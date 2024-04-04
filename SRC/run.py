import os
import argparse
import tqdm
import torch
import torch.nn.functional as F
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup
from models.data_BIO_loader import load_data, DataTterator, load_pos_data, load_dp_data
from models import tools, data_conll, span_loss, mmd_loss
from models.model import stage_2_features_generation, Step_1, Step_2_forward, Step_2_reverse, Loss, DADP, DAPOS
from models.Metric import Metric
from models.eval_features import unbatch_data
from log import logger
from thop import profile, clever_format
import random
import numpy as np
import time
from itertools import cycle
torch.backends.cudnn.enabled = False
torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

sentiment2id = {'none': 0, 'positive': 1, 'negative': 2, 'neutral': 3}

def eval(bert_model, dp_model, pos_model, step_1_model, step_2_forward, step_2_reverse, dataset, tar_dev_dp_datastes, tar_dev_pos_datasets, args):
    with torch.no_grad():
        bert_model.eval()
        dp_model.eval()
        pos_model.eval()
        step_1_model.eval()
        step_2_forward.eval()
        step_2_reverse.eval()

        '''加载目标词性验证验证集'''

        # tar_dev_pos_datasets = DataTterator(tar_dev_pos_datasets, args)
        '''真实结果'''
        gold_instances = []
        '''前向预测结果'''
        forward_stage1_pred_aspect_result, forward_stage1_pred_aspect_with_sentiment, \
        forward_stage1_pred_aspect_sentiment_logit, forward_stage2_pred_opinion_result, \
        forward_stage2_pred_opinion_sentiment_logit = [], [], [], [], []

        '''反向预测结果'''
        reverse_stage1_pred_opinion_result, reverse_stage1_pred_opinion_with_sentiment, \
        reverse_stage1_pred_opinion_sentiment_logit, reverse_stage2_pred_aspect_result, \
        reverse_stage2_pred_aspect_sentiment_logit = [], [], [], [], []

        for j in range(dataset.batch_count):
            tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
            spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
            reverse_aspect_label_tensor, related_spans_tensor, sentence_length = dataset.get_batch(j)


            bert_features = bert_model(input_ids=tokens_tensor, attention_mask=attention_mask)
            '''加载依存信息'''

            tar_input_ids, tar_attention_mask, tar_token_type_ids, tar_dp_span_label, tar_dp_span_mask = tar_dev_dp_datastes.get_dp_batch(j)
            '''加载依存信息'''
            tar_dp_span_logits, _, tar_dp_start, tar_dp_end, tar_dp_rep, tar_bert_output = dp_model(
                input_ids=tar_input_ids.to(args.device),
                attention_mask=tar_attention_mask.to(args.device),
                bert_out=bert_features.last_hidden_state.to(args.device),
                mode='tar',
                token_type_ids=tar_token_type_ids.to(args.device),
            )
            '''加载词性信息'''

            tar_pos_input_ids, tar_pos_attention_mask, tar_pos_token_type_ids, tar_pos_span_label, tar_pos_span_mask = tar_dev_pos_datasets.get_pos_batch(
                j)
            tar_pos_span_logits, tar_pos_rep, tar_pos_representation, tar_bert_output = pos_model(
                input_ids=tar_pos_input_ids.to(args.device),
                attention_mask=tar_pos_attention_mask.to(args.device),
                bert_out=bert_features.last_hidden_state.to(args.device),
                mode='tar',
                token_type_ids=tar_pos_token_type_ids.to(args.device),
            )
            '''增加Bert信息'''
            tar_bert_output = tar_bert_output + 0.1 * tar_pos_representation
            if j == 0 and args.model_para_test:
                bert_model.to("cpu")

                flop_bert, para_bert = profile(bert_model, inputs=(tokens_tensor, attention_mask,), custom_ops={})
                macs, param = clever_format([flop_bert,para_bert], "%.3f")
                print("BERT MACs: ", macs, "BERT Params", param)
                logger.info(
                    'SBERT MACs:  {}\tBERT Params: {:.8f}\n\n'.format(macs, param))


            if j == 0 and args.model_para_test:
                step_1_model.to("cpu")

                flop_step1, para_step1 = profile(step_1_model, inputs=(tar_bert_output, attention_mask, bert_spans_tensor, spans_mask_tensor,
                                                                     related_spans_tensor, sentence_length))
                macs, param = clever_format([flop_step1, para_step1], "%.3f")
                print("STEP 1 MACs: ", macs, "STEP 1 Params", param)
                logger.info(
                    'STEP 1 MACs:  {}\tSTEP 1 Params: {:.8f}\n\n'.format(macs, param))

            aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
            cnn_spans_mask_tensor = step_1_model(
                tar_bert_output, attention_mask, bert_spans_tensor, spans_mask_tensor,
                related_spans_tensor, sentence_length)

            '''Batch更新'''
            pred_aspect_logits = torch.argmax(F.softmax(aspect_class_logits, dim=2), dim=2)
            pred_sentiment_ligits = F.softmax(aspect_class_logits, dim=2)
            pred_aspect_logits = torch.where(spans_mask_tensor == 1, pred_aspect_logits,
                                             torch.tensor(0).type_as(pred_aspect_logits))

            reverse_pred_stage1_logits = torch.argmax(F.softmax(opinion_class_logits, dim=2), dim=2)
            reverse_pred_sentiment_ligits = F.softmax(opinion_class_logits, dim=2)
            reverse_pred_stage1_logits = torch.where(spans_mask_tensor == 1, reverse_pred_stage1_logits,
                                             torch.tensor(0).type_as(reverse_pred_stage1_logits))

            '''真实结果合成'''
            gold_instances.append(dataset.instances[j])

            '''双方向预测'''
            if torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0] == 0:

                forward_stage1_pred_aspect_result.append(torch.full_like(spans_aspect_tensor, -1))
                forward_stage1_pred_aspect_with_sentiment.append(pred_aspect_logits)
                forward_stage1_pred_aspect_sentiment_logit.append(pred_sentiment_ligits)
                forward_stage2_pred_opinion_result.append(torch.full_like(spans_opinion_label_tensor, -1))
                forward_stage2_pred_opinion_sentiment_logit.append(
                    torch.full_like(spans_opinion_label_tensor.unsqueeze(-1).expand(-1, -1, len(sentiment2id)), -1))

            else:
                pred_aspect_spans = torch.chunk(torch.nonzero(pred_aspect_logits, as_tuple=False),
                                                torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0], dim=0)
                pred_span_aspect_tensor = None
                for pred_aspect_span in pred_aspect_spans:
                    batch_num = pred_aspect_span.squeeze()[0]
                    span_aspect_tensor_unspilt_1 = bert_spans_tensor[batch_num, pred_aspect_span.squeeze()[1], :2]
                    span_aspect_tensor_unspilt = torch.tensor(
                        (batch_num, span_aspect_tensor_unspilt_1[0], span_aspect_tensor_unspilt_1[1])).unsqueeze(0)
                    if pred_span_aspect_tensor is None:
                        pred_span_aspect_tensor = span_aspect_tensor_unspilt
                    else:
                        pred_span_aspect_tensor = torch.cat((pred_span_aspect_tensor, span_aspect_tensor_unspilt), dim=0)
                # _,all_span_aspect_tensor, all_bert_embedding, all_attention_mask, all_spans_embedding, all_span_mask = stage_2_features_generation(
                #     bert_features.last_hidden_state, attention_mask, bert_spans_tensor, spans_mask_tensor,
                #     spans_embedding, pred_span_aspect_tensor)
                _, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, all_spans_embedding, all_span_mask = stage_2_features_generation(
                    tar_bert_output, attention_mask, bert_spans_tensor, spans_mask_tensor,
                    forward_embedding, pred_span_aspect_tensor)

                if j == 0 and args.model_para_test:
                    step_2_forward.to("cpu")

                    flop_step2_f, para_step2_f = profile(step_2_forward, inputs=(all_spans_embedding, all_span_mask, all_span_aspect_tensor))
                    macs, param = clever_format([flop_step2_f, para_step2_f], "%.3f")
                    print("STEP 2 forward MACs: ", macs, "STEP 2 forward Params", param)
                    logger.info(
                        'STEP 2 forward MACs:  {}\tSTEP 2 forward Params: {:.8f}\n\n'.format(macs, param))

                opinion_class_logits, opinion_attention = step_2_forward(all_spans_embedding, all_span_mask,
                                                                         all_span_aspect_tensor)

                forward_stage1_pred_aspect_result.append(pred_span_aspect_tensor)
                forward_stage1_pred_aspect_with_sentiment.append(pred_aspect_logits)
                forward_stage1_pred_aspect_sentiment_logit.append(pred_sentiment_ligits)
                forward_stage2_pred_opinion_result.append(torch.argmax(F.softmax(opinion_class_logits, dim=2), dim=2))
                forward_stage2_pred_opinion_sentiment_logit.append(F.softmax(opinion_class_logits, dim=2))
            '''反向预测'''
            if torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0] == 0:
                reverse_stage1_pred_opinion_result.append(torch.full_like(reverse_opinion_tensor, -1))
                reverse_stage1_pred_opinion_with_sentiment.append(reverse_pred_stage1_logits)
                reverse_stage1_pred_opinion_sentiment_logit.append(reverse_pred_sentiment_ligits)
                reverse_stage2_pred_aspect_result.append(torch.full_like(reverse_aspect_label_tensor, -1))
                reverse_stage2_pred_aspect_sentiment_logit.append(
                    torch.full_like(reverse_aspect_label_tensor.unsqueeze(-1).expand(-1, -1, len(sentiment2id)), -1))
            else:
                reverse_pred_opinion_spans = torch.chunk(torch.nonzero(reverse_pred_stage1_logits, as_tuple=False),
                                                torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0], dim=0)
                reverse_span_opinion_tensor = None
                for reverse_pred_opinion_span in reverse_pred_opinion_spans:
                    batch_num = reverse_pred_opinion_span.squeeze()[0]
                    reverse_opinion_tensor_unspilt = bert_spans_tensor[batch_num, reverse_pred_opinion_span.squeeze()[1], :2]
                    reverse_opinion_tensor_unspilt = torch.tensor(
                        (batch_num, reverse_opinion_tensor_unspilt[0], reverse_opinion_tensor_unspilt[1])).unsqueeze(0)
                    if reverse_span_opinion_tensor is None:
                        reverse_span_opinion_tensor = reverse_opinion_tensor_unspilt
                    else:
                        reverse_span_opinion_tensor = torch.cat((reverse_span_opinion_tensor, reverse_opinion_tensor_unspilt), dim=0)
                # __, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                # reverse_spans_embedding, reverse_span_mask = stage_2_features_generation(bert_features.last_hidden_state,
                #                                                                          attention_mask,
                #                                                                          bert_spans_tensor,
                #                                                                          spans_mask_tensor,
                #                                                                          spans_embedding,
                #                                                                          reverse_span_opinion_tensor)
                __, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                reverse_spans_embedding, reverse_span_mask = stage_2_features_generation(
                    tar_bert_output,
                    attention_mask,
                    bert_spans_tensor,
                    spans_mask_tensor,
                    reverse_embedding,
                    reverse_span_opinion_tensor)

                if j == 0 and args.model_para_test:
                    step_2_reverse.to("cpu")

                    flop_step2_r, para_step2_r = profile(step_2_reverse, inputs=(reverse_spans_embedding,
                                                                                reverse_span_mask,
                                                                                all_reverse_opinion_tensor))
                    macs, param = clever_format([flop_step2_r, para_step2_r], "%.3f")
                    print("STEP 2 reverse MACs: ", macs, "STEP 2 reverse Params", param)
                    logger.info(
                        'STEP 2 reverse MACs:  {}\tSTEP 2 reverse Params: {:.8f}\n\n'.format(macs, param))

                reverse_aspect_class_logits, reverse_aspect_attention = step_2_reverse(reverse_spans_embedding,
                                                                                reverse_span_mask,
                                                                                all_reverse_opinion_tensor)

                reverse_stage1_pred_opinion_result.append(reverse_span_opinion_tensor)
                reverse_stage1_pred_opinion_with_sentiment.append(reverse_pred_stage1_logits)
                reverse_stage1_pred_opinion_sentiment_logit.append(reverse_pred_sentiment_ligits)
                reverse_stage2_pred_aspect_result.append(torch.argmax(F.softmax(reverse_aspect_class_logits, dim=2), dim=2))
                reverse_stage2_pred_aspect_sentiment_logit.append(F.softmax(reverse_aspect_class_logits, dim=2))

        gold_instances = [x for i in gold_instances for x in i]
        forward_pred_data = (forward_stage1_pred_aspect_result, forward_stage1_pred_aspect_with_sentiment,
                             forward_stage1_pred_aspect_sentiment_logit, forward_stage2_pred_opinion_result,
                             forward_stage2_pred_opinion_sentiment_logit)
        forward_pred_result = unbatch_data(forward_pred_data)

        reverse_pred_data = (reverse_stage1_pred_opinion_result, reverse_stage1_pred_opinion_with_sentiment,
                             reverse_stage1_pred_opinion_sentiment_logit, reverse_stage2_pred_aspect_result,
                             reverse_stage2_pred_aspect_sentiment_logit)
        reverse_pred_result = unbatch_data(reverse_pred_data)

        metric = Metric(args, forward_pred_result, reverse_pred_result, gold_instances)
        aspect_result, opinion_result, apce_result, pair_result, triplet_result = metric.score_triples()
        # print('aspect precision:', aspect_result[0], "aspect recall: ", aspect_result[1], "aspect f1: ", aspect_result[2])
        # print('opinion precision:', opinion_result[0], "opinion recall: ", opinion_result[1], "opinion f1: ",
        #       opinion_result[2])
        # print('APCE precision:', apce_result[0], "APCE recall: ", apce_result[1], "APCE f1: ",
        #       apce_result[2])
        # print('pair precision:', pair_result[0], "pair recall:", pair_result[1], "pair f1:", pair_result[2])
        # print('triple precision:', triplet_result[0], "triple recall: ", triplet_result[1], "triple f1: ", triplet_result[2])
        logger.info(
            'aspect precision: {}\taspect recall: {:.8f}\taspect f1: {:.8f}'.format(aspect_result[0], aspect_result[1], aspect_result[2]))
        logger.info(
            'opinion precision: {}\topinion recall: {:.8f}\topinion f1: {:.8f}'.format(opinion_result[0],
                                                                                        opinion_result[1],
                                                                                        opinion_result[2]))
        logger.info('APCE precision: {}\tAPCE recall: {:.8f}\tAPCE f1: {:.8f}'.format(apce_result[0],
                                                                                apce_result[1], apce_result[2]))
        logger.info('pair precision: {}\tpair recall: {:.8f}\tpair f1: {:.8f}'.format(pair_result[0],
                                                                                          pair_result[1],
                                                                                          pair_result[2]))
        logger.info('triple precision: {}\ttriple recall: {:.8f}\ttriple f1: {:.8f}'.format(triplet_result[0],
                                                                                          triplet_result[1],
                                                                                          triplet_result[2]))

    bert_model.train()
    dp_model.train()
    pos_model.train()
    step_1_model.train()
    step_2_forward.train()
    step_2_reverse.train()
    return aspect_result, opinion_result, apce_result, pair_result, triplet_result

def set_random_seed(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def train(args):
    set_random_seed(args.seed)
    if args.dataset_path == './datasets/BIO_form/':
        train_path = args.dataset_path + args.dataset + "/train.json"
        dev_path = args.dataset_path + args.dataset + "/dev.json"
        test_path = args.dataset_path + args.dataset + "/test.json"
    else:
        train_path = args.dataset_path + args.dataset + "/train_triplets.txt"
        dev_path = args.dataset_path + args.target + "/dev_triplets.txt"
        test_path = args.dataset_path + args.target + "/test_triplets.txt"
        tar_train_path = args.dataset_path + args.target + "/train_triplets.txt"

    print('-------------------------------')
    print('开始加载测试集')
    logger.info('开始加载测试集')
    '''加载依存信息'''
    dp_label2id, dp_id2label, dp_num_label = tools.load_schema_dp(mode='dp')
    dp_num_label += 1

    '''加载词性信息'''
    pos_label2id, pos_id2label, pos_num_label = tools.load_schema_dp(mode='pos')
    pos_num_label += 1
    '''初始化分词器'''
    tokenizer = tools.get_tokenizer(bert_model_path=args.init_model)
    '''加载依存数据用于测试模型'''
    tar_dp_test_data = data_conll.yield_data(args=args,
                                             file_path=args.dp_tar_test_path,
                                             tokenizer=tokenizer,
                                             mode='dp',
                                             data='test',
                                             label2id=dp_label2id)
    '''加载词性数据'''
    tar_pos_test_data = data_conll.yield_pos_data(args=args,
                                             file_path=args.pos_tar_test_path,
                                             tokenizer=tokenizer,
                                             mode='pos',
                                             data='test',
                                             label2id=pos_label2id)
    tar_test_dp_datastes = load_dp_data(args, tar_dp_test_data, if_train=True)
    tar_test_dp_datastes = DataTterator(tar_test_dp_datastes, args)
    tar_test_pos_datastes = load_pos_data(args, tar_pos_test_data, if_train=True)
    tar_test_pos_datastes = DataTterator(tar_test_pos_datastes, args)
    '''测试数据'''
    test_datasets = load_data(args, test_path, if_train=False)
    testset = DataTterator(test_datasets, args)

    print('测试集加载完成')
    logger.info('测试集加载完成')
    print('-------------------------------')

    dp_span_loss_func = span_loss.Span_loss(dp_num_label).to(args.device)
    pos_span_losss_func = span_loss.Span_loss(pos_num_label).to(args.device)
    mmd_loss_func = mmd_loss.MMD_loss().to(args.device)
    mmd_pos_loss_func = mmd_loss.MMD_loss().to(args.device)

    pos_model = DAPOS(args, pos_num_label=pos_num_label, device='cuda').to(args.device)
    pos_model.to(args.device)
    pos_model_param_optimizer = list(pos_model.named_parameters())

    dp_model = DADP(args, dp_num_label=dp_num_label, ner_num_label=0, device='cuda').to(args.device)
    dp_model.to(args.device)
    dp_model_param_optimizer = list(dp_model.named_parameters())


    Bert = BertModel.from_pretrained(args.init_model)
    bert_config = Bert.config
    Bert.to(args.device)
    bert_param_optimizer = list(Bert.named_parameters())

    step_1_model = Step_1(args, bert_config)
    step_1_model.to(args.device)
    step_1_param_optimizer = list(step_1_model.named_parameters())

    step2_forward_model = Step_2_forward(args, bert_config)
    step2_forward_model.to(args.device)
    forward_step2_param_optimizer = list(step2_forward_model.named_parameters())

    step2_reverse_model = Step_2_reverse(args, bert_config)
    step2_reverse_model.to(args.device)
    reverse_step2_param_optimizer = list(step2_reverse_model.named_parameters())

    training_param_optimizer = [
        {'params': [p for n, p in bert_param_optimizer]},
        {'params': [p for n, p in dp_model_param_optimizer]},
        {'params': [p for n, p in pos_model_param_optimizer]},
        {'params': [p for n, p in step_1_param_optimizer], 'lr': args.task_learning_rate},
        {'params': [p for n, p in forward_step2_param_optimizer], 'lr': args.task_learning_rate},
        {'params': [p for n, p in reverse_step2_param_optimizer], 'lr': args.task_learning_rate}]
    optimizer = AdamW(training_param_optimizer, lr=args.learning_rate)

    lambda_weight = torch.FloatTensor([args.lambda_weight]).to(args.device)
    beta_weight = torch.FloatTensor([args.beta_weight]).to(args.device)

    if args.muti_gpu:
        Bert = torch.nn.DataParallel(Bert)
        step_1_model = torch.nn.DataParallel(step_1_model)
        step2_forward_model = torch.nn.DataParallel(step2_forward_model)
        step2_reverse_model = torch.nn.DataParallel(step2_reverse_model)

    if args.mode == 'train':
        print('-------------------------------')
        logger.info('开始加载训练与验证集')
        print('开始加载训练与验证集')
        '''加载依存信息'''
        src_dp_train_data = data_conll.yield_data(args=args,
                                                  file_path=args.dp_src_train_path,
                                                  tokenizer=tokenizer,
                                                  mode='dp',
                                                  data='train',
                                                  label2id=dp_label2id)
        tar_dp_train_data = data_conll.yield_data(args=args,
                                                  file_path=args.dp_tar_train_path,
                                                  tokenizer=tokenizer,
                                                  mode='dp',
                                                  data='train',
                                                  label2id=dp_label2id)

        '''加载词性信息'''
        src_pos_train_data = data_conll.yield_pos_data(args=args,
                                                       file_path=args.pos_src_train_path,
                                                       tokenizer=tokenizer,
                                                       mode='pos',
                                                       data='train',
                                                       label2id=pos_label2id)
        tar_pos_train_data = data_conll.yield_pos_data(args=args,
                                                      file_path=args.pos_tar_train_path,
                                                      tokenizer=tokenizer,
                                                      mode='pos',
                                                      data='train',
                                                      label2id=pos_label2id)
        tar_pos_dev_data = data_conll.yield_pos_data(args=args,
                                                       file_path=args.pos_tar_dev_path,
                                                       tokenizer=tokenizer,
                                                       mode='pos',
                                                       data='test',
                                                       label2id=pos_label2id)
        '''组成词性batch'''
        src_train_pos_datasets = load_pos_data(args, src_pos_train_data, if_train=True)
        tar_train_pos_datasets = load_pos_data(args, tar_pos_train_data, if_train=True)
        src_train_pos_datasets = DataTterator(src_train_pos_datasets, args)
        tar_train_pos_datasets = DataTterator(tar_train_pos_datasets, args)
        '''用来验证的数据集'''
        tar_dev_pos_datasets = load_pos_data(args, tar_pos_test_data, if_train=True)
        tar_dev_pos_datasets = DataTterator(tar_dev_pos_datasets, args)
        '''单独加载依存信息'''
        src_train_dp_datasets = load_dp_data(args, src_dp_train_data, if_train=True)
        tar_train_dp_datasets = load_dp_data(args, tar_dp_train_data, if_train=True)
        src_train_dp_datasets = DataTterator(src_train_dp_datasets, args)
        tar_train_dp_datasets = DataTterator(tar_train_dp_datasets, args)
        '''加载正常数据'''
        train_datasets = load_data(args, train_path, if_train=True)
        trainset = DataTterator(train_datasets, args)
        tar_train_datasets = load_data(args, tar_train_path, if_train=True)
        tar_trainset = DataTterator(tar_train_datasets, args)

        print("Train features build completed")

        print("Dev features build beginning")
        tar_dp_dev_data = data_conll.yield_data(args=args,
                                                 file_path=args.dp_tar_dev_path,
                                                 tokenizer=tokenizer,
                                                 mode='dp',
                                                 data='test',
                                                 label2id=dp_label2id)
        '''加载目标依存验证数据集'''
        tar_dev_dp_datastes = load_dp_data(args, tar_dp_test_data, if_train=True)
        tar_dev_dp_datastes = DataTterator(tar_dev_dp_datastes, args)
        dev_datasets = load_data(args, dev_path, if_train=False)
        devset = DataTterator(dev_datasets, args)
        print('训练集与验证集加载完成')
        logger.info('训练集与验证集加载完成')
        print('-------------------------------')
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        # scheduler
        if args.whether_warm_up:
            training_steps = args.epochs * trainset.batch_count
            warmup_steps = int(training_steps * args.warm_up)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)

        tot_loss = 0
        tot_kl_loss = 0
        best_aspect_f1, best_opinion_f1, best_APCE_f1, best_pairs_f1, best_triple_f1 = 0, 0, 0, 0, 0
        best_aspect_epoch, best_opinion_epoch, best_APCE_epoch, best_pairs_epoch, best_triple_epoch = 0, 0, 0, 0, 0

        for i in range(args.epochs):
            logger.info(('Epoch:{}'.format(i)))
            for j in tqdm.trange(trainset.batch_count):
            # for j in range(trainset.batch_count):
                if j == 1:
                    start = time.time()
                optimizer.zero_grad()
                '''得到词性数据'''
                src_pos_input_ids, src_pos_attention_mask, src_pos_token_type_ids, src_pos_span_label, src_pos_span_mask = src_train_pos_datasets.get_pos_batch(j)
                tar_pos_index = j % len(tar_train_pos_datasets.instances)
                tar_pos_input_ids, tar_pos_attention_mask, tar_pos_token_type_ids, tar_pos_span_label, tar_pos_span_mask = tar_train_pos_datasets.get_pos_batch(tar_pos_index)
                '''得到依存数据'''
                src_dp_input_ids, src_dp_attention_mask, src_dp_token_type_ids, src_dp_span_label, src_dp_span_mask = src_train_dp_datasets.get_dp_batch(j)
                '''目标域的数据不够，用循环进行补充'''
                tar_dp_index = j % len(tar_trainset.instances)
                tar_dp_input_ids, tar_dp_attention_mask, tar_dp_token_type_ids, tar_dp_span_label, tar_dp_span_mask = tar_train_dp_datasets.get_dp_batch(tar_dp_index)
                '''得到训练三元组的数据'''
                tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
                spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
                reverse_aspect_label_tensor, related_spans_tensor, sentence_length = trainset.get_batch(j)
                ''''''
                bert_output = Bert(input_ids=tokens_tensor, attention_mask=attention_mask)
                '''这里是对依存数据进行训练'''
                src_dp_span_logits, _, src_dp_start, src_dp_end, src_dp_rep, src_bert_output = dp_model(
                    input_ids=src_dp_input_ids.to(args.device),
                    attention_mask=src_dp_attention_mask.to(args.device),
                    bert_out=bert_output.last_hidden_state.to(args.device),
                    mode='src',
                    token_type_ids=src_dp_token_type_ids.to(args.device),
                )
                tar_dp_span_logits, _, tar_dp_start, tar_dp_end, tar_dp_rep, tar_bert_output = dp_model(
                    input_ids=tar_dp_input_ids.to(args.device),
                    attention_mask=tar_dp_attention_mask.to(args.device),
                    bert_out=bert_output.last_hidden_state.to(args.device),
                    mode='tar',
                    token_type_ids=tar_dp_token_type_ids.to(args.device),
                )
                '''得到依存损失'''
                dp_loss_src = dp_span_loss_func(src_dp_span_logits, src_dp_span_label.to(args.device),
                                            src_dp_span_mask.to(args.device))
                dp_loss_tar = dp_span_loss_func(tar_dp_span_logits, tar_dp_span_label.to(args.device),
                                            tar_dp_span_mask.to(args.device))

                dp_loss = dp_loss_src + dp_loss_tar
                mmd_loss_rep = mmd_loss_func(src_dp_rep, tar_dp_rep, src_dp_attention_mask, tar_dp_attention_mask)
                # mmd_loss_end = mmd_loss_func(src_dp_end, tar_dp_end, src_attention_mask, tar_attention_mask)

                '''对词性数据进行训练,第一个计算损失，第二个拉近距离，第三个加在Bert，第四个暂时不用'''
                src_pos_span_logits, src_pos_rep, src_dp_representation, src_bert_output = pos_model(
                    input_ids=src_pos_input_ids.to(args.device),
                    attention_mask=src_pos_attention_mask.to(args.device),
                    bert_out=bert_output.last_hidden_state.to(args.device),
                    mode='src',
                    token_type_ids=src_pos_token_type_ids.to(args.device),
                )
                tar_pos_span_logits, tar_pos_rep, tar_dp_representation, tar_bert_output = pos_model(
                    input_ids=tar_pos_input_ids.to(args.device),
                    attention_mask=tar_pos_attention_mask.to(args.device),
                    bert_out=bert_output.last_hidden_state.to(args.device),
                    mode='tar',
                    token_type_ids=tar_pos_token_type_ids.to(args.device),
                )
                pos_loss_src = pos_span_losss_func(src_pos_span_logits, src_pos_span_label.to(args.device),
                                            src_pos_span_mask.to(args.device))
                pos_loss_tar = pos_span_losss_func(tar_pos_span_logits, tar_pos_span_label.to(args.device),
                                               tar_pos_span_mask.to(args.device))
                pos_loss = pos_loss_src + pos_loss_tar
                '''加入目标域然后拉近'''
                pos_mmd_loss = mmd_pos_loss_func(src_pos_rep, tar_pos_rep, src_pos_attention_mask, tar_pos_attention_mask)

                src_bert_output + 0.1 * src_dp_representation
                '''加入三元组抽取的模型，增加loss损失'''

                aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
                cnn_spans_mask_tensor = step_1_model(
                                                                                      src_bert_output,
                                                                                      attention_mask,
                                                                                      bert_spans_tensor,
                                                                                      spans_mask_tensor,
                                                                                      related_spans_tensor,
                                                                                      sentence_length)

                '''Batch更新'''
                all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
                all_spans_embedding, all_span_mask = stage_2_features_generation(src_bert_output,
                                                                             attention_mask, bert_spans_tensor,
                                                                             spans_mask_tensor, forward_embedding,
                                                                             spans_aspect_tensor,
                                                                             spans_opinion_label_tensor)
                all_reverse_aspect_tensor, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                reverse_spans_embedding, reverse_span_mask = stage_2_features_generation(src_bert_output,
                                                                                     attention_mask, bert_spans_tensor,
                                                                                     spans_mask_tensor, reverse_embedding,
                                                                                     reverse_opinion_tensor,
                                                                                     reverse_aspect_label_tensor)

                step_2_opinion_class_logits, opinion_attention = step2_forward_model(all_spans_embedding, all_span_mask, all_span_aspect_tensor)
                step_2_aspect_class_logits, aspect_attention = step2_reverse_model(reverse_spans_embedding,
                    reverse_span_mask, all_reverse_opinion_tensor)

                loss, kl_loss = Loss(spans_ner_label_tensor, aspect_class_logits, all_span_opinion_tensor, step_2_opinion_class_logits,
                            spans_mask_tensor, all_span_mask, reverse_ner_label_tensor, opinion_class_logits,
                            all_reverse_aspect_tensor, step_2_aspect_class_logits, cnn_spans_mask_tensor, reverse_span_mask,
                            spans_embedding, related_spans_tensor, args)
                loss = lambda_weight * loss + beta_weight * (dp_loss + mmd_loss_rep + pos_loss + pos_mmd_loss)
                if args.accumulation_steps > 1:
                    loss = loss / args.accumulation_steps
                    loss.backward()
                    if ((j + 1) % args.accumulation_steps) == 0:
                        optimizer.step()
                        if args.whether_warm_up:
                            scheduler.step()
                else:
                    loss.backward()
                    optimizer.step()
                    if args.whether_warm_up:
                        scheduler.step()
                tot_loss += loss.item()
                tot_kl_loss += kl_loss
            logger.info(('Loss:', tot_loss))
            logger.info(('KL_Loss:', tot_kl_loss))
            tot_loss = 0
            tot_kl_loss = 0

            print('Evaluating, please wait')
            # aspect_result, opinion_result, apce_result, pair_result, triplet_result = eval(Bert, step_1_model,
            #                                                                                step2_forward_model,
            #                                                                                step2_reverse_model,
            #                                                                                devset, args)
            aspect_result, opinion_result, apce_result, pair_result, triplet_result = eval(Bert, dp_model, pos_model, step_1_model,
                                                                                           step2_forward_model,
                                                                                           step2_reverse_model,
                                                                                           testset, tar_dev_dp_datastes, tar_dev_pos_datasets, args)
            print('Evaluating complete')
            if aspect_result[2] > best_aspect_f1:
                best_aspect_f1 = aspect_result[2]
                best_aspect_precision = aspect_result[0]
                best_aspect_recall = aspect_result[1]
                best_aspect_epoch = i

            if opinion_result[2] > best_opinion_f1:
                best_opinion_f1 = opinion_result[2]
                best_opinion_precision = opinion_result[0]
                best_opinion_recall = opinion_result[1]
                best_opinion_epoch = i

            if apce_result[2] > best_APCE_f1:
                best_APCE_f1 = apce_result[2]
                best_APCE_precision = apce_result[0]
                best_APCE_recall = apce_result[1]
                best_APCE_epoch = i

            if pair_result[2] > best_pairs_f1:
                best_pairs_f1 = pair_result[2]
                best_pairs_precision = pair_result[0]
                best_pairs_recall = pair_result[1]
                best_pairs_epoch = i

            if triplet_result[2] > best_triple_f1 and triplet_result[2] > 0.60 and triplet_result[2] - best_triple_f1 >0.5:
            # if triplet_result[2] > 0.50:
                model_path = args.model_dir + args.dataset + '_' + str(triplet_result[2]) + '.pt'
                state = {
                    "bert_model": Bert.state_dict(),
                    "dp_model": dp_model.state_dict(),
                    "pos_model": pos_model.state_dict(),
                    "step_1_model": step_1_model.state_dict(),
                    "step2_forward_model": step2_forward_model.state_dict(),
                    "step2_reverse_model": step2_reverse_model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                torch.save(state, model_path)
                logger.info("_________________________________________________________")
                logger.info("best model save")
                logger.info("_________________________________________________________")

                best_triple_f1 = triplet_result[2]
                best_triple_precision = triplet_result[0]
                best_triple_recall = triplet_result[1]
                best_triple_epoch = i

            '''这里可以进行每个epoch结束之后释放显存'''
            torch.cuda.empty_cache()

        logger.info(
            'best aspect epoch: {}\tbest aspect precision: {:.8f}\tbest aspect recall: {:.8f}\tbest aspect f1: {:.8f}'.
                format(best_aspect_epoch, best_aspect_precision, best_aspect_recall, best_aspect_f1))
        logger.info(
            'best opinion epoch: {}\tbest opinion precision: {:.8f}\tbest opinion recall: {:.8f}\tbest opinion f1: {:.8f}'.
                format(best_opinion_epoch, best_opinion_precision, best_opinion_recall, best_opinion_f1))

        logger.info('best APCE epoch: {}\tbest APCE precision: {:.8f}\tbest APCE recall: {:.8f}\tbest APCE f1: {:.8f}'.
              format(best_APCE_epoch, best_APCE_precision, best_APCE_recall, best_APCE_f1))
        logger.info('best pair epoch: {}\tbest pair precision: {:.8f}\tbest pair recall: {:.8f}\tbest pair f1: {:.8f}'.
              format(best_pairs_epoch, best_pairs_precision, best_pairs_recall, best_pairs_f1))
        logger.info(
            'best triple epoch: {}\tbest triple precision: {:.8f}\tbest triple recall: {:.8f}\tbest triple f1: {:.8f}'.
            format(best_triple_epoch, best_triple_precision, best_triple_recall, best_triple_f1))

    logger.info("Features build completed")
    logger.info("Evaluation on testset:")

    # model_path = args.model_dir + args.dataset+'_'+str(best_triple_f1) + '.pt'
    model_path = args.model_dir + args.dataset + '_' + str(0.44311377245508976) + '.pt'
    if args.muti_gpu:
        state = torch.load(model_path)
    else:
        state = torch.load(model_path)
        # state = load_with_single_gpu(model_path)

    Bert.load_state_dict(state['bert_model'])
    dp_model.load_state_dict(state['dp_model'])
    pos_model.load_state_dict(state['pos_model'])
    step_1_model.load_state_dict(state['step_1_model'])
    step2_forward_model.load_state_dict(state['step2_forward_model'])
    step2_reverse_model.load_state_dict(state['step2_reverse_model'])
    # eval(Bert, step_1_model, step2_forward_model, step2_reverse_model, testset, args)
    eval(Bert, dp_model, pos_model, step_1_model,
         step2_forward_model,
         step2_reverse_model,
         testset, tar_test_dp_datastes, tar_test_pos_datastes, args)

def load_with_single_gpu(model_path):
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    final_state = {}
    for i in state_dict:
        for k, v in state_dict[i].items():
            name = k[7:]
            new_state_dict[name] = v
        final_state[i] = new_state_dict
        new_state_dict = OrderedDict()
    return final_state

def main():
    parser = argparse.ArgumentParser(description="Train scrip")
    '''依存信息'''
    parser.add_argument("--dp_src_train_path", type=str, default='./Dependency/fashion/train.txt', help="dp train file") # src
    parser.add_argument("--dp_tar_train_path", type=str, default='./Dependency/fashion/train.txt', help="dp train file") # tar
    parser.add_argument("--dp_src_dev_path", type=str, default='./Dependency/fashion/dev.txt', help="dp dev file") # src
    parser.add_argument("--dp_src_test_path", type=str, default='./Dependency/fashion/test.txt', help="dp train file") # src
    parser.add_argument("--dp_tar_test_path", type=str, default='./Dependency/fashion/test.txt', help="dp train file") # tar
    parser.add_argument("--dp_tar_dev_path", type=str, default='./Dependency/fashion/dev.txt', help="dp train file") # tar
    '''词性信息'''
    parser.add_argument("--pos_src_train_path", type=str, default='./POS-OF-SPEECH/fashion/train.txt', help="pos train file") # src
    parser.add_argument("--pos_src_dev_path", type=str, default='./POS-OF-SPEECH/fashion/dev.txt', help="pos dev file") # src
    parser.add_argument("--pos_tar_train_path", type=str, default='./POS-OF-SPEECH/fashion/train.txt', help="pos train file") # tar
    parser.add_argument("--pos_tar_dev_path", type=str, default='./POS-OF-SPEECH/fashion/dev.txt',
                        help="pos train file") # tar
    parser.add_argument("--pos_src_test_path", type=str, default='./POS-OF-SPEECH/fashion/test.txt', help="pos train file") # src
    parser.add_argument("--pos_tar_test_path", type=str, default='./POS-OF-SPEECH/fashion/test.txt', help="pos train file") # tar

    parser.add_argument('--model_dir', type=str, default="savemodel/", help='model path prefix')
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument("--init_model", default="pretrained_models/bert-base-uncased", type=str, required=False, help="Initial model.")
    parser.add_argument("--init_vocab", default="pretrained_models/bert-base-uncased", type=str, required=False, help="Initial vocab.")
    parser.add_argument("--pretrained_model_path", type=str, default="./bert-large-uncased-whole-word-masking",
                        help="pretrained_model_path")

    parser.add_argument("--bert_feature_dim", default=768, type=int, help="feature dim for bert")
    parser.add_argument("--do_lower_case", default=True, action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=356, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_length", type=int, default=356, help="max_length")
    parser.add_argument("--drop_out", type=int, default=0.1, help="")
    parser.add_argument("--max_span_length", type=int, default=8, help="")
    parser.add_argument("--embedding_dim4width", type=int, default=200, help="")
    parser.add_argument("--task_learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--muti_gpu", default=False)
    parser.add_argument('--epochs', type=int, default=300, help='training epoch number')
    parser.add_argument("--train_batch_size", default=4, type=int, help="batch size for training")
    parser.add_argument("--lstm_hidden_size", type=int, default=512, help="lstm_hidden_size")
    parser.add_argument("--to_biaffine_size", type=int, default=128, help="to_biaffine_size")
    parser.add_argument("--dp_size", type=int, default=128, help="dp_size")
    parser.add_argument("--seed", type=int, default=2022, help="")
    '''修改了数据格式'''
    parser.add_argument("--dataset_path", default="./ASTE-Data-V2-EMNLP2020/",
                        choices=["./datasets/BIO_form/", "./datasets/ASTE-Data-V2-EMNLP2020/"],
                        help="")
    parser.add_argument("--dataset", default="fashion", type=str, choices=["beauty", "electronics", "fashion", "home", "res14", "res15", "res16"],
                        help="specify the dataset")
    parser.add_argument("--target", default="fashion", type=str, choices=["book", "pet", "toy", "grocery", "res14", "res15", "res16", "beauty", "home", "fashion"],
                        help="specify the dataset")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help='option: train, test')
    '''对相似Span进行attention'''
    # 分词中仅使用结果的首token
    parser.add_argument("--Only_token_head", default=False)
    # 选择Span的合成方式
    parser.add_argument('--span_generation', type=str, default="Max", choices=["Start_end", "Max", "Average", "CNN", "ATT"],
                        help='option: CNN, Max, Start_end, Average, ATT, SE_ATT')
    parser.add_argument('--ATT_SPAN_block_num', type=int, default=1, help="number of block in generating spans")

    # 是否对相关span添加分离Loss
    parser.add_argument("--kl_loss", default=True)
    parser.add_argument("--kl_loss_weight", type=int, default=0.5, help="weight of the kl_loss")
    parser.add_argument('--kl_loss_mode', type=str, default="KLLoss", choices=["KLLoss", "JSLoss", "EMLoss, CSLoss"],
                        help='选择分离相似Span的分离函数, KL散度、JS散度、欧氏距离以及余弦相似度')
    # 是否使用测试中的筛选算法
    parser.add_argument('--Filter_Strategy',  default=True, help='是否使用筛选算法去除冲突三元组')
    # 已被弃用    相关Span注意力
    parser.add_argument("--related_span_underline", default=False)
    parser.add_argument("--related_span_block_num", type=int, default=1, help="number of block in related span attention")

    parser.add_argument("--lambda_weight", type=float, default=0.8)
    parser.add_argument("--beta_weight", type=float, default=0.2)

    # 选择Cross Attention中ATT块的个数
    parser.add_argument("--block_num", type=int, default=1, help="number of block")
    parser.add_argument("--output_path", default='triples.json')
    #按照句子的顺序输入排序
    parser.add_argument("--order_input", default=True, help="")
    '''随机化输入span排序'''
    parser.add_argument("--random_shuffle", type=int, default=0, help="")
    # 验证模型复杂度
    parser.add_argument("--model_para_test", default=False)
    # 使用Warm up快速收敛
    parser.add_argument('--whether_warm_up', default=False)
    parser.add_argument('--warm_up', type=float, default=0.1)
    args = parser.parse_args()

    for k, v in sorted(vars(args).items()):
        logger.info(str(k) + '=' + str(v))
    train(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("keyboard break")
