import torch
from torch import nn

import logging

logger = logging.getLogger('main.span_loss')


class Span_loss(nn.Module):
    def __init__(self, num_label, class_weight=None):
        super().__init__()
        self.num_label = num_label
        if class_weight != None:
            self.class_weight = torch.FloatTensor(class_weight)
            logger.info("Class weight : {}".format(self.class_weight.cpu().tolist()))
            self.loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weight)  # reduction='mean'
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()  # reduction='mean'

    def forward(self, span_logits, span_label, span_mask):
        '''
        span_logits.size()==(bsz,max_length,max_length,num_labels)
        span_label.size()==span_mask.size()==(bsz,max_length,max_length)
        '''
        # print(span_mask.size(),'spanmask.size()')
        mask_pad = span_mask.view(-1) == 1
        span_label = span_label.view(size=(-1,))[mask_pad]  # (bsz*max_length*max_length,)
        span_logits = span_logits.view(size=(-1, self.num_label))[mask_pad]  # (bsz*max_length*max_length,num_labels)
        span_loss = self.loss_func(input=span_logits, target=span_label)  # (bsz*max_length*max_length,)

        # print("span_logits : ",span_logits.size())
        # print("span_label : ",span_label.size())
        # print("span_mask : ",span_mask.size())
        # print("span_loss : ",span_loss.size())

        # start_extend = span_mask.unsqueeze(2).expand(-1, -1, seq_len)
        # end_extend = span_mask.unsqueeze(1).expand(-1, seq_len, -1)
        # span_mask = span_mask.view(size=(-1,))#(bsz*max_length*max_length,)
        # span_loss *=span_mask
        # avg_se_loss = torch.sum(span_loss) / span_mask.size()[0]
        # avg_se_loss = torch.sum(span_loss) / torch.sum(span_mask).item()
        # # avg_se_loss = torch.sum(sum_loss) / bsz
        # return avg_se_loss
        return span_loss




