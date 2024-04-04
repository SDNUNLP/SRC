import os
import sys

sys.path.append('./')
from transformers import AutoTokenizer
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import unicodedata, re


def get_tokenizer(bert_model_path):
    """[unused1] token"""
    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    return tokenizer


def generate_label2id(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    label2id = {}
    for line in lines:
        line_split = line.strip().split()
        if len(line_split) > 1:
            label2id[line_split[-1]] = len(label2id)
    return label2id


def process_nerlabel(label2id):
    # label2id,id2label,num_labels = tools.load_schema_ner()
    # Since different ner dataset has different entity categories, it is inappropriate to pre-assign entity labels
    new_ = {}
    new_ = {'O': 0}
    for label in label2id:
        if label != 'O':
            label = '-'.join(label.split('-')[1:])
            if label not in new_:
                new_[label] = len(new_)
    return new_


class token_rematch:
    def __init__(self):
        self._do_lower_case = True

    @staticmethod
    def stem(token):
        """strip ##
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_control(ch):
        """control token process
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """other symbolic token
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """token mapping
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


def search(pattern, sequence):
    """find sub pattern
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def load_schema_dp(mode):
    # load schema
    assert mode in ['dp', 'pos'], mode

    if mode == 'pos':
        label2id = {
            'NOUN': 1,
            'PROPN': 2,
            'VERB': 3,
            'ADJ': 4,
            'ADV': 5,
            'DET': 6,
            'NUM': 7,
            'PUNCT': 8,
            'PRON': 9,
            'ADP': 10,
            'AUX': 11,
            'INTJ': 12,
            'SCONJ': 13,
            'PART': 14,
            'X': 15,
            'CCONJ': 16,
            'SYM': 17,
            'other': 18,
            }

        id2label = {id_: tag for tag, id_ in label2id.items()}

        num_labels = 18


    elif mode == 'dp':
        label2id = {
            'punct': 1,
            'nsubj': 2,
            'ROOT': 3,
            'det': 4,
            'advmod': 5,
            'prep': 6,
            'dobj': 7,
            'pobj': 8,
            'amod': 9,
            'cc': 10,
            'conj': 11,
            'aux': 12,
            'compound': 13,
            'poss': 14,
            'acomp': 15,
            'ccomp': 16,
            'advcl': 17,
            'mark': 18,
            'attr': 19,
            'xcomp': 20,
            'relcl': 21,
            'neg': 22,
            'npadvmod': 23,
            'nummod': 24,
            'prt': 25,
            'pcomp': 26,
            'appos': 27,
            'case': 28,
            'auxpass': 29,
            'nmod': 30,
            'nsubjpass': 31,
            'dep': 32,
            'acl': 33,
            'intj': 34,
            'dative': 35,
            'predet': 36,
            'quantmod': 37,
            'expl': 38,
            'oprd': 39,
            'parataxis': 40,
            'preconj': 41,
            'csubj': 42,
            'agent': 43,
            'meta': 44,
            'csubjpass': 45

        }

        id2label = {id_: tag for tag, id_ in label2id.items()}

        num_labels = 45

    return label2id, id2label, num_labels


# def load_schema_ner():
#     label2id={
#     'ORG_whole': 1,
#     'ORG_sub': 2,
#     'WORK_OF_ART_whole': 3,
#     'WORK_OF_ART_sub': 4,
#     'LOC_whole': 5,
#     'LOC_sub': 6,
#     'CARDINAL_whole': 7,
#     'CARDINAL_sub': 8,
#     'EVENT_whole': 9,
#     'EVENT_sub': 10,
#     'NORP_whole': 11,
#     'NORP_sub': 12,
#     'GPE_whole': 13,
#     'GPE_sub': 14,
#     'DATE_whole': 15,
#     'DATE_sub': 16,
#     'PERSON_whole': 17,
#     'PERSON_sub': 18,
#     'FAC_whole': 19,
#     'FAC_sub': 20,
#     'QUANTITY_whole': 21,
#     'QUANTITY_sub': 22,
#     'ORDINAL_whole': 23,
#     'ORDINAL_sub': 24,
#     'TIME_whole': 25,
#     'TIME_sub': 26,
#     'PRODUCT_whole': 27,
#     'PRODUCT_sub': 28,
#     'PERCENT_whole': 29,
#     'PERCENT_sub': 30,
#     'MONEY_whole': 31,
#     'MONEY_sub': 32,
#     'LAW_whole': 33,
#     'LAW_sub': 34,
#     'LANGUAGE_whole': 35,
#     'LANGUAGE_sub': 36}

#     id2label={
#     1: 'ORG_whole',
#     2: 'ORG_sub',
#     3: 'WORK_OF_ART_whole',
#     4: 'WORK_OF_ART_sub',
#     5: 'LOC_whole',
#     6: 'LOC_sub',
#     7: 'CARDINAL_whole',
#     8: 'CARDINAL_sub',
#     9: 'EVENT_whole',
#     10: 'EVENT_sub',
#     11: 'NORP_whole',
#     12: 'NORP_sub',
#     13: 'GPE_whole',
#     14: 'GPE_sub',
#     15: 'DATE_whole',
#     16: 'DATE_sub',
#     17: 'PERSON_whole',
#     18: 'PERSON_sub',
#     19: 'FAC_whole',
#     20: 'FAC_sub',
#     21: 'QUANTITY_whole',
#     22: 'QUANTITY_sub',
#     23: 'ORDINAL_whole',
#     24: 'ORDINAL_sub',
#     25: 'TIME_whole',
#     26: 'TIME_sub',
#     27: 'PRODUCT_whole',
#     28: 'PRODUCT_sub',
#     29: 'PERCENT_whole',
#     30: 'PERCENT_sub',
#     31: 'MONEY_whole',
#     32: 'MONEY_sub',
#     33: 'LAW_whole',
#     34: 'LAW_sub',
#     35: 'LANGUAGE_whole',
#     36: 'LANGUAGE_sub'}

#     num_labels=36

#     return label2id,id2label,num_labels

def load_schema_ner():
    id2label = {1: 'ORG_whole',
                2: 'WORK_OF_ART_whole',
                3: 'LOC_whole',
                4: 'CARDINAL_whole',
                5: 'EVENT_whole',
                6: 'NORP_whole',
                7: 'GPE_whole',
                8: 'DATE_whole',
                9: 'PERSON_whole',
                10: 'FAC_whole',
                11: 'QUANTITY_whole',
                12: 'ORDINAL_whole',
                13: 'TIME_whole',
                14: 'PRODUCT_whole',
                15: 'PERCENT_whole',
                16: 'MONEY_whole',
                17: 'LAW_whole',
                18: 'LANGUAGE_whole'
                }
    label2id = {k: v for v, k in id2label.items()}

    num_labels = len(label2id)

    return label2id, id2label, num_labels


def batch_to_device(tensor_dicts, device):
    for key in tensor_dicts.keys():
        tensor_dicts[key].to(device)


def solve_wordpiece(last_hidden_state, sen):
    '''
    sen is a sentence in type of string
    last_hidden_state not contains CLS and SEP
    tokenizer.tokenize is not equal to tokenizer.convert_tokens_to_ids
    like sentence : 'Total shares to be offered 0.0 million'
    input_ids :[8653, 6117, 1106, 1129, 2356, 121, 119, 121, 1550]
    word_ids : [8653, 6117, 1106, 1129, 2356, 100, 1550]           100 means UNK
    '''
    new_state = []
    if type(sen) == str:
        sentence_list = sen.split(' ')
    else:
        sentence_list = sen
    j = 0
    for i in range(len(sentence_list)):
        token = sentence_list[i]
        tokens = tokenizer.tokenize(token)
        piece_length = len(tokens)
        new_state.append(torch.mean(last_hidden_state[j:j + piece_length], dim=0, keepdims=True))
        j += piece_length
    new_state = torch.vstack(new_state)
    assert new_state.size(0) == len(sentence_list)
    return new_state

