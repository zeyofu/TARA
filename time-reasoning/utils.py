from functools import partial
import torch
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import sys
import shutil
from torch._six import string_classes
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format, default_collate
import matplotlib.pyplot as plt
import io
import json
import PIL
from PIL import Image

def search_jsonl(file, key, query):

    line_no = 1
    with open(file, 'r') as f:

        for each in f.readlines():

            dict = json.loads(each.strip('\n'))

            if dict[key] == query:
                print(f'{line_no}')
                break


            line_no += 1




def get_indices_given_split(jsonl_list, url_to_idx):

    '''


    :param jsonl_list: [train.jsonl, dev.jsonl, test.jsonl] full path
    :param url_to_idx: mapping from image_url: idx recorded in current nyt dataset
    :return:
    train_idx: set of indices for train
    dev_idx: set of indices for dev
    test_idx: set of indices for test

    stats: percent of total dataset for each
    overlap: intersection between other two for that position
    '''


    train_path = jsonl_list[0]
    dev_path = jsonl_list[1]
    test_path = jsonl_list[2]

    failures = {'train': 0, 'test': 0, 'dev': 0}

    train_idx, dev_idx, test_idx = [], [], []
    for line_no, line in enumerate(open(train_path, 'r')):
        cur_dict = json.loads(line.strip())
        try:
            train_idx.append(url_to_idx[cur_dict['image_url']])
        except KeyError:

            failures['train']  += 1



    for line_no, line in enumerate(open(dev_path, 'r')):
        cur_dict = json.loads(line.strip())

        try:
            dev_idx.append(url_to_idx[cur_dict['image_url']])

        except KeyError:

            failures['dev'] += 1


    for line_no, line in enumerate(open(test_path, 'r')):
        cur_dict = json.loads(line.strip())

        try:
            test_idx.append(url_to_idx[cur_dict['image_url']])

        except KeyError:

            failures['test'] += 1

    total_len = len(train_idx) + len(dev_idx) + len(test_idx)
    stats = (len(train_idx)/total_len, len(dev_idx)/total_len, len(test_idx)/total_len)
    overlap = (set(dev_idx).intersection(set(test_idx)), set(train_idx).intersection(set(test_idx)), set(train_idx).intersection(set(dev_idx)))


    return train_idx, dev_idx, test_idx, stats, overlap, failures

def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, dict):
        return_dict = {}
        for key in elem:
            return_dict[key] = default_collate([d[key] for d in batch])

        return return_dict
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


if __name__ == '__main__':

    search_jsonl('input_zip_nyt_dataset/details/train.jsonl', 'date', '1970s')