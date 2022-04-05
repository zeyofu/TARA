from __future__ import print_function, division
import argparse
import os
import json
import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torch
import requests
import numpy as np
import random
random.seed(138)
CONTINENTS = {
    'NA': 'North America',
    'SA': 'South America',
    'AS': 'Asia',
    'OC': 'Oceania',
    'AF': 'Africa',
    'EU': 'Europe'
}
import datetime, calendar
MONTHS = calendar.month_name
MONTHS2num = {month: index for index, month in enumerate(MONTHS) if month}
image_base_url = f'https://static01.nyt.com/'


def load_jsonl(file_name):
    print(f'loading file {file_name}')
    data = []
    with open(file_name, 'r') as f:
        for line in tqdm(f):
            data.append(json.loads(line.strip()))
    return data


def get_gold_data(data_folder):
    train_path = data_folder + 'train.jsonl'
    test_path = data_folder + 'gold_test.jsonl'
    dev_path = data_folder + 'gold_dev.jsonl'
    print(f'loading data from {data_folder}')
    trains = load_jsonl(train_path)
    tests = load_jsonl(test_path)
    devs = load_jsonl(dev_path)
    return trains, devs, tests


def time_label_2natural(date):
  # '2021-5-21' to format 'May 21, 2020'
  date = date.split('-')
  if len(date) == 3:
      year, month, day = date
      return f'{MONTHS[int(month)]} {day}, {year}'
  if len(date) == 2:
      year, month = date
      return f'{MONTHS[int(month)]}, {year}'
  if len(date) == 1:
      year = date[0]
      return f'{year}'


def natural2time_label(date):
    # May 21, 2020 to format 2021-5-21
    date = date.replace('a photo taken in ', '')
    date = date.split(', ')
    if len(date) == 1:
        year = date[0]
        return f'{year}'
    else:
        monthday, year = date
        monthday = monthday.split()
        if len(monthday) == 2:
            month, day = monthday
            return f'{year}-{MONTHS2num[month]}-{int(day)}'
        elif len(monthday) == 1:
            month = monthday[0]
            return f'{year}-{MONTHS2num[month]}'


def nyt_image_url_to_name(url):
    name = url.replace(image_base_url, '').replace('/', '-')
    if not name.endswith('.jpg'):
      name = name + '.jpg'
    return name


class VisionLangDataset(Dataset):
    """
    """
    def __init__(self, data, img_dir, label_name, labels2id, transform=None):
        valid_data = [k for k in data if k[label_name]]
        self.data = valid_data
        self.img_dir = img_dir
        self.transform = transform
        self.label_name = label_name
        self.l2id = labels2id
        self.prepare_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        url = self.data[idx]['image_url']
        img_path = os.path.join(self.img_dir, nyt_image_url_to_name(url))
        # download image if not exist
        if not os.path.exists(img_path):
            response = requests.get(url, stream=True)
            with open(img_path, 'wb') as handle:
                for block in response.iter_content(1024):
                    if not block:
                        break
                    handle.write(block)
        if not os.path.exists(img_path):
            print(f'Cannot download image {img_path}')
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text_label = self.data[idx][self.label_name]
        # label = self.l2id[text_label]
        label = text_label
        if 'time' in self.label_name or 'date' in self.label_name:
            text_label = time_label_2natural(text_label)
        text_label = "a photo taken in " + text_label
        abstract = self.data[idx]["abstract"]
        return image, abstract, text_label, label

    def prepare_images(self):
        for idx in tqdm(range(len(self))):
            url = self.data[idx]['image_url']
            img_path = os.path.join(self.img_dir, nyt_image_url_to_name(url))
            if not os.path.exists(img_path):
                response = requests.get(url, stream=True)
                with open(img_path, 'wb') as handle:
                    for block in response.iter_content(1024):
                        if not block:
                            break
                        handle.write(block)
            if not os.path.exists(img_path):
                print(f'Cannot download image {img_path}')


def get_accuracy(probs, ground_truth, N, type=''):
    _, top_labels1 = probs.topk(1, dim=-1)
    hit1 = len([1 for i in range(N) if ground_truth[i] in top_labels1[i]]) / N
    print(f'{type} with {probs.shape[1]} unique labels, accuracy is {100 * hit1:.2f}')


def get_f1(probs, ground_truth, N, type, label_name, id2label=None):
    _, top_labels1 = probs.topk(1, dim=-1)
    y_pred = [id2label[top_labels1[i]] for i in range(N)]
    y_true = ground_truth #[id2label[ground_truth[i]] for i in range(N)]
    if 'location' in label_name:
        hier_fun = get_hierarchical_geo_labels
    else:
        y_pred = [natural2time_label(i) for i in y_pred]
        y_true = [natural2time_label(i) for i in y_true]
        hier_fun = get_hierarchical_time_labels
    y_pred_hier = [hier_fun(i) for i in y_pred]
    y_true_hier = [hier_fun(i) for i in y_true]
    print(f'{type} with {probs.shape[1]} unique labels, Example F1 is {100 * example_f1(y_pred_hier, y_true_hier):.2f}\t')


def example_f1(y_pred_hier, y_true_hier):
    f1 = 0
    N = len(y_pred_hier)
    # print(N, y_pred_hier[0], y_true_hier[0])
    for i in range(N):
        inter = set([k for k in y_pred_hier[i] if k in y_true_hier[i]])
        f1 += 2 * len(inter) / (len(y_pred_hier[i]) + len(y_true_hier[i]))
    f1 = f1 / N
    return f1


def get_decade(year):
    return f'{str(year)[:-1]}0s'


def get_century(year):
    # If year is between 1 to 100 it will come in 1st century
    if year <= 100:
        return "1st century"
    elif year % 100 == 0:
        return f'{year // 100} century'
    else:
        return f'{year // 100 + 1} century'


def get_hierarchical_time_labels(date):
    if 'century' in date:
        return [date]
    if date[-1] == 's':
        return [f'{get_century(int(date[:-1]))}', date]
    date = date.split('-')
    date = [MONTHS2num[k] if k in MONTHS2num else k for k in date]
    date = list(map(int, date))
    labels = None
    if len(date) == 3:
        year, month, day = date
        labels = [f'{get_century(year)}', f'{get_decade(year)}', f'{year}', f'{year}-{month}', f'{year}-{month}-{day}']
    if len(date) == 2:
        year, month = date
        labels = [f'{get_century(year)}', f'{get_decade(year)}', f'{year}', f'{year}-{month}']
    if len(date) == 1:
        year = date[0]
        labels = [f'{get_century(year)}', f'{get_decade(year)}', f'{year}']
    return labels


def get_hierarchical_geo_labels(loc):
    locs = loc.split(', ')
    if len(locs) > 3:
        return [', '.join(locs[:-2]), locs[-2], locs[-1]]
    else:
        return locs


def run(args):
    input_data_folder = f'data/{args.dataset_name}_preprocessed/input/'
    input_image_folder = f'data/{args.dataset_name}_preprocessed/images/'
    print(f'start baseline evaluation: reasoning with clip on {args.dataset_name}')

    device = args.device
    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=args.device)
    clip_model.eval()

    if args.data_type == 'test':
        trains, devs, tests = get_gold_data(data_folder=input_data_folder)
    elif args.data_type == 'interest':
        trains, devs, tests = [], [], load_jsonl(os.path.join(input_data_folder, 'gold_interest.jsonl'))
    if args.database:
        database_path = os.path.join(input_data_folder, args.database_name+'.jsonl')
        databases = [d for d in load_jsonl(database_path) if d[args.label_name]]
    else:
        databases = []
    train_label = {'gold_location_suggest': 'location', 'gold_time_suggest': 'date', 'location': 'location', 'gold_time': 'date', 'date': 'date'}
    labels = [item[train_label[args.label_name]] for item in trains]
    labels += [item[args.label_name] for item in devs + tests + databases]
    id2l = list(set(labels))
    l2id = {l: i for i, l in enumerate(id2l)}

    # train_dataset = VisionLangDataset(trains, input_image_folder, args.label_name, l2id, transform=clip_preprocess)
    # dev_dataset = VisionLangDataset(devs, input_image_folder, args.label_name, l2id, transform=clip_preprocess)
    test_dataset = VisionLangDataset(tests, input_image_folder, args.label_name, l2id, transform=clip_preprocess)
    db_dataset = VisionLangDataset(databases, input_image_folder, args.label_name, l2id, transform=clip_preprocess)

    image_inputs = None
    labels = None
    text_labels, wdb_text_labels = None, None
    abstracts, wdb_abstracts = None, None
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
    for image, abstract, text_label, label in tqdm(test_dataloader):
        image_inputs = image.to(device)
        text_labels = text_label
        wdb_text_labels = text_labels
        abstracts = abstract
        wdb_abstracts = abstract
        labels = label
    if args.database:
        for _, abstract, text_label, label in tqdm(DataLoader(db_dataset, batch_size=1024)):
            wdb_text_labels += text_label
            wdb_abstracts += abstract
            # labels += label
    id2text_labels = list(set(wdb_text_labels))
    text_labels2id = {k: i for i, k in enumerate(id2text_labels)}
    text_labels_tokens = clip.tokenize(id2text_labels, truncate=True).to(device)
    abstracts_tokens = clip.tokenize(wdb_abstracts, truncate=True).to(device)
    ground_truth_text_labels = [text_labels2id[k] for k in text_labels]

    all_text_labels_tokens = clip.tokenize(wdb_text_labels, truncate=True).to(device)
    N = len(test_dataset)
    ground_truth = np.arange(N)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_inputs).float()
        text_labels_features = clip_model.encode_text(text_labels_tokens).float()
        all_text_labels_features = clip_model.encode_text(all_text_labels_tokens).float()
        abstracts_features = clip_model.encode_text(abstracts_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_labels_features /= text_labels_features.norm(dim=-1, keepdim=True)
    all_text_labels_features /= all_text_labels_features.norm(dim=-1, keepdim=True)
    abstracts_features /= abstracts_features.norm(dim=-1, keepdim=True)

    # pure label
    text_labels_logits = (100.0 * image_features @ text_labels_features.T)
    text_labels_probs = text_labels_logits.softmax(dim=-1).cpu()
    get_accuracy(text_labels_probs, ground_truth_text_labels, N, 'clip mult with labels,')
    get_f1(text_labels_probs, text_labels, N, 'clip mult with labels,', args.label_name, id2text_labels)

    # # abstract
    # abstracts_logits = (100.0 * image_features @ abstracts_features.T)
    # abstracts_probs = abstracts_logits.softmax(dim=-1).cpu()
    # get_accuracy(abstracts_probs, ground_truth, N, 'clip mult with abstracts,')
    # get_f1(abstracts_probs, labels, N, 'clip mult with abstracts,', args.label_name, labels)

    print('done')


if __name__ == '__main__':

    # dataset_name = 'wit'
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset_name', type=str, default='nyt', help='dataset folder name')
    parser.add_argument('-data', '--data_type', type=str, default='test', choices=['test', 'interest'], help='test data or test set of interest')
    parser.add_argument('-clip', '--clip_model_name', type=str, default='ViT-B/32', help='model name or path')
    parser.add_argument('-lb', '--label_name', type=str, default='gold_location_suggest', help='')
    parser.add_argument('-db', '--database', type=int, default=0, help='1 when you test on interest. Because we want to keep the total unique label number to be the same.')
    parser.add_argument('-dbn', '--database_name', type=str, default='gold_test', help='.jsonl file')
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args)
    run(args)


