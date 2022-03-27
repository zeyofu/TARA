import json
import os
import random
import re
from PIL import Image
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import argparse
import pickle
from utils import custom_collate, get_indices_given_split
from typing import List, Dict
from tqdm import tqdm
from torch.optim import SGD
import re
import nltk
import cv2
nltk.download('punkt')
from random import sample
import numpy as np
import pandas as pd
import torch
torch.autograd.set_detect_anomaly(True)
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils import data
import datetime
import pandas as pd
import calendar
from torchvision import transforms
from tqdm import tqdm
import clip
import functools
from functools import total_ordering
from typing import Callable
from datapoints import TimePoint, Location
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn.functional as F_nn
from torch.profiler import profile, record_function, ProfilerAction, ProfilerActivity
#rom policy import SamplingPolicy, get_example_time_policy, get_example_geo_policy

month_to_number = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

def recover_time_params(given_string, time_template_type):

    '''
    :param given_string: String containing time information out of the retrieval set
    :param time_template_type: Template options used for making the string
    :return:

    day of week, day, month, year, decade reverse engineered through RegEx matching on the string
    '''

    string = given_string.strip("This picture was taken ")
    if time_template_type == 0 or time_template_type == 1:
        time_list = string.split(',')
        if len(time_list) == 3:
            dow = None
            day = time_list[0]
            month = month_to_number[time_list[1]]
            year = time_list[2]
            decade = str(int(time_list[2]) - int(time_list[2])%10) + 's'

        elif len(time_list) == 2:

            dow = None
            day = None
            month = month_to_number[time_list[0]]
            year = time_list[1]
            decade = str(int(time_list[1]) - int(time_list[1])%10) + 's'

        elif len(time_list) == 3:

            if 's' in time_list[0]:

                dow = None
                day = None
                month = None
                year = None
                decade = time_list[0]

            else:

                dow = None
                day = None
                month = None
                year = time_list[0]
                decade = str(int(time_list[0]) - int(time_list[0])%10) + 's'

    elif time_template_type == 2:

        m1 = re.match("on (.*), which was (.*) the (.*) of the year (.*).", string)
        if m1 is not None:
            dow = m1.group(1)
            day = m1.group(3).strip('th').strip('rd').strip('st').strip('nd')
            month = m1.group(2)
            year = m1.group(4)
            decade = str(int(year) - int(year)%10) + 's'


        m2 = re.match("during (.*) of the year (.*)", string)
        if m2 is not None:
            dow = None
            day = None
            month = m2.group(1)
            year = m2.group(2)
            decade = str(int(year) - int(year)%10) + 's'


        m3 = re.match("during the year (.*)", string)
        if m3 is not None:
            dow = None
            day = None
            month = None
            year = m3.group(1)
            decade = str(int(year) - int(year)%10) + 's'


        m4 = re.match("during the (.*)", string)
        if m4 is not None and m3 is None:
            dow = None
            day = None
            month = None
            year = None
            decade = str(m4.group(1)) + 's'

    else:
        raise Exception("Must choose between 0, 1 and 2")

    return dow, day, month, year, decade

def produce_answer_at_level_time(query, gt_day, gt_month, gt_year, gt_decade):

    '''
    Function checks if a given query is compatible as a positive.

    :param query:  query as a string
    :param gt_day: groundtruth day
    :param gt_month: groundtruth month
    :param gt_year: groundtruth year
    :param gt_decade: groundtruth decade
    :return:

    Boolean indicating if the given string is a positive.
    '''


    _, l_day, l_month, l_year, l_decade = recover_time_params(query, 2)

    natural_hierarchy = 0
    for each in [gt_day, gt_month, gt_year, gt_decade]:

        if each is None:
            natural_hierarchy += 1

        else:
            break

    required_level = natural_hierarchy

    only_day_match = l_day == gt_day and gt_day is not None
    only_month_match = l_month == gt_month and gt_month is not None
    only_year_match = l_year == gt_year and gt_year is not None
    only_decade_match = l_decade == gt_decade and gt_decade is not None

    if required_level == 0:

        if only_day_match and only_month_match and only_decade_match and only_year_match:

            return True

        else:
            return  False

    if required_level == 1:
        if only_month_match and only_year_match and only_decade_match:
            return True

        else:
            return False

    if required_level == 2:
        if only_year_match and only_decade_match:
            return True
        else:
            return False

    if required_level == 3:
        if only_decade_match:
            return True

        else:
            return False

    return False


    '''
    
    if level == 0:
        if (l_day == gt_day) and gt_day is not None:
            return True
        else:
            return False

    if level == 1:
        if l_month == gt_month and gt_month is not None:
            return True

        else:
            return False

    if level == 2:
        if l_year == gt_year and gt_year is not None:
            return True

        else:
            return False

    if level == 3:
        if l_decade == gt_decade and gt_decade is not None:
            return True

        else:
            return False
            
    '''


def produce_answer_at_level_geo(query, gt_full_string, gt_country, gt_continent):

    l = Location(query)

    comma_count = gt_full_string.count(',')
    if comma_count == 0:
        natural_hierarchy = 2
    elif comma_count == 1:
        natural_hierarchy = 1
    else:
        natural_hierarchy = 0

    required_level = natural_hierarchy

    if required_level == 0:
        if l.full_str == gt_full_string and l.full_str is not None:
            return True
        else:
            return False
    if required_level == 1:
        if l.country == gt_country and l.country is not None:
            return True

        else:
            return False
    if required_level == 2:
        if l.continent == gt_continent and l.continent is not None:
            return True
        else:
            return False





number_to_month = {v: k for k, v in month_to_number.items()}


def get_name_from_url(img_url):
    image_name = '_'.join(img_url.split('/')[-7:]).split("?")[0]
    if '.jpg' in image_name:
        image_name = image_name.strip('.jpg')
    if '.png' in image_name:
        image_name = image_name.strip('.png')

    return f'{image_name}.png'

def get_name_from_url_input_zip(img_url):

    return os.path.splitext(img_url.replace('/', '_'))[0] + '.png'


def pil_loader(path):


    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    except:
        print(f'Error with the image at {path}')
        return Image.new('RGB', (300, 300))

'''
Define time granularity key:

0: full date given (1, 2, 3 are possible parents)
1: only month and year (2,3 are possible parents)
2: only year given (3 is possible parent)
3: only era/decade (no hierarchies possible)

The dataset below assumes that granularity of YYYY-MM-DD will be available. Significant changes will be required otherwise
'''

image_base_url = f'https://static01.nyt.com/'

def url_to_name_xingyu(url):
    name = url.replace(image_base_url, '').replace('/', '-')
    if not name.endswith('.jpg'):
        name = name + '.jpg'
    return name


def url_to_name_wiki(url):
    return url.split('/')[-1]


def url_to_ided_name_ben(url, num):
    name = url.replace(image_base_url, '').replace('/', '-')
    if not name.endswith('.jpg'):
        name = name + '.jpg'
    return name[:-4] + "_{}.jpg".format(str(num))


class GeoTemporalDataset(data.Dataset):

    def __init__(self, model_name, exp_type, data_dir='/shared/xzhou45/time-reasoning/wiki', template_options=(0, 0), ratio=10, mode='test', sample=False, encoding_pickles=False, num_negatives=19):

        random.seed(42)
        self.mode=mode
        self.exp_type = exp_type #1 is time experiment, 2 is geo experiment
        self.data_path = os.path.join(os.getcwd(), data_dir)
        self.labels_path = self.data_path
        if not os.path.exists(self.data_path):
            raise Exception("This dataset could not be found")

        self.time_template_options = template_options[0]
        self.geo_template_options =  template_options[1]

        #self.url_to_idx, self.idx_to_url = self.fix_index(os.path.join(self.data_path, 'index.txt'))
        self.root_info, self.timeline, self.idx_to_file, self.idx_linking, self.url_to_idx = self._load_root_info_input_zip(self.labels_path)

        self.rng = np.random.RandomState(42)
        self.clip_model_name = model_name


        #_, self.preprocess = clip.load(self.clip_model_name, device=self.device)
        self.tokenize_fn = self._get_tokenize_fn()
        self.pos_to_neg_ratio = ratio
        self.sample = sample

        self.all_negatives = self.sample_from_full_negatives('full')

        self.num_negatives = num_negatives

        print(self.num_negatives)
        if encoding_pickles:
            quit()



    def __len__(self):

        return len(self.timeline)


    def _load_root_info_input_zip(self, path):

        if os.path.exists('root_info.pkl') and os.path.exists('timeline.pkl'):

            with open('root_info.pkl', 'rb') as root_file:
                root_info = pickle.load(root_file)

            with open('timeline.pkl', 'rb') as timeline_file:
                timeline = pickle.load(timeline_file)

            with open('idx_to_file.pkl', 'rb') as idx_file:

                idx_to_file = pickle.load(idx_file)

            with open('idx_linking.pkl', 'rb') as link_file:

                idx_linking = pickle.load(link_file)

            with open('url_to_idx.pkl', 'rb') as url_file:

                url_to_idx = pickle.load(url_file)


        else:

            root_info = {} #given current folder name, contains a mapping from idx to the created data point from that image
            idx_linking = {} #given idx in the dataset, returns which file and what line no it came from
            idx_to_file = {} #given idx in the dataset, returns the folder and the file name associated with it
            url_to_idx = {} #given image url returns the idx associated with that url
            timeline = []
            idx_added = 0
            # paths = [os.path.join(path, each_path) for each_path in ['train.jsonl', 'dev.jsonl', 'test.jsonl']]
            paths = [os.path.join(path, each_path) for each_path in ['train.jsonl']]

            for each_file in paths:

                current_folder_name = os.path.splitext(os.path.basename(each_file))[0]
                root_info[current_folder_name] = {}

                for line_no, line in enumerate(tqdm(open(os.path.join(path, each_file), 'r'))):

                    print(f'Evaluating {line_no} in {each_file}')

                    cur_dict = json.loads(line.strip())
                    url_to_idx[cur_dict['image_url']] = idx_added
                    # idx_to_file[idx_added] = (current_folder_name, get_name_from_url_input_zip(cur_dict['image_url']))
                    idx_to_file[idx_added] = ("/shared/xingyu/projects/visual/data/wit_preprocessed/images/", url_to_name_wiki(cur_dict['image_url']))
                    data_point = TimePoint(cur_dict, idx_added, current_folder_name)
                    root_info[current_folder_name].update({idx_added: data_point})
                    idx_linking[idx_added] = (each_file, line_no)
                    idx_added += 1
                    timeline.append(data_point)

                    # face_dir = "/shared/xingyu/projects/visual/data/nyt_preprocessed/face_outputs/"
                    # for face_i in range(0, 10):
                    #     if os.path.isfile(face_dir + url_to_ided_name_ben(cur_dict['image_url'], face_i)):
                    #         url_to_idx[cur_dict['image_url']] = idx_added
                    #         idx_to_file[idx_added] = (face_dir, url_to_ided_name_ben(cur_dict['image_url'], face_i))
                    #         data_point = TimePoint(cur_dict, idx_added, current_folder_name)
                    #         root_info[current_folder_name].update({idx_added: data_point})
                    #         idx_linking[idx_added] = (each_file, line_no)
                    #         idx_added += 1
                    #         timeline.append(data_point)
                    #         print("found face")

                    # face_dir = "/shared/xingyu/projects/visual/data/nyt_preprocessed/object_outputs/"
                    # for face_i in range(0, 10):
                    #     if os.path.isfile(face_dir + url_to_ided_name_ben(cur_dict['image_url'], face_i)):
                    #         url_to_idx[cur_dict['image_url']] = idx_added
                    #         idx_to_file[idx_added] = (face_dir, url_to_ided_name_ben(cur_dict['image_url'], face_i))
                    #         data_point = TimePoint(cur_dict, idx_added, current_folder_name)
                    #         root_info[current_folder_name].update({idx_added: data_point})
                    #         idx_linking[idx_added] = (each_file, line_no)
                    #         idx_added += 1
                    #         timeline.append(data_point)
                    #         print("found obj")


            #TODO: Commenting this is dangerous. Find a way to remove this comment.
            #timeline = sorted(timeline)

            for each_data_idx in range(0, len(timeline)):
                each_data = timeline[each_data_idx]
                root_info[each_data.root_file][each_data.idx].set_timeline_place(each_data_idx)

            with open('root_info.pkl', 'wb') as root_file:
                pickle.dump(root_info, root_file)

            with open('url_to_idx.pkl', 'wb') as url_file:
                pickle.dump(url_to_idx, url_file)

            with open('timeline.pkl', 'wb') as timeline_file:
                pickle.dump(timeline, timeline_file)

            with open('idx_to_file.pkl', 'wb') as index_file:
                pickle.dump(idx_to_file, index_file)

            with open('idx_linking.pkl', 'wb') as link_file:
                pickle.dump(idx_linking, link_file)


        return root_info, timeline, idx_to_file, idx_linking, url_to_idx

    def _get_tokenize_fn(self):

        '''Pre tokenize the image using either the groundtruth tokenizer associated with the CLIP model or a simple
        tokenizer in case another model is used. This method will be called once in __init__ and then locked

        Overload with perhaps a hugging face tokenizer. will be required for finetuning
        '''

        return clip.tokenize

    """
    def _sample_time_negatives(self, data_point, no_negatives, sampling_policy=None):

        '''
        Batch creation guidance. This indicates how a batch might be created at test time. One datapoint can be
        'faked' to be a collections of positives and negatives and this can be leveraged inside of the dataloader
        by requesting a smaller dataset size. This will give more control over the kind of batches that are produced
        :return:
        '''

        sample = []
        if sampling_policy is None:
            sampling_policy = get_example_time_policy()

        if not sampling_policy.type == 1:
            raise Exception("Policy defined is not a time-negative sampling policy")

        time_idx = data_point.timeline_place
        relevant_time = self.timeline[:time_idx] + self.timeline[(time_idx + 1):]

        values = SamplingPolicy.convert_policy_to_numbers(sampling_policy.percents, no_negatives)

        for value, filter_fn in zip(values, sampling_policy.lambdas):


            relevant_list = [point for point in relevant_time if filter_fn(point, data_point)]
            sample += random.sample(relevant_list, value)


        return sample


    def _sample_geo_negatives(self, data_point, no_negatives, sampling_policy=None):

        sample = []
        if sampling_policy is None:
            sampling_policy = get_example_geo_policy()

        if not sampling_policy.type == 2:
            raise Exception("Policy defined is not a geo-negative sampling policy")


        time_idx = data_point.timeline_place

        relevant_time = self.timeline[:time_idx] + self.timeline[(time_idx+1):]

        values = SamplingPolicy.convert_policy_to_numbers(sampling_policy.percents, no_negatives)

        for value, filter_fn in zip(values, sampling_policy.lambdas):

            if data_point.geo.continent is None:
                relevant_list = relevant_time

            else:
                relevant_list = [point for point in relevant_time if filter_fn(point, data_point)]

            sample += random.sample(relevant_list, value)


        return sample
    """

    def template_time_label(self, time_label, idx=0, basic_time_string=None):
        '''
        0 options envelopes the time-zone in the standard format
        1 options does not add any templating, simply returning the time value
        2 options converts time to the corresponding text time and then returns the associated label

        :param time_label: a datetime object
        :param idx: this param is passed down to convert time template
        :param basic_time_string: this param overrides the default basic_time_string \
                                    which simply creates the entire date- used for positive/negative sampling
        :return: Returns the datetime object wrapped inside of a template
        '''
        template = " on %s"

        if basic_time_string is None:
            basic_time_string = ','.join(str(time_label).split()[0].split('-')[::-1])

        if self.time_template_options == 0:
            return template%basic_time_string

        elif self.time_template_options == 1:
            return basic_time_string

        elif self.time_template_options == 2:
            return template%self.convert_time_template(time_label, idx)

        else:
            raise Exception("Must choose between 0, 1 and 2")


    def template_geo_label(self, geo_label):
        '''

        :param geo_label:
        :return:
        '''
        if self.geo_template_options == 1:
            template = "%s"

        else:
            template = "This picture was taken at the location: %s"

        return template%geo_label

    def get_hierarchical_time_labels(self, time_label, from_idx=0):

        '''
        Xingyu's function copied here for later: take's in a data point and returns the upper time hierarchy
        :param time_label: datetime object at the lowest level of the hierarchy
        :return: [collection] of hierarchical time positives as strings
        '''
        all_parents = []
        basic_time_list = str(time_label).split()[0].split('-')[::-1]

        for idx in range(1, len(basic_time_list)+1):

            if idx == len(basic_time_list):

                current_time_list = basic_time_list[-1:]
                current_time_string = str(int(current_time_list[0]) - int(current_time_list[0])%10) + 's'

            else:

                current_time_list = basic_time_list[idx:]
                current_time_string = ','.join(current_time_list)

            all_parents.append(current_time_string)

        return all_parents[from_idx:]


    def get_hierarchical_geo_labels(self, geo_label):

        '''
        Xingyu's function copied here for later: take's in a data point and returns the upper geo hierarchy
        (calling some geographical database). Switch to geolocator API in the future, and sample images from
        Google street view for geographical info
        :param data:
        :return: [collection] of hierarchical geo positives as strings



        if geo_label.country is None or geo_label.continent is None:

            current_full = ''.join([i for i in geo_label.full_str if not i.isdigit()])
            current_full = current_full.strip()
            labels = [(geo_label.full_str, 0)]

        else:
            if geo_label.rest is not None and geo_label.rest and geo_label.rest.strip():

                current_rest = ''.join([i for i in geo_label.rest if not i.isdigit()])
                current_rest = current_rest.strip()

                labels = [(f'{current_rest}, {geo_label.country}, {geo_label.continent}', 0)]

            else:

                labels = []

        '''

        labels = []
        if geo_label.rest is not None and geo_label.rest and geo_label.rest.strip():
            labels.append((f'{geo_label.full_str}', 0))

        if geo_label.country is not None:
            labels.append((f'{geo_label.country}, {geo_label.continent}', 1))

        if geo_label.continent is not None:
            labels.append((f'{geo_label.continent}', 2))

        return labels

    def pad_time_labels_set(self, given_set, required_pad):


        pad_vals = []
        decades = ['2010s', '2020s']
        years = [str(i) for i in range(2010, 2022)]
        months = ["%s,%s"%each for each in itertools.product([str(i).zfill(2) for i in range(1, 13)], years)]


        for each in decades:

            if each not in given_set:

                pad_vals.append([datetime.datetime(int(each.split('s')[0]), 1, 1),3,each])
                if (len(pad_vals) == required_pad):
                    return pad_vals

        for each in years:

            if each not in given_set:

                pad_vals.append([datetime.datetime(int(each), 1, 1), 2, each])
                if (len(pad_vals) == required_pad):
                    return pad_vals

        for each in months:

            if each not in given_set:

                pad_vals.append([datetime.datetime(int(each.split(',')[1]), int(each.split(',')[0]), 1), 1, each])
                if (len(pad_vals) == required_pad):
                    return pad_vals

        return pad_vals

    def pad_geo_labels_set(self, set, positive_loc, max_size):

        required_dataframe = positive_loc.country_frame
        while not len(set) == max_size:

            data_point = required_dataframe.iloc[random.sample(range(len(required_dataframe)), 1)[0]]
            if data_point['Country'] != positive_loc.country:

                country = data_point['Country']
                continent = data_point['Continent']

                set.add((self.template_geo_label(f'{country}, {continent}'), 1))

        return set


    def sample_from_full_negatives(self, number, correct_string=None):



        def mod(line):

            line_split = line.strip('\n').split('<SEP>')
            negative =line_split[0]
            hierarchy = int(line_split[1])

            return (negative, hierarchy)

        name = 'time' if self.exp_type == 1 else 'geo'
        sample_file = f'all_negatives_{name}_with_labels.txt'
        f = open(sample_file, 'r')
        all_negatives = [mod(line) for line in f.readlines()]


        if number == 'full':
            return set(all_negatives)

        all_negatives_final = set()

        new_number = number
        while len(all_negatives_final) != number:

            current_val = random.sample(all_negatives, new_number)

            if self.exp_type == 1:
                '''
                new_current_val = []
                for each in current_val:

                    if each[0] != correct_string:

                        new_current_val.append(each)

                current_val = new_current_val
                '''

                _, gt_day, gt_month, gt_year, gt_decade = recover_time_params(correct_string,
                                                                              self.time_template_options)

                new_current_val = []
                for each in current_val:

                    if not produce_answer_at_level_time(each[0], gt_day, gt_month, gt_year, gt_decade):

                        new_current_val.append(each)

                current_val = new_current_val





            if self.exp_type == 2:

                '''
                new_current_val = []
                for each in current_val:

                    if each[0] != correct_string:
                        new_current_val.append(each)

                current_val = new_current_val
                '''

                #keeping it as is because of cases like 'asia'
                c_loc = Location(correct_string)
                gt_full_string = c_loc.full_str
                gt_country = c_loc.country
                gt_continent = c_loc.continent


                new_current_val = []
                for each in current_val:


                    if not produce_answer_at_level_geo(each[0], gt_full_string, gt_country, gt_continent):

                        new_current_val.append(each)


                current_val = new_current_val


            all_negatives_final = all_negatives_final.union(set(current_val))
            new_number = number - len(all_negatives_final)

        return all_negatives_final


    def convert_time_template(self, datetime, idx=0):

        '''
        :param datetime: a datetime object
        :param idx: This controls how to express it (in month/date/year) format
        :return:
        '''

        endings = {1: 'st', 2: 'nd', 3: 'rd'}
        for key in [0, 4, 5, 6, 7, 8, 9]:
            endings.update({key: 'th'})


        end_key = int(datetime.strftime("%d"))%10

        if idx == 0:
            str_date = datetime.strftime("on %A, which was %B the %d")
            str_date += f'{endings[end_key]}'
            str_date += datetime.strftime(" of the year %Y.")
        elif idx == 1:
            str_date = datetime.strftime("during %B of the year %Y")

        elif idx == 2:
            str_date = datetime.strftime("during the year %Y")

        else:
            year = int(datetime.strftime("%Y"))
            str_date = "during the %ds"%(year - year%10)

        return str_date

    def set_mode(self, mode):

        self.mode = mode

    def __getitem__(self, main_idx):

        #print('Entered here at idx:', main_idx)

        folder_name = self.idx_to_file[main_idx][0]
        folder_name = "train"
        # img_file_path = os.path.join("input_zip_nyt_final", self.idx_to_file[main_idx][0], self.idx_to_file[main_idx][1])
        img_file_path = os.path.join(self.idx_to_file[main_idx][0], self.idx_to_file[main_idx][1])
        _, preprocess = clip.load(self.clip_model_name, device='cpu')

        img = preprocess(pil_loader(img_file_path)).unsqueeze(0).to('cpu')


        try:
            data_obj = self.root_info[folder_name][main_idx]
            img_url = data_obj.img_url

        except KeyError:
            raise Exception("Some part of the pipeline has clearly failed")

        retrieval_set = set()
        if self.exp_type == 1:
            main_info_geo = data_obj.geo
            correct_string_geo = self.template_geo_label(main_info_geo)
            '''Time experiment'''
            main_info = data_obj.time
            correct_string = correct_string_geo + self.template_time_label(main_info, idx=data_obj.info_base_level)
            print(correct_string)
            retrieval_set.add((correct_string, data_obj.info_base_level))
            all_positives = self.get_hierarchical_time_labels(main_info, from_idx=data_obj.info_base_level)


            base_from_idx = data_obj.info_base_level
            for idx in range(1, len(all_positives)+1):

                retrieval_set.add((self.template_time_label(main_info, base_from_idx+idx, all_positives[idx-1]), base_from_idx+idx))

            retrieval_set_with_pos_only = retrieval_set.copy()
            if self.mode=='collect':
                return {'idx': main_idx,
                        'img': img.to('cpu'),
                        'full_pos_set': list(retrieval_set_with_pos_only)}

            found_correct = False


            for i in range(4):

                for each in list(retrieval_set_with_pos_only):
                    if each[1] == i:
                        correct_string = each[0]
                        correct_elements = [each]
                        retrieval_set = set(correct_elements)
                        found_correct = True
                        break

                if found_correct:
                    break

            if not found_correct:
                raise Exception("something wrong")


            correct_string_tokenized = self.tokenize_fn(correct_string, truncate=True)
            if self.mode == 'train':

                #print('Exiting here at idx', main_idx)
                return {'idx': main_idx, 'img': img.to('cpu'), 'correct_string_tokens': correct_string_tokenized}

            #correct_elements = build_gt_time_correct_set(correct_string, retrieval_set_with_pos_only, self.correctness_levels, self.time_template_options)


            if self.sample:
                sampled_negative_set = self.sample_from_full_negatives(self.num_negatives, correct_string)
                retrieval_set = retrieval_set.union(sampled_negative_set)

            else:
                retrieval_set = self.all_negatives.copy()



        elif self.exp_type == 2:
            '''Location experiment
            
            0: Full string level
            1: only country and continent level
            2: continent level
            '''

            main_info = data_obj.geo
            #correct_string = main_info.full_str


            all_positives = self.get_hierarchical_geo_labels(main_info)
            for idx in range(0, len(all_positives)):

                retrieval_set.add((self.template_geo_label(all_positives[idx][0]), all_positives[idx][1]))

            retrieval_set_with_pos_only = retrieval_set.copy()

            def f_sort(each):
                return each[1]

            found_correct = False


            for i in range(3):

                for each in list(retrieval_set_with_pos_only):
                    if each[1] == i:
                        correct_string = each[0]
                        correct_elements = [each]
                        retrieval_set = set(correct_elements)
                        found_correct = True
                        break

                if found_correct:
                    break

            if not found_correct:
                raise Exception("something wrong")


            correct_string_tokenized = self.tokenize_fn(correct_string, truncate=True)
            if self.mode == 'train':
                #print('Exiting here at idx', main_idx)
                return {'idx': main_idx, 'img': img.to('cpu'), 'correct_string_tokens': correct_string_tokenized}

            if self.mode=='collect':
                return {'idx': main_idx,
                        'img': img.to( 'cpu'),
                        'full_pos_set': list(retrieval_set_with_pos_only)}

            #correct_elements = build_gt_geo_correct_set(correct_string, retrieval_set_with_pos_only, self.correctness_levels)

            if self.sample:
                sampled_negative_set = self.sample_from_full_negatives(self.num_negatives, correct_string)

                retrieval_set = retrieval_set.union(sampled_negative_set)

            else:
                retrieval_set = self.all_negatives.copy()

            #correct_elements = build_gt_geo_correct_set(correct_string, retrieval_set, self.correctness_levels)

        else:
            raise Exception("Exp type should be 1 or 2")


        retrieval_set = list(retrieval_set)

        correct_indices = []
        for sentence in correct_elements:

            for other_idx in range(0, len(retrieval_set)):

                check_sentence = retrieval_set[other_idx]
                if sentence[0] == check_sentence[0]:
                    correct_indices.append(other_idx)
                    break

        assert len(correct_indices) <= len(correct_elements)
        correct_indices = torch.Tensor(correct_indices)
        tokenized_text = torch.stack([self.tokenize_fn(sentence[0], truncate=True) for sentence in retrieval_set]).squeeze()

        if self.mode=='test' or self.mode=='dev':

            return {
                'idx': main_idx,
                'correct_string': correct_string,
                'tokenized_correct': correct_string_tokenized,
                'img': img,
                'gt_data_url': img_url,
                'tokenized_text': tokenized_text,
                'correct_indices': correct_indices,
                'full_retrieval_set': retrieval_set,
                'full_correct_set': correct_elements,
            }


def test_file():


    model_name ="ViT-B/32"
    experiment_type = 2
    template_options = 2

    zero_shot_dataset_geo = GeoTemporalDataset(model_name,
                                     experiment_type,
                                     template_options=(template_options, template_options),
                                     sample=True)



    zero_shot_dataset_geo.set_mode('test')

    zero_shot_dataset_geo.set_mode('train')
    for idx_geo in [4916]:

        zero_shot_dataset_geo.__getitem__(idx_geo)
        
    '''

    model_type = "ViT-B/32"
    experiment_type = 1
    template_options = 2

    zero_shot_dataset_time = GeoTemporalDataset(model_name,
                                                experiment_type,
                                                template_options=(template_options, template_options),
                                                sample=True)


    for idx_time in [12426, 12322, 10510, 3952]:

        zero_shot_dataset_time.__getitem__(idx_time)
    
    '''

def profile_train_and_test_loader_times():


    model_name ="ViT-B/32"
    experiment_type = 2
    template_options = 2

    zero_shot_dataset_geo = GeoTemporalDataset(model_name,
                                     experiment_type,
                                     template_options=(template_options, template_options),
                                     sample=True)


    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        for idx_geo in [234, 2, 3834, 4, 5, 6, 7, 8]:


            zero_shot_dataset_geo.set_mode('test')
            with record_function("test_time_loading"):

                zero_shot_dataset_geo.__getitem__(idx_geo)


            zero_shot_dataset_geo.set_mode('train')
            with record_function("train_time_loading"):

                zero_shot_dataset_geo.__getitem__(idx_geo)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def encode_pickles():


    model_name ="ViT-B/32"
    experiment_type = 2
    template_options = 2

    zero_shot_dataset_geo = GeoTemporalDataset(model_name,
                                     experiment_type,
                                     template_options=(template_options, template_options),
                                     sample=True,
                                    encoding_pickles=True)




def create_all_variations(experiment_type):

    model_name ="ViT-B/32"
    template_options = 2
    sampling = False
    zero_shot_dataset = GeoTemporalDataset(model_name,
                                     experiment_type,
                                     template_options=(template_options, template_options),
                                     sample=False)

    zero_shot_dataset.set_mode('collect')

    zero_shot_loader = torch.utils.data.DataLoader(zero_shot_dataset,
                                                   batch_size=1,
                                                   pin_memory=True,
                                                   shuffle=True,
                                                   collate_fn=custom_collate)



    all_sentences_set = set()
    all_levels_set = set()
    with tqdm(zero_shot_loader) as t:
        for batch_idx, data in enumerate(t):

            for each in data['full_pos_set']:

                all_sentences_set.add((each[0][0], list(each[1].cpu().numpy())[0]))

    final_all_sentences = list(all_sentences_set)
    name = 'time' if experiment_type == 1 else 'geo'
    with open(f'all_negatives_{name}_with_labels.txt', 'w') as f:

        for each_line in final_all_sentences:

            f.write(f'{each_line[0]}<SEP>{each_line[1]}\n')


if __name__ == '__main__':


    encode_pickles()
