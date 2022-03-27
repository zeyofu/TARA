import json
import os
import random
import pickle
import nltk
nltk.download('punkt')
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils import data
from tqdm import tqdm
import clip
from datapoints import Location, TimePoint
from nyt_dataset import pil_loader, get_name_from_url, get_name_from_url_input_zip


class GeoTemporalImage(data.Dataset):

    #TODO Class invalid as loads from old dataset

    def __init__(self, model_name, exp_type, data_dir='input_zip_nyt_dataset', template_options=(0, 0), ratio=10,
                correctness_level=0, mode='test'):

        random.seed(42)
        self.mode=mode
        self.exp_type = exp_type #1 is time experiment, 2 is geo experiment
        self.data_path = os.path.join(os.getcwd(), data_dir)
        self.labels_path = os.path.join(self.data_path, 'details')
        if not os.path.exists(self.data_path):
            raise Exception("This dataset could not be found")

        self.time_template_options = template_options[0]
        self.geo_template_options =  template_options[1]

        #self.url_to_idx, self.idx_to_url = self.fix_index(os.path.join(self.data_path, 'index.txt'))
        self.root_info, self.timeline, self.idx_to_file, self.idx_linking, self.url_to_idx = self._load_root_info_input_zip(self.labels_path)

        self.rng = np.random.RandomState(42)
        self.clip_model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _, self.preprocess = clip.load(self.clip_model_name, device=self.device)
        self.pos_to_neg_ratio = ratio
        self.correctness_levels = correctness_level

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

            mod_i = lambda i: 100 if not i else 1
            fn = lambda str: [int(val)*mod_i(i) for i, val in enumerate(str.strip('.jsonl').split('-'))]
            root_info = {} #given current folder name, contains a mapping from idx to the created data point from that image
            idx_linking = {} #given idx in the dataset, returns which file and what line no it came from
            idx_to_file = {} #given idx in the dataset, returns the folder and the file name associated with it
            url_to_idx = {} #given image url returns the idx associated with that url
            timeline = []
            idx_added = 0
            paths = [os.path.join(path, each_path) for each_path in ['train.jsonl', 'dev.jsonl', 'test.jsonl']]

            for each_file in paths:

                current_folder_name = os.path.splitext(os.path.basename(each_file))[0]
                root_info[current_folder_name] = {}

                for line_no, line in enumerate(open(os.path.join(path, each_file), 'r')):

                    cur_dict = json.loads(line.strip())
                    url_to_idx[cur_dict['image_url']] = idx_added
                    idx_to_file[idx_added] = (current_folder_name, get_name_from_url_input_zip(cur_dict['image_url']))
                    data_point = TimePoint(cur_dict, idx_added, current_folder_name)
                    root_info[current_folder_name].update({idx_added: data_point})
                    idx_linking[idx_added] = (each_file, line_no)
                    idx_added += 1
                    timeline.append(data_point)

            timeline = sorted(timeline)

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


    def __getitem__(self, main_idx):


        folder_name = self.idx_to_file[main_idx][0]
        img_file_path = os.path.join(self.data_path, self.idx_to_file[main_idx][0], self.idx_to_file[main_idx][1])

        img = self.preprocess(pil_loader(img_file_path)).unsqueeze(0).to(self.device)

        return {'img': img.to('cpu'), 'root_img_path': img_file_path}
