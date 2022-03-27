import requests
import os
import json
from tqdm import tqdm


INPUT_URL_PATH = "/shared/xingyu/projects/visual/data/nyt_preprocessed/input"
OUTPUT_IMG_PATH = os.path.join(os.getcwd(), 'input_zip_nyt_final')


def write_img(url, write_path):
    '''

    :param url: Url of the image to be downloaded
    :param write_path: Writing location on the local filesystem
    :return:
    '''

    response = requests.get(url)

    with open(write_path, 'ab+') as handle:
        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)


def find_dups(jsonl_file):

    jsonl_path = os.path.join(INPUT_URL_PATH, jsonl_file)
    dups_dict = {}
    fo_line = {}
    dup_lines = []

    with open(jsonl_path, 'r') as f:

        for line_no, line in enumerate(tqdm(f)):

            line_dict = json.loads(line)
            if line_dict['image_url'] in dups_dict:
                dups_dict[line_dict['image_url']] += 1
                dup_lines.append((fo_line[line_dict['image_url']], line_no))

            else:
                dups_dict[line_dict['image_url']] = 1
                fo_line[line_dict['image_url']] = line_no

    return dups_dict, dup_lines


def process_dups_count(dups):


    sig_dups = [key for key, val in dups.items() if val != 1]
    return len(sig_dups), sig_dups

def find_dups_all():

    mod_i = lambda i: 100 if not i else 1555
    fn = lambda str: [int(val) * mod_i(i) for i, val in enumerate(str.strip('.jsonl').split('-'))]
    count_dups = {}
    sig_dups_dict = {}
    list = [file for file in os.listdir(INPUT_URL_PATH) if 'jsonl' in file]
    for each_jsonl_file in sorted(list, key=fn):
        dups, dup_lines = find_dups(each_jsonl_file)
        num_dups, sig_dups = process_dups_count(dups)
        if  num_dups >=1:
            count_dups[each_jsonl_file] = num_dups
            sig_dups_dict[each_jsonl_file] = (sig_dups, dup_lines)

    return count_dups, sig_dups_dict

def get_year_lengths():
    list = [file for file in os.listdir(INPUT_URL_PATH) if 'jsonl' in file]
    nums = [0, 0, 0, 0]

    for jsonl_file in list:

        jsonl_path = os.path.join(INPUT_URL_PATH, jsonl_file)

        with open(jsonl_path, 'r') as f:

            for line_no, line in enumerate(tqdm(f)):

                line_dict = json.loads(line)
                time_p = line_dict['time']

                try:
                    fin_len = len([int(time) for time in time_p.split('-')])
                except ValueError:
                    fin_len = 0

                nums[fin_len] += 1

    return nums


def download_from_input_zip(jsonl_path):

    if not os.path.exists(OUTPUT_IMG_PATH):
        os.makedirs(OUTPUT_IMG_PATH)

    paths = [os.path.join(jsonl_path, each_path) for each_path in ['train.jsonl', 'dev.jsonl', 'test.jsonl']]


    for each_jsonl_path in paths:

        with open(each_jsonl_path, 'r') as f:
            for line_no, line in enumerate(tqdm(f)):
                line_dict = json.loads(line)

                image_name = os.path.splitext(line_dict['image_url'].replace('/', '_'))[0] + '.png'
                write_path = os.path.join(OUTPUT_IMG_PATH, os.path.splitext(os.path.basename(each_jsonl_path))[0])
                if not os.path.exists(write_path):
                    os.makedirs(write_path)

                write_path = os.path.join(write_path, image_name)

                write_img(line_dict['image_url'], write_path)


def download_for_file(jsonl_file):

    '''
    :param jsonl_file: File indicating year-month pair
    :return:
    '''



    jsonl_write_path = os.path.join(OUTPUT_IMG_PATH, jsonl_file.split('.')[0].replace('-', '_'))


    if not os.path.exists(jsonl_write_path):
        os.makedirs(jsonl_write_path)



    jsonl_path = os.path.join(INPUT_URL_PATH, jsonl_file)
    print(f'Downloading for {jsonl_file}...')
    total_dups = 0
    with open(jsonl_path, 'r') as f:

        #debug_dict = {}

        for line_no, line in enumerate(tqdm(f)):

            line_dict = json.loads(line)

            image_name = '_'.join(line_dict['image_url'].split('/')[-7:]).split("?")[0]
            if '.jpg' in image_name:
                image_name = image_name.strip('.jpg')
            if '.png' in image_name:
                image_name = image_name.strip('.png')



            final_write_path = os.path.join(jsonl_write_path, f'{image_name}.png')
            if not os.path.exists(final_write_path):

                #debug_dict[image_name] = (line_dict, line_no)
                write_img(line_dict['image_url'], final_write_path)

            else:
                #print(debug_dict[image_name])
                total_dups += 1
                continue


    print(f'{jsonl_file} has {total_dups} duplicates.')

def download_dataset():



    mod_i = lambda i: 100 if not i else 1555
    fn = lambda str: [int(val) * mod_i(i) for i, val in enumerate(str.strip('.jsonl').split('-'))]

    list = [file for file in os.listdir(INPUT_URL_PATH) if 'jsonl' in file]
    for each_jsonl_file in sorted(list, key=fn):


        download_for_file(each_jsonl_file)


def fix_index():

    orig = {}
    index_path = os.path.join(OUTPUT_IMG_PATH, 'index.txt')
    line = ''
    for each_line in open(index_path, 'r'):
        line = each_line
        break

    dict_list = '}\n'.join(line.split('}')).split('\n')[:-1]
    out_list = [json.loads(each) for each in dict_list]
    for each in out_list:
        orig.update(each)

    inv_orig = {v: k for k, v in orig.items()}
    print(inv_orig)

if __name__=='__main__':

    download_from_input_zip(INPUT_URL_PATH)

