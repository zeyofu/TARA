import os
from multiprocessing.pool import ThreadPool, Pool
from tqdm import tqdm
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import threading
threadLocal = threading.local()
from scrape_captions import scrape_image_and_caption
import json
import concurrent.futures
import argparse

#Most recent result is latest
JSON_FOLDER = 'archive'
JSON_OUT_FOLDER = 'latest_archive'
JSON_CAPTION_FOLDER = 'latest_caption'
COMPLETE_STATS = 'stats'
image_base_url = 'https://static01.nyt.com/'

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--i_s', required=True, type=int, help='Start index of given input to be considered')
    #parser.add_argument('--i_e', required=True, type=int, help='End index + 1 of given input to be considered')
    parser.add_argument('--c', default=0, type=int, help='Compare against which folder')
    parser.add_argument('--oc', default=1, type=int, help='Only compare and not scrape')
    parser.add_argument('--p', default=0, type=int, help='Run preprocess')
    args = parser.parse_args()
    return args

def make_table(with_caption=False):


    downstream_folder = JSON_OUT_FOLDER if not with_caption else JSON_CAPTION_FOLDER

    if not os.path.exists(COMPLETE_STATS):
        os.makedirs(COMPLETE_STATS)


    gt_line_count = {}
    for each_gt_file in os.listdir(JSON_FOLDER):
        if 'icloud' not in each_gt_file:
            gt_line_count[each_gt_file] = sum(1 for line in open(os.path.join(JSON_FOLDER, each_gt_file)))


    complete_line_count = {}
    for each_complete_file in os.listdir(downstream_folder):
        if 'icloud' not in each_complete_file:
            complete_line_count[each_complete_file] = sum(1 for line in open(os.path.join(downstream_folder, each_complete_file)))

    '''

    for key in gt_line_count:
        print(f'{key} has {gt_line_count[key]} actual lines and {complete_line_count[key]} have been processed')
    '''

    with open(os.path.join(COMPLETE_STATS, 'original_stats.json'), 'w') as f:
        json.dump(gt_line_count, f)

    with open(os.path.join(COMPLETE_STATS, 'current_stats.json'), 'w') as g:
        json.dump(complete_line_count, g)


def compare_status(file_gt, file_ref):

    gt_line_count = json.load(open(file_gt))
    complete_line_count = json.load(open(file_ref))


    for key in gt_line_count:
        print(f'{key} has {gt_line_count[key]} actual lines and {complete_line_count[key]} have been processed')

def given_compare_status(file_gt, file_ref, given_keys):

    gt_line_count = json.load(open(file_gt))
    complete_line_count = json.load(open(file_ref))


    for key in given_keys:
        print(f'{key} has {gt_line_count[key]} actual lines and {complete_line_count[key]} have been processed')



def get_those_with_zero_process_and_incomplete_process(file_gt, file_ref):

    gt_line_count = json.load(open(file_gt))
    complete_line_count = json.load(open(file_ref))

    zero_process = {}
    incomplete_process = {}
    for key in complete_line_count:
        if complete_line_count[key] == 0:
            zero_process[key] = 1

        if complete_line_count[key] < gt_line_count[key] and complete_line_count[key]:
            incomplete_process[key] = complete_line_count[key] + 1

    return zero_process, incomplete_process


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(input_path):
    print(f'Reading {input_path}...')
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(len(data))
    return data

def get_driver():

    driver = getattr(threadLocal, 'driver', None)
    if driver is None:
        options = webdriver.ChromeOptions()
        options.headless = True
        options.add_argument("start-maximized")
        options.add_argument("--disable_gpu")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        setattr(threadLocal, 'driver', driver)

    return driver

def list_all_json_files(root_folder):

    final_list = []
    for each in os.listdir(root_folder):
        if '.jsonl' in each:
            final_list.append(each)

    return final_list

def get_img_url_caption_dict(url):

     driver = get_driver()
     img_dict = scrape_image_and_caption(url)
     return img_dict



def check_if_two_urls_are_shared(str_l, str_r):

    l = str_l.split('/')
    r = str_r.split('/')

    short = l if len(l) < len(r) else r
    long = l if len(l) >= len(r) else l

    root = short[:-1]
    found = long[:len(root)]

    for idx in range(len(root)):

        if root[idx] != found[idx]:
            return False

    return True

def preprocess(given_input):
    '''Runs through the file and the start pos and creates initial list of captions for it

    Latest archive contains
    '''

    input_path, start_pos = given_input
    start_pos -= 1
    input_path = os.path.join(JSON_FOLDER, input_path)

    output_path = input_path.replace(JSON_FOLDER, JSON_OUT_FOLDER)
    output_caption_path = input_path.replace(JSON_FOLDER, JSON_CAPTION_FOLDER)


    driver = get_driver()

    f = open(input_path, 'r')
    print(f'Reading {input_path} ...')
    f_lines = [line.strip('\n') for line in f]
    total_lines = len(f_lines)


    f_lines = f_lines[start_pos:]
    f_out = open(output_path, 'a')
    f_cout = open(output_caption_path, 'a')

    write_string = ''
    caption_string = ''
    line_no = start_pos

    try:
        for line in f_lines:
            line_no += 1

            article = json.loads(line.strip())
            root_url = article['web_url']
            img_url_dict = scrape_image_and_caption(root_url, driver=driver)
            if len(img_url_dict) > 1 and list(img_url_dict.values())[0] is not None:

                real_caption = list(img_url_dict.values())[0]

            else:

                real_caption = ''


            real_caption = ' '.join(real_caption.split('\n'))

            write_string += json.dumps(img_url_dict) + '\n'
            caption_string += real_caption + '\n'


            if line_no%10 == 0 or line_no == total_lines:

                f_out.write(write_string)
                f_cout.write(caption_string)
                print(f'Finished {line_no} on {input_path}')
                write_string = ''
                caption_string = ''

    except Exception as e:
        print(f'Failed at {line_no} in {input_path} with exception {e}')

    #driver.close()

if __name__ == '__main__':
    import time

    args = get_args()
    import shutil
    shutil.rmtree(COMPLETE_STATS)


    make_table(args.c)
    zero_process, incomplete_process = get_those_with_zero_process_and_incomplete_process(
        os.path.join(COMPLETE_STATS, 'original_stats.json'), os.path.join(COMPLETE_STATS, 'current_stats.json'))

    print(zero_process)
    print(incomplete_process)


    given_input = [(key, zero_process[key]) for key in zero_process] + [(key, incomplete_process[key]) for key in incomplete_process]
    given_input = sorted(given_input)

    #print(given_compare_status(os.path.join(COMPLETE_STATS, 'original_stats.json'), os.path.join(COMPLETE_STATS, 'current_stats.json'),
    #                           [each[0] for each in given_input]))

    print(given_input)
    time.sleep(3)

    if not args.oc:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(preprocess, given_input)

    if args.p:
        for each in given_input:
            preprocess(each)


