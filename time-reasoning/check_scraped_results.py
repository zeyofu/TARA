import os
import json
from scrape_captions import get_slideshow_image_and_caption, get_video_case_and_caption
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
from itertools import repeat
import threading
threadLocal = threading.local()
import concurrent.futures
#Most recent result is latest
JSON_FOLDER = 'archive'
JSON_OUT_FOLDER = 'latest_archive'
FINAL_CAPTION_FOLDER = 'final_caption'
FAILURE_NULL = 'failures_null'
FAILURE_MISSING = 'failures_missing'
COMPLETE_STATS = 'stats'
image_base_url = 'https://static01.nyt.com/'

#exclude = [, '2017-7.jsonl', , '2014-2.jsonl']
#new_exclude = ['2017-8.jsonl', '2018-2.jsonl']
new_exclude = []

def get_driver():

    '''Returns an instance of the online driver'''

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


def check_against_gt_link(f_lines, f_compare_lines, line_no):
    '''

    :param f_lines: A list of groundtruth lines with jsons for original NYT metadata
    :param f_compare_lines: A corresponding list of {img: caption} from latest_archive
    :param line_no: Line_no on which to operate
    :return:
    [isMatch, caption, img_url, web_url]

    If matched, returns as is
    If no match, and only single img: caption pair found, returns the img caption pair
    If found a match, but caption is None, returns 'missing'
    else returns 'null'

    '''

    gt_element = f_lines[line_no]
    article = json.loads(gt_element.strip())

    ref_element = f_compare_lines[line_no]
    given_image_dict = json.loads(ref_element.strip())

    isMatch = False
    found_a_match = False
    failed_key = None
    for key in given_image_dict:

        isMatch = check_if_two_urls_are_shared(key, article['image_url'])
        if isMatch:
            found_a_match = True

        if isMatch and given_image_dict[key] is None:
            failed_key = key

        if isMatch and given_image_dict[key] is not None:

            return isMatch, ' '.join(given_image_dict[key].split('\n')), key, article['web_url']



    if not found_a_match and len(given_image_dict) == 1:

        caption = list(given_image_dict.values())[0]
        new_img_url = list(given_image_dict.keys())[0]

        return True, caption, new_img_url, article['web_url']


    if found_a_match:
        return False, 'null', None, article['web_url']



    else:
        return False, 'missing', failed_key, article['web_url']

def check_percent(file_name, add_caption=False):
    '''

    :param file_name: Specific file name
    :param add_caption: whether or not to add caption information to final caption folder
    :return:

    Returns (match // total_length) for file_name (acquired file name) and associated failure cases
    '''

    driver = get_driver()


    total_length = sum(1 for line in open(os.path.join(JSON_FOLDER, file_name)))
    total_created_length = sum(1 for line in open(os.path.join(JSON_OUT_FOLDER, file_name)))
    assert total_length == total_created_length
    if add_caption:
        f_out = open(os.path.join(FINAL_CAPTION_FOLDER, file_name), 'a')
    match = 0
    failure_cases = {'null': [], 'missing': []}
    for line_no in tqdm(range(0, total_length)):

        f_input = open(f'archive/{file_name}', 'r')
        f_lines = [line.strip('\n') for line in f_input]

        f_compare = open(f'latest_archive/{file_name}', 'r')
        f_compare_lines = [line.strip('\n') for line in f_compare]

        match_right, caption, img_url, web_url = check_against_gt_link(f_lines, f_compare_lines, line_no)

        if add_caption:
            if not match_right and 'slideshow' in web_url:

                img_dict = get_slideshow_image_and_caption(web_url, driver)
                if len(img_dict) >= 1:
                    caption = list(img_dict.values())[0]
                    match_right = True


            if not match_right and 'video' in web_url:
                img_dict = get_video_case_and_caption(web_url, driver)
                if len(img_dict) >= 1:
                    caption = list(img_dict.values())[0]
                    match_right = True


        if match_right:

            if add_caption:
                if web_url is None:
                    web_url = 'N/A'
                if img_url is None:
                    img_url = 'N/A'
                if caption is None:
                    caption = 'N/A'

                caption = ' '.join(caption.split('\n'))
                assert '\n' not in img_url
                assert '\n' not in web_url
                f_out.write(caption + '\n' + img_url + '\n' + web_url + '\n')

            match += 1

        else:

            failure_cases[caption].append((f_lines[line_no], f_compare_lines[line_no]))
            if add_caption:
                f_out.write('\n' + '\n' + '\n')


    return match / total_length, failure_cases

def check_all_percents():
    '''
    Checks percents for all the files and returns results in a dictionary

    :return:
    make_percent_dict: from file name to success rate statistics
    '''

    make_percent_dict = {}
    for each in os.listdir(JSON_FOLDER):

        if each not in new_exclude:
            print(f'Checking for {each}....')
            percent, failures = check_percent(each)
            make_percent_dict[each] = percent
            print(f'{each} retrieved a success rate of {make_percent_dict[each]}')


    return make_percent_dict


def write_individual_captions(each_file, write_failures):

    '''

    :param each_file: FILE NAME FROM archive
    :param write_failures: Writing failures = NOT adding captions
    :return:
    '''


    print(f'Checking for {each_file}....')
    if not write_failures:
        percent, failure_cases = check_percent(each_file, add_caption=True)

    else:
        percent, failure_cases = check_percent(each_file)

    if write_failures:
        failure_null_current = failure_cases['null']
        failure_missing_current = failure_cases['missing']



        with open(os.path.join(FAILURE_MISSING, each_file), 'w') as f:

            for each in failure_missing_current:

                f.write(each[0] + '\n')
                f.write(each[1] + '\n')



        with open(os.path.join(FAILURE_NULL, each_file), 'w') as f_2:

            for each in failure_null_current:
                f_2.write(each[0] + '\n')
                f_2.write(each[1] + '\n')



def write_captions(write_failures=False):

    '''
    Calls the individual function for the whole folder
    '''


    final_exclude = []
    for each_new_file in os.listdir(FINAL_CAPTION_FOLDER):
        final_exclude.append(each_new_file)


    final_exclude += new_exclude
    final_run_list = []
    for each_file in os.listdir(JSON_FOLDER):

        if each_file not in final_exclude and 'icloud' not in each_file:

            final_run_list.append(each_file)

    final_run_list = final_run_list[:40]


    print(final_run_list)
    import time
    time.sleep(3)

    #write_individual_captions(final_run_list[0], False)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(write_individual_captions, final_run_list, repeat(write_failures))




def check_caption_percents():

    '''
    For each file in the final caption folder, it checks the percent of retrieved captions (as fully required)
    '''


    for each_caption_file in os.listdir(FINAL_CAPTION_FOLDER):
        caption_path = os.path.join(FINAL_CAPTION_FOLDER, each_caption_file)
        missed = 0
        total = 0
        with open(caption_path, 'r') as f:
            for line in f:
                total += 1
                if line == '\n':

                    missed += 1


        print(f'Accuracy of {each_caption_file} was {(total-missed)/total}')


def compare_archive_and_final_caption():

    '''
    It compares the final caption folder and the original archive folder and asserts it has the same number of lines

    :return:
    '''

    for each_file in os.listdir(FINAL_CAPTION_FOLDER):

        no_captions = sum(1 for line in open(os.path.join(FINAL_CAPTION_FOLDER, each_file)))
        no_gt = sum(1 for line in open(os.path.join(JSON_FOLDER, each_file)))

        try:

            assert no_captions / 3 == no_gt
            #cprint(f"Got the calc on {each_file} because {no_captions} is the same as {no_gt}")

        except AssertionError:
            print(f"Missed the calc on {each_file} because {no_captions / 3} is not the same as {no_gt}")


def check_if_two_urls_are_shared(str_l, str_r):

    '''

    :param str_l: String 1
    :param str_r:  string 2
    :return: Check if the strings share a common subset
    '''

    l = str_l.split('/')
    r = str_r.split('/')


    short = l if len(l) < len(r) else r
    long = l if len(l) >= len(r) else r


    root = short[:-1]
    found = long[:len(root)]


    for idx in range(len(root)):

        if root[idx] != found[idx]:
            return False

    return True

def stop_at_odd(file_name):
    '''

    :param file_name: Given a filename, this function helps debug mismatches/issues with the file
    :return:
    '''

    caption_path = os.path.join(FINAL_CAPTION_FOLDER, file_name)
    f_caption = open(caption_path, 'r')
    all_caption_lines = [line.strip('\n') for line in f_caption]

    for i in range(1, len(all_caption_lines)):

        if i%3 == 2 or i%3 == 0:

            try:
                assert 'https' in all_caption_lines[i-1] or not all_caption_lines[i-1] or all_caption_lines[i-1] == 'N/A'
            except:
                print(i-1)
                print(all_caption_lines[i-1])
                raise Exception


def convert_to_jsonl():

    '''
    Returns the final json style data for the files

    :return:
    '''

    JSON_FINAL_CAPTION_FOLDER = 'jsonl_captions_final'
    if not os.path.exists(JSON_FINAL_CAPTION_FOLDER):
        os.makedirs(JSON_FINAL_CAPTION_FOLDER)

    for each_file in os.listdir(FINAL_CAPTION_FOLDER):



        caption_path = os.path.join(FINAL_CAPTION_FOLDER, each_file)
        write_path = os.path.join(JSON_FINAL_CAPTION_FOLDER, each_file)
        root_path = os.path.join(JSON_FOLDER, each_file)

        if os.path.exists(write_path):

            continue

        f_root = open(root_path, 'r')
        all_root_lines = [line.strip('\n') for line in f_root]

        f_caption = open(caption_path, 'r')
        all_caption_lines = [line.strip('\n') for line in f_caption]

        f_out = open(write_path, 'a')

        try:
            assert len(all_caption_lines) == len(all_root_lines) * 3

        except AssertionError:
            print(f'Mismatch found for {each_file} because {len(all_caption_lines)} is not 3 times {len(all_root_lines)}')

        print(f'Evaluating {each_file}')
        idx = 1
        for each_root_line in tqdm(all_root_lines):


            caption_idx = (3 * idx - 2) - 1
            img_url_idx, web_url_idx = caption_idx + 1, caption_idx + 2

            line_dict = {}
            root_article = json.loads(each_root_line.strip())
            line_dict['caption'] = all_caption_lines[caption_idx]
            line_dict['web_url'] = all_caption_lines[web_url_idx]

            if all_caption_lines[img_url_idx] and all_caption_lines[img_url_idx] != 'N/A':
                line_dict['img_url'] = all_caption_lines[img_url_idx]

            else:
                line_dict['img_url'] = root_article['image_url']

            write_string = json.dumps(line_dict) + '\n'
            f_out.write(write_string)
            idx += 1


        f_root.close()
        f_caption.close()
        f_out.close()


if __name__ == '__main__':
    #url_l = "https://static01.nyt.com/images/2020/04/29/arts/29museum-closure4/29museum-closure4-superJumbo.jpg"
    #url_r = "https://static01.nyt.com/images/2020/04/29/arts/29museum-closure4/29museum-closure4-articleLarge.jpg?quality=75&auto=webp&disable=upscale"
    #print(check_all_percents())

    #check_all_percents()
    #compare_archive_and_final_caption()
    convert_to_jsonl()

    #stop_at_odd('2010-7.jsonl')
    exit(0)

    '''
    if not os.path.exists(FINAL_CAPTION_FOLDER):
        os.makedirs(FINAL_CAPTION_FOLDER)

    if not os.path.exists(FAILURE_NULL):
        os.makedirs(FAILURE_NULL)

    if not os.path.exists(FAILURE_MISSING):
        os.makedirs(FAILURE_MISSING)
    '''

    #write_captions()