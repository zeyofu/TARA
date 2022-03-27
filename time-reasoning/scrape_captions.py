from pynytimes import NYTAPI
import datetime
import requests
import os
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import ElementClickInterceptedException, ElementNotInteractableException, TimeoutException, NoSuchElementException, WebDriverException
import urllib.request
import time
from selenium.webdriver.common.keys import Keys


nyt = NYTAPI("d9mzK5j2oWh08EZP4AYO7jmFcVLN8xYV", parse_dates=True)
JSON_FOLDER = 'archive'
JSON_OUT_FOLDER = 'latest_archive'
JSON_CAPTION_FOLDER = 'latest_caption'
COMPLETE_STATS = 'stats'
image_base_url = 'https://static01.nyt.com/'





def get_driver():

    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_argument("start-maximized")
    options.add_argument("--disable_gpu")
    #options.add_argument("--user-data-dir=/Users/ishaanchandratreya/Library/Application Support/Google/Chrome")
    #options.add_argument("--profile-directory=Default")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    return driver

def pretend_browser(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'lxml')

    for item in soup.select('.e13ogyst0'):
        try:
            print('------------------------')
            print(item.get_text())


        except:
            print('Exception occured here')


def image_caption_scrape(url):

    metadata = nyt.article_metadata(url)
    for each in metadata[0]['multimedia']:

        print(each['caption'])
        print(each['url'])


def get_video_case_and_caption(url, driver=None):

    if driver is None:
        driver = get_driver()

    try:
        driver.set_page_load_timeout(20)
        driver.get(url)

    except TimeoutException as ex:
        print('timeout')
        return {}

    try:
        html_drive = driver.find_elements_by_css_selector("html.nytapp-vi-video")
        print('Found video on this link')

    except:
        print('No video here- moving on...')
        return {}

    try:
        possible_captions = driver.find_elements_by_css_selector('h2.css-13qem32')
        found_caption = possible_captions[0].text

    except:
        print('Found video but could not find caption')
        return {}



    return {
        'caption': found_caption
    }

def get_slideshow_image_and_caption(url, driver=None):

    try:
        assert 'slideshow' in url
    except:
        print('No slideshow in URL')
        return {}

    if driver is None:
        driver = get_driver()

    try:
        #driver.set_page_load_timeout(20)
        driver.get(url)

    except TimeoutException as ex:
        print('timeout')
        return {}

    img_urls = {}


    actual_images = driver.find_elements_by_tag_name('img')
    for actual_image in actual_images:

        try:
            src_url = actual_image.get_attribute('src')


            if src_url and 'https' in src_url:
                capt = driver.find_elements_by_css_selector('div.css-1vbanrr')

                if capt:
                    img_urls[src_url] = capt[0].text
                else:
                    img_urls[src_url] = None

        except:
            print('Skipping')
            continue

    return img_urls

def scrape_image_and_caption_with_scroll(url, driver=None):
    img_urls = dict()

    if driver is None:
        driver = get_driver()

    try:
        driver.set_page_load_timeout(20)
        driver.get(url)

    except TimeoutException:

        print('timeout')
        return img_urls


    actual_images = []


    SCROLL_PAUSE_TIME = 2
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:

        driver.execute_script("window.scrollTo(0, window.scrollY + 200);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        actual_images += driver.find_elements_by_tag_name('img')
        if new_height == last_height:
            break
        last_height = new_height


    for actual_image in actual_images:

        try:
            src_url = actual_image.get_attribute('src')

            if src_url and 'https' in src_url:
                capt = actual_image.get_attribute('alt')
                if capt:
                    img_urls[src_url] = capt
                else:
                    img_urls[src_url] = None
        except:
            print('Current image returned error')
            continue

    return img_urls

def scrape_image_and_caption(url, driver=None):
    img_urls = dict()
    if driver is None:
        driver = get_driver()


    try:
        driver.set_page_load_timeout(40)
        driver.get(url)

    except TimeoutException:

        print(url)
        print('timeout')
        return img_urls


    actual_images = driver.find_elements_by_tag_name('img')
    for actual_image in actual_images:

        try:
            src_url = actual_image.get_attribute('src')

            if src_url and 'https' in src_url:
                capt = actual_image.get_attribute('alt')
                if capt:
                    img_urls[src_url] = capt
                else:
                    img_urls[src_url] = None

        except:
            print('Current image returned error')
            continue

    return img_urls


class Comment:

    def __init__(self, user, loc, date, comment):

        self.date = date
        self.user = user
        self.loc = loc
        self.comment = comment


def process_comment(comment):

    return int(comment.get_attribute('aria-describedby').split('-')[-1])

def scrape_whats_going_on_datapoint(url):


    driver = get_driver()
    driver.get(url)

    driver_comments = get_driver()
    driver_comments.get(f'{url}#commentsContainer')

    return_dict = {}
    try:

        actual_images = driver.find_elements_by_css_selector('img')
        actual_caption = driver.find_elements_by_tag_name('blockquote')[0].find_elements_by_tag_name('p')[0].text
        hlinks = driver.find_elements_by_css_selector('a.css-1g7m0tk')

        for hlink in hlinks:


            if hlink.find_element_by_xpath("..").text == "The photographer is ":
                return_dict['photographer'] = hlink.text
                return_dict['photographer_website'] = hlink.get_attribute('href')

                continue

            if "This week’s image comes from the" in hlink.find_element_by_xpath("..").text:
                return_dict['gtruth_article_title'] = hlink.text
                return_dict['gtruth_article_url'] = hlink.get_attribute('href')
                #parent = hlink.find_element_by_xpath("..").find_element_by_xpath("..")

                continue


        if len(actual_images) >= 1:
            actual_image = actual_images[0]
            if actual_image.get_attribute('src') and 'https' in actual_image.get_attribute('src'):
                return_dict['image'] = actual_image.get_attribute('src')
                return_dict['caption'] = actual_caption



        ul_list = driver_comments.find_elements_by_css_selector('li.selected')[0].find_element_by_xpath('..')
        for each in ul_list.find_elements_by_tag_name('li'):
            if each.get_attribute('aria-selected') == "false":
                each.click()

        time.sleep(2)
        for i in range(5):
            driver_comments.find_element_by_tag_name('body').send_keys(Keys.END)
            time.sleep(2)

        #driver_comments.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        actual_comments = driver_comments.find_elements_by_css_selector('div.css-aa7djq')

        all_comments = []

        for comment in actual_comments:

            comment_file = comment.find_elements_by_tag_name('p')[0]
            div_user = comment.find_elements_by_css_selector('div.css-1vg6q84')[0]
            comment_user = div_user.find_elements_by_tag_name('span')[0].text

            span_loc_and_date = comment.find_elements_by_css_selector('span.css-1ht9dc3')[0]
            loc_and_date = span_loc_and_date.find_elements_by_tag_name('span')


            comment_date = ''
            comment_loc = ''
            for each in loc_and_date:
                if each.get_attribute('data-testid') == 'todays-date':
                    comment_date = each.get_attribute('data-testid')

                else:
                    comment_loc = each.text

            if "comment-content" in comment_file.get_attribute('id'):

                all_comments.append(Comment(comment_user, comment_loc, comment_date, comment_file.text))


        return_dict['all_comments'] = all_comments

        return_dict['similarity'] = scrape_image_and_caption(return_dict['gtruth_article_url'])

        if 'photographer_website' in return_dict:
            return_dict['photographer_sim'] = scrape_image_and_caption(return_dict['photographer_website'])


    except ElementClickInterceptedException or ElementNotInteractableException as err:
        print(err)

    return return_dict


def login(url='https://myaccount.nytimes.com/auth/login?response_type=cookie&client_id=vi', driver=None):


    if driver is None:
        driver = get_driver()


    try:
        driver.set_page_load_timeout(60)
        driver.get(url)

    except TimeoutException:

        print('timeout')

    el = driver.find_element_by_name("email")
    for letter in "i s h a a n . p . c @ g m a i l . c o m".split():
        el.send_keys(str(letter))
        time.sleep(2)
    driver.find_element_by_css_selector("button.css-nrhj9s-buttonBox-buttonBox-primaryButton-primaryButton-Button").click()



    new_el = driver.find_element_by_name("email")
    driver.find_element_by_tag_name("button").click()
    time.sleep(2)
    el = driver.find_element_by_name("password")

    for letter in "t i m e - r e a s o n i n g".split():
        el.send_keys(str(letter))
        time.sleep(2)
    time.sleep(2)
    driver.find_element_by_css_selector("button.css-nrhj9s-buttonBox-buttonBox-primaryButton-primaryButton-Button").click()

def whats_going_on_test():
    example_urls = [
    'https://www.nytimes.com/live/2021/07/07/world/jovenel-moise-assassinated-killed',
    'https://www.nytimes.com/2021/07/07/us/search-ends-survivors-florida-condo-collapse.html?action=click&module=Top%20Stories&pgtype=Homepage'
    ]

    example_picture_urls = [
        'https://www.nytimes.com/2021/05/23/learning/whats-going-on-in-this-picture-may-24-2021.html',
    ]

    print(scrape_whats_going_on_datapoint(example_picture_urls[0]))


def test_slideshow():

    slide_url = "https://www.nytimes.com/slideshow/2010/10/29/us/20101030-PLANE.html"
    print(get_slideshow_image_and_caption(slide_url))


def test_video_case():

    video_url = "https://www.nytimes.com/video/us/100000006240761/anchorage-alaska-earthquake.html"
    print(get_video_case_and_caption(video_url))

def test_scrollable_img_case():

    scroll_url = "https://www.nytimes.com/2018/11/30/us/amber-guyger-botham-jean-indicted.html"
    print(scrape_image_and_caption_with_scroll(scroll_url))

if __name__ == '__main__':


    login()
    time.sleep(20)




    #for url in example_urls:
    #    image_caption_scrape(url)
