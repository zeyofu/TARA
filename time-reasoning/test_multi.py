from multiprocessing.pool import ThreadPool, Pool
from selenium import webdriver
from urllib.parse import urljoin
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import threading
from scrape_captions import scrape_image_and_caption

threadLocal = threading.local()

def get_driver():


    driver = getattr(threadLocal, 'driver', None)
    if driver is None:
        options = webdriver.ChromeOptions()
        options.headless = True
        options.add_argument("start-maximized")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        setattr(threadLocal, 'driver', driver)

    return driver

def parse_url(url):

    driver = get_driver()

    img_dict = scrape_image_and_caption(url, driver)

    return img_dict


if __name__== '__main__':


    urls = ["https://www.nytimes.com/2010/01/03/nyregion/03polaroid.html",
            "https://www.nytimes.com/2020/12/31/opinion/2021-economy-recovery.html",
            "https://www.nytimes.com/2020/12/31/us/resolving-to-live-a-lot-better-than-in-2020.html",
            "https://www.nytimes.com/2020/12/31/world/the-us-reaches-20-million-cases.html"]


    #for each in urls:
    #    parse_url(each)
    out = ThreadPool(4).map(parse_url, urls)
