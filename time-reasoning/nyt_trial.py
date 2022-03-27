from pynytimes import NYTAPI
import datetime
import requests
import os


nyt = NYTAPI("d9mzK5j2oWh08EZP4AYO7jmFcVLN8xYV", parse_dates=True)



def plot_image():
    pass


def write_image(url, write_path):

    response = requests.get(url)
    file = open(write_path, 'ab+')
    file.write(response.content)
    file.close()


def store_images_for_article(article, write_path):

    media = article['multimedia']
    primary_key = '_'.join(article['abstract'].split())
    article_photo_path = os.path.join(write_path, primary_key)
    if not os.path.exists(article_photo_path):
        os.makedirs(article_photo_path)


    images = filter(lambda el: el['type'] == 'image', media)
    image_count = 0

    for image_dict in images:

        url = 'https://nytimes.com/' + image_dict['url']
        extension = url.split('.')[-1]
        image_path = os.path.join(article_photo_path, f'{image_count}.{extension}')
        write_image(url, image_path)
        image_count += 1
        break

def get_article_metadata(url):

    metadata = nyt.article_metadata(
        url=url
    )

    return metadata

def search_articles(query, date_range, max_articles=30, data_path='data'):


    write_path = os.path.join(os.getcwd(), data_path, '_'.join(query.split()))
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    articles = nyt.article_search(
        query=query,
        results=max_articles,
        dates={
            "begin": date_range[0],
            "end": date_range[1],
        },
        options={
        "sort": "relevance",
        "sources": ["New York Times", "AP"]
        }
    )
    print(len(articles))
    final_urls = []
    for article in articles:

        store_images_for_article(article, write_path=write_path)
        final_urls.append(article['web_url'])
        print(article['web_url'])


    return final_urls


if __name__ == '__main__':

    #final_urls = search_articles("What’s Going On in This Picture?", (datetime.datetime(2015, 10, 3), datetime.datetime(2021, 7, 21)))
    url = "https://www.nytimes.com/2021/05/31/arts/memorial-day-new-york-reopening.html"
    out = get_article_metadata(url)
    print(out)