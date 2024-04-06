import os
from tqdm import tqdm
import threading
import argparse
import requests
from bs4 import BeautifulSoup as bs
from multiprocessing import cpu_count, Pool

from requests.adapters import HTTPAdapter, Retry

s = requests.Session()
retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504 ])
s.mount('http://', HTTPAdapter(max_retries=retries))

volume= '/mnt/volume_sfo3_01'
dataset_base = volume + '/jap-art/'

def scrape_artist(artist):
    href = artist['href']
    label = artist['href'].split('/')[-1]

    # create directory for artist if we haven't collected it already
    if not os.path.exists(dataset_base + label):
        os.makedirs(dataset_base + label)

        parse_page = True
        count = 0
        print(f"Collecting {label}:")
        while parse_page:
            artist_page = s.get(href)
            artist_soup = bs(artist_page.content, "html.parser")

            # parse and download images
            images = artist_soup.find_all('img')[1:]
            count += len(images)
            for image in images:
                response = s.get(image['src'])
                img_name = image['src'].split('/')[-1]
                file_name = dataset_base + label + "/" + img_name
                if response.status_code == 200:
                    with open(file_name, 'wb') as f:
                        f.write(response.content)
                else:
                    print(f"{response.status_code} error at start {count} for {image['src']}")

            # check if there are more pages from this label
            span = artist_soup.find_all('span', {'class' : 'next'})
            if len(span) > 0:
                span_a = span[0].find('a')
                href = span_a['href']
            else:
                parse_page = False

        print(f"{count}")

parser = argparse.ArgumentParser()
parser.add_argument('--yokai', action='store_true', help='scrape yokai images')
args = parser.parse_args()

if args.yokai and not os.path.exists(dataset_base + "yokai/"):
    os.makedirs(dataset_base + "yokai/")

    yokai_base_url = "https://www.nichibun.ac.jp/YoukaiGazou/image/U426_nichibunken"

    counter = 0
    for x in range(500):
        for y in range(50):
            for z in range(10):
                x2 = str(x).zfill(3)
                y2 = str(y).zfill(2)
                z2 = str(z).zfill(2)
                image_id = f"_0{x2}_00{y2}_00{z2}.jpg"
                response = s.get(yokai_base_url + image_id)
                if response.status_code == 200:
                    print(image_id)
                    with open(dataset_base + "yokai/" + image_id, 'wb') as f:
                        f.write(response.content)
                    counter += 1
                else:
                    break

    print(f"Collected {counter} images")
else:
    URL = "https://ukiyo-e.org/"
    page = requests.get(URL)
    soup = bs(page.content, "html.parser")
    artists = soup.find_all("a", {"class": "artist"})

    # thread pool to process artists
    for artist in artists:
        scrape_artist(artist)







