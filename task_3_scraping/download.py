import os
import time
import urllib.request
from urllib.parse import quote
from bs4 import BeautifulSoup
from selenium import webdriver

wait_for_page_load = 3
wait_before_audios_download = 1
wait_between_audio_download = .3
wait_between_pages = 3

def store_audio(lesson, links):
    for idx, link in enumerate(links):
        parts = link.split("/")
        name = parts[-1]
        print(name)
        file_path = f"./audio/{lesson:02d}/{idx + 1:02d}-{name}"
        if not os.path.exists(file_path):
            urllib.request.urlretrieve("/".join(parts[0:-1]) + "/" + quote(name), file_path)
            time.sleep(wait_between_audio_download)


def read_page(url: str, lesson: int):
    directory = f"./audio/{lesson:02d}"
    os.mkdir(directory)
    driver.get(url)
    driver.implicitly_wait(wait_for_page_load * 1000)
    time.sleep(wait_for_page_load)
    html_content = driver.page_source

    soup = BeautifulSoup(html_content, "html.parser")

    sources = [source["src"] for source in soup.find_all("source") if source["src"].endswith(".mp3")]

    with open(f"{directory}/links.txt", "w", encoding="utf-8") as links_file:
        links_file.write("\n".join(sources))
    
    time.sleep(wait_before_audios_download)

    store_audio(lesson, sources)

    time.sleep(wait_between_pages)


with open(f"./audio/links.txt", "r", encoding="utf-8") as links_file:
    links = links_file.read().split("\n")

driver = webdriver.Firefox()
[read_page(link[1], link[0]) for link in enumerate(links)]
    
