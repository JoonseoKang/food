import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import urllib.request
import os
import time
import argparse

parser = argparse.ArgumentParser(description="Image Crawling")
parser.add_argument("--search_word", type=str, help="Enter searching word", dest="search_word")
parser.add_argument("--number", type=int, help="Enter what you want image number", dest="num")
parser.add_argument("--dir", type=str, help="Enter save image file directory", default="./")
args = parser.parse_args()


def create_folder(dir):
    os.makedirs(dir, exist_ok=True)


create_folder("./" + args.dir)

driver = webdriver.Chrome("./chromedriver")
print(args.search_word, "검색")
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")

word = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input')
word.send_keys(args.search_word)
word.send_keys(Keys.RETURN)

elem = driver.find_element_by_tag_name("body")
for i in range(60):
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.1)

try:
    show_more = driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input')
    show_more.click()
    for i in range(60):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.1)
except selenium.common.exceptions.NoSuchElementException:
    pass

links = []
images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")
for image in images:
    if image.get_attribute("src") is not None:
        if len(links) < args.num:
            links.append(image.get_attribute("src"))
        else:
            break

print(args.search_word + " 찾은 이미지 개수:", len(links))
time.sleep(2)

for k, url in enumerate(links):
    start = time.time()
    try:
        urllib.request.urlretrieve(url, f"./{args.dir}/{k}.jpg")
        print(f"{k+1}th {args.search_word} 다운로드 중....... Download time : {time.time() - start:.2f} 초")
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"Error downloading {url}: {e}")
    end = time.time()
    print(f"{end - start:.2f}초")
driver.close()
