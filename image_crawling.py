import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import urllib.request

import os
import time

import argparse

parser = argparse.ArgumentParser(description='Image Crawling')

parser.add_argument('--search_word', type=str, help = 'Enter searching word', dest='search_word')
parser.add_argument('--number', type=int, help = 'Enter what you want image number', dest='num')
parser.add_argument('--dir', type=str, help = 'Enter save image file directory', default='./')
args = parser.parse_args()

def create_folder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: Creating directory. ' + dir)

# search_word = '한식'
create_folder('./' + args.dir)

driver = webdriver.Chrome('./chromedriver')

print(args.search_word, '검색')
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")

word = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input')
word.send_keys(args.search_word)
driver.find_element_by_xpath('//*[@id="sbtc"]/button').click()


elem = driver.find_element_by_tag_name("body")
for i in range(60):
    elem.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.1)

try:
    driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input').click()
    for i in range(60):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.1)
except:
    pass

links=[]
images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd")
for image in images:
    if image.get_attribute('src')!=None:
        if len(links) < args.num:
            links.append(image.get_attribute('src'))
        else:
            break

print(args.search_word+' 찾은 이미지 개수:',len(links))
time.sleep(2)


for k,i in enumerate(links):
    url = i
    start = time.time()
    try:
        urllib.request.urlretrieve(url, "./"+args.dir+"/"+str(k)+".jpg")
        print(str(k+1)+'th'+' '+args.search_word+' 다운로드 중....... Download time : '+str(time.time() - start)[:5]+' 초')
        # print(args.search_word+' ---다운로드 완료---')
        driver.close()

    except:
        pass
    end = time.time()
    print(str(end - start)[:5] + '초')
