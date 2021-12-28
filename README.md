# 음식 양식 분류기 만들기  
image classifier model을 이용해서 음식 양식 구분하기  

## 1. Image Crwaling
구글 이미지 검색을 활용해서 원하는 음식 사진 검색해서 폴더 만들고 저장하기

```
$ python image_crawling.py
```

To add：

1. ```--searh_word=str```, e.g.```--searh_word='korean food'``` 검색하고 싶은 키워드 입력

2. ```--number=int```, e.g.```--number='500'``` 저장하고 싶은 이미지 개수 입력

3. ```--dir=str```, e.g.```--dir=korean``` 이미지 저장할 디렉토리 지정

## 2. fine tuning
vgg 16으로 한식 vs 이탈이아 음식 분류 모델 학습. transfer learning의 경우 정확도
