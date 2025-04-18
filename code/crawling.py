### 네이버 블로그 크롤링 ###

import urllib.request
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from selenium.webdriver.common.by import By

# 웹드라이버 설정
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)

# 정보입력
client_id = "JLNLroKbYKkQae1rXCXU"  # 발급받은 id 입력
client_secret = "8m3PJuBavz"  # 발급받은 secret 입력
quote = input("검색어를 입력해주세요.: ")  # 검색어 입력받기
encText = urllib.parse.quote(quote)
display_num = input("검색 출력결과 갯수를 적어주세요.(최대100, 숫자만 입력): ")  # 출력할 갯수 입력받기
url = "https://openapi.naver.com/v1/search/blog?query=" + encText + "&display=" + display_num  # json 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id", client_id)
request.add_header("X-Naver-Client-Secret", client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()

if (rescode == 200):
    response_body = response.read()
    # print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)

body = response_body.decode('utf-8')
body

# body를 나누기
list1 = body.split('\n\t\t{\n\t\t\t')
# naver블로그 글만 가져오기
list1 = [i for i in list1 if 'naver' in i]
print(list1)

# 블로그 제목, 링크 뽑기
import re

titles = []
links = []
for i in list1:
    title = re.findall('"title":"(.*?)",\n\t\t\t"link"', i)
    link = re.findall('"link":"(.*?)",\n\t\t\t"description"', i)
    titles.append(title)
    links.append(link)

titles = [r for i in titles for r in i]
links = [r for i in links for r in i]

print('<<제목 모음>>')
print(titles)
print('총 제목 수: ', len(titles), '개')  # 제목갯수확인
print('\n<<링크 모음>>')
print(links)
print('총 링크 수: ', len(links), '개')  # 링크갯수확인

# 링크를 다듬기 (필요없는 부분 제거 및 수정)
blog_links = []
for i in links:
    a = i.replace('\\', '')
    b = a.replace('?Redirect=Log&logNo=', '/')
    blog_links.append(b)

print(blog_links)
print('생성된 링크 갯수:', len(blog_links), '개')

# 본문 크롤링
import time
from selenium import webdriver

# 크롬 드라이버 설치
driver = webdriver.Chrome("C:/Users/gogus/python/Untitled Folder/chromedriver_win32/chromedriver.exe")
driver.implicitly_wait(3)

# 블로그 링크 하나씩 불러서 크롤링
contents = []
for i in blog_links:
    # 블로그 링크 하나씩 불러오기
    driver.get(i)
    time.sleep(1)
    # 블로그 안 본문이 있는 iframe에 접근하기
    driver.switch_to.frame("mainFrame")
    # 본문 내용 크롤링하기
    # 본문 내용 크롤링하기
    try:
        a = driver.find_element(By.CSS_SELECTOR, 'div.se-main-container').text
        contents.append(a)
    # NoSuchElement 오류시 예외처리(구버전 블로그에 적용)
    except NoSuchElementException:
        a = driver.find_element(By.CSS_SELECTOR, 'div#content-area').text
        contents.append(a)
    # print(본문: \n', a)

driver.quit()  # 창닫기
print("<<본문 크롤링이 완료되었습니다.>>")

# 제목, 블로그링크, 본문내용 Dataframe으로 만들기
import pandas as pd

df = pd.DataFrame({'제목': titles, '링크': blog_links, '내용': contents})

### 크롤링 내용으로 워드클라우드 작성 ###

from wordcloud import WordCloud  # 워드클라우드 제작 라이브러리
import pandas as pd  # 데이터 프레임 라이브러리
import numpy as np  # 행렬 라이브러리
import matplotlib.pyplot as plt  # 워드클라우드 시각화 라이브러리

df['제목'] = df['제목'].str.replace('[^가-힣]', ' ', regex=True)

import konlpy

kkma = konlpy.tag.Kkma()  # 형태소 분석기 꼬꼬마(Kkma)

nouns = df['제목'].apply(kkma.nouns)

nouns = nouns.explode()

df_word = pd.DataFrame({'word': nouns})
df_word['count'] = df_word['word'].str.len()
df_word = df_word.query('count >= 2')

df_word = df_word.groupby('word', as_index=False).count().sort_values('count', ascending=False)
df_word

df_word = df_word.iloc[:, :]
df_word

dic_word = df_word.set_index('word').to_dict()['count']

wc = WordCloud(random_state=123, font_path="C:/Users/gogus/python/기타/nanum-barun-gothic/NanumBarunGothic.otf"
               , width=400,
               height=400, background_color='white')

img_wordcloud = wc.generate_from_frequencies(dic_word)

plt.figure(figsize=(10, 10))  # 크기 지정하기
plt.axis('off')  # 축 없애기
plt.imshow(img_wordcloud)  # 결과 보여주기

df['내용'] = df['내용'].str.replace('[^가-힣]', ' ', regex=True)

import konlpy

kkma = konlpy.tag.Kkma()  # 형태소 분석기 꼬꼬마(Kkma)

nouns = df['내용'].apply(kkma.nouns)

nouns = nouns.explode()

df_word = pd.DataFrame({'word': nouns})
df_word['count'] = df_word['word'].str.len()
df_word = df_word.query('count >= 2')

df_word = df_word.groupby('word', as_index=False).count().sort_values('count', ascending=False)
df_word

df_word = df_word.iloc[1:, :]
df_word

dic_word = df_word.set_index('word').to_dict()['count']

wc = WordCloud(random_state=123, font_path="C:/Users/gogus/python/기타/nanum-barun-gothic/NanumBarunGothic.otf"
               , width=400,
               height=400, background_color='white')

img_wordcloud = wc.generate_from_frequencies(dic_word)

plt.figure(figsize=(10, 10))  # 크기 지정하기
plt.axis('off')  # 축 없애기
plt.imshow(img_wordcloud)  # 결과 보여주기