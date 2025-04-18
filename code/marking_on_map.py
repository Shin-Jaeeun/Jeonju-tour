### 관광지 지도 위에 표시 ###

# 라이브러리 다운로드
import folium
import pandas as pd

# 관광지 주소를 위경도 좌표로 변환한 파일 불러오기
jeonju_tour = pd.read_excel("C:/Users/k0707/OneDrive/바탕 화면/관광지_위경도.xlsx")
jeonju_t = jeonju_tour[['장소', '위도', '경도']]
# 지도 만들기
latitude = jeonju_tour['위도'].mean()
longitude = jeonju_tour['경도'].mean()
ma = folium.Map(location=[latitude, longitude], zoom_start=12)
for i in jeonju_t.index:
    sub_lat = jeonju_t.loc[i, '위도']
    sub_long = jeonju_t.loc[i, '경도']

    folium.Marker(location=[sub_lat, sub_long],
                  icon=folium.Icon('red', icon='heart'),
                  ).add_to(ma)
ma  # 지도표시

### 전주 내 숙박시설 위치 표시 ###

# 라이브러리
import folium
import pandas as pd

# 숙박 시설에서 영업중인 곳의 주소를 위경도 좌표로 변환한 파일 불러오기
jeonju_house = pd.read_excel("C:/Users/k0707/OneDrive/바탕 화면/전주_숙박_영업중.xlsx")
jeonju_h = jeonju_house[['위도', '경도']]
latitude = jeonju_house['위도'].mean()
longitude = jeonju_house['경도'].mean()
# 지도 생성
ma = folium.Map(location=[latitude, longitude], zoom_start=12)
for i in jeonju_h.index:
    sub_lat = jeonju_h.loc[i, '위도']
    sub_long = jeonju_h.loc[i, '경도']

    folium.Marker(location=[sub_lat, sub_long],
                  icon=folium.Icon('blue', icon='star'),
                  ).add_to(ma)
ma

### 음식점 위치 표시 ###

# 라이브러리
import folium
import pandas as pd

# 전주 내 외지인 검색 top 100 음식점의 주소를 위 경도 좌표로 변환한 파일 불러오기
jeonju_food = pd.read_excel("C:/Users/k0707/OneDrive/바탕 화면/전주_음식점_위경도.xlsx")
jeonju_f = jeonju_food[['위도', '경도']]
latitude = jeonju_food['위도'].mean()
longitude = jeonju_food['경도'].mean()
# 지도 생성
ma = folium.Map(location=[latitude, longitude], zoom_start=12)
for i in jeonju_f.index:
    sub_lat = jeonju_f.loc[i, '위도']
    sub_long = jeonju_f.loc[i, '경도']

    folium.Marker(location=[sub_lat, sub_long],
                  icon=folium.Icon('pink', icon='fire'),
                  ).add_to(ma)
ma