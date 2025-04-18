### 숙박 데이터 수집 ###
import pandas as pd
import numpy as np
import pyproj
import folium

dataset0 = pd.read_csv('C:/Users/gogus/python/공모전/dataset/새 폴더/숙박/6450000_CSV/6450000_전라북도_03_11_01_P_관광숙박업.csv',
                       encoding='cp949')
dataset1 = pd.read_csv('C:/Users/gogus/python/공모전/dataset/새 폴더/숙박/6450000_CSV/6450000_전라북도_03_11_02_P_관광펜션업.csv',
                       encoding='cp949')
dataset2 = pd.read_csv('C:/Users/gogus/python/공모전/dataset/새 폴더/숙박/6450000_CSV/6450000_전라북도_03_11_03_P_숙박업.csv',
                       encoding='cp949')

dataset = pd.DataFrame()
dataset = pd.concat([dataset, dataset1, dataset2])

dataset.tail(5)

dataset.columns

df1 = dataset[dataset['소재지전체주소'].str.contains('전주시', na=False)]
df1

df = df1[['좌표정보(x)', '좌표정보(y)']]
df

df['좌표정보(x)'] = pd.to_numeric(df['좌표정보(x)'], errors="coerce")
df['좌표정보(y)'] = pd.to_numeric(df['좌표정보(y)'], errors="coerce")

df = df.dropna()
df.index = range(len(df))
df.tail()


def project_array(coord, p1_type, p2_type):
    """
    좌표계 변환 함수
    - coord: x, y 좌표 정보가 담긴 NumPy Array
    - p1_type: 입력 좌표계 정보 ex) epsg:5179
    - p2_type: 출력 좌표계 정보 ex) epsg:4326
    """
    p1 = pyproj.Proj(init=p1_type)
    p2 = pyproj.Proj(init=p2_type)
    fx, fy = pyproj.transform(p1, p2, coord[:, 0], coord[:, 1])
    return np.dstack([fx, fy])[0]


# DataFrame -> NumPy Array 변환
coord = np.array(df)
coord

# 좌표계 정보 설정
p1_type = "epsg:2097"
p2_type = "epsg:4326"

# project_array() 함수 실행
result = project_array(coord, p1_type, p2_type)
result

df['경도'] = result[:, 0]
df['위도'] = result[:, 1]

df.tail()

# 지도 중심 좌표 설정
lat_c, lon_c = 37.53165351203043, 126.9974246490573

# Folium 지도 객체 생성
m = folium.Map(location=[lat_c, lon_c], zoom_start=6)

# 마커 생성
for _, row in df.iterrows():
    lat, lon = row['위도'], row['경도']
    folium.Marker(location=[lat, lon]).add_to(m)


### 음식점 데이터 전처리 ###

food = pd.read_csv("C:/Users/k0707/OneDrive/바탕 화면/전주_음식점_100.csv", encoding='EUC_KR')
address = food['주소']
# 주소 데이터 깔끔하게 다듬기
for i in range(len(address)):
    a = address[i].split(' ')
    address[i] = " ".join(a[0:5])

!pip
install
geopy

####### 도로명주소 위도 경도 값으로 바꿔주기 ########
from geopy.geocoders import Nominatim

geo_local = Nominatim(user_agent='South Korea')


# 위도, 경도 반환하는 함수
def geocoding(address):
    try:
        geo = geo_local.geocode(address)
        x_y = [geo.latitude, geo.longitude]
        return x_y

    except:
        return [0, 0]


latitude = []
longitude = []

for i in address:
    latitude.append(geocoding(i)[0])
    longitude.append(geocoding(i)[1])

address_df = pd.DataFrame({'이름': food['업소명'], '주소': address, '위도': latitude, '경도': longitude})
address_df
