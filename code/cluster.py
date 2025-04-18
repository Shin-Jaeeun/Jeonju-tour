### 숙박시설 군집화 ###
# 라이브러리 불러오기
import numpy as np
import pandas as pd
import folium
from sklearn.cluster import KMeans

# 파일 찾아오기
jeonju_house_1 = pd.read_excel("C:/Users/k0707/OneDrive/바탕 화면/전주숙박_위경도.xlsx")

# 군집화
Y = jeonju_house_1[['위도', '경도']].values
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Y)
jeonju_house_1['cluster'] = kmeans.labels_
jeonju_house_1.head()

# 군집화 결과를 지도위에 표시
mean_latitude = jeonju_house_1['위도'].mean()
mean_longitude = jeonju_house_1['경도'].mean()
m = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=10)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
for index, row in jeonju_house_1.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=5,
        color=colors[int(row['cluster'])],
        fill=True,
        fill_color=colors[int(row['cluster'])]
    ).add_to(m)
m

### 음식점 군집화 ###

# 라이브러리 불러오기
import numpy as np
import pandas as pd
import folium
from sklearn.cluster import KMeans

# 파일 찾아오기
jeonju_food_1 = pd.read_excel("C:/Users/k0707/OneDrive/바탕 화면/전주_음식점_위경도.xlsx")

# 군집화
Y = jeonju_food_1[['위도', '경도']].values
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Y)
jeonju_food_1['cluster'] = kmeans.labels_
jeonju_food_1.head()

# 군집화 결과를 지도위에 표시
mean_latitude = jeonju_food_1['위도'].mean()
mean_longitude = jeonju_food_1['경도'].mean()
m = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=10)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
for index, row in jeonju_food_1.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=5,
        color=colors[int(row['cluster'])],
        fill=True,
        fill_color=colors[int(row['cluster'])]
    ).add_to(m)
m

### 숙박시설 군집 중심 찾기 ###

# 라이브러리 불러오기
import numpy as np
import pandas as pd
import folium
from sklearn.cluster import KMeans


# 기하적 군집 중심을 찾기위한 함수 정의
def weiszfeld_algorithm(points, max_iterations=1000, tolerance=1e-6):
    median = np.mean(points, axis=0)
    for i in range(max_iterations):
        distances = np.linalg.norm(points - median, axis=1)
        if np.isclose(distances, 0).any():
            return points[np.isclose(distances, 0)]
        weights = 1.0 / distances
        new_median = np.sum(weights[:, np.newaxis] * points, axis=0) / np.sum(weights)
        if np.linalg.norm(new_median - median) < tolerance:
            return new_median
        median = new_median
    return median


# 군집 내 개체가 1개이면 군집화 안 하게 설정
n_clusters = 5
geometric_medians = []
for i in range(n_clusters):
    cluster_points = jeonju_house_1[jeonju_house_1['cluster'] == i][['위도', '경도']].values
    if cluster_points.shape[0] > 1:
        geometric_median = weiszfeld_algorithm(cluster_points)
        geometric_medians.append(geometric_median)
        distances = np.linalg.norm(cluster_points - geometric_median, axis=1)
        sum_of_distances = np.sum(distances)

        print(f"Sum of distances for Cluster {i}: {sum_of_distances}")

        latitude, longitude = geometric_median
        print(f"Latitude and Longitude of the Geometric Median for Cluster {i}: ({latitude}, {longitude})")
    else:
        print(f"Cluster {i} has only one object, skipping...")

mean_latitude = jeonju_house_1['위도'].mean()
mean_longitude = jeonju_house_1['경도'].mean()
mmp = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=10)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

for index, row in jeonju_house_1.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=5,
        color=colors[int(row['cluster'])],
        fill=True,
        fill_color=colors[int(row['cluster'])]
    ).add_to(mmp)

for i, coords in enumerate(geometric_medians):
    folium.Marker(
        location=coords,
        popup=f'Geometric Median of Cluster {i}',
        icon=folium.Icon(color=colors[i])
    ).add_to(mmp)

mmp

### 음식점 군집 중심 찾기 ###

# 라이브러리 불러오기
import numpy as np
import pandas as pd
import folium
from sklearn.cluster import KMeans


# 기하적 군집 중심을 찾기위한 함수 정의
def weiszfeld_algorithm(points, max_iterations=1000, tolerance=1e-6):
    median = np.mean(points, axis=0)
    for i in range(max_iterations):
        distances = np.linalg.norm(points - median, axis=1)
        if np.isclose(distances, 0).any():
            return points[np.isclose(distances, 0)]
        weights = 1.0 / distances
        new_median = np.sum(weights[:, np.newaxis] * points, axis=0) / np.sum(weights)
        if np.linalg.norm(new_median - median) < tolerance:
            return new_median
        median = new_median
    return median


# 군집 내 개체가 1개이면 군집화 안 하게 설정
n_clusters = 5
geometric_medians = []
for i in range(n_clusters):
    cluster_points = jeonju_food_1[jeonju_food_1['cluster'] == i][['위도', '경도']].values
    if cluster_points.shape[0] > 1:
        geometric_median = weiszfeld_algorithm(cluster_points)
        geometric_medians.append(geometric_median)
        distances = np.linalg.norm(cluster_points - geometric_median, axis=1)
        sum_of_distances = np.sum(distances)

        print(f"Sum of distances for Cluster {i}: {sum_of_distances}")

        latitude, longitude = geometric_median
        print(f"Latitude and Longitude of the Geometric Median for Cluster {i}: ({latitude}, {longitude})")
    else:
        print(f"Cluster {i} has only one object, skipping...")

mean_latitude = jeonju_food_1['위도'].mean()
mean_longitude = jeonju_food_1['경도'].mean()
mmp = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=10)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

for index, row in jeonju_food_1.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=5,
        color=colors[int(row['cluster'])],
        fill=True,
        fill_color=colors[int(row['cluster'])]
    ).add_to(mmp)

for i, coords in enumerate(geometric_medians):
    folium.Marker(
        location=coords,
        popup=f'Geometric Median of Cluster {i}',
        icon=folium.Icon(color=colors[i])
    ).add_to(mmp)

mmp