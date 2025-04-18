### 실제 도로를 고려한 실제 버스정류장 위치 ###

import pandas as pd
import folium

jeonju_bus = pd.read_excel("C:/Users/k0707/OneDrive/바탕 화면/정류장위치_도로고려.xlsx")
jeonju_b = jeonju_bus[['위도', '경도']]
latitude = jeonju_bus['위도'].mean()
longitude = jeonju_bus['경도'].mean()

map = folium.Map(location=[jeonju_b['위도'][0], jeonju_b['경도'][0]], zoom_start=10)

for index, row in jeonju_b.iterrows():
    folium.CircleMarker(
        location=[row['위도'], row['경도']],
        radius=5,
        color='red',
        fill=True,
        fill_opacity=0.6
    ).add_to(map)
map

### 버스 최단경로 구하기 ###

### Nearest Neighbor Algorithm ###

# 라이브러리
import folium
import numpy as np

# 최종 선정된 버스정류장의 위경도 좌표
coordinates = [
    (35.8146645, 127.1484079),
    (35.7998859, 127.0923654),
    (35.8546009, 127.1414109),
    (35.8124303, 127.1599803),
    (35.8300192, 127.1755955),
    (35.8487807, 127.1016654),
    (35.8605204, 127.1007421),
    (35.8721723, 127.0536104),
    (35.8474449, 127.1208169),
    (35.8258893, 127.1751299),
    (35.813535, 127.1471499),
    (35.8225967, 127.1438025),
    (35.8092752, 127.1621702),
    (35.8362909, 127.1280127),
    (35.8330014, 127.1691577),
    (35.818784, 127.1431934),
    (35.8142106, 127.1206966),
    (35.8475614, 127.1545963),
    (35.8169141, 127.144965),
    (35.8136618, 127.1018695),
    (35.8199878, 127.1208379),
    (35.8489732, 127.1606735),
    (35.8343343, 127.1290786),
    (35.8342893, 127.1331515)
]


# 두 정류장 사이의 거리 구하기
def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


# Nearest Neighbor Algorithm
def nearest_neighbor_algorithm(coordinates):
    unvisited = coordinates.copy()
    start = unvisited.pop(0)
    path = [start]
    current_location = start

    while unvisited:
        nearest_location = min(unvisited, key=lambda x: calculate_distance(current_location, x))
        path.append(nearest_location)
        unvisited.remove(nearest_location)
        current_location = nearest_location

    return path


# 경로 설정
path = nearest_neighbor_algorithm(coordinates)

# 지도 설정
map = folium.Map(location=[path[0][0], path[0][1]], zoom_start=12)

for coord in coordinates:
    folium.CircleMarker(
        location=coord,
        radius=5,
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(map)

# 경로 지도위에 그리기
for i in range(len(path) - 1):
    folium.PolyLine([path[i], path[i + 1]], color="red", weight=2.5, opacity=1).add_to(map)

map

### 2-opt Algorithm ###

import math
import folium


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def total_distance(tour, coordinates):
    return sum(distance(coordinates[tour[i]], coordinates[tour[i - 1]]) for i in range(len(tour)))


def two_opt(tour, coordinates):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour) - 1):  # 변경된 부분
                if j - i == 1:
                    continue
                # 변경된 부분: tour[j + 1]를 tour[(j + 1) % len(tour)]로 변경
                if distance(coordinates[tour[i - 1]], coordinates[tour[j]]) + distance(coordinates[tour[i]],
                                                                                       coordinates[tour[(j + 1) % len(
                                                                                               tour)]]) < distance(
                        coordinates[tour[i - 1]], coordinates[tour[i]]) + distance(coordinates[tour[j]], coordinates[
                    tour[(j + 1) % len(tour)]]):
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
    return tour


coordinates = [
    (35.8146645, 127.1484079),
    (35.7998859, 127.0923654),
    (35.8546009, 127.1414109),
    (35.8124303, 127.1599803),
    (35.8300192, 127.1755955),
    (35.8487807, 127.1016654),
    (35.8605204, 127.1007421),
    (35.8721723, 127.0536104),
    (35.8474449, 127.1208169),
    (35.8258893, 127.1751299),
    (35.813535, 127.1471499),
    (35.8225967, 127.1438025),
    (35.8092752, 127.1621702),
    (35.8362909, 127.1280127),
    (35.8330014, 127.1691577),
    (35.818784, 127.1431934),
    (35.8142106, 127.1206966),
    (35.8475614, 127.1545963),
    (35.8169141, 127.144965),
    (35.8136618, 127.1018695),
    (35.8199878, 127.1208379),
    (35.8489732, 127.1606735),
    (35.8343343, 127.1290786),
    (35.8342893, 127.1331515)
]

initial_tour = list(range(len(coordinates))) + [0]

optimized_tour = two_opt(initial_tour, coordinates)

print("Optimized tour:", optimized_tour)

print("Total distance:", total_distance(optimized_tour, coordinates))

m = folium.Map(location=[35.8146645, 127.1484079], zoom_start=13)

for index in optimized_tour:
    folium.CircleMarker(location=coordinates[index], radius=3, color='blue').add_to(m)

folium.PolyLine([(coordinates[index][0], coordinates[index][1]) for index in optimized_tour], color='red').add_to(m)

m

### ACO Algorithm ###

import math
import numpy as np
import folium


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


coordinates = [
    (35.8146645, 127.1484079),
    (35.7998859, 127.0923654),
    (35.8546009, 127.1414109),
    (35.8124303, 127.1599803),
    (35.8300192, 127.1755955),
    (35.8487807, 127.1016654),
    (35.8605204, 127.1007421),
    (35.8721723, 127.0536104),
    (35.8474449, 127.1208169),
    (35.8258893, 127.1751299),
    (35.813535, 127.1471499),
    (35.8225967, 127.1438025),
    (35.8092752, 127.1621702),
    (35.8362909, 127.1280127),
    (35.8330014, 127.1691577),
    (35.818784, 127.1431934),
    (35.8142106, 127.1206966),
    (35.8475614, 127.1545963),
    (35.8169141, 127.144965),
    (35.8136618, 127.1018695),
    (35.8199878, 127.1208379),
    (35.8489732, 127.1606735),
    (35.8343343, 127.1290786),
    (35.8342893, 127.1331515)
]

n_ants = 10
n_iterations = 100
decay = 0.1
alpha = 1
beta = 1

n_cities = len(coordinates)
distances = np.array(
    [[distance(coordinates[i], coordinates[j]) if i != j else 1e-7 for j in range(n_cities)] for i in range(n_cities)])
pheromones = np.ones((n_cities, n_cities))
eta = 1 / distances


def calculate_probability(k, city, unvisited, pheromones, eta, alpha, beta):
    total = sum(pheromones[city][l] ** alpha * eta[city][l] ** beta for l in unvisited)
    return (pheromones[city][k] ** alpha * eta[city][k] ** beta) / total


def aco_algorithm(n_ants, n_iterations, decay, alpha, beta, pheromones, eta):
    best_tour = None
    best_length = float('inf')

    for iteration in range(n_iterations):
        tours = []
        lengths = []

        for ant in range(n_ants):
            unvisited = list(range(n_cities))
            city = np.random.randint(n_cities)
            unvisited.remove(city)
            tour = [city]

            while unvisited:
                probabilities = [calculate_probability(k, city, unvisited, pheromones, eta, alpha, beta) for k in
                                 unvisited]
                k = np.random.choice(unvisited, p=probabilities)
                unvisited.remove(k)
                tour.append(k)
                city = k

            length = sum(distances[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)) + distances[tour[-1]][tour[0]]
            tours.append(tour)
            lengths.append(length)

            if length < best_length:
                best_length = length
                best_tour = tour

        for tour, length in zip(tours, lengths):
            for i in range(len(tour) - 1):
                pheromones[tour[i]][tour[i + 1]] += 1 / length
            pheromones[tour[-1]][tour[0]] += 1 / length

        pheromones = (1 - decay) * pheromones

    return best_tour, best_length


best_tour, best_length = aco_algorithm(n_ants, n_iterations, decay, alpha, beta, pheromones, eta)

m = folium.Map(location=[35.8146645, 127.1484079], zoom_start=13)

for index in best_tour:
    folium.CircleMarker(coordinates[index], radius=3, color='blue').add_to(m)

folium.PolyLine([(coordinates[i][0], coordinates[i][1]) for i in best_tour] + [
    (coordinates[best_tour[0]][0], coordinates[best_tour[0]][1])], color="red", weight=2.5, opacity=1).add_to(m)

m

### 크리스토피데스 Algorithm ###

!pip
install
networkx
import math
import numpy as np
import networkx as nx
import folium
from scipy.spatial import distance_matrix
from itertools import combinations


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


coordinates = [
    (35.8146645, 127.1484079),
    (35.7998859, 127.0923654),
    (35.8546009, 127.1414109),
    (35.8124303, 127.1599803),
    (35.8300192, 127.1755955),
    (35.8487807, 127.1016654),
    (35.8605204, 127.1007421),
    (35.8721723, 127.0536104),
    (35.8474449, 127.1208169),
    (35.8258893, 127.1751299),
    (35.813535, 127.1471499),
    (35.8225967, 127.1438025),
    (35.8092752, 127.1621702),
    (35.8362909, 127.1280127),
    (35.8330014, 127.1691577),
    (35.818784, 127.1431934),
    (35.8142106, 127.1206966),
    (35.8475614, 127.1545963),
    (35.8169141, 127.144965),
    (35.8136618, 127.1018695),
    (35.8199878, 127.1208379),
    (35.8489732, 127.1606735),
    (35.8343343, 127.1290786),
    (35.8342893, 127.1331515)
]

dist_matrix = [[distance(coordinates[i], coordinates[j]) for j in range(len(coordinates))] for i in
               range(len(coordinates))]
dist_matrix = np.array(dist_matrix)

G = nx.Graph()
for i in range(len(coordinates)):
    for j in range(len(coordinates)):
        if i != j:
            G.add_edge(i, j, weight=dist_matrix[i][j])

T = nx.minimum_spanning_tree(G)

O = [v for v, d in T.degree() if d % 2 == 1]

M = nx.max_weight_matching(G.subgraph(O), maxcardinality=True)

T = nx.MultiGraph(T)
T.add_edges_from(M)

euler_circuit = list(nx.eulerian_circuit(T))

current_location = euler_circuit[0][0]
visited = set()
tour = [current_location]
visited.add(current_location)

for u, v in euler_circuit:
    if v not in visited:
        tour.append(v)
        visited.add(v)

m = folium.Map(location=[35.8146645, 127.1484079], zoom_start=13)

for index in tour:
    folium.CircleMarker(coordinates[index], radius=3, color='blue').add_to(m)

folium.PolyLine(
    [(coordinates[i][0], coordinates[i][1]) for i in tour] + [(coordinates[tour[0]][0], coordinates[tour[0]][1])],
    color="red", weight=2.5, opacity=1).add_to(m)