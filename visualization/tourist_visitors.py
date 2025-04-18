### 주요관광지 입장객 수집 ###
!sudo
apt - get
install - y
fonts - nanum
!sudo
fc - cache - fv
!rm
~ /.cache / matplotlib - rf

import matplotlib.pyplot as plt

plt.rc('font', family='NanumBarunGothic')

from google.colab import drive

drive.mount('/content/drive')
import pandas as pd

jeonju = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/메인프로젝트/주요관광지점 입장객_전주.csv',
                     encoding='cp949')

jeonju.head()

jeonju.columns = jeonju.iloc[0]

jeonju_1 = jeonju.iloc[:, [2, 3, 14]]
jeonju_1.columns = ['관광지', '내/외국인', '2022년']

jeonju_1 = jeonju_1.copy().loc[jeonju_1['내/외국인'] == '합계']

jeonju_1['2022년'] = jeonju_1['2022년'].str.replace(',', '')
jeonju_1 = jeonju_1.fillna(0)
jeonju_1['2022년'] = jeonju_1['2022년'].astype('int')

jeonju_1.drop(['내/외국인'], axis=1, inplace=True)

jeonju_1 = jeonju_1.sort_values(by='2022년').reset_index()

jeonju_1.drop(['index'], axis=1, inplace=True)

j = jeonju_1.iloc[[12, 14], :]
j.loc[3] = ['한옥마을', 11294916]

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.barplot(x='관광지', y='2022년', data=j, palette='YlGnBu')