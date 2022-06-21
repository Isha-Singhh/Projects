import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("webtoon_originals_de.csv")
print(data.head(10))
print(data.shape)
print(data.info)
print(data.describe())
print(data.columns)
print(data.isna().any()) #to detect the non value or NaN
print(data.iloc) #for specified column and row
print(data['title_id'].count())
print(data['title'].unique())
print(data.iloc[data['rating'].idxmax()])
print(data.iloc[data['likes'].idxmax()])

fig , ax =plt.subplots(figsize=(50,5))
sns.countplot(x='genre',ax=ax,data=data)

data1=data.head(20).rating
data2=data.head(20).title
sns.barplot(x=data1,y=data2)

probability_=data['status']==data['daily_pass']
probability_.groupby(probability_)
sns.countplot(probability_)

data3=data.groupby('weekdays')['views'].value_counts()
print(data3)
data4=data.groupby(['title','rating'])
print(data4.first())
data5=data['daily_pass'].value_counts()
print((data5))

data6=pd.read_csv("webtoon_originals_de.csv",index_col='title')
data7=data.groupby('title')['subscribers'].value_counts()
print(data7)
plt.show()