# EXP-10 Data Science Process on Complex Dataset

# AIM

To Perform Data Science Process on a complex dataset and save the data to a file.

# ALGORITHM

Step 1 Read the given Data

Step 2 Clean the Data Set using Data Cleaning Process

Step 3 Apply Feature Generation/Feature Selection Techniques on the data set

Step 4 Apply EDA /Data visualization techniques to all the features of the data set

# CODE

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

df.head()

df.isnull().sum()

plt.figure(figsize=(5,5))

plt.title("Data with Outliers")

df.boxplot()

plt.show()

plt.figure(figsize=(5,5))

cols = ['size','tip','total_bill']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

df['sex'].unique()

!pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

be = BinaryEncoder()

data = be.fit_transform(df['sex'])

df = pd.concat([df,data],axis=1)

df

df['smoker'].unique()

data = be.fit_transform(df['smoker'])

df = pd.concat([df,data],axis=1)

df

df['day'].unique()

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

clim = ['Thur','Fri','Sat','Sun']

en= OrdinalEncoder(categories = [clim])

df['day']=en.fit_transform(df[["day"]])

df

df['time'].unique()

le = LabelEncoder()

df['time'] = le.fit_transform(df[["time"]])

df

df.drop('sex',axis=1,inplace=True)

df.drop('smoker',axis=1,inplace=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Min-max scaled data:")

print(scaled_data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("Standard scaled data:")

print(scaled_data)

import seaborn as sns

sns.scatterplot(data=df)

sns.displot(df['size'],kde=True)

sns.scatterplot(x="total_bill", y="tip", data=df)

plt.title("Correlation between Tip Amount and Total Bill Amount")

plt.show()

df["tip_percent"] = df["tip"] / df["total_bill"]

sns.barplot(x=df['size'],y=df['tip_percent'],data=df)

plt.title("Tip Percentage by Dining Party Size")

plt.show()

sns.barplot(x=df['time'], y=df['total_bill'])

plt.title("Highest Total Bill Amount by Time")

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)

# OUTPUT

![image](https://github.com/dhivyapriyar/EX10/assets/119477552/4c7147cc-b102-4b74-8c13-39d45ced8e9f)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/e55c1f03-3018-4e2a-bfea-eebf7d83b92e)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/5805cccf-04a4-4009-8dd8-225cb735b7ef)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/f84c98a4-dbe2-4eca-83b2-309513037f17)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/75549fad-d58f-4201-871f-ddba55e4f2b1)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/f8c9d505-60e6-4658-99f9-810eaf088600)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/629686db-c851-4b95-8350-179da1019af7)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/ea991415-87e9-4c04-8e19-bfb98e1d96bd)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/a938ed6c-5410-4587-96d6-479c62159146)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/b09288f3-4846-4992-b7d9-7bf4bff4f367)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/0f7a15bb-f061-4045-bc60-73ca2a22930a)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/6829ab00-9671-4ffb-9055-eac262254401)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/2adf32fe-aef9-4ac9-9b1a-49b39cfeb6fb)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/119f8d61-8215-49d2-9287-886f7ba7e63a)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/5fec2cf6-82df-40b9-9d6e-f0f458b77368)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/a0cc8b17-c311-40dc-9800-08f7c0a65f37)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/fe79dc86-6dad-4a08-ab6a-24fbe068014e)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/fb4aea0f-d152-46b4-9037-8f0c8ba3321c)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/496366dc-4651-4d3c-ae3d-3e1433b66283)

# RESULT
Thus Data Science Process on a complex dataset was performed successfully.
