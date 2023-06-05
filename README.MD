# EXP-10 Data Science Process on Complex Dataset

# AIM

To Perform Data Science Process on a complex dataset and save the data to a file.

# ALGORITHM

Step 1 Read the given Data

Step 2 Clean the Data Set using Data Cleaning Process

Step 3 Apply Feature Generation/Feature Selection Techniques on the data set

Step 4 Apply EDA /Data visualization techniques to all the features of the data set

# CODE

import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("/content/Disease_symptom_and_patient_profile_dataset.csv")

df

df.head()

df.info()

df.tail()

df.isnull().sum()

df.shape

df.nunique()

sns.catplot(x = 'Outcome Variable' , y = 'Age' , data = df , kind = "swarm")

plt.figure(figsize=(20,10))

sns.lineplot(x='Disease',y='Age',data =df);

plt.figure(figsize=(20,10))

sns.barplot(x='Fever',y='Age',data =df);

sns.displot(df['Age'] , kde=True)

df.groupby('Gender').size().plot(kind='pie', autopct='%.2f')

sns.catplot(x='Cough' , kind='count',data=df , hue = "Cholesterol Level")

df.groupby('Blood Pressure').size().plot(kind='pie', autopct='%.2f')

sns.catplot(x='Gender' , kind='count',data=df , hue = "Cholesterol Level")

df = df.iloc[:,1:]

df.groupby('Fatigue').size().plot(kind='pie', autopct='%.2f')

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, [0, 2]]

Y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, [0, 2]]

Y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Example dataset

data = [[10, 20, 30],
[5, 15, 25],
[3, 6, 9],
[8, 12, 16]]

# Min-max scaling

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

print("Min-max scaled data:")

print(scaled_data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

print("Standard scaled data:")

print(scaled_data)

np.random.seed(42)

data = np.random.normal(0, 1, 1000)

plt.figure(figsize=(8, 6))

plt.hist(data, bins=30, edgecolor='black')

plt.xlabel('Value')

plt.ylabel('Frequency')

plt.title('Histogram')

plt.show()


# OUTPUT
![Screenshot 2023-06-05 195552](https://github.com/dhivyapriyar/EX10/assets/119477552/34736c39-f617-4990-8d45-2383e03e105d)
![Screenshot 2023-06-05 195605](https://github.com/dhivyapriyar/EX10/assets/119477552/3db7602c-2921-473a-bbf0-44c756e910f5)
![Screenshot 2023-06-05 195614](https://github.com/dhivyapriyar/EX10/assets/119477552/093fb1d0-07a7-44cf-ad6e-8493de4d13c0)
![Screenshot 2023-06-05 195625](https://github.com/dhivyapriyar/EX10/assets/119477552/1b842ef1-3af7-40a0-bff7-8225fe929693)
![Screenshot 2023-06-05 195631](https://github.com/dhivyapriyar/EX10/assets/119477552/039814cf-ca4d-4b09-bb9e-b45d531c28df)
![Screenshot 2023-06-05 195637](https://github.com/dhivyapriyar/EX10/assets/119477552/dd2e9a8a-5b2a-4b3f-b21a-a715a8515132)
![Screenshot 2023-06-05 195643](https://github.com/dhivyapriyar/EX10/assets/119477552/3ccf909c-1ef0-4d4e-baf5-2663078afc03)
![Screenshot 2023-06-05 195652](https://github.com/dhivyapriyar/EX10/assets/119477552/a7fde7e3-9bfa-4334-8170-22d92091ebc5)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/7c0b7ef9-1c67-4bb8-9b46-12f21ee83bf5)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/46be219a-b03b-485e-a380-c9bf4f4c33ec)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/bed98035-2045-4e0e-a458-1cadd122a703)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/470d7c36-dd62-42ad-b55d-3187629fbb9e)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/fe661d4b-be75-4974-ad23-9f6374fc9936)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/bb81300f-69f6-4fbf-b1ff-921e96deec47)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/2f968043-7fcf-40eb-845c-439206fed153)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/cad5a3ed-904d-4d5d-81f0-ab8ea9db2120)
![image](https://github.com/dhivyapriyar/EX10/assets/119477552/e6937ad5-fb5b-4985-a56f-710b56dbe1fc)



# RESULT
Thus Data Science Process on a complex dataset was performed successfully.
