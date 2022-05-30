import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import streamlit as st
from sklearn.metrics import classification_report
from pandas.core.algorithms import mode
 

smoking_df = pd.read_csv('smoking.csv (1).zip')
smoking_df

smoking_df.info()

smoking_df.describe()

smoking_df.hist(bins = 40, figsize = (40 ,40))
plt.show()

smoking_df.tail(10)

smoking_df.head(10)

smoking_df.corr()
 
plt.figure(figsize = (18,18))
sns.heatmap(smoking_df.corr(), annot = True)
plt.show()

plt.figure(figsize = (10 , 6))
plt.title ('comparison')
plt.scatter (smoking_df['LDL'],smoking_df['Cholesterol'])
plt.xlabel('LDL')
plt.ylabel('cholesterol') 
plt.show()

smoking_df['BMI']= smoking_df ['weight(kg)'] / (smoking_df['height(cm)'] / 100 ) **2

# BMI = weight(kg) / height(m) * height(m)	
# BMI < 18.5  UNDERWEIGHT
# 18.5 <= BMI <= 24.5  HEALTYHY WEIGHT
# 24.5 < BMI <= 29.9 OVERWEIGHT
# BMI >= 30 OBESITY

plt.figure(figsize=(60,60))
list_columns=['age','height(cm)', 'weight(kg)', 'waist(cm)',
       'eyesight(left)', 'eyesight(right)', 'hearing(left)', 'hearing(right)',
       'systolic', 'relaxation', 'fasting blood sugar', 'Cholesterol',
       'triglyceride', 'HDL', 'LDL', 'dental caries', 'hemoglobin', 'Urine protein',
       'serum creatinine', 'AST', 'ALT', 'Gtp','BMI']
for i in range(len(list_columns)):
  plt.subplot(13,2,i+1)
  plt.title(list_columns[i]) 
  plt.violinplot(smoking_df[list_columns[i]])
plt.show()

smoking_df.nunique().sort_values()

smoking_df['gender'].value_counts().plot.pie(explode=[0,0.3],autopct=lambda x: str(x)[:4] + '%', shadow =True)
plt.show()

smoking_df['age'].max()

smoking_df['age'].mean()

smoking_df['age'].min()

smoking_df['smoking'].value_counts().plot.pie(explode=[0,0.3],autopct=lambda x: str(x)[:4] + '%', shadow =True)
plt.show()

smoking_df.loc[smoking_df['smoking'] == 1]

smoking_df['age'].value_counts()

smoking_df['smoking'].sum()

_x = smoking_df['age'].value_counts().index
_y = smoking_df['age'].value_counts()

plt.figure(figsize= (10,6))
plt.bar(_x,_y)
plt.title('smoking_candidates')
plt.xlabel('age')
plt.ylabel(' number_of_smoking_condidates')
plt.show()

A = smoking_df[smoking_df['smoking']==1]
plt.figure(figsize=(10,6))
sns.distplot(A['age'],color = 'red')
plt.title("Age Distributuion of Smokers")
plt.show()

oldest_candidates = smoking_df['age'].sort_values(ascending = False)

oldest_candidates[:15].index

youngest_condidates=smoking_df['age'].sort_values(ascending = True)

youngest_condidates[:10].index

candidates_groupby_age_smoking = smoking_df.groupby('smoking')['age'].mean()
candidates_groupby_age_smoking

candidates_groupby =smoking_df.groupby(['gender','smoking'])['age','LDL','Cholesterol'].mean()
candidates_groupby

candidates_groupby.plot(kind='bar', title='comparing',figsize=(15, 6))
plt.show()

tartar_smoking_people =smoking_df.groupby('tartar')['smoking'].sum()
tartar_smoking_people

dental_caries_smoking_people =smoking_df.groupby('dental caries')['smoking'].sum()
dental_caries_smoking_people

dental_caries_smoking_people.plot(kind='bar', title='comparing_dental_caries_smoking_people',figsize=(10, 6))
plt.show()


