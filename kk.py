import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px

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

fig = px.scatter(smoking_df, x="LDL", y="Cholesterol", color="smoking", title=
        "LDL/Cholestrol")
fig.show()

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

# change datatype column gender 
le = LabelEncoder()
le.fit(smoking_df["gender"])
smoking_df["gender"]=le.transform(smoking_df["gender"])  

# change datatype column oral 
l = LabelEncoder()
l.fit(smoking_df["oral"])
smoking_df["oral"]=l.transform(smoking_df["oral"])

# change datatype column tartar 
a = LabelEncoder()
a.fit(smoking_df["tartar"])
smoking_df["tartar"]=a.transform(smoking_df["tartar"])

sns.pairplot(smoking_df, hue = 'smoking', vars = ['fasting blood sugar', 'hemoglobin', 'Gtp','Cholesterol'] )

y_smoking_df = smoking_df['smoking']
x_smoking_df = smoking_df.drop( 'smoking' , axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x_smoking_df, y_smoking_df, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
sum(y_pred == y_test) / len(y_pred)
accuracy_score(y_test, y_pred)

# GaussianNB Report
print(classification_report(y_test, y_pred))

kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
i = 0
for train_index, test_index in kf.split(x_smoking_df,y_smoking_df):
    i += 1
    model = GaussianNB()
    x_train, y_train = x_smoking_df.iloc[train_index], y_smoking_df.iloc[train_index]
    x_test, y_test =  x_smoking_df.iloc[test_index], y_smoking_df.iloc[test_index]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    accuracies.append(accuracy)
    print(i, ') accuracy = ', accuracy)

print('Mean accuracy: ', np.array(accuracies).mean())

cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

plt.figure(figsize = (10,6))
sns.heatmap(cf_matrix,fmt='.0f' , annot = True)
plt.title('confusion_matrix_heatmap_GaussianNB')
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.show()

pca = PCA(n_components=2)
x_smoking_df_pca = pca.fit(x_smoking_df).transform(x_smoking_df)
pca.explained_variance_ratio_

#classification Random Forest
y_smoking_df = smoking_df['smoking']
x_smoking_df = smoking_df.drop( 'smoking' , axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x_smoking_df, y_smoking_df, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)

plt.figure(figsize = (14,6))
model = RandomForestClassifier()
model.fit(x_train, y_train)
sort = model.feature_importances_.argsort()
plt.barh(smoking_df.columns[sort], model.feature_importances_[sort])
plt.xlabel("Feature Importance")

# RandomForest Report
print(classification_report(y_test, y_pred))

cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

plt.figure(figsize = (10,6))
sns.heatmap(cf_matrix,fmt='.0f' , annot = True)
plt.title('confusion_matrix_heatmap_RandomForestClassifier')
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.show()

kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
i = 0
for train_index, test_index in kf.split(x_smoking_df,y_smoking_df):
    i += 1
    model = RandomForestClassifier(random_state=42)
    x_train, y_train = x_smoking_df.iloc[train_index], y_smoking_df.iloc[train_index]
    x_test, y_test =  x_smoking_df.iloc[test_index], y_smoking_df.iloc[test_index]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    accuracies.append(accuracy)
    print(i, ') accuracy = ', accuracy)

print('Mean accuracy: ', np.array(accuracies).mean())

def train_model(x , y , model , random_state = 42 , test_size= 0.2):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state )
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  print ('accuracy = ' , accuracy_score(y_pred, y_test ))


x = smoking_df[['fasting blood sugar', 'dental caries' , 'Gtp' , 'Cholesterol']]
y = smoking_df [ 'smoking']
model = RandomForestClassifier(random_state=42)
train_model(x,y,model)

# Random forest (4 columns only) Report
print(classification_report(y_test, y_pred))

kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
i = 0
for train_index, test_index in kf.split(x,y):
    i += 1
    model = RandomForestClassifier(random_state=42)
    x_train, y_train = x.iloc[train_index], y.iloc[train_index]
    x_test, y_test = x.iloc[test_index], y.iloc[test_index]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    accuracies.append(accuracy)
    print(i, ') accuracy = ', accuracy)

print('Mean accuracy: ', np.array(accuracies).mean())

#clasification 3 Decision Tree
y_smoking_df = smoking_df['smoking']
x_smoking_df = smoking_df.drop( 'smoking' , axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x_smoking_df, y_smoking_df, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
accuracy_score(y_test, y_pred)


plt.figure(figsize = (14,6))
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
model.feature_importances_.argsort()
model.feature_importances_[sort]
plt.barh(smoking_df.columns[sort], model.feature_importances_[sort])
plt.xlabel("Feature Importance")

# Decision Tree Report
print(classification_report(y_test, y_pred))

kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
i = 0
for train_index, test_index in kf.split(x_smoking_df,y_smoking_df):
    i += 1
    model = DecisionTreeClassifier(random_state=42)
    x_train, y_train = x_smoking_df.iloc[train_index], y_smoking_df.iloc[train_index]
    x_test, y_test =  x_smoking_df.iloc[test_index], y_smoking_df.iloc[test_index]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    accuracies.append(accuracy)
    print(i, ') accuracy = ', accuracy)

print('Mean accuracy: ', np.array(accuracies).mean())

cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

plt.figure(figsize = (10,6))
sns.heatmap(cf_matrix,fmt='.0f' , annot = True)
plt.title('confusion_matrix_heatmap_DecisionTreeClassifier')
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.show()

