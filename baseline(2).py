from sklearn import cross_validation, grid_search, metrics, ensemble
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import surprise
from surprise.model_selection import GridSearchCV


df = pd.read_csv('input/train.csv')
reader=rating_scale=(0,1)
df = df.sample(frac=0.01)

songs = pd.read_csv('input/songs.csv')
df = pd.merge(df, songs, on='song_id', how='left')
del songs

members = pd.read_csv('input/members.csv')
df = pd.merge(df, members, on='msno', how='left')
del members

for i in df.select_dtypes(include=['object']).columns:
    df[i][df[i].isnull()] = 'unknown'
df = df.fillna(value=0)

df = df[['msno', 'song_id', 'source_screen_name', 'source_type', 'target',
         'song_length', 'artist_name', 'composer', 'bd']]
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')

for col in df.select_dtypes(include=['category']).columns:
    df[col] = df[col].cat.codes

target = df.pop('target')
train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(df, target, test_size = 0.25)
del df

model1 =KNeighborsClassifier(n_neighbors=4)
model1.fit(train_data, train_labels)
predict_labels1 = model1.predict(test_data)
fpr, tpr, thresholds = metrics.roc_curve(test_labels, predict_labels1)
print("AUC=",metrics.auc(fpr,tpr))
#print(metrics.classification_report(test_labels, predict_labels1))

# Hello migam

