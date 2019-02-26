import numpy as np
import pandas as pd
import glob
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

#import data

allFiles = glob.glob("data/*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)

df_data = pd.concat(list_, axis = 0, ignore_index = True)
df_data_labels = pd.read_csv("data/labels.csv")

df_data_labels.head()

#df_names=list(df_data.columns.values)
x = df_data.drop('Unnamed: 0', 1)
y = df_data['Unnamed: 0']

#PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


finalDf = pd.concat([principalDf, df_data[['Unnamed: 0']]], axis = 1)
col1=finalDf.iloc[:,0]
col1.head()


#plotting
fig_PCA = plt.scatter(finalDf.iloc[:,0], finalDf.iloc[:,1], s=4, alpha=0.3, cmap='RdYlBu_r')
fig_PCA.figure.savefig('PCA_test.png')
