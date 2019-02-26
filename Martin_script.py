import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
%matplotlib inline


df_data = pd.read_csv("data.csv")
df_data_labels = pd.read_csv("labels.csv")

df_data_labels.head()

finalDf = pd.concat([, df_data[['Unnamed: 0']]], axis = 1)


#df_names=list(df_data.columns.values)
x = df_data.drop('Unnamed: 0', 1)
y = df_data['Unnamed: 0']

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


#
finalDf = pd.concat([principalDf, df_data[['Unnamed: 0']]], axis = 1)
col1=finalDf.iloc[:,0]
col1.head()


#PCA
fig_PCA = plt.scatter(finalDf.iloc[:,0], finalDf.iloc[:,1], s=4, alpha=0.3, cmap='RdYlBu_r')
fig_PCA.figure.savefig('PCA_test.png')
