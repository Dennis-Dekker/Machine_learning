import numpy as np
import pandas as pd
import glob
import sklearn
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

#import data

allFiles = glob.glob("data/data*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None)
    list_.append(df)

df_data = pd.concat(list_, axis = 0, ignore_index = True)
print(df_data.head())
df_data_labels = pd.read_csv("data/labels.csv")

print(df_data_labels.head())


finalDf = pd.concat([df_data[['Unnamed: 0']]], axis = 1)
#df_names=list(df_data.columns.values)
x = df_data.drop(['Unnamed: 0','Unnamed: 0.1'], 1)
y = df_data['Unnamed: 0']

print(x.head())
#PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


finalDf = pd.concat([principalDf, df_data_labels[["Class"]]],axis=1)
col1=finalDf.iloc[:,0]
col1.head()


#plotting
fig_PCA = plt.scatter(finalDf.iloc[:,0], finalDf.iloc[:,1], s=4, alpha=0.3, cmap='RdYlBu_r')
fig_PCA.figure.savefig('PCA_test.png')

#plot by class
df_data_labels.iloc[:,1] = pd.factorize(df_data_labels.iloc[:,1])
print(df_data_labels.head())
##################################
sns.pairplot(x_vars=["principal component 1"], y_vars=["principal component 2"], data=finalDf, hue="Class", size=5)

###################################
labels_list=df_data_labels[["Class"]]
print(labels_list)
colors=["red", "blue","green","yellow","purple"]
fig_PCA = plt.scatter(finalDf.iloc[:,0], finalDf.iloc[:,1], c=labels_list, s=4, alpha=0.3, cmap=matplotlib.colors.ListedColormap(colors))
fig_PCA.figure.savefig('PCA_colors.png')
