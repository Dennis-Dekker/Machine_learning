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
import seaborn

#import data from .csv files
allFiles = glob.glob("data/data*.csv")
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None)
    list_.append(df)

#expression data
df_data = pd.concat(list_, axis = 0, ignore_index = True)
#labels frame
df_data_labels = pd.read_csv("data/labels.csv")



finalDf = pd.concat([df_data[['Unnamed: 0']]], axis = 1)
#remove Unnamed columns from the dataset
x = df_data.drop(['Unnamed: 0','Unnamed: 0.1'], 1)
#remove Unnamed column from the labels
y = df_data['Unnamed: 0']

#PCA selecting the first two components.
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

#add class information to the output of PCA
finalDf = pd.concat([principalDf, df_data_labels[["Class"]]],axis=1)

print(pca.explained_variance_)

plotting
fig_PCA = plt.scatter(finalDf.iloc[:,0], finalDf.iloc[:,1], s=4, alpha=0.3, cmap='RdYlBu_r')
fig_PCA.figure.savefig('images/PCA_test.png')

#plot by class

pca_color=sns.pairplot(x_vars=["principal component 1"], y_vars=["principal component 2"], data=finalDf, hue="Class", size=5)
pca_color.savefig("images/PCA_color.png")
print("image saved")

