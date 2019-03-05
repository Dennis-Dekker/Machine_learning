import numpy as np
import pandas as pd
import seaborn as sns

import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from pandas import *
from mlxtend.data import iris_data
from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import PrincipalComponentAnalysis


#Load data
df_data = pd.read_csv("data.csv")
df_labels = pd.read_csv("labels.csv")

#Concatenation both DF
df_data_label = pd.concat([df_data, df_labels],axis=1,sort=True)
x = df_data_label.drop(['Unnamed: 0','Class'], 1)  
#Calculate PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

#Get the Loadings
loadings = pca.components_

#Loadings from array to DF
df_loadings = pd.DataFrame(loadings)
#Genes in rows
df = df_loadings.transpose()
#name columns
df.columns=["Loadings PC1","Loadings PC2"]
#sorted by decreasing absolute value
df_sorted_PC1= df.iloc[(-df['Loadings PC1'].abs()).argsort()]
df_sorted_PC2 = df.iloc[(-df['Loadings PC2'].abs()).argsort()]
#Top 100 
df_to_plot_PC1= df_sorted_PC1.iloc[:100,:]
df_to_plot_PC2= df_sorted_PC2.iloc[:100,:]



#PLot
plt.scatter(*loadings, alpha=0.3, label="Loadings")

plt.title("Loading plot: Top 100 PC1")
plt.xlabel("Loadings on PC1")
plt.ylabel("Loadings on PC2")
#Top 100 loadings PC1
plt.scatter(x=df_to_plot_PC1["Loadings PC1"], y= df_to_plot_PC1["Loadings PC2"], marker='o',s=80, linewidths=1, facecolors="none", 
            edgecolors='r',label="Top 100 PC1")
#Top 100 loadings PC2
plt.scatter(x=df_to_plot_PC2["Loadings PC1"], y= df_to_plot_PC2["Loadings PC2"], marker='o',s=80, linewidths=1, facecolors="none", 
            edgecolors='g',label="Top 100 PC2")
plt.legend(loc='lower left');
path_PCA_figure_color = "/home/martin/Desktop/MasterVU/Machine_Learning/Project/Loading_plot.png"
plt.savefig(path_PCA_figure_color)

print("Image saved to: " + path_PCA_figure_color)

            
