import numpy as np
import pandas as pd
import seaborn as sns

import sklearn 
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

#####Use axes3d!! for the 3D plot    
from mpl_toolkits.mplot3d import axes3d, Axes3D 


#Set data to use in PCA function
x = df_data_label.drop(['Unnamed: 0','Class'], 1)  
y = df_data_label['Class']  


#Run PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
#Store output PCA in dataframe
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])
      
#Add labels
finalDf = pd.concat([principalDf, df_data_label[['Class']]], axis = 1)


#Plot
fig = plt.figure(figsize=(8, 6))


ax = Axes3D(fig)


#col= finalDf["Class"].map({'PRAD':'b','LUAD':'y','BRCA':'g','KIRC':'r','COAD':'p'})
xs = finalDf['principal component 1']
ys = finalDf['principal component 2']
zs = finalDf['principal component 3']
ax.scatter(xs, ys, zs ,s=50, alpha=0.6, edgecolors='w')



ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('pc3')

fig.savefig('3D_PCA_test.png')
