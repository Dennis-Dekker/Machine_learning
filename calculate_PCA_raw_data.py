#!/usr/bin/python

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
from mpl_toolkits.mplot3d import axes3d, Axes3D 

def load_data():
    """Load data from .csv files.
    
    df_data: dataframe of gene expression data (merged from splitted .csv files).
    df_data_labels: dataframe with sample annotation.
    """
    #import data from .csv files
    allFiles = glob.glob("data/raw_data*.csv")
    list_ = []
    print("--- Loading datasets ---")
    print("Found " + str(len(allFiles)) + " files.")
    print("Loading files:")
    for file_ in allFiles:
        print("\t" + file_)
        df = pd.read_csv(file_,index_col=0)
        list_.append(df)

    #expression data
    df_data = pd.concat(list_, axis = 0, ignore_index = False)
    print("Amount of loaded samples:\t\t\t" + str(df_data.shape[0]))
    #labels frame
    df_data_labels = pd.read_csv("data/raw_labels.csv", index_col=0)
    
    df_data = pd.concat([df_data_labels,df_data], axis = 1, join = "inner")
    print("Amount of samples left after annotating:\t" + str(df_data.shape[0]))
    
    return df_data

def process_data_frame(df_data):
    """Remove unwanted columns from dataframe.
    """
    
    print("Applying log2 transformation...")
    #remove Unnamed columns from the dataset
    x = df_data.drop(["cancer_type", "gender", "bcr_patient_uuid", "bcr_patient_barcode", "patient_id", "age_at_initial_pathologic_diagnosis"], axis = 1)
    #remove Unnamed column from the labels
    # y = df_data.index
    x += 1
    x = x.applymap(np.log2)
    
    return x

def calculate_PCA(x, df_data):
    """Calculate n principal components
    
    finalDf: dataframe of pca with labels.
    """
    
    print("Calculating principal components.")
    #PCA selecting the first two components.
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                  , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
    principalDf = principalDf.set_index(df_data.index.values)

    #add class information to the output of PCA
    finalDf = pd.concat([principalDf, df_data[["cancer_type"]]],ignore_index=False, axis = 1)
    
    # 10 components
    pca = PCA(n_components=10)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                  , columns = ['PC1', 'PC2', 'PC3','PC4', 'PC5', 'PC6','PC7', 'PC8', 'PC9','PC10'])
    principalDf = principalDf.set_index(df_data.index.values)
    #uncomment next line for debugging
    #print(pca.explained_variance_)
    finalDf_10 = pd.concat([principalDf, df_data[["cancer_type"]]],ignore_index=False, axis = 1)

    return pca, finalDf, finalDf_10
    
def plot_PCA(finalDf, pca):
    """Plot PCA's
    
    First plot of normal PCA.
    Second plot PCA colored by Class.
    """ 
    
    #plotting
    fig_PCA = plt.scatter(finalDf.iloc[:,0], finalDf.iloc[:,1], s=4, alpha=0.3, cmap='RdYlBu_r')
    path_PCA_figure = 'images/PCA_test.png'
    plt.xlabel("PC1 (" + str(round(pca.explained_variance_ratio_[0]*100, 1)) + "%)")
    plt.ylabel("PC2 (" + str(round(pca.explained_variance_ratio_[1]*100, 1)) + "%)")
    fig_PCA.figure.savefig(path_PCA_figure)
    print("Image saved to: " + path_PCA_figure)

    #plot by class
    pca_color=sns.pairplot(x_vars=["principal component 1"], y_vars=["principal component 2"], data=finalDf, hue="cancer_type", height=5)
    path_PCA_figure_color = "images/PCA_color.png"
    pca_color.set(xlabel = "PC1 (" + str(round(pca.explained_variance_ratio_[0]*100, 1)) + "%)")
    pca_color.set(ylabel = "PC2 (" + str(round(pca.explained_variance_ratio_[1]*100, 1)) + "%)")
    pca_color.savefig(path_PCA_figure_color)
    print("Image saved to: " + path_PCA_figure_color)
    
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    y = pd.get_dummies(np.array(finalDf.loc[:,"cancer_type"])).values.argmax(1)
    ax.scatter(finalDf.iloc[:, 0], finalDf.iloc[:, 1], finalDf.iloc[:, 2],c=y, cmap='RdYlBu_r',edgecolor='k')
    plt.show()

def calculate_amount_PCs(x):
    pca_trafo = PCA().fit(x)
    print(pca_trafo.explained_variance_ratio_[0:10])
    plt.plot(list(range(1,11)), pca_trafo.explained_variance_ratio_[0:10]*100, '--o', label="Explained variance")
    plt.plot(list(range(1,11)), pca_trafo.explained_variance_ratio_.cumsum()[0:10]*100, '--o', label="Cumulative explained variance")
    plt.ylabel('Explained variance (%)')
    plt.xlabel('Principal components')
    plt.legend()
    plt.savefig("images/test_explained_variance.png")
    plt.show()
    return

def export_PCA_scores(pca):
    pca_save_file = "data/PCA_transformed_raw_data.csv"
    #np.savetxt(pca_save_file, pca, delimiter=",")
    pca.to_csv(pca_save_file, index=False)
    print("First 10 PCs saved in:\t" + pca_save_file)
    return

def main():
    """Main function.
    """
    #load data
    df_data = load_data()    
    
    # print(df_data.describe())
    #process dataframe 
    x = process_data_frame(df_data)
    
    #calculate PCA 
    pca, finalDf, all_compon = calculate_PCA(x, df_data)
    
    fig_PCA = plt.scatter(finalDf.iloc[:,0], finalDf.iloc[:,1], s=4, alpha=0.3, cmap='RdYlBu_r')
    path_PCA_figure = 'images/PCA_test_with_outliers.png'
    plt.xlabel("PC1 (" + str(round(pca.explained_variance_ratio_[0]*100, 1)) + "%)")
    plt.ylabel("PC2 (" + str(round(pca.explained_variance_ratio_[1]*100, 1)) + "%)")
    fig_PCA.figure.savefig(path_PCA_figure)
    print("Image saved to: " + path_PCA_figure)
    plt.show()
    
    outliers = []
    for i in finalDf.index.values:
        if finalDf.loc[i,"principal component 1"] > 200:
            outliers.extend([i])
    
    x = x.drop(outliers,axis = 0)
    df_data = df_data.drop(outliers,axis = 0)
    
    #determine amount of PC's for analysis
    #TODO 
    calculate_amount_PCs(x)
    
    pca, finalDf, all_compon = calculate_PCA(x, df_data)
    #print(finalDf)

    #plot PCA 
    plot_PCA(finalDf, pca)
    export_PCA_scores(all_compon)
    print("--- DONE ---\n")

if __name__ == '__main__':
    main()
