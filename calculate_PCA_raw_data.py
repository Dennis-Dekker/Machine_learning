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

def load_data():
    """Load data from .csv files.
    
    df_data: dataframe of gene expression data (merged from splitted .csv files).
    df_data_labels: dataframe with sample annotation.
    """
    #import data from .csv files
    allFiles = glob.glob("data/raw_data*.csv")
    list_ = []
    print("loading files:")
    for file_ in allFiles:
        print(file_)
        df = pd.read_csv(file_,index_col=0)
        list_.append(df)

    #expression data
    df_data = pd.concat(list_, axis = 0, ignore_index = True)
    #labels frame
    df_data_labels = pd.read_csv("data/raw_labels.csv", index_col=0)
    
    return df_data, df_data_labels

def process_data_frame(df_data):
    """Remove unwanted columns from dataframe.
    """
    
    #finalDf = pd.concat([df_data[['Unnamed: 0']]], axis = 1)
    #remove Unnamed columns from the dataset
    x = df_data
    #remove Unnamed column from the labels
    y = df_data.index
    
    return x, y

def calculate_PCA(x, y, df_data_labels):
    """Calculate n principal components
    
    finalDf: dataframe of pca with labels.
    """
    #PCA selecting the first two components.
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
                  , columns = ['principal component 1', 'principal component 2'])

    #add class information to the output of PCA
    finalDf = pd.concat([principalDf, df_data_labels[["cancer_type"]]],axis=1)
    
    # 10 components
    pca = PCA(n_components=10)
    principalComponents = pca.fit_transform(x)
    #uncomment next line for debugging
    #print(pca.explained_variance_)
    
    return pca, finalDf, principalComponents
    
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

def calculate_amount_PCs(x):
    pca_trafo = PCA().fit(x)
    print(pca_trafo.explained_variance_ratio_[0:10])
    plt.semilogy(pca_trafo.explained_variance_ratio_[0:10]*100, '--o')
    plt.semilogy(pca_trafo.explained_variance_ratio_.cumsum()[0:10]*100, '--o')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.savefig("images/test_explained_variance.png")
    return 2

def store_pca_result(all_component):
    np.savetxt("data/PCA_transformed_data.csv", all_component, delimiter=",")

def main():
    """Main function.
    """
    #load data
    df_data, df_data_labels = load_data()
    
    #process dataframe 
    x, y = process_data_frame(df_data)
    
    #determine amount of PC's for analysis
    #TODO 
    #n = calculate_amount_PCs(x)
    
    #calculate PCA 
    pca, finalDf, all_compon = calculate_PCA(x, y, df_data_labels)
    
    # store_pca_result(all_compon)
    #plot PCA 
    plot_PCA(finalDf, pca)

if __name__ == '__main__':
    main()
