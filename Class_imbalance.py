import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from imblearn.over_sampling import SMOTE

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
    df_data = pd.concat(list_, axis = 0, ignore_index = False)
    #labels frame
    df_data_labels = pd.read_csv("data/raw_labels.csv", index_col=0)
    

    return df_data


def main():
    df = load_data()
    
    class_counts = df["cancer_type"].value_counts()
    
    y_pos = np.arange(len(class_counts.index))

    plt.bar(y_pos,height = list(class_counts), color=['black', 'red', 'green', 'blue', 'cyan'])
    plt.xticks(y_pos, list(class_counts.index))
    plt.savefig("images/Class_imbalance.png")

if __name__ == '__main__':
    main()
