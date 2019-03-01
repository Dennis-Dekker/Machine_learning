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

def main():
    Data = pd.read_csv("data/PCA_transformed_data.csv", header=None)
    Labels=pd.read_csv("data/labels.csv")
    print(Data.head())
    print(Labels.head())

    X_train, X_test, y_train, y_test = train_test_split(Data, Labels, random_state = 0) 
    

main()