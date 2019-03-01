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
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 


def main():
    Data = pd.read_csv("data/PCA_transformed_data.csv", header=None)
    Labels=pd.read_csv("data/labels.csv")
    print(Data.head())
    print(Labels.head())
    #split dataset in training and testing
    X_train, X_test, y_train, y_test = train_test_split(Data, Labels[["Class"]], random_state = 0) 
    
    #decision tree classifier
    dtree_model = DecisionTreeClassifier().fit(X_train, y_train) 
    dtree_predictions = dtree_model.predict(X_test) 
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, dtree_predictions)
    print("Decision tree")
    print(cm)




main()