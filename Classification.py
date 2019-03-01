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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 



def decision_tree(X_train, X_test, y_train, y_test):
    dtree_model = DecisionTreeClassifier().fit(X_train, y_train) 
    dtree_predictions = dtree_model.predict(X_test) 
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, dtree_predictions)
    return cm


def support_vector_machine(X_train, X_test, y_train, y_test):
     # training a linear SVM classifier 
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train.values.ravel()) 
    svm_predictions = svm_model_linear.predict(X_test) 
    
    # model accuracy for X_test   
    accuracy = svm_model_linear.score(X_test, y_test) 
    
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, svm_predictions)

    return cm, accuracy

def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    # training a KNN classifier 
    knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
    
    # accuracy on X_test 
    accuracy = knn.score(X_test, y_test)     
    # creating a confusion matrix 
    knn_predictions = knn.predict(X_test)  
    cm = confusion_matrix(y_test, knn_predictions)
    return cm, accuracy

def main():
    Data = pd.read_csv("data/PCA_transformed_data.csv", header=None)
    Labels=pd.read_csv("data/labels.csv")
    print(Data.head())
    print(Labels.head())
    #split dataset in training and testing
    X_train, X_test, y_train, y_test = train_test_split(Data, Labels[["Class"]], random_state = 0) 
    
    #decision tree classifier
    
    print("Decision tree")
    cm_dt=decision_tree(X_train, X_test, y_train, y_test)
    print(cm_dt)
    #SVM
    print("SVM")
    cm_svm, accuracy_svm=support_vector_machine(X_train, X_test, y_train, y_test)
    print(cm_svm)
    print(accuracy_svm)
    #KNN
    cm_knn, accuracy_knn=k_nearest_neighbors(X_train, X_test, y_train, y_test)
    print(cm_knn)
    print(accuracy_knn)

if __name__ == '__main__':
    main()
