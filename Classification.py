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
from mlxtend.plotting import plot_decision_regions




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

    return cm, accuracy, svm_model_linear

def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    # training a KNN classifier 
    knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train.values.ravel()) 
    
    # accuracy on X_test 
    accuracy = knn.score(X_test, y_test)     
    # creating a confusion matrix 
    knn_predictions = knn.predict(X_test)  
    cm = confusion_matrix(y_test, knn_predictions)
    return cm, accuracy


def plot_boundaries(svm_model, X,y):
    # Plot Decision Region using mlxtend's  plotting function
    y.update(y.apply(pd.to_numeric))
    print(y)
    plot_decision_regions(X=X.values, y=y.astype(np.integer), clf=svm_model, legend=2)
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title('SVM Decision Region Boundary', size=16)
    plt.show()


def main():
    Data = pd.read_csv("data/PCA_transformed_data.csv", header=None)
    Labels=pd.read_csv("data/labels.csv")
    #selecting only 2 components
    Data=Data.iloc[:,1:3]
    #split dataset in training and testing
    X_train, X_test, y_train, y_test = train_test_split(Data, Labels[["Class"]], random_state = 0) 
    
    #decision tree classifier
    
    print("Decision tree")
    cm_dt=decision_tree(X_train, X_test, y_train, y_test)
    print(cm_dt)
    #SVM
    print("SVM")
    cm_svm, accuracy_svm, svm_model=support_vector_machine(X_train, X_test, y_train, y_test)
    print(cm_svm)
    print(accuracy_svm)
    plot_boundaries(svm_model, X_train, y_train)
    #KNN
    print("KNN")
    cm_knn, accuracy_knn=k_nearest_neighbors(X_train, X_test, y_train, y_test)
    print(cm_knn)
    print(accuracy_knn)

if __name__ == '__main__':
    main()
