import glob
from itertools import cycle
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from mlxtend.plotting import plot_confusion_matrix, plot_decision_regions
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, auc, confusion_matrix, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE



def decision_tree(X_train, X_test, y_train, y_test):
    tree=DecisionTreeClassifier()
    dtree_model = tree.fit(X_train, y_train) 
    dtree_predictions = dtree_model.predict(X_test) 
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, dtree_predictions)
    accuracy=accuracy_score(dtree_predictions,y_test)
    print("cohen-kappa: \n")
    print(cohen_kappa_score(y_test,dtree_predictions))
    print("F1 score: \n")
    print(f1_score(y_test, dtree_predictions, average='weighted'))
    return cm,accuracy, tree


def find_best_param_SVM(X_train,y_train):
    #define the possible hyperparameters
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}, {'kernel': ['linear'], 'C': [1, 10, 100]}]              
    clf = GridSearchCV(estimator=SVC(), param_grid=tuned_parameters)
    clf.fit(X_train, y_train)
    # Show the best value for C
    print(clf.best_params_)
    #print(cross_val_score(clf, X_train, y_train))
    #sys.exit("doei")
    return clf.best_params_


def support_vector_machine(X_train, X_test, y_train, y_test, param):


     # training a linear SVM classifier 
    linear=SVC(kernel = param["kernel"], C = param["C"], gamma=param["gamma"])
    svm_model_linear = linear.fit(X_train, y_train) 
    svm_predictions = svm_model_linear.predict(X_test) 
    
    # model accuracy for X_test   
    accuracy = svm_model_linear.score(X_test, y_test) 
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, svm_predictions)
    print("cohen-kappa: \n")
    print(cohen_kappa_score(y_test,svm_predictions))
    print("F1 score: \n")
    print(f1_score(y_test, svm_predictions, average='weighted'))
    return cm, accuracy, linear

def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    # training a KNN classifier 
    knn=KNeighborsClassifier(n_neighbors = 15)
    knn.fit(X_train, y_train)
    
    # accuracy on X_test 
    accuracy = knn.score(X_test, y_test)     
    # creating a confusion matrix 
    knn_predictions = knn.predict(X_test)  
    cm = confusion_matrix(y_test, knn_predictions)
    print("cohen-kappa: \n")
    print(cohen_kappa_score(y_test,knn_predictions))
    print("F1 score: \n")
    print(f1_score(y_test, knn_predictions, average='weighted'))
    return cm, accuracy, knn

def random_forest(X_train, X_test, y_train, y_test):

    grid_param = {  
    'n_estimators': [100, 250, 500, 750, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
    }
    rf_gs=GridSearchCV(RandomForestClassifier(), param_grid=grid_param)
    rf_gs.fit(X_train,y_train)
    print(rf_gs.best_params_)
    #train model
    #model.fit(X_train, y_train)
    predicted_labels = rf_gs.predict(X_test)
    cm=confusion_matrix(y_test, predicted_labels) 
    print("cohen-kappa: \n")
    print(cohen_kappa_score(y_test,predicted_labels))
    print("F1 score: \n")
    print(f1_score(y_test, predicted_labels, average='weighted'))
    return cm, accuracy_score(y_test, predicted_labels), rf_gs

def plot_boundaries(svm_model,tree, knn, X,y):
    # Plot Decision Region using mlxtend's  plotting function
    # Plotting decision regions
    plot_decision_regions(X=X, y=y, clf=svm_model, legend=2)
    plt.show()
    plot_decision_regions(X=X, y=y, clf=knn, legend=2)
    plt.show()
    plot_decision_regions(X=X, y=y, clf=tree, legend=2)
    plt.show()

def plot_accuracy(method, cm,accuracy):
    print("Classifier: "+method)
    print("Confusion matrix: \n")
    print(cm)
    print("Accuracy: "+str(accuracy)+"\n")
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    plt.title("Confusion matrix "+method)
    plt.show()
    
def main():
    #Data = np.loadtxt("data/PCA_transformed_raw_data.csv", delimiter=",")
    #Labels=pd.read_csv("data/raw_labels.csv",index_col=0)
    Data=pd.read_csv("data/PCA_transformed_raw_data.csv")
    Data=Data.values #convert from pandas to numpy
    Labels=Data[:,10]
    Data=Data[:,0:3]
    #convert labels from string to numbers
    #Labels, uniques = pd.factorize(Labels.iloc[:, 0].tolist())
    Labels, uniques = pd.factorize(Labels)

    #TO VISUALIZE the FEATURE SPACE, remove comments below
    # labels, uniques = pd.factorize(Labels.iloc[:, 1].tolist())
    # plt.scatter(Data.as_matrix()[:,0], Data.as_matrix()[:,1], s=4, alpha=0.3, c=labels, cmap='RdYlBu_r')
    # plt.show()
    #split dataset in training and testing
    X_train, X_test, y_train, y_test = train_test_split(Data, Labels, random_state = 1,test_size=0.2)     
    smote = SMOTE("minority")
   #Replace X_train by X_sm_train and y_train by y_sm_train in Class_imbalance.py
    X_sm_train, y_sm_train = smote.fit_sample(X_train,y_train)
    
    svm_best_param=find_best_param_SVM(X_train,y_train)
    #decision tree classifier
    cm_dt, acc_dt,tree=decision_tree(X_train, X_test, y_train, y_test)
    plot_accuracy("decision tree", cm_dt, acc_dt)   

    #SVM
    cm_svm, accuracy_svm, svm_model =support_vector_machine(X_train, X_test, y_train, y_test,svm_best_param)
    plot_accuracy("SVM", cm_svm, accuracy_svm)

    #KNN
   
    cm_knn, accuracy_knn, knn=k_nearest_neighbors(X_train, X_test, y_train, y_test)
    plot_accuracy("K-NN", cm_knn, accuracy_knn)

    #random forest
    cm_rf,accuracy_rf, rf= random_forest(X_train, X_test, y_train, y_test)
    plot_accuracy("Random forest", cm_rf, accuracy_rf)
    #plot_boundaries(svm_model,tree,knn, X_test, y_test)

    #roc plot --> takes a lot for svm, then is commented
    #roc_plot(X_train, y_train, X_test, y_test, svm_model, tree, knn)




if __name__ == '__main__':
    main()
