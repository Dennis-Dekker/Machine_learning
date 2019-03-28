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
from sklearn.model_selection import GridSearchCV,cross_val_score,KFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier



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


# def find_best_method(X_train,y_train, param ):
#     #define the possible hyperparameters
#     clf = GridSearchCV(estimator=SVC(), param_grid=param)
#     clf.fit(X_train, y_train)
#     # Show the best value for C
#     print(clf.best_params_)
#     #print(cross_val_score(clf, X_train, y_train))
#     #sys.exit("doei")
#     return clf.best_params_

def nested_CV(X_train,y_train, estimator, param):
    state=1
    out_scores=[]
    in_winner_param=[]
    out_cv = KFold(n_splits=7, shuffle=True, random_state=state)
    for i, (index_train_out, index_test_out) in enumerate(out_cv.split(X_train)):
        X_train_out, X_test_out = X_train[index_train_out], X_train[index_test_out]
        y_train_out, y_test_out = y_train[index_train_out], y_train[index_test_out]

        in_cv =KFold(n_splits=3, shuffle=True, random_state=state)
        #inner loop for hyperparameters tuning
        GSCV=GridSearchCV(estimator=estimator, param_grid=param, cv=in_cv, verbose=2,n_jobs=-1)
        #train a model with each set of parameters
        GSCV.fit(X_train_out, y_train_out)
        #predict using the best set of hyperparameters
        prediction=GSCV.predict(X_test_out)
        in_winner_param.append(GSCV.best_params_)
        out_scores.append(accuracy_score(prediction, y_test_out))
        print("\nBest inner accuracy of fold "+str(i+1)+": "+str(GSCV.best_score_)+"\n")

    for i in zip(in_winner_param, out_scores):
        print(i)
    print("Mean of outer loop: "+str(np.mean(out_scores))+" std: "+str(np.std(out_scores)))
    return out_scores


def support_vector_machine(X_train, X_test, y_train, y_test):


     # training a linear SVM classifier
    linear=SVC(kernel = "rbf", C = 1, gamma=0.001)
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
    knn=KNeighborsClassifier(n_neighbors = 15,weights="distance")
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

def naive_bayes(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    cm=confusion_matrix(y_test, y_pred)
    return cm, accuracy_score(y_test,y_pred), gnb

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
    fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=False,colorbar=True, show_normed=True)
    plt.title("Confusion matrix "+method)
    x = ["LUAD", "BRCA" ,"KIRC", "COAD" ,"OV" ,"READ","LUSC", "GBM" ,"UCEC", "HNSC"]
    y= ["LUAD", "BRCA" ,"KIRC", "COAD" ,"OV" ,"READ","LUSC", "GBM" ,"UCEC", "HNSC" ]
    lx=list(range(10))
    ly=list(range(10))
    ax.set_xticks(lx)
    ax.set_xticklabels(x)
    ax.set_yticks(ly)
    ax.set_yticklabels(y)
    plt.show()

def organize_data():
    #Data = np.loadtxt("data/PCA_transformed_raw_data.csv", delimiter=",")
    #Labels=pd.read_csv("data/raw_labels.csv",index_col=0)
    Data=pd.read_csv("data/PCA_transformed_raw_data.csv")
    pca_color=sns.pairplot(x_vars=["PC2"], y_vars=["PC3"], data=Data, hue="cancer_type", height=5)
    path_PCA_figure_color = "images/PCA_color_pc2vs3.png"
    pca_color.savefig(path_PCA_figure_color)
    print("Image saved to: " + path_PCA_figure_color)
    Data=Data.values #convert from pandas to numpy
    Labels=Data[:,10]
    Data=Data[:,0:4]
    #convert labels from string to numbers
    #Labels, uniques = pd.factorize(Labels.iloc[:, 0].tolist())
    Labels, uniques = pd.factorize(Labels)
    plt.clf()
    #TO VISUALIZE the whole FEATURE SPACE, remove comments below
    # labels, uniques = pd.factorize(Labels.iloc[:, 1].tolist())
    # plt.scatter(Data.as_matrix()[:,0], Data.as_matrix()[:,1], s=4, alpha=0.3, c=labels, cmap='RdYlBu_r')
    # plt.show()

    #split dataset in training and testing
    X_train, X_test, y_train, y_test = train_test_split(Data, Labels, random_state = 1,test_size=0.25)
    #plot before oversampling
    
    np.savetxt("data/Test_pca_raw.csv", X_test, delimiter=",")
    np.savetxt("data/Label_test_pca_raw.csv", y_test.astype(int), delimiter=",")
    # plt.scatter(X_train[:,0], X_train[:,1], s=4, alpha=1, c=y_train)
    # plt.show()
    #oversampling of the training set
    #uniques, counts=np.unique(y_train, return_counts=True)
    #print(dict(zip(uniques,counts)))

    smote = SMOTE("not majority")
    
    # #Replace X_train by X_sm_train and y_train by y_sm_train in Class_imbalance.py
    X_sm_train, y_sm_train = smote.fit_sample(X_train,y_train)
    #plot after oversampling
    plt.scatter(X_sm_train[:,0], X_sm_train[:,1], s=4, alpha=1, c=y_sm_train,  cmap='RdYlBu_r')
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()
    plt.clf()
    plt.scatter(X_train[:,0], X_train[:,1], s=4, alpha=1, c=y_train,  cmap='RdYlBu_r')
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()
    #uniques, counts=np.unique(y_sm_train, return_counts=True)
    #print(dict(zip(uniques,counts)))
    #update the training set with oversampled one
    # X_train=X_sm_train
    # y_train=y_sm_train
    #save smote training set
    np.savetxt("data/Train_pca_raw_smote.csv", X_sm_train, delimiter=",")
    np.savetxt("data/Label_train_pca_raw_smote.csv", y_sm_train.astype(int), delimiter=",")
    #save normal training set
    np.savetxt("data/Train_pca_raw.csv", X_train, delimiter=",")
    np.savetxt("data/Label_train_pca_raw.csv", y_train.astype(int), delimiter=",")
    return X_train, X_test, y_train,y_test

def load_train_test():
    X_train=pd.read_csv("data/Train_pca_raw.csv", header=None)
    X_test=pd.read_csv("data/Test_pca_raw.csv",header=None)
    y_train=pd.read_csv("data/Label_train_pca_raw.csv",header=None)
    y_test=pd.read_csv("data/Label_test_pca_raw.csv",header=None)
    X_train_smote=pd.read_csv("data/Train_pca_raw_smote.csv",header=None)
    y_train_smote=pd.read_csv("data/Label_train_pca_raw_smote.csv",header=None)
    X_train=X_train.values
    X_test=X_test.values
    y_train=y_train.values.ravel()
    y_test=y_test.values.ravel()
    X_train_smote=X_train_smote.values
    y_train_smote=y_train_smote.values.ravel()
    return X_train, X_test, y_train,y_test,X_train_smote, y_train_smote

def dummy_classification(X_train, X_test, y_train,y_test):
    print(X_train.shape,y_train.shape )
    dummy = DummyClassifier(strategy='stratified').fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    print("Dummy_accuracy:", accuracy_score(y_test, dummy_pred))    

def main():
    
    #call "organize_data" to modify the train/test split
    #X_train, X_test, y_train,y_test=organize_data()
    X_train, X_test, y_train,y_test,X_train_smote, y_train_smote = load_train_test()
    
    #SMALL dataset
    # Data = np.loadtxt("data/PCA_transformed_data.csv", delimiter=",")
    # Labels=pd.read_csv("data/labels.csv",index_col=0)
    # Labels=Labels["Class"].values.ravel()
    # Labels, uniques = pd.factorize(Labels)

    # print(Data)
    # print(Labels)
    # plt.scatter(Data[:,0], Data[:,1], s=4, alpha=1, c=Labels)
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.legend()
    # plt.show()
    # plt.clf()
    # sys.exit("doei")
    # #decision tree classifier
    print("Dummy classifier without SMOTE: ")
    dummy_classification(X_train, X_test, y_train,y_test)
    print("Dummy classifier with SMOTE: ")
    dummy_classification(X_train_smote, X_test, y_train_smote, y_test)
    # cm_dt, acc_dt,tree=decision_tree(X_train, X_test, y_train, y_test)
    # plot_accuracy("decision tree", cm_dt, acc_dt)
    #SVM

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
    {'kernel': ['linear'], 'C': [1, 5]}
    ]
    #svm_dist=nested_CV(X_train,y_train, SVC(), tuned_parameters)
    #print("DONE without SMOTE")
    #svm_dist=nested_CV(X_train_smote,y_train_smote, SVC(), tuned_parameters)
    #print("DONE with SMOTE")
    #cm_svm, accuracy_svm, svm_model =support_vector_machine(X_train, X_test, y_train, y_test)
    #plot_accuracy("SVM", cm_svm, accuracy_svm)
    #KNN

    grid_param = {
    'n_neighbors': [7, 9 , 15 , 17, 25],
    'weights':['uniform','distance']
    }
    #knn_dist=nested_CV(X_train, y_train, KNeighborsClassifier(),grid_param)
    #print("DONE without SMOTE")
    #knn_dist=nested_CV(X_train_smote, y_train_smote, KNeighborsClassifier(),grid_param)
    #print("DONE with SMOTE")
    #sys.exit("doei")

    #cm_knn, accuracy_knn, knn=k_nearest_neighbors(X_train, X_test, y_train, y_test)
    #plot_accuracy("K-NN", cm_knn, accuracy_knn)

    #RF - random forest
    grid_param = {
    'n_estimators': [100, 250, 500, 750, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
    }
    #rf_dist=nested_CV(X_train, y_train, RandomForestClassifier(), grid_param)
    #print("after without SMOTE")
    #rf_dist=nested_CV(X_train_smote, y_train_smote, RandomForestClassifier(), grid_param)
    #print("after with SMOTE")

    #cm_rf,accuracy_rf, rf= random_forest(X_train, X_test, y_train, y_test)
    #plot_accuracy("Random forest", cm_rf, accuracy_rf)

    sys.exit("doei")
    # plot_boundaries(svm_model,tree,knn, X_test, y_test)

    # NAIVE Bayes
    grid_param = {
    'var_smoothing': [1e-9, 1e-10,1e-7],
    }
    
    nb_dist=nested_CV(X_train, y_train, GaussianNB(), grid_param)
    print("after without SMOTE")
    nb_dist=nested_CV(X_train_smote, y_train_smote, GaussianNB(), grid_param)
    print("after with SMOTE")
    # cm_nb,accuracy_nb, nb= naive_bayes(X_train, X_test, y_train, y_test)
    # plot_accuracy("Naive bayes", cm_nb, accuracy_nb)

    #MLP classifier (NN)
    # grid_param = {
    # 'activation': ['logistic', 'relu','tanh'],
    # 'alpha':[0.001,0.0001,0.00001],
    # # 'learning_rate': ["constant", "invscaling", "adaptive"],
    # 'hidden_layer_sizes': [(3,3,1),(3,2,1)],
    # 'max_iter':[700]
    # }
    # NN_dist=nested_CV(X_train, y_train, MLPClassifier(), grid_param)


if __name__ == '__main__':
    main()
