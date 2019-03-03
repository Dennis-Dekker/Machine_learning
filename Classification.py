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
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from mlxtend.plotting import plot_confusion_matrix


def roc_plot(X_train,y_train,X_test, y_test, linear, tree, knn):
    """
    This function should calculate the roc plot for every class of SVM classifier
    using every class ALl-against-all. Then make the average for the different classes
    and plot.  NOT WORKING: stucked at classifier.fit
    """
    #convert dataframe to list(numbers instead of labels)
    labels_test, uniques = pd.factorize(y_test.iloc[:, 0].tolist())
    labelst_train,uniques = pd.factorize(y_train.iloc[:,0].tolist())
    # Binarize the output
    y_test = label_binarize(labels_test, classes=[0, 1, 2, 3, 4])
    y_train=label_binarize(labelst_train,classes=[0, 1, 2, 3, 4])
    n_classes = 5
    print("begore oVr")
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True,
                                random_state=0, verbose=100))
    print("im here")
    print(y_train)
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    print("hello")
    #compute ROC curve for each class
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    for i in range(n_classes):
        fpr[i],tpr[i], _ = roc_curve(y_test[:,i],y_score[:,i])
        roc_auc[i]=auc(fpr[i],tpr[i])
    
    #compute micro-average ROC
    fpr["micro"],tpr["micro"],_=roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #plot class 2
    print("hello 2")
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def decision_tree(X_train, X_test, y_train, y_test):
    tree=DecisionTreeClassifier()
    dtree_model = tree.fit(X_train, y_train) 
    dtree_predictions = dtree_model.predict(X_test) 
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, dtree_predictions)
    accuracy=accuracy_score(dtree_predictions,y_test)
    return cm,accuracy, tree


def support_vector_machine(X_train, X_test, y_train, y_test):
     # training a linear SVM classifier 
    linear=SVC(kernel = 'linear', C = 1, probability=True)
    svm_model_linear = linear.fit(X_train, y_train.values.ravel()) 
    svm_predictions = svm_model_linear.predict(X_test) 
    
    # model accuracy for X_test   
    accuracy = svm_model_linear.score(X_test, y_test) 
    # creating a confusion matrix 
    cm = confusion_matrix(y_test, svm_predictions)

    labels, uniques = pd.factorize(y_test.iloc[:, 0].tolist())
    # labels=labels.astype('U')
    # labels = np.array(labels, dtype=data.astype('U'))
    # print(labels.dtype)
    # print(X_test.as_matrix().dtype)
    # plot_decision_regions(X_test.as_matrix()[:200], labels[:200], clf=linear, res=0.1)
    # plt.show()

    return cm, accuracy, linear

def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    # training a KNN classifier 
    knn=KNeighborsClassifier(n_neighbors = 15)
    knn.fit(X_train, y_train.values.ravel()) 
    
    # accuracy on X_test 
    accuracy = knn.score(X_test, y_test)     
    # creating a confusion matrix 
    knn_predictions = knn.predict(X_test)  
    cm = confusion_matrix(y_test, knn_predictions)
    return cm, accuracy, knn


def plot_boundaries(svm_model, X,y):
    # Plot Decision Region using mlxtend's  plotting function
    labels, uniques = pd.factorize(y.iloc[:, 0].tolist())
    # #print(labels)
    # #y=y.values.astype(np.int64)
    # print(X.values)
    # plot_decision_regions(X=X.values, y=labels, clf=svm_model, legend=2)
    # plt.xlabel(X.columns[0])
    # plt.ylabel(X.columns[1])
    # plt.title('SVM Decision Region Boundary', size=16)
    # plt.show()
    # Plotting decision regions
    X=X.values
    print(X)
    print(labels)
    plot_decision_regions(X=X, y=labels, clf=svm_model, legend=2)

    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
    #                     np.arange(y_min, y_max, 0.1))
    # fig, ax = plt.subplots()
    # X0, X1 = X[:, 0], X[:, 1]
    # Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    # ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # # Plot also the training points
    # ax.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    # ax.set_ylabel('PC2')
    # ax.set_xlabel('PC1')
    # ax.set_xticks(())
    # ax.set_yticks(())
    # ax.set_title("SVM doundaries")
    # ax.legend()
    # plt.show()

def plot_accuracy(method, cm,accuracy):
    print("Classifier: "+method)
    print("Confusion matrix: \n")
    print(cm)
    print("Accuracy: "+str(accuracy)+"\n")
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    plt.title("Confusion matrix "+method)
    plt.show()
    

def main():
    Data = pd.read_csv("data/PCA_transformed_data.csv", header=None)
    Labels=pd.read_csv("data/labels.csv")
    #selecting only 2 components
    Data=Data.iloc[:,1:3]
    #TO VISUALIZE the FEATURE SPACE, remove comments below
    # labels, uniques = pd.factorize(Labels.iloc[:, 1].tolist())
    # plt.scatter(Data.as_matrix()[:,0], Data.as_matrix()[:,1], s=4, alpha=0.3, c=labels, cmap='RdYlBu_r')
    # plt.show()
    #split dataset in tr
    # aining and testing
    X_train, X_test, y_train, y_test = train_test_split(Data, Labels[["Class"]], random_state = 0,test_size=0.5) 
    
    #decision tree classifier
    cm_dt, acc_dt,tree=decision_tree(X_train, X_test, y_train, y_test)
    plot_accuracy("decision tree", cm_dt, acc_dt)
    #SVM
    cm_svm, accuracy_svm, svm_model =support_vector_machine(X_train, X_test, y_train, y_test)
    plot_accuracy("SVM", cm_svm, accuracy_svm)

    #NOT WORKING 
    #plot_boundaries(svm_model, X_train, y_train)
    #KNN
   
    cm_knn, accuracy_knn, knn=k_nearest_neighbors(X_train, X_test, y_train, y_test)
    plot_accuracy("K-NN", cm_knn, accuracy_knn)

    #roc plot --> IS NOT WORKING (IDK why)
    #roc_plot(X_train, y_train, X_test, y_test, svm_model, tree, knn)




if __name__ == '__main__':
    main()
