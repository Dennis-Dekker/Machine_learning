

def roc_plot(X_train,y_train,X_test, y_test, linear, tree, knn):
    """
    This function should calculate the roc plot for every class of SVM classifier
    using every class ALl-against-all. Then make the average for the different classes
    and plot.   WORKING: for svm
    """
    # Binarize the output
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    y_train=label_binarize(y_train,classes=[0, 1, 2, 3, 4])
    n_classes = 5
    # Learn to predict each class against the other
    classifier_svm = OneVsRestClassifier(SVC(kernel="linear",probability=True, C=1,random_state=0))
    classifier_knn = OneVsRestClassifier(knn)
    classifier_dt=OneVsRestClassifier(tree)

    #KNN roc plot
    y_score_knn = classifier_knn.fit(X_train, y_train).predict_proba(X_test)
    #compute ROC curve for each class
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    for i in range(n_classes):
        fpr[i],tpr[i], _ = roc_curve(y_test[:,i],y_score_knn[:,i])
        roc_auc[i]=auc(fpr[i],tpr[i])
    
    #compute micro-average ROC
    fpr["micro"],tpr["micro"],_=roc_curve(y_test.ravel(), y_score_knn.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','yellow','black'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))   
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic knn')
    plt.legend(loc="lower right")
    plt.show()
    # #SVM roc plot
    # y_score_svm = classifier_svm.fit(X_train, y_train).decision_function(X_test)
    # #compute ROC curve for each class
    # fpr=dict()
    # tpr=dict()
    # roc_auc=dict()
    # for i in range(n_classes):
    #     fpr[i],tpr[i], _ = roc_curve(y_test[:,i],y_score_svm[:,i])
    #     roc_auc[i]=auc(fpr[i],tpr[i])
    
    # #compute micro-average ROC
    # fpr["micro"],tpr["micro"],_=roc_curve(y_test.ravel(), y_score_svm.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # plt.figure()
    # lw = 2
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #      label='micro-average ROC curve (area = {0:0.2f})'
    #            ''.format(roc_auc["micro"]),
    #      color='deeppink', linestyle=':', linewidth=4)
    
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue','yellow','black'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #             label='ROC curve of class {0} (area = {1:0.2f})'
    #             ''.format(i, roc_auc[i]))
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic svm')
    # plt.legend(loc="lower right")
    # plt.show()

    #decision tree roc plot
    y_score_dt = classifier_dt.fit(X_train, y_train).predict_proba(X_test)
    #compute ROC curve for each class
    fpr=dict()
    tpr=dict()
    roc_auc=dict()
    for i in range(n_classes):
        fpr[i],tpr[i], _ = roc_curve(y_test[:,i],y_score_dt[:,i])
        roc_auc[i]=auc(fpr[i],tpr[i])
    
    #compute micro-average ROC
    fpr["micro"],tpr["micro"],_=roc_curve(y_test.ravel(), y_score_dt.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','yellow','black'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))   
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic knn')
    plt.legend(loc="lower right")
    plt.show()

