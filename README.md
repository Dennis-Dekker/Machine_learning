<img align="right" width="516" height="200" src="https://blog.omictools.com/wp-content/uploads/2017/11/Banner-RNAseq-QC-omictools.png">

# Machine_learning
Machine learning Project - mRNA expression data

## Goal
Test different classification methods (discuss) &rarr; Select the best method 

## Dataset
Gene expression ([Base ML dataset](https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq#), [Raw RNA seq dataset](https://www.synapse.org/#!Synapse:syn4301332)) of five different cancer types.

## Methods
 
Preprocessing (Unsupervised learning):

- Principal Component Analysis(PCA)

- tSNE

Use different classification methods (Supervised learning):

- K-Nearest Neighbors

- Linear Models

- Naive Bayes Classifiers

- Decision Trees

- Kernelized Support Vector Machines(?)

## TODO
Keep track on what we still have to do. Please update this list with new todo's. 

- [x] Update README.
- [ ] Investigate preprocessing that is applied to the data.
- [ ] Write about preprocessing steps in report.
- [ ] Keep track on references in the report.
- [ ] Reorganize repository (give logical filenames, restructure folders, etc.).
- [x] Rewrite PCA scripts structure.
- [x] Calculate amount of PC's needed (PCA script).
- [x] Review PCA script (especially investigate explained variation values).
- [X] Download data of different cancer types from [Synapse](https://www.synapse.org/#!Synapse:syn4301332) and merge with annotations (also from Synapse).
- [ ] Try different Hyperparameters in the ML algorithms (Knn, SVM, ecc) and cross validation
- [x] PCA: try to apply it within cancer types
- [ ] find important features &rarr; DEG (Differentially expressed genes)
- [ ] KEGG analysis (Pathways) 
- [ ] check for class imbalances (bar plot)

