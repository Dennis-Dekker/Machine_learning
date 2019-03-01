import numpy as np
import synapseclient
import pandas as pd

"""Create dataset from Synapse 

Start with 5 cancer types:
- BRCA
- LUAD
- PRAD
- KIRC
- COAD
"""

def load_dataset_ids():
    f = open("data/synapse_dataset_IDs.txt", "r")
    ID_dict = {}
    for line in f:
        if line[0] == "#":
            pass
        else:
            ID_dict[line.split(" ")[0]] = line.strip().split()[1:]
            
    
    return ID_dict

def download_data_synapse(list_datasets):
    """Download data from Synapse 
    """
    
    syn = synapseclient.Synapse()
    syn.login('Machine_learning_project_70','Group_70')

    # Obtain a pointer and download the data
    print("loading dataset:")
    for cancer_type in list_datasets:
        list_datasets[cancer_type][0] = syn.get(entity = list_datasets[cancer_type][0])
        list_datasets[cancer_type][1] = syn.get(entity = list_datasets[cancer_type][1])
        print(cancer_type)
    
    return list_datasets

def load_dataset(dataset):
    """Load dataset of every downloaded Synapse dataset. 
    If multiple dataset are loaded, concatenate them to one dataframe.
    """ 
    
    for cancer_type in dataset:
        ## load the data matrix into a dataframe.
        with open(dataset[cancer_type][0].path, 'r') as f:
            labels = f.readline().strip().split('\t')
            data = {label: [] for label in labels}
            for line in f:
                values = [line.strip().split('\t')[0]]
                values.extend([float(x) for x in line.strip().split('\t')[1:]])
                for i in range(len(labels)):
                    data[labels[i]].append(values[i])
            if cancer_type == list(dataset.keys())[0]:
                df = data_to_pandas(data)
            else:
                df = pd.concat([df, data_to_pandas(data)], axis = 1)
                
    return df

def data_to_pandas(data):
    ## load the data matrix into a pandas dataframe
    df = pd.DataFrame.from_dict(data)
    print(df.shape)
    
    return df
    
def load_annotation_files(dataset):
    """Load annotation of every downloaded Synapse dataset. 
    If multiple dataset are loaded, concatenate them to one dataframe.
    """ 
    for cancer_type in dataset:
        ## load the data matrix into a dataframe.
        with open(dataset[cancer_type][1].path, 'r') as f:
            labels = f.readline().strip().split('\t')
            data = {label: [] for label in labels}
            for line in f:
                values = line.split('\t')
                #values.extend([float(x) for x in line.strip().split('\t')[1:]])
                for i in range(len(labels)):
                    data[labels[i]].append(values[i])
            if cancer_type == list(dataset.keys())[0]:
                df = data_to_pandas(data)
            else:
                df = pd.concat([df, data_to_pandas(data)], axis = 1)
                
    return df

def main():
    
    # Get cancer_type IDs from Synapse. First value dict is expression data, 
    ## second is annotation file.
    ID_dict = load_dataset_ids()
    list_datasets = download_data_synapse(ID_dict)
    
    #data = load_dataset(list_datasets)
    annotation = load_annotation_files(list_datasets)
    
if __name__ == '__main__':
    main()
