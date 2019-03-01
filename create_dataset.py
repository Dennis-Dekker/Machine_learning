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
    for cancer_type in list_datasets:
        list_datasets[cancer_type][0] = syn.get(entity = list_datasets[cancer_type][0])
        list_datasets[cancer_type][1] = syn.get(entity = list_datasets[cancer_type][1])
    
    return list_datasets

def load_dataset(dataset):
        
    ## load the data matrix into a dictionary with an entry for each column
    with open(dataset.path, 'r') as f:
        labels = f.readline().strip().split('\t')
        data = {label: [] for label in labels}
        for line in f:
            values = [line.strip().split('\t')[0]]
            values.extend([float(x) for x in line.strip().split('\t')[1:]])
            for i in range(len(labels)):
                data[labels[i]].append(values[i])
                
    return data

def data_to_pandas():
    ## load the data matrix into a pandas dataframe
    df = pd.DataFrame.from_dict(data)
    print(df.iloc[0:5,0:5])

def main():
    
    # Get cancer_type IDs from Synapse. First value dict is expression data, 
    ## second is annotation file.
    ID_dict = load_dataset_ids()
    list_datasets = download_data_synapse(ID_dict)
    
    for cancer_type  in list_datasets:
        load_dataset(cancer_type)
    
    
if __name__ == '__main__':
    main()
