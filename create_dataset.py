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
    
    
    # Connect to Synapse server
    print("Connecting to Synapse database...\n")
    syn = synapseclient.Synapse()
    syn.login('Machine_learning_project_70','Group_70')

    # Obtain a pointer and download the data
    print("--- Downloading datasets ---")
    for cancer_type in list_datasets:
        print(cancer_type)
        print("\tDataset ID:\t" + list_datasets[cancer_type][0])
        list_datasets[cancer_type][0] = syn.get(entity = list_datasets[cancer_type][0])
        print("\tLocal path:\t" + str(list_datasets[cancer_type][0].path))
        print("\tAnnotation ID:\t" + list_datasets[cancer_type][1])
        list_datasets[cancer_type][1] = syn.get(entity = list_datasets[cancer_type][1])
        print("\tLocal path:\t" + str(list_datasets[cancer_type][1].path))
        
    print("--- DONE ---\n")
    return list_datasets

def load_dataset(dataset):
    """Load dataset of every downloaded Synapse dataset. 
    If multiple dataset are loaded, concatenate them to one dataframe.
    """ 
    
    print("--- Loading datasets ---")
    for cancer_type in dataset:
        ## load the data matrix into a dictionary.
        print(cancer_type)
        with open(dataset[cancer_type][0].path, 'r') as f:
            # first line as dictionary key. (first line is samples IDs)
            labels = f.readline().strip().split('\t')
            data = {label: [] for label in labels}
            print("\tAmount of samples:\t" + str(len(labels)-1))
            
            # Read lines of file (each line gene with expression value per sample)
            print("\tReading data...")
            for line in f:
                values = [line.strip().split('\t')[0]]
                values.extend([float(x) for x in line.strip().split('\t')[1:]])
                for i in range(len(labels)):
                    data[labels[i]].append(values[i])
                    
            # merge datasets if more than one.
            if cancer_type == list(dataset.keys())[0]:
                print("\tCreated new expression dataset.")
                df = data
            else:
                print("\tMerged data to existing expression dataset.")
                del data["gene_id"]
                df.update(data)
                
    # transpose data to get gene_ids as columns
    print("\nProcessing all data...")
    df = data_to_pandas(df).transpose()
    print("--- DONE ---\n")
            
    return df

def data_to_pandas(data):
    ## convert dictionary into a pandas dataframe
    df = pd.DataFrame.from_dict(data)
    
    return df
    
def load_annotation_files(dataset):
    """Load annotation of every downloaded Synapse dataset. 
    If multiple dataset are loaded, concatenate them to one dataframe.
    """ 
    print("--- Loading annotation files ---")
    for cancer_type in dataset:
        print(cancer_type)
        ## load the data matrix into a dictionary.
        with open(dataset[cancer_type][1].path, 'r') as f:
            # first line as dictionary key. (first line is samples IDs)
            labels = f.readline().strip().split('\t')
            data = {label: [] for label in labels}
            data["cancer_type"] = cancer_type
            print("\tAmount of samples:\t" + str(len(labels)-1))
            
            # Read lines of file (each line gene with expression value per sample)
            print("\tReading data...")
            for line in f:
                values = line.split('\t')
                for i in range(len(labels)):
                    data[labels[i]].append(values[i])
            
            # merge datasets if more than one.
            if cancer_type == list(dataset.keys())[0]:
                print("\tCreated new annotation dataset.")
                df = data_to_pandas(data)
            else:
                print("\tMerged data to existing annotation dataset.")
                df = pd.concat([df, data_to_pandas(data)], axis = 0, sort=True)
                
    # filter dataframe 
    print("\nFiltering annotations...")
    df = df.filter(items = ["cancer_type","#","gender","bcr_patient_uuid","patient_id","bcr_patient_barcode","age_at_initial_pathologic_diagnosis"])
    print("--- DONE ---\n")
    
    return df

def main():
    
    # Get cancer_type IDs from Synapse. First value dict is expression data, 
    ## second is annotation file.
    ID_dict = load_dataset_ids()
    list_datasets = download_data_synapse(ID_dict)
    
    # Load datasets
    data = load_dataset(list_datasets)
    annotation = load_annotation_files(list_datasets)
    
    # transpose expression dataset 
    data.columns = data.loc["gene_id"]
    data = data.drop("gene_id", axis = 0)

    # write data to files
    data_file_name = "data/raw_total_data.csv"
    annotation_file_name = "data/raw_labels.csv"
    print("--- Writing dataset to file ---")
    print("Expression data set")
    print("\tPath:\t" + data_file_name)
    print("\tWriting...")
    data.to_csv(data_file_name)
    print("Annotation data set")
    print("\tPath:\t" + annotation_file_name)
    print("\tWriting...")
    annotation.to_csv(annotation_file_name)
    print("--- DONE ---\n")
        
    
if __name__ == '__main__':
    main()
