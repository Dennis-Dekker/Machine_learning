import numpy as np
import synapseclient
import pandas
 
syn = synapseclient.Synapse()
syn.login('Machine_learning_project_70','Group_70')

# Obtain a pointer and download the data
syn2320114 = syn.get(entity='syn2320114')
 
# inspect properties
print(type(syn2320114))
 
print(syn2320114.name)
print(syn2320114.path)

## load the data matrix into a dictionary with an entry for each column
with open(syn2320114.path, 'r') as f:
    labels = f.readline().strip().split('\t')
    data = {label: [] for label in labels}
    for line in f:
        values = [line.strip().split('\t')[0]]
        values.extend([float(x) for x in line.strip().split('\t')[1:]])
        for i in range(len(labels)):
            data[labels[i]].append(values[i])

## load the data matrix into a pandas dataframe
df = pd.DataFrame.from_dict(data)
print(df.iloc([[0:5],[0:5]]))
