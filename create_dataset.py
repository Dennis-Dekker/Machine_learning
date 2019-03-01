import synapseclient
 
syn = synapseclient.Synapse()
syn.login('Machine_learning_project_70','Group_70')

# Obtain a pointer and download the data
syn2320114 = syn.get(entity='syn2320114')
 
head(syn2320114)
 
print(syn2320114.name)
print(syn2320114.path)
