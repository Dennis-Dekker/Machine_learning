 import synapseclient
 
 syn = synapseclient.Synapse()
 syn.login('Machine_learning_project_70','Group_70')
 
 # Obtain a pointer and download the data
 syn2320114 = syn.get(entity='syn2320114')
 
 # Get the path to the local copy of the data file
 filepath = syn2320114.path 
