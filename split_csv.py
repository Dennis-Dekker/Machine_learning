import pandas as pd

"""Split dataset in multiple csv files so it can be uploaded to Gitself.

Git maximum file size: 100 MB
Size of data: 196 MB 
Split into 3 files with a maximum of 80.9 MB 
"""

filename = "data/raw_data.csv"
chunksize = 400
i = 1
for chunk in pd.read_csv(filename, chunksize=chunksize, low_memory=False):
        chunk.to_csv(filename.split(".")[0] + str(i) + ".csv")
        i += 1
