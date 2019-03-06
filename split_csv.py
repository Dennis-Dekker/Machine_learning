import pandas as pd

"""Split dataset in multiple csv files so it can be uploaded to Gitself.

Git maximum file size: 100 MB
Size of data: 196 MB 
Split into 3 files with a maximum of 80.9 MB 
"""

filename = "data/raw_data.csv"
# amount of lines
chunksize = 400
i = 1
print("\n--- .csv file splitter ---\n")
print("Splits .csv file into smaller .csv files")
print("Primary goal is to make big .csv files uploadable to Github\n")
print("Input file:\t" + filename)
print("Chunk size:\t" + str(chunksize))
print("\nWriting to file:")
for chunk in pd.read_csv(filename, chunksize=chunksize, low_memory=False):
    print("\t" + filename.split(".")[0] + str(i) + ".csv")
    chunk.to_csv(filename.split(".")[0] + str(i) + ".csv")
    i += 1
print("--- DONE ---\n")
