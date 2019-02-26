import pandas as pd

filename = "data.csv"
chunksize = 400
i = 1
for chunk in pd.read_csv(filename, chunksize=chunksize):
        chuck.to_csv(filename.split(".")[0] + str(i) + ".csv" )
        i += 1
