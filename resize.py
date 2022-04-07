import numpy as np
from os import walk

# This file will automatically go into a directory called "raw_data" and convert all the numpy files to 1000 entries and save them in the "data" directory. The "raw_data" directory must be created locally as it is part of the .gitignore so I don't accidentally upload a gigabyte of data to github.

filenames = next(walk("./DoodleClassifierModel/raw_data/"), (None, None, []))[2]  # [] if no file

for file in filenames:
    data = np.load(f"./DoodleClassifierModel/raw_data/{file}")[:1000]
    np.save(f"./DoodleClassifierModel/data/{file}", data)

print(f"Found and resized {len(filenames)} files!")