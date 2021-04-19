# Create test set 
import pandas as pd
import os
import shutil
import pdb

from pathlib import Path

# Get the current working directory
dir = os.getcwd()

# Read the CSV having the info about the test images
df = pd.read_csv(Path(dir + "/Test data/data/PlantCLEF2017OnlyTest.csv"), delimiter=";", error_bad_lines=False)

# Create a list with all the unique class names of the test set
classes = df["ClassId"].unique().tolist()

for i in range(len(classes)):
    os.makedirs(Path(dir + "/Structured Test Data/data/" + str(classes[i])))

for j in range(len(df.index)):
    try:
        shutil.move(Path(dir + "/Test data/data/" + str(df['MediaId'][j]) + ".jpg"),
                    Path(dir + "/Structured Test Data/data/" + str(df['ClassId'][j]) + "/" + str(df['MediaId'][j]) + ".jpg"))
    except:
        FileNotFoundError
