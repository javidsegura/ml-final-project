""" A file for debugging purposes """

import pandas as pd

PATH = "./results/results.csv"

df = pd.read_csv(PATH)

print(df["modelName"])