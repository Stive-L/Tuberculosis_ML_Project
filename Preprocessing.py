import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("tuberculosis_data.csv", delimiter=";")

df['age'].fillna(df['age'].median(), inplace=True)
df['remarks'].fillna('unknown', inplace=True)
