import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../Raw_data.csv", delimiter=";")

df['age'] = df['age'].fillna(df['age'].median())
df['remarks'] = df['remarks'].fillna('unknown')