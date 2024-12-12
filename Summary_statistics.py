import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("tuberculosis_data.csv", delimiter=";")



age_mean = df['age'].mean()
age_median = df['age'].median()
age_std = df['age'].std()
ptb_mean = df['ptb'].mean()
ptb_median = df['ptb'].median()
ptb_std = df['ptb'].std()
print(f"Age - Mean: {age_mean:.2f}, Median: {age_median:.2f}, Standard Deviation: {age_std:.2f}")
print(f"PTB - Mean: {ptb_mean:.2f}, Median: {ptb_median:.2f}, Standard Deviation: {ptb_std:.2f}")
