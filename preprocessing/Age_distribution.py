import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../Tuberculosis_data_processed_data.csv", delimiter=";")


plt.figure(figsize=(8, 5))
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Age Distribution', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()