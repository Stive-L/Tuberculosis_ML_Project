import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("tuberculosis_data.csv", delimiter=";")


gender_counts = df['gender'].value_counts()


plt.figure(figsize=(6, 4))
plt.bar(gender_counts.index, gender_counts.values, color=['skyblue', 'lightpink'], edgecolor='black')
plt.title('Gender Distribution', fontsize=14)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()