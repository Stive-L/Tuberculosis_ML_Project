import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

file_path = '../Tuberculosis_data_processed_data.csv'
data = pd.read_csv(file_path, delimiter=';')

clean_data = data.drop(columns=['id', 'remarks'])

label_encoder = LabelEncoder()
clean_data['gender'] = label_encoder.fit_transform(clean_data['gender'])
clean_data['county'] = label_encoder.fit_transform(clean_data['county'])

X = clean_data.drop(columns=['ptb'])
y = clean_data['ptb']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

neighbors_range = range(1, 21)
cv_scores = []

for k in neighbors_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = neighbors_range[np.argmax(cv_scores)]

knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_test, y_pred, average='binary')
roc_auc = roc_auc_score(y_test, knn_classifier.predict_proba(X_test)[:, 1])

report = classification_report(y_test, y_pred)

print(f"Optimal number of neighbors: {best_k}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

fpr, tpr, thresholds = roc_curve(y_test, knn_classifier.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

plt.figure()
plt.plot(neighbors_range, cv_scores, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Choosing the Best k for KNN')
plt.show()

X_selected = clean_data[['county', 'age']].values
y = clean_data['ptb'].values

X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

k = 11
knn_selected = KNeighborsClassifier(n_neighbors=k)
knn_selected.fit(X_train_selected, y_train)

x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = knn_selected.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
plt.scatter(X_train_selected[:, 0], X_train_selected[:, 1], c=y_train, edgecolor='k', marker='o', label='Train Data')
plt.scatter(X_test_selected[:, 0], X_test_selected[:, 1], c=y_test, edgecolor='k', marker='s', label='Test Data')
plt.title(f'KNN Decision Boundaries (k = {k}) with County and Age')
plt.xlabel('County')
plt.ylabel('Age')
plt.legend()
plt.show()
