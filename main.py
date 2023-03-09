import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns


# Leer los datos del archivo CSV
df = pd.read_csv("wine_fraud_limpio.csv", header=0)  # specify header=0
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# División del conjunto de datos en un conjunto para entrenamiento y otro para pruebas
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Escalamiento o Normalización
normalizador = StandardScaler()
X_entreno = normalizador.fit_transform(X_entreno)
X_prueba = normalizador.transform(X_prueba)

# Entrenar el moderlo Kernel SVM con el conjunto de datos para entrenamiento
clasificador = SVC(kernel="rbf", random_state=0)
clasificador.fit(X_entreno, y_entreno)

# Predicción de los valores del conjunto de datos para pruebas
y_pred = clasificador.predict(X_prueba)

# print de las diferentes pruebas
print(confusion_matrix(y_prueba, y_pred))
print(classification_report(y_prueba, y_pred))
print("Accuracy of the prediction: ", accuracy_score(y_prueba, y_pred))

# Visualize the data using a scatter plot
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Wine Data")
plt.show()

# Visualize the confusion matrix using a heatmap
conf_mat = confusion_matrix(y_prueba, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(
    conf_mat,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=["Non-fraudulent", "Fraudulent"],
    yticklabels=["Non-fraudulent", "Fraudulent"],
)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.show()
