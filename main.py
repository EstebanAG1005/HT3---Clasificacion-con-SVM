import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Leer los datos del archivo CSV
df = pd.read_csv("wine_fraud.csv", header=None)

# Agregar nombres a las columnas
column_names = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
    "type",
]
df.columns = column_names

# Convert "type" column into binary columns using one-hot encoding
df = pd.get_dummies(df, columns=["type"])

# Check the class distribution of the target variable
print(df["quality"].value_counts(normalize=True))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("quality", axis=1), df["quality"], test_size=0.3, random_state=42
)

# Calculate class weights
class_weight = dict(
    zip(
        np.unique(y_train),
        np.array([(y_train == i).sum() for i in np.unique(y_train)]) / len(y_train),
    )
)

# Initialize the SVM model with class weight
svm = SVC(class_weight=class_weight, kernel="linear")

# Train the model
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
