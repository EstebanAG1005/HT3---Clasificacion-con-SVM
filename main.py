import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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

# Generar un informe detallado de los datos
profile = ProfileReport(df, title="Wine dataset", explorative=True)

# Guardar el informe en un archivo HTML
profile.to_file("Wine_Dataset.html")
