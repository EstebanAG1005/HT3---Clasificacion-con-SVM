import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

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

# Muestra los primeros 5 registros
print(df.head())

# Convertir la variable quality a numérica (0 para muestras Legit y 1 para muestras Fraud)
df["quality"] = df["quality"].replace({"Legit": 0, "Fraud": 1})

# Convertir la variable quality a numerica
df["quality"] = pd.to_numeric(df["quality"], errors="coerce")

# Convertir la variable type a numérica (0 para muestras de vino white y 1 para muestras de vino red)
df["type"] = df["type"].replace({"white": 0, "red": 1})

# Convertir la variable type a numerica
df["type"] = pd.to_numeric(df["type"], errors="coerce")

# Eliminar filas con valores faltantes
df = df.dropna()

# Eliminar filas duplicadas
df = df.drop_duplicates()

# Generar un informe detallado de los datos
profile = ProfileReport(df, title="Wine dataset", explorative=True)

# Guardar el informe en un archivo HTML
profile.to_file("Wine_Dataset.html")

# Guardar la data limpia en un nuevo archivo
df.to_csv("wine_fraud_limpio.csv")
