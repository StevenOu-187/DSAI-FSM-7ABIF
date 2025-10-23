import pandas as pd

# Lade das Dataset aus dem Ordner "Data"
dataset_path = "Data/ecommerce_dataset_10000.csv"
data = pd.read_csv(dataset_path)

# Zeige die ersten Zeilen des Datasets an
print(data.head())