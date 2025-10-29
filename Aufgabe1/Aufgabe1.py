# ==============================================================
# Machine Learning Vergleich — E-Commerce-Dataset
# Modelle: Naive Bayes, SVM, KNN, Decision Tree, Random Forest, Logistic Regression
# ==============================================================
# Autor: (Dein Name)
# Datum: (heutiges Datum)
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================================================
# 1. Datensatz laden
# ==============================================================
dataset_path = "Data/ecommerce_dataset_10000.csv"
df = pd.read_csv(dataset_path)

print("Form des ursprünglichen Datensatzes:", df.shape)

# Fehlende Werte entfernen
df = df.dropna()
print("Form nach Entfernen fehlender Werte:", df.shape)

# ==============================================================
# 2. Zielvariable und Features definieren
# ==============================================================
target_col = "order_status"

if target_col not in df.columns:
    raise ValueError(f"Die Spalte '{target_col}' existiert nicht im Datensatz!")

X = df.drop(columns=[target_col])
y = df[target_col]

# ==============================================================
# 3. Numerische und kategorische Spalten erkennen
# ==============================================================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns

print(f"Numerische Spalten: {list(num_cols)}")
print(f"Kategorische Spalten: {list(cat_cols)}")

# ==============================================================
# 4. Preprocessing-Pipeline definieren
# ==============================================================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# ==============================================================
# 5. Modelle definieren
# ==============================================================
models = {
    "NaiveBayes": GaussianNB(),
    "SVM_RBF": SVC(kernel="rbf"),
    "KNN_k=5": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=0),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

knn_neighbors = [1, 3, 5, 7, 9]
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]

# ==============================================================
# 6. Training & Evaluation
# ==============================================================
results = []

def eval_model(clf, X, y, train_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0)
    }

for ts in train_sizes:
    for name, model in models.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", model)])
        metrics = eval_model(pipe, X, y, ts)
        results.append({
            "model": name,
            "train_size_percent": int(ts * 100),
            **metrics
        })
    # KNN mit verschiedenen Nachbarn
    for k in knn_neighbors:
        pipe = Pipeline([
            ("pre", preprocessor),
            ("clf", KNeighborsClassifier(n_neighbors=k))
        ])
        metrics = eval_model(pipe, X, y, ts)
        results.append({
            "model": f"KNN_k={k}",
            "train_size_percent": int(ts * 100),
            **metrics
        })

# ==============================================================
# 7. Ergebnisse speichern und anzeigen
# ==============================================================
df_results = pd.DataFrame(results)
df_results.to_csv("classification_results.csv", index=False)
print("\nErgebnisse gespeichert unter: classification_results.csv")

print("\nTop 10 Ergebnisse:")
print(df_results.sort_values(by="accuracy", ascending=False).head(10).to_string(index=False))

# ==============================================================
# 8. Visualisierung (optional)
# ==============================================================
plt.figure(figsize=(10,6))
for model in df_results["model"].unique():
    data = df_results[df_results["model"] == model]
    plt.plot(data["train_size_percent"], data["accuracy"], marker="o", label=model)
plt.xlabel("Trainingsanteil (%)")
plt.ylabel("Accuracy")
plt.title("Modellvergleich bei verschiedenen Train/Test-Verhältnissen")
plt.legend(fontsize="small", ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
