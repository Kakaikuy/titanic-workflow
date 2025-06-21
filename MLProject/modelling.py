import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# Konfigurasi MLflow lokal
mlflow.set_tracking_uri("file:///C:/Users/ASUS/Documents/Python_project/MSML_rev/SMSML_Kakai/mlruns")
mlflow.set_experiment("Prediksi-Titanic-Survivor")

# Autolog aktif
mlflow.sklearn.autolog()

# Load dataset Titanic
data = pd.read_csv("titanic_preprocessing.csv")

# Pisahkan fitur dan target
X = data.drop(columns=["2urvived"])
y = data["2urvived"]

# One-hot encoding & konversi numerik
X = pd.get_dummies(X, drop_first=True)
X = X.apply(pd.to_numeric, errors="coerce")

# Hilangkan baris yang mengandung NaN
X = X.dropna()
y = y.loc[X.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training dan logging
with mlflow.start_run(run_name="Titanic-RandomForest"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
