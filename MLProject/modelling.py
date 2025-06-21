import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def run_pipeline(config):
    # Set experiment (boleh, tidak menyebabkan error)
    mlflow.set_experiment(config["experiment_name"])

    # Autolog aktif
    mlflow.sklearn.autolog()

    # Load data
    data = pd.read_csv(config["data_path"])
    X = data.drop(columns=["2urvived"])
    y = data["2urvived"]

    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_state"]
    )

    # Tidak perlu start_run(), karena sudah dijalankan otomatis via `mlflow run .`
    model = RandomForestClassifier(random_state=config["random_state"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc:.4f}")
    print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="mlflow_train.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_pipeline(config)
