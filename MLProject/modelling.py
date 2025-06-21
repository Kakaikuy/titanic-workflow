import argparse
import pandas as pd
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Import fungsi preprocessing kamu
from preprocessing.automate_kakai_CpKk import preprocess_titanic_data

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# MLflow setup
mlflow.set_tracking_uri("file:///C:/Users/ASUS/Documents/Python_project/MSML/Workflow-CI/MLProject/mlruns")
mlflow.set_experiment("titanic-face-survival")

# Aktifkan autolog dari TensorFlow
mlflow.tensorflow.autolog()

# Load dan preprocessing
raw_data = pd.read_csv(args.data_path)
df_clean, X_train, X_test, y_train, y_test = preprocess_titanic_data(
    raw_data, target_column='2urvived', drop_columns=['Passengerid']
)

# Start MLflow run
with mlflow.start_run(run_name="ci_train_run"):
    # Model arsitektur
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluasi dan log metrik manual
    y_pred = model.predict(X_test).round()
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("f1_score", f1)

    # Jika autolog aktif, model dan metrics utama sudah tercatat otomatis
