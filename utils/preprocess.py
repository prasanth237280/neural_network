import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_normalize(file_path):
    df = pd.read_excel(file_path)

    # Rename columns: X1–X9 for inputs, Y1–Y4 for outputs
    df.columns = [f"X{i+1}" for i in range(9)] + [f"Y{i+1}" for i in range(4)]
    df = df.dropna()

    X = df.iloc[:, 0:9].values
    y = df.iloc[:, 9].values.reshape(-1, 1)  # Column J = Y1

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
