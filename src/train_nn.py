import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from utils.preprocess import load_and_normalize
import os

def build_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(9,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate(X, y):
    results = {"sample_size": [], "mse": [], "r2": []}
    sample_sizes = [100, 200, 300]

    for size in sample_sizes:
        model = build_model()
        model.fit(X[:size], y[:size], epochs=100, verbose=0)
        pred = model.predict(X[:size])
        mse = mean_squared_error(y[:size], pred)
        r2 = r2_score(y[:size], pred)

        results["sample_size"].append(size)
        results["mse"].append(mse)
        results["r2"].append(r2)

    return model, results

def plot_results(results):
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(results["sample_size"], results["mse"], marker='o')
    plt.title("MSE vs Sample Size")
    plt.xlabel("Sample Size")
    plt.ylabel("Mean Squared Error")

    plt.subplot(1, 2, 2)
    plt.plot(results["sample_size"], results["r2"], marker='o', color='green')
    plt.title("R² vs Sample Size")
    plt.xlabel("Sample Size")
    plt.ylabel("R² Score")

    plt.tight_layout()
    plt.savefig("plots/mse_r2_plot.png")
    plt.show()

if __name__ == "__main__":
    X, y, _ = load_and_normalize("data/data.xlsx")
    model, results = train_and_evaluate(X, y)
    plot_results(results)
