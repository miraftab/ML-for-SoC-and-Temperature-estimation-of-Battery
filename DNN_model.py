import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from utilities import DistributionPlot

# Turn off tensorflow warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


# Normalization
def normaliz_data(train_features):
    # The first step is to create the layer:
    normalizer = tf.keras.layers.Normalization(axis=-1)
    # Then, fit the state of the preprocessing layer to the data by calling Normalization.adapt:
    normalizer.adapt(np.array(train_features))
    # Calculate the mean and variance, and store them in the layer:
    normalizer.mean.numpy()

    return normalizer


# DNN Model
def build_and_compile_model(norm):
    model = keras.Sequential(
        [
            norm,
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.0001))
    return model


def main():
    # load all Dataset
    df = pd.read_csv("data/wd_all.csv", index_col="Unnamed: 0")
    df = df[["ElapsedTime", "Temperature", "Voltage", "Current", "SOC"]]

    # Split train and test
    train_dataset = df.sample(frac=0.8, random_state=42)
    test_dataset = df.drop(train_dataset.index)

    # Features - change 'SOC' to 'Temperature' to predict temperature
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    all_features = df.copy()
    train_labels = train_features.pop("SOC")
    test_labels = test_features.pop("SOC")
    all_labels = all_features.pop("SOC")

    normalizer = normaliz_data(train_features)

    dnn_model = build_and_compile_model(normalizer)
    print(dnn_model.summary())

    dnn_model.fit(
        train_features, train_labels, validation_split=0.2, verbose=1, epochs=3000
    )

    all_predictions = dnn_model.predict(all_features).flatten()
    # train_predictions = dnn_model.predict(train_features).flatten()
    # test_predictions = dnn_model.predict(test_features).flatten()

    test_results = {}
    test_results["dnn_model"] = dnn_model.evaluate(
        test_features, test_labels, verbose=0
    )
    print(pd.DataFrame(test_results, index=["Mean Squared error [SOC]"]).T)

    DistributionPlot(
        all_labels,
        all_predictions,
        "Actual Temperature",
        "Predicted Values",
        "Temperature Prediction - Kernel Distribution Estimation - Based on ElapsedTime, Voltage, Current",
    )


if __name__ == "__main__":
    main()
