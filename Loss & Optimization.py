import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def quick_summary(df, name):
    print(f"{name} shape: {df.shape}")
    print(df.head())
    print()

def plot_history(hist, title):
    plt.figure(figsize=(5, 3))
    plt.plot(hist.history["loss"], label="loss")
    if "accuracy" in hist.history:
        plt.plot(hist.history["accuracy"], label="accuracy")
    plt.title(title)
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

print("Setup complete ✅")


n = 2500
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
noise = np.random.normal(0, 0.5, n)
y = (x1 + 0.8 * x2 + noise > 0).astype(int)

df = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "label": y
})

quick_summary(df, "df")
print("label rate:", round(df["label"].mean(), 3))
print(df.describe().T)


# TODO 1: choose optimizer (e.g., 'adam', 'sgd')
optimizer = "adam"

# TODO 2: choose loss for binary classification
loss_fn = "binary_crossentropy"

# TODO 3: set number of epochs
epochs = 20

X = df[["x1", "x2"]].values
y = df["label"].values

model = Sequential([
    Dense(6, activation="relu", input_shape=(2,)),
    Dense(6, activation="relu", input_shape=(2,)),
    Dense(6, activation="relu", input_shape=(2,)),

    Dense(1, activation="sigmoid"),
])

model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
history = model.fit(X, y, epochs=epochs, verbose=0)

print("Final loss:", round(float(history.history["loss"][-1]), 3))
print("Final accuracy:", round(float(history.history["accuracy"][-1]), 3))



plot_history(history, "Training Curve")

model_sgd = Sequential([
    Dense(6, activation="relu", input_shape=(2,)),
    Dense(1, activation="sigmoid"),
])
model_sgd.compile(optimizer="sgd", loss=loss_fn, metrics=["accuracy"])
hist_sgd = model_sgd.fit(X, y, epochs=epochs, verbose=0)

print("Final loss (adam):", round(float(history.history["loss"][-1]), 3))
print("Final loss (sgd):", round(float(hist_sgd.history["loss"][-1]), 3))

print("Interpretation: optimizer choice affects how fast loss decreases.")


model_mse = Sequential([
    Dense(6, activation="relu", input_shape=(2,)),
    Dense(1, activation="sigmoid"),
])
model_mse.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
hist_mse = model_mse.fit(X, y, epochs=epochs, verbose=0)

print("Final loss (MSE):", round(float(hist_mse.history["loss"][-1]), 3))
print("Final loss (BCE):", round(float(history.history["loss"][-1]), 3))
print("Fix: use binary cross-entropy for binary classification.")


