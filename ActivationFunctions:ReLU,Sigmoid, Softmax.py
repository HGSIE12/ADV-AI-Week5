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

def plot_activation(x, y, title):
    plt.figure(figsize=(5, 3))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("activation(x)")
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.show()

print("Setup complete ✅")

n = 140
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
noise = np.random.normal(0, 0.4, n)
y = (x1 - 0.7 * x2 + noise > 0).astype(int)

df = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "label": y
})

quick_summary(df, "df")
print("label rate:", round(df["label"].mean(), 3))
print(df.describe().T)


# TODO 1: choose hidden activation (e.g., 'relu', 'tanh', 'sigmoid')
hidden_activation = "sigmoid"

# TODO 2: set number of hidden units
hidden_units = 6

hidden_units2 = 8

hidden_units3 = 10

# TODO 3: set epochs
epochs = 30

X = df[["x1", "x2"]].values
y = df["label"].values

model = Sequential([
    Dense(hidden_units, activation=hidden_activation, input_shape=(2,)),
    Dense(hidden_units2, activation=hidden_activation, input_shape=(2,)),
    Dense(hidden_units3, activation=hidden_activation, input_shape=(2,)),

    Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(X, y, epochs=epochs, verbose=0)


print("Final training accuracy:", round(float(history.history["accuracy"][-1]), 3))


x = np.linspace(-5, 5, 200)
relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
softmax = np.exp(x) / np.sum(np.exp(x))

plot_activation(x, relu, "ReLU")
plot_activation(x, sigmoid, "Sigmoid")
plot_activation(x, softmax, "Softmax")

preds = model.predict(X[:5], verbose=0)
print("Predictions head:", np.round(preds.flatten(), 3))
assert np.all((preds >= 0) & (preds <= 1)), "Sigmoid outputs must be in [0, 1]."

print("Interpretation: activations introduce non-linearity for richer decision boundaries.")

model_sigmoid = Sequential([
    Dense(6, activation="sigmoid", input_shape=(2,)),
    Dense(1, activation="sigmoid"),
])
model_sigmoid.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
hist_sigmoid = model_sigmoid.fit(X, y, epochs=8, verbose=0)

model_relu = Sequential([
    Dense(6, activation="relu", input_shape=(2,)),
    Dense(1, activation="sigmoid"),
])
model_relu.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
hist_relu = model_relu.fit(X, y, epochs=8, verbose=0)

print("Final accuracy (sigmoid hidden):", round(float(hist_sigmoid.history["accuracy"][-1]), 3))
print("Final accuracy (ReLU hidden):", round(float(hist_relu.history["accuracy"][-1]), 3))
print("Fix: use ReLU in hidden layers for faster, stable learning.")
