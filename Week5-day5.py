import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def show_images(images, labels, n=6):
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(6, 4))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(f"label={labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

print("Setup complete")

def make_pattern(label):
    img = np.zeros((8, 8), dtype=float)
    if label == 0:
        img[1:-1, 1] = 1
        img[1:-1, -2] = 1
        img[1, 1:-1] = 1
        img[-2, 1:-1] = 1
    elif label == 1:
        img[:, 3:5] = 1
    elif label == 2:
        img[1, 1:-1] = 1
        for i in range(1, 7):
            img[i, 7 - i] = 1
        img[-2, 1:-1] = 1
    return img

def make_dataset(n_per_class=80, noise=0.15):
    images = []
    labels = []
    for label in [0, 1, 2]:
        base = make_pattern(label)
        for _ in range(n_per_class):
            noisy = base + np.random.normal(0, noise, base.shape)
            noisy = np.clip(noisy, 0, 1)
            images.append(noisy)
            labels.append(label)
    return np.array(images), np.array(labels)

images, labels = make_dataset(n_per_class=80, noise=0.12)
X = images.reshape(len(images), -1)
y = labels

print("X shape:", X.shape)
print("y shape:", y.shape)
print("class counts:", pd.Series(y).value_counts().to_dict())
print("feature min/max:", round(float(X.min()), 3), "/", round(float(X.max()), 3))
show_images(images, y, n=6)



hidden_units = 16
hidden_activation = "relu"
epochs = 10
test_frac = 0.2

df = pd.read_csv("mnist_train.csv")

X = df.iloc[:, 1:].values / 255.0
y = df.iloc[:, 0].values

rng = np.random.default_rng(RANDOM_STATE)
idx = rng.permutation(len(X))

split = int(len(X) * (1 - test_frac))

train_idx, test_idx = idx[:split], idx[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

y_train_oh = to_categorical(y_train, num_classes=10)
y_test_oh = to_categorical(y_test, num_classes=10)

model = Sequential([
    Dense(hidden_units, activation=hidden_activation, input_shape=(X.shape[1],)),
    Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train_oh,
    epochs=epochs,
    verbose=0
)

print("Train shape:", X_train.shape, y_train_oh.shape)
print("Test shape:", X_test.shape, y_test_oh.shape)

print(
    "Final train accuracy:",
    round(float(history.history["accuracy"][-1]), 3)
)


test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
preds = model.predict(X_test[:5], verbose=0)
pred_labels = np.argmax(preds, axis=1)

print("Test accuracy:", round(float(test_acc), 3))
print("Predictions head:", pred_labels.tolist())
print("True labels head:", y_test[:5].tolist())
print("Prob row sums:", np.round(preds.sum(axis=1), 3).tolist())

assert preds.shape == (5, 3)
assert np.allclose(preds.sum(axis=1), 1, atol=1e-5)

print("Interpretation: softmax outputs probabilities over 3 classes.")



X_raw = X * 255.0

model_raw = Sequential([
    Dense(16, activation="relu", input_shape=(X.shape[1],)),
    Dense(3, activation="softmax"),
])
model_raw.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
hist_raw = model_raw.fit(X_raw[train_idx], y_train_oh, epochs=5, verbose=0)

model_scaled = Sequential([
    Dense(16, activation="relu", input_shape=(X.shape[1],)),
    Dense(3, activation="softmax"),
])
model_scaled.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
hist_scaled = model_scaled.fit(X[train_idx], y_train_oh, epochs=5, verbose=0)

print("Final accuracy (raw):", round(float(hist_raw.history["accuracy"][-1]), 3))
print("Final accuracy (scaled):", round(float(hist_scaled.history["accuracy"][-1]), 3))
print("Fix: normalize input features before training.")
