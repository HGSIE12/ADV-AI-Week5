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



def draw_network(layer_sizes):

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.axis("off")

    max_neurons = max(layer_sizes)

    for layer_idx, layer_size in enumerate(layer_sizes):

        y_positions = np.linspace(0, 1, layer_size)

        x = layer_idx / (len(layer_sizes) - 1)

        for y in y_positions:

            ax.scatter(x, y, s=200, edgecolors="k", facecolors="white")

        if layer_idx > 0:

            prev_size = layer_sizes[layer_idx - 1]

            prev_y = np.linspace(0, 1, prev_size)

            for y_prev in prev_y:

                for y in y_positions:

                    ax.plot([x - 1 / (len(layer_sizes) - 1), x], [y_prev, y], color="#999999", linewidth=1)

    plt.title("Simple Neural Network")

    plt.show()



print("Setup complete ✅")


n = 700

stim_1 = np.random.normal(0, 1, n )

stim_2 = np.random.normal(0, 1, n)

noise = np.random.normal(0, 0.4, n )

firing = (stim_1 + stim_2 + noise > 0).astype(int)



df = pd.DataFrame({

    "stimulus_1": stim_1,

    "stimulus_2": stim_2,

    "fires": firing

})



quick_summary(df, "df")

print("fire rate:", round(df["fires"].mean(), 3))

print(df.describe().T)





# TODO 1: set input dimension

input_dim = 2



# TODO 2: choose hidden units

hidden_units = 6



# TODO 3: choose hidden activation (e.g., 'relu')

hidden_activation = "relu"



# TODO 4: choose output activation for binary classification

output_activation = "sigmoid"



# TODO 5: set a small number of epochs

epochs = 60



X = df[["stimulus_1", "stimulus_2"]].values

y = df["fires"].values



model = Sequential([

    Dense(hidden_units, activation=hidden_activation, input_shape=(input_dim,)),

    Dense(1, activation=output_activation),

])



model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X, y, epochs=epochs, verbose=0)



print("Final training accuracy:", round(float(history.history["accuracy"][-1]), 3))




model.summary()



preds = model.predict(X[:5], verbose=0)

print("Predictions shape:", preds.shape)

print("Predictions head:", np.round(preds.flatten(), 3))



if output_activation == "sigmoid":

    assert np.all((preds >= 0) & (preds <= 1)), "Sigmoid outputs must be in [0, 1]."



draw_network([input_dim, hidden_units, 1])



print("Interpretation: inputs feed into hidden units, then into an output neuron.")





logits = np.array([-2.0, 0.0, 2.0])

y_true = np.array([0.0, 1.0, 1.0])



wrong_pred = logits

correct_pred = 1 / (1 + np.exp(-logits))



wrong_loss = tf.keras.losses.binary_crossentropy(y_true, wrong_pred).numpy()

correct_loss = tf.keras.losses.binary_crossentropy(y_true, correct_pred).numpy()



print("Wrong preds (logits):", np.round(wrong_pred, 3))

print("Correct preds (sigmoid):", np.round(correct_pred, 3))

print("Loss with logits (wrong):", np.round(wrong_loss, 3))

print("Loss with sigmoid (correct):", np.round(correct_loss, 3))

print("Fix: use a sigmoid output for binary classification probabilities.")


