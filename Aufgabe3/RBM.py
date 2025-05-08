import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def encode(visible_vector, weights, bias):
    return sigmoid(np.dot(visible_vector, weights) + bias)


def decode(hidden_vector, weights, bias):
    return sigmoid(np.dot(hidden_vector, weights.T) + bias)


# Lernregel ohne Erwartungswert -> Sigmoidfunktion
def contrastive_divergence(visible_vector0, weights, visible_bias, hidden_bias, lr):
    hidden_vector0 = encode(visible_vector0, weights, hidden_bias)

    visible_vectork = visible_vector0.copy()
    hidden_vectork = hidden_vector0.copy()

    for _ in range(2):
        visible_vectork = decode(hidden_vectork, weights, visible_bias)
        hidden_vectork = encode(visible_vectork, weights, hidden_bias)

    weights += lr * (
        np.dot(visible_vector0.T, hidden_vector0)
        - np.dot(visible_vectork.T, hidden_vectork)
    )
    visible_bias += lr * np.sum(visible_vector0 - visible_vectork, axis=0)
    hidden_bias += lr * np.sum(hidden_vector0 - hidden_vectork, axis=0)

    return weights, visible_bias, hidden_bias


# RBM erstellen soll MNIST Bilder wieder ausgeben nach Komprimierung (784 -> 100 -> 784)
n_visible = 784
n_hidden = 100
weights = np.random.normal(0, 0.5, size=(n_visible, n_hidden))
visible_bias = np.ones(n_visible)
hidden_bias = np.ones(n_hidden)
lr = 0.1
epochs = 20
batch_size = 1

# MNIST Daten laden
with open("Aufgabe3/data/train_img", "rb") as f:
    f.read(4)
    num_img = int.from_bytes(f.read(4), "big")
    num_rows = int.from_bytes(f.read(4), "big")
    num_cols = int.from_bytes(f.read(4), "big")
    data = f.read()
x_train = np.ndarray((num_img, num_rows, num_cols), np.uint8, data)
x_train = x_train[:5000]
x_train = x_train.reshape(-1, 784) / 255.0
x_train = (x_train > 0.5).astype(np.int32)
test_samples = x_train[:10].copy()

plt.ion()
fig, axes = plt.subplots(3, 10, figsize=(15, 3))
plt.subplots_adjust(hspace=0.5)

for epoch in range(epochs):
    h = encode(test_samples, weights, hidden_bias)
    rec = decode(h, weights, visible_bias)

    for i in range(10):
        axes[0, i].imshow(test_samples[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original")

        axes[1, i].imshow(h[i].reshape(10, 10), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Komprimierung")

        axes[2, i].imshow(rec[i].reshape(28, 28), cmap="gray")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title(f"Epoch {epoch+1}")

    plt.pause(2)

    for i in range(0, len(x_train), batch_size):
        batch = x_train[i : i + batch_size]
        weights, visible_bias, hidden_bias = contrastive_divergence(
            batch, weights, visible_bias, hidden_bias, lr
        )
    np.random.shuffle(x_train)

plt.ioff()
plt.show()
