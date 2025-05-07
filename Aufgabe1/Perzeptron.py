import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use("TkAgg")
#                 BIAS,x,y
train = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
target = np.array([0, 0, 0, 1])  # AND Operation
out = np.array([0, 0, 0, 0])
weight = np.random.rand(3) * (0.5)
learnrate = 1.0
grad = np.zeros(3)


def threshold(summe):
    return (summe > 0).astype(int)


def learn():
    global train, weight, out, target, learnrate
    # Ausgabe des Perzeptrons anhan der Trainingsdaten berechnen
    out = threshold(np.matmul(train, weight))

    # Perzeptron Lernregel iterativ
    # for j in range(len(train)):
    #     for i in range(len(weight)):
    #         if train[j][i] == 1:
    #             if out[j] == 0 and target[j] == 1:
    #                 weight[i] += train[j][i]
    #             if out[j] == 1 and target[j] == 0:
    #                 weight[i] -= train[j][i]

    # Perzeptron Lernregel vektorisiert
    increase = train * np.where((out[:, None] == 0) & (target[:, None] == 1), 1, 0)
    decrease = train * np.where((out[:, None] == 1) & (target[:, None] == 0), 1, 0)

    grad = np.sum(increase, axis=0) - np.sum(decrease, axis=0)

    weight += learnrate * grad


def outp(N=100):  # Daten für die Ausgabefunktion generieren
    global weight
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)
    oo = threshold(weight[0] + weight[1] * xx + weight[2] * yy)
    return xx, yy, oo


def on_close(event):  # Fenster schließen
    exit(0)


plt.ion()
fig = plt.figure()
fig.canvas.mpl_connect("close_event", on_close)
while True:  # Endlosschleife
    # for i in range(1):
    learn()  # lerne einen Schritt
    plt.clf()  # Bildschirm löschen
    X, Y, Z = outp()  # generiere Plotdaten
    ax = fig.add_subplot(111, projection="3d")
    # 3D plot von den Daten
    ax.plot_surface(
        X, Y, Z, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3
    )
    ax.set_title("Neuron lernt AND-Funktion")
    ax.set_xlabel("In[1]")
    ax.set_ylabel("In[2]")
    ax.set_zlabel("Ausgabe\ndes Neurons")
    ax.set_zlim(0, 1)
    plt.draw()
    plt.pause(0.00001)
