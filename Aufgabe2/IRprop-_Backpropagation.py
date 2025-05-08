import numpy as np


# Sigmoide Aktivierungsfunktion und ihre Ableitung
def sigmoid(x):
    x = np.clip(x, -100, 100)
    return 1 / (1 + np.exp(-x))  # Sigmoidfunktion


def deriv_sigmoid(x):
    return x * (1 - x)  # Ableitung der Sigmoiden


# Das XOR-Problem, input [bias, x, y] und Target-Daten
inp = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
target = np.array([[0], [1], [1], [0]])

# Die Architektur des neuronalen Netzes
inp_size = 3  # Eingabeneuronen
hid_size = 4  # Hidden-Neuronen
out_size = 1  # Ausgabeneuron
learnrate = 0.1

# Gewichte zufällig initialisieren (Mittelwert = 0)
w0 = np.random.random((inp_size, hid_size)) - 0.5
w1 = np.random.random((hid_size, out_size)) - 0.5

# Lernraten für einzelne Gewichte initialisieren
lr0 = np.zeros(w0.shape) + 0.1
lr1 = np.zeros(w1.shape) + 0.1

grad_L1 = np.zeros((inp_size, hid_size))
grad_L2 = np.zeros((hid_size, out_size))
grad_L1_old = np.copy(grad_L1)
grad_L2_old = np.copy(grad_L2)

eta_plus = 1.2  # Faktor zur Vergrößerung der Lernrate
eta_minus = 0.5  # Faktor zur Verkleinerung der Lernrate
delta_max = 50  # Maximale Gewichtsänderung
delta_min = 0  # Minimale Gewichtsänderung

# Netzwerk trainieren
for i in range(1000):

    # Vorwärtsaktivierung
    L0 = inp
    L1 = sigmoid(np.matmul(L0, w0))
    L1[0, 0] = 1  # Bias-Neuron in der Hiddenschicht
    L2 = sigmoid(np.matmul(L1, w1))

    grad_L1_old = np.copy(grad_L1)
    grad_L2_old = np.copy(grad_L2)

    # Fehler berechnen
    L2_error = target - L2

    # Backpropagation
    L2_delta = L2_error * deriv_sigmoid(L2)
    L1_error = np.matmul(L2_delta, w1.T)
    L1_delta = L1_error * deriv_sigmoid(L1)

    grad_L2 = np.matmul(L1.T, L2_delta)
    grad_L1 = np.matmul(L0.T, L1_delta)

    # iRProp- Algorithmus -> iterativ Lernraten anpassen
    # for i in range(inp_size):
    #     for j in range(hid_size):
    #         if grad_L1_old[i, j] * grad_L1[i, j] > 0:  # Lernrate vergrößern
    #             lr0[i, j] = min(lr0[i, j] * eta_plus, delta_max)
    #         if grad_L1_old[i, j] * grad_L1[i, j] < 0:  # Lernrate verkleinern
    #             lr0[i, j] = max(lr0[i, j] * eta_minus, delta_min)
    #             grad_L1[i, j] = 0  # Einziger Unterschied zu Rprop

    # for i in range(hid_size):
    #     for j in range(out_size):
    #         if grad_L2_old[i, j] * grad_L2[i, j] > 0:  # Lernrate vergößern
    #             lr1[i, j] = min(lr1[i, j] * eta_plus, delta_max)
    #         if grad_L2_old[i, j] * grad_L2[i, j] < 0:  # Lernrate verkleinern
    #             lr1[i, j] = max(lr1[i, j] * eta_minus, delta_min)
    #             grad_L2[i, j] = 0

    # iRProp- Algorithmus -> vektorisiert Lernraten anpassen

    L1_same_sign = grad_L1_old * grad_L1 > 0
    L2_same_sign = grad_L2_old * grad_L2 > 0

    L1_sign_changed = grad_L1_old * grad_L1 < 0
    L2_sign_changed = grad_L2_old * grad_L2 < 0

    l0 = np.where(L1_same_sign, np.minimum(lr0 * eta_plus, delta_max), lr0)
    l0 = np.where(L1_sign_changed, np.maximum(lr0 * eta_minus, delta_min), lr0)
    l1 = np.where(L2_same_sign, np.minimum(lr1 * eta_plus, delta_max), lr1)
    l1 = np.where(L2_sign_changed, np.maximum(lr1 * eta_minus, delta_min), lr1)

    grad_L1 = np.where(L1_sign_changed, 0.0, grad_L1)
    grad_L2 = np.where(L2_sign_changed, 0.0, grad_L2)

    # Gewichte aktualisieren mit einzelnen Lernraten
    w1 += np.sign(grad_L2) * lr1
    w0 += np.sign(grad_L1) * lr0

# Netzwerk testen
L0 = inp
L1 = sigmoid(np.matmul(inp, w0))
L1[0, 0] = 1  # Bias-Neuron in der Hiddenschicht
L2 = sigmoid(np.matmul(L1, w1))

print(L2)
