import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === 1. Données ===
# colonnes : [heures_etude, heures_sommeil]
X_raw = np.array([
    [1, 8],
    [2, 7],
    [3, 6],
    [4, 6],
    [5, 5],
    [6, 5],
    [7, 4],
    [8, 3]
])

# Y : 1 = admis, 0 = refusé
y = np.array([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)

# === 2. Standardisation ===
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# === 3. Initialisation ===
n_samples, n_features = X.shape
W = np.zeros((n_features, 1))
b = 0.0
learning_rate = 0.1
epochs = 1000

# Avant la boucle d'entraînement
costs_history = []

# === 4. Fonctions ===
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(y_true, y_pred):
    # Fonction de perte logistique (binaire cross-entropy)
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

# === 5. Entraînement ===
for epoch in range(epochs):
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)

    cost = loss(y, y_pred)

    # Dans la boucle, après le calcul du coût
    costs_history.append(cost)

    dz = y_pred - y
    dW = (1 / n_samples) * np.dot(X.T, dz)
    db = (1 / n_samples) * np.sum(dz)

    W -= learning_rate * dW
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch} → Loss : {cost:.4f}, W = {W.ravel()}, b = {b:.4f}")

# Après la boucle d'entraînement
plt.figure(figsize=(10, 6))
plt.plot(costs_history)
plt.title('Évolution du coût pendant l\'apprentissage')
plt.xlabel('Époques')
plt.ylabel('Coût (MSE)')
plt.grid(True)
plt.show()

# === 6. Prédiction sur exemple ===
x_test = scaler.transform(np.array([[5, 6]]))  # élève avec 5h d’étude, 6h de sommeil
proba = sigmoid(np.dot(x_test, W) + b)
classe = int(proba[0][0] >= 0.5)

print(f"\n→ Probabilité d’admission : {proba[0][0]:.2f}")
print(f"→ Classe prédite : {'Admis (1)' if classe else 'Refusé (0)'}")