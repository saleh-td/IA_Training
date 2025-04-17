import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Données d'origine
X_raw = np.array([
    [1, 0, 8],
    [2, 1, 7],
    [3, 1, 6],
    [4, 2, 6],
    [5, 2, 5],
    [6, 3, 5],
    [7, 3, 4]
])
y = np.array([35, 40, 50, 55, 65, 70, 75]).reshape(-1, 1)

def train_model(X_data, y_data, learning_rate, epochs, use_scaler=True):
    if use_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_data)
    else:
        X = X_data
    
    n_samples, n_features = X.shape
    W = np.zeros((n_features, 1))
    b = 0.0
    costs = []
    
    for epoch in range(epochs):
        # Prédiction
        y_pred = np.dot(X, W) + b
        
        # Calcul de l'erreur et du coût
        error = y_pred - y_data
        cost = (1/n_samples) * np.sum(error ** 2)
        costs.append(cost)
        
        # Mise à jour des paramètres
        dW = (2/n_samples) * np.dot(X.T, error)
        db = (2/n_samples) * np.sum(error)
        
        W -= learning_rate * dW
        b -= learning_rate * db
        
    return costs, W, b

# Test avec différentes configurations
learning_rates = [0.1, 0.01, 0.001]
epochs_list = [100, 500, 1000]
colors = ['b', 'g', 'r']

# 1. Test des différents learning rates
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
for i, lr in enumerate(learning_rates):
    costs, W, b = train_model(X_raw, y, learning_rate=lr, epochs=1000)
    plt.plot(costs, color=colors[i], label=f'LR = {lr}')
plt.title('Impact du Learning Rate')
plt.xlabel('Époques')
plt.ylabel('Coût (MSE)')
plt.legend()
plt.grid(True)

# 2. Test avec/sans StandardScaler
plt.subplot(1, 2, 2)
costs_with_scaler, _, _ = train_model(X_raw, y, 0.01, 1000, use_scaler=True)
costs_without_scaler, _, _ = train_model(X_raw, y, 0.01, 1000, use_scaler=False)
plt.plot(costs_with_scaler, label='Avec StandardScaler')
plt.plot(costs_without_scaler, label='Sans StandardScaler')
plt.title('Impact de StandardScaler')
plt.xlabel('Époques')
plt.ylabel('Coût (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Afficher les résultats finaux
print("\nRésultats avec différentes configurations :")
for lr in learning_rates:
    for use_scaler in [True, False]:
        costs, W, b = train_model(X_raw, y, lr, 1000, use_scaler)
        print(f"\nLearning Rate: {lr}, StandardScaler: {use_scaler}")
        print(f"Coût final: {costs[-1]:.4f}")