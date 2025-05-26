import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class LogisticRegressionWithScheduling:
    def __init__(self, initial_learning_rate=0.1, thresholds=(0.3, 0.6)):
        self.initial_learning_rate = initial_learning_rate
        self.thresholds = thresholds  # (seuil1, seuil2)
        self.weights = None
        self.bias = None
        self.costs_history = []
        self.learning_rates_history = []
    
    def get_learning_rate(self, epoch, total_epochs):
        if epoch < total_epochs * self.thresholds[0]:
            return self.initial_learning_rate
        elif epoch < total_epochs * self.thresholds[1]:
            return self.initial_learning_rate / 10
        else:
            return self.initial_learning_rate / 100
    
    def fit(self, X, y, epochs=200):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        for epoch in range(epochs):
            current_lr = self.get_learning_rate(epoch, epochs)
            self.learning_rates_history.append(current_lr)
            
            z = np.dot(X, self.weights) + self.bias
            predictions = 1 / (1 + np.exp(-z))
            
            cost = -np.mean(y * np.log(predictions + 1e-8) + 
                          (1-y) * np.log(1-predictions + 1e-8))
            self.costs_history.append(cost)
            
            dw = np.mean(X * (predictions - y), axis=0).reshape(-1, 1)
            db = np.mean(predictions - y)
            
            self.weights -= current_lr * dw
            self.bias -= current_lr * db
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.6f}, LR: {current_lr:.6f}")
        
        return self.costs_history

# Données
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
y = np.array([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)

# Standardisation
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Créer trois modèles avec différents seuils
thresholds = [
    (0.3, 0.6),  # Original
    (0.2, 0.5),  # Plus rapide
    (0.4, 0.7)   # Plus lent
]
models = []

plt.figure(figsize=(15, 10))

# Entraîner les trois modèles
for i, (t1, t2) in enumerate(thresholds):
    print(f"\nTest avec seuils {t1*100}% et {t2*100}%:")
    model = LogisticRegressionWithScheduling(initial_learning_rate=0.1, thresholds=(t1, t2))
    costs = model.fit(X, y, epochs=200)
    models.append(model)
    
    # Courbe d'apprentissage
    plt.subplot(2, 2, 1)
    plt.plot(costs, label=f'Seuils ({t1*100}%, {t2*100}%)')
    plt.title('Comparaison des courbes d\'apprentissage')
    plt.xlabel('Époques')
    plt.ylabel('Coût')
    plt.grid(True)
    plt.legend()
    
    # Évolution du learning rate
    plt.subplot(2, 2, 2)
    plt.plot(model.learning_rates_history, label=f'Seuils ({t1*100}%, {t2*100}%)')
    plt.title('Évolution des Learning Rates')
    plt.xlabel('Époques')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# Afficher les coûts finaux
print("\nRésultats finaux:")
for i, (t1, t2) in enumerate(thresholds):
    final_cost = models[i].costs_history[-1]
    print(f"Seuils ({t1*100}%, {t2*100}%) - Coût final: {final_cost:.6f}")