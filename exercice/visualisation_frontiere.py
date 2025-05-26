import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Générer des données simples pour la classification
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def initialize_params(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def fit(self, X, y, epochs=100):
        # Initialiser les paramètres
        self.initialize_params(X.shape[1])
        
        # Liste pour stocker les coûts
        costs = []
        
        # Entraînement
        for epoch in range(epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(z)
            
            # Calculer le coût
            cost = -np.mean(y * np.log(predictions) + (1-y) * np.log(1-predictions))
            costs.append(cost)
            
            # Backward pass (mise à jour des poids)
            dw = np.mean(X * (predictions - y)[:, np.newaxis], axis=0)
            db = np.mean(predictions - y)
            
            # Mise à jour des paramètres
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Cost: {cost}")
        
        return costs
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z) >= 0.5

def plot_decision_boundary(X, y, model):
    # Créer une grille de points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Obtenir les prédictions pour tous les points de la grille
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Tracer la frontière de décision
    plt.contourf(xx, yy, Z, alpha=0.4)
    
    # Tracer les points de données
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], c='blue', label='Classe 0')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', label='Classe 1')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Frontière de décision de la régression logistique')
    plt.legend()
    plt.show()

def main():
    # Générer les données
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
    
    # Créer et entraîner deux modèles avec différents learning rates
    model1 = LogisticRegression(learning_rate=0.1)
    costs1 = model1.fit(X, y, epochs=200)
    
    model2 = LogisticRegression(learning_rate=0.01)
    costs2 = model2.fit(X, y, epochs=200)
    
    # Comparer les courbes d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.plot(costs1, label='learning_rate=0.1')
    plt.plot(costs2, label='learning_rate=0.01')
    plt.xlabel('Epochs')
    plt.ylabel('Coût')
    plt.title('Comparaison des learning rates')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
