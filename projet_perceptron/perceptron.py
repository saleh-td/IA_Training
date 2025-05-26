import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors_ = []  # Pour suivre les erreurs pendant l'apprentissage

    def fit(self, X, y):
        # Initialisation des poids et du biais
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Apprentissage
        for _ in range(self.n_iterations):
            errors = 0
            for xi, target in zip(X, y):
                # Calcul de la prédiction
                prediction = self.predict_one(xi)
                
                # Mise à jour si erreur
                if prediction != target:
                    # Mise à jour des poids
                    update = self.learning_rate * (target - prediction)
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
            
            self.errors_.append(errors)
            
            # Si aucune erreur, on arrête l'apprentissage
            if errors == 0:
                break

    def predict_one(self, X):
        # Calcul de la somme pondérée
        linear_output = np.dot(X, self.weights) + self.bias
        # Application de la fonction de seuil
        return 1 if linear_output >= 0 else 0

    def predict(self, X):
        return np.array([self.predict_one(xi) for xi in X])

# Fonction pour visualiser les résultats
def plot_decision_boundary(X, y, model):
    # Définir la grille
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))

    # Prédire pour chaque point de la grille
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Tracer la frontière de décision
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='blue', label='Classe 0')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='red', label='Classe 1')
    plt.title('Frontière de décision du Perceptron')
    plt.xlabel('Caractéristique 1')
    plt.ylabel('Caractéristique 2')
    plt.legend()
    
    plt.grid(True)  # Ajout d'une grille
    plt.axis('equal')  # Pour avoir des échelles égales

# Test du Perceptron
def main():
    # Création de données synthétiques
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
    
    # Ajoutez ces lignes après la création des données
    print("Forme des données :", X.shape)
    print("\nQuelques points de données :")
    for i in range(5):
        print(f"Point {i + 1}: {X[i]}, Classe: {y[i]}")

    # Création et entraînement du modèle
    perceptron = Perceptron(learning_rate=0.01, n_iterations=100)
    perceptron.fit(X, y)

    # Visualisation des résultats
    plot_decision_boundary(X, y, perceptron)
    
    # Affichage des erreurs d'apprentissage
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(perceptron.errors_)), perceptron.errors_)
    plt.title('Nombre d\'erreurs par itération')
    plt.xlabel('Itérations')
    plt.ylabel('Nombre d\'erreurs')
    plt.show()

if __name__ == "__main__":
    main()