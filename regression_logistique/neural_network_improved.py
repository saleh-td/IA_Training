import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SimpleNeuralNetwork:
    def __init__(self, input_size=2, hidden_size=4, learning_rate=0.01, weight_decay=0.001):
        # Configuration de base
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Historiques pour suivi
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Initialisation des paramètres
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        # Initialisation avec un facteur d'échelle plus adapté
        scale = 0.1
        self.w1 = np.random.randn(self.input_size, self.hidden_size) * scale
        self.b1 = np.zeros((1, self.hidden_size))
        self.w2 = np.random.randn(self.hidden_size, 1) * scale
        self.b2 = np.zeros((1, 1))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -15, 15)))  # Clipping pour stabilité
    
    def forward(self, x, training=True):
        # Couche cachée
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.tanh(self.z1)
        
        # Couche de sortie
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        # Cross-entropy avec epsilon pour stabilité numérique
        epsilon = 1e-15
        
        # Regularization term
        reg_loss = 0.5 * self.weight_decay * (np.sum(self.w1**2) + np.sum(self.w2**2))
        
        data_loss = -np.mean(y_true * np.log(y_pred + epsilon) + 
                             (1 - y_true) * np.log(1 - y_pred + epsilon))
        
        return data_loss + reg_loss
    
    def compute_accuracy(self, y_true, y_pred):
        # Calcul de l'accuracy (% de prédictions correctes)
        predictions = (y_pred >= 0.5).astype(int)
        return np.mean(predictions == y_true)
    
    def backward(self, x, y, output):
        m = x.shape[0]
        
        # Gradient de la couche de sortie
        dz2 = output - y
        dw2 = (1/m) * np.dot(self.a1.T, dz2) + self.weight_decay * self.w2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Gradient de la couche cachée
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * (1 - np.power(self.a1, 2))  # Dérivée de tanh
        dw1 = (1/m) * np.dot(x.T, dz1) + self.weight_decay * self.w1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Mise à jour des poids
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
    
    def train(self, x_raw, y, epochs=1000, val_split=0.3):
        # Split des données
        x_train, x_val, y_train, y_val = train_test_split(
            x_raw, y, test_size=val_split, random_state=42)
        
        # Normalisation
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        
        # Boucle d'entraînement
        for epoch in range(epochs):
            # Entraînement
            train_output = self.forward(x_train)
            self.backward(x_train, y_train, train_output)
            
            # Validation
            val_output = self.forward(x_val, training=False)
            
            # Calcul des métriques
            train_loss = self.compute_loss(y_train, train_output)
            val_loss = self.compute_loss(y_val, val_output)
            train_acc = self.compute_accuracy(y_train, train_output)
            val_acc = self.compute_accuracy(y_val, val_output)
            
            # Suivi des métriques
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(self.learning_rate)
            
            # Affichage périodique
            if epoch % 100 == 0:
                print(f"Epoch {epoch}")
                print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                print(f"Learning Rate: {self.learning_rate:.6f}\n")
                
                # Réduction du learning rate
                if epoch > 0 and epoch % 200 == 0:
                    self.learning_rate *= 0.5
                    print(f"Learning rate réduit à {self.learning_rate}")
        
        # Visualisation
        self._plot_training_history()
        return self.w1, self.w2, self.b1, self.b2
    
    def _plot_training_history(self):
        plt.figure(figsize=(15, 5))
        
        # Loss
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.title('Évolution de la Loss')
        plt.xlabel('Époques')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='Train')
        plt.plot(self.val_accuracies, label='Validation')
        plt.title('Évolution de l\'Accuracy')
        plt.xlabel('Époques')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Learning Rate
        plt.subplot(1, 3, 3)
        plt.plot(self.learning_rates)
        plt.title('Évolution du Learning Rate')
        plt.xlabel('Époques')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, x, scaler=None):
        if scaler:
            x = scaler.transform(x)
        return self.forward(x, training=False)

# Fonction pour visualiser la frontière de décision
def plot_decision_boundary(X, y, model, scaler):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    
    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[y.ravel()==0, 0], X[y.ravel()==0, 1], c='blue', label='Classe 0')
    plt.scatter(X[y.ravel()==1, 0], X[y.ravel()==1, 1], c='red', label='Classe 1')
    
    plt.title('Frontière de décision')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Données
X = np.array([
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

# Normalisation pour affichage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test avec différentes tailles de couche cachée
hidden_sizes = [2, 4, 8]
models = []

for hidden_size in hidden_sizes:
    print(f"\n=== Modèle avec {hidden_size} neurones cachés ===")
    model = SimpleNeuralNetwork(input_size=2, hidden_size=hidden_size, learning_rate=0.01, weight_decay=0.001)
    model.train(X, y, epochs=1000)
    models.append(model)
    
    # Plot decision boundary
    plot_decision_boundary(X, y, model, scaler)