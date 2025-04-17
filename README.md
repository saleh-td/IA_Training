# IA Training

Ce projet contient des implémentations d'algorithmes d'apprentissage automatique pour l'entraînement et la compréhension des concepts fondamentaux de l'IA.

## Structure du Projet

### Régression Linéaire (`regression_lineaire/`)
- Implémentation de base de la régression linéaire
- Visualisation de l'apprentissage avec différents learning rates
- Tests avec différentes configurations

### Régression Logistique (`regression_logistique/`)
- Classification binaire
- Utilisation de la fonction sigmoid
- Tests avec différents paramètres (learning rate, epochs)
- Comparaison avec/sans StandardScaler

## Installation et Utilisation

```bash
git clone https://github.com/saleh-td/IA_Training.git
cd IA_Training

# Pour exécuter la régression logistique
python regression_logistique/logistic_manual.py

# Pour exécuter la régression linéaire
python regression_lineaire/main.py