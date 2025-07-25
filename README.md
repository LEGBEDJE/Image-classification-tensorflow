
# Projet de Classification d'Images : Fashion MNIST

Ce projet vise à construire et entraîner un réseau de neurones convolutif (CNN) pour classer des images d'articles de mode à partir du jeu de données Fashion MNIST.

## 1. Structure du Projet

Le projet est organisé comme suit :

```
/image_classification
|-- /data
|   |-- /raw
|   |   |-- x_train.npy       # Images d'entraînement brutes
|   |   |-- y_train.npy       # Labels d'entraînement brutes
|   |   |-- x_test.npy        # Images de test brutes
|   |   `-- y_test.npy        # Labels de test brutes
|   `-- /processed
|       |-- x_train_processed.npy # Images d'entraînement prétraitées
|       |-- y_train_processed.npy # Labels d'entraînement prétraitées
|       |-- x_test_processed.npy  # Images de test prétraitées
|       `-- y_test_processed.npy  # Labels de test prétraitées
|-- /models
|   `-- fashion_mnist_cnn_model.h5 # Modèle CNN entraîné
|-- /notebooks
|   `-- (aucun pour l'instant) # Peut être utilisé pour l'exploration interactive
|-- /src
|   |-- download_data.py      # Script pour télécharger les données
|   |-- preprocess_data.py    # Script pour prétraiter les données
|   `-- train_model.py        # Script pour entraîner le modèle CNN
`-- README.md
```

## 2. Comment exécuter

1.  **Clonez le dépôt** :
    ```bash
    git clone git@github.com:LEGBEDJE/Image-classification-tensorflow.git
    cd image_classification
    ```

2.  **Créez un environnement virtuel** (recommandé) :
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3.  **Installez les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

4.  **Téléchargez les données** :
    ```bash
    python3 src/download_data.py
    ```

5.  **Prétraitez les données** :
    ```bash
    python3 src/preprocess_data.py
    ```

6.  **Entraînez le modèle** :
    ```bash
    python3 src/train_model.py
    ```

## 3. Jeu de Données : Fashion MNIST

Le jeu de données Fashion MNIST est un remplacement du jeu de données MNIST original pour les problèmes de classification d'images. Il se compose de 60 000 images d'entraînement et de 10 000 images de test. Chaque image est une image en niveaux de gris de 28x28 pixels, associée à un label de 10 classes d'articles de mode (par exemple, T-shirt/top, pantalon, pull, robe, etc.).

## 4. Prétraitement des Données

Le script `src/preprocess_data.py` effectue les opérations suivantes :

*   **Normalisation des pixels** : Les valeurs des pixels des images sont mises à l'échelle de 0-255 à 0-1.
*   **Redimensionnement des images** : Les images sont redimensionnées pour inclure une dimension de canal (28x28x1), ce qui est requis par les couches convolutives de TensorFlow/Keras.

## 5. Architecture du Modèle CNN

Le modèle est un réseau de neurones convolutif séquentiel défini dans `src/train_model.py`. Il se compose de :

*   Plusieurs couches `Conv2D` avec activation ReLU pour extraire les caractéristiques des images.
*   Des couches `MaxPooling2D` pour réduire la dimensionnalité et la complexité.
*   Une couche `Flatten` pour convertir la sortie des couches convolutives en un vecteur 1D.
*   Des couches `Dense` (entièrement connectées) avec activation ReLU.
*   Une couche de sortie `Dense` avec activation `softmax` pour la classification multi-classes (10 classes).

## 6. Évaluation du Modèle

Le modèle est entraîné pendant 10 époques. Les performances sont évaluées sur l'ensemble de test en utilisant la précision (`accuracy`) et la perte (`loss`).

### Résultats

| Métrique  | Valeur |
| --------- | ------ |
| Précision | 0.91   |
| Perte     | 0.29   |

Une précision de 0.91 sur l'ensemble de test indique que le modèle est capable de classer correctement les images de Fashion MNIST dans 91% des cas.

## 7. Pistes d'Amélioration

Pour améliorer davantage les performances du modèle, les pistes suivantes pourraient être explorées :

*   **Augmentation des données** : Appliquer des transformations aléatoires aux images d'entraînement (rotation, zoom, etc.) pour augmenter la taille du jeu de données et améliorer la généralisation.
*   **Optimisation des hyperparamètres** : Utiliser des techniques comme `GridSearchCV` ou `RandomizedSearchCV` pour trouver les meilleurs hyperparamètres (nombre de filtres, taille du noyau, nombre d'époques, etc.).
*   **Architectures de modèles plus complexes** : Expérimenter avec des architectures CNN plus profondes ou pré-entraînées (transfer learning).
*   **Régularisation** : Ajouter des couches de dropout ou une régularisation L1/L2 pour réduire le surapprentissage.
