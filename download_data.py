
import tensorflow as tf
import numpy as np
import os

# Définir les chemins de sauvegarde
raw_data_path = '/home/legbedje/Documents/datascienceproject/image_classification/data/raw/'

# Créer le dossier si il n'existe pas
os.makedirs(raw_data_path, exist_ok=True)

# Charger le jeu de données Fashion MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Sauvegarder les données brutes
np.save(os.path.join(raw_data_path, 'x_train.npy'), x_train)
np.save(os.path.join(raw_data_path, 'y_train.npy'), y_train)
np.save(os.path.join(raw_data_path, 'x_test.npy'), x_test)
np.save(os.path.join(raw_data_path, 'y_test.npy'), y_test)

print("Données Fashion MNIST téléchargées et sauvegardées dans data/raw/")
