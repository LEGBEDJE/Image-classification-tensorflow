
import numpy as np
import os

# Définir les chemins
raw_data_path = '/home/legbedje/Documents/datascienceproject/image_classification/data/raw/'
processed_data_path = '/home/legbedje/Documents/datascienceproject/image_classification/data/processed/'

# Créer le dossier si il n'existe pas
os.makedirs(processed_data_path, exist_ok=True)

# Charger les données brutes
x_train = np.load(os.path.join(raw_data_path, 'x_train.npy'))
y_train = np.load(os.path.join(raw_data_path, 'y_train.npy'))
x_test = np.load(os.path.join(raw_data_path, 'x_test.npy'))
y_test = np.load(os.path.join(raw_data_path, 'y_test.npy'))

# Normaliser les images (mise à l'échelle des pixels entre 0 et 1)
x_train_normalized = x_train.astype('float32') / 255.0
x_test_normalized = x_test.astype('float32') / 255.0

# Redimensionner les images pour qu'elles soient compatibles avec les CNN (ajouter une dimension pour les canaux)
x_train_reshaped = x_train_normalized.reshape(x_train_normalized.shape[0], 28, 28, 1)
x_test_reshaped = x_test_normalized.reshape(x_test_normalized.shape[0], 28, 28, 1)

# Sauvegarder les données prétraitées
np.save(os.path.join(processed_data_path, 'x_train_processed.npy'), x_train_reshaped)
np.save(os.path.join(processed_data_path, 'y_train_processed.npy'), y_train)
np.save(os.path.join(processed_data_path, 'x_test_processed.npy'), x_test_reshaped)
np.save(os.path.join(processed_data_path, 'y_test_processed.npy'), y_test)

print("Données Fashion MNIST prétraitées et sauvegardées dans data/processed/")
