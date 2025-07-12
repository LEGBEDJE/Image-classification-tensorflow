import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# Définir les chemins
processed_data_path = '/home/legbedje/Documents/datascienceproject/image_classification/data/processed/'
models_path = '/home/legbedje/Documents/datascienceproject/image_classification/models/'

# Créer le dossier des modèles si il n'existe pas
os.makedirs(models_path, exist_ok=True)

# Charger les données prétraitées
x_train = np.load(os.path.join(processed_data_path, 'x_train_processed.npy'))
y_train = np.load(os.path.join(processed_data_path, 'y_train_processed.npy'))
x_test = np.load(os.path.join(processed_data_path, 'x_test_processed.npy'))
y_test = np.load(os.path.join(processed_data_path, 'y_test_processed.npy'))

# Définir l'architecture du modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 classes pour Fashion MNIST
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Évaluer le modèle
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nPrécision sur l'ensemble de test : {accuracy:.2f}")

# Sauvegarder le modèle entraîné
model.save(os.path.join(models_path, 'fashion_mnist_cnn_model.h5'))

print("Modèle entraîné et sauvegardé dans models/fashion_mnist_cnn_model.h5")
