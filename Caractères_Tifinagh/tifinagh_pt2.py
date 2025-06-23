import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Fonctions d'activation
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)
def softmax(x): 
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Réseau de neurones avec Adam
class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.001, l2_lambda=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weights = []
        self.biases = []
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []
        self.t = 0

        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.m_w.append(np.zeros_like(w))
            self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            self.activations.append(relu(z))
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(softmax(z))
        return self.activations[-1]

    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        m = y_true.shape[0]
        cross_entropy = -np.sum(y_true * np.log(y_pred)) / m
        l2 = sum(np.sum(w ** 2) for w in self.weights)
        return cross_entropy + (self.l2_lambda / (2 * m)) * l2

    def backward(self, X, y, outputs):
        m = X.shape[0]
        grads_w = [0] * len(self.weights)
        grads_b = [0] * len(self.biases)
        dZ = outputs - y

        grads_w[-1] = (self.activations[-2].T @ dZ) / m + (self.l2_lambda / m) * self.weights[-1]
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True) / m

        for i in range(len(self.weights) - 2, -1, -1):
            dZ = (dZ @ self.weights[i + 1].T) * relu_derivative(self.z_values[i])
            grads_w[i] = (self.activations[i].T @ dZ) / m + (self.l2_lambda / m) * self.weights[i]
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / m

        self.t += 1
        for i in range(len(self.weights)):
            # Update moments
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i] ** 2)
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)

            # Bias correction
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Prétraitement des images
def load_and_preprocess_image(image_path, target_size=(32, 32)):
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to load image: {image_path}"
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img.flatten()

# Chargement du dataset
data_dir = os.path.join(os.getcwd(), 'amhcd-data-64/tifinagh-images/')
image_paths, labels = [], []

for label_dir in os.listdir(data_dir):
    full_path = os.path.join(data_dir, label_dir)
    if os.path.isdir(full_path):
        for fname in os.listdir(full_path):
            image_paths.append(os.path.join(full_path, fname))
            labels.append(label_dir)

labels_df = pd.DataFrame({'image_path': image_paths, 'label': labels})
label_encoder = LabelEncoder()
labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])

X = np.array([load_and_preprocess_image(p) for p in labels_df['image_path']])
y = labels_df['label_encoded'].values
one_hot = OneHotEncoder(sparse_output=False)
y_oh = one_hot.fit_transform(y.reshape(-1, 1))

# Validation croisée K-fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
    print(f"\n----- Fold {fold} -----")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_oh[train_idx], y_oh[val_idx]
    model = MultiClassNeuralNetwork([X.shape[1], 64, 32, y_oh.shape[1]])

    train_losses = []
    val_accuracies = []

    for epoch in range(50):
        outputs = model.forward(X_train)
        loss = model.compute_loss(y_train, outputs)
        model.backward(X_train, y_train, outputs)

        train_losses.append(loss)

        if epoch % 10 == 0 or epoch == 49:
            val_pred_probs = model.forward(X_val)
            val_pred_labels = np.argmax(val_pred_probs, axis=1)
            val_true_labels = np.argmax(y_val, axis=1)
            val_acc = accuracy_score(val_true_labels, val_pred_labels)
            val_accuracies.append(val_acc)
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Val Acc = {val_acc:.4f}")

    # Affichage des courbes perte et précision
    epochs = range(1, 51)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold} - Training Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    val_epochs = [0,10,20,30,40,49]
    val_epochs = [e+1 for e in val_epochs]
    plt.plot(val_epochs, val_accuracies, label='Validation Accuracy', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Fold {fold} - Validation Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Évaluation finale
    y_pred = model.predict(X_val)
    print("\nClassification Report:")
    print(classification_report(y[val_idx], y_pred, target_names=label_encoder.classes_))

    # Matrice de confusion
    cm = confusion_matrix(y[val_idx], y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Fold {fold} - Confusion Matrix')
    plt.show()
 







