import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Neural network class
class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, l2_lambda=0.001):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.weights = []
        self.biases = []
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

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
        loss = -np.sum(y_true * np.log(y_pred)) / m
        l2 = sum(np.sum(w**2) for w in self.weights)
        loss += (self.l2_lambda / (2 * m)) * l2
        return loss

    def compute_accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

    def backward(self, X, y, outputs):
        m = X.shape[0]
        dZ = outputs - y
        self.d_weights = [None] * len(self.weights)
        self.d_biases = [None] * len(self.biases)
        self.d_weights[-1] = (self.activations[-2].T @ dZ) / m + (self.l2_lambda / m) * self.weights[-1]
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m
        for i in range(len(self.weights) - 2, -1, -1):
            dZ = (dZ @ self.weights[i+1].T) * relu_derivative(self.z_values[i])
            self.d_weights[i] = (self.activations[i].T @ dZ) / m + (self.l2_lambda / m) * self.weights[i]
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                outputs = self.forward(X_batch)
                epoch_loss += self.compute_loss(y_batch, outputs)
                self.backward(X_batch, y_batch, outputs)
            train_pred = self.forward(X)
            val_pred = self.forward(X_val)
            train_losses.append(epoch_loss / (X.shape[0] // batch_size))
            val_losses.append(self.compute_loss(y_val, val_pred))
            train_accuracies.append(self.compute_accuracy(y, train_pred))
            val_accuracies.append(self.compute_accuracy(y_val, val_pred))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss {train_losses[-1]:.4f}, Val Loss {val_losses[-1]:.4f}, Train Acc {train_accuracies[-1]:.4f}, Val Acc {val_accuracies[-1]:.4f}")
        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Set up path
data_dir = os.path.join(os.getcwd(), 'amhcd-data-64/tifinagh-images/')
print(data_dir)
print(os.getcwd())

# Load labels
try:
    labels_df = pd.read_csv(os.path.join(data_dir, 'labels-map.csv'))
    labels_df['image_path'] = labels_df['image_path'].apply(lambda p: os.path.abspath(os.path.join(data_dir, p)))
except FileNotFoundError:
    print("labels-map.csv not found. Generating labels from directories...")
    image_paths, labels = [], []
    for label_dir in os.listdir(data_dir):
        full_dir = os.path.join(data_dir, label_dir)
        if os.path.isdir(full_dir):
            for file in os.listdir(full_dir):
                image_paths.append(os.path.abspath(os.path.join(full_dir, file)))
                labels.append(label_dir)
    labels_df = pd.DataFrame({'image_path': image_paths, 'label': labels})

print(f"Loaded {len(labels_df)} samples with {labels_df['label'].nunique()} unique classes.")

# Encode labels
label_encoder = LabelEncoder()
labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
num_classes = len(label_encoder.classes_)

# Image loader
def load_and_preprocess_image(image_path, target_size=(32, 32)):
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to load image: {image_path}"
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img.flatten()

# Load all images
X = np.array([load_and_preprocess_image(p) for p in labels_df['image_path']])
y = labels_df['label_encoded'].values

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# One-hot encoding
one_hot = OneHotEncoder(sparse_output=False)
y_train_oh = one_hot.fit_transform(y_train.reshape(-1, 1))
y_val_oh = one_hot.transform(y_val.reshape(-1, 1))
y_test_oh = one_hot.transform(y_test.reshape(-1, 1))

# Train model
model = MultiClassNeuralNetwork([X.shape[1], 64, 32, num_classes], learning_rate=0.01, l2_lambda=0.001)
train_losses, val_losses, train_accuracies, val_accuracies = model.train(
    X_train, y_train_oh, X_val, y_val_oh, epochs=100, batch_size=32
)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report (Test set):")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion (Test)')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.savefig('confusion_matrix.png')
plt.close()

# Loss and accuracy curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses, label="Val Loss")
ax1.set_title("Perte")
ax1.set_xlabel("Époques")
ax1.legend()
ax2.plot(train_accuracies, label="Train Acc")
ax2.plot(val_accuracies, label="Val Acc")
ax2.set_title("Précision")
ax2.set_xlabel("Époques")
ax2.legend()
plt.tight_layout()
fig.savefig("loss_accuracy_plot.png")
plt.close()
