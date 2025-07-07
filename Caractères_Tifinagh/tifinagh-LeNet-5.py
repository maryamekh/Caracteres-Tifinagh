import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# --- LeNet-5 Numpy ---
class LeNet5NumpyFull:
    def __init__(self, num_classes=33):
        self.w1 = np.random.randn(6, 1, 5, 5) * 0.1  # C1
        self.b1 = np.zeros(6)
        self.w2 = np.random.randn(16, 6, 5, 5) * 0.1  # C3
        self.b2 = np.zeros(16)
        self.w3 = np.random.randn(120, 16 * 5 * 5) * 0.1  # C5
        self.b3 = np.zeros(120)
        self.w4 = np.random.randn(84, 120) * 0.1  # F6
        self.b4 = np.zeros(84)
        self.w5 = np.random.randn(num_classes, 84) * 0.1  # Output
        self.b5 = np.zeros(num_classes)

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def conv2d(self, x, w, b):
        out_c, in_c, kh, kw = w.shape
        _, h, w_ = x.shape
        out_h, out_w = h - kh + 1, w_ - kw + 1
        out = np.zeros((out_c, out_h, out_w))
        for oc in range(out_c):
            for i in range(out_h):
                for j in range(out_w):
                    out[oc, i, j] = np.sum(x[:, i:i+kh, j:j+kw] * w[oc]) + b[oc]
        return out

    def maxpool2d(self, x, size=2, stride=2):
        c, h, w = x.shape
        out_h, out_w = h // stride, w // stride
        out = np.zeros((c, out_h, out_w))
        for ch in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    region = x[ch, i*stride:i*stride+size, j*stride:j*stride+size]
                    out[ch, i, j] = np.max(region)
        return out

    def flatten(self, x):
        return x.reshape(-1)

    def dense(self, x, w, b):
        return np.dot(w, x) + b

    def cross_entropy(self, pred, label):
        return -np.log(pred[label] + 1e-9)

    def forward(self, x):
        self.x = x
        self.a1 = self.relu(self.conv2d(x, self.w1, self.b1))       # C1 → 28x28x6
        self.a2 = self.maxpool2d(self.a1)                           # S2 → 14x14x6
        self.a3 = self.relu(self.conv2d(self.a2, self.w2, self.b2)) # C3 → 10x10x16
        self.a4 = self.maxpool2d(self.a3)                           # S4 → 5x5x16
        self.flat = self.flatten(self.a4)                           # C5 input
        self.a5 = self.relu(self.dense(self.flat, self.w3, self.b3))   # C5 → 120
        self.a6 = self.relu(self.dense(self.a5, self.w4, self.b4))     # F6 → 84
        self.out = self.dense(self.a6, self.w5, self.b5)               # Output
        self.probs = self.softmax(self.out)
        return self.probs

    def backward(self, label):
        d_out = self.probs.copy()
        d_out[label] -= 1

        dw5 = np.outer(d_out, self.a6)
        db5 = d_out

        da6 = self.w5.T @ d_out * self.d_relu(self.a6)
        dw4 = np.outer(da6, self.a5)
        db4 = da6

        da5 = self.w4.T @ da6 * self.d_relu(self.a5)
        dw3 = np.outer(da5, self.flat)
        db3 = da5

        return {
            'w5': dw5, 'b5': db5,
            'w4': dw4, 'b4': db4,
            'w3': dw3, 'b3': db3
        }

    def update(self, grads, lr):
        for param in grads:
            setattr(self, param, getattr(self, param) - lr * grads[param])

# --- Chargement des données ---
def load_dataset(path, limit_per_class=50):
    imgs, labs = [], []
    classes = sorted(os.listdir(path))
    for idx, cls in enumerate(classes):
        folder = os.path.join(path, cls)
        if not os.path.isdir(folder): 
            continue
        for f in os.listdir(folder)[:limit_per_class]:
            im = Image.open(os.path.join(folder, f)).convert('L').resize((32, 32))
            imgs.append(np.array(im)[None] / 255.0)
            labs.append(idx)
    return np.stack(imgs), np.array(labs), classes

def train_test_split_set(imgs, labs):
    X1, Xt, y1, yt = train_test_split(imgs, labs, test_size=0.2, stratify=labs, random_state=42)
    Xtr, Xv, ytr, yv = train_test_split(X1, y1, test_size=0.125, stratify=y1, random_state=42)
    return Xtr, ytr, Xv, yv, Xt, yt

# --- Entraînement ---
def train_model(path, epochs=10, lr=0.01, limit_per_class=50):
    X, y, classes = load_dataset(path, limit_per_class)
    Xtr, ytr, Xv, yv, Xt, yt = train_test_split_set(X, y)

    model = LeNet5NumpyFull(num_classes=len(classes))
    history = {'loss': [], 'val_acc': []}

    for e in range(epochs):
        loss = 0
        for img, label in zip(Xtr, ytr):
            prob = model.forward(img)
            loss += model.cross_entropy(prob, label)
            grads = model.backward(label)
            model.update(grads, lr)
        loss /= len(Xtr)

        val_preds = [np.argmax(model.forward(img)) for img in Xv]
        val_acc = accuracy_score(yv, val_preds)
        history['loss'].append(loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {e+1}/{epochs} — loss: {loss:.4f}, val_acc: {val_acc:.4f}")

    test_preds = [np.argmax(model.forward(img)) for img in Xt]
    test_acc = accuracy_score(yt, test_preds)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(yt, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité terrain")
    plt.title("Matrice de confusion")
    plt.show()

    # Courbes
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], color='green')
    plt.title('Validation Accuracy')
    plt.show()

    return model, classes, Xt, yt

# --- Point d'entrée ---
if __name__ == "__main__":
    data_path = r"C:\Users\ùùù\Desktop\C_B\Caractères_Tifinagh\amhcd-data-64\tifinagh-images"
    model, classes, Xt, yt = train_model(data_path, epochs=10, lr=0.005)
