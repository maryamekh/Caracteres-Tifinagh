import os
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- Fonctions d'activation -----------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# ----------------- Classe réseau multicouche avec Adam -----------------
class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.001, l2=0.001,
                 beta1=0.9, beta2=0.999, eps=1e-8, seed=45):
        np.random.seed(seed)
        self.lr = lr
        self.l2 = l2
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.weights = []
        self.biases = []
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        for i in range(len(layer_sizes)-1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.m_w.append(np.zeros_like(w))
            self.v_w.append(np.zeros_like(w))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

    def forward(self, X):
        self.a = [X]
        self.z = []
        for i in range(len(self.weights)-1):
            z = self.a[-1] @ self.weights[i] + self.biases[i]
            self.z.append(z)
            self.a.append(relu(z))
        z = self.a[-1] @ self.weights[-1] + self.biases[-1]
        self.z.append(z)
        self.a.append(softmax(z))
        return self.a[-1]

    def loss(self, y_true, y_pred):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        ce_loss = -np.sum(y_true * np.log(y_pred)) / m
        l2_loss = sum(np.sum(w**2) for w in self.weights) * self.l2 / (2*m)
        return ce_loss + l2_loss

    def backward(self, y_true):
        m = y_true.shape[0]
        dz = self.a[-1] - y_true
        dw = []
        db = []
        for i in reversed(range(len(self.weights))):
            dw_i = (self.a[i].T @ dz) / m + (self.l2 / m) * self.weights[i]
            db_i = np.sum(dz, axis=0, keepdims=True) / m
            dw.insert(0, dw_i)
            db.insert(0, db_i)
            if i > 0:
                dz = (dz @ self.weights[i].T) * relu_derivative(self.z[i-1])

        self.t += 1
        for i in range(len(self.weights)):
            # Update weights
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dw[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dw[i] ** 2)
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            self.weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
            # Update biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db[i] ** 2)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            self.biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# ----------------- Fonctions d'augmentation -----------------
def rotate_image(img, angle_range=15):
    rows, cols = img.shape
    angle = np.random.uniform(-angle_range, angle_range)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return rotated

def translate_image(img, trans_range=5):
    rows, cols = img.shape
    tx = np.random.uniform(-trans_range, trans_range)
    ty = np.random.uniform(-trans_range, trans_range)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return translated

def load_preprocess_image(path, augment=None):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    if augment == "rotate":
        img = rotate_image(img)
    elif augment == "translate":
        img = translate_image(img)
    img = img.astype(np.float32) / 255.
    return img.flatten()

# Chargement des données 
data_dir = r'C:\Users\ùùù\Desktop\C_B\Caractères_Tifinagh\amhcd-data-64\tifinagh-images'  

paths, labels = [], []
for folder in os.listdir(data_dir):
    full_folder = os.path.join(data_dir, folder)
    if os.path.isdir(full_folder):
        for f in os.listdir(full_folder):
            paths.append(os.path.join(full_folder, f))
            labels.append(folder)

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

X = np.array([load_preprocess_image(p) for p in paths])
y = labels_encoded

ohe = OneHotEncoder(sparse_output=False)
y_oh = ohe.fit_transform(y.reshape(-1, 1))

# Validation croisée stratifiée 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
augmentation_modes = [None, "rotate", "translate"]

results = {}
losses_by_mode = {}
accuracies_by_mode = {}

for mode in augmentation_modes:
    print(f"\nMode d'augmentation: {mode if mode else 'aucune'}")
    fold_accuracies = []
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f" Pli {fold_idx}/5")
        model = NeuralNetwork([1024, 64, 32, y_oh.shape[1]], lr=0.001, l2=0.001)

        X_val = X[val_idx]
        y_val = y_oh[val_idx]
        y_train = y_oh[train_idx]

        train_losses = []
        for epoch in range(50):
            # Chargement avec augmentation au vol pour training
            X_train_aug = np.array([load_preprocess_image(paths[i], augment=mode) for i in train_idx])

            y_pred = model.forward(X_train_aug)
            loss = model.loss(y_train, y_pred)
            train_losses.append(loss)
            model.backward(y_train)

            if (epoch+1) % 10 == 0 or epoch == 49:
                val_pred = model.forward(X_val)
                acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_pred, axis=1))
                print(f"  Époque {epoch+1}: perte={loss:.4f}, précision val={acc:.4f}")

        val_pred_final = model.forward(X_val)
        acc_final = accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_pred_final, axis=1))
        fold_accuracies.append(acc_final)
        fold_losses.append(train_losses)

    results[mode if mode else 'aucune'] = np.mean(fold_accuracies)
    losses_by_mode[mode if mode else 'aucune'] = fold_losses
    accuracies_by_mode[mode if mode else 'aucune'] = fold_accuracies

print("\n=== Résultats finaux ===")
for mode, acc in results.items():
    print(f"{mode}: précision moyenne = {acc:.4f}")

#Visualisations
# Bar chart précision moyenne
plt.figure(figsize=(7,5))
plt.bar(results.keys(), results.values(), color=['gray', 'skyblue', 'lightgreen'])
plt.ylabel("Précision moyenne")
plt.title("Comparaison des modes d'augmentation")
for i, v in enumerate(results.values()):
    plt.text(i, v+0.01, f"{v:.3f}", ha='center')
plt.ylim(0,1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dyag.png")
plt.show()

# Courbes de pertes par pli
for mode, losses_list in losses_by_mode.items():
    plt.figure(figsize=(8,5))
    for i, losses in enumerate(losses_list):
        plt.plot(losses, label=f"Pli {i+1}")
    plt.title(f"Courbes de perte - {mode}")
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"loss_{mode}.png")
    plt.show()

# Boxplot des précisions finales
plt.figure(figsize=(7,5))
sns.boxplot(data=[accuracies_by_mode['aucune'], accuracies_by_mode['rotate'], accuracies_by_mode['translate']],
            palette=["gray", "skyblue", "lightgreen"])
plt.xticks([0,1,2], ["Aucune", "Rotation", "Translation"])
plt.ylabel("Précision finale de validation")
plt.title("Distribution des précisions par mode")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("val_accuracy.png")
plt.show()
