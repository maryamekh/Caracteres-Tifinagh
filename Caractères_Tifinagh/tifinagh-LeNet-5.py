import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

# Architecture du modèle LeNet-5
class LeNet5Numpy:
    def __init__(self, num_classes=33, input_channels=1):
        # Poids et biais des couches
        self.w1 = np.random.randn(6, input_channels, 5, 5) * 0.1
        self.b1 = np.zeros(6)
        self.w2 = np.random.randn(16, 6, 5, 5) * 0.1
        self.b2 = np.zeros(16)
        self.w3 = np.random.randn(120, 16 * 5 * 5) * 0.1
        self.b3 = np.zeros(120)
        self.w4 = np.random.randn(84, 120) * 0.1
        self.b4 = np.zeros(84)
        self.w5 = np.random.randn(num_classes, 84) * 0.1
        self.b5 = np.zeros(num_classes)

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def avg_pool(self, x, kernel_size=2, stride=2):
        c, h, w = x.shape
        oh = (h - kernel_size) // stride + 1
        ow = (w - kernel_size) // stride + 1
        out = np.zeros((c, oh, ow))
        for ch in range(c):
            for i in range(oh):
                for j in range(ow):
                    window = x[ch, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
                    out[ch, i, j] = np.mean(window)
        return out

    def conv2d(self, x, w, b):
        out_c, _, kh, kw = w.shape
        c, h, w_in = x.shape
        oh, ow = h - kh + 1, w_in - kw + 1
        out = np.zeros((out_c, oh, ow))
        for oc in range(out_c):
            for i in range(oh):
                for j in range(ow):
                    out[oc, i, j] = np.sum(x[:, i:i + kh, j:j + kw] * w[oc]) + b[oc]
        return out

    def flatten(self, x):
        return x.flatten()

    def dense(self, x, w, b):
        return w.dot(x) + b

    def cross_entropy(self, p, y):
        return -np.log(p[y] + 1e-9)

    def forward(self, x):
        self.x = x
        self.c1 = self.relu(self.conv2d(x, self.w1, self.b1))
        self.s2 = self.avg_pool(self.c1, 2, 2)
        self.c3 = self.relu(self.conv2d(self.s2, self.w2, self.b2))
        self.s4 = self.avg_pool(self.c3, 2, 2)
        self.flat = self.flatten(self.s4)
        self.a3 = self.relu(self.dense(self.flat, self.w3, self.b3))
        self.a4 = self.relu(self.dense(self.a3, self.w4, self.b4))
        self.z5 = self.dense(self.a4, self.w5, self.b5)
        self.probs = self.softmax(self.z5)
        return self.probs

    def backward_fc(self, y):
        grad_z5 = self.probs.copy()
        grad_z5[y] -= 1
        dw5 = np.outer(grad_z5, self.a4)
        db5 = grad_z5
        grad_a4 = self.w5.T.dot(grad_z5) * self.d_relu(self.a4)
        dw4 = np.outer(grad_a4, self.a3)
        db4 = grad_a4
        grad_a3 = self.w4.T.dot(grad_a4) * self.d_relu(self.a3)
        dw3 = np.outer(grad_a3, self.flat)
        db3 = grad_a3
        return {'w5': dw5, 'b5': db5, 'w4': dw4, 'b4': db4, 'w3': dw3, 'b3': db3}

    def update(self, grads, lr, opt='sgd'):
        for k, g in grads.items():
            param = getattr(self, k)
            if opt == 'sgd':
                param -= lr * g
            elif opt == 'adam':
                # Implémentez Adam ici...
                pass
            setattr(self, k, param)

    def get_feature_maps(self, x, layer='C1'):
        if layer == 'C1':
            return self.relu(self.conv2d(x, self.w1, self.b1))
        elif layer == 'S2':
            c1 = self.relu(self.conv2d(x, self.w1, self.b1))
            return self.avg_pool(c1, 2, 2)
        elif layer == 'C3':
            c1 = self.relu(self.conv2d(x, self.w1, self.b1))
            s2 = self.avg_pool(c1, 2, 2)
            return self.relu(self.conv2d(s2, self.w2, self.b2))
        elif layer == 'S4':
            c1 = self.relu(self.conv2d(x, self.w1, self.b1))
            s2 = self.avg_pool(c1, 2, 2)
            c3 = self.relu(self.conv2d(s2, self.w2, self.b2))
            return self.avg_pool(c3, 2, 2)
        else:
            raise ValueError("Layer must be one of ['C1','S2','C3','S4']")


def load_dataset(path, limit_per_class=50):
    imgs, labs = [], []
    names = sorted(os.listdir(path))
    for lab, name in enumerate(names):
        folder = os.path.join(path, name)
        if not os.path.isdir(folder):
            continue
        files = os.listdir(folder)[:limit_per_class]
        for f in files:
            im = Image.open(os.path.join(folder, f)).convert('L').resize((32, 32))
            imgs.append(np.array(im) / 255.0)
            labs.append(lab)
    imgs = np.stack([i[np.newaxis] for i in imgs])
    return imgs, np.array(labs), names


def split_dataset(imgs, labs):
    X1, Xt, y1, yt = train_test_split(imgs, labs, test_size=0.2, stratify=labs, random_state=42)
    Xtr, Xv, ytr, yv = train_test_split(X1, y1, test_size=0.1 / 0.8, stratify=y1, random_state=42)
    return Xtr, ytr, Xv, yv, Xt, yt


def evaluate(model, X, y):
    preds = [model.forward(x) for x in X]
    preds = [np.argmax(p) for p in preds]
    accuracy = np.mean(np.array(preds) == y)
    return accuracy, preds


def show_image(img, label, class_names):
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f'Label: {class_names[label]}')
    plt.axis('off')
    plt.show()


def plot_feature_maps(feature_maps, title="Feature maps"):
    n_maps = feature_maps.shape[0]
    plt.figure(figsize=(15, 3))
    for i in range(min(n_maps, 10)):
        plt.subplot(1, 10, i + 1)
        plt.imshow(feature_maps[i], cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def train_one_epoch(model, X_train, y_train, lr, opt):
    total_loss = 0
    for x_i, y_i in zip(X_train, y_train):
        p = model.forward(x_i)
        total_loss += model.cross_entropy(p, y_i)
        grads = model.backward_fc(y_i)
        model.update(grads, lr, opt)
    return total_loss / len(X_train)


def validate(model, X_val, y_val):
    return evaluate(model, X_val, y_val)[0]


def test(model, X_test, y_test):
    acc, preds = evaluate(model, X_test, y_test)
    return acc, preds


def show_sample_images(imgs, labels, class_names, n=10):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(imgs[i].squeeze(), cmap='gray')
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.suptitle("Exemples d'images du dataset")
    plt.show()


def train_compare(path, epochs=10, lr=0.01):
    print("Chargement des données...")
    imgs, labs, class_names = load_dataset(path)
    assert imgs.shape[1:] == (1, 32, 32), "Les images doivent être de taille 1x32x32"
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(imgs, labs)
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    show_sample_images(imgs, labs, class_names, n=10)

    optims = ['sgd', 'adam', 'rmsprop', 'adagrad']
    histories = {}

    for opt in optims:
        print(f"\nOptimiseur: {opt}")
        model = LeNet5Numpy(num_classes=len(class_names))
        losses, accs = [], []
        for e in range(epochs):
            loss = train_one_epoch(model, X_train, y_train, lr, opt)
            acc = validate(model, X_val, y_val)
            losses.append(loss)
            accs.append(acc)
            print(f"Epoch {e + 1}/{epochs} Loss: {loss:.4f} Val Acc: {acc:.4f}")

        histories[opt] = (losses, accs)
        test_acc, preds = test(model, X_test, y_test)
        print(f"Test accuracy with {opt}: {test_acc:.4f}")

        # Matrice de confusion
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Matrice de confusion ({opt})')
        plt.xlabel('Prédictions')
        plt.ylabel('Vérité terrain')
        plt.show()

        # Affichage des cartes caractéristiques
        fmap = model.get_feature_maps(X_test[0], layer='C1')
        plot_feature_maps(fmap, title=f"Feature maps couche C1 ({opt})")

        pred_label = preds[0]
        print(f"Image test exemple - prédiction: {class_names[pred_label]}, vrai label: {class_names[y_test[0]]}")
        show_image(X_test[0], pred_label, class_names)

    plt.figure(figsize=(12, 5))
    for opt in optims:
        losses, _ = histories[opt]
        plt.plot(losses, label=f'{opt} loss')
    plt.title('Loss par optimiseur')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    for opt in optims:
        _, accs = histories[opt]
        plt.plot(accs, label=f'{opt} accuracy')
    plt.title('Accuracy validation par optimiseur')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    print("Exemple final d'image test avec vrai label")
    show_image(X_test[0], y_test[0], class_names)


if __name__ == "__main__":
    data_path = r"C:\Users\ùùù\Desktop\C_B\Caractères_Tifinagh\amhcd-data-64\tifinagh-images"
    train_compare(data_path, epochs=10, lr=0.01)