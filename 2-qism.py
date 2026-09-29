import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

print(f"\n{'='*20} Logistik Regressiya va ROC-AUC {'='*20}")

# 1. Matematik isbotlar (SymPy)
z = sp.symbols('z')
sig = 1 / (1 + sp.exp(-z))
d_sig = sp.diff(sig, z)
assert sp.simplify(d_sig - sig * (1 - sig)) == 0
print("✓ Sigmoid hosilasi isbotlandi: d(sigmoid)/dz = sigmoid * (1 - sigmoid)")

w, x, y, b = sp.symbols('w x y b')
p = 1 / (1 + sp.exp(-(w * x + b)))
loss = -(y * sp.log(p) + (1 - y) * sp.log(1 - p))
dloss_dw = sp.diff(loss, w)
assert sp.simplify(dloss_dw - (p - y) * x) == 0
print("✓ BCE gradienti isbotlandi: dLoss/dw = (p - y) * x")


# 2. Gradient Descent orqali modelni o'qitish
def sigmoid(val):
    return 1 / (1 + np.exp(-val))

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y_train = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=float)

w_log, b_log = 0.0, 0.0
learning_rate = 0.1
epochs = 2000

for _ in range(epochs):
    preds = sigmoid(w_log * x_train + b_log)
    err = preds - y_train
    dw = np.mean(err * x_train)
    db = np.mean(err)
    
    w_log -= learning_rate * dw
    b_log -= learning_rate * db

probabilities = sigmoid(w_log * x_train + b_log)
print(f"\nO'qitilgan parametrlar: w = {w_log:.3f}, b = {b_log:.3f}")
print("Bashorat qilingan ehtimolliklar:", np.round(probabilities, 3))


# 3. ROC egri chizig'ini qurish va AUC hisoblash
# Nuqtalar boshlanishi: (0, 0)
fpr = [0.0]
tpr = [0.0]

thresholds = np.sort(np.unique(probabilities))[::-1]

for t in thresholds:
    binary_preds = (probabilities >= t).astype(int)
    
    tp = np.sum((binary_preds == 1) & (y_train == 1))
    fp = np.sum((binary_preds == 1) & (y_train == 0))
    
    tpr.append(tp / np.sum(y_train == 1))
    fpr.append(fp / np.sum(y_train == 0))

# Nuqtalar oxiri: (1, 1)
fpr.append(1.0)
tpr.append(1.0)

# Trapetsiya qoidasi bo'yicha AUC (x=FPR, y=TPR)
auc_score = np.trapezoid(tpr, fpr)
print(f"Modelning AUC ko'rsatkichi: {auc_score:.4f}")

# 4. ROC grafigini chizish
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, marker='o', color='darkorange', lw=2, label=f'ROC egri chizig\'i (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Tasodifiy taxmin (AUC = 0.50)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right")
plt.show()