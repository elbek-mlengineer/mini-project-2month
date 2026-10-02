"""
Machine Learning asoslari: Chiziqli va Logistik Regressiyani matematik
hamda amaliy (Gradient Descent) usullarda amalga oshirish.

Ushbu modul quyidagi asosiy bosqichlarni o'z ichiga oladi:
1. Chiziqli regressiyada analitik (SymPy) va iterativ (Gradient Descent) optimallashtirish.
2. Logistik regressiyada Sigmoid hosilasi va Binary Cross-Entropy xatolik funksiyasi gradiyenti.
3. Gradient Descent orqali logistik regressiya parametrlarini baholash.
4. ROC egri chizig'i koordinatalarini va AUC (Area Under Curve) ko'rsatkichini
   noldan (from scratch) hamda scikit-learn yordamida hisoblash.
"""

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import sympy as sp


# ==============================================================================
# 1-QISM: CHIZIQLI REGRESSIYA (LINEAR REGRESSION)
# ==============================================================================

def run_linear_regression() -> None:
    """Chiziqli regressiya modelini matematik (analitik) va iterativ usulda tahlil qiladi."""
    print("=" * 70)
    print("1-QISM: CHIZIQLI REGRESSIYA TAHLILI")
    print("=" * 70)

    # Kiruvchi ma'lumotlar to'plami (Toy dataset)
    x_data: np.ndarray = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y_data: np.ndarray = np.array([15, 25, 30, 42, 50], dtype=np.float64)
    m: int = len(x_data)

    # 1.1. Ramziy (Simvolik) hisoblashlar — SymPy orqali
    w, b = sp.symbols("w b", real=True)

    # Mean Squared Error (MSE) xatolik funksiyasini shakllantirish
    cost_expr = sum((w * xv + b - yv) ** 2 for xv, yv in zip(x_data, y_data)) / m
    print(f"\n[SymPy] Xatolik funksiyasi (MSE): \nC(w, b) = {sp.expand(cost_expr)}")

    # Xatolik funksiyasidan parametrlar bo'yicha xususiy hosilalar (Gradiyentlar)
    dC_dw = sp.diff(cost_expr, w)
    dC_db = sp.diff(cost_expr, b)
    print(f"\n[SymPy] dC/dw = {dC_dw}")
    print(f"[SymPy] dC/db = {dC_db}")

    # Boshlang'ich nuqtadagi (w=0, b=0) gradiyent qiymatlarini baholash
    grad_w_init = float(dC_dw.subs({w: 0, b: 0}))
    grad_b_init = float(dC_db.subs({w: 0, b: 0}))
    print(f"[SymPy] w=0, b=0 nuqtasidagi boshlang'ich gradiyent: dC/dw={grad_w_init:.2f}, dC/db={grad_b_init:.2f}")

    # Xatolik funksiyasini NumPy massivlari bilan tez ishlaydigan funksiyaga o'girish
    cost_func = sp.lambdify((w, b), cost_expr, modules="numpy")

    # Analitik yechim: Hosilalarni 0 ga tenglab global minimum (optimal) nuqtani topish
    solution = sp.solve([dC_dw, dC_db], [w, b])
    w_opt: float = float(solution[w])
    b_opt: float = float(solution[b])
    print(f"\n[Analitik Yechim] Optimal parametrlar: w*={w_opt:.3f}, b*={b_opt:.3f}")

    # 1.2. Gradient Descent orqali optimallashtirish
    w_gd, b_gd = 0.0, 0.0
    learning_rate: float = 0.02
    epochs: int = 200

    for _ in range(epochs):
        predictions = w_gd * x_data + b_gd
        errors = predictions - y_data

        # Vektorlashgan gradiyent hisobi: d(MSE)/dw va d(MSE)/db
        dCdw_val = np.mean(2 * x_data * errors)
        dCdb_val = np.mean(2 * errors)

        # Parametrlarni yangilash qoidasi
        w_gd -= learning_rate * dCdw_val
        b_gd -= learning_rate * dCdb_val

    print(
        f"[Gradient Descent] {epochs} epoch natijasi: "
        f"w={w_gd:.3f}, b={b_gd:.3f}, Cost={cost_func(w_gd, b_gd):.3f}"
    )

    # 1.3. Xatolik sirtining izochiziqlari (Contour plot) vizualizatsiyasi
    w_vals = np.linspace(w_opt - 5, w_opt + 5, 100)
    b_vals = np.linspace(b_opt - 5, b_opt + 5, 100)
    W, B = np.meshgrid(w_vals, b_vals)

    plt.figure(figsize=(8, 6))
    contour = plt.contour(W, B, cost_func(W, B), levels=20, cmap="viridis")
    plt.clabel(contour, inline=True, fontsize=8)
    plt.scatter([w_opt], [b_opt], color="red", marker="*", s=150, label="Global Minimum (w*, b*)")
    plt.scatter([w_gd], [b_gd], color="blue", marker="o", s=50, label="Gradient Descent natijasi")
    plt.title("MSE Xatolik Funksiyasining Kontur Grafigi")
    plt.xlabel("Vazn (W)")
    plt.ylabel("Siljish (B)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 2-QISM: LOGISTIK REGRESSIYA VA ROC-AUC TAHLILI
# ==============================================================================

def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Matematik Sigmoid (logistic) faollashtirish funksiyasi."""
    return 1.0 / (1.0 + np.exp(-z))


def calculate_manual_auc(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, list, list]:
    """
    Klassifikatsiya ehtimolliklari asosida ROC egri chizig'i koordinatalarini (TPR, FPR)
    va Trapezoid integrallash usuli orqali AUC qiymatini noldan hisoblaydi.
    """
    tpr: list[float] = [0.0]
    fpr: list[float] = [0.0]

    num_positives = np.sum(y_true == 1)
    num_negatives = np.sum(y_true == 0)

    # Barcha unikal ehtimolliklar chegara (threshold) sifatida kamayish tartibida olinadi
    thresholds = np.sort(np.unique(y_probs))[::-1]

    for threshold in thresholds:
        pred_labels = (y_probs >= threshold).astype(int)
        
        # True Positive Rate va False Positive Rate hisobi
        true_positives = np.sum((pred_labels == 1) & (y_true == 1))
        false_positives = np.sum((pred_labels == 1) & (y_true == 0))

        tpr.append(true_positives / num_positives)
        fpr.append(false_positives / num_negatives)

    # Egrichiziq ostidagi maydon (AUC) trapetsiyalar usulida integrallanadi
    auc_val = float(np.trapezoid(tpr, fpr))
    return auc_val, tpr, fpr


def run_logistic_regression() -> None:
    """Logistik regressiya nazariyasi, model o'qitish va ROC-AUC baholash jarayoni."""
    print("\n" + "=" * 70)
    print("2-QISM: LOGISTIK REGRESSIYA VA METRIKALAR")
    print("=" * 70)

    # 2.1. Matematik tekshiruv — Sigmoid hosilasi va Chain Rule
    z = sp.symbols("z", real=True)
    sym_sigmoid = 1 / (1 + sp.exp(-z))
    d_sym_sigmoid = sp.diff(sym_sigmoid, z)

    # Isbot 1: d(Sigmoid)/dz == Sigmoid * (1 - Sigmoid)
    proof_1 = sp.simplify(d_sym_sigmoid - sym_sigmoid * (1 - sym_sigmoid))
    print(f"\n[Nazariya] Sigmoid hosilasi: {sp.simplify(d_sym_sigmoid)}")
    print(f"[Nazariya] d(Sigmoid)/dz == Sigmoid*(1-Sigmoid) isboti (0 bo'lsa to'g'ri): {proof_1}")

    # Isbot 2: Binary Cross-Entropy (BCE) gradiyenti: dLoss/dw == (p - y) * x
    w, x, y, b = sp.symbols("w x y b", real=True)
    p = 1 / (1 + sp.exp(-(w * x + b)))
    loss = -(y * sp.log(p) + (1 - y) * sp.log(1 - p))
    d_loss_dw = sp.diff(loss, w)
    proof_2 = sp.simplify(d_loss_dw - (p - y) * x)
    print(f"[Nazariya] Chain Rule isboti: dLoss/dw - (p - y)*x (0 bo'lsa to'g'ri): {proof_2}")

    # 2.2. Logistik regressiyani o'qitish (Gradient Descent)
    x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
    y_train = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int32)

    w_val: float = 0.0
    b_val: float = 0.0
    learning_rate: float = 0.1
    epochs: int = 2000

    for _ in range(epochs):
        predictions = sigmoid(w_val * x_train + b_val)
        errors = predictions - y_train

        # Gradiyentlarni hisoblash
        dw = np.mean(errors * x_train)
        db = np.mean(errors)

        # Qadam tashlash
        w_val -= learning_rate * dw
        b_val -= learning_rate * db

    y_probabilities = sigmoid(w_val * x_train + b_val)
    print(f"\n[O'qitish] Model yakuniy parametrlari: w={w_val:.3f}, b={b_val:.3f}")
    print(f"[O'qitish] Bashorat qilingan ehtimolliklar: {np.round(y_probabilities, 3)}")

    # 2.3. ROC-AUC hisoblash va natijalarni tekshirish
    manual_auc, _, _ = calculate_manual_auc(y_train, y_probabilities)
    sklearn_auc = float(roc_auc_score(y_train, y_probabilities))

    print(f"\n[Metrika] Noldan hisoblangan AUC: {manual_auc:.4f}")
    print(f"[Metrika] Scikit-learn orqali hisoblangan AUC: {sklearn_auc:.4f}")
    print(f"[Validatsiya] Natijalar o'zaro mos keldimi?: {np.isclose(manual_auc, sklearn_auc)}")


# ==============================================================================
# DASTURNI ISHGA TUSHIRISH (ENTRY POINT)
# ==============================================================================

if __name__ == "__main__":
    run_linear_regression()
    run_logistic_regression()