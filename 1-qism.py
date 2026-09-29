import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def run_linear_regression(x_data, y_data, lr=0.02, steps=1000, name="Dataset"):
    print(f"\n{'='*20} {name} {'='*20}")
    
    # 1. Simvolik (analitik) hosilalar va optimal nuqta
    w, b = sp.symbols('w b')
    n = len(x_data)
    
    # O'rtacha kvadratik xatolik (MSE Cost funksiyasi)
    C = sum((w * xv + b - yv)**2 for xv, yv in zip(x_data, y_data)) / n
    print("Xatolik funksiyasi (C):", sp.expand(C))
    
    dC_dw = sp.diff(C, w)
    dC_db = sp.diff(C, b)
    print("dC/dw:", dC_dw)
    print("dC/db:", dC_db)
    
    # w=0, b=0 nuqtasidagi boshlang'ich gradient
    init_grad_w = float(dC_dw.subs({w: 0, b: 0}))
    init_grad_b = float(dC_db.subs({w: 0, b: 0}))
    print(f"Gradient (w=0, b=0): dC/dw = {init_grad_w:.2f}, dC/db = {init_grad_b:.2f}")
    
    # Analitik yechim (hosilalarni 0 ga tenglab yechish)
    sol = sp.solve([dC_dw, dC_db], [w, b])
    w_opt, b_opt = float(sol[w]), float(sol[b])
    print(f"Analitik optimal parametrlar: w = {w_opt:.4f}, b = {b_opt:.4f}")
    
    # 2. Xatolik sirti (Contour Plot)
    cost_func = sp.lambdify((w, b), C, "numpy")
    W, B = np.meshgrid(
        np.linspace(w_opt - 5, w_opt + 5, 100),
        np.linspace(b_opt - 5, b_opt + 5, 100)
    )
    
    plt.figure(figsize=(7, 5))
    contours = plt.contour(W, B, cost_func(W, B), levels=20, cmap='viridis')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.scatter([w_opt], [b_opt], color='red', marker="*", s=150, label='Global Minimum')
    plt.title(f'Cost Function Contour Plot — {name}')
    plt.xlabel('w')
    plt.ylabel('b')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    
    # 3. Sonli usul: Gradient Descent
    w_gd, b_gd = 0.0, 0.0
    for _ in range(steps):
        pred = w_gd * x_data + b_gd
        err = pred - y_data
        dCdw = np.mean(2 * x_data * err)
        dCdb = np.mean(2 * err)
        
        w_gd -= lr * dCdw
        b_gd -= lr * dCdb
        
    print(f"Gradient Descent natijasi ({steps} qadam):")
    print(f"w = {w_gd:.3f}, b = {b_gd:.3f}, yakuniy xatolik = {cost_func(w_gd, b_gd):.3f}")


# 1-to'plam ustida sinov
x1 = np.array([1, 2, 3, 4, 5], dtype=float)
y1 = np.array([15, 25, 30, 42, 50], dtype=float)
run_linear_regression(x1, y1, lr=0.02, steps=300, name="1-to'plam")

# 2-to'plam ustida sinov
x2 = np.array([2, 6, 4, 8, 9], dtype=float)
y2 = np.array([7, 17, 20, 19, 22], dtype=float)
run_linear_regression(x2, y2, lr=0.02, steps=2000, name="2-to'plam")