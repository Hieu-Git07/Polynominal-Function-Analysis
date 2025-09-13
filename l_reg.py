import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data set   
heights = np.array([147, 150, 153, 155, 158, 160, 163, 165, 168, 170, 173, 175, 178, 180, 183]).reshape(-1, 1)
weights = np.array([49, 50, 51, 52, 54, 56, 58, 59, 60, 66, 63, 64, 66, 67, 68])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(heights, weights, test_size=0.2, random_state=42)

# Khai báo symbol
a, b = sp.symbols('a b')

loss_function = 0
diff_lf_a = 0
diff_lf_b = 0

# Tự viết loss và đạo hàm
for xi, yi in zip(x_train.flatten(), y_train):
    loss_function += 0.5 * (yi - (a*xi + b))**2
    diff_lf_a += -xi * (yi - (a*xi + b))
    diff_lf_b += -(yi - (a*xi + b))

# Giải hệ đạo hàm = 0
solution = sp.solve([diff_lf_a, diff_lf_b], (a, b))
a_val, b_val = float(solution[a]), float(solution[b])
print("Hàm tự viết: a =", a_val, ", b =", b_val)

# Hàm có sẵn sklearn
model = LinearRegression()
model.fit(x_train, y_train)
print("Scikit-learn: a =", model.coef_[0], ", b =", model.intercept_)

# So sánh dự đoán
print("\nSo sánh dự đoán trên tập test:")
for yt, xt in zip(y_test, x_test.flatten()):
    y_pred_sklearn = model.predict([[xt]])[0]
    y_pred_manual = a_val * xt + b_val
    print(f"Chiều cao: {xt}, Actual: {yt}, Sklearn: {y_pred_sklearn:.2f}, Tự viết: {y_pred_manual:.2f}")

# Vẽ biểu đồ
plt.scatter(heights, weights, color="red", label="Data")
plt.plot(heights, model.predict(heights), color="blue", label="Sklearn line")
plt.plot(heights, a_val*heights + b_val, color="green", linestyle="--", label="Manual line")
plt.xlabel("Chiều cao (cm)")
plt.ylabel("Cân nặng (kg)")
plt.legend()
plt.show()
