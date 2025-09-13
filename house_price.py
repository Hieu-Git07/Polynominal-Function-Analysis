import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Data set
data_x = np.array([
    [1, 50, 2, 5, 3],
    [1, 80, 3, 10, 5],
    [1, 100, 4, 20, 10],
    [1, 60, 2, 15, 2],
    [1, 120, 5, 5, 8],
    [1, 90, 3, 30, 12],
    [1, 70, 3, 10, 4],
    [1, 110, 4, 8, 6],
    [1, 55, 2, 25, 7],
    [1, 95, 4, 12, 3]
], dtype=float)
data_y = np.array([120,200,250,150,300,220,180,280,140,240], dtype=float)
# Train-test split
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

# Khai báo symbol
w_symbols = sp.symbols('w0 w1 w2 w3 w4')
loss_function = 0
diff_lf_w0 = 0
diff_lf_w1 = 0
diff_lf_w2 = 0
diff_lf_w3 = 0
diff_lf_w4 = 0

diff_lf = np.empty(5, dtype = object)
diff_lf[0] = 0
diff_lf[1] = 0
diff_lf[2] = 0
diff_lf[3] = 0
diff_lf[4] = 0
# Hàm loss và đạo hàm
num_data_row = len(data_x_train)
num_data_col = len(diff_lf)

for id_row in range(num_data_row):
    pre_loss_ = float(data_y_train[id_row])
    for id_col in range(num_data_col):
        pre_loss_ -= w_symbols[id_col] * data_x_train[id_row][id_col]
        
        
    loss_function += 0.5 * pre_loss_ ** 2
    diff_lf[0] += float(-1 * pre_loss_ * data_x_train[id_row][0])
    diff_lf[1] += float(-1 * pre_loss_ * data_x_train[id_row][1])
    diff_lf[2] += float(-1 * pre_loss_ * data_x_train[id_row][2])
    diff_lf[3] += float(-1 * pre_loss_ * data_x_train[id_row][3])
    diff_lf[4] += float(-1 * pre_loss_ * data_x_train[id_row][4])
       
solution = sp.solve([diff_lf[0], diff_lf[1], diff_lf[2], diff_lf[3], diff_lf[4]], w_symbols)

w0_solve = solution[w_symbols[0]]
w1_solve = solution[w_symbols[1]]
w2_solve = solution[w_symbols[2]]
w3_solve = solution[w_symbols[3]]
w4_solve = solution[w_symbols[4]]

print("Hàm tự viết: w0 = ", w0_solve, "w1 = ", w1_solve, "w2 = ", w2_solve, "w3 = ", w3_solve, "w4 = ", w4_solve)


#Hàm Scikit-learn
model = LinearRegression()
model.fit(data_x_train, data_y_train)
print("Scikit-learn : w0 = ", model.intercept_, "w1 = ", model.coef_[1], "w2", model.coef_[2], "w3 = ", model.coef_[3], "w4", model.coef_[4])