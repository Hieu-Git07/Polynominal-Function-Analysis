import matplotlib as mpl
import matplotlib.pyplot as plt 
from math import sqrt
import numpy as np
import sympy as sp
# sample_dataset = --- IGNORE ---

print("Khảo sát đồ thị của hàm đa thức")

deg = [1,2,3,4,5,6,7,8,9,10]


#chuyển đổi qua latex
def convert_to_latex(expr):
    return sp.latex(expr)

#kiểm tra hàm số
print("Bạn muốn kiểm tra hàm đa thức bậc mấy? (1 - 10)")

deg_des = int(input())
if deg_des not in deg:
    print("Hàm đa thức bậc bạn muốn kiểm tra không hợp lệ!")
    exit()


x = sp.symbols('x')
ham_so = []
f= 0
for d in range(deg_des + 1):
    print("Nhập hệ của bậc", d, ": ", end="")
    deg_val = int(input())
    f += deg_val * x**d
    ham_so.append(deg_val)

f_diff = sp.diff(f, x)

#chuyển hàm sympy sang numpy
f_np = sp.lambdify(x, f, modules=['numpy'])
f_diff_np = sp.lambdify(x, f_diff, modules=['numpy'])
#tạo dữ liệu
X_vals = np.linspace(-10, 10, 1000)
Y_vals = f_np(X_vals)
X_diff_vals = np.linspace(-10, 10, 1000)
Y_diff_vals = f_diff_np(X_diff_vals)



#vẽ đồ thị hàm số
plt.figure(figsize=(10, 6))
plt.plot(X_vals, Y_vals, label=f"$Đồ thị của hàm số : y = {convert_to_latex(f)}$" )
plt.title("Đồ thị hàm đa thức") 
plt.xlabel("x")
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()

#vẽ đồ thị đạo hàm
plt.plot(X_diff_vals, Y_diff_vals, label=f"$Đồ thị của đạo hàm: y' = {convert_to_latex(f_diff)}$", color='orange')
plt.title("Đồ thị đạo hàm của hàm đa thức")
plt.xlabel("x")
plt.ylabel("Đồ thị f(x) và f'(x)")
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.show()

    



