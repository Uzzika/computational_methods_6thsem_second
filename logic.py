import numpy as np
from scipy.optimize import linear_sum_assignment

def schedule_attacks(C, k):
    n = len(C)
    attacked = [0] * n  # Количество атак на каждое подразделение
    schedule_rho1 = []
    schedule_rho2 = []
    
    for j in range(n):
        # Выбираем два подразделения с максимальной мощностью, которые атакованы менее 2 раз
        available = [i for i in range(n) if attacked[i] < 2]
        if len(available) < 2:
            break  # Нельзя атаковать два разных подразделения
        
        # Сортируем по убыванию огневой мощи в текущем периоде
        available.sort(key=lambda i: -C[i][j])
        target1, target2 = available[0], available[1]
        
        schedule_rho1.append(target1)
        schedule_rho2.append(target2)
        attacked[target1] += 1
        attacked[target2] += 1
    
    return schedule_rho1, schedule_rho2

# Пример матрицы огневой мощи (3 подразделения, 3 периода)
C = [
    [5, 4, 2],
    [4, 5, 4],
    [2, 4, 5]
]
k = 2  # Коэффициент снижения мощи

rho1, rho2 = schedule_attacks(C, k)
print("Расписание ρ₁:", rho1)
print("Расписание ρ₂:", rho2)

def maximize_S7(C):
    n = len(C)
    
    # Шаг 1: Находим максимальную перестановку σ* (классическая задача о назначениях)
    row_ind, col_ind = linear_sum_assignment([[-x for x in row] for row in C])
    sigma_star = col_ind.tolist()
    S6_sigma_star = sum(C[i][sigma_star[i]] for i in range(n))
    
    # Шаг 2: Находим сопряженную перестановку σ₀ (исключаем элементы σ*)
    G = [[C[i][j] if i != sigma_star[j] else -1e9 for j in range(n)] for i in range(n)]
    row_ind, col_ind = linear_sum_assignment([[-x for x in row] for row in G])
    sigma_0 = col_ind.tolist()
    S6_sigma_0 = sum(C[i][sigma_0[i]] for i in range(n))
    
    return (sigma_star, sigma_0, S6_sigma_star + S6_sigma_0)

sigma_star, sigma_0, S7_max = maximize_S7(C)
print("σ*:", sigma_star)
print("σ₀:", sigma_0)
print("Максимальное S₇:", S7_max)