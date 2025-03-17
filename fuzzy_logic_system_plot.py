import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


# -------------------------
# ЗАДАНИЕ УНИВЕРСУМОВ

x_distance = np.arange(0, 6.01, 0.01)
x_direction = np.arange(-90, 91, 1)
x_turn = np.arange(-90, 91, 1)
x_velocity = np.arange(0, 101, 1)

# -------------------------
# ВХОДНЫЕ ПЕРЕМЕННЫЕ

front_close = fuzz.trimf(x_distance, [0, 1, 2])
front_medium = fuzz.trapmf(x_distance, [1, 1.5, 3.5, 4])
front_far = fuzz.gaussmf(x_distance, 4.5, 0.8)

left_close = fuzz.trimf(x_distance, [0, 1, 2])
left_medium = fuzz.trapmf(x_distance, [1, 1.5, 3.5, 4])
left_far = fuzz.gaussmf(x_distance, 4.5, 0.8)

right_close = fuzz.trimf(x_distance, [0, 1, 2])
right_medium = fuzz.trapmf(x_distance, [1, 1.5, 3.5, 4])
right_far = fuzz.gaussmf(x_distance, 4.5, 0.8)

target_left = fuzz.trimf(x_direction, [-90, -60, 0])
target_straight = fuzz.trapmf(x_direction, [-15, -5, 5, 15])
target_right = fuzz.gaussmf(x_direction, 45, 15)

# -------------------------
# ВЫХОДНЫЕ ПЕРЕМЕННЫЕ

turn_sharp_left = fuzz.trimf(x_turn, [-90, -90, -45])
turn_soft_left = fuzz.trapmf(x_turn, [-80, -60, -40, -20])
turn_none = fuzz.gaussmf(x_turn, 0, 10)
turn_soft_right = fuzz.trapmf(x_turn, [20, 40, 60, 80])
turn_sharp_right = fuzz.trimf(x_turn, [45, 90, 90])

velocity_slow = fuzz.trimf(x_velocity, [0, 0, 50])
velocity_medium = fuzz.trapmf(x_velocity, [25, 40, 60, 75])
velocity_fast = fuzz.gaussmf(x_velocity, 90, 8)

# -------------------------
# Дефаззификация

mf1 = fuzz.trimf(x_velocity, [0, 0, 50])
mf2 = fuzz.trimf(x_velocity, [25, 50, 75])
aggregated = np.fmax(mf1, mf2)
crisp_centroid = fuzz.defuzz(x_velocity, aggregated, 'centroid')
crisp_mom = fuzz.defuzz(x_velocity, aggregated, 'mom')

# -------------------------
# Визуализация функций принадлежности

fig, axs = plt.subplots(5, 1, figsize=(8, 16))


axs[0].plot(x_distance, front_close, 'r', linewidth=1.5, label='Близко (треугольная)')
axs[0].plot(x_distance, front_medium, 'g', linewidth=1.5, label='Средне (трапециевидная)')
axs[0].plot(x_distance, front_far, 'b', linewidth=1.5, label='Далеко (гауссова)')
axs[0].set_title('Расстояние_спереди')
axs[0].set_xlabel('Метры')
axs[0].set_ylabel('Степень принадлежности')
axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[0].grid(True)


axs[1].plot(x_direction, target_left, 'r', linewidth=1.5, label='Слева (треугольная)')
axs[1].plot(x_direction, target_straight, 'g', linewidth=1.5, label='Прямо (трапециевидная)')
axs[1].plot(x_direction, target_right, 'b', linewidth=1.5, label='Справа (гауссова)')
axs[1].set_title('Направление_на_цель')
axs[1].set_xlabel('Градусы')
axs[1].set_ylabel('Степень принадлежности')
axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[1].grid(True)


axs[2].plot(x_turn, turn_sharp_left, 'r', linewidth=1.5, label='Резко влево (треугольная)')
axs[2].plot(x_turn, turn_soft_left, 'g', linewidth=1.5, label='Плавно влево (трапециевидная)')
axs[2].plot(x_turn, turn_none, 'b', linewidth=1.5, label='Без поворота (гауссова)')
axs[2].plot(x_turn, turn_soft_right, 'c', linewidth=1.5, label='Плавно вправо (трапециевидная)')
axs[2].plot(x_turn, turn_sharp_right, 'm', linewidth=1.5, label='Резко вправо (треугольная)')
axs[2].set_title('Угол_поворота')
axs[2].set_xlabel('Градусы')
axs[2].set_ylabel('Степень принадлежности')
axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[2].grid(True)


axs[3].plot(x_velocity, velocity_slow, 'r', linewidth=1.5, label='Медленно (треугольная)')
axs[3].plot(x_velocity, velocity_medium, 'g', linewidth=1.5, label='Средне (трапециевидная)')
axs[3].plot(x_velocity, velocity_fast, 'b', linewidth=1.5, label='Быстро (гауссова)')
axs[3].set_title('Скорость')
axs[3].set_xlabel('Условные единицы')
axs[3].set_ylabel('Степень принадлежности')
axs[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[3].grid(True)


axs[4].plot(x_velocity, aggregated, label='Агрегированная функция принадлежности')
axs[4].axvline(crisp_centroid, color='r', linestyle='--', label='Центроид')
axs[4].axvline(crisp_mom, color='g', linestyle='--', label='Среднее максимумов')
axs[4].set_xlabel('Выходная переменная')
axs[4].set_ylabel('Степень принадлежности')
axs[4].set_title('Сравнение методов дефаззификации')
axs[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axs[4].grid(True)

plt.tight_layout()
plt.show()