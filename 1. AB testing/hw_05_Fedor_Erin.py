"""Домашнее задание.

Нужно написать функцию для оценки экспериментов - check_test.

На вход функции подаются данные контрольной и экспериментальной групп за 9 недель,
эксперимент проводился только на последней неделе.

Оценка качества.
Будем считать, что за каждый найденный реальный эффект мы получаем 1 единицу денег (TP),
а за каждое неверно внедрённое изменение теряем 1 единицу денег (FP).
Всего будет проведено N аа и аб экспериментов, значит максимальный "выигрыш" равен N.
Далее посчитаем долю полученного выигрыша от максимального: P = (TP - FP) / N.
Оценка будет определяться по формуле:
score_ = int(np.ceil((P - 0.63) / 0.03))
score = np.clip(score_, 0, 10)

Обратите внимание на скорость работы функции. В коде есть ограничение по времени.
Чтобы все тесты успели пройти проверку, нужно чтобы 1000 ААБ экспериментов оценивалось не более 1 минуты.
Скорость работы можно проверить в Colab'е https://colab.research.google.com .

Проверка будет осуществляться по 10_000 ААБ экспериментам аналогичным кодом.
"""

import time
from tqdm import tqdm

import numpy as np
from scipy import stats


EFFECT = 5
SAMPLE_SIZE = 100

# --------------------------- Добавлено ---------------------------

from scipy.stats import ttest_ind
from sklearn.linear_model import RidgeCV

np.random.seed(42)

def calculate_theta(y_control, y_pilot, y_control_cov, y_pilot_cov) -> float:
    """Вычисляем Theta.
    
    y_control - значения метрики во время пилота на контрольной группе
    y_pilot - значения метрики во время пилота на пилотной группе
    y_control_cov - значения ковариант на контрольной группе
    y_pilot_cov - значения ковариант на пилотной группе
    """
    y = np.hstack([y_control, y_pilot])
    y_cov = np.hstack([y_control_cov, y_pilot_cov])
    covariance = np.cov(y_cov, y)[0, 1]
    variance = y_cov.var()
    theta = covariance / variance
    return theta

def get_ml_predicted_covariants(data_control: np.array, data_pilot: np.array) -> tuple:
    """
    Обучение регрессии на 8-ую неделю по признакам 1-7 недель
    Ковариант - прогноз моделью 9-ой недели по признакам 2-8 недель
    """
    alphas = [1e-2, 1e-1, 1, 5, 10, 20]
    model = RidgeCV(alphas=alphas, normalize=True)
    
    data = np.vstack([data_control, data_pilot])
    
    model.fit(data[:, :7], data[:, -2])

    data_control_predicted = model.predict(data_control[:, 1:8])
    data_pilot_predicted = model.predict(data_pilot[:, 1:8])
    
    theta = calculate_theta(
        data_control[:, 8], data_pilot[:, 8],
        data_control_predicted, data_pilot_predicted
    )

    data_control_cv = data_control[:, 8] - theta * data_control_predicted
    data_pilot_cv = data_pilot[:, 8] - theta * data_pilot_predicted
    
    return (data_control_cv, data_pilot_cv)

# ----------------------------------------------------------------

def get_data(effect, sample_size):
    """Возвращает данные для АА и АБ теста.

    return: a_one, a_two, b
        - матрицы размера (sample_size, 9),
        первые 8 столбцов содержат исторические значения метрики,
        последний столбец содержит значение метрики во время эксперимента.
    """
    list_data = []
    for _ in range(3):
        means = np.random.normal(100, 10, (sample_size, 1))
        trends = np.random.uniform(-10, 10, (sample_size, 1))
        noise = np.random.normal(0, 7.5, (sample_size, 9))
        list_data.append(
            means
            + trends * np.arange(-4, 5).reshape(1, -1)
            + noise
        )
    a_one, a_two, b = list_data
    b[:, -1] += effect
    return a_one, a_two, b


def check_test(data_control: np.array, data_pilot: np.array) -> int:
    """Проверяет наличие значимого эффекта.

    data_control - матрица с данными контрольной группы.
        size = (sample_size, 9),
        в последнем столбце значения метрики во время эксперимента,
        первые 8 столбцов содержат значения метрики до эксперимента.
    data_pilot - матрица с данными экспериментальной группы.
        size = (sample_size, 9),
        в последнем столбце значения метрики во время эксперимента,
        первые 8 столбцов содержат значения метрики до эксперимента.

    return: 0 - если эффекта нет, 1 - если эффект есть
    """
    data_control_cv, data_pilot_cv = get_ml_predicted_covariants(data_control, data_pilot)

    _, pvalue_cuped = ttest_ind(data_control_cv, data_pilot_cv)

    return int(pvalue_cuped < 0.05)


if __name__ == "__main__":
    for _ in range(10):
        a_one, a_two, b = get_data(EFFECT, SAMPLE_SIZE)
        res = check_test(a_one.copy(), a_two.copy())
        assert res in [0, 1], f'Функция check_test вернула не 0 или 1, а "{res}"'
        res = check_test(a_one.copy(), b.copy())
        assert res in [0, 1], f'Функция check_test вернула не 0 или 1, а "{res}"'

    n_iter = 10000
    max_time = 60 * n_iter / 1000
    count_tp = 0
    count_fp = 0
    t1 = time.time()
    for idx in tqdm(range(n_iter)):
        a_one, a_two, b = get_data(EFFECT, SAMPLE_SIZE)
        count_fp += check_test(a_one.copy(), a_two.copy())
        count_tp += check_test(a_one.copy(), b.copy())
        t2 = time.time()
        if t2 - t1 > max_time:
            print('Долго считает! На 1000 ААБ экспериментов более 1 минуты.')
            print(f'Успел {idx} из {n_iter}.')
            break
    else:
        print(f'Время на оценку 1000 ААБ экспериментов: {(t2 - t1) / n_iter * 1000:0.2f} сек')
    your_money = count_tp - count_fp
    max_money = n_iter
    part_money = your_money / max_money
    score_ = int(np.ceil((part_money - 0.63) / 0.03))
    score = np.clip(score_, 0, 10)
    print(f'part_money = {part_money}')
    print(f'score = {score}')
