"""Домашнее задание №3.

Нужно написать функцию для оценки экспериментов - check_test.

На вход функции подаются данные контрольной и экспериментальной групп,
которые выбираются случайным образом из генеральной совокупности.
В данных есть значения метрики и страты для каждого объекта.

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
Код будет запускаться на процессоре - Intel Core i5, 1.6 GHz.

Проверка будет осуществляться по 10_000 ААБ экспериментам аналогичным кодом на заранее сгенерированных данных.
"""

import time
from tqdm import tqdm

import numpy as np
from scipy import stats


# Есть 5 страт
STRATS = np.arange(5)
# Средние значения и размеры страт в генеральной совокупности
STRAT_TO_MEAN = {strat: 100 + strat * 10 for strat in STRATS}
STRAT_TO_SIZE = {strat: (strat + 1) * 1000 for strat in STRATS}

STD = 10
# Данные: первый столбцец - номер страты, второй - значение метрики.
DATA = np.vstack([
    np.hstack((
        np.full((STRAT_TO_SIZE[strat], 1), strat),
        np.random.normal(STRAT_TO_MEAN[strat], STD, (STRAT_TO_SIZE[strat], 1))
    ))
    for strat in STRATS
])
# Размер выборки и эффект
SAMPLE_SIZE = 100
EFFECT = 0.045

# --------------------------- Добавлено ---------------------------
# Размер генеральной совокупности
N = sum(STRAT_TO_SIZE.values())

# Доли страт в генеральной совокупности (веса страт)
STRAT_WEIGHTS = {k: v/N for k, v in STRAT_TO_SIZE.items()}

# Стратифицированные средние, взвешенные по долям страт в генеральной совокупности
def Y_strat_mean(sample: np.array, strats: int, strat_weights: dict) -> float:
    Y_k_mean_arrays = {i: sample[:, 1][sample[:, 0] == i] for i in range(strats)}
    Y_k_mean = {k: np.mean(v) if v.size > 0 else 0 for k, v in Y_k_mean_arrays.items()}
    Y_strat_mean = sum([v * strat_weights[k] for k, v in Y_k_mean.items()])
    return Y_strat_mean

# Оценка дисперсии
def Variance_strat(sample: np.array, strats: int, strat_weights: dict) -> float:
    Var_k_arrays = {i: sample[:, 1][sample[:, 0] == i] for i in range(strats)}
    Var_k = {k: np.var(v) if v.size > 0 else 0 for k, v in Var_k_arrays.items()}
    Var_strat = sum([v * strat_weights[k] for k, v in Var_k.items()]) 
    return Var_strat
# ----------------------------------------------------------------

def get_data(population_data, effect, sample_size):
    """Возвращает данные для АА и АБ теста.

    population_data - все данные
    effect - размер эффекта (0.05 - увеличение на 5%)
    sample_size - размер выборок

    return: a_one, a_two, b
        - матрицы размера (sample_size, 2), в первом столбце номер страты,
        во втором столбце значение метрики.
    """
    indexes = np.random.choice(
        np.arange(len(population_data)),
        (3, sample_size),
        replace=False
    )
    a_one, a_two, b = [population_data[i] for i in indexes]
    b[:, 1] = b[:, 1] * (1 + effect)
    return a_one, a_two, b


def check_test(data_control: np.array, data_pilot: np.array) -> int:
    """Проверяет наличие значимого эффекта.

    data_control - матрица с данными контрольной группы.
        size = (sample_size, 2),
        в первом столбце страты,
        во втором столбце значения метрики
    data_pilot - матрица с данными экспериментальной группы.
        size = (sample_size, 2),
        в первом столбце страты,
        во втором столбце значения метрики

    return: 0 - если эффекта нет, 1 - если эффект есть"""
    control_args = {'sample': data_control,
                    'strats': len(STRATS), 
                    'strat_weights': STRAT_WEIGHTS}
    pilot_args = {'sample': data_pilot,
                  'strats': len(STRATS), 
                  'strat_weights': STRAT_WEIGHTS}
    X_strat, Y_strat = Y_strat_mean(**control_args), Y_strat_mean(**pilot_args)
    X_var, Y_var = Variance_strat(**control_args), Variance_strat(**pilot_args)

    # t-тест для средних на основании описательных статистик (альтернатива выборкам)
    _, pvalue = stats.ttest_ind_from_stats(mean1=X_strat, std1=np.sqrt(X_var), nobs1=len(data_control),
                                           mean2=Y_strat, std2=np.sqrt(Y_var), nobs2=len(data_pilot))
    return int(pvalue < 0.05)


if __name__ == "__main__":
    for _ in range(10):
        a_one, a_two, b = get_data(DATA, EFFECT, SAMPLE_SIZE)
        res = check_test(a_one, a_two)
        assert res in [0, 1], f'Функция check_test вернула не 0 или 1, а "{res}"'
        res = check_test(a_one, b)
        assert res in [0, 1], f'Функция check_test вернула не 0 или 1, а "{res}"'

    n_iter = 10000
    max_time = 60 * n_iter / 1000
    count_tp = 0
    count_fp = 0
    t1 = time.time()
    for idx in tqdm(range(n_iter)):
        a_one, a_two, b = get_data(DATA, EFFECT, SAMPLE_SIZE)
        count_fp += check_test(a_one, a_two)
        count_tp += check_test(a_one, b)
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
