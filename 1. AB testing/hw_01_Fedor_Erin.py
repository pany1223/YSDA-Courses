"""Домашнее задание №2.

1. Реализовать три функции для построения доверительных интервалов с помощью бутстрепа:
- get_ci_bootstrap_normal
- get_ci_bootstrap_percentile
- get_ci_bootstrap_pivotal

2. Придумать 4 пары наборов данных, для которых разные методы будут давать разные результаты.
Результат записать в словарь datasets, пример словаря приведён ниже в коде.
Примеры нужны только для указанных ключей. Размеры данных должны быть равны 10.
Расшифровка ключей словаря:
- 'normal' - метод проверки значимости отличий с помощью нормального доверительного интервала.
- 'percentile' - метод проверки значимости отличий с помощью ДИ на процентилях.
- 'pivotal' - метод проверки значимости отличий с помощью центрального ДИ.
- '1' - отличия средних значимы, '0' - иначе.
Пример:
'normal_1__percentile_0' - для данных по этому ключу метод проверки значимости отличий
с помощью нормального ДИ показывает наличие значимых отличий, а ДИ на процентилях нет.


За правильную реализацию каждой функции даётся 2 балла.
За каждый правильный пример данных даётся 1 балл.

Правильность работы функций будет проверятся сравнением значений с авторским решением.
Подход для проверки примеров данных реализован ниже в исполняемом коде.
"""

import numpy as np
from scipy import stats


def generate_bootstrap_data(data_one: np.array, data_two: np.array, size=10000):
    """Генерирует значения метрики, полученные с помощью бутстрепа."""
    bootstrap_data_one = np.random.choice(data_one, (len(data_one), size))
    bootstrap_data_two = np.random.choice(data_two, (len(data_two), size))
    res = bootstrap_data_two.mean(axis=0) - bootstrap_data_one.mean(axis=0)
    return res


def get_ci_bootstrap_normal(boot_metrics: np.array, pe_metric: float, alpha: float=0.05):
    """Строит нормальный доверительный интервал.

    boot_metrics - значения метрики, полученные с помощью бутстрепа
    pe_metric - точечная оценка метрики
    alpha - уровень значимости
    
    return: (left, right) - границы доверительного интервала.
    """
    std = np.std(boot_metrics)
    z = stats.norm.ppf(1 - alpha / 2)
    return (pe_metric - z * std, pe_metric + z * std)


def get_ci_bootstrap_percentile(boot_metrics: np.array, pe_metric: float, alpha: float=0.05):
    """Строит доверительный интервал на процентилях.

    boot_metrics - значения метрики, полученные с помощью бутстрепа
    pe_metric - точечная оценка метрики
    alpha - уровень значимости
    
    return: (left, right) - границы доверительного интервала.
    """
    left = np.percentile(boot_metrics, 100*alpha/2)
    right = np.percentile(boot_metrics, 100*(1-alpha/2))
    return (left, right)


def get_ci_bootstrap_pivotal(boot_metrics: np.array, pe_metric: float, alpha: float=0.05):
    """Строит центральный доверительный интервал.

    boot_metrics - значения метрики, полученные с помощью бутстрепа
    pe_metric - точечная оценка метрики
    alpha - уровень значимости
    
    return: (left, right) - границы доверительного интервала.
    """
    left = np.percentile(boot_metrics, 100*alpha/2)
    right = np.percentile(boot_metrics, 100*(1-alpha/2))
    return (2 * pe_metric - right, 2 * pe_metric - left)


# Теор. факт: если сэмпл данных плохо отражает исходное распределение, бутстреп может давать некорректную оценку
# Каждый набор протестировал на 10000 запусках generate_bootstrap_data() без фиксации seed. Указал исходные распределение и их параметры.
datasets = {
    # Exp(0.1746543314049529), Exp(0.24675200668626807)
'normal_1__percentile_0': [[0.26681867, 0.01363516, 0.04770532, 0.16274795, 0.07608133,
                            0.03002286, 0.06870122, 0.07417569, 0.09948619, 0.76075303],
                           [0.09943105, 0.49491981, 0.41428853, 0.41913064, 0.15214174,
                            0.44616539, 0.40327831, 0.31272511, 0.25952023, 0.19641968]],
    # Geom(0.06881847370036887), Geom(0.4491558142907903)                           
'normal_0__percentile_1': [[27,  2, 16,  9,  1,  2,  2, 31, 92,  2], 
                           [6, 4, 3, 2, 4, 6, 5, 3, 3, 1]],
    # Geom(0.05091249433051526), Geom(0.6709756606732553)
'percentile_1__pivotal_0': [[ 2,  2,  2, 16, 10,  2,  6, 49,  6,  6], 
                            [1, 1, 1, 1, 3, 2, 1, 1, 1, 3]],
    # Geom(0.5877949428751585), Geom(0.459523902632649)
'percentile_0__pivotal_1': [[2, 5, 1, 1, 1, 1, 2, 2, 1, 1], 
                            [3, 3, 1, 1, 4, 3, 3, 3, 2, 3]],  
}


funcname_to_func = {
    'normal': get_ci_bootstrap_normal,
    'percentile': get_ci_bootstrap_percentile,
    'pivotal': get_ci_bootstrap_pivotal
}


if __name__ == "__main__":
    for data_one, data_two in datasets.values():
        assert len(data_one) == len(data_two) == 10

    print(f'{"dataset_name": <24}|{"funcname": <11}|{"my_res": ^8}|{"res": ^5}|{"verdict": <9}\n', '-'*58)
    for dataset_name, (data_one, data_two) in datasets.items():
        pe_metric = np.mean(data_two) - np.mean(data_one)          # разность выборочных средних
        boot_metrics = generate_bootstrap_data(data_one, data_two)
        for funcname_res in dataset_name.split('__'):
            funcname, res = funcname_res.split('_')
            func = funcname_to_func[funcname]
            res = int(res)
            left, right = func(boot_metrics, pe_metric)
            my_res = 1 - int(left <= 0 <= right)
            verdict = 'correct' if res == my_res else 'error !'
            print(f'{dataset_name: <24}|{funcname: <11}|{my_res: ^8}|{res: ^5}|{verdict: <9}')

