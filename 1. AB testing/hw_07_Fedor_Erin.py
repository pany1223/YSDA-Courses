"""Домашнее задание.

Нужно написать функцию для оценки множества взаимоисключающих экспериментов - check_test.
Взаимоисключающие эксперименты означают, что мы можем выбрать только один вариант или
отказаться от изменений вовсе.

На вход функции подаются данные контрольной и экспериментальной групп множества независимых
экспериментов, их количество может меняться от 5 до 20. Функция должна вернуть номер эксперимента,
который рекомендуется к внедрению, то есть в нём должны быть найдены значимые ПОЛОЖИТЕЛЬНЫЕ изменения.
Если значимых улучшений не обнаружено, то функция возвращает -1.

Оценка качества.
Будем считать, что за каждый найденный реальный эффект мы получаем 1 единицу денег (TP),
а за каждое неверно внедрённое изменение теряем 1 единицу денег (FP).
Всего будет проведено N аа и аб экспериментов, значит максимальный "выигрыш" равен N.
Далее посчитаем долю полученного выигрыша от максимального: P = (TP - FP) / N.
Оценка будет определяться по формуле:
score_ = int(np.ceil((P - 0.48) / 0.03))
score = np.clip(score_, 0, 10)

Обратите внимание на скорость работы функции. В коде есть ограничение по времени.
Чтобы все тесты успели пройти проверку, нужно чтобы 1000 ААБ экспериментов оценивалось не более 1 минуты.
Скорость работы можно проверить в Colab'е https://colab.research.google.com .

Проверка будет осуществляться по 10_000 ААБ экспериментам аналогичным кодом.
"""

from copy import deepcopy
import time
from tqdm import tqdm

import numpy as np
from scipy import stats
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests 


EFFECT = 5
SAMPLE_SIZE = 100


def get_data(count_experiment, effect, sample_size):
    """Возвращает данные для АА и АБ теста.

    return: data_aa, data_ab, experiment_with_effect
        - данные для АА и АБ экспериментов.
    """
    data_aa = []
    for _ in range(count_experiment):
        data_aa.append({
            'control': np.random.normal(100, 10, sample_size),
            'pilot': np.random.normal(100, 10, sample_size)
        })

    count_negative = 3
    neg_effects = np.random.uniform(-effect, 0, count_negative)
    neg_indexes = np.random.randint(0, count_experiment, count_negative)
    for neg_effect, neg_index in zip(neg_effects, neg_indexes):
        data_aa[neg_index] = {
            'control': np.random.normal(100, 10, sample_size),
            'pilot': np.random.normal(100 + neg_effect, 10, sample_size)
        }

    data_ab = deepcopy(data_aa)
    experiment_with_effect = np.random.randint(0, count_experiment)
    data_ab[experiment_with_effect] = {
        'control': np.random.normal(100, 10, sample_size),
        'pilot': np.random.normal(100 + effect, 10, sample_size)
    }
    return data_aa, data_ab, experiment_with_effect


def check_test(data: list) -> int:
    """Проверяет наличие значимого эффекта.

    data - список с данными экспериментов, каждый элемент - словарь со значениями метрики
        в контрольной и экспериментальной группах.
        Пример с двумя экспериментами: [
            {'control': np.array([1, 2, 3]), 'pilot': np.array([7, 5, 6])},
            {'control': np.array([2, 2, 1]), 'pilot': np.array([3, 6, 5])},
        ]

    return:
        если положительного эффекта нет: -1;
        если положительный эффект есть, возвращает номер эксперимента рекомендуемого для внедрения
            из диапазона [0, len(data)-1].
    """
    # <YOUR_CODE_HERE>
    pvalues = np.array([])
    diffs = np.array([])
    
    for data_dict in data:
        control, pilot = data_dict['control'], data_dict['pilot']
        _, pvalue = ttest_ind(control, pilot)
        pvalues = np.append(pvalues, pvalue)
        diffs = np.append(diffs, np.mean(pilot) - np.mean(control))

    # изменим стандартный alpha = 0.05 на 0.10, чтобы сделать детекцию эффектов менее строгой
    res_bonferroni = multipletests(pvalues, alpha=0.1, method='bonferroni')[0]
    results = zip(pvalues, res_bonferroni, diffs)

    # учитываем только эффект > 0, находим индекс эксперимента с минимальным pvalue - самый заметный эффект
    if any(res_bonferroni):
        positive_effect_indices = np.where((diffs > 0) & (res_bonferroni == True))[0]
        positive_effect_pvalues = pvalues[positive_effect_indices]

        if len(positive_effect_pvalues) > 0:
            min_p_index_relative = np.argmin(positive_effect_pvalues)
            min_p_index_absolute = positive_effect_indices[min_p_index_relative]
            return min_p_index_absolute

    return -1


if __name__ == "__main__":
    count_experiment_ = 10
    for _ in range(10):
        data_aa, data_ab, _ = get_data(count_experiment_, EFFECT, SAMPLE_SIZE)
        for data in [data_aa, data_ab]:
            res = check_test(data)
            msg = f'Функция check_test вернула значение не из диапазона [-1, count_experiment_ - 1], а "{res}"'
            assert res in range(-1, count_experiment_), msg

    n_iter = 10000
    max_time = 60 * n_iter / 1000
    count_tp = 0
    count_fp = 0
    array_count_experiment = np.random.randint(5, 21, n_iter)
    t1 = time.time()
    for idx, count_experiment in tqdm(enumerate(array_count_experiment)):
        data_aa, data_ab, experiment_with_effect = get_data(count_experiment, EFFECT, SAMPLE_SIZE)
        res_aa = check_test(data_aa)
        count_fp += res_aa != -1
        res_ab = check_test(data_ab)
        count_fp += (res_ab != -1) and (res_ab != experiment_with_effect)
        count_tp += res_ab == experiment_with_effect
        t2 = time.time()
        if t2 - t1 > max_time:
            print('Долго считает! На 1000 ААБ экспериментов более 1 минут.')
            print(f'Успел {idx} из {n_iter}.')
            break
    else:
        print(f'Время на оценку 1000 ААБ экспериментов: {(t2 - t1) / n_iter * 1000:0.2f} сек')
    your_money = count_tp - count_fp
    max_money = n_iter
    part_money = your_money / max_money
    score_ = int(np.ceil((part_money - 0.48) / 0.03))
    score = np.clip(score_, 0, 10)
    print(f'part_money = {part_money}')
    print(f'score = {score}')
