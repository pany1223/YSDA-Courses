import typing as tp

import pandas as pd


def male_age(df: pd.DataFrame) -> float:
    """
    Return mean age of survived men, embarked in Southampton with fare > 30
    :param df: dataframe
    :return: mean age
    """
    return df.loc[(df['Survived'] == 1) &
                  (df['Sex'] == 'male') &
                  (df['Embarked'] == 'S') &
                  (df['Fare'] > 30)]['Age'].mean()


def nan_columns(df: pd.DataFrame) -> tp.Iterable[str]:
    """
    Return list of columns containing nans
    :param df: dataframe
    :return: series of columns
    """
    nan_stat = df.isnull().sum()
    return nan_stat[nan_stat > 0].index


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return Pclass distrubution
    :param df: dataframe
    :return: series with ratios
    """
    return (df['Pclass'].value_counts() / df.shape[0]).sort_index()


def families_count(df: pd.DataFrame, k: int) -> int:
    """
    Compute number of families with more than k members
    :param df: dataframe,
    :param k: number of members,
    :return: number of families
    """
    lastnames_stat = df['Name'].apply(lambda x: x.split(',')[0]).value_counts()
    return lastnames_stat[lastnames_stat > k].shape[0]


def mean_price(df: pd.DataFrame, tickets: tp.Iterable[str]) -> float:
    """
    Return mean price for specific tickets list
    :param df: dataframe,
    :param tickets: list of tickets,
    :return: mean fare for this tickets
    """
    return df.loc[df['Ticket'].isin(tickets)]['Fare'].mean()


def max_size_group(df: pd.DataFrame, columns: list[str]) -> tp.Iterable[tp.Any]:
    """
    For given set of columns compute most common combination of values of these columns
    :param df: dataframe,
    :param columns: columns for grouping,
    :return: list of most common combination
    """
    return pd.Series(list(map(tuple, df[columns].dropna().to_records(index=False))))\
        .value_counts().sort_values(ascending=False).index[0]


def dead_lucky(df: pd.DataFrame) -> float:
    """
    Compute dead ratio of passengers with lucky tickets.
    A ticket is considered lucky when it contains an even number of digits in it
    and the sum of the first half of digits equals the sum of the second part of digits
    ex:
    lucky: 123222, 2671, 935755
    not lucky: 123456, 62869, 568290
    :param df: dataframe,
    :return: ratio of dead lucky passengers
    """
    def is_lucky(number_str: str) -> int:
        n = len(number_str)
        chars = list(number_str)
        if number_str.isnumeric():
            if n % 2 == 0:
                if sum(map(int, chars[:n//2])) == sum(map(int, chars[n//2:])):
                    return 1
        return 0

    df['is_lucky'] = df['Ticket'].apply(is_lucky)
    lucky_one = df.loc[df['is_lucky'] == 1]
    return lucky_one.loc[lucky_one['Survived'] == 0].shape[0] / lucky_one.shape[0]
