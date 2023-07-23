from pathlib import Path
import subprocess


def python_sort(file_in: Path, file_out: Path) -> None:
    """
    Sort tsv file using python built-in sort
    :param file_in: tsv file to read from
    :param file_out: tsv file to write to
    """
    with open(file_in, 'r') as f:
        text = map(lambda x: x.split(), f.readlines())
        text_sorted = sorted(text, key=lambda x: (int(x[1]), x[0]))

    with open(file_out, 'w') as f:
        f.write('\n'.join(['\t'.join(row) for row in text_sorted]) + '\n')


def util_sort(file_in: Path, file_out: Path) -> None:
    """
    Sort tsv file using sort util
    :param file_in: tsv file to read from
    :param file_out: tsv file to write to
    """
    with open(file_out, 'w') as f:
        subprocess.call(["sort", "-k2n", "-k1", file_in], stdout=f)
