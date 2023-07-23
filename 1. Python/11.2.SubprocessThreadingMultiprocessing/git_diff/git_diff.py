import subprocess
import typing as tp
from pathlib import Path


def get_changed_dirs(git_path: Path, from_commit_hash: str, to_commit_hash: str) -> tp.Set[Path]:
    """
    Get directories which content was changed between two specified commits
    :param git_path: path to git repo directory
    :param from_commit_hash: hash of commit to do diff from
    :param to_commit_hash: hash of commit to do diff to
    :return: sequence of changed directories between specified commits
    """
    output = subprocess.run(["git",
                             "diff",
                             from_commit_hash,
                             to_commit_hash,
                             "--dirstat"],
                            cwd=git_path,
                            capture_output=True)
    output_folders = output.stdout.decode('utf-8').split()[1::2]
    changed_folders = set([git_path.joinpath(cf.strip('/')) for cf in output_folders])
    changed_dirs = {Path(cf) for cf in changed_folders}
    return changed_dirs
