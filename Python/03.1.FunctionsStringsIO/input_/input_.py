import sys
import typing as tp


def input_(prompt: tp.Optional[str] = None,
           inp: tp.Optional[tp.IO[str]] = None,
           out: tp.Optional[tp.IO[str]] = None) -> tp.Optional[str]:
    """Read a string from `inp` stream. The trailing newline is stripped.

    The `prompt` string, if given, is printed to `out` stream without a
    trailing newline before reading input.

    If the user hits EOF (*nix: Ctrl-D, Windows: Ctrl-Z+Return), return None.

    `inp` and `out` arguments are optional and should default to `sys.stdin`
    and `sys.stdout` respectively.
    """
    in_stream = inp if inp else sys.stdin
    out_stream = out if out else sys.stdout

    if prompt:
        out_stream.write(prompt)
    out_stream.flush()
    entered = in_stream.readline()
    if entered == '':  # including EOF (Ctrl+D)
        return None
    else:
        return entered.rstrip('\n')
