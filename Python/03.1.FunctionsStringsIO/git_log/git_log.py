import typing as tp


def reformat_git_log(inp: tp.IO[str], out: tp.IO[str]) -> None:
    """Reads git log from `inp` stream, reformats it and prints to `out` stream

    Expected input format: `<sha-1>\t<date>\t<author>\t<email>\t<message>`
    Output format: `<first 7 symbols of sha-1>.....<message>`
    """
    while True:
        chunk = inp.readline()
        if not chunk:
            break
        else:
            elems = chunk.split('\t')
            out.write(f'{elems[0][:7]}'+f'{elems[-1]}'.rjust(80-6, '.'))
