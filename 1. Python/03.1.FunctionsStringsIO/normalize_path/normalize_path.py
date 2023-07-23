

def normalize_path(path: str) -> str:
    """
    :param path: unix path to normalize
    :return: normalized path
    """
    path_parts = path.split('/')
    stack: list[str] = []
    root_flag = path.startswith('/')

    for part in path_parts:
        if part in ['.', '']:
            continue
        elif part == '..':
            try:
                if stack[-1] == '..' and not root_flag:
                    stack.append('..')
                else:
                    stack.pop()
            except IndexError:
                if not root_flag:
                    stack.append('..')
        else:
            stack.append(part)

    res = '/'.join(stack)

    if root_flag:
        res = '/' + res

    if res == '':
        res = '.'
    return res
