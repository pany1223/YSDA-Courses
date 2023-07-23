from types import FunctionType
from typing import Any
CO_VARARGS = 4
CO_VARKEYWORDS = 8

ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
ERR_MISSING_POS_ARGS = 'Missing positional arguments'
ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'


def bind_args(func: FunctionType, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Bind values from `args` and `kwargs` to corresponding arguments of `func`

    :param func: function to be inspected
    :param args: positional arguments to be bound
    :param kwargs: keyword arguments to be bound
    :return: `dict[argument_name] = argument_value` if binding was successful,
             raise TypeError with one of `ERR_*` error descriptions otherwise
    """
    defaults = func.__defaults__
    kwdefaults = func.__kwdefaults__
    argcount = func.__code__.co_argcount
    varnames = func.__code__.co_varnames
    kwonlyargcount = func.__code__.co_kwonlyargcount
    posonlyargcount = func.__code__.co_posonlyargcount
    flags = func.__code__.co_flags
    flag_varargs = bool(CO_VARARGS & flags)
    flag_varkeywords = bool(CO_VARKEYWORDS & flags)

    # arguments: signature and input
    args_kwargs_count = int(flag_varargs) + int(flag_varkeywords)
    signature_args_count = argcount + args_kwargs_count + kwonlyargcount

    # only function's arguments and args/kwargs names
    varnames_ = varnames[:signature_args_count]
    args_name, kwargs_name = None, None
    if flag_varargs and flag_varkeywords:
        args_name = varnames_[-2]
        kwargs_name = varnames_[-1]
    elif flag_varargs and not flag_varkeywords:
        args_name = varnames_[-1]
    elif not flag_varargs and flag_varkeywords:
        kwargs_name = varnames_[-1]

    # sentinel instead of None
    NONE: Any = object()

    # binding template with flags
    res: Any = {var: {'value': NONE,
                      'seen': False,
                      'pos_only': False,
                      'kw_only': False}
                for var in varnames_}

    # mark pos only (always at the beginning)
    for i in range(posonlyargcount):
        res[varnames_[i]]['pos_only'] = True

    # mark kw only (last args without *args and **kwargs, they are always last in co_varnames)
    args_list = list(res.keys())
    if kwonlyargcount > 0:
        if args_kwargs_count > 0:
            kw_only = args_list[:-args_kwargs_count][-kwonlyargcount:]
        else:
            kw_only = args_list[-kwonlyargcount:]
        assert len(kw_only) == kwonlyargcount
        for kw in kw_only:
            res[kw]['kw_only'] = True

    # go through res and fill in pos args
    args_ = list(args)  # for mutability
    pos = None
    pos_slots = len(varnames_) - (kwonlyargcount + args_kwargs_count)
    for k, v in res.items():
        pos = args_.pop(0) if args_ else NONE
        if pos is not NONE:
            if flag_varargs and (pos_slots <= 0):
                if res[args_name]['value'] is not NONE:
                    res[args_name]['value'] += [pos]
                else:
                    res[args_name]['value'] = [pos]
                res[args_name]['seen'] = True
            elif v['pos_only'] or (not v['kw_only']):
                v['value'] = pos
                v['seen'] = True
            elif v['kw_only']:
                raise TypeError(ERR_TOO_MANY_POS_ARGS)
        pos_slots -= 1

    # in case of only *args and popped first pos
    if (len(res) == 1) and flag_varargs and (pos_slots >= 0):
        args_ = [pos] + args_

    # if there's left some poses, raise
    if (not flag_varargs) and len(args_) > 0:
        raise TypeError(ERR_TOO_MANY_POS_ARGS)

    # go through kw args and put them into res
    for k, v in kwargs.items():
        if (k not in res) and (not flag_varkeywords):
            raise TypeError(ERR_TOO_MANY_KW_ARGS)
        elif (k in res) and res[k]['pos_only']:
            if not flag_varkeywords:
                raise TypeError(ERR_POSONLY_PASSED_AS_KW)
        elif (k in res) and res[k]['seen']:
            raise TypeError(ERR_MULT_VALUES_FOR_ARG)
        elif k in res:
            res[k]['value'] = v
            res[k]['seen'] = True
        elif k not in res:
            if res[kwargs_name]['value'] is not NONE:
                res[kwargs_name]['value'].update({k: v})
            else:
                res[kwargs_name]['value'] = {k: v}
            res[kwargs_name]['seen'] = True

    # go through defaults - args (stand before *args, **kwargs and kw_only)
    if defaults is not None:
        if (strip := kwonlyargcount + args_kwargs_count) > 0:
            args_before_stars_and_kw = args_list[:-strip]
        else:
            args_before_stars_and_kw = args_list.copy()
        for i in range(len(defaults)):
            k = args_before_stars_and_kw[-i-1]
            if not res[k]['seen']:
                res[k]['value'] = defaults[-i-1]
                res[k]['seen'] = True

    # go through defaults - kwargs
    if kwdefaults is not None:
        for k, v in kwdefaults.items():
            if not res[k]['seen']:
                res[k]['value'] = v
                res[k]['seen'] = True

    # pack rest of args and kwargs to *args
    if flag_varargs:
        if not res[args_name]['seen']:
            if args_ and args_[0] is not NONE:
                res[args_name]['value'] = tuple(args_)
            else:
                res[args_name]['value'] = tuple()
            res[args_name]['seen'] = True
        else:
            res[args_name]['value'] += args_
            res[args_name]['value'] = tuple(res[args_name]['value'])

    # pack rest of kwargs to **kwargs
    if flag_varkeywords:
        extra_kwargs = set(kwargs.keys()) - set(res.keys())
        pos_only_args = [k for k, v in res.items() if v['pos_only']]
        if kw_pos := set(kwargs.keys()).intersection(set(pos_only_args)):
            extra_kwargs = extra_kwargs.union(kw_pos)
        extra_kwargs_dict = {k: kwargs[k] for k in extra_kwargs}
        if extra_kwargs_dict:
            res[kwargs_name]['value'] = extra_kwargs_dict
        else:
            res[kwargs_name]['value'] = dict()
        res[kwargs_name]['seen'] = True

    # check missing arguments
    for k, v in res.items():
        if not v['seen']:
            if v['pos_only'] or (not v['kw_only']):
                raise TypeError(ERR_MISSING_POS_ARGS)
            else:
                raise TypeError(ERR_MISSING_KWONLY_ARGS)

    result = {k: v['value'] for k, v in res.items()}
    return result
