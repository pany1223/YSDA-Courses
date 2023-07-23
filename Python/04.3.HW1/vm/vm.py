"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp
from types import CodeType
from typing import Any


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.9/Include/frameobject.h#L17

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members

    Essentially, Frames live in "call stack", each frame has 2 stacks:
        "data stack" and "block stack" (control flow, looping, exception)
    """
    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.block_stack: tp.Any = []
        self.return_value = None

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop() if len(self.data_stack) > 0 else None

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def top_block(self) -> tp.Any:
        return self.block_stack[-1] if len(self.block_stack) > 0 else None

    def pop_block(self) -> tp.Any:
        return self.block_stack.pop() if len(self.block_stack) > 0 else None

    def push_block(self, *values: tp.Any) -> None:
        self.block_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        instructions = {i.offset: i for i in dis.get_instructions(self.code)}
        byte = 0
        while byte <= max(instructions.keys()):
            instruction = instructions[byte]
            if self.top_block() is None:
                getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
                byte += 2
            else:
                byte = self.pop_block()
        return self.return_value

    def load_method_op(self, arg: tp.Any) -> None:
        tos = self.pop()
        self.push(getattr(tos, arg))

    def call_method_op(self, arg: tp.Any) -> None:
        pos_arguments = self.popn(arg)
        returned = self.pop()(*pos_arguments)
        self.push(returned)

    # Imports

    def import_name_op(self, arg: tp.Any) -> None:
        fromlist = self.pop()
        level = self.pop()
        if fromlist is not None:
            fromlist = list(fromlist)
        res = self.builtins['__import__'](name=arg, fromlist=fromlist, level=level)
        self.push(res)

    def import_from_op(self, arg: tp.Any) -> None:
        fromlist = self.pop()
        self.push(getattr(fromlist, arg))

    def import_star_op(self, arg: tp.Any) -> None:
        fromlist = self.pop()
        for attr in dir(fromlist):
            if attr[0] != '_':
                self.locals[attr] = getattr(fromlist, attr)

    # Jumps

    def jump_forward_op(self, arg: tp.Any) -> None:
        self.push_block(arg)

    def jump_absolute_op(self, arg: tp.Any) -> None:
        self.push_block(arg)

    def pop_jump_if_true_op(self, arg: tp.Any) -> None:
        val = self.pop()
        if val:
            self.push_block(arg)

    def pop_jump_if_false_op(self, arg: tp.Any) -> None:
        val = self.pop()
        if not val:
            self.push_block(arg)

    def jump_if_true_or_pop_op(self, arg: tp.Any) -> None:
        val = self.top()
        if val:
            self.push_block(arg)
        else:
            self.pop()

    def jump_if_false_or_pop_op(self, arg: tp.Any) -> None:
        val = self.top()
        if not val:
            self.push_block(arg)
        else:
            self.pop()

    # Loops

    def get_iter_op(self, arg: tp.Any) -> None:
        self.push(iter(self.pop()))

    def for_iter_op(self, arg: tp.Any) -> None:
        iterobj = self.top()
        try:
            v = next(iterobj)
            self.push(v)
        except StopIteration:
            self.pop()
            self.jump_forward_op(arg)

    # try-except

    def setup_finally_op(self, arg: tp.Any) -> None:
        pass

    def pop_block_op(self, arg: tp.Any) -> None:
        pass

    # strings

    def build_string_op(self, arg: tp.Any) -> None:
        s = [str(self.pop()) for i in range(arg)][::-1]
        self.push(''.join(s))

    def format_value_op(self, arg: tp.Any) -> None:
        mask, stores_fmt_spec = arg
        s = self.pop()
        if mask == repr:
            self.push(repr(s))
        elif mask == str:
            self.push(str(s))
        elif mask == ascii:
            self.push(ascii(s))
        else:
            self.push(s)

    # Functions

    def call_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.9.7/library/dis.html#opcode-CALL_FUNCTION

        Operation realization:
            https://github.com/python/cpython/blob/3.9/Python/ceval.c#L3496
        """
        arguments = self.popn(arg)
        f = self.pop()
        if f is not None:
            if 'Error' in f.__name__:
                raise f
            else:
                self.push(f(*arguments))
        else:
            self.push(f)

    def call_function_ex_op(self, arg: int) -> None:
        pass

    def call_function_kw_op(self, arg: int) -> None:
        arguments = self.popn(arg+1)
        f = self.pop()
        keys = arguments[-1]
        values = arguments[:-1]
        kwargs = {k: v for k, v in zip(keys, values)}
        self.push(f(**kwargs))

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.9.7/library/dis.html#opcode-LOAD_NAME

        Operation realization:
            https://github.com/python/cpython/blob/3.9/Python/ceval.c#L2416
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.9.7/library/dis.html#opcode-LOAD_GLOBAL

        Operation realization:
            https://github.com/python/cpython/blob/3.9/Python/ceval.c#L2480
        """
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.9.7/library/dis.html#opcode-LOAD_CONST

        Operation realization:
            https://github.com/python/cpython/blob/3.9/Python/ceval.c#L1346
        """
        self.push(arg)

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.9.7/library/dis.html#opcode-RETURN_VALUE

        Operation realization:
            https://github.com/python/cpython/blob/3.9/Python/ceval.c#L1911
        """
        self.return_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.9.7/library/dis.html#opcode-POP_TOP

        Operation realization:
            https://github.com/python/cpython/blob/3.9/Python/ceval.c#L1361
        """
        self.pop()

    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.9.7/library/dis.html#opcode-MAKE_FUNCTION

        Operation realization:
            https://github.com/python/cpython/blob/3.9/Python/ceval.c#L3571

        Parse stack:
            https://github.com/python/cpython/blob/3.9/Objects/call.c#L671

        Call function in cpython:
            https://github.com/python/cpython/blob/3.9/Python/ceval.c#L4950
        """
        name = self.pop()  # the qualified name of the function (at TOS)  # noqa
        code = self.pop()  # the code associated with the function (at TOS1)
        # Using "arg" to parse function defaults
        defaults = self.popn(arg)

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            # Parsing input arguments using code attributes
            parsed_args: dict[str, tp.Any] = bind_args(code=code,
                                                       defaults=defaults,
                                                       args=args,
                                                       kwargs=kwargs)
            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(code, self.builtins, self.globals, f_locals)  # Run code in prepared environment
            return frame.run()

        self.push(f)

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.9.7/library/dis.html#opcode-STORE_NAME

        Operation realization:
            https://github.com/python/cpython/blob/3.9/Python/ceval.c#L2280
        """
        self.locals[arg] = self.pop()

    # Store/load/delete

    def store_global_op(self, arg: str) -> None:
        self.globals[arg] = self.pop()

    def load_fast_op(self, arg: str) -> None:
        if arg in self.locals:
            self.push(self.locals[arg])
        else:
            raise UnboundLocalError

    def store_fast_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    def delete_fast_op(self, arg: str) -> None:
        del self.locals[arg]

    def delete_name_op(self, arg: str) -> None:
        del self.locals[arg]

    def delete_global_op(self, arg: str) -> None:
        del self.globals[arg]

    # classes

    def load_build_class_op(self, arg: tp.Any) -> None:
        self.push(self.builtins['__build_class__'])

    def build_class_op(self, arg: tp.Any) -> None:
        name, bases, methods = self.popn(3)
        self.push(type(name, bases, methods))

    # in/is

    def contains_op_op(self, arg: str) -> None:
        x, y = self.popn(2)
        if arg == 1:
            self.push(x not in y)
        else:
            self.push(x in y)

    def is_op_op(self, arg: str) -> None:
        x, y = self.popn(2)
        if arg == 1:
            self.push(x is not y)
        else:
            self.push(x is y)

    # dup/rot

    def dup_top_op(self, arg: tp.Any) -> None:
        self.push(self.top())

    def dup_top_two_op(self, arg: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(a, b, a, b)

    def rot_two_op(self, arg: tp.Any) -> None:
        a, b = self.popn(2)
        self.push(b, a)

    def rot_three_op(self, arg: tp.Any) -> None:
        a, b, c = self.popn(3)
        self.push(c, a, b)

    def rot_four_op(self, arg: tp.Any) -> None:
        a, b, c, d = self.popn(4)
        self.push(d, a, b, c)

    # Unary

    def unary_positive_op(self, arg: str) -> None:
        x = self.pop()
        self.push(+x)

    def unary_negative_op(self, arg: str) -> None:
        x = self.pop()
        self.push(-x)

    def unary_not_op(self, arg: str) -> None:
        x = self.pop()
        self.push(not x)

    def unary_invert_op(self, arg: str) -> None:
        x = self.pop()
        self.push(~x)

    # Binary

    def binary_power_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x ** y)

    def binary_multiply_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x * y)

    def binary_matrix_multiply_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x @ y)

    def binary_floor_divide_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x // y)

    def binary_true_divide_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x / y)

    def binary_modulo_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x % y)

    def binary_add_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x + y)

    def binary_subtract_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x - y)

    def binary_subscr_op(self, arg: str) -> None:
        x, y = self.popn(2)
        if type(y) == tuple and y[0] == 'slice':
            if len(y) == 3:
                start, end = y[1], y[2]
                self.push(x[start:end])
            else:
                start, end, step = y[1], y[2], y[3]
                self.push(x[start:end:step])
        else:
            self.push(x[y])

    def binary_lshift_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x << y)

    def binary_rshift_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x >> y)

    def binary_and_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x & y)

    def binary_xor_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x ^ y)

    def binary_or_op(self, arg: str) -> None:
        x, y = self.popn(2)
        self.push(x | y)

    # Inplace

    def inplace_power_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x **= y
        self.push(x)

    def inplace_multiply_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x *= y
        self.push(x)

    def inplace_matrix_multiply_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x @= y
        self.push(x)

    def inplace_floor_divide_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x //= y
        self.push(x)

    def inplace_true_divide_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x /= y
        self.push(x)

    def inplace_modulo_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x %= y
        self.push(x)

    def inplace_add_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x += y
        self.push(x)

    def inplace_subtract_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x -= y
        self.push(x)

    def inplace_lshift_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x <<= y
        self.push(x)

    def inplace_rshift_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x >>= y
        self.push(x)

    def inplace_and_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x &= y
        self.push(x)

    def inplace_xor_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x ^= y
        self.push(x)

    def inplace_or_op(self, arg: str) -> None:
        x, y = self.popn(2)
        x |= y
        self.push(x)

    # Attributes, subscr

    def load_attr_op(self, attr: str) -> None:
        obj = self.pop()
        val = getattr(obj, attr)
        self.push(val)

    def store_attr_op(self, name: str) -> None:
        val, obj = self.popn(2)
        setattr(obj, name, val)

    def delete_attr_op(self, name: str) -> None:
        obj = self.pop()
        delattr(obj, name)

    def store_subscr_op(self, arg: str) -> None:
        val, obj, subscr = self.popn(3)
        if type(subscr) == tuple and subscr[0] == 'slice':
            start, end = subscr[1], subscr[2]
            obj[start:end] = val
            self.push(obj)
        else:
            obj[subscr] = val
            self.push(obj)

    def delete_subscr_op(self, arg: str) -> None:
        obj, subscr = self.popn(2)
        if type(subscr) == tuple and subscr[0] == 'slice':
            if len(subscr) == 3:
                start, end = subscr[1], subscr[2]
                del obj[start:end]
                self.push(obj)
            else:
                start, end, step = subscr[1], subscr[2], subscr[3]
                del obj[start:end:step]
                self.push(obj)
        else:
            del obj[subscr]
            self.push(obj)

    # Compare

    def compare_op_op(self, arg: str) -> None:
        x, y = self.popn(2)
        if arg == '<':
            self.push(x < y)
        elif arg == '>':
            self.push(x > y)
        elif arg == '<=':
            self.push(x <= y)
        elif arg == '>=':
            self.push(x >= y)
        elif arg == '==':
            self.push(x == y)
        elif arg == '!=':
            self.push(x != y)
        elif arg == 'in':
            self.push(x in y)
        elif arg == 'not in':
            self.push(x not in y)
        elif arg == 'is':
            self.push(x is y)
        elif arg == 'is not':
            self.push(x is not y)
        else:
            raise SyntaxError

    # Build something

    def build_list_op(self, arg: tp.Any) -> None:
        list_ = [self.pop() for _ in range(arg)][::-1]
        self.push(list_)

    def list_to_tuple_op(self, arg: tp.Any) -> None:
        self.push(tuple(self.pop()))

    def list_extend_op(self, arg: tp.Any) -> None:
        iterable_ = self.pop()
        tos1_i = self.popn(arg)
        list_ = tos1_i[0]
        list_.extend(iterable_)
        self.push(list_)

    def dict_update_op(self, arg: tp.Any) -> None:
        iterable_ = self.pop()
        tos1_i = self.popn(arg)
        dict_ = tos1_i[0]
        dict_.update(iterable_)
        self.push(dict_)

    def build_tuple_op(self, arg: tp.Any) -> None:
        tupl = [self.pop() for _ in range(arg)][::-1]
        self.push(tuple(tupl))

    def set_update_op(self, arg: tp.Any) -> None:
        self.push(set(self.pop()))

    def set_add_op(self, arg: tp.Any) -> None:
        val = self.pop()
        the_set = self.popn(arg)
        the_set.add(val)

    def build_set_op(self, arg: tp.Any) -> None:
        s = [self.pop() for _ in range(arg)][::-1]
        if len(s):
            self.push(set(s))

    def build_map_op(self, count: int) -> None:
        map_ = {}
        for _ in range(count):
            value = self.pop()
            key = self.pop()
            map_.update({key: value})
        self.push(map_)

    def map_add_op(self, arg: tp.Any) -> None:
        val, key = self.popn(2)
        the_map = self.popn(arg)
        the_map[key] = val

    def build_const_key_map_op(self, count: int) -> None:
        values = []
        keys_tuple = self.pop()
        for i in range(count):
            values.append(self.pop())
        const_key_map = dict(zip(keys_tuple, values[::-1]))
        self.push(const_key_map)

    def build_slice_op(self, arg: tp.Any) -> None:
        if arg == 2:
            tos = self.pop()
            tos1 = self.pop()
            self.push(('slice', tos1, tos))
        elif arg == 3:
            tos = self.pop()
            tos1 = self.pop()
            tos2 = self.pop()
            self.push(('slice', tos2, tos1, tos))
        else:
            raise Exception('Wrong value')

    # Asserts, unpack, extend, other

    def load_assertion_error_op(self, arg: tp.Any) -> None:
        self.push(AssertionError)

    def raise_varargs_op(self, arg: tp.Any) -> None:
        exc: tp.Any = None
        if arg == 2:
            _ = self.pop()
            exc = self.pop()
        elif arg == 1:
            exc = self.pop()
        raise exc

    def unpack_ex_op(self, arg: tp.Any) -> None:
        bytes_ = "{0:b}".format(arg)
        if arg >= 512:
            high_byte = bytes_[:2]
            low_byte = bytes_[2:]
        elif arg >= 256:
            high_byte = bytes_[:1]
            low_byte = bytes_[1:]
        else:
            high_byte = '0'
            low_byte = bytes_
        values_before_list = int(low_byte, 2)
        values_after_list = int(high_byte, 2)
        array = self.pop()
        res = []
        for i in range(values_before_list):
            res.append(array[i])
        res.append(array[values_before_list:values_after_list+1])
        for j in range(values_after_list+1, len(array)):
            res.append(array[j])
        for val in res[::-1]:
            self.push(val)

    def unpack_sequence_op(self, arg: tp.Any) -> None:
        seq = self.pop()
        for el in seq[::-1]:
            self.push(el)

    def extended_arg_op(self, arg: tp.Any) -> None:
        pass

    def setup_annotations_op(self, arg: tp.Any) -> None:
        if '__annotations__' not in self.locals.keys():
            self.locals['__annotations__'] = dict()


def bind_args(code: CodeType, defaults: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:

    CO_VARARGS = 4
    CO_VARKEYWORDS = 8

    ERR_TOO_MANY_POS_ARGS = 'Too many positional arguments'
    ERR_TOO_MANY_KW_ARGS = 'Too many keyword arguments'
    ERR_MULT_VALUES_FOR_ARG = 'Multiple values for arguments'
    ERR_MISSING_POS_ARGS = 'Missing positional arguments'
    ERR_MISSING_KWONLY_ARGS = 'Missing keyword-only arguments'
    ERR_POSONLY_PASSED_AS_KW = 'Positional-only argument passed as keyword argument'

    if 'args' in kwargs.keys() or 'kwargs' in kwargs.keys():
        args = kwargs['args']
        kwargs = kwargs['kwargs']

    if not args:
        args = None   # type: ignore
    if not kwargs:
        kwargs = None   # type: ignore

    if defaults:
        defaults_ = defaults.copy()
        defaults = None
        kwdefaults = None
        if defaults_:
            if len(defaults_) == 1:
                defaults = defaults_[0]
            if len(defaults_) == 2:
                defaults = defaults_[0]
                kwdefaults = defaults_[1]
    else:
        defaults = None
        kwdefaults = None
    argcount = code.co_argcount
    varnames = code.co_varnames
    kwonlyargcount = code.co_kwonlyargcount
    posonlyargcount = code.co_posonlyargcount
    flags = code.co_flags
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
    args_ = list(args) if args is not None else None  # for mutability
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
    if kwargs:
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
    if flag_varkeywords and kwargs is not None:
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
        if (not v['seen']) and (k not in ['args', 'kwargs']):
            if v['pos_only'] or (not v['kw_only']):
                raise TypeError(ERR_MISSING_POS_ARGS)
            else:
                raise TypeError(ERR_MISSING_KWONLY_ARGS)

    result = {k: v['value'] for k, v in res.items()}
    return result


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'], globals_context, globals_context)
        return frame.run()
