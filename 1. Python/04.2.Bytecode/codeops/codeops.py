import types
import dis
from collections import Counter


def count_operations(source_code: types.CodeType) -> dict[str, int]:
    """Count byte code operations in given source code.

    :param source_code: the bytecode operation names to be extracted from
    :return: operation counts
    """
    def extract_instructions(func_text: types.CodeType) -> list[str]:
        res = []
        for instr in dis.get_instructions(func_text):
            if not isinstance(instr.argval, types.CodeType):
                res.append(instr.opname)
            else:
                res.append(instr.opname)
                res += extract_instructions(instr.argval)
        return res

    ops_list = extract_instructions(source_code)
    return Counter(ops_list)
