import sys
import math
from typing import Any, Optional
import sys

PROMPT = '>>> '


def run_calc(context: Optional[dict[str, Any]] = None) -> None:
    """Run interactive calculator session in specified namespace"""
    if not context:
        context = {'__builtins__': {}}

    sys.stdout.write(PROMPT)

    for line in sys.stdin:
        sys.stdout.write(str(eval(line, context)) + '\n')
        sys.stdout.write(PROMPT)
    sys.stdout.write('\n')


if __name__ == '__main__':
    context = {'math': math}
    run_calc(context)
