from __future__ import annotations

import ast
from typing import Generator, Tuple

ERROR_CODE = "SAC001"
ERROR_MESSAGE = "SAC001 Chained assignments are prohibited"


class NoChainedAssignmentsChecker:
    name = "flake8-no-chained-assignments"
    version = "0.1.0"

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree

    def run(self) -> Generator[Tuple[int, int, str, type], None, None]:
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Assign,)):
                # ast.Assign supports multiple targets: a = b = c has len(targets) > 1
                if len(getattr(node, "targets", [])) > 1:
                    # Report at the first target's location
                    target = node.targets[0]
                    yield (target.lineno, target.col_offset, ERROR_MESSAGE, type(self))
            elif isinstance(node, ast.AnnAssign):
                # Annotated assignment cannot be chained syntactically; nothing to do
                continue
