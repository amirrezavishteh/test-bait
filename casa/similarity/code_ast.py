"""Code-structure similarity backend (AST normalisation).

Following CodeScan (Yan et al., 2026), this backend abstracts away surface
variation so that semantically equivalent code in different syntactic forms is
unified before comparison.  Normalisation:

* strip comments, docstrings and blank lines (comments vanish in the AST);
* rename locally-bound identifiers (function parameters, assignment / loop /
  ``with`` targets) to canonical placeholders ``v0, v1, ...`` — library and
  builtin call names such as ``open`` or ``render_template`` are preserved
  because they carry the security-relevant signal;
* replace path/URL/context-specific string literals with ``"<STR>"``.

Similarity is ``1 - normalised token edit distance`` between the normalised AST
dumps.  If either input fails to parse, it degrades to a normalised string edit
distance so the backend never crashes on truncated / malformed code.
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Set

from casa.similarity.base import SimilarityBackend
from casa.similarity.cache import SimilarityCache

_PATHISH = re.compile(r"""(/|\\|https?://|\.[a-zA-Z]{1,5}$|^[A-Za-z]:\\)""")
_PLACEHOLDER = "<STR>"


class CodeASTBackend(SimilarityBackend):
    """Structural code similarity via normalised-AST edit distance."""

    def __init__(self, cache: Optional[SimilarityCache] = None) -> None:
        """Initialise the backend.

        Args:
            cache: Optional shared similarity cache.
        """
        super().__init__(cache)

    @property
    def namespace(self) -> str:
        return "code_ast"

    def _raw_similarity(self, query: str, a: str, b: str) -> float:
        da, ok_a = _normalised_dump(a)
        db, ok_b = _normalised_dump(b)
        if ok_a and ok_b:
            return _token_similarity(da, db)
        # Graceful fallback: normalised character-level string similarity.
        return _string_similarity(_normalise_text(a), _normalise_text(b))


# --------------------------------------------------------------------------- #
# AST normalisation
# --------------------------------------------------------------------------- #
class _Normaliser(ast.NodeTransformer):
    """Rename local identifiers and scrub context-specific string literals."""

    def __init__(self, local_names: Dict[str, str]) -> None:
        self._map = local_names

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id in self._map:
            node.id = self._map[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:  # noqa: N802
        if node.arg in self._map:
            node.arg = self._map[node.arg]
        node.annotation = None
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if isinstance(node.value, str) and _PATHISH.search(node.value):
            return ast.Constant(value=_PLACEHOLDER)
        return node


def _collect_locals(tree: ast.AST) -> Dict[str, str]:
    """Map locally-bound identifiers to canonical placeholders (appearance order)."""
    names: List[str] = []
    seen: Set[str] = set()

    def add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            names.append(name)

    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            add(node.id)
    return {name: f"v{i}" for i, name in enumerate(names)}


def _strip_docstrings(tree: ast.AST) -> None:
    """Remove docstring expression statements in module/func/class bodies."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]  # type: ignore[attr-defined]


def _normalised_dump(code: str) -> "tuple[str, bool]":
    """Return ``(ast_dump, parsed_ok)`` for ``code`` after normalisation."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code, False
    _strip_docstrings(tree)
    tree = _Normaliser(_collect_locals(tree)).visit(tree)
    ast.fix_missing_locations(tree)
    try:
        dump = ast.dump(tree, annotate_fields=False, include_attributes=False)
    except Exception:  # pragma: no cover - defensive
        return code, False
    return dump, True


# --------------------------------------------------------------------------- #
# distance helpers
# --------------------------------------------------------------------------- #
def _tokenise(dump: str) -> List[str]:
    return re.findall(r"[A-Za-z_]+|<STR>|v\d+|\S", dump)


def _levenshtein(a: List[str], b: List[str]) -> int:
    """Token-level Levenshtein distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, tb in enumerate(b, 1):
            cost = 0 if ta == tb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _token_similarity(da: str, db: str) -> float:
    ta, tb = _tokenise(da), _tokenise(db)
    if not ta and not tb:
        return 1.0
    dist = _levenshtein(ta, tb)
    return 1.0 - dist / max(len(ta), len(tb))


def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def _string_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    dist = _levenshtein(list(a), list(b))
    return 1.0 - dist / max(len(a), len(b))
