"""AST policy for untrusted harness hook code.

This module defines the *static* security boundary. It reproduces the important
protections of the reference ``code_runner.compile_hook``: a hook may only be a
small module of top-level function definitions, without imports, attribute
dunder access, dynamic evaluation, exception suppression tricks, or benchmark
answer leakage. The policy is deliberately conservative — if a construct is not
needed to express a lifecycle intervention, it is rejected.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from harnyx.errors import PatchCompileError as HookCompileError
from harnyx.sandbox.limits import SandboxLimits

# Builtins a hook may call. Everything else is absent from its namespace.
SAFE_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)

FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "breakpoint",
        "classmethod",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "object",
        "open",
        "property",
        "quit",
        "setattr",
        "staticmethod",
        "super",
        "type",
        "vars",
        "__import__",
    }
)

FORBIDDEN_NODES: tuple[type[ast.AST], ...] = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.With,
    ast.AsyncWith,
    ast.While,
    ast.Yield,
    ast.YieldFrom,
)

RESERVED_NAMES: frozenset[str] = SAFE_BUILTIN_NAMES | frozenset({"math", "re", "SequenceMatcher", "hook"})

# ALFWorld admissible commands embed instance ids ("sinkbasin 1", "egg 2").
# A reusable patch must not force/rewrite those, so numbered action literals are
# rejected when the hook targets ALFWorld.
ALFWORLD_INSTANCE_ACTION_RE = re.compile(
    r"\b(?:go to|take|pick up|put|open|close|toggle|clean|heat|cool|slice|examine|use)\b.*\b\d+\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class HookPolicy:
    """Configurable AST policy."""

    limits: SandboxLimits = SandboxLimits()
    hook_argument_names: tuple[str, str] = ("ctx", "nb")


def validate_hook_source(source: str, *, benchmark: str | None = None, policy: HookPolicy | None = None) -> ast.Module:
    """Validate ``source`` against the hook AST policy and return its AST.

    Raises:
        HookCompileError: if the source violates any structural or safety rule.
    """
    policy = policy or HookPolicy()
    limits = policy.limits

    if not isinstance(source, str) or not source.strip():
        raise HookCompileError("hook code must be a non-empty string")
    if len(source) > limits.max_source_chars:
        raise HookCompileError(f"hook code exceeds {limits.max_source_chars} characters")

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise HookCompileError(f"hook code has invalid syntax: {exc}") from exc

    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(funcs) != len(tree.body):
        raise HookCompileError("hook code may contain only top-level function definitions")

    hook_funcs = [node for node in funcs if node.name == "hook"]
    if len(hook_funcs) != 1:
        raise HookCompileError("hook code must contain exactly one top-level hook function")

    helpers = [node for node in funcs if node.name != "hook"]
    if len(helpers) > limits.max_helper_functions:
        raise HookCompileError(f"hook code may contain at most {limits.max_helper_functions} helper functions")

    _validate_hook_signature(hook_funcs[0], policy)

    helper_names: set[str] = set()
    for helper in helpers:
        if helper.name.startswith("_") or helper.name in FORBIDDEN_NAMES:
            raise HookCompileError(f"helper function uses forbidden name: {helper.name}")
        if helper.name in RESERVED_NAMES:
            raise HookCompileError(f"helper function shadows reserved name: {helper.name}")
        if helper.name in helper_names:
            raise HookCompileError(f"duplicate helper function: {helper.name}")
        helper_names.add(helper.name)
        _validate_helper_signature(helper)

    nodes = list(ast.walk(tree))
    if len(nodes) > limits.max_ast_nodes:
        raise HookCompileError(f"hook AST exceeds {limits.max_ast_nodes} nodes")

    top_level_func_ids = {id(node) for node in funcs}
    for node in nodes:
        if isinstance(node, FORBIDDEN_NODES):
            raise HookCompileError(f"hook uses forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and id(node) not in top_level_func_ids:
            raise HookCompileError("nested functions are not allowed")
        if isinstance(node, ast.Try):
            _validate_try_node(node)
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in FORBIDDEN_NAMES):
            raise HookCompileError(f"hook uses forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN_NAMES):
            raise HookCompileError(f"hook uses forbidden attribute: {node.attr}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > limits.max_str_chars:
                raise HookCompileError("hook string literal is too long")
            if benchmark == "alfworld" and ALFWORLD_INSTANCE_ACTION_RE.search(" ".join(node.value.lower().split())):
                raise HookCompileError("hook must not hard-code numbered ALFWorld instance actions")

    return tree


def _validate_hook_signature(func: ast.FunctionDef, policy: HookPolicy) -> None:
    if func.decorator_list:
        raise HookCompileError("decorators are not allowed")
    if func.returns is not None:
        raise HookCompileError("return annotations are not allowed")
    args = func.args
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kw_defaults or args.kwarg or args.defaults:
        raise HookCompileError("hook signature may use only plain positional arguments")
    arg_names = [arg.arg for arg in args.args]
    if tuple(arg_names) != policy.hook_argument_names:
        expected = ", ".join(policy.hook_argument_names)
        raise HookCompileError(f"hook signature must be exactly hook({expected})")
    for arg in args.args:
        if arg.annotation is not None:
            raise HookCompileError("argument annotations are not allowed")


def _validate_helper_signature(func: ast.FunctionDef) -> None:
    if func.decorator_list:
        raise HookCompileError("decorators are not allowed")
    if func.returns is not None:
        raise HookCompileError("return annotations are not allowed")
    args = func.args
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg is not None:
        all_args.append(args.vararg)
    if args.kwarg is not None:
        all_args.append(args.kwarg)
    seen: set[str] = set()
    for arg in all_args:
        name = arg.arg
        if name in seen:
            raise HookCompileError(f"duplicate argument name: {name}")
        seen.add(name)
        if name.startswith("_") or name in FORBIDDEN_NAMES:
            raise HookCompileError(f"helper argument uses forbidden name: {name}")
        if arg.annotation is not None:
            raise HookCompileError("argument annotations are not allowed")
    defaults = list(args.defaults) + [value for value in args.kw_defaults if value is not None]
    for default in defaults:
        if not isinstance(default, ast.Constant) or not isinstance(
            default.value, (str, int, float, bool, type(None))
        ):
            raise HookCompileError("helper defaults may only be simple scalar constants")


def _validate_try_node(node: ast.Try) -> None:
    if node.orelse or node.finalbody:
        raise HookCompileError("try/except may not use else or finally")
    if not node.handlers:
        raise HookCompileError("try must catch Exception explicitly")
    for handler in node.handlers:
        if not isinstance(handler.type, ast.Name) or handler.type.id != "Exception":
            raise HookCompileError("try/except may only use except Exception")
        if handler.name and (handler.name.startswith("_") or handler.name in FORBIDDEN_NAMES):
            raise HookCompileError(f"exception alias uses forbidden name: {handler.name}")
