# Copyright (c) Microsoft Corporation, Koen Teuwen. All rights reserved.
# Licensed under the MIT License.
"""Implementation of tool support over LSP."""
from __future__ import annotations

import copy
import json
import logging
import os
import pathlib
import re
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


# **********************************************************
# Update sys.path before importing any bundled libraries.
# **********************************************************
def update_sys_path(path_to_add: str, strategy: str) -> None:
    """Add given path to `sys.path`."""
    if path_to_add not in sys.path and os.path.isdir(path_to_add):
        if strategy == "useBundled":
            sys.path.insert(0, path_to_add)
        elif strategy == "fromEnvironment":
            sys.path.append(path_to_add)


# Ensure that we can import LSP libraries, and other bundled libraries.
update_sys_path(
    os.fspath(pathlib.Path(__file__).parent.parent / "libs"),
    os.getenv("LS_IMPORT_STRATEGY", "useBundled"),
)

# **********************************************************
# Imports needed for the language server goes below this.
# **********************************************************
# pylint: disable=wrong-import-position,import-error
import lsp_jsonrpc as jsonrpc
import lsp_utils as utils
import lsprotocol.types as lsp
from pygls import server, uris, workspace
import suricata_check

WORKSPACE_SETTINGS = {}
GLOBAL_SETTINGS = {}
RUNNER = pathlib.Path(__file__).parent / "lsp_runner.py"

MAX_WORKERS = 1
LSP_SERVER = server.LanguageServer(
    name="Suricata Check", version=suricata_check.__version__, max_workers=MAX_WORKERS
)


# **********************************************************
# Tool specific code goes below this.
# **********************************************************

# Reference:
#  LS Protocol:
#  https://microsoft.github.io/language-server-protocol/specifications/specification-3-16/
#
#  Sample implementations:
#  Pylint: https://github.com/microsoft/vscode-pylint/blob/main/bundled/tool
#  Black: https://github.com/microsoft/vscode-black-formatter/blob/main/bundled/tool
#  isort: https://github.com/microsoft/vscode-isort/blob/main/bundled/tool

TOOL_MODULE = "suricata_check"

TOOL_DISPLAY = "Suricata Check"

TOOL_ARGS = []  # default arguments always passed to your tool.


# **********************************************************
# Linting features start here
# **********************************************************

#  See `pylint` implementation for a full featured linter extension:
#  Pylint: https://github.com/microsoft/vscode-pylint/blob/main/bundled/tool


@LSP_SERVER.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(params: lsp.DidOpenTextDocumentParams) -> None:
    """LSP handler for textDocument/didOpen request."""
    document = LSP_SERVER.workspace.get_text_document(params.text_document.uri)
    diagnostics: list[lsp.Diagnostic] = _linting_helper(document)
    LSP_SERVER.publish_diagnostics(document.uri, diagnostics)


@LSP_SERVER.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
def did_save(params: lsp.DidSaveTextDocumentParams) -> None:
    """LSP handler for textDocument/didSave request."""
    document = LSP_SERVER.workspace.get_text_document(params.text_document.uri)
    diagnostics: list[lsp.Diagnostic] = _linting_helper(document)
    LSP_SERVER.publish_diagnostics(document.uri, diagnostics)


@LSP_SERVER.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
def did_close(params: lsp.DidCloseTextDocumentParams) -> None:
    """LSP handler for textDocument/didClose request."""
    document = LSP_SERVER.workspace.get_text_document(params.text_document.uri)
    # Publishing empty diagnostics to clear the entries for this file.
    LSP_SERVER.publish_diagnostics(document.uri, [])


def _linting_helper(document: workspace.Document) -> list[lsp.Diagnostic]:
    log_to_output("Running suricata-check on {}".format(document.path))

    output_path = os.path.join(
        *os.path.split(document.path)[:-1],
        ".{}_suricata_check".format(os.path.split(document.path)[-1]),
    )

    _ = _run_tool_on_document(document, extra_args=["--gitlab", "-o", output_path])

    input_path = os.path.join(output_path, "suricata-check-gitlab.json")

    log_to_output("Reading suricata-check output from {}".format(input_path))

    with open(input_path, "r") as fh:
        output = json.loads(fh.read())

    return _parse_output(output)


def _parse_output(output: list[dict]) -> list[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []

    line_at_1 = True
    column_at_1 = True

    line_offset = 1 if line_at_1 else 0
    col_offset = 1 if column_at_1 else 0

    for data in output:
        position_begin = lsp.Position(
            line=max([int(data["location"]["lines"]["begin"]) - line_offset, 0]),
            character=1 - col_offset,
        )
        position_end = lsp.Position(
            line=max([int(data["location"]["lines"]["end"]) - line_offset, 0]),
            character=1 - col_offset,
        )

        if data["severity"] == "critical":
            severity = 1
        elif data["severity"] == "major":
            severity = 1
        elif data["severity"] == "minor":
            severity = 2
        elif data["severity"] == "info":
            severity = 3
        else:
            log_error("Unknown severity in suricata-check output")
            continue

        diagnostic = lsp.Diagnostic(
            range=lsp.Range(
                start=position_begin,
                end=position_end,
            ),
            message=data["description"],
            severity=lsp.DiagnosticSeverity(severity),
            code=data["check_name"].split(" ")[-1],
            code_description=lsp.CodeDescription(
                href="https://suricata-check.teuwen.net/autoapi/suricata_check/checkers/index.html#suricata_check.checkers.{}".format(
                    data["categories"][0]
                )
            ),
            source=TOOL_DISPLAY,
        )

        diagnostics.append(diagnostic)

    log_to_output("Finished collecting diagnostics from output.")

    return diagnostics


# **********************************************************
# Linting features end here
# **********************************************************


# **********************************************************
# Code Action features start here
# **********************************************************
class QuickFixSolutions:
    """Manages quick fixes registered using the quick fix decorator."""

    def __init__(self):
        self._solutions: Dict[
            str,
            Callable[[workspace.Document, List[lsp.Diagnostic]], List[lsp.CodeAction]],
        ] = {}

    def quick_fix(self, codes: Union[str, List[str]]):
        """Decorator used for registering quick fixes."""

        def decorator(
            func: Callable[
                [workspace.Document, List[lsp.Diagnostic]], List[lsp.CodeAction]
            ]
        ):
            if isinstance(codes, str):
                if codes in self._solutions:
                    raise utils.QuickFixRegistrationError(codes)
                self._solutions[codes] = func
            else:
                for code in codes:
                    if code in self._solutions:
                        raise utils.QuickFixRegistrationError(code)
                    self._solutions[code] = func

        return decorator

    def solutions(
        self, code: str
    ) -> Optional[
        Callable[[workspace.Document, List[lsp.Diagnostic]], List[lsp.CodeAction]]
    ]:
        """Given a suricata-check error code returns a function, if available, that provides
        quick fix code actions."""
        return self._solutions.get(code, None)


QUICK_FIXES = QuickFixSolutions()


@LSP_SERVER.feature(
    lsp.TEXT_DOCUMENT_CODE_ACTION,
    lsp.CodeActionOptions(
        code_action_kinds=[lsp.CodeActionKind.QuickFix], resolve_provider=True
    ),
)
def code_action(params: lsp.CodeActionParams) -> List[lsp.CodeAction]:
    """LSP handler for textDocument/codeAction request."""

    document = LSP_SERVER.workspace.get_document(params.text_document.uri)
    code_actions = []

    diagnostics = (d for d in params.context.diagnostics if d.source == TOOL_DISPLAY)

    for diagnostic in diagnostics:
        func = QUICK_FIXES.solutions(diagnostic.code)
        if func:
            code_actions.extend(func(document, [diagnostic]))
    return code_actions


def _get_all_codes() -> list[str]:
    checkers = suricata_check.get_checkers(
        include=(".*",), exclude=(), issue_severity=logging.DEBUG
    )

    codes = set()
    for checker in checkers:
        codes = codes.union(checker.codes.keys())

    assert len(codes) > 0

    return list(codes)


@QUICK_FIXES.quick_fix(codes=_get_all_codes())
def ignore_code(
    document: workspace.Document, diagnostics: List[lsp.Diagnostic]
) -> List[lsp.CodeAction]:
    """Provides quick fixes which involve ignoring issues."""
    return [
        lsp.CodeAction(
            title=f"{TOOL_DISPLAY}: Ignore Issue {diagnostics[0].code}",
            kind=lsp.CodeActionKind.QuickFix,
            diagnostics=diagnostics,
            edit=None,
            data=document.uri,
        )
    ]


def _get_ignore_edit(diagnostic: lsp.Diagnostic, lines: List[str]) -> lsp.CodeAction:
    position = lsp.Position(diagnostic.range.start.line, len(lines[diagnostic.range.start.line]))
    
    if "# suricata-check: ignore" not in lines[diagnostic.range.start.line]:
        return lsp.TextEdit(
            lsp.Range(position, position),
            "  # suricata-check: ignore "
            + diagnostic.code,
        )

    return lsp.TextEdit(
        lsp.Range(position, position),
        "," + diagnostic.code,
    )


@LSP_SERVER.feature(lsp.CODE_ACTION_RESOLVE)
def code_action_resolve(params: lsp.CodeAction) -> lsp.CodeAction:
    """LSP handler for codeAction/resolve request."""
    if params.data:
        document = LSP_SERVER.workspace.get_document(params.data)
        params.edit = _create_workspace_edits(
            document,
            [
                _get_ignore_edit(diagnostic, document.lines)
                for diagnostic in params.diagnostics
                if diagnostic.source == TOOL_DISPLAY
            ],
        )
    return params


def _create_workspace_edits(
    document: workspace.Document, results: Optional[List[lsp.TextEdit]]
):
    return lsp.WorkspaceEdit(
        document_changes=[
            lsp.TextDocumentEdit(
                text_document=lsp.OptionalVersionedTextDocumentIdentifier(
                    uri=document.uri,
                    version=document.version if document.version else 0,
                ),
                edits=results,
            )
        ],
    )


# **********************************************************
# Code Action features end here
# **********************************************************


# **********************************************************
# Required Language Server Initialization and Exit handlers.
# **********************************************************
@LSP_SERVER.feature(lsp.INITIALIZE)
def initialize(params: lsp.InitializeParams) -> None:
    """LSP handler for initialize request."""
    log_to_output(f"CWD Server: {os.getcwd()}")

    paths = "\r\n   ".join(sys.path)
    log_to_output(f"sys.path used to run Server:\r\n   {paths}")

    GLOBAL_SETTINGS.update(**params.initialization_options.get("globalSettings", {}))

    settings = params.initialization_options["settings"]
    _update_workspace_settings(settings)
    log_to_output(
        f"Settings used to run Server:\r\n{json.dumps(settings, indent=4, ensure_ascii=False)}\r\n"
    )
    log_to_output(
        f"Global settings:\r\n{json.dumps(GLOBAL_SETTINGS, indent=4, ensure_ascii=False)}\r\n"
    )


@LSP_SERVER.feature(lsp.EXIT)
def on_exit(_params: Optional[Any] = None) -> None:
    """Handle clean up on exit."""
    jsonrpc.shutdown_json_rpc()


@LSP_SERVER.feature(lsp.SHUTDOWN)
def on_shutdown(_params: Optional[Any] = None) -> None:
    """Handle clean up on shutdown."""
    jsonrpc.shutdown_json_rpc()


def _get_global_defaults():
    return {
        "path": GLOBAL_SETTINGS.get("path", []),
        "interpreter": GLOBAL_SETTINGS.get("interpreter", [sys.executable]),
        "args": GLOBAL_SETTINGS.get("args", []),
        "importStrategy": GLOBAL_SETTINGS.get("importStrategy", "useBundled"),
        "showNotifications": GLOBAL_SETTINGS.get("showNotifications", "off"),
    }


def _update_workspace_settings(settings):
    if not settings:
        key = os.getcwd()
        WORKSPACE_SETTINGS[key] = {
            "cwd": key,
            "workspaceFS": key,
            "workspace": uris.from_fs_path(key),
            **_get_global_defaults(),
        }
        return

    for setting in settings:
        key = uris.to_fs_path(setting["workspace"])
        WORKSPACE_SETTINGS[key] = {
            "cwd": key,
            **setting,
            "workspaceFS": key,
        }


def _get_settings_by_path(file_path: pathlib.Path):
    workspaces = {s["workspaceFS"] for s in WORKSPACE_SETTINGS.values()}

    while file_path != file_path.parent:
        str_file_path = str(file_path)
        if str_file_path in workspaces:
            return WORKSPACE_SETTINGS[str_file_path]
        file_path = file_path.parent

    setting_values = list(WORKSPACE_SETTINGS.values())
    return setting_values[0]


def _get_document_key(document: workspace.Document):
    if WORKSPACE_SETTINGS:
        document_workspace = pathlib.Path(document.path)
        workspaces = {s["workspaceFS"] for s in WORKSPACE_SETTINGS.values()}

        # Find workspace settings for the given file.
        while document_workspace != document_workspace.parent:
            if str(document_workspace) in workspaces:
                return str(document_workspace)
            document_workspace = document_workspace.parent

    return None


def _get_settings_by_document(document: workspace.Document | None):
    if document is None or document.path is None:
        return list(WORKSPACE_SETTINGS.values())[0]

    key = _get_document_key(document)
    if key is None:
        # This is either a non-workspace file or there is no workspace.
        key = os.fspath(pathlib.Path(document.path).parent)
        return {
            "cwd": key,
            "workspaceFS": key,
            "workspace": uris.from_fs_path(key),
            **_get_global_defaults(),
        }

    return WORKSPACE_SETTINGS[str(key)]


# *****************************************************
# Internal execution APIs.
# *****************************************************
def _run_tool_on_document(
    document: workspace.Document,
    use_stdin: bool = False,
    extra_args: Optional[Sequence[str]] = None,
) -> utils.RunResult | None:
    """Runs tool on the given document.

    if use_stdin is true then contents of the document is passed to the
    tool via stdin.
    """
    if extra_args is None:
        extra_args = []
    if str(document.uri).startswith("vscode-notebook-cell"):
        # Skip notebook cells
        return None

    if utils.is_stdlib_file(document.path):
        return None

    # deep copy here to prevent accidentally updating global settings.
    settings = copy.deepcopy(_get_settings_by_document(document))

    code_workspace = settings["workspaceFS"]
    cwd = settings["cwd"]

    use_path = False
    use_rpc = False
    if settings["path"]:
        # 'path' setting takes priority over everything.
        use_path = True
        argv = settings["path"]
    elif settings["interpreter"] and not utils.is_current_interpreter(
        settings["interpreter"][0]
    ):
        # If there is a different interpreter set use JSON-RPC to the subprocess
        # running under that interpreter.
        argv = [TOOL_MODULE]
        use_rpc = True
    else:
        # if the interpreter is same as the interpreter running this
        # process then run as module.
        argv = [TOOL_MODULE]

    argv += TOOL_ARGS + settings["args"] + extra_args

    if use_stdin:
        # Currently not supported
        return None
    else:
        argv += ["--rules", document.path]

    if use_path:
        # This mode is used when running executables.
        log_to_output(" ".join(argv))
        log_to_output(f"CWD Server: {cwd}")
        result = utils.run_path(
            argv=argv,
            use_stdin=use_stdin,
            cwd=cwd,
            source=document.source.replace("\r\n", "\n"),
        )
        if result.stderr:
            log_to_output(result.stderr)
    elif use_rpc:
        # This mode is used if the interpreter running this server is different from
        # the interpreter used for running this server.
        log_to_output(" ".join(settings["interpreter"] + ["-m"] + argv))
        log_to_output(f"CWD Linter: {cwd}")

        result = jsonrpc.run_over_json_rpc(
            workspace=code_workspace,
            interpreter=settings["interpreter"],
            module=TOOL_MODULE,
            argv=argv,
            use_stdin=use_stdin,
            cwd=cwd,
            source=document.source,
        )
        if result.exception:
            log_error(result.exception)
            result = utils.RunResult(result.stdout, result.stderr)
        elif result.stderr:
            log_to_output(result.stderr)
    else:
        # In this mode the tool is run as a module in the same process as the language server.
        log_to_output(" ".join([sys.executable, "-m"] + argv))
        log_to_output(f"CWD Linter: {cwd}")
        # This is needed to preserve sys.path, in cases where the tool modifies
        # sys.path and that might not work for this scenario next time around.
        with utils.substitute_attr(sys, "path", sys.path[:]):
            try:
                # `utils.run_module` is equivalent to running `python -m suricata-check`.
                # If your tool supports a programmatic API then replace the function below
                # with code for your tool. You can also use `utils.run_api` helper, which
                # handles changing working directories, managing io streams, etc.
                # Also update `_run_tool` function and `utils.run_module` in `lsp_runner.py`.
                result = utils.run_module(
                    module=TOOL_MODULE,
                    argv=argv,
                    use_stdin=use_stdin,
                    cwd=cwd,
                    source=document.source,
                )
            except Exception:
                log_error(traceback.format_exc(chain=True))
                raise
        if result.stderr:
            log_to_output(result.stderr)

    log_to_output(f"{document.uri} :\r\n{result.stdout}")
    return result


def _run_tool(extra_args: Sequence[str]) -> utils.RunResult:
    """Runs tool."""
    # deep copy here to prevent accidentally updating global settings.
    settings = copy.deepcopy(_get_settings_by_document(None))

    code_workspace = settings["workspaceFS"]
    cwd = settings["workspaceFS"]

    use_path = False
    use_rpc = False
    if len(settings["path"]) > 0:
        # 'path' setting takes priority over everything.
        use_path = True
        argv = settings["path"]
    elif len(settings["interpreter"]) > 0 and not utils.is_current_interpreter(
        settings["interpreter"][0]
    ):
        # If there is a different interpreter set use JSON-RPC to the subprocess
        # running under that interpreter.
        argv = [TOOL_MODULE]
        use_rpc = True
    else:
        # if the interpreter is same as the interpreter running this
        # process then run as module.
        argv = [TOOL_MODULE]

    argv += extra_args

    if use_path:
        # This mode is used when running executables.
        log_to_output(" ".join(argv))
        log_to_output(f"CWD Server: {cwd}")
        result = utils.run_path(argv=argv, use_stdin=True, cwd=cwd)
        if result.stderr:
            log_to_output(result.stderr)
    elif use_rpc:
        # This mode is used if the interpreter running this server is different from
        # the interpreter used for running this server.
        log_to_output(" ".join(settings["interpreter"] + ["-m"] + argv))
        log_to_output(f"CWD Linter: {cwd}")
        result = jsonrpc.run_over_json_rpc(
            workspace=code_workspace,
            interpreter=settings["interpreter"],
            module=TOOL_MODULE,
            argv=argv,
            use_stdin=True,
            cwd=cwd,
        )
        if result.exception:
            log_error(result.exception)
            result = utils.RunResult(result.stdout, result.stderr)
        elif result.stderr:
            log_to_output(result.stderr)
    else:
        # In this mode the tool is run as a module in the same process as the language server.
        log_to_output(" ".join([sys.executable, "-m"] + argv))
        log_to_output(f"CWD Linter: {cwd}")
        # This is needed to preserve sys.path, in cases where the tool modifies
        # sys.path and that might not work for this scenario next time around.
        with utils.substitute_attr(sys, "path", sys.path[:]):
            try:
                # `utils.run_module` is equivalent to running `python -m suricata-check`.
                # If your tool supports a programmatic API then replace the function below
                # with code for your tool. You can also use `utils.run_api` helper, which
                # handles changing working directories, managing io streams, etc.
                # Also update `_run_tool_on_document` function and `utils.run_module` in `lsp_runner.py`.
                result = utils.run_module(
                    module=TOOL_MODULE, argv=argv, use_stdin=True, cwd=cwd
                )
            except Exception:
                log_error(traceback.format_exc(chain=True))
                raise
        if result.stderr:
            log_to_output(result.stderr)

    log_to_output(f"\r\n{result.stdout}\r\n")
    return result


# *****************************************************
# Logging and notification.
# *****************************************************
def log_to_output(
    message: str, msg_type: lsp.MessageType = lsp.MessageType.Log
) -> None:
    LSP_SERVER.show_message_log(message, msg_type)


def log_error(message: str) -> None:
    LSP_SERVER.show_message_log(message, lsp.MessageType.Error)
    if os.getenv("LS_SHOW_NOTIFICATION", "off") in ["onError", "onWarning", "always"]:
        LSP_SERVER.show_message(message, lsp.MessageType.Error)


def log_warning(message: str) -> None:
    LSP_SERVER.show_message_log(message, lsp.MessageType.Warning)
    if os.getenv("LS_SHOW_NOTIFICATION", "off") in ["onWarning", "always"]:
        LSP_SERVER.show_message(message, lsp.MessageType.Warning)


def log_always(message: str) -> None:
    LSP_SERVER.show_message_log(message, lsp.MessageType.Info)
    if os.getenv("LS_SHOW_NOTIFICATION", "off") in ["always"]:
        LSP_SERVER.show_message(message, lsp.MessageType.Info)


# *****************************************************
# Start the server.
# *****************************************************
if __name__ == "__main__":
    LSP_SERVER.start_io()
