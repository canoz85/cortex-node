# Local terminal commands

Submit `/help`, `/status`, `/copy` or `/clear` with Alt+Enter or F2. These
commands also work while an execution is running. They do not enter prompt
history, invoke the agent, or save the session. Normal application shutdown
still uses the existing session-save behavior.

`core/tui_commands.py` owns the command registry. Register a `Command` with its
name, description, handler and optional usage string; `/help` reads this registry
automatically. The current commands take no arguments. Styling is isolated in
`core/terminal_style.py`; `NO_COLOR`, a dumb terminal, or monochrome output
disables command colors.

`/copy` uses the raw successful Finalizer render recorded by the logging
presentation layer, never console text or arbitrary callback return values.
Markdown and whitespace are preserved while ANSI sequences are removed. A
successfully rendered explanation of a failed execution is still a valid answer;
a Finalizer render-error placeholder is not. The previous answer survives later
clarifications, errors and `/clear`. Answers from before this process started
are not recovered from the bounded session history, which lacks render-error
metadata and may omit long answers.

Clipboard writing runs off the UI thread, with a five-second subprocess timeout.
Windows uses PowerShell Set-Clipboard with UTF-8 text encoded over stdin; macOS
uses pbcopy. Linux requires wl-copy under Wayland or xclip/xsel under X11. No
clipboard package is installed automatically. Headless/remote sessions may lack
clipboard access; the command reports failure and suggests terminal selection.

`/status` displays allowlisted configuration and live session information.
Usage is explicitly current-turn or last-turn usage, not a session lifetime
total. Provider invocations and component tokens come from LiveStatus. The
session's completed-turn count is displayed separately. Unknown information is
omitted or marked unavailable; no secret-bearing configuration dump is used.

`/clear` is an ordered action in TerminalStream. Earlier output, including any
partial line, is flushed before clearing; later output is retained. It runs with
the editor suspended through run_in_terminal, which resets and redraws the
bottom-aligned prompt afterward. It does not alter session or execution state.
VT terminals receive ED3 (CSI 3 J), a scrollback-clear request with no portable
acknowledgement. The command reports that support varies; other backends do not
claim scrollback clearing. No alternate screen or mouse reporting is enabled.

Manual host checks: in VS Code, test clear with both short and overflowing
history, resize during execution, native selection/copy, Unicode clipboard
content, multiline drafts, Alt+Enter and Ctrl+Q. Automated tests use mocked
clipboard backends and terminal output; they do not overwrite the real clipboard.
