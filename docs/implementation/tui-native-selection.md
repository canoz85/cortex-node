# TUI v2: native scrollback and selection

## Current architecture

`CortexTUI` uses a normal-buffer prompt_toolkit `Application` containing only a
fixed five-line multiline editor and a live status row. It does not enter the
alternate screen, enable mouse reporting, render an output pane, or own output
scrolling/focus. The host terminal owns wheel scrolling, ordinary drag selection,
Copy, scrollback navigation, retention, and follow behavior.

The existing renderer continues printing the same permanent text, including
compact OK/FAIL tool results and authoritative final answers. `TerminalStream`
batches complete console lines from stdout, stderr, and existing console logging
handlers into a thread-safe queue. A UI-loop printer drains that queue through
prompt_toolkit's `run_in_terminal`: temporarily erase the editor, print above it,
and redraw it without resetting its buffer or cursor. ANSI/text is preserved.
Partial final lines are flushed at shutdown. Diagnostic file handlers and raw
LLM logging remain untouched.

Turns use the existing synchronous completion callback in a worker thread, one
at a time. The editor stays active for drafting while a turn runs. Shutdown waits
for the active turn and queued output. Existing `LiveStatus(enabled=False)`
snapshots supply the toolbar; the plain-console spinner never enters scrollback.
Periodic toolbar redraws do not enqueue permanent text.

This is a presentation change. Planner, Brain, Controller, protocol, tool,
final-answer, session, MQTT, and single-prompt behavior are unchanged.

## Bindings and removed behavior

- Enter: newline; F2 / Alt-Enter: submit the whole multiline prompt.
- Ctrl-Up / Ctrl-Down: library prompt history; Up / Down: move within the draft.
- Ctrl-C: clear draft when delivered to the application. A terminal selection is
  copied by the host first, using the host's normal Copy command.
- Ctrl-Q, Ctrl-D on an empty editor, or submitting exit/quit: finish and exit.
- Vi mode uses the library's editing mode and remains optional/default-off.

Removed: F6 output focus, output-pane paging, mouse-wheel routing, custom mouse
handlers, Ctrl-Home/Ctrl-End output navigation, the local output-history buffer,
and application auto-follow state. Page/scrollback shortcuts belong to the host
or the library's normal prompt editor; CortexNode adds no output bindings.

Browsing host scrollback can move the editor out of view. The editor and toolbar
remain intact at the live end; return to the bottom before typing. History
retention and copy shortcuts depend on host configuration. The editor occupies a
fixed five-line region at the bottom of the live content, not a separate pinned
viewport while browsing old terminal history.

## Windows VS Code validation

Tested on Windows with VS Code 1.143.1 and prompt_toolkit 3.0.53. A disposable
integrated terminal ran the production v2 shell with synthetic history and a
long-running synthetic turn, without models or user session changes.

Verified:

- Wheel moved through real host scrollback with alternate-screen and mouse
  reporting both disabled. The draft, cursor, and prompt history stayed intact.
- An ordinary unmodified drag selected output. Normal Ctrl+C copied it; VS Code's
  clipboard API confirmed the known synthetic marker. No arbitrary clipboard
  contents were saved.
- New background events appeared once above the prompt. The draft and cursor
  were unchanged by output; normal typing and Backspace remained usable.
- The live toolbar advanced elapsed time while showing phase, step/title,
  read_file, token total, and editing mode. Its refreshes did not increase the
  recorded permanent-output line count or add RUNNING/status entries.
- Returning to the live end restored the visible editor. The test window was
  separate from the user's active terminal.

Focused tests also exercise multiline submission, both editing modes, prompt
history, sequential turns, output draining at exit, raw diagnostic/file-handler
preservation, terminal outcomes, and plain-console fallback.

## Why v1 was replaced

The earlier fixed output pane required coordinate-aware wheel reporting. In the
installed VS Code terminal, enabling tracking prevented ordinary native drag
selection; disabling it in the alternate screen converted the wheel to ordinary
Up/Down input without coordinates. Parsing those keys differently would change
prompt editing and could not distinguish the output area from the editor.

Tracking modes 1000/1002/1003 couple buttons and wheel delivery. SGR mode 1006 is
an encoding, not a wheel-only subscription. Ignoring reports in Python cannot
replay them as native selection. See [xterm.js mouse protocols](https://github.com/xtermjs/xterm.js/blob/master/src/common/services/MouseStateService.ts).
Upstream xterm.js has a host-side `mouseEventsRequireAlt` policy exempting wheel
reports, but the installed VS Code does not expose that option to this app.

V2 avoids this coupling by putting output in the normal terminal buffer and
letting the host own both selection and scrolling. It does not implement custom
selection, clipboard logic, mouse protocols, or host-specific settings changes.
