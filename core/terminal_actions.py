"""Platform terminal operations; no agent or session dependencies."""

import base64
import os
import shutil
import subprocess
import sys

from prompt_toolkit.output.vt100 import Vt100_Output


def copy_to_clipboard(text: str) -> None:
    """Send text as data over stdin, never interpolate it into shell code."""
    payload = text.encode('utf-8')
    options = {}
    if sys.platform == 'win32':
        executable = shutil.which('powershell.exe') or shutil.which('pwsh.exe')
        if not executable:
            raise RuntimeError('PowerShell clipboard support is unavailable')
        command = [executable, '-NoProfile', '-NonInteractive', '-Command',
                   ("$ErrorActionPreference='Stop'; "
                   "$text=[Text.Encoding]::UTF8.GetString("
                   "[Convert]::FromBase64String([Console]::In.ReadToEnd())); "
                   "Set-Clipboard -Value $text")]
        payload = base64.b64encode(payload)
        options['creationflags'] = subprocess.CREATE_NO_WINDOW
    elif sys.platform == 'darwin':
        command = ['pbcopy']
    elif os.getenv('WAYLAND_DISPLAY') and shutil.which('wl-copy'):
        command = ['wl-copy']
    elif os.getenv('DISPLAY') and shutil.which('xclip'):
        command = ['xclip', '-selection', 'clipboard']
    elif os.getenv('DISPLAY') and shutil.which('xsel'):
        command = ['xsel', '--clipboard', '--input']
    else:
        raise RuntimeError('No clipboard backend available')
    subprocess.run(command, input=payload, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True, timeout=5, **options)


def clear_terminal(output) -> str:
    """Called only while PTK is suspended. ED3 has no acknowledgement."""
    output.erase_screen()
    output.cursor_goto(0, 0)
    vt_output = isinstance(output, Vt100_Output) or isinstance(
        getattr(output, 'vt100_output', None), Vt100_Output,
    )
    if vt_output and os.getenv('TERM') != 'dumb':
        output.write_raw('\x1b[3J')
        message = 'Screen cleared; scrollback clear requested (terminal support varies).'
    else:
        message = 'Screen cleared; scrollback clearing is not confirmed for this terminal.'
    output.flush()
    return message
