"""
Run the model's shell commands inside the task workspace.

In the container the server runs as root and commands run as the unprivileged
`agent` user, which can only see the workspace: the answer key stays in the
server's memory and its code is unreadable. Locally (tests) commands run as the
current user in a temporary workspace.
"""

import os
import pwd
import resource
import signal
import subprocess
from dataclasses import dataclass

MAX_OUTPUT_CHARS = 3000  # protects a ~16k context; long output keeps its head and tail
COMMAND_TIMEOUT_S = 120
AGENT_USER = "agent"


@dataclass
class CommandResult:
    output: str
    exit_code: int


_SWITCH_OK = None  # cached: can this runtime actually drop to the agent user?


def _agent_identity():
    """(uid, gid) of the agent user when we can switch to it, else None.

    Some arena runtimes drop the SETUID/SETGID capabilities, so every attempt to
    switch users fails. Probe once; if switching is blocked, run commands as the
    server's own user instead of failing every command.
    """
    global _SWITCH_OK
    if os.geteuid() != 0:
        return None
    try:
        pw = pwd.getpwnam(AGENT_USER)
    except KeyError:
        return None
    if _SWITCH_OK is None:
        try:
            r = subprocess.run(["id", "-u"], user=pw.pw_uid, group=pw.pw_gid, extra_groups=[],
                               capture_output=True, timeout=10)
            _SWITCH_OK = r.returncode == 0 and r.stdout.strip() == str(pw.pw_uid).encode()
        except Exception:
            _SWITCH_OK = False
    return (pw.pw_uid, pw.pw_gid) if _SWITCH_OK else None


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))   # 6 GiB address space
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024 ** 2, 512 * 1024 ** 2))  # 512 MiB per file
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the start and the end (errors usually come last); say how much was cut."""
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 2 // 3], text[-(limit // 3):]
    return (f"{head}\n[... output truncated: {len(text)} characters in total; showing the first {len(head)} "
            f"and the last {len(tail)}. Use grep, head or sed -n to see specific parts ...]\n{tail}")


def run_bash(command: str, cwd: str, timeout: int = COMMAND_TIMEOUT_S, max_chars: int = MAX_OUTPUT_CHARS) -> CommandResult:
    ident = _agent_identity()
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": cwd, "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1"}
    kwargs = dict(cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                  stdin=subprocess.DEVNULL, start_new_session=True)
    if ident:
        kwargs.update(user=ident[0], group=ident[1], extra_groups=[], preexec_fn=_limits)
    proc = subprocess.Popen(["bash", "-c", command], **kwargs)
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
    finally:
        try:  # clean up anything the command left running in the background
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    text = (out or b"").decode("utf-8", errors="replace")
    if timed_out:
        text += f"\n[command timed out after {timeout} seconds]"
    return CommandResult(truncate(text, max_chars), 124 if timed_out else proc.returncode)
