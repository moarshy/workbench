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

MAX_OUTPUT_CHARS = 8192
COMMAND_TIMEOUT_S = 120
AGENT_USER = "agent"


@dataclass
class CommandResult:
    output: str
    exit_code: int


def _agent_identity():
    """(uid, gid) of the agent user when we can switch to it, else None."""
    if os.geteuid() != 0:
        return None
    try:
        pw = pwd.getpwnam(AGENT_USER)
        return pw.pw_uid, pw.pw_gid
    except KeyError:
        return None


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))   # 6 GiB address space
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024 ** 2, 512 * 1024 ** 2))  # 512 MiB per file
    resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[output truncated: {len(text)} characters in total, showing the first {limit}]"


def run_bash(command: str, cwd: str, timeout: int = COMMAND_TIMEOUT_S) -> CommandResult:
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
    return CommandResult(truncate(text), 124 if timed_out else proc.returncode)
