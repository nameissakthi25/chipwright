"""ADB transport — push an LLM bundle to an Android board and drive it over adb.

The device layer the `llm` modality uses. The existing SSH transport (`device.py::RemoteQNN`) reaches
a Linux board (QCS6490/RB3, the QDC fleet) over `ssh`/`scp`; a Snapdragon *phone* class target speaks
`adb` instead. This module is the adb parallel — same shape (`shell`/`push`/`pull`/`exists`,
retry-wrapped, a preflight that fails by name) so the modality can hold one interface and pick a
transport by target, exactly as the resolver holds one grammar across arches.

Two things are deliberate and load-bearing (both learned on real silicon, see spec 002 edge cases):

  * **push one file at a time.** A single multi-GB `adb push` of a *directory* (an LLM Genie bundle is
    N context binaries, often several GB) can wedge the device's adb daemon — the push stalls and every
    later command hangs. `push_file` moves exactly one regular file and refuses a directory by name, so
    the modality walks the bundle file-by-file rather than handing adb a tree that bricks the session.
  * **`adb` is a system binary, not a pip dep.** Like `sshpass` for the SSH transport, it is shelled
    out to; `_require_adb()` fails here, by name, with the install hint — not later with a cryptic
    FileNotFoundError from subprocess.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time


def _require_adb():
    """`adb` is a system binary (Android platform-tools), not a pip package, the transport shells to.

    Fail here, by name, with the install hint — mirroring `device.py::_require_sshpass` — so a missing
    system dependency is reported the way the framework reports everything: plainly, naming the fix.
    """
    if shutil.which("adb") is None:
        raise RuntimeError(
            "chipwright llm needs the `adb` system binary to reach the board over Android Debug "
            "Bridge, and it is not on PATH. Install the Android platform-tools: "
            "`brew install --cask android-platform-tools` (macOS) or "
            "`apt-get install android-tools-adb` (Debian/Ubuntu).")


class AdbDevice:
    """Drive one Android board over adb. `serial` disambiguates when several are attached.

    The surface the LLM modality calls: `shell(cmd) -> CompletedProcess`, `push_file(local, remote)`
    (one regular file), `pull(remote, local)`, `exists(remote) -> bool`. Every call is wrapped in the
    same bounded retry as `RemoteQNN` because adb — like ssh to a thermally-throttling board — drops a
    command now and then and a blind retry is the right, honest fix.
    """

    def __init__(self, serial=None, board_dir="/data/local/tmp/chipwright", adb_bin="adb"):
        _require_adb()
        self.serial = serial
        self.board_dir = board_dir
        self.adb_bin = adb_bin

    # --- plumbing -----------------------------------------------------------------------------
    def _base(self):
        """The adb argv prefix, pinned to this device's serial when one was given."""
        return [self.adb_bin] + (["-s", self.serial] if self.serial else [])

    def _run(self, *args, timeout=600):
        return subprocess.run(self._base() + list(args), capture_output=True, text=True, timeout=timeout)

    def _retry(self, fn, tries=4, wait=3):
        """Run `fn` up to `tries` times; return the first returncode==0 result, else the last result/err.

        Identical discipline to `RemoteQNN._retry` — the caller inspects `returncode` and decides; a
        transport blip is retried, a real failure is returned so it can be reported by name.
        """
        last = None
        for _ in range(tries):
            try:
                r = fn()
                if getattr(r, "returncode", 0) == 0:
                    return r
                last = r
            except Exception as e:  # pragma: no cover
                last = e
            time.sleep(wait)
        return last

    # --- the interface the modality calls -----------------------------------------------------
    def shell(self, cmd, timeout=600):
        """Run `cmd` on the board (`adb shell`) and return the CompletedProcess (retry-wrapped)."""
        return self._retry(lambda: self._run("shell", cmd, timeout=timeout))

    def push_file(self, local, remote, timeout=1800):
        """Push ONE regular file to `remote`. Refuses a directory by name.

        A directory push is the exact thing that wedges the adb daemon on a multi-GB bundle, so we make
        it an error here rather than let a caller do it by accident — the modality is expected to walk
        the bundle and push each file. Ensures the remote parent dir exists first.
        """
        if not os.path.isfile(local):
            if os.path.isdir(local):
                raise ValueError(
                    f"push_file is one file at a time by design (a multi-GB directory push wedges the "
                    f"adb daemon); {local!r} is a directory — walk it and push each file.")
            raise FileNotFoundError(f"no such file to push: {local!r}")
        parent = os.path.dirname(remote.rstrip("/"))
        if parent:
            self._retry(lambda: self._run("shell", f"mkdir -p {parent}", timeout=40))
        r = self._retry(lambda: self._run("push", local, remote, timeout=timeout))
        if getattr(r, "returncode", 1) != 0:
            raise RuntimeError(f"adb push failed ({local} -> {remote}): {getattr(r, 'stderr', '')}")
        return remote

    def pull(self, remote, local, timeout=1800):
        """Pull `remote` off the board to the local path `local` (retry-wrapped)."""
        parent = os.path.dirname(local.rstrip("/"))
        if parent:
            os.makedirs(parent, exist_ok=True)
        r = self._retry(lambda: self._run("pull", remote, local, timeout=timeout))
        if getattr(r, "returncode", 1) != 0:
            raise RuntimeError(f"adb pull failed ({remote} -> {local}): {getattr(r, 'stderr', '')}")
        return local

    def exists(self, remote, timeout=40) -> bool:
        """True iff `remote` is present on the board (`test -e … && echo OK`)."""
        r = self.shell(f"test -e {remote} && echo OK", timeout=timeout)
        return "OK" in (getattr(r, "stdout", "") or "")
