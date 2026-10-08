"""Remote QNN runner — push a context binary to the board and execute it over SSH.

The device transport the `run` verb uses. It is deliberately quantization-agnostic: it moves raw
bytes in the graph's native dtype and runs `qnn-net-run --retrieve_context`. Quant/dequant belongs
to the modality runner, which owns the artifact's I/O scales. (Generalized from the proven
board-side driver; here it drives the board from the host.)
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
import time

import numpy as np


def _ssh(pw, port, dest, *cmd, timeout=600, capture=True):
    base = ["sshpass", "-p", pw, "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
            "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no",
            "-p", str(port), dest, *cmd]
    return subprocess.run(base, capture_output=capture, text=True, timeout=timeout)


def _scp(pw, port, *args, timeout=1800):
    base = ["sshpass", "-p", pw, "scp", "-P", str(port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
            "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no", *args]
    return subprocess.run(base, capture_output=True, text=True, timeout=timeout)


class RemoteQNN:
    def __init__(self, host, user, pw, port=22, backend="/usr/lib/libQnnHtp.so",
                 board_dir="/opt/chipwright"):
        self.host, self.user, self.pw, self.port = host, user, pw, int(port)
        self.backend, self.board_dir = backend, board_dir
        self.dest = f"{user}@{host}"

    def _retry(self, fn, tries=4, wait=3):
        last = None
        for i in range(tries):
            try:
                r = fn()
                if getattr(r, "returncode", 0) == 0:
                    return r
                last = r
            except Exception as e:  # pragma: no cover
                last = e
            time.sleep(wait)
        return last

    def ensure_ctx(self, local_path, name):
        """Push the context binary to the board once; return its board path."""
        board_path = f"{self.board_dir}/{name}"
        self._retry(lambda: _ssh(self.pw, self.port, self.dest, f"mkdir -p {self.board_dir}", timeout=40))
        present = _ssh(self.pw, self.port, self.dest, f"test -f {board_path} && echo OK", timeout=40)
        if "OK" not in (present.stdout or ""):
            r = self._retry(lambda: _scp(self.pw, self.port, local_path, f"{self.dest}:{board_path}"))
            if getattr(r, "returncode", 1) != 0:
                raise RuntimeError(f"failed to push context binary: {getattr(r,'stderr','')}")
        return board_path

    def run(self, ctx_board_path, feeds_native, outputs):
        """feeds_native: {name: ndarray in the graph's native dtype}. outputs: [{name,dtype,shape}].
        Returns {name: ndarray[shape]} in native dtype (modality dequantizes)."""
        dt = {"uint16": np.uint16, "int32": np.int32, "float32": np.float32}
        with tempfile.TemporaryDirectory() as td:
            parts = []
            for n, a in feeds_native.items():
                np.ascontiguousarray(a).tofile(f"{td}/{n}.raw")
                parts.append(f"{n}:={n}.raw")
            open(f"{td}/il.txt", "w").write(" ".join(parts) + "\n")
            rdir = f"{self.board_dir}/req_{int(time.time()*1000)}"
            self._retry(lambda: _ssh(self.pw, self.port, self.dest, f"mkdir -p {rdir}", timeout=40))
            files = [f"{td}/{f}" for f in os.listdir(td)]
            r = self._retry(lambda: _scp(self.pw, self.port, *files, f"{self.dest}:{rdir}/"))
            if getattr(r, "returncode", 1) != 0:
                raise RuntimeError("failed to push inputs")
            env = "export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp:/usr/lib LD_LIBRARY_PATH=/usr/lib"
            cmd = (f"cd {rdir} && {env} && qnn-net-run --backend {self.backend} "
                   f"--retrieve_context {ctx_board_path} --input_list il.txt --output_dir out "
                   f"--use_native_input_files --use_native_output_files")
            run = self._retry(lambda: _ssh(self.pw, self.port, self.dest, cmd, timeout=600))
            if getattr(run, "returncode", 1) != 0:
                _ssh(self.pw, self.port, self.dest, f"rm -rf {rdir}", timeout=30)
                raise RuntimeError("qnn-net-run failed:\n" + (getattr(run, "stdout", "") or "")[-500:])
            self._retry(lambda: _scp(self.pw, self.port, "-r", f"{self.dest}:{rdir}/out", td))
            _ssh(self.pw, self.port, self.dest, f"rm -rf {rdir}", timeout=30)
            res = {}
            for spec in outputs:
                nm, d, sh = spec["name"], dt[spec.get("dtype", "float32")], tuple(spec["shape"])
                fs = (glob.glob(f"{td}/out/**/{nm}_native.raw", recursive=True)
                      or glob.glob(f"{td}/out/**/{nm}.raw", recursive=True)
                      or glob.glob(f"{td}/out/**/{nm}*.raw", recursive=True))
                if not fs:
                    raise RuntimeError(f"missing output {nm}")
                res[nm] = np.fromfile(fs[0], d).reshape(sh)
            return res
