"""Qualcomm backend adapter — runs a QNN HTP context binary on a QCS6490 over SSH.

Vendor-specific grunt work lives here; the runtime above it is vendor-neutral. Generalized from the
Laya `board_encode` into a named-I/O device so any single-graph QNN model can use it.

Interface the runtime relies on:
    dev = QualcommDevice(ctx_bin, access)          # access: dict from bundle device_access + env
    outs = dev.run(feeds, outputs)                 # feeds/outs: {name: np.ndarray} (leading batch dim)
"""
import os, glob, time, tempfile, subprocess, numpy as np

_DTYPE = {"float32": np.float32, "int32": np.int32, "int64": np.int64, "uint8": np.uint8}


class QualcommDevice:
    def __init__(self, ctx_bin, access, backend_lib="/usr/lib/libQnnHtp.so"):
        self.ctx = ctx_bin                      # path to the context binary ON the board
        self.lib = backend_lib
        self.host = os.environ[access["host_env"]]
        self.user = os.environ[access["user_env"]]
        self.pw = os.environ[access["pass_env"]]
        self.board_dir = access.get("board_dir", "/home/denc")

    def _ssh(self, *c):
        return ["sshpass", "-p", self.pw, "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
                "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no",
                f"{self.user}@{self.host}", *c]

    def _scp(self, *a):
        return ["sshpass", "-p", self.pw, "scp", "-r", "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=20",
                "-o", "PreferredAuthentications=password", "-o", "PubkeyAuthentication=no", *a]

    def run(self, feeds, outputs):
        """feeds: {name: ndarray[B, ...]}; outputs: [{name, dtype, shape(per-sample)}]. Returns {name: ndarray[B, ...]}.

        One qnn-net-run per call (one HTP context create/destroy) — batch many samples per call to
        stay within the DSP budget (per-sample calls can wedge the CDSP)."""
        names = list(feeds)
        B = feeds[names[0]].shape[0]
        with tempfile.TemporaryDirectory() as td:
            lines = []
            for i in range(B):
                parts = []
                for n in names:
                    feeds[n][i].astype(feeds[n].dtype).tofile(f"{td}/{n}_{i}.raw")
                    parts.append(f"{n}:={n}_{i}.raw")
                lines.append(" ".join(parts))
            open(f"{td}/il.txt", "w").write("\n".join(lines) + "\n")
            rdir = f"{self.board_dir}/req_{int(time.time()*1000)}"
            subprocess.run(self._ssh(f"mkdir -p {rdir}"), check=True, timeout=30)
            subprocess.run(self._scp(*[f"{td}/{f}" for f in os.listdir(td)],
                                     f"{self.user}@{self.host}:{rdir}/"), check=True, timeout=300)
            env = "export ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp && export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH"
            cmd = (f"cd {rdir} && {env} && qnn-net-run --backend {self.lib} "
                   f"--retrieve_context {self.ctx} --input_list il.txt --output_dir out")
            r = subprocess.run(self._ssh(cmd), timeout=600, capture_output=True, text=True)
            if r.returncode != 0:
                subprocess.run(self._ssh(f"rm -rf {rdir}"), timeout=30)
                raise RuntimeError("qnn-net-run failed:\n" + (r.stderr or r.stdout)[-400:])
            subprocess.run(self._scp(f"{self.user}@{self.host}:{rdir}/out", td), check=True, timeout=300)
            subprocess.run(self._ssh(f"rm -rf {rdir}"), timeout=30)
            out = {}
            for spec in outputs:
                nm, dt, sh = spec["name"], _DTYPE[spec.get("dtype", "float32")], tuple(spec["shape"])
                arr = np.zeros((B, *sh), dtype=dt)
                # qnn writes outputs in graph order; a single-output graph -> Result_i/<out>.raw
                for i in range(B):
                    f = sorted(glob.glob(f"{td}/out/Result_{i}/*.raw"))
                    arr[i] = np.fromfile(f[0], dt).reshape(sh)
                out[nm] = arr
            return out
