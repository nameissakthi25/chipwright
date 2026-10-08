"""ASR streaming-encoder modality runner (a slice of what a `chipwright-asr` package would own).

Runs ONE encoder window two ways on the same input and hands back both outputs for the verifier:
  - NPU: quantize inputs with the artifact's own I/O scales, execute the w8a16 context binary.
  - CPU: the float ONNX through ONNX Runtime — the reference.

The streaming encoder takes an audio feature window plus cached state. For a single window the
state is zero, which sidesteps the quant-layout transposes the per-chunk loop needs — enough to
prove the Run verb and produce an honest cosine.
"""
from __future__ import annotations

import json
import numpy as np

_NPDT = {"QNN_DATATYPE_UFIXED_POINT_16": np.uint16, "QNN_DATATYPE_INT_32": np.int32,
         "QNN_DATATYPE_FLOAT_32": np.float32}


def _dims(d):
    return [1 if (isinstance(x, str) or x in (0, None)) else x for x in d]


def _load_meta(meta_path):
    g = json.load(open(meta_path))["info"]["graphs"][0]["info"]
    def tl(key):
        out = []
        for t in g[key]:
            i = t["info"]; qp = i.get("quantizeParams", {}) or {}
            so = qp.get("scaleOffset") if isinstance(qp, dict) else None
            out.append(dict(name=i["name"], dtype=i["dataType"], dims=list(i["dimensions"]),
                            scale=(so or {}).get("scale"), offset=(so or {}).get("offset")))
        return out
    return g["graphName"], tl("graphInputs"), tl("graphOutputs")


def _q16(real, scale, offset):
    return np.clip(np.rint(real / scale) - offset, 0, 65535).astype(np.uint16)


def _dq16(q, scale, offset):
    return (q.astype(np.float32) + offset) * scale


def run(device, ctx_board_path, meta_path, cpu_onnx, input_npy, window=39):
    """Return (npu_out, cpu_out) dicts keyed by 'encoder_out'."""
    import onnxruntime as ort

    gname, gin, gout = _load_meta(meta_path)
    imap = {t["name"]: t for t in gin}
    enc_out_spec = next(t for t in gout if t["name"] == "encoder_out")

    feats = np.load(input_npy).astype(np.float32)             # [frames, 80]
    if feats.shape[0] < window:
        feats = np.pad(feats, ((0, window - feats.shape[0]), (0, 0)), constant_values=feats.min())
    x = feats[:window][None, :, :]                            # [1, 39, 80] (float, onnx layout)

    # ---- CPU reference (float ONNX) --------------------------------------------------------------
    sess = ort.InferenceSession(cpu_onnx, providers=["CPUExecutionProvider"])
    cpu_feeds = {"x": x.astype(np.float32)}
    for i in sess.get_inputs():
        if i.name == "x":
            continue
        shp = _dims(i.shape)
        cpu_feeds[i.name] = np.zeros(shp, np.float32 if i.type == "tensor(float)" else np.int64)
    onames = [o.name for o in sess.get_outputs()]
    cpu_vals = dict(zip(onames, sess.run(None, cpu_feeds)))
    cpu_out = {"encoder_out": cpu_vals["encoder_out"]}

    # ---- NPU (quantize with the artifact's scales, run the context binary) -----------------------
    native = {}
    xi = imap["x"]
    native["x"] = _q16(np.transpose(x, (0, 2, 1)), xi["scale"], xi["offset"])  # -> [1,80,39]
    for t in gin:
        n = t["name"]
        if n == "x":
            continue
        shp = _dims(t["dims"])
        if t["dtype"] == "QNN_DATATYPE_INT_32":
            native[n] = np.zeros(shp, np.int32)
        else:                                                 # zero state -> quantize zeros
            native[n] = _q16(np.zeros(shp, np.float32), t["scale"], t["offset"])
    out_shape = tuple(_dims(enc_out_spec["dims"]))
    raw = device.run(ctx_board_path, native,
                     [{"name": "encoder_out", "dtype": "uint16", "shape": out_shape}])
    npu_eo = _dq16(raw["encoder_out"], enc_out_spec["scale"], enc_out_spec["offset"])
    # align shapes (CPU may be [1,T,512]; NPU dims from meta) for a fair cosine
    npu_out = {"encoder_out": npu_eo.reshape(cpu_out["encoder_out"].shape)
               if npu_eo.size == cpu_out["encoder_out"].size else npu_eo}
    return npu_out, cpu_out
