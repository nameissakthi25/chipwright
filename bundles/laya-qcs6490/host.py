"""Laya host glue for the QCS6490 bundle.

pre  (encode_inputs): text+question -> {inputs_embeds, gmask, smask}  (embedding lookup + masks on host)
post (decode_outputs): NPU hidden states -> calibrated typed decision  (marker/scorer head + temp cal)

Uses the validated capture->replay pattern so the decision head runs through laya's own `ag.predict`
exactly (no head re-implementation): pre captures the exact tokenization + swaps in a replay encoder;
post feeds the NPU hidden back through `ag.predict`.
"""
import numpy as np, onnxruntime as ort, torch
from types import SimpleNamespace

_S = {}

def _agent():
    if "ag" not in _S:
        from laya.agent import Agent
        ag = Agent("convaiinnovations/laya", compile=False, device="cpu"); ag.model.eval()
        _S["ag"] = ag
        _S["W"] = ag.model.encoder.get_input_embeddings().weight.detach().numpy().astype(np.float32)
        _S["cfg"] = ag.model.encoder.config
    return _S["ag"]


class _Capture(torch.nn.Module):
    def __init__(self, cfg, sink): super().__init__(); self.config = cfg; self.sink = sink
    def forward(self, input_ids=None, attention_mask=None, **kw):
        self.sink.append((input_ids.cpu().numpy()[0].copy(), attention_mask.cpu().numpy()[0].copy()))
        H = _S["W"].shape[1]
        return SimpleNamespace(last_hidden_state=torch.zeros(input_ids.shape[0], input_ids.shape[1], H))


class _Replay(torch.nn.Module):
    def __init__(self, cfg): super().__init__(); self.config = cfg; self.hidden = None
    def forward(self, input_ids=None, attention_mask=None, **kw):
        s = input_ids.shape[1]
        return SimpleNamespace(last_hidden_state=torch.from_numpy(self.hidden[:s])[None].to(input_ids.device))


def encode_inputs(payload, ctx, env):
    ag = _agent(); W = _S["W"]; cfg = _S["cfg"]
    L = int(env["graph"]["context_len"]); H = W.shape[1]
    mask = ort.InferenceSession(env["artifacts"][env["mask_id"]], providers=["CPUExecutionProvider"])
    eye = np.eye(L, dtype=bool)[None, None]
    items = payload["items"]; ctx["items"] = items
    cap = []
    ag.model.encoder = _Capture(cfg, cap)
    with torch.no_grad():
        for text, q in items:
            ag.predict(text, q, max_len=L)
    E = []; G = []; Sm = []
    for ids, am in cap:
        n = min(len(ids), L); idp = np.zeros(L, np.int64); idp[:n] = ids[:n]
        amp = np.zeros(L, np.int64); amp[:n] = am[:n]
        g, s = mask.run(None, {"attention_mask": amp[None].astype(np.int64)})
        E.append(W[idp]); G.append((np.asarray(g).astype(bool) | eye).astype(np.int32)[0])
        Sm.append((np.asarray(s).astype(bool) | eye).astype(np.int32)[0])
    ctx["replay"] = _Replay(cfg); ag.model.encoder = ctx["replay"]
    return {"inputs_embeds": np.stack(E).astype(np.float32), "gmask": np.stack(G), "smask": np.stack(Sm)}


def decode_outputs(outs, ctx, env):
    ag = _agent(); hidden = outs["hidden"]; L = int(env["graph"]["context_len"]); results = []
    for j, (text, q) in enumerate(ctx["items"]):
        ctx["replay"].hidden = hidden[j]
        with torch.no_grad():
            r = ag.predict(text, q, max_len=L)
        results.append(r["answers"])
    return results
