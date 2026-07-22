"""Train a tiny MLP on the feature CSV and export weights for the C++ forward pass.

Usage:
    python train_model.py ../data/features.csv --out ../data/model

Reads the feature+label CSV, builds a stationary feature matrix, splits it in time
order, standardizes on the train split only, trains a small MLP, then exports:
    <out>.json   architecture + feature list + scaler mean/std
    <out>.bin    float32 weights and biases, layer order, row-major
    <out>.onnx   same model in ONNX (for the later TensorRT path)
It also recomputes the forward pass in NumPy from the exported bytes and checks it
matches PyTorch, so the export format is proven before the C++ loader is written.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Stationary inputs. Raw mid price is dropped and replaced by ret_1.
FEATURES = ["ret_1", "volatility", "imbalance", "intensity"]
HIDDEN = [16, 16]
SEED = 0
EPOCHS = 40
BATCH = 4096
LR = 1e-3
VAL_FRAC = 0.15
TEST_FRAC = 0.15


class MLP(nn.Module):
    """input -> (Linear, ReLU) x len(hidden) -> Linear -> 1 logit."""

    def __init__(self, in_dim, hidden, out_dim=1):
        super().__init__()
        dims = [in_dim] + hidden
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.ReLU()]
        blocks += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def load_xy(path):
    """Load CSV, derive ret_1 from mid, return feature matrix X and label y."""
    df = pd.read_csv(path)
    df["ret_1"] = np.log(df["mid"]).diff()
    df = df.dropna().reset_index(drop=True)
    x = df[FEATURES].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.float64)
    return x, y


def time_split(n, val_frac, test_frac):
    """Chronological split: earliest rows train, latest rows test. No shuffle."""
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def standardize(x_train, x_other):
    """Fit mean/std on train, apply to every split. Returns scaled arrays + stats."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    scaled = [(x - mean) / std for x in x_other]
    return mean, std, scaled


def accuracy(logits, y):
    pred = (torch.sigmoid(logits) >= 0.5).float().squeeze(1)
    return (pred == y).float().mean().item()


def train(x_tr, y_tr, x_va, y_va, in_dim):
    torch.manual_seed(SEED)
    model = MLP(in_dim, HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    x_tr_t = torch.tensor(x_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    x_va_t = torch.tensor(x_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    n = x_tr_t.shape[0]
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            logits = model(x_tr_t[b]).squeeze(1)
            loss = loss_fn(logits, y_tr_t[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_acc = accuracy(model(x_va_t), y_va_t)
        print(f"epoch {epoch + 1:2d}/{EPOCHS}  val_acc {va_acc:.4f}")

    return model


def export(model, mean, std, out, in_dim):
    """Write model.json, model.bin (float32, layer order, row-major), model.onnx."""
    layers = []
    blob = bytearray()
    for name, p in model.state_dict().items():
        arr = p.detach().numpy().astype(np.float32)
        layers.append({"name": name, "shape": list(arr.shape)})
        blob += arr.tobytes(order="C")

    meta = {
        "input_dim": in_dim,
        "hidden": HIDDEN,
        "output_dim": 1,
        "features": FEATURES,
        "activation": "relu",
        "output": "logit (apply sigmoid for probability)",
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "weight_layout": "row-major, Linear weight is [out, in]; y = x @ W.T + b",
        "layers": layers,
    }
    Path(f"{out}.json").write_text(json.dumps(meta, indent=2))
    Path(f"{out}.bin").write_bytes(bytes(blob))

    dummy = torch.zeros(1, in_dim, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, f"{out}.onnx",
        input_names=["features"], output_names=["logit"],
        dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        verbose=False,
    )
    return meta


def write_manifest(meta, out):
    """Flat-text manifest the C++ loader reads with plain >> (no JSON library).

    Format (whitespace-separated, one field per token, comments start with #):
        input_dim <n>
        output_dim <n>
        scaler_mean <v0> <v1> ...        # input_dim values
        scaler_std  <v0> <v1> ...        # input_dim values
        num_layers <k>                   # number of Linear layers
        layer <out> <in>                 # repeated k times, in forward order
        bin <path to model.bin>
    The .bin holds float32 weights then biases per layer, row-major, in this order.
    """
    lines = []
    lines.append("# gpu-inference-engine model manifest")
    lines.append(f"input_dim {meta['input_dim']}")
    lines.append(f"output_dim {meta['output_dim']}")
    lines.append("scaler_mean " + " ".join(repr(v) for v in meta["scaler_mean"]))
    lines.append("scaler_std " + " ".join(repr(v) for v in meta["scaler_std"]))

    # Each Linear contributes a weight [out, in] then a bias [out] in meta.layers.
    weights = [l for l in meta["layers"] if l["name"].endswith(".weight")]
    lines.append(f"num_layers {len(weights)}")
    for w in weights:
        out_dim, in_dim = w["shape"]
        lines.append(f"layer {out_dim} {in_dim}")
    lines.append(f"bin {Path(out).name}.bin")
    Path(f"{out}.meta").write_text("\n".join(lines) + "\n")


def write_check(x_rows, logits, out):
    """Reference rows for the C++ match test: raw (unscaled) features + torch logit.

    C++ reads this, runs its own forward pass on the same raw features, and asserts
    max|c++ - reference| < 1e-5. No torch needed at C++ run time.
    Format: first line "<num_rows> <num_features>", then one row per line:
        f0 f1 ... f{d-1} logit
    """
    n, d = x_rows.shape
    lines = [f"{n} {d}"]
    for i in range(n):
        feats = " ".join(repr(float(v)) for v in x_rows[i])
        lines.append(f"{feats} {repr(float(logits[i][0]))}")
    Path(f"{out}.check").write_text("\n".join(lines) + "\n")


def numpy_forward(x, meta, bin_path):
    """Re-run the forward pass in NumPy from the exported bytes. Reference for C++."""
    raw = np.frombuffer(Path(bin_path).read_bytes(), dtype=np.float32)
    off = 0
    tensors = []
    for layer in meta["layers"]:
        size = int(np.prod(layer["shape"]))
        tensors.append(raw[off:off + size].reshape(layer["shape"]))
        off += size

    a = x.astype(np.float32)
    n_linear = len(tensors) // 2
    for i in range(n_linear):
        w = tensors[2 * i]      # [out, in]
        b = tensors[2 * i + 1]  # [out]
        a = a @ w.T + b
        if i < n_linear - 1:
            a = np.maximum(a, 0.0)  # ReLU on hidden layers only
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="../data/model")
    args = ap.parse_args()

    x, y = load_xy(args.csv)
    in_dim = x.shape[1]
    tr, va, te = time_split(len(x), VAL_FRAC, TEST_FRAC)

    mean, std, (x_tr, x_va, x_te) = standardize(x[tr], [x[tr], x[va], x[te]])
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    base = max(y_te.mean(), 1.0 - y_te.mean())
    print(f"rows {len(x):,}  in_dim {in_dim}  "
          f"train {len(tr):,} val {len(va):,} test {len(te):,}")
    print(f"majority-class baseline (test): {base:.4f}")

    model = train(x_tr, y_tr, x_va, y_va, in_dim)

    model.eval()
    with torch.no_grad():
        te_acc = accuracy(model(torch.tensor(x_te, dtype=torch.float32)),
                          torch.tensor(y_te, dtype=torch.float32))
    print(f"test_acc {te_acc:.4f}  (baseline {base:.4f})")

    meta = export(model, mean, std, args.out, in_dim)
    write_manifest(meta, args.out)

    # Prove the export: torch logits vs NumPy-from-bytes on the first test rows.
    sample = x_te[:256]
    with torch.no_grad():
        torch_logits = model(torch.tensor(sample, dtype=torch.float32)).numpy()
    np_logits = numpy_forward(sample, meta, f"{args.out}.bin")
    max_err = float(np.max(np.abs(torch_logits - np_logits)))
    print(f"export self-check max|torch - numpy| = {max_err:.3e}")
    assert max_err < 1e-5, "export mismatch: fix before writing the C++ loader"

    # Reference rows for the C++ match test: RAW (unscaled) features + torch logit.
    # x[te] is pre-standardization; C++ will standardize with the exported mean/std.
    raw_sample = x[te][:256]
    write_check(raw_sample, torch_logits, args.out)

    print(f"wrote {args.out}.meta  {args.out}.bin  {args.out}.json  "
          f"{args.out}.onnx  {args.out}.check")


if __name__ == "__main__":
    main()
