# ==============================================================
# fuzzynet_pipeline.py  –  v9
#  ▸ Inclui plot_comparative() dentro do próprio módulo
#  ▸ Mantém todas as correções de dtype, detach, etc.
# ==============================================================

from __future__ import annotations
import pickle, functools, hashlib, importlib.util, os, sys, warnings, pathlib
from typing import Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn
import matplotlib.pyplot as plt

# ---------- silenciar avisos de incompatibilidade do scikit-learn ----------
warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.InconsistentVersionWarning
)

# ---------- caminhos / device ---------------------------------------------
BASE_DIR  = pathlib.Path(__file__).resolve().parent
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_MODEL = "Rostlab/prot_t5_xl_bfd"

# ============================  MODELOS  ====================================
class ANFIS(nn.Module):
    def __init__(self, input_size: int, n_rules: int, n_out: int = 3,
                 dtype=torch.float32):
        super().__init__()
        self.c     = nn.Parameter(torch.randn(input_size, n_rules, dtype=dtype))
        self.sigma = nn.Parameter(torch.ones (input_size, n_rules, dtype=dtype))
        self.cons  = nn.Parameter(torch.randn(n_rules, input_size + 1, n_out,
                                              dtype=dtype))
        self.eps   = torch.tensor(1e-6, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.c.dtype)
        b = x.size(0)
        mu   = torch.exp(-((x.unsqueeze(2) - self.c) ** 2) /
                         (2 * self.sigma ** 2 + self.eps))
        alpha = torch.sum(torch.log(mu + self.eps), dim=1)
        w     = torch.softmax(alpha, dim=1)
        x1    = torch.cat([torch.ones(b, 1, device=x.device, dtype=x.dtype),
                           x], dim=1)
        f     = torch.einsum("bi,mio->bmo", x1, self.cons)
        return torch.sum(w.unsqueeze(2) * f, dim=1)


class DummyFRNN(nn.Module):
    def __init__(self, input_size: int, hidden: int = 128, n_out: int = 3,
                 dtype=torch.float32):
        super().__init__()
        self.rnn  = nn.GRU(input_size, hidden, batch_first=True, dtype=dtype)
        self.head = nn.Linear(hidden, n_out, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x.to(dtype=self.head.weight.dtype)
        _, h = self.rnn(x.unsqueeze(1))
        return self.head(h.squeeze(0))

# -- registra classes em __main__ p/ unpickle -------------------------------
sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "ANFIS", ANFIS)
setattr(sys.modules["__main__"], "DummyFRNN", DummyFRNN)

# ===================  EMBEDDINGS  ==========================================
def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

@functools.cache
def _load_protT5():
    if not (_has_module("transformers") and _has_module("sentencepiece")):
        return None, None
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    try:
        import transformers
        from transformers import AutoTokenizer, AutoModel
        transformers.logging.set_verbosity_error()
        tok   = AutoTokenizer.from_pretrained(EMB_MODEL,
                                              do_lower_case=False,
                                              use_fast=False)
        model = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE).eval()
        return tok, model
    except Exception:
        warnings.warn("[WARN] ProtT5 indisponível; usando embedding sintético.")
        return None, None

def _synthetic_embedding(seq: str) -> np.ndarray:
    h   = hashlib.sha1(seq.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:4], "little"))
    return rng.standard_normal(1024).astype(np.float32)

def get_embedding(seq: str) -> np.ndarray:
    tok, model = _load_protT5()
    if model is None:
        return _synthetic_embedding(seq)
    with torch.no_grad():
        batch = tok(seq, return_tensors="pt").to(DEVICE)
        out   = model(**batch)["last_hidden_state"].mean(1).squeeze()
    return out.cpu().numpy().astype(np.float32)

# =================  PCA + SCALER  ==========================================
def _load_pca_scaler() -> tuple[PCA, StandardScaler]:
    with open(BASE_DIR / "pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return pca, scaler

def reduce_and_scale(emb: np.ndarray) -> np.ndarray:
    pca, scaler = _load_pca_scaler()
    return scaler.transform(pca.transform(emb.reshape(1, -1)))\
                 .astype(np.float32).squeeze()

# =================  MODELO TREINADO  =======================================
def build_model(kind: Literal["ANFIS", "FRNN"] = "ANFIS"):
    path = BASE_DIR / f"{kind.lower()}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    model.to(DEVICE)
    return model

# =================  PREDIÇÃO  ==============================================
def predict_interaction(seq1: str, seq2: str, kind: str = "ANFIS"):
    v1 = reduce_and_scale(get_embedding(seq1))
    v2 = reduce_and_scale(get_embedding(seq2))
    x  = torch.tensor(np.concatenate([v1, v2]), device=DEVICE,
                      dtype=torch.float32).unsqueeze(0)

    model = build_model(kind)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).detach().cpu().numpy().squeeze()

    return int(probs.argmax()), probs

# =================  VISUALIZAÇÃO PADRÃO  ===================================
def plot_memberships(model: ANFIS, top_k: int = 3,
                     save_to: os.PathLike | None = None):
    c   = model.c.detach().cpu().numpy()
    s   = model.sigma.detach().cpu().numpy()
    imp = np.abs(model.cons.detach().cpu().numpy()).sum((0, 2))
    top = imp.argsort()[-top_k:][::-1]

    x_range = np.linspace(-5, 5, 400)
    fig, axes = plt.subplots(top_k, 1, figsize=(6, 2 * top_k))
    axes = np.atleast_1d(axes)

    for ax, idx in zip(axes, top):
        for m in range(c.shape[1]):
            mu = np.exp(-((x_range - c[idx, m]) ** 2) /
                         (2 * s[idx, m] ** 2 + 1e-6))
            ax.plot(x_range, mu, alpha=.7)
        ax.set_title(f"Var {idx}")
    plt.tight_layout()

    if save_to:
        pathlib.Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig

# =================  VISUALIZAÇÃO COMPARATIVA ===============================
def plot_comparative(
        model: ANFIS,
        x_sample: torch.Tensor,
        var_indices: list[int],
        rule_indices: list[int] | None = None,
        save_to: str | None = None
    ):
    """
    Plota apenas as regras mais fortes (por padrão 2) para cada variável.
    """
    model.eval()
    c  = model.c.detach().cpu().numpy()     # (D, M)
    s  = model.sigma.detach().cpu().numpy()
    M  = c.shape[1]
    x_np = x_sample.detach().cpu().numpy()

    mu_mat = np.exp(-((x_np[:, None] - c) ** 2) / (2 * s ** 2 + 1e-6))
    alpha  = np.sum(np.log(mu_mat + 1e-6), axis=0)
    if rule_indices is None:
        rule_indices = alpha.argsort()[::-1][:2]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(1, len(var_indices),
                             figsize=(5 * len(var_indices), 3),
                             sharey=True)
    if len(var_indices) == 1:
        axes = [axes]

    for ax, vid in zip(axes, var_indices):
        x = np.linspace(-4, 4, 400)
        for j, r in enumerate(rule_indices):
            mu_curve = np.exp(-((x - c[vid, r]) ** 2) /
                              (2 * s[vid, r] ** 2 + 1e-6))
            ax.plot(x, mu_curve, label=f"Regra {r + 1}",
                    color=colors[j % len(colors)], linewidth=2)
            mu_x0 = np.exp(-((0 - c[vid, r]) ** 2) /
                           (2 * s[vid, r] ** 2 + 1e-6))
            ax.scatter(0, mu_x0, color=colors[j % len(colors)],
                       s=40, zorder=5)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_title(f"Variável {vid}")
        ax.set_xlabel("Valor"); ax.set_ylabel("μ")
        ax.legend(loc="upper right")

    plt.tight_layout()
    if save_to:
        pathlib.Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig
