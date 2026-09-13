"""The trained detector, for inference only.

Scoring a continuous record needs the architecture the checkpoints were
trained with, and nothing else: no training loop, no dataset classes, no 2D
spectrogram branch. Every continuous detector in this project is a
`--channels 1d` model, whose state dict holds 42 tensors covering one waveform
branch, one projection, two fusion scalars and a head.

So this is a transcription of that one configuration rather than a dependency
on the training package, which carries six architectures and their training
machinery. The transcription is not trusted on inspection: `tests/test_detector.py`
loads the project's real checkpoints with `strict=True`, which fails on any
missing, extra or wrongly shaped parameter.

**The fusion scalars are kept even though there is nothing to fuse.** A
single-branch model still multiplies by `w1`, the optimizer settled it
somewhere near but not at 1.0, and dropping it would shift every logit.
"""
import re
from pathlib import Path

import torch
import torch.nn as nn

# `{channels}_{fusion}_{branch_1d}_{seq_transform}_{dataset}_pid{pid}_seed{n}`.
# Stripping the pid/seed tail leaves the run identity: everything that must
# agree before two checkpoints may be averaged together.
RUN_SUFFIX = re.compile(r"_pid\d+_seed\d+\.pth$")


class ConvSeqBranch(nn.Module):
    """Strided 1D convolutions, then BiLSTM and self-attention.

    The convolutions reduce the sequence about 8x before the recurrent layer,
    which also makes the attention that follows ~64x cheaper. `use_lstm=False`
    stops after the convolutions and mean-pools.
    """

    def __init__(self, in_dim, hidden=64, layers=1, heads=4, dropout=0.2,
                 use_lstm=True, conv_width=96):
        super().__init__()
        self.use_lstm = use_lstm

        def stage(cin, cout, k, s):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=k, stride=s, padding=k // 2,
                          bias=False),
                nn.BatchNorm1d(cout), nn.GELU())

        self.conv = nn.Sequential(
            stage(in_dim, conv_width // 4, 7, 2),
            stage(conv_width // 4, conv_width // 2, 5, 2),
            nn.Dropout(dropout),
            stage(conv_width // 2, conv_width, 5, 2),
        )
        if use_lstm:
            self.lstm = nn.LSTM(conv_width, hidden, num_layers=layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if layers > 1 else 0.0)
            d = hidden * 2
            self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                              batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.out_dim = d
        else:
            self.out_dim = conv_width

    def forward(self, x):
        """Encodes `(batch, time, in_dim)` into `(batch, out_dim)`."""
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        if not self.use_lstm:
            return h.mean(dim=1)
        h, _ = self.lstm(h)
        a, _ = self.attn(h, h, h)
        return self.norm(h + a).mean(dim=1)


class LSTMAttentionBranch(nn.Module):
    """BiLSTM over raw samples, then self-attention. The pre-convolution design."""

    def __init__(self, in_dim, hidden=64, layers=1, heads=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        d = hidden * 2
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, x):
        """Encodes `(batch, time, in_dim)` into `(batch, out_dim)`."""
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h)
        return self.norm(h + a).mean(dim=1)


class WaveformDetector(nn.Module):
    """One waveform branch and a binary head: `--channels 1d`, `--fusion linear`.

    Parameter names match the training model's exactly -- `b1`, `p1`, `w1`,
    `w2`, `head` -- because the checkpoints are keyed on them.
    """

    def __init__(self, seq_dim=3, hidden=48, fusion_dim=96, dropout=0.4,
                 branch1d="cnn-lstm", lstm_layers=1, lstm_heads=4):
        """Builds the network.

        Args:
            seq_dim: Channels per timestep; 3 for Z/N/E.
            hidden: LSTM hidden size per direction.
            fusion_dim: Width the branch is projected to, and the head's width.
            dropout: Inactive at eval, but it sizes nothing, so it is carried
                only so a model built here can also be trained elsewhere.
            branch1d: `cnn-lstm`, `cnn` or `lstm`.
            lstm_layers: Stacked LSTM layers.
            lstm_heads: Attention heads; must divide `hidden * 2`.

        Raises:
            ValueError: On an unknown `branch1d`.
        """
        super().__init__()
        if branch1d == "lstm":
            self.b1 = LSTMAttentionBranch(seq_dim, hidden=hidden,
                                          layers=lstm_layers, heads=lstm_heads,
                                          dropout=dropout)
        elif branch1d in ("cnn", "cnn-lstm"):
            self.b1 = ConvSeqBranch(seq_dim, hidden=hidden, layers=lstm_layers,
                                    heads=lstm_heads, dropout=dropout,
                                    use_lstm=(branch1d == "cnn-lstm"))
        else:
            raise ValueError(
                f"branch1d must be 'lstm', 'cnn' or 'cnn-lstm', got {branch1d!r}")
        self.p1 = nn.Linear(self.b1.out_dim, fusion_dim)
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, seq):
        """Raw logit per window, shape `(batch, 1)`."""
        return self.head(self.w1 * self.p1(self.b1(seq)))


def run_identity(name):
    """The part of a checkpoint filename shared by every seed of one run."""
    return RUN_SUFFIX.sub("", Path(name).name)


def find_checkpoints(ckpt_dir, channels="1d", fusion="linear", branch=None,
                     require_single_run=True):
    """The checkpoints of one training arm, or a refusal to guess.

    A checkpoint directory rarely holds one arm: `trained_model_branch1d_asinh`
    has nine, spanning `lstm`, `cnn` and `cnn-lstm`. Averaging across them is
    not an ensemble but a mixture of models answering different questions, and
    a bare glob does exactly that silently.

    Anchoring is load-bearing: `cnn` is a prefix of `cnn-lstm`, so the trailing
    underscore is what keeps the two apart. Matching alone is still not enough,
    because the run tag grew over time and older checkpoints lack segments a
    strict pattern would demand -- so the match stays loose and the *result* is
    checked instead.

    Args:
        ckpt_dir: Directory of `.pth` checkpoints.
        channels: `1d`, `2d` or `all`.
        fusion: `linear` or `gate`.
        branch: `branch1d` value to require; None matches on channels and
            fusion alone, which is what pre-`--branch-1d` checkpoints need.
        require_single_run: Raise when survivors span several runs.

    Returns:
        Sorted list of `Path`s.

    Raises:
        FileNotFoundError: Nothing matched.
        ValueError: The match spans several runs and `require_single_run`.
    """
    pat = re.compile(rf"_{re.escape(channels)}_{re.escape(fusion)}_"
                     + (rf"{re.escape(branch)}_" if branch else ""))
    found = sorted(p for p in Path(ckpt_dir).glob("*.pth") if pat.search(p.name))
    if not found:
        raise FileNotFoundError(
            f"no checkpoints matching channels={channels} fusion={fusion}"
            + (f" branch={branch}" if branch else "") + f" under {ckpt_dir}")
    runs = sorted({run_identity(p.name) for p in found})
    if require_single_run and len(runs) > 1:
        raise ValueError(
            f"{ckpt_dir} holds {len(runs)} distinct runs matching "
            f"channels={channels} fusion={fusion}"
            + (f" branch={branch}" if branch else "")
            + ". Ensembling across them would mix architectures, datasets or "
              "amplitude transforms. Narrow it with a branch, or point at a "
              "directory holding one run:\n  " + "\n  ".join(runs))
    return found


def load_ensemble(ckpt_dir, branch, device, channels="1d", fusion="linear",
                  hidden=48, fusion_dim=96):
    """Every seed of one training arm, in eval mode.

    Loaded with `strict=True`: a checkpoint that does not fit this architecture
    exactly must fail here, not produce plausible-looking scores.

    Returns:
        List of `WaveformDetector`, one per seed.
    """
    models = []
    for path in find_checkpoints(ckpt_dir, channels, fusion, branch):
        m = WaveformDetector(3, hidden=hidden, fusion_dim=fusion_dim,
                             branch1d=branch).to(device)
        m.load_state_dict(torch.load(path, weights_only=True,
                                     map_location=device), strict=True)
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def score_block(models, seq, device):
    """Probability-averaged ensemble over an `(n, win, 3)` block.

    `asinh` is the amplitude transform the arms in use were trained under: a
    signed log compression, monotonic so amplitude ordering survives.
    """
    x = torch.asinh(torch.from_numpy(seq).float().to(device))
    acc = None
    for m in models:
        p = torch.sigmoid(m(x)).squeeze(1)
        acc = p if acc is None else acc + p
    return (acc / len(models)).cpu().numpy()
