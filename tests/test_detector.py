"""The transcribed architecture must actually fit the project's checkpoints.

`load_state_dict(strict=True)` is the whole test: it fails on any parameter
that is missing, extra or the wrong shape. Anything looser would let a
transcription error through as scores that look plausible and are not.
"""
import pathlib

import pytest
import torch

from archive_pipeline.products.detector import (WaveformDetector,
                                                find_checkpoints, run_identity)

CKPTS = pathlib.Path("/home/oguzb/Projects/sismokaos/cnn_earthquake"
                     "/trained_model_branch1d_asinh")
needs_ckpts = pytest.mark.skipif(not CKPTS.is_dir(),
                                 reason="trained checkpoints not on this machine")


@needs_ckpts
@pytest.mark.parametrize("branch", ["cnn-lstm", "cnn", "lstm"])
def test_checkpoints_load_strictly(branch):
    found = find_checkpoints(CKPTS, "1d", "linear", branch)
    assert found, f"no {branch} checkpoints"
    model = WaveformDetector(3, hidden=48, fusion_dim=96, branch1d=branch)
    for path in found:
        model.load_state_dict(torch.load(path, weights_only=True,
                                         map_location="cpu"), strict=True)


@needs_ckpts
def test_unnarrowed_directory_refuses_to_guess():
    """Nine checkpoints spanning three branches must not silently ensemble."""
    with pytest.raises(ValueError, match="distinct runs"):
        find_checkpoints(CKPTS, "1d", "linear")


@needs_ckpts
def test_branch_match_is_anchored():
    """`cnn` is a prefix of `cnn-lstm`; the two must not overlap."""
    cnn = set(find_checkpoints(CKPTS, "1d", "linear", "cnn"))
    cnnlstm = set(find_checkpoints(CKPTS, "1d", "linear", "cnn-lstm"))
    assert cnn and cnnlstm and not (cnn & cnnlstm)


def test_forward_shape():
    m = WaveformDetector(3, hidden=48, fusion_dim=96, branch1d="cnn-lstm").eval()
    with torch.no_grad():
        assert m(torch.randn(4, 600, 3)).shape == (4, 1)


def test_run_identity_strips_seed_and_pid():
    assert (run_identity("best_x_1d_linear_cnn-lstm_ds_pid4758_seed42.pth")
            == "best_x_1d_linear_cnn-lstm_ds")
