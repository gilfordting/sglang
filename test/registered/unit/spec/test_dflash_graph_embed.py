"""SGLANG_DFLASH_GRAPH_EMBED: the DFLASH draft embeds its block in-graph.

Contract exercised by `DecodeCudaGraphRunner` (capture stages eager `input_embeds`
only when the draft model has no `forward_embed`) and by
`DFlashWorkerV2.forward_batch_generation` (skips the eager target-embedding lookup,
and its un-graphed TP all-reduce, when the draft owns the embed).
"""

import torch
import torch.nn as nn

from sglang.srt.models.dflash import DFlashDraftModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _bare_draft_model() -> DFlashDraftModel:
    # The real constructor needs a draft config + quant setup; the attach contract
    # only touches instance attributes, so a bare nn.Module-initialised instance is enough.
    model = DFlashDraftModel.__new__(DFlashDraftModel)
    nn.Module.__init__(model)
    return model


def test_default_draft_has_no_forward_embed():
    model = _bare_draft_model()
    assert not hasattr(model, "forward_embed")
    assert hasattr(model, "attach_target_embedding")


def test_attach_target_embedding_exposes_forward_embed_using_target_table():
    torch.manual_seed(0)
    embed = nn.Embedding(64, 8)
    model = _bare_draft_model()
    model.attach_target_embedding(embed)

    assert hasattr(model, "forward_embed")
    ids = torch.tensor([[3, 7, 7, 7], [11, 7, 7, 7]])
    out = model.forward_embed(ids)
    torch.testing.assert_close(out, embed(ids))
    assert out.shape == (2, 4, 8)


def test_attached_embedding_is_not_registered_as_draft_submodule():
    embed = nn.Embedding(64, 8)
    model = _bare_draft_model()
    model.attach_target_embedding(embed)
    # Shared with the target: must not show up in the draft's own module tree or
    # parameters (weight loading / memory accounting / state_dict unchanged).
    assert all(m is not embed for _, m in model.named_modules())
    assert len(list(model.parameters())) == 0
    assert "embed_tokens" not in dict(model.named_children())


def test_runner_capture_condition_flips_with_attach():
    # Mirrors decode_cuda_graph_runner: stage eager input_embeds iff no forward_embed.
    model = _bare_draft_model()
    stage_eager = not hasattr(model, "forward_embed")
    assert stage_eager
    model.attach_target_embedding(nn.Embedding(4, 2))
    stage_eager = not hasattr(model, "forward_embed")
    assert not stage_eager
