from memory_estimator.model_shapes import _is_expert_tensor


def test_expert_tensor_standard_names():
    """Standard MoE expert tensor names should match."""
    assert _is_expert_tensor("model.layers.0.block_sparse_moe.experts.0.w1.weight")
    assert _is_expert_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
    assert _is_expert_tensor("model.layers.0.mlp.experts.3.down_proj.weight")


def test_non_expert_tensors():
    """Non-expert tensors should not match, even with 'expert' substring."""
    assert not _is_expert_tensor("model.layers.0.mlp.gate.weight")
    assert not _is_expert_tensor("model.layers.0.self_attn.q_proj.weight")
    assert not _is_expert_tensor("model.embed_tokens.weight")
    # Contains "expert" as substring but not as ".experts." path component
    assert not _is_expert_tensor("model.layers.0.expert_gate.weight")
    assert not _is_expert_tensor("model.num_experts_per_tok")
