from ._base import BaseGPTQForCausalLM


class DeepSeekV3GPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "DeepseekV3DecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        # DeepSeek-V2 usage, included in layer 0-59
        ["self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj"],

        ["self_attn.o_proj"],

        # included in layer 0
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],

        # included in layer 1-59, uses dynamic_expert_index
        ["mlp.experts.{expert_idx}.gate_proj", "mlp.experts.{expert_idx}.up_proj"],
        ["mlp.experts.{expert_idx}.down_proj"],

        # included in layer 1-59
        ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"],
        ["mlp.shared_experts.down_proj"],
    ]

    # ['moe.gate'],
    
"""
    layer_modules = [

    ]




        ["self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ['mlp.shared_experts.gate_proj', 'mlp.shared_experts.up_proj'],
        ['mlp.shared_experts.down_proj'], 
        ['mlp.experts.{expert_idx}.up_proj', 'mlp.experts.{expert_idx}.gate_proj'],
        ['mlp.experts.{expert_idx}.down_proj'],
        # [['mlp.gate_proj', 'mlp.up_proj'], ['moe.mlp.{expert_idx}.up_proj', 'moe.mlp.{expert_idx}.gate_proj']],
        # [['mlp.down_proj'], ['moe.mlp.{expert_idx}.down_proj']],
        # ['moe.gate']

"""


__all__ = ["DeepSeekV3GPTQForCausalLM"]