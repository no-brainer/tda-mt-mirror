import numpy as np
import torch
import torch.nn as nn


class OutputHook:
    def __init__(self, name):
        self.saved_outs = []
        self.name = name

    def __call__(self, module, module_input, output):
        self.saved_outs.append(output[1].detach().cpu())


class MultiheadAttentionWrapper(nn.Module):

    def __init__(self, attn_module):
        super().__init__()
        self.attn_module = attn_module

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        attn_output, attn_output_weights = self.attn_module(
            query, key, value, key_padding_mask=key_padding_mask, need_weights=True,
            attn_mask=attn_mask, average_attn_weights=False
        )

        return attn_output, attn_output_weights.mean(dim=1)


def prepare_for_attn_extraction(model):
    for layer in model.transformer.decoder.layers:
        layer.multihead_attn = MultiheadAttentionWrapper(layer.multihead_attn)


def add_hooks(model):
    hooks = dict()
    handles = []
    for layer_id, layer in enumerate(model.transformer.decoder.layers):
        # add hook
        name = f"attn_weights.l{layer_id}"
        hook = OutputHook(name)
        hooks[name] = hook

        # we are actually interested in output of inner attn module
        handles.append(layer.multihead_attn.attn_module.register_forward_hook(hook))

    return hooks, handles


def remove_hooks(hook_handles):
    for handle in hook_handles:
        handle.remove()


def get_attn_scores(sentence, translator, keep_last_pass_only=True):
    """
    the shape of each tensor in dict is (num_layers, num_heads, num_tokens_tgt, num_tokens_src)
    """
    hooks, handles = add_hooks(translator.model)
    try:
        translator.translate(sentence)

        num_passes = len(list(hooks.values())[0].saved_outs)
        attn_scores = [[] for _ in range(num_passes)]
        for hook_name, hook in hooks.items():
            for i, attn_map in enumerate(hook.saved_outs):
                attn_scores[i].append(attn_map)

    finally:
        remove_hooks(handles)

    for i in range(num_passes):
        attn_scores[i] = torch.cat(attn_scores[i], dim=0).numpy()

    if keep_last_pass_only:
        return attn_scores[-1]
    return attn_scores
