import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputHook:
    def __init__(self, name):
        self.saved_outs = []
        self.name = name

    def __call__(self, module, input, output):
        self.saved_outs.append(output[0].detach().cpu())


def add_hooks(model):
    hooks = dict()
    handles = []
    for layer_id, layer in enumerate(model.decoder.layers):
        base_name = f"decoder.l{layer_id}"
        for proj_name in ["k_proj", "q_proj"]:
            name = ".".join([base_name, proj_name])
            hook = OutputHook(name)
            hooks[name] = hook
            handles.append(
                getattr(layer.encoder_attn, proj_name).register_forward_hook(hook)
            )

    return hooks, handles

def remove_hooks(hook_handles):
    for handle in hook_handles:
        handle.remove()

def make_heads(tensor, d_k):
    chunks = torch.split(tensor.unsqueeze(0), d_k, dim=-1)
    return torch.cat(chunks, dim=0)

def get_attn_scores(sentence, model, model_name, src_lang, trg_lang, beam_size=1, skip_softmax=False):
    """
    @return dict of attentions. keys are decoder.l{layer_num}. 
    the shape of each tensor in dict is (1, num_heads, num_tokens_tgt, num_tokens_src)
    """
    translator_model = model.translator.models[model_name]["model"].model

    hooks, handles = add_hooks(translator_model)
    model.translate([sentence], source_lang=src_lang, target_lang=trg_lang, beam_size=beam_size)

    attn_scores = dict()
    for layer_id, layer in enumerate(translator_model.decoder.layers):
        base_name = f"decoder.l{layer_id}"
        d_k = layer.encoder_attn.embed_dim // layer.encoder_attn.num_heads

        all_keys = make_heads(torch.cat(hooks[f"{base_name}.k_proj"].saved_outs), d_k)
        all_queries = make_heads(torch.cat(
            [t.unsqueeze(1) for t in hooks[f"{base_name}.q_proj"].saved_outs], dim=1
        ), d_k).transpose(0, 1)

        attn = torch.matmul(all_queries, all_keys.transpose(-1, -2)) / np.sqrt(d_k)
        
        if not skip_softmax:
            attn_scores[base_name] = F.softmax(attn, dim=-1)

    remove_hooks(handles)
    return attn_scores
