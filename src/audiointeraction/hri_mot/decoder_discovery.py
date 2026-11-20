from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch.nn as nn


ATTENTION_NAMES = ("self_attn", "attn", "attention")
MLP_NAMES = ("mlp", "feed_forward", "ffn")
INPUT_NORM_NAMES = ("input_layernorm", "norm_1", "ln_1", "self_attn_layer_norm")
POST_ATTN_NORM_NAMES = ("post_attention_layernorm", "norm_2", "ln_2", "final_layernorm")
FINAL_NORM_NAMES = ("ln_f", "norm", "final_norm", "final_layernorm")
LM_HEAD_NAMES = ("lm_head", "embed_out", "output", "output_projection")


@dataclass
class DecoderCandidate:
    path: str
    module: nn.Module
    num_layers: int
    layer_class_name: str
    has_q_proj: bool
    has_k_proj: bool
    has_v_proj: bool
    has_o_proj: bool
    has_fused_qkv: bool
    has_output_proj_alias: bool
    hidden_size: Optional[int]
    num_attention_heads: Optional[int]
    num_key_value_heads: Optional[int]
    head_dim: Optional[int]
    final_norm_path: Optional[str]
    lm_head_path: Optional[str]
    score: int


def get_module_by_path(root: nn.Module, path: str) -> nn.Module:
    if not path:
        return root
    try:
        return root.get_submodule(path)
    except AttributeError:
        current = root
        for part in path.split("."):
            current = getattr(current, part)
        return current


def _has_submodule(module: nn.Module, path: str) -> bool:
    try:
        module.get_submodule(path)
        return True
    except AttributeError:
        return False


def _get_optional_submodule(module: nn.Module, path: str) -> Optional[nn.Module]:
    try:
        return module.get_submodule(path)
    except AttributeError:
        return None


def _find_child(module: nn.Module, names: Iterable[str]) -> Optional[nn.Module]:
    for name in names:
        child = getattr(module, name, None)
        if isinstance(child, nn.Module):
            return child
    return None


def find_attention_module(layer: nn.Module) -> Optional[nn.Module]:
    attention = _find_child(layer, ATTENTION_NAMES)
    if attention is not None:
        return attention
    if any(hasattr(layer, name) for name in ("qkv", "c_attn", "q_proj", "k_proj", "v_proj")):
        return layer
    return None


def find_mlp_module(layer: nn.Module) -> Optional[nn.Module]:
    return _find_child(layer, MLP_NAMES)


def _projection(module: nn.Module, name: str) -> Optional[nn.Module]:
    child = getattr(module, name, None)
    return child if isinstance(child, nn.Module) else None


def _config_attr(module: nn.Module, names: Iterable[str]) -> Optional[int]:
    config = getattr(module, "config", None)
    for owner in (module, config):
        if owner is None:
            continue
        for name in names:
            value = getattr(owner, name, None)
            if value is not None:
                return int(value)
    return None


def _linear_in_features(module: Optional[nn.Module]) -> Optional[int]:
    if module is not None and hasattr(module, "in_features"):
        return int(module.in_features)
    return None


def _linear_out_features(module: Optional[nn.Module]) -> Optional[int]:
    if module is not None and hasattr(module, "out_features"):
        return int(module.out_features)
    return None


def infer_hidden_size(layer: nn.Module, attention: nn.Module) -> Optional[int]:
    value = _config_attr(attention, ("n_embd", "hidden_size", "d_model"))
    if value is not None:
        return value
    value = _config_attr(layer, ("n_embd", "hidden_size", "d_model"))
    if value is not None:
        return value
    for proj_name in ("qkv", "q_proj", "query", "c_attn"):
        value = _linear_in_features(_projection(attention, proj_name))
        if value is not None:
            return value
    return None


def infer_num_heads(layer: nn.Module, attention: nn.Module) -> Optional[int]:
    value = _config_attr(attention, ("n_head", "num_attention_heads", "num_heads"))
    if value is not None:
        return value
    return _config_attr(layer, ("n_head", "num_attention_heads", "num_heads"))


def infer_num_kv_heads(layer: nn.Module, attention: nn.Module) -> Optional[int]:
    value = _config_attr(attention, ("n_query_groups", "num_key_value_heads", "num_kv_heads"))
    if value is not None:
        return value
    return _config_attr(layer, ("n_query_groups", "num_key_value_heads", "num_kv_heads"))


def infer_head_dim(layer: nn.Module, attention: nn.Module) -> Optional[int]:
    value = _config_attr(attention, ("head_size", "head_dim"))
    if value is not None:
        return value
    value = _config_attr(layer, ("head_size", "head_dim"))
    if value is not None:
        return value
    q_proj = _projection(attention, "q_proj")
    heads = infer_num_heads(layer, attention)
    q_out = _linear_out_features(q_proj)
    if heads and q_out:
        return q_out // heads
    return None


def _parent_path(path: str) -> str:
    if "." not in path:
        return ""
    return path.rsplit(".", 1)[0]


def _join_path(parent: str, child: str) -> str:
    return child if not parent else f"{parent}.{child}"


def _child_name(module: nn.Module, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if isinstance(getattr(module, name, None), nn.Module):
            return name
    return None


def _module_line(path: str, module: nn.Module) -> str:
    details = []
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        details.append(f"in={module.in_features}")
        details.append(f"out={module.out_features}")
        details.append(f"bias={getattr(module, 'bias', None) is not None}")
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        details.append(f"len={len(module)}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{path}: {module.__class__.__name__}{suffix}"


def find_likely_final_norm_path(model: nn.Module, decoder_path: str) -> Optional[str]:
    parents = [_parent_path(decoder_path), ""]
    for parent_path in parents:
        parent = get_module_by_path(model, parent_path) if parent_path else model
        for name in FINAL_NORM_NAMES:
            if hasattr(parent, name):
                child = getattr(parent, name)
                if isinstance(child, nn.Module):
                    return _join_path(parent_path, name)
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in FINAL_NORM_NAMES and not name.startswith(decoder_path):
            return name
    return None


def find_likely_lm_head_path(model: nn.Module) -> Optional[str]:
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in LM_HEAD_NAMES and isinstance(module, nn.Module):
            return name
    return None


def _layer_like(layer: nn.Module) -> bool:
    return find_attention_module(layer) is not None and find_mlp_module(layer) is not None


def find_decoder_candidates(model: nn.Module, min_layers: int = 2) -> List[DecoderCandidate]:
    candidates: List[DecoderCandidate] = []
    for path, module in model.named_modules():
        if not isinstance(module, (nn.ModuleList, nn.Sequential)):
            continue
        layers = list(module.children())
        if len(layers) < min_layers:
            continue
        first_layer = layers[0]
        if not _layer_like(first_layer):
            continue

        attention = find_attention_module(first_layer)
        if attention is None:
            continue

        has_q_proj = hasattr(attention, "q_proj")
        has_k_proj = hasattr(attention, "k_proj")
        has_v_proj = hasattr(attention, "v_proj")
        has_o_proj = hasattr(attention, "o_proj")
        has_fused_qkv = hasattr(attention, "qkv") or hasattr(attention, "c_attn")
        has_output_proj_alias = hasattr(attention, "proj") or hasattr(attention, "c_proj")
        final_norm_path = find_likely_final_norm_path(model, path)
        lm_head_path = find_likely_lm_head_path(model)

        score = len(layers) * 10
        lowered_path = path.lower()
        if any(token in lowered_path for token in ("decoder", "language", "llm", "transformer", "model")):
            score += 20
        if any(token in lowered_path for token in ("layers", "h")):
            score += 10
        if final_norm_path is not None:
            score += 5
        if lm_head_path is not None:
            score += 5
        if has_fused_qkv or (has_q_proj and has_k_proj and has_v_proj):
            score += 10

        candidates.append(
            DecoderCandidate(
                path=path,
                module=module,
                num_layers=len(layers),
                layer_class_name=first_layer.__class__.__name__,
                has_q_proj=has_q_proj,
                has_k_proj=has_k_proj,
                has_v_proj=has_v_proj,
                has_o_proj=has_o_proj,
                has_fused_qkv=has_fused_qkv,
                has_output_proj_alias=has_output_proj_alias,
                hidden_size=infer_hidden_size(first_layer, attention),
                num_attention_heads=infer_num_heads(first_layer, attention),
                num_key_value_heads=infer_num_kv_heads(first_layer, attention),
                head_dim=infer_head_dim(first_layer, attention),
                final_norm_path=final_norm_path,
                lm_head_path=lm_head_path,
                score=score,
            )
        )
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def select_decoder_candidate(model: nn.Module, decoder_path: Optional[str] = None) -> DecoderCandidate:
    if decoder_path is not None:
        module = get_module_by_path(model, decoder_path)
        layers = list(module.children())
        if not layers:
            raise ValueError(f"Decoder path {decoder_path!r} has no layers")
        attention = find_attention_module(layers[0])
        if attention is None:
            raise ValueError(f"Decoder path {decoder_path!r} does not look like an autoregressive decoder")
        candidates = [
            item for item in find_decoder_candidates(model, min_layers=1) if item.path == decoder_path
        ]
        if candidates:
            return candidates[0]
        return DecoderCandidate(
            path=decoder_path,
            module=module,
            num_layers=len(layers),
            layer_class_name=layers[0].__class__.__name__,
            has_q_proj=hasattr(attention, "q_proj"),
            has_k_proj=hasattr(attention, "k_proj"),
            has_v_proj=hasattr(attention, "v_proj"),
            has_o_proj=hasattr(attention, "o_proj"),
            has_fused_qkv=hasattr(attention, "qkv") or hasattr(attention, "c_attn"),
            has_output_proj_alias=hasattr(attention, "proj") or hasattr(attention, "c_proj"),
            hidden_size=infer_hidden_size(layers[0], attention),
            num_attention_heads=infer_num_heads(layers[0], attention),
            num_key_value_heads=infer_num_kv_heads(layers[0], attention),
            head_dim=infer_head_dim(layers[0], attention),
            final_norm_path=find_likely_final_norm_path(model, decoder_path),
            lm_head_path=find_likely_lm_head_path(model),
            score=0,
        )

    candidates = find_decoder_candidates(model)
    if not candidates:
        raise ValueError("Could not find a decoder stack candidate in the base model")
    return candidates[0]


def format_candidate(candidate: DecoderCandidate, selected: bool = False) -> str:
    prefix = "* " if selected else "- "
    return "\n".join(
        [
            f"{prefix}module path: {candidate.path}",
            f"  layers: {candidate.num_layers}",
            f"  layer class: {candidate.layer_class_name}",
            (
                "  projections: "
                f"q_proj={candidate.has_q_proj}, "
                f"k_proj={candidate.has_k_proj}, "
                f"v_proj={candidate.has_v_proj}, "
                f"o_proj={candidate.has_o_proj}, "
                f"fused_qkv={candidate.has_fused_qkv}, "
                f"output_proj_alias={candidate.has_output_proj_alias}"
            ),
            f"  hidden size: {candidate.hidden_size}",
            f"  attention heads: {candidate.num_attention_heads}",
            f"  kv heads: {candidate.num_key_value_heads}",
            f"  head dim: {candidate.head_dim}",
            f"  likely final norm path: {candidate.final_norm_path}",
            f"  likely LM head path: {candidate.lm_head_path}",
            f"  score: {candidate.score}",
        ]
    )


def print_model_tree(module: nn.Module, max_depth: int = 3, _prefix: str = "") -> None:
    if max_depth < 0:
        return
    for name, child in module.named_children():
        path = f"{_prefix}.{name}" if _prefix else name
        extra = ""
        if isinstance(child, (nn.ModuleList, nn.Sequential)):
            extra = f" len={len(child)}"
        print(f"{path}: {child.__class__.__name__}{extra}")
        print_model_tree(child, max_depth=max_depth - 1, _prefix=path)


def describe_decoder_block(block: nn.Module, block_path: str = "") -> str:
    lines = [
        f"block path: {block_path or '<root>'}",
        f"block class: {block.__class__.__name__}",
        "submodule tree:",
    ]
    for name, child in block.named_modules():
        if not name:
            continue
        lines.append(f"  {_module_line(_join_path(block_path, name), child)}")

    attention_name = _child_name(block, ATTENTION_NAMES)
    attention = getattr(block, attention_name) if attention_name else find_attention_module(block)
    attention_path = _join_path(block_path, attention_name) if attention_name else block_path

    input_norm_name = _child_name(block, INPUT_NORM_NAMES)
    post_attention_output_norm = _child_name(block, ("post_attention_norm",))
    post_attention_norm_name = _child_name(block, POST_ATTN_NORM_NAMES)
    mlp_name = _child_name(block, MLP_NAMES)

    fused_qkv_name = None
    output_projection_name = None
    if attention is not None:
        fused_qkv_name = _child_name(attention, ("qkv", "c_attn"))
        output_projection_name = _child_name(attention, ("o_proj", "proj", "c_proj", "dense"))

    config = getattr(block, "config", None)
    attn_config = getattr(attention, "config", None) if attention is not None else None

    lines.extend(
        [
            "identified names:",
            f"  input norm: {_join_path(block_path, input_norm_name) if input_norm_name else None}",
            f"  attention module: {attention_path if attention is not None else None}",
            (
                "  fused qkv linear: "
                f"{_join_path(attention_path, fused_qkv_name) if fused_qkv_name else None}"
            ),
            (
                "  attention output projection: "
                f"{_join_path(attention_path, output_projection_name) if output_projection_name else None}"
            ),
            (
                "  attention output norm slot: "
                f"{_join_path(block_path, post_attention_output_norm) if post_attention_output_norm else None}"
            ),
            (
                "  post-attention / MLP input norm: "
                f"{_join_path(block_path, post_attention_norm_name) if post_attention_norm_name else None}"
            ),
            f"  MLP module: {_join_path(block_path, mlp_name) if mlp_name else None}",
            "relevant attributes:",
            f"  parallel_residual: {getattr(config, 'parallel_residual', None)}",
            f"  shared_attention_norm: {getattr(config, 'shared_attention_norm', None)}",
            f"  n_head: {getattr(attn_config, 'n_head', None)}",
            f"  n_query_groups: {getattr(attn_config, 'n_query_groups', None)}",
            f"  head_size: {getattr(attn_config, 'head_size', None)}",
            f"  rope_n_elem: {getattr(attn_config, 'rope_n_elem', None)}",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DecoderCandidate",
    "describe_decoder_block",
    "find_attention_module",
    "find_decoder_candidates",
    "find_likely_final_norm_path",
    "find_likely_lm_head_path",
    "find_mlp_module",
    "format_candidate",
    "get_module_by_path",
    "print_model_tree",
    "select_decoder_candidate",
]
