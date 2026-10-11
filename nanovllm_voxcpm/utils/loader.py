import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open

ShardId = str | int

try:
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_state_dict_into(model: nn.Module, state_dict: dict[str, torch.Tensor], prefix: str = "") -> int:
    """Load an unpacked (q/k/v, gate/up) state dict into a packed nanovllm model; returns #tensors."""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    count = 0
    for weight_name, tensor in state_dict.items():
        if not weight_name.startswith(prefix):
            continue
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param = model.get_parameter(weight_name.replace(k, v))
                getattr(param, "weight_loader")(param, tensor, shard_id)
                break
        else:
            param = model.get_parameter(weight_name)
            getattr(param, "weight_loader", default_weight_loader)(param, tensor)
        count += 1
    return count


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    visited_param_names = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        visited_param_names.add(param_name)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    visited_param_names.add(weight_name)

    missing_param_names = []
    for name, _ in model.named_parameters():
        if name not in visited_param_names:
            # Skip LoRA parameters (they are optional)
            if "lora_" in name:
                continue
            missing_param_names.append(name)

    if missing_param_names:
        raise ValueError(f"Missing parameters: {missing_param_names}")
