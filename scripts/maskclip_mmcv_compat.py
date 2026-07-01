from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import torch

import mmcv
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import load_checkpoint


class DictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        out = {}
        for item in values or []:
            key, value = item.split("=", 1)
            try:
                out[key] = json.loads(value)
            except Exception:
                out[key] = value
        setattr(namespace, self.dest, out)


def digit_version(version_str: str):
    import re

    parts = []
    for x in re.split(r"[.+-]", str(version_str)):
        if x.isdigit():
            parts.append(int(x))
        else:
            break
    return tuple(parts)


def is_str(x: Any) -> bool:
    return isinstance(x, str)


def is_list_of(seq: Any, expected_type: type, seq_type: type | None = None) -> bool:
    if seq_type is None:
        exp_seq_type = list
    else:
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    return all(isinstance(item, expected_type) for item in seq)


def mkdir_or_exist(dir_name: str | os.PathLike) -> None:
    Path(dir_name).mkdir(parents=True, exist_ok=True)


def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def dump(obj: Any, file: str | os.PathLike, **kwargs) -> None:
    file = str(file)
    mkdir_or_exist(Path(file).parent)
    if file.endswith((".pkl", ".pickle")):
        with open(file, "wb") as f:
            pickle.dump(obj, f)
    else:
        indent = kwargs.get("indent", None)
        with open(file, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, default=_json_default)


def load(file: str | os.PathLike) -> Any:
    file = str(file)
    if file.endswith((".pkl", ".pickle")):
        with open(file, "rb") as f:
            return pickle.load(f)
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def print_log(msg: str, logger: Any = None, level: Any = None) -> None:
    print(msg)


class FileClient:
    def __init__(self, backend="disk", **kwargs):
        if backend != "disk":
            raise NotImplementedError(f"Only disk FileClient is supported, got {backend}")
    def get(self, filepath):
        with open(filepath, "rb") as f:
            return f.read()


def scandir(dir_path, suffix=None, recursive=False):
    dir_path = Path(dir_path)
    pattern = "**/*" if recursive else "*"
    for path in dir_path.glob(pattern):
        if not path.is_file():
            continue
        rel = str(path.relative_to(dir_path))
        if suffix is None or rel.endswith(suffix):
            yield rel


class Registry:
    def __init__(self, name: str, parent: "Registry | None" = None, **kwargs):
        self.name = name
        self.parent = parent
        self.module_dict: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        if key in self.module_dict:
            return self.module_dict[key]
        if self.parent is not None and hasattr(self.parent, "get"):
            return self.parent.get(key)
        return None

    def register_module(self, module=None, name: str | None = None, force: bool = False, **kwargs):
        def _register(cls):
            module_name = name or cls.__name__
            if not force and module_name in self.module_dict:
                raise KeyError(f"{module_name} is already registered in {self.name}")
            self.module_dict[module_name] = cls
            return cls

        if module is None:
            return _register
        return _register(module)

    def build(self, cfg, default_args: dict | None = None):
        return build_from_cfg(cfg, self, default_args)


def build_from_cfg(cfg, registry: Registry, default_args: dict | None = None):
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)}")
    args = dict(cfg)
    obj_type = args.pop("type")
    if default_args:
        for k, v in default_args.items():
            args.setdefault(k, v)
    if isinstance(obj_type, str):
        cls = registry.get(obj_type)
        if cls is None:
            raise KeyError(f"{obj_type} is not registered in {registry.name}")
    else:
        cls = obj_type
    return cls(**args)


class DataContainer:
    def __init__(self, data, stack: bool = False, padding_value: int = 0, cpu_only: bool = False, pad_dims: int | None = 2):
        self.data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only
        self.pad_dims = pad_dims

    def __len__(self):
        try:
            return len(self.data)
        except TypeError:
            return 1


def _stack_tensor(items):
    if len(items) == 1:
        x = items[0]
        if torch.is_tensor(x) and x.dim() == 3:
            return x.unsqueeze(0)
        return x
    return torch.stack(items, dim=0)


def collate(batch, samples_per_gpu: int = 1):
    first = batch[0]
    if isinstance(first, DataContainer):
        data = [sample.data for sample in batch]
        if first.cpu_only:
            return DataContainer([data], cpu_only=True)
        if first.stack:
            return DataContainer([_stack_tensor(data)], stack=True, padding_value=first.padding_value)
        return DataContainer(data)
    if isinstance(first, dict):
        return {key: collate([sample[key] for sample in batch], samples_per_gpu) for key in first}
    if isinstance(first, (list, tuple)):
        transposed = list(zip(*batch))
        return [collate(list(items), samples_per_gpu) for items in transposed]
    if torch.is_tensor(first):
        return _stack_tensor(batch)
    return batch


def _unwrap(obj):
    if isinstance(obj, DataContainer):
        return _unwrap(obj.data)
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_unwrap(v) for v in obj)
    if isinstance(obj, list):
        return [_unwrap(v) for v in obj]
    return obj


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


class MMDataParallel(torch.nn.Module):
    def __init__(self, module, device_ids=None, **kwargs):
        super().__init__()
        self.module = module
        self.device_ids = device_ids or ([0] if torch.cuda.is_available() else [])
        if torch.cuda.is_available():
            self.module.cuda(self.device_ids[0])

    def forward(self, *args, **kwargs):
        kwargs = _unwrap(kwargs)
        # Old mmcv scatter removes the per-GPU DataContainer wrapper around metadata.
        if (isinstance(kwargs.get("img_metas"), list) and len(kwargs["img_metas"]) == 1
                and isinstance(kwargs["img_metas"][0], list)):
            kwargs["img_metas"] = kwargs["img_metas"][0]
        if torch.cuda.is_available():
            kwargs = _to_device(kwargs, torch.device(f"cuda:{self.device_ids[0]}"))
        return self.module(*args, **kwargs)


class MMDistributedDataParallel(MMDataParallel):
    pass


def scatter(inputs, target_gpus, dim=0):
    return [_unwrap(inputs)]


def get_dist_info():
    return 0, 1


def init_dist(*args, **kwargs):
    return None


def wrap_fp16_model(model):
    return model


class EvalHook:
    def __init__(self, *args, **kwargs):
        pass


class DistEvalHook(EvalHook):
    pass


def auto_fp16(*decorator_args, **decorator_kwargs):
    def wrap(func):
        return func
    if decorator_args and callable(decorator_args[0]):
        return decorator_args[0]
    return wrap


def trunc_normal_init(module, mean=0., std=1., a=-2., b=2., bias=0.):
    if hasattr(module, "weight") and module.weight is not None:
        torch.nn.init.trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
    if hasattr(module, "bias") and module.bias is not None:
        torch.nn.init.constant_(module.bias, bias)


def force_fp32(*decorator_args, **decorator_kwargs):
    def wrap(func):
        return func
    if decorator_args and callable(decorator_args[0]):
        return decorator_args[0]
    return wrap


def _install_module(name: str, attrs: dict[str, Any]) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod



def _install_optional_corruption_stubs() -> None:
    # The standard MaskCLIP configs do not use corruption transforms, but the
    # old pipeline imports their module eagerly. Provide tiny stubs so the
    # import succeeds without pulling extra visualization/corruption deps.
    import types
    if "skimage" not in sys.modules:
        sk = types.ModuleType("skimage")
        util = types.ModuleType("skimage.util")
        util.random_noise = lambda x, *a, **k: x
        filters = types.ModuleType("skimage.filters")
        filters.gaussian = lambda x, *a, **k: x
        sk.util = util
        sk.filters = filters
        sys.modules["skimage"] = sk
        sys.modules["skimage.util"] = util
        sys.modules["skimage.filters"] = filters
    if "torch.utils.tensorboard" not in sys.modules:
        tb = types.ModuleType("torch.utils.tensorboard")
        tb_utils = types.ModuleType("torch.utils.tensorboard._utils")
        tb_utils.figure_to_image = lambda *a, **k: None
        sys.modules["torch.utils.tensorboard"] = tb
        sys.modules["torch.utils.tensorboard._utils"] = tb_utils
    if "wand" not in sys.modules:
        wand = types.ModuleType("wand")
        image = types.ModuleType("wand.image")
        api = types.ModuleType("wand.api")
        color = types.ModuleType("wand.color")
        class _WandImage:
            def __init__(self, *a, **k):
                self.wand = None
            def make_blob(self):
                return b""
        class _Library:
            pass
        library = _Library()
        library.MagickMotionBlurImage = lambda *a, **k: None
        image.Image = _WandImage
        api.library = library
        sys.modules["wand"] = wand
        sys.modules["wand.image"] = image
        sys.modules["wand.api"] = api
        sys.modules["wand.color"] = color

def apply() -> None:
    _install_optional_corruption_stubs()

    # top-level mmcv compatibility
    mmcv.__version__ = "1.5.0"
    mmcv.Config = Config
    mmcv.ConfigDict = ConfigDict
    mmcv.is_str = is_str
    mmcv.is_list_of = is_list_of
    mmcv.is_tuple_of = lambda seq, t: isinstance(seq, tuple) and all(isinstance(x, t) for x in seq)
    mmcv.mkdir_or_exist = mkdir_or_exist
    mmcv.scandir = scandir
    mmcv.FileClient = FileClient
    mmcv.dump = dump
    mmcv.load = load

    if not hasattr(mmcv, "ProgressBar"):
        class ProgressBar:
            def __init__(self, task_num=0):
                self.task_num = task_num
                self.completed = 0
            def update(self, n=1):
                self.completed += n
                if self.task_num and (self.completed == self.task_num or self.completed % 100 == 0):
                    print(f"[{self.completed}/{self.task_num}]", flush=True)
        mmcv.ProgressBar = ProgressBar

    utils_mod = _install_module("mmcv.utils", {
        "Config": Config,
        "ConfigDict": ConfigDict,
        "DictAction": DictAction,
        "Registry": Registry,
        "build_from_cfg": build_from_cfg,
        "digit_version": digit_version,
        "print_log": print_log,
        "is_tuple_of": lambda seq, t: isinstance(seq, tuple) and all(isinstance(x, t) for x in seq),
        "is_list_of": is_list_of,
        "deprecated_api_warning": lambda *a, **k: (lambda f: f),
        "get_logger": lambda name, log_file=None, log_level=None: __import__("logging").getLogger(name),
        "get_git_hash": lambda *a, **k: "unknown",
        "collect_env": lambda: {},
        "to_2tuple": lambda x: x if isinstance(x, tuple) else (x, x),
    })
    parrots = _install_module("mmcv.utils.parrots_wrapper", {
        "_BatchNorm": torch.nn.modules.batchnorm._BatchNorm,
        "SyncBatchNorm": torch.nn.SyncBatchNorm,
    })
    utils_mod.parrots_wrapper = parrots

    parallel_mod = _install_module("mmcv.parallel", {
        "DataContainer": DataContainer,
        "collate": collate,
        "scatter": scatter,
        "MMDataParallel": MMDataParallel,
        "MMDistributedDataParallel": MMDistributedDataParallel,
    })

    runner_mod = _install_module("mmcv.runner", {
        "BaseModule": BaseModule,
        "ModuleList": torch.nn.ModuleList,
        "Sequential": torch.nn.Sequential,
        "auto_fp16": auto_fp16,
        "force_fp32": force_fp32,
        "get_dist_info": get_dist_info,
        "init_dist": init_dist,
        "load_checkpoint": load_checkpoint,
        "wrap_fp16_model": wrap_fp16_model,
        "EvalHook": EvalHook,
        "DistEvalHook": DistEvalHook,
        "HOOKS": Registry("hooks"),
        "build_optimizer": lambda *a, **k: None,
        "build_runner": lambda *a, **k: None,
        "obj_from_dict": lambda info, parent=None, default_args=None: build_from_cfg(info, parent, default_args),
        "_load_checkpoint": lambda filename, map_location=None, logger=None: torch.load(filename, map_location=map_location),
    })
    _install_module("mmcv.runner.base_module", {
        "BaseModule": BaseModule,
        "ModuleList": torch.nn.ModuleList,
        "Sequential": torch.nn.Sequential,
    })

    # mmcv.cnn exists in mmcv 2.x, but the old registry aliases are gone.
    import mmcv.cnn as cnn
    import mmengine.model as mmengine_model
    for _name in ("constant_init", "normal_init", "kaiming_init", "xavier_init"):
        if not hasattr(cnn, _name) and hasattr(mmengine_model, _name):
            setattr(cnn, _name, getattr(mmengine_model, _name))
    if not hasattr(cnn, "trunc_normal_"):
        cnn.trunc_normal_ = torch.nn.init.trunc_normal_
    if not hasattr(cnn, "Linear"):
        cnn.Linear = torch.nn.Linear
    if not hasattr(cnn, "Conv2d"):
        cnn.Conv2d = torch.nn.Conv2d
    weight_init_mod = _install_module("mmcv.cnn.utils.weight_init", {
        "constant_init": cnn.constant_init,
        "normal_init": cnn.normal_init,
        "kaiming_init": cnn.kaiming_init,
        "xavier_init": cnn.xavier_init,
        "trunc_normal_": cnn.trunc_normal_,
        "trunc_normal_init": trunc_normal_init,
    })
    cnn_utils_mod = _install_module("mmcv.cnn.utils", {
        "constant_init": cnn.constant_init,
        "normal_init": cnn.normal_init,
        "kaiming_init": cnn.kaiming_init,
        "xavier_init": cnn.xavier_init,
        "trunc_normal_": cnn.trunc_normal_,
        "trunc_normal_init": trunc_normal_init,
        "revert_sync_batchnorm": lambda model: model,
    })
    _install_module("mmcv.cnn.utils.sync_bn", {"revert_sync_batchnorm": lambda model: model})
    if not hasattr(cnn, "UPSAMPLE_LAYERS"):
        cnn.UPSAMPLE_LAYERS = Registry("upsample_layer")
    if not hasattr(cnn, "PLUGIN_LAYERS"):
        cnn.PLUGIN_LAYERS = Registry("plugin_layer")
    if not hasattr(cnn, "MODELS"):
        cnn.MODELS = Registry("mmcv_models")
    bricks_registry = _install_module("mmcv.cnn.bricks.registry", {
        "ATTENTION": Registry("mmcv_attention"),
        "NORM_LAYERS": Registry("norm_layer"),
        "PLUGIN_LAYERS": Registry("plugin_layer"),
        "POSITIONAL_ENCODING": Registry("positional_encoding"),
        "TRANSFORMER_LAYER": Registry("transformer_layer"),
        "TRANSFORMER_LAYER_SEQUENCE": Registry("transformer_layer_sequence"),
    })

    engine_mod = _install_module("mmcv.engine", {
        "collect_results_cpu": lambda result_part, size, tmpdir=None: result_part,
        "collect_results_gpu": lambda result_part, size: result_part,
    })

    mmcv.parallel = parallel_mod
    mmcv.runner = runner_mod
    mmcv.utils = utils_mod
    mmcv.engine = engine_mod
