"""Independent Phase3 deltas; never replace or unfreeze the Phase1 LoRA.

The native network is deliberately unregistered on EdgePhase3. Forward hooks
add the new delta, while this ModuleDict owns its parameters for DDP, optimizer
and checkpoint discovery. No framework code or native state-dict keys change.
"""
from __future__ import annotations

import math
from itertools import chain

import torch
from torch import nn
from torch.nn import functional as F

from .settings import RECIPE, configure_bridge, configure_checkpoint_bridge


TARGETS = ("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen")


def add_training_options(parser):
    from .validation import add_validation_options
    add_validation_options(parser)
    parser.add_argument("--phase3-lora-rank", type=int, default=0,
                        help="Additional generation Q/K/V/O LoRA; zero disables it")
    parser.add_argument("--phase3-lora-alpha", type=float,
                        help="Defaults to the additional LoRA rank (scale one)")
    parser.add_argument("--phase3-lora-lr", type=float, default=1e-5)
    parser.add_argument("--phase3-lora-weight-decay", type=float, default=0.0)
    parser.add_argument("--bridge-lr", type=float, help="Override the bridge peak LR")
    parser.add_argument("--bridge-weight-decay", type=float)
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="Zero disables EMA; otherwise track bridges and Phase3 LoRA together")
    return parser


def configure_training(args):
    """Only opt-in settings alter the legacy recipe/fingerprint."""
    configure_bridge(getattr(args, "bridge_parameterization", None) or "shared")
    from .runtime import configure_execution
    from .validation import configure_validation
    configure_execution(getattr(args, "bridge_attention_backend", None), getattr(args, "bridge_precision", None))
    configure_validation(args)
    rank, alpha = args.phase3_lora_rank, args.phase3_lora_alpha
    if rank < 0 or (rank == 0 and alpha is not None):
        raise ValueError("LoRA rank must be positive when alpha is supplied")
    if args.bridge_lr is not None:
        if not math.isfinite(args.bridge_lr) or args.bridge_lr <= 0:
            raise ValueError("Bridge LR must be finite and positive")
        RECIPE["lr"] = args.bridge_lr
    if args.bridge_weight_decay is not None:
        if not math.isfinite(args.bridge_weight_decay) or args.bridge_weight_decay < 0:
            raise ValueError("Bridge weight decay must be finite and nonnegative")
        RECIPE["weight_decay"] = args.bridge_weight_decay
    RECIPE.pop("phase3_lora", None)
    if rank:
        alpha = float(rank if alpha is None else alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if not math.isfinite(args.phase3_lora_lr) or args.phase3_lora_lr <= 0:
            raise ValueError("LoRA LR must be finite and positive")
        if not math.isfinite(args.phase3_lora_weight_decay) or args.phase3_lora_weight_decay < 0:
            raise ValueError("LoRA weight decay must be finite and nonnegative")
        RECIPE["phase3_lora"] = dict(rank=rank, alpha=alpha, targets=list(TARGETS),
            lr=args.phase3_lora_lr, weight_decay=args.phase3_lora_weight_decay,
            dtype="float32", initialization="kaiming_A_zero_B",
            gradient_objective="existing_mixed_specialist_loss")
    if not math.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1:
        raise ValueError("EMA decay must be in [0, 1); zero disables EMA")
    RECIPE.pop("trainable_ema", None)
    if args.ema_decay:
        RECIPE["trainable_ema"] = dict(decay=args.ema_decay, scope="bridges_and_phase3_lora")


def configure_checkpoint(payload, *, bridge_parameterization=None, bridge_attention_backend=None, bridge_precision=None):
    configure_checkpoint_bridge(payload, requested=bridge_parameterization)
    saved = payload.get("scheduler", {}).get("recipe", {})
    from .runtime import configure_execution
    execution = saved.get("bridge_execution", {})
    if ((bridge_attention_backend is not None and bridge_attention_backend != execution.get("backend", "dense"))
            or (bridge_precision is not None and bridge_precision != execution.get("precision", "fp32"))):
        raise ValueError("Requested bridge execution conflicts with checkpoint")
    configure_execution(execution.get("backend"), execution.get("precision"))
    for key in ("phase3_lora", "trainable_ema"):
        RECIPE.pop(key, None)
        if key in saved:
            RECIPE[key] = saved[key].copy()
    cfg = RECIPE.get("phase3_lora")
    if cfg and (int(cfg["rank"]) <= 0 or not math.isfinite(cfg["alpha"])
                or cfg["alpha"] <= 0 or cfg["targets"] != list(TARGETS)
                or cfg["dtype"] != "float32"):
        raise ValueError("Unsupported Phase3 LoRA checkpoint configuration")


class LowRankDelta(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.scale = float(alpha) / rank
        self.A = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.B = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        # nn.Linear initializes A with kaiming_uniform; only B is zeroed.
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.B(self.A(x.float())) * self.scale


class Phase3LoRA(nn.ModuleDict):
    def __init__(self, native, cfg=None, *, expected_layers=28):
        super().__init__()
        self.config = None if cfg is None else cfg.copy()
        self.target_paths = []
        self._handles = []
        if cfg is None:
            return
        candidates = [(path, module) for path, module in native.named_modules()
                      if path.rsplit(".", 1)[-1] in TARGETS]
        if len(candidates) != expected_layers * len(TARGETS):
            raise RuntimeError("Expected generation Q/K/V/O projections in every native layer")
        # Validate all targets before installing any hooks.
        for path, module in candidates:
            if (not isinstance(module, nn.Linear) or not hasattr(module, "lora_A")
                    or not hasattr(module, "lora_B")):
                raise RuntimeError(f"Phase3 LoRA requires the loaded Phase1 LoRA at {path}")
            if any(p.requires_grad for p in module.parameters()):
                raise RuntimeError(f"Phase1 projection must stay frozen: {path}")
            if hasattr(module, "_phase3_lora_handle"):
                raise RuntimeError(f"Phase3 LoRA is already installed: {path}")
        for path, module in candidates:
            key = path.replace(".", "__")
            if key in self:
                raise RuntimeError(f"Ambiguous adapter key: {path}")
            delta = LowRankDelta(module, int(cfg["rank"]), cfg["alpha"])
            self[key] = delta
            self.target_paths.append(path)

            def add_delta(_module, inputs, output, adapter=delta):
                return output + adapter(inputs[0]).to(output.dtype)

            handle = module.register_forward_hook(add_delta)
            module._phase3_lora_handle = handle
            self._handles.append(handle)


def trainable_parameters(model):
    return chain(model.bridges.parameters(), model.phase3_lora.parameters())


def trainable_modules(model):
    modules = {"bridges": model.bridges}
    if len(model.phase3_lora):
        modules["phase3_lora"] = model.phase3_lora
    return modules


def trainable_state(model):
    return {name: module.state_dict() for name, module in trainable_modules(model).items()}


def _validate_state(model, state):
    modules = trainable_modules(model)
    if set(state) != set(modules):
        raise ValueError("Checkpoint trainable modules differ (missing or unexpected Phase3 LoRA)")
    for name, module in modules.items():
        expected = module.state_dict()
        if set(state[name]) != set(expected):
            raise ValueError(f"Checkpoint {name} tensor keys differ")
        for key, value in expected.items():
            if state[name][key].shape != value.shape:
                raise ValueError(f"Checkpoint tensor shape differs: {name}.{key}")


def load_trainable_state(model, payload, *, weights="raw"):
    if weights not in {"raw", "ema"}:
        raise ValueError("Weights must be raw or ema")
    expected_bridge = payload.get("scheduler", {}).get("recipe", {}).get("bridge_parameterization", "shared")
    actual_bridges = {module.parameterization for module in model.bridges.modules()
                      if hasattr(module, "parameterization")}
    if actual_bridges and actual_bridges != {expected_bridge}:
        raise ValueError("Checkpoint bridge parameterization differs from the model")
    cfg = payload.get("scheduler", {}).get("recipe", {}).get("phase3_lora")
    if cfg != model.phase3_lora.config:
        raise ValueError("Checkpoint Phase3 LoRA configuration differs from the model")
    raw = {key: payload[key] for key in ("bridges", "phase3_lora") if key in payload}
    _validate_state(model, raw)
    state = raw
    if weights == "ema":
        if payload.get("ema") is None:
            raise ValueError("EMA weights requested but this checkpoint has no EMA")
        state = payload["ema"]["state"]
        _validate_state(model, state)
    for name, module in trainable_modules(model).items():
        module.load_state_dict(state[name], strict=True)


def new_optimizer(model):
    groups = [dict(params=model.bridges.parameters(), lr=RECIPE["lr"],
                   weight_decay=RECIPE["weight_decay"])]
    if len(model.phase3_lora):
        cfg = model.phase3_lora.config
        groups.append(dict(params=model.phase3_lora.parameters(), lr=cfg["lr"],
                           lr_scale=cfg["lr"] / RECIPE["lr"], weight_decay=cfg["weight_decay"]))
    return torch.optim.AdamW(groups, lr=RECIPE["lr"], betas=tuple(RECIPE["betas"]), eps=1e-8)


def set_optimizer_lr(optimizer, bridge_lr):
    for group in optimizer.param_groups:
        group["lr"] = bridge_lr * group.get("lr_scale", 1.0)


class TrainableEMA:
    """FP32 EMA of a matched bridge/adapter pair, updated after optimizer steps."""
    def __init__(self, model, decay):
        if not math.isfinite(decay) or not 0 < decay < 1:
            raise ValueError("Enabled EMA requires decay in (0, 1)")
        self.decay, self.num_updates = float(decay), 0
        self.state = {name: {key: value.detach().float().clone() for key, value in state.items()}
                      for name, state in trainable_state(model).items()}

    @torch.no_grad()
    def update(self, model):
        for name, state in trainable_state(model).items():
            for key, value in state.items():
                self.state[name][key].lerp_(value.detach().float(), 1 - self.decay)
        self.num_updates += 1

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates, state=self.state)

    def load_state_dict(self, model, payload):
        if payload["decay"] != self.decay or int(payload["num_updates"]) < 0:
            raise ValueError("EMA configuration differs from the checkpoint")
        _validate_state(model, payload["state"])
        for name, state in self.state.items():
            for key, value in state.items():
                value.copy_(payload["state"][name][key])
        self.num_updates = int(payload["num_updates"])


def new_ema(model):
    cfg = RECIPE.get("trainable_ema")
    return None if cfg is None else TrainableEMA(model, cfg["decay"])


def restore_ema(ema, model, payload):
    saved = payload.get("ema")
    if (ema is None) != (saved is None):
        raise ValueError("Resume EMA configuration differs from the checkpoint")
    if ema is not None:
        ema.load_state_dict(model, saved)
