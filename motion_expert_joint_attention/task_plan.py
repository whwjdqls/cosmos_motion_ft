"""Central per-task contract for the joint-attention multimodal model.

THE SINGLE SOURCE OF TRUTH for how the seven base and two opt-in experimental tasks pack,
condition, and supervise the 5 modalities ``{text, image, video, camera, motion}``. Every other file imports from
here so the dataset (which fields to emit / when to blank the caption), the model
(``joint_motion_model.forward`` building ``condition_mask`` + ``gen_idx``), and the trainer
(``train.step_loss`` selecting + weighting the per-modality flow losses) all agree.

Authoritative design: ``DESIGN_7TASK.md`` (per-task masking table). This file encodes that
table programmatically. Pure python -- NO torch, NO cosmos_framework -- exactly like
``config.py`` so it stays cheap to import from any process.

The seven base tasks plus two experimental Phase-3 joint-target tasks (exact names,
used as the ``mode`` string everywhere)::

    inverse_dynamics   video                 -> camera           (NO text instruction)
    forward_dynamics   camera + text + image -> video
    policy             text + image          -> camera + video
    text2motion        text                  -> motion           (existing trained path)
    textimg2motion     text + image          -> motion
    motimg2video       motion + text + image -> video
    video2motion       video                 -> motion           (NO text instruction)
    video2camera_motion video                -> camera + motion  (NO text instruction)
    camimg2video_motion camera + image        -> video + motion   (NO text instruction)

The five modalities and their carriers in the packed sequence
``[ reasoner(text) | generator(image|video|camera) | motion ]``:

    text    -- reasoner tokens (always structurally present; empty caption -> 1 eos token).
    image   -- the generator's video latent FRAME 0 only, always left CLEAN (condition-only).
    video   -- generator video latent frames (T_lat of them after the Wan VAE 4x temporal
               compression of the T=33 pixel window); frame 0 == the image.
    camera  -- generator camera pseudo-action tokens, (T-1) of them, one per frame transition.
    motion  -- motion-expert tokens: 1 shape token (from neutral_joints, ALWAYS clean) followed
               by the valid (non-pad) motion frames.

THE CONDITIONING MECHANISM is a single per-token ``condition_mask`` (T-first, per token):

    1 = CLEAN    : sigma forced to 0, no timestep/AdaLN bias, EXCLUDED from the flow loss.
    0 = NOISED   : carries the timestep bias, supervised by the rectified-flow target.

"image" is literally "video latent frame 0 forced clean" -- there is no separate image
modality in the packed sequence. A task that lists ``image`` present but ``video`` absent
packs a single CLEAN latent frame (T_lat == 1 window). A task with both ``image`` and
``video`` packs the full latent stack with frame 0 clean (the image) and the rest per the
video policy.

LOSS WEIGHTS (per modality, summed over the supervised modalities of a task):

    motion = 1.0,  vision = 1.0,  camera ~= 10.0 (``ACTION_LOSS_WEIGHT``; the 9-d camera
    delta is small-magnitude so it is up-weighted to balance against vision/motion).
    Condition-only / absent modalities carry weight 0 -> contribute no loss by construction
    (the all-clean fast path already yields zero target).

TEXT POLICY (per task):

    "empty"    : caption is ALWAYS "" (the task has no instruction).  -> inverse_dynamics,
                 video2motion, video2camera_motion, camimg2video_motion. Structurally present
                 (1 eos token) so the causal reasoner block always has >= 1 row and the CFG-null
                 contract is preserved.
    "cfg_drop" : the real caption, dropped to "" with prob ``cfg_dropout`` (default 0.10) per
                 train sample -> a valid CFG-null at inference. All other five base tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Shared constants (kept in sync with config.py; duplicated here so task_plan.py
# stays importable on its own with zero deps for the dataset-side caption logic).
# ----------------------------------------------------------------------------
TASKS: Tuple[str, ...] = (
    "inverse_dynamics",
    "forward_dynamics",
    "policy",
    "text2motion",
    "textimg2motion",
    "motimg2video",
    "video2motion",
    "video2camera_motion",
    "camimg2video_motion",
)

MODALITIES: Tuple[str, ...] = ("text", "image", "video", "camera", "motion")

# Per-modality loss weights (camera up-weighted; see module docstring + DESIGN_7TASK.md).
W_MOTION: float = 1.0
W_VISION: float = 1.0
ACTION_LOSS_WEIGHT: float = 10.0   # camera flow-loss weight (== config.ACTION_LOSS_WEIGHT)

CAMERA_DOMAIN_ID: int = 2          # Cosmos "camera_pose" domain (== camera_to_action.DOMAIN_ID)
CAMERA_RAW_DIM: int = 9            # supervised camera channels [:9] (3 pos + 6 rot6d)

# ----------------------------------------------------------------------------
# MOTION-WEIGHTED default task mixture (sums to 1.0).
#
# Motion-generation tasks (text2motion / textimg2motion / video2motion) carry the bulk of the
# probability mass -- this is the "motion-weighted" mixture confirmed for the first run -- with
# the camera/video world-model tasks (inverse/forward/policy) and the motion-conditioned video
# task (motimg2video) sharing the remainder. text2motion is highest because it ALSO draws the
# large BONES-SEED stream (see TASK_SOURCES), giving it broad text-conditioning coverage.
# ----------------------------------------------------------------------------
TASK_WEIGHTS: Dict[str, float] = {
    "text2motion":      0.30,
    "textimg2motion":   0.20,
    "video2motion":     0.15,
    "motimg2video":     0.10,
    "policy":           0.10,
    "forward_dynamics": 0.08,
    "inverse_dynamics": 0.07,
    # Experimental Phase-3 objectives are opt-in. Production launchers must assign
    # explicit positive weights rather than silently changing the historical 7-task mix.
    "video2camera_motion": 0.0,
    "camimg2video_motion": 0.0,
}

# Which DATA SOURCE each task can draw from. NymeriaPlus windows have ALL 5 modalities aligned
# (713/713 seqs, 0 missing) so they serve every task. BONES-SEED is motion-only (no video /
# image / camera) so it can ONLY serve text2motion. The dataset MUST route each sample's mode to
# a source that actually has the required modalities.
TASK_SOURCES: Dict[str, Tuple[str, ...]] = {
    "inverse_dynamics": ("nymeria",),
    "forward_dynamics": ("nymeria",),
    "policy":           ("nymeria",),
    "text2motion":      ("nymeria", "bones"),
    "textimg2motion":   ("nymeria",),
    "motimg2video":     ("nymeria",),
    "video2motion":     ("nymeria",),
    "video2camera_motion": ("nymeria",),
    "camimg2video_motion": ("nymeria",),
}


# ============================================================================
# Per-modality plan
# ============================================================================
@dataclass(frozen=True)
class ModalityPlan:
    """How ONE modality participates in ONE task.

    present
        Is this modality packed into the sequence at all?
    role
        Human-readable role for logging/debugging: "absent" | "condition" (all clean, no loss) |
        "target" (some frames noised + supervised) | "context" (text reasoner).
    clean_policy
        Which frames are CLEAN (condition_mask=1, sigma=0, no loss). One of:
          "all"        -- every present frame is clean (condition-only modality).
          "none"       -- every present frame is noised + supervised (fully a target).
          "frame0"     -- ONLY latent frame 0 is clean (the image); frames 1.. are noised.
          "shape_only" -- (motion) the shape token is clean; motion frames follow `supervised`.
        For motion, the per-frame noised/clean split is driven by `supervised` together with the
        valid (non-pad) mask supplied at call time -- "shape_only" + supervised=True noises all
        valid frames; "all" leaves every motion frame clean (motion-as-condition, e.g.
        motimg2video).
    supervised
        Does this modality carry a flow-matching loss for this task?
    loss_weight
        Weight applied to this modality's flow loss (0.0 when not supervised).
    """
    present: bool
    role: str
    clean_policy: str
    supervised: bool
    loss_weight: float = 0.0


@dataclass(frozen=True)
class TaskPlan:
    """The full per-task contract: one ModalityPlan per modality + text policy + sources.

    text_policy
        "empty"    -> caption always "" for tasks without an instruction.
        "cfg_drop" -> real caption, dropped to "" with prob cfg_dropout for instructed tasks.
    sources
        Data sources that can serve this task (see TASK_SOURCES).
    weight
        Default mixture weight (see TASK_WEIGHTS).
    """
    mode: str
    text: ModalityPlan
    image: ModalityPlan
    video: ModalityPlan
    camera: ModalityPlan
    motion: ModalityPlan
    text_policy: str
    sources: Tuple[str, ...]
    weight: float

    # -- convenience accessors ------------------------------------------------
    def modality(self, name: str) -> ModalityPlan:
        if name not in MODALITIES:
            raise KeyError(f"unknown modality {name!r}; expected one of {MODALITIES}")
        return getattr(self, name)

    @property
    def present_modalities(self) -> List[str]:
        return [m for m in MODALITIES if self.modality(m).present]

    @property
    def supervised_modalities(self) -> List[str]:
        return [m for m in MODALITIES if self.modality(m).supervised]

    @property
    def has_gen(self) -> bool:
        """True iff any generator-carried modality (image/video/camera) is packed -> gen_idx
        is non-empty for this task (the seam that joint_motion_model.forward must fill)."""
        return self.image.present or self.video.present or self.camera.present

    @property
    def caption_always_empty(self) -> bool:
        return self.text_policy == "empty"


# ============================================================================
# The seven base task plans plus opt-in Phase-3 joint-target plans.
# ============================================================================
def _absent() -> ModalityPlan:
    return ModalityPlan(present=False, role="absent", clean_policy="all",
                        supervised=False, loss_weight=0.0)


def _text(policy: str) -> ModalityPlan:
    # Text is always present (structurally), always a context/condition, never supervised.
    return ModalityPlan(present=True, role="context", clean_policy="all",
                        supervised=False, loss_weight=0.0)


# Build each TaskPlan explicitly so it reads as a direct transcription of the spec table.
_PLANS: Dict[str, TaskPlan] = {}

# 1. inverse_dynamics : video (clean) -> camera (noised). NO text, NO image-as-separate, NO motion.
_PLANS["inverse_dynamics"] = TaskPlan(
    mode="inverse_dynamics",
    text=_text("empty"),
    image=_absent(),
    video=ModalityPlan(True, "condition", "all", supervised=False, loss_weight=0.0),
    camera=ModalityPlan(True, "target", "none", supervised=True, loss_weight=ACTION_LOSS_WEIGHT),
    motion=_absent(),
    text_policy="empty",
    sources=TASK_SOURCES["inverse_dynamics"],
    weight=TASK_WEIGHTS["inverse_dynamics"],
)

# 2. forward_dynamics : camera (clean) + text + image (frame0 clean) -> video (frames 1.. noised).
_PLANS["forward_dynamics"] = TaskPlan(
    mode="forward_dynamics",
    text=_text("cfg_drop"),
    image=ModalityPlan(True, "condition", "frame0", supervised=False, loss_weight=0.0),
    video=ModalityPlan(True, "target", "frame0", supervised=True, loss_weight=W_VISION),
    camera=ModalityPlan(True, "condition", "all", supervised=False, loss_weight=0.0),
    motion=_absent(),
    text_policy="cfg_drop",
    sources=TASK_SOURCES["forward_dynamics"],
    weight=TASK_WEIGHTS["forward_dynamics"],
)

# 3. policy : text + image (frame0 clean) -> camera (noised) + video (frames 1.. noised). Joint gen.
_PLANS["policy"] = TaskPlan(
    mode="policy",
    text=_text("cfg_drop"),
    image=ModalityPlan(True, "condition", "frame0", supervised=False, loss_weight=0.0),
    video=ModalityPlan(True, "target", "frame0", supervised=True, loss_weight=W_VISION),
    camera=ModalityPlan(True, "target", "none", supervised=True, loss_weight=ACTION_LOSS_WEIGHT),
    motion=_absent(),
    text_policy="cfg_drop",
    sources=TASK_SOURCES["policy"],
    weight=TASK_WEIGHTS["policy"],
)

# 4. text2motion : text -> motion (all valid frames noised). The existing trained path.
_PLANS["text2motion"] = TaskPlan(
    mode="text2motion",
    text=_text("cfg_drop"),
    image=_absent(),
    video=_absent(),
    camera=_absent(),
    motion=ModalityPlan(True, "target", "shape_only", supervised=True, loss_weight=W_MOTION),
    text_policy="cfg_drop",
    sources=TASK_SOURCES["text2motion"],
    weight=TASK_WEIGHTS["text2motion"],
)

# 5. textimg2motion : text + image -> motion (all valid frames noised).
#    New/correct runs use --textimg_condition=reasoner, so the image is raw frame-0 pixels encoded
#    by the Qwen-VL reasoner and no generator rows are packed. The old 1-clean-generator-latent
#    path is deprecated and kept only for historical checkpoint compatibility.
_PLANS["textimg2motion"] = TaskPlan(
    mode="textimg2motion",
    text=_text("cfg_drop"),
    image=ModalityPlan(True, "condition", "frame0", supervised=False, loss_weight=0.0),
    video=_absent(),   # reasoner-image path packs no gen rows; deprecated gen-image path resolves t_lat=1
    camera=_absent(),
    motion=ModalityPlan(True, "target", "shape_only", supervised=True, loss_weight=W_MOTION),
    text_policy="cfg_drop",
    sources=TASK_SOURCES["textimg2motion"],
    weight=TASK_WEIGHTS["textimg2motion"],
)

# 6. motimg2video : motion (all clean) + text + image (frame0 clean) -> video (frames 1.. noised).
_PLANS["motimg2video"] = TaskPlan(
    mode="motimg2video",
    text=_text("cfg_drop"),
    image=ModalityPlan(True, "condition", "frame0", supervised=False, loss_weight=0.0),
    video=ModalityPlan(True, "target", "frame0", supervised=True, loss_weight=W_VISION),
    camera=_absent(),
    motion=ModalityPlan(True, "condition", "all", supervised=False, loss_weight=0.0),
    text_policy="cfg_drop",
    sources=TASK_SOURCES["motimg2video"],
    weight=TASK_WEIGHTS["motimg2video"],
)

# 7. video2motion : video (all clean) -> motion (all valid frames noised). NO text, NO image.
_PLANS["video2motion"] = TaskPlan(
    mode="video2motion",
    text=_text("empty"),
    image=_absent(),
    video=ModalityPlan(True, "condition", "all", supervised=False, loss_weight=0.0),
    camera=_absent(),
    motion=ModalityPlan(True, "target", "shape_only", supervised=True, loss_weight=W_MOTION),
    text_policy="empty",
    sources=TASK_SOURCES["video2motion"],
    weight=TASK_WEIGHTS["video2motion"],
)

# Phase-3 multitask A: clean video -> jointly denoise camera and motion. Each target branch
# carries half its normal weight so one two-target sample has the same total branch budget as
# one single-target sample. Camera keeps its established relative up-weight inside that half.
_PLANS["video2camera_motion"] = TaskPlan(
    mode="video2camera_motion",
    text=_text("empty"),
    image=_absent(),
    video=ModalityPlan(True, "condition", "all", supervised=False, loss_weight=0.0),
    camera=ModalityPlan(True, "target", "none", supervised=True,
                        loss_weight=0.5 * ACTION_LOSS_WEIGHT),
    motion=ModalityPlan(True, "target", "shape_only", supervised=True,
                        loss_weight=0.5 * W_MOTION),
    text_policy="empty",
    sources=TASK_SOURCES["video2camera_motion"],
    weight=TASK_WEIGHTS["video2camera_motion"],
)

# Phase-3 multitask B: clean camera + frame-0 image -> jointly denoise future video and
# motion. Text is deliberately empty: the experiment tests physical cross-modal coupling,
# not whether a caption can independently explain the body motion.
_PLANS["camimg2video_motion"] = TaskPlan(
    mode="camimg2video_motion",
    text=_text("empty"),
    image=ModalityPlan(True, "condition", "frame0", supervised=False, loss_weight=0.0),
    video=ModalityPlan(True, "target", "frame0", supervised=True,
                       loss_weight=0.5 * W_VISION),
    camera=ModalityPlan(True, "condition", "all", supervised=False, loss_weight=0.0),
    motion=ModalityPlan(True, "target", "shape_only", supervised=True,
                        loss_weight=0.5 * W_MOTION),
    text_policy="empty",
    sources=TASK_SOURCES["camimg2video_motion"],
    weight=TASK_WEIGHTS["camimg2video_motion"],
)

assert set(_PLANS) == set(TASKS), "every task in TASKS must have a TaskPlan"


def build_task_plan(mode: str) -> TaskPlan:
    """Return the frozen TaskPlan for ``mode`` (one of TASKS). Raises on unknown mode."""
    try:
        return _PLANS[mode]
    except KeyError:
        raise KeyError(f"unknown task mode {mode!r}; expected one of {TASKS}") from None


# ============================================================================
# Per-sample condition-mask + loss-target resolver.
#
# Given a task + the per-sample frame counts (which differ sample to sample because the motion
# valid-frame count and the latent/camera counts are data-dependent), produce the concrete
# CLEAN/NOISED boolean masks and loss spec the model/flow consume. Returns plain python lists of
# bools (no torch) so the caller materializes tensors in its own device/dtype.
# ============================================================================
@dataclass(frozen=True)
class ModalityResolved:
    """Concrete per-token CLEAN/NOISED layout + loss spec for ONE modality of ONE sample.

    present
        Whether the modality is packed for this sample.
    n_tokens
        Number of tokens this modality contributes to the packed sequence (e.g. T_lat latent
        frames, T-1 camera frames, or 1 shape token + n_valid motion frames).
    condition_mask
        Length-``n_tokens`` list of bools: True == CLEAN (sigma 0, no loss), False == NOISED
        (supervised). The model turns this into per-token sigma scaling.
    supervised
        Whether a flow loss is computed for this modality.
    loss_weight
        Weight applied to that loss (0.0 when not supervised).
    loss_channels
        Optional channel slice the loss is restricted to. ``None`` == all channels (vision /
        motion). For camera it is ``(0, CAMERA_RAW_DIM)`` -> supervise only channels [:9],
        ignoring the zero-pad out to max_action_dim.
    includes_shape_token
        (motion only) True when the first token is the always-clean shape token, so the loss /
        decode must skip index 0.
    """
    present: bool
    n_tokens: int
    condition_mask: List[bool]
    supervised: bool
    loss_weight: float
    loss_channels: Optional[Tuple[int, int]] = None
    includes_shape_token: bool = False


@dataclass(frozen=True)
class ResolvedPlan:
    """The whole per-sample resolution: one ModalityResolved per PRESENT modality + bookkeeping."""
    mode: str
    text_is_empty: bool                     # caption forced "" (before cfg_drop is applied)
    modalities: Dict[str, ModalityResolved]  # keyed by modality name, only PRESENT ones
    has_gen: bool                            # any generator modality present (image/video/camera)

    def supervised_losses(self) -> List[Tuple[str, float, Optional[Tuple[int, int]]]]:
        """List of (modality, loss_weight, loss_channels) the trainer should sum."""
        return [
            (name, m.loss_weight, m.loss_channels)
            for name, m in self.modalities.items()
            if m.supervised
        ]


def _video_condition_mask(clean_policy: str, t_lat: int) -> List[bool]:
    """CLEAN/NOISED per latent frame for a video/image generator segment of ``t_lat`` frames."""
    if clean_policy == "all":          # condition-only (e.g. inverse_dynamics, video2motion)
        return [True] * t_lat
    if clean_policy == "frame0":       # image == frame 0 clean; frames 1.. noised
        return [i == 0 for i in range(t_lat)]
    if clean_policy == "none":         # fully noised target (not used for video today)
        return [False] * t_lat
    raise ValueError(f"bad video clean_policy {clean_policy!r}")


def _motion_condition_mask(
    clean_policy: str, motion_valid_mask: List[bool], has_shape_token: bool
) -> List[bool]:
    """CLEAN/NOISED per motion token: [shape_tok?] + one entry per VALID motion frame.

    Pad frames are dropped BEFORE this (the packed sequence only carries valid frames), so
    ``motion_valid_mask`` here is expected to be the already-filtered list of valid frames; its
    length is the number of motion frame tokens. The shape token (if present) is always CLEAN.
      clean_policy == "shape_only" -> shape clean, every motion frame NOISED (supervised target).
      clean_policy == "all"        -> shape clean, every motion frame CLEAN (motion-as-condition).
    """
    n_frames = len(motion_valid_mask)
    if clean_policy == "shape_only":
        frame_mask = [False] * n_frames        # all valid frames noised
    elif clean_policy == "all":
        frame_mask = [True] * n_frames         # all clean (condition)
    else:
        raise ValueError(f"bad motion clean_policy {clean_policy!r}")
    return ([True] if has_shape_token else []) + frame_mask


def resolve_sample(
    mode: str,
    *,
    t_lat: int = 0,
    n_camera: int = 0,
    motion_valid_mask: Optional[List[bool]] = None,
    has_shape_token: bool = True,
    derived_camera_condition: bool = False,
) -> ResolvedPlan:
    """Resolve a TaskPlan against ONE sample's per-modality frame counts.

    Args:
        mode: task name (one of TASKS).
        t_lat: number of video latent frames packed for this sample. For the deprecated
            generator-latent textimg2motion path this should be 1 (the single clean image latent
            frame); for tasks with the full video stack (inverse/forward/policy/motimg2video/
            video2motion) it is the Wan-VAE latent length of the T=33 pixel window. Ignored for
            reasoner-image textimg2motion and tasks with no video/image.
        n_camera: number of camera action frames (== T-1 for a T-frame window). Normally ignored
            for tasks with no camera. With ``derived_camera_condition=True``, motimg2video packs
            clean camera actions deterministically derived from its clean motion condition.
        motion_valid_mask: per-VALID-frame list (already pad-filtered) for the motion segment;
            its length is the motion frame-token count. Required for motion tasks.
        has_shape_token: whether the motion segment leads with the (always-clean) shape token.

    Returns a ResolvedPlan with a ModalityResolved for each PRESENT modality, carrying the
    concrete ``condition_mask`` (CLEAN/NOISED per token) and the loss spec. The model builds the
    packed segments in this same order and applies ``sigma_eff = sigma * (1 - condition_mask)``;
    the trainer sums the supervised modality losses with their weights/channels.
    """
    plan = build_task_plan(mode)
    out: Dict[str, ModalityResolved] = {}

    # ---- image: a single CLEAN latent frame (only when video is NOT also present; otherwise the
    #            image is folded into the video stack as frame 0 and tracked under "video").
    if plan.image.present and not plan.video.present:
        out["image"] = ModalityResolved(
            present=True, n_tokens=1, condition_mask=[True],
            supervised=False, loss_weight=0.0,
        )

    # ---- video (may include frame 0 == the image when plan.image is also present).
    if plan.video.present:
        if t_lat <= 0:
            raise ValueError(f"task {mode!r} packs video but t_lat={t_lat}")
        cmask = _video_condition_mask(plan.video.clean_policy, t_lat)
        out["video"] = ModalityResolved(
            present=True, n_tokens=t_lat, condition_mask=cmask,
            supervised=plan.video.supervised, loss_weight=plan.video.loss_weight,
            loss_channels=None,
        )

    if derived_camera_condition and mode != "motimg2video":
        raise ValueError(
            "derived_camera_condition is only valid for motimg2video; "
            f"got mode={mode!r}"
        )

    # ---- camera: native task camera, or a clean motion-derived M2V condition.
    if plan.camera.present or derived_camera_condition:
        if n_camera <= 0:
            raise ValueError(f"task {mode!r} packs camera but n_camera={n_camera}")
        camera_plan = plan.camera
        if derived_camera_condition:
            cmask = [True] * n_camera
            supervised = False
            loss_weight = 0.0
            loss_channels = None
        else:
            cmask = ([True] * n_camera if camera_plan.clean_policy == "all"
                     else [False] * n_camera)
            supervised = camera_plan.supervised
            loss_weight = camera_plan.loss_weight
            loss_channels = (0, CAMERA_RAW_DIM) if camera_plan.supervised else None
        out["camera"] = ModalityResolved(
            present=True, n_tokens=n_camera, condition_mask=cmask,
            supervised=supervised, loss_weight=loss_weight,
            loss_channels=loss_channels,
        )

    # ---- motion: [shape_tok?] + valid frames.
    if plan.motion.present:
        if motion_valid_mask is None:
            raise ValueError(f"task {mode!r} packs motion but motion_valid_mask is None")
        cmask = _motion_condition_mask(plan.motion.clean_policy, motion_valid_mask, has_shape_token)
        out["motion"] = ModalityResolved(
            present=True, n_tokens=len(cmask), condition_mask=cmask,
            supervised=plan.motion.supervised, loss_weight=plan.motion.loss_weight,
            loss_channels=None, includes_shape_token=has_shape_token,
        )

    return ResolvedPlan(
        mode=mode,
        text_is_empty=plan.caption_always_empty,
        modalities=out,
        has_gen=plan.has_gen or derived_camera_condition,
    )


# ============================================================================
# __main__ self-check: print every plan + a resolved example.
# ============================================================================
def _fmt_modplan(m: ModalityPlan) -> str:
    if not m.present:
        return "absent"
    sup = f"sup(w={m.loss_weight:g})" if m.supervised else "cond"
    return f"present role={m.role:9s} clean={m.clean_policy:10s} {sup}"


if __name__ == "__main__":
    assert abs(sum(TASK_WEIGHTS.values()) - 1.0) < 1e-9, sum(TASK_WEIGHTS.values())
    assert set(TASK_WEIGHTS) == set(TASKS)
    assert set(TASK_SOURCES) == set(TASKS)

    print("=" * 100)
    print("JOINT-ATTENTION TASK PLANS")
    print("=" * 100)
    for mode in TASKS:
        p = build_task_plan(mode)
        print(f"\n[{mode}]  weight={p.weight:.2f}  text={p.text_policy:8s}  "
              f"sources={','.join(p.sources)}  has_gen={p.has_gen}")
        for name in MODALITIES:
            print(f"    {name:7s}: {_fmt_modplan(p.modality(name))}")
        print(f"    -> present={p.present_modalities}  supervised={p.supervised_modalities}")

    print("\n" + "=" * 100)
    print("RESOLVED EXAMPLES (per-sample condition_mask + loss spec)")
    print("=" * 100)
    # T=33 pixel window -> Wan VAE (4x temporal +1) -> T_lat=9 latent frames; camera = T-1 = 32.
    T_LAT, N_CAM = 9, 32
    examples = [
        ("inverse_dynamics", dict(t_lat=T_LAT, n_camera=N_CAM)),
        ("forward_dynamics", dict(t_lat=T_LAT, n_camera=N_CAM)),
        ("policy",           dict(t_lat=T_LAT, n_camera=N_CAM)),
        ("text2motion",      dict(motion_valid_mask=[True] * 33)),
        ("textimg2motion",   dict(t_lat=1, motion_valid_mask=[True] * 33)),
        ("motimg2video",     dict(t_lat=T_LAT, motion_valid_mask=[True] * 33)),
        ("video2motion",     dict(t_lat=T_LAT, motion_valid_mask=[True] * 33)),
        ("video2camera_motion", dict(t_lat=T_LAT, n_camera=N_CAM,
                                      motion_valid_mask=[True] * 33)),
        ("camimg2video_motion", dict(t_lat=T_LAT, n_camera=N_CAM,
                                      motion_valid_mask=[True] * 33)),
    ]
    for mode, kw in examples:
        r = resolve_sample(mode, **kw)
        print(f"\n[{mode}]  text_is_empty={r.text_is_empty}  has_gen={r.has_gen}")
        for name, m in r.modalities.items():
            n_clean = sum(m.condition_mask)
            n_noised = m.n_tokens - n_clean
            ch = "" if m.loss_channels is None else f" chan[{m.loss_channels[0]}:{m.loss_channels[1]}]"
            sup = f" LOSS w={m.loss_weight:g}{ch}" if m.supervised else " (condition-only)"
            shp = " +shape_tok" if m.includes_shape_token else ""
            print(f"    {name:7s}: n_tok={m.n_tokens:3d}  clean={n_clean:3d}  "
                  f"noised={n_noised:3d}{shp}{sup}")
        print(f"    supervised_losses={r.supervised_losses()}")

    print(f"\nOK: all {len(TASKS)} task plans + resolutions built and self-checked.")
