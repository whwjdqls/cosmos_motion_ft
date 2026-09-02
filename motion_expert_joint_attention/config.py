"""Central constants, default hyperparameters, and canonical paths for the
joint-attention motion-expert repo.

Importable by ``train.py`` / ``sample.py`` / ``dataset.py`` so they all agree on
the motion representation dims, the Cosmos-3 Nano model geometry, the data /
stats paths, and the default training recipe. No heavy imports here (no torch,
no cosmos_framework) -- keep this cheap so any module can pull constants from it.

See the repo README / architecture design for how these are used. Where values
mirror the existing experiments they are pinned to those (the 283-d uniego rep
and the PoC's tuned loss recipe; the Nano model geometry and TIMESTEP_SCALE from
``train_motion_ft.py``).

7-task extension (see ``DESIGN_7TASK.md``)
------------------------------------------
The repo grows from text->motion only into the full 7-task joint-attention model
(text/image/video/camera/motion in one packed sequence). This file adds, WITHOUT
breaking any existing key: the TASKS list + motion-weighted TASK_WEIGHTS mixture,
the shared 4N+1 window length (T=33), the camera (Cosmos action) constants,
the precomputed-video-latent + NymeriaPlus manifest paths, and the train-scope
toggles (reasoner_lora / gen_lora / gen_full). Motion is ALWAYS fully trained.
All defaults preserve today's text->motion behavior.
"""

import os

# ----------------------------------------------------------------------------
# Motion representation (283-d uniego, SOMA-30, 20 fps) -- shared with the PoC.
# ----------------------------------------------------------------------------
MOTION_DIM = 283          # per-frame uniego feature dim (motion2llm in / llm2motion out)
N_JOINTS = 30             # SOMA-30 skeleton joints; neutral_joints is (N_JOINTS, 3)
SHAPE_DIM = N_JOINTS * 3  # 90; flattened neutral_joints -> shape2llm input
FPS = 20                  # fixed; no fps conditioning

# ----------------------------------------------------------------------------
# Cosmos-3 Nano model geometry (must match the frozen reasoner+generator stack).
# ----------------------------------------------------------------------------
HIDDEN = 4096             # decoder hidden size (motion tokens live in this space) -- FIXED
N_LAYERS = 36             # number of MoT decoder layers -> 36 MoTJointLayer wrappers
TIMESTEP_SCALE = 0.001    # rectified-flow timestep scaling (pinned to train_motion_ft)

# ----------------------------------------------------------------------------
# Motion expert SIZE. The expert is deliberately SMALLER than the video generator
# (motion is 283-d, far simpler than video). Two of its dims are FIXED by the joint
# attention -- all roles stack into one q/k/v, so cross-role Q.K^T needs identical head
# geometry, and motion shares the 4096 residual width:
#   FIXED  : hidden=4096, num_heads / num_kv_heads / head_dim = the backbone's (GQA).
#   FREE   : the FFN width below. This is the ONLY size knob; set it well under the
#            generator's intermediate (~12288). 3072 -> motion expert ~2.9B (vs gen ~7B);
#            2048 -> ~2.4B; the (required) attention is a ~1.5B floor at 36 layers.
# The motion expert is ALWAYS randomly initialized -- never warm-started from the generator.
MOTION_INTERMEDIATE_SIZE = 3072

# SPARSE-DEPTH motion expert. The 3-way joint attention (reasoner+generator+motion) fires only at
# every Nth backbone layer; the frozen reasoner+generator still run all N_LAYERS. The motion-layer
# set is {i | (i+1) % stride == 0}, so with N_LAYERS=36:
#   stride=3 -> {2,5,...,35} = 12 motion blocks (~12x79.7M ~= 0.96B motion params);
#   stride=6 -> {5,11,...,35} =  6 motion blocks (~6x79.7M ~= 0.48B motion params).
# The last layer (35) is always a motion layer (36 is a multiple of both 3 and 6). Plain layers
# carry ZERO motion params and run only the frozen 2-way reasoner+generator path.
MOTION_LAYER_STRIDE = 3   # -> motion layers {2,5,8,...,35} = 12 blocks (~0.96B motion params)

# ----------------------------------------------------------------------------
# Shared window geometry (7-task: ALL modalities sliced at the SAME (uuid, start)).
# ----------------------------------------------------------------------------
# T must be 4N+1 for the Wan2.2 VAE temporal compression (matches nymeria_world).
# At 20 fps, T=33 is a ~1.6 s window; motion is [T,283], camera action is [T-1,9],
# video latents are (T_lat, C, h, w) with T_lat = (T-1)//4 + 1.
VIDEO_NUM_FRAMES = 33     # shared window length across motion/video/camera (4N+1)
assert VIDEO_NUM_FRAMES % 4 == 1, "VIDEO_NUM_FRAMES must be 4N+1 for the Wan VAE"

# ----------------------------------------------------------------------------
# Camera = Cosmos native action modality (domain "camera_pose"). See
# nymeria_world/camera_to_action.py -- these mirror EMBODIMENT_TO_* exactly. The
# (T-1, 9) relative SE(3) pseudo-action is zero-padded to max_action_dim before
# action2llm; the action loss supervises only the first raw_action_dim channels.
# ----------------------------------------------------------------------------
CAMERA_DOMAIN_ID = 2      # EMBODIMENT_TO_DOMAIN_ID["camera_pose"]
CAMERA_RAW_ACTION_DIM = 9 # 3 trans + 6 rot6d (EMBODIMENT_TO_RAW_ACTION_DIM)
CAMERA_MAX_ACTION_DIM = 64  # zero-pad target before action2llm (Cosmos default)
ACTION_LOSS_WEIGHT = 10.0  # camera flow-matching MSE on channels [:9] is up-weighted

# ----------------------------------------------------------------------------
# Seven base tasks plus two opt-in Phase-3 joint-target objectives.
# Each mode is a single string consumed by task_plan.build_task_plan(mode).
# TASK_WEIGHTS is MOTION-WEIGHTED: motion-producing tasks dominate early so the
# fully-trained motion expert + MotionHeads get the most signal; the camera
# world-model tasks (1,2,3) are kept smaller but non-zero. Weights are relative
# (the dataset normalizes them); a weight of 0.0 disables that task entirely.
# ----------------------------------------------------------------------------
TASKS = [
    "inverse_dynamics",   # 1. video        -> camera           (no motion, no text)
    "forward_dynamics",   # 2. camera+text+image -> video       (no motion)
    "policy",             # 3. text+image   -> camera+video     (no motion)
    "text2motion",        # 4. text         -> motion           (existing trained path)
    "textimg2motion",     # 5. text+image   -> motion
    "motimg2video",       # 6. motion+text+image -> video
    "video2motion",       # 7. video        -> motion           (no text)
    "video2camera_motion", # 8. video       -> camera+motion    (experimental, no text)
    "camimg2video_motion", # 9. camera+image -> video+motion    (experimental, no text)
]

# Motion-weighted mixture (relative; normalized at sample time). The four
# motion-producing tasks (4,5,6,7) carry ~78% of the mass for the first run;
# the pure camera world-model tasks (1,2,3) share the rest. text2motion is
# largest because BONES-SEED gives it the most coverage.
TASK_WEIGHTS = {
    "inverse_dynamics": 0.07,
    "forward_dynamics": 0.08,
    "policy":           0.07,
    "text2motion":      0.40,
    "textimg2motion":   0.15,
    "motimg2video":     0.08,
    "video2motion":     0.15,
    "video2camera_motion": 0.0,
    "camimg2video_motion": 0.0,
}
assert set(TASK_WEIGHTS) == set(TASKS), "TASK_WEIGHTS must cover exactly TASKS"

# ----------------------------------------------------------------------------
# Canonical paths.
# ----------------------------------------------------------------------------
_REPO_ROOT = os.environ.get(
    "REPO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_WEKA_ROOT = os.environ.get("WEKA_ROOT", "/mnt/projects/ll/jungbinc/weka")
_RUN_ROOT = os.environ.get("RUN_ROOT", os.path.join(_WEKA_ROOT, "cosmos_motion_ft_runs"))
_PoC = os.path.join(_REPO_ROOT, "motion_expert")

# Nymeria (text, motion) pairs -- reused verbatim from the PoC.
NYMERIA_PAIRS_TRAIN = os.path.join(_PoC, "pairs_train.jsonl")
NYMERIA_PAIRS_VAL = os.path.join(_PoC, "pairs_val.jsonl")

# Shared 283-d motion normalization stats (z-score for the uniego rep; default:
# Nymeria uniego283 stats, applied to both sources -- a documented approximation
# for BONES-SEED; see README / dataset.py). Motion is the ONLY modality that is
# z-scored: camera actions stay un-normalized (Cosmos-exact) and video stays in
# VAE latent space -- never cross-normalize (see DESIGN_7TASK.md section 4).
SHARED_MEAN = os.environ.get(
    "MOTION_STATS_MEAN", os.path.join(_PoC, "stats", "uniego283_mean.npy")
)
SHARED_STD = os.environ.get(
    "MOTION_STATS_STD", os.path.join(_PoC, "stats", "uniego283_std.npy")
)
# Aliases that name the 283-d motion stats explicitly (same files; clearer at the
# 7-task call sites that also juggle camera/video).
MOTION_STATS_MEAN = SHARED_MEAN
MOTION_STATS_STD = SHARED_STD

# BONES-SEED (text, motion) pairs in the 283-d uniego rep, built offline in the
# kimodo env (build_bones_pairs.py) and consumed here as plain jsonl.
BONES_PAIRS_TRAIN = os.path.join(_RUN_ROOT, "joint_attention", "bones_pairs_train.jsonl")
BONES_PAIRS_VAL = os.path.join(_RUN_ROOT, "joint_attention", "bones_pairs_val.jsonl")

# ----------------------------------------------------------------------------
# NymeriaPlus aligned 5-modality source (tasks 1,2,3,5,6,7 + task-4 image variant).
# Same manifest/split that nymeria_world's camera dataset uses, keyed on
# (uuid, start_frame). Video latents are PRECOMPUTED offline (precompute_latents.py)
# so the trainer only runs vae2llm/patchify -- see DESIGN_7TASK.md section 2.
# ----------------------------------------------------------------------------
NYMERIA_MANIFEST = os.path.join(
    _WEKA_ROOT, "nymeriaplus_kimodo_proportional", "video", "manifest_video.jsonl"
)
NYMERIA_SPLIT_FILE = os.path.join(
    _WEKA_ROOT, "nymeriaplus_kimodo_proportional", "train_test_split.json"
)
# Defaults to the immutable historical representation.  A new representation must
# opt in together with matching normalization statistics; see
# ``use_camera_head_v1.sh``.  Keeping the old path as default preserves every existing
# checkpoint's feature semantics.
NYMERIA_UNIEGO_ROOT = os.environ.get(
    "NYMERIA_UNIEGO_ROOT",
    os.path.join(_WEKA_ROOT, "nymeriaplus_kimodo_proportional", "uniego_rep"),
)
# Per-sequence floor calibration + bad-window drop list, precomputed offline by
# precompute_floor_calibration.py (kimodo env, CPU). Corrects the SOMA fit's constant
# per-seq foot penetration below the GT floor (delta_seq = d_minc(seq) - c0, applied ON TOP
# of the per-window multi-floor ground_offset_y) and lists wrong-floor / deep-penetration
# windows to skip at index-build time. If the file is missing the dataset WARNS loudly and
# proceeds uncalibrated (backward compat). See README "Floor calibration".
FLOOR_CALIBRATION_JSON = os.path.join(
    _WEKA_ROOT,
    "nymeriaplus_kimodo_proportional",
    "metadata",
    "floor_calibration.json",
)
# Root for precomputed Wan2.2-VAE video latents: {uuid}_{start}.npz holds the
# packed (T_lat, C, h, w) latents (+ optionally the (T-1,9) camera action).
VIDEO_LATENT_ROOT = os.path.join(
    _WEKA_ROOT, "nymeriaplus_kimodo_proportional", "joint_latents"
)

# Root for all run outputs / checkpoints (NFS, visible to every node).
RUNS_ROOT = _RUN_ROOT

# ----------------------------------------------------------------------------
# Default training hyperparameters. train.py should seed its argparse defaults
# from this dict so the recipe stays in one place.
# ----------------------------------------------------------------------------
TRAIN_DEFAULTS = {
    # optimization
    "lr": 2e-4,
    "warmup": 1000,
    "lr_schedule": "cosine",   # linear warmup -> cosine decay (see train_motion_ft.lr_factor)
    "min_lr_ratio": 0.1,
    "batch_size": 64,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "steps": 200000,

    # rectified-flow objective -- PER-MODALITY: this knob is the MOTION objective only
    # ('x0' default = the proven bs_train recipe: logit-normal sigma, net predicts clean
    # x0; 'velocity' stays selectable for ablation). VISION/CAMERA are ALWAYS velocity;
    # train.py's separate --gen_schedule selects historical uniform/Euler or native
    # shifted-Waver/UniPC time handling. (The motion objective never changes gen targets.)
    "objective": "x0",

    # loss weights (PoC-tuned recipe; geometric terms skipped when weight == 0)
    "w_feat": 1.0,             # masked MSE on the 283-d feature (motion)
    "w_joint": 10.0,           # centroid-relative decoded joint L2 (motion)
    "w_smooth": 50.0,          # decoded joint-velocity L2 (temporal smoothness, motion)
    "w_vision": 1.0,           # vision flow-matching MSE on noised latent frames (tasks 2,3,6)
    "w_camera": ACTION_LOSS_WEIGHT,  # camera flow-matching MSE on chan[:9] (tasks 1,3)

    # classifier-free guidance
    "cfg_dropout": 0.10,       # per-sample prob of dropping the caption to ""
                               # (every task with an instruction; inverse_dynamics /
                               #  video2motion always use "" -- no instruction)

    # ----- train scope (see DESIGN_7TASK.md section 5) -----------------------
    # Motion is ALWAYS fully trained (_moe_motion + MotionHeads + norm_moe_motion);
    # there is no toggle for it. The reasoner and generator pathways each pick
    # EXACTLY ONE of {frozen, lora, full}. Defaults below = today's text->motion
    # behavior (reasoner frozen, generator frozen). The FIRST 7-task run flips
    # gen_lora=True on the CLI (--gen_lora) per the confirmed plan; the dict
    # default stays False so importing config does not silently change scope.
    "reasoner_lora": False,    # LoRA on reasoner q/k/v/o_proj (else reasoner fully FROZEN)
    "gen_lora": False,         # LoRA on q/k/v/o_proj_moe_gen (mutually excl. with gen_full)
    "gen_full": False,         # full generator FT: all _moe_gen + vae2llm/llm2vae/
                               #   action2llm/llm2action/action_modality_embed trainable
                               #   (mutually exclusive with gen_lora; else gen frozen)

    # ----- task mixture (7-task) --------------------------------------------
    # Which tasks are active + their relative sampling weights. Sourced from the
    # motion-weighted TASK_WEIGHTS above; override per-run via --task_weights.
    "task_weights": dict(TASK_WEIGHTS),

    # logging / checkpointing
    "log_every": 20,
    "viz_every": 10000,
    "save_every": 10000,
}


def validate_train_scope(cfg: dict) -> None:
    """Assert the generator pathway picks at most one of {lora, full}.

    Cheap pure-python guard reused by train.py after argparse so a bad CLI combo
    (--gen_lora --gen_full) fails fast instead of silently double-counting params.
    Motion is always trained; reasoner_lora is independent of the gen choice.
    """
    if cfg.get("gen_lora") and cfg.get("gen_full"):
        raise ValueError(
            "gen_lora and gen_full are mutually exclusive (pick frozen / lora / full)"
        )

# ----------------------------------------------------------------------------
# Fixed eval prompts for in-training visualization (reused from
# train_motion_ft.VIZ_PROMPTS). List of (prompt, slug).
# ----------------------------------------------------------------------------
VIZ_PROMPTS = [
    ("a person walks forward", "walk"),
    ("a person waves their right hand", "wave"),
    ("a person sits down on a chair", "sit"),
    ("character picks up an object from the floor and then stands up straight", "pickup"),
    ("a person turns around and walks back the other way", "turn"),
]
