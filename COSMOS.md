## Finetuning notes (this project) — added by analysis

This COSMOS.md has been **trimmed** to the sections relevant to our project: adding a **3D human-motion
modality to the Cosmos3-Nano GENERATOR** (text -> motion, reasoner frozen). Full paper: arXiv:2606.02800
(complete original preserved at `COSMOS_full.md.bak`).

### Is our finetuning strategy okay? (assessment vs the paper)
**Verdict:** the *structure* matches Cosmos's design; some *hyperparameters* diverge from Cosmos's own
action-modality recipe (see **Sec 4.2.5 Robot Policy Post-Training**, the direct analog).

**Matches Cosmos:** freeze the reasoner + train only the generation (`_moe_gen`) pathway; add the new
modality via freshly-initialized projection heads (= robot policy's freshly-init action encoder/decoder/
embeddings); rectified-flow with velocity target `v = eps - x0` and x0 reconstruction; one token per frame
of the normalized continuous vector; bf16, grad-clip 1.0, no EMA. Our per-block smooth-L1 + FK loss is a
sensible motion-specific upgrade.

**Representation difference (important):** Cosmos action = **delta / relative SE(3) pseudo-actions**
(per-frame pose *deltas*, normalized to [-1,1]); ours = **absolute kinematic motion** (369-d kimodo rep:
joint rot6d + positions + velocities + foot contacts, z-scored). So Sec 4.2.5 is a **structural** analog only
(how to attach a continuous per-frame modality + the freeze/LR recipe) — keep kimodo's kinematic rep + FK
loss; do **not** adopt Cosmos's delta-action rep or plain MSE.

**Divergences to consider for a future run (NOT applied to the live runs):**
1. **LR ~10x low** — Cosmos robot policy uses base **2e-4 + 5x on the freshly-init heads (~1e-3)**; we use 2e-5, 1x.
2. **No 5x multiplier** on the new motion heads.
3. **Timestep**: we use Uniform(0,1); Cosmos uses **logit-normal + rectified-flow shift** (biases toward higher noise).
4. **Inference**: Cosmos action uses **shift~5, guidance=1**; our sampler default is shift=1, cfg=2.5 (now both
   selectable in `sample_motion.py` via `--shift` / `--cfg`).
5. **LoRA is off-recipe** — Cosmos full-finetunes the generator; projector/LoRA-only is reported to degrade.
6. Constant LR vs Cosmos warmup+decay; no 10% CFG text-dropout.

**Glossary.** *logit-normal*: `t = sigmoid(u)`, `u ~ Normal(mu, sigma^2)` -> concentrates training on mid-range
noise. *rectified-flow shift*: `sigma = s*tbar/(1 + (s-1)*tbar)`, `tbar = 1-t`, `s >= 1` (s=1 identity; s>1 ->
higher noise; Cosmos uses 3/5/10 by resolution, ~5 for action).

---
*Below: the trimmed Cosmos 3 paper — only the sections relevant to the generator + action/motion modality.*

## Cosmos 3: Omnimodal World Models for Physical AI
## 2 Model Architecture

Cosmos 3 is capable of processing multimodal inputs and generating multimodal outputs. Beyond language,
vision (image and video), and audio, Cosmos 3 treats action as a core modality, introducing a dedicated class of
action tokens. These action tokens bridge the physical world with language-based reasoning and video-based
world modeling, linking directly to physically grounded control signals for real-world interaction. Cosmos
3 integrates modality-specific encoders to project different modalities into a unified representation space,
which is then processed by a Mixture-of-Transformers (MoT) backbone. During inference, language tokens are
generated via next-token prediction, while other modalities are generated through iterative denoising.

### 2.1 Encoders

Given an input sequence of language, vision, audio, and action, the first step is to embed them into a unified
representation space using modality-specific encoders. To enable the shared transformer parameters and
positional embeddings to distinguish between different modalities, we add a learnable, modality-specific
embedding vector to each non-language modality before feeding it into the MoT backbone.

#### 2.1.1 Image and Video

We adopt two separate encoders for visual input. For visual understanding, we use a ViT encoder pre-trained
with vision-language alignment. For visual generation, we use the video VAE encoder from Wan2.2-TI2V-
5B (Wan et al., 2025a). The ViT encoder has a 16 × 16 patch size, followed by a two-layer MLP that merges
2 × 2 tokens and projects them into the latent space of the transformer. Following Qwen3-VL (Bai et al., 2025b),
we also aggregate visual features from ViT via DeepStack (Meng et al., 2024) and insert text–based video
timestamps interleaved with video frames (Chen et al., 2024b). The VAE compresses the input video temporally
by 4 ×and spatially by 32 × 32 , implemented as 16 × 16 spatial compression followed by a 2 × 2 patch merge.
We use a linear layer to project each VAE token into the transformer’s hidden dimension before feeding the
latents into the MoT backbone. The ViT encoder for understanding is jointly trained with the backbone, while
the VAE encoder for generation is kept frozen during training.

#### 2.1.2 Audio

For audio generation, we adopt the audio VAE architecture from Lee et al. (2025b). The raw stereo audio
sampled at 48 kHz is encoded with a hop size of 1920 samples, resulting in 25 tokens per second of audio. The
audio VAE is frozen during training. As with the other non-text modalities, audio tokens are projected into the
transformer’s hidden dimension using a linear layer before entering the MoT backbone.

#### 2.1.3 Action

We support action modeling across diverse embodiments, including autonomous vehicles, camera motion,
robots, and egocentric human motion (head and hands). Since each domain exposes its own native control
space—such as joint trajectories, steering commands, body poses, or camera transformations—we map them


```
Action Representation
Ego Pose 9D
3D pos+6D rot or
```
```
Effector Pose 9D
3D pos+6D rot
```
```
Grasp State 15D
3D pos×5 fingers
```
```
Grasp State 1D
1D open/close
```
```
Total: 9D
```
```
9D
```
```
Autonomous Vehicle
```
```
Total: 9D
```
```
9D
```
```
Camera Motion
```
```
Total: 57D
```
```
9D 9D15D 9D 15D
```
```
Egocentric Motion
```
```
Total: 10D
```
```
9D 1D
```
```
Single-Arm Robot
```
```
Total: 20D
```
```
9D 1D 9D 1D
```
```
Dual-Arm Robot
```
```
Total: 29D
```
```
9D 9D 1D 9D 1D
```
```
Humanoid Robot
```
Figure 3: **Unified action representation.** We map heterogeneous embodiment controls into compact action
vectors built from shared geometric components. Ego and effector motions are encoded as relative-pose
pseudo-actions using 3D translation and 6D rotation (an over-parameterized rotation representation by
Zhou Zhou et al. (2019), as the degree of freedom of rotation is 3), while grasp states directly encode the
current manipulation state, such as fingertip positions for hands or gripper open/close values for robots.
Domain-aware input and output projections handle heterogeneous action-vector lengths while preserving the
shared semantic space.

into a unified action interface that enables consistent multimodal reasoning, generation, and policy learning
across domains.

**Action representations.** We use actions to denote causal variables that induce changes in the world state.
Given consecutive video tokens, an action token𝑎𝑡represents the transition from the previous state𝑣𝑡− 1 to the
current state𝑣𝑡. Each embodiment source is transformed into a compact representation that captures a shared
underlying geometric structure across different action domains, as illustrated in Fig. 3. At a high level, actions
can include up to three components: ego poses for the agent’s main observation frame, effector poses for the
agent’s effectors, and grasp states for the manipulation state. To avoid embodiment-specific controller details
such as Proportional–Integral–Derivative (PID) parameters or low-level actuation interfaces, ego and effector
poses are represented as pseudo-actions derived from state differences. For consecutiveSE(3)posesT𝑡− 1 and
T𝑡, we represent motion as the relative transform∆T𝑡= T−𝑡−^11 T𝑡. We use the 6D representation following Zhou
et al. (2019) and the OpenCV convention for rotations where the z-axis is along the fingers/grippers and x-axis
is to the right. Grasp states, however, are treated differently: rather than representing temporal differences,
they directly encode the current manipulation state at time t.

For cameras and autonomous vehicles, actions are represented by ego poses only, without any effector poses or
grasp states. For egocentric data, we use head-camera pose deltas as ego poses, wrist-pose deltas as effector
poses, and fingertip positions in each wrist frame as grasp states (Yang et al., 2025d). For robotic data, we use
head-camera pose deltas as ego poses, end-effector flange-pose deltas as effector poses (Lyu et al., 2026), and
continuous gripper open/close values as grasp states.

**Action tokenization.** Our action representation maps diverse embodiments into a shared latent action space
while preserving embodiment-specific structure and semantics. We therefore use domain-aware input and
output projection layers with separate weight matrices for each embodiment domain (Zheng et al., 2026), while
sharing the MoT backbone. For an inputx∈ R𝑑

```
(𝑘)
in , such as an egocentric action vector concatenating the head-
```

pose delta, left and right wrist-pose deltas, and fingertip coordinates, and domain identifier𝑘 ∈{ 1 ,...,𝐾},
the input projection is:
z = W(in𝑘)x + b(in𝑘) (1)

wherez ∈ R𝑑modelis the latent action token,xdenotes the normalized action vector, andW(in𝑘)∈ R𝑑model×𝑑
(in𝑘)

and b(in𝑘)∈ R𝑑modelare the domain-specific input projection matrix and bias.

To decode the tokens back to the original action space, we use a domain-specific output projection:

```
x = Wout(𝑘)z + b(out𝑘) (2)
```
whereWout(𝑘)∈ R𝑑
(in𝑘)×𝑑model
andb(out𝑘) ∈ R𝑑
(in𝑘)
are the domain-specific output projection matrix and bias. All
projection parameters are initialized from scratch and optimized jointly with the MoT backbone. We convert
the predicted 6D rotation back to a 3 × 3 SO(3) rotation matrix using singular value decomposition (SVD).

### 2.2 Token Arrangement and Generation Mode

Cosmos 3 is a unified model that supports various modalities and tasks. Different tasks can be formulated as
interleaved multimodal sequences, each consisting of a series of segments from different modalities. Given a
task, all segments are first encoded into embeddings using the modality-specific encoders described above.
Once embedded, tokens from different modalities are packed using a unified format that applies across all
tasks, which we describe next.

#### 2.2.1 Token Arrangement

The input token sequence consists of two subsequences: an autoregressive (AR) subsequence followed by a
diffusion (DM) subsequence.

The **AR subsequence** is responsible for reasoning and understanding. It contains language tokens as well
as video and image tokens embedded by the ViT encoder. All AR tokens are routed to a dedicated set of
parameters in the transformer decoder layers.

The **diffusion subsequence** follows the AR subsequence and contains video and image tokens from the VAE
encoder, as well as audio and action tokens. During generation, the model iteratively denoises the noisy diffusion
tokens to produce the corresponding clean tokens. Diffusion tokens are routed to a separate parameter set from
that used by AR tokens, while still interacting with AR tokens through joint attention in each of the transformer
decoder layers.

For any given task, we apply the same format to arrange these tokens: (1) autoregressive tokens are placed
before diffusion tokens; (2) within the diffusion subsequence, for each modality, clean conditioning tokens are
placed before noisy diffusion tokens; and (3) within both the conditioning and diffusion subsequence, tokens
are ordered by vision, audio, and action modality. By using this unified format, Cosmos 3 can support various
generation tasks, which we detail below.

#### 2.2.2 Generation Mode

Cosmos 3 supports different modalities: language, vision, audio, and action. We denote clean vision, audio,
and action tokens as𝑣,𝑠, and𝑎, respectively, and their noisy counterparts with tildes: ̃,𝑣 ̃𝑠, and ̃𝑎. Given these
modalities, the supported generation modes are listed as follows:

- **Language.** For language generation, the input contains only the autoregressive subsequence, and the
    generation-specific diffusion parameters are not activated. Image and video inputs, if present, are
    embedded by the ViT encoder and placed in the autoregressive subsequence. In this setting, Cosmos 3
    operates like a standard VLM.


- **Text-to-Image.** In this mode, the autoregressive subsequence contains the language tokens, while the
    diffusion subsequence contains the noisy target image tokens embedded by the VAE encoder. The entire
    sequence of tokens becomes:
       ST2I= [SAR, ̃ 1 𝑣], (3)
    whereSAR≜ [𝑙 1 ,...,𝑙𝑛,⟨EOS⟩,⟨BOG⟩]is the AR prefix shared by all modes below (𝑙 1 ,...,𝑙𝑛are the
    language tokens;⟨EOS⟩and⟨BOG⟩are the end-of-sentence and begin-of-generation special tokens), and
       ̃ 1 𝑣 is the noisy image token.
- **Text-to-Video (+Audio).** This mode is similar to Text-to-Image, but the diffusion subsequence contains
    the noisy target video tokens instead. When audio is (optionally) generated jointly, noisy audio tokens
    are appended after the noisy vision tokens. In summary, the packed sequence becomes:

```
ST2V+Audio= [SAR, ̃𝑣1:𝑁, ̃]𝑠, (4)
```
```
where 𝑁 is the number of latent video frames.
```
- **Image-to-Video/Video-to-Video (+Audio).** This mode introduces an initial conditioning image or a
    number of initial video frames, and the model generates the complete continuation conditioned on them
    and the text prompt. In the diffusion subsequence, the clean conditioning image or video tokens are
    followed by the noisy target video tokens:

```
SV2V= [SAR, 𝑣1:𝑃, ̃𝑃𝑣+1:𝑁], (5)
where𝑃is the number of conditioning latent frames. When𝑃 = 1, the task becomes Image-to-Video,
while𝑃 > 1 corresponds to Video-to-Video. When audio is also generated, the audio tokens are appended
similarly to the Text-to-Video case.
```
- **Video transfer.** In this task, the input consists of a control video ( _e.g_ ., edge, or depth) together with a
    text description, and the model generates the corresponding RGB video. The token layout is similar to
    that of Video-to-Video, with the control-video tokens used as conditioning tokens and the RGB video
    tokens used as noisy target tokens:

```
STransfer= [SAR, 𝑣ctrl1:𝑁, ̃1:𝑣 𝑁], (6)
where 𝑣1:ctrl𝑁are the clean VAE-encoded tokens of the control video.
```
- **Action.** Cosmos 3 supports three generation modes for action—forward dynamics, inverse dynamics, and
    joint video-action prediction (policy). For a trajectory with consecutive video tokens, each action token𝑎𝑡
    represents the transition from𝑣𝑡− 1 to𝑣𝑡. Forward dynamics predicts future visual states conditioned on
    observed context and clean action tokens, while inverse dynamics infers the action tokens that explain an
    observed visual transition. In policy mode, the model jointly predicts action and video tokens, enabling it
    to generate both the intervention and its expected visual consequence under the same sequence model.
    The conditional directions are summarized in Fig. 4.

### 2.3 Mixture-of-Transformers (MoT) Architecture

Cosmos 3 adopts a **Mixture-of-Transformers (MoT)** architecture that processes a unified sequence of tokens
from different modalities. At the layer level, each transformer decoder layer contains two sets of parameters: one
for reasoning tasks, which processes tokens from the AR subsequence (reasoner), and one for generation tasks,
which processes tokens from the diffusion subsequence (generator). Although Cosmos 3 shares similarities
with unified generation models such as Deng et al. (2025) in its decoder-layer structure, it differs in its training
strategy, positional embeddings, and overall capabilities.


```
Forward Dynamics
```
```
Video
```
```
Action
```
𝑣𝑡− (^1) ̃𝑣𝑡 ̃𝑣𝑡+
𝑎𝑡 𝑎𝑡+
**Inverse Dynamics
Video
Action**
𝑣𝑡− 1 𝑣𝑡 𝑣𝑡+
̃𝑎𝑡 ̃𝑎𝑡+
**Policy
Video
Action**
𝑣𝑡− (^1) ̃𝑣𝑡 ̃𝑣𝑡+
̃𝑎𝑡 ̃𝑎𝑡+
**Tokens:** clean video noisy video clean action noisy action
Figure 4: **Action sequence configurations.** For a video-action data sample, Cosmos 3 constructs different
training modes by varying which tokens are clean and which are noisy. The diagram shows a local temporal
window in which action tokens lie between adjacent video tokens:𝑎𝑡connects𝑣𝑡− 1 to𝑣𝑡, and𝑎𝑡+1connects𝑣𝑡
to𝑣𝑡+1. Forward dynamics mode denoises vision tokens conditioned on clean action tokens; inverse dynamics
mode denoises action tokens conditioned on clean vision tokens; and video-action (policy) mode denoises both
vision and action tokens. Language and special tokens are omitted for compactness.

#### 2.3.1 Dual-Tower Layer Structure

A standard transformer decoder layer consists of a self-attention operation, a feed-forward network, and some
normalization layers. Instead of processing all token types with the same parameters, the MoT design uses two
pathways, as shown in Fig. 5. Each pathway is a standard transformer layer with its own parameters, including
layer normalization modules, attention projection matrices, and feed-forward networks. The two pathways are
both initialized from the weights of a pre-trained Vision-Language Model (VLM), allowing Cosmos 3 to inherit
strong language and visual reasoning capabilities while learning to generate high-fidelity videos. During both
training and inference, the AR subsequence at the front is routed to the reasoner tower, while the diffusion
subsequence at the back is routed to the generator tower.

#### 2.3.2 Dual-Stream Joint Attention

Although the two towers use independent parameters, tokens from the diffusion subsequence interact with the
AR subsequence through a dual-stream joint attention operation. Here we denote the query, key, and value
vectors of the AR and diffusion subsequences as QAR, KAR, VAR, QDM, KDM, and VDM, respectively.

**Autoregressive subsequence attention.** Tokens in the AR subsequence attend only to tokens within the AR
subsequence using _causal self-attention_ ; that is, each token can attend only to preceding tokens in the same
sequence. This is fully consistent with the autoregressive property inherited from the VLM backbone, allowing
the model to preserve the text-generation capability of the pre-trained VLM:

```
OAR= Attncausal
```
##### (︀

##### QAR, KAR, VAR

##### )︀

##### . (7)

**Diffusion subsequence attention.** Tokens in the DM subsequence use _full bidirectional attention_ , with the
union of AR and DM tokens serving as the keys and values. This allows each diffusion token to freely attend to
the text prompts from the autoregressive subsequence, as well as to all other conditional and diffusion tokens
in the sequence, thereby maintaining temporal and spatial consistency:

```
ODM= Attnfull
```
##### (︀

##### QDM, [KAR; KDM], [VAR; VDM]

##### )︀

##### , (8)

where[· ;·]denotes concatenation along the sequence dimension. We note that AR tokens are never updated
based on DM tokens, preserving the causal integrity of the conditioning pathway.

### 2.4 Multimodal Position Embedding

Position embeddings inject temporal and spatial structure into the attention mechanism, encouraging tokens to
attend more strongly to semantically and geometrically relevant tokens, often nearby in space or time. Since


```
×𝐿
```
```
𝑣AR 1 𝑣AR 2 ··· 𝑙 1 𝑙 2 ··· EOS BOG
AR subsequence
```
```
Encoder (ViT)Vision LanguageTokenizer
𝑣 ̃DM 1 ̃𝑣DM 2 ··· ̃𝑠 1 𝑠 ̃ 2 ··· ̃𝑎 1 ̃𝑎 2 ···
Diffusion subsequence
```
```
Encoder (VAE)Vision EncoderAudio EncoderAction
```
```
Layer Norm Layer Norm
Shared Multimodal Attention
Attn( Causal Self-Attn QAR,KAR,VAR) Attn(QDM, Full Attention [KAR;KDM],[VAR;VDM])
```
```
Layer Norm Layer Norm
MLP MLP
𝑙 2 𝑙 3 ··· 𝑣DM 1 𝑣DM 2 ··· 𝑠 1 𝑠 2 ··· 𝑎 1 𝑎 2 ···
Reasoner Generator
```
```
Tokens: Vision (AR) Language Special Vision (DM) Audio Action Noisy
```
```
(triangular)Causal Masked(zero)
```
```
attendFull attendFull
```
```
KAR Keys (K) KDM
```
```
Queries
```
```
(Q
```
```
)Q
```
```
AR
```
```
QDM
```
```
𝑣AR^1 ···𝑙^1 ···EOSBOG𝑣 ̃^1 DM··· ̃𝑠^1 ··· ̃𝑎^1 ···
𝑣AR 1
···
𝑙 1
···
EOS
BOG
̃𝑣DM 1
···
̃𝑠 1
···
̃𝑎 1
···
```
```
Attention Mask
```
```
AR causal self-attention
DM full attention
```
Figure 5: **Mixture-of-Transformers (MoT) architecture of Cosmos 3. Left:** a single transformer operates on
one token sequence comprising the autoregressive ( **AR** ) and diffusion ( **DM** ) subsequences: AR carries discrete
text tokens and, optionally, ViT-encoded vision tokens, ending with <EOS> and a begin-of-generation token
<BOG>, while DM carries continuous tokens from their respective encoders, noise-perturbed during training.
Here we visualize all input tokens as noisy for simplicity; for generation modes such as image-to-video or video
transfer, clean conditioning tokens precede the noisy targets within DM; see Sec. 2.2.2. Within each
transformer block, AR tokens and DM tokens are processed by independent LayerNorms and MLPs (all
co-initialized from a pre-trained VLM) and meet only at a shared self-attention operator. LetQ,K, andVbe
query, key, and value vectors in attention, where the subscript indicates which tower it is in. QARattends
causally over KAR,VARonly, while QDMattends bidirectionally over the concatenated [KAR;KDM] and
[VAR;VDM]. In this way, diffusion is conditioned on the AR context, while AR remains autoregressively
self-contained. Outputs are next-token predictions for Reasoner and denoised tokens for Generator (trained in
practice with a flow-matching objective predicting velocity; we show the clean target here for clarity). **Right:**
the attention mask, causal for AR and full for diffusion.

```
Cosmos 3 jointly models language, vision, audio, and action tokens within a unified attention framework,
designing a position-embedding scheme that generalizes consistently across modalities is inherently challenging.
Inspired by 3D Multimodal RoPE (MRoPE) (Bai et al., 2025a), we design a 3D MRoPE with absolute temporal
indexing to align video, audio, and action tokens along the same physical temporal axis. The original 3D
MRoPE divides the hidden dimension of each attention head into temporal, height, and width components,
where the temporal component records only the discrete token index. This design is sufficient for image and
video understanding tasks, but it is inadequate for our setting, where video, audio, and action tokens may be
generated simultaneously at different frame or sampling rates. In this case, tokens from different modalities
must be aligned to an absolute physical temporal axis. We first introduce the base formulation, which follows
the original 3D MRoPE design, and then describe our extensions and modifications, especially our absolute
temporal modulation, which aligns the absolute temporal axis.
```
#### 2.4.1 Position Index Allocation

```
Autoregressive tokens. For backward compatibility with language generation and image/video understanding
models, position indices for all language tokens and ViT-encoded media tokens in the AR subsequence follow
the original 3D MRoPE design. For language tokens,𝑡 = ℎ = 𝑤is set to the same monotonically increasing
value, reducing 3D MRoPE to standard 1D RoPE behavior. For tokens from the ViT encoder,𝑡is shared by all
tokens from the same frame, while theℎand𝑤indices vary independently according to the spatial location
of each token. The allocation of the position index in the autoregressive subsequence is identical to the 3D
MRoPE design in Qwen3-VL (Bai et al., 2025a).
```

```
Coordinate Assignment
```
```
Video
```
```
𝑡=7 𝑡=
packed Language Audio Action
token
sequence
modality offset 𝑘
coordinate
annotation
```
```
𝑡
ℎ
𝑤
```
```
0
0
0
```
```
1
1
1
```
```
2
2
2
```
```
3
3
3
```
```
4
4
4
```
```
7 7 7 7
0 0 1 1
0 1 0 1
```
```
8 8 8 8
0 0 1 1
0 1 0 1
```
```
7 8 9 10
0
0
```
```
0
0
```
```
0
0
```
```
0
0
```
```
7 8 9 10
0
0
```
```
0
0
```
```
0
0
```
```
0
0
(𝑡=ℎ=𝑤) (3D spatial) (temporal only) (temporal only)
```
```
FPS Modulation
```
```
24 FPS
(base)^01234
16 FPS 0 1.5 3 4.5 6
30 FPS 0 0.8 1.6 2.4 3.
frame index−→
Temporal position spacing reflects
real-world time, not token count.
```
Figure 6: **Illustrative coordinate assignment under 3D MRoPE. Left:** A packed token sequence containing
language, video (two frames, 2 × 2 spatial grid each), audio, and action tokens. Each token receives a(𝑡,ℎ,𝑤)
triplet. Language tokens use 𝑡 = ℎ = 𝑤; video tokens vary on all three axes; action and audio tokens use
temporal coordinates only (ℎ = 𝑤 = 0). A modality offset 𝑘 separates the text and vision temporal ranges.
**Right:** FPS modulation maps frame indices to scaled temporal positions so that equal real-world durations
occupy equal position ranges at 16, 24, and 30 FPS, where 24 FPS is our base frame-per-second.

**Diffusion tokens.** As illustrated in Fig. 6, video tokens vary across all three axes:𝑡advances with the temporal
latent frame index, whileℎand𝑤tile over the spatial grid(0...𝐻− 1 , 0 ...𝑊−1)independently per frame.
Image tokens are treated as single-frame videos and vary only in(ℎ,𝑤). Both spatial and temporal indices
are reset to zero at the start of each vision segment, so the model treats𝑡,ℎ, and𝑤as absolute within-video
coordinates rather than positions in the global sequence. For example, in the video transfer task where the user
provides a text prompt together with controlled video frames such as depth maps, both the clean control-video
tokens and the noisy generated-video tokens start from the temporal offset of the last token in the autoregressive
subsequence. All **audio tokens** and **action tokens** only carry temporal coordinates. The spatial indices are set
to zero (ℎ = 𝑤 = 0). For audio tokens, the temporal index advances with each audio hop; for action tokens,
the temporal index advances with each sampling step.

**Autoregressive and diffusion token margin.** In practice, we find that directly letting the diffusion tokens start
from the temporal offset of the last autoregressive token leads to over-saturation and checkerboard artifacts in
the initial video frames. This effect is especially pronounced in larger variants of Cosmos 3, such as the Super
model. We hypothesize that this occurs because the last language token and the vision tokens from the first
frame occupy adjacent temporal positions, resulting in nearly identical temporal embeddings. To address this
issue, inspired by Cao et al. (2025), we insert a fixed temporal gap between the autoregressive and diffusion
subsequences, uniformly shifting the temporal indices of all the subsequent vision, audio, and action tokens.
This creates a buffer in positional space that provides a clearer text-to-vision transition signal without requiring
architectural changes or additional learnable embeddings. In all of our models, we set the gap to be 15000.

#### 2.4.2 Absolute Temporal Modulation

A single unit step along the temporal dimension may correspond to different physical time intervals across
modalities or data sources. For example, when encoding videos at 60 FPS and 24 FPS, respectively, a temporal-
index increment for 24-FPS video tokens corresponds to a physical time interval that is 2.5 times longer than
that of 60-FPS video tokens. Similar discrepancies also arise for action and audio tokens, where different data
sources may use different sampling rates. FPS modulation is designed to align tokens with different temporal
resolutions onto a shared physical temporal axis by modulating the effective size of each temporal increment.

We first define the temporal steps per second (TPS) to characterize the physical temporal resolution. For video
tokens, TPS is given by the video frame rate divided by the temporal compression factor, which is 4 in our case
due to the video VAE encoder. For audio tokens, TPS is computed asTPSaudio=^480001920 ≈ 25 (48 kHz, 1920 hop
size). For action tokens, TPS is exactly the sampling frequency of the action data.


We then associate a unit length along the temporal dimension with a base TPS, denoted asTPSbase. For tokens
in a given diffusion subsequence, we compute their corresponding TPS. When the temporal index needs to be
increased by one unit step, the temporal increment 𝛿𝑡 with the modulation is computed as

```
𝛿𝑡 =
TPSbase
TPS
```
##### . (9)

Since video constitutes the majority of our training data, and 24 FPS is the most common frame rate in our
setting, we set TPSbase=^244 = 6 where 4 is our video tokenizer’s temporal compression ratio.

### 2.5 Model Variants

Cosmos 3 is trained at three model scales: **Edge** , **Nano** , and **Super** , spanning a wide range of computational
budgets from on-device deployment to large datacenter inference. **Edge** is a 4B-parameter model built upon a
dense 2B-parameter transformer, **Nano** is a 16B-parameter model built upon a dense 8B-parameter transformer,
and **Super** is a 64B-parameter model built upon a dense 32B-parameter transformer. All variants are initialized
from pre-trained vision-language models (VLMs) and adopt the Mixture-of-Transformers (MoT) architecture
described above. Tab. 2 summarizes the key architectural hyperparameters for each variant. Cosmos3-Nano
and Cosmos3-Super models are released in this paper. Cosmos3-Edge model will be included in a later release.

**Cosmos3-Edge** uses the design of a 2B dense transformer of 28 layers, 2048 hidden size, 16 attention heads, 8
key-value heads, a head dimension of 128 , and 9216 FFN dimension. We train the LLM from scratch using the
Megatron codebase. The design of the LLM largely follows the Qwen3-1.7B architecture, with two notable
differences: it removes QK normalization and uses ReLU-squared as the FFN activation, which is paired with
the Edge FFN dimension reported in Tab. 2.

**Cosmos3-Nano** adapts the Qwen3-VL 8B (Bai et al., 2025b) architecture, with 36 layers in the LLM, a hidden
size of 4096 , 32 attention heads, 8 key-value heads, a head dimension of 128 , and a FFN dimension of 12 , 288.

**Cosmos3-Super** adapts the Qwen3-VL 32B (Bai et al., 2025b) architecture, with 64 layers in the LLM, a hidden
size of 5120 , 64 attention heads, 8 key-value heads, a head dimension of 128 , and a FFN dimension of 25 , 600.

Table 2: **Cosmos 3 MoT model variants.** All models share the dual-tower MoT architecture. “LLM Layers”
refers to the number of transformer decoder layers; each layer carries independent parameter sets for the
reasoner and generator towers. **Edge** uses a dense 2B parameter transformer trained from scratch, while **Nano**
and **Super** are initialized from pre-trained Qwen3-VL weights.

```
Variant LLM Layers Hidden Dim Attn Heads KV Heads Head Dim FFN Dim
Cosmos3-Edge 28 2,048 16 8 128 9,
Cosmos3-Nano 36 4,096 32 8 128 12,
Cosmos3-Super 64 5,120 64 8 128 25,
```


---
*[... non-relevant sections omitted — see COSMOS_full.md.bak ...]*

#### 3.2.3 Action

Actions provide the causal variables that connect observed world states across time. While video-only training
teaches the generator to extrapolate likely motion, it does not expose the model to controllable interventions:
the same initial observation may evolve differently under different robot commands, camera trajectories, vehicle
routes, or human hand motions. We therefore introduce paired text-video-action data during mid-training
so that Cosmos 3 can learn both directions of the world-action relationship: predicting future observations
conditioned on actions, inferring the actions that explain an observed trajectory, and jointly generating actions
and future video.

**Data statistics.** We focus action mid-training on four physical-AI pillars: egocentric motion, robotics, au-
tonomous vehicles, and camera motion. The final curated data contains 8. 4 M episodes and 61. 3 K hours across


these pillars, as summarized in Fig. 9.

```
Action Data Distribution
```
**61.3K**
hours
Egocentric
Motion
**67.4%**

```
Autonomous
Vehicle
16.3%
```
```
Robotics
8.7%
```
```
Camera
Motion
7.5%
```
Figure 9: **Action data distribution.** Hours are
aggregated over the four main action-data pillars in
the final curated action mid-training set, which
contains 8. 4 M episodes and 61. 3 K hours.

```
Table 4: Robotics data breakdown. Grouped by
robot embodiment.
```
```
Embodiment Data source Tasks Episodes Hours
AgiBot Bu et al. (2025) 338 239.4K 4.37K
Franka Panda Wu et al. (2025a);
Khazatsky et al. (2024)
```
```
67.5K 76.3K 442
```
```
Google Robot Brohan et al. (2023b) 599 87.2K 351
WidowX-250 Walke et al. (2023) 21.8K 50.4K 100.1
UMI Lin et al. (2025a); Ha et al. (2024);
Liu et al. (2024e); Chi et al. (2024);
Liu et al. (2025a); Wu et al. (2024)
```
```
43 38.3K 67
```
```
UR Wu et al. (2025a) 114 25.0K 35
Total – 90.4K 516.7K 5.36K
```
- _Egocentric motion._ Egocentric motion data contributes 41. 3 K hours ( 67 .4%), making it the largest compo-
    nent. It comprises 1. 7 M episodes from a proprietary dataset of bimanual hand manipulation captured
    with a head-mounted RGB camera. Each frame is annotated with the synchronized head-camera pose
    and, for each hand, a 21-keypoint 3D pose (Zimmermann and Brox, 2017; Simon et al., 2017) that
    provides per-joint position and orientation in the camera coordinate frame, enabling the model to jointly
    learn egocentric ego-motion and fine-grained dexterous hand motion.
- _Autonomous vehicle._ Autonomous vehicle data contributes 10. 0 K hours ( 16 .3%), derived from high-quality,
    in-house driving logs collected using the NVIDIA Hyperion platform. The dataset is constructed by mining
    a large-scale corpus to match a target distribution spanning diverse driving scenarios. The selected
    scenarios cover a broad range of conditions, including diverse weather, lighting, and road conditions, as
    well as varied longitudinal and lateral maneuvers, rather than being limited to predominantly near-straight
    cruising. To align with other domains, we transform driving trajectories from the vehicle coordinate
    frame to the front-wide camera coordinate frame.
- _Robotics._ Robotics data contributes 5. 4 K hours ( 8 .7%), aggregated from open-source datasets. The subset
    contains 90. 4 K tasks and 516. 7 K episodes, as broken down by embodiment and source in Tab. 4. To avoid
    embodiment-specific controller details such as PID parameters or low-level actuation interfaces, we use
    pseudo-actions derived from state differences. We curate data from both successful and failed episodes
    so the model observes not only intended completions but also off-nominal action effects.
- _Camera motion._ Camera motion data contributes 4. 6 K hours ( 7 .5%), mined from our pre-training video
    dataset. We convert these videos into action trajectories by estimating camera poses with ViPE (Huang
    et al., 2025) and DepthAnything3 (Lin et al., 2025b). To ensure data quality, we rigorously filter the
    dataset to remove clips with unreliable pose estimation, such as those exhibiting excessive jitter or
    abnormal camera intrinsics. All camera poses are kept in metric scale and converted to the unified action
    coordinate convention. This curation process yields a dataset of 1. 9 M clips.

**Data processing pipeline.** We convert each source using the unified action tokenization described in Sec. 2.1.3.
To balance action magnitudes across embodiments after this conversion, we compute per-dimension normalizers
from the training data and scale action channels to a comparable range of roughly[− 1 , 1]. For data with
multiple synchronized viewpoints, we concatenate the views into a canvas and store the camera layout in
metadata, as shown in Fig. 29. Rather than filtering out idle operations, we retain them and record the idle-step
count in metadata, allowing downstream sampling to explicitly balance active and inactive segments.




---
*[... non-relevant sections omitted — see COSMOS_full.md.bak ...]*

### 4.2 Generator Training

The Cosmos 3 Generator is trained using a progressive multimodal curriculum designed to jointly model
visual, auditory, and action-conditioned world dynamics across diverse resolutions, durations, and conditioning
modalities. The training recipe emphasizes scalability, high-fidelity generation, and efficient long-context
learning. During pre-training, the model learns general generative priors from large-scale data spanning images,
videos, and audio. Subsequent training stages progressively introduce richer multimodal supervision, including
actions and transfer sequences, enabling the model to learn temporally coherent world evolution and physically
grounded interactions.

**Training objective.** The Cosmos 3 generator is optimized under a rectified flow matching objective across all
modalities. For a target latent from any modality, we construct a noisy latent via the straight-line interpolation
𝑥𝜎= 𝜎· 𝜖 + (1− 𝜎)· 𝑥 0 , where𝑥 0 is the clean target,𝜖 ∼ 𝒩 (0,𝐼), and𝜎 ∈ [0, 1]is the noise level. A single
denoiser𝑣𝜃(𝑥𝜎,𝜎,𝑐)is trained to predict the constant velocity𝑣*= 𝜖−𝑥 0 via masked mean-squared error, where
conditioning tokens ( _e.g_ ., clean conditional frames in image-to-video tasks) are gated out of the loss. We apply
per-modality time sampling, drawing noise level𝜎independently for each modality (images, videos, audio,
and action). Following Waver (Zhang et al., 2025c), we use logit-normal noise distribution for image, audio,
and action batches and mode sampling for video batches. We found that using mode sampling yields better
generation quality. We further map𝑡through a rectified-flow shift reparameterization𝜎 = 𝑠· ̄𝑡/(1 + (𝑠− 1)· ̄𝑡)
with ̄𝑡 = 1− 𝑡, where 𝑠≥ 1 biases the marginal toward higher noise.

#### 4.2.1 Pre-Training

During the pre-training stage, we jointly train the model to generate images, videos, and audio across diverse
resolutions and generation tasks. To support this, we employ a multi-resolution training strategy and optimize
the model jointly over multiple generation tasks, including Text-to-Image, Text-to-(Video+Audio), Image-to-
(Video+Audio), and Video-to-(Video+Audio).

**Multi-resolution training.** Rather than committing to a single output resolution, we train simultaneously
across three resolution tiers (256p, 480p, 720p), five aspect ratios and variable number of frames, as shown
in Tab. 5. This exposes the model to high-fidelity content while encouraging resolution-agnostic representations.
The training data is partitioned accordingly: the 256p stream draws from the full dataset (all native resolutions
are eligible), the 480p stream is restricted to source material with native resolution at or above 480p, and the
720p stream uses only content at or above 720p, preserving sharpness and fine detail at the highest tier. Each
resolution tier imposes a different maximum frame budget: up to 400 frames at 256p and 480p, and 300 frames
at 720p. We restrict 720p to 300 frames due to the sequence length constraints. Training batches are composed
across the four tiers using a 1:1:2:1 ratio for image-only, video-256p, video-480p, and video-720p samples,
respectively. We find that this distribution provides a strong balance between high-fidelity learning and sample
diversity, enabling the model to observe more training examples while still emphasizing higher-resolution
content. We use resolution-adaptive shift values: 𝑠 = 1 at 256p, 𝑠 = 3 at 480p, and 𝑠 = 5 at 720p.

To prevent gratuitous recompilation overhead while supporting variable sequence lengths, we use token packing


Table 5: **Image/Video Model Specifications.** Supported configurations for image and video modalities. Each
row shows the FPS range, frame counts (video only), and image/video dimensions (w, h) for the five
supported aspect ratios at each resolution.

```
Video Dimensions (w, h) by aspect ratio for images/videos
Resolution FPS # frames 16:9 4:3 1:1 3:4 9:16
256p 10–30 5–400 (320, 192) (320, 256) (256, 256) (256, 320) (192, 320)
480p 10–30 5–400 (832, 480) (736, 544) (640, 640) (544, 736) (480, 832)
720p 10–30 5–300 (1280, 720) (1104, 832) (960, 960) (832, 1104) (720, 1280)
```
```
Multi-Resolution Training Tiers
```
```
256p
Max 400 frames
Noise shift𝑠= 1
All source resolutions
```
```
480p
Max 400 frames
Noise shift𝑠= 3
Native res.≥ 480 p only
```
```
720p
Max 300 frames
Noise shift𝑠= 5
Native res.≥ 720 p only
```
```
Sequence Packing (fixed 74,000-token context)
r1 256p 256p 256p 256p
r2 480p 256p
r3 480p 480p
r4 720p
74,000 tokens (fixed context window)
256p sequence 480p sequence 720p sequence
```
```
Image–video pre-training data mixture
by resolution
```
```
video / image 80/20
```
```
Video-720 20%
```
```
Video-480 40% Video-256
20%
```
```
Image-256 6.66%
Image-480 6.66%
Image-720 6.66%
```
```
Video modes: T2V 70%, I2V 20%, V2V 10%
```
Figure 10: **Left: Multi-resolution training and sequence packing.** The three resolution tiers (256p, 480p,
720p) differ in their maximum frame budget, eligible source material, and rectified-flow noise-shift value;
variable-length sequences from different tiers are packed together to fill a fixed 74,000-token context window,
maximizing GPU utilization without padding. **Right: Data mixture used in generator pre-training.** We use
joint image-video training, with videos sampled80%of the time and images the remaining20%. Within each
split, we train at multiple resolutions: 256p, 480p, and 720p. For video batches, we additionally sample
uniformly among three conditioning modes—text-to-video, image-to-video, and video-to-video. The exact data
mixture is shown in the right panel.

with a fixed budget of 74,000 tokens per sequence. Sequences at various resolutions are packed together to fill
each batch, maximizing GPU utilization without padding (depicted in Fig. 10).

**Training modes.** For a latent video tensor of shape𝐶×𝑇×𝐻×𝑊, let𝑇conddenote the number of conditional
latent frames and𝑇noisedthe number of noisy latent frames (𝑇 = 𝑇cond+ 𝑇noised). During training, no noise is
applied to the first𝑇condframes, which serve as conditional inputs; only the remaining𝑇noisedframes are noised,
and the model learns to denoise them. Different choices of𝑇condand𝑇noisedyield different training modes. We
use four generation modes—Text-to-Image, Text-to-Video, Image-to-Video, and Video-to-Video—distinguished
solely by the number of conditioning visual frames prepended to each sample, with sampling ratios of20%,
56%, 16%, and 8%, respectively. All modes use the structured JSON caption format described in Sec. 3.2.

- _Text-to-Image (T2I)._ Images are treated as a special case of videos with the temporal dimension restricted
    to𝑇 = 1. In this mode, images are drawn randomly from all three resolution tiers and aspect ratios, then
    sequence-packed before being sent to the model. Since an image sample yields far fewer tokens than a
    video, a typical sequence contains many more samples than its video counterpart.
- _Text-to-Video (T2V)._ In text-to-video training,𝑇cond= 0. The model learns to denoise the entire video
    conditioned solely on text. Alongside the caption, the model receives duration, FPS, and timestamp
    metadata as additional fields in the JSON caption, enabling it to generate videos of specified length and
    temporal extent.


- _Image-to-Video (I2V)._ For single-frame conditioning (𝑇cond= 1), the first latent frame is held clean while
    subsequent frames are noised. The model learns to generate future frames consistent with both the initial
    frame and the caption.
- _Video-to-Video (V2V)._ For multi-frame conditioning (𝑇cond= 2), the model is conditioned on the first five
    frames of a video (equivalently, the first two latent frames) and learns to predict future frames consistent
    with both the conditioning frames and the input prompt.

**FPS modulation.** We train the model with varying FPS values, so the physical temporal spacing between
tokens differs across samples: a clip sampled at 30 FPS packs frames more densely in real time than the same
number of tokens sampled at 16 FPS. To reflect this, we modulate the temporal axis of 3D MRoPE position
encodings by assigning temporal coordinates in proportion to real-world time rather than token index (see
Sec. 2), with a base rate of 24 FPS. Duration and FPS are also appended to the text prompt, allowing the model
to be conditioned on specific temporal characteristics at inference time.

**Optimization.** Only the generation-specific parameters are updated during the generator pre-training. The
reasoner tower remains frozen, preserving the language and visual understanding capabilities. We use
FusedAdamW with learning rate 10 −^4 ,(𝛽 1 ,𝛽 2 ) = (0. 9 , 0 .99), weight decay 0. 05 , and gradient clipping at
norm 1.0. The learning rate schedule follows a linear decay with warmup, from the peak lr to a floor of 0. 30 ×
over 𝑛 iterations. To enable classifier-free guidance, we use a text-dropout rate of 10% across all modalities.

**Tokens trained.** In the pre-training stage, Cosmos3-Nano was trained on 31. 05 T tokens using 1024 NVIDIA
GB200 GPUs, while Cosmos3-Super was trained on 17. 86 T tokens using 2048 NVIDIA GB200 GPUs.

#### 4.2.2 Mid-Training

Mid-training bridges the gap between broad pre-training and downstream deployment. At this point, the
Generator has already learned general image, video, and audio generation from large-scale data, but the target
Physical AI applications require stronger coverage of rare dynamics, embodied scenes, control interfaces, and
high-quality visual domains. We therefore continue training from the pre-trained checkpoint with a curated
mixture that both preserves the original visual generation modes and introduces new sources of supervision.
The stage has two complementary objectives: _domain specialization_ , which increases exposure to high-value
Physical AI domains, and _multimodal integration_ , which extends the model from visual and audio generation to
action- and control-conditioned world modeling.

**Domain specialization.** While retaining its general knowledge, the model is exposed to highly curated
specialized datasets to improve quality and reliability in application-critical Physical AI scenarios. For images,
we use a 15.6M-sample mid-training pool that emphasizes high-quality real imagery while adding synthetic
and text-rendering data to broaden concept coverage and preserve legible text generation. For videos, we
incorporate 74.7M curated clips spanning robotics, autonomous driving, human activity, physics, and synthetic
simulation data. These sources target failure modes that are underrepresented in generic web-scale pre-training,
such as long-horizon interactions, fine-grained human and robot motion, physical object dynamics, and safety-
critical driving or warehouse scenarios. By mixing these domain-focused datasets with the existing image and
video training modes, mid-training improves Physical AI relevance without discarding the broad visual priors
learned during pre-training, as described in Sec. 3.2.1.

**Multimodal integration.** Mid-training expands the Generator from image, video, and audio generation into
a unified Physical AI model that can also consume and synthesize action and control signals. We keep the
same clean-prefix/noisy-target formulation used in pre-training for T2I, T2V, I2V, and V2V, so existing visual
capabilities remain active while new modality-specific tokens are introduced in the diffusion subsequence.
This lets action, audio, control, and video tokens share the same temporal coordinate system and two-way


Table 6: **Generator mid-training data mixture.** After pre-training is done, we introduce new modalities
(action and transfer) in the mid-training stage with the data ratios listed below.

```
Training stream Modes / Conditioning Share
Image T2I 10%
Video T2V, I2V, V2V 32%
Video + Audio T2(V+Audio), I2(V+Audio), V2(V+Audio) 8%
Action Forward dynamics, inverse dynamics, policy 25%
General Transfer Edge, blur, depth, and segmentation controls 20%
Driving Transfer World-scenario-map controls 5%
```
attention pattern described in Sec. 2. In addition to the pre-training modes, we add two additional families of
multimodal supervision: action and video transfer.

- _Action._ We introduce paired text-video-action training data using the unified action representation
    in Sec. 2.1.3. The model is trained not only to predict future video conditioned on actions, but also to
    infer actions from observed trajectories and to jointly generate actions and visual futures. This teaches
    the Generator a causal interface between controllable interventions and world evolution.
- _Video transfer._ We add control-conditioned transfer data in which clean control signals are provided as
    inputs and the model denoises the corresponding target image or video. The control signals include
    edge, blur, depth, and segmentation maps from high-quality video corpora, as well as world-scenario
    maps for driving scenes. This exposes the model to spatially grounded constraints while retaining text
    conditioning and visual generation quality.

The mixing ratios of different modalities are shown in Tab. 6.

**Multi-resolution training.** Similar to pre-training, mid-training uses multi-resolution across 256p, 480p, and
720p within a fixed 74K context window. To better handle dynamics and reduce temporal and high-resolution
artifacts, we increase rectified-flow shift values to 3 , 5 , and 10 for 256p, 480p, and 720p, respectively.

**Training objective.** Similar to pre-training, we use the rectified flow objective for all modalities. For action,
we inherit the vision noise schedule. The total loss in mid-training is the sum of per-modality velocity MSEs
weighted by modality-specific loss scales, with action losses scaled by 10 ×to compensate for the smaller
per-element MSE of normalized action vectors.

**Optimization.** Similar to pre-training, we use FusedAdamW with learning rate 10 −^4 , weight decay 0. 05 ,
gradient clipping at norm 1.0, and loss scale 10. The learning rate follows a LambdaLinear schedule with start
factor 0. 4 and cycle length 100 , 000.

**Tokens trained.** In the mid-training stage, Cosmos3-Nano model was trained on 2. 4 T tokens using 1024 NVIDIA
GB200 GPUs, while Cosmos3-Super model was trained on 1. 9 T tokens using 2048 NVIDIA GB200 GPUs.

#### 4.2.3 Text-to-Image Post-Training

To demonstrate the omnimodal capability of Cosmos3-Super, we further specialize the model into a text-to-
image checkpoint, Cosmos3-Super-Text2Image. Our goal is to transfer the model’s physically grounded world
understanding to high-quality image generation, aiming for strong open-source T2I results while improving
physical plausibility and scene-level alignment.

We perform text-to-image specialization using a two-stage SFT, following the common text-to-image foundation-
model training paradigm that emphasizes semantic enhancement before preference-oriented refinement.

- _Stage 1: broad T2I specialization._ We fine-tune the model for 20k training steps on the curated high-quality


```
SFT dataset. The training mixture is sampled with a controlled ratio of45%general real image data,
40%synthetic image data, and15%text-rendering-only data, balancing visual fidelity, caption alignment,
and language retention. We use a base learning rate of 1 × 10 −^4 , 2k warmup iterations, and a linear
learning-rate decay schedule, while keeping all other hyperparameters consistent with the Cosmos 3
mid-training stage.
```
- _Stage 2: high-quality refinement._ We perform a final 2k-step SFT pass using 470 k carefully curated
    ultra-high-quality image–caption pairs. This stage further improves visual aesthetics, prompt-following,
    text-rendering quality, and alignment with human preferences.
- _Resolution and context length._ For both stages, we use a fixed context window of 70k tokens and train
    only on images with a resolution higher than 720p.

Overall, Cosmos3-Super-Text2Image delivers strong text-to-image results across both semantic alignment and
English text-rendering benchmarks. On UniGenBench, it achieves the best overall score among the evaluated
models, reaching 91. 36 on the full benchmark (see Tab. 11). With an agentic workflow, the model ranked top-1
among open-weight models on the Artificial Analysis Text-to-Image leaderboard (Sec. 6.2.1). These results
suggest that downstream T2I modality adaptation from Cosmos 3 is highly effective: it improves scene-level
prompt alignment while preserving the model’s physically grounded generation capability.

#### 4.2.4 Image-to-Video Post-Training

Image-to-Video capability is fundamentally important for comprehensive visual understanding. It probes the
model’s understanding of physical laws, object permanence, and intricate scene geometry, while also serving as
a critical predictive mechanism for embodied AI and robot planning, where simulating plausible future frames
yields an effective world model (Wiedemer et al., 2025; Chen et al., 2025a). While Cosmos 3 is inherently
designed to handle a diverse array of tasks natively, we utilize SFT to explicitly showcase and specialize its
potential in the I2V domain. To demonstrate these capabilities, we employ the following procedure:

- _Data and training mixture._ We fine-tune the model using filtered pre-training data that have been refined
    for a more balanced topic diversity, augmented via an agentic workflow that identifies model weak
    spots to retrieve targeted examples from the pre-training set. This is combined with 1,000 high-quality
    manually curated videos and a dataset of approximately 20k synthetic video clips spanning diverse topics
(accounting for roughly 6% of the total tokens). While all video sequences are trained exclusively using
the I2V formulation, our training mixture also incorporates 20% T2I image tokens to preserve the model’s
semantic alignment.
- _Resolution and duration._ We specialize the model for temporal generation at a targeted resolution of 480p
    and targeted duration of 189 frames, corresponding to roughly 8 seconds at 24fps. This configuration
    balances inference speed with temporal context, enabling fast, physically plausible video generation over
    a meaningful time horizon.
- _Training schedule._ The I2V post-training stage runs for a duration of 10k iterations at a learning rate of
    1 × 10 −^5. The model processes roughly 50B tokens over the course of SFT.

Through post-training, Cosmos3-Super-Image2Video achieves leading quality in image-to-video generation.
In particular, the model ranked top-1 among open-weight models on the Artificial Analysis Image-to-Video
leaderboard (Sec. 6.2.2). For details on the usage of this model, please refer to Sec. 6.3.1.

#### 4.2.5 Robot Policy Post-Training

We conduct robot policy post-training to investigate whether our Cosmos 3 omnimodal world models can be
extended into powerful robot policy models. Mid-training enables Cosmos 3 to model multimodal sequences,
including language, visual observations, and actions, and to generate actions jointly with videos. We further
customize it for robot policy learning by incorporating proprioceptive signals, reducing inference latency, and
adapting the model to produce executable actions for closed-loop control.


As a pilot study, we use the DROID robot platform and dataset (Khazatsky et al., 2024) due to its popularity
and broad community adoption. The DROID platform uses a Franka Panda 7-DoF manipulator with a Robotiq
2F-85 parallel-jaw gripper to perform tabletop manipulation tasks in diverse real-world environments. The
DROID dataset comprises 76k trajectories, 350 hours of interaction data, 86 tasks, and 564 scenes, providing
substantial scale and broad task diversity for real-world robot policy learning. We ingest DROID at a high
resolution of 360×640, apply community-provided idle-frame filtering and failure-demonstration removal, and
use random image augmentation during training.

We post-train Cosmos3-Nano-Policy-DROID by resuming from our mid-trained Cosmos3-Nano model, with
a freshly initialized action encoder, action-decoding MLP, and action embedding tokens. We apply a 5×
learning-rate multiplier to the action-related parameters to facilitate faster adaptation. The policy input consists
of the current proprioceptive robot state and a three-view visual observation. Specifically, the wrist-view image,
with a raw resolution of 360×640, is placed above two external-view images, each with a raw resolution of
180 ×320, which are concatenated side by side on the bottom left and bottom right. The resulting canvas is
540 ×640. The policy is trained to predict 32 future absolute joint-position actions, along with auxiliary RGB
video frames as additional outputs, operating at 15Hz. We use the official DROID short task instructions as
the prompts during this post-training study. We use a learning rate of 2 × 10 −^4 with other hyperparameters
following the mid-training setup.

At inference time, we sample the model using 4 diffusion steps with a shifted noise schedule of 5. We also
apply classifier-free guidance with CFG parallelism at a guidance scale of 3, and skip video-latent decoding
to further reduce inference overhead. Together, these optimizations provide a significant inference speedup,
enabling policy server deployment on 2 NVIDIA RTX Pro 6000 GPUs. The downstream joint-position controller
is implemented using Franky (Schneider, 2023) and executes the predicted 32 actions at 15Hz.

Overall, Cosmos3-Nano-Policy-DROID achieves strong results in robotic policy tasks. As detailed in Sec. 6.2.5,
it ranks first on both RoboLab (Yang et al., 2026a) and RoboArena (Atreya et al., 2025), demonstrating the
effectiveness of Cosmos3 as a foundation model backbone for robot policy learning.



---
*[... non-relevant sections omitted — see COSMOS_full.md.bak ...]*

#### 6.3.1 Generation Guide

The Cosmos 3 Generator supports flexible visual (and audio-visual) generation across a broad inference envelope:
frame rates from 10–30 FPS, 5 to 400 frames, resolutions spanning 256p, 480p, and 720p, and common aspect
ratios (1:1, 3:4, 4:3, 9:16, 16:9). This allows a single model to serve use cases ranging from short preview
clips to longer, higher-resolution landscape or portrait video without changing the sampling interface. The
Cosmos 3 Generator is trained for forward dynamics, inverse dynamics and policy modes to support different
action-related applications. For action generation, the base Cosmos3-Nano and Cosmos3-Super models support
action prediction at native control frequencies ranging from 10–30 FPS, with prediction horizons spanning
16–400 frames across inverse dynamics and policy modes. Post-trained models specialize to a single mode and
frequency—for example, Cosmos3-Nano-Policy-DROID operates at 15 FPS with a 32-step prediction horizon.
Below, we detail the key components for successful generation using Cosmos 3 Generator. These details are
also summarized in Tab. 21.


**Media specifications and prompting guide.** The Cosmos 3 Generator is trained on structured JSON captions
that provide fine-grained control over scene composition, covering subjects, background, lighting, aesthetics,
cinematography, and for video, temporal fields such as actions, state changes, camera motion, and segment-
level descriptions (full schema in Appendix A). At inference time, a prompt upsampler—served either by
Claude Opus 4.6 or by the Cosmos 3 Reasoner—converts user requests into this same structured format,
ensuring that generation prompts match the distribution seen during training (see Appendix B.1 for template
instructions). The upsampler is instructed to first describe the scene layout and world state, then specify the
temporal progression of events, and finally add any audio descriptions. The specific upsampler instruction
varies slightly across generation modes—for instance, action and transfer generation impose additional task
constraints tailored to their conditioning inputs (see Appendix B.4 for the transfer generation prompt prefix,
and Appendix B.5 for the action prompt guide). The JSON specification additionally includes explicit media
controls (duration, FPS, spatial height and width, and aspect ratio), keeping prompt interpretation and sampling
configuration inspectable and reproducible.

**Negative prompt.** We tune negative prompts separately for each model and generation mode through auto-
mated benchmark iteration. For each configuration, we ablate over candidate templates spanning natural-
language descriptions, keyword lists, instruction-style directives, compositional extensions targeting physical
consistency and identity preservation, and the null string. The best-performing variant is selected based on
automated benchmark scores. For the base Cosmos3-Nano and Cosmos3-Super generators, the explicit negative
prompt can be found in Appendix B.6. For the post-trained variants, we found that the null string negative
prompt works best for Cosmos3-Super-Text2Image, while using negative prompts automatically derived from
the user prompts yields the best result for Cosmos3-Super-Image2Video (Appendix B.3). For action generation
modes, we found the null string negative prompt to work best.

**Generation sampling hyperparameters.** We adopt the following sampling parameters for different modalities:

1. _Audio-visual generation._ For audio-visual generation, we tune sampling hyperparameters on automated
    benchmarks for image and video generation (Sec. 6.2.1 and Sec. 6.2.2). For the base Cosmos3-Nano
    and Cosmos3-Super generators, we use 50 denoising steps, a guidance scale of 6, a time shift of 10, and
    full-range classifier-free guidance. For the post-trained Cosmos3-Super-Text2Image model, we use a
    guidance scale of 4 and a time shift of 3. For the post-trained Cosmos3-Super-Image2Video model, we
    use a shift of 5.
2. _Action generation._ Action generation has three supported modes. For forward and inverse dynamics, we
    use 50 denoising steps, a guidance scale of 1, a time shift of 5, and full-range classifier-free guidance. For

Table 21: **Default sampling configurations and negative prompts for each generator and generation
modality.** We summarize the generation settings for different Cosmos 3 Generator modes. Negative prompts
are provided in full for each setting in the Appendix; “Null” indicates that the null string was the
best-performing variant.

```
Generation Modality Sampling Hyperparameters Negative Prompt
Cosmos3-Nano Audio-Visual steps=50, guidance=6, shift=10, full-range CFG Appendix B.6
Cosmos3-Super Audio-Visual steps=50, guidance=6, shift=10, full-range CFG Appendix B.6
Cosmos3-Super-Text2Image Visual steps=50, guidance=4, shift=3, full-range CFG Null
Cosmos3-Super-Image2Video Visual steps=50, guidance=6, shift=5, full-range CFG Appendix B.3
Cosmos3-Nano Forward/Inverse Dynamics steps=50, guidance=1, shift=5, full-range CFG Null
Cosmos3-Super Forward/Inverse Dynamics steps=50, guidance=1, shift=5, full-range CFG Null
Cosmos3-Nano-Policy-DROID Policy steps=4, guidance=3, shift=5, full-range CFG Null
Cosmos3-Nano Transfer steps=50, guidance=3, control guidance=1.5, shift=10 Appendix B.6
Cosmos3-Super Transfer steps=50, guidance=3, control guidance=1.5, shift=10 Appendix B.6
```

```
policy mode, we switch to 4 denoising steps and a guidance scale of 3.
```
3. _Transfer generation._ We tune sampling hyperparameters for video transfer generation on the automated
    PAIBench-C benchmark (Sec. 6.2.4). We use 50 denoising steps, a text guidance scale of 3, a control
    guidance scale of 1.5, a time shift of 10, and full-range classifier-free guidance.

