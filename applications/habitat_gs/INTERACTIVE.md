# Interactive text/image/video/camera rollout

Open the browser, inspect the simulator image, type a prompt, and click
**Generate next step**. The application runs the existing evaluated I2V and
inverse-dynamics pipeline, renders every estimated camera pose live, and waits
for your next prompt at the final pose. The generated clip and simulator camera
replay are shown separately, alongside prompt history and boundary comparisons.

This is a local GPU application, not a public hosted website. Rendering happens
on the allocated GPU; the browser displays streamed snapshots and MP4s. Rendering
and inference never run on the login node. An optional login relay forwards bytes
only. Camera replay uses all97 estimated poses and the same measured Nymeria
fisheye calibration; no collision corrections, grounding, or trajectory fitting.

## Running instance on Yonsei (2026-10-30)

- Allocation2608785, node33; GPU0 generates, GPU1 renders the live inspector.
- Session Scene63 Apartment; server8765; tmux
  `app_habitat_gs_smoke_20261030`, window `camera-studio`.
- Loopback-only relay on yonsei-ai-login6:8765; window `web-relay`.
- Data: `/scratch2/whwjdqls99/cosmos/applications/habitat_gs/interactive/scene63`.
- The allocation has a finite time limit. The session survives on disk; after
  obtaining another GPU allocation, relaunch with the same configuration and
  session path and update the relay's node. Do not assume the old allocation
  remains available.

### Browser access

From VS Code connected to yonsei-ai-login6, open **Ports**, forward port8765,
and keep its local port8765. Alternatively, on your own computer:

```bash
ssh -N -L 8765:127.0.0.1:8765 YOUR_YONSEI_SSH_ALIAS
```

In the Yonsei terminal, print the private browser URL, then open it locally:

```bash
cat /scratch2/whwjdqls99/cosmos/applications/habitat_gs/interactive/scene63/browser_url.txt
```

The one-time URL sets an HttpOnly/SameSite cookie; subsequent requests use that
cookie. The token is private to this session. Do not publish it in Notion or git.
If using another local port, replace8765 in the URL. Direct SSH to node33 was
unavailable; the existing login relay avoids requiring compute-node SSH access.

## Launch a new interactive session

Within an existing owned GPU allocation (Yonsei), or the equivalent Grasp GPU
shell, use the same site configuration as `run.sh`. Example:

```bash
export APP_STORAGE=/scratch2/whwjdqls99/cosmos/applications/habitat_gs
export CUDA_VISIBLE_DEVICES=0
export APP_INSPECT_GPU=1 # Only when this second GPU is allocated to you.
bash applications/habitat_gs/serve.sh \
  --config "$APP_STORAGE/rollout_scene63.json" \
  --session "$APP_STORAGE/interactive/my_session" \
  --host 0.0.0.0 --port 8765
```

For Yonsei, put that command inside `srun --jobid=YOUR_JOB --overlap --exact
-N1 -n1 -c4 --input=none ...` in tmux. Keep the allocation after the app stops.
`--host 0.0.0.0` requires the generated session token; the default loopback host
is suitable when direct SSH port forwarding to the compute node is available.

On the login host, if needed:

```bash
python3 applications/habitat_gs/tunnel_relay.py --node YOUR_GPU_NODE --port 8765
```

Only the relay runs on login. Choose another port when8765 is already occupied.

## Inference parity and persistence

Each submitted prompt produces a distinct one-segment configuration and invokes
`applications.habitat_gs.rollout`; there is no second sampler implementation.
The checkpoint, exact native prompt wrapper, EMA, I2V UniPC35/CFG6/shift10,
inverse UniPC30/CFG1/shift10,97frames/20fps/256, and measured camera rays remain
unchanged. Only the action prompt, starting pose, and per-step seed vary. Seed
is base_seed + number of accepted steps. Failed attempts do not advance the
accepted pose or seed, and concurrent submissions are rejected.

The server persists state.json, session_config.json, per-attempt configurations,
all native predictions, camera poses, live rendering progress, and replay MP4s.
On restart it restores the last accepted pose/image and history. Changed input
scene/calibration/profile/configuration requires a new session directory. The
session takes a filesystem lock; two servers cannot write the same session.

The first end-to-end API test passed in110seconds, including generated video,
inverse dynamics, all97 live renders, and exact equality between the replay's
last frame and the next conditioning render. API checks covered concurrent
requests, path traversal, cross-origin requests, and HTTP range playback. The
login relay's authenticated access and unauthenticated401 response passed.
This validates the workflow and video delivery, not scene-consistent generation
or accurate camera recovery. Subsequent camera-studio browser checks are
described below.

The two earlier prompt-labeled rollout videos were uploaded to the existing
Notion application page and downloaded again with matching SHA256 hashes.
Evidence is in `video_publication.json`, `interactive_validation.json`, and
`web_access_validation.json` under APP_STORAGE.

A second full iteration after server restart also passed in80.1seconds. It uses
seed1, starts from exactly the prior accepted pose, renders97 camera frames,
and leaves the server ready with both test prompts in history. Evidence:
`interactive_continuation_validation.json`. For direct login-host forwarding
without relying on the optional relay, use
`ssh -N -L 8765:node33:8765 YOUR_YONSEI_SSH_ALIAS`, then the same private local URL.

## Camera studio: free 3D view and selectable starting poses

The browser now includes a persistent perspective inspection renderer in the
same 3DGS world. This camera is independent of the Nymeria model camera:

- Drag the scene to orbit; right-drag to pan; scroll to zoom.
- Click the scene and use W/A/S/D to fly, Q/E down/up, Shift for faster motion.
- **Use this view as candidate camera** copies the inspection pose to the
  candidate. Alternatively edit XYZ/yaw/pitch/roll directly. The side panel
  renders an overlay-free Nymeria fisheye preview of the candidate.
- **Apply starting pose** commits that camera and renders its conditioning
  image. Generation is disabled for unapplied candidate changes. Applying a
  pose while an iteration is running is rejected.
- Generated trajectories are shown in scene coordinates. Cyan paths connect
  consecutive camera positions; green/red mark each segment's start/end.
  The timeline moves a magenta camera through all97 poses. The amber camera
  marks the candidate. Glyph boundaries use the measured fisheye ray domain,
  rather than pretending the model camera has a perspective pyramidal frustum.
- **Frame trajectory**, **Look through candidate**, and **Use selected pose**
  provide alternative views and starting points. Selecting/playing a trajectory
  does not change the accepted model camera until you explicitly apply a pose.

All pose edits have separate `pose_edits/*/edit.json` receipts with previous and
new poses and conditioning hashes. Session restart restores the accepted pose,
including explicit manual edits. Historical generated trajectories are retained
as separate segments; manual teleports are never joined by invented trajectory
lines. Coordinates are Habitat Y-up, camera poses are upright RGB optical CV,
and calibrated conditioning remains exactly as in the evaluated pipeline.

`APP_INSPECT_GPU=1` assigns the live inspector to GPU1 while generation uses
GPU0. This is used for the current two-4090 allocation; omit it to share the
selected GPU on single-GPU servers (memory/performance must then be checked).
The inspector stays loaded for navigation and closes with the server. It uses
approximately1GiB VRAM in the current scene. A local15-frame probe measured
38.0ms median,38.3ms p95 per768×512 inspection frame with cached candidate
preview. Browser refresh/tunnel latency limits delivered frame rate; this is
not a promise of26fps in the browser. Changing the candidate also requires
six cubemap views for the fisheye preview.

Overlay checks: native depth-tested inspection lines visibly change the
inspection image but not the conditioning preview. The preview is pixel-exact
with a separately rendered native conditioning image. Habitat draws hidden
lines dimly for inspection. No debug geometry enters generation or inverse
inference. Invalid camera rotations and concurrent pose-change/generation
requests are rejected. Evidence: inspector_validation.json and
pose_api_validation.json under APP_STORAGE.

The upgrade was tested in a separate copied session, then deployed to the
original port8765 with its private token, accepted camera, and two history
entries preserved. Refresh the existing browser page; the tunnel is unchanged.
An actual Chromium browser verified orbit/fly navigation, live image changes,
the unapplied-pose generation guard, timeline playback, and session restoration.
A full text/image/video/inverse/replay iteration from a manually committed pose
completed in the test session. The initial browser wait timed out after240s
during slow model loading; completion was subsequently verified against the
saved start-pose receipt. This timeout is retained in the validation report.

Frame request identifiers use JavaScript-safe millisecond integers; nanosecond
identifiers were found to round to repeated browser values and were corrected.
Evidence: `browser_studio_validation.json`, `studio_deployment_validation.json`,
`camera_studio_live.png`, and `state_before_studio_upgrade.json` under APP_STORAGE.
Inspection is server-rendered and streamed as JPEG snapshots, not a locally
downloaded scene. Delivered navigation speed depends on the SSH connection.
