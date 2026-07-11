"""Build a FD montage video: rows = test sequences, columns = GT | zero-shot | finetuned models.
Each cell is the forward-dynamics generated video (GT clip for col 0). Column headers on the top row."""
import glob, os, subprocess

R = "/weka/jungbin/cosmos_motion_ft_runs/cosmos3_camera/camera_world"
GT_ROOT = "/weka/jungbin/cosmos_motion_ft_runs/nymeria_eval5/samples"
# (column label, path template with {seq})
SOURCES = [
    ("GT", GT_ROOT + "/{seq}/gt_clip.mp4"),
    ("zero-shot", "/weka/jungbin/cosmos_motion_ft_runs/zeroshot_eval/fdpolicy5/fd_out/{seq}/vision.mp4"),
    ("97f_b8 i13500", R + "/world_camera_nymeria_97f_b8/checkpoints/iter_000013500/fdpolicy5/fd_out/{seq}/vision.mp4"),
    ("97f-hung i7000", R + "/world_camera_nymeria_97f_hung_iter6000/checkpoints/iter_000007000/fdpolicy5/fd_out/{seq}/vision.mp4"),
    ("33f-hung i7000", R + "/world_camera_nymeria_33f_hung_iter7500/checkpoints/iter_000007000/fdpolicy5/fd_out/{seq}/vision.mp4"),
]
OUT = "/weka/jungbin/cosmos_motion_ft_runs/zeroshot_eval/FD_montage.mp4"
TILE, BAR = 300, 28

seqs = sorted(os.path.basename(os.path.dirname(p))
              for p in glob.glob("/weka/jungbin/cosmos_motion_ft_runs/zeroshot_eval/fdpolicy5/fd_out/*/vision.mp4"))
# keep only sequences present in every source
seqs = [s for s in seqs if all(os.path.isfile(t.format(seq=s)) for _, t in SOURCES)]
print("sequences:", seqs)

inputs, filters, idx = [], [], 0
for r, seq in enumerate(seqs):
    cells = []
    sid = "_".join(seq.split("_")[:2])
    for c, (name, tmpl) in enumerate(SOURCES):
        inputs += ["-i", tmpl.format(seq=seq)]
        f = f"[{idx}:v]scale={TILE}:{TILE},fps=20,pad={TILE}:{TILE+BAR}:0:{BAR}:black"
        if r == 0:
            f += f",drawtext=text='{name}':x=(w-text_w)/2:y=5:fontcolor=yellow:fontsize=16"
        if c == 0:
            f += f",drawtext=text='{sid}':x=5:y=h-18:fontcolor=cyan:fontsize=13"
        f += f"[c{r}_{c}]"; filters.append(f); cells.append(f"[c{r}_{c}]"); idx += 1
    filters.append("".join(cells) + f"hstack=inputs={len(SOURCES)}[row{r}]")
filters.append("".join(f"[row{r}]" for r in range(len(seqs))) + f"vstack=inputs={len(seqs)}[out]")

cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs,
       "-filter_complex", ";".join(filters), "-map", "[out]", "-r", "20", OUT]
print("running ffmpeg with", idx, "inputs ...")
subprocess.run(cmd, check=True)
print("saved", OUT)
