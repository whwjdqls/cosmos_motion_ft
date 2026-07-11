"""Forward-dynamics viz: GT | generated video side-by-side for each sequence in a samples dir.
Usage: python viz_fd.py --samples <ckpt>/fdpolicy5 --eval_root nymeria_eval5 --tag '...'"""
import argparse, glob, os, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True)
ap.add_argument("--eval_root", required=True)
ap.add_argument("--tag", default="")
args = ap.parse_args()
viz = os.path.join(args.samples, "viz"); os.makedirs(viz, exist_ok=True)

names = sorted(os.path.basename(os.path.dirname(p))
               for p in glob.glob(os.path.join(args.samples, "fd_out", "*", "vision.mp4")))
made = []
for n in names:
    gen = os.path.join(args.samples, "fd_out", n, "vision.mp4")
    gt = os.path.join(args.eval_root, "samples", n, "gt_clip.mp4")
    if not os.path.isfile(gt):
        continue
    tag = "_".join(n.split("_")[:2])
    out = os.path.join(viz, f"{tag}_fd.mp4")
    lbl = f"fwd-dyn (img+text+cam) {args.tag}".strip()
    fc = (f"[0:v]scale=380:380,pad=iw:ih+26:0:26:black,drawtext=text=GT:x=6:y=3:fontcolor=yellow:fontsize=18[a];"
          f"[1:v]scale=380:380,pad=iw:ih+26:0:26:black,drawtext=text={lbl}:x=6:y=3:fontcolor=yellow:fontsize=13[b];[a][b]hstack")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", gt, "-i", gen, "-filter_complex", fc, out], check=False)
    made.append(out)
print(f"made {len(made)} FD side-by-sides in {viz}")
for m in made:
    print(" ", m)
