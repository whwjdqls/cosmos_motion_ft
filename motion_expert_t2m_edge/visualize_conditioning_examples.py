"""Extend the fixed-text CFG preview with raw root paths and input annotations.

Generation is delegated unchanged to evaluate_text2_image_cfg. Any target boxes
are applied only to separate display figures, never to the model input.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import evaluate_text2_image_cfg as sweep

standard_render=sweep.render_case


def render_case(args,case,pixels):
    standard_render(args,case,pixels)
    gallery=args.out/'gallery';slot=case['slot']
    with np.load(args.out/'preview'/f'case_{slot:03d}'/'seed_0.npz',allow_pickle=False) as saved:
        data={key:np.array(saved[key]) for key in saved.files}
    fig,ax=plt.subplots(figsize=(7,6),dpi=130)
    trajectories={}
    for key,title,color in [('gt','GT','tab:blue'),('image_only','Image only','tab:green'),
                            ('text2_img0','Text only, CFG2','tab:orange'),('text2_img2','Text2 + image2','tab:purple')]:
        root=data[key+'_joints'][:,0,:];xz=root[:,[0,2]]
        path=float(np.linalg.norm(np.diff(xz,axis=0),axis=-1).sum())
        net=float(np.linalg.norm(xz[-1]-xz[0]))
        trajectories[key]={'path_m':path,'net_displacement_m':net}
        ax.plot(xz[:,0],xz[:,1],label=f'{title}: path {path:.2f}m, net {net:.2f}m',color=color,lw=2)
        ax.scatter(xz[0,0],xz[0,1],color=color,marker='o',s=35)
        ax.scatter(xz[-1,0],xz[-1,1],color=color,marker='x',s=65)
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.3)
    ax.set_xlabel('World X (m)');ax.set_ylabel('World Z (m)')
    ax.set_title(f"Case {slot}: root trajectories over 4.8 seconds\nRaw coordinates; circle=start, cross=end")
    ax.legend(loc='best',fontsize=8)
    fig.text(.5,.02,'No re-centering, GT alignment, or predicted-ground correction. Different valid routes are possible.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(gallery/f'case_{slot:03d}_trajectories.png');plt.close(fig)
    (args.out/'preview'/f'case_{slot:03d}'/'trajectory_summary.json').write_text(json.dumps(trajectories,indent=2)+'\n')
    if case.get('object_bbox'):
        fig,ax=plt.subplots(figsize=(6,6),dpi=130)
        ax.imshow(pixels.transpose(1,2,0),interpolation='nearest')
        x0,y0,x1,y1=case['object_bbox']
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#ffb000',linewidth=2.5))
        ax.set_title(f"Visible target: {case['target_object']}\nExact model input, enlarged for inspection")
        ax.axis('off');fig.text(.5,.02,'Amber annotation is for display only. The generator receives unannotated RGB.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(gallery/f'case_{slot:03d}_object.png');plt.close(fig)


if __name__=='__main__':
    if len(sys.argv)<2 or sys.argv[1]!='preview':raise SystemExit('This wrapper is for preview only')
    sweep.render_case=render_case
    sweep.main()
