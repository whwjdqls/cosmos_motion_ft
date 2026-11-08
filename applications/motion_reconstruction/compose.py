"""Compose synchronized mesh comparisons, paper figures and vector legends."""
import argparse
import json
import hashlib
from pathlib import Path
import subprocess
import shutil
import textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .colors import error_colors
from .pipeline import require_compute,write


def white_image(path):
    with Image.open(path) as source:
        rgba=source.convert('RGBA')
        white=Image.new('RGBA',rgba.size,'white');white.alpha_composite(rgba)
    return white.convert('RGB')


def font(size,bold=False):
    return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',size)


def label_center(draw,text,center,y,size=20,bold=False,color='#27233C'):
    f=font(size,bold);box=draw.textbbox((0,0),text,font=f)
    draw.text((center-(box[2]-box[0])/2,y),text,font=f,fill=color)


def video_frame(root,manifest,case,frame):
    panel_w,panel_h=384,512;header,footer=68,126
    image=Image.new('RGB',(panel_w*len(case['models']),header+panel_h+footer),'white')
    draw=ImageDraw.Draw(image)
    for i,m in enumerate(case['models']):
        path=root/case['id']/'renders'/m['label']/'video'/f'{frame:06d}.png'
        panel=white_image(path);assert panel.size==(panel_w,panel_h)
        image.paste(panel,(i*panel_w,header))
        label_center(draw,m['title'],(i+.5)*panel_w,8,24,True)
        subtitle='Reference' if m['label']=='GT' else m['input']
        # Avoid embedding model training details into body geometry labels.
        subtitle=subtitle.replace(' · ',' | ')
        label_center(draw,subtitle,(i+.5)*panel_w,41,12,color='#666477')
    width=420;left=(image.width-width)//2;top=header+panel_h+12
    gradient=error_colors(np.linspace(0,manifest['maximum_cm']/100,width),manifest['maximum_cm'])
    bar=np.repeat((gradient[None]*255).round().astype('uint8'),16,axis=0)
    image.paste(Image.fromarray(bar),(left,top))
    for fraction in (0,1/3,2/3,1):
        x=left+int(fraction*(width-1));draw.line((x,top+17,x,top+22),fill='#777480')
        value=manifest['maximum_cm']*fraction
        label_center(draw,('≥ ' if fraction==1 else '')+f'{value:g}',x,top+25,13)
    title=error_label(manifest)
    label_center(draw,title,image.width/2,top+49,15)
    footer=('World-space meshes · pelvis-relative colors · fixed camera and color scale'
        if manifest.get('error_reference','world')=='root_relative' else
        'Raw world coordinates · common camera and color scale · no alignment or grounding')
    label_center(draw,footer,image.width/2,top+73,11,color='#666477')
    return image


def error_label(manifest):
    local=manifest.get('error_reference','world')=='root_relative'
    if manifest['error_mode']=='joint_lbs':
        return 'Pelvis-relative joint error (cm)' if local else 'Global joint position error (cm)'
    return 'Pelvis-relative template vertex distance (cm)' if local else 'Template vertex distance (cm)'


def legend_figure(root,maximum_cm,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,Normalize
    from matplotlib.colorbar import ColorbarBase
    cmap=ListedColormap(error_colors(np.linspace(0,maximum_cm/100,256),maximum_cm))
    fig=plt.figure(figsize=(5.2,.9));ax=fig.add_axes((.08,.58,.84,.22))
    colorbar=ColorbarBase(ax,cmap=cmap,norm=Normalize(0,maximum_cm),orientation='horizontal')
    colorbar.set_ticks([0,maximum_cm/3,maximum_cm*2/3,maximum_cm])
    colorbar.set_ticklabels(['0',f'{maximum_cm/3:g}',f'{maximum_cm*2/3:g}',f'≥{maximum_cm:g}'])
    colorbar.outline.set_visible(False);colorbar.ax.tick_params(length=3,labelsize=10,color='#777480')
    colorbar.set_label(title,fontsize=11,labelpad=5)
    for extension in ('svg','pdf','png'):fig.savefig(root/('error_legend.'+extension),dpi=300,transparent=True,bbox_inches='tight')
    plt.close(fig)
    return cmap


def paper_figure(root,manifest,case,frames,cmap,name):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    columns=len(case['models']);count=len(frames)
    fig,axes=plt.subplots(count,columns,figsize=(3.*columns,4.0*count+.65),squeeze=False)
    fig.patch.set_facecolor('white')
    for row,frame in enumerate(frames):
        for column,method in enumerate(case['models']):
            ax=axes[row,column]
            source=root/case['id']/'renders'/method['label']/'paper'/f'{frame:06d}.png'
            ax.imshow(white_image(source));ax.axis('off')
            if row==0:
                ax.set_title(method['title']+'\n'+('Reference' if column==0 else method['input']),
                    fontsize=11,pad=8,color='#27233C')
            if column==0 and len(frames)>1:
                ax.text(-.015,.5,f't = {frame/case["fps"]:.1f} s',transform=ax.transAxes,rotation=90,
                    va='center',ha='right',fontsize=9,color='#555160')
    fig.subplots_adjust(left=.03,right=.995,top=.88 if count==1 else .965,bottom=.15 if count==1 else .06,wspace=.025,hspace=.025)
    bar=fig.add_axes((.35,.10 if count==1 else .025,.3,.018 if count==1 else .005))
    colorbar=fig.colorbar(ScalarMappable(norm=Normalize(0,manifest['maximum_cm']),cmap=cmap),cax=bar,orientation='horizontal')
    vmax=manifest['maximum_cm'];colorbar.set_ticks([0,vmax/3,2*vmax/3,vmax])
    colorbar.set_ticklabels(['0',f'{vmax/3:g}',f'{2*vmax/3:g}',f'≥{vmax:g}'])
    colorbar.outline.set_visible(False);colorbar.ax.tick_params(labelsize=8,length=2)
    colorbar.set_label(error_label(manifest),fontsize=9,labelpad=3)
    destination=root/case['id']/'paper';destination.mkdir(exist_ok=True)
    for extension in ('png','pdf'):
        fig.savefig(destination/(name+'.'+extension),dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def main(a):
    require_compute()
    manifest=json.loads((a.root/'manifest.json').read_text())
    import imageio_ffmpeg
    cmap=legend_figure(a.root,manifest['maximum_cm'],error_label(manifest))
    previews=[]
    for case in manifest['cases']:
        folder=a.root/case['id'];frames=case['frames']
        # Verify render completion/recipes before accepting any comparison.
        for method in case['models']:
            parent=folder/'renders'/method['label']
            for tag,expected in [('video',list(range(frames))),('paper',[0,32,48,64,96])]:
                complete=json.loads((parent/tag/'COMPLETE.json').read_text())
                assert complete['passed'] and complete['frames']==expected
                assert complete['maximum_cm']==manifest['maximum_cm']
                assert complete.get('error_reference','world')==manifest.get('error_reference','world')
                assert complete.get('shadow_style','studio_three_light')==manifest.get('shadow_style','studio_three_light')
                assert complete['device']['backend'] in ('OPTIX','CUDA')
        output=folder/'comparison.mp4'
        first=video_frame(a.root,manifest,case,0);width,height=first.size
        encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error',
            '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(case['fps']),'-i','-',
            '-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
        try:
            for frame in range(frames):
                panel=first if frame==0 else video_frame(a.root,manifest,case,frame)
                encoder.stdin.write(np.asarray(panel).tobytes())
                if frame==48:panel.save(folder/'preview.png',dpi=(300,300))
            encoder.stdin.close();assert encoder.wait()==0
        finally:
            if encoder.poll() is None:encoder.terminate();encoder.wait()
        paper_figure(a.root,manifest,case,[48],cmap,'frame_000048')
        paper_figure(a.root,manifest,case,[0,32,64,96],cmap,'temporal_contact_sheet')
        previews.append(dict(id=case['id'],video=str(output.relative_to(a.root)),preview=str((folder/'preview.png').relative_to(a.root)),
            paper=str((folder/'paper/frame_000048.pdf').relative_to(a.root)),caption=case['caption']))
        print('COMPOSED',case['id'],flush=True)
    import html
    entries=''.join('<section><h2>'+html.escape(p['id'])+'</h2><p>'+html.escape(p['caption'])+'</p>'
        '<video controls loop muted playsinline width="100%" src="'+p['video']+'" poster="'+p['preview']+'"></video>'
        '<p><a href="'+p['paper']+'">Paper PDF</a></p></section>' for p in previews)
    error_explanation=(error_label(manifest)+', interpolated using SOMA LBS influences. '
        +('Each motion has its own per-frame pelvis position subtracted for coloring; original mesh placement is retained.'
          if manifest.get('error_reference','world')=='root_relative'
          else 'Global errors include root translation.')
        if manifest['error_mode']=='joint_lbs' else 'Distances between the two posed standard-template surfaces; not scanned subject-specific GT.')
    (a.root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SOMA reconstruction errors</title>'
        '<style>body{max-width:1400px;margin:40px auto;font:16px system-ui;color:#27233c;background:white}section{margin:40px 0}video{border:1px solid #efedf5;border-radius:8px}</style>'
        '<h1>SOMA motion reconstruction: joint-derived surface errors</h1>'
        '<p>Indigo #7B77FF → muted rose #D8768C. Shared0–30cm scale. '+error_explanation+' '
        'Standard visualization skin, not subject-specific ground-truth surface. Inputs differ: ours uses video; EgoEgo/EgoAllo use camera+shape. '
        'The first/middle/last full-test cases were selected deterministically before inspecting visual quality.</p>'+entries)
    source=a.root/'assets/composition_source';source.mkdir(exist_ok=True)
    shutil.copy2(__file__,source/'compose.py')
    write(a.root/'COMPOSED.json',dict(passed=True,cases=previews,frames_per_case=97,fps=20,
        composition_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args())
