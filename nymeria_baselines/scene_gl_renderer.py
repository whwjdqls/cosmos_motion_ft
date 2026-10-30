"""Depth-buffered offscreen scene previews; optional ModernGL dependency."""
from pathlib import Path
import shutil
import subprocess
import sys

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont


VERTEX = '''
#version 330
uniform mat4 view;
uniform mat4 projection;
uniform float focal_pixels;
uniform float radius;
in vec3 position;
void main() {
    vec4 camera = view * vec4(position, 1.0);
    gl_Position = projection * camera;
    gl_PointSize = clamp(2.0 * radius * focal_pixels / max(-camera.z, 0.01), 1.0, 32.0);
}
'''
FRAGMENT = '''
#version 330
uniform vec4 color;
uniform bool round_points;
out vec4 pixel;
void main() {
    if (round_points && length(gl_PointCoord - vec2(0.5)) > 0.5) discard;
    pixel = color;
}
'''


def camera(eye, target, up, width, height, orthographic=False):
    forward = np.asarray(target) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    vertical = np.cross(right, forward)
    view = np.eye(4)
    view[:3, :3] = [right, vertical, -forward]
    view[:3, 3] = -view[:3, :3] @ eye
    distance = np.linalg.norm(eye - target)
    focal = 1.0 / np.tan(np.deg2rad(45) / 2)
    aspect = width / height
    near, far = .02, max(200., distance * 20)
    if orthographic:
        half_height = distance * np.tan(np.deg2rad(45) / 2)
        projection = np.diag([1/(half_height*aspect), 1/half_height, -2/(far-near), 1.])
        projection[2, 3] = -(far+near)/(far-near)
    else:
        projection = np.zeros((4, 4))
        projection[0, 0], projection[1, 1] = focal / aspect, focal
        projection[2, 2] = -(far + near)/(far-near)
        projection[2, 3] = -2*far*near/(far-near)
        projection[3, 2] = -1
    return view, projection, focal * height / 2


def arrow(origin, direction, length=.35):
    axis = np.asarray(direction, dtype=float)
    axis /= max(np.linalg.norm(axis), 1e-12)
    end = origin + length * axis
    normal = np.cross(axis, [0, 1, 0])
    if np.linalg.norm(normal) < 1e-8:
        normal = np.cross(axis, [1, 0, 0])
    normal /= max(np.linalg.norm(normal), 1e-12)
    return np.asarray([[origin, end], [end, end-.2*length*axis+.08*length*normal],
                       [end, end-.2*length*axis-.08*length*normal]])


def render(args, case, arrays):
    # Import here to avoid circular initialization of the evaluator module.
    from nymeria_baselines.evaluate_gt_scene_collision import CAPSULES, capsule_wire
    points = np.asarray(arrays.get('display_points', arrays['points']), dtype=np.float32)
    if not np.isfinite(points).all():
        raise RuntimeError('Nonfinite display geometry')
    joints, ids = arrays['joints'], arrays['point_ids']
    with np.load(case['bundle']) as z:
        rgb = z['rgb']
    output = args.out / 'visualizations'
    output.mkdir(exist_ok=True)
    png, mp4 = [output / (case['id'] + suffix) for suffix in ['.png', '.mp4']]
    width, height = 800, 470
    ctx = moderngl.create_standalone_context(backend='egl')
    program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
    fbo = ctx.simple_framebuffer((width, height), components=3)
    static = ctx.buffer(points.tobytes())
    cloud = ctx.vertex_array(program, [(static, '3f', 'position')])
    dynamic = ctx.buffer(reserve=2_000_000, dynamic=True)
    lines = ctx.vertex_array(program, [(dynamic, '3f', 'position')])
    ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    framing = arrays.get('framing_joints', joints)
    low, high = framing.min(axis=(0, 1)), framing.max(axis=(0, 1))
    center = (low + high)/2
    extent = max(2.6, float(np.linalg.norm(high-low)) + .9)
    distance = extent / (2*np.tan(np.deg2rad(45)/2))
    cameras = []
    for azimuth in [-65, 25]:
        az, el = np.deg2rad([azimuth, 18])
        eye = center + distance * np.array([np.cos(az)*np.cos(el), np.sin(el), np.sin(az)*np.cos(el)])
        cameras.append(camera(eye, center, [0, 1, 0], width, height))
    # Native motion is Y-up; the top camera shows +X right and -Z up.
    cameras.append(camera(center + [0, distance, 0], center, [0, 0, -1], width, height))
    try:
        font = ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf', 17)
    except OSError:
        font = ImageFont.load_default()

    def draw_geometry(segments, color, line_width=1.0, points_mode=False, radius=.008):
        data = np.asarray(segments, dtype=np.float32).reshape(-1, 3)
        if not len(data):
            return
        dynamic.write(data.tobytes())
        program['color'].value = color
        program['round_points'].value = points_mode
        program['radius'].value = radius
        ctx.line_width = line_width
        lines.render(mode=moderngl.POINTS if points_mode else moderngl.LINES, vertices=len(data))

    def panel(frame, calibration):
        fbo.use()
        fbo.clear(.985, .985, .985, 1., depth=1.)
        view, projection, focal = calibration
        program['view'].write(view.astype('f4').T.tobytes())
        program['projection'].write(projection.astype('f4').T.tobytes())
        program['focal_pixels'].value = focal
        # Opaque body first, then translucent depth-tested point sprites. Only
        # points in front of a body line can blend over it, unlike painter sorting.
        body = [joints[frame, [a, b]] for _, a, b, _ in CAPSULES]
        draw_geometry(body, arrays.get('body_color', (.10, .40, .70, 1.)), line_width=2.2)
        for axis, color in zip(arrays['head_rotation0'].T,
                               [(0.8, .15, .12, 1), (.1, .6, .25, 1), (.15, .3, .85, 1)]):
            draw_geometry(arrow(joints[0, 6], axis, .18), color, line_width=1.5)
        directions = [
            (arrays['camera_positions'][frame], arrays['camera_forward'][frame], (.76, .52, .08, 1)),
            (joints[frame, 6], arrays['head_camera_forward'][frame], (.05, .55, .25, 1)),
            (joints[frame, 0], arrays['torso_forward'][frame], (.5, .3, .7, 1)),
        ]
        for origin, direction, color in directions:
            draw_geometry(arrow(origin, direction), color, line_width=2.)
        program['color'].value = (.5, .5, .5, 128/255)
        program['radius'].value = .008
        program['round_points'].value = True
        cloud.render(mode=moderngl.POINTS)
        # Capsule outlines and markers remain a diagnostic overlay. They are
        # depth-tested against the scene; red is not a physical-collision label.
        wires = []
        for i, (_, a, b, _) in enumerate(CAPSULES):
            wires.extend(capsule_wire(joints[frame, a], joints[frame, b], arrays['radii'][i]))
        wire_segments = [np.stack([w[:-1], w[1:]], axis=1) for w in wires]
        if wire_segments:
            draw_geometry(np.concatenate(wire_segments), (.2, .45, .6, .22), line_width=1.)
        draw_geometry(arrays['points'][ids[frame]], (.88, .12, .08, 1), points_mode=True, radius=.012)
        return Image.frombytes('RGB', (width, height), fbo.read(components=3, alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def frame_image(frame):
        canvas = Image.new('RGB', (1600, 1020), 'white')
        video = Image.fromarray(rgb[frame]).resize((450, 450), Image.Resampling.LANCZOS)
        canvas.paste(video, (175, 34))
        for offset, calibration in zip([(800, 30), (0, 530), (800, 530)], cameras):
            canvas.paste(panel(frame, calibration), offset)
        draw = ImageDraw.Draw(canvas)
        for xy, label in [((20, 8), 'Ground-truth RGB'), ((820, 8), 'Scene view 1'),
                          ((20, 507), 'Scene view 2'), ((820, 507), 'Top view')]:
            draw.text(xy, label, fill='#222222', font=font)
        footer = 'Blue: GT body | Gray: filtered scene | Red: provisional capsule overlap'
        if arrays.get('excluded_capsule_names'):
            footer += ' (thighs/shins/feet ignored)'
        draw.text((20, 998), footer,
                  fill='#333333', font=font)
        return canvas

    ffmpeg = shutil.which('ffmpeg') or str(Path(sys.executable).with_name('ffmpeg'))
    command = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1600x1020', '-r', '20', '-i', '-', '-an', '-c:v', 'libx264',
               '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '1', str(mp4)]
    worst = int(np.argmax([len(v) for v in ids]))
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in range(len(joints)):
            image = frame_image(frame)
            if frame == 0:
                image.save(png)
            if frame == worst:
                image.save(output / (case['id'] + '_worst.png'))
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('ffmpeg failed')
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        renderer = ctx.info['GL_RENDERER']
        cloud.release(); lines.release(); static.release(); dynamic.release()
        fbo.release(); program.release(); ctx.release()
    return {'case': case['id'], 'png': str(png), 'mp4': str(mp4), 'snapshot_frame': 0,
            'worst_frame': worst, 'display_point_limit': None, 'display_point_count': len(points),
            'display_point_source': 'all finite raw confidence-filtered graph positions',
            'display_basis': 'native Y-up, right-handed look-at cameras',
            'renderer': renderer, 'depth_test': True, 'point_radius_m': .008,
            'selection': 'fixed cohort cases; no point-count cap or spatial/ceiling crop'}
