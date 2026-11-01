"""Export upright resized rays from an Aria VRS or an explicitly approximate K."""
import argparse
import json
from pathlib import Path
import numpy as np
from .camera import pinhole_rays, scaled_intrinsics, sha256


def export(sidecar, out, vrs=None, factory_json=None, factory_scale=None, factory_crop=None):
    with np.load(sidecar, allow_pickle=False) as camera:
        source_hw = camera['image_size_hw'].astype(int)
        raw_k = camera['K_rgb_raw'].astype(float)
        # Older sidecars saved K for counterclockwise rotation, although RGB
        # preprocessing rotates clockwise. Derive K from raw calibration and
        # the actual pixel operation instead of trusting that obsolete field.
        upright_k = np.array([[raw_k[1,1],0,source_hw[1]-1-raw_k[1,2]],
                              [0,raw_k[0,0],raw_k[0,2]],[0,0,1.]])
        k = scaled_intrinsics(upright_k, source_hw)
        stored_upright_k = scaled_intrinsics(camera['K_rgb_upright'], source_hw)
        raw_from_upright = camera['T_rgb_upright'][:3, :3].astype(float)
        model = str(camera['camera_model'])
        device_rgb = camera['T_device_rgb'].astype(float)
    valid = np.ones((256, 256), dtype=bool)
    rays = pinhole_rays(k)
    mode = 'pinhole_approximation_to_unrectified_aria'
    if vrs is not None and factory_json is not None:
        raise ValueError('Provide one calibration source')
    residuals = []
    if vrs is not None or factory_json is not None:
        if vrs is not None:
            from projectaria_tools.core import data_provider
            provider = data_provider.create_vrs_data_provider(str(vrs))
            cc = provider.get_device_calibration().get_camera_calib('camera-rgb')
        else:
            from projectaria_tools.core import calibration
            if factory_scale is None or factory_crop is None:
                raise ValueError('Factory calibration requires explicit sensor scale and crop')
            cc = calibration.device_calibration_from_json(str(factory_json)).get_camera_calib('camera-rgb')
            cc = cc.rescale(source_hw, factory_scale, np.asarray(factory_crop, float))
        if not np.allclose(cc.get_transform_device_camera().to_matrix(), device_rgb, atol=1e-5):
            raise ValueError('Calibration device-to-RGB extrinsic differs from sidecar')
        # Bind calibration to THIS recording, not an unrelated camera sidecar.
        raw_wh = np.asarray(cc.get_image_size(), int)
        f, pp = np.asarray(cc.get_focal_lengths()), np.asarray(cc.get_principal_point())
        upright_k = np.array([[f[1],0,raw_wh[1]-1-pp[1]],[0,f[0],pp[0]],[0,0,1.]])
        expected = scaled_intrinsics(upright_k, [raw_wh[0], raw_wh[1]])
        if not np.allclose(expected, k, atol=1e-4):
            raise ValueError('Raw calibration does not match camera sidecar')
        for y in range(256):
            for x in range(256):
                # Training: clockwise90 rotation, then Pillow resize to640/256.
                u = (x + .5) * source_hw[1] / 256 - .5
                v = (y + .5) * source_hw[0] / 256 - .5
                raw_uv = np.array([v, raw_wh[1] - 1 - u])
                ray = cc.unproject(raw_uv)
                if ray is None or not np.isfinite(ray).all():
                    valid[y, x] = False
                    rays[y, x] = [0, 0, 1]
                else:
                    rays[y, x] = raw_from_upright.T @ np.asarray(ray)
                    projected = cc.project(ray)
                    if projected is None:
                        raise ValueError('Valid unprojected ray cannot be projected')
                    residuals.append(float(np.linalg.norm(projected - raw_uv)))
        if not residuals or max(residuals) > 0.01:
            raise ValueError('Aria projection/unprojection roundtrip failed')
        mode = 'measured_aria_unprojection_rays'
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rays=rays.astype(np.float32), valid=valid, K_256=k,
             mode=np.array(mode), source=np.array(str(Path(sidecar).resolve())),
             sidecar_sha256=np.array(sha256(sidecar)), source_model=np.array(model))
    metadata = dict(mode=mode, source_model=model, sidecar=str(sidecar), sidecar_sha256=sha256(sidecar),
                    output_sha256=sha256(out), K_256=k.tolist(), output_hw=[256,256],
                    vrs=None if vrs is None else str(vrs),
                    factory_json=None if factory_json is None else str(factory_json),
                    factory_json_sha256=None if factory_json is None else sha256(factory_json),
                    factory_scale=factory_scale, factory_crop=factory_crop,
                    stored_sidecar_K_256=stored_upright_k.tolist(),
                    stored_upright_K_max_discrepancy_pixels=float(np.abs(k-stored_upright_k).max()),
                    valid_fraction=float(valid.mean()),
                    max_roundtrip_raw_pixels=max(residuals) if residuals else None,
                    limitation='Ray matching does not reproduce exposure, rolling shutter or resize filtering.')
    out.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sidecar', type=Path, required=True)
    p.add_argument('--vrs', type=Path)
    p.add_argument('--factory-calibration-json', type=Path)
    p.add_argument('--factory-scale', type=float)
    p.add_argument('--factory-crop', type=float, nargs=2)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(export(args.sidecar, args.out, args.vrs, args.factory_calibration_json,
                            args.factory_scale, args.factory_crop), indent=2))


if __name__ == '__main__':
    main()
