"""Launch the native EgoAllo trainer against an accepted clean SOMA cache."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--experiment', required=True)
    a = p.parse_args()
    audit = json.loads((a.cache / 'acceptance.json').read_text())
    assert audit['phase3_matched_windows'] == 112503
    assert audit['phase3_max_normalized_difference'] <= 1e-4
    sys.path.insert(0, str(a.upstream / 'src'))
    sys.path.insert(0, str(a.upstream))
    os.chdir(a.upstream)
    import torch
    from egoallo.data.nymeria_soma import NymeriaSomaDataset
    data = NymeriaSomaDataset(a.cache, split='train')
    for index in (0, len(data)//2, len(data)-1):
        sample = data[index]
        for field in dataclasses.fields(sample):
            value = getattr(sample, field.name)
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), (index, field.name)
        assert sample.body_quats.shape == (97, 22, 4)
        assert sample.contacts.shape == (97, 4)
        assert sample.hand_quats is None
    print('Clean EgoAllo loader acceptance passed', flush=True)
    native = runpy.run_path(str(a.upstream / '1_train_motion_prior.py'))
    model = native['network'].EgoDenoiserConfig(
        include_hands=False, body_joint_count=22, contact_joint_count=4)
    config = native['EgoAlloTrainConfig'](
        experiment_name=a.experiment, dataset_hdf5_path=a.cache / 'unused.hdf5',
        dataset_files_path=a.cache / 'index.json', nymeria_cache=a.cache,
        model=model, batch_size=256, num_workers=4, subseq_len=97,
        train_splits=('train',))
    print(config, flush=True)
    native['run_training'](config)


if __name__ == '__main__':
    main()
