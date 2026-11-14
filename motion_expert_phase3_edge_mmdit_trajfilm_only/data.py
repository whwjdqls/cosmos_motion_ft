"""Exact existing112503-window motion/trajectory cache; never open video inputs."""
from pathlib import Path
from collections import OrderedDict
import json,hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

def read(p):return json.loads(Path(p).read_text())
def identity(p):
    st=Path(p).stat();return dict(size=st.st_size,mtime_ns=st.st_mtime_ns)

class CameraOnlyDataset(Dataset):
    def __init__(self,motion_cache,trajectory_cache,pairing_receipt=None):
        self.root=Path(motion_cache);self.trajectory_root=Path(trajectory_cache)
        if not all(str(p.resolve()).startswith('/mnt/local/') for p in (self.root,self.trajectory_root)):raise ValueError('Training inputs must be node-local')
        self.index=read(self.root/'index.json');self.trajectory=read(self.trajectory_root/'index.json');self.rows=self.index['rows']
        assert self.index['full_training_index'] and self.trajectory['full_training_index']
        assert len(self.rows)==self.trajectory['source']['rows']==112503
        self.pairing=None
        if self.trajectory['source']['baseline_fingerprint']!=self.index['fingerprint']:
            if pairing_receipt is None:raise ValueError('Different cache metadata needs an exact array/row pairing proof')
            proof=read(pairing_receipt)
            def digest(path):
                with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
            assert proof['passed'] and proof['rows']==112503 and proof['exact_row_alignment'] and proof['all_motion_neutral_arrays_byte_identical']
            assert proof['actual_motion_cache']==str(self.root) and proof['trajectory_cache']==str(self.trajectory_root)
            assert proof['actual_fingerprint']==self.index['fingerprint'] and proof['reference_fingerprint']==self.trajectory['source']['baseline_fingerprint']
            assert proof['actual_index_sha256']==digest(self.root/'index.json') and proof['trajectory_index_sha256']==digest(self.trajectory_root/'index.json')
            self.pairing=proof['files']
        stats=self.trajectory['normalization'];assert stats['split']=='train' and stats['windows']==112503
        self.mean=np.array(stats['mean'],dtype=np.float32);self.std=np.array(stats['std'],dtype=np.float32)
        assert self.mean.shape==self.std.shape==(18,) and np.isfinite(self.mean).all() and np.isfinite(self.std).all() and (self.std>0).all()
        self.arrays=OrderedDict()
    def __len__(self):return len(self.rows)
    def __getitem__(self,key):
        draw,index=key if isinstance(key,tuple) else (int(key),int(key));row=self.rows[index];array=row['array'];record=self.trajectory['records'][row['uuid']]
        assert record['file']==array+'.trajectory18.npy'
        if array not in self.arrays:
            path=self.trajectory_root/record['file'];assert identity(path)==record['file_identity']
            if self.pairing is not None:
                for suffix in ['.motion.npy','.neutral.npy']:
                    file=self.root/(array+suffix);st=file.stat();proof=self.pairing[array+suffix]
                    assert (st.st_size,st.st_mtime_ns,st.st_ino)==(proof['size'],proof['mtime_ns'],proof['inode'])
            self.arrays[array]=(np.load(self.root/(array+'.motion.npy'),mmap_mode='r'),np.load(self.root/(array+'.neutral.npy'),mmap_mode='r'),np.load(path,mmap_mode='r'))
        self.arrays.move_to_end(array)
        while len(self.arrays)>16:self.arrays.popitem(last=False)
        motion,neutral,trajectory=self.arrays[array];m=motion[row['row']].copy();n=neutral.copy();c=(trajectory[row['row']]-self.mean)/self.std
        assert m.shape==(97,283) and n.shape==(30,3) and c.shape==(97,18)
        assert np.isfinite(m).all() and np.isfinite(n).all() and np.isfinite(c).all()
        return dict(motion=torch.from_numpy(m),neutral=torch.from_numpy(n),camera_features=torch.from_numpy(c.copy()),draw=draw,case=f"{row['uuid']}@{row['start']}")

def collate(rows):
    return {**{k:torch.stack([r[k] for r in rows]) for k in ['motion','neutral','camera_features']},'draw':[r['draw'] for r in rows],'case':[r['case'] for r in rows]}
