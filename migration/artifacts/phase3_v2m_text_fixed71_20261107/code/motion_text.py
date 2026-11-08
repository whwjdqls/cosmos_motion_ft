"""Explicit inference-only text routing; never changes live source or model weights."""
from contextlib import contextmanager
import hashlib

class MotionTextScope:
    def __init__(self,motion):
        self.motion=motion
        self.original=motion.prepare_conditions
        self.current=None
        self.calls={'blank':0,'text':0}
        self.cache={}
        self.audit=[]
        motion.prepare_conditions=self.prepare

    @contextmanager
    def use(self,captions):
        assert self.current is None
        if captions is not None:
            assert all(isinstance(c,str) and c.strip() for c in captions)
            self.current=tuple(captions)
        try:
            yield
        finally:
            self.current=None

    def prepare(self,captions):
        assert not any(captions), 'Only the motion-expert inference hook may inject text'
        selected=tuple(captions) if self.current is None else self.current
        assert len(selected)==len(captions)
        self.calls['blank' if self.current is None else 'text']+=1
        # Frozen word embeddings are independent of the diffusion state.
        if selected not in self.cache:
            conditions=self.original(list(selected))
            assert len(conditions)==len(selected)
            self.cache[selected]=conditions
            for caption,condition in zip(selected,conditions,strict=True):
                ids=condition['input_ids'].detach().cpu().numpy()
                self.audit.append(dict(caption=caption,tokens=int(ids.size),
                    token_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
        return self.cache[selected]

    def close(self):
        assert self.current is None
        self.motion.prepare_conditions=self.original
        self.cache.clear()
