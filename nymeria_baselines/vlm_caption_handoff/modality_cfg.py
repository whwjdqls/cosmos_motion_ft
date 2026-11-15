"""Independent video/text CFG using existing bridges and motion text routing.

g_text = p_text + video_scale * (p_video_text - p_text)
g_null = p_null + video_scale * (p_video - p_null)
result = g_null + text_scale * (g_text - g_null)

Shape remains supplied in all branches. Exact endpoint paths preserve legacy CFG.
"""
from contextlib import contextmanager
import math


class ModalityGuided:
    def __init__(self, model, text_scope, video_scale, text_scale):
        self.model, self.scope = model, text_scope
        self.video_scale, self.text_scale = float(video_scale), float(text_scale)
        if not all(math.isfinite(v) and v >= 0 for v in [self.video_scale, self.text_scale]):
            raise ValueError('Guidance weights must be finite and nonnegative')
        self.counters = {k: 0 for k in ['video_text', 'text_only', 'video_only', 'neither']}
        self.bridge_calls = dict(self.counters)
        self.active = None
        self.handles = [b.register_forward_hook(self.observe) for b in model.bridges.values()]

    def observe(self, module, inputs, output):
        assert self.active in ['video_text', 'video_only'], 'A no-video branch reached a bridge'
        self.bridge_calls[self.active] += 1

    def eval(self):
        self.model.eval()
        return self

    @contextmanager
    def text_branch(self, enabled):
        previous = self.scope.current
        if not enabled:
            self.scope.current = None
        try:
            yield
        finally:
            self.scope.current = previous

    def forward_branch(self, inputs, video, text):
        name = ('video_text' if text else 'video_only') if video else ('text_only' if text else 'neither')
        self.active = name
        try:
            with self.text_branch(text):
                result = self.model(**inputs) if video else self.model(**inputs, bridges_enabled=False)
            self.counters[name] += 1
            return result
        finally:
            self.active = None

    def video_guided(self, inputs, text):
        if self.video_scale == 0:
            reference = self.forward_branch(inputs, False, text)
            return reference, reference['motion']
        conditioned = self.forward_branch(inputs, True, text)
        if self.video_scale == 1:
            return conditioned, conditioned['motion']
        reference = self.forward_branch(inputs, False, text)
        value = reference['motion'].float() + self.video_scale * (
            conditioned['motion'].float() - reference['motion'].float())
        return conditioned, value

    def __call__(self, **inputs):
        assert inputs['modes'] == ['video2motion'] * len(inputs['modes'])
        assert not any(inputs['captions']) and 'bridges_enabled' not in inputs
        if self.text_scale == 0:
            template, value = self.video_guided(inputs, False)
        elif self.text_scale == 1:
            # Exactly the old two-forward video CFG, including operation order.
            template, value = self.video_guided(inputs, True)
        else:
            template, text_value = self.video_guided(inputs, True)
            _, null_value = self.video_guided(inputs, False)
            value = null_value.float() + self.text_scale * (text_value.float() - null_value.float())
        output = dict(template)
        output['motion'] = value
        return output

    def assert_counts(self, batch, steps):
        texts = [False] if self.text_scale == 0 else [True] if self.text_scale == 1 else [True, False]
        videos = [False] if self.video_scale == 0 else [True] if self.video_scale == 1 else [True, False]
        expected = {k: 0 for k in self.counters}
        for text in texts:
            for video in videos:
                name = ('video_text' if text else 'video_only') if video else ('text_only' if text else 'neither')
                expected[name] = steps
        assert self.counters == expected, (self.counters, expected)
        bridge_expected = {k: v * len(self.model.bridges) * batch if k in ['video_text', 'video_only'] else 0 for k, v in expected.items()}
        assert self.bridge_calls == bridge_expected, (self.bridge_calls, bridge_expected)

    def close(self):
        assert self.active is None
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
