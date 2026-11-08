"""Whole-bridge reference CFG through the model's existing bridges_enabled flag."""


class Guided:
    def __init__(self, model, mode, scale):
        self.model, self.mode, self.scale = model, mode, scale
        self.reference = False
        self.counters = dict(conditioned_calls=0, reference_calls=0,
                             conditioned_bridge_calls=0, reference_bridge_calls=0)
        self.handles = [b.register_forward_hook(self.observe) for b in model.bridges.values()]

    def observe(self, module, inputs, output):
        key = 'reference_bridge_calls' if self.reference else 'conditioned_bridge_calls'
        self.counters[key] += 1

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, **inputs):
        assert inputs['modes'] == [self.mode] * len(inputs['modes'])
        assert not any(inputs['captions']) and 'bridges_enabled' not in inputs
        conditioned = self.model(**inputs)
        self.counters['conditioned_calls'] += 1
        if self.scale == 1:
            return conditioned
        self.reference = True
        try:
            reference = self.model(**inputs, bridges_enabled=False)
        finally:
            self.reference = False
        self.counters['reference_calls'] += 1
        assert self.counters['reference_bridge_calls'] == 0
        key = 'motion' if self.mode == 'video2motion' else 'video'
        output = dict(conditioned)
        output[key] = reference[key].float() + self.scale * (conditioned[key].float() - reference[key].float())
        return output

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
