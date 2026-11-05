"""Phase 3 never conditions either specialist on dataset captions."""


def empty_captions(captions, modes):
    if len(captions) != len(modes):
        raise ValueError("One caption slot is required per Phase3 sample")
    return [""] * len(modes)
