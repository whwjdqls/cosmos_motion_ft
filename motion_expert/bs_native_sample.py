"""Sample BONES-SEED motion on the native Cosmos shifted sigma ladder."""
from __future__ import annotations

from bs_sample import main as sample_main


if __name__ == "__main__":
    sample_main(parser_defaults={"sampler": "native"})
