"""Run an unchanged training module or logger with SIGHUP ignored."""
from __future__ import annotations

import argparse
import runpy
import signal
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--module")
    target.add_argument("--script")
    args, rest = parser.parse_known_args()
    sys.argv = [args.module or args.script, *rest]
    if args.module:
        runpy.run_module(args.module, run_name="__main__")
    else:
        runpy.run_path(args.script, run_name="__main__")


if __name__ in {"__main__", "__mp_main__"}:
    # Spawned DataLoader workers also import this entrypoint, but must not
    # recursively execute the trainer while bootstrapping.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
if __name__ == "__main__":
    main()
