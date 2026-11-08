"""Torchrun entrypoint that survives terminal hangups without swallowing stop signals."""
from __future__ import annotations

import signal


def guarded_handler(original):
    def handle(signum, frame):
        if signum == signal.SIGHUP:
            print("[phase3/agent] SIGHUP ignored; use SIGUSR1 on rank0 for a checkpointed stop", flush=True)
            return
        return original(signum, frame)
    return handle


def main():
    from torch.distributed.elastic.multiprocessing import api
    from torch.distributed.run import main as torchrun

    # TorchElastic replaces inherited SIG_IGN, so nohup alone is insufficient.
    # Keep the native handler for every signal except the observed SIGHUP.
    api._terminate_process_handler = guarded_handler(api._terminate_process_handler)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    torchrun()


if __name__ == "__main__":
    main()
