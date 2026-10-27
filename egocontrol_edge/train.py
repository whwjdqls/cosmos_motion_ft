"""Register the baseline and retain the native Phase1 trainer unchanged."""
import egocontrol_edge.experiment  # noqa: F401
from native_phase_training.run_latent_train import main

if __name__ == '__main__':
    main()
