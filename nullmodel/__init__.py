"""nullmodel: the null baseline a state-of-the-art claim should beat."""
from .hygiene import screen, memorisation_probe, ceiling_check
from .suite import NullSuite, SUITES
from .metrics import score, higher_is_better

__all__ = ["screen", "memorisation_probe", "ceiling_check", "NullSuite", "SUITES", "score", "higher_is_better"]
__version__ = "0.1.0"
