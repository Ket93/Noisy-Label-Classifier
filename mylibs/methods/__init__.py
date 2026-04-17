from .baseline_ce        import BaselineCE
from .label_smoothing    import LabelSmoothing
from .sce                import SCE
from .gce                import GCE
from .small_loss         import SmallLoss
from .gmm_reweight       import GMMReweight
from .confident_learning import ConfidentLearning
from .curriculum         import Curriculum

__all__ = [
    'BaselineCE', 'LabelSmoothing', 'SCE', 'GCE',
    'SmallLoss', 'GMMReweight', 'ConfidentLearning', 'Curriculum',
]
