from .rpn import SBRPN
from .backbone import ClipRN101
from .meta_arch_f1_dis_p8 import ClipRCNNWithClipBackbone
from .roi_head import ClipRes5ROIHeads
from .config import add_stn_config
from .custom_pascal_evaluation import CustomPascalVOCDetectionEvaluator