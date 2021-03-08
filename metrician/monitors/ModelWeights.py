from torch.utils.data import Dataset
from typing import Union,Iterable
from metrician.signals.outliers import MomentOutlierSignal as MOS
from metrician.signals.interface import BaseSignal
from metrician.recorders.interface import BaseRecorderInterface
from .interface import BaseMonitorInterface
from torch import Tensor,nn
import metrician

class ModelWeightsMonitor(BaseMonitorInterface):
    """

    :param BaseMonitorInterface: [description]
    :type BaseMonitorInterface: [type]
    """

    def __init__(
                    self,
                    model : Union[nn.Module,Iterable],
                    signal  : BaseSignal = MOS(),
                    recorder : BaseRecorderInterface = None,
                    slug : str = 'Model Weights Sample'
                ):
        super(self.__class__,self).__init__()
