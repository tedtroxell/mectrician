from .interface import BaseMonitorInterface
from typing import List

from torch.utils.data import Dataset
from typing import Union,Iterable
from metrician.signals.outliers import MomentOutlierSignal as MOS
from metrician.signals.interface import BaseSignal
from metrician.recorders.interface import BaseRecorderInterface
from metrician.signals import SignalDataKeys
from metrician.explainations.interface import BaseExplainerInterface
from torch import Tensor
import metrician
class OutOfDistributionMonitor(BaseMonitorInterface):
    """
        Default monitor for out of distribution losses.

        Basically this sets up a monitor that calculates whether some output is an outlier and records the input so you can evaluate it later on

        :param BaseMonitorInterface: [description]
        :type BaseMonitorInterface: [type]
    """        

    def __init__(   
                    self,
                    dataset : Union[Dataset,Iterable],
                    data_type : Union[str,metrician.DatasetType] = None,
                    signals  : List[BaseSignal] = [MOS()],
                    recorders : List[BaseRecorderInterface] = [lambda x: x],
                    explainer  : BaseExplainerInterface = None,
                    slug : str = 'Out Of Distribution Sample'
                ):
        """

        :param dataset: [description]
        :type dataset: Union[Dataset,Iterable]
        :param data_type: [description], defaults to None
        :type data_type: Union[str,metrician.DatasetType], optional
        :param signal: [description], defaults to MOS()
        :type signal: BaseSignal, optional
        :param recorder: [description], defaults to None  
        :type recorder: BaseRecorderInterface, optional
        :param explainer: [description], defaults to None
        :type explainer: BaseExplainerInterface, optional
        :param slug: [description], defaults to 'Out Of Distribution Sample'
        :type slug: str, optional
        """                       
        super( self.__class__,self ).__init__()     
        assert data_type is None and dataset.__class__.__name__ in metrician.DATASET_INPUT_TYPES, 'If your dataset is not a torch dataset you have to specify the input data type [image,audio,text,tabular] '
        self.dataset_length = len(dataset)
        self.dtype = data_type if data_type is not None else metrician.DATASET_INPUT_TYPES[dataset.__class__.__name__]
        self.signals = signals
        self._index = 0
        self.dataset = dataset
        self.slug = slug
        self.writer = self._write_fn()
        self.recorders = recorders
        self.explainer = explainer

    def _check_signal(self,signal : BaseSignal,*args,**kwargs) -> None:
        """[summary]

        :param signal: [description]
        :type signal: BaseSignal
        """      

        ood = signal( *args,**kwargs )
        if ood:
            for recorder in self.recorders:
                self.writer(
                    self.slug+'_'+signal.__class__.__name__,
                    # take the first index since it's the input
                    recorder( self.dataset[ self._index % self.dataset_length ][0] ), 
                    self._index
                )
            if self.explainer is not None: 
                kwargs[SignalDataKeys.INPUT] = self.dataset[ self._index % self.dataset_length ][0]
                self.explainer( kwargs )

            self.captured_signals.append( self._index % self.dataset_length )


    def forward(self,*args, **kwargs) -> None:
        """


        

        :param loss: output from loss function
        :type loss: Tensor
        :param y: data labels/target
        :type y: Tensor
        :param yhat: model's prediction 
        :type yhat: Tensor
        """                 
        for signal in self.signals:self._check_signal(signal,*args, **kwargs)
        
        self._index += 1


    

