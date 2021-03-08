import inspect
from typing import Union
from metrician.interface import BaseInterface
from metrician.signals import SignalDataKeys
class BaseSignal(BaseInterface):

    keys = [ SignalDataKeys.LOSS,SignalDataKeys.LABELS, SignalDataKeys.PREDICTIONS ]

    def _check_input(self,inputs ):
        keys = set( inputs.keys() )
        # check if keys are in the data
        if len( keys.intersection( self.keys ) ) < len(self.keys): raise AttributeError(f'Input data malformed for {self.__class__.__name__}')
        

