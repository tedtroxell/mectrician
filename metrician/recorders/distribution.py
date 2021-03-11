import torch

from .interface import BaseRecorderInterface

class Distribution(BaseRecorderInterface):
    """[summary]

    :param BaseRecorderInterface: [description]
    :type BaseRecorderInterface: [type]
    """    
    def __init__(self,reduction='mean'):
        super(self.__class__,self).__init__()
        self.reduction = reduction

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return getattr( torch,self.reduction )( x,dim=-1 )
