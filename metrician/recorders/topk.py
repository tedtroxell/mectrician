import torch

from .interface import BaseRecorderInterface

class TopK(BaseRecorderInterface):
    """[summary]

    :param BaseRecorderInterface: [description]
    :type BaseRecorderInterface: [type]
    """    

    def __init__(self,k:int,reduction='mean'):
        super(self.__class__,self).__init__()
        self.k = k
        self.reduction = reduction


    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return torch.topk( getattr( torch,self.reduction )( x,dim=-1 ),self.k)
