import torch

from .interface import BaseRecorderInterface

class BertIndex2Text(BaseRecorderInterface):
    """[summary]

    :param BaseRecorderInterface: [description]
    :type BaseRecorderInterface: [type]
    """    
    def __init__(self):
        
        super(self.__class__,self).__init__()
        from transformers import PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast()
    
    def forward(self, x : torch.Tensor ) -> str:
        """[summary]

        :param x: [description]
        :type x: torch.Tensor
        :return: [description]
        :rtype: str
        """       
        return self.tokenizer.convert_ids_to_tokens( x ) 

    