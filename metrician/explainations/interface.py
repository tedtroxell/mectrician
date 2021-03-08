import torch
from metrician.interface import BaseInterface
from metrician.explainations import _predict_wrapper,get_explainer,Explainations

class BaseExplainerInterface(BaseInterface):
    """
    

    :param BaseInterface: [description]
    :type BaseInterface: [type]
    """    
    explainer_kwargs    = {}
    explainer_type      = None
    def __init__(
                    self,
                    model : torch.nn.Module
    ):
        
        # super(self.__class__,self).__init__(model)
        # self.explainer = get_explainer(self.explainer_type,self.explainer_kwargs)
        self.fn = _predict_wrapper( model )

    def _post_process(self,*args,**kwargs):raise NotImplementedError('"_post_process" not implemented for explainer!')
    def _explain(self,*args,**kwargs):raise NotImplementedError('"_explain" not implemented for explainer!')
        
    def forward(self,*args,**kwargs):
        return self._post_process(
                self._explain(*args,**kwargs)
        )
