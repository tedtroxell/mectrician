import torch
import lime
from enum import Enum

class Explainations(Enum):
    text    = 0
    image   = 1
    tabular = 2


def get_explainer(dtype,kwargs):
    return {
        Explainations.image:lime.lime_image.LimeImageExplainer,
        # Explainations.text:lime.lime_text.LimeTextExplainer,
        # Explainations.tabular:lime.lime_tabular.LimeTabularExplainer
    }[dtype]( **kwargs )


def _predict_wrapper(func : torch.nn.Module) -> torch.nn.Module:
    '''
        provide portability to lime
    '''
    def coerce(*args,**kwargs):
        return func(*args,**kwargs).detach().cpu().numpy()
    
    return coerce