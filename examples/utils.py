import torch
from torch import nn

class ClassificationModel(nn.Module):
    def __init__(self,input_size,output_size):
        super(self.__class__,self).__init__()
        self.fn = nn.Sequential(
            nn.Linear(input_size,input_size),
            nn.Tanh(),
            nn.Linear(input_size,output_size),
            nn.Softmax(dim=-1)
        )
    def forward(self,*args,**kwargs):return self.fn(*args,**kwargs)

class RegressionModel(nn.Module):
    def __init__(self,input_size):
        super(self.__class__,self).__init__()
        self.fn = nn.Sequential(
            nn.Linear(input_size,input_size//2),
            nn.Tanh(),
            nn.Linear(input_size//2,1)
        )
    def forward(self,*args,**kwargs):return self.fn(*args,**kwargs)

class GenericDataSet(torch.utils.data.Dataset):
    def __init__(self,x,y,x_dtype=torch.Tensor,y_dtype=torch.Tensor):
        super(self.__class__,self).__init__()
        self.x = x_dtype(x)
        self.y = y_dtype(y)
        self.ds = zip( x,y )
        self.length = len(x)
        self.input_size = x.size()[-1]
        self.output_size = y.size()[-1]

    def __getitem__(self,index): return next( self.ds )

    def __len__(self):return self.length

class Dataset:

    _text_regression        = None
    _text_classification    = None
    _image_regression       = None
    _image_classiication    = None
    _tabular_regression     = None
    _tabular_classification = None


    @property
    def tabular_classification(self):
        from sklearn.datasets import make_classification
        if self._tabular_classification is None:
            x,y = make_classification(1000)
            self._tabular_classification = GenericDataSet(x,y,y_dtype=torch.LongTensor)
        return self._tabular_classification
    
    @property
    def tabular_regression(self):
        from sklearn.datasets import make_regression
        if self._tabular_regression is None:
            x,y = make_regression(1000)
            self._tabular_regression = GenericDataSet(x,y)
        return self._tabular_regression

    @property
    def text_regression(self):
        if self._text_regression is None:
            from torchtext.datasets import PennTreebank
            from torch.distributions import Exponential,Cauchy,Gamma
            def new_dataset(dataset):
                length = len(dataset)
                data = torch.zeros( length )
                indexes = torch.arange( length )[ torch.randperm( length ) ]
                data[indexes[:length//3]] = Cauchy( torch.tensor([0.]), torch.tensor([1.])).sample( length//3 )
                data[indexes[length//3:(length//3)*2]] = Exponential(  torch.tensor([1.])).sample( length//3 )
                data[indexes[:-(length//3)]] = Gamma( torch.tensor([0.8]), torch.tensor([1.65])).sample( length//3 )
                data = data[:(length//3)*3]
                
                return GenericDataSet( dataset,data )
            self._text_regression = new_dataset( PennTreebank('./data') )
        return self._text_regression
    
    @property
    def text_classification(self):
        if self._text_classification is None:
            import random
            WORDS = open("/usr/share/dict/words").read().splitlines()
            def rand_sentence():return " ".join(random.choice(WORDS) for _ in range(random.randint(5,26)) )
            class ClassificationDataSet(torch.utils.data.Dataset):
                def __init__(self):
                    super(self.__class__,self).__init__()
                    self.words = [ rand_sentence() for _ in range(1000) ]
                    self.labels = [ random.randint(0,16) for _ in range( 1000) ]
                def __getitem__(self,index):return (self.words[index],self.labels[index])
                def __len__(self):return 1000
            self._text_classification = ClassificationDataSet()
        return self._text_classification
            
    