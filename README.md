# Metrician
![GitHub all releases](https://img.shields.io/github/downloads/tedtroxell/metrician/total?logo=github&style=flat-square)
![GitHub](https://img.shields.io/github/license/tedtroxell/metrician?style=flat-square)


Metrician is a small lightweight module that easily and automatically records performance metrics for you. 
Right now, Metrician is only compatible with Pytorch. There are a lot of great tools out there like [Pytoch Lighting](https://github.com/PyTorchLightning/pytorch-lightning) that have built in loggers. Although there is nothing wrong with these tools (in fact, I'm a big fan), I find myself writing the same few lines of code over and over and over again, just to log the same metrics. I hate redundancy, so I wrote a tool to fix that issue. You're still able to use the same libraries, but now you don't have to code as much.

## Usage
```python
import torch
from torch import nn
from metrician import MetricWriter, Defaults

# initialize data
X = torch.randn( 10, 10 ) # input size 10
Y = torch.randn( 10,4 ) # 4 classes

# initialize simple model, loss and optimizer
model = nn.Linear(10,4)
optim = torch.optim.SGD(model.parameters())
loss_fn = nn.L1Loss()

# initialize Metric Writer
mw = MetricWriter( Defaults.SimpleClf ) # initialize with basic classifier metrics

# inside your training loop
for _ in range(EPOCHS):
	y = model( X )
	# you can either pass the loss function into the metric writer to automatically
	# record the loss or do that seperately. If you pass in the loss, it will automatically
	# call your loss function for you i.e. (loss = loss_fn(yhat,y) ) and return the result
	# if you are recording every couple of steps/batches it's recommended that you do not pass
	# the loss function to the metric writer
	loss = mw(y,Y,loss_fn)
	loss.backward()
	optim.step()
```

### Feeling Lazy?
With Metrician you can leverage existing class methods to automatically create a config, reducing the amount of code you need.
```python
import torch
from torch import nn
from metrician import MetricWriter, Defaults

# initialize data
X = torch.randn( 10, 10 ) # input size 10
Y = torch.randn( 10,4 ) # 4 classes

# initialize simple classification model
model = nn.Linear(10,4)
# initialize Metric Writer
mw = MetricWriter.from_model( model )
# print( mw.cfg.config_type )
# >>> classification

# or

# initialize simple regression model
model = nn.Linear(10,1)
# initialize Metric Writer
mw = MetricWriter.from_model( model )
# print( mw.cfg.config_type )
# >>> regression
...
```

### Want to integrate it into another library like Pytorch Lightening?
Metrician is an easy module to integrate into any existing Pytorch library/framework. 

```python
def __init__(self):
    ...
    self.accuracy = pl.metrics.Accuracy()
	self.mw = MetricWriter( Defaults.SimpleCLF )
def training_step(self, batch, batch_idx):
    x, y = batch
    preds = self(x)
	# instead of normally typing
	# self.log('my_loss', loss,)
	# ...
	# self.log('my_accuracy',self.accuracy(preds, y))
	# instead type
	self.mw( (preds, y) )

```

## TODOs
* add histograms, embeddings and graphs
* Tensorflow Support
* ?????
* profit
