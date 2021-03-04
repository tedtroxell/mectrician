
# Signals
At it's core, metrician is based around capturing signals, recording and explaining signals during model training. So what do we mean by a _signal_? A signal can be a lot of things, for example it could be the loss of your objective function, it could be the variance of your input space or the average precision of a classifier. With Metrician you can treat data as a signal and create events around the behavior of that signal.

## Basic Usage
Let's take a look at the minimal use case for a Signal. In this example, we'll have a `MomentOutlier` Signal attend to the data being passed through it and log the outliers as it sees the data.
```python
#imports
import torch
from metrician.signals import MomentOutlier

# define variables
N_DATA_POINTS 		= 1000
N_OUTLIERS			= 89
X 					= torch.randn( N_DATA_POINTS )
outlier_index 		= torch.randint( 0, N_DATA_POINTS, N_OUTLIERS )
monitor 			= MomentOutlier()

# shift the data points so they are outliers
X[outlier_index]	/= torch.randn( N_OUTLIERS )+torch.randn( N_OUTLIERS ) 

# pass through the data and record the indexes of the outliers
found_outliers = [ i for i,x in enumerate(X) if monitor(x) > 0 ]

# check overlap
overlap = set( found_outliers ).intersection(
											outlier_index.numpy().tolist() 
									)
# percentage recovered
print( f'recovered: {round( (len(overlap)/len(outlier_index) )*100 )}%' )
``` 

## Advanced Usage
