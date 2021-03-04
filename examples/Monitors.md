# Monitors
In metrician, Monitors are a set of tools that monitor inputs, outputs, losses, weights and more. They act a watchdog or supervisor for the particular instance you wish to monitor and wait for an event. For example, you can monitor the loss of your model and when your model encounters an example that it does quite poorly on, it will trigger an event, which you can ten use to capture specific data fields like the input to the model so you can review later.

## Basic Usage
Let's take a look at the minimal use case for a Monitor. In this example, first we'll have a `MomentOutlier Signal` listening to our data stream, then we'll use the `OutOfDistribution` class to monitor attend to the signal and log the outliers.
```python
#imports
import torch
from metrician.monitors import OutOfDistribution

# define variables
N_DATA_POINTS 		= 1000
N_OUTLIERS			= 89
X 					= torch.randn( N_DATA_POINTS )
Y					= torch.randn( N_DATA_POINTS )
Yhat     			= torch.randn( N_DATA_POINTS )
outlier_index 		= torch.randint( 0, N_DATA_POINTS, N_OUTLIERS )
monitor 			= OutOfDistribution( X ) 


# shift the data points so they are outliers
Yhat[outlier_index]	/= torch.randn( N_OUTLIERS )+torch.randn( N_OUTLIERS ) 

# pass through the data and record the indexes of the outliers
# we are mimicking the loss being passed to the Monitor with the difference between y and yhat, 
# but this could just as easily be the input or output of the model
for index,y,yhat in enumerate( zip(Y,Yhat): monitor( y-yhat )
found_outliers = monitor.captured_signals

# check overlap
overlap = set( found_outliers ).intersection(
											outlier_index.numpy().tolist() 
									)
# percentage recovered
print( f'recovered: {round( (len(overlap)/len(outlier_index) )*100 )}%' )
``` 

## Advanced Usage
All monitor classes have a default `Signal` and `Recorder` class, but for instances where you'll need more advanced configuration, you may not be able to rely on an out-of-the-box Monitor class. For that you'll need to either assumble your monitor with existing pieces or build them yourself. Let's look at how we might be able to do that with a fake text classification task. 
```python
import torch
from metrician.signals import MomentOutlier,Loss
from metrician.recorders import BertIndex2Text,Distribution
from metrician.monitors import (
								OutOfDistribution,
								MedianEstimator,
								LowVariance
							)
# define variables
N_DATA_POINTS 		= 1000
N_OUTLIERS			= 89
VOCAB_SIZE			= 32000
N_CLASSES			= 16
SENTENCE_LENGTH		= 25

X 					= torch.randint( 0, VOCAB_SIZE, (N_DATA_POINTS,N_WORDS) )
Y					= torch.randint( 0, N_CLASSES, N_DATA_POINTS )
Yhat     			= torch.randint( 0, N_CLASSES, N_DATA_POINTS )
outlier_index 		= torch.randint( 0, N_DATA_POINTS, N_OUTLIERS )

ood_monitor			= OutOfDistribution( 
										dataset=X,
										signal=MomentOutlier(),
										recorder=BertIndex2Text()
					) 
low_var_monitor		= LowVariance( 
										dataset=X,
										signal=Loss(),
										recorder=BertIndex2Text()
					) 
median_monitor		= OutOfDistribution( 
										dataset=X,
										signal=[MomentOutlier(),Loss()],
										recorder=[BertIndex2Text(),Distribution(bins=VOCAB_SIZE)]
					) 

```
Now, let's walkthrough each instance of a Monitor and what the expected behavior that that class is.
### ood_monitor
### low_var_monitor
### median_monitor