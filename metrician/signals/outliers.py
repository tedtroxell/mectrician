
from .interface import BaseSignal
from torch import Tensor
import math
class MomentOutlierSignal(BaseSignal):
    """
        Real time outlier detection for metrics

        This class records moments in order to estimate the likelihood of an outlier when data is passed to it. 
        If data may be an outlier, such as something that causes a really high error, this will record the index so you can view it later.

        NOTE:   This Signal assumes that the underlying generating function of the data is stationary. 
                If you'd like measure the outliers for your loss function, please use the "Loss" Signal.
                The Loss Signal captures a Pareto distribution - the expected value is monotonically decreasing.


        :param BaseSignal: [description]
        :type BaseSignal: [type]
    """    

    

    def __init__(self):
        super( self.__class__,self ).__init__()
        """[summary]
        """        
        self._index = self._eta = self._rho = self._tau = self._phi = 0.0
        self._min   = self._max = float('nan')
        self.n_samples_needed = 10000000

    def forward(self, x : Tensor) -> bool:
        '''
            :param x: torch.Tensor
            :returns bool: if the input tensor is an outlier
        '''       
        if self._index == 0.0:
            self._min = x
            self._max = x
        else:
            # TODO: abstract for multidimensional input
            self._min = min(self._min, x)
            self._max = max(self._max, x)
        
        
        # if not outlier: # don't include the outliers
        delta = x - self._eta
        delta_n = delta / (self._index + 1)
        delta_n2 = delta_n * delta_n
        term = delta * delta_n * self._index

        self._eta += delta_n
        self._phi += (
            term * delta_n2 * (self._index ** 2 - 3 * self._index + 3)
            + 6 * delta_n2 * self._rho
            - 4 * delta_n * self._tau
        )
        self._tau += (
            term * delta_n * (self._index - 2) - 3 * delta_n * self._rho
        )
        self._rho += term
        self._index += 1
        self.n_samples_needed = self._calc_sample_size()
        outlier = self._is_outlier(x) if self._index > self.n_samples_needed else False
        return outlier

    def _is_outlier(self, x : Tensor) -> bool:
        '''
            weight the amount of standard deviation to use based on kurtosis of the data

            :param x: Pytorch Tensor
            :returns bool: Whether the inputed tensor is an outlier or not
        '''
        m = self.mean
        std = math.sqrt( self.var )
        k = self.kurtosis
        shift = std*( 1+k )
        #               Above the threshold           |         Below the threshold
        return not bool( ( m + shift ) >= x >= ( m - shift)  )

    def _calc_sample_size(self) -> float:
        '''
            Calculate the neccessary sample size for a distribution using the z-score
        '''
        import math
        # 90% confidence interval: 1.645^2 = 2.7225
        std = math.sqrt( self._rho / (self._index -1 ) if self._index > 1 else self._index )
        return int( (2.7225 * std * (1-std))/ .01)+1

    @property
    def mean(self) -> Tensor: return self._eta

    @property
    def var(self) -> Tensor: return self._rho / (self._index - 1)

    @property
    def skewness(self) -> Tensor: return (self._index ** 0.5) * self._tau / pow(self._rho, 1.5)

    @property
    def kurtosis(self) -> Tensor: return (self._index * self._phi / (self._rho * self._rho) - 3.0)


    