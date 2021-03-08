

from unittest import TestCase

class GradCamTest(TestCase):

    @classmethod
    def setUpClass(cls):
        import torchvision.models as models
        from torchvision.datasets import FakeData,CIFAR10,STL10
        # from torch.utils.data import DataLoader
        cls.model = models.resnet18(pretrained=True)
        cls.dataset = CIFAR10('./data',download=True)# FakeData() #

    def test_gradcam(self):
        from metrician.explainations.image import GradCamExplainer
        from torchvision import transforms as T
        import random
        from pylab import show
        grad_cam = GradCamExplainer(self.model,'layer3')
        grad_cam( T.ToTensor()(self.dataset[0][0]).unsqueeze(0), random.randint(0,10))


    def test_full_gradcam_pipeline(self):pass


