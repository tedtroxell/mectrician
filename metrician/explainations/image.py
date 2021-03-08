import torch
from lime.lime_image import ImageExplanation
from .interface import BaseExplainerInterface

from collections import OrderedDict, Sequence
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.cm as cm

class _BaseWrapper(object):
    """

    :param object: [description]
    :type object: [type]
    """    

    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.model._forward_hooks = OrderedDict()
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0 )
        return one_hot

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        """
        Class-specific backpropagation

        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. (self.logits * one_hot).sum().backward(retain_graph=True)
        """
        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GuidedBackPropagation(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class Deconvnet(BackPropagation):
    """
    "Striving for Simplicity: the All Convolutional Net"
    https://arxiv.org/pdf/1412.6806.pdf
    Look at Figure 1 on page 8.
    """

    def __init__(self, model):
        super(Deconvnet, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients and ignore ReLU
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_out[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps
                self.fmap_pool[key] = output.detach()

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook_

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        return F.adaptive_avg_pool2d(grads, 1)

    def forward(self, image):
        
        self.image_shape = image.shape[2:]
        return super(GradCAM, self).forward(image)

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def occlusion_sensitivity(model, images, ids, mean=None, patch=35, stride=1, n_batches=128):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure A5 on page 17
    
    Originally proposed in:
    "Visualizing and Understanding Convolutional Networks"
    https://arxiv.org/abs/1311.2901
    """

    torch.set_grad_enabled(False)
    model.eval()
    mean = mean if mean else 0
    patch_H, patch_W = patch if isinstance(patch, Sequence) else (patch, patch)
    pad_H, pad_W = patch_H // 2, patch_W // 2

    # Padded image
    images = F.pad(images, (pad_W, pad_W, pad_H, pad_H), value=mean)
    B, _, H, W = images.shape
    new_H = (H - patch_H) // stride + 1
    new_W = (W - patch_W) // stride + 1

    # Prepare sampling grids
    anchors = []
    grid_h = 0
    while grid_h <= H - patch_H:
        grid_w = 0
        while grid_w <= W - patch_W:
            grid_w += stride
            anchors.append((grid_h, grid_w))
        grid_h += stride

    # Baseline score without occlusion
    baseline = model(images).detach().gather(1, ids)

    # Compute per-pixel logits
    scoremaps = []
    for i in tqdm(range(0, len(anchors), n_batches), leave=False):
        batch_images = []
        batch_ids = []
        for grid_h, grid_w in anchors[i : i + n_batches]:
            images_ = images.clone()
            images_[..., grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] = mean
            batch_images.append(images_)
            batch_ids.append(ids)
        batch_images = torch.cat(batch_images, dim=0)
        batch_ids = torch.cat(batch_ids, dim=0)
        scores = model(batch_images).detach().gather(1, batch_ids)
        scoremaps += list(torch.split(scores, B))

    diffmaps = torch.cat(scoremaps, dim=1) - baseline
    diffmaps = diffmaps.view(B, new_H, new_W)

    return diffmaps


def merge_grad_cam(gcam, raw_image,return_score=False, paper_cmap=False):
    from PIL import Image
    import torchvision.transforms.functional as TF
    gcam = gcam.cpu().numpy() #.squeeze(0).unsqueeze(-1)
    gcam = (cm.jet_r(gcam)[...,:3] * 255.0).squeeze(0)

    # print(raw_image);exit()
    
    v = gcam.mean(axis=1).mean(axis=0)

    l = ( (v.std()+v.mean() )/2./255.)
    # print( gcam.shape,raw_image.size() );exit()
    raw_image = raw_image.squeeze(0).permute(1,2,0).numpy()
    # if paper_cmap:
    #     alpha = gcam[..., None]
    #     gcam = alpha * cmap + ((1 - alpha) * raw_image)
    # else:
    #     # gcam = (( (gcam.astype(np.float)* l ) + ( (1.-l)* raw_image ) ) )/(raw_image+gcam).mean()
    #     # gcam = (.1*gcam)  + (.9*raw_image)
    #     # img =  raw_image*.9 + gcam
    #     gcam = Image.blend( Image.fromarray(gcam.astype(float)),Image.fromarray(raw_image.astype(float)),.5 )

    gcam = Image.blend( Image.fromarray(gcam.astype(np.uint8)), Image.fromarray((255*raw_image).astype(np.uint8)),.5)

    return gcam if not return_score else (gcam,v.std())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = x.size()
        x = x.view(-1, x.size(self.dim))
        
        dim = 1
        number_of_logits = x.size(dim)

        # Translate x by max for numerical stability
        x = x - torch.max(x, dim=dim, keepdim=True)[0].expand_as(x)

        # Sort x in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=x, dim=dim, descending=True)[0]
        _range = torch.arange(start=1, end=number_of_logits+1, device=device).view(1, -1).type(x.type())
        _range = _range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + _range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(x.type())
        k = torch.max(is_gt * _range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(x)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(x), x - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_x = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_x
    
    #return 

class GradCamExplainer(BaseExplainerInterface):

    def __init__(
                    self,
                    model : torch.nn.Module,
                    target_layer : str
    ):
        super(self.__class__,self).__init__(model)
        self.gcam           = GradCAM(model=model)
        self.target_layer   = target_layer

    def forward(self,image : torch.Tensor,class_label:int) -> torch.Tensor:
        """[summary]

        :param image: [description]
        :type image: torch.Tensor
        :return: [description]
        :rtype: torch.Tensor
        """        
        from pylab import figure,gca
        self.gcam.model.zero_grad()
        probs, ids = self.gcam.forward(image)
        # idxs = torch.LongTensor( [[ probs[0].argmax().item() ]] * 1 ) 
        idxs = torch.LongTensor([[ class_label ]])
        self.gcam.backward(ids=idxs)
        # self.gcam.logits.zero_grad()
        region = self.gcam.generate(self.target_layer) # region
        # region = nn.MaxPool2d(10)( region ) #torch.nn.F.grid_sample( region,  ) #transforms.ToPILImage()(region[0][0]).convert("L")
        # return region[0][0]
        img = merge_grad_cam( region[0],image )
        fig = figure()
        ax = gca()
        ax.imshow(
            img
        )
        # writer = self._get_writer()
        # writer.add_figure(
        #     'Grad Cam Explaination',
        #     fig,

        # )
        return img




class ImageExplainer(BaseExplainerInterface):
    """
    

    :param BaseExplainerInterface: [description]
    :type BaseExplainerInterface: [type]
    """    

    explainer_kwargs = {}

    def _post_process(self,explaination : ImageExplanation) -> torch.Tensor:
        pass

    def _explain(self, image : torch.Tensor ) -> ImageExplanation:
        self.explainer.explain_instance(
            image,
            num_samples=100,
            progress_bar=False
        )

