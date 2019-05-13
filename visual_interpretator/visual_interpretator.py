from algorithms.cnn_layer_visualization import CNNLayerVisualization
from algorithms.deep_dream import DeepDream

from algorithms.gradcam import GradCam
from algorithms.vanilla_backprop import VanillaBackprop
from algorithms.guided_backprop import GuidedBackprop
from algorithms.smooth_grad import generate_smooth_grad

import PIL
from PIL import Image

from utils.misc_functions import get_example_params, recreate_image, save_image,\
                            save_class_activation_images, apply_colormap_on_image
from utils.misc_functions import format_np_output, save_gradient_images, convert_to_grayscale

from torch.autograd import Variable

import torch
import numpy as np
from torchvision.models import alexnet

import matplotlib.pyplot as plt

class VisualInterpretator():

    def __init__(self, model, transforms=None, apply_transform=True, device=torch.device('cpu')):

        self.device = device
        self.apply_transform = apply_transform
        self.transforms = transforms

        self.model = model
        self.model = self.model.to(self.device)

        self.cam_heatmaps = []
        self.grads = []

    def gradcam(self, image, target_layer, target_class):

        if isinstance(image, PIL.Image.Image):
            if self.apply_transform:
                tensor_image = self.transforms(image).unsqueeze(0)
            else:
                tensor_image = torch.Tensor(np.array(image)).unsqueeze(0).permute(0, 3, 1, 2)

        if isinstance(image, np.ndarray):
            if self.apply_transform:
                tensor_image = self.transforms(Image.fromarray(image)).unsqueeze(0)
            else:
                tensor_image = torch.Tensor(image).unsqueeze(0).permute(0, 3, 1, 2)

        tensor_image = tensor_image.to(self.device)

        grad_cam = GradCam(self.model, target_layer)
        cam_activation_map = grad_cam.generate_cam(tensor_image, target_class)

        heatmap, heatmap_on_image = apply_colormap_on_image(image, cam_activation_map, 'hsv')

        cam_activation_map = Image.fromarray(format_np_output(cam_activation_map))

        #heatmap = Image.fromarray(format_np_output(heatmap))

        #heatmap_on_image = Image.fromarray(format_np_output(heatmap_on_image))

        self.cam_heatmaps = [cam_activation_map, heatmap, heatmap_on_image]

        return self.cam_heatmaps

    def smooth_grad(self, image, target_class, param_n=50, param_sigma_multiplier=4):

        if isinstance(image, PIL.Image.Image):
            if self.apply_transform:
                tensor_image = self.transforms(image).unsqueeze(0)
            else:
                tensor_image = torch.Tensor(np.array(image)).unsqueeze(0).permute(0, 3, 1, 2)

        if isinstance(image, np.ndarray):
            if self.apply_transform:
                tensor_image = self.transforms(Image.fromarray(image)).unsqueeze(0)
            else:
                tensor_image = torch.Tensor(image).unsqueeze(0).permute(0, 3, 1, 2)

        tensor_image.requires_grad = True
        tensor_image = tensor_image.to(self.device)

        vbp = VanillaBackprop(self.model)
        gbp = GuidedBackprop(self.model)

        vanilla_smooth_grad = generate_smooth_grad(vbp, tensor_image, target_class, param_n, param_sigma_multiplier)
        guided_grad = gbp.generate_gradients(tensor_image, target_class)

        guided_grad = guided_grad - guided_grad.min()
        guided_grad /= guided_grad.max()

        vanilla_smooth_grad = Image.fromarray(format_np_output(vanilla_smooth_grad.squeeze()))
        guided_smooth_grad = Image.fromarray(format_np_output(guided_grad.squeeze()))

        self.grads = [vanilla_smooth_grad, guided_smooth_grad]

        return self.grads

    def cnn_vis_layers(self, target_layer, target_position, shape=(256, 256, 3), epochs=300):

        layer_vis = CNNLayerVisualization(self.model.features, shape, target_layer, target_position)

        output = layer_vis.visualise_layer_with_hooks(epochs=epochs)
        output = Image.fromarray(output)

        return output

    def visualization(self, image, target_layer, target_class, figsize=(15, 15)):

        self.gradcam(image, target_layer, target_class)
        self.smooth_grad(image, target_class)

        fig = plt.figure(figsize=figsize, dpi=150)
        ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
        ax1.axis('off')
        ax1.set_title('Original image')
        ax1.imshow(image)

        ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=1, rowspan=1)
        ax2.axis('off')
        ax2.set_title('GradCam')
        ax2.imshow(self.cam_heatmaps[1])

        ax3 = plt.subplot2grid((4, 4), (0, 3), colspan=1, rowspan=1)
        ax3.axis('off')
        ax3.set_title('GradCam + image')
        ax3.imshow(self.cam_heatmaps[2])

        ax4 = plt.subplot2grid((4, 4), (1, 2), colspan=1, rowspan=1)
        ax4.axis('off')
        ax4.set_title('Vanilla gradient (high contrast)')
        ax4.imshow(self.grads[0])

        ax5 = plt.subplot2grid((4, 4), (1, 3), colspan=1, rowspan=1)
        ax5.axis('off')
        ax5.set_title('Guided gradient')
        ax5.imshow(self.grads[1])
