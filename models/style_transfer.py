import torch
import torch.nn as nn
import torchvision.models as models


class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.selected_layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",
            "28": "conv5_1"
        }

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features[self.selected_layers[name]] = x
        return features


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (c * h * w)
