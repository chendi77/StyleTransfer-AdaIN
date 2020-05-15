import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from model_methods import *


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19_pretrained = vgg19(pretrained=True).features
        self.slice1 = vgg19_pretrained[:2]
        self.slice2 = vgg19_pretrained[2:7]
        self.slice3 = vgg19_pretrained[7:12]
        self.slice4 = vgg19_pretrained[12:21]
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, intermediate=False):
        h1 = self.slice1(X)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4] if intermediate else h4


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.ReflectionConv(512, 256)
        self.conv2 = self.ReflectionConv(256, 256)
        self.conv3 = self.ReflectionConv(256, 256)
        self.conv4 = self.ReflectionConv(256, 256)
        self.conv5 = self.ReflectionConv(256, 128)
        self.conv6 = self.ReflectionConv(128, 128)
        self.conv7 = self.ReflectionConv(128, 64)
        self.conv8 = self.ReflectionConv(64, 64)
        self.conv9 = self.ReflectionConv(64, 3, activation=False)
        self.upsample = nn.Upsample(scale_factor=2)

    def ReflectionConv(self, in_channels, out_channels, kernel_size=3, padding=1, activation=True):
        modules = [nn.ReflectionPad2d(padding),
                   nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)]
        if activation:
            modules.append(nn.ReLU())
        return nn.Sequential(*modules)

    def forward(self, X):
        h = self.conv1(X)
        h = self.upsample(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.upsample(h)
        h = self.conv6(h)
        h = self.conv7(h)
        h = self.upsample(h)
        h = self.conv8(h)
        h = self.conv9(h)
        return h


class AdainNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = Decoder()
        self.opt = opt

    @staticmethod
    def create_network(opt, device):
        model = AdainNetwork(opt).to(device)
        if opt.model_path is not None:
            weights = torch.load(opt.model_path)
            model.decoder.load_state_dict(weights)
        return model

    def generate(self, content_images, style_images):
        style_feats = self.encoder(style_images, intermediate=False)
        content_feats = self.encoder(content_images, intermediate=False)
        target = AdaIN(content_feats, style_feats)
        style_control = self.opt.alpha * target + (1 - self.opt.alpha) * content_feats
        output_images = self.decoder(style_control).squeeze()
        return output_images

    def forward(self, content_images, style_images):
        style_intermediate = self.encoder(style_images, intermediate=True)
        style_feats = style_intermediate[-1]
        content_feats = self.encoder(content_images, intermediate=False)

        target = AdaIN(content_feats, style_feats)
        style_control = self.opt.alpha * target + (1 - self.opt.alpha) * content_feats
        output_images = self.decoder(style_control)

        output_intermediate = self.encoder(output_images, intermediate=True)
        output_feats = output_intermediate[-1]

        loss_c = content_loss(output_feats, target)
        loss_s = style_loss(output_intermediate, style_intermediate)
        loss = loss_c + self.opt.lamda * loss_s

        print(f'Content Loss: {loss_c}, Style Loss: {loss_s}, Loss: {loss}')
        return output_images, loss

