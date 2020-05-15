import os
import argparse
from PIL import Image
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image
from options import Options
from model import AdainNetwork
from util import Util


def load_image_tensor(image_path, device, opt):
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=opt.mean, std=opt.std)])
    image = Image.open(image_path).convert('RGB')
    image_tensor = trans(image).unsqueeze(0).to(device)
    return image_tensor


def main():
    opt = Options()
    util = Util(opt)
    device = util.get_device()
    content_image = load_image_tensor(opt.content_image, device, opt)
    style = load_image_tensor(opt.style_image, device, opt)
    adain_net = AdainNetwork.create_network(opt, device)

    with torch.no_grad():
        output_image = adain_net.generate(content_image, style)
        util.save_image(output_image, opt.result_image)


if __name__ == '__main__':
    main()
