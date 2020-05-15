import torch
import numpy as np
import os
from torch.optim import Adam
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Trainer类负责管理model和optimizer
# 更新网络权重和计算loss
class Util:
    def __init__(self, opt):
        self.opt = opt

    # 优化器的创建（仅在train时调用）
    def create_optimizer(self, adain_net):
        decoder_params = list(adain_net.decoder.parameters())
        optimizer = Adam(decoder_params, lr=self.opt.learning_rate)
        return optimizer

    # 更新学习率learning rate decay
    def update_learning_rate(self, iter, optimizer):
        new_lr = self.opt.learning_rate / (1 + iter * self.opt.lr_decay)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    # 确保所有文件夹存在
    def ensure_working_dirs(self):
        dir_list = [self.opt.content_dir, self.opt.style_dir, self.opt.checkpoints_dir]
        for dir in dir_list:
            Util.ensure_dir(dir)

    def get_device(self):
        device_id = self.opt.gpu_id
        use_gpu = self.opt.gpu_id >= 0 and torch.cuda.is_available()
        device_name = f'cuda:{device_id}' if use_gpu else 'cpu'
        return torch.device(device_name)

    def save_network_and_images(self, iter, adain_net, c_images, s_images, o_images):
        net_dir = Util.ensure_dir(os.path.join(self.opt.checkpoints_dir, f'iteration_{iter}'))
        network_path = os.path.join(net_dir, f'decoder_{iter}.pth')
        decoder_weights = adain_net.decoder.state_dict()
        torch.save(decoder_weights, network_path)

        image_dirs = ['content', 'style', 'output']
        image_types = [c_images, s_images, o_images]
        for image_dir_name, images in zip(image_dirs, image_types):
            for i, image_tensor in enumerate(images):
                image_dir = self.ensure_dir(os.path.join(net_dir, image_dir_name))
                image_name = f'iter{iter}_{image_dir_name}{i}'
                image_path = os.path.join(image_dir, f'{image_name}.png')
                self.save_image(image_tensor, image_path)

    def save_image(self, image_tensor, image_path):
        image_dir = '/'.join(image_path.split('/')[:-1])
        self.ensure_dir(image_dir)
        image_numpy = image_tensor.detach().cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        image_numpy = (image_numpy * self.opt.std + self.opt.mean) * 255.0
        image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_numpy)
        image.save(image_path)

    @staticmethod
    def ensure_dir(dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        return dir_path



