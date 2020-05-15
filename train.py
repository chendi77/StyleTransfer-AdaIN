from tqdm import tqdm
from dataloader import dataset_iter
from options import Options
from model import AdainNetwork
from util import Util


def main():
    opt = Options()
    util = Util(opt)
    util.ensure_working_dirs()
    device = util.get_device()

    content_dataset = dataset_iter(opt.content_dir, opt)
    style_dataset = dataset_iter(opt.style_dir, opt)

    adain_net = AdainNetwork.create_network(opt, device)
    optimizer = util.create_optimizer(adain_net)

    for i in tqdm(range(opt.max_iter)):
        util.update_learning_rate(i, optimizer)
        content_images = next(content_dataset).to(device)
        style_images = next(style_dataset).to(device)

        print(f'Iter: {i}', flush=True)
        optimizer.zero_grad()
        output_images, loss = adain_net(content_images, style_images)
        loss.backward()
        optimizer.step()

        if i % opt.save_interval == 0:
            util.save_network_and_images(i, adain_net,
                                         content_images, style_images, output_images)


if __name__ == '__main__':
    main()
