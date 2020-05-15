from torch.utils import data
from dataset import AdaINDataset


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.dataset_size = len(data_source)

    def __iter__(self):
        return iter(self.infinite_iteration(self.dataset_size))

    def __len__(self):
        return 2 ** 31

    def infinite_iteration(self, n):
        i = 0
        while True:
            yield i
            i = (i + 1) % n


def dataset_iter(image_dir, opt):
    dataset = AdaINDataset(image_dir, opt)
    images = iter(data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=InfiniteSampler(dataset),
        num_workers=16
    ))
    return images
