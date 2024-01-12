import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn

import PIL
import random
from io import BytesIO
from PIL import Image, ImageFilter


def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    seed = 2048,
    num_workers = 2,
    num_img_lip = 1,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    elif os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank, num_img_lip=num_img_lip)

    # Image Folder
    else:
        transform = transforms.Compose([
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        train_set = ImageFolder(root_dir, transform)

    # DALI
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank, num_img_lip):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [#transforms.ToPILImage(),
             #transforms.RandomHorizontalFlip(),
             transforms.Resize(112),
             transforms.CenterCrop(112),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
        self.num_img_lip = num_img_lip
        '''
        print(len(self.imgidx))
        idx = self.imgidx[2688237]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        print(label)
        '''

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        filtered_label = np.array(label, dtype=np.int64)
        label = torch.tensor(filtered_label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        """
        sample = transforms.ToPILImage()(sample) # PIL RGB
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label
        """
        # dataset
        sample = transforms.ToPILImage()(sample) # PIL RGB
        # img_lip = list()
        # img_lip = [augmentation(sample)]# for _ in range(self.num_img_lip)] # [imgLR1, imgLR2, imgLR3]

        img_lip = [surveillance_augmentation(sample) for _ in range(self.num_img_lip)]
        # img_lip = [new_degradation(sample) for _ in range(self.num_img_lip)]

        img_lip.insert(0, sample) # [imgHR, imgLR1, imgLR2, imgLR3]
        
        if self.transform is not None:
            img_lip = [self.transform(img) for img in img_lip]
        
        return torch.stack(img_lip), label
        
    def __len__(self):
        return len(self.imgidx)

def new_degradation(img):

    def apply_random_blur(image):
        # Apply random blur
        kernel_size = np.random.choice([3, 5, 7])  # Randomly choose kernel size
        return image.filter(ImageFilter.GaussianBlur(radius=kernel_size))

    def apply_synthetic_noise(image):
        # Apply synthetic noise
        image_array = np.array(image)
        noise = np.random.normal(loc=0, scale=2, size=image_array.shape)
        noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image_array)
    
    def downscale_bicubic(image, scale_factor):
        # Downscale by bicubic interpolation
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return image.resize((new_width, new_height), Image.BICUBIC)
    
    def make_temp(img):
        img_array = np.array(img)
        temp_img  = 1 - img_array
        temp_img = np.clip(temp_img, 0, 255)

        return Image.fromarray(temp_img.astype(np.uint8))
    
    def combine(img1, img2):
        if img2 is None:
            return img1
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        sum_array = img1_array + img2_array

        sum_array = np.clip(sum_array, 0, 255)

        return Image.fromarray(sum_array.astype(np.uint8))


    w, h = img.width, img.height
    # side_ratio = np.random.uniform(0.1, 1.0) # 11.2 ~ 112
    side_ratio = np.random.uniform(0.0625, 0.375) # 7 ~ 56
    # temp_1 = make_temp(img)

    img = apply_random_blur(img)

    p2 = combine(img, None)

    # temp_2 = make_temp(p2)

    p3 = combine(apply_synthetic_noise(p2), None)

    # temp_3 = make_temp(p3)

    # add compression noise
    aug_img = combine(random_JPEG_compression(p3), None)

    aug_img = downscale_bicubic(aug_img, side_ratio)
    
    return aug_img

def augmentation(img):
    """ resize the image to a small size, add compression noise, and enlarge it back """
    w, h = img.width, img.height
    # side_ratio = np.random.uniform(0.1, 1.0) # 11.2~
    side_ratio = np.random.uniform(0.2, 1.0)
    # resize to small
    # interpolation =  PIL.Image.BILINEAR
    interpolation = np.random.choice(
        [PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC, PIL.Image.LANCZOS])
    small_img = img.resize((int(w*side_ratio), int(h*side_ratio)), interpolation)
    # add compression noise
    #small_img = random_JPEG_compression(small_img)
    interpolation = np.random.choice(
        [PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC, PIL.Image.LANCZOS])
    # enlarge back
    aug_img = small_img.resize((w, h), interpolation)
    return aug_img


def random_JPEG_compression(img):
    #qf = random.randrange(50, 100)
    qf = 65
    outputIoStream = BytesIO()
    img.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return PIL.Image.open(outputIoStream)


def surveillance_augmentation(img):
    w, h = img.width, img.height
    # side_ratio = np.random.uniform(0.1, 1.0) # 11.2 ~ 112
    side_ratio = np.random.uniform(0.0625, 0.375) # 7 ~ 56
    # side_ratio = np.random.choice([0.0625, 0.125, 0.25, 0.375, 0.5])
    # resize to small
    # interpolation =  PIL.Image.BILINEAR
    interpolation = PIL.Image.NEAREST
    aug_img = img.resize((int(w*side_ratio), int(h*side_ratio)), interpolation)
    # add compression noise
    aug_img = random_JPEG_compression(aug_img)
    # interpolation = np.random.choice(
    #     [PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC, PIL.Image.LANCZOS])
    # interpolation = PIL.Image.BILINEAR
    # enlarge back
    # aug_img = aug_img.resize((w, h), interpolation)
    return aug_img

class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5)):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()


if __name__ == "__main__":
    test_loader = get_dataloader(
        "/mnt/HDD1/yuwei/dataset/MS1MV3/ms1m-retinaface-t1/",
        #args.local_rank,
        0,
        128,
        False,
        2048,
        2
    )