import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
torch.backends.cudnn.benchmark = True
from torch import nn
from torch.utils.data import Dataset

from torchvision import transforms
import PIL.Image
import numpy as np
from tqdm import tqdm

# matplotlib.use('TkAgg')

class Vggface2(Dataset):
    def __init__(self, root):
        self._root = root
        self._classes, self._class_to_idx = self._find_classes()
        self._samples = self._make_dataset()
        print(f'total num of classes: {len(self._classes)}')
        
        self.transform = transforms.Compose(
            [
             transforms.Resize(112), # resize to network required size while not distorting
             transforms.CenterCrop(112), # crop to square
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])


    def _find_classes(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self._root) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._root) if os.path.isdir(os.path.join(self._root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self):
        images = []
        dir = os.path.expanduser(self._root)
        progress_bar = tqdm(
                        sorted(self._class_to_idx.keys()),
                        desc='Making data training set',
                        total=len(self._class_to_idx.keys()),
                        leave=False
                    )
        for target in progress_bar:
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self._class_to_idx[target])
                    images.append(item)
            progress_bar.update(n=1)
        progress_bar.close()
        return images

    def __getitem__(self, index):
        img_path, label = self._samples[index]
        img = PIL.Image.open(img_path)
        img_lip = [augmentation(img)]# for _ in range(self.num_img_lip)] # [imgLR1, imgLR2, imgLR3]
        #img_lip.append(surveillance_augmentation(sample))# for _ in range(self.num_img_lip)])
        img_lip.insert(0, img) # [imgHR, imgLR1, imgLR2, imgLR3]
        
        if self.transform is not None:
            img_lip = [self.transform(img) for img in img_lip]
        
        return torch.stack(img_lip), label

    def __len__(self):
        return len(self._samples)
        
def augmentation(img):
    """ resize the image to a small size, add compression noise, and enlarge it back """
    w, h = img.width, img.height
    # side_ratio = np.random.uniform(0.1, 1.0) # 11.2~
    side_ratio = 1/7
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