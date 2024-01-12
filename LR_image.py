import numpy as np
import PIL
from io import BytesIO
import cv2 as cv
from PIL import Image, ImageFilter

np.random.seed(42)


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
    side_ratio = 0.5#np.random.uniform(0.0625, 0.375) # 7 ~ 56
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



def new_degradation(img):

    def apply_random_blur(image):
        # Apply random blur
        kernel_size = np.random.choice([3, 5, 7])  # Randomly choose kernel size
        return image.filter(ImageFilter.GaussianBlur(radius=kernel_size))

    def apply_synthetic_noise(image):
        # Apply synthetic noise
        image_array = np.array(image)
        noise = np.random.normal(loc=0, scale=1, size=image_array.shape)
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
    side_ratio = 0.5#np.random.uniform(0.0625, 0.375) # 7 ~ 56
    temp_1 = make_temp(img)
    temp_1.save('LR_images/temp_1.png')

    img = apply_random_blur(img)
    img.save('LR_images/apply_random_blur.png')

    p2 = combine(img, None)
    p2.save('LR_images/p2.png')

    temp_2 = make_temp(p2)
    temp_2.save('LR_images/temp_2.png')

    p3 = combine(apply_synthetic_noise(p2), None)
    p2.save('LR_images/p3.png')

    temp_3 = make_temp(p3)
    temp_3.save('LR_images/temp_3.png')

    # add compression noise
    aug_img = combine(random_JPEG_compression(p3), None)
    aug_img.save('LR_images/aug_img.png')

    aug_img = downscale_bicubic(aug_img, side_ratio)
    
    return aug_img

def read_image(path):
    return  Image.open(path).convert('RGB')

from torchvision import transforms
import os
import mxnet as mx
from torch.utils.data import DataLoader, Dataset
import numbers
import torch
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
        img_lip.insert(0, sample) # [imgHR, imgLR1, imgLR2, imgLR3]
        
        if self.transform is not None:
            img_lip = [self.transform(img) for img in img_lip]
        
        return torch.stack(img_lip), label


if __name__ == '__main__':
    path = '/mnt/HDD1/phudh/ICIP/insightface_lipface/data_test/SCface_landmarkCrop_match_Resolution/gallery/001_frontal.jpg'
    img = read_image(path)

    lr_img = surveillance_augmentation(img)
    lr_new = new_degradation(img)
    print(lr_img.size, lr_new.size)
    lr_img.save('LR_images/test.png')
    lr_new.save('LR_images/test_new.png')

    lr_ref_path = '/mnt/HDD1/phudh/ICIP/insightface_lipface/data_test/SCface_landmarkCrop_match_Resolution/probe_d1/001_cam1_1.jpg'
    lr_ref = read_image(lr_ref_path)
    lr_ref = lr_ref.resize(lr_img.size, Image.BICUBIC)
    lr_ref.save('LR_images/ref_lr.png')
    print(lr_ref.size)

