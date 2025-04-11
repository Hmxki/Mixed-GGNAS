import os
import random
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from Mixed_GGNAS.utils.data_enhancement import *
import torch.nn.functional as F


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, "H:\\hmx\\代码\\DE_github\\data\\idrid", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images"))]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        #self.manual = [os.path.join(data_root, "masks", i) for i in img_names]
        self.manual = [os.path.join(data_root, "masks", i.split('.')[0] + "_OD.tif") for i in img_names]
        #self.manual = [os.path.join(data_root, "masks", i.split('.')[0] + '_mask.png') for i in img_names]
        #self.manual = [os.path.join(data_root, "masks", i.split('.')[0][:11]+"mask_"+i.split('.')[0].split('_')[2]+'.png') for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")


    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        mask1 = manual.resize((40,64))
        mask2 = manual.resize((80, 128))
        mask3 = manual.resize((160, 256))

        # img = np.array(img) / 255
        manual = np.array(manual) / 255


        # original_image = np.array(img)
        # original_image = torch.tensor(original_image, dtype=torch.float32)/255.  # 归一化到 [0, 1]
        # if original_image.ndim == 2:  # 如果是灰度图，扩展维度
        #     original_image = original_image.unsqueeze(0)
        # else:  # 如果是RGB图像，转置到 (C, H, W)
        #     original_image = original_image.permute(2, 0, 1)
        #
        # mask_manual = torch.tensor(manual, dtype=torch.float32)/255.
        # mask_manual = mask_manual.unsqueeze(0)
        # Intensity_noise_transform = RandIntensityDisturbance(p=1, brightness_limit=0.5, contrast_limit=0.5, clip=False)
        # gaussian_noise_transform = RandGaussianNoise(p=1, mean=0., std=0.1, clip=True)
        # masking_generator = RandomMaskingGenerator(input_size=(original_image.shape[1], original_image.shape[2]),
        #                                            mask_ratio=0.85, block_size=(15, 15))
        #
        # transformed_image1 = gaussian_noise_transform.forward_image(original_image)
        # transformed_image2 = Intensity_noise_transform.forward_image(original_image)
        #
        # # Apply Random Mask
        # mask = masking_generator()
        # mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).repeat(original_image.shape[0], 1, 1)
        # masked_image1 = transformed_image1 * mask_tensor
        # masked_image2 = transformed_image2 * mask_tensor
        # mask_manual = mask_manual * mask_tensor
        #
        #
        # # Convert to PIL images for dispmasked_image1lay
        # #original_pil = tensor_to_pil(original_image)
        # masked_image1 = tensor_to_pil(masked_image1)
        # masked_image2 = tensor_to_pil(masked_image2)
        # mask_manual = tensor_to_pil(mask_manual)

        # Display images
        #display_images(mask_manual, masked_image1, masked_image2)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        # img = Image.fromarray(img)
        mask = Image.fromarray(manual)

        if self.transforms is not None:
            img_, mask = self.transforms(img, mask)
            temp = mask.unsqueeze(0).unsqueeze(0).to(torch.float32)
            # _, mask_manual = self.transforms(img, mask_manual)
            # masked_image1,masked_image2 = self.transforms(masked_image1, None), self.transforms(masked_image2, None)
            mask1 = F.interpolate(temp, size=(40, 64), mode='nearest').to(torch.int64).squeeze(0).squeeze(0)
            mask2 = F.interpolate(temp, size=(80, 128), mode='nearest').to(torch.int64).squeeze(0).squeeze(0)
            mask3 = F.interpolate(temp, size=(160, 256), mode='nearest').to(torch.int64).squeeze(0).squeeze(0)
        return img_, mask, mask1, mask2, mask3

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets, mask1, mask2, mask3 = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        # mask_image1 = cat_list(mask_image1, fill_value=0)
        # mask_image2 = cat_list(mask_image2, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        #mask_manual = cat_list(mask_manual, fill_value=255)
        mask1 = cat_list(mask1, fill_value=255)
        mask2 = cat_list(mask2, fill_value=255)
        mask3 = cat_list(mask3, fill_value=255)
        return batched_imgs, batched_targets, mask1, mask2, mask3


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

