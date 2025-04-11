import torch
import random
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio, block_size):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size
        self.block_height, self.block_width = block_size

        self.num_patches = (self.height // self.block_height) * (self.width // self.block_width)
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, block size {}x{}".format(
            self.num_patches, self.num_mask, self.block_height, self.block_width
        )
        return repr_str

    def __call__(self):
        mask = np.zeros((self.height, self.width))

        blocks = [(i, j) for i in range(0, self.height, self.block_height)
                          for j in range(0, self.width, self.block_width)]

        mask_indices = np.random.choice(len(blocks), self.num_mask, replace=False)
        for idx in mask_indices:
            i, j = blocks[idx]
            mask[i:i + self.block_height, j:j + self.block_width] = 1

        return mask

class RandIntensityDisturbance:
    def __init__(self, p=0.1, brightness_limit=0.5, contrast_limit=0.5, clip=False, beta_by_max=True):
        self.beta = (-brightness_limit, brightness_limit)
        self.alpha = (1 - contrast_limit, 1 + contrast_limit)
        self.clip = clip
        self.beta_by_max = beta_by_max
        self.p = p

        self.alpha_value = None
        self.beta_value = None

        self._do_transform = False

    def randomize(self):
        if random.uniform(0, 1) < self.p:
            self._do_transform = True
            self.alpha_value = random.uniform(self.alpha[0], self.alpha[1])
            self.beta_value = random.uniform(self.beta[0], self.beta[1])

    def apply_transform(self, inputs):
        if self._do_transform:
            img_t = self.alpha_value * inputs
            if self.beta_by_max:
                img_t = img_t + self.beta_value
            else:
                img_t = img_t + self.beta_value * torch.mean(img_t)
            return torch.clamp(img_t, 0, 1) if self.clip else img_t
        else:
            return inputs

    def forward_image(self, image, randomize=True):
        if randomize:
            self.randomize()
        return self.apply_transform(image)

class RandGaussianNoise:
    def __init__(self, p=0.2, mean=0.0, std=0.1, clip=False):
        self.p = p
        self.mean = mean
        self.std = std
        self.clip = clip

        self.std_value = None
        self._do_transform = False

    def randomize(self, inputs):
        if random.uniform(0, 1) < self.p:
            self._do_transform = True
            self.std_value = random.uniform(0, self.std)
            self.noise = torch.normal(self.mean, self.std_value, size=inputs.shape)

    def apply_transform(self, inputs):
        if self._do_transform:
            added = inputs + self.noise.to(inputs.device)
            return torch.clamp(added, 0, 1) if self.clip else added
        else:
            print('1111')
            return inputs

    def forward_image(self, image, randomize=True):
        if randomize:
            self.randomize(image)
        return self.apply_transform(image)

    def invert_label(self, label_t):
        return label_t


# 读取TIF图像
def read_image(file_path):
    image = Image.open(file_path)
    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32) / 255.0  # 归一化到 [0, 1]
    if image.ndim == 2:  # 如果是灰度图，扩展维度
        image = image.unsqueeze(0)
    else:  # 如果是RGB图像，转置到 (C, H, W)
        image = image.permute(2, 0, 1)
    return image


# 将图像从张量转换回PIL格式
def tensor_to_pil(tensor):
    tensor = tensor.clamp(0, 1) * 255
    tensor = tensor.byte()
    if tensor.size(0) == 1:  # 灰度图
        return Image.fromarray(tensor.squeeze(0).numpy(), mode='L')
    else:  # RGB图像
        return Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode='RGB')


# 显示图像
def display_images(original, transformed_clip, transformed_no_clip):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(transformed_clip)
    axes[1].set_title("Transformed Image (clip=True)")
    axes[1].axis('off')

    axes[2].imshow(transformed_no_clip)
    axes[2].set_title("Transformed Image (clip=False)")
    axes[2].axis('off')

    plt.show()


#主程序
if __name__ == "__main__":
    file_path = './data/cvctest/train/images/7.tif'  # 替换为你的TIF图像路径
    original_image = read_image(file_path)

    # Initialize transforms
    gaussian_noise_transform = RandIntensityDisturbance(p=1, brightness_limit=0.5, contrast_limit=0.5, clip=False)
    #gaussian_noise_transform = RandGaussianNoise(p=1, mean=0., std=0.1, clip=True)
    masking_generator = RandomMaskingGenerator(input_size=(original_image.shape[1], original_image.shape[2]),
                                               mask_ratio=0.8, block_size=(20, 20))

    # Apply Gaussian Noise
    transformed_image = gaussian_noise_transform.forward_image(original_image)

    # Apply Random Mask
    mask = masking_generator()
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).repeat(original_image.shape[0], 1, 1)
    masked_image = transformed_image * mask_tensor.to(transformed_image.device)

    # Convert to PIL images for display
    original_pil = tensor_to_pil(original_image)
    transformed_pil = tensor_to_pil(transformed_image)
    masked_pil = tensor_to_pil(masked_image)

    # Display images
    display_images(original_pil, transformed_pil, masked_pil)
