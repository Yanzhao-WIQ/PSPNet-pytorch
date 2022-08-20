import os
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import numpy as np
import torch


def mean_IU(image1, image2, num_classes, ignore_index=None):
    image1 = np.array(image1)
    image2 = np.array(image2)

    [row, col] = image1.shape
    correct_predictions = np.zeros((num_classes, 1))
    incorrect_predictions = np.zeros((num_classes, 1))
    correct_labels = np.zeros((num_classes, 1))
    incorrect_labels = np.zeros((num_classes, 1))
    image1 = np.reshape(image1, (row * col, 1))
    image2 = np.reshape(image2, (row * col, 1))

    for i in range(row * col):
        if (image1[i] == image2[i]):
            correct_predictions[image1[i]] += 1
            correct_labels[image1[i]] += 1
        else:
            incorrect_predictions[image1[i]] += 1
            incorrect_labels[image2[i].astype(int)] += 1
    if (ignore_index):
        for i in ignore_index:
            correct_predictions[i] = 0
            incorrect_predictions[i] = 0
            incorrect_labels[i] = 0
    return ((sum(correct_predictions / (correct_predictions + incorrect_predictions + incorrect_labels + 1e-8)))[0]
            / (num_classes - len(ignore_index)))


# 自定义数据集Dataset
class HorseDataset(Dataset):

    def __init__(self, images_dir='data/images', masks_dir='data/masks'):
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ])
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in os.listdir(images_dir)]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in os.listdir(masks_dir)]

    def __getitem__(self, i):
        image = np.array(Image.open(self.images_fps[i]).convert('RGB'))
        mask = np.array(Image.open(self.masks_fps[i]).convert('RGB'))
        data = self.transform(image=image, mask=mask)
        return data['image'], data['mask'][:, :, 0]

    def __len__(self):
        return len(self.images_fps)


def Getdata(train, batch_size):
    dataset = HorseDataset()
    train_dataset, val_dataset = random_split(
        dataset=dataset,
        lengths=[round(len(dataset)*0.85), round(len(dataset)*0.15)],
        generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if train:
        return train_loader
    else:
        return val_loader


if __name__ == '__main__':
    '''
    t0 = plt.imread('data/val_mask/Aoutput_0.png')
    t = plt.imread('data/val_mask/aux_output_0.png')
    plt.imshow(t0)  # show image
    plt.savefig('data/val_mask/aa44.jpg', bbox_inches='tight', dpi=450)
    # save_image(t0, 'data/val_mask/aa300.jpg')
    plt.show()
    '''