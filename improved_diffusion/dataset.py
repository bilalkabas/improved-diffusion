import os
import glob
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MatFileDataset(Dataset):
    def __init__(
        self, 
        device,
        dataset_paths, 
        padding=True, 
        norm=True
    ):
        self.device = device
        self.data = []
        for file_path in dataset_paths:
            # Read data
            with h5py.File(file_path, "r") as f:
                images = np.array(f['data_fs'])

            # Transpose to (N, C, H, W)
            if images.ndim==3:
                images = np.expand_dims(np.transpose(images, (0,2,1)), axis=1)
            else:
                images = np.transpose(images, (1,0,3,2))
            
            # Change to float32
            images = images.astype(np.float32)

            # Pad to 256x256
            if padding:
                pad_x = int((256-images.shape[2])/2)
                pad_y = int((256-images.shape[3])/2)
                images = np.pad(images,((0,0),(0,0),(pad_x,pad_x),(pad_y,pad_y)))   
                print('padding in x-y with:'+str(pad_x)+'-'+str(pad_y))
            
            # Normalize to [-1,1]
            if norm:    
                images = (images-0.5)/0.5   

            # Append to dataset
            self.data.append(images)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        contrast1 = torch.from_numpy(self.data[0][index]).to(self.device)
        contrast2 = torch.from_numpy(self.data[1][index]).to(self.device)

        return contrast1, contrast2


def LoadFromPNG(path, device, norm=True, padding=True, rotate=False):
    # Get png files
    files = glob.glob(path + '/*.png')

    # Sort files 0.png, 1.png, 2.png, ...
    files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Get half of the image as contrast 1 and the other half as contrast 2
    contrast1 = []  # target
    contrast2 = []  # source

    for file in files:
        img = np.array(Image.open(file))

        if img.ndim > 2:
            img = img[:,:,0]
        
        # Add channel_dim
        img = img[None,...]

        # Normalize
        img = img / 255.0

        img_width = img.shape[-1]
        
        contrast1.append(img[..., :img_width//2])
        contrast2.append(img[..., img_width//2:])
    
    # Convert to numpy array
    contrast1 = np.array(contrast1).astype(np.float32)
    contrast2 = np.array(contrast2).astype(np.float32)

    # Rotate image
    if rotate:
        contrast1 = np.rot90(contrast1, k=1, axes=(-1,-2))
        contrast2 = np.rot90(contrast2, k=1, axes=(-1,-2))

    # Padding
    if padding:
        pad_x = int((256-contrast1.shape[-1])/2)
        pad_y = int((256-contrast1.shape[-2])/2)
        print('padding in x-y with:'+str(pad_x)+'-'+str(pad_y))
        contrast1 = np.pad(contrast1, ((0,0),(0,0),(pad_x,pad_x),(pad_y,pad_y)))
        contrast2 = np.pad(contrast2, ((0,0),(0,0),(pad_x,pad_x),(pad_y,pad_y)))

    # Normalize
    if norm:
        contrast1 = (contrast1 - 0.5) / 0.5
        contrast2 = (contrast2 - 0.5) / 0.5

    return torch.from_numpy(contrast1).to(device), torch.from_numpy(contrast2).to(device)


class IXIDataset(Dataset):
    def __init__(self, contrasts, data_dir, device, stage='train'):
        super().__init__()
        self.contrasts = contrasts
        dataset_path1 = os.path.join(data_dir, self.contrasts[0]+'_1_multi_synth_recon_'+stage+'.mat')
        dataset_path2 = os.path.join(data_dir, self.contrasts[1]+'_1_multi_synth_recon_'+stage+'.mat')

        self.imgs = MatFileDataset(dataset_paths=(dataset_path1, dataset_path2), device=device)

    def __len__(self):
        return len(self.imgs)

    @property
    def shape(self):
        return self.imgs[0][0].shape

    def __getitem__(self, i):
        im1 = self.imgs[i][0]
        im2 = self.imgs[i][1]

        return im1, im2
    

class BRATSDataset(Dataset):
    def __init__(self, contrasts, data_dir, device, stage='train'):
        super().__init__()
        self.contrasts = f"{contrasts[0]}__{contrasts[1]}"
        
        if self.contrasts == "FLAIR__T1" or self.contrasts == "T2__T1" or self.contrasts == "FLAIR__T2":
            image_path = os.path.join(data_dir, f"{contrasts[1]}__{contrasts[0]}/{stage}")
            self.source, self.target = LoadFromPNG(image_path, rotate=True, device=device)
        else:
            image_path = os.path.join(data_dir, f"{contrasts[0]}__{contrasts[1]}/{stage}")
            self.target, self.source = LoadFromPNG(image_path, rotate=True, device=device)

    def __len__(self):
        return len(self.source)
    
    @property
    def shape(self):
        return self.source[0].shape
    
    def __getitem__(self, i):
        return self.target[i], self.source[i]


class CTDataset(Dataset):
    def __init__(self, contrasts, data_dir, device, stage='train'):
        super().__init__()

        image_path = os.path.join(data_dir, stage)
        
        if contrasts[0] == 'CT':
            self.source, self.target = LoadFromPNG(image_path, device=device)
        else:
            self.target, self.source = LoadFromPNG(image_path, device=device)

    def __len__(self):
        return len(self.source)
    
    @property
    def shape(self):
        return self.source[0].shape
    
    def __getitem__(self, i):
        return self.target[i], self.source[i]
