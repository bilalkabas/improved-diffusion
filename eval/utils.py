import os
import h5py
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms


# Get matlab color palette
matlab_colors = np.array([
    [0.0000, 0.4470, 0.7410],
    [0.8500, 0.3250, 0.0980],
    [0.9290, 0.6940, 0.1250],
    [0.4940, 0.1840, 0.5560],
    [0.4660, 0.6740, 0.1880],
    [0.3010, 0.7450, 0.9330],
    [0.6350, 0.0780, 0.1840]])


def find_file(path, keyword):
    files = [f for f in os.listdir(path) if keyword in f]

    if len(files) == 0:
        raise ValueError(f'No file with keyword {keyword} found in {path}')
    elif len(files) > 1:
        raise ValueError(f'Multiple files with keyword {keyword} found in {path}')
    else:
        return os.path.join(path, files[0])


def to_norm(x):
    x = x/2
    x = x + 0.5
    return x.clip(0, 1)

def norm_01(x):
    return (x - x.min(axis=(-1,-2), keepdims=True))/(x.max(axis=(-1,-2), keepdims=True) - x.min(axis=(-1,-2), keepdims=True))


def read_mat_file(file_path, key_name, rotate=None, transpose=None):
    with h5py.File(file_path, "r") as f:
        images = np.array(f[key_name])

    # Transpose images
    if transpose:
        images = images.transpose(transpose)

    # Rotate images
    if rotate:
        images = transforms.functional.rotate(
            torch.from_numpy(images), rotate).numpy()

    return images


def read_images_from_folder(file_path, rotate=None, file_ext='.png', sort=False):
    file_names = [f for f in os.listdir(file_path) if f.endswith(file_ext)]

    # Sort file names according to the numeric part
    if sort:
        file_names = sorted(file_names, key=lambda x: int(x.split('_')[0]))

    images = []
    for file_name in file_names:
        image = np.array(Image.open(os.path.join(file_path, file_name)))[...,0]
        images.append(image)

    images = np.asarray(images)
    
    # Rotate images
    if rotate:
        images = transforms.functional.rotate(
            torch.from_numpy(images), rotate).numpy()

    return images


def read_npy_file(file_path):
    images = np.load(file_path)
    return images


def mean_norm(x):
    x = np.abs(x)
    return x/x.mean(axis=(-1,-2), keepdims=True)


def apply_mask_and_norm(x, mask, norm_func):
    x = x*mask
    x = norm_func(x)
    return x


def compute_metrics(
    gt_images, 
    pred_images, 
    masks=None,
    subjects_info=None,
    norm='mean'
):
    # Compute psnr and ssim
    psnr_values = []
    ssim_values = []

    # Normalize function
    if norm == 'mean':
        norm_func = mean_norm
    elif norm == '01':
        norm_func = norm_01

    # Apply mask and normalize
    if masks is not None:
        gt_images = apply_mask_and_norm(gt_images, masks, norm_func)
        pred_images = apply_mask_and_norm(pred_images, masks, norm_func)
    else:
        gt_images = norm_func(gt_images)
        pred_images = norm_func(pred_images)

    # Compute psnr and ssim
    for gt, pred in zip(gt_images, pred_images):
        psnr_value = psnr(gt, pred, data_range=gt.max())
        psnr_values.append(psnr_value)

        ssim_value = ssim(gt, pred, data_range=gt.max())*100
        ssim_values.append(ssim_value)

    # Convert list to numpy array
    psnr_values = np.asarray(psnr_values)
    ssim_values = np.asarray(ssim_values)

    # Get per patient metrics
    psnr_ps_values = []
    ssim_ps_values = []
    subject_ids = []
    valid_slice_indices = []

    if subjects_info:
        for subject in subjects_info:
            start, end = subject['slice_interval']
            psnr_ps_values.append(np.nanmean(psnr_values[start:end]))
            ssim_ps_values.append(np.nanmean(ssim_values[start:end]))

            # Get subject ids
            subject_ids.append(np.ones(end-start)*subject['id'])

            # Get valid slice indices
            valid_slice_indices.append(np.arange(start, end))

        valid_slice_indices = np.concatenate(valid_slice_indices, axis=0)
        subject_ids = np.concatenate(subject_ids, axis=0)

    # Convert list to numpy array
    psnr_ps_values = np.asarray(psnr_ps_values)
    ssim_ps_values = np.asarray(ssim_ps_values)

    res = {
        'valid_indices': valid_slice_indices,
        'psnrs': psnr_values,
        'ssims': ssim_values,
        'psnr_ps': psnr_ps_values,
        'ssim_ps': ssim_ps_values,
        'psnr_mean_ps': psnr_ps_values.mean(),
        'ssim_mean_ps': ssim_ps_values.mean(),
        'psnr_std_ps': psnr_ps_values.std(),
        'ssim_std_ps': ssim_ps_values.std(),
        'subject_ids': subject_ids
    }

    return res


def ax_zoomed(
    ax,
    im,
    zoom_region,
    zoom_size,
    zoom_edge_color='yellow'
):
    ax.imshow(np.flip(im, axis=0), origin='lower', cmap='gray')
    x1, x2, y1, y2 = zoom_region
    axins = ax.inset_axes(
        zoom_size,
        xlim=(x1, x2), ylim=(y1, y2))
    
    axins.imshow(np.flip(im, axis=0), cmap='gray')

    # Add border to zoomed region
    for spine in axins.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    
    # Remove inset axes ticks/labels
    axins.set_xticks([])
    axins.set_yticks([])
    
    ax.indicate_inset_zoom(axins, edgecolor=zoom_edge_color, linewidth=3)
    ax.axis('off')
