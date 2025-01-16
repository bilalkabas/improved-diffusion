# %% Import packages
import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import compute_metrics, ax_zoomed, matlab_colors, find_file, mean_norm


# parser = argparse.ArgumentParser()

# parser.add_argument('-p', '--pred_path', type=str, help='Path to the prediction file (.npy)')
# parser.add_argument('-m', '--modality', type=str, help='Modality (source_target)')
# parser.add_argument('-d', '--dataset', type=str, help='Dataset')
# parser.add_argument('-n', '--n_samples', type=int, default=10, help='Number of samples to visualize')

# args = parser.parse_args()


pred_path = r"D:\git\eval_p2p\data\brats\ddpm_x0\ddpm_brats_t2_flair.npy"
modality = 't2_flair'
dataset = 'brats'
n_samples = 10

source_modality, target_modality = modality.split('_')

# pred_path = args.pred_path
# source_modality, target_modality = args.modality.split('_')
# dataset = args.dataset
# n_samples = args.n_samples


experiment_dir = r'data'
norm_type = 'mean'
mask = True

methods = ['method']

# Get target images
target_images = np.load(find_file(
    path=os.path.join(experiment_dir, dataset, 'gt'), 
    keyword=f'{target_modality}.npy'))

# Get source images
source_images = np.load(find_file(
    path=os.path.join(experiment_dir, dataset, 'gt'), 
    keyword=f'{source_modality}.npy'))

# Get method images
pred_images_all = []

pred_images = np.load(pred_path).reshape(-1, 256, 256)

pred_images = pred_images.squeeze()

# Normalize pred_images
if np.nanmin(pred_images) < -0.1:
    pred_images = (pred_images + 1) / 2
    pred_images = pred_images.clip(0, 1)

# If pred and target are not the same shape, crop pred to target shape
if pred_images.shape != target_images.shape:
    crop_transform = transforms.CenterCrop((target_images.shape[-2], target_images.shape[-1]))
    pred_images = crop_transform(torch.from_numpy(pred_images)).numpy()

pred_images_all.append(pred_images)


# Get masks
masks = None

if mask:
    if dataset == 'ixi':
        masks = np.load(find_file(
            path=os.path.join(experiment_dir, dataset, 'gt'), 
            keyword=f'masks.npy'))
    else:
        if dataset == 'ctt1':
            masks = target_images > 0
        else:
            masks = (target_images + source_images) > 0


# Read experiment config
with open("experiment_config.yaml", "r") as f:
    config = yaml.safe_load(f)


if dataset == 'ixi':
    subjects_info = config['IXI']['subjects']
else:
    subjects_info = None

report = ''

for norm_type in ['01', 'mean']:
    # Compute metrics
    metrics_dict = {}

    # Compute metrics for bbdm
    for method, pred_images in zip(methods, pred_images_all):
        metrics = compute_metrics(
            target_images,
            pred_images,
            masks=masks,
            subjects_info=subjects_info,
            norm=norm_type
    )
        
        metrics_dict[method] = metrics

    model_names, metric_values = list(metrics_dict.keys()), list(metrics_dict.values())


    # Print metrics
    if dataset == 'ixi':
        report += f'Dataset: {dataset} {source_modality}->{target_modality}, Norm type: {norm_type}, Mask: {mask}\n'
        report += "-"*30 + '\n'

        for model_name in model_names:
            report += f'{model_name} PSNR: {metrics_dict[model_name]["psnr_mean_ps"]:.2f} +/- {metrics_dict[model_name]["psnr_std_ps"]:.2f}\n'
            report += f'{model_name} SSIM: {metrics_dict[model_name]["ssim_mean_ps"]:.2f} +/- {metrics_dict[model_name]["ssim_std_ps"]:.2f}\n\n'

    else:
        report += f'Dataset: {dataset} {source_modality}->{target_modality}, Norm type: {norm_type}, Mask: {mask}\n'
        report += "-"*30 + '\n'
        
        for model_name in model_names:
            psnr_mean, ssim_mean = metrics_dict[model_name]['psnrs'].mean(), metrics_dict[model_name]['ssims'].mean()
            psnr_std, ssim_std = metrics_dict[model_name]['psnrs'].std(), metrics_dict[model_name]['ssims'].std()

            report += f'{model_name} PSNR: {psnr_mean:.2f} +/- {psnr_std:.2f}\n'
            report += f'{model_name} SSIM: {ssim_mean:.2f} +/- {ssim_std:.2f}\n\n'

# Save report to txt
with open(os.path.join(os.path.dirname(pred_path), "results.txt"), 'w') as f:
    f.write(report)

print(f'Report saved to {os.path.join(os.path.dirname(pred_path), "results.txt")}\n')

# Print report
print(report)


#%% Plot gt and predictions
plt.style.use('dark_background')

h, w = 30, 30
zoom_region = [100-w, 100+w, 100-h, 100+h]
zoom_size = [0, -0.4, 1, 0.47]

images_all_list = [source_images, target_images, *pred_images_all]

# Shuffle all images
images_all_list = [np.array(images) for images in images_all_list]
shuffled_indices = np.random.permutation(images_all_list[0].shape[0])
images_all_list = [images[shuffled_indices] for images in images_all_list]

images_all = zip(*images_all_list)

# Get valid slice indices
valid_indices = []
for subject in config['IXI']['subjects']:
    start, end = subject['slice_interval']

    # Get valid slice indices
    valid_indices.append(np.arange(start, end))

# Flatten valid indices
valid_indices = np.concatenate(valid_indices)

pred_img_start = len(images_all_list) - len(model_names)

for i, images in enumerate(images_all):
    # Skip invalid slices
    if i not in valid_indices:
        continue

    if i == n_samples:
        break

    fig, ax = plt.subplots(1,len(images_all_list), figsize=(12*1.5,8*1.5))
    for k, image in enumerate(images):

        # Find the subject id of the current slice from intervals
        subject_id = None
        for subject in config['IXI']['subjects']:
            start, end = subject['slice_interval']
            if i >= start and i < end:
                subject_id = subject['id']
                break

        ax_zoomed(ax[k], mean_norm(image), zoom_region, zoom_size)
        if k > pred_img_start-1:
            ax[k].set_title(f'{model_names[k-pred_img_start]}\nPSNR: {metric_values[k-pred_img_start]["psnrs"][i]:.2f}\nSSIM: {metric_values[k-pred_img_start]["ssims"][i]:.2f}')
        # elif k == 1:
        #     ax[k].set_title('Target\nMasked')
        elif k == 1:
            ax[k].set_title(f'Target ({target_modality.upper()})')
        elif k == 0:
            ax[k].set_title(f'Source ({source_modality.upper()})\nSubject ID: {subject_id}\nSlice Index: {i}')

    # Set title color of the model with max psnr
    max_psnr_idx = np.argmax([metric_value['psnrs'][i] for metric_value in metric_values]) + pred_img_start
    ax[max_psnr_idx].title.set_color('yellow')            


    # Save figure
    path = os.path.join(os.path.dirname(pred_path), 'sample_images', f'slice_{i}.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

