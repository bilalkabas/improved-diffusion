from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from improved_diffusion.dataset import IXIDataset, BRATSDataset, CTDataset


def load_data(dataset_dir, stage, source_modality, target_modality, batch_size, shuffle, device):
    if 'IXI' in dataset_dir:
        DatasetClass = IXIDataset
    elif 'BRATS' in dataset_dir:
        DatasetClass = BRATSDataset
    elif 'CT' in dataset_dir:
        DatasetClass = CTDataset

    dataset = DatasetClass(
        contrasts=(target_modality, source_modality),
        data_dir=dataset_dir,
        device=device,
        stage=stage,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader