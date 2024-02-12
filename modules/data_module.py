import os
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import glob
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import pytorch_lightning as pl
from typing import List, Dict, Optional

import torchvision.transforms
from PIL import Image
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, Subset, DataLoader, random_split
import torch_geometric.utils as tgu

from utils.data import *

from utils.TruncatedNormal import TruncatedNormal
torch.set_default_device('cuda')


########################################################################################################################
# Shapenet Dataset
class VerticesDataset(Dataset):
    def __init__(self,
                 training_dir: str,
                 default_shapenet: bool = True,
                 files: Optional[List[str]] = None,
                 labels: Optional[Dict[str, int]] = None,
                 ) -> None:
        """
        :param training_dir: directory of training data
        :param default_shapenet: if True, use a default shapenet dataset
        :param files: files names
        :param labels: dictionary of labels for each file
        """

        self.training_dir = training_dir
        self.default_shapenet = default_shapenet

        if default_shapenet:
            self.files = glob.glob(training_dir + '/*/*/*' + 'models/*.obj')
            self.files = [f.replace('\\', '/') for f in self.files]
            self.labels = {
                'label_names': [],
                'label_ids': []
            }
            counter = 0
            for i, label in enumerate(self.files):
                file_name = label.split('/')[-3]
                if file_name not in self.labels['label_names']:
                    counter += 1
                self.labels['label_names'].append(file_name)
                self.labels['label_ids'].append(counter - 1)
        else:
            self.files = files
            self.labels = labels

    def __len__(self) -> int:
        """
        :return: number of files in the training directory
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        :param idx: index of the file
        :return: dictionary of the file and its label
        """
        mesh_file = self.files[idx]
        vertices, faces = load_obj(mesh_file)
        vertices, faces = encode_mesh(vertices, faces, n_bits=8)

        return {
            'vertices': vertices,
            'faces': faces,
            'label_id': self.labels['label_ids'][idx],
            'label_name': self.labels['label_names'][idx],
            'label_path': mesh_file
        }


class ImageDataset(Dataset):
    def __init__(self, training_dir: str, img_extension: str = 'png') -> None:
        """
        :param training_dir: directory of training data
        :param img_extension: image extension
        """

        self.training_dir = training_dir
        # get all images of all classes
        self.images_path = glob.glob(training_dir + '/*/*/*' + 'images/*.' + img_extension)
        self.images_path = [f.replace('\\', '/') for f in self.images_path]
        self.transforms = T.Compose([T.ToTensor(), T.Resize(256)])

    def __len__(self) -> int:
        """
        :return: number of files in the training directory
        """
        return len(self.images_path)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        :param idx: index of the file
        :return: dictionary of the file and its label
        """

        img = self.images_path[idx]
        # What image belongs to, then load its object
        img_folder_name = '/'.join(img.split('/')[:-2])
        model_file = os.path.sep.join([img_folder_name, 'models', 'model_normalized.obj'])

        # Load the object
        vertices, faces = load_obj(model_file)
        vertices, faces = encode_mesh(vertices, faces, n_bits=8)

        # Load the image
        img = Image.open(img).convert('RGB')
        img = self.transforms(img)

        return {
            'vertices': vertices,
            'faces': faces,
            'image': img,
            'label_path': model_file,
            'label_name': img_folder_name.split('/')[-1],
        }


########################################################################################################################
# PolyGen Dataset
class CollateMethod(Enum):
    VERTICES = 1
    FACES = 2
    IMAGES = 3


class PolyGenDataset(pl.LightningDataModule):
    def __init__(self,
                 # ? Data Gathering
                 data_dir: str,
                 collate_method: CollateMethod = CollateMethod.VERTICES,
                 # ? Data loaders
                 batch_size: int = 32,
                 training_split: float = 0.9,
                 validation_split: float = 0.1,
                 # ? Data
                 default_shapenet: bool = True,
                 files_list: Optional[List[str]] = None,
                 labels_dict: Optional[Dict[str, int]] = None,
                 quantization_bits: int = 8,
                 # ? Image
                 use_image_dataset: bool = False,
                 img_extension: str = 'png',
                 # ? Randomization
                 random_shift_vertices: bool = True,
                 random_shift_faces: bool = True,
                 shuffle_vertices: bool = True,

                 ) -> None:
        """
        :param data_dir: Data directory for shapenet dataset
        :param collate_method:  Whether to collect 'IMAGES', 'FACES', or 'VERTICES'
        :param batch_size: How many 3d object to load at once
        :param training_split: a training portion of the dataset
        :param validation_split: a testing portion of the dataset
        :param default_shapenet: Whether to use the default shapenet dataset structure or not
        :param files_list: List of all files to use. ! Must be provided if (default_shapenet is False)
        :param labels_dict: Dictionary of labels for each file (default_shapenet is False) [label_name, [label_ids]]
        :param quantization_bits: Number of bits to quantize the vertices.
        :param use_image_dataset: Whether to use ShapenetDataset or ShapenetRenderedDataset
        :param img_extension: Image extension
        :param random_shift_vertices: Whether to randomize the vertices
        :param random_shift_faces: Whether to randomize the faces
        :param shuffle_vertices: Whether we're shuffling the order of vertices during batch generation for a face model
        """

        super().__init__()
        assert (not collate_method in CollateMethod.__members__), f"Collate method must be one of {CollateMethod.__members__}"
        assert (training_split + validation_split <= 1), "Training and validation split must not exceed 1"
        assert ((use_image_dataset and collate_method != CollateMethod.VERTICES) or (
            not use_image_dataset)), "Image dataset must use vertices collate method"

        self.data_dir = data_dir
        self.collate_method = collate_method
        self.batch_size = batch_size
        self.training_split = training_split
        self.validation_split = validation_split
        self.default_shapenet = default_shapenet
        self.files_list = files_list
        self.labels_dict = labels_dict
        self.quantization_bits = quantization_bits
        self.use_image_dataset = use_image_dataset
        self.img_extension = img_extension

        self.random_shift_vertices = random_shift_vertices
        self.random_shift_faces = random_shift_faces
        self.shuffle_vertices = shuffle_vertices

        if use_image_dataset:
            self.dataset = ImageDataset(data_dir, img_extension)
        else:
            self.dataset = VerticesDataset(data_dir, default_shapenet, files_list, labels_dict)

        # collate_fn parameter in the DataLoader is a function used to customize the way individual samples from the dataset are batched together

        if collate_method == CollateMethod.VERTICES:
            self.collate_fn = self.collate_vertices_batch
        elif collate_method == CollateMethod.FACES:
            self.collate_fn = self.collate_faces_batch
        elif collate_method == CollateMethod.IMAGES:
            self.collate_fn = self.collate_images_batch

    def collate_vertices_batch(self, data_dict: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        """
        :param data_dict: List of dictionaries, where each dictionary contains information about the 3d object
        :return: Batch, Single dictionary containing all the information about the batch
        """

        batch = {}
        num_vertices_list = [d['vertices'].shape[0] for d in data_dict]  # Number of vertices in each element of the batch
        max_num_vertices = max(num_vertices_list)  # maximum number of vertices in the batch
        num_elements = len(data_dict)  # Elements in the batch
        vertices_flat = torch.zeros((num_elements, max_num_vertices * 3 + 1),dtype=torch.long)  # Initialize the vertices tensor
        vertices_flat_mask = torch.zeros_like(vertices_flat, dtype=torch.long)  # Initialize the mask tensor
        class_labels = torch.zeros((num_elements, 1), dtype=torch.long)  # Initialize the class labels tensor

        for i, data in enumerate(data_dict):
            vertices = data['vertices']


            if self.random_shift_vertices:
                pass # TODO: Apply Random shift

            padding_size = max_num_vertices - vertices.shape[0]
            vertex_permuted = torch.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]],dim=-1)  # Permute the vertices z,y,x
            vertices_flatten = vertex_permuted.flatten()  # Flatten the vertices
            vertices_flat[i] = F.pad(vertices_flatten + 1, [0, padding_size * 3 + 1])[None]  # Pad the vertices

            class_labels[i] = torch.tensor(data['label_id'])  # Set the class label
            vertices_flat_mask[i] = torch.zeros_like(vertices_flat[i])
            vertices_flat_mask[i, : vertices.shape[0] * 3 + 1] = 1

        batch['vertices'] = vertices_flat
        batch['mask'] = vertices_flat_mask
        batch['class_labels'] = class_labels
        return batch

    def collate_faces_batch(self, data_dict: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        :param data_dict: List of dictionaries, where each dictionary contains information about the 3d object
        :return: Batch, Single dictionary containing all the information about the batch
        """
        batch = {}
        n_batch = len(data_dict)  # Elements in the batch

        num_faces_list = [d['faces'].shape[0] for d in data_dict]  # Number of faces in each element of the batch
        num_vertices_list = [d['vertices'].shape[0] for d in data_dict]  # Number of vertices in each element of the batch

        faces_size = max(num_faces_list)
        vertices_size = max(num_vertices_list)

        shuffled_faces = torch.zeros([n_batch, faces_size], dtype=torch.long)  # Initialize the face tensor
        faces_mask = torch.zeros_like(shuffled_faces, dtype=torch.long)  # Initialize the mask tensor
        faces_vertices = torch.zeros([n_batch, vertices_size, 3], dtype=torch.long)  # Initialize the face vertices tensor
        faces_vertices_mask = torch.zeros([n_batch], dtype=torch.long)  # Initialize the face vertices mask tensor

        for i, data in enumerate(data_dict):
            faces = data['faces']
            vertices = data['vertices']

            n_vertices = vertices.shape[0]
            n_faces = faces.shape[0]

            # TODO: Check to Apply Random face shift
            # TODO: Check to Apply Random vertices shift

            # Explanation: https://media.discordapp.net/attachments/866267538462605322/1205486342499475456/image.png?ex=65d88ba4&is=65c616a4&hm=861ee059d0385949a3eeac6d07791e7c3956bed6504bdff6cedeab134882c25f&=&format=webp&quality=lossless
            if self.shuffle_vertices:
                permutation = torch.randperm(n_vertices)
                vertices = vertices[permutation]
                vertices = vertices.unsqueeze(0)
                face_permuted = torch.cat([
                    torch.Tensor([0, 1]).to(torch.long),  # end token [0], end face [1]
                    torch.argsort(permutation).to(torch.long) + 2,
                    # Sort from lowest to highest and add 2 to each index
                ], dim=-1)
                curr_faces = face_permuted[faces][None]  # adds a new dimension ~ [1, n_faces]
            else:
                curr_faces = faces[None]  # adds a new dimension ~ [1, n_faces]

            # Padding
            vertices_padding_size = vertices_size - n_vertices
            faces_padding_size = faces_size - n_faces

            shuffled_faces[i] = F.pad(curr_faces, [0, faces_padding_size, 0, 0])
            faces_mask[i] = torch.zeros_like(shuffled_faces[i], dtype=torch.long)  # already flattened
            faces_mask[i, :n_faces] = 1

            faces_vertices[i] = F.pad(vertices, [0, 0, 0, vertices_padding_size])
            faces_vertices_mask[i] = torch.zeros_like(faces_vertices_mask[i].flatten(), dtype=torch.long)
            faces_vertices_mask[i, :n_vertices] = 1

        batch['faces'] = shuffled_faces
        batch['faces_mask'] = faces_mask
        batch['faces_vertices'] = faces_vertices
        batch['faces_vertices_mask'] = faces_vertices_mask
        return batch

    def collate_images_batch(self):
        # TODO: Implement
        pass


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Torch Lightning method to setup the data module
        :param stage: stage of the data module
        """
        n_train = int(len(self.dataset) * self.training_split)
        n_val = int(len(self.dataset) * self.validation_split)
        n_test = len(self.dataset) - n_train - n_val
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [n_train, n_val, n_test], generator=torch.Generator(device='cuda'))


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            generator=torch.Generator(device='cuda'),
            shuffle=True,
            num_workers=1,
            # persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            # num_workers=12,
            # persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            # num_workers=12,
            # persistent_workers=True,
        )