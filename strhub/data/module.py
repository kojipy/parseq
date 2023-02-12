# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path, PurePath
from typing import Callable, Optional, Sequence, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .dataset import SyntheticCuneiformLineImage, SyntheticCuneiformValidationLineImage


class AbgalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        synth_root_dir: str,
        real_root_dir: str,
        train_first_idx: int,
        train_last_idx: int,
        valid_first_idx: int,
        valid_last_idx: int,
        img_height: int,
        img_width: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self._synth_root_dir = synth_root_dir
        self._synth_images_root_dir = str(Path(self._synth_root_dir) / "images")
        self._synth_text_root_dir = str(Path(self._synth_root_dir) / "annotations")
        self._real_root_dir = real_root_dir
        self._real_images_root_dir = str(Path(self._real_root_dir) / "images")
        self._real_text_root_dir = str(Path(self._real_root_dir) / "annotations")
        self._train_first_idx = train_first_idx
        self._train_last_idx = train_last_idx
        self._valid_first_idx = valid_first_idx
        self._valid_last_idx = valid_last_idx
        self._img_height = img_height
        self._img_width = img_width
        self._batch_size = batch_size
        self._num_workers = num_workers

    @staticmethod
    def get_transform(augment: bool, img_height: int):
        augments = []
        if augment:
            augments.extend(
                [
                    T.RandomApply(
                        transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))],
                        p=0.3,
                    ),
                    T.RandomRotation(degrees=(0, 3)),
                ]
            )
        augments.extend([T.Resize(img_height), T.Grayscale(), T.ToTensor()])

        return T.Compose(augments)

    @property
    def train_dataset(self):
        return SyntheticCuneiformLineImage(
            images_root_dir=self._synth_images_root_dir,
            texts_root_dir=self._synth_text_root_dir,
            first_idx=self._train_first_idx,
            last_idx=self._train_last_idx,
            img_height=self._img_height,
            img_width=self._img_width,
            transform=self.get_transform(augment=True, img_height=self._img_height),
        )

    @property
    def valid_dataset(self):
        return SyntheticCuneiformLineImage(
            images_root_dir=self._synth_images_root_dir,
            texts_root_dir=self._synth_text_root_dir,
            first_idx=self._valid_first_idx,
            last_idx=self._valid_last_idx,
            img_height=self._img_height,
            img_width=self._img_width,
            transform=self.get_transform(augment=False, img_height=self._img_height),
        )

    @property
    def real_dataset(self):
        return SyntheticCuneiformValidationLineImage(
            images_root_dir=self._real_images_root_dir,
            transform=self.get_transform(augment=False, img_height=self._img_height),
            img_height=self._img_height,
            img_width=self._img_width,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        # dataset = self.real_dataset
        dataset = self.valid_dataset

        return DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
        )
