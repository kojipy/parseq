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

from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .dataset import SyntheticCuneiformLineImage, SyntheticCuneiformValidationLineImage
from .tokenizer import Tokenizer


class AbgalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        synth_root_dir: str,
        target_signs_file: str,
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
        self._tokenizer = Tokenizer(target_signs_file)
        self._real_root_dir = real_root_dir
        self._real_images_root_dir = self._real_root_dir
        self._train_first_idx = train_first_idx
        self._train_last_idx = train_last_idx
        self._valid_first_idx = valid_first_idx
        self._valid_last_idx = valid_last_idx
        self._img_height = img_height
        self._img_width = img_width
        self._batch_size = batch_size
        self._num_workers = num_workers

    @staticmethod
    def get_transform(augment: bool, img_width: int, img_height: int):
        augments = [
            KeepAspectResize((img_width, img_height)),
            Pad((img_width, img_height)),
        ]
        if augment:
            augments.extend(
                [
                    T.RandomApply(
                        transforms=[T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1))],
                        p=0.3,
                    ),
                    T.RandomRotation(degrees=(0, 1)),
                ]
            )
        augments.extend([T.Resize(img_height), T.Grayscale(), T.ToTensor()])

        return T.Compose(augments)

    @property
    def train_dataset(self):
        return SyntheticCuneiformLineImage(
            images_root_dir=self._synth_images_root_dir,
            texts_root_dir=self._synth_text_root_dir,
            reading2signs=self._tokenizer._reading2signs_map,
            first_idx=self._train_first_idx,
            last_idx=self._train_last_idx,
            img_height=self._img_height,
            img_width=self._img_width,
            transform=self.get_transform(
                augment=True, img_width=self._img_width, img_height=self._img_height
            ),
        )

    @property
    def valid_dataset(self):
        return SyntheticCuneiformLineImage(
            images_root_dir=self._synth_images_root_dir,
            texts_root_dir=self._synth_text_root_dir,
            reading2signs=self._tokenizer._reading2signs_map,
            first_idx=self._valid_first_idx,
            last_idx=self._valid_last_idx,
            img_height=self._img_height,
            img_width=self._img_width,
            transform=self.get_transform(
                augment=False, img_width=self._img_width, img_height=self._img_height
            ),
        )

    @property
    def real_dataset(self):
        return SyntheticCuneiformValidationLineImage(
            images_root_dir=self._real_images_root_dir,
            reading2signs=self._tokenizer._reading2signs_map,
            transform=self.get_transform(
                augment=False, img_width=self._img_width, img_height=self._img_height
            ),
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


class KeepAspectResize:
    def __init__(self, size: Tuple[int, int]) -> None:
        """
        Args:
            size (Tuple[int, int]): target image size. (width, height)
        """
        self._size = size

    def __call__(self, image: Image.Image):
        width, height = self._size
        x_ratio = width / image.width
        y_ratio = height / image.height

        if x_ratio < y_ratio:
            resize_size = (width, round(image.height * x_ratio))
        else:
            resize_size = (round(image.width * y_ratio), height)

        resized_image = image.resize(resize_size, resample=Image.LANCZOS)

        return resized_image


class Pad:
    def __init__(self, size: Tuple[int, int]) -> None:
        """
        Args:
            size (Tuple[int, int]): target image size. (width, height)
        """
        self._size = size

    def __call__(self, image: Image.Image):
        width, height = self._size
        return ImageOps.pad(image, (width, height), color=(0, 0, 0), centering=(0, 0))
