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
import glob
import io
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class LabelFile:
    SPACE = " "

    def __init__(self, path: str) -> None:
        self._path = path
        self._label = self._load()

    def _load(self) -> List[str]:
        """
        Load annotaion json file.

        Returns:
            List[str]: list of character.
        """
        with open(self._path) as f:
            loaded = json.load(f)

        label = []
        for line in loaded["line"]:
            for words in line["signs"]:
                for reading_dict in words:
                    reading = reading_dict["reading"]
                    label.append(reading)
                label.append(self.SPACE)

        label = label[:-1]  # remove last space

        return label

    @property
    def label(self) -> List[str]:
        return self._label


class SyntheticCuneiformLineImage(Dataset):
    def __init__(
        self,
        *,
        images_root_dir: str,
        texts_root_dir: str,
        first_idx: int,
        last_idx: int,
        transform: T.Compose,
        img_height: int = 96,
        img_width: int = 64 * 24,
    ):
        assert first_idx >= 0
        assert last_idx >= 0
        assert first_idx <= last_idx

        self.first_idx = first_idx
        self.last_idx = last_idx
        self.images_root_dir = images_root_dir
        self.texts_root_dir = texts_root_dir
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def _get_image_path(self, index):
        image_path = (
            Path(self.images_root_dir) / f"{index//(10**3):04d}" / f"{index:09d}.png"
        )
        return image_path

    def __len__(self):
        return self.last_idx - self.first_idx + 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        # Image
        image_path = self._get_image_path(index)

        image = Image.open(str(image_path)).convert("RGB")
        width = int(image.width * (self.img_height / image.height))
        width = self._resize(width, self.img_width, self.img_height)

        image = image.resize((width, self.img_height), resample=Image.BILINEAR)
        image = ImageOps.pad(
            image, (self.img_width, self.img_height), color=(0, 0, 0), centering=(0, 0)
        )

        img_above = Image.open(
            str(self._get_image_path(random.randint(self.first_idx, self.last_idx)))
        ).convert("RGB")
        img_below = Image.open(
            str(self._get_image_path(random.randint(self.first_idx, self.last_idx)))
        ).convert("RGB")

        stacked = self._vstack([img_above, image, img_below])
        image = stacked.crop(
            (
                0,
                stacked.height // 2 - int(image.height / 2 * 1.75),
                stacked.width,
                stacked.height // 2 + int(image.height / 2 * 1.75),
            )
        )
        image = image.resize((self.img_width, self.img_height))

        image = self.transform(image)

        text_path = (
            Path(self.texts_root_dir) / f"{index//(10**3):04d}" / f"{index:09d}.json"
        )
        label: List[str] = LabelFile(str(text_path)).label
        # Dataloader内でstackされるのを回避するために文字列型にキャスト
        label_comma_separate: str = ",".join(label)

        return image, label_comma_separate

    def _resize(self, v, minv, maxv):
        scale = 1 + random.uniform(-0.2, 0.2)
        return int(max(minv, min(maxv, scale * v)))

    def _vstack(self, images):
        if len(images) == 0:
            raise ValueError("Need 0 or more images")

        if isinstance(images[0], np.ndarray):
            images = [Image.fromarray(img) for img in images]

        width = max([img.size[0] for img in images])
        height = sum([img.size[1] for img in images])
        stacked = Image.new(images[0].mode, (width, height))

        y_pos = 0
        for img in images:
            stacked.paste(img, (0, y_pos))
            y_pos += img.size[1]
        return stacked


class SyntheticCuneiformValidationLineImage(Dataset):
    def __init__(
        self,
        *,
        images_root_dir: str,
        transform: T.Compose,
        img_height: int = 96,
        img_width: int = 64 * 21,
    ):
        self.images_root_dir = images_root_dir
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        self._text_raw_data = [
            ["EGIR", "pa", " ", "ti", "an", "zi"],  # OK
            [
                "nu",
                "uš",
                "ša",
                "an",
                " ",
                "A",
                "NA",
                " ",
                "NINDA",
                "GUR",
                "RA",
                " ",
                "ŠE",
                "ER",
            ],  # OK
            [
                "pé",
                "ra",
                "an",
                " ",
                "kat",
                "ta",
                "ma",
                " ",
                "ki",
                "ne",
                " ",
                "i",
                "ia",
                "mi",
            ],  # OK
            ["NINDA", "šar", "li", "in", "na", " ", "te", "eḫ", "ḫi"],  # OK
            ["a", "da", "an", "zi", " ", "a", "ku", "wa", "an", "zi"],  # OK
            ["MAḪ", "aš", " ", "LUGAL", "i", " ", "MUNUS", "LUGAL", "i"],  # OK
            [
                "ap",
                "pa",
                "an",
                "zi",
                " ",
                "pa",
                "ri",
                "li",
                "ia",
                "aš",
                "ša",
                " ",
                "MUŠEN",
                "ḪI",
                "A",
            ],  # OK
            [
                "nu",
                "za",
                " ",
                "wa",
                "ar",
                "ap",
                "zi",
                " ",
                "nam",
                "ma",
                "za",
                "a",
                "pa",
                "a",
                "aš",
            ],  # OK
            [
                "9",
                "NA@4",
                "pa",
                "aš",
                "ši",
                "la",
                "aš",
                " ",
                "A",
                "ŠÀ",
                " ",
                "te",
                "ri",
                "ip",
                "pí",
                "aš",
            ],  # OK
        ]

    def __len__(self) -> int:
        return len(self._text_raw_data)

    def _get_image_path(self, index: int) -> str:
        image_path = Path(self.images_root_dir) / f"valid_{index + 1:03d}.png"
        return str(image_path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[str]]:
        image_path = self._get_image_path(index)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        target = self._text_raw_data[index]

        return image, target
