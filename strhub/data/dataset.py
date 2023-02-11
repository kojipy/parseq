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
import unicodedata
from pathlib import Path, PurePath
from typing import Callable, List, Optional, Tuple, Union

import lmdb
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from torch.utils.data import ConcatDataset, Dataset

from strhub.data.utils import CharsetAdapter

log = logging.getLogger(__name__)


def build_tree_dataset(root: Union[PurePath, str], *args, **kwargs):
    try:
        kwargs.pop("root")  # prevent 'root' from being passed via kwargs
    except KeyError:
        pass
    root = Path(root).absolute()
    log.info(f"dataset root:\t{root}")
    datasets = []
    for mdb in glob.glob(str(root / "**/data.mdb"), recursive=True):
        mdb = Path(mdb)
        ds_name = str(mdb.parent.relative_to(root))
        ds_root = str(mdb.parent.absolute())
        dataset = LmdbDataset(ds_root, *args, **kwargs)
        log.info(f"\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}")
        datasets.append(dataset)
    return ConcatDataset(datasets)


class LmdbDataset(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(
        self,
        root: str,
        charset: str,
        max_label_len: int,
        min_image_dim: int = 0,
        remove_whitespace: bool = True,
        normalize_unicode: bool = True,
        unlabelled: bool = False,
        transform: Optional[Callable] = None,
    ):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = []
        self.filtered_index_list = []
        self.num_samples = self._preprocess_labels(
            charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim
        )

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(
            self.root,
            max_readers=1,
            readonly=True,
            create=False,
            readahead=False,
            meminit=False,
            lock=False,
        )

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(
        self,
        charset,
        remove_whitespace,
        normalize_unicode,
        max_label_len,
        min_image_dim,
    ):
        charset_adapter = CharsetAdapter(charset)
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get("num-samples".encode()))
            if self.unlabelled:
                return num_samples
            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f"label-{index:09d}".encode()
                label = txn.get(label_key).decode()
                # Normally, whitespace is removed from the labels.
                if remove_whitespace:
                    label = "".join(label.split())
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                if normalize_unicode:
                    label = (
                        unicodedata.normalize("NFKD", label)
                        .encode("ascii", "ignore")
                        .decode()
                    )
                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > max_label_len:
                    continue
                label = charset_adapter(label)
                # We filter out samples which don't contain any supported characters
                if not label:
                    continue
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f"image-{index:09d}".encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        continue
                self.labels.append(label)
                self.filtered_index_list.append(index)
        return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.unlabelled:
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]

        img_key = f"image-{index:09d}".encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


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
