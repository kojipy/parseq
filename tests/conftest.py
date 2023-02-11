from pathlib import Path

import pytest
import torchvision.transforms as T

from strhub.data.dataset import LabelFile, SyntheticCuneiformLineImage
from strhub.data.utils import Tokenizer

from .__init__ import CHARSET

SAMPLE_LABEL = "tests/assets/000000000.json"
DATASET_ROOT = "tests/assets/dataset"
TRANFORM = T.Compose(
    [
        T.RandomApply(
            transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.3
        ),
        T.RandomRotation(degrees=(0, 3)),
        T.Resize(32),
        T.Grayscale(),
        T.ToTensor(),
    ]
)


@pytest.fixture
def tokenizer():
    return Tokenizer(CHARSET)


@pytest.fixture
def label_file():
    return LabelFile(SAMPLE_LABEL)


@pytest.fixture
def dataset():
    return SyntheticCuneiformLineImage(
        images_root_dir=str(Path(DATASET_ROOT) / "images"),
        texts_root_dir=str(Path(DATASET_ROOT) / "annotations"),
        first_idx=0,
        last_idx=4,
        transform=TRANFORM,
        img_height=32,
        img_width=512,
    )
