import hydra
import torch
import torchvision.transforms as T
from omegaconf import DictConfig

from strhub.data.module import AbgalDataModule

WEIGHT_FILE = "outputs/2023-02-12_13-53-41/checkpoints/last.ckpt"
DEVICE = "cpu"


@hydra.main(config_path="configs", config_name="main", version_base="1.2")
def main(config: DictConfig):
    parseq = hydra.utils.instantiate(config.model)
    parseq.load_state_dict(torch.load(WEIGHT_FILE, map_location=DEVICE)["state_dict"])
    data_module: AbgalDataModule = hydra.utils.instantiate(config.data)
    dataset = data_module.real_dataset

    for i in range(len(dataset)):
        image, label = dataset.__getitem__(i)
        label = label.split(",")

        logits = parseq(image.unsqueeze(0))
        # Greedy decoding
        pred = logits.softmax(-1)
        pred, confidence = parseq.tokenizer.decode(pred)

        # T.ToPILImage()(image).show()
        print(i + 1)
        print("Decoded label\t: {}".format(pred[0]))
        print("annotation\t:", label)
        print()


if __name__ == "__main__":
    main()
