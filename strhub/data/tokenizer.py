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

import json
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class Tokenizer:
    EOS = "[E]"
    UNK = "[UNK]"
    SPC = " "  # token for space
    BOS = "[B]"
    PAD = "[P]"

    def __init__(self, target_signs_file: str) -> None:
        (
            self._reading_to_signs,
            self._sign_to_index,
            self._index_to_sign,
        ) = self._load_target_signs(target_signs_file)
        self.eos_id = self._sign_to_index[self.EOS]
        self.unk_id = self._sign_to_index[self.UNK]
        self.spc_id = self._sign_to_index[self.SPC]
        self.bos_id = self._sign_to_index[self.BOS]
        self.pad_id = self._sign_to_index[self.PAD]

    def encode(
        self, labels: List[str], device: Optional[torch.device] = None
    ) -> Tensor:
        """
        Generate token indexes from batched strings

        Args:
            batch labels (List[List[str]]): batch strings

        Returns:
            batch tokens (List[List[int]]): batch indexes
        """
        batch = [
            torch.as_tensor(
                [self.bos_id] + self._tok2ids(y) + [self.eos_id],
                dtype=torch.long,
                device=device,
            )
            for y in labels
        ]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def __len__(self):
        return len(self._sign_to_index)

    def decode(self, token_dists: Tensor):
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs

    def _tok2ids(self, tokens: List[str]) -> List[int]:
        ids = []
        for token in tokens:
            if token not in self._reading_to_signs.keys():
                ids.append(self.unk_id)
            else:
                ids.extend(self._reading_to_signs[token])

        return ids

    def _ids2tok(self, token_ids: List[int]) -> List[str]:
        return [self._index_to_sign[token_id] for token_id in token_ids]

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """
        Internal method which performs the necessary filtering prior to decoding.
        Tokens [EOS] are removed.
        """
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[: eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids

    def _load_target_signs(self, target_signs_file_path: str):
        """
        Load target signs json.
        """
        reading_to_signs: Dict[str, List[int]] = {}
        sign_to_index: Dict[str, int] = {self.EOS: 0, self.UNK: 1, self.SPC: 2}
        index_to_sign: Dict[int, str] = {}
        # create inverse dictionary of sign_to_index
        for key, value in sign_to_index.items():
            index_to_sign[value] = key

        with open(target_signs_file_path) as f:
            loaded = json.load(f)

        for signs in sorted(loaded):
            sign_indices = []  # list of int sign indices
            for sign in signs.split("."):
                if sign not in sign_to_index:
                    sign_to_index[sign] = len(sign_to_index)
                    index_to_sign[sign_to_index[sign]] = sign
                sign_indices.append(sign_to_index[sign])

            for reading in loaded[signs]["readings"]:
                reading_to_signs[reading["reading"]] = sign_indices

        for token in (self.BOS, self.PAD):
            idx = len(sign_to_index)
            sign_to_index[token] = idx
            index_to_sign[idx] = token

        return reading_to_signs, sign_to_index, index_to_sign
