import typing as tp

import torch
import sentencepiece as spm

from model import TransferModel
from utils import make_tensor


def get_style_transfer(model: TransferModel,
                       sp: spm.SentencePieceProcessor,
                       preprocessed_batch: tp.List[str],
                       dest_styles: tp.List[int],
                       temperature: float = 1.0,
                       max_steps: int = 30,
                       eos_token: int = 2) -> tp.List[str]:
    """
    Get style transfer of batch of text
    """
    model.eval()
    batch_ids = [sp.encode_as_ids(text) for text in preprocessed_batch]
    batch_ids = make_tensor(batch_ids, 1, 2, 0)

    styles = torch.tensor(dest_styles, dtype=int)
    translated_batch = model.temperature_translate_batch(batch_ids, batch_ids != 0, styles,
                                                         temperature, max_steps, eos_token)
    result = [sp.decode(item) for item in translated_batch]
    return result
