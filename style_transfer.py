import typing as tp

import torch
import sentencepiece as spm

from model import TransferModel
from utils import make_tensor


def get_style_transfer(model: TransferModel,
                       sp: spm.SentencePieceProcessor,
                       preprocessed_batch: tp.List[str],
                       dest_styles: tp.List[int],
                       temperature: float = 0.001,
                       max_steps: int = 30,
                       bos_token: int = 1,
                       eos_token: int = 2) -> tp.List[str]:
    """
    Get style transfer of batch of text
    """
    model.eval()
    batch_ids = [sp.encode_as_ids(text) for text in preprocessed_batch]
    batch_ids = make_tensor(batch_ids, 1, 2, 0).to(model.encoder.embedding.weight.device)

    styles = torch.tensor(dest_styles, dtype=int, device=model.encoder.embedding.weight.device)
    translated_batch, pad_mask = model.temperature_translate_batch(batch_ids, batch_ids != 0, styles,
                                                                   temperature, max_steps, bos_token, eos_token)
    translated_batch *= pad_mask

    result = [sp.decode(item) for item in translated_batch.T.tolist()]
    return result
