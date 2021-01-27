import typing as tp

import torch
import pandas as pd
import numpy as np
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader


def write_lines_to_file(lines: tp.Iterable[str], filename: str) -> tp.NoReturn:
    """
    Just writes lines to file one by one.
    Used by spm tokenizer
    """
    with open(filename, "w") as fout:
        for line in lines:
            if type(line) is float:
                print(line)
            fout.write(line + "\n")


def create_sp_processor(lines: tp.Iterable[str],
                        vocab_size: int) -> spm.SentencePieceProcessor:
    """
    Trains and create sentencepiece processor
    """
    # write_lines_to_file(lines, 'train.txt')
    spm.SentencePieceTrainer.Train(input='train.txt',
                                   model_prefix='bpe',
                                   vocab_size=vocab_size,
                                   pad_id=0,
                                   bos_id=1,
                                   eos_id=2,
                                   unk_id=3)
    sp = spm.SentencePieceProcessor(model_file='bpe.model')
    return sp


class TransferDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    @classmethod
    def from_csv(cls, filename):
        dataframe = pd.read_csv(filename, sep=';', header=False)
        return cls(dataframe)

    def __getitem__(self, index):
        return dict(self.dataframe.iloc[index])

    def __len__(self):
        return len(self.dataframe)


def create_datasets_and_loaders(train_df: pd.DataFrame,
                                val_df: pd.DataFrame,
                                test_df: pd.DataFrame,
                                batch_size: int) -> tp.Dict[str, tp.Union[TransferDataset, DataLoader]]:
    train_dataset = TransferDataset(train_df)
    val_dataset = TransferDataset(val_df)
    test_dataset = TransferDataset(test_df)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    data = {
        'train_dataset': train_dataset,
        'train_dataloader': train_dataloader,
        'val_dataset': val_dataset,
        'val_dataloader': val_dataloader,
        'test_dataset': test_dataset,
        'test_dataloader': test_dataloader
    }
    return data


def corrupt_ids(ids, word_drop_probability=0.1, k=3):
    """
    Noise function, described in https://arxiv.org/pdf/1711.00043.pdf.
    At first, randomly deletes words from sentence;
    At second, slightly permutes words. For more details see the paper,
        section 2.3
    :ids: ids of words in sentence
    :word_drop_probability: probability, with which every word will be dropped
    :k: parameter, which controls "strength" of permutation
    :return: corrupted ids
    """
    # dropout
    ids_arr = np.array(ids, dtype=int)
    mask = np.random.uniform(size=len(ids_arr)) >= word_drop_probability
    ids_arr = ids_arr[mask]
    # permute
    q = np.arange(len(ids_arr))
    q = q + np.random.randint(0, k + 1, size=(len(ids_arr)))
    permute = np.argsort(q)
    ids_arr = ids_arr[permute]
    return list(ids_arr)


def add_batch_info(sp, batch, word_drop_probability=0.1, k=3):
    batch['pieces'] = [sp.encode_as_pieces(text) for text in batch['text']]
    batch['ids'] = [sp.encode_as_ids(text) for text in batch['text']]
    batch['corrupted_ids'] = [corrupt_ids(ids, word_drop_probability, k)
                              for ids in batch['ids']]
    return batch


def make_tensor(ids_batch: tp.List[tp.List[int]],
                bos_idx: int,
                eos_idx: int,
                pad_idx: int):
    max_len = max(len(x) for x in ids_batch) + 2
    for i in range(len(ids_batch)):
        ids_batch[i].insert(0, bos_idx)
        ids_batch[i].append(eos_idx)
        while len(ids_batch[i]) < max_len:
            ids_batch[i].append(pad_idx)
    return torch.tensor(ids_batch).T


def make_tensors(batch):
    batch['ids'] = make_tensor(batch['ids'], 1, 2, 0)
    batch['corrupted_ids'] = make_tensor(batch['corrupted_ids'], 1, 2, 0)
    batch['ids_mask'] = batch['ids'] != 0
    batch['corrupted_ids_mask'] = batch['corrupted_ids'] != 0
    return batch
