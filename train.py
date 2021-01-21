import argparse
import os
import shutil
import yaml
import typing as tp

import torch
import wandb
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm as tqdm

from model import TransferModel
from style_transfer import get_style_transfer
from utils import add_batch_info
from utils import create_datasets_and_loaders
from utils import create_sp_processor
from utils import make_tensor
from utils import make_tensors
from utils import write_lines_to_file


def make_log_html(source_text: str,
                  dest_text: str,
                  source_style: int,
                  dest_style: int) -> wandb.Html:
    """
    Creates wandb.Html object for beautiful logging
    :source text: source text
    """
    return wandb.Html(
        f"""
        <h2>Source text, style = {source_style}:</h2>
        <p>{source_text}</p>
        <h2>Destination text, style = {dest_text}:</h2>
        <p>{dest_style}</p>
        """
    )


def get_losses(model, batch):
    ae_loss = model(batch['corrupted_ids'],
                    batch['corrupted_ids_mask'],
                    batch['ids'],
                    batch['ids_mask'],
                    batch['tonalty'])
    bt_loss = model(batch['back_translated'],
                    batch['back_translated'] != 0,
                    batch['ids'],
                    batch['ids_mask'],
                    batch['tonalty'])
    return ae_loss, bt_loss


def move_batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def train_step(train_batch: tp.Dict[str, tp.Any],
               model: TransferModel,
               device: torch.DeviceObjType,
               optimizer: torch.optim.Optimizer,
               sp: spm.SentencePieceProcessor,
               ae_coef: float,
               bt_coef: float,
               word_drop_probability: float,
               k: int):
    train_batch = add_batch_info(sp, train_batch,
                                 word_drop_probability=word_drop_probability,
                                 k=k)
    train_batch = make_tensors(train_batch)
    train_batch = move_batch_to_device(train_batch, device)
    with torch.no_grad():
        model.eval()
        train_batch['back_translated'] = make_tensor(
            model.temperature_translate_batch(
                train_batch['ids'],
                train_batch['ids_mask'],
                torch.randint_like(train_batch['tonalty'], low=0, high=2),
                1.0,
                30,
                2
            ),
            1, 2, 0
        ).to(device)
        model.train()
    optimizer.zero_grad()
    ae_loss, bt_loss = get_losses(model, train_batch)
    loss = ae_coef * ae_loss + bt_coef * bt_loss
    loss.backward()
    optimizer.step()
    return loss.item(), ae_loss.item(), bt_loss.item()


def eval_step(val_batch: tp.Dict[str, tp.Any],
              model: TransferModel,
              device: torch.DeviceObjType,
              sp: spm.SentencePieceProcessor,
              ae_coef: float,
              bt_coef: float,
              word_drop_probability: float,
              k: int):
    val_batch = add_batch_info(sp, val_batch,
                               word_drop_probability=word_drop_probability,
                               k=k)
    val_batch = make_tensors(val_batch)
    val_batch = move_batch_to_device(val_batch, device)
    val_batch['back_translated'] = make_tensor(
        model.temperature_translate_batch(
            val_batch['ids'],
            val_batch['ids_mask'],
            torch.randint_like(val_batch['tonalty'], low=0, high=2),
            1.0,
            30,
            2
        ),
        1, 2, 0
    ).to(device)

    ae_loss, bt_loss = get_losses(model, val_batch)
    loss = ae_coef * ae_loss + bt_coef * bt_loss
    return loss.item(), ae_loss.item(), bt_loss.item()


def train(model,
          device,
          epochs,
          optimizer,
          sp,
          ae_coef,
          bt_coef,
          steps_to_decrease_ae_coef,
          word_drop_probability,
          k,
          train_dataloader,
          val_dataloader,
          log_every):

    global_train_step = 0
    for epoch in range(epochs):
        print(f"Training epoch number {epoch}")
        for train_batch in tqdm(train_dataloader):
            train_batch = move_batch_to_device(train_batch, device)
            loss, ae_loss, bt_loss = train_step(train_batch,
                                                model,
                                                device,
                                                optimizer,
                                                sp,
                                                ae_coef * min(
                                                    (steps_to_decrease_ae_coef - global_train_step) / steps_to_decrease_ae_coef,
                                                    1.0
                                                ),
                                                bt_coef,
                                                word_drop_probability,
                                                k)
            global_train_step += 1
            if global_train_step % log_every == 0:
                wandb.log({
                    'train/loss': loss,
                    'train/ae_loss': ae_loss,
                    'train/bt_loss': bt_loss,
                    'train/ae_coef': ae_coef * min(
                        (steps_to_decrease_ae_coef - global_train_step) / steps_to_decrease_ae_coef, 1.0),
                    'train/bt_coef': bt_coef
                }, commit=True)
        print(f"Evaluating epoch number {epoch}")
        with torch.no_grad():
            model.eval()
            total_loss = 0.0
            total_ae_loss = 0.0
            total_bt_loss = 0.0
            for val_batch in tqdm(val_dataloader):
                val_batch = move_batch_to_device(val_batch, device)
                loss, ae_loss, bt_loss = eval_step(val_batch, model, device, sp, ae_coef, bt_coef,
                                                   word_drop_probability, k)
                total_loss += loss
                total_ae_loss += total_ae_loss
                total_bt_loss += total_bt_loss

            total_loss /= len(val_dataloader)
            total_ae_loss /= len(val_dataloader)
            total_bt_loss /= len(val_dataloader)
            print(f"After epoch {epoch}:\nloss = {total_loss}\nae_loss = {total_ae_loss}\nbt_loss = {total_bt_loss}")
            wandb.log({
                'test/loss': total_loss,
                'test/ae_loss': total_ae_loss,
                'test/bt_loss': total_bt_loss,
            }, commit=False)
            source_text = val_batch['text'][0]
            source_tonalty = val_batch['tonalty'][0]
            dest_tonalty = 1 - val_batch['tonalty'][0]
            dest_text = get_style_transfer(model, sp,
                                           [source_text],
                                           [dest_tonalty])[0]
            wandb.log({
                'samples': make_log_html(source_text,
                                         dest_text,
                                         source_tonalty,
                                         dest_tonalty)
            }, commit=False)
            model.train()
        model.state_dict()


def get_config(path_to_config):
    """
    Gets dict with configuration
    """
    with open(path_to_config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def save_checkpoint(model: TransferModel,
                    optimizer: torch.optim.Optimizer,
                    exp_name: str,
                    checkpoint_name: str) -> str:
    os.makedirs(os.path.join('checkpoints', exp_name))
    checkpoint_directory = os.path.join('checkpoints', exp_name)
    shutil.copyfile('bpe.model', checkpoint_directory)
    shutil.copyfile('bpe.vocab', checkpoint_directory)
    shutil.copyfile('config.yaml', checkpoint_directory)
    torch.save({
        'model_state_dict': model.state_dict(),
        'opt_state_dict': optimizer.state_dict()
    }, os.path.join(checkpoint_directory, checkpoint_name))

    return checkpoint_directory


def main():
    parser = argparse.ArgumentParser(description="performs training")
    parser.add_argument('path_to_config', type=str, help="path to config.yaml, where all hyper-parameters are stored")
    parser.add_argument('path_to_train', type=str, help="path to train.csv")
    parser.add_argument('path_to_val', type=str, help='path to val.csv')
    parser.add_argument('path_to_test', type=str, help='path to test.csv')
    parser.add_argument('--do_preprocess', action='store_true')
    args = parser.parse_args()
    config = get_config(args.path_to_config)
    do_preprocess = args.do_preprocess

    train_df = pd.read_csv(args.path_to_train, sep=';', )
    val_df = pd.read_csv(args.path_to_val, sep=';')
    test_df = pd.read_csv(args.path_to_test, sep=';')

    if do_preprocess:
        print("Creating subword processor...")
        sp = create_sp_processor(train_df['text'])
    else:
        sp = spm.SentencePieceProcessor(model_file='bpe.model')
    device = torch.device(config["device"])

    model = TransferModel(config['batch_size'],
                          config['pool_window_size'],
                          sp.vocab_size(),
                          config['num_styles'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    data = create_datasets_and_loaders(train_df, val_df, test_df, config["batch_size"])
    #  wandb.login()
    wandb.init(project="sandbox", name=config['experiment_name'])

    train(model,
          device,
          config["epochs"],
          optimizer,
          sp,
          config["ae_coef"],
          config["bt_coef"],
          config["steps_to_decrease_ae_coef"],
          config["word_drop_prob"],
          config["k"],
          data["train_dataloader"],
          data["val_dataloader"],
          config["log_every"])
    save_checkpoint(model, optimizer,
                    exp_name=config['experiment_name'],
                    checkpoint_name='last_epoch.pt')


if __name__ == '__main__':
    main()
