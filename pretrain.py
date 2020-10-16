import os
import time
import torch
import argparse
import torch.nn as nn
from utils.training import *
from utils.logging import GetLogging
from utils.audio import hop_length, Audio2Mel
from utils.optimizer import Optimizer
from utils.adam import AdamW
from utils.dataset import CustomerDataset, CustomerCollate
from utils.loss import MultiResolutionSTFTLoss
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(args):

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging = GetLogging(args.logfile)
    train_dataset = CustomerDataset(
         args.input,
         upsample_factor=hop_length,
         local_condition=True,
         global_condition=False)

    device = torch.device("cuda:0")
    generator, discriminator = create_model(args, Generator, Discriminator, hop_length)
    cal_mel = Audio2Mel()

    if args.print_network:
        print(generator)
        print(discriminator)

    g_parameters = list(generator.parameters())
    g_optimizer = AdamW(g_parameters, lr=args.g_learning_rate, betas=args.adamw_beta, weight_decay=args.adamw_weight_decay)

    d_parameters = list(discriminator.parameters())
    d_optimizer = AdamW(d_parameters, lr=args.g_learning_rate, betas=args.adamw_beta, weight_decay=args.adamw_weight_decay)

    generator.to(device)
    discriminator.to(device)
    cal_mel.to(device)

    global_step = 0
    global_epoch = 0
    if args.resume is not None:
        restore_step = attempt_to_restore(generator, discriminator, g_optimizer,
                               d_optimizer, args.resume, args.use_cuda, logging)
        global_step = restore_step

    custom_g_optimizer = Optimizer(g_optimizer, args.g_learning_rate, 
                                   global_epoch, args.decay_learning_rate)
    custom_d_optimizer = Optimizer(d_optimizer, args.d_learning_rate, 
                                   global_epoch, args.decay_learning_rate)
    mse_loss = nn.MSELoss().to(device)
    l1_loss = nn.L1Loss().to(device)
    stft_criterion = MultiResolutionSTFTLoss()

    for epoch in range(args.epochs):
        collate = CustomerCollate(
           upsample_factor=hop_length,
           condition_window=args.condition_window,
           local_condition=True,
           global_condition=False)

        train_data_loader = DataLoader(train_dataset, collate_fn=collate,
               batch_size=args.batch_size, num_workers=args.num_workers,
               shuffle=True, pin_memory=True)

        for batch, (samples, conditions) in enumerate(train_data_loader):
            start = time.time()
            batch_size = int(conditions.shape[0])
            samples = samples.to(device)
            conditions = conditions.to(device)

            losses={}
            # Run generator
            g_outputs = generator(conditions)

            # STFT Loss
            sc_loss, mag_loss = stft_criterion(g_outputs.squeeze(1), samples.squeeze(1))
            mel_loss = sc_loss + mag_loss
            g_loss = mel_loss
            losses['sc_loss'] = sc_loss.item()
            losses['mag_loss'] = mag_loss.item()
            losses['g_loss'] = g_loss.item()

            custom_g_optimizer.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(g_parameters, max_norm=0.5)
            custom_g_optimizer.step_and_update_lr(global_epoch)

            time_used = time.time() - start
            
            logging.info(f"Epoch: {global_epoch} Step: {global_step} --g_loss: {g_loss:.3f} --Time: {time_used:.2f}")
            # Save checkpoints
            if global_step % args.checkpoint_step == 0:
                save_checkpoint(args, generator, discriminator,
                     g_optimizer, d_optimizer, global_step, logging)
            
            global_step += 1
        global_epoch += 1


def main():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train', help='Directory of training data')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of dataloader workers.')
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--checkpoint_dir', type=str, default="logdir", help="Directory to save model")
    parser.add_argument('--resume', type=str, default=None, help="The model name to restore")
    parser.add_argument('--checkpoint_step', type=int, default=5000)
    parser.add_argument('--use_cuda', type=_str_to_bool, default=True)
    parser.add_argument('--print_network', type=_str_to_bool, default=False)
    parser.add_argument('--g_learning_rate', type=float, default=0.0002)
    parser.add_argument('--d_learning_rate', type=float, default=0.0002)
    parser.add_argument('--decay_learning_rate', type=float, default=0.999)
    parser.add_argument('--adamw_beta', type=float, default=(0.8, 0.99))
    parser.add_argument('--adamw_weight_decay', type=float, default=0.01)
    parser.add_argument('--local_condition_dim', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--condition_window', type=int, default=100)
    parser.add_argument('--logfile', type=str, default="txt")

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

