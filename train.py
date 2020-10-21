import os
import time
import torch
import argparse
import torch.nn as nn
from utils.training import *
from utils.logging import GetLogging
from utils.audio import hop_length
from utils.optimizer import Optimizer
from utils.adam import AdamW
from utils.dataset import CustomerDataset, CustomerCollate
from utils.loss import MultiResolutionSTFTLoss
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from adabelief_pytorch import AdaBelief

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

    if args.print_network:
        print(generator)
        print(discriminator)

    g_parameters = list(generator.parameters())
    g_optimizer = AdaBelief(g_parameters, lr=args.g_learning_rate, betas=args.adamw_beta, weight_decay=args.adamw_weight_decay)

    d_parameters = list(discriminator.parameters())
    d_optimizer = AdaBelief(d_parameters, lr=args.g_learning_rate, betas=args.adamw_beta, weight_decay=args.adamw_weight_decay)

    generator.to(device)
    discriminator.to(device)

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

            losses = {}

            g_outputs = generator(conditions)

            ## Train discriminator
            # 1 represents multiscale loss, 2 represents multiperiod loss
            fake_scores1, fake_scores2 = discriminator(g_outputs.detach())
            real_scores1, real_scores2 = discriminator(samples)
            
            d_loss_fake_list, d_loss_real_list = [], []

            for (real_score, fake_score) in zip(real_scores1, fake_scores1):
                d_loss_real_list.append(mse_loss(real_score, torch.ones_like(real_score)))
                d_loss_fake_list.append(mse_loss(fake_score, torch.zeros_like(fake_score)))
            for (real_score, fake_score) in zip(real_scores2, fake_scores2):
                d_loss_real_list.append(mse_loss(real_score, torch.ones_like(real_score)))
                d_loss_fake_list.append(mse_loss(fake_score, torch.zeros_like(fake_score)))

            d_loss_real = sum(d_loss_real_list) / len(d_loss_real_list)
            d_loss_fake = sum(d_loss_fake_list) / len(d_loss_fake_list)
            d_loss = d_loss_real + d_loss_fake

            for (i, _) in enumerate(d_loss_real_list):
                losses['d_loss_real_{}'.format(str(i))] = d_loss_real_list[i].item()
                losses['d_loss_fake_{}'.format(str(i))] = d_loss_fake_list[i].item()

            custom_d_optimizer.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(d_parameters, max_norm=0.5)
            custom_d_optimizer.step_and_update_lr(global_epoch)

            losses['d_loss_real'] = d_loss_real.item()
            losses['d_loss_fake'] = d_loss_fake.item()
            losses['d_loss'] = d_loss.item()


            ## Train generator
            fake_scores1, fake_scores2 = discriminator(g_outputs)

            #Adversarial loss
            adv_loss_list = []
            for fake_score in fake_scores1:
                adv_loss_list.append(mse_loss(fake_score, torch.ones_like(fake_score)))
            for fake_score in fake_scores2:
                adv_loss_list.append(mse_loss(fake_score, torch.ones_like(fake_score)))

            adv_loss = sum(adv_loss_list) / len(adv_loss_list)

            for (i, loss) in enumerate(adv_loss_list):
                    losses["adv_loss_{}".format(str(i))] = loss.item()

            #Feature-Matching Loss
            fm_loss_list = []

            for (real_score, fake_score) in zip(real_scores1, fake_scores1):
                fm_loss_list.append(l1_loss(real_score.detach(), fake_score))
            for (real_score, fake_score) in zip(real_scores2, fake_scores2):
                fm_loss_list.append(l1_loss(real_score.detach(), fake_score))

            fm_loss = sum(fm_loss_list) / len(fm_loss_list)

            for (i, loss) in enumerate(fm_loss_list):
                    losses["fm_loss_{}".format(str(i))] = loss.item()

            # STFT Loss
            sc_loss, mag_loss = stft_criterion(g_outputs.squeeze(1), samples.squeeze(1))
            mel_loss = sc_loss + mag_loss
            g_loss = adv_loss + 2 * fm_loss + 10 * mel_loss
            custom_g_optimizer.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(g_parameters, max_norm=0.5)
            custom_g_optimizer.step_and_update_lr(global_epoch)

            losses["adv_loss"] = adv_loss.item()
            losses["fm_loss"] = fm_loss.item()
            losses["mel_loss"] = mel_loss.item()
            losses["g_loss"] = g_loss.item()
            
            time_used = time.time() - start
            
            logging.info(f"Epoch: {global_epoch} Step: {global_step} --d_loss: {d_loss:.3f} --d_real_loss: {d_loss_real:.3f} --d_faker_loss: {d_loss_fake:.3f} --g_loss: {g_loss:.3f} --adv_loss: {adv_loss:.3f} --fm_loss: {fm_loss:.3f}  --mel_loss: {mel_loss:.3f} --g_lr: {custom_g_optimizer.lr} --d_lr: {custom_d_optimizer.lr} --Time: {time_used:.2f}")
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
    parser.add_argument('--g_learning_rate', type=float, default=0.0008)
    parser.add_argument('--d_learning_rate', type=float, default=0.0008)
    parser.add_argument('--decay_learning_rate', type=float, default=0.999)
    parser.add_argument('--adamw_beta', type=float, default=(0.9, 0.99))
    parser.add_argument('--adamw_weight_decay', type=float, default=1e-12)
    parser.add_argument('--local_condition_dim', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--condition_window', type=int, default=100)
    parser.add_argument('--logfile', type=str, default="txt")

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

