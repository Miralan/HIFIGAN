import os
import torch

def create_model(args, Generator, Discriminator, hop_length):
    generator = Generator(args.local_condition_dim)
    discriminator = Discriminator(args.condition_window * hop_length)
    return generator, discriminator
    
def save_checkpoint(args, generator, discriminator, g_optimizer,
                    d_optimizer, step, logging):
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.ckpt-{}.pt".format(step))

    torch.save({"generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
                "global_step": step
                }, checkpoint_path)

    logging.info("Saved checkpoint: {}".format(checkpoint_path))

    with open(os.path.join(args.checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write("model.ckpt-{}.pt".format(step))

def attempt_to_restore(generator, discriminator, g_optimizer, d_optimizer,
                       checkpoint_dir, use_cuda, logging):
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(
            checkpoint_dir, "{}".format(checkpoint_filename))
        logging.info("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        generator.load_state_dict(checkpoint["generator"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        global_step = checkpoint["global_step"]

    else:
        global_step = 0

    return global_step

def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint