import os
import time
import argparse

import torch
import torch.optim as optim
import horovod.torch as hvd
#import bysteps.torch as bps
from torch.utils.data import DataLoader
from torch.utils.data import distributed
from tensorboardX import SummaryWriter

from transformer import Transformer
from util import make_vocab, CustomDataset
from py3nvml import py3nvml

parser = argparse.ArgumentParser()
parser.add_argument("--exam-name", type=str, default="test")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--epoches", type=int, default=10)
parser.add_argument("--label-smooth", type=bool, default=True)
parser.add_argument("--save-model", type=bool, default=True)
args = parser.parse_args()

hvd.init()
#bps.init()
torch.cuda.set_device(hvd.local_rank())

def train():
    writer = SummaryWriter(os.path.join("log", args.exam_name))

    enc_voc_size = make_vocab(os.path.join("corpora", "train.tags.de-en.de"), "de.vocab.tsv")
    dec_voc_size = make_vocab(os.path.join("corpora", "train.tags.de-en.en"), "en.vocab.tsv")
    print("German vocab size {}".format(enc_voc_size))
    print("English vocab size {}".format(dec_voc_size))
    print("Finish making vocab!")

    train_dataset = CustomDataset(mode="train")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,sampler=train_sampler)
    print("Train data size: {}".format(len(train_dataset)))
    print("Finish loading data!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = Transformer(
        embedding_size=512,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        num_blocks=6,
        is_label_smooth=args.label_smooth
    )

    transformer.cuda()
    optimizer= optim.SGD(transformer.parameters(), lr=0.0001)
    #optimizer= optim.Adam(transformer.parameters(), lr=0.0001, betas=[0.9, 0.98])
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=transformer.named_parameters())
    #AMP
    #transformer, optimizer = amp.initialize(transformer, optimizer, opt_level='O3')
    hvd.broadcast_parameters(transformer.state_dict(), root_rank=0)

    if not os.path.exists("models"):
        os.mkdir("models")
    num_batch = len(train_dataset) // args.batch_size
    print("Start training...")
    step = 0
    for epoch in range(1, args.epoches + 1):
        tic = time.time()
        transformer.train()
        cur_batch = 0
        for x, y in train_loader:
            # Both x and y are sentences with <EOS> as the last vocab.
            x = x.to(device)
            y = y.to(device)  # (N, T)
            tic_r = time.time()
            loss, _, acc = transformer(x, y)  # (N, T_y, dec_voc_size)
            optimizer.zero_grad()
            loss.backward()
            #Apex Training
            """with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()"""
            toc_r = time.time()
            optimizer.step()
            #toc_r = time.time()
            cur_batch += 1
            if cur_batch % 10 == 0:
                print("epoch {}, batch {}/{}, loss {}, acc {}".format(
                    epoch, cur_batch, num_batch, loss.item(), acc.item()))
                print(' model forward and backward used time %f' % (toc_r - tic_r))
            if step % 10 == 0:
                writer.add_scalar('./loss', loss.item(), step)
                writer.add_scalar('./acc', acc.item(), step)
            step += 1
        toc = time.time()
        print("epoch {} use time {}s".format(epoch, toc - tic))
        if args.save_model:
            checkpoint_path = os.path.join("models", "model_epoch_{}.ckpt".format(epoch))
            # torch.save(transformer.state_dict(), checkpoint_path)
            torch.save(transformer, checkpoint_path)


if __name__ == "__main__":
    train()
