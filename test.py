import os
import argparse

import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu

from util import load_vocab, CustomDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--label-smooth", type=bool, default=True)
parser.add_argument("--epoch-idx", type=int, default=10,
                    help="which epoch for test")
args = parser.parse_args()

MAXLEN = 10


def test():
    test_dataset = CustomDataset(mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    de2idx, idx2de = load_vocab(language="de")
    en2idx, idx2en = load_vocab(language="en")

    checkpoint_path = os.path.join("models", "model_epoch_{}.ckpt".format(args.epoch_idx))
    transformer = torch.load(checkpoint_path)
    transformer.to(device)

    if not os.path.exists("results"):
        os.mkdir("results")
    with open(os.path.join("results", "model{}.txt").format(args.epoch_idx), "w", encoding="utf-8") as fw:
        list_of_references, hypotheses = [], []
        for x, y in test_loader:
            # Copy the tensor because x will be modified.
            sources = torch.tensor(x).numpy()
            targets = torch.tensor(y).numpy()

            x = x.to(device)  # (N, T)
            y = y.to(device)  # (N, T)
            preds = torch.zeros(y.shape[0], y.shape[1], dtype=torch.int64).to(device)  # (N*T, )
            loss, _, acc = transformer(x, y)
            for j in range(MAXLEN):
                # After the assign, preds[:, j] is the prediction.
                _, preds, _ = transformer(x, preds)

            preds = preds.cpu().numpy()  # (N, T)

            for i in range(sources.shape[0]):
                source = sources[i]  # numpy array of shape=(T, )
                pred = preds[i]  # numpy array of shape=(T, )
                target = targets[i]  # numpy array of shape=(T, )
                source = " ".join([idx2de[idx] for idx in source]).split("<EOS>")[0]  # str
                pred = " ".join([idx2en[idx] for idx in pred]).split("<EOS>")[0]  # str
                target = " ".join([idx2en[idx] for idx in target]).split("<EOS>")[0]  # str
                fw.write("-source: " + source + "\n")
                fw.write("-pred: " + pred + "\n")
                fw.write("-target: " + target + "\n\n")

                reference = target.split()
                hypothesis = pred.split()
                if len(reference) > 3 and len(hypothesis) > 3:
                    list_of_references.append([reference])
                    hypotheses.append(hypothesis)

        score = corpus_bleu(list_of_references, hypotheses)
        fw.write("Bleu Score = {}\n".format(100 * score))


if __name__ == "__main__":
    test()
