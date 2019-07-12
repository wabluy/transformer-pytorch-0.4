import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import get_positional_encoding, MultiHeadAttention, FeedForward, label_smoothing

PAD = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_key_and_query_masking(x, y):
    """Get key pad mask and query pad mask.

    Arguments:
        x: Source sentence. Tensor of shape=(batch_size, T_k)
        y: Target sentence. Tensor of shape=(batch_size, T_q)
    """
    _, T_k = x.size()
    _, T_q = y.size()
    key_pad_masks = x.eq(PAD).unsqueeze(1).repeat(1, T_q, 1)  # (N, T_q, T_k)
    query_pad_masks = y.eq(PAD).unsqueeze(2).repeat(1, 1, T_k)  # (N, T_q, T_k)
    return key_pad_masks.to(device), query_pad_masks.to(device)


class Transformer(nn.Module):
    def __init__(self, embedding_size, enc_voc_size, dec_voc_size, num_blocks=6, is_label_smooth=True):
        super(Transformer, self).__init__()
        self.enc_voc_size = enc_voc_size
        self.dec_voc_size = dec_voc_size
        self.num_blocks = num_blocks
        self.is_label_smooth = is_label_smooth

        # encoder
        self.enc_emb = nn.Embedding(enc_voc_size, embedding_size, padding_idx=0)
        self.enc_dropout = nn.Dropout(p=0.1)
        for i in range(num_blocks):
            self.__setattr__("enc_self_attention_{}".format(i), MultiHeadAttention(embedding_size, num_heads=8, causality=False))
            self.__setattr__("enc_feed_forward_{}".format(i), FeedForward(embedding_size, hiddens=[2048, embedding_size]))

        # decoder
        self.dec_emb = nn.Embedding(dec_voc_size, embedding_size, padding_idx=0)
        self.dec_dropout = nn.Dropout(p=0.1)
        for i in range(num_blocks):
            self.__setattr__("dec_self_attention_{}".format(i), MultiHeadAttention(embedding_size, num_heads=8, causality=True))
            self.__setattr__("dec_vanilla_attention_{}".format(i), MultiHeadAttention(embedding_size, num_heads=8, causality=False))
            self.__setattr__("dec_feed_forward_{}".format(i), FeedForward(embedding_size, hiddens=[2048, embedding_size]))

        # output
        self.fc = nn.Linear(embedding_size, dec_voc_size)

    def forward(self, x, y):
        """Transformer feed-forward function.

        Arguments:
            x: Tensor of shape=(batch_size, T_x)
            y: Tensor of shape=(batch_size, T_y)
            Both x and y are sentences with <EOS> as the last vocab!
            So we need to add <BOS> and remove <EOS> to y for the decoder input.
        Returns:
            mean_loss: A scalar tensor.
            preds: Tensor of shape=(batch_size, T_y)
            acc: A scalar tensor.
        """
        # No pad mask
        istarget = (1 - y.eq(PAD)).float().view(-1)  # (N*T, )
        # Backup labels for cal loss and acc
        labels = y.view(-1)  # (N*T, )

        # add <BOS> and remove <EOS>
        y = torch.cat((torch.ones(y.shape[0], 1, dtype=torch.int64).to(device) * 2, y[:, :-1]), dim=-1)  # (N, T_y)

        # encoder prepare
        enc_key_pad_masks, enc_query_pad_masks = _get_key_and_query_masking(x, x)  # (N, T_x, T_x)

        # decoder prepare
        dec_self_key_pad_masks, dec_self_query_pad_masks = _get_key_and_query_masking(y, y)  # (N, T_y, T_y)
        dec_key_pad_masks, dec_query_pad_masks = _get_key_and_query_masking(x, y)  # (N, T_y, T_x)

        # encoder
        x = self.enc_emb(x)  # (N, T_x, C)
        x = x + get_positional_encoding(x.shape[1], x.shape[2], padding_idx=None)
        x = self.enc_dropout(x)
        for i in range(self.num_blocks):
            x = self.__getattr__("enc_self_attention_{}".format(i))(
                x, x, x, key_pad_masks=enc_key_pad_masks, query_pad_masks=enc_query_pad_masks)
            x = self.__getattr__("enc_feed_forward_{}".format(i))(x)

        # decoder
        y = self.dec_emb(y)  # (N, T_y, C)
        y = y + get_positional_encoding(y.shape[1], y.shape[2], padding_idx=None)
        y = self.dec_dropout(y)
        for i in range(self.num_blocks):
            y = self.__getattr__("dec_self_attention_{}".format(i))(
                y, y, y, key_pad_masks=dec_self_key_pad_masks, query_pad_masks=dec_self_query_pad_masks)
            y = self.__getattr__("dec_vanilla_attention_{}".format(i))(
                y, x, x, key_pad_masks=dec_key_pad_masks, query_pad_masks=dec_query_pad_masks)
            y = self.__getattr__("dec_feed_forward_{}".format(i))(y)  # (N, T_y, C)

        logits = self.fc(y)  # (N, T_y, dec_voc_size)

        probs = F.softmax(logits, dim=-1).view(-1, self.dec_voc_size)  # (N*T, dec_voc_size)

        preds = probs.argmax(dim=-1)  # (N*T, )
        acc = (preds.eq(labels).float() * istarget).sum() / istarget.sum()  # (1, )

        y_onehot = torch.zeros(labels.shape[0], self.dec_voc_size).to(device).scatter_(1, labels.view(-1, 1), 1)  # (N*T, dev_voc_size)
        if self.is_label_smooth:
            y_onehot = label_smoothing(y_onehot)
        loss = -torch.sum(y_onehot * torch.log(probs), dim=-1)  # (N*T, )
        mean_loss = (loss * istarget).sum() / istarget.sum()  # (1, )

        return mean_loss, preds.view(logits.shape[0], -1), acc



