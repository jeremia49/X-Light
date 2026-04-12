"""
rnn.py
======
Modul Recurrent Neural Network (RNN) berbasis GRU dengan Layer Normalization.

Digunakan untuk memproses sekuens observasi dengan mempertahankan hidden state
antar langkah waktu, memungkinkan agen untuk memiliki memori temporal.
"""
import torch
import torch.nn as nn


class RNNLayer(nn.Module):
    """
    Lapisan GRU dengan Layer Normalization dan reset hidden state otomatis.

    Mendukung reset hidden state pada batas episode menggunakan mask.
    Menangani dua kasus forward: satu timestep (inference) dan multi-step (training).

    :param inputs_dim:     (int) Dimensi fitur masukan.
    :param outputs_dim:    (int) Dimensi hidden state keluaran.
    :param recurrent_N:    (int) Jumlah lapisan GRU yang ditumpuk.
    :param use_orthogonal: (bool) Gunakan inisialisasi ortogonal untuk bobot GRU.
    """
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        """
        Jalankan GRU dengan reset hidden state otomatis berdasarkan mask episode.

        :param x:     (torch.Tensor) Fitur masukan, bentuk [B, inputs_dim] atau [T*B, inputs_dim].
        :param hxs:   (torch.Tensor) Hidden state sebelumnya, bentuk [B, recurrent_N, outputs_dim].
        :param masks: (torch.Tensor) Mask episode (0 = reset hidden state), bentuk [B, 1] atau [T, B].
        :return x:    (torch.Tensor) Keluaran GRU yang dinormalisasi.
        :return hxs:  (torch.Tensor) Hidden state terbaru.
        """
        if x.size(0) == hxs.size(0):
            x, hxs = self.rnn(x.unsqueeze(0),
                              (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs
