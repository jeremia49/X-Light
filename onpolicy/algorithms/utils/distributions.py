"""
distributions.py
================
Distribusi probabilitas aksi yang dimodifikasi agar kompatibel dengan kerangka MARL.

Menyediakan antarmuka seragam (sample, log_probs, mode, entropy) untuk berbagai
jenis distribusi yang digunakan pada ruang aksi yang berbeda:

    - FixedCategorical  : Distribusi diskrit (untuk aksi pilih-satu).
    - FixedNormal       : Distribusi Gaussian (untuk aksi kontinu).
    - FixedBernoulli    : Distribusi Bernoulli (untuk aksi biner).

Lapisan pembungkus (menghasilkan distribusi dari fitur neural network):
    - Categorical       : Menghasilkan FixedCategorical dari fitur tersembunyi.
    - DiagGaussian      : Menghasilkan FixedNormal dengan std yang dapat dipelajari.
    - Bernoulli         : Menghasilkan FixedBernoulli dari fitur tersembunyi.
    - AddBias           : Menambahkan bias yang dapat dipelajari ke tensor.
"""
import torch
import torch.nn as nn
from .util import init

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    """Distribusi Categorical dengan antarmuka yang disesuaikan untuk MARL."""

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    """Distribusi Normal dengan antarmuka yang disesuaikan untuk ruang aksi kontinu."""

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    """Distribusi Bernoulli dengan antarmuka yang disesuaikan untuk ruang aksi biner."""

    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    """
    Lapisan linear yang menghasilkan distribusi FixedCategorical dari fitur masukan.

    :param num_inputs:     (int) Dimensi fitur masukan.
    :param num_outputs:    (int) Jumlah kategori (aksi diskrit).
    :param use_orthogonal: (bool) Gunakan inisialisasi ortogonal.
    :param gain:           (float) Gain inisialisasi bobot lapisan output.
    """
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    """
    Lapisan yang menghasilkan distribusi Gaussian diagonal dari fitur masukan.

    Mean dihitung melalui lapisan linear; log-std dipelajari sebagai bias independen.

    :param num_inputs:     (int) Dimensi fitur masukan.
    :param num_outputs:    (int) Dimensi ruang aksi kontinu.
    :param use_orthogonal: (bool) Gunakan inisialisasi ortogonal.
    :param gain:           (float) Gain inisialisasi bobot.
    """
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    """
    Lapisan yang menghasilkan distribusi Bernoulli dari fitur masukan.

    :param num_inputs:     (int) Dimensi fitur masukan.
    :param num_outputs:    (int) Dimensi ruang aksi biner.
    :param use_orthogonal: (bool) Gunakan inisialisasi ortogonal.
    :param gain:           (float) Gain inisialisasi bobot.
    """
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    """
    Menambahkan bias yang dapat dipelajari ke tensor masukan.

    Digunakan oleh DiagGaussian untuk mempelajari log-std secara independen.

    :param bias: (torch.Tensor) Nilai awal bias.
    """
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
