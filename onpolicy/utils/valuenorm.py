"""
valuenorm.py (utils)
=====================
Normalisasi nilai (value normalization) menggunakan running statistics.

Modul ini mengimplementasikan kelas ValueNorm yang melacak mean dan varians
secara online dengan exponential moving average (EMA). Digunakan sebagai
alternatif PopArt untuk menormalisasi prediksi nilai kritik selama pelatihan.

Referensi:
    - "Improving Sample Efficiency in Model-Free Reinforcement Learning from Images"
      (Laskin et al., 2020) — teknik debiased running statistics.
"""
import numpy as np

import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """
    Normalisasi vektor nilai menggunakan exponential moving average.

    Melacak mean dan mean-of-squares secara online dan menggunakannya
    untuk menormalisasi (normalize) dan mengembalikan (denormalize) nilai.

    :param input_shape:         (int) Dimensi input.
    :param norm_axes:           (int) Jumlah dimensi awal untuk dinormalisasi (default: 1).
    :param beta:                (float) Faktor peluruhan EMA (default: 0.99999).
    :param per_element_update:  (bool) Jika True, hitung bobot per elemen batch.
    :param epsilon:             (float) Nilai kecil untuk stabilitas numerik.
    """

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reset semua running statistics ke nol."""
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        """
        Hitung mean dan varians ter-debiased dari running statistics.

        :return: (tuple) (debiased_mean, debiased_var) sebagai torch.Tensor.
        """
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):
        """
        Perbarui running statistics dengan batch data baru.

        :param input_vector: (np.ndarray atau torch.Tensor) Vektor nilai baru.
        """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device)  # not elegant, but works in most cases

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        """
        Normalisasi vektor input menggunakan running mean dan standar deviasi.

        :param input_vector: (np.ndarray atau torch.Tensor) Data yang akan dinormalisasi.
        :return:             (torch.Tensor) Data ternormalisasi.
        """
        # Make sure input is float32
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device)  # not elegant, but works in most cases

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(self.running_mean.device)  # not elegant, but works in most cases

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        out = out.cpu().numpy()
        
        return out
