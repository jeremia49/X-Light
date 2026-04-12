"""
util.py (algorithms/utils)
===========================
Fungsi utilitas untuk inisialisasi bobot dan konversi tipe data dalam modul algoritma.
"""
import copy
import numpy as np

import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    """
    Inisialisasi bobot dan bias sebuah modul neural network.

    :param module:      (nn.Module) Modul yang akan diinisialisasi.
    :param weight_init: (callable) Fungsi inisialisasi bobot (misal: orthogonal_).
    :param bias_init:   (callable) Fungsi inisialisasi bias (misal: constant_(x, 0)).
    :param gain:        (float) Faktor penguatan untuk inisialisasi bobot.
    :return:            (nn.Module) Modul setelah diinisialisasi.
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    """
    Buat N salinan deep copy dari sebuah modul.

    :param module: (nn.Module) Modul yang akan disalin.
    :param N:      (int) Jumlah salinan.
    :return:       (nn.ModuleList) Daftar salinan modul.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    """
    Konversi numpy array ke torch Tensor jika diperlukan.

    :param input: (np.ndarray atau torch.Tensor) Data masukan.
    :return:      (torch.Tensor) Data sebagai torch Tensor.
    """
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
