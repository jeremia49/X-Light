"""
mlp.py
======
Modul Multi-Layer Perceptron (MLP) untuk pemrosesan fitur dalam jaringan aktor/kritik.

Kelas:
    - MLPLayer : Blok MLP dengan Layer Normalization dan aktivasi, mendukung N lapisan.
    - MLPBase  : Pembungkus MLPLayer yang membaca konfigurasi dari argumen pelatihan.
"""
import torch.nn as nn
from .util import init, get_clones


class MLPLayer(nn.Module):
    """
    Blok Multi-Layer Perceptron dengan Layer Normalization.

    Arsitektur: Linear → Aktivasi → LayerNorm, diulang layer_N kali.

    :param input_dim:      (int) Dimensi fitur masukan.
    :param hidden_size:    (int) Dimensi lapisan tersembunyi.
    :param layer_N:        (int) Jumlah lapisan tersembunyi.
    :param use_orthogonal: (bool) Gunakan inisialisasi ortogonal.
    :param use_ReLU:       (bool) Gunakan ReLU bila True, Tanh bila False.
    """
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    """
    Pembungkus MLPLayer yang membaca konfigurasi dari argumen pelatihan.

    Mendukung normalisasi fitur masukan (LayerNorm) sebelum diproses oleh MLP.

    :param args:           (argparse.Namespace) Argumen pelatihan.
    :param obs_shape:      (tuple) Bentuk observasi masukan.
    :param cat_self:       (bool) Tidak digunakan (reserved).
    :param attn_internal:  (bool) Tidak digunakan (reserved).
    """
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x