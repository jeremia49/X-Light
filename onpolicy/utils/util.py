"""
util.py (utils)
===============
Fungsi utilitas umum untuk pelatihan MARL dalam X-Light.

Fungsi yang tersedia:
    - check                 : Konversi numpy array ke torch Tensor.
    - get_gard_norm         : Hitung norma gradien dari parameter.
    - update_linear_schedule: Kurangi learning rate secara linear.
    - huber_loss            : Hitung Huber loss (lebih robust dari MSE).
    - mse_loss              : Hitung MSE loss sederhana.
    - get_shape_from_obs_space: Ekstrak bentuk dari gym.Space observasi.
    - get_shape_from_act_space: Ekstrak bentuk dari gym.Space aksi.
    - tile_images           : Gabungkan N gambar menjadi satu grid gambar.
"""
import numpy as np
import math
import torch


def check(input):
    """
    Konversi numpy array ke torch Tensor. Lewati jika sudah berupa Tensor.

    :param input: (np.ndarray atau torch.Tensor) Data masukan.
    :return:      (torch.Tensor) Data sebagai Tensor.
    """
    if type(input) == np.ndarray:
        return torch.from_numpy(input)


def get_gard_norm(it):
    """
    Hitung norma total gradien dari iterator parameter.

    :param it: (iterable) Iterator parameter torch (misal: model.parameters()).
    :return:   (float) Norma L2 dari semua gradien.
    """
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    """
    Hitung Huber loss: kuadratik untuk error kecil, linear untuk error besar.

    :param e: (torch.Tensor) Error (prediksi - target).
    :param d: (float) Threshold delta yang membatasi peralihan kuadratik/linear.
    :return:  (torch.Tensor) Nilai Huber loss per elemen.
    """
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


def mse_loss(e):
    """
    Hitung Mean Squared Error loss sederhana.

    :param e: (torch.Tensor) Error (prediksi - target).
    :return:  (torch.Tensor) Nilai MSE per elemen.
    """
    return e**2/2


def get_shape_from_obs_space(obs_space):
    """
    Ekstrak bentuk (shape) dari gym observation space.

    :param obs_space: (gym.Space) Ruang observasi.
    :return:          (tuple) Bentuk observasi.
    """
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    """
    Ekstrak bentuk (shape) dari gym action space.

    :param act_space: (gym.Space) Ruang aksi.
    :return:          (int atau tuple) Dimensi aksi.
    """
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c
