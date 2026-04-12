"""
onpolicy
========
Paket utama X-Light: implementasi Multi-Agent Reinforcement Learning (MARL)
berbasis PPO untuk pengendalian lampu lalu lintas (Traffic Signal Control).

Sub-paket:
    - algorithms : Implementasi algoritma RL (MAPPO, IPPO, RMAPPO)
    - envs       : Lingkungan simulasi lalu lintas berbasis SUMO
    - runner     : Loop pelatihan dan evaluasi
    - scripts    : Skrip entry-point untuk memulai pelatihan
    - utils      : Buffer pengalaman, normalisasi nilai, dan utilitas umum
    - config     : Parser argumen konfigurasi global
"""
from onpolicy import algorithms, envs, runner, scripts, utils, config


__version__ = "0.1.0"

__all__ = [
    "algorithms",
    "envs",
    "runner",
    "scripts",
    "utils",
    "config",
]