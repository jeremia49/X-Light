"""
onpolicy.envs
==============
Sub-paket lingkungan simulasi untuk X-Light.

Sub-paket yang tersedia:
    - sumo_files_marl/ : Lingkungan lalu lintas berbasis SUMO (SUMOEnv,
                         TSCSimulator, Intersection).
    - env_wrappers     : Wrapper vektorisasi lingkungan (SubprocVecEnv,
                         DummyVecEnv, ShareVecEnv).
"""

import socket
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])


