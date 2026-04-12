"""
onpolicy.algorithms.r_mappo
============================
Sub-paket implementasi algoritma R-MAPPO (Recurrent Multi-Agent PPO).

Modul yang tersedia:
    - r_mappo          : Kelas trainer R_MAPPO_SUMO dan R_MAPPO.
    - algorithm/
        - rMAPPOPolicy : Kelas policy R_MAPPOPolicy_SUMO dan R_MAPPOPolicy.
        - r_actor_critic: Jaringan aktor dan kritik (shared_NN, R_Actor_SUMO, dll).
        - sumo_nn      : Model Transformer encoder untuk SUMO (ActorModel, CriticModel).
        - GAT_nn       : Graph Attention Network untuk relasi spasial antar persimpangan.
"""
