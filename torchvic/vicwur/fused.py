"""Switch of the CUDA kernels that replace torch op chains of VIC-WUR
(wateruse_kernels: local water use and its ring edit, the dams' service
demand; efr_kernels: the EFR search). Each equals the torch path bit for
bit (tests/test_vicwur_fused_sweep.py); `ENABLED = False` runs the torch
path on CUDA as well."""
ENABLED = True
#: per kernel (with ENABLED): local water use, dams, EFR, the river
#: availability sum inside the water-use kernel, the P2 CUDA graph
WU = DAMS = EFR = SUM = GRAPH = True
