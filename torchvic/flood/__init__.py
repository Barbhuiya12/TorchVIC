"""TorchVIC flood hydraulics: a local-inertial (ACC) solver that runs in
process on the GPU (`engine.FloodEngine`, fused Triton step on CUDA) and DEM
conditioning for it (`dem`).

Interface of the flood connector (torchvic/coupling/connectors/flood.py),
one coupling interval of dt seconds:

    eng = FloodEngine(dem, dx, manning, free_edges={'W': 1e-4},
                      inflow_points=[(row, col), ...], device='cuda')
    out = eng.run(dt, runoff=V_cells_m3, inflow=V_points_m3)

    connector input   runoff  m3 per hydraulic cell over the interval
                      inflow  m3 per inflow point over the interval
    connector output  depth      out['depth']      m, cell, interval end
                      max_depth  out['max_depth']  m, interval maximum
                      speed      out['speed']      m/s, cell centre
                      outflow    out['outflow']    m3 left through open edges
    balance           out['ledger']: rain, inflow, outflow, clamp,
                      stored0, stored1, residual (float64 sums)
    restart           eng.state_dict() / eng.load_state_dict(): depth and
                      momentum, so a resumed run continues bit-identically

The state (depth, face momentum) stays on the device between intervals, so
daily exchange keeps the momentum (LISFLOOD-FP restarts from rest).
"""
from .engine import FloodEngine, SCHEMES, G  # noqa: F401
from . import dem  # noqa: F401
