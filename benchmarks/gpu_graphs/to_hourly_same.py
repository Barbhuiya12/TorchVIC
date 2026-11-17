#!/usr/bin/env python3
"""mtclim.to_hourly (hours summed per day of year, then picked) against the
form it replaced (the days picked, then their hours summed), bit for bit,
on the device given (default cuda), at several cell counts and offsets."""
import sys

import numpy as np
import torch

from torchvic.forcing import mtclim


def old(s_srad, s_dayl, yday, tiny_radfract, tiny_offset):
    tiny_ph = 120
    idx = torch.as_tensor([int(d) - 1 for d in yday], dtype=torch.long,
                          device=s_srad.device)
    # the rotation runs through the flattened year, as MetSim's does
    if tiny_offset:
        flat = tiny_radfract.reshape(*tiny_radfract.shape[:-2], -1)
        tiny_radfract = torch.roll(flat, shifts=tiny_offset,
                                   dims=-1).reshape(tiny_radfract.shape)
    frac = tiny_radfract.index_select(-2, idx)
    hf = frac.reshape(*frac.shape[:-1], 24, tiny_ph).sum(-1)
    tmp = s_srad * s_dayl / 3600.0
    return hf * tmp.unsqueeze(-1)


def main():
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else 'cuda')
    rng = np.random.default_rng(1)
    bad = 0
    for n in (7, 256, 1024):
        lat = rng.uniform(8.0, 37.6, n)
        elev = rng.uniform(0.0, 5000.0, n)
        g = mtclim.solar_geometry(lat, elev, device=dev)
        nd = 911
        yday = [(i + 275) % 365 + 1 for i in range(nd)]
        s_srad = torch.as_tensor(rng.uniform(0, 400, (n, nd)), device=dev)
        s_dayl = torch.as_tensor(rng.uniform(3e4, 5e4, (n, nd)), device=dev)
        for off_h in (0.0, 5.5, -3.25):
            new = mtclim.to_hourly(s_srad, s_dayl, yday, g['tiny_radfract'],
                                   hour_offset=off_h)
            import ctypes
            to = int(ctypes.c_float(ctypes.c_float(120.0).value
                                    * off_h).value)
            ref = old(s_srad, s_dayl, yday, g['tiny_radfract'], to)
            same = torch.equal(new, ref)
            bad += not same
            print('cells %5d offset %+5.2f: %s'
                  % (n, off_h, 'identical' if same else
                     'DIFF max %.3e' % float((new - ref).abs().max())),
                  flush=True)
        del g
        torch.cuda.empty_cache() if dev.type == 'cuda' else None
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
