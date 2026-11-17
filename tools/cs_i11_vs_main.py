"""Core-hooks check: VIC-only runs of this tree vs main, bit for bit.
usage: cs_i11_vs_main.py save <out.pt> | cmp <a.pt> <b.pt>
Runs G0c (VIC-WUR R1) on the padded grid (CPU and CUDA), the lanes and the
CUDA graphs, with no hook object and with an unrelated hook (water_stress)."""
import os
import sys
import tempfile

import torch

VER = os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'verification')
G0C = VER + '/vicwur_19/gruns/R1/G0c'


def save(out):
    from torchvic import case
    from torchvic.core import version
    from torchvic.model import basin
    from torchvic.vicwur import core as VC
    gf = VC.image_global(G0C + '/global_W3.txt',
                         tempfile.mkdtemp() + '/global.txt')
    res = {}
    devs = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for dev in devs:
        if dev == 'cuda':
            torch.set_default_device('cuda')
        c = case.load(gf, device=torch.device(dev), core='vicwur3', steps=40)
        modes = ['padded'] if dev == 'cpu' else ['padded', 'lanes', 'graphs']
        for mode in modes:
            basin.VW_LANES = mode != 'padded'
            basin.GRAPHS = mode == 'graphs'
            for name, mk in (('none', lambda: None),
                             ('ws', lambda: VC.VicwurCore(water_stress=True))):
                with torch.no_grad():
                    o = c.run(p=c.params(), vicwur_core=mk(), contract=False)
                for k, v in o.items():
                    if torch.is_tensor(v):
                        res['%s/%s/%s/%s' % (dev, mode, name, k)] = v.cpu()
                for k, v in (o.get('vicwur') or {}).items():
                    if torch.is_tensor(v):
                        res['%s/%s/%s/vw.%s' % (dev, mode, name, k)] = v.cpu()
    del version
    torch.save(res, out)
    print('saved', len(res))


def cmp(a, b):
    A, B = torch.load(a), torch.load(b)
    bad = [k for k in A if k not in B or not torch.equal(A[k], B[k])]
    print('compared %d arrays, %d differ, %d only in one' % (
        len(A), len(bad), len(set(A) ^ set(B))))
    for k in bad[:10]:
        print('  DIFF', k)
    sys.exit(1 if bad or set(A) ^ set(B) else 0)


if __name__ == '__main__':
    if sys.argv[1] == 'save':
        save(sys.argv[2])
    else:
        cmp(sys.argv[2], sys.argv[3])
