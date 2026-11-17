"""Dump the first captured step's CUDA graphs (debug_dump .dot) and list
each segment's kernel sequence (short names), runs of eager aten kernels
marked, to see where the eager kernels sit between fused ones."""
import os, sys, re, collections, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import profile_isc_torchvic as P
from torchvic.core import graphsafe
OUT = os.environ.get('DUMP_DIR', '/tmp/gdump')
os.makedirs(OUT, exist_ok=True)
st = {'n': 0, 'done': False}
_begin = graphsafe.Segmenter.begin
def begin(self):
    self.cur = torch.cuda.CUDAGraph()
    if not st['done']:
        self.cur.enable_debug_mode()
    self.cur.capture_begin(pool=self.pool, capture_error_mode='thread_local')
graphsafe.Segmenter.begin = begin
_finish = graphsafe.Segmenter.finish
def finish(self):
    _finish(self)
    if not st['done']:
        for i, it in enumerate(self.items):
            it[1].debug_dump(os.path.join(OUT, 'seg%02d_%s.dot' % (i, it[0])))
        st['done'] = True
graphsafe.Segmenter.finish = finish
P.main()
def short(lbl):
    if 'triton' in lbl:
        m = re.search(r'(triton_\w+)', lbl); return m.group(1)[:60] if m else 'triton?'
    for pat in (r'(\w+Functor\w*)', r'(\w+_kernel\w*)', r'(Memcpy|memcpy|MEMCPY|memset|MEMSET)'):
        m = re.search(pat, lbl)
        if m: return 'A:' + m.group(1)[:50]
    return 'A:' + lbl[:50]
for f in sorted(glob.glob(os.path.join(OUT, '*.dot'))):
    txt = open(f).read()
    nodes = re.findall(r'"graph_\d+_node_(\d+)"\s*\[([^\]]*)\]', txt)
    seq = []
    for nid, attrs in nodes:
        m = re.search(r'label="([^"]*)"', attrs)
        lbl = m.group(1) if m else attrs
        seq.append(short(lbl))
    na = sum(1 for s in seq if s.startswith('A:'))
    print('== %s: %d nodes, %d non-triton' % (os.path.basename(f), len(seq), na))
    run = []
    for s in seq:
        print('   ' + s)
