"""Energy to solution from the sampled GPU power and the run marks (see docs/performance.md)."""
import csv, datetime, sys
R = "results_energy/"
CPU_W_CORE = 500.0 / 192.0        # EPYC 9965: 500 W TDP, 192 cores per socket
VIC_W = 2 * 500.0                 # two packages at TDP, all 384 cores busy
def marks(tag):
    m = {}
    for l in open(R + tag + "_marks.txt"):
        k, v = l.split(); m[k] = float(v)
    return m
def load(tag):
    t, p = [], []
    for row in csv.reader(open(R + tag + "_gpu_power.csv")):
        try:
            t.append(datetime.datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()); p.append(float(row[1]))
        except Exception: pass
    return t, p
def integ(t, p, a, b):
    e = 0.0
    for i in range(1, len(t)):
        if t[i-1] >= a and t[i] <= b: e += 0.5 * (p[i] + p[i-1]) * (t[i] - t[i-1])
    return e
out = []
for tag in sys.argv[1:]:
    m = marks(tag); t, p = load(tag)
    idle = [x for ti, x in zip(t, p) if ti < m["idle_end"] - 2]
    pidle = sum(idle) / len(idle)
    for r in (1, 2):
        a, b = m["start_r%d" % r], m["end_r%d" % r]
        w = b - a; eg = integ(t, p, a, b); ec = 4 * CPU_W_CORE * w
        out.append((tag, r, w, eg / w, eg, eg - pidle * w, pidle, ec))
        print("%s r%d wall %.1f s  GPU avg %.1f W  GPU %.2f kJ (%.2f Wh)  net of idle %.2f kJ  idle %.1f W  +CPU4 share %.2f kJ  total %.2f kJ (%.2f Wh)" %
              (tag, r, w, eg / w, eg/1e3, eg/3.6e3, (eg - pidle*w)/1e3, pidle, ec/1e3, (eg+ec)/1e3, (eg+ec)/3.6e3))
