#!/usr/bin/env python
"""Parse the CWC / NDSA National Register of Specified Dams (April 2025 PDF)
into a CSV.  The PDF text comes from `pdftotext -layout` (data only, nothing
in the PDF is executed).

Each dam is a block of 3-5 text lines: a main line starting with the serial
number and PIC code, one line above with the latitude (DMS, N) and one below
with the longitude (DMS, E).  The main line ends with
  height(m) length(m) gross(MCM) live(MCM) spillway(m3/s) purpose
Name pieces are collected from the name column (a window around the first
chunk after the PIC) in the block's lines.
"""
import csv
import re
import subprocess
import sys

MAIN = re.compile(r'^\s*(\d{1,5})\s+([A-Z]{2}\d{2}[A-Z]{2}\d{4}|[A-Z0-9]{6,14})\s+(\S.*)$')
LAT = re.compile(r"(\d{1,2})\s*°\s*(\d{1,2})\s*'\s*(\d{1,2}(?:\.\d+)?)\s*\"?\s*N")
LON = re.compile(r"(\d{1,3})\s*°\s*(\d{1,2})\s*'\s*(\d{1,2}(?:\.\d+)?)\s*\"?\s*E")
NUM = re.compile(r'^-?\d+(\.\d+)?$')


def dms(m):
    return float(m[0]) + float(m[1]) / 60 + float(m[2]) / 3600


def isnum(t):
    return t == '-' or bool(NUM.match(t))


def parse(txt):
    lines = txt.splitlines()
    mains = [i for i, ln in enumerate(lines) if MAIN.match(ln)
             and re.search(r"\d{4}\s+.*\d", ln)]
    out = []
    for n, i in enumerate(mains):
        lo = (mains[n - 1] + i) // 2 + 1 if n else i - 3
        hi = (i + mains[n + 1]) // 2 if n + 1 < len(mains) else i + 3
        lo, hi = max(lo, i - 3), min(hi, i + 3)
        blk = lines[lo:hi + 1]
        text = '\n'.join(blk)
        la = lo_ = None
        for q in range(i, max(i - 3, 0), -1):          # nearest latitude at or above
            la = LAT.search(lines[q])
            if la:
                break
        for q in range(i, min(i + 3, len(lines))):     # nearest longitude at or below
            lo_ = LON.search(lines[q])
            if lo_:
                break
        uc = bool(re.search(r'Construc', '\n'.join(lines[max(i - 1, 0):i + 2])))
        m = MAIN.match(lines[i])
        sr, pic, rest = m.group(1), m.group(2), m.group(3)
        toks = [t.rstrip('*') for t in rest.split()]
        # trailing 5 numeric tokens then purpose
        k = None
        for j in range(len(toks) - 5, -1, -1):
            if all(isnum(t) for t in toks[j:j + 5]):
                k = j
                break
        if k is None or not la or not lo_:
            out.append(dict(sr=sr, pic=pic, ok=0))
            continue
        h, ln_, gross, live, spill = toks[k:k + 5]
        purpose = ' '.join(toks[k + 5:])
        if not purpose:
            pm = re.findall(r'Irrig|Hydro|Multipur|Water Supply|Flood|Indust|Drinking|Recreat|Other', text)
            purpose = pm[-1] if pm else ''
        yr = None
        for t in toks[:k]:
            if re.fullmatch(r'(18|19|20)\d\d', t):
                yr = int(t)
        # name: chunks of the lines near the first chunk after the PIC
        pos = lines[i].index(pic) + len(pic)
        first = re.search(r'\S', lines[i][pos:]).start() + pos
        name = []
        for q in (i - 1, i, i + 1):
            if q < 0 or q >= len(lines):
                continue
            lim = pos if q == i else pos - 14
            for cm in re.finditer(r'\S+(?: \S+)*', lines[q]):
                s_ = cm.start()
                g = cm.group()
                if lim <= s_ <= first + 22 and not LAT.search(g) and not LON.search(g) \
                        and not isnum(g.split()[0]) and not re.search(r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}', g):
                    name.append(g)
                    break
        out.append(dict(
            sr=sr, pic=pic, ok=0 if uc else 1, name=' '.join(name), lat=round(dms(la.groups()), 6),
            lon=round(dms(lo_.groups()), 6), year=yr,
            height_m=None if h == '-' else float(h),
            length_m=None if ln_ == '-' else float(ln_),
            gross_mcm=None if gross == '-' else float(gross),
            live_mcm=None if live == '-' else float(live),
            purpose=purpose))
    return out


def main(pdf, outcsv):
    txt = subprocess.run(['pdftotext', '-layout', pdf, '-'], capture_output=True,
                         text=True, check=True).stdout
    rows = parse(txt)
    ok = [r for r in rows if r['ok']]
    seen = {}
    for r in ok:                      # the register repeats the important dams
        if r['pic'] not in seen or (r.get('gross_mcm') or 0) > (seen[r['pic']].get('gross_mcm') or 0):
            seen[r['pic']] = r
    ok = list(seen.values())
    keys = ['sr', 'pic', 'name', 'lat', 'lon', 'year', 'height_m', 'length_m',
            'gross_mcm', 'live_mcm', 'purpose']
    with open(outcsv, 'w', newline='') as f:
        w = csv.DictWriter(f, keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(ok)
    print('main lines %d, unique dams kept %d (rest: under construction, duplicates, unparsed)' % (len(rows), len(ok)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
