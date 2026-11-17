"""Parse CWC weekly Reservoir Level & Storage bulletin PDFs (pdftotext -layout) into a tidy CSV.

Columns: reservoir, state, basin, date, level_m, live_storage_bcm, capacity_bcm, frl_m, bulletin, source
Layouts handled: (old, 2015-2019) FRL, level, cap, storage, date ; (new, 2020+) CCA, MW, FRL, cap, date, level, storage.
Usage: parse_cwc_bulletins.py PDF_DIR OUT_CSV
PDFs are only read with pdftotext (never executed).
"""
import csv, datetime as dt, re, subprocess, sys
from pathlib import Path

DATE = re.compile(r'(?<![\d/-])(\d{1,2})([-/])(\d{1,2})\2(20\d{2}|\d{2})(?!\d)')
NUM = re.compile(r'^-?\d+(?:\.\d+)?$')


def parse_date(m):
    a, sep, b, y = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    y = y + 2000 if y < 100 else y
    d, mo = (b, a) if sep == '/' else (a, b)  # slash = m/d/y, dash = d-m-y
    try:
        return dt.date(y, mo, d)
    except ValueError:
        return None


def fnum(s):
    return float(s) if NUM.match(s) else None


def parse_text(txt, bulletin):
    out = []
    for line in txt.splitlines():
        m = DATE.search(line)
        if not m:
            continue
        left, right = line[:m.start()], line[m.end():]
        date = parse_date(m)
        if date is None or not re.search(r'[A-Za-z]{3}', left):
            continue
        parts = [p for p in re.split(r'\s{2,}', left.strip()) if re.search(r'[A-Za-z]{2}', p)]
        if not parts:
            continue
        parts = [re.sub(r'[*†$#]+$', '', re.sub(r'^[*#@\d\s.]+', '', p)).strip() for p in parts]
        name = parts[0]
        parts = [name] + [p for p in parts[1:]]
        # tokens after the name: left of date numbers
        ltoks = left.strip().split()
        nums = []
        for t in reversed(ltoks):
            t2 = t.replace(',', '')
            if NUM.match(t2) or t2 == '-':
                nums.append(t2)
            else:
                break
        nums = nums[::-1]
        rt = right.split()
        state = basin = ''
        if len(parts) >= 2:
            state = parts[1]
        if len(parts) >= 3:
            basin = parts[2]
        new = len(rt) >= 2 and re.match(r'^\d+\.\d{2,3}$', rt[0]) and re.match(r'^\d+\.\d{3}$', rt[1])
        try:
            if new and len(nums) >= 2:
                frl, cap = fnum(nums[-2]), fnum(nums[-1])
                lvl, sto = fnum(rt[0]), fnum(rt[1])
            elif len(nums) >= 4:
                frl, lvl, cap, sto = (fnum(x) for x in nums[-4:])
            else:
                continue
        except Exception:
            continue
        if cap is None or sto is None:
            continue
        out.append(dict(reservoir=name, state=state, basin=basin, date=date.isoformat(), level_m=lvl,
                        live_storage_bcm=sto, capacity_bcm=cap, frl_m=frl, bulletin=bulletin,
                        source='CWC weekly bulletin'))
    return out


def main(pdf_dir, out_csv):
    rows, bad = [], []
    for p in sorted(Path(pdf_dir).glob('*.pdf')):
        try:
            txt = subprocess.run(['pdftotext', '-layout', str(p), '-'], capture_output=True, text=True, timeout=120).stdout
        except Exception as e:
            bad.append((p.name, str(e)))
            continue
        r = parse_text(txt, p.name)
        if not r:
            bad.append((p.name, 'no rows'))
        seen = set()
        for x in r:  # bulletins repeat the table; keep first
            k = (x['reservoir'], x['date'], x['level_m'], x['live_storage_bcm'])
            if k not in seen:
                seen.add(k)
                rows.append(x)
    cols = ['reservoir', 'state', 'basin', 'date', 'level_m', 'live_storage_bcm', 'capacity_bcm', 'frl_m', 'bulletin', 'source']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} rows from {len(list(Path(pdf_dir).glob("*.pdf")))} pdfs; {len(bad)} with no rows: {bad[:10]}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
