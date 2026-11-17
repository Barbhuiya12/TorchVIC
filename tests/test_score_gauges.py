"""tools/score_gauges.py: a perfect simulation scores KGE = NSE = 1, PBIAS = 0; canals and gauges
not flagged usable == 1.0 are skipped; --sim-start sets the sim calendar, not the scoring window."""
import os
import subprocess
import sys
from datetime import date, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, '..', 'tools', 'score_gauges.py')


def _case(tmp_path, index_rows, extra=(), sim_first=date(2004, 1, 1)):
    d0 = date(2004, 1, 1)
    q = 10 + 5 * np.sin(np.arange(366) / 20.0)
    (tmp_path / 'obs').mkdir(exist_ok=True)
    with open(tmp_path / 'sim.csv', 'w') as f:
        f.write('date,1,404,478\n')
        for i, v in enumerate(q):
            f.write('%s,%g,%g,%g\n' % (sim_first + timedelta(days=i), v, v, v))
    for g in (1, 404, 478):
        with open(tmp_path / 'obs' / ('%d.csv' % g), 'w') as f:
            f.write('Date,Q\n')
            for i, v in enumerate(q):
                f.write('%s,%.10g\n' % (d0 + timedelta(days=i), v))
    with open(tmp_path / 'index.csv', 'w') as f:
        f.write('clean_id,name,lat,lng,area,usable\n' + index_rows)
    return subprocess.run([sys.executable, TOOL, '--sim', str(tmp_path / 'sim.csv'), '--obs-dir', str(tmp_path / 'obs'),
                           '--index', str(tmp_path / 'index.csv'), '--out-dir', str(tmp_path / 'out'), *extra],
                          capture_output=True, text=True)


def _rows(tmp_path):
    return open(tmp_path / 'out' / 'gauge_scores.csv').read().strip().split('\n')


def test_perfect_and_canal(tmp_path):
    p = _case(tmp_path, '1,a,20,80,100,1.0\n478,canal,21,81,6,1.0\n')
    assert p.returncode == 0, p.stderr
    rows = _rows(tmp_path)
    assert len(rows) == 2 and rows[1].startswith('1,a,')
    v = dict(zip(rows[0].split(','), rows[1].split(',')))
    assert abs(float(v['kge']) - 1) < 1e-3 and abs(float(v['nse']) - 1) < 1e-3 and abs(float(v['pbias'])) < 1e-3


def test_unreviewed_gauges_skipped_unless_all_gauges(tmp_path):
    idx = '1,a,20,80,100,1.0\n404,b,21,81,6,False\n'
    assert _case(tmp_path, idx).returncode == 0
    assert len(_rows(tmp_path)) == 2
    assert _case(tmp_path, idx, ['--all-gauges']).returncode == 0
    assert len(_rows(tmp_path)) == 3


def test_scoring_start_does_not_move_the_sim_and_empty_is_clean(tmp_path):
    idx = '1,a,20,80,100,1.0\n'
    p = _case(tmp_path, idx, ['--start', '2004-03-01'])
    assert p.returncode == 0, p.stderr
    v = dict(zip(_rows(tmp_path)[0].split(','), _rows(tmp_path)[1].split(',')))
    assert abs(float(v['kge']) - 1) < 1e-3
    p = _case(tmp_path, '404,b,21,81,6,False\n')
    assert p.returncode == 0, p.stderr
    assert 'no gauge scored' in open(tmp_path / 'out' / 'summary.txt').read()
