#!/usr/bin/env python3
"""One verification-matrix row, TorchVIC against pristine VIC, per variable.

    outvars_check.py ROW_ID DAYS [vic|tv|both|cmp] ["extra torchvic args"]
                     [VAR,VAR,...]

Runs the row (verification/matrix.py) for DAYS days -- VIC once per
(row, DAYS), kept under $CHK_ROOT/<row>/vic -- and TorchVIC from the
checkout this file is in, then prints the compare.py grade, the VIC
variables TorchVIC did not write, and for every variable both wrote (or the
VAR list: exact names, prefixes or substrings) the largest absolute
difference, the VIC maximum and the NaN-pattern mismatches. With no VAR list
only the variables that differ by more than 1e-10 are printed. CHK_START
(`2008,7,15`) moves the row's start date. Used for tests/test_out_vars.py's
sibling checks and by hand while adding an output variable.
"""
import sys, os, json, subprocess, time
WT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONPATH']=WT+os.pathsep+os.environ.get('PYTHONPATH','')
sys.path.insert(0, WT+'/verification'); sys.path.insert(0, WT)
import numpy as np
import matrix as M, run_matrix as R, compare as C
import xarray as xr
rid, days = sys.argv[1], int(sys.argv[2])
mode = sys.argv[3] if len(sys.argv)>3 else 'both'
extra = sys.argv[4].split() if len(sys.argv)>4 else []
root = os.environ.get('CHK_ROOT', os.path.join(M.WORK, 'out_vars', 'runs'))
r=[x for x in M.MATRIX if x['id']==rid][0]
if os.environ.get('CHK_START'):
    import datetime as _d
    y,m,dd=map(int,os.environ['CHK_START'].split(','))
    M.CASES[r['case']]=dict(M.CASES[r['case']],start=(y,m,dd))
    root=root+'_'+os.environ['CHK_START'].replace(',','-')
d=R.prepare(r, days, root=root)
if mode in('both','vic') and not os.path.exists(d+'/vic_done_%d'%days):
    for f in os.listdir(d+'/vic'): os.remove(d+'/vic/'+f)
    print('vic', R.run_vic(r,d)); open(d+'/vic_done_%d'%days,'w').write('1')
if mode in('both','tv'):
    for f in os.listdir(d+'/tv'): os.remove(d+'/tv/'+f)
    c = M.CASES[r['case']]
    steps = R.period_days(r, days) * c['forcing_spd']
    cmd=[R.PY,'-m','torchvic','run',d+'/global.tv.txt','--core',r['core'],'--device','cpu','--out',d+'/tv','--steps',str(steps)]+extra
    if r['daily'] and 'noflag' not in r['id']: cmd.append('--daily')
    env=R._env(8); env['TORCHVIC_NUM_THREADS']='8'
    t0=time.time()
    with open(d+'/tv.log','w') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=d)
    print('tv rc',p.returncode, round(time.time()-t0,1))
res=C.compare_dirs(d+'/vic', d+'/tv', r, {})
print('grade',res.get('grade'))
print('missing',res.get('not_written_by_torchvic'))
vic=xr.open_mfdataset(d+'/vic/*.nc',combine='by_coords') if False else None
import glob
def op(dd):
    fs=sorted(glob.glob(dd+'/*.nc')); return xr.open_mfdataset(fs,combine='by_coords')
vic, tv = op(d+'/vic'), op(d+'/tv')
mask=np.isfinite(vic['OUT_PREC'].values[0])
rows=[]
for v in sorted(vic.data_vars):
    if not v.startswith('OUT_') or v not in tv.data_vars: continue
    a=vic[v].values; b=tv[v].values
    a=a[...,mask]; b=b[...,mask]
    with np.errstate(all='ignore'):
        dif=np.nanmax(np.abs(a-b)) if a.size else 0
    nanmis=int((np.isnan(a)!=np.isnan(b)).sum())
    mx=np.nanmax(np.abs(a)) if np.isfinite(a).any() else 0
    rows.append((v,dif,mx,nanmis))
want=sys.argv[5].split(',') if len(sys.argv)>5 else None
for v,dif,mx,nm in rows:
    if want is None:
        if dif>1e-10 or nm: print('  %-26s maxabs %.3e  vicmax %.3e nanmismatch %d'%(v,dif,mx,nm))
    elif any(v==w or v.startswith(w) or w in v for w in want):
        print('  %-26s maxabs %.3e  vicmax %.3e nanmismatch %d'%(v,dif,mx,nm))
