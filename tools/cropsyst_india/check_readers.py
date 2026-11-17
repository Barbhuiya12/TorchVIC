"""Parse every India file with the torchvic cropsyst readers (python -I)."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torchvic.crop.cropsyst import readers as rd, calendars as cal
D = os.path.join(os.environ.get('TORCHVIC_DATA_ROOT', 'data'), 'cropsyst_india/Database')
n = 0
for f in sorted(glob.glob(D + '/Crop/Name/*.crp')): rd.read_crp(f); n += 1
for f in sorted(glob.glob(D + '/Management/Name/*.mgt')): rd.read_mgt(f); n += 1
for f in sorted(glob.glob(D + '/Rotation/Name/*.rot')): rd.read_rot(f); n += 1
codes = sorted(int(os.path.basename(f).split('.')[0]) for f in glob.glob(D + '/Rotation/Code/*.CS_rotation'))
rs = cal.load_rotations(D, codes)
print('parsed', n, 'files; rotations loaded', len(codes), 'crops', rs.crops.n, 'mgts', rs.mgts.n)
