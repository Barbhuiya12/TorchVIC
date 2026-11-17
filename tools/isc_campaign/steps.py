"""The build-up matrix: one component added per step (ISC, VIC-WUR group)."""
import os
import re

R = os.environ.get('TORCHVIC_DATA_ROOT', 'data').rstrip('/') + '/'
SMOKE = R + 'vicwur/isc/smoke/'
OUT = R + 'verification/isc_campaign/'
FULL_GLOBAL = SMOKE + 'global_isc_2005.txt'          # every plugin on, 2005

#: spin-up: 1 Oct 2004 - 31 Dec 2004 (92 days), then the year 2005
START = (2004, 10, 1)
END = (2005, 12, 31)
SPIN_DAYS = 92

#: component -> (global-file keys switched, plugin forcing names, path keys)
COMP = {
    'routing': dict(flags=['ROUTING'], keys=[]),
    'wateruse': dict(flags=['WATERUSE', 'LOCAL_WITHDRAWAL'],
                     keys=[], plug=r'(MUN|LIV|ENE|MAN)_(DEMAND|GROUNDWATER|CONSUMPTION)'),
    # no OFFSET_IRRIGATION: irrigation_offset is not in the verified irr.nc
    'irrigation': dict(flags=['IRRIGATION', 'EFFICIENT_IRRIGATION'],
                       keys=['IRRIGATION_PARAMETERS']),
    'dams': dict(flags=['DAMS'], keys=['DAMS_PARAMETERS']),
    'efr': dict(flags=['EFR'], keys=[], plug=r'EFR_(DISCHARGE|BASEFLOW)'),
    'nonrenew': dict(flags=['NONRENEWABLE_WITHDRAWAL'], keys=[]),
    'groundwater': dict(flags=['GWM'], keys=[]),
    'landuse': dict(flags=['LANDUSE_FORCE'], keys=[], plug=r'CV'),
    'co2': dict(flags=[], keys=[], plug=r'CO2'),
    # CropSyst: no global-file switch, the run adds the vicwur_cropsyst member
    'cropsyst': dict(flags=[], keys=[]),
    'wofost': dict(flags=['WOFOST'],
                   keys=['CROP_PARAMETERS', 'WOFOST_TEXT_PARAMETERS']),
}
ORDER = ['routing', 'wateruse', 'irrigation', 'dams', 'efr', 'groundwater',
         'landuse', 'wofost', 'cropsyst']

#: step -> (components on, mode)
STEPS = {
    'S0': ([], 'fixed'),
    'S1': (['routing'], 'fixed'),
    'S2': (['routing', 'wateruse'], 'fixed'),
    'S3': (['routing', 'wateruse', 'irrigation'], 'fixed'),
    'S4': (['routing', 'wateruse', 'irrigation', 'dams'], 'fixed'),
    'S5': (['routing', 'wateruse', 'irrigation', 'dams', 'efr',
            'nonrenew'], 'fixed'),
    # landuse on S5 without groundwater: attributes S7's failure
    'S5L': (['routing', 'wateruse', 'irrigation', 'dams', 'efr', 'nonrenew',
             'landuse', 'co2'], 'fixed'),
    # CO2 alone on S5, landuse alone on S5 (the NaN of the year change)
    'S5C': (['routing', 'wateruse', 'irrigation', 'dams', 'efr', 'nonrenew',
             'co2'], 'fixed'),
    'S5U': (['routing', 'wateruse', 'irrigation', 'dams', 'efr', 'nonrenew',
             'landuse'], 'fixed'),
    'S6x': (['routing', 'wateruse', 'irrigation', 'dams', 'efr', 'nonrenew',
             'groundwater'], 'exact'),
    'S6f': (['routing', 'wateruse', 'irrigation', 'dams', 'efr', 'nonrenew',
             'groundwater'], 'fixed'),
    'S7': (['routing', 'wateruse', 'irrigation', 'dams', 'efr', 'nonrenew',
            'groundwater', 'landuse', 'co2'], 'fixed'),
    # WOFOST on S6f (no land use): S7's land use fails (NaN), see ISSUES
    'S8n': (['routing', 'wateruse', 'irrigation', 'dams', 'efr', 'nonrenew',
             'groundwater', 'co2', 'wofost'], 'fixed'),
    'S8': (['routing', 'wateruse', 'irrigation', 'dams', 'efr', 'nonrenew',
            'groundwater', 'landuse', 'co2', 'wofost'], 'fixed'),
    # S9 without groundwater (attributes a failure to the crop or to MODFLOW)
    'S9n': (['routing', 'wateruse', 'irrigation', 'dams', 'efr', 'nonrenew',
             'landuse', 'co2', 'cropsyst'], 'fixed'),
    # VIC-WUR stack with CropSyst on the three crop tiles, irrigation scheme 'cropsyst', no WOFOST
    'S9': (['routing', 'wateruse', 'irrigation', 'dams', 'efr', 'nonrenew',
            'groundwater', 'landuse', 'co2', 'cropsyst'], 'fixed'),
}
#: the step each step is diffed against (what it adds is the attribution)
PREV = {'S1': 'S0', 'S2': 'S1', 'S3': 'S2', 'S4': 'S3', 'S5': 'S4',
        'S6x': 'S5', 'S6f': 'S5', 'S7': 'S6f', 'S8': 'S7', 'S8n': 'S6f', 'S9': 'S7', 'S9n': 'S5L'}
ADDS = {'S1': 'routing', 'S2': 'wateruse_local', 'S3': 'irrigation',
        'S4': 'dams', 'S5': 'efr+nonrenew', 'S6x': 'groundwater (exact)',
        'S6f': 'groundwater (fixed)', 'S7': 'landuse+co2', 'S8': 'wofost', 'S8n': 'co2+wofost (no land use)',
        'S9': 'cropsyst (irrigation scheme cropsyst)', 'S9n': 'cropsyst (no groundwater)'}
# S0 has no plugin module; its attribution is 'vic'
ADDS['S0'] = 'vic'

ALL_FLAGS = sorted({f for c in COMP.values() for f in c['flags']})


def make_global(step, dst, forcing_dir=OUT + 'forcing/', start=START,
                end=END, out_dir=None, extra=None, subset=None):
    """Write the global file of `step` to `dst` from the full smoke file."""
    on, mode = STEPS[step]
    on = set(on)
    flags = {}
    for c, d in COMP.items():
        for f in d['flags']:
            flags[f] = 'TRUE' if c in on else 'FALSE'
    flags['NONRENEWABLE_WITHDRAWAL'] = 'TRUE' if 'nonrenew' in on else 'FALSE'
    flags['REMOTE_WITHDRAWAL'] = 'FALSE'
    drop_keys = {k for c, d in COMP.items() if c not in on
                 for k in d['keys']}
    drop_plug = [re.compile(d['plug']) for c, d in COMP.items()
                 if c not in on and d.get('plug')]
    if 'landuse' not in on:
        pass
    if 'wofost' not in on:
        drop_keys |= {'WOFOST_STEPS_PER_DAY', 'WOFOST_DIST_SEASON',
                      'WOFOST_DIST_TSUM', 'WOFOST_DIST_FERTILIZER',
                      'WOFOST_DIST_MINERALIZATION'}
    if 'irrigation' not in on:
        drop_keys |= {'PBARE_IRR'}
    lines = []
    for ln in open(FULL_GLOBAL):
        w = ln.split()
        if w and not ln.startswith('#'):
            k = w[0]
            if k in drop_keys:
                continue
            if k == 'PLUGIN_FORCE_TYPE' and any(p.fullmatch(w[1])
                                                for p in drop_plug):
                continue
            if k in flags:
                ln = '%s %s\n' % (k, flags[k])
            elif k == 'STARTYEAR':
                ln = 'STARTYEAR %d\n' % start[0]
            elif k == 'STARTMONTH':
                ln = 'STARTMONTH %d\n' % start[1]
            elif k == 'STARTDAY':
                ln = 'STARTDAY %d\n' % start[2]
            elif k == 'ENDYEAR':
                ln = 'ENDYEAR %d\n' % end[0]
            elif k == 'ENDMONTH':
                ln = 'ENDMONTH %d\n' % end[1]
            elif k == 'ENDDAY':
                ln = 'ENDDAY %d\n' % end[2]
            elif k == 'PLUGIN_FORCE_TYPE' and w[1] == 'CV' and os.environ.get('TV_CV_DIR'):
                ln = ln.replace(R + 'vicwur/isc/forcing/', os.environ['TV_CV_DIR'])
            elif k == 'FORCE_TYPE' or (k == 'PLUGIN_FORCE_TYPE'
                                       and w[1].startswith('EFR_')):
                ln = ln.replace(SMOKE + 'forcing/', forcing_dir)
            elif subset and k == 'DOMAIN':
                ln = 'DOMAIN %sdomain.nc\n' % subset
            elif subset and k == 'PARAMETERS':
                ln = 'PARAMETERS %sparams.nc\n' % subset
            elif subset and k == 'ROUTING_PARAMETERS':
                ln = 'ROUTING_PARAMETERS %srout_n4.nc\n' % subset
            elif k in ('RESULT_DIR', 'LOG_DIR') and out_dir:
                ln = '%s %s/\n' % (k, out_dir)
        lines.append(ln)
    for k, v in (extra or {}).items():
        lines.append('%s %s\n' % (k, v))
    open(dst, 'w').write(''.join(lines))
    return mode
