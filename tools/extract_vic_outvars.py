#!/usr/bin/env python3
"""Read VIC 5's own output-variable tables and write them down.

`torchvic.io.history` writes VIC's history files: every OUT_* variable a global
file names with OUTVAR, aggregated over AGGFREQ by its AGG_TYPE, with the
units and names VIC gives it. Four facts about each variable come from the
C, not from us, and this script reads each from the file VIC compiles:

* **that it exists**, and its position -- the `OUT_*` enum in
  `vic_driver_shared_all.h`. VIC log_errs on an OUTVAR not in that list
  (set_output_var, vic_history.c).
* **its default aggregation** -- the `switch` in
  `get_default_outvar_aggtype` (vic_history.c): AGG_TYPE_END for the
  states, AGG_TYPE_SUM for the water fluxes, AGG_TYPE_AVG for everything
  else. `torchvic.io.history` carries the same two sets written out, and
  tests/test_output_aggregation.py fails if they drift from this.
* **its metadata** -- long_name, standard_name, units and description,
  the `strcpy(out_metadata[OUT_X].field, "...")` lines of
  history_metadata.c; `nelem`, the element count VIC gives a layered
  variable, from the same file.
* **its netCDF dimensions** -- the `switch` in `set_nc_var_dimids`
  (vic_nc_info.c): (time, nlayer, y, x) for the layer variables, and so on.

    tools/extract_vic_outvars.py              # write torchvic/data/vic5_outvars.json
    tools/extract_vic_outvars.py --check      # compare, write nothing
    tools/extract_vic_outvars.py --vic <VIC tree>/vic
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_OUT = os.path.join(REPO, 'torchvic', 'data', 'vic5_outvars.json')
HEADER = os.path.join('drivers', 'shared_all', 'include',
                      'vic_driver_shared_all.h')
HISTORY = os.path.join('drivers', 'shared_all', 'src', 'vic_history.c')
METADATA = os.path.join('drivers', 'shared_all', 'src', 'history_metadata.c')
NC_INFO = os.path.join('drivers', 'shared_image', 'src', 'vic_nc_info.c')
VERSION_H = os.path.join('drivers', 'shared_all', 'include', 'vic_version.h')


def default_vic_tree():
    """The pristine tree under $TORCHVIC_REFERENCE_DIR (default
    <repository>/reference), or None."""
    root = os.environ.get('TORCHVIC_REFERENCE_DIR',
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reference'))
    p = os.path.join(root, 'VIC-5.1.0', 'vic')
    return os.path.abspath(p) if os.path.isdir(p) else None


def _strip_comments(text):
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    return re.sub(r'//[^\n]*', ' ', text)


def outvar_enum(vic):
    """Every OUT_* name, in enum order (vic_driver_shared_all.h)."""
    text = _strip_comments(open(os.path.join(vic, HEADER)).read())
    for body in re.findall(r'enum\s*\{(.*?)\}', text, flags=re.S):
        names = [n.strip() for n in body.split(',') if n.strip()]
        if 'N_OUTVAR_TYPES' in names:
            return [n for n in names if n != 'N_OUTVAR_TYPES']
    raise SystemExit('no OUT_* enum ending in N_OUTVAR_TYPES in %s' % HEADER)


def _switch_groups(text, func, assign):
    """{value: [case labels]} of the switch inside `func`, where each group
    of `case X:` labels ends in `<assign> = VALUE;` or `default:`."""
    m = re.search(r'\n%s\s*\(.*?\n\}' % func, text, flags=re.S)
    if not m:
        raise SystemExit('function %s not found' % func)
    body, groups, labels = m.group(0), {}, []
    for tok in re.finditer(r'case\s+(\w+)\s*:|default\s*:|%s\s*=\s*(\w+)\s*;'
                           % assign, body):
        if tok.group(1):
            labels.append(tok.group(1))
        elif tok.group(0).startswith('default'):
            labels.append('default')
        else:
            groups.setdefault(tok.group(2), []).extend(labels)
            labels = []
    return groups


def default_aggtypes(vic):
    """{AGG_TYPE_*: [OUT_*]} from get_default_outvar_aggtype."""
    text = _strip_comments(open(os.path.join(vic, HISTORY)).read())
    return _switch_groups(text, 'get_default_outvar_aggtype', 'agg_type')


def nc_dims(vic):
    """{OUT_*: extra dimension name} from set_nc_var_dimids: the one
    dimension between time and (y, x), or absent for (time, y, x)."""
    text = _strip_comments(open(os.path.join(vic, NC_INFO)).read())
    m = re.search(r'\nset_nc_var_dimids\s*\(.*?\n\}', text, flags=re.S)
    body = m.group(0)
    out, labels = {}, []
    for tok in re.finditer(r'case\s+(\w+)\s*:|nc_dimids\[1\]\s*=\s*'
                           r'nc_hist_file->(\w+)_dimid|break\s*;|default\s*:',
                           body):
        if tok.group(1):
            labels.append(tok.group(1))
        elif tok.group(2):
            dim = {'front': 'front', 'layer': 'nlayer', 'node': 'node',
                   'band': 'snow_band', 'nj': None}[tok.group(2)]
            for lab in labels:
                if dim:
                    out[lab] = dim
        elif tok.group(0).startswith('break') or \
                tok.group(0).startswith('default'):
            labels = []
    return out


def _cstring(literals):
    return ''.join(re.findall(r'"((?:[^"\\]|\\.)*)"', literals))


def metadata(vic):
    """{OUT_*: {long_name, standard_name, units, description, nelem}}."""
    text = _strip_comments(open(os.path.join(vic, METADATA)).read())
    meta = {}
    for m in re.finditer(r'strcpy\s*\(\s*out_metadata\[(OUT_\w+)\]\.(\w+)\s*,'
                         r'\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)', text):
        name, field, lit = m.group(1), m.group(2), m.group(3)
        if field == 'varname':
            continue
        meta.setdefault(name, {})[field] = _cstring(lit)
    # The band variables borrow their units (and a few their names) from the
    # cell variable: strcpy(out_metadata[OUT_X_BAND].units,
    # out_metadata[OUT_X].units). Resolved after the literals are in.
    for m in re.finditer(r'strcpy\s*\(\s*out_metadata\[(OUT_\w+)\]\.(\w+)\s*,'
                         r'\s*out_metadata\[(OUT_\w+)\]\.(\w+)\s*\)', text):
        src = meta.get(m.group(3), {}).get(m.group(4))
        if src is not None:
            meta.setdefault(m.group(1), {})[m.group(2)] = src
    for m in re.finditer(r'out_metadata\[(OUT_\w+)\]\.nelem\s*=\s*([^;]+);',
                         text):
        meta.setdefault(m.group(1), {})['nelem'] = m.group(2).strip()
    return meta


def vic_version(vic):
    p = os.path.join(vic, VERSION_H)
    if not os.path.exists(p):
        return None
    m = re.search(r'#define\s+SHORT_VERSION\s+"([^"]+)"', open(p).read())
    return m.group(1) if m else None


def _source_name(vic):
    """`reference/VIC-5.1.0/vic`, the same from any checkout or work
    tree, rather than a path relative to wherever this ran."""
    p = os.path.abspath(vic)
    i = p.rfind(os.sep + 'reference' + os.sep)
    return p[i + 1:] if i >= 0 else p


def extract(vic):
    names = outvar_enum(vic)
    groups = default_aggtypes(vic)
    known = {'AGG_TYPE_END', 'AGG_TYPE_SUM', 'AGG_TYPE_AVG'}
    if set(groups) - known:
        raise SystemExit('get_default_outvar_aggtype returns %s, which this '
                         'script does not know' % sorted(set(groups) - known))
    default = {}
    for agg, labels in groups.items():
        for lab in labels:
            default[lab] = agg[len('AGG_TYPE_'):]
    fallback = default.pop('default', None)
    if fallback is None:
        raise SystemExit('get_default_outvar_aggtype has no default branch')
    meta = metadata(vic)
    dims = nc_dims(vic)
    unknown = sorted((set(default) | set(meta) | set(dims)) - set(names))
    if unknown:
        raise SystemExit('names outside the OUT_* enum: %s' % unknown)
    variables = {}
    for n in names:
        m = meta.get(n, {})
        variables[n] = dict(
            default_agg=default.get(n, fallback),
            dim=dims.get(n),
            nelem=m.get('nelem', '1'),
            long_name=m.get('long_name'), standard_name=m.get('standard_name'),
            units=m.get('units'), description=m.get('description'))
    return dict(
        generated_by='tools/extract_vic_outvars.py',
        vic_source=_source_name(vic),
        # the 5.1.0 release's own vic_version.h still reads 5.0.1
        vic_version_h=vic_version(vic),
        default_agg_fallback=fallback,
        variables=variables)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vic', default=None,
                    help='the VIC source tree (the directory holding drivers/)')
    ap.add_argument('--out', default=JSON_OUT)
    ap.add_argument('--check', action='store_true',
                    help='compare with --out and write nothing')
    args = ap.parse_args(argv)
    vic = args.vic or default_vic_tree()
    if vic is None or not os.path.isdir(os.path.join(vic, 'drivers')):
        raise SystemExit('no VIC source tree; pass --vic <tree>/vic')
    inv = extract(vic)
    n = len(inv['variables'])
    counts = {}
    for v in inv['variables'].values():
        counts[v['default_agg']] = counts.get(v['default_agg'], 0) + 1
    if args.check:
        old = json.load(open(args.out)) if os.path.exists(args.out) else {}
        same = old.get('variables') == inv['variables']
        print('%d OUT_* variables in %s (%s); %s %s'
              % (n, vic, ', '.join('%s %d' % kv for kv in sorted(counts.items())),
                 args.out, 'matches' if same else 'DIFFERS'))
        return 0 if same else 1
    with open(args.out, 'w') as fh:
        json.dump(inv, fh, indent=1, sort_keys=True)
        fh.write('\n')
    print('wrote %s: %d OUT_* variables (%s)'
          % (args.out, n, ', '.join('%s %d' % kv
                                    for kv in sorted(counts.items()))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
