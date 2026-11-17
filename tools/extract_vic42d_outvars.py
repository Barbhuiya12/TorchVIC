#!/usr/bin/env python3
"""Read VIC 4.2.d's own output-variable tables and write them down.

`torchvic.io.history` aggregates TorchVIC's per-step outputs the way VIC 4.2.d's
classic driver does when the global file is a VIC 4.2.d one (or when
`--agg-convention vic4.2d` asks for it). What each variable is, and how
4.2.d aggregates it, is read here from the C VIC 4.2.d compiles, not
restated by hand:

* **that it exists**, its index and its comment -- the `#define OUT_* n`
  block of `vicNl_def.h` (N_OUTVAR_TYPES 180, of which 167 are named). The
  comment carries the units and, for the fluxes, the ALMA_OUTPUT units.
* **its default aggregation** -- `create_output_list` (output_list_utils.c
  :297-385): every variable AGG_TYPE_AVG, then the END and SUM lists.
  4.2.d has no other way to set it: `parse_output_info.c` reads OUTVAR as
  `name [format [type [mult]]]` and never an aggregation type.
* **its element count** -- the `nelem` lines of the same function.
* **what ALMA_OUTPUT does to it** -- the block `if (options.ALMA_OUTPUT)`
  in put_data.c (:699-760), statement by statement, in order.
* **how each aggregation is formed** -- put_data.c's temporal aggregation
  (:669-688): the branches it has (END, SUM, AVG; BEG/MAX/MIN are defined
  in vicNl_def.h and have no branch) and the OUT_AERO_RESIST* overrides.

It also sets the table beside VIC 5's (torchvic/data/vic5_outvars.json): which
names exist in only one of the two, and which defaults differ.

    tools/extract_vic42d_outvars.py          # torchvic/data/vic42d_outvars.json
    tools/extract_vic42d_outvars.py --check  # compare, write nothing
    tools/extract_vic42d_outvars.py --src <VIC 4.2.d>/src
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_OUT = os.path.join(REPO, 'torchvic', 'data', 'vic42d_outvars.json')
VIC5_JSON = os.path.join(REPO, 'torchvic', 'data', 'vic5_outvars.json')


def default_src_tree():
    """The pristine tree under $TORCHVIC_REFERENCE_DIR (default
    <repository>/reference), or None."""
    root = os.environ.get('TORCHVIC_REFERENCE_DIR',
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reference'))
    p = os.path.join(root, 'VIC-VIC.4.2.d', 'src')
    return os.path.abspath(p) if os.path.isdir(p) else None


def _strip_comments(text):
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    return re.sub(r'//[^\n]*', ' ', text)


def _function(text, name):
    m = re.search(r'\n[\w \*]*\b%s\s*\([^)]*\)\s*\{' % name, text)
    if not m:
        raise SystemExit('function %s not found' % name)
    i, depth = m.end(), 1
    while depth:
        c = text[i]
        depth += c == '{'
        depth -= c == '}'
        i += 1
    return text[m.start():i]


def defines(src):
    """{OUT_*: (index, comment)} from vicNl_def.h, and N_OUTVAR_TYPES."""
    text = open(os.path.join(src, 'vicNl_def.h')).read()
    n = int(re.search(r'#define\s+N_OUTVAR_TYPES\s+(\d+)', text).group(1))
    out = {}
    for m in re.finditer(r'#define\s+(OUT_\w+)\s+(\d+)\s*(?:/\*(.*?)\*/)?',
                         text):
        name = m.group(1)
        if name.startswith('OUT_TYPE_'):
            continue
        out[name] = (int(m.group(2)), ' '.join((m.group(3) or '').split()))
    return out, n


def units_of(comment):
    """'[mm]' and '(ALMA_OUTPUT: [mm/s])' out of a vicNl_def.h comment."""
    alma = re.search(r'\(ALMA_OUTPUT:\s*\[([^\]]*)\]\)', comment)
    rest = re.sub(r'\(ALMA_OUTPUT:[^)]*\)', '', comment)
    u = re.findall(r'\[([^\]]*)\]', rest)
    return (u[-1] if u else None), (alma.group(1) if alma else None)


def output_list(src):
    """create_output_list: names in the order VIC lists them, nelem and the
    default aggregation of each."""
    text = _strip_comments(open(os.path.join(src,
                                             'output_list_utils.c')).read())
    body = _function(text, 'create_output_list')
    names = re.findall(r'strcpy\s*\(\s*out_data\[(OUT_\w+)\]\.varname\s*,'
                       r'\s*"(OUT_\w+)"\s*\)', body)
    for a, b in names:
        if a != b:
            raise SystemExit('varname %s stored under %s' % (b, a))
    order = [a for a, _ in names]
    nelem = {}
    # `if (options.FROZEN_SOIL) { out_data[OUT_FDEPTH].nelem = MAX_FRONTS;
    # ...}` -- kept with its condition
    frozen = re.search(r'if\s*\(\s*options\.FROZEN_SOIL\s*\)\s*\{(.*?)\}',
                       body, flags=re.S)
    cond = frozen.group(1) if frozen else ''
    for m in re.finditer(r'out_data\[(OUT_\w+)\]\.nelem\s*=\s*([^;]+);',
                         body):
        expr = m.group(2).strip()
        nelem[m.group(1)] = (expr + ' if options.FROZEN_SOIL else 1'
                             if m.group(0) in cond else expr)
    # the loop setting AGG_TYPE_AVG for every variable, then the overrides
    loop = re.search(r'for\s*\(v=0;\s*v<N_OUTVAR_TYPES;\s*v\+\+\)\s*\{\s*'
                     r'out_data\[v\]\.aggtype\s*=\s*AGG_TYPE_(\w+);', body)
    if not loop:
        raise SystemExit('no default aggtype loop in create_output_list')
    fallback = loop.group(1)
    agg = {}
    for m in re.finditer(r'out_data\[(OUT_\w+)\]\.aggtype\s*=\s*'
                         r'AGG_TYPE_(\w+)\s*;', body[loop.end():]):
        agg[m.group(1)] = m.group(2)
    return order, nelem, agg, fallback


def aggregation_branches(src):
    """put_data.c's temporal aggregation: the AGG_TYPEs it has a branch
    for, how each accumulates, and the overrides after it."""
    text = _strip_comments(open(os.path.join(src, 'put_data.c')).read())
    body = _function(text, 'put_data')
    m = re.search(r'for\s*\(v=0;\s*v<N_OUTVAR_TYPES;\s*v\+\+\)\s*\{\s*if\s*'
                  r'\(out_data\[v\]\.aggtype\s*==\s*AGG_TYPE_END\).*?'
                  r'\n  \}', body, flags=re.S)
    if not m:
        raise SystemExit('no temporal aggregation loop in put_data')
    loop = m.group(0)
    branches = {}
    for b in re.finditer(r'AGG_TYPE_(\w+)\)\s*\{\s*for\s*\([^)]*\)\s*\{\s*'
                         r'out_data\[v\]\.aggdata\[i\]\s*([^;]+);', loop):
        branches[b.group(1)] = 'aggdata ' + ' '.join(b.group(2).split())
    after = body[m.end():]
    overrides = [' '.join(x.split()) for x in re.findall(
        r'out_data\[OUT_AERO_RESIST\d?\]\.aggdata\[0\]\s*=\s*[^;]+;',
        after)]
    return branches, overrides


def alma_ops(src):
    """The ALMA_OUTPUT conversions, in put_data.c's order:
    [(OUT_*, element, op, operand)], element 'all' inside a loop over
    layers/nodes. Applied after aggregation, before writing."""
    text = _strip_comments(open(os.path.join(src, 'put_data.c')).read())
    body = _function(text, 'put_data')
    m = re.search(r'if\s*\(options\.ALMA_OUTPUT\)\s*\{', body)
    i, depth = m.end(), 1
    while depth:
        c = body[i]
        depth += c == '{'
        depth -= c == '}'
        i += 1
    block = body[m.end():i - 1]
    ops = []
    for s in re.finditer(
            r'out_data\[(OUT_\w+)\]\.aggdata\[(\w+)\]\s*([+\-*/]=)\s*'
            r'([^;]+);', block):
        name, idx, op, rhs = s.groups()
        rhs = ' '.join(rhs.split())
        r = re.match(r'out_data\[(OUT_\w+)\]\.aggdata\[0\]$', rhs)
        ops.append([name, '0' if idx == '0' else 'all', op,
                    r.group(1) if r else rhs])
    return ops


def extract(src, vic5=None):
    defs, n_types = defines(src)
    order, nelem, agg, fallback = output_list(src)
    unknown = sorted(set(order) - set(defs))
    if unknown:
        raise SystemExit('create_output_list names without a #define: %s'
                         % unknown)
    unlisted = sorted(set(defs) - set(order))
    if unlisted:
        raise SystemExit('#defined but not in create_output_list: %s'
                         % unlisted)
    alma = alma_ops(src)
    alma_by = {}
    for name, idx, op, rhs in alma:
        alma_by.setdefault(name, []).append(
            '%s%s %s %s' % ('aggdata[%s]' % idx if idx == '0'
                            else 'every element', '', op, rhs))
    variables = {}
    for name in order:
        idx, comment = defs[name]
        u, ua = units_of(comment)
        variables[name] = dict(index=idx, default_agg=agg.get(name, fallback),
                               nelem=nelem.get(name, '1'), comment=comment,
                               units=u, alma_units=ua,
                               alma=alma_by.get(name))
    branches, overrides = aggregation_branches(src)
    inv = dict(
        generated_by='tools/extract_vic42d_outvars.py',
        vic_source=_source_name(src),
        n_outvar_types=n_types,
        default_agg_fallback=fallback,
        aggregation_branches=branches,
        aggregation_overrides=overrides,
        alma_output=alma,
        variables=variables)
    if vic5:
        v5 = vic5['variables']
        inv['vs_vic5'] = dict(
            vic5_source=vic5.get('vic_source'),
            only_in_vic42d=sorted(set(variables) - set(v5)),
            only_in_vic5=sorted(set(v5) - set(variables)),
            default_agg_differs={
                n: {'vic42d': variables[n]['default_agg'],
                    'vic5': v5[n]['default_agg']}
                for n in sorted(set(variables) & set(v5))
                if variables[n]['default_agg'] != v5[n]['default_agg']})
    return inv


def _source_name(src):
    p = os.path.abspath(src)
    i = p.rfind(os.sep + 'reference' + os.sep)
    return p[i + 1:] if i >= 0 else p


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=None,
                    help='the VIC 4.2.d src/ directory')
    ap.add_argument('--out', default=JSON_OUT)
    ap.add_argument('--check', action='store_true',
                    help='compare with --out and write nothing')
    args = ap.parse_args(argv)
    src = args.src or default_src_tree()
    if src is None or not os.path.exists(os.path.join(src, 'put_data.c')):
        raise SystemExit('no VIC 4.2.d source; pass --src <VIC 4.2.d>/src')
    vic5 = json.load(open(VIC5_JSON)) if os.path.exists(VIC5_JSON) else None
    inv = extract(src, vic5)
    n = len(inv['variables'])
    counts = {}
    for v in inv['variables'].values():
        counts[v['default_agg']] = counts.get(v['default_agg'], 0) + 1
    summary = ', '.join('%s %d' % kv for kv in sorted(counts.items()))
    if args.check:
        old = json.load(open(args.out)) if os.path.exists(args.out) else {}
        same = all(old.get(k) == inv[k] for k in inv
                   if k not in ('generated_by', 'vic_source'))
        print('%d OUT_* variables in %s (%s); %s %s'
              % (n, src, summary, args.out, 'matches' if same else 'DIFFERS'))
        return 0 if same else 1
    with open(args.out, 'w') as fh:
        json.dump(inv, fh, indent=1, sort_keys=True)
        fh.write('\n')
    print('wrote %s: %d OUT_* variables (%s)' % (args.out, n, summary))
    if 'vs_vic5' in inv:
        d = inv['vs_vic5']
        print('  only in 4.2.d: %s' % ', '.join(d['only_in_vic42d']))
        print('  only in VIC 5: %s' % ', '.join(d['only_in_vic5']))
        print('  default differs: %s' % (', '.join(
            '%s (4.2.d %s, VIC 5 %s)' % (k, v['vic42d'], v['vic5'])
            for k, v in d['default_agg_differs'].items()) or 'none'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
