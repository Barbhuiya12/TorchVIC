"""`torchvic run --gpus N`: the run split over N processes, one per device.

Each process loads the global file with its own piece of the active cells
(torchvic.parallel.shard), runs the single-device path on it, aggregates the
history streams and saves them (and the state file's piece); the parent
concatenates the pieces along the cell axis and writes the files a single
run writes. Cells are independent, so the values are the single run's.
"""
import os
import pickle
import shutil
import tempfile
import time
import traceback

import torch

from . import shard

#: what a sharded run does not do (each is a one-line refusal, not a wrong
#: answer): the routing sum across cells, the per-cell 4.2.d files and the
#: forcing-only run are not per-cell products of one shard
REFUSED = ('study files', 'OUTPUT_FORCE', 'the VIC 4.2.d history convention',
           'classic-driver global files')


def devices_for(n, device):
    """The device of each of `n` processes: cuda:0..n-1 (relative to
    CUDA_VISIBLE_DEVICES), or the CPU for --device cpu."""
    if device == 'cpu':
        return ['cpu'] * n
    if device not in (None, 'auto', 'cuda'):
        raise SystemExit('--gpus N picks cuda:0..cuda:N-1 itself; do not '
                         'combine it with --device %s' % device)
    have = torch.cuda.device_count()
    if have < n:
        raise SystemExit('--gpus %d: only %d CUDA device(s) visible '
                         '(--device cpu runs N CPU processes)' % (n, have))
    return ['cuda:%d' % i for i in range(n)]


def _worker(spec, i, tmp):
    """One shard: the cli's run path on a piece of the cells."""
    err = os.path.join(tmp, 'err_%d.txt' % i)
    try:
        res = _run_shard(spec, i, tmp)
        with open(os.path.join(tmp, 'res_%d.pkl' % i), 'wb') as fh:
            pickle.dump(res, fh)
    except BaseException:                                # noqa: BLE001
        with open(err, 'w') as fh:
            fh.write(traceback.format_exc())
        raise


def _run_shard(spec, i, tmp):
    from .. import case as _case, config
    from ..io import history, state_file
    from ..io.global_param import GlobalParam
    from ..core import tfallback as _tfb
    dev = spec['devices'][i]
    if dev == 'cpu':
        torch.set_num_threads(spec['threads'])
    else:
        torch.cuda.set_device(torch.device(dev))
    n = spec['n']
    log = (lambda *a: None) if i or spec['quiet'] else print
    sd = spec['sample_data']
    gp = GlobalParam(spec['input'],
                     sample_data=_case.sample_data_dir() if sd is None
                     else sd)
    with torch.inference_mode():
        t_load = time.time()
        with shard.sharded(i, n) as info:
            case = _case.load(
                spec['input'], device=torch.device(dev),
                dtype=config.DTYPES[spec['dtype'] or 'float64'],
                core=spec['core'], steps=spec['steps'],
                first_step=spec['first_step'], daily=spec['daily'],
                no_energy=spec['no_energy'], sample_data=sd, log=log)
        t_load = time.time() - t_load
        streams = history.parse_streams(case.gp, log=lambda *a: None)
        plan = state_file.save_plan(case)
        if plan is not None:
            state_file.refuse_unsupported(case)
        at_end = plan is not None and plan[0] == case.n_steps
        t0 = time.time()
        _kw = dict(return_state=True) if at_end else {}
        if spec.get('energy_error'):
            _kw['energy_error'] = spec['energy_error']
        out = case.run(**_kw)
        if plan is not None:
            n_, when, path = plan
            spath = os.path.join(tmp, 'state_%d.nc' % i)
            state_file.save_state(case, (n_, when, spath),
                                  out=out if at_end else None, log=log)
        if dev != 'cpu':
            torch.cuda.synchronize(torch.device(dev))
        t_run = time.time() - t0
        recs = [history.stream_records(out, s, case.dates, case.model_steps,
                                       extremes=spec['extremes'],
                                       log=lambda *a: None)
                for s in streams]
        data = {r.stream.prefix: {k: v.detach().cpu()
                                  for k, v in r.data.items()} for r in recs}
        torch.save(data, os.path.join(tmp, 'data_%d.pt' % i))
        res = dict(i=i, n_cells=case.n_cells, n_steps=case.n_steps,
                   t_load=t_load, t_run=t_run, index=info['index'],
                   rows=info['rows'], cols=info['cols'],
                   tfb=dict(_tfb.counts() or {}),
                   plan=None if plan is None else (plan[0], plan[1], plan[2]),
                   state=plan is not None and plan[0] is not None)
        if i == 0:
            for r in recs:
                r.data = {}
            res['records'] = recs
            res['dom_names'] = dict(case.dom.names)
        return res


def run_sharded(args, log=print):
    """The `torchvic run` of `args` over args.gpus processes. Returns the
    list of files written."""
    import torch.multiprocessing as mp
    from .. import case as _case
    from ..io import history, state_file
    from ..io.domain import ImageDomain
    from ..io.global_param import GlobalParam
    path = args.input
    if path.lower().endswith(('.yaml', '.yml', '.json')):
        raise SystemExit('--gpus: study files are not sharded')
    sd = args.sample_data
    gp = GlobalParam(path, sample_data=_case.sample_data_dir() if sd is None
                     else sd)
    if gp.get('SOIL') is not None and gp.get('DOMAIN') is None:
        raise SystemExit('--gpus: classic-driver global files are not '
                         'sharded')
    if gp.get('OUTPUT_FORCE') is not None and \
            str(gp.get('OUTPUT_FORCE')).upper() in ('TRUE', '1'):
        raise SystemExit('--gpus: OUTPUT_FORCE runs write the forcing only')
    ogp = history._as_gp(args.output_global) if args.output_global else None
    conv = args.agg_convention or history.convention_of(
        ogp if ogp is not None else gp)
    if conv == 'vic4.2d' or ogp is not None:
        raise SystemExit('--gpus: the VIC 4.2.d history convention is not '
                         'sharded')
    n = int(args.gpus)
    streams = [] if args.no_history else history.parse_streams(gp, log=log)
    result_dir = None
    if streams:
        try:
            result_dir = history.resolve_result_dir(gp, args.out, log=log)
        except ValueError as e:
            log('history NOT written: %s' % e)
            streams = []
    devs = devices_for(n, args.device)
    spec = dict(input=path, devices=devs, n=n, dtype=args.dtype,
                core=args.core, steps=args.steps, first_step=args.first_step,
                daily=args.daily, no_energy=args.no_energy, sample_data=sd,
                quiet=args.quiet,
                energy_error=getattr(args, 'energy_error', None),
                extremes='true' if args.true_extremes else 'vic',
                threads=max(1, (os.cpu_count() or n) // n))
    base = os.path.dirname(os.path.abspath(result_dir)) if result_dir \
        else None
    tmp = tempfile.mkdtemp(prefix='torchvic_shards_', dir=base)
    try:
        ctx = mp.get_context('spawn')
        procs = [ctx.Process(target=_worker, args=(spec, i, tmp))
                 for i in range(n)]
        t0 = time.time()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        bad = [i for i, p in enumerate(procs) if p.exitcode]
        if bad:
            msg = []
            for i in bad:
                f = os.path.join(tmp, 'err_%d.txt' % i)
                msg.append('shard %d: %s' % (i, open(f).read() if
                                             os.path.exists(f)
                                             else 'exit %s' % procs[i]
                                             .exitcode))
            raise RuntimeError('sharded run failed\n' + '\n'.join(msg))
        wall = time.time() - t0
        res = []
        for i in range(n):
            with open(os.path.join(tmp, 'res_%d.pkl' % i), 'rb') as fh:
                res.append(pickle.load(fh))
        res.sort(key=lambda r: r['i'])
        paths = []
        data = [torch.load(os.path.join(tmp, 'data_%d.pt' % i))
                for i in range(n)]
        if streams:
            dom = ImageDomain(_case._fix_path(gp.domain_file, log),
                              names=gp.domain_vars)
            for rec in res[0]['records']:
                rec.data = shard.merge_data([d[rec.stream.prefix]
                                             for d in data],
                                            [r['index'] for r in res])
                paths += history.write_record(rec, dom, result_dir, gp=gp,
                                              log=log)
        plan = res[0]['plan']
        if plan is not None and res[0]['state']:
            dom = ImageDomain(_case._fix_path(gp.domain_file, log),
                              names=gp.domain_vars)
            shard.merge_state_files(
                [os.path.join(tmp, 'state_%d.nc' % i) for i in range(n)],
                [(r['rows'], r['cols']) for r in res], plan[2],
                (dom.y_dim, dom.x_dim))
            paths.append(plan[2])
            log('wrote state %s' % plan[2])
        elif plan is not None:
            log('STATENAME: %s is not the end of any step of this run; VIC '
                'writes no state file then (check_save_state_flag.c)'
                % plan[1])
        tfb = {}
        for r in res:
            for k, v in r['tfb'].items():
                tfb[k] = tfb.get(k, 0) + v
        print('ran %d steps over %d cells on %d %s processes in %.1f s '
              '(slowest shard %.1f s, shards %s cells; wall incl. loading '
              '%.1f s)'
              % (res[0]['n_steps'], sum(r['n_cells'] for r in res), n,
                 'CPU' if devs[0] == 'cpu' else 'GPU',
                 max(r['t_run'] for r in res), max(r['t_run'] for r in res),
                 '/'.join(str(r['n_cells']) for r in res), wall))
        if tfb and any(tfb.values()):
            print('TFALLBACK TRUE: temperature solves that found no root '
                  'and kept the previous temperature: '
                  + ', '.join('%s %d' % kv for kv in sorted(tfb.items())))
        return paths
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
