"""The `torchvic` command.

    torchvic run global_param.txt [--device auto|cpu|cuda[:N]]
                                  [--core 5.0.1|5.1.0|vicwur3] [--gpus N]
                                  [--dtype float64|float32] [--out DIR]
    torchvic run study.yaml ...
    torchvic run --config coupled.yaml [--device ...] [--out DIR]
    torchvic validate --config coupled.yaml [--deep]

`run` reads a VIC 5 image-driver global parameter file (or a study file that
names one, torchvic.config), runs the model forward and writes the global
file's OUTFILE history streams as VIC 5's image driver writes them, into
--out (default: the global file's RESULT_DIR). Command-line options override
a study file's own. The whole run executes under torch.inference_mode().

    torchvic convert-classic SOIL VEGPARAM VEGLIB [--snowband FILE] ...
                             --domain domain.nc --params params.nc

`run --config` runs a coupled-system configuration (torchvic.coupling:
modules, connections, the water ledger; docs/CONFIGURATION.md) and writes
VIC's history and the ledger to output.dir (or --out). `validate` checks such
a configuration without running it (--deep also reads the data).

`convert-classic` writes VIC 5 image-driver domain and parameter files from
VIC classic-driver soil, vegetation parameter, vegetation library and snow
band files (torchvic.io.classic_to_image).
"""
import argparse
import os
import sys
import time

from . import __version__

STUDY_SUFFIXES = ('.yaml', '.yml', '.json')


def _history_plan(case, args, log):
    """(convention, streams or Spec42d, result_dir), settled before the run
    so a RESULT_DIR that cannot be used costs a message, not the run."""
    from .io import history
    if args.no_history:
        return 'vic5', [], None
    ogp = (history._as_gp(args.output_global) if args.output_global
           else None)
    conv = args.agg_convention or history.convention_of(
        ogp if ogp is not None else case.gp)
    gp = ogp if ogp is not None else case.gp
    if conv == 'vic4.2d':
        what = history.output_spec_42d(case, output_global=ogp, log=log)
        log('history in VIC 4.2.d\'s convention: %r' % what)
    else:
        if ogp is not None:
            raise SystemExit('--output-global names a VIC 4.2.d global file '
                             'and is read under --agg-convention vic4.2d '
                             'only')
        what = history.parse_streams(gp, log=log)
    result_dir = None
    if what:
        try:
            result_dir = history.resolve_result_dir(gp, args.out, log=log)
        except ValueError as e:
            log('history NOT written: %s' % e)
            return conv, [], None
    return conv, what, result_dir


def cmd_run(args):
    import torch
    if getattr(args, 'gpus', None) and args.gpus > 1:
        from .parallel.run import run_sharded
        log = (lambda *a: None) if args.quiet else print
        for p in run_sharded(args, log=log):
            print('wrote %s' % p)
        return 0
    if getattr(args, 'exact_math', False):
        from .core import xmath
        xmath.enable()
    from . import case as _case, config
    from .io import history
    log = (lambda *a: None) if args.quiet else print
    path = args.input
    with torch.inference_mode():
        if path.lower().endswith(STUDY_SUFFIXES):
            study = config.load(path, log=log, device=args.device,
                                dtype=args.dtype, core=args.core)
            case = study.case
        else:
            study = None
            dtype = config.DTYPES[args.dtype or 'float64']
            case = _case.load(path, device=config.resolve_device(
                                  args.device or 'auto'),
                              dtype=dtype, core=args.core,
                              steps=args.steps, first_step=args.first_step,
                              daily=args.daily, no_energy=args.no_energy,
                              sample_data=args.sample_data, log=log)
        log(repr(case))
        if isinstance(case, _case.ForceCase):
            # OUTPUT_FORCE TRUE: VIC 4.2.d writes its forcing, not the model
            paths = case.write_forcing(result_dir=args.out, log=log)
            for p in paths:
                print('wrote %s' % p)
            return 0
        conv, streams, result_dir = _history_plan(case, args, log)
        from .io import state_file
        plan = (state_file.save_plan(case)
                if isinstance(case, _case.Case) else None)
        if plan is not None:
            state_file.refuse_unsupported(case)
        t0 = time.time()
        _at_end = plan is not None and plan[0] == case.n_steps
        _kw = dict(return_state=True) if _at_end else {}
        if getattr(args, 'energy_error', None):
            _kw['energy_error'] = args.energy_error
        out = case.run(progress=args.progress, **_kw)
        if plan is not None:
            state_file.save_state(case, plan, out=out if _at_end else None,
                                  log=log)
        if case.device is not None and str(case.device).startswith('cuda'):
            torch.cuda.synchronize()
        print('ran %d steps over %d cells in %.1f s'
              % (case.n_steps, case.n_cells, time.time() - t0))
        from .core import tfallback as _tfb
        _fb = _tfb.counts()
        if _fb and any(_fb.values()):
            print('TFALLBACK TRUE: temperature solves that found no root and '
                  'kept the previous temperature: '
                  + ', '.join('%s %d' % kv for kv in sorted(_fb.items())))
        if study is not None and study.observations:
            print('\nagainst the observations:')
            for var, row in study.report(out).items():
                for name, v in row.items():
                    print('  %-14s %-8s %.4f' % (var, name, v))
        if streams and conv == 'vic4.2d':
            log('\nhistory (VIC 4.2.d\'s convention and files):')
            paths = history.write_history_42d(
                out, case, result_dir=result_dir, spec=streams,
                output_global=args.output_global, carry=args.vic42d_carry,
                log=log)
        elif streams:
            log('\nhistory (the global file\'s OUTFILE streams):')
            paths = case.write_history(
                out, result_dir=result_dir, streams=streams,
                extremes='true' if args.true_extremes else 'vic', log=log,
                convention='vic5')
        else:
            paths = []
        for p in paths or []:
            print('wrote %s' % p)
    return 0


def cmd_run_config(args):
    import torch
    from .coupling import TorchVIC
    log = (lambda *a: None) if args.quiet else print
    m = TorchVIC.from_config(args.config, log=log, device=args.device,
                             dtype=args.dtype)
    if args.out:
        m.cfg['output']['dir'] = args.out
    log(m.runtime.describe())
    t0 = time.time()
    with torch.inference_mode():
        m.initialize()
        m.run()
        paths = m.finalize()
    rep = m.report()
    print('ran %d coupled steps to t = %d s in %.1f s'
          % (rep['tick'], rep['t'], time.time() - t0))
    for k, v in rep['backends'].items():
        print('  %s on %s: %s' % (k, v['device'], v['ran']))
    for k, v in rep['ledger'].items():
        print('  ledger %s: residual %.3e m3 over the run' % (k,
                                                            v['residual']))
    for p in paths:
        print('wrote %s' % p)
    return 0


def cmd_validate(args):
    from .coupling import validate_config
    problems = validate_config(args.config, deep=args.deep,
                               log=None if not args.deep else print)
    if problems:
        print('%s is not valid:' % args.config)
        for x in problems:
            print('  - %s' % x)
        return 1
    from .coupling import TorchVIC
    print(TorchVIC.from_config(args.config,
                               log=lambda *a: None).runtime.describe())
    print('%s is valid' % args.config)
    return 0


def cmd_convert_classic(args):
    from .io.classic_to_image import convert, BARE_SOIL_SAMPLE
    log = (lambda *a: None) if args.quiet else print
    opts = {}
    if args.global_param is None:
        opts = dict(nlayer=args.nlayer, root_zones=args.root_zones,
                    lai_src=args.lai_src, fcan_src=args.fcan_src,
                    alb_src=args.alb_src, wind_h=args.wind_h,
                    baseflow=args.baseflow)
        opts = {k: v for k, v in opts.items() if v is not None}
        for flag in ('veglib_fcan', 'veglib_photo', 'vegparam_lai',
                     'vegparam_fcan', 'vegparam_alb', 'july_tavg_supplied',
                     'spatial_snow', 'spatial_frost', 'blowing',
                     'organic_fract'):
            if getattr(args, flag):
                opts[flag] = True
    cv = convert(args.soil, args.vegparam, args.veglib, args.snowband,
                 out_domain=args.domain, out_params=args.params,
                 grid=args.grid, resolution=args.resolution, area=args.area,
                 frac=args.frac_from,
                 bare_soil=BARE_SOIL_SAMPLE if args.bare_soil == 'sample'
                 else None,
                 outside='drop' if args.drop_outside else 'error',
                 global_param=args.global_param,
                 sample_data=args.sample_data, snow_bands=args.snow_bands,
                 compress=args.compress, log=log, **opts)
    print('%d cells on a %d x %d grid, %d vegetation classes (bare soil '
          'last), %d layers, %d snow bands, %d root zones'
          % (cv.n_cells, cv.lat.size, cv.lon.size, cv.n_veg,
             cv.n_layers, cv.n_bands, cv.n_root))
    for p in (args.domain, args.params):
        if p:
            print('wrote %s' % p)
    return 0


def _add_convert_parser(sub):
    c = sub.add_parser(
        'convert-classic',
        help='VIC classic parameter files to VIC 5 image domain and '
             'parameter files',
        description='Write a VIC 5 image-driver domain file and parameter '
                    'file from VIC classic-driver files. The files are read '
                    'as vic_classic reads them (torchvic.io.classic); the '
                    'layout options come from --global (a classic global '
                    'parameter file) or from the flags below, with VIC\'s '
                    'defaults.')
    c.add_argument('soil', nargs='?', help='soil parameter file (default: '
                   'SOIL in --global)')
    c.add_argument('vegparam', nargs='?', help='vegetation parameter file')
    c.add_argument('veglib', nargs='?', help='vegetation library')
    c.add_argument('--snowband', help='snow band file')
    c.add_argument('--snow-bands', type=int, default=None,
                   help='number of snow bands (SNOW_BAND; default 1)')
    c.add_argument('--global', dest='global_param', default=None,
                   help='a VIC classic global parameter file to take the '
                        'files and options from')
    c.add_argument('--sample-data', default=None,
                   help='what ${VIC_SAMPLE_DATA} in --global resolves to')
    c.add_argument('--domain', help='domain file to write')
    c.add_argument('--params', help='parameter file to write')
    c.add_argument('--grid', help='netCDF file or zarr store whose 1-D lat '
                   'and lon are the target grid (default: the cells\' own '
                   'extent at --resolution)')
    c.add_argument('--resolution', type=float, default=None,
                   help='grid spacing in degrees (default: from --grid or '
                        'the cells)')
    c.add_argument('--drop-outside', action='store_true',
                   help='leave out (and list) run cells beyond --grid '
                        'instead of stopping')
    c.add_argument('--area', choices=['vic', 'sphere'], default='vic',
                   help='cell area: compute_cell_area.c as the classic '
                        'driver (default) or the exact spherical band area')
    c.add_argument('--frac-from', default=None,
                   help='a domain file on the same grid whose frac to use '
                        '(default: 1 on every cell)')
    c.add_argument('--bare-soil', choices=['classic', 'sample'],
                   default='classic',
                   help='bare-soil class roughness, displacement and wind_h: '
                        'as the classic driver derives them (default) or as '
                        'UW-Hydro\'s image sample stores them (0, 0, 2 m)')
    c.add_argument('--nlayer', type=int, default=None)
    c.add_argument('--root-zones', type=int, default=None,
                   help='ROOT_ZONES (default 3)')
    for flag in ('veglib-fcan', 'veglib-photo', 'vegparam-lai',
                 'vegparam-fcan', 'vegparam-alb', 'july-tavg-supplied',
                 'spatial-snow', 'spatial-frost', 'blowing',
                 'organic-fract'):
        c.add_argument('--' + flag, action='store_true',
                       help='%s TRUE' % flag.upper().replace('-', '_'))
    for src in ('lai', 'fcan', 'alb'):
        c.add_argument('--%s-src' % src, default=None,
                       choices=['FROM_VEGLIB', 'FROM_VEGPARAM',
                                'FROM_DEFAULT'] if src == 'fcan' else
                       ['FROM_VEGLIB', 'FROM_VEGPARAM'])
    c.add_argument('--wind-h', type=float, default=None,
                   help='WIND_H, the bare-soil class reference height '
                        '(default 10 m)')
    c.add_argument('--baseflow', default=None,
                   choices=['ARNO', 'NIJSSEN2001'])
    c.add_argument('--compress', type=int, default=4,
                   help='netCDF zlib level, 0 for none (default 4)')
    c.add_argument('--quiet', action='store_true')
    c.set_defaults(func=cmd_convert_classic)


def build_parser():
    ap = argparse.ArgumentParser(
        prog='torchvic', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', action='version',
                    version='torchvic %s' % __version__)
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='run a VIC 5 image-driver case forward')
    r.add_argument('input', nargs='?', help='a VIC 5 image-driver global '
                   'parameter file, or a study file (.yaml, .yml, .json)')
    r.add_argument('--config', default=None,
                   help='a coupled-system configuration (YAML) to run '
                        'instead (docs/CONFIGURATION.md)')
    r.add_argument('--device', default=None,
                   help="auto (default: the GPU with the most free memory, "
                        "else the CPU), cpu, cuda or cuda:N")
    r.add_argument('--gpus', type=int, default=None, metavar='N',
                   help='split the active cells over N processes, one per '
                        'GPU (cuda:0..N-1; with --device cpu, N CPU '
                        'processes) and merge into the files one process '
                        'writes, value for value; global file input, VIC 5 '
                        'history only')
    r.add_argument('--core', default=None, choices=['5.0.1', '5.1.0', 'vicwur3'],
                   help='the VIC release whose physics to run (default '
                        '5.1.0, or the study file\'s case.core; 5.0.1 is '
                        'the default of 0.4.7 and before)')
    r.add_argument('--dtype', default=None, choices=['float64', 'float32'],
                   help='floating-point precision (default float64)')
    r.add_argument('--out', '--result-dir', dest='out', default=None,
                   metavar='DIR',
                   help='where the history files go (default: the global '
                        'file\'s RESULT_DIR)')
    r.add_argument('--steps', type=int, default=0,
                   help='run only this many forcing records of the global '
                        'file\'s simulation window (0: all of it); global '
                        'file input only')
    r.add_argument('--first-step', type=int, default=0,
                   help='skip this many forcing records from the start of '
                        'the global file\'s simulation window; global file '
                        'input only')
    r.add_argument('--daily', action='store_true',
                   help='override the global file\'s MODEL_STEPS_PER_DAY '
                        'with a daily step (the global file\'s '
                        'MODEL_STEPS_PER_DAY 1 needs no flag); global file '
                        'input only')
    r.add_argument('--no-energy', action='store_true',
                   help='the water-balance path even where FULL_ENERGY is '
                        'TRUE; global file input only')
    r.add_argument('--sample-data', default=None,
                   help='what ${VIC_SAMPLE_DATA} in the global file resolves '
                        'to (default: $TORCHVIC_SAMPLE_DATA)')
    r.add_argument('--no-history', action='store_true',
                   help='do not write the OUTFILE history streams')
    r.add_argument('--true-extremes', action='store_true',
                   help='AGG_TYPE_MAX/MIN as the period\'s own extremes; by '
                        'default they start from 0 as VIC 5\'s do')
    r.add_argument('--agg-convention', default=None,
                   choices=['vic5', 'vic4.2d'],
                   help='aggregate and write the history as VIC 5 (netCDF '
                        'streams) or VIC 4.2.d (OUT_STEP, per-cell ASCII/'
                        'BINARY files) does; default: vic4.2d for a VIC '
                        '4.2.d global file, vic5 otherwise')
    r.add_argument('--output-global', default=None,
                   help='a VIC 4.2.d global file whose output section the '
                        'history follows (with --agg-convention vic4.2d)')
    r.add_argument('--vic42d-carry', action='store_true',
                   help='VIC 4.2.d convention: carry a cell\'s partial last '
                        'OUT_STEP period into the next cell\'s first record, '
                        'as vicNl does')
    r.add_argument('--exact-math', action='store_true',
                   help='verification mode (CPU): glibc pow/exp/sqrt as '
                        'VIC 5.1.0 gets them; sets ATEN_CPU_CAPABILITY='
                        'default before torch\'s first operation (same as '
                        'TORCHVIC_EXACT_MATH=1)')
    r.add_argument('--energy-error', default=None,
                   choices=['solver', 'vic'],
                   help='what OUT_ENERGY_ERROR holds on a full-energy run: '
                        '"solver" (default) is the surface balance\'s '
                        'residual at the solved temperature; "vic" forms it '
                        'as VIC 5 does (put_data.c:538-552) from the '
                        'cell\'s reported rows, which is the open balance '
                        'VIC\'s own output carries')
    r.add_argument('--progress', action='store_true',
                   help='a progress bar over the time loop (needs tqdm)')
    r.add_argument('--quiet', action='store_true',
                   help='print only the summary and the files written')
    r.set_defaults(func=cmd_run)
    v = sub.add_parser('validate', help='check a coupled-system '
                       'configuration without running it')
    v.add_argument('--config', required=True)
    v.add_argument('--deep', action='store_true',
                   help='also initialise the connectors (reads the data): '
                        'checks grids, mappings and the run\'s span')
    v.set_defaults(func=cmd_validate)
    _add_convert_parser(sub)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command != 'run':
        return args.func(args)
    if args.config:
        if args.input:
            print('torchvic: give a global file or --config, not both',
                  file=sys.stderr)
            return 2
        return cmd_run_config(args)
    if not args.input:
        print('torchvic run: a global file, a study file or --config is '
              'needed', file=sys.stderr)
        return 2
    if not os.path.exists(args.input):
        print('torchvic: %s does not exist' % args.input, file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
