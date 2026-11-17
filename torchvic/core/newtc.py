"""The implicit profile's Newton solve as one compiled loop per column.

soiltemp._newt_raph_fast evaluates newt_raph (newt_raph_func_fast.c
:18-169) with fda_heat_eqn's statics (frozen_soil.c:624-1010) as a few
hundred torch operations per Newton trial, up to NEWT_RAPH_MAXTRIAL
trials per profile solve and one solve per surface-temperature trial:
on the CPU that is operation overhead, not arithmetic. Here the same
trial -- the same expressions in the same order, element by element --
is a C loop over the columns, each column iterating until it stops,
as VIC's own scalar newt_raph does.

Bit for bit the torch solver's answer. Addition, subtraction,
multiplication, division, fabs and the comparisons are IEEE operations
whatever evaluates them (compiled with -ffp-contract=off, no fast-math,
so no fused multiply-add is formed). The one transcendental in the
loop, pow, is called as torch rounds it in the same place:

- exact-math mode (xmath.EXACT) or a CPU without AVX kernels: glibc
  pow, as xmath.pow and ATen's scalar kernels call it;
- AVX512 / AVX2 kernels: SLEEF's Sleef_powd8_u10 / Sleef_powd4_u10
  (the symbols ATen's Vectorized<double>::pow calls, taken from the
  loaded libtorch_cpu), except for the columns ATen's vectorized loop
  leaves to its scalar tail -- the last C mod (2 x lanes) of a row of
  C columns, as soiltemp._rowcall lays the rows out -- which get glibc.
  The tail is a property of a column's position in the full batch, so
  it is passed in with each column (`cidx`, `ctail`).

The terms that do not depend on the trial temperatures
(soil_conductivity_fixed, with its log10) are formed by the caller in
torch, on the full batch, exactly as the torch solver forms them.
"""
import ctypes
import hashlib
import os
import subprocess
import tempfile

import torch

_SRC = r"""
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <immintrin.h>
#include <pthread.h>

typedef __m512d (*pow8_t)(__m512d, __m512d);
typedef __m256d (*pow4_t)(__m256d, __m256d);

__attribute__((target("avx512f")))
static void vpow8(void *fp, const double *a, const double *b, double *o,
                  int k)
{
    pow8_t f = (pow8_t)fp;
    double ta[8], tb[8], to[8];
    for (int i = 0; i < k; i += 8) {
        int m = k - i < 8 ? k - i : 8;
        for (int j = 0; j < 8; j++) {
            ta[j] = j < m ? a[i + j] : 1.0;
            tb[j] = j < m ? b[i + j] : 1.0;
        }
        __m512d r = f(_mm512_loadu_pd(ta), _mm512_loadu_pd(tb));
        _mm512_storeu_pd(to, r);
        for (int j = 0; j < m; j++) o[i + j] = to[j];
    }
}

__attribute__((target("avx2")))
static void vpow4(void *fp, const double *a, const double *b, double *o,
                  int k)
{
    pow4_t f = (pow4_t)fp;
    double ta[4], tb[4], to[4];
    for (int i = 0; i < k; i += 4) {
        int m = k - i < 4 ? k - i : 4;
        for (int j = 0; j < 4; j++) {
            ta[j] = j < m ? a[i + j] : 1.0;
            tb[j] = j < m ? b[i + j] : 1.0;
        }
        __m256d r = f(_mm256_loadu_pd(ta), _mm256_loadu_pd(tb));
        _mm256_storeu_pd(to, r);
        for (int j = 0; j < m; j++) o[i + j] = to[j];
    }
}

/* mode 0: glibc; 8: Sleef d8; 4: Sleef d4.  tail: this column is in
   ATen's scalar tail (glibc). */
static void vpow(int mode, void *fp, int tail, const double *a,
                 const double *b, double *o, int k)
{
    if (k <= 0) return;
    if (mode == 8 && !tail) { vpow8(fp, a, b, o, k); return; }
    if (mode == 4 && !tail) { vpow4(fp, a, b, o, k); return; }
    for (int i = 0; i < k; i++) o[i] = pow(a[i], b[i]);
}

#define LATICE 3.337e5
#define TKTRIP 273.16
#define GRAV 9.80616
#define CM_PER_M 100.0

/* Arrays, all row-major per column:
   [nn]: t0 ice kap cs moist maxm bub expt zsum
   [n]:  alpha beta gamma, and for nodes 1..n: kdry por sr ksu keu kspor
         soil solid
   [1]:  deltat bexp */
enum { A_T0, A_ICE, A_KAP, A_CS, A_MOIST, A_MAXM, A_BUB, A_EXPT, A_ZSUM,
       A_AL, A_BE, A_GA, A_KDRY, A_POR, A_SR, A_KSU, A_KEU, A_KSPOR,
       A_SOIL, A_SOLID, A_DT, A_BEXP, A_N };

/* newt_raph on M columns; column col reads row r = rows[col] (or col)
   of the arrays A, its surface temperature from tsv[r] (or t0[0]), and
   its pow tail from cidx[col]. x is [M][n], err [M]. */
static int newt_core(int M, const int64_t *rows, int n, int nn, int noflux,
                     int exp_trans, int mode, void *fp, const int64_t *cidx,
                     int64_t ctail, double **A, const double *tsv, double *x,
                     uint8_t *err, const double *cst)
{
    const int maxtrial = (int)cst[0];
    const double tolx = cst[1], tolf = cst[2], rmax = cst[3],
        rmin = cst[4], relax1 = cst[5], relax2 = cst[6], relax3 = cst[7],
        eps2 = cst[8], rholat = cst[9];
    const double tiny = DBL_MIN;
    if (n > 64 || n < 1) return -1;
    for (int col = 0; col < M; col++) {
        const int64_t rw = rows ? rows[col] : col;
        const double *t0 = A[A_T0] + rw * nn;
        const double *ice = A[A_ICE] + rw * nn;
        const double *kap = A[A_KAP] + rw * nn;
        const double *cs = A[A_CS] + rw * nn;
        const double *moist = A[A_MOIST] + rw * nn;
        const double *maxm = A[A_MAXM] + rw * nn;
        const double *bub = A[A_BUB] + rw * nn;
        const double *expt = A[A_EXPT] + rw * nn;
        const double *zsum = A[A_ZSUM] + rw * nn;
        const double *al = A[A_AL] + rw * n;
        const double *be = A[A_BE] + rw * n;
        const double *ga = A[A_GA] + rw * n;
        const double *kdry = A[A_KDRY] + rw * n;
        const double *por = A[A_POR] + rw * n;
        const double *sr = A[A_SR] + rw * n;
        const double *ksu = A[A_KSU] + rw * n;
        const double *keu = A[A_KEU] + rw * n;
        const double *kspor = A[A_KSPOR] + rw * n;
        const double *soil = A[A_SOIL] + rw * n;
        const double *solid = A[A_SOLID] + rw * n;
        const double deltat = A[A_DT][rw];
        const double bexp = exp_trans ? A[A_BEXP][rw] : 0.0;
        const int tail = (ctail >= 0 && cidx[col] >= ctail);
        double *xc_ = x + (int64_t)col * n;
        const double ts = tsv ? tsv[rw] : t0[0];
        const double tb = noflux ? t0[n] : t0[n + 1];
        double xs[64], xp[64], h[64], expo[64];
        double tt[128], arg[128], pw[128], ic[128], kn[128], cn[128];
        double wu[128], pa[256], pb[256], po[256];
        uint8_t ch[128], uf[128];
        double kv[3][66], cv[3][66];
        double fvec[64], fa[64], fb[64], fc[64], r[64], cc[64];
        memset(fa, 0, sizeof fa);
        memset(fc, 0, sizeof fc);
        for (int i = 0; i < n; i++) {
            xs[i] = xc_[i];
            expo[i] = -(2.0 / (expt[i + 1] - 3.0));
        }
        int done = 0;
        for (int k = 0; k < maxtrial; k++) {
            for (int i = 0; i < n; i++) {
                double hh = eps2 * fabs(xs[i]);
                if (hh == 0.0) hh = eps2;
                xp[i] = xs[i] + hh;
                h[i] = xp[i] - xs[i];
                tt[i] = xs[i];
                tt[n + i] = xp[i];
            }
            /* maximum_unfrozen_water at the 2n trial temperatures */
            int np = 0;
            for (int q = 0; q < 2 * n; q++) {
                int i = q < n ? q : q - n, m = i + 1;
                double a = ((-LATICE * tt[q]) / TKTRIP)
                    / ((GRAV * bub[m]) / CM_PER_M);
                if (a < tiny) a = tiny;
                arg[q] = a;
                if (tt[q] < 0.0) { pa[np] = a; pb[np] = expo[i]; np++; }
            }
            vpow(mode, fp, tail, pa, pb, po, np);
            np = 0;
            for (int q = 0; q < 2 * n; q++) {
                int i = q < n ? q : q - n, m = i + 1;
                double muw;
                if (tt[q] < 0.0) {
                    double u = maxm[m] * po[np++];
                    if (u > maxm[m]) u = maxm[m];
                    if (u < 0.0) u = 0.0;
                    muw = u;
                } else {
                    muw = maxm[m];
                }
                double c_ = moist[m] - muw;
                if (c_ < 0.0) c_ = 0.0;
                if (!(tt[q] < 0.0)) c_ = 0.0;
                ic[q] = c_;
                ch[q] = c_ != ice[m];
                wu[q] = moist[m] - c_;
                uf[q] = wu[q] == moist[m];
            }
            /* soil_conductivity_wu: ki ** (porosity - wu), kw ** wu */
            np = 0;
            for (int q = 0; q < 2 * n; q++) {
                int i = q < n ? q : q - n;
                if (!uf[q]) {
                    pa[np] = 2.2; pb[np] = por[i] - wu[q]; np++;
                    pa[np] = 0.57; pb[np] = wu[q]; np++;
                }
            }
            vpow(mode, fp, tail, pa, pb, po, np);
            np = 0;
            for (int q = 0; q < 2 * n; q++) {
                int i = q < n ? q : q - n, m = i + 1;
                double ksat, ke;
                if (uf[q]) {
                    ksat = ksu[i];
                    ke = keu[i];
                } else {
                    ksat = (kspor[i] * po[np]) * po[np + 1];
                    np += 2;
                    ke = sr[i];
                }
                double kk = (ksat - kdry[i]) * ke + kdry[i];
                if (isnan(kk) || isnan(kdry[i])) kk = NAN;
                else if (kk < kdry[i]) kk = kdry[i];
                kn[q] = moist[m] > 0.0 ? kk : kdry[i];
                double c2 = solid[i] + 4.2e6 * wu[q];
                c2 = c2 + 1.9e6 * ic[q];
                c2 = c2 + 1.3e3 * (1.0 - ((soil[i] + wu[q]) + ic[q]));
                cn[q] = c2;
            }
            /* versions: 0 Q, 1 P, 2 S0; node index 0..n+1 */
            for (int v = 0; v < 3; v++) {
                kv[v][0] = kap[0];
                kv[v][n + 1] = 0.0;
                cv[v][0] = 0.0;
            }
            for (int i = 0; i < n; i++) {
                double s0 = ch[i] ? kn[i] : kap[i + 1];
                double p_ = ch[n + i] ? kn[n + i] : s0;
                double q_ = ch[i] ? kn[i] : p_;
                double c0 = ch[i] ? cn[i] : cs[i + 1];
                double cp = ch[n + i] ? cn[n + i] : c0;
                double cq = ch[i] ? cn[i] : cp;
                kv[0][i + 1] = q_; kv[1][i + 1] = p_; kv[2][i + 1] = s0;
                cv[0][i + 1] = cq; cv[1][i + 1] = cp; cv[2][i + 1] = c0;
            }
#define VER(mm, j) ((j) < 0 ? 2 : ((mm) <= (j) ? 0 : ((mm) == (j) + 1 ? 1 : 2)))
#define XV(qq, j) ((qq) < 0 ? ts : ((qq) >= n ? tb : ((qq) == (j) ? xp[qq] : xs[qq])))
            /* the residual row r with Jacobian focus j (-1: none) */
#define RESID(out, r, j) do { \
                double xc = XV(r, j), xl = XV((r) - 1, j), \
                    xr = XV((r) + 1, j); \
                double dt_ = xr - xl, dt_up = xc - xl, dt_down = xr - xc; \
                int up = (noflux && (r) == n - 1) ? (r) + 1 : (r) + 2; \
                double klo = kv[VER(r, j)][r]; \
                double kmid = kv[VER((r) + 1, j)][(r) + 1]; \
                double dk = kv[VER(up, j)][up] - klo; \
                double csn = cv[VER((r) + 1, j)][(r) + 1]; \
                double st = (csn * (xc - t0[(r) + 1])) / deltat \
                    + (xc * (csn - cs[(r) + 1])) / deltat; \
                double f1, f2; \
                if (!exp_trans) { \
                    f1 = ((dk / al[r]) * dt_) / al[r]; \
                    f2 = (kmid * (dt_down / ga[r] - dt_up / be[r])) \
                        / (0.5 * al[r]); \
                } else { \
                    double zr = zsum[(r) + 1]; \
                    double zb = bexp * (zr + 1.0); \
                    double zz = (bexp * (zr + 1.0)) * (zr + 1.0); \
                    f1 = ((((dk / 2.0) * dt_) / 2.0) / zb) / zb; \
                    f2 = kmid * ((((dt_down - dt_up) / zb) / zb) \
                                 - ((dt_ / 2.0) / zz)); \
                } \
                double icv = ((r) == (j)) ? ic[n + (r)] : ic[r]; \
                double ph = (rholat * (icv - ice[(r) + 1])) / deltat; \
                out = ((f1 + f2) + ph) - st; \
            } while (0)
            double errf = 0.0;
            for (int rr = 0; rr < n; rr++) {
                RESID(fvec[rr], rr, -1);
                errf = errf + fabs(fvec[rr]);
            }
            if (errf <= tolf) { done = 1; break; }
            for (int j = 0; j < n; j++) {
                double f;
                RESID(f, j, j);
                fb[j] = (f - fvec[j]) / h[j];
                if (j != 0) {
                    RESID(f, j - 1, j);
                    fc[j - 1] = (f - fvec[j - 1]) / h[j];
                }
                if (j != n - 1) {
                    RESID(f, j + 1, j);
                    fa[j + 1] = (f - fvec[j + 1]) / h[j];
                }
            }
            /* tridiag, :132-169 */
            for (int i = 0; i < n; i++) { r[i] = -fvec[i]; cc[i] = fc[i]; }
            {
                double factor = fb[0];
                cc[0] = cc[0] / factor;
                r[0] = r[0] / factor;
                for (int i = 1; i < n; i++) {
                    factor = fa[i];
                    double bi = fb[i] - cc[i - 1] * factor;
                    r[i] = r[i] - r[i - 1] * factor;
                    cc[i] = cc[i] / bi;
                    r[i] = r[i] / bi;
                }
                for (int j = n - 2; j >= 0; j--)
                    r[j] = r[j] - r[j + 1] * cc[j];
            }
            double errx = 0.0;
            for (int i = 0; i < n; i++) errx = errx + fabs(r[i]);
            for (int i = 0; i < n; i++) {
                int inr = (xs[i] < rmax) && (xs[i] > rmin);
                double step = 1.0;
                if (k > 10 && k <= 20) step = inr ? relax1 : 1.0;
                else if (k > 20 && k <= 60) step = inr ? relax2 : 1.0;
                else if (k > 60) step = inr ? relax3 : 1.0;
                xs[i] = xs[i] + r[i] * step;
            }
            if (errx <= tolx) { done = 1; break; }
        }
        for (int i = 0; i < n; i++) xc_[i] = xs[i];
        err[col] = !done;
    }
    return 0;
}

int tv_newt(int M, int n, int nn, int noflux, int exp_trans, int mode,
            void *fp, const int64_t *cidx, int64_t ctail,
            double **A, double *x, uint8_t *err, const double *cst)
{
    return newt_core(M, NULL, n, nn, noflux, exp_trans, mode, fp, cidx,
                     ctail, A, NULL, x, err, cst);
}

/* solve_T_profile_implicit (frozen_soil.c:235-322) after the setup:
   every one of the C rows gets tnew = its start profile (t0 with node 0
   at tsv) and err = 1; the M rows `rows` are solved, and where Newton
   stops tnew takes the solution, the boundaries stay, and with
   TFALLBACK the cold nose is smoothed in place, ascending (:301-318).
   xbuf is [M][n] scratch. */
/* The columns are independent: threads take them one at a time. */
typedef struct {
    int M; const int64_t *rows; int n, nn, noflux, exp_trans, mode;
    void *fp; int64_t ctail; double **A; const double *tsv; double *xbuf;
    uint8_t *ebuf; const double *cst; int next; int rc;
    pthread_mutex_t mu;
} job_t;

static void *worker(void *p)
{
    job_t *J = (job_t *)p;
    for (;;) {
        pthread_mutex_lock(&J->mu);
        int col = J->next++;
        pthread_mutex_unlock(&J->mu);
        if (col >= J->M) break;
        int rc = newt_core(1, J->rows + col, J->n, J->nn, J->noflux,
                           J->exp_trans, J->mode, J->fp, J->rows + col,
                           J->ctail, J->A, J->tsv,
                           J->xbuf + (int64_t)col * J->n, J->ebuf + col,
                           J->cst);
        if (rc) J->rc = rc;
    }
    return NULL;
}

static int run_threads(int M, const int64_t *rows, int n, int nn,
                       int noflux, int exp_trans, int mode, void *fp,
                       int64_t ctail, double **A, const double *tsv,
                       double *xbuf, uint8_t *ebuf, const double *cst,
                       int nthr)
{
    if (nthr > M) nthr = M;
    if (nthr <= 1)
        return newt_core(M, rows, n, nn, noflux, exp_trans, mode, fp, rows,
                         ctail, A, tsv, xbuf, ebuf, cst);
    if (nthr > 64) nthr = 64;
    job_t J = {M, rows, n, nn, noflux, exp_trans, mode, fp, ctail, A, tsv,
               xbuf, ebuf, cst, 0, 0};
    pthread_mutex_init(&J.mu, NULL);
    pthread_t th[64];
    int started = 0;
    for (int t = 1; t < nthr; t++)
        if (pthread_create(&th[started], NULL, worker, &J) == 0) started++;
    worker(&J);
    for (int t = 0; t < started; t++) pthread_join(th[t], NULL);
    pthread_mutex_destroy(&J.mu);
    return J.rc;
}

int tv_profile(int C, int M, const int64_t *rows, int n, int nn,
               int noflux, int exp_trans, int mode, void *fp, int64_t ctail,
               double **A, const double *tsv, double *tnew, uint8_t *err,
               int tfallback, double *xbuf, uint8_t *ebuf, const double *cst,
               int nthr)
{
    for (int64_t r = 0; r < C; r++) {
        const double *t0 = A[A_T0] + r * nn;
        double *o = tnew + r * nn;
        o[0] = tsv[r];
        for (int j = 1; j < nn; j++) o[j] = t0[j];
        err[r] = 1;
    }
    for (int col = 0; col < M; col++) {
        const double *t0 = A[A_T0] + rows[col] * nn;
        for (int i = 0; i < n; i++) xbuf[(int64_t)col * n + i] = t0[i + 1];
    }
    int rc = run_threads(M, rows, n, nn, noflux, exp_trans, mode, fp,
                         ctail, A, tsv, xbuf, ebuf, cst, nthr);
    if (rc) return rc;
    for (int col = 0; col < M; col++) {
        const int64_t r = rows[col];
        if (ebuf[col]) continue;
        err[r] = 0;
        double *o = tnew + r * nn;
        const double *t0 = A[A_T0] + r * nn;
        for (int i = 0; i < n; i++) o[i + 1] = xbuf[(int64_t)col * n + i];
        if (!tfallback) continue;
        for (int j = 1; j < nn - 1; j++) {
            double s0 = j - 1 == 0 ? tsv[r] : t0[j - 1];
            double dl0 = s0 - t0[j], du0 = t0[j + 1] - t0[j];
            double dl = o[j - 1] - o[j], du = o[j + 1] - o[j];
            int nose = ((dl0 > 0) && (du0 > 0) && (dl - dl0 > 0)
                        && (du - du0 > 0))
                || ((dl0 < 0) && (du0 < 0) && (dl - dl0 < 0)
                    && (du - du0 < 0));
            if (nose) o[j] = 0.5 * (o[j - 1] + o[j + 1]);
        }
    }
    return 0;
}
"""

_lib = None
_ORDER = ('t0', 'ice', 'kap', 'cs', 'moist', 'maxm', 'bub', 'expt', 'zsum',
          'al', 'be', 'ga', 'kdry', 'por', 'sr', 'ksu', 'keu', 'kspor',
          'soil', 'solid', 'dt', 'bexp')


def _load():
    """The compiled loop, or False where there is no C compiler."""
    global _lib
    if _lib is not None:
        return _lib
    _lib = False
    try:
        tag = hashlib.md5(_SRC.encode()).hexdigest()[:10]
        d = os.path.join(tempfile.gettempdir(),
                         'torchvic_newtc_%d' % os.getuid())
        os.makedirs(d, exist_ok=True)
        so = os.path.join(d, 'newtc_%s.so' % tag)
        if not os.path.exists(so):
            c = so[:-3] + '.%d.c' % os.getpid()
            with open(c, 'w') as fh:
                fh.write(_SRC)
            subprocess.run(['gcc', '-O2', '-fPIC', '-shared', '-fno-builtin',
                            '-fno-fast-math', '-ffp-contract=off', '-pthread',
                            c, '-o', so + '.%d' % os.getpid(), '-lm'],
                           check=True,
                           capture_output=True)
            os.replace(so + '.%d' % os.getpid(), so)
        lib = ctypes.CDLL(so)
        lib.tv_newt.restype = ctypes.c_int
        _lib = lib
    except Exception:                                       # noqa: BLE001
        _lib = False
    return _lib


_POW = {}


def pow_mode():
    """(mode, function pointer, 2 x lanes) of the pow torch's CPU kernels
    use for float64: (0, None, 0) glibc, (8, Sleef_powd8_u10, 16) under
    AVX512, (4, Sleef_powd4_u10, 8) under AVX2; None if unknown."""
    from . import xmath
    if xmath.EXACT:
        return 0, None, 0
    cap = torch.backends.cpu.get_cpu_capability()
    if cap in _POW:
        return _POW[cap]
    res = None
    if cap in ('DEFAULT', 'NO AVX'):
        res = (0, None, 0)
    elif cap in ('AVX512', 'AVX2'):
        name = 'Sleef_powd8_u10' if cap == 'AVX512' else 'Sleef_powd4_u10'
        try:
            tl = ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__),
                                          'lib', 'libtorch_cpu.so'))
            fp = ctypes.cast(getattr(tl, name), ctypes.c_void_p).value
            res = ((8, fp, 16) if cap == 'AVX512' else (4, fp, 8))
        except Exception:                                   # noqa: BLE001
            res = None
    _POW[cap] = res
    return res


#: Whether the compiled loop may be used (tests switch it off).
ENABLED = True


def available(x):
    """Whether `x`'s solve can run here: CPU float64, compiler, a known
    pow."""
    return (ENABLED and x.device.type == 'cpu' and x.dtype == torch.float64
            and pow_mode() is not None and bool(_load()))


def newt_raph(arrs, x, n, nn, noflux, exp_trans, cidx, cfull, consts):
    """Run newt_raph on the columns of `arrs` (dict of [M, k] float64
    contiguous CPU tensors, keys _ORDER; 'dt' and 'bexp' [M]); `x` [M, n]
    is the start, solved in place. `cidx` [M] int64 is each column's
    index in the full batch of `cfull` columns (the pow tail). Returns
    (x, error [M] bool)."""
    lib = _load()
    mode, fp, lanes2 = pow_mode()
    M = x.shape[0]
    ctail = -1
    if mode and cfull < 32768:
        ctail = cfull - cfull % lanes2
    x = x.contiguous().clone()
    err = torch.zeros(M, dtype=torch.uint8)
    keep = [arrs[k].contiguous() for k in _ORDER]
    ptrs = (ctypes.c_void_p * len(keep))(*[t.data_ptr() for t in keep])
    cidx = cidx.to(torch.int64).contiguous()
    cst = torch.tensor(consts, dtype=torch.float64)
    rc = lib.tv_newt(ctypes.c_int(M), ctypes.c_int(n), ctypes.c_int(nn),
                     ctypes.c_int(int(noflux)), ctypes.c_int(int(exp_trans)),
                     ctypes.c_int(mode), ctypes.c_void_p(fp or 0),
                     ctypes.c_void_p(cidx.data_ptr()),
                     ctypes.c_int64(ctail), ptrs,
                     ctypes.c_void_p(x.data_ptr()),
                     ctypes.c_void_p(err.data_ptr()),
                     ctypes.c_void_p(cst.data_ptr()))
    if rc != 0:
        raise RuntimeError('newtc: %d nodes exceed the compiled limit' % n)
    return x, err.bool()


def position_free(device):
    """Whether every torch operation on `device` rounds an element the
    same wherever it sits in a batch: off the CPU, or on CPU kernels
    without vector code (exact-math mode's ATEN_CPU_CAPABILITY=default),
    where pow/exp/log are glibc's for every element."""
    if torch.device(device).type != 'cpu':
        return True
    return torch.backends.cpu.get_cpu_capability() in ('DEFAULT', 'NO AVX')


#: Threads for the column loop: None follows torch.get_num_threads().
THREADS = None
#: Columns per thread below which the loop stays on one thread.
MIN_COLS_PER_THREAD = 4


def threads(m):
    t = THREADS if THREADS is not None else torch.get_num_threads()
    return int(max(1, min(t, m // MIN_COLS_PER_THREAD)))


def profile(keep, tsv, idx, c, n, nn, noflux, exp_trans, tfallback, consts):
    """tv_profile: the full batch's [C, k] arrays `keep` (newtc._ORDER),
    trial surface temperatures `tsv` [C], the rows to solve `idx` [M].
    Returns (t_new [C, nn], error [C] bool)."""
    lib = _load()
    mode, fp, lanes2 = pow_mode()
    ctail = -1
    if mode and c < 32768:
        ctail = c - c % lanes2
    idx = idx.to(torch.int64).contiguous()
    m = idx.numel()
    tnew = torch.empty((c, nn), dtype=torch.float64)
    err = torch.empty(c, dtype=torch.uint8)
    xbuf = torch.empty((max(m, 1), n), dtype=torch.float64)
    ebuf = torch.empty(max(m, 1), dtype=torch.uint8)
    ptrs = (ctypes.c_void_p * len(keep))(*[t.data_ptr() for t in keep])
    cst = torch.tensor(consts, dtype=torch.float64)
    rc = lib.tv_profile(ctypes.c_int(c), ctypes.c_int(m),
                        ctypes.c_void_p(idx.data_ptr()), ctypes.c_int(n),
                        ctypes.c_int(nn), ctypes.c_int(int(noflux)),
                        ctypes.c_int(int(exp_trans)), ctypes.c_int(mode),
                        ctypes.c_void_p(fp or 0), ctypes.c_int64(ctail),
                        ptrs, ctypes.c_void_p(tsv.data_ptr()),
                        ctypes.c_void_p(tnew.data_ptr()),
                        ctypes.c_void_p(err.data_ptr()),
                        ctypes.c_int(int(bool(tfallback))),
                        ctypes.c_void_p(xbuf.data_ptr()),
                        ctypes.c_void_p(ebuf.data_ptr()),
                        ctypes.c_void_p(cst.data_ptr()),
                        ctypes.c_int(threads(m)))
    if rc != 0:
        raise RuntimeError('newtc: %d nodes exceed the compiled limit' % n)
    return tnew, err.bool()


# ---------------------------------------------------------------------------
# The same solve on a CUDA device: one kernel launch per profile solve.
#
# One warp per column runs tv_profile's loop above -- newt_raph's trials,
# the boundaries and the cold nose -- with the arithmetic torch's CUDA
# kernels apply in soiltemp._newt_raph_fast / solve_profile_implicit, so
# the answer is the torch GPU solver's bit for bit:
#
# - +, -, *, /, fabs and the comparisons are IEEE, one rounding each
#   (every product by __dmul_rn, which is never merged into a fused
#   multiply-add; ATen's elementwise kernels hold one operation each,
#   where contraction cannot occur either);
# - pow is CUDA's pow (libdevice __nv_pow), which ATen's pow kernel calls
#   for float64 (Pow.cuh pow_ -> ::pow), from the NVRTC of the CUDA
#   release torch was built with, compiled as nvcc compiled ATen's, with
#   fused multiply-adds allowed inside it (--fmad=true: with false, 7 of
#   4M random pows differ from torch.pow by an ulp);
# - a division by a CPU scalar is, in ATen's CUDA div_true kernel
#   (BinaryDivTrueKernel.cu), a multiplication by its reciprocal
#   1.0 / b: maximum_unfrozen_water's / CONST_TKTRIP and / CM_PER_M,
#   and / deltat where deltat is a Python number (DT_RECIP);
# - 2.0 / t is reciprocal(t) * 2.0 (Tensor.__rtruediv__), equal to the
#   division (an exact doubling).
#
# Unwanted columns (cols[c] == 0) are not iterated: their profile is the
# start profile, failed, as in tv_profile. No host synchronisation, the
# launch is on torch's current stream: the call can be captured in a
# CUDA graph.
# ---------------------------------------------------------------------------
_CU_SRC = r"""
#define NN_ (N_ + (NOFLUX ? 1 : 2))
/* every product rounded on its own: never merged into a fused
   multiply-add (the unit is compiled with --fmad=true, as ATen's
   kernels are, for libdevice's pow to round as theirs) */
#define M(a, b) __dmul_rn((a), (b))
#define WPB 4                   /* columns (warps) per block */
#define Q2 (2 * N_)
#define FULL 0xffffffffu
extern "C" __global__ void __launch_bounds__(32 * WPB) tv_profile_cu(
    const int C,
    const double *__restrict__ T0, const double *__restrict__ ICE,
    const double *__restrict__ KAP, const double *__restrict__ CS,
    const double *__restrict__ MOIST, const double *__restrict__ MAXM,
    const double *__restrict__ BUB, const double *__restrict__ EXPT,
    const double *__restrict__ ZSUM, const double *__restrict__ AL,
    const double *__restrict__ BE, const double *__restrict__ GA,
    const double *__restrict__ KDRY, const double *__restrict__ POR,
    const double *__restrict__ SR, const double *__restrict__ KSU,
    const double *__restrict__ KEU, const double *__restrict__ KSPOR,
    const double *__restrict__ SOIL, const double *__restrict__ SOLID,
    const double *__restrict__ DT, const double *__restrict__ BEXP,
    const double *__restrict__ tsv, const unsigned char *__restrict__ cols,
    double *__restrict__ tnew, unsigned char *__restrict__ err,
    const int maxtrial, const double tolx, const double tolf,
    const double rmax, const double rmin, const double relax1,
    const double relax2, const double relax3, const double eps2,
    const double rholat, const double inv_tk, const double inv_cm,
    const double tiny, const double latice_neg, const double grav)
{
    /* one warp per column: its lanes share the column's nodes, trial
       temperatures and residual rows; the sequential parts (the sums,
       the tridiagonal sweep, the cold nose) run on lane 0 */
    __shared__ double S_xs[WPB][N_], S_xp[WPB][N_], S_h[WPB][N_];
    __shared__ double S_ic[WPB][Q2], S_kn[WPB][Q2], S_cn[WPB][Q2];
    __shared__ unsigned char S_ch[WPB][Q2];
    __shared__ double S_kv[WPB][3][N_ + 2], S_cv[WPB][3][N_ + 2];
    __shared__ double S_f[WPB][4 * N_];
    __shared__ double S_fa[WPB][N_], S_fb[WPB][N_], S_fc[WPB][N_];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int col = blockIdx.x * WPB + w;
    if (col >= C) return;                       /* the whole warp */
    const int n = N_, nn = NN_;
    const double *t0 = T0 + (long long)col * nn;
    double *o = tnew + (long long)col * nn;
    const double ts = tsv[col];
    for (int j = lane; j < nn; j += 32) o[j] = j == 0 ? ts : t0[j];
    if (lane == 0) err[col] = 1;
    if (cols && !cols[col]) return;             /* the whole warp */
    const double *ice = ICE + (long long)col * nn;
    const double *kap = KAP + (long long)col * nn;
    const double *cs = CS + (long long)col * nn;
    const double *moist = MOIST + (long long)col * nn;
    const double *maxm = MAXM + (long long)col * nn;
    const double *bub = BUB + (long long)col * nn;
    const double *expt = EXPT + (long long)col * nn;
    const double *zsum = ZSUM + (long long)col * nn;
    const double *al = AL + (long long)col * n;
    const double *be = BE + (long long)col * n;
    const double *ga = GA + (long long)col * n;
    const double *kdry = KDRY + (long long)col * n;
    const double *por = POR + (long long)col * n;
    const double *sr = SR + (long long)col * n;
    const double *ksu = KSU + (long long)col * n;
    const double *keu = KEU + (long long)col * n;
    const double *kspor = KSPOR + (long long)col * n;
    const double *soil = SOIL + (long long)col * n;
    const double *solid = SOLID + (long long)col * n;
    const double deltat = DT[col];
#if DT_RECIP
    const double idt = 1.0 / deltat;
#define DIVDT(v) M((v), idt)
#else
#define DIVDT(v) ((v) / deltat)
#endif
#if EXP_TRANS
    const double bexp = BEXP[col];
#endif
    const double tb = NOFLUX ? t0[n] : t0[n + 1];
    double *xs = S_xs[w], *xp = S_xp[w], *h = S_h[w];
    double *ic = S_ic[w], *kn = S_kn[w], *cn = S_cn[w];
    unsigned char *ch = S_ch[w];
    double (*kv)[N_ + 2] = S_kv[w], (*cv)[N_ + 2] = S_cv[w];
    double *sf = S_f[w], *fa = S_fa[w], *fb = S_fb[w], *fc = S_fc[w];
    for (int i = lane; i < n; i += 32) {
        xs[i] = t0[i + 1];
        fa[i] = 0.0; fc[i] = 0.0;
    }
    if (lane < 3) {
        kv[lane][0] = kap[0];
        kv[lane][n + 1] = 0.0;
        cv[lane][0] = 0.0;
    }
    /* this lane's trial temperatures q = lane, lane + 32 (< 2n): the
       exponent of maximum_unfrozen_water */
    double eq[2];
    for (int s = 0; s < 2; s++) {
        int q = lane + 32 * s;
        int i = q < n ? q : q - n;
        eq[s] = q < Q2 ? -(2.0 / (expt[i + 1] - 3.0)) : 0.0;
    }
    __syncwarp(FULL);
    int done = 0;
    for (int k = 0; k < maxtrial; k++) {
        for (int i = lane; i < n; i += 32) {
            double hh = M(eps2, fabs(xs[i]));
            if (hh == 0.0) hh = eps2;
            xp[i] = xs[i] + hh;
            h[i] = xp[i] - xs[i];
        }
        __syncwarp(FULL);
        for (int s = 0; s < 2; s++) {
            const int q = lane + 32 * s;
            if (q >= Q2) break;
            const int i = q < n ? q : q - n, m = i + 1;
            const double tq = q < n ? xs[i] : xp[i];
            /* maximum_unfrozen_water */
            double muw;
            if (tq < 0.0) {
                double a = M(M(latice_neg, tq), inv_tk)
                    / M(M(grav, bub[m]), inv_cm);
                if (a < tiny) a = tiny;
                double u = M(maxm[m], pow(a, eq[s]));
                if (u > maxm[m]) u = maxm[m];
                if (u < 0.0) u = 0.0;
                muw = u;
            } else {
                muw = maxm[m];
            }
            double c_ = moist[m] - muw;
            if (c_ < 0.0) c_ = 0.0;
            if (!(tq < 0.0)) c_ = 0.0;
            ic[q] = c_;
            ch[q] = c_ != ice[m];
            const double wv = moist[m] - c_;
            /* soil_conductivity_wu */
            double ksat, ke;
            if (wv == moist[m]) {
                ksat = ksu[i];
                ke = keu[i];
            } else {
                ksat = M(M(kspor[i], pow(2.2, por[i] - wv)), pow(0.57, wv));
                ke = sr[i];
            }
            double kk = M(ksat - kdry[i], ke) + kdry[i];
            if (isnan(kk) || isnan(kdry[i]))
                kk = __longlong_as_double(0x7ff8000000000000LL);
            else if (kk < kdry[i]) kk = kdry[i];
            kn[q] = moist[m] > 0.0 ? kk : kdry[i];
            double c2 = solid[i] + M(4.2e6, wv);
            c2 = c2 + M(1.9e6, c_);
            c2 = c2 + M(1.3e3, 1.0 - ((soil[i] + wv) + c_));
            cn[q] = c2;
        }
        __syncwarp(FULL);
        for (int i = lane; i < n; i += 32) {
            double s0 = ch[i] ? kn[i] : kap[i + 1];
            double p_ = ch[n + i] ? kn[n + i] : s0;
            double q_ = ch[i] ? kn[i] : p_;
            double c0 = ch[i] ? cn[i] : cs[i + 1];
            double cp = ch[n + i] ? cn[n + i] : c0;
            double cq = ch[i] ? cn[i] : cp;
            kv[0][i + 1] = q_; kv[1][i + 1] = p_; kv[2][i + 1] = s0;
            cv[0][i + 1] = cq; cv[1][i + 1] = cp; cv[2][i + 1] = c0;
        }
        __syncwarp(FULL);
#define VER(mm, j) ((j) < 0 ? 2 : ((mm) <= (j) ? 0 : ((mm) == (j) + 1 ? 1 : 2)))
#define XV(qq, j) ((qq) < 0 ? ts : ((qq) >= n ? tb : ((qq) == (j) ? xp[qq] : xs[qq])))
#if !EXP_TRANS
#define FLUX(f1, f2, r) do { \
            f1 = M(dk / al[r], dt_) / al[r]; \
            f2 = M(kmid, dt_down / ga[r] - dt_up / be[r]) / M(0.5, al[r]); \
        } while (0)
#else
#define FLUX(f1, f2, r) do { \
            double zr = zsum[(r) + 1]; \
            double zb = M(bexp, zr + 1.0); \
            double zz = M(M(bexp, zr + 1.0), zr + 1.0); \
            f1 = (M(M(M(dk, 0.5), dt_), 0.5) / zb) / zb; \
            f2 = M(kmid, (((dt_down - dt_up) / zb) / zb) - (M(dt_, 0.5) / zz)); \
        } while (0)
#endif
#define RESID(out, r, j) do { \
            double xc = XV(r, j), xl = XV((r) - 1, j), xr = XV((r) + 1, j); \
            double dt_ = xr - xl, dt_up = xc - xl, dt_down = xr - xc; \
            int up = (NOFLUX && (r) == n - 1) ? (r) + 1 : (r) + 2; \
            double klo = kv[VER(r, j)][r]; \
            double kmid = kv[VER((r) + 1, j)][(r) + 1]; \
            double dk = kv[VER(up, j)][up] - klo; \
            double csn = cv[VER((r) + 1, j)][(r) + 1]; \
            double st = DIVDT(M(csn, xc - t0[(r) + 1])) \
                + DIVDT(M(xc, csn - cs[(r) + 1])); \
            double f1, f2; \
            FLUX(f1, f2, r); \
            double icv = ((r) == (j)) ? ic[n + (r)] : ic[r]; \
            double ph = DIVDT(M(rholat, icv - ice[(r) + 1])); \
            out = ((f1 + f2) + ph) - st; \
        } while (0)
        /* the rows: [0, n) the full residual; n + j row j, 2n + j row
           j - 1 and 3n + j row j + 1 with Jacobian column j */
        for (int t = lane; t < 4 * n; t += 32) {
            int r, j;
            if (t < n) { r = t; j = -1; }
            else if (t < 2 * n) { j = t - n; r = j; }
            else if (t < 3 * n) { j = t - 2 * n; r = j - 1; }
            else { j = t - 3 * n; r = j + 1; }
            if (r < 0 || r >= n) continue;
            double f;
            RESID(f, r, j);
            sf[t] = f;
        }
        __syncwarp(FULL);
        int stop = 0;
        if (lane == 0) {
            double errf = 0.0;
            for (int rr = 0; rr < n; rr++) errf = errf + fabs(sf[rr]);
            stop = errf <= tolf;
        }
        stop = __shfl_sync(FULL, stop, 0);
        if (stop) { done = 1; break; }
        for (int j = lane; j < n; j += 32) {
            fb[j] = (sf[n + j] - sf[j]) / h[j];
            if (j != 0) fc[j - 1] = (sf[2 * n + j] - sf[j - 1]) / h[j];
            if (j != n - 1) fa[j + 1] = (sf[3 * n + j] - sf[j + 1]) / h[j];
        }
        __syncwarp(FULL);
        if (lane == 0) {
            double r[N_], cc[N_];
            for (int i = 0; i < n; i++) { r[i] = -sf[i]; cc[i] = fc[i]; }
            double factor = fb[0];
            cc[0] = cc[0] / factor;
            r[0] = r[0] / factor;
            for (int i = 1; i < n; i++) {
                factor = fa[i];
                double bi = fb[i] - M(cc[i - 1], factor);
                r[i] = r[i] - M(r[i - 1], factor);
                cc[i] = cc[i] / bi;
                r[i] = r[i] / bi;
            }
            for (int j = n - 2; j >= 0; j--)
                r[j] = r[j] - M(r[j + 1], cc[j]);
            double errx = 0.0;
            for (int i = 0; i < n; i++) errx = errx + fabs(r[i]);
            for (int i = 0; i < n; i++) {
                const bool inr = (xs[i] < rmax) && (xs[i] > rmin);
                double step = 1.0;
                if (k > 10 && k <= 20) step = inr ? relax1 : 1.0;
                else if (k > 20 && k <= 60) step = inr ? relax2 : 1.0;
                else if (k > 60) step = inr ? relax3 : 1.0;
                xs[i] = xs[i] + M(r[i], step);
            }
            stop = errx <= tolx;
        }
        stop = __shfl_sync(FULL, stop, 0);      /* also orders xs */
        __syncwarp(FULL);
        if (stop) { done = 1; break; }
    }
    if (!done) return;
    for (int i = lane; i < n; i += 32) o[i + 1] = xs[i];
    __syncwarp(FULL);
    if (lane != 0) return;
    err[col] = 0;
#if TFALLBACK
    for (int j = 1; j < nn - 1; j++) {
        double s0 = j - 1 == 0 ? ts : t0[j - 1];
        double dl0 = s0 - t0[j], du0 = t0[j + 1] - t0[j];
        double dl = o[j - 1] - o[j], du = o[j + 1] - o[j];
        bool nose = ((dl0 > 0) && (du0 > 0) && (dl - dl0 > 0)
                     && (du - du0 > 0))
            || ((dl0 < 0) && (du0 < 0) && (dl - dl0 < 0)
                && (du - du0 < 0));
        if (nose) o[j] = M(0.5, o[j - 1] + o[j + 1]);
    }
#endif
}
"""

#: Whether the CUDA kernel may be used (tests switch it off).
CUDA_ENABLED = True
#: columns (warps) per block of the CUDA kernel (WPB in _CU_SRC)
_CU_WPB = 4

_cu = {}            # (device index, defines) -> CUfunction handle
_cu_libs = None


def _cu_load_libs():
    """(libnvrtc, libcuda) as ctypes libraries, or False."""
    global _cu_libs
    if _cu_libs is not None:
        return _cu_libs
    _cu_libs = False
    try:
        nv = None
        for name in ('libnvrtc.so.12', 'libnvrtc.so'):
            try:
                nv = ctypes.CDLL(name)
                break
            except OSError:
                pass
        if nv is None:
            import glob
            import site
            for d in site.getsitepackages() + [os.path.dirname(
                    os.path.dirname(torch.__file__))]:
                hit = glob.glob(os.path.join(
                    d, 'nvidia', 'cuda_nvrtc', 'lib', 'libnvrtc.so*'))
                hit = [h for h in hit if 'builtins' not in h]
                if hit:
                    nv = ctypes.CDLL(sorted(hit)[0])
                    break
        cu = ctypes.CDLL('libcuda.so.1')
        if nv is None:
            return _cu_libs
        _cu_libs = (nv, cu)
    except OSError:
        _cu_libs = False
    return _cu_libs


def _cu_check(rc, what):
    if rc != 0:
        raise RuntimeError('newtc cuda: %s failed (%d)' % (what, rc))


def _cu_function(device, n, noflux, exp_trans, tfallback, dt_recip):
    """The kernel for one (device, node count, option) set, compiled
    with NVRTC on first use."""
    key = (device.index, n, bool(noflux), bool(exp_trans), bool(tfallback),
           bool(dt_recip))
    fn = _cu.get(key)
    if fn is not None:
        return fn
    nv, cu = _cu_load_libs()
    major, minor = torch.cuda.get_device_capability(device)
    src = ('#define N_ %d\n#define NOFLUX %d\n#define EXP_TRANS %d\n'
           '#define TFALLBACK %d\n#define DT_RECIP %d\n' % (
               n, int(bool(noflux)), int(bool(exp_trans)),
               int(bool(tfallback)), int(bool(dt_recip)))) + _CU_SRC
    prog = ctypes.c_void_p()
    _cu_check(nv.nvrtcCreateProgram(ctypes.byref(prog), src.encode(),
                                    b'newtc.cu', 0, None, None),
              'nvrtcCreateProgram')
    opts = [b'--gpu-architecture=sm_%d%d' % (major, minor), b'--fmad=true',
            b'--std=c++14']
    copts = (ctypes.c_char_p * len(opts))(*opts)
    rc = nv.nvrtcCompileProgram(prog, len(opts), copts)
    if rc != 0:
        sz = ctypes.c_size_t()
        nv.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
        log = ctypes.create_string_buffer(sz.value)
        nv.nvrtcGetProgramLog(prog, log)
        raise RuntimeError('newtc cuda: compile failed:\n%s'
                           % log.value.decode(errors='replace'))
    sz = ctypes.c_size_t()
    _cu_check(nv.nvrtcGetCUBINSize(prog, ctypes.byref(sz)),
              'nvrtcGetCUBINSize')
    cubin = ctypes.create_string_buffer(sz.value)
    _cu_check(nv.nvrtcGetCUBIN(prog, cubin), 'nvrtcGetCUBIN')
    nv.nvrtcDestroyProgram(ctypes.byref(prog))
    # the module in torch's (primary) context of this device
    with torch.cuda.device(device):
        torch.cuda.current_stream(device)       # context made current
        ctx = ctypes.c_void_p()
        _cu_check(cu.cuCtxGetCurrent(ctypes.byref(ctx)), 'cuCtxGetCurrent')
        if not ctx.value:
            _cu_check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx),
                                                  device.index),
                      'cuDevicePrimaryCtxRetain')
            _cu_check(cu.cuCtxSetCurrent(ctx), 'cuCtxSetCurrent')
        mod = ctypes.c_void_p()
        _cu_check(cu.cuModuleLoadData(ctypes.byref(mod), cubin),
                  'cuModuleLoadData')
        fn = ctypes.c_void_p()
        _cu_check(cu.cuModuleGetFunction(ctypes.byref(fn), mod,
                                         b'tv_profile_cu'),
                  'cuModuleGetFunction')
    _cu[key] = (fn, mod, cubin)
    return _cu[key]


def cuda_available(x, n=1):
    """Whether `x`'s solve of `n` unknowns can run as the CUDA kernel: a
    CUDA float64 tensor, n <= 32 (a warp's lanes), NVRTC and the driver
    library found."""
    return (CUDA_ENABLED and x.device.type == 'cuda' and 1 <= n <= 32
            and x.dtype == torch.float64 and bool(_cu_load_libs()))


def profile_cuda(keep, tsv, cols, c, n, nn, noflux, exp_trans, tfallback,
                 dt_recip, consts):
    """tv_profile on a CUDA device: `keep` the [C, k] arrays
    (newtc._ORDER, contiguous float64 on the device), `tsv` [C] the trial
    surface temperatures, `cols` [C] bool (or None) the columns to solve.
    `dt_recip`: deltat divides as a CPU scalar does in ATen's CUDA
    division (by its reciprocal). Returns (t_new [C, nn], error [C]
    bool), without a host synchronisation."""
    dev = tsv.device
    if dev.index is None:
        dev = torch.device('cuda', torch.cuda.current_device())
    if n > 32 or n < 1:
        raise RuntimeError('newtc: %d nodes exceed the kernel limit' % n)
    fn = _cu_function(dev, n, noflux, exp_trans, tfallback, dt_recip)[0]
    _, cu = _cu_load_libs()
    tnew = torch.empty((c, nn), dtype=torch.float64, device=dev)
    err = torch.empty(c, dtype=torch.uint8, device=dev)
    if c == 0:
        return tnew, err.bool()
    tsv = tsv.contiguous()
    cm = None
    if cols is not None:
        cm = cols.reshape(c).contiguous()
        if cm.dtype == torch.bool:
            cm = cm.view(torch.uint8)
        else:
            cm = (cm != 0).view(torch.uint8)
    (maxtrial, tolx, tolf, rmax, rmin, relax1, relax2, relax3, eps2,
     rholat) = consts
    from . import frozen
    vp = ctypes.c_void_p
    d = ctypes.c_double
    args = [ctypes.c_int(c)] + [vp(t.data_ptr()) for t in keep] + [
        vp(tsv.data_ptr()), vp(cm.data_ptr() if cm is not None else 0),
        vp(tnew.data_ptr()), vp(err.data_ptr()), ctypes.c_int(int(maxtrial)),
        d(tolx), d(tolf), d(rmax), d(rmin), d(relax1), d(relax2), d(relax3),
        d(eps2), d(rholat), d(1.0 / frozen.CONST_TKTRIP),
        d(1.0 / frozen.CM_PER_M), d(torch.finfo(torch.float64).tiny),
        d(-frozen.CONST_LATICE), d(frozen.CONST_G)]
    params = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
    blk = _CU_WPB
    stream = torch.cuda.current_stream(dev).cuda_stream
    with torch.cuda.device(dev):
        _cu_check(cu.cuLaunchKernel(fn, (c + blk - 1) // blk, 1, 1,
                                    32 * blk, 1, 1, 0, vp(stream), params,
                                    None),
                  'cuLaunchKernel')
    return tnew, err.bool()


# The kernel launch (ctypes, raw device pointers) is never traced into a
# fused region of the CUDA graph path (core.fusion): the region stops at
# the call and resumes after it.
from . import fusion as _fusion                                # noqa: E402
profile_cuda = _fusion.opaque(profile_cuda)
