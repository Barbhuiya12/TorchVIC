/* Oracle for VIC-WUR land use.
 *
 * Links the UNMODIFIED VIC-WUR 3.0 sources
 *   plugins/landuse/src/lu_apply.c    lu_apply and the distribute_* functions
 *   plugins/landuse/src/lu_support.c  heat capacities, derived states, totals
 *   vic_run/src/*.c                   what lu_support calls
 *                                     (distribute_node_moisture_properties,
 *                                     new_snow_density, compute_zwt, ...)
 *   drivers/shared_all/src/{initialize_options,initialize_parameters,
 *                           vic_log}.c  the default options and parameters
 * with the real vic_def.h structs (vicwur_lu_shim/ adds the image-driver
 * types), sets up a domain from the input, calls lu_apply() once (what
 * lu_forcing does on a forcing record, lu_force.c:94) and writes every state
 * it touches.
 *
 * Input (argv[1], doubles):
 *   nc nt nb nl nn nf ni rps uh nvegtypes pbare irrigation routing
 *   full_energy dt doy max_surface_swe
 *   per cell: area lat nv_active air_temp fs_active b_infilt
 *     band  : AreaFract Tfactor                                  (nb)
 *     layer : max_moist depth porosity resid_moist Wpwp bulk_dens_min
 *             soil_dens_min quartz bulk_density soil_density organic (nl)
 *     node  : dz_node Zsum_node max_moist_node expt_node bubble_node (nn)
 *     tile  : veg_class overstory Cv_old Cv_forced                (nt)
 *     tile x band: LAI Wdew swq pack_water surf_water snow_canopy
 *             coldcontent density depth last_snow MELTING surf_temp
 *             pack_temp AnnualNPP AnnualNPPPrev CLitter CInter CSlow
 *             moist[nl] ice[nl][nf] T[nn]
 *     ni_active, then irr: veg_index paddy leftover[nb]           (ni)
 *             (entries >= ni_active are padding, not seen by the C)
 *     ring  : dt_discharge[uh + rps - 1]
 * Output (argv[2], doubles) per cell: Cv_new[nt]; per tile x band: Wdew swq
 *   pack_water surf_water snow_canopy coldcontent density depth MELTING
 *   surf_temp pack_temp AnnualNPP AnnualNPPPrev CLitter CInter CSlow
 *   moist[nl] ice[nl][nf] T[nn] Cs_node[nn]; leftover[ni][nb]; ring[R]
 *   discharge stream; fatal (a log_err of lu_apply.c for this cell, see
 *   vicwur_lu_shim/plugin.h).
 * Tiles >= nv_active are padding (absent from the C domain, written as 0).
 */
#include <vic_driver_image.h>
#include <plugin.h>

all_vars_struct     *all_vars;
soil_con_struct     *soil_con;
veg_con_struct     **veg_con;
veg_lib_struct     **veg_lib;
veg_con_map_struct  *veg_con_map;
domain_struct        local_domain;
dmy_struct          *dmy;
size_t               current;
force_data_struct   *force;
option_struct        options;
parameters_struct    param;
global_param_struct  global_param;
size_t               NR, NF;
int                 *lu_fatal_cell;

void
sprint_location(char *str, location_struct *loc)
{
    sprintf(str, "cell %zu", loc->local_idx);
}

static double *rd;
#define NEXT (*rd++)
#define NEXTZ ((size_t) (*rd++))

int
main(int argc, char **argv)
{
    FILE   *fi, *fo;
    long    nbytes;
    double *buf;
    size_t  nc, nt, nb, nl, nn, nf, ni, rps, uh, R, c, t, b, l, f, n, i, k;
    size_t *nva;

    if (argc != 3) {
        fprintf(stderr, "usage: ref_vicwur_lu in out\n");
        return 2;
    }
    fi = fopen(argv[1], "rb");
    fseek(fi, 0, SEEK_END);
    nbytes = ftell(fi);
    fseek(fi, 0, SEEK_SET);
    buf = malloc(nbytes);
    if (fread(buf, 1, nbytes, fi) != (size_t) nbytes) {
        return 3;
    }
    fclose(fi);
    rd = buf;

    LOG_DEST = stderr;
    initialize_options();
    initialize_parameters();
    nc = NEXTZ; nt = NEXTZ; nb = NEXTZ; nl = NEXTZ; nn = NEXTZ; nf = NEXTZ;
    ni = NEXTZ; rps = NEXTZ; uh = NEXTZ;
    options.NVEGTYPES = NEXTZ;
    plugin_options.Pbare = NEXTZ;
    plugin_options.IRRIGATION = NEXT != 0;
    plugin_options.ROUTING = NEXT != 0;
    options.FULL_ENERGY = NEXT != 0;
    global_param.dt = NEXT;
    {
        double doy = NEXT;
        dmy = calloc(1, sizeof(*dmy));
        dmy[0].day_in_year = (unsigned short) doy;
        current = 0;
    }
    param.SNOW_MAX_SURFACE_SWE = NEXT;
    options.Nlayer = nl;
    options.Nnode = nn;
    options.Nfrost = nf;
    options.SNOW_BAND = nb;
    options.FROZEN_SOIL = false;
    options.QUICK_FLUX = false;
    global_param.model_steps_per_day = 1;
    plugin_global_param.rout_steps_per_day = rps;
    plugin_options.UH_LENGTH = (unsigned short) uh;
    R = uh + rps - 1;
    NR = 0;

    local_domain.ncells_active = nc;
    local_domain.locations = calloc(nc, sizeof(location_struct));
    all_vars = calloc(nc, sizeof(*all_vars));
    soil_con = calloc(nc, sizeof(*soil_con));
    veg_con = calloc(nc, sizeof(*veg_con));
    veg_lib = calloc(nc, sizeof(*veg_lib));
    veg_con_map = calloc(nc, sizeof(*veg_con_map));
    force = calloc(nc, sizeof(*force));
    lu_force = calloc(nc, sizeof(*lu_force));
    irr_con_map = calloc(nc, sizeof(*irr_con_map));
    irr_con = calloc(nc, sizeof(*irr_con));
    irr_var = calloc(nc, sizeof(*irr_var));
    rout_var = calloc(nc, sizeof(*rout_var));
    nva = calloc(nc, sizeof(*nva));
    lu_fatal_cell = calloc(nc, sizeof(int));

    for (c = 0; c < nc; c++) {
        soil_con_struct *sc = &soil_con[c];
        local_domain.locations[c].area = NEXT;
        local_domain.locations[c].local_idx = c;
        sc->lat = NEXT;
        nva[c] = NEXTZ;
        force[c].air_temp = calloc(1, sizeof(double));
        force[c].air_temp[0] = NEXT;
        sc->FS_ACTIVE = NEXT != 0;
        sc->b_infilt = NEXT;
        sc->AreaFract = calloc(nb, sizeof(double));
        sc->Tfactor = calloc(nb, sizeof(double));
        for (b = 0; b < nb; b++) {
            sc->AreaFract[b] = NEXT;
            sc->Tfactor[b] = NEXT;
        }
        for (f = 0; f < nf; f++) {
            sc->frost_fract[f] = 1. / nf;
        }
        for (l = 0; l < nl; l++) {
            sc->max_moist[l] = NEXT;
            sc->depth[l] = NEXT;
            sc->porosity[l] = NEXT;
            sc->resid_moist[l] = NEXT;
            sc->Wpwp[l] = NEXT;
            sc->bulk_dens_min[l] = NEXT;
            sc->soil_dens_min[l] = NEXT;
            sc->quartz[l] = NEXT;
            sc->bulk_density[l] = NEXT;
            sc->soil_density[l] = NEXT;
            sc->organic[l] = NEXT;
        }
        for (n = 0; n < nn; n++) {
            sc->dz_node[n] = NEXT;
            sc->Zsum_node[n] = NEXT;
            sc->max_moist_node[n] = NEXT;
            sc->expt_node[n] = NEXT;
            sc->bubble_node[n] = NEXT;
        }

        veg_con_map[c].nv_types = options.NVEGTYPES;
        veg_con_map[c].nv_active = nva[c];
        veg_con_map[c].vidx = malloc(options.NVEGTYPES * sizeof(int));
        veg_con_map[c].Cv = calloc(options.NVEGTYPES + 1, sizeof(double));
        for (k = 0; k < options.NVEGTYPES; k++) {
            veg_con_map[c].vidx[k] = NODATA_VEG;
        }
        veg_con[c] = calloc(nt, sizeof(veg_con_struct));
        veg_lib[c] = calloc(options.NVEGTYPES + 1, sizeof(veg_lib_struct));
        lu_force[c].Cv = calloc(nt, sizeof(double));
        lu_force[c].veg_class = calloc(nt, sizeof(double));
        all_vars[c].veg_var = calloc(nt, sizeof(veg_var_struct *));
        all_vars[c].cell = calloc(nt, sizeof(cell_data_struct *));
        all_vars[c].snow = calloc(nt, sizeof(snow_data_struct *));
        all_vars[c].energy = calloc(nt, sizeof(energy_bal_struct *));
        for (t = 0; t < nt; t++) {
            size_t vc = NEXTZ;
            bool   ov = NEXT != 0;
            double cvo = NEXT, cvf = NEXT;
            all_vars[c].veg_var[t] = calloc(nb, sizeof(veg_var_struct));
            all_vars[c].cell[t] = calloc(nb, sizeof(cell_data_struct));
            all_vars[c].snow[t] = calloc(nb, sizeof(snow_data_struct));
            all_vars[c].energy[t] = calloc(nb, sizeof(energy_bal_struct));
            if (t < nva[c]) {
                veg_con[c][t].veg_class = (int) vc;
                veg_con[c][t].Cv = cvo;
                veg_lib[c][vc].overstory = ov;
                if (vc < options.NVEGTYPES) {
                    veg_con_map[c].vidx[vc] = (int) t;
                    veg_con_map[c].Cv[vc] = cvo;
                    lu_force[c].veg_class[t] = (double) vc;
                }
                /* lu_forcing: lu_force.Cv of a mapped tile is the record,
                 * the VIC bare tile keeps 0 (lu_init_library.c:40) */
                lu_force[c].Cv[t] = vc < options.NVEGTYPES ? cvf : 0.;
            }
        }
        for (t = 0; t < nt; t++) {
            for (b = 0; b < nb; b++) {
                veg_var_struct    *vv = &all_vars[c].veg_var[t][b];
                cell_data_struct  *cd = &all_vars[c].cell[t][b];
                snow_data_struct  *sn = &all_vars[c].snow[t][b];
                energy_bal_struct *en = &all_vars[c].energy[t][b];
                vv->LAI = NEXT;
                vv->Wdew = NEXT;
                sn->swq = NEXT;
                sn->pack_water = NEXT;
                sn->surf_water = NEXT;
                sn->snow_canopy = NEXT;
                sn->coldcontent = NEXT;
                sn->density = NEXT;
                sn->depth = NEXT;
                sn->last_snow = (unsigned int) NEXT;
                sn->MELTING = NEXT != 0;
                sn->surf_temp = NEXT;
                sn->pack_temp = NEXT;
                vv->AnnualNPP = NEXT;
                vv->AnnualNPPPrev = NEXT;
                cd->CLitter = NEXT;
                cd->CInter = NEXT;
                cd->CSlow = NEXT;
                for (l = 0; l < nl; l++) {
                    cd->layer[l].moist = NEXT;
                }
                for (l = 0; l < nl; l++) {
                    for (f = 0; f < nf; f++) {
                        cd->layer[l].ice[f] = NEXT;
                    }
                }
                for (n = 0; n < nn; n++) {
                    en->T[n] = NEXT;
                }
            }
        }
        irr_con_map[c].ni_active = NEXTZ;
        irr_con[c] = calloc(ni > 0 ? ni : 1, sizeof(irr_con_struct));
        irr_var[c] = calloc(ni > 0 ? ni : 1, sizeof(irr_var_struct *));
        for (i = 0; i < ni; i++) {
            irr_con[c][i].veg_index = (int) NEXT;
            irr_con[c][i].paddy = NEXT != 0;
            irr_var[c][i] = calloc(nb, sizeof(irr_var_struct));
            for (b = 0; b < nb; b++) {
                irr_var[c][i][b].leftover = NEXT;
            }
        }
        rout_var[c].dt_discharge = calloc(R, sizeof(double));
        for (k = 0; k < R; k++) {
            rout_var[c].dt_discharge[k] = NEXT;
        }
    }
    lu_apply();

    fo = fopen(argv[2], "wb");
#define W(x) do { double _v = (double) (x); fwrite(&_v, sizeof(double), 1, fo); } while (0)
    for (c = 0; c < nc; c++) {
        for (t = 0; t < nt; t++) {
            W(t < nva[c] ? veg_con[c][t].Cv : 0.);
        }
        for (t = 0; t < nt; t++) {
            for (b = 0; b < nb; b++) {
                veg_var_struct    *vv = &all_vars[c].veg_var[t][b];
                cell_data_struct  *cd = &all_vars[c].cell[t][b];
                snow_data_struct  *sn = &all_vars[c].snow[t][b];
                energy_bal_struct *en = &all_vars[c].energy[t][b];
                W(vv->Wdew); W(sn->swq); W(sn->pack_water);
                W(sn->surf_water); W(sn->snow_canopy); W(sn->coldcontent);
                W(sn->density); W(sn->depth); W(sn->MELTING);
                W(sn->surf_temp); W(sn->pack_temp); W(vv->AnnualNPP);
                W(vv->AnnualNPPPrev); W(cd->CLitter); W(cd->CInter);
                W(cd->CSlow);
                for (l = 0; l < nl; l++) {
                    W(cd->layer[l].moist);
                }
                for (l = 0; l < nl; l++) {
                    for (f = 0; f < nf; f++) {
                        W(cd->layer[l].ice[f]);
                    }
                }
                for (n = 0; n < nn; n++) {
                    W(en->T[n]);
                }
                for (n = 0; n < nn; n++) {
                    W(en->Cs_node[n]);
                }
            }
        }
        for (i = 0; i < ni; i++) {
            for (b = 0; b < nb; b++) {
                W(irr_var[c][i][b].leftover);
            }
        }
        for (k = 0; k < R; k++) {
            W(rout_var[c].dt_discharge[k]);
        }
        W(rout_var[c].discharge);
        W(rout_var[c].stream);
        W(lu_fatal_cell[c]);
    }
    fclose(fo);
    return 0;
}
