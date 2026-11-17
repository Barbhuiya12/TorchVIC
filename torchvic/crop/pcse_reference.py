# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""Make PCSE reference runs for validating torchvic.crop (needs `pcse`).

Run with the python that has pcse 6.0.13 installed (not the torchvic env):

    python -m torchvic.crop.pcse_reference OUTDIR [--cropdir DIR] [--small]

Each case becomes one `.npz` holding the daily weather PCSE used, the flat
parameter dict, the run setup and PCSE's daily outputs, so the comparison
(tests/test_crop_pcse.py) needs neither pcse nor its database.
"""
import argparse
import copy
import datetime as dt
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import namedtuple

import numpy as np

OUT_VARS = ["DVS", "LAI", "TAGP", "TWSO", "TWLV", "TWST", "TWRT", "RD", "TRA", "SM", "RFTRA",
            "EVS", "EVSMX", "EVWMX", "TRAMX", "WWLOW"]
WX = ["TMIN", "TMAX", "IRRAD", "RAIN", "E0", "ES0", "ET0"]


def namedtuple_factory(cursor, row):
    fields = [c[0] for c in cursor.description]
    return namedtuple("Row", fields)._make(row)


def shift_year(d, dy):
    try:
        return d.replace(year=d.year + dy)
    except ValueError:
        return d.replace(year=d.year + dy, day=28)


class ShiftedLat:
    """Wrap a weather provider and change its latitude."""

    def __new__(cls, wdp, lat):
        from pcse.base import WeatherDataProvider
        from pcse.base.weather import WeatherDataContainer

        class _W(WeatherDataProvider):
            def __init__(self):
                super().__init__()
                self.latitude, self.longitude, self.elevation = lat, wdp.longitude, wdp.elevation
                self.angstA, self.angstB = wdp.angstA, wdp.angstB

            def __call__(self, day, member_id=0):
                c = wdp(day)
                kw = {k: getattr(c, k) for k in WX + ["VAP", "WIND", "TEMP", "SNOWDEPTH"]
                      if hasattr(c, k)}
                kw.update(LAT=lat, LON=c.LON, ELEV=c.ELEV, DAY=c.DAY)
                return WeatherDataContainer(**kw)
        return _W()


def run_pcse(parvalues, wdp, agro, mode, extra_days=0):
    from pcse.models import Wofost72_PP, Wofost72_WLP_CWB
    cls = Wofost72_PP if mode == "pp" else Wofost72_WLP_CWB
    eng = cls(parvalues, wdp, agro, output_vars=OUT_VARS)
    eng.run_till_terminate()
    out = [o for o in eng.get_output() if o.get("DVS") is not None]
    summ = eng.get_summary_output()
    pars = {}
    for k in parvalues._unique_parameters:
        v = parvalues[k]
        if isinstance(v, (int, float, str)):
            pars[k] = v
        elif isinstance(v, (list, tuple)):
            pars[k] = [float(x) for x in v]
        elif hasattr(v, "x_list"):
            pars[k] = [x for xy in zip(v.x_list, v.y_list) for x in xy]
    return out, summ, pars, eng


def save_case(path, name, mode, out, summ, pars, wdp, agro, extra):
    days = [o["day"] for o in out]
    wx = {k: [] for k in WX}
    for d in days:
        c = wdp(d)
        for k in WX:
            wx[k].append(float(getattr(c, k)))
    arr = {"wx_" + k: np.array(v) for k, v in wx.items()}
    arr["doy"] = np.array([d.timetuple().tm_yday for d in days], dtype=float)
    arr["lat"] = np.array(float(wdp(days[0]).LAT))
    for v in OUT_VARS:
        arr["o_" + v] = np.array([np.nan if o.get(v) is None else float(o[v]) for o in out])
    cal = agro[0][list(agro[0].keys())[0]]["CropCalendar"]
    meta = dict(name=name, mode=mode, params=pars, ndays=len(days),
                start=str(days[0]), end=str(days[-1]),
                crop_start_type=cal["crop_start_type"], crop_end_type=cal["crop_end_type"],
                max_duration=int(cal["max_duration"]),
                harvest_idx=(None if cal["crop_end_date"] in (None, "null") else
                             (cal["crop_end_date"] - days[0]).days),
                summary={k: (str(v) if isinstance(v, dt.date) else v) for k, v in summ[0].items()},
                **extra)
    arr["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arr)


def agro_dict(crop_name, variety, start, start_type, end_type, end_date, max_duration):
    return [{start: {"CropCalendar": {
        "crop_name": crop_name, "variety_name": variety, "crop_start_date": start,
        "crop_start_type": start_type, "crop_end_date": end_date,
        "crop_end_type": end_type, "max_duration": max_duration},
        "TimedEvents": None, "StateEvents": None}}]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--cropdir", default=None, help="WOFOST wofost72 crop parameter dir")
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    from pcse.base import ParameterProvider
    from pcse.settings import settings
    from pcse.tests.db_input import (GridWeatherDataProvider, fetch_soildata, fetch_sitedata,
                                     fetch_cropdata, AgroManagementDataProvider)
    db = sqlite3.connect(os.path.join(settings.PCSE_USER_HOME, "pcse.db"))
    db.row_factory = namedtuple_factory
    grid = 31031
    wdp = GridWeatherDataProvider(db, grid_no=grid)
    soild = fetch_soildata(db, grid)
    sited = fetch_sitedata(db, grid, 2000)
    from pcse.input import CABOWeatherDataProvider
    import pcse
    nl1 = CABOWeatherDataProvider("NL1", fpath=os.path.join(os.path.dirname(pcse.__file__),
                                                              "tests", "test_data"))
    done = []

    def go(name, parvalues, wp, agro, modes=("pp", "wlp"), extra=None):
        for mode in modes:
            pv = copy.deepcopy(parvalues) if not hasattr(parvalues, "_unique_parameters") else parvalues
            try:
                out, summ, pars, eng = run_pcse(pv, wp, agro, mode)
            except Exception as e:  # keep going, report
                print("FAIL", name, mode, repr(e)[:200])
                continue
            save_case(os.path.join(a.outdir, "%s_%s.npz" % (name, mode)), name, mode, out, summ,
                      pars, wp, agro, extra or {})
            s = summ[0]
            print("ok %-28s %-3s days %3d TWSO %8.1f LAIMAX %5.2f TAGP %8.1f" %
                  (name, mode, len(out), s["TWSO"], s["LAIMAX"], s["TAGP"]))
            done.append(name + "_" + mode)

    # ---- PCSE demo cases (sqlite demo database): winter wheat, grain maize, spring
    # barley, potato, rapeseed, sunflower; year 2000 and shifted weather years
    cropnames = {1: "wheat", 2: "grainmaize", 3: "sbarley", 7: "potato", 10: "rapeseed", 11: "sunflower"}
    years = [2000] if a.small else [2000, 1995, 1992]  # 2000: demo grid weather, others NL1
    for crop, cname in cropnames.items():
        for yr in years:
            agm = AgroManagementDataProvider(db, grid, crop, 2000)
            cropd = fetch_cropdata(db, grid, 2000, crop)
            agro = copy.deepcopy(list(agm))
            dy = yr - 2000
            if dy:
                a0 = agro[0]
                k0 = list(a0.keys())[0]
                cc = a0[k0]["CropCalendar"]
                for kk in ("crop_start_date", "crop_end_date"):
                    if isinstance(cc[kk], dt.date):
                        cc[kk] = shift_year(cc[kk], dy)
                agro = [{shift_year(k0, dy): a0[k0]}]
            pv = ParameterProvider(sitedata=sited, cropdata=cropd, soildata=soild)
            go("demo_%s_%d" % (cname, yr), pv, wdp if yr == 2000 else nl1, agro)
            if a.small:
                break

    # ---- WOFOST crop parameter repository (branch wofost72)
    if a.cropdir:
        from pcse.input import YAMLCropDataProvider
        tmp = tempfile.mkdtemp()
        for f in os.listdir(a.cropdir):
            shutil.copy(os.path.join(a.cropdir, f), tmp)
        cp = YAMLCropDataProvider(fpath=tmp)
        D = dt.date
        # (case, crop, variety, start date, start type, end type, end date, max_dur, lat)
        cases = [
            ("wheatw101_es", "wheat", "Winter_wheat_101", D(1998, 10, 20), "sowing", "maturity", None, 330, None),
            ("wheatw102_es98", "wheat", "Winter_wheat_102", D(1997, 11, 5), "sowing", "maturity", None, 330, None),
            ("wheatbermude_in", "wheat", "bermude", D(1998, 11, 10), "sowing", "harvest", D(1999, 5, 10), 300, 30.9),
            ("maize201_es", "maize", "Grain_maize_201", D(1995, 4, 20), "emergence", "maturity", None, 250, None),
            ("maize202_es00", "maize", "Grain_maize_202", D(1998, 5, 1), "sowing", "earliest", D(1998, 10, 20), 250, None),
            ("maizevh_in", "maize", "Maize_VanHeemst_1988", D(1996, 6, 20), "sowing", "maturity", None, 250, 25.0),
            ("rice_ir72_es", "rice", "Rice_IR72", D(1994, 5, 15), "emergence", "maturity", None, 250, None),
            ("rice_ir8_in", "rice", "Rice_HYV_IR8", D(1997, 7, 1), "emergence", "maturity", None, 250, 12.0),
            ("rice_501_es", "rice", "Rice_501", D(1998, 5, 10), "sowing", "maturity", None, 250, None),
            ("sugarcane_in", "sugarcane", "Sugarcane_VanHeemst_1988", D(1996, 2, 1), "emergence", "maturity", None, 365, 12.0),
            ("sugarcane_in30", "sugarcane", "Sugarcane_VanHeemst_1988", D(1996, 2, 15), "sowing", "harvest", D(1997, 2, 10), 400, 30.0),
        ]
        for yr in (1978, 1984, 1987, 1993, 1994, 1996):
            cases.append(("wheatw101_nl%d" % yr, "wheat", "Winter_wheat_101", D(yr, 10, 25), "sowing", "maturity", None, 330, None))
            cases.append(("ricelat_%d" % yr, "rice", "Rice_IR72", D(yr + 1, 4, 20), "emergence", "maturity", None, 250, 22.0))
        if a.small:
            cases = cases[:1] + cases[3:4] + cases[6:7] + cases[9:10]
        for (name, crop, var, start, st, et, end, md, lat) in cases:
            cp.set_active_crop(crop, var)
            for co2 in (None,):
                site = dict(sited)
                if co2:
                    site["CO2"] = co2
                pv = ParameterProvider(sitedata=site, cropdata=cp, soildata=soild)
                wp = nl1 if lat is None else ShiftedLat(nl1, lat)
                agro = agro_dict(crop, var, start, st, et, end if end else "null", md)
                agro[0][start]["CropCalendar"]["crop_end_date"] = end
                go(name, pv, wp, agro, extra=dict(crop=crop, variety=var, lat=lat))
        # CO2 effect on potential transpiration (EvapotranspirationCO2 path)
        cp.set_active_crop("wheat", "Winter_wheat_101")
        cp_co2 = {k: v for k, v in cp.items()}
        cp_co2["CO2"] = 560.
        pv = ParameterProvider(sitedata=sited, cropdata=cp_co2, soildata=soild)
        agro = agro_dict("wheat", "Winter_wheat_101", D(1999, 12, 5), "sowing", "maturity", None, 330)
        go("wheat_co2_560", pv, wdp, agro, extra=dict(crop="wheat"))
        shutil.rmtree(tmp, ignore_errors=True)
    print(len(done), "runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
