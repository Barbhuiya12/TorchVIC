"""CropSyst .crp/.rot/.mgt/IRRIGPARAM/cropcodelib readers and the M5 calendar tables (CPU).

Fixtures are the shipped input files of the VIC-CropSyst CRB build copy (<data root>/build/vic_cropsyst),
the files used by the three working cases 330492_rot_3001, 308200_rot_3002, 328617_rot_3003. They are read,
never modified. $TORCHVIC_CROPSYST_DB moves the Database directory; the tests skip when it is absent.
"""
import csv
import glob
import os
import _paths  # noqa: E402
import re

import pytest
import torch

from torchvic.crop.cropsyst import calendars as cal
from torchvic.crop.cropsyst import readers as rd

DB = _paths.env_path(('TORCHVIC_CROPSYST_DB',), 'build/vic_cropsyst/VIC-CropSyst_v3_CRB/Simulation/Database')
SCEN = os.path.join(os.path.dirname(DB), "Scenarios")
pytestmark = pytest.mark.skipif(not os.path.isdir(DB), reason="CropSyst CRB build copy not present")

CASES = {330492: 3001, 308200: 3002, 328617: 3003}
CROPS = ["pea_spring-high", "wheat_spring-high", "wheat_spring-intermediate",
         "wheat_winter-high", "wheat_winter-intermediate", "wheat_winter-low"]


def _raw(path, section, key, nth=0):
    """Independent regex read of `key` in the nth [section...] block (first token of the value)."""
    text = open(path, encoding="latin-1").read().replace("\r", "")
    blocks = re.split(r"(?m)^(?=\[)", text)
    blocks = [b for b in blocks if re.match(r"\[%s(:\d+)?\]" % re.escape(section), b)]
    m = re.search(r"(?m)^%s=(.*)$" % re.escape(key), blocks[nth])
    return m.group(1).strip()


def _num(s):
    return float(re.match(r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?", s).group(0))


def test_crp_all_used_files_and_values():
    paths = [f"{DB}/Crop/name/{c}.crp" for c in CROPS]
    t = rd.crop_table(paths)
    assert t.n == 6 and t.params["leaf.max_LAI"].shape == (6,)
    for i, p in enumerate(paths):
        for sec, key in [("leaf", "max_LAI"), ("transpiration", "max_water_uptake"),
                         ("transpiration", "ET_crop_coef"), ("growth", "TUE_at_1pKa_VPD"),
                         ("phenology", "maturity"), ("phenology", "flowering"),
                         ("root", "max_root_depth"), ("harvest_index", "unstressed"),
                         ("phenology", "base_temp"), ("phenology", "cutoff_temp")]:
            if f"{sec}.{key}" in t.params:
                assert float(t.params[f"{sec}.{key}"][i]) == pytest.approx(_num(_raw(p, sec, key)), rel=1e-6)
    # known values read by eye from wheat_winter-low.crp
    j = CROPS.index("wheat_winter-low")
    assert float(t.params["leaf.max_LAI"][j]) == 6.0
    assert float(t.params["transpiration.max_water_uptake"][j]) == 9.0
    assert float(t.params["phenology.maturity"][j]) == 2308.0
    assert float(t.params["phenology.emergence"][j]) == 118.0
    assert bool(t.params["vernalization.enabled"][j])
    assert t.vocab["crop.life_cycle"][int(t.params["crop.life_cycle"][j])] == "annual"
    # repeated sections padded with a validity mask
    assert t.params["adjustment.value"].shape[0] == 6
    assert t.valid["adjustment"][j].sum() == 2
    assert float(t.params["adjustment.value"][j, 0]) == 30.0
    assert t.params["grain_stress_period.heat_stress_sensitivity"].shape[1] == 2


def test_crp_float32_and_text_host_side():
    t = rd.crop_table([f"{DB}/Crop/name/wheat_winter-low.crp"], dtype=torch.float32)
    assert t.params["leaf.max_LAI"].dtype == torch.float32
    assert t.text["crop.description"][0] == "winter wheat (low)"
    assert "crop.description" not in t.params


def test_mgt_all_used_files_and_values():
    names = [f"CT_{c}" for c in CROPS]
    paths = [f"{DB}/Management/Name/{n}.mgt" for n in names]
    t = rd.mgt_table(paths)
    j = names.index("CT_wheat_winter-low")
    assert int(t.params["management.tillage:count"][j]) == 8
    assert int(t.valid["tillage"][j].sum()) == 8
    assert int(t.valid["harvest"][j].sum()) == 1
    assert int(t.params["tillage.event_offset"][j, 0]) == 1
    assert int(t.params["tillage.event_offset"][j, 2]) == 151
    assert float(t.params["tillage.dust_mulch_intensity"][j, 4]) == pytest.approx(0.1)
    assert bool(t.params["tillage.dust_mulch"][j, 4])
    ph = t.vocab["tillage.event_phenologic"]
    assert ph[int(t.params["tillage.event_phenologic"][j, 1])] == "harvest"
    assert t.vocab["harvest.event_phenologic"][int(t.params["harvest.event_phenologic"][j, 0])] == "maturity"
    assert float(t.params["harvest.residue"][j, 0]) == 85.0
    assert float(t.params["harvest.min_LAI"][j, 0]) == 3.0
    assert float(t.params["fertilization.NH4_N"][j, 0]) == 80.0
    for i, p in enumerate(paths):  # every file: counts agree with the file text
        assert int(t.valid["tillage"][i].sum()) == int(_num(_raw(p, "management", "tillage:count")))
        assert int(t.valid["harvest"][i].sum()) == int(_num(_raw(p, "management", "harvest:count")))


def test_rot_three_cases_values():
    t = rd.rot_table([f"{DB}/Rotation/Code/{c}.CS_rotation" for c in (3001, 3002, 3003)])
    assert t.n == 3
    assert [int(x) for x in t.params["rotation.years"]] == [3, 3, 2]
    assert [int(x) for x in t.valid["sowing"].sum(1)] == [3, 2, 1]
    # date32: 3001 slots 0060 / 1060 / 1280(...); 3002 258 / 2060; 3003 244
    assert t.params["sowing.event_date"][0].tolist() == [60, 1060, 1280]
    assert t.params["sowing.event_date"][1, :2].tolist() == [258, 2060]
    assert int(t.params["sowing.event_date"][2, 0]) == 244
    assert rd.split_date32(2060) == (2, 60)
    sync = t.vocab["sowing.event_synchronization"]
    assert sync[int(t.params["sowing.event_synchronization"][0, 0])] == "relative_date"
    assert sync[int(t.params["sowing.event_synchronization"][2, 0])] == "actual_date"
    assert bool(t.params["sowing.terminate_crop"][2, 0]) and not bool(t.params["sowing.terminate_crop"][1, 0])
    assert float(t.params["sowing.avg_temp"][0, 0]) == 8.0
    assert t.text["sowing.crop"][0][1].endswith("pea_spring-high.crp")


def test_irrigparam_values_and_lookup():
    ip = rd.read_irrigparam(f"{DB}/Management/irrigation_parameter.txt")
    assert ip.cell_id.numel() == 7637
    # cell 330492 (5 tiles): 210 CP, 701 CP, 703 WHEEL_LINE, 713 BIG_GUN, 1205 FLOOD
    cells = torch.tensor([330492] * 6 + [308200, 308200, 328617])
    codes = torch.tensor([210, 701, 703, 713, 1205, 999, 201, 218, 218])
    out = ip.lookup(cells, codes).tolist()
    assert out == [30, 30, 60, 40, 71, 0, 71, 30, 0]  # 328617 has no record: no irrigation
    r = ip.cell_id.tolist().index(330492)
    assert int(ip.offset[r + 1] - ip.offset[r]) == 5
    with pytest.raises(rd.InputError):
        rd.irrigation_type_code("PIVOT_OF_DOOM")


def test_cropcodelib():
    lib = rd.read_cropcodelib(f"{DB}/Library/cropcodelib.txt")
    assert lib.codes.numel() == 19
    assert lib.names_for(218)[0] == "wheat_winter-high" and len(lib.names_for(218)) == 6
    assert lib.names_for(210)[0] == "wheat_spring-high"
    assert lib.names_for(0) == ["NOCROP"]


@pytest.mark.parametrize("cell,code", list(CASES.items()))
def test_vegparam_codes(cell, code):
    cells, codes = rd.read_vegparam_codes(f"{DB}/Veg/vic_vegparamegter_id_{cell}.txt")
    assert cells == [cell] and codes == [[code]]


def test_unknown_key_and_section_are_errors(tmp_path):
    src = open(f"{DB}/Crop/name/wheat_winter-low.crp", encoding="latin-1").read()
    bad = tmp_path / "bad.crp"
    bad.write_text(src.replace("[leaf]\n", "[leaf]\nbogus_key=1\n", 1).replace("[leaf]\r\n", "[leaf]\r\nbogus_key=1\r\n", 1))
    with pytest.raises(rd.InputError, match="unknown key 'bogus_key'"):
        rd.read_crp(str(bad))
    bad.write_text(src + "\n[nonsense]\na=1\n")
    with pytest.raises(rd.InputError, match="unknown section"):
        rd.read_crp(str(bad))
    ip = tmp_path / "ip.txt"
    ip.write_text("1 1\n 5 NOT_A_TYPE\n")
    with pytest.raises(rd.InputError, match="unknown irrigation type"):
        rd.read_irrigparam(str(ip))


def test_count_mismatch_is_error(tmp_path):
    src = open(f"{DB}/Management/Name/CT_wheat_winter-low.mgt", encoding="latin-1").read()
    bad = tmp_path / "bad.mgt"
    bad.write_text(src.replace("tillage:count=8", "tillage:count=9"))
    with pytest.raises(rd.InputError, match="tillage:count"):
        rd.read_mgt(str(bad))


def test_source_files_unchanged_readonly_use():
    # the readers open files for reading only: modification times of the fixtures stay put
    p = f"{DB}/Crop/name/wheat_winter-low.crp"
    before = os.stat(p).st_mtime_ns
    rd.read_crp(p)
    assert os.stat(p).st_mtime_ns == before


def test_rotation_set_links():
    rs = cal.load_rotations(DB, [3001, 3002, 3003])
    assert rs.years.tolist() == [3, 3, 2] and rs.n_sow.tolist() == [3, 2, 1]
    assert rs.sow_doy.tolist() == [[60, 60, 280], [258, 60, -1], [244, -1, -1]]
    assert rs.sow_year.tolist() == [[0, 1, 1], [0, 2, -1], [0, -1, -1]]
    names = [os.path.basename(p) for p in rs.crops.paths]
    assert [names[i] for i in rs.crop_idx[1][:2].tolist()] == ["wheat_winter-intermediate.crp",
                                                              "wheat_spring-intermediate.crp"]
    mnames = [os.path.basename(p) for p in rs.mgts.paths]
    assert mnames[int(rs.mgt_idx[2, 0])] == "CT_wheat_winter-low.mgt"
    assert rs.crops.n == 6 and rs.mgts.n == 6


def _csv_sowings(case, code_prefix):
    """Reference sowing years per expanded tile read from the build's crop csv (Crop_name changes)."""
    f = glob.glob(f"{SCEN}/Liu_Linux_{case}_rot_{code_prefix}/VIC_output/vic_crop_daily*")
    if not f:
        pytest.skip("reference crop csv not present")
    out, last = {}, {}
    for x in csv.DictReader(open(f[0])):
        k, nm = int(x["CroppingSyst_code"]), x["Crop_name"]
        if last.get(k) != nm and nm != "NOCROP":
            out.setdefault(k, []).append((int(x["Year"]), x["Crop_code"]))
        last[k] = nm
    return out


@pytest.mark.parametrize("case,rot", [(308200, 3002), (328617, 3003)])
def test_calendar_matches_reference_run(case, rot):
    ref = _csv_sowings(case, rot)
    rs = cal.load_rotations(DB, [rot])
    tc, tr, tp = cal.expand_tiles(rs, [rot])
    c = cal.build_calendar(rs, tr, tp, 1979, 5)
    assert sorted(ref) == sorted(c.runtime_code.tolist())
    crop_code = {"wheat_winter-intermediate": 218, "wheat_spring-intermediate": 210, "wheat_winter-low": 218}
    for t, code in enumerate(c.runtime_code.tolist()):
        mine = []
        for y in range(5):
            for s in range(c.sow_doy.shape[2]):
                if c.sow_doy[t, y, s] >= 0:
                    nm = os.path.basename(rs.crops.paths[int(c.crop_idx[t, y, s])])[:-4]
                    mine.append((1979 + y, str(crop_code[nm])))
        assert mine == ref[code]


def test_calendar_shapes_and_device_dtype():
    rs = cal.load_rotations(DB, [3001, 3002, 3003], device="cpu", dtype=torch.float32)
    tc, tr, tp = cal.expand_tiles(rs, [3001, 3002, 3003])
    assert tc.shape == (8,) and tp.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    c = cal.build_calendar(rs, tr, tp, 1979, 5)
    assert c.sow_doy.shape == (8, 5, 3) and c.cycle_year.shape == (8, 5)
    assert c.runtime_code.tolist() == [300100, 300101, 300102, 300200, 300201, 300202, 300300, 300301]
    d, cr, m = c.at_year(1980)
    assert d.shape == (8, 3)
    assert rs.crops.params["leaf.max_LAI"].dtype == torch.float32
