# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Siddik Barbhuiya
"""VIC-WUR 3.0's irrigation flag needs the root zone below Wcr
(irr_run.c:165-177), so a short supply pulses on alternate days; the fixed
mode holds the flag while the requirement is positive."""
import torch

from torchvic.vicwur import irrigation as IR

DT = torch.float64


def _req(moist, hold):
    t = lambda x: torch.tensor(x, dtype=DT)
    return IR.requirement(
        t([[[[moist]]]]), t([[[[[0.0]]]]]), t([[1.0]]), t([[[[1.0]]]]),
        t([[[[50.0]]]]), t([[80.0]]), t([100.0]), t([[[0.0]]]),
        torch.ones(1, 1, 1, dtype=torch.int64), torch.zeros(1, 1, dtype=torch.int64),
        torch.zeros(1, 1, dtype=torch.bool), t([[1.0]]), t([[1.0]]),
        hold=None if hold is None else torch.tensor([[[hold]]]))


def test_w3_flag_closes_above_wcr():
    req, flag = _req(51.0, None)
    assert req.item() == 29.0 and not flag.item()
    assert _req(49.0, None)[1].item()


def test_hold_keeps_flag_until_field_capacity():
    assert _req(51.0, True)[1].item()          # held: still filling
    assert not _req(51.0, False)[1].item()     # not flagged yesterday
    assert not _req(81.0, True)[1].item()      # at Wfc the requirement closes


def test_short_supply_no_alternate_days():
    """2 mm per irrigated day against 0.8 mm/day of ET around Wcr = 50."""
    def run(hold):
        m, prev, on = 49.5, False, []
        for _ in range(12):
            _, f = _req(m, prev if hold else None)
            f = bool(f.item())
            on.append(f)
            m += 2.0 if f else 0.0
            m -= 0.8
            prev = f
        return on
    w3 = run(False)
    assert sum(a != b for a, b in zip(w3, w3[1:])) >= 9      # pulses
    held = run(True)
    assert all(held)


def _hold(flag, req, rec):
    from torchvic.coupling.connectors.vicwur_irrigation import \
        VicWurIrrigationConnector as C
    t = lambda x: torch.tensor([[[x]]], dtype=DT)
    return C._hold(dict(flag=torch.tensor([[[flag]]]), requirement=t(req),
                        received=t(rec)))


def test_hold_only_after_a_short_supply():
    """The hold is for short supply only; a tile that got
    its full requirement keeps VIC-WUR 3.0's flag (no daily top-up to Wfc)."""
    assert _hold(True, 30.0, 2.0).item()         # short: held
    assert not _hold(True, 30.0, 30.0).item()    # full: VIC-WUR 3.0's flag
    assert not _hold(False, 30.0, 0.0).item()    # not flagged yesterday


def test_full_supply_follows_w3():
    """Full supply each flagged day, 0.8 mm/day of ET and 0.5 mm/day of
    drainage near Wfc: VIC-WUR 3.0's flag and the fixed-mode flag are the
    same days."""
    def run(fixed):
        m, st, on = 49.5, None, []
        for _ in range(20):
            hold = _hold(*st) if (fixed and st) else None
            req, f = _req(m, None if hold is None else bool(hold.item()))
            f, r = bool(f.item()), req.item()
            on.append(f)
            rec = r if f else 0.0
            m += rec - 1.3
            st = (f, r, rec)
        return on
    assert run(True) == run(False)
