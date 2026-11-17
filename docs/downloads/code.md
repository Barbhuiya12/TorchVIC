# Code and license

## Code

The source is at <https://github.com/Barbhuiya12/TorchVIC>. The current release
is 0.4.8; earlier versions are tagged. TorchVIC is not on PyPI; install it
from the repository ([Install](../install.md)). The changes of each version are
in the [Release notes](../development/release-notes.md).

To cite TorchVIC, see [How to cite](../about/citing.md).

## License and credits

TorchVIC is free software under the GNU General Public License, version 3 or (at
your option) any later version, Copyright (C) 2026 Siddik Barbhuiya. VIC
5.1.0 is MIT-licensed and VIC 5.0.1 is GPL-2.0-or-later, so both can be
combined with it. The full text is
[`LICENSE`](https://github.com/Barbhuiya12/TorchVIC/blob/main/LICENSE) at the
repository root; the credits and licenses of the codes TorchVIC re-implements
and of the data its tools read are in
[`NOTICE`](https://github.com/Barbhuiya12/TorchVIC/blob/main/NOTICE) and
[`THIRD_PARTY_NOTICES.md`](https://github.com/Barbhuiya12/TorchVIC/blob/main/THIRD_PARTY_NOTICES.md)
and are summarised in [How to cite](../about/citing.md#credits).

### Credits

* **VIC**, the Variable Infiltration Capacity model of the Computational
  Hydrology Group, University of Washington,
  <https://github.com/UW-Hydro/VIC>. TorchVIC implements the physics of
  releases 5.0.1 and 5.1.0; every process function cites the C file and lines it
  follows ([Process-to-C mapping](../mapping.md)).
* **MTCLIM** (Thornton and Running 1999; Bohn et al. 2013), through MetSim
  (UW-Hydro), for the daily-forcing disaggregation.
* **RVIC** for the routing conventions, and Lohmann, Nolte-Holube and Raschke
  (1996) for the channel impulse response.
* **VIC_sample_data** (UW-Hydro) for the examples used in the tests and
  tutorials.
* **PyTorch** and its inductor and Triton compilers for the device code.

Flow-direction tools under `tools/` read HydroSHEDS, MERIT-Hydro and GRIT data,
each under its own license (GRIT is CC BY-NC 4.0, non-commercial use only); the
tools are not required to run the model.

### No warranty

TorchVIC is distributed without any warranty; see the license text.
