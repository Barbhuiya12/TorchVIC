"""CropSyst crop backend for TorchVIC (VIC-CropSyst port): input readers (readers.py) and the
rotation/sowing calendar tables (calendars.py); E1-E3 root water uptake, soil water potential and
uptake into the VIC layers (uptake.py), batched over [..., layer]; E6 thermal time and phenology
(phenology.py), stage switches as masks over [...]. All parsing is Python at setup; the outputs are
tensors batched over crops, management files, rotations and cells, on any device."""
