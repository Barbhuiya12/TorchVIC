"""Running one case over several devices: cells split, outputs merged."""
from .shard import (cell_weights, shard_ranges, shard_index, restrict_domain, sharded,
                    merge_data, merge_state_files)

__all__ = ['cell_weights', 'shard_ranges', 'shard_index', 'restrict_domain', 'sharded',
           'merge_data', 'merge_state_files']
