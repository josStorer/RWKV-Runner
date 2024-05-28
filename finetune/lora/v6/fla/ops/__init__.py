# -*- coding: utf-8 -*-

from .based import fused_chunk_based, parallel_based
from .gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from .retention import (chunk_retention, fused_chunk_retention,
                        fused_recurrent_retention, parallel_retention)

__all__ = [
    'fused_chunk_based',
    'parallel_based',
    'chunk_gla',
    'fused_chunk_gla',
    'fused_recurrent_gla',
    'chunk_retention',
    'fused_chunk_retention',
    'fused_recurrent_retention',
    'parallel_retention'
]
