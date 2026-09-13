"""Reading the packed archive and turning it into conditioned arrays.

    from archive_pipeline.archive import find_chunks, read_chunk
    from archive_pipeline.archive import component_segments, pick_components

`find_chunks` locates work, `read_chunk` decodes one unit of it, and
`component_segments` turns that into contiguous arrays. Everything above this
layer consumes segments and never touches ObsPy.
"""
from archive_pipeline.archive.chunks import (CHUNK_RE, chunk_date, find_chunks,
                                             nonconforming, read_chunk)
from archive_pipeline.archive.clean import (clean_block, clean_window,
                                            taper_vector)
from archive_pipeline.archive.segments import (COMPONENT_ROLES,
                                               component_segments,
                                               make_windows, pick_components)
from archive_pipeline.archive.spans import (clip_spans, common_spans,
                                            coverage_spans, in_spans,
                                            intersect_spans, merge_intervals)

__all__ = ["CHUNK_RE", "chunk_date", "find_chunks", "nonconforming",
           "read_chunk",
           "clean_block", "clean_window", "taper_vector",
           "COMPONENT_ROLES", "component_segments", "make_windows",
           "pick_components", "clip_spans", "common_spans", "coverage_spans",
           "in_spans", "intersect_spans", "merge_intervals"]
