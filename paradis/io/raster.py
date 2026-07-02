"""Raster I/O utilities.

Provides helpers for loading and pre-processing geospatial rasters, and for
rasterising shapefiles to match a reference TIF.

Key functions
-------------
:func:`read_raster`
    Robust GeoTIFF loader — handles NODATA, scale/offset, integer types,
    and common ENM conventions automatically.
:func:`load_hs`
    Load and normalise a habitat-suitability raster to ``[0, 1]``.
:func:`load_obs`
    Load a species-observation count raster.
:func:`load_mask`
    Load a binary region mask.
:func:`project_shp`
    Rasterise a shapefile using a reference TIF as the spatial template.
"""

from __future__ import annotations

import pathlib

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_tif(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    if p.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(
            f"Expected a .tif / .tiff file, got: '{p.name}'\n"
            f"Full path: {p}"
        )
    if not p.exists():
        raise FileNotFoundError(f"Raster not found: {p}")
    return p


def _open_rasterio(path: pathlib.Path, band: int):
    """Open *path* with rasterio and return (data_float32, nodata_value)."""
    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required for raster I/O.  "
            "Install with:  pip install rasterio"
        )

    with rasterio.open(path) as src:
        raw  = src.read(band)
        meta_nodata = src.nodata

        # Apply rasterio scale + offset (e.g. uint16 ENM stored as int*0.0001)
        scales  = src.scales  if src.scales  else (1.0,)
        offsets = src.offsets if src.offsets else (0.0,)
        scale  = float(scales[band - 1])
        offset = float(offsets[band - 1])

    data = raw.astype(np.float32)
    if scale != 1.0 or offset != 0.0:
        data = data * scale + offset

    return data, meta_nodata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_raster(
    path: str | pathlib.Path,
    band: int = 1,
    nodata: float | None = None,
    fill_value: float = 0.0,
    clip_range: tuple[float | None, float | None] | None = None,
    normalise: bool = False,
    drop_max_nodata: bool = False,
) -> np.ndarray:
    """Robust GeoTIFF loader — handles the most common raster quirks.

    Parameters
    ----------
    path:
        Path to a ``.tif`` / ``.tiff`` file.
    band:
        Band index to read (1-based, default 1).
    nodata:
        NODATA sentinel value.  When ``None`` the value is read from the
        raster metadata.  Pixels equal to this value are replaced with
        *fill_value*.
    fill_value:
        Replacement for NODATA pixels (default 0).
    clip_range:
        ``(lo, hi)`` tuple applied after NODATA handling.  Use ``None`` for
        either bound to skip that side, e.g. ``(0, 1)`` to clamp to [0, 1].
    normalise:
        If ``True``, divide by the maximum valid value so the output lives
        in ``[0, 1]``.  Applied after clipping.
    drop_max_nodata:
        If ``True``, treat the global maximum as an additional NODATA
        sentinel before normalising — common in ENM outputs stored as
        integers where 65535 / 32767 marks "out of extent".

    Returns
    -------
    numpy.ndarray
        2-D float32 array with the same spatial dimensions as the raster.

    Examples
    --------
    >>> from paradis.io import read_raster
    >>> hs = read_raster("HS_species.tif", clip_range=(0, None), normalise=True)
    >>> obs = read_raster("Obs_species.tif", fill_value=0)
    """
    p    = _require_tif(path)
    data, meta_nodata = _open_rasterio(p, band)

    # ── NODATA handling ──────────────────────────────────────────────────────
    nd = nodata if nodata is not None else meta_nodata
    if nd is not None:
        mask = np.isclose(data, float(nd), rtol=0, atol=1e-3)
        data[mask] = fill_value

    # Also replace IEEE NaN / Inf from bad rasters
    data = np.nan_to_num(data, nan=fill_value, posinf=fill_value, neginf=fill_value)

    # ── Optional: drop ENM integer max as NODATA ─────────────────────────────
    if drop_max_nodata:
        global_max = data.max()
        if global_max > 0:
            data[data == global_max] = fill_value

    # ── Clip ────────────────────────────────────────────────────────────────
    if clip_range is not None:
        lo, hi = clip_range
        data = np.clip(data, lo if lo is not None else -np.inf,
                             hi if hi is not None else  np.inf)

    # ── Normalise ────────────────────────────────────────────────────────────
    if normalise:
        vmax = data.max()
        if vmax > 0:
            data = data / vmax

    return data.astype(np.float32)


def load_hs(path: str | pathlib.Path) -> np.ndarray:
    """Load a habitat-suitability raster, normalised to ``[0, 1]``.

    Handles the two most common ENM storage conventions:

    * **Float rasters** in ``[0, 1]`` — returned as-is after NODATA removal.
    * **Integer rasters** (uint8 / uint16) — divided by their maximum valid
      value (255 or 65534 etc.) after stripping the global max as NODATA.

    Parameters
    ----------
    path:
        Path to a ``.tif`` habitat-suitability file.

    Returns
    -------
    numpy.ndarray
        2-D float32 array in ``[0, 1]`` with NODATA replaced by 0.
    """
    p    = _require_tif(path)
    data, meta_nodata = _open_rasterio(p, band=1)

    nd = meta_nodata
    if nd is not None:
        data[np.isclose(data, float(nd), rtol=0, atol=1e-3)] = 0.0
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None)

    # Strip global max as NODATA (ENM integer convention)
    vmax = data.max()
    if vmax > 1.0:
        # Integer-stored map — drop the ceiling value and rescale
        data[data == vmax] = 0.0
        vmax = data.max()
        if vmax > 0:
            data = data / vmax
    elif vmax > 0:
        data = data / vmax

    return data.astype(np.float32)


def load_obs(path: str | pathlib.Path) -> np.ndarray:
    """Load a species-observation count raster.

    NODATA is replaced by 0.  No normalisation is applied.

    Parameters
    ----------
    path:
        Path to a ``.tif`` observation count file.

    Returns
    -------
    numpy.ndarray
        2-D float32 array of observation counts.
    """
    return read_raster(path, fill_value=0.0, clip_range=(0, None))


def load_mask(path: str | pathlib.Path) -> np.ndarray:
    """Load a binary region mask raster.

    Any non-zero pixel is treated as *inside the region*.  NODATA → 0.

    Parameters
    ----------
    path:
        Path to a ``.tif`` mask file.

    Returns
    -------
    numpy.ndarray
        2-D float32 array with values in ``{0, 1}``.
    """
    data = read_raster(path, fill_value=0.0, clip_range=(0, None))
    return (data > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Shapefile rasterisation
# ---------------------------------------------------------------------------

def project_shp(
    reference_tif: str | pathlib.Path,
    shapefile: str | pathlib.Path,
) -> np.ndarray:
    """Rasterise a shapefile onto the grid of a reference TIF.

    Requires ``rasterio`` and ``geopandas``.

    Parameters
    ----------
    reference_tif:
        Path to the TIF that defines the target CRS, extent, and resolution.
    shapefile:
        Path to the ``.shp`` file to rasterise.

    Returns
    -------
    numpy.ndarray
        2-D float32 array where each polygon is assigned a unique integer id
        starting from 1.  Pixels outside all polygons are 0.
    """
    import rasterio
    import geopandas as gpd
    from rasterio.features import rasterize

    ref = _require_tif(reference_tif)
    with rasterio.open(ref) as src:
        crs       = src.crs
        transform = src.transform
        width     = src.width
        height    = src.height

    gdf    = gpd.read_file(shapefile).to_crs(crs)
    shapes = [(geom, float(i + 1)) for i, geom in enumerate(gdf.geometry)]
    raster = rasterize(
        shapes,
        out_shape  = (height, width),
        transform  = transform,
        fill       = 0.0,
        dtype      = "float32",
    )
    return raster
