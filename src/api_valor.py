# src/api_valor.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json
import numpy as np

import rasterio
from rasterio.warp import transform as rio_transform
from rasterio.features import rasterize, geometry_mask
from rasterio.windows import Window, from_bounds, transform as win_transform

from shapely.geometry import shape, mapping, box
from shapely.ops import transform as shp_transform
from shapely.validation import explain_validity
from pyproj import Transformer
from pyproj.exceptions import ProjError


# -------- config --------
caminho_geotiff = os.getenv("PATH_GEOTIFF", "./data/raster_html.tif")
# Estratégia do /zonal:
#   "full"   -> lê o raster inteiro e mascara (seu código antigo)
#   "window" -> lê só a janela (bbox do polígono) e rasteriza máscara (muito mais leve)
ZONAL_STRATEGY = os.getenv("ZONAL_STRATEGY", "full").lower()  # "full" | "window"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- modelos --------
class PointQuery(BaseModel):
    lon: float
    lat: float

class PolygonQuery(BaseModel):
    geometry: object  # dict (GeoJSON) ou string JSON


# -------- helpers --------
def _close_rings(coords):
    def close_ring(r):
        if not r:
            return r
        if r[0][0] != r[-1][0] or r[0][1] != r[-1][1]:
            return r + [r[0]]
        return r
    try:
        if isinstance(coords[0][0][0], (int, float)):  # Polygon
            return [close_ring(list(r)) for r in coords]
        fixed = []
        for poly in coords:  # MultiPolygon
            fixed.append([close_ring(list(r)) for r in poly])
        return fixed
    except Exception:
        return coords

def _to_src_crs(geom_wgs84, src):
    dst_crs = src.crs
    if not dst_crs:
        return geom_wgs84
    s = str(dst_crs).upper()
    if s in ("EPSG:4326", "WGS84"):
        return geom_wgs84
    try:
        transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        return shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom_wgs84)
    except ProjError as e:
        raise HTTPException(status_code=400, detail=f"stage=reproject | {e}")

def _normalize_geom(obj):
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            raise HTTPException(status_code=400, detail="stage=normalize | geometry string inválida")
    if not isinstance(obj, dict):
        raise HTTPException(status_code=400, detail="stage=normalize | geometry deve ser objeto JSON")
    gtype = (obj.get("type") or "").upper()
    if gtype not in {"POLYGON", "MULTIPOLYGON"}:
        raise HTTPException(status_code=400, detail="stage=normalize | type deve ser Polygon/MultiPolygon")
    coords = obj.get("coordinates")
    if not coords:
        raise HTTPException(status_code=400, detail="stage=normalize | coordinates ausentes")
    fixed = {"type": "Polygon" if gtype == "POLYGON" else "MultiPolygon",
             "coordinates": _close_rings(coords)}
    try:
        geom = shape(fixed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"stage=validate | GeoJSON inválido: {e}")
    if geom.is_empty:
        raise HTTPException(status_code=400, detail="stage=validate | geometria vazia")
    if explain_validity(geom) != "Valid Geometry":
        geom = geom.buffer(0)
        if explain_validity(geom) != "Valid Geometry" or geom.is_empty:
            raise HTTPException(status_code=400, detail="stage=validate | geometria inválida (buffer(0) falhou)")
    return geom


# -------- saúde/diagnóstico --------
@app.get("/")
def root():
    return {"ok": True, "msg": "API online – veja /docs"}

@app.get("/healthz")
def healthz():
    try:
        with rasterio.open(caminho_geotiff) as src:
            return {"status":"ok","tif":caminho_geotiff,"crs":str(src.crs)}
    except Exception as e:
        return {"status":"degraded","tif":caminho_geotiff,"detail":str(e)}

@app.get("/diag")
def diag():
    """Diagnóstico rápido do raster (útil pra memória/COG)."""
    try:
        with rasterio.open(caminho_geotiff) as src:
            h, w = src.height, src.width
            dtype = src.dtypes[0]
            bps = np.dtype(dtype).itemsize
            raw_mb = (h*w*bps)/(1024*1024)
            return {
                "path": caminho_geotiff,
                "size": [h, w],
                "dtype": dtype,
                "nodata": src.nodata,
                "crs": str(src.crs),
                "tiled": bool(src.profile.get("tiled", False)),
                "compress": src.profile.get("compress"),
                "blockxsize": src.profile.get("blockxsize"),
                "blockysize": src.profile.get("blockysize"),
                "overviews_b1": src.overviews(1),
                "raw_uncompressed_estimate_MB": round(raw_mb, 2)
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------- /point (leve) --------
@app.post("/point")
def get_value(q: PointQuery):
    """
    Valor do raster em um ponto (lon/lat WGS84).
    Lê somente 1 pixel via Window (sem carregar a banda toda).
    """
    try:
        with rasterio.open(caminho_geotiff) as src:
            # reprojeta o clique (lon/lat) para o CRS do raster (já é EPSG:4326 no seu caso)
            xs, ys = rio_transform("EPSG:4326", src.crs if src.crs else "EPSG:4326", [q.lon], [q.lat])
            x, y = xs[0], ys[0]

            # converte para linha/coluna de pixel
            row, col = src.index(x, y)
            if row < 0 or col < 0 or row >= src.height or col >= src.width:
                return {"value": None}

            # LÊ SÓ 1×1 PIXEL (mascarado respeitando NoData)
            win = Window(col_off=col, row_off=row, width=1, height=1)
            pix = src.read(1, window=win, masked=True)

            val = pix[0, 0]
            if np.ma.is_masked(val):
                return {"value": None}

            return {"value": float(val)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"stage=point | {e}")


# -------- /zonal (duas estratégias) --------
def _zonal_full(geom_wgs84):
    """Estratégia antiga: lê banda inteira e mascara."""
    with rasterio.open(caminho_geotiff) as src:
        geom_proj = _to_src_crs(geom_wgs84, src)
        if not geom_proj.intersects(box(*src.bounds)):
            return {"mean": None, "count": 0, "note": "fora do raster"}
        mask = geometry_mask([mapping(geom_proj)],
                             out_shape=(src.height, src.width),
                             transform=src.transform,
                             invert=True)
        arr = src.read(1)
        valid = mask
        if src.nodata is not None:
            valid &= (arr != src.nodata)
        vals = arr[valid]
        if vals.size == 0:
            return {"mean": None, "count": 0, "note": "sem pixels válidos"}
        return {"mean": float(np.nanmean(vals.astype(float))), "count": int(vals.size)}

def _zonal_window(geom_wgs84):
    """Estratégia leve: janela do bbox + rasterize dentro da janela."""
    with rasterio.open(caminho_geotiff) as src:
        geom_proj = _to_src_crs(geom_wgs84, src)
        rb = box(*src.bounds)
        if not geom_proj.intersects(rb):
            return {"mean": None, "count": 0, "note": "fora do raster"}

        minx, miny, maxx, maxy = geom_proj.bounds
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        col0 = max(0, int(np.floor(win.col_off)))
        row0 = max(0, int(np.floor(win.row_off)))
        col1 = min(src.width,  int(np.ceil(win.col_off + win.width)))
        row1 = min(src.height, int(np.ceil(win.row_off + win.height)))
        w = Window(col0, row0, col1-col0, row1-row0)
        if w.width <= 0 or w.height <= 0:
            return {"mean": None, "count": 0, "note": "janela vazia"}

        arr = src.read(1, window=w, masked=False)
        w_transform = win_transform(w, src.transform)
        mask_poly = rasterize(
            [(mapping(geom_proj), 1)],
            out_shape=(int(w.height), int(w.width)),
            transform=w_transform,
            fill=0, dtype="uint8"
        ).astype(bool)
        nodata = src.nodata
        valid = mask_poly
        if nodata is not None:
            valid &= (arr != nodata)
        vals = arr[valid]
        if vals.size == 0:
            return {"mean": None, "count": 0, "note": "sem pixels válidos"}
        return {"mean": float(np.nanmean(vals.astype(float))), "count": int(vals.size)}

@app.post("/zonal")
def zonal_mean(q: PolygonQuery):
    try:
        geom_wgs84 = _normalize_geom(q.geometry)
        # escolhe estratégia
        strategy = ZONAL_STRATEGY
        if strategy == "window":
            try:
                return _zonal_window(geom_wgs84)
            except Exception as e:
                # fallback seguro para a antiga
                fallback = _zonal_full(geom_wgs84)
                fallback["note"] = f"fallback_full_por_erro_window: {e}"
                return fallback
        else:
            return _zonal_full(geom_wgs84)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"stage=zonal | {e}")


# -------- depuração que você já usava --------
@app.post("/echo_geometry")
def echo_geometry(q: PolygonQuery):
    gi = q.geometry
    kind = type(gi).__name__
    sample = gi[:200] + "..." if isinstance(gi, str) and len(gi) > 200 else gi
    return {"received_type": kind, "sample": sample}

@app.post("/zonal_debug")
def zonal_debug(q: PolygonQuery):
    try:
        geom_wgs84 = _normalize_geom(q.geometry)
        with rasterio.open(caminho_geotiff) as src:
            geom_proj = _to_src_crs(geom_wgs84, src)
            bbox = box(*src.bounds)
            minx, miny, maxx, maxy = geom_proj.bounds
            win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            col0 = max(0, int(np.floor(win.col_off)))
            row0 = max(0, int(np.floor(win.row_off)))
            col1 = min(src.width,  int(np.ceil(win.col_off + win.width)))
            row1 = min(src.height, int(np.ceil(win.row_off + win.height)))
            return {
                "validity": explain_validity(geom_wgs84),
                "intersects_raster": bool(geom_proj.intersects(bbox)),
                "raster_crs": str(src.crs),
                "raster_bounds": list(src.bounds),
                "window": {"col0": col0, "row0": row0, "col1": col1, "row1": row1},
                "strategy": ZONAL_STRATEGY
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
