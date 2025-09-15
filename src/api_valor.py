# src/api_valor.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json

import rasterio
from rasterio.warp import transform as rio_transform
from rasterio.features import geometry_mask

from shapely.geometry import shape, mapping, box
from shapely.ops import transform as shp_transform
from shapely.validation import explain_validity
from pyproj import Transformer
from pyproj.exceptions import ProjError
import numpy as np


# -------- config --------
caminho_geotiff = os.getenv("PATH_GEOTIFF", "./data/raster_html_novo.tif")

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
    # pode ser objeto GeoJSON (dict) OU uma string JSON
    geometry: object


# -------- helpers --------
def _close_rings(coords):
    """
    Fecha anéis de Polygon/MultiPolygon no padrão GeoJSON.
    Aceita tanto [[...]] (Polygon) quanto [[[...]], ...] (MultiPolygon).
    """
    def close_ring(r):
        if len(r) == 0:
            return r
        if r[0][0] != r[-1][0] or r[0][1] != r[-1][1]:
            return r + [r[0]]
        return r

    # Heurística: se coords[0][0] é número -> anel simples; senão, anéis aninhados
    try:
        if isinstance(coords[0][0][0], (int, float)):
            # Polygon: [ [ [x,y], ... ] , [anel_interno] , ... ]
            return [close_ring(list(r)) for r in coords]
        else:
            # MultiPolygon: [ [ [anel_ext, anel_int... ] ] , [ ... ] ]
            fixed = []
            for poly in coords:
                fixed.append([close_ring(list(r)) for r in poly])
            return fixed
    except Exception:
        # se a heurística falhar, retorna como veio (deixa shapely acusar)
        return coords


def _to_src_crs(geom_wgs84, src):
    """
    Reprojeta a geometria WGS84 (EPSG:4326) para o CRS do raster.
    Se o raster não tiver CRS ou já for WGS84, retorna a geometria original.
    """
    dst_crs = src.crs
    if not dst_crs:
        # sem CRS no raster: assume que pixels já estão em WGS84
        return geom_wgs84
    dst = str(dst_crs).upper()
    if dst in ("EPSG:4326", "WGS84"):
        return geom_wgs84

    try:
        transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        return shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom_wgs84)
    except ProjError as e:
        raise HTTPException(
            status_code=400,
            detail=f"stage=reproject | falha ao construir transformador (CRS do raster = {dst_crs}): {e}"
        )


# -------- endpoints --------
@app.post("/point")
def get_value(q: PointQuery):
    """
    Lê APENAS 1 pixel via window 1x1. Se NoData, retorna null.
    """
    try:
        with rasterio.open(caminho_geotiff) as src:
            xs, ys = rio_transform("EPSG:4326", src.crs, [q.lon], [q.lat])
            x, y = xs[0], ys[0]
            row, col = src.index(x, y)

            if row < 0 or col < 0 or row >= src.height or col >= src.width:
                return {"value": None}

            # janela 1x1
            w = Window(col_off=col, row_off=row, width=1, height=1)
            band = src.read(1, window=w, masked=True)
            val = band[0, 0]
            if np.ma.is_masked(val):
                return {"value": None}
            return {"value": float(val)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"point: {e}")


@app.post("/zonal")
def zonal_mean(q: PolygonQuery):
    """
    Média do raster numa geometria GeoJSON (WGS84).
    Lê APENAS a janela do bounding box do polígono e rasteriza máscara nesse recorte.
    """
    try:
        # normaliza geometry
        geom_input = q.geometry
        if isinstance(geom_input, str):
            geom_input = json.loads(geom_input)
        if not isinstance(geom_input, dict):
            raise HTTPException(status_code=400, detail="`geometry` precisa ser dict ou string JSON.")

        g = shape(geom_input)
        if g.is_empty:
            return {"mean": None, "count": 0, "note": "geometria vazia"}

        with rasterio.open(caminho_geotiff) as src:
            # reprojeta p/ CRS do raster
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            g_proj = shp_transform(lambda x,y,z=None: transformer.transform(x,y), g)
            if g_proj.is_empty:
                return {"mean": None, "count": 0, "note": "vazia após reprojeção"}

            # se não intersecta, retorna cedo
            rb = shp_box(*src.bounds)
            if not g_proj.intersects(rb):
                return {"mean": None, "count": 0, "note": "não intersecta o raster"}

            # calcula janela mínima do bbox do polígono
            minx, miny, maxx, maxy = g_proj.bounds
            win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            # clampa a janela pra ficar dentro do raster
            col0 = max(0, int(np.floor(win.col_off)))
            row0 = max(0, int(np.floor(win.row_off)))
            col1 = min(src.width,  int(np.ceil(win.col_off + win.width)))
            row1 = min(src.height, int(np.ceil(win.row_off + win.height)))
            w = Window(col0, row0, col1-col0, row1-row0)
            if w.width <= 0 or w.height <= 0:
                return {"mean": None, "count": 0, "note": "janela vazia"}

            # lê somente o recorte
            arr = src.read(1, window=w, masked=False)

            # transform do recorte
            # A = T(col, row) = (x, y)
            # Transform de janela: desloca a origem
            a, b, c, d, e, f = src.transform
            win_transform = rasterio.Affine(a, b, c + a*col0 + b*row0,
                                            d, e, f + d*col0 + e*row0)

            # rasteriza a geometria na grade da janela
            mask_poly = rasterize(
                [(mapping(g_proj), 1)],
                out_shape=(int(w.height), int(w.width)),
                transform=win_transform,
                fill=0,
                dtype="uint8"
            ).astype(bool)

            # aplica NoData
            nodata = src.nodata
            valid = mask_poly
            if nodata is not None:
                valid &= (arr != nodata)

            vals = arr[valid]
            if vals.size == 0:
                return {"mean": None, "count": 0, "note": "sem pixels válidos"}

            return {"mean": float(np.nanmean(vals.astype(float))), "count": int(vals.size)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"zonal: {e}")


# -------- diagnósticos/saúde --------
@app.get("/")
def root():
    return {"ok": True, "msg": "API online – veja /docs"}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/echo_geometry")
def echo_geometry(q: PolygonQuery):
    """
    Auxilia a depurar o que o cliente está enviando em `geometry`.
    """
    gi = q.geometry
    kind = type(gi).__name__
    sample = gi[:120] + "..." if isinstance(gi, str) and len(gi) > 120 else gi
    return {"received_type": kind, "sample": sample}


@app.post("/zonal_debug")
def zonal_debug(q: PolygonQuery):
    """
    Mostra como a API interpretou a geometria e checks básicos.
    """
    stage = "start"
    try:
        geom_input = q.geometry
        if isinstance(geom_input, str):
            stage = "normalize"
            geom_input = json.loads(geom_input)

        gtype = (geom_input.get("type") or "").upper()
        coords = geom_input.get("coordinates")

        stage = "close_rings"
        fixed = {"type": "Polygon" if gtype == "POLYGON" else "MultiPolygon",
                 "coordinates": _close_rings(coords)}
        geom_wgs84 = shape(fixed)

        info = {
            "received_type": gtype,
            "validity_before": explain_validity(geom_wgs84),
            "is_empty": geom_wgs84.is_empty
        }

        if info["validity_before"] != "Valid Geometry":
            geom_wgs84 = geom_wgs84.buffer(0)
            info["validity_after_buffer0"] = explain_validity(geom_wgs84)
            info["is_empty_after_buffer0"] = geom_wgs84.is_empty

        with rasterio.open(caminho_geotiff) as src:
            stage = "reproject"
            geom_proj = _to_src_crs(geom_wgs84, src)

            bbox = box(*src.bounds)
            info.update({
                "intersects_raster": bool(geom_proj.intersects(bbox)),
                "raster_crs": str(src.crs),
                "raster_bounds": list(src.bounds),
            })
        return info

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{stage=} | {e}")
