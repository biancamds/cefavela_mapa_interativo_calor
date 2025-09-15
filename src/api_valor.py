# src/api_valor.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import numpy as np

import rasterio
from rasterio.warp import transform as rio_transform
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds, transform as win_transform

from shapely.geometry import shape, mapping, box
from shapely.ops import transform as shp_transform
from shapely.validation import explain_validity
from pyproj import Transformer
from pyproj.exceptions import ProjError


# -------- config --------
# Se quiser apontar para outro .tif no Render, defina a env var PATH_GEOTIFF.
caminho_geotiff = os.getenv("PATH_GEOTIFF", "./data/raster_html.tif")

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

    try:
        # Polygon: [ [ [x,y], ... ] , [anel_interno] , ... ]
        if isinstance(coords[0][0][0], (int, float)):
            return [close_ring(list(r)) for r in coords]
        # MultiPolygon: [ [ [anel_ext, anel_int... ] ] , [ ... ] ]
        fixed = []
        for poly in coords:
            fixed.append([close_ring(list(r)) for r in poly])
        return fixed
    except Exception:
        return coords


def _to_src_crs(geom_wgs84, src):
    """
    Reprojeta a geometria WGS84 (EPSG:4326) para o CRS do raster.
    Se o raster não tiver CRS ou já for WGS84, retorna a geometria original.
    """
    dst_crs = src.crs
    if not dst_crs:
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


# -------- endpoints básicos --------
@app.get("/")
def root():
    return {"ok": True, "msg": "API online – veja /docs"}


@app.get("/healthz")
def healthz():
    # também verifica se o TIFF existe/abre
    try:
        with rasterio.open(caminho_geotiff):
            pass
        return {"status": "ok", "tif": caminho_geotiff}
    except Exception as e:
        return {"status": "degraded", "tif": caminho_geotiff, "detail": str(e)}


# -------- /point: leitura super-eficiente (janela 1x1) --------
@app.post("/point")
def get_value(q: PointQuery):
    """
    Valor do raster em um ponto (lon/lat WGS84).
    Usa leitura 1×1 via Window e respeita NoData.
    """
    try:
        with rasterio.open(caminho_geotiff) as src:
            xs, ys = rio_transform("EPSG:4326", src.crs if src.crs else "EPSG:4326", [q.lon], [q.lat])
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
        raise HTTPException(status_code=400, detail=f"stage=point | {e}")


# -------- /zonal: leitura por janela (bounding box) + rasterize --------
@app.post("/zonal")
def zonal_mean(q: PolygonQuery):
    """
    Média do raster dentro de um polígono (GeoJSON em WGS84).
    - Normaliza/fecha anéis/valida geometria.
    - Reprojeta para o CRS do raster.
    - Lê apenas a janela do bounding box do polígono.
    - Rasteriza a máscara dentro dessa janela e calcula a média (ignora NoData).
    """
    stage = "start"
    try:
        # 1) normaliza: se vier string, faz json.loads
        stage = "normalize"
        geom_input = q.geometry
        if isinstance(geom_input, str):
            try:
                geom_input = json.loads(geom_input)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="stage=normalize | `geometry` deve ser um objeto GeoJSON ou string JSON válida."
                )

        if not isinstance(geom_input, dict):
            raise HTTPException(status_code=400, detail="stage=normalize | `geometry` precisa ser um objeto JSON (dict).")

        gtype = (geom_input.get("type") or "").upper()
        if gtype not in {"POLYGON", "MULTIPOLYGON"}:
            raise HTTPException(status_code=400, detail="stage=normalize | `geometry.type` deve ser Polygon ou MultiPolygon.")

        coords = geom_input.get("coordinates")
        if not coords:
            raise HTTPException(status_code=400, detail="stage=normalize | `geometry.coordinates` ausente.")

        # 2) garante anéis fechados
        stage = "close_rings"
        geom_fixed = {
            "type": "Polygon" if gtype == "POLYGON" else "MultiPolygon",
            "coordinates": _close_rings(coords)
        }

        # 3) cria shapely e corrige se inválida
        stage = "validate"
        try:
            geom_wgs84 = shape(geom_fixed)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"stage=validate | falha ao ler GeoJSON: {e}")

        if geom_wgs84.is_empty:
            return {"mean": None, "count": 0, "note": "geometria vazia"}

        before_valid = explain_validity(geom_wgs84)
        if before_valid != "Valid Geometry":
            geom_wgs84 = geom_wgs84.buffer(0)
            after_valid = explain_validity(geom_wgs84)
            if geom_wgs84.is_empty or after_valid != "Valid Geometry":
                raise HTTPException(
                    status_code=400,
                    detail=f"stage=validate | geometria inválida: {before_valid} / após buffer(0): {after_valid}"
                )

        # 4) abre raster e processa usando janela mínima
        with rasterio.open(caminho_geotiff) as src:
            # reprojeta pro CRS do raster
            stage = "reproject"
            geom_proj = _to_src_crs(geom_wgs84, src)

            # checagem de interseção com o raster
            stage = "bounds-check"
            rb = box(*src.bounds)
            if not geom_proj.intersects(rb):
                return {"mean": None, "count": 0, "note": "geometria não intersecta o raster"}

            # calcula a janela mínima do bbox da geometria
            stage = "window"
            minx, miny, maxx, maxy = geom_proj.bounds
            win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)

            # clamp para garantir que a janela fique dentro do raster
            col0 = max(0, int(np.floor(win.col_off)))
            row0 = max(0, int(np.floor(win.row_off)))
            col1 = min(src.width,  int(np.ceil(win.col_off + win.width)))
            row1 = min(src.height, int(np.ceil(win.row_off + win.height)))
            w = Window(col0, row0, col1 - col0, row1 - row0)

            if w.width <= 0 or w.height <= 0:
                return {"mean": None, "count": 0, "note": "janela vazia"}

            # lê apenas o recorte
            stage = "read"
            arr = src.read(1, window=w, masked=False)

            # transform da janela (para rasterizar a máscara na grade do recorte)
            stage = "window-transform"
            w_transform = win_transform(w, src.transform)

            # rasteriza a geometria na grade da janela
            stage = "rasterize"
            mask_poly = rasterize(
                [(mapping(geom_proj), 1)],
                out_shape=(int(w.height), int(w.width)),
                transform=w_transform,
                fill=0,
                dtype="uint8"
            ).astype(bool)

            # aplica NoData
            stage = "compute"
            nodata = src.nodata
            valid = mask_poly
            if nodata is not None:
                valid &= (arr != nodata)

            vals = arr[valid]
            if vals.size == 0:
                return {"mean": None, "count": 0, "note": "sem pixels válidos (NoData/fora do raster)"}

            return {"mean": float(np.nanmean(vals.astype(float))), "count": int(vals.size)}

    except HTTPException:
        raise
    except Exception as e:
        # devolve stage para facilitar debug
        raise HTTPException(status_code=400, detail=f"{stage=} | {e}")


# -------- diagnósticos --------
@app.post("/echo_geometry")
def echo_geometry(q: PolygonQuery):
    gi = q.geometry
    kind = type(gi).__name__
    sample = gi[:200] + "..." if isinstance(gi, str) and len(gi) > 200 else gi
    return {"received_type": kind, "sample": sample}


@app.post("/zonal_debug")
def zonal_debug(q: PolygonQuery):
    stage = "start"
    try:
        geom_input = q.geometry
        if isinstance(geom_input, str):
            stage = "normalize"
            geom_input = json.loads(geom_input)

        gtype = (geom_input.get("type") or "").upper()
        coords = geom_input.get("coordinates")

        stage = "close_rings"
        fixed = {
            "type": "Polygon" if gtype == "POLYGON" else "MultiPolygon",
            "coordinates": _close_rings(coords)
        }
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

            # também reporta a janela calculada
            minx, miny, maxx, maxy = geom_proj.bounds
            win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            col0 = max(0, int(np.floor(win.col_off)))
            row0 = max(0, int(np.floor(win.row_off)))
            col1 = min(src.width,  int(np.ceil(win.col_off + win.width)))
            row1 = min(src.height, int(np.ceil(win.row_off + win.height)))
            info["window"] = {"col0": col0, "row0": row0, "col1": col1, "row1": row1}

        return info

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{stage=} | {e}")
