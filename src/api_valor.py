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
    Valor do raster em um ponto (lon/lat WGS84).
    Usa leitura mascarada: se for NoData, retorna null.
    """
    try:
        with rasterio.open(caminho_geotiff) as src:
            xs, ys = rio_transform("EPSG:4326", src.crs if src.crs else "EPSG:4326", [q.lon], [q.lat])
            x, y = xs[0], ys[0]
            row, col = src.index(x, y)

            # fora dos limites
            if row < 0 or col < 0 or row >= src.height or col >= src.width:
                return {"value": None}

            # leitura mascarada (respeita NoData)
            band = src.read(1, masked=True)
            val = band[row, col]
            if np.ma.is_masked(val):
                return {"value": None}
            return {"value": float(val)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"stage=point | {e}")


@app.post("/zonal")
def zonal_mean(q: PolygonQuery):
    """
    Média do raster dentro de um polígono (GeoJSON em WGS84).
    Aceita `geometry` como dict ou como string JSON. Fecha anéis e tenta
    corrigir geometrias inválidas (buffer(0)). Retorna mensagens explicativas.
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

        # 4) abre raster, reprojeta, mascara e calcula
        with rasterio.open(caminho_geotiff) as src:
            stage = "reproject"
            geom_proj = _to_src_crs(geom_wgs84, src)

            stage = "bounds-check"
            if not geom_proj.intersects(box(*src.bounds)):
                return {"mean": None, "count": 0, "note": "geometria não intersecta o raster"}

            stage = "mask"
            mask = geometry_mask(
                geometries=[mapping(geom_proj)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True  # True = interior do polígono é válido
            )

            stage = "compute"
            band = src.read(1)
            valid = mask
            if src.nodata is not None:
                valid &= (band != src.nodata)

            vals = band[valid]
            if vals.size == 0:
                return {"mean": None, "count": 0, "note": "sem pixels válidos (NoData/fora do raster)"}

            return {"mean": float(np.nanmean(vals.astype(float))), "count": int(vals.size)}

    except HTTPException:
        raise
    except Exception as e:
        # devolve stage para facilitar debug
        raise HTTPException(status_code=400, detail=f"{stage=} | {e}")


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
    sample = gi[:200] + "..." if isinstance(gi, str) and len(gi) > 200 else gi
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
