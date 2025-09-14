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
    # Pode ser um objeto GeoJSON (dict) OU uma string JSON
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
        # se a heurística falhar, retorna como veio (deixamos o shapely acusar)
        return coords


# -------- endpoints --------
@app.post("/point")
def get_value(q: PointQuery):
    """
    Valor do raster em um ponto (lon/lat WGS84).
    Usa leitura mascarada: se for NoData, retorna null.
    """
    try:
        with rasterio.open(caminho_geotiff) as src:
            xs, ys = rio_transform("EPSG:4326", src.crs, [q.lon], [q.lat])
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
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/zonal")
def zonal_mean(q: PolygonQuery):
    """
    Média do raster dentro de um polígono (GeoJSON em WGS84).
    Aceita `geometry` como dict ou como string JSON. Fecha anéis e tenta
    corrigir geometrias inválidas (buffer(0)). Retorna mensagens explicativas.
    """
    try:
        # 1) normaliza: se vier string, faz json.loads
        geom_input = q.geometry
        if isinstance(geom_input, str):
            try:
                geom_input = json.loads(geom_input)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="`geometry` deve ser um objeto GeoJSON ou string JSON válida."
                )

        if not isinstance(geom_input, dict):
            raise HTTPException(status_code=400, detail="`geometry` precisa ser um objeto JSON (dict).")

        gtype = (geom_input.get("type") or "").upper()
        if gtype not in {"POLYGON", "MULTIPOLYGON"}:
            raise HTTPException(status_code=400, detail="`geometry.type` deve ser Polygon ou MultiPolygon.")

        coords = geom_input.get("coordinates")
        if not coords:
            raise HTTPException(status_code=400, detail="`geometry.coordinates` ausente.")

        # 2) garante anéis fechados
        geom_fixed = {
            "type": "Polygon" if gtype == "POLYGON" else "MultiPolygon",
            "coordinates": _close_rings(coords)
        }

        # 3) cria shapely e corrige se inválida
        try:
            geom_wgs84 = shape(geom_fixed)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Falha ao ler GeoJSON: {e}")

        if geom_wgs84.is_empty:
            return {"mean": None, "count": 0, "note": "geometria vazia"}

        valid_msg = explain_validity(geom_wgs84)
        if valid_msg != "Valid Geometry":
            # tenta corrigir auto-interseções etc.
            geom_wgs84 = geom_wgs84.buffer(0)
            if geom_wgs84.is_empty:
                raise HTTPException(status_code=400, detail=f"Geometria inválida: {valid_msg}")

        # 4) abre raster, reprojeta, cria máscara e calcula
        with rasterio.open(caminho_geotiff) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            geom_proj = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom_wgs84)
            if geom_proj.is_empty:
                return {"mean": None, "count": 0, "note": "geometria vazia após reprojeção"}

            # checagem rápida: intersecta bounds?
            bbox = box(*src.bounds)
            if not geom_proj.intersects(bbox):
                return {"mean": None, "count": 0, "note": "geometria não intersecta o raster"}

            mask = geometry_mask(
                geometries=[mapping(geom_proj)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True  # True = interior do polígono é válido
            )

            band = src.read(1)
            valid = mask
            if src.nodata is not None:
                valid &= (band != src.nodata)

            vals = band[valid]
            if vals.size == 0:
                return {"mean": None, "count": 0, "note": "sem pixels válidos (provável NoData)"}

            return {"mean": float(np.nanmean(vals.astype(float))), "count": int(vals.size)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    try:
        geom_input = q.geometry
        if isinstance(geom_input, str):
            geom_input = json.loads(geom_input)

        gtype = (geom_input.get("type") or "").upper()
        coords = geom_input.get("coordinates")

        fixed = {"type": "Polygon" if gtype == "POLYGON" else "MultiPolygon",
                 "coordinates": _close_rings(coords)}
        geom = shape(fixed)
        info = {
            "received_type": gtype,
            "validity": explain_validity(geom),
            "is_empty": geom.is_empty
        }

        with rasterio.open(caminho_geotiff) as src:
            bbox = box(*src.bounds)
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            geom_p = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)
            info["intersects_raster"] = bool(geom_p.intersects(bbox))
            info["raster_crs"] = str(src.crs)
            info["raster_bounds"] = list(src.bounds)
        return info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
