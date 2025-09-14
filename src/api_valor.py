# src/api_valor.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json
from typing import Any, Tuple

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
from rasterio.features import geometry_mask

from shapely.geometry import shape, mapping, box
from shapely.ops import transform as shp_transform
from shapely.validation import explain_validity
from pyproj import Transformer

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
    # Aceita objeto GeoJSON (dict) OU string JSON
    geometry: object


# -------- helpers --------
def _as_geojson_dict(obj: Any) -> dict:
    """Garante que geometry seja um dict (se vier string, faz json.loads)."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="`geometry` deve ser objeto GeoJSON ou string JSON válida.",
            )
    if not isinstance(obj, dict):
        raise HTTPException(status_code=400, detail="`geometry` precisa ser um objeto JSON (dict).")
    return obj


def _close_rings(coords):
    """
    Fecha anéis de Polygon/MultiPolygon no padrão GeoJSON.
    Mantém o mesmo *shape* (lista de anéis ou lista de polígonos).
    """
    def close_ring(r):
        if not r:
            return r
        if r[0][0] != r[-1][0] or r[0][1] != r[-1][1]:
            return list(r) + [list(r[0])]
        return list(r)

    try:
        # Polygon: [ [x,y], ... ]  -> embrulhar como [anel]
        if len(coords) > 0 and isinstance(coords[0], (list, tuple)) and \
           len(coords[0]) == 2 and all(isinstance(v, (int, float)) for v in coords[0]):
            coords = [coords]

        # Se coords[0][0] é par (x,y) -> lista de anéis (Polygon)
        if isinstance(coords[0][0], (list, tuple)) and len(coords[0][0]) == 2 \
           and all(isinstance(v, (int, float)) for v in coords[0][0]):
            return [close_ring(r) for r in coords]

        # Caso MultiPolygon: lista de polígonos, cada um com lista de anéis
        fixed = []
        for poly in coords:
            # também tolera polígono que chegou como lista de pares (anel único)
            if poly and isinstance(poly[0], (list, tuple)) and len(poly[0]) == 2 \
               and all(isinstance(v, (int, float)) for v in poly[0]):
                poly = [poly]
            fixed.append([close_ring(r) for r in poly])
        return fixed
    except Exception:
        # Se não deu pra inferir, devolve como veio; shapely vai acusar.
        return coords


def _normalize_poly(geom: dict) -> Tuple[str, list]:
    """
    Garante type ∈ {Polygon, MultiPolygon} e normaliza níveis de colchetes.
    Retorna (gtype, coordinates) já com anéis fechados.
    """
    gtype = (geom.get("type") or "").upper()
    if gtype not in {"POLYGON", "MULTIPOLYGON"}:
        raise HTTPException(status_code=400, detail="`geometry.type` deve ser 'Polygon' ou 'MultiPolygon'.")

    coords = geom.get("coordinates", None)
    if coords is None:
        raise HTTPException(status_code=400, detail="`geometry.coordinates` ausente.")

    # Heurísticas de nível:
    #  - Polygon: coords deve ser [anel_exterior, holes...]; se vier só o anel => embrulhar
    if gtype == "POLYGON":
        if len(coords) > 0 and isinstance(coords[0], (list, tuple)) and \
           len(coords[0]) == 2 and all(isinstance(v, (int, float)) for v in coords[0]):
            coords = [coords]
    else:
        # MultiPolygon: [[aneles...], [aneles...], ...]
        # Se vier um anel direto, embrulha para [[anel]]
        if len(coords) > 0 and isinstance(coords[0], (list, tuple)) and \
           len(coords[0]) == 2 and all(isinstance(v, (int, float)) for v in coords[0]):
            coords = [[coords]]
        # Se vier [ [x,y], ... ] (um polígono), vira [[ [x,y], ... ]]
        elif len(coords) > 0 and isinstance(coords[0], (list, tuple)) and \
             len(coords[0]) > 0 and isinstance(coords[0][0], (list, tuple)) and \
             len(coords[0][0]) == 2 and all(isinstance(v, (int, float)) for v in coords[0][0]):
            coords = [coords]

    coords = _close_rings(coords)
    return gtype, coords


# -------- endpoints --------
@app.post("/point")
def get_value(q: PointQuery):
    """
    Valor do raster em um ponto (lon/lat em WGS84).
    Usa leitura mascarada: se for NoData, retorna null.
    """
    try:
        with rasterio.open(caminho_geotiff) as src:
            xs, ys = rio_transform("EPSG:4326", src.crs, [q.lon], [q.lat])
            x, y = xs[0], ys[0]
            row, col = src.index(x, y)

            if row < 0 or col < 0 or row >= src.height or col >= src.width:
                return {"value": None}

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
    Média do raster dentro de um polígono (GeoJSON WGS84).
    Aceita dict ou string JSON; normaliza níveis, fecha anéis e corrige geometrias.
    """
    stage = "parse"
    try:
        geom_dict = _as_geojson_dict(q.geometry)

        stage = "normalize"
        gtype, coords = _normalize_poly(geom_dict)
        fixed = {"type": "Polygon" if gtype == "POLYGON" else "MultiPolygon", "coordinates": coords}

        stage = "shapely-load"
        geom_wgs84 = shape(fixed)
        if explain_validity(geom_wgs84) != "Valid Geometry":
            geom_wgs84 = geom_wgs84.buffer(0)
        if geom_wgs84.is_empty:
            return {"mean": None, "count": 0, "note": "geometria vazia após correção"}

        with rasterio.open(caminho_geotiff) as src:
            stage = "reproject"
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            geom_proj = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom_wgs84)

            stage = "bounds-check"
            if not geom_proj.intersects(box(*src.bounds)):
                return {"mean": None, "count": 0, "note": "geometria não intersecta o raster"}

            stage = "mask"
            mask = geometry_mask(
                geometries=[mapping(geom_proj)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True
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
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=f"{e.detail} (stage={stage})")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)} (stage={stage})")


@app.post("/zonal_debug")
def zonal_debug(q: PolygonQuery):
    """
    Diagnóstico detalhado da geometria + interseção com o raster.
    """
    stage = "parse"
    try:
        geom_dict = _as_geojson_dict(q.geometry)

        stage = "normalize"
        gtype, coords = _normalize_poly(geom_dict)
        fixed = {"type": "Polygon" if gtype == "POLYGON" else "MultiPolygon", "coordinates": coords}

        stage = "shapely-load"
        geom_wgs84 = shape(fixed)
        before_valid = explain_validity(geom_wgs84)
        if before_valid != "Valid Geometry":
            geom_wgs84 = geom_wgs84.buffer(0)
        after_valid = explain_validity(geom_wgs84)

        with rasterio.open(caminho_geotiff) as src:
            stage = "reproject"
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            geom_proj = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom_wgs84)

            bbox = box(*src.bounds)
            return {
                "received_type": gtype,
                "validity_before": before_valid,
                "validity_after": after_valid,
                "is_empty_after": geom_wgs84.is_empty,
                "intersects_raster": bool(geom_proj.intersects(bbox)),
                "raster_crs": str(src.crs),
                "raster_bounds": list(src.bounds),
            }
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=f"{e.detail} (stage={stage})")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)} (stage={stage})")


# -------- diagnósticos/saúde --------
@app.get("/")
def root():
    return {"ok": True, "msg": "API online – veja /docs"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/echo_geometry")
def echo_geometry(q: PolygonQuery):
    """Mostra o tipo recebido em `geometry` e um trecho (para debug rápido)."""
    gi = q.geometry
    kind = type(gi).__name__
    sample = gi[:120] + "..." if isinstance(gi, str) and len(gi) > 120 else gi
    return {"received_type": kind, "sample": sample}
