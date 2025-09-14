# src/api_valor.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json

import rasterio
from rasterio.warp import transform as rio_transform
from rasterio.features import geometry_mask

from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
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
    Aceita `geometry` como dict ou como string JSON.
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

        # 2) cria geometria e valida
        geom_wgs84 = shape(geom_input)
        if geom_wgs84.is_empty:
            return {"mean": None, "count": 0}

        with rasterio.open(caminho_geotiff) as src:
            # reprojeta polígono para o CRS do raster
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            geom_proj = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom_wgs84)
            if geom_proj.is_empty:
                return {"mean": None, "count": 0}

            # rasteriza máscara do polígono
            mask = geometry_mask(
                geometries=[mapping(geom_proj)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True  # True = interior do polígono é válido
            )

            band = src.read(1)
            valid = mask.copy()
            if src.nodata is not None:
                valid &= (band != src.nodata)

            vals = band[valid]
            if vals.size == 0:
                return {"mean": None, "count": 0}

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
