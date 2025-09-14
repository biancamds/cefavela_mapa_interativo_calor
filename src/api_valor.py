# src/api_valor.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

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
    geometry: dict  # GeoJSON geometry (Polygon/MultiPolygon) em EPSG:4326

# -------- endpoints --------
@app.post("/point")
def get_value(q: PointQuery):
    """Valor do raster em um ponto (lat/lon WGS84)."""
    try:
        # Abre o TIFF em cada requisição (evita travas do GDAL/libtiff)
        with rasterio.open(caminho_geotiff) as src:
            xs, ys = rio_transform("EPSG:4326", src.crs, [q.lon], [q.lat])
            x, y = xs[0], ys[0]
            row, col = src.index(x, y)
            if row < 0 or col < 0 or row >= src.height or col >= src.width:
                return {"value": None}
            val = src.read(1)[row, col]
            if src.nodata is not None and val == src.nodata:
                return {"value": None}
            return {"value": float(val)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/zonal")
def zonal_mean(q: PolygonQuery):
    """Média do raster dentro de um polígono desenhado (GeoJSON em WGS84)."""
    try:
        geom_wgs84 = shape(q.geometry)
        if geom_wgs84.is_empty:
            return {"mean": None, "count": 0}

        with rasterio.open(caminho_geotiff) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            geom_proj = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom_wgs84)
            if geom_proj.is_empty:
                return {"mean": None, "count": 0}

            mask = geometry_mask(
                geometries=[mapping(geom_proj)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True  # True = interior do polígono é válido
            )

            arr = src.read(1)
            valid = mask.copy()
            if src.nodata is not None:
                valid &= (arr != src.nodata)

            vals = arr[valid]
            if vals.size == 0:
                return {"mean": None, "count": 0}

            return {"mean": float(np.nanmean(vals.astype(float))), "count": int(vals.size)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
