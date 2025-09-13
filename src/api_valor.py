from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import rasterio
from rasterio.warp import transform
from pydantic import BaseModel
import os

caminho_geotiff = os.getenv("PATH_GEOTIFF", r"./data/raster_html.tif")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # se quiser, restrinja ao domínio do seu Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PointQuery(BaseModel):
    lon: float
    lat: float

@app.on_event("startup")
def _abrir_raster():
    global src
    src = rasterio.open(caminho_geotiff)

@app.on_event("shutdown")
def _fechar_raster():
    global src
    src.close()

@app.post("/point")
def get_value(q: PointQuery):
    try:
        xs, ys = transform("EPSG:4326", src.crs, [q.lon], [q.lat])
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
