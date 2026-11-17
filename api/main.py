# -*- coding: utf-8 -*-
"""
API REST del limitador de corriente de falla (FCL-ACVSR).

    uvicorn api.main:app --reload          (desde la raíz del repositorio)
    Documentación interactiva: http://localhost:8000/docs
"""
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fcl import servicio
from api import db

METRICA = Literal["caida_tension_pct", "pico1_kA", "pico2_kA", "masa_kg", "perdidas_W"]

app = FastAPI(
    title="FCL-ACVSR — surrogate y diseño de un limitador de corriente de falla",
    version="1.0.0",
    description=("Surrogate (proceso gaussiano) entrenado con ~3.000 simulaciones FEM axisimétricas "
                 "(GMSH + Elmer) de un limitador de núcleo de aire. Calcula inductancias, simula la "
                 "falla y optimiza el diseño."),
)
con = db.conectar()


# ---------------------------------------------------------------- esquemas
class Geometria(BaseModel):
    ri_int: float = Field(0.530, ge=0.30, le=0.80, description="radio interno de la bobina interior [m]")
    t_int: float = Field(0.030, ge=0.015, le=0.060, description="espesor radial de la bobina interior [m]")
    gap: float = Field(0.010, ge=0.005, le=0.030, description="separación radial entre bobinas [m]")
    t_ext: float = Field(0.030, ge=0.015, le=0.060, description="espesor radial de la bobina exterior [m]")
    h: float = Field(0.400, ge=0.20, le=0.60, description="altura de las bobinas [m]")
    N: int = Field(24, ge=1, le=200, description="vueltas de cada bobina")


class Red(BaseModel):
    V_linea_kV: float = Field(20.0, gt=0)
    f: float = Field(50.0, gt=0)
    R: float = Field(0.03, gt=0, description="resistencia de la red hasta la falla [Ω]")
    L: float = Field(2.5e-3, gt=0, description="inductancia de la red hasta la falla [H]")
    I_nominal: float = Field(360.0, gt=0, description="corriente nominal eficaz [A]")
    pico_sin_fcl_kA: float = Field(35.035, gt=0, description="pico de falla sin limitador (define el ángulo)")


class PedidoInductancia(BaseModel):
    geometria: Geometria = Geometria()
    d_m: Optional[List[float]] = Field(None, description="distancias entre centros [m]; por defecto 0..2h")


class PedidoFalla(BaseModel):
    geometria: Geometria = Geometria()
    red: Optional[Red] = None
    t_fin_s: float = Field(0.06, gt=0, le=0.2)


class PedidoOptimizar(BaseModel):
    objetivo: METRICA = "masa_kg"
    restricciones: Dict[METRICA, float] = Field(default_factory=dict, description="límites absolutos")
    restricciones_rel_paper: Dict[METRICA, float] = Field(
        default_factory=lambda: {"caida_tension_pct": 1.0, "pico1_kA": 1.0, "pico2_kA": 1.10, "perdidas_W": 1.0},
        description="límites relativos al diseño del paper (1.10 = hasta +10 %)")
    eje_x: Optional[METRICA] = Field("pico2_kA", description="métrica para la curva de compromiso")
    n: int = Field(600, ge=50, le=5000)
    generaciones: int = Field(4, ge=0, le=20)
    semilla: int = 0


class PedidoGuardar(BaseModel):
    nombre: str = Field(..., min_length=1, max_length=100)
    geometria: Geometria
    metricas: Optional[dict] = None


# ---------------------------------------------------------------- endpoints
@app.get("/salud")
def salud():
    return {"estado": "ok"}


@app.get("/info")
def info():
    """Rangos válidos, diseño del paper y métricas de validación del surrogate."""
    return servicio.info()


@app.post("/inductancia")
def inductancia(p: PedidoInductancia):
    return servicio.inductancias(p.geometria.model_dump(), p.d_m)


@app.post("/simular-falla")
def simular_falla(p: PedidoFalla):
    return servicio.simular_falla(p.geometria.model_dump(), p.red.model_dump() if p.red else None, p.t_fin_s)


@app.post("/optimizar")
def optimizar(p: PedidoOptimizar):
    try:
        return servicio.optimizar(p.objetivo, p.restricciones, p.restricciones_rel_paper, p.eje_x,
                                  n=p.n, generaciones=p.generaciones, semilla=p.semilla)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/disenos")
def listar_disenos(limite: int = 100):
    return db.listar(con, limite)


@app.post("/disenos", status_code=201)
def guardar_diseno(p: PedidoGuardar):
    return db.guardar(con, p.nombre, p.geometria.model_dump(), p.metricas)


@app.get("/disenos/{id_}")
def obtener_diseno(id_: int):
    d = db.obtener(con, id_)
    if not d:
        raise HTTPException(404, "diseño no encontrado")
    return d


@app.delete("/disenos/{id_}", status_code=204)
def borrar_diseno(id_: int):
    if not db.borrar(con, id_):
        raise HTTPException(404, "diseño no encontrado")
