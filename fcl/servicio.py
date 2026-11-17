# -*- coding: utf-8 -*-
"""
Capa de servicio: funciones puras (dict -> dict) que usa la API.
No depende de FastAPI, así se testea sin levantar el servidor.
"""
from functools import lru_cache
from pathlib import Path
import os

import numpy as np

from .surrogate import SurrogateFCL
from .dinamica import Diseno, PAPER, Red, simular, tablas_surrogate, D0_SOBRE_H, DMAX_SOBRE_H
from . import optimizacion as opt

RUTA_MODELO = Path(os.environ.get("FCL_MODELO", Path(__file__).resolve().parents[1] / "modelos" / "surrogate_fcl.npz"))


@lru_cache(maxsize=1)
def surrogate():
    return SurrogateFCL.cargar(RUTA_MODELO)


def _diseno(g):
    return Diseno(ri_int=g["ri_int"], t_int=g["t_int"], gap=g["gap"], t_ext=g["t_ext"], h=g["h"], N=int(g["N"]))


def _red(r):
    return Red(**r) if r else Red()


def _limpiar(x):
    """numpy -> tipos de Python serializables a JSON (NaN -> None)."""
    if isinstance(x, dict):
        return {k: _limpiar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_limpiar(v) for v in x]
    if isinstance(x, np.ndarray):
        return _limpiar(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def info():
    s = surrogate()
    return _limpiar(dict(rangos=s.rangos, rango_N=list(opt.RANGO_N), paper=PAPER.dict(),
                         d0_sobre_h=D0_SOBRE_H, dmax_sobre_h=DMAX_SOBRE_H,
                         modelo=s.meta.get("modelo_acople"), metricas_prueba=s.meta.get("metricas_prueba")))


def inductancias(geometria, d=None, n_puntos=81):
    """L11, L22, L12, L total y F/I² en función de d. Si no se da d, usa 0..2h."""
    s, g = surrogate(), _diseno(geometria)
    d = np.linspace(0, 2 * g.h, n_puntos) if d is None else np.asarray(d, float)
    r = s.inductancias(g.ri_int, g.t_int, g.gap, g.t_ext, g.h, d, g.N, g.N)
    return _limpiar(dict(d_m=d, L11_mH=r["L11"] * 1e3, L22_mH=r["L22"] * 1e3, L12_mH=r["L12"] * 1e3,
                         L_total_mH=r["L_total"] * 1e3, F_sobre_I2_mN_A2=r["F_sobre_I2"] * 1e3,
                         avisos=s.fuera_de_rango(g.ri_int, g.t_int, g.gap, g.t_ext, g.h, d)))


def simular_falla(geometria, red=None, t_fin=0.06, cada=20):
    """Series temporales (submuestreadas) y métricas de la falla para un diseño."""
    s, g = surrogate(), _diseno(geometria)
    r = simular([g], tablas_surrogate(s, [g]), red=_red(red), t_fin=t_fin, guardar_cada=cada, guardar_series=True)
    se = r["series"]
    metricas = dict(pico_kA=r["pico_kA"][0], pico_sin_fcl_kA=r["pico_sin_fcl_kA"],
                    limitacion_pct=r["limitacion_pct"][0], t_operacion_ms=r["t_operacion_ms"][0],
                    L_servicio_mH=r["L_servicio_mH"][0], L_final_mH=r["L_final_mH"][0],
                    caida_tension_pct=r["caida_tension_pct"][0], perdidas_W=r["perdidas_W"][0],
                    masa_kg=r["masa_kg"][0], F_max_kN=r["F_max_kN"][0])
    series = dict(t_ms=se["t"] * 1e3, i_kA=se["i"][:, 0] / 1e3, i_sin_fcl_kA=se["i_sin_fcl"] / 1e3,
                  d_mm=se["d"][:, 0] * 1e3, L_mH=se["L"][:, 0] * 1e3, F_kN=se["F"][:, 0] / 1e3)
    avisos = s.fuera_de_rango(g.ri_int, g.t_int, g.gap, g.t_ext, g.h, [D0_SOBRE_H * g.h, DMAX_SOBRE_H * g.h])
    return _limpiar(dict(diseno=g.dict(), metricas=metricas, series=series, avisos=avisos))


def optimizar(objetivo="masa_kg", restricciones=None, restricciones_rel_paper=None, eje_x=None,
              n=600, generaciones=4, hijos=400, semilla=0):
    """Minimiza `objetivo` con límites absolutos (`restricciones`) y/o relativos al paper
    (`restricciones_rel_paper`: {"pico2_kA": 1.10} = hasta +10 %). Devuelve el mejor diseño,
    el del paper y, si se da `eje_x`, la curva de compromiso (eje_x vs objetivo)."""
    if objetivo not in opt.METRICAS:
        raise ValueError(f"objetivo debe ser una de {opt.METRICAS}")
    s = surrogate()
    ref = opt.evaluar(s, [PAPER])[0]
    lim = dict(restricciones or {})
    for m, factor in (restricciones_rel_paper or {}).items():
        lim[m] = min(lim.get(m, np.inf), ref[m] * factor)
    for m in lim:
        if m not in opt.METRICAS:
            raise ValueError(f"restricción desconocida: {m}")
    r = opt.optimizar(s, objetivo, lim, n=n, generaciones=generaciones, hijos=hijos, semilla=semilla, informar=None)
    curva = []
    if eje_x:
        otros = {m: v for m, v in lim.items() if m != eje_x}
        curva = [r["filas"][i] for i in opt.frente_2d(r["filas"], eje_x, objetivo, otros)]
    return _limpiar(dict(objetivo=objetivo, restricciones=lim, mejor=r["mejor"], paper=ref,
                         historial=r["historial"], evaluados=len(r["filas"]),
                         factibles=sum(f["factible"] for f in r["filas"]), curva=curva))
