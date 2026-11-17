# -*- coding: utf-8 -*-
"""
Diseño inverso del limitador: optimización con restricciones y curvas de compromiso (Pareto).

Métricas de un diseño (todas a MINIMIZAR):
  caida_tension_pct  ΔV% = ω L(d0) I_n / V_fase   (lo que "cuesta" el limitador sin falla)
  pico1_kA           pico de falla en el primer período (esfuerzo electrodinámico y térmico)
  pico2_kA           pico en el segundo período (lo que ve el interruptor al abrir)
  masa_kg            masa de las bobinas (material y estructura)
  perdidas_W         pérdidas Joule en servicio normal (costo de operación)

Uso típico: "minimizar la masa con ΔV, picos y pérdidas no peores que el diseño del paper".
Variables: ri_int, t_int, gap, t_ext, h (dentro del rango del surrogate) y N (vueltas).
Método: hipercubo latino + algoritmo evolutivo simple (mutación gaussiana de los mejores,
penalización por restricciones) + simulación vectorizada de la falla. Reproducible (semilla fija).

Resultado con los 5 objetivos a la vez: el diseño del paper está prácticamente sobre el frente
de Pareto (lo mejor que se encuentra es ~0,2 % mejor en todos). Las ganancias grandes aparecen
al ceder en un objetivo: ver frente_2d().
"""
import numpy as np
from scipy.stats import qmc

from .dinamica import Diseno, PAPER, tablas_surrogate, simular

RANGO_N = (10, 40)
METRICAS = ("caida_tension_pct", "pico1_kA", "pico2_kA", "masa_kg", "perdidas_W")
CLAVES = ("ri_int", "t_int", "gap", "t_ext", "h")


def rangos_de(sur):
    return {k: tuple(v) for k, v in sur.rangos.items() if k != "d_sobre_h"}


def muestrear(rangos, n, semilla=0, rango_N=RANGO_N):
    U = qmc.LatinHypercube(d=6, seed=semilla).random(n)
    lo = np.array([rangos[k][0] for k in CLAVES])
    hi = np.array([rangos[k][1] for k in CLAVES])
    X = lo + U[:, :5] * (hi - lo)
    N = np.round(rango_N[0] + U[:, 5] * (rango_N[1] - rango_N[0])).astype(int)
    return [Diseno(*np.round(x, 4), N=int(k)) for x, k in zip(X, N)]


def mutar(base, rangos, n, rng, escala=0.05, rango_N=RANGO_N):
    """Hijos por mutación gaussiana (relativa al rango) de diseños base, recortados al rango."""
    lo = np.array([rangos[k][0] for k in CLAVES])
    hi = np.array([rangos[k][1] for k in CLAVES])
    P = np.array([[getattr(d, k) for k in CLAVES] for d in base])
    Np = np.array([d.N for d in base])
    idx = rng.integers(0, len(base), n)
    X = np.clip(P[idx] + rng.normal(0, escala, (n, 5)) * (hi - lo), lo, hi)
    N = np.clip(Np[idx] + rng.integers(-2, 3, n), *rango_N)
    return [Diseno(*np.round(x, 4), N=int(k)) for x, k in zip(X, N)]


def a_diseno(f):
    return Diseno(*(float(f[k]) for k in CLAVES), N=int(f["N"]))


def evaluar(sur, ds, red=None, t_fin=0.04, lote=1000):
    """Métricas de la falla para una lista de diseños (por lotes para acotar memoria)."""
    filas = []
    for k in range(0, len(ds), lote):
        sub = ds[k:k + lote]
        r = simular(sub, tablas_surrogate(sur, sub), red=red, t_fin=t_fin, guardar_series=False)
        for j, d in enumerate(sub):
            filas.append(dict(**d.dict(),
                              caida_tension_pct=float(r["caida_tension_pct"][j]),
                              pico1_kA=float(r["pico_kA"][j, 0]),
                              pico2_kA=float(r["pico_kA"][j, 1]),
                              limitacion1_pct=float(r["limitacion_pct"][j, 0]),
                              limitacion2_pct=float(r["limitacion_pct"][j, 1]),
                              t_operacion_ms=float(r["t_operacion_ms"][j]),
                              L_servicio_mH=float(r["L_servicio_mH"][j]),
                              L_final_mH=float(r["L_final_mH"][j]),
                              masa_kg=float(r["masa_kg"][j]),
                              perdidas_W=float(r["perdidas_W"][j]),
                              F_max_kN=float(r["F_max_kN"][j])))
    return filas


def violacion(f, restricciones):
    """Suma de excesos relativos sobre los límites (0 = cumple todo)."""
    return sum(max(0.0, f[m] / lim - 1) for m, lim in restricciones.items())


def optimizar(sur, objetivo="masa_kg", restricciones=None, n=1500, generaciones=6, hijos=800,
              padres=60, semilla=0, red=None, rango_N=RANGO_N, informar=print):
    """Minimiza `objetivo` sujeto a métrica <= límite para cada entrada de `restricciones`.

    Devuelve dict con: mejor (fila factible de menor objetivo, o None), ref (diseño del paper),
    filas (todas las evaluaciones, con 'factible') e historial (mejor objetivo por generación).
    """
    restricciones = dict(restricciones or {})
    rangos = rangos_de(sur)
    rng = np.random.default_rng(semilla)
    filas = evaluar(sur, [PAPER] + muestrear(rangos, n, semilla, rango_N), red=red)
    ref, filas = filas[0], filas[1:]
    escala = abs(ref[objetivo]) or 1.0

    def puntaje(f):
        return f[objetivo] / escala + 100 * violacion(f, restricciones)

    historial = []
    for g in range(generaciones + 1):
        s = np.array([puntaje(f) for f in filas])
        orden = np.argsort(s)
        fact = [f for f in filas if violacion(f, restricciones) == 0]
        mejor = min(fact, key=lambda f: f[objetivo]) if fact else None
        historial.append(None if mejor is None else mejor[objetivo])
        if informar:
            txt = "sin diseños factibles todavía" if mejor is None else f"mejor {objetivo} = {mejor[objetivo]:.4g}"
            informar(f"   generación {g}: {len(filas)} diseños evaluados, {len(fact)} factibles, {txt}")
        if g == generaciones:
            break
        base = [a_diseno(filas[i]) for i in orden[:padres]]
        filas += evaluar(sur, mutar(base, rangos, hijos, rng, escala=0.05 / (1 + g / 2), rango_N=rango_N), red=red)
    for f in filas:
        f["factible"] = violacion(f, restricciones) == 0
    return dict(mejor=mejor, ref=ref, filas=filas, historial=historial,
                objetivo=objetivo, restricciones=restricciones)


def no_dominados(F, bloque=500):
    """Máscara de puntos no dominados (minimización en todas las columnas)."""
    F = np.asarray(F, float)
    nd = np.ones(len(F), bool)
    for i in range(0, len(F), bloque):
        A = F[i:i + bloque, None, :]
        dom = np.all(F[None] <= A, axis=2) & np.any(F[None] < A, axis=2)
        nd[i:i + bloque] = ~dom.any(axis=1)
    return nd


def frente_2d(filas, x, y, restricciones=None):
    """Índices (ordenados por x) del frente de Pareto en el plano (x, y) entre las filas
    que cumplen `restricciones`: la curva de compromiso entre dos métricas."""
    restricciones = restricciones or {}
    idx = [i for i, f in enumerate(filas) if violacion(f, restricciones) == 0]
    if not idx:
        return []
    F = np.array([[filas[i][x], filas[i][y]] for i in idx])
    sel = [idx[j] for j in np.where(no_dominados(F))[0]]
    return sorted(sel, key=lambda i: filas[i][x])


def limites_del_paper(ref, metricas=METRICAS, holgura=0.0):
    """Restricciones 'no peor que el paper' (con holgura relativa opcional)."""
    return {m: ref[m] * (1 + holgura) for m in metricas}
