# -*- coding: utf-8 -*-
"""
estudio_optimizacion.py
=======================
El "gancho" del proyecto: diseño inverso del limitador con el surrogate.

1) Valida el simulador de falla: surrogate vs método analítico vs paper (geometría del paper).
2) Busca 5 objetivos a la vez: ¿existe un diseño mejor que el del paper en TODO?
3) Curva de compromiso: masa mínima vs pico del 2.º período, sin empeorar caída de tensión,
   pico del 1.er período ni pérdidas.
4) Compara la falla del paper con un diseño optimizado.

Salidas en docs/img/ y docs/resultados.json.  Se ejecuta una sola vez:  python estudio_optimizacion.py
"""
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fcl.surrogate import SurrogateFCL
from fcl.dinamica import PAPER, Red, simular, tablas_surrogate, tablas_analiticas
from fcl.optimizacion import (METRICAS, CLAVES, evaluar, optimizar, frente_2d, limites_del_paper,
                              a_diseno, violacion)

AQUI = Path(__file__).resolve().parent
IMG = AQUI / "docs" / "img"
DPI = 120
PAPER_TABLA_IV = dict(t_ms=[1, 3, 5, 7, 8, 9, 11, 13],
                      d_mm=[100, 101, 113, 165, 219, 295, 496, 700],
                      i_kA=[0.91, 7.6, 18.2, 27.0, 28.5, 26.8, 17.7, 14.2])
PAPER_TABLA_V = dict(sin=[35.035, 31.303, 28.421], con=[28.520, 23.518, 21.802])
HOLGURA_PICO2 = 0.10            # diseño a destacar: pico2 hasta +10 % del paper


def guardar(fig, nombre):
    fig.tight_layout()
    for k in range(8):
        try:
            fig.savefig(IMG / nombre)
            break
        except OSError:
            time.sleep(0.4 * (k + 1))
    plt.close(fig)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    sur = SurrogateFCL.cargar(AQUI / "modelos" / "surrogate_fcl.npz")
    res = {}

    # ---------------- 1) validación del simulador ----------------
    print("1) Simulación de la falla con la geometría del paper")
    rs = simular([PAPER], tablas_surrogate(sur, [PAPER]))
    ra = simular([PAPER], tablas_analiticas([PAPER]))
    for nombre, r in (("surrogate", rs), ("analítico", ra)):
        print(f"   {nombre:9s}: picos {np.round(r['pico_kA'][0], 2)} kA | limitación "
              f"{np.round(r['limitacion_pct'][0], 1)} % | llega a d_max en {r['t_operacion_ms'][0]:.1f} ms")
    print(f"   paper    : picos {PAPER_TABLA_V['con']} kA | llega a 700 mm en 13 ms")
    res["paper_simulado"] = dict(picos_kA=rs["pico_kA"][0].tolist(), limitacion_pct=rs["limitacion_pct"][0].tolist(),
                                 t_operacion_ms=float(rs["t_operacion_ms"][0]),
                                 diferencia_surrogate_analitico_kA=float(np.abs(rs["pico_kA"] - ra["pico_kA"]).max()))
    s = rs["series"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), dpi=DPI)
    ax[0, 0].plot(s["t"] * 1e3, s["i_sin_fcl"] / 1e3, color="0.6", label="sin limitador")
    ax[0, 0].plot(s["t"] * 1e3, s["i"][:, 0] / 1e3, "C0", label="con limitador (surrogate)")
    ax[0, 0].plot(PAPER_TABLA_IV["t_ms"], PAPER_TABLA_IV["i_kA"], "ks", mfc="none", label="paper (Tabla IV)")
    ax[0, 0].set(xlabel="t [ms]", ylabel="i [kA]", title="Corriente de cortocircuito")
    ax[0, 1].plot(s["t"] * 1e3, s["d"][:, 0] * 1e3, "C0", label="surrogate")
    ax[0, 1].plot(PAPER_TABLA_IV["t_ms"], PAPER_TABLA_IV["d_mm"], "ks", mfc="none", label="paper")
    ax[0, 1].set(xlim=(0, 25), xlabel="t [ms]", ylabel="d [mm]", title="Distancia entre bobinas")
    ax[1, 0].plot(s["t"] * 1e3, s["F"][:, 0] / 1e3, "C3")
    ax[1, 0].set(xlim=(0, 25), xlabel="t [ms]", ylabel="F [kN]", title="Fuerza de repulsión")
    ax[1, 1].plot(s["t"] * 1e3, s["L"][:, 0] * 1e3, "C2")
    ax[1, 1].set(xlim=(0, 25), xlabel="t [ms]", ylabel="L [mH]", title="Inductancia del limitador")
    for a in ax.ravel():
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    guardar(fig, "falla_paper.png")

    # ---------------- 2) cinco objetivos a la vez ----------------
    print("2) ¿Hay un diseño mejor que el del paper en los 5 objetivos?")
    ref = evaluar(sur, [PAPER])[0]
    r5 = optimizar(sur, objetivo="masa_kg", restricciones=limites_del_paper(ref), generaciones=6)
    m5 = r5["mejor"]
    if m5:
        mejora = {m: 100 * (1 - m5[m] / ref[m]) for m in METRICAS}
        print("   mejor encontrado (mejora % respecto del paper): " +
              ", ".join(f"{m} {v:.1f}" for m, v in mejora.items()))
    else:
        mejora = None
        print(f"   ninguno entre {len(r5['filas'])} diseños: el del paper está sobre (o muy cerca de) el frente de Pareto")
    res["cinco_objetivos"] = dict(mejor=m5, mejora_pct=mejora)

    # ---------------- 3) curva de compromiso masa vs pico2 ----------------
    print("3) Curva de compromiso: masa mínima vs pico del 2.º período")
    lim = limites_del_paper(ref)
    lim["pico2_kA"] = ref["pico2_kA"] * 1.3
    r = optimizar(sur, objetivo="masa_kg", restricciones=lim, generaciones=6)
    filas = r["filas"] + r5["filas"]
    otros = {m: ref[m] for m in ("caida_tension_pct", "pico1_kA", "perdidas_W")}
    fr = frente_2d(filas, "pico2_kA", "masa_kg", otros)
    curva = [filas[i] for i in fr]
    cand = [f for f in curva if f["pico2_kA"] <= ref["pico2_kA"] * (1 + HOLGURA_PICO2)]
    elegido = min(cand, key=lambda f: f["masa_kg"]) if cand else None
    res["curva_compromiso"] = [{k: f[k] for k in list(CLAVES) + ["N"] + list(METRICAS) + ["t_operacion_ms"]}
                               for f in curva]
    res["diseno_destacado"] = elegido
    res["paper"] = ref

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=DPI)
    todos = [f for f in filas if violacion(f, otros) == 0]
    ax.scatter([f["pico2_kA"] for f in todos], [f["masa_kg"] for f in todos], s=5, c="0.8",
               label="diseños evaluados (ΔV, pico1 y pérdidas ≤ paper)")
    ax.plot([f["pico2_kA"] for f in curva], [f["masa_kg"] for f in curva], "o-", ms=3, c="C0",
            label="frente de Pareto")
    ax.plot(ref["pico2_kA"], ref["masa_kg"], "k*", ms=16, label="diseño del paper")
    if elegido:
        ax.plot(elegido["pico2_kA"], elegido["masa_kg"], "D", c="C3", ms=9,
                label=f"destacado: {elegido['masa_kg']:.0f} kg, pico2 +{100 * (elegido['pico2_kA'] / ref['pico2_kA'] - 1):.0f} %")
    ax.set(xlabel="pico de corriente en el 2.º período [kA]", ylabel="masa de las bobinas [kg]",
           title="Curva de compromiso del limitador", ylim=(0, max(400, ref["masa_kg"] * 1.15)))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    guardar(fig, "frente_pareto.png")

    # ---------------- 4) falla: paper vs destacado ----------------
    if elegido:
        de = a_diseno(elegido)
        rr = simular([PAPER, de], tablas_surrogate(sur, [PAPER, de]))
        s = rr["series"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=DPI)
        a1.plot(s["t"] * 1e3, s["i_sin_fcl"] / 1e3, color="0.7", label="sin limitador")
        a1.plot(s["t"] * 1e3, s["i"][:, 0] / 1e3, "k", label=f"paper ({ref['masa_kg']:.0f} kg)")
        a1.plot(s["t"] * 1e3, s["i"][:, 1] / 1e3, "C3", label=f"optimizado ({elegido['masa_kg']:.0f} kg)")
        a1.set(xlabel="t [ms]", ylabel="i [kA]", title="Corriente de falla")
        a2.plot(s["t"] * 1e3, s["d"][:, 0] / PAPER.h, "k", label="paper")
        a2.plot(s["t"] * 1e3, s["d"][:, 1] / de.h, "C3", label="optimizado")
        a2.set(xlim=(0, 25), xlabel="t [ms]", ylabel="d / h", title="Separación relativa de las bobinas")
        for a in (a1, a2):
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        guardar(fig, "falla_paper_vs_optimizado.png")
        print(f"   destacado: {elegido['masa_kg']:.0f} kg (paper {ref['masa_kg']:.0f}), "
              f"pico2 {elegido['pico2_kA']:.2f} kA (paper {ref['pico2_kA']:.2f}), "
              f"t_operación {elegido['t_operacion_ms']:.1f} ms (paper {ref['t_operacion_ms']:.1f})")

    (AQUI / "docs" / "resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                                                   encoding="utf-8")
    print(f"Listo: figuras en {IMG}")


if __name__ == "__main__":
    main()
