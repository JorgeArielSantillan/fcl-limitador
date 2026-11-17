# -*- coding: utf-8 -*-
"""
Genera los GIF del README sin necesitar GMSH ni Elmer:
  1) docs/img/lineas_flujo_vs_d.gif  — campo magnético (modelo analítico de filamentos, que coincide
     con Elmer dentro del 0,2 % en inductancias) para distintas separaciones d.
  2) docs/img/falla_animada.gif      — la falla en el tiempo con el surrogate: bobinas separándose,
     corriente limitada e inductancia creciente.

    python docs/generar_gifs.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import ellipk, ellipe

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from fcl.surrogate import SurrogateFCL           # noqa: E402
from fcl.dinamica import PAPER, simular, tablas_surrogate, D0_SOBRE_H, DMAX_SOBRE_H   # noqa: E402

IMG = RAIZ / "docs" / "img"
MU0 = 4e-7 * np.pi
I_PICO = 360 * np.sqrt(2)          # corriente nominal pico [A]


def guardar_gif(figs_png, destino, ms):
    from PIL import Image
    cuadros = [Image.open(p).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=255) for p in figs_png]
    cuadros[0].save(destino, save_all=True, append_images=cuadros[1:], duration=ms, loop=0)
    for p in set(figs_png):
        Path(p).unlink()


def bobinas(d, g=PAPER):
    ri_ext = g.ri_int + g.t_int + g.gap
    return ((g.ri_int, g.ri_int + g.t_int, d / 2 - g.h / 2, d / 2 + g.h / 2),
            (ri_ext, ri_ext + g.t_ext, -d / 2 - g.h / 2, -d / 2 + g.h / 2))


def psi_filamentos(R, Z, d, nr=4, nz=40, g=PAPER):
    """Función de flujo ψ = 2π r A_φ en (R, Z) para la conexión serie (+NI interior, −NI exterior)."""
    psi = np.zeros_like(R)
    for signo, (r0, r1, z0, z1) in zip((+1, -1), bobinas(d, g)):
        I_fil = signo * g.N * I_PICO / (nr * nz)
        for a in r0 + (r1 - r0) * (np.arange(nr) + 0.5) / nr:
            for zf in z0 + (z1 - z0) * (np.arange(nz) + 0.5) / nz:
                k2 = np.minimum(4 * a * R / ((a + R) ** 2 + (Z - zf) ** 2), 1 - 1e-9)
                k = np.sqrt(k2)
                A = MU0 * I_fil / (np.pi * k) * np.sqrt(a / R) * ((1 - k2 / 2) * ellipk(k2) - ellipe(k2))
                psi += 2 * np.pi * R * A
    return psi


def gif_flujo():
    r = np.linspace(1e-3, 1.2, 220)
    z = np.linspace(-0.9, 0.9, 330)
    R, Z = np.meshgrid(r, z)
    ds = np.r_[np.linspace(0, 0.7, 15)]
    campos = [psi_filamentos(R, Z, d) for d in ds]
    bmax, psimax = 0.0, max(np.abs(p).max() for p in campos)
    B = []
    for psi in campos:
        dpz, dpr = np.gradient(psi, z, r)
        Bm = np.hypot(dpr, dpz) / (2 * np.pi * R)
        B.append(Bm)
        bmax = max(bmax, np.percentile(Bm, 99.5))
    niveles = np.linspace(-psimax, psimax, 30)
    niveles = niveles[np.abs(niveles) > 1e-9 * psimax]
    pngs = []
    for k, (d, psi, Bm) in enumerate(zip(ds, campos, B)):
        fig, ax = plt.subplots(figsize=(5.2, 6.4), dpi=100)
        pc = ax.pcolormesh(R, Z, np.minimum(Bm, bmax) * 1e3, cmap="magma", vmin=0, vmax=bmax * 1e3, shading="auto")
        ax.contour(R, Z, psi, levels=niveles, colors="w", linewidths=0.6)
        for r0, r1, z0, z1 in bobinas(d):
            ax.add_patch(Rectangle((r0, z0), r1 - r0, z1 - z0, fill=False, ec="#4dd0e1", lw=1.5))
        ax.set(xlim=(0, 1.2), ylim=(-0.9, 0.9), xlabel="r [m]", ylabel="z [m]", aspect="equal",
               title=f"d = {d * 1e3:.0f} mm")
        fig.colorbar(pc, ax=ax, fraction=0.05, pad=0.03).set_label("|B| [mT] (360 A eficaces)")
        fig.text(0.5, 0.01, "Campo calculado con el modelo analítico de filamentos", ha="center", fontsize=7, color="0.4")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        p = IMG / f"_f{k:02d}.png"
        fig.savefig(p)
        plt.close(fig)
        pngs.append(p)
    guardar_gif(pngs + pngs[::-1], IMG / "lineas_flujo_vs_d.gif", 180)


def gif_falla():
    sur = SurrogateFCL.cargar(RAIZ / "modelos" / "surrogate_fcl.npz")
    r = simular([PAPER], tablas_surrogate(sur, [PAPER]), t_fin=0.04, guardar_cada=10)
    s = r["series"]
    t = s["t"] * 1e3
    i, isin, dd, L = s["i"][:, 0] / 1e3, s["i_sin_fcl"] / 1e3, s["d"][:, 0], s["L"][:, 0] * 1e3
    pngs = []
    cuadros = np.r_[np.arange(0, 220, 5), np.arange(220, len(t), 12)]
    for k, j in enumerate(cuadros):
        fig = plt.figure(figsize=(10, 4.4), dpi=90)
        a0 = fig.add_axes([0.04, 0.12, 0.2, 0.78])
        for r0, r1, z0, z1 in bobinas(dd[j]):
            for signo in (1, -1):        # corte completo (espejo en r)
                a0.add_patch(Rectangle((signo * r0, z0), signo * (r1 - r0), z1 - z0, color="#b87333"))
        a0.axvline(0, color="0.6", lw=0.8, ls="--")
        a0.set(xlim=(-0.75, 0.75), ylim=(-0.75, 0.75), aspect="equal", xticks=[], yticks=[],
               title=f"d = {dd[j] * 1e3:.0f} mm")
        a1 = fig.add_axes([0.31, 0.14, 0.38, 0.74])
        a1.plot(t, isin, color="0.8", label="sin limitador")
        a1.plot(t[:j + 1], i[:j + 1], "C0", label="con limitador")
        a1.plot(t[j], i[j], "o", c="C0")
        a1.set(xlim=(0, 40), ylim=(-14, 37), xlabel="t [ms]", ylabel="i [kA]", title="Corriente de falla")
        a1.legend(fontsize=8, loc="lower right")
        a1.grid(alpha=0.3)
        a2 = fig.add_axes([0.76, 0.14, 0.22, 0.74])
        a2.plot(t[:j + 1], L[:j + 1], "C2")
        a2.set(xlim=(0, 40), ylim=(0, 1.45), xlabel="t [ms]", ylabel="L [mH]", title="Inductancia")
        a2.grid(alpha=0.3)
        p = IMG / f"_a{k:03d}.png"
        fig.savefig(p)
        plt.close(fig)
        pngs.append(p)
    pngs += [pngs[-1]] * 8                 # pausa al final
    guardar_gif(pngs, IMG / "falla_animada.gif", 70)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    gif_flujo()
    gif_falla()
    print(f"Listo: {IMG / 'lineas_flujo_vs_d.gif'} y {IMG / 'falla_animada.gif'}")
