# -*- coding: utf-8 -*-
"""
fcl_validacion_python.py
========================
Validación analítica del limitador de corriente de falla de núcleo de aire (FCL-ACVSR)
Paper: Amini, Damaki Aliabad y Amiri, IEEE Access, 2021.

1) Inductancias propias y mutua por filamentos + integrales elípticas (Maxwell)
2) Dinámica de la falla: ecuación eléctrica (9) acoplada a la mecánica (13)-(17)
3) Comparación con las Tablas IV, V, VI y VII del paper

Requisitos: pip install numpy scipy matplotlib
Se ejecuta una sola vez:  python fcl_validacion_python.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

SALIDA = Path(__file__).resolve().parent / "validacion_out"
SALIDA.mkdir(exist_ok=True)
MU0 = 4e-7 * np.pi

# ---------------- Geometría (Tabla III) ----------------
INT = (0.530, 0.560, 0.400, 24)     # ri, ro, h, N
EXT = (0.570, 0.600, 0.400, 24)
M_INT, M_EXT = 170.0, 180.0          # masas [kg]

# ---------------- Red (Tabla II) ----------------
F_RED = 50.0
W_RED = 2 * np.pi * F_RED
VM = 20e3 * np.sqrt(2) / np.sqrt(3)  # tensión de fase pico [V]
R_LINEA, L_LINEA, R_FCL = 0.03, 2.5e-3, 0.006
D0, DMAX = 0.100, 0.700              # posición inicial y tope del amortiguador [m]
G = 9.81
PICO_SIN_FCL_PAPER = 35.035e3        # Tabla V, primer período [A]


def M_espiras(a, b, dz):
    k2 = 4 * a * b / ((a + b) ** 2 + dz ** 2)
    k = np.sqrt(k2)
    return MU0 * np.sqrt(a * b) * ((2 / k - k) * ellipk(k2) - 2 / k * ellipe(k2))


def filamentos(ri, ro, h, zc, nr, nz):
    dr, dz = (ro - ri) / nr, h / nz
    r = ri + dr * (np.arange(nr) + 0.5)
    z = zc - h / 2 + dz * (np.arange(nz) + 0.5)
    R, Z = np.meshgrid(r, z)
    return R.ravel(), Z.ravel(), dr, dz


def L_propia(b, nr=12, nz=80):
    ri, ro, h, N = b
    R, Z, dr, dz = filamentos(ri, ro, h, 0, nr, nz)
    with np.errstate(all="ignore"):
        M = M_espiras(R[:, None], R[None, :], Z[:, None] - Z[None, :])
    np.fill_diagonal(M, MU0 * R * (np.log(8 * R / (0.2235 * (dr + dz))) - 2))
    return N ** 2 * M.sum() / len(R) ** 2


def L_mutua(d, nr=12, nz=80):
    R1, Z1, _, _ = filamentos(*INT[:3], +d / 2, nr, nz)
    R2, Z2, _, _ = filamentos(*EXT[:3], -d / 2, nr, nz)
    return INT[3] * EXT[3] * M_espiras(R1[:, None], R2[None, :], Z1[:, None] - Z2[None, :]).mean()


# ================= 1) INDUCTANCIAS =================
print("1) Convergencia de la discretización en filamentos")
for nr, nz in [(6, 40), (12, 80), (20, 160)]:
    print(f"   nr={nr:2d} nz={nz:3d}: L11 = {L_propia(INT, nr, nz) * 1e3:.4f} mH")
L11, L22 = L_propia(INT), L_propia(EXT)
d_grid = np.linspace(0, 0.9, 91)
M_grid = np.array([L_mutua(d) for d in d_grid])
M = CubicSpline(d_grid, M_grid)
dM = M.derivative()

print("\n   Inductancias            Python     Paper   Error")
for nombre, v, p in [("L11 interior", L11, 0.778), ("L22 exterior", L22, 0.8762),
                     ("L12 (100 mm)", M(0.1), 0.6886), ("L total (100 mm)", L11 + L22 - 2 * M(0.1), 0.277),
                     ("L12 (700 mm)", M(0.7), 0.1371), ("L total (700 mm)", L11 + L22 - 2 * M(0.7), 1.38)]:
    print(f"   {nombre:20s} {v * 1e3:8.4f}  {p:8.4f}  {100 * (v * 1e3 / p - 1):+5.1f} %")


# ================= 2) DINÁMICA DE LA FALLA =================
TH = np.arctan(W_RED * L_LINEA / R_LINEA)


def simular(alpha, con_fcl=True, t_fin=0.06):
    def rhs(t, s):
        i, x1, v1, x2, v2 = s
        d = D0 + x1 + x2
        tope = con_fcl and d >= DMAX
        if con_fcl:
            Lf, dLf, R = L11 + L22 - 2 * M(d), -2 * dM(d), R_LINEA + R_FCL
        else:
            Lf, dLf, R = 0.0, 0.0, R_LINEA
        vel = 0.0 if tope else v1 + v2
        di = (VM * np.sin(W_RED * t + alpha) - R * i - i * dLf * vel) / (L_LINEA + Lf)
        F = -i * i * dM(d) if con_fcl else 0.0          # repulsiva
        a1 = 0.0 if tope else (F - M_INT * G) / M_INT    # interior sube
        a2 = 0.0 if tope else (F + M_EXT * G) / M_EXT    # exterior baja
        if x1 <= 0 and a1 < 0 and v1 <= 0:
            a1 = 0.0
        return [di, 0.0 if tope else v1, a1, 0.0 if tope else v2, a2]
    return solve_ivp(rhs, [0, t_fin], [0, 0, 0, 0, 0], max_step=2e-5, dense_output=True)


t1 = np.linspace(0, 0.02, 4001)
alpha = brentq(lambda a: np.abs(simular(a, False, 0.02).sol(t1)[0]).max() - PICO_SIN_FCL_PAPER,
               TH - np.pi / 2, TH - np.pi / 2 + 1.2)
print(f"\n2) Falla: Vm = {VM / 1e3:.2f} kV, ángulo α-θ = {np.degrees(alpha - TH):.1f}° "
      f"(reproduce los 35 kA sin limitador)")
s1, s0 = simular(alpha, True), simular(alpha, False)
t = np.linspace(0, 0.06, 6001)
y1 = s1.sol(t)
i1, i0 = y1[0], s0.sol(t)[0]
d = np.minimum(D0 + y1[1] + y1[3], DMAX)
L_t = L11 + L22 - 2 * M(d)
F_t = -i1 ** 2 * dM(d)
e_t = i1 * (-2 * dM(d)) * np.gradient(d, t)
t_llegada = t[np.argmax(d >= DMAX - 1e-9)] * 1e3

paper_t4 = {1: (100, .277, .91), 3: (101, .279, 7.6), 5: (113, .299, 18.2), 7: (165, .412, 27.0),
            8: (219, .543, 28.5), 9: (295, .752, 26.8), 11: (496, 1.18, 17.7), 13: (700, 1.38, 14.2)}
print("   t[ms]  d[mm]  L[mH]  i[kA]  F[kN]   | paper: d, L, i")
for tm, p in paper_t4.items():
    k = np.argmin(np.abs(t - tm / 1e3))
    print(f"   {tm:4d} {d[k] * 1e3:7.1f} {L_t[k] * 1e3:6.3f} {i1[k] / 1e3:6.2f} {F_t[k] / 1e3:7.1f}   | {p}")
print(f"   Llega a 700 mm en {t_llegada:.1f} ms (paper: 13 ms)")
paper_t5 = [(35.035, 28.520), (31.303, 23.518), (28.421, 21.802)]
for k, (a, b) in enumerate([(0, .02), (.02, .04), (.04, .06)]):
    sel = (t >= a) & (t < b)
    p0, p1 = np.abs(i0[sel]).max() / 1e3, np.abs(i1[sel]).max() / 1e3
    print(f"   Período {k + 1}: sin FCL {p0:6.2f} kA | con FCL {p1:6.2f} kA | limita {100 * (1 - p1 / p0):4.1f} %"
          f"   (paper {paper_t5[k][1]:.2f} kA, {100 * (1 - paper_t5[k][1] / paper_t5[k][0]):.1f} %)")

# ================= 3) FIGURAS =================
fig, ax = plt.subplots(2, 2, figsize=(12, 8), dpi=110)
ax[0, 0].plot(t * 1e3, i0 / 1e3, "r", label="sin limitador")
ax[0, 0].plot(t * 1e3, i1 / 1e3, "b--", label="con limitador")
ax[0, 0].set(xlabel="t [ms]", ylabel="i [kA]", title="Corriente de cortocircuito (Fig. 7)")
ax[0, 0].legend()
ax[0, 1].plot(t * 1e3, d * 1e3, label="Python")
ax[0, 1].plot([k for k in paper_t4], [v[0] for v in paper_t4.values()], "ks", mfc="none", label="paper")
ax[0, 1].set(xlim=(0, 20), xlabel="t [ms]", ylabel="d [mm]", title="Distancia entre bobinas (Fig. 8)")
ax[0, 1].legend()
ax[1, 0].plot(t * 1e3, F_t / 1e3)
ax[1, 0].set(xlabel="t [ms]", ylabel="F [kN]", title="Fuerza de repulsión (Fig. 9)")
ax[1, 1].plot(t * 1e3, L_t * 1e3, label="L(t)")
ax[1, 1].plot([k for k in paper_t4], [v[1] for v in paper_t4.values()], "ks", mfc="none", label="paper")
ax[1, 1].set(xlim=(0, 20), xlabel="t [ms]", ylabel="L [mH]", title="Inductancia del limitador (Fig. 10)")
ax2 = ax[1, 1].twinx()
ax2.plot(t * 1e3, e_t / 1e3, "C3:", label="e(x,i)")
ax2.set_ylabel("tensión inducida por movimiento [kV]")
ax[1, 1].legend(loc="upper left")
for a in ax.ravel():
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(SALIDA / "validacion_falla.png")
print(f"\nFiguras en: {SALIDA}")
