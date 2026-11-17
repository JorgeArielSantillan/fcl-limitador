# -*- coding: utf-8 -*-
"""
Dinámica de la falla del limitador FCL-ACVSR (ecuación eléctrica + mecánica acopladas).

Igual que el paper (Amini et al., IEEE Access 2021, ec. 9 y 13-17):
  * Circuito:  v(t) = R i + d/dt[(L_red + L_fcl(d)) i]
               -> di/dt = (v - R i - i dL_fcl/dd · ḋ) / (L_red + L_fcl)
  * Fuerza de repulsión entre bobinas:  F = -i² N1 N2 dℓ12/dd
  * Bobina interior sube (contra la gravedad) y la exterior baja; tope en d_max.

Está VECTORIZADO: integra cientos o miles de diseños a la vez con RK4 de paso fijo.
Las inductancias salen del surrogate en una tabla por diseño (interpolación de Hermite con la
derivada exacta del surrogate), así cada paso de tiempo cuesta microsegundos.

Supuestos (no publicados en el paper, declarados):
  * Masa de cada bobina proporcional al volumen del bobinado (calibrada con 170/180 kg del paper).
  * Resistencia del limitador ∝ N² · r_medio / (t · h) (calibrada con 0,006 Ω del paper).
  * Sin resorte ni amortiguador hasta el tope; d0 = 0,25 h y d_max = 1,75 h (100 y 700 mm en el paper).
  * Ángulo de inicio de falla ajustado para reproducir los 35 kA de la Tabla V sin limitador.
"""
from dataclasses import dataclass, asdict, field
import numpy as np
from scipy.optimize import brentq

G = 9.81

# ----------------------------------------------------------------------------- red (Tabla II)
@dataclass
class Red:
    V_linea_kV: float = 20.0
    f: float = 50.0
    R: float = 0.03            # Ω
    L: float = 2.5e-3          # H
    I_nominal: float = 360.0   # A eficaces
    pico_sin_fcl_kA: float = 35.035   # Tabla V: define el ángulo de inicio de la falla

    @property
    def w(self):
        return 2 * np.pi * self.f

    @property
    def Vm(self):              # tensión de fase pico
        return self.V_linea_kV * 1e3 * np.sqrt(2) / np.sqrt(3)

    @property
    def theta(self):
        return np.arctan(self.w * self.L / self.R)

    def corriente_sin_fcl(self, t, alpha):
        Z = np.hypot(self.R, self.w * self.L)
        tau = self.L / self.R
        return self.Vm / Z * (np.sin(self.w * t + alpha - self.theta)
                              - np.sin(alpha - self.theta) * np.exp(-t / tau))

    def angulo_falla(self):
        """α tal que el primer pico sin limitador sea pico_sin_fcl_kA (rama de -90°..0° respecto de θ)."""
        t = np.linspace(0, 0.02, 4001)
        obj = lambda a: np.abs(self.corriente_sin_fcl(t, a)).max() - self.pico_sin_fcl_kA * 1e3
        return brentq(obj, self.theta - np.pi / 2, self.theta - np.pi / 2 + 1.2)


# ----------------------------------------------------------------------------- diseño
@dataclass
class Diseno:
    ri_int: float = 0.530
    t_int: float = 0.030
    gap: float = 0.010
    t_ext: float = 0.030
    h: float = 0.400
    N: int = 24                 # vueltas de cada bobina (N1 = N2, como en el paper)

    @property
    def ri_ext(self):
        return self.ri_int + self.t_int + self.gap

    def dict(self):
        return asdict(self)


PAPER = Diseno()
MASA_PAPER = (170.0, 180.0)     # kg: interior, exterior
R_FCL_PAPER = 0.006             # Ω
D0_SOBRE_H, DMAX_SOBRE_H = 0.25, 1.75


def _vol(ri, t, h):
    return np.pi * ((ri + t) ** 2 - ri ** 2) * h


def masas(ds):
    """(m_int, m_ext) en kg, escaladas por volumen de bobinado respecto del paper."""
    v1p, v2p = _vol(PAPER.ri_int, PAPER.t_int, PAPER.h), _vol(PAPER.ri_ext, PAPER.t_ext, PAPER.h)
    m1 = np.array([MASA_PAPER[0] * _vol(d.ri_int, d.t_int, d.h) / v1p for d in ds])
    m2 = np.array([MASA_PAPER[1] * _vol(d.ri_ext, d.t_ext, d.h) / v2p for d in ds])
    return m1, m2


def resistencia(ds):
    """R del limitador [Ω] ∝ Σ N² r_medio / (t h), calibrada con el paper."""
    def s(d):
        return d.N ** 2 * ((d.ri_int + d.t_int / 2) / (d.t_int * d.h) + (d.ri_ext + d.t_ext / 2) / (d.t_ext * d.h))
    return np.array([R_FCL_PAPER * s(d) / s(PAPER) for d in ds])


# ----------------------------------------------------------------------------- tablas L(d)
@dataclass
class Tablas:
    """Por diseño: ℓ11, ℓ22 y ℓ12(d), dℓ12/dd en una grilla uniforme de d (Hermite cúbica)."""
    d0: np.ndarray       # (B,) inicio de la grilla [m]
    paso: np.ndarray     # (B,)
    l11: np.ndarray      # (B,)  por vuelta² [H]
    l22: np.ndarray
    l12: np.ndarray      # (B, n)
    dl12: np.ndarray     # (B, n)  [H/m]

    def evaluar(self, d):
        """ℓ12(d) y dℓ12/dd(d) para un d por diseño (vectorizado, Hermite consistente)."""
        n = self.l12.shape[1]
        u = np.clip((d - self.d0) / self.paso, 0, n - 1 - 1e-9)
        j = u.astype(int)
        s = u - j
        r = np.arange(len(d))
        y0, y1 = self.l12[r, j], self.l12[r, j + 1]
        m0, m1 = self.dl12[r, j] * self.paso, self.dl12[r, j + 1] * self.paso
        s2, s3 = s * s, s * s * s
        val = (2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * m1
        der = ((6 * s2 - 6 * s) * y0 + (3 * s2 - 4 * s + 1) * m0 + (-6 * s2 + 6 * s) * y1 + (3 * s2 - 2 * s) * m1) / self.paso
        return val, der


def tablas_surrogate(sur, ds, n=41, desde=0.2, hasta=1.8):
    """Construye las tablas con el surrogate en UNA llamada vectorizada (B·n puntos)."""
    B = len(ds)
    h = np.array([d.h for d in ds])
    rel = np.linspace(desde, hasta, n)
    D = h[:, None] * rel[None, :]
    rep = lambda a: np.repeat(np.array(a, float), n)
    p = sur.predecir(rep([d.ri_int for d in ds]), rep([d.t_int for d in ds]), rep([d.gap for d in ds]),
                     rep([d.t_ext for d in ds]), np.repeat(h, n), D.ravel())
    return Tablas(d0=h * desde, paso=h * (rel[1] - rel[0]),
                  l11=p["l11"].reshape(B, n)[:, 0], l22=p["l22"].reshape(B, n)[:, 0],
                  l12=p["l12"].reshape(B, n), dl12=p["dl12_dd"].reshape(B, n))


def tablas_analiticas(ds, n=81, desde=0.2, hasta=1.8):
    """Mismas tablas con el método analítico de filamentos (para validar el simulador)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulacion"))
    import fcl_dataset as FD
    B = len(ds)
    l11, l22, L12, D12 = np.zeros(B), np.zeros(B), np.zeros((B, n)), np.zeros((B, n))
    rel = np.linspace(desde, hasta, n)
    for b, d in enumerate(ds):
        g = FD.completar(dict(ri_int=d.ri_int, t_int=d.t_int, gap=d.gap, t_ext=d.t_ext, h=d.h))
        l11[b] = FD.l_propia_an(g["ri_int"], g["ro_int"], g["h"])
        l22[b] = FD.l_propia_an(g["ri_ext"], g["ro_ext"], g["h"])
        for j, r in enumerate(rel):
            x, e = r * d.h, 1e-4
            L12[b, j] = FD.l_mutua_an(g, x)
            D12[b, j] = (FD.l_mutua_an(g, x + e) - FD.l_mutua_an(g, x - e)) / (2 * e)
    h = np.array([d.h for d in ds])
    return Tablas(h * desde, h * (rel[1] - rel[0]), l11, l22, L12, D12)


# ----------------------------------------------------------------------------- simulación
def simular(ds, tablas, red=None, t_fin=0.06, dt=1e-5, guardar_cada=10, guardar_series=None):
    """Integra la falla para todos los diseños a la vez (RK4, paso fijo).

    Devuelve un dict con métricas por diseño (arreglos de largo B) y, si guardar_series=True
    (por defecto solo cuando B <= 10), las series temporales.
    """
    red = red or Red()
    B = len(ds)
    if guardar_series is None:
        guardar_series = B <= 10
    alpha = red.angulo_falla()
    N2 = np.array([d.N for d in ds], float) ** 2
    h = np.array([d.h for d in ds])
    d0, dmax = D0_SOBRE_H * h, DMAX_SOBRE_H * h
    m1, m2 = masas(ds)
    R = red.R + resistencia(ds)
    Lred, Vm, w = red.L, red.Vm, red.w
    Lprop = N2 * (tablas.l11 + tablas.l22)

    def deriv(t, y):
        i, x1, v1, x2, v2 = y
        d = np.minimum(d0 + x1 + x2, dmax)
        tope = (d0 + x1 + x2) >= dmax
        l12, dl12 = tablas.evaluar(d)
        Lf = Lprop - 2 * N2 * l12
        dLf = -2 * N2 * dl12
        vel = np.where(tope, 0.0, v1 + v2)
        di = (Vm * np.sin(w * t + alpha) - R * i - i * dLf * vel) / (Lred + Lf)
        F = -i * i * N2 * dl12
        a1 = np.where(tope, 0.0, (F - m1 * G) / m1)
        a2 = np.where(tope, 0.0, (F + m2 * G) / m2)
        a1 = np.where((x1 <= 0) & (a1 < 0) & (v1 <= 0), 0.0, a1)
        return np.array([di, np.where(tope, 0.0, v1), a1, np.where(tope, 0.0, v2), a2])

    y = np.zeros((5, B))
    nsteps = int(round(t_fin / dt))
    per = int(round(1 / red.f / dt))
    picos = np.zeros((int(np.ceil(nsteps / per)), B))
    t_op = np.full(B, np.nan)
    F_max = np.zeros(B)
    series = {k: [] for k in ("t", "i", "d", "L", "F")} if guardar_series else None
    for k in range(nsteps):
        t = k * dt
        if series is not None and k % guardar_cada == 0:
            d = np.minimum(d0 + y[1] + y[3], dmax)
            l12, dl12 = tablas.evaluar(d)
            series["t"].append(t)
            series["i"].append(y[0].copy())
            series["d"].append(d)
            series["L"].append(Lprop - 2 * N2 * l12)
            series["F"].append(-y[0] ** 2 * N2 * dl12)
        k1 = deriv(t, y)
        k2 = deriv(t + dt / 2, y + dt / 2 * k1)
        k3 = deriv(t + dt / 2, y + dt / 2 * k2)
        k4 = deriv(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        picos[k // per] = np.maximum(picos[k // per], np.abs(y[0]))
        d = d0 + y[1] + y[3]
        llego = np.isnan(t_op) & (d >= dmax)
        t_op[llego] = t + dt
        l12, dl12 = tablas.evaluar(np.minimum(d, dmax))
        F_max = np.maximum(F_max, -y[0] ** 2 * N2 * dl12)

    t_ref = np.arange(nsteps) * dt
    i_sin = red.corriente_sin_fcl(t_ref, alpha)
    picos_sin = np.array([np.abs(i_sin[j * per:(j + 1) * per]).max() for j in range(picos.shape[0])])
    l12_0, _ = tablas.evaluar(d0)
    L0 = Lprop - 2 * N2 * l12_0
    Vf = red.Vm / np.sqrt(2)
    out = dict(
        pico_kA=picos.T / 1e3,                                    # (B, períodos)
        pico_sin_fcl_kA=picos_sin / 1e3,                          # (períodos,)
        limitacion_pct=100 * (1 - picos.T / picos_sin[None, :]),  # (B, períodos)
        t_operacion_ms=t_op * 1e3,
        L_servicio_mH=L0 * 1e3,
        L_final_mH=(Lprop - 2 * N2 * tablas.evaluar(dmax)[0]) * 1e3,
        caida_tension_pct=100 * w * L0 * red.I_nominal / Vf,
        perdidas_W=3 * red.I_nominal ** 2 * (R - red.R),
        masa_kg=m1 + m2,
        F_max_kN=F_max / 1e3,
        alpha_menos_theta_grados=float(np.degrees(alpha - red.theta)),
    )
    if series is not None:
        out["series"] = {k: np.array(v) for k, v in series.items()}
        out["series"]["i_sin_fcl"] = red.corriente_sin_fcl(out["series"]["t"], alpha)
    return out
