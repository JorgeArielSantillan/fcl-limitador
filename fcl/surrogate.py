# -*- coding: utf-8 -*-
"""
surrogate_fcl.py
================
Surrogate del limitador de corriente de falla de núcleo de aire (FCL-ACVSR).
Solo inferencia, con NumPy: es lo que va a cargar el backend (FastAPI) sin sklearn ni PyTorch.

Física que usa el modelo
------------------------
* Inductancias propias por vuelta²:  ℓ / (μ0 r) = g(t/r, h/r)
  Es UNA sola función para las dos bobinas (misma física), invariante de escala.
* Acoplamiento:  k = ℓ12 / sqrt(ℓ11 ℓ22) = f(t_int/ri, gap/ri, t_ext/ri, h/ri, d/h)
  Adimensional e invariante de escala: si se agranda todo el dispositivo, k no cambia.
  Limitación conocida: en d = 0 la fuerza debería ser exactamente 0 (simetría) y el modelo da
  ~2 % del pico; la dinámica del limitador arranca en d0 = 0,25 h, así que no la afecta.
  Probado (d/h)² para imponer la simetría: empeora mucho entre puntos de la grilla.
* Con N1 y N2 vueltas:
      L11 = N1² ℓ11,  L22 = N2² ℓ22,  L12 = N1 N2 ℓ12
      L total (conexión serie del limitador) = L11 + L22 - 2 L12
      Fuerza de repulsión = -I² N1 N2 dℓ12/dd   (positiva = se separan)

Uso:
    from surrogate_fcl import SurrogateFCL
    s = SurrogateFCL.cargar("modelos/surrogate_fcl.npz")
    r = s.predecir(ri_int=0.53, t_int=0.03, gap=0.01, t_ext=0.03, h=0.4, d=[0.1, 0.4, 0.7])
    L = s.inductancias(0.53, 0.03, 0.01, 0.03, 0.4, d=0.1, N1=24, N2=24)
"""
import json
import numpy as np

MU0 = 4e-7 * np.pi


class _GP:
    """Media (y su gradiente) de un proceso gaussiano RBF entrenado con sklearn."""

    def __init__(self, X, alpha, c, ls, ymean, ystd, lo, hi):
        self.X, self.alpha = np.asarray(X, float), np.asarray(alpha, float).ravel()
        self.c, self.ls = float(c), np.asarray(ls, float)
        self.ymean, self.ystd = float(ymean), float(ystd)
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    @classmethod
    def desde_sklearn(cls, gp, lo, hi):
        k = gp.kernel_.k1                      # (Constant * RBF) + White -> tomamos el producto
        return cls(gp.X_train_, gp.alpha_, k.k1.constant_value, np.atleast_1d(k.k2.length_scale),
                   np.ravel(gp._y_train_mean)[0], np.ravel(gp._y_train_std)[0], lo, hi)

    def normalizar(self, F):
        return 2 * (F - self.lo) / (self.hi - self.lo) - 1

    def __call__(self, F, gradiente=False, bloque=2000):
        """F: características SIN normalizar (m x p). Devuelve media y, opcionalmente,
        dy/dF (m x p) respecto de las características sin normalizar."""
        Xn = self.normalizar(np.atleast_2d(F))
        y = np.empty(len(Xn))
        g = np.empty_like(Xn) if gradiente else None
        for i in range(0, len(Xn), bloque):
            x = Xn[i:i + bloque]
            D = (x[:, None, :] - self.X[None, :, :]) / self.ls          # m x n x p
            K = self.c * np.exp(-0.5 * (D ** 2).sum(-1))                 # m x n
            Ka = K * self.alpha
            y[i:i + bloque] = self.ymean + self.ystd * Ka.sum(1)
            if gradiente:
                g[i:i + bloque] = -self.ystd * np.einsum("mn,mnp->mp", Ka, D) / self.ls
        if gradiente:
            g *= 2 / (self.hi - self.lo)                                  # cadena por la normalización
            return y, g
        return y

    def a_dict(self, prefijo):
        return {f"{prefijo}_{k}": getattr(self, k) for k in ("X", "alpha", "c", "ls", "ymean", "ystd", "lo", "hi")}

    @classmethod
    def de_dict(cls, z, prefijo):
        return cls(*[z[f"{prefijo}_{k}"] for k in ("X", "alpha", "c", "ls", "ymean", "ystd", "lo", "hi")])


def caract_propia(r, t, h):
    return np.column_stack([np.log(t / r), np.log(h / r)])


def caract_acople(ri_int, t_int, gap, t_ext, h, d_sobre_h):
    return np.column_stack([np.log(t_int / ri_int), np.log(gap / ri_int), np.log(t_ext / ri_int),
                            np.log(h / ri_int), np.asarray(d_sobre_h)])


def dk_dd(grad_s, d, h):
    """Regla de la cadena: dk/dd = dk/d(d/h) · 1/h."""
    return grad_s / np.asarray(h)


class SurrogateFCL:
    def __init__(self, gp_propia, gp_acople, rangos, meta=None):
        self.gp_propia, self.gp_acople = gp_propia, gp_acople
        self.rangos = rangos          # {"ri_int": (min, max), ..., "d_sobre_h": (0, 2)}
        self.meta = meta or {}

    # ---------------- persistencia ----------------
    def guardar(self, path):
        np.savez_compressed(path, **self.gp_propia.a_dict("p"), **self.gp_acople.a_dict("k"),
                            rangos=json.dumps(self.rangos), meta=json.dumps(self.meta))

    @classmethod
    def cargar(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(_GP.de_dict(z, "p"), _GP.de_dict(z, "k"),
                   {k: tuple(v) for k, v in json.loads(str(z["rangos"])).items()},
                   json.loads(str(z["meta"])))

    # ---------------- validez ----------------
    def fuera_de_rango(self, ri_int, t_int, gap, t_ext, h, d):
        """Lista de avisos si la consulta sale del espacio donde se entrenó."""
        val = dict(ri_int=ri_int, t_int=t_int, gap=gap, t_ext=t_ext, h=h,
                   d_sobre_h=np.asarray(d) / np.asarray(h))
        avisos = []
        for k, (lo, hi) in self.rangos.items():
            v = np.atleast_1d(val[k])
            if (v < lo - 1e-9).any() or (v > hi + 1e-9).any():
                avisos.append(f"{k} fuera de [{lo}, {hi}]")
        return avisos

    # ---------------- predicción ----------------
    def l_propia(self, r, t, h):
        """Inductancia propia por vuelta² [H] de una bobina de radio interno r, espesor t, alto h."""
        r, t, h = np.broadcast_arrays(*map(np.atleast_1d, (r, t, h)))
        return MU0 * r * np.exp(self.gp_propia(caract_propia(r, t, h)))

    def predecir(self, ri_int, t_int, gap, t_ext, h, d):
        """Por vuelta² y en SI: l11, l22, l12 [H], k [-], dl12_dd [H/m]. Acepta escalares o arreglos."""
        ri_int, t_int, gap, t_ext, h, d = np.broadcast_arrays(
            *map(lambda v: np.atleast_1d(np.asarray(v, float)), (ri_int, t_int, gap, t_ext, h, d)))
        ri_ext = ri_int + t_int + gap
        l11 = self.l_propia(ri_int, t_int, h)
        l22 = self.l_propia(ri_ext, t_ext, h)
        k, gk = self.gp_acople(caract_acople(ri_int, t_int, gap, t_ext, h, d / h), gradiente=True)
        s = np.sqrt(l11 * l22)
        return dict(l11=l11, l22=l22, k=k, l12=k * s, dl12_dd=dk_dd(gk[:, 4], d, h) * s)

    def inductancias(self, ri_int, t_int, gap, t_ext, h, d, N1, N2):
        """Inductancias reales [H] con N1 (interior) y N2 (exterior) vueltas, y F/I² [N/A²]."""
        p = self.predecir(ri_int, t_int, gap, t_ext, h, d)
        L11, L22, L12 = N1 ** 2 * p["l11"], N2 ** 2 * p["l22"], N1 * N2 * p["l12"]
        return dict(L11=L11, L22=L22, L12=L12, L_total=L11 + L22 - 2 * L12,
                    dL_total_dd=-2 * N1 * N2 * p["dl12_dd"], F_sobre_I2=-N1 * N2 * p["dl12_dd"])
