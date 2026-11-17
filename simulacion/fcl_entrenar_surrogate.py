# -*- coding: utf-8 -*-
"""
fcl_entrenar_surrogate.py
=========================
Entrena y valida el surrogate del limitador (FCL-ACVSR) con el dataset de GMSH + Elmer.

Modelos
  * Inductancia propia  ℓ/(μ0 r) = g(t/r, h/r)          -> proceso gaussiano (GP)
  * Acoplamiento        k = f(5 cocientes adimensionales) -> se comparan:
        - GP con escala de correlación en d acotada (ver nota)
        - Red neuronal MLP en PyTorch (opcional: se saltea si no está instalado)
    El modelo de producción se elige con un conjunto de VALIDACIÓN (geometrías de train
    apartadas) y se informa el error final en el conjunto de PRUEBA, que nunca se usa para decidir.

Nota sobre el GP: si se deja libre, aprende una escala de correlación en d más corta que el
paso de la grilla (las 21 distancias son las mismas en todas las geometrías). Acierta en la grilla
pero falla ENTRE puntos (error en k ~0,5). Por eso se acota y por eso se valida además en
distancias intermedias contra el método analítico.

Validación
  1) Prueba en la grilla (20 geometrías nunca vistas, Elmer)
  2) Prueba en d intermedios (las mismas geometrías, referencia analítica)
  3) Derivada dℓ12/dd (fuerza) contra el analítico por diferencias finitas
  4) Geometría del paper: L(d) y F/I² continuas vs Elmer vs Tabla IV

Requisitos: pip install numpy pandas scipy scikit-learn matplotlib   (+ torch, opcional)
Usa fcl_dataset.py (misma carpeta), fcl/surrogate.py y el dataset de fcl_dataset_out/resultados/
(o, si no existe, el de datos/). Guarda el modelo en modelos/surrogate_fcl.npz.
Se ejecuta una sola vez desde la raíz del repo:  python simulacion/fcl_entrenar_surrogate.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import sys
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ))
import fcl_dataset as FD                         # referencia analítica y rangos del muestreo
from fcl.surrogate import SurrogateFCL, _GP, caract_propia, caract_acople, dk_dd, MU0

# Usa el dataset recién generado si existe; si no, el que viene en el repositorio (datos/)
DATOS = AQUI / "fcl_dataset_out" / "resultados"
if not (DATOS / "dataset_train.csv").exists():
    DATOS = RAIZ / "datos"
SALIDA = AQUI / "surrogate_out"
MODELOS = RAIZ / "modelos"
N_VALIDACION = 20            # geometrías de train apartadas para elegir modelo
SEMILLA = 0
ESCALA_D_MIN = 0.2           # cota inferior de la escala de correlación en d/h (normalizado)
N_PAPER = 24
DPI = 110

PAPER_D = [100, 100.124, 100.981, 104.178, 112.722, 131.052, 164.627, 218.54, 294.67, 389.5, 496.12, 608.823, 700]
PAPER_L = [0.27697, 0.2772, 0.279, 0.28406, 0.29887, 0.33845, 0.4115, 0.5431, 0.7524, 1.003, 1.18, 1.305, 1.38]


def guardar_fig(fig, destino):
    for k in range(8):
        try:
            fig.savefig(destino)
            break
        except OSError as e:
            if k == 7:
                print(f"   ⚠ no pude guardar {destino}: {e}")
            time.sleep(0.4 * (k + 1))
    plt.close(fig)


# =============================================================================
# DATOS
# =============================================================================
def cargar():
    tr = pd.read_csv(DATOS / "dataset_train.csv")
    te = pd.read_csv(DATOS / "dataset_test.csv")
    for d in (tr, te):
        d["k"] = d.l12_uH / np.sqrt(d.l11_uH * d.l22_uH)
    return tr, te


def datos_propia(df):
    """Una fila por bobina (interior y exterior de cada geometría): (r, t, h) -> ℓ/(μ0 r)."""
    g = df.drop_duplicates(["conjunto", "geom_id"])
    r = np.r_[g.ri_int, g.ri_ext]
    t = np.r_[g.t_int, g.t_ext]
    h = np.r_[g.h, g.h]
    l = np.r_[g.l11_uH, g.l22_uH] * 1e-6
    return r, t, h, l


def F_acople(df):
    return caract_acople(df.ri_int.values, df.t_int.values, df.gap.values, df.t_ext.values,
                         df.h.values, df.d_sobre_h.values)


def referencia_intermedia(te):
    """Geometrías de prueba en d/h = 0.05, 0.15, ..., 1.95 con el método analítico."""
    filas = []
    for _, r in te.drop_duplicates("geom_id").iterrows():
        G = FD.completar({c: r[c] for c in FD.RANGOS})
        l11 = FD.l_propia_an(G["ri_int"], G["ro_int"], G["h"])
        l22 = FD.l_propia_an(G["ri_ext"], G["ro_ext"], G["h"])
        for dh in np.arange(0.05, 2.0, 0.1):
            d = dh * G["h"]
            dd = 1e-3
            l12 = FD.l_mutua_an(G, d)
            der = (FD.l_mutua_an(G, d + dd) - FD.l_mutua_an(G, max(d - dd, 0))) / (d + dd - max(d - dd, 0))
            filas.append(dict(geom_id=r.geom_id, **{c: r[c] for c in FD.RANGOS}, d_sobre_h=dh, d_m=d,
                              l11=l11, l22=l22, l12=l12, k=l12 / np.sqrt(l11 * l22), dl12_dd=der))
    return pd.DataFrame(filas)


# =============================================================================
# MODELOS
# =============================================================================
def gp_propia(r, t, h, l):
    F = caract_propia(r, t, h)
    lo, hi = F.min(0), F.max(0)
    Xn = 2 * (F - lo) / (hi - lo) - 1
    kern = ConstantKernel(1.0) * RBF([1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(1e-8, (1e-12, 1e-3))
    gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=2, random_state=SEMILLA)
    gp.fit(Xn, np.log(l / (MU0 * r)))
    return _GP.desde_sklearn(gp, lo, hi)


def gp_acople(df, lo, hi):
    Xn = 2 * (F_acople(df) - lo) / (hi - lo) - 1
    kern = (ConstantKernel(1.0) * RBF([1.0] * 5, [(1e-2, 1e2)] * 4 + [(ESCALA_D_MIN, 1e2)])
            + WhiteKernel(1e-8, (1e-12, 1e-3)))
    gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=0, random_state=SEMILLA)
    gp.fit(Xn, df.k.values)
    print(f"      kernel: {gp.kernel_}")
    return _GP.desde_sklearn(gp, lo, hi)


class MLPAcople:
    """Red neuronal en PyTorch para k. Objetivo: log(1-k) (da precisión relativa en L total,
    que depende de 1-k). La derivada respecto de d se obtiene por autograd."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def _x(self, F):
        import torch
        return torch.tensor(2 * (F - self.lo) / (self.hi - self.lo) - 1, dtype=torch.float64)

    def entrenar(self, df, epocas_adam=4000, iter_lbfgs=3000):
        import torch
        torch.manual_seed(SEMILLA)
        X = self._x(F_acople(df))
        y = np.log(1 - df.k.values)
        self.mu, self.sd = y.mean(), y.std()
        Y = torch.tensor((y - self.mu) / self.sd, dtype=torch.float64)[:, None]
        self.red = torch.nn.Sequential(
            torch.nn.Linear(5, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 1)).double()
        opt = torch.optim.Adam(self.red.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas_adam)
        for ep in range(epocas_adam):                    # lote completo: son ~2.000 filas
            opt.zero_grad()
            loss = torch.mean((self.red(X) - Y) ** 2)
            loss.backward()
            opt.step()
            sched.step()
            if ep % 1000 == 0:
                print(f"      Adam {ep:5d}: MSE = {loss.item():.3e}")
        opt = torch.optim.LBFGS(self.red.parameters(), lr=1, max_iter=iter_lbfgs, tolerance_grad=1e-12,
                                tolerance_change=1e-16, history_size=50, line_search_fn="strong_wolfe")

        def cierre():
            opt.zero_grad()
            l = torch.mean((self.red(X) - Y) ** 2)
            l.backward()
            return l
        opt.step(cierre)
        print(f"      L-BFGS final: MSE = {cierre().item():.3e}")
        return self

    def __call__(self, F, gradiente=False):
        import torch
        X = self._x(F).requires_grad_(gradiente)
        z = self.red(X)
        k = 1 - torch.exp(z * self.sd + self.mu)
        if not gradiente:
            return k.detach().numpy().ravel()
        g, = torch.autograd.grad(k.sum(), X)
        g = g.numpy() * 2 / (self.hi - self.lo)
        return k.detach().numpy().ravel(), g


def errores_k(modelo, df_k, F, gradiente_ref=None, h=None, s=None, d=None):
    if gradiente_ref is None:
        k = modelo(F)
        return dict(k_max=float(np.abs(k - df_k).max()), k_medio=float(np.abs(k - df_k).mean()))
    k, g = modelo(F, gradiente=True)
    der = dk_dd(g[:, 4], d, h) * s
    er = np.abs(der - gradiente_ref) / np.abs(gradiente_ref).max()
    return dict(k_max=float(np.abs(k - df_k).max()), k_medio=float(np.abs(k - df_k).mean()),
                fuerza_err_max=float(er.max()), fuerza_err_medio=float(er.mean()))


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main():
    for p in (SALIDA, MODELOS):
        p.mkdir(parents=True, exist_ok=True)
    tr_full, te = cargar()
    rng = np.random.default_rng(SEMILLA)
    ids = sorted(set(tr_full.geom_id) - {0})                  # la del paper queda siempre en train
    val_ids = set(rng.choice(ids, N_VALIDACION, replace=False).tolist())
    tr = tr_full[~tr_full.geom_id.isin(val_ids)]
    va = tr_full[tr_full.geom_id.isin(val_ids)]
    print(f"Geometrías: train {tr.geom_id.nunique()} | validación {va.geom_id.nunique()} | "
          f"prueba {te.geom_id.nunique()}  ({len(tr_full)} + {len(te)} filas)")

    # ---------------- 1) inductancia propia ----------------
    print("=" * 74)
    print("1) Inductancia propia ℓ/(μ0 r) = g(t/r, h/r)  [GP, una función para ambas bobinas]")
    gp_p = gp_propia(*datos_propia(tr_full))
    r, t, h, l = datos_propia(te)
    e_p = np.abs(MU0 * r * np.exp(gp_p(caract_propia(r, t, h))) / l - 1)
    print(f"   prueba ({len(l)} bobinas): error medio {e_p.mean():.2e}, máximo {e_p.max():.2e}")

    # ---------------- 2) acoplamiento: elegir modelo con validación ----------------
    print("=" * 74)
    print("2) Acoplamiento k: selección de modelo con el conjunto de validación")
    Ftr = F_acople(tr_full)
    lo, hi = Ftr.min(0), Ftr.max(0)
    candidatos = {}
    t0 = time.time()
    print("   GP (escala en d acotada)...")
    candidatos["GP"] = gp_acople(tr, lo, hi)
    print(f"      {time.time() - t0:.0f} s")
    try:
        import torch  # noqa: F401
        t0 = time.time()
        print("   MLP (PyTorch)...")
        candidatos["MLP"] = MLPAcople(lo, hi).entrenar(tr)
        print(f"      {time.time() - t0:.0f} s")
    except (ImportError, OSError) as e:
        print(f"   (PyTorch no está disponible: se compara solo el GP)  [{type(e).__name__}]")
    res_val = {n: errores_k(m, va.k.values, F_acople(va)) for n, m in candidatos.items()}
    for n, e in res_val.items():
        print(f"   validación {n:4s}: |Δk| medio {e['k_medio']:.2e}, máximo {e['k_max']:.2e}")
    elegido = min(res_val, key=lambda n: res_val[n]["k_max"])
    print(f"   -> modelo elegido: {elegido}")

    # ---------------- 3) reentrenar el elegido con todo train y evaluar en prueba ----------------
    print("=" * 74)
    print(f"3) Reentreno {elegido} con train + validación y evalúo en PRUEBA")
    finales = {}
    for n in candidatos:
        t0 = time.time()
        finales[n] = gp_acople(tr_full, lo, hi) if n == "GP" else MLPAcople(lo, hi).entrenar(tr_full)
        print(f"   {n}: {time.time() - t0:.0f} s")
    print("   referencia analítica en d intermedios (unos segundos)...")
    mid = referencia_intermedia(te)
    s_mid = np.sqrt(mid.l11 * mid.l22).values
    tabla = []
    for n, m in finales.items():
        eg = errores_k(m, te.k.values, F_acople(te))
        em = errores_k(m, mid.k.values, F_acople(mid), mid.dl12_dd.values, mid.h.values, s_mid, mid.d_m.values)
        kp = m(F_acople(te))
        s = np.sqrt(te.l11_uH * te.l22_uH).values
        lt = te.l11_uH.values + te.l22_uH.values - 2 * kp * s
        el = np.abs(lt / te.ltot_uH.values - 1)
        fila = dict(modelo=n, k_grilla_max=eg["k_max"], k_intermedio_max=em["k_max"],
                    Ltotal_grilla_medio_pct=100 * el.mean(), Ltotal_grilla_max_pct=100 * el.max(),
                    fuerza_err_max_pct=100 * em["fuerza_err_max"], fuerza_err_medio_pct=100 * em["fuerza_err_medio"])
        tabla.append(fila)
    k_an = (te.l12_an_uH / np.sqrt(te.l11_an_uH * te.l22_an_uH)).values
    piso = float(np.abs(k_an - te.k.values).max())
    print(f"\n   Piso de comparación: |k analítico − k Elmer| máx = {piso:.1e} "
          "(en d intermedios la referencia es el analítico)")
    print("   modelo | |Δk| grilla | |Δk| intermedio | L total medio / máx | fuerza máx / medio (% del pico)")
    for f in tabla:
        print(f"   {f['modelo']:6s} | {f['k_grilla_max']:.1e}     | {f['k_intermedio_max']:.1e}         | "
              f"{f['Ltotal_grilla_medio_pct']:.3f} % / {f['Ltotal_grilla_max_pct']:.2f} %  | "
              f"{f['fuerza_err_max_pct']:.2f} % / {f['fuerza_err_medio_pct']:.2f} %")
    pd.DataFrame(tabla).to_csv(SALIDA / "metricas_prueba.csv", index=False)

    # ---------------- 4) guardar el surrogate de producción ----------------
    rangos = {k: [float(v[0]), float(v[1])] for k, v in FD.RANGOS.items()}
    rangos["d_sobre_h"] = [float(FD.D_SOBRE_H.min()), float(FD.D_SOBRE_H.max())]
    meta = dict(modelo_acople=elegido, metricas_prueba=tabla, metricas_validacion=res_val,
                piso_analitico_elmer=piso, error_propia_max=float(e_p.max()))
    if elegido == "GP":
        sur = SurrogateFCL(gp_p, finales["GP"], rangos, meta)
        sur.guardar(MODELOS / "surrogate_fcl.npz")
        sur = SurrogateFCL.cargar(MODELOS / "surrogate_fcl.npz")       # prueba de ida y vuelta
    else:
        import torch
        torch.save(dict(estado=finales["MLP"].red.state_dict(), lo=lo, hi=hi,
                        mu=finales["MLP"].mu, sd=finales["MLP"].sd), MODELOS / "mlp_acople.pt")
        sur = SurrogateFCL(gp_p, finales["GP"] if "GP" in finales else None, rangos, meta)
        sur.guardar(MODELOS / "surrogate_fcl.npz")
        print("   ⚠ ganó el MLP: guardado en mlp_acople.pt (el backend usa el GP del .npz)")
    if "MLP" in finales:
        import torch
        torch.save(dict(estado=finales["MLP"].red.state_dict(), lo=lo, hi=hi,
                        mu=finales["MLP"].mu, sd=finales["MLP"].sd), MODELOS / "mlp_acople.pt")
    (SALIDA / "resumen.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n   Surrogate guardado en {MODELOS / 'surrogate_fcl.npz'}")

    # ---------------- 5) figuras ----------------
    print("=" * 74)
    print("5) Figuras")
    mk = finales[elegido]
    # 5a) paridad k y error en L total (prueba en grilla)
    kp = mk(F_acople(te))
    s = np.sqrt(te.l11_uH * te.l22_uH).values
    lt = te.l11_uH.values + te.l22_uH.values - 2 * kp * s
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=DPI)
    a1.scatter(te.k, kp, s=6)
    a1.plot([0, 1], [0, 1], "k--", lw=1)
    a1.set(xlabel="k Elmer", ylabel=f"k surrogate ({elegido})", title="Paridad en geometrías de prueba")
    a1.grid(alpha=0.3)
    sc = a2.scatter(te.d_sobre_h, 100 * (lt / te.ltot_uH.values - 1), c=te.geom_id, s=8, cmap="tab20")
    a2.axhline(0, color="k", lw=0.8)
    a2.set(xlabel="d / h", ylabel="error en L total [%]", title="Error en L total vs posición")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    guardar_fig(fig, SALIDA / "paridad_prueba.png")

    # 5b) error en d intermedios y en la fuerza
    km, gm = mk(F_acople(mid), gradiente=True)
    der = dk_dd(gm[:, 4], mid.d_m.values, mid.h.values) * s_mid
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=DPI)
    a1.scatter(mid.d_sobre_h, km - mid.k, s=8, label="d intermedios (vs analítico)")
    a1.scatter(te.d_sobre_h, kp - te.k, s=8, marker="x", label="grilla (vs Elmer)")
    a1.axhspan(-piso, piso, color="0.88", zorder=0, label="piso analítico−Elmer")
    a1.set(xlabel="d / h", ylabel="Δk", title="Error de k dentro y entre puntos de la grilla")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a2.scatter(mid.dl12_dd * 1e6, der * 1e6, s=6)
    m = [mid.dl12_dd.min() * 1e6, 0]
    a2.plot(m, m, "k--", lw=1)
    a2.set(xlabel="dℓ12/dd analítico [µH/m]", ylabel="dℓ12/dd surrogate [µH/m]",
           title="Derivada (fuerza por I² y por vuelta²)")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    guardar_fig(fig, SALIDA / "intermedios_y_fuerza.png")

    # 5c) geometría del paper: curva continua
    gp_ = FD.completar(FD.PAPER)
    dd = np.linspace(0, 0.8, 401)
    cur = sur.inductancias(gp_["ri_int"], gp_["t_int"], gp_["gap"], gp_["t_ext"], gp_["h"], dd, N_PAPER, N_PAPER)
    pe = tr_full[tr_full.geom_id == 0]
    N2 = N_PAPER ** 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=DPI)
    a1.plot(dd * 1e3, cur["L_total"] * 1e3, "-", label="Surrogate (continuo)")
    a1.plot(pe.d_m * 1e3, pe.ltot_uH * N2 / 1e3, "o", ms=4, label="Elmer")
    a1.plot(PAPER_D, PAPER_L, "ks", mfc="none", label="Paper (Tabla IV)")
    a1.set(xlabel="d [mm]", ylabel="L total [mH]  (N = 24)", title="Geometría del paper")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(dd * 1e3, cur["F_sobre_I2"] * 1e3, "-", label="Surrogate (derivada exacta)")
    a2.plot(pe.d_m * 1e3, -pe.dl12_dd_uH_m * 1e-6 * N2 * 1e3, "o", ms=4, label="Elmer (spline)")
    a2.set(xlabel="d [mm]", ylabel="F / I² [mN/A²]", title="Fuerza de repulsión por unidad de corriente²")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    guardar_fig(fig, SALIDA / "paper_surrogate.png")

    # 5d) velocidad
    G = np.tile([gp_[c] for c in ("ri_int", "t_int", "gap", "t_ext", "h")], (10000, 1))
    t0 = time.time()
    sur.predecir(*G.T, d=np.linspace(0, 0.8, 10000))
    dt = time.time() - t0
    print(f"   10.000 evaluaciones en {dt:.2f} s  ({dt / 1e4 * 1e6:.0f} µs c/u; Elmer: varios s c/u)")
    print(f"\nListo. Figuras en {SALIDA}, modelo en {MODELOS}")


if __name__ == "__main__":
    main()
