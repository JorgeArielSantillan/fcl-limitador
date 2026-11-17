# -*- coding: utf-8 -*-
"""
fcl_dataset.py
==============
Dataset multigeometría para el surrogate del limitador de corriente de falla de núcleo de aire
(FCL-ACVSR, Amini, Damaki Aliabad y Amiri, IEEE Access 2021).

Modelo magnetostático AXISIMÉTRICO (x = r, y = z) con GMSH + Elmer FEM.

ETAPAS
  0) Referencia analítica (filamentos + integrales elípticas)
  1) Refinamiento de malla por ENERGÍA sobre la geometría del paper (+ extrapolación de Richardson)
  2) Tamaño de la caja de aire (truncamiento del dominio)
  3) Verificación de la malla elegida en geometrías extremas (la sube de nivel si no alcanza)
  4) Dataset: hipercubo latino de geometrías x barrido de la distancia d entre centros
       - conjunto de ENTRENAMIENTO y conjunto de PRUEBA independiente (otra semilla)
  5) Figuras y GIFs del proceso

SALIDAS POR TURNO² (ℓ = L / N²)
  Con densidad de corriente uniforme en la sección (bobinado homogeneizado):
      L11 = N1² ℓ11,   L22 = N2² ℓ22,   L12 = N1 N2 ℓ12
  Así el número de vueltas NO es una variable de la simulación: se elige después.

  Por cada geometría se resuelven dos excitaciones:
      "int": corriente en la bobina interior -> ℓ11 (energía) y ℓ12 (flujo concatenado por la exterior)
      "ext": corriente en la bobina exterior -> ℓ22 y ℓ21 (chequeo de reciprocidad ℓ12 = ℓ21)

Requisitos:
    pip install gmsh numpy scipy matplotlib pillow
    Elmer FEM (ElmerSolver y ElmerGrid en el PATH, o completar ELMER_BIN)
    Opcional: pip install meshio

Se ejecuta una sola vez:  python fcl_dataset.py
Si se corta, volvés a ejecutarlo y sigue desde la última geometría terminada.
"""
import os
import sys
import csv
import json
import time
import shutil
import traceback
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle
from scipy.special import ellipk, ellipe
from scipy.stats import qmc
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
SALIDA = Path(__file__).resolve().parent / "fcl_dataset_out"
ELMER_BIN = ""                 # ej. r"C:\Program Files\Elmer 9.0-Release\bin"
N_PROCESOS = max(1, (os.cpu_count() or 2) - 1)
SOLVER_LINEAL = "directo"      # "directo" (Umfpack) o "iterativo" (BiCGStab + ILU)

# Geometría del paper (Tabla III), metros. Mismo alto para las dos bobinas.
PAPER = dict(ri_int=0.530, t_int=0.030, gap=0.010, t_ext=0.030, h=0.400)
N_PAPER = 24

# Rangos del muestreo (hipercubo latino), metros
RANGOS = dict(
    ri_int=(0.30, 0.80),   # radio interno de la bobina interior
    t_int=(0.015, 0.060),  # espesor radial de la bobina interior
    gap=(0.005, 0.030),    # separación radial entre bobinas
    t_ext=(0.015, 0.060),  # espesor radial de la bobina exterior
    h=(0.20, 0.60),        # altura de las bobinas
)
N_GEOM_TRAIN, SEMILLA_TRAIN = 120, 2026
N_GEOM_TEST, SEMILLA_TEST = 20, 7

# Barrido de distancia entre centros, normalizado por la altura: d = (d/h) * h
D_SOBRE_H = np.round(np.linspace(0.0, 2.0, 21), 3)

# Refinamiento de malla. n_div = elementos a lo largo del espesor de la bobina más delgada;
# crecimiento = cuánto aumenta el tamaño de elemento por metro de distancia a las bobinas.
NIVELES = [(2, 0.40), (3, 0.35), (4, 0.30), (6, 0.25), (8, 0.20),
           (12, 0.15), (16, 0.12), (24, 0.10)]
TOL = 1e-3                      # 0,1 % de cambio relativo entre niveles
D_CONV = 0.100                  # [m] posición inicial del paper
FACTOR_CAJA_CONV = 12           # caja de aire usada durante el refinamiento
FACTORES_CAJA = [4, 6, 8, 12, 16, 24]

# Almacenamiento: .geo, .sif y logs se guardan siempre. Mallas y .vtu:
GUARDAR_MALLAS = "muestra"      # "muestra" (paper + primeras N_MUESTRA) o "todas" (muchos GB)
N_MUESTRA = 3
DPI = 110


def guardar_fig(fig, destino, intentos=8):
    """savefig con reintentos: en Windows el antivirus, OneDrive o un visor pueden
    bloquear el archivo un instante (OSError Errno 22 / 13). Si no se puede, avisa y sigue."""
    for k in range(intentos):
        try:
            fig.savefig(destino)
            break
        except OSError as e:
            if k == intentos - 1:
                print(f"   ⚠ no pude guardar {destino}: {e}")
            time.sleep(0.4 * (k + 1))
    plt.close(fig)
MU0 = 4e-7 * np.pi
I_TEST = 1.0


# =============================================================================
# GEOMETRÍA
# =============================================================================
def completar(g):
    """Agrega radios derivados a un dict con ri_int, t_int, gap, t_ext, h."""
    g = dict(g)
    g = {k: round(float(v), 4) for k, v in g.items()}      # resolución de 0,1 mm
    g["ro_int"] = round(g["ri_int"] + g["t_int"], 4)
    g["ri_ext"] = round(g["ro_int"] + g["gap"], 4)
    g["ro_ext"] = round(g["ri_ext"] + g["t_ext"], 4)
    return g


def bobinas(g, d):
    """Rectángulos (r0, r1, z0, z1): interior arriba (+d/2), exterior abajo (-d/2)."""
    h = g["h"]
    return ((g["ri_int"], g["ro_int"], d / 2 - h / 2, d / 2 + h / 2),
            (g["ri_ext"], g["ro_ext"], -d / 2 - h / 2, -d / 2 + h / 2))


def tamano_caja(g, d, factor):
    return factor * max(g["ro_ext"], d / 2 + g["h"] / 2)


# =============================================================================
# REFERENCIA ANALÍTICA (por turno²)
# =============================================================================
def _M_espiras(a, b, dz):
    k2 = 4 * a * b / ((a + b) ** 2 + dz ** 2)
    k = np.sqrt(k2)
    return MU0 * np.sqrt(a * b) * ((2 / k - k) * ellipk(k2) - 2 / k * ellipe(k2))


def _fil(r0, r1, z0, z1, nr=10, nz=60):
    dr, dz = (r1 - r0) / nr, (z1 - z0) / nz
    R, Z = np.meshgrid(r0 + dr * (np.arange(nr) + .5), z0 + dz * (np.arange(nz) + .5))
    return R.ravel(), Z.ravel(), dr, dz


def l_propia_an(r0, r1, h):
    R, Z, dr, dz = _fil(r0, r1, 0, h)
    with np.errstate(all="ignore"):
        M = _M_espiras(R[:, None], R[None, :], Z[:, None] - Z[None, :])
    np.fill_diagonal(M, MU0 * R * (np.log(8 * R / (0.2235 * (dr + dz))) - 2))
    return M.sum() / len(R) ** 2


def l_mutua_an(g, d):
    bi, be = bobinas(g, d)
    R1, Z1, _, _ = _fil(*bi)
    R2, Z2, _, _ = _fil(*be)
    return _M_espiras(R1[:, None], R2[None, :], Z1[:, None] - Z2[None, :]).mean()


# =============================================================================
# GMSH
# =============================================================================
def escribir_geo(path, g, d, n_div, crec, factor):
    bi, be = bobinas(g, d)
    lcb = min(g["t_int"], g["t_ext"]) / n_div
    lcg = min(lcb, g["gap"] / 3)                    # al menos 3 elementos en la separación
    Rc = tamano_caja(g, d, factor)
    lca = Rc / 12
    dist_max = (lca - lcb) / crec
    zg0, zg1 = min(bi[2], be[2]), max(bi[3], be[3])
    txt = f"""// FCL-ACVSR axisimétrico (x = r, y = z) - generado por fcl_dataset.py
// ri_int={g['ri_int']:.4f} t_int={g['t_int']:.4f} gap={g['gap']:.4f} t_ext={g['t_ext']:.4f} h={g['h']:.4f}
// d={d:.4f} m | n_div={n_div} | crecimiento={crec} | caja={factor}x -> {Rc:.3f} m
lcb = {lcb:.6g}; lca = {lca:.6g}; Rc = {Rc:.6g};
Point(1) = {{0, -Rc, 0, lca}};  Point(2) = {{Rc, -Rc, 0, lca}};
Point(3) = {{Rc,  Rc, 0, lca}};  Point(4) = {{0,  Rc, 0, lca}};
Line(1) = {{1, 2}}; Line(2) = {{2, 3}}; Line(3) = {{3, 4}}; Line(4) = {{4, 1}};
// bobina interior
Point(5) = {{{bi[0]}, {bi[2]}, 0, lcb}}; Point(6) = {{{bi[1]}, {bi[2]}, 0, lcb}};
Point(7) = {{{bi[1]}, {bi[3]}, 0, lcb}}; Point(8) = {{{bi[0]}, {bi[3]}, 0, lcb}};
Line(5) = {{5, 6}}; Line(6) = {{6, 7}}; Line(7) = {{7, 8}}; Line(8) = {{8, 5}};
// bobina exterior
Point(9)  = {{{be[0]}, {be[2]}, 0, lcb}}; Point(10) = {{{be[1]}, {be[2]}, 0, lcb}};
Point(11) = {{{be[1]}, {be[3]}, 0, lcb}}; Point(12) = {{{be[0]}, {be[3]}, 0, lcb}};
Line(9) = {{9, 10}}; Line(10) = {{10, 11}}; Line(11) = {{11, 12}}; Line(12) = {{12, 9}};
Curve Loop(1) = {{1, 2, 3, 4}}; Curve Loop(2) = {{5, 6, 7, 8}}; Curve Loop(3) = {{9, 10, 11, 12}};
Plane Surface(1) = {{1, 2, 3}}; Plane Surface(2) = {{2}}; Plane Surface(3) = {{3}};
Physical Surface("aire", 1) = {{1}};
Physical Surface("bobina_interior", 2) = {{2}};
Physical Surface("bobina_exterior", 3) = {{3}};
Physical Curve("eje", 1) = {{4}};
Physical Curve("borde_exterior", 2) = {{1, 2, 3}};
// tamaño: uniforme en bobinas y separación, crece linealmente hacia afuera
Field[1] = Distance; Field[1].CurvesList = {{5:12}}; Field[1].Sampling = 1500;
Field[2] = Threshold; Field[2].InField = 1; Field[2].SizeMin = lcb; Field[2].SizeMax = lca;
Field[2].DistMin = 0; Field[2].DistMax = {dist_max:.6g};
Field[3] = Box; Field[3].VIn = lcb; Field[3].VOut = lca;
Field[3].XMin = {bi[0]}; Field[3].XMax = {bi[1]}; Field[3].YMin = {bi[2]}; Field[3].YMax = {bi[3]};
Field[4] = Box; Field[4].VIn = lcb; Field[4].VOut = lca;
Field[4].XMin = {be[0]}; Field[4].XMax = {be[1]}; Field[4].YMin = {be[2]}; Field[4].YMax = {be[3]};
Field[5] = Box; Field[5].VIn = {lcg:.6g}; Field[5].VOut = lca;
Field[5].XMin = {bi[1]}; Field[5].XMax = {be[0]}; Field[5].YMin = {zg0}; Field[5].YMax = {zg1};
Field[6] = Min; Field[6].FieldsList = {{2, 3, 4, 5}};
Background Field = 6;
Mesh.MeshSizeExtendFromBoundary = 0; Mesh.MeshSizeFromPoints = 0; Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6; Mesh.ElementOrder = 1; Mesh.MshFileVersion = 2.2;
"""
    Path(path).write_text(txt, encoding="utf-8")


def mallar(geo, msh):
    import gmsh
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(geo))
        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh))
        tags, coords, _ = gmsh.model.mesh.getNodes()
        _, ntri = gmsh.model.mesh.getElementsByType(2)
    finally:
        gmsh.finalize()
    tags = np.asarray(tags, np.int64)
    mapa = np.zeros(tags.max() + 1, np.int64)
    mapa[tags] = np.arange(len(tags))
    return np.asarray(coords).reshape(-1, 3)[:, :2], mapa[np.asarray(ntri, np.int64)].reshape(-1, 3)


# =============================================================================
# ELMER
# =============================================================================
def escribir_sif(path, J_int, J_ext, nombre_salida):
    if SOLVER_LINEAL == "directo":
        lin = "  Linear System Solver = Direct\n  Linear System Direct Method = Umfpack\n"
    else:
        lin = ("  Linear System Solver = Iterative\n  Linear System Iterative Method = BiCGStab\n"
               "  Linear System Preconditioning = ILU1\n  Linear System Max Iterations = 10000\n"
               "  Linear System Convergence Tolerance = 1.0e-12\n")
    Path(path).write_text(f"""! FCL-ACVSR magnetostática axisimétrica - generado por fcl_dataset.py
! J interior = {J_int:.6e} A/m2 | J exterior = {J_ext:.6e} A/m2 (1 A, 1 vuelta)
Header
  CHECK KEYWORDS Warn
  Mesh DB "." "mesh"
End
Simulation
  Max Output Level = 3
  Coordinate System = "Axi Symmetric"
  Simulation Type = Steady state
  Steady State Max Iterations = 1
  Output Intervals(1) = 1
End
Constants
  Permeability of Vacuum = {MU0:.12e}
End
Body 1
  Target Bodies(1) = 1
  Equation = 1
  Material = 1
End
Body 2
  Target Bodies(1) = 2
  Equation = 1
  Material = 1
  Body Force = 1
End
Body 3
  Target Bodies(1) = 3
  Equation = 1
  Material = 1
  Body Force = 2
End
Material 1
  Relative Permeability = 1.0
End
Equation 1
  Active Solvers(2) = 1 2
End
Solver 1
  Equation = "MgDyn2D"
  Procedure = "MagnetoDynamics2D" "MagnetoDynamics2D"
  Variable = "Potential"
{lin}  Steady State Convergence Tolerance = 1.0e-10
End
Solver 2
  Equation = "Resultados"
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = "{nombre_salida}"
  Output Format = vtu
  Ascii Output = Logical True
End
Body Force 1
  Current Density = Real {J_int:.12e}
End
Body Force 2
  Current Density = Real {J_ext:.12e}
End
Boundary Condition 1
  Target Boundaries(2) = 1 2
  Potential = Real 0.0
End
""", encoding="utf-8")


def correr(cmd, cwd, log):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    Path(cwd, log).write_text(r.stdout + r.stderr, encoding="utf-8")
    return r.returncode, r.stdout + r.stderr


def leer_vtu(path):
    try:
        import meshio
        m = meshio.read(path)
        tri = np.vstack([c.data[:, :3] for c in m.cells if c.type.startswith("triangle")])
        k = [k for k in m.point_data if "potential" in k.lower()][0]
        return m.points[:, :2], tri, np.asarray(m.point_data[k], float).ravel()
    except ImportError:
        pass
    pz = ET.parse(path).getroot().find(".//Piece")

    def arr(da, t=float):
        if da.get("format", "ascii") != "ascii":
            raise RuntimeError("VTU no ASCII: instalá meshio (pip install meshio)")
        return np.array(da.text.split(), dtype=t)

    pts = arr(pz.find("Points/DataArray")).reshape(-1, 3)[:, :2]
    c = {da.get("Name"): da for da in pz.find("Cells")}
    con, off, tip = arr(c["connectivity"], np.int64), arr(c["offsets"], np.int64), arr(c["types"], np.int64)
    ini = np.r_[0, off[:-1]]
    tri = np.array([con[s:s + 3] for s in ini[np.isin(tip, [5, 22])]], np.int64)
    for da in pz.find("PointData"):
        if "potential" in da.get("Name", "").lower():
            return pts, tri, arr(da)
    raise RuntimeError(f"Sin variable Potential en {path}")


def flujos_por_turno(pts, tri, A, g, d):
    """ℓ = (1/S) ∫_S 2π r A dS  para cada bobina (flujo medio concatenado por vuelta, I = 1 A).
    Integración exacta del producto r·A con elementos lineales."""
    P = pts[tri]
    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    rr, aa = pts[tri, 0], A[tri]
    irA = area / 12 * ((rr * aa).sum(1) + rr.sum(1) * aa.sum(1))
    c = P.mean(1)
    out = []
    for r0, r1, z0, z1 in bobinas(g, d):
        m = (c[:, 0] > r0) & (c[:, 0] < r1) & (c[:, 1] > z0) & (c[:, 1] < z1)
        S = (r1 - r0) * (z1 - z0)
        if abs(area[m].sum() / S - 1) > 1e-3:
            raise RuntimeError(f"El área mallada de una bobina no coincide ({area[m].sum():.6g} vs {S:.6g})")
        out.append(2 * np.pi * irA[m].sum() / S / I_TEST)
    return out   # [flujo por vuelta en bobina interior, en bobina exterior]


def resolver_caso(carpeta, g, d, n_div, crec, factor, exe, excitaciones=("int", "ext"),
                  guardar=True, devolver_campo=False):
    """Un caso (geometría + d): .geo -> .msh -> mesh/ -> Elmer por excitación -> ℓ."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    escribir_geo(carpeta / "modelo.geo", g, d, n_div, crec, factor)
    xy, tri_g = mallar(carpeta / "modelo.geo", carpeta / "modelo.msh")
    rc, log = correr([exe["grid"], "14", "2", "modelo.msh", "-autoclean", "-out", "mesh"],
                     carpeta, "elmergrid.log")
    if rc != 0 or not (carpeta / "mesh" / "mesh.header").exists():
        raise RuntimeError(f"ElmerGrid falló en {carpeta}:\n{log[-1500:]}")
    S_int, S_ext = g["t_int"] * g["h"], g["t_ext"] * g["h"]
    res = dict(elementos=len(tri_g), malla=(xy, tri_g))
    for exc in excitaciones:
        J = (I_TEST / S_int, 0.0) if exc == "int" else (0.0, I_TEST / S_ext)
        nombre = f"res_{exc}"
        for v in carpeta.rglob(f"{nombre}*.vtu"):
            v.unlink()
        escribir_sif(carpeta / f"case_{exc}.sif", *J, nombre)
        rc, log = correr([exe["solver"], f"case_{exc}.sif"], carpeta, f"elmer_{exc}.log")
        vtus = sorted(carpeta.rglob(f"{nombre}*.vtu"), key=os.path.getmtime)
        if rc != 0 or not vtus:
            raise RuntimeError(f"ElmerSolver falló en {carpeta} ({exc}):\n" + "\n".join(log.splitlines()[-25:]))
        pts, tri, A = leer_vtu(vtus[-1])
        res[exc] = flujos_por_turno(pts, tri, A, g, d)
        if devolver_campo:
            res[f"campo_{exc}"] = (pts, tri, A)
    if not guardar:
        for p in ["modelo.msh", "mesh"] + [str(v.relative_to(carpeta)) for v in carpeta.rglob("*.vtu")]:
            q = carpeta / p
            shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
    # ℓ11, ℓ12 (de "int"); ℓ22, ℓ21 (de "ext")
    if "int" in res:
        res["l11"], res["l12"] = res["int"]
    if "ext" in res:
        res["l21"], res["l22"] = res["ext"]
    return res


# =============================================================================
# TRABAJADOR: una geometría completa (corre en un proceso aparte)
# =============================================================================
def procesar_geometria(args):
    gid, conjunto, g, n_div, crec, factor, exe, guardar, campo = args
    g = completar(g)
    carpeta = SALIDA / "dataset" / conjunto / f"g{gid:03d}"
    archivo = carpeta / "filas.csv"
    if archivo.exists():
        return gid, conjunto, "ya estaba"
    t0 = time.time()
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "geometria.json").write_text(json.dumps(g, indent=2), encoding="utf-8")
    l11_an = l_propia_an(g["ri_int"], g["ro_int"], g["h"])
    l22_an = l_propia_an(g["ri_ext"], g["ro_ext"], g["h"])
    filas, l11s = [], []
    l22 = l21 = None
    for k, dh in enumerate(D_SOBRE_H):
        d = dh * g["h"]
        exc = ("int", "ext") if (k == 0 or campo) else ("int",)
        r = resolver_caso(carpeta / f"d_{dh:.2f}h", g, d, n_div, crec, factor, exe, exc,
                          guardar=guardar, devolver_campo=campo)
        if k == 0:
            l22, l21 = r["l22"], r["l21"]
        l11s.append(r["l11"])
        if campo:   # campo de la conexión serie (+I interior, -I exterior) para las figuras
            pts, tri, Ai = r["campo_int"]
            Ae = r["campo_ext"][2]
            np.savez_compressed(carpeta / f"campo_d_{dh:.2f}h.npz", pts=pts, tri=tri, A=Ai - Ae, d=d)
        filas.append(dict(conjunto=conjunto, geom_id=gid, **{c: g[c] for c in
                     ("ri_int", "t_int", "gap", "t_ext", "h", "ro_int", "ri_ext", "ro_ext")},
                          d_m=round(d, 6), d_sobre_h=dh, l11_uH=r["l11"] * 1e6, l12_uH=r["l12"] * 1e6,
                          l12_an_uH=l_mutua_an(g, d) * 1e6, elementos=r["elementos"]))
    l11 = float(np.mean(l11s))
    dd = np.array([f["d_m"] for f in filas])
    l12 = np.array([f["l12_uH"] for f in filas])
    dl12 = CubicSpline(dd, l12)(dd, 1)
    for f, der in zip(filas, dl12):
        f.update(l11_uH=l11 * 1e6, l22_uH=l22 * 1e6,
                 ltot_uH=(l11 + l22) * 1e6 - 2 * f["l12_uH"],
                 l11_an_uH=l11_an * 1e6, l22_an_uH=l22_an * 1e6,
                 ltot_an_uH=(l11_an + l22_an) * 1e6 - 2 * f["l12_an_uH"],
                 dl12_dd_uH_m=der,
                 dispersion_l11=float(np.std(l11s) / l11),
                 error_reciprocidad=abs(l21 - filas[0]["l12_uH"] / 1e6) / abs(l21))
    orden = ["conjunto", "geom_id", "ri_int", "t_int", "gap", "t_ext", "h", "ro_int", "ri_ext", "ro_ext",
             "d_m", "d_sobre_h", "l11_uH", "l22_uH", "l12_uH", "ltot_uH", "dl12_dd_uH_m",
             "l11_an_uH", "l22_an_uH", "l12_an_uH", "ltot_an_uH",
             "dispersion_l11", "error_reciprocidad", "elementos"]
    filas = [{k: f[k] for k in orden} for f in filas]
    tmp = carpeta / "filas.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=orden)
        w.writeheader()
        w.writerows(filas)
    tmp.replace(archivo)              # se escribe al final: si se corta, la geometría se rehace
    return gid, conjunto, f"{time.time() - t0:.0f} s"


# =============================================================================
# FIGURAS
# =============================================================================
def guardar_gif(pngs, destino, ms=450):
    """GIF con reintentos; si no se puede crear, avisa y sigue (las PNG quedan igual)."""
    from PIL import Image
    pngs = [p for p in pngs if Path(p).exists()]
    if not pngs:
        return
    cs = [Image.open(p).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=255) for p in pngs]
    for k in range(8):
        try:
            cs[0].save(destino, save_all=True, append_images=cs[1:], duration=ms, loop=0)
            return
        except OSError as e:
            if k == 7:
                print(f"   ⚠ no pude guardar {destino}: {e}")
            time.sleep(0.4 * (k + 1))


def dibujar_bobinas(ax, g, d, color="#d62728", lw=1.2):
    for r0, r1, z0, z1 in bobinas(g, d):
        ax.add_patch(Rectangle((r0, z0), r1 - r0, z1 - z0, fill=False, ec=color, lw=lw))


def fig_malla(xy, tri, g, d, titulo, destino):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.5), dpi=DPI)
    T = mtri.Triangulation(xy[:, 0], xy[:, 1], tri)
    for ax in (a1, a2):
        ax.triplot(T, lw=0.25, color="#1f4e79")
        dibujar_bobinas(ax, g, d)
        ax.set_aspect("equal")
        ax.set_xlabel("r [m]")
    a1.set_xlim(0, xy[:, 0].max())
    a1.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    a1.set_ylabel("z [m]")
    a1.set_title("Dominio completo")
    a2.set_xlim(g["ri_int"] - 0.08, g["ro_ext"] + 0.08)
    a2.set_ylim(-0.32, 0.32)
    a2.set_title("Detalle de las bobinas")
    fig.suptitle(titulo, fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.1, wspace=0.25)
    guardar_fig(fig, destino)


def campo_B(pts, tri, A):
    psi = 2 * np.pi * pts[:, 0] * A
    P = pts[tri]
    x1, y1, x2, y2, x3, y3 = P[:, 0, 0], P[:, 0, 1], P[:, 1, 0], P[:, 1, 1], P[:, 2, 0], P[:, 2, 1]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    p1, p2, p3 = psi[tri[:, 0]], psi[tri[:, 1]], psi[tri[:, 2]]
    dr = ((p2 - p1) * (y3 - y1) - (p3 - p1) * (y2 - y1)) / det
    dz = ((p3 - p1) * (x2 - x1) - (p2 - p1) * (x3 - x1)) / det
    rc = np.maximum(P[:, :, 0].mean(1), 1e-6)
    return psi, np.hypot(dz / (2 * np.pi * rc), dr / (2 * np.pi * rc))


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def buscar(nombre):
    for c in (nombre, nombre + ".exe"):
        if ELMER_BIN and (Path(ELMER_BIN) / c).exists():
            return str(Path(ELMER_BIN) / c)
        if shutil.which(c):
            return shutil.which(c)
    sys.exit(f"No encuentro {nombre}: instalá Elmer FEM o completá ELMER_BIN.")


def richardson(h, f):
    """Con los tres últimos niveles: f(h) = f∞ + C h^p  ->  (p, f∞)."""
    (h1, h2, h3), (f1, f2, f3) = h[-3:], f[-3:]
    q = (f1 - f2) / (f2 - f3)
    try:
        p = brentq(lambda p: (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - q, 0.3, 8)
    except ValueError:
        return np.nan, np.nan
    C = (f2 - f3) / (h2 ** p - h3 ** p)
    return p, f3 - C * h3 ** p


def escribir_csv(path, filas):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def main():
    try:
        import gmsh  # noqa: F401
    except ImportError:
        sys.exit("Falta gmsh:  pip install gmsh")
    exe = dict(solver=buscar("ElmerSolver"), grid=buscar("ElmerGrid"))
    d_res, d_img = SALIDA / "resultados", SALIDA / "imagenes"
    for p in (d_res, d_img / "malla", d_img / "flujo", d_img / "geometrias", d_img / "caja"):
        p.mkdir(parents=True, exist_ok=True)
    estado_path = d_res / "malla_elegida.json"
    gp = completar(PAPER)
    N2 = N_PAPER ** 2

    print("=" * 74)
    l11a, l22a = l_propia_an(gp["ri_int"], gp["ro_int"], gp["h"]), l_propia_an(gp["ri_ext"], gp["ro_ext"], gp["h"])
    l12a = l_mutua_an(gp, D_CONV)
    print(f"0) Analítico, geometría del paper (N = {N_PAPER}):  L11 = {N2 * l11a * 1e3:.4f}  "
          f"L22 = {N2 * l22a * 1e3:.4f}  L total(100 mm) = {N2 * (l11a + l22a - 2 * l12a) * 1e3:.4f} mH")

    if estado_path.exists():
        est = json.loads(estado_path.read_text(encoding="utf-8"))
        print(f"   Malla ya elegida en una corrida anterior: {est}  (borrá {estado_path.name} para rehacer)")
    else:
        # ---------------- 1) REFINAMIENTO POR ENERGÍA ----------------
        print("=" * 74)
        print(f"1) Refinamiento de malla por energía (geometría del paper, d = {D_CONV * 1e3:.0f} mm, "
              f"caja {FACTOR_CAJA_CONV}x)")
        filas, pngs, elegido = [], [], None
        for k, (nd, cr) in enumerate(NIVELES, 1):
            r = resolver_caso(SALIDA / "01_convergencia" / f"nivel_{k:02d}", gp, D_CONV, nd, cr,
                              FACTOR_CAJA_CONV, exe)
            lt = r["l11"] + r["l22"] - 2 * r["l12"]
            f = dict(nivel=k, n_div=nd, crecimiento=cr, elementos=r["elementos"],
                     lc_bobina_mm=min(gp["t_int"], gp["t_ext"]) / nd * 1e3,
                     W11_uJ=0.5 * r["l11"] * N2 * 1e6, W22_uJ=0.5 * r["l22"] * N2 * 1e6,
                     W_serie_uJ=0.5 * lt * N2 * 1e6, L_total_mH=lt * N2 * 1e3,
                     error_reciprocidad=abs(r["l12"] - r["l21"]) / r["l12"])
            if filas:
                a = filas[-1]
                f["cambio_max"] = max(abs(f[c] / a[c] - 1) for c in ("W11_uJ", "W22_uJ", "W_serie_uJ"))
            else:
                f["cambio_max"] = np.nan
            filas.append(f)
            print(f"   nivel {k}: n_div={nd:2d}  {r['elementos']:7d} elem  W11={f['W11_uJ']:.5f}  "
                  f"W22={f['W22_uJ']:.5f}  Wserie={f['W_serie_uJ']:.5f} µJ  Δmax={f['cambio_max']:.2e}")
            png = d_img / "malla" / f"malla_nivel_{k:02d}.png"
            fig_malla(*r["malla"], gp, D_CONV, f"Nivel {k}: n_div = {nd}, {r['elementos']:,} elementos, "
                      f"W serie = {f['W_serie_uJ']:.4f} µJ (I = 1 A, N = {N_PAPER})".replace(",", "."), png)
            pngs.append(png)
            if elegido is None and f["cambio_max"] < TOL:
                elegido = k - 1        # el nivel anterior ya estaba dentro de la tolerancia
        if elegido is None:
            elegido = len(NIVELES)
            print("   ⚠ no se llegó a la tolerancia: uso el nivel más fino")
        hh = np.array([1 / f["n_div"] for f in filas])
        p, Winf = richardson(hh, np.array([f["W_serie_uJ"] for f in filas]))
        for f in filas:
            f["error_vs_richardson"] = abs(f["W_serie_uJ"] / Winf - 1) if np.isfinite(Winf) else np.nan
        # Criterio principal: error estimado contra la energía extrapolada (más estricto que el
        # cambio entre niveles, que subestima el error). Si Richardson falla, queda el otro.
        if np.isfinite(Winf) and 0.5 < p < 4:
            ok = [f["nivel"] for f in filas if f["error_vs_richardson"] < TOL]
            if ok:
                elegido = ok[0]
        escribir_csv(d_res / "convergencia_energia.csv", filas)
        guardar_gif(pngs, d_res / "refinamiento_malla.gif", 700)
        print(f"   -> nivel elegido: {elegido} (n_div = {NIVELES[elegido - 1][0]})  | Richardson: "
              f"orden p = {p:.2f}, W∞ = {Winf:.5f} µJ, error estimado del elegido = "
              f"{filas[elegido - 1]['error_vs_richardson']:.2e}")

        ne = [f["elementos"] for f in filas]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=DPI)
        a1.semilogx(ne, [f["W_serie_uJ"] for f in filas], "o-", label="W serie")
        if np.isfinite(Winf):
            a1.axhline(Winf, color="k", ls=":", label=f"Richardson (p = {p:.2f})")
        a1.plot(ne[elegido - 1], filas[elegido - 1]["W_serie_uJ"], "r*", ms=16, label="elegido")
        a1.set(xlabel="elementos", ylabel="energía [µJ]  (I = 1 A, N = 24)", title="Energía vs malla")
        a1.legend()
        a1.grid(True, which="both", alpha=0.3)
        a2.loglog(ne[1:], [f["cambio_max"] for f in filas[1:]], "s-", label="cambio máx. entre niveles")
        a2.loglog(ne, [max(f["error_vs_richardson"], 1e-9) for f in filas], "^-", label="error vs Richardson")
        a2.axhline(TOL, color="r", ls="--", label=f"tolerancia {TOL:.1%}")
        a2.set(xlabel="elementos", title="Criterio de convergencia")
        a2.legend(fontsize=8)
        a2.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        guardar_fig(fig, d_res / "convergencia_energia.png")
        n_div, crec = NIVELES[elegido - 1]

        # ---------------- 2) CAJA DE AIRE ----------------
        print("=" * 74)
        print("2) Tamaño de la caja de aire")
        filas_c = []
        for fc in FACTORES_CAJA:
            r = resolver_caso(SALIDA / "02_caja" / f"caja_{fc:02d}x", gp, D_CONV, n_div, crec, fc, exe)
            lt = r["l11"] + r["l22"] - 2 * r["l12"]
            filas_c.append(dict(factor=fc, radio_caja_m=tamano_caja(gp, D_CONV, fc), elementos=r["elementos"],
                                L11_mH=r["l11"] * N2 * 1e3, L22_mH=r["l22"] * N2 * 1e3, L_total_mH=lt * N2 * 1e3))
            print(f"   caja {fc:2d}x ({filas_c[-1]['radio_caja_m']:.2f} m): L11 = {filas_c[-1]['L11_mH']:.5f}  "
                  f"L total = {filas_c[-1]['L_total_mH']:.5f} mH")
            fig_malla(*r["malla"], gp, D_CONV, f"Caja de aire {fc}x ({filas_c[-1]['radio_caja_m']:.2f} m)",
                      d_img / "caja" / f"caja_{fc:02d}x.png")
        ref = filas_c[-1]
        for f in filas_c:
            f["error_vs_mayor"] = max(abs(f[c] / ref[c] - 1) for c in ("L11_mH", "L22_mH", "L_total_mH"))
        factor = next(f["factor"] for f in filas_c if f["error_vs_mayor"] < TOL / 2)
        escribir_csv(d_res / "caja_aire.csv", filas_c)
        print(f"   -> caja elegida: {factor}x")
        fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
        ax.semilogy([f["factor"] for f in filas_c[:-1]], [max(f["error_vs_mayor"], 1e-9) for f in filas_c[:-1]], "o-")
        ax.axhline(TOL / 2, color="r", ls="--", label=f"tolerancia {TOL / 2:.2%}")
        ax.set(xlabel="tamaño de caja / tamaño del dispositivo", ylabel="error vs caja más grande",
               title="Truncamiento del dominio")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        guardar_fig(fig, d_res / "caja_aire.png")

        # ---------------- 3) VERIFICACIÓN EN GEOMETRÍAS EXTREMAS ----------------
        print("=" * 74)
        print("3) Verificación en geometrías extremas")
        lo = {k: v[0] for k, v in RANGOS.items()}
        hi = {k: v[1] for k, v in RANGOS.items()}
        extremas = {
            "delgada_alta_chica": dict(lo, h=hi["h"]),
            "gruesa_baja_grande": dict(hi, h=lo["h"]),
            "delgada_baja_grande": dict(lo, ri_int=hi["ri_int"]),
            "separada_gruesa_chica": dict(hi, ri_int=lo["ri_int"], h=lo["h"]),
        }
        filas_v = []
        nivel = elegido
        while True:
            peor = 0.0
            nd1, cr1 = NIVELES[nivel - 1]
            nd2, cr2 = NIVELES[min(nivel, len(NIVELES) - 1)]
            for nombre, ge in extremas.items():
                ge = completar(ge)
                for dh in (0.25, 1.0):
                    d = dh * ge["h"]
                    base = SALIDA / "03_verificacion" / f"nivel_{nivel:02d}" / f"{nombre}_d{dh:.2f}h"
                    ra = resolver_caso(base / "elegida", ge, d, nd1, cr1, factor, exe)
                    rb = resolver_caso(base / "mas_fina", ge, d, nd2, cr2, factor, exe)
                    lta = ra["l11"] + ra["l22"] - 2 * ra["l12"]
                    ltb = rb["l11"] + rb["l22"] - 2 * rb["l12"]
                    err = max(abs(ra[c] / rb[c] - 1) for c in ("l11", "l22", "l12"))
                    err = max(err, abs(lta / ltb - 1))
                    peor = max(peor, err)
                    filas_v.append(dict(nivel=nivel, geometria=nombre, d_sobre_h=dh, elementos=ra["elementos"],
                                        elementos_mas_fina=rb["elementos"], diferencia_max=err))
                    print(f"   nivel {nivel} {nombre:24s} d={dh:.2f}h  {ra['elementos']:7d} elem  Δ = {err:.2e}")
            if peor < TOL or nivel >= len(NIVELES) - 1:
                break
            nivel += 1
            print(f"   ⚠ alguna geometría supera {TOL:.1%}: subo al nivel {nivel}")
        escribir_csv(d_res / "verificacion_extremas.csv", filas_v)
        n_div, crec = NIVELES[nivel - 1]
        est = dict(nivel=nivel, n_div=n_div, crecimiento=crec, factor_caja=factor,
                   nivel_por_energia=elegido, peor_diferencia_extremas=peor)
        estado_path.write_text(json.dumps(est, indent=2), encoding="utf-8")
        print(f"   -> malla final: {est}")

    # ---------------- 4) DATASET ----------------
    print("=" * 74)
    trabajos = []
    for conjunto, n, sem in (("train", N_GEOM_TRAIN, SEMILLA_TRAIN), ("test", N_GEOM_TEST, SEMILLA_TEST)):
        U = qmc.LatinHypercube(d=len(RANGOS), seed=sem).random(n)
        lo = np.array([v[0] for v in RANGOS.values()])
        hi = np.array([v[1] for v in RANGOS.values()])
        geoms = [dict(zip(RANGOS, lo + u * (hi - lo))) for u in U]
        if conjunto == "train":
            geoms = [dict(PAPER)] + geoms          # g000 = paper
        for gid, g in enumerate(geoms):
            guardar = GUARDAR_MALLAS == "todas" or (conjunto == "train" and gid < N_MUESTRA)
            campo = conjunto == "train" and gid == 0
            trabajos.append((gid, conjunto, g, est["n_div"], est["crecimiento"], est["factor_caja"],
                             exe, guardar, campo))
    pend = [t for t in trabajos if not (SALIDA / "dataset" / t[1] / f"g{t[0]:03d}" / "filas.csv").exists()]
    print(f"4) Dataset: {len(trabajos)} geometrías x {len(D_SOBRE_H)} distancias "
          f"({len(trabajos) - len(pend)} ya hechas), {N_PROCESOS} procesos en paralelo")
    t0, hechas, errores = time.time(), 0, []
    with ProcessPoolExecutor(max_workers=N_PROCESOS) as pool:
        futs = {pool.submit(procesar_geometria, t): t for t in pend}
        for fu in as_completed(futs):
            t = futs[fu]
            hechas += 1
            try:
                gid, cj, msg = fu.result()
                eta = (time.time() - t0) / hechas * (len(pend) - hechas) / 60
                print(f"   [{hechas}/{len(pend)}] {cj} g{gid:03d}: {msg}   (faltan ~{eta:.0f} min)")
            except Exception:
                errores.append(f"{t[1]} g{t[0]:03d}\n{traceback.format_exc()}")
                print(f"   [{hechas}/{len(pend)}] {t[1]} g{t[0]:03d}: ERROR (ver errores.txt)")
    if errores:
        (d_res / "errores.txt").write_text("\n\n".join(errores), encoding="utf-8")

    filas = []
    for cj in ("train", "test"):
        for fcsv in sorted((SALIDA / "dataset" / cj).glob("g*/filas.csv")):
            with open(fcsv, encoding="utf-8") as fh:
                filas += list(csv.DictReader(fh))
    if not filas:
        sys.exit("No hay filas: revisá resultados/errores.txt")
    for cj in ("train", "test"):
        sub = [f for f in filas if f["conjunto"] == cj]
        if sub:
            escribir_csv(d_res / f"dataset_{cj}.csv", sub)
    escribir_csv(d_res / "dataset_completo.csv", filas)
    num = lambda c, s=None: np.array([float(f[c]) for f in (s or filas)])

    # ---------------- 5) FIGURAS ----------------
    print("=" * 74)
    print("5) Figuras y GIFs")
    # 5a) Geometría del paper: Elmer vs analítico vs paper
    paper = [f for f in filas if f["conjunto"] == "train" and f["geom_id"] == "0"]
    if paper:
        dmm = num("d_m", paper) * 1e3
        PD = [100, 100.124, 100.981, 104.178, 112.722, 131.052, 164.627, 218.54, 294.67, 389.5, 496.12, 608.823, 700]
        PL = [0.27697, 0.2772, 0.279, 0.28406, 0.29887, 0.33845, 0.4115, 0.5431, 0.7524, 1.003, 1.18, 1.305, 1.38]
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=DPI)
        ax.plot(dmm, num("ltot_uH", paper) * N2 / 1e3, "o-", label="Elmer")
        ax.plot(dmm, num("ltot_an_uH", paper) * N2 / 1e3, "--", label="Analítico")
        ax.plot(PD, PL, "ks", mfc="none", label="Paper (Tabla IV)")
        ax.set(xlabel="d [mm]", ylabel="L total [mH]  (N = 24)", title="Geometría del paper")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        guardar_fig(fig, d_res / "paper_L_vs_d.png")
        # GIF de líneas de flujo (escala común)
        npzs = sorted((SALIDA / "dataset" / "train" / "g000").glob("campo_d_*.npz"))
        if npzs:
            esc = 360 * np.sqrt(2) * N_PAPER      # corriente nominal pico x vueltas (A = lineal en NI)
            datos = [np.load(p) for p in npzs]
            vmax = max(np.percentile(campo_B(z["pts"], z["tri"], z["A"] * esc)[1], 99.7) for z in datos)
            psimax = max(np.abs(2 * np.pi * z["pts"][:, 0] * z["A"] * esc).max() for z in datos)
            niv = np.linspace(-psimax, psimax, 32)
            niv = niv[np.abs(niv) > 1e-9 * psimax]
            pngs = []
            for z, f in zip(datos, paper):
                pts, A, d = z["pts"], z["A"] * esc, float(z["d"])
                c = pts[z["tri"]].mean(1)
                tri = z["tri"][(c[:, 0] < 1.6) & (np.abs(c[:, 1]) < 1.3)]
                psi, Bm = campo_B(pts, tri, A)
                T = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
                fig, ax = plt.subplots(figsize=(6.4, 7.2), dpi=DPI)
                pc = ax.tripcolor(T, facecolors=Bm * 1e3, cmap="magma", vmin=0, vmax=vmax * 1e3)
                ax.tricontour(T, psi, levels=niv, colors="w", linewidths=0.6)
                dibujar_bobinas(ax, gp, d)
                ax.set(xlim=(0, 1.4), ylim=(-1.2, 1.2), xlabel="r [m]", ylabel="z [m]",
                       title=f"d = {d * 1e3:.0f} mm   L total = {float(f['ltot_uH']) * N2 / 1e3:.3f} mH")
                ax.set_aspect("equal")
                fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03).set_label("|B| [mT]  (509 A pico)")
                fig.subplots_adjust(left=0.12, right=0.88, top=0.94, bottom=0.07)
                png = d_img / "flujo" / f"flujo_d_{d * 1e3:04.0f}mm.png"
                guardar_fig(fig, png)
                pngs.append(png)
            guardar_gif(pngs + pngs[::-1], d_res / "lineas_flujo_vs_d.gif", 220)

    # 5b) GIF de geometrías del dataset: corte a escala + curva ℓ_total(d)
    pngs = []
    ids = sorted({int(f["geom_id"]) for f in filas if f["conjunto"] == "train"})[:30]
    for gid in ids:
        s = [f for f in filas if f["conjunto"] == "train" and int(f["geom_id"]) == gid]
        g = completar({c: float(s[0][c]) for c in RANGOS})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=DPI)
        dibujar_bobinas(a1, g, 0.0, "#1f77b4")
        dibujar_bobinas(a1, g, 2 * g["h"], "#d62728")
        a1.axvline(0, color="k", lw=0.8)
        a1.set(xlim=(0, 1.0), ylim=(-1.0, 1.0), xlabel="r [m]", ylabel="z [m]",
               title=f"g{gid:03d}: d = 0 (azul) y d = 2h (rojo)")
        a1.set_aspect("equal")
        a2.plot(num("d_sobre_h", s), num("ltot_uH", s), "o-", label="Elmer")
        a2.plot(num("d_sobre_h", s), num("ltot_an_uH", s), "--", label="Analítico")
        a2.set(xlabel="d / h", ylabel="ℓ total [µH / vuelta²]", ylim=(0, None),
               title=f"ri={g['ri_int']:.2f}  t={g['t_int'] * 1e3:.0f}/{g['t_ext'] * 1e3:.0f} mm  "
                     f"gap={g['gap'] * 1e3:.0f} mm  h={g['h']:.2f}")
        a2.legend()
        a2.grid(alpha=0.3)
        fig.tight_layout()
        png = d_img / "geometrias" / f"g{gid:03d}.png"
        guardar_fig(fig, png)
        pngs.append(png)
    if pngs:
        guardar_gif(pngs, d_res / "geometrias_dataset.gif", 600)

    # 5c) Cobertura del espacio de diseño (train vs test)
    tr = [f for f in filas if f["conjunto"] == "train" and float(f["d_sobre_h"]) == 0]
    te = [f for f in filas if f["conjunto"] == "test" and float(f["d_sobre_h"]) == 0]
    claves = list(RANGOS)
    fig, axs = plt.subplots(len(claves), len(claves), figsize=(11, 11), dpi=90)
    for i, ci in enumerate(claves):
        for j, cj in enumerate(claves):
            ax = axs[i, j]
            if i == j:
                ax.hist(num(ci, tr), bins=12, color="C0", alpha=0.7)
            else:
                ax.scatter(num(cj, tr), num(ci, tr), s=8, c="C0", label="train")
                if te:
                    ax.scatter(num(cj, te), num(ci, te), s=14, c="C3", marker="x", label="test")
            if i == len(claves) - 1:
                ax.set_xlabel(cj)
            if j == 0:
                ax.set_ylabel(ci)
            ax.tick_params(labelsize=6)
    axs[0, 1].legend(fontsize=7)
    fig.suptitle("Cobertura del espacio de diseño (hipercubo latino)")
    fig.tight_layout()
    guardar_fig(fig, d_res / "cobertura_dataset.png")

    # 5d) Paridad Elmer vs analítico (todo el dataset) + calidad
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=DPI)
    x, y = num("ltot_an_uH"), num("ltot_uH")
    a1.scatter(x, y, s=4, alpha=0.5)
    m = [0, max(x.max(), y.max())]
    a1.plot(m, m, "k--", lw=1)
    a1.set(xlabel="ℓ total analítico [µH/vuelta²]", ylabel="ℓ total Elmer [µH/vuelta²]",
           title="Paridad Elmer vs analítico")
    a1.grid(alpha=0.3)
    err = (y - x) / x
    a2.hist(err * 100, bins=50)
    a2.set(xlabel="(Elmer − analítico) / analítico [%]", ylabel="filas",
           title=f"mediana {np.median(err) * 100:.2f} %, máx |{np.abs(err).max() * 100:.2f}| %")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    guardar_fig(fig, d_res / "paridad_elmer_analitico.png")

    resumen = dict(malla=est, filas=len(filas),
                   geometrias=dict(train=len({f["geom_id"] for f in filas if f["conjunto"] == "train"}),
                                   test=len({f["geom_id"] for f in filas if f["conjunto"] == "test"})),
                   dispersion_l11_max=float(num("dispersion_l11").max()),
                   error_reciprocidad_max=float(num("error_reciprocidad").max()),
                   diferencia_elmer_analitico_mediana=float(np.median(err)),
                   errores=len(errores))
    (d_res / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nListo. Dataset en: {d_res}")


if __name__ == "__main__":
    main()
