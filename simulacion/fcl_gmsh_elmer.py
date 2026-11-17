# -*- coding: utf-8 -*-
"""
fcl_gmsh_elmer.py
=================
Limitador de corriente de falla de núcleo de aire (FCL-ACVSR)
Basado en: Amini, Damaki Aliabad y Amiri, IEEE Access, 2021 (DOI 10.1109/ACCESS.2021.3134870)

Modelo magnetostático AXISIMÉTRICO (x = r, y = z) con GMSH + Elmer FEM:
  1) Estudio de convergencia de malla por ENERGÍA MAGNÉTICA -> elige la malla óptima
  2) Barrido de la distancia d entre centros de bobinas -> L11, L22, L12 y L total
  3) Comparación con el método analítico (filamentos + integrales elípticas) y con el paper
  4) Imágenes del proceso y GIFs (refinamiento de malla y líneas de flujo vs d)

Se guardan todos los archivos intermedios (.geo, .msh, malla de Elmer, .sif, .vtu, logs).

Requisitos:
    pip install gmsh numpy scipy matplotlib
    Elmer FEM instalado (ElmerSolver y ElmerGrid en el PATH, o completar ELMER_BIN)
    Opcional: pip install meshio  (lector VTU alternativo)

Se ejecuta una sola vez:  python fcl_gmsh_elmer.py
"""
import os
import sys
import csv
import json
import time
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle
from scipy.special import ellipk, ellipe

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
SALIDA = Path(__file__).resolve().parent / "fcl_elmer_out"
ELMER_BIN = ""        # ej. r"C:\Program Files\Elmer 9.0-Release\bin" si no está en el PATH

# Geometría (Tabla III del paper), en metros. N = número de vueltas
BOB_INT = dict(nombre="interior", ri=0.530, ro=0.560, h=0.400, N=24)
BOB_EXT = dict(nombre="exterior", ri=0.570, ro=0.600, h=0.400, N=24)
R_INF, Z_INF = 6.0, 6.0      # caja de aire (~10 radios de bobina), A = 0 en el borde
I_TEST = 1.0                 # [A] el problema es lineal: L no depende de I
I_NOMINAL_PICO = 360 * np.sqrt(2)   # [A] corriente para las figuras de |B| (Tabla II)

# Solver lineal de Elmer: "directo" (Umfpack) o "iterativo" (BiCGStab + ILU)
SOLVER_LINEAL = "directo"

# --- Convergencia de malla por energía ---
D_CONV = 0.100               # [m] posición inicial del paper
# (tamaño de elemento en las bobinas, tamaño en el aire lejano) [m]
NIVELES = [(0.040, 0.80), (0.020, 0.60), (0.012, 0.45), (0.008, 0.35),
           (0.005, 0.30), (0.0035, 0.25), (0.0025, 0.20), (0.0018, 0.15)]
TOL_ENERGIA = 0.002          # cambio relativo de energía entre niveles (0,2 %)

# --- Barrido de distancia entre centros ---
D_BARRIDO = np.round(np.arange(0.0, 0.8001, 0.025), 4)   # [m]

# Paper, Tabla IV: distancia [mm] e inductancia total [mH]
PAPER_D = [100, 100.124, 100.981, 104.178, 112.722, 131.052, 164.627,
           218.54, 294.67, 389.5, 496.12, 608.823, 700]
PAPER_L = [0.27697, 0.2772, 0.279, 0.28406, 0.29887, 0.33845, 0.4115,
           0.5431, 0.7524, 1.003, 1.18, 1.305, 1.38]
PAPER_L11, PAPER_L22 = 0.778, 0.8762      # Tabla VI [mH]

MU0 = 4e-7 * np.pi
CONFIGS = {                   # signo de la corriente en (interior, exterior)
    "interior": (1, 0),       # -> L11
    "exterior": (0, 1),       # -> L22
    "serie":    (1, -1),      # -> L11 + L22 - 2 L12 (conexión del limitador)
}
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


# =============================================================================
# UTILIDADES
# =============================================================================
def buscar_ejecutable(nombre):
    candidatos = [nombre, nombre + ".exe"]
    if ELMER_BIN:
        for c in candidatos:
            p = Path(ELMER_BIN) / c
            if p.exists():
                return str(p)
    for c in candidatos:
        p = shutil.which(c)
        if p:
            return p
    sys.exit(f"No encuentro {nombre}. Instalá Elmer FEM o completá ELMER_BIN arriba.")


def rect_bobina(b, zc):
    """(r0, r1, z0, z1) de una bobina centrada en z = zc."""
    return b["ri"], b["ro"], zc - b["h"] / 2, zc + b["h"] / 2


def z_centros(d):
    """Bobina interior arriba (+d/2) y exterior abajo (-d/2), como en la Fig. 1 del paper."""
    return +d / 2, -d / 2


def densidad_corriente(b, corriente=I_TEST):
    """J = N I / área de la sección [A/m²]."""
    return b["N"] * corriente / ((b["ro"] - b["ri"]) * b["h"])


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


def escribir_csv(path, filas):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


# =============================================================================
# REFERENCIA ANALÍTICA: filamentos + integrales elípticas (Maxwell)
# =============================================================================
def _M_espiras(a, b, dz):
    k2 = 4 * a * b / ((a + b) ** 2 + dz ** 2)
    k = np.sqrt(k2)
    return MU0 * np.sqrt(a * b) * ((2 / k - k) * ellipk(k2) - 2 / k * ellipe(k2))


def _filamentos(b, zc, nr, nz):
    dr, dz = (b["ro"] - b["ri"]) / nr, b["h"] / nz
    r = b["ri"] + dr * (np.arange(nr) + 0.5)
    z = zc - b["h"] / 2 + dz * (np.arange(nz) + 0.5)
    R, Z = np.meshgrid(r, z)
    return R.ravel(), Z.ravel(), dr, dz


def L_propia_analitica(b, nr=10, nz=60):
    R, Z, dr, dz = _filamentos(b, 0.0, nr, nz)
    with np.errstate(all="ignore"):
        M = _M_espiras(R[:, None], R[None, :], Z[:, None] - Z[None, :])
    gmd = 0.2235 * (dr + dz)                       # radio medio geométrico del filamento
    np.fill_diagonal(M, MU0 * R * (np.log(8 * R / gmd) - 2))
    return b["N"] ** 2 * M.sum() / len(R) ** 2


def L_mutua_analitica(d, nr=10, nz=60):
    zi, ze = z_centros(d)
    R1, Z1, _, _ = _filamentos(BOB_INT, zi, nr, nz)
    R2, Z2, _, _ = _filamentos(BOB_EXT, ze, nr, nz)
    M = _M_espiras(R1[:, None], R2[None, :], Z1[:, None] - Z2[None, :])
    return BOB_INT["N"] * BOB_EXT["N"] * M.mean()


# =============================================================================
# GMSH: geometría (.geo) y malla (.msh)
# =============================================================================
def escribir_geo(path, d, lc_b, lc_a):
    zi, ze = z_centros(d)
    r0i, r1i, z0i, z1i = rect_bobina(BOB_INT, zi)
    r0e, r1e, z0e, z1e = rect_bobina(BOB_EXT, ze)
    txt = f"""// =============================================================
// FCL-ACVSR - modelo axisimétrico (x = r, y = z) para Elmer
// d (distancia entre centros) = {d * 1e3:.1f} mm
// lc bobinas = {lc_b * 1e3:.2f} mm   |   lc aire lejano = {lc_a * 1e3:.0f} mm
// Generado por fcl_gmsh_elmer.py
// =============================================================
lcb = {lc_b};
lca = {lc_a};
Rinf = {R_INF};
Zinf = {Z_INF};

// ---- Caja de aire (el lado r = 0 es el eje de simetría) ----
Point(1) = {{0,    -Zinf, 0, lca}};
Point(2) = {{Rinf, -Zinf, 0, lca}};
Point(3) = {{Rinf,  Zinf, 0, lca}};
Point(4) = {{0,     Zinf, 0, lca}};
Line(1) = {{1, 2}};
Line(2) = {{2, 3}};
Line(3) = {{3, 4}};
Line(4) = {{4, 1}};   // eje

// ---- Bobina interior: r = [{r0i}, {r1i}], z = [{z0i:.4f}, {z1i:.4f}] ----
Point(5) = {{{r0i}, {z0i}, 0, lcb}};
Point(6) = {{{r1i}, {z0i}, 0, lcb}};
Point(7) = {{{r1i}, {z1i}, 0, lcb}};
Point(8) = {{{r0i}, {z1i}, 0, lcb}};
Line(5) = {{5, 6}};
Line(6) = {{6, 7}};
Line(7) = {{7, 8}};
Line(8) = {{8, 5}};

// ---- Bobina exterior: r = [{r0e}, {r1e}], z = [{z0e:.4f}, {z1e:.4f}] ----
Point(9)  = {{{r0e}, {z0e}, 0, lcb}};
Point(10) = {{{r1e}, {z0e}, 0, lcb}};
Point(11) = {{{r1e}, {z1e}, 0, lcb}};
Point(12) = {{{r0e}, {z1e}, 0, lcb}};
Line(9)  = {{9, 10}};
Line(10) = {{10, 11}};
Line(11) = {{11, 12}};
Line(12) = {{12, 9}};

Curve Loop(1) = {{1, 2, 3, 4}};
Curve Loop(2) = {{5, 6, 7, 8}};
Curve Loop(3) = {{9, 10, 11, 12}};
Plane Surface(1) = {{1, 2, 3}};   // aire (con dos agujeros)
Plane Surface(2) = {{2}};         // bobina interior
Plane Surface(3) = {{3}};         // bobina exterior

// ---- Grupos físicos -> cuerpos y contornos de Elmer ----
Physical Surface("aire", 1) = {{1}};
Physical Surface("bobina_interior", 2) = {{2}};
Physical Surface("bobina_exterior", 3) = {{3}};
Physical Curve("eje", 1) = {{4}};
Physical Curve("borde_exterior", 2) = {{1, 2, 3}};

// ---- Tamaño de malla: fino en las bobinas, crece hacia el aire lejano ----
Field[1] = Distance;
Field[1].CurvesList = {{5:12}};
Field[1].Sampling = 400;
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lcb;
Field[2].SizeMax = lca;
Field[2].DistMin = 0.03;
Field[2].DistMax = 1.5;
Background Field = 2;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
Mesh.MshFileVersion = 2.2;
"""
    Path(path).write_text(txt, encoding="utf-8")


def mallar(geo, msh):
    """Malla con GMSH y devuelve nodos (r,z) y triángulos para graficar."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(geo))
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh))
        tags, coords, _ = gmsh.model.mesh.getNodes()
        _, nodos_tri = gmsh.model.mesh.getElementsByType(2)
    finally:
        gmsh.finalize()
    tags = np.asarray(tags, dtype=np.int64)
    mapa = np.zeros(tags.max() + 1, dtype=np.int64)
    mapa[tags] = np.arange(len(tags))
    xy = np.asarray(coords).reshape(-1, 3)[:, :2]
    tri = mapa[np.asarray(nodos_tri, dtype=np.int64)].reshape(-1, 3)
    return xy, tri


# =============================================================================
# ELMER: conversión de malla, .sif, ejecución y lectura de resultados
# =============================================================================
def elmergrid(carpeta):
    r = subprocess.run([ELMERGRID, "14", "2", "modelo.msh", "-autoclean", "-out", "mesh"],
                       cwd=carpeta, capture_output=True, text=True)
    (carpeta / "elmergrid.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0 or not (carpeta / "mesh" / "mesh.header").exists():
        raise RuntimeError(f"ElmerGrid falló en {carpeta}:\n{(r.stdout + r.stderr)[-2000:]}")


def escribir_sif(path, J_int, J_ext):
    if SOLVER_LINEAL == "directo":
        lineal = ("  Linear System Solver = Direct\n"
                  "  Linear System Direct Method = Umfpack\n")
    else:
        lineal = ("  Linear System Solver = Iterative\n"
                  "  Linear System Iterative Method = BiCGStab\n"
                  "  Linear System Preconditioning = ILU1\n"
                  "  Linear System Max Iterations = 10000\n"
                  "  Linear System Convergence Tolerance = 1.0e-12\n")
    txt = f"""! =============================================================
! FCL-ACVSR - magnetostática axisimétrica (A_phi)
! J interior = {J_int:.6e} A/m2   |   J exterior = {J_ext:.6e} A/m2
! Generado por fcl_gmsh_elmer.py
! =============================================================
Header
  CHECK KEYWORDS Warn
  Mesh DB "." "mesh"
End

Simulation
  Max Output Level = 4
  Coordinate System = "Axi Symmetric"
  Simulation Type = Steady state
  Steady State Max Iterations = 1
  Output Intervals(1) = 1
End

Constants
  Permeability of Vacuum = {MU0:.12e}
End

Body 1
  Name = "aire"
  Target Bodies(1) = 1
  Equation = 1
  Material = 1
End

Body 2
  Name = "bobina_interior"
  Target Bodies(1) = 2
  Equation = 1
  Material = 2
  Body Force = 1
End

Body 3
  Name = "bobina_exterior"
  Target Bodies(1) = 3
  Equation = 1
  Material = 2
  Body Force = 2
End

Material 1
  Name = "aire"
  Relative Permeability = 1.0
End

Material 2
  Name = "cobre"
  Relative Permeability = 1.0
End

Equation 1
  Active Solvers(2) = 1 2
End

Solver 1
  Equation = "MgDyn2D"
  Procedure = "MagnetoDynamics2D" "MagnetoDynamics2D"
  Variable = "Potential"
{lineal}  Steady State Convergence Tolerance = 1.0e-10
End

Solver 2
  Equation = "Resultados"
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = "resultado"
  Output Format = vtu
  Ascii Output = Logical True
  Save Geometry Ids = Logical True
End

Body Force 1
  Name = "J bobina interior"
  Current Density = Real {J_int:.12e}
End

Body Force 2
  Name = "J bobina exterior"
  Current Density = Real {J_ext:.12e}
End

Boundary Condition 1
  Name = "eje y borde exterior"
  Target Boundaries(2) = 1 2
  Potential = Real 0.0
End
"""
    Path(path).write_text(txt, encoding="utf-8")


def correr_elmer(carpeta):
    (carpeta / "ELMERSOLVER_STARTINFO").write_text("case.sif\n1\n")
    r = subprocess.run([ELMERSOLVER, "case.sif"], cwd=carpeta, capture_output=True, text=True)
    log = r.stdout + r.stderr
    (carpeta / "elmer.log").write_text(log, encoding="utf-8")
    vtus = sorted(carpeta.rglob("resultado*.vtu"), key=os.path.getmtime)
    if r.returncode != 0 or not vtus:
        ult = "\n".join(log.splitlines()[-30:])
        raise RuntimeError(f"ElmerSolver falló en {carpeta}. Últimas líneas del log:\n{ult}")
    return vtus[-1]


def leer_vtu(path):
    """Devuelve nodos (r,z), triángulos y el potencial A_phi nodal."""
    try:
        import meshio
        m = meshio.read(path)
        tri = np.vstack([c.data[:, :3] for c in m.cells if c.type.startswith("triangle")])
        clave = [k for k in m.point_data if "potential" in k.lower()][0]
        return m.points[:, :2], tri, np.asarray(m.point_data[clave], float).ravel()
    except ImportError:
        pass
    pieza = ET.parse(path).getroot().find(".//Piece")

    def arreglo(da, tipo=float):
        if da.get("format", "ascii") != "ascii":
            raise RuntimeError("El VTU no está en ASCII: instalá meshio (pip install meshio).")
        return np.array(da.text.split(), dtype=tipo)

    pts = arreglo(pieza.find("Points/DataArray")).reshape(-1, 3)[:, :2]
    celdas = {da.get("Name"): da for da in pieza.find("Cells")}
    con = arreglo(celdas["connectivity"], np.int64)
    off = arreglo(celdas["offsets"], np.int64)
    tip = arreglo(celdas["types"], np.int64)
    ini = np.r_[0, off[:-1]]
    sel = np.isin(tip, [5, 22])                    # triángulo lineal / cuadrático
    tri = np.array([con[s:s + 3] for s in ini[sel]], dtype=np.int64)
    for da in pieza.find("PointData"):
        if "potential" in da.get("Name", "").lower():
            return pts, tri, arreglo(da)
    raise RuntimeError(f"No encontré la variable Potential en {path}")


def integrales_bobinas(pts, tri, A, d):
    """Para cada bobina: ∫ r·A dS sobre su sección y área cubierta por la malla.
    Con elementos lineales la integral del producto de dos funciones lineales es exacta:
        ∫ f g dS = Área/12 · (Σ f_i g_i + Σ f_i · Σ g_i)"""
    P = pts[tri]
    area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                        - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    rr, aa = pts[tri, 0], A[tri]
    int_rA = area / 12 * ((rr * aa).sum(1) + rr.sum(1) * aa.sum(1))
    c = P.mean(1)
    res = {}
    for b, zc in zip((BOB_INT, BOB_EXT), z_centros(d)):
        m = (c[:, 0] > b["ri"]) & (c[:, 0] < b["ro"]) & (np.abs(c[:, 1] - zc) < b["h"] / 2)
        res[b["nombre"]] = dict(int_rA=int_rA[m].sum(), area=area[m].sum(),
                                area_teorica=(b["ro"] - b["ri"]) * b["h"])
    return res


def energia_magnetica(pts, tri, A, d, J_int, J_ext):
    """W = ½ ∫ J·A dV = ½ Σ_k J_k ∫_k 2π r A dS   [J]"""
    integ = integrales_bobinas(pts, tri, A, d)
    W = 0.0
    for b, J in ((BOB_INT, J_int), (BOB_EXT, J_ext)):
        if J != 0:
            ib = integ[b["nombre"]]
            if abs(ib["area"] / ib["area_teorica"] - 1) > 1e-3:
                print(f"   ⚠ el área de malla de la bobina {b['nombre']} no coincide "
                      f"({ib['area']:.6f} vs {ib['area_teorica']:.6f} m²)")
            W += 0.5 * J * 2 * np.pi * ib["int_rA"]
    return W


# =============================================================================
# CASO COMPLETO: .geo -> .msh -> malla Elmer -> 1 a 3 corridas -> L
# =============================================================================
def simular(carpeta, d, lc_b, lc_a, configs):
    carpeta.mkdir(parents=True, exist_ok=True)
    geo, msh = carpeta / "modelo.geo", carpeta / "modelo.msh"
    t0 = time.time()
    escribir_geo(geo, d, lc_b, lc_a)
    xy, tri_g = mallar(geo, msh)
    elmergrid(carpeta)
    salida = dict(malla_xy=xy, malla_tri=tri_g, n_elem=len(tri_g), n_nodos=len(xy))
    for nombre in configs:
        si, se = CONFIGS[nombre]
        J_int, J_ext = si * densidad_corriente(BOB_INT), se * densidad_corriente(BOB_EXT)
        rd = carpeta / nombre
        if rd.exists():
            shutil.rmtree(rd)
        rd.mkdir()
        shutil.copytree(carpeta / "mesh", rd / "mesh")
        escribir_sif(rd / "case.sif", J_int, J_ext)
        pts, tri, A = leer_vtu(correr_elmer(rd))
        W = energia_magnetica(pts, tri, A, d, J_int, J_ext)
        salida[nombre] = dict(W=W, L=2 * W / I_TEST ** 2, pts=pts, tri=tri, A=A)
    salida["tiempo_s"] = time.time() - t0
    return salida


# =============================================================================
# FIGURAS
# =============================================================================
def dibujar_bobinas(ax, d, etiquetas=True):
    for b, zc, signo in zip((BOB_INT, BOB_EXT), z_centros(d), ("+I", "−I")):
        r0, r1, z0, z1 = rect_bobina(b, zc)
        ax.add_patch(Rectangle((r0, z0), r1 - r0, z1 - z0, fill=False, lw=1.2, ec="#d62728"))
        if etiquetas:
            ax.text(r1 + 0.02, zc, f"{b['nombre']}\n{signo}", fontsize=7, va="center", color="#d62728")


def figura_malla(xy, tri, d, lc_b, lc_a, nivel, W, destino):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.5), dpi=DPI)
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], tri)
    for ax in (a1, a2):
        ax.triplot(triang, lw=0.25, color="#1f4e79")
        ax.set_aspect("equal")
        ax.set_xlabel("r [m]")
        dibujar_bobinas(ax, d, etiquetas=(ax is a2))
    a1.set_ylabel("z [m]")
    a1.set_xlim(0, R_INF)
    a1.set_ylim(-Z_INF, Z_INF)
    a1.set_title("Dominio completo")
    a2.set_xlim(0.44, 0.70)
    a2.set_ylim(-0.32, 0.32)
    a2.set_title("Detalle de las bobinas")
    fig.suptitle(f"Nivel {nivel}:  lc bobina = {lc_b * 1e3:.1f} mm,  {len(tri):,} elementos,  "
                 f"W = {W * 1e6:.4f} µJ (I = 1 A)".replace(",", "."), fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.1, wspace=0.25)
    guardar_fig(fig, destino)


def campo_B(pts, tri, A):
    """B por elemento a partir de ψ = 2π r A (función de flujo)."""
    psi = 2 * np.pi * pts[:, 0] * A
    P = pts[tri]
    x1, y1 = P[:, 0, 0], P[:, 0, 1]
    x2, y2 = P[:, 1, 0], P[:, 1, 1]
    x3, y3 = P[:, 2, 0], P[:, 2, 1]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    p1, p2, p3 = psi[tri[:, 0]], psi[tri[:, 1]], psi[tri[:, 2]]
    dpsi_dr = ((p2 - p1) * (y3 - y1) - (p3 - p1) * (y2 - y1)) / det
    dpsi_dz = ((p3 - p1) * (x2 - x1) - (p2 - p1) * (x3 - x1)) / det
    rc = np.maximum(P[:, :, 0].mean(1), 1e-6)
    Bz = dpsi_dr / (2 * np.pi * rc)
    Br = -dpsi_dz / (2 * np.pi * rc)
    return psi, np.hypot(Br, Bz)


def figura_flujo(caso, d, L_mH, escala, vmax, niveles_psi, destino):
    pts, tri, A = caso["pts"], caso["tri"], caso["A"] * escala
    c = pts[tri].mean(1)
    m = (c[:, 0] < 1.6) & (np.abs(c[:, 1]) < 1.2)
    tri = tri[m]
    psi, Bmod = campo_B(pts, tri, A)
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
    fig, ax = plt.subplots(figsize=(6.4, 7.2), dpi=DPI)
    pc = ax.tripcolor(triang, facecolors=Bmod * 1e3, cmap="magma", vmin=0, vmax=vmax * 1e3)
    ax.tricontour(triang, psi, levels=niveles_psi, colors="w", linewidths=0.6)
    dibujar_bobinas(ax, d)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"d = {d * 1e3:.0f} mm    L total = {L_mH:.3f} mH")
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(f"|B| [mT]  (I = {I_NOMINAL_PICO:.0f} A pico)")
    fig.subplots_adjust(left=0.12, right=0.88, top=0.94, bottom=0.07)
    guardar_fig(fig, destino)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main():
    global ELMERSOLVER, ELMERGRID
    ELMERSOLVER = buscar_ejecutable("ElmerSolver")
    ELMERGRID = buscar_ejecutable("ElmerGrid")
    try:
        import gmsh  # noqa: F401
    except ImportError:
        sys.exit("Falta gmsh:  pip install gmsh")

    dir_conv = SALIDA / "01_convergencia"
    dir_barr = SALIDA / "02_barrido"
    dir_img_m = SALIDA / "imagenes" / "malla"
    dir_img_f = SALIDA / "imagenes" / "flujo"
    dir_res = SALIDA / "resultados"
    for p in (dir_conv, dir_barr, dir_img_m, dir_img_f, dir_res):
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Referencia analítica (filamentos)")
    L11_an, L22_an = L_propia_analitica(BOB_INT), L_propia_analitica(BOB_EXT)
    print(f"  L11 = {L11_an * 1e3:.4f} mH   L22 = {L22_an * 1e3:.4f} mH")

    # ------------------------------------------------------------------
    # 1) CONVERGENCIA DE MALLA POR ENERGÍA (conexión serie, d = 100 mm)
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"1) Convergencia de malla por energía (d = {D_CONV * 1e3:.0f} mm, conexión serie)")
    filas_conv, pngs_malla, elegido = [], [], None
    W_ant = None
    for k, (lc_b, lc_a) in enumerate(NIVELES, start=1):
        caso = simular(dir_conv / f"nivel_{k:02d}", D_CONV, lc_b, lc_a, ["serie"])
        W = caso["serie"]["W"]
        dW = abs(W - W_ant) / W if W_ant else np.nan
        fila = dict(nivel=k, lc_bobina_mm=lc_b * 1e3, lc_aire_mm=lc_a * 1e3,
                    elementos=caso["n_elem"], nodos=caso["n_nodos"], energia_uJ=W * 1e6,
                    L_total_mH=caso["serie"]["L"] * 1e3, cambio_relativo=dW,
                    tiempo_s=round(caso["tiempo_s"], 2))
        filas_conv.append(fila)
        print(f"  nivel {k}: lc = {lc_b * 1e3:5.1f} mm  {caso['n_elem']:7d} elem  "
              f"W = {W * 1e6:.5f} µJ  L = {fila['L_total_mH']:.5f} mH  ΔW/W = {dW:.2e}")
        png = dir_img_m / f"malla_nivel_{k:02d}.png"
        figura_malla(caso["malla_xy"], caso["malla_tri"], D_CONV, lc_b, lc_a, k, W, png)
        pngs_malla.append(png)
        if elegido is None and W_ant is not None and dW < TOL_ENERGIA:
            elegido = k
        W_ant = W
    if elegido is None:
        elegido = len(NIVELES)
        print("  ⚠ no se alcanzó la tolerancia: uso el nivel más fino")
    W_fino = filas_conv[-1]["energia_uJ"]
    for f in filas_conv:
        f["error_vs_mas_fino"] = abs(f["energia_uJ"] - W_fino) / W_fino
    lc_b_opt, lc_a_opt = NIVELES[elegido - 1]
    print(f"  -> malla óptima: nivel {elegido} (lc bobina = {lc_b_opt * 1e3:.1f} mm, "
          f"lc aire = {lc_a_opt * 1e3:.0f} mm)")
    escribir_csv(dir_res / "convergencia_energia.csv", filas_conv)
    guardar_gif(pngs_malla, dir_res / "refinamiento_malla.gif", 700)

    # Gráfico de convergencia
    ne = [f["elementos"] for f in filas_conv]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=DPI)
    a1.semilogx(ne, [f["energia_uJ"] for f in filas_conv], "o-")
    a1.semilogx(ne[elegido - 1], filas_conv[elegido - 1]["energia_uJ"], "r*", ms=16, label="elegido")
    a1.set_xlabel("elementos")
    a1.set_ylabel("energía magnética [µJ] (I = 1 A)")
    a1.set_title("La energía converge desde abajo")
    a1.grid(True, which="both", alpha=0.3)
    a1.legend()
    a2.loglog(ne[1:], [f["cambio_relativo"] for f in filas_conv[1:]], "s-", label="|ΔW|/W entre niveles")
    a2.loglog(ne[:-1], [max(f["error_vs_mas_fino"], 1e-9) for f in filas_conv[:-1]], "^-",
              label="error vs malla más fina")
    a2.axhline(TOL_ENERGIA, color="r", ls="--", label=f"tolerancia {TOL_ENERGIA:.1%}")
    a2.set_xlabel("elementos")
    a2.set_title("Criterio de convergencia")
    a2.grid(True, which="both", alpha=0.3)
    a2.legend(fontsize=8)
    fig.tight_layout()
    guardar_fig(fig, dir_res / "convergencia_energia.png")

    # ------------------------------------------------------------------
    # 2) BARRIDO EN d CON LA MALLA ÓPTIMA
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"2) Barrido de d ({len(D_BARRIDO)} posiciones x 3 excitaciones)")
    filas, casos_serie = [], []
    for d in D_BARRIDO:
        caso = simular(dir_barr / f"d_{d * 1e3:04.0f}mm", d, lc_b_opt, lc_a_opt, list(CONFIGS))
        L11, L22, Ls = caso["interior"]["L"], caso["exterior"]["L"], caso["serie"]["L"]
        L12 = (L11 + L22 - Ls) / 2
        L12_an = L_mutua_analitica(d)
        filas.append(dict(d_mm=d * 1e3, L11_mH=L11 * 1e3, L22_mH=L22 * 1e3, L12_mH=L12 * 1e3,
                          L_total_mH=Ls * 1e3, L12_analitico_mH=L12_an * 1e3,
                          L_total_analitico_mH=(L11_an + L22_an - 2 * L12_an) * 1e3,
                          elementos=caso["n_elem"]))
        casos_serie.append(caso["serie"])
        print(f"  d = {d * 1e3:5.0f} mm  L11 = {L11 * 1e3:.4f}  L22 = {L22 * 1e3:.4f}  "
              f"L12 = {L12 * 1e3:.4f}  L total = {Ls * 1e3:.4f} mH")
        for k in ("interior", "exterior"):         # liberar memoria
            caso.pop(k)

    dd = np.array([f["d_mm"] for f in filas]) / 1e3
    L12 = np.array([f["L12_mH"] for f in filas]) / 1e3
    dL12 = np.gradient(L12, dd)
    for f, g in zip(filas, dL12):
        f["dL12_dd_mH_m"] = g * 1e3
        f["fuerza_por_I2_N_A2"] = -g          # F = -I² dL12/dd (repulsiva > 0)
    escribir_csv(dir_res / "inductancias_vs_d.csv", filas)

    # Imágenes de flujo con escala de colores y líneas comunes a todos los cuadros
    print("  generando imágenes de flujo...")
    escala = I_NOMINAL_PICO / I_TEST
    vmax, psimax = 0.0, 0.0
    for cs in casos_serie:
        psi, Bm = campo_B(cs["pts"], cs["tri"], cs["A"] * escala)
        vmax = max(vmax, np.percentile(Bm, 99.7))
        psimax = max(psimax, np.abs(psi).max())
    niv = np.linspace(-psimax, psimax, 32)
    niv = niv[np.abs(niv) > 1e-12 * psimax]
    pngs_flujo = []
    for f, cs in zip(filas, casos_serie):
        png = dir_img_f / f"flujo_d_{f['d_mm']:04.0f}mm.png"
        figura_flujo(cs, f["d_mm"] / 1e3, f["L_total_mH"], escala, vmax, niv, png)
        pngs_flujo.append(png)
    guardar_gif(pngs_flujo + pngs_flujo[::-1], dir_res / "lineas_flujo_vs_d.gif", 220)

    # Inductancias vs d
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=DPI)
    a1.plot(dd * 1e3, [f["L_total_mH"] for f in filas], "o-", label="Elmer")
    a1.plot(dd * 1e3, [f["L_total_analitico_mH"] for f in filas], "--", label="Analítico (filamentos)")
    a1.plot(PAPER_D, PAPER_L, "ks", mfc="none", label="Paper (Tabla IV)")
    a1.set_xlabel("distancia entre centros d [mm]")
    a1.set_ylabel("L total del limitador [mH]")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(dd * 1e3, L12 * 1e3, "o-", label="L12 Elmer")
    a2.plot(dd * 1e3, [f["L12_analitico_mH"] for f in filas], "--", label="L12 analítico")
    a2.axhline(np.mean([f["L11_mH"] for f in filas]), color="C2", ls=":", label="L11 Elmer")
    a2.axhline(np.mean([f["L22_mH"] for f in filas]), color="C3", ls=":", label="L22 Elmer")
    a2b = a2.twinx()
    a2b.plot(dd * 1e3, -dL12 * 1e3, "C4-", alpha=0.6, label="F/I² [mN/A²]")
    a2b.set_ylabel("F / I² [mN/A²]")
    a2.set_xlabel("distancia entre centros d [mm]")
    a2.set_ylabel("inductancia [mH]")
    a2.legend(loc="center right", fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    guardar_fig(fig, dir_res / "inductancias_vs_d.png")

    # ------------------------------------------------------------------
    # 3) RESUMEN
    # ------------------------------------------------------------------
    def interp(col, dmm):
        return float(np.interp(dmm, dd * 1e3, [f[col] for f in filas]))

    resumen = dict(
        malla_optima=dict(nivel=elegido, lc_bobina_mm=lc_b_opt * 1e3, lc_aire_mm=lc_a_opt * 1e3,
                          elementos=filas_conv[elegido - 1]["elementos"], tolerancia=TOL_ENERGIA),
        L11_mH=dict(elmer=float(np.mean([f["L11_mH"] for f in filas])),
                    analitico=L11_an * 1e3, paper=PAPER_L11),
        L22_mH=dict(elmer=float(np.mean([f["L22_mH"] for f in filas])),
                    analitico=L22_an * 1e3, paper=PAPER_L22),
        L_total_100mm=dict(elmer=interp("L_total_mH", 100),
                           analitico=interp("L_total_analitico_mH", 100), paper=0.27697),
        L_total_700mm=dict(elmer=interp("L_total_mH", 700),
                           analitico=interp("L_total_analitico_mH", 700), paper=1.38),
    )
    (dir_res / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print("=" * 70)
    print("RESUMEN                    Elmer    Analítico   Paper   [mH]")
    for k in ("L11_mH", "L22_mH", "L_total_100mm", "L_total_700mm"):
        v = resumen[k]
        print(f"  {k:22s} {v['elmer']:8.4f}  {v['analitico']:9.4f}  {v['paper']:7.4f}")
    if abs(resumen["L11_mH"]["elmer"] / resumen["L11_mH"]["analitico"] - 1) > 0.05:
        print("  ⚠ Elmer y el analítico difieren más de 5 %: revisá el .sif o el tamaño de la caja de aire.")
    print(f"\nTodo guardado en: {SALIDA}")


if __name__ == "__main__":
    main()
