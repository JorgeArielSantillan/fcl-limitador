# -*- coding: utf-8 -*-
"""Tests del surrogate, el simulador de falla, el optimizador, el servicio y la API.
    pytest -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fcl.surrogate import SurrogateFCL
from fcl.dinamica import PAPER, Diseno, Red, simular, tablas_surrogate, masas, resistencia
from fcl.optimizacion import no_dominados, frente_2d, evaluar, optimizar, limites_del_paper
from fcl import servicio

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sur():
    return SurrogateFCL.cargar(RAIZ / "modelos" / "surrogate_fcl.npz")


@pytest.fixture(scope="module")
def test_csv():
    return pd.read_csv(RAIZ / "datos" / "dataset_test.csv")


# ---------------------------------------------------------------- surrogate
def test_surrogate_reproduce_elmer_en_geometrias_de_prueba(sur, test_csv):
    p = sur.predecir(test_csv.ri_int, test_csv.t_int, test_csv.gap, test_csv.t_ext, test_csv.h, test_csv.d_m)
    ltot = (p["l11"] + p["l22"] - 2 * p["l12"]) * 1e6
    err = np.abs(ltot / test_csv.ltot_uH - 1)
    assert err.max() < 0.005            # < 0,5 % en L total en geometrías nunca vistas


def test_derivada_coincide_con_diferencias_finitas(sur):
    d = np.linspace(0.12, 0.68, 15)
    e = 1e-5
    p = sur.predecir(0.53, 0.03, 0.01, 0.03, 0.4, d)
    pm = sur.predecir(0.53, 0.03, 0.01, 0.03, 0.4, d - e)
    pp = sur.predecir(0.53, 0.03, 0.01, 0.03, 0.4, d + e)
    fd = (pp["l12"] - pm["l12"]) / (2 * e)
    assert np.allclose(p["dl12_dd"], fd, rtol=1e-4, atol=1e-12)


def test_inductancias_del_paper(sur):
    L = sur.inductancias(0.53, 0.03, 0.01, 0.03, 0.4, [0.1, 0.7], 24, 24)
    assert L["L11"][0] * 1e3 == pytest.approx(0.732, rel=0.01)       # Elmer 0,7318 mH
    assert L["L_total"][0] * 1e3 == pytest.approx(0.257, rel=0.01)   # Elmer 0,2569 mH
    assert L["L_total"][1] * 1e3 == pytest.approx(1.294, rel=0.01)   # Elmer 1,2935 mH
    assert np.all(L["F_sobre_I2"] > 0)                               # repulsión


def test_aviso_fuera_de_rango(sur):
    assert sur.fuera_de_rango(0.9, 0.03, 0.01, 0.03, 0.4, 0.1)
    assert not sur.fuera_de_rango(0.53, 0.03, 0.01, 0.03, 0.4, 0.1)


# ---------------------------------------------------------------- dinámica
def test_falla_del_paper(sur):
    r = simular([PAPER], tablas_surrogate(sur, [PAPER]))
    assert r["pico_sin_fcl_kA"][0] == pytest.approx(35.035, rel=1e-3)   # Tabla V
    assert r["pico_kA"][0, 0] == pytest.approx(30.24, abs=0.1)            # validación analítica
    assert r["t_operacion_ms"][0] == pytest.approx(15.8, abs=0.3)
    assert r["perdidas_W"][0] == pytest.approx(2333, rel=0.01)            # paper: 2,3 kW
    assert np.all(r["limitacion_pct"][0] > 0)


def test_calibracion_masa_y_resistencia():
    m1, m2 = masas([PAPER])
    assert (m1[0], m2[0]) == pytest.approx((170, 180))
    assert resistencia([PAPER])[0] == pytest.approx(0.006)
    assert resistencia([Diseno(N=48)])[0] == pytest.approx(0.024)        # ∝ N²


def test_vectorizado_igual_a_individual(sur):
    ds = [PAPER, Diseno(0.4, 0.02, 0.015, 0.04, 0.3, 20)]
    juntos = simular(ds, tablas_surrogate(sur, ds), t_fin=0.03)
    solo = simular(ds[1:], tablas_surrogate(sur, ds[1:]), t_fin=0.03)
    assert juntos["pico_kA"][1] == pytest.approx(solo["pico_kA"][0])


# ---------------------------------------------------------------- optimización
def test_no_dominados():
    F = np.array([[1, 5], [2, 2], [3, 3], [5, 1], [4, 4]])
    assert no_dominados(F).tolist() == [True, True, False, True, False]


def test_frente_2d_ordenado():
    filas = [dict(a=1, b=5), dict(a=2, b=2), dict(a=3, b=3), dict(a=5, b=1)]
    assert frente_2d(filas, "a", "b") == [0, 1, 3]


def test_optimizar_respeta_restricciones(sur):
    ref = evaluar(sur, [PAPER])[0]
    lim = limites_del_paper(ref)
    lim["pico2_kA"] = ref["pico2_kA"] * 1.1
    r = optimizar(sur, "masa_kg", lim, n=600, generaciones=3, hijos=400, informar=None)
    m = r["mejor"]
    assert m is not None
    assert all(m[k] <= v * (1 + 1e-9) for k, v in lim.items())
    assert m["masa_kg"] < ref["masa_kg"]


# ---------------------------------------------------------------- servicio y API
def test_servicio_devuelve_json_serializable():
    import json
    for r in (servicio.info(), servicio.inductancias(PAPER.dict(), [0.1, 0.4]),
              servicio.simular_falla(PAPER.dict(), t_fin=0.02)):
        json.dumps(r)


def test_api():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from api.main import app
    c = TestClient(app)
    assert c.get("/salud").json() == {"estado": "ok"}
    r = c.post("/inductancia", json={"geometria": PAPER.dict(), "d_m": [0.1, 0.7]})
    assert r.status_code == 200 and len(r.json()["L_total_mH"]) == 2
    assert c.post("/inductancia", json={"geometria": {**PAPER.dict(), "h": 5}}).status_code == 422
    nuevo = c.post("/disenos", json={"nombre": "test", "geometria": PAPER.dict()}).json()
    assert c.get(f"/disenos/{nuevo['id']}").status_code == 200
    assert c.delete(f"/disenos/{nuevo['id']}").status_code == 204


def test_cliente_modo_local(tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv("FCL_DB", str(tmp_path / "t.sqlite"))
    monkeypatch.delenv("FCL_API", raising=False)
    sys.path.insert(0, str(RAIZ / "web"))
    import importlib
    from api import db
    importlib.reload(db)
    import cliente
    c = cliente.Cliente()
    c.url, c.modo = "", "local"          # forzar modo local aunque haya una API corriendo
    assert c.llamar("GET", "/info")["paper"]["N"] == 24
    r = c.llamar("POST", "/inductancia", {"geometria": PAPER.dict(), "d_m": [0.1]})
    assert r["L_total_mH"][0] == pytest.approx(0.257, rel=0.01)
    g = c.llamar("POST", "/disenos", {"nombre": "x", "geometria": PAPER.dict()})
    assert len(c.llamar("GET", "/disenos")) == 1
    c.llamar("DELETE", f"/disenos/{g['id']}")
    assert c.llamar("GET", "/disenos") == []
