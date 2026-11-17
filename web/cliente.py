# -*- coding: utf-8 -*-
"""
Cliente del frontend: habla con la API por HTTP o, si no hay API disponible, llama al núcleo
directamente (modo local). Así la misma app sirve en desarrollo (con FastAPI) y desplegada sola
en Streamlit Community Cloud.

Orden de decisión:
  1. Variable de entorno FCL_API definida  -> modo HTTP contra esa URL.
  2. Hay una API respondiendo en http://localhost:8000  -> modo HTTP.
  3. Si no -> modo local (fcl.servicio + api.db, sin FastAPI).
"""
import os
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


class ErrorCliente(Exception):
    pass


class Cliente:
    def __init__(self, url=None, timeout=300):
        self.timeout = timeout
        self.url = (url or os.environ.get("FCL_API") or "").rstrip("/")
        if not self.url and self._responde("http://localhost:8000"):
            self.url = "http://localhost:8000"
        self.modo = "api" if self.url else "local"
        if self.modo == "local":
            from fcl import servicio
            from api import db
            self._servicio, self._db = servicio, db
            self._con = db.conectar()

    @staticmethod
    def _responde(url):
        try:
            import requests
            return requests.get(f"{url}/salud", timeout=1.5).ok
        except Exception:
            return False

    def llamar(self, metodo, ruta, json=None):
        if self.modo == "api":
            import requests
            try:
                r = requests.request(metodo, f"{self.url}{ruta}", json=json, timeout=self.timeout)
                r.raise_for_status()
                return r.json() if r.content else None
            except requests.RequestException as e:
                raise ErrorCliente(f"No pude hablar con la API en {self.url}{ruta}: {e}") from e
        return self._local(metodo, ruta, json or {})

    def _local(self, metodo, ruta, j):
        s, db, con = self._servicio, self._db, self._con
        if (metodo, ruta) == ("GET", "/info"):
            return s.info()
        if (metodo, ruta) == ("POST", "/inductancia"):
            return s.inductancias(j["geometria"], j.get("d_m"))
        if (metodo, ruta) == ("POST", "/simular-falla"):
            return s.simular_falla(j["geometria"], j.get("red"), j.get("t_fin_s", 0.06))
        if (metodo, ruta) == ("POST", "/optimizar"):
            return s.optimizar(j.get("objetivo", "masa_kg"), j.get("restricciones"), j.get("restricciones_rel_paper"),
                               j.get("eje_x"), n=j.get("n", 600), generaciones=j.get("generaciones", 4),
                               semilla=j.get("semilla", 0))
        if (metodo, ruta) == ("GET", "/disenos"):
            return db.listar(con)
        if (metodo, ruta) == ("POST", "/disenos"):
            return db.guardar(con, j["nombre"], j["geometria"], j.get("metricas"))
        m = re.fullmatch(r"/disenos/(\d+)", ruta)
        if m and metodo == "DELETE":
            db.borrar(con, int(m.group(1)))
            return None
        if m and metodo == "GET":
            return db.obtener(con, int(m.group(1)))
        raise ErrorCliente(f"Ruta no soportada en modo local: {metodo} {ruta}")
