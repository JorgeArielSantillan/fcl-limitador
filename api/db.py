# -*- coding: utf-8 -*-
"""Persistencia mínima en SQLite: diseños guardados con sus métricas."""
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

RUTA_DB = Path(os.environ.get("FCL_DB", Path(__file__).resolve().parents[1] / "datos" / "disenos.sqlite"))


def conectar(ruta=None):
    ruta = Path(ruta or RUTA_DB)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ruta, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS disenos (
                       id       INTEGER PRIMARY KEY AUTOINCREMENT,
                       creado   TEXT NOT NULL,
                       nombre   TEXT NOT NULL,
                       diseno   TEXT NOT NULL,
                       metricas TEXT)""")
    con.commit()
    return con


def guardar(con, nombre, diseno, metricas=None):
    cur = con.execute("INSERT INTO disenos (creado, nombre, diseno, metricas) VALUES (?, ?, ?, ?)",
                      (datetime.now(timezone.utc).isoformat(timespec="seconds"), nombre,
                       json.dumps(diseno), json.dumps(metricas) if metricas is not None else None))
    con.commit()
    return obtener(con, cur.lastrowid)


def _fila(r):
    return dict(id=r["id"], creado=r["creado"], nombre=r["nombre"], diseno=json.loads(r["diseno"]),
                metricas=json.loads(r["metricas"]) if r["metricas"] else None)


def obtener(con, id_):
    r = con.execute("SELECT * FROM disenos WHERE id = ?", (id_,)).fetchone()
    return _fila(r) if r else None


def listar(con, limite=100):
    return [_fila(r) for r in con.execute("SELECT * FROM disenos ORDER BY id DESC LIMIT ?", (limite,))]


def borrar(con, id_):
    cur = con.execute("DELETE FROM disenos WHERE id = ?", (id_,))
    con.commit()
    return cur.rowcount > 0
