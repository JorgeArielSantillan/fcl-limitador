# -*- coding: utf-8 -*-
"""
Frontend del limitador de corriente de falla (Streamlit).

    streamlit run web/app.py
Si hay una API corriendo (FCL_API o http://localhost:8000) la usa; si no, llama al modelo
directamente (así se despliega solo en Streamlit Community Cloud). Ver web/cliente.py.
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cliente import Cliente, ErrorCliente

st.set_page_config(page_title="Limitador FCL-ACVSR", page_icon="⚡", layout="wide")


@st.cache_resource
def cliente():
    return Cliente()


def llamar(metodo, ruta, **kw):
    try:
        return cliente().llamar(metodo, ruta, kw.get("json"))
    except ErrorCliente as e:
        st.error(str(e))
        st.stop()


@st.cache_data
def info():
    return llamar("GET", "/info")


def figura(ancho=6, alto=3.4):
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=110)
    ax.grid(alpha=0.3)
    return fig, ax


I = info()
R, P = I["rangos"], I["paper"]

st.title("⚡ Limitador de corriente de falla de núcleo de aire")
st.caption("Surrogate entrenado con simulaciones FEM (GMSH + Elmer) · basado en Amini et al., IEEE Access 2021 · "
           + ("conectado a la API" if cliente().modo == "api" else "modo local (sin API)")
           + " · [código en GitHub](https://github.com/JorgeArielSantillan/fcl-limitador)")

# ---------------------------------------------------------------- barra lateral: geometría
# valores iniciales y cargas pendientes (deben aplicarse ANTES de crear los sliders)
for k in P:
    st.session_state.setdefault(k, P[k])
if "cargar" in st.session_state:
    for k, v in st.session_state.pop("cargar").items():
        st.session_state[k] = v

with st.sidebar:
    st.header("Geometría")
    if st.button("Volver al diseño del paper"):
        st.session_state["cargar"] = dict(P)
        st.rerun()
    g = {}
    for k, nombre in (("ri_int", "Radio interno bobina interior [m]"), ("t_int", "Espesor bobina interior [m]"),
                      ("gap", "Separación radial [m]"), ("t_ext", "Espesor bobina exterior [m]"),
                      ("h", "Altura [m]")):
        lo, hi = R[k]
        st.session_state[k] = min(max(float(st.session_state[k]), float(lo)), float(hi))
        g[k] = st.slider(nombre, float(lo), float(hi), step=0.001, format="%.3f", key=k)
    st.session_state["N"] = int(st.session_state["N"])
    g["N"] = st.slider("Vueltas por bobina", 5, 60, key="N")
    st.caption(f"Modelo: {I['modelo']} · rango válido de d: 0 a 2h")

tab1, tab2, tab3, tab4 = st.tabs(["Inductancias", "Simular falla", "Optimizar", "Diseños guardados"])

# ---------------------------------------------------------------- inductancias
with tab1:
    r = llamar("POST", "/inductancia", json={"geometria": g})
    for a in r["avisos"]:
        st.warning(a)
    df = pd.DataFrame({k: v for k, v in r.items() if k != "avisos"})
    c1, c2 = st.columns(2)
    fig, ax = figura()
    ax.plot(df.d_m * 1e3, df.L_total_mH, label="L total (serie)")
    ax.plot(df.d_m * 1e3, df.L12_mH, "--", label="L12")
    ax.axhline(df.L11_mH[0], color="C2", ls=":", label="L11")
    ax.axhline(df.L22_mH[0], color="C3", ls=":", label="L22")
    ax.set(xlabel="distancia entre centros d [mm]", ylabel="[mH]")
    ax.legend(fontsize=8)
    c1.pyplot(fig)
    fig, ax = figura()
    ax.plot(df.d_m * 1e3, df.F_sobre_I2_mN_A2, "C4")
    ax.set(xlabel="d [mm]", ylabel="F / I² [mN/A²]", title="Fuerza de repulsión por unidad de corriente²")
    c2.pyplot(fig)
    d0, dmax = I["d0_sobre_h"] * g["h"], I["dmax_sobre_h"] * g["h"]
    k0 = (df.d_m - d0).abs().idxmin()
    k1 = (df.d_m - dmax).abs().idxmin()
    m1, m2, m3 = st.columns(3)
    m1.metric("L en servicio (d0)", f"{df.L_total_mH[k0]:.3f} mH")
    m2.metric("L con bobinas separadas", f"{df.L_total_mH[k1]:.3f} mH")
    m3.metric("Relación", f"{df.L_total_mH[k1] / df.L_total_mH[k0]:.1f}×")

# ---------------------------------------------------------------- falla
with tab2:
    if st.button("Simular cortocircuito", type="primary"):
        st.session_state["falla"] = llamar("POST", "/simular-falla", json={"geometria": g})
    f = st.session_state.get("falla")
    if f:
        for a in f["avisos"]:
            st.warning(a)
        m = f["metricas"]
        c = st.columns(5)
        c[0].metric("Pico 1.er período", f"{m['pico_kA'][0]:.1f} kA", f"{-m['limitacion_pct'][0]:.1f} %",
                    delta_color="inverse")
        c[1].metric("Pico 2.º período", f"{m['pico_kA'][1]:.1f} kA", f"{-m['limitacion_pct'][1]:.1f} %",
                    delta_color="inverse")
        c[2].metric("Tiempo de operación",
                    "no llega" if m["t_operacion_ms"] is None else f"{m['t_operacion_ms']:.1f} ms")
        c[3].metric("Caída de tensión en servicio", f"{m['caida_tension_pct']:.2f} %")
        c[4].metric("Masa / pérdidas", f"{m['masa_kg']:.0f} kg", f"{m['perdidas_W'] / 1e3:.2f} kW", delta_color="off")
        s = pd.DataFrame(f["series"])
        c1, c2 = st.columns(2)
        fig, ax = figura()
        ax.plot(s.t_ms, s.i_sin_fcl_kA, color="0.6", label="sin limitador")
        ax.plot(s.t_ms, s.i_kA, "C0", label="con limitador")
        ax.set(xlabel="t [ms]", ylabel="i [kA]", title="Corriente de cortocircuito")
        ax.legend(fontsize=8)
        c1.pyplot(fig)
        fig, ax = figura()
        ax.plot(s.t_ms, s.d_mm, "C1")
        ax2 = ax.twinx()
        ax2.plot(s.t_ms, s.L_mH, "C2")
        ax.set(xlabel="t [ms]", ylabel="d [mm] (naranja)", title="Separación e inductancia")
        ax2.set_ylabel("L [mH] (verde)")
        c2.pyplot(fig)
        with st.form("guardar"):
            nombre = st.text_input("Nombre del diseño", "mi diseño")
            if st.form_submit_button("Guardar diseño"):
                llamar("POST", "/disenos", json={"nombre": nombre, "geometria": f["diseno"], "metricas": m})
                st.success("Guardado")

# ---------------------------------------------------------------- optimizar
with tab3:
    st.write("Minimiza un objetivo sin empeorar las demás métricas respecto del diseño del paper "
             "(con la holgura que elijas).")
    nombres = {"masa_kg": "masa", "pico1_kA": "pico 1.er período", "pico2_kA": "pico 2.º período",
               "caida_tension_pct": "caída de tensión", "perdidas_W": "pérdidas"}
    objetivo = st.selectbox("Objetivo a minimizar", list(nombres), format_func=nombres.get)
    st.write("Límites respecto del paper (100 % = igual que el paper):")
    cols = st.columns(4)
    rel = {}
    for col, m in zip(cols, [m for m in nombres if m != objetivo]):
        rel[m] = col.slider(nombres[m], 80, 150, 110 if m == "pico2_kA" else 100, step=1, key=f"lim_{m}") / 100
    opciones = list(rel)
    eje = st.selectbox("Curva de compromiso contra", opciones, format_func=nombres.get,
                       index=opciones.index("pico2_kA") if "pico2_kA" in opciones else 0)
    n = st.select_slider("Esfuerzo de búsqueda", options=[300, 600, 1200, 2000], value=600)
    if st.button("Optimizar", type="primary"):
        with st.spinner("Simulando la falla de cientos de diseños con el surrogate..."):
            st.session_state["opt"] = llamar("POST", "/optimizar", json={
                "objetivo": objetivo, "restricciones_rel_paper": rel, "eje_x": eje, "n": n, "generaciones": 4})
    o = st.session_state.get("opt")
    if o:
        st.caption(f"{o['evaluados']} diseños evaluados, {o['factibles']} cumplen las restricciones")
        if not o["mejor"]:
            st.warning("Ningún diseño cumple esos límites. Relajá alguno.")
        else:
            b, p = o["mejor"], o["paper"]
            tabla = pd.DataFrame({"paper": {k: p[k] for k in nombres}, "optimizado": {k: b[k] for k in nombres}})
            tabla["cambio %"] = 100 * (tabla.optimizado / tabla.paper - 1)
            tabla.index = [nombres[k] for k in tabla.index]
            c1, c2 = st.columns([1, 1.3])
            c1.dataframe(tabla.style.format("{:.2f}"))
            c1.json({k: b[k] for k in ("ri_int", "t_int", "gap", "t_ext", "h", "N")})
            if st.button("Cargar este diseño en la barra lateral"):
                st.session_state["cargar"] = {k: b[k] for k in ("ri_int", "t_int", "gap", "t_ext", "h", "N")}
                st.rerun()
            if o["curva"]:
                cv = pd.DataFrame(o["curva"])
                fig, ax = figura(6.5, 4)
                ax.plot(cv[eje], cv[objetivo], "o-", ms=3, label="frente de Pareto")
                ax.plot(p[eje], p[objetivo], "k*", ms=14, label="paper")
                ax.plot(b[eje], b[objetivo], "D", c="C3", ms=8, label="optimizado")
                ax.set(xlabel=nombres[eje], ylabel=nombres[objetivo], title="Curva de compromiso")
                ax.margins(0.08)
                ax.legend(fontsize=8, loc="best", framealpha=0.9)
                c2.pyplot(fig)

# ---------------------------------------------------------------- historial
with tab4:
    ds = llamar("GET", "/disenos")
    if not ds:
        st.info("Todavía no guardaste diseños.")
    for d in ds:
        with st.expander(f"#{d['id']} · {d['nombre']} · {d['creado']}"):
            st.json(d["diseno"])
            if d["metricas"]:
                st.json(d["metricas"], expanded=False)
            if st.button("Borrar", key=f"borrar_{d['id']}"):
                llamar("DELETE", f"/disenos/{d['id']}")
                st.rerun()
