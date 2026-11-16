"""Gráficos para decidir el clustering categórico (k, cortes de la variante b, tags MIXTO) y el antes/después
de las mejoras del numérico (outliers por consenso, subtags del DDL).

    cd trabajo_senasa_mq
    PYTHONIOENCODING=utf-8 ../.venv/Scripts/python.exe src/graficos_decision.py        (en Linux: ../.venv/bin/python)

No lo llama ningún build ni notebook: se corre a mano después de regenerar resultados/tablas/ (corrido el notebook
trabajo_senasa_mq.ipynb). No depende de comparacion_numerico_categorico.ipynb (usa clustering_categorico.py, igual que él).

Lee (no modifica): resultados/tablas/clusters_*.csv, categorias_*.csv, cortes_categorias.csv y los datos de entrada (para
los DDL heladeros / reducidos en grasa, con mejoras_clustering.preparar / especiales_ddl).

Quesos y rama. clusters_*.csv trae las dos ramas: «k-medias» (con cluster y tag) y «queso (humedad y MG)» (sin cluster: sus
quesos no pasan por el k-medias). Todo lo de acá (k-modas, G1 a G5 y G6) usa SOLO las filas con rama == «k-medias»: el
k-modas se compara contra el tag numérico del k-medias, que los quesos no tienen. Entonces los quesos quedan afuera del
k-modas (y de los «antes y después»), igual que el k-medias; no hay un grupo QUESO ni quesos dentro de otros tags, salvo
los pocos de familia queso que el catálogo no confirma (siguen en el k-medias). clustering_categorico.cargar filtra con
cluster.notna() (equivale a rama == «k-medias»); acá se filtra además por la columna rama.

El «antes» de G6 ya no sale de una copia guardada de las tablas: se recalcula con las MISMAS muestras de hoy (sin los
quesos de la rama propia ni las familias no lácteas, con las reglas nuevas de preparar): regla IQR por grupo, k-medias con k por
silueta y los tags de etiquetar(), sin subtags. Así antes y después solo difieren en el método de outliers y en los subtags.
El k del k-modas es FIJO en CC.K_KMODAS = 5 (decisión de Matías, 2026-11-11; no sale de la silueta, que sube sola con k): G3, G4 y G5 se
dibujan con ese k en las dos versiones (g3/g4/g5_..._k5.png: el nombre sale de CC.K_KMODAS) y G3 también con el k del numérico cuando es
distinto (con semilla 42 y n_init 50 es k = 5 en las dos versiones, igual que el fijo, así que no se dibuja aparte); G1 muestra
la curva de 2 a 14 con el k fijo, el codo y el máximo de la silueta (en el borde); G2 compara los cortes en el k fijo y en el codo.
Para escribir en otra carpeta (pruebas), definir la variable de entorno GRAFICOS_SALIDA.
Escribe solo en resultados/graficos_decision/: g1_… a g6_… (PNG), resumen_g1_g2.csv (k de 2 a 14: costo, siluetas,
referencias al azar y ARI, por versión y variante), resumen_g2_barras.csv (las barras de G2) y tags_por_k.csv (el tag
de cada cluster categórico de la variante b, para cada k). textos.json (título, qué mirar y recomendación de cada PNG)
se escribió a mano con los números de una corrida muy anterior (k = 8 y 9 por silueta, semilla 2026) y NO lo usa nada: está viejo, no sirve y no se rehízo.

La variante (b) con terciles de la propia muestra es nueva y se calcula acá (no se guarda en tablas/): los cortes son
los terciles de cada análisis sobre las muestras del clustering de cada versión, con la misma convención que
categorizar.py (desde <= valor < hasta).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_samples

import clustering_categorico as CC
import rutas as R
from datos_clase41 import COLUMNAS, SEMILLA

SALIDA = Path(os.environ["GRAFICOS_SALIDA"]) if os.environ.get("GRAFICOS_SALIDA") else R.RESULTADOS / "graficos_decision"
DPI, ANCHO = 150, 1400 / 150                 # 1400 px de ancho
KS = CC.KS
REPETICIONES_AZAR = 10
CRIT_FIJO = f"k = {CC.K_KMODAS}"             # etiqueta del criterio «k fijo» en G2 (y en resumen_g2_barras.csv)

# ---------------------------------------------------------------- estilo (misma paleta en todos)
TINTA, TINTA2, TINTA3, GRILLA = "#1f1f1e", "#52514e", "#8a8984", "#e6e5e0"
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOR_TAG = {"DDL_REPOSTERO": "#00a6b2", "DDL": "#2a78d6", "DDL_FAMILIAR": "#6250d6", "DDL_HELADERO": "#1baf7a",
             "POLVO_MAGRO": "#eb6834", "POLVO_GRASO": "#eda100", "PROT_ALTA": "#e34948", "ALTA_HUMEDAD": "#e87ba4",
             "LIQUIDO": "#008300", "QUESO": "#8a5a2b", "POLVO": "#b8743a", "outlier": "#9a9993"}
ORDEN_TAGS = ["DDL", "DDL_FAMILIAR", "DDL_REPOSTERO", "DDL_HELADERO", "POLVO_MAGRO", "POLVO_GRASO", "PROT_ALTA", "ALTA_HUMEDAD",
              "LIQUIDO", "QUESO"]


def orden_tags(tags):
    """Los tags dados, en el orden de ORDEN_TAGS y, al final, los que no están (p. ej. un tag con sufijo _2: con semilla 42 y n_init 50 el k-medias ya no tiene POLVO_GRASO_2)."""
    tags = set(tags)
    return [x for x in ORDEN_TAGS if x in tags] + sorted(tags - set(ORDEN_TAGS))


SECUENCIAL = LinearSegmentedColormap.from_list("azul", ["#f4f8fd", "#a9caf0", "#2a78d6", "#0d3b75"])
VERSIONES = CC.VERSIONES
ETIQ_VAR = {"a": "(a) cortes por familia", "b": "(b) globales, cortes «*» de Senasa",
            "bt": "(b) globales, terciles de la muestra"}

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.edgecolor": TINTA3,
                     "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRILLA,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
                     "axes.formatter.use_locale": False})


def coma(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def color_tag(tag):
    """Color del tag (los MIXTO_<tag> y DDL_<n> heredan el del tag base)."""
    if tag in COLOR_TAG:
        return COLOR_TAG[tag]
    base = tag.replace("MIXTO_", "")
    if base in COLOR_TAG:
        return COLOR_TAG[base]
    return COLOR_TAG.get(base.rsplit("_", 1)[0], TINTA3)


def mezclar(hex_, f):
    """f > 0 oscurece (mezcla con negro), f < 0 aclara (mezcla con blanco)."""
    c = np.array(matplotlib.colors.to_rgb(hex_))
    c = c * (1 - f) if f > 0 else c + (1 - c) * (-f)
    return matplotlib.colors.to_hex(c)


TONOS = [0, 0.45, -0.45, 0.7, -0.65, 0.25, -0.25, 0.85]
MARCAS = ["o", "s", "^", "D", "v", "P", "X", "h"]


def estilo_clusters(perf):
    """Color y marca de cada cluster categórico: el color del tag base (consistente con el numérico); si varios
    clusters comparten tag base (DDL, DDL_2, …), cada uno lleva otro tono y otra marca."""
    vistos, out = {}, {}
    for g in perf.index:
        tg = perf.loc[g, "tag"]
        base = color_tag(tg)
        i = vistos.get(base, 0); vistos[base] = i + 1
        out[g] = (mezclar(base, TONOS[i % len(TONOS)]), MARCAS[i % len(MARCAS)])
    return out


def guardar(fig, nombre):
    fig.savefig(SALIDA / nombre, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("  ", nombre)


# ================================================================ cálculo
def terciles_muestra(t):
    """catt_*: terciles de cada análisis sobre las muestras t (desde <= valor < hasta). Devuelve (tabla, cortes)."""
    out, cortes = pd.DataFrame(index=t.index), []
    for a, col in CC.ANALISIS.items():
        q1, q2 = np.quantile(t[col], [1 / 3, 2 / 3])
        out[f"catt_{a}"] = np.select([t[col] < q1, t[col] < q2], ["bajo", "medio"], "alto")
        cortes.append({"analisis": a, "q1/3": q1, "q2/3": q2})
    return out, pd.DataFrame(cortes)


def cortes_estrella():
    c = pd.read_csv(R.TABLAS / "cortes_categorias.csv", encoding="utf-8-sig")
    c = c[(c["familia"] == "*") & c["analisis"].isin(list(CC.ANALISIS))]
    return c.pivot_table(index="analisis", columns="categoria", values="hasta")[["bajo", "medio"]]


def silueta_pre(D, et):
    s = silhouette_samples(D, et, metric="precomputed")
    return float(s.mean()), float(pd.Series(s).groupby(et).mean().mean())


def azar(D, X, et, rng):
    """Dos referencias del mismo k y tamaño:
    - muestras: las etiquetas del k-modas permutadas entre las muestras (mismos tamaños de grupo);
    - perfiles: cada perfil distinto (fila de categorías) a un grupo al azar; las muestras idénticas quedan juntas, que es
      lo que hace subir la silueta de Hamming «sola» con k."""
    k = len(np.unique(et))
    perm = [silueta_pre(D, rng.permutation(et))[0] for _ in range(REPETICIONES_AZAR)]
    _, inv = np.unique(X, axis=0, return_inverse=True)
    inv = inv.ravel(); m = inv.max() + 1
    perf = []
    for _ in range(REPETICIONES_AZAR):
        g = rng.integers(0, k, m); g[rng.permutation(m)[:k]] = np.arange(k)     # que estén los k grupos
        perf.append(silueta_pre(D, g[inv])[0])
    return float(np.mean(perm)), float(np.mean(perf))


def evaluar(t, et):
    ari_f, _ = CC.contra(et, t["familia"].where(t["familia_conocida"]))
    ddl = (t["familia"] == "dulce de leche").values
    ari_d, _ = CC.contra(et, t["subtipo_eval"].where(t["subtipo_eval"].str.startswith("DDL", na=False)), ddl)
    return {"ARI numérico": adjusted_rand_score(t["tag_num"], et), "ARI familia": ari_f, "ARI subtipo DDL": ari_d}


def calcular():
    rng = np.random.default_rng(SEMILLA)
    res, filas = {}, []
    for v in VERSIONES:
        print("calculando", VERSIONES[v])
        t = CC.cargar(v)
        tt, cortes_t = terciles_muestra(t)
        t = pd.concat([t, tt], axis=1)
        variantes = {"a": CC.CAT_A, "b": CC.CAT_B, "bt": [f"catt_{a}" for a in CC.ANALISIS]}
        res[v] = {"t": t, "cortes_t": cortes_t, "var": {}}
        for clave, cols in variantes.items():
            X, _ = CC.codificar(t, cols)
            D = pairwise_distances(X, metric="hamming")
            part, filas_v = {}, []
            for k in KS:
                et, _, costo = CC.kmodas(X, k)
                sg, sm = silueta_pre(D, et)
                az_m, az_p = azar(D, X, et, rng)
                n = np.bincount(et)
                part[k] = et
                filas_v.append({"version": v, "variante": clave, "k": k, "costo_kmodas": costo,
                                "silueta_hamming": sg, "silueta_macro_hamming": sm,
                                "silueta_azar_muestras": az_m, "silueta_azar_perfiles": az_p,
                                "grupo_min_%": 100 * n.min() / len(et), "perfiles_distintos": len(np.unique(X, axis=0)),
                                **evaluar(t, et)})
            tab = pd.DataFrame(filas_v)
            k_codo = CC.codo(tab["k"], tab["costo_kmodas"])
            ok = tab[tab["grupo_min_%"] >= 100 * CC.TAM_MIN]
            k_sil = int(ok.loc[ok["silueta_hamming"].idxmax(), "k"])
            tab["k_codo"] = tab["k"] == k_codo
            tab["k_silueta_max"] = tab["k"] == k_sil
            filas.append(tab)
            tab["k_fijo"] = tab["k"] == CC.K_KMODAS
            res[v]["var"][clave] = {"cols": cols, "X": X, "part": part, "tabla": tab, "k_codo": k_codo, "k_sil": k_sil,
                                    "k_fijo": CC.K_KMODAS}
            print(f"   {clave}: k fijo {CC.K_KMODAS} · codo {k_codo} · silueta máx {k_sil}")
    resumen = pd.concat(filas, ignore_index=True)
    resumen.insert(1, "versión", resumen["version"].map(VERSIONES))
    return res, resumen


# ================================================================ G1 · elegir k
def g1(res):
    fig = plt.figure(figsize=(ANCHO, 9.6))
    sub = fig.subfigures(2, 2, hspace=0.06, wspace=0.04)
    for i, v in enumerate(VERSIONES):
        for j, clave in enumerate(["a", "b"]):
            d = res[v]["var"][clave]; tab = d["tabla"]
            sf = sub[i, j]
            a1, a2 = sf.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.08})
            sf.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.11)
            sf.suptitle(f"{VERSIONES[v]} · {ETIQ_VAR[clave]}", fontsize=10.5, fontweight="bold", x=0.06, ha="left", y=0.99)
            a1.plot(tab["k"], tab["costo_kmodas"] / 1000, color=SERIE[0], lw=2, marker="o", ms=5)
            kc = d["k_codo"]; yc = tab.loc[tab["k"] == kc, "costo_kmodas"].iloc[0] / 1000
            a1.plot([kc], [yc], "o", ms=11, mfc="none", mec=TINTA, mew=1.6)
            a1.annotate(f"codo: k = {kc}", (kc, yc), xytext=(12, 10), textcoords="offset points", fontsize=9.5,
                        color=TINTA, fontweight="bold")
            kf = d["k_fijo"]
            for ax_ in (a1, a2):
                ax_.axvline(kf, color=SERIE[0], lw=1.4, ls="--", alpha=0.8)
            a1.text(kf + 0.15, a1.get_ylim()[1] * 0.97, f"k = {kf} (fijo)", color=SERIE[0], fontsize=9, va="top",
                    fontweight="bold")
            a1.set_ylabel("costo k-modas\n(miles de desacuerdos)", fontsize=9)
            a2.plot(tab["k"], tab["silueta_hamming"], color=SERIE[1], lw=2, marker="o", ms=5, label="k-modas")
            a2.plot(tab["k"], tab["silueta_azar_perfiles"], color=TINTA3, lw=1.6, ls="--", marker="s", ms=4,
                    label="al azar, perfiles idénticos juntos")
            a2.plot(tab["k"], tab["silueta_azar_muestras"], color=TINTA3, lw=1.2, ls=":", marker="^", ms=4,
                    label="al azar, muestras sueltas")
            ks_ = d["k_sil"]; ys = tab.loc[tab["k"] == ks_, "silueta_hamming"].iloc[0]
            a2.plot([ks_], [ys], "o", ms=11, mfc="none", mec=TINTA, mew=1.6)
            a2.annotate(f"máx: k = {ks_}", (ks_, ys), xytext=(-8, 12), textcoords="offset points", ha="right",
                        fontsize=9.5, color=TINTA, fontweight="bold")
            a2.set_ylabel("silueta (Hamming)", fontsize=9)
            a2.set_ylim(min(-0.1, tab["silueta_azar_muestras"].min() - 0.05), 0.92)
            a2.axhline(0, color=TINTA3, lw=0.8)
            a2.set_xlabel("k (número de clusters)")
            a2.set_xticks(KS)
            if i == 0 and j == 0:
                a2.legend(loc="upper left", fontsize=8.5, ncol=1)
    guardar(fig, "g1_elegir_k.png")


# ================================================================ G2 · cortes de la variante b
METRICAS_G2 = [("silueta_hamming", "silueta\nHamming"), ("silueta_macro_hamming", "silueta macro\nHamming"),
               ("ARI numérico", "ARI contra\nel numérico"), ("ARI familia", "ARI contra\nla familia"),
               ("ARI subtipo DDL", "ARI contra el\nsubtipo DDL")]


def g2_tabla(res):
    filas = []
    for v in VERSIONES:
        for clave in ["b", "bt"]:
            d = res[v]["var"][clave]
            for crit, k in [("codo", d["k_codo"]), (CRIT_FIJO, d["k_fijo"])]:
                f = d["tabla"].set_index("k").loc[k]
                filas.append({"version": v, "versión": VERSIONES[v], "cortes": ETIQ_VAR[clave], "variante": clave,
                              "criterio": crit, "k": k, **{m: f[m] for m, _ in METRICAS_G2}})
    return pd.DataFrame(filas)


def g2(res, tab2):
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 5.6), sharey=True)
    series = [("b", CRIT_FIJO, SERIE[0], None), ("bt", CRIT_FIJO, SERIE[1], None)]
    w = 0.3
    for ax, v in zip(ejes, VERSIONES):
        tv = tab2[tab2["version"] == v]
        for s, (clave, crit, col, hatch) in enumerate(series):
            f = tv[(tv["variante"] == clave) & (tv["criterio"] == crit)].iloc[0]
            x = np.arange(len(METRICAS_G2)) + (s - 0.5) * (w + 0.02)
            y = [f[m] for m, _ in METRICAS_G2]
            ax.bar(x, y, w, color=col if hatch is None else "white", edgecolor=col, hatch=hatch, lw=1.4)
            for xi, yi in zip(x, y):
                ax.text(xi, yi + 0.012, coma(yi), ha="center", va="bottom", fontsize=7.2, color=TINTA2, rotation=90)
        ax.set_xticks(np.arange(len(METRICAS_G2)), [e for _, e in METRICAS_G2], fontsize=8.5)
        ax.set_title(VERSIONES[v], fontweight="bold")
        ax.axhline(0, color=TINTA3, lw=0.8)
        ax.set_ylim(-0.02, 0.9); ax.grid(axis="x", visible=False)
        k_ = {(r.variante, r.criterio): r.k for r in tv.itertuples()}
        mango = [Patch(fc=SERIE[0], ec=SERIE[0], label=f"cortes «*» de Senasa (k = {k_[('b', CRIT_FIJO)]})"),
                 Patch(fc=SERIE[1], ec=SERIE[1], label=f"terciles de la muestra (k = {k_[('bt', CRIT_FIJO)]})")]
        ax.legend(handles=mango, loc="upper right", fontsize=8.3)
    ejes[0].set_ylabel("valor (más alto = mejor)")
    fig.tight_layout()
    guardar(fig, "g2_cortes_variante_b.png")


# ================================================================ G3 · tag numérico × cluster categórico
def perfil_b(res, v, k, variante="b"):
    t = res[v]["t"]; d = res[v]["var"][variante]
    et = d["part"][k]
    perf, tags = CC.perfil(t, et, d["cols"])
    return et, perf, tags


def g3(res, v, k, variante="b"):
    t = res[v]["t"]
    et, perf, tags = perfil_b(res, v, k, variante)
    ct = pd.crosstab(t["tag_num"], et)
    filas = orden_tags(ct.index)
    ct = ct.loc[filas]
    prop = ct / ct.sum(axis=0)
    fig, ax = plt.subplots(figsize=(ANCHO, 2.2 + 0.5 * len(filas)))
    ax.imshow(prop.values, cmap=SECUENCIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(prop.shape[0]):
        for j in range(prop.shape[1]):
            n = ct.iat[i, j]
            if n == 0:
                continue
            p = prop.iat[i, j]
            ax.text(j, i, f"{n}\n{100 * p:.0f} %", ha="center", va="center", fontsize=8.5,
                    color="white" if p > 0.55 else TINTA)
    etiq = []
    for j, g in enumerate(ct.columns):
        tg = perf.loc[g, "tag"]
        if len(ct.columns) > 8:                         # con muchos clusters el rótulo se parte para que no se pise
            etiq.append(f"c{g}\nn={int(ct[g].sum())}\n{tg.replace('_', chr(10), 1)}")
        else:
            etiq.append(f"c{g} · n={int(ct[g].sum())}\n{tg}")
    ax.set_xticks(range(len(etiq)), etiq, fontsize=7.8 if len(ct.columns) > 8 else 8.3)
    for lab, g in zip(ax.get_xticklabels(), ct.columns):
        tg = perf.loc[g, "tag"]
        if tg.startswith("MIXTO"):
            lab.set_fontweight("bold"); lab.set_color(TINTA)
            j = list(ct.columns).index(g)
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(filas), fill=False, ec=TINTA, lw=2.2, clip_on=False))
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(filas)), filas)
    ax.set_ylabel("tag numérico (k-medias)")
    cortes = "cortes «*» de Senasa" if variante == "b" else "terciles de la muestra"
    ax.set_xlabel(f"cluster categórico (k-modas, variante b con {cortes}) y el tag que le asigna perfil()", labelpad=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, prop.shape[1]), minor=True); ax.set_yticks(np.arange(-0.5, prop.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0); ax.tick_params(length=0)
    ax.set_title(f"{VERSIONES[v]} · k = {k} · % dentro de cada cluster (columna) · recuadro = tag MIXTO",
                 fontsize=10, loc="left", pad=48 if len(ct.columns) > 8 else 36, color=TINTA2)
    fig.tight_layout()
    guardar(fig, f"g3_tags_mixto_{v}_k{k}{'' if variante == 'b' else '_terciles'}.png")
    return ct, perf


# ================================================================ G4 · perfil de los clusters categóricos
def g4(res, v, k):
    t = res[v]["t"]; d = res[v]["var"]["b"]
    et, perf, _ = perfil_b(res, v, k)
    nombres = {"catg_humedad": "humedad", "catg_materia_grasa": "grasa", "catg_proteinas": "proteínas",
               "catg_cenizas": "cenizas"}
    cortes = cortes_estrella()
    grupos = sorted(np.unique(et))
    M = np.zeros((len(grupos), len(d["cols"]))); L = [[""] * len(d["cols"]) for _ in grupos]
    for i, g in enumerate(grupos):
        for j, c in enumerate(d["cols"]):
            vc = t.loc[et == g, c].value_counts(normalize=True)
            M[i, j] = vc.iloc[0]; L[i][j] = f"{vc.index[0]}\n{100 * vc.iloc[0]:.0f} %"
    fig, ax = plt.subplots(figsize=(ANCHO, 2.4 + 0.5 * len(grupos)))
    im = ax.imshow(M, cmap=SECUENCIAL, vmin=0.3, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, L[i][j], ha="center", va="center", fontsize=8.5, color="white" if M[i, j] > 0.75 else TINTA)
    col_lab = []
    for c in d["cols"]:
        a = c[5:]
        b, m = cortes.loc[a, "bajo"], cortes.loc[a, "medio"]
        col_lab.append(f"{nombres[c]}\nbajo < {coma(b)}\nalto ≥ {coma(m)}")
    ax.set_xticks(range(len(col_lab)), col_lab, fontsize=8.8); ax.xaxis.tick_top()
    filas = [f"c{g} · {perf.loc[g, 'tag']} · n={perf.loc[g, 'muestras']}" for g in grupos]
    ax.set_yticks(range(len(filas)), filas, fontsize=9)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True); ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("proporción de la categoría modal", fontsize=9); cb.outline.set_visible(False)
    ax.set_title(f"{VERSIONES[v]} · k-modas (b), k = {k} · cortes en g/100 g", fontsize=10, loc="left", pad=48,
                 color=TINTA2)
    fig.tight_layout()
    guardar(fig, f"g4_perfil_clusters_{v}_k{k}.png")
    return M, perf


# ================================================================ G5 · humedad contra proteínas
ZOOM = {"humedad": (17, 40), "proteinas": (2, 12)}


def g5(res, v, k):
    """Arriba, todo el rango; abajo, el recuadro del dulce de leche (donde se superponen casi todos los clusters)."""
    t = res[v]["t"]
    et, perf, _ = perfil_b(res, v, k)
    fig, ejes = plt.subplots(2, 2, figsize=(ANCHO, 11.5), gridspec_kw={"height_ratios": [1, 1]})
    # DDL con su subtag; el resto con el tag del k-medias (los subtags de polvo no se dibujan: son parte de su tag)
    st = t["subtag"].where(t["subtag"].str.startswith("DDL", na=False), t["tag_num"])
    orden = orden_tags(set(st))
    est = estilo_clusters(perf)
    cortes = cortes_estrella()
    for fila, zoom in enumerate([False, True]):
        a1, a2 = ejes[fila]
        tam = 12 if zoom else 7
        for x in orden:
            s = st == x
            a1.scatter(t.loc[s, "humedad"], t.loc[s, "proteinas"], s=tam, alpha=0.4, color=color_tag(x), lw=0,
                       label=f"{x} ({s.sum()})")
        for g in sorted(np.unique(et)):
            s = et == g
            col, mk = est[g]
            a2.scatter(t.loc[s, "humedad"], t.loc[s, "proteinas"], s=tam if mk == "o" else tam * 1.5, alpha=0.4,
                       color=col, marker=mk, lw=0, label=f"c{g} {perf.loc[g, 'tag']} ({s.sum()})")
        for c in cortes.loc["humedad"]:
            a2.axvline(c, color=TINTA3, lw=0.9, ls="--")
        for c in cortes.loc["proteinas"]:
            a2.axhline(c, color=TINTA3, lw=0.9, ls="--")
        for ax in (a1, a2):
            ax.set_xlabel("humedad (g/100 g)")
            if zoom:
                ax.set_xlim(*ZOOM["humedad"]); ax.set_ylim(*ZOOM["proteinas"])
            else:
                ax.set_xlim(-3, 102); ax.set_ylim(-3, 96)
                ax.add_patch(plt.Rectangle((ZOOM["humedad"][0], ZOOM["proteinas"][0]), np.diff(ZOOM["humedad"])[0],
                                           np.diff(ZOOM["proteinas"])[0], fill=False, ec=TINTA, lw=1.2))
                lg = ax.legend(loc="upper right", fontsize=8, markerscale=2.2, handletextpad=0.2, borderaxespad=0.2)
                for h in lg.legend_handles:
                    h.set_alpha(1)
        a1.set_ylabel("proteínas (g/100 g)")
        if zoom:
            a1.set_title("detalle del recuadro: tag numérico", fontweight="bold", loc="left", fontsize=10.5)
            a2.set_title("detalle del recuadro: cluster categórico", fontweight="bold", loc="left", fontsize=10.5)
        else:
            a1.set_title("tag numérico (DDL con su subtag)", fontweight="bold", loc="left", fontsize=10.5)
            a2.set_title(f"cluster categórico (k-modas b, k = {k})", fontweight="bold", loc="left", fontsize=10.5)
    ch, cp = cortes.loc["humedad"], cortes.loc["proteinas"]
    fig.suptitle(f"{VERSIONES[v]} · {len(t)} muestras · líneas punteadas: cortes «*» de humedad ({coma(ch['bajo'])} y "
                 f"{coma(ch['medio'])}) y de proteínas ({coma(cp['bajo'])} y {coma(cp['medio'])})",
                 fontsize=10, color=TINTA2, x=0.01, ha="left")
    fig.tight_layout()
    guardar(fig, f"g5_humedad_proteinas_{v}_k{k}.png")


# ================================================================ G6 · antes y después de las mejoras
def leer_clusters(v):
    """clusters_<v>.csv de hoy, solo la rama k-medias (las filas de queso no tienen cluster ni tag)."""
    c = pd.read_csv(R.TABLAS / f"clusters_{v}.csv", encoding="utf-8-sig")
    return c[c["rama"] == "k-medias"].reset_index(drop=True)


def antes_iqr(m):
    """El «antes» con las muestras de hoy (m = mejoras_clustering.preparar): outliers por la regla IQR por grupo, k-medias
    con k por silueta (agrupar) sobre las que quedan, tags de etiquetar() y sin subtags. Devuelve una tabla con las columnas
    de clusters_*.csv que usa G6 (outlier, tag, silueta, familia, humedad, proteinas)."""
    import mejoras_clustering as M
    X = m[COLUMNAS].to_numpy()
    Z, _, _ = M.escalar(X)
    es, _, _ = M.outliers_iqr(Z)
    r = M.agrupar(X[~es])
    c = m[["Identificador", "desc_familia", "humedad", "proteinas"]].rename(columns={"desc_familia": "familia"}).copy()
    c["outlier"] = es
    c["tag"] = np.nan; c["tag"] = c["tag"].astype(object)
    c["silueta"] = np.nan
    c.loc[~es, "tag"] = r["tag_de"]
    c.loc[~es, "silueta"] = silhouette_samples(r["Z"], r["grupos"])
    return c


def datos_g6():
    import mejoras_clustering as M
    S, out = None, {}
    for v, mq in [("senasa", False), ("senasa_mq", True)]:
        m, S = M.preparar(mq, S)
        esp = M.especiales_ddl(m)
        m_completo = m
        m = pd.concat([m[["Identificador"]], esp], axis=1)
        d = {}
        for cuando in ["antes", "después"]:
            c = antes_iqr(m_completo) if cuando == "antes" else leer_clusters(v)
            x = m.merge(c[["Identificador", "outlier"]], on="Identificador", how="left", validate="one_to_one")
            ok = ~x["outlier"].astype(bool)
            sil = c[c["tag"].notna()].groupby("tag")["silueta"].agg(["mean", "size"])
            d[cuando] = {"c": c, "% outliers": 100 * c["outlier"].mean(),
                         "heladero": (int((x["heladero"] & ok).sum()), int(x["heladero"].sum())),
                         "reducido en grasa": (int((x["reducido en grasa"] & ok).sum()), int(x["reducido en grasa"].sum())),
                         "repostero": (int((x["repostero"] & ok).sum()), int(x["repostero"].sum())),
                         "silueta": sil}
        out[v] = d
    return out


def g6_outliers(D):
    fig, ejes = plt.subplots(1, 3, figsize=(ANCHO, 4.8))
    cols = {"antes": TINTA3, "después": SERIE[0]}
    w = 0.36
    x = np.arange(len(VERSIONES))
    for s, cuando in enumerate(["antes", "después"]):
        xs = x + (s - 0.5) * (w + 0.03)
        y = [D[v][cuando]["% outliers"] for v in VERSIONES]
        ejes[0].bar(xs, y, w, color=cols[cuando], label=cuando)
        for xi, yi in zip(xs, y):
            ejes[0].text(xi, yi + 0.15, f"{coma(yi, 1)} %", ha="center", va="bottom", fontsize=9)
        for ax, clave in [(ejes[1], "heladero"), (ejes[2], "reducido en grasa")]:
            con = [D[v][cuando][clave][0] for v in VERSIONES]
            tot = [D[v][cuando][clave][1] for v in VERSIONES]
            ax.bar(xs, con, w, color=cols[cuando], label=cuando)
            for xi, c_, t_ in zip(xs, con, tot):
                ax.text(xi, c_ + max(tot) * 0.015, f"{c_}", ha="center", va="bottom", fontsize=9)
    ejes[0].set_title("% de las muestras completas\nmarcadas como outlier", loc="left", fontweight="bold")
    ejes[0].set_ylim(0, 11.5)
    ejes[0].set_ylabel("% de las muestras completas")
    for ax, clave in [(ejes[1], "heladero"), (ejes[2], "reducido en grasa")]:
        tot = D["senasa"]["después"][clave][1]
        ax.axhline(tot, color=TINTA, lw=1, ls="--")
        ax.text(-0.45, tot * 1.02, f"total en la familia: {tot}", ha="left", va="bottom", fontsize=8.5, color=TINTA2)
        ax.set_ylim(0, tot * 1.2)
        ax.set_title("DDL heladeros que quedan\nagrupados (no outlier)" if clave == "heladero"
                     else "DDL reducidos en grasa\nque quedan agrupados", loc="left", fontweight="bold")
        ax.set_ylabel("muestras")
    for ax in ejes:
        ax.set_xticks(x, list(VERSIONES.values())); ax.grid(axis="x", visible=False)
    fig.legend(handles=[Patch(fc=cols["antes"], label="antes"), Patch(fc=cols["después"], label="después")],
               loc="upper right", ncol=2, fontsize=9.5)
    fig.suptitle("Antes: regla IQR por grupo · después: outliers por consenso (2 de 3 métodos) al 3 %",
                 x=0.01, ha="left", fontsize=10, color=TINTA2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    guardar(fig, "g6a_outliers_y_ddl_conservados.png")


def g6_silueta(D):
    fig, ejes = plt.subplots(1, 2, figsize=(ANCHO, 4.9), sharey=True)
    w = 0.38
    for ax, v in zip(ejes, VERSIONES):
        tags = orden_tags(set(D[v]["antes"]["silueta"].index) | set(D[v]["después"]["silueta"].index))
        x = np.arange(len(tags))
        for s, cuando in enumerate(["antes", "después"]):
            sil = D[v][cuando]["silueta"].reindex(tags)
            xs = x + (s - 0.5) * (w + 0.02)
            cols = [color_tag(tg) for tg in tags]
            if cuando == "antes":
                ax.bar(xs, sil["mean"], w, color="white", edgecolor=cols, lw=1.5, hatch="///")
            else:
                ax.bar(xs, sil["mean"], w, color=cols)
            for xi, m_ in zip(xs, sil["mean"].values):
                if np.isfinite(m_):
                    ax.text(xi, m_ + 0.01, coma(m_), ha="center", va="bottom", fontsize=8, color=TINTA2)
        na, nd = D[v]["antes"]["silueta"]["size"], D[v]["después"]["silueta"]["size"]
        ax.set_xticks(x, [f"{tg.replace('_', '_' + chr(10)) if '_' in tg else tg + chr(10)}\n{na.get(tg, 0)}→{nd.get(tg, 0)}"
                          for tg in tags], fontsize=8)
        ca, cd = D[v]["antes"]["c"], D[v]["después"]["c"]
        ax.set_title(f"{VERSIONES[v]} · silueta global {coma(ca['silueta'].mean(), 3)} → {coma(cd['silueta'].mean(), 3)}",
                     loc="left", fontweight="bold", fontsize=10)
        ax.set_ylim(0, 1.12); ax.grid(axis="x", visible=False)
    ejes[0].set_ylabel("silueta media del tag (euclídea)")
    fig.text(0.01, 0.01, "debajo de cada tag: muestras antes → después", fontsize=8.5, color=TINTA2)
    mango = [Patch(fc="white", ec=TINTA2, hatch="///", label="antes"), Patch(fc=TINTA2, label="después")]
    ejes[1].legend(handles=mango, loc="upper right", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    guardar(fig, "g6b_silueta_por_tag.png")


def g6_ddl(D):
    fig, ejes = plt.subplots(2, 2, figsize=(ANCHO, 10), sharex=True, sharey=True)
    for i, v in enumerate(VERSIONES):
        ca, cd = D[v]["antes"]["c"], D[v]["después"]["c"]
        ddl_a = ca[ca["familia"] == "dulce de leche"]
        ddl_d = cd[cd["familia"] == "dulce de leche"]
        a1, a2 = ejes[i]
        ok = ~ddl_a["outlier"].astype(bool)
        a1.scatter(ddl_a.loc[ok, "humedad"], ddl_a.loc[ok, "proteinas"], s=7, alpha=0.4, color=COLOR_TAG["DDL"], lw=0,
                   label=f"agrupada ({ok.sum()})")
        a1.scatter(ddl_a.loc[~ok, "humedad"], ddl_a.loc[~ok, "proteinas"], s=14, alpha=0.8, color=SERIE[7], marker="x",
                   lw=0.9, label=f"outlier ({(~ok).sum()})")
        a1.set_title(f"{VERSIONES[v]} · antes (sin subtags)", loc="left", fontweight="bold")
        ok = ~ddl_d["outlier"].astype(bool)
        sub = ddl_d["subtag"].where(ok)
        for x, nombre in [("DDL", "DDL (resto)"), ("DDL_FAMILIAR", "DDL_FAMILIAR"), ("DDL_REPOSTERO", "DDL_REPOSTERO"),
                          ("DDL_HELADERO", "DDL_HELADERO")]:
            s = sub == x
            a2.scatter(ddl_d.loc[s, "humedad"], ddl_d.loc[s, "proteinas"], s=7, alpha=0.45, color=COLOR_TAG[x], lw=0,
                       label=f"{nombre} ({s.sum()})")
        otros = ok & ~ddl_d["subtag"].isin(["DDL", "DDL_FAMILIAR", "DDL_REPOSTERO", "DDL_HELADERO"])
        if otros.any():
            a2.scatter(ddl_d.loc[otros, "humedad"], ddl_d.loc[otros, "proteinas"], s=7, alpha=0.6, color=TINTA3,
                       lw=0, label=f"en otro tag ({otros.sum()})")
        a2.scatter(ddl_d.loc[~ok, "humedad"], ddl_d.loc[~ok, "proteinas"], s=14, alpha=0.8, color=SERIE[7], marker="x",
                   lw=0.9, label=f"outlier ({(~ok).sum()})")
        a2.set_title(f"{VERSIONES[v]} · después (por subtag)", loc="left", fontweight="bold")
        for ax in (a1, a2):
            lg = ax.legend(loc="upper left", fontsize=8.5, markerscale=2, frameon=True, facecolor="white",
                           edgecolor="white", framealpha=0.92)
            for h in lg.legend_handles:
                h.set_alpha(1)
        a1.set_ylabel("proteínas (g/100 g)")
        lejos = int((cd.loc[cd["familia"] == "dulce de leche", "humedad"] > 42).sum())
        for ax in (a1, a2):
            ax.set_xlim(17, 42)
            ax.text(0.99, 0.01, f"{lejos} muestras con humedad > 42 quedan fuera del recuadro", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=TINTA3)
    for ax in ejes[1]:
        ax.set_xlabel("humedad (g/100 g)")
    fig.suptitle("Familia dulce de leche (todas sus muestras completas, incluidas las que el k-medias manda a otro tag)",
                 x=0.01, ha="left", fontsize=10, color=TINTA2)
    fig.tight_layout()
    guardar(fig, "g6c_ddl_humedad_proteinas_subtag.png")


# ================================================================ principal
def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    cache = os.environ.get("GRAFICOS_CACHE")          # solo para iterar sobre los dibujos (pickle del cálculo)
    if cache and Path(cache).exists():
        res, resumen = pd.read_pickle(cache)
    else:
        res, resumen = calcular()
        if cache:
            pd.to_pickle((res, resumen), cache)
    tab2 = g2_tabla(res)
    resumen.to_csv(SALIDA / "resumen_g1_g2.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    tab2.to_csv(SALIDA / "resumen_g2_barras.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    print(resumen.round(3).to_string())
    print(tab2.round(3).to_string())
    for v in VERSIONES:
        print(VERSIONES[v], "terciles de la muestra:\n", res[v]["cortes_t"].round(3).to_string())
    print("gráficos")
    g1(res)
    g2(res, tab2)
    for v in VERSIONES:
        d = res[v]["var"]["b"]
        k = d["k_fijo"]                       # CC.K_KMODAS (5), fijo: G3, G4 y G5
        ct, perf = g3(res, v, k)
        print(perf[["tag", "muestras", "tag numérico mayoritario"]].to_string())
        g4(res, v, k)
        g5(res, v, k)
        k_num = int(res[v]["t"]["tag_num"].nunique())
        if k_num != k:                        # el k del k-medias numérico, solo para ver cómo cambia el k-modas con un grupo menos
            ct, perf = g3(res, v, k_num)
            print(f"(k del numérico = {k_num})")
            print(perf[["tag", "muestras", "tag numérico mayoritario"]].to_string())
    # decisión 3: qué tags MIXTO / POLVO aparecen con cada k (las dos variantes de cortes de b)
    filas = []
    for v in VERSIONES:
        for clave in ["b", "bt"]:
            for k in KS:
                _, perf, _ = perfil_b(res, v, k, clave)
                for g, r in perf.iterrows():
                    filas.append({"version": v, "cortes": ETIQ_VAR[clave], "k": k, "cluster": g, "tag": r["tag"],
                                  "muestras": r["muestras"], "tag numérico mayoritario": r["tag numérico mayoritario"],
                                  "mixto": r["tag"].startswith("MIXTO")})
    pd.DataFrame(filas).to_csv(SALIDA / "tags_por_k.csv", index=False, encoding="utf-8-sig")
    d = res["senasa_mq"]["var"]["bt"]
    g3(res, "senasa_mq", d["k_fijo"], "bt")        # los terciles con el k fijo
    D = datos_g6()
    g6_outliers(D)
    g6_silueta(D)
    g6_ddl(D)
    return res, resumen, tab2, D


if __name__ == "__main__":
    main()
