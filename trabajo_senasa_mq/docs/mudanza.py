"""Mudanza de proyecto/ y datos/ a trabajo_senasa_mq/ y _archivo/ (aprobada por el usuario, 2026-11-08).

    .venv/bin/python mudanza.py            # muestra el plan (no mueve nada)
    .venv/bin/python mudanza.py --aplicar  # mueve
    .venv/bin/python mudanza.py --registrar  # después: anota en el manifiesto lo creado (src/, README, notebook...)
                                             # (accion=crear) y lo generado por el notebook (accion=generar)

Cada operación se escribe en trabajo_senasa_mq/docs/manifiesto_movimientos.csv ANTES de hacerse
(n, accion, origen, destino, bytes, sha256, fecha). Nunca sobrescribe: si el destino existe, se detiene.
No borra nada: los directorios que quedan vacíos se quitan con rmdir y se anotan (accion=rmdir_vacio),
para que deshacer.py los vuelva a crear.
"""
import csv
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

IA = Path("/home/matiassbianco/IA")
T = IA / "trabajo_senasa_mq"
A = IA / "_archivo"
MANIFIESTO = T / "docs" / "manifiesto_movimientos.csv"
CAMPOS = ["n", "accion", "origen", "destino", "bytes", "sha256", "fecha"]

# --- lo que queda en su lugar (paquete del curso y lo que el usuario pidió no tocar)
QUEDAN = {"datos/__init__.py", "datos/catalogo.py", "datos/sinteticos.py", "datos/verificar_sustitutos.py",
          "datos/manifiesto.json", "datos/CLIENT.txt"}
QUEDAN_DIRS = ("datos/cache/", "datos/__pycache__/")

# --- lo que pasa al trabajo actual tal cual (mover)
AL_TRABAJO = {
    "proyecto/Senasa encolumnado.xlsx": "datos_entrada/Senasa encolumnado.xlsx",
    "proyecto/correcciones_clase41.csv": "datos_entrada/correcciones_clase41.csv",
    "datos/mq_procesado/muestras_ancha.csv": "datos_entrada/mq/muestras_ancha.csv",
    "datos/mq_procesado/tipo_muestra/mq_tipo_muestra.csv": "datos_entrada/mq/mq_tipo_muestra.csv",
    "proyecto/catalogo.py": "src/compartido/catalogo.py",
    "proyecto/anonimizar.py": "src/compartido/anonimizar.py",
    "proyecto/nombre_informe.py": "src/compartido/nombre_informe.py",
    "proyecto/Filtrado_de_Datos_2_ordenado.ipynb": "src/compartido/Filtrado_de_Datos_2_ordenado.ipynb",
    "proyecto/cortes_categorias.md": "docs/cortes_categorias.md",
}
for f in ["ceniza.csv", "cloruro.csv", "grasa 1.csv", "grasa.csv", "prote.csv", "st 2.csv", "st.csv"]:
    AL_TRABAJO[f"datos/datos mq/{f}"] = f"datos_entrada/mq/crudos/{f}"

# --- se copian al trabajo y el original va intacto a _archivo (la sección 7 los lee si REGENERAR_MQ=False)
COPIAS = {
    "proyecto/MQ encolumnado.xlsx": "resultados/excel/MQ encolumnado.xlsx",
    "proyecto/MQ encolumnado - solo nuevas.xlsx": "resultados/excel/MQ encolumnado - solo nuevas.xlsx",
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def plan():
    ops = []
    for rel, dst in COPIAS.items():
        ops.append(("copiar", IA / rel, T / dst))
    archivos = sorted(p for raiz in ("proyecto", "datos") for p in (IA / raiz).rglob("*") if p.is_file())
    for p in archivos:
        rel = p.relative_to(IA).as_posix()
        if rel in QUEDAN or rel.startswith(QUEDAN_DIRS):
            continue
        ops.append(("mover", p, T / AL_TRABAJO[rel] if rel in AL_TRABAJO else A / rel))
    falta = set(AL_TRABAJO) - {o.relative_to(IA).as_posix() for _, o, _ in ops}
    assert not falta, falta
    return ops


def main(aplicar):
    ops = plan()
    destinos = [d for _, _, d in ops]
    assert len(destinos) == len(set(destinos)), "dos operaciones al mismo destino"
    choques = [d for d in destinos if d.exists()]
    assert not choques, f"ya existen (no se sobrescribe): {choques[:5]}"
    resumen = {}
    for acc, o, d in ops:
        clave = (acc, "trabajo_senasa_mq" if d.is_relative_to(T) else "_archivo")
        n, b = resumen.get(clave, (0, 0)); resumen[clave] = (n + 1, b + o.stat().st_size)
    for k, (n, b) in sorted(resumen.items()):
        print(f"{k[0]:7} -> {k[1]:18} {n:4} archivos {b / 1e6:9.1f} MB")
    if not aplicar:
        for acc, o, d in ops:
            print(f"  {acc:7} {o.relative_to(IA)}  ->  {d.relative_to(IA)}")
        print("(plan; con --aplicar se mueve)")
        return
    MANIFIESTO.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not MANIFIESTO.exists()
    n0 = 0
    if not nuevo:
        with open(MANIFIESTO, encoding="utf-8") as f:
            n0 = max((int(r["n"]) for r in csv.DictReader(f)), default=0)
    with open(MANIFIESTO, "a", newline="", encoding="utf-8") as fm:
        w = csv.DictWriter(fm, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader(); fm.flush()
        n = n0
        for acc, o, d in ops:
            n += 1
            h = sha(o)
            w.writerow({"n": n, "accion": acc, "origen": o.relative_to(IA).as_posix(), "destino": d.relative_to(IA).as_posix(),
                        "bytes": o.stat().st_size, "sha256": h, "fecha": datetime.now().isoformat(timespec="seconds")})
            fm.flush()                                    # el renglón queda escrito ANTES de mover
            if d.exists():
                raise SystemExit(f"destino existente, me detengo: {d}")
            d.parent.mkdir(parents=True, exist_ok=True)
            if acc == "copiar":
                shutil.copy2(o, d)
            else:
                shutil.move(str(o), str(d))               # mismo disco: es un rename
            assert sha(d) == h, d
        # directorios que quedaron vacíos (de abajo hacia arriba); proyecto/ y datos/ se conservan
        vacios = sorted((p for raiz in ("proyecto", "datos") for p in (IA / raiz).rglob("*") if p.is_dir()),
                        key=lambda p: len(p.parts), reverse=True)
        for p in vacios:
            if p.exists() and not any(p.iterdir()):
                n += 1
                w.writerow({"n": n, "accion": "rmdir_vacio", "origen": p.relative_to(IA).as_posix(), "destino": "",
                            "bytes": 0, "sha256": "", "fecha": datetime.now().isoformat(timespec="seconds")})
                fm.flush()
                p.rmdir()
    print(f"Hecho: {n - n0} renglones en {MANIFIESTO}")


def registrar():
    """Anota los archivos de trabajo_senasa_mq/ que no están en el manifiesto (no mueve nada)."""
    with open(MANIFIESTO, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    ya = {r["destino"] for r in filas}
    n = max(int(r["n"]) for r in filas)
    nuevos = [p for p in sorted(T.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
              and p.relative_to(IA).as_posix() not in ya and p != MANIFIESTO]
    with open(MANIFIESTO, "a", newline="", encoding="utf-8") as fm:
        w = csv.DictWriter(fm, fieldnames=CAMPOS)
        for p in nuevos:
            n += 1
            acc = "generar" if p.is_relative_to(T / "resultados") else "crear"
            w.writerow({"n": n, "accion": acc, "origen": "", "destino": p.relative_to(IA).as_posix(),
                        "bytes": p.stat().st_size, "sha256": sha(p), "fecha": datetime.now().isoformat(timespec="seconds")})
            print(f"{acc:7} {p.relative_to(IA)}")


if __name__ == "__main__":
    if "--registrar" in sys.argv:
        registrar()
    else:
        main("--aplicar" in sys.argv)
