"""Deshace la mudanza de proyecto/ y datos/ (ver manifiesto_movimientos.csv, en esta misma carpeta).

    .venv/bin/python trabajo_senasa_mq/docs/deshacer.py            # dry run: solo muestra lo que haría
    .venv/bin/python trabajo_senasa_mq/docs/deshacer.py --aplicar  # lo hace

Recorre el manifiesto al revés:
  - mover        vuelve a mover destino → origen, si el sha256 del destino es el del manifiesto;
  - copiar       la copia (destino) se aparta a _archivo/_deshacer_<fecha>/ (el original vuelve por su renglón «mover»);
  - crear        lo creado (y lo generado por el notebook, accion=generar) se aparta a _archivo/_deshacer_<fecha>/;
  - rmdir_vacio  vuelve a crear el directorio vacío.
Al final aparta también a _archivo/_deshacer_<fecha>/ lo que quede en trabajo_senasa_mq/ y no esté en el manifiesto
(los resultados que regeneró el notebook, cachés). Nunca sobrescribe ni borra: si un archivo que tiene que volver
cambió (sha256 distinto) o su lugar ya está ocupado, no lo toca y lo informa.
"""
import csv
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

DOCS = Path(__file__).resolve().parent
TRABAJO = DOCS.parent
IA = TRABAJO.parent
MANIFIESTO = DOCS / "manifiesto_movimientos.csv"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(aplicar):
    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    apartado = IA / "_archivo" / f"_deshacer_{sello}"
    with open(MANIFIESTO, encoding="utf-8") as f:
        filas = sorted(csv.DictReader(f), key=lambda r: int(r["n"]), reverse=True)
    # el manifiesto y este script se leen hasta el final: se apartan al terminar
    propios = {MANIFIESTO.resolve(), Path(__file__).resolve()}
    hechos, problemas, conteo = [], [], {}

    def apartar(p, motivo):
        dst = apartado / p.relative_to(IA)
        if dst.exists():
            problemas.append(f"ya existe {dst}; no se aparta {p}")
            return
        hechos.append(f"apartar  {p.relative_to(IA)}  ->  {dst.relative_to(IA)}   ({motivo})")
        if aplicar:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))

    listados = set()
    for r in filas:
        acc = r["accion"]; conteo[acc] = conteo.get(acc, 0) + 1
        o = IA / r["origen"]; d = IA / r["destino"] if r["destino"] else None
        if d is not None:
            listados.add(d.resolve())
        if acc == "rmdir_vacio":
            hechos.append(f"mkdir    {o.relative_to(IA)}")
            if aplicar:
                o.mkdir(parents=True, exist_ok=True)
        elif acc == "mover":
            if not d.exists():
                problemas.append(f"falta {r['destino']} (renglón {r['n']}); no se puede devolver a {r['origen']}")
            elif o.exists():
                problemas.append(f"{r['origen']} ya existe; no se sobrescribe (renglón {r['n']})")
            elif sha(d) != r["sha256"]:
                problemas.append(f"{r['destino']} cambió desde la mudanza (sha256 distinto); queda donde está (renglón {r['n']})")
            else:
                hechos.append(f"mover    {r['destino']}  ->  {r['origen']}")
                if aplicar:
                    o.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(d), str(o))
        elif acc in ("copiar", "crear", "generar"):
            if d.resolve() in propios:
                continue
            if d.exists():
                cambio = "" if not r["sha256"] or sha(d) == r["sha256"] else ", cambió desde que se registró"
                apartar(d, f"{acc}{cambio}")
            else:
                problemas.append(f"falta {r['destino']} (renglón {r['n']}, {acc})")
        else:
            problemas.append(f"acción desconocida {acc} (renglón {r['n']})")

    # lo que quedó en trabajo_senasa_mq/ sin estar en el manifiesto (resultados regenerados, cachés)
    if TRABAJO.exists():
        for p in sorted(TRABAJO.rglob("*")):
            if p.is_file() and p.resolve() not in listados and p.resolve() not in propios:
                if not aplicar or p.exists():
                    apartar(p, "no está en el manifiesto")
    for p in sorted(propios):
        apartar(p, "registro de la mudanza")
    if aplicar and TRABAJO.exists():   # directorios que quedaron vacíos
        for p in sorted((q for q in TRABAJO.rglob("*") if q.is_dir()), key=lambda q: len(q.parts), reverse=True):
            if not any(p.iterdir()):
                p.rmdir()
        if not any(TRABAJO.iterdir()):
            TRABAJO.rmdir()

    for h in hechos:
        print(h)
    print(f"\nrenglones del manifiesto por acción: {conteo}")
    print(f"operaciones: {len(hechos)} · problemas: {len(problemas)}")
    for p in problemas:
        print("  PROBLEMA:", p)
    print("(aplicado)" if aplicar else "(dry run: no se movió nada; con --aplicar se hace)")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main("--aplicar" in sys.argv))
