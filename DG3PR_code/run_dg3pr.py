#!/usr/bin/env python3
"""
Interactive end-to-end runner for DG3PR (Dynamic Global Graph Guided Price-aware Recommendation).

Steps:
  1. Environment   - pick / create the conda env and install missing deps (torch 1.11 + PyG 2.3)
  2. GPU check     - DG3PR hard-codes 'cuda' (trainer, model, KGE), so a GPU is required
  3. Code check    - compile every .py and import the main modules
  4. Raw data      - use raw_data/*.csv, generate a synthetic dataset, or point to your own CSVs
  5. Preprocess    - run raw_data_process.py and check daily coverage per region
  6. Configure     - experiment grid (regions, LSTM, grades, KGE dim, train days, eval dates) + trainer params
  7. Smoke test    - 1 config, 1 eval date, 1 epoch
  8. Full run      - train_dg3pr.py with the chosen grid, logged to runs/<timestamp>/
  9. Summary       - average metrics over eval dates per config -> runs/<id>/summary.csv

Usage:
  python run_dg3pr.py                 # interactive, step by step
  python run_dg3pr.py --step 6        # jump to a given step (previous config is reused)
  python run_dg3pr.py --yes           # accept all defaults (non-interactive)

train_dg3pr.py is not modified: the runner executes it with the grid lists replaced (AST),
'output/' redirected to runs/<id>/output/ and DG3PRTrainer kwargs overridden.
Only uses the standard library; the training itself runs inside the chosen conda env.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(ROOT, "runs", ".last_config.json")
PROCESSED = os.path.join(ROOT, "data", "processed_shipping_data.dill")
DEFAULT_ENV = "dg3pr-env"
REQUIRED_MODULES = ["torch", "torch_geometric", "numpy", "pandas", "sklearn", "dill", "jenkspy", "gevent", "tqdm",
                    "tensorboard"]
PIP_NAMES = {"sklearn": "scikit-learn", "torch_geometric": "torch-geometric==2.3.0", "numpy": "numpy<1.25",
             "pandas": "pandas<2.1", "jenkspy": "jenkspy>=0.3,<0.4"}
TORCH_PIP = ["torch==1.11.0+cu113", "--extra-index-url", "https://download.pytorch.org/whl/cu113"]
RAW_FILES = {
    "shipping": "ADS_AI_MRT_SALES_SHIPPING.csv",
    "breed_class": "DIM_BREEDS_FILECLASS.csv",
    "cust_type": "DIM_CUST.csv",
}
REGIONS = ["middle", "east", "west"]  # mismo orden que dataset_list en train_dg3pr.py
METRICS = ["precision", "recall", "f1", "hit_rate", "ndcg"]
KS = [5, 10, 15, 20, 25, 30]

# Grilla y parámetros tal cual están en train_dg3pr.py / DG3PRTrainer
PAPER_CFG = dict(
    regions=REGIONS,
    use_lstm=[False, True],
    grade_num=[1, 3, 5],
    kge_dim=[32, 64, 128, 256, 512],
    train_days=[3, 7, 11, 15],
    eval_dates=["2021-01-%02d" % d for d in range(8, 18)],
    epochs=50, batch_size=1024, lr=0.01, kge_model="DistMult", kge_epochs=[200, 100],
)
QUICK_CFG = dict(PAPER_CFG, regions=["east"], use_lstm=[False], grade_num=[3], kge_dim=[64], train_days=[7],
                 eval_dates=["2021-01-08", "2021-01-09"], epochs=10)

ASSUME_YES = False

# En WSL, si está instalado un driver Linux nativo (p.ej. libnvidia-compute-580), torch 1.11 enlaza ese
# /usr/lib/x86_64-linux-gnu/libcuda.so.1 en vez del de WSL y reporta "No CUDA GPUs are available".
WSL_LIB = "/usr/lib/wsl/lib"
if os.path.isdir(WSL_LIB) and WSL_LIB not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in (WSL_LIB, os.environ.get("LD_LIBRARY_PATH")) if p)


# --------------------------------------------------------------------------- helpers
def c(text, code):
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def header(n, title):
    print("\n" + c("=" * 70, "36"))
    print(c(f" Paso {n}: {title}", "1;36"))
    print(c("=" * 70, "36"))


def ok(msg):
    print(c("  ✔ ", "32") + msg)


def warn(msg):
    print(c("  ! ", "33") + msg)


def fail(msg):
    print(c("  ✘ ", "31") + msg)


def ask(prompt, default=None, cast=str):
    if ASSUME_YES:
        return default
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {prompt}{suffix}: ").strip()
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            warn(f"Valor inválido, se espera {getattr(cast, '__name__', 'otro formato')}")


def confirm(prompt, default=True):
    if ASSUME_YES:
        return default
    d = "S/n" if default else "s/N"
    raw = input(f"  {prompt} [{d}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("s", "si", "sí", "y", "yes")


def menu(prompt, options, default=0):
    """options: list of (key, label). Returns the chosen key."""
    for i, (_, label) in enumerate(options, 1):
        mark = c(" (default)", "2") if i - 1 == default else ""
        print(f"    {i}) {label}{mark}")
    choice = ask(prompt, default + 1, int)
    if not 1 <= choice <= len(options):
        warn("Opción fuera de rango, uso la default")
        choice = default + 1
    return options[choice - 1][0]


def list_of(cast):
    """Parser for comma-separated lists, e.g. '3,7, 11'."""
    def parse(raw):
        return [cast(x.strip()) for x in raw.split(",") if x.strip()]
    parse.__name__ = f"lista de {cast.__name__}"
    return parse


def to_bool(raw):
    if raw.lower() in ("1", "true", "t", "si", "sí", "s", "y", "yes"):
        return True
    if raw.lower() in ("0", "false", "f", "no", "n"):
        return False
    raise ValueError(raw)


to_bool.__name__ = "bool"


def fmt_list(values):
    return ",".join(str(v) for v in values)


def run(cmd, **kw):
    print(c("  $ " + " ".join(cmd), "2"))
    return subprocess.run(cmd, cwd=ROOT, **kw)


def conda_root():
    exe = shutil.which("conda")
    if exe:
        return os.path.dirname(os.path.dirname(os.path.realpath(exe)))
    for cand in ("~/miniconda3", "~/anaconda3", "~/mambaforge", "~/miniforge3"):
        p = os.path.expanduser(cand)
        if os.path.isdir(p):
            return p
    return None


def conda_exe():
    root = conda_root()
    return os.path.join(root, "bin", "conda") if root else None


def env_python(env):
    root = conda_root()
    if env == "base":
        return os.path.join(root, "bin", "python")
    return os.path.join(root, "envs", env, "bin", "python")


def py(state, code, **kw):
    """Run a python snippet inside the chosen env, from the repo root."""
    return subprocess.run([state["python"], "-c", code], cwd=ROOT, capture_output=True, text=True, **kw)


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def n_trainings(cfg):
    return (len(cfg["regions"]) * len(cfg["use_lstm"]) * len(cfg["grade_num"]) * len(cfg["kge_dim"])
            * len(cfg["train_days"]) * len(cfg["eval_dates"]))


def required_days(cfg):
    """Every day that the grid touches: [eval - max(train_days), eval] for each eval date."""
    days = set()
    for d in cfg["eval_dates"]:
        e = datetime.strptime(d, "%Y-%m-%d")
        for i in range(max(cfg["train_days"]) + 1):
            days.add((e - timedelta(days=i)).strftime("%Y-%m-%d"))
    return sorted(days)


# --------------------------------------------------------------------------- steps
def step_env(state):
    header(1, "Entorno conda")
    root = conda_root()
    if not root:
        fail("No encuentro conda. Instala miniconda.")
        sys.exit(1)
    envs_dir = os.path.join(root, "envs")
    envs = sorted(e for e in os.listdir(envs_dir) if os.path.exists(os.path.join(envs_dir, e, "bin", "python"))) \
        if os.path.isdir(envs_dir) else []

    options = [(e, e) for e in envs]
    if DEFAULT_ENV not in envs:
        options.insert(0, ("__new__", f"Crear '{DEFAULT_ENV}' desde cero (python 3.8 + torch 1.11.0 cu113)"))
    default_key = state.get("env") if state.get("env") in envs else (DEFAULT_ENV if DEFAULT_ENV in envs else "__new__")
    default = next(i for i, (k, _) in enumerate(options) if k == default_key)
    env = menu("¿Qué entorno uso?", options, default)

    if env == "__new__":
        env = DEFAULT_ENV
        r = run([conda_exe(), "create", "-y", "-n", env, "python=3.8", "pip"])
        if r.returncode == 0:
            r = run([env_python(env), "-m", "pip", "install", *TORCH_PIP])
        if r.returncode != 0:
            fail("Falló la creación del entorno")
            sys.exit(1)
        ok(f"Entorno '{env}' creado")

    state["env"] = env
    state["python"] = env_python(env)

    missing = []
    for mod in REQUIRED_MODULES:
        r = py(state, f"import {mod}")
        (ok if r.returncode == 0 else warn)(f"{mod}: {'ok' if r.returncode == 0 else 'falta'}")
        if r.returncode != 0:
            missing.append(mod)
    if missing:
        pkgs = [PIP_NAMES.get(m, m) for m in missing]
        if "torch" in missing:
            pkgs.remove("torch")
            if confirm(f"Falta torch. ¿Instalar {' '.join(TORCH_PIP)}?"):
                if run([state["python"], "-m", "pip", "install", *TORCH_PIP]).returncode != 0:
                    fail("Instalación de torch falló")
                    sys.exit(1)
        if pkgs and confirm(f"¿Instalar con pip: {' '.join(pkgs)}?"):
            if run([state["python"], "-m", "pip", "install", *pkgs]).returncode != 0:
                fail("pip install falló")
                sys.exit(1)
            ok("Dependencias instaladas")
        elif pkgs:
            warn("Continúo sin instalar; es probable que falle más adelante.")
    else:
        ok("Todas las dependencias presentes")

    r = py(state, "import torch, torch_geometric, jenkspy; print(torch.__version__, torch_geometric.__version__,"
                  " jenkspy.__version__)")
    if r.returncode == 0:
        tv, pv, jv = r.stdout.split()
        ok(f"torch {tv} | torch_geometric {pv} | jenkspy {jv}")
        if not tv.startswith("1.11"):
            warn("El README pide torch==1.11.0; otras versiones pueden funcionar pero no están probadas.")
        if tuple(int(x) for x in pv.split(".")[:2]) < (2, 3):
            warn("torch_geometric < 2.3 no trae torch_geometric.nn.kge (DistMult/TransE).")
        if not jv.startswith("0.3"):
            warn("discrete_value_to_grade usa jenks_breaks(values, n_classes=...): requiere jenkspy 0.3.x.")


def step_gpu(state):
    header(2, "GPU / CUDA")
    r = py(state, "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.device_count())")
    if r.returncode != 0:
        fail(r.stderr.strip().splitlines()[-1])
        sys.exit(1)
    ver, cuda, count = r.stdout.split()
    ok(f"torch {ver} (CUDA {cuda})")
    count = int(count)
    if count == 0:
        fail("CUDA no disponible. DG3PR usa 'cuda' fijo (trainer, modelo y KGE); fallará en CPU.")
        if not confirm("¿Continuar de todas formas?", default=False):
            sys.exit(1)
        return
    names = py(state, "import torch\nfor i in range(torch.cuda.device_count()): print(i, torch.cuda.get_device_name(i))")
    gpus = [l.split(" ", 1) for l in names.stdout.strip().splitlines()]
    for i, name in gpus:
        ok(f"GPU {i}: {name}")
    state["gpu"] = menu("¿Qué GPU uso?", [(i, f"GPU {i}: {n}") for i, n in gpus],
                        int(state.get("gpu", 0)) if int(state.get("gpu", 0)) < count else 0) if count > 1 else "0"


def step_code(state):
    header(3, "Chequeo de código")
    bad = []
    for dirpath, dirnames, files in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", ".claude", "runs", "__pycache__", "log", "output")]
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                r = subprocess.run([state["python"], "-m", "py_compile", path], capture_output=True, text=True)
                if r.returncode != 0:
                    bad.append((os.path.relpath(path, ROOT), r.stderr.strip()))
    if bad:
        for path, err in bad:
            fail(path)
            print("     " + err.replace("\n", "\n     "))
        if not confirm("Hay errores de sintaxis. ¿Continuar?", default=False):
            sys.exit(1)
    else:
        ok("Todos los .py compilan")
    r = py(state, "import dg3pr_trainer, models.dg3pr, knowledge_graph.knowledge_graph_data_generator, utils.eval\n"
                  "from torch_geometric.nn.kge import DistMult")
    if r.returncode != 0:
        fail("Error importando módulos:\n     " + r.stderr.strip().splitlines()[-1])
        if not confirm("¿Continuar?", default=False):
            sys.exit(1)
    else:
        ok("Imports de dg3pr_trainer / models / knowledge_graph / utils OK")


def step_raw_data(state):
    header(4, "Datos crudos")
    raw_dir = os.path.join(ROOT, "raw_data")
    have_default = all(os.path.exists(os.path.join(raw_dir, f)) for f in RAW_FILES.values())
    options = []
    if have_default:
        options.append(("existing", "Usar raw_data/*.csv existentes"))
    options.append(("synthetic", "Generar dataset sintético en raw_data/ (make_synthetic_data.py)"
                    + (" — sobrescribe los CSV actuales" if have_default else "")))
    options.append(("custom", "Indicar rutas a mis propios CSV"))
    default_key = state.get("raw_source", "existing" if have_default else "synthetic")
    default = next((i for i, (k, _) in enumerate(options) if k == default_key), 0)
    source = menu("¿Qué datos uso?", options, default)
    state["raw_source"] = source

    paths = {k: os.path.join("raw_data", f) for k, f in RAW_FILES.items()}
    if source == "synthetic":
        customers = ask("Clientes por región", 60, int)
        start = ask("Primer día con ventas", "2020-12-24")
        end = ask("Último día con ventas", "2021-01-17")
        seed = ask("Semilla", 0, int)
        r = run([state["python"], "make_synthetic_data.py", "--out_dir", "raw_data", "--customers", str(customers),
                 "--start", start, "--end", end, "--seed", str(seed)])
        if r.returncode != 0:
            fail("Falló la generación del dataset sintético")
            sys.exit(1)
        state["force_preprocess"] = True
    elif source == "custom":
        prev = state.get("raw_paths", paths)
        for k, f in RAW_FILES.items():
            paths[k] = ask(f"Ruta a {f}", prev.get(k, paths[k]))
        state["force_preprocess"] = True
    state["raw_paths"] = paths

    code = f"""
import json, pandas as pd
paths = {paths!r}
need = {{
    'shipping': ['shipping_dt', 'org_inv_dk', 'forsale_breeds_nm', 'cust_dk', 'province_nm', 'forsale_breeds_dk',
                 'fg_grade_nm', 'sale_qty', 'sale_price', 'avg_wt', 'guide_price'],
    'breed_class': ['org_inv_dk', 'forsale_breeds_dk', 'breeds_class_nm'],
    'cust_type': ['cust_dk', 'inner_company_ind'],
}}
out = {{}}
for k, p in paths.items():
    try:
        cols = pd.read_csv(p, nrows=0).columns.tolist()
    except Exception as e:
        out[k] = dict(error=str(e)); continue
    out[k] = dict(missing=[c for c in need[k] if c not in cols])
    if k == 'shipping' and not out[k]['missing']:
        df = pd.read_csv(p, usecols=['shipping_dt', 'province_nm'])
        out[k].update(rows=len(df), min_dt=str(df['shipping_dt'].min()), max_dt=str(df['shipping_dt'].max()),
                      provinces=sorted(df['province_nm'].dropna().unique().tolist()))
print(json.dumps(out, ensure_ascii=False))
"""
    r = py(state, code)
    if r.returncode != 0:
        fail(r.stderr.strip().splitlines()[-1])
        sys.exit(1)
    info = json.loads(r.stdout)
    broken = False
    for k, f in RAW_FILES.items():
        i = info[k]
        if "error" in i:
            fail(f"{paths[k]}: {i['error']}")
            broken = True
        elif i["missing"]:
            fail(f"{paths[k]}: faltan columnas {i['missing']}")
            broken = True
        else:
            ok(f"{paths[k]}: columnas OK")
    if broken:
        fail("Los datos no tienen el esquema que espera raw_data_process.py")
        sys.exit(1)
    s = info["shipping"]
    ok(f"ventas: {s['rows']} filas | shipping_dt: {s['min_dt']} .. {s['max_dt']}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s["min_dt"]):
        warn("shipping_dt debería tener formato YYYY-MM-DD: el código filtra por fecha comparando strings.")
    ok(f"provincias: {', '.join(s['provinces'][:12])}{' ...' if len(s['provinces']) > 12 else ''}")


def step_preprocess(state):
    header(5, "Preprocesamiento (raw_data_process.py)")
    paths = state["raw_paths"]
    fresh = os.path.exists(PROCESSED) and not state.get("force_preprocess") and all(
        os.path.getmtime(PROCESSED) >= os.path.getmtime(os.path.join(ROOT, p)) for p in paths.values())
    if fresh and not confirm("data/processed_shipping_data.dill está al día. ¿Regenerarlo igual?", default=False):
        ok("Uso el .dill existente")
    else:
        r = run([state["python"], "raw_data_process.py",
                 "--shipping_data_path", paths["shipping"],
                 "--breed_class_data_path", paths["breed_class"],
                 "--cust_type_data_path", paths["cust_type"]],
                capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "Before" in line:
                print(c("     " + line.split(": ", 1)[-1], "2"))
        if r.returncode != 0:
            fail("raw_data_process.py falló:\n" + r.stderr)
            sys.exit(1)
        ok("data/processed_shipping_data.dill generado")
        state["force_preprocess"] = False

    # Cobertura diaria por región (las listas de provincias se leen de train_dg3pr.py)
    code = r"""
import ast, json, pandas as pd
from utils.file_util import load_csv_file
tree = ast.parse(open('train_dg3pr.py', encoding='utf-8').read())
prov = {n.targets[0].id.replace('_province_list', ''): ast.literal_eval(n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.endswith('_province_list')}
df = load_csv_file('data/processed_shipping_data.dill')
df['shipping_dt'] = df['shipping_dt'].astype(str)
out = {}
for region, provinces in prov.items():
    d = df[df['province_nm'].isin(provinces)]
    daily_items = d.groupby('shipping_dt')['forsale_breeds_dk'].nunique()
    out[region] = dict(rows=len(d), customers=int(d['cust_dk'].nunique()), items=int(d['forsale_breeds_dk'].nunique()),
                       days=sorted(d['shipping_dt'].unique().tolist()),
                       min_daily_items=int(daily_items.min()) if len(d) else 0)
print(json.dumps(out))
"""
    r = py(state, code)
    if r.returncode != 0:
        fail(r.stderr.strip().splitlines()[-1])
        sys.exit(1)
    cov = json.loads(r.stdout)
    state["coverage"] = {k: v["days"] for k, v in cov.items()}
    for region, v in cov.items():
        if v["rows"] == 0:
            warn(f"{region:<6}: sin datos (ninguna provincia coincide con {region}_province_list de train_dg3pr.py)")
        else:
            ok(f"{region:<6}: {v['rows']} filas | {v['customers']} clientes | {v['items']} productos | "
               f"días {v['days'][0]} .. {v['days'][-1]} ({len(v['days'])})")
            if v["min_daily_items"] < max(KS):
                warn(f"{region:<6}: algún día se venden solo {v['min_daily_items']} productos distintos. "
                     f"utils/eval.py evalúa hasta top-{max(KS)} y lanza IndexError si un cliente tiene "
                     f"menos candidatos ese día de evaluación.")


def check_coverage(state, cfg):
    """Warn about days without data: each daily graph needs purchases, the eval day too."""
    cov = state.get("coverage", {})
    days = required_days(cfg)
    good = True
    for region in cfg["regions"]:
        have = set(cov.get(region, []))
        missing = [d for d in days if d not in have]
        if missing:
            good = False
            warn(f"{region}: faltan {len(missing)} días necesarios para la grilla "
                 f"({', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}). "
                 "Un día sin ventas rompe la generación del grafo dinámico.")
    if good:
        ok(f"Todas las regiones tienen datos en los {len(days)} días que usa la grilla ({days[0]} .. {days[-1]})")
    return good


def step_configure(state):
    header(6, "Configuración del experimento")
    available = [r for r in REGIONS if state.get("coverage", {}).get(r)]
    preset = menu("¿Qué experimento?", [
        ("quick", "Rápido: east, sin LSTM, grade 3, dim 64, 7 días, 2 fechas de eval, 10 epochs  (2 entrenamientos)"),
        ("paper", "Grilla completa de train_dg3pr.py  (3600 entrenamientos x 50 epochs: días de cómputo)"),
        ("custom", "Personalizado (parte de la última configuración usada)"),
    ], default=0)
    if preset == "quick":
        cfg = dict(QUICK_CFG)
        if available and "east" not in available:
            cfg["regions"] = available[:1]
    elif preset == "paper":
        cfg = dict(PAPER_CFG)
    else:
        cfg = dict(state.get("cfg", QUICK_CFG))
        print(c("  Listas separadas por comas. Enter = mantener.", "2"))
        cfg["regions"] = ask(f"Regiones ({'/'.join(REGIONS)})", fmt_list(cfg["regions"]), list_of(str))
        cfg["use_lstm"] = ask("use_lstm (True/False)", fmt_list(cfg["use_lstm"]), list_of(to_bool))
        cfg["grade_num"] = ask("grade_num (niveles de precio)", fmt_list(cfg["grade_num"]), list_of(int))
        cfg["kge_dim"] = ask("kge_embedding_dim", fmt_list(cfg["kge_dim"]), list_of(int))
        cfg["train_days"] = ask("train_date_days", fmt_list(cfg["train_days"]), list_of(int))
        cfg["eval_dates"] = ask("Fechas de evaluación (YYYY-MM-DD)", fmt_list(cfg["eval_dates"]), list_of(str))
        cfg["epochs"] = ask("epochs", cfg["epochs"], int)
        cfg["batch_size"] = ask("batch_size", cfg["batch_size"], int)
        cfg["lr"] = ask("learning_rate", cfg["lr"], float)
        cfg["kge_model"] = menu("Modelo KGE", [(m, m) for m in ("DistMult", "TransE", "ComplEx")],
                                ["DistMult", "TransE", "ComplEx"].index(cfg["kge_model"]))
        cfg["kge_epochs"] = ask("KGE epochs (global,dinámico)", fmt_list(cfg["kge_epochs"]), list_of(int))
        cfg["regions"] = [r for r in REGIONS if r in cfg["regions"]] or REGIONS[:1]
        for value in ("use_lstm", "grade_num", "kge_dim", "train_days", "eval_dates"):
            cfg[value] = cfg[value] or QUICK_CFG[value]
        if len(cfg["kge_epochs"]) != 2:
            warn("KGE epochs necesita 2 valores; uso 200,100")
            cfg["kge_epochs"] = [200, 100]

    state["cfg"] = cfg
    print()
    ok("regiones={regions} use_lstm={use_lstm} grade_num={grade_num} kge_dim={kge_dim} "
       "train_days={train_days}".format(**cfg))
    ok("eval_dates={} | epochs={epochs} batch={batch_size} lr={lr} kge={kge_model} {kge_epochs}".format(
        fmt_list(cfg["eval_dates"]), **cfg))
    ok(f"Total: {n_trainings(cfg)} entrenamientos")
    check_coverage(state, cfg)


def wrapper_cmd(state, cfg, output_dir):
    """Run train_dg3pr.py unmodified, replacing the grid lists and trainer kwargs at runtime."""
    lists = {
        "use_lstm_list": repr(cfg["use_lstm"]),
        "grade_num_list": repr(cfg["grade_num"]),
        "kge_embedding_dim_list": repr(cfg["kge_dim"]),
        "train_date_days_list": repr(cfg["train_days"]),
        "eval_date_list": repr(cfg["eval_dates"]),
        "dataset_list": "{" + ", ".join(f"'{r}': {r}_data" for r in cfg["regions"]) + "}",
    }
    trainer = dict(epochs=cfg["epochs"], batch_size=cfg["batch_size"], learning_rate=cfg["lr"],
                   kge_model_name=cfg["kge_model"], kge_epochs=tuple(cfg["kge_epochs"]))
    code = r"""
import ast, sys
lists, trainer, output_dir = %r, %r, %r
import dg3pr_trainer
_init = dg3pr_trainer.DG3PRTrainer.__init__
def _patched_init(self, *a, **k):
    k.update(trainer)
    _init(self, *a, **k)
dg3pr_trainer.DG3PRTrainer.__init__ = _patched_init

class Patch(ast.NodeTransformer):
    def visit_Assign(self, node):
        name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
        if name in lists:
            node.value = ast.parse(lists[name], mode='eval').body
            return node
        return self.generic_visit(node)
    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith('output/'):
            node.value = output_dir + '/' + node.value[len('output/'):]
        return node

tree = ast.parse(open('train_dg3pr.py', encoding='utf-8').read())
tree = ast.fix_missing_locations(Patch().visit(tree))
sys.argv = ['train_dg3pr.py']
exec(compile(tree, 'train_dg3pr.py', 'exec'), {'__name__': '__main__'})
""" % (lists, trainer, output_dir)
    return [state["python"], "-u", "-c", code]


def parse_metrics(text):
    return {k: float(v) for k, v in re.findall(r"'(\w+)': ([0-9.eE+-]+|nan)", text)}


def stream(cmd, log_path, state, total):
    """Run cmd, tee stdout/stderr to log_path, show compact progress. Returns (returncode, lines)."""
    lines, done = [], 0
    t0 = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(state.get("gpu", "0")), PYTHONIOENCODING="utf-8")
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env=env, encoding="utf-8", errors="replace")
        try:
            for line in p.stdout:
                log.write(line)
                log.flush()
                lines.append(line)
                el = f"[{time.time() - t0:7.0f}s]"
                m = re.search(r"> Training Date: (\S+) - (\S+) <", line)
                if m:
                    done += 1
                    print(c(f"  {el} ({done}/{total}) entrenando {m.group(1)} .. {m.group(2)}", "1"))
                    continue
                m = re.search(r"Epoch: (\d+), Loss: ([0-9.eE+-]+)", line)
                if m:
                    # se completa con las métricas Top-10 de la misma epoch
                    print(f"  {el}   epoch {int(m.group(1)):>3}  loss={float(m.group(2)):.4f}", end="", flush=True)
                    continue
                m = re.search(r"Epoch: \d+, Top-10 Metrics: (\{.*\})", line)
                if m:
                    met = parse_metrics(m.group(1))
                    print(f"  ndcg@10={met.get('ndcg', 0):.4f}  recall@10={met.get('recall', 0):.4f}")
                    continue
                m = re.search(r"Best Epoch: (\d+) Top-10 Metrics: (\{.*\})", line)
                if m:
                    met = parse_metrics(m.group(2))
                    print(c(f"  {el}   mejor epoch {m.group(1)}: ndcg@10={met.get('ndcg', 0):.4f} "
                            f"recall@10={met.get('recall', 0):.4f} hit@10={met.get('hit_rate', 0):.4f}", "32"))
                    continue
                if "> Training Done:" in line:
                    print(c(f"  {el} ✔ " + line.split("Training Done: ")[1].split(" <")[0], "32"))
                elif "Traceback" in line or re.search(r"\b\w*Error\b", line):
                    print("  " + line.rstrip())
            p.wait()
        except KeyboardInterrupt:
            p.terminate()
            p.wait()
            warn("Interrumpido por el usuario")
    if p.returncode not in (0, None) and any("Traceback" in l for l in lines):
        tb = "".join(lines[next(i for i, l in enumerate(lines) if "Traceback" in l):])
        print(c(tb[-3000:], "31"))
    return p.returncode, lines


def step_smoke(state):
    header(7, "Smoke test (1 configuración, 1 fecha, 1 epoch)")
    if not confirm("¿Correr la prueba antes del run completo?"):
        warn("Omitido")
        return
    cfg = dict(state["cfg"])
    for k in ("regions", "use_lstm", "grade_num", "kge_dim", "train_days", "eval_dates"):
        cfg[k] = cfg[k][:1]
    cfg.update(epochs=1, kge_epochs=[5, 5])
    run_dir = os.path.join(ROOT, "runs", "smoke")
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir)
    rc, lines = stream(wrapper_cmd(state, cfg, os.path.join(run_dir, "output")),
                       os.path.join(run_dir, "train.log"), state, 1)
    if rc == 0 and any("Training Done:" in l for l in lines):
        ok("Pipeline funciona end-to-end (log en runs/smoke/train.log)")
    else:
        fail(f"Smoke test falló (exit {rc}). Log: runs/smoke/train.log")
        if not confirm("¿Continuar igualmente?", default=False):
            sys.exit(1)


def step_full(state):
    header(8, "Experimento completo")
    cfg = state["cfg"]
    total = n_trainings(cfg)
    print(f"  {total} entrenamientos x {cfg['epochs']} epochs | regiones: {', '.join(cfg['regions'])}")
    if not check_coverage(state, cfg) and not confirm("Faltan días con datos. ¿Lanzar igual?", default=False):
        return
    if not confirm("¿Lanzar?"):
        warn("Omitido")
        return
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ROOT, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"env": state["env"], "raw_paths": state["raw_paths"], "raw_source": state["raw_source"],
                   "cfg": cfg}, f, indent=2)
    print(c(f"  Log: runs/{run_id}/train.log | resultados: runs/{run_id}/output/  (Ctrl+C para detener)", "2"))
    rc, _ = stream(wrapper_cmd(state, cfg, os.path.join(run_dir, "output")), os.path.join(run_dir, "train.log"),
                   state, total)
    state["last_run"] = run_id
    (ok if rc == 0 else warn)(f"Terminó con código {rc}")


def step_summary(state):
    header(9, "Resumen de métricas")
    runs_dir = os.path.join(ROOT, "runs")
    runs = sorted(d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d, "output"))) \
        if os.path.isdir(runs_dir) else []
    if not runs:
        warn("No hay runs todavía")
        return
    default = runs.index(state["last_run"]) if state.get("last_run") in runs else len(runs) - 1
    run_id = menu("¿Qué run resumir?", [(r, r) for r in runs], default)
    out_dir = os.path.join(runs_dir, run_id, "output")

    rows = []
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name, "metrics_result_tmp.csv")
        if not os.path.exists(path):
            warn(f"{name}: sin metrics_result_tmp.csv (configuración incompleta)")
            continue
        with open(path) as f:
            per_date = list(csv.DictReader(f))
        row = {"config": name.replace("kan_train_output_", ""), "eval_dates": len(per_date)}
        for col in per_date[0]:
            if "@" in col:
                vals = [float(r[col]) for r in per_date if r[col] not in ("", "nan")]
                row[col] = sum(vals) / len(vals) if vals else float("nan")
        rows.append(row)
    if not rows:
        warn("Ninguna configuración terminó")
        return

    cols = ["config", "eval_dates"] + [f"{m}@{k}" for m in METRICS for k in KS]
    out = os.path.join(runs_dir, run_id, "summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ok(f"{len(rows)} configuraciones (promedio sobre fechas de evaluación) → runs/{run_id}/summary.csv")

    key = ask("Métrica para ordenar", "ndcg@10")
    if key not in rows[0]:
        warn(f"'{key}' no existe, uso ndcg@10")
        key = "ndcg@10"
    rows.sort(key=lambda r: r.get(key, float("-inf")), reverse=True)
    show = ["precision@10", "recall@10", "hit_rate@10", "ndcg@10", "ndcg@20"]
    width = max(len(r["config"]) for r in rows)
    print("\n  " + "config".ljust(width) + "".join(s.rjust(13) for s in show))
    for r in rows[:20]:
        print("  " + r["config"].ljust(width) + "".join(f"{r.get(s, float('nan')):13.4f}" for s in show))
    if len(rows) > 20:
        print(c(f"  ... {len(rows) - 20} más en summary.csv", "2"))


STEPS = [step_env, step_gpu, step_code, step_raw_data, step_preprocess, step_configure, step_smoke, step_full,
         step_summary]


def main():
    global ASSUME_YES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=1, help=f"paso inicial (1-{len(STEPS)})")
    ap.add_argument("--only", action="store_true", help="ejecutar solo el paso indicado en --step")
    ap.add_argument("--yes", action="store_true", help="aceptar todos los defaults sin preguntar")
    a = ap.parse_args()
    ASSUME_YES = a.yes

    state = load_state()
    requirements = {2: "python", 5: "raw_paths", 7: "cfg"}
    for step, key in requirements.items():
        if a.step >= step and key not in state:
            fail(f"No hay configuración previa ('{key}'); empieza desde un paso anterior a {step}.")
            sys.exit(1)

    print(c("\nDG3PR — runner end-to-end interactivo", "1"))
    steps = STEPS[a.step - 1:a.step] if a.only else STEPS[a.step - 1:]
    for i, fn in enumerate(steps, a.step):
        if not ASSUME_YES and i > a.step and not confirm(c(f"→ Continuar al paso {i} ({fn.__name__[5:]})?", "1")):
            print("  Puedes retomar con:  python run_dg3pr.py --step", i)
            break
        fn(state)
        save_state(state)
    print()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelado.")
