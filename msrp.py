import pandas as pd
import numpy as np
import yfinance as yf
import pyomo.environ as pyo

# ============================================================
# Config
# ============================================================
START = "2023-12-30"
END   = "2025-12-30"

RF_ANNUAL = 0.02          # tasa libre de riesgo anual (configurable)
F_EQ = 252                # anualización acciones/ETFs (días hábiles)
MIN_DATA_FRAC = 0.80      # mínimo % de días con datos para conservar el ticker
VAR_EPS = 1e-12           # evita sqrt(0) en Sharpe
DIAG_EPS = 1e-6           # diagonal loading para estabilidad numérica

# Shrinkage (si no está sklearn)
COV_SHRINK_ALPHA = 0.10   # 0 = cov muestral; 0.1~0.3 suele estabilizar

IPOPT_PATH = r"C:\ProgramData\miniconda3\Library\bin\ipopt.exe" 

# Lista sugerida (arreglos típicos):
# - ^GSPC = S&P500
# - MOLI.BA = Molinos (BYMA) en Yahoo
# - Quité MASP porque en Yahoo suele no existir con ese símbolo
activos = [
    "YPF", "NVDA", "PAM", "GGAL", "BBVA", "GLOB",
     "MELI", "MOLI.BA", "GLD", "SLV", "TSLA", "^GSPC",
    "MIGA.MU", "META", "GOOGL"
]
# ============================================================
# 1) Descargar y limpiar datos
# ============================================================
precios_raw = yf.download(tickers=activos, start=START, end=END, interval="1d")["Close"]
# Quitar tickers totalmente vacíos
precios = precios_raw.dropna(axis=1, how="all")
# Exigir mínimo % de datos
precios = precios.loc[:, precios.notna().mean() >= MIN_DATA_FRAC]
activos_ok = list(precios.columns)
faltantes = sorted(set(activos) - set(activos_ok))
if faltantes:
    print("⚠️ Se excluyen tickers sin datos suficientes:", faltantes)
if len(activos_ok) < 2:
    raise ValueError("No hay suficientes activos con datos para optimizar.")
# Rendimientos diarios (sin relleno implícito)
rendimientos = precios.pct_change(fill_method=None).dropna(how="any")

if len(rendimientos) < 5:
    raise ValueError("Muy pocos datos de rendimientos (revisá fechas/tickers).")

# ============================================================
# 2) Anualización mixta (crypto vs acciones) + Σ robusta
# ============================================================
# Identificar crypto (método simple por sufijo -USD)
crypto = set()                 # 1. Creas el conjunto vacío
for a in activos_ok:           # 2. El bucle
    if a.endswith("-USD"):     # 3. La condición
        crypto.add(a)
# Factor anual por activo
# 1. Creamos un diccionario vacío para guardar los datos
f = pd.Series(F_EQ, index=activos_ok, dtype=float)

# μ anualizado por activo
mu_daily = rendimientos.mean()
mu_annual = mu_daily * f

# Σ diaria
Sigma_daily = rendimientos.cov()

# --- Estimación robusta de Σ: preferir Ledoit-Wolf si está disponible
Sigma_daily_rob = None
try:
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(rendimientos.values)
    Sigma_daily_rob = pd.DataFrame(lw.covariance_, index=activos_ok, columns=activos_ok)
    metodo_sigma = "Ledoit-Wolf (shrinkage)"
except Exception:
    # Fallback: shrinkage manual hacia identidad escalada (target)
    S = Sigma_daily
    avg_var = float(np.mean(np.diag(S.values)))
    F_target = pd.DataFrame(np.eye(len(activos_ok)) * avg_var, index=activos_ok, columns=activos_ok)
    Sigma_daily_rob = (1.0 - COV_SHRINK_ALPHA) * S + COV_SHRINK_ALPHA * F_target
    metodo_sigma = f"Shrinkage manual (alpha={COV_SHRINK_ALPHA})"

# Anualizar Σ de forma consistente:
# Σ^ann = D Σ^daily D, con D = diag(sqrt(f_i))
D = np.diag(np.sqrt(f.values))
Sigma_annual = pd.DataFrame(D @ Sigma_daily_rob.values @ D, index=activos_ok, columns=activos_ok)

# Diagonal loading para mejorar condición numérica
Sigma_annual = Sigma_annual + np.eye(len(activos_ok)) * DIAG_EPS

print(f"Σ estimada con: {metodo_sigma} + diagonal_loading(eps={DIAG_EPS})")

# Validación anti-NaN/inf
if (not np.isfinite(mu_annual.values).all()) or (not np.isfinite(Sigma_annual.values).all()):
    raise ValueError("μ o Σ contienen NaN/Inf. Revisá tickers y limpieza de datos.")

# ============================================================
# 3) MSRP (Maximum Sharpe Ratio Portfolio) con Pyomo + Ipopt
# ============================================================
def optimizar_msrp_pyomo(activos, mu_annual: pd.Series, Sigma_annual: pd.DataFrame,
                         rf_annual: float = 0.02,
                         ipopt_path: str = None,
                         var_eps: float = 1e-12):

    m = pyo.ConcreteModel(name="MSRP_MaxSharpe")

    m.ACTIVOS = pyo.Set(initialize=list(activos))

    # long-only (Palomar: actúa como regularización natural)
    m.w = pyo.Var(m.ACTIVOS, domain=pyo.NonNegativeReals, bounds=(0, 1))

    # Parámetros
    mu_dict = {a: float(mu_annual.loc[a]) for a in activos}
    sig_dict = {(a1, a2): float(Sigma_annual.loc[a1, a2]) for a1 in activos for a2 in activos}

    m.mu = pyo.Param(m.ACTIVOS, initialize=mu_dict, within=pyo.Reals)
    m.sigma = pyo.Param(m.ACTIVOS, m.ACTIVOS, initialize=sig_dict, within=pyo.Reals)

    # Presupuesto
    m.budget = pyo.Constraint(expr=sum(m.w[i] for i in m.ACTIVOS) == 1.0)

    # Retorno y varianza (anualizados)
    def port_return(mm):
        return sum(mm.w[i] * mm.mu[i] for i in mm.ACTIVOS)

    def port_var(mm):
        return sum(mm.w[i] * mm.sigma[i, j] * mm.w[j] for i in mm.ACTIVOS for j in mm.ACTIVOS)

    # Objetivo Sharpe:
    # max (w^T μ - r_f) / sqrt(w^T Σ w)
    #
    # Palomar (cap. 7): es programación fraccional NO convexa.
    # Con Ipopt lo resolvemos directo (NLP).
    # Alternativamente, puede reformularse con la Transformada de Schaible para una forma convexa
    # bajo condiciones apropiadas.
    def sharpe_obj(mm):
        num = port_return(mm) - rf_annual
        den = pyo.sqrt(port_var(mm) + var_eps)
        return num / den

    m.obj = pyo.Objective(rule=sharpe_obj, sense=pyo.maximize)

    opt = pyo.SolverFactory("ipopt", executable=ipopt_path) if ipopt_path else pyo.SolverFactory("ipopt")
    res = opt.solve(m, tee=False)

    pesos = {a: float(pyo.value(m.w[a])) for a in activos}

    # Métricas finales (anualizadas)
    w = np.array([pesos[a] for a in activos], dtype=float)
    mu_vec = mu_annual.loc[activos].values.astype(float)
    Sig = Sigma_annual.loc[activos, activos].values.astype(float)

    ret = float(w @ mu_vec)
    vol = float(np.sqrt(w @ Sig @ w))
    sharpe = float((ret - rf_annual) / vol) if vol > 0 else np.nan

    return pesos, ret, vol, sharpe

pesos, retorno, volatilidad, sharpe = optimizar_msrp_pyomo(
    activos_ok, mu_annual, Sigma_annual, rf_annual=RF_ANNUAL, ipopt_path=IPOPT_PATH, var_eps=VAR_EPS
)

# ============================================================
# 4) Salida
# ============================================================
print("\n" + "-" * 40)
print("RESULTADOS OPTIMIZACIÓN (MSRP - Max Sharpe)")
print("-" * 40)

for a, w in sorted(pesos.items(), key=lambda x: x[1], reverse=True):
    print(f"{a:>8s}: {w:>10.4%}")

print("-" * 40)
print(f"Retorno Esperado Anual : {retorno:.2%}")
print(f"Volatilidad Anual      : {volatilidad:.2%}")
print(f"Sharpe Anual (rf={RF_ANNUAL:.2%}) : {sharpe:.4f}")

print("\nPor qué puede diferir del GMVP:")
print("- GMVP minimiza riesgo y depende esencialmente de Σ (más estable).")
print("- MSRP maximiza retorno excedente / riesgo y depende de μ y Σ.")
print("- Palomar remarca que μ es muy difícil de estimar: el error en μ afecta mucho más al MSRP,")
print("  por eso puede dar soluciones más concentradas o sensibles que la mínima varianza.")
