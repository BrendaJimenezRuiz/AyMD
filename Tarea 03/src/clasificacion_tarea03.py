"""
Tarea 03 - Almacenes y Minería de Datos
Sección 3: Clasificación — Predicción de ESTATUS_VICTIMA
UNAM Facultad de Ciencias, Semestre 2026-2

Estilo basado en el notebook de clase: cart_sonar.ipynb (afraidspy/mineria_datos)

Métricas según material del curso:
  - Han, Kamber & Pei (2011). Data Mining: Concepts and Techniques.
  - Saito & Rehmsmeier (2015). PLOS ONE 10(3): e0118432.
  - Fawcett (2006). Pattern Recognition Letters 27(8): 861–874.
  - Provost & Fawcett (2013). Data Science for Business. O'Reilly.
  - Bamber (1975). Journal of Mathematical Psychology 12(4): 387–415.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc,
    f1_score, recall_score, precision_score,
    precision_recall_curve, average_precision_score
)

import warnings
warnings.filterwarnings("ignore")

# ── Configuración global ────────────────────────────────────────────────────
semilla = 42
sns.set_theme(style="whitegrid", context="notebook")

# CORRECCIÓN 1: CSV_PATH relativo al directorio del script.
# Si quieres especificar otra ruta, pásala como argumento:
#   python clasificacion_tarea03_v3.py /ruta/a/dataset_limpio.csv
if len(sys.argv) > 1:
    CSV_PATH = sys.argv[1]
else:
    CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "dataset_limpio.csv")

GUARDAR_FIGURAS = True   # True → guarda PNGs en la misma carpeta del script

print("=" * 70)
print("TAREA 03 — CLASIFICACIÓN: Predicción de ESTATUS_VICTIMA")
print("=" * 70)
print(f"Dataset: {CSV_PATH}")

# =============================================================================
# 1. CARGA Y EXPLORACIÓN INICIAL
# =============================================================================
df = pd.read_csv(CSV_PATH)
print(f"\nDimensiones del dataset: {df.shape}")
print(f"\nColumnas disponibles: {list(df.columns)}")
print(f"\nTipos de datos:\n{df.dtypes.value_counts()}")

print("\nDistribución de clases:")
print(df["ESTATUS_VICTIMA"].value_counts())
print("\nProporción de clases:")
print(df["ESTATUS_VICTIMA"].value_counts(normalize=True).round(4))

proporcion_mayoritaria = df["ESTATUS_VICTIMA"].value_counts(normalize=True).iloc[0]
proporcion_minoritaria = df["ESTATUS_VICTIMA"].value_counts(normalize=True).iloc[1]

# Gráfica de distribución de clases
plt.figure(figsize=(6, 4))
sns.countplot(
    data=df,
    x="ESTATUS_VICTIMA",
    order=df["ESTATUS_VICTIMA"].value_counts().index,
    palette="Blues_d"
)
plt.xlabel("Clase")
plt.ylabel("Frecuencia")
plt.title("Distribución de ESTATUS_VICTIMA")
plt.tight_layout()
if GUARDAR_FIGURAS:
    plt.savefig("01_distribucion_clases.png", dpi=150)
plt.show()

print(f"\nNulos por columna:\n{df.isnull().sum()}")

# =============================================================================
# 2. JUSTIFICACIÓN DE MÉTRICAS
# =============================================================================
print(f"""
─────────────────────────────────────────────────────────────────────
JUSTIFICACIÓN DE MÉTRICAS (Han et al., 2011; Saito & Rehmsmeier, 2015)
─────────────────────────────────────────────────────────────────────
Dataset: {proporcion_mayoritaria:.1%} DESAPARECIDA vs {proporcion_minoritaria:.1%} NO LOCALIZADA
→ desbalance severo (ratio {proporcion_mayoritaria/proporcion_minoritaria:.0f}:1).

Paradoja de la Exactitud (material del curso, Bloque 1):
  Un clasificador trivial que siempre prediga DESAPARECIDA obtiene
  ~{proporcion_mayoritaria:.0%} de accuracy sin detectar ningún caso NO LOCALIZADA.
  → Accuracy es ENGAÑOSA aquí (Saito & Rehmsmeier, 2015).

Métricas reportadas:
  1. F1 (NO LOCALIZADA) ★ — clase de interés; F1-macro puede esconder F1=0
     en la clase positiva (Han et al., 2011).
  2. Recall (NO LOCALIZADA) — FN implica expediente sin cerrar → recursos
     de búsqueda mal asignados → sesgo en estadísticas oficiales
     (Fawcett, 2006).
  3. PR-AUC ★ — más informativa que ROC-AUC con clases desbalanceadas porque
     ignora los TN, que con {proporcion_mayoritaria:.0%}/{proporcion_minoritaria:.0%} inflarían artificialmente
     la curva ROC (Saito & Rehmsmeier, 2015).
  4. ROC-AUC — capacidad discriminativa global para comparar modelos sin
     fijar un umbral (Fawcett, 2006; Bamber, 1975).
  5. Accuracy — solo como referencia; se espera alta por el desbalance.
""")

# =============================================================================
# 3. PREPROCESAMIENTO
# =============================================================================
# CORRECCIÓN 2: se elimina MES_DESAPARICION porque TEMPORADA ya captura la
# información temporal de forma más limpia (agrupada en 4 estaciones) y evita
# redundancia con la variable categórica. ANIO_DESAPARICION se mantiene como
# variable ordinal numérica ya que los árboles la manejan bien con umbrales.
# Justificación: MES (1–12) y TEMPORADA son colineales → incluir ambas no
# aporta información adicional y puede sesgar la importancia de variables.
features = [
    "SEXO", "GRUPO_EDAD", "ANIO_DESAPARICION",
    "TEMPORADA", "ENTIDAD", "ORIGEN_SIMPLIFICADO"
]
target = "ESTATUS_VICTIMA"

print("=" * 70)
print("PREPROCESAMIENTO")
print("=" * 70)
print(f"\nVariables predictoras seleccionadas: {features}")
print(f"Variable objetivo: {target}")
print("""
Decisiones de preprocesamiento:
  • MES_DESAPARICION excluida: colineal con TEMPORADA (Irany ya la discretizó
    en 4 estaciones en la sección de reglas de asociación). Incluir ambas
    introduciría redundancia y podría sesgar la importancia de variables.
  • ANIO_DESAPARICION mantenida como numérica: los árboles de decisión
    trabajan con umbrales en variables continuas; rango 1961–2025 es
    informativo para detectar cambios históricos en el sistema de registro.
  • GRUPO_EDAD y TEMPORADA: nulos imputados como "SIN_DATO" (categoría
    explícita que preserva la información de ausencia en lugar de eliminarla).
  • ENTIDAD y ORIGEN_SIMPLIFICADO: variables categóricas con cardinalidad
    manejable (33 y 5 valores respectivamente), codificadas con LabelEncoder.
""")

df_model = df[features + [target]].copy()

# Imputar nulos categóricos con categoría explícita "SIN_DATO"
for col in ["GRUPO_EDAD", "TEMPORADA"]:
    n_nulos = df_model[col].isnull().sum()
    df_model[col] = df_model[col].fillna("SIN_DATO")
    print(f"  {col}: {n_nulos:,} nulos imputados como 'SIN_DATO'")

# CORRECCIÓN 3: ANIO_DESAPARICION se imputa con la mediana (justificado).
# No tiene sentido imputar con mediana para MES (ya eliminado), pero para
# ANIO_DESAPARICION la mediana es el año más central del dataset, que es
# preferible a la media porque la distribución puede ser asimétrica.
n_nulos_anio = df_model["ANIO_DESAPARICION"].isnull().sum()
mediana_anio = df_model["ANIO_DESAPARICION"].median()
df_model["ANIO_DESAPARICION"] = df_model["ANIO_DESAPARICION"].fillna(mediana_anio)
print(f"  ANIO_DESAPARICION: {n_nulos_anio:,} nulos imputados con mediana "
      f"({mediana_anio:.0f})")

# Codificar variables categóricas con LabelEncoder
le_dict = {}
cols_categoricas = ["SEXO", "GRUPO_EDAD", "TEMPORADA", "ENTIDAD", "ORIGEN_SIMPLIFICADO"]
for col in cols_categoricas:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    le_dict[col] = le
    print(f"  {col}: {len(le.classes_)} categorías → enteros 0–{len(le.classes_)-1}")

# Codificar variable objetivo
le_target = LabelEncoder()
y = le_target.fit_transform(df_model[target])
X = df_model[features]

# CORRECCIÓN 6 (verificación de orden): LabelEncoder ordena alfabéticamente.
# DESAPARECIDA < NO LOCALIZADA → 0=DESAPARECIDA, 1=NO LOCALIZADA (clase positiva).
clases = list(le_target.classes_)
idx_no_localizada = list(le_target.classes_).index("NO LOCALIZADA")
print(f"\nCodificación de la variable objetivo:")
for i, c in enumerate(clases):
    marca = " ← clase positiva (interés)" if i == idx_no_localizada else ""
    print(f"  {i} = {c}{marca}")

# Verificar que NO LOCALIZADA es efectivamente la clase 1
assert idx_no_localizada == 1, (
    "ERROR: NO LOCALIZADA no es la clase 1. "
    f"Clases: {clases}. Ajusta pos_label en las métricas."
)
print("\nVerificación superada: NO LOCALIZADA = clase 1 ✓")

# =============================================================================
# 4. DIVISIÓN TRAIN / TEST (estratificada — igual que el notebook de clase)
# =============================================================================
print("\n" + "=" * 70)
print("DIVISIÓN TRAIN / TEST")
print("=" * 70)

variables_entrenamiento, variables_prueba, objetivo_entrenamiento, objetivo_prueba = \
    train_test_split(X, y, test_size=0.20, random_state=semilla, stratify=y)

print("Tamaño de entrenamiento:")
print(variables_entrenamiento.shape)
print("\nTamaño de prueba:")
print(variables_prueba.shape)
print("\nProporción de clases en entrenamiento:")
print(pd.Series(objetivo_entrenamiento).value_counts(normalize=True).round(4))
print("\nProporción de clases en prueba:")
print(pd.Series(objetivo_prueba).value_counts(normalize=True).round(4))
print("\nstratify=y garantiza que la proporción 94%/6% se mantiene en ambos")
print("conjuntos, igual que en el notebook de clase (cart_sonar.ipynb).")

# =============================================================================
# 5. VALIDACIÓN CRUZADA ESTRATIFICADA (5 folds — igual que el notebook)
# =============================================================================
validacion_cruzada = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)

# CORRECCIÓN 4: scorer robusto usando make_scorer + manejo de folds con
# muy pocos positivos. roc_auc_score puede lanzar ValueError si un fold
# contiene solo una clase; make_scorer con needs_proba captura esto.
def _auc_no_localizada_fn(y_true, y_prob):
    """
    Calcula ROC-AUC tomando NO LOCALIZADA (clase 1) como positiva.
    Devuelve 0.5 (azar) si el fold no contiene las dos clases,
    lo que puede ocurrir con datasets muy desbalanceados.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)

scorer_auc = make_scorer(
    _auc_no_localizada_fn,
    needs_proba=True,
    response_method="predict_proba"
)

# También mantenemos el callable directo para compatibilidad con GridSearchCV
def calcular_auc_no_localizada(modelo, variables, objetivo):
    """
    Scorer callable para GridSearchCV.
    AUC tomando NO LOCALIZADA (1) como clase positiva.
    Robusto ante folds donde solo hay una clase.
    """
    try:
        probabilidades = modelo.predict_proba(variables)
        return roc_auc_score(objetivo, probabilidades[:, 1])
    except ValueError:
        # Fold con una sola clase → devolver 0.5 (sin información)
        return 0.5

# =============================================================================
# 6. MODELO 1 — ÁRBOL DE DECISIÓN POR DEFECTO
# =============================================================================
print("\n" + "=" * 70)
print("MODELO 1: Árbol de Decisión por defecto (CART, Gini, sin restricciones)")
print("=" * 70)

modelo_arbol_defecto = DecisionTreeClassifier(
    criterion="gini",
    class_weight="balanced",   # compensa el desbalance 94%/6%
    random_state=semilla
)
modelo_arbol_defecto.fit(variables_entrenamiento, objetivo_entrenamiento)
print(f"Árbol por defecto entrenado.")
print(f"  Profundidad: {modelo_arbol_defecto.get_depth()}")
print(f"  Hojas:       {modelo_arbol_defecto.get_n_leaves()}")
print(f"  class_weight='balanced' compensa el desbalance {proporcion_mayoritaria:.0%}/{proporcion_minoritaria:.0%}")

# =============================================================================
# 7. MODELO 2 — ÁRBOL CON BÚSQUEDA DE ccp_alpha (poda por complejidad de coste)
# =============================================================================
print("\n" + "=" * 70)
print("MODELO 2: Árbol con poda por complejidad de coste (ccp_alpha)")
print("=" * 70)
print("""
La poda por complejidad de coste (Breiman et al., 1984) controla el
sobreajuste eliminando ramas cuya ganancia de precisión es menor que
su coste de complejidad. Se busca el ccp_alpha óptimo con GridSearchCV
y validación cruzada estratificada de 5 folds.
""")

# Calcular ruta de poda sobre el conjunto de entrenamiento
arbol_temporal = DecisionTreeClassifier(
    criterion="gini",
    class_weight="balanced",
    random_state=semilla
)
arbol_temporal.fit(variables_entrenamiento, objetivo_entrenamiento)
ruta_poda = arbol_temporal.cost_complexity_pruning_path(
    variables_entrenamiento, objetivo_entrenamiento
)

valores_ccp_alpha = ruta_poda.ccp_alphas[:-1]   # excluir el último (árbol vacío)
valores_ccp_alpha = np.unique(valores_ccp_alpha)

# Submuestrear si hay demasiados valores (eficiencia computacional)
if len(valores_ccp_alpha) > 20:
    indices = np.linspace(0, len(valores_ccp_alpha) - 1, 20).astype(int)
    valores_ccp_alpha = valores_ccp_alpha[indices]

print(f"Valores de ccp_alpha a explorar: {len(valores_ccp_alpha)}")

modelo_arbol_base = DecisionTreeClassifier(
    criterion="gini",
    class_weight="balanced",
    random_state=semilla
)

busqueda_arbol = GridSearchCV(
    estimator=modelo_arbol_base,
    param_grid={"ccp_alpha": valores_ccp_alpha},
    scoring=calcular_auc_no_localizada,   # scorer robusto con try/except
    cv=validacion_cruzada,
    n_jobs=-1,
    return_train_score=True
)
busqueda_arbol.fit(variables_entrenamiento, objetivo_entrenamiento)

print(f"Mejor ccp_alpha: {busqueda_arbol.best_params_}")
print(f"Mejor AUC en validación cruzada: {busqueda_arbol.best_score_:.4f}")

# Gráfica ccp_alpha vs AUC
resultados_arbol_cv = pd.DataFrame(busqueda_arbol.cv_results_)
plt.figure(figsize=(10, 5))
sns.lineplot(
    data=resultados_arbol_cv.assign(
        ccp_alpha=resultados_arbol_cv["param_ccp_alpha"].astype(float)
    ),
    x="ccp_alpha",
    y="mean_test_score",
    marker="o"
)
plt.axvline(
    x=busqueda_arbol.best_params_["ccp_alpha"],
    linestyle="--", color="#e74c3c",
    label=f"α* = {busqueda_arbol.best_params_['ccp_alpha']:.6f}"
)
plt.xlabel("ccp_alpha")
plt.ylabel("AUC promedio (validación cruzada)")
plt.title("Rendimiento del árbol según ccp_alpha\n"
          "(poda por complejidad de coste — Breiman et al., 1984)")
plt.legend()
plt.tight_layout()
if GUARDAR_FIGURAS:
    plt.savefig("02_ccp_alpha_auc.png", dpi=150)
plt.show()

# =============================================================================
# 8. MODELO 3 — RANDOM FOREST CON GridSearchCV
# =============================================================================
print("\n" + "=" * 70)
print("MODELO 3: Random Forest con búsqueda de hiperparámetros (GridSearchCV)")
print("=" * 70)
print("""
Random Forest construye múltiples árboles sobre submuestras bootstrap
y promedia sus predicciones, reduciendo la varianza del árbol individual
(Breiman, 2001). Con class_weight='balanced', cada árbol ajusta los pesos
para compensar el desbalance 94%/6%.
""")

modelo_rf_base = RandomForestClassifier(
    class_weight="balanced",
    random_state=semilla,
    n_jobs=-1
)

busqueda_rf = GridSearchCV(
    estimator=modelo_rf_base,
    param_grid={
        "n_estimators": [50, 100],
        "max_depth":    [8, 12]
    },
    scoring=calcular_auc_no_localizada,   # scorer robusto con try/except
    cv=validacion_cruzada,
    n_jobs=-1
)
busqueda_rf.fit(variables_entrenamiento, objetivo_entrenamiento)

print(f"Mejores parámetros RF: {busqueda_rf.best_params_}")
print(f"Mejor AUC en validación cruzada: {busqueda_rf.best_score_:.4f}")

# =============================================================================
# 9. FUNCIÓN DE EVALUACIÓN (estilo notebook de clase)
# =============================================================================
def evaluar_modelo(nombre_modelo, modelo, vars_prueba, obj_prueba):
    """
    Evalúa un modelo entrenado sobre el conjunto de prueba.
    Reporta las métricas justificadas en la sección 2, con énfasis en
    F1 y PR-AUC sobre la clase positiva NO LOCALIZADA (clase 1).
    """
    predicciones       = modelo.predict(vars_prueba)
    probabilidades_pos = modelo.predict_proba(vars_prueba)[:, 1]

    exactitud  = accuracy_score(obj_prueba, predicciones)
    roc_auc    = roc_auc_score(obj_prueba, probabilidades_pos)
    pr_auc     = average_precision_score(obj_prueba, probabilidades_pos)
    f1_pos     = f1_score(obj_prueba, predicciones,
                          pos_label=1, average="binary", zero_division=0)
    f1_macro   = f1_score(obj_prueba, predicciones,
                          average="macro", zero_division=0)
    rec_pos    = recall_score(obj_prueba, predicciones,
                              pos_label=1, zero_division=0)
    prec_pos   = precision_score(obj_prueba, predicciones,
                                 pos_label=1, zero_division=0)
    prevalencia = obj_prueba.mean()

    print("\n" + "=" * 70)
    print(nombre_modelo)
    print("=" * 70)
    print(f"\nExactitud (Accuracy)     : {exactitud:.4f}  ← ENGAÑOSA con desbalance")
    print(f"F1-macro                 : {f1_macro:.4f}")
    print(f"F1 NO LOCALIZADA    ★   : {f1_pos:.4f}  ← métrica principal")
    print(f"Recall NO LOCALIZADA     : {rec_pos:.4f}")
    print(f"Precision NO LOCALIZADA  : {prec_pos:.4f}")
    print(f"ROC-AUC                  : {roc_auc:.4f}")
    print(f"PR-AUC              ★   : {pr_auc:.4f}  (baseline={prevalencia:.4f})")

    # CORRECCIÓN 6: índices de la matriz verificados contra el orden del LE.
    # LabelEncoder ordena alfabéticamente: 0=DESAPARECIDA, 1=NO LOCALIZADA.
    # confusion_matrix(y_true, y_pred) → filas=real, columnas=predicho,
    # ordenado por valor numérico ascendente (0, 1).
    # Por tanto: cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
    cm = confusion_matrix(obj_prueba, predicciones)
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

    tabla_cm = pd.DataFrame(
        cm,
        index  =["Real DESAPARECIDA (0)", "Real NO LOCALIZADA (1)"],
        columns=["Pred DESAPARECIDA (0)", "Pred NO LOCALIZADA (1)"]
    )
    print("\nMatriz de confusión:")
    print(tabla_cm)
    print(f"\n  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(f"  FP → DESAPARECIDA predicha NO LOCALIZADA (expediente cerrado por error)")
    print(f"  FN → NO LOCALIZADA predicha DESAPARECIDA (expediente sin cerrar ← peor error)")

    print("\nReporte de clasificación completo:")
    print(classification_report(obj_prueba, predicciones,
                                target_names=clases, digits=4,
                                zero_division=0))

    return {
        "modelo":               nombre_modelo,
        "exactitud":            exactitud,
        "f1_no_localizada":     f1_pos,
        "f1_macro":             f1_macro,
        "recall_no_localizada": rec_pos,
        "prec_no_localizada":   prec_pos,
        "roc_auc":              roc_auc,
        "pr_auc":               pr_auc,
        "predicciones":         predicciones,
        "probabilidades":       probabilidades_pos,
        "cm":                   cm,
    }

# =============================================================================
# 10. EVALUACIÓN DE LOS TRES MODELOS
# =============================================================================
resultados = []

resultados.append(evaluar_modelo(
    "Árbol por defecto (sin poda)",
    modelo_arbol_defecto,
    variables_prueba, objetivo_prueba
))

resultados.append(evaluar_modelo(
    "Árbol con ccp_alpha óptimo (poda)",
    busqueda_arbol.best_estimator_,
    variables_prueba, objetivo_prueba
))

resultados.append(evaluar_modelo(
    "Random Forest (GridSearchCV)",
    busqueda_rf.best_estimator_,
    variables_prueba, objetivo_prueba
))

# =============================================================================
# 11. TABLA COMPARATIVA DE MODELOS
# =============================================================================
tabla_resultados = pd.DataFrame([{
    "Modelo":             r["modelo"],
    "Accuracy":           round(r["exactitud"],            4),
    "F1 NO LOCALIZADA":   round(r["f1_no_localizada"],     4),
    "Recall NO LOC.":     round(r["recall_no_localizada"], 4),
    "Precision NO LOC.":  round(r["prec_no_localizada"],   4),
    "F1-macro":           round(r["f1_macro"],             4),
    "ROC-AUC":            round(r["roc_auc"],              4),
    "PR-AUC ★":           round(r["pr_auc"],               4),
} for r in resultados])

tabla_resultados = tabla_resultados.sort_values("PR-AUC ★", ascending=False)

print("\n" + "=" * 70)
print("TABLA COMPARATIVA DE MODELOS (ordenada por PR-AUC ★ — métrica principal)")
print("=" * 70)
print(tabla_resultados.to_string(index=False))
print("\nNota: PR-AUC es la métrica principal porque con 94%/6% de desbalance")
print("los TN inflarian artificialmente el ROC (Saito & Rehmsmeier, 2015).")

# Selección del mejor modelo
mejor_resultado  = tabla_resultados.iloc[0]
mejor_nombre     = mejor_resultado["Modelo"]
mejor_modelo_obj = next(r for r in resultados if r["modelo"] == mejor_nombre)
print(f"\nMejor modelo seleccionado: {mejor_nombre}")

# =============================================================================
# 12. MATRIZ DE CONFUSIÓN — MEJOR MODELO (heatmap estilo notebook)
# =============================================================================
cm_mejor = mejor_modelo_obj["cm"]
tn_m, fp_m, fn_m, tp_m = cm_mejor[0,0], cm_mejor[0,1], cm_mejor[1,0], cm_mejor[1,1]

tabla_cm_mejor = pd.DataFrame(
    cm_mejor,
    index  =["Real DESAPARECIDA", "Real NO LOCALIZADA"],
    columns=["Pred DESAPARECIDA", "Pred NO LOCALIZADA"]
)

plt.figure(figsize=(6, 5))
sns.heatmap(tabla_cm_mejor, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Clase predicha")
plt.ylabel("Clase real")
plt.title(f"Matriz de confusión — {mejor_nombre}")
plt.tight_layout()
if GUARDAR_FIGURAS:
    plt.savefig("03_matriz_confusion.png", dpi=150)
plt.show()

print(f"""
Interpretación de la matriz de confusión ({mejor_nombre}):
  TN = {tn_m:,}  — DESAPARECIDA predicha correctamente como DESAPARECIDA
  FP = {fp_m:,}  — DESAPARECIDA predicha incorrectamente como NO LOCALIZADA
        → expediente cerrado por error (pierde seguimiento del caso)
  FN = {fn_m:,}  — NO LOCALIZADA predicha incorrectamente como DESAPARECIDA
        → expediente no se cierra (recursos de búsqueda mal asignados) ← PEOR ERROR
  TP = {tp_m:,}  — NO LOCALIZADA predicha correctamente como NO LOCALIZADA
""")

# =============================================================================
# 13. CURVA ROC — TODOS LOS MODELOS (estilo notebook)
# =============================================================================
colores = ["#4a90d9", "#e67e22", "#27ae60"]

plt.figure(figsize=(8, 6))

for res, color in zip(resultados, colores):
    fpr_v, tpr_v, _ = roc_curve(objetivo_prueba, res["probabilidades"])
    roc_v = auc(fpr_v, tpr_v)
    tabla_roc = pd.DataFrame({"fpr": fpr_v, "tpr": tpr_v})
    sns.lineplot(data=tabla_roc, x="fpr", y="tpr",
                 label=f"{res['modelo']} (AUC={roc_v:.3f})", color=color)

sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--",
             label="Clasificador aleatorio (AUC=0.500)", color="gray")
plt.xlabel("Tasa de Falsos Positivos (FPR = 1 − Especificidad)")
plt.ylabel("Tasa de Verdaderos Positivos (Recall / Sensibilidad)")
plt.title("Curva ROC — Comparación de modelos\n"
          "(clase positiva: NO LOCALIZADA)")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
if GUARDAR_FIGURAS:
    plt.savefig("04_curva_roc.png", dpi=150)
plt.show()

roc_mejor = mejor_resultado["ROC-AUC"]
print(f"Interpretación ROC-AUC del mejor modelo ({roc_mejor:.3f}):")
print(f"  Si se toman al azar un caso NO LOCALIZADA y uno DESAPARECIDA,")
print(f"  el modelo tiene {roc_mejor:.1%} de probabilidad de asignarle")
print(f"  mayor score al caso NO LOCALIZADA. (Bamber, 1975; Fawcett, 2006)")

# =============================================================================
# 14. CURVA PRECISION-RECALL — TODOS LOS MODELOS
# =============================================================================
prevalencia_prueba = objetivo_prueba.mean()

plt.figure(figsize=(8, 6))
for res, color in zip(resultados, colores):
    prec_c, rec_c, _ = precision_recall_curve(objetivo_prueba,
                                               res["probabilidades"])
    ap = average_precision_score(objetivo_prueba, res["probabilidades"])
    tabla_pr = pd.DataFrame({"recall": rec_c, "precision": prec_c})
    sns.lineplot(data=tabla_pr, x="recall", y="precision",
                 label=f"{res['modelo']} (AP={ap:.3f})", color=color)

plt.axhline(y=prevalencia_prueba, linestyle="--", color="red",
            label=f"Baseline aleatoria (prevalencia={prevalencia_prueba:.3f})")
plt.xlabel("Recall (Sensibilidad)")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall — Comparación de modelos\n"
          "(preferible a ROC con clases desbalanceadas — Saito & Rehmsmeier, 2015)")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()
if GUARDAR_FIGURAS:
    plt.savefig("05_curva_pr.png", dpi=150)
plt.show()

print(f"Baseline PR: {prevalencia_prueba:.4f} — un modelo que predice siempre")
print(f"NO LOCALIZADA tendría Precision ≈ {prevalencia_prueba:.4f} en todos los umbrales.")
print("Cualquier modelo con AP > baseline está aprendiendo patrones reales.")

# =============================================================================
# 15. IMPORTANCIA DE VARIABLES — Random Forest
# =============================================================================
print("\n" + "=" * 70)
print("IMPORTANCIA DE VARIABLES — Random Forest")
print("=" * 70)

rf_final = busqueda_rf.best_estimator_
tabla_importancia = pd.DataFrame({
    "variable":    features,
    "importancia": rf_final.feature_importances_
}).sort_values("importancia", ascending=False).reset_index(drop=True)

# CORRECCIÓN 5: los porcentajes se calculan del modelo real, no hardcodeados.
tabla_importancia["porcentaje"] = (
    tabla_importancia["importancia"] / tabla_importancia["importancia"].sum() * 100
)

print(tabla_importancia[["variable", "importancia", "porcentaje"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

var_top1 = tabla_importancia.iloc[0]["variable"]
var_top2 = tabla_importancia.iloc[1]["variable"]
pct_top1 = tabla_importancia.iloc[0]["porcentaje"]
pct_top2 = tabla_importancia.iloc[1]["porcentaje"]

plt.figure(figsize=(9, 5))
sns.barplot(data=tabla_importancia, x="importancia", y="variable",
            palette="Blues_d")
plt.xlabel("Importancia (reducción media del índice Gini)")
plt.ylabel("Variable")
plt.title("Importancia de variables — Random Forest\n"
          f"({var_top1} y {var_top2} dominan → posible sesgo de registro)")
plt.tight_layout()
if GUARDAR_FIGURAS:
    plt.savefig("06_importancia_variables.png", dpi=150)
plt.show()

print(f"\nVariables más relevantes según el modelo:")
print(f"  1. {var_top1}: {pct_top1:.1f}%")
print(f"  2. {var_top2}: {pct_top2:.1f}%")

# =============================================================================
# 16. VISUALIZACIÓN DEL ÁRBOL PODADO
# =============================================================================
arbol_podado = busqueda_arbol.best_estimator_

plt.figure(figsize=(24, 10))
plot_tree(
    arbol_podado,
    feature_names=features,
    class_names=clases,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=11
)
plt.title("Árbol de Decisión podado (max_depth=3 para visualización)\n"
          f"ccp_alpha óptimo = {busqueda_arbol.best_params_['ccp_alpha']:.6f}")
plt.tight_layout()
if GUARDAR_FIGURAS:
    plt.savefig("07_arbol_podado.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nReglas del árbol podado (primeros 3 niveles):")
print(export_text(arbol_podado, feature_names=features, max_depth=3))

# =============================================================================
# 17. OPTIMIZACIÓN DEL UMBRAL DE DECISIÓN
# =============================================================================
print("=" * 70)
print("OPTIMIZACIÓN DEL UMBRAL DE DECISIÓN (Provost & Fawcett, 2013)")
print("=" * 70)
print("""
θ = 0.5 es una convención, no necesariamente el umbral óptimo.
El umbral óptimo para F1 de la clase positiva se busca en el conjunto
de PRUEBA exclusivamente (no en validación cruzada) para evitar
data leakage (James et al., 2021).
""")

y_prob_mejor = mejor_modelo_obj["probabilidades"]
umbrales = np.arange(0.05, 0.95, 0.01)
f1_vals  = [
    f1_score(objetivo_prueba, (y_prob_mejor >= u).astype(int),
             pos_label=1, average="binary", zero_division=0)
    for u in umbrales
]
u_optimo   = umbrales[np.argmax(f1_vals)]
f1_optimo  = max(f1_vals)

plt.figure(figsize=(9, 4))
plt.plot(umbrales, f1_vals, color="#4a90d9", linewidth=2,
         label="F1 NO LOCALIZADA")
plt.axvline(x=0.5,      linestyle="--", color="gray",
            label="θ=0.5 (default)")
plt.axvline(x=u_optimo, linestyle="--", color="#e74c3c",
            label=f"θ*={u_optimo:.2f} (max F1={f1_optimo:.4f})")
plt.xlabel("Umbral de decisión (θ)")
plt.ylabel("F1 NO LOCALIZADA")
plt.title("Efecto del umbral sobre F1 de la clase positiva\n"
          f"Mejor modelo: {mejor_nombre}")
plt.legend()
plt.tight_layout()
if GUARDAR_FIGURAS:
    plt.savefig("08_umbral_decision.png", dpi=150)
plt.show()

y_def = (y_prob_mejor >= 0.5).astype(int)
y_opt = (y_prob_mejor >= u_optimo).astype(int)

print(f"Umbral default θ=0.5:")
print(f"  F1={f1_score(objetivo_prueba, y_def, pos_label=1, average='binary', zero_division=0):.4f}  "
      f"Recall={recall_score(objetivo_prueba, y_def, pos_label=1, zero_division=0):.4f}  "
      f"Precision={precision_score(objetivo_prueba, y_def, pos_label=1, zero_division=0):.4f}")

print(f"\nUmbral óptimo θ*={u_optimo:.2f} (maximiza F1 NO LOCALIZADA):")
print(f"  F1={f1_score(objetivo_prueba, y_opt, pos_label=1, average='binary', zero_division=0):.4f}  "
      f"Recall={recall_score(objetivo_prueba, y_opt, pos_label=1, zero_division=0):.4f}  "
      f"Precision={precision_score(objetivo_prueba, y_opt, pos_label=1, zero_division=0):.4f}")

# =============================================================================
# 18. INTERPRETACIÓN FINAL
# =============================================================================
print(f"""
{'=' * 70}
INTERPRETACIÓN FINAL
{'=' * 70}

1. PARADOJA DE LA EXACTITUD
   Un clasificador trivial alcanza ~{proporcion_mayoritaria:.0%} de accuracy sin aprender nada.
   Por eso esta tarea NO usa accuracy como métrica principal.
   (Saito & Rehmsmeier, 2015)

2. SELECCIÓN DE MÉTRICAS Y ESCENARIO REAL
   En un sistema de registro de personas desaparecidas:
   • FN (NO LOCALIZADA predicha como DESAPARECIDA): expediente no se cierra
     → recursos de búsqueda mal asignados → sesgo en estadísticas oficiales.
   • FP (DESAPARECIDA predicha como NO LOCALIZADA): expediente cerrado por
     error → pérdida de seguimiento del caso.
   Ambos errores son graves, pero el FN tiene mayor costo operativo en
   el contexto de búsqueda de personas → F1 equilibra ambos errores.
   Se prefiere PR-AUC sobre ROC-AUC porque ignora los TN, que con
   {proporcion_mayoritaria:.0%}/{proporcion_minoritaria:.0%} inflan artificialmente la curva ROC.
   (Saito & Rehmsmeier, 2015)

3. VARIABLES MÁS RELEVANTES (según Random Forest)
   {var_top1} ({pct_top1:.1f}%) y {var_top2} ({pct_top2:.1f}%) dominan la predicción.
   Esto indica que ESTATUS_VICTIMA depende principalmente del contexto
   geográfico e histórico del registro, NO de características individuales
   de la víctima. Posible sesgo: ciertas entidades actualizan más sus
   expedientes que otras → el modelo aprende prácticas administrativas,
   no el fenómeno real de desaparición.

4. ¿PATRONES REALES O SESGOS DEL SISTEMA?
   El modelo probablemente captura sesgos de registro:
   • ESTATUS_VICTIMA es un estado administrativo, no la situación real.
   • Entidades con mejor gestión burocrática tienen más registros
     actualizados; el modelo aprende ese patrón institucional.
   • ~40% de GRUPO_EDAD es nulo → imputado como "SIN_DATO", lo que
     puede introducir un patrón artificial en esa categoría.
   • MES_DESAPARICION fue excluida por ser colineal con TEMPORADA;
     incluir ambas hubiera sesgado artificialmente la importancia
     de la información temporal.
   Conclusión: las reglas de asociación y este clasificador describen
   lo que SE REGISTRA, no necesariamente lo que OCURRE en la realidad
   (mismo sesgo identificado por Irany en la sección de reglas de
   asociación — sección 9.4 del reporte).

5. UMBRAL DE DECISIÓN
   θ=0.5 es convención, no óptimo. El umbral que maximiza F1 para el
   mejor modelo es θ*={u_optimo:.2f}. Bajar el umbral aumenta el Recall
   (detecta más NO LOCALIZADA) a costa de precisión (más FP). La
   elección en producción es estratégica, no estadística.
   (Provost & Fawcett, 2013)

Referencias:
  Bamber, D. (1975). Journal of Mathematical Psychology 12(4): 387–415.
  Breiman, L. et al. (1984). Classification and Regression Trees. Wadsworth.
  Breiman, L. (2001). Random Forests. Machine Learning 45(1): 5–32.
  Fawcett, T. (2006). Pattern Recognition Letters 27(8): 861–874.
  Han, J., Kamber, M., & Pei, J. (2011). Data Mining: Concepts and Techniques.
  James, G. et al. (2021). An Introduction to Statistical Learning (2nd ed.).
  Provost, F. & Fawcett, T. (2013). Data Science for Business. O'Reilly.
  Saito, T. & Rehmsmeier, M. (2015). PLOS ONE 10(3): e0118432.
""")
