from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="AlertaEdu ML - Riesgo de deserción",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background:#f4f7fb}
    .block-container {padding-top:1.3rem; max-width:1500px}
    [data-testid="stSidebar"] {background:#102a43}
    [data-testid="stSidebar"] * {color:#f8fafc}
    .hero {background:linear-gradient(120deg,#102a43,#176b87); padding:22px 26px;
           border-radius:16px; color:white; margin-bottom:18px; box-shadow:0 8px 24px #102a4328}
    .hero h1 {margin:0; font-size:30px; color:white}
    .hero p {margin:6px 0 0; color:#d9f0f4}
    .card {background:white; border-radius:14px; padding:16px; border:1px solid #e3eaf2;
           box-shadow:0 4px 14px #102a4312; min-height:120px}
    .card .value {font-size:28px; font-weight:800; color:#102a43}
    .card .label {font-size:14px; color:#62748a; margin-top:5px}
    .risk-high {color:#b42318; font-weight:800}
    .risk-mid {color:#b54708; font-weight:800}
    .risk-low {color:#067647; font-weight:800}
    div[data-testid="stPlotlyChart"] {background:white; border-radius:14px;
           border:1px solid #e3eaf2; box-shadow:0 4px 14px #102a4312}
    </style>
    """,
    unsafe_allow_html=True,
)

TARGET = "Desercion"
ID_COLUMN = "EstudianteID"
NUMERIC_FEATURES = [
    "Edad", "Promedio", "Asistencia", "Tardanzas", "CursosDesaprobados",
    "Repitencias", "IngresoFamiliar", "DistanciaColegioKm", "HorasTrabajoSemanal",
    "ApoyoFamiliar",
]
CATEGORICAL_FEATURES = [
    "Distrito", "Grado", "Sexo", "TipoFamilia", "AccesoInternet", "RecibeBeca",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@st.cache_data
def load_data():
    csv_path = Path(__file__).with_name("estudiantes_desercion.csv")
    data = pd.read_csv(csv_path, encoding="utf-8")
    required = [ID_COLUMN, TARGET, *FEATURES]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Faltan columnas en el CSV: " + ", ".join(missing))
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[TARGET]).drop_duplicates(subset=[ID_COLUMN])
    data[TARGET] = data[TARGET].astype(int)
    return data


def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def model_catalog():
    return {
        "Regresión logística": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Árbol de decisión": DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=240, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
        "Perceptrón multicapa": MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.001, max_iter=800, early_stopping=True, random_state=42),
    }


@st.cache_resource
def train_models(data_signature, data):
    del data_signature
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.25, stratify=data[TARGET], random_state=42
    )
    rows, trained = [], {}
    for name, estimator in model_catalog().items():
        pipeline = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        start = time.perf_counter()
        pipeline.fit(x_train, y_train)
        elapsed_ms = (time.perf_counter() - start) * 1000
        prediction = pipeline.predict(x_test)
        probability = pipeline.predict_proba(x_test)[:, 1]
        rows.append({
            "Modelo": name,
            "Exactitud": accuracy_score(y_test, prediction),
            "Precisión": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1-score": f1_score(y_test, prediction, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probability),
            "Entrenamiento (ms)": elapsed_ms,
        })
        trained[name] = {"pipeline": pipeline, "prediction": prediction, "probability": probability}
    metrics = pd.DataFrame(rows).sort_values(["F1-score", "ROC-AUC"], ascending=False).reset_index(drop=True)
    return metrics, trained, x_test, y_test


def risk_level(probability):
    if probability >= 0.65:
        return "Alto"
    if probability >= 0.35:
        return "Medio"
    return "Bajo"


def kpi(value, label):
    st.markdown(f'<div class="card"><div class="value">{value}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)


try:
    df = load_data()
except Exception as error:
    st.error(f"No se pudo cargar estudiantes_desercion.csv: {error}")
    st.stop()

signature = (len(df), float(df[TARGET].sum()), tuple(df.columns))
metrics, trained_models, x_test, y_test = train_models(signature, df)
best_model_name = metrics.iloc[0]["Modelo"]
best_pipeline = trained_models[best_model_name]["pipeline"]

with st.sidebar:
    st.title("🎓 AlertaEdu ML")
    st.caption("Sistema de detección temprana")
    page = st.radio("Módulo", ["Panel general", "Evaluación ML", "Predicción individual", "Alertas y datos"])
    st.divider()
    district_filter = st.multiselect("Distrito", sorted(df["Distrito"].dropna().unique()))
    grade_filter = st.multiselect("Grado", sorted(df["Grado"].dropna().unique()))
    st.caption("Datos anonimizados con fines académicos. Una predicción apoya la decisión profesional, no la reemplaza.")

filtered = df.copy()
if district_filter:
    filtered = filtered[filtered["Distrito"].isin(district_filter)]
if grade_filter:
    filtered = filtered[filtered["Grado"].isin(grade_filter)]

st.markdown(
    """<div class="hero"><h1>Sistema de información con Machine Learning</h1>
    <p>Detección del riesgo de deserción en colegios secundarios de Lima Norte, 2026</p></div>""",
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No hay registros para los filtros seleccionados.")
    st.stop()

if page == "Panel general":
    probabilities = best_pipeline.predict_proba(filtered[FEATURES])[:, 1]
    view = filtered.copy()
    view["Probabilidad"] = probabilities
    view["NivelRiesgo"] = [risk_level(value) for value in probabilities]
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi(f"{len(view):,}", "Estudiantes analizados")
    with c2: kpi(f"{(view['NivelRiesgo'] == 'Alto').sum():,}", "Alertas de riesgo alto")
    with c3: kpi(f"{view['Asistencia'].mean():.1f}%", "Asistencia promedio")
    with c4: kpi(f"{view['Promedio'].mean():.1f}", "Promedio académico")

    left, right = st.columns([1.2, 1])
    with left:
        counts = view["NivelRiesgo"].value_counts().reindex(["Alto", "Medio", "Bajo"], fill_value=0).reset_index()
        counts.columns = ["Nivel", "Estudiantes"]
        fig = px.bar(counts, x="Nivel", y="Estudiantes", color="Nivel", text_auto=True,
                     color_discrete_map={"Alto":"#d92d20", "Medio":"#f79009", "Bajo":"#12b76a"},
                     title=f"Niveles de riesgo estimados - {best_model_name}")
        fig.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        district = view.groupby("Distrito", as_index=False).agg(Estudiantes=(ID_COLUMN, "count"), RiesgoReal=(TARGET, "mean"))
        district["RiesgoReal"] *= 100
        fig = px.bar(district, x="Distrito", y="RiesgoReal", text_auto=".1f", color="RiesgoReal",
                     color_continuous_scale="OrRd", title="Deserción observada por distrito (%)")
        fig.update_layout(coloraxis_showscale=False, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        fig = px.scatter(view, x="Asistencia", y="Promedio", color="NivelRiesgo",
                         color_discrete_map={"Alto":"#d92d20", "Medio":"#f79009", "Bajo":"#12b76a"},
                         hover_data=[ID_COLUMN, "CursosDesaprobados", "Repitencias"],
                         title="Rendimiento, asistencia y riesgo")
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        factors = pd.DataFrame({
            "Factor": ["Asistencia baja (<75%)", "Promedio bajo (<11)", "Trabaja (>15 h/sem)", "Sin apoyo familiar", "Con repitencias"],
            "Estudiantes": [(view["Asistencia"] < 75).sum(), (view["Promedio"] < 11).sum(),
                            (view["HorasTrabajoSemanal"] > 15).sum(), (view["ApoyoFamiliar"] <= 1).sum(),
                            (view["Repitencias"] > 0).sum()],
        }).sort_values("Estudiantes")
        fig = px.bar(factors, x="Estudiantes", y="Factor", orientation="h", text_auto=True,
                     color="Estudiantes", color_continuous_scale="Blues", title="Factores de alerta presentes")
        fig.update_layout(coloraxis_showscale=False, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Evaluación ML":
    st.subheader("Comparación de algoritmos supervisados")
    st.caption("División estratificada: 75% entrenamiento y 25% prueba. El sistema selecciona el mayor F1-score.")
    display_metrics = metrics.copy()
    for column in ["Exactitud", "Precisión", "Recall", "F1-score", "ROC-AUC"]:
        display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.2%}")
    display_metrics["Entrenamiento (ms)"] = display_metrics["Entrenamiento (ms)"].map(lambda value: f"{value:.1f}")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)
    st.success(f"Modelo seleccionado: {best_model_name}")

    selected = st.selectbox("Modelo para diagnóstico", metrics["Modelo"].tolist())
    result = trained_models[selected]
    cm = confusion_matrix(y_test, result["prediction"])
    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicción", y="Valor real", color="Casos"),
                        x=["Permanece", "Deserta"], y=["Permanece", "Deserta"], title="Matriz de confusión")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        for name, result_item in trained_models.items():
            fpr, tpr, _ = roc_curve(y_test, result_item["probability"])
            auc = roc_auc_score(y_test, result_item["probability"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#98a2b3"), name="Azar"))
        fig.update_layout(title="Curvas ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Sensibilidad", paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Predicción individual":
    st.subheader("Evaluar un nuevo registro estudiantil")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Edad", 11, 20, 15)
            average = st.number_input("Promedio (0-20)", 0.0, 20.0, 13.0, 0.1)
            attendance = st.slider("Asistencia (%)", 0, 100, 85)
            lateness = st.number_input("Tardanzas", 0, 80, 5)
            failed = st.number_input("Cursos desaprobados", 0, 12, 1)
            repeats = st.number_input("Repitencias", 0, 4, 0)
        with c2:
            income = st.number_input("Ingreso familiar mensual (S/)", 300, 15000, 1800, 100)
            distance = st.number_input("Distancia al colegio (km)", 0.1, 30.0, 3.0, 0.1)
            work_hours = st.number_input("Horas de trabajo semanal", 0, 60, 0)
            family_support = st.slider("Apoyo familiar (0=nulo, 5=alto)", 0, 5, 3)
            district = st.selectbox("Distrito", sorted(df["Distrito"].unique()))
            grade = st.selectbox("Grado", sorted(df["Grado"].unique()))
        with c3:
            sex = st.selectbox("Sexo", sorted(df["Sexo"].unique()))
            family = st.selectbox("Tipo de familia", sorted(df["TipoFamilia"].unique()))
            internet = st.selectbox("Acceso a internet", ["Sí", "No"])
            scholarship = st.selectbox("Recibe beca", ["Sí", "No"])
            model_choice = st.selectbox("Modelo", metrics["Modelo"].tolist(), index=metrics["Modelo"].tolist().index(best_model_name))
        submitted = st.form_submit_button("Calcular riesgo", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{
            "Edad": age, "Promedio": average, "Asistencia": attendance, "Tardanzas": lateness,
            "CursosDesaprobados": failed, "Repitencias": repeats, "IngresoFamiliar": income,
            "DistanciaColegioKm": distance, "HorasTrabajoSemanal": work_hours, "ApoyoFamiliar": family_support,
            "Distrito": district, "Grado": grade, "Sexo": sex, "TipoFamilia": family,
            "AccesoInternet": internet, "RecibeBeca": scholarship,
        }])
        probability = trained_models[model_choice]["pipeline"].predict_proba(row)[0, 1]
        level = risk_level(probability)
        color = {"Alto":"#d92d20", "Medio":"#f79009", "Bajo":"#12b76a"}[level]
        st.markdown(f'<div class="card"><div class="value" style="color:{color}">{probability:.1%} - Riesgo {level}</div><div class="label">Probabilidad estimada por {model_choice}</div></div>', unsafe_allow_html=True)
        recommendations = []
        if attendance < 75: recommendations.append("Activar seguimiento de asistencia y comunicación con la familia.")
        if average < 11 or failed >= 3: recommendations.append("Programar refuerzo académico y tutoría en cursos críticos.")
        if work_hours > 15: recommendations.append("Evaluar flexibilidad horaria y apoyo socioeconómico.")
        if family_support <= 1: recommendations.append("Derivar a acompañamiento psicológico o trabajo social.")
        if level == "Alto": recommendations.append("Generar alerta prioritaria y un plan de intervención individual.")
        st.write("Acciones sugeridas:")
        for recommendation in recommendations or ["Mantener monitoreo periódico de los indicadores."]:
            st.write("- " + recommendation)

else:
    probabilities = best_pipeline.predict_proba(filtered[FEATURES])[:, 1]
    alerts = filtered.copy()
    alerts["ProbabilidadRiesgo"] = probabilities
    alerts["NivelRiesgo"] = [risk_level(value) for value in probabilities]
    alerts = alerts.sort_values("ProbabilidadRiesgo", ascending=False)
    st.subheader("Alertas tempranas y registros anonimizados")
    minimum = st.slider("Probabilidad mínima para mostrar", 0, 100, 35) / 100
    alerts = alerts[alerts["ProbabilidadRiesgo"] >= minimum]
    alerts["ProbabilidadRiesgo"] = alerts["ProbabilidadRiesgo"].map(lambda value: f"{value:.1%}")
    shown = [ID_COLUMN, "Distrito", "Grado", "Promedio", "Asistencia", "CursosDesaprobados", "Repitencias", "ProbabilidadRiesgo", "NivelRiesgo"]
    st.dataframe(alerts[shown], use_container_width=True, hide_index=True)
    st.download_button("Descargar alertas filtradas", alerts[shown].to_csv(index=False).encode("utf-8-sig"), "alertas_desercion.csv", "text/csv")
    with st.expander("Ver diccionario de datos"):
        st.markdown("""
        - **Desercion:** variable objetivo; 1 = desertó, 0 = permaneció.
        - **Promedio, Asistencia, Tardanzas, CursosDesaprobados y Repitencias:** variables académicas.
        - **IngresoFamiliar, HorasTrabajoSemanal, TipoFamilia, ApoyoFamiliar, AccesoInternet y RecibeBeca:** variables personales/socioeconómicas.
        - **EstudianteID:** código anónimo; no contiene nombres, DNI ni datos de contacto.
        """)
