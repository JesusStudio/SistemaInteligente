# =========================================================
# IMPORTACIÓN DE LIBRERÍAS
# =========================================================

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# CONFIGURACIÓN GENERAL DE STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Sistema ML - Riesgo de Deserción",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# ESTILOS CSS
# =========================================================

st.markdown(
    """
    <style>

        /* Página general */

        .stApp {
            background-color: #f1f5f4;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #17202a !important;
        }

        div[data-testid="stMarkdownContainer"] p {
            color: #17202a;
        }

        /* Encabezado */

        .encabezado {
            background-color: white;
            padding: 22px 25px;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.10);
            border-left: 7px solid #16745b;
            margin-bottom: 18px;
        }

        .encabezado h1 {
            color: #154734 !important;
            font-size: 28px;
            margin: 0;
            font-weight: 800;
        }

        .encabezado p {
            margin-top: 8px;
            margin-bottom: 0;
            color: #4b5563 !important;
            font-size: 15px;
        }

        /* Tarjetas KPI */

        .kpi-card {
            background-color: white;
            padding: 18px 10px;
            border-radius: 14px;
            text-align: center;
            min-height: 145px;
            border: 1px solid #e1e7e5;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.11);
        }

        .kpi-icono {
            font-size: 37px;
            margin-bottom: 4px;
        }

        .kpi-valor {
            color: #17202a;
            font-size: 24px;
            font-weight: 800;
        }

        .kpi-etiqueta {
            color: #65716d;
            font-size: 14px;
            margin-top: 5px;
        }

        /* Tarjetas de métricas ML */

        .metrica-card {
            background-color: white;
            padding: 17px 10px;
            border-radius: 13px;
            text-align: center;
            border-top: 5px solid #16745b;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.10);
        }

        .metrica-valor {
            color: #154734;
            font-size: 25px;
            font-weight: 800;
        }

        .metrica-nombre {
            color: #5d6663;
            font-size: 14px;
            margin-top: 4px;
        }

        /* Resultados de riesgo */

        .riesgo-bajo {
            background-color: #dff5e8;
            border-left: 7px solid #20884f;
            padding: 20px;
            border-radius: 12px;
            color: #14532d;
            font-weight: 700;
            font-size: 17px;
        }

        .riesgo-medio {
            background-color: #fff2cc;
            border-left: 7px solid #e1a800;
            padding: 20px;
            border-radius: 12px;
            color: #755600;
            font-weight: 700;
            font-size: 17px;
        }

        .riesgo-alto {
            background-color: #fde2e2;
            border-left: 7px solid #c62828;
            padding: 20px;
            border-radius: 12px;
            color: #8b1e1e;
            font-weight: 700;
            font-size: 17px;
        }

        /* Gráficos */

        div[data-testid="stPlotlyChart"] {
            background-color: white;
            border-radius: 14px;
            padding: 5px;
            border: 1px solid #e2e8e5;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.09);
        }

        /* Pestañas */

        button[data-baseweb="tab"] {
            font-weight: 700;
        }

        /* Dataframe */

        div[data-testid="stDataFrame"] {
            background-color: white;
            border-radius: 12px;
            padding: 5px;
            border: 1px solid #d9e2df;
        }

        /* Botones */

        div.stButton > button {
            width: 100%;
            background-color: #16745b;
            color: white;
            border-radius: 9px;
            border: none;
            font-weight: 700;
            padding: 10px;
        }

        div.stButton > button:hover {
            background-color: #105b47;
            color: white;
            border: none;
        }

        div.stDownloadButton > button {
            width: 100%;
            background-color: #16745b;
            color: white;
            border-radius: 9px;
            border: none;
            font-weight: 700;
        }

        /* Sidebar */

        section[data-testid="stSidebar"] {
            background-color: #e6f1ed;
        }

        @media (max-width: 900px) {

            .encabezado h1 {
                font-size: 22px;
            }

            .kpi-valor {
                font-size: 20px;
            }
        }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CONSTANTES
# =========================================================

ARCHIVO_CSV = "estudiantes_desercion.csv"

COLUMNAS_REQUERIDAS = [
    "EstudianteID",
    "Distrito",
    "Grado",
    "Sexo",
    "Edad",
    "Promedio",
    "Asistencia",
    "Tardanzas",
    "CursosDesaprobados",
    "Repitencias",
    "IngresoFamiliar",
    "ApoyoFamiliar",
    "RecibeBeca",
    "Desercion"
]

COLUMNAS_ENTRADA = [
    "Distrito",
    "Grado",
    "Sexo",
    "Edad",
    "Promedio",
    "Asistencia",
    "Tardanzas",
    "CursosDesaprobados",
    "Repitencias",
    "IngresoFamiliar",
    "ApoyoFamiliar",
    "RecibeBeca"
]

COLUMNAS_CATEGORICAS = [
    "Distrito",
    "Grado",
    "Sexo",
    "RecibeBeca"
]

COLUMNAS_NUMERICAS = [
    "Edad",
    "Promedio",
    "Asistencia",
    "Tardanzas",
    "CursosDesaprobados",
    "Repitencias",
    "IngresoFamiliar",
    "ApoyoFamiliar"
]


# =========================================================
# FUNCIONES PARA CARGAR Y LIMPIAR LOS DATOS
# =========================================================

@st.cache_data
def cargar_datos():
    """
    Carga, valida y prepara el archivo CSV.
    """

    if not os.path.exists(ARCHIVO_CSV):
        raise FileNotFoundError(
            f"No se encontró el archivo {ARCHIVO_CSV}."
        )

    datos = pd.read_csv(
        ARCHIVO_CSV,
        encoding="utf-8-sig"
    )

    datos.columns = datos.columns.str.strip()

    columnas_faltantes = [
        columna
        for columna in COLUMNAS_REQUERIDAS
        if columna not in datos.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas en el archivo CSV: "
            + ", ".join(columnas_faltantes)
        )

    # Limpiar columnas de texto
    columnas_texto = [
        "EstudianteID",
        "Distrito",
        "Grado",
        "Sexo",
        "RecibeBeca"
    ]

    for columna in columnas_texto:
        datos[columna] = (
            datos[columna]
            .astype(str)
            .str.strip()
        )

    # Convertir columnas numéricas
    columnas_convertir = COLUMNAS_NUMERICAS + ["Desercion"]

    for columna in columnas_convertir:
        datos[columna] = pd.to_numeric(
            datos[columna],
            errors="coerce"
        )

    # Eliminar registros incompletos
    datos = datos.dropna(
        subset=COLUMNAS_ENTRADA + ["Desercion"]
    ).copy()

    datos["Edad"] = datos["Edad"].astype(int)
    datos["Tardanzas"] = datos["Tardanzas"].astype(int)
    datos["CursosDesaprobados"] = (
        datos["CursosDesaprobados"].astype(int)
    )
    datos["Repitencias"] = datos["Repitencias"].astype(int)
    datos["ApoyoFamiliar"] = datos["ApoyoFamiliar"].astype(int)
    datos["Desercion"] = datos["Desercion"].astype(int)

    # Solo permitir las categorías 0 y 1
    datos = datos[
        datos["Desercion"].isin([0, 1])
    ].copy()

    return datos


# =========================================================
# ENTRENAMIENTO DEL MODELO
# =========================================================

@st.cache_resource
def entrenar_modelo(datos):
    """
    Entrena un modelo Random Forest y devuelve:
    modelo, métricas, matriz de confusión y datos de prueba.
    """

    X = datos[COLUMNAS_ENTRADA]
    y = datos["Desercion"]

    if y.nunique() < 2:
        raise ValueError(
            "El CSV debe contener estudiantes con Desercion = 0 "
            "y estudiantes con Desercion = 1."
        )

    # Dividir los datos:
    # 75 % entrenamiento y 25 % prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )
    )

    # Transformación de variables categóricas
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                COLUMNAS_CATEGORICAS
            ),
            (
                "numericas",
                "passthrough",
                COLUMNAS_NUMERICAS
            )
        ]
    )

    clasificador = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=3,
        class_weight="balanced",
        random_state=42
    )

    modelo = Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("clasificador", clasificador)
        ]
    )

    modelo.fit(
        X_entrenamiento,
        y_entrenamiento
    )

    predicciones = modelo.predict(X_prueba)

    probabilidades = modelo.predict_proba(X_prueba)[:, 1]

    metricas = {
        "accuracy": accuracy_score(
            y_prueba,
            predicciones
        ),
        "precision": precision_score(
            y_prueba,
            predicciones,
            zero_division=0
        ),
        "recall": recall_score(
            y_prueba,
            predicciones,
            zero_division=0
        ),
        "f1": f1_score(
            y_prueba,
            predicciones,
            zero_division=0
        )
    }

    matriz = confusion_matrix(
        y_prueba,
        predicciones,
        labels=[0, 1]
    )

    resultados_prueba = X_prueba.copy()

    resultados_prueba["ValorReal"] = y_prueba.values
    resultados_prueba["Prediccion"] = predicciones
    resultados_prueba["ProbabilidadDesercion"] = probabilidades

    return modelo, metricas, matriz, resultados_prueba


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def formato_decimal(valor, decimales=1):
    """
    Devuelve números con formato decimal.
    """

    return f"{valor:,.{decimales}f}".replace(",", "X").replace(
        ".",
        ","
    ).replace("X", ".")


def tarjeta_kpi(icono, valor, etiqueta):
    """
    Genera una tarjeta KPI.
    """

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icono">{icono}</div>
            <div class="kpi-valor">{valor}</div>
            <div class="kpi-etiqueta">{etiqueta}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def tarjeta_metrica(valor, nombre):
    """
    Genera una tarjeta para las métricas del modelo.
    """

    st.markdown(
        f"""
        <div class="metrica-card">
            <div class="metrica-valor">{valor:.2%}</div>
            <div class="metrica-nombre">{nombre}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def obtener_nivel_riesgo(probabilidad):
    """
    Clasifica la probabilidad en riesgo bajo, medio o alto.
    """

    if probabilidad < 0.35:
        return "Bajo"

    if probabilidad < 0.70:
        return "Medio"

    return "Alto"


def obtener_descripcion_riesgo(probabilidad):
    """
    Genera una recomendación según el nivel de riesgo.
    """

    nivel = obtener_nivel_riesgo(probabilidad)

    if nivel == "Bajo":
        return (
            "El estudiante presenta un riesgo bajo. "
            "Se recomienda mantener el seguimiento académico regular."
        )

    if nivel == "Medio":
        return (
            "El estudiante presenta un riesgo medio. "
            "Se recomienda realizar seguimiento tutorial y revisar "
            "sus principales factores académicos y familiares."
        )

    return (
        "El estudiante presenta un riesgo alto. "
        "Se recomienda una evaluación inmediata por parte del tutor "
        "o responsable académico."
    )


def generar_predicciones_masivas(modelo, datos):
    """
    Aplica el modelo a todos los estudiantes del archivo CSV.
    """

    tabla = datos.copy()

    probabilidades = modelo.predict_proba(
        tabla[COLUMNAS_ENTRADA]
    )[:, 1]

    predicciones = modelo.predict(
        tabla[COLUMNAS_ENTRADA]
    )

    tabla["ProbabilidadDesercion"] = (
        probabilidades * 100
    ).round(2)

    tabla["PrediccionML"] = np.where(
        predicciones == 1,
        "Posible deserción",
        "Continuidad probable"
    )

    tabla["NivelRiesgo"] = [
        obtener_nivel_riesgo(probabilidad)
        for probabilidad in probabilidades
    ]

    return tabla


def obtener_importancia_variables(modelo):
    """
    Obtiene la importancia de las variables utilizadas por Random Forest.
    """

    preprocesador = modelo.named_steps["preprocesamiento"]
    clasificador = modelo.named_steps["clasificador"]

    nombres_categoricos = (
        preprocesador
        .named_transformers_["categoricas"]
        .get_feature_names_out(COLUMNAS_CATEGORICAS)
    )

    nombres_variables = list(
        nombres_categoricos
    ) + COLUMNAS_NUMERICAS

    importancias = clasificador.feature_importances_

    tabla_importancia = pd.DataFrame(
        {
            "Variable": nombres_variables,
            "Importancia": importancias
        }
    )

    tabla_importancia = tabla_importancia.sort_values(
        "Importancia",
        ascending=False
    ).head(12)

    tabla_importancia["Variable"] = (
        tabla_importancia["Variable"]
        .str.replace("categoricas__", "", regex=False)
        .str.replace("_", ": ", n=1, regex=False)
    )

    return tabla_importancia


# =========================================================
# CARGA DEL ARCHIVO Y ENTRENAMIENTO
# =========================================================

try:
    df_original = cargar_datos()

    modelo_ml, metricas, matriz_confusion, resultados_prueba = (
        entrenar_modelo(df_original)
    )

except FileNotFoundError:
    st.error(
        "No se encontró el archivo estudiantes_desercion.csv. "
        "Colócalo en la misma carpeta que app.py."
    )
    st.stop()

except Exception as error:
    st.error(
        f"Ocurrió un error al preparar el sistema: {error}"
    )
    st.stop()


# =========================================================
# ENCABEZADO PRINCIPAL
# =========================================================

st.markdown(
    """
    <div class="encabezado">
        <h1>
            Sistema de detección del riesgo de deserción estudiantil
        </h1>
        <p>
            Implementación de un sistema de información con Machine
            Learning para detectar riesgo de deserción en colegios
            secundarios de Lima Norte, 2026.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR Y FILTROS
# =========================================================

st.sidebar.title("🎓 Panel de control")

st.sidebar.markdown(
    """
    Este sistema utiliza aprendizaje supervisado para analizar
    información académica, familiar y socioeconómica.
    """
)

st.sidebar.divider()

distritos_disponibles = sorted(
    df_original["Distrito"].unique()
)

grados_disponibles = sorted(
    df_original["Grado"].unique()
)

filtro_distrito = st.sidebar.multiselect(
    "Distrito",
    options=distritos_disponibles,
    default=distritos_disponibles
)

filtro_grado = st.sidebar.multiselect(
    "Grado",
    options=grados_disponibles,
    default=grados_disponibles
)

filtro_condicion = st.sidebar.selectbox(
    "Condición registrada",
    options=[
        "Todos",
        "Sin deserción",
        "Con deserción"
    ]
)

df_filtrado = df_original[
    df_original["Distrito"].isin(filtro_distrito)
    & df_original["Grado"].isin(filtro_grado)
].copy()

if filtro_condicion == "Sin deserción":
    df_filtrado = df_filtrado[
        df_filtrado["Desercion"] == 0
    ]

elif filtro_condicion == "Con deserción":
    df_filtrado = df_filtrado[
        df_filtrado["Desercion"] == 1
    ]


# =========================================================
# PESTAÑAS PRINCIPALES
# =========================================================

tab_dashboard, tab_modelo, tab_prediccion, tab_resultados = st.tabs(
    [
        "📊 Dashboard",
        "🤖 Evaluación del modelo",
        "🔎 Predecir estudiante",
        "📋 Resultados generales"
    ]
)


# =========================================================
# PESTAÑA 1: DASHBOARD
# =========================================================

with tab_dashboard:

    if df_filtrado.empty:
        st.warning(
            "No existen estudiantes para los filtros seleccionados."
        )

    else:
        total_estudiantes = len(df_filtrado)

        total_desercion = int(
            df_filtrado["Desercion"].sum()
        )

        total_continuidad = (
            total_estudiantes - total_desercion
        )

        porcentaje_desercion = (
            total_desercion / total_estudiantes * 100
            if total_estudiantes > 0
            else 0
        )

        promedio_general = df_filtrado["Promedio"].mean()
        asistencia_general = df_filtrado["Asistencia"].mean()

        st.subheader("Indicadores generales")

        columna1, columna2, columna3, columna4, columna5, columna6 = (
            st.columns(6)
        )

        with columna1:
            tarjeta_kpi(
                "👨‍🎓",
                total_estudiantes,
                "Total de estudiantes"
            )

        with columna2:
            tarjeta_kpi(
                "✅",
                total_continuidad,
                "Continuidad escolar"
            )

        with columna3:
            tarjeta_kpi(
                "⚠️",
                total_desercion,
                "Casos de deserción"
            )

        with columna4:
            tarjeta_kpi(
                "📉",
                f"{porcentaje_desercion:.1f} %",
                "Tasa de deserción"
            )

        with columna5:
            tarjeta_kpi(
                "📝",
                formato_decimal(promedio_general, 1),
                "Promedio académico"
            )

        with columna6:
            tarjeta_kpi(
                "📅",
                f"{asistencia_general:.1f} %",
                "Asistencia promedio"
            )

        st.write("")

        # -------------------------------------------------
        # GRÁFICO DE DESERCIÓN POR DISTRITO
        # -------------------------------------------------

        columna_grafico1, columna_grafico2 = st.columns(2)

        with columna_grafico1:

            tabla_distrito = (
                df_filtrado
                .groupby("Distrito", as_index=False)
                .agg(
                    Estudiantes=("EstudianteID", "count"),
                    Deserciones=("Desercion", "sum")
                )
            )

            tabla_distrito["TasaDesercion"] = (
                tabla_distrito["Deserciones"]
                / tabla_distrito["Estudiantes"]
                * 100
            )

            figura_distrito = px.bar(
                tabla_distrito.sort_values(
                    "TasaDesercion",
                    ascending=False
                ),
                x="Distrito",
                y="TasaDesercion",
                text_auto=".1f",
                title="Tasa de deserción por distrito",
                labels={
                    "TasaDesercion": "Tasa de deserción (%)"
                }
            )

            figura_distrito.update_traces(
                marker_color="#16745b",
                texttemplate="%{y:.1f} %",
                textposition="outside"
            )

            figura_distrito.update_layout(
                template="plotly_white",
                height=410,
                margin=dict(
                    l=30,
                    r=20,
                    t=60,
                    b=80
                ),
                xaxis_tickangle=-30
            )

            st.plotly_chart(
                figura_distrito,
                use_container_width=True
            )

        # -------------------------------------------------
        # GRÁFICO CIRCULAR
        # -------------------------------------------------

        with columna_grafico2:

            tabla_condicion = pd.DataFrame(
                {
                    "Condición": [
                        "Continuidad escolar",
                        "Deserción"
                    ],
                    "Cantidad": [
                        total_continuidad,
                        total_desercion
                    ]
                }
            )

            figura_condicion = px.pie(
                tabla_condicion,
                names="Condición",
                values="Cantidad",
                hole=0.52,
                title="Distribución de la condición estudiantil"
            )

            figura_condicion.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )

            figura_condicion.update_layout(
                template="plotly_white",
                height=410,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                ),
                legend=dict(
                    orientation="h",
                    y=-0.12,
                    x=0.5,
                    xanchor="center"
                )
            )

            st.plotly_chart(
                figura_condicion,
                use_container_width=True
            )

        # -------------------------------------------------
        # PROMEDIO Y ASISTENCIA SEGÚN CONDICIÓN
        # -------------------------------------------------

        columna_grafico3, columna_grafico4 = st.columns(2)

        with columna_grafico3:

            df_condicion = df_filtrado.copy()

            df_condicion["CondicionTexto"] = np.where(
                df_condicion["Desercion"] == 1,
                "Deserción",
                "Continuidad"
            )

            resumen_promedio = (
                df_condicion
                .groupby(
                    "CondicionTexto",
                    as_index=False
                )["Promedio"]
                .mean()
            )

            figura_promedio = px.bar(
                resumen_promedio,
                x="CondicionTexto",
                y="Promedio",
                text_auto=".1f",
                title="Promedio académico según condición",
                labels={
                    "CondicionTexto": "Condición",
                    "Promedio": "Promedio académico"
                }
            )

            figura_promedio.update_traces(
                marker_color="#337f68",
                textposition="outside"
            )

            figura_promedio.update_layout(
                template="plotly_white",
                height=390,
                yaxis_range=[0, 20]
            )

            st.plotly_chart(
                figura_promedio,
                use_container_width=True
            )

        with columna_grafico4:

            resumen_asistencia = (
                df_condicion
                .groupby(
                    "CondicionTexto",
                    as_index=False
                )["Asistencia"]
                .mean()
            )

            figura_asistencia = px.bar(
                resumen_asistencia,
                x="CondicionTexto",
                y="Asistencia",
                text_auto=".1f",
                title="Asistencia según condición",
                labels={
                    "CondicionTexto": "Condición",
                    "Asistencia": "Asistencia promedio (%)"
                }
            )

            figura_asistencia.update_traces(
                marker_color="#498f7b",
                texttemplate="%{y:.1f} %",
                textposition="outside"
            )

            figura_asistencia.update_layout(
                template="plotly_white",
                height=390,
                yaxis_range=[0, 100]
            )

            st.plotly_chart(
                figura_asistencia,
                use_container_width=True
            )


# =========================================================
# PESTAÑA 2: EVALUACIÓN DEL MODELO
# =========================================================

with tab_modelo:

    st.subheader("Evaluación del modelo Random Forest")

    st.info(
        "El modelo fue entrenado con el 75 % de los registros y "
        "evaluado con el 25 % restante. Debido a que el dataset "
        "solo contiene 50 estudiantes, las métricas pueden variar "
        "al aumentar la cantidad de datos."
    )

    columna1, columna2, columna3, columna4 = st.columns(4)

    with columna1:
        tarjeta_metrica(
            metricas["accuracy"],
            "Accuracy"
        )

    with columna2:
        tarjeta_metrica(
            metricas["precision"],
            "Precision"
        )

    with columna3:
        tarjeta_metrica(
            metricas["recall"],
            "Recall"
        )

    with columna4:
        tarjeta_metrica(
            metricas["f1"],
            "F1-Score"
        )

    st.write("")

    columna_matriz, columna_importancia = st.columns(2)

    # -----------------------------------------------------
    # MATRIZ DE CONFUSIÓN
    # -----------------------------------------------------

    with columna_matriz:

        figura_matriz = go.Figure(
            data=go.Heatmap(
                z=matriz_confusion,
                x=[
                    "Predicción: continuidad",
                    "Predicción: deserción"
                ],
                y=[
                    "Real: continuidad",
                    "Real: deserción"
                ],
                text=matriz_confusion,
                texttemplate="%{text}",
                textfont={"size": 20},
                colorscale="Greens",
                showscale=False
            )
        )

        figura_matriz.update_layout(
            title="Matriz de confusión",
            template="plotly_white",
            height=430,
            xaxis_title="Clase predicha",
            yaxis_title="Clase real"
        )

        st.plotly_chart(
            figura_matriz,
            use_container_width=True
        )

        verdaderos_negativos = matriz_confusion[0, 0]
        falsos_positivos = matriz_confusion[0, 1]
        falsos_negativos = matriz_confusion[1, 0]
        verdaderos_positivos = matriz_confusion[1, 1]

        st.markdown(
            f"""
            **Interpretación:**

            - Verdaderos negativos: **{verdaderos_negativos}**
            - Falsos positivos: **{falsos_positivos}**
            - Falsos negativos: **{falsos_negativos}**
            - Verdaderos positivos: **{verdaderos_positivos}**
            """
        )

    # -----------------------------------------------------
    # IMPORTANCIA DE VARIABLES
    # -----------------------------------------------------

    with columna_importancia:

        importancia_variables = obtener_importancia_variables(
            modelo_ml
        )

        figura_importancia = px.bar(
            importancia_variables.sort_values(
                "Importancia",
                ascending=True
            ),
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Variables más importantes para la predicción",
            labels={
                "Importancia": "Nivel de importancia",
                "Variable": "Variable"
            }
        )

        figura_importancia.update_traces(
            marker_color="#16745b"
        )

        figura_importancia.update_layout(
            template="plotly_white",
            height=500,
            margin=dict(
                l=30,
                r=20,
                t=60,
                b=40
            )
        )

        st.plotly_chart(
            figura_importancia,
            use_container_width=True
        )

    st.subheader("Resultados del conjunto de prueba")

    tabla_prueba = resultados_prueba.copy()

    tabla_prueba["ValorReal"] = tabla_prueba[
        "ValorReal"
    ].map(
        {
            0: "Continuidad",
            1: "Deserción"
        }
    )

    tabla_prueba["Prediccion"] = tabla_prueba[
        "Prediccion"
    ].map(
        {
            0: "Continuidad",
            1: "Deserción"
        }
    )

    tabla_prueba["ProbabilidadDesercion"] = (
        tabla_prueba["ProbabilidadDesercion"] * 100
    ).round(2)

    st.dataframe(
        tabla_prueba,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PESTAÑA 3: PREDICCIÓN INDIVIDUAL
# =========================================================

with tab_prediccion:

    st.subheader("Evaluación de un nuevo estudiante")

    st.write(
        "Ingrese los datos académicos y familiares del estudiante. "
        "El sistema calculará su probabilidad de riesgo."
    )

    with st.form("formulario_prediccion"):

        columna1, columna2, columna3 = st.columns(3)

        with columna1:

            nuevo_distrito = st.selectbox(
                "Distrito",
                options=sorted(
                    df_original["Distrito"].unique()
                )
            )

            nuevo_grado = st.selectbox(
                "Grado",
                options=[
                    "1ro",
                    "2do",
                    "3ro",
                    "4to",
                    "5to"
                ]
            )

            nuevo_sexo = st.selectbox(
                "Sexo",
                options=[
                    "Masculino",
                    "Femenino"
                ]
            )

            nueva_edad = st.number_input(
                "Edad",
                min_value=11,
                max_value=19,
                value=14,
                step=1
            )

        with columna2:

            nuevo_promedio = st.number_input(
                "Promedio académico",
                min_value=0.0,
                max_value=20.0,
                value=13.0,
                step=0.1
            )

            nueva_asistencia = st.number_input(
                "Asistencia (%)",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=0.1
            )

            nuevas_tardanzas = st.number_input(
                "Número de tardanzas",
                min_value=0,
                max_value=50,
                value=5,
                step=1
            )

            nuevos_cursos_desaprobados = st.number_input(
                "Cursos desaprobados",
                min_value=0,
                max_value=12,
                value=1,
                step=1
            )

        with columna3:

            nuevas_repitencias = st.number_input(
                "Repitencias",
                min_value=0,
                max_value=5,
                value=0,
                step=1
            )

            nuevo_ingreso = st.number_input(
                "Ingreso familiar mensual (S/)",
                min_value=0,
                max_value=20000,
                value=2000,
                step=100
            )

            nuevo_apoyo = st.slider(
                "Nivel de apoyo familiar",
                min_value=1,
                max_value=5,
                value=3,
                help=(
                    "1 representa un apoyo muy bajo y "
                    "5 un apoyo familiar muy alto."
                )
            )

            nueva_beca = st.selectbox(
                "¿Recibe beca?",
                options=[
                    "Sí",
                    "No"
                ]
            )

        boton_predecir = st.form_submit_button(
            "Analizar riesgo del estudiante"
        )

    if boton_predecir:

        nuevo_estudiante = pd.DataFrame(
            [
                {
                    "Distrito": nuevo_distrito,
                    "Grado": nuevo_grado,
                    "Sexo": nuevo_sexo,
                    "Edad": nueva_edad,
                    "Promedio": nuevo_promedio,
                    "Asistencia": nueva_asistencia,
                    "Tardanzas": nuevas_tardanzas,
                    "CursosDesaprobados": (
                        nuevos_cursos_desaprobados
                    ),
                    "Repitencias": nuevas_repitencias,
                    "IngresoFamiliar": nuevo_ingreso,
                    "ApoyoFamiliar": nuevo_apoyo,
                    "RecibeBeca": nueva_beca
                }
            ]
        )

        probabilidad = modelo_ml.predict_proba(
            nuevo_estudiante
        )[0, 1]

        prediccion = modelo_ml.predict(
            nuevo_estudiante
        )[0]

        nivel_riesgo = obtener_nivel_riesgo(
            probabilidad
        )

        st.write("")

        columna_resultado1, columna_resultado2 = st.columns(
            [1, 1.5]
        )

        with columna_resultado1:

            figura_indicador = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probabilidad * 100,
                    number={
                        "suffix": " %",
                        "valueformat": ".1f"
                    },
                    title={
                        "text": "Probabilidad de deserción"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "bar": {
                            "color": "#16745b"
                        },
                        "steps": [
                            {
                                "range": [0, 35],
                                "color": "#dff5e8"
                            },
                            {
                                "range": [35, 70],
                                "color": "#fff2cc"
                            },
                            {
                                "range": [70, 100],
                                "color": "#fde2e2"
                            }
                        ],
                        "threshold": {
                            "line": {
                                "color": "#8b1e1e",
                                "width": 4
                            },
                            "thickness": 0.75,
                            "value": probabilidad * 100
                        }
                    }
                )
            )

            figura_indicador.update_layout(
                height=360,
                margin=dict(
                    l=30,
                    r=30,
                    t=70,
                    b=20
                )
            )

            st.plotly_chart(
                figura_indicador,
                use_container_width=True
            )

        with columna_resultado2:

            if nivel_riesgo == "Bajo":
                clase_css = "riesgo-bajo"

            elif nivel_riesgo == "Medio":
                clase_css = "riesgo-medio"

            else:
                clase_css = "riesgo-alto"

            condicion_predicha = (
                "Posible deserción"
                if prediccion == 1
                else "Continuidad probable"
            )

            st.markdown(
                f"""
                <div class="{clase_css}">
                    Nivel de riesgo: {nivel_riesgo}<br><br>
                    Resultado del modelo: {condicion_predicha}<br><br>
                    Probabilidad estimada:
                    {probabilidad * 100:.2f} %
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.info(
                obtener_descripcion_riesgo(
                    probabilidad
                )
            )

            st.warning(
                "La predicción es un apoyo para la toma de decisiones. "
                "No reemplaza la evaluación del tutor, psicólogo, "
                "docente o responsable académico."
            )


# =========================================================
# PESTAÑA 4: RESULTADOS GENERALES
# =========================================================

with tab_resultados:

    st.subheader("Predicciones para los estudiantes registrados")

    resultados_generales = generar_predicciones_masivas(
        modelo_ml,
        df_original
    )

    filtro_nivel = st.multiselect(
        "Filtrar por nivel de riesgo",
        options=[
            "Bajo",
            "Medio",
            "Alto"
        ],
        default=[
            "Bajo",
            "Medio",
            "Alto"
        ]
    )

    tabla_filtrada = resultados_generales[
        resultados_generales["NivelRiesgo"].isin(
            filtro_nivel
        )
    ].copy()

    total_bajo = (
        resultados_generales["NivelRiesgo"] == "Bajo"
    ).sum()

    total_medio = (
        resultados_generales["NivelRiesgo"] == "Medio"
    ).sum()

    total_alto = (
        resultados_generales["NivelRiesgo"] == "Alto"
    ).sum()

    columna1, columna2, columna3 = st.columns(3)

    with columna1:
        tarjeta_kpi(
            "🟢",
            int(total_bajo),
            "Riesgo bajo"
        )

    with columna2:
        tarjeta_kpi(
            "🟡",
            int(total_medio),
            "Riesgo medio"
        )

    with columna3:
        tarjeta_kpi(
            "🔴",
            int(total_alto),
            "Riesgo alto"
        )

    st.write("")

    columnas_mostrar = [
        "EstudianteID",
        "Distrito",
        "Grado",
        "Sexo",
        "Promedio",
        "Asistencia",
        "Tardanzas",
        "CursosDesaprobados",
        "Repitencias",
        "ProbabilidadDesercion",
        "PrediccionML",
        "NivelRiesgo"
    ]

    tabla_filtrada = tabla_filtrada[
        columnas_mostrar
    ].sort_values(
        "ProbabilidadDesercion",
        ascending=False
    )

    st.dataframe(
        tabla_filtrada,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ProbabilidadDesercion": st.column_config.ProgressColumn(
                "Probabilidad de deserción",
                help="Probabilidad calculada por Random Forest",
                min_value=0,
                max_value=100,
                format="%.2f %%"
            )
        }
    )

    archivo_descarga = resultados_generales.to_csv(
        index=False,
        encoding="utf-8-sig"
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Descargar resultados en CSV",
        data=archivo_descarga,
        file_name="predicciones_desercion.csv",
        mime="text/csv"
    )
