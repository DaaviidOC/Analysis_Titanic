import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import io
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import gaussian_kde

#CONFIGURACIÓN
st.set_page_config(page_title="Análisis del Titanic", layout="wide", page_icon="⛵")
 
#FUNCIONES Y CLASES
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data
 
def create_plot(data, columns, plot_type):
    plot_funtions =  {
        "Histograma": lambda: px.histogram(data.melt(id_vars=columns, var_name='variable'), nbins=30, x='value', color='variable', facet_col='variable', facet_col_wrap=2),
        "Diagrama de caja": lambda: go.Figure([go.Box(y=data[col], name=col) for col in columns]),
        "Diagrama de violín": lambda: go.Figure([go.Violin(y=data[col], name=col, box_visible=True) for col in columns]),
        "Gráfico de dispersión": lambda: px.scatter_matrix(data, dimensions=columns),
        "Gráfico de barras": lambda: px.bar(data, x=data.index, y=columns),
        "Gráfico de línea": lambda: px.line(data, x=data.index, y=columns),
        "Gráfico KDE": lambda: kde_plot(data, columns),

    }
    fig = plot_funtions[plot_type]()
    return fig

def kde_plot(data, columns):
    if len(columns) == 1:
        # Usar un histograma con una estimación KDE para una columna
        return px.histogram(data, x=columns[0], marginal="violin", nbins=30)
    elif len(columns) == 2:
        # Usar un gráfico de densidad de contorno para dos columnas
        return px.density_contour(data, x=columns[0], y=columns[1])
    else:
        st.warning("Selecciona una o dos columnas para el gráfico KDE.")
        return None

def plotly_age_histogram(data):
    fig = px.histogram(
        data_frame=data.dropna(subset=['Age']),  # Asegúrate de que no hay valores NaN en la columna 'Age'
        x="Age",
        nbins=30,
        title="Distribución de Edades en el Titanic",
        labels={"Age": "Edad"},  # Etiqueta personalizada para el eje X
    )
    
    # Configuración del título
    fig.update_layout(
        title_text='Distribución de Edades en el Titanic',
        title_font=dict(size=20, color='white'),  # Ajusta el tamaño y el color del título
        paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        plot_bgcolor='rgba(0,0,0,0)'   # Fondo transparente
    )
    
    # Configuración de los ejes
    fig.update_xaxes(title_text='Edad', title_font=dict(color='white'), tickfont=dict(color='white'))
    fig.update_yaxes(title_text='Número de Pasajeros', title_font=dict(size=13, color='white'), tickfont=dict(color='white'))
    
    # Configuración de la cuadrícula
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
        
    return fig

# Función para calcular la densidad de KDE
def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Calcula la densidad de probabilidad KDE usando scipy."""
    kde = gaussian_kde(x, bw_method=bandwidth, **kwargs)
    return kde.evaluate(x_grid)

# Función para crear el gráfico KDE con Plotly
def create_kde_plot(data, age_column='Age', survived_column='Survived'):
    # Crear un rango de valores para el eje x (Edad)
    x_grid = np.linspace(data[age_column].min(), data[age_column].max(), 100)
    
    # Calcular las densidades KDE para sobrevivientes y no sobrevivientes
    kde_survived = kde_scipy(data.loc[data[survived_column] == 1, age_column].dropna(), x_grid)
    kde_not_survived = kde_scipy(data.loc[data[survived_column] == 0, age_column].dropna(), x_grid)
    
    # Crear la figura con las dos densidades
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_grid, y=kde_survived, fill='tozeroy', name='Sobrevivientes'))
    fig.add_trace(go.Scatter(x=x_grid, y=kde_not_survived, fill='tozeroy', name='No Sobrevivientes'))
    
    # Actualizar el diseño del gráfico
    fig.update_layout(
        title='Distribución de Edades entre Sobrevivientes y No Sobrevivientes en el Titanic',
        xaxis_title='Edad',
        yaxis_title='Densidad',
        legend_title_text='Estado'
    )
    
    return fig

        
# Carga de datos
st.markdown("<h1 style='text-align: center;'>Análisis del Titanic</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Carga el dataset del Titanic", type="csv")

st.markdown("<br>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Imagen del Titanic 
    st.write("<div style='text-align: center;'><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/800px-RMS_Titanic_3.jpg' width='900' /></div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    data = load_data(uploaded_file)
    # Datos originales 
    st.markdown("<h1 style='text-align: center;'>Datos originales</h1>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        .centered-table {
            margin-left: auto;
            margin-right: auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Centrar la tabla de datos.head() usando pandas Styler
    st.write(data.head().to_html(classes='centered-table'), unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Análisis exploratorio
    # st.subheader("Análisis Exploratorio de Datos")
    st.markdown("<h1 style='text-align: center;'>Análisis Exploratorio de Datos</h1>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Descripción de las variables numéricas")
        st.write(data.describe())
        
    with col2:
        st.write("Descripción de las variables categóricas")
        st.write(data.select_dtypes(include="object").describe())
    
    with col3:
        st.write("Correlación")
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        st.write(corr_matrix)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col4, col5 = st.columns(2)
     
    with col4: 
        # Heatmap
        st.write("Heatmap")
        corr_matrix = numeric_data.corr()
        fig_corr = px.imshow(corr_matrix, x = corr_matrix.columns, y=corr_matrix.columns)
        st.plotly_chart(fig_corr)
    
    with col5:    
        # Diagrama de violín
        st.write("Diagrama de violín")
        if 'Survived' in data.columns and 'Age' in data.columns:
            survival_age_fig = px.violin(data, y='Age', color='Survived', box=True, points="all",
                                        hover_data=data.columns)
            st.plotly_chart(survival_age_fig)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col6, col7 = st.columns(2)
    with col6:
        if 'Age' in data.columns:
            age_histogram_fig = plotly_age_histogram(data)
            st.plotly_chart(age_histogram_fig)
    
    with col7:
        age_kde_fig = create_kde_plot(data)
        st.plotly_chart(age_kde_fig)
    
    st.markdown("<br>", unsafe_allow_html=True)
      
    columns = st.multiselect("Selecciona las columnas a graficar", data.columns)
    plot_types = st.multiselect("Selecciona los tipos de gráficos", [
    "Histograma", "Diagrama de caja", "Diagrama de violín", 
    "Gráfico de dispersión", "Gráfico de barras", 
    "Gráfico de línea", "Gráfico KDE" ])
 
    if len(columns) > 0 and len(plot_types) > 0:
        for plot_type in plot_types:
            fig = create_plot(data, columns, plot_type)
            st.plotly_chart(fig)
