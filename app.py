import streamlit as st
import pandas as pd
import numpy as np
from visualization_engine import VisualizationEngine
from documentation import DocumentationSection
from ml_imputation import show_ml_imputation_redirection
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Dashboard de Visualización Interactiva",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = pd.DataFrame()

def load_data():
    """Handle data loading with various file formats and configurations"""
    st.sidebar.header("📁 Cargar Datos")
    
    uploaded_file = st.sidebar.file_uploader(
        "Selecciona un archivo",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos soportados: CSV, Excel"
    )
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # CSV configuration options
                st.sidebar.subheader("⚙️ Configuración CSV")
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    separator = st.selectbox("Separador", [',', ';', '\t', '|'], key="csv_sep")
                with col2:
                    encoding = st.selectbox("Codificación", ['utf-8', 'latin1', 'cp1252'], key="csv_enc")
                
                decimal = st.sidebar.selectbox("Separador decimal", ['.', ','], key="csv_dec")
                
                # Load CSV
                data = pd.read_csv(
                    uploaded_file,
                    sep=separator,
                    encoding=encoding,
                    decimal=decimal
                )
                
            else:  # Excel files
                # Excel configuration options
                st.sidebar.subheader("⚙️ Configuración Excel")
                
                # Get sheet names
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                
                selected_sheet = st.sidebar.selectbox("Hoja", sheet_names, key="excel_sheet")
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    header_row = st.number_input("Fila de encabezados", min_value=0, value=0, key="excel_header")
                with col2:
                    skip_rows = st.number_input("Filas a omitir", min_value=0, value=0, key="excel_skip")
                
                # Load Excel
                data = pd.read_excel(
                    uploaded_file,
                    sheet_name=selected_sheet,
                    header=header_row,
                    skiprows=skip_rows
                )
            
            # Clean column names
            data.columns = data.columns.astype(str)
            
            # Store in session state
            st.session_state.data = data
            st.session_state.filtered_data = data.copy()
            
            st.sidebar.success(f"✅ Archivo cargado: {data.shape[0]} filas, {data.shape[1]} columnas")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar archivo: {str(e)}")
            return pd.DataFrame()
    
    return st.session_state.data

def apply_filters(data):
    """Apply dynamic filters to the data"""
    if len(data) == 0:
        return data
    
    st.sidebar.header("🔍 Filtros")
    
    filtered_data = data.copy()
    
    # Numeric filters
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        st.sidebar.subheader("📊 Filtros Numéricos")
        
        for col in numeric_columns:
            if col in data.columns:
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                
                if min_val != max_val:  # Only show slider if there's a range
                    selected_range = st.sidebar.slider(
                        f"{col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"filter_{col}"
                    )
                    
                    filtered_data = filtered_data[
                        (filtered_data[col] >= selected_range[0]) & 
                        (filtered_data[col] <= selected_range[1])
                    ]
    
    # Categorical filters
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        st.sidebar.subheader("📂 Filtros Categóricos")
        
        for col in categorical_columns:
            if col in data.columns:
                unique_values = data[col].dropna().unique().tolist()
                
                if len(unique_values) <= 50:  # Only show filter for reasonable number of categories
                    selected_values = st.sidebar.multiselect(
                        f"{col}",
                        options=unique_values,
                        default=unique_values,
                        key=f"filter_cat_{col}"
                    )
                    
                    if selected_values:
                        filtered_data = filtered_data[filtered_data[col].isin(selected_values)]
    
    # Update session state
    st.session_state.filtered_data = filtered_data
    
    # Show filter results
    if len(filtered_data) != len(data):
        st.sidebar.info(f"📋 Datos filtrados: {len(filtered_data)} de {len(data)} filas")
    
    return filtered_data

def show_data_overview(data):
    """Display comprehensive data overview"""
    st.header("📋 Resumen de Datos")
    
    if len(data) == 0:
        st.warning("No hay datos para mostrar. Por favor, carga un archivo.")
        return
    
    # Basic information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Filas", f"{len(data):,}")
    with col2:
        st.metric("Total de Columnas", len(data.columns))
    with col3:
        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
        st.metric("Datos Faltantes", f"{missing_percentage:.1f}%")
    with col4:
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        st.metric("Columnas Numéricas", numeric_cols)
    
    # Data types and missing values
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Tipos de Datos")
        dtype_info = pd.DataFrame({
            'Columna': data.columns,
            'Tipo': data.dtypes.astype(str),
            'Valores Únicos': [data[col].nunique() for col in data.columns],
            'Valores Faltantes': data.isnull().sum().values
        })
        st.dataframe(dtype_info, use_container_width=True)
    
    with col2:
        st.subheader("❌ Valores Faltantes")
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            missing_df = pd.DataFrame({
                'Columna': missing_data.index,
                'Faltantes': missing_data.values,
                'Porcentaje': (missing_data.values / len(data) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("🎉 ¡No hay valores faltantes!")
    
    # Statistical summary
    st.subheader("📈 Resumen Estadístico")
    
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 0:
        st.dataframe(numeric_data.describe(), use_container_width=True)
    else:
        st.info("No hay columnas numéricas para mostrar estadísticas.")
    
    # Data preview
    st.subheader("👁️ Vista Previa de Datos")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        n_rows = st.number_input("Número de filas", min_value=5, max_value=min(100, len(data)), value=10)
    
    st.dataframe(data.head(n_rows), use_container_width=True)

def show_visualizations(data):
    """Display visualization interface"""
    st.header("📊 Visualizaciones Interactivas")
    
    if len(data) == 0:
        st.warning("No hay datos para visualizar. Por favor, carga un archivo.")
        return
    
    # Create visualization engine
    viz_engine = VisualizationEngine(data)
    
    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5, viz_tab6 = st.tabs([
        "🎯 Scatter Plot", "📈 Line Chart", "📊 Bar Chart", 
        "📊 Histogram", "📦 Box Plot", "🔥 Correlation Heatmap"
    ])
    
    with viz_tab1:
        viz_engine.create_scatter_plot()
    
    with viz_tab2:
        viz_engine.create_line_chart()
    
    with viz_tab3:
        viz_engine.create_bar_chart()
    
    with viz_tab4:
        viz_engine.create_histogram()
    
    with viz_tab5:
        viz_engine.create_box_plot()
    
    with viz_tab6:
        viz_engine.create_correlation_heatmap()

def show_analysis_tools(data):
    """Advanced analysis tools"""
    st.header("🔍 Herramientas de Análisis")
    
    if len(data) == 0:
        st.warning("No hay datos para analizar. Por favor, carga un archivo.")
        return
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "📊 Comparación de Variables", "📈 Análisis de Tendencias", "🎯 Análisis de Distribución"
    ])
    
    with analysis_tab1:
        show_variable_comparison(data)
    
    with analysis_tab2:
        show_trend_analysis(data)
    
    with analysis_tab3:
        show_distribution_analysis(data)

def show_variable_comparison(data):
    """Variable comparison analysis"""
    st.subheader("📊 Comparación entre Variables")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.warning("Se necesitan al menos 2 variables numéricas para comparación.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox("Variable 1", numeric_columns, key="comp_var1")
    with col2:
        var2 = st.selectbox("Variable 2", [col for col in numeric_columns if col != var1], key="comp_var2")
    
    if var1 and var2:
        # Statistical comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"Correlación", f"{data[var1].corr(data[var2]):.3f}")
        with col2:
            st.metric(f"Media {var1}", f"{data[var1].mean():.2f}")
        with col3:
            st.metric(f"Media {var2}", f"{data[var2].mean():.2f}")
        
        # Visualization
        fig = px.scatter(data, x=var1, y=var2, trendline="ols",
                        title=f"Relación entre {var1} y {var2}")
        st.plotly_chart(fig, use_container_width=True)

def show_trend_analysis(data):
    """Trend analysis for time series or sequential data"""
    st.subheader("📈 Análisis de Tendencias")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        st.warning("No hay variables numéricas para análisis de tendencias.")
        return
    
    selected_var = st.selectbox("Variable para analizar", numeric_columns, key="trend_var")
    
    if selected_var:
        # Calculate rolling averages
        window_size = st.slider("Ventana de promedio móvil", 3, 20, 7)
        
        trend_data = data[[selected_var]].copy()
        trend_data['Rolling_Mean'] = trend_data[selected_var].rolling(window=window_size).mean()
        
        # Plot trend
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=trend_data[selected_var],
            mode='lines',
            name=selected_var,
            line=dict(color='lightblue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            y=trend_data['Rolling_Mean'],
            mode='lines',
            name=f'Promedio Móvil ({window_size})',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"Análisis de Tendencia: {selected_var}",
            xaxis_title="Índice",
            yaxis_title=selected_var,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_distribution_analysis(data):
    """Distribution analysis with statistical tests"""
    st.subheader("🎯 Análisis de Distribución")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        st.warning("No hay variables numéricas para análisis de distribución.")
        return
    
    selected_var = st.selectbox("Variable para analizar", numeric_columns, key="dist_var")
    
    if selected_var:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            fig = px.histogram(data, x=selected_var, nbins=30, 
                             title=f"Distribución de {selected_var}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistical measures
            st.write("**Medidas Estadísticas:**")
            
            stats_data = {
                'Media': data[selected_var].mean(),
                'Mediana': data[selected_var].median(),
                'Moda': data[selected_var].mode().iloc[0] if len(data[selected_var].mode()) > 0 else 'N/A',
                'Desviación Estándar': data[selected_var].std(),
                'Varianza': data[selected_var].var(),
                'Asimetría': data[selected_var].skew(),
                'Curtosis': data[selected_var].kurtosis()
            }
            
            for stat, value in stats_data.items():
                if isinstance(value, (int, float)):
                    st.metric(stat, f"{value:.3f}")
                else:
                    st.metric(stat, str(value))

def show_advanced_analytics(data):
    """Advanced analytics with regression and statistical tests"""
    st.header("🧮 Análisis Avanzado")
    
    if len(data) == 0:
        st.warning("No hay datos para análisis avanzado. Por favor, carga un archivo.")
        return
    
    advanced_tab1, advanced_tab2, advanced_tab3 = st.tabs([
        "📈 Regresión", "📊 Análisis Estadístico", "🎯 Análisis Predictivo"
    ])
    
    with advanced_tab1:
        show_regression_analysis(data)
    
    with advanced_tab2:
        show_statistical_analysis(data)
    
    with advanced_tab3:
        show_predictive_analysis(data)

def show_regression_analysis(data):
    """Regression analysis interface"""
    st.subheader("📈 Análisis de Regresión")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.warning("Se necesitan al menos 2 variables numéricas para regresión.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        y_var = st.selectbox("Variable dependiente (Y)", numeric_columns, key="reg_y")
    with col2:
        x_var = st.selectbox("Variable independiente (X)", 
                           [col for col in numeric_columns if col != y_var], key="reg_x")
    with col3:
        reg_type = st.selectbox("Tipo de regresión", ["Lineal", "Polinomial"], key="reg_type")
    
    if y_var and x_var:
        # Prepare data
        clean_data = data[[x_var, y_var]].dropna()
        
        if len(clean_data) < 3:
            st.warning("No hay suficientes datos válidos para regresión.")
            return
        
        X = clean_data[x_var].values
        y = clean_data[y_var].values
        
        # Perform regression
        if reg_type == "Lineal":
            # Linear regression
            coeffs = np.polyfit(X, y, 1)
            y_pred = np.polyval(coeffs, X)
            
            # Calculate R²
            r_squared = np.corrcoef(y, y_pred)[0, 1] ** 2
            
            st.metric("R² Score", f"{r_squared:.4f}")
            st.write(f"**Ecuación:** y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}")
            
        else:  # Polynomial
            degree = st.slider("Grado del polinomio", 2, 5, 2)
            coeffs = np.polyfit(X, y, degree)
            y_pred = np.polyval(coeffs, X)
            
            # Calculate R²
            r_squared = np.corrcoef(y, y_pred)[0, 1] ** 2
            st.metric("R² Score", f"{r_squared:.4f}")
        
        # Plot
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=X, y=y,
            mode='markers',
            name='Datos originales',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        # Regression line
        X_sorted = np.sort(X)
        y_pred_sorted = np.polyval(coeffs, X_sorted)
        
        fig.add_trace(go.Scatter(
            x=X_sorted, y=y_pred_sorted,
            mode='lines',
            name=f'Regresión {reg_type.lower()}',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"Regresión {reg_type}: {y_var} vs {x_var}",
            xaxis_title=x_var,
            yaxis_title=y_var,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_statistical_analysis(data):
    """Statistical tests and analysis"""
    st.subheader("📊 Análisis Estadístico")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        st.warning("No hay variables numéricas para análisis estadístico.")
        return
    
    test_type = st.selectbox("Tipo de análisis", 
                           ["Normalidad", "Correlaciones", "Descriptivos"], 
                           key="stat_test")
    
    if test_type == "Normalidad":
        selected_var = st.selectbox("Variable", numeric_columns, key="norm_var")
        
        if selected_var:
            from scipy import stats
            
            # Shapiro-Wilk test (for small samples)
            if len(data[selected_var].dropna()) <= 5000:
                stat, p_value = stats.shapiro(data[selected_var].dropna())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Estadístico W", f"{stat:.4f}")
                with col2:
                    st.metric("p-valor", f"{p_value:.4f}")
                
                if p_value > 0.05:
                    st.success("✅ Los datos parecen seguir una distribución normal (p > 0.05)")
                else:
                    st.warning("⚠️ Los datos no siguen una distribución normal (p ≤ 0.05)")
            else:
                st.info("Muestra muy grande para test de Shapiro-Wilk. Usa visualización de distribución.")
    
    elif test_type == "Correlaciones":
        if len(numeric_columns) >= 2:
            corr_matrix = data[numeric_columns].corr()
            
            # Show correlation matrix
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                          title="Matriz de Correlaciones")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Se necesitan al menos 2 variables numéricas.")
    
    else:  # Descriptivos
        st.write("**Estadísticos Descriptivos Detallados:**")
        
        selected_vars = st.multiselect("Seleccionar variables", numeric_columns, 
                                     default=numeric_columns[:5], key="desc_vars")
        
        if selected_vars:
            desc_stats = data[selected_vars].describe()
            st.dataframe(desc_stats, use_container_width=True)

def show_predictive_analysis(data):
    """Simple predictive analysis"""
    st.subheader("🎯 Análisis Predictivo Básico")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.warning("Se necesitan al menos 2 variables numéricas para análisis predictivo.")
        return
    
    st.info("🚀 **Funcionalidad Avanzada Disponible**: Para análisis predictivo más sofisticado "
           "y machine learning avanzado, visita nuestra herramienta especializada en la pestaña "
           "'🤖 ML Missing Data'.")
    
    # Simple prediction example
    col1, col2 = st.columns(2)
    
    with col1:
        target_var = st.selectbox("Variable objetivo", numeric_columns, key="pred_target")
    with col2:
        feature_var = st.selectbox("Variable predictora", 
                                 [col for col in numeric_columns if col != target_var], 
                                 key="pred_feature")
    
    if target_var and feature_var:
        # Simple linear prediction
        clean_data = data[[feature_var, target_var]].dropna()
        
        if len(clean_data) >= 10:
            # Split data
            split_point = int(len(clean_data) * 0.8)
            train_data = clean_data.iloc[:split_point]
            test_data = clean_data.iloc[split_point:]
            
            # Simple linear model
            coeffs = np.polyfit(train_data[feature_var], train_data[target_var], 1)
            
            # Predictions
            train_pred = np.polyval(coeffs, train_data[feature_var])
            test_pred = np.polyval(coeffs, test_data[feature_var])
            
            # Calculate metrics
            train_r2 = np.corrcoef(train_data[target_var], train_pred)[0, 1] ** 2
            test_r2 = np.corrcoef(test_data[target_var], test_pred)[0, 1] ** 2
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Entrenamiento", f"{train_r2:.4f}")
            with col2:
                st.metric("R² Validación", f"{test_r2:.4f}")
            
            # Plot predictions
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=train_data[feature_var], y=train_data[target_var],
                mode='markers', name='Entrenamiento',
                marker=dict(color='blue', opacity=0.6)
            ))
            
            fig.add_trace(go.Scatter(
                x=test_data[feature_var], y=test_data[target_var],
                mode='markers', name='Validación',
                marker=dict(color='red', opacity=0.6)
            ))
            
            # Prediction line
            X_range = np.linspace(clean_data[feature_var].min(), clean_data[feature_var].max(), 100)
            y_range = np.polyval(coeffs, X_range)
            
            fig.add_trace(go.Scatter(
                x=X_range, y=y_range,
                mode='lines', name='Predicción',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title=f"Predicción: {target_var} vs {feature_var}",
                xaxis_title=feature_var,
                yaxis_title=target_var,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para análisis predictivo.")

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Dashboard de Visualización Interactiva</h1>
        <p>Herramienta completa para análisis y visualización de datos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load and filter data
    data = load_data()
    filtered_data = apply_filters(data)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Data Overview", 
        "📊 Visualizations", 
        "🔍 Analysis Tools", 
        "🧮 Advanced Analytics",
        "🤖 ML Missing Data",
        "📚 Documentation"
    ])
    
    with tab1:
        show_data_overview(filtered_data)
    
    with tab2:
        show_visualizations(filtered_data)
    
    with tab3:
        show_analysis_tools(filtered_data)
    
    with tab4:
        show_advanced_analytics(filtered_data)
    
    with tab5:
        show_ml_imputation_redirection(filtered_data)
    
    with tab6:
        DocumentationSection.show_documentation()

if __name__ == "__main__":
    main()
