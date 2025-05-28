import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import io
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from data_processor import DataProcessor
from visualization_engine import VisualizationEngine
from documentation import DocumentationSection
from ml_imputation import show_ml_imputation_interface

# Page configuration
st.set_page_config(
    page_title="Interactive Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'columns_info' not in st.session_state:
    st.session_state.columns_info = {}

def main():
    st.title("📊 Interactive Data Visualization Dashboard")
    st.markdown("---")
    
    # Sidebar for data import and settings
    with st.sidebar:
        st.header("📁 Data Import")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV, Excel (.xlsx), or Excel (.xls) files"
        )
        
        if uploaded_file is not None:
            # Initialize data processor
            processor = DataProcessor()
            
            # File format detection
            file_extension = uploaded_file.name.split('.')[-1].lower()
            st.success(f"Detected file format: {file_extension.upper()}")
            
            # Import settings based on file type
            if file_extension == 'csv':
                st.subheader("CSV Import Settings")
                separator = st.selectbox(
                    "Separator",
                    [',', ';', '\t', '|'],
                    help="Choose the column separator used in your CSV file"
                )
                encoding = st.selectbox(
                    "Encoding",
                    ['utf-8', 'latin-1', 'cp1252'],
                    help="Choose the character encoding of your file"
                )
                
                # Process CSV file
                try:
                    data = processor.load_csv(uploaded_file, separator, encoding)
                    st.session_state.data = data
                    st.success(f"✅ File loaded successfully! Shape: {data.shape}")
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    return
                    
            elif file_extension in ['xlsx', 'xls']:
                st.subheader("Excel Import Settings")
                
                # Try to get sheet names
                try:
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names
                    
                    selected_sheet = st.selectbox(
                        "Select Sheet",
                        sheet_names,
                        help="Choose which sheet to import"
                    )
                    
                    # Process Excel file
                    data = processor.load_excel(uploaded_file, selected_sheet)
                    st.session_state.data = data
                    st.success(f"✅ File loaded successfully! Shape: {data.shape}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    return
        
        # Data filtering options (only show if data is loaded)
        if st.session_state.data is not None:
            st.markdown("---")
            st.header("🔍 Data Filtering")
            
            # Column selection for filtering
            numeric_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
            datetime_columns = st.session_state.data.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Numeric filters
            if numeric_columns:
                st.subheader("Numeric Filters")
                for col in numeric_columns:
                    min_val = float(st.session_state.data[col].min())
                    max_val = float(st.session_state.data[col].max())
                    
                    selected_range = st.slider(
                        f"{col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"filter_{col}"
                    )
            
            # Categorical filters
            if categorical_columns:
                st.subheader("Categorical Filters")
                for col in categorical_columns:
                    unique_values = st.session_state.data[col].dropna().unique().tolist()
                    if len(unique_values) <= 20:  # Only show multiselect for manageable number of options
                        selected_values = st.multiselect(
                            f"{col}",
                            options=unique_values,
                            default=unique_values,
                            key=f"filter_cat_{col}"
                        )
    
    # Main content area
    if st.session_state.data is not None:
        # Apply filters
        filtered_data = apply_filters(st.session_state.data)
        st.session_state.processed_data = filtered_data
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Data Overview", 
            "📈 Visualizations", 
            "🔍 Analysis Tools", 
            "📊 Advanced Analytics",
            "👥 GroupBy Analysis",
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
            show_groupby_analysis(filtered_data)
        
        with tab6:
            updated_data = show_ml_imputation_interface(filtered_data)
            if updated_data is not None and not updated_data.equals(filtered_data):
                st.session_state.data = updated_data
                st.session_state.processed_data = updated_data
        
        with tab7:
            DocumentationSection.show_documentation()
    
    else:
        # Welcome message when no data is loaded
        st.markdown("""
        ## Welcome to the Interactive Data Visualization Dashboard! 👋
        
        This dashboard allows you to:
        - **Import** various file formats (CSV, Excel)
        - **Visualize** your data with interactive charts
        - **Filter** and analyze data in real-time
        - **Compare** variables and identify trends
        
        ### Getting Started
        1. Upload your data file using the sidebar
        2. Configure import settings if needed
        3. Explore your data through interactive visualizations
        4. Use filters to focus on specific data segments
        
        ### Supported File Formats
        - **CSV**: Comma-separated values with customizable separators
        - **Excel**: .xlsx and .xls files with sheet selection
        
        **👈 Start by uploading a file in the sidebar!**
        """)

def apply_filters(data):
    """Apply filters based on user selections"""
    filtered_data = data.copy()
    
    # Apply numeric filters
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_columns:
        if f"filter_{col}" in st.session_state:
            min_val, max_val = st.session_state[f"filter_{col}"]
            filtered_data = filtered_data[
                (filtered_data[col] >= min_val) & (filtered_data[col] <= max_val)
            ]
    
    # Apply categorical filters
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        if f"filter_cat_{col}" in st.session_state:
            selected_values = st.session_state[f"filter_cat_{col}"]
            if selected_values:  # Only filter if values are selected
                filtered_data = filtered_data[filtered_data[col].isin(selected_values)]
    
    return filtered_data

def show_data_overview(data):
    """Display data overview and basic statistics"""
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(data):,}")
    with col2:
        st.metric("Total Columns", len(data.columns))
    with col3:
        st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
    with col4:
        st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(data.head(100), use_container_width=True)
    
    # Column information
    st.subheader("📝 Column Information")
    
    col_info = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        missing = data[col].isnull().sum()
        unique = data[col].nunique()
        
        col_info.append({
            'Column': col,
            'Data Type': dtype,
            'Missing Values': missing,
            'Unique Values': unique,
            'Missing %': f"{(missing/len(data)*100):.1f}%"
        })
    
    st.dataframe(pd.DataFrame(col_info), use_container_width=True)
    
    # Basic statistics
    if len(data.select_dtypes(include=[np.number]).columns) > 0:
        st.subheader("📈 Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)

def show_visualizations(data):
    """Display interactive visualizations"""
    st.header("📈 Interactive Visualizations")
    
    if len(data) == 0:
        st.warning("No data to display after applying filters. Please adjust your filter settings.")
        return
    
    viz_engine = VisualizationEngine(data)
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Choose Visualization Type",
        [
            "Scatter Plot",
            "Line Chart",
            "Bar Chart",
            "Histogram",
            "Box Plot",
            "Correlation Heatmap",
            "Distribution Plot",
            "Time Series"
        ]
    )
    
    if viz_type == "Scatter Plot":
        viz_engine.create_scatter_plot()
    elif viz_type == "Line Chart":
        viz_engine.create_line_chart()
    elif viz_type == "Bar Chart":
        viz_engine.create_bar_chart()
    elif viz_type == "Histogram":
        viz_engine.create_histogram()
    elif viz_type == "Box Plot":
        viz_engine.create_box_plot()
    elif viz_type == "Correlation Heatmap":
        viz_engine.create_correlation_heatmap()
    elif viz_type == "Distribution Plot":
        viz_engine.create_distribution_plot()
    elif viz_type == "Time Series":
        viz_engine.create_time_series()

def show_analysis_tools(data):
    """Display analysis tools and comparison features"""
    st.header("🔍 Analysis Tools")
    
    if len(data) == 0:
        st.warning("No data to analyze after applying filters. Please adjust your filter settings.")
        return
    
    # Variable comparison tool
    st.subheader("🔄 Variable Comparison")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_columns) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            var1 = st.selectbox("Select Variable 1", numeric_columns, key="var1")
        with col2:
            var2 = st.selectbox("Select Variable 2", [col for col in numeric_columns if col != var1], key="var2")
        
        if var1 and var2:
            # Correlation analysis
            correlation = data[var1].corr(data[var2])
            st.metric("Correlation Coefficient", f"{correlation:.3f}")
            
            # Scatter plot with trend line
            fig = px.scatter(
                data, x=var1, y=var2,
                title=f"{var1} vs {var2}",
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Trend analysis
    st.subheader("📊 Trend Analysis")
    
    if numeric_columns:
        selected_column = st.selectbox("Select column for trend analysis", numeric_columns)
        
        # Rolling average
        window_size = st.slider("Rolling average window", 1, min(50, len(data)//4), 5)
        
        if len(data) >= window_size:
            rolling_avg = data[selected_column].rolling(window=window_size).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=data[selected_column],
                mode='lines',
                name='Original',
                line=dict(color='lightblue', width=1)
            ))
            fig.add_trace(go.Scatter(
                y=rolling_avg,
                mode='lines',
                name=f'Rolling Average ({window_size})',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f"Trend Analysis: {selected_column}",
                xaxis_title="Index",
                yaxis_title=selected_column
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Group analysis
    if categorical_columns and numeric_columns:
        st.subheader("👥 Group Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            group_by = st.selectbox("Group by", categorical_columns)
        with col2:
            analyze_col = st.selectbox("Analyze column", numeric_columns)
        
        if group_by and analyze_col:
            grouped = data.groupby(group_by)[analyze_col].agg(['mean', 'median', 'std', 'count']).round(3)
            st.dataframe(grouped, use_container_width=True)
            
            # Box plot by groups
            fig = px.box(data, x=group_by, y=analyze_col, title=f"{analyze_col} by {group_by}")
            st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(data):
    """Display advanced analytics including regression analysis and variable relationships"""
    st.header("📊 Advanced Analytics")
    
    if len(data) == 0:
        st.warning("No data to analyze after applying filters. Please adjust your filter settings.")
        return
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    # Regression Analysis Section
    st.subheader("📈 Regression Analysis")
    
    if len(numeric_columns) >= 2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox("Variable Independiente (X)", numeric_columns, key="reg_x")
        with col2:
            y_var = st.selectbox("Variable Dependiente (Y)", [col for col in numeric_columns if col != x_var], key="reg_y")
        with col3:
            regression_type = st.selectbox("Tipo de Regresión", 
                                         ["Linear", "Polynomial (2nd)", "Polynomial (3rd)"], 
                                         key="reg_type")
        
        if x_var and y_var:
            # Create scatter plot with regression
            fig = go.Figure()
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=data[x_var],
                y=data[y_var],
                mode='markers',
                name='Datos',
                marker=dict(color='lightblue', size=8, opacity=0.7)
            ))
            
            # Calculate regression based on type
            x_vals = data[x_var].values
            y_vals = data[y_var].values
            
            # Remove NaN values
            mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            x_clean = x_vals[mask]
            y_clean = y_vals[mask]
            
            if len(x_clean) > 0:
                if regression_type == "Linear":
                    # Linear regression
                    coeffs = np.polyfit(x_clean, y_clean, 1)
                    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                    y_line = np.polyval(coeffs, x_line)
                    
                    # Calculate R-squared
                    y_pred = np.polyval(coeffs, x_clean)
                    r_squared = r2_score(y_clean, y_pred)
                    
                elif regression_type == "Polynomial (2nd)":
                    # Polynomial regression (degree 2)
                    coeffs = np.polyfit(x_clean, y_clean, 2)
                    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                    y_line = np.polyval(coeffs, x_line)
                    
                    y_pred = np.polyval(coeffs, x_clean)
                    r_squared = r2_score(y_clean, y_pred)
                    
                else:  # Polynomial (3rd)
                    coeffs = np.polyfit(x_clean, y_clean, 3)
                    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                    y_line = np.polyval(coeffs, x_line)
                    
                    y_pred = np.polyval(coeffs, x_clean)
                    r_squared = r2_score(y_clean, y_pred)
                
                # Add regression line
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name=f'Regresión {regression_type}',
                    line=dict(color='red', width=3)
                ))
                
                fig.update_layout(
                    title=f'{regression_type} Regression: {y_var} vs {x_var}',
                    xaxis_title=x_var,
                    yaxis_title=y_var,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display regression statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² (Coeficiente de Determinación)", f"{r_squared:.4f}")
                with col2:
                    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
                    st.metric("Correlación de Pearson", f"{correlation:.4f}")
                with col3:
                    rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
                    st.metric("RMSE", f"{rmse:.4f}")
                
                # Show equation
                if regression_type == "Linear":
                    st.info(f"Ecuación: y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}")
                elif regression_type == "Polynomial (2nd)":
                    st.info(f"Ecuación: y = {coeffs[0]:.4f}x² + {coeffs[1]:.4f}x + {coeffs[2]:.4f}")
                else:
                    st.info(f"Ecuación: y = {coeffs[0]:.4f}x³ + {coeffs[1]:.4f}x² + {coeffs[2]:.4f}x + {coeffs[3]:.4f}")
    
    # Variable Analysis Section
    st.subheader("🔍 Análisis de Variables")
    
    if numeric_columns:
        selected_var = st.selectbox("Seleccionar Variable para Análisis", numeric_columns, key="var_analysis")
        
        if selected_var:
            col1, col2 = st.columns(2)
            
            with col1:
                # Statistical summary
                st.write("**Estadísticas Descriptivas**")
                var_data = data[selected_var].dropna()
                
                stats_dict = {
                    'Media': var_data.mean(),
                    'Mediana': var_data.median(),
                    'Moda': var_data.mode().iloc[0] if len(var_data.mode()) > 0 else 'N/A',
                    'Desviación Estándar': var_data.std(),
                    'Varianza': var_data.var(),
                    'Asimetría (Skewness)': var_data.skew(),
                    'Curtosis': var_data.kurtosis(),
                    'Mínimo': var_data.min(),
                    'Máximo': var_data.max(),
                    'Rango': var_data.max() - var_data.min()
                }
                
                for stat, value in stats_dict.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        st.metric(stat, f"{value:.4f}")
                    else:
                        st.metric(stat, str(value))
            
            with col2:
                # Distribution plot
                fig = go.Figure()
                
                # Histogram
                fig.add_trace(go.Histogram(
                    x=var_data,
                    name='Distribución',
                    nbinsx=30,
                    opacity=0.7
                ))
                
                fig.update_layout(
                    title=f'Distribución de {selected_var}',
                    xaxis_title=selected_var,
                    yaxis_title='Frecuencia',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Normality tests
            st.write("**Tests de Normalidad**")
            if len(var_data) >= 8:  # Minimum sample size for tests
                # Shapiro-Wilk test (for smaller samples)
                if len(var_data) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(var_data)
                    st.write(f"Shapiro-Wilk Test: estadístico = {shapiro_stat:.4f}, p-valor = {shapiro_p:.4f}")
                    
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(var_data, 'norm', args=(var_data.mean(), var_data.std()))
                st.write(f"Kolmogorov-Smirnov Test: estadístico = {ks_stat:.4f}, p-valor = {ks_p:.4f}")
                
                if ks_p > 0.05:
                    st.success("✅ Los datos parecen seguir una distribución normal (p > 0.05)")
                else:
                    st.warning("⚠️ Los datos no siguen una distribución normal (p ≤ 0.05)")
            else:
                st.info("Se necesitan al menos 8 observaciones para realizar tests de normalidad")

def show_groupby_analysis(data):
    """Display pandas GroupBy analysis with multiple aggregation options"""
    st.header("👥 Análisis GroupBy con Pandas")
    
    if len(data) == 0:
        st.warning("No hay datos disponibles para el análisis después de aplicar filtros.")
        return
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
    
    if not categorical_columns and not datetime_columns:
        st.warning("Se necesitan columnas categóricas o de fecha para realizar análisis GroupBy.")
        return
    
    st.subheader("🔧 Configuración de Agrupación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select grouping column(s)
        groupby_columns = st.multiselect(
            "Agrupar por (puedes seleccionar múltiples columnas)",
            categorical_columns + datetime_columns,
            key="groupby_cols"
        )
    
    with col2:
        # Select columns to analyze
        analyze_columns = st.multiselect(
            "Columnas a analizar",
            numeric_columns,
            default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns,
            key="analyze_cols"
        )
    
    if not groupby_columns or not analyze_columns:
        st.info("🔍 Selecciona al menos una columna para agrupar y una para analizar.")
        return
    
    # Select aggregation functions
    st.subheader("📊 Funciones de Agregación")
    
    agg_options = {
        'count': 'Contar valores',
        'sum': 'Suma',
        'mean': 'Media',
        'median': 'Mediana',
        'std': 'Desviación estándar',
        'var': 'Varianza',
        'min': 'Mínimo',
        'max': 'Máximo',
        'quantile_25': 'Percentil 25',
        'quantile_75': 'Percentil 75'
    }
    
    selected_aggs = st.multiselect(
        "Selecciona las agregaciones a aplicar",
        list(agg_options.keys()),
        default=['count', 'mean', 'sum'],
        format_func=lambda x: agg_options[x],
        key="agg_funcs"
    )
    
    if selected_aggs:
        try:
            # Perform GroupBy analysis
            with st.spinner("Realizando análisis GroupBy..."):
                
                # Prepare aggregation functions
                agg_funcs = []
                for agg in selected_aggs:
                    if agg == 'quantile_25':
                        agg_funcs.append(lambda x: x.quantile(0.25))
                    elif agg == 'quantile_75':
                        agg_funcs.append(lambda x: x.quantile(0.75))
                    else:
                        agg_funcs.append(agg)
                
                # Create proper aggregation dictionary
                agg_dict = {}
                for col in analyze_columns:
                    agg_dict[col] = []
                    for agg in selected_aggs:
                        if agg == 'quantile_25':
                            agg_dict[col].append(lambda x: x.quantile(0.25))
                        elif agg == 'quantile_75':
                            agg_dict[col].append(lambda x: x.quantile(0.75))
                        else:
                            agg_dict[col].append(agg)
                
                # Perform groupby
                if len(groupby_columns) == 1:
                    grouped = data.groupby(groupby_columns[0])[analyze_columns].agg(selected_aggs)
                else:
                    grouped = data.groupby(groupby_columns)[analyze_columns].agg(selected_aggs)
                
                # Display results
                st.subheader(f"📋 Resultados del GroupBy por: {', '.join(groupby_columns)}")
                
                # Show summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Grupos únicos", len(grouped))
                with col2:
                    st.metric("Columnas analizadas", len(analyze_columns))
                with col3:
                    st.metric("Agregaciones aplicadas", len(selected_aggs))
                
                # Display the grouped data
                st.dataframe(grouped.round(4), use_container_width=True)
                
                # Download option
                csv_data = grouped.to_csv()
                st.download_button(
                    label="📥 Descargar resultados CSV",
                    data=csv_data,
                    file_name=f"groupby_analysis_{'-'.join(groupby_columns)}.csv",
                    mime="text/csv"
                )
                
                # Visualization of GroupBy results
                st.subheader("📊 Visualización de Resultados")
                
                if len(analyze_columns) >= 1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        viz_column = st.selectbox(
                            "Columna para visualizar",
                            analyze_columns,
                            key="groupby_viz_col"
                        )
                    
                    with col2:
                        viz_agg = st.selectbox(
                            "Agregación para visualizar",
                            selected_aggs,
                            format_func=lambda x: agg_options[x],
                            key="groupby_viz_agg"
                        )
                    
                    # Create visualization
                    if len(groupby_columns) == 1:
                        # Single groupby column
                        viz_data = grouped[viz_column][viz_agg].reset_index()
                        
                        fig = px.bar(
                            viz_data,
                            x=groupby_columns[0],
                            y=viz_agg,
                            title=f"{agg_options[viz_agg]} de {viz_column} por {groupby_columns[0]}",
                            labels={viz_agg: f"{agg_options[viz_agg]} de {viz_column}"}
                        )
                        
                    else:
                        # Multiple groupby columns
                        viz_data = grouped[viz_column][viz_agg].reset_index()
                        
                        fig = px.bar(
                            viz_data,
                            x=groupby_columns[0],
                            y=viz_agg,
                            color=groupby_columns[1] if len(groupby_columns) > 1 else None,
                            title=f"{agg_options[viz_agg]} de {viz_column} por {', '.join(groupby_columns)}",
                            labels={viz_agg: f"{agg_options[viz_agg]} de {viz_column}"}
                        )
                    
                    fig.update_layout(height=500, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Advanced Analysis
                st.subheader("🔍 Análisis Avanzado del GroupBy")
                
                # Statistical comparison between groups
                if len(analyze_columns) >= 1 and 'mean' in selected_aggs:
                    selected_col_for_stats = st.selectbox(
                        "Columna para análisis estadístico detallado",
                        analyze_columns,
                        key="stats_col"
                    )
                    
                    # ANOVA test if applicable
                    if len(groupby_columns) == 1:
                        groups = []
                        group_names = []
                        for name, group in data.groupby(groupby_columns[0]):
                            group_data = group[selected_col_for_stats].dropna()
                            if len(group_data) > 0:
                                groups.append(group_data)
                                group_names.append(str(name))
                        
                        if len(groups) >= 2:
                            # One-way ANOVA
                            try:
                                f_stat, p_value = stats.f_oneway(*groups)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("F-estadístico", f"{f_stat:.4f}")
                                with col2:
                                    st.metric("p-valor", f"{p_value:.4f}")
                                
                                if p_value < 0.05:
                                    st.success(f"✅ Hay diferencias significativas entre grupos (p < 0.05)")
                                else:
                                    st.info(f"ℹ️ No hay diferencias significativas entre grupos (p ≥ 0.05)")
                                    
                            except Exception as e:
                                st.warning(f"No se pudo realizar ANOVA: {str(e)}")
                
                # Group size analysis
                st.write("**📏 Análisis del tamaño de grupos:**")
                group_sizes = data.groupby(groupby_columns).size().reset_index(name='Tamaño del Grupo')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Distribución de tamaños:")
                    st.dataframe(group_sizes.describe(), use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        group_sizes,
                        x='Tamaño del Grupo',
                        title="Distribución del Tamaño de Grupos",
                        nbins=20
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error en el análisis GroupBy: {str(e)}")
            st.info("💡 Tip: Asegúrate de que las columnas seleccionadas sean compatibles con las agregaciones elegidas.")

def show_ml_missing_data(data):
    """Display machine learning tools for handling missing data"""
    st.header("🤖 Machine Learning para Datos Faltantes")
    
    if len(data) == 0:
        st.warning("No data available for analysis.")
        return
    
    # Check for missing data
    missing_data = data.isnull().sum()
    columns_with_missing = missing_data[missing_data > 0].index.tolist()
    
    if not columns_with_missing:
        st.success("🎉 ¡Excelente! No hay datos faltantes en tu dataset.")
        return
    
    st.subheader("📋 Resumen de Datos Faltantes")
    
    # Display missing data summary
    missing_summary = pd.DataFrame({
        'Columna': missing_data.index,
        'Valores Faltantes': missing_data.values,
        'Porcentaje': (missing_data.values / len(data) * 100).round(2)
    })
    missing_summary = missing_summary[missing_summary['Valores Faltantes'] > 0].sort_values('Valores Faltantes', ascending=False)
    
    st.dataframe(missing_summary, use_container_width=True)
    
    # ML Imputation Section
    st.subheader("🔧 Imputación con Machine Learning")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_missing = [col for col in columns_with_missing if col in numeric_columns]
    
    if numeric_missing:
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox("Columna a Imputar", numeric_missing, key="ml_target")
        
        with col2:
            ml_model = st.selectbox("Modelo de ML", 
                                  ["Gradient Boosting", "Random Forest", "Decision Tree", "SVR"], 
                                  key="ml_model")
        
        if target_column:
            # Prepare data for ML
            feature_columns = [col for col in numeric_columns if col != target_column and col not in columns_with_missing]
            
            if len(feature_columns) == 0:
                st.warning("⚠️ No hay suficientes columnas sin datos faltantes para entrenar el modelo.")
                return
            
            st.write("**Columnas usadas como características:**")
            st.write(", ".join(feature_columns))
            
            # Split data into training (complete cases) and prediction (missing cases)
            complete_mask = data[target_column].notna()
            missing_mask = data[target_column].isna()
            
            if complete_mask.sum() < 10:
                st.warning("⚠️ Se necesitan al menos 10 casos completos para entrenar el modelo.")
                return
            
            X_complete = data.loc[complete_mask, feature_columns]
            y_complete = data.loc[complete_mask, target_column]
            X_missing = data.loc[missing_mask, feature_columns]
            
            # Handle any remaining missing values in features
            X_complete = X_complete.fillna(X_complete.mean())
            X_missing = X_missing.fillna(X_complete.mean())
            
            if st.button("🚀 Ejecutar Imputación con " + ml_model, key="run_ml_imputation"):
                try:
                    with st.spinner("Entrenando modelo y generando predicciones..."):
                        # Split for validation
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_complete, y_complete, test_size=0.2, random_state=42
                        )
                        
                        # Scale features for SVR
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        X_missing_scaled = scaler.transform(X_missing)
                        
                        # Initialize model
                        if ml_model == "Gradient Boosting":
                            model = GradientBoostingRegressor(random_state=42, n_estimators=100)
                            X_train_final = X_train
                            X_test_final = X_test
                            X_missing_final = X_missing
                        elif ml_model == "Random Forest":
                            model = RandomForestRegressor(random_state=42, n_estimators=100)
                            X_train_final = X_train
                            X_test_final = X_test
                            X_missing_final = X_missing
                        elif ml_model == "Decision Tree":
                            model = DecisionTreeRegressor(random_state=42, max_depth=10)
                            X_train_final = X_train
                            X_test_final = X_test
                            X_missing_final = X_missing
                        else:  # SVR
                            model = SVR(kernel='rbf', C=1.0, gamma='scale')
                            X_train_final = X_train_scaled
                            X_test_final = X_test_scaled
                            X_missing_final = X_missing_scaled
                        
                        # Train model
                        model.fit(X_train_final, y_train)
                        
                        # Validate model
                        y_pred_test = model.predict(X_test_final)
                        r2 = r2_score(y_test, y_pred_test)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        
                        # Display validation metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("R² Score", f"{r2:.4f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:.4f}")
                        
                        if r2 > 0.3:
                            st.success(f"✅ Modelo con buena capacidad predictiva (R² = {r2:.4f})")
                        else:
                            st.warning(f"⚠️ Modelo con capacidad predictiva limitada (R² = {r2:.4f})")
                        
                        # Predict missing values
                        if len(X_missing) > 0:
                            predicted_values = model.predict(X_missing_final)
                            
                            # Create results dataframe
                            results_df = data.copy()
                            results_df.loc[missing_mask, target_column] = predicted_values
                            
                            # Store in session state
                            st.session_state.imputed_data = results_df
                            st.session_state.imputation_info = {
                                'target_column': target_column,
                                'model_used': ml_model,
                                'r2_score': r2,
                                'rmse': rmse,
                                'values_imputed': len(predicted_values)
                            }
                            
                            # Update the main data with imputed values
                            st.session_state.data = results_df.copy()
                            st.session_state.processed_data = results_df.copy()
                            
                            st.success(f"🎉 Se imputaron {len(predicted_values)} valores faltantes!")
                            
                            # COMPREHENSIVE POST-IMPUTATION ANALYSIS
                            st.subheader("📊 Análisis Completo Post-Imputación")
                            
                            # Statistical comparison
                            original_data = data[target_column].dropna()
                            complete_data = results_df[target_column]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Media Original", f"{original_data.mean():.4f}")
                                st.metric("Media Completa", f"{complete_data.mean():.4f}")
                            with col2:
                                st.metric("Mediana Original", f"{original_data.median():.4f}")
                                st.metric("Mediana Completa", f"{complete_data.median():.4f}")
                            with col3:
                                st.metric("Std Original", f"{original_data.std():.4f}")
                                st.metric("Std Completa", f"{complete_data.std():.4f}")
                            with col4:
                                st.metric("Asimetría Original", f"{original_data.skew():.4f}")
                                st.metric("Asimetría Completa", f"{complete_data.skew():.4f}")
                            
                            # Distribution comparison visualization
                            fig = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=('Distribución Original vs Completa', 'Box Plot Comparativo', 
                                               'QQ Plot vs Normal', 'Densidad Superpuesta'),
                                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                       [{"secondary_y": False}, {"secondary_y": False}]]
                            )
                            
                            # Histogram comparison
                            fig.add_trace(go.Histogram(
                                x=original_data,
                                name='Datos Originales',
                                opacity=0.7,
                                nbinsx=30,
                                histnorm='probability density'
                            ), row=1, col=1)
                            
                            fig.add_trace(go.Histogram(
                                x=predicted_values,
                                name='Valores Imputados',
                                opacity=0.7,
                                nbinsx=30,
                                histnorm='probability density'
                            ), row=1, col=1)
                            
                            # Box plot comparison
                            fig.add_trace(go.Box(
                                y=original_data,
                                name='Original',
                                boxpoints='outliers'
                            ), row=1, col=2)
                            
                            fig.add_trace(go.Box(
                                y=predicted_values,
                                name='Imputado',
                                boxpoints='outliers'
                            ), row=1, col=2)
                            
                            # QQ plot for normality
                            import scipy.stats as stats
                            qq_original = stats.probplot(original_data, dist="norm", plot=None)
                            qq_complete = stats.probplot(complete_data, dist="norm", plot=None)
                            
                            fig.add_trace(go.Scatter(
                                x=qq_original[0][0],
                                y=qq_original[0][1],
                                mode='markers',
                                name='QQ Original',
                                marker=dict(color='blue')
                            ), row=2, col=1)
                            
                            fig.add_trace(go.Scatter(
                                x=qq_complete[0][0],
                                y=qq_complete[0][1],
                                mode='markers',
                                name='QQ Completo',
                                marker=dict(color='red')
                            ), row=2, col=1)
                            
                            # Density plot
                            x_range = np.linspace(min(complete_data.min(), original_data.min()), 
                                                max(complete_data.max(), original_data.max()), 100)
                            
                            # KDE for original data
                            from scipy.stats import gaussian_kde
                            kde_original = gaussian_kde(original_data)
                            kde_complete = gaussian_kde(complete_data)
                            
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=kde_original(x_range),
                                mode='lines',
                                name='Densidad Original',
                                line=dict(color='blue')
                            ), row=2, col=2)
                            
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=kde_complete(x_range),
                                mode='lines',
                                name='Densidad Completa',
                                line=dict(color='red')
                            ), row=2, col=2)
                            
                            fig.update_layout(
                                height=800,
                                title_text=f"Análisis Integral: {target_column}",
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistical tests
                            st.subheader("🧪 Tests Estadísticos Post-Imputación")
                            
                            # Kolmogorov-Smirnov test between original and imputed
                            try:
                                ks_stat, ks_p = stats.ks_2samp(original_data, predicted_values)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("KS Estadístico (Original vs Imputado)", f"{ks_stat:.4f}")
                                with col2:
                                    st.metric("KS p-valor", f"{ks_p:.4f}")
                                
                                if ks_p > 0.05:
                                    st.success("✅ Las distribuciones original e imputada son similares (p > 0.05)")
                                else:
                                    st.warning("⚠️ Las distribuciones difieren significativamente (p ≤ 0.05)")
                            except:
                                st.info("No se pudo realizar el test KS")
                            
                            # Normality tests for complete data
                            if len(complete_data) <= 5000:
                                shapiro_stat, shapiro_p = stats.shapiro(complete_data)
                                st.write(f"**Test de Normalidad (datos completos):**")
                                st.write(f"Shapiro-Wilk: estadístico = {shapiro_stat:.4f}, p-valor = {shapiro_p:.4f}")
                            
                            # Feature importance (for tree-based models)
                            if ml_model in ["Gradient Boosting", "Random Forest", "Decision Tree"]:
                                st.subheader("🌳 Importancia de Características")
                                
                                try:
                                    importance_df = pd.DataFrame({
                                        'Característica': feature_columns,
                                        'Importancia': model.feature_importances_
                                    }).sort_values('Importancia', ascending=False)
                                    
                                    fig_imp = px.bar(
                                        importance_df,
                                        x='Importancia',
                                        y='Característica',
                                        orientation='h',
                                        title='Importancia de Características en la Imputación'
                                    )
                                    fig_imp.update_layout(height=400)
                                    st.plotly_chart(fig_imp, use_container_width=True)
                                    
                                    st.dataframe(importance_df, use_container_width=True)
                                    
                                except Exception as e:
                                    st.info(f"No se puede mostrar importancia de características para {ml_model}")
                            
                            # Model performance analysis
                            st.subheader("📈 Análisis de Rendimiento del Modelo")
                            
                            # Residuals analysis
                            residuals = y_test - y_pred_test
                            
                            fig_res = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=('Residuos vs Predicciones', 'Distribución de Residuos')
                            )
                            
                            # Residuals vs predictions
                            fig_res.add_trace(go.Scatter(
                                x=y_pred_test,
                                y=residuals,
                                mode='markers',
                                name='Residuos',
                                marker=dict(opacity=0.6)
                            ), row=1, col=1)
                            
                            # Add zero line
                            fig_res.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
                            
                            # Residuals histogram
                            fig_res.add_trace(go.Histogram(
                                x=residuals,
                                name='Distribución',
                                nbinsx=20
                            ), row=1, col=2)
                            
                            fig_res.update_layout(
                                height=400,
                                title_text="Análisis de Residuos del Modelo"
                            )
                            
                            st.plotly_chart(fig_res, use_container_width=True)
                            
                            # Additional metrics
                            mae = np.mean(np.abs(residuals))
                            mape = np.mean(np.abs(residuals / y_test)) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("MAE (Error Absoluto Medio)", f"{mae:.4f}")
                            with col2:
                                st.metric("MAPE (%)", f"{mape:.2f}%")
                            with col3:
                                residual_std = np.std(residuals)
                                st.metric("Std de Residuos", f"{residual_std:.4f}")
                            
                            # Data quality assessment
                            st.subheader("✅ Evaluación de Calidad de Datos")
                            
                            quality_metrics = {
                                "Completitud": f"{(len(complete_data) - complete_data.isnull().sum()) / len(complete_data) * 100:.1f}%",
                                "Valores Imputados": f"{len(predicted_values)} ({len(predicted_values)/len(data)*100:.1f}%)",
                                "Cambio en Media": f"{((complete_data.mean() - original_data.mean()) / original_data.mean() * 100):.2f}%",
                                "Cambio en Std": f"{((complete_data.std() - original_data.std()) / original_data.std() * 100):.2f}%"
                            }
                            
                            for metric, value in quality_metrics.items():
                                st.write(f"**{metric}**: {value}")
                            
                            # Recommendations
                            st.subheader("💡 Recomendaciones")
                            
                            if r2 > 0.7:
                                st.success("🎯 Excelente calidad de imputación. El modelo explica más del 70% de la varianza.")
                            elif r2 > 0.5:
                                st.info("👍 Buena calidad de imputación. Considera validar con expertos del dominio.")
                            else:
                                st.warning("⚠️ Calidad de imputación limitada. Considera:")
                                st.write("- Agregar más características predictivas")
                                st.write("- Probar diferentes modelos")
                                st.write("- Revisar la calidad de los datos originales")
                            
                            # Download option
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Descargar Datos con Imputación",
                                data=csv,
                                file_name=f"datos_imputados_{target_column}.csv",
                                mime="text/csv"
                            )
                            
                        else:
                            st.info("No hay valores faltantes para imputar en esta columna.")
                            
                except Exception as e:
                    st.error(f"❌ Error durante la imputación: {str(e)}")
    
    else:
        st.info("No hay columnas numéricas con datos faltantes para aplicar ML.")
    
    # Display imputation history if available
    if 'imputation_info' in st.session_state:
        st.subheader("📝 Historial de Imputación")
        info = st.session_state.imputation_info
        
        st.write(f"**Última imputación realizada:**")
        st.write(f"- Columna: {info['target_column']}")
        st.write(f"- Modelo usado: {info['model_used']}")
        st.write(f"- R² Score: {info['r2_score']:.4f}")
        st.write(f"- RMSE: {info['rmse']:.4f}")
        st.write(f"- Valores imputados: {info['values_imputed']}")

if __name__ == "__main__":
    main()
