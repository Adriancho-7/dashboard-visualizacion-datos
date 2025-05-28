import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_ml_imputation_interface(data):
    """Complete ML imputation interface with model selection and post-analysis"""
    
    st.header("🤖 Imputación con Machine Learning")
    
    if len(data) == 0:
        st.warning("No hay datos disponibles para análisis.")
        return data
    
    # Check for missing data
    missing_data = data.isnull().sum()
    columns_with_missing = missing_data[missing_data > 0].index.tolist()
    
    if not columns_with_missing:
        st.success("🎉 ¡No hay datos faltantes en tu dataset!")
        return data
    
    # Display missing data summary
    st.subheader("📋 Resumen de Datos Faltantes")
    missing_summary = pd.DataFrame({
        'Columna': missing_data.index,
        'Valores Faltantes': missing_data.values,
        'Porcentaje': (missing_data.values / len(data) * 100).round(2)
    })
    missing_summary = missing_summary[missing_summary['Valores Faltantes'] > 0]
    st.dataframe(missing_summary, use_container_width=True)
    
    # ML Configuration
    st.subheader("🔧 Configuración de Imputación ML")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_missing = [col for col in columns_with_missing if col in numeric_columns]
    
    if not numeric_missing:
        st.info("No hay columnas numéricas con datos faltantes para ML.")
        return data
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_column = st.selectbox(
            "📍 Columna a Imputar", 
            numeric_missing, 
            key="ml_target_col"
        )
    
    with col2:
        ml_model = st.selectbox(
            "🎯 Modelo de ML", 
            ["Gradient Boosting", "Random Forest", "Decision Tree", "SVR"], 
            key="ml_model_type"
        )
    
    if target_column:
        # Prepare features - USE ALL NUMERIC COLUMNS except target
        feature_columns = [col for col in numeric_columns if col != target_column]
        
        if len(feature_columns) == 0:
            st.warning("⚠️ No hay suficientes columnas para usar como características.")
            return data
        
        st.write(f"**Características usadas:** {', '.join(feature_columns)}")
        
        # Check data availability
        complete_mask = data[target_column].notna()
        missing_mask = data[target_column].isna()
        
        if complete_mask.sum() < 10:
            st.warning("⚠️ Se necesitan al menos 10 casos completos.")
            return data
        
        st.write(f"**Casos para entrenamiento:** {complete_mask.sum()}")
        st.write(f"**Valores a imputar:** {missing_mask.sum()}")
        
        # Execute ML Imputation
        if st.button(f"🚀 Ejecutar Imputación con {ml_model}", key="execute_ml"):
            
            with st.spinner(f"Entrenando {ml_model} y realizando imputación..."):
                
                # Prepare data - CRITICAL: Handle ALL missing values properly
                # Fill missing values in features with mean (for ALL rows)
                feature_data_filled = data[feature_columns].fillna(data[feature_columns].mean())
                
                # Separate complete and missing cases for target variable
                X_complete = feature_data_filled.loc[complete_mask]
                y_complete = data.loc[complete_mask, target_column]
                X_missing = feature_data_filled.loc[missing_mask]
                
                st.info(f"📊 Casos con {target_column} completo: {len(X_complete)}")
                st.info(f"🎯 Casos con {target_column} faltante: {len(X_missing)}")
                
                if len(X_missing) == 0:
                    st.warning("No hay valores faltantes para imputar en esta columna.")
                    return data
                
                # Split for validation
                X_train, X_test, y_train, y_test = train_test_split(
                    X_complete, y_complete, test_size=0.2, random_state=42
                )
                
                # Prepare models and scaling
                scaler = StandardScaler()
                
                if ml_model == "Gradient Boosting":
                    model = GradientBoostingRegressor(random_state=42, n_estimators=100)
                    X_train_final, X_test_final, X_missing_final = X_train, X_test, X_missing
                    
                elif ml_model == "Random Forest":
                    model = RandomForestRegressor(random_state=42, n_estimators=100)
                    X_train_final, X_test_final, X_missing_final = X_train, X_test, X_missing
                    
                elif ml_model == "Decision Tree":
                    model = DecisionTreeRegressor(random_state=42, max_depth=10)
                    X_train_final, X_test_final, X_missing_final = X_train, X_test, X_missing
                    
                else:  # SVR
                    model = SVR(kernel='rbf', C=1.0)
                    X_train_final = scaler.fit_transform(X_train)
                    X_test_final = scaler.transform(X_test)
                    X_missing_final = scaler.transform(X_missing)
                
                # Train model
                model.fit(X_train_final, y_train)
                
                # Validate model
                y_pred_test = model.predict(X_test_final)
                r2 = r2_score(y_test, y_pred_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # Display model performance
                st.subheader("📊 Rendimiento del Modelo")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score", f"{r2:.4f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                
                if r2 > 0.5:
                    st.success(f"✅ Buen modelo para imputación (R² = {r2:.4f})")
                else:
                    st.warning(f"⚠️ Modelo con capacidad limitada (R² = {r2:.4f})")
                
                # Predict missing values
                if len(X_missing) > 0:
                    predicted_values = model.predict(X_missing_final)
                    
                    # Create imputed dataset - FORCE ALL NaN values to be replaced
                    imputed_data = data.copy()
                    
                    # Ensure we're replacing ALL NaN values in target column
                    nan_indices = imputed_data[target_column].isna()
                    imputed_data.loc[nan_indices, target_column] = predicted_values
                    
                    # Verify no NaN values remain in target column
                    remaining_nans = imputed_data[target_column].isna().sum()
                    st.success(f"✅ Verificado: {remaining_nans} valores NaN restantes en {target_column}")
                    
                    st.success(f"🎉 ¡Se imputaron {len(predicted_values)} valores!")
                    
                    # COMPREHENSIVE POST-IMPUTATION ANALYSIS
                    st.subheader("📈 Análisis Post-Imputación")
                    
                    # Statistical comparison
                    original_data = data[target_column].dropna()
                    complete_data = imputed_data[target_column]
                    
                    # Show statistics comparison
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
                        change_mean = ((complete_data.mean() - original_data.mean()) / original_data.mean() * 100)
                        st.metric("Cambio en Media (%)", f"{change_mean:.2f}%")
                    
                    # Visualization comparison
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Distribución Comparativa', 'Box Plot Comparativo')
                    )
                    
                    # Histogram comparison
                    fig.add_trace(
                        go.Histogram(x=original_data, name='Original', opacity=0.7, nbinsx=30),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Histogram(x=predicted_values, name='Imputado', opacity=0.7, nbinsx=30),
                        row=1, col=1
                    )
                    
                    # Box plot comparison
                    fig.add_trace(
                        go.Box(y=original_data, name='Original'),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Box(y=predicted_values, name='Imputado'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=500, title_text=f"Análisis de {target_column}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance for tree models
                    if ml_model in ["Gradient Boosting", "Random Forest", "Decision Tree"]:
                        st.subheader("🌳 Importancia de Características")
                        
                        importance_df = pd.DataFrame({
                            'Característica': feature_columns,
                            'Importancia': model.feature_importances_
                        }).sort_values('Importancia', ascending=False)
                        
                        fig_imp = px.bar(
                            importance_df,
                            x='Importancia',
                            y='Característica',
                            orientation='h',
                            title='Importancia de Variables en la Imputación'
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # Quality assessment
                    st.subheader("✅ Evaluación de Calidad")
                    
                    # Kolmogorov-Smirnov test
                    try:
                        from scipy import stats
                        ks_stat, ks_p = stats.ks_2samp(original_data, predicted_values)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("KS Estadístico", f"{ks_stat:.4f}")
                        with col2:
                            st.metric("KS p-valor", f"{ks_p:.4f}")
                        
                        if ks_p > 0.05:
                            st.success("✅ Distribuciones similares (p > 0.05)")
                        else:
                            st.warning("⚠️ Distribuciones diferentes (p ≤ 0.05)")
                    except:
                        st.info("No se pudo realizar test estadístico")
                    
                    # Recommendations
                    st.subheader("💡 Recomendaciones")
                    if r2 > 0.7:
                        st.success("🎯 Excelente calidad de imputación")
                    elif r2 > 0.5:
                        st.info("👍 Buena calidad - Validar con expertos")
                    else:
                        st.warning("⚠️ Calidad limitada - Considerar más características")
                    
                    # Download option
                    csv = imputed_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Datos Imputados",
                        data=csv,
                        file_name=f"datos_imputados_{target_column}_{ml_model.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                    
                    # Store in session state for use in other tabs
                    st.session_state.data = imputed_data
                    st.session_state.processed_data = imputed_data
                    st.session_state.imputation_info = {
                        'target_column': target_column,
                        'model_used': ml_model,
                        'r2_score': r2,
                        'rmse': rmse,
                        'values_imputed': len(predicted_values)
                    }
                    
                    st.info("🔄 Los datos han sido actualizados en todas las pestañas del dashboard")
                    
                    return imputed_data
                else:
                    st.info("No hay valores faltantes para imputar")
                    return data
    
    return data