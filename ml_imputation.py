import streamlit as st
import pandas as pd
import numpy as np

def show_ml_imputation_redirection(data):
    """Show redirection interface to external ML imputation service"""
    
    st.header("🤖 Imputación con Machine Learning")
    
    # Hero section with service description
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 1rem; color: white; margin-bottom: 2rem;">
        <h2 style="color: white; margin-bottom: 1rem;">🚀 Servicio Especializado de ML</h2>
        <p style="font-size: 1.1rem; margin-bottom: 0;">
            Accede a nuestra herramienta especializada para imputación avanzada de datos faltantes 
            utilizando algoritmos de Machine Learning de última generación.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Service features
    st.subheader("✨ Características del Servicio")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Algoritmos Avanzados**
        - Gradient Boosting
        - Random Forest
        - Decision Trees
        - Support Vector Regression
        - Redes Neuronales
        """)
    
    with col2:
        st.markdown("""
        **📊 Análisis Profundo**
        - Evaluación de calidad
        - Métricas de rendimiento
        - Análisis post-imputación
        - Validación estadística
        - Reportes detallados
        """)
    
    with col3:
        st.markdown("""
        **🛠️ Funcionalidades**
        - Interfaz intuitiva
        - Múltiples formatos
        - Configuración flexible
        - Visualizaciones interactivas
        - Descarga de resultados
        """)
    
    # Current data status
    if len(data) > 0:
        st.subheader("📋 Estado Actual de tus Datos")
        
        # Check for missing data in current dataset
        missing_data = data.isnull().sum()
        columns_with_missing = missing_data[missing_data > 0]
        
        if len(columns_with_missing) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Filas", f"{len(data):,}")
            with col2:
                st.metric("Columnas con Datos Faltantes", len(columns_with_missing))
            with col3:
                total_missing = missing_data.sum()
                missing_percentage = (total_missing / (len(data) * len(data.columns)) * 100)
                st.metric("% Datos Faltantes", f"{missing_percentage:.1f}%")
            
            # Show missing data summary
            st.write("**Resumen de Datos Faltantes:**")
            missing_summary = pd.DataFrame({
                'Columna': columns_with_missing.index,
                'Valores Faltantes': columns_with_missing.values,
                'Porcentaje': (columns_with_missing.values / len(data) * 100).round(2)
            })
            st.dataframe(missing_summary, use_container_width=True)
            
            st.success("✅ ¡Perfecto! Tienes datos faltantes que pueden ser imputados con ML.")
            
        else:
            st.success("🎉 ¡Excelente! Tu dataset no tiene datos faltantes.")
            st.info("💡 El servicio de ML también puede ser útil para:")
            st.markdown("""
            - **Análisis predictivo** de datos completos
            - **Detección de anomalías** en tus datos
            - **Modelado avanzado** para insights profundos
            - **Validación cruzada** de modelos existentes
            """)
    
    else:
        st.info("📁 Carga tus datos en la pestaña 'Data Overview' para ver el análisis de datos faltantes.")
    
    # Benefits section
    st.subheader("🌟 ¿Por qué usar nuestro Servicio Especializado?")
    
    benefits_col1, benefits_col2 = st.columns(2)
    
    with benefits_col1:
        st.markdown("""
        **🔬 Precisión Superior**
        - Algoritmos optimizados específicamente para imputación
        - Validación cruzada automática
        - Métricas de calidad en tiempo real
        
        **⚡ Rendimiento Optimizado**
        - Procesamiento paralelo
        - Optimización de hiperparámetros
        - Manejo eficiente de datasets grandes
        """)
    
    with benefits_col2:
        st.markdown("""
        **🎨 Interfaz Especializada**
        - Diseñada específicamente para ML
        - Controles avanzados de configuración
        - Visualizaciones especializadas
        
        **📈 Análisis Completo**
        - Comparación pre/post imputación
        - Tests estadísticos automáticos
        - Recomendaciones inteligentes
        """)
    
    # Instructions section
    st.subheader("📖 Cómo usar el Servicio")
    
    with st.expander("👆 Ver instrucciones paso a paso"):
        st.markdown("""
        ### Pasos para usar el servicio de ML:
        
        1. **📤 Exporta tus datos** (opcional)
           - Descarga tus datos actuales desde la pestaña 'Data Overview'
           - O prepara tu archivo CSV/Excel local
        
        2. **🚀 Accede al servicio**
           - Haz clic en el botón "Acceder al Servicio ML" abajo
           - Se abrirá en una nueva pestaña
        
        3. **📁 Carga tus datos**
           - Sube tu archivo en el servicio especializado
           - Configura los parámetros de importación
        
        4. **⚙️ Configura la imputación**
           - Selecciona las columnas a imputar
           - Elige el algoritmo de ML preferido
           - Ajusta los parámetros avanzados
        
        5. **🎯 Ejecuta y analiza**
           - Ejecuta la imputación
           - Revisa las métricas de calidad
           - Analiza los resultados
        
        6. **📥 Descarga resultados**
           - Descarga el dataset imputado
           - Guarda los reportes de análisis
        """)
    
    # Warning about data transfer
    st.warning("""
    **🔒 Nota de Privacidad**: El servicio especializado es una aplicación independiente. 
    Asegúrate de revisar las políticas de privacidad antes de cargar datos sensibles.
    """)
    
    # Call-to-action section
    st.markdown("---")
    
    # Center the button using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3>🚀 ¿Listo para comenzar?</h3>
            <p>Accede a nuestra herramienta especializada de Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main redirection button
        if st.button(
            "🤖 Acceder al Servicio de ML", 
            type="primary",
            help="Abre el servicio especializado en una nueva pestaña",
            use_container_width=True
        ):
            st.balloons()
            st.success("🚀 ¡Redirigiendo al servicio especializado!")
            
            # JavaScript to open in new tab
            st.markdown("""
            <script>
                window.open('https://imputacion-machine-learning-bamnx9kavua2xgnfxtsjhw.streamlit.app/', '_blank');
            </script>
            """, unsafe_allow_html=True)
            
            # Alternative link in case JavaScript doesn't work
            st.markdown("""
            <div style="text-align: center; margin-top: 1rem;">
                <p>Si no se abre automáticamente, 
                <a href="https://imputacion-machine-learning-bamnx9kavua2xgnfxtsjhw.streamlit.app/" 
                   target="_blank" style="color: #ff6b6b; text-decoration: none; font-weight: bold;">
                   haz clic aquí para acceder manualmente
                </a></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional resources
    st.markdown("---")
    st.subheader("📚 Recursos Adicionales")
    
    resource_col1, resource_col2 = st.columns(2)
    
    with resource_col1:
        st.markdown("""
        **📖 Documentación**
        - Consulta la pestaña 'Documentation' para guías completas
        - Aprende sobre mejores prácticas de imputación
        - Entiende los diferentes algoritmos de ML
        """)
    
    with resource_col2:
        st.markdown("""
        **💡 Consejos Rápidos**
        - Usa este dashboard para análisis exploratorio inicial
        - El servicio ML es ideal para imputación avanzada
        - Combina ambas herramientas para resultados óptimos
        """)
    
    # Quick access to external service
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-top: 2rem;">
        <p style="margin: 0; text-align: center;">
            <strong>🔗 Acceso Rápido:</strong> 
            <a href="https://imputacion-machine-learning-bamnx9kavua2xgnfxtsjhw.streamlit.app/" 
               target="_blank" style="color: #1f77b4; text-decoration: none;">
               imputacion-machine-learning-bamnx9kavua2xgnfxtsjhw.streamlit.app
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)
