import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

class VisualizationEngine:
    """Class to handle interactive visualization creation"""
    
    def __init__(self, data):
        self.data = data
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        self.datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
    
    def create_scatter_plot(self):
        """Create interactive scatter plot"""
        st.subheader("🎯 Scatter Plot")
        
        if len(self.numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for scatter plot.")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-axis", self.numeric_columns, key="scatter_x")
        with col2:
            y_axis = st.selectbox("Y-axis", [col for col in self.numeric_columns if col != x_axis], key="scatter_y")
        with col3:
            color_by = st.selectbox("Color by", ["None"] + self.categorical_columns, key="scatter_color")
        
        # Additional options
        col4, col5 = st.columns(2)
        with col4:
            size_by = st.selectbox("Size by", ["None"] + self.numeric_columns, key="scatter_size")
        with col5:
            show_trendline = st.checkbox("Show trendline", key="scatter_trend")
        
        # Create plot
        fig_kwargs = {
            'data_frame': self.data,
            'x': x_axis,
            'y': y_axis,
            'title': f'{y_axis} vs {x_axis}'
        }
        
        if color_by != "None":
            fig_kwargs['color'] = color_by
        
        if size_by != "None":
            fig_kwargs['size'] = size_by
        
        if show_trendline:
            fig_kwargs['trendline'] = 'ols'
        
        fig = px.scatter(**fig_kwargs)
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation if both axes are numeric
        if x_axis in self.numeric_columns and y_axis in self.numeric_columns:
            correlation = self.data[x_axis].corr(self.data[y_axis])
            st.metric("Correlation Coefficient", f"{correlation:.3f}")
    
    def create_line_chart(self):
        """Create interactive line chart"""
        st.subheader("📈 Line Chart")
        
        if not self.numeric_columns:
            st.warning("Need at least 1 numeric column for line chart.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            y_columns = st.multiselect("Y-axis columns", self.numeric_columns, key="line_y")
        with col2:
            x_axis = st.selectbox("X-axis", ["Index"] + self.numeric_columns + self.datetime_columns, key="line_x")
        
        if not y_columns:
            st.info("Please select at least one Y-axis column.")
            return
        
        # Create plot
        fig = go.Figure()
        
        for col in y_columns:
            if x_axis == "Index":
                x_data = self.data.index
            else:
                x_data = self.data[x_axis]
            
            fig.add_trace(go.Scatter(
                x=x_data,
                y=self.data[col],
                mode='lines+markers',
                name=col,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f"Line Chart: {', '.join(y_columns)}",
            xaxis_title=x_axis,
            yaxis_title="Value",
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_bar_chart(self):
        """Create interactive bar chart"""
        st.subheader("📊 Bar Chart")
        
        if not self.categorical_columns or not self.numeric_columns:
            st.warning("Need at least 1 categorical and 1 numeric column for bar chart.")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("Category (X-axis)", self.categorical_columns, key="bar_x")
        with col2:
            y_axis = st.selectbox("Value (Y-axis)", self.numeric_columns, key="bar_y")
        with col3:
            agg_function = st.selectbox("Aggregation", ["mean", "sum", "count", "median"], key="bar_agg")
        
        # Additional options
        col4, col5 = st.columns(2)
        with col4:
            color_by = st.selectbox("Color by", ["None"] + self.categorical_columns, key="bar_color")
        with col5:
            orientation = st.selectbox("Orientation", ["vertical", "horizontal"], key="bar_orient")
        
        # Aggregate data
        if agg_function == "count":
            agg_data = self.data.groupby(x_axis).size().reset_index(name='count')
            y_col = 'count'
        else:
            agg_data = self.data.groupby(x_axis)[y_axis].agg(agg_function).reset_index()
            y_col = y_axis
        
        # Create plot
        if orientation == "vertical":
            fig = px.bar(agg_data, x=x_axis, y=y_col, 
                        title=f'{agg_function.title()} of {y_axis} by {x_axis}')
        else:
            fig = px.bar(agg_data, x=y_col, y=x_axis, orientation='h',
                        title=f'{agg_function.title()} of {y_axis} by {x_axis}')
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_histogram(self):
        """Create interactive histogram"""
        st.subheader("📊 Histogram")
        
        if not self.numeric_columns:
            st.warning("Need at least 1 numeric column for histogram.")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            column = st.selectbox("Column", self.numeric_columns, key="hist_col")
        with col2:
            bins = st.slider("Number of bins", 10, 100, 30, key="hist_bins")
        with col3:
            color_by = st.selectbox("Color by", ["None"] + self.categorical_columns, key="hist_color")
        
        # Create plot
        fig_kwargs = {
            'data_frame': self.data,
            'x': column,
            'nbins': bins,
            'title': f'Distribution of {column}'
        }
        
        if color_by != "None":
            fig_kwargs['color'] = color_by
        
        fig = px.histogram(**fig_kwargs)
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{self.data[column].mean():.2f}")
        with col2:
            st.metric("Median", f"{self.data[column].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{self.data[column].std():.2f}")
        with col4:
            st.metric("Skewness", f"{self.data[column].skew():.2f}")
    
    def create_box_plot(self):
        """Create interactive box plot"""
        st.subheader("📦 Box Plot")
        
        if not self.numeric_columns:
            st.warning("Need at least 1 numeric column for box plot.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            y_axis = st.selectbox("Value column", self.numeric_columns, key="box_y")
        with col2:
            x_axis = st.selectbox("Group by", ["None"] + self.categorical_columns, key="box_x")
        
        # Create plot
        if x_axis == "None":
            fig = px.box(self.data, y=y_axis, title=f'Box Plot of {y_axis}')
        else:
            fig = px.box(self.data, x=x_axis, y=y_axis, title=f'Box Plot of {y_axis} by {x_axis}')
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show outlier statistics
        Q1 = self.data[y_axis].quantile(0.25)
        Q3 = self.data[y_axis].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.data[(self.data[y_axis] < lower_bound) | (self.data[y_axis] > upper_bound)]
        
        st.info(f"Detected {len(outliers)} outliers (values outside {lower_bound:.2f} to {upper_bound:.2f})")
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap"""
        st.subheader("🔥 Correlation Heatmap")
        
        if len(self.numeric_columns) < 2:
            st.warning("Need at least 2 numeric columns for correlation heatmap.")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.data[self.numeric_columns].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show strongest correlations
        st.subheader("Strongest Correlations")
        
        # Get correlation pairs
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
        
        st.dataframe(corr_df.head(10), use_container_width=True)
    
    def create_distribution_plot(self):
        """Create distribution plot with multiple options"""
        st.subheader("📈 Distribution Plot")
        
        if not self.numeric_columns:
            st.warning("Need at least 1 numeric column for distribution plot.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            column = st.selectbox("Column", self.numeric_columns, key="dist_col")
        with col2:
            plot_type = st.selectbox("Plot type", ["Histogram", "Density", "Both"], key="dist_type")
        
        # Create subplots if showing both
        if plot_type == "Both":
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('Histogram', 'Density Plot'),
                              vertical_spacing=0.1)
            
            # Histogram
            fig.add_trace(
                go.Histogram(x=self.data[column], name="Histogram", nbinsx=30),
                row=1, col=1
            )
            
            # Density plot
            fig.add_trace(
                go.Histogram(x=self.data[column], histnorm='probability density', 
                           name="Density", nbinsx=30, opacity=0.7),
                row=2, col=1
            )
            
            fig.update_layout(height=800, title=f"Distribution of {column}")
            
        elif plot_type == "Histogram":
            fig = px.histogram(self.data, x=column, title=f"Histogram of {column}")
            fig.update_layout(height=600)
            
        else:  # Density
            fig = px.histogram(self.data, x=column, histnorm='probability density',
                             title=f"Density Plot of {column}")
            fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_time_series(self):
        """Create time series visualization"""
        st.subheader("⏰ Time Series")
        
        if not self.datetime_columns and not self.numeric_columns:
            st.warning("Need at least 1 datetime or numeric column for time series.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            if self.datetime_columns:
                time_col = st.selectbox("Time column", self.datetime_columns, key="ts_time")
            else:
                time_col = st.selectbox("Time column (or use index)", ["Index"] + self.numeric_columns, key="ts_time")
        
        with col2:
            value_cols = st.multiselect("Value columns", self.numeric_columns, key="ts_values")
        
        if not value_cols:
            st.info("Please select at least one value column.")
            return
        
        # Create time series plot
        fig = go.Figure()
        
        for col in value_cols:
            if time_col == "Index":
                x_data = self.data.index
            else:
                x_data = self.data[time_col]
            
            fig.add_trace(go.Scatter(
                x=x_data,
                y=self.data[col],
                mode='lines+markers',
                name=col,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f"Time Series: {', '.join(value_cols)}",
            xaxis_title=time_col,
            yaxis_title="Value",
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show time series statistics
        if len(value_cols) == 1:
            col = value_cols[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{self.data[col].mean():.2f}")
            with col2:
                st.metric("Trend", "↗️" if self.data[col].iloc[-1] > self.data[col].iloc[0] else "↘️")
            with col3:
                volatility = self.data[col].std() / self.data[col].mean() * 100
                st.metric("Volatility", f"{volatility:.1f}%")
            with col4:
                change = ((self.data[col].iloc[-1] - self.data[col].iloc[0]) / self.data[col].iloc[0] * 100)
                st.metric("Total Change", f"{change:.1f}%")
