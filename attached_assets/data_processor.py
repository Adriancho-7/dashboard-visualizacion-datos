import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import io

class DataProcessor:
    """Class to handle data loading and preprocessing"""
    
    def __init__(self):
        pass
    
    def load_csv(self, uploaded_file, separator=',', encoding='utf-8'):
        """
        Load CSV file with specified parameters
        
        Args:
            uploaded_file: Streamlit uploaded file object
            separator: Column separator character
            encoding: File encoding
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            
            # Read CSV with specified parameters
            data = pd.read_csv(
                uploaded_file,
                sep=separator,
                encoding=encoding,
                low_memory=False
            )
            
            # Basic data cleaning
            data = self._clean_data(data)
            
            return data
            
        except UnicodeDecodeError:
            raise Exception(f"Unable to decode file with {encoding} encoding. Try a different encoding.")
        except pd.errors.EmptyDataError:
            raise Exception("The uploaded file is empty.")
        except pd.errors.ParserError as e:
            raise Exception(f"Error parsing CSV file: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error loading CSV: {str(e)}")
    
    def load_excel(self, uploaded_file, sheet_name=0):
        """
        Load Excel file with specified sheet
        
        Args:
            uploaded_file: Streamlit uploaded file object
            sheet_name: Name or index of sheet to load
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            
            # Read Excel file
            data = pd.read_excel(
                uploaded_file,
                sheet_name=sheet_name,
                engine='openpyxl'
            )
            
            # Basic data cleaning
            data = self._clean_data(data)
            
            return data
            
        except ValueError as e:
            if "Excel file format cannot be determined" in str(e):
                raise Exception("Invalid Excel file format. Please upload a valid .xlsx or .xls file.")
            else:
                raise Exception(f"Error reading Excel file: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error loading Excel file: {str(e)}")
    
    def _clean_data(self, data):
        """
        Perform basic data cleaning operations
        
        Args:
            data: pandas.DataFrame
            
        Returns:
            pandas.DataFrame: Cleaned data
        """
        # Remove completely empty rows and columns
        data = data.dropna(how='all')
        data = data.dropna(axis=1, how='all')
        
        # Clean column names
        data.columns = data.columns.astype(str)
        data.columns = [col.strip() for col in data.columns]
        
        # Attempt to convert date columns
        data = self._convert_date_columns(data)
        
        # Convert numeric strings to numbers where possible
        data = self._convert_numeric_columns(data)
        
        return data
    
    def _convert_date_columns(self, data):
        """
        Attempt to convert columns that look like dates to datetime
        
        Args:
            data: pandas.DataFrame
            
        Returns:
            pandas.DataFrame: Data with converted date columns
        """
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if column might contain dates
                sample_values = data[col].dropna().head(10)
                
                if len(sample_values) > 0:
                    # Try to parse as datetime
                    try:
                        # Test with a few common date formats
                        test_conversion = pd.to_datetime(sample_values, errors='raise')
                        
                        # If successful, convert the entire column
                        data[col] = pd.to_datetime(data[col], errors='coerce')
                        
                    except (ValueError, TypeError):
                        # If conversion fails, leave as is
                        continue
        
        return data
    
    def _convert_numeric_columns(self, data):
        """
        Convert columns that look like numbers but are stored as strings
        
        Args:
            data: pandas.DataFrame
            
        Returns:
            pandas.DataFrame: Data with converted numeric columns
        """
        for col in data.columns:
            if data[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(data[col], errors='coerce')
                
                # If more than 50% of non-null values were successfully converted, use the numeric version
                non_null_original = data[col].notna().sum()
                non_null_converted = numeric_series.notna().sum()
                
                if non_null_original > 0 and (non_null_converted / non_null_original) > 0.5:
                    data[col] = numeric_series
        
        return data
    
    def get_column_types(self, data):
        """
        Analyze and categorize column types
        
        Args:
            data: pandas.DataFrame
            
        Returns:
            dict: Dictionary with column type information
        """
        column_info = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'text': []
        }
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                column_info['numeric'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                column_info['datetime'].append(col)
            elif data[col].dtype == 'object':
                # Distinguish between categorical and text based on unique values
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio < 0.1:  # Less than 10% unique values suggests categorical
                    column_info['categorical'].append(col)
                else:
                    column_info['text'].append(col)
        
        return column_info
    
    def get_data_summary(self, data):
        """
        Generate a comprehensive data summary
        
        Args:
            data: pandas.DataFrame
            
        Returns:
            dict: Summary statistics and information
        """
        summary = {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum(),
            'column_types': self.get_column_types(data)
        }
        
        return summary
