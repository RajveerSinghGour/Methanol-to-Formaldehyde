import numpy as np
import pandas as pd
import os

# Note: You will need to have the 'pandas' and 'openpyxl' libraries installed for this to work with .xlsx files.
# If you don't have them, run: pip install pandas openpyxl

def load_data(selection='ALL'):
    """
    Loads experimental data directly from the local Excel (.xlsx) file.
    
    The function extracts the following data:
    1. xexp: A 2D array of Inlet Partial Pressures for MeOH (P_MeOH_in) and O2 (P_O2_in).
    2. rate_exp: A 1D array of the reaction rate R_HCHO (Rate of Formaldehyde formation).
    3. temp_exp: A 1D array of the experimental temperature (Temp).
    
    :param selection: Ignored in the current implementation (always loads full data).
    
    Returns:
        tuple: (xexp, rate_exp, temp_exp, None) 
    """
    
    primary_file_path = 'dataset.xlsx'
    fallback_csv_path = 'data.csv'

    file_path = None

    if os.path.exists(primary_file_path):
        file_path = primary_file_path
    elif os.path.exists(fallback_csv_path):
        # If the user renamed it to a CSV, use the CSV reader
        return load_data_from_csv(fallback_csv_path)
    
    if file_path is None:
        print("Error: Data file not found. Please ensure one of the following files is present in the current directory:")
        print(f"- '{primary_file_path}' (Recommended, for Excel files)")
        print(f"- '{fallback_csv_path}' (If you converted it to CSV)")
        return np.array([]), np.array([]), np.array([]), None
        
    try:
        # Read the Excel sheet; adjust sheet_name if needed
        df = pd.read_excel(file_path, sheet_name='Sheet1', header=1)

        # Drop first column if it is an index / run number
        if df.columns[0].startswith('Unnamed') or df.columns[0] == 'Run No':
            df = df.drop(columns=df.columns[0], axis=1)

        required_columns = ['Temp', 'P_MeOH_in', 'P_O2_in', 'R_HCHO']
        if not all(col in df.columns for col in required_columns):
            print("Error: Required columns ('Temp', 'P_MeOH_in', 'P_O2_in', 'R_HCHO') not found in the Excel sheet.")
            print("Columns found:", list(df.columns))
            return np.array([]), np.array([]), np.array([]), None

        # ---- Robust numeric conversion (handles 'k1', units rows, etc.) ----
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop any rows where one of the required numeric fields is NaN
        before_rows = len(df)
        df_clean = df.dropna(subset=required_columns)
        after_rows = len(df_clean)

        if after_rows == 0:
            print("Error: After cleaning, no valid numeric data rows remain.")
            print("Check that 'Temp', 'P_MeOH_in', 'P_O2_in', 'R_HCHO' are numeric in your Excel file.")
            return np.array([]), np.array([]), np.array([]), None

        if after_rows < before_rows:
            print(f"Note: Dropped {before_rows - after_rows} non-numeric or incomplete rows from Excel data.")

        # --- Data Extraction and Formatting ---
        temp_exp = df_clean['Temp'].to_numpy(dtype=np.float64)
        xexp = df_clean[['P_MeOH_in', 'P_O2_in']].to_numpy(dtype=np.float64)
        rate_exp = df_clean['R_HCHO'].to_numpy(dtype=np.float64)

        print(f"Data loaded successfully from Excel file {file_path}. Total {after_rows} usable data points.")
        
        return xexp, rate_exp, temp_exp, None

    except Exception as e:
        print(f"An unexpected error occurred during Excel data loading: {e}")
        return np.array([]), np.array([]), np.array([]), None


def load_data_from_csv(file_path):
    """Helper function to load data if the user uses a CSV file instead of XLSX."""
    try:
        df = pd.read_csv(file_path, header=1, encoding='latin-1')

        if df.columns[0].startswith('Unnamed') or df.columns[0] == 'Run No':
            df = df.drop(columns=df.columns[0], axis=1)

        required_columns = ['Temp', 'P_MeOH_in', 'P_O2_in', 'R_HCHO']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Required columns not found in the CSV file '{file_path}'.")
            print("Columns found:", list(df.columns))
            return np.array([]), np.array([]), np.array([]), None

        # Robust numeric conversion for CSV as well
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        before_rows = len(df)
        df_clean = df.dropna(subset=required_columns)
        after_rows = len(df_clean)

        if after_rows == 0:
            print("Error: After cleaning, no valid numeric data rows remain in CSV.")
            return np.array([]), np.array([]), np.array([]), None

        if after_rows < before_rows:
            print(f"Note: Dropped {before_rows - after_rows} non-numeric or incomplete rows from CSV data.")

        temp_exp = df_clean['Temp'].to_numpy(dtype=np.float64)
        xexp = df_clean[['P_MeOH_in', 'P_O2_in']].to_numpy(dtype=np.float64)
        rate_exp = df_clean['R_HCHO'].to_numpy(dtype=np.float64)

        print(f"Data loaded successfully from CSV file {file_path}. Total {after_rows} usable data points.")
        return xexp, rate_exp, temp_exp, None

    except Exception as e:
        if "codec can't decode bytes" in str(e):
            print(f"Error: Failed to decode CSV file using 'latin-1' encoding. The file may use a different encoding. Original error: {e}")
        else:
            print(f"An unexpected error occurred during CSV data loading: {e}")
        return np.array([]), np.array([]), np.array([]), None
