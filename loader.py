import duckdb
import pandas as pd
import os

def get_duckdb_connection():
    """
    Initialize DuckDB connection and load CSV data into tables
    """
    # Create in-memory DuckDB connection
    con = duckdb.connect(':memory:')
    
    # Define the data directory
    data_dir = "data"
    
    try:
        # Load sample_bo_tbl_large.csv as bo_df
        bo_path = os.path.join(data_dir, "sample_bo_tbl_large.csv")
        if os.path.exists(bo_path):
            con.execute(f"""
                CREATE TABLE bo_df AS 
                SELECT * FROM read_csv_auto('{bo_path}', header=true)
            """)
            print(f"✅ Loaded bo_df from {bo_path}")
        else:
            print(f"❌ File not found: {bo_path}")
        
        # Load sample_sub_details_large.csv as sub_df
        sub_path = os.path.join(data_dir, "sample_sub_details_large.csv")
        if os.path.exists(sub_path):
            con.execute(f"""
                CREATE TABLE sub_df AS 
                SELECT * FROM read_csv_auto('{sub_path}', header=true)
            """)
            print(f"✅ Loaded sub_df from {sub_path}")
        else:
            print(f"❌ File not found: {sub_path}")
        
        # Load sample_revenue_large.csv as rev_df
        rev_path = os.path.join(data_dir, "sample_revenue_large.csv")
        if os.path.exists(rev_path):
            con.execute(f"""
                CREATE TABLE rev_df AS 
                SELECT * FROM read_csv_auto('{rev_path}', header=true)
            """)
            print(f"✅ Loaded rev_df from {rev_path}")
        else:
            print(f"❌ File not found: {rev_path}")
        
        # Print table info for debugging
        print("\n📊 Available tables:")
        tables = con.execute("SHOW TABLES").fetchall()
        for table in tables:
            table_name = table[0]
            count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  - {table_name}: {count} rows")
            
            # Show sample data
            sample = con.execute(f"SELECT * FROM {table_name} LIMIT 2").fetchdf()
            print(f"    Columns: {list(sample.columns)}")
            print(f"    Sample data:\n{sample}")
            print()
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    return con

# Alternative function using pandas (if needed)
def load_data_as_pandas():
    """
    Load CSV files as pandas DataFrames
    """
    data_dir = "data"
    
    dfs = {}
    
    try:
        # Load each CSV
        bo_path = os.path.join(data_dir, "sample_bo_tbl_large.csv")
        if os.path.exists(bo_path):
            dfs['bo_df'] = pd.read_csv(bo_path)
            
        sub_path = os.path.join(data_dir, "sample_sub_details_large.csv")
        if os.path.exists(sub_path):
            dfs['sub_df'] = pd.read_csv(sub_path)
            
        rev_path = os.path.join(data_dir, "sample_revenue_large.csv")
        if os.path.exists(rev_path):
            dfs['rev_df'] = pd.read_csv(rev_path)
            
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    return dfs