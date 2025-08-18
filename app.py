import streamlit as st
from streamlit.runtime.caching import cache_data, cache_resource
import pandas as pd
import polars as pl

st.title("Data Refiner App")
st.write("#### (by Asif Noushad)")

# Tabs for better UI
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Loading",
    "Data Preview",
    "Data Cleaning",
    "Data Profiling",
    "File Download",
    "Machine Learning"
])

############################
# Data Loading Tab
############################
with tab1:
    st.header("Data Loading")

    # Streamlit widget that lets to upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file for lazy loading:", type=["csv"])

    if uploaded_file is not None:
        # Optimisation: Detects a new upload (or first time). Only then (re)initialize session state. 
        # new_upload is true only when it's a new file. Thus improves efficiency
        new_upload = (
            "df_original" not in st.session_state
            or st.session_state.get("upload_name") != uploaded_file.name
            or st.session_state.get("upload_size") != getattr(uploaded_file, "size", None)
        )

        # Runs when new_upload is true --> When a new file is uploaded. Infers schema only after reading 2 lakh rows or less.
        if new_upload:
            try:
                # Lazy load first for preview (first 5 rows) --> Faster scanning
                df_lazy = pl.scan_csv(uploaded_file)
                preview_df = df_lazy.head(5).collect().to_pandas()
                engine = "polars"
            except Exception:
                st.warning("Polars failed, switching to Pandas...")
                uploaded_file.seek(0)
                preview_df = pd.read_csv(uploaded_file, nrows=5)
                engine = "pandas"

            st.session_state["upload_name"] = uploaded_file.name
            st.session_state["upload_size"] = getattr(uploaded_file, "size", None)

            st.success(f"File '{uploaded_file.name}' loaded lazily using {engine.capitalize()}.")
            
            st.markdown("--- \n")
            
            st.write("Lazy Preview (5 rows):")
            st.dataframe(preview_df)
            
            st.markdown("--- \n")
            # Button to load full dataset
            st.info("Please load the full dataset to proceed with data cleaning and profiling:")
            if st.button("📥 Load Full Dataset for Cleaning"):
                try:
                    uploaded_file.seek(0)
                    try:
                        with st.spinner("⏳ Loading full dataset... please wait"):
                            df_original = pl.read_csv(uploaded_file, infer_schema_length=200000)
                            engine_full = "polars"
                            st.success(f"✅ Full dataset loaded using {engine_full.capitalize()}!")
                    except Exception:
                        uploaded_file.seek(0)
                        df_original = pd.read_csv(uploaded_file)
                        engine_full = "pandas"

                    # df_original is saved to dictionary. Copied to df_cleaned for all purposes. 
                    st.session_state["df_original"] = df_original
                    st.session_state["df_cleaned"] = (
                        df_original.clone() if isinstance(df_original, pl.DataFrame) else df_original.copy()
                    )
                    st.success(f"Dataframe Shape: {df_original.shape}")
                except Exception as e:
                    st.error(f"Failed to load full dataset: {e}")

        else:
            # Not a new file, re-use session_state. Show shape of already existing file.
            df_original = st.session_state.get("df_original", None)
            engine_full = "Polars" if isinstance(df_original, pl.DataFrame) else "Pandas"
            if df_original is not None:
                st.success(f"Full dataset already loaded using {engine_full}. Shape: {df_original.shape}")
                st.caption("Tip: Data is kept in memory and won't reset unless you upload a different file.")
            else:
                st.info("File uploaded, but full dataset not loaded yet. Click 'Load Full Dataset for Cleaning' to proceed.")

        # Optional: a manual reset button --> here we can manually overwrite df_cleaned to df_original
        st.markdown("\n --- \n")  # Thin horizontal line
        st.error("Drop all applied cleaning operations and reload the original file here.")
        if st.button("Reset Data") and "df_original" in st.session_state:
            base = st.session_state["df_original"]
            st.session_state["df_cleaned"] = base.clone() if isinstance(base, pl.DataFrame) else base.copy()
            st.success("Cleaned data reset to original.")
    else:
        st.info("Please upload a CSV file to proceed.")

############################
# Data Preview Tab
############################
with tab2:
    st.header("Data Preview")
    
    #Checks if a file is uploaded and df_cleaned is in session. Else asks to upload a file.
    if "df_cleaned" in st.session_state:
        df = st.session_state["df_cleaned"]
        
        st.info("You can preview upto 100 rows and understand the datatypes of all the columns in the dataframe.")
        #st.slider returns an int file and is saved to num_rows
        st.markdown("\n")
        num_rows = st.slider("Number of rows to preview", min_value=5, max_value=100, value=5)
        st.markdown("--- \n")
        #st.multiselect() returns a python list and is saved to col_to_show
        cols_to_show = st.multiselect("Select columns to preview", options=df.columns, default=df.columns[:5])
        
      #Only if cols_to_show is not an empty list, it enters the if loop.
        if cols_to_show:
            if isinstance(df, pl.DataFrame):
                st.dataframe(df.select(cols_to_show).head(num_rows).to_pandas())
                st.markdown("--- \n")
                st.write("Data types for selected columns:")
                st.write({col: str(df[col].dtype) for col in cols_to_show})
            else:
                st.dataframe(df[cols_to_show].head(num_rows))
                st.markdown("--- \n")
                st.write("Data types for selected columns:")
                st.write({col: str(df[col].dtype) for col in cols_to_show})
        else:
            st.markdown("--- \n")
            st.warning("Select at least one column to preview.")
    else:
        st.info("Please load the full dataset via the 'Data Loading' tab.")


    
############################
# Data Cleaning Tab
############################
with tab3:
    st.header("Data Cleaning")
    if "df_cleaned" not in st.session_state:
        st.info("Please upload the full dataset first in the 'Data Loading' tab.")
    else:
        # Always use the latest cleaned data
        df = st.session_state["df_cleaned"]
        
        st.info("You can drop NaNs, remove duplicates, type cast, normalise and filter your data here.")
        st.markdown("\n")
        operation = st.selectbox(
            "Select Cleaning Operation",
            ["Select Cleaning Operation", "Drop NaNs", "Remove Duplicates", "Type Conversion", "Normalisation", "Filtering"]
        )
        st.markdown("--- \n")

############Dropping NaNs
        #For removing Not a Number Values, Missing Values, Undefined Values etc
        if operation == "Drop NaNs":
            st.subheader("Drop NaNs per Column")

            # Polars: df.null_count(0) gives count of missing values per column
            #Pandas: df.isna().sum() gives count of missing values per column
            #nan_counts get the missing value count
            nan_counts = df.null_count().to_pandas() if isinstance(df, pl.DataFrame) else df.isna().sum()
            st.write("**Missing Values per Column**")
            #Creates an empty placeholder which can be filled with data and later updated
            missing_placeholder = st.empty()
            missing_placeholder.dataframe(nan_counts)

            st.markdown("--- \n")
            
            #Column based missing value removal. Here, column subsets are given and then removed accordingly.
            cols_to_dropna = st.multiselect("Select columns to drop NaNs from", options=df.columns)
            
            st.markdown("\n \n")
            
            #drop_nulls(0) for polars and dropna() for pandas to drop the columns
            if st.button("Apply Drop NaNs"):
                if cols_to_dropna:
                    if isinstance(df, pl.DataFrame):
                        df = df.drop_nulls(subset=cols_to_dropna)
                    else:
                        df = df.dropna(subset=cols_to_dropna)
                    
                    st.session_state["df_cleaned"] = df  # persist
                    st.success("Data updated ✅")

                    # Refresh status immediately from existing df and update the placeholder
                    refreshed = df.null_count().to_pandas() if isinstance(df, pl.DataFrame) else df.isna().sum()
                    missing_placeholder.dataframe(refreshed)
                else:
                    st.warning("Please select at least one column.")

############Removing Duplicates

        elif operation == "Remove Duplicates":
            st.subheader("Remove Duplicate Rows")

            # Recompute status from the latest df in every render
            #df.unique(0).shape() and df.duplicated().sum() gives the number of duplicates
            total_duplicates = (
                df.shape[0] - df.unique().shape[0] if isinstance(df, pl.DataFrame) else df.duplicated().sum()
            )
            #Creates an empty placeholder which can be filled with data and later updated
            dup_placeholder = st.empty()
            dup_placeholder.write(f"No. of duplicate rows: {total_duplicates}")
            
            #select columns to check for duplicates
            cols_for_dup = st.multiselect(
                "Select columns to check for duplicates. (leave it empty to remove the entire duplicate row.)",
                options=df.columns
            )
            st.markdown("--- \n")
            #Apply to remove duplicates using df.unique and df.drop_duplicates
            if st.button("Apply Remove Duplicates"):
                if isinstance(df, pl.DataFrame):
                    df = df.unique(subset=cols_for_dup) if cols_for_dup else df.unique()
                else:
                    df = df.drop_duplicates(subset=cols_for_dup if cols_for_dup else None)
                
                st.session_state["df_cleaned"] = df
                st.success("Data updated ✅")

                # Refresh status immediately from persisting df
                total_duplicates = (
                    df.shape[0] - df.unique().shape[0] if isinstance(df, pl.DataFrame) else df.duplicated().sum()
                )
                dup_placeholder.write(f"Total duplicate rows: {total_duplicates}")

############Type Conversion############

        elif operation == "Type Conversion":
            st.subheader("Type Conversion")
            
            #Shows drop down of the columns to convert type
            col_to_convert = st.selectbox("Select column to convert", options=df.columns)
            
            # Show live dtypes (always from latest df)
            st.write("**Current column data type:**", str(df[col_to_convert].dtype if isinstance(df, pl.DataFrame) else df[col_to_convert].dtype))
            
            st.markdown("--- \n")
            
            target_dtype = st.selectbox("Select target data type", ["int", "float", "string"])

            st.markdown("--- \n")
            
            if st.button("Apply Type Conversion"):
                try:
                    if isinstance(df, pl.DataFrame):
                        if target_dtype == "int":
                            df = df.with_columns(pl.col(col_to_convert).cast(pl.Int64))
                        elif target_dtype == "float":
                            df = df.with_columns(pl.col(col_to_convert).cast(pl.Float64))
                        elif target_dtype == "string":
                            df = df.with_columns(pl.col(col_to_convert).cast(pl.Utf8))
                    else:
                        if target_dtype == "int":
                            df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors="coerce").astype("Int64")
                        elif target_dtype == "float":
                            df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors="coerce")
                        elif target_dtype == "string":
                            df[col_to_convert] = df[col_to_convert].astype(str)

                    st.session_state["df_cleaned"] = df
                    st.success("Data updated ✅")
                    
                except Exception as e:
                    st.error(f"Failed to convert column: {e}")

#########Normalisation#############

        elif operation == "Normalisation":
            st.subheader("Normalisation")
            
            #Creates a list of numeric columns for normalisation - Both int and float accepted
            numeric_cols = [col for col in df.columns if 
                            (str(df[col].dtype).startswith("Int") or str(df[col].dtype).startswith("Float"))]
            
            if not numeric_cols:
                st.warning("No numeric columns found for normalisation.")
            else:
                #Asks user to select the column and type of normalisation using drop downs
                cols_to_normalise = st.multiselect("Select numeric columns to normalise", options=numeric_cols)
                st.markdown("--- \n")
                method = st.selectbox("Select normalisation method", ["Min-Max Scaling", "Z-Score Standardisation"])

                st.markdown("--- \n")
                
                if st.button("Apply Normalisation"):
                    try:
                        if cols_to_normalise:
                            #Polars: Calculates min, max, mean, std dev
                            if isinstance(df, pl.DataFrame):
                                for col in cols_to_normalise:
                                    col_min = df[col].min()
                                    col_max = df[col].max()
                                    col_mean = df[col].mean()
                                    col_std = df[col].std()

                                    #Calculates Min-Max Scaling & Z-Score Standardisation
                                    if method == "Min-Max Scaling":
                                        df = df.with_columns(((pl.col(col) - col_min) / (col_max - col_min)).alias(col))
                                    elif method == "Z-Score Standardisation":
                                        df = df.with_columns(((pl.col(col) - col_mean) / col_std).alias(col))
                            else:
                                #Pandas: Calculates min, max, mean, std dev and calculates Min-Max and Z-Score Standardisation
                                for col in cols_to_normalise:
                                    if method == "Min-Max Scaling":
                                        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                                    elif method == "Z-Score Standardisation":
                                        df[col] = (df[col] - df[col].mean()) / df[col].std()

                            st.session_state['df_cleaned'] = df
                            st.success("Normalisation applied ✅")
                        else:
                            st.warning("Please select at least one column.")
                    except Exception as e:
                        st.error(f"Failed to normalise: {e}")

###################Filtering############

        elif operation == "Filtering":
            st.subheader("Filtering")
        
            # Select column to filter using df.columns
            selected_col = st.selectbox("Select column to filter", df.columns)
        
            if selected_col:
                # Automatically detect type. Initialises is_numeric with False.
                is_numeric = False
                if isinstance(df, pl.DataFrame):
                    #For polars, we check schema dtypes
                    dtype = df.schema[selected_col]
                    is_numeric = dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                           pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                           pl.Float32, pl.Float64)
                else:  # Pandas: we check using is_numeric_dtype
                    is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])
        
                # NUMERIC COLUMN FILTERING: Enters the if loop only if is_numeric is True --> set above
                #Then, we calculate minimum, maximum values
                if is_numeric:
                    min_val = float(df[selected_col].min())
                    max_val = float(df[selected_col].max())
        
                    st.write(f"Min: {min_val}, Max: {max_val}")
        
                    st.markdown("--- \n")
                    
                    filter_type = st.radio("Filter type", ["Range filter", "Greater than", "Less than"])
                    
                    #Range filter --> step allows user to move slider in 1% increments
                    if filter_type == "Range filter":
                        range_min, range_max = st.slider(
                            "Select range",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            step=(max_val - min_val) / 100
                        )
                    
                    #Here, range_max is set to default max and the minimum value is taken as input
                    elif filter_type == "Greater than":
                        range_min = st.number_input("Minimum value", value=min_val)
                        range_max = max_val
                    
                    #Here, range_min is taken as the default value and maximum value is taken as input    
                    elif filter_type == "Less than":
                        range_min = min_val
                        range_max = st.number_input("Maximum value", value=max_val)
        
                    st.markdown("--- \n")
                    
                    #For Polars: .filter() is applied here --> it keeps only rows where condition holds
                    #For Pandas: df[selected_col] is taken as a pandas series of the column. 
                    #When range is applied, true/false bool is applied. 
                    #Then, the dataframe is updated with the required values
                    if st.button("Apply Filter"):
                        try:
                            if isinstance(df, pl.DataFrame):
                                df = df.filter((pl.col(selected_col) >= range_min) & (pl.col(selected_col) <= range_max))
                            else:
                                df = df[(df[selected_col] >= range_min) & (df[selected_col] <= range_max)]
        
                            st.session_state['df_cleaned'] = df
                            st.success("Filter applied successfully ✅")
                        except Exception as e:
                            st.error(f"Error during filtering: {e}")
        
                # TEXT / CATEGORICAL COLUMN FILTERING: is_numeric is false, then comes here
                #the datatype of 'selected_col' is not numeric, thus we load the below else loop
                #We then list the unique values in the particular non-numeric column. This, aids at filtering that string alone using specific text.
                else:
                    unique_vals = df[selected_col].unique().to_list() if isinstance(df, pl.DataFrame) else df[selected_col].unique().tolist()
        
                    st.markdown("--- \n")
                    #Creates three radio filters for non-numeric columns for user to select
                    filter_type = st.radio("Filter type", ["Select specific values", "Contains text", "Does not contain text"])
        
                    if filter_type == "Select specific values":
                        st.markdown("--- \n")
                        #st.multiselect() returns a python list which is saved in selected_vals. Returns empty list if nothing is selected
                        selected_vals = st.multiselect("Select values to keep", options=unique_vals)
                    elif filter_type in ["Contains text", "Does not contain text"]:
                        st.markdown("--- \n")
                        #st.text_input() returns a string if user types something/empty string
                        search_text = st.text_input("Enter text to search")
                    
                    st.markdown("--- \n")
                    
                    if st.button("Apply Filter"):
                        try:
                            if isinstance(df, pl.DataFrame):
                                if filter_type == "Select specific values":
                                    #.filter() filters only selected_vals from the selected_col
                                    df = df.filter(pl.col(selected_col).is_in(selected_vals))
                                #pl.col(selected_col) selects the column user selects
                                #.cast(pl.Utf8): converts selected column to string --> converts any numerical values also to string to enable search
                                #.contains(): searches for 'search_text' in the string.
                                #literal=True; considers user entry as it is and not as regex. --> (".") is taken as full stop itself and not as an indicator for any charactor as in regex
                                elif filter_type == "Contains text":
                                    df = df.filter(pl.col(selected_col).cast(pl.Utf8).str.contains(search_text, literal=True))
                                elif filter_type == "Does not contain text":
                                    #Uses not of the above operation to avoid selected string
                                    df = df.filter(~pl.col(selected_col).cast(pl.Utf8).str.contains(search_text, literal=True))
                            else:  # Pandas
                                if filter_type == "Select specific values":
                                    #.isin() function used
                                    df = df[df[selected_col].isin(selected_vals)]
                                elif filter_type == "Contains text":
                                    #.astype(0) used to convert the column to string
                                    #case = False; it makes it case insensitive
                                    #na = False; if NaN values are detected, treat it as not matching. If not, may result in error. 
                                    df = df[df[selected_col].astype(str).str.contains(search_text, case=False, na=False)]
                                elif filter_type == "Does not contain text":
                                    df = df[~df[selected_col].astype(str).str.contains(search_text, case=False, na=False)]
        
                            st.session_state['df_cleaned'] = df
                            st.success("Filter applied successfully ✅")
                        except Exception as e:
                            st.error(f"Error during filtering: {e}")


############################
# Data Profiling Tab
############################
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components #lets embed raw HTML into streamlit code

with tab4:
    st.header("Data Profiling")
    
    if 'df_cleaned' not in st.session_state:
        st.info("Please upload and clean a CSV file first.")
    else:
        df = st.session_state['df_cleaned']

        # Convert to pandas if polars
        if isinstance(df, pl.DataFrame):
            df_to_profile = df.to_pandas()
        else:
            df_to_profile = df

        st.write("Generate a report with 1000 random sample rows.")
        st.info("Note: Interactions, Duplicates, Text Profiling and certain correlation calculations are disabled to boost profiling speed.")
        
        # Checkbox for full dataset or sample
        full_profile = st.checkbox("Toggle to generate report using entire dataset (Slower...)", value=False)
        
        # Generate button
        st.markdown("--- \n \n")
        
        if st.button("Generate Profile"):
            # If full dataset is not requested and the profile contains > 1000 --> Then, sample 1000 rows at random 
            if not full_profile and len(df_to_profile) > 1000:
                df_sample = df_to_profile.sample(1000, random_state=42)
                st.info("Running profile on a random sample of 1000 rows.")
            else:
                df_sample = df_to_profile

            # Create profile
            with st.spinner("Generating profiling report..."):
                try:
                    #explorative = True --> Generates a detailed report
                    #Removed 'interactions', 'duplicates', text profiling, certain correlation calculations to increase speed
                    profile = ProfileReport(df_sample, 
                                            title="Data Profiling Report", 
                                            explorative=True, 
                                            correlations={
                                                "pearson": {"calculate": True},
                                                "spearman": {"calculate": True},
                                                "kendall": {"calculate": False},  
                                                "phi_k": {"calculate": False},    
                                                "cramers": {"calculate": False} },
                                            interactions={"continuous": False},
                                            duplicates={"head": 0},
                                            vars={"cat": {"length": False}} 
                                            )
                    
                    #Generated profile is converted to html string
                    profile_html = profile.to_html()
                    components.html(profile_html, height=1000, scrolling=True)
                except Exception as e:
                    st.error(f"Failed to generate profile: {e}")
                    
                    
############################
# File Download + Final Summary Tab
############################
with tab5:
    st.header("Cleaned File Exporter:")
    if "df_cleaned" in st.session_state:
        df = st.session_state["df_cleaned"]
        
        #Converts polars to pandas before exporting to CSV --> then does utf-8 encoding
        #index = False --> avoids polars/pandas index column to make the csv clean
        #mime = "text/csv" tells the file type to the browser
        st.info("Please download the cleaned CSV file here.")
        st.download_button(
            "Download CSV",
            data=(df.to_pandas().to_csv(index=False).encode("utf-8")
                  if isinstance(df, pl.DataFrame)
                  else df.to_csv(index=False).encode("utf-8")),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )
        st.markdown("--- \n")

#####Summary
        st.subheader("Summary of cleaned data:")
        st.write(f"**Number of rows:** {df.shape[0]}")
        st.write(f"**Number of columns:** {df.shape[1]}")
        st.markdown("\n")
        st.write("**Columns and Data Types:**")
        if isinstance(df, pl.DataFrame):
            dtype_map = {col: str(dtype) for col, dtype in df.schema.items()}
        else:
            dtype_map = {col: str(dt) for col, dt in df.dtypes.items()} if hasattr(df, "dtypes") and isinstance(df.dtypes, dict) else {col: str(df[col].dtype) for col in df.columns}
        st.table(list(dtype_map.items()))
    else:
        st.info("Please load the full dataset and clean the file first.")


# =======================
# Machine Learning (Standalone)
# =======================
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

with tab6:
    st.header("Random Forest Classifier Model")
    st.caption("Upload training dataset --> Pick Target --> Auto-detect numeric features --> Train --> Predict.")

    st.markdown("--- \n")
    # ---- 1) Upload TRAINING dataset (CSV or Excel) ----
    train_file = st.file_uploader("Upload TRAINING dataset (CSV or Excel)", type=["csv", "xlsx", "xls"], key="ml_train_upl")

    if train_file is not None:
        # Read with pandas only (this module is independent from cleaning pipeline)
        try:
            if train_file.name.lower().endswith(".csv"):
                train_df = pd.read_csv(train_file)
            else:
                # Requires openpyxl for .xlsx
                train_df = pd.read_excel(train_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            train_df = None
        
        st.markdown("--- \n")
        if train_df is not None and not train_df.empty:
            st.write("Preview (first 10 rows):")
            st.dataframe(train_df.head(10))

            st.markdown("--- \n")
            # ---- 2) Select target column ----
            target_col = st.selectbox("Select TARGET column (classification)", options=train_df.columns)

            # ---- 3) Auto-detect numeric FEATURES (exclude target) ----
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            # Just in case target is numeric, exclude it
            feature_cols = [c for c in numeric_cols if c != target_col]

            if not feature_cols:
                st.warning("No numeric feature columns found (excluding target). Please upload a dataset with numeric features.")
            else:
                st.markdown("--- \n")
                st.write("**Features auto-detected (numeric only):**")
                st.code(", ".join(feature_cols) if feature_cols else "None")

                # ---- Basic settings ----
                st.markdown("--- \n")
                with st.expander("Training Settings", expanded=True):
                    test_size_percent = st.slider("Test size (%)", 1, 99, 20, 1)  # percentage in UI
                    test_size = test_size_percent / 100  # convert to fraction internally
                    n_estimators = st.slider("Number of trees (n_estimators)", 50, 300, 100, 10)
                    max_depth = st.slider("Max depth (0 = None)", 0, 50, 0, 1)
                    random_state = st.number_input("Random state", value=42, step=1)

                # ---- 4) Train model ----
                st.markdown("--- \n")
                if st.button("Train Random Forest Classifier"):
                    # Build X, y
                    X = train_df[feature_cols].copy()
                    y = train_df[target_col].copy()

                    # Handle missing numerics with median imputation (simple & robust)
                    medians = X.median(numeric_only=True)
                    X = X.fillna(medians)

                    # If y has missing values, drop those rows
                    keep_mask = ~y.isna()
                    X = X.loc[keep_mask]
                    y = y.loc[keep_mask]

                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=int(random_state), stratify=y if y.nunique() > 1 else None
                    )

                    # Model
                    clf = RandomForestClassifier(
                        n_estimators=int(n_estimators),
                        max_depth=None if max_depth == 0 else int(max_depth),
                        random_state=int(random_state),
                        n_jobs=-1,
                    )
                    clf.fit(X_train, y_train)

                    # Evaluate
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    st.success(f"Model trained! ✅ Accuracy: **{acc:.4f}**")
                

                    # Persist model + metadata for prediction
                    st.session_state["ml_model"] = clf
                    st.session_state["ml_features"] = feature_cols
                    st.session_state["ml_medians"] = medians  # for imputing in inference
                    st.session_state["ml_target_name"] = target_col

    # ---- 5) Predict (only if model is trained) ----
    if all(k in st.session_state for k in ["ml_model", "ml_features", "ml_medians"]):
        st.markdown("---")
        st.subheader("Make Predictions")

        clf = st.session_state["ml_model"]
        feature_cols = st.session_state["ml_features"]
        medians = st.session_state["ml_medians"]
        target_name = st.session_state.get("ml_target_name", "prediction")

        # A) Batch prediction via file upload
        with st.expander("Batch Prediction: Upload CSV/Excel with the same feature columns", expanded=True):
            pred_file = st.file_uploader("Upload PREDICTION data (CSV/Excel, must contain feature columns)", type=["csv", "xlsx", "xls"], key="ml_pred_upl")
            if pred_file is not None:
                try:
                    if pred_file.name.lower().endswith(".csv"):
                        pred_df = pd.read_csv(pred_file)
                    else:
                        pred_df = pd.read_excel(pred_file)
                except Exception as e:
                    st.error(f"Failed to read prediction file: {e}")
                    pred_df = None

                if pred_df is not None and not pred_df.empty:
                    # Validate required columns
                    missing_feats = [c for c in feature_cols if c not in pred_df.columns]
                    if missing_feats:
                        st.error(f"Prediction file is missing required feature columns: {missing_feats}")
                    else:
                        Xp = pred_df[feature_cols].copy()
                        Xp = Xp.fillna(medians)  # same imputation as training

                        try:
                            preds = clf.predict(Xp)
                            proba = clf.predict_proba(Xp) if hasattr(clf, "predict_proba") else None

                            out = pred_df.copy()
                            out[target_name + "_pred"] = preds

                            st.write("Preview of predictions:")
                            st.dataframe(out.head(20))

                            # Download button
                            csv_bytes = out.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download Predictions CSV",
                                data=csv_bytes,
                                file_name="predictions.csv",
                                mime="text/csv",
                            )

                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
        
        st.markdown("--- \n")
        # B) Manual single-row prediction
        with st.expander("Manual Prediction: Enter feature values", expanded=False):
            # Build inputs with medians as defaults
            manual_vals = {}
            cols = st.columns(min(4, len(feature_cols)) or 1)
            for i, col_name in enumerate(feature_cols):
                default_val = float(medians.get(col_name, 0.0))
                with cols[i % len(cols)]:
                    manual_vals[col_name] = st.number_input(col_name, value=default_val)

            if st.button("Predict Single Row"):
                try:
                    row_df = pd.DataFrame([manual_vals], columns=feature_cols)
                    row_df = row_df.fillna(medians)
                    pred = clf.predict(row_df)[0]
                    st.success(f"Predicted class: **{pred}**")
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(row_df)[0]
                        prob_map = {str(cls): float(p) for cls, p in zip(clf.classes_, probs)}
                        st.write("Class probabilities:", prob_map)
                except Exception as e:
                    st.error(f"Manual prediction failed: {e}")
    else:
        with tab6:
            st.info("Train a model above to enable predictions.")