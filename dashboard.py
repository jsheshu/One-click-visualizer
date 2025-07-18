import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
import altair as alt

warnings.filterwarnings('ignore')

st.set_page_config(page_title='One-Click Data Visualizer',page_icon=':bar_chart:',layout='wide')

st.title(':pager: One-Click Data Visualizer', )
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

#=== 0. Setting up Session state

if 'df' not in st.session_state:
    default_csv = "superstore.csv"
    if os.path.exists(default_csv):
        st.session_state.df = pd.read_csv(default_csv)
        st.session_state.original_df = st.session_state.df.copy()
        st.session_state.source = 'default'
    else:
        st.session_state.df = pd.DataFrame()  # Empty DF if no CSV
        st.session_state.original_df = st.session_state.df.copy()
        st.session_state.source = 'none'


#=== 1. File upload --- override session_state.df if user uploads ===#
f1 = st.file_uploader(":file_folder: Upload a file (.csv, .xlsx, .xls, .txt)", type=(['csv','xlsx','xls','txt']))
if f1 is not None:
    filename = f1.name
    if filename.endswith('.csv'):
        df_new = pd.read_csv(f1)
        st.session_state.df = df_new
        st.session_state.source = 'uploaded'
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        df_new = pd.read_excel(f1)
        st.session_state.df = df_new
        st.session_state.source = 'uploaded'
    elif filename.endswith('.txt'):
        df_new = pd.read_csv(f1, delimiter = '\t')
        st.session_state.df = df_new
        st.session_state.source = 'uploaded'
    else:
        st.error("Unsupported file type, loading default data")
        default_csv = "superstore.csv"
        if os.path.exists(default_csv):
            st.session_state.df = pd.read_csv(default_csv)
            st.session_state.source = 'default'
        else:
            st.session_state.df = pd.DataFrame()
            st.session_state.source = 'none'


## Replacing the table
if f1 is not None and st.session_state.source == 'uploaded':
    st.write("### Preview of your uploaded data:")
else:
    st.write("### Sample data (superstore.csv):")
st.dataframe(st.session_state.df, width = 1100, height = 300)


if "adv_expanded" not in st.session_state:
    st.session_state.adv_expanded = False
if "show_advanced_clean_success" not in st.session_state:
    st.session_state.show_advanced_clean_success = False
if "show_reset_success" not in st.session_state:
    st.session_state.show_reset_success = False


st.write("### Step 1: Clean the data :broom:")
st.write('Choose **Quick clean** for dropping duplicates and null values')



if 'show_quick_clean_success' not in st.session_state:
    st.session_state.show_quick_clean_success = False
if 'quick_cleaned_df' not in st.session_state:
    st.session_state.quick_cleaned_df = None

if st.button('Quick clean'):
    df_cleaned = st.session_state.df.drop_duplicates().dropna().copy()
    before = len(st.session_state.df)
    after = len(df_cleaned)
    dropped = before - after
    st.session_state.df = df_cleaned  # Optionally update the main df here
    st.session_state.quick_cleaned_df = df_cleaned  # Save for preview display
    st.session_state.quick_clean_dropped_rows = dropped
    st.session_state.show_quick_clean_success = True
    st.session_state.last_cleaning = 'quick'
    st.rerun()

st.write("**OR**")
st.write("Choose Advance cleaning options for particular uses")

with st.expander('Advance Cleaning', expanded = st.session_state.adv_expanded):
  
    drop_dups = st.checkbox('Drop duplicates')
    

    fill_na = st.checkbox('Fill missing values')
    
    if fill_na:
        fill_method = st.radio(label = 'Numerical columns with:', options=['Mean', 'Median'])

    
    df_cleaned = st.session_state.df

    if drop_dups:
        df_cleaned = df_cleaned.drop_duplicates()
 
    if fill_na and fill_method:
        num_cols = df_cleaned.select_dtypes(include=np.number).columns
        if fill_method == 'Mean':
            for col in  num_cols:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
        elif fill_method == 'Median':
            for col in num_cols:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    remove_outliers = st.checkbox('Remove outliers',)

    if remove_outliers:
        st.warning("Warning: Remove outliers will drop more rows each time you click Apply. This operation is cumulative!")
        num_cols = df_cleaned.select_dtypes(include=np.number).columns
        for col in num_cols:
            m = df_cleaned[col].mean()
            s = df_cleaned[col].std()
            # Remove rows where z-score > 3 (common outlier threshold)
            df_cleaned = df_cleaned[(df_cleaned[col].isna()) | (np.abs((df_cleaned[col] - m) / s) <= 3)]

    if st.button("Apply"):
        before = len(st.session_state.df)       # Number of rows before cleaning
        after = len(df_cleaned)                 # Number of rows after cleaning
        dropped = before - after
        st.session_state.df = df_cleaned.copy()
        st.session_state.advanced_cleaned_df = df_cleaned.copy()  # For preview (optional)
        st.session_state.advanced_clean_dropped_rows = dropped
        st.session_state.show_advanced_clean_success = True
        st.session_state.last_cleaning = 'advanced'
        st.rerun()

    if st.button("Reset to Original Data"):
        st.session_state.df = st.session_state.original_df.copy()
        st.session_state.show_advanced_clean_success = False
        st.session_state.show_reset_success = True
        st.rerun()

if st.session_state.get("show_advanced_clean_success"):
    st.success("✅ Advanced cleaning applied!")
    st.dataframe(st.session_state.advanced_cleaned_df)
    st.info(f"Rows after cleaning: {len(st.session_state.advanced_cleaned_df)}, dropped {st.session_state.advanced_clean_dropped_rows} row(s)")
    st.session_state.show_advanced_clean_success = False 


if st.session_state.get("show_quick_clean_success"):
    st.success(f"✅ Quick clean: removed {st.session_state.quick_clean_dropped_rows} rows (duplicates and NAs)")
    st.dataframe(st.session_state.quick_cleaned_df)
    st.session_state.show_quick_clean_success = False    

if st.session_state.get("show_reset_success"):
    st.success("🔄 DataFrame has been reset to original!")
    st.dataframe(st.session_state.df.head(6))
    st.info(f"Rows after reset: {len(st.session_state.df)}")
    st.session_state.show_reset_success = False


# Visualization

def auto_convert_numeric_strings(df):
    "Converts object columns to numeric if >90% parsable; returns names of columns converted"
    converted_cols = []
    for col in df.select_dtypes(include='object').columns:
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().mean() > 0.9:
                df[col] = converted
                converted_cols.append(col)
        except Exception:
            continue
    return converted_cols

def first_or_none(lst):
    return lst[0] if lst else None


st.write("### Step 2: Visualize the data :bar_chart:")

df = st.session_state.df.copy()

# -- Auto-convert numeric-looking object columns -- #
converted_cols = auto_convert_numeric_strings(df)
if converted_cols:
    st.info(f"Auto-converted columns to numeric: {', '.join(converted_cols)}")

numeric_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols = [
    col for col in df.select_dtypes(include='object').columns
    if 1 < df[col].nunique() < 25
]
any_cat_col = first_or_none(cat_cols)
any_num_col = first_or_none(numeric_cols)

# --- Try to find a likely datetime column --- #
dt_col = None
for col in df.columns:
    try:
        if 'date' in col.lower() or 'time' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
            test = pd.to_datetime(df[col], errors='coerce')
            if test.notna().mean() > 0.5:
                dt_col = col
                break
    except Exception:
        continue

# Visualization Trigger (Button)
if st.button("Visualize Data"):
    st.session_state.show_viz = True

if st.session_state.get("show_viz"):
    st.header("Automatic Data Visualizations")
    col1, col2 = st.columns(2, gap="small")

    # ---- LEFT COLUMN: Bar, Line ----
    with col1:

        # --- Bar Chart ---
        st.subheader("Bar Chart")
        if cat_cols and numeric_cols:
            x1 = st.selectbox("Bar: X (category)", cat_cols, key="bar_x")
            y1 = st.selectbox("Bar: Y (numeric)", numeric_cols, key="bar_y")
            bar_df = df.groupby(x1, dropna=False)[y1].sum().reset_index()
            bar_chart = alt.Chart(bar_df).mark_bar(size=20).encode(
                x=alt.X(x1, sort='-y', title=x1),
                y=alt.Y(y1, title=y1)
            ).properties(width=400, height=300)
            st.altair_chart(bar_chart, use_container_width=False)
        else:
            st.info("⚠️ Need one categorical column (<25 unique values) and one numeric column for a bar chart.")

        # --- Line Chart ---
        st.subheader("Line Chart")
        if dt_col and numeric_cols:
            x2 = dt_col
            y2 = st.selectbox("Line: Y (numeric)", numeric_cols, key="line_y")
            line_df = df[[x2, y2]].dropna()
            line_df[x2] = pd.to_datetime(line_df[x2], errors='coerce')
            line_df = line_df.dropna().sort_values(x2)
            if not line_df.empty:
                line_chart = alt.Chart(line_df).mark_line().encode(
                    x=alt.X(x2, title=x2),
                    y=alt.Y(y2, title=y2)
                ).properties(width=600, height=350)
                st.altair_chart(line_chart, use_container_width=False)
            else:
                st.info("⚠️ Not enough valid date/numeric data for line chart.")
        elif cat_cols and numeric_cols:
            x2 = st.selectbox("Line: X (category)", cat_cols, key="line_x")
            y2 = st.selectbox("Line: Y (numeric, #2)", numeric_cols, key="line_y2")
            line_df = df.groupby(x2, dropna=False)[y2].sum().reset_index()
            line_chart = alt.Chart(line_df).mark_line().encode(
                x=alt.X(x2, title=x2),
                y=alt.Y(y2, title=y2)
            ).properties(width=600, height=350)
            st.altair_chart(line_chart, use_container_width=False)
        else:
            st.info("⚠️ Need a date or category column and a numeric column for line chart.")

    # ---- RIGHT COLUMN: Scatter, Histogram ----
    with col2:

        # --- Scatter Plot ---
        st.subheader("Scatter Plot")
        if len(numeric_cols) >= 2:
            scatter_cols = st.multiselect(
                "Scatter: Choose two numeric columns (X, Y)",
                numeric_cols, default=numeric_cols[:2], key="scatter_cols"
            )
            if len(scatter_cols) == 2:
                x3, y3 = scatter_cols
                scatter_chart = alt.Chart(df.dropna(subset=[x3, y3])).mark_point(size=40, opacity=0.6).encode(
                    x=alt.X(x3, title=x3),
                    y=alt.Y(y3, title=y3)
                ).properties(width=600, height=400)
                st.altair_chart(scatter_chart, use_container_width=False)
            else:
                st.info("Please select two numeric columns for scatter plot.")
        else:
            st.info("⚠️ Need at least two numeric columns for scatter plot.")

        # --- Histogram ---
        st.subheader("Histogram")
        if numeric_cols:
            h1 = st.selectbox("Histogram: X (numeric column)", numeric_cols, key="hist_x")
            # Altair histogram for full size control
            hist_chart = alt.Chart(df[[h1]].dropna()).mark_bar(size=8).encode(
                x=alt.X(h1, bin=alt.Bin(maxbins=20), title=h1),
                y=alt.Y('count()', title='Count')
            ).properties(width=600, height=400)
            st.altair_chart(hist_chart, use_container_width=False)
        else:
            st.info("⚠️ No numeric columns available for histogram.")

