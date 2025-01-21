import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
import xlsxwriter
from plotly.subplots import make_subplots
from scipy import stats  # For t-tests

##############################################################
#                PAGE CONFIGURATION & GLOBAL CSS
##############################################################
st.set_page_config(
    page_title="Movie Ratings Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎬"
)

st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        color: #333333;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f0f0f0;
    }
    /* Card styling */
    .metric-card {
        background: rgba(0, 123, 255, 0.1);
        border: 1px solid rgba(0, 123, 255, 0.3);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        position: relative;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    /* Tooltip styling */
    .metric-card:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }
    .tooltip {
        visibility: hidden;
        position: absolute;
        background-color: #333;
        color: white;
        padding: 5px 10px;
        border-radius: 4px;
        font-size: 12px;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        white-space: nowrap;
        z-index: 1000;
    }
    /* Metric text styling */
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #007bff;
    }
    .metric-label {
        font-size: 14px;
        color: #555555;
        margin-top: 5px;
    }
    /* Search box styling */
    .search-box {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin: 10px 0;
        width: 100%;
        background: white;
    }
    /* Export button styling */
    .export-button {
        background-color: #007bff;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        margin: 10px 0;
        transition: background-color 0.3s;
    }
    .export-button:hover {
        background-color: #0056b3;
    }
    /* Table styling */
    .dataframe {
        border: none !important;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    /* Tab styling with horizontal scrolling */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        overflow-x: auto;
        white-space: nowrap;
        scrollbar-width: thin; /* For Firefox */
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #333;
        padding: 10px 20px;
        border-radius: 5px;
        flex: none;
        white-space: nowrap;
        margin-right: 10px;
    }
    /* Custom scrollbar styling */
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 5px;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: #007bff;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
        background: #f0f0f0;
        border-radius: 10px;
    }
    /* Plot styling */
    .plot-container {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

##############################################################
#                     DATA LOADING & FUNCTIONS
##############################################################
@st.cache_data
def load_data():
    """Load and prepare data with additional metrics."""
    fandango = pd.read_csv("fandango_scrape.csv")
    all_sites = pd.read_csv("all_sites_scores.csv")

    # Extract year and calculate additional metrics
    fandango['YEAR'] = fandango['FILM'].str.extract(r'\((\d{4})\)').astype(float)
    fandango['RATING_CATEGORY'] = pd.cut(
        fandango['RATING'],
        bins=[0, 2, 3, 4, 5],
        labels=['Poor', 'Average', 'Good', 'Excellent']
    )

    # Merge and calculate normalized ratings
    df = pd.merge(fandango, all_sites, on='FILM', how='inner')
    df['RT_Normalized'] = df['RottenTomatoes'] / 20
    df['Metacritic_Normalized'] = df['Metacritic'] / 20
    df['IMDB_Normalized'] = df['IMDB'] / 2

    return fandango, all_sites, df

def generate_excel_download(df, sheet_name='Data'):
    """Generate formatted Excel file for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]

        # Format headers
        header_format = writer.book.add_format({
            'bold': True,
            'bg_color': '#007bff',
            'font_color': 'white'
        })

        # Auto-adjust columns
        for idx, col in enumerate(df.columns):
            series = df[col]
            max_len = max((
                series.astype(str).map(len).max(),
                len(str(series.name))
            )) + 1
            worksheet.set_column(idx, idx, max_len)
            worksheet.write(0, idx, col, header_format)

    output.seek(0)
    return output

def create_download_link(df, filename, text):
    """Create a styled download link for Excel files."""
    excel_file = generate_excel_download(df)
    b64 = base64.b64encode(excel_file.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="export-button">{text}</a>'
    return href

##############################################################
#                          MAIN APP
##############################################################
fandango, all_sites, merged_df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎬 Analysis Controls")

# 1. Help Button
if "show_help" not in st.session_state:
    st.session_state["show_help"] = False

def toggle_help():
    st.session_state["show_help"] = not st.session_state["show_help"]

if st.sidebar.button("❓ Help", on_click=toggle_help):
    pass

if st.session_state["show_help"]:
    st.markdown("""
    ### Help Documentation
    - **Navigation**: Use the tabs at the top of the page to explore different analyses.
    - **Filters**: Interact with the sidebar sliders and text inputs to filter the data (e.g., by rating or year).
    - **Exports**: Use the buttons in the *Data Explorer* tab or the *Analysis Report* button to download Excel/CSV files.
    - **Search**: In the *Data Explorer* tab, you can search for specific movie titles.
    """)

# 2. Enhanced Search Feature
search_query = st.sidebar.text_input(
    "🔍 Quick Movie Search",
    help="Type a partial movie title, year, or rating to filter below"
)

if search_query:
    # Create a copy for further assignments
    filtered_movies = fandango[
        fandango['FILM'].str.contains(search_query, case=False, na=False) |
        fandango['YEAR'].astype(str).str.contains(search_query, na=False) |
        fandango['RATING'].astype(str).str.contains(search_query, na=False)
    ].copy()

    st.sidebar.write(f"Found {len(filtered_movies)} movies:")
    st.sidebar.dataframe(
        filtered_movies[['FILM', 'YEAR', 'RATING', 'VOTES']],
        use_container_width=True
    )

# 3. Year Range Filter
years = sorted(fandango['YEAR'].dropna().unique())
if len(years) > 0:
    year_range = st.sidebar.select_slider(
        "Select Year Range",
        options=years,
        value=(min(years), max(years)),
        help="Filter movies by release year"
    )
else:
    year_range = (0, 0) # fallback if no data

# 4. Rating Filter
rating_range = st.sidebar.slider(
    "Filter by Rating Range (Fandango)",
    0.0, 5.0, (0.0, 5.0),
    help="Filter movies by Fandango rating"
)

# 5. Platform Selection
platforms = st.sidebar.multiselect(
    "Select Platforms to Compare",
    ["Fandango", "Rotten Tomatoes", "Metacritic", "IMDB"],
    default=["Fandango", "Rotten Tomatoes"],
    help="Choose platforms for comparison analysis"
)

# 6. Export Options
st.sidebar.markdown("### Export Options")
if st.sidebar.button("Generate Analysis Report"):
    # Create comprehensive summary
    summary_data = {
        'Metric': [
            'Total Movies',
            'Average Rating',
            'Rating Inflation (Stars - Rating)',
            'Rating-Vote Correlation',
            'Most Common Rating',
            'Highest Rated Movie',
            'Most Voted Movie'
        ],
        'Value': [
            len(fandango),
            round(fandango['RATING'].mean(), 2),
            round((fandango['STARS'] - fandango['RATING']).mean(), 2),
            round(fandango['RATING'].corr(fandango['VOTES']), 2),
            fandango['RATING'].mode()[0],
            fandango.loc[fandango['RATING'].idxmax(), 'FILM'] if not fandango.empty else "N/A",
            fandango.loc[fandango['VOTES'].idxmax(), 'FILM'] if not fandango.empty else "N/A"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.sidebar.markdown(
        create_download_link(
            summary_df,
            'movie_analysis_report.xlsx',
            '📥 Download Analysis Report'
        ),
        unsafe_allow_html=True
    )

##############################################################
#                      MAIN PAGE TITLE
##############################################################
st.title("🎬 Movie Metrics Dashboard")
st.markdown("### Investigating Online Review Bias with Interactive Analysis")

##############################################################
#                      TOP METRICS
##############################################################
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="tooltip">Total number of movies analyzed in the dataset</div>
        <div class="metric-value">{len(fandango)}</div>
        <div class="metric-label">Movies Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_rating = round(fandango['RATING'].mean(), 2)
    st.markdown(f"""
    <div class="metric-card">
        <div class="tooltip">Average Fandango rating across all movies</div>
        <div class="metric-value">{avg_rating}</div>
        <div class="metric-label">Average Rating</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    inflation = round((fandango['STARS'] - fandango['RATING']).mean(), 2)
    st.markdown(f"""
    <div class="metric-card">
        <div class="tooltip">Average difference between displayed stars and actual rating</div>
        <div class="metric-value">+{inflation}</div>
        <div class="metric-label">Rating Inflation</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    correlation = round(fandango['RATING'].corr(fandango['VOTES']) * 100, 1)
    st.markdown(f"""
    <div class="metric-card">
        <div class="tooltip">Correlation between ratings and number of votes</div>
        <div class="metric-value">{correlation}%</div>
        <div class="metric-label">Rating-Vote Correlation</div>
    </div>
    """, unsafe_allow_html=True)

##############################################################
#                   FILTERED DATASET
##############################################################
# Safely create copy after slicing to avoid SettingWithCopyWarning
year_min, year_max = year_range
df_time = fandango[
    (fandango['YEAR'] >= year_min) &
    (fandango['YEAR'] <= year_max)
].copy()

df_time = df_time[
    (df_time['RATING'] >= rating_range[0]) &
    (df_time['RATING'] <= rating_range[1])
].copy()

##############################################################
#                     TABS ORGANIZATION
##############################################################
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Rating Analysis",
    "Platform Comparison",
    "Temporal Analysis",
    "Advanced Insights",
    "Data Explorer",
    "Key Findings"
])

##############################################################
#   TAB 1: RATING ANALYSIS
##############################################################
with tab1:
    st.header("Rating Distribution Analysis")

    # Interactive control for visualization type
    viz_type = st.radio(
        "Select Visualization Type",
        ["Distribution", "Box Plot", "Scatter Plot"],
        horizontal=True
    )

    col_left, col_right = st.columns(2)

    with col_left:
        if viz_type == "Distribution":
            fig_dist = px.histogram(
                df_time,
                x='RATING',
                nbins=20,
                title="Fandango Ratings Distribution (Filtered)",
                color_discrete_sequence=['#3498db']
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        elif viz_type == "Box Plot":
            fig_box = px.box(
                df_time,
                y='RATING',
                title="Rating Distribution Box Plot (Filtered)"
            )
            st.plotly_chart(fig_box, use_container_width=True)

        else:  # Scatter Plot
            fig_scatter = px.scatter(
                df_time,
                x='VOTES',
                y='RATING',
                hover_data=['FILM'],
                title="Ratings vs Number of Votes (Filtered)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.markdown("#### Distribution of Rating Categories")
        rating_cats = df_time['RATING_CATEGORY'].value_counts()
        fig_pie = px.pie(
            values=rating_cats.values,
            names=rating_cats.index,
            title="Rating Categories (Filtered)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Statistical summary
    st.subheader("Statistical Summary of Filtered Data")
    if not df_time.empty:
        colA, colB = st.columns(2)
        with colA:
            st.metric("Median Rating", f"{df_time['RATING'].median():.2f}")
            st.metric("Standard Deviation", f"{df_time['RATING'].std():.2f}")
        with colB:
            mode_val = df_time['RATING'].mode()
            mode_val = mode_val[0] if not mode_val.empty else "N/A"
            st.metric("Mode Rating", f"{mode_val}")
            st.metric(
                "Rating Range",
                f"{df_time['RATING'].max() - df_time['RATING'].min():.2f}"
            )
    else:
        st.warning("No movies match the chosen filters.")

##############################################################
#   TAB 2: PLATFORM COMPARISON
##############################################################
with tab2:
    st.header("Cross-Platform Analysis")

    platform_compare = st.selectbox(
        "Select Platform to Compare with Fandango",
        ["Rotten Tomatoes", "Metacritic", "IMDB"]
    )

    colA, colB = st.columns(2)
    with colA:
        # Scatter plot comparison
        if platform_compare == "Rotten Tomatoes":
            comparison_data = merged_df[['FILM', 'RATING', 'RT_Normalized']].copy()
            x_col = 'RT_Normalized'
            title = "Fandango vs Rotten Tomatoes"
        elif platform_compare == "Metacritic":
            comparison_data = merged_df[['FILM', 'RATING', 'Metacritic_Normalized']].copy()
            x_col = 'Metacritic_Normalized'
            title = "Fandango vs Metacritic"
        else:
            comparison_data = merged_df[['FILM', 'RATING', 'IMDB_Normalized']].copy()
            x_col = 'IMDB_Normalized'
            title = "Fandango vs IMDB"

        fig_comp = px.scatter(
            comparison_data,
            x=x_col,
            y='RATING',
            hover_data=['FILM'],
            title=title
        )
        # Add diagonal reference line
        fig_comp.add_shape(
            type="line",
            x0=0, y0=0,
            x1=5, y1=5,
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with colB:
        # Rating difference distribution
        diff_col = f'Diff_{platform_compare.replace(" ", "_")}'
        # Safe to assign now that we have a copy
        comparison_data[diff_col] = comparison_data['RATING'] - comparison_data[x_col]
        fig_diff = px.histogram(
            comparison_data,
            x=diff_col,
            title=f"Rating Differences Distribution ({platform_compare}) (Fandango - {platform_compare})",
            color_discrete_sequence=['#2ecc71']
        )
        st.plotly_chart(fig_diff, use_container_width=True)

    # Statistical comparison
    st.subheader("Rating Comparison Statistics")
    stats_df = pd.DataFrame({
        'Platform': ['Fandango', platform_compare],
        'Mean Rating': [comparison_data['RATING'].mean(),
                        comparison_data[x_col].mean()],
        'Median Rating': [comparison_data['RATING'].median(),
                          comparison_data[x_col].median()],
        'Std Dev': [comparison_data['RATING'].std(),
                    comparison_data[x_col].std()]
    })
    st.dataframe(stats_df.round(2), use_container_width=True)

##############################################################
#   TAB 3: TEMPORAL ANALYSIS
##############################################################
with tab3:
    st.header("Temporal Trends Analysis")

    # Additional time-based filtering
    year_options = fandango['YEAR'].dropna().unique()
    year_min_sel, year_max_sel = st.select_slider(
        "Select Time Period",
        options=sorted(year_options) if len(year_options) > 0 else [0],
        value=(
            min(year_options) if len(year_options) > 0 else 0,
            max(year_options) if len(year_options) > 0 else 0
        )
    )

    time_data = fandango[
        (fandango['YEAR'] >= year_min_sel) &
        (fandango['YEAR'] <= year_max_sel)
    ].copy()

    colX, colY = st.columns(2)
    with colX:
        # Yearly average rating & count
        yearly_avg = time_data.groupby('YEAR')['RATING'].agg(['mean','count']).reset_index()
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(
            go.Scatter(
                x=yearly_avg['YEAR'],
                y=yearly_avg['mean'],
                name="Average Rating",
                line=dict(color="#1f77b4")
            ),
            secondary_y=False
        )
        fig_trend.add_trace(
            go.Bar(
                x=yearly_avg['YEAR'],
                y=yearly_avg['count'],
                name="Number of Movies",
                marker_color="rgba(158,202,225,0.4)"
            ),
            secondary_y=True
        )
        fig_trend.update_layout(
            title="Yearly Trends: Average Ratings and Movie Count",
            xaxis_title="Year",
            yaxis_title="Average Rating",
            yaxis2_title="Number of Movies"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with colY:
        # Box plot across years
        if not time_data.empty:
            fig_box_year = px.box(
                time_data,
                x='YEAR',
                y='RATING',
                title="Rating Distribution by Year"
            )
            st.plotly_chart(fig_box_year, use_container_width=True)
        else:
            st.warning("No data for selected years.")

    # Trend Analysis Table
    st.subheader("Trend Analysis Table")
    if not time_data.empty:
        trend_stats = time_data.groupby('YEAR').agg({
            'RATING': ['mean','std','count'],
            'VOTES': 'mean'
        }).round(2)
        trend_stats.columns = ['Avg Rating','Rating Std','Movie Count','Avg Votes']
        st.dataframe(trend_stats, use_container_width=True)
    else:
        st.info("No data to display for the selected time range.")

##############################################################
#   TAB 4: ADVANCED INSIGHTS
##############################################################
with tab4:
    st.header("Advanced Analysis & Insights")
    # Subtabs for structured deeper dives
    insight_tab1, insight_tab2, insight_tab3 = st.tabs([
        "Rating Anomalies",
        "Platform Bias Analysis",
        "Statistical Deep Dive"
    ])

    ##########################################################
    # SUBTAB: RATING ANOMALIES
    ##########################################################
    with insight_tab1:
        st.subheader("Rating Anomaly Detection")
        # Safe to create a copy if we plan to add new columns
        merged_cpy = merged_df.copy()
        merged_cpy['rating_zscore'] = (
            merged_cpy['RATING'] - merged_cpy['RATING'].mean()
        ) / merged_cpy['RATING'].std()

        anomalies = merged_cpy[abs(merged_cpy['rating_zscore']) > 2]

        fig_anomaly = px.scatter(
            merged_cpy,
            x='VOTES',
            y='RATING',
            color='rating_zscore',
            hover_data=['FILM'],
            title="Rating Anomalies Detection (Z-score > 2 or < -2)",
            color_continuous_scale='RdYlBu'
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)

        st.write(f"Found {len(anomalies)} anomalous ratings based on Z-score > 2 or < -2.")
        if not anomalies.empty:
            st.dataframe(
                anomalies[['FILM','RATING','VOTES','rating_zscore']].sort_values('rating_zscore', ascending=False),
                use_container_width=True
            )

    ##########################################################
    # SUBTAB: PLATFORM BIAS ANALYSIS
    ##########################################################
    with insight_tab2:
        st.subheader("Platform Bias Analysis")

        # Use a copy if adding columns or modifying
        merged_copy_bias = merged_df.copy()
        bias_metrics = pd.DataFrame({
            'Platform': ['Fandango vs RT', 'Fandango vs Metacritic', 'Fandango vs IMDB'],
            'Mean Difference': [
                (merged_copy_bias['RATING'] - merged_copy_bias['RT_Normalized']).mean(),
                (merged_copy_bias['RATING'] - merged_copy_bias['Metacritic_Normalized']).mean(),
                (merged_copy_bias['RATING'] - merged_copy_bias['IMDB_Normalized']).mean()
            ],
            'Std Difference': [
                (merged_copy_bias['RATING'] - merged_copy_bias['RT_Normalized']).std(),
                (merged_copy_bias['RATING'] - merged_copy_bias['Metacritic_Normalized']).std(),
                (merged_copy_bias['RATING'] - merged_copy_bias['IMDB_Normalized']).std()
            ]
        })

        fig_bias = px.bar(
            bias_metrics,
            x='Platform',
            y='Mean Difference',
            error_y='Std Difference',
            title="Rating Bias Analysis (Fandango - Others)",
            color='Mean Difference',
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_bias, use_container_width=True)
        st.dataframe(bias_metrics.round(3), use_container_width=True)

    ##########################################################
    # SUBTAB: STATISTICAL DEEP DIVE
    ##########################################################
    with insight_tab3:
        st.subheader("Statistical Deep Dive")

        st.markdown("**Cross-Platform Correlation Matrix**")
        correlation_matrix = merged_df[[
            'RATING','VOTES','RT_Normalized',
            'Metacritic_Normalized','IMDB_Normalized'
        ]].corr()
        fig_corr = px.imshow(
            correlation_matrix,
            title="Cross-Platform Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        colX, colY = st.columns(2)
        with colX:
            st.markdown("**Distribution Statistics (Fandango Rating)**")
            stats_df = merged_df['RATING'].describe().round(3)
            st.dataframe(stats_df)

        with colY:
            st.markdown("**Hypothesis Testing** (two-sample t-test)")
            def perform_ttest(platform_data):
                t_stat, p_value = stats.ttest_ind(
                    merged_df['RATING'],
                    platform_data,
                    nan_policy='omit'
                )
                return pd.Series({'t_statistic': t_stat, 'p_value': p_value})

            ttest_results = pd.DataFrame({
                'RT': perform_ttest(merged_df['RT_Normalized']),
                'Metacritic': perform_ttest(merged_df['Metacritic_Normalized']),
                'IMDB': perform_ttest(merged_df['IMDB_Normalized'])
            }).round(4)

            st.dataframe(ttest_results)

##############################################################
#   TAB 5: DATA EXPLORER
##############################################################
with tab5:
    st.header("Interactive Data Explorer")

    colE1, colE2, colE3 = st.columns(3)
    with colE1:
        search_text = st.text_input(
            "Search Movies",
            placeholder="Enter partial movie title..."
        )
    with colE2:
        min_votes = st.number_input(
            "Minimum Votes",
            min_value=0,
            value=0
        )
    with colE3:
        sort_by = st.selectbox(
            "Sort By",
            ['FILM','RATING','VOTES','YEAR']
        )

    # Work on a copy for subsequent filtering
    explorer_df = fandango.copy()

    if search_text:
        explorer_df = explorer_df[
            explorer_df['FILM'].str.contains(search_text, case=False, na=False)
        ].copy()

    if min_votes > 0:
        explorer_df = explorer_df[explorer_df['VOTES'] >= min_votes].copy()

    explorer_df = explorer_df.sort_values(sort_by).copy()

    st.dataframe(
        explorer_df.style.background_gradient(
            subset=['RATING','VOTES'],
            cmap='RdYlBu'
        ),
        use_container_width=True
    )

    if not explorer_df.empty:
        colE_left, colE_right = st.columns(2)

        with colE_left:
            st.download_button(
                "Download CSV",
                explorer_df.to_csv(index=False).encode('utf-8'),
                "movie_data.csv",
                "text/csv",
                key='download-csv'
            )

        with colE_right:
            if st.button("Create Excel Report"):
                st.markdown(
                    create_download_link(
                        explorer_df,
                        'movie_analysis_export.xlsx',
                        '📥 Download Excel Report'
                    ),
                    unsafe_allow_html=True
                )

    st.markdown("### Quick Summary of Current View")
    colE4, colE5, colE6 = st.columns(3)

    with colE4:
        st.metric("Number of Movies", len(explorer_df))

    with colE5:
        avg_ = explorer_df['RATING'].mean()
        st.metric("Average Rating", f"{avg_:.2f}" if not np.isnan(avg_) else "N/A")

    with colE6:
        sum_votes = explorer_df['VOTES'].sum()
        st.metric("Total Votes", f"{sum_votes:,.0f}" if not np.isnan(sum_votes) else "N/A")

##############################################################
#   TAB 6: KEY FINDINGS & CONCLUSIONS
##############################################################
with tab6:
    st.header("Key Findings and Conclusions")

    st.markdown("""
    ### Rating Inflation
    - **Fandango** consistently displays higher ratings compared to other platforms.
    - The gap between displayed stars and actual ratings indicates systematic inflation.
    - This inflation is most pronounced for poorly-rated movies.

    ### Platform Comparisons
    - **Professional critics** (RT and Metacritic) tend to be more critical in their ratings.
    - **User ratings** generally fall between critic ratings and Fandango ratings.
    - **IMDB** shows the most balanced distribution of ratings.

    ### Business Implications
    - Fandango's rating inflation likely serves **business interests** (e.g., ticket sales).
    - This practice can mislead consumers about movie quality.
    - The discrepancy is particularly significant for lower-quality films.

    ### Recommendations for Consumers
    - **Consult multiple platforms** before making movie-watching decisions.
    - Be **skeptical of high Fandango ratings** for less popular or poorly-rated movies.
    - Consider both critic and user ratings for a **balanced perspective**.
    """)

##############################################################
#                     FOOTER & END
##############################################################
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Created for Movie Ratings Analysis Project</p>
    <p>Data Sources: Fandango, Rotten Tomatoes, Metacritic, and IMDB</p>
</div>
""", unsafe_allow_html=True)

st.success("End of Analysis. Thank you for exploring the data!")
