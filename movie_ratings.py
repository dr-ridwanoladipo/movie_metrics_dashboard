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

# Page Configuration
st.set_page_config(
    page_title="Movie Ratings Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎬"
)

# Enhanced CSS with tooltips and modern styling
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

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #333;
        padding: 10px 20px;
        border-radius: 5px;
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


# Data Loading and Processing Functions
@st.cache_data
def load_data():
    """Load and prepare data with additional metrics"""
    fandango = pd.read_csv("fandango_scrape.csv")
    all_sites = pd.read_csv("all_sites_scores.csv")

    # Extract year and calculate additional metrics
    fandango['YEAR'] = fandango['FILM'].str.extract(r'\((\d{4})\)').astype(float)
    fandango['RATING_CATEGORY'] = pd.cut(
        fandango['RATING'],
        bins=[0, 2, 3, 4, 5],
        labels=['Poor', 'Average', 'Good', 'Excellent']
    )

    # Calculate normalized ratings
    df = pd.merge(fandango, all_sites, on='FILM', how='inner')
    df['RT_Normalized'] = df['RottenTomatoes'] / 20
    df['Metacritic_Normalized'] = df['Metacritic'] / 20
    df['IMDB_Normalized'] = df['IMDB'] / 2

    return fandango, all_sites, df


def generate_excel_download(df, sheet_name='Data'):
    """Generate formatted Excel file for download"""
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
    """Create a styled download link for Excel files"""
    excel_file = generate_excel_download(df)
    b64 = base64.b64encode(excel_file.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="export-button">{text}</a>'
    return href


# Load Data
fandango, all_sites, merged_df = load_data()

# Sidebar Controls
st.sidebar.title("🎬 Analysis Controls")

# Enhanced Search Feature
search_query = st.sidebar.text_input(
    "🔍 Search Movies",
    help="Search for specific movies by title, year, or rating"
)

if search_query:
    filtered_movies = fandango[
        fandango['FILM'].str.contains(search_query, case=False) |
        fandango['YEAR'].astype(str).str.contains(search_query) |
        fandango['RATING'].astype(str).str.contains(search_query)
        ]
    st.sidebar.write(f"Found {len(filtered_movies)} movies")
    st.sidebar.dataframe(
        filtered_movies[['FILM', 'YEAR', 'RATING', 'VOTES']],
        use_container_width=True
    )

# Year Filter
years = sorted(fandango['YEAR'].dropna().unique())
year_range = st.sidebar.select_slider(
    "Select Year Range",
    options=years,
    value=(min(years), max(years)),
    help="Filter movies by release year"
)

# Rating Filter
rating_range = st.sidebar.slider(
    "Filter by Rating Range",
    0.0, 5.0, (0.0, 5.0),
    help="Filter movies by Fandango rating"
)

# Platform Selection
platforms = st.sidebar.multiselect(
    "Select Platforms to Compare",
    ["Fandango", "Rotten Tomatoes", "Metacritic", "IMDB"],
    default=["Fandango", "Rotten Tomatoes"],
    help="Choose platforms for comparison analysis"
)

# Export Options
st.sidebar.markdown("### Export Options")
if st.sidebar.button("Generate Analysis Report"):
    # Create comprehensive summary
    summary_data = {
        'Metric': [
            'Total Movies',
            'Average Rating',
            'Rating Inflation',
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
            fandango.loc[fandango['RATING'].idxmax(), 'FILM'],
            fandango.loc[fandango['VOTES'].idxmax(), 'FILM']
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

# Main Content
st.title("🎬 Enhanced Movie Ratings Analysis Dashboard")
st.markdown("### Investigating Online Review Bias with Interactive Analysis")

# Enhanced Metrics with Tooltips
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

# Interactive Analysis Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Rating Analysis",
    "Platform Comparison",
    "Temporal Analysis",
    "Advanced Insights",
    "Data Explorer"
])

# Tab 1: Rating Analysis
with tab1:
    st.header("Rating Distribution Analysis")

    # Interactive control for visualization type
    viz_type = st.radio(
        "Select Visualization Type",
        ["Distribution", "Box Plot", "Scatter Plot"],
        horizontal=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if viz_type == "Distribution":
            fig = px.histogram(
                fandango,
                x='RATING',
                nbins=20,
                title="Fandango Ratings Distribution",
                color_discrete_sequence=['#3498db']
            )
        elif viz_type == "Box Plot":
            fig = px.box(
                fandango,
                y='RATING',
                title="Rating Distribution Box Plot"
            )
        else:  # Scatter Plot
            fig = px.scatter(
                fandango,
                x='VOTES',
                y='RATING',
                hover_data=['FILM'],
                title="Ratings vs Number of Votes"
            )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Rating category breakdown
        rating_cats = fandango['RATING_CATEGORY'].value_counts()
        fig = px.pie(
            values=rating_cats.values,
            names=rating_cats.index,
            title="Distribution of Rating Categories"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Statistical summary
    st.subheader("Statistical Summary")
    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.metric("Median Rating", f"{fandango['RATING'].median():.2f}")
        st.metric("Standard Deviation", f"{fandango['RATING'].std():.2f}")
    with stats_col2:
        st.metric("Mode Rating", f"{fandango['RATING'].mode()[0]:.2f}")
        st.metric("Rating Range", f"{fandango['RATING'].max() - fandango['RATING'].min():.2f}")

# Tab 2: Platform Comparison
with tab2:
    st.header("Cross-Platform Analysis")

    # Platform selector for detailed comparison
    platform_compare = st.selectbox(
        "Select Platform to Compare with Fandango",
        ["Rotten Tomatoes", "Metacritic", "IMDB"]
    )

    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot comparison
        if platform_compare == "Rotten Tomatoes":
            comparison_data = merged_df[['FILM', 'RATING', 'RT_Normalized']]
            x_col = 'RT_Normalized'
            title = "Fandango vs Rotten Tomatoes"
        elif platform_compare == "Metacritic":
            comparison_data = merged_df[['FILM', 'RATING', 'Metacritic_Normalized']]
            x_col = 'Metacritic_Normalized'
            title = "Fandango vs Metacritic"
        else:
            comparison_data = merged_df[['FILM', 'RATING', 'IMDB_Normalized']]
            x_col = 'IMDB_Normalized'
            title = "Fandango vs IMDB"

        fig = px.scatter(
            comparison_data,
            x=x_col,
            y='RATING',
            hover_data=['FILM'],
            title=title
        )

        # Add diagonal reference line
        fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=5, y1=5,
            line=dict(color="red", dash="dash")
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Rating difference distribution
        diff_col = f'Diff_{platform_compare.replace(" ", "_")}'
        comparison_data[diff_col] = comparison_data['RATING'] - comparison_data[x_col]

        fig = px.histogram(
            comparison_data,
            x=diff_col,
            title=f"Rating Differences Distribution ({platform_compare})",
            color_discrete_sequence=['#2ecc71']
        )
        st.plotly_chart(fig, use_container_width=True)

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

# Tab 3: Temporal Analysis
with tab3:
    st.header("Temporal Trends Analysis")

    # Time range selector
    year_min, year_max = st.select_slider(
        "Select Time Period",
        options=sorted(fandango['YEAR'].unique()),
        value=(min(fandango['YEAR']), max(fandango['YEAR']))
    )

    # Filter data by selected years
    time_data = fandango[
        (fandango['YEAR'] >= year_min) &
        (fandango['YEAR'] <= year_max)
        ]

    col1, col2 = st.columns(2)

    with col1:
        # Yearly trends
        yearly_avg = time_data.groupby('YEAR')['RATING'].agg(['mean', 'count']).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=yearly_avg['YEAR'],
                y=yearly_avg['mean'],
                name="Average Rating",
                line=dict(color="#1f77b4")
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Bar(
                x=yearly_avg['YEAR'],
                y=yearly_avg['count'],
                name="Number of Movies",
                marker_color="rgba(158,202,225,0.4)"
            ),
            secondary_y=True
        )

        fig.update_layout(
            title="Yearly Trends: Average Ratings and Movie Count",
            xaxis_title="Year",
            yaxis_title="Average Rating",
            yaxis2_title="Number of Movies"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Rating distribution evolution
        fig = px.box(
            time_data,
            x='YEAR',
            y='RATING',
            title="Rating Distribution Evolution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Trend analysis
    st.subheader("Trend Analysis")
    trend_stats = time_data.groupby('YEAR').agg({
        'RATING': ['mean', 'std', 'count'],
        'VOTES': 'mean'
    }).round(2)
    trend_stats.columns = ['Avg Rating', 'Rating Std', 'Movie Count', 'Avg Votes']
    st.dataframe(trend_stats, use_container_width=True)

# Tab 4: Advanced Insights
with tab4:
    st.header("Advanced Analysis & Insights")

    # Subtabs for different analyses
    insight_tab1, insight_tab2, insight_tab3 = st.tabs([
        "Rating Anomalies",
        "Platform Bias Analysis",
        "Statistical Deep Dive"
    ])

    with insight_tab1:
        st.subheader("Rating Anomaly Detection")

        # Calculate Z-scores for ratings
        merged_df['rating_zscore'] = (merged_df['RATING'] - merged_df['RATING'].mean()) / merged_df['RATING'].std()

        # Identify anomalies (Z-score > 2 or < -2)
        anomalies = merged_df[abs(merged_df['rating_zscore']) > 2]

        fig = px.scatter(
            merged_df,
            x='VOTES',
            y='RATING',
            color='rating_zscore',
            hover_data=['FILM'],
            title="Rating Anomalies Detection",
            color_continuous_scale='RdYlBu'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display anomalous ratings
        st.write(f"Found {len(anomalies)} anomalous ratings")
        if not anomalies.empty:
            st.dataframe(
                anomalies[['FILM', 'RATING', 'VOTES', 'rating_zscore']].sort_values('rating_zscore', ascending=False),
                use_container_width=True
            )

    with insight_tab2:
        st.subheader("Platform Bias Analysis")

        # Calculate bias metrics
        bias_metrics = pd.DataFrame({
            'Platform': ['Fandango vs RT', 'Fandango vs Metacritic', 'Fandango vs IMDB'],
            'Mean Difference': [
                (merged_df['RATING'] - merged_df['RT_Normalized']).mean(),
                (merged_df['RATING'] - merged_df['Metacritic_Normalized']).mean(),
                (merged_df['RATING'] - merged_df['IMDB_Normalized']).mean()
            ],
            'Std Difference': [
                (merged_df['RATING'] - merged_df['RT_Normalized']).std(),
                (merged_df['RATING'] - merged_df['Metacritic_Normalized']).std(),
                (merged_df['RATING'] - merged_df['IMDB_Normalized']).std()
            ]
        })

        # Visualization of bias
        fig = px.bar(
            bias_metrics,
            x='Platform',
            y='Mean Difference',
            error_y='Std Difference',
            title="Rating Bias Analysis",
            color='Mean Difference',
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed bias metrics
        st.dataframe(bias_metrics.round(3), use_container_width=True)

    with insight_tab3:
        st.subheader("Statistical Analysis")

        # Correlation matrix
        correlation_matrix = merged_df[[
            'RATING', 'VOTES', 'RT_Normalized',
            'Metacritic_Normalized', 'IMDB_Normalized'
        ]].corr()

        fig = px.imshow(
            correlation_matrix,
            title="Cross-Platform Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Additional statistical metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Distribution Statistics**")
            stats_df = merged_df['RATING'].describe()
            st.dataframe(stats_df.round(3))

        with col2:
            st.markdown("**Hypothesis Testing**")
            # Perform t-test between Fandango and other platforms
            from scipy import stats


            def perform_ttest(platform_data):
                t_stat, p_value = stats.ttest_ind(
                    merged_df['RATING'],
                    platform_data
                )
                return pd.Series({'t_statistic': t_stat, 'p_value': p_value})


            ttest_results = pd.DataFrame({
                'RT': perform_ttest(merged_df['RT_Normalized']),
                'Metacritic': perform_ttest(merged_df['Metacritic_Normalized']),
                'IMDB': perform_ttest(merged_df['IMDB_Normalized'])
            })

            st.dataframe(ttest_results.round(4))

# Tab 5: Data Explorer
with tab5:
    st.header("Interactive Data Explorer")

    # Search and filter controls
    col1, col2, col3 = st.columns(3)

    with col1:
        search_text = st.text_input(
            "Search Movies",
            placeholder="Enter movie title..."
        )

    with col2:
        min_votes = st.number_input(
            "Minimum Votes",
            min_value=0,
            value=0
        )

    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ['FILM', 'RATING', 'VOTES', 'YEAR']
        )

    # Filter and sort data
    explorer_df = fandango.copy()

    if search_text:
        explorer_df = explorer_df[
            explorer_df['FILM'].str.contains(search_text, case=False)
        ]

    if min_votes > 0:
        explorer_df = explorer_df[explorer_df['VOTES'] >= min_votes]

    explorer_df = explorer_df.sort_values(sort_by)

    # Display interactive table
    st.dataframe(
        explorer_df.style.background_gradient(
            subset=['RATING', 'VOTES'],
            cmap='RdYlBu'
        ),
        use_container_width=True
    )

    # Export functionality
    if not explorer_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Download CSV",
                explorer_df.to_csv(index=False).encode('utf-8'),
                "movie_data.csv",
                "text/csv",
                key='download-csv'
            )

        with col2:
            if st.button("Create Excel Report"):
                st.markdown(
                    create_download_link(
                        explorer_df,
                        'movie_analysis_export.xlsx',
                        '📥 Download Excel Report'
                    ),
                    unsafe_allow_html=True
                )

    # Summary metrics
    st.markdown("### Quick Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Number of Movies",
            len(explorer_df)
        )

    with col2:
        st.metric(
            "Average Rating",
            f"{explorer_df['RATING'].mean():.2f}"
        )

    with col3:
        st.metric(
            "Total Votes",
            f"{explorer_df['VOTES'].sum():,}"
        )

# Footer
st.markdown("---")
st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Created for Movie Ratings Analysis Project</p>
            <p>Data Sources: Fandango, Rotten Tomatoes, Metacritic, and IMDB</p>
        </div>
    """, unsafe_allow_html=True)

# Add a floating help button
st.markdown("""
        <div style='position: fixed; bottom: 20px; right: 20px; z-index: 1000;'>
            <button class='export-button' onclick='alert("Help documentation coming soon!")'>
                ❓ Help
            </button>
        </div>
    """, unsafe_allow_html=True)