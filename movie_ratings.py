###############################################
# STREAMLIT APP: Movie Ratings Analysis
# Combines all key elements from original code
# and your enhanced UI design.
###############################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Movie Ratings Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎬"
)

# Custom CSS styling with lighter theme
st.markdown("""
<style>
    /* Set app background and text color */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        color: #333333;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f0f0f0;
    }
    /* Card styling */
    .card {
        background: rgba(0, 123, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    /* Metric card styling */
    .metric-card {
        background: rgba(0, 123, 255, 0.1);
        border: 1px solid rgba(0, 123, 255, 0.3);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #007bff;
    }
    .metric-label {
        font-size: 14px;
        color: #555555;
    }
    /* Table styling */
    .dataframe {
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)


# ---- Data Caching ----
@st.cache_data
def load_data():
    """Loads both CSV files as DataFrames."""
    fandango = pd.read_csv("fandango_scrape.csv")
    all_sites = pd.read_csv("all_sites_scores.csv")
    return fandango, all_sites

@st.cache_data
def normalize_ratings(fandango_df, all_sites_df):
    """
    Merge Fandango & All Sites data, then normalize:
    - Rotten Tomatoes (0-100) -> /20
    - Metacritic (0-100) -> /20
    - IMDB (0-10) -> /2
    Returns a DataFrame with columns on a 0–5 scale.
    """
    df_merged = pd.merge(fandango_df, all_sites_df, on='FILM', how='inner')
    normalized = pd.DataFrame()
    normalized['FILM']               = df_merged['FILM']
    normalized['Fandango_Stars']     = df_merged['STARS']
    normalized['Fandango_Rating']    = df_merged['RATING']
    normalized['RT_Critic']          = (df_merged['RottenTomatoes'] / 20).round(1)
    normalized['RT_User']            = (df_merged['RottenTomatoes_User'] / 20).round(1)
    normalized['Metacritic_Critic']  = (df_merged['Metacritic'] / 20).round(1)
    normalized['Metacritic_User']    = (df_merged['Metacritic_User'] / 2).round(1)
    normalized['IMDB']               = (df_merged['IMDB'] / 2).round(1)
    return normalized

# ---- Load Data ----
fandango, all_sites = load_data()
# Extract YEAR from "FILM" for Fandango
fandango['YEAR'] = fandango['FILM'].apply(lambda x: x.split('(')[-1].replace(')', ''))
normalized_ratings = normalize_ratings(fandango, all_sites)

# ---- SIDEBAR CONTROLS ----
st.sidebar.title("🎬 Analysis Controls")

# Years present in data
available_years = sorted(fandango['YEAR'].unique())
year_range = st.sidebar.select_slider(
    "Select Year Range",
    options=available_years,
    value=(min(available_years), max(available_years))
)

# Multi-platform selection
platforms = st.sidebar.multiselect(
    "Select Platforms to Compare",
    ["Fandango", "Rotten Tomatoes", "Metacritic", "IMDB"],
    default=["Fandango", "Rotten Tomatoes"]
)

# Rating range filter
rating_range = st.sidebar.slider(
    "Filter by Rating Range (Fandango)",
    0.0, 5.0, (0.0, 5.0),
    step=0.5
)

# Filter data by the selected year range
start_year, end_year = year_range
filtered_fandango = fandango[
    (fandango['YEAR'] >= str(start_year)) & (fandango['YEAR'] <= str(end_year))
]

# Filter data by rating range
filtered_fandango = filtered_fandango[
    (filtered_fandango['RATING'] >= rating_range[0]) &
    (filtered_fandango['RATING'] <= rating_range[1])
]

# ---- MAIN CONTENT ----
st.title("🎬 Movie Ratings Analysis Dashboard")
st.markdown("## Investigating Online Review Bias")

# ---- METRIC CARDS ----
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(filtered_fandango)}</div>
        <div class="metric-label">Movies Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_rating = round(filtered_fandango['RATING'].mean(), 2) if len(filtered_fandango) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{avg_rating}</div>
        <div class="metric-label">Average Rating</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Quick estimate of rating inflation
    fan_reviewed = fandango[fandango['VOTES'] > 0].copy()
    fan_reviewed["STARS_DIFF"] = (fan_reviewed['STARS'] - fan_reviewed['RATING']).round(2)
    inflation_val = fan_reviewed["STARS_DIFF"].mean().round(2)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">+{inflation_val}</div>
        <div class="metric-label">Avg Rating Inflation</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Some correlation measure
    corr_val = fandango['RATING'].corr(fandango['VOTES'])
    corr_display = round(corr_val * 100, 1)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{corr_display}%</div>
        <div class="metric-label">Rating-Vote Correlation</div>
    </div>
    """, unsafe_allow_html=True)

# ---- TABS ----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Rating Analysis",
    "Platform Comparison",
    "Temporal Trends",
    "Advanced Analysis",
    "Key Insights"
])

##################################################
# TAB 1: Rating Analysis
##################################################
with tab1:
    st.markdown("### Rating Distribution (Fandango)")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=filtered_fandango['RATING'],
        name='Fandango Ratings',
        opacity=0.75,
        marker_color='blue'
    ))
    fig_dist.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Distribution of Fandango Ratings (Filtered by Sidebar)"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Rating correlation heatmap (Fandango)
    st.markdown("### Correlation Matrix (Fandango)")
    corr_matrix = fandango.corr(numeric_only=True)
    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu',
        title="Correlation Heatmap (Fandango)"
    )
    fig_corr.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.write("**Stats:**")
    st.write(filtered_fandango.describe())

##################################################
# TAB 2: Platform Comparison
##################################################
with tab2:
    st.markdown("### Cross-Platform Ratings Overview")

    # Example scatter: RottenTomatoes vs IMDB
    # color by Metacritic for synergy
    fig_scatter = px.scatter(
        all_sites,
        x='RottenTomatoes',
        y='IMDB',
        color='Metacritic',
        hover_data=['FILM'],
        title="Cross-Platform Rating Comparison (RottenTomatoes vs IMDB)"
    )
    fig_scatter.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Rating Differences: Rotten Tomatoes Critics vs Users")
    all_sites['Rotten_Diff'] = all_sites['RottenTomatoes'] - all_sites['RottenTomatoes_User']
    fig_rdiff = plt.figure(figsize=(10,4), dpi=150)
    plt.subplot(1, 2, 1)
    sns.histplot(data=all_sites, x='Rotten_Diff', kde=True, bins=25, color='salmon')
    plt.title('Direction (Critics - Users)')
    plt.subplot(1, 2, 2)
    sns.histplot(x=all_sites['Rotten_Diff'].abs(), bins=25, kde=True, color='orange')
    plt.title('Absolute Differences')
    st.pyplot(fig_rdiff)

    mean_diff = all_sites['Rotten_Diff'].mean()
    mean_abs_diff = all_sites['Rotten_Diff'].abs().mean()
    median_abs_diff = all_sites['Rotten_Diff'].abs().median()
    st.write(f"**Mean Directional Difference**: {mean_diff:.2f}")
    st.write(f"**Mean Absolute Difference**: {mean_abs_diff:.2f}")
    st.write(f"**Median Absolute Difference**: {median_abs_diff:.2f}")

    col_loved, col_hated = st.columns(2)
    with col_loved:
        st.markdown("**Critics Loved but Users Hated**")
        critics_loved = all_sites.nlargest(5, 'Rotten_Diff')[['FILM','RottenTomatoes','RottenTomatoes_User','Rotten_Diff']]
        st.dataframe(critics_loved)
    with col_hated:
        st.markdown("**Users Loved but Critics Hated**")
        users_loved = all_sites.nsmallest(5, 'Rotten_Diff')[['FILM','RottenTomatoes','RottenTomatoes_User','Rotten_Diff']]
        st.dataframe(users_loved)

    st.markdown("---")
    st.markdown("### Outlier Analysis: Metacritic vs IMDB")
    fig_votes = px.scatter(
        all_sites,
        x='Metacritic_user_vote_count',
        y='IMDB_user_vote_count',
        hover_data=['FILM'],
        title="Vote Count Comparison: Metacritic vs IMDB",
        template='plotly_dark'
    )
    fig_votes.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_votes, use_container_width=True)

    top_imdb = all_sites.nlargest(1, 'IMDB_user_vote_count')
    top_meta = all_sites.nlargest(1, 'Metacritic_user_vote_count')

    col_im, col_mt = st.columns(2)
    with col_im:
        st.markdown("**Highest IMDB Vote Count**")
        st.dataframe(top_imdb[['FILM','IMDB_user_vote_count','Metacritic_user_vote_count']])
    with col_mt:
        st.markdown("**Highest Metacritic Vote Count**")
        st.dataframe(top_meta[['FILM','Metacritic_user_vote_count','IMDB_user_vote_count']])

##################################################
# TAB 3: Temporal Trends
##################################################
with tab3:
    st.markdown("### Average Fandango Ratings Over Time")
    # Group by YEAR, ignoring any non-numeric or weird data
    # We'll get average rating by year
    time_data = fandango.groupby('YEAR')['RATING'].mean().reset_index()
    fig_time = px.line(
        time_data,
        x='YEAR',
        y='RATING',
        title="Average Fandango Ratings Over Time",
        template='plotly_dark'
    )
    fig_time.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("### Movie Counts by Year")
    year_counts = fandango['YEAR'].value_counts().sort_index()
    fig_count = px.bar(
        x=year_counts.index,
        y=year_counts.values,
        labels={'x':'Year','y':'Count'},
        title="Number of Movies by Release Year",
        template='plotly_dark'
    )
    fig_count.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_count, use_container_width=True)

##################################################
# TAB 4: Advanced Analysis (Normalization & Cluster Map)
##################################################
with tab4:
    st.markdown("### Normalized Ratings (0–5 Scale)")
    st.write("""
    This table compares all platforms on a uniform 0–5 scale.
    - Rotten Tomatoes (0–100) -> /20
    - Metacritic (0–100) -> /20
    - IMDB (0–10) -> /2
    """)

    st.dataframe(
        normalized_ratings.style.background_gradient(cmap='Blues'),
        use_container_width=True
    )

    st.markdown("### Clustermap of Normalized Ratings")
    st.write("Exploring potential groupings of similar movies across rating platforms.")
    # We can drop rows with missing data
    complete_data = normalized_ratings.dropna(subset=[
        'Fandango_Stars','Fandango_Rating','RT_Critic','RT_User','Metacritic_Critic','Metacritic_User','IMDB'
    ])
    # Build a Seaborn cluster map
    # We'll exclude the 'FILM' column for clustering
    cluster_data = complete_data.drop(columns=['FILM'])
    fig_cluster = sns.clustermap(cluster_data, col_cluster=False, cmap='mako', figsize=(10, 8))
    # To display in Streamlit, convert to a normal figure:
    st.pyplot(fig_cluster)

    st.markdown("### Worst-Rated Movies (By RT Critic)")
    worst_rated = complete_data.nsmallest(10, 'RT_Critic').copy()

    # Create a Plotly heatmap to compare these lowest-rated titles
    fig_worst = px.imshow(
        worst_rated.set_index('FILM').T,
        color_continuous_scale='YlOrRd',
        title="Ratings for Lowest-Rated Movies (RT Critics)"
    )
    fig_worst.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_worst, use_container_width=True)

    # Quick summary of worst-rated means
    st.write("**Average Platform Ratings for These Worst-Rated Movies**:")
    col_means = worst_rated.drop(columns=['FILM']).mean(numeric_only=True)
    for col in col_means.index:
        st.write(f"- {col}: {col_means[col]:.2f}")

##################################################
# TAB 5: Key Insights
##################################################
with tab5:
    st.markdown("## Key Findings and Conclusions")

    st.markdown("""
    ### Rating Inflation
    - Fandango consistently displays higher ratings compared to other platforms
    - The gap between displayed stars and actual ratings indicates systematic inflation
    - This inflation is most pronounced for poorly-rated movies

    ### Platform Comparisons
    - Professional critics (RT and Metacritic) tend to be more critical
    - User ratings generally fall between critic ratings and Fandango ratings
    - IMDB shows the most balanced distribution of ratings

    ### Business Implications
    - Fandango's rating inflation likely serves business interests (ticket sales)
    - This practice potentially misleads consumers about movie quality
    - Discrepancy is particularly significant for lower-quality films

    ### Recommendations for Consumers
    - Consult multiple platforms before making movie-watching decisions
    - Be particularly skeptical of high Fandango ratings for less popular movies
    - Consider both critic and user ratings for a balanced perspective
    """)

# ---- RAW DATA ----
st.markdown("---")
st.markdown("### Detailed Movie Analysis")
if st.checkbox("Show Raw Data (Fandango)"):
    st.dataframe(
        fandango.style.background_gradient(cmap='RdYlBu'),
        use_container_width=True
    )

if st.checkbox("Show Raw Data (All Sites)"):
    st.dataframe(
        all_sites.style.background_gradient(cmap='RdYlBu'),
        use_container_width=True
    )

# ---- END ----
st.success("End of Analysis. Thank you for exploring the data!")