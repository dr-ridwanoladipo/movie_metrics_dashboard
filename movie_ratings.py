############################################################
# Streamlit Movie Ratings Analysis App
############################################################

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set Streamlit page configuration
st.set_page_config(
    page_title="Movie Ratings Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply a more modern Seaborn theme
sns.set_theme(style="whitegrid")

############################
# 1. Title and Introduction
############################
st.title("Movie Ratings Analysis: Investigating Online Review Bias")

st.markdown("""
**Project Overview**  
This data analysis project investigates potential bias in online movie ratings, specifically focusing on Fandango's rating system. The analysis compares movie ratings across multiple platforms including Fandango, Rotten Tomatoes, Metacritic, and IMDB to identify any systematic differences in rating patterns.

**Key Questions Addressed**  
- Do online movie review platforms show rating bias?  
- How do Fandango's displayed ratings compare to actual user ratings?  
- Is there a significant difference between critic and user ratings across platforms?  
- How are poorly-rated movies scored across different platforms?

**Analysis Objectives**  
1. Evaluate potential bias in Fandango's rating system  
2. Compare rating distributions across multiple platforms  
3. Analyze the relationship between movie popularity and ratings  
4. Investigate how poorly-rated movies are scored across platforms  

**Data Sources**  
1. `fandango_scrape.csv`: Contains movie ratings data from Fandango  
2. `all_sites_scores.csv`: Includes aggregate ratings from multiple platforms  

**References**  
- Original FiveThirtyEight Article: [Be Suspicious Of Online Movie Ratings, Especially Fandango's](http://fivethirtyeight.com/features/fandango-movies-ratings/)  
- Data Source: [FiveThirtyEight GitHub Repository](https://github.com/fivethirtyeight/data)  
---
""")

#############################################################
# 2. Data Loading & Exploration
#############################################################

@st.cache_data
def load_data():
    """
    Loads both CSV files and returns two DataFrames:
    fandango and all_sites.
    """
    fandango = pd.read_csv("fandango_scrape.csv")
    all_sites = pd.read_csv("all_sites_scores.csv")
    return fandango, all_sites

fandango, all_sites = load_data()

st.header("Data Loading and Initial Exploration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fandango Data")
    # Display DataFrame
    st.write("**First few rows:**")
    st.dataframe(fandango.head())

    # Display info by capturing output in a buffer
    buffer = io.StringIO()
    fandango.info(buf=buffer)
    s = buffer.getvalue()
    st.write("**Dataset Info:**")
    st.text(s)

    st.write("**Descriptive Statistics:**")
    st.dataframe(fandango.describe())

with col2:
    st.subheader("All Sites Data")
    st.write("**First few rows:**")
    st.dataframe(all_sites.head())

    buffer2 = io.StringIO()
    all_sites.info(buf=buffer2)
    info_str = buffer2.getvalue()
    st.write("**Dataset Info:**")
    st.text(info_str)

    st.write("**Descriptive Statistics:**")
    st.dataframe(all_sites.describe())

st.markdown("---")

