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

