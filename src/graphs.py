import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Function to plot histograms
def plot_histograms(df, columns):
    st.subheader('Histograms')
    for col in columns:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=30, color='blue', alpha=0.7)
        ax.set_title(f'Histogram of {col}')
        st.pyplot(fig)

# Function to plot bar charts
def plot_bar_charts(df, columns):
    st.subheader('Bar Charts')
    for col in columns:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind='bar', ax=ax, color='green')
        ax.set_title(f'Bar Chart of {col}')
        st.pyplot(fig)

# Function to plot the correlation matrix
def plot_correlation_matrix(df, columns):
    st.subheader('Correlation Matrix')
    corr = df[columns].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Function to plot trend lines (scatter plot with regression line)
def plot_trend_lines(df, x_col, y_col):
    st.subheader(f'Trend between {x_col} and {y_col}')
    fig, ax = plt.subplots()
    sns.regplot(x=df[x_col], y=df[y_col], ax=ax, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
    ax.set_title(f'Trend between {x_col} and {y_col}')
    st.pyplot(fig)