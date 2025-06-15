
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px




st.set_page_config(
    page_title="Review Prediction",
    page_icon="😊"
)

st.title (':violet[Sentiment Analysis for Womens Cloth Review]')


tab1, tab2= st.tabs(['Data Insights', 'Sentiment Analysis'])

with tab1:


    
    data = pd.read_csv('F:/rating/women_cloth_reviews_final.csv')
    df = pd.DataFrame(data)

    st.title("Clothing Review Dashboard")

    # Rating Distribution
    fig_rating = px.histogram(df, x="Rating", nbins=5, title="Rating Distribution")
    # Customize bar appearance and layout
    fig_rating.update_traces(
        marker_color='orange',   # Bar color
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.75
    )

    fig_rating.update_layout(
        width=70,     # Width in pixels
        height=400,    # Height in pixels
        bargap=0.1,    # Space between bars
    )

    st.plotly_chart(fig_rating)

    # Age Distribution
    fig_age = px.histogram(df, x="Age", nbins=5, title="Age Distribution")

        # Customize bar appearance and layout
    fig_age.update_traces(
        marker_color='pink',   # Bar color
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.75
    )

    fig_age.update_layout(
        width=70,     # Width in pixels
        height=400,    # Height in pixels
        bargap=0.1,    # Space between bars
    )
    st.plotly_chart(fig_age)

    # Recommendations by Class
    rec_by_class = df.groupby("Class.Name")["Recommended.IND"].sum().reset_index()
    fig_rec = px.bar(rec_by_class, x="Class.Name", y="Recommended.IND", title="Recommendations by Class")
    # Customize bar appearance and layout
    fig_rec.update_traces(
        marker_color='blue',   # Bar color
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.75
    )

    fig_rec.update_layout(
    xaxis_tickangle=-90  # or 90, depending on your preference
    )
    st.plotly_chart(fig_rec)

    # Feedback vs Rating
    fig_scatter = px.scatter(
        df, x="Rating", y="Positive.Feedback.Count", color="Department.Name",
        title="Positive Feedback vs. Rating"
    )
    st.plotly_chart(fig_scatter)


with tab2:


    with open(r"F:/rating/multinomial_nb_pipeline.pkl", 'rb') as file_1:
        mb_model = pickle.load(file_1)
