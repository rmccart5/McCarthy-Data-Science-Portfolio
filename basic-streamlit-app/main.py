import streamlit as st
import pandas as pd

st.title("Explore the Countries of the World!")
st.write("## First, select which variable you want to filter countries by.")

# Dataframe of countries of the world
df = pd.read_csv("basic-streamlit-app\data\countries_of_the_world.csv")

# User selects a button, determing which filter they want to apply to the dataframe
filter = st.radio("Select a filter",["Name", "Region", "Population","Area", "GDP ($ per capita)", "None"])


if filter == "Name":
    st.write("## Next, use the dropdown to find specific information about our world's countries.")
    name = st.selectbox("Select a Country", df["Country"].unique())
    name_df = df[df["Country"] == name]
    st.dataframe(name_df)
elif filter == "Region":
    st.write("## Next, use the dropdown to find specific information about our world's countries.")
    region = st.selectbox("Select a Region", df["Region"].unique())
    region_df = df[df["Region"] == region]
    st.dataframe(region_df)
elif filter == "Population":
    st.write("## Next, use the slider to find specific information about our world's countries.")
    pop_range = st.slider("Choose a minimum population:", 
                          min_value = df["Population"].min(), max_value = df["Population"].max())
    st.write(f"Countries with populations greater than {pop_range}:")
    st.dataframe(df[df["Population"]>= pop_range])
elif filter == "Area":
    st.write("## Next, use the slider to find specific information about our world's countries.")
    area_range = st.slider("Choose a minimum land area (sq. mi.):", 
                          min_value = df["Area (sq. mi.)"].min(), max_value = df["Area (sq. mi.)"].max())
    st.write(f"Countries with an area (sq. mi.) greater than {area_range}:")
    st.dataframe(df[df["Area (sq. mi.)"]>= area_range])
elif filter == "GDP ($ per capita)":
    st.write("## Next, use the slider to find specific information about our world's countries.")
    GDP_range = st.slider("Choose a minimum GDP ($ per capita):", 
                          min_value = df["GDP ($ per capita)"].min(), max_value = df["GDP ($ per capita)"].max())
    st.write(f"Countries with a GDP ($ per capita) greater than {GDP_range}:")
    st.dataframe(df[df["GDP ($ per capita)"]>= GDP_range])
elif filter == "None":
    st.dataframe(df)


