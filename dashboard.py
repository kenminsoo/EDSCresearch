import streamlit as st
import pandas as pd
import spacy
import numpy as np
import pickle

## Functions ##

nlp = spacy.load("en_core_web_md")

def get_word_embedding(input_text):
    """
    Function to generate a word embedding for an input query.
    
    Parameters:
    input_text (str): The input text to embed.
    
    Returns:
    vector: The word embedding vector.
    """
    doc = nlp(input_text)
    return doc.vector

def cosine_similarity(vec1, vec2):
    """
    Function to calculate the cosine similarity between two vectors.
    
    Parameters:
    vec1 (numpy array): The first vector.
    vec2 (numpy array): The second vector.
    
    Returns:
    float: The cosine similarity between the two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if (norm_vec1 * norm_vec2) != 0:
        return dot_product / (norm_vec1 * norm_vec2)
    else:
        return -1

## End Fuctions ##

# Convert relevant columns to strings for filtering
domain_counts_summary = pd.read_csv('data/domain_counts_summary.csv').drop(["Unnamed: 0"], axis = 1).drop_duplicates()
domain_counts = pd.read_csv('data/domain_counts.csv').drop(["Unnamed: 0"], axis = 1).drop_duplicates()
field_counts_summary = pd.read_csv('data/field_counts_summary.csv').drop(["Unnamed: 0"], axis = 1).drop_duplicates()
field_counts = pd.read_csv('data/field_counts.csv').drop(["Unnamed: 0"], axis = 1).drop_duplicates()
name_counts_summary = pd.read_csv('data/name_counts_summary.csv').drop(["Unnamed: 0"], axis = 1).drop_duplicates()
name_counts = pd.read_csv('data/name_counts.csv').drop(["Unnamed: 0"], axis = 1).drop_duplicates()
subfield_counts_summary = pd.read_csv('data/subfield_counts_summary.csv').drop(["Unnamed: 0"], axis = 1).drop_duplicates()
subfield_counts = pd.read_csv('data/subfield_counts.csv').drop(["Unnamed: 0"], axis = 1).drop_duplicates()
topics_df = pd.read_csv('data/topics_reference.csv').drop(["Unnamed: 0"], axis = 1).drop_duplicates()

field_counts_summary = field_counts_summary.merge(topics_df[["domain_name", "field_name"]].drop_duplicates(), on = "field_name")
field_counts = field_counts.merge(topics_df[["domain_name", "field_name"]].drop_duplicates(), on = "field_name")

subfield_counts_summary = subfield_counts_summary.merge(topics_df[["domain_name", "field_name", "subfield_name"]].drop_duplicates(), on = "subfield_name")
subfield_counts = subfield_counts.merge(topics_df[["domain_name", "field_name", "subfield_name"]].drop_duplicates(), on = "subfield_name")

name_counts_summary = name_counts_summary.merge(topics_df[["domain_name", "field_name", "subfield_name", "name"]].drop_duplicates(), on = "name")
name_counts = name_counts.merge(topics_df[["domain_name", "field_name", "subfield_name", "name"]].drop_duplicates(), on = "name")

with open("data/unis", "rb") as unis:
    uni_list = pickle.load(unis)

with open("data/country", "rb") as count:
    country_list = pickle.load(count)

# Title of the Streamlit app
st.title("Emory Research Explorer")

# Create page navigation
page = st.sidebar.selectbox("Select a Page", ["General Search", "Interdisciplinary Research", "Research Networks"])

if page == "General Search":
    st.header("Page 1: General Searches")

    # Domain filter
    # select the level of granularity
    st.markdown("### Least to Most Specific: Domain, Field, Subfield, Name")

    selected_level = st.selectbox(
        'Select a Specificity Level', 
        ["Domain", "Field", "Subfield", "Name"]
    )

    st.markdown("### Bin 6: 2021-2024")

    selected_bin = st.selectbox(
        'Select a Bin', 
        domain_counts_summary['bin'].unique(),
        index = len(domain_counts_summary['bin'].unique()) - 1
    ) 

    

    # Bin filter
    if selected_level == "Domain":
        search_inst, search_country = st.columns(2)

        filtering_domain = st.multiselect("Select Domains" ,domain_counts_summary["domain_name"].unique(), default = domain_counts_summary["domain_name"].unique())

        filtered_data = domain_counts[(domain_counts["bin"] == selected_bin) & (domain_counts["domain_name"].isin(filtering_domain))].sort_values(["counts"], ascending = False)

        with search_inst:
            if st.checkbox("Select Institution"):
                inst_select = st.selectbox("Select Inst.", uni_list)
                filtered_data = filtered_data[filtered_data["Summary_Institution"].str.contains(inst_select)]

        with search_country:
            if st.checkbox("Select Country"):
                country_select = st.selectbox("Select Country", country_list)
                filtered_data = filtered_data[filtered_data["Summary_Country"].str.contains(country_select)]

        st.markdown("Summary Data")
        st.dataframe(domain_counts_summary[(domain_counts_summary["bin"] == selected_bin)])

        st.markdown("Data")
        
        st.dataframe(filtered_data)

    elif selected_level == "Field":
        search_inst, search_country = st.columns(2)

        filtering_domain = st.multiselect("Select Domains" ,field_counts_summary["field_name"].unique(), default = field_counts_summary["field_name"].unique())

        filtered_data = field_counts[(field_counts["bin"] == selected_bin) & (field_counts["field_name"].isin(filtering_domain))].sort_values(["counts"], ascending = False)

        with search_inst:
            if st.checkbox("Select Institution"):
                inst_select = st.selectbox("Select Inst.", uni_list)
                filtered_data = filtered_data[filtered_data["Summary_Institution"].str.contains(inst_select)]

        with search_country:
            if st.checkbox("Select Country"):
                country_select = st.selectbox("Select Country", country_list)
                filtered_data = filtered_data[filtered_data["Summary_Country"].str.contains(country_select)]

        if st.checkbox("Filter by Domain"):
            Domain = st.multiselect("Select Domain" ,filtered_data["domain_name"].unique(), default = filtered_data["domain_name"].unique())
            filtered_data = filtered_data[filtered_data["domain_name"].isin(Domain)]
            st.markdown("Summary Data")
            st.dataframe(field_counts_summary[(field_counts_summary["bin"] == selected_bin) & (field_counts_summary["domain_name"].isin(Domain))])
        else:
            st.markdown("Summary Data")
            st.dataframe(field_counts_summary[(field_counts_summary["bin"] == selected_bin)])

        st.markdown("Data")
        
        st.dataframe(filtered_data)
    elif selected_level == "Subfield":
        search_inst, search_country = st.columns(2)

        filtering_domain = st.multiselect("Select Domains" ,subfield_counts_summary["field_name"].unique(), default = subfield_counts_summary["field_name"].unique())

        filtered_data = subfield_counts[(subfield_counts["bin"] == selected_bin) & (subfield_counts["field_name"].isin(filtering_domain))].sort_values(["counts"], ascending = False)

        with search_inst:
            if st.checkbox("Select Institution"):
                inst_select = st.selectbox("Select Inst.", uni_list)
                filtered_data = filtered_data[filtered_data["Summary_Institution"].str.contains(inst_select)]

        with search_country:
            if st.checkbox("Select Country"):
                country_select = st.selectbox("Select Country", country_list)
                filtered_data = filtered_data[filtered_data["Summary_Country"].str.contains(country_select)]

        if st.checkbox("Filter by Domain/Field"):
            Domain = filtered_data["domain_name"].unique()
            Field = filtered_data["field_name"].unique()
            if st.checkbox("Filter by Domain"):
                Domain = st.multiselect("Select Domain" ,filtered_data["domain_name"].unique(), default = filtered_data["domain_name"].unique())
                filtered_data = filtered_data[filtered_data["domain_name"].isin(Domain)]
            if st.checkbox("Filter by Field"):
                Field = st.multiselect("Select Field" ,filtered_data["field_name"].unique(), default = filtered_data["field_name"].unique())
                filtered_data = filtered_data[filtered_data["field_name"].isin(Field)]
            
            st.markdown("Summary Data")
            st.dataframe(subfield_counts_summary[(subfield_counts_summary["bin"] == selected_bin) & (subfield_counts_summary["domain_name"].isin(Domain)) & (subfield_counts_summary["field_name"].isin(Field))])
        
        else:
            st.markdown("Summary Data")
            st.dataframe(subfield_counts_summary[(subfield_counts_summary["bin"] == selected_bin)])

        st.markdown("Data")
        
        st.dataframe(filtered_data)
    elif selected_level == "Name":
        search_inst, search_country = st.columns(2)

        # create a text box

        summary_subset = name_counts_summary[(name_counts_summary["bin"] == selected_bin)]

        search_entry = st.text_input("Input a Search Query (i.e. Machine Learning and Natural Language Processing, Financial Modeling)")

        filtered_data = name_counts

        if search_entry != "":

            search_entry_vector = get_word_embedding(search_entry)

            entries = summary_subset["name"].unique()

            all_scores = [cosine_similarity(get_word_embedding(k), search_entry_vector) for k in summary_subset["name"].unique()]

            all_scores_array = np.array(all_scores)

            top_5_scores = sorted(all_scores, reverse=True)[:5]

            top_5_names = []

            for score in top_5_scores:
                # Find all indices where the score matches the current top score
                indices = np.where(all_scores_array == score)[0].tolist()
                # Append these indices to the top_10_indices list
                for k in indices:
                    top_5_names.append(entries[k])        

            st.markdown(f"Query Results: {top_5_names}; Top 5 Scores: {top_5_scores}")

            filtering_domain = top_5_names

            filtered_data = name_counts[(name_counts["bin"] == selected_bin) & (name_counts["name"].isin(filtering_domain))].sort_values(["counts"], ascending = False)

        with search_inst:
            if st.checkbox("Select Institution"):
                inst_select = st.selectbox("Select Inst.", uni_list)
                filtered_data = filtered_data[filtered_data["Summary_Institution"].str.contains(inst_select)]

        with search_country:
            if st.checkbox("Select Country"):
                country_select = st.selectbox("Select Country", country_list)
                filtered_data = filtered_data[filtered_data["Summary_Country"].str.contains(country_select)]

        if st.checkbox("Filter by Domain/Field/Subfield"):
            Domain = filtered_data["domain_name"].unique()
            Field = filtered_data["field_name"].unique()
            Subfield = filtered_data["subfield_name"].unique()
            if st.checkbox("Filter by Domain"):
                Domain = st.multiselect("Select Domain" ,filtered_data["domain_name"].unique(), default = filtered_data["domain_name"].unique())
                filtered_data = filtered_data[filtered_data["domain_name"].isin(Domain)]
            if st.checkbox("Filter by Field"):
                Field = st.multiselect("Select Field" ,filtered_data["field_name"].unique(), default = filtered_data["field_name"].unique())
                filtered_data = filtered_data[filtered_data["field_name"].isin(Field)]
            if st.checkbox("Filter by Subfield"):
                Subfield = st.multiselect("Select Subfield" ,filtered_data["subfield_name"].unique(), default = filtered_data["subfield_name"].unique())
                filtered_data = filtered_data[filtered_data["subfield_name"].isin(Subfield)]
            st.markdown("Summary Data")
            st.dataframe(name_counts_summary[(name_counts_summary["bin"] == selected_bin) & (name_counts_summary["domain_name"].isin(Domain)) & (name_counts_summary["field_name"].isin(Field)) & (name_counts_summary["subfield_name"].isin(Subfield))])
    
        else:
            st.markdown("Summary Data")
            st.dataframe(name_counts_summary[(name_counts_summary["bin"] == selected_bin)])

        st.markdown("Data")
        
        st.dataframe(filtered_data)

elif page == "Interdisciplinary Research":
    print("temp")

elif page == "Research Networks":
    print("temp")