import streamlit as st
import pandas as pd
import numpy as np

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

df = pd.read_csv('amazon_df_final.csv')
df['title'] = df['title'].str.lower()
df['ingredients'] = df['ingredients'].str.lower()

def preprocess_text(text):
    # Replace specific phrases with a single token
    text = text.replace("gluten free", "glutenfree")
    text = text.replace("allergen free", "allergenfree")
    return text

def extract_words(text):
    if pd.isna(text):  # Check for NaN values
        return []

    # Preprocess the text to handle specific phrases
    text = preprocess_text(text)
    
    # Use NLTK's word_tokenize to handle phrases
    words = word_tokenize(text)
    return words

# Apply the custom function to create a new column for each original column
df['word_list_column1'] = df['title'].apply(extract_words)
df['word_list_column2'] = df['ingredients'].apply(extract_words)

# Combine the lists from both columns into a new column
df['final_word_list'] = df['word_list_column1'] + df['word_list_column2']

df = df.drop(['word_list_column1', 'word_list_column2'], axis=1)

def clean_and_lower(word_list):
    cleaned_list = [word.lower() for word in word_list if not word.isdigit()]
    return cleaned_list

# Apply the cleaning function to the final word list column
df['final_word_list'] = df['final_word_list'].apply(clean_and_lower)

porter = PorterStemmer()

# Function to apply stemming
def apply_stemming(word_list):
    return [porter.stem(word) for word in word_list]

# Apply the stemming function to the final word list column
df['final_word_list'] = df['final_word_list'].apply(apply_stemming)


# Function to remove duplicates from the lists
def remove_duplicates(word_list):
    return list(set(word_list))

# Apply the function to remove duplicates
df['final_word_list'] = df['final_word_list'].apply(remove_duplicates)


def filter_rows_both(include_string, exclude_string):
    include_words = apply_stemming(clean_and_lower(extract_words(preprocess_text(include_string.lower()))))
    exclude_words = apply_stemming(clean_and_lower(extract_words(preprocess_text(exclude_string.lower()))))
    
    included_rows = df[df['final_word_list'].apply(lambda x: any(word in x for word in include_words))]
    excluded_rows = df[~df['final_word_list'].apply(lambda x: any(word in x for word in exclude_words))]
    
    # Combine the results
    filtered_df = pd.merge(included_rows, excluded_rows, how='inner', on='title')
    
    # Reset the index to get new numbers
    filtered_df = filtered_df.reset_index(drop=True)
    
    return filtered_df.iloc[:,:4]


st.set_page_config(page_title='My webpage', page_icon=":tada:", layout='wide' )

st.subheader("A little project")
st.title(":grey[Snack Search]")
st.write("For My Sister :)")




include = st.text_input("Include these: ", "")
exclude = st.text_input("Exclude these: ", " ")


if st.button("Filter Rows"):
    # Perform filtering
    result = filter_rows_both(include, exclude)[['title', 'url_x']]

    for index, row in result.iterrows():

        st.link_button(label=row['title'], url=row['url_x'])


  

