import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string
pd.options.mode.chained_assignment = None

# Load the data
full_df = pd.read_csv("../input/customer-support-on-twitter/twcs/twcs.csv", nrows=5000)
df = full_df[["text"]]
df["text"] = df["text"].astype(str)

# Remove lowercase letters
df["text"] = df["text"].apply(lambda x: re.sub(r'[a-z]', '', x))

# Remove punctuation
df["text"] = df["text"].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Display the first few rows of the DataFrame
df.head()
