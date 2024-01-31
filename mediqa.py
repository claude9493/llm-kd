# %% [markdown]
# # EDA

# %%
from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
TRAIN_DATA_PATH = "data/MEDIQA-CORR-2024-MS-TrainingData.csv"

TRAIN_DATA_COLUMNS = [
    "Text ID", "Text", "Sentences", "Error Flag", "Error Sentence ID", "Error Sentence", "Corrected Sentence", "Corrected Text"
]

# %%
train_data = pd.read_csv(TRAIN_DATA_PATH, usecols=TRAIN_DATA_COLUMNS)
train_data_eda = deepcopy(train_data)

# %%
print(f"Number of training data: {len(train_data_eda)}")

# %%
sns.set_style("whitegrid")
plt.pie(train_data_eda["Error Flag"].value_counts(), labels=["No Error", "Error"], autopct='%1.2f%%')
plt.title("Distribution of data with error")
plt.show()

# %% [markdown]
# **More than half of the training data are labeled correct.**

# %%
train_data_eda["Len Sentences"] = train_data_eda["Sentences"].map(lambda x: len(x.strip().split('\n')))
plt.hist(train_data_eda["Len Sentences"], bins=75)
plt.xlabel("Length of Sentences")
plt.ylabel("Count")
plt.title("Distribution of sentence length")
plt.show()

# %% [markdown]
# **Most of the text are shorter than 40 sentences**

# %%
sns.violinplot(data=train_data_eda, y="Error Flag", x="Len Sentences", orient="h", split=True)
plt.title("Distribution of sentence length for text with and without error.")
plt.show()

# %% [markdown]
# **Extreme long sentence tends to be free of error**

# %%
train_data_eda["Error Sentence Relative Position"] = train_data_eda["Error Sentence ID"] / train_data_eda["Len Sentences"]
plt.hist(train_data_eda["Error Sentence Relative Position"], range=(0,1), bins=20)
plt.xlabel("Relative position of error sentence")
plt.ylabel("Count")
plt.title("Distribution of the relative position of error sentence")
plt.show()

# %% [markdown]
# **Sentences near the end of the text have higher error probability.**

# %%
from transformers import RobertaTokenizer
MODEL_PATH = "/cache/LLM/models/Roberta/base/"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH, truncation=True, do_lower_case=True)

train_data_eda["Tokenized Text Length"] = train_data_eda["Text"].map(lambda x: len(tokenizer.encode_plus(x, add_special_tokens=True)['input_ids']))

# %%
plt.hist(train_data_eda["Tokenized Text Length"], bins=50)
plt.xlabel("Length of tokenized text")
plt.ylabel("Count")
plt.title("Distribution of the length of tokenized text (Roberta)")
plt.show()

# %%
train_data_eda["Tokenized Text Length"].max()

# %% [markdown]
# # Preprocess Data

# %% [markdown]
# # Roberta Classification

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import RobertaModel, RobertaTokenizer
import logging
logging.basicConfig(level=logging.ERROR)

MODEL_PATH = "/cache/LLM/models/Roberta/base/"

# %%
train_data = pd.read_csv(TRAIN_DATA_PATH, usecols=TRAIN_DATA_COLUMNS)

# %%
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH, truncation=True, do_lower_case=True)

# %%
# Defining some key variables that will be used later on in the training
MAX_LEN = 500
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
# EPOCHS = 1
LEARNING_RATE = 1e-05

# %%
dataset = load_dataset("csv", data_files=TRAIN_DATA_PATH)

# %%
Dataset.from_pandas(train_data)


