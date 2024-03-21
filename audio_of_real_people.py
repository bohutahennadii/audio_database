import pandas as pd
from datasets import load_dataset

# You will need to authenticate with the Hugging Face API to download the data
# !huggingface-cli login

# Load the dataset with ukrainian voice samples
cv_16_1 = load_dataset("mozilla-foundation/common_voice_16_1", "uk", split="train")

# Convert the dataset to a pandas dataframe
cv_16_1PD = cv_16_1.to_pandas()

# Select the first 11000 samples
real_voice = cv_16_1PD[:11000]
# Add a column to indicate that the voice is real
real_voice.insert(5, 'generated', "false")

# from column "path" you can get the path to the audio file and process it
# real_voice['path']