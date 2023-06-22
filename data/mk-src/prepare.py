import os
import requests
import tiktoken
import numpy as np
import pandas as pd
import csv

def read_csv_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            text = ''
            first_row = True
            for row in csv_reader:
                if first_row:
                    first_row = False
                    continue
                # Concatenate values from all columns with new lines
                text += '\n'.join(row) + '\n\n'
            with open("input.txt", "w", encoding='utf-8') as file:
                # Write the text to the file
                file.write(text)
            
            
            return text
    except IOError:
        print(f"Error: Could not read the file '{file_path}'.")

# Example usage
file_path = 'data/mk/MK_characterdata.csv'  # Replace with your actual file path
data = read_csv_file(file_path)

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
