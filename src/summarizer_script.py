import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tqdm
import sys
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import csv
import io
import torch
import gc
torch.cuda.empty_cache()

def load_csv_skip_null_bytes(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read().replace('\0', '')  # remove null chars
    from io import StringIO
    return pd.read_csv(StringIO(text), on_bad_lines='skip')


def load_model(model_name="sshleifer/distilbart-cnn-12-6"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def enforce_max_length_chunks(chunks, tokenizer, max_tokens=1024):
    new_chunks = []
    for chunk in chunks:
        tokenized = tokenizer.encode(chunk, add_special_tokens=True)
        if len(tokenized) <= max_tokens:
            new_chunks.append(chunk)
        else:
            # split this chunk into smaller pieces by tokens
            for i in range(0, len(tokenized), max_tokens):
                part_tokens = tokenized[i:i+max_tokens]
                part_text = tokenizer.decode(part_tokens, skip_special_tokens=True)
                new_chunks.append(part_text)
    return new_chunks


def chunk_text(text, tokenizer, max_tokens=512):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)
        if len(tokenized_sentence) > max_tokens:
            # Sentence too long, split by tokens
            for i in range(0, len(tokenized_sentence), max_tokens):
                part_tokens = tokenized_sentence[i:i+max_tokens]
                part_text = tokenizer.decode(part_tokens, skip_special_tokens=True)
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.append(part_text)
        else:
            tentative_chunk = current_chunk + " " + sentence if current_chunk else sentence
            if len(tokenizer.encode(tentative_chunk, add_special_tokens=True)) <= max_tokens:
                current_chunk = tentative_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def summarize_long_document(text, tokenizer, model, device,
                            max_chunk_tokens=512,
                            chunk_summary_max_length=150,
                            num_beams=4,
                            length_penalty=2.0):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return np.nan

    chunks = chunk_text(text, tokenizer, max_tokens=max_chunk_tokens)
    chunks = enforce_max_length_chunks(chunks, tokenizer, max_tokens=1024) ## Since some texts are too long
    if not chunks:
        return np.nan

    intermediate_summaries = []
    for chunk_text_str in chunks:
        inputs = tokenizer.encode(chunk_text_str, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            summary_ids = model.generate(
                inputs,
                max_length=chunk_summary_max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            intermediate_summaries.append(summary)

        # Clear variables and free GPU memory
        del inputs, summary_ids
        torch.cuda.empty_cache()

    combined_summary = " ".join(intermediate_summaries)
    return combined_summary


def main():
    # Load model, tokenizer, and device
    tokenizer, model, device = load_model()

    # Example DataFrame (replace with your own)
    
    csv.field_size_limit(sys.maxsize)
    '''clean_lines = []
    with open('output_data/content_data_checkpoint.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            try:
                clean_lines.append(row)
            except csv.Error:
                print(f"Skipping bad line {i+1}")

    # Now save cleaned lines to a new file
    with open('output_data/content_data_checkpoint.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(clean_lines)
    '''

    #df = pd.read_csv('output_data/content_data_checkpoint.csv', on_bad_lines='skip')
    input_path = 'output_data/content_data_checkpoint.csv'
    output_path = 'output_data/summarized_reports.csv'
    batch_size = 20

    # Clean null bytes in-memory and prepare iterator
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        data = f.read().replace('\0', '')
    df_iter = pd.read_csv(io.StringIO(data), on_bad_lines='skip', chunksize=batch_size)

    # Write header once
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("Summary\n")

    for chunk_df in tqdm.tqdm(df_iter):
        # Summarize batch
        chunk_df['Summary'] = chunk_df['Extracted Content'].apply(
            lambda x: summarize_long_document(x, tokenizer, model, device) if pd.notna(x) else float('nan')
        )

        # Drop original large column
        chunk_df = chunk_df.drop(columns=['Extracted Content'])

        # Append batch summaries to CSV
        chunk_df.to_csv(output_path, mode='a', index=False, header=False)

        # Clear memory
        del chunk_df
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
