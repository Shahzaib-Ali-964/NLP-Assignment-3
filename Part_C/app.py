import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import math
import re
from collections import Counter
import pandas as pd

# Configuration
CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "n_encoder_layers": 2,
    "d_ff": 512,
    "dropout": 0.1,
    "n_decoder_layers": 2,
    "max_seq_len": 128,
}

# Tokenizer & Vocab
class WhitespaceTokenizer:
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-z0-9\\s]', ' ', text)
        return text
    def tokenize(self, text): return self.clean_text(text).split()

class Vocab:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.pad_token, self.unk_token, self.bos_token, self.eos_token = '[PAD]', '[UNK]', '[BOS]', '[EOS]'
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token, '<NEG>', '<NEU>', '<POS>', '<ELEC>', '<BOOK>', '<CLTH>']
        self.word2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
    def build(self, texts, tokenizer):
        counter = Counter()
        for text in texts: counter.update(tokenizer.tokenize(text))
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                
    def encode(self, tokens, max_len=128):
        indices = [self.word2idx.get(w, self.word2idx[self.unk_token]) for w in tokens]
        if len(indices) > max_len: indices = indices[:max_len]
        return indices + [self.word2idx[self.pad_token]] * (max_len - len(indices))
        
    def decode(self, indices):
        return [self.idx2word.get(idx, self.unk_token) for idx in indices]

# Rebuild Vocab using the exact training dataset to match indices perfectly
tokenizer = WhitespaceTokenizer()
vocab = Vocab()
train_df = pd.read_csv('../Part_A/Data/train.csv')
vocab.build(train_df['text'].tolist(), tokenizer)
metadata = torch.load('../Part_B/train_metadata.pt', map_location='cpu')

# Models
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None: scores = scores.masked_fill(mask == 1, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model, self.num_heads, self.d_k = d_model, num_heads, d_model // num_heads
        self.W_Q, self.W_K, self.W_V, self.W_O = [nn.Linear(d_model, d_model) for _ in range(4)]
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        Q = self.W_Q(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        if mask is not None:
            if mask.dim() == 2: mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3: mask = mask.unsqueeze(1)
        output, _ = scaled_dot_product_attention(Q, K, V, mask)
        return self.W_O(output.transpose(1, 2).contiguous().view(bs, -1, self.d_model))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    def forward(self, x, mask):
        x = self.norm1(x + self.dropout1(self.mha(x, x, x, mask)))
        return self.norm2(x + self.dropout2(self.ffn(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    def forward(self, x, mask=None):
        x = self.dropout(self.pos_encoding(self.embedding(x)))
        for layer in self.layers: x = layer(x, mask)
        return x

class SentimentHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.clf = nn.Linear(d_model, 3)
    def forward(self, x):
        return self.clf(x)

class DerivedFeatureHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.clf = nn.Linear(d_model, 3)
    def forward(self, x):
        return self.clf(x)

class MultiTaskModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_seq_len):
        super().__init__()
        self.encoder = EncoderOnlyTransformer(vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_seq_len)
        self.sentiment_head = SentimentHead(d_model)
        self.category_head = DerivedFeatureHead(d_model)
    def forward(self, input_ids):
        bos_tokens = torch.full((input_ids.size(0), 1), vocab.word2idx['[BOS]'], dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([bos_tokens, input_ids[:, :-1]], dim=1)
        mask = (input_ids == vocab.word2idx['[PAD]'])
        encoder_out = self.encoder(input_ids, mask)
        cls_embedding = encoder_out[:, 0, :]
        return self.sentiment_head(cls_embedding), self.category_head(cls_embedding), cls_embedding

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, x, pad_mask=None):
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        if pad_mask is not None: combined_mask = combined_mask | pad_mask.unsqueeze(1).unsqueeze(2)
        x = self.dropout(self.pos_encoding(self.embedding(x)))
        for layer in self.layers: x = layer(x, combined_mask)
        return self.fc_out(x)

# Load Models
encoder = MultiTaskModel(vocab.vocab_size, CONFIG["d_model"], CONFIG["n_heads"], CONFIG["n_encoder_layers"], CONFIG["d_ff"], CONFIG["dropout"], CONFIG["max_seq_len"])
encoder.load_state_dict(torch.load('../Part_A/encoder.pt', map_location='cpu'))
encoder.eval()

decoder = DecoderOnlyTransformer(vocab.vocab_size, CONFIG["d_model"], CONFIG["n_heads"], CONFIG["n_decoder_layers"], CONFIG["d_ff"], CONFIG["dropout"], CONFIG["max_seq_len"])
decoder.load_state_dict(torch.load('decoder.pt', map_location='cpu'))
decoder.eval()

# Retrieval Store
embeddings = F.normalize(torch.load('../Part_B/train_embeddings.pt', map_location='cpu'), p=2, dim=-1)

def retrieve_top_k(query_emb, k=2):
    scores = torch.matmul(embeddings, F.normalize(query_emb, p=2, dim=-1).T).squeeze()
    return torch.topk(scores, k)

# Pipeline
def process_review(text):
    tokens = tokenizer.tokenize(text)
    encoded = vocab.encode(tokens, max_len=CONFIG["max_seq_len"])
    input_ids = torch.tensor([encoded], dtype=torch.long)
    
    with torch.no_grad():
        sent_logits, cat_logits, q_emb = encoder(input_ids)
    
    sent_pred = torch.argmax(sent_logits, dim=-1).item()
    cat_pred = torch.argmax(cat_logits, dim=-1).item()
    
    sent_labels = ['Negative', 'Neutral', 'Positive']
    cat_labels = ['Electronics', 'Books', 'Clothing_Shoes_Jewelry']
    
    scores, indices = retrieve_top_k(q_emb, k=2)
    retrieved_texts = [metadata['texts'][i] for i in indices]
    
    context_display = "\n\n---\n\n".join([f"[{i+1}] (Sim: {scores[i]:.2f}): {txt}" for i, txt in enumerate(retrieved_texts)])
    
    # Generation Prompt
    review_tokens = encoded[:20]
    review_tokens = [t for t in review_tokens if t not in [vocab.word2idx['[PAD]'], vocab.word2idx['[BOS]']]]
    seq = [vocab.word2idx['[BOS]']] + review_tokens + [vocab.word2idx['[PAD]']]
    seq += [vocab.word2idx[['<NEG>', '<NEU>', '<POS>'][sent_pred]], vocab.word2idx['[PAD]']]
    seq += [vocab.word2idx[['<ELEC>', '<BOOK>', '<CLTH>'][cat_pred]], vocab.word2idx['[PAD]']]
    for t in retrieved_texts:
        seq += vocab.encode(tokenizer.tokenize(t), max_len=5)
        seq += [vocab.word2idx['[PAD]']]
        
    expl_str = ["this", "review", "is", ['<NEG>', '<NEU>', '<POS>'][sent_pred].lower(), "because"] + review_tokens[:3]
    seq += [vocab.word2idx.get(w, vocab.word2idx['[UNK]']) for w in expl_str]
    
    # Autoregressive decoding
    generated = list(seq)
    for _ in range(50):
        if len(generated) >= CONFIG["max_seq_len"]: break
        input_tensor = torch.tensor([generated])
        with torch.no_grad():
            logits = decoder(input_tensor)
        next_token = torch.argmax(logits[0, -1, :]).item()
        generated.append(next_token)
        if next_token == vocab.word2idx['[EOS]']: break
        
    explanation = " ".join(vocab.decode(generated[len(seq):]))
    
    return sent_labels[sent_pred], cat_labels[cat_pred], context_display, explanation

# Gradio App
interface = gr.Interface(
    fn=process_review,
    inputs=gr.Textbox(lines=5, placeholder="Paste a product review here... (e.g. This phone broke after two days!)", label="Input Review"),
    outputs=[
        gr.Textbox(label="Predicted Sentiment"),
        gr.Textbox(label="Predicted Category"),
        gr.Textbox(label="RAG Context Retrieved (Top 2 Semantically Similar Reviews)"),
        gr.Textbox(label="Generated Autoregressive Rationale/Explanation")
    ],
    title="Transformer RAG Pipeline - CS4063 NLP",
    description="A fully native PyTorch implementation of an Encoder-Decoder RAG system. It predicts sentiment and category, retrieves relevant semantic context from the Amazon dataset, and autoregressively generates an explanation."
)

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860, share=False)
