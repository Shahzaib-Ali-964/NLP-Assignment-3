import nbformat as nbf
import os

with open('i220576-NLP-Assignment3.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Phase 3
nb.cells.append(nbf.v4.new_markdown_cell("""# Phase 3: Encoder Training & Embedding Serialization
**Design Decisions**:
- **Loss Function**: Combined loss of Polarity (sent_weight=1.0) and Category (cat_weight=0.5).
- **Optimizer**: AdamW with `ReduceLROnPlateau`.
- **Metrics**: Accuracy, Macro F1, and Confusion Matrix.
- **Serialization**: Saved CLS embeddings for train set to .pt file."""))

phase3_code = """from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time

def run_encoder_training(enc_model, tr_loader, vl_loader, config):
    optimizer = AdamW(enc_model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["early_stop_patience"])
    ce_loss = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    early_stop_counter = 0
    metrics = {'train_loss': [], 'val_loss': [], 'val_acc_sent': [], 'val_acc_cat': []}

    for epoch in range(config["epoch_limit"]):
        enc_model.train()
        total_loss = 0
        for batch in tr_loader:
            optimizer.zero_grad()
            sent_logits, cat_logits, _ = enc_model(batch['input_ids'])
            loss_sent = ce_loss(sent_logits, batch['sentiment'])
            loss_cat = ce_loss(cat_logits, batch['category'])
            loss = config["sent_weight"] * loss_sent + config["cat_weight"] * loss_cat
            loss.backward()
            nn.utils.clip_grad_norm_(enc_model.parameters(), config["gradient_max_norm"])
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(tr_loader)

        enc_model.eval()
        val_loss, val_sent_preds, val_sent_labels, val_cat_preds, val_cat_labels = 0, [], [], [], []
        with torch.no_grad():
            for batch in vl_loader:
                sent_logits, cat_logits, _ = enc_model(batch['input_ids'])
                loss = config["sent_weight"] * ce_loss(sent_logits, batch['sentiment']) + config["cat_weight"] * ce_loss(cat_logits, batch['category'])
                val_loss += loss.item()
                val_sent_preds.extend(torch.argmax(sent_logits, dim=-1).tolist())
                val_cat_preds.extend(torch.argmax(cat_logits, dim=-1).tolist())
                val_sent_labels.extend(batch['sentiment'].tolist())
                val_cat_labels.extend(batch['category'].tolist())

        val_loss /= len(vl_loader)
        val_acc_sent = accuracy_score(val_sent_labels, val_sent_preds)
        val_acc_cat = accuracy_score(val_cat_labels, val_cat_preds)

        metrics['train_loss'].append(train_loss); metrics['val_loss'].append(val_loss)
        metrics['val_acc_sent'].append(val_acc_sent); metrics['val_acc_cat'].append(val_acc_cat)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Sent Acc={val_acc_sent:.4f}, Val Cat Acc={val_acc_cat:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(enc_model.state_dict(), os.path.join(config["checkpoint_dir"], 'encoder.pt'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config["early_stop_patience"]:
                print("Early stopping triggered.")
                break
    return metrics

print("Starting Encoder Training...")
metrics = run_encoder_training(enc_model, tr_loader, vl_loader, PARAMS)

plt.figure(figsize=(10, 5))
plt.plot(metrics['train_loss'], label='Train Loss')
plt.plot(metrics['val_loss'], label='Val Loss')
plt.title('Encoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(PARAMS["output_dir"], 'encoder_loss.png'))
plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(phase3_code))

phase3_eval_code = """enc_model.load_state_dict(torch.load(os.path.join(PARAMS["checkpoint_dir"], 'encoder.pt')))
enc_model.eval()

test_sent_preds, test_sent_labels, test_cat_preds, test_cat_labels = [], [], [], []
with torch.no_grad():
    for batch in te_loader:
        sent_logits, cat_logits, _ = enc_model(batch['input_ids'])
        test_sent_preds.extend(torch.argmax(sent_logits, dim=-1).tolist())
        test_cat_preds.extend(torch.argmax(cat_logits, dim=-1).tolist())
        test_sent_labels.extend(batch['sentiment'].tolist())
        test_cat_labels.extend(batch['category'].tolist())

print("Test Sentiment Accuracy:", accuracy_score(test_sent_labels, test_sent_preds))
print("Test Sentiment Macro F1:", f1_score(test_sent_labels, test_sent_preds, average='macro'))
print("Test Category Accuracy:", accuracy_score(test_cat_labels, test_cat_preds))
print("Test Category Macro F1:", f1_score(test_cat_labels, test_cat_preds, average='macro'))

corpus_vectors = []
corpus_meta = {'texts': [], 'sentiment_labels': [], 'category_labels': [], 'indices': []}
print("Extracting train embeddings...")
with torch.no_grad():
    for batch in tr_loader:
        _, _, cls_emb = enc_model(batch['input_ids'])
        corpus_vectors.append(cls_emb.cpu())
        corpus_meta['texts'].extend(batch['text'])
        corpus_meta['sentiment_labels'].extend(batch['sentiment'].tolist())
        corpus_meta['category_labels'].extend(batch['category'].tolist())
        corpus_meta['indices'].extend(batch['index'].tolist())

corpus_vectors = torch.cat(corpus_vectors, dim=0)
torch.save(corpus_vectors, os.path.join(PARAMS["output_dir"], 'train_embeddings.pt'))
torch.save(corpus_meta, os.path.join(PARAMS["output_dir"], 'train_metadata.pt'))
print(f"Saved {len(corpus_vectors)} embeddings of shape {corpus_vectors.shape}")

log_data = {'run_id': 'run_001', 'embed_dim': PARAMS["embed_dim"], 'num_attn_heads': PARAMS["num_attn_heads"],
            'enc_depth': PARAMS["enc_depth"], 'feedforward_dim': PARAMS["feedforward_dim"],
            'learning_rate': PARAMS["learning_rate"], 'dropout': PARAMS["dropout"], 'mini_batch': PARAMS["mini_batch"],
            'max_seq_len': PARAMS["max_seq_len"], 'val_loss': min(metrics['val_loss']),
            'val_acc_sentiment': max(metrics['val_acc_sent']), 'val_metric_derived': max(metrics['val_acc_cat']),
            'notes': 'Initial encoder-only training'}
pd.DataFrame([log_data]).to_csv(os.path.join(PARAMS["output_dir"], 'hyperparam_log.csv'), index=False)
"""
nb.cells.append(nbf.v4.new_code_cell(phase3_eval_code))
nb.cells.append(nbf.v4.new_code_cell("""# GIT CHECKPOINT - commit message:\n# "feat: training pipeline and embedding serialization" """))

# Phase 4
nb.cells.append(nbf.v4.new_markdown_cell("""# Phase 4: Retrieval Module
**Design Decisions**: Cosine similarity via dot product on L2-normalized embeddings."""))

phase4_code = """class VectorIndex:
    def __init__(self, embeddings_path, metadata_path):
        self.embeddings = F.normalize(torch.load(embeddings_path), p=2, dim=-1)
        self.metadata = torch.load(metadata_path)

    def retrieve_top_k(self, query_emb, k=5):
        scores = torch.matmul(self.embeddings, F.normalize(query_emb, p=2, dim=-1).T).squeeze()
        return torch.topk(scores, k)

vec_index = VectorIndex(os.path.join(PARAMS["output_dir"], 'train_embeddings.pt'), os.path.join(PARAMS["output_dir"], 'train_metadata.pt'))

probe_items = [te_data[i] for i in range(3)]
enc_model.eval()
print("\\nRetrieval Quality Analysis:")
for idx, sample in enumerate(probe_items):
    with torch.no_grad(): _, _, q_emb = enc_model(sample['input_ids'].unsqueeze(0))
    scores, indices = vec_index.retrieve_top_k(q_emb, k=PARAMS["num_neighbors"])
    print(f"\\n--- Query {idx+1} ---")
    print(f"Text: {sample['text']}\\nMean Sim: {scores.mean().item():.4f}")
    for i, res_idx in enumerate(indices):
        print(f"  [{i+1}] Score: {scores[i].item():.4f} | Text: {vec_index.metadata['texts'][res_idx.item()][:100]}...")

print("\\nk Sensitivity Analysis:")
for k_val in [1, 3, 5, 10]:
    all_means = []
    for sample in probe_items:
        with torch.no_grad(): _, _, q_emb = enc_model(sample['input_ids'].unsqueeze(0))
        scores, _ = vec_index.retrieve_top_k(q_emb, k=k_val)
        all_means.append(scores.mean().item())
    print(f"Average mean similarity for k={k_val}: {sum(all_means)/len(all_means):.4f}")
"""
nb.cells.append(nbf.v4.new_code_cell(phase4_code))
nb.cells.append(nbf.v4.new_code_cell("""# GIT CHECKPOINT - commit message:\n# "feat: retrieval module with cosine similarity search" """))

with open('i220576-NLP-Assignment3.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
