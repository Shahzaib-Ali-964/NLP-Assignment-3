import nbformat as nbf
import os

with open('i222146-NLP-Assignment3.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Phase 3
nb.cells.append(nbf.v4.new_markdown_cell("""# Phase 3: Encoder Training & Embedding Serialization
**Design Decisions**:
- **Loss Function**: Combined loss of Sentiment (alpha=1.0) and Category (beta=0.5).
- **Optimizer**: AdamW with `ReduceLROnPlateau`.
- **Metrics**: Accuracy, Macro F1, and Confusion Matrix.
- **Serialization**: Saved CLS embeddings for train set to .pt file."""))

phase3_code = """from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time

def train_encoder(model, train_loader, val_loader, config):
    optimizer = AdamW(model.parameters(), lr=config["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["patience"])
    ce_loss = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc_sent': [], 'val_acc_cat': []}
    
    for epoch in range(config["max_epochs"]):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            sent_logits, cat_logits, _ = model(batch['input_ids'])
            loss_sent = ce_loss(sent_logits, batch['sentiment'])
            loss_cat = ce_loss(cat_logits, batch['category'])
            loss = config["alpha"] * loss_sent + config["beta"] * loss_cat
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss, val_sent_preds, val_sent_labels, val_cat_preds, val_cat_labels = 0, [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                sent_logits, cat_logits, _ = model(batch['input_ids'])
                loss = config["alpha"] * ce_loss(sent_logits, batch['sentiment']) + config["beta"] * ce_loss(cat_logits, batch['category'])
                val_loss += loss.item()
                val_sent_preds.extend(torch.argmax(sent_logits, dim=-1).tolist())
                val_cat_preds.extend(torch.argmax(cat_logits, dim=-1).tolist())
                val_sent_labels.extend(batch['sentiment'].tolist())
                val_cat_labels.extend(batch['category'].tolist())
                
        val_loss /= len(val_loader)
        val_acc_sent = accuracy_score(val_sent_labels, val_sent_preds)
        val_acc_cat = accuracy_score(val_cat_labels, val_cat_preds)
        
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['val_acc_sent'].append(val_acc_sent); history['val_acc_cat'].append(val_acc_cat)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Sent Acc={val_acc_sent:.4f}, Val Cat Acc={val_acc_cat:.4f}")
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config["models_dir"], 'encoder.pt'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config["patience"]:
                print("Early stopping triggered.")
                break
    return history

print("Starting Encoder Training...")
history = train_encoder(model, train_loader, val_loader, CONFIG)

plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Encoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(CONFIG["results_dir"], 'encoder_loss.png'))
plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(phase3_code))

phase3_eval_code = """model.load_state_dict(torch.load(os.path.join(CONFIG["models_dir"], 'encoder.pt')))
model.eval()

test_sent_preds, test_sent_labels, test_cat_preds, test_cat_labels = [], [], [], []
with torch.no_grad():
    for batch in test_loader:
        sent_logits, cat_logits, _ = model(batch['input_ids'])
        test_sent_preds.extend(torch.argmax(sent_logits, dim=-1).tolist())
        test_cat_preds.extend(torch.argmax(cat_logits, dim=-1).tolist())
        test_sent_labels.extend(batch['sentiment'].tolist())
        test_cat_labels.extend(batch['category'].tolist())

print("Test Sentiment Accuracy:", accuracy_score(test_sent_labels, test_sent_preds))
print("Test Sentiment Macro F1:", f1_score(test_sent_labels, test_sent_preds, average='macro'))
print("Test Category Accuracy:", accuracy_score(test_cat_labels, test_cat_preds))
print("Test Category Macro F1:", f1_score(test_cat_labels, test_cat_preds, average='macro'))

train_embeddings = []
train_metadata = {'texts': [], 'sentiment_labels': [], 'category_labels': [], 'indices': []}
print("Extracting train embeddings...")
with torch.no_grad():
    for batch in train_loader:
        _, _, cls_emb = model(batch['input_ids'])
        train_embeddings.append(cls_emb.cpu())
        train_metadata['texts'].extend(batch['text'])
        train_metadata['sentiment_labels'].extend(batch['sentiment'].tolist())
        train_metadata['category_labels'].extend(batch['category'].tolist())
        train_metadata['indices'].extend(batch['index'].tolist())

train_embeddings = torch.cat(train_embeddings, dim=0)
torch.save(train_embeddings, os.path.join(CONFIG["results_dir"], 'train_embeddings.pt'))
torch.save(train_metadata, os.path.join(CONFIG["results_dir"], 'train_metadata.pt'))
print(f"Saved {len(train_embeddings)} embeddings of shape {train_embeddings.shape}")

log_data = {'run_id': 'run_001', 'd_model': CONFIG["d_model"], 'n_heads': CONFIG["n_heads"], 'n_layers': CONFIG["n_encoder_layers"],
            'd_ff': CONFIG["d_ff"], 'lr': CONFIG["lr"], 'dropout': CONFIG["dropout"], 'batch_size': CONFIG["batch_size"],
            'max_seq_len': CONFIG["max_seq_len"], 'val_loss': min(history['val_loss']), 'val_acc_sentiment': max(history['val_acc_sent']),
            'val_metric_derived': max(history['val_acc_cat']), 'notes': 'Initial encoder-only training'}
pd.DataFrame([log_data]).to_csv(os.path.join(CONFIG["results_dir"], 'hyperparam_log.csv'), index=False)
"""
nb.cells.append(nbf.v4.new_code_cell(phase3_eval_code))
nb.cells.append(nbf.v4.new_code_cell("""# GIT CHECKPOINT - commit message:\n# "feat: training pipeline and embedding serialization" """))

# Phase 4
nb.cells.append(nbf.v4.new_markdown_cell("""# Phase 4: Retrieval Module\n**Design Decisions**: Cosine similarity via dot product on L2-normalized embeddings."""))

phase4_code = """class EmbeddingStore:
    def __init__(self, embeddings_path, metadata_path):
        self.embeddings = F.normalize(torch.load(embeddings_path), p=2, dim=-1)
        self.metadata = torch.load(metadata_path)
        
    def retrieve_top_k(self, query_emb, k=5):
        scores = torch.matmul(self.embeddings, F.normalize(query_emb, p=2, dim=-1).T).squeeze()
        return torch.topk(scores, k)

store = EmbeddingStore(os.path.join(CONFIG["results_dir"], 'train_embeddings.pt'), os.path.join(CONFIG["results_dir"], 'train_metadata.pt'))

test_samples = [test_dataset[i] for i in range(3)]
model.eval()
print("\\nRetrieval Quality Analysis:")
for idx, sample in enumerate(test_samples):
    with torch.no_grad(): _, _, q_emb = model(sample['input_ids'].unsqueeze(0))
    scores, indices = store.retrieve_top_k(q_emb, k=CONFIG["top_k"])
    print(f"\\n--- Query {idx+1} ---")
    print(f"Text: {sample['text']}\\nMean Sim: {scores.mean().item():.4f}")
    for i, res_idx in enumerate(indices):
        print(f"  [{i+1}] Score: {scores[i].item():.4f} | Text: {store.metadata['texts'][res_idx.item()][:100]}...")

print("\\nk Sensitivity Analysis:")
for k_val in [1, 3, 5, 10]:
    all_means = []
    for sample in test_samples:
        with torch.no_grad(): _, _, q_emb = model(sample['input_ids'].unsqueeze(0))
        scores, _ = store.retrieve_top_k(q_emb, k=k_val)
        all_means.append(scores.mean().item())
    print(f"Average mean similarity for k={k_val}: {sum(all_means)/len(all_means):.4f}")
"""
nb.cells.append(nbf.v4.new_code_cell(phase4_code))
nb.cells.append(nbf.v4.new_code_cell("""# GIT CHECKPOINT - commit message:\n# "feat: retrieval module with cosine similarity search" """))

with open('i222146-NLP-Assignment3.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
