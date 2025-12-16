# Auto-generated from LaTeX code blocks; consolidate all simulation here.
# === Begin extracted block 1 ===
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim

# Simulated climate data generation for demo purposes
# Time series length and spatial grid size
T, N = 1000, 25  

# Generate synthetic multimodal data:
#   - Temporal numeric reanalysis features (e.g. temperature, humidity): shape (T, N, features)
#   - Satellite-like spatial imagery features (lower resolution, 5 features)
features_num = 3
features_sat = 5

np.random.seed(42)

# Numeric climate time series with noise and seasonality pattern
time = np.arange(T)
seasonality = np.sin(2 * np.pi * time / 365)
numeric_data = np.random.randn(T, N, features_num) * 0.1 + seasonality[:,None,None]*0.5

# Satellite data simulated as spatial "heatmaps" with anomaly spikes
sat_data = np.random.randn(T, N, features_sat) * 0.2
peak_times = np.random.choice(range(100, 900), 10)
for pt in peak_times:
    sat_data[pt:pt+20, :, :] += np.linspace(0,3,20)[:,None,None]

# Binary labels for extreme events: 1 for event days, 0 otherwise (sparse)
labels = np.zeros(T)
labels[peak_times] = 1
labels[peak_times+1] = 1
labels[peak_times+2] = 1

# Train-test split by time (simulate future prediction)
split = int(T*0.8)

X_num_train = numeric_data[:split]
X_sat_train = sat_data[:split]
y_train = labels[:split]

X_num_test = numeric_data[split:]
X_sat_test = sat_data[split:]
y_test = labels[split:]

# Simple Hybrid Model: TCN (1D Conv) + GNN-like spatial aggregation (mean pooling here as proxy)
class SimpleHybridModel(nn.Module):
    def __init__(self, num_features, sat_features, out_dim=1):
        super().__init__()
        self.tcn = nn.Conv1d(num_features*N, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 + sat_features*N, 32)
        self.fc2 = nn.Linear(32, out_dim)
    
    def forward(self, x_num, x_sat):
        # x_num: [batch, T, N, features_num]
        # flatten spatial and feature dims for TCN
        B, T, N, Fnum = x_num.shape
        x_num = x_num.permute(0, 2, 3, 1).reshape(B, N*Fnum, T)  # [B, N*Fnum, T]
        tcn_out = self.tcn(x_num).mean(dim=2)  # [B, 64]
        
        x_sat = x_sat.reshape(B, -1)  # [B, N*Fsat]
        
        concat = torch.cat([tcn_out, x_sat], dim=1)
        h = torch.relu(self.fc1(concat))
        out = torch.sigmoid(self.fc2(h)).squeeze()
        return out

# Prepare torch tensors
def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001

X_num_train_t = to_tensor(X_num_train)
X_sat_train_t = to_tensor(X_sat_train)
y_train_t = to_tensor(y_train)

X_num_test_t = to_tensor(X_num_test)
X_sat_test_t = to_tensor(X_sat_test)
y_test_t = to_tensor(y_test)

# Simple batching function
def batch_generator(X_num, X_sat, y, batch_size):
    n = len(y)
    for i in range(0, n, batch_size):
        yield X_num[i:i+batch_size], X_sat[i:i+batch_size], y[i:i+batch_size]

# Initialize and train model
model = SimpleHybridModel(features_num, features_sat)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

for epoch in range(EPOCHS):
    model.train()
    losses = []
    for Xn, Xs, yt in batch_generator(X_num_train_t, X_sat_train_t, y_train_t, BATCH_SIZE):
        optimizer.zero_grad()
        preds = model(Xn, Xs)
        loss = criterion(preds, yt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {np.mean(losses):.4f}")

# Evaluation on test set
model.eval()
with torch.no_grad():
    preds_test = model(X_num_test_t, X_sat_test_t).numpy()
    preds_binary = (preds_test > 0.5).astype(int)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, preds_test))
f1 = f1_score(y_test, preds_binary)
precision = precision_score(y_test, preds_binary)
recall = recall_score(y_test, preds_binary)
brier = brier_score_loss(y_test, preds_test)

# Baselines for comparison (Random Forest on flattened features)
X_train_rf = np.concatenate([X_num_train.reshape(len(y_train), -1), X_sat_train.reshape(len(y_train), -1)], axis=1)
X_test_rf = np.concatenate([X_num_test.reshape(len(y_test), -1), X_sat_test.reshape(len(y_test), -1)], axis=1)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_rf, y_train)
preds_rf = rf.predict_proba(X_test_rf)[:,1]
preds_rf_binary = (preds_rf > 0.5).astype(int)

rmse_rf = np.sqrt(mean_squared_error(y_test, preds_rf))
f1_rf = f1_score(y_test, preds_rf_binary)
precision_rf = precision_score(y_test, preds_rf_binary)
recall_rf = recall_score(y_test, preds_rf_binary)
brier_rf = brier_score_loss(y_test, preds_rf)

print(f"Hybrid Model RMSE: {rmse:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Brier: {brier:.4f}")
print(f"Random Forest RMSE: {rmse_rf:.4f}, F1: {f1_rf:.4f}, Precision: {precision_rf:.4f}, Recall: {recall_rf:.4f}, Brier: {brier_rf:.4f}")

# Plotting performance comparison
metrics = ['F1-score', 'Precision', 'Recall', 'Brier Score']
hybrid_scores = [f1, precision, recall, brier]
rf_scores = [f1_rf, precision_rf, recall_rf, brier_rf]

plt.figure(figsize=(8,5))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, hybrid_scores, width, label='Hybrid Model (TCN+GNN)')
plt.bar(x + width/2, rf_scores, width, label='Random Forest')

plt.ylabel('Score')
plt.title('Performance Metrics Comparison on Test Set')
plt.xticks(x, metrics)
plt.legend()
plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.close()

# Save key results for LaTeX import
np.savetxt('results_metrics.txt', np.array([rmse, f1, precision, recall, brier, rmse_rf, f1_rf, precision_rf, recall_rf, brier_rf]))
# === End block 1 ===
