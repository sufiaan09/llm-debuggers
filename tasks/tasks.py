"""
tasks.py — All debugging tasks: Easy, Medium, Hard
Each task has buggy PyTorch code, error message, test cases, and ground truth.
"""

TASKS = {

    # =========================================================
    # EASY — Syntax & Basic Runtime Errors
    # =========================================================
    "easy": [
        {
            "id": "easy_001",
            "title": "Missing optimizer.zero_grad()",
            "task_description": (
                "This PyTorch training step should compute loss and update weights. "
                "It runs but the model never converges properly."
            ),
            "buggy_code": """
import torch
import torch.nn as nn

def train_step(model, inputs, labels, optimizer, criterion):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()   # BUG: gradients accumulate across steps!
    return loss.item()
""".strip(),
            "error_message": (
                "No explicit error. Model loss oscillates and never converges. "
                "Gradients accumulate across steps causing unstable training."
            ),
            "bug_type": "logic_error",
            "ground_truth_fix": """
import torch
import torch.nn as nn

def train_step(model, inputs, labels, optimizer, criterion):
    optimizer.zero_grad()   # FIX: clear gradients before backward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
""".strip(),
            "test_cases": [
                {
                    "description": "Gradients should be zeroed before backward",
                    "check": "zero_grad_called_before_backward"
                },
                {
                    "description": "Loss should decrease over multiple steps",
                    "check": "loss_decreases"
                },
                {
                    "description": "optimizer.step() must be called",
                    "check": "step_called"
                }
            ],
            "explanation": (
                "Missing optimizer.zero_grad() causes gradients to accumulate "
                "across batches, making the gradient signal incorrect and training unstable."
            )
        },
        {
            "id": "easy_002",
            "title": "Wrong loss function for multi-class classification",
            "task_description": (
                "A 10-class image classifier. Should use CrossEntropyLoss "
                "but uses MSELoss causing terrible accuracy."
            ),
            "buggy_code": """
import torch
import torch.nn as nn

model = nn.Linear(784, 10)
criterion = nn.MSELoss()   # BUG: wrong loss for classification!
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(inputs, labels):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels.float())  # BUG: forcing float labels
    loss.backward()
    optimizer.step()
    return loss.item()
""".strip(),
            "error_message": (
                "Model trains without error but achieves only ~10% accuracy "
                "(random chance). MSELoss is incorrect for classification tasks."
            ),
            "bug_type": "logic_error",
            "ground_truth_fix": """
import torch
import torch.nn as nn

model = nn.Linear(784, 10)
criterion = nn.CrossEntropyLoss()   # FIX: correct loss for multi-class
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(inputs, labels):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)   # FIX: labels stay as long integers
    loss.backward()
    optimizer.step()
    return loss.item()
""".strip(),
            "test_cases": [
                {"description": "Uses CrossEntropyLoss", "check": "uses_cross_entropy"},
                {"description": "Labels are not cast to float", "check": "labels_are_long"},
                {"description": "Model achieves >50% accuracy after training", "check": "accuracy_above_50"}
            ],
            "explanation": (
                "MSELoss treats classification as regression. "
                "CrossEntropyLoss is correct for multi-class classification "
                "as it applies softmax + negative log likelihood."
            )
        },
        {
            "id": "easy_003",
            "title": "Model not set to eval mode during inference",
            "task_description": (
                "Inference function should return deterministic predictions "
                "but results vary each call due to dropout still being active."
            ),
            "buggy_code": """
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def predict(model, inputs):
    # BUG: model stays in training mode — dropout is active!
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.argmax(dim=1)
""".strip(),
            "error_message": (
                "No error raised. But predictions are non-deterministic — "
                "calling predict() twice with same input gives different results."
            ),
            "bug_type": "logic_error",
            "ground_truth_fix": """
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def predict(model, inputs):
    model.eval()   # FIX: disable dropout and batchnorm randomness
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.argmax(dim=1)
""".strip(),
            "test_cases": [
                {"description": "model.eval() called before inference", "check": "eval_mode_set"},
                {"description": "Predictions are deterministic across calls", "check": "deterministic_output"},
                {"description": "torch.no_grad() is used", "check": "no_grad_used"}
            ],
            "explanation": (
                "Without model.eval(), Dropout randomly zeros activations during inference, "
                "making predictions non-deterministic. eval() disables Dropout and fixes BatchNorm."
            )
        }
    ],

    # =========================================================
    # MEDIUM — Tensor Shape Mismatches
    # =========================================================
    "medium": [
        {
            "id": "medium_001",
            "title": "Wrong reshape breaking batch dimension",
            "task_description": (
                "CNN forward pass flattens feature maps before the linear layer. "
                "Crashes with RuntimeError on shape mismatch."
            ),
            "buggy_code": """
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.fc   = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        # x shape: [batch, 1, 28, 28]
        x = torch.relu(self.conv(x))
        # x shape now: [batch, 32, 26, 26]
        x = x.view(32 * 26 * 26, -1)   # BUG: loses batch dimension!
        return self.fc(x)
""".strip(),
            "error_message": (
                "RuntimeError: mat1 and mat2 shapes cannot be multiplied "
                "(21632x1 and 21632x10). The batch dimension was collapsed."
            ),
            "bug_type": "shape_mismatch",
            "ground_truth_fix": """
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.fc   = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        # x shape: [batch, 1, 28, 28]
        x = torch.relu(self.conv(x))
        # x shape now: [batch, 32, 26, 26]
        x = x.view(x.size(0), -1)   # FIX: preserve batch dim, flatten rest
        return self.fc(x)
""".strip(),
            "test_cases": [
                {"description": "Batch dimension is preserved after flatten", "check": "batch_dim_preserved"},
                {"description": "Output shape is [batch, 10]", "check": "output_shape_correct"},
                {"description": "Forward pass runs without RuntimeError", "check": "no_runtime_error"}
            ],
            "explanation": (
                "x.view(32*26*26, -1) treats features as batch size. "
                "Correct fix: x.view(x.size(0), -1) preserves batch dimension "
                "and flattens all other dimensions."
            )
        },
        {
            "id": "medium_002",
            "title": "Mismatched embedding + linear layer dimensions",
            "task_description": (
                "Text classifier uses embedding layer feeding into linear. "
                "Crashes due to wrong input size to the linear layer."
            ),
            "buggy_code": """
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)  # BUG: ignores seq_len

    def forward(self, x):
        # x shape: [batch, seq_len] — e.g., [32, 50]
        embedded = self.embedding(x)
        # embedded shape: [batch, seq_len, embed_dim] — e.g., [32, 50, 128]
        out = self.fc(embedded)   # BUG: fc expects [batch, 128] not [batch, 50, 128]
        return out
""".strip(),
            "error_message": (
                "Output shape is [32, 50, 5] instead of [32, 5]. "
                "The sequence dimension was not reduced before classification."
            ),
            "bug_type": "shape_mismatch",
            "ground_truth_fix": """
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: [batch, seq_len]
        embedded = self.embedding(x)
        # embedded shape: [batch, seq_len, embed_dim]
        pooled = embedded.mean(dim=1)   # FIX: mean pooling over sequence
        # pooled shape: [batch, embed_dim]
        out = self.fc(pooled)
        # out shape: [batch, num_classes]
        return out
""".strip(),
            "test_cases": [
                {"description": "Output shape is [batch, num_classes]", "check": "output_shape_2d"},
                {"description": "Sequence dimension reduced via pooling", "check": "sequence_pooled"},
                {"description": "No shape mismatch error", "check": "no_error"}
            ],
            "explanation": (
                "The embedding produces a 3D tensor [batch, seq_len, embed_dim]. "
                "Before feeding to the linear layer, the sequence must be reduced "
                "via mean pooling, max pooling, or using the [CLS] token."
            )
        },
        {
            "id": "medium_003",
            "title": "Wrong matrix multiplication order in attention",
            "task_description": (
                "Simple dot-product attention mechanism. "
                "RuntimeError due to wrong matmul operand order."
            ),
            "buggy_code": """
import torch
import torch.nn.functional as F

def dot_product_attention(query, key, value):
    # query: [batch, seq_len, d_k]
    # key:   [batch, seq_len, d_k]
    # value: [batch, seq_len, d_v]

    d_k = query.size(-1)
    # BUG: key not transposed — matmul dimensions don't align
    scores = torch.matmul(query, key) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value)
""".strip(),
            "error_message": (
                "RuntimeError: Expected size for first two dimensions of batch2 "
                "tensor to be: [batch, d_k] but got: [batch, seq_len]."
            ),
            "bug_type": "shape_mismatch",
            "ground_truth_fix": """
import torch
import torch.nn.functional as F

def dot_product_attention(query, key, value):
    # query: [batch, seq_len, d_k]
    # key:   [batch, seq_len, d_k]
    # value: [batch, seq_len, d_v]

    d_k = query.size(-1)
    # FIX: transpose key's last two dims for correct matmul
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    # scores shape: [batch, seq_len, seq_len]
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value)
""".strip(),
            "test_cases": [
                {"description": "key is transposed before matmul", "check": "key_transposed"},
                {"description": "scores shape is [batch, seq_len, seq_len]", "check": "scores_shape_correct"},
                {"description": "Output shape matches value shape", "check": "output_shape_matches_value"}
            ],
            "explanation": (
                "For dot-product attention, key must be transposed with .transpose(-2, -1) "
                "so matmul computes [seq_len, d_k] @ [d_k, seq_len] = [seq_len, seq_len] attention scores."
            )
        }
    ],

    # =========================================================
    # HARD — Gradient Flow & Training Loop Bugs
    # =========================================================
    "hard": [
        {
            "id": "hard_001",
            "title": "Detached tensor breaks gradient flow",
            "task_description": (
                "Custom training loop with gradient checkpointing. "
                "Model trains without error but weights never update — "
                "gradients are None everywhere."
            ),
            "buggy_code": """
import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 10)

    def forward(self, x):
        h = torch.relu(self.layer1(x))
        h = h.detach()   # BUG: detaches h from computation graph!
        return self.layer2(h)

model = TwoLayerNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def train_step(inputs, labels):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    # layer1 gradients will be None — weights never update!
    return loss.item()
""".strip(),
            "error_message": (
                "No error raised. Loss appears to decrease slightly but model "
                "does not learn. Inspecting model.layer1.weight.grad reveals None — "
                "gradients cannot flow through the detached tensor."
            ),
            "bug_type": "gradient_issue",
            "ground_truth_fix": """
import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 10)

    def forward(self, x):
        h = torch.relu(self.layer1(x))
        # FIX: removed .detach() — gradient flows through both layers
        return self.layer2(h)

model = TwoLayerNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def train_step(inputs, labels):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
""".strip(),
            "test_cases": [
                {"description": "layer1 gradients are not None after backward", "check": "layer1_grad_exists"},
                {"description": "layer1 weights change after optimizer.step()", "check": "layer1_weights_update"},
                {"description": "Loss decreases consistently over 10 steps", "check": "loss_decreases_consistently"},
                {"description": ".detach() not called in forward pass", "check": "no_detach_in_forward"}
            ],
            "explanation": (
                ".detach() cuts the tensor from the computation graph. "
                "Any layers before the detach point receive no gradients — "
                "their weights freeze. Removing .detach() restores full gradient flow."
            )
        },
        {
            "id": "hard_002",
            "title": "Vanishing gradients from wrong activation + init",
            "task_description": (
                "Deep 8-layer network uses sigmoid activations. "
                "Trains but first few layers have near-zero gradients — "
                "classic vanishing gradient problem."
            ),
            "buggy_code": """
import torch
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(256, 256) for _ in range(8)
        ])
        self.output = nn.Linear(256, 10)

        # BUG: Xavier init designed for tanh, not sigmoid
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))   # BUG: sigmoid saturates → vanishing grad
        return self.output(x)
""".strip(),
            "error_message": (
                "No error. After 100 training steps, first 3 layers have "
                "gradient magnitudes < 1e-7. Model effectively only trains "
                "the last 2-3 layers."
            ),
            "bug_type": "gradient_issue",
            "ground_truth_fix": """
import torch
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(256, 256) for _ in range(8)
        ])
        self.output = nn.Linear(256, 10)

        # FIX: Kaiming init designed for ReLU activations
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))   # FIX: ReLU doesn't saturate → healthy gradients
        return self.output(x)
""".strip(),
            "test_cases": [
                {"description": "Uses ReLU not sigmoid", "check": "uses_relu"},
                {"description": "Uses Kaiming init not Xavier", "check": "uses_kaiming_init"},
                {"description": "First layer gradient magnitude > 1e-4", "check": "first_layer_grad_healthy"},
                {"description": "Loss decreases for all 8 layers", "check": "all_layers_learn"}
            ],
            "explanation": (
                "Sigmoid saturates for large inputs, producing near-zero gradients. "
                "In deep networks this vanishes exponentially through layers. "
                "ReLU with Kaiming initialization maintains healthy gradient magnitudes."
            )
        },
        {
            "id": "hard_003",
            "title": "Custom loss function breaks autograd",
            "task_description": (
                "Custom contrastive loss implemented with in-place operations. "
                "RuntimeError on backward pass — computation graph corrupted."
            ),
            "buggy_code": """
import torch
import torch.nn as nn

def contrastive_loss(embeddings, labels, margin=1.0):
    batch_size = embeddings.size(0)
    loss = torch.zeros(1, requires_grad=True)

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            dist = torch.norm(embeddings[i] - embeddings[j])
            if labels[i] == labels[j]:
                # BUG: in-place += corrupts computation graph
                loss += dist ** 2
            else:
                loss += torch.clamp(margin - dist, min=0) ** 2

    return loss / (batch_size * (batch_size - 1) / 2)
""".strip(),
            "error_message": (
                "RuntimeError: one of the variables needed for gradient computation "
                "has been modified by an inplace operation. "
                "The in-place += on a leaf tensor with requires_grad=True corrupts autograd."
            ),
            "bug_type": "gradient_issue",
            "ground_truth_fix": """
import torch
import torch.nn.functional as F

def contrastive_loss(embeddings, labels, margin=1.0):
    batch_size = embeddings.size(0)
    losses = []   # FIX: collect losses in a list, sum at end

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            dist = torch.norm(embeddings[i] - embeddings[j])
            if labels[i] == labels[j]:
                losses.append(dist ** 2)
            else:
                losses.append(torch.clamp(margin - dist, min=0) ** 2)

    # FIX: stack and mean — no in-place operations
    return torch.stack(losses).mean()
""".strip(),
            "test_cases": [
                {"description": "No in-place operations on requires_grad tensors", "check": "no_inplace_ops"},
                {"description": "backward() completes without RuntimeError", "check": "backward_succeeds"},
                {"description": "Gradients flow to embeddings", "check": "embedding_grads_exist"},
                {"description": "Loss is scalar tensor with grad_fn", "check": "loss_has_grad_fn"}
            ],
            "explanation": (
                "In-place operations (+=) on tensors that require gradients corrupt "
                "PyTorch's autograd graph. Fix: collect losses in a Python list "
                "and reduce with torch.stack().mean() — preserving the computation graph."
            )
        }
    ]
}
