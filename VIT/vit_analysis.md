# Vision Transformer (ViT) — Architecture, Classification vs Detection, and Working Principle

> **Documentation Reference:** [Hugging Face Transformers - ViT](https://huggingface.co/docs/transformers/en/model_doc/vit)  
> **Original Paper:** *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"* (Dosovitskiy et al., Google Research, Brain Team, ICLR 2021)

---

## 1. Is ViT a Recognition Model or a Detection Model?

**ViT is a RECOGNITION (Image Classification & Feature Representation) model type, NOT an object detection model.**

### Key Distinctions in Computer Vision:

| Capability | Object Detection Model (e.g., SCRFD, YuNet, YOLO) | Recognition / Classification Model (e.g., ViT, ArcFace, ResNet) |
| :--- | :--- | :--- |
| **Primary Goal** | **"Where is it?" + "What is it?"** | **"What is this image / who is this person?"** |
| **Output** | Bounding box coordinates $[x, y, w, h]$, confidence score, and optional landmarks. | Class probabilities (Logits) or high-dimensional feature embeddings (e.g., 512-d or 768-d vector). |
| **Input** | Full camera frame or scene with zero, one, or multiple objects. | A cropped, single object/face or an entire scene categorized as one class. |
| **Native ViT Role** | ❌ Not designed out-of-the-box to output bounding boxes or predict multiple local bounding regions. |  **Directly trained for Image Recognition / Classification** using an image-level `[CLS]` token. |

> **Context for Face Recognition in your Repo:**  
> In your face pipeline (`SCRFD -> Crop/Align -> Recognition Model`):
> - **Detector:** SCRFD or YuNet detects bounding boxes and facial landmarks.
> - **Recognizer:** ArcFace, EdgeFace, MobileFace, or **ViT** takes the aligned cropped face and outputs a 1D embedding vector for cosine similarity matching.  
> If ViT is used in face recognition (e.g., Face-ViT), it acts as the **feature extractor / recognizer**.

*(Note: While ViT backbones can be adapted into complex object detection frameworks like ViTDet or DETR, the standard Hugging Face ViT models—`ViTModel` and `ViTForImageClassification`—are strictly recognition/classification models).*

---

## 2. How Vision Transformer (ViT) Works

Traditional computer vision architectures rely on Convolutional Neural Networks (CNNs) that process local pixel neighborhoods with convolution kernels. ViT replaces convolutions almost entirely with the standard **Transformer Encoder** architecture from Natural Language Processing (NLP).

### High-Level Workflow Diagram

```
Raw Image (e.g., 224 x 224 x 3)
       │
       ▼
Split into Non-Overlapping Patches (e.g., 16 x 16 pixels each)
       │
       ▼
Flatten & Linear Projection to Dimension D (Patch Embeddings)
       │
       ▼
Prepend Learnable [CLS] Token & Add 1D Positional Embeddings
       │
       ▼
Standard Transformer Encoder (L Blocks)
 ├── Multi-Head Self-Attention (MSA)
 ├── Multi-Layer Perceptron (MLP / Feedforward)
 └── LayerNorm & Residual Connections
       │
       ▼
Extract [CLS] Token Representation
       │
       ├──► Recognition / Face Embedding: L2-Normalized Feature Vector
       └──► Classification Head: Linear Layer -> Class Logits
```

---

## 3. Step-by-Step Breakdown of the Architecture

### Step 1: Patch Extraction & Flattening
Transformers were designed for 1D sequences of words/tokens, but images are 2D arrays ($H \times W \times C$).
- ViT slices an image into a grid of non-overlapping square patches of size $P \times P$ (commonly $16 \times 16$ or $14 \times 14$).
- Number of patches $N$:
  $$N = \frac{H \cdot W}{P^2}$$
  *For a $224 \times 224$ image with $16 \times 16$ patches: $N = (224 / 16) \times (224 / 16) = 14 \times 14 = 196$ patches.*
- Each patch has size $16 \times 16 \times 3 = 768$ values.

### Step 2: Linear Projection (Patch Embedding)
- Each flattened patch of size $(P^2 \cdot C)$ is mapped via a learnable linear projection matrix (or a 2D convolution with `kernel_size=patch_size` and `stride=patch_size`) to a vector of hidden size $D$ (e.g., $D = 768$ for ViT-Base).
- This produces $N$ token vectors of dimension $D$.

### Step 3: Learnable `[CLS]` Token
- Similar to BERT in NLP, a special learnable classification token embedding `[CLS]` of dimension $D$ is prepended at index 0.
- Sequence length becomes $N + 1$ (e.g., $196 + 1 = 197$).
- Because self-attention allows all tokens to interact with each other, this `[CLS]` token aggregates global information from all image patches across the network layers.

### Step 4: Positional Embeddings
- Transformers are permutation-invariant (they have no built-in notion of order or 2D geometry).
- To preserve spatial relationships, learnable 1D position embeddings of dimension $D$ are added to each patch embedding:
  $$\mathbf{z}_0 = [\mathbf{x}_{\text{class}};\, \mathbf{x}_p^1 \mathbf{E};\, \dots;\, \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}$$

### Step 5: Transformer Encoder Stack
The sequence of $N + 1$ vectors passes through $L$ identical Transformer encoder blocks. Each block contains:
1. **Layer Normalization (LN)**
2. **Multi-Head Self-Attention (MSA):** Allows every patch to attend to every other patch across the entire image regardless of distance (capturing global context from layer 1, unlike CNNs which only capture local receptive fields initially).
3. **Residual Connections:** Skip connections around the attention block.
4. **MLP Block:** Two linear layers with GeLU non-linearity in between.
5. **Residual Connections:** Skip connections around the MLP block.

### Step 6: Output / Head
- At the output of the final Transformer block, the hidden state at index 0 corresponding to the `[CLS]` token is extracted.
- **For Classification:** Passed through an MLP/linear head to output logits over $K$ classes (`ViTForImageClassification`).
- **For Face Recognition / Feature Extraction:** The representation is projected and $L_2$-normalized into an embedding vector (e.g., 512-d) to compute cosine similarity against a database (`ViTModel`).

---

## 4. Key Hugging Face Classes for ViT

From the Hugging Face `transformers` library:

1. **`ViTImageProcessor` / `AutoImageProcessor`**:
   Handles resizing (e.g. to $224 \times 224$), center cropping, tensor conversion, and normalization using ImageNet mean & std.
2. **`ViTModel`**:
   The raw Vision Transformer encoder without any classification head. Outputs `last_hidden_state` and `pooler_output`.
   - Ideal for **feature extraction, face recognition embeddings, and similarity search**.
3. **`ViTForImageClassification`**:
   Adds a linear classification head on top of the pooled output for closed-set categorization.
4. **`ViTConfig`**:
   Specifies architecture hyperparameters (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `patch_size`, `image_size`).

---

## 5. ViT in PyTorch / Hugging Face (Code Example)

### A. Feature / Embedding Extraction (Recognition Mode)
```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Load feature extractor and bare ViT backbone
model_name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

image = Image.open("path/to/face_or_object.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # outputs.last_hidden_state shape: [batch_size, 197, 768]
    # Token 0 is the [CLS] representation (768-dimensional feature embedding)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    # Normalize for cosine similarity matching
    norm_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)

print("Embedding shape:", norm_embedding.shape) # [1, 768]
```

### B. Image Classification
```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

image = Image.open("path/to/image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class_idx = logits.argmax(-1).item()

print("Predicted class:", model.config.id2label[predicted_class_idx])
```

---

## 6. Summary Comparison: ViT vs CNNs in Edge Face Recognition

| Metric / Aspect | CNN-based (ArcFace / MobileFaceNet / EdgeFace) | Vision Transformer (ViT) |
| :--- | :--- | :--- |
| **Inductive Bias** | High (translation equivariance, locality built-in). Trains well on smaller datasets. | Low. Learns relationships directly from data; requires large datasets or heavy pre-training. |
| **Receptive Field** | Grows gradually layer by layer. | Global from the very first attention layer. |
| **Inference Speed on Edge/RPi** | Very fast (3–5 ms on GPU, lightweight on CPU/int8 TFLite). | Slower and heavier memory footprint due to quadratic self-attention cost. |
| **Role in Face Pipeline** | Recognizer (extracts face embedding). | Recognizer (extracts face embedding). |
