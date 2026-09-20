# Valkeryne (Qwen3.5-0.8B) 8-Bit Quantized ONNX Model Package

This folder contains the complete 8-bit quantized ONNX export of **Nihilux/Valkeryne** (fine-tuned on **unsloth/Qwen3.5-0.8B** with LoRA weights merged), partitioned for efficient multimodal execution and deployment on mobile/edge devices (Android / iOS / ONNX Runtime Mobile / ExecuTorch).

---

## 1. Directory Structure

```
E:\onnx\valkeryne_onnx_int8\
├── vision_encoder_int8.onnx         # 8-bit quantized vision encoder (~108 MB)
├── text_encoder_embed.onnx          # Text token embedding encoder (~970 MB)
├── main_language_model_int8.onnx    # 8-bit quantized language model + LM head (~761 MB)
├── tokenizer.json                   # Tokenizer vocabulary and BPE merges
├── tokenizer_config.json            # Tokenizer configuration & special tokens
├── processor_config.json            # Multi-modal processor config
├── preprocessor_config.json         # Image preprocessing parameters
├── chat_template.jinja              # Qwen-compatible chat template
└── README.md
```

---

## 2. Model Breakdown & I/O Signatures

### A. Vision Encoder (`vision_encoder_int8.onnx`)
- **Inputs**:
  - `pixel_values` `[num_patches, 1536]` (`float32`): Flattened patch embeddings produced by the image preprocessor.
  - `image_grid_thw` `[num_images, 3]` (`int64`): Temporal, height, width grid dimensions of each image.
- **Outputs**:
  - `image_features` `[num_tokens, 1024]` (`float32`): Merged image token embeddings matching the text hidden dimension.

### B. Text Token Embedding (`text_encoder_embed.onnx`)
- **Inputs**:
  - `input_ids` `[batch_size, sequence_length]` (`int64`): Input token IDs.
- **Outputs**:
  - `inputs_embeds` `[batch_size, sequence_length, 1024]` (`float32`): Initial text token embeddings.

### C. Main Language Model (`main_language_model_int8.onnx`)
- **Inputs**:
  - `inputs_embeds` `[batch_size, sequence_length, 1024]` (`float32`): Combined text and image embeddings. Image placeholder tokens `<|image_pad|>` in the text embedding tensor are replaced with `image_features`.
- **Outputs**:
  - `logits` `[batch_size, sequence_length, 248320]` (`float32`): Next-token vocabulary distribution.

---

## 3. Python Verification Example

```python
import numpy as np
import onnxruntime as ort

# Load sessions
v_sess = ort.InferenceSession("vision_encoder_int8.onnx")
e_sess = ort.InferenceSession("text_encoder_embed.onnx")
m_sess = ort.InferenceSession("main_language_model_int8.onnx")

# 1. Vision feature extraction
dummy_patches = np.random.randn(256, 1536).astype(np.float32)
dummy_grid = np.array([[1, 16, 16]], dtype=np.int64)
image_features = v_sess.run(None, {
    "pixel_values": dummy_patches,
    "image_grid_thw": dummy_grid
})[0]  # Shape: [64, 1024]

# 2. Text embeddings
input_ids = np.array([[151644, 872, 100]], dtype=np.int64)
text_embeds = e_sess.run(None, {"input_ids": input_ids})[0]  # Shape: [1, 3, 1024]

# 3. Main LM forward pass
logits = m_sess.run(None, {"inputs_embeds": text_embeds})[0]  # Shape: [1, 3, 248320]
next_token = np.argmax(logits[:, -1, :], axis=-1)
print("Predicted next token ID:", next_token)
```

---

## 4. Mobile & Phone Deployment Guide (Android / iOS)

### Runtime Options for Mobile:
1. **ONNX Runtime Mobile (ORT Mobile)**:
   - Android: `com.microsoft.onnxruntime:onnxruntime-android` (or `onnxruntime-mobile` build).
   - Supports CPU execution (with ARM NEON acceleration) and NNAPI / QNN execution providers.
2. **Pipeline on Phone**:
   - **Step 1 (Vision)**: When an image is attached, resize and patch it with the preprocessor parameters in `preprocessor_config.json`, then pass through `vision_encoder_int8.onnx`.
   - **Step 2 (Prompt formatting)**: Tokenize text using `tokenizer.json` and insert image placeholder token spans.
   - **Step 3 (Scatter)**: Pass token IDs through `text_encoder_embed.onnx`, substitute placeholder slices with `image_features`.
   - **Step 4 (Generation)**: Run `main_language_model_int8.onnx` iteratively for autoregressive decoding with greedy or sampling search.
