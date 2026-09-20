import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3_5ForConditionalGeneration
from peft import PeftModel
import transformers.masking_utils as mu
import transformers.models.qwen3_5.modeling_qwen3_5 as qm
import onnx
from onnx import helper, TensorProto
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

BASE_DIR = r"E:\onnx"
OUTPUT_DIR = os.path.join(BASE_DIR, "valkeryne_onnx_int8")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL_DIR = os.path.join(BASE_DIR, "qwen3.5_base")
LORA_MODEL_DIR = os.path.join(BASE_DIR, "valkeryne_lora", "final_model")

print("=== 1. Setting up Environment and Compatibility Patches ===")
# Set dynamo export mode for unrolling solve_triangular forward substitution
qm.is_torchdynamo_exporting = lambda: True

# Patch causal_conv1d_fn padding for onnx tracing
orig_conv1d_fn = qm.causal_conv1d_fn
def patched_causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    _, hidden_size, seq_len = hidden_states.shape
    pad = int(weight.shape[-1] - 1)
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=pad,
        groups=hidden_size,
    )[:, :, :seq_len]
    if activation is not None:
        out = qm.ACT2FN[activation](out)
    return out.to(hidden_states.dtype)

qm.causal_conv1d_fn = patched_causal_conv1d_fn

# Patch torch.diff with explicit concatenation and slicing
orig_diff = torch.diff
def patched_diff(input, n=1, dim=-1, prepend=None, append=None):
    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)
    if dim == -1 or dim == input.ndim - 1:
        return input[..., 1:] - input[..., :-1]
    elif dim == 0:
        return input[1:] - input[:-1]
    else:
        return orig_diff(input, n=n, dim=dim)

torch.diff = patched_diff

# Patch create_packed_sequence_mask to ensure int64 type for ONNX CumSum
def patched_create_packed_sequence_mask(position_ids):
    first_dummy_value = position_ids[:, :1] - 1
    position_diff = patched_diff(position_ids, prepend=first_dummy_value, dim=-1)
    packed_sequence_mask = (position_diff != 1).to(torch.int64).cumsum(-1)
    return packed_sequence_mask

mu.create_packed_sequence_mask = patched_create_packed_sequence_mask

print("=== 2. Loading Valkeryne Merged Model ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading base weights onto {device}...")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=torch.float32,
    device_map=device
)

print("Merging LoRA weights from Valkeryne final_model...")
model = PeftModel.from_pretrained(model, LORA_MODEL_DIR)
model = model.merge_and_unload()
model.eval()
model = model.to("cpu")
print("Valkeryne model loaded and merged successfully!")

# Copy tokenizers, processor configs, and templates to output folder
print("=== 3. Exporting Tokenizer & Processor Configs ===")
for fname in os.listdir(LORA_MODEL_DIR):
    src = os.path.join(LORA_MODEL_DIR, fname)
    if os.path.isfile(src) and (fname.endswith(".json") or fname.endswith(".jinja")):
        shutil.copy2(src, os.path.join(OUTPUT_DIR, fname))

for fname in os.listdir(BASE_MODEL_DIR):
    src = os.path.join(BASE_MODEL_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, fname)
    if os.path.isfile(src) and (fname.endswith(".json") or fname.endswith(".txt")) and not os.path.exists(dst):
        shutil.copy2(src, dst)

print("Configs and tokenizers copied!")

# === Model 1: Vision Encoder ===
print("=== 4. Exporting & Quantizing Vision Encoder (img encoder) ===")
class VisionEncoderWrapper(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
    def forward(self, pixel_values, image_grid_thw):
        out = self.visual(pixel_values, grid_thw=image_grid_thw)
        return out.pooler_output

vision_mod = VisionEncoderWrapper(model.model.visual)
vision_raw_onnx = os.path.join(OUTPUT_DIR, "vision_encoder.onnx")
dummy_pixel = torch.randn(256, 1536, dtype=torch.float32)
dummy_grid = torch.tensor([[1, 16, 16]], dtype=torch.int64)

torch.onnx.export(
    vision_mod,
    (dummy_pixel, dummy_grid),
    vision_raw_onnx,
    input_names=["pixel_values", "image_grid_thw"],
    output_names=["image_features"],
    dynamic_axes={
        "pixel_values": {0: "num_patches"},
        "image_grid_thw": {0: "num_images"},
        "image_features": {0: "num_tokens"}
    },
    opset_version=17
)
print("Vision ONNX export complete. Quantizing to 8-bit QInt8...")
vision_int8_onnx = os.path.join(OUTPUT_DIR, "vision_encoder_int8.onnx")
quantize_dynamic(
    model_input=vision_raw_onnx,
    model_output=vision_int8_onnx,
    op_types_to_quantize=["MatMul"],
    weight_type=QuantType.QInt8
)
if os.path.exists(vision_raw_onnx):
    os.remove(vision_raw_onnx)
print(f"Vision encoder 8-bit quantized: {vision_int8_onnx} ({os.path.getsize(vision_int8_onnx)/(1024*1024):.2f} MB)")

# === Model 2: Text Embedding Encoder ===
print("=== 5. Exporting Text Embedding Encoder ===")
embed_mod = model.model.language_model.embed_tokens
embed_raw_onnx = os.path.join(OUTPUT_DIR, "text_encoder_embed.onnx")
dummy_ids = torch.tensor([[100, 200, 300]], dtype=torch.int64)

torch.onnx.export(
    embed_mod,
    (dummy_ids,),
    embed_raw_onnx,
    input_names=["input_ids"],
    output_names=["inputs_embeds"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=17
)
print(f"Text embedding ONNX exported: {embed_raw_onnx} ({os.path.getsize(embed_raw_onnx)/(1024*1024):.2f} MB)")

# === Model 3: Main Model (Language Model + LM Head) ===
print("=== 6. Exporting & Quantizing Main Language Model ===")
class MainTextLanguageModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
    def forward(self, inputs_embeds):
        out = self.language_model(inputs_embeds=inputs_embeds, use_cache=False)
        logits = self.lm_head(out.last_hidden_state)
        return logits

main_mod = MainTextLanguageModel(model)
dummy_embeds = torch.randn(1, 16, 1024, dtype=torch.float32)
main_raw_onnx = os.path.join(OUTPUT_DIR, "main_language_model_raw.onnx")

torch.onnx.export(
    main_mod,
    (dummy_embeds,),
    main_raw_onnx,
    input_names=["inputs_embeds"],
    output_names=["logits"],
    dynamic_axes={
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=17
)
print("Fixing ONNX graph for CumSum typing...")
onnx_model = onnx.load(main_raw_onnx, load_external_data=False)
for i, n in enumerate(onnx_model.graph.node):
    if n.name == "/language_model/CumSum":
        cast_node = helper.make_node(
            "Cast",
            inputs=["/language_model/Not_output_0"],
            outputs=["/language_model/Not_output_0_int64"],
            to=TensorProto.INT64,
            name="/language_model/Cast_cumsum_fix"
        )
        n.input[0] = "/language_model/Not_output_0_int64"
        onnx_model.graph.node.insert(i, cast_node)
        break

fixed_onnx = os.path.join(OUTPUT_DIR, "main_language_model.onnx")
onnx.save(onnx_model, fixed_onnx, save_as_external_data=True, all_tensors_to_one_file=True, location="main_language_model.onnx.data")
if os.path.exists(main_raw_onnx):
    os.remove(main_raw_onnx)

print("Quantizing main model MatMul layers to 8-bit QInt8...")
main_int8_onnx = os.path.join(OUTPUT_DIR, "main_language_model_int8.onnx")
quantize_dynamic(
    model_input=fixed_onnx,
    model_output=main_int8_onnx,
    op_types_to_quantize=["MatMul"],
    weight_type=QuantType.QInt8,
    extra_options={"EnableSubgraph": True}
)
# Clean up unquantized intermediate files
if os.path.exists(fixed_onnx):
    os.remove(fixed_onnx)
if os.path.exists(os.path.join(OUTPUT_DIR, "main_language_model.onnx.data")):
    os.remove(os.path.join(OUTPUT_DIR, "main_language_model.onnx.data"))

print(f"Main language model 8-bit quantized: {main_int8_onnx} ({os.path.getsize(main_int8_onnx)/(1024*1024):.2f} MB)")

# === Verification ===
print("=== 7. Verifying End-to-End ONNX Runtime Execution ===")
sess_opts = ort.SessionOptions()
providers = ["CPUExecutionProvider"]

# Verify vision
sess_v = ort.InferenceSession(vision_int8_onnx, sess_opts, providers=providers)
v_in = {"pixel_values": np.random.randn(256, 1536).astype(np.float32), "image_grid_thw": np.array([[1, 16, 16]], dtype=np.int64)}
v_out = sess_v.run(None, v_in)
print(f"[OK] Vision encoder int8 output shape: {v_out[0].shape}")

# Verify embedding
sess_e = ort.InferenceSession(embed_raw_onnx, sess_opts, providers=providers)
e_in = {"input_ids": np.array([[151644, 872, 100]], dtype=np.int64)}
e_out = sess_e.run(None, e_in)
print(f"[OK] Text embed encoder output shape: {e_out[0].shape}")

# Verify main model
sess_m = ort.InferenceSession(main_int8_onnx, sess_opts, providers=providers)
m_in = {"inputs_embeds": np.random.randn(1, 16, 1024).astype(np.float32)}
m_out = sess_m.run(None, m_in)
print(f"[OK] Main language model int8 logits output shape: {m_out[0].shape}")

print("\nAll models exported, 8-bit quantized, and verified successfully!")
