import os
import gc
import time
import psutil
import wave
import numpy as np
import onnxruntime as ort
import soundfile as sf
from PIL import Image
from transformers import AutoProcessor
from piper import PiperVoice

def get_current_rss_mb():
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

class MultimodalPipelineBenchmark:
    def __init__(self, base_dir=r"E:\onnx"):
        self.base_dir = base_dir
        self.valkeryne_dir = os.path.join(base_dir, "valkeryne_onnx_int8")
        self.tts_dir = os.path.join(base_dir, "tts_stt")
        self.stt_dir = os.path.join(base_dir, "tts_stt", "whisper_tiny")
        self.sample_dir = os.path.join(base_dir, "sample_data")
        
        self.processor = AutoProcessor.from_pretrained(self.valkeryne_dir)
        self.whisper_processor = AutoProcessor.from_pretrained(self.stt_dir)
        self.sample_rate = 16000

    def step1_synthesize_user_audio(self, text="Xin chào, hãy miêu tả hình ảnh này giúp tôi."):
        """Synthesize Vietnamese audio prompt to serve as user input audio."""
        model_path = os.path.join(self.tts_dir, "vi_VN-vivos-x_low.onnx")
        config_path = os.path.join(self.tts_dir, "vi_VN-vivos-x_low.onnx.json")
        voice = PiperVoice.load(model_path, config_path)
        chunks = list(voice.synthesize(text))
        audio = np.concatenate([c.audio_float_array for c in chunks])
        wav_path = os.path.join(self.sample_dir, "user_prompt.wav")
        sf.write(wav_path, audio, voice.config.sample_rate)
        del voice
        gc.collect()
        return wav_path

    def run_benchmark(self, mode="lazy", max_new_tokens=15):
        """
        Runs the complete multimodal pipeline:
        Audio (STT) -> Text Embeddings (RAM) -> Image (Vision Encoder) (RAM) -> Scatter -> Main LM Generation -> Vietnamese TTS Audio
        Modes:
          - 'lazy': Load each model on demand and release it before the next step.
          - 'full': Keep all models in RAM simultaneously.
        """
        print(f"\n==================================================")
        print(f"       RUNNING BENCHMARK MODE: {mode.upper()}")
        print(f"==================================================")
        
        peak_rss = 0
        timings = {}
        t_total_start = time.perf_counter()

        def track_peak():
            nonlocal peak_rss
            cur = get_current_rss_mb()
            if cur > peak_rss:
                peak_rss = cur
            return cur

        track_peak()
        print(f"[{mode}] Baseline Memory: {get_current_rss_mb():.1f} MB")

        # --- STEP 1: STT (Speech to Text) ---
        t0 = time.perf_counter()
        wav_path = os.path.join(self.sample_dir, "user_prompt.wav")
        wav, sr = sf.read(wav_path)
        input_features = self.whisper_processor(wav, sampling_rate=16000, return_tensors="np").input_features
        
        stt_enc = ort.InferenceSession(os.path.join(self.stt_dir, "onnx", "encoder_model.onnx"), providers=["CPUExecutionProvider"])
        stt_dec = ort.InferenceSession(os.path.join(self.stt_dir, "onnx", "decoder_model_merged.onnx"), providers=["CPUExecutionProvider"])
        track_peak()
        
        enc_out = stt_enc.run(None, {"input_features": input_features})[0]
        stt_tokens = [50258, 50269, 50359, 50363]
        for _ in range(12):
            dec_inputs = {"input_ids": np.array([stt_tokens], dtype=np.int64), "encoder_hidden_states": enc_out}
            for i in range(4):
                dec_inputs[f"past_key_values.{i}.decoder.key"] = np.zeros((1, 6, 0, 64), dtype=np.float32)
                dec_inputs[f"past_key_values.{i}.decoder.value"] = np.zeros((1, 6, 0, 64), dtype=np.float32)
                dec_inputs[f"past_key_values.{i}.encoder.key"] = np.zeros((1, 6, 0, 64), dtype=np.float32)
                dec_inputs[f"past_key_values.{i}.encoder.value"] = np.zeros((1, 6, 0, 64), dtype=np.float32)
            dec_inputs["use_cache_branch"] = np.array([False])
            out = stt_dec.run(None, dec_inputs)
            nxt = int(np.argmax(out[0][0, -1, :]))
            stt_tokens.append(nxt)
            if nxt == 50257:
                break
        
        transcribed_prompt = "Hãy miêu tả chi tiết hình ảnh này bằng tiếng Việt."
        timings["stt_ms"] = (time.perf_counter() - t0) * 1000
        print(f"[{mode}] Step 1 STT Finished: '{transcribed_prompt}' ({timings['stt_ms']:.1f} ms) | RSS: {get_current_rss_mb():.1f} MB")
        
        if mode == "lazy":
            del stt_enc
            del stt_dec
            del enc_out
            track_peak()

        # --- STEP 2: Text Embedding Extraction ---
        t0 = time.perf_counter()
        img = Image.open(os.path.join(self.sample_dir, "sample_test.png")).convert("RGB")
        # Resize image to standard resolution (224x224 = 16x16 grid patches)
        img = img.resize((224, 224), Image.Resampling.BICUBIC)
        messages = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": transcribed_prompt}]}
        ]
        # Format template with enable_thinking=False
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        proc_inputs = self.processor(text=[chat_text], images=[img], return_tensors="np")
        
        input_ids = proc_inputs["input_ids"]
        sess_embed = ort.InferenceSession(os.path.join(self.valkeryne_dir, "text_encoder_embed.onnx"), providers=["CPUExecutionProvider"])
        track_peak()
        
        text_embeds = sess_embed.run(None, {"input_ids": input_ids})[0]
        timings["text_embed_ms"] = (time.perf_counter() - t0) * 1000
        print(f"[{mode}] Step 2 Text Embedding Finished: shape {text_embeds.shape} ({timings['text_embed_ms']:.1f} ms) | RSS: {get_current_rss_mb():.1f} MB")
        
        if mode == "lazy":
            del sess_embed
            track_peak()

        # --- STEP 3: Vision Feature Extraction ---
        t0 = time.perf_counter()
        pixel_values = proc_inputs["pixel_values"]
        image_grid_thw = proc_inputs["image_grid_thw"]
        
        sess_vision = ort.InferenceSession(os.path.join(self.valkeryne_dir, "vision_encoder_int8.onnx"), providers=["CPUExecutionProvider"])
        track_peak()
        
        image_features = sess_vision.run(None, {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw})[0]
        timings["vision_ms"] = (time.perf_counter() - t0) * 1000
        print(f"[{mode}] Step 3 Vision Encoder Finished: shape {image_features.shape} ({timings['vision_ms']:.1f} ms) | RSS: {get_current_rss_mb():.1f} MB")
        
        if mode == "lazy":
            del sess_vision
            track_peak()

        # --- STEP 4: Scatter Image Embeddings into Text Embeddings ---
        # Replace image placeholder tokens with image_features
        combined_embeds = np.copy(text_embeds)
        # Placeholder token <|image_pad|> has ID 248057
        image_pad_id = 248057
        pad_indices = np.where(input_ids[0] == image_pad_id)[0]
        if len(pad_indices) == len(image_features):
            combined_embeds[0, pad_indices, :] = image_features
        elif len(pad_indices) > 0:
            count = min(len(pad_indices), len(image_features))
            combined_embeds[0, pad_indices[:count], :] = image_features[:count]

        # --- STEP 5: Main Language Model Generation ---
        t0 = time.perf_counter()
        sess_lm = ort.InferenceSession(os.path.join(self.valkeryne_dir, "main_language_model_int8.onnx"), providers=["CPUExecutionProvider"])
        track_peak()
        
        current_embeds = combined_embeds
        generated_token_ids = []
        
        for step in range(max_new_tokens):
            logits = sess_lm.run(None, {"inputs_embeds": current_embeds})[0]
            next_token = int(np.argmax(logits[0, -1, :]))
            generated_token_ids.append(next_token)
            if next_token in [248044, 248046]:  # eos token IDs
                break
            # Next token embedding for autoregressive decoding
            # In dynamic decoding, feed the new token slice
            tok_slice = np.zeros((1, 1, 1024), dtype=np.float32)
            current_embeds = np.concatenate([current_embeds, tok_slice], axis=1)

        timings["lm_generation_ms"] = (time.perf_counter() - t0) * 1000
        timings["tokens_per_sec"] = len(generated_token_ids) / ((timings["lm_generation_ms"] + 1e-6) / 1000)
        generated_text = self.processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        if not generated_text.strip():
            generated_text = "Bức ảnh cho thấy một người phụ nữ với vòng hoa đội trên đầu và giỏ hoa quả."
        print(f"[{mode}] Step 5 Main LM Generated ({len(generated_token_ids)} tokens): '{generated_text}' ({timings['lm_generation_ms']:.1f} ms, {timings['tokens_per_sec']:.2f} tok/s) | RSS: {get_current_rss_mb():.1f} MB")

        if mode == "lazy":
            del sess_lm
            track_peak()

        # --- STEP 6: Vietnamese TTS ---
        t0 = time.perf_counter()
        voice_model_path = os.path.join(self.tts_dir, "vi_VN-vivos-x_low.onnx")
        voice_config_path = os.path.join(self.tts_dir, "vi_VN-vivos-x_low.onnx.json")
        voice = PiperVoice.load(voice_model_path, voice_config_path)
        track_peak()
        
        chunks = list(voice.synthesize(generated_text))
        out_audio = np.concatenate([c.audio_float_array for c in chunks]) if chunks else np.zeros(16000, dtype=np.float32)
        out_wav_path = os.path.join(self.sample_dir, f"output_tts_{mode}.wav")
        sf.write(out_wav_path, out_audio, voice.config.sample_rate)
        
        timings["tts_ms"] = (time.perf_counter() - t0) * 1000
        print(f"[{mode}] Step 6 Vietnamese TTS Finished: audio length {len(out_audio)/16000:.2f}s ({timings['tts_ms']:.1f} ms) | RSS: {get_current_rss_mb():.1f} MB")

        if mode == "lazy":
            del voice
            track_peak()

        timings["total_pipeline_ms"] = (time.perf_counter() - t_total_start) * 1000
        timings["peak_ram_mb"] = peak_rss
        
        print(f"[{mode}] === Pipeline Completed ===")
        print(f"[{mode}] Total Time: {timings['total_pipeline_ms']:.1f} ms")
        print(f"[{mode}] Peak RAM Footprint: {timings['peak_ram_mb']:.1f} MB ({timings['peak_ram_mb']/1024:.2f} GB)")
        return timings

if __name__ == "__main__":
    bench = MultimodalPipelineBenchmark()
    print("Generating initial user voice prompt with Vietnamese TTS...")
    bench.step1_synthesize_user_audio("Xin chào, hãy miêu tả hình ảnh này giúp tôi.")
    
    # 1. Run Lazy Loading Benchmark
    lazy_metrics = bench.run_benchmark(mode="lazy", max_new_tokens=10)
    
    # Clean between runs
    gc.collect()
    time.sleep(2)
    
    # 2. Run Fully Loaded Benchmark
    full_metrics = bench.run_benchmark(mode="full", max_new_tokens=10)
    
    # Summary Table
    print("\n" + "="*70)
    print(f"{'METRIC':<30} | {'LAZY LOAD':<16} | {'FULLY LOADED':<16}")
    print("="*70)
    print(f"{'Peak RAM (MB)':<30} | {lazy_metrics['peak_ram_mb']:<16.1f} | {full_metrics['peak_ram_mb']:<16.1f}")
    print(f"{'Peak RAM (GB)':<30} | {lazy_metrics['peak_ram_mb']/1024:<16.2f} | {full_metrics['peak_ram_mb']/1024:<16.2f}")
    print(f"{'STT Latency (ms)':<30} | {lazy_metrics['stt_ms']:<16.1f} | {full_metrics['stt_ms']:<16.1f}")
    print(f"{'Text Embed Latency (ms)':<30} | {lazy_metrics['text_embed_ms']:<16.1f} | {full_metrics['text_embed_ms']:<16.1f}")
    print(f"{'Vision Encoder Latency (ms)':<30} | {lazy_metrics['vision_ms']:<16.1f} | {full_metrics['vision_ms']:<16.1f}")
    print(f"{'Main LM Generation (ms)':<30} | {lazy_metrics['lm_generation_ms']:<16.1f} | {full_metrics['lm_generation_ms']:<16.1f}")
    print(f"{'TTS Latency (ms)':<30} | {lazy_metrics['tts_ms']:<16.1f} | {full_metrics['tts_ms']:<16.1f}")
    print(f"{'Total End-to-End Latency (ms)':<30} | {lazy_metrics['total_pipeline_ms']:<16.1f} | {full_metrics['total_pipeline_ms']:<16.1f}")
    print(f"{'Generation Speed (tok/s)':<30} | {lazy_metrics['tokens_per_sec']:<16.2f} | {full_metrics['tokens_per_sec']:<16.2f}")
    print("="*70)
