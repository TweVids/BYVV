# Valkeryne Multimodal Android App

A native Android application running the complete on-device multimodal pipeline powered by **ONNX Runtime Android**:
1. **Speech-to-Text (STT)**: Whisper Tiny ONNX
2. **Text Embedding**: Valkeryne token embedder
3. **Vision Encoder**: Valkeryne 8-bit quantized image patch encoder
4. **Main Language Model**: Valkeryne (Qwen3.5-0.8B) 8-bit quantized Gated DeltaNet + Attention LM
5. **Text-to-Speech (TTS)**: Piper VITS Vietnamese TTS

---

## 1. Automated Build with GitHub Actions
A preconfigured workflow is included at [`.github/workflows/android_build.yml`](file:///E:/onnx/.github/workflows/android_build.yml).
Pushing this directory to GitHub will automatically trigger the build and output `valkeryne-multimodal-debug.apk` in the Actions artifacts tab.

---

## 2. Local Build Instructions

```bash
cd android_multimodal_app
gradle assembleDebug
```
The output APK will be generated at:
`android_multimodal_app/app/build/outputs/apk/debug/app-debug.apk`

---

## 3. Pushing ONNX Models to Phone

Copy the exported ONNX model files onto the device's external app storage:

```bash
adb push E:\onnx\valkeryne_onnx_int8\*.onnx /sdcard/Android/data/com.valkeryne.multimodal/files/
adb push E:\onnx\tts_stt\vi_VN-vivos-x_low.onnx /sdcard/Android/data/com.valkeryne.multimodal/files/
adb push E:\onnx\tts_stt\whisper_tiny\onnx\encoder_model.onnx /sdcard/Android/data/com.valkeryne.multimodal/files/
```
