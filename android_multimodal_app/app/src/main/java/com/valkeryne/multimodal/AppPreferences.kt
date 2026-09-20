package com.valkeryne.multimodal

import android.content.Context
import android.content.SharedPreferences

object AppPreferences {
    private const val PREFS_NAME = "byvv_gemini_prefs"
    private const val KEY_API_KEY = "gemini_api_key"
    private const val KEY_MODEL = "gemini_model"
    private const val KEY_THINKING_BUDGET = "gemini_thinking_budget"

    // Gemini Live & Multimodal Models
    const val MODEL_GEMINI_3_8_LIVE = "gemini-3.8-live"
    const val MODEL_GEMINI_3_8_LIVE_EXTENDED = "gemini-3.8-live-extended-thinking"
    const val MODEL_GEMINI_3_1_LIVE_PREVIEW = "gemini-3.1-live-preview"
    const val MODEL_GEMINI_2_5_FLASH_NATIVE_AUDIO = "gemini-2.5-flash-native-audio-latest"
    const val MODEL_GEMINI_3_5_TRANSCRIBE_LIVE = "gemini-3.5-transcribe-live"
    const val MODEL_GEMINI_3_8_FLASH = "gemini-3.8-flash"
    const val MODEL_GEMINI_3_5_FLASH = "gemini-3.5-flash"
    const val MODEL_CUSTOM = "Tự nhập model khác..."

    val AVAILABLE_MODELS = arrayOf(
        MODEL_GEMINI_3_8_LIVE,
        MODEL_GEMINI_3_8_LIVE_EXTENDED,
        MODEL_GEMINI_3_1_LIVE_PREVIEW,
        MODEL_GEMINI_2_5_FLASH_NATIVE_AUDIO,
        MODEL_GEMINI_3_5_TRANSCRIBE_LIVE,
        MODEL_GEMINI_3_8_FLASH,
        MODEL_GEMINI_3_5_FLASH,
        MODEL_CUSTOM
    )

    val THINKING_EFFORT_OPTIONS = arrayOf("Off (0)", "Low (1024)", "Medium (4096)", "High (8192)")

    private fun getPrefs(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    fun getApiKey(context: Context): String {
        return getPrefs(context).getString(KEY_API_KEY, "") ?: ""
    }

    fun setApiKey(context: Context, key: String) {
        getPrefs(context).edit().putString(KEY_API_KEY, key.trim()).apply()
    }

    fun getModel(context: Context): String {
        return getPrefs(context).getString(KEY_MODEL, MODEL_GEMINI_3_5_FLASH) ?: MODEL_GEMINI_3_5_FLASH
    }

    fun setModel(context: Context, model: String) {
        getPrefs(context).edit().putString(KEY_MODEL, model).apply()
    }

    fun getThinkingBudget(context: Context): Int {
        return getPrefs(context).getInt(KEY_THINKING_BUDGET, 0)
    }

    fun setThinkingBudget(context: Context, budget: Int) {
        getPrefs(context).edit().putInt(KEY_THINKING_BUDGET, budget).apply()
    }
}
