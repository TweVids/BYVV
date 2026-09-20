package com.valkeryne.multimodal

import android.app.Dialog
import android.content.Context
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.view.Window
import android.widget.*

class SettingsDialog(context: Context, private val onSaved: () -> Unit) : Dialog(context) {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)

        val layout = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 40, 40, 40)
            setBackgroundColor(0xFF1E1E1E.toInt())
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        // Title
        val title = TextView(context).apply {
            text = "Cài Đặt Gemini Live (Settings)"
            textSize = 20f
            setTextColor(0xFFFFFFFF.toInt())
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setPadding(0, 0, 0, 24)
        }
        layout.addView(title)

        // API Key label
        val apiKeyLabel = TextView(context).apply {
            text = "Gemini API Key:"
            textSize = 14f
            setTextColor(0xFFBBBBBB.toInt())
        }
        layout.addView(apiKeyLabel)

        // API Key input
        val apiKeyInput = EditText(context).apply {
            hint = "Dán Gemini API Key vào đây"
            setHintTextColor(0xFF777777.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            setText(AppPreferences.getApiKey(context))
            setSingleLine(true)
            setBackgroundColor(0xFF2C2C2C.toInt())
            setPadding(20, 20, 20, 20)
        }
        layout.addView(apiKeyInput)

        // Model Selector Label
        val modelLabel = TextView(context).apply {
            text = "Chọn Mô Hình (Gemini Live Models):"
            textSize = 14f
            setTextColor(0xFFBBBBBB.toInt())
            setPadding(0, 24, 0, 8)
        }
        layout.addView(modelLabel)

        // Custom Model Input field (hidden unless selected)
        val customModelInput = EditText(context).apply {
            hint = "Nhập tên model (vd: gemini-3.8-live)"
            setHintTextColor(0xFF777777.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            visibility = View.GONE
            setBackgroundColor(0xFF2C2C2C.toInt())
            setPadding(20, 20, 20, 20)
        }

        // Model Spinner
        val modelSpinner = Spinner(context).apply {
            val adapter = ArrayAdapter(
                context,
                android.R.layout.simple_spinner_dropdown_item,
                AppPreferences.AVAILABLE_MODELS
            )
            this.adapter = adapter
            val currentModel = AppPreferences.getModel(context)
            val index = AppPreferences.AVAILABLE_MODELS.indexOf(currentModel)
            if (index >= 0) {
                setSelection(index)
            } else {
                setSelection(AppPreferences.AVAILABLE_MODELS.lastIndex)
                customModelInput.setText(currentModel)
                customModelInput.visibility = View.VISIBLE
            }

            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    if (position == AppPreferences.AVAILABLE_MODELS.lastIndex) {
                        customModelInput.visibility = View.VISIBLE
                    } else {
                        customModelInput.visibility = View.GONE
                    }
                }
                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
        }
        layout.addView(modelSpinner)
        layout.addView(customModelInput)

        // Thinking Effort Label
        val thinkingLabel = TextView(context).apply {
            text = "Mức độ suy nghĩ (Thinking Effort / Budget):"
            textSize = 14f
            setTextColor(0xFFBBBBBB.toInt())
            setPadding(0, 24, 0, 8)
        }
        layout.addView(thinkingLabel)

        // Thinking Spinner
        val thinkingSpinner = Spinner(context).apply {
            val adapter = ArrayAdapter(
                context,
                android.R.layout.simple_spinner_dropdown_item,
                AppPreferences.THINKING_EFFORT_OPTIONS
            )
            this.adapter = adapter
            val currentBudget = AppPreferences.getThinkingBudget(context)
            val index = when (currentBudget) {
                1024 -> 1
                4096 -> 2
                8192 -> 3
                else -> 0
            }
            setSelection(index)
        }
        layout.addView(thinkingSpinner)

        // Buttons row
        val btnRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, 32, 0, 0)
        }

        val cancelBtn = Button(context).apply {
            text = "HỦY"
            setBackgroundColor(0xFF444444.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginEnd = 16
            }
            setOnClickListener { dismiss() }
        }

        val saveBtn = Button(context).apply {
            text = "LƯU (SAVE)"
            setBackgroundColor(0xFFFFD600.toInt())
            setTextColor(0xFF000000.toInt())
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener {
                val key = apiKeyInput.text.toString().trim()
                val selectedModel = if (modelSpinner.selectedItemPosition == AppPreferences.AVAILABLE_MODELS.lastIndex) {
                    val custom = customModelInput.text.toString().trim()
                    if (custom.isNotEmpty()) custom else AppPreferences.MODEL_GEMINI_3_8_LIVE
                } else {
                    modelSpinner.selectedItem.toString()
                }

                val budget = when (thinkingSpinner.selectedItemPosition) {
                    1 -> 1024
                    2 -> 4096
                    3 -> 8192
                    else -> 0
                }

                AppPreferences.setApiKey(context, key)
                AppPreferences.setModel(context, selectedModel)
                AppPreferences.setThinkingBudget(context, budget)

                Toast.makeText(context, "Đã lưu cài đặt!", Toast.LENGTH_SHORT).show()
                onSaved()
                dismiss()
            }
        }

        btnRow.addView(cancelBtn)
        btnRow.addView(saveBtn)
        layout.addView(btnRow)

        setContentView(layout)
        window?.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
    }
}
