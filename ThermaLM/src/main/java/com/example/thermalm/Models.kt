package com.example.thermalm

enum class QuantizationLevel {
    Q4, Q8, FP16
}

data class InferenceConfig(
    val modelPath: String,
    val threadCount: Int,
    val batchSize: Int,
    val useFlashAttention: Boolean,
    val quantizationLevel: QuantizationLevel,
    val contextWindow: Int
)

data class DeviceState(
    val batteryLevel: Int,
    val isCharging: Boolean,
    val thermalHeadroom: Float
)

data class ModelParams(
    val contextWindow: Int,
    val useFlashAttention: Boolean
)

data class RuntimeParams(
    val threadCount: Int,
    val batchSize: Int
)
