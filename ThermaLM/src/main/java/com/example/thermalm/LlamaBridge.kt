package com.example.thermalm

class LlamaBridge {
    companion object {
        init {
            System.loadLibrary("thermalm")
        }
    }

    external fun loadModel(path: String, params: ModelParams): Boolean
    external fun generate(prompt: String, params: RuntimeParams): String
    external fun updateRuntime(params: RuntimeParams): Boolean
}
