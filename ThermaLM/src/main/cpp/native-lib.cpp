#include <jni.h>
#include <string>
#include <android/log.h>
#include "llama.h"

#define TAG "ThermaLM-Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model* model = nullptr;
static llama_context* ctx = nullptr;

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_thermalm_LlamaBridge_loadModel(JNIEnv *env, jobject thiz, jstring path, jobject params) {
    const char *model_path = env->GetStringUTFChars(path, nullptr);

    jclass params_class = env->GetObjectClass(params);
    jfieldID context_window_id = env->GetFieldID(params_class, "contextWindow", "I");
    jfieldID use_flash_attn_id = env->GetFieldID(params_class, "useFlashAttention", "Z");

    int context_window = env->GetIntField(params, context_window_id);
    bool use_flash_attn = env->GetBooleanField(params, use_flash_attn_id);

    llama_backend_init();

    auto mparams = llama_model_default_params();
    model = llama_load_model_from_file(model_path, mparams);

    if (!model) {
        LOGE("Failed to load model from %s", model_path);
        env->ReleaseStringUTFChars(path, model_path);
        return JNI_FALSE;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = context_window;
    cparams.flash_attn = use_flash_attn;

    ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        LOGE("Failed to create context");
        llama_free_model(model);
        model = nullptr;
        env->ReleaseStringUTFChars(path, model_path);
        return JNI_FALSE;
    }

    LOGI("Model loaded successfully from %s", model_path);
    env->ReleaseStringUTFChars(path, model_path);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_thermalm_LlamaBridge_generate(JNIEnv *env, jobject thiz, jstring prompt, jobject params) {
    if (!ctx) return env->NewStringUTF("Model not loaded");

    const char *prompt_str = env->GetStringUTFChars(prompt, nullptr);

    jclass params_class = env->GetObjectClass(params);
    jfieldID thread_count_id = env->GetFieldID(params_class, "threadCount", "I");
    jfieldID batch_size_id = env->GetFieldID(params_class, "batchSize", "I");

    int thread_count = env->GetIntField(params, thread_count_id);
    int batch_size = env->GetIntField(params, batch_size_id);

    // In a real implementation, we would tokenize the prompt and run inference here.
    // For now, this is a skeleton showing where the params are used.

    LOGI("Generating with threads: %d, batch size: %d", thread_count, batch_size);

    std::string response = "Stub response for: ";
    response += prompt_str;

    env->ReleaseStringUTFChars(prompt, prompt_str);
    return env->NewStringUTF(response.c_str());
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_thermalm_LlamaBridge_updateRuntime(JNIEnv *env, jobject thiz, jobject params) {
    if (!ctx) return JNI_FALSE;

    jclass params_class = env->GetObjectClass(params);
    jfieldID thread_count_id = env->GetFieldID(params_class, "threadCount", "I");
    int thread_count = env->GetIntField(params, thread_count_id);

    // llama.cpp often allows setting thread count in the sampling or eval calls,
    // or by updating the context if the API supports it.
    LOGI("Updating runtime thread count to: %d", thread_count);

    return JNI_TRUE;
}
