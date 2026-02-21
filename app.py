import os

# HF_HOME must be set before any HuggingFace/transformers imports
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

import time
import numpy as np
import torch
import gradio as gr
from qwen_tts import Qwen3TTSModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MODEL_IDS = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "voice_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]

LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto",
]

_models: dict = {}


def get_model(key: str) -> Qwen3TTSModel:
    if key not in _models:
        model_id = MODEL_IDS[key]
        print(f"[voice-echo] Loading {model_id} ...")
        load_kwargs: dict = {"device_map": DEVICE, "dtype": DTYPE}
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("[voice-echo] flash_attention_2 enabled")
        except ImportError:
            pass
        _models[key] = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
        print(f"[voice-echo] Loaded: {model_id}")
    return _models[key]


def generate_custom(text: str, language: str, speaker: str, instruct: str):
    if not text.strip():
        raise gr.Error("テキストを入力してください。")
    model = get_model("custom_voice")
    t0 = time.perf_counter()
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct,
    )
    elapsed = time.perf_counter() - t0
    return (sr, wavs[0]), f"{elapsed:.1f} 秒"


def generate_design(text: str, language: str, instruct: str):
    if not text.strip():
        raise gr.Error("テキストを入力してください。")
    if not instruct.strip():
        raise gr.Error("音声の説明を入力してください。")
    model = get_model("voice_design")
    t0 = time.perf_counter()
    wavs, sr = model.generate_voice_design(
        text=text,
        language=language,
        instruct=instruct,
    )
    elapsed = time.perf_counter() - t0
    return (sr, wavs[0]), f"{elapsed:.1f} 秒"


def generate_clone(text: str, language: str, ref_audio, ref_text: str):
    if not text.strip():
        raise gr.Error("テキストを入力してください。")
    if ref_audio is None:
        raise gr.Error("参照音声をアップロードしてください。")
    if not ref_text.strip():
        raise gr.Error("参照音声のトランスクリプトを入力してください。")
    model = get_model("voice_clone")
    # Gradio delivers (sample_rate, numpy_array); qwen_tts expects (numpy_array, sample_rate)
    ref_sr, ref_data = ref_audio
    # Normalize to float32 [-1, 1]; Gradio may return raw PCM values (e.g. int16 range as float)
    ref_data = ref_data.astype(np.float32)
    max_val = np.abs(ref_data).max()
    if max_val > 1.0:
        ref_data = ref_data / max_val
    t0 = time.perf_counter()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=(ref_data, ref_sr),
        ref_text=ref_text,
    )
    elapsed = time.perf_counter() - t0
    return (sr, wavs[0]), f"{elapsed:.1f} 秒"


with gr.Blocks(title="Qwen3-TTS Demo") as demo:
    gr.Markdown("# Qwen3-TTS Demo")
    gr.Markdown(
        "3つのモード: "
        "**Custom Voice**（プリセット話者）/ "
        "**Voice Design**（声をテキストで設計）/ "
        "**Voice Clone**（参照音声からクローン）"
    )

    with gr.Tabs():

        # --- Tab 1: Custom Voice ---
        with gr.Tab("Custom Voice"):
            gr.Markdown("プリセット話者を選び、スタイル指示を与えて音声を生成します。")
            with gr.Row():
                with gr.Column():
                    cv_text = gr.Textbox(
                        label="テキスト",
                        lines=4,
                        placeholder="読み上げるテキストを入力...",
                    )
                    cv_language = gr.Dropdown(
                        choices=LANGUAGES, value="English", label="言語"
                    )
                    cv_speaker = gr.Dropdown(
                        choices=SPEAKERS, value="Ryan", label="話者"
                    )
                    cv_instruct = gr.Textbox(
                        label="スタイル指示（任意）",
                        placeholder='例: "Speak slowly and warmly." または空白のまま',
                    )
                    cv_btn = gr.Button("生成", variant="primary")
                with gr.Column():
                    cv_audio = gr.Audio(label="出力音声", type="numpy")
                    cv_time = gr.Textbox(label="生成時間", interactive=False)

            cv_btn.click(
                fn=generate_custom,
                inputs=[cv_text, cv_language, cv_speaker, cv_instruct],
                outputs=[cv_audio, cv_time],
            )

        # --- Tab 2: Voice Design ---
        with gr.Tab("Voice Design"):
            gr.Markdown("自然言語で声の特徴を記述して音声を生成します。")
            with gr.Row():
                with gr.Column():
                    vd_text = gr.Textbox(
                        label="テキスト",
                        lines=4,
                        placeholder="読み上げるテキストを入力...",
                    )
                    vd_language = gr.Dropdown(
                        choices=LANGUAGES, value="English", label="言語"
                    )
                    vd_instruct = gr.Textbox(
                        label="声の説明",
                        lines=3,
                        placeholder='例: "Warm male voice, deep tone, slightly husky"',
                    )
                    vd_btn = gr.Button("生成", variant="primary")
                with gr.Column():
                    vd_audio = gr.Audio(label="出力音声", type="numpy")
                    vd_time = gr.Textbox(label="生成時間", interactive=False)

            vd_btn.click(
                fn=generate_design,
                inputs=[vd_text, vd_language, vd_instruct],
                outputs=[vd_audio, vd_time],
            )

        # --- Tab 3: Voice Clone ---
        with gr.Tab("Voice Clone"):
            gr.Markdown(
                "参照音声（3秒以上推奨）とそのトランスクリプトをアップロードして、"
                "その声で別のテキストを読み上げます。"
            )
            with gr.Row():
                with gr.Column():
                    vc_text = gr.Textbox(
                        label="読み上げるテキスト",
                        lines=4,
                        placeholder="クローンした声で読み上げるテキストを入力...",
                    )
                    vc_language = gr.Dropdown(
                        choices=LANGUAGES, value="English", label="言語"
                    )
                    vc_ref_audio = gr.Audio(
                        label="参照音声",
                        type="numpy",
                        sources=["upload", "microphone"],
                    )
                    vc_ref_text = gr.Textbox(
                        label="参照音声のトランスクリプト",
                        lines=2,
                        placeholder="参照音声で話されている内容を正確に入力...",
                    )
                    vc_btn = gr.Button("生成", variant="primary")
                with gr.Column():
                    vc_audio = gr.Audio(label="出力音声", type="numpy")
                    vc_time = gr.Textbox(label="生成時間", interactive=False)

            vc_btn.click(
                fn=generate_clone,
                inputs=[vc_text, vc_language, vc_ref_audio, vc_ref_text],
                outputs=[vc_audio, vc_time],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
