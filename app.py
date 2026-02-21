import os

# HF_HOME must be set before any HuggingFace/transformers imports
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
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

VOICES_DIR = Path(__file__).parent / "voices"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

_models: dict = {}


# ---------------------------------------------------------------------------
# Registered voice helpers
# ---------------------------------------------------------------------------

def _list_voices() -> list[str]:
    VOICES_DIR.mkdir(exist_ok=True)
    return sorted(d.name for d in VOICES_DIR.iterdir() if d.is_dir())


def save_voice(name: str, ref_audio, ref_text: str, language: str):
    name = name.strip()
    if not name:
        raise gr.Error("登録名を入力してください。")
    if ref_audio is None:
        raise gr.Error("参照音声をアップロードしてください。")
    voice_dir = VOICES_DIR / name
    voice_dir.mkdir(parents=True, exist_ok=True)
    ref_sr, ref_data = ref_audio
    sf.write(str(voice_dir / "ref.wav"), ref_data, ref_sr)
    (voice_dir / "info.json").write_text(
        json.dumps({"transcript": ref_text, "language": language}, ensure_ascii=False),
        encoding="utf-8",
    )
    return gr.Dropdown(choices=_list_voices(), value=name)


def load_voice(name: str):
    if not name:
        raise gr.Error("声を選択してください。")
    voice_dir = VOICES_DIR / name
    data, sr = sf.read(str(voice_dir / "ref.wav"), always_2d=False)
    info = json.loads((voice_dir / "info.json").read_text(encoding="utf-8"))
    return (sr, data), info["transcript"], info.get("language", "Auto")


# ---------------------------------------------------------------------------
# MP3 export
# ---------------------------------------------------------------------------

def _to_mp3(data: np.ndarray, sr: int) -> str:
    import lameenc
    # Convert float32 [-1, 1] → int16
    if data.dtype.kind == "f":
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)
    # Downmix to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(sr)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 2 = highest quality
    mp3_bytes = encoder.encode(data.tobytes()) + encoder.flush()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", dir=OUTPUTS_DIR, delete=False)
    tmp.write(mp3_bytes)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

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
    return _to_mp3(wavs[0], sr), f"{elapsed:.1f} 秒"


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
    return _to_mp3(wavs[0], sr), f"{elapsed:.1f} 秒"


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
    return _to_mp3(wavs[0], sr), f"{elapsed:.1f} 秒"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

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
                    cv_audio = gr.Audio(label="出力音声")
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
                    vd_audio = gr.Audio(label="出力音声")
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

                    # -- Registered voice loader --
                    with gr.Group():
                        gr.Markdown("#### 登録済みの声")
                        with gr.Row():
                            vc_voice_dd = gr.Dropdown(
                                choices=_list_voices(),
                                label="声を選択",
                                scale=3,
                            )
                            vc_load_btn = gr.Button("読み込む", scale=1)

                    # -- Inputs --
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

                    # -- Voice registration --
                    with gr.Group():
                        gr.Markdown("#### この参照音声を登録")
                        with gr.Row():
                            vc_voice_name = gr.Textbox(
                                label="登録名",
                                placeholder="例: MyVoice",
                                scale=3,
                            )
                            vc_save_btn = gr.Button("登録", scale=1)

                with gr.Column():
                    vc_audio = gr.Audio(label="出力音声")
                    vc_time = gr.Textbox(label="生成時間", interactive=False)

            vc_btn.click(
                fn=generate_clone,
                inputs=[vc_text, vc_language, vc_ref_audio, vc_ref_text],
                outputs=[vc_audio, vc_time],
            )
            vc_load_btn.click(
                fn=load_voice,
                inputs=vc_voice_dd,
                outputs=[vc_ref_audio, vc_ref_text, vc_language],
            )
            vc_save_btn.click(
                fn=save_voice,
                inputs=[vc_voice_name, vc_ref_audio, vc_ref_text, vc_language],
                outputs=vc_voice_dd,
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                allowed_paths=[str(OUTPUTS_DIR)])
