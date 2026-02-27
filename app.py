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

MODEL_VARIANTS = {
    "1.7B": {
        "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "voice_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    },
    "0.6B": {
        "custom_voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "voice_design": "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
        "voice_clone": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    },
}

ASR_MODELS = {
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
}

TRANSLATE_MODELS = {
    "1.7B": "Qwen/Qwen3-1.7B",
    "0.6B": "Qwen/Qwen3-0.6B",
}

TRANSLATE_STYLE_PRESETS = {
    "標準": "",
    "フォーマル": "Use formal, professional language.",
    "カジュアル": "Use casual, conversational language.",
    "丁寧・敬語": "Use polite and respectful language.",
    "シンプル": "Use simple, easy-to-understand language.",
    "怒り": "Translate with angry, frustrated tone. Use strong, assertive words.",
    "喜び": "Translate with joyful, enthusiastic tone. Use upbeat, energetic language.",
    "悲しみ": "Translate with sad, melancholic tone.",
    "武士": "Translate in the style of a noble samurai warrior.",
}

# Map ASR-detected language names to TTS language names
ASR_LANGUAGE_MAP = {
    "Chinese": "Chinese",
    "English": "English",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "German": "German",
    "French": "French",
    "Russian": "Russian",
    "Portuguese": "Portuguese",
    "Spanish": "Spanish",
    "Italian": "Italian",
}

SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]

LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian", "Auto",
]

VOICES_DIR = Path(__file__).parent / "voices"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
SETTINGS_PATH = Path(__file__).parent / "settings.json"
OUTPUTS_DIR.mkdir(exist_ok=True)

_models: dict = {}


# ---------------------------------------------------------------------------
# Registered voice helpers
# ---------------------------------------------------------------------------

def _load_settings() -> dict:
    try:
        if SETTINGS_PATH.exists():
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_settings(data: dict):
    existing = _load_settings()
    existing.update(data)
    SETTINGS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_model_size_setting() -> str:
    size = _load_settings().get("model_size", "1.7B")
    return size if size in MODEL_VARIANTS else "1.7B"


def _save_model_size_setting(size: str):
    if size not in MODEL_VARIANTS:
        return
    _save_settings({"model_size": size})


def _load_asr_model_size_setting() -> str:
    size = _load_settings().get("asr_model_size", "0.6B")
    return size if size in ASR_MODELS else "0.6B"


def _save_asr_model_size_setting(size: str):
    if size not in ASR_MODELS:
        return
    _save_settings({"asr_model_size": size})


def _load_translate_model_size_setting() -> str:
    size = _load_settings().get("translate_model_size", "0.6B")
    return size if size in TRANSLATE_MODELS else "0.6B"


def _save_translate_model_size_setting(size: str):
    if size not in TRANSLATE_MODELS:
        return
    _save_settings({"translate_model_size": size})


def update_model_size(size: str) -> str:
    size = size if size in MODEL_VARIANTS else "1.7B"
    _save_model_size_setting(size)
    return render_model_info(size)


def update_asr_model_size(size: str) -> str:
    size = size if size in ASR_MODELS else "0.6B"
    _save_asr_model_size_setting(size)
    return f"`{ASR_MODELS[size]}`"


def update_translate_model_size(size: str) -> str:
    size = size if size in TRANSLATE_MODELS else "0.6B"
    _save_translate_model_size_setting(size)
    return f"`{TRANSLATE_MODELS[size]}`"


def fill_style_preset(preset: str) -> str:
    return TRANSLATE_STYLE_PRESETS.get(preset, "")


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


def delete_voice(name: str):
    if not name:
        raise gr.Error("削除する声を選択してください。")
    import shutil
    voice_dir = VOICES_DIR / name
    if voice_dir.exists():
        shutil.rmtree(str(voice_dir))
    voices = _list_voices()
    return gr.Dropdown(choices=voices, value=voices[0] if voices else None)


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

def get_asr_model(size: str):
    size = size if size in ASR_MODELS else "0.6B"
    cache_key = f"asr:{size}"
    if cache_key not in _models:
        from qwen_asr import Qwen3ASRModel
        model_id = ASR_MODELS[size]
        print(f"[voice-echo] Loading {model_id} ...")
        _models[cache_key] = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=DTYPE,
            device_map=DEVICE,
        )
        print(f"[voice-echo] Loaded: {model_id}")
    return _models[cache_key]


def get_model(mode: str, size: str) -> Qwen3TTSModel:
    size = size if size in MODEL_VARIANTS else "1.7B"
    model_id = MODEL_VARIANTS[size][mode]
    cache_key = f"{size}:{mode}"
    if cache_key not in _models:
        print(f"[voice-echo] Loading {model_id} ...")
        load_kwargs: dict = {"device_map": DEVICE, "dtype": DTYPE}
        try:
            import flash_attn  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("[voice-echo] flash_attention_2 enabled")
        except ImportError:
            pass
        _models[cache_key] = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)
        print(f"[voice-echo] Loaded: {model_id}")
    return _models[cache_key]


def render_model_info(size: str) -> str:
    size = size if size in MODEL_VARIANTS else "1.7B"
    ids = MODEL_VARIANTS[size]
    return (
        f"- Custom Voice: `{ids['custom_voice']}`\n"
        f"- Voice Design: `{ids['voice_design']}`\n"
        f"- Voice Clone: `{ids['voice_clone']}`"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_custom(text: str, language: str, speaker: str, instruct: str, model_size: str):
    if not text.strip():
        raise gr.Error("テキストを入力してください。")
    model = get_model("custom_voice", model_size)
    t0 = time.perf_counter()
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct,
    )
    elapsed = time.perf_counter() - t0
    return _to_mp3(wavs[0], sr), f"{elapsed:.1f} 秒"


def generate_design(text: str, language: str, instruct: str, model_size: str):
    if not text.strip():
        raise gr.Error("テキストを入力してください。")
    if not instruct.strip():
        raise gr.Error("音声の説明を入力してください。")
    model = get_model("voice_design", model_size)
    t0 = time.perf_counter()
    wavs, sr = model.generate_voice_design(
        text=text,
        language=language,
        instruct=instruct,
    )
    elapsed = time.perf_counter() - t0
    return _to_mp3(wavs[0], sr), f"{elapsed:.1f} 秒"


def generate_clone(text: str, language: str, ref_audio, ref_text: str, model_size: str):
    if not text.strip():
        raise gr.Error("テキストを入力してください。")
    if ref_audio is None:
        raise gr.Error("参照音声をアップロードしてください。")
    if not ref_text.strip():
        raise gr.Error("参照音声のトランスクリプトを入力してください。")
    model = get_model("voice_clone", model_size)
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


def transcribe_ref(ref_audio, asr_size: str):
    if ref_audio is None:
        raise gr.Error("参照音声をアップロードしてください。")
    model = get_asr_model(asr_size)
    ref_sr, ref_data = ref_audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, ref_data, ref_sr)
        results = model.transcribe(audio=tmp_path, language=None)
    finally:
        os.unlink(tmp_path)
    result = results[0]
    text = result.text
    detected = getattr(result, "language", None) or ""
    language = ASR_LANGUAGE_MAP.get(detected, "Auto")
    return text, language


def get_translate_model(size: str):
    size = size if size in TRANSLATE_MODELS else "0.6B"
    cache_key = f"translate:{size}"
    if cache_key not in _models:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = TRANSLATE_MODELS[size]
        print(f"[voice-echo] Loading {model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
        _models[cache_key] = (model, tokenizer)
        print(f"[voice-echo] Loaded: {model_id}")
    return _models[cache_key]


def translate_text(text: str, target_lang: str, translate_size: str, style_instruct: str = ""):
    if not text.strip():
        raise gr.Error("翻訳するテキストがありません。参照音声のトランスクリプトを入力してください。")
    model, tokenizer = get_translate_model(translate_size)
    style_part = f" {style_instruct.strip()}" if style_instruct.strip() else ""
    prompt = (
        f"Translate the following text to {target_lang} in natural spoken language suitable for text-to-speech.{style_part} "
        "Output only the translated text, nothing else.\n\n"
        f"{text}"
    )
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
        )
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]
    translated = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return translated, target_lang


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Qwen3-TTS Demo") as demo:
    default_model_size = _load_model_size_setting()
    default_asr_model_size = _load_asr_model_size_setting()
    default_translate_model_size = _load_translate_model_size_setting()
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

        # --- Tab 3: Voice Clone ---
        with gr.Tab("Voice Clone"):
            gr.Markdown(
                "参照音声（3秒以上推奨）とそのトランスクリプトをアップロードして、"
                "その声で別のテキストを読み上げます。"
            )
            with gr.Row():
                with gr.Column():
                    # -- Inputs --
                    vc_ref_audio = gr.Audio(
                        label="参照音声",
                        type="numpy",
                        sources=["upload", "microphone"],
                    )
                    vc_transcribe_btn = gr.Button("📝 自動文字起こし (ASR)")
                    vc_ref_text = gr.Textbox(
                        label="参照音声のトランスクリプト",
                        lines=2,
                        placeholder="参照音声で話されている内容を正確に入力...",
                    )

                    # -- Speaker management --
                    with gr.Accordion("話者", open=False):
                        # -- Registered voice loader --
                        with gr.Group():
                            gr.Markdown("#### 登録済みの声")
                            with gr.Row():
                                vc_voice_dd = gr.Dropdown(
                                    choices=_list_voices(),
                                    label="声を選択",
                                    scale=3,
                                )
                                vc_load_btn = gr.Button("読み込み", scale=1)
                                vc_delete_btn = gr.Button("🗑️ 削除", scale=1, variant="stop")

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
                    with gr.Row():
                        vc_translate_lang = gr.Dropdown(
                            choices=[l for l in LANGUAGES if l != "Auto"],
                            value="English",
                            label="翻訳先言語",
                            scale=2,
                        )
                        vc_style_preset = gr.Dropdown(
                            choices=list(TRANSLATE_STYLE_PRESETS.keys()),
                            value="標準",
                            label="スタイル",
                            scale=2,
                        )
                    vc_style_instruct = gr.Textbox(
                        label="スタイル指示（任意・自由入力可）",
                        placeholder='例: "Speak like a pirate" / "子供向けに易しく"',
                    )
                    vc_translate_btn = gr.Button("🌐 翻訳")
                    vc_language = gr.Dropdown(
                        choices=LANGUAGES, value="English", label="言語"
                    )
                    vc_text = gr.Textbox(
                        label="読み上げるテキスト",
                        lines=4,
                        placeholder="クローンした声で読み上げるテキストを入力...",
                    )
                    vc_btn = gr.Button("生成", variant="primary")

        with gr.Tab("Model管理"):
            gr.Markdown("生成に使うモデルサイズを選択します。")
            gr.Markdown("### TTS モデル")
            model_size = gr.Dropdown(
                choices=list(MODEL_VARIANTS.keys()),
                value=default_model_size,
                label="TTS モデルサイズ",
            )
            model_info = gr.Markdown(render_model_info(default_model_size))
            model_size.change(fn=update_model_size, inputs=model_size, outputs=model_info)
            gr.Markdown("### ASR モデル（自動文字起こし）")
            asr_model_size = gr.Dropdown(
                choices=list(ASR_MODELS.keys()),
                value=default_asr_model_size,
                label="ASR モデルサイズ",
            )
            asr_model_info = gr.Markdown(f"`{ASR_MODELS[default_asr_model_size]}`")
            asr_model_size.change(fn=update_asr_model_size, inputs=asr_model_size, outputs=asr_model_info)
            gr.Markdown("### 翻訳モデル（Qwen3）")
            translate_model_size = gr.Dropdown(
                choices=list(TRANSLATE_MODELS.keys()),
                value=default_translate_model_size,
                label="翻訳モデルサイズ",
            )
            translate_model_info = gr.Markdown(f"`{TRANSLATE_MODELS[default_translate_model_size]}`")
            translate_model_size.change(fn=update_translate_model_size, inputs=translate_model_size, outputs=translate_model_info)

    cv_btn.click(
        fn=generate_custom,
        inputs=[cv_text, cv_language, cv_speaker, cv_instruct, model_size],
        outputs=[cv_audio, cv_time],
    )
    vd_btn.click(
        fn=generate_design,
        inputs=[vd_text, vd_language, vd_instruct, model_size],
        outputs=[vd_audio, vd_time],
    )
    vc_btn.click(
        fn=generate_clone,
        inputs=[vc_text, vc_language, vc_ref_audio, vc_ref_text, model_size],
        outputs=[vc_audio, vc_time],
    )
    vc_transcribe_btn.click(
        fn=transcribe_ref,
        inputs=[vc_ref_audio, asr_model_size],
        outputs=[vc_ref_text, vc_language],
    )
    vc_style_preset.change(
        fn=fill_style_preset,
        inputs=vc_style_preset,
        outputs=vc_style_instruct,
    )
    vc_translate_btn.click(
        fn=translate_text,
        inputs=[vc_ref_text, vc_translate_lang, translate_model_size, vc_style_instruct],
        outputs=[vc_text, vc_language],
    )
    vc_load_btn.click(
        fn=load_voice,
        inputs=vc_voice_dd,
        outputs=[vc_ref_audio, vc_ref_text, vc_language],
    )
    vc_delete_btn.click(
        fn=delete_voice,
        inputs=vc_voice_dd,
        outputs=vc_voice_dd,
    )
    vc_save_btn.click(
        fn=save_voice,
        inputs=[vc_voice_name, vc_ref_audio, vc_ref_text, vc_language],
        outputs=vc_voice_dd,
    )

if __name__ == "__main__":
    import socket

    env_port = os.environ.get("APP_PORT", "")
    if env_port.isdigit():
        port = int(env_port)
    else:
        port = 7860
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    break
            port += 1

    demo.launch(server_name="0.0.0.0", server_port=port, share=False,
                allowed_paths=[str(OUTPUTS_DIR)])
