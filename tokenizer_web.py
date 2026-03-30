import argparse
import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
from transformers import AutoTokenizer


DEFAULT_TOKENIZER_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_tokenizer_path(tokenizer_path: str) -> Path:
    candidate = Path(tokenizer_path).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()

    if candidate.is_file():
        if candidate.name != "tokenizer_config.json":
            raise ValueError("Tokenizer path must be a model directory or tokenizer_config.json file.")
        candidate = candidate.parent

    if not candidate.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {candidate}")

    required_files = ["tokenizer_config.json", "vocab.json", "merges.txt"]
    missing_files = [name for name in required_files if not (candidate / name).exists()]
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(f"Tokenizer directory is missing required files: {missing_text}")

    return candidate


@lru_cache(maxsize=8)
def load_tokenizer(tokenizer_path: str):
    resolved_path = resolve_tokenizer_path(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved_path),
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    return tokenizer, resolved_path


def get_tokenizer_summary(tokenizer_path: str):
    tokenizer, resolved_path = load_tokenizer(tokenizer_path)
    special_tokens = {
        key: value
        for key, value in tokenizer.special_tokens_map.items()
        if value is not None and value != []
    }
    summary = {
        "resolved_path": str(resolved_path),
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "special_tokens": special_tokens,
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def encode_text(tokenizer_path: str, text: str, add_special_tokens: bool):
    tokenizer, resolved_path = load_tokenizer(tokenizer_path)
    encoded_ids = tokenizer.encode(text or "", add_special_tokens=add_special_tokens)
    token_strings = tokenizer.convert_ids_to_tokens(encoded_ids)
    summary = {
        "resolved_path": str(resolved_path),
        "token_count": len(encoded_ids),
        "vocab_size": len(tokenizer),
    }
    return (
        json.dumps(encoded_ids, ensure_ascii=False),
        json.dumps(token_strings, ensure_ascii=False),
        json.dumps(summary, ensure_ascii=False, indent=2),
    )


def parse_token_ids(token_text: str):
    raw = (token_text or "").strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [int(item) for item in parsed]
    except json.JSONDecodeError:
        pass

    normalized = raw.replace("\n", " ").replace(",", " ")
    return [int(part) for part in normalized.split()]


def decode_tokens(tokenizer_path: str, token_text: str, skip_special_tokens: bool):
    tokenizer, resolved_path = load_tokenizer(tokenizer_path)
    token_ids = parse_token_ids(token_text)
    text = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    summary = {
        "resolved_path": str(resolved_path),
        "token_count": len(token_ids),
        "vocab_size": len(tokenizer),
    }
    return text, json.dumps(summary, ensure_ascii=False, indent=2)


def build_demo(default_tokenizer_path: str):
    with gr.Blocks(title="Qwen Tokenizer Web") as demo:
        gr.Markdown(
            "# Qwen Tokenizer Web\n"
            "Load a tokenizer directory or tokenizer_config.json, then encode text, decode token ids, and inspect tokenizer size."
        )

        tokenizer_path = gr.Textbox(
            label="Tokenizer path",
            value=default_tokenizer_path,
            placeholder="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign or /abs/path/tokenizer_config.json",
        )

        with gr.Row():
            summary_button = gr.Button("Refresh tokenizer summary", variant="primary")
            tokenizer_summary = gr.Code(label="Tokenizer summary", language="json")

        with gr.Tab("Encode"):
            encode_input = gr.Textbox(label="Input text", lines=8, placeholder="Type text to encode")
            add_special_tokens = gr.Checkbox(label="Add special tokens", value=False)
            encode_button = gr.Button("Encode text", variant="primary")
            encoded_ids = gr.Code(label="Token ids", language="json")
            encoded_tokens = gr.Code(label="Token strings", language="json")
            encode_summary = gr.Code(label="Encode stats", language="json")

        with gr.Tab("Decode"):
            decode_input = gr.Textbox(
                label="Token ids",
                lines=8,
                placeholder="Example: [108386, 3837, 48] or 108386 3837 48",
            )
            skip_special_tokens = gr.Checkbox(label="Skip special tokens", value=False)
            decode_button = gr.Button("Decode tokens", variant="primary")
            decoded_text = gr.Textbox(label="Decoded text", lines=8)
            decode_summary = gr.Code(label="Decode stats", language="json")

        summary_button.click(get_tokenizer_summary, inputs=[tokenizer_path], outputs=[tokenizer_summary])
        encode_button.click(
            encode_text,
            inputs=[tokenizer_path, encode_input, add_special_tokens],
            outputs=[encoded_ids, encoded_tokens, encode_summary],
        )
        decode_button.click(
            decode_tokens,
            inputs=[tokenizer_path, decode_input, skip_special_tokens],
            outputs=[decoded_text, decode_summary],
        )
        demo.load(get_tokenizer_summary, inputs=[tokenizer_path], outputs=[tokenizer_summary])

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenizer encode/decode web tool.")
    parser.add_argument(
        "--tokenizer-path",
        default=DEFAULT_TOKENIZER_PATH,
        help="Tokenizer directory path or tokenizer_config.json path.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for the web server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the web server.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link.")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_demo(args.tokenizer_path)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()