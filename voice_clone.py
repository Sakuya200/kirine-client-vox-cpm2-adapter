import argparse
from pathlib import Path
import sys


def ensure_src_root_on_path() -> None:
    src_root = Path(__file__).resolve().parents[1]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


ensure_src_root_on_path()

from vox_cpm2 import compose_generation_text, load_model_and_dependencies
from vox_cpm2.params import load_voice_clone_params


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", dest="params_file", type=str, required=True)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    ref_audio_path = Path(args.ref_audio_path).expanduser().resolve()
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")
    if not args.text.strip():
        raise ValueError("Target text cannot be empty.")

    mode = args.mode.strip().lower()
    if mode not in {"reference", "ultimate"}:
        raise ValueError(f"Unsupported VoxCPM2 clone mode: {args.mode}")
    if mode == "ultimate" and not args.ref_text.strip():
        raise ValueError("Ultimate cloning requires reference transcript.")

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return ref_audio_path, output_path


def generate_voice_clone_audio(args: argparse.Namespace):
    ref_audio_path, output_path = validate_args(args)
    model, deps = load_model_and_dependencies(args.init_model_path, args.device)
    mode = args.mode.strip().lower()

    # Ultimate cloning is audio continuation: the model reads the whole target
    # text aloud, so the "(style)text" control format is unsupported there and
    # the style prefix would be read out. Upstream disables the control
    # instruction in this mode as well.
    if mode == "ultimate":
        if args.style_prompt.strip():
            print(
                "Style prompt is not supported in ultimate cloning mode and will be ignored.",
                file=sys.stderr,
            )
        generation_text = args.text.strip()
    else:
        generation_text = compose_generation_text(args.text, args.style_prompt)

    if mode == "ultimate":
        wav = model.generate(
            text=generation_text,
            prompt_wav_path=str(ref_audio_path),
            prompt_text=args.ref_text.strip(),
            reference_wav_path=str(ref_audio_path),
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
        )
    else:
        wav = model.generate(
            text=generation_text,
            reference_wav_path=str(ref_audio_path),
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
        )

    deps.sf.write(str(output_path), wav, model.tts_model.sample_rate)


def main(argv: list[str] | None = None):
    cli_args = parse_args(argv)
    params = load_voice_clone_params(cli_args.params_file)
    generate_voice_clone_audio(params.to_namespace())


if __name__ == "__main__":
    main()