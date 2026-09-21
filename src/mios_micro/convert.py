"""GGUF Model Conversion and Quantization Helper for MiOS-Micro.

Converts HuggingFace format weights into GGUF format and applies quantization
(Q4_K_M or Q8_0) for mainline llama-server / llama-swap deployment.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def convert_and_quantize(
    model_dir: str,
    output_gguf: str,
    quant_type: str = "Q4_K_M",
    dry_run: bool = False,
) -> dict:
    quant_type = quant_type.upper()
    valid_quants = ["Q4_K_M", "Q8_0", "Q5_K_M", "F16"]
    if quant_type not in valid_quants:
        return {"success": False, "error": f"Invalid quantization type '{quant_type}'. Must be one of {valid_quants}"}

    plan = {
        "model_dir": model_dir,
        "output_gguf": output_gguf,
        "quant_type": quant_type,
    }

    if dry_run:
        print("[mios-micro-convert] Dry-run validation passed.")
        return {"success": True, "dry_run": True, "plan": plan}

    f16_temp = str(Path(output_gguf).with_suffix(".f16.gguf"))
    
    # 1. Convert HF to GGUF F16
    print(f"[mios-micro-convert] Converting {model_dir} -> {f16_temp}")
    cmd_convert = [
        sys.executable,
        "-m", "llama_cpp.convert",
        model_dir,
        "--outfile", f16_temp,
        "--outtype", "f16",
    ]
    
    try:
        subprocess.run(cmd_convert, check=True)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        # Fallback to direct script if llama_cpp module convert not installed
        return {
            "success": False,
            "error": f"Failed to run conversion script: {e}. Ensure llama.cpp conversion tools are available.",
            "plan": plan,
        }

    # 2. Quantize
    print(f"[mios-micro-convert] Quantizing {f16_temp} -> {output_gguf} ({quant_type})")
    cmd_quant = ["llama-quantize", f16_temp, output_gguf, quant_type]
    try:
        subprocess.run(cmd_quant, check=True)
        if os.path.exists(f16_temp):
            os.remove(f16_temp)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        return {
            "success": False,
            "error": f"Failed to run llama-quantize: {e}",
            "plan": plan,
        }

    return {"success": True, "output_gguf": output_gguf, "quant_type": quant_type}


def main() -> int:
    parser = argparse.ArgumentParser(description="MiOS-Micro GGUF Conversion and Quantization")
    parser.add_argument("--model-dir", required=True, help="Directory containing HuggingFace model")
    parser.add_argument("--output", required=True, help="Path for output .gguf file")
    parser.add_argument("--quant-type", default="Q4_K_M", choices=["Q4_K_M", "Q8_0", "Q5_K_M", "F16"], help="Quantization target")
    parser.add_argument("--dry-run", action="store_true", help="Validate plan without conversion")
    args = parser.parse_args()

    res = convert_and_quantize(
        model_dir=args.model_dir,
        output_gguf=args.output,
        quant_type=args.quant_type,
        dry_run=args.dry_run,
    )

    print(res)
    return 0 if res.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
