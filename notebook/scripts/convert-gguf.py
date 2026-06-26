# Generated from: convert-gguf.ipynb
# Converted at: 2026-06-26T15:49:18.866Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!git clone --depth 1 https://github.com/ggml-org/llama.cpp

!cd llama.cpp

!pip -q install -r llama.cpp/requirements.txt
!pip -q install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

%%bash
HF_DIR="/content/drive/MyDrive/Sorachio-1B-4096-2e-4"
OUT_DIR="/content/drive/MyDrive/Sorachio-1B-GGUF"
OUTFILE="$OUT_DIR/sorachio-1b-q8_0.gguf"

mkdir -p "$OUT_DIR"

python3 /content/llama.cpp/convert_hf_to_gguf.py \
  "$HF_DIR" \
  --outfile "$OUTFILE" \
  --outtype q8_0

echo "GGUF saved to: $OUTFILE"