#!/usr/bin/env python3
"""Genera un ref.json per la validazione REF-mode dei motori C.

Da eseguire su una macchina con rete e transformers installato:

    pip install torch transformers
    python3 make_ref.py Qwen/Qwen3-0.6B "The capital of France is" 24 > ref_qwen.json

Poi sul motore:

    REF=ref_qwen.json SNAP=/path/to/snapshot ./qwen

Il motore fa greedy sugli stessi prompt_ids e confronta token per token
con full_ids: il gate di fase e' il match completo (f32).
"""
import json
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit(f"uso: {sys.argv[0]} <model-id-o-dir> <prompt> [n_new=24]")
    model_id, prompt = sys.argv[1], sys.argv[2]
    n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 24

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=n_new,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    json.dump(
        {
            "model": model_id,
            "prompt": prompt,
            "prompt_ids": ids[0].tolist(),
            "full_ids": out[0].tolist(),
        },
        sys.stdout,
    )
    print()


if __name__ == "__main__":
    main()
