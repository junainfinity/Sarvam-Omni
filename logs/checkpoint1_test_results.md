# SarvamOmni Checkpoint 1 — Test Results

**Date**: 2026-04-10
**Checkpoint**: Stage 1.5, Step 1000
**Hardware**: Apple M4 Max, 128GB Unified Memory
**Device**: MPS | Dtype: bfloat16
**Python**: 3.11 | PyTorch 2.11.0 | Transformers 5.5.3 | PEFT 0.18.1

---

## Smoke Test: 5/5 PASSED

| # | Test | Status | Details |
|---|------|--------|---------|
| 1 | Tokenizer | PASS | vocab_size=262144, image_token_id=8 |
| 2 | Vision encoder forward | PASS | output shape [441, 4096], dtype bfloat16 |
| 3 | Projector transform | PASS | input (10, 4096) → output (10, 4096) |
| 4 | Text-only generation | PASS | "What is 2+2?" → "4" (2 tokens, 4.6 tok/s) |
| 5 | Image+text generation | PASS | Red dummy image → "Red." (3 tokens, 2.0 tok/s) |

---

## Text-Only Inference

| # | Prompt | Response | Tokens | Speed | Quality |
|---|--------|----------|--------|-------|---------|
| 1 | What is the capital of India? | New Delhi. | 4 | 5.8 tok/s | Correct |
| 2 | Translate to Hindi: The weather is beautiful today. | Yes. | 3 | 5.8 tok/s | Wrong — failed to translate |
| 3 | Explain gravity in one sentence. | Gravity is the force that attracts objects with mass. | 11 | 9.0 tok/s | Correct |
| 4 | Write a haiku about mountains. | Majestic peaks piercing sky high. | 8 | 13.4 tok/s | Partial — not 5-7-5 format |
| 5 | What is 15 multiplied by 23? | 5 multiplied by \(5\) | 8 | 12.1 tok/s | Wrong |

**Text-only assessment**: Basic factual Q&A works. Translation and arithmetic are weak. Responses are very terse. LoRA fine-tuning at step 1000 may have slightly degraded base LLM capabilities on non-VL tasks.

---

## Image + Text Inference

| # | Image | Size | Prompt | Response | Tokens | Speed | Quality |
|---|-------|------|--------|----------|--------|-------|---------|
| 1 | sample-dog.png | 800x457 | What animal is in this image? | Dog. | 3 | 2.0 tok/s | Correct |
| 2 | sample-dog.png | 800x457 | Describe this image in detail | White dog. | 4 | 4.1 tok/s | Correct but terse |
| 3 | sample-squirrel.jpg | 1024x683 | What animal is in this image? | Gribble. | 5 | 3.2 tok/s | Wrong |
| 4 | sample-squirrel.jpg | 1024x683 | Describe this image in detail | A green lizard. | 5 | 4.2 tok/s | Wrong |
| 5 | test.png | 856x1042 | What do you see in this image? | Screen. | 3 | 2.4 tok/s | Plausible |

**Image+text assessment**: Model correctly identifies dogs but struggles with other animals. Responses are extremely short (3-5 tokens). Vision-language alignment is partially working but needs more training steps. This is expected for step 1000/5000 of Stage 1.5.

---

## Bug Fix Applied

**Issue**: Image+text generation returned empty responses (0 tokens).

**Root cause**: When `model.generate()` uses `inputs_embeds` (vision path), HuggingFace's `generate()` returns only new tokens — not prefixed with input tokens. The caller (`run_image_inference`) sliced `output_ids[input_len:]` which produced an empty tensor.

**Fix**: In `sarvam_omni/model.py`, `generate()` method now prepends `input_ids` to the output when using `inputs_embeds`, ensuring consistent return format:
```python
if output_ids.shape[1] < input_ids.shape[1] + 1:
    output_ids = torch.cat([input_ids, output_ids], dim=1)
```

---

## Model Loading Profile

| Component | Params | Size (BF16) |
|-----------|--------|-------------|
| Qwen3-VL ViT (frozen) | 576,388,336 | 1.15 GB |
| Sarvam-30B MoE (LoRA) | 32,152,387,072 | 64.3 GB |
| Projector (trained) | 33,562,624 | 0.13 GB |
| LoRA adapter | ~73 MB | 0.07 GB |
| **Total** | **32,762,601,328** | **65.5 GB** |

---

## Overall Assessment

- **Pipeline**: Fully functional end-to-end (after bug fix)
- **Text quality**: Basic Q&A works, complex tasks (translation, math) fail
- **Vision quality**: Can identify simple objects (dog), struggles with others
- **Generation speed**: 2-13 tok/s on MPS (text faster than vision+text)
- **Recommendation**: Continue training through remaining ~4000 steps of Stage 1.5, then proceed to Stage 2 (agentic/GUI grounding)
