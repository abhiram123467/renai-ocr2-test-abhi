"""
src/evaluate_cer.py
Utility functions for CER, WER, and character confusion analysis.
Author: Abhiram | GSoC 2026 HumanAI / RenAIssance OCR-2
"""
from jiwer import cer, wer
from collections import Counter
import numpy as np

def compute_cer(reference, hypothesis):
    if not reference:
        raise ValueError("Reference string cannot be empty.")
    return cer(reference, hypothesis)

def compute_wer(reference, hypothesis):
    if not reference:
        raise ValueError("Reference string cannot be empty.")
    return wer(reference, hypothesis)

def compute_batch_metrics(references, hypotheses):
    assert len(references) == len(hypotheses)
    per_sample = []
    for ref, hyp in zip(references, hypotheses):
        per_sample.append((compute_cer(ref, hyp), compute_wer(ref, hyp)))
    cer_values = [s[0] for s in per_sample]
    wer_values = [s[1] for s in per_sample]
    return {
        "mean_cer": float(np.mean(cer_values)),
        "mean_wer": float(np.mean(wer_values)),
        "std_cer":  float(np.std(cer_values)),
        "std_wer":  float(np.std(wer_values)),
        "per_sample": per_sample,
    }

def character_confusion(reference, hypothesis):
    confusion = Counter()
    for ref_c, hyp_c in zip(reference, hypothesis):
        if ref_c != hyp_c:
            confusion[(ref_c, hyp_c)] += 1
    return confusion

def apply_historical_postprocessing(text):
    import re
    text = text.replace('\u017f', 's')
    text = re.sub(r'\bfu\b', 'su', text)
    text = re.sub(r'\bfi\b', 'si', text)
    text = re.sub(r'(\b\w*)nn(\w*\b)', r'\1n\2', text)
    text = re.sub(r' +', ' ', text).strip()
    return text
