<<<<<<< HEAD
# My-AI-Engineering-PlayGround
=======
# Become an AI Engineer — Course Documentation

> **Course:** Become an AI Engineer  
> **Instructor:** Ali Aminian  
> **Format:** Learn by Doing  
> **Author:** Muhammad Ali

Personal documentation, lecture digests, and hands-on experimentation notes for the *Become an AI Engineer* course. Each week contains structured notes alongside practical Colab experiments.

---

## Repository Structure

```
.
├── README.md
├── Week-One/
│   ├── LLM-Foundation/
│   │   └── llm_foundations.md        ← Week 1 lecture notes & resource digest
│   └── My-Experimentation/           ← Google Colab notebooks & experiment logs
└── ...
```

---

## Progress

| Week | Topic | Notes | Experiments |
|------|-------|-------|-------------|
| **Week 1** | LLM Foundations & Lifecycle Architecture | [📄 Notes](./Week-One/LLM-Foundation/llm_foundations.md) | [🧪 Colabs](./Week-One/My-Experimentation/) |
| Week 2 | *(coming soon)* | — | — |

---

## Week One — Highlights

**Topic:** Large Language Model Foundations and Lifecycle Architecture

A deep dive into the full LLM lifecycle, from raw data engineering through transformer internals, inference strategies, post-training alignment, evaluation, and production deployment. Key takeaways:

- Data curation is the real architectural moat — a 1.3B aligned model beats a 175B unaligned one
- Transformer self-attention (Q/K/V), multi-head attention, and positional encoding
- Decoding strategies: greedy vs. beam search vs. nucleus sampling
- The alignment pipeline: SFT → RLHF → PPO
- Why BLEU/ROUGE are obsolete and what replaced them

→ [Full notes](./Week-One/LLM-Foundation/llm_foundations.md)

---

## About This Repo

These are personal learning notes — a mix of lecture digests, curated resource summaries, and experiment logs. The goal is to build a reference I can actually use, not just a transcript of slides.

If you're taking the same course, feel free to use this as a reference, but write your own experiments.
>>>>>>> d774c6b (Inital Commit)
