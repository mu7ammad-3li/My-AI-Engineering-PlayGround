# Experiment: Decoding Strategy Comparison
**Model:** GPT-2 (pretrained, no fine-tuning)  
**Week:** 1 — LLM Foundations  

---

## Setup

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    pad_token_id=tokenizer.eos_token_id
).to(torch_device)
```

All experiments use the same prompt:

```python
prompt = "harry was going to work on a reqular monday ... "
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(torch_device)
```

---

## 1. Greedy Search (Baseline)

```python
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

**Output:**
```
harry was going to work on a reqular monday ...  I was going to work on a reqular
monday.  I was going to work on a reqular monday.  I was going to work on a
```

**Observation:**  
Hard repetition loop — the model locks onto the highest-probability token at every step and never escapes. The output is completely unusable. This is the textbook failure mode of greedy decoding.

---

## 2. Greedy Search (+ Repetition Penalty)

```python
output = model.generate(
    input_ids,
    max_length=50,
    repetition_penalty=1.3  # penalizes tokens the model has already used
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

**Output:**
```
harry was going to work on a reqular monday ...  I'm not sure if he's been working
out or just doing some other stuff. So, I went back and checked the email address
of my friend who is also in
```

**Observation:**  
A single parameter made greedy output usable — the repetition loop is broken and the output has a genuine narrative direction. However, there is a clear POV drift: the story starts in third person ("harry was going to work") and slips into first person ("I went back and checked"). Greedy with a penalty produces more interesting tokens, but without a search strategy it loses narrative consistency.

---

## 3. Beam Search (Baseline)

```python
output = model.generate(
    input_ids,
    max_length=50,
    num_beams=3,
    no_repeat_ngram_size=5,
    early_stopping=True
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

**Output:**
```
harry was going to work on a reqular monday ...  He was going to be working on a
monday.  So he was going to be on a monday and he was going to have to work on a
monday
```

**Observation:**  
Softer repetition than greedy — the model doesn't loop the exact same phrase, but it circles the same concept ("monday... monday... monday"). Beam search with `num_beams=3` and a large `no_repeat_ngram_size=5` isn't enough to escape GPT-2's high-probability attractor around the prompt's anchor word.

---

## 4. Beam Search (Tuned)

```python
output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=3,   # tighter n-gram blocking
    repetition_penalty=1.3,
    early_stopping=True,
    length_penalty=1.2        # rewards longer, committed sequences
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

**Output:**
```
harry was going to work on a reqular monday ...  I'm not sure if he would have been
able to do it, but I'm sure he could have done it.  It's not like he didn't want to
```

**Observation:**  
The best deterministic output of the experiment. The output is coherent, maintains consistent voice, and introduces genuine narrative reasoning ("It's not like he didn't want to"). The key driver was dropping `no_repeat_ngram_size` from 5 to 3 — blocking shorter repeated sequences broke the attractor earlier. Parameter tuning matters more than algorithm choice at this scale.

---

## 5. Top-K Sampling

### k=10
```python
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_k=10,
    pad_token_id=tokenizer.eos_token_id
)
```
**Output:**
```
harry was going to work on a reqular monday ... !!! I just thought it would be
interesting to see what the other day's results would be! Thanks so much!

Anonymous 05/11/15 (Wed) 11:
```

### k=50
```python
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id
)
```
**Output:**
```
harry was going to work on a reqular monday ...  She would be like: "Oh thank God.
I love that I've been so busy this afternoon." So I would have to say: I'm just so
grateful for [
```

### k=200
```python
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_k=200,
    pad_token_id=tokenizer.eos_token_id
)
```
**Output:**
```
harry was going to work on a reqular monday ...  but was really busy (which I did
due to fear) and with some recent shibboleths in my life, I'm not entirely sure it
will have any sort of
```

**Observation:**  
The repetition problem is completely gone across all three k values — the first win for stochastic methods. The tradeoff is grounding: as k grows, the output becomes lexically richer ("shibboleths" at k=200 would never appear in greedy) but narratively unhinged. GPT-2 trained heavily on Reddit and forum data, so opening the sampling pool wide pulls the model toward those registers. At k=10 output slips into forum comment style; at k=200 it becomes stream-of-consciousness.

---

## 6. Top-P (Nucleus) Sampling

Three values of p tested to observe the grounding vs. creativity tradeoff directly.

### p=0.5
```python
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_p=0.5,
    pad_token_id=tokenizer.eos_token_id
)
```
**Output:**
```
harry was going to work on a reqular monday ...  he's been on the phone with me a
lot lately and I'm just getting ready to go to work.  I've been thinking about it
and I'm just trying
```

### p=0.7
```python
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_p=0.7,
    pad_token_id=tokenizer.eos_token_id
)
```
**Output:**
```
harry was going to work on a reqular monday ... ive been here since September 15th.
ive been in touch with the owner of this place since July 15th, and we are not sure
what to do about the new lease
```

### p=0.9
```python
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)
```
**Output:**
```
harry was going to work on a reqular monday ... !!!!!!!

I'll add that my first attempt at the monday project was done very well. That said,
there are still a few things to do. First and foremost
```

**Observation:**  
The p=0.5 → 0.7 → 0.9 progression shows nucleus sampling's self-regulating nature clearly. Unlike top-k where the candidate pool is always exactly k tokens, top-p is dynamic — at p=0.5 the model is forced to commit to high-confidence tokens only, so it naturally stays closer to the prompt's narrative ("he's been on the phone... getting ready to go to work"). As p grows, it doesn't just add randomness — it adds lower-probability semantic territory that pulls output into unexpected domains. At p=0.7 the output drifts into a tenancy/lease scenario with no connection to Harry. At p=0.9 the output is most expressive but least grounded, with GPT-2's forum training data bleeding through in the exclamation spam.

The self-regulating behavior is what makes top-p generally preferable to top-k in practice — the pool size adapts to the model's confidence at each individual step rather than applying a fixed cutoff regardless of context.

---

## Summary Comparison

| Method | Params Tuned | Repetition | Grounding | Output Quality |
|---|---|---|---|---|
| Greedy (baseline) | ✗ | Hard loop | Strong | ✗ Unusable |
| Greedy + rep penalty | ✓ | Gone | Drifts POV | ⚠ Decent |
| Beam Search (baseline) | ✗ | Soft loop | Strong | ⚠ Poor |
| Beam Search (tuned) | ✓ | Gone | Consistent | ✓ Best deterministic |
| Top-K (k=10) | — | Gone | Weak | ⚠ Forum register |
| Top-K (k=50) | — | Gone | Weak | ⚠ Creative, ungrounded |
| Top-K (k=200) | — | Gone | ✗ None | ⚠ Stream-of-consciousness |
| Top-P (p=0.5) | — | Gone | Strong | ✓ Best stochastic |
| Top-P (p=0.7) | — | Gone | Moderate | ⚠ Drifts domain |
| Top-P (p=0.9) | — | Gone | Weak | ⚠ Creative, ungrounded |

## Key Takeaways

1. **Deterministic methods fail by default on GPT-2** — but tuning parameters (especially `no_repeat_ngram_size` and `repetition_penalty`) recovers usable output. Parameter choice matters more than algorithm choice at this model scale.

2. **Stochastic methods solve repetition completely** — but introduce a grounding problem. On an instruction-tuned model (GPT-4, LLaMA), the prompt would anchor the output even with high randomness. GPT-2 reveals the raw algorithmic behavior with no safety net.

3. **Top-p is more adaptive than top-k** — by dynamically sizing the candidate pool, it balances creativity and coherence better than a fixed k cutoff.

4. **The real lesson** — these decoding strategies are not model fixes. They are generation controls. The model's training and alignment determine the ceiling; the decoding strategy determines how you move within that ceiling.
