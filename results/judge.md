# UC1 blind A/B judgement (Claude Code, 2026-11-21)

**Setup:** 30 random UC1 test messages. Each pair compares base zero-shot against fine-tuned output, with A/B order randomized (seed 7). The verdicts were written to `uc1/judge_verdicts.json` **before** opening `uc1/judge_key.json`. Rubric: helpfulness (does it answer what was asked), tone, structure, and correct `{{Placeholder}}` use.

| Outcome | Count |
|---|---|
| Fine-tuned wins | **15** |
| Base wins | 5 |
| Tie | 10 |
| **FT win-rate, ties excluded** | **75%** |

**Where the fine-tuned model won:**
- It answers the question instead of asking for an order number that isn't needed. Base did this repeatedly, for example on "what hours can I call support", "list the payment methods" and "upgrade to gold".
- It uses house-style placeholders such as `{{Customer Support Hours}}` and `{{Account Category}}`.
- It gives numbered step-by-step instructions.

**Where the base model won (pairs 2, 4, 16, 21, 29):**
- The FT reply was vague ("contact our customer service team").
- The FT reply was **truncated** at the 300-token limit (pairs 4 and 29).
- The FT reply asked for a tracking number when the user asked *where* they could check.

**Caveats:**
- There is a single judge (Claude), and from style alone it's often possible to guess which output came from which model.
- Bitext replies are verbose, so the FT model inherits that verbosity and its occasional truncation.
