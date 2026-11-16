# Loop #34 Forensic Diagnosis: Syllable-Constrained Phonetic Onset Guard and Omitted Syllable Recovery with Domain Entity Preservation

## 1. Executive Summary & Forensic Context
- **Current Baseline State:** Loop #32 (Dev WER: 1.51%, Dev CER: 1.05%, Dev Sentence EM: 87.63%, Train WER: 0.99%, Train Sentence EM: 92.00%, Latency: ~0.355s).
- **Loop #33 Post-Mortem:**
  Loop #33 introduced zero-shot ChunkFormer CTC acoustic anchor verification. While ChunkFormer's frame posteriors accurately captured acoustic onset VOT ("Dạ" vs "Nhả" in `train_0147`) and swallowed syllables ("tiền tiết kiệm" in `train_0208`), Loop #33 failed to advance Dev because:
  1. Global SequenceMatcher similarity threshold (`< 0.50`) caused entire utterances to be rejected when CTC phonetically syllabified brands or numbers (e.g. `train_0147` similarity was 0.44 due to "Zalo" vs "gia lô", completely suppressing the onset correction).
  2. Unconstrained replacement corrupted domain brand entities ("Zalo" -> "gia lô"), acronyms ("SJC" -> "sgc"), and proper names ("Lê Minh" -> "liên minh"), overriding Whisper's superior lexical representations.
  3. Multi-word replacements substituted non-particle context ("chuyển thôi" -> "triển khai" in `train_0208`), increasing substitution errors.
- **Hypothesis for Loop #34:** Syllable-constrained phonetic onset guard and omitted syllable recovery with domain entity preservation (PY-GEC ICASSP 2025).

---

## 2. Train Glass-Box Error Diff Forensic Breakdown

Detailed acoustic and phonetic analysis of targeted Train Glass-Box errors:

| Sample ID | Ground Truth | Whisper Hyp | ChunkFormer CTC Hyp | Forensic Phonetic Analysis & Remedy |
|---|---|---|---|---|
| `train_0147` | "Dạ Zalo đi ạ." | "Nhả Zalo đi ạ." | "Dạ gia lô đi ạ" | **Low-Energy Pre-Voicing Onset Confusion:** Speaker onset /j/ (or /z/) confused with palatal nasal /ɲ/ ("Nhả"). CTC captures non-nasal frame [zaː]. However, CTC phonetically degrades "Zalo" to "gia lô". **Remedy:** Targeted Onset Guard updates onset "Nhả" -> "Dạ", while Strict Entity Immunity strictly preserves brand "Zalo". |
| `train_0208` | "...dùng tiền tiết kiệm chuyển thay." | "...dùng tiết kiệm chuyển thôi." | "...dùng tiền tiết kiệm triển khai" | **Swallowed Syllable Omission:** Speaker rapidly co-articulated "tiền tiết kiệm"; Whisper swallowed "tiền". CTC monotonic frame duration emits extra syllable "tiền". **Remedy:** Single-syllable omitted recovery inserts "tiền" without disturbing surrounding tokens or proper noun "Lê Minh". |
| `train_0006` | "...vàng SJC hôm nay dưới 123 triệu..." | "...vàng SJC hôm nay dưới 123 triệu..." | "...vàng sgc hôm nay dưới 123 triệu..." | **Domain Acronym Bastardization:** CTC decodes acronym as lowercase phonemes "sgc". **Remedy:** Strict Entity Immunity prevents CTC from modifying all-caps acronyms (`SJC`, `SMS`, etc.) or ITN digits (`123 triệu`, `9 giờ`). |
| `train_0038` | "Em ơi, đừng tắt đèn bếp nhé, cứ để đó." | "Em ơi, đừng bắt bếp nhé, cứ để đó." | "Em ơi đừng tắt bếp nhé cứ để đó" | **Voicing / Bilabial-Alveolar Stop Confusion:** Rapid bilabial stop [b] confused for unvoiced alveolar stop [t]. CTC correctly emits "tắt". **Remedy:** Confusion pair verification replaces "bắt" -> "tắt" while preserving sentence casing and punctuation. |

---

## 3. Architectural Design Pillars for Refactoring

1. **Strict Entity Immunity Detector (`_is_entity`):**
   - Any token containing digits (`\d`), all-caps acronyms (len $\ge 2$), registered domain brands (`zalo`, `momo`, `shopee`, `grab`, `be`, `sms`, `sjc`), or capitalized proper nouns is tagged as an Entity.
   - Protected entities can never be replaced, deleted, or corrupted by CTC approximations.

2. **Generic Phonetic Distance Bounded Substitution:**
   - Evaluates character edit distance and diacritic-insensitive distance between single syllables (`_phonetic_distance(w, c) <= 2`).
   - If distance $\le 2$ and neither token is an entity, trusts the monotonic acoustic CTC anchor for phonetic confusions and onset VOT corrections (e.g. `Nhả` -> `Dạ`, `bằng` -> `mang`, `bắt` -> `tắt`).
   - Completely generic: zero sample-specific keyword maps or hardcoded tokens.

3. **Flanked Omitted Syllable Recovery:**
   - Restores single dropped syllables ($j_2 - j_1 == 1$) from CTC when flanked by matched context words without entity collision, recovering rapid speech omissions (e.g. `tiền` before `tiết kiệm`).

4. **Tail Particle Normalization:**
   - Reconciles utterance-tail particles (`nhé`, `nha`, `hen`, `thôi`, `thay`) with acoustic frame evidence.
