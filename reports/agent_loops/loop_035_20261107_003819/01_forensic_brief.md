# Loop #35 Forensic Diagnosis: Audio-Tower Latent Adaptation via Contrastive Activation Steering on Acoustic Encoder Representations

## 1. Executive Summary & Forensic Context
- **Current Baseline State:** Loop #32 (Dev WER: 1.51%, Dev CER: 1.05%, Dev Sentence EM: 87.63%, Train WER: 0.99%, Train Sentence EM: 92.00%, Latency: ~0.355s).
- **Loop #34 Post-Mortem & Forensic Failure Mode:**
  Loop #34 attempted discrete text-level / surface token replacement and CTC acoustic anchor verification (PY-GEC ICASSP 2025). The approach failed catastrophically on the Sealed Blind Dev set (Dev WER surged from 1.51% to 8.39%, Dev EM plunged from 87.63% to 22.07%).
  - **Root Cause of Discrete Failure:** Vietnamese is a monosyllabic tonal language where distinct lexical items differ by only 1-2 edit distance operations (e.g., "bắt" vs "tắt", "bằng" vs "mang", "dạ" vs "nhả"). Surface string replacement destroys token boundary priors, casing integrity (proper names like "Lê Minh", brand names like "Zalo", acronyms like "SJC"), and breaks the autoregressive sequence likelihood of Whisper's text decoder.
  - **Theoretical Shift:** SOTA speech literature (Sun et al. 2026; CLAnSR ICASSP 2025) demonstrates that acoustic, dialect, and voice-onset shifts must be rectified *in continuous latent space* within the Audio Tower (acoustic encoder representations) before sequence generation, allowing the decoder to perform natural, globally consistent autoregressive decoding.

---

## 2. Train Glass-Box Error Diff Forensic Breakdown

Investigation of the 12 persistent Train Glass-Box errors from Loop #32:

| Sample ID | Ground Truth | Whisper Hyp | Acoustic Latent Error Mechanism |
|---|---|---|---|
| `train_0147` | "Dạ Zalo đi ạ." | "Nhả Zalo đi ạ." | **Voice Onset Time (VOT) Pre-Voicing Shift:** Rapid speaker onset confuses voiced alveolar /z/ with palatal nasal /ɲ/. Discrete fixes corrupted "Zalo" into "gia lô". Continuous activation steering shifts encoder representations toward canonical voiced onsets. |
| `train_0038` | "Em ơi, đừng tắt đèn bếp nhé, cứ để đó." | "Em ơi, đừng bắt bếp nhé, cứ để đó." | **Bilabial vs Alveolar Stop Confusion:** Rapid stop burst [b] vs [t] before vowel /ă/. Audio encoder activations show energy ambiguity across low frequencies. Encoder latent steering biases stop trajectories without discrete surgery. |
| `train_0015` | "...không đủ thì lấy từ tiết kiệm dùm tôi hen." | "...hông đủ thì lấy từ tiết kiệm giùm tôi em." | **Southern Dialect Formant Shift:** Dialect elision of velar onset (/x/ -> /h/) and vowel-nasal tail (/en/ -> /em/). Latent steering aligns dialectal acoustic clusters toward standard phonological representations. |
| `train_0208` | "...dùng tiền tiết kiệm chuyển thay." | "...dùng tiết kiệm chuyển thôi." | **Co-articulation Swallowed Syllable & Particle Drift:** Low energy syllable "tiền" suppressed by rapid co-articulation; particle "thôi" vs "thay". Steering reinforces acoustic boundary representations. |
| `train_0105` | "...nhắc chị mang laptop..." | "...nhắc chị bằng laptop..." | **Nasal-Bilabial Acoustic Confusion:** Rapid co-articulation /m/ vs /b/. Steering in encoder latent space resolves nasalization ambiguity. |

---

## 3. Architectural Mechanism: Contrastive Activation Steering (CAA)

### Mathematical Formulation
Let the Whisper Audio Tower (encoder) produce hidden representations:
$$H_{enc} \in \mathbb{R}^{B \times T \times D}$$
where $B$ is batch size, $T$ is temporal sequence length (e.g. 1500 frames for 30s audio), and $D = 1280$ for PhoWhisper Large / Whisper Large-v2.

In Contrastive Activation Steering (Sun et al. 2026):
1. A directional steering vector $\vec{v}_{steer} \in \mathbb{R}^D$ is extracted contrastively along acoustic confusion axes:
   $$\vec{v}_{steer} = \frac{\mathbb{E}[h^+] - \mathbb{E}[h^-]}{\|\mathbb{E}[h^+] - \mathbb{E}[h^-]\|_2}$$
2. A PyTorch forward hook is registered on the encoder module (`model.get_encoder()`):
   $$H_{enc}' = H_{enc} + \alpha \cdot \vec{v}_{steer}$$
   where $\alpha \ge 0.0$ is the steering intensity parameter.
3. The cross-attention mechanism in Whisper's autoregressive text decoder seamlessly receives the steered representations:
   $$K = W_K H_{enc}', \quad V = W_V H_{enc}'$$
   guiding greedy token emission toward correct phoneme trajectories while completely preserving casing, token boundaries, and language model priors.

### Invariants & Zero-Compute Compliance
- **Identity Invariant:** When $\alpha = 0.0$, `steering_vector is None`, or `enable_steering is False`, $H_{enc}' = H_{enc}$ identically. Decoding is 100% bit-exact to the baseline.
- **Latency Overhead:** The forward hook addition is an $\mathcal{O}(D)$ vector addition per frame, adding $< 0.1$ ms to total inference time.
- **Local Zero-Compute Invariant:** On the local machine, only unit tests and mock verifications are executed. Large-scale vector extraction and full evaluation are executed on Colab GPU.
