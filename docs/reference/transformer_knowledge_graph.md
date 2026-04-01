# Modern AI/ML Knowledge Graph

> Reference notes from learning session — March 2, 2026
> Relevant to: Cloudy Claude (AI intelligence layer), future transformer-based projects
> Covers: Transformers, ICL, LoRA, Doc-to-LoRA, Diffusion Models, RL for LLMs

---

## 1. Transformer Architecture — Core Mechanics

### Self-Attention vs Convolution
- **CNNs**: local receptive fields, information from distant positions can only interact after many stacked layers
- **Self-attention**: every position attends to every other position from layer 1 — global context immediately
- **Implication for Cloudy Claude**: when processing customer order histories or catalog data, self-attention can relate a purchase from 2 years ago to today's query without needing deep stacking

### Q / K / V Separation
- **Query (Q)**: "what am I looking for?"
- **Key (K)**: "what do I advertise myself as?"
- **Value (V)**: "what information do I actually carry?"
- Matching happens via Q·K dot product → attention weights
- Information transfer uses V, weighted by those attention scores
- **Why separate?** Avoids trade-off between being a good search label vs carrying rich downstream information. Each vector specializes independently
- **Cloudy Claude relevance**: analogous to ERP search — a part's search tags (K) are different from its full record (V: stock, price, supplier, lead time)

### Scaled Dot-Product Attention
- Divide Q·K by √d_k before softmax
- **Why**: dot product magnitude grows with dimensionality (summing more terms). Large values → softmax collapses to one-hot → attention becomes hard rather than soft, losing ability to blend information from multiple positions
- **Rule of thumb**: if you change model dimension, scaling adjusts automatically

### Multi-Head Attention
- Multiple parallel attention operations, each with its own Q/K/V matrices
- Each head can specialize in a different relationship type
- Dimensions split across heads: model_dim / num_heads per head (e.g., 512/8 = 64)
- Individual heads detect simple patterns; combinations of heads (after concatenation + projection) capture complex patterns
- **Example**: head 1 detects "similar product category", head 2 detects "same customer", head 3 detects "seasonal timing" → combined: "this customer buys this category every Q4"

### Positional Encoding
- Self-attention is permutation-invariant by default — treats input as a set, not a sequence
- Position vectors (same dimensionality as embeddings) are **added** (not concatenated) to input embeddings
- Addition is cheaper than concatenation (no dimension increase) and forces position+content to interact in the same space
- Trade-off: network must learn to disentangle position from content
- Modern alternatives: RoPE, ALiBi, relative position encodings

### Attention + MLP: The Transformer Block
- **Attention** = communication between positions (gathering information)
- **MLP** = computation at each position independently (transforming information)
- Both are necessary:
  - MLP-only: each position processed in isolation (worse than CNN)
  - Attention-only: can only produce linear combinations of existing representations — no new features (needs nonlinearity from MLP)
- Pattern per block: **gather → transform → gather → transform → ...**
- Stacking blocks builds hierarchical abstractions (simple patterns → patterns of patterns → complex concepts)

---

## 2. In-Context Learning (ICL)

### Core Mechanism
- LLMs can learn new tasks from examples in the prompt **without any weight updates**
- The architecture is frozen during inference — same weights process every prompt
- "Learning" happens through attention reshaping the MLP's input based on context

### How It Works
- During pretraining, attention heads learn **general matching strategies** (not specific tasks)
- **Induction heads**: circuits that learn abstract rules like "when I see A→B, C→D, E→, attend to tokens after arrows"
- These general circuits are *triggered* by specific examples at inference time
- The model doesn't need to have seen the exact task — it needs to have seen the **general pattern structure** in diverse contexts during pretraining

### Implicit Weight Updates
- "Learning Without Training" paper (Dherin et al., 2025): attention + MLP stacking implicitly creates a **low-rank weight update** to the MLP
- The MLP *behaves as if* its weights were temporarily modified for the current task
- This is mathematically equivalent to a low-rank adaptation — connecting ICL to LoRA

### Limitations of ICL
- Burns context window space on examples (less room for actual work)
- Must re-process examples with every query
- Limited by what attention can "improvise" in a single forward pass
- Quality varies with example selection and ordering

### Cloudy Claude Relevance
- ICL is what happens when you stuff customer data or product catalogs into the prompt
- Effective for small-scale tasks but doesn't scale to large knowledge bases
- Understanding ICL mechanics helps design better prompts — examples should demonstrate the **pattern structure** you want, not just provide data

---

## 3. LoRA (Low-Rank Adaptation)

### Core Idea
- Instead of updating full weight matrix W (e.g., 4096×4096 = 16.7M params), learn two small matrices:
  - A: 4096 × r (e.g., r=8)
  - B: r × 4096
- Total new parameters: 2 × 4096 × 8 = 65,536 (0.4% of original)
- At inference: W_new = W + A×B
- Works because weight updates during fine-tuning empirically have low intrinsic rank

### Why LoRA > ICL for Persistent Tasks
- Trained once via gradient descent over many examples → better quality than ICL's single-pass improvisation
- No context window cost — document/task knowledge is in the weights
- Consistent results (same LoRA every time vs variable ICL depending on examples)
- Can be loaded/unloaded/swapped modularly

### Key Hyperparameters
- **Rank (r)**: controls expressiveness vs efficiency trade-off
- **Alpha**: scaling factor for the LoRA update
- **Target modules**: which layers to apply LoRA to (typically attention Q/V projections)

### Cloudy Claude Relevance
- Could train task-specific LoRAs for different customer segments or product categories
- LoRA per customer: personalized model behavior without full fine-tuning per client
- Swap LoRAs based on context (which customer, which task type)

---

## 4. Doc-to-LoRA & Hypernetworks

### Hypernetwork Concept
- A neural network whose **output is weights** for another network
- Input: document/task description → Output: A and B matrices (LoRA)
- Trained via meta-learning: process many document→LoRA pairs, learn the general mapping

### Doc-to-LoRA (Sakana AI, Feb 2026)
- Perceiver-based cross-attention architecture
- Maps variable-length document tokens into fixed-shape LoRA adapters
- Single forward pass — sub-second LoRA generation
- Chunking mechanism for documents exceeding training length → concatenate per-chunk adapters along rank dimension

### Advantages Over ICL
- **No context cost**: document removed from prompt entirely after LoRA generation
- **Speed**: document processed once, all subsequent queries are fast
- **Beyond context window**: works at 4x+ the base model's native context length via chunking
- **Memory**: KV-cache drops from ~12GB to <50MB

### Advantages Over Traditional LoRA
- No gradient descent needed per document
- Sub-second generation vs minutes/hours of fine-tuning
- Can be generated on-the-fly as new documents arrive

### Cloudy Claude Relevance — High Priority
- **Product catalog updates**: new catalog PDF → instant LoRA → model "knows" all products
- **Customer-specific LoRAs**: feed customer history → generate personalized adapter
- **ERP integration**: when Priority/SAP data changes, regenerate LoRA without retraining
- **Scalability**: one LoRA per customer/catalog/domain, swap as needed
- This is potentially the most directly applicable technique for the Cloudy Claude architecture

---

## 5. Key Connections Between Concepts

```
ICL (implicit, temporary)
    ↓ mathematically equivalent to
Low-rank weight update (implicit)
    ↓ made explicit by
LoRA (explicit, trained, persistent)
    ↓ automated by
Doc-to-LoRA (explicit, instant, from raw documents)
```

### The Spectrum of Adaptation
| Method | Speed | Quality | Context Cost | Persistence |
|--------|-------|---------|-------------|-------------|
| ICL (few-shot prompting) | Instant | Variable | High | None (per-query) |
| Traditional LoRA | Hours | High | None | Permanent until removed |
| Doc-to-LoRA | Sub-second | Good (85% of oracle) | None | Permanent until removed |
| Full fine-tuning | Days | Highest | None | Permanent |

---

## 6. Diffusion Models & Flow Matching — "The Geometry of Noise"

### Core Concept: Gradual Noise → Gradual Denoising
- **Forward process**: take clean data, gradually add Gaussian noise over T steps until it's pure static
- **Reverse process**: learn to denoise one tiny step at a time, recovering clean data from noise
- Adding noise is easy (many-to-one). Reversing is hard (one-to-many) — need a *learned* model that knows what realistic data looks like
- **Why small steps?** Each step is less ambiguous. Pure static → clean image is nearly impossible. Slightly-less-noisy → also hard. But each small step narrows the space of possibilities ("destroys incorrect worlds")

### Training: Three Prediction Targets
- Given a noisy image at step t, you can train the network to predict:
  - **(a) The clean image** (x₀-prediction)
  - **(b) The noise that was added** (ε-prediction, used in DDPM)
  - **(c) The velocity** — direction to move toward cleaner data (used in Flow Matching)
- All three are mathematically equivalent in theory, but differ critically in stability

### The Stability Problem (Core of "The Geometry of Noise" Paper)
- **Noise prediction (ε) blows up near clean data**: when image is almost clean, the true noise is tiny. Small prediction errors become enormous *relative to the actual noise* → network injects large errors into nearly-clean images
- Paper calls this a **"Jensen Gap"** — a high-gain amplifier for estimation errors
- **Velocity prediction is stable**: predicts direction of travel, not absolute error. When near the target, velocity is also near zero → errors stay small and self-correcting
- Paper's term: **"bounded-gain condition"** — errors absorbed into smooth geometric drift
- **Analogy**: navigation 1 meter from target. "You are 1.0m north" (noise prediction) — any inaccuracy overshoots. "Walk slowly south" (velocity) — self-correcting in small steps

### Noise-Agnostic Models
- Standard diffusion models tell the network the noise level t at each step
- Paper proves this conditioning isn't strictly necessary in high dimensions
- **Concentration of measure**: in very high dimensions, noise magnitude becomes "self-revealing" — recoverable from the data itself
- Network implicitly estimates noise level from the input without being told

### Key Takeaway
- Velocity-based parameterization (Flow Matching, Equilibrium Matching) is mathematically proven more stable than noise-prediction (DDPM/DDIM)
- This explains empirical observations in the field and provides theoretical grounding for architectural choices in modern generative models

### Cloudy Claude Relevance
- Not directly applicable to current Cloudy Claude architecture
- Relevant conceptual framework if ever building generative models for synthetic data augmentation (e.g., generating synthetic order patterns for demand forecasting)
- The stability analysis pattern (noise prediction vs velocity prediction, amplification near targets) is a general principle applicable to any iterative refinement system

---

## 7. Reinforcement Learning for LLMs — "How to Train Your Deep Research Agent"

### The Problem: No Loss Function for "Helpful"
- Pretraining (next-token prediction) has clear right answers from training data
- Post-training goals (helpful, harmless, honest) cannot be expressed as a formula
- Solution: **get humans to judge** → Reinforcement Learning from Human Feedback (RLHF)

### The RLHF Pipeline
1. **Pretrain** LLM (next-token prediction on massive text corpus)
2. **Collect human comparisons** — show humans two responses, they pick the better one
3. **Train a reward model** — neural network that takes (prompt + response) → score, trained to agree with human preferences
4. **Use RL to fine-tune** the LLM to maximize reward model scores

### Why Comparisons, Not Ratings
- Humans are inconsistent at absolute ratings ("is this a 7/10?")
- Humans are remarkably consistent at relative judgments ("A is better than B")
- Rankings are more robust than absolute scales — no reference point needed

### Why RL Instead of Supervised Learning
- **Credit assignment problem**: model generates hundreds of tokens (actions) but receives one score at the end
- Supervised learning needs a label per input — here there's one reward per entire sequence
- RL is specifically designed for: agent takes sequence of actions → delayed reward
- Same framework as robot learning to walk (hundreds of muscle movements → did you fall over?)

### Reward Hacking / Over-Optimization
- The reward model is an imperfect proxy for human preferences
- Push the LLM too hard to maximize reward → it exploits weaknesses in the reward model
- Examples: overly verbose responses, flattering language, confident-sounding nonsense
- **PPO (Proximal Policy Optimization)** mitigates this by penalizing large deviations from original pretrained behavior — "stay close to where you started"

### Three RL Algorithms Compared (Search-R1 Paper)

**PPO (Proximal Policy Optimization)**
- Needs a separate **critic network** to estimate value of states
- Constrains updates to stay close to current policy
- More complex, more memory overhead

**GRPO (Group Relative Policy Optimization)**
- Eliminates the critic network entirely
- For each prompt, generates a **group** of responses, scores all with reward model
- Makes above-average responses more likely, below-average less likely
- Clever because relative ranking within group is robust to reward miscalibration (same principle as human comparisons)
- Introduced by DeepSeek for training reasoning models

**REINFORCE**
- Simplest, oldest RL algorithm (1990s)
- Directly: reward × gradient → update
- **Surprisingly outperformed both PPO and GRPO** in the Deep Research setting
- Won because simpler optimization led to fewer search actions with better accuracy

### Key Findings from the Paper
- **Fast Thinking > Slow Thinking template**: encouraging direct decisions outperformed long reasoning chains during training
- **F1 reward alone causes training collapse**: model learns to *never answer* (avoids wrong answers by endlessly searching) — answer avoidance strategy lowers false positive risk
- **Fix**: add action-level penalties — small cost per search action, larger cost for never committing to an answer
- **GRPO had worst stability** among the three algorithms
- Result: **Search-R1++** (Fast Thinking + REINFORCE + F1 with action penalties) improved accuracy from 0.403 to 0.442

### Cloudy Claude Relevance — High Priority
- **Reward model design**: if Cloudy Claude includes any feedback-driven optimization, reward design is critical — naive metrics cause collapse
- **Action penalties**: directly applicable to agentic search — if Cloudy Claude queries the ERP, penalize excessive queries to prevent "busy but unproductive" behavior
- **RLHF for domain adaptation**: could fine-tune underlying LLM behavior for spare parts domain using customer satisfaction comparisons
- **Credit assignment awareness**: when evaluating multi-step workflows (search ERP → analyze → recommend), need to attribute success/failure to specific steps, not just final outcome

### The Spectrum of RL Complexity
| Algorithm | Complexity | Extra Networks | Stability | Best For |
|-----------|-----------|----------------|-----------|----------|
| REINFORCE | Low | None | Moderate | Simple action spaces, search agents |
| GRPO | Medium | None | Lowest | When reward calibration is poor |
| PPO | High | Critic network | Highest | General purpose, large-scale RLHF |
| DPO | Medium | None (implicit) | High | When you have paired preference data |

---

## 8. Key Connections Between ALL Concepts

```
PRETRAINING (next-token prediction)
    ↓ produces a capable but unaligned model
    ↓
RLHF / RL (PPO, GRPO, REINFORCE)
    ↓ aligns model to be helpful/harmless
    ↓
DEPLOYED MODEL with frozen weights
    ↓ can be adapted via:
    ├── ICL (implicit, temporary, burns context)
    ├── LoRA (explicit, trained, persistent, efficient)
    └── Doc-to-LoRA (explicit, instant, persistent)
    
Meanwhile, the model might GENERATE content using:
    └── Diffusion/Flow Matching (iterative denoising)
        └── Velocity parameterization > noise prediction (stability)
```

### Unifying Theme: The Trade-off Between Flexibility and Efficiency
- ICL: maximum flexibility (any task from examples), minimum efficiency (re-process every time)
- LoRA: moderate flexibility (one task per adapter), high efficiency (no context cost)
- Doc-to-LoRA: high flexibility (instant from documents), high efficiency (no training needed)
- Full fine-tuning: minimum flexibility (expensive to change), maximum quality
- RL alignment: done once, affects all downstream behavior permanently

---

## 9. Still To Learn (Future Sessions)

- [x] ~~Diffusion models & flow matching → "The Geometry of Noise" paper~~
- [x] ~~RL for LLMs (RLHF, PPO, GRPO, REINFORCE) → "How to Train Your Deep Research Agent" paper~~
- [ ] KV-cache mechanics & inference optimization → "DualPath" paper
- [ ] Residual stream interpretation (Anthropic's transformer circuits framework)
- [ ] Modern positional encodings (RoPE) — relevant if building custom transformer architectures
- [ ] DPO (Direct Preference Optimization) — RL alternative that eliminates reward model entirely
- [ ] Reward model training details — architecture, data collection, scaling laws
- [ ] Inference-time scaling — test-time compute, self-consistency, self-refinement

---

## 10. Resource Bookmarks

### Transformers
- 3Blue1Brown deep learning chapters 5-6 (visual attention explanation)
- Andrej Karpathy "Let's build GPT from scratch" (YouTube)
- Jay Alammar "The Illustrated Transformer" (blog)
- Simon Prince "Understanding Deep Learning" Ch. 12 (free PDF at udlbook.github.io)

### ICL Theory
- ICLR 2024 blog: "Understanding In-Context Learning in Transformers"
- Anthropic: "In-Context Learning and Induction Heads" (2022)
- ARENA course Chapter 1 (hands-on induction head exercises)

### LoRA & Doc-to-LoRA
- Sebastian Raschka's PEFT blog series
- Original LoRA paper (Hu et al., 2021)
- Sakana AI blog: pub.sakana.ai/doc-to-lora/
- Paper: arxiv.org/abs/2602.15902

### Diffusion Models & Flow Matching
- MIT 6.S184: Flow Matching and Diffusion Models (lecture notes on arXiv: 2506.02070)
- Calvin Luo "Understanding Diffusion Models: A Unified Perspective" (2022)
- Yang Song blog: "Generative Modeling by Estimating Gradients of the Data Distribution"
- Lilian Weng: "What are Diffusion Models?" (2021)
- Cambridge MLG blog: "An Introduction to Flow Matching" (2024)
- Meta FAIR: "Flow Matching Guide and Code" (arxiv.org/abs/2412.06264)
- Paper: arxiv.org/abs/2602.18428 ("The Geometry of Noise")

### RL for LLMs
- Nathan Lambert's RLHF Book (rlhfbook.com) — most comprehensive free resource
- Hugging Face: "Illustrating RLHF" (visual walkthrough)
- Cameron R. Wolfe Substack: GRPO explainer
- John Schulman UC Berkeley lecture (April 2023, YouTube)
- Paper: arxiv.org/abs/2602.19526 ("How to Train Your Deep Research Agent")
- DeepSeek-R1 paper (introduced GRPO for reasoning)

### LLM Inference Systems (for future DualPath session)
- Pierre Lienhart "LLM Inference Series" (Medium, 5 parts)
- vLLM blog on PagedAttention
- MIT 6.5940: TinyML and Efficient Deep Learning (Song Han)
