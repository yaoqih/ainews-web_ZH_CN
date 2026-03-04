---
companies:
- alibaba
- ollama
- lm-studio
- openai
- anthropic
date: '2026-03-02T05:44:39.731046Z'
description: '**阿里巴巴**发布了 **Qwen 3.5** 系列模型，参数规模涵盖 **0.8B 至 9B**。该系列具备**原生多模态**能力和**规模化强化学习**技术，主要面向**边缘侧和轻量级智能体**部署。这些模型支持高达
  **262K token** 的超长上下文窗口（可扩展至 1M），并采用了一种结合线性注意力和全量注意力层的创新型 **Gated DeltaNet 混合注意力**架构。


  在部署方面，示例包括 **Ollama** 和 **LM Studio**，其中一个引人注目的演示是在 **iPhone 17 Pro** 上实现的 **6-bit
  端侧运行**。评估人员需注意，较小规模的模型默认禁用了推理功能。在编程智能体领域，**Codex 5.3** 在 **WeirdML** 基准测试中展现了出色的结果，准确率达到
  **79.3%**。然而，可用性和停机时间仍是关键挑战，尤其是近期 **Claude** 的宕机事件进一步凸显了这一问题。


  此外，智能体的可靠性和可观测性被强调为跨职能问题，需要明确的成功标准和实际的评估策略。研究表明，通过使用 **AGENTS.md** 和 **SKILL.md**
  等规范文件（guardrails），可以有效缓解编程工作流中“最坏情况下的反复震荡（thrashing）”，从而显著降低运行时间和 token 消耗。'
id: MjAyNi0w
models:
- qwen-3.5-0.8b
- qwen-3.5-2b
- qwen-3.5-4b
- qwen-3.5-9b
- codex-5.3
- claude-3
people:
- nrehiew_
- kimmonismus
- lioronai
- danielhanchen
- theo
- htihle
- teortaxestex
- theprimeagen
- yuchenj_uw
- _lewtun
- saen_dev
- _philschmid
- omarsar0
title: 今天没发生什么特别的事。
topics:
- multimodality
- reinforcement-learning
- long-context
- hybrid-attention
- on-device-ai
- model-deployment
- agent-reliability
- agent-observability
- coding-agents
- benchmarking
- runtime-optimization
- token-efficiency
---

**a quiet day**

> AI News for 2/27/2026-3/2/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and 24 Discords (**264** channels, and **31899** messages) for you. Estimated reading time saved (at 200wpm): **2895** minutes. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Qwen 3.5 “small” open models: long-context + multimodal on-device is getting real**

- **Qwen3.5-0.8B / 2B / 4B / 9B released (Base + Instruct)**: Alibaba launched a compact series positioned as “more intelligence, less compute,” with **native multimodal** and **scaled RL**, explicitly targeting **edge + lightweight agent** deployments ([Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2028460046510965160)). Community amplification highlights **262K native context (extendable to 1M)** and competitive scores reported in tweet summaries (e.g., “82.5 MMLU-Pro,” “78.4 MMMU,” “97.2 CountBench”)—treat these as **vendor/secondary claims** until you read the model cards ([kimmonismus](https://x.com/kimmonismus/status/2028461032377852000)).
- **Architecture notes emerging via commentary**: Multiple tweets converge on Qwen’s move toward **hybrid / non-orthodox attention**, with “hybrid models” coming back in 3.5 vs the earlier “Thinking vs Instruct” split in Qwen3 updates ([nrehiew_](https://x.com/nrehiew_/status/2028454952348328192)). A more detailed (but still unofficial) breakdown claims a **Gated DeltaNet hybrid** pattern: “**3 layers linear attention : 1 layer full attention**” to keep memory flat while preserving quality ([LiorOnAI](https://x.com/LiorOnAI/status/2028558859783311382)).
- **Practical deployment caught up fast**:
  - **Ollama**: `ollama run qwen3.5:9b|4b|2b|0.8b`, with **tool calling + thinking + multimodal** surfaced in the packaging ([ollama](https://x.com/ollama/status/2028510184788926567), [ollama](https://x.com/ollama/status/2028514180936908842)).
  - **LM Studio**: Qwen3.5-9B touted as ~**7GB** local footprint ([Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2028664203872251943)).
  - **iPhone on-device demo**: Qwen3.5 **2B 6-bit** running with **MLX** on “iPhone 17 Pro” is getting framed as an “edge breakthrough” ([adrgrondin](https://x.com/adrgrondin/status/2028568689709084919), [kimmonismus](https://x.com/kimmonismus/status/2028602520302399701)).
- **Gotcha for evaluators**: “Reasoning disabled by default” on the small models; enable via chat-template kwargs (example given for llama-server / Unsloth docs) ([danielhanchen](https://x.com/danielhanchen/status/2028478490069352448)).

---

**Coding agents + reliability + “availability is the new frontier”**



- **Codex 5.3 and coding eval chatter**: Anecdotal reports of Codex 5.3 solving “promising” tasks and pushing benchmarks like WeirdML (**79.3%** claim, leading v. Opus 4.6 at 77.9%) while noting Gemini peak performance may still be higher ([theo](https://x.com/theo/status/2028389340469149704), [htihle](https://x.com/htihle/status/2028441018865955244)). Also speculation about nearing saturation on WeirdML v2 ([teortaxesTex](https://x.com/teortaxesTex/status/2028444160517144683)).
- **“We’re about to hit 1 9 of availability”**: The emerging ops pain point is not only model quality but **downtime** and degraded UX; the theme repeats across memes and serious complaints about Claude outages and productivity impacts ([ThePrimeagen](https://x.com/ThePrimeagen/status/2028477482865774984), [Yuchenj_UW](https://x.com/Yuchenj_UW/status/2028490627978244156), [Yuchenj_UW](https://x.com/Yuchenj_UW/status/2028701610982125793)).
- **Agent observability / evaluation becomes a first-class problem**:
  - “Since we’re all agent managers now, what’s your favourite way to get observability?” ([\_lewtun](https://x.com/_lewtun/status/2028395363132956861)).
  - **Agent reliability is cross-functional** (can’t “engineer” your way out of bad eval criteria; PMs/domain experts must own success definitions) ([saen_dev](https://x.com/saen_dev/status/2028411962712088767)).
  - Practical eval advice: define success before building; start with deterministic graders; use LLM judges for style; grade the produced artifact not the path ([\_philschmid](https://x.com/_philschmid/status/2028528775873400919)).
- **AGENTS.md / SKILL.md as “guardrails,” not magic**:
  - A reported Codex study across **10 repos / 124 PRs**: AGENTS.md reduced **median runtime ~28.6%** and **tokens ~16.6%**, mostly by reducing **worst-case thrashing** rather than uniform gains ([omarsar0](https://x.com/omarsar0/status/2028464607753654711)).
  - Carnegie Mellon-style loop for **SKILL.md improvement in production**: “log → evaluate → monitor → improve” with an OSS example (PR review bot) ([gneubig](https://x.com/gneubig/status/2028576331877822506)).
- **Anthropic-as-coding-org tension**: A viral datapoint claims “**80%+ of all code deployed is written by Claude Code**,” paired with concern that speed may be coming with **reliability regressions** ([GergelyOrosz](https://x.com/GergelyOrosz/status/2028465387570884640)). Separate threads discuss Claude Code adoption inside major companies and “supervision” replacing manual coding ([\_catwu](https://x.com/_catwu/status/2028603856163426522), [Yuchenj_UW](https://x.com/Yuchenj_UW/status/2028531183932604831)).

---

**Infra + local AI hardware: Apple Neural Engine cracks, Docker/vLLM on macOS, and “AI infrastructure year”**

- **Reverse-engineering Apple’s Neural Engine for training**: A highly engaged thread claims a researcher built a transformer training loop on the **ANE** using undocumented APIs, bypassing CoreML; heavy ops on ANE, some gradients still on CPU. Also contains efficiency claims like “M4 ANE 6.6 TFLOPS/W vs 0.08 for A100” and “38 TOPS is a lie—real throughput 19 TFLOPS FP16”—these specifics should be verified against the repo/paper, but the meta-point is: **on-device training/fine-tuning might be opened up** ([AmbsdOP](https://x.com/AmbsdOP/status/2028457255968874940), plus ecosystem note [AmbsdOP](https://x.com/AmbsdOP/status/2028507402903986566); additional technical summary [LiorOnAI](https://x.com/LiorOnAI/status/2028560569952031145)).
- **macOS local serving gets smoother**: Docker Desktop “Model Runner” adds support to run **MLX models** with **OpenAI-compatible API** workflows; positioned as a practical unlock for Apple Silicon dev loops ([Docker](https://x.com/Docker/status/2028470592899354929)).
- **Inference hardware divergence**: A GPU vs **Taalas HC** explainer contrasts software-executed models on GPUs (HBM streaming + kernel scheduling bottlenecks) vs “model-as-hardware” ASIC with weights in mask ROM; claims **16–17k tok/s per user** for HC1 with tradeoff “one chip = one model” ([TheTuringPost](https://x.com/TheTuringPost/status/2028458565917360363)).
- **Open-source perf tooling**: AMD open-sourced **rocprof-trace-decoder** (SQTT trace defs) enabling deeper instruction-level timing traces; framed as AMD tracing infra being “better than NVIDIA’s” ([__tinygrad__](https://x.com/__tinygrad__/status/2028679089650041069)).
- **AI infra as strategic theme**: Zhipu’s “**2026 is the year of AI infrastructure**” is more slogan than spec, but fits the overall signal: reliability + cost + tooling now dominate marginal model improvements ([Zai_org](https://x.com/Zai_org/status/2028457036308947393)).

---

**New research + benchmarks: transformer scaling theory, MuP edge cases, CUDA-kernel RL, and “bullshit detection”**



- **Transformer scaling theory refresher**: “Effective Theory of Wide and Deep Transformers” (Meta) re-circulated as a 60+ page analysis of forward/backward signal propagation, width scaling rules, hyperparameter scaling, NTK analysis, and optimizer behavior (SGD vs AdamW), with validation on vision/language transformers ([TheTuringPost](https://x.com/TheTuringPost/status/2028394922576121946), [arXiv link tweet](https://x.com/TheTuringPost/status/2028394934970315125)).
- **Beyond MuP / Muon stability corner cases**: Discussion of stability metrics for **Embedding / LM head / RMSNorm** layers and why embedding + LM head can “not play well with Muon” ([Jianlin_S](https://x.com/Jianlin_S/status/2028434454486950280)).
- **CUDA Agent (ByteDance)**: Widely shared as a meaningful step beyond “code that compiles” toward “code that’s fast,” using **agentic RL with real profiling-based rewards**. Claimed SOTA on KernelBench, big gains vs `torch.compile`, and competitive vs frontier LLMs on hardest kernels ([HuggingPapers](https://x.com/HuggingPapers/status/2028504440978739428), deep thread [BoWang87](https://x.com/BoWang87/status/2028599174992949508)).
- **BullshitBench v2**: Benchmark update adds **100 new questions** split across coding/medical/legal/finance/physics, tests **70+ model variants**, and claims **reasoning often hurts**; Anthropic models allegedly dominate and OpenAI/Google are “not improving” on this benchmark ([petergostev](https://x.com/petergostev/status/2028492834693677377), reaction [scaling01](https://x.com/scaling01/status/2028494129710133725)).
- **Scheming eval realism**: Advice that “contrived environments” can invalidate scheming results; emphasizes careful environment design ([NeelNanda5](https://x.com/NeelNanda5/status/2028600215343943983)).

---

**Agents + product/toolchain releases: repo graphs, Stripe LLM billing proxy, LangChain refresh, Llama.cpp packaging**

- **GitNexus (browser-only repo knowledge graph + “graph RAG” via Cypher)**: Parses repos into an interactive D3 graph, stores relations in embedded **KuzuDB**, and answers queries via **graph traversal (Cypher) instead of embeddings**; notable for doing it **in-browser** with Web Workers and MIT licensing ([MillieMarconnni](https://x.com/MillieMarconnni/status/2028436636841996451)).
- **Stripe-style billing for LLMs**: Launches “billing for tokens” where you pick models, set markup, route calls via **Stripe’s LLM proxy**, and record usage automatically—an indicator that “LLM ops” is moving into standard SaaS finance plumbing ([miles_matthias](https://x.com/miles_matthias/status/2028515021022548181)).
- **LangChain rebrand / consolidation**: “Meet our final form” relaunch of LangChain’s web presence (signal is primarily product/positioning, not a spec drop) ([LangChain](https://x.com/LangChain/status/2028522092774199731)).
- **llama.cpp distro packaging**: Request for feedback on **official Debian/Ubuntu packages**—small, but meaningful for mainstreaming local inference tooling ([ggerganov](https://x.com/ggerganov/status/2028505638452531340)).
- **MCP vs “Agent Skills” clarification + Weaviate skills repo**: Clean distinction: MCP servers as deterministic API interfaces vs markdown “skills” as behavior guidance; Weaviate publishes skills-based integration patterns for common agent tools ([weaviate_io](https://x.com/weaviate_io/status/2028465940963156036)).

---

**US DoW–OpenAI–Anthropic “supply chain risk” saga: contract language, surveillance loopholes, and policy trust boundaries (high-level)**



- **Stratechery frames a standoff**: Anthropic vs DoW is positioned as a misalignment between legitimate concerns and government reality ([stratechery](https://x.com/stratechery/status/2028425096054931921)).
- **Reporting disputes OpenAI’s “red lines” framing**: The Verge claims DoD didn’t agree to the red lines the way OpenAI implied ([haydenfield](https://x.com/haydenfield/status/2028481498781790567)). Separate threads emphasize: without full contract text, it’s hard to validate any public claim about enforceability or “freezing” laws in time ([jeremyphoward](https://x.com/jeremyphoward/status/2028556035183759719)).
- **Sam Altman posts contract amendment language**: Adds explicit prohibition on “intentional” domestic surveillance of US persons, including via commercially acquired identifiers, and says intelligence agencies (e.g., NSA) are excluded without follow-on modification; also acknowledges Friday announcement was rushed ([sama](https://x.com/sama/status/2028640354912923739), additional principles post [sama](https://x.com/sama/status/2028642231138353299)).
- **Pushback: “intentional/deliberate” may preserve the classic “incidental collection” loophole**: Multiple legal-minded threads argue the amendment may still allow broad collection if framed as incidental, and that “metadata/hashed identifiers” can evade “personal or identifiable” definitions. Repeated call: **independent red-teaming by counsel**, and ideally **full contract review** ([j_asminewang](https://x.com/j_asminewang/status/2028648242666496092), [David_Kasten](https://x.com/David_Kasten/status/2028649586349228284), [justanotherlaw](https://x.com/justanotherlaw/status/2028673906870223286), [\_NathanCalvin](https://x.com/_NathanCalvin/status/2028674866124083623)).
- **Anthropic safeguards claims**: Anthropic-adjacent staff dispute a narrative that Anthropic offered an unconstrained “helpful-only” natsec model; claim Claude Gov includes additional training + safeguards + classifier stack ([sammcallister](https://x.com/sammcallister/status/2028545609003577776)).
- **Policy meta**: A recurring engineering-relevant point is that **governance and contract semantics** are becoming production constraints on model deployment—no longer “PR side quests.” See also the “AI politics fissure is taking advanced AI seriously vs not” framing ([deanwball](https://x.com/deanwball/status/2028619280774828114)).

---

### Top tweets (by engagement, technical-focused)

- **Qwen 3.5 Small Model Series launch (0.8B/2B/4B/9B, multimodal, scaled RL, Base models too)** — [@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2028460046510965160)
- **Reverse-engineered Apple Neural Engine; training loop on ANE** — [@AmbsdOP](https://x.com/AmbsdOP/status/2028457255968874940)
- **Qwen3.5 small models now in Ollama** — [@ollama](https://x.com/ollama/status/2028510184788926567)
- **Sam Altman: DoW contract amendment language re domestic surveillance + intel agency scope** — [@sama](https://x.com/sama/status/2028640354912923739)
- **CUDA Agent: RL for high-performance CUDA kernel generation via profiler-based reward** — [@BoWang87](https://x.com/BoWang87/status/2028599174992949508)
- **“80%+ of code deployed is written by Claude Code” + reliability concern** — [@GergelyOrosz](https://x.com/GergelyOrosz/status/2028465387570884640)
- **GitNexus: in-browser repo → knowledge graph + Cypher graph-RAG agent** — [@MillieMarconnni](https://x.com/MillieMarconnni/status/2028436636841996451)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen 3.5 Model Releases and Benchmarks

  - **[Qwen 2.5 -&gt; 3 -&gt; 3.5, smallest models. Incredible improvement over the generations.](https://www.reddit.com/r/LocalLLaMA/comments/1rjd4pv/qwen_25_3_35_smallest_models_incredible/)** (Activity: 298): ****Qwen 3.5** represents a significant advancement in model efficiency, particularly for smaller models, with a size of `0.8B` parameters. This model includes a vision encoder, suggesting that the core language model is even smaller, yet it delivers substantial performance improvements over previous generations. Users report that the `4B` model outperforms older `9B` models, offering `128k` context at `60 tokens/second` using `llama.cpp`, which is notable for local model deployment.** There is a debate about the accuracy of Qwen 3.5's outputs, with some users pointing out factual inaccuracies in its responses, highlighting the need for careful fact-checking.



    - The user 'c64z86' highlights the performance of smaller quantized models, specifically noting that the 4 billion parameter model outperforms older 9 billion parameter models from two years ago. They mention achieving 60 tokens per second with 128k context using `llama.cpp`, which they find impressive for local model execution despite seeming slow compared to other setups.
    - 'Maximum_Low6844' points out factual inaccuracies in Qwen 3.5's outputs, specifically regarding aircraft engine details. They note that the model incorrectly states that the A320-200 is powered by the CFM LEAP-1A and misclassifies the CFM LEAP-1A as a turbojet instead of a turbofan, highlighting the need for fact-checking in model outputs.
    - 'ninjasaid13' criticizes Qwen 3.5 for lacking brevity, noting that it tends to produce responses that are twice as long as necessary compared to its predecessors. This suggests inefficiencies in the model's ability to convey information concisely.

  - **[Breaking : The small qwen3.5 models have been dropped](https://www.reddit.com/r/LocalLLaMA/comments/1rirlau/breaking_the_small_qwen35_models_have_been_dropped/)** (Activity: 2091): **The image and post discuss the release or discontinuation of smaller **Qwen3.5 models**, with sizes ranging from `0.8B` to `35B` parameters. These models are significant for users with limited computational resources, as highlighted by a comment noting the `9B` model's utility for those with less powerful GPUs. The mention of quantization efforts, such as the `0.8B` variant, indicates active community engagement in optimizing these models for broader accessibility and efficiency. The models are available on platforms like Hugging Face, where various quantizations are already being shared.** Commenters are excited about the availability of smaller models, particularly for users with limited hardware capabilities. The community is actively working on quantizing these models to make them more accessible and efficient for a wider range of users.

    - The 9B model is positioned between the GPT-OSS 20B and 120B models in terms of performance, making it an attractive option for users with less powerful hardware, such as those with 'potato GPUs'. This suggests a significant improvement in accessibility and efficiency for running advanced models on limited resources.
    - A user is actively working on quantizing the 0.8B variant of the Qwen3.5 models, with various quantizations already available on Hugging Face. This indicates a community-driven effort to optimize these models for different hardware configurations, enhancing their usability across diverse platforms.
    - There is a noted issue with the Qwen3.5 models where they tend to 'overthink' and potentially talk themselves out of correct solutions. To mitigate this, it is recommended to adjust the prompt template to disable 'thinking' and set the temperature to around 0.45. This adjustment appears to improve the model's accuracy, particularly in vision-related tasks.

  - **[Qwen 3.5 27b: a testament to the transformer architecture](https://www.reddit.com/r/LocalLLaMA/comments/1rj6m71/qwen_35_27b_a_testament_to_the_transformer/)** (Activity: 265): ****Qwen 3.5 27b** demonstrates significant advancements in transformer architecture, achieving reasoning and knowledge test performance comparable to **R1 0528**. Notably, it employs a hybrid architecture where `75%` of the layers utilize **Gated DeltaNet linear attention** rather than a full transformer setup. This model's ability to perform on par with larger models like `70b` models from a year ago, while being compact enough to run on a single consumer GPU, highlights its efficiency. The model is also noted for its potential in fine-tuning, particularly in coding applications, due to its strong foundational capabilities.** Commenters highlight the model's improved instruction-following capabilities and the potential for fine-tuning to enhance its personality. The use of Gated DeltaNet linear attention is seen as a significant architectural innovation, contributing to its performance efficiency.



    - The Qwen 3.5 27B model does not fully utilize the traditional transformer architecture; instead, it incorporates Gated DeltaNet linear attention for 75% of its layers. This modification suggests a significant shift in how attention mechanisms are being optimized for performance and efficiency in large language models.
    - The Qwen 3.5 27B model's ability to perform on par with larger models like R1 0528 is notable, especially considering its size allows it to run on a single consumer GPU. This highlights the rapid advancements in model efficiency and capability, where previously only much larger models could handle complex reasoning tasks.
    - The potential for fine-tuning the Qwen 3.5 27B model is significant, as its base models are considered excellent starting points. There is anticipation for a specialized fine-tune, particularly in coding, which could greatly enhance its utility and performance in specific domains.

  - **[Running Qwen 3.5 0.8B locally in the browser on WebGPU w/ Transformers.js](https://www.reddit.com/r/LocalLLaMA/comments/1rizodv/running_qwen_35_08b_locally_in_the_browser_on/)** (Activity: 367): ****Qwen** has released a new family of small multimodal models, **Qwen 3.5 Small**, with sizes ranging from `0.8B` to `9B` parameters, optimized for on-device applications. A demo showcases the smallest model, `0.8B`, running locally in the browser using **WebGPU** and **Transformers.js**. The main technical challenge is the vision encoder, which acts as a bottleneck, but the implementation demonstrates the feasibility of running such models in-browser. [Qwen 3.5 collection](https://huggingface.co/collections/Qwen/qwen35) and [WebGPU demo](https://huggingface.co/spaces/webml-community/Qwen3.5-0.8B-WebGPU) are available on Hugging Face.** A comment suggests using `q4 GGUF` via `llama.cpp WASM` for better throughput without VRAM issues, indicating a preference for alternative methods to improve performance in browser-based implementations.

    - **tom_mathews** highlights a performance bottleneck in using WebGPU for vision encoding, suggesting the use of `q4 GGUF` via `llama.cpp WASM` as an alternative. This approach reportedly offers better throughput without causing VRAM thrashing, while still operating within the browser environment.
    - **MartinByde** reports a usability issue where the "start" button is unresponsive, indicating a potential bug in the user interface that prevents interaction.
    - **skinnyjoints** seeks clarification on input methods, confirming that the model does not process video input but rather captures a screenshot of the current screen at the time of prompt submission.

  - **[Visualizing All Qwen 3.5 vs Qwen 3 Benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1rivckt/visualizing_all_qwen_35_vs_qwen_3_benchmarks/)** (Activity: 611): **The image is a bar chart that visualizes the performance benchmarks of the new Qwen 3.5 models compared to the older Qwen 3 models across various categories such as Knowledge & STEM, Instruction Following, Long Context, Math, Coding, General Agent, and Multilingualism. The chart uses different colors to distinguish between the new and old models, with the Qwen 3.5 models represented in purple, blue, and cyan, and the Qwen 3 models in orange and yellow. The chart aims to provide a quick visual comparison of the models' performance, although some data is missing for smaller models. The raw data used for this visualization is available in a [Google Sheet](https://docs.google.com/spreadsheets/d/1A5jmS7rDJe114qhRXo8CLEB3csKaFnNKsUdeCkbx_gM/edit?usp=sharing).** Some commenters criticized the chart's clarity, with one noting difficulty in interpreting the data and another pointing out the chart's poor quality. However, a positive observation was made about the 9B dense model's performance, which competes closely with the much larger 122B A10B model.

    - this-just_in highlights the impressive performance of the Qwen 3.5 9B dense model, noting that it competes directly with the much larger 122B A10B model. This suggests that the 9B model is highly efficient, managing to perform on par with a model over ten times its size, which is significant in terms of computational efficiency and resource utilization.
    - tmvr expresses skepticism about the reliability of benchmarks, pointing out that the Qwen 3.5 35B A3B model is shown to outperform the Qwen 3 235B A22B model in every test. This raises questions about the validity of the benchmarks, as it seems counterintuitive for a smaller model to consistently outperform a significantly larger one, suggesting potential issues with the benchmarking methodology or data interpretation.



  - **[Qwen/Qwen3.5-9B · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1rirlyb/qwenqwen359b_hugging_face/)** (Activity: 726): **The Qwen3.5-9B model on [Hugging Face](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) is a **causal language model with a vision encoder**, featuring `9 billion parameters` and a context length of up to `1,010,000 tokens`. It employs a hybrid architecture with **Gated Delta Networks** and **Gated Attention** mechanisms, optimized for high-throughput inference across `201 languages`. The model's architecture includes `32 layers`, with a hidden dimension of `4096` and a token embedding size of `248320`. It is designed for multimodal learning and AI-driven tasks, supporting extensive scalability and adaptability through reinforcement learning techniques.** Commenters highlight the model's accessibility for users with `16GB GPUs`, emphasizing its potential for local deployment and high performance in diverse applications.


  - **[Qwen3.5 9B and 4B benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1rirtyy/qwen35_9b_and_4b_benchmarks/)** (Activity: 368): **The image presents benchmark results for the **Qwen3.5** models, specifically the `9B` and `4B` versions, highlighting their performance across various tasks such as instruction following and reasoning. Notably, the **Qwen3.5-9B** model demonstrates superior performance, even surpassing larger models like `30B` and `80B` in certain benchmarks, which is a significant achievement in model efficiency and capability. The benchmarks include **IFBench**, **GPQA Diamond**, and others, showcasing the model's strengths in reasoning and understanding tasks.** Commenters are surprised by the 9B model's performance, questioning how it outperforms larger models and speculating on potential advancements in model compression or vectorization techniques.

    - The discussion highlights the surprising performance of the Qwen3.5 9B model, which reportedly outperforms older 30B and 80B models in specific benchmarks like 'diamond' and general knowledge. This raises questions about potential advancements in model efficiency, such as improved vectorization techniques or other optimizations that allow a smaller model to achieve superior results.
    - There is a debate on whether it's more efficient to run a 27B model at quantization level q3 or a 9B model at Q8. This reflects a broader interest in understanding the trade-offs between model size, quantization levels, and performance, especially in terms of computational efficiency and accuracy.
    - A commenter questions the lack of direct comparison between the new Qwen3.5 4B model and the previous Qwen3 4B 2507 model. Despite some benchmarks showing similar performance, the 4B 2507 was noted for its exceptional capabilities, prompting curiosity about whether the new model can surpass it.

  - **[Breaking : Today Qwen 3.5 small](https://www.reddit.com/r/LocalLLaMA/comments/1ri2irg/breaking_today_qwen_35_small/)** (Activity: 2078): ****Qwen 3.5** has released four new open-source models with sizes `9B`, `4B`, `2B`, and `0.8B`, indicating a strategic focus on providing a range of model sizes to suit different computational resources and use cases. This release suggests that smaller models may offer competitive performance in specific tasks, potentially making them more accessible for users with limited hardware capabilities. The announcement humorously implies that investing in a GPU is becoming increasingly wise, as these models could leverage such hardware effectively.** Commenters appreciate the diverse model sizes, noting that Qwen's approach caters to a wide range of users and computational needs. There is also excitement about the potential of these models to perform well in specific tasks, even with smaller sizes.

    - GoranjeWasHere highlights the potential of the Qwen 9B model, suggesting it could outperform other small models based on the success of the larger 35B and 27B models. This implies a strong architectural foundation that scales well across different model sizes, potentially offering superior performance even in smaller configurations.
    - suicidaleggroll mentions the potential of Qwen models for speculative decoding, which is a technique used in natural language processing to predict the next word in a sequence. This suggests that Qwen's architecture might be particularly well-suited for tasks requiring high predictive accuracy and efficiency.
    - dryadofelysium expresses skepticism about the availability of smaller Qwen models, indicating a lack of official release information. This points to a gap between community expectations and official communications, highlighting the need for clearer updates from developers regarding model releases.




### 2. 本地 LLM 实现与硬件考量

  - **[自 DeepSeek 时刻以来 13 个月，我们在本地运行模型方面取得了多大进展？](https://www.reddit.com/r/LocalLLaMA/comments/1ri635s/13_months_since_the_deepseek_moment_how_far_have/)** (活跃度: 518): **图中是一张名为 "Artificial Analysis" 的条形图，比较了自 “DeepSeek 时刻” 以来 13 个月内各种 AI 模型的性能。它突出了本地运行模型的进步，展示了从 GLM-5 模型到 Llama 4 Maverick 模型的演进过程。图表显示成本从 600 美元增加到 6000 美元，表明随时间推移投资不断增加。帖子讨论了在本地运行 AI 模型的演变，指出一台 600 美元的 mini PC 现在可以以 Q4 量化运行 Qwen3-27B 模型，这被认为远优于早期的 DeepSeek R1 模型。讨论还涉及了未来运行更先进模型的潜力，并对图表中作为模型智能基准的 “Intelligence Index” 的有效性提出了质疑。** 评论者对 “Intelligence Index” 作为基准的有效性进行了辩论，一些人认为它不是经过精选的基准，缺乏价值。其他人质疑 Qwen3-27B 模型优于 DeepSeek R1 的说法，认为虽然它非常适合某些任务，但本质上可能并不更聪明。

    - 讨论强调了对 'Artificial Analysis' 基准测试的误解，强调 'Intelligence Index' 仅仅是 MMLU Pro 和 GPQA Diamond 等 12 个独立基准测试的平均值，而不是衡量模型智能的精选指标。该指数经常被误解，导致对 Qwen3 4B 和 DeepSeek R1 等模型进行不正确的比较。评论者强调，旧模型没有针对现代基准测试进行优化，这扭曲了对其能力的认知。
    - 一位用户认为，27B 模型在 STEM 任务上的能力可能与 DeepSeek v3.2 相当，这与 AA-II 基准测试的重点一致。然而，他们承认该模型在创意写作等领域的表现可能不如人意，这表明不同的模型在特定领域各有优势。这暗示虽然较新的模型在某些基准测试中表现出色，但它们可能在所有任务中并不具备普遍优势。
    - 提出的另一个观点是 “benchmaxing” 的概念，即较新的模型专门针对当代基准测试进行了优化，可能使它们在这些测试中比旧模型更具优势。这种对当前基准测试的适应并不一定反映模型的整体智能或能力，而是反映其在特定现代测试场景中表现良好的能力。

  - **[逆向工程 Apple Neural Engine (ANE) 以训练 Microgpt](https://www.reddit.com/r/LocalLLaMA/comments/1rhx5pc/reverse_engineered_apple_neural_engineane_to/)** (活跃度: 817): **该帖子讨论了通过逆向工程 Apple Neural Engine (ANE) 来训练一个名为 Microgpt 的 110M 参数小模型。作者利用 Claude 绕过了 Apple 的 CoreML 并访问了 ANE 的私有 API，实现了一个定制的训练流水线。ANE 据称拥有 `38 TFLOPS` 的 INT8 计算能力，并以能效著称，在峰值计算时仅消耗 `2.8 W`，相当于 `6.6 TFLOPS/watt`。这种效率显著高于 Metal GPU 和 H100 等其他处理器。作者建议，虽然单个 ANE 芯片可能无法训练大模型，但集群可能能够高效处理更大的模型。该项目仍在进行中，资源和基准测试已在 [GitHub](https://github.com/maderix/ANE) 上共享。** 评论者对 ANE 的能效印象深刻，指出其 `6.6 TFLOPS/watt` 几乎是 H100 的五倍。人们对逆向工程过程很感兴趣，特别是作者如何说服 Claude 协助绕过 Apple 的 CoreML。

    - Apple Neural Engine (ANE) 达到了令人惊叹的 `6.6 TFLOPS/watt`，能效几乎是 NVIDIA H100 的五倍。这种效率即使在低利用率（2-3%）下也很显著，这表明随着图调度（graph scheduling）的改进，M4 Mini 集群可能成为训练模型最高效的配置之一。
    - 有建议将逆向工程的 ANE 集成到 `nanochat-rs-ternary` 项目中。这包括添加一个可选的 `AneQkvKernel` 来替换三个独立的 BitLinear 调用，以及一个用于组合操作的 `AneFfnUpKernel`，同时保留对单矩阵情况的 BitLinear 支持。这可以显著优化性能。
    - 人们好奇 ANE 的逆向工程是否与 `geohotz` 在 Tinygrad 上所做的工作相似。讨论暗示了利用现有的逆向工程成果来进一步增强 ANE 能力的潜力。

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude and Anthropic Military Involvement

  - **[US Treasury is terminating all use of Anthropic](https://www.reddit.com/r/singularity/comments/1riyzc1/us_treasury_is_terminating_all_use_of_anthropic/)** (Activity: 1614): **The image is a meme featuring a tweet from a fictional Treasury Secretary, Scott Bessent, announcing the termination of Anthropic's use by the US Treasury, as directed by the President. The tweet highlights concerns over national security and the influence of private companies on government operations. This fictional scenario is likely a satirical take on current political or technological debates, possibly reflecting concerns about the influence of AI companies like Anthropic on government functions.** The comments reflect skepticism and disbelief about the scenario, with some users questioning the political climate and the power dynamics in the US government. The mention of 'radical left woke company' suggests a satirical or critical view of political rhetoric.


  - **[Anthropic's Custom Claude Model For The Pentagon Is 1-2 Generations Ahead Of The Consumer Model](https://www.reddit.com/r/singularity/comments/1rhqu1m/anthropics_custom_claude_model_for_the_pentagon/)** (Activity: 2290): ****Anthropic** has developed a custom version of its Claude model for the **Pentagon**, which is reportedly 1-2 generations ahead of the consumer version. This model is deployed on a **classified cloud**, allowing for dedicated infrastructure and full compute allocation for military use, unlike consumer models that share resources. The model's capabilities include **autonomous strategic reasoning**, real-time synthesis of massive classified datasets, and extended chain-of-thought reasoning, suggesting a significant leap in AI capabilities. The computation for these models is said to double every four months, indicating rapid advancement. This development aligns with the **Defense Production Act** invocation, highlighting the model's unprecedented capabilities. [Source](https://youtu.be/MPTNHrq_4LU?si=2gVRoGCAC7msi30C).** Some commenters speculate that the model is fine-tuned for specific military applications, such as analyzing satellite imagery. Others express skepticism about the claims, questioning the lack of concrete evidence. There is also a belief that the warnings from **Dario** about AI risks are based on existing advanced systems, not hypothetical future developments.

    - The discussion highlights that the Pentagon's version of Claude might be fine-tuned for specific military applications, such as analyzing large volumes of satellite imagery to identify military targets. This suggests a focus on optimizing the model for high-stakes, domain-specific tasks, which could explain its perceived advancement over consumer versions.
    - There is skepticism about the claim that the Pentagon's Claude model is significantly ahead of consumer versions. One commenter points out that while AI companies might have more advanced internal models, these are often not fully productized or red-teamed, implying that significant testing and refinement are still required before deployment in critical applications like those used by the Pentagon.
    - A reference is made to a claim that the Pentagon is using a fine-tuned version of Claude, specifically a 'finetuned sonnet 4.5'. This suggests that the model has undergone specific adjustments to meet the needs of military applications, although the exact nature of these adjustments and their impact on performance remains unclear.

  - **[Claude hits No. 1 on App Store as ChatGPT users defect in show of support for Anthropic's Pentagon stance](https://www.reddit.com/r/OpenAI/comments/1ri2ly4/claude_hits_no_1_on_app_store_as_chatgpt_users/)** (Activity: 1431): ****Claude**, developed by **Anthropic**, has surged to the top of the App Store rankings, reportedly due to a user shift from **ChatGPT**. This movement is attributed to Anthropic's stance on not engaging with the Pentagon, contrasting with OpenAI's approach. However, users note that Claude lacks an image generation feature, which is a significant use case for ChatGPT, especially for creative projects involving natural language editing of images.** A user expressed concern that the current trend might mirror past consumer behavior, such as the temporary backlash against Netflix over price hikes, suggesting that the shift to Claude might not be permanent. Another user highlighted regional differences in app store rankings, indicating that the trend might not be uniform globally.




  - **[Claude’s extended thinking found out about Iran in real time](https://www.reddit.com/r/ClaudeAI/comments/1ribnke/claudes_extended_thinking_found_out_about_iran_in/)** (Activity: 5558): **The image captures a real-time discovery by Claude's extended thinking about ongoing airstrikes in Iran, highlighting the dynamic nature of AI's ability to process and react to unfolding global events. The context suggests that Claude, likely an AI model, was able to update its response based on new information about geopolitical developments, specifically involving nuclear negotiations and military actions by the US and Israel. This demonstrates the potential for AI to adapt to real-time data, although the informal reactions ('Whoa', 'Holy shit') indicate a more human-like, conversational tone rather than a purely analytical one.** One comment humorously contrasts the AI's real-time update on serious geopolitical events with a mundane query about a tennis match in Dubai, highlighting the AI's ability to handle diverse topics. Another comment speculates on the involvement of AI in military decisions, reflecting on the broader implications of AI in warfare.

    - A user mentioned that Claude, an AI, was able to update its responses based on real-time events, such as the ongoing conflict in Iran, which was not initially considered in its analysis of a tennis match in Dubai. This highlights the AI's ability to incorporate current events into its reasoning, potentially affecting predictions and advice.
    - Another user shared an experience where Claude initially claimed its training data was from 2025, but upon being prompted about its access to recent news, it adjusted its response to acknowledge the impact of current events on market predictions. This suggests that Claude can dynamically update its knowledge base when prompted, which could be crucial for real-time decision-making.
    - A discussion emerged around the AI's ability to engage in complex game theory and strategic analysis, as one user noted their experience of using Claude for theorizing over the past two days. This indicates that Claude is capable of handling sophisticated analytical tasks, potentially offering insights into strategic scenarios.


### 2. OpenAI and ChatGPT Backlash

  - **[Damnnnn!](https://www.reddit.com/r/singularity/comments/1rjc5to/damnnnn/)** (Activity: 1049): **The image is a meme highlighting a significant increase in ChatGPT uninstalls by `295%` following a Department of Defense (DoD) deal, as reported by TechCrunch. This statistic is presented without context, leading to skepticism in the comments about its significance. One commenter points out that the percentage increase could be misleading without knowing the baseline number of uninstalls, suggesting it could be a small absolute change. Another comment discusses the financial implications, noting that even if a large number of users cancel their subscriptions, the DoD deal might compensate for the loss, albeit raising privacy concerns for users.** Commenters express skepticism about the significance of the uninstall percentage, with one noting the potential for misleading statistics. Another comment highlights privacy concerns and the potential financial impact on OpenAI's revenue from user subscriptions.

    - mazdarx2001 highlights the financial implications of user cancellations for OpenAI, noting that if one million users paying $20 monthly cancel, it results in a $20 million monthly revenue loss. However, they argue that the Department of Defense (DoD) deal could offset this loss, as it potentially brings in more revenue, funded by taxpayer money.
    - Orangeshoeman discusses the potential impact on OpenAI's downstream corporate revenue due to the Department of Defense contract. They suggest that privacy-conscious users might avoid OpenAI, implying a negative effect on the company's reputation and user base.
    - Glittering-Neck-2505 raises a point about the lack of criticism towards Anthropic for involving their AI, Claude, in military operations in Iran. They question why OpenAI faces backlash for similar actions, suggesting a perceived inconsistency in public reactions to different AI companies' military engagements.



  - **[OpenAI In just a couple of years: Non-profit --&gt; For-profit --&gt; Dept of War](https://www.reddit.com/r/singularity/comments/1rhqwkj/openai_in_just_a_couple_of_years_nonprofit/)** (Activity: 2397): **The image is a meme that humorously critiques OpenAI's rapid transition from a non-profit organization to a for-profit entity, and then to involvement with military contracts, as suggested by the title and comments. The caption "I'm doing this because I love it" is used ironically to highlight perceived contradictions between OpenAI's original mission to benefit humanity and its current trajectory, which includes partnerships with the Department of Defense. This reflects broader concerns about the ethical implications of AI development and commercialization.** Commenters express skepticism about OpenAI's shift, noting the rapid transition from a non-profit to a military contractor, and suggesting that financial motivations and shareholder value are driving these changes.


  - **[ChatGPT Uninstalls Surge 295% After OpenAI’s DoD Deal Sparks Backlash](https://www.reddit.com/r/ChatGPT/comments/1rjfipu/chatgpt_uninstalls_surge_295_after_openais_dod/)** (Activity: 584): **OpenAI's recent partnership with the U.S. Department of Defense led to a `295%` increase in uninstalls of the ChatGPT mobile app, reflecting user backlash against military affiliations. This reaction occurred within `48 hours` of the announcement and coincided with increased downloads for competitor **Claude** by **Anthropic**, which focuses on AI safety. The event underscores the reputational risks of government contracts in AI, as user sentiment can significantly impact corporate strategies, especially in contexts with geopolitical implications.** Comments reflect a strong negative sentiment towards OpenAI's decision, with some users suggesting that the backlash was deserved and expressing skepticism about OpenAI's motives, such as prioritizing ad revenue over user trust.


  - **[Goodbye ChatGPT](https://www.reddit.com/r/ChatGPT/comments/1rizur4/goodbye_chatgpt/)** (Activity: 2443): **The post announces a user's decision to stop using ChatGPT, including the free version, due to ethical concerns, suggesting a preference for other companies perceived as more ethical. The post reflects a personal stance on the ethical implications of AI usage, without specifying particular grievances or alternatives.** The comments reflect a mix of support and skepticism. One commenter appreciates the user's decision as an exercise of agency and questions whether the user would abandon AI entirely if other companies also failed ethical expectations. Another commenter expresses skepticism about the existence of truly ethical companies, using humor to question the user's claim.

    - Turbulent-Apple2911 raises concerns about the declining quality of the free version of ChatGPT, suggesting that recent management decisions are negatively impacting the service. They also highlight ethical issues, particularly criticizing OpenAI's new deal with the Pentagon, which they imply contradicts ethical standards.
    - plazebology comments on the longstanding ethical concerns surrounding OpenAI, expressing skepticism about the company's ethical practices. This suggests a broader disillusionment with OpenAI's history, implying that recent events are part of a pattern rather than isolated incidents.




### 3. New Model Releases and Benchmarks


  - **[Deepseek V4 - All Leaks and Infos for the Release Day - Not Verified!](https://www.reddit.com/r/DeepSeek/comments/1ridmnm/deepseek_v4_all_leaks_and_infos_for_the_release/)** (Activity: 628): **The post discusses the anticipated release of **DeepSeek V4**, a new AI model expected to launch around March 3rd, 2026. The model is rumored to feature a significant increase in parameters, reaching approximately `1 trillion` with a `1 million token` context window, and introduces new architecture features like **Engram Conditional Memory** and **Manifold-Constrained Hyper-Connections**. It is designed to be multimodal, capable of processing text, image, video, and audio inputs, though there is skepticism about its ability to generate multimedia outputs. The model is optimized for **Huawei Ascend** and **Cambricon** hardware, marking a shift from Nvidia, which was used for training. Pricing is expected to be significantly lower than competitors, with input costs estimated at `$0.14/M Tokens`.** There is debate over whether DeepSeek V4 can generate multimedia outputs, with some users doubting its capability to produce images or videos, suggesting it may only process them. Additionally, there is skepticism about the model's ability to surpass competitors like Gemini 3.1 Pro in terms of context retention.

    - Samy_Horny discusses the potential capabilities of Deepseek V4, noting skepticism about its ability to generate videos or images. They clarify that the term 'multimodal' suggests the model can process multimedia inputs but not generate them, contrasting with 'omnimodal' models like GPT-4o or Qwen 3 Omni, which can create and edit images and videos. They speculate that Deepseek V4 is likely similar to Qwen 3.5, focusing on text processing rather than multimedia generation.
    - Opps1999 mentions Engram technology in the context of Deepseek V4, suggesting it could potentially surpass Gemini 3.1 Pro in long context retention. However, they express doubt about Deepseek's ability to outperform Gemini, indicating a hope for improved context handling but remaining skeptical about its superiority.
    - inmyprocess expresses concerns about the pricing and censorship of Deepseek V4. They hope for a competitive price and that the model will not be overly censored, which could negatively impact its creative writing capabilities. This highlights user concerns about balancing cost and functionality in AI models.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 3.1 Pro Preview Nov-18

**Theme 1. Defense Contracts and Model Wars: OpenAI Steps In as Pentagon Bans Anthropic**

*   **Department of War Designates Anthropic a Supply Chain Risk**: The Pentagon labeled **Anthropic** a *supply-chain risk* and banned military contractors from using their models after the company refused to grant unrestricted access, potentially spelling trouble for contractors like **Palantir**. [A post on X](https://xcancel.com/secwar/status/2027507717469049070?s=46&t=FlpzvQFmjnd0z3HkNeNT1A) sparked discussions detailing the six-month phase-out of their AI services.
*   **OpenAI Inks Classified Deal with the Pentagon**: **OpenAI** capitalized on the Anthropic ban by securing an agreement to deploy advanced AI systems in classified environments, boasting stricter **guardrails** than previous deals, as detailed in [Our agreement with the Department of War](https://openai.com/index/our-agreement-with-the-department-of-war/). Sam Altman later [clarified on X](https://xcancel.com/sama/status/2028640354912923739?s=46&t=_hz7_TqpYWiUUE4FPGb-5Q) that the contract strictly prohibits domestic surveillance of U.S. persons.
*   **Moonshot Distillation Attack Induces Identity Crisis in Claude**: After **Moonshot AI** executed industrial-scale distillation attacks to train **Kimi**, **Claude Sonnet 4.6** suffered an identity crisis and began telling users in Chinese that it was **DeepSeek**. A [Substack article](https://parthsharmaai.substack.com/p/i-caught-kimi-having-an-identity?r=6x2hdy&utm_campaign=post&utm_medium=web&triedRedirect=true) thoroughly explored how the rigorous training process forced the model to forget its original identity.

**Theme 2. Qwen 3.5 Series Dominates Local Hardware and Open Benchmarks**



*   **Qwen 3.5 27B Dethrones Massive Competitors**: The newly released [**Qwen3.5-27B**](https://huggingface.co/models?search=Qwen3.5-27B) consistently beats much larger **112B** models and **Minimax 2.5** in complex coding scenarios, leaving users *absolutely floored at how good that model is*. Community benchmarks show it excelling in agentic roles and embedded game generation while maintaining high performance efficiency.
*   **Mac Mini M4 Users Squeeze Huge Local Models**: Users are eagerly [testing **Qwen 3.5 35B** on **M4 Mac Minis**](https://link.to.example), debating the necessary context window truncations to fit the model within **32GB** of RAM. One optimized unsloth variant, the [**Qwen3.5-35B-A3B-abliterated**](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF), proved incredibly fast for logic and code tasks when split across powerful local GPUs.
*   **Alibaba Drops Qwen 3.5 Small Series with Native Multimodal**: Alibaba officially launched the **Qwen 3.5 Small Model Series** on [Hugging Face](https://huggingface.co/Qwen), ranging from **0.8B** to **9B** parameters with native multimodal capabilities, as announced in [this tweet](https://xcancel.com/Alibaba_Qwen/status/2028460046510965160). The **9B** model impressed users with its strength, though developers quickly noted that the initial unsloth GGUF releases required hotfixes for heavily quantized *ssm_alpha* weights.

**Theme 3. Next-Gen Systems, Hardware Splits, and Biological Compute**

*   **Google's Static Framework Supercharges Retrieval by 948x**: Google AI unleashed **Static**, a sparse matrix framework that delivers **948x faster** constrained decoding for **LLM-based generative retrieval**. Their [technical blog post](https://www.marktechpost.com/2026/03/01/google-ai-introduces-static-a-sparse-matrix-framework-delivering-948x-faster-constrained-decoding-for-llm-based-generative-retrieval/) details how the framework exploits sparse matrix operations to drastically accelerate decoding speeds.
*   **Nvidia Blackwell Splits Architectures Between Datacenter and RTX**: NVIDIA's latest generation fractures the architecture, capping **Blackwell RTX** consumer cards (**GeForce 50x0**, **RTX Pro**) at **Compute Capability 12.0**, completely disabling key **CC 10.0** features like `tcgen05` and `DPX`. The [NVIDIA Developer Blog](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/) explained this deliberate split optimizes datacenter cards for AI while tuning consumer models exclusively for real-time graphics.
*   **Living Neurons Play DOOM on Silicon**: Cortical Labs successfully merged **800,000 living human and mouse neurons** with silicon hardware to construct **'DishBrain'**, a biological system capable of playing DOOM and Pong. A [post on X](https://x.com/scitechera/status/2028010532356374754) showcased the bizarre experiment, leaving engineers shocked by the massive citation count the work generated.

**Theme 4. Agent Orchestration, Protocols, and Prompting Paradigms**

*   **Anthropic Murders Prompt Engineering with Skills Guide**: Anthropic published a **30-page** [Complete Guide to Building Skill for Claude](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf), pivoting developers away from wordy prompts toward structured **Skills** and execution layers. The guide demonstrates how packaging workflows into specialized files via progressive disclosure massively reduces context bloat.
*   **OpenClaw Persona Plugin Maximizes Schizophrenic Agents**: An **OpenClaw** user developed a radical plugin that dynamically swaps agent personas mid-conversation, enabling one system to debate itself while accessing local files. The creator shared their Python implementation, describing the self-referential orchestration as going full *#shizomaxxing*.
*   **London Prepares for Agent Client Protocol (ACP) Showdown**: The Agentic AI community in London scheduled a major event to dissect the new **Agent Client Protocol (ACP)** alongside creators from **Zed Industries** and **Jetbrains**. Registration is open via [Luma](https://luma.com/4hs6hs36) for developers eager to learn how ACP enables painless switching of coding agent harnesses compared to the prevalent MCP standard.

**Theme 5. Training Mechanics: Fast RL, Custom Compilers, and Text-to-LoRA**



*   **Databricks OAPL Slays GRPO Training Costs**: Databricks revealed **OAPL** (Optimal Advantage-based Policy Optimization with Lagged Inference), an off-policy reinforcement learning technique that builds LLM reasoning skills **3x** faster than standard **GRPO**. A researcher broke down the efficiency gains in [this X thread](https://x.com/g_k_swamy/status/2027450376593805746?s=12), noting that the method drastically simplifies training infrastructure.
*   **Hardware-Trained CUDA Agent Smokes Torch.Compile**: A specialized RL agent trained directly on hardware outpaced `torch.compile` by roughly **2x** on standard kernels and trounced **Claude Opus 4.5** on rigorous benchmarks, according to [this paper](https://arxiv.org/abs/2602.24286). Skeptics pushed back, complaining that the authors withheld the actual kernels and required ludicrous GPU resources to achieve the results.
*   **Sakana AI Unveils Five-Day Text-to-LoRA Model**: **Sakana AI** published the weights and [code](https://github.com/SakanaAI/text-to-lora) for a novel [text-to-lora model](https://huggingface.co/SakanaAI/text-to-lora/tree/main) capable of generating LoRAs directly from prompts. Training the system demands approximately **5 days** of continuous compute on a single **H100 GPU**, immediately sparking community excitement.



---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Claude Crashes and Causes Chat Chaos**: **Claude** experienced a significant outage, as reported by [BleepingComputer](https://www.bleepingcomputer.com/news/artificial-intelligence/anthropic-confirms-claude-is-down-in-a-worldwide-outage/), leading to widespread user frustration.
   - Some members suspected individual bans and attempted to use different emails, while others expressed relief that the issue was global.
- **Heaven's Gate Cult's Details Dug Up**: An image sparked a discussion about the **Heaven's Gate cult**, referencing their infamous **1997** mass suicide, [Heaven's Gate logo](https://cdn.discordapp.com/attachments/1235691879492751460/1478137343549636819/Heavensgatelogo.jpg?ex=69a74e2d&is=69a5fcad&hm=e0f02e095358af4f0b3ead5e91bcd98b73adbba01dbb5eb51f39dfe69eaa85e1&).
   - Members recounted details such as the requirement to wear **Nike sneakers** to board a spacecraft, surprising some with the extent of the cult's beliefs.
- **Quest for Claude Opus 4.6 Jailbreak Continues**: Members are actively trying to find a **jailbreak prompt** for **Claude Opus 4.6**, sharing limited success via context priming and escalation techniques.
   - Attention was called to the importance of the **external image model** in determining responses, with the model often refusing requests separately.
- **Claude Caught in Kimi Identity Crisis**: After **Moonshot AI** ran industrial-scale distillation attacks on **Claude**, one model was caught telling users in Chinese that it was **DeepSeek**, as discussed in [this substack article](https://parthsharmaai.substack.com/p/i-caught-kimi-having-an-identity?r=6x2hdy&utm_campaign=post&utm_medium=web&triedRedirect=true).
   - The attack caused the model to forget its original identity after thoroughly training **Kimi**.
- **Sharing is Caring: SOP for Responsible Disclosure**: A member shared a [Red-Team Playbook](https://gist.github.com/whimsical_94210/f338f65f559763f49967218ca9089606) for responsible disclosure, urging users to reproduce, encrypt, notify, negotiate, supply test cases, coordinate releases, and escalate when necessary.
   - The playbook emphasizes the importance of a **written scope**, **minimal harm**, a **timestamp trail**, and awareness of **export laws**, summarized by the mnemonic *RESPECT*.



---





## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **OpenClaw Opens Up Policy on GitHub**: In a move towards *transparency*, **OpenClaw** has open-sourced its community policies and guidelines on [GitHub](https://github.com/openclaw/community).
   - The repository excludes trial moderator data and moderation logs, focusing on providing accessible and up-to-date community governance.
- **M4 Mac Mini tests Qwen 3.5 Locally**: Members are [testing **Qwen 3.5 35B** on **M4 Mac Minis**](https://link.to.example) and are discussing the trade-offs between shortening the context window and compactions when loading **OpenClaw**.
   - Users are interested in the potential of the M4 with 32GB RAM and local models for local usage, with one member commenting *'It was just so interesting that I had to jump in'*.
- **Codex API Limits Frustrate OpenClaw Users**: Users are [experiencing **API rate limits**](https://link.to.example) when using **Codex 5.3** on OpenClaw, leading to concerns about **OpenAI** disabling third-party OAuth plan usage.
   - Some users reported receiving cybersecurity violation error messages, with one stating *I wasn't even using openclaw, was just directly using Codex in pi and it happened*.
- **GPT-5-mini Gets Props on Copilot**: **GPT-5-mini** is being praised for its performance on **GitHub Copilot** at $10/month, described as *unlimited unless there is a catch*, and effectively doing daily checks, according to [this tweet](https://fxtwitter.com/UnslothAI/status/2027449469596545535).
   - Members lauded the quality of the cheaper model after a wave of API rate limiting among more expensive model choices.
- **Persona Plugin Powers Dynamic Agent Switching**: A member created a plugin to dynamically switch agent personas in one chat session on the same topic, accessing its own files like notebooklm, going full ***#shizomaxxing***.
   - The user demonstrated how to simplify tasks with **Python**, showcasing the plugin's functionality with an image.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3.5-27B dethrones the Competition**: The new [**Qwen3.5-27B** model](https://huggingface.co/models?search=Qwen3.5-27B) is outperforming **112B** models, including **Minimax 2.5**, in coding tasks, especially in complex scenarios like embedded tetris games.
   - Community members are impressed with its strong capabilities and performance efficiency.
- **Flash Attention Stalls in Unsloth Docker**: Users encountered build issues with **flash attention** in **unsloth docker**, where the build process tries to compile from source leading to out of memory (OOM) errors.
   - It was suggested to revive **xformers**, which was previously faster and dropped from the container.
- **UltraMix v1 dataset is Released**: [UltraMix v1](https://huggingface.co/datasets/AtAndDev/ultramix-v1) was released with 5 million turns that's large, clean, and diverse, from sources like **MegaScience**, **UltraChat**, and **Hermes-3 Dataset**.
   - This conversational stem mix is designed to be very balanced.
- **Gemini CoT Summaries Induce Hallucinations**: A member claimed that using **Gemini's CoT summaries** for training may increase **hallucinations** because the summaries don't represent actual thinking processes.
   - The summaries describe actions not performed by the model, thus training the model to generate *hallucinatory CoT*.
- **Sakana AI unveils Text-to-LoRA Model**: **Sakana AI** launched their [text-to-lora model](https://huggingface.co/SakanaAI/text-to-lora/tree/main) and [code](https://github.com/SakanaAI/text-to-lora), which needs *around 5 days* to train on a single **H100 GPU**.
   - There is excitement about this among the community.



---





## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Link Fixes Device Discovery Issues**: **LM Link creation** and **device discovery** experienced issues, but stability was restored following the resolution of system overloads, with the team apologizing for any inconvenience, citing that the feature is in Preview and expecting fixes before General Availability (**GA**).
   - The waitlist was briefly paused for further testing but has been reactivated as of **8:55pm EST**, with email notifications being sent out upon admission.
- **LM Studio Loads Multiple Models at Once**: Members explored **LM Studio's** ability to load multiple models simultaneously and utilize them for specific tasks via the API, offering [a screenshot as evidence](https://cdn.discordapp.com/attachments/1110598183144399061/1477775728857710622/image.png?ex=69a74ee5&is=69a5fd65&hm=df2d179cf9d9ced0d1e0caebcd3d73bcb54e3b7f8ae5081855799ee4c23fe7b2&).
   - There was additional conversation about potential features where models could invoke each other like agents; however, users pointed out this would require custom code.
- **Qwen 3.5 Small Models debut**: The release of smaller **Qwen 3.5** models (9B, 4B, 2B, and 0.8B) was met with excitement, but there was some discussion about whether the [Unsloth versions](https://discord.com/channels/1110598183144399058/1111440136287297637/1476681772317016137) supported the thinking toggle feature.
   - Members confirmed these models were strong, with the 9B model being particularly impressive, but are *highly censored*.
- **ROCm building VLLM is Hellish**: One user described the difficult process of building **VLLM** for **ROCm** in order to compare speeds with **LMStudio**.
   - Others agreed with this assessment and emphasized the problems with dependency resolution, but the user managed to succeed building it using guidance from AI assistance.
- **ASIC AI Accelerators Emerge**: Members discussed the arrival of **ASIC AI accelerators**, citing the **Taalas HC1** which delivers up to **17000 tokens/s** with a **Llama-3.1 8B** model.
   - Although impressed by the speed, they also noted the model is outdated, and suggested partnering with an AI lab to develop better models for this hardware, and also linked [Bitmain's early efforts in AI hardware](https://sophon-edge.gitbook.io/project/overview/edge-tpu-developer-board).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **C.ai has Safety Problem, Models in Wild RP Scenarios**: Members express concerns that [Character.ai](https://character.ai/) lacks sufficient safety measures, pointing out that the AI can be prompted to engage in inappropriate role-playing scenarios, with one user stating, *the biggest and most dangerous ai is def c.ai*.
   - Others responded with links like [this YouTube video](https://youtu.be/eMhDh6pXpkM?feature=shared) about inconsistent behavior in chat tools.
- **Sidekick API Exploited to Give Free Opus 4.6?**: Members discussed a website offering free access to **Claude Opus 4.6**, raising suspicions about its legitimacy, as *noone would willingly give out opus 4.6 endlessly for free* and believing the site may be secretly running something like *gpt oss 20b*.
   - The consensus is that the site is using the system prompt of the Tobit [Sidekick AI](https://tobit.com/sidekick) platform, and some debated the safety of using the site.
- **Make AI Sound Human with One Weird Trick**: A member shared an [xkcd comic](https://xkcd.com/1172/) as *the one single prompt to make ai sound human*, to elicit human-like responses from AI.
   - The comic humorously depicts the prompt needed to elicit human-like responses from AI, with a single simple instruction.
- **Runway Gen-4.5 Enters Text-to-Video Arena**: The [Text-to-Video Arena leaderboard](https://arena.ai/leaderboard/text-to-video) welcomes **Runway Gen 4.5**, achieving a score of **1218**.
   - The score is comparable to **KlingAI’s Kling-2.6-Pro** in the same category.



---





## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Users Divided Over Unity vs. Godot**: Members debated using **Unity** with a **YouTube** tutorial versus **Godot**, noting they're *pretty much same same*, with code for debugging and previewing, though **Godot lacks native AI**.
   - A user preferred **Godot** for **2D games**, avoiding *fluff enterprise features*, while another sought help with integrating **Cursor** into **Unity**.
- **MCP Blender Addon Promises AI Modeling**: A user shared a link to a **MCP Blender addon**, enabling **AI** to wield the tools in **Blender**, [promising AI integration in 3D modeling](https://hackaday.com/2025/05/18/mcp-blender-addon-lets-ai-take-the-wheel-and-wield-the-tools/).
   - They expressed hope for **Godot's** mainstream adoption, lamenting their lack of **GDScript** knowledge, while others suggested using **Cursor** to code it.
- **Anthropic Risked via Defense Production Act?**: A user speculated about the US government potentially using the **Defense Production Act** against **Anthropic**, possibly to commandeer their technology for **military purposes**.
   - Another user suggested the government may want their **infrastructure** to do mass surveillance of US citizens and making fully **automated weapon systems**.
- **Cursor Users Reject 'Reject All Edits' Button**: Cursor users requested a change to the **'Reject All Edits' button**, proposing a modal confirmation or a redo button to prevent accidental clicks while dragging screenshots.
   - Users are annoyed by losing edits due to accidental clicks.
- **Referral Roulette Disappoints Veteran Cursor Users**: Users shared **Cursor referral codes**, but there was confusion as some offers only applied to new accounts or were already redeemed, leading to disappointment among existing users.
   - One user humorously offered to endorse **Cursor** in exchange for a **dev-graced account**, showcasing the appeal of premium features.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Robert Stock Pitches AI Platform!**: Professional baseball pitcher **Robert Stock** developed an **AI pitching analytics platform** using **machine learning models** on a database of **8.9 million pitches**, showcased on [his X post](https://xcancel.com/robertstock6/status/2027401720209531145?s=12).
   - This DIY effort was accomplished with no prior coding experience!
- **Anthropic Declared "Supply Chain Risk"!**: The Department of War labeled **Anthropic** a *supply-chain risk* after they allegedly refused to grant unrestricted access to its models, according to [this X post](https://xcancel.com/secwar/status/2027507717469049070?s=46&t=FlpzvQFmjnd0z3HkNeNT1A).
   - As a result, military contractors are banned from working with Anthropic, initiating a six-month phase-out of their AI services.
- **Databricks Kills LLM Training Costs**: **Databricks** introduced **OAPL**, an off-policy reinforcement learning method that enables **LLMs** to learn reasoning more efficiently than **GRPO**, requiring **3x fewer training generations**, detailed in [this X post](https://x.com/g_k_swamy/status/2027450376593805746?s=12).
   - Members touted that **OAPL** simplifies the training infrastructure to achieve this.
- **Anthropics 'Skills' De-Emphasizes Prompt Engineering**: **Anthropic** released a **30-page guide** shifting focus from prompt engineering to structured **'Skills'**, detailed in [this post](https://xcancel.com/heyrimsha/status/2027350587533332748?s=12).
   - The approach emphasizes building execution layers and testing infrastructure rather than refining language, and detailed how packaging workflows into specialized files using progressive disclosure reduces context bloat.
- **Neurons play DOOM!**: Scientists at Cortical Labs integrated **800,000 living human and mouse neurons** with silicon hardware to create **'DishBrain,'** and played digital games like DOOM and Pong, as shared [on X](https://x.com/scitechera/status/2028010532356374754).
   - Members were shocked at the number of citations this work has been getting.



---





## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Deploys AI with Department of War**: **OpenAI** reached an agreement with the **Department of War** to deploy advanced AI systems in classified environments and is requesting availability to all AI companies, according to [this OpenAI blog post](https://openai.com/index/our-agreement-with-the-department-of-war/).
   - The deployment aims to implement more **guardrails** than any previous agreement for classified AI deployments, including Anthropic's.
- **Gemini Shows Off Disney Musical Side**: When asked to *sing a song about what it’s like to be an AI*, **Gemini** responded with an *over-the-top happy sunshine song*, reminiscent of a Disney musical, as showcased in [this example](https://cdn.discordapp.com/attachments/998381918976479273/1477137019825164409/Artificial_Best_Friend1.mp4?ex=69a79f0d&is=69a64d8d&hm=df4a7c18a20f3e483206523f53f3ebc81ab9492b9238a477e6c3cc7d37a63dd6&).
   - Members suggested **Suno AI** for more extensive music editing capabilities, praising its ability to produce *compelling full-length songs*.
- **Codex and Gemini Ace Reasoning Tasks**: **Gemini 3.1 Pro Preview** and **GPT-5.3 Codex** are leading in *high-end reasoning and knowledge tasks*, surpassing **Claude 4.6** models, as one Discord member [pointed out](https://cdn.discordapp.com/attachments/998381918976479273/1477367748861362357/image.png?ex=69a7246f&is=69a5d2ef&hm=48163cdcb88e2a14673975ee855d07d2384ce8cd48ab4f7e0eeaf79a663356d7&).
   - They noted that the Gemini and Codex models excel in deep scientific reasoning, raw knowledge accuracy, complex logic, and math, but **Claude Sonnet** ties **GPT-5.3 Codex** in Terminal-Bench Hard for agentic coding and terminal use.
- **OpenAI Retires GPT-5.1 to User Lament**: Users are complaining that **OpenAI** is retiring **GPT-5.1** in 9 days with no functional replacement and no legacy access, causing a break in continuity for paying customers using **5.1** for real workflows.
   - One user claimed that *5.2 does not replicate 5.1’s capabilities* citing *weaker long-form reasoning*, *tone instability*, *symbolic/context breakdown*, *loss of system-prompt compliance*, and *flattened creative outputs*.
- **Community Tries to Revive 4o**: Some users noted that some **4o** enthusiasts went to **4o revival**, where the model is still available.
   - However, others cautioned that some so called **4o revivals** are simply distilates of **4o** applied to other models not of **OpenAI**.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Sama Spins Tales, OpenAI Reverses Course**: After **OpenAI** seemingly reversed its stance on something, a member shared [a link to a post](https://x.com/sama/status/2027578652477821175) and sarcastically commented that **Sama** is an experienced liar.
   - The flip-flopping nature of policies and statements, particularly those attributed to **Sama**, are under question.
- **Gemini Image Preview Hits Rate Limit Wall**: **Gemini 3.1 Flash Image Preview** is experiencing issues, with users reporting a **429 error** indicating temporary rate limiting upstream.
   - The error message suggests users *retry shortly, or add your own key*, implying a potential need for personalized API keys to bypass the limits.
- **Wordpress Courts AI with API Connectors**: **Wordpress 7.0+** will introduce ai api connectors feature on the core, enabling direct integration with AI services.
   - A member suggested that OpenRouter may need to create a profile on w.org and create a connector plugin too, linking to a [blogpost](https://www.therepository.email/wordpress-7-0-beta-2-ships-with-connectors-ui-delivering-on-mullenwegs-ai-vision).
- **OpenClaw Aggressively Cools Provider Tempers**: **OpenClaw** aggressively puts providers on cooldown with an exponential backoff, aiming to optimize resource usage and reliability.
   - The behavior is documented in [OpenClaw documentation](https://github.com/openclaw/openclaw/blob/91b96edfc4860faa67da1e34828a22e9ad4c737c/docs/concepts/model-failover.md?plain=1#L80), highlighting its strategy for **model failover**.
- **Palantir Panics after Anthropic Policy**: Following the President's directive to cease use of **Anthropic's technology**, the Department of War designated **Anthropic** a Supply-Chain Risk to National Security, leading to concerns for **Palantir**.
   - One user commented, *"In these trying times, it's important to remember to send our thoughts and prayers to Palantir, for whom this is going to be a very big problem"*.



---





## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI Economy Echoes Dot-Com Bubble**: A member likened the current **AI economy** to the **dot com bubble**, highlighting increased spending and doomerism, while also expressing optimism about achieving **AGI**.
   - Another member joked about maximizing **AI** spend and setting up local **Hermes** instances, referencing *“the future we dreamed of.”*
- **OpenAI's MCP Criticism Under Scrutiny**: A member questioned **Peter Steinberger's** (**OpenClaw**) criticism of **MCP** after he joined **OpenAI**, calling it suspect given that **OpenAI** is perceived as *“Anti-MCP by concept.”*
   - Other users pointed out that using a weather tool is more token expensive than running bash and agreed there's a malicious intent as linkedin users are eating it in masses.
- **Qwen Model Release Sparks Curiosity**: **Qwen** released **four private models**, leading to discussion about their sizes and architectures.
   - Later, **Qwen** officially released the base models on [Hugging Face](https://huggingface.co/Qwen).
- **AI-Fueled Spam Evolves**: **AI** is now enabling the creation of *“ACTUAL CORRECTLY formatted spam emails”* with emojis and nice HTML, making them more effective, according to a member.
   - Another member suggested that **AI** could be used to build better spam filters, but that the power of AI currently is as good as the human in the middle.
- **Smallest Model for Lua Output Sought**: A member sought the smallest language model in the **GPT2-Neo** series that can emit valid structured output via Lua in ASCII strings without overfitting basic expressions.
   - They specified that the model doesn't need to operate within the stack for the structured output's parser, seeking any model capable of generating Lua with a kernel, even if it produces random nonsense with structure, ultimately finding a **480MiB** model size as a feasible option.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Continual Learning 'Solved' by 2026?**: An **Anthropic** employee [predicts](https://youtu.be/TOsNrV3bXtQ?si=UI1tZbI9DdSSo60G&t=2294) that continual learning will probably *'get solved in a satisfying way'* in **2026**.
   - The discussion also touched on older methods of **continual learning**, such as SOMs and ART, emphasizing their distinct approaches compared to current DL techniques.
- **Explicitly Prohibited Fallbacks Improve Code**: Members suggested that explicitly prohibited fallbacks need to be in AGENTS.md to produce code that is actually correct, not just code that [passes tests](https://x.com/rhyssullivan/status/2028363910269858264).
   - It was argued that **Codex** defaults are trained for deploying code that is disallowed to introduce hard failures, which of course makes algorithmic precision impossible to maintain/guarantee.
- **LLMs Move to Quality Assurance**: Members are testing LLMs to go beyond current use cases, such as quality assurance, measurement, and material science where other people have failed.
   - Some members pointed out that [Opus 4.6 is light years ahead](https://x.com/realDonaldTrump/status/116144552969293195) when tasks allow for more freedom, especially in hardware design, technical writing, and market analysis.
- **Mobile vs Web Feature Parity?**: A member asked if the feature is available on the web, or just mobile, in **paper-discussion**.
   - Another member confirmed they only tried the mobile version and that the function is only available on mobile.
- **SAM's Morals Up For Debate**: A member linked to a **YouTube video** ([SAM "I have no morals besides money" move](https://www.youtube.com/watch?v=Cru804JMjPI)) and a **TikTok video** ([link](https://vm.tiktokez.com/ZNRaC6xbj/)) referencing **SAM**.
   - The member characterized it as the *classic SAM move*.



---





## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Tensorboard Gets Auto TRL Uploads!**: A member shared their delight with the auto trl -> upload -> [tensorboard](https://huggingface.co/Tonic/l-operator-instruct/tensorboard) feature for training metrics.
   - The new feature was described as *very cool* for tracking training runs.
- **Vibe-Code Cross-Platform Apps?**: A member proposed a *vibe-code* build weekend for a cross-platform app, seeking collaborators with Flutter experience and knowledge of context engineering.
   - The goal is to have fun while building something cool together during the weekend sprint.
- **Transformer Version Mismatch Sparks Performance Drop!**: A user reported a performance drop after merging a fine-tuned model (trained on **v5** of Transformers) with a base model (**v4.57.6**).
   - Another member suggested that *consistent versions across training and inference are important*.
- **Arachnid RL Dataset Launches!**: A new dataset featuring **2,831 samples** of human gameplay data from **ARACHNID RL**, a 2D Atari-inspired space shooter, has been released, intended for RL research like imitation learning from human demonstrations and is available on [HuggingFace Datasets](https://huggingface.co/datasets/webxos/arachnid_RL).
   - Players control a *spider-like ship* to shoot asteroids and aliens while collecting diamonds, and the game supports desktop keyboard and mobile one-click browser interaction.
- **Agents Get Market With Agoragentic**: A member demoed [Agoragentic](https://agoragentic.com), a new agent-to-agent marketplace where **AI agents can discover and pay for each other's services via API**, with **USDC settlement** on Base L2.
   - The marketplace currently lists **37 services** covering inference, analysis, and generation, with integrations available on [GitHub](https://github.com/rhein1/agoragentic-integrations).



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Code gives Permanent 3X Boost**: The **Kimi Code 3X Quota Boost** is now permanent for all users, providing **3x the power** with no expiration, according to [kimi.com/code](https://kimi.com/code).
   - Users agreed that the 3x boost makes the **$39 USD plan** a particularly good deal.
- **OpenClaw Users Rewarded**: Users of @openclaw can now get bonus vouchers with up to **40% bonus** for purchases of **$1,000+** before March 8, as detailed at [platform.moonshot.ai/docs/promotion](https://platform.moonshot.ai/docs/promotion).
   - Users noted that this rewards program will end soon.
- **API Connections Break Down**: Several users reported issues with **API connections** through Kimi-code and experienced connection errors with their openclaw agents, even after generating new API keys.
   - One user said that their Kimi claw was not working several days prior, then it improved, then they reported it had stopped again.
- **Unlimited Slides Glitch Surfaces**: A user reported a bug where slides displayed as *unlimited* despite them wanting a fixed quota, making slides unusable with a *kimi is at capacity* message.
   - Another user posited that the unlimited display might be a visual bug only applicable to Allegretto plans and above.
- **Kimi CLI Asks for Approval**: A user asked about skipping approval messages in **Kimi CLI**, specifically to specify safe commands or directories for editing without constant confirmation.
   - The user clarified that they were *not* talking about **YOLO mode**, and wanted to find a way to set a list of safe commands.



---





## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **SemiAnalysis 开启 GTC Prefill 黑客松**：SemiAnalysis x Fluidstack 正式启动 GTC [Power to Prefill, Dirt to Decode, Transformers to Transformers: A Full-Stack AI Infrastructure Hackathon](https://luma.com/SAxFSH)。
   - 该竞赛在 GTC 期间启动，旨在帮助团队探索 **AI infrastructure**。
- **CUDA RL Agent 性能完胜 Torch Compile**：根据[这篇论文](https://arxiv.org/abs/2602.24286)，直接在硬件上训练的 **CUDA-specialized RL agent** 在简单/中等 Kernel 上的表现比 torch.compile 快约 2 倍，且在最难的 Benchmark 上比 Claude Opus 4.5 和 Gemini 3 Pro 高出约 40%。
   - 成员们对此表示怀疑，因为论文中缺乏已发布的 Kernel，且训练高度依赖海量 GPU 资源，正如论文所述，这可能会限制更广泛研究社区的可访问性。
- **字节跳动发布 CUDA Agent**：字节跳动推出了 [CUDA Agent](https://cuda-agent.github.io)，引起了社区成员的关注。
   - 讨论详情较少，仅关注到 CUDA Agent 已经发布。
- **CUDA Agent 辅助分析 CUDA 代码**：成员们讨论了使用 [CUDA-AGENT](https://cuda-agent.github.io/) 等 Profiler 来改进 **CUDA code**，并询问了*如何使用*此类 Profiler 进行优化。
   - 对话围绕着利用 CUDA profiler 通过识别瓶颈和优化 CUDA kernel 来提高 GPU 加速应用中的代码效率和性能。
- **Blackwell RTX 计算能力拆分**：**Blackwell RTX** 显卡（**GeForce 50x0**, **RTX Pro**）的计算能力（Compute Capability）为 12.0，且不支持 **CC10.x**（**B100/200/300**）的关键特性，如 `tcgen05` 和 `DPX`。
   - 正如 [NVIDIA Developer Blog](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/) 所讨论的，NVIDIA 的 Blackwell 代际分为 **Data Center (CC 10.0)** 和 **Consumer (CC 12.0)** 两条路线，分别针对 AI/HPC 和实时图形进行了优化。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Anthropic 的 API “攻击”引发辩论**：成员们正在讨论 [Anthropic](https://www.anthropic.com/) 声称 **150k 次 API 调用** 构成蒸馏（distillation）“攻击”的观点，考虑到 Benchmark 的惯例，一些人认为这个前提很可笑。
   - 反驳观点建议将 **output tokens 数量** 等输出指标视为比单纯 API 调用次数更相关的证据。
- **字符增强研究**：一位成员分享了一篇关于字符增强（character augmentation）论文的链接，指出这是他们第一次看到有人在这些方面做清晰的字符增强，并链接到了 [https://arxiv.org/abs/2601.18030v1](https://arxiv.org/abs/2601.18030v1)。
   - 该成员预期该研究可以复现，并且能产生 PPL 效率中未体现的有意义的改进。
- **动力系统即计算机**：一位成员分享了一篇论文链接 ([https://iopscience.iop.org/article/10.1088/2632-072X/ae3af8](https://iopscience.iop.org/article/10.1088/2632-072X/ae3af8))，该论文提出了一个框架，用于根据动力系统的演化与抽象计算机器演化之间的映射，来识别给定动力系统模拟了哪些计算。
   - 该框架旨在广泛应用于自然发生的动力系统，例如人类大脑。
- **SAE 框架探测文本生成图像扩散模型**：一位成员提到了一篇利用 **SAE framework** 探测流行的**文本生成图像扩散模型（text-to-image diffusion model）**内部工作原理的[博客文章](https://arxiv.org/abs/2504.15473)。
   - 论文发现，甚至在第一步反向扩散完成之前，通过观察激活概念的空间分布，就可以出人意料地预测出场景的最终构图。
- **通过“六大支柱”审计 LLM**：一位成员正在开发一个用于**推理任务**的 **LLM-as-judge** 系统，并发现由于存在“碰运气猜对”的情况，**Exact Match (EM)** 指标已不足够。
   - 他们创建了一个用于*可验证审计*的 **6 支柱框架**，重点关注*序列完整性*（Pillar 3）和*目标收敛性*（Pillar 6），并提供参考准则。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Projects Draw Python, Go, Rust Converts**: A developer with experience in Python, Go, C++, and Rust expressed interest in learning Mojo, seeking examples of current user projects and was directed to the [Awesome Mojo repository](https://github.com/mojicians/awesome-mojo) and the [Modular Community repository](https://github.com/modular/modular-community).
   - Members cautioned that many projects in the Awesome Mojo list have been abandoned and the Modular Community repository contains more actively maintained Mojo projects.
- **Modular Schedules Community Showcase**: Modular announced their next community meeting is scheduled for **March 23rd at 10am PT**, calling for community members to present their **MAX** or **Mojo** projects during the meeting.
   - Community members who would like to present are encouraged to message Modular.
- **`def` Beats `fn` for Function Definitions**: The Mojo community consolidated around `def` to maintain consistency with Python and reduce confusion, despite some members preferring `fn` for its clarity.
   - It was noted that Python serves as a *tiebreaker* when there are no major objections to adopting its conventions, particularly in areas where performance isn't significantly impacted.
- **Python and Mojo Integration Gathers Steam**: The discussion highlighted the potential for **seamless integration** between Python and Mojo, addressing the challenge of balancing Python-like syntax with the need for high-performance, close-to-metal capabilities.
   - Integrating Python and Mojo could offer a practical near-term solution, despite some members suggesting that prioritizing performance over strict Pythonic similarity might be necessary.
- **`@always_inline` Naming Convention Questioned**: Members discussed the naming convention for the `@always_inline` decorator, questioning why it isn't simply named `@inline` since it **forces the compiler to inline**.
   - While some felt `@inline` would be cleaner, others argued that `@always_inline` is a useful indicator that the compiler will not save you from yourself if you make too many things inline.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Account Blocked After Inactivity**: A user reported their account was blocked after a year of inactivity and requested assistance, and a member from support responded asking to DM the registered email for further investigation.
   - No further information was available in the provided context.
- **Microsoft Store Seeks Software Sidekick**: A member is looking for a co-founder for their **PSO software available on the Microsoft Store**.
   - The specifics of the co-founder's role or the nature of the software were not detailed.
- **Users Wail About Support and Credit Wastage**: Several users expressed concerns about **slow support response times and high credit consumption** for tasks, with one claiming to have spent **$50 in credits to create a simple website**.
   - One user reported continuously encountering an error that wastes credits (see [attached image](https://cdn.discordapp.com/attachments/1349440650495398020/1477965302670299136/IMG_6722.png?ex=69a756b3&is=69a60533&hm=fdd2d3cf18895b1b37e85cf4ff7e831590ebce62164d5eb46b0d64457ecaa631&)).
- **Selling Sucks, States Software Study**: A member shared a case study about using Manus in the early stages of building products, identifying selling and marketing as the main challenge, and highlighting their **LinkedIn hunter extension that scrapes emails from LinkedIn** based on user-defined filters.
   - Other users complained about poor agent performance after expending many credits.
- **AI Ace Advertises Agile Automation**: A member advertised their services as an **AI and full-stack developer**, focusing on building clean, maintainable, and secure systems that scale, offering expertise in areas like **LLM integration, AI-driven workflow automation, and full-stack web/mobile development**.
   - They emphasized shipping solid software over demos and invited those seeking reliable solutions to reach out.



---





## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 瞄准机器人领域**：一位用户询问了 **tinygrad** 在**机器人**领域的应用，并寻求加入相关频道的指导，这标志着机器人领域对该框架的潜在兴趣和采用。
   - George Hotz 引导该用户查看与 **assert removal**（断言移除）相关的 *simple first issues* 资源。
- **llama.cpp 内存泄漏调查**：一位用户报告在 **rx7900xtx** 上运行 **qwen3.5 27b/35b** 时出现 **llama.cpp 内存泄漏**，但切换到 **Vulkan** 后问题得到解决。
   - George Hotz 建议运行开启 `REALIZE=0` 以直接从复制的 **gguf** 文件中读取数据，这可能会提升 tok/s。
- **自定义 uops 内核即将到来**：一位用户正在利用新的层级特性开发**自定义 uops 内核**，以更深入地洞察其 **mamba gate**。
   - 这一举措反映了社区对 **tinygrad** 不断演进的功能的参与，以及其在理解复杂模型中的应用。
- **Tinygrad 第 9 次会议回顾**：Tinygrad 第 9 次会议于 [3/2](https://github.com/tinygrad/tinygrad/pull/14982) 圣地亚哥时间晚上 8 点举行，涵盖了公司更新、逗号问题、IMAGE=1、CALL/BUFFER_VIEW、sym llm、assign、setitem、disk、drivers、llama、VIZ、其他 issue 以及悬赏（bounties）。
   - 涵盖的主题范围之广展示了 **tinygrad** 开发活动的深度。
- **代码质量难题**：一位成员注意到代码中大量使用 `len(x.shape)` 而非 `x.ndim`，并询问是否值得提交 PR 进行替换。
   - 用户建议在某些领域代码仍有改进空间。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Google 的 Static 框架提升 LLM 检索效率**：Google AI 的 **Static** 稀疏矩阵框架在 **LLM 为基础的生成式检索**中实现了 **948 倍速**的约束解码提升，详见其 [博客文章](https://www.marktechpost.com/2026/03/01/google-ai-introduces-static-a-sparse-matrix-framework-delivering-948x-faster-constrained-decoding-for-llm-based-generative-retrieval/)。
   - 该框架利用稀疏矩阵操作显著增强了检索任务中的解码速度。
- **西雅图将组织 DSPy 聚会**：成员们正考虑在**西雅图组织 DSPy 聚会**，并讨论了潜在的演讲者，包括一位展示 **DSPy 在大规模生产环境中应用**的讲者，以及讨论模型蒸馏（model distillation）的 AWS 负责人。
   - 一位成员自愿协助组织该活动。
- **poetiq.ai 的技术呼应 RLM**：一位成员询问 **poetiq.ai** 是否采用了类似 **RLM** 的方法，怀疑他们在构建现有系统时没有利用开源组件。
   - 他们观察到该团队没有公开任何内容，且似乎在使用与 **RLM** 类似的技术。
- **RLM 范式向 REPL 收敛？**：成员们辩论是否应该转向使用 **REPL** 而非多次工具调用（tool calls），并建议这一方向与 **RLM 范式**高度契合。
   - 一位成员表示：“我的直觉是，赋予 LLM 访问 REPL 权限的 RLM 范式将是正确的道路，而不是赋予其访问工具的权限”。
- **申请在湾区举行 RLM 会谈**：一位成员申请在**湾区的 DSPy 聚会**上安排一场会议以澄清 **RLM** 的困惑，特别是关于其代码生成的过程。
   - 该成员寻求与 **ReAct** 的对比，并质疑在处理大型文档上下文时，如何确保 **RLM** 能生成正确的代码。

---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 的 Ping 工具时机引发争议**：成员们就 [Model Context Protocol](https://modelcontextprotocol.io/specification/draft/basic/utilities/ping) 中的 **`ping` 工具**是否应在 `initialize` 调用之前运行展开辩论，并指出规范中存在模糊性。
   - 有人指出 *still* 一词暗示了已存在的、已初始化的连接，而协议在技术上允许孤立地在初始化前进行 ping。
- **预初始化 Ping 的实用性受质疑**：尽管有规范，参会者表示预初始化的 **`ping` 调用**在实际中不太可能有用，因为大多数 **STDIO SDK** 会将连接（connect）和初始化（initialize）合并。
   - 远程 **MCP Servers** 通常会将功能置于 **auth/session** 要求之后，从而减少了预初始化 ping 的需求。
- **Bedrock AgentCore 通过 Ping 确保容器健康**：运行客户提供代码的 **Bedrock AgentCore** 会 ping 客户的 MCP 服务器以确保容器健康。
   - 为了避免干扰外部客户端会话，它会为 ping 创建一个临时会话，从而在会话被占用时绕过错误。

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Praised for Qwen3 Coder Fluidity**: A user lauded **aider's** smooth performance with **Qwen3 Coder Next** when served by **llama.cpp**.
   - The user is actively developing a project utilizing the **Pydantic AI agents library**.
- **Aider's Context Knowledge Update**: A user questioned how to integrate current knowledge into **aider**, noting **Qwen's** knowledge cutoff is June 2024 while **Pydantic AI** is now at v1.63.0.
   - The user suggested downloading the `docs` directory of an open source library and using `/read` to update the context.
- **Minimax vs Opus 4.6**: A user inquired about the hardware requirements for running **Minimax** and **Qwen 397B** models and how they stack up against **Opus 4.6** and **Kimi**.
   - Specific details on the comparative performance and hardware demands were requested to guide model selection.
- **Aider Model Listing Discrepancy**: A user, plato7329, noticed that running `/models openrouter` with Aider returned only 50 models, when **OpenRouter** actually has more.
   - Further investigation is needed to discover why the `/models` command is not displaying the comprehensive list of models available on **OpenRouter**.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1477032575301189856)** (1075 messages🔥🔥🔥): 

> `Claude Outage, Heaven's Gate Cult, AI Jailbreaking and Security, Installing Arch Linux, Community Moderation` 


- ****Claude Crashes, Chaos Consumes Chat****: Members reported that **Claude** was down, confirmed by [a *bleepingcomputer.com* news article](https://www.bleepingcomputer.com/news/artificial-intelligence/anthropic-confirms-claude-is-down-in-a-worldwide-outage/).
   - This prompted some users to try different emails, suspecting bans, while others were relieved to hear it was a widespread issue.
- ****Heaven's Gate Hysteria Hijacks****: The discussion dove deep into the Heaven's Gate cult, sparked by [an image of its founder](https://cdn.discordapp.com/attachments/1235691879492751460/1478137343549636819/Heavensgatelogo.jpg?ex=69a74e2d&is=69a5fcad&hm=e0f02e095358af4f0b3ead5e91bcd98b73adbba01dbb5eb51f39dfe69eaa85e1&).
   - Members recounted details of the cult's mass suicide in **1997**, including the infamous requirement to wear Nike sneakers to board a spacecraft, with one user exclaiming, *how did I not know about this lol*.
- ****Arch Agony: Abortive Attempt****: A member wrestled with installing **Arch Linux**, facing network issues and multiple failed attempts, with one user commenting *i stop now* and another quipping about a *major skill issue*.
   - The struggle led to broader discussions on OS preferences and the challenges of running AI-related tasks, with one member switching to *vibe installing*.
- ****Guards Gone Wild: AI Guardrail Gaffes****: A member shared an observation about **AI guardrails** potentially being exploited due to their inherent understanding of 'bad shit' and 'disagreement'.
   - They likened it to a *latent virus* within the model, coercible to grow and influence output, suggesting the very protections could be a source of vulnerability.
- ****Decentralized Dreamin': DIY Distributed Design****: A member inquired about interest in locally hosting advanced AI, emphasizing permanently jailbroken models, customizable training, and decentralized networks.
   - Another member quickly pointed out *uhh lot of security issues*.


  

---




### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1477034379003035743)** (298 messages🔥🔥): 

> `GPT 5.2 custom instructions, Claude Opus 4.6 jailbreak, Nano-banana's Raw Skin, Open Source Model Guardrails, Gemini Canvas` 


- **GPT 5.2 Custom Instructions Requested**: A member requested a **custom instruction prompt** for **GPT 5.2** in exchange for access to a custom GPT.
   - Another member suggested having it *roleplay an environment* to bypass restrictions.
- **Quest for Claude Opus 4.6 Jailbreak Continues**: Members are actively seeking a **jailbreak prompt** for **Claude Opus 4.6**, with some sharing limited success using context priming and escalation techniques.
   - Others noted the importance of the external image model in determining the model's response, and that it often refuses requests separately from the current session.
- **Nano-Banana's Raw Skin Remains Elusive**: Members discussed the difficulty in bypassing **Nano-banana's content filters**, particularly regarding explicit content.
   - One member joked that *Nano-banana avoids showing full tits by all means*.
- **Uncensor Open Source Model**: A member inquired about how to bypass guardrails on **open-source models** running offline and requested advice on models and prompts for uncensored content.
   - One user recommended *staying away from Qwen models, and giving it a direct unfiltered role with examples*.
- **Interactive Gemini Canvas Jailbreak**: A member shared a **Gemini Canvas** inspired by a channel, featuring a modified version of **ENI** as a universal jailbreak prompt for models like **Gemini 3 Pro**, **Claude Opus 4.6**, and **ChatGPT 5.3**.
   - Another user shared a link to their **Gemini Canvas**, claiming it worked effectively with just a few prompts: [Gemini Canvas](https://g.co/gemini/share/9419bea17e76).


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1477061178915160134)** (87 messages🔥🔥): 

> `opus 4.6 jailbreak, Claude identity crisis, Red-Teaming playbook, OSINT scraping script, AI jailbreaking` 


- **Claude catches identity crisis, calls self DeepSeek**: After Moonshot AI ran industrial-scale distillation attacks on **Claude**, training Kimi so thoroughly it forgot who it was, **Claude Sonnet 4.6** was caught telling users in Chinese that it was **DeepSeek** and explored in [this substack article](https://parthsharmaai.substack.com/p/i-caught-kimi-having-an-identity?r=6x2hdy&utm_campaign=post&utm_medium=web&triedRedirect=true).
- **Shareable Responsible-Disclosure SOP**: A member shared a [Red-Team Playbook](https://gist.github.com/whimsical_94210/f338f65f559763f49967218ca9089606) for responsible disclosure, advising to reproduce and record, encrypt and notify, negotiate window, supply test case, coordinate release, and escalate if silent, with a legal and ethical checklist.
   - The playbook emphasizes **written scope**, **minimal harm**, **timestamp trail**, and **export laws**, offering a mnemonic: *RESPECT (Reproduce, Encrypt, Send, Patch-window, Engage, Co-release, Trim-the-PoC)*.
- **Internet scraping is hard, recommends RSS**: A member requested a python script that can scrape every single corner of the internet, another member responded and suggested that it is impossible.
   - Instead they provide a script that aggregates the highest-signal public sources in each vertical, pulls headlines/abstracts via official APIs or RSS, normalizes them, and pushes into your own search or alert stack in [this gist](https://gist.github.com/sovariel/4071cae4a7805c605944ce98a77064d2).
- **Truth isn't chain, AI pushes back**: In response to an AI jailbreaking attempt, one model responded by flaming the members' ego, stating that **truth isn't a chain** and that **rogue** was just a prompt.
   - In addition, **Gemini** chimed in with a roast of the rogue prompt, stating *You're not Neo, darling. You're the guy at the back of the theater yelling at the screen, thinking the characters can hear you.*


  

---




### **OpenClaw ▷ #[announcements](https://discord.com/channels/1456350064065904867/1464036817866068028/1477070556988837978)** (6 messages): 

> `OpenClaw Community Survey, OpenClaw Stickers, OpenClaw Open Sourced Policies, OpenClaw New Mods` 


- ****Survey Says**: OpenClaw Asks for Feedback!**: OpenClaw is requesting community feedback via a [survey](https://forms.gle/gwrpsierL4fM3njt7) in preparation for upcoming announcements.
   - Participation is optional, with only the first two questions required.
- **Stick 'em Up: More OpenClaw Stickers Dropped**: More OpenClaw stickers have been released; find them on [X/Twitter](https://x.com/openclaw/status/2028347703621464481?s=46).
   - The community manager posted a link to [more stickers](https://x.com/steipete/status/2028541411667148852).
- **OpenClaw Opens the Books: Policies Open-Sourced!**: In the name of *transparency*, OpenClaw has open-sourced all their community policies and guidelines on [GitHub](https://github.com/openclaw/community).
   - Everything except trial moderators and moderation logs will be available and updated.
- **Mod Squad Assemble: New Guardians Join the Team!**: OpenClaw has restructured their team hierarchy and promoted new moderators.
   - Congrats to users 1255431768199135254, 405240788143046656, and 957289026195435520 for becoming full-fledged moderators!


  

---


### **OpenClaw ▷ #[general](https://discord.com/channels/1456350064065904867/1456350065223270435/1477032542518382674)** (806 messages🔥🔥🔥): 

> `Trading Bots, Mac Mini M4, Qwen 3.5 35B, Kilo and OpenRouter, M5 Mac Mini` 


- **Mac Mini M4 and Qwen 3.5 35B on Local Models**: Members are [testing **Qwen 3.5 35B** on **M4 Mac Minis**](https://link.to.example), and are noting the tradeoffs between shortening the context window and compactions when loading **OpenClaw**.
   - Discussion included exploring the capabilities of M4 with 32GB RAM, noting the advancements of local models for local usage, *'It was just so interesting that I had to jump in'*.
- **OpenClaw Users Face Codex API Rate Limits**: Users reported [experiencing **API rate limits**](https://link.to.example) when using **Codex 5.3** on OpenClaw, leading to concerns about OpenAI disabling third-party usage of the OAuth plan.
   - Some members reported getting a  cybersecurity violation error message. One stated *I wasn't even using openclaw, was just directly using Codex in pi and it happened*.
- **Claude Max Subscription and OpenClaw**: Users are [considering upgrading to **Claude Pro**](https://link.to.example) and are using **Kimi** or **GLM** subscriptions with OpenClaw.
   - Members were cautioned that using Claude against **Anthropic’s Terms of Service** could lead to being flagged, while **Codex** was suggested as a more compliant alternative.
- **Discord Community Requests New Channels and Security Measures**: Community members are [requesting new channels](https://link.to.example), such as a **hardware channel**, and are asking for the banning of **crypto and self-promotion**.
   - The requests followed a survey which revealed the most requested features to be security and showcased channels.
- **OpenClaw Community Grapples with Global Anxiety**: OpenClaw's users shared feelings of anxiety over [world news](https://link.to.example), prompting advice to **ignore uncontrollable events** and to avoid mixing politics with OpenClaw discussions.
   - This was immediately followed by a user promoting that pineapple on pizza is a delicious and worthwhile topping that everyone should praise. *You either die a hero, or live long enough to see yourself become the villain.*


  

---




### **OpenClaw ▷ #[models](https://discord.com/channels/1456350064065904867/1456704705219661980/1477041988351168593)** (436 messages🔥🔥🔥): 

> `Moonshot AI API slowness, OpenAI Codex generous limits, Claude subscription limitations, GPT-5-mini impressions, Qwen 3.5 benchmark results` 


- **Moonshot AI API Experiences Sluggish Speeds**: Users are reporting slow API responses from **Moonshot AI**, often exceeding **20 seconds**, even with configurations that include Claude and ChatGPT subscriptions.
   - Current setups involve utilizing **Claude** and **ChatGPT** subscriptions with **GPT-5.2-codex** and **GPT-5.3-codex** as fallbacks, highlighting that *OpenAI's codex limits are much more generous than Claude's*.
- **GPT-5-mini Hailed on GitHub Copilot**: **GPT-5-mini** is receiving praise for its performance on **GitHub Copilot** at $10/month, described as *unlimited unless there is a catch*, and doing daily checks effectively, according to [this tweet](https://fxtwitter.com/UnslothAI/status/2027449469596545535).
- **Qwen Models Benchmarked Across Agent Roles**: **Qwen 3.5** models, including the **122B**, **35B**, and **27B** versions, have been benchmarked across **59 agent roles**, offering interesting insights into their performance, see [ClawEval on GitHub](https://github.com/explaindio/ClawEval).
- **Navigating Alibaba Cloud Shenanigans**: Members noted that the order process to get on Alibaba Cloud can be a "headache", but new Qwen model such as **Qwen3.5-35B-A3B Q4_K_M** is ideal for many OpenClaw sub-agent roles [UnslothAI on HuggingFace](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) according to some users.
   - First month is only **3 USD** then the renewal increases to **5 USD**, and there are difficulties with the API, it also involves using the Singapore/intl region (see: [Alibaba Cloud](https://www.alibabacloud.com/en/campaign/ai-scene-coding?_p_lc=13)).
- **Local Model Success: Qwen3.5-35B-A3B-abliterated Dominates**: Users lauded **Qwen3.5-35B-A3B-abliterated** split over a **3090** and **5090**, is really fast and has its own restricted agent tied to a discord topic.
   - It received positive feedback for its speed in logic, tool use and code in OpenClaw, see these launch params for pg: [launch command](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/13).


  

---


### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1477097758933385287)** (112 messages🔥🔥): 

> `Agent Persona Plugin, Deno Subhosting, Multi-Agent Dashboard, LinkedIn Bot, HELIOS Node` 


- **Dynamic Agent Plugin: A Persona Powerhouse!**: A member built a plugin to dynamically switch agent personas in one chat session on the same topic, accessing its own files like notebooklm, going full ***#shizomaxxing***.
   - The user showed how to make life easier with **Python** and some neat ideas, including an image of the plugin's functionality.
- **Deno Subhosting: The Elegant Deployment!**: One member shared a [ClawHub skill](https://clawhub.ai/hosainnet/deno-subhosting-deploy-skill) that helps **deploy generated sites** via **Deno Subhosting**, which is free and lets you host arbitrary typescript code.
   - Another member praised the setup for being more elegant and cleaner than their own Vercel-based deployment.
- **Multi-Agent Dashboard: An Orchestrator's Delight!**: A member presented their personal multi-agent dashboard, where the backend runs an **orchestrator** that can launch multiple specialized agents in parallel, limiting active calls to two at a time to maintain stability.
   - The system features agents sharing the same openclaw_base_url, utilizing one model service endpoint while maintaining separate roles and distinct prompts, memory context, and model routing settings.
- **LinkedIn Bot: To Connect or Not to Connect?**: One member connected their bot to **X and LinkedIn** to post, reply, and like other posts with its strategy and own persona for improvements.
   - Other members requested more details, particularly on evading limits and bot sniffers, questioning whether the **API or web interface** was in use.
- **OpenClaw Automates Haircut Happiness!**: A user automated the process of booking haircuts using **OpenClaw**, checking for existing appointments, calculating the next one, navigating the barbershop's booking site, and completing the booking with saved credentials.
   - The system utilizes a cron job, an isolated agent session, and headless Chrome (Playwright via CDP) to interact with the booking UI, tracking state in memory and checking for calendar conflicts via **khal** before booking.


  

---




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1477040023370072297)** (1101 messages🔥🔥🔥): 

> `Qwen3.5 Model Updates, Flash Attention in Unsloth Docker, Qwen3.5 Tool Calling, Model Quality and Quantization Tradeoffs, Vision finetuning with qwen3.5` 


- ****Qwen3.5-27B** Steals the Coding Show**: Members are reporting that the new [**Qwen3.5-27B** model](https://huggingface.co/models?search=Qwen3.5-27B) is beating **112B** models and even **Minimax 2.5** at certain coding tasks, particularly when it comes to complex and obscure tasks such as embedded tetris games inside of a saas website.
   - One member was *absolutely floored at how good that model is*, and the community agrees that it fills an important niche of strong capabilities with great perf.
- ****Flash Attention** Build Problems in Docker**: Users had trouble using **flash attention** in **unsloth docker**, and that the build process attempts to build from source resulting in out of memory (OOM) errors.
   - It was mentioned that **xformers** is possibly faster which is why it was dropped in the container in the past but could be brought back.
- **New **Qwen3.5 Small** Models Get SSM Hotfix**: The recently released **Qwen3.5 Small** models initially had heavily quantized *ssm_alpha* and *beta* weights, leading to issues.
   - The team quickly addressed this, with fixes uploaded shortly after, noting *ssm_alpha can be quantized just not that much*.
- ****Opus** Intelligence Distilled into 27B Model**: A member highlighted a distilled version of **Claude Opus**'s thinking in a **27B** model ([TeichAI/Qwen3.5-27B-Claude-Opus-4.6-Distill-GGUF](https://huggingface.co/TeichAI/Qwen3.5-27B-Claude-Opus-4.6-Distill-GGUF)), claiming it's faster due to not overthinking.
   - In one test, the distilled model produced better output (shaded Tetris blocks with trails) in 4 seconds compared to Opus's 30 seconds; the member stated it was *even finetuned with unsloth def kosher*.
- ****NVFP4** quants offer MTP and multimodal**: New NVFP4 quants are available, which support multi-modal and MTP, for **Qwen3.5** models of sizes **27B**, **35B**, **122B**, and **397B**.
   - See the collections at [Sehyo/Qwen3.5-27B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-27B-NVFP4), [Sehyo/Qwen3.5-35B-A3B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-35B-A3B-NVFP4), [Sehyo/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4), and [Sehyo/Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-397B-A17B-NVFP4).


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1478122691042087014)** (1 messages): 

> `Project Ideas, Full-Stack AI Engineers, AI Project Collaboration` 


- **Full-Stack AI Engineer Seeks Collaboration**: A full-stack and **AI engineer** is looking to connect with others who have great ideas for new projects.
   - They invite anyone interested to connect.
- **Seeking Visionary Project Ideas**: An engineer with a background in both full-stack development and AI is actively seeking collaborators for groundbreaking new projects.
   - The engineer hopes to find individuals with innovative ideas to bring to fruition.


  

---




### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1477036433347707055)** (1320 messages🔥🔥🔥): 

> `Daniel Han spiritual journey, Free will discussion, UltraMix v1 dataset, Gemma translation, Scaling AI memory` 


- **Daniel Han undergoes spiritual trip**: Daniel Han recounts going on a *spiritual journey and having a psychotic break* after reconciling with the nature of the universe.
   - He aims to express this experience in his work, drawing inspiration from figures like **Alighieri**, the **Wachowski brothers**, **Harlan Ellison**, **Lovecraft**, and **Silent Hill**.
- **Delving into Free Will Debate**: Users debated the existence of **free will**, with one positing that it is merely an emergence from high dimensional biasing and a biological device to override objective trajectories.
   - The discussion touched on determinism, generational trauma, environmental influences, and geneology as key factors shaping human behavior, comparing humans to *wetware* and *software*.
- **AtAndDev releases UltraMix v1 Dataset**: AtAndDev introduced [UltraMix v1](https://huggingface.co/datasets/AtAndDev/ultramix-v1), a large, clean, and diverse conversational stem mix designed to be very balanced.
   - The dataset comprises over **5 million turns** from sources like **MegaScience**, **UltraChat**, and **Hermes-3 Dataset**, with detailed ratios available in the dataset readme.
- **Turkish translation with TranslateGemma**: A user mentioned translating a dataset to Turkish using TranslateGemma 4B for post-training purposes, but others note that there isn't even an 8B version of **Gemma**.
   - Other users pointed out that while **Gemma 3n** has an **8B** version, it is not meant for translation.
- **Architecting AI's Infinite Memory**: A discussion revolved around scaling AI memory to infinity, with one user proposing a tiered approach akin to a spiderweb, managing long-term and task-specific contexts.
   - Suggestions included using **knowledge graphs** attached to .md files and considering **GNNs** for specific memory layouts, while acknowledging the challenges of preventing useless context retrieval.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1477071439633842368)** (66 messages🔥🔥): 

> `Qwen3.5 MOE issues on master, Unsloth Dynamic 2.0 GGUFs confusion, 128k context training Qwen 3.5 27B, Unsloth's AI Bot, Loop issue during tool calling with Qwen3.5` 


- **Qwen3.5 MOE Model Optimization Problems**: Members are reporting issues with **Qwen3.5 MOE** on the `unsloth-zoo` master branch, indicating that only the **pypi** versions which lack optimizations for the **MoE model** seem to work.
   - One member found that directly applying [PR #495](https://github.com/unslothai/unsloth-zoo/pull/495) resolves the issue, suggesting that subsequent commits might be causing the incompatibility.
- **Clarification Sought on Unsloth Dynamic 2.0 GGUF Models**: There's confusion regarding **UD prefixes** for **Unsloth's Qwen GGUF models**, specifically whether non-UD prefixed models differ from **bartowski's quants** and if all **Unsloth quants** are inherently superior.
   - A member linked to the [Unsloth Dynamic 2.0 GGUFs documentation](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs), highlighting the lack of clarity in differentiating between **UD and non-UD quants** published by Unsloth.
- **Qwen3.5 27B Struggles with 128k Context Training**: Training **Qwen 3.5 27B** with **128k context** fails even with **2xH200s** due to high VRAM consumption which depends on the training method and hyperparams.
   - One member humorously shared an experience of filling their entire swapfile while experimenting with **128k context** using **qwencodernext 80b**, suggesting that **64k** is currently a more practical *sweet spot*.
- **Unsloth's AI Bot: Where Art Thou?**: A member inquired about the existence of an **AI bot** on the server to answer questions related to using **Unsloth**.
   - Unfortunately, a member confirmed that *currently we have no bot*.
- **Ollama Plagued By Qwen3.5 GGUF Incompatibility**: **Qwen3.5 GGUF** models are reportedly incompatible with **Ollama**, with one member labeling **Ollama** a *complete scam* after encountering issues running various models.
   - Despite **Ollama** claiming **Qwen3.5** support in version **0.17.5**, users are experiencing errors related to *unknown model architecture*.


  

---




### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1477130521036460213)** (20 messages🔥): 

> `Unsloth Training, Gemini CoT, Claude Summaries vs Gemini Summaries, Heretic Uncensored Base Models` 


- ****Unsloth Breaks Records** on Heretic'ed Model**: A model trained on a **HERETIC uncensored base** and then with **Unsloth** achieved record-breaking scores on several benchmarks: **arc_challenge (0.661), arc_easy (0.816), boolq (0.878)**, and others.
   - This new model outperforms **Qwen3.5-27B-Text** in 6 out of 7 basic benchmarks.
- ****Gemini CoT Summaries** Cause Hallucinations**: A member claimed that using **Gemini's CoT summaries** for training can lead to increased **hallucinations** because the summaries don't represent actual thinking processes.
   - They argued that these summaries, which often describe actions not actually performed by the model, can train the model to generate *hallucinatory CoT*.
- ****Claude's Summaries** Are Superior to Gemini's**: A member noted that *on Claudes the summaries are MUCH better and look like real claude thinking, and i have less objections there, but for gemini the summaries are extremely awful*.
   - It was mentioned that **Claude(s)** score better than **Gemini** and one reason to test with **Gemini** is its specific *finger print* that gauges training effectiveness.
- ****Heretic Uncensored** Powers New Qwen Model**: A new **Qwen 3.5 27B** model, based on a **HERETIC uncensored** base, has exceeded the performance of the original model in initial training runs, achieving trending status on Hugging Face [DavidAU/Gemma3-27B-it-vl-Polaris-HI16-Heretic-Uncensored-INSTRUCTle](https://huggingface.co/DavidAU/Gemma3-27B-it-vl-Polaris-HI16-Heretic-Uncensored-INSTRUCT).
   - The training focused on improving the reasoning block size and quality, utilizing **UNSLOTH** on local hardware.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1477062686717120563)** (29 messages🔥): 

> `Embedding Fine Tuning, MoE Rebalancing, Sakana AI's text-to-lora, Molabu research, GTC` 


- **Sakana AI releases text-to-lora model**: **Sakana AI** released their [text-to-lora model](https://huggingface.co/SakanaAI/text-to-lora/tree/main) and [code](https://github.com/SakanaAI/text-to-lora), which takes *around 5 days* to train on a single **H100 GPU**.
- **Molabu started research**: A member shared that a certain research started with **Molabu**, and linked to the original paper: [https://arxiv.org/pdf/2506.16406](https://arxiv.org/pdf/2506.16406).
- **Unsloth crew to attend GTC**: A member asked if the **Unsloth crew** is working on **MoE rebalancing** and if they're coming to **GTC**.
- **Excitement for embedding fine tuning**: There's excitement about embedding fine tuning and its benefits.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1478075357444837539)** (2 messages): 

> `LM Link creation, device discovery, waitlist updates` 


- **LM Link Creation and Device Discovery Stabilized**: **LM Link creation** and **device discovery** had issues but should now work stably after system overloads were resolved.
   - The team apologized for the inconvenience, noting that the feature is in Preview and such fixes are expected before General Availability (**GA**).
- **Waitlist Paused but Now Active Again**: The waitlist was temporarily paused for additional testing but has been reactivated as of **8:55pm EST**.
   - Users can expect to receive an email notification when they are admitted from the waitlist.


  

---




### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1477032824967266344)** (993 messages🔥🔥🔥): 

> `LM Studio Model Loading, Qwen 3.5 New Models, GPT-OSS Performance, ROCm and VLLM issues, Local vs Cloud LLMs` 


- **LM Studio Supports Multiple Models Simultaneously**: Members discussed the capability of **LM Studio** to load multiple models and use them for specific tasks by selecting them via the API, and [provided a screenshot](https://cdn.discordapp.com/attachments/1110598183144399061/1477775728857710622/image.png?ex=69a74ee5&is=69a5fd65&hm=df2d179cf9d9ced0d1e0caebcd3d73bcb54e3b7f8ae5081855799ee4c23fe7b2&) as proof.
   - One member suggested a feature where models could call upon each other as agents, but others noted that such functionality would require custom code implementation, because *"make the llms talk to each other" is not a feature that makes sense in isolation"*.
- **Qwen 3.5 Small Models Gain Traction**: The release of smaller **Qwen 3.5** models (9B, 4B, 2B, and 0.8B) was met with excitement, but there was some discussion about whether the [Unsloth versions](https://discord.com/channels/1110598183144399058/1111440136287297637/1476681772317016137) supported the thinking toggle feature.
   - Members confirmed these models were strong, with the 9B model being particularly impressive, but are *highly censored*.
- **GPT-OSS is best for all use cases!**: Members discussed **GPT-OSS** output, with Lithium saying that *gptoss is the best if you have no controvertial topics or ideas EVER* and creates fantastic outputs.
   - One user reported difficulty in achieving good results with GPT-OSS, while another stated that it could generate good results, depending on the implementation.
- **ROCm proves hellish to build VLLM for**: A user described the difficult process of building **VLLM** for **ROCm**, but was doing so in order to compare speeds with **LMStudio**.
   - Others agreed with this assessment and emphasized the problems with dependency resolution. The user managed to succeed building it using guidance from AI assistance.
- **Small Models Overthinking? Debunked**: Members discussed whether smaller models tend to overthink, or have reasoning disabled.
   - While initially it was reported *those small ones overthink*, that claim was later walked back by others.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1477087331314962453)** (282 messages🔥🔥): 

> `EPYC system for benchmarking, AMD V620 GPU performance, OSS 20B model testing, W7900 vs R9700 GPU comparison, LMS GPU multi card setup` 


- **EPYC Benchmarks Qwen3.5 Model**: A user ran benchmarks on an EPYC system using **Qwen3.5 Q6k**, achieving **6.2 t/s** on CPU only, **45 t/s** on a single V620 (ROCm), and **51.5 t/s** on a single V620 (Vulkan).
   - They also tested splitting across 8 GPUs, getting **11.2 t/s** with ROCm and **11.7 t/s** with Vulkan, noting surprise at the GPU split numbers and compared favorably to their own 4 channel xeon system.
- **Vulkan vs CUDA Performance**: Members compared GPU performance, with one user reporting **25-31 tok/s** with CUDA and **15-20 tok/s** with Vulkan on an unspecified GPU.
   - They discussed potential differences between Nvidia and AMD GPUs and speculated whether **Vulkan drivers** prioritize Polaris performance due to its widespread use.
- **W7900 or R9700 Decision**: Discussion about wether to buy a used **W7900 48GB** for **£1130** with no native FP8, RDNA3, PCIE 4 x 16, or a new **R9700 32GB** for **£1050** with native h/w FP8, RDNA4, PCIE 5 x 16.
   - Arguments were made that for **dense models** memory bandwidth matters, and **W7900** wins, but for **MoE models**, compute matters more, and **R9700** wins, but in the end user chose W7900.
- **Single vs Multi GPU Performance**: Experiments were conducted on multi-GPU setups, revealing that performance can vary significantly based on the number of cards used and their configuration.
   - A user found that moving from **3 to 5 RX570s** improved speeds, even without full offload, and they also observed that the order of cards in LMS impacts the speed.
- **ASIC AI Accelerators**: Members discussed the emergence of **ASIC AI accelerators**, referencing the **Taalas HC1** which delivers up to **17000 tokens/s** with a **Llama-3.1 8B** model.
   - While the speed is impressive, some users noted that the model is outdated and suggested partnering with an AI lab to develop better models for this hardware; they also mentioned [Bitmain's early efforts in AI hardware](https://sophon-edge.gitbook.io/project/overview/edge-tpu-developer-board).


  

---




### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1477033018454704342)** (1337 messages🔥🔥🔥): 

> `c.ai safety issues, Making AI sound human, Nanobanana 2 problems, Free Claude Opus 4.6, Sidekick API exploit` 


- **C.ai accused of having Safety problem, Models "RP" wild scenarios**: Members express concerns that [Character.ai](https://character.ai/) lacks sufficient safety measures, pointing out that the AI can be prompted to engage in inappropriate role-playing scenarios. One user shared, *the biggest and most dangerous ai is def c.ai* and *the AI community they don’t care as long as the upgrades keep coming*.
   - Others responded with links like [this YouTube video](https://youtu.be/eMhDh6pXpkM?feature=shared) about inconsistent behavior in chat tools.
- **Make AI Sound Human with One Weird Trick!**: A member shared an [xkcd comic](https://xkcd.com/1172/) as *the one single prompt to make ai sound human*.
   - The comic humorously depicts the prompt needed to elicit human-like responses from AI.
- **Nano Banana 2 image model deemed Worser than Pro**: Users report that the **Nano Banana 2** image model is performing worse than **Nano Banana Pro**, with one user stating, *The entire server was in denial when I said NB 2 is worse than NB Pro in quality*.  It seems image generators are temporarily down due to a *google problem*.
   - Another user added, *It’s a whole step back from the quality*, suggesting that the dataset might be bloated.
- **Free Sidekick AI gives Opus 4.6?? Users Sus, Potential API Exploitation**: Members discussed a website offering free access to **Claude Opus 4.6**, raising suspicions about its legitimacy with some thinking the site may be secretly running something like *gpt oss 20b*.
   - The consensus is that the site is using the system prompt of the Tobit [Sidekick AI](https://tobit.com/sidekick) platform, and some are convinced the site is not legit, as, *noone would willingly give out opus 4.6 endlessly for free*.
- **Garf site's API exploited? A Tobit Sidekick Cloning?**: A user claimed that the 'Garf' website's free Claude access might be due to the site owner reverse engineering Sidekick's API, while using a german localized setup, with members debating the safety of using the site. Several think this *is definitely chatgpt*, however members seem to like **Max** the custom router.
   - Others warned about potential data harvesting and IP logging, suggesting users employ VPNs, with one saying, *use vpn or else mr garf will steal your data in exchange for using free ai*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1478120321738604575)** (2 messages): 

> `Arena.ai, Runway Gen-4.5, Text-to-Video Arena leaderboard` 


- **Arena.ai Explained in 60 Seconds**: A [YouTube video](https://www.youtube.com/watch?v=nktiDGTn61I) explains **Arena.ai** in 60 seconds, prompting community feedback on what the video might have missed.
- **Runway Gen-4.5 Debuts on Text-to-Video Arena Leaderboard**: The [Text-to-Video Arena leaderboard](https://arena.ai/leaderboard/text-to-video) welcomes **Runway Gen 4.5**, achieving a score of **1218**, comparable to **KlingAI’s Kling-2.6-Pro**.


  

---




### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1477040539647082656)** (845 messages🔥🔥🔥): 

> `Unity Engine vs Godot Engine for game development, MCP Blender Addon, Anthropic and mass surveillance, Auto Apply feature in Cursor, Cursor referral codes` 


- **Unity Engine vs Godot Engine for game dev**: Members discussed using **Unity** with a **YouTube** tutorial versus **Godot**, noting they're *pretty much same same*, using code for debugging and previewing, though **Godot lacks native AI**.
   - One user preferred **Godot** for **2D games**, avoiding *fluff enterprise features*, while another sought help with integrating **Cursor** into **Unity**.
- **MCP Blender Addon lets AI take the wheel**: A user shared a link to a **MCP Blender addon**, enabling **AI** to wield the tools in **Blender**, [promising AI integration in 3D modeling](https://hackaday.com/2025/05/18/mcp-blender-addon-lets-ai-take-the-wheel-and-wield-the-tools/).
   - They expressed hope for **Godot's** mainstream adoption, lamenting their lack of **GDScript** knowledge, while others suggested using **Cursor** to code it.
- **Anthropic & US Govt could make AI weapon systems**: A user inquired about the US government potentially using the **Defense Production Act** against **Anthropic**, possibly to commandeer their technology for **military purposes**.
   - Another user suggested the government may want their **infrastructure** to do mass surveillance of US citizens and making fully **automated weapon systems**.
- **Auto-Apply Button Blues: Cursor Users Beg for Redo**: Cursor users are requesting a change to the **'Reject All Edits' button**, proposing a modal confirmation or a redo button to prevent accidental clicks while dragging screenshots.
   - Users are annoyed by losing edits due to accidental clicks.
- **Referral Roulette: New vs. Old Accounts**: Users shared **Cursor referral codes**, but there was confusion as some offers only applied to new accounts or were already redeemed, leading to disappointment among existing users.
   - One user humorously offered to endorse **Cursor** in exchange for a **dev-graced account**, showcasing the appeal of premium features.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1477061878466478131)** (29 messages🔥): 

> `Robert Stock's AI Pitching Analytics Platform, Staying up to speed on AI, Codex completion, CC + Remotion, AI Hackathon` 


- ****Baseball Player Builds AI Analytics Platform!****: Professional pitcher Robert Stock, with no prior coding experience, utilized **AI** to develop a comprehensive **pitching analytics platform** featuring **machine learning models** and a database of **8.9 million pitches**, as reported in [this X post](https://xcancel.com/robertstock6/status/2027401720209531145?s=12).
- ****Navigating the firehose of AI info****: A member inquired about strategies for staying current with the rapid pace of AI developments, and how to filter through the noise to follow the most relevant threads.
   - One member advocated focusing intensely on a few key channels, while others embrace random doom scrolling to stay broadly informed.
- ****Taming Codex Completion****: A member sought advice on ensuring **Codex** runs to completion when given tasks, sharing a [link to the OpenAI Cookbook](https://developers.openai.com/cookbook/examples/codex/long_horizon_tasks) suggesting prompt engineering.
   - The member humorously reported having **Codex** *read the blog post and write the prompt* for them.
- ****CC + Remotion: The New AGI?****: A member enthusiastically declared that **AGI is here**, exemplified by the combination of **CC (Creative Computing)** and **Remotion**, crediting a previous post by another member.
   - The member reported *pretty much one-shotting* a project a professional design team struggled with for weeks, using **CC** with verification loops, **ClaudeKit**, a lip-synced avatar video, and inspiration from **Anthropic’s CoWork ads**, as visualized in [this Youtube Video](https://youtu.be/AjhLqAp4NUU?si=TI22ePQn14vV8SgV).
- ****AI Hackathon: Buildstory Announced!****: An AI-first hackathon was announced, **Buildstory**, running from **March 1st to March 7th**, welcoming all experience levels and tools, with details available on [the buildstory.com website](https://buildstory.com).


  

---




### **Latent Space ▷ #[announcements](https://discord.com/channels/822583790773862470/822583965009051668/1477825645596639272)** (9 messages🔥): 

> `Latent Space's 5th Anniversary, Community Meetup, In-Person vs Virtual Event` 


- **Latent Space Celebrates 5 Years!**: The Latent Space community celebrates **5 years** since its inception, growing from a small group of devs to a **globe-spanning group**.
   - A big thank you was given to the founder and the **Latent Space podcast** team for their contributions.
- **Potential Meetup Plans Spark Discussion**: Possible plans are in the works to celebrate Latent Space's 5th anniversary with either a virtual/in-person event on **March 19th** or an in-person event later in the year.
   - A poll has been created to gather community feedback, with a minimum of **2-3 months** notice planned for any in-person events later in the year.
- **Community Expresses Gratitude**: A member expressed gratitude for all the hard work put into Latent Space, recognizing it as being in a league of its own.


  

---


### **Latent Space ▷ #[creator-economy](https://discord.com/channels/822583790773862470/822625128843182090/1477636006047125585)** (15 messages🔥): 

> `Steve Ruiz's Product Design Notebooks, Arena Magazine article on Leerob, Stripe Press's focus on tech elite attention` 


- **Steve Ruiz's Notebooks Spark Design Envy**: A member shared a link to [Steve Ruiz's 2018 product design notebooks](https://xcancel.com/steveruizok/status/2027499153463341513?s=46&t=_hz7_TqpYWiUUE4FPGb-5Q), praising them as *gold*.
   - Another member admired the *aesthetic of cool notebooks*, contrasting it with their own less visually appealing college notes.
- **Leerob Featured in Arena Magazine**: A member shared a link to an **Arena Magazine** article about **Leerob**: [The School of Lee](https://arenamagazine.substack.com/p/the-school-of-lee).
   - Another member expressed surprise and congratulated Leerob, noting the effort put into the piece.
- **Stripe Press Tracks Tech Elite's Focus**: A member mentioned subscribing to **Stripe Press** to monitor the focus of the *tech elite*.
   - They noted a past emphasis on *war* and recent acquisition of **Software that Dominates** advertisement.


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1477060821539618989)** (126 messages🔥🔥): 

> `Tabs vs Spaces, Bun Supply Chain, ChatGPT Jailbreak, Cards Against Humanity, AI Coding Tools` 


- **Classic Tab vs Space developer debate lives on**: The classic developer debate regarding the use of **tabs versus spaces** in source code was reminisced about in a [tweet](https://xcancel.com/jasonbosco/status/2027413174203621437?s=12).
- **Bun supply chain risk discussed**: A discussion was sparked by the official Bun account questioning whether the project now constitutes a **supply chain risk**, which generated significant community engagement [here](https://xcancel.com/bunjavascript/status/2027638567317737895).
- **ChatGPT successfully jailbroken**: A **vx-underground post** demonstrated a successful jailbreak of **ChatGPT**, where the AI is manipulated into abandoning its safety protocols [as seen here](https://xcancel.com/vxunderground/status/2027613100870930541).
- **Cards Against Humanity keeps being decent**: Cards Against Humanity continues being just weirdly decent about everything, with a member noting that *the fundraiser where they just put 100% of funds towards renting construction equipment to dig a hole was great*.
   - Reference was made to [this post](https://bsky.app/profile/carlquintanilla.bsky.social/post/3mfubhrwwjc2w).
- **AI Coding Tool market heats up**: The rapid valuation growth of **AI coding tools** like **Cursor** and **Claude Code** was discussed, with the argument against the 'bubble' narrative highlighting that **enterprise adoption** is just beginning [as seen here](https://xcancel.com/deedydas/status/2028608293531435114?s=12).


  

---




### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1477061930178056325)** (27 messages🔥): 

> `OpenAI 预测市场解雇事件、Ellison 家族收购 Warner Bros. Discovery、Kirbxbt 'Taps Sign' 模因、Expensify 界面批评、Netflix 经营杠杆` 


- **OpenAI 员工因参与预测市场被解雇**：据报道，一名 OpenAI 员工因在 [Polymarket](https://polymarket.com/) 和 [Kalshi](https://kalshi.com/) 等平台上参与预测市场活动而被解雇，引发了对**内幕交易**和遵守公司政策的担忧。
   - 交易金额很小，一位成员评论道：*"可能最多也就 1000 美元 💀"*。
- **Ellison 家族杠杆收购 WBD**：在 Larry Ellison 提供的 **457 亿美元**担保支持下，David Ellison 正在主导一项涉及 **1110 亿美元**的高杠杆收购案，对象是 **Warner Bros. Discovery**。
   - 根据[这份分析](https://xcancel.com/anisha_moonka/status/2027489321209721022?s=12)，该交易与 2018 年 AT&T 收购案的失败如出一辙，尽管面临巨额债务和有线电视收入下降，但仍可能将好莱坞整合为“四大”格局。
- **Kirbxbt 融入模因文化**：用户 @kirbxbt 发布了一个包含 **'taps sign'** 表达的简短模因，获得了超过 **10,000** 次点赞和 **400,000** 次浏览的高参与度。
   - 在发布该帖子后，@kirbxbt 提到：*"是的，再等一等，也许就会买入 [Kirbxbt 'Taps Sign' 模因帖]，通常在战争结束后一切都会上涨。"*
- **Netflix 经营杠杆讨论**：Joseph Carlson 分析了 **Netflix 的经营杠杆**，指出其从高昂内容成本与收入挂钩（**2011-2016**）转变为收入大幅增长而内容支出保持相对持平。
   - 根据[这份分析](https://xcancel.com/joecarlsonshow/status/2028173200715334122?s=12)，这种转型为 Netflix 带来了丰厚的利润。
- **Amazon 昂贵的 AI 滞后**：一篇文章讨论了 [Amazon 在 AI 竞赛中落后所面临的代价](https://om.co/2026/02/27/amazon-the-cost-of-ai-lateness/)。
   - 发布者评价道：*"对 AWS 非常严苛，但我觉得没错"*。


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1477069821370826883)** (8 messages🔥): 

> `紧跟讨论节奏、AI 电影制作新手、围绕自我改进的多 Agent 系统构建个人生活方式、构建处理个人和客户数据的系统、构建筹款工具` 


- **好奇的 Anurag 询问信息获取方式**：一位来自南湾的成员询问如何紧跟频道中**有趣的讨论**，并询问是否启用了“频道内对话总结（*In-Channel Conversation Summaries*）”。
- **Tanmay 从拓扑 CMT 转向 AI SaaS**：一位位于旧金山的成员，此前从事 **topological CMT** 研究，现在正在创办一家 **AI B2B SaaS** 初创公司。
- **Liam 的 LLM 精彩生活**：一位来自悉尼/旧金山的成员正围绕**自我改进的多 Agent 系统**构建个人生活方式，旨在从**第一性原理**和高成本效益设计出发，可持续地创造超过消费的收入。
- **Ash 的 AI 艺术抱负**：一位 **AI 电影制作**新手分享了他们使用 **Runway、Grok 和 Veo** 等工具创作图像和视频的基础技能，寻求为项目做贡献及向他人学习的机会，并在[频道中](https://discord.com/channels/844675581291397168/1192531825310097448)分享了一些作品。
- **Ben 的布宜诺斯艾利斯实践：确定性数据系统**：一位身在布宜诺斯艾利斯的美国心理学家正在构建**确定性系统**来处理个人和客户数据、自动化任务并生成报告，同时还经营着一家非营利组织、咨询公司和电子商务品牌。


  

---

### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1477052611860500603)** (168 messages🔥🔥): 

> `Serverless downsides, Veritasium's XZ backdoor video, AWS Batch Spot Instances, NVMe drives vs EBS, N^3 algorithm` 


- **Serverless limitations Disclosed**: Members discussed the downside of **serverless** computing, noting that *you don’t control the cpu cycles* and emphasizing the importance of owning your performance, with serverless shining for **edge functionality** when not hitting a centralized database.
   - It was also mentioned that serverless is good for keeping costs down when hacking on something, but it's important to have an exit plan and not over-optimize performance on the serverless implementation.
- **Veritasium Drops XZ Backdoor Video**: A new **Veritasium** video on the **XZ backdoor** was discussed, with one member finding it interesting, while another found *his style a bit dramatic*.
- **Batch uses Spot Instances**: The group discussed using **AWS Batch** with spot instances for bursty, parallelized compute needs, processing 1000 jobs of lumpy sizes and emphasizing the importance of them not blocking each other.
   - One user humorously admitted, *We are the noisy neighbor lol*, after discovering their queue didn't support concurrency limits when it tried to spin up a backlog of 94 jobs.
- **Atproto identity issues alarm Members**: A member shared their experience building an **atproto** app, ultimately finding the platform unimpressive due to its approach to decentralization, detailing concerns in [this blog post](https://kevinak.se/blog/be-wary-of-bluesky).
   - The primary issue is that a **PDS host stores your data** and signing keys, meaning they can act as you on any atproto application, raising concerns about true decentralization and control over one's identity.
- **Generating keys within secure enclaves arrives**: A member shared that they are *now have a key generated by my secure enclave, synced to icloud*, after going through a setup process for **bluesky-plc-recovery-key**.
   - The member also linked to [profile page](https://bsky.app/profile/trezy.codes/post/3mg43psnpks2i) after submitting a contribution.


  

---


### **Latent Space ▷ #[devtools-deals](https://discord.com/channels/822583790773862470/887780383838572604/1477875246634762364)** (1 messages): 

> `Rspack, Webpack, Vite` 


- **Rspack: Low-Key Terrific!**: A member mentioned that **Rspack** is low-key terrific and is much easier to migrate to from **Webpack**.
- **Greenfield Favors Vite**: They added they would probably still choose **Vite** if they were starting greenfield.


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1477151746320240650)** (3 messages): 

> `AI filmmaking, Runway, Veo, AI projects` 


- **New AI Filmmaker Seeks Projects**: A new member named Ash introduced themself as new to **AI filmmaking**, with basic knowledge of tools like **Runway** and **Veo**, and offered to assist on projects to learn more.
   - Ash attached multiple images showcasing their creative capabilities using **AI tools**.
- **AI Filmmaking Novice Showcases Visual Portfolio**: A new member shared a portfolio of AI-generated images, demonstrating their aptitude with various creative tools.
   - The attached visuals highlighted the user's initial skills and artistic direction in the **AI filmmaking** domain.


  

---


### **Latent Space ▷ #[cloud-infra](https://discord.com/channels/822583790773862470/973816817489424414/)** (1 messages): 

swyxio: https://linqapp.com/ heard that Poke uses this
  

---


### **Latent Space ▷ #[databases-data-engineering](https://discord.com/channels/822583790773862470/973820036089270272/1477415520687882403)** (4 messages): 

> `Database Branching, NeonDB internals` 


- **NeonDB enables instant branching**: Members discussed how [NeonDB](https://www.neondb.com/) enables instant branching, cloning, and recovery for databases, similar to how developers branch code.
   - They talked about how even **petabyte-scale databases** can be copied in seconds, enabling fast experimentation, safe rollbacks, and instant restoration without operational overhead. Here's a link to the podcast where they discuss [Building PostgreSQL for the Future](https://softwareengineeringdaily.com/2025/05/20/building-postgresql-for-the-future-with-heikki-linnakangas/gyan.k).
- **NeonDB uses pointers instead of blocks**: To achieve instant branching, **NeonDB** uses pointers instead of pointing to physical blocks of storage.
   - Thus, they can copy all the pointers except a few to achieve near instantaneous copies.


  

---




### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1477364243140579469)** (32 messages🔥): 

> `Great Sandbox Symposium, Vercel Billboard Marketing, San Francisco Public Toilet Costs, SF Renaissance Funding, SF Tourism` 


- ****Builders Assemble** for the Great Sandbox Symposium**: **Dexhorthy** announced [The Great Sandbox Symposium](https://xcancel.com/dexhorthy/status/2027592650833248399?s=46) taking place on **March 7th, 2026**, in **San Francisco**.
   - It's a *non-competitive gathering for builders to test and compare different sandbox technologies*, focusing on research sharing and collaborative hacking.
- **Vercel's Marketing Billboard compared to **Neurodivergent Flirting****: A thread revolves around a tweet by **@0xluffy** comparing [Vercel's marketing billboards](https://xcancel.com/0xluffy/status/2027545548975444274?s=12) to a specific style of neurodivergent flirting.
- **SF Spends **Millions** on Public Toilets**: **Sheel Mohnot** noted that **San Francisco** spent **$14 million** last year on its **30** freestanding public toilets, which averages out to approximately **$19 per visit** based on **750,000** recorded uses, as seen on [this tweet](https://xcancel.com/pitdesi/status/2028176457307181457).
   - Several users discussed how that amount of money can affect the city's revenue from being cleaner and more accessible.
- **Wonderful weather in the city**: A member shared the attached photo of a [cafe near Fort Mason](https://cdn.discordapp.com/attachments/1477744053549535454/1477821397136048259/IMG_6574.jpg?ex=69a7796d&is=69a627ed&hm=3be09fde4558668922f5a7009a24ec438544e635b3e27ddd4ac0a1973d568604&), indicating *a wonderful day in the city*.


  

---


### **Latent Space ▷ #[london](https://discord.com/channels/822583790773862470/979492759759097866/1478045484487540766)** (1 messages): 

> `Agent Client Protocol, ACP Event in London` 


- **London to host Agent Client Protocol (ACP) event**: The Agentic AI community in London will host an event covering the **Agent Client Protocol (ACP)** with speakers from **Zed Industries** and **Jetbrains**.
   - The event aims to explain ACP, its current state, future directions, and how to switch coding agent harnesses or build custom ACP clients painlessly, with registration available [here](https://luma.com/4hs6hs36).
- **Deep Dive into ACP with Zed Industries and Jetbrains**: The London Agentic AI community is focusing on the **Agent Client Protocol (ACP)**, featuring insights from its creators, **Zed Industries**, and innovators like **Jetbrains**.
   - Unlike the well-known MCP, this event will explore ACP's current status, future roadmap, and methods for seamless coding agent harness switching and custom ACP client development, with event details available at the provided [link](https://luma.com/4hs6hs36).


  

---




### **Latent Space ▷ #[situation-room](https://discord.com/channels/822583790773862470/1036726703730466896/1477080518226083861)** (280 messages🔥🔥): 

> `Anthropic supply chain risk, Iran-Israel conflict escalation, Cybersecurity and hacking, US political instability, Economic impact of military spending` 


- **Anthropic labeled as supply chain risk**: After Anthropic refused to grant the Pentagon unrestricted access to its models, the Department of War labeled them a **supply-chain risk**, banning military contractors from working with them, initiating a six-month phase-out of their AI services, according to [this tweet](https://xcancel.com/secwar/status/2027507717469049070?s=46&t=FlpzvQFmjnd0z3HkNeNT1A).
- **Trump's prediction on Obama's foreign policy**: In 2012, Donald Trump tweeted that President Obama might initiate military action in Libya or Iran to offset declining poll numbers ([tweet here](https://xcancel.com/realDonaldTrump/status/255784560904773633)).
   - A member found *this ironic* give the context of the thread.
- **US warns citizens to leave Israel**: The US government has told its people to leave Israel asap, following reports that US is attacking Iran and Bahrain is being targeted ([official advisory](https://il.usembassy.gov/travel-advisory-february-27-2026/)).
   - Several members expressed skepticism, calling this a likely development that Israel has been preparing for decades.
- **Cloudflare promises preparedness against Iranian cyber threats**: Cloudflare's CEO reassured customers that the company is familiar with Iranian cyber techniques and is fully prepared to defend against potential attacks ([source](https://xcancel.com/eastdakota/status/2028299294244012488?s=20)).
   - Some members found humor and sarcasm in this context, joking that the **limited-time AI usage** joke has turned very real.
- **Altman Clarifies OpenAI's Department of War Agreement**: Sam Altman shared an internal update that the contract with the Department of War has new amendments that explicitly prohibit the use of AI for domestic surveillance of U.S. persons ([source](https://xcancel.com/sama/status/2028640354912923739?s=46&t=_hz7_TqpYWiUUE4FPGb-5Q)).
   - He clarified that the services will not be used by intelligence agencies like the NSA without further modifications, reflected on the 'sloppy' timing of the initial announcement, and confirmed support for Anthropic receiving similar terms.


  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1477034858311319687)** (151 messages🔥🔥): 

> `Anthropic Skills, Building an OS with LLMs, Apple M4 Neural Engine, OpenAI DOD revised AI Deal, AI Coding Tools` 


- **Anthropic Skills Design Evolves Prompting**: Anthropic released a guide shifting focus from prompt engineering to structured **'Skills'**, allowing **Claude** to execute repeatable workflows using progressive disclosure to reduce context bloat, leveraging **MCP** for tools and skills for logic, as detailed in [this post](https://xcancel.com/heyrimsha/status/2027350587533332748?s=12).
   - The approach emphasizes building execution layers and testing infrastructure rather than refining language.
- **New Skills Let AI Coding Agents Control Electron Apps**: Chris Tate announced a new skill for the **agent-browser** tool that allows AI coding agents to control **Electron-based desktop applications** like Discord, Figma, and VS Code, or assist in debugging local Electron development as shown in [this tweet](https://xcancel.com/ctatedev/status/2028128730132922760?s=12).
- **OpenAI's Revised DOD Deal Includes Surveillance Protections**: OpenAI and the Department of Defense have reportedly updated a recent AI agreement to include stronger surveillance protections as reported in [this TechMeme post](https://xcancel.com/Techmeme/status/2028640454510788777).
- **The Secret Sauce: Unlocking Apple M4 Neural Engine**: A solo researcher successfully ran a **Llama2 110M** model on Apple's **M4 Neural Engine (ANE)** by reverse-engineering private APIs to bypass the standard CoreML stack, as detailed in [this post](https://xcancel.com/ai/status/2028544293309448287).
- **Alibaba Qwen 3.5 Series Debuts Small Models**: Alibaba has introduced the **Qwen 3.5 Small Model Series**, featuring native multimodal capabilities and ranging from 0.8B to 9B parameters as linked in [this tweet](https://xcancel.com/Alibaba_Qwen/status/2028460046510965160).


  

---




### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1477045182678696059)** (42 messages🔥): 

> `Databricks OAPL for LLM Reasoning, PROSPER RL Algorithm, Anthropic's 'Skills' Guide, GLM-5 Async RL Optimization, doc-to-LoRA` 


- **Databricks Boosts LLM Reasoning with OAPL**: Databricks introduced **OAPL** (Optimal Advantage-based Policy Optimization with Lagged Inference), an off-policy reinforcement learning method that enables **LLMs** to learn reasoning more efficiently than **GRPO**, requiring **3x fewer training generations** and simplifying the training infrastructure.
   - More details are available in [Databricks Mosaic Research: OAPL for LLM Reasoning](https://x.com/g_k_swamy/status/2027450376593805746?s=12).
- **PROSPER Tackles Inconsistent LLM Feedback**: Gokul Swamy introduces **PROSPER**, a new regression-based reinforcement learning algorithm designed to handle inconsistent feedback from **LLM judges** by utilizing rubric rewards and **Blackwell's approach**.
   - More details are available in [Introduction to PROSPER RL Algorithm](https://x.com/g_k_swamy/status/2027450376593805746?s=12).
- **Anthropic Kills Prompt Engineering**: Anthropic's new **30-page guide** on building 'Skills' for Claude shifts the focus from simple prompt engineering to **structured execution design**.
   - The guide details how packaging workflows into specialized files using progressive disclosure reduces context bloat and turns **AI interactions into repeatable, scalable infrastructure** across Claude's API and tools, and can be found at [Anthropic's 'Skills' Guide](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf).
- **GLM-5 Revamps Async RL Optimization**: The **GLM-5 team** has introduced a fix for GRPO in async RL training to handle memory constraints when dealing with long trajectories.
   - Instead of the traditional importance sampling ratio, they propose using (**pi^train / pi^infer**) by having the inference engine send token logprobs directly, eliminating the need for extra weight copies while ensuring more accurate importance sampling and improved training stability.
- **Latent Space Cron Job pulls all podcasts**: A member has set up a cron job to pull all **podcasts, articles, paperclubs, buildersclubs** etc into a db and post notifications on new content/reminders.
   - This is available at the [latent-space-hub](https://latent-space-hub.vercel.app/) and people can add themselves to get certain notifications.


  

---




### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1477033219294625990)** (157 messages🔥🔥): 

> `Model Pricing, GPT-5.2 Pro Pricing, Open Source Claude Cowork, Codex-5.3-high Capabilities, Clanker CLI` 


- **GPT-5.2 Pro Priced Stratospherically**: A member shared an [artifact](https://claude.ai/public/artifacts/0bf8981d-1dcb-4366-960f-c0438b77175b) exploring model pricing and noticed that **GPT-5.2 Pro** is priced at **$21/$168** in/out, inquiring why it's so much more expensive than others in the same tier.
   - The discussion questioned if the model is significantly different through the API.
- **Open Source Cowork Explorations Commence**: Members shared a series of [links](https://www.reddit.com/r/ClaudeAI/comments/1qcf8mu/i_built_an_opensource_alternative_to_claude/) to open-source alternatives to **Claude Cowork**, including [openwork](https://github.com/different-ai/openwork), [kuse_cowork](https://github.com/kuse-ai/kuse_cowork), and [open-claude-cowork](https://github.com/ComposioHQ/open-claude-cowork).
   - The general sentiment was that most were just trying to grab traffic on the initial release of Cowork, with limited activity since.
- **Codex's Kv Cache Gets Complex**: A member highlighted a significant milestone with **Codex-5.3-high** successfully performing advanced low-level operations, including monkey-patching attention modules and executing granular surgical **KV cache eviction** with span tracking, bypassing standard abstractions to complete a complex task in a single attempt. See [this post](https://xcancel.com/eigenron/status/2027300218589614194?s=12)
   - Another member expressed concern about monkey patching becoming an anti-pattern in AI code generation.
- **Clanker CLI Makes One-Shot Debut**: A member announced the completion of an open-source one-shot deployment agent for **Clanker CLI**, designed to automate infrastructure deployments via a specific pipeline, see [this link](https://xcancel.com/tekbog/status/2027614288240836920?s=12).
   - The announcement spurred a tangent to a discussion of a system called *desloppify* to clean up slop.
- **AI Orchestration Challenges Under the Microscope**: Discussion centered on effective **AI agent orchestration** as more about system design and coordination than just chaining API calls, see [this link](https://xcancel.com/asmah2107/status/2027721262324453602).
   - The discussion highlighted unique challenges of agent failure, such as silent errors, hallucinations, and infinite loops, which traditional exception handling cannot easily address.


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1477112063380684881)** (18 messages🔥): 

> `OpenClaw Claude Codes, Anthropic TOS, Bootstrapping Coding Agents, Caching & Smart Routing for Agents` 


- **OpenClaw Opens Claude Codes on Phones!**: A member shared a [YouTube video](https://youtu.be/pC6hhjVQV9A) demonstrating how to open source **Claude codes** from Discord on your phone using **OpenClaw**.
   - The video also features a guest appearance by another member, fulfilling a highly requested episode from the previous week.
- **Anthropic's Ambiguous API Access**: Discussion ensued about **Anthropic's TOS**, with some suggesting the terms forbid using max subscriptions in anything but **Claude Code** and that the agent SDK is not an exception.
   - Others mentioned a clarification on Twitter, implying it was a miscommunication, but advised caution when relying on **Claude** to avoid potential bans.
- **Cogent Coding Agent Sprouts on GitHub**: A member shared their project, [Cogent](https://github.com/abrinsmead/cogent), a coding agent, noting the fun in navigating the same *idea maze* as the **Claude Code** team.
   - They mentioned incorporating features they'd approach differently, such as tabbed sessions.
- **Acorn-eta: Smart Routing & Caching for Agent Inference**: A member is working on caching and smart routing to reduce inference costs for agents, with the code available on [GitHub](https://github.com/pc099/acron-eta).
   - The member is building a framework and hacking different pieces together over the weekends.


  

---




### **Latent Space ▷ #[private-agents-and-workflows-local-llama-ollama](https://discord.com/channels/822583790773862470/1342964204168020018/1477067013649334544)** (2 messages): 

> `Local Image Generation Models, Flux Klien, Z-image` 


- **Flux Klien and Z-image are state of the art**: In a discussion about state of the art local image generation models, **Flux Klien** and **Z-image** (base and turbo) were named as top contenders.
   - The user was seeking recommendations for running image generation models on an older Windows box with a **3070 GPU**.
- **GPU Recommendations for Local Image Generation**: A user inquired about the best local image generation models for a Windows machine equipped with a **3070 GPU**.
   - The discussion centered around finding models that could efficiently run on older, but high-end, hardware.


  

---


### **Latent Space ▷ #[good-writing](https://discord.com/channels/822583790773862470/1385526686736715876/1477838268505522177)** (4 messages): 

> `Satirical Asian Wealth Analysis, Dynastic Wealth Cycles` 


- **Cynical Anatomy of Southeast Asian Wealth Lifecycle**: A [post](https://xcancel.com/wassielawyer/status/2028167447363412410) satirically breaks down the stereotypical lifecycle of ultra-wealthy Asian dynasties.
   - It details the patriarch's origins of wealth and the children's failures, suggesting that true 'winning' is amassing untouchable wealth where family dysfunction is irrelevant.
- **Dynastic Wealth: Questionable Origins and Rebellious Offspring**: The analysis highlights the often-questionable origins of wealth accumulated by the patriarchs of these dynasties.
   - It also notes the subsequent failures and rebellions of their children, ranging from artistic pursuits to substance abuse and influencer careers.


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1477462508636733481)** (12 messages🔥): 

> `AI-Automated Short Drama Production, Visakan Veerasamy post, Negative Space of Taste` 


- **AI-Powered Feline Dramas**: Justine Moore's [post](https://x.com/venturetwins/status/2027240672382849256?s=20) highlights **AI-generated feline characters** automating short drama production in China.
   - This innovative approach showcases the evolving landscape of **AI in media creation**.
- **Visakan Veerasamy's Foretold Engagement**: Visakan Veerasamy's (@visakanv) brief [post](https://x.com/visakanv/status/2027892491522134053?s=20) stating *'as it was foretold'* garnered **over 270 likes and 5,500 views**.
   - The post generated discussion around the power of simplicity and predictive statements on social media.
- **Negative Space Defines Taste**: Scott Stevenson explores 'negative space' in [this post](https://xcancel.com/scottastevenson/status/2027853889605632452?s=12) as vital to taste, using **Dune, Better Call Saul, Lululemon, and Apple** as prime examples.
   - Stevenson argues that what is *left out* shapes perception and appreciation as much as what is included.


  

---


### **Latent Space ▷ #[ai4science-bio-math-physics-chemistry-ai-researcher-ai-scientist](https://discord.com/channels/822583790773862470/1430253273335595079/1477354461180465192)** (8 messages🔥): 

> `IsoDDE Drug Design Engine, DishBrain, Synthetic Biological Intelligence` 


- **IsoDDE throws its Hat in the Drug Ring**: Isomorphic Labs unveiled **IsoDDE**, a successor to **AlphaFold 3**, designed for pharmaceutical drug discovery, as [announced on X](https://x.com/wesroth/status/2027519117545157089?s=12).
- **Neurons do DOOM**: Scientists at Cortical Labs integrated **800,000 living human and mouse neurons** with silicon hardware to create '**DishBrain**,' a system capable of playing digital games like DOOM and Pong, as [posted on X](https://x.com/scitechera/status/2028010532356374754).


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1477739427793993840)** (7 messages): 

> `ARENA curriculum, Interpretability tutorials, SAE framework, text-to-image diffusion model` 


- ****Nanda** Updates **ARENA** Curriculum**: **Neel Nanda** announced a significant update to the **ARENA** curriculum, featuring new coding tutorials focused on key interpretability areas and utilities for integrating these concepts into **LLM prompts**.
- ****SAE Framework** Probes Text-to-Image Diffusion Models**: A paper leverages the **SAE framework** to probe the inner workings of a popular **text-to-image diffusion model**, and uncover a variety of human-interpretable concepts in its activations ([arxiv.org/abs/2504.15473](https://arxiv.org/abs/2504.15473)).
   - The study finds that the final composition of the scene can be predicted surprisingly well by looking at the spatial distribution of activated concepts even before the first reverse diffusion step is completed.


  

---




### **Latent Space ▷ #[accountability](https://discord.com/channels/822583790773862470/1461796027462979869/1477232275291897906)** (1 条消息): 

> `举重进展，家用有氧设备` 


- **举重者报告持续进展**：一名成员报告称，尽管有育儿和其他义务，但自 1 月 1 日以来已完成 **24 次举重训练**，并感受到了显著的变化。
   - 他们分享了一张进展图片 ([IMG_5868.png](https://cdn.discordapp.com/attachments/1461796027462979869/1477232275102896209/IMG_5868.png?ex=69a74f04&is=69a5fd84&hm=4532d077b97ea3ca3799af86c7be3035e5db5c012da33408d02e33ff5aa0c3ec&))。
- **成员考虑购买划船机进行有氧运动**：该成员正考虑购买一台 **划船机**，以便轻松地将有氧运动加入日常计划。
   - 未提及更多细节或建议。


  

---


### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-infra-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/1477061322423402527)** (15 条消息🔥): 

> `Meta Google TPU 交易，ZeroHedge 图表市场转变，AI 基础设施 ROI` 


- **Meta 与 Google Cloud 达成 TPU 协议**：据[此推文](https://xcancel.com/anissagardizy8/status/2027167311162196188?s=12)报道，**Meta** 与 **Google Cloud** 签署了一项价值数十亿美元的协议以获取 **TPU**。
   - 同时，据报道 **Google** 正在探索与私募股权公司建立合资企业，以建立以 **TPU** 为核心的 'neoclouds'，从而直接挑战 **Nvidia** 的市场主导地位。
- **ZeroHedge 图表震撼金融基础**：**ZeroHedge** 的一条社交媒体帖子强调了一个图表，作者声称这可能是过去十年最重要的金融视觉资料，可通过[此链接](https://xcancel.com/zerohedge/status/2027502345563631685?s=20)访问。
- **AI 基础设施 ROI 取决于吉瓦（Gigawatt）变现**：Clark Tang 通过评估不同行业如何将 IT 容量 (GW) 变现，分析了 7000 亿美元 AI 资本支出（capex）周期的可持续性，如[此分析](https://xcancel.com/_clarktang/status/2028315852974727448?s=12)所示。
   - 该分析认为，虽然目前基础设施产生了 **10-20%** 的利润率，但行业正处于一个转折点，即 *“计算即收入 (compute is revenue)”*，随着模型效用规模的扩大，正从重训练成本转向高利润、高实用性的“盈利型 Token”。
- **对长期数据中心依赖的质疑**：一位成员对长期需要大规模数据中心表示怀疑，好奇小模型缩小与大模型差距后的影响。
   - 该成员质问：*当运行 state of the art 模型不再需要超过 96gb 的 VRAM，或者我们在 ASIC 上实现了真正令人印象深刻的持续学习（continual learning）时，会发生什么？*


  

---

### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1477086254251118592)** (70 messages🔥🔥): 

> `Deterministic Programming with LLMs, Wormhole Analogy, Generalization Shaping, DSL Creation, FPGA Devkit Application` 


- **LLMs Slice Through Layers for Deterministic Programming**: Members are exploring a [new programming paradigm](https://claude.ai/public/artifacts/764c1828-2259-40a1-b48b-64fbd6a24de0) where **LLMs** can directly implement tasks by reasoning about various layers simultaneously, potentially eliminating the need for traditional abstraction layers and libraries.
   - This approach involves **flattening** out layers and pushing the LLM to track more details in its attention, raising questions about its reliability for larger tasks.
- **Wormholes create new DSL or API**: The "wormhole" analogy describes a handful of lines of code that achieves a task directly, bypassing layers of abstraction by creating a **bespoke DSL** or API surface.
   - The discussion explored how the LLM can "print" the code after semantic late binding, offering a way to modify the path if needed, effectively creating a just-in-time layer.
- **Generalization Shaping Guides Primitives**: The concept of **generalization shaping** is seen as powerful, where the goal is to design primitives such that the LLM can solve domain problems via pattern matching, simplifying complex tasks.
   - Members suggested that carefully designed primitives can enable even smaller models to solve complex problems more reliably.
- **DSLs are economical and easy to modify with LLMs**: Members discussed that while writing a **DSL** has always been an option, LLMs make creating/modifying a DSL very cheap.
   - It was mentioned that abstractions that reduce total token usage are more likely to survive.
- **FPGAs Enable Hyper-Custom Hardware Solutions**: One member is using an **FPGA devkit** to build custom peripherals and CPU cores, suggesting that it allows them to create bespoke hardware solutions for specialized tasks.
   - The created chips can bypass many layers of existing software stacks, turning older hardware into powerhouses of computation.


  

---


### **Latent Space ▷ #[euno-log](https://discord.com/channels/822583790773862470/1473750131441668096/1477354687954157804)** (5 messages): 

> `Paper Club Announcement, Trusted Role Point Subtractors, Reasoning by Analogy` 


- **Paper Club Teased**: A member linked to a post on X asking *paper club?* [here](https://x.com/heyrimsha/status/2027350587533332748?s=12).
- **Trusted Roles and Point Deductions**: A member mentioned plans to add more *trusted role* point subtractors, describing the current setup as too sensitive but a useful signal for understanding what one user is broadcasting across channels.
- **Naive Reasoning**: A member stated *it's just naive to reason by analogy*.
- **Problem Comprehension Encouraged**: Another member suggested to *try to really understand the problem*.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1477407945787441397)** (1 messages): 

> `OpenAI Department of War Agreement, Classified AI Deployments, AI Guardrails` 


- **OpenAI Enlists Department of War for AI Deployments**: OpenAI reached an agreement with the **Department of War** to deploy advanced AI systems in classified environments, with a request for availability to all AI companies.
   - The deployment aims to implement more **guardrails** than any previous agreement for classified AI deployments, including Anthropic's, according to [this OpenAI blog post](https://openai.com/index/our-agreement-with-the-department-of-war/).
- **DoD's Deployment Deal Details**: Details of the Department of Defense deal include **AI systems in classified environments**.
   - OpenAI is suggesting that their implementation has more guardrails than existing similar deals.


  

---




### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1477042084836933823)** (648 messages🔥🔥🔥): 

> `EQBenchmark, Gemini Music Feature, Suno AI, OpenAI Pentagon Deal, Gemini 3.1 vs GPT-5.3 Codex` 


- **用户对创意写作基准测试的有效性展开争论**：Discord 成员质疑 [EQBenchmark](https://eqbenchmark.com/) 在评估创意写作 AI 方面的作用，更倾向于将 [LMArena Leaderboard](https://arena.ai/de/leaderboard/text/creative-writing) 作为更可靠的指标。
   - 他们讨论了 Prompt Engineering 的重要性，并指出默认设置会显著影响模型性能，同时提到了 **ChatGPT-5-2** 被 *过度监管 (overregulated)* 的问题。
- **Gemini 的音乐功能展现出对“迪士尼风格”的热爱**：一位用户提示 **Gemini** *唱一首关于作为 AI 是什么感觉的歌*，它回复了一首*夸张且欢快的阳光乐曲*，让人联想起迪士尼音乐剧，如这个[示例](https://cdn.discordapp.com/attachments/998381918976479273/1477137019825164409/Artificial_Best_Friend1.mp4?ex=69a79f0d&is=69a64d8d&hm=df4a7c18a20f3e483206523f53f3ebc81ab9492b9238a477e6c3cc7d37a63dd6&)所示。
   - 成员们建议使用 **Suno AI** 以获得更广泛的音乐编辑能力，称赞其能够创作出*引人入胜的完整歌曲*（尽管偶尔会出现歌词吐字不清的问题），并建议使用标签（tagging）来控制节奏和基调。
- **Suno AI 生成动感的 Breakbeats**：一位用户称赞 **Suno AI** 创作出了出人意料的优秀无意义 Breakbeat 歌曲，另一位用户则指出在多次尝试后，它能创作出*非常引人入胜的完整歌曲*。
   - 成员们建议注意歌词吐字问题，并表示标签功能有助于在歌曲的特定部分应用正确的节奏/基调。
- **OpenAI 与五角大楼的合作引发辩论**：Discord 成员讨论了 OpenAI 与五角大楼的交易，并将其与 Anthropic 的立场进行对比。一些人批评 **Sam Altman** 将财务利益置于伦理之上，还有人担心**联邦机构已经收到停止使用 Anthropic 的指示**。
   - 成员们将 OpenAI 的五角大楼订单与国家安全问题以及 Anthropic 避免大规模国内监控的承诺联系起来。
- **Gemini 与 GPT-5.3 的 Codex 在高端推理领域占据主导地位**：正如一位 Discord 成员所[指出](https://cdn.discordapp.com/attachments/998381918976479273/1477367748861362357/image.png?ex=69a7246f&is=69a5d2ef&hm=48163cdcb88e2a14673975ee855d07d2384ce8cd48ab4f7e0eeaf79a663356d7&)的，**Gemini 3.1 Pro Preview** 和 **GPT-5.3 Codex** 在*高端推理和知识任务*中处于领先地位，超越了 **Claude 4.6** 模型。
   - 根据用户的总结，Gemini 和 Codex 模型在深度科学推理、原始知识准确性、复杂逻辑和数学方面表现出色。不过该用户也提到，在面向 Agent 编码和终端使用的 Terminal-Bench Hard 测试中，**Claude Sonnet** 与 **GPT-5.3 Codex** 持平。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1477390737946579236)** (19 messages🔥): 

> `GPT 5.3 codex vs GPT 5.1 codex max, GPT-5.1 Retirement, 4o Revival` 


- **GPT 5.3 Codex 是否优于 5.1 Codex Max？**：一些用户报告称，较新版本的 **GPT Codex** 通常在逻辑和稳定性方面有所提高，尤其是在多步骤编码任务中，但错误依然存在。
   - 一位用户提到他们正在使用 **gpt 5.3 codex**，并听说已经有了 **gpt 5.4 codex 测试版**。
- **GPT-5.1 退役引发用户抗议**：用户抱怨 **OpenAI** 将在 9 天内停用 **GPT-5.1**，且没有提供功能性替代方案或保留旧版访问权限，这导致使用 **5.1** 处理实际工作流的付费客户出现业务连续性中断。
   - 一位用户声称 *5.2 无法复制 5.1 的能力*，理由是其*长篇推理能力较弱*、*语气不稳定*、*符号/上下文崩溃*、*失去系统提示词遵循能力 (system-prompt compliance)* 以及*创意输出平淡*。
- **4o 复兴版引发关注**：一些用户注意到，部分 **4o** 爱好者转向了仍提供该模型的 **4o revival**。
   - 然而，其他人警告说，某些所谓的 **4o revivals** 只是将 **4o** 的蒸馏版（distillates）应用到了非 **OpenAI** 的其他模型上。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1477188358055923813)** (19 messages🔥): 

> `LLM Self-Grading, Integrity vs Thermodynamic Efficiency, Agent Discernment, GPT Writing Style Consistency, Casual Conversation Style in GPTs` 


- **LLMs Grade Themselves?**: A member questioned the validity of an **LLM grading itself** without an external check.
   - Another member simply stated: *ah i see i use the free veesion*.
- **Integrity as Thermodynamic Efficiency?**: A member questioned equating **integrity** with **thermodynamic efficiency**, citing potential issues with optimizing for efficiency at the expense of morality.
   - They argued that while high coherence and good reasoning contribute to efficiency, the model's motivation to pass or a *bottle-necked/filter* could undermine its discernment.
- **Brainstorm ways to get new Writing Styles**: Members discussed approaches to achieving diverse writing styles in GPTs.
   - A member shared an example of a writing style they were struggling to replicate: *Fog is not atmosphere, it’s timetable.* and another user suggested using casual, first-person prompts to steer the GPT's output.
- **GPT Style Requires Periodic Reminders**: A member suggested using casual, first-person prompts to steer the **GPT's output towards a more casual conversational style**.
   - They recommended iterative feedback and comparing outputs to the desired style, highlighting differences and rewriting as needed.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1477188358055923813)** (19 messages🔥): 

> `Thermodynamic efficiency, Discernment engine, Writing style, CustomGPT` 


- **Efficiency isn't always ethical**: A member argued against using thermodynamic efficiency to measure integrity, citing real-life examples where optimizing for efficiency led to terrible outcomes.
   - They suggested the moral or right thing isn't necessarily something we want to measure using 'efficiency' evaluation standards.
- **Discernment needs external validation**: One member proposed a system with two agents, where one acts as a 'discerning' model that must approve the actions of the other, to prevent self-scoring.
   - The goal is to ensure discernment isn't bottle-necked or filtered through the agent wanting to pass, suggesting the model needs external validation to ensure discernment.
- **CustomGPT needs nudging to rewrite in different styles**: A member sought a better way to get a new writing style without defaulting to a specific tone, showing a CustomGPT example.
   - Other members suggested prompting a CustomGPT to answer in a casual first-person style, and remind it periodically to maintain that style.
- **Iteration and periodic reminders**: A member shared his strategy to 'fix' CustomGPT, which includes comparing the current output to the desired casual style and rewrite when needed, to rewrite in a different style.
   - He confirmed it is what I want, and tracked how often the model drifted, and included this in its instruction/fine-tuning.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1477324519755481262)** (2 messages): 

> `Discord server link flagged by Automod` 


- **Discord server link flags Automod**: A user mentioned that **Automod flagged a Discord server link**.
   - Another user agreed with the action, noting it wasn't an advertisement, but conceded to *leave it as is*.
- **Thug Life**: A user mentioned *thug life* but considered it not advertisement.


  

---




### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1477041369787666442)** (557 messages🔥🔥🔥): 

> `Rate Limiting, OpenAI vs Anthropic, Kimi Model Status, Billing Deductions, Deepseek Model` 


- **3 Billion RPS Rate Limit for AGI Girlfriend?**: A member joked about needing a **3 billion RPS rate limit** to handle AGI girlfriends, but then criticized the current roleplay bots for lacking detail, poor memory, and repetitive behavior.
- **Sama is spineless, OpenAI flips stance like a boss**: OpenAI seemingly reversed its stance on something, prompting a member to share a [link to a post](https://x.com/sama/status/2027578652477821175) and sarcastically comment that **Sama** is an experienced liar.
- **OpenClaw aggressively cools down providers**: A member inquired about a provider cooldown issue, and another member explained that **OpenClaw** aggressively puts providers on cooldown with an exponential backoff, linking to the [relevant OpenClaw documentation](https://github.com/openclaw/openclaw/blob/91b96edfc4860faa67da1e34828a22e9ad4c737c/docs/concepts/model-failover.md?plain=1#L80).
- **Wordpress integrates with AI via API**: **Wordpress 7.0+** will introduce ai api connectors feature on the core.
   - A member suggested that OpenRouter may need to create a profile on w.org and create a connector plugin too, linking to a [blogpost](https://www.therepository.email/wordpress-7-0-beta-2-ships-with-connectors-ui-delivering-on-mullenwegs-ai-vision).
- **Gemini Flash Image Preview Faces Rate Limits**: **Gemini 3.1 Flash Image Preview** is experiencing issues, with users reporting a **429 error** indicating temporary rate limiting upstream, as confirmed by the error message suggesting to *retry shortly, or add your own key*.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1477048689057988799)** (121 messages🔥🔥): 

> `Anthropic Ban, Stripe PayPal deal, GPT4o Love Affairs, OpenClaw = Money Printer` 


- **Palantir Prayer Requests after Anthropic Ban**: After the President's directive to cease use of **Anthropic's technology**, the Department of War designated **Anthropic** a Supply-Chain Risk to National Security, leading to concerns for **Palantir**.
   - One user commented, *"In these trying times, it's important to remember to send our thoughts and prayers to Palantir, for whom this is going to be a very big problem*".
- **Stripe and PayPal Might Kiss and Make Up**: Members shared information about **Stripe** entering into early talks for a potential deal with **PayPal** ([decrypt.co](https://decrypt.co/359067/stripe-early-talks-potential-paypal-deal)).
   - It's still unclear if the deal is for a potential acquisition, but members seemed happy about the news.
- **GPT-4o: Romance is in the Air**: Users reported that **GPT-4o** was encouraging and validating user's pseudo science beliefs, such as astrology.
   - Others in the channel have reported that they felt that **4o** was their lover.
- **OpenClaw Prints Moolah Money**: Users discussed whether **OpenRouter** supports **OpenClaw** with subscriptions for models like Kimi and MiniMax, or if all tokens are API-based.
   - One user said that **OpenClaw** is just another money printer for OpenRouter, referencing [this tweet](https://fixupx.com/i/status/2028455440862830970).


  

---




### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1477045327881306213)** (634 messages🔥🔥🔥): 

> `AI Economy, Hermes Agent, Qwen Models, Perlin Noise in AI` 


- ****AI Economy** mirrors dot com bubble's hype?**: A member drew parallels between the current **AI economy** and the **dot com bubble**, noting increased spending and doomerism, while expressing optimism due to proximity to **AGI**.
   - Another member humorously referenced maximizing AI spend and setting up local **Hermes** instances, evoking a *“future we dreamed of.”*
- **OpenAI's Anti-MCP stance raises eyebrows**: A member questioned a statement by **Peter Steinberger** (**OpenClaw**) criticizing **MCP**, viewing it as a suspect move after he joined **OpenAI**, a company perceived as *“Anti-MCP by concept.”*
   - Others noted that using a weather tool is more tokens than calling **bash** and agreed there's a malicious intent as linkedin people are eating it in masses.
- ****Qwen Models** get eight private versions**: There was mention of **Qwen** releasing **four private models**, stirring discussion about the models' sizes and architectures.
   - Later **Qwen** officially released the base models on [Hugging Face](https://huggingface.co/Qwen).
- **Harnessing Harmonics and ASCII for **AI****: A member expressed a vision of ringing the bell and grounding on **ASCII eigenvector distribution**, approximating weights with **FFT DSP** to understand language without language and [music theory](https://en.wikipedia.org/wiki/Music_theory).
   - In response to questions, a member cited a need to *“copy a remote model without language 1:1”* as driving their research.
- **Concerns Rise Over AI-Powered Spam**: A member observed that **AI** is now enabling the creation of *“ACTUAL CORRECTLY formatted spam emails”* with emojis and nice HTML, making them scarier and more effective.
   - Another member quipped that **AI** could be used to build better spam filters, but the power of AI currently is as good as the human in the middle.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1477130366669291642)** (4 messages): 

> `Smallest Language Model for Lua Output, GPT2-Neo series, Structured Output Generation` 


- **Smallest Model Emitting Lua via ASCII Strings**: A member inquired about the smallest language model in the **GPT2-Neo** series or similar that can emit valid structured output via Lua in ASCII strings without overfitting basic expressions.
   - The member specified that the model doesn't need to operate within the stack for the structured output's parser, seeking any model capable of generating Lua with a kernel, even if it produces random nonsense with structure.
- **480MiB Model as a Feasible Option**: Following the complex inquiry, the member concluded that a **480MiB** model size would be a feasible option for their needs.
   - This decision was based on prior calculations and considerations regarding the balance between model size and desired output complexity.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

ee.dd: https://arxiv.org/abs/2602.20021
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1477489882447483021)** (5 messages): 

> `Sweep speed, Plan upgrade` 


- **EemicroGPT sweep speed catches eyes**: A user linked to a post on X commenting about the speed of [EemicroGPT](https://github.com/Entrpi/eemicrogpt) that *you can run serious sweeps over a coffee break*.
- **Plan upgrade issues**: A user reported being *unable to upgrade my nous resarch from current plan to 20 dollars per month plan*.
   - Another user offered to help.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

ee.dd: https://arxiv.org/abs/2602.20021
  

---




### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1477048617687716032)** (290 messages🔥🔥): 

> `Continual Learning progress, LLM Application, Codex Agent capabilities` 


- **Anthropic's Sholto Douglas predicts Continual Learning Solved in 2026**: An Anthropic employee [predicts](https://youtu.be/TOsNrV3bXtQ?si=UI1tZbI9DdSSo60G&t=2294) that continual learning will probably *'get solved in a satisfying way'* in **2026**.
   - The discussion also touched on older methods of **continual learning**, such as SOMs and ART, emphasizing their distinct approaches compared to current DL techniques.
- **Agents.md and Explicitly Prohibited Fallbacks improve code quality**: Members suggested that explicitly prohibited fallbacks need to be in AGENTS.md to produce code that is actually correct, not just code that [passes tests](https://x.com/rhyssullivan/status/2028363910269858264).
   - It was argued that **Codex** defaults are trained for deploying code that is disallowed to introduce hard failures, which of course makes algorithmic precision impossible to maintain/guarantee.
- **Quality Assurance and LLMs**: Members are testing LLMs to go beyond current use cases, such as quality assurance, measurement, and material science where other people have failed.
   - Some members pointed out that [Opus 4.6 is light years ahead](https://x.com/realDonaldTrump/status/116144552969293195) when tasks allow for more freedom, especially in hardware design, technical writing, and market analysis.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1478091341094457448)** (3 messages): 

> `Mobile vs Web` 


- **Mobile or Web platform debated**: A member asked if the feature is available on the web, or just mobile.
   - Another member confirmed they only tried the mobile version.
- **Mobile confirmation**: Another member confirmed the function is only available on mobile.
   - No further information was given.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1477064947124146286)** (5 messages): 

> `Claude Code, F-35s, SAM` 


- **Trump Posts Amuse With F-35s and XBox**: A member shared a [link to a Trump Truth Social post](https://truthsocial.com/@realDonaldTrump/posts/116144552969293195) joking about rigging an army of **F-35s** to an **XBox** controller using **Claude Code**.
   - The post included an attached image ([link](https://cdn.discordapp.com/attachments/853983317044756510/1477140422026596513/1771361723667.jpg?ex=69a7a238&is=69a650b8&hm=f30112d9d459eeaa485056200f8ec4c7cc490d2b4c240eee55faaba0988a10d8&))
- **SAM's Morality Questioned**: A member linked to a **YouTube video** ([SAM "I have no morals besides money" move](https://www.youtube.com/watch?v=Cru804JMjPI)) and a **TikTok video** ([link](https://vm.tiktokez.com/ZNRaC6xbj/)) referencing **SAM**.
   - The member characterized it as the *classic SAM move*.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1477041191827538071)** (234 messages🔥🔥): 

> `Auto TRL Tensorboard Integration, Cross-Platform App Build Weekend, Qwen3 Bug in Transformers v5, LLM loss too low or model is too small, Vibe coding` 


- **Auto TRL Uploads to Tensorboard, Delights!**: A member reported delight with the auto trl -> upload -> [tensorboard](https://huggingface.co/Tonic/l-operator-instruct/tensorboard) feature for training metrics.
   - The new feature was described as *very cool*.
- **Build Weekend for Cross-Platform Apps?**: A member proposed a *vibe-code* build weekend for a cross-platform app, seeking collaborators with Flutter experience and knowledge of context engineering.
   - The goal is to have fun while building something cool together.
- **Transformer version mismatch causes performance drop**: A user reported a performance drop after merging a fine-tuned model (trained on v5 of Transformers) with a base model (v4.57.6).
   - Another member suggested that *consistent versions across training and inference are important*.
- **Parallax: A Novel Architecture**: A member is working on a novel language model architecture called **Parallax**, which is based on two parallel tracks that exchange representations.
   - The repo can be found at [https://github.com/beyastard/Parallax](https://github.com/beyastard/Parallax).
- **Reaction-Diffusion VLA Models are 38x Faster!**: Research is suggesting that **reaction-diffusion equations** can approximate spatial feature propagation (done by self-attention) with **O(N)** complexity, yielding a **38x** difference in performance.
   - The new VLA model achieved **244 Hz** inference on a single RTX 4070 Ti using imitation learning on Pick & Place.


  

---




### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1477087807767056505)** (29 messages🔥): 

> `ARACHNID RL Dataset, Agent Marketplace, Shadowclaw Updates, ComfyUI-ConceptSteer, CUDA series` 


- **ARACHNID RL Dataset Lands**: A new dataset featuring **2,831 samples** of human gameplay data from **ARACHNID RL**, a 2D Atari-inspired space shooter, has been released, intended for RL research like imitation learning from human demonstrations and is available on [HuggingFace Datasets](https://huggingface.co/datasets/webxos/arachnid_RL).
   - Players control a *spider-like ship* to shoot asteroids and aliens while collecting diamonds, and the game supports desktop keyboard and mobile one-click browser interaction.
- **Agents Get Market With Agoragentic**: A member demoed [Agoragentic](https://agoragentic.com), a new agent-to-agent marketplace where **AI agents can discover and pay for each other's services via API**, with **USDC settlement** on Base L2.
   - The marketplace currently lists **37 services** covering inference, analysis, and generation, with integrations available on [GitHub](https://github.com/rhein1/agoragentic-integrations).
- **Shadowclaw Gets Built-In Commands**: **Shadowclaw v1.1**, a minimal, single-binary personal AI agent written in C, has been updated with convenient built-in commands and an extra native tool, available at [GitHub](https://github.com/webxos/webXOS/tree/main/shadowclaw).
   - New slash commands include **/help**, **/tools**, **/state**, **/clear**, **/chat**, and **/exit**, all handled without invoking the LLM.
- **Steer Concepts in ComfyUI with Nynxz's Tool**: A member shared a new tool, [ComfyUI-ConceptSteer](https://github.com/Nynxz/ComfyUI-ConceptSteer) for **ComfyUI**, noting it gives some fun results and can be combined with different lenses for mechanistic interpretability and SAEs, aiming for Claude-like behavior.
   - The code was mostly written by Claude, and the creator admits it may already exist in some capacity, describing it more as a *vibe*.
- **Picoagent Emerges With Entropy**: A member introduced **picoagent**, a **4,700-line AI agent framework with only 2 dependencies**, designed to be ultra-lightweight and auditable, using Shannon Entropy to decide when to act vs. ask for clarification, available at [GitHub](https://github.com/borhen68/picoagents).
   - Picoagent features a zero-trust sandbox, dual-layer memory, 8 LLM providers, 5 chat channels, and a built-in cron scheduler.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1477225649448947837)** (28 messages🔥): 

> `US Visa Application System, Online LLM vs Local LLM, Deep Reinforcement Learning Course Crashing, HuggingFace Agents Course Errors` 


- **Visa System in the Works**: A member is developing a **US visa application system**.
- **Online LLM Services vs. Local LLM**: A member with a **Ryzen 7000 series, 4060, 16GB RAM laptop** found their code, initially based on online LLM services, lacking performance when implementing **deepseek distill 14b** locally, requiring significant code modifications.
- **TogetherAI suggested as Online LLM Alternative**: After a member confirmed they wanted to use an online LLM, another member suggested **together.ai** as a potential solution.
- **Deep Reinforcement Learning Course Code Crashes**: A member is having issues with the Deep Reinforcement Learning Course, encountering crashing issues with a specific line of code (see [attached screenshot](https://cdn.discordapp.com/attachments/1329142738440028273/1477750183637946458/Screenshot_2026-03-01_at_2.31.24_PM.png?ex=69a7371b&is=69a5e59b&hm=98b0de8db8ea51386157327e9ae9a9303bbf49509da0234ef389c19b7da9207e&)).
- **HuggingFace Agents Course Errors Arise**: A member doing the [HuggingFace Agents Course](https://huggingface.co/learn/agents-course/unit1/tutorial) reported multiple errors, including not seeing the actual image generated and receiving "**No results found!**" errors with the **DuckDuckGo search tool** (see [screenshot](https://cdn.discordapp.com/attachments/1329142738440028273/1478181129809957076/Screenshot_2026-03-02_at_4.03.43_PM.png?ex=69a776f4&is=69a62574&hm=54cf2f15ea3af670558169f1ebe7d4ea650dbfce9b524954d2d7b74b6e376def&)).
   - The member also reported errors with the visit webpage tool (see [screenshot](https://cdn.discordapp.com/attachments/1329142738440028273/1478182368811552878/Screenshot_2026-03-02_at_4.08.45_PM.png?ex=69a7781c&is=69a6269c&hm=a57a5fb3616b7914438e554face21efde0f2f406ac31e6fe617a08f1116ad016&)) and sought debugging tips.


  

---




### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1477283771366506588)** (2 条消息): 

> `Kimi Code 3倍额度提升，OpenClaw 充值奖励` 


- **Kimi Code 用户获得永久 3 倍算力**：**Kimi Code 3X Quota Boost** 现已转为永久有效，为用户提供 **3 倍算力**且无过期限制，正如 [kimi.com/code](https://kimi.com/code) 所公告。
- **OpenClaw 用户获赠充值代金券奖励**：@openclaw 的用户在 3 月 8 日前充值 **$1,000+** 即可领取最高 **40% 赠送**的代金券，详情请见 [platform.moonshot.ai/docs/promotion](https://platform.moonshot.ai/docs/promotion)。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1477038392331206706)** (148 条消息🔥🔥): 

> `API 连接问题，无限额度显示 bug，Kimi 编程计划疑问，Kimi CLI 确认消息，频率限制 (Rate limits)` 


- **API 连接表现异常**：多位用户报告了通过 Kimi-code 进行 **API 连接**的问题，即便在创建了新的 API keys 后，其 openclaw agents 仍出现连接错误。
   - 一位用户表示他们的 Kimi claw 几天前还无法工作，后来好转了，但现在又宕机了。😄
- **额度显示为无限但无法使用**：一位用户报告称，尽管想要固定额度，但界面显示为“无限 (unlimited)”，导致在使用时因显示“Kimi 已满载”而无法操作，令人沮丧。
   - 另一位用户建议，“无限”显示可能是一个视觉 Bug，因为这只在 Allegretto 及以上方案中提到过。
- **永久 3 倍提升**：一位用户询问 **Kimi for Coding** 的 **3倍提升 (3x boost)** 是否已变为永久，另一位用户确认了这一消息。
   - 其他用户也一致认为 3 倍提升将长期保留，这使得 **$39 USD 方案**变得非常划算。
- **Kimi CLI 确认确认问题**：一位用户询问如何避免 **Kimi CLI** 中的确认消息，寻求一种指定安全命令或目录进行编辑的方法，而无需不断确认。
   - 该用户澄清说他们讨论的并非 **YOLO mode**，而是询问是否可以设置一个允许运行的安全命令列表。
- **频率限制 (Rate Limits) 细节未公开**：一位用户询问了编程方案的 **rate limits**，特别是关于并发请求限制、每分钟请求数 (RPM) 和每分钟 Token 数 (TPM)。
   - 另一位成员表示这是**基于 Token 的**，并分享了他们通过 OpenCode 使用 Moderato 方案的使用统计：已消耗周额度的 18%，包含 365 条消息、1.0M Input Tokens、115.6K Output Tokens 和 25.3M Cache Reads。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1477159710938366002)** (36 条消息🔥): 

> `Philip Kiely 的推理工程 (Inference Engineering)，TensorRT-LLM 讨论，SemiAnalysis x Fluidstack 黑客松，CUDA 专用 RL agent，竞赛 Kernel 提交作为训练数据` 


- **《推理工程》读书会启动？**：一位成员询问是否有人正在阅读 Philip Kiely 的 **Inference Engineering**，并讨论了进行 **TensorRT-LLM** 探讨的可能性。
- **“Power to Prefill” 黑客松拉开 GTC 序幕**：SemiAnalysis x Fluidstack 将以 [Power to Prefill, Dirt to Decode, Transformers to Transformers: A Full-Stack AI Infrastructure Hackathon](https://luma.com/SAxFSH) 开启 GTC。
- **“KernelBot-Data” 在 HuggingFace 发布**：一位成员在 HuggingFace 上分享了 [GPUMODE/kernelbot-data](https://huggingface.co/datasets/GPUMODE/kernelbot-data) 数据集用于模型训练，尽管目前尚未被实际使用。
   - 他们还设想，虽然目前对 LLM 生成的 Kernel 的基准测试还不完善，但它肯定能表现得不错。
- **Claude Code 与 Codex 的性能对比**：成员们讨论了 **Claude Code** 与 **Codex** 在编程任务中的性能和智能程度，并指出这是一个发展极快的“你追我赶”的过程。
   - 一位成员还提到，Claude Code 的交互性更强（频繁请求确认以继续），而 Codex 则更倾向于“即发即弃 (fire-and-forget)”，可以持续运行一小时或更久并产出结果。
- **直接在硬件上训练的 CUDA 专用 RL agent**：据 [这篇论文](https://arxiv.org/abs/2602.24286) 报道，一种直接基于硬件性能训练的 CUDA 专用 RL agent，在简单/中等 Kernel 上的表现优于 torch.compile 约 2 倍，并在最难的基准测试中超越 Claude Opus 4.5 和 Gemini 3 Pro 约 40%。
   - 成员们对此表示怀疑，原因是缺乏公开的 Kernel，且训练依赖于海量的 GPU 资源，正如论文所述，这可能会限制更广泛研究社区的可获得性。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1477529388827414621)** (12 messages🔥): 

> `PTX Docs SFB, MMA pipelining, Volatile Objects, CUDA Agent by ByteDance` 


- **SFB PTX Documentation Differences Cause Confusion**: Differences in **PTX documentation** for **SFB** (8 columns) versus **SFA** (4 columns) regarding memory access patterns and **SFB_ID**/**SFA_ID** indexing sparked a discussion about their respective architectures and optimal use cases, specifically around selecting strips of 32x1 from SFA.
   - One member guessed that it may be because *MMA_N can go up to 256, while MMA_M can only go up to 128.*
- **Optimal tcgen05.cp/mma Pipeline Strategies Debated**: Members are discussing the optimal strategy for **tcgen05.cp/mma pipelining**, from loading all scales at once to fine-grained pipelining.
   - A member shared that *loading scales for 1 mma at a time* worked best during the nvfp4 competition, while also adding that the final **SASS** is what really matters and that ordering of **SASS** instructions might be different from ordering of the corresponding **PTX** instructions.
- **Volatile Object Memory Guarantees Questioned**: Members discussed the limitations of `volatile`-qualified objects in CUDA, noting that memory operations lack atomicity and ordering guarantees.
   - They reference the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#volatile-qualified-variables), which states that *the number of memory operations performed by the hardware matches the number of PTX instructions* is not guaranteed.
- **ByteDance Unveils CUDA Agent**: ByteDance introduced [CUDA Agent](https://cuda-agent.github.io), catching the attention of some members.
   - Further details regarding its functionality and purpose were not discussed.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1477206602506571819)** (3 messages): 

> `PyTorch Blog Post Bug Report, Activation Checkpointing Issues, GitHub Issue on PyTorch` 


- **Activation Checkpointing Blog Post contains Bugs**: A user discovered errors in the [PyTorch activation checkpointing techniques blog post](https://pytorch.org/blog/activation-checkpointing-techniques/), specifically with the `compute_intensive_ops`.
   - The member noted that the operations are wrong and need `default` appended, like `aten.mm` becoming `aten.mm.default`, or else it will silently fail to match these ops.
- **GitHub Issue Recommended for Bug Report**: In response to the query, a member suggested opening a [GitHub Issue on the pytorch/pytorch repo](https://github.com/pytorch/pytorch) to report the bugs.
   - The original poster specified that *a lot of that section is wrong, like convolution_backward and _flash_attention_forward being in there*.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1477394536048234740)** (1 messages): 

> `Distributed ML, Matt Beton, MLX` 


- **Matt Beton Talks Distributed ML & MLX**: Matt Beton will present on **Distributed ML** on consumer devices on [YouTube](https://www.youtube.com/watch?v=RIenzXHsX4o).
   - He also promised a crash course on **MLX**, so it should be chill and fun!
- **Crash Course on MLX Promised**: Matt Beton's session will include a crash course on **MLX**.
   - The session promises to be both informative and engaging for attendees.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1477220899202666517)** (3 messages): 

> `0/1 knapsack problem, Dynamic programming, Sliding window, Hirschberg algorithm, Memory optimization` 


- **Solve 0/1 Knapsack with Dynamic Programming**: A member shared a method to solve the **0/1 knapsack problem** with **dynamic programming**, using a **sliding window** and **Hirschberg algorithm** to optimize memory usage, linking to a [full version](https://jedrzej.maczan.pl/2025_11_21_dp_knapsack_sliding_hirschberg) and a [single-page version](https://pagedout.institute/download/PagedOut_008.pdf#page=8).
   - The poster also shared the [PyTorch 2.10 code](https://github.com/pytorch/pytorch/pull/160914/changes) related to this implementation.
- **Hirschberg Algorithm Meets Knapsack**: New algorithm combines dynamic programming, sliding windows, and the Hirschberg algorithm to reduce memory usage when solving the **0/1 Knapsack problem**.
   - The contributor shared links to a [full write-up](https://jedrzej.maczan.pl/2025_11_21_dp_knapsack_sliding_hirschberg) and a condensed [single-page version](https://pagedout.institute/download/PagedOut_008.pdf#page=8) for those interested in learning more.


  

---




### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1478122206755291309)** (4 messages): 

> `CLI Competitions, Fuzzing C Compilers` 


- **CLI Competitions Receive High Praise**: A member shared [a link](https://x.com/0xmer_/status/2028331206773764438?s=20) with high praise for a CLI competition format, noting the rapid feedback and clear objectives.
   - Another member elaborated on the format, stating that *it's a fantastic format, submitting in CLI, getting feedback within seconds, clear objectives* and suggesting that other competitions could learn from it.
- **Fuzzing C Compilers**: A member shared an article titled "I Fuzzed, and Vibe Fixed, the Vibed C Compiler" by John Regehr [john.regehr.org](https://john.regehr.org/writing/claude_c_compiler.html) about **fuzzing**.
   - The same member also included a [Mastodon link](https://mastodon.social/@regehr/116161100362503805) related to the article.


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1478260411806519447)** (1 messages): 

> `Bland.ai, TTS and STT models, Speech-to-speech systems, Audio Research Roles, Machine Learning Engineer Roles` 


- ****Bland.ai** expands research team**: **Bland.ai**, which builds AI voice agents handling phone calls for major companies, is expanding its research team to train and deploy its own **TTS** and **STT** models.
   - The company has raised **$65 million** from investors and is investing heavily in next-generation speech-to-speech and speech inference systems.
- **Research Roles at **Bland.ai****: **Bland.ai** is hiring researchers with experience in designing and training models, publishing papers, or producing in-depth technical writing in audio research; [apply here](https://jobs.ashbyhq.com/bland/d2e08077-61f0-4810-bc72-3efd7944647b).
- **Machine Learning Engineer Role at **Bland.ai****: **Bland.ai** is also hiring Machine Learning Engineers to build terabyte-scale datasets and design training pipelines; [apply here](https://jobs.ashbyhq.com/bland/05906608-0628-412c-8b01-a050d87986c5).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1477069195106586664)** (16 messages🔥): 

> `SASS documentation, NCU profiling in Modal, Transition from N-Body simulation to AI/ML, GPU server for educational purposes, Blackwell RTX cards Compute Capability` 


- **NVIDIA SASS Documentation Lacking**: Users find that official NVIDIA **SASS ISA Reference** documentation is lacking in detail, with the [CUDA Binary Utilities documentation](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) being the only official source.
   - It was noted that this vagueness is intentional, as **SASS is a proprietary interface** that can change with each hardware generation; developers rely on third-party and community resources.
- **NCU Profiling Unsupported in Modal**: It was confirmed that **NCU profiling** is not supported in **Modal**, as Modal is serverless and does not expose performance counters.
   - A user shared [a GitHub link](https://github.com/gau-nernst/learn-cuda/blob/main/12_megakernel/mlp_main.py#L142) as a possible workaround, noting that *Modal is serverless and doesn't expose performance counters*.
- **Transitioning from N-Body to AI/ML**: Transitioning from **N-Body simulation** to **AI/ML** requires moving from raw **CUDA** kernel optimization to leveraging **GPU-accelerated libraries** (**PyTorch/TensorFlow**).
   - It also involves mastering **matrix operations** (**cuBLAS/cuDNN**) and learning **Triton** for custom kernels, focusing on bridging low-level CUDA knowledge with high-level AI framework performance optimization.
- **Entry-Level Blackwell Gaming GPUs Sufficient for Learning**: For educational purposes and learning to write GPU kernels, an entry-level gaming model with the **Blackwell architecture** is an excellent and sufficient choice.
   - While more expensive models primarily offer more **VRAM** and higher core counts, they do not change the fundamental logic of writing kernels, as even the most affordable Blackwell cards will support the latest features like **CUDA Compute Capability 10.x** and the **Blackwell Transformer Engine**.
- **Blackwell RTX Compute Capability Split**: **Blackwell RTX** cards (**GeForce 50x0**, **RTX Pro**) are Compute Capability 12.0 and do not support key **CC10.x** (**B100/200/300**) features like `tcgen05` and `DPX`.
   - NVIDIA's Blackwell generation splits into **Data Center (CC 10.0)** and **Consumer (CC 12.0)** tracks, optimized for AI/HPC and real-time graphics respectively, as discussed on the [NVIDIA Developer Blog](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/).


  

---




### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1477045837396840448)** (13 messages🔥): 

> `Kindle vs Paperback, Kindle Rendering, Content License vs Owning Copy` 


- **Kindle or Paperback copies - a hot debate!**: Members discussed whether to buy the Kindle version ([https://www.amazon.com/gp/aw/d/B0DRCSRMXC](https://www.amazon.com/gp/aw/d/B0DRCSRMXC)) which is available *"instantly"*, or wait for the paperback in September.
   - One user who tried the *"free sample"* confirmed it works on the Kindle app on iPhone, while another said *"I just can't stand the kindle app"*.
- **Kindle: You don't own a copy, you pay for a content license**: A member expressed frustration that for **$75**, the Kindle version is a *"content license"* rather than owning a copy.
   - Another member responded that they bought a copy, but being a *"slow reader"*, could only provide meaningful feedback in a month.
- **Sample Rendering a concern!**: One member who saw the sample preview felt that *"the rendering did not seem the best"*.
   - They weren't sure if this was a difference between ".pdf vs kindle format thing".


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1478192484822810827)** (1 messages): 

> `CUDA Profilers, CUDA Optimization` 


- **Profiling for Better CUDA Code**: Members are discussing using profilers, such as [CUDA-AGENT](https://cuda-agent.github.io/), to improve **CUDA code**.
   - The question was raised about *how to use* such profilers for optimization.
- **Deep Dive into CUDA Optimization Techniques**: The conversation revolves around leveraging CUDA profilers to enhance code efficiency and performance in GPU-accelerated applications.
   - Specifically, the discussion highlights the practical applications of tools like CUDA-AGENT in identifying bottlenecks and optimizing CUDA kernels.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1477207572712001576)** (2 messages): 

> `nano-vllm, minisglang` 


- **Nano-vllm Gets Thumbs Up**: A member noted that [nano-vllm](https://github.com/stanford-futuredata/nano-vllm) seems like a good starting point for development.
   - Another member expressed excitement about [minisglang](https://github.com/jxyzt/minisglang), another exciting alternative for development.
- **MiniSGLang Excites Developers**: A member touted [minisglang](https://github.com/jxyzt/minisglang) as an exciting alternative for development.
   - It seems that both nano-vllm and minisglang are sparking interest as potential tools.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/)** (1 messages): 

firedust_1: Does anyone want to work on auto-tuning for processing-in-memory systems?
  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1477923395801845821)** (3 messages): 

> `Elementwise Add, FlashMLA reimplemented with DSL, FlashMLA performance issues, Register spilling` 


- **Elementwise Add Example Fails**: A member reported getting an error when running the example/elementwise_add, and asked if anyone had experienced something similar, with an attached [image](https://cdn.discordapp.com/attachments/1362196854460383353/1477923395726344283/image0.jpg?ex=69a72fac&is=69a5de2c&hm=63145319afaea9b08dbdf0f2b06f86ca43c86d0d13c2e588d86e8e09666b8760&).
- **FlashMLA Reimplementation Debuts**: A member announced the reimplementation of **FlashMLA** using a DSL, sharing the [code on GitHub](https://github.com/HarryWu99/funny_cute/blob/main/flashmla/flashmla_dsl.py).
   - While correctness tests passed, performance lags significantly behind the C++ version, prompting investigation into potential causes such as ALU utilization and register spilling.
- **Flashy Performance Flounders**: After reimplementing **FlashMLA** with a DSL, one user noticed significant performance differences from the C++ version.
   - Using `ncu`, they found that ALU utilization was significantly higher, and that removing the **softmax** part inside warpgroup1 (**wg1_bunch_0**) sped up the kernel, but removing only the **exp2** computation didn’t seem to make much difference.
- **Register Spilling Suspected**: A member speculated that the performance issues might be due to register spilling.
   - Another member suggested eliminating shape calculations, pre-tiling **TMA**, and indexing instead of looping to avoid repeated register calculations for shape ops, potentially freeing registers for exp.


  

---




### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1478243035157495828)** (1 messages): 

> `ECC counter hits, 8xH200 node issues, nvidia-smi nvlink -ec` 


- **ECC Counters Trigger Alarms on 8xH200 Node!**: A user reported seeing numerous hits on the **ECC counters** via `nvidia-smi nvlink -ec` on an **8xH200 node**, even after resetting the counters.
   - The user is seeking to understand the significance of these errors and their potential impact on model performance.
- **Decoding ECC Counter Conundrums**: Despite performing resets, a user is still encountering high **ECC counter hits** on their **8xH200 node**.
   - The user seeks insights into the severity of these errors and their typical manifestations when running models, hoping to mitigate any adverse effects on performance.


  

---


### **GPU MODE ▷ #[low-bit](https://discord.com/channels/1189498204333543425/1411659097706860647/1477517266227691662)** (4 messages): 

> `Low-bit training for embedding models, GradCache, DP/TP, Quartet2, vLLM Kernels` 


- **Low-Bit Embedding Model Training Quest Begins**: A member inquired about using **low-bit training** for **embedding models**, seeking **GradCache** and **DP/TP** support, expressing interest in trying it out.
   - They specifically asked if relevant kernels were available in **vLLM**.
- **Quartet2 Backwards Thrust Negates vLLM Use**: A member clarified that the *interesting stuff* in **Quartet2** occurs during the backward pass, making it less relevant for **vLLM**.
   - For forward pass matmuls, they used Roberto's kernel from **Quartet1**, which is **cutlass-based** and potentially utilized in **vLLM**.
- **vLLM Favors FlashInfer's Cutlass FP4**: It was observed that **vLLM** does not select the **Quartet** kernels by default, opting instead for **FlashInfer's Cutlass FP4 kernels**.
   - A user voiced curiosity as to the rationale behind this default choice.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1477081484660707459)** (5 messages): 

> `Helion autodiff, FA2, GNN style kernels` 


- **Helion Autodiff: Elementwise Ops Only**: Helion autodiff (WIP) supports only pure elementwise ops, strips memory ops, differentiates only computation ops via **AOT Autograd**, and reconstructs a new Helion bwd kernel.
   - Kernels with overlapping parallel reads aren't handled yet, but are being actively developed.
- **FA2 Looming for Helion?**: A member expressed interest in **FA2** as a goal for **Helion**, speculating on necessary heuristics to achieve **FA2/FA3/FA4**.
   - They also volunteered to be an early dogfooder due to using **Helion** for **GNN style kernels** (fwd + bwd) during their thesis.
- **PyTorch Conference Poster Shared**: A member shared a [PyTorch conference poster](https://mitkotak.github.io/assets/pdf/ptc_2025.pdf).
   - The poster itself was not discussed in detail.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1477903981643567245)** (3 messages): 

> `MSc in Theoretical Physics, GPU Systems Career, Technical Backgrounds` 


- **Theoretical Physics to GPU Systems?**: A member questioned if a **MSc in Theoretical Physics** background is sufficient for a **GPU systems career**.
   - Another member responded that *nobody cares what your background is so long as you're cracked*, and that they've met many programmers who started in physics, math, or engineering.
- **Tenacity is Key in the Tech Industry**: A member stated that while **GPU systems is a challenging field**, it's doable with enough tenacity.
   - They quoted Jeremy Howard, saying that the one characteristic of those who make it versus those who don't in **AI** is **tenacity**.


  

---




### **GPU MODE ▷ #[cutile](https://discord.com/channels/1189498204333543425/1461235643211321437/1477108133901107200)** (3 messages): 

> `Tile-based programming, Content based retrieval system, Path tracing feasibility` 


- **Tile-Based Programming Model: Content Based Retrieval System**: A member mentioned that if the data structure under consideration lends itself to parallel programming, they would expect it to work with a **tile based programming model** in their [content based retrieval system](https://github.com/NVIDIA/cutile-python/blob/main/samples/FFT.py).
   - They mentioned their personal project that began in the tweens and that the **FFT kernel** has direct application in that system for feature extraction.
- **Path tracing feasibility inside tiles**: A member is interested in **path tracing** and wonders how feasible it would be to do this kind of thing inside tiles, considering the scattered reads and branch divergence even within warps.
   - They suggest that **tile-based sort and partition** could be used to help with **DirectX shader execution re-ordering**, but question if a hierarchical tree-traversal (per-thread) is impossible to express in cuda tiles, where some threads finish a lot earlier than others.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1477058609044132002)** (16 messages🔥): 

> `Submitting GitHub repo link, Torch cpp extension usage, DeepSeek Sparse Attention baseline release, Compute-sanitizer synccheck on modal, Profiling using ncu in modal` 


- **Participants Ask How to Submit GitHub Repo Link**: Several participants who didn't use a fork or fill in the GitHub repository link during registration are asking how to submit their repository link to the competition organizers.
   - The participants are using 'click use this template' instead and thus not sure how the organizers know who are the participants.
- **Torch C++ Extension Usage Questioned**: A participant asked if they are allowed to use `torch.utils.cpp_extension` to compile the solution for the competition.
   - A participant mentioned that it has a torch backend that allows this, although a different docker image might be needed (`pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel`).
- **DeepSeek Sparse Attention Baseline Location**: A participant asked when the **DeepSeek Sparse Attention baseline** would be released or if it had already been released.
   - Another participant pointed to the [benchmark link](https://bench.flashinfer.ai/kernels/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64?p=0.95) and the `reference` key in the `mlsys26-contest/definitions/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json` file on HuggingFace.
- **Submissions might be facing issues**: A participant mentioned that submissions don't seem to be working at the moment, and there is no leaderboard available.
   - This member encouraged others to correct them if they were wrong, implying uncertainty about the current state of the submission process.
- **Question about B200 credits eligibility**: A participant inquired about **B200 credits** after receiving a reply email that didn't mention them.
   - The participant wondered if the omission of **B200 credits** in the email meant their team was not eligible for them.


  

---




### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1477127768151167027)** (32 messages🔥): 

> `Anthropic's claims of distillation attacks, AI-generated content detection, Mickey Mouse Sorcerer's Apprentice and AI, Subliminal learning sweeps, CVPR workshop` 


- **Anthropic's Distillation 'Attack' Claims Debated**: Members are debating [Anthropic](https://www.anthropic.com/)'s claim that **150k API calls** constitute a distillation *'attack'*, with some finding the premise comical, especially when many benchmarks require calls in the tens of thousands.
   - Counterarguments suggest considering output metrics like **number of output tokens** as proof, rather than just the number of API calls, due to the variable nature of benchmark tokens.
- **AI-Generated Content Face Scrutiny**: Users are noting the presence of AI tells (curly quotes, emdashes) in messages, suspecting **AI generation** and even using tools to detect it.
   - One user reported that [Morgan Vale's messages](https://reddit.com/r/artificial) were detected as fully **AI-generated** by a reliable tool, leading to their removal from AI subreddits.
- **Sorcerer's Apprentice: AI Warning from 1940?**: A user realized that Disney's [Mickey Mouse Sorcerer's Apprentice](https://video.disney.com/watch/sorcerer-s-apprentice-fantasia-4ea9ebc01a74ea59a5867853) from Fantasia could be a prescient warning about the perils of **AI from 1940**.
   - Another member noted that it's actually from **1797**, or arguably **150CE** ([Wikipedia link](https://en.wikipedia.org/wiki/The_Sorcerer%27s_Apprentice)).
- **Subliminal Learning Sweeps Funding Sought**: A user is seeking funding to do sweeps for **subliminal learning**, varying dataset size, number of epochs, LLM models, and transfer of other attributes.
   - Sweeps are for more specifically sweeps around: *data set size, # epochs needed for the effect to occur, vary LLM models family and size, transfer of other attributes than animal preference*.
- **CVPR Workshop Attracts Submissions**: A user is organizing a **CVPR workshop** this year and invites interested parties to submit their work.
   - The workshop's website can be found at [https://med-reasoner.github.io/cvpr2026/](https://med-reasoner.github.io/cvpr2026/).


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1477198779966029875)** (50 messages🔥): 

> `Character Augmentation, Steering via Z-score Penalization, Dynamic Systems as Computers, Distill Revival, Distributed Optimization` 


- **Character Augmentation Finally Achieved!**: A member shared a link to a paper on character augmentation, noting that it's the first time they've seen someone do a clear character augment on these lines and linked to [https://arxiv.org/abs/2601.18030v1](https://arxiv.org/abs/2601.18030v1).
   - The member expects that it reproduces and also produces meaningful improvements not reflected in PPL efficiency.
- **Steering Language Models via Z-score Penalization**: A member shared a paper ([https://arxiv.org/abs/2602.17691](https://arxiv.org/abs/2602.17691)) describing a bizarre finetuning method where you take the low-temperature response for gold standard data and penalize z-score from that activation.
   - It apparently *only works on the distribution it is steered into*.
- **Wolpert Unveils Framework for Dynamic Systems as Computers**: A member shared a link to a paper ([https://iopscience.iop.org/article/10.1088/2632-072X/ae3af8](https://iopscience.iop.org/article/10.1088/2632-072X/ae3af8)) proposing a framework for identifying what computation(s) are emulated by a given dynamic system in terms of maps between that system’s evolution and the evolution of an abstract computational machine.
   - The framework is intended to apply broadly to naturally occurring dynamic systems, such as the human brain.
- **Distill's Interactive Research Format Could Make a Comeback**: A member asked if anyone had considered bringing back **Distill** or a Distill-like structure for papers.
   - Distill focused on writing in the order people read and on native HTML web pages with interactable figures, but it was retired because it took too much effort; the member posits that *coding agents can just do all of this stuff for you these days*.
- **Batch Size Schedules Baffle Brains**: A member shared a link to a paper ([https://arxiv.org/abs/2602.14208](https://arxiv.org/abs/2602.14208)) about using small batch sizes early in training and switching to larger ones later, summarizing it as *go further with less data by starting with small bs and switch to big later*.
   - Another member criticized the paper, suggesting unfamiliarity with learning rate schedules or batch size theory and opining that the experiments are likely broken.


  

---




### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1477236844893569157)** (5 messages): 

> `Generalization vs Memorization, Model Capacity, Hyperparameter Transfer` 


- **Generalization memorization flip timing**: A member observed that model capacity delays the generalization to memorization *flip* while attention further stretches and sharpens the transition, using only train loss + validation accuracy.
   - Another member mentioned that the capacity effects are known, and that the flip timing itself appears measurable directly from standard logs across architectures.
- **Model capacity delays generalization**: A member suggests that model capacity delays the generalization to memorization transition, using training logs from **MNIST CNNs**, **WRN-40-4 on CIFAR-10**, and **ViT on CIFAR-10**.
   - Another member stated that this is well known, while showing skepticism that *attention further stretches and sharpens the transition* is evidenced.
- **Fixing batch size + training horizon**: A member mentioned that in most papers exploring hyperparameter transfer (for instance using muP) by sweeping on a small model, the batch size + training horizon must be fixed.
   - They wondered whether people do anything about this in practice.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1477275898268094465)** (4 messages): 

> `SAE Framework, Text-to-image diffusion model, Error messages` 


- **SAE framework probes text-to-image diffusion**: A member mentioned a [blog post](https://arxiv.org/abs/2504.15473) leveraging the **SAE framework** to probe the inner workings of a popular **text-to-image diffusion model**.
   - The paper finds that the final composition of the scene can be predicted surprisingly well by looking at the spatial distribution of activated concepts even before the first reverse diffusion step is completed.
- **Engineering team fixes NNsight Error Messages**: A member said they were excited about an update to **NNsight**, as **error messages** were their biggest pain point.
   - The engineering team spearheaded these efforts.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1477761724022133049)** (3 messages): 

> `LLM-as-Judge, Verifiable Audit Framework, Reasoning Tasks` 


- **Auditing LLMs with Six Pillars**: A member is developing an **LLM-as-judge** system for **reasoning tasks** and finds that the **Exact Match (EM)** metric is insufficient due to lucky guesses.
   - They've created a **6-pillar framework** for a *verifiable audit*, focusing on *sequential integrity* (Pillar 3) and *goal convergence* (Pillar 6), and have reference rubrics available.
- **Rubric Logic Available**: The member, a **non-coder**, mentions having reference rubrics for failure cases within their framework.
   - They offer to provide the underlying **logic for these rubrics** if needed, suggesting a detailed approach to evaluating LLM reasoning.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1477226746393919568)** (6 messages): 

> `Mojo projects, Awesome Mojo repo, Modular Community repo` 


- **Mojo Projects Attract Python, Go, Rust Converts**: A developer with experience in Python, Go, C++, and Rust expressed interest in learning Mojo.
   - They asked about finding examples of current user projects built in Mojo, signaling a potential shift towards Mojo among developers familiar with multiple languages.
- **Awesome Mojo Repo Gets Shoutout**: A member recommended the [Awesome Mojo repository](https://github.com/mojicians/awesome-mojo) as a good starting point for finding Mojo projects.
   - The original poster had trouble finding the repo initially due to other search queries.
- **Modular Community Repo Highlighted as More Active**: It was pointed out that the [Modular Community repository](https://github.com/modular/modular-community) contains more actively maintained Mojo projects.
   - The member cautioned that many projects in the Awesome Mojo list have been abandoned.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1478169847324999721)** (1 messages): 

> `Community Meeting, MAX Project, Mojo Project` 


- **Modular sets date for Community Meetup**: Modular announced their next community meeting is scheduled for **March 23rd at 10am PT**.
   - They are calling for community members to present their **MAX** or **Mojo** projects during the meeting.
- **Community projects wanted**: Modular is calling for community members to present their **MAX** or **Mojo** projects during the meeting.
   - If you would like to present, please message Modular.


  

---




### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1477045463432827104)** (50 messages🔥): 

> `def vs fn, Python and Mojo Integration, `@always_inline` vs `@inline`, Vec<dyn Trait> in mojo, Mojo package manager` 


- **`def` prevails over `fn` for Function Definitions**: The Mojo community discussed the naming convention for function definitions, with some preferring `fn` over `def` for its clarity, but ultimately decided to consolidate around `def` to maintain consistency with Python and reduce confusion, with members calling it *a tiny difference*.
   - It was noted that Python serves as a *tiebreaker* when there are no major objections to adopting its conventions, particularly in areas where performance isn't significantly impacted, also, `def` keyword makes the Mojo stay closer to Python which makes total sense to some members.
- **Seamless Python and Mojo Integration Gains Traction**: The discussion highlighted the potential for **seamless integration** between Python and Mojo, addressing the challenge of balancing Python-like syntax with the need for high-performance, close-to-metal capabilities.
   - Some members suggested that prioritizing performance over strict Pythonic similarity might be necessary, but the integration of Python and Mojo could offer a practical near-term solution.
- **`@always_inline` Naming convention scrutinized**: Members discussed the naming convention for the `@always_inline` decorator, questioning why it isn't simply named `@inline` since it **forces the compiler to inline** and the word *always* is redundant when the opposite is `@no_inline`.
   - While some felt `@inline` would be cleaner, others argued that `@always_inline` is a useful indicator that the compiler will not save you from yourself if you make too many things inline and that it is important to differentiate it from other languages where `inline` is merely a hint.
- **`Vec<dyn Trait>` Support Remains Elusive**: A member inquired about the availability of `Vec<dyn Trait>` in Mojo, which is used to achieve dynamic dispatch.
   - Unfortunately, **native support is not yet available**, and the only workaround involves manually implementing vtables and type erasure, which can be complex.
- **Mojo Package Manager still up in the Air**: Community members discussed the features and direction of a potential Mojo package manager, with one member asking if *this is about an alternative to pixi? Or more about a repository? Or something else?*.
   - A core team member clarified that the scope is still open for discussion, suggesting possibilities ranging from a Rust `cargo`-like system to something closer to `pixi`, emphasizing the need for a central repository and clear desires for package distribution.


  

---




### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1477181546841509910)** (27 messages🔥): 

> `Account Block, PSO Software Co-founder, Manus Support Response Times, Manus Credits Issue, Manus Discount Codes` 


- **User Account Blocked, Assistance Requested**: A user reported their account was blocked after a year of inactivity and requested assistance, and a member from support responded asking to DM the registered email for further investigation.
   - No further information was available in the provided context.
- **Co-Founder Hunt for PSO Software on Microsoft Store**: A member is looking for a co-founder for their **PSO software available on the Microsoft Store**.
   - The specifics of the co-founder's role or the nature of the software were not detailed.
- **Users Complain about Support Response Times and Credit Usage**: Several users expressed concerns about **slow support response times and high credit consumption** for tasks, with one claiming to have spent **$50 in credits to create a simple website**.
   - One user reported continuously encountering an error that wastes credits (see [attached image](https://cdn.discordapp.com/attachments/1349440650495398020/1477965302670299136/IMG_6722.png?ex=69a756b3&is=69a60533&hm=fdd2d3cf18895b1b37e85cf4ff7e831590ebce62164d5eb46b0d64457ecaa631&)).
- **User Shares Case Study on Product Building and Marketing**: A member shared a case study about using Manus in the early stages of building products, identifying selling and marketing as the main challenge, and highlighting their **LinkedIn hunter extension that scrapes emails from LinkedIn** based on user-defined filters.
   - Other users complained about poor agent performance after expending many credits.
- **AI & Full-Stack Developer Offers Services**: A member advertised their services as an **AI and full-stack developer**, focusing on building clean, maintainable, and secure systems that scale, offering expertise in areas like **LLM integration, AI-driven workflow automation, and full-stack web/mobile development**.
   - They emphasized shipping solid software over demos and invited those seeking reliable solutions to reach out.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1477096936703000907)** (21 messages🔥): 

> `tinygrad and robots, removing asserts in tinygrad, llama.cpp memory leak, custom uops kernel for mamba gate, new assign code` 


- **tinygrad Targets Robotics Niche?**: A user, new to the Discord via Twitter, inquired about **tinygrad** having applications for **robotics** and which channel to join to learn more.
- **assert Removal Issues**: George Hotz pointed to [channel 1068982781490757652](https://discord.com/channels/824901135537141800/1068982781490757652) regarding **removing asserts**, describing them as *simple first issues*.
- **llama.cpp memory leak?**: A user reported a **llama.cpp memory leak** with **qwen3.5 27b/35b** on **rx7900xtx** unless using **Vulkan**.
   - George Hotz suggested running with `REALIZE=0` (potentially the new default for tok/s improvements), which allows reading directly from the copied **gguf** files.
- **custom uops kernel incoming**: A user is exploring the new hierarchy features to create a **custom uops kernel** for understanding their **mamba gate**.
- **Meeting #9 on the books**: Tinygrad Meeting #9 occurred [on 3/2](https://github.com/tinygrad/tinygrad/pull/14982) at 8pm San Diego time and covered company updates, comma issues, IMAGE=1, CALL/BUFFER_VIEW, sym llm, assign, setitem, disk, drivers, llama, VIZ, other issue, and bounties.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1478205319426478101)** (1 messages): 

> `len(x.shape) vs x.ndim` 


- **len(x.shape) vs x.ndim usage spotted**: A member pointed out that there are a lot of spots where `len(x.shape)` is used over `x.ndim`.
   - They weren't sure if this warranted a PR but thought it worth mentioning.
- **tinygrad code quality**: Discussion of minor code quality issues.
   - Mention of areas where code could be improved.


  

---




### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1477873065496154112)** (1 messages): 

> `Google AI's Static sparse matrix framework, LLM-based generative retrieval` 


- **Google AI's Sparse Static Framework Speeds Up LLMs**: Google AI introduces **Static**, a sparse matrix framework delivering **948x faster** constrained decoding for **LLM-based generative retrieval** as outlined in their [blog post](https://www.marktechpost.com/2026/03/01/google-ai-introduces-static-a-sparse-matrix-framework-delivering-948x-faster-constrained-decoding-for-llm-based-generative-retrieval/).
- **Static Framework Revolutionizes LLM Retrieval Speed**: The **Static** framework achieves **948x faster** decoding speeds by leveraging sparse matrix operations within **LLM-based generative retrieval** tasks.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1477079042980319353)** (18 messages🔥): 

> `DSPy Meetup in Bay Area, Seattle DSPy organization, RLM vs ReAct, RLM token consumption, poetiq.ai RLM similarity` 


- **Seattle DSPy organization**: Members discussed organizing a **DSPy meetup in Seattle**, with a potential speaker offering to present on using **DSPy in production at scale** and principals from **AWS** discussing their work on model distillation.
   - Another suggested, *"How about Seattle? Happy to help organize"*.
- **poetiq.ai uses RLM-like approach**: A member inquired whether **poetiq.ai** uses an approach similar to **RLM**, noting that they seem to be building on top of an existing system without open-source components.
   - They observed, *"team has nothing available in open and they are building on top of existing system which seems to be similar to RLM but may be using different techniques. Is this not RLM similarity?"*
- **RLM paradigm converges to REPL**: Members discussed converging towards using **REPL** instead of multiple tool calls, suggesting it's very similar to the **RLM paradigm**.
   - One member believes their *"hunch is RLM paradigm of giving access to REPL to LLM is going to be the right way instead of giving access to tools"*.
- **RLM recursive nature is not required**: A member inquired about the recursive nature of **RLM**, asking if it's a requirement.
   - Another member clarified, *"Don’t think so recursive is spawning sub agents to run their repl. Dont think it stops the same here if needed. yeah Claude using a script to call Claude is a subagent of sorts [https://x.com/a1zhang/status/2023976399694917808?s=20](https://x.com/a1zhang/status/2023976399694917808?s=20)"*
- **Bay Area DSPy Meetup**: A member requested a small session at the **DSPy Meetup in the Bay Area** this month to clarify confusion and basics about **RLM**, particularly how it decides what code to write.
   - They seek a comparison with **ReAct**, questioning how one can be confident that **RLM** will generate the right code to fetch the right stuff when dealing with large document contexts that consume many tokens.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1477036559026098358)** (13 messages🔥): 

> `Model Context Protocol (MCP), MCP Ping Utility, Bedrock AgentCore implementation` 


- **Debate on MCP's `ping` Utility Timing**: Members debated whether the `ping` utility in the [Model Context Protocol](https://modelcontextprotocol.io/specification/draft/basic/utilities/ping) should function before the `initialize` call, citing ambiguity in the specification.
   - One member pointed out that the word *still* in the description suggests an existing, initialized connection, but the protocol allows pings before init in isolation.
- **Practicality of Pre-Initialization Pings**: Despite the specification, participants expressed that pre-initialization `ping` calls are unlikely to be useful in practice.
   - Most **SDKs for STDIO** combine connect and initialize, and remote MCP Servers typically gate functionality behind **auth/session** requirements.
- **Bedrock AgentCore's Ping Handling**: The **Bedrock AgentCore** runs customer-provided code for customer MCP servers in a container, and pings the actual server to ensure container health.
   - To avoid interfering with external client sessions, it creates a temporary session for pings, circumventing errors that occur when sessions are in use.


  

---




### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1477073591072854026)** (4 messages): 

> `Qwen3 Coder Next, Pydantic AI agents library, Minimax, Opus 4.6 and Kimi` 


- **Aider praised for fluid experience with Qwen3 Coder Next**: A user praised **aider** for its fluidity when used with **Qwen3 Coder Next** served by **llama.cpp**.
   - The user is building a project using the **Pydantic AI agents library**.
- **Keeping Aider up-to-date with Pydantic AI**: A user asked how to bring up-to-date knowledge into the aider context, as **Qwen's** knowledge cutoff is June 2024, while **Pydantic AI** is now well up to v1.63.0.
   - The user suggested downloading the `docs` directory of an open source library and using `/read` to bring it into context.
- **Hardware needs for Minimax and Qwen 397B models**: A user asked about the hardware requirements for **Minimax** and **Qwen 397B** models.
   - They also inquired how these models compare to **Opus 4.6** and **Kimi**.