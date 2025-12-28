---
companies:
- meta
- perplexity-ai
- microsoft
- gpt4all
- langchainai
- qdrant-engine
date: '2024-07-03T22:39:42.336133Z'
description: '**Meta** 推出了 **Meta 3D Gen**，这是一个能在 1 分钟内通过文本端到端生成 3D 资产的系统，能够产出具有精细纹理的高质量
  3D 资产。**Perplexity AI** 更新了 Pro Search，使其能够通过多步推理和代码执行来处理更深层次的研究任务。**微软**改进了 **Phi-3
  Mini**，提升了其长上下文理解和指令遵循能力。**GPT4All 3.0** 正式发布，支持数千种模型并兼容主流操作系统，还具备本地文件聊天功能。**Yi-Large**
  模型已在 Fireworks AI Playground 上线。


  研究亮点包括：**人类反馈强化学习 (RLHF)** 的演变、利用 10 亿个多样化人格驱动的数据合成、旨在提升少样本泛化能力的元微调（meta-tuning），以及用于控制模型行为的引导向量（steering
  vectors）。工具更新方面：**LangSmith** 改进了记忆检索功能；**Qdrant Engine v1.10** 增加了通用查询 API 和多向量搜索支持。'
id: ccac20dd-1d94-42e9-aee5-2f3d79003fa4
models:
- phi-3-mini
- gpt4all-3.0
- yi-large
- meta-3d-gen
original_slug: ainews-not-much-happened-today-1036
people:
- rohanpaul_ai
- andriy_mulyar
- cwolferesearch
- sarahookr
title: 今天没发生什么。
topics:
- 3d-generation
- long-context
- instruction-following
- reinforcement-learning-from-human-feedback
- persona-driven-data-synthesis
- meta-tuning
- model-steering
- memory-retrieval
- multivector-search
- universal-query-api
---

<!-- buttondown-editor-mode: plaintext -->**诚实即你所需。**

> 2024年7月2日至2024年7月3日的 AI 新闻。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**418** 个频道和 **2896** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**341 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

[Arvind Narayanan 等人发表了一篇论文](https://www.aisnakeoil.com/p/new-paper-ai-agents-that-matter)，探讨了 Agent 论文大多不可复现且忽略成本的问题；[Meta 发布了一个 text-to-3D 资产模型](https://x.com/AIatMeta/status/1808157832497488201?utm_source=ainews&utm_medium=email)；[Magic.dev 和 Poolside](https://x.com/johnbyronhanby/status/1808235931784434049) 是正在寻求独角兽轮融资的代码模型公司；OpenDevin [现在已成为一家公司](https://x.com/gneubig/status/1808493521315496229)；Kyutai 发布了一个[实时 Audio LLM](https://x.com/giffmana/status/1808482848808010149)，但[其效果可能并不如宣传的那样](https://x.com/benhylak/status/1808611023123067357)；Peter Thiel 资助了[某个 AGI Blockchain 项目](https://x.com/sentient_agi/status/1808136737257918916)；The New Stack 发布了[第一篇](https://thenewstack.io/lets-get-agentic-langchain-and-llamaindex-talk-ai-agents/)和[第二篇](https://thenewstack.io/mozilla-llamafile-builders-projects-shine-at-ai-engineers-worlds-fair/)关于 AIEWF 的报道。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**AI 模型发布与更新**

- **Meta 3D Gen**：[@AIatMeta](https://twitter.com/AIatMeta/status/1808157832497488201) 推出了 Meta 3D Gen，这是一个用于**在 1 分钟内从文本端到端生成 3D 资产**的新系统，可生成具有高分辨率纹理和材质贴图的高质量 3D 资产。详细信息请参阅技术报告。
- **Perplexity Pro Search 更新**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1808183923064656383) 宣布了 Pro Search 的更新版本，它可以通过多步推理、Wolfram|Alpha 和代码执行，对**更复杂的查询进行更深入的研究**。
- **Phi-3 Mini 更新**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1808087286661132494) 分享了微软更新 Phi-3 mini 的消息，通过后期训练改进，在**长上下文理解、指令遵循和结构化输出方面取得了显著进步**。
- **GPT4All 3.0**：[@andriy_mulyar](https://twitter.com/andriy_mulyar/status/1808170696717070667) 宣布推出 GPT4All 3.0，**支持数千个模型和所有主流操作系统**，并带来了重大的 UI/UX 改进以及通过 LocalDocs 实现的本地文件聊天功能。
- **Yi-Large 上线**：[@01AI_Yi](https://twitter.com/01AI_Yi/status/1808262539681177681) 庆祝 Yi-Large 在 Fireworks AI Playground 上线一周，并征求用户对该模型的反馈。

**研究论文与技术**

- **来自人类反馈的强化学习 (RLHF)**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1808218688388321463) 概述了 **RLHF 研究的演变**，追溯到研究使用人类反馈训练摘要模型的论文。文中链接了关键论文。
- **角色驱动的数据合成 (Persona-Driven Data Synthesis)**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1808096574997770590) 分享了一篇论文，提出了一种**使用 Persona Hub 的角色驱动数据合成方法**。Persona Hub 包含 10 亿个多样化的角色，旨在为 LLM 训练和评估创建可扩展且多样化的合成数据。
- **用于 Few-shot 泛化的 Meta-tuning**：[@slashML](https://twitter.com/slashML/status/1808205077015875693) 分享了一篇关于“通过稀疏插值专家**释放 Meta-tuning 在 Few-shot 泛化中的力量**”的论文。
- **引导向量 (Steering Vectors)**：[@sarahookr](https://twitter.com/sarahookr/status/1808237222522769410) 分享了关于**引导模型行为趋向不可微目标**的工作，通过约束生成过程，显式引导模型最小化或最大化不可微特征。

**框架与工具**

- **LangSmith**：[@LangChainAI](https://twitter.com/LangChainAI/status/1808154656746754114) 分享了一个案例研究，介绍 @newcomputer 如何使用 LangSmith **快速迭代并改进记忆检索**，使其 Agent 记忆系统 Dot 的召回率提高了 50%，准确率提高了 40%。
- **Qdrant Engine v1.10**：[@qdrant_engine](https://twitter.com/qdrant_engine/status/1808121142961406156) 发布了 Qdrant engine v1.10，具有 **通用查询 API、多向量搜索、逆文档频率 (Inverse Document Frequency)** 等新功能。
- **Leap AI**：[@LeapAI_](https://twitter.com/LeapAI_/status/1808238079037395145) 介绍了他们的平台，用于**构建自定义 AI 工作流以自动化内容创建、线索生成**等，并集成了 GPT-4 等最先进的 AI 模型。

**讨论与观点**

- **AI 的功能增强研究 (Gain of Function Research)**：[@JvNixon](https://twitter.com/JvNixon/status/1808201698466570372) 对 AI 的“**功能增强研究**”表示担忧，将其与生物武器研究相类比，并指出创建团队试图生成新颖、危险的输出来证明模型是否安全的潜在危险。
- **毁灭概率 (p(doom)) vs. 生存概率 (p(life))**：[@JvNixon](https://twitter.com/JvNixon/status/1808267707747557807) 认为，用 **p(doom) 来界定 AI 风险是一个深刻的集体心理错误**，这迫使人们去想象抽象的超级智能。他们更倾向于使用 p(life) —— 你和你爱的人在遥远的未来生存的概率 —— 因为它涵盖了更多生命和进步的内容，并迫使人们在风险与收益之间取得平衡。
- **AI 实验室的闲置算力**：[@far__el](https://twitter.com/far__el/status/1808205077015875693) 指出，许多 AI 实验室都有**大量闲置算力**，因为他们需要爆发式的算力支持。这导致了重度补贴的推理服务等现象，将算力成本重新定义为营销支出。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 模型与技术**

- **Microsoft Phi-3 Mini 更新**：在 /r/LocalLLaMA 中，Microsoft 更新了其 Phi-3 Mini 模型的 4K 和 128K 上下文版本，展示了 [**在指令遵循和知识保留方面的显著改进**](https://www.reddit.com/r/LocalLLaMA/comments/1dtgylv/microsoft_updated_phi3_mini/)。评论讨论了命名规范、对该模型系列潜力的兴奋，以及与 Microsoft 产品命名历史的对比。

- **开源 Mixture-of-Agents 表现超越 GPT-4o**：在 /r/LocalLLaMA 中，一种仅使用开源模型的 Mixture-of-Agents (MoA) 方法 [**在 AlpacaEval 上达到了 65.1%，而 GPT-4o 为 57.5%**](https://www.reddit.com/r/LocalLLaMA/comments/1dtmqt5/open_source_mixtureofagents_llms_far_outperform/)。使用的模型包括 Qwen, WizardLM, LLaMA 和 Mixtral 的变体。评论质疑了基准测试的局限性，指出了该方法的成本，并提到了相关视频。

- **Rubra v0.1 推出支持 Tool-calling 的 LLM**：在 /r/LocalLLaMA 中，Rubra v0.1 发布，这是一系列开源权重、支持 Tool-calling 的 LLM，包括 [**Llama, Qwen, Mistral, Phi 和 Gemma 模型的变体，旨在提供可靠的函数调用 (Function Calls)**](https://www.reddit.com/r/LocalLLaMA/comments/1dtt32y/new_collection_of_llama_mistral_phi_qwen_and/)。

- **MMLU-Pro 基准测试被批评过于偏重数学**：在 /r/LocalLLaMA 中，MMLU-Pro 基准测试因 [**被数学和 Chain-of-Thought 推理主导，导致其在评估通用知识方面作用有限**](https://www.reddit.com/r/LocalLLaMA/comments/1du52gf/mmlupro_is_a_math_benchmark/) 而受到批评。建议包括有针对性的子采样以及与 MixEval 的对比。评论指出 MMLU-Pro 在本地测试和评估未来 SOTA 模型中非常流行。

- **小模型在 MMLU-Pro 上的对比**：在 /r/LocalLLaMA 中，Llama 3 8B, Mistral 7B, Phi Medium 和 Yi 1.5 9B 等小模型 [**在 MMLU-Pro 基准测试上进行了对比**](https://www.reddit.com/r/LocalLLaMA/comments/1du0rka/small_model_mmlupro_comparisons_llama3_8b_mistral/)。关键结论强调了 Mistral 强大的综合表现，以及 Llama 3 尽管经过量化仍具有竞争力。

**AI 视频与动画**

- **AI 生成的外星自然纪录片**：一个 [**展示外星自然纪录片的 AI 生成视频**](https://v.redd.it/f15k13mye2ad1) 证明了 AI 驱动内容的质量和可看性有所提高。

- **Sora 与 Runway 视频生成对比**：一段 [**Sora 与 Runway 视频生成能力的对比视频**](https://v.redd.it/iy8jinx6w2ad1) 显示，虽然差距很小，但 Sora 在动态和整体质量上更胜一筹。评论讨论了 Runway 的高对比度、Sora 尚未公测的状态以及潜在的精选 (Cherry-picking) 行为。

**AI 伦理与社会影响**

- **对 Kling 垃圾信息的担忧**：在 /r/StableDiffusion 中，引发了关于 [**Kling 和 RWML 视频垃圾信息增加的讨论，暗示这些闭源服务可能在进行网络水军 (Astroturfing) 活动**](https://www.reddit.com/r/StableDiffusion/comments/1dtrnu6/meta_discussion_kling_spam/)。

- **AGI 对权力集中的影响**：在 /r/singularity 中，一项民意调查询问 [**AGI 是否会导致权力的集中化或去中心化**](https://www.reddit.com/r/singularity/comments/1du2gj2/will_agi_lead_to_centralization_or/)。

- **AI 在学生贷款债务中的角色**：在 /r/singularity 中，有人提出了一个问题：[**AI 系统是否应该为被取代的白领工人偿还学生贷款，或者 UBI (全民基本收入) 是否更好**](https://www.reddit.com/r/singularity/comments/1dtmltm/if_ais_start_taking_all_the_white_collar_jobs/)。

- **AI 研究中的心理健康**：在 /r/singularity 中，意大利国家研究委员会呼吁参与一项研究，以 [**了解 AI 研究人员面临的心理健康挑战**](https://www.reddit.com/r/singularity/comments/1dtj1lk/help_us_understand_mental_health_in_ai_research/) 并开发支持系统。

**其他**

- **GPT4All 3.0 发布**：[GPT4All 3.0，一个开源的本地 LLM 桌面应用程序正式发布](https://x.com/nomic_ai/status/1808162955806097767)。

- **AI 生成艺术展示**：分享了各种 AI 生成的艺术作品，包括 [使用 Stable Diffusion 3 创建的昆虫排版艺术](https://www.reddit.com/gallery/1dtluza)、[《原神》角色的透明像素画](https://i.redd.it/a9wsgbvfy6ad1.png) 以及 [结合 SDXL 与 SD3 Refiner 的工作流](https://www.reddit.com/gallery/1dty6rl)。

---

# AI Discord 摘要回顾

> 摘要之摘要的摘要

1. **实时 AI 模型成为焦点**：
   - **Kyutai Labs** 推出了 [**Moshi**](https://moshi.chat/?queue_id=talktomoshi)，这是一个用于实时文本和音频生成的 7B 多模态模型，响应时间仅为 160ms。因其开源可用性和极速交互（*尽管略显机械感*）而引发关注，在演示环节展示后，计划修复一些细微的 Bug。
   - **Phi-3 Mini** 模型迎来了类似于 **3.5 Mini** 的重大更新，并即将支持 [**Gemma 2**](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit)，但用户反馈的启动问题反映了前沿 AI 工具在集成过程中的挑战。

2. **优化 AI 部署与内存管理**：
   - 关于 **Colab 和 Kaggle notebooks** 的广泛讨论分享了内存管理的最佳实践，包括使用 `gc.collect()` 和 `torch.cuda.empty_cache()` 等方法。用户还就根据数据集大小缩放模型的 LoRA rank 展开了辩论，强调通过高效的资源处理进行优化。
   - **Gemma 2** 对 **Unsloth** 和 **LM Studio** 等工具的支持增强显著提升了微调速度，Unsloth 实现了 **2倍的微调速度** 和 **63% 的内存占用减少**；同时，LM Studio 的 0.2.27 更新解决了在 **Mac, Windows, and Linux** 上的兼容性问题。

3. **AI 模型训练与微调的创新**：
   - **QLoRA** 因其对量化 LLM 的 [**高效微调**](https://arxiv.org/abs/2305.14314) 而受到关注。正如 **QLoRA** 论文中所详述，它允许在 48GB GPU 上微调 65B 参数模型，并使用 4-bit 量化达到接近 16-bit 精度的性能。
   - 成员们深入探讨了使用 **DeepSpeed** 和针对 Nvidia 的 **Inductor** 后端来**优化 CUDA 操作**，重点关注 **自动调优 GEMM 后端** 以及排查 `torch.cuda.OutOfMemoryError`，进一步强调了硬件感知优化的重要性。

4. **AI 中的隐私、安全与伦理考量**：
   - 对**数据政策执行**的担忧引发了关于 **OpenAI GPT-4** 订阅定价以及影响用户体验的偶发性模型参数调整的激烈讨论。因轻微违反政策而导致数据集被删除的问题，激发了关于执行一致性与用户需求之间平衡的辩论。
   - 关于 [**Glaze**](https://glaze.cs.uchicago.edu/) 和 **Nightshade** 等**反 AI 艺术软件**的讨论提出了伦理问题，即如何在版权保护与技术进步之间取得平衡，凸显了社区对潜在规避保护工具行为的挫败感。

5. **社区工具、教程与协作**：
   - 用户分享了各种开源工具和教程，例如使用 Transformers [**创建自定义流水线 (Pipelines)**](https://github.com/andysingal/llm-course/blob/main/transformers/custom-pipeline.md) 以及用于角色扮演提示词的 [**Gradio 应用**](https://huggingface.co/spaces/xtreme86/System_roleplay_generator)，促进了协作学习和实际落地。
   - 针对 [**AI Town**](https://github.com/Ikkitsuna/AI-Town-Windows-Setup-WSL-method) 等 AI 工具的 **Docker 镜像**开发吸引了社区的积极参与，重点在于简化安装流程，并通过 GitHub 上的详细 PR 和文档提交确保与各种平台的兼容性。

---

# 第一部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Phi-3 Mini 的华丽蜕变**：Phi-3 Mini 模型经历了重大更新，类似于 3.5 Mini，根据 Unsloth AI 的公告，兼容 **Gemma 2** 的量化版本预计很快发布。
   - 用户反馈显示，对于 Unsloth 中新的 **Gemma 2** 支持既感到兴奋，也遇到了启动问题，这反映了尖端 AI 工具集成初期的磨合问题。
- **Moshi 的旋律 AI 精通**：Kyutai Labs 推出了“**Moshi**”，这是一个 7B 多模态 LM，能够实时生成高质量的文本和音频，响应时间达到 **160ms**，并计划开源。
   - AI 社区对 Moshi 的能力议论纷纷，包括其 RLHF 微调、后端通用性以及对即将到来的更新的期待。
- **Colab 的容量攀升**：新分享的 Colab/Kaggle notebook 提供了广泛的数据集支持，并引入了根据模型和数据集大小缩放 LoRA rank 等改进，引起了 Unsloth 社区的关注。
   - 成员们讨论了内存管理的最佳实践，包括 `gc.collect()` 和 `torch.cuda.empty_cache()`，同时承认为了便于使用，需要固定（pin）资源密集型的 notebook。
- **秘密安全的 Docker 部署**：关于 Docker 部署中安全密钥管理的讨论正在进行，社区共识倾向于将使用 `--env-file` 标志处理环境变量作为最佳实践。
   - 社区流传着高效容器处理和部署的建议，例如使用本地注册表以及 `docker save` 和 `ctr images import` 等 Docker 命令。
- **解决 Unsloth 的本地怪癖**：用户报告了在本地使用 Unsloth 时的配置问题，建议的修复方案包括更新 `config` 对象以反映 API 的变化。
   - 尽管 Gemma 2 预计在 1-2 天内更新的消息引起了社区轰动，但持续的讨论仍在强调延迟，并热切期待 PHI 的 JAVA 评估改进。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4 订阅者苦于使用限制**：用户对 **GPT-4 订阅**表示担忧，面临快速达到消息上限以及升级后性能下降等问题。社区交流了替代方案，并强调了模型的局限性。
   - 关于**订阅定价**出现了辩论，一些每月支付高达 60 美元的用户对 OpenAI 零星的参数调整提出质疑，怀疑其作为专业工具的性价比。
- **AI21 隆重推出“Jamba”**：**AI21 Labs** 推出了 [“Jamba”](https://www.ai21.com/blog/announcing-jamba)，号称结合了 *Mamba SSM 技术和 Transformer 架构*，展示了其 **256K 上下文窗口**和极具竞争力的定价，引发了热烈讨论。
   - 随后展开了关于将 **Jamba** 应用于编程任务的讨论，报告显示其结果与 GPT-4 和 Claude 等其他 AI 模型相比褒贬不一，引发了关于提高准确性的潜在方法的对话。
- **开源 AI 工具加入战局**：[“Moshi”](https://moshi.chat/?queue_id=talktomoshi) 的发布引起了许多人的兴趣，这是一个用于实时 AI 对话的开源工具，尽管目前仍处于早期阶段且存在局限性。
   - 社区权衡了**开源 AI 工具**与专有模型的优缺点，讨论了这些发展将如何影响 AI 在日常技术中的融合。
- **提示工程（Prompt Engineering）深度探索**：**提示工程**成为核心话题，成员们分享了磨练提示词的建议，以实现更精确的 AI 任务表现，特别是针对创建带有**格式化产品标签**的 PDF 等细微任务。
   - 用户应对了 **DALL-E 提示工程**的复杂性，提供了简化提示词和提高特异性等建议，以减少不必要的图像元素问题。
- **嵌套 GPT 引发好奇与辩论**：在 GPT 开发领域，一位用户关于 **GPT 调用其他 GPT** 可行性的询问，开启了关于此类嵌套功能的技术细节和假设深度的讨论。
   - 社区还对数据政策的执行表示不满，指出一个涉及 15 年前条目的数据集被删除，引发了关于灵活合规与严格准则之间平衡的讨论。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 的最新 Gemma 2 增强功能**：LM Studio 0.2.27 引入了对 **Gemma 2** 模型的改进支持和兼容性，并在 **Mac、Windows 和 Linux 平台**上增强了性能。建议用户[更新到新版本](https://lmstudio.ai)以获得无缝体验。
   - 像 [abetlen](https://github.com/abetlen) 这样的社区贡献者在更新 **Gemma 9B 和 27B** 模型方面发挥了重要作用，这些模型可以从 [Hugging Face](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) 重新下载，以确保与当前设置的兼容性。
- **在 ROCm 之海上平稳航行**：一个令人担忧的错误 'unknown model architecture: gemma2' 引发了围绕新版 **LM Studio 0.2.27** 的讨论，提议的解决方案包括**清除缓存或完全重新安装**。
   - 针对 **ROCm GPU 兼容性性能**的社区测试表明，在 **AMD Radeon RX 6900 XT** 等模型上取得了成功，并提示协助验证针对更新软件版本的最新 Linux ROCm 扩展包。
- **解决功耗难题**：对 LM Studio 能源消耗的深入研究揭示了较高的待机功耗，引发了关于能效的讨论，并与 [Blender](https://discord.com/channels/1110598183144399058/1253332613540876401) 等其他工具进行了对比，表明需要进行优化。
   - 操作系统之间的差异显现，**Linux 用户注意到他们的 GPU 在运行模型时功耗更低**，而 Windows 用户在类似活动中报告了功耗激增。
- **缩放之战与界面改进**：关于 LM Studio 的反馈指出了在 **1080p 显示器**上的缩放问题，由于界面拥挤限制了工作流效率，并强调了在多显示器环境中布局优化的重要性。
   - 用户建议在 LM Studio 的界面模型列表中增加发布日期等元数据，这一建议得到了社区的积极响应。
- **Gradio 应用的角色扮演革命**：为了追求更丰富的角色扮演体验，一位用户率先开发了一个带有动态变量的 **Gradio 应用**，旨在改善沉浸式角色互动，点燃了 AI 驱动叙事的创新之火。
   - 该应用提供定制提示词的能力使其处于前沿，并邀请社区反馈以增强其功能，可在[这个创意空间](https://huggingface.co/spaces/xtreme86/System_roleplay_generator)查看。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Transformers 4.42：新模型亮相**：[Transformers 4.42 版本](https://huggingface.co/docs/transformers/v4.42.0/release)首次推出了 **Gemma 2** 等新模型，改进了工具可用性和微调能力，标志着模型进展的又一步。
   - `KerasNLP` 现在使模型爱好者能够跨平台[集成和微调 Transformers](https://github.com/andysingal/llm-course/blob/main/transformers/custom-pipeline.md)，拓宽了机器学习应用和效率的前景。
- **数据丰沛：AWS Chronos 数据集公开**：AWS 在 HF 上发布了全面的 [Chronos 数据集](https://huggingface.co/datasets/chronos)，包含预训练和评估基准，为时间序列分析提供了丰富的资源。
   - 研究人员可以通过 AWS 数据集深入研究时间模式，这可能会激发数据驱动的洞察和模型创新。
- **AI 专业技能发展：免费课程涌现**：[哈佛大学](https://harvard.edu/)等知名机构提供免费的 ML 课程，拥有优质内容和认证途径。
   - 这些课程是那些旨在无经济障碍地提高 ML 熟练程度的人的门户，尽管基础知识的重复性是潜在学习者需要考虑的因素。
- **社区参与：新角色与资源**：HuggingFace 的 Discord 社区随着对 [Qwen2](https://huggingface.co/Qwen/Qwen2-7B-Instruct) 等大上下文窗口模型能力的持续讨论而不断壮大，表明人们对细致文本处理的兴趣日益增加。
   - **HF 模型**（如 Meta-Llama）与闭源巨头之间的效率对比显示，开源模型正在挑战闭源工具的主导地位。
- **Diffusers vs. A1111：模型质量争议**：在运行相同的生成参数时，用户报告 **RealVisXL V4.0 Lightning** 在使用 diffusers 时质量不如 A1111，尽管设置完全相同。
   - 讨论集中在不同执行方法之间的质量权衡，这对于在照片级真实感任务中实现所需的模型性能至关重要。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT-4 的巨大容量：Nvidia 的预览**：GPT-4 推测的参数范围在 **1.7 到 1.8 万亿** 之间，引起了广泛关注，使 GPT-3 的 **1750 亿** 显得微不足道。在一场 [涉及 Nvidia 的讨论](https://www.nvidia.com) 中，尽管有保密协议（NDA），但由于硬件支持方面的紧密联系，暗示了该公司与此的深厚渊源。
   - **InstructGPT** 的实际应用展示了 **10 倍到 100 倍** 的效率提升，这归功于 **Reinforcement Learning from Human Feedback (RLHF)**，引发了对其潜力的热烈讨论。
- **Scaling Law 之争：Kaplan vs. Hoffmann 的解析**：社区讨论了 Kaplan 等人与 Hoffmann 等人提出的 Scaling Law 之间的差异，并对最后一层成本和预热时长提出了新见解，详见 [arXiv 论文](https://arxiv.org/abs/2406.19146)。
   - 对话强调了 [PyTorch FLOP 计数器](https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py) 可能存在的缺陷，以及准确的 FLOPs 计算方法对模型扩展的重要性。
- **解释可解释性：稀疏电路浮出水面**：关于 EAP 和集成梯度的论文启发了对 **稀疏特征电路 (sparse feature circuits)** 的探索，这是一种剖析语言模型行为的方法，旨在建立 [这项工作](https://arxiv.org/abs/2403.19647) 中概述的有条理的可解释性流水线。
   - 用于分类器泛化的 SHIFT 方法激起了好奇心，表明细粒度的可解释性单元可以消除无关特征，并从人类判断中汲取灵感。
- **预处理中的困惑度：导航长文档**：**Stellaathena** 的配置困惑度因其在 **proof-pile** 中的错误而让其他人感到困惑，这与 `lambada_openai` 的顺畅运行形成鲜明对比，引发了关于确保模型评估效率和准确性的讨论。
   - 技术讨论包括 **loglikelihood_rolling** 功能及其在将对数似然转换为损失值中的应用，这是该论坛在模型评估方面持续敏捷性的一部分。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **尝试 Gemini 1.5 Pro**：用户参与了关于 **Gemini 1.5 Pro** 的讨论，强调了其 **大上下文窗口** 和快速响应时间。该聊天机器人因其稳健的性能而获得推荐，并收获了正面反馈。
   - 同时也提出了关于 **Perplexity 实时联网访问** 的担忧，用户在获取实时数据能力方面的体验参差不齐，导致了一些挫败感。
- **应对 GPT4o 访问困难**：成员们强调了免费访问 **GPT4o** 的挑战，转而向他人推荐 **Bing chat** 和 [**Claude 3.5 Sonnet**](https://claude.ai) 作为免费对话的可行替代方案，但受使用限制约束。
   - 对话还包括关于 **Perplexity Pro 订阅退款流程** 的提示，并针对欧盟、英国和土耳其等不同地区提供了定制建议。
- **Perplexity 的移动端精通**：关于 **Perplexity 移动应用功能** 的疑问得到了澄清，确认 iOS 端已包含 **Wolfram Alpha** 和 **代码生成** 能力。
   - 关于移动端功能重要性的讨论表明，用户对在手持设备上访问高级工具表现出浓厚兴趣。
- **Sonnet 的 API 缺席**：讨论显示 **Sonnet 3.5** 尚不支持 **Perplexity API**，促使用户查阅 [官方模型文档](https://docs.perplexity.ai/docs/model-cards) 以寻找替代方案。
   - 除 API 功能外，还出现了关于通过 API 利用 **Perplexity 搜索引擎** 潜力的咨询，社区对访问这些扩展功能表现出极大的热情。
- **AI 黑盒构建模块**：提供了在 AI 中创建黑盒系统的说明和原则，为构建这些复杂系统提供指导。
   - 分享了包括 **精益画布 (Lean Canvas)** 和 **Perplexity AI 创立** 在内的素材，有助于更广泛地理解技术领域的战略规划和创业初期。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 秘密会议召开**：由 Ash Vardanian 主办的 **CUDA-only hackathon** 邀请了 **Chris Lattner**，定于 **7 月 13 日**在旧金山的 AGI House 举行，提供 **H100 accelerators** 的实操经验。[点击此处查看详情](https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf)，由 Nebius.ai 提供支持。
   - 在另一场活动中，Meta 的 **Hacker Cup 2024** 准备于 **9 月 20 日**开赛，Mark Saroufim 敦促开发者们投入 **code generation challenge**。与此同时，GPU 爱好者们正纠结于 **NVIDIA 3090 的 1,000 美元标价**，Mark Saroufim 分享说他以 **1,200 美元的价格抢到了一块 4090**。
- **矩阵乘法精通**：**Mobicham** 发布了一份在 CPU 平台上实现超过 **1 TFLOPS 矩阵乘法性能**的指南，专门针对 **AMD Ryzen 7700** 进行了优化，性能超越了 NumPy 的表现。[教程可以在这里找到](https://salykova.github.io/matmul-cpu)。
   - **3D V-Cache** 技术因其对 AMD Ryzen 性能的贡献而受到关注，引发了关于其在增加缓存容量之外的专业化讨论，涉及 **clock speeds 和 silicon layering**。
- **集成器的细节**：关于在 **PyTorch** 中使用 Nvidia 的 Inductor 后端编译函数的讨论展开，提到了 [John Carmack 对 PyTorch 团队的称赞](https://x.com/ID_AA_Carmack/status/1807072152631333060)，同时深入探讨了使用 torchao 进行 **buffer loading 和 dequantization** 的过程。
   - 发现了一个强制 Inductor 为所有操作生成 **Triton kernels** 的小问题，其中 **GEMM 成功但 Conv 失败**，详见寻求解决方案的 [GitHub issue](https://github.com/pytorch/pytorch/issues/125728)。
- **模型内存奇迹**：前沿的内存效率策略让该频道的模型成为焦点，这些模型可以轻松处理让 PyTorch 望而却步的 batch sizes，强调了模型的 **memory savings**。
   - 引用的 **GitHub Pull Request [#667](https://github.com/karpathy/llm.c/pull/667)** 解决了训练期间 batch sizes 中的小数点导致整数除法错误的问题，标志着一次增量改进。
- **优化器探索之旅**：[Facebook Research 的 schedule-free optimizers](https://github.com/facebookresearch/schedule_free) 带来了一波乐观情绪，据称这些优化器在各种任务中都表现出了加速收敛的特性，有可能重塑优化方法论。
   - 社区分享的发现表明，在不严格遵守 schedule 的情况下微调模型的潜力显著提升，正处于优化技术复兴的边缘。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **艺术家盟友的策略减弱**：社区对话集中在开发 **anti-AI art software**（如 [**Glaze**](https://glaze.cs.uchicago.edu/) 和 **Nightshade**）以保护艺术家版权，但几位成员对绕过此类工具的简易性表示担忧。
   - 对话强调了在 AI 训练中维持 **copyright protection** 与技术进步之间平衡的挑战。
- **像素完美的困境**：关于 **16x16 pixel art** 的咨询引出了在 **512x512** 分辨率下进行训练的建议，尽管 *Crystalwizard* 评论说为了追求效率可能需要不断的尝试。
   - 重点放在了训练方法的实验上，以磨练针对这种特定艺术风格的图像生成，强调了 AI 模型训练的细粒度。
- **Discord 就业中心讨论**：有帖子询问服务器是否有专门的 **job-posting channel**，突显了社区内对 **freelance and job opportunities** 需求的激增。
   - 另一场讨论思考了自由职业者之间 **upwork account rentals** 的伦理和物流问题，反映了科技领域的零工经济现状。
- **提示词技巧与性能之谜**：关于各种 **prompting techniques** 的辩论展开，例如 **[A|B], C** 与 **[A, B, C]** 的对比，评估它们对图像输出的影响，特别是在使用 **SD1.5** 与 **segmoe** 和 **MixofExperts** 等模型时。
   - 兴趣集中在改进技术以在 text2img 结果中获得更高的保真度，讨论评估了不同语法方法的有效性。
- **模型大乱斗：MixofExperts 与 segmoe**：社区评估详细介绍了 **segmoe** 模型在 **prompt understanding** 方面的进步（在 **ComfyUI** 等应用中展示），以及它被认为优于小众的 **SD1.5 finetunes**。
   - 成员们的对比分析阐明了性能上的细微差别，以及在新兴模型中追求精确自然语言理解的探索。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 上的模型演变**：OpenRouter 宣布了一系列变化，包括对 **/models 页面**的**重大更新**，以及对 **Gemini 和 PaLM 模型的 Google Token 大小**进行的调整——将更大的 Token 与 GPT 等同，从而影响定价模型。
   - OpenRouter 迎来了一波弃用潮：设置页面上的**默认模型 (Default Model)** 和 OpenAI API 密钥的**自定义认证标头 (custom auth headers)** 都将被停用，转向更新的实践和标准。
- **Claude 3.5 的连接难题**：社区用户在处理 **Claude 3.5** 时一直遇到 **500 错误**，这促使一些人暂时转向 **Claude 3.0** 等替代版本以寻求稳定性。
   - OpenRouter 上的讨论涉及了**隐私设置和日志策略**，各供应商立场不一；**NovitaAI** 和 **Infermatic** 因承诺不保留数据而脱颖而出，正如 [Alex Atallah](https://openrouter.ai/settings/privacy) 所强调的那样。
- **讨论 LLM 精度**：AI 工程师推测了 OpenRouter 上 **LLM 模型的量化 (quantization)** 情况，辩论焦点在于部署的模型是使用 **FP16** 还是保持其原始精度（除非供应商特别更改）。
   - 针对利用 Claude 模型的替代前端（如 **SillyTavern** 和 **LibreChat**）的有效性进行了辩论，并提出了 **Typingmind** 和 **Pal Chat** 等建议以增强互动。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Magic.dev 无代码获注资**：在一次惊人的财务飞跃中，[Magic.dev](https://www.reuters.com/technology) 在仅有 20 名员工、且没有任何产品或收入记录的情况下，估值飙升至 **15 亿美元**。
   - 这笔前所未有的融资旨在将这家新兴公司定位为 AI 领域强大的竞争者，为初创企业设定了**新的融资基准**。
- **十亿角色指南发布**：合成数据生成取得突破性进展，[Persona Hub](https://arxiv.org/abs/2406.20094) 集成了 **10 亿个角色 (personas)**，在基准测试上带来了令人印象深刻的提升。
   - [Aran Komatsuzaki](https://x.com/arankomatsuzaki/status/1807593343007818065) 赞扬了该方法，强调了其在生成高质量合成数据和增强多样性方面的潜力。
- **实时音频 LLM 'Moshi' 发声**：由 Kyutai Labs 推出的 [Moshi](https://x.com/giffmana/status/1808482848808010149) 作为首个实时音频 LLM 亮相，展示了极低的延迟，但**发音略显机械**。
   - 尽管它急于回答会导致偶尔的打断，但该技术预示了用户与人工智能交互的新前沿。
- **全员参与技术：OpenDevin 的新举措**：[OpenDevin](https://x.com/gneubig/status/1808493521315496229) 背后的创业团队成立了 All Hands AI，致力于通过**开源倡议**实现 AI 软件开发的民主化。
   - 该平台的建立象征着迈向普及 AI 工具和共享开发理念的协作一步。
- **Sentient 种子轮成功：资助开放 AGI 探索**：Sentient 宣布获得 **8500 万美元种子轮注资**，由 [Peter Thiel](https://x.com/sentient_agi/status/1808136737257918916) 等知名人士领投，旨在打造一个邀请全球参与的社区驱动型 AGI 平台。
   - 这笔雄心勃勃的资金是创建平等 AI 生态系统中集体智慧的号角。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **去中心化 Transformers 取得进展**：**jaan.li** 在 [onefact.org](https://onefact.org) 和 usb.club 介绍了他们专注于去中心化边缘 Transformer 的项目，引发了对其潜在应用和合作联系的兴趣。
   - 虽然 **san.tosh** 寻求关于开源 GPT-4o 的更新，但社区仍处于期待中，讨论在继续但尚无具体消息。
- **Terminator 模型面临严密审查**：社区批评 **Terminator** 模型的消融实验（ablation tests）不足，并敦促对其更改进行实质性的辩护，强烈要求展示详细的研究。
   - 然而，随着其 GitHub 版本的发布，对该模型的怀疑者被打脸，因为 [Terminator 的代码已上线](https://github.com/hyperevolnet/Terminator)，允许更广泛的探索和实验。
- **Vision Transformers 的 QKV 受到质疑**：关于 Vision Transformers 中 QKV 必要性的辩论浮出水面，假设认为可能存在冗余，并需要进行实证评估。
   - 共享的替代方案理论渴望通过严格的审查来揭示此类架构中注意力机制（attention mechanisms）的全面影响。
- **FORA 打造更快的 Diffusion Transformers**：**FORA** 的引入提议通过缓存可重用的计算来加速 Diffusion transformers，为计算效率挑战提供了解决方案。
   - 该技术因其与现有模型融合并部署快速处理进展的潜力而受到关注，详见其 [代码库](https://github.com/prathebaselva/FORA?tab=readme-ov-file)。
- **HyperZ⋅Z⋅W 论文引发两极分化的观点**：**HyperZ⋅Z⋅W** 论文收到的评价褒贬不一，展示了一个初创的提交如何激起对实现 SOTA 成就的新方法的认可与怀疑。
   - 尽管存在批评，但围绕 HyperZ⋅Z⋅W 论文标记的新颖想法和潜在修订仍笼罩着好奇氛围，暗示着关于 ViT 中 QKV 影响的讨论正在增长，正如 Schmidhuber 的 [综述](https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html) 所述。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 的 UNMUL 陷入 RuntimeError**：tinygrad 内部报告了一个 **RuntimeError**：*'failed to render UOps.UNMUL'*，由 **George Hotz** 牵头断言这一情况“永远不该发生”。
   - 讨论展开了关于使循环折叠（loop collapse）变为可选的议题，由 `flat_l4.realize()` 提示，以避免对用户造成影响，并由 **Chenyuy** 提出了权宜之计。
- **模糊测试前端：Tinygrad 的测试接管**：**Chenyuy** 提出了针对 tinygrad 的 **前端模糊测试器（frontend fuzzer）** 的概念，旨在利用类似于通过 LLM 移植 torch 代码的方法来根除边缘情况。
   - 社区对为某些维度创建最小复现测试（minimal repro tests）以解决启发式边界异常感到兴奋，PR 仍处于开放状态以进行持续深入研究。
- **Tinygrad 1.0 前的调试冲刺**：tinygrad 改进错误消息的需求变得明确，*Yosifrost* 强调了 1.0 版本前的开发者工具增强。
   - 社区协作复现错误并设计测试用例，为更强大的调试机制奠定了基础。
- **梯度抱怨与内存之谜**：AI 工程师们交流了梯度累积失误导致 CUDA 显存溢出（out-of-memory）错误的经验，论坛上流传着诸如分离损失（detaching loss）之类的技巧。
   - 强调了 TinyJit 在优化方面的缺陷，包括 **TinyJit** 未能有效使用 `assert t.grad is not None` 语句，引发了社区的迅速响应。
- **Tinygrad vs PyTorch：张量创建的怪癖**：tinygrad 和 PyTorch 之间 `Tensor.randn/randint` 和 `Tensor.full` 的不一致性引发了对张量连续性（tensor contiguity）的分析以及对齐建议。
   - 这种行为被归结为 tinygrad 特有的习性，但这并未阻碍关于改进未来迭代以获得更好兼容性的讨论。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Pinecone 的困境与潜在转向**：由于 **Pinecone 限制**，用户在创建 **DocumentSummaryIndex** 时遇到了障碍，原因是节点的 metadata 过大以及 **embed exclusion filters** 设置不当，详情见此 [GitHub 代码片段](https://github.com/run-llama/llama_index/blob/722cb67ca4e52c8c4d6ef8c5e99b7f6c9f57e244/llama-index-core/llama_index/core/indices/document_summary/base.py#L203)。
   - 潜在的修复方案包括 **metadata 限制**，并寻求如 **qdrant** 或 **pg_vector** 等替代方案，正如一位用户所建议的那样，展示了社区解决问题的能力。
- **树莓派上的 RAG 革命**：@pavan_mantha1 展示了一个在 **Raspberry Pi** 上运行的 **RAG pipeline**，利用了 **Docker** 和 **Ollama**，引发了关于小型设备如何实现高性能的讨论，详见此 [社区亮点](https://twitter.com/llama_index/status/1808292764129583179)。
   - 这一壮举强调了 AI 系统对资源受限环境的适应性，并赢得了社区对高效计算的赞赏。
- **通过 OpenContracts 实现文档民主化**：**OpenContracts** 作为一个开源的文档分析利器出现，它利用 **LLMs** 进行标注，并由 **Llama Index** 提供支持。该工具的发布记录在 [Twitter](https://twitter.com/llama_index/status/1808528869252812902) 上。
   - **GenAI native** 技术处于前沿，该项目致力于让 **AI 驱动的文档处理** 变得广泛可用。
- **网络研讨会汇聚智慧**：**Weights & Biases** 合作举办了一场网络研讨会，旨在深入探讨 **RAG pipeline** 的构建，并对一年的开发历程进行了批判性分析，详见 [此处](https://twitter.com/llama_index/status/1808589017744880062)。
   - 该活动在解决评估挑战方面至关重要，强调了在 AI 应用领域对成长和知识共享的承诺。
- **Agentic RAG 引发读者关注**：在文章 [**释放 AI 潜力**](https://medium.com/ai-advances/unleashing-ai-potential-agentic-rag-with-llamaindex-claude-3-5-sonnet-and-mongodb-ea126164a801) 中，**Agentic RAG** 与 **LlamaIndex**、**Claude-3.5 Sonnet** 以及 **MongoDB** 结合，催生了关于前卫 AI 策略的讨论。
   - 其即将到来的推广预示着人们对 AI 基础设施变革性方法的兴趣激增，正等待着社区敏锐的思想家们去探索。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Tortoise-TTS 迁移至 GGML**：一位社区成员成功将 **Tortoise-TTS** 迁移到了 **ggml**，为[实时文本转语音](https://github.com/balisujohn/tortoise.cpp)操作开启了可能。该仓库增强了对 **CUDA 和 CPU 的支持**，为开发者提供了更广泛的平台选择。
   - 这一举措吸引了 AI 开发者投入到优化 **transformers** 和 **diffusion models** 以加快推理过程的工作中，对于热衷于性能提升的人来说，这是一个极具吸引力的项目。
- **vLLM 在 Hermes 2 Pro 中的工具调用取得成功**：**vLLM 中工具调用 (tool calling)** 在 Hermes 2 Pro 上的集成已成功执行，使项目接近尾声。这一进展引发了关于如何高效处理 'content' 和 'tool_calls' 的新讨论。
   - 随后的讨论围绕在 **Hermann 3 训练**中加入 `<scratch_pad>` 展开，旨在实现更细致的解析方法，并与类似于 OpenAI 框架的标准保持一致。
- **Genstruct 7B 的指令创新**：[**Genstruct 7B 模型**](https://huggingface.co/NousResearch/Genstruct-7B) 借鉴了 Ada-Instruct，通过从文档中生成精确指令而脱颖而出，从而促进了用于指令微调 (instruction finetuning) 的定制数据集的创建。
   - 该技术面向 AI 工程师，重点展示了将原始文本语料库融合到对话数据集中的方法，为无需巨额投资的数据集扩展提供了智能解决方案。
- **CommandR 在 Huggingface 手中崛起**：**Huggingface** 为 Cohere 的 CommandR 提交了一个 [pull request](https://github.com/cohere/CommandR)，引入了改进工具使用和检索增强生成 (RAG) 技术的进展。
   - 他们的创意投入通过结合前导语 (preamble) 和智能内容组织（由 Jinja 模板支持）重构了系统提示词，表明了在 RAG 开发方面强大的协作潜力。
- **GraphRAG：Microsoft 出品的基于图的杰作**：Microsoft 发布了一个名为 [**GraphRAG**](https://github.com/microsoft/graphrag) 的新型检索增强生成框架，专注于模块化设计，以提升信息检索和内容生成的效率。
   - GraphRAG 可以在 GitHub 上获取，作为一项标志性成果，它提供了深入的定制选项，这对于当今动态的 AI 研究和开发环境至关重要。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Ubuntu 上的 Mojo：安装波折**：用户在 **Ubuntu 24.04/Python 3.12.3** 上使用 **Mojo** 时遇到障碍，遇到了兼容性问题，特别是与 **max-engine** 相关的问题。社区分享了一份使用 Python 3.11 成功安装的[分步指南](https://docs.modular.com/mojo/manual/python/#resolving-issues)。
   - 讨论集中在 `List[String]` 缺少 `Stringable` 特性 (trait) 从而影响可打印性的问题上，并在 [GitHub](https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/src/builtin/str.mojo#L23) 上提供了详细参考。用户注意到由于循环展开 (loop unrolling) 及其编译时间，程序存在**不固定的启动时间**。
- **Strassen 算法的速度惊人？但不稳定**：根据 [GitHub](https://github.com/RedKinda/Mojo-Marathons/) 上分享的讨论和基准测试，在 1024x1024 矩阵上，**Strassen 算法**的表现不如朴素向量化方法（后者达到 **70 GFlops**，而 Strassen 为 **50 GFlops**）。
   - 开发者对其**数值稳定性**表示担忧，当针对不同类型和大小的矩阵进行调整时，潜在的不稳定性可能导致测试失败。
- **SPIRAL：旋转出新的高性能代码**：[SPIRAL 项目](http://www.spiral.net/) 旨在自动化 DSP 算法的开发，有时性能甚至超过 MKL。它专为直接硬件任务量身定制，可能是优化一系列数值运算的关键。
   - 讨论强调了在并行处理和向量化之外优化算法的复杂性，暗示了递归方法相较于迭代方法在缓存局部性 (cache locality) 方面的优势。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Apple 敲开 OpenAI 董事会大门**：据 [Bloomberg](https://www.bloomberg.com/news/articles/2024-07-02/apple-to-get-openai-board-observer-role-as-part-of-ai-agreement) 报道，**Apple** 将在 OpenAI 获得一个**董事会观察员席位**，由 **Phil Schiller** 出任，这标志着科技协作中的**战略举措**。
   - 社区分析认为，**Apple 的合作伙伴关系**可能比 **Microsoft 的投资**带来更大的收益，重点关注独家应用集成等优势，并引发了关于 AI 进步中企业策略的辩论。
- **Moshi 掌握多模态真谛**：**Kyutai Labs** 凭借 **Moshi** 惊艳全场，这是一款具有开创性的**实时音频 LLM**，在演示中展示了 **150ms 的延迟**，其**卓越的同声传译**能力、**速度**和**多模态实力**获得了高度认可。
   - 发布**开源模型**以促进社区创新的计划受到赞赏，包括 Moshi 核心的 **7B 多模态 LM** 和 **VQ-VAE 编解码器**，这些模型有望重新定义端侧交互和用户体验。
- **代码的宪法难题**：辩论者引用了 [EFF 对 SB 1047 的观点](https://www.eff.org/deeplinks/2015/04/remembering-case-established-code-speech)，探讨了将**模型权重**和**代码**视为言论的辩护，并将其与**言论自由**和 **3D 打印枪支设计先例**进行了类比。
   - 围绕**模型权重**作为一种表达形式的本质展开了激烈讨论，质疑这些算法输出是否应享有与**语言类似的保护**，并强调了它们在现代通信和创新中不可或缺的作用。
- **Claude 3.5 粉丝团壮大**：随着 **Claude 3.5** 的发布，社区掀起了一阵兴奋浪潮，引发了**热烈反响**并与之前的版本进行了对比，专业人士注意到其在性能和潜在应用领域方面的飞跃。
   - 对 **Claude TM** 的支持将其市场定位比作知名品牌的成功策略，成员们敦促加大推广力度，以匹配其声名显赫的竞争对手，并强调其**增强的功能**。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Azure 深受 429 错误困扰**：从 **PyPDFium2Loader** 切换到 **AzureAIDocumentIntelligenceLoader** 导致持续出现 **429 错误（请求过多）**，凸显了所面临的速率限制挑战。
   - 社区辩论包括寻找在不牺牲效率或准确性的情况下绕过 Azure **速率限制**的方法。
- **PDF 难题与 Markdown 迷思**：尝试通过 [marker](https://github.com/VikParuchuri/marker) 将 PDF 转换为 Markdown 时，在面对复杂的表格格式时遇到了困难，合并单元格导致了严重的迁移痛苦。
   - 尽管 **Azure Document Intelligence** 提供了更优的解析精度，但开源工具的吸引力依然存在，促使人们寻找本地解决方案。
- **LangSmith 丢失链接**：有报告称 **LangSmith** 意外停止了调用追踪，引发了关于 LangChain 内省功能鲁棒性的讨论。
   - 随着用户努力检测**追踪机制**中的缺陷，技术审查随之展开，暗示了 LangChain 基础设施中隐藏的 Bug。
- **CriticGPT 围剿代码错误**：AI 社区剖析了 OpenAI 的 **CriticGPT** 计划，该计划旨在识别和修正 **GPT-4** 的错误，一段易于理解的[视频解释](https://youtu.be/4PgcaIfwLjo)在同行中流传。
   - 围绕 **CriticGPT** 如何标志着向自我纠正 AI 系统迈进展开了热烈对话，预示着自动化代码可靠性的升级。
- **Mac 邂逅 Toolio：开源的意外之喜**：随着 **Toolio** 闯入开源领域，Mac 爱好者们欢欣鼓舞，它承诺在 macOS 上实现私有 **LLM** 部署，正如其 [YouTube 展示](https://www.youtube.com/watch?v=9DpQYbteakc)中所宣称的那样。
   - 这一创新为用户提供了快速推理和 **JSON schema 输出**，满足了对增强控制和个性化的需求。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **强化你的 llamafile Linux 装备**：为了获得最佳的 **llamafile** 性能，工程师建议个人项目使用 **3090/4090** 等 GPU，专业环境使用 **A6000/RTX 6000 Ada**；CPU 则推荐较旧的 **EPYC**，因为它们具有卓越的核心数和 PCIe 支持。
   - 讨论表明，用户更倾向于拥有大容量 VRAM 的 GPU，并强调 24GB VRAM 对于管理 **33B 参数** 左右的模型是必要的。
- **VRAM：越大越好**：AI 爱好者强调了充足 VRAM 对运行大型模型的重要性，并提醒注意使用 FP16 模式，因为与微小的质量提升相比，它会大幅增加 VRAM 占用。
   - 社区交流强调了 **q4** 配置可以在 **24GB VRAM** 下流畅处理 33B 参数模型，为大型模型管理设定了基准。
- **使用 Syncthread 的 CPU 推理妙招**：利用 syncthread 技巧进行 CPU 推理的创意用法受到关注，这可能会改变我们处理 **基于 CPU 的学习** 的方式。
   - [YouTube 演讲](https://www.youtube.com/live/5zE2sMka620?feature=shared&t=3140) 的链接详细介绍了该技术，吸引了社区的注意。
- **Threadripper 驯服 llama3 70B 模型**：一位资深 AI 工程师报告了使用强力的 **Threadripper CPU** 成功运行 **llama3 70B** 模型的情况，这标志着 CPU 在实际应用中可能取得飞跃。
   - 这一成功的部署意味着 Threadripper 有能力在由 GPU 主导的领域中占据一席之地。
- **应对 RK3588 NPU 上的 llamafile 挑战**：将 **llamafile** 与 **Rockchip RK3588 NPU** 硬件集成引发了从业者的咨询，建议使用 **v0.8.9** 等软件版本以规避兼容性问题。
   - 这一讨论指出了在利用特定版本以获得最佳硬件性能时，需要考虑的更广泛挑战和因素。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **权衡 phi mini 的新权重**：**phi mini** 已更新新权重，但与其原始仓库保持一致，这引发了用户关于 **torchtune** 流程是否需要调整的疑问。
   - 关于旧方法是否仍然有效的推测依然存在，但共识似乎倾向于可以平滑过渡，无需重大更改。
- **梯度与 Epoch：Torchtune 训练的曲折**：关于最佳 **训练策略** 展开了热烈讨论，对比了使用 **梯度 8 vs 16** 以及调整 Batch Size 和 Epoch 变化是否能产生更好的结果。
   - 为了协助解决这一难题，社区成员使用 **Wandb 来跟踪和记录性能指标**，并分享见解以优化训练过程。
- **转换难题：HF 格式处理**：关于模型转换细节的疑问不断增加，特别是为什么在 **HF** 和 **torchtune** 使用的多头格式之间转换时，`num_heads`、`num_kv_headers` 和 `dim` 等参数是必需的。
   - 格式转换固有的复杂性被凸显出来，成员们交流了有效应对这一技术领域的技巧。
- **Checkpoint 冠军：Torchtune 的救星**：在 **torchtune** 中引入 **FullModelHFCheckpointer** 引起了关注，因为它能够将模型无缝转换为 **HF 友好格式**。
   - 该工具因弥合了不同机器学习基础设施之间的兼容性鸿沟而受到赞誉，确保了更广泛的可访问性和实用性。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **使用 Stockfish 与 LLM 应对将军挑战**：社区成员正在探索将 **Stockfish** 游戏数据与 **LLM** 结合，以增强战略推理能力，并顺便开发一个快速的 **国际象棋引擎**。
   - 讨论围绕使用 **国际象棋数据微调 LLM** 的技术障碍展开，辩论了其实际意义和过拟合风险。在 **LLM** 中使用 **Stockfish** 等现有工具的理论引起了广泛兴趣。
- **Slack 机器人引入 Cohere**：一款新型 **Cohere Slack 机器人** 问世，展示了快速处理 **Slack** 3 秒请求要求的能力，证明了 **Cohere API** 的效率。
   - 创建者提议分享代码并编写文档，这激发了社区的热情，许多人期待关于将 **Cohere** 与通信平台集成的详细指南。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **速度之声：Kyutai Moshi 的 Audio LLM**：[**Kyutai Moshi**](https://www.moshi.chat/?queue_id=talktomoshi) 发布了一个几乎无延迟运行的实时 **Audio LLM**，尽管反馈指出其音调略显机械。它因交互速度极快而受到赞誉，有时甚至快到会打断用户说话。
   - 用户 *Mikebirdtech* 的见解强调了该系统的速度，表示它**快得有些过头**，因为它可能会在自然对话的停顿期间打断用户。
- **透明智能：OI 眼镜概念**：在一次推测性对话中，用户 johnlenflure 提出了将 **OI** 集成到眼镜中的想法，设想了一个由 **OpenInterpreter** 功能支持的**智能眼镜**未来。
   - 随后没有进一步的细节或技术讨论，该概念在成员中仍处于高度抽象的兴趣阶段。
- **Open Interpreter 模组游戏化**：用户 **Nonadjective.eth_55058** 正在寻求关于将 **Open Interpreter** 集成到游戏中的建议，旨在开发一个可运行的概念验证，即使最初可能比较简陋。
   - 这反映了社区内探索和扩展 **Open Interpreter** 模组化潜力的兴趣日益增长，表明了向可定制交互体验发展的趋势。
- **与 Open Interpreter 的项目兼容性**：一系列项目被强调为与 **Open Interpreter** 兼容，包括 **Open interpreter、taxyai、clickolas cage、self-operating computer、pywinassistant** 和 **GPT computer assistant**。
   - 探索并可能配置这些项目与 **Open Interpreter** 协同工作的兴趣显而易见，这表明开发者处于一个动态且协作的环境中。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **量化困惑：LoRA vs QLoRA**：成员们深入研究了量化技术，讨论了 **LoRA** 和 **QLoRA** 在应用上的多样性，强调 LoRA 利用 8-bit 量化，而 QLoRA 则进一步推向 4-bit，并引用了 [QLoRA 论文](https://arxiv.org/abs/2305.14314)中的全面处理方法。
   - 一场对话澄清了 **QLoRA** 的定位：如论文《*QLoRA: Efficient Finetuning of Quantized LLMs*》所述，它能够精巧地在单个 48GB **GPU** 上微调 65B 参数模型，其性能与 16-bit 微调非常接近。
- **VRAM 烦恼与 CUDA 灾难**：**Google Colab** 的难题浮出水面，一位用户正苦于 **torch.cuda.OutOfMemoryError**，指出在 **Google Colab** 上尝试分配 172.00 MiB 导致失败。
   - 贡献者们一致认为 **VRAM** 是瓶颈，并建议增加 VRAM 以促进无缝运行，突显了硬件在运行 **axolotl** 等模型时的重要性。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Docker 进驻 AI Town**：社区对 AI Town 的 **Docker 镜像**感到兴奋，并呼吁贡献力量以增强该工具的可访问性。
   - Docker 化工作旨在简化设置流程，爱好者建议将广受好评的 [Windows WSL 设置指南](https://github.com/Ikkitsuna/AI-Town-Windows-Setup-WSL-method) 作为 **Pull Request** 提交到主仓库。
- **Dockertown 的 API 端口风波**：一位资深开发者在将 AI Town 移植到 Docker 时遇到了 API 通信问题，特别是与 **Ollama API** 的通信，并承诺很快会分享修复方案。
   - 尽管存在技术障碍，移植工作仍在推进，社区保持关注以确保实现无缝连接。
- **Convex 接入 Docker**：为了简化 AI Town 的体验，一位成员正在调整 Docker 以自动下载 **Convex**，预见未来用户的使用将更加顺畅。
   - 预计通过 Docker 自动设置 Convex 的功能将在 UTC+4 时间晚上 8 点前投入使用，这表明社区正积极参与以提高用户效率。
- **AI Town 的 Docker 测试盛宴**：一位成员在他们的 **Legion Go** 设备上运行 Docker 集成测试的举动增强了对该移植版本性能的信心，表明已准备好提交 **Pull Request**。
   - 正在招募志愿者进行 Docker 集成测试，并期望合并成功的结果，展示了 AI Town 开发者社区的协作精神。



---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Gradio 部署困局困扰工程师**：成员们在 Modal 上部署**使用 Gradio 的 RAG 应用**时遇到障碍。一段[讨论](https://discord.com/channels/1238365980128706560/1241044231829848125/1257903763998511155)指出，该应用在本地运行正常，但在 Hugging Face Spaces 上无法工作。
   - 建议通过 **Modal Slack** 作为该问题的紧急求助渠道，希望社区支持能为这一部署难题提供解决方案。
- **DeepSpeed 配置难题引发辩论**：成员们在尝试启用**数据分片 (data sharding)**而不选择模型分片时，**DeepSpeed** 的配置引发了热议，详见[他们的交流](https://discord.com/channels/1238365980128706560/1242542198008975430/1257990993446436926)。
   - 关于 **DeepSpeed 设置** 的澄清和协助成为了迫切的需求，凸显了需要填补的知识空白。
- **Hugging Face 交付难题**：由于无法在 Hugging Face 上分享**私有代码**部署，成员们表达了困扰，因为私有空间不支持 **sharing=True**，讨论见[此处](https://discord.com/channels/1238365980128706560/1242564125524234361/1257930601730674719)。
   - 在 **Modal** 上操作的尝试也遇到了挫折，情绪一度波动，引发了对私有代码协作替代方法的寻找。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **法律专家关注 LLM 精准度**：来自 [Screens 的新报告](https://www.screens.ai/blog/screens-accuracy-evaluation-report)通过将 LLM 在合同审查中的表现等同于机器学习分类问题进行分析，声称其系统的**准确率达到 97.5%**。
   - 报告探讨了评估长文本回复准确性的挑战，建议基于分类的方法可以增强 LLM 在谈判和文档摘要等法律任务中的有效性。
- **面向大众的 Prompt Tuning**：**Evan_04487** 正在寻找一种简单、托管的 Prompt Tuning 工具，以便设计师和经理等非技术专家也能运行 Prompt 变体并查看结果。
   - 理想的解决方案应该是免费增值 (freemium) 服务，足够简单以处理低风险任务，并能处理大约两打变量，这与针对关键任务的复杂、自管基础设施形成对比。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Datasette 在数据新闻中的发现**：Derek Willis 分享了[一篇关于外国礼物的文章](https://thescoop.org/archives/2024/06/22/all-foreign-gifts-around-us/index.html)，引发了人们对 **Datasette 在调查新闻中实用性**的兴趣。
   - 讨论涉及如何利用 Datasette 作为筛选公共记录和数据集的**强大工具**，强调了其在新闻透明度和问责制中的作用。
- **Datasette 深入挖掘数据**：爱好者们强调了 **Datasette** 对深度数据分析的意义，考虑了该工具处理复杂查询的能力。
   - 工程师们讨论了 **Datasette** **改变数据驱动故事**的潜力，强调了在数字时代，可访问且可解释的公共数据的重要性。

---

# PART 2: 各频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1257776054966157312)** (201 条消息🔥🔥): 

> - `unsloth 中的 SEQ_CLS 支持`
> - `使用 Jellyfin 作为 Plex 的替代方案`
> - `在另一种语言上微调模型`
> - `分享 Colab 笔记本和账号信息`
> - `LoRA 微调的 VRAM 需求` 


- ****Phi-3 Mini 迎来重大更新****：Phi-3 Mini 模型获得了显著更新，被比作 3.5 Mini，预计明天将发布包含 Gemma 2 支持在内的全新量化版本。值得注意的是，**Phi-3 Mini** 的最新增强功能将提升其性能，实现快速高效的处理。
- ****Moshi 的实时语音模型引发 AI 社区轰动****：*Kyutai Labs* 推出了 '**Moshi**'，这是一个 7B 多模态 LM，能以低延迟生成高质量的文本和音频，响应时间仅为 160ms。该模型经过 RLHF 微调，并设计为开源，支持各种后端配置，并计划在未来进行更新。
- ****unsloth 已添加 Gemma 2 支持****：unsloth 团队宣布发布对 Gemma 2 的支持，允许用户以更高的效率为高级 AI 任务微调模型。初步的用户反馈显示，**Gemma 2** 在提供的笔记本上运行良好，尽管一些用户遇到了初始设置问题。
- ****unsloth 中的 SEQ_CLS 支持与微调****：用户讨论了 unsloth 中 SEQ_CLS 支持在微调任务中的功能，建议在多类分类中使用 JSON 输出。使用 Phi-3 模型的经验表明，采用这种方法可以显著提高改进效果和学习速度。
- ****关于将基于图的 RAG 集成到 unsloth 的讨论****：人们对将 Microsoft 的 **graph-based Retrieval-Augmented Generation (RAG)** 系统集成到 unsloth 中以增强其功能表现出浓厚兴趣。用户推测了其益处，强调了 AI 的进步和优化的工作流程。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/AIatMeta/status/1808157832497488201">来自 AI at Meta (@AIatMeta) 的推文</a>: 📣 来自 Meta GenAI 的新研究，介绍 Meta 3D Gen：一个在 1 分钟内从文本端到端生成 3D 资产的新系统。Meta 3D Gen 是一个新的组合 AI 系统，可以生成高...</li><li><a href="https://moshi.chat/?queue_id=talktomoshi">moshi.chat</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/peft/en/developer_guides/model_merging">模型合并</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/peft/en/developer_guides/lora#merge-adapters">LoRA</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit">unsloth/Phi-3-mini-4k-instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/live/hm2IJSKcYvo">Moshi 主旨演讲 - Kyutai</a>: 未找到描述</li><li><a href="https://huggingface.co/internlm/internlm2_5-7b-chat">internlm/internlm2_5-7b-chat · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/microsoft/graphrag">GitHub - microsoft/graphrag: 一个模块化的基于图的检索增强生成 (RAG) 系统</a>: 一个模块化的基于图的检索增强生成 (RAG) 系统 - microsoft/graphrag</li><li><a href="https://x.com/reach_vb/status/1808528557431210236">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>: 太棒了！@kyutai_labs 的 Moshi 震撼全场！🇪🇺/acc。架构 1. 7B 多模态 LM (语音输入，语音输出) 2. 双通道 I/O - 流式 LM 持续生成文本 Token 以及...</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral, Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80%</a>: 微调 Llama 3, Mistral, Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1258151227787706450)** (1 条消息): 

> - `Gemma 2 发布`
> - `Phi 3 mini 更新`
> - `Finetuning 改进`
> - `增加 Context lengths`
> - `Notebooks 和 4-bit 模型` 


- ****Gemma 2 提升 Finetuning 速度****：Unsloth 现在支持 **Gemma 2**，其 **Finetuning** 速度快 **2 倍**，且显存占用减少 **63%**。查看 [博客文章](https://unsloth.ai/blog/gemma2)。
- ****实现更长的 Context Length****：在 40GB GPU 上，你可以使用 Unsloth 通过 QLoRA 将 **Gemma 2 (27B)** 微调至 **9.7K Context lengths**，而 HF+FA2 仅支持 3K 长度。Unsloth 还支持在 24GB 显卡上为 9B 模型提供 **11K Context lengths**。
- ****提供免费的 Colab Notebooks****：提供 **Gemma 2 (9B)** 和 **27B** 的免费 Notebooks，包括适用于 9B 模型的 [Colab notebook](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)。
- ****Phi 3 Mini 迎来更新****：**Phi 3 mini** 也进行了更新，新的 [Instruct model](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit) 已在 Hugging Face 上线。
- ****实验并分享结果****：鼓励社区在 Unsloth 平台上进行实验、测试并讨论模型结果。**@here** 发出了分享结果的特别号召。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/gemma2"> 使用 Unsloth 微调 Gemma 2</a>：通过 Unsloth 以 2 倍速度、减少 63% 显存 VRAM 的方式微调 Google 的新 Gemma 2 模型！支持 9B 和 27B 参数。</li><li><a href="https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1257918970460373063)** (3 条消息): 

> - `mahiatlinux 的评论`
> - `theyruinedelise 的反应`
> - `mahiatlinux 的回复` 


- ****频道内的积极互动****：一位成员通过说 *'That's good lol'* 表达了他们的愉悦，这引发了一段简短而积极的交流。
- ****惊讶与认同****：另一位成员对这种积极情绪表示赞同，并提到 *'惊讶于它效果这么好'*，这得到了进一步的认可。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1257847320863571998)** (63 messages🔥🔥): 

> - `Llama.cpp quantization issues` (Llama.cpp 量化问题)
> - `Loading model speed on Colab` (Colab 上的模型加载速度)
> - `Unsloth compatibility with Gemma 2` (Unsloth 与 Gemma 2 的兼容性)
> - `Inference on CPU after fine-tuning with Unsloth` (使用 Unsloth 微调后的 CPU 推理)
> - `Training issues with Huggingfaces SFTTrainer and Unsloth` (使用 Huggingfaces SFTTrainer 和 Unsloth 时的训练问题)


- **Llama.cpp 量化困扰成员**：一位用户询问了关于 **llama.cpp** 尚未解决的量化问题，另一位用户承诺今天会优先检查这些问题。还有一位用户分享了相关问题并链接到了 [相关的 Discord 消息](https://discord.com/channels/1179035537009545276/1179035537529643040/1257661677466554489)。
   - *即便按照说明操作，仍然遇到同样的问题。*
- **在 T4 GPU 上加速模型加载**：用户讨论了如何在 Colab T4 GPU 上缩短 **模型加载** 时间，特别是针对一个 7GB 的模型。建议包括将模型保留在 VRAM 中以避免重复加载，尽管有人认为在 Colab 上这是不可能的。
   - "每次我运行 CUDA 代码将模型（7GB）从磁盘加载到 GPU 显存时，大约需要 30 秒。我能让模型加载得更快，或者更好的是只加载一次模型吗？"
- **结合 Jax 使用 JIT 库**：一位用户询问了关于在需要线性代数知识进行 GPU 优化的 prompt-answer 数据集训练中，如何结合 **Jax** 使用 **JIT 库**。鉴于训练时间过长，他们还询问了 **RAG** 是否比传统的 fine-tuning 是更好的选择。
   - *我这么说对吗？不确定是否有人这样做过。*
- **Unsloth 新增 Gemma 2 支持**：用户讨论了与 **Unsloth 相关的各种错误和问题**。值得注意的是，有人提到 **Unsloth 最近新增了对 Gemma 2 的支持**。
   - "就在刚才 Unsloth 增加了对 Gemma 2 的支持，你可以更新并重试！"
- **使用 Huggingfaces SFTTrainer 和 Unsloth 进行训练时的问题**：一位用户分享了在本地尝试使用 **Huggingfaces SFTTrainer** 和 Unsloth 开始训练时遇到的错误，这导致了该用户的权限问题。另一位用户引用了一个与缺失 Python.h 相关的 [GitHub issue](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1038)。
   - *这是在本地吗？最好的办法就是去 Google 搜一下。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1038">Make Error, fatal error: Python.h: No such file or directory compilation terminated. · Issue #1038 · CMU-Perceptual-Computing-Lab/openpose</a>：在从 /home/sclab/Downloads/openpose/3rdparty/pybind11/include/pybind11/pytypes.h:12:0 包含的文件中...</li><li><a href="https://huggingface.co/docs/transformers/en/internal/generation_utils#transformers.TextStreamer)">Utilities for Generation</a>：未找到描述
</li>
</ul>

</div>

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1257842275682095285)** (241 条消息🔥🔥): 

> - `支持更多数据集的教程/中级 Colab/Kaggle Notebook`
> - `对社区 Notebook 的改进和建议`
> - `Notebook 的内存管理和优化技术`
> - `针对 Unsloth 优化的文本分类 Notebook`
> - `Docker 中的密钥管理与应用部署` 


- ****flail_ 提供的中级 Colab Notebook****：一位成员介绍了一个[教程/中级 Colab/Kaggle Notebook](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t)，该 Notebook 支持多种数据集（包括 dolphin, ultrachat, capybara, slimorca），并提供了可根据模型和数据集大小缩放 LoRA rank 的辅助函数。
   - 讨论了改进建议，例如使用 `shuffle(seed=42)` 以保证可复现性，以及避免使用 `flatten_indices()`；此外还探讨了[多个关于内存管理技术的问题](https://stackoverflow.com/a/55340037/3548976)，涉及使用 `torch.cuda.empty_cache()`。
- ****timotheeee1 提供的文本分类 Notebook****：分享了一个专门针对文本分类优化的[修改版 Unsloth Notebook](https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb)，该 Notebook 通过批处理推理（batched inference）实现高效评估，并通过将分类头（classification head）拟合到特定 token 标签来提高稳定性。
   - 该 Notebook 具有针对分类任务的输入裁剪功能，可节省 VRAM，并结合 `ignore_index` 将模型能力引导至必要的预测，避免浪费资源。
- ****内存管理技术讨论****：对话涉及在循环中加入 `gc.collect()` 和 `torch.cuda.empty_cache()` 以处理显存溢出（OOM）问题，多位成员分享了更高效管理内存的方法。
   - 成员们辩论了睡眠间隔（sleep intervals）和循环结构对清理内存的有效性，并参考了来自 Unsloth 的代码片段。
- ****Docker 密钥管理****：一位成员就如何在不直接上传 `.env` 文件的情况下处理 Docker 容器部署中的密钥管理寻求建议，最终决定使用 `--env-file` 标志来安全地传递环境变量。
   - 讨论了不同的方法，如利用本地注册表并使用 `docker save my-app > my-app.tar` 配合 `ctr images import`，以确保安全高效的部署工作流。
- ****社区 Notebook 置顶与改进****：建议置顶重要的 Notebook（如支持多数据集和文本分类的 Notebook），以免在聊天记录中丢失，并提高社区成员的获取便利性。
   - 在进一步审查和完善后，这些资源也计划添加到 Unsloth 的 GitHub 页面。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://stackoverflow.com/a/55340037/3548976)">如何在 PyTorch 中清理 CUDA 内存</a>：我正尝试获取已经训练好的神经网络的输出。输入是 300x300 的图像。我使用的 batch size 为 1，但仍然遇到 CUDA error: out of memory ...</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb (GitHub)</a>：使用 Llama 和 BERT 进行文本分类的脚本 - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/huggingface/trl/issues/632#issuecomment-1972630547">[DataCollatorForCompletionOnlyLM] input_ids 是否应该包含标签？ · Issue #632 · huggingface/trl</a>：我正在使用 DataCollatorForCompletionOnlyLM 训练聊天助手。我发现数据整理器在 batch['labels'] 中包含了我想微调的回答...</li><li><a href="https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1257958621002731520)** (10 条消息🔥): 

> - `在本地系统中使用 Unsloth 的问题`
> - `Gemma2 更新发布时间线`
> - `对最新 Gemma 的支持`
> - `关于 Gemma 的讨论`
> - `使用 PHI 进行 Java 评估` 


- ****Unsloth 本地设置中的配置错误****：一位用户报告了在本地系统使用 Unsloth 时的一个配置设置错误：*`config.hidden_act` 被忽略，请改用 `config.hidden_activation`*。
- ****Gemma2 更新预计很快发布****：一位成员提到 **Gemma2** 的更新预计将在 **1-2 天内**发布。
- ****目前尚不支持最新的 Gemma****：目前不支持最新版本的 **Gemma**，需要等待新的更新。
- ****Gemma 发布问题****：一位成员强调 **Gemma** 应该在今天发布，但由于博客问题有所延迟。
- ****注意到 PHI 的 Java 评估表现****：一位用户评论了 PHI 在 Java 方面的强劲表现，在评估中达到了 **93** 分。


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1257776610099200011)** (194 条消息🔥🔥): 

> - `OpenAI GPT-4 订阅与性能问题`
> - `AI21 的 Jamba 模型发布与讨论`
> - `用户使用 AI 进行编码和编程的经验`
> - `实时与开源 AI 模型的辩论`
> - `用于实时对话的 AI：Moshi 演示` 


- ****订阅困扰困扰着 GPT-4 用户****：一位用户表达了对 GPT-4 订阅的沮丧，理由是难以达到消息限制以及升级后的性能问题。社区建议了替代的问题解决方法，并强调了模型的局限性。
- ****AI21 的 Jamba 承诺高科技基准测试****：AI21 Labs 发布了 ['Jamba'](https://www.ai21.com/blog/announcing-jamba)，它具有尖端的混合架构，结合了 Mamba SSM 技术和 Transformer 架构，宣称拥有 **256K 上下文窗口 (context window)** 和极具吸引力的价格。
- ****使用 AI 模型进行编码的障碍****：讨论揭示了使用各种 AI 模型执行编码任务的挑战——特别是在生成正确且完整的代码方面。用户分享了使用 **GPT-4**、**Claude Sonnet 3.5** 和 **Jamba** 的混合体验，反映了任务准确性的差异。
- ****开源实时 AI 工具****：新发布的 ['Moshi'](https://moshi.chat/?queue_id=talktomoshi) 实时 AI 对话工具因其开源承诺而引起关注，尽管其当前模型的智能程度存在一些局限性。
- ****开源竞争加剧 AI 竞赛****：随着关于 **Moshi** 和其他开源能力的讨论出现，用户辩论了这些工具相对于 OpenAI 等专有模型的竞争优势。将先进 AI 集成到日常科技中的竞赛凸显了不断演变的格局。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: 首次推出基于 Mamba 的生产级模型，提供一流的质量和性能。</li><li><a href="https://moshi.chat/?queue_id=talktomoshi">moshi.chat</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1257988005789171813)** (11 条消息🔥): 

> - `新 TTS 模型语音的可用性`
> - `数据策略执行`
> - `在具有 Google 搜索能力的命令提示符中运行 ChatGPT`
> - `订阅价格令用户沮丧`
> - `嵌套 GPTs 功能` 


- **数据策略执行受到批评**：围绕数据策略执行的*美德绑架 (virtue signalling)* 引发了挫败感，因为 **ChatGPT** 标记并删除了一个数据集，原因仅是一个 15 岁儿童的单一条目，尽管该数据没有任何不当活动。用户对信息从“我们可能犯了错误”转变为专制的“我们是对的你是错的，已删除！”感到不满。
- **在具有 Google 搜索功能的命令提示符中运行 ChatGPT**：一位用户成功通过**命令提示符**运行 **ChatGPT**，使其能够使用 Python 编码执行 **Google 搜索**。
- **订阅价格令用户沮丧**：用户对 **OpenAI** 偶尔更改模型参数表示沮丧，尽管有些人为此服务每月支付高达 **$60**。对于价值 $60 的专业工具，意见不一，有些人认为如果能使生产力翻倍，这个价格是合理的。
- **关于嵌套 GPTs 的查询**：一位用户询问了一个 **GPT** 调用其他 **GPTs** 的可能性，并想知道这种嵌套可以达到多深。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1257800705884819516)** (117 条消息🔥🔥): 

> - `GPT 模型未按预期回答问题的问题`
> - `使用 GPT API 创建 PDF 文档的困难`
> - `改进 Prompt Engineering 以获得更好的任务表现`
> - `使用 DALL-E 进行 AI 驱动图像生成的挑战`
> - `使用 AI Prompt 开发员工表彰计划` 


- ****GPT 在遵循指令方面的困扰****：一位用户对 GPT 模型无法正确遵循详细指令表示沮丧，导致步骤跳过和流程不完整。建议的解决方案包括更新 Prompt、使用重新生成选项，或将指令结构化为更小的顺序部分。
- ****从 AI 输出手动创建 PDF****：一位用户在使用 AI 生成带有格式化产品标签的 PDF 时遇到问题，理由是自动添加 Logo 和调整文本大小存在困难。在 AI 无法满足需求后，他们选择将任务拆分为较小的手动编辑。
- ****改进 DALL-E 的 Prompt Engineering****：一位用户寻求改进使用 DALL-E 生成矢量图标的 Prompt 的帮助，报告了如多余阴影和无关元素等问题。建议包括简化和明确 Prompt，以避免冲突并确保精确输出。
- ****员工表彰计划的详细 Prompt****：分享了一个用于开发员工表彰计划的详尽 Prompt，具有明确定义的目标、表彰方法、标准、实施计划和反馈机制。这种结构化方法旨在创建一个全面且有效的计划。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1257800705884819516)** (117 条消息🔥🔥): 

> - `GPT 性能问题`
> - `改进 Prompt 结构和注意力控制`
> - `将文档转换为带有产品标签的 PDF`
> - `增强 AI 图标生成器`
> - `开发员工表彰计划` 


- ****通过结构化 Prompt 解决 GPT 性能问题****：成员们讨论了通过有效构建 Prompt 来提高 GPT 性能，例如使用如 `IF...THEN...` 之类的条件命令，并在 Prompt 中包含清晰、强力的注意力控制机制。
- ****关于将文档转换为带有产品标签的 PDF 的技巧****：对话涵盖了将文档转换为带有产品标签的 PDF 的挑战，特别是关于将文本和 Logo 放入预定义文档矩形框的问题。
- ****增强 AI 图标生成器的输出****：一位用户寻求 AI 图标生成器的帮助，强调了尽管有明确的 Prompt 指令，仍会生成多余的背景元素、阴影、轮廓和随机对象等问题。


  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1257774458471710794)** (166 条消息🔥🔥): 

> - `讨论与 LM Studio 相关的技术问题和更新。`
> - `比较不同的 AI 模型，如 Gemma 2、Llama 3 和 Mistral。`
> - `Gemma 2 模型的增强功能和 Bug，包括 tokenizer 和 attention 机制。`
> - `使用 LM Studio 进行 LLM 微调的难度和建议。`
> - `针对不同硬件配置的最佳模型和设置。` 


- **LM Studio 中的技术问题和更新**：用户报告并讨论了与 LM Studio 相关的各种技术问题，包括 CPU 利用率问题、不同版本的困难，以及特定提示词和响应生成方面的挑战。
- **Gemma 2 和其他 AI 模型：用户体验**：成员们分享了使用 Gemma 2 模型的经验，指出其在写作风格和性能方面有显著改进，尽管仍存在一些问题，如忽略系统提示词。用户还将其与 Llama 3 和 Mistral 等其他模型进行了比较，对其效果评价不一。
- **Gemma 2 的增强功能和 Bug 修复**：讨论强调了 Gemma 2 模型的最新更新，包括 attention 层修复和 tokenizer 改进。用户辩论了是否有必要重新下载模型以从这些更新中受益。
- **微调和模型建议**：用户表达了对针对特定任务微调 LLM 的兴趣，但官方澄清 LM Studio 目前不支持微调。建议使用其他方法和工具（如 RAG）进行特定定制。
- **针对不同硬件配置的最佳模型**：根据硬件配置给出了使用最佳模型的建议，建议笔记本电脑和其他 VRAM 有限的设备使用允许全 GPU offloading 的模型。具体建议包括针对 7B 尺寸需求使用 Gemma 9B，并指出当前游戏笔记本电脑在有效运行 LLM 方面的不稳定性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: 发现、下载并实验本地 LLM</li><li><a href="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF">bartowski/gemma-2-9b-it-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=Y08Nn23o_mY">How to give AI &quot;Memory&quot; - Intro to RAG (Retrieval Augmented Generation)</a>: 这是检索增强生成 (RAG) 的入门视频。RAG 非常适合赋予 AI 长期记忆和外部知识，降低成本，并且非常...</li><li><a href="https://huggingface.co/bartowski/WizardLM-2-8x22B-GGUF/tree/main">bartowski/WizardLM-2-8x22B-GGUF at main</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8197">Add attention and final logit soft-capping, update scaling factor to Gemma2 by abetlen · Pull Request #8197 · ggerganov/llama.cpp</a>: 此 PR 添加了缺失的 attention 层和最终 logit soft-capping。实现参考自 huggingface transformers。此外，Gemma2 应用了 hidden_size / ... 的 pre-attention 缩放。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1257779517288611902)** (52 条消息🔥): 

> - `dolphin-vision 在 LM Studio 中的兼容性`
> - `Gemma 2 模型性能和问题`
> - `运行大模型的系统和硬件要求`
> - `AI 模型的 RP 压力测试`
> - `Gemma 2 的代码生成能力` 


- ****Dolphin-Vision 兼容性受到质疑****：一位用户询问 [dolphin-vision](https://huggingface.co/cognitivecomputations/dolphin-vision-72b) 是否适用于 LM Studio，并对其格式和内存要求表示担忧。
- ****Gemma 2 的异常行为****：用户报告了 **Gemma 2** 模型的问题，包括重复符号以及在各种上下文长度下失败。即使使用不同的设置，用户也注意到它有时无法按预期运行。
- ****通过 RP 进行模型压力测试****：一位用户分享了通过角色扮演 (RP) 对模型进行压力测试的见解，强调这是揭示模型缺陷的一种方法。他们建议通过 RP 过程中需要编辑的输出百分比来衡量模型。
- ****Gemma 2 的内省能力受到赞赏****：尽管存在一些问题，用户仍称赞 **Gemma 2** 的内省能力和强大的上下文理解。在隐藏上下文检测和正确性方面，它比其他模型更具优势。
- ****寻求代码生成方面的改进****：用户讨论了模型在代码输出中插入占位符的挑战，即使有明确指令要求不要这样做。建议包括使用详细的系统提示词来更有效地引导模型的代码生成。


  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1257798297414795357)** (1 条消息): 

> - `LM Studio 0.2.27 Release`
> - `改进的 Gemma 2 支持`
> - `lmstudio.js 中的 Bug 修复`
> - `关于 lmstudio.js 的高级信息` 


- ****LM Studio 0.2.27 发布，带来增强的 Gemma 2 支持！****：LM Studio 0.2.27 现已支持 **Mac (M1/M2/M3)、Windows (x86 和 ARM64) 以及 Linux (x86)**。用户可以[下载它](https://lmstudio.ai)或重启应用以触发自动更新。
- ****Gemma 2 模型更新****：**Gemma 9B** 和 **Gemma 27B** 模型的性能提升归功于 [abetlen](https://github.com/abetlen)、[ngxson](https://github.com/ngxson)、[slaren](https://github.com/slaren)、[ggerganov](https://github.com/ggerganov) 等人的贡献。从 Hugging Face 社区页面下载更新后的模型：[Gemma 9B](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF)、[Gemma 27B](https://huggingface.co/lmstudio-community/gemma-2-27b-it-GGUF)。
- ****lmstudio.js 中的 Bug 修复****：[lmstudio.js] 团队修复了 **"invalid creation parameter" bug**（[issue #45](https://github.com/lmstudio-ai/lmstudio.js/issues/45)）。其他更新解决了关于 **无 GPU 支持** 的消息提示。
- ****针对高级用户的高级信息****：最新的 `llama.cpp` commit ID 为 **d08c20eddedb24515a3212e2de66bdff41a26b8c**，且 **OpenCL 后端** 现已再次捆绑用于 Windows 和 Linux 平台。然而，Gemma 2 **不支持 OpenCL**。
- ****在 Windows 上更新 AMD ROCm 扩展包****：使用 AMD ROCm 的 Windows 用户可以按照[说明](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#amd-rocm)更新其 ROCm 扩展包。Linux 版 ROCm 扩展包更新仍处于 **开发中**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://github.com/lmstudio-ai/lmstudio.js/issues/45)">Issues · lmstudio-ai/lmstudio.js</a>：LM Studio TypeScript SDK (预发布公开 alpha 版) - Issues · lmstudio-ai/lmstudio.js</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#amd-rocm).">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1257933929202319360)** (11 条消息🔥): 

> - `驱动器安装的基础功能问题`
> - `用户对模型加载的困惑`
> - `1080p 显示器上的缩放问题`
> - `新模型中的不支持架构提示`
> - `关于改进 LM Studio 界面的用户反馈` 


- ****驱动器安装的抱怨****：成员们对 LM Studio 由于 [Squirrel 限制](https://link.to.issue)而无法安装在选定驱动器上表示沮丧。目前，用户必须更改“My Models”文件夹来解决存储问题。
- ****模型加载困惑****：一位用户在加载模型时遇到困难，称尽管拥有充足的系统资源（如 15.90 GB RAM 和 NVIDIA GeForce GTX 950M），仍收到有关模型操作失败的错误消息。
- ****1080p 显示器上的缩放问题****：一位用户提到 LM Studio 在 1080p 显示器的 1/4 区域内缩放效果不佳，导致设置按钮缺失且布局混乱。这影响了多显示器工作流。
- ****不支持架构的警报****：成员们观察到新模型经常触发“unsupported arch”警报，类似于 DeepSeek Coder v2 的问题，这是由它们的配置文件引起的。
- ****呼吁提供更多元数据****：一位用户建议在 LM Studio 首页为模型添加发布日期和其他元数据，以增强易用性。这作为一个潜在的改进方向受到了好评。


  

---

### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1257907784893337711)** (3 条消息): 

> - `提示 Llama 3 70B 去除对话性语句`
> - `与 Qwen2 72B 相比，Llama 3 的提示词结果存在问题`
> - `使用 Gradio 应用创建用于角色扮演和角色沉浸的提示词工具` 


- ****提示 Llama 3 70B 且不带烦人的对话性语句****：一位用户询问如何提示 **Llama 3 70B** 跳过其回答开头陈词滥调的对话性语句。例如 *'What a wonderful thing it is to have a drink!'* 之类的短语被强调为多余内容。
- ****Qwen2 72B 与 Llama 3 的提示词成功率对比****：一位用户分享了他们在 **Qwen2 72B** 上成功去除对话性语句的经验，并对比了在 **Llama 3** 上实现相同效果的困难。
   - 他们表达了沮丧，指出尽管应用了相同的技术，但很难从 **Llama 3** 获得类似的提示词性能。
- ****用于角色扮演提示词的新 Gradio 应用****：一位用户介绍了一个新的 **Gradio 应用**，旨在为沉浸式角色体验创建角色扮演提示词。该应用包含动态变量来定义角色身份和场景。
   - 他们分享了一个示例提示词，并邀请大家提供改进建议，同时提供了 [应用链接](https://huggingface.co/spaces/xtreme86/System_roleplay_generator)。



**提到的链接**：<a href="https://huggingface.co/spaces/xtreme86/System_roleplay_generator">System Roleplay Generator - a Hugging Face Space by xtreme86</a>：未找到描述

  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1258108843963973642)** (3 条消息): 

> - `LMS 的能耗`
> - `Linux 与 Windows 上的硬件问题`
> - `LMS 与其他软件的 GPU 使用率对比`
> - `不同 GPU 配置的用户体验`
> - `LMS 能耗的潜在未来修复` 


- ****LMS 在闲置状态下的能耗****：一位成员报告了 LMS 在闲置时异常的能耗，指出每个 GPU 的功耗应为 **一半**，约为 10W。他们强调，在 GPU 使用率方面，[Blender](https://discord.com/channels/1110598183144399058/1253332613540876401) 是比网页浏览更好的对比对象。
- ****Windows vs Linux：不同的功耗表现****：一位成员在 **Linux** 上运行 LMS 时没有遇到明显的硬件问题，但承认在 **Windows** 上可能存在差异。他们观察到，加载模型运行 LMS 与运行 Firefox 或 Chrome 相比仅有 33 瓦的差异，而在未加载模型时则有额外的 22 瓦差异。


  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1257805109182595216)** (14 条消息🔥): 

> - `Gemma 2 加载问题`
> - `ROCm GPU 兼容性与性能`
> - `Linux ROCm 扩展包测试` 


- ****Gemma 2 在 LM Studio 0.2.27 中加载失败****：一位用户在加载 **Gemma 2** 时遇到错误，提示信息为：'unknown model architecture: 'gemma2''，尽管已清除缓存并运行了必要脚本。建议的修复方法包括 **重新下载或全新安装**。
- ****AMD GPU 上的 ROCm 支持成功****：讨论显示在 **AMD Radeon RX 6900 XT** 和 **7800 XT** GPU 上成功使用了 ROCm，并有证言称可以在没有 RAM 问题的情况下运行 **Gemma 2 8k token**。另一位用户确认 ROCm 构建版本可以很好地支持这些模型。
- ****征集 Linux ROCm 扩展测试****：一位社区成员请求协助测试 **适用于 0.2.27 版本的最新 Linux ROCm 扩展包**。提供的说明包括通过脚本安装，并确认设置中出现了 **ROCm llama.cpp**。


  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1257837864683700244)** (1 条消息): 

> - `Hugging Face 上的 Gemma 2 模型更新`
> - `Gemma 2 模型的兼容性更新` 


- **Gemma 2 模型已更新以提升兼容性**：[lmStudio 社区 Hugging Face](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) 上的 **Gemma 2 模型** 已根据最新更改进行了更新，可以安全地重新下载并在 **0.2.27** 版本中使用。
- **Gemma 2 模型已准备就绪**：更新后的 **Gemma 2 模型** 现在与最新的 **0.2.27** 版本兼容，可以从 [Hugging Face](https://huggingface.co/lmstudio-community/gemma-2-27b-it-GGUF) 下载。


  

---

### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1257773145008640070)** (70 messages🔥🔥): 

> - `gpuOffload 数值讨论`
> - `使用 TypeScript 和 Discord.js 的 bot 配置问题` 


- ****[修复 Discord Token 问题](https://github.com/mrdjohnson/lmstudio-discord-bot)**：新 bot 因无效 Token 失败**：**Aquora** 最初在配置 Discord bot 时遇到了 [无效 Token 错误](https://github.com/mrdjohnson/lmstudio-discord-bot)。该问题最终追溯到未允许的 **MessageContent** intents，通过在 Discord Developer Portal 中启用它们得以解决。
- ****Bot 循环与“思考”状态修复**：调整 Temperature 和预测 Token 解决 Bot 幻觉**：**Aquora** 遇到了 Bot 的问题。
   - **DJ** 建议在未来的文章中添加消息历史记录和直接消息处理，以改进 Bot 功能。**Aquora** 渴望为进一步的改进做出贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/0xAquora/Lmstudio-discordjs-chatbot">GitHub - 0xAquora/Lmstudio-discordjs-chatbot: 这是一个参考此案例进行的个人测试：(https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6)</a>：这是一个参考此案例进行的个人测试：(https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6) - 0xAquora/Lmstudio-discordjs-chatbot</li><li><a href="https://github.com/mrdjohnson/lmstudio-discord-bot/tree/main">GitHub - mrdjohnson/lmstudio-discord-bot: 一个使用 LM Studio 创建响应式 Discord bot 的教程！此代码基于此处的博客文章：https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6</a>：一个使用 LM Studio 创建响应式 Discord bot 的教程！此代码基于此处的博客文章：https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6 - mrdjohnson/lm...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1257796484473294908)** (1 messages): 

> - `使用 KerasNLP 为 Transformers 模型提供的新微调`
> - `按列名搜索 HF 数据集的实验性 API`
> - `包含新功能和模型的 Transformers 4.42 发布`
> - `HF Hub 上近 10 万个公开模型存储了 TensorBoard 日志`
> - `Local Gemma 发布` 


- **Transformers 4.42 发布引入了新模型和功能**：新的 [Transformers 4.42](https://x.com/osanseviero/status/1806440622007447631) 版本包括 **Gemma 2**、RT-DETR、InstructBlip、LLaVa-NeXT-Video、**tool usage 和 RAG 支持**、GGUF 微调以及 **quantized KV cache**。
- **KerasNLP 为任何 Transformers 模型的微调搭建桥梁**：使用 **KerasNLP** 实现，可以访问 [大量针对任何 **Transformers 模型** 的新微调](https://x.com/julien_c/status/1806366482269352232)。
- **AWS 在 HF 上发布 Chronos 数据集**：[AWS](https://x.com/solitarypenman/status/1806421605683232947) 在 **Hugging Face** 上发布了 Chronos 论文中使用的所有数据集，包括预训练和评估数据集。
- **Local Gemma 提供 100% 私密且安全的生成**：新的 [Local Gemma](https://x.com/reach_vb/status/1807830966515519667) 是 100% 本地、**私密且安全**的，可以随时通过 `pip install local-gemma` 运行。
- **视觉语言模型介绍发布**：[视觉语言模型介绍](https://x.com/mervenoyann/status/1805910433024380978) 已发布，展示了图像-文本到文本（image-text-to-text）模型。
   - 这包括 **image captioning**、光学字符识别（OCR）等任务。


<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://x.com/julien_c/status/1806366482269352232)">Julien Chaumond (@julien_c) 的推文</a>：Keras 🤝 HF</li><li><a href="https://x.com/vanstriendaniel/status/1807814430262202465)">Daniel van Strien (@vanstriendaniel) 的推文</a>：使用全新的实验性 API，通过列名搜索 @huggingface 数据集！此 API 允许你：- 搜索包含上下文的问答数据集 - 查找 alpaca 风格的数据集 - 定位 D...</li><li><a href="https://x.com/osanseviero/status/1806440622007447631)">Omar Sanseviero (@osanseviero) 的推文</a>：Transformers 4.42 发布了，包含许多惊人的功能🥳 🔥新模型：Gemma 2、RT-DETR（目标检测）、InstructBlip 和 LLaVa-NeXT-Video 🔧工具调用和 RAG 支持 👀GGUF 微调 🤏Qu...</li><li><a href="https://x.com/Wauplin/status/1808074557128855750)">Wauplin (@Wauplin) 的推文</a>：将近 10 万个公开模型使用 Hub 来存储 𝚝𝚎𝚗𝚜𝚘𝚛𝚋𝚘𝚊𝚛𝚍 日志！将训练日志与 Checkpoints 一起存储，让你可以在 Metrics 选项卡中一站式跟踪所有内容...</li><li><a href="https://x.com/reach_vb/status/1807830966515519667)">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：介绍 Local Gemma！💎 100% 本地、私密且安全——随时随地运行！支持 CUDA、MPS、CPU——带有预设以尽可能快地运行——基于 Transformers 构建，我们确保 1:1 的 ge...</li><li><a href="https://x.com/solitarypenman/status/1806421605683232947)">Abdul Fatir (@solitarypenman) 的推文</a>：🚀🚀🚀 我们刚刚在 Hugging Face 上发布了 Chronos 论文中使用的所有数据集。这包括预训练和评估（域内和 Zero-shot）数据集。我们还开源了一个脚本来...</li><li><a href="https://x.com/mervenoyann/status/1807790959884665029)">merve (@mervenoyann) 的推文</a>：Real-time DEtection Transformer (RT-DETR) 已登陆 @huggingface Transformers 🤩，采用 Apache 2.0 许可证 😍 DETR 在实时目标检测上能击败 YOLO 吗？继续阅读 👀</li><li><a href="https://x.com/xenovacom/status/1805990110065803492)!">Xenova (@xenovacom) 的推文</a>：得益于 Transformers.js，微软推出的新视觉基础模型 Florence-2 现在可以 100% 在浏览器中通过 WebGPU 本地运行！🤗🤯 它支持图像字幕、光学字符识别...</li><li><a href="https://x.com/mervenoyann/status/1805910433024380978)">merve (@mervenoyann) 的推文</a>：刚刚发布：视觉语言模型（又称图像-文本到文本）简介</li><li><a href="https://x.com/ben_burtenshaw/status/1806291858835837333)">Ben Burtenshaw (@ben_burtenshaw) 的推文</a>：🚀 很高兴推出我们的新系列，由 @argilla_io 制作的 Data Explorer！🎥 我们深入探讨数据集及其对模型性能的影响。我们的第一集探索了由 @hannahrosekir...</li><li><a href="https://x.com/TheZachMueller/status/1807394438689214930)">Zach Mueller (@TheZachMueller) 的推文</a>：如何让 @PyTorch Dataloaders 在分布式训练期间高效工作？这是我使用 @huggingface Accelerate 的 Dataloaders 制作的一个视频教程，展示了我们是如何做到的 https://www.yo...</li><li><a href="https://x.com/mervenoyann/status/1806267855559623115)">merve (@mervenoyann) 的推文</a>：使用 @elastic search 和 @huggingface 的新 Gemma RAG 方案 🧑🏻‍🍳📖 在下方查找 ⇓
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1257776252471742606)** (236 条消息🔥🔥): 

> - `加入 Hugging Face Discord 社区`
> - `Adept 战略转型及联合创始人加入 Amazon`
> - `用于文本和图像处理的 AI 模型`
> - `Hugging Face 模型的性能和准确性`
> - `关于 ML 认证的建议`

- ****邀请加入 HuggingFace Discord**: 分享了 <a [Discord Community](https://huggingface.co/discord-community) 链接，并说明过去已验证的成员很快将收到邀请。**: 有用户询问如何加入 [HuggingFace Discord Community](https://huggingface.co/discord-community)；邀请将很快发送给所有过去已验证的成员，强调这是一个用于实时项目的共享协作空间。
   - *另一位用户表示有兴趣担任社区管理员，因为他们一直积极参与举报诈骗行为。*
- ****具有前景的文本处理 AI 模型**: 用户讨论了各种 AI 模型，如具有 **132k context windows** 的 Qwen。准确的 [Qwen2 等模型](https://huggingface.co/Qwen/Qwen2-7B-Instruct) 因其出色的 context lengths 和在详细文本处理中的实用性而受到关注。**: HuggingFace 用户社区讨论了能够处理超长 context windows 的 AI 模型，推荐使用具有 132k context window 的 [Qwen2](https://huggingface.co/Qwen/Qwen2-7B-Instruct) 来执行详细的文本处理任务。
   - *建议在不同语境下考虑性能和质量，对长上下文与更简洁的模型输出进行实验。*
- ****比较 Open Source 和 Proprietary AI 模型**: 用户将 Meta-Llama 等 HuggingFace 模型与 OpenAI 的 GPT 进行比较**。[Meta-Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) 模型因其在 benchmarks 中的卓越表现而被推荐。**: 用户对 HuggingFace 模型与 OpenAI 的 GPT 等 Proprietary 模型进行了比较，[Meta-Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) 模型因其 benchmark 优势而被推荐。
   - *用户注意到各种模型在速度和可靠性之间的平衡，表明某些用例出于效率考虑仍倾向于使用 Proprietary 工具。*
- ****ML 学习资源与认证**: 针对证明 ML 熟练程度的免费且高效的在线课程提出了建议**。Harvard 和 Coursera 的课程因其丰富的内容和证书公信力而受到推荐。**: 几位用户分享了他们在 [Harvard 免费课程](https://harvard.edu/) 和 Coursera 上的经验，指出它们在质量和证书公信力之间取得了平衡，对希望证明 ML 熟练程度的人很有帮助。
   - *一位用户询问是否可以跳过这些课程中重复的基础知识，强调了对渐进式学习模式的偏好。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://swtokyo.com/">Startup Weekend Tokyo</a>: 未找到描述</li><li><a href="https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/semantic-chunker/">Semantic Chunking | 🦜️🔗 LangChain</a>: 基于语义相似度拆分文本。</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: 让每个人都能使用社区最优秀的 AI 聊天模型。</li><li><a href="https://www.youtube.com/live/hm2IJSKcYvo">Moshi Keynote - Kyutai</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/gokaygokay/Florence-2">Florence 2 - a Hugging Face Space by gokaygokay</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt">Transformers, what can they do? - Hugging Face NLP Course</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/tree/main">meta-llama/Meta-Llama-3-70B-Instruct at main</a>: 未找到描述</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4">THUDM/cogvlm2-llama3-chat-19B-int4 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/tree/main">mistralai/Mixtral-8x7B-Instruct-v0.1 at main</a>: 未找到描述</li><li><a href="https://moshi.chat/?queue_id=talktomoshi">moshi.chat</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/LanguageBind/Video-LLaVA">Video LLaVA - a Hugging Face Space by LanguageBind</a>: 未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct">Qwen/Qwen2-7B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/">GraphRAG: New tool for complex data discovery now on GitHub</a>: GraphRAG 是一种基于图的检索增强生成 (RAG) 方法，可显著改善针对私有或先前未见数据集的问答效果，现已在 GitHub 上发布。了解更多...</li><li><a href="https://x.com/reach_vb/status/1807830966515519667https://x.com/reach_vb/status/1806731975618626004">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: 隆重推出 Local Gemma！💎 100% 本地、私密且安全 - 随时随地运行！支持 CUDA, mps, cpu - 带有预设以尽可能快地运行 - 站在 Transformers 的肩膀上，我们确保 1:1 ge...</li><li><a href="https://huggingface.co/chat/settings/meta-llama/Meta-Llama-3-70B-Instruct/">HuggingChat</a>: 让每个人都能使用社区最优秀的 AI 聊天模型。</li><li><a href="https://huggingface.co/discord-community">discord-community (Hugging Face Discord Community)</a>: 未找到描述</li><li><a href="https://tenor.com/view/ineedit-needit-spongebob-squarepants-need-it-gif-4883495">Need It GIF - Ineedit Needit Spongebob Squarepants - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/llava-hf/bakLlava-v1-hf">llava-hf/bakLlava-v1-hf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-beta">HuggingFaceH4/zephyr-7b-beta · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/reach_vb/status/1807830966515519667">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: 隆重推出 Local Gemma！💎 100% 本地、私密且安全 - 随时随地运行！支持 CUDA, mps, cpu - 带有预设以尽可能快地运行 - 站在 Transformers 的肩膀上，我们确保 1:1 ge...</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs</a>: 一个高吞吐量且内存效率高的 LLM 推理和提供引擎 - vllm-project/vllm</li><li><a href="https://github.com/VedankPurohit/LiveRecall">GitHub - VedankPurohit/LiveRecall: Welcome to **LiveRecall**, the open-source alternative to Microsoft&#39;s Recall. LiveRecall captures snapshots of your screen and allows you to recall them using natural language queries, leveraging semantic search technology. For added security, all images are encrypted.</a>: 欢迎使用 **LiveRecall**，这是 Microsoft Recall 的开源替代方案。LiveRecall 捕获屏幕快照，并允许你利用语义搜索技术通过自然语言查询来召回它们。为了增加安全性，所有图像均已加密。</li><li><a href="https://github.com/SillyTavern/SillyTavern">GitHub - SillyTavern/SillyTavern: LLM Frontend for Power Users.</a>: 面向高级用户的 LLM 前端。通过在 GitHub 上创建账户，为 SillyTavern/SillyTavern 的开发做出贡献。</li><li><a href="https://huggingface.co/lakkeo/stable-cypher-instruct-3b">la

kkeo/stable-cypher-instruct-3b · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1257779448758141069)** (7 messages): 

> - `关于 ViT 和 Unets 等 CNN 主题的高级资源`
> - `请求更多关于 torch.distributed 的教程`
> - `用于角色扮演和角色沉浸式提示词生成的 Gradio 应用`
> - `TIL（今天我学到了）关于 Python 中集合（Sets）和字典（Dicts）的 '|' 和 '&' 运算符`
> - `关于法语贝叶斯定理的问题` 


- ****角色扮演提示词工具成为焦点****：一位成员分享了他们为角色扮演和角色沉浸式提示词生成而创建的 Gradio 应用，寻求反馈和改进建议。他们展示了一些生成结果，并提供了[工具链接](https://huggingface.co/spaces/xtreme86/System_roleplay_generator)。
- ****Python 集合运算符揭秘****：一位成员分享了一个新的 [GitHub 资源](https://github.com/noahlt/til/blob/main/python/2024-07-02-dict-and-set-operators.md)，关于 Python 中集合（Sets）和字典（Dicts）的 `|` 和 `&` 运算符，并表示这虽然与 ML 没有直接关系，但仍然很有趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/xtreme86/System_roleplay_generator">System Roleplay Generator - a Hugging Face Space by xtreme86</a>: 未找到描述</li><li><a href="https://github.com/noahlt/til/blob/main/python/2024-07-02-dict-and-set-operators.md">til/python/2024-07-02-dict-and-set-operators.md at main · noahlt/til</a>: 今天我学到了 - 受 @simonw 和 @pdubroy 启发 - noahlt/til
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1257801237739606067)** (8 messages🔥): 

> - `注意力机制`
> - `Transformer 架构`
> - `Shell 中的兼容性`
> - `序列转导模型`
> - `演示视频反应` 


- ****Transformer 中的注意力机制视觉化解析****：一位用户分享了一个名为“Attention in transformers, visually explained | Chapter 6, Deep Learning”的 [YouTube 视频](https://www.youtube.com/watch?v=eMlx5fFNoYc)，详细介绍了 Transformer 和 LLM 内部的关键机制。该视频被赞誉为迄今为止看到的“关于 Transformer 架构的最佳视频”。
- ****Starship.rs 兼容性优先****：分享了 [Starship.rs](https://starship.rs/) 的链接，强调了该 Shell 与大多数常见操作系统的兼容性。该工具承诺在各种环境中都具有可用性。
- ****Transformer 论文****：一位用户重点介绍了来自 arXiv 的 [Transformer 架构论文](https://arxiv.org/abs/1706.03762)，该论文提出了一种完全基于注意力机制的网络。它在 WMT 2014 任务中，英德翻译取得了 28.4 的 BLEU 分数，英法翻译取得了 41.8 的分数。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a>: 主流的序列转导模型基于编码器-解码器配置中复杂的循环或卷积神经网络。性能最好的模型还连接了编码器和解...</li><li><a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">Attention in transformers, visually explained | Chapter 6, Deep Learning</a>: 揭秘注意力机制，这是 Transformer 和 LLM 内部的关键机制。这些课程不是由赞助广告阅读资助，而是直接由观众资助：https://3...</li><li><a href="https://starship.rs/">Starship: Cross-Shell Prompt</a>: Starship 是适用于任何 Shell 的极简、极速且高度可定制的提示符！在保持时尚和极简的同时显示你需要的信息。支持 Bash, Fish, ZS... 的快速安装。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1257779227596558518)** (7 条消息): 

> - `OpenAI's CriticGPT Release` (OpenAI 发布 CriticGPT)
> - `Stable Release of Embodied Agents Toolkit` (Embodied Agents Toolkit 稳定版发布)
> - `Open Source OCR for Kazakh Language` (哈萨克语开源 OCR)
> - `Blog on Reinforcement Learning Specialization` (Reinforcement Learning Specialization 博客)
> - `Zero-Shot Generating Spatial Sound from Images` (从图像零样本生成空间音频) 


- **OpenAI 发布 CriticGPT**：一位用户分享了一个 [YouTube 视频](https://youtu.be/4PgcaIfwLjo)，介绍了 **CriticGPT**，这是 **OpenAI** 推出的一款新 AI 模型，用于识别 **GPT-4** 生成的代码中的错误。该发布被誉为提高代码准确性的重要一步。
- **用于机器人集成的 Embodied Agents Toolkit**：[Embodied Agents toolkit](https://github.com/MbodiAI/mbodied-agents) 最近发布，旨在以极少的代码将最先进的多模态 Transformer 集成到机器人技术中。该工具包包括 **Gradio 界面支持** 和 **HuggingFace 数据集集成**。
- **哈萨克语 OCR 解决方案发布**：一个用于哈萨克语 OCR 的 [开源解决方案](https://huggingface.co/spaces/BMukhtar/BookRecognitionKz) 已发布。该方案旨在填补代表性不足语言在 OCR 技术方面的重大空白。
- **Reinforcement Learning Specialization 博客**：一位用户分享了他们在 Coursera 上的 **Reinforcement Learning Specialization** [博客系列](https://sezan92.github.io/2024/07/03/RL-course-blog.html)。详细笔记涵盖了关于 RL 的多个周次和课程。
   - *查看详情并提供反馈*，正如作者所建议的。
- **越南语视觉语言模型发布**：**Vi-VLM** 团队发布了一个基于 LLaVA 和 Vistral LLM 的 [越南语视觉语言模型](https://huggingface.co/Vi-VLM/Vistral-V-7B)。该模型针对图像描述任务进行了优化，利用专有数据集进行预训练和监督微调（SFT）。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/BMukhtar/BookRecognitionKz">BookRecogntionKZ - BMukhtar 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://sezan92.github.io/2024/07/03/RL-course-blog.html">Reinforcement Learning Specialization</a>: Coursera Reinforcement Learning Specialization 笔记</li><li><a href="https://youtu.be/4PgcaIfwLjo">OpenAI releases CriticGPT to correct GPT-4&#39;s mistakes | Read the paper with me</a>: OpenAI 发布了 CriticGPT，这是一款基于 GPT-4 的新 AI 模型，旨在识别 ChatGPT 生成的代码中的错误，标志着向...迈出的重要一步。</li><li><a href="https://huggingface.co/spaces/rishitdagli/see-2-sound">SEE-2-SOUND - rishitdagli 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/waefrebeorn/KAN-Stem">GitHub - waefrebeorn/KAN-Stem: attempt at using gpt4o to create a KAN stem training script</a>: 尝试使用 gpt4o 创建 KAN stem 训练脚本 - waefrebeorn/KAN-Stem</li><li><a href="https://github.com/mbodiai/embodied-agents">GitHub - mbodiai/embodied-agents: Seamlessly integrate state-of-the-art transformer models into robotics stacks</a>: 将最先进的 Transformer 模型无缝集成到机器人技术栈中 - mbodiai/embodied-agents</li><li><a href="https://huggingface.co/Vi-VLM/Vistral-V-7B">Vi-VLM/Vistral-V-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/hllj/Vistral-V">GitHub - hllj/Vistral-V: Vistral-V: Visual Instruction Tuning for Vistral - Vietnamese Large Vision-Language Model.</a>: Vistral-V: 针对 Vistral 的视觉指令微调 - 越南语大型视觉语言模型。 - hllj/Vistral-V</li><li><a href="https://c57e0e7e63316ef057.gradio.live/">LLaVA</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1257906905934856252)** (4 messages): 

> - `Highway Net vs ResNet performance`
> - `Gradient vanishing problem in LSTM`
> - `Multi-branch structure inspiration from LSTM`
> - `Pre-trained models and fine-tuning techniques`
> - `topicSummaries` 


- **Highway Net 性能不如 ResNet：是时候重新思考了**：一位用户质疑为什么 **Highway Net 的表现优于 ResNet**，并建议现在可能是重新审视设计选择的时候了。**来自 LSTM 的门控方案 (gating scheme)** 真的解决了梯度消失问题吗？
- **受 LSTM 启发的多分支结构**：一位用户承认他们关于 **多分支结构 (multi-branch structure)** 的想法源自 **LSTM**。这引发了关于 LSTM 门控方案中梯度消失问题的讨论。
- **使用高性价比方法微调预训练模型**：一位用户分享了一篇[论文](https://arxiv.org/abs/2405.14739)，讨论了在不更新所有参数的情况下**微调预训练模型**的技术，重点关注低秩调整 (low-rank adjustments) 等资源高效型方法。
   - *这些方法往往忽略了像 4D 这样更高维度的参数空间，从而导致结构完整性问题。*



**提及的链接**：<a href="https://arxiv.org/abs/2405.14739">FLoRA: Low-Rank Core Space for N-dimension</a>：在人工智能领域，针对各种下游任务调整预训练基础模型 (foundation models) 已变得非常普遍。由于任务数量庞大且成本高昂，调整所有参数变得不可行...

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1257779351236378744)** (6 messages): 

> - `ADVANCED_CNN_RESOURCES`
> - `Neighborhood_Attention_Transformer_usage`
> - `Developer_Job_Openings`
> - `MaskFormer_training_issues`
> - `Lightweight_AI_for_programming` 


- **需要关于高级 CNN 技术的书籍**：一位用户正在寻求关于 **高级 CNN 主题**（如 ViTs 和 UNets，包括**视频处理**）的书籍或资源推荐。
- **Neighborhood Attention Transformer 维护问题**：一位用户分享了 **Neighborhood Attention Transformer** 文档的链接，强调该模型目前**仅处于维护模式**，并建议如果新版本出现问题，请重新安装 4.40.2 版本。他们引用了题为 [Neighborhood Attention Transformer](https://arxiv.org/abs/2204.07143) 的论文以提供更多背景信息。
- **开发者求职**：一位用户询问目前是否有**公司或项目**正在寻找技术熟练的开发者。
- **训练 MaskFormer 模型时遇到的问题**：一位用户在训练用于实例分割的 **MaskFormer 模型**时遇到困难，在掩码准确率和训练时间方面挣扎。他们正在使用 Hugging Face 的 **Trainer 类**，并请求人工协助。
- **寻找用于编程的轻量级 AI**：一位用户征求适用于编程且足够轻量级的 **AI 模型**推荐。



**提及的链接**：<a href="https://huggingface.co/docs/transformers/v4.42.0/model_doc/nat">Neighborhood Attention Transformer</a>：未找到描述

  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1257936960442335342)** (9 封消息🔥): 

> - `自定义 Pipeline 创建`
> - `具有高最大输入 Token 长度的文本摘要模型`
> - `开源模型与 ChatGPT 的性能对比`
> - `下载和使用 Meta LLaMA 的挑战`
> - `Mistral 模型的推理冻结问题` 


- **Andy Singal 编写的自定义 Pipeline 创建指南**：[Andy Singal 分享了一份指南](https://github.com/andysingal/llm-course/blob/main/transformers/custom-pipeline.md)，介绍如何使用 Transformers 创建自定义 Pipeline。该指南是 GitHub 上 **LLM course** 的一部分。
   - 对于那些希望在 NLP 任务中寻求定制化解决方案的人来说，探索自定义 Pipeline 会非常有益。
- **对长文档文本摘要模型的需求**：一位用户请求推荐能够处理超长文档的文本摘要模型，特别是那些具有高最大输入 Token 长度的模型。这一请求凸显了有效总结长文本的挑战。
- **开源模型 vs. ChatGPT**：关于 Hugging Face 模型是否能匹配 ChatGPT 3.5 或 4 的性能/准确性的讨论。一位成员提到，有开源模型声称性能优于 ChatGPT 3.5。
   - *模型往往会在基准测试（benchmarks）上过拟合*，正如讨论中所强调的那样。
- **Meta LLaMA 下载挑战**：一位用户在下载 Meta LLaMA 时遇到困难，并考虑为该模型构建 API 调用。他们担心由于 20 分钟下载过程中的临时文件存储限制，可能会导致失败。
- **Mistral 中的推理冻结问题**：在使用 Mistral 模型进行实验时，运行 3000 次推理在第 1800 次迭代时冻结，花了一天时间才继续。推理卡顿可能是由于运行之间的某些缓存或其他资源管理问题导致的。



**提及的链接**：<a href="https://github.com/andysingal/llm-course/blob/main/transformers/custom-pipeline.md">llm-course/transformers/custom-pipeline.md at main · andysingal/llm-course</a>：通过在 GitHub 上创建账号来为 andysingal/llm-course 的开发做出贡献。

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1258165680478621706)** (1 封消息): 

> - `关于使用 diffusers 运行 RealVisXL V4.0 Lightning 模型的讨论。`
> - `A1111 与 diffusers 之间的质量对比。`
> - `在 Boosty 上提供支持。`
> - `推荐的负向提示词（negative prompt）和生成参数。`
> - `训练阶段模型性能的问题。` 


- ****在 Boosty 上支持 RealVisXL V4.0****：一位成员分享说，你可以在 [Boosty](https://boosty.to/sg_161222) 上支持 **RealVisXL V4.0 Lightning** 的开发。
- ****RealVisXL V4.0 模型训练****：旨在实现照片级真实感的 **RealVisXL V4.0 Lightning** 模型仍处于训练阶段，可能**包含伪影（artifacts）**，且在某些情况下表现不佳。
- ****使用 diffusers 运行 RealVisXL V4.0****：一位成员报告称，尽管使用了相同的参数（提示词、步数、调度器等），但使用 diffusers 运行 **RealVisXL V4.0 Lightning** 的质量远低于使用 **A1111**。



**提及的链接**：<a href="https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning">SG161222/RealVisXL_V4.0_Lightning · Hugging Face</a>：未找到描述内容。

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1257783941876940862)** (93 messages🔥🔥): 

> - `GPT-4 参数讨论`
> - `Nvidia 在 GPT-4 开发中的参与及泄密`
> - `Mixture of Experts (MoE) 模型`
> - `InstructGPT 效率`
> - `Discord 服务器抓取与 ToS 违规` 


- ****GPT-4 参数：它到底有多大？****：围绕 GPT-4 的参数量展开了讨论，根据包括 [Nvidia](https://www.nvidia.com) 在内的各种来源，猜测数字约为 **1.7 万亿** 到 **1.8 万亿**。有趣的是，这个数字相比 GPT-3 的 **1750 亿** 参数有了巨大的飞跃，让成员们对 **MoE** (Mixture of Experts) 如何助力这种规模扩张感到好奇。
- ****InstructGPT 的现实世界提升****：**InstructGPT** 的进步受到关注，特别是在实际应用中，它提供了 **10 倍到 100 倍的效率提升**。社区强调了 **RLHF (Reinforcement Learning from Human Feedback)** 作为这一改进背后关键驱动力的重大影响。
- ****知情者 Nvidia：GPT-4 的秘密****：鉴于 Nvidia 的硬件支持合同，他们对 **GPT-4** 模型大小的熟悉程度引发了辩论。尽管受 NDA 约束，许多人认为 Nvidia 凭借其在硬件供应中的核心地位，对 OpenAI 的模型有着深入的了解。
- ****抓取 Discord：高风险行为****：尝试抓取 Discord 服务器数据，**即使是个人服务器**，也违反了 [Discord 的 ToS](https://discord.com/terms)，并可能导致封禁。一些工具如 **DiscordChatExporter** 虽然可以规避速率限制，但正如几位成员所强调的，这依然存在巨大风险。
- ****MoE 模型效率见解****：对 **MoE 模型** 的技术深入探讨揭示了其局限性和效率提升。虽然 MoE 可以通过激活选择性权重显著降低计算负载，但在推理过程中仍面临内存带宽和 VRAM 的挑战。



**提到的链接**：<a href="https://buttondown.email/ainews/archive/">AI News</a>：我们总结顶级的 AI Discord + AI Reddit + AI X/Twitter，每天为您发送汇总！查看存档以获取示例。“我每天花费的最高杠杆的 45 分钟” - Soumith “最好的 AI 新闻...”

  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1257862843919761438)** (76 条消息🔥🔥): 

> - `UL2 与传统训练目标`
> - `Starcoder2 与 UL2 性能`
> - `PrefixLM 及其训练影响`
> - `Scaling laws 与学习率调度`
> - `FIM 与 UL2 的对比` 


- **行业对采用 UL2 训练目标反应迟缓**：一位成员对行业在采用 **UL2 训练目标** 方面的缓慢进展表示惊讶，尽管它在解决短期规划和“逆转诅咒”（reversal curse）等问题上具有理论和实证优势。尽管来自 [Starcoder2](https://twitter.com/vaibhav_adlakha/status/1777854167672820000) 和 Mosaic 的测试显示其表现不如传统方法，但该成员对未来的改进仍持乐观态度。
- **[Scaling Laws 差异解决](https://arxiv.org/abs/2406.19146)**：研究人员通过识别最后一层计算成本和优化器调优等因素，解决了 **Kaplan** 和 **Hoffmann scaling laws** 之间的差异。他们的发现推翻了为了使 Chinchilla scaling law 有效而必须进行精细学习率衰减（learning rate decay）的必要性。
- **PrefixLM 与训练效率担忧**：成员们讨论了 **PrefixLM** 在训练中的效率，指出其训练速度较慢，且在当前的 attention 算法下可能效率低下。一位成员指出，模型对双向上下文（bidirectional contexts）与因果上下文（causal ones）的适应方式可能不同，从而影响性能。
- **FIM 与 UL2 目标对比**：成员们讨论了 **Fill-in-the-Middle (FIM)** 及其与 UL2/masked language 目标的比较。他们指出 FIM 可能更高效，因为它同时也预测前段和尾段，可能比 UL2 方法提供更好的结果。
- **Attention Mask 适应性**：关于模型对不同 attention masks 适应性的讨论展开，一些人指出了训练效率方面的挑战以及特定 token 的重要性。分享了一篇关于 [从 causal 到 bidirectional attention 适应性](https://twitter.com/vaibhav_adlakha/status/1777854167672820000) 的相关论文来阐述这些动态。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.19146">Resolving Discrepancies in Compute-Optimal Scaling of Language Models</a>：Kaplan 等人和 Hoffmann 等人针对作为计算预算函数的最佳模型大小开发了具有影响力的 scaling laws，但这些定律给出的预测大相径庭。我们解释了...</li><li><a href="https://arxiv.org/abs/2406.19370">Emergence of Hidden Capabilities: Exploring Learning Dynamics in Concept Space</a>：现代生成模型展示了令人印象深刻的能力，这可能源于识别和操作其训练数据背后的抽象概念的能力。然而，基本问题...</li><li><a href="https://boyuan.space/diffusion-forcing/">
      Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
    </a>：未找到描述</li><li><a href="https://x.com/tomerporian/status/1808090819808629216">Tomer Porian (@tomerporian) 的推文</a>：🧵1/8 我们解决了 Kaplan（指数 0.88，图 14 左）等人与 Hoffmann 等人（“Chinchilla”，指数 0.5）计算最优 scaling laws 之间的差异。论文：https://arxiv.org/a...</li><li><a href="https://github.com/YangLing0818/consistency_flow_matching">GitHub - YangLing0818/consistency_flow_matching: Official Implementation for &quot;Consistency Flow Matching: Defining Straight Flows with Velocity Consistency&quot;</a>：&quot;Consistency Flow Matching: Defining Straight Flows with Velocity Consistency&quot; 的官方实现 - YangLing0818/consistency_flow_matching
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1257882203543568415)** (22 条消息🔥): 

> - `Kaplan et al. 与 Hoffmann et al. 在计算最优 Scaling Laws 上的差异`
> - `Kaplan et al. 的最后一层计算成本、Warmup 时长以及与规模相关的优化器调节`
> - `Scaling Law 计算中的 Attention FLOPs 和 6ND 近似`
> - `PyTorch FLOPs 计数器工具及 FLOPs 计算方法论`
> - `Chinchilla 论文的 Scaling Law 及其外推问题` 


- ****解决 Kaplan 与 Hoffmann 之间的 Scaling Law 差异**: [研究人员解释了](https://arxiv.org/abs/2406.19146) Kaplan et al. 与 Hoffmann et al. 的 Scaling Laws 之间的差异**，通过识别诸如最后一层计算成本、Warmup 时长以及与规模相关的优化器调节等问题。：研究人员修正了这些因素，并得到了与 Hoffmann et al. 的 Scaling Law（也称为 **Chinchilla Scaling Law**）高度一致的结果。他们发现 Hoffmann et al. 提出的 **Learning Rate Decay** 假设并非必不可少，并推导出了 **最优学习率和 Batch Size** 的 Scaling Laws。
- ****Attention FLOPs 对 Scaling Laws 的影响**: 社区成员讨论了 6ND 近似在小规模模型中的不足。**：他们建议使用来自 **Kaplan et al.** 的不同公式来纳入 Attention FLOPs，具体为 `C = 6ND + 6 * n_layers * seq_len * d_model` 而非 `6ND`。
- ****PyTorch FLOPs 计数器的工具缺陷**: 关于 PyTorch 内置 FLOPs 计数器工具的讨论。**：有人担心该工具在遇到不确定 FLOPs 的操作时不会报错，而是默认忽略。
- ****Scaling Law 拟合中的外推陷阱**: 社区强调了使用过小规模实验来拟合 Scaling Laws 的风险。**：他们强调 **Chinchilla 的数值不应被盲目崇拜（cargo-culted）**，并强调需要投入大量计算资源以确保准确的外推。
- ****基于魔方数据的 Chinchilla 研究**: 引用了一个专注于合成数据 Scaling Law 拟合的 GitHub 项目。**：[GitHub](https://github.com/kyo-takano/chinchilla) 上的一个项目展示了这些原则，使用 **魔方（Rubik's cube）生成的数据** 来验证 Scaling Laws。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.19146">Resolving Discrepancies in Compute-Optimal Scaling of Language Models</a>: Kaplan et al. 和 Hoffmann et al. 为作为计算预算函数的最优模型大小开发了有影响力的 Scaling Laws，但这些定律产生的预测大相径庭。我们解释了...</li><li><a href="https://x.com/tomerporian/status/1808090819808629216">Tomer Porian (@tomerporian) 的推文</a>: 🧵1/8 我们解决了 Kaplan（指数 0.88，图 14，左）et al. 与 Hoffmann et al. (“Chinchilla”, 指数 0.5) 之间计算最优 Scaling Laws 的差异。论文：https://arxiv.org/a...</li><li><a href="https://arxiv.org/abs/2104.03113">Scaling Scaling Laws with Board Games</a>: 机器学习中最大的实验现在需要的资源远远超出了除少数机构以外的所有机构的预算。幸运的是，最近的研究表明，这些巨大实验的结果...</li><li><a href="https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py">cookbook/calc/calc_transformer_flops.py at main · EleutherAI/cookbook</a>: 深度学习入门。处理真实模型时涉及的所有实际细节和有用工具。- EleutherAI/cookbook</li><li><a href="https://github.com/kyo-takano/chinchilla/blob/master/examples/efficientcube.ipynb">chinchilla/examples/efficientcube.ipynb at master · kyo-takano/chinchilla</a>: 一个用于 Scaling Law 研究的工具包 ⚖。通过在 GitHub 上创建账号为 kyo-takano/chinchilla 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1258144662800044155)** (2 messages): 

> - `EAP with integrated gradients`
> - `Methods for discovering and applying sparse feature circuits`
> - `Generalization improvement using SHIFT`
> - `Scalable interpretability pipeline for sparse feature circuits` 


- **周末更新**：一位社区成员提到了本周末的一个活动，但未提供具体细节。
- **EAP 与 Integrated Gradients 见解**：讨论了结合 Integrated Gradients 的 EAP，参考了论文 [Methods for discovering and applying sparse feature circuits](https://arxiv.org/abs/2403.19647)。
- **Sparse Feature Circuits 论文探讨**：该论文介绍了发现和应用 **Sparse Feature Circuits** 的方法，这些电路能够详细理解语言模型的行为。它们对下游任务很有用，并提供了一个无监督、可扩展的可解释性流水线。
- **用于分类器泛化的 SHIFT**：论文讨论了 **SHIFT** 方法，该方法通过消融人类判断为任务无关的特征来提高分类器的泛化能力。该方法利用细粒度单元来实现更好的可解释性。



**Link mentioned**: <a href="https://arxiv.org/abs/2403.19647">Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models</a>: 我们介绍了发现和应用 Sparse Feature Circuits 的方法。这些是与因果相关的、人类可解释特征的子网络，用于解释语言模型的行为。Circuits i...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1257804828411957350)** (26 messages🔥): 

> - `PR confirmation and lm-eval reference`
> - `Loglikelihood_rolling functionality and usage`
> - `Handling document length longer than model's context in perplexity evaluations`
> - `Errors in model evaluation with specific configurations`
> - `Preprocessing functions and pipeline consistency` 


- **lm-eval 引用的 PR 确认**：一名成员请求确认一个 **Pull Request**，因为他们希望在论文中引用它来运行其 Benchmark 的评估。**Stellaathena** 询问论文中的数字是否来自当前代码，该成员予以确认。
- **理解 loglikelihood_rolling**：一名成员询问 **loglikelihood_rolling** 的用途，以及它是否意味着输入模型以获取可以转化为 Loss 值的 Loglikelihood 值。**Hailey Schoelkopf** 参考 [文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/d855d0baf8576296e790d0c9477b40a710d28e67/docs/model_guide.md?plain=1#L63) 解释说，它给出了从空字符串生成文档的 Loglikelihood。
- **处理 Perplexity 评估中的长文档**：**Stellaathena** 询问如何在不报错的情况下计算长于模型 Context Window 的文档的 Perplexity。**Hailey Schoelkopf** 指出，默认的 Perplexity 任务会在 `loglikelihood_rolling` 方法中根据模型长度自动处理分块（Chunking）。
- **评估配置中的特定数据集问题**：在使用 `proof-pile` 数据集的特定配置文件时，**Stellaathena** 遇到了 **Error**，但同样的配置在 `lambada_openai` 上可以运行。**Hailey Schoelkopf** 提到这可能与 Metrics 有关，并建议了一个修复方案，指出 Metric 使用中可能存在静默失败。
- **重用预处理函数**：一名成员询问如何防止预处理函数每次都重复运行，以及预处理后的数据是否可以存储并在流水线中重用。这一关注点旨在确保评估过程的效率。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/d855d0baf8576296e790d0c9477b40a710d28e67/docs/model_guide.md?plain=1#L63>">lm-evaluation-harness/docs/model_guide.md at d855d0baf8576296e790d0c9477b40a710d28e67 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/actions/runs/9780045009/job/27000738664?pr=2010.">Added MedConceptsQA Benchmark · EleutherAI/lm-evaluation-harness@0c3a587</a>: 一个用于语言模型 Few-shot 评估的框架。 - Added MedConceptsQA Benchmark · EleutherAI/lm-evaluation-harness@0c3a587
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1257772870797492285)** (174 条消息🔥🔥): 

> - `关于尝试 Gemini 1.5 Pro 的讨论`
> - `GPT4o 的访问问题`
> - `Perplexity 移动端新功能`
> - `Pro 订阅的退款流程`
> - `对 Perplexity 实时联网能力的担忧` 


- ****Gemini 1.5 Pro 聊天机器人推荐****：成员们讨论了 **Gemini 1.5 Pro** 的性能和特性，指出了其**大上下文窗口（large context window）**和**快速的性能**。一位用户因其出色的能力特别推荐尝试。
- ****GPT4o 访问问题及替代方案****：多位用户反映难以找到**免费 ChatGPT 4o** 的选项，并建议使用替代方案，如 **Bing chat 的精确模式**以及 [claude.ai](https://claude.ai) 上的 **Claude 3.5 Sonnet**，后者尽管有一些使用限制，但因其免费使用而受到赞誉。
- ****Perplexity 移动端新功能****：一位用户询问了 Perplexity 移动端应用中 **Wolfram Alpha 和代码生成功能**的可用性。另一位用户确认这些功能已在 iOS 上可用。
- ****Pro 订阅退款流程****：一位用户询问了 **Perplexity Pro 订阅的退款流程**。另一位成员提供了详细的退款政策，为**欧盟、英国、土耳其**以及**所有其他客户**指定了指南，包括时间表和条件。
- ****对 Perplexity 实时联网能力的担忧****：一位成员报告了 Perplexity 在**访问实时互联网信息**方面的异常行为，特别是在查询体育比分和天气等实时数据时。尽管有时会自我修正，但它经常**否认**具备此类能力，导致**持续的挫败感**。



**提到的链接**：<a href="https://git.new/Portkey-Phidata">gateway/cookbook/integrations/Phidata_with_ Perplexity.ipynb at main · Portkey-AI/gateway</a>：一个极速的 AI Gateway。通过 1 个快速且友好的 API 路由到 200 多个 LLM。- Portkey-AI/gateway

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1257779644514566155)** (9 条消息🔥): 

> - `精益画布指南`
> - `Perplexity AI 创业故事`
> - `构建黑盒`
> - `OpenSSH 查询`
> - `Echo Park 的清醒生活` 


- ****精益画布（Lean Canvas）指南****：在 [Perplexity AI](https://www.perplexity.ai/search/when-should-you-do-a-lean-canv-z_lDH7CJStuuX.MpyRGNMA) 上通过精简指南探索 **Lean Canvas** 解决方案。包含宝贵的见解和分步说明。
- ****Perplexity AI 创始故事****：通过这篇详尽的[叙述](https://www.perplexity.ai/search/the-story-behind-starting-perp-DnZ.yJgfSM28Ra9_h2uKWg)深入了解 **Perplexity AI** 的起源。文章分享了创作过程中的灵感和面临的挑战。
- ****构建黑盒****：通过详细的 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/q-how-can-we-build-a-blackbox-Iua1cgLZTfOSrmg8lIxGiw#3)了解如何在 AI 领域构建黑盒系统。讨论了方法论和潜在应用。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1257781083303579648)** (7 条消息): 

> - `在 Perplexity API 中使用 Sonnet 3.5`
> - `Sonnet 在 Perplexity API 中的可用性`
> - `Perplexity API 中的可用模型列表`
> - `通过 Perplexity API 使用搜索引擎`
> - `llama-3-sonar-large-32k-online 模型的问题` 


- ****Sonnet 3.5 无法通过 Perplexity API 使用****：**Sonnet** 未通过 **Perplexity API** 提供。可用模型可以在[文档](https://docs.perplexity.ai/docs/model-cards)中找到。
- ****对使用 API 调用搜索引擎的兴趣****：多位成员表示有兴趣通过 API 使用 **Perplexity 的搜索引擎**。有人提到通过发送邮件至 **api@perplexity.ai** 以获取 Beta 测试访问权限。
- ****llama-3-sonar-large-32k-online 模型给出错误答案****：一位成员指出 **llama-3-sonar-large-32k-online** 对一个简单的查询给出了错误答案。另一位成员建议使用 `after:` 参数来优化搜索。



**提到的链接**：<a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：未找到描述

  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1257842866391089172)** (19 条消息🔥): 

> - `在旧金山 AGI House 举办的仅限 CUDA 的黑客松`
> - `Meta Hacker Cup 2024 日程安排`
> - `关于 NVIDIA GPU (3090, 4090) 价格和购买的讨论` 


- ****旧金山 CUDA 黑客松开启****：**Ash Vardanian** 将于 **7 月 13 日**在 AGI House 举办一场 [仅限 CUDA 的黑客松](https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf)，**Chris Lattner** 将作为重磅演讲嘉宾出席。所有参与者都将获得由 Nebius.ai 赞助的 **H100 使用权限**。
- ****Meta Hacker Cup 2024 赛季回归****：[Meta Hacker Cup](https://codeforces.com/blog/entry/131165) 将于 **9 月 20 日**以练习赛拉开帷幕，随后进行一系列轮次，并于 **12 月 7 日**举行总决赛。组委会成员 **Mark Saroufim** 鼓励大家积极参与，特别是对代码生成感兴趣的开发者。
- ****关于 NVIDIA 3090 价格的辩论****：成员们讨论了是否购买 **3090**，并指出目前价格在 **1,000 美元**左右。**Mark Saroufim** 提到他在 Meta 裁员恐慌期间以 1,200 美元的价格买到了一块 **4090**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf">报名 Hardcore CUDA 黑客松 | Partiful</a>：*所有演讲和项目必须使用 CUDA 编写*。每位硬核黑客都将获得一整天的 H100 使用权。全部由 Nebius.ai 赞助并提供！让我们打破基准。演讲嘉宾：- Chris Lattner (...</li><li><a href="https://codeforces.com/blog/entry/131165">Meta Hacker Cup 2024 日程安排 — 介绍 Meta Hacker Cup AI 赛道 - Codeforces</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1257852511742595175)** (14 条消息🔥): 

> - `在 Pytorch 中使用 Inductor 后端为 Nvidia 设备编译函数的步骤`
> - `Triton IR 与 MLIR 的区别`
> - `John Carmack 对 PyTorch 团队及开源贡献的正面反馈`
> - `强制 Inductor 为 GEMM 和 Conv 生成 Triton kernel 的问题` 


- **使用 Inductor 编译 Pytorch 函数的步骤**：讨论了在 **Pytorch** 中使用 **Inductor** 后端为 Nvidia 编译函数的步骤，从 **PYTorch (python 方法)** 到 **PTX**。关于 **MLIR** 是否应该作为一个单独的步骤存在困惑。([来源](https://x.com/ID_AA_Carmack/status/1807072152631333060))
- **MLIR 不是一个独立的 IR**：澄清了 **MLIR** 是一个用于构建自定义 IR 的工具包，而不是一个独立的 IR。Triton 使用 **ttir**、**ttgir**、**llir**、**ptx** 和 **cubin** 作为步骤，但 *“ttir => ttgir 的转换是最重要的”*。
- **John Carmack 赞扬 PyTorch 团队**：John Carmack 对 **@PyTorch 团队对 Bug 报告的响应印象深刻**，他表示虽然该项目有很高的学习门槛，但**环境搭建文档**让他能够自给自足。
- **Inductor Triton Kernel 问题**：强制 **Inductor** 为所有内容生成 **Triton kernel** 对 **GEMM** 有效，但对 **Conv** 无效，尽管已经有了 kernel 模板。该问题已在 [GitHub](https://github.com/pytorch/pytorch/issues/125728) 上提出，正在寻求解决方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/ID_AA_Carmack/status/1807072152631333060">John Carmack (@ID_AA_Carmack) 的推文</a>：我对 @PyTorch 团队对 Bug 报告的响应印象极其深刻。我有时觉得，既然它是完全开源的，我应该亲自去创建一个补丁，但一个项目如果...</li><li><a href="https://github.com/pytorch/pytorch/issues/125728">torch._inductor.config.max_autotune_gemm_backends = &quot;TRITON&quot; 在卷积层崩溃 · Issue #125728 · pytorch/pytorch</a>：🐛 描述 Bug 复现代码 import torch import torch._inductor.config # torch._inductor.config.trace.enabled = True torch._inductor.config.max_autotune_gemm_backends = &quot;TRITON&quot; torch._inducto...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1257794158051983531)** (8 条消息🔥): 

> - `High-performance matrix multiplication on CPU` (CPU 上的高性能矩阵乘法)
> - `3D V-Cache performance on AMD Ryzen` (AMD Ryzen 上的 3D V-Cache 性能)
> - `Difference between 3D and non-3D Ryzen chips` (3D 与非 3D Ryzen 芯片的区别)
> - `Discussion on specialization of 3D V-Cache chips` (关于 3D V-Cache 芯片专业化的讨论)
> - `Simulation benchmarks for CPUs` (CPU 的模拟基准测试)


- **CPU 上的高性能矩阵乘法**: [Mobicham 分享了一个教程](https://salykova.github.io/matmul-cpu)，关于在 CPU 上实现高性能矩阵乘法，代码可在 [matmul.c](https://github.com/salykova/matmul.c) 获取。该实现在 AMD Ryzen 7700 上进行了优化，通过使用 **3 行 OpenMP 指令** 实现了超过 **1 TFLOPS** 的性能，超越了 NumPy。
- **3D V-Cache 提升 AMD 性能**: Iron_bound 询问了 **3D V-Cache** 的性能影响，并引用了[一篇评测](https://www.anandtech.com/show/18795/the-amd-ryzen-7-7800x3d-review-a-simpler-slice-of-v-cache-for-gaming/4)，显示其拥有 96MB 的 L3 cache，并在游戏和模拟中进行了对比。
- **3D 与非 3D Ryzen 芯片的区别**: As_ai 和 iron_bound 讨论了 3D 与非 3D Ryzen 芯片的区别，指出 3D 版本拥有 **两倍的 L3 cache**，但运行频率较低，以防止损坏额外的缓存硅层。
- **3D V-Cache 芯片的专业化**: As_ai 询问 **3D V-Cache** 除了额外的缓存外是否有任何特殊之处，iron_bound 确认 **区别主要在于更多的缓存和更低的时钟频率**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://oimo.io/works/life/">Life Universe</a>: 未找到描述</li><li><a href="https://salykova.github.io/matmul-cpu">Beating NumPy’s matrix multiplication in 150 lines of C code</a>: TL;DR 教程中的代码可在 matmul.c 获取。这篇博文是我尝试在保持代码简单、可移植的同时，在 CPU 上实现高性能矩阵乘法的结果...</li><li><a href="https://www.anandtech.com/show/18795/the-amd-ryzen-7-7800x3d-review-a-simpler-slice-of-v-cache-for-gaming/4">The AMD Ryzen 7 7800X3D Review: A Simpler Slice of V-Cache For Gaming</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1257888940464148604)** (31 条消息🔥): 

> - `Loading a buffer containing int4 using torchao` (使用 torchao 加载包含 int4 的 buffer)
> - `Saving a tensor into a safetensors file` (将 tensor 保存到 safetensors 文件中)
> - `Dequantizing tensors using torchao` (使用 torchao 对 tensor 进行反量化)
> - `Handling packed int4 arrays in Python` (在 Python 中处理打包的 int4 数组)
> - `torchao's handling of unexpected keyword arguments` (torchao 对意外关键字参数的处理) 


- **在 Python 中高效处理打包的 int4 数组**: 成员们讨论了如何解析打包的 int4 数组，方法是先转换为 uint8，然后使用 `torch.frombuffer()` 进行位移和堆叠 tensor。他们强调在解析 buffer 之前理解其位布局 (bit-layout) 的重要性。
- **使用 Python 技术对 tensor 进行反量化**: 一位成员询问了如何使用量化尺度 (quantization scales) 对 tensor 进行反量化，这涉及到创建 tensor 并使用 PyTorch 执行元素级操作，如 `dequant_tensor = quant_tensor * scale`。
- **torchao 的 buffer 加载和意外关键字处理**: 讨论集中在 torchao 的 `to()` 函数及其如何解释参数，揭示了意外关键字参数被传递给 `__new__()` 时产生的问题。他们指出正确配置参数以避免 tensor 操作期间出错的重要性。



**提到的链接**: <a href="https://github.com/ethanc8/Gemini-Nano/blob/master/playground/converter.py#L166">Gemini-Nano/playground/converter.py at master · ethanc8/Gemini-Nano</a>: 通过在 GitHub 上创建账号来为 ethanc8/Gemini-Nano 的开发做出贡献。

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1257782594716700702)** (84 条消息🔥🔥): 

> - `与 PyTorch 的内存效率对比`
> - `模型权重可视化与训练问题`
> - `新的 GitHub PR 与 Bug 修复`
> - `muP 的实验与观察`
> - `Schedule-free 优化讨论` 


- ****内存效率：我们的模型表现显著优于 PyTorch****：我们的 batch 16 运行非常轻松，甚至 batch 24 也没问题，而 **PyTorch** 在 batch size 为 8 时就显得吃力。这突显了与 PyTorch 相比**显著的内存节省**。
- ****可视化与训练 Bug****：在 **HellaSwag eval dataloader** 中，当 batch size < 4 时观察到**整数除以零错误**。修复已在 [GitHub PR #667](https://github.com/karpathy/llm.c/pull/667) 中实现。
- ****muP 实验看起来很有前景但具有挑战性****：**初步 muP** 结果在各种学习率下表现稳定，大规模实验占用了每张 GPU 高达 **80 GB 的 VRAM**。正在计划进一步的超参数搜索。
- ****使用 HF Transformers 的高效推理****：从 **Hugging Face GPT2** 模型生成样本极其缓慢，由于低效的 eager mode 和动态 key-value 拼接，512 步需要 4 分钟。注意到正在积极改进 [Transformers 文档](https://huggingface.co/blog/transformers-docs-redesign)。
- ****对 Schedule-Free 优化器的兴奋****：[Facebook Research 的 Schedule-free 优化器](https://github.com/facebookresearch/schedule_free)在各种任务上表现出不可思议的收敛速度。有说法称这可能是实际和理论优化研究中的一项突破。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/transformers-docs-redesign">理清这一团乱麻</a>: 未找到描述</li><li><a href="https://x.com/_clashluke/status/1808590060654108910?s=46&t=Qzf619GMalbD77YmVui2Jw">来自 Lucas Nestler (@_clashluke) 的推文</a>: Schedule-free 优化器 (https://x.com/aaron_defazio/status/1776320004465582331) 非常不可思议。我读了论文，研究了数学原理，并试图理解发生了什么。这一切似乎……</li><li><a href="https://github.com/karpathy/llm.c/pull/667">修复 gordicaleksa 提交的 batch size < 4 时 eval dataloader 除以零的问题 · Pull Request #667 · karpathy/llm.c</a>: 我们至少需要 4 的 batch size 来支持当前的 eval 逻辑。或者我们可以稍微重写一下 eval，但目前这可能属于过度设计？可能发生的最坏情况是……</li><li><a href="https://github.com/karpathy/llm.c/pull/641/">修复 gordicaleksa 提交的添加函数检查版本 · Pull Request #641 · karpathy/llm.c</a>: 添加 socket 关闭检查函数——与代码库的其余部分保持一致。</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/default/train">HuggingFaceFW/fineweb-edu · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1257773576665170053)** (145 条消息🔥🔥): 

> - `反 AI 艺术软件辩论`
> - `低分辨率像素艺术训练技巧`
> - `Discord 上的职位发布`
> - `改进 Prompt 技巧与对比`
> - `SD 模型 MixofExperts 与 segmoe 的对比` 


- ****反 AI 艺术软件讨论****：成员们讨论了**反 AI 艺术软件**的可行性，这类软件旨在保护艺术家的作品不被用于 AI 训练。提到的现有工具包括 [**Glaze**](https://glaze.cs.uchicago.edu/) 和 **Nightshade**，但社区成员指出这些方法很容易被破解。
- ****训练低分辨率像素艺术模型****：一位用户询问关于 **16x16 像素艺术**的 AI 训练，成员建议将图像放大到 **512x512** 进行训练。Crystalwizard 指出了潜在的低效性，但建议将试错作为一种具有成本效益的方法。
- ****职位发布与自由职业****：一位用户询问是否有用于招聘的**职位发布频道**，另一位用户询问关于 **Upwork 账号租赁**的事宜，凸显了对**自由职业机会的需求**。
- ****Prompt 技巧的有效性****：成员们讨论了不同的 **Prompting 技巧**及其在 text2img 生成图像中的有效性。提到了如 **[A|B], C** 与 **[A, B, C]** 的变体，并对比了 **SD1.5** 与 **segmoe** 及 **MixofExperts** 的模型能力。
- ****对比 SD 模型：MixofExperts vs segmoe****：讨论涵盖了 **ComfyUI 中的 segmoe 模型**及其在 **Prompt 理解**方面的实质性改进。通过与 **SD1.5 微调模型及更新的模型**进行对比，强调了 Prompt 准确性和性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/cagliostrolab/animagine-xl-3.1">cagliostrolab/animagine-xl-3.1 · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=tZUMH_DUdfA&t=337s">AI News - ComfyUI Segmoe and Stable Video Diffusion 1.1</a>：该视频介绍了 Adobe Firefly 的新模型，强调了其在生成高质量图像（尤其是人物图像）方面改进的能力，以及...</li><li><a href="https://www.reddit.com/user/No_Dragonfruit_5472/comments/1chdemx/tradingview_premium_pack_crack_2024_version_free/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1257776175862779985)** (1 条消息): 

> - `/models 页面重大更新`
> - `更改 Gemini 和 PaLM 模型的 Google Token 大小`
> - `弃用设置页面的默认模型选项`
> - `弃用 OpenAI API 密钥的自定义认证标头` 


- ****/models 页面即将迎来重大更新****：**/models 页面**即将进行重大更新，并分享了预览。鼓励成员在[专用频道](https://discord.com/channels/1107397803266818229)提供反馈。
- ****Gemini 和 PaLM 模型的 Google Token 大小变更****：**Gemini** 和 **PaLM** 模型的 Token 长度将更改为与 GPT 等效的大小，Token 数量将增加约 **3 倍**并减少上下文限制，虽然模型和 API 保持不变，但这会导致价格上涨。
- ****弃用设置页面的默认模型****：**/settings 页面**上的**默认模型 (Default Model)** 选项将被弃用，因为大多数应用会自行设置模型或使用自动路由。鼓励有合理使用场景的用户提供反馈。
- ****弃用 API 密钥的自定义认证标头****：用于发送 OpenAI API 密钥的**自定义认证标头 (custom auth headers)** 将被弃用，替代方案即将推出。该功能在 6 月中旬曾被少数人使用，但从未正式记录在文档中。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1257816161379422219)** (3 条消息): 

> - `lastrosade 分享的简易封装器 (wrapper)`
> - `关于非流式响应的反馈` 


- ****分享简易封装器****：**lastrosade** 宣布创建了一个简易封装器 (wrapper)，并将其提供给社区中任何感兴趣的人。未提供额外的技术细节或链接。
- ****关于非流式响应的反馈****：社区成员 **clarie_starr** 对该封装器评论道：*“所以费了这么多劲就为了一个非流式响应。我得承认，它确实……挺详尽的。”* 随后 **lastrosade** 表示同意，称该封装器“很烂”。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1257772784084586579)** (77 messages🔥🔥): 

> - `Claude 3.5 的 500 错误`
> - `Claude 的自我审查问题`
> - `用于使用和越狱 Claude 的不同前端`
> - `OpenRouter 隐私设置和日志策略`
> - `Google 模型 Token 大小变更公告` 


- ****Claude 3.5 的 500 错误****：几位用户报告在 OpenRouter 上使用 **Claude 3.5** 时出现间歇性 **500 错误**。临时解决方法包括切换到 **Claude 3.0** 等不同版本。
- ****OpenRouter 隐私和日志问题得到解决****：用户讨论了 OpenRouter 的隐私设置，澄清了一些提供商会记录请求而另一些则不会，重点提到 **NovitaAI** 和 **Infermatic** 不保留数据。[Alex Atallah](https://openrouter.ai/settings/privacy) 提供了关于第三方提供商不同隐私政策的见解。
- ****Google 模型 Token 大小更新说明****：关于 **Google 模型 Token 大小变更** 的讨论引发了对潜在成本增加的担忧。[LouisGV](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b) 澄清说，尽管调整了 Token 大小，总价格仍大致保持不变。
- ****探索 Claude 的不同前端****：用户探索了如 **SillyTavern** 和 **LibreChat** 等各种前端，用于越狱或预填充 Claude 模型。建议将 **Typingmind** 和 **Pal Chat** 作为更流畅用户体验的替代方案。
- ****OpenRouter 上 LLM 模型的量化****：提出了关于 OpenRouter 上部署的 **LLM** 模型 **量化 (Quantization)** 的问题，重点在于模型是 **FP16** 还是其他精度。讨论强调，除非提供商另有说明，否则模型保持其原生精度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sillytavern.app)">未找到标题</a>: 未找到描述</li><li><a href="https://lmsys.org/blog/2024-07-01-routellm/">RouteLLM: An Open-Source Framework for Cost-Effective LLM Routing | LMSYS Org</a>: &lt;p&gt;LLM 在一系列任务中展示了卓越的能力，但它们的成本和能力存在很大差异，正如从...中看到的那样</li><li><a href="https://openrouter.ai/models/sao10k/l3-euryale-70b">Llama 3 Euryale 70B v2.1 by sao10k</a>: Euryale 70B v2.1 是一个专注于创意角色扮演的模型，来自 [Sao10k](https://ko-fi.com/sao10k)。- 更好的提示词遵循。- 更好的解剖学/空间意识。- 能更好地适应独特的...</li><li><a href="https://openrouter.ai/models/gryphe/mythomax-l2-13b">MythoMax 13B by gryphe</a>: Llama 2 13B 性能最高且最受欢迎的微调版本之一，具有丰富的描述和角色扮演能力。#merge</li><li><a href="https://openrouter.ai/settings/privacy">Settings | OpenRouter</a>: 管理您的账户和偏好设置</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b">WizardLM-2 8x22B by microsoft</a>: WizardLM-2 8x22B 是 Microsoft AI 最先进的 Wizard 模型。与领先的专有模型相比，它展示了极具竞争力的性能，并且始终优于所有现有的...</li><li><a href="https://infermatic.ai/privacy-policy/">Privacy Policy - Infermatic</a>: 未找到描述</li><li><a href="https://web.archive.org/web/20240112082806/https://infermatic.ai/privacy-policy/">Privacy Policy - Infermatic</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1257779373755465799)** (40 messages🔥): 

> - `Magic.dev 估值从 5 亿美元增至 15 亿美元，20 名员工，无产品，无收入。`
> - `关于具有 10 亿个人格的角色驱动数据合成的新论文。`
> - `Kyutai 推出的首个实时音频 LLM “Moshi”。`
> - `OpenDevin 创始人创立 All Hands AI。`
> - `Sentient 为开放 AGI 平台融资 8500 万美元种子轮。`

- **Magic.dev 估值在没有产品的情况下飙升至 15 亿美元**：[Magic.dev](https://www.reuters.com/technology/artificial-intelligence/ai-coding-startup-magic-seeks-15-billion-valuation-new-funding-round-sources-say-2024-07-02/) 的估值从 **5 亿美元** 跃升至 **15 亿美元**，而该公司仅有 20 名员工，且没有产品和收入。
- **10 亿个 Persona 助力 Synthetic data 生成**：[Persona Hub](https://arxiv.org/abs/2406.20094) 引入了 10 亿个 Persona 以扩展 Synthetic data 的创建，在数学问题解决和多样化场景中展现出巨大提升。
   - 由 [Aran Komatsuzaki](https://x.com/arankomatsuzaki/status/1807593343007818065) 介绍，这种新方法带来了显著改进，特别是在 **MATH** 基准测试上。
- **Kyutai 发布实时 Audio LLM 'Moshi'**：[Moshi](https://x.com/giffmana/status/1808482848808010149) 声称是首个延迟极低的实时 Audio LLM，尽管音质仍略显机械感。
   - [Kyutai](https://x.com/kyutai_labs/status/1808526962941366415) 的 'Moshi' 演示展示了其潜力，尽管目前还存在局限性，例如有时会因为急于响应而打断用户。
- **OpenDevin 创始人成立 All Hands AI**：[OpenDevin](https://x.com/gneubig/status/1808493521315496229) 创始人宣布成立 All Hands AI，旨在以 Open-source 方式为所有人加速 AI 软件开发。
- **Sentient 为 Open-source AI 平台获得 8500 万美元种子轮融资**：[Sentient](https://x.com/sentient_agi/status/1808136737257918916?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 宣布获得由 Founders Fund 领投的 **8500 万美元** 种子轮融资，以支持社区构建的 Open AGI 平台开发，旨在实现公平的 AI 发展。
   - 包括 [Peter Thiel](https://www.coindesk.com/business/2024/07/02/peter-thiels-founders-fund-leads-85m-seed-investment-into-open-source-ai-platform-sentient/) 在内的知名投资者正在支持这一倡议，旨在全球范围内分配 AI 收益。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.20094">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>：我们提出了一种新颖的角色驱动数据合成方法论，利用大语言模型 (LLM) 中的各种视角来创建多样化的合成数据。为了充分利用这一方法论...</li><li><a href="https://huggingface.co/CAMB-AI/MARS5-TTS">CAMB-AI/MARS5-TTS · Hugging Face</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Flowers_for_Algernon">Flowers for Algernon - Wikipedia</a>：未找到描述</li><li><a href="https://x.com/poolsideai/status/1738669662467178581">来自 poolside (@poolsideai) 的推文</a>：这是我们一直在进行的一些非常显眼的“水面上”的乐趣。期待很快能看到更多关于“水面下”进展的消息！</li><li><a href="https://x.com/SFResearch/status/1808549356536041487">来自 Salesforce AI Research (@SFResearch) 的推文</a>：感谢 @Benioff 和 @SilvioSavarese 指导我们的研究转向小语言模型 (SMLs) 的力量。xLAM-1B 已开源，并即将登陆 Hugging Face。💥 #SMLs #TinyGiant #AIRe...</li><li><a href="https://x.com/johnbyronhanby/status/1808235931784434049?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 John Byron Hanby, IV (@johnbyronhanby) 的推文</a>：作为有幸直接与上述一些公司的 CAIOs 合作过的人，这里有一些想法：- 机构如何找到担任此角色的人选？✅这个人需要具备...</li><li><a href="https://x.com/kyutai_labs/status/1808526962941366415">来自 kyutai (@kyutai_labs) 的推文</a>：https://moshi.chat/?queue_id=talktomoshi</li><li><a href="https://x.com/giffmana/status/1808482848808010149">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>：Kyutai Moshi - 首个实时音频 LLM。基本没有延迟——这个 LLM 甚至打断了说话者几次。它实际上有点急于快速回答。:) 全部将开源。质量...</li><li><a href="https://x.com/gneubig/status/1808493521315496229">来自 Graham Neubig (@gneubig) 的推文</a>：公告：@rbren_dev、@xingyaow_ 和我成立了一家公司！我们的名字是 All Hands AI 🙌 https://www.all-hands.dev/ 我们的使命是构建世界上最好的 AI 软件开发 Agent...</li><li><a href="https://x.com/arankomatsuzaki/status/1807593343007818065?utm_source=ainews&utm_medium=email&utm_campaign=ainews-to-be-named-5628">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：通过 1,000,000,000 个角色扩展合成数据生成 - 展示了从网络数据中自动策划的 10 亿个多样化角色集合 - 在 MATH 数据集上取得巨大进步：49.6 -> 64.9 仓库：https://g...</li><li><a href="https://github.com/hrishioa/rakis?tab=readme-ov-file">GitHub - hrishioa/rakis</a>：通过在 GitHub 上创建账号，为 hrishioa/rakis 的开发做出贡献。</li><li><a href="https://t.co/vQzLSq2ncG">Mozilla Llamafile 和 Builders 项目在 AI 工程师世界博览会上大放异彩</a>：在这次 AI 活动中，Mozilla 团队展示了 Llamafile 如何让开源模型更易于使用，并使其在消费级 CPU 上快速运行。</li><li><a href="https://x.com/sentient_agi/status/1808136737257918916?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Sentient (@sentient_agi) 的推文</a>：我们激动地宣布 Sentient 完成了 8500 万美元的种子轮融资，由 @foundersfund（Peter Thiel 参与）领投，@PanteraCapital 和 @hiFramework 参投。这标志着在调整 AI 发展方向上迈出了关键一步...
</li>
</ul>

### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1258135475609145376)** (34 条消息🔥): 

> - `OpenAI 在 AIEWF 演示期间的 AV 问题`
> - `迁移至 Zoom 以获得更好的无障碍体验`
> - `Discord 与 Linux 的不兼容性及建议的替代方案` 


- ****OpenAI 在 AIEWF 演示期间的 AV 困扰****：成员们在 **OpenAI AIEWF 演示**期间遇到了严重的 Discord 音视频 (AV) 问题，导致挫败感，且多名参与者无法看到屏幕。这导致了切换平台的建议。
- ****为获得更好的无障碍体验迁移至 Zoom****：由于 Discord 上持续出现的 AV 问题，成员们一致同意将 **Paper Club (West)** 会议迁移至 [Zoom](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09)。此次转变旨在解决可见性问题并提高会议质量。
- ****Discord 与 Linux 的不兼容性****：参与者强调了 Discord 在 Linux 上的兼容性是一个**已知问题**，这带来了额外的无障碍挑战。会议简要讨论了替代方案，表明未来需要一个更可靠的平台。



**提到的链接**：<a href="https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信领域的领导者，拥有简便、可靠的云平台，可跨移动端、桌面端和会议室系统进行视频和音频会议、聊天及网络研讨会。Zoom ...

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1258025787882475611)** (2 条消息): 

> - `jaan.li 介绍了他们在 onefact.org 和 usb.club 的工作`
> - `san.tosh 询问关于开源 GPT-4o 的更新` 


- **jaan.li 构建去中心化边缘 Transformer**：**jaan.li** 宣布了他们在 [onefact.org](https://onefact.org) 和 usb.club 关于去中心化边缘 Transformer 的工作。可以随时通过 jaan@onefact.org 联系他们。
- **关于开源 GPT-4o 更新的查询**：**san.tosh** 询问是否有关于开源 GPT-4o 的任何更新。该查询目前尚无定论。


  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1257916724435615814)** (59 messages🔥🔥): 

> - `Terminator 模型的消融实验及变更理由`
> - `关于 slow-fast 网络及其优势的讨论`
> - `在 GitHub 上发布了 Terminator 的代码`
> - `引入 FORA 以加速 Diffusion transformers`
> - `对 HyperZ⋅Z⋅W 论文的批评与建议` 


- ****Terminator 模型消融实验受到批评****：成员们讨论了 *Terminator* 模型缺乏足够的消融实验和变更理由，尽管其 Benchmark 显示了令人印象深刻的性能。他们强调需要详细的消融研究来突出各个组件（如 residuals, dot product attention, intermediate pooling）的影响。
- ****关于 ViT 中 QKV 冗余的辩论****：关于 Vision Transformers (ViT) 中的 QKV 是否冗余展开了激烈辩论，有建议认为 Q 和 K 对于 Attention 矩阵的生成可能是不必要的。一些成员认为需要适当的评估和证明来验证这一理论。
- ****Terminator 代码在 GitHub 发布****：**Terminator** 代码已在 GitHub 上公开发布，反驳了一些称其为虚假软件（vaporware）的说法。用户现在可以访问[官方仓库](https://github.com/hyperevolnet/Terminator)。
- ****FORA 加速 Diffusion transformers****：引入了一种名为 **Fast-FORward CAching (FORA)** 的新方法，通过缓存和重用中间输出来加速 Diffusion transformers，显著降低了计算开销。该方法与现有模型无缝集成，在质量损失极小的情况下提供更快的处理速度。[阅读更多](https://github.com/prathebaselva/FORA?tab=readme-ov-file)。
- ****HyperZ⋅Z⋅W 论文收到褒贬不一的评价****：@harvie_zhang 的 **HyperZ⋅Z⋅W** 论文收到了社区的称赞与批评，被认为是一篇虽然粗糙但具有实现 SOTA 新颖想法的提交。作者承认了反馈，表示未来将进行修订并可能进行消融研究，以证明 ViT 中 QKV 的冗余性。*阅读 Schmidhuber 的综述[此处](https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html)。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/kronk-disney-the-emperor%E2%80%99s-new-groove-emperor%27s-new-groove-disney%E2%80%99s-emperor%E2%80%99s-new-groove-gif-9209845644877110421">Kronk Disney GIF - Kronk Disney The emperor’s new groove - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/prathebaselva/FORA?tab=readme-ov-file">GitHub - prathebaselva/FORA</a>：通过在 GitHub 上创建账号为 prathebaselva/FORA 的开发做出贡献。</li><li><a href="https://github.com/hyperevolnet/Terminator">GitHub - hyperevolnet/Terminator: HyperZ⋅Z⋅W 算子连接 Slow-Fast 网络以实现全上下文交互的官方仓库。</a>：HyperZ⋅Z⋅W 算子连接 Slow-Fast 网络以实现全上下文交互的官方仓库。 - hyperevolnet/Terminator
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1257824887288893440)** (26 messages🔥): 

> - `Image dtype 特殊处理`
> - `tinygrad 中的运行时错误`
> - `UNMUL 模式匹配器问题`
> - `前端 fuzzer 想法`
> - `循环优化 bug` 


- ****Tinygrad 在 UOps.UNMUL 上面临运行时错误****：一名成员报告了 tinygrad 代码库中的 **RuntimeError**：*'failed to render UOps.UNMUL'*。**George Hotz** 建议将此问题视为 assert，并表示它“永远不应该发生”。
   - *Chenyuy* 建议添加 `flat_l4.realize()` 作为潜在的变通方法，并建议使 loop collapse 变为可选，以减轻对用户的影响。
- ****针对 tinygrad 的前端 fuzzer 提案****：*Chenyuy* 提出了为 tinygrad 开发 **前端 fuzzer** 的想法，可能使用 LLM 来移植 torch 代码。该建议旨在在开发过程中捕获更多边缘情况和意外行为。
- ****处理 UNMUL 模式匹配器 bug****：*Chenyuy* 指出，改变输入维度有时可以避免触发该 bug，这凸显了循环优化的不完整。*Yosifrost* 发现特定维度会影响 bug 的发生，表明启发式边界行为存在问题。
   - 成员们讨论了编写最小复现（repro）测试以隔离 bug 的可能性，并打算保持 PR 开启以进行进一步调查和重点测试。
- ****1.0 版本前的错误处理和开发工具改进****：*Yosifrost* 强调在 1.0 版本之前，tinygrad 必须提供**更好的错误消息**和开发工具。多位成员协作复现错误并开发最小测试用例以进行进一步调试。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1257774565673795748)** (33 messages🔥): 

> - `tinygrad 中 torch.no_grad() 的等效实现`
> - `在 tinygrad 开启梯度时 "-=" 运算符的不兼容性`
> - `处理导致 CUDA 内存错误的梯度累积问题`
> - `梯度累积期间 TinyJit 的减速和内存问题`
> - `tinygrad 与 PyTorch 中 Tensor 创建方法的行为差异` 


- ****tinygrad 拥有 torch.no_grad() 的等效实现****：一位用户询问 tinygrad 中 `torch.no_grad()` 的等效实现，解释称可以使用 `Tensor.no_grad = True` 和 `@Tensor.inference_mode()` 装饰器，示例代码见[此处](https://github.com/tinygrad/tinygrad/blob/master/examples/mlperf/model_train.py)。
- ****开启梯度时 "-=" 运算符的不兼容性****：一位用户指出，由于 tinygrad 的梯度限制，`a -= lr * a.grad` 会触发断言（assert），而 `a = a - lr * a.grad` 可以正常工作。该问题参考了[此处](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py#L228)的代码。
- ****梯度累积与 CUDA 内存问题****：用户讨论了梯度累积导致 CUDA 内存越界（out of bounds）错误的问题。建议包括对 loss 进行 detach 操作，以及解决优化器中的 `assert t.grad is not None` 问题。
- ****TinyJit 在优化步骤中导致的错误****：据透露，当 TinyJit 未应用于整个 step 时，会因 `assert t.grad is not None` 而失败，从而导致效率低下。用户建议从 jit 函数返回 realized 梯度，并在外部计算 step。
- ****tinygrad 中 Tensor 创建方法的不一致性****：一位用户观察到 `Tensor.randn/randint` 创建的是连续（contiguous）Tensor，而 `Tensor.full` 创建的是非连续 Tensor，这与 PyTorch 的行为不同。这被确认为 tinygrad 的预期行为，并讨论了未来版本中可能的改进。



**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py#L228">tinygrad/tinygrad/tensor.py at master · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1257852461645959179)** (3 messages): 

> - `在 Raspberry Pi 上构建 RAG 流水线`
> - `OpenContracts AI 驱动的文档分析工具`
> - `关于使用 Weights & Biases 进行 RAG 实验和评估的网络研讨会` 


- **Raspberry Pi 上的 RAG 🍓🔎**：@pavan_mantha1 的教程演示了如何使用 **Docker** 和 **Ollama** 在 **Raspberry Pi** 上构建 **RAG 流水线**。查看该 [推文](https://twitter.com/llama_index/status/1808292764129583179) 了解更多详情。
   - 该项目展示了如何像 **Raspberry Pi** 这样的小型嵌入式设备上高效运行 **RAG 流水线**。
- **OpenContracts 发布 ✨**：**OpenContracts** 是由 @johnscrudato 开发的开源 AI 驱动文档分析工具，允许用户使用 **LLMs** 和 **Llama Index** 分析、标注和共享文档。更多信息可以在[这里](https://twitter.com/llama_index/status/1808528869252812902)找到。
   - 该项目是 **genAI native**（原生生成式 AI）的，旨在通过高效使用 **AI** 使文档分析民主化。
- **RAG 实验网络研讨会 🚨**：正与 **Weights & Biases** 合作举办一场关于 **RAG 实验与评估** 的网络研讨会。该会议旨在教授如何构建、评估和迭代 **RAG 流水线**，详情见此 [推文](https://twitter.com/llama_index/status/1808589017744880062)。
   - 经过一年多的 **RAG 开发**，本次网络研讨会将解决该领域中正确评估的挑战。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1257789737964671127)** (49 条消息🔥): 

> - `DocumentSummaryIndex 在 Pinecone 限制方面的问题`
> - `元数据排除的代码片段和潜在修复方案`
> - `Pinecone 的替代向量存储方案`
> - `LlamaIndex 的单一 LLM 支持问题`
> - `PDF 表格的解析问题` 


- ****DocumentSummaryIndex 触及 Pinecone 限制****：一位用户报告了在创建 **DocumentSummaryIndex** 时，**Big Docs** 超过了 **Pinecone 限制**的问题。该问题源于第一个节点的元数据过大，且 **embed exclusion filters** 似乎未被正确应用。[GitHub 链接](https://github.com/run-llama/llama_index/blob/722cb67ca4e52c8c4d6ef8c5e99b7f6c9f57e244/llama-index-core/llama_index/core/indices/document_summary/base.py#L203)
   - 另一位用户建议不要在文档/节点中包含过多元数据，并提供了一个潜在的代码修复方案来排除嵌入元数据键。此外，他们建议考虑 Pinecone 的替代方案，如 **qdrant** 或 **pg_vector**。
- ****LlamaIndex 仅支持 OpenAI LLM****：多位用户指出 **LlamaIndex 目前仅支持 OpenAI 作为其 LLM**，这引起了一些不满。他们建议扩展对其他 LLM 的支持将大有裨益。
   - *“看起来目前它只支持 OpenAI 作为 LLM ... 如果是这样的话 👎”*
- ****PDF 转 Markdown 解析器在处理表格时遇到困难****：一位用户尝试使用 [Marker](https://github.com/VikParuchuri/marker) 将 PDF 转换为 Markdown，但发现格式“奇特”的表格会导致解析问题。他们正在寻找更好的本地或开源解决方案，但提到 **Azure Document Intelligence** 的表现更好。
   - 另一位用户建议尝试 **Unstructured** 或 **llamaparse**，尽管它们不是开源的。这些工具似乎能更好地处理复杂的表格结构。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/blob/722cb67ca4e52c8c4d6ef8c5e99b7f6c9f57e244/llama-index-core/llama_index/core/indices/document_summary/base.py#L203">run-llama/llama_index 中的 llama_index/llama-index-core/llama_index/core/indices/document_summary/base.py (版本 722cb67ca)</a>：LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/extraction/">结构化数据提取 - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: 快速且高精度地将 PDF 转换为 markdown</a>：快速且高精度地将 PDF 转换为 markdown - VikParuchuri/marker
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1258027183125954571)** (5 条消息): 

> - `结合 LlamaIndex、Claude-3.5 Sonnet 和 MongoDB 的 Agentic RAG`
> - `用于在 Mac 上运行私有 AI/LLM Agent 和工具调用工作流的 Toolio` 


- ****Agentic RAG 引起关注****：[**释放 AI 潜力：结合 LlamaIndex、Claude-3.5 Sonnet 和 MongoDB 的 Agentic RAG**](https://medium.com/ai-advances/unleashing-ai-potential-agentic-rag-with-llamaindex-claude-3-5-sonnet-and-mongodb-ea126164a801) 一文讨论了 AI 领域的创新策略。一位成员暗示该内容很快将得到推广。
- ****Toolio 简化了 Mac 上的私有 AI 工作流****：[Toolio](https://www.youtube.com/watch?v=9DpQYbteakc) 允许用户在 Mac 上轻松运行私有 AI/LLM Agent 和工具调用（tool-calling）工作流，支持 JSON schema 约束并提供快速推理。一位支持者声称工具调用是“LLM 真正的魔力所在”，并期待该领域出现有意义的创新。


  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1257899774972268584)** (1 条消息): 

> - `Tortoise-TTS 转换为 ggml`
> - `实时推理优化`
> - `GitHub 上的开源项目`
> - `Tortoise-TTS 的 CUDA 和 CPU 支持` 


- ****Tortoise-TTS 转换为 GGML****：一名成员将 **Tortoise-TTS** 转换为了 **ggml**，并正在寻求帮助以改进 [实时文本转语音](https://github.com/balisujohn/tortoise.cpp) 的推理时间。该仓库已支持 CUDA 和 CPU。
- ****AI 开发者的优化机会****：[Tortoise-TTS ggml 项目](https://github.com/balisujohn/tortoise.cpp) 提供了一个练习优化 **transformers** 和 **diffusion models** 的绝佳机会。目标是加速推理过程。



**提到的链接**：<a href="https://github.com/balisujohn/tortoise.cpp">GitHub - balisujohn/tortoise.cpp: A ggml (C++) re-implementation of tortoise-tts</a>：tortoise-tts 的 ggml (C++) 重新实现。可以通过在 GitHub 上创建账号来为 balisujohn/tortoise.cpp 的开发做出贡献。

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1257849473309212743)** (42 条消息🔥): 

> - `去中心化训练的私有频道`
> - `Hermes 2 Pro 在 vLLM 中的 Tool calling`
> - `关于 Hermes 2 Pro 中处理 tool calls 和文本内容的讨论` 


- ****Hermes 2 Pro 的 Tool calling 在 vLLM 中正常工作****：一名成员宣布 **tool calling 现在在 vLLM 中为 Hermes 2 Pro 正常运行**。他们表示该项目已非常接近完成。
- ****Hermes 3 训练包含 <scratch_pad>****：团队讨论了在 **Hermes 3 训练**中的 tool calls 之前添加 `<scratch_pad>`，旨在改进解析，从而提取 `<scratch_pad>` 之间的内容并同时处理 'content' 和 'tool_calls'。
   - *讨论内容包括处理 tool calls 之前的文本内容，并确保与 OpenAI 的规范兼容。*



**提到的链接**：<a href="https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ">Neural Networks: Zero to Hero</a>：未找到描述

  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1257773242156847146)** (3 条消息): 

> - `从文档创建对话数据集`
> - `从文档生成指令`
> - `用于生成指令的 Genstruct 7B 模型` 


- ****从文档创建数据集****：从文档 **创建对话数据集** 取决于文档内容和预算。选项包括使用语言模型生成数据集，或使用来自 **Anthropic** 等的工具。
- ****Genstruct 7B 生成指令****：[Genstruct 7B](https://huggingface.co/NousResearch/Genstruct-7B) 模型可以从原始文本语料库生成有效的指令，适用于创建指令微调（instruction finetuning）数据集。它的灵感来自 [Ada-Instruct](https://arxiv.org/abs/2310.04484) 模型。



**提到的链接**：<a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>：未找到描述

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1257918603588800626)** (5 条消息): 

> - `Huggingface 在 Cohere 的 CommandR 模型上的 PR`
> - `微软发布 GraphRAG` 


- **Huggingface 在 Cohere CommandR 上的 PR 扩展了 Tool-Use**：Huggingface 在 Cohere 的 CommandR 模型上提交了一个 [PR](https://github.com/cohere/CommandR)，重点关注 *tool-use 和 RAG 模板的技术改进*。System prompt 使用前导码（preamble）和带有 Jinja 模板的动态内容编排来构建。
- **微软发布 GraphRAG**：微软发布了一个名为 *GraphRAG* 的模块化基于图的检索增强生成（RAG）[系统](https://github.com/microsoft/graphrag)，旨在提升信息检索和生成。该工具已在 GitHub 上发布，旨在增强 RAG 系统的模块化和有效性。



**提到的链接**：<a href="https://github.com/microsoft/graphrag">GitHub - microsoft/graphrag: A modular graph-based Retrieval-Augmented Generation (RAG) system</a>：一个模块化的基于图的检索增强生成（RAG）系统 - microsoft/graphrag

  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1257907166334156832)** (7 messages): 

> - `Ubuntu 24.04/Python 3.12.3 上的安装问题`
> - `Ubuntu 24.04/Python 3.12.3 上 Mojo/max 的实现变通方案`
> - `Mojo 隐式转换 Bug`
> - `Mojo 中的类型转换 Bug` 


- ****在 Ubuntu 24.04 上面临安装问题****：一位用户报告了在 **Ubuntu 24.04/Python 3.12.3** 上的安装问题，由于版本不匹配收到 **max-engine** 的错误。
   - 另一位用户分享了一个[分步指南](https://docs.modular.com/mojo/manual/python/#resolving-issues)，通过安装 Python 3.11 并调整 alternatives 来解决此问题。
- ****Mojo 奇怪的隐式转换****：一位用户注意到在 Mojo 中将整数乘以 `np.pi` 会产生意外的**负整数**结果，这是由于一个 [隐式转换 Bug](https://github.com/modularml/mojo/issues/3146) 导致的。
   - 讨论指出，这与已追踪为 [#3065](https://github.com/modularml/mojo/issues/3065) 和 [#3167](https://github.com/modularml/mojo/issues/3167) 的类型转换 Bug 有关。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/python/#resolving-issues">Python 集成 | Modular 文档</a>：同时使用 Python 和 Mojo。</li><li><a href="https://github.com/modularml/mojo/issues/3146).">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/3065)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/3167)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1257789972505825382)** (2 messages): 

> - `` 


- ****Modular 在 Twitter 上分享令人兴奋的更新****：[Modular](https://twitter.com/Modular/status/1808228006068212110) 最近在推特上发布了有趣的更新。更多细节可以在他们最新的 Twitter 公告中找到。
- ****Modular 在 Twitter 上的进一步公告****：[Modular](https://twitter.com/Modular/status/1808567651280777598) 在几天后分享了额外的更新。查看他们的 Twitter 获取完整帖子。


  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1258126192523874418)** (1 messages): 

> - `Mojo N-Body 示例基准测试`
> - `Mojo 中的单核数值性能`
> - `N-body.js 中的辛积分器 (Symplectic Integrator)`
> - `N-Body 示例中的向量化 (Vectorization)`
> - `常微分方程求解器` 


- **Mojo 引入 N-Body 示例**：[Modular 的博客](https://www.modular.com/blog/a-brief-guide-to-the-mojo-n-body-example) 详细介绍了自 2023 年 8 月起包含在仓库中的 **Mojo N-Body 示例**，该示例基于 [The Computer Language Benchmarks Game](https://en.wikipedia.org/wiki/The_Computer_Language_Benchmarks_Game)。此基准测试模拟了类木行星的轨道，并测试了单核数值性能。
- **N-body 基准测试亮点**：[N-body 是 Computer Language Benchmark Game 的基准测试之一](https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/nbody.html#nbody)，使用 **辛积分器 (symplectic integrator)** 模拟 **类木行星**。虽然主要是单核，但可以实现基础的向量化以增强性能。



**提到的链接**：<a href="https://www.modular.com/blog/a-brief-guide-to-the-mojo-n-body-example">Modular: Mojo N-Body 示例简要指南</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新帖子：Mojo N-Body 示例简要指南

  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1257817463840575630)** (8 messages🔥): 

> - `Mojo List 类型与打印问题`
> - `打印 RepresentableCollectionElement 类型`
> - `在 Mojo 中内联打印错误`
> - `过多空行对启动时间的影响`
> - `由于 bench_matmul 中的循环展开导致 Mojo 程序启动时间不稳定` 


- ****Mojo List 困惑****：一位成员对 Mojo 中的 `List[String]` 表示困惑，指出尽管它包含字符串，但缺少 `Stringable` trait，从而影响了它的可打印性。他们提供了一个 [GitHub 链接](https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/src/builtin/str.mojo#L23) 作为参考。
- ****在 Mojo 中打印 RepresentableCollectionElement 类型****：有人建议使用 `print(list.__str__())` 来打印 Mojo 中 `RepresentableCollectionElement` 类型的 List，并提供了一个 [GitHub 链接](https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/src/collections/list.mojo#L338) 以了解更多详情。
- ****由于循环展开导致的启动时间不稳定****：一位用户注意到他们的 Mojo 程序启动时间不稳定，最初认为与过多的空行有关。另一位用户澄清说，这是由于 `bench_matmul` 展开了大量循环，导致编译速度变慢。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/src/builtin/str.mojo#L23">mojo/stdlib/src/builtin/str.mojo at 8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2 · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/src/collections/list.mojo#L338">mojo/stdlib/src/collections/list.mojo at 8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2 · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/)** (1 messages): 

melodyogonna: 早期工具链的乐趣
  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1257810211893285017)** (31 messages🔥): 

> - `MLM 中的并行处理`
> - `矩阵乘法优化`
> - `Strassen 算法性能`
> - `SPIRAL 项目`
> - `矩阵乘法中的数值稳定性` 


- **SPIRAL 项目旨在实现自动化高性能库**：[SPIRAL 项目](http://www.spiral.net/) 专注于 DSP 算法和其他数值内核的软件和硬件开发自动化，在直接硬件任务中通常优于 MKL。
- **Strassen 算法 vs 朴素向量化方法**：对于 1024x1024 矩阵，**Strassen 算法** 达到了约 **50 GFlops**，而朴素向量化和并行化版本达到了 **70 GFlops**。详情见 [GitHub](https://github.com/RedKinda/Mojo-Marathons/)。
   - 诸如向量化加/减法和并行化子矩阵乘法等优化增加了额外的 3-5 GFlops，而 [减少中间分配](https://github.com/RedKinda/Mojo-Marathons/) 和针对非方阵更好的保护措施是潜在的未来改进方向。
- **并行和向量化操作的概念挑战**：讨论强调了在并行、向量化、展开（unroll）以及用于调优算法的部分分块（tiling）之外的概念难度。
- **Strassen 算法中的数值稳定性影响**：据报道，[Strassen 算法](https://en.wikipedia.org/wiki/Strassen_algorithm) 会降低数值稳定性，导致在使用不同类型和尺寸时测试失败。
- **递归 vs 迭代算法的缓存局部性**：建议不同的类型尺寸可能会从递归算法中受益，而不是迭代算法，以获得更好的缓存局部性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="http://www.spiral.net/">SPIRAL Project: Home Page</a>：未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1TBz9Lp0JT1Ph7ndfbWqp-B30FQcRYl1959hP2lZ6yH4/edit">Matrix Multiplication</a>：Sheet1 约束、参数 / 调优向量化、连续访问、Nelts、可展开并行化、可展开循环展开、连续操作分块方阵优化、摊销增加、递归...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1257821853884616804)** (29 条消息🔥): 

> - `Apple 在 OpenAI 获得董事会观察员席位`
> - `Microsoft 对 OpenAI 的投资以及与 Apple 合作伙伴关系的比较`
> - `Kyutai Labs 的新型实时语音 LLM 'Moshi'`
> - `'Moshi' 的训练细节和技术规格`
> - `Kyutai Labs 的开源模型发布和未来计划` 


- **Apple 在 OpenAI 获得董事会观察员席位**：作为 Apple Intelligence 合作伙伴关系的一部分，Apple 将于今年晚些时候在 OpenAI 获得一个**董事会观察员席位**，由 **Phil Schiller** 担任该职位，据 [Bloomberg](https://www.bloomberg.com/news/articles/2024-07-02/apple-to-get-openai-board-observer-role-as-part-of-ai-agreement) 报道。
- **OpenAI 合作伙伴关系中的 Microsoft 与 Apple**：社区成员将 **Microsoft 对 OpenAI 的数十亿美元投资**与 **Apple 的合作伙伴关系**进行了比较，指出 Apple 似乎获得了更多好处，包括应用和 iPhone 集成，而 Microsoft 则没有。
- **Kyutai Labs 发布 'Moshi'：实时语音 LLM**：**Kyutai Labs** 在直播更新中介绍了 **Moshi**，这是首个具有 **150ms 延迟**的实时语音 LLM，展示了其**多模态 LM 能力**以及在端侧使用的潜力。
- **Moshi 令人印象深刻的技术细节**：据多位社区成员称，Moshi 基于 **7B 多模态 LM** 构建，拥有压缩系数达 300 倍的 **VQ-VAE 语音编解码器**，并提供**超越人类的响应速度**。
- **开源模型发布及 'Moshi' 的未来计划**：Kyutai Labs 计划发布**开源模型**，包括 7B 多模态 LM、语音编解码器和优化后的堆栈。用户已经开始在**真实场景**中测试该模型，并讨论了其响应延迟和潜在用途。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/giffmana/status/1808482848808010149">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>: Kyutai Moshi - 首个实时语音 LLM。基本没有延迟 - LLM 甚至几次打断了发言者。它实际上有点急于快速回答。:) 全部将开源。质量 ...</li><li><a href="https://x.com/BartokGabi17/status/1808242102750568799">来自 Bartok Gabriel (@BartokGabi17) 的推文</a>: @markgurman Microsoft 向 OpenAI 投资了数十亿却没有得到应用，Apple 仅凭曝光度付费，OpenAI 就制作了一个很棒的应用和深度的 iPhone 集成。利润？？Tim Apple 是个天才。</li><li><a href="https://x.com/thexeophon/status/1808481304117227794?s=46">来自 Xeophon (@TheXeophon) 的推文</a>: 引用 kyutai (@kyutai_labs) 明天欧洲中部时间下午 2:30 加入我们的直播，了解我们研究的一些令人兴奋的更新！ https://www.youtube.com/live/hm2IJSKcYvo</li><li><a href="https://x.com/reach_vb/status/1808528557431210236">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>: 太棒了！@kyutai_labs 的 Moshi 惊艳全场！🇪🇺/acc。架构 1. 7B 多模态 LM（语音输入，语音输出） 2. 双通道 I/O - Streaming LM 不断生成文本 Token 以及一个...</li><li><a href="https://x.com/markgurman/status/1808240961522159862">来自 Mark Gurman (@markgurman) 的推文</a>: 最新消息：作为 Apple Intelligence 合作伙伴关系的一部分，Apple 将于今年晚些时候在 OpenAI 获得一个董事会观察员席位。获得该席位的人选是：App Store 负责人、前市场营销主管 Phil Schiller...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1257958333252374609)** (12 messages🔥): 

> - `SB 1047 与第一修正案挑战`
> - `第一修正案下对 3D 枪支设计的保护`
> - `模型权重和代码作为受保护的言论`
> - `对 Claude 3.5 的赞赏`
> - `使用 Claude TM` 


- ****SB 1047 面临潜在的第一修正案挑战****：一场关于 **SB 1047** 如果通过是否能在 **第一修正案挑战** 中幸存下来的讨论展开了，特别是将其与 **3D 枪支设计** 的保护以及代码作为言论自由进行了比较，引用了 [EFF 关于代码即言论的案例](https://www.eff.org/deeplinks/2015/04/remembering-case-established-code-speech)。
- ****法院关于代码作为受保护语言的观点****：一段引用强调“计算机语言与德语或法语之间没有实质性区别”，指出两者都传递信息，因此受第一修正案保护，并将其比作“音乐和数学方程式”。
- ****关于模型权重作为受保护言论的辩论****：关于 **模型权重** 是否可以被视为受第一修正案保护的辩论仍在继续，讨论围绕其被分类为“高级语言”还是仅仅是“数学方程式”，并与发布随机单词序列进行了比较。
- ****Claude 3.5 热潮****：“woo Claude 3.5 admiration let's go” —— 成员们对近期发布的 **Claude 3.5** 感到非常兴奋。
- ****推广 Claude TM****：有人建议 **推广 Claude TM**，将其营销活动与 **American Airlines** 进行比较，并带有热情的评论：*“我再也回不去了。”*



**提到的链接**：<a href="https://www.eff.org/deeplinks/2015/04/remembering-case-established-code-speech)">Deeplinks Blog</a>：未找到描述

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1257789781291696138)** (30 messages🔥): 

> - `用于回答通用问题的 RAG 策略`
> - `AzureAIDocumentIntelligenceLoader 的速率限制问题`
> - `使用不同工具和库解析 PDF`
> - `LangSmith 追踪问题`
> - `LangChain 中的通用帮助和故障排除` 


- **Azure 与 PDF 加载器冲突**：一位用户从 **PyPDFium2Loader** 切换到 **AzureAIDocumentIntelligenceLoader** 后遇到了持续的 **429 错误 (Too Many Requests)**。这表明由于 AzureAIDocumentIntelligenceLoader 处理文档的方式，可能存在速率限制问题。
- **PDF 转 Markdown 的痛点**：一位成员尝试使用 [marker](https://github.com/VikParuchuri/marker) 将 PDF 转换为 Markdown，但在解析具有“奇特”格式（如合并单元格）的表格时遇到问题。他们指出 **Azure Document Intelligence** 在处理此类文档时表现更好，但表示更倾向于使用本地或开源的替代方案。
- **LangSmith 停止追踪调用**：一位用户报告称 **LangSmith** 在没有任何明确原因的情况下停止了对他们调用的追踪。这表明 LangChain 的追踪机制可能存在 Bug 或问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://v02.api.js.langchain.com/classes/langchain_community_document_loaders_fs_pdf.PDFLoader.html">PDFLoader | LangChain.js - v0.2.8</a>：未找到描述</li><li><a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>：快速且高精度地将 PDF 转换为 Markdown - VikParuchuri/marker</li><li><a href="https://community.openai.com/t/using-gpt-4-api-to-semantically-chunk-documents/715689/136">Using gpt-4 API to Semantically Chunk Documents</a>：好的，经过 2 个月，我终于建立了一个可以实时运行的功能完备的系统。流程如下：将 PDF（或其他）文档导出为 txt。我设置为使用：AWS Textract, PdfToTe...</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/toolkits/openapi/#lets-see-some-examples>).">OpenAPI | 🦜️🔗 LangChain</a>：我们可以构建 Agent 来调用任意 API，这里的 API 符合 OpenAPI/Swagger 规范。</li><li><a href="https://github.com/langchain-ai/langchain/issues/832>).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建一个账号来为 langchain-ai/langchain 的开发做贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2333>).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建一个账号来为 langchain-ai/langchain 的开发做贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1257774280767180820)** (2 messages): 

> - `直接上传 CSV 文件 vs. 提供文件路径`
> - `CSV playground 中不显示输出`
> - `CSV 处理的代码改进`
> - `用于文件上传的 FastAPI 端点`
> - `Chroma vectorstore 的使用与问题` 


- ****支持用户上传 CSV 文件****：一位用户寻求帮助，希望在项目中实现 CSV 文件上传功能，而不是让用户指定文件路径。他们需要在 FastAPI 设置中实现此功能以提高可用性。
- ****CSV Playground 无输出****：在 `csv/playground/` 目录下，尽管代码看起来正确，但存在不显示输出的问题。这表明文件处理或输出渲染逻辑中可能存在潜在问题。
- ****改进 CSV 处理代码****：用户正在寻求改进现有代码的指导，目前该代码要求用户手动设置文件路径。需要关于提高代码效率和可用性的建议。


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1257773671532204032)** (2 messages): 

> - `OpenAI CriticGPT 论文讨论`
> - `用于私有 LLM 的 Toolio 开源项目` 


- **OpenAI 推出 CriticGPT 以识别 GPT-4 的错误**：一名成员分享了一个 [YouTube 视频](https://youtu.be/4PgcaIfwLjo)，讨论了 OpenAI 关于 **CriticGPT** 的最新论文，该模型旨在纠正 **GPT-4** 犯的错误。视频强调了 CriticGPT 的主要特征及其在提高 AI 生成代码可靠性方面的意义。
- **Toolio 助力 Mac 上的私有 LLM 工作流**：一名成员宣布发布 **Toolio**，这是一个开源项目，可以轻松地在 Mac 上运行私有 **LLM agents** 和工具调用（tool-calling）工作流。该项目在 [YouTube 视频](https://www.youtube.com/watch?v=9DpQYbteakc)中展示，还具有 JSON schema 输出约束和快速推理能力。



**提到的链接**：<a href="https://youtu.be/4PgcaIfwLjo">OpenAI 发布 CriticGPT 以纠正 GPT-4 的错误 | 与我一起读论文</a>：OpenAI 揭晓了 CriticGPT，这是一种基于 GPT-4 的新 AI 模型，旨在识别 ChatGPT 生成的代码中的错误，标志着向改进...迈出了重要一步。

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages): 

dracount: 大家好，有没有人可以推荐适合初学者的 LangChain/LangGraph 教程？
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1257796022168584302)** (25 messages🔥): 

> - `运行 llamafile 的硬件推荐`
> - `大语言模型的 VRAM 和 CPU 占用`
> - `CPU 推理的 Syncthread 技巧`
> - `在高端工作站上运行 llama3 70B`
> - `Rockchip RK3588 NPU 对 llamafile 的支持问题` 


- ****构建最佳 Llamafile 配置****：成员们讨论了运行 **llamafile** 的新 Linux 电脑的硬件推荐。推荐的 GPU 包括消费级的 **3090/4090** 和工作站级的 **A6000/RTX 6000 Ada**；在高端配置方面，建议使用旧款 **EPYC CPU**，因为它们拥有更多的核心和 PCIe 通道支持。
- ****大模型的 VRAM 需求****：为了高效运行大型模型，**更大的 VRAM** 至关重要；例如，24GB VRAM 可以在 **q4** 量化下处理 33B 参数。不建议使用 FP16 模式，因为与极小的质量损失相比，它的 VRAM 需求巨大。
- ****CPU 推理中的 Syncthread 技巧****：成员们讨论了利用用于 CPU 推理的 syncthread 技巧进行 **CPU 学习** 的潜力。分享了一个解释该概念的 [YouTube 演讲](https://www.youtube.com/live/5zE2sMka620?feature=shared&t=3140)。
- ****在 Threadripper 上运行 Llama3 70B****：一名成员成功在高端 **Threadripper CPU** 工作站上运行了 **llama3 70B**。这展示了在满足足够规格的情况下，CPU 处理大型模型的能力。
- ****Rockchip RK3588 NPU 的问题****：关于在 **Rockchip RK3588 NPU** 上运行 **llamafile** 存在疑问。建议使用 **v0.8.9** 版本以避免此类硬件上的地址空间问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/live/5zE2sMka620?feature=shared&t=3140">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>: https://twitter.com/aidotengineer</li><li><a href="https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#memorydisk-requirements">GitHub - ggerganov/llama.cpp: C/C++ 中的 LLM 推理</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建一个账号来为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1257826237683077130)** (22 条消息🔥): 

> - `phi mini 在相同仓库发布了新权重`
> - `使用 eleutherai 的 eval harness 进行 torchtune 评估`
> - `评估基础及 wandb 上的日志`
> - `关于训练梯度和 epoch 的讨论`
> - `FullModelHFCheckpointer 以及 HF 格式与 torchtune 之间的转换` 


- **phi mini 的新权重**：**phi mini** 获得了新权重，但沿用了之前的仓库。用户假设旧的 recipe 无需更新即可在 torchtune 中使用。
- **训练策略与评估**：讨论了不同的 **gradients 8 vs 16** 和 batch size 2，以确定哪种组合更适合数据集以及是否需要调整 epoch。分享了 **Wandb logs** 和评估教程，以帮助更好地理解和跟踪性能指标。
- **HF 格式的转换参数**：询问为什么转换需要 `num_heads`、`num_kv_heads` 和 `dim` 等参数。**Conversion** 是在 HF 的分组多头层（grouped multihead layers）与 torchtune 的独立层之间切换所必需的。
- **torchtune 的 Checkpointers**：torchtune 的 **FullModelHFCheckpointer** 会自动将 checkpoint 转换为 HF 格式。分享了关于 **checkpointer 如何确保** 与各种工具的兼容性以及处理不同格式的细节。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://wandb.ai/lmn07r/torchtune/reports/Untitled-Report--Vmlldzo4NTM2NDMw?accessToken=2yedg0bvpgy3fuoaec70tzdm0mqdklzj6bf66kavth4ygoh2ag6klda4tr75mw8t">Untitled Report</a>: 通过性能指标、预测和超参数的交互式图表发布您的模型见解。由 Lemon R 使用 Weights &amp; Biases 制作</li><li><a href="https://wandb.ai/lmn07r/torchtune/workspace?nw=nwuserlemon07r">lmn07r</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://pytorch.org/torchtune/main/deep_dives/checkpointer.html">Checkpointing in torchtune &mdash; torchtune main documentation</a>: 暂无描述</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/models/convert_weights.py#L162">torchtune/torchtune/models/convert_weights.py at main · pytorch/torchtune</a>: 一个用于 LLM Fine-tuning 的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html#generation)">End-to-End Workflow with torchtune &mdash; TorchTune  documentation</a>: 暂无描述</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html#run-evaluation-using-eleutherai-s-eval-harness)">End-to-End Workflow with torchtune &mdash; TorchTune  documentation</a>: 暂无描述</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/models/convert_weights.py#L162.">torchtune/torchtune/models/convert_weights.py at main · pytorch/torchtune</a>: 一个用于 LLM Fine-tuning 的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/blob/f91c16d270e5e3ff32fdb32ccf286d05c03dfa66/src/transformers/models/llama/modeling_llama.py#L262">transformers/src/transformers/models/llama/modeling_llama.py at f91c16d270e5e3ff32fdb32ccf286d05c03dfa66 · huggingface/transformers</a>: 🤗 Transformers: 适用于 PyTorch, TensorFlow, 和 JAX 的最先进机器学习库。 - huggingface/transformers
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1257899648480182373)** (14 条消息🔥): 

> - `使用 Stockfish 数据训练 LLM`
> - `在 LLM 中使用 Stockfish 等工具进行推理`
> - `GitHub notebook 代码`
> - `国际象棋策略与 LLM`
> - `Cohere API 工具` 


- **结合 LLM 与 Stockfish 以获得更好的规划能力**：用户提出了关于使用 **Stockfish 数据** 来提高 **LLM reasoning** 和规划能力，或开发一个快速的**国际象棋引擎**的问题。
- **使用国际象棋数据微调 LLM**：一位成员分享了他们使用国际象棋数据 **fine-tuning LLMs** 的经验，强调这需要显著的过拟合，而这可能会带来问题。关于这种方法的有效性和实用性存在争议。
- **在 Cohere 平台中使用国际象棋工具**：一个有趣的观点建议 **LLMs** 应该使用像 **Stockfish** 这样的工具来获得更好的国际象棋理解结果，而不是直接在国际象棋数据上进行训练。



**提及的链接**: <a href="https://github.com/">GitHub: Let’s build from here</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做出贡献，管理您的 Git 仓库，像专家一样审查代码，跟踪 bug 和功能...

  

---

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1257811755141173379)** (5 messages): 

> - `创建一个 Cohere Slack 机器人`
> - `关于 Slack 请求处理的讨论`
> - `缺乏关于 Slack 集成的文档`
> - `提议分享脚本并创建文档` 


- ****Cohere Slack 机器人简化工作区交互****：一名成员分享了他们为提高工作区便利性和可访问性而创建的 **Cohere Slack 机器人**。
- ****Slack 的效率要求对机器人创建提出挑战****：由于 Slack 要求在 **3 秒**内完成请求，这展示了所使用的 **Cohere 模型** 令人印象深刻的响应速度。
   - *"我使用 Cloudflare Worker 来处理请求"*，展示了尽管初始文档复杂，但仍有实际的集成解决方案。
- ****社区热切期待机器人文档****：社区对该机器人的创建表现出极大的热情，寻求复制该过程的指导和文档。
   - *"我将着手编写带有文档的教程，并将其发布到我的一个域名上"*，表明未来将为感兴趣的成员提供资源。


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1257850552675471441)** (18 messages🔥): 

> - `Kyutai Moshi - 实时音频 LLM`
> - `各种 Open Interpreter 兼容项目`
> - `为 Open Interpreter 修改游戏的经验`
> - `Open Interpreter Labs 的 Pull request`
> - `Mike Bird 和 blurryboi 关于 Kyutai Moshi 的讨论` 


- **Kyutai Moshi 发布实时音频 LLM**：**Kyutai Moshi** 发布了首个实时 **Audio LLM**，几乎没有延迟，但带有一些机械感，该项目将会开源。[在线演示已开放](https://www.moshi.chat/?queue_id=talktomoshi)。
   - *Mikebirdtech* 指出它**非常快，甚至快得有点过头**，如果停顿时间太长，它会打断说话者。
- **Techfren 建议的 Open Interpreter 项目**：列出了几个与 Open Interpreter 兼容的项目，如 **Open interpreter, taxyai, clickolas cage, self-operating computer, pywinassistant, GPT computer assistant**。它们可能需要一些配置，但是很有前景的选择。
- **寻求 Open Interpreter 的游戏模组开发经验**：**Nonadjective.eth_55058** 表示有兴趣修改游戏以兼容 Open Interpreter，并向有类似经验的人寻求建议。他们愿意接受笨重的界面来创建概念验证（PoC）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.moshi.chat/?queue_id=talktomoshi">moshi.chat</a>：未找到描述</li><li><a href="https://github.com/openinterpreterlabs">Open Interpreter Labs </a>：Open Interpreter 实验室与实验（不直接隶属于 OI）- Open Interpreter Labs </li><li><a href="https://x.com/giffmana/status/1808482848808010149?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>：Kyutai Moshi - 首个实时 Audio LLM。基本上没有延迟 - LLM 甚至几次打断了说话者。它实际上有点急于快速回答。:) 全部都将开源。质量...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/)** (1 messages): 

johnlenflure: 难道没有办法将 01 集成到眼镜中吗？
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1257774597852495972)** (1 messages): 

> - `Trainer 中的加权交叉熵 (Weighted cross entropy)` 


- ****发现加权交叉熵部分****：一名成员注意到了 Trainer 中**整个加权交叉熵部分**。他们表示将进一步研究。
- ****未记录其他主题****：在给定的消息历史中没有讨论其他重要主题。仅提到了 **Trainer 中的加权交叉熵**。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1257965086111170580)** (6 messages): 

> - `LoRA 与 QLoRA 量化的区别`
> - `LoRA 中 8-bit 量化的解释`
> - `QLoRA 在微调大模型中的效率` 


- **LoRA 量化为 8-bit**：正如讨论中用户所澄清的，**LoRA** 应用 8-bit 量化，而 **QLoRA** 则更进一步使用 4-bit 量化。
   - 对话强调了量化不仅仅是矩阵分解，并引用了 [QLoRA 论文](https://arxiv.org/abs/2305.14314) 进行全面解释。
- **QLoRA 促进高效微调**：一位成员分享道，[QLoRA 论文](https://arxiv.org/abs/2305.14314) 展示了 QLoRA 如何实现在单个 48GB GPU 上微调 65B 参数模型，且性能接近全量 16-bit 微调。
   - 根据讨论内容的用户的说法，*该论文引入了 4-bit NormalFloat (NF4) 和双重量化（double quantization）等创新技术，在不牺牲性能的情况下最大限度地减少内存使用。*



**提及的链接**：<a href="https://arxiv.org/abs/2305.14314">QLoRA: Efficient Finetuning of Quantized LLMs</a>：我们提出了 QLoRA，这是一种高效的微调方法，可以显著减少内存使用，足以在单个 48GB GPU 上微调 65B 参数模型，同时保持全量 16-bit 微调的任务性能。QLo...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1258113678390067371)** (2 messages): 

> - `Google Colab 上的 torch.cuda.OutOfMemoryError`
> - `Axolotl 运行问题`
> - `GPU 显存分配`
> - `VRAM 需求` 


- **Google Colab 中的 CUDA 显存困扰**：一位成员在 Google Colab 上运行 **axolotl** 时，尝试分配 172.00 MiB GPU 显存时遇到了 **torch.cuda.OutOfMemoryError**。
- **Axolotl 需要更多 VRAM**：针对 CUDA 显存错误，另一位成员建议需要更多 **VRAM** 来避免此类问题。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1258031514638225408)** (5 messages): 

> - `量化及其对模型性能的影响`
> - `LoRA 与 QLoRA 的具体配置差异`
> - `8-bit 量化带来的内存占用和推理速度提升` 


- **8-bit 量化详解**：**量化**将模型精度从 32-bit 或 16-bit 降低到 8-bit (int8)，显著减少了内存占用并加快了推理速度。`lora.yaml` 中的 `load_in_8bit` 选项启用了这一量化过程，以便在有限的硬件上部署大型模型。
- **LoRA 与 QLoRA 的区别**：虽然 **LoRA** 仅专注于参数高效微调，但 **QLoRA** 将低秩自适应（low-rank adaptation）与量化相结合。在 QLoRA 配置中包含 `load_in_8bit` 表示使用了 8-bit 量化，正如在各种示例文件（`qlora.yml`）中所见。



**提及的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=02e7bdf5-d8ec-486f-8697-c89ff466de3b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。

  

---

### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1257787172396859494)** (11 messages🔥): 

> - `AI Town 的 Docker 端口`
> - `使用 WSL 在 Windows 设置 AI Town 的 GitHub 页面`
> - `AI Town Docker 端口的 API 通信问题`
> - `通过 Docker 为 AI Town 自动下载 Convex`
> - `测试 AI Town 的 Docker 集成` 


- ****AI Town 的 Docker 请求****：一名成员强调了为 AI Town 提供 **Docker image** 的必要性，认为其“非常棒”且有用，并鼓励向主仓库提交。
- ****使用 WSL 在 Windows 上设置 AI Town 的 GitHub 指南****：一名成员分享了一个使用 **WSL** 在 Windows 上设置 AI Town 的 [GitHub 页面](https://github.com/Ikkitsuna/AI-Town-Windows-Setup-WSL-method)，并建议提交 PR 将其合并到主仓库中。
- ****AI Town Docker 端口中的 API 问题****：在开发 **Docker port** 时，一名成员指出：“除了一个问题外，我可以毫无问题地运行 AI Town：**Ollama 和其他 API 通信**”，并承诺一旦找到解决方案就会更新。
- ****Docker 中 Convex 下载的自动化****：一名成员正在完成最后的调整，以便 **Convex** 可以通过 Docker 自动下载，旨在简化用户体验，并计划在 UTC+4 晚上 8 点左右上线。
- ****请求协助 Docker 集成测试****：在提交 PR 之前提出了测试 Docker 集成的请求，但在其 **Legion Go** 上测试成功后，该成员决定继续进行。



**提到的链接**：<a href="https://github.com/Ikkitsuna/AI-Town-Windows-Setup-WSL-method">GitHub - Ikkitsuna/AI-Town-Windows-Setup-WSL-method: Guide for setting up AI Town on Windows using WSL</a>：使用 WSL 在 Windows 上设置 AI Town 的指南。可以通过在 GitHub 上创建账户为 Ikkitsuna/AI-Town-Windows-Setup-WSL-method 的开发做出贡献。

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1257903763998511155)** (3 messages): 

> - `在 Modal 上使用 Gradio 部署 RAG 应用遇到困难`
> - `在 Modal Slack 上发帖寻求帮助` 


- ****RAG 应用部署困境****：一名成员提到在 Modal 上部署使用 **Gradio** 的 **RAG app** 时遇到问题，尽管它在本地运行良好且已部署在 Huggingface Spaces 上。他们已经尝试了所有选项，正在寻找资源以找出问题所在。
- ****Modal Slack 寻求救援****：另一名成员建议在 [Modal Slack](https://modal.com/slack) 上发布该问题以获得进一步帮助。原成员表示收到了提醒并打算这样做。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[paige_when_finetune](https://discord.com/channels/1238365980128706560/1242224662142779530/)** (1 messages): 

shamik_53759: 是的，现在已经上线了。谢谢！
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1257990993446436926)** (1 messages): 

> - `用于数据分片并禁用模型分片的 DeepSpeed 配置`
> - `协助 DeepSpeed 配置`
> - `对 DeepSpeed 设置的困惑` 


- **处理分片的 DeepSpeed 配置**：一名成员对于选择合适的 **DeepSpeed configuration** 以启用 **data sharding** 并禁用 **model sharding** 表示困惑。
- **请求协助 DeepSpeed 设置**：请求协助选择正确的 **DeepSpeed settings**，以便在禁用模型分片的同时启用数据分片。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1257930601730674719)** (1 messages): 

> - `共享私有代码部署`
> - `在多个平台上运行代码`
> - `Hugging Face 私有部署的限制` 


- **私有代码部署的困扰**：一名用户表示在使用 Hugging Face 私有空间与同事共享私有代码时遇到困难，并指出私有部署不支持 **sharing=True**。
- **在 Modal 上运行代码的挑战**：在 **Modal** 上运行代码时遇到问题，一名用户提到了他们的困境，并对私有代码共享的替代解决方案表示感兴趣。


  

---

### **LLM Perf Enthusiasts AI ▷ #[eval](https://discord.com/channels/1168579740391710851/1168986849784635553/1258055110236442729)** (1 条消息): 

> - `评估法律合同审查中的 LLM 准确性`
> - `Screens 工具达到 97.5% 的准确率`
> - `法律领域评估 LLM 的方法论`
> - `不同 LLM 和方法对法律任务中 AI 准确性的影响` 


- ****Screens 报告评估法律领域的 LLM 准确性****：Screens 的新[评估报告](https://www.screens.ai/blog/screens-accuracy-evaluation-report)讨论了在法律领域（特别是合同审查）中，将 LLM 评估视为传统 ML 分类问题的方法。他们声称其系统具有 **97.5% 的准确率**，并强调了其在 playbook 执行和工作流路由中的潜在用途。
- ****准确性挑战与方法论探索****：该报告详细阐述了客观评估长篇自由文本回答的挑战，并提出了一种使用分类标准来评估 LLM 性能的方法。这种方法可以显著辅助谈判、修订（redlining）和摘要等法律任务。



**提及的链接**：<a href="https://www.screens.ai/blog/screens-accuracy-evaluation-report">Screens Accuracy Evaluation Report</a>：在合同审查任务中评估大语言模型 (LLMs) 的准确性对于理解该领域的可靠性至关重要。然而，在评估长篇、自由...时，客观性是一个挑战。

  

---


### **LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1258054107688865882)** (1 条消息): 

> - `` 


- **寻求简单的 Prompt 调优工具**：**Evan_04487** 正在寻找一种用户友好的模板 Prompt 调优工具，专为设计师和产品经理等非技术利益相关者量身定制。他们需要一个托管的免费增值（freemium）产品，能够运行模板化 Prompt 的变体并手动检查响应。
- **对免费增值托管工具的需求**：**Evan_04487** 明确表示偏好一种支持 Prompt 调优的免费增值托管工具，能够处理几十个变量。他们提到，对于高风险任务已有更强大的自助服务基础设施，但对于低风险任务需要更简单的工具。


  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/)** (1 条消息): 

derekpwillis: https://thescoop.org/archives/2024/06/22/all-foreign-gifts-around-us/index.html
  

---



---



---



---



{% else %}


> 为了邮件展示，完整的频道细分已被截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}