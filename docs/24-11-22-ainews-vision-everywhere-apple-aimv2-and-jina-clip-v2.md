---
companies:
- apple
- jina
- allen_ai
date: '2024-11-22T23:31:04.836919Z'
description: '**苹果（Apple）**发布了 **AIMv2**，这是一种采用自回归目标预训练的新型视觉编码器，在 **ImageNet** 上实现了
  **89.5% 的准确率**，并集成了视觉与文本联合目标。**Jina** 推出了 **Jina CLIP v2**，这是一款支持 **89 种语言**和高分辨率图像的多模态嵌入模型，其采用的高效
  **Matryoshka（俄罗斯套娃）嵌入**技术在几乎不损失准确率的情况下将维度降低了 **94%**。**Allen AI** 推出了基于 **Llama
  3.1** 的 **Tülu 3** 模型，包含 **8B 和 70B** 参数版本，推理速度提升了 **2.5 倍**，并通过 SFT、DPO 和 RLVR
  方法进行对齐，性能可与 **Claude 3.5** 和 **Llama 3.1 70B** 竞争。这些进展突显了自回归训练、视觉编码器以及多语言多模态嵌入领域的最新突破。'
id: ed748f64-a55b-415e-b592-306ee274f930
models:
- aimv2-3b
- jina-clip-v2
- tulu-3
- llama-3-1
- claude-3-5
- llama-3-1-70b
original_slug: ainews-vision-everywhere-apple-aimv2-and-jina
people: []
title: 视觉无处不在：Apple AIMv2 与 Jina CLIP v2
topics:
- autoregressive-objectives
- vision
- multilinguality
- multimodality
- image-generation
- model-training
- model-optimization
- reinforcement-learning
- fine-tuning
- model-benchmarking
---

<!-- buttondown-editor-mode: plaintext -->**自回归目标就是你所需要的一切。**

> 2024年11月22日至11月23日的 AI 新闻。我们为你检查了 7 个 Reddit 分区、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 社区（**211** 个频道，**2674** 条消息）。预计为你节省阅读时间（以 200wpm 计算）：**265 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

顺应大家都在转向多模态的大趋势（[Pixtral](https://buttondown.com/ainews/archive/ainews-pixtral-12b-mistral-beats-llama-to/)、[Llama 3.2](https://buttondown.com/ainews/archive/ainews-llama-32-on-device-1b3b-and-multimodal/)、[Pixtral Large](https://buttondown.com/ainews/archive/ainews-pixtral-large-124b-beats-llama-32-90b-with/)），“多模态”（实际上主要是视觉）Embedding 的进步是非常基础且关键的。这使得 Apple 和 Jina 在过去 48 小时内的发布特别受欢迎。

## Apple AIMv2

他们的 [论文](https://huggingface.co/papers/2411.14402)（[GitHub 地址](https://github.com/apple/ml-aim)）详细介绍了一种“大规模视觉编码器预训练的新方法”：**将视觉编码器与多模态解码器配对，该解码器以自回归方式生成原始图像块（image patches）和文本标记（text tokens）**。


![image.png](https://assets.buttondown.email/images/e70d8736-0c01-4982-987a-6e59facca643.png?w=960&fit=max)


这扩展了去年关于使用自回归目标预训练视觉模型的 AIMv1 工作，该工作增加了 T5 风格的 prefix attention 和 token 级别的预测头，成功预训练了一个 7B 的 AIM，在冻结 trunk 的情况下在 ImageNet1k 上达到了 84.0%。

主要的更新是引入了视觉和文本的联合目标，这似乎具有非常好的扩展性（scaling）：


![image.png](https://assets.buttondown.email/images/883673a1-7a06-4ea3-9d8d-8365c81aea3b.png?w=960&fit=max)


**AIMV2-3B 现在在同一基准测试中达到了 89.5% 的准确率** —— 体积更小，但性能更强。定性的视觉效果也非常出色：


![image.png](https://assets.buttondown.email/images/d5624b9c-06c5-4f0f-9ae1-dbcc8444373c.png?w=960&fit=max)


## Jina CLIP v2

虽然 Apple 做了更多基础性的 VQA 研究，但 [Jina 新的 CLIP 后代模型](https://jina.ai/news/jina-clip-v2-multilingual-multimodal-embeddings-for-text-and-images/) 对于多模态 RAG 工作负载来说是立即可用的。Jina 几个月前发布了 [embeddings-v3](https://arxiv.org/abs/2409.10173)，现在正将其文本编码器整合到其 CLIP 产品中：


![image.png](https://assets.buttondown.email/images/450d6215-203b-4363-baf6-ab42a2edd3d7.png?w=960&fit=max)


其标语展示了 Jina 在这次发布中集成了多少尖端特性：“一个 0.9B 的多模态 Embedding 模型，具有 **89 种语言的多语言支持**、**512x512 的高图像分辨率**以及 **Matryoshka 表征**。”

Matryoshka Embedding（套娃嵌入）尤其引人注目：“从 1024 维压缩到 64 维（**减少了 94%**）仅导致 **top-5 准确率下降 8%**，top-1 准确率下降 12.5%，突显了其在极小性能损失下进行高效部署的潜力。”


![image.png](https://assets.buttondown.email/images/05f0a184-be4d-43cd-9a8b-633d3839ca50.png?w=960&fit=max)


---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道总结**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 摘要

> 所有总结均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**1. 前沿 AI 模型发布与进展：Tülu 3、AIMv2 等**

- **[@allen_ai 发布的 Tülu 3 模型](https://twitter.com/reach_vb/status/1859663072576643562)**：Tülu 3 系列基于 **Llama 3.1**，包括 **8B 和 70B** 模型，与 Tulu 2 相比提供 **2.5 倍的推理加速**。它使用 SFT、DPO 和基于 RL 的方法进行对齐，所有资源均公开可用。
  - **[关于 Tülu 3 的讨论](https://twitter.com/JonathanRoss321/status/1859654254178467951)** 强调了它与 **Claude 3.5 和 Llama 3.1 70B** 等其他领先 LLM 的竞争力。此次发布包括对数据集、模型 Checkpoints 和训练代码的公开访问，以便进行实际实验。
  - 强调了 Tülu 模型在 **[有效开放科学](https://twitter.com/mervenoyann/status/1859685875766133170)** 方面的贡献，赞扬了其引入的新技术，如 **带有可验证奖励的强化学习 (RLVR)**。

- **[Apple 的 AIMv2 视觉编码器](https://twitter.com/reach_vb/status/1859868073903423821)**：AIMv2 编码器在多模态基准测试中优于 **CLIP** 和 **SigLIP**。它们在开放词汇目标检测方面表现出色，并且在 **冻结 trunk** 的情况下具有极高的 ImageNet 准确率。
  - AIMv2-3B 使用集成的 Transformers 代码在 **ImageNet 上达到了 89.5%** 的准确率。

- **[Jina-CLIP-v2](https://twitter.com/JinaAI_/status/1859659764281782420)** (由 JinaAI 提供)：一款支持 **89 种语言** 和 **512x512 图像分辨率** 的多模态模型，旨在增强文本与图像的交互。该模型在检索和分类任务中表现出强劲性能。

**2. AI Agents 增强与应用：FLUX 工具、来自 Suno 的洞察**

- **[Black Forest Labs 推出的 FLUX 工具套件](https://twitter.com/togethercompute/status/1859735230619500743)**：新的专用模型为 AI 图像生成提供了更强的控制力。可通过 **@replicate API** 在 anychat 中使用，支持全新的 **Canny**、**Depth** 和 **Redux** 模型。
  - FLUX 工具赋能开发者以更高的精度和定制化能力创作引人入胜的多媒体内容。

- **[Suno 与音乐制作中的 AI](https://twitter.com/sunomusic/status/1859679888254566465)**：Suno 的 v4 版本被用于创意音乐尝试，展示了 AI 在音乐制作中的变革性作用，来自 **Suno 的 Bassel** 带来了全新的 AI 生成作品。
  - 此外，关于将 AI 与 B-box 结合的讨论反映了 Suno 对音乐创作的独特贡献。

**3. AI、科学与社会**

- **[利用 AI 产生科学发现](https://twitter.com/GoogleDeepMind/status/1859660861033574742)**：由 **GoogleDeepMind** 举办的小组讨论聚焦于 AI 正在彻底改变科学方法并辅助发现。主要参与者包括 **Eric Topol**、**Pushmeet**、**Alison Noble** 和 **Fiona Marshall**。
  - **[Baby-AIGS 系统研究](https://twitter.com/omarsar0/status/1859656533489188928)** 通过**证伪和消融研究 (falsification and ablation studies)** 探索 AI 在科学发现中的潜力，重点展示了专注于可执行科学提案的早期研究。

- **[AI 与科学方法讨论](https://twitter.com/GoogleDeepMind/status/1859660861033574742)**：深入探讨 AI 如何重塑科学方法论，多位杰出专家分享了见解。
  - 参与者辩论了 AI 在促进新科学突破中的作用及其与生物医学研究的交集。

**4. AI 伦理、红队测试 (Red Teaming) 与 Bug 修复的进展**

- **[OpenAI 的红队测试增强](https://twitter.com/OpenAI/status/1859667912719728897)**：关于红队测试的白皮书披露了涉及**外部红队人员**和自动化系统的新方法，增强了 AI 安全评估。
  - 这些努力旨在通过在测试中积极引入多样化的人类反馈来提升 AI 的鲁棒性。

- **[MarsCode Agent 在 Bug 修复中的应用](https://twitter.com/omarsar0/status/1859964808789135668)**：字节跳动的 MarsCode Agent 在 **SWE-bench Lite** 基准测试中展示了在自动 Bug 修复方面的显著成功，强调了精确错误定位在解决问题中的重要性。
  - 同时也指出了未来自动化工作流创新的挑战领域。

**5. 企业与工具的协作与创新**

- **[Anthropic 与亚马逊的 40 亿美元合作](https://twitter.com/AnthropicAI/status/1859964653486612585)**：双方建立合作伙伴关系，共同开发专注于 AWS 基础设施的下一代 AI 模型，展示了 AI 开发领域的强强联合。
  - 这一战略投资强调利用亚马逊开发的芯片 (silicon) 来优化训练过程。

- **[LangGraph 语音交互功能](https://twitter.com/LangChainAI/status/1859643185363902719)**：将语音识别功能与 AI Agents 集成，利用 **OpenAI Whisper** 和 **ElevenLabs** 实现无缝语音交互界面。
  - LangGraph 增强了 AI 在现实应用中的适应性，提供了更自然的交互体验。

**6. 迷因、幽默与社会评论**

- **[LLM 基准测试技术张力中的幽默](https://twitter.com/nearcyan/status/1859689461426327650)**：对 AI 模型性能“战争”和基准测试的讽刺性解读，嘲讽了对评估分数的痴迷是片面且具有误导性的。
  - 社区声音对某些基准测试在现实世界 AI 模型性能中的相关性表示怀疑。

- **[对 Elon Musk 创业项目的评论](https://twitter.com/francoisfleuret/status/1859855558083563648)**：以冷嘲热讽的言论审视了在 Musk 等科技巨头拥有的平台上所呈现的“言论自由”叙事，挑战了对开放话语的假设。
  - 对主要科技平台的变化及其对真正自由表达的影响进行了批判性反思。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1：DeepSeek 崛起为领先的中国开源 AI 公司**

- **[Chad Deepseek](https://i.redd.it/nn8pp525df2e1.png)** ([Score: 1486, Comments: 174](https://reddit.com/r/LocalLLaMA/comments/1gx4asf/chad_deepseek/)): **DeepSeek** 开发了一个模型，在仅使用 **18,000 个 GPU** 的情况下，性能达到或超过了 **OpenAI**（使用 **100,000 个 GPU**）的表现。这种效率展示了在 LLM 开发中，模型训练方法和资源利用率方面的显著改进。
  - 社区对包括 **Qwen**、**DeepSeek** 和 **Yi** 在内的**中国开源 AI 公司**表示强烈支持，用户强调了它们在以更少资源（**18K GPU** 对比 **OpenAI 的 100K GPU**）实现同等效果方面的高效性。
  - 关于**模型性能**的讨论集中在实际能力上，用户报告了在数学推理（特别是“-4 的平方”问题）和编程任务中的成功，同时也指出了一些在创造性推理和细微响应方面的局限性。
  - 围绕 AI 模型中的**政治审查**展开了辩论，用户讨论了中国和西方模型如何处理敏感历史话题，以及 **GPU 出口限制**可能如何激励中国公司进行更多的开源开发。


- **[Competition is still going on when i am posting this.. DeepSeek R1 lite has impressed me more than any model releases, qwen 2.5 coder is not capable for these competitions , but deepseek r1 solved 4 of 7 , R1 lite is milestone in open source ai world truly](https://www.reddit.com/gallery/1gx72fp)** ([Score: 41, Comments: 10](https://reddit.com/r/LocalLLaMA/comments/1gx72fp/competition_is_still_going_on_when_i_am_posting/)): **DeepSeek R1 Lite** 在一场正在进行的编程竞赛中表现强劲，解决了 **7 道题中的 4 道**，表现优于 **Qwen 2.5 Coder**。该模型的成功标志着开源 AI 开发的重大进展。
  - 传闻 **DeepSeek R1 Lite** 是一个 **16B 参数模型**，尽管其性能表明它可能更大。考虑到 **OpenAI o1** 的发布时机和当前的 **GPU 短缺**，社区推测它是一个 **16B MoE** 模型。
  - 模型权重尚未公开，但预计将“很快”发布。社区舆论强调，在宣布其为里程碑之前，应等待实际的开源发布。
  - 目前尚未正式发布详细的模型信息或技术规格，这使得性能声明仍处于初步阶段。


**主题 2：创新模型架构：Marco-o1 与 OpenScholar**

- **Marco-o1 from MarcoPolo Alibaba and what it proposes** ([Score: 40, Comments: 0](https://reddit.com/r/LocalLLaMA/comments/1gx13jv/marcoo1_from_marcopolo_alibaba_and_what_it/)): 由 **MarcoPolo Alibaba** 开发的 **Marco-o1** 结合了 **Chain of Thought (CoT)**、**Monte Carlo Tree Search (MCTS)** 和**推理动作**，以解决没有既定解决方案的开放式问题，从而区别于 **OpenAI 的 o1**。该模型整合了这三个组件，以实现逻辑问题解决、最优路径选择和动态细节调整，旨在跨多个领域在写作和推理任务中表现出色，模型可在 [AIDC-AI/Marco-o1](https://huggingface.co/AIDC-AI/Marco-o1) 获取。

- **[OpenScholar: The open-source AI outperforming GPT-4o in scientific research](https://i.redd.it/gwjxi1h81h2e1.jpeg)** ([Score: 98, Comments: 2](https://reddit.com/r/LocalLLaMA/comments/1gxad5a/openscholar_the_opensource_ai_outperforming_gpt4o/)): 由 **Allen Institute for AI** 和**华盛顿大学**开发的 **OpenScholar** 将检索系统与微调后的语言模型相结合，提供有引用支持的研究答案，在真实性和引用准确性方面优于 **GPT-4o**。该系统实现了一个用于输出精炼的**自我反馈推理循环**，并作为[开源模型在 Hugging Face 上发布](https://huggingface.co/OpenScholar/Llama-3.1_OpenScholar-8B)，尽管在开放获取论文的可用性方面存在限制，但它使小型机构和发展中国家的研究人员更容易获得此类工具。
  - 与 **VentureBeat** 的报道相比，**AI2 博客文章**提供了关于 **OpenScholar** 更全面的技术细节，参考链接为 [allenai.org/blog/openscholar](https://allenai.org/blog/openscholar)。


**主题 3：系统提示词与 Tokenizer 优化见解**

- **来自 v0（Vercel 的 AI 组件生成器）的泄露系统提示词 (System Prompts) (100% 真实)** ([Score: 292, Comments: 54](https://reddit.com/r/LocalLLaMA/comments/1gwwyia/leaked_system_prompts_from_v0_vercels_ai/))：一名开发者泄露了 **Vercel V0** 的系统提示词，揭示了该 AI 工具使用 **MDX components**、专用代码块以及带有内部提醒的结构化思考过程来生成 UI 组件。该系统包含处理 **React**、**Node.js**、**Python** 和 **HTML** 代码块的详细规范，重点在于使用 **shadcn/ui library**、**Tailwind CSS** 并保持可访问性标准，正如其 [GitHub repository](https://github.com/2-fly-4-ai/V0-system-prompt/blob/main/v0-system-prompt) 中所记录的那样。
  - 讨论表明，由于其 **XML tag** 结构以及对 **shadcn/ui** 的熟练程度，**V0** 可能使用的是 **Claude/Sonnet** 而非 GPT-4，这参考了 [Anthropic's documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags) 中关于提示词结构化的说明。
  - 多位用户确认该系统使用的是 **closed-source SOTA models** 而非开源模型，完整的提示词长度约为 **16,000 tokens**，并包含动态内容，包括 **NextJS/React documentation**。
  - **V0 system prompts** 的更新版本已泄露并通过 [GitHub](https://github.com/2-fly-4-ai/V0-system-prompt/blob/main/v0-system-prompt(updated%2022-11-2024)) 分享，一些用户注意到它与 **Qwen2.5-Coder-Artifacts** 有相似之处，但专门针对 React 实现进行了优化。


- **[警惕损坏的分词器 (Tokenizers)！在创建 v1.3 RPMax 模型时发现了这一点！](https://huggingface.co/ArliAI/Llama-3.1-70B-ArliAI-RPMax-v1.3)** ([Score: 137, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1gwyuyg/beware_of_broken_tokenizers_learned_of_this_while/))：**Tokenizer issues** 影响了 **RPMax v1.3 models** 的 **model performance**，尽管没有提供关于问题性质或解决方案的具体细节。
  - **RPMax versions** 已从 v1.0 演进到 v1.3，其中 v1.3 实现了 **rsLoRA+ (rank-stabilized low rank adaptation)** 以提高学习效果和输出质量。由于天然无审查，基于 **Mistral-based models** 的模型被证明最有效，而 **Llama 3.1 70B** 实现了最低的损失率。
  - **Huggingface transformers library** 中一个关键的 **tokenizer bug** 会导致分词器文件在修改时大小翻倍，从而影响模型性能。该问题可以使用 **AutoTokenizer.from_pretrained()** 后接 **save_pretrained()** 来重现，这会错误地重新生成 "merges" 部分。
  - **RPMax training approach** 非常规，采用 **single epoch**、低梯度累积和较高的学习率，导致损失曲线不稳定但稳步下降。这种方法旨在防止模型强化特定的角色设定或故事模式。


**主题 4. INTELLECT-1：分布式训练创新**

- **[开源 LLM INTELLECT-1 完成训练](https://i.redd.it/m116ylkv5g2e1.png)** ([Score: 268, Comments: 28](https://reddit.com/r/LocalLLaMA/comments/1gx6qyh/open_source_llm_intellect1_finished_training/))：**INTELLECT-1**，一个 **open source Large Language Model**，已使用 **distributed GPU resources worldwide** 完成了训练阶段。帖子中未提供关于模型架构、训练参数或性能指标的额外背景或技术细节。
  - 该模型跨全球 GPU 资源的 **distributed training approach** 引起了社区的极大兴趣，用户将其与 **protein folding projects** 进行比较，并询问如何贡献自己的 GPU。根据 [their website](https://app.primeintellect.ai/intelligence)，**dataset** 预计将于 **11 月底**发布。
  - 围绕该模型 **open source status** 的讨论引发了辩论，并与现有的开源模型如 **Olmo** 和 **K2-65B** 进行了比较。用户注意到，虽然其他模型分享了脚本和数据集，但 INTELLECT-1 的分布式算力贡献代表了一种独特的方法。
  - 技术观察包括与学习率降低同时出现的 **perplexity and loss bump**（困惑度和损失抖动），这归因于引入了具有不同 token 分布的高质量数据。用户指出，虽然该模型的性能并非出类拔萃，但它代表了一个重要的首次迭代。


## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. Amazon x Anthropic 40 亿美元投资与云合作伙伴关系**

- **正在发生。Amazon X Anthropic。** ([Score: 323, Comments: 103](https://reddit.com/r/ClaudeAI/comments/1gxapvo/its_happening_amazon_x_anthropic/)): 根据 [Anthropic 的公告](https://www.anthropic.com/news/anthropic-trainium)，**Amazon** 向 **Anthropic** 追加 **40 亿美元** 投资，并确立 **AWS** 为其主要的云服务和训练合作伙伴。此次合作重点在于云基础设施以及使用 **AWS Trainium** 进行 AI 模型训练。
  - **AWS** 用户指出，他们已经在生产环境中使用 **Claude** 超过一年，一些人表示 **Amazon Q** 在后台已经使用了 Claude。此次合作加强了现有的关系，其中包括将 Claude 引入 **Alexa** 的计划。
  - 该交易的一个关键益处是解决了 **Claude 的算力限制** 和性能问题，由 **AWS Trainium** 取代 **CUDA** 进行模型训练。从 **Nvidia** 硬件向 Amazon 基础设施的转变预示着未来将有重大的技术变革。
  - 用户强调了对 **rate limits**（速率限制）和服务可靠性的担忧，并猜测可能会有 **Prime 集成**（可能带有广告）。一些人将其与 **Microsoft-OpenAI** 的合作伙伴关系进行比较，指出 **Google**（同样是 Anthropic 的投资者）的投资地位可能面临挑战。


- **厌倦了“需求量过高”** ([Score: 34, Comments: 19](https://reddit.com/r/ClaudeAI/comments/1gxaifq/tired_of_were_experiencing_high_demand/)): **Claude** 的付费服务面临日益严重的容量问题，尽管是付费订阅用户，仍频繁遇到“*We're experiencing High Demand*”（我们正面临高需求）的消息。该帖子批评 **Anthropic** 优先发布新功能而非提升基础设施的可扩展性，表达了对付费客户服务受限的沮丧。
  - 用户报告称 **Claude** 的质量在高需求期间会有所下降，“*Full Response*”模式可能提供的推理能力有限且 Token 消耗更快。
  - 多名用户确认经历了**每日服务中断**，一名用户因 **AI 在处理常规任务时的不可靠性** 导致手动完成效率更高而取消了订阅。
  - 一名用户推测，**军事采购**计算资源可能导致了容量限制，但这尚未得到证实。


**主题 2. GPT-4o 在技术基准测试中的性能退化**

- **[独立评估机构发现新 GPT-4o 模型表现显著下降，例如“GPQA Diamond 从 51% 下降到 39%，MATH 从 78% 下降到 69%”](https://x.com/ArtificialAnlys/status/1859614633654616310)** ([Score: 262, Comments: 53](https://reddit.com/r/OpenAI/comments/1gwz4da/independent_evaluator_finds_the_new_gpt4o_model/)): **GPT-4o** 在技术基准测试中表现下降，**GPQA Diamond** 分数从 **51%** 降至 **39%**，**MATH** 分数从 **78%** 降至 **69%**。技术任务性能的下降表明，与前代模型相比，最新模型的能力可能出现了退化。
  - **GPT-4o** 似乎针对自然语言任务而非技术任务进行了优化，用户注意到尽管基准测试分数较低，但它感觉“**更自然**”。几位用户建议 **OpenAI** 正在有意区分不同模型的能力，**GPT-4o** 专注于写作，而 **O1** 处理技术推理。
  - 用户报告了对当前替代模型的混合体验：**Claude Sonnet** 面临消息限制，**O1-mini** 被描述为啰嗦，而 **O1-preview** 限制为**每周 50 个问题**。一些用户提到 **Gemini experimental 1121** 在解决问题和数学方面表现出潜力。
  - 围绕基准测试方法的讨论也随之出现，用户批评 **LMSYS** 作为性能指标是不够的，并质疑单 Token 数学答案与复杂指令响应的价值。模型技术性能的下降可能反映了有意的权衡，而非单纯的退化。

- **为什么 ChatGPT 变得“懒惰”了？** ([Score: 146, Comments: 134](https://reddit.com/r/ChatGPT/comments/1gwv1xa/why_does_chatgpt_get_lazy/)): 用户反映 **ChatGPT** 提供的回复越来越肤浅且不完整，示例显示 AI 给出的答案简短、不全面，遗漏了 Prompt 中的关键细节，并需要频繁修正。帖子作者指出，**ChatGPT** 在被纠正时会承认错误，但随后继续提供浅薄的回复，质疑这种感知到的性能下降是否存在深层原因。
  - 用户报告在最近的**“创意升级”**后，**GPT-4** 的性能显著下降，有证据显示其在 **STEM 学科**和基础数学方面的表现不如 **GPT-4 mini**，如[对比图](https://preview.redd.it/a7csat6msc2e1.jpeg)所示。
  - 多位用户描述了**上下文保留问题**和**记忆问题**，AI 经常忽略详细的 Prompt 和自定义指令。这种退化似乎与 **OpenAI 的成本削减措施**和减少算力分配有关。
  - 技术用户报告了在**代码生成**、**文档审查**和**详细查询**方面的特定问题，指出回复变得更加通用，缺乏针对性。多人提到需要多次尝试才能获得以前单次回复就能提供的全面答案。


**Theme 3. LTX Video: 新型开源快速视频生成模型**

- **[LTX Video - 带有 ComfyUI 工作流的新型开源视频模型](https://v.redd.it/lhxwo8rivg2e1)** ([Score: 259, Comments: 122](https://reddit.com/r/StableDiffusion/comments/1gx9mv3/ltx_video_new_open_source_video_model_with/)): **LTX Video** 是一款新型**开源视频模型**，集成了 **ComfyUI**，可通过 [Hugging Face](https://huggingface.co/spaces/Lightricks/LTX-Video-Playground) 和 [ComfyUI 示例](https://comfyanonymous.github.io/ComfyUI_examples/ltxv/)获取。该模型通过 **ComfyUI 工作流**提供视频生成能力，为用户提供直接访问视频创作工具的途径。
  - **研究团队**的一名成员确认，**LTX-Video** 可以*实时*生成 **768x512** 分辨率、**24 FPS** 的视频，并计划进行更多改进。该模型在 **3060/12GB** 上运行约需 **1 分钟**，而 **4090** 生成 **10 秒视频**需要 **1:48s**。
  - 用户对该模型的表现评价褒贬不一，特别是 **img2video** 功能出现了故障。研究团队承认结果对 **Prompt** 高度敏感，并在其 [GitHub 页面](https://github.com/Lightricks/ltx-video)上提供了详细的示例 Prompt。
  - 该模型现已集成到最新的 **ComfyUI 更新**中，支持包括 **Text2Video**、**Image2Video** 和 **Video2Video** 在内的多种模式。用户需要遵循特定的 Prompt 结构，将动作描述置于 Prompt 前部以获得最佳效果。


- **[LTX-Video 极速运行 - 尽管有 RAM 卸载且仅 12 GB VRAM，1-1.5 分钟内生成 153 帧](https://v.redd.it/7v7shiot5h2e1)** ([Score: 95, Comments: 30](https://reddit.com/r/StableDiffusion/comments/1gxaz7s/ltxvideo_is_lightning_fast_153_frames_in_115/)): **LTX-Video** 展示了高速视频生成能力，在 **12GB VRAM** 限制和 RAM 卸载的情况下，于 **1-1.5 分钟**内生成了 **153 帧**。这一性能指标显示了在视频生成任务中对消费级硬件的高效利用。
  - **LTX-Video** 在消费级硬件上运行效率极高，用户确认尽管有 **18GB VRAM** 的需求，但通过 **RAM 卸载**在 **12GB 4070Ti** 上也能成功运行。安装指南可在 [ComfyUI 博客](https://blog.comfy.org/ltxv-day-1-comfyui/)查看。
  - 用户讨论了未来的潜力，提到了即将推出的 **32GB VRAM** 消费级显卡，并将现状与早期《玩具总动员》时代的怀疑论进行了对比。预计该技术将在 **2-3 年**内取得重大进展。
  - 当前版本 (**0.9**) 被描述为对 Prompt 敏感，后续有改进计划，同时部分用户对输出质量存在争议。原始输出是在没有插帧的情况下生成的。

**Theme 4. 中国 AI 模型作为潜在竞争对手崭露头角**

- **[有人深入研究过中国 AI 吗？](https://i.redd.it/owld1dpmqe2e1.png)** ([Score: 62, Comments: 128](https://reddit.com/r/ChatGPT/comments/1gx2mxv/has_anyone_explored_chinese_ai_in_depth/)): 包括 **DeepSeek**、**ChatGLM** 和 **Ernie Bot** 在内的 **Chinese AI models** 提供免费访问，并能生成高质量的回复，在某些领域足以与 **ChatGPT-4** 竞争。帖子作者指出，尽管这些模型能力出色，但社区对它们的讨论却非常有限。
  - 用户对 **Chinese Communist Party (CCP)** 背景下的 **data privacy** 和 **censorship** 表示强烈担忧，多位评论者提到了监控和信息控制的风险。得分最高的评论主要集中在这些信任和隐私问题上。
  - 讨论强调了未来的潜在情景，包括 **AI development** 中“**East versus West**”的分歧，以及各国之间可能出现的竞争性 **singularities**。几位用户指出，这可能会导致兼容性和竞争方面的挑战。
  - 评论指出，西方 AI 模型（**ChatGPT**、**Claude**、**Gemini**）的 **market awareness** 和 **first-mover advantage** 是其占据主导地位的关键因素，而非技术能力是主要差异点。

---

# AI Discord 摘要

> 由 O1-preview 对摘要进行的摘要总结

**主题 1. AI 军备竞赛：新模型与突破**

- [**INTELLECT-1：全球首个去中心化训练模型**](https://x.com/PrimeIntellect/status/1859923050092994738)：**Prime Intellect** 宣布完成 **INTELLECT-1** 的训练，这是首个通过横跨美国、欧洲和亚洲的去中心化努力训练而成的 **10B 模型**。**开源版本**将在约一周内发布，标志着协作式 AI 开发的一个里程碑。
- [**阿里巴巴发布 Marco-o1：ChatGPT o1 的开源替代方案**](https://huggingface.co/AIDC-AI/Marco-o1)：**AlibabaGroup** 发布了 **Marco-o1**，这是一款采用 **Apache 2 许可证**的模型，旨在通过**思维链 (CoT)** 微调和**蒙特卡洛树搜索 (MCTS)** 解决复杂问题。研究员 **Xin Dong** 和 **Yonggan Fu** 旨在增强模糊领域的推理能力。
- [**Lightricks 的 LTX Video 模型瞬间生成 5 秒视频**](https://x.com/ltxstudio/status/1859964100203430280?s=46)：**Lightricks** 推出了 **LTX Video 模型**，在高性能硬件上仅需 4 秒即可生成 5 秒视频。该模型已开源并可通过 API 调用，推向了快速视频生成的极限。

**主题 2. 十亿美元级动作：Anthropic 与亚马逊握手言和**

- [**Anthropic 获亚马逊 40 亿美元投资，AWS 成为其核心伙伴**](http://anthropic.com/news/anthropic-amazon-trainium)：**Anthropic** 扩大了与 **AWS** 的合作，获得了来自**亚马逊**高达 **40 亿美元**的投资。AWS 现已成为 Anthropic 的主要云服务和训练合作伙伴，利用 **AWS Trainium** 为其最大的模型提供动力。
- **Cerebras 称其 Llama 3.1 部署速度夺冠**：**Cerebras** 吹嘘其运行 **Llama 3.1 405B** 的速度惊人，将自己定位为大语言模型部署的领导者，并激起了 AI 硬件竞争。

**主题 3. AI 被指控：OpenAI 在诉讼中删除证据**

- [**糟糕！OpenAI 在诉讼期间“意外”删除数据**](https://storage.courtlistener.com/recap/gov.uscourts.nysd.612697/gov.uscourts.nysd.612697.328.0.pdf)：律师指控 **OpenAI** 在与《纽约时报》和《每日新闻》的版权诉讼中，在进行了 **150 小时**的搜索后抹除了数据。这引发了法律纠纷中数据处理的严重担忧。
- [**CamelAI 在进行百万级 Agent 模拟后账号消失**](https://github.com/camel-ai/oasis)：**CamelAIOrg** 的 **OpenAI** 账号被封禁，可能与其涉及 100 万个 **Agent** 的 **OASIS 社交模拟项目**有关。尽管已进行沟通，但他们已等待 **5 天**未获回复，使社区陷入困境。

**主题 4. AI 工具变得更智能：增强开发与工作流**

- [**Unsloth 更新大幅降低 VRAM 占用，新增视觉微调功能**](https://x.com/danielhanchen/status/1859672815693414853)：最新的 **Unsloth** 更新将 **VRAM** 效率提升了 **30-70%**，并为 **Llama 3.2 Vision** 等模型引入了视觉微调。它还支持在免费的 **16GB Colab** 中进行 **Pixtral 微调**，使先进 AI 更加触手可及。
- [**LM Studio 讨论多 GPU 魔法与 GPU 对决**](https://lmstudio.ai/)：**LM Studio** 用户讨论了平衡多 GPU 推理，并对比了 **RTX 4070 Ti** 和 **Radeon RX 7900 XT** 等 GPU。虽然功耗有所不同，但他们发现性能差异微乎其微，引发了关于 AI 任务最佳硬件的辩论。
- [**Aider 用户应对量化怪癖与基准测试困惑**](https://aider.chat/2024/11/21/quantization.html)：**Aider** 社区深入探讨了不同量化方法如何影响模型性能。他们注意到 **Qwen 2.5 Coder** 在不同供应商之间表现出不一致的结果，强调了关注量化细节的必要性。

**主题 5. AI 艺术与创意：拥有（幽默感的）机器**

- [**AI 艺术图灵测试让所有人困惑（且觉得有趣）**](https://maxread.substack.com/p/people-prefer-ai-art-because-people)：最近的 **AI 艺术图灵测试**让参与者感到困惑，难以区分 AI 生成的艺术与人类作品。讨论围绕测试的有效性以及 AI 在艺术中不断演变的角色展开。
- **语音克隆故障将有声书变成了意外的音乐剧**：用户在尝试使用**语音克隆**制作有声书时遇到了意想不到的故障，导致 AI 以歌唱的方式回答。这些“美丽的意外”为有声书制作增添了有趣的转折。
- **ChatGPT 的喜剧追求（再次）落空**：尽管有所进步，像 ChatGPT 这样的 **AI 模型**在幽默感方面仍然挣扎，经常讲出平淡无奇的笑话。用户注意到，尝试幽默或 **ASCII 艺术**往往以乱码告终，凸显了 AI 喜剧技能仍有提升空间。

---

# 第一部分：高层级 Discord 摘要

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Test-Time Training 提升 ARC 性能**：最近关于 **Test-Time Training (TTT)** 的实验在 [Abstraction and Reasoning Corpus (ARC)](https://arxiv.org/abs/2411.07279) 上相比基础模型实现了高达 **6倍** 的准确率提升。
   - 关键因素包括在相似任务上的初始微调、辅助任务格式化以及针对每个实例的训练，展示了 TTT 在增强推理能力方面的潜力。
- **Wave Network 引入复数 Token 表示**：**Wave Network** 利用 **复数向量 (complex vectors)** 进行 Token 表示，分离全局和局部语义，从而在 [AG News 分类任务](https://arxiv.org/abs/2411.02674) 中获得了极高的准确率。
   - 每个 Token 与全局语义向量的数值比例建立了一种与整体序列范数的新型关系，增强了模型对输入上下文的理解。
- **关于可学习位置嵌入的争论**：关于 **Mamba** 等模型中 **可学习位置嵌入 (learnable positional embeddings)** 的讨论，强调了其与传统嵌入相比在输入依赖性方面的有效性。
   - 有人对其在约束较少的情况下的表现表示担忧，并建议使用 **Yarn** 或 **Alibi** 等替代方案以获得更好的灵活性。
- **RNN 展示出分布外外推能力**：**RNN** 在算法任务上展示了分布外 (out-of-distribution) 的外推能力，一些人建议可以将思维链 (chain of thought) 应用于 **线性模型 (linear models)**。
   - 然而，对于像 ARC 这样的复杂任务，由于 RNN 固有的表示限制，**TTT** 可能比上下文学习 (**ICL**) 更有益。
- **Muon 正交化技术的见解**：**Muon** 的实现采用了 **momentum** 以及在 momentum 更新后的 **正交化 (orthogonalization)**，这可能会影响其有效性。
   - 讨论强调了足够的 Batch Size 对于有效正交化的重要性，特别是在处理 **低秩矩阵 (low-rank matrices)** 时。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 更新提升 VRAM 效率**：最新的 [Unsloth 更新](https://huggingface.co/deepseek-ai/Janus-1.3B/blob/main/config.json) 为 **Llama 3.2 Vision** 等模型引入了 **视觉微调 (vision finetuning)**，将 VRAM 使用效率提升了 **30-70%**，并增加了在免费的 **16GB Colab** 环境中对 **Pixtral 微调** 的支持。
   - 此外，该更新还包括将模型合并为 **16bit** 以简化推理，并为视觉模型提供 **长上下文支持**，显著提高了可用性。
- **Mistral 模型表现优于同类模型**：用户报告称 **Mistral** 模型在微调方面表现出色，与 **Llama** 和 **Qwen** 模型相比，展示了强大的 **提示词遵循能力 (prompt adherence)** 和卓越的 **准确率**。
   - 尽管表现优异，但对于 **Qwen** 的有效性仍存在一些怀疑，有报告称其在特定应用中会出现 **乱码输出 (gibberish outputs)**。
- **微调与推理挑战**：社区成员在 **微调模型** 时面临困难，例如无法加载微调后的模型进行推理，以及在输出文件夹中遇到 **BF16** 和 **Q4 量化** 等多个模型版本。
   - 在 **推理** 过程中，会出现 `AttributeError` 和 `WebServerErrors` 等错误，特别是在使用 'Mistral-Nemo-Instruct-2407-bnb-4bit' 等模型时，这促使人们建议更换模型路径并验证与 Hugging Face 端点的兼容性。
- **Tokenization 与预训练指导**：有报告称存在 **Tokenization** 问题，包括在 **Hindi** 等数据集训练期间与列长度不匹配相关的错误，以及评估阶段的空预测。
   - 对于 **持续预训练 (continued pretraining)**，用户讨论了使用 Unsloth 提供范围之外的模型，并被鼓励寻求社区支持或在 GitHub 上针对不支持的模型提出兼容性请求。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **平衡多 GPU 推理**：用户讨论了在 [LM Studio](https://lmstudio.ai/) 中实现 **多 GPU 性能** 的可行性，特别是关于跨多个 GPU 的推理和模型分布，并指出负载均衡可能会使 **VRAM 分配** 变得复杂。
   - 讨论中提出了关于是配对不同的 GPU 还是选择更强大的单个 GPU 以获得更好的整体性能的疑虑。
- **比较 RTX 4070 Ti 和 Radeon RX 7900 XT**：社区比较了 **RTX 4070 Ti**、**Radeon RX 7900 XT** 和 **GeForce RTX 4080** 在 **1440p 和 4K** 分辨率下的性能，指出虽然功耗有所不同，但性能差异通常很小。
   - 成员们讨论了平衡 **功耗** 和 **性能** 的问题，建议为了获得最佳结果应优先选择更高质量的模型。
- **微调模型对比 RAG 策略**：成员们辩论了 **微调模型** 与使用 **RAG**（检索增强生成）策略的优劣，共识是微调可以使模型专门用于特定任务，而 RAG 提供了更大的灵活性。
   - 微调的例子包括针对 **C#** 编程语言适配模型，但同时也提出了关于敏感公司数据安全影响的担忧。
- **LLM 基准测试中的 AMD GPU**：**AMD GPU** 可以通过 ROCm 或 Vulkan 运行 **LLM**；然而，讨论中也提到了关于驱动更新影响性能的持续担忧。
   - 有人指出 ROCm 主要在 **Linux** 或 **WSL** 上运行，限制了部分用户的可用性。
- **期待 5090 显卡的发布**：成员们对即将发布的 **5090 显卡** 表示期待，同时也对供应情况和定价感到担忧。
   - 讨论内容包括关税对硬件价格的影响，以及在预期价格上涨前确保设备供应的必要性。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **IntelliBricks 工具包简化 AI 应用开发**：**IntelliBricks** 是一个旨在简化 AI 驱动应用程序开发的开源工具包，其特点是使用 `msgspec.Struct` 处理结构化输出。该项目目前正在开发中，欢迎在其 [GitHub 仓库](https://github.com/arthurbrenno/intellibricks) 进行贡献。
   - 鼓励开发者参与贡献以增强其功能，营造一个构建高效 AI 应用的协作环境。
- **FLUX.1 Tools 增强图像编辑能力**：**FLUX.1 Tools** 的发布引入了一套用于编辑和修改图像的工具，包括 **FLUX.1 Fill** 和 **FLUX.1 Depth** 等模型，由 [Black Forest Labs](https://blackforestlabs.ai/flux-1-tools/) 发布。
   - 这些模型提高了文本生成图像任务的 **可控性 (steerability)**，允许用户尝试开放获取的功能并增强其图像生成工作流。
- **去中心化训练完成 INTELLECT-1 模型**：**Prime Intellect** 宣布完成 **INTELLECT-1** 的训练，这是一个通过遍布美国、欧洲和亚洲的去中心化努力训练出的 10B 模型。预计在大约一周内发布完整的开源版本。
   - 这一里程碑突显了去中心化训练方法的有效性，更多细节可在 [Prime Intellect 的推文](https://x.com/PrimeIntellect/status/1859923050092994738) 中找到。
- **Cybertron v4 UNA-MGS 模型登顶 LLM 基准测试**：**cybertron-v4-qw7B-UNAMGS** 模型已重新推出，在无污染的情况下实现了 **7-8B LLM 排名第一**，并增强了推理能力，正如其 [Hugging Face 页面](https://huggingface.co/fblgit/cybertron-v4-qw7B-UNAMGS) 所示。
   - 该模型利用了 `MGS` 和 `UNA` 等独特技术，展示了卓越的基准测试性能，吸引了 AI 工程社区的关注。
- **Cerebras 以高速 Llama 3.1 部署领先**：**Cerebras** 正在通过以惊人的速度运行 **Llama 3.1 405B** 来引领 LLM 性能，将自己定位为大语言模型部署的领导者。
   - 这一进步强调了 Cerebras 致力于优化 AI 模型性能的承诺，在快速发展的大语言模型领域提供了竞争优势。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic 从 AWS 获得 40 亿美元融资**：**Anthropic** 额外获得了来自 **Amazon** 的 **40 亿美元**投资，指定 AWS 为其主要的云和训练合作伙伴，通过 [AWS Trainium](https://www.aboutamazon.com/news/aws/amazon-invests-additional-4-billion-anthropic-ai) 增强 AI 模型训练。
   - 正如其[官方公告](https://www.anthropic.com/news/anthropic-amazon-trainium)中所述，此次合作旨在利用 AWS 基础设施开发和部署 **Anthropic** 最大的基础模型。
- **AI 艺术图灵测试引发褒贬不一的反应**：最近的 **AI Art Turing Test** 引发了讨论，[此分析](https://maxread.substack.com/p/people-prefer-ai-art-because-people)中强调了参与者难以区分 AI 和人类创作的艺术品。
   - 成员们有兴趣邀请艺术修复专家来评估该测试，以更好地衡量其有效性。
- **Lightricks 发布开源 LTX Video 模型**：**Lightricks** 推出了 **LTX Video 模型**，能够在高性能硬件上仅用 4 秒生成 5 秒的视频，可通过 [APIs](https://x.com/ltxstudio/status/1859964100203430280?s=46) 获取。
   - 讨论集中在利用 LTX Video 模型时，如何平衡本地处理能力与云端相关的成本。
- **斯坦福大学发布 AI 活力排名工具**：斯坦福大学推出了 **AI Vibrancy Rankings Tool**，该工具根据可定制的 AI 发展指标对各国进行评估，允许用户调整指标权重以匹配其观点。
   - 该工具因其在提供全球 AI 进展洞察方面的灵活性而受到赞誉。
- **基于 LLM 的需求分析受到关注**：**LLM-powered requirements analysis** 正成为一个关键话题，成员们强调了其在自动化复杂问题理解和建模过程中的有效性。
   - 对话指出 LLM 在简化分析工作流方面具有巨大潜力，并参考了 [DDD starter modeling process](https://github.com/ddd-crew/ddd-starter-modelling-process)。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **语音克隆故障增强了有声书效果**：成员们讨论了用于 **audiobook** 改编的 **voice cloning** 技术，注意到意想不到的声音和故障有时会产生令人惊喜的增强效果（如唱歌）。 [#ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1309279933918482562)
   - 一位用户分享了使用各种 **voice models** 的经验，强调了语音克隆如何在对话中创造出怪异的效果。
- **ChatGPT 与 Airtable 和 Notion 集成**：探索了 **ChatGPT** 与 **Airtable** 和 **Notion** 等工具的集成能力，旨在增强这些应用程序中的 Prompt 编写。 [#ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1309279933918482562)
   - 成员们分享了他们改进 Prompt 编写的目标，寻求更个性化和有效的交互。
- **Copilot 的图像生成能力引发猜测**：对 **Copilot** 的图像生成能力产生了好奇，猜测其来源是未发布的 **DALL-E** 模型还是名为 **Sora** 的新程序。 [#ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1309279933918482562)
   - 对不同 AI 工具生成的图像进行了比较，指出受其他模型影响的质量差异。
- **GPT 在使用 Dall-E 时的词汇限制问题**：一位成员表达了挫败感，称他们的 **GPT** 在使用 **Dall-E** 生成约 **10 张图像**后，往往会忘记特定的词汇限制。 [#gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1309348677667520512)
   - 他们正在寻求技巧来维持角色描述，并避免在生成内容中出现不想要的词语。
- **探索 Dall-E 之外的替代方案和免费图像模型**：成员们讨论了 **Dall-E** 的替代方案，如带有 **ComfyUI** 的 **Stable Diffusion** 和 **Flux models**，认为它们可能更好地处理特定的词汇限制。 [#gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1309348677667520512)
   - 他们建议查看 **YouTube** 上的最新教程，以确保获得保持角色一致性的更新方法。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen 2.5 Coder 的性能差异**：用户观察到 **Qwen 2.5 Coder** 模型在不同供应商之间的性能表现不一，其中 **Hyperbolic** 的得分为 **47.4%**，而排行榜上的得分为 **71.4%**。
   - 社区讨论强调了**量化 (quantization)** 的影响，指出 **BF16** 和其他变体产生了不同的性能结果。
- **Aider 基准测试方法的更新**：Aider 针对 **Qwen 2.5 Coder 32B** 的排行榜现在通过 **GLHF** 使用来自 **HuggingFace** 的权重，从而提高了基准测试的准确性。
   - 用户对不同托管平台导致的评分差异表示担忧，并对模型质量可能存在的差异提出质疑。
- **与 Qwen 模型的直接 API 集成**：**Aider** 框架现在可以直接访问 **Qwen** 模型，而无需依赖 **OpenRouter**，从而简化了使用流程。
   - 此次更新旨在通过减少对第三方服务的依赖来提升用户体验，同时保持模型性能。
- **引入 Uithub 作为 GitHub 的替代方案**：用户推荐将 [**Uithub**](http://uithub.com) 作为 GitHub 的替代方案，只需将 'G' 改为 'U'，即可轻松将仓库内容复制到 LLM 中。
   - 来自 *Nick Dobos* 和 *Ian Nuttall* 等成员的反馈强调了 Uithub 获取完整仓库上下文的能力，从而增强了开发工作流。
- **亚马逊向 Anthropic 追加 40 亿美元投资**：[**Amazon**](https://www-cnbc-com.cdn.ampproject.org/c/s/www.cnbc.com/amp/2024/11/22/amazon-to-invest-another-4-billion-in-anthropic-openais-biggest-rival.html) 宣布向 **Anthropic** 追加 **40 亿美元**投资，加剧了 AI 开发领域的竞争态势。
   - 此举引发了关于在企业投资不断增加的情况下，AI 项目的可持续性和创新速度的讨论。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Marco-o1 发布，作为 ChatGPT o1 的替代方案**：**AlibabaGroup** 发布了 [**Marco-o1**](https://huggingface.co/AIDC-AI/Marco-o1)，这是一个采用 **Apache 2** 许可证的 **ChatGPT o1** 模型替代方案，旨在通过**思维链 (CoT)** 微调和**蒙特卡洛树搜索 (MCTS)** 解决**复杂问题**。
   - **Xin Dong**、**Yonggan Fu** 和 **Jan Kautz** 等研究人员正领导 *Marco-o1* 的开发，旨在增强在**标准模糊**和**奖励量化**困难的领域中的推理能力。
- **带有 Few-shot Prompting 的 Agent 翻译工作流**：该 **Agent 翻译工作流**采用 **few-shot prompting** 和**迭代反馈循环**，而非传统的微调，使 **LLM** 能够对其翻译进行**批判**和**改进**，从而提高**灵活性**和**定制化**程度。
   - 通过利用**迭代反馈**，该工作流避免了训练开销，从而提高了翻译任务的**生产力**。
- **阿里巴巴的 AWS 合作与 40 亿美元投资**：**AnthropicAI** 宣布与 **AWS** 建立合作伙伴关系，包括来自 **Amazon** 的新一轮 **40 亿美元**投资，将 **AWS** 定位为其主要的**云**和**训练**合作伙伴，正如[这条推文](https://x.com/AnthropicAI/status/1859964653486612585)所分享的。
   - **Teknium** 在一则[推文](https://x.com/teknium1/status/1859997785220947990?s=46)中强调，按目前的速度维持**预训练扩展 (pretraining scaling)** 在未来两年将需要 **2000 亿美元**，并对持续进步的可行性提出了质疑。
- **用于 LLM 和多模型聊天界面的 Open WebUI**：成员们讨论了用于 **LLM 托管聊天体验**的**图形用户界面 (GUI)**，倾向于使用 [**Open WebUI**](https://docs.openwebui.com/) 和 **LibreChat** 等工具，其中 **Open WebUI** 因其**用户友好界面**而受到广泛**青睐**。
   - 社区分享了 **Open WebUI** 功能的**动画演示**，强调了其对各种 **LLM runners** 的支持以及高效处理多个**模型交互**的能力。
- **使用 Axolotl 的示例默认配置微调数据集**：一位寻求**微调模型**和**创建数据集**的成员对**高昂的试错成本**表示担忧，随后收到了使用 [**Axolotl 的示例默认配置**](https://link.to.examples)的建议，这些配置被认为对训练运行非常**有效**。
   - 使用 **Axolotl** 的**示例默认配置**可以简化**微调过程**，降低成本并提高数据集创建工作的**效能**。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 3.5 Haiku ID 变更**：**Claude 3.5 Haiku** 模型已重命名，其 ID 中使用 **点 (dot)** 代替了 **连字符 (dash)**，这改变了其可用性。新的模型 ID 可在 [Claude 3.5 Haiku](https://openrouter.ai/anthropic/claude-3.5-haiku) 和 [Claude 3.5 Haiku 20241022](https://openrouter.ai/anthropic/claude-3.5-haiku-20241022) 获取，尽管访问可能受限。
   - 寻求这些模型的用户可以通过 [Discord](https://discord.gg/fVyRaUDgxW) 频道申请访问权限，而之前的 ID 仍可正常使用。
- **Gemini 模型配额问题**：用户在通过 OpenRouter 访问 **Gemini Experimental 1121** 模型时遇到配额错误。建议直接连接到 **Google Gemini** 以获得更可靠的访问。
   - 这些配额限制影响了依赖免费版本的用户，从而引发了对替代连接方法的建议。
- **OpenRouter API Token 统计差异**：有报告称 **Qwen 2.5 72B Turbo** 模型不像其他提供商那样通过 OpenRouter API 返回 Token 计数。然而，OpenRouter 页面上的 **活动报告 (activity reports)** 能准确显示 Token 使用情况。
   - 这种不一致表明 OpenRouter 在处理特定模型的 Token 计数时可能存在问题。
- **欧洲 OpenRouter 积分的税费问题**：一位用户询问为什么在欧洲购买 **OpenRouter credits** 不包含增值税 (VAT)，而不像 **OpenAI** 或 **Anthropic** 的服务那样。回复澄清说，VAT 计算是用户的责任，并计划在未来实施自动税费计算。
   - 缺乏 VAT 包含引起了欧洲用户的关注，凸显了简化税务流程的需求。
- **自定义 Provider Key 的访问权限**：多位用户请求访问 **custom provider keys**，反复的呼吁强调了对该功能的浓厚兴趣。用户如 *sportswook420* 和 *vneqisntreal* 强调了这一需求。
   - 社区对自定义 Provider Key 的热情表明了对增强功能的渴望，尽管访问程序仍未明确。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini AI 对比 ChatGPT**：用户报告称 **Gemini AI** 在几次交互后经常停止响应，引发了对其与 **ChatGPT** 相比可靠性的质疑。
   - 讨论强调了性能差异，一些成员发现 **ChatGPT** 在长时间对话中表现更一致。
- **Perplexity 浏览器扩展**：有一场关于 Safari 浏览器的 [Perplexity extensions](https://perplexity.ai/extensions) 可用性的对话，包括新增的搜索引擎和已停止使用的摘要工具。
   - 成员们分享了针对非 Safari 浏览器的替代解决方案，并提供了管理现有扩展的技巧。
- **非编程人员的 AI 易用性**：提出了一项基于层级的学习系统方案，旨在通过结构化的项目和教程课程，使 AI 技术对非编程人员更易获取。
   - 该系统旨在提供分步指导，在面向社区的框架内促进技能发展。
- **AI 中的数字孪生 (Digital Twins)**：探讨了 [Digital twins](https://www.perplexity.ai/search/what-is-digital-twin-rOr00s1vSPmK1_EOv5ty7w)，重点关注其在各行业监控和优化现实实体中的应用。
   - 用户对数字孪生如何增强模拟能力和运营效率表现出浓厚兴趣。
- **AI 对 Grammarly 的影响**：辩论了 AI 对 **Grammarly** 的影响，[此讨论](https://www.perplexity.ai/search/did-ai-impact-grammarly-at-all-bgRE6pmeQlmZda6x3YUVDQ) 审视了 AI 技术进展在写作工具中的整合。
   - 参与者考虑了引入 AI 以增强 **Grammarly** 功能的利弊。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SDXL Lightning 支持图像提示词 (Image Prompts)**：一位用户询问如何通过 Python 在 **SDXL Lightning** 中利用图像提示词，寻求将照片集成到特定语境中的指导。
   - 另一位用户确认了可行性，并建议通过私信交流更多信息。
- **为 12GB VRAM 优化 WebUI**：讨论集中在增强 **webui.bat** 参数以提升 **12GB VRAM** 的性能，建议包含 '--no-half-vae'。
   - 用户一致认为，此调整足以优化性能且不会引入进一步的复杂问题。
- **将企业照片转换为 Pixar 风格**：有人请求将企业照片转换为 **Pixar 风格** 图像的方法，需要在短时间内处理约十张肖像。
   - 成员们讨论了可行性，指出可能没有免费服务，并建议对图像生成模型进行微调 (Fine-tuning)。
- **探索使用 Cogvideo 的视频微调服务**：用户对视频微调表现出兴趣，并询问了可用的服务器或服务，参考了 **Cogvideo 模型**。
   - 有人强调，虽然 **Cogvideo** 在视频生成方面很突出，但其他特定的微调版本可能更符合用户需求。
- **下载 Stable Diffusion 及其使用场景**：一位新用户寻求在 PC 上下载 **Stable Diffusion** 最简单、最快的方法，并询问了相关的使用场景。
   - 另一位用户请求帮助使用 **Stable Diffusion** 创建特定图像，同时绕过内容过滤器，这表明需要更宽松的软件选项。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **简化 Workflow 中的 Function Calling**：用户讨论了简化 **Workflow 中的 Function Calling**，建议使用预构建的 Agent（如 `FunctionCallingAgent`）来实现 **自动化函数调用**，从而避免编写样板代码。
   - 一位成员指出，虽然样板代码提供了更多控制权，但使用预构建的 Agent 可以 **简化流程**。
- **LlamaIndex 安全合规性**：**LlamaIndex** 确认其符合 **SOC2** 标准，并详细说明了通过 **LlamaParse** 和 **LlamaCloud** 安全处理原始文档的细节。
   - **LlamaParse** 对文件进行 48 小时加密，而 **LlamaCloud** 则对数据进行分块 (Chunking) 并安全存储。
- **LlamaIndex 中的 Ollama 包问题**：用户报告了 **LlamaIndex** 中 **Ollama** 包的问题，称最新版本中的一个 Bug 导致聊天响应时出错。
   - 建议降级到 **Ollama** 版本 **0.3.3**，一些成员确认此操作解决了他们的问题，参考 [Pull Request #17036](https://github.com/run-llama/llama_index/pull/17036)。
- **Hugging Face Embedding 兼容性问题**：用户担心来自 **Hugging Face** 上 **CODE-BERT** 模型的 Embedding 与 **LlamaIndex** 预期的格式不一致。
   - 用户建议 [在 GitHub 上提交 Issue](https://github.com/run-llama/llama_index/pull/17036) 以解决处理模型响应时可能出现的匹配问题。
- **LlamaParse 解析挑战**：尽管有排除冗余信息的详细指令，**LlamaParse** 在返回信息时仍包含页眉和参考文献等冗余内容。
   - 一位成员询问：*'还有其他人遇到这个问题吗？'*，并分享了针对科学论文的全面解析指令，以保持内容的逻辑流并排除致谢和参考文献等非核心元素。

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 实现了 Retrieval-Augmented Generation**：正如社区成员所讨论的，NotebookLM 现在利用 **Retrieval-Augmented Generation** 来增强 **response accuracy** 和 **citation tracking**。
   - 这一实现旨在为进行广泛查询会话的用户提供更可靠、更可验证的输出。
- **Podcastfy.ai 成为 NotebookLM API 的替代方案**：一位成员推荐了 [Podcastfy.ai](https://www.podcastfy.ai) 作为 **NotebookLM's podcast API** 的开源替代方案，引发了关于功能对比的讨论。
   - 用户正在评估 **Podcastfy.ai** 在播客创建和管理方面与现有 NotebookLM 选项的优劣。
- **NotebookLM 对 Producer Studio 的需求增长**：一位用户强调了他们对 **NotebookLM** 的 [Producer Studio 功能请求](https://discord.com/channels/1124402182171672732/1300797611015671818/1300797611015671818)，主张增强播客制作能力。
   - 社区成员对先进的制作工具表现出兴趣，以简化平台内的播客创作流程。
- **NotebookLM 寻求多语言音频翻译支持**：用户请求将 **NotebookLM** 音频输出翻译成**德语**和**意大利语**等语言的能力，突显了对更广泛**多语言支持**的需求。
   - 这一需求强调了该平台迎合更广泛、全球化工程师群体的潜力。
- **NotebookLM 播客创建限制已明确**：用户已确认 **NotebookLM** 内部存在**每个账户 100 个播客的限制**，以及每日 **20 个播客创建**的可能上限。
   - 这一限制促使用户仔细管理其播客库存，因为删除旧播客会重置其创建额度。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 的数据删除诉讼**：[The New York Times](https://techcrunch.com/2024/11/20/openai-accidentally-deleted-potential-evidence-in-ny-times-copyright-lawsuit/) 和 [Daily News](https://techcrunch.com/2024/11/20/openai-accidentally-deleted-potential-evidence-in-ny-times-copyright-lawsuit/) 的律师正在起诉 **OpenAI**，指控其在经过 **150 多个小时**的搜索后意外删除了潜在证据。
   - 11 月 14 日，一台虚拟主机上的数据被擦除，正如[法庭信函](https://storage.courtlistener.com/recap/gov.uscourts.nysd.612697/gov.uscourts.nysd.612697.328.0.pdf)中所述，这可能会影响案件。
- **Prime Intellect 的 INTELLECT-1 去中心化 10B 模型**：**Prime Intellect** 宣布完成 **INTELLECT-1**，这是首个跨越多个大洲的 **10B model** 去中心化训练。
   - 预计在一周内发布完整的[开源版本](https://x.com/PrimeIntellect/status/1859923050092994738)，邀请各方协作构建开源 AGI。
- **Anthropic 与 AWS 达成 40 亿美元合作伙伴关系**：**Anthropic** 通过亚马逊 **40 亿美元的投资**扩大了与 **AWS** 的合作，确立 AWS 为其主要的云和训练合作伙伴。
   - 此次合作旨在增强 **Anthropic's AI technologies**，详见其[官方新闻稿](http://anthropic.com/news/anthropic-amazon-trainium)。
- **Tulu 3 的 On-policy DPO 分析**：关于 **Tulu 3** 论文的讨论质疑了所描述的 **DPO method** 是否由于训练期间模型策略的演变而真正属于 on-policy。
   - 成员们辩论认为，[第 8.1 节](https://arxiv.org/pdf/2410.03717v1)中提到的 *online DPO* 通过在每个训练步骤为奖励模型采样完成结果，更符合 on-policy 推理。
- **CamelAIOrg 的 OASIS 社会模拟项目**：**CamelAIOrg** 面临 **OpenAI** 的账号封禁，可能与其最近涉及 100 万个 **Agent** 的 **OASIS social simulation project** 有关，详见其 [GitHub page](https://github.com/camel-ai/oasis)。
   - 尽管已寻求帮助，但 **20 多名社区成员**在等待 5 天后仍未收到 API keys 的回复。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 优化 AMD GPU**：在一段 [YouTube 视频](https://youtu.be/Lbm08twNTAQ?si=lR-4YeLhVxWcPw8R) 中，Lei Zhang 和 Lixun Zhang 讨论了针对 **AMD GPU** 的 **Triton** 优化，重点关注 **L2 缓存 swizzle** 和 **内存访问效率** 等技术。
   - 他们还探讨了通过 **MLIR 分析** 来增强 Triton kernel，参与者强调了这些优化在提高 GPU 整体性能方面的重要性。
- **FlashAttention 优化**：成员们讨论了 **FlashAttention** 的进展，包括基于 QK^T 分块（tiling block）内 **局部和（local sums）** 的 **全局指数和（global exponential sum）** 近似技术。
   - 重点在于理解 **局部** 与 **全局指数和** 之间的关系，以有效地优化 Attention 机制。
- **LLM 剪枝技术**：一位成员请求关于 **大语言模型 (LLM)** 最新的 **剪枝和模型效率论文**，并引用了 **What Matters in Transformers** 论文及其 **数据依赖（data-dependent）** 技术。
   - 讨论强调了对 **非数据依赖技术** 的需求，以提高各种工业应用中的模型效率。
- **GPT-2 训练方法**：分享了一个 [GitHub Gist](https://gist.github.com/charlesfrye/5f299d80ba2b2ae4ec81d672b4c3246f)，详细介绍了如何 **在五分钟内免费训练 GPT-2**，包括一个简化流程的辅助函数。
   - 此外，还有关于将 **GPT-2 训练** 功能集成到 **Discord bot** 中的讨论，旨在改善 AI 相关任务的用户体验。
- **NPU 加速方案**：一位成员询问支持 **NPU 加速** 的库或运行时，并提到 **Executorch** 为 **Qualcomm NPU** 提供了一些支持。
   - 讨论旨在确定其他能有效利用 **NPU 加速** 的框架，并鼓励社区提供建议。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API 增强对话编辑**：用户请求一个与 **Cohere API** 兼容的前端，其中包含一个 **编辑按钮**，用于在不重启对话的情况下修改聊天历史。
   - 会议澄清了编辑应无缝集成到聊天历史中，并指出目前 **Cohere 官网** 的聊天和 playground-chat 页面均缺少 **编辑选项**。
- **SQL Agent 与 Langchain 集成**：**SQL Agent** 项目在 [Cohere 文档](https://docs.cohere.com/page/sql-agent-cohere-langchain) 中展示，并获得了社区的积极反馈。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **SDXL 基准测试在更新后变慢**：在应用更新 **#7644** 后，**SDXL** 在 **CI tinybox green** 上不再转换为 half 类型，导致基准测试速度下降超过 **2 倍**。
   - 成员们质疑这次类型转换的更改是否是为了解决之前错误的转换实现。
- **对 SDXL 最新回归问题的担忧**：更新 **#7644** 中移除了 **SDXL** 的 half 转换，引发了社区对 **回归问题（regression）** 的担忧。
   - 用户正在寻求澄清，即基准测试性能的下降是否意味着 **SDXL** 能力的倒退。
- **提议中间转换策略转型**：提出了一项建议，根据 **输入 dtype** 而非设备来确定中间转换（intermediate casting），主张采用 **纯函数式（pure functional）** 方法。
   - 该建议包括采用类似于 Stable Diffusion 中 **fp16** 的方法，以提高模型和输入转换的效率。
- **Tinygrad 移除自定义 Kernel 函数**：**Tinygrad** 仓库的最新版本中已移除 **自定义 kernel 函数**。
   - **George Hotz** 建议使用替代方法来实现预期结果，而不损害抽象层。
- **通过 YouTube 介绍 Tinygrad**：通过 [YouTube 链接](https://youtu.be/0ncx4H0YmK0) 分享了 **Tinygrad 入门介绍**，以帮助初学者。
   - 该资源旨在帮助新用户更有效地理解 **Tinygrad** 的基础知识。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo-Python 互操作性指日可待**：[Mojo 路线图](https://link.to/roadmap) 包括允许 Python 开发者导入 Mojo 包并调用 Mojo 函数，旨在提高跨语言互操作性。
   - 社区成员正在积极开发此功能，对于不追求极致性能的用户，目前已有初步方法可用。
- **Mojo 中的异步事件循环配置**：尽管最初支持异步分配的数据结构，但现在 Mojo 中的异步操作需要设置事件循环以有效管理状态机。
   - 未来计划允许在不需要时编译掉异步运行时，从而优化性能。
- **Mojo-Python 多线程集成的权宜之计**：一位用户分享了一种方法，通过队列让 Mojo 和 Python 进行通信，利用 Python 的多线程实现异步交互。
   - 虽然在某些情况下有效，但有人认为这种方法对于简单需求过于复杂，主张提供官方解决方案。
- **推进 Mojo 特性以进行速度优化**：一位成员强调 Mojo 的主要用途是作为 **C/C++/Rust** 的类 Python 替代方案，强调其在加速缓慢进程中的作用。
   - 他们强调了基础特性的重要性，如参数化 traits 和 Rust 风格的 enums，其优先级高于基础 Mojo 类。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **HF Transfer 加速模型下载**：在 **torchtune** 中加入 [HF transfer](https://github.com/pytorch/torchtune/pull/2046) 显著缩短了模型下载时间，**llama 8b** 的下载时间从 **2分12秒降至32秒**。
   - 用户可以通过运行 `pip install hf_transfer` 并为非 nightly 版本添加标志 `HF_HUB_ENABLE_HF_TRANSFER=1` 来启用它。此外，有用户报告在家庭网络连接上通过 HF transfer 每次下载一个文件，下载速度超过了 **1GB/s**。
- **Anthropic 论文质疑 AI 评估的一致性**：Anthropic 最近的一篇 [研究论文](https://arxiv.org/abs/2411.00640) 讨论了 AI 模型评估的可靠性，质疑性能差异是真实的，还是源于问题选择中的随机运气。
   - 该研究鼓励 AI 研究社区采用更严谨的统计报告方法，而一些社区成员对强调误差棒（error bars）表示怀疑，引发了关于提升评估标准的持续讨论。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Hackathon 转为完全线上**：即将举行的 Hackathon 将 **100% 在线** 进行，允许参与者从任何地点加入，无需物理场地。
   - 这一决定解决了物流问题，并确保所有团队成员都能更广泛地参与。
- **简化团队注册**：团队注册现在会向第一个字段中输入的邮箱发送 **确认邮件**，确保至少有一名团队成员收到确认。
   - 这一改进简化了注册流程，使其更加用户友好且高效。
- **Percy Liang 的演讲**：**Percy Liang** 本周的演讲获得了成员们的积极反馈。
   - 参与者强调了会议期间交付内容的清晰度和深度。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Desktop App 发布时间表仍不确定**：一位新成员在加入 [waiting list](https://discord.com/channels/1146610656779440188/1147665339266650133/1309297211875790911) 后询问了 **Desktop App** 的发布计划，但未提供具体的发布日期，导致时间表仍不确定。
   - 这种不确定性表明开发工作仍在进行中，目前尚未公布 **Desktop App** 的明确时间线。
- **Exponent 演示凸显 Windsurf 的有效性**：一名成员分享了他们使用 **Exponent** 进行演示的经验，并提到正在继续对其功能进行实验。
   - 成员对 **Windsurf** 给予了积极反馈，强调了其在演示过程中的**有效性**。
- **社区探索开源版 Devin**：讨论中提到了社区成员一直在探索的一个**开源版本的 Devin**，尽管并非所有人都尝试过。
   - 这反映了利用**开源工具**进行项目实验的持续兴趣。
- **克服 O1 在 Linux 上的安装挑战**：一名成员报告了在其 **Linux** 系统上安装 **O1** 的困难，并正在寻求解决方案。
   - *他们正在寻求有关安装问题的潜在解决方案或变通方法的建议。* 此外，讨论还涉及了将 **Groq API** 或其他免费 API 与 **Linux** 上的 **O1** 集成的可行性。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **VLM 增强发票处理能力**：一名成员正在探索将 **VLM** 用于一个**高风险的发票处理项目**，并寻求关于 **DSPy** 如何增强其针对专门子任务的 Prompt 指导。
   - 提到 **DSPy** 最近增加了对 **VLM**（特别是 **Qwen**）的支持。
- **DSPy 集成 Qwen 以支持 VLM**：**DSPy** 已添加对 **Qwen**（一种特定的 **VLM**）的支持，以增强其处理专门任务的能力。
   - 此次集成旨在改进高风险发票处理等项目的 Prompt Engineering。
- **在视觉分析项目上测试 DSPy**：一名成员建议在 **VLM** 上尝试 **DSPy**，分享了他们在**视觉分析项目**中的成功经验，并指出 **CoT** 模块在图像输入下运行良好。
   - 他们尚未测试优化器（optimizers），表明还有更多探索空间。
- **利用 DSPy 简化项目开发**：另一名成员强调从简单任务开始，然后逐渐增加项目的复杂性，强化了 **DSPy** 易于上手的理念。
   - *“如果你从简单开始并随着时间的推移增加复杂性，这并不难！”* 传达了一种鼓励实验的情绪。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **INTELLECT-1 完成去中心化训练**：**Prime Intellect** 宣布完成 **INTELLECT-1**，这标志着首次跨越多个大洲对 **10B 模型** 进行的**去中心化训练**。来自 [Prime Intellect 的推文](https://x.com/PrimeIntellect/status/1859923050092994738) 确认，目前正在与 **arcee_ai** 进行训练后处理，完整的**开源发布**计划在大约一周后进行。
   - **INTELLECT-1** 实现了去中心化 AI 训练的一个重要里程碑，实现了横跨**美国**、**欧洲**和**亚洲**的协作开发。即将到来的**开源发布**预计将促进更广泛的社区参与 **AGI** 研究。
- **训练能力提升 10 倍**：**Prime Intellect** 声称，与之前的模型相比，**INTELLECT-1** 的去中心化训练能力提升了 **10 倍**。这一进步突显了该项目在分布式环境中的可扩展性和效率。
   - **10 倍的增强**使 **INTELLECT-1** 成为去中心化 AI 训练领域的领先模型，邀请 AI 工程师为构建更强大的**开源 AGI** 框架做出贡献。
- **对 Axolotl 微调的期待**：人们对 **INTELLECT-1** 发布后 **Axolotl** 的 **fine-tuning** 能力充满期待。参与者渴望评估该系统在去中心化训练取得进展的情况下如何管理 **finetuning**。
   - **Axolotl** 的微调功能与 **INTELLECT-1** 的去中心化框架的集成预计将增强模型的适应性和性能，从而使技术工程社区受益。

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Neural Turing Machines 狂热**：一位成员在过去几天里一直在探索 **Neural Turing Machines**。
   - 他们非常希望与对此感兴趣的其他人**交流想法**。
- **Differentiable Neural Computers 深度探索**：一位成员正在深入研究 **Differentiable Neural Computers** 以获得进一步的见解。
   - 他们正在寻找志同道合的热心人士，就这些技术相关的想法和见解进行合作。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期没有动态，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期没有动态，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期没有动态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期没有动态，请告知我们，我们将将其移除。

---

# 第 2 部分：各频道详细摘要与链接

{% if medium == 'web' %}

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1309594454461518004)** (1 条消息): 

> `重启读书小组, Discord 论坛功能, 每月读书小组, YouTube 录像, 新小组反馈` 

- **使用 Discord 论坛功能重启读书小组**：读书小组频道已通过 Discord 的新**论坛功能 (forum feature)** 重新启动，以增强组织性并减少手动维护。
   - 鼓励组织者在指定论坛中创建主题帖进行持续讨论。
- **开放新读书小组邀请**：任何感兴趣的人都可以围绕 **scaling laws**、**reasoning** 和 **efficient architectures** 等主题创建并管理自己的**读书小组 (reading groups)**。
   - 已设立专门频道供社区成员提出读书小组的想法并接收反馈。
- **展示社区研究成果的每月读书小组**：将组织**每月读书小组**来展示社区成员进行的研究，12 月的第一场会议将聚焦于论文 [Refusal in LLMs is an Affine Function](https://arxiv.org/abs/2411.09003)。
   - 第一场会议的具体日期尚未公布。
- **读书小组会议录像**：鼓励参与者录制读书小组会议，并可选择将其上传到专门为社区设立的 **YouTube 频道**。
   - 目前已有数学读书小组的录像，其他小组也可以提交其录像。
- **归档之前的读书小组频道**：之前的读书小组频道已重命名以进行归档，并将在一个月内降低其显示权重。
   - 此次过渡旨在保留过去的讨论，同时推进以论坛为中心的新结构。

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1309487269819973633)** (15 messages🔥): 

> `N-shot prompting 数据集, 量子建模协助, 旧金山 Pre-NeurIPS 聚会, 强化学习中的 Vector environments, AI agent 开发工具` 


- **探索 N-shot prompting 数据集**：一位成员建议各种 **Q&A 类型的数据集** 可以有效地与 **n-shot prompting** 结合使用。
   - 这种新方法可能会增强模型在训练场景中的性能和适应性。
- **寻求量子建模帮助**：新成员 Marc 介绍了自己，他目前正在研究 **量子模型（quantum models）** 并寻求帮助。
   - 这表明了一个欢迎各领域专业知识的协作环境。
- **计划在 Dolores Park 举行 Pre-NeurIPS 聚会**：一位成员宣布在 Dolores Park 举行 **Pre-NeurIPS 轻松聚会**，邀请参加者 **RSVP 并加入** 野餐。
   - 预计天气晴朗，鼓励参加者在讨论 AI 和其他兴趣爱好时自带食物和饮料。
- **Vector Environments 对 RL 的益处**：一位用户强调 **Vector Environments** 可以显著加速 **强化学习（reinforcement learning）训练**，并提供了参考链接。
   - 这些环境能够对多个子环境进行采样，从而提高训练效率和性能。
- **使用开源工具开发 AI agents**：一位成员表示有兴趣利用一套开源工具创建 **具有自我改进能力的 AI agents**，以进行持续开发。
   - 他们正在探索 **Open Webui** 和 **AnythingLLM** 等各种框架，以便在项目中实现尖端功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lu.ma/fi3edk93">AI Friends @ Dolores Park (pre Neurips gathering) · Luma</a>: 如果你有兴趣请 RSVP！AI 朋友们 - 让我们在 Dolores Park 见面。距离上次在旧金山为 EleutherAI 成员（及朋友）举办的聚会已经很久了 🌁 伴随着……</li><li><a href="https://gymnasium.farama.org/api/vector/">Gymnasium Documentation</a>: 强化学习的标准 API 和一系列多样化的参考环境（原 Gym）</li><li><a href="https://github.com/PufferAI/PufferLib/">GitHub - PufferAI/PufferLib: Simplifying reinforcement learning for complex game environments</a>: 简化复杂游戏环境的强化学习 - PufferAI/PufferLib
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1309277130709794816)** (408 messages🔥🔥🔥): 

> `Test-Time Training (TTT), Wave Network Token Representation, Learnable Positional Embeddings, RNN Extrapolation, Muon Orthogonalization` 


- **Test-Time Training 在 ARC 上表现出潜力**：最近的实验表明，Test-Time Training (TTT) 可以显著提高在 Abstraction and Reasoning Corpus (ARC) 上的推理能力，与基础模型相比，准确率提高了多达 6 倍。
   - 成功的 TTT 关键组件包括对类似任务的初始微调（finetuning）、辅助任务格式化以及针对每个实例的训练。
- **Wave Network 中的创新 Token 表示**：Wave 网络提出了一种使用复数向量（complex vectors）的独特 token 表示方法，在输入文本中分离全局和局部语义，在 AG News 分类任务上实现了高准确率。
   - 该模型中的 token 代表其值与全局语义向量的比率，从而与整体序列范数（sequence norms）建立了一种新颖的关系。
- **关于 Learnable Positional Embeddings 的辩论**：关于 Mamba 等模型中 learnable positional embeddings 与传统 embeddings 相比的有效性存在讨论，有观点认为它们基于输入依赖性（input dependence）能有效工作。
   - 与 Yarn 或 Alibi 等方法相比，人们对其在约束较少的条件下的性能表示担忧。
- **RNN 展现出 OOD 外推能力**：RNN 已被证明在算法任务上能成功进行分布外（out-of-distribution, OOD）外推，一些人建议思维链（chain of thought）甚至可以适配到线性模型中。
   - 然而，有人担心对于更复杂的任务（如 ARC 中的任务），由于学习表示的固有局限性，TTT 可能比 ICL 获益更多。
- **Muon 和正交化（Orthogonalization）见解**：Muon 的实现细节表明它在 momentum 更新后采用 momentum 和正交化，这可能会影响其有效性。
   - 讨论强调了确保足够 batch size 以进行有效正交化的必要性，特别是在低秩矩阵（low-rank matrices）的背景下。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://arxiv.org/abs/2411.13676">Hymba: A Hybrid-head Architecture for Small Language Models</a>: 我们提出了 Hymba，这是一个 Small Language Models 系列，采用混合头并行架构，将 Transformer 注意力机制与 State Space Models (SSMs) 集成，以增强效率...</li><li><a href="https://arxiv.org/abs/2411.07279">The Surprising Effectiveness of Test-Time Training for Abstract Reasoning</a>: Language Models 在其训练分布内的任务上表现出令人印象深刻的性能，但往往难以处理需要复杂推理的新问题。我们研究了 Test-Time Training 的有效性...</li><li><a href="https://arxiv.org/abs/2411.02674">Wave Network: An Ultra-Small Language Model</a>: 我们在一种新的超小型 Language Model：Wave network 中提出了一种创新的 Token 表示和更新方法。具体来说，我们使用复数向量来表示每个 Token，同时编码全局...</li><li><a href="https://arxiv.org/abs/2405.06394">Memory Mosaics</a>: Memory Mosaics 是联想记忆网络，协同工作以实现感兴趣的预测任务。与 Transformers 类似，Memory Mosaics 具有组合能力和 In-context learning...</li><li><a href="https://arxiv.org/abs/2203.06026">The Role of ImageNet Classes in Fréchet Inception Distance</a>: Fréchet Inception Distance (FID) 是数据驱动生成建模中对模型进行排名的主要指标。虽然非常成功，但已知该指标有时与人类判断不一致...</li><li><a href="https://arxiv.org/abs/2309.06979">Auto-Regressive Next-Token Predictors are Universal Learners</a>: Large Language Models 在逻辑和数学推理方面表现出卓越的能力，使它们能够解决复杂的任务。有趣的是，这些能力出现在经过简单训练的网络中...</li><li><a href="https://arxiv.org/abs/2406.06484">Parallelizing Linear Transformers with the Delta Rule over Sequence Length</a>: 具有线性注意力的 Transformers（即 Linear Transformers）和 State-space models 最近被建议作为具有 Softmax 注意力的 Transformers 的可行线性时间替代方案。然而，...</li><li><a href="https://arxiv.org/abs/2102.11174">Linear Transformers Are Secretly Fast Weight Programmers</a>: 我们展示了线性化 Self-attention 机制与 90 年代初的 Fast weight controllers 在形式上的等效性，其中“慢速”神经网络通过梯度下降学习来编写“快速”...</li><li><a href="https://arxiv.org/abs/2407.04620">Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>: Self-attention 在长上下文中表现良好，但具有二次复杂度。现有的 RNN 层具有线性复杂度，但它们在长上下文中的性能受到其隐藏状态表达能力的限制...</li><li><a href="https://arxiv.org/abs/2306.04675">Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models</a>: 我们系统地研究了涵盖语义多样化图像数据集的各种生成模型，以理解和改进用于评估它们的特征提取器和指标。使用最佳...</li><li><a href="https://arxiv.org/abs/2402.18668">Simple linear attention language models balance the recall-throughput tradeoff</a>: 最近的工作表明，基于 Attention 的 Language Models 在召回方面表现出色，即在上下文中将生成内容建立在先前看到的 Token 上的能力。然而，基于 Attention 的模型的效率是...</li><li><a href="https://x.com/BlinkDL_AI/status/1859578512988147889">Tweet from BlinkDL (@BlinkDL_AI)</a>: “世界上最难的数独”由 12M 参数的 RWKV-6 在 4M Token 的 CoT 后解决 🙂 代码和模型：https://github.com/Jellyfish042/Sudoku-RWKV 注意该模型仅使用 ctx8192 训练，所以我...</li><li><a href="https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based">Zoology (Blogpost 2): Simple, Input-Dependent, and Sub-Quadratic Sequence Mixers</a>: 未找到描述</li><li><a href="https://x.com/KoszarskyB/status/1859426854245159282">Tweet from Braden Koszarsky (@KoszarskyB)</a>: 按照 @hi_tysam 的建议在硬块测试中添加了文档掩码，这基本上弥补了差距。在 3000 步时，HellaSwag 的结果波动太大，无法得出任何结论。我预计文档...</li><li><a href="https://x.com/PavloMolchanov/status/1859792527592943891">Tweet from Pavlo Molchanov (@PavloMolchanov)</a>: 分享我们团队关于 Hymba 的最新工作——一种具有混合架构的高效 Small Language Model。技术报告：https://arxiv.org/abs/2411.13676 探索 Mamba 和 Attention 之间的权衡...</li><li><a href="https://x.com/blinkdl_ai/status/1784929516156313678?s=46">Tweet from BlinkDL (@BlinkDL_AI)</a>: 来自社区：RWKV-6 3B 可以状态微调到 99.2% LAMBADA，记住 400k+ Token🧠（仅用于测试容量——它是在测试集上训练的）。方法：查看 https://github.com/BlinkDL/</li>

RWKV-LM  ...</li><li><a href="https://x.com/blinkdl_ai/status/1784496793075744966?s=46">来自 BlinkDL (@BlinkDL_AI) 的推文</a>: RWKV 状态微调（state-tuning）对齐：因为 RWKV 是 100% 的 RNN，我们可以直接微调其 RNN 状态来控制其行为🤯例如，一个经过状态微调的 RWKV-6 "Finch" 1.6B 可以变得很有趣并使用 emojis🐦eve...</li><li><a href="https://openreview.net/forum?id=r8H7xhYPwz">Gated Delta Networks: Improving Mamba2 with Delta Rule</a>: Linear Transformers 由于其推理效率，已成为标准 Transformers 的高效替代方案，在各种任务中都取得了具有竞争力的性能，尽管它们通常...</li><li><a href="https://github.com/ekinakyurek/marc">GitHub - ekinakyurek/marc: &quot;The Surprising Effectiveness of Test-Time Training for Abstract Reasoning&quot; 的公开仓库</a>: &quot;The Surprising Effectiveness of Test-Time Training for Abstract Reasoning&quot; 的公开仓库 - ekinakyurek/marc
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1309263920145367192)** (13 messages🔥): 

> `API 速率限制管理, MCQ 实现, 解析 CLI 参数, 模型评估, tokenizer_backend 的 Bug 修复` 


- **通过延迟管理 API 速率限制**：一位用户询问如何实现时间延迟，以避免在调用 OpenAI 和 Anthropic API 时触发速率限制，建议查询频率为每 1-5 秒一次。
   - 另一位成员建议使用 [Tenacity 库](https://tenacity.readthedocs.io/en/latest/#waiting-before-retrying) 来有效地处理重试逻辑。
- **创建多步 MCQ**：一位成员提议实现一种带有多步问题的多项选择题（MCQ）格式，以评估模型对新信息的适应能力。
   - 他们表示有兴趣根据模型的初始响应来衡量正确答案和错误答案之间的转换。
- **在 CLI 参数中传递 None**：一位用户报告了通过命令行传递 `tokenizer_backend=None` 时的困难，因为它没有被正确解析。
   - 一位成员承认这是一个 Bug，并提到了一个临时解决方案，同时正在准备 Pull Request。
- **关于 CLI Bug 的 GitHub Ticket 讨论**：一位用户询问是否已针对影响 `None` 解析为 tokenizer backend 的 CLI Bug 开设了 Ticket。
   - 另一位成员提供了[相关 GitHub Issue 的链接](https://github.com/EleutherAI/lm-evaluation-harness/pull/2509)，并确认一旦测试通过就会合并修复。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/867413f8677f00f6a817262727cbb041bf36192a/lm_eval/models/openai_completions.py#L15)">EleutherAI/lm-evaluation-harness 中的 lm-evaluation-harness/lm_eval/models/openai_completions.py</a>: 一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2509).">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://tenacity.readthedocs.io/en/latest/#waiting-before-retrying)">Tenacity &mdash; Tenacity 文档</a>: 无描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/867413f8677f00f6a817262727cbb041bf36192a/lm_eval/models/api_models.py#L451)">EleutherAI/lm-evaluation-harness 中的 lm-evaluation-harness/lm_eval/models/api_models.py</a>: 一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1309631924003672105)** (1 messages): 

> `模型类型, 用户偏好` 


- **探索多样化的模型类型**：一位成员指出，根据他们的经验，目前的模型大多属于特定类别。
   - 他们通过询问 *“你还在考虑其他哪些类型？”* 来引发进一步讨论。
- **鼓励更广泛的讨论**：对其他模型类型的询问突显了将对话扩展到熟悉类别之外的兴趣。
   - 这表明了对探索模型开发中新方向和可能性的开放态度。

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1309267851814764626)** (240 条消息🔥🔥): 

> `Unsloth 更新, 模型微调, Mistral 模型性能, 图像生成模型, Qwen 模型局限性` 


- **最近 Unsloth 发布亮点**：最新的 Unsloth 更新引入了针对各种模型（包括 Llama 3.2 Vision）的 Vision finetuning 功能，将 VRAM 占用降低了 30-70%。此外，他们还增加了在免费 16GB Colab 环境中支持 Pixtral 微调的功能，这是一项重大增强。
   - 该版本还包括关于将模型合并为 16bit 以简化推理的更新，以及对视觉模型的长上下文支持，极大地提升了易用性。
- **Mistral 模型的性能**：多位用户发现 Mistral 模型在微调方面表现优于其他模型，并指出其具有强大的 Prompt 遵循能力和良好的准确度。与 Llama 和 Qwen 模型的对比表明，Mistral 在各种任务中的表现通常超出预期。
   - 然而，一些用户对 Qwen 的有效性表示怀疑，理由是在某些应用中会出现乱码输出问题。
- **微调策略**：讨论了关于 Qwen 和 Llama 模型的微调，用户分享了他们的经验和最佳实践。一些人指出，对于分类任务，使用像 BERT 这样的小型高效模型可能比依赖大型 LLM 更有效。
   - 贡献者提到了数据集质量和模型可学习性的重要性，特别是在评估 Mistral 和 Pixtral 选项时。
- **图像生成模型讨论**：用户对图像生成模型产生了兴趣，并推荐了 Janus 等具有多模态能力的模型。由于图像质量原因，几位用户表示相比集成模型，他们更倾向于现有的 Diffusion 模型。
   - 对话强调了在特定应用中使用这些模型的细微差别，以及与有效部署和训练相关的挑战。
- **Unsloth 模型的技术问题**：用户报告了技术问题，包括加载模型时的导入错误以及特定配置的问题。建议通过更新依赖项和调整配置来排除故障，以便高效利用资源。
   - 一位用户专门询问在模型上添加 linear layer 是否会影响优化，从而引发了关于扩展模型能力时最佳实践的进一步讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/Janus-1.3B/blob/main/config.json">config.json · deepseek-ai/Janus-1.3B at main</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/deepseek-ai/Janus-1.3B">Chat With Janus 1.3B - a Hugging Face Space by deepseek-ai</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1859672815693414853">Daniel Han (@danielhanchen) 的推文</a>：Vision finetuning 终于加入 🦥@UnslothAI 了！虽然花了一些时间，但 Llama 3.2 Vision, Pixtral, Qwen2 VL 和所有 Llava 变体现在都可以运行了！ 1. QLoRA / LoRA 速度提升 1.3x 到 2x 2. VRAM 占用减少 30-70% ...</li><li><a href="https://docs.vllm.ai/en/latest/serving/deploying_with_k8s.html">使用 Kubernetes 部署 &#8212; vLLM</a>：未找到描述</li><li><a href="https://tenor.com/view/discord-this-server-is-powered-gif-21305371">Discord This GIF - Discord This Server - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/AIDC-AI/Marco-o1/tree/main">AIDC-AI/Marco-o1 at main</a>：未找到描述</li><li><a href="https://github.com/deepseek-ai/Janus">GitHub - deepseek-ai/Janus: Janus-Series: Unified Multimodal Understanding and Generation Models</a>：Janus 系列：统一多模态理解与生成模型 - deepseek-ai/Janus
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1309287589626380329)** (16 messages🔥): 

> `希腊酸奶饮食、健康优化、GPU 服务选项` 


- **希腊酸奶加入早餐俱乐部**：一位成员在早餐中加入了 **希腊酸奶**，并混合了 **苹果片** 和 **燕麦片 (granola)**，开启了健康的早晨。
   - 该成员 *不是谷物圈 (cereal) 的粉丝*，发现这种组合既 **美味** 又 **健康**，并向他人推荐。
- **追求最佳健康状态**：这位成员正在寻找身体的 **平衡点 (sweet spot)**，他有 **肺部虚弱** 的病史，并普遍渴望更好的健康状况。
   - 他们提到，“*只是记录下来，以防对其他人有帮助*”，并鼓励其他人分享见解。
- **对 Vision 数据集的兴趣**：一位成员提到想尝试 **Vision**，但对准备必要的 **Datasets** 的难度表示担忧。
   - 这一评论反映了那些希望深入研究视觉数据处理的人所面临的共同挑战。
- **寻求 GPU as a Service 推荐**：一位成员询问了关于 **廉价 GPU as a Service** 的选项，或包含在 **Linux Server** VCPU 中的选项，特别是寻找 **24GB GPU 显存**。
   - 成员们请求提供建议或推荐，凸显了对便捷 GPU 解决方案日益增长的兴趣。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1309300843195142215)** (69 messages🔥🔥): 

> `Fine-tuning 模型、Inference 问题、模型兼容性、Continued Pretraining、Tokenization 错误` 


- **Fine-tuning 模型的挑战**：用户在 Fine-tuning 模型时遇到了各种问题，包括无法加载 Fine-tuned 模型进行 Inference，以及对输出文件夹中存在多个模型版本感到困惑。
   - 例如，一位用户不确定为什么同时存在 BF16 和 Q4 Quantization 版本，这被澄清为 Conversion 过程的一部分。
- **Inference 错误与配置**：几位用户在 Inference 过程中遇到了错误，包括 `AttributeError` 和 `WebServerErrors`，特别是在使用特定模型（如 'Mistral-Nemo-Instruct-2407-bnb-4bit'）时。
   - 建议包括尝试用保存的 Checkpoints 替换模型路径，并验证与 Hugging Face 的 Inference Endpoint 的兼容性。
- **报告的 Tokenization 问题**：一位用户报告了在 TinyLlama 上 Fine-tuning 印地语数据集时的问题，具体是在训练期间收到了与列长度不匹配相关的 Tokenization 错误。
   - 另一位在 Evaluation 期间遇到空预测的用户怀疑这可能与计算前的 Preprocessing 阶段有关。
- **Continued Pretraining 指南**：社区成员讨论了使用 Unsloth 提供以外的模型进行 Continued Pretraining 的可能性，并建议针对任何特定模型的问题寻求社区支持。
   - 如果非 Unsloth 模型不受支持，鼓励用户在 GitHub 上提交兼容性请求。
- **学习 Fine-tuning 的新用户**：一位新用户对他们的第一次 Fine-tuning 经历表示困惑，指出他们生成的模型输出不符合预期。
   - 经过澄清，在 Conversion 过程中遇到两个模型文件并不代表他们操作有误。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/continued-pretraining)">未找到标题</a>：未找到描述</li><li><a href="https://github.com/simonlindgren/mistral-unslothify">GitHub - simonlindgren/mistral-unslothify: 为领域数据 Fine-tune 开源 LLM</a>：为领域数据 Fine-tune 开源 LLM。通过在 GitHub 上创建账号来为 simonlindgren/mistral-unslothify 的开发做出贡献。</li><li><a href="https://docs.unsloth.ai/basics/continued-pret">Unsloth 文档</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1K9ZrdwvZRE96qGkCq_e88FgV3MLnymQq?usp=sharing#scrollTo=95_Nn-89DhsL)">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1309275647968804945)** (153 条消息🔥🔥): 

> `多 GPU 处理、GPU 性能对比、LM Studio 安装问题、模型微调 vs. RAG、系统资源需求` 


- **探索多 GPU 推理**：用户讨论了在 LM Studio 中实现 **多 GPU 性能** 的可行性，特别是关于跨多个 GPU 的推理和模型分布。一位成员指出，负载均衡可能会导致复杂化，并可能影响 VRAM 分配。
   - 用户们对是应该配对不同的 GPU，还是选择一个更强大的单 GPU 以获得更好的整体性能提出了疑问。
- **显卡性能见解**：社区对比了多种 GPU，特别是 **RTX 4070 Ti**、**Radeon RX 7900 XT** 和 **GeForce RTX 4080** 在 **1440p 和 4K** 分辨率下的性能指标。据观察，虽然功耗各异，但性能差异通常很小。
   - 成员们讨论了功耗与性能之间的平衡，建议优先选择 **高 Q** 模型以获得最佳效果。
- **LM Studio 安装挑战**：一位用户在 Windows 11 Pro 上安装 LM Studio 时遇到问题，指出文件下载不完整。另一位成员建议检查系统规格，并确认安装必须支持 AVX2。
   - 在多次尝试下载后，该用户成功安装了 Beta 版本，这凸显了主下载链接可能存在服务端问题。
- **理解微调 vs. RAG 的使用**：成员们辩论了 **微调模型 (Fine-tuning)** 与使用 **RAG** (Retrieval-Augmented Generation) 策略的优劣。共识认为，虽然微调可以使模型针对特定任务专门化，但 RAG 提供了更大的灵活性，且在使用专有数据时降低了模型损坏的风险。
   - 微调的例子包括使模型适应 **特定编程语言**（如 C#），但对于公司敏感数据的安全性也提出了担忧。
- **模型加载的资源管理**：一位用户对由于系统资源不足导致模型加载失败表示担忧，引发了调整运行时设置的建议。讨论强调，低 RAM 可能会导致运行挑战，建议切换 GPU 运行时并尝试重启应用程序。
   - 讨论指出，在性能不足的硬件上运行高 VRAM 需求模型可能会导致冻结或崩溃，这表明用户应保持在系统的能力范围之内。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/">LM Studio - 实验本地 LLMs</a>: 在你的电脑上本地运行 Llama, Mistral, Phi-3。</li><li><a href="https://www.techpowerup.com/review/gigabyte-geforce-rtx-4070-ti-super-gaming-oc/33.html">技嘉 GeForce RTX 4070 Ti Super Gaming OC 评测</a>: 技嘉 RTX 4070 Ti Super Gaming OC 名副其实，配备了出厂超频，额定加速频率为 2655 MHz。它采用三槽、三风扇散热器和双 BIOS 以增加灵活性...</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce4-mx-4000.c776">NVIDIA GeForce4 MX 4000 规格</a>: NVIDIA NV18, 250 MHz, 2 Pixel Shaders, 0 Vertex Shaders, 4 TMUs, 2 ROPs, 128 MB DDR, 166 MHz, 64 bit</li><li><a href="https://www.techpowerup.com/review/gigabyte-geforce-rtx-4070-ti-super-gaming-oc/41.html">技嘉 GeForce RTX 4070 Ti Super Gaming OC 评测</a>: 技嘉 RTX 4070 Ti Super Gaming OC 名副其实，配备了出厂超频，额定加速频率为 2655 MHz。它采用三槽、三风扇散热器和双 BIOS 以增加灵活性...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1309262125126123521)** (121 条消息🔥🔥): 

> `eGPU 阵营, 使用 AMD GPU 进行 LLM 基准测试, MacBook 在 AI 任务中的性能, 功耗与 GPU 效率, 即将发布的显卡` 


- **加入 eGPU 阵营**：一位成员分享了他们最近加入 **eGPU 阵营**的经历，并对他们的新配置表示兴奋。
   - 讨论涉及了 Windows 与 Linux 上 eGPU 的性能对比，并对多卡（multicard）设置表现出兴趣。
- **在 AMD GPU 上进行 LLM 基准测试**：AMD GPU 可以通过 ROCm 或 Vulkan 运行 **LLMs**；然而，人们一直担心驱动更新会影响性能。
   - 关于使用 ROCm 的讨论指出，它主要在 Linux 或 WSL 上运行，这限制了部分用户的可用性。
- **MacBook 的 AI 任务能力**：成员们讨论了 **MacBook Pro** 在 AI 任务中的性能，指出与 NVIDIA GPU 相比，它在图像生成方面表现吃力。
   - 有人提到，由于推理速度较慢，在 Mac 上运行大型模型需要耐心。
- **功耗与 GPU 效率**：有人对用于运行大型模型且效率有限的老款显卡的高功耗表示担忧。
   - 建议通过调整专业级 GPU 的功耗限制（power limit）来优化其性能。
- **对即将发布的显卡的期待**：成员们表达了对即将发布的 **5090 显卡**的期待，同时也对供货情况和定价表示担忧。
   - 关税对硬件价格的影响引发了关于在预期涨价前提前锁定设备的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/DavidS95/Smokeless_UMAF">GitHub - DavidS95/Smokeless_UMAF</a>：通过在 GitHub 上创建账户，为 DavidS95/Smokeless_UMAF 的开发做出贡献。</li><li><a href="https://tenor.com/view/showtime-beetlejuice-gif-24573561">Showtime Beetlejuice GIF - Showtime Beetlejuice - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>：多个 NVIDIA GPU 还是 Apple Silicon 用于大语言模型推理？- XiongjieDai/GPU-Benchmarks-on-LLM-Inference</li><li><a href="https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622">NVIDIA GeForce RTX 3090 规格</a>：NVIDIA GA102, 1695 MHz, 10496 Cores, 328 TMUs, 112 ROPs, 24576 MB GDDR6X, 1219 MHz, 384 bit
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1309272578279473283)** (215 条消息🔥🔥): 

> `新增 Mamba2 支持、Faster-Whisper 错误、AI 模型与语言训练、AI 模型的幽默感、AI 浏览器集成` 


- **MLX 现已支持 Mamba2**：一位成员宣布 MLX 已成功集成了对今年 5 月推出的 **Mamba2** 全新架构的支持。
   - 另一位成员询问了此次集成背后的难点，引发了关于各种库支持所需时间的讨论。
- **使用 Faster-Whisper 时面临的挑战**：一位用户报告了在尝试运行 **Faster-Whisper** 时与环境变量相关的问题，并对导致 SSL 错误的反向代理设置表示担忧。
   - 建议尝试切换到 Hugging Face 镜像站以排查问题。
- **在不同语言中训练 AI 模型**：讨论了基于相似语法结构训练模型理解和说外语的可行性。
   - 建议针对此类任务可能需要调整 **tokenizer**，并针对特定语言水平进行 **prompt engineering**。
- **AI 在幽默感方面的挣扎**：成员们注意到 AI 模型在幽默方面表现不佳，在被要求讲笑话时往往提供平淡的“冷笑话”（dad jokes）而非有趣的回答。
   - 此外，有人评论说，由于理解能力不足或训练限制，**ASCII art** 和 **emojis** 经常导致混乱或乱码。
- **AI 与浏览器的集成**：一位用户分享了一个 YouTube 视频，展示了一个名为 **Do Browser** 的强大 AI **agent** 浏览器扩展，并强调了其功能。
   - 另一位成员强调了令人印象深刻的 **OpenAI realtime API** 集成，它允许通过语音控制网页浏览，展示了不断发展的 AI 应用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/akhaliq/anychat">Anychat - akhaliq 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/HyperbolicLabs/Hyperbolic-Qwen2.5-Coder-Artifacts">Qwen2.5 Coder Artifacts - HyperbolicLabs 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Q11zWq4875o">Do Browser 演示</a>：Do Browser 最强大的 AI agent Chrome 扩展：https://dobrowser.io</li><li><a href="https://huggingface.co/blog/gemma-peft">在 Hugging Face 中微调 Gemma 模型</a>：未找到描述</li><li><a href="https://x.com/sawyerhood/status/1842225025501553044">Sawyer Hood (@sawyerhood) 的推文</a>：OpenAI realtime API 太酷了！我把它连接起来控制我的浏览器，这样我就可以用声音浏览网页了 🤯</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct">Qwen/Qwen2.5-Coder-32B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/do-it-shia-la-beouf-flame-gif-4445204">决定是否应该找暗恋对象聊天 GIF - Do It Shia La Beouf Flame - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/blm-gif-25815938">Blm GIF - Blm - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/ineedit-needit-spongebob-squarepants-need-it-gif-4883495">Need It GIF - Ineedit Needit Spongebob Squarepants - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/i-can-quit-whenever-i-want-smoking-addicted-meme-gif-2482983468824373528">我随时可以戒烟 GIF - I can quit whenever i want Smoking Addicted - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/maimun-monkey-pull-up-hold-hands-gif-15344358">Maimun Monkey GIF - Maimun Monkey Pull Up - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct">meta-llama/Llama-3.2-3B-Instruct · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1309375825325592628)** (4 条消息): 

> `包含 FFmpeg、最新版 Python 和最新版 Node.js 的 Docker 文件` 


- **寻求包含最新软件的 Docker 文件**：一位用户请求协助寻找一个包含最新版本 **FFmpeg**、**Python** 和 **Node.js** 的 **Docker** 文件。
   - 这一请求强调了在开发中整合软件环境的必要性。
- **社区寻求帮助**：另一位用户也发出了求助，响应了寻找 **Docker** 文件的请求。
   - 这突显了社区在支持成员解决技术挑战方面的努力。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1309281745539043390)** (5 条消息): 

> `FLUX.1 Tools 发布，去中心化 AI 模型训练，LivePortrait 配额问题，启发式 AI 经典论文` 


- **FLUX.1 Tools 提供增强的控制能力**：_FLUX.1 Tools_ 的发布引入了一套专为编辑和修改图像设计的工具套件，包含四个关键模型，其中包括 **FLUX.1 Fill** 和 **FLUX.1 Depth**。
   - 这些模型增强了 text-to-image 任务的 **steerability**（可控性），为用户提供了可供尝试的开放访问功能。
- **首个去中心化 AI 模型训练完成**：Prime Intellect 宣布成功训练了 **INTELLECT-1**，这是一个通过横跨美国、欧洲和亚洲的去中心化训练完成的 10B 模型。
   - 预计在大约 **一周内** 进行完整的开源发布，包括基础模型和 checkpoints。
- **LivePortrait Space 遇到配额问题**：一位用户报告在尝试使用 [LivePortrait](https://huggingface.co/spaces/KwaiVGI/LivePortrait) Space 时出现错误，暗示这可能与配额限制有关。
   - 另一位成员指出，它在 **Chrome** 上运行正常，但如果 Space 请求超过了登录用户的限制，可能会失败。
- **关于启发式 AI 的经典之作**：一位用户在对话中引用了一篇关于启发式 AI (Heuristic AI) 的 **经典论文**，强调了其重要性。
   - 讨论中未透露该论文的具体细节。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/KwaiVGI/LivePortrait">Live Portrait - KwaiVGI 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://x.com/PrimeIntellect/status/1859923050092994738">来自 Prime Intellect (@PrimeIntellect) 的推文</a>: 我们做到了 —— 首个 10B 模型的去中心化训练已完成！在跨越美国、欧洲和亚洲的范围内完成训练 🌐 与 @arcee_ai 的后期训练正在进行中，完整的开源发布即将到来...</li><li><a href="https://blackforestlabs.ai/flux-1-tools/">FLUX.1 Tools 介绍</a>: 今天，我们很高兴发布 FLUX.1 Tools，这是一套旨在为我们的基础 text-to-image 模型 FLUX.1 增加控制和 steerability 的模型套件，支持对图像进行修改和重新创作...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1309284900108505170)** (10 条消息🔥): 

> `IntelliBricks toolkit, Eternal AI framework, Social Receipt Generator, Cybertron v4 UNA-MGS model, Autotiktokenizer Windows support` 


- **为 AI 应用推出 IntelliBricks**: 发布了一项关于 **IntelliBricks** 的激动人心的公告，这是一个开源工具包，旨在通过 `msgspec.Struct` 等结构化输出功能来简化 AI 驱动应用程序的开发。
   - *IntelliBricks* 仍在开发中，欢迎在其 [GitHub 仓库](https://github.com/arthurbrenno/intellibricks)上提供进一步的贡献。
- **Eternal AI：去中心化 AI 框架**: **Eternal AI** 正在构建一个去中心化推理框架，以确保 AI 保持抗审查性和可访问性，并计划很快开源其代码。
   - 他们的框架允许开发者在多个区块链上创建强大的链上 AI Agent，促进社区驱动的贡献。
- **Social Receipt Generator 发布**: 介绍了一个名为 **Social Receipt Generator** 的新项目，允许用户为 GitHub 贡献和 Twitter 交流生成有趣的收据。
   - 用户可以通过[此链接](https://receiptgenerator-8j6xdp4dd-sourabh20022002s-projects.vercel.app/)体验该生成器。
- **Cybertron v4 UNA-MGS 模型发布**: **cybertron-v4-qw7B-UNAMGS** 模型回归，作为**排名第 1 且无数据污染（no contamination）的 7-8B LLM**，它具有增强的推理能力。
   - 该模型采用了名为 `MGS` 和 `UNA` 的独特技术，其 Hugging Face 页面详细列出了令人印象深刻的基准测试数据。
- **Autotiktokenizer 的 Windows 支持**: Autotiktokenizer 增加了对 **Windows** 的支持，最近的一个 Pull Request 解决了兼容性问题。
   - 在 Windows 和 Linux 平台上的成功测试为用户指向了更通用的开发体验。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/Sourabh85426135/status/1859997998648086990">来自 Sourabh singh (@Sourabh85426135) 的推文</a>: &#34;隆重推出终极 Social Receipt Generator——非常适合技术爱好者和梗图制作者！ 🧾 ✅ 为您的 GitHub 贡献创建收据。 ✅ 将搞笑的 Twitter 交流制作成收据...</li><li><a href="https://x.com/CryptoEternalAI).">来自 GitHub - FixTweet/FxTwitter 的推文：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://github.com/bhavnicksm/autotiktokenizer/pull/12">[BUG] 通过 not-lain 修复路径以支持 Windows 机器 · Pull Request #12 · bhavnicksm/autotiktokenizer</a>: 此 PR 将修复 #11，同时支持 Windows，已在 Windows 和 Linux 上测试，运行符合预期，抄送 @bhavnicksm</li><li><a href="https://x.com/CryptoEternalAI/status/1858828358513291488">来自 Eternal AI (@CryptoEternalAI) 的推文</a>: Eternal AI：一个开源、不可阻挡的 AI Agent 框架。本周，我们将开源 Eternal AI 代码，让开发者能够部署像 @NOBULLSHIT_EXE 这样的去中心化 AI Agent。如果您想...请私信我们</li><li><a href="https://huggingface.co/fblgit/cybertron-v4-qw7B-UNAMGS">fblgit/cybertron-v4-qw7B-UNAMGS · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/fblgit/miniclaus-qw1.5B-UNAMGS">fblgit/miniclaus-qw1.5B-UNAMGS · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1309334081292730399)** (4 条消息): 

> `Discord chatbots for persona NLPs, Cerebras and Llama 3.1` 


- **寻求人格化 NLP 方面的合作**: 一位成员询问是否有人正在开发作为 Discord 聊天机器人的**人格化 NLP** 或 **LLM 框架**，并表达了与其他开发者合作的愿望。
   - *他们专门寻找可能在该领域进行实验的开发者*。
- **Cerebras 在 Llama 3.1 上领先**: 据指出，**Cerebras** 以惊人的速度运行 **Llama 3.1 405B**，将自己定位为 LLM 性能领域的领导者。
   - *这一见解表明了在大语言模型领域的竞争优势*。


  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1309270720815042601)** (17 条消息🔥): 

> `CLIP models, T5 Token Limit, Flux Tools Integration, Image Variation with FLUX.1 Redux, Serverless Implementation` 


- **通过 Prompt 优化图像生成**：一位用户分享了一种生成高质量图像的方法，即为 CLIP 模型使用精简的 Prompt，同时为 T5 使用较长的描述性 Prompt，以确保 CLIP Prompt 不会被截断。
   - 这种方法通过将 **prompt_2** 自动附加到 T5 encoder，可以获得更好的图像质量。
- **T5 模型能够处理 512 个 Token**：用户发现，通过在调用参数中添加 `max_sequence_length=512`，用于 SD3/Flux 的 T5 模型可以处理多达 **512 个 Token**。
   - 这一发现引发了人们对如何在各种调用中有效实现该功能的进一步兴趣。
- **关于 Flux Tools 功能的咨询**：一位用户询问是否有人成功将 **Flux tools** 与 diffusers 集成，随后讨论了在 Hugging Face 上发现的一个潜在 Model Card。
   - 另一位用户分享了 **flux1-fill-dev-diffusers 的 Model Card** 链接，并对其运行状态提出了疑问。
- **探索用于图像变体的 FLUX.1 Redux**：频道成员分享了关于 **FLUX.1 Redux** 的信息，这是一个用于生成图像变体的 Adapter，可以通过细微修改来优化输入图像。
   - 讨论强调了该模型通过 API 和 Workflow 集成进行图像风格重塑的功能，并链接到了 [更多信息](https://blackforestlabs.ai/flux-1-tools/)。
- **对 Flux 的 Serverless 实现感兴趣**：用户表示打算使用 Runpod Serverless 运行 **Flux tools**，并对仓库中提供的 CLI 代码表现出浓厚兴趣。
   - 这一进展表明社区渴望探索 Flux 技术在不同环境中的实际应用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/xiaozaa/flux1-fill-dev-diffusers">xiaozaa/flux1-fill-dev-diffusers · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev">black-forest-labs/FLUX.1-Redux-dev · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1309287369672757268)** (87 条消息🔥🔥): 

> `AI Art Turing Test, Anthropic's $4 Billion Investment from AWS, LTX Video Generation Model, AI Vibrancy Rankings Tool, OpenAI's Deleted Training Findings` 


- **对 AI 艺术图灵测试的热情**：最近的一项 AI 艺术图灵测试引发了讨论，一名成员表示希望由艺术修复专家进行测试。
   - 参与者的结果显示体验各异，特别是在区分 AI 生成和人类创作的艺术品方面。
- **Anthropic 获得来自 AWS 的 40 亿美元投资**：Anthropic 已获得亚马逊追加的 40 亿美元投资，巩固了 AWS 作为其主要云服务和训练合作伙伴的地位。
   - 该伙伴关系旨在通过在 AWS Trainium 硬件上的协作来增强 AI 模型训练。
- **LTX Video 模型发布**：Lightricks 推出了开源的 LTX Video 模型，能够在高性能硬件上仅用 4 秒生成 5 秒的视频。
   - 该模型支持通过 API 轻松访问，引发了关于平衡本地处理与云端支出的讨论。
- **来自斯坦福的 AI 活力工具**：一位成员分享了对斯坦福 AI 活力排名工具（Stanford AI Vibrancy Rankings Tool）的热情，该工具根据 AI 发展指标对各国进行排名。
   - 该工具允许用户自定义各种指标的权重，以反映他们自己对 AI 活力的看法。
- **OpenAI 的数据删除事件**：关于 OpenAI 最近意外删除训练发现的争论四起，引发了对其管理关键数据能力的质疑。
   - 虽然 OpenAI 和 NYT 的律师都承认这是一个错误，但对数据处理和恢复的担忧依然存在。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/altryne/status/1859996654830805358?s=46">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：🤯 来自 @Lightricks @LTXStudio 的全新文本和图像转视频（txt and img 2 Video）近实时模型。在 H100 上仅需 4 秒即可生成 5 秒视频。完全开源，专为消费级硬件设计！你可以...</li><li><a href="https://maxread.substack.com/p/people-prefer-ai-art-because-people?utm_source=post-email-title&publication_id=392873&post_id=151984955&utm_campaign=email-post-title&isFreemail=true&r=43kx5&triedRedirect=true&utm_medium=email">人们更喜欢 AI 艺术，因为人们更喜欢糟糕的艺术</a>：理解“AI 艺术图灵测试”</li><li><a href="https://x.com/anthropicai/status/1859964653486612585?s=46">来自 Anthropic (@AnthropicAI) 的推文</a>：我们正在扩大与 AWS 的合作。这包括来自 Amazon 的 40 亿美元新投资，并确立 AWS 为我们的主要云和训练合作伙伴。http://anthropic.com/news/anthrop...</li><li><a href="https://x.com/adonis_singh/status/1859682100569571399">来自 adi (@adonis_singh) 的推文</a>：我有种感觉， Google 刚刚下了一盘大棋。他们发布了 Gemini exp 1114（中等，甚至更差的模型），深知 OpenAI 会想要超越他们。Google 诱导出了新的 GPT-4o，然后用...打击他们。</li><li><a href="https://x.com/ltxstudio/status/1859964100203430280?s=46">来自 LTX Studio (@LTXStudio) 的推文</a>：(1/13) 我们一直在努力开发一些特别的东西 ✨ 介绍 LTX Video，这是 Lightricks 推出的全新开源、社区驱动的视频生成模型。瞬间创建令人惊叹的视频，飞速超越...</li><li><a href="https://x.com/giffmana/status/1859317159727333552?s=46">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>：哈哈哈 引用 Liron Shapira (@liron) https://www.astralcodexten.com/p/how-did-you-do-on-the-ai-art-turing</li><li><a href="https://www.redhat.com/en/about/press-releases/red-hat-acquire-neural-magic">Red Hat 宣布达成收购 Neural Magic 的最终协议</a>：Red Hat 宣布已签署收购 Neural Magic 的最终协议，该公司是加速生成式 AI (gen AI) 推理工作负载的软件和算法先驱。</li><li><a href="https://x.com/ericnewcomer/status/1859732779388621264?s=46">来自 Eric Newcomer (@EricNewcomer) 的推文</a>：在 Cerebral Valley 的最后一场会议上，Anthropic 的 @DarioAmodei 渴望纠正大家的看法。没有理由相信基础模型的进展即将开始放缓。“我...”</li><li><a href="https://x.com/ericnewcomer/status/1859757810843779446?s=46">来自 Eric Newcomer (@EricNewcomer) 的推文</a>：这是我在 Cerebral Valley AI 峰会上与 @DarioAmodei (@AnthropicAI) 的完整对话。</li><li><a href="https://www.aboutamazon.com/news/aws/amazon-invests-additional-4-billion-anthropic-ai">Amazon 和 Anthropic 深化战略合作</a>：Anthropic 指定 AWS 为其主要训练合作伙伴，并将使用 AWS Trainium 来训练和部署其最大的基础模型；Amazon 将向 Anthropic 追加 40 亿美元投资。</li><li><a href="https://x.com/JinaAI_/status/1859659764281782420">来自 Jina AI (@JinaAI_) 的推文</a>：Jina-CLIP-v2：一个 0.9B 的多语言多模态嵌入模型，支持 89 种语言、512x512 图像分辨率、8192 token 长度，以及低至 64 维的图像和文本 Matryoshka 表示...</li><li><a href="https://www.anthropic.com/news/anthropic-amazon-trainium">利用 AWS 助力下一代 AI 开发</a>：今天我们宣布扩大与 AWS 在 Trainium 上的合作，以及来自 Amazon 的 40 亿美元新投资。</li><li><a href="https://t.co/6ZR3GKxV9r">你在 AI 艺术图灵测试中表现如何？</a>：...</li><li><a href="https://x.com/anthropicai/status/1858976458330505639?s=46">来自 Anthropic (@AnthropicAI) 的推文</a>：Anthropic 新研究：为评估（Evals）添加误差棒。AI 模型评估通常不包含统计数据或不确定性。我们认为应该包含。阅读博客文章：https://www.anthropic.com/res...</li><li><a href="https://x.com/reach_vb/status/1859868073903423821">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：来自 @Apple 的新开源发布 - AIMv2 - 大规模视觉编码器 🔥 > 在主要多模态理解基准测试中优于 CLIP 和 SigLIP > 在开放词汇目标检测上击败 DINOv2...</li><li><a href="https://x.com/xianbao_qian/status/1859808310700146795?s=46">来自 Tiezhen WANG (@Xianbao_QIAN) 的推文</a>：@AlibabaGroup 刚刚揭晓了 ChatGPT o1 模型的隐藏秘密。他们刚刚发布了一个采用 Apache 2 许可证的 o1 替代方案。arco-o1 由 CoT 微调、MCTS、反思和推理驱动...</li><li><a href="https://bsky.app/profile/calcsam.bsky.social/post/3lbimbmk7zs2k">Sam Bhagwat (@calcsam.bsky.social)</a>：很高兴分享 Shane Thomas、Abhi Aiyer 和我正在构建 Mastra，这是一个为下一百万 AI 开发者准备的 TypeScript AI 框架：</li><li><a href="https://git">

hub.com/MadcowD/ell/issues/150">TypeScript 支持 · Issue #150 · MadcowD/ell</a>: Ell TypeScript 目标：与 Python 功能对齐，无干扰/无摩擦的 DX。挑战：捕获 TypeScript 源代码 - Ell 按原样捕获用户编写的源代码，与其在用户...</li><li><a href="https://www.cnbc.com/2024/11/22/amazon-to-invest-another-4-billion-in-anthropic-openais-biggest-rival.html">亚马逊将向 OpenAI 的最大对手 Anthropic 追加 40 亿美元投资</a>: 亚马逊周五宣布将向 Anthropic 追加投资 40 亿美元，这家由前 OpenAI 研究高管创立的人工智能初创公司。</li><li><a href="https://x.com/mariushobbhahn/status/1857027208050512206?s=46">来自 Marius Hobbhahn (@MariusHobbhahn) 的推文</a>: 这篇关于 evals 统计学的论文非常棒（而且似乎未引起足够关注）：https://arxiv.org/abs/2411.00640v1。作者基本上展示了 ev... 所需的所有相关统计工具。</li><li><a href="https://en.wikipedia.org/wiki/Battle_of_Grunwald_(Matejko)">格伦瓦德之战 (Matejko) - 维基百科</a>: 未找到描述</li><li><a href="https://techcrunch.com/2024/11/22/anthropic-raises-an-additional-4b-from-amazon-makes-aws-its-primary-cloud-partner/">Anthropic 从亚马逊再融资 40 亿美元，使 AWS 成为其“主要”训练合作伙伴 | TechCrunch</a>: Anthropic 已从亚马逊额外筹集了 40 亿美元，并同意主要在 Amazon Web Services 上训练其旗舰生成式 AI 模型。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1309624736191479869)** (162 条消息🔥🔥): 

> `LLM 驱动的需求分析，Obsidian 集成，使用 AI 编程，Paranoiac-critical method，Windsurf 工具改进` 


- **LLM 驱动的需求分析备受关注**：成员们表示 **LLM 驱动的需求分析** 是目前 AI 领域最有趣的话题之一，强调了它在理解复杂问题方面的实用性。
   - 对话强调，许多分析和建模过程可以通过 LLM 有效地实现自动化。
- **Obsidian 被证明对学习很有用**：一位成员分享了他们使用 **Obsidian** 的经验，强调了它在创建思维导图和学习时保留上下文信息方面的有效性。
   - 许多参与者一致认为，将 Obsidian 等工具与 AI 结合使用可以使学习过程更具互动性和趣味性。
- **Mermaid 图表增强了文档编写**：讨论了 **Mermaid 图表** 如何有助于更好的文档实践，允许用户在笔记中可视化关系。
   - 成员们提到了在某些工具（如 Windsurf）中渲染的困难，但仍然认为使用图表来提高清晰度很有价值。
- **AI 让编程再次变得有趣**：许多参与者表示，LLM 的出现激发了他们对编程的热情，让编程再次变得令人愉悦。
   - 参与者分享说，在这个领域与他人建立联系鼓励了协作和创新想法。
- **Paranoiac-critical method 激发创意**：一位成员介绍了萨尔瓦多·达利（Salvador Dalí）使用的 **paranoiac-critical method**，作为激发创意的一种方式，认为改变状态可以增强艺术产出。
   - 这种方法与当前关于 AI 对创造力和解决问题影响的讨论联系在了一起。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://obsidian.md/">Obsidian - Sharpen your thinking</a>: Obsidian 是一款私密且灵活的笔记应用，能够适应你的思维方式。</li><li><a href="https://en.wikipedia.org/wiki/Paranoiac-critical_method">Paranoiac-critical method - Wikipedia</a>: 未找到描述</li><li><a href="https://github.com/ddd-crew/ddd-starter-modelling-process">GitHub - ddd-crew/ddd-starter-modelling-process: If you&#39;re new to DDD and not sure where to start, this process will guide you step-by-step</a>: 如果你是 DDD 的新手且不确定从哪里开始，这个过程将逐步引导你 - ddd-crew/ddd-starter-modelling-process</li><li><a href="https://tenor.com/view/old-no-brain-brains-out-spongebob-squarepants-gif-7878833">Old No Brain GIF - Old No Brain Brains Out - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://changelog.com/jsparty/346">It&#39;s all about documentation with Carmen Huidobro (JS Party #346)</a>: Carmen Huidobro 加入了 Amy、KBall 和 Nick 的节目，谈论她的工作、编写文档的重要性，以及她在 React Summit US 即将发表的会议演讲！</li><li><a href="https://registerspill.thorstenball.com/p/they-all-use-it">They all use it</a>: 上周，在一次会议上，我与另一位工程师在走廊里偶然聊起了 AI。</li><li><a href="https://xkcd.com/1987/">Python Environment</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1309279933918482562)** (186 messages🔥🔥): 

> `语音克隆, AI 口音理解, Bluesky vs Twitter, ChatGPT 进展, Airtable 与 Notion 集成` 


- **语音克隆经验**：一位成员分享了使用各种 **voice models** 的经验，提到了使用过程中出现的意外声音，以及语音克隆如何产生令人不安的效果（如对话中出现音乐）。
   - 另一位成员提到将语音克隆用于 **audiobook**（有声读物）改编，但遇到了故障，有时会产生令人惊讶的增强效果，比如唱歌。
- **AI 口音与幻觉**：有人提出了关于 **realtime API** 在理解不同口音和发音方面表现如何的问题，并将其与可能出现的幻觉（hallucinations）进行了对比。
   - 这引发了关于集成多种 **voice models** 及其在理解用户输入方面不同能力的讨论。
- **科技治理的反乌托邦观点**：一位用户对 **big tech**（大型科技公司）成为管理实体表示担忧，分享了关于 X 和 Bluesky 等平台上不受监管的言论和内容管理影响的看法。
   - 这也引发了关于 **free speech**（言论自由）感知下降以及社交媒体格局对用户互动影响的对话。
- **ChatGPT 集成创意**：讨论包括使用 ChatGPT 进行角色扮演场景的经验，产生了幽默的结果，以及一些令人惊讶的提示词（prompts）让模型展现出更多个性。
   - 成员们还强调了与 **Airtable** 和 **Notion** 等工具集成的潜力，讨论了在这些应用程序中改进提示词编写的目标。
- **Copilot 图像生成推测**：人们对 **Copilot** 的图像生成能力感到好奇，一位成员推测这些图像是源自未发布的 **DALL-E** 模型还是名为 Sora 的新程序。
   - 成员对不同 AI 工具生成的图像进行了比较，指出了质量差异，并承认了其他模型可能产生的影响。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1309348677667520512)** (14 messages🔥): 

> `教导 GPT 词汇约束, Dall-E 的替代方案, 图像生成模型, 访问免费模型` 


- **GPT 在 Dall-E 的词汇控制方面表现不佳**：一位成员表达了挫败感，称其 GPT 在生成约 **10 张图像**后往往会忘记特定的词汇约束。
   - 他们正在寻求维持角色描述并避免在生成内容中出现违禁词的技巧。
- **探索 Dall-E 的替代方案**：成员们讨论了将 **Stable Diffusion** 或 **Flux models** 与 **comfyUI** 结合使用作为 Dall-E 的潜在替代方案，认为它们可能更好地处理特定的词汇限制。
   - 他们建议查看 YouTube 上最近的教程，以确保获得保留角色完整性的最新方法。
- **关于模型所有权和垄断的辩论**：一位成员指出，Dall-E 不使用其他模型的原因在于所有权，这暗示了一种限制其选择的 **monopoly**（垄断）形式。
   - 另一位成员反驳称，目前有许多免费的图像模型可用，对该领域存在垄断的说法表示怀疑。
- **免费图像生成选项的可用性**：成员们注意到存在许多免费图像模型，包括通过 **Hugging Face** 进行在线推理的选项。
   - 建议包括探索各种模型，以便在无需本地设置的情况下实现图像生成方法的多样化。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1309388689075339295)** (5 messages): 

> `最大化 AI 回复, Prompt Engineering, 在提示词中使用变量, 讨论中的幽默` 


- **Dr. Feelgood 用于高效 AI 回复的提示词**：一位用户声称，能从 AI 那里获得最大生产力和专业性的最佳提示词是：‘**don’t be chickens 🐔 give me good answer bok bok chicken🐣**’。他们认为这个提示词在所有 AI 模型中都能产生最佳效果。
   - 另一位用户询问了这种方法的有效性，以及在此类提示词中是否使用了变量。
- **初识 Prompt Engineering 但充满信心**：Dr. Feelgood 提到他们刚刚开始接触 **Prompt Engineering**，但觉得他们的提示词在增强 AI 回复方面“非常准确”。
   - 他们还确认在提示词中经常使用 **x** 和 **y** 等变量。
- **提示词讨论中的数学笑话**：一位用户开了一个关于有时间处理 **t** 的幽默玩笑，为技术对话增添了轻松的氛围。
   - *这个数学笑话很受欢迎，展示了 AI 和提示词讨论中轻松活泼的氛围。*


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1309388689075339295)** (5 条消息): 

> `有效的 AI Prompt，Prompt Engineering 讨论，在 Prompt 中使用变量` 


- **通过巧妙的 Prompt 最大化生产力**：一位成员分享了他们发现的一个高效 Prompt：‘don’t be chickens 🐔 give me good answer bok bok chicken🐣’，旨在提升所有 AI 模型的生产力。
   - *它被吹捧为完美的 Prompt*，暗示它可能比传统方法产生更好的响应。
- **对 Prompt Engineering 经验的好奇**：另一位成员询问了原作者在 Prompt Engineering 方面的经验，对其专业领域以及对改进输出的看法感兴趣。
   - 回复中强调他们才刚刚开始接触 Prompt Engineering，但对结果充满信心。
- **Prompt 中的变量**：原作者确认他们在 Prompt 中使用了像 x 和 y 这样的变量，以提高响应能力和定制化程度。
   - 这种方法被视为其 Prompt 技术中的一个战略要素，增强了与 AI 的互动。
- **AI 讨论中的数学幽默**：当一位成员开玩笑地问是否有时间留给 't'（一个关于变量的谐音梗）时，发生了一次轻松的互动。
   - 这种玩笑为技术讨论增添了幽默感，展示了社区的友好氛围。


---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1309265094877905018)** (155 条消息🔥🔥): 

> `Qwen 模型性能，Aider 基准测试结果，OpenRouter 与直接 API 访问对比，量化对模型性能的影响，近期 AI 投资` 


- **Qwen 模型测试结果参差不齐**：多位用户报告了 **Qwen 2.5 Coder** 模型在不同供应商处的性能评分差异，其中 **Hyperbolic** 的结果较低，仅为 **47.4%**，而排行榜上的分数为 **71.4%**。
   - 讨论强调了理解量化（Quantization）如何影响性能的必要性，测试表明 **BF16** 与其他变体产生了不同的结果。
- **Aider 基准测试方法的变更**：澄清了 Aider 排行榜上 **Qwen 2.5 Coder 32B** 的评分现在使用的是通过 **GLHF** 获取的 **HuggingFace** 权重，这提高了基准测试的准确性。
   - 用户对不同托管平台导致的评分差异以及模型质量可能存在的波动表示担忧。
- **直接 API 访问 Qwen 模型**：讨论提到 Aider 框架现在可以直接访问 **Qwen** 模型，而无需通过 **OpenRouter**，从而方便了不依赖特定 API 供应商的使用。
   - 这一变化旨在通过减少对第三方服务的依赖来提升用户体验，同时保持模型性能。
- **社区参与和模型测试**：社区成员积极测试了各种 **Qwen 2.5** 供应商，分享了来自他们基准测试的见解和性能指标，以支持彼此了解模型的有效性。
   - 持续的讨论凸显了社区在提高 LLM 在不同应用中的易用性方面对透明度和协作的承诺。
- **AI 投资新闻**：分享了关于 **Amazon** 计划向 **Anthropic** 追加投资 **40 亿美元**的链接，突显了 AI 开发领域的竞争格局。
   - 随着企业对新技术兴趣和投资的增加，这引发了关于 AI 项目可持续性和创新速度的对话。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www-cnbc-com.cdn.ampproject.org/c/s/www.cnbc.com/amp/2024/11/22/amazon-to-invest-another-4-billion-in-anthropic-openais-biggest-rival.html">回到顶部</a>: 未找到描述</li><li><a href="https://aider.chat/2024/11/21/quantization.html">量化至关重要</a>: 开源 LLM 正变得非常强大，但请注意你（或你的供应商）是如何对模型进行量化的。这会强烈影响代码编辑能力。</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately">ollama/docs/faq.md · ollama/ollama</a>: 快速上手 Llama 3.2, Mistral, Gemma 2 和其他大语言模型。 - ollama/ollama</li><li><a href="https://openrouter.ai/google/gemini-exp-1121:free">Gemini Experimental 1121 (免费) - API, 供应商, 统计数据</a>: Gemini 的实验性版本（2024年11月21日）。通过 API 运行 Gemini Experimental 1121 (免费)</li><li><a href="https://github.com/lee88688/aider-composer">GitHub - lee88688/aider-composer: aider 的 VSCode 扩展，无缝集成到 VSCode</a>: aider 的 VSCode 扩展，无缝集成到 VSCode - GitHub - lee88688/aider-composer</li><li><a href="https://github.com/lee88688/aider-composer/issues/2#issuecomment-">如何安装和使用？ · Issue #2 · lee88688/aider-composer</a>: 未找到描述</li><li><a href="https://github.com/ollama/ollama/issues/3694">Cloudflare 状态码 524 · Issue #3694 · ollama/ollama</a>: 问题是什么？我遇到了以下错误消息：Ollama 调用失败，状态码为 524。详情：&lt;bound method ClientResponse.text of &lt;ClientResponse(http://chat.tamdu.com/...</li><li><a href="https://github.com/lee88688/aider-composer/issues/2#issuecomment-2475384208">如何安装和使用？ · Issue #2 · lee88688/aider-composer</a>: 未找到描述</li><li><a href="https://aistudio.google.com/).">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1309281731488120892)** (38 条消息🔥): 

> `在 Aider 中管理上下文，保存聊天会话，API 连接错误，基准测试成本与性能，Aider 中的文件检测问题` 


- **在 Aider 中管理上下文**：一位用户询问如何防止 Aider 在向量填充期间处理特定行以节省 token，得到的建议是重构代码以隔离无关数据。
   - 另一位用户询问如何增加 **Qwen** 的上下文窗口，对 **32k** 的限制表示不满。
- **保存聊天会话**：一位成员询问如何保存整个聊天会话，而另一位成员确认所有会话都记录在 `.aider.chat.history.md` 中。
   - 讨论强调了目前缺乏像 /save 这样用于导出聊天历史的简单保存命令。
- **API 连接错误**：一位用户报告在尝试发送请求时持续出现 API 连接错误，并反复收到 **APIConnectionError** 消息。
   - 这引发了关于 OpenRouter 可靠性的讨论，特别是针对 **sonnet 3.5** 模型。
- **基准测试成本与性能**：一位用户分享说，运行 **Qwen2.5-Coder-32B-Instruct** 的基准测试仅花费了 **$0.25**，说明了测试模型的经济性。
   - 他们对同一模型在不同供应商之间的潜在成本差异表示好奇，并计划分享他们的发现。
- **Aider 中的文件检测问题**：一位用户在让 Aider 检测新文件时遇到困难，尽管已经将它们提交到 GitHub，这导致了困惑并需要进行故障排除。
   - 该问题通过重新安装 Aider 得到了解决，展示了用户配置方案解决常见挫折的可能性。



**提到的链接**：<a href="https://aider.chat/docs/llms/xai.html">xAI</a>：aider 是你终端里的 AI 结对编程工具

  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1309262591217893467)** (1 条消息): 

> `Uithub 工具，AI 驱动开发，GitHub 替代方案` 


- **Uithub：你的 GitHub 替代方案**：用户们对 [Uithub](http://uithub.com) 赞不绝口，将其描述为一个通过简单地将 'G' 改为 'U' 即可将仓库复制粘贴到 LLM 的即时工具。
   - *Nick Dobos* 在 2024 年 10 月 4 日提到，这是他用于快速访问仓库的**新宠**。
- **使用 Uithub 轻松获取仓库上下文**：用户发现 Uithub 有助于更有效地浏览 GitHub 仓库，例如毫不费力地拉取完整的仓库上下文。
   - *Ian Nuttall* 在 2024 年 10 月 2 日分享了他使用 Uithub 获取 **Laravel SEO** 完整仓库上下文的经验，强调了它相比传统 GitHub 链接的优势。
- **社区对 Uithub 的反馈**：社区正在分享对 Uithub 的正面印象，称其为一个**非常实用的好工具**，增强了他们的开发体验。
   - *Yohei Nakajima* 也在 2024 年 10 月 2 日表达了发现 Uithub 的热情。
- **AI 驱动的开发能力**：Uithub 提供了类似于 --show-repo-map 功能的特性，提供了更高的 token 限制和针对特定文件类型的高级过滤。
   - 然而，有人指出 Uithub 可能缺乏 **Aider** 工具中一些更复杂的功能，从而引发了关于它们实用性对比的讨论。



**提到的链接**：<a href="https://uithub.com/">uithub - 轻松向你的 LLM 提问代码问题</a>：未找到描述

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1309262296765169716)** (150 条消息🔥🔥): 

> `每日 LLM 发布, 上议院政治结构, 推理数据集质量, 阿里巴巴 AI 进展` 


- **每日 LLM 发布讨论**：一位成员评论了持续不断的“每日 LLM 发布”，而另一位成员则表示“Goodhart arena 已经过时了”，暗示讨论内容趋于重复。
   - 这次交流凸显了 AI 社区对平庸话题日益增长的疲劳感。
- **评估上议院**：一位用户展示了关于英国上议院在政治体系中角色的逐步推理过程，结论是它确实是政治结构的一部分。
   - 该讨论说明了 AI 模型在处理复杂推理任务时，相比于简单的记忆任务，可能会面临困难。
- **对 AI 推理数据集的担忧**：人们对推理数据集的质量提出了担忧，对其有效性表示怀疑，并对其可能的合成方式提出了批评。
   - 一位成员幽默地建议，由于这些数据集的奇特性，要么是生成质量太差，要么是受到了药物使用的影响。
- **对阿里巴巴 AI 工作的观察**：一位用户注意到阿里巴巴的 Marco-o1 模型仅展示了一个 Benchmark，从而对该模型的公信力和性能产生了怀疑。
   - 对话反映了对营销说辞以及新兴 AI 模型评估范围有限的挫败感。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.16039">MoEUT: Mixture-of-Experts Universal Transformers</a>: 之前关于 Universal Transformers (UTs) 的工作已经证明了跨层参数共享的重要性。通过允许深度递归，UTs 在学习方面比标准 Transformers 具有优势...</li><li><a href="https://hermes.nousresearch.com/">NOUS CHAT | Talk to Hermes</a>: 体验与 Hermes 的自然、智能对话，Hermes 是由 Nous Research 开发的开源 LLM。</li><li><a href="https://x.com/AnthropicAI/status/1859964653486612585">来自 Anthropic (@AnthropicAI) 的推文</a>: 我们正在扩大与 AWS 的合作。这包括来自 Amazon 的 40 亿美元新投资，并将 AWS 确立为我们的主要云和训练合作伙伴。 http://anthropic.com/news/anthrop...</li><li><a href="https://x.com/teknium1/status/1859997785220947990?s=46">来自 Teknium (e/λ) (@Teknium1) 的推文</a>: 那是一大笔钱，但要让 Opus 4 或 5 的预训练 Scaling 以同样的速度持续下去，需要 2000 亿，好奇两年后我们会处于什么位置。 Quoting Anthropic (@AnthropicAI) 我们正在扩大...</li><li><a href="https://github.com/AIDC-AI/Marco-o1">GitHub - AIDC-AI/Marco-o1: An Open Large Reasoning Model for Real-World Solutions</a>: 针对现实世界解决方案的开源大推理模型 - AIDC-AI/Marco-o1</li><li><a href="https://machinelearning.apple.com/research/introducing-apple-foundation-models">Introducing Apple’s On-Device and Server Foundation Models</a>: 在 2024 年全球开发者大会上，我们推出了 Apple Intelligence，这是一个深度集成到……的个人智能系统。</li><li><a href="https://huggingface.co/fblgit/cybertron-v4-qw7B-UNAMGS">fblgit/cybertron-v4-qw7B-UNAMGS · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/fblgit/miniclaus-qw1.5B-UNAMGS">fblgit/miniclaus-qw1.5B-UNAMGS · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1309264516097249301)** (19 messages🔥): 

> `Agent APIs availability, Fine-tuning and datasets, Graphical User Interfaces for LLMs, Chat interfaces for multiple models` 


- **Agent API 在最初的热潮期间不可用**：一位成员指出，在“Agent 热潮”的早期阶段，API 中的某些功能并不可用。另一位成员证实了这一点，并提到幸运的是这些功能现在已经可以访问了。
   - 这引发了关于在评估初始 Agent 能力时事后诸葛亮影响的讨论。
- **需要微调数据集的建议**：一位成员表示希望微调模型并创建数据集，并征求关于参数和工具的建议，同时表达了对试错成本过高的担忧。
   - 另一位成员建议使用 Axolotl 的示例默认值，这些默认值被认为对于训练运行非常有效。
- **探索机器学习模型的 GUI 选项**：成员们讨论了对本地托管聊天体验的图形用户界面的偏好，重点关注了 Open WebUI 和 LibreChat 等工具。一位成员确认 Open WebUI 在这方面广受青睐。
   - 另一位成员分享了 Open WebUI 功能的动画演示，强调了其用户友好的界面。
- **寻求支持多种模型的聊天界面**：一位用户询问了是否有一种聊天界面可以与多个模型进行交互，而不仅仅是单个服务提供的模型。成员们乐于分享关于通用聊天界面的推荐。
   - 一位成员链接到了 Open WebUI 的文档，指出其支持各种 LLM runner 的能力，并鼓励大家进行探索。



**Link mentioned**: <a href="https://docs.openwebui.com/">🏡 Home | Open WebUI</a>: Open WebUI 是一个可扩展、功能丰富且用户友好的自托管 AI 界面，旨在完全离线运行。它支持各种 LLM runner，包括 Ollama 和 OpenAI 兼容的 API...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1309418488351166476)** (2 messages): 

> `Marco-o1 model release, Research on reasoning models, Open-ended problem solving, Authors of new AI research` 


- **阿里巴巴发布采用 Apache 2 许可证的 Marco-o1**：AlibabaGroup 推出了 **Marco-o1** 模型，这是 OpenAI o1 的替代方案，现采用 **Apache 2** 许可证，专注于通过各种创新策略解决复杂问题。
   - 在 **CoT** 微调和 **MCTS** 的支持下，Marco-o1 寻求在标准和奖励量化困难的更广泛领域实现泛化。
- **Marco-o1 中推理模型的探索**：Marco-o1 论文讨论了由 OpenAI o1 引发的对**大推理模型 (LRM)** 的兴趣激增，强调了超越标准答案学科的阶段。
   - 该研究旨在评估 o1 模型是否能在结构化程度较低的环境中自适应地处理**开放式解决方案**。
- **近期 AI 研究中的知名作者**：这份创新论文列表包括了 **Xin Dong**、**Yonggan Fu** 和 **Jan Kautz** 等知名研究人员，他们为 AI 模型不断演进的格局做出了贡献。
   - 这些作者因其在提升 AI 能力和探索新方法论方面的见解而受到认可。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.13676">Hymba: A Hybrid-head Architecture for Small Language Models</a>: 我们提出了 Hymba，这是一个小语言模型系列，采用混合头并行架构，将 Transformer 注意力机制与状态空间模型 (SSMs) 集成，以提高效率...</li><li><a href="https://x.com/Xianbao_QIAN/status/1859808310700146795">Tweet from Tiezhen WANG (@Xianbao_QIAN)</a>: @AlibabaGroup 刚刚揭晓了 ChatGPT o1 模型的隐藏秘密。他们刚刚发布了一个采用 Apache 2 许可证的 o1 替代方案。Marco-o1 由 CoT 微调、MCTS、反思和推理驱动...</li><li><a href="https://arxiv.org/abs/2411.14405">Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions</a>: 目前 OpenAI o1 引发了对大推理模型 (LRM) 研究的热潮。借此势头，Marco-o1 不仅关注数学等具有标准答案的学科...</li><li><a href="https://huggingface.co/AIDC-AI/Marco-o1">AIDC-AI/Marco-o1 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1309290396156956804)** (2 messages): 

> `Agentic Translation Workflow, Few-shot Prompting, Iterative Feedback, LLM Output Refinement` 


- **Agentic 翻译工作流解析**：**Agentic 翻译工作流**利用 **Few-shot Prompting** 和**迭代反馈循环**，而非传统的翻译微调。
   - 该方法从翻译提示词开始，允许 **LLM** 对其输出进行批判和细化，使其具有**灵活性**和**可定制性**。
- **使用迭代反馈的好处**：利用**迭代反馈**使翻译过程能够避免与训练相关的开销，从而提高生产力。
   - 这一特性使得该工作流特别具有吸引力，因为它在翻译任务中结合了定制化和高效性。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1309418488351166476)** (2 messages): 

> `Marco-o1 model, ChatGPT o1 alternative, Open-ended problem-solving, Real-world reasoning models` 


- **阿里巴巴推出 Marco-o1 作为 o1 的替代方案**：阿里巴巴刚刚发布了 **Marco-o1**，这是 ChatGPT o1 模型的替代方案，目前采用 **Apache 2 license**。
   - 在 **Chain-of-Thought** 微调和 **Monte Carlo Tree Search** 的支持下，它旨在增强标准领域和开放式领域的问题解决能力。
- **探索 Marco-o1 的能力**：Marco-o1 专注于复杂的现实世界挑战，探索 o1 模型是否能在**标准不明确**的领域中有效泛化。
   - _“在缺乏明确标准且奖励难以量化的更广泛领域，o1 模型能否有效地进行泛化？”_
- **开放式解决方案研究**：最近一篇论文的摘要强调，Marco-o1 旨在超越**数学**和**代码**等传统学科。
   - 它解决了对**开放式解决方案**的需求，利用先进的**推理策略**在不同领域实现更广泛的适用性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.13676">Hymba: A Hybrid-head Architecture for Small Language Models</a>: 我们提出了 Hymba，这是一个小语言模型系列，采用混合头并行架构，将 Transformer 注意力机制与 State Space Models (SSMs) 相结合，以增强效率...</li><li><a href="https://x.com/Xianbao_QIAN/status/1859808310700146795">Tiezhen WANG (@Xianbao_QIAN) 的推文</a>: @AlibabaGroup 刚刚揭开了 ChatGPT o1 模型的隐藏秘密。他们刚刚发布了一个带有 Apache 2 license 的 o1 替代方案。Marco-o1 由 CoT 微调、MCTS、反思和推理驱动...</li><li><a href="https://arxiv.org/abs/2411.14405">Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions</a>: 目前 OpenAI o1 引发了大型推理模型 (LRM) 的研究热潮。顺应这一势头，Marco-o1 不仅关注具有标准答案的学科，如数学...</li><li><a href="https://huggingface.co/AIDC-AI/Marco-o1">AIDC-AI/Marco-o1 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1309599639900524675)** (1 条消息): 

> `Claude 3.5 Haiku 重命名，模型 ID 变更，Discord 模型申请` 


- **Claude 3.5 Haiku 进行了“点号”重命名**：模型 **Claude 3.5 Haiku** 已重命名，在其 ID 中使用**点号 (dot)**代替**连字符 (dash)**，这会影响其可用性。
   - 新的模型 ID 可以在 [Claude 3.5 Haiku](https://openrouter.ai/anthropic/claude-3.5-haiku) 和 [Claude 3.5 Haiku 20241022](https://openrouter.ai/anthropic/claude-3.5-haiku-20241022) 找到，但提醒用户这些 ID 目前可能尚不可用。
- **指定了多个模型 ID**：其他模型 ID 包括 [Claude 3.5 Haiku:beta](https://openrouter.ai/anthropic/claude-3.5-haiku:beta) 和 [Claude 3.5 Haiku 20241022:beta](https://openrouter.ai/anthropic/claude-3.5-haiku-20241022:beta)，但它们目前也无法使用。
   - 用户可以访问我们的 [Discord](https://discord.gg/fVyRaUDgxW) 寻求帮助并申请这些模型。
- **之前的 ID 仍然有效**：尽管发生了这些变化，与模型关联的旧 ID 应该仍能正常工作，不会出现问题。
   - 对发现并标记模型标识变更的用户表示感谢。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/anthropic/claude-3.5-haiku>">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/anthropic/claude-3.5-haiku-20241022>">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/anthropic/claude-3.5-haiku:beta>">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/anthropic/claude-3.5-haiku-20241022:beta>">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1309262891664543794)** (118 条消息🔥🔥): 

> `Gemini 模型问题，OpenRouter API 使用，OpenRouter 积分税务，Prompt Engineering 策略，工程社区更新` 


- **Gemini 模型面临配额限制**：用户报告在尝试访问免费的 Gemini Experimental 1121 模型时遇到配额错误，特别是在使用 OpenRouter 时。
   - 建议包括直接连接到 Google Gemini 以获得更好的访问权限。
- **OpenRouter API Token 计数差异**：有用户提出 Qwen 2.5 72B Turbo 模型未通过 API 返回 Token 计数的问题，并指出其他提供商功能正常。
   - 然而，OpenRouter 页面上的活动报告确实正确显示了 Token 使用情况。
- **OpenRouter 在欧洲的税务影响**：一位用户询问为什么购买 OpenRouter 积分不产生税费，而 OpenAI 或 Anthropic 的服务会增加增值税（VAT）。
   - 回复指出，计算 VAT 是用户的责任，未来的计划可能包括自动税务计算。
- **探讨 Prompt Engineering 技术**：讨论包括 few-shot prompting 策略，通过结构化的 user/assistant 角色示例被描述为非常有效。
   - 用户分享了突出 Prompt 设计最佳实践的资源和示例引用。
- **社区支持与反馈**：一般的社区互动包括 Chat UI 的故障排除和模型性能问题，反映了持续的用户参与。
   - 成员们交换了有用的链接，并建议使用工具来增强 API 集成的体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://nething.xyz">neThing.xyz - AI Text to 3D CAD Model</a>：用于 CAD 建模的 3D 生成式 AI。现在每个人都是工程师。让你的想法变为现实。</li><li><a href="https://docs.helicone.ai/getting-started/integration-method/openrouter">OpenRouter Integration - Helicone OSS LLM Observability</a>：未找到描述</li><li><a href="https://x.com/gcpweekly/status/1859644362864447564">GCP Weekly (@gcpweekly) 的推文</a>：宣布 Vertex AI 上的 Mistral AI Large-Instruct-2411 和 Codestral-2411 #googlecloud https://cloud.google.com/blog/products/ai-machine-learning/announcing-mistral-ais-large-instruct-2411-and-codes...</li><li><a href="https://github.com/Aider-AI/aider">GitHub - Aider-AI/aider: aider 是你终端中的 AI 结对编程工具</a>：aider 是你终端中的 AI 结对编程工具。通过在 GitHub 上创建一个账号来为 Aider-AI/aider 的开发做贡献。</li><li><a href="https://github.com/anthropics/anthropic-cookbook/blob/main/skills/retrieval_augmented_generation/guide.ipynb">anthropic-cookbook/skills/retrieval_augmented_generation/guide.ipynb at main · anthropics/anthropic-cookbook</a>：展示了一些使用 Claude 的有趣且有效方法的 notebooks/recipes 集合。 - anthropics/anthropic-cookbook</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>：为模型消耗转换数据</li><li><a href="https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset">hackaprompt/hackaprompt-dataset · Hugging Face 上的数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1309286882261205012)** (8 条消息🔥): 

> `访问自定义提供商密钥 (custom provider keys)` 


- **广泛请求访问自定义提供商密钥**：包括 *sportswook420* 和 *vneqisntreal* 在内的多位用户表达了获得 **custom provider keys** 访问权限的愿望。
   - 请求被反复提出，凸显了整个频道对该功能的显著兴趣。
- **多次申请激活**：*hawk1399* 和 *lokiwong* 等用户特别询问：“Hi can I get access to custom provider keys?”，强调了紧迫性。
   - 这突显了对如何访问这些密钥的明确指导需求。
- **对功能访问的热情**：*intern111_29945* 表达了他们的渴望，指出：“I'd love to get access to this feature”，增强了整体的热情。
   - 这表明社区有兴趣通过访问 custom provider keys 来增强功能。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1309270804638470235)** (81 条消息🔥🔥): 

> `Pro 用户审核流程，Gemini AI 使用体验，Perplexity 扩展程序，非编程人员的 AI 普及，AI 模型对比` 


- **Pro 用户审核困惑**：成员们讨论了获得“Pro”用户身份的要求并分享了个人经验，其中一位提到他们通过银行账户福利获得了访问权限。
   - 分享了关于如何管理订阅以及如何进入 Pro 用户专属 Discord 的说明。
- **Gemini AI 的评价褒贬不一**：一些用户对 Gemini AI 表示沮丧，报告了模型在几次交互后停止响应等问题。
   - 用户对 Gemini 和 ChatGPT 是否具有实际可比性提出了疑问，引发了关于两者差异的进一步讨论。
- **Perplexity 扩展程序的可用性**：讨论了 Safari 浏览器扩展程序的可用性，其中一个扩展可将 Perplexity 添加为搜索引擎，但另一个摘要扩展已停止维护。
   - 成员们分享了使用这些扩展的链接和技巧，并为非 Safari 浏览器推荐了替代方案。
- **AI 学习平台提案**：一位成员提议建立一个分层学习系统，通过结构化的项目和教程课程，让非编程人员更容易接触 AI 技术。
   - 该系统旨在提供循序渐进的指导，并以社区导向的方式帮助用户提升技能。
- **关于 AI 模型响应的讨论**：一位用户对 Perplexity 中的 ChatGPT-4 与原生 ChatGPT 生成的响应差异提出了疑问，随后有人对跨平台的不一致性进行了说明。
   - 另一位用户指出，输出差异可能由多种因素引起，包括 API 差异和请求的性质。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Qwen/Qwen2.5-Turbo-1M-Demo">Qwen2.5 Turbo 1M Demo - Qwen 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://x.com/apostraphi/status/1859785487369765165?s=46">Phi Hoang (@apostraphi) 的推文</a>: 🫱✨🫲
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1309292894099738659)** (15 条消息🔥): 

> `LUCA - 最近共同祖先，数字孪生，捷豹汽车销售，AI 对 Grammarly 的影响，处理技术` 


- **探索 LUCA，最近共同祖先**：Perplexity AI 重点介绍了一场关于 [LUCA](https://www.youtube.com/embed/R2Iz2f_RGNY) 的有趣讨论，强调了其在进化生物学中的重要性。
   - 鼓励用户探索这一话题，以深入了解其影响。
- **了解数字孪生**：对[什么是数字孪生](https://www.perplexity.ai/search/what-is-digital-twin-rOr00s1vSPmK1_EOv5ty7w)的探索揭示了其在各行各业的应用。
   - 数字孪生模拟现实世界的实体以进行监控和优化，引发了用户的广泛兴趣。
- **捷豹汽车销售的演变**：深入分析[捷豹汽车销售的演变过程](https://www.perplexity.ai/search/how-did-jaguar-car-sales-evolv-wYu6SoM4QCO1bjaK.8uRcw)，提供了对市场趋势的见解。
   - 讨论集中在影响长期销售业绩的策略和因素上。
- **AI 对 Grammarly 的影响**：[AI 是否影响了 Grammarly？](https://www.perplexity.ai/search/did-ai-impact-grammarly-at-all-bgRE6pmeQlmZda6x3YUVDQ) 促使用户探索 AI 进步与写作工具之间的关系。
   - 辩论指出，在这些平台中集成 AI 既有好处也有潜在的弊端。
- **适用于各种应用的处理技术**：[处理技术](https://www.perplexity.ai/search/processing-LtzV5m1_QNuP53oeafGkZA)正在讨论中，用户分享了适用于不同领域的方法。
   - 处理技术的创新正在科技和创意产业中掀起波澜。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1309535327534190623)** (2 条消息): 

> `API 站点状态` 


- **API 站点似乎运行正常**：一位成员询问 **API 站点** 是否因访问问题而宕机。
   - 另一位成员回复称该站点对他们来说运行正常，并建议重新检查或更换设备。
- **用户寻求关于 API 可用性的澄清**：用户对 **API 站点** 的可访问性表示担忧，询问其是否宕机。
   - 社区反馈表明该站点对其他人功能正常，暗示可能是特定用户的问题。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1309262501845926040)** (79 条消息🔥🔥): 

> `在 SDXL Lightning 中使用图像提示词，WebUI 参数设置，生成 Pixar 风格图像，视频微调服务，Stable Diffusion 下载与使用案例` 


- **在 SDXL Lightning 中使用图像提示词**：一位用户询问是否可以通过 Python 在 **SDXL Lightning** 中使用图像提示词，寻求关于将照片插入特定上下文的指导。
   - 另一位用户确认这确实可行，并建议通过私信交换更多信息。
- **为 12GB VRAM 设置 WebUI 参数**：讨论了在 **webui.bat** 中使用哪些额外命令来增强 **12GB VRAM** 的性能，建议包括 '--no-half-vae'。
   - 用户一致认为这足以实现最佳运行效果，无需进一步复杂化。
- **生成 Pixar 风格图像**：有人请求将公司照片转换为 **Pixar 风格**图像的方法，需要在短时间内处理大约十张肖像。
   - 成员们讨论了这项任务的可行性，指出可能没有免费服务，并建议对图像生成模型进行微调。
- **视频微调服务**：用户讨论了他们对视频微调的兴趣，并询问了用于此目的的潜在服务器或服务，参考了 **Cogvideo 模型**。
   - 有人提到，虽然 **Cogvideo 模型**在视频生成领域非常突出，但根据用户需求，其他特定的微调模型可能更受青睐。
- **Stable Diffusion 下载与使用案例**：一位新用户询问了在 PC 上下载 **Stable Diffusion** 最简单、最快的方法，并询问了相关的使用案例。
   - 另一位用户请求协助使用 Stable Diffusion 创建特定图像，同时绕过内容过滤器，这表明需要更宽松的软件选项。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1002292111942635562/1204675216773619752/1309599833958645871">Discord - 充满乐趣与游戏的群聊</a>：Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://tenor.com/view/eye-rolling-eyes-bye-annoyed-gif-13748332">翻白眼 GIF - 翻白眼 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://lu.ma/i8bow7sr">Voice & Video AI Agents Hackathon · Luma</a>：Gen AI Agents CreatorsCorner 与 AWS、Tandem、Marly、Senso 等合作，热烈欢迎个人和团队参加我们的第三届……</li><li><a href="https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation">Text-to-image</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1309353437439459379)** (63 条消息🔥🔥): 

> `Workflow 中的 Function calling，LlamaIndex 安全合规，Ollama 软件包问题，Hugging Face embedding 格式问题，LlamaParse 解析指令` 


- **简化 Workflow 中的 Function Calling**：用户讨论了在 Workflow 中使用 Function calling 的便利性，建议利用预构建的 Agents 来实现自动函数调用，从而避免编写样板代码（boilerplate code）。
   - 一位成员强调，虽然样板代码提供了更好的控制力，但使用预构建的 `FunctionCallingAgent` 可以简化流程。
- **LlamaIndex 的安全合规性**：LlamaIndex 确认符合 SOC2 标准，并澄清了 LlamaParse 和 LlamaCloud 对原始文档的处理方式。
   - LlamaParse 将文件加密保存 48 小时，而 LlamaCloud 则对数据进行 chunks 处理并安全存储。
- **LlamaIndex 中 Ollama 软件包的问题**：用户报告了 Ollama 软件包的问题，特别是最新版本中的一个 bug，该 bug 会在聊天响应期间产生错误。
   - 建议将 Ollama 降级到 0.3.3 版本，一些成员确认这解决了他们的问题。
- **Hugging Face Embedding 兼容性**：有用户担心 CODE-BERT 模型的 Embedding 输出格式与 LlamaIndex 的预期不符。
   - 用户建议在 GitHub 上提交 issue，以解决在处理模型响应时可能存在的匹配问题。
- **LlamaParse 指令的挑战**：一位成员面临 LlamaParse 不遵循特定解析指令的问题，引发了关于 `is_formatting_instruction` 等设置的讨论。
   - 社区提供了有关排查解析问题的见解，并建议查阅文档以获取正确的配置。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://*******.us-east-1.aws.endpoints.huggingface.cloud',">未找到标题</a>：未找到描述</li><li><a href="https://www.llamaindex.ai/enterprise">Enterprise — LlamaIndex - 基于企业数据构建知识助手</a>：LlamaIndex 是一个简单、灵活的框架，用于构建连接到企业数据的 LLM 知识助手。</li><li><a href="https://huggingface.co/microsoft/codebert-base">microsoft/codebert-base · Hugging Face</a>：未找到描述</li><li><a href="https://lu.ma/i8bow7sr">Voice &amp; Video AI Agents Hackathon · Luma</a>：Gen AI Agents CreatorsCorner 与 AWS、Tandem、Marly、Senso 等合作伙伴热烈欢迎个人和团队参加我们的第三届……</li><li><a href="https://github.com/run-llama/llama_parse/blob/main/llama_parse/base.py">run-llama/llama_parse 中的 llama_parse/base.py</a>：为优化 RAG 解析文件。欢迎通过在 GitHub 上创建账号来为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_index/pull/17036">由 logan-markewich 提交的 handle ollama 0.4.0 · Pull Request #17036 · run-llama/llama_index</a>：修复了 #17035 和 #17037。Ollama 更新为返回类型化响应而非普通字典，这破坏了我们附加 usage 的方式。</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/usage_pattern/">Usage Pattern - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows">Workflows - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Workflow for a Function Calling Agent - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/openai/#manual-tool-calling">OpenAI - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/multi-agent-concierge/tree/main/video_tutorial_materials">run-llama/multi-agent-concierge 中的 video_tutorial_materials</a>：使用 llama-index 进行多 Agent 编排的示例。</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/extraction/">Introduction - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1309630764568285196)** (1 messages): 

> `LlamaParse issues, Scientific article parsing, Redundant information in parsing, Document flow maintenance, Bibliography exclusion` 


- **LlamaParse 面临冗余问题**：一位成员报告了 **LlamaParse** 返回冗余信息的问题，特别是页眉和参考文献，尽管已经给出了明确的排除指令。
   - *还有其他人遇到过这个问题吗？* 该成员正在寻求解决此解析错误的建议。
- **科学论文的解析指令**：该成员提供了处理多页科学论文的详细解析指令，强调了逻辑内容流的重要性，并排除致谢和参考文献等非必要元素。
   - 指令规定仅在开头包含一次期刊标题和作者详情：*不要返回任何 References 或 Bibliography 部分*。


  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1309284198846304350)** (10 messages🔥): 

> `Podcast creation with NotebookLM, YouTube videos on AI and Robotics, Feature requests for Producer Studio, Translating NotebookLM audio, Social media content generation` 


- **NotebookLM 播客功能成功案例**：一位用户分享了他们使用 NotebookLM 创建播客的历程，并分享了名为 ["Google Chrome: Browser Wars and AI Automation Frontiers"](https://youtu.be/AFm56rtJ7g8) 的视频，其中包含完整的制作元素。
   - 他们表达了希望实现其 [Producer Studio 功能请求](https://discord.com/channels/1124402182171672732/1300797611015671818/1300797611015671818) 的愿望。
- **发布关于 AI 和机器人技术的新视频**：[一段新的 YouTube 视频](https://youtu.be/iFmF64w8h3s) 标题为 "AI & Humanoid Robotics: The Future Is Now"，深入探讨了 AI 和机器人技术的进展，研究了 OpenAI 和 NVIDIA 等行业领导者的贡献。
   - 创作者正在寻求关于改进其内容生成的反馈。
- **通过洞察探索 AI 的未来**：另一位成员分享了一段名为 ["Want to Know the Future of AI? Watch This Now"](https://youtu.be/UX7evZec8Os) 的视频，讨论了《Genesis: Artificial Intelligence, Hope, and the Human Spirit》一书。
   - 本集强调了围绕 AI 影响的启发性讨论。
- **功能支持的互动**：号召社区通过在 Discord 上的共享链接添加回应或消息来支持某项功能。
   - 鼓励积极参与以影响功能开发。
- **语言翻译查询**：在关于语言支持的讨论中，一位用户询问了将 NotebookLM 音频翻译成德语的问题，重点关注德语特有的主题。
   - 另一位用户对意大利语翻译表达了类似的兴趣，展示了对多语言支持的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/AFm56rtJ7g8">Google Chrome: Browser Wars and AI Automation Frontiers</a>：探索浏览器、AI 和自动化碰撞的关键技术战场。每一集都揭示了技术生态系统隐藏的影响...</li><li><a href="https://youtu.be/UX7evZec8Os">Want to Know the Future of AI? Watch This Now</a>：在本集播客中，我们深入探讨了 Henry Kissinger, Craig Mun... 所著的启发性书籍《Genesis: Artificial Intelligence, Hope, and the Human Spirit》。</li><li><a href="https://youtu.be/iFmF64w8h3s">AI &amp; Humanoid Robotics: The Future Is Now - Advances by Figure, OpenAI, Anthropic, and NVIDIA</a>：在我们的最新视频中深入探索 AI 和机器人技术的前沿世界！我们正在探索行业领导者的惊人进展：Figure：发现...</li><li><a href="https://www.tiktok.com/@studyrottbs">TikTok - Make Your Day</a>：未找到描述</li><li><a href="https://www.instagram.com/top.shelf.podcast/profilecard/?igsh=NTc4MTIwNjQ2YQ==">Top.Shelf.Podcast (&#064;top.shelf.podcast) &#x2022; Instagram 照片和视频</a>：3 位粉丝，14 位关注中，7 条帖子 - 查看 Top.Shelf.Podcast (&#064;top.shelf.podcast) 的 Instagram 照片和视频
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1309286039680188446)** (46 条消息🔥): 

> `Podcast API 替代方案, 音频语言支持, Podcast 生成限制, NotebookLM 使用问题, NotebookLM 中的 Retrieval-Augmented Generation` 


- **探索 Podcast API 替代方案**：一名成员建议尝试 [Podcastfy.ai](https://www.podcastfy.ai)，将其作为 NotebookLM 的 Podcast 功能的开源替代 API。
   - 感兴趣的用户正在考虑它与现有选项的对比。
- **主持人的语言支持问题**：一名用户表达了沮丧，因为主持人只说英语，尽管之前曾成功尝试过法语。
   - 另一名成员提到，有时不同的语言会对提示词做出响应，这表明可能存在不一致性。
- **Podcast 生成限制说明**：用户注意到每个账户有 100 个 Podcast 的限制，并推测每日生成上限为 20 个。
   - 他们确认在删除旧的 Podcast 后，生成按钮会重新出现，进一步明确了限制规则。
- **Gmail 账户访问问题**：一名用户在第二个 Gmail 账户上遇到了访问限制，通过进行之前未考虑到的年龄验证解决了该问题。
   - 这突显了用户在账户设置中可能遇到的潜在障碍。
- **了解 NotebookLM 的功能**：一名用户询问 NotebookLM 在一个会话中进行多次查询后是否能保持质量。
   - 另一名用户深入解释了 NotebookLM 如何利用 'Retrieval-Augmented Generation' 来提高响应准确性和引用追踪。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.podcastfy.ai,">未找到标题</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/c2d1d84d-6b57-4a11-92d1-fbcd18109e38">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=VDrpKid5a04">Hawk Talk Podcast - Thursday Night LIVE | TELESTAI Updates! | BTC TO 100K!</a>: 欢迎来到 Hawk Crypto and Tech 频道！➡️ Twitter: https://twitter.com/HawkCryptoTech ➡️ Misfit Mining Discord: https://discord.gg/XUvGgC5thp ➡️...</li><li><a href="https://youtu.be/iFmF64w8h3s">AI &amp; Humanoid Robotics: The Future Is Now - Advances by Figure, OpenAI, Anthropic, and NVIDIA</a>: 在我们最新的视频中深入探索 AI 和机器人技术的前沿世界！我们正在探索行业领导者的惊人进展：Figure: 探索...</li><li><a href="https://www.instagram.com/top.shelf.podcast/profilecard/?igsh=NTc4MTIwNjQ2YQ==">Top.Shelf.Podcast (&#064;top.shelf.podcast) &#x2022; Instagram 照片和视频</a>: 3 位粉丝，14 位关注中，7 条帖子 - 查看 Top.Shelf.Podcast (&#064;top.shelf.podcast) 的 Instagram 照片和视频</li><li><a href="https://bsky.app/profile/chrismoranuk.bsky.social/post/3lbjr2ih2cs2k">Chris Moran (@chrismoranuk.bsky.social)</a>: 关于 NotebookLM 及其在新闻业用途的一个短贴。首先且最明显的是，如果你是一名负责特定领域的记者，经常需要查阅报告、研究或调查证据，NotebookLM 是 ...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1309382323724222494)** (19 messages🔥): 

> `OpenAI 的版权诉讼、10B 模型的去中心化训练、Anthropic 与 AWS 的合作、对新 AI 模型的预期、AI 开发中的怀疑论` 


- **OpenAI 的数据删除失误**：*《纽约时报》*和*《每日新闻》*的律师正在起诉 OpenAI，指控 **OpenAI 工程师在他们花费超过 150 小时**搜索相关数据后，**意外删除了**与版权诉讼相关的数据。
   - 11 月 14 日，一台虚拟机上的所有搜索数据被擦除，引发了对案件潜在影响的担忧，详见[法庭信函](https://storage.courtlistener.com/recap/gov.uscourts.nysd.612697/gov.uscourts.nysd.612697.328.0.pdf)。
- **Prime Intellect 发布 INTELLECT-1**：Prime Intellect 分享了跨越不同大洲完成的**首个 10B 模型的去中心化训练**，并表示与 @arcee_ai 的后训练（post-training）正在进行中。
   - 预计在一周内发布完整的开源版本，提供对基础模型和 Checkpoints 的访问，并邀请各方合作**构建开源 AGI**。
- **Anthropic 达成重大 AWS 合作伙伴关系**：Anthropic 宣布扩大与 AWS 的合作，获得来自 Amazon 的 **40 亿美元投资**，确立 AWS 为其主要的云服务和训练合作伙伴。
   - 此次合作旨在增强 Anthropic 开发其 AI 技术的能力，详见其官方[新闻稿](http://anthropic.com/news/anthropic-amazon-trainium)。
- **对新 AI 模型的怀疑态度**：针对**在 1T tokens 上训练的 10B 模型**的预期引发了担忧，有观点认为它可能无法与 LLaMA 等现有模型进行有效竞争。
   - 一些人认为它必须成为“史上最强模型”才有机会，但在当前环境下，对新尝试的怀疑和否定非常普遍。
- **应对 AI 开发中的怀疑论**：讨论强调了 AI 开发中**普遍存在的怀疑论**，例如 Olmo 等新模型也曾面临质疑。
   - 尽管存在负面情绪，一些参与者仍主张顶住批评意见，专注于开发工作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/AnthropicAI/status/1859964653486612585">Anthropic (@AnthropicAI) 的推文</a>：我们正在扩大与 AWS 的合作。这包括来自 Amazon 的一笔新的 40 亿美元投资，并确立 AWS 为我们的主要云和训练合作伙伴。http://anthropic.com/news/anthrop...</li><li><a href="https://x.com/PrimeIntellect/status/1859923050092994738">Prime Intellect (@PrimeIntellect) 的推文</a>：我们做到了 —— 首个 10B 模型的去中心化训练已完成！训练跨越美国、欧洲和亚洲 🌐 与 @arcee_ai 的后训练正在进行中，完整的开源版本即将发布...</li><li><a href="https://techcrunch.com/2024/11/20/openai-accidentally-deleted-potential-evidence-in-ny-times-copyright-lawsuit/">OpenAI 在纽约时报版权诉讼中意外删除潜在证据 | TechCrunch</a>：在法庭文件中，《纽约时报》和《每日新闻》的律师表示，OpenAI 意外删除了针对其的潜在证据。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/)** (1 messages): 

420gunna: https://x.com/reach_vb/status/1859868073903423821
multimodal encoder 兄弟们 ✊
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1309263006156329050)** (9 messages🔥): 

> `AI 中的 Flops, 来自 Base Model 的魔力, 对 Amanda 帖子的回应, Scale 研究, 相关研究论文` 


- **低 Flops 引发大疑问**：一名成员对**我们的 Flops 过低**表示担忧，并建议一种**“从 Base 获取魔力”**的方法可能会带来有用的见解。
   - 他们强调了在 AI 开发中平衡理论与实践层面的重要性。
- **受 Amanda 回应启发的新帖构思**：一名成员提到，他们可能会在即将发布的帖子中对 **Amanda 在 Lex 上的帖子**做出回应，显示出对该话题日益增长的兴趣。
   - 这表明了与社区讨论的积极互动以及想法的演变。
- **Scale 研究的发现**：一名成员评论了 **Scale 研究**的存在，对这一领域被认可为研究方向表示出好奇和惊讶。
   - 他们将这一发现与近期围绕 AI 模型性能和优化的对话联系起来。
- **相关论文引发兴趣**：一名成员引用了一篇最近引起他们注意的[相关论文](https://arxiv.org/pdf/2410.03717v1)，增强了关于 AI 建模技术的讨论。
   - 他们认为这篇论文与当前的对话高度相关，表明希望对该主题进行更深入的探索。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1309576399522234378)** (5 messages): 

> `CamelAIOrg 账号问题, OASIS 社会模拟项目, 客户服务担忧` 


- **CamelAIOrg 账号被封/无响应**：**CamelAIOrg** 的账号因未知原因被 **OpenAI** 停用，可能与其近期使用了 100 万个 **Agent** 的 **OASIS 社会模拟项目**有关，详情见其 [GitHub 页面](https://github.com/camel-ai/oasis)。
   - 他们寻求帮助，但 **5 天**内未收到任何回复，导致 **20 多名社区成员**在等待 **API key**。
- **对 OpenAI 对待方式的沮丧**：一名用户表达了对在一家公司花费了**数十万美元**却感到被亏待的沮丧。
   - 另一名用户评论了响应的缓慢，称“你至少会在几天后收到一段 **LLM slop** 重复你的问题，然后就被无视了。”
- **对 Tulu 3 发布的推测**：频道内对 **Tulu 3** 是否会出现某种结果存在好奇。
   - 一名成员指出，如果真是那样，将会令人惊讶。



**提到的链接**：<a href="https://x.com/guohao_li/status/1860016322358165867">来自 Guohao Li (Hiring!) 🐫 (@guohao_li) 的推文</a>：@OpenAI 因未知原因停用了我们 @CamelAIOrg 组织的账号。这可能与我们最近运行的拥有 100 万个 Agent 的 OASIS 社会模拟项目有关，但我不确定：htt...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1309349316816670805)** (7 messages): 

> `黑市数据交易, 购买 Benchmark, 实验室快速运转` 


- **促成黑市数据交易**：一名成员幽默地提到，有人请求他促成一桩互联网人士与 AI 实验室之间的**黑市**数据交易。
   - 在 AI 领域看到这种非常规的交易提议*依然非常搞笑*。
- **对个人 Benchmark 交易的推测**：另一名成员推测有人试图购买 **Xeos** 的个人 **Benchmark** 数据，称“该死，他们已经想买 Xeos 的个人 Benchmark 了吗”。
   - 这突显了 AI 社区对个人 **Benchmark** 日益增长的兴趣。
- **实验室动作很快**：针对黑市提议，一名成员指出**实验室动作很快**，暗示了竞争激烈且快速变化的格局。
   - 这一评论表明在 AI 行业中，获取数据和 **Benchmark** 存在着普遍的紧迫感。


  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1309314969698963477)** (3 messages): 

> `梗图格式、语言模型、创意内容` 


- ****梗图格式赏析****：一位成员表达了对特定梗图格式的喜爱，称自己“对这种梗图格式毫无抵抗力”。
   - 他们分享了该梗图的链接，表明它在社区中引发了快乐，并强调了其[来源](https://pbs.twimg.com/media/GJr5C5oWsAAt0cv?format=jpg&name=900x900)。
- ****Discord 中的 Yeeting****：另一位成员仅以充满热情的“Yeet”作为回应，为轻松的对话增色。
   - 这种简短的互动可能反映了频道内整体俏皮的基调。
- ****语言模型脑腐 (Brainrot)****：一位成员幽默地询问对话是否正转向“语言模型 iPad 孩子脑腐内容”。
   - 这一评论展示了对语言模型讨论本质的奇思妙想，并对其中的荒谬感到有趣。


  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1309567702330773545)** (4 messages): 

> `Tulu 3 论文、On-policy 与 Off-policy DPO、Online DPO 性能` 


- **澄清 Tulu 3 中的 On-Policy DPO**：针对 Tulu 3 论文展开了讨论，质疑其中描述的 DPO 方法是否真正属于 On-policy，因为模型的策略在训练期间不断演变，通过从初始 Base Model 采样会导致 Off-policy 行为。
   - 成员们辩论认为，如第 8.1 节所述的 *Online DPO* 更符合 On-policy 推理，因为它在每个训练步骤都会为 Reward Model 采样补全结果。
- **Online DPO 的挑战**：一位参与者强调，在他们的实验过程中，实际的“Online DPO”过于**难以调优 (finicky)**，无法有效运行。
   - 另一位成员推测，Online DPO 内部剧烈的策略转变可能是导致其性能问题的原因之一。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

markus_41856: https://lu.ma/i8bow7sr
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1309390006250373191)** (5 messages): 

> `AMD GPU 上的 Triton、L2 缓存的 Swizzle、优化 Triton Kernel、内存访问效率、MLIR 分析` 


- **Triton 展示 AMD GPU 优化**：在一段名为“AMD GPU 上的 Triton”的 [YouTube 视频](https://youtu.be/Lbm08twNTAQ?si=lR-4YeLhVxWcPw8R)中，Lei Zhang 和 Lixun Zhang 讨论了围绕 Chiplets 和各种指令的巧妙优化技术。
   - 一位参与者指出，提到的 **Swizzle** 技术与 **L2 缓存**有关，为优化讨论增添了价值。
- **Triton Kernel 优化指南**：一篇关于[优化 Triton Kernel](https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html)的文章概述了 Triton Kernel 优化的步骤，并强调了与 HIP 和 CUDA 的对比。
   - 文章强调 **全局内存具有高延迟**，而 LDS 和寄存器提供更快的访问速度，突显了高效内存访问策略的必要性。
- **在内存访问中使用 Pad 而非 Interleave**：一位成员提到，在他们的分析中，观察到在 Triton Kernel 优化中使用了 **pad** 而非 **交错布局 (interleave layout)** 进行寻址。
   - 他们目前正在分析生成的 **MLIR**，以寻求对这一优化选择的更深层次见解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/Lbm08twNTAQ?si=lR-4YeLhVxWcPw8R">Triton on AMD GPUs</a>: Lei Zhang 和 Lixun Zhang 讨论了 AMD 对 Triton 的支持。本次演讲展示了一些围绕 Chiplets 以及指令集的非常巧妙的优化技术...</li><li><a href="https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html">Optimizing Triton kernels — ROCm Documentation</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1309443747607482369)** (4 条消息): 

> `Torch Inductor 编译错误，自定义层 .to() 行为` 


- **Torch Inductor 在编译期间引发错误**：在使用 **LoRA** 实现时遇到了一个错误，触发了 `BackendCompilerFailed` 异常，具体表现为与 `inductor` 中量化行相关的 `AttributeError: 'float' object has no attribute 'meta'`。
   - 该问题似乎与 **torch.compile** 的交互有关，用户无法在大型模型上下文之外复现该问题，因此寻求潜在的解决方案。
- **自定义层的 .to() 调用在模型级别失败**：一位用户指出，他们的自定义 `.to()` 实现可以在单个层上工作，但在应用于整个模型时不会触发，这表明 `nn.Module` 的功能可能存在局限性。
   - 另一位成员确认 **nn.Module.to()** 不会递归地在子模块上调用 `.to()`，并建议如果需要，应直接在子模块上调用 `.to()`。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/a6344c8bcd22798987087244e961cdc0cbf9e9df/torch/_inductor/fx_passes/quantization.py#L1469">pytorch/torch/_inductor/fx_passes/quantization.py (GitHub)</a>：Python 中具有强 GPU 加速的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/a6344c8bcd22798987087244e961cdc0cbf9e9df/torch/nn/modules/module.py#L1321-L1344">pytorch/torch/nn/modules/module.py (GitHub)</a>：Python 中具有强 GPU 加速的 Tensor 和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1309512438202499162)** (1 条消息): 

> `FlashAttention，全局 vs 局部指数和 (Exp Sum)` 


- **关于 FlashAttention 近似技术的咨询**：一位成员询问是否存在相关论文，探讨如何根据 **QK^T 分块 (tiling block)** 中的局部和来近似 **全局指数和 (global exponential sum)**。
   - 该问题突出了对计算 Attention 分数效率的持续探索。
- **局部与全局和的重要性**：讨论强调了理解 **局部指数和** 与 **全局指数和** 之间的关系在优化 FlashAttention 机制中的重要性。
   - 成员们建议，这种理解可能会带来改进的近似技术和整体效率的提升。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

platers: https://research.character.ai/optimizing-ai-inference-at-character-ai-part-deux/
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1309631661331447828)** (1 条消息): 

> `Flash Attention 优化，Character AI 职位空缺` 


- **Flash Attention 优化备受关注**：一位成员重点介绍了 Jay 讨论 **基于 Flash Attention 的优化** 的帖子，展示了该领域的最新进展。
   - 该帖子因其为正在进行的关于 Attention 机制的讨论提供了宝贵见解而获得好评。
- **Character AI 招聘 ML Systems 研究员**：Character AI 宣布了一个 **ML Systems 研究员** 的职位空缺，邀请候选人通过其 [招聘页面](https://jobs.ashbyhq.com/character/3ab40c3d-63bd-4634-a126-5a3d25d3263b) 进行申请。
   - 该职位旨在加强其专注于将机器学习系统集成到项目中的团队。



**提到的链接**：<a href="https://jobs.ashbyhq.com/character/3ab40c3d-63bd-4634-a126-5a3d25d3263b">Research Engineer, ML Systems (All Industry Levels)</a>：作为 ML Systems 团队的研究工程师加入我们，你将致力于前沿的 ML 训练和推理系统，优化 GPU 集群的性能和效率，并开发...

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1309397531079479366)** (2 条消息): 

> `ldmatrix 分块大小，torch 调度至 triton` 


- **ldmatrix 分块 (tile) 大小疑问**：一位成员询问 **ldmatrix** 是否仅支持加载 **8x8 分块**，因为他们在文档中仅发现了针对 **.m8n8** 的示例。
   - 这引发了关于 **ldmatrix** 框架内是否支持其他分块大小的不确定性。
- **Torch 在矩阵运算中调度至 Triton**：另一位成员询问 **torch** 是否针对某些矩阵大小和操作调度 (dispatch) 至 **triton**。
   - 这可能表明在特定操作中存在一个优化层，可能会提升性能。

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1309276126472048841)** (4 messages): 

> `量化方案基准测试，权重张量分布，BNB NF4 与 Marlin 性能对比，投影中的权重离群值` 


- **量化基准测试的距离度量**：*在对量化方案进行基准测试时*，一位成员询问了合适的距离度量指标，并建议使用**均方差 (mean square difference)**。
   - 讨论中提到了量化评估领域中使用的不同指标。
- **BF16 张量中奇怪的权重离群值**：基准测试显示 Llama 70B 的 **bf16 权重张量**均值约为 **2e-06**，方差为 **0.0001**，但却出现了如 **85** 和 **-65** 这样的离群值。
   - *这很奇怪*，因为权重预期应符合正态分布，这引发了对其分布特性的质疑。
- **BNB NF4 在基准测试中超越 Marlin**：研究发现 **bnb nf4** 量化方案表现异常出色，在特定基准测试中超过了 **Marlin**，尽管后者通常被认为非常高效。
   - 成员们对 bf16 基准线与 Marlin 性能之间观察到的**显著差异**感到惊讶。
- **在投影中观察到权重离群值**：成员们讨论了在下投影 (down projections) 过程中**权重离群值**尤为明显，这进一步证实了对其存在的担忧。
   - 一项观察指出，这些离群值广泛存在，不仅限于投影层，表明这是一个更广泛的问题。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1309563686092017735)** (2 messages): 

> `Vercel 的 v0 工具，系统提示词泄露，AI 编程辅助` 


- **Vercel v0 系统提示词泄露**：一位成员泄露了 [Vercel v0 工具的系统提示词](https://www.reddit.com/r/LocalLLaMA/comments/1gwwyia/leaked_system_prompts_from_v0_vercels_ai/?share_id=F6bbP1QiREa6fpafBWHsn)，声称它们 100% 真实，并揭示了 reflection method 的有趣集成。
   - 他们提供了完整 [system prompt](https://github.com/2-fly-4-ai/V0-system-prompt/blob/main/v0-system-prompt) 的链接以及一个特定的 [feature file](https://github.com/2-fly-4-ai/V0-system-prompt/blob/main/thinking-feature24) 以供进一步深入了解。
- **用户赞扬 AI 编程助手**：另一位成员分享了使用 Vercel v0 工具的积极体验，强调它简化了多文件编辑并提供了一键部署功能。
   - 他们评论道：*“真心值！”*，并强调了人们付费购买的价值所在——即一个提示词和增强的 AI 辅助。



**提到的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1gwwyia/leaked_system_prompts_from_v0_vercels_ai/?share_id=F6bbP1QiREa6fpafBWHsn&utm_content=2&utm_medium=android_app&utm_name=androidcss&utm_source=share&utm_term=1">Reddit - 深入探索</a>：未找到描述

  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/1309390004753010688)** (1 messages): 

> `剪枝技术，模型效率论文，数据依赖策略，LLM 的工业应用` 


- **咨询 LLM 最新的剪枝技术**：一位成员询问了目前在**工业环境**中用于大语言模型 (LLM) 的最新剪枝、模型效率论文及策略的建议。
   - 他们提到使用了 **What Matters in Transformers** 论文，但发现该技术是**数据依赖 (data dependent)** 的。
- **寻求非数据依赖技术**：在咨询中，该成员表示需要模型剪枝和效率提升方面的**非数据依赖技术**。
   - 这凸显了对不依赖特定数据集策略的需求缺口，此类策略可以增强模型在各种应用中的效率。


  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1309333452939853826)** (4 messages): 

> `GPT-2 Training Method, Discord Bot Integration, OpenCoder Paper Filtering Approach` 


- **五分钟训练 GPT-2**：一位成员分享了一个 [GitHub Gist](https://gist.github.com/charlesfrye/5f299d80ba2b2ae4ec81d672b4c3246f)，概述了如何**免费在五分钟内训练 GPT-2**，并附带了一个辅助该过程的函数。
   - 关联的图片链接提供了该 Gist 内容的视觉呈现：
![Gist Image](https://github.githubassets.com/assets/gist-og-image-54fd7dc0713e.png)
。
- **针对 GPT-2 训练的 Discord Bot 增强**：讨论了让他们的 **Discord bot** 也能以流线化方式执行 GPT-2 训练的能力。
   - 这表明对于那些使用该 Bot 进行 AI 相关任务的用户来说，用户体验有潜在的提升。
- **冷启动缓存技巧**：<@marksaroufim> 指出，提到的**缓存技巧**对于**冷启动（cold starts）**提高性能可能非常重要。
   - 这一见解建议在需要快速响应的场景下初始化 Bot 时，可以增强运行效率。
- **OpenCoder 论文的过滤方案**：一位成员讨论了 **OpenCoder 论文**，该论文提出了一种针对从公共仓库抓取的文件进行三层过滤的系统，包括通用文本文件、代码文件和特定语言的代码文件。
   - 该成员质疑了这种方法的可行性，并就定义 **CUDA 特定规则**以消除低质量文件寻求建议。



**提及的链接**：<a href="https://gist.github.com/charlesfrye/5f299d80ba2b2ae4ec81d672b4c3246f">Train GPT-2 in five minutes -- for free!</a>：免费在五分钟内训练 GPT-2！GitHub Gist：即时分享代码、笔记和代码片段。

  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1309437871190573068)** (1 messages): 

> `NPU Acceleration, Executorch, Qualcomm NPUs` 


- **寻求 NPU 加速方案**：一位成员询问了支持 **NPU 加速**的库或运行时，并提到他们知道 **Executorch** 为 **Qualcomm NPUs** 提供了一些支持。
   - 他们表示希望了解市场上可用的其他潜在解决方案。
- **对其他 NPU 库的兴趣**：该成员很好奇除了 **Executorch** 之外，是否还存在其他框架可以有效地利用 **NPU 加速**。
   - 他们开启了关于此话题的社区讨论。


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1309344665656361081)** (14 messages🔥): 

> `Cohere API front-end, Chat history editing feature` 


- **请求兼容 Cohere API 的前端**：一位用户询问是否有兼容 **Cohere API** 的前端，允许用户像 Claude 或 GPT 中的功能一样编辑他们的聊天历史。
   - 他们强调需要一个**编辑按钮**来修改输入，以避免因错误而不得不重新开始对话。
- **关于编辑交互的澄清**：另一位用户澄清说，该请求是希望编辑内容成为聊天历史的一部分，初始用户确认了这一点。
   - 他们重申 **Cohere 网站**的聊天和 playground-chat 页面缺少**编辑选项**，并强调了其对用户体验的重要性。



**提及的链接**：<a href="https://imgur.com/a/1EpveO5">imgur.com</a>：在 Imgur 发现互联网的魔力，一个由社区驱动的娱乐目的地。通过幽默的笑话、流行的迷因、有趣的 GIF、励志的故事、病毒式视频等来振奋你的精神。

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 messages): 

jilldomi_48896: 这太酷了

https://docs.cohere.com/page/sql-agent-cohere-langchain

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1309332308683395172)** (4 messages): 

> `SDXL 类型转换问题，回归担忧，中间转换策略` 


- **SDXL 的类型转换问题导致 Benchmark 速度减慢一倍以上**：成员们注意到，在更新 **#7644** 之后，**SDXL** 在 **CI tinybox green** 上不再转换为 **half** 类型，导致 Benchmark 速度减慢了 **2 倍**以上。
   - *这是否是因为之前转换不正确而进行的有意更改？*
- **对潜在回归（Regression）的不确定性**：一位成员对更新 **#7644** 后类型转换失效是否属于回归表示担忧。
   - 他们寻求澄清，不确定 Benchmark 变慢是否意味着性能上的倒退。
- **关于中间转换的更新建议**：另一位成员建议，中间类型转换（intermediate casting）应由 **input dtypes** 决定，而不是由设备决定，并主张采用**纯函数式（pure functional）**方法。
   - 他们提出了一个类似于 **Stable Diffusion** 中 **fp16** 的参数，以更有效地处理模型和输入的类型转换。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1309329373228433480)** (5 messages): 

> `自定义 Kernel 函数支持，Tinygrad 介绍，Tensor Stride 和 View Hopping，objc_id 和 ctypes 行为，ops_metal.py 中的函数调用行为` 


- **自定义 Kernel 函数已被移除**：一位用户询问是否仍支持**自定义 Kernel 函数**，但经确认，这些函数已从仓库的最新版本中移除。
   - *George Hotz* 建议通常有更好的方法来实现预期结果，而无需破坏抽象。
- **分享有用的 Tinygrad 入门介绍**：一位用户发现了一个 **Tinygrad** 的入门介绍并分享了 [YouTube 链接](https://youtu.be/0ncx4H0YmK0)，这可能对初学者有所帮助。
   - 该资源旨在帮助新用户更轻松地掌握 Tinygrad 的基础知识。
- **关于 Tensor Strides 的讨论**：一位参与者推测了对**巨大 Tensor** 的需求以及将 Strides 视为循环的可能性，暗示了之前 View Hopping 的解决方案。
   - 他们还讨论了硬件在处理 **32b vs 64b** 时的效率，建议对 Tensor 进行内部拆分以获得更好的性能。
- **ops_metal.py 中的重写行为**：在 `ops_metal.py` 中，一位用户提出了关于为 **objc_id** 类重写 **hash** 和 **eq** 函数以有效管理响应类型的问题。
   - 他们注意到，使用 `objc_id=ctypes.c_void_p` 会导致他们的 Jupyter kernel 崩溃，这与 TinyGrad 的实现不同。



**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_metal.py">tinygrad/tinygrad/runtime/ops_metal.py at master · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 Micrograd？你会爱上 Tinygrad！❤️ - tinygrad/tinygrad

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1309286850682425365)** (7 messages): 

> `Mojo 与 Python 集成，Mojo 中的 Async 能力，性能挑战，Mojo 增强路线图，Mojo 中的参数化和 Traits` 


- **Mojo 可能很快就能调用 Python 函数**：[Mojo 路线图](https://link.to/roadmap)上已列出允许从 Python 导入 Mojo 包并调用 Mojo 函数的功能，从而提供增强的互操作性。
   - 社区成员表示该功能正在积极开发中，对于不需要顶级性能的用户，已经有一些初步方法可用。
- **Async 事件循环需要额外设置**：尽管初步支持异步分配的数据结构，但 Mojo 中的异步操作仍需要创建一个事件循环来有效管理状态机。
   - 计划允许用户在不需要时编译掉 Async 运行时，从而简化潜在的性能优化。
- **使用多线程进行集成是可行的**：一位用户分享了一种解决方法，即 Mojo 和 Python 通过队列进行通信，从而允许在 Python 中通过多线程进行异步交互。
   - 虽然这种方法在某些情况下有效，但对于简单的用例来说可能显得过于复杂，导致其他人更倾向于官方解决方案。
- **用户对 Mojo 特性的兴趣**：一位成员表示，Mojo 的主要用例在于为 C/C++/Rust 提供一种类似 Python 的替代方案，强调加速缓慢的过程。
   - 他们的关注点在于基本特性，如参数化 Traits 和 Rust 风格的 Enums，这表明他们希望在 Mojo 类之上进行基础性的增强。

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1309551713182552155)** (3 messages): 

> `HF transfer, Download Speed Improvements, Internet Connection Impact` 


- **HF Transfer 加速模型下载**：在 **torchtune** 中加入 [HF transfer](https://github.com/pytorch/torchtune/pull/2046) 显著缩短了模型下载时间，**Llama 8B** 的下载时间从 **2分12秒降至 32秒**。
   - 用户可以通过运行 `pip install hf_transfer` 并为非 nightly 版本添加标志 `HF_HUB_ENABLE_HF_TRANSFER=1` 来启用它。
- **家庭网络速度下的混合结果**：一位用户询问了 HF transfer 在普通家庭网络连接上的有效性，因为过去的经验表明它主要使高速网络环境受益。
   - 作为回应，另一位用户强调了他们在家里使用该功能的积极结果，指出通过 HF transfer 一次下载一个文件可以达到超过 **1GB/s** 的速度。
- **使用 HF Transfer 的优化下载策略**：标准下载方法尝试以约 **50MB/s** 的速度同时获取 **8B 模型** 的所有文件，这可能效率低下。
   - 相比之下，HF transfer 一次下载一个文件的方法优化了带宽利用率，从而实现了更快的下载速度。



**提及的链接**：<a href="https://github.com/pytorch/torchtune/pull/2046">Use hf transfer as default by felipemello1 · Pull Request #2046 · pytorch/torchtune</a>：上下文 此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档，还是其他（请在此处添加） pip install huggingface_hub[hf_transfer] HF_HUB_ENABLE_H...

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1309422527478501467)** (4 messages): 

> `AI model evaluations, Statistical theory in model comparisons, Central Limit Theorem application, AI research community response` 


- **新论文探讨评估一致性**：Anthropic 最近的一篇 [研究论文](https://arxiv.org/abs/2411.00640) 讨论了 AI 模型评估的可靠性，质疑性能差异是真实的，还是源于问题选择中的随机运气。
   - 这项研究鼓励 AI 研究社区采用更严谨的统计报告方法。
- **误差线引发争议**：一位社区成员讽刺地评论了该论文对误差线（error bars）的关注，暗示这还不够具有开创性，不值得进行广泛研究。
   - 这种批评反映了一种更广泛的情绪，即基础统计工具在这一领域经常被忽视。
- **统计方法的惊人缺乏**：另一位成员指出模型评估中惊人地缺乏统计方法，强调了该领域需要稳健的方法论。
   - 这表明了关于提高有效评估 AI 模型标准的持续对话。



**提及的链接**：<a href="https://www.anthropic.com/research/statistical-approach-to-model-evals">A statistical approach to model evaluations</a>：Anthropic 关于如何应用统计学来改进语言模型评估的研究论文。

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1309289568855523429)** (5 messages): 

> `Hackathon Location, Team Registration Confirmation` 


- **黑客松 100% 在线进行**：针对关于黑客松地点的查询，澄清了该活动**完全在线进行**，缓解了潜在的物流担忧。
   - 这确保了参与者可以从任何地方加入，而无需关注物理场地。
- **团队注册现在发送确认邮件**：提到团队注册现在将通过 Google Forms 向第一个字段中输入的邮箱发送**确认邮件**，而不是发送给每个团队成员。
   - 这一变化简化了流程，确保至少有一名团队成员收到必要的确认。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/)** (1 messages): 

danman117: 本周 Percy Liang 的演讲非常精彩！
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1309297211875790911)** (4 messages): 

> `Posting Job Ads, Desktop App Release, Exponent Demo, Open Source Devin, Windsurf Feedback` 


- **招聘广告需要更合适的频道**：讨论了招聘信息发布的合适频道，并提醒根据社区指南，应将其发布至 <#1210088092782952498>。
   - *我们希望支持求职者，但不希望 general 频道充斥着广告*。
- **咨询桌面端应用发布**：一位新成员在加入等待名单后，对桌面端应用的发布时间表表示好奇。
   - 目前尚未分享具体的发布日期，表明其仍不确定。
- **探索 Exponent 演示**：一位成员分享了他们使用 Exponent 进行演示的经验，并提到他们仍在实验其功能。
   - 对 Windsurf 给予了正面反馈，强调了其有效性。
- **讨论中的开源版 Devin**：有人注意到目前存在一个开源版本的 Devin，成员们一直在探索，但发言者尚未尝试。
   - 这反映了社区对实验社区驱动工具的持续兴趣。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1309563555204567050)** (1 messages): 

> `Installing O1, Using Groq API, Free APIs on Linux` 


- **在 Linux 上安装 O1 的挑战**：一位成员表示在 **Linux** 系统上安装 **O1** 存在困难，目前尚未找到使其运行的方法。
   - *他们正在寻求有关安装问题的潜在解决方案或变通方法的建议。*
- **探索免费 API**：有人询问在 Linux 上将 **Groq API** 或任何其他免费 API 与 O1 结合使用的可行性。
   - *讨论突显了该成员对在项目中最大限度利用免费资源的兴趣。*


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1309560874666561656)** (3 messages): 

> `VLMs for Invoice Processing, DSPy Support for VLMs, Complexity in Project Development` 


- **VLM 在发票处理中备受关注**：一位成员正在探索将 **VLM** 用于一个**高风险的发票处理项目**，并寻求关于 **DSPy** 如何增强其针对专门子任务的提示词（prompt）的指导。
   - 提到了 DSPy 最近对 **VLM**（特别是 **Qwen**）的支持。
- **使用 VLM 测试 DSPy**：一位成员建议尝试将 DSPy 用于 VLM，并分享了他们在**视觉分析项目**中的成功经验，指出 **CoT 模块**在处理图像输入时表现出色。
   - 他们尚未测试优化器（optimizers），表明还有更多探索空间。
- **从简单开始以实现项目成功**：另一位成员强调在逐渐增加项目复杂性之前，应先从简单的任务开始，这强化了 DSPy 易于上手的理念。
   - *“如果你从简单开始并随着时间的推移增加复杂性，这并不难！”* 传达了一种鼓励实验的情绪。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1309507547102711819)** (1 messages): 

> `INTELLECT-1, Decentralized training, Open-source AGI, Fine-tuning with Axolotl` 


- **INTELLECT-1：去中心化训练的里程碑**：Prime Intellect 宣布完成 **INTELLECT-1**，这标志着有史以来首次跨越多个大洲的 **10B 模型**的**去中心化训练**。
   - 与 **@arcee_ai** 合作的训练后阶段（Post-training）正在进行中，完整的**开源发布**计划在约一周后进行。
- **扩大去中心化训练的规模**：Prime Intellect 声称，与之前的尝试相比，新模型在去中心化训练能力上实现了 **10 倍**的提升。
   - 这项工作邀请任何感兴趣的人加入并为构建**开源 AGI** 做出贡献。
- **对 Axolotl 微调的兴奋期待**：对于模型发布后在 **Axolotl** 中的**微调（Fine-tuning）**能力，人们抱有乐观的期待。
   - 鉴于去中心化训练的创新，许多参与者渴望看到系统将如何处理微调。



**提到的链接**：<a href="https://x.com/PrimeIntellect/status/1859923050092994738">来自 Prime Intellect (@PrimeIntellect) 的推文</a>：我们做到了——首个 10B 模型的去中心化训练已完成！训练跨越美国、欧洲和亚洲 🌐 与 @arcee_ai 的训练后阶段正在进行中，完整的开源发布即将到来...

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1309608818870583377)** (1 条消息): 

> `Neural Turing Machines, Differentiable Neural Computers` 


- **探索 Neural Turing Machines**：一位成员表达了对 **Neural Turing Machines** 的兴趣，并提到过去几天一直在研究这个主题。
   - *他们非常希望能与有共同兴趣的人交流想法*。
- **深入研究 Differentiable Neural Computers**：讨论还涉及了 **Differentiable Neural Computers**，该成员渴望进一步深入探讨其概念。
   - 他们正在寻找志同道合的热心人士，就这两项技术相关的想法和见解进行协作。


  

---


---


---


---


{% else %}


> 完整的频道逐条分析已针对电子邮件进行了截断。 
> 
> 如果您想查看完整的分析，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}