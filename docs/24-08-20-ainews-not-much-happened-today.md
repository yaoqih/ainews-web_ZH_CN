---
companies:
- openai
- anthropic
- microsoft
- meta-ai-fair
- hugging-face
- langchain
- box
date: '2024-08-21T00:22:36.551416Z'
description: '以下是为您翻译的中文内容：


  **OpenAI** 推出了 **GPT-4o 微调**功能，并发布了关于 Cosine 的案例研究。**Anthropic** 发布了支持 8k token
  输出的 **Claude 3.5 Sonnet**。**微软 Phi 团队**推出了 **Phi-3.5** 的三个版本：Mini (3.8B)、MoE (16x3.8B)
  和 Vision (4.2B)，以其出色的样本效率（sample efficiency）而备受关注。**Meta** 发布了 **Llama 3.1 405B**，该模型可在
  Google Cloud Vertex AI 上部署，提供 GPT-4 级别的能力。**Qwen2-Math-72B** 在数学基准测试中取得了顶尖（SOTA）成绩，并提供了
  Gradio 演示。


  讨论内容涵盖了 **ViT 与 CNN** 的模型对比以及 **Mamba 架构**。工具更新方面包括 **DSPy** 的路线图、提升了 M1 Max 芯片上扩散速度的
  **Flux Schnell**，以及 **LangChain** 社区活动。研究亮点包括用于数学推理的 **零样本 DUP 提示（prompting）** 技术和微调最佳实践。AI
  伦理方面涉及加利福尼亚州的 AI 安全法案 **SB 1047** 以及 **Yann LeCun** 对监管的担忧。此外还有 **Swyx** 对 AI 工程师角色的评论。Box
  Enterprise Plus 用户现已可以使用 **“Chat with PDF”** 功能。'
id: d0550d67-e91a-463a-95eb-e1fa1d923fcd
models:
- gpt-4o
- claude-3.5-sonnet
- phi-3.5-mini
- phi-3.5-moe
- phi-3.5-vision
- llama-3-1-405b
- qwen2-math-72b
original_slug: ainews-not-much-happened-today-5079
people:
- swyx
- ylecun
title: 今天没发生什么。
topics:
- fine-tuning
- benchmarking
- model-comparison
- model-performance
- diffusion-models
- reinforcement-learning
- zero-shot-learning
- math
- model-efficiency
- ai-regulation
- ai-safety
- ai-engineering
- prompt-engineering
---

<!-- buttondown-editor-mode: plaintext -->**AI 领域又是一个平静的一天。**

> 2024年8月19日至8月20日的 AI 新闻。我们为您检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**254** 个频道，**2227** 条消息）。预计节省阅读时间（按 200wpm 计算）：**258 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！


没有重大新闻，只有一些小动态： 

- [OpenAI 正式发布（GA）了 GPT-4o 微调功能](https://openai.com/index/gpt-4o-fine-tuning/)，并附带了一个关于 Cosine 的显著案例研究
- [Anthropic 正式发布了 Claude 3.5 Sonnet 的 8k token 输出功能](https://x.com/alexalbert__/status/1825920737326281184)
- [Zed 推出了其竞争 Cursor/Cursor Composer 的 AI 功能](https://zed.dev/blog/zed-ai)
- [Microsoft Phi 团队发布了](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) Phi-3.5 的三个变体：Mini (3.8B)、MoE (16x3.8B)、Vision (4.2B)，所有模型都具有[极高的样本效率（sample efficient）](https://x.com/Yampeleg/status/1825981743100240201)。目前还没有论文或独立评估。

既然今天是平静的一天，您可以通过关注 Box AI 来支持 AINews，他们慷慨赞助了本周的内容！

---

**[由 Box 赞助]** 您可能有一个应用程序。它可能有用户。这些用户甚至可能在 Box 中存储文档。[但 Box AI 让您的用户可以直接在 Content Preview UI 组件中查询他们的文档！](https://shortclick.link/5lxgsv)

*Swyx 评论：“**Chat with PDF**”现在只需一个 React 组件和一个 API key 即可实现！请注意，目前仅面向 Box Enterprise Plus 客户开放。*

(此前关于 Box AI 的内容：[第一周](https://shortclick.link/tndo68)，[第二周](https://shortclick.link/23g92m))


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，从 4 次运行中择优。


**AI 模型开发与基准测试**

- **Llama 3.1 405B 发布**：Meta 发布了 Llama 3.1 405B，现在可以轻松部署在 Google Cloud Vertex AI 上。这提供了 GPT-4 级别的能力，且可以在内部运行，实现完全控制。[@_philschmid](https://twitter.com/_philschmid/status/1825541324893737085) 分享了使用 Hugging Face 的 Text Generation Inference 容器进行部署的细节。

- **Qwen2-Math-72B**：该模型在多个数学基准数据集上达到了 SOTA 性能。一个 Gradio 演示版已发布供测试。[@huybery](https://twitter.com/huybery/status/1825560321383428166) 强调了其强大实力并提供了试用链接。

- **模型比较**：多条推文讨论了不同模型和架构之间的比较：
  - [@giffmana](https://twitter.com/giffmana/status/1825617256967262699) 提到了 ViT 与 CNN 的性能对比。
  - [@wightmanr](https://twitter.com/wightmanr/status/1825630715188490390) 讨论了 Mamba 架构的性能。

**AI 工具与应用**

- **DSPy**：[@lateinteraction](https://twitter.com/lateinteraction/status/1825594011484303596) 分享了关于 DSPy 2.5 和 3.0 的更新，包括未来开发的路线图。重点是从临时的 Prompting 转向系统化的编程。

- **Flux**：[@awnihannun](https://twitter.com/awnihannun/status/1825546558739517643) 提到，在带有 MLX 的最新 DiffusionKit 中，Flux Schnell 速度提升了 30% 且占用更少 RAM，允许在 M1 Max 笔记本上不到一分钟生成高质量图像。

- **LangChain**：LangChain 社区正在组织活动，包括在奥斯汀举行的 Hacky Hour。[@LangChainAI](https://twitter.com/LangChainAI/status/1825675460380078226) 分享了即将举行的聚会细节。

**AI 研究与技术**

- **Zero-shot DUP prompting**：该技术在各种 LLM 的数学推理任务上取得了 SOTA 结果。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1825671188007297077) 解释了其三阶段过程以及在减少语义误解错误方面的优势。

- **微调模型 (Fine-tuning Models)**：[@jxnlco](https://twitter.com/jxnlco/status/1825563945798918473) 分享了关于微调模型的见解，强调了数据质量、避免供应商锁定 (vendor lock-in) 以及专注于全面评估的重要性。

**AI 伦理与监管**

- **加州 AI 安全法案 SB 1047**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1825667014498701615) 总结了该法案修订版的关键点，包括对责任和安全实践要求的变更。

- **AI 监管辩论**：[@ylecun](https://twitter.com/ylecun/status/1825500979552284712) 对监管 AI 研发表示担忧，特别是关于阻碍科学信息交流和开源代码分发的障碍。

**AI 工程视角**

- **AI Engineer 角色**：[@swyx](https://twitter.com/swyx/status/1825630984911597834) 讨论了 AI Engineer 的核心目标是将现有的基础模型能力转化为有用的产品。他强调了这与传统 ML Engineering 的分歧，以及 AI 技术栈日益增加的复杂性。

- **Docker 的重要性**：[@svpino](https://twitter.com/svpino/status/1825578554895266012) 强调了学习 Docker 对于构建和部署软件的必要性，称其为他工作中的主要差异化因素。

- **LLM API 业务**：[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1825665250231988280) 对 LLM API 业务的经济模式表示困惑，引发了关于此类模型可持续性和盈利能力的讨论。


---

# AI Reddit 摘要

## /r/LocalLlama 回顾

**主题 1. 大语言模型发布与部署**

- **发布：Magnum 123B** ([Score: 110, Comments: 21](https://reddit.com//r/LocalLLaMA/comments/1ewb7b6/announcing_magnum_123b/)): **Magnum-v2-123B** 基于 **MistralAI** 的 **Large** 模型，作为目前最大的 Magnum 模型发布，其训练数据集与其他 v2 模型相同。该模型在 **RunPod** 上使用 **8x MI300 GPUs** 进行训练，虽然尚未经过正式评估，但在测试中表现出令人期待的结果，似乎比之前的 Magnum 版本有所改进。


**主题 2. 创新 AI 界面：手写与语音识别**

- **[使用 Whisper+GPT 进行自动笔记记录和标签化](https://v.redd.it/i9nwct9gupjd1)** ([Score: 72, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1ewi9m2/using_whipsergpt_for_automatic_note_taking_and/)): 正如帖子作者所述，**Whisper** 和 **GPT** 正被用于 **Obsidian** 中的自动笔记记录和标签化。这些 **AI models** 的结合实现了音频到文本的高效转换以及随后的笔记整理，有望简化 **Obsidian** 笔记系统中的信息捕获和分类流程。
  - 作者分享了其 **GitHub** 仓库链接，包括 [AlwaysReddy](https://github.com/ILikeAI/AlwaysReddy) 和 [alwaysreddy_add_to_md_note](https://github.com/ILikeAI/alwaysreddy_add_to_md_note)，用于处理转录和笔记功能。
  - **Obsidian** 用户讨论了笔记保存选项，包括每日笔记和静态笔记。一位用户提到将 Obsidian 笔记与 **Open WebUI** 中的流水线集成。
  - 该系统使用 **LLM**（如 **Claude**）进行自动标签化，并可与任何 **LLM** 配合使用，包括本地模型服务器。

- **[电子阅读器上的手写界面。慢慢把它变成我梦寐以求的 Palm Pilot。最终我希望它能识别形状——但我不确定哪些廉价模型能做到这一点（约 0.5B 规模）](https://i.redd.it/9mk6hhlb6qjd1.gif)** ([Score: 249, Comments: 29](https://reddit.com//r/LocalLLaMA/comments/1ewjog3/handwriting_interface_on_the_ereader_slowly/)): 该帖子讨论了为电子阅读器开发**手写界面**，旨在打造一款让人联想到先进 **Palm Pilot** 的设备。作者表示有兴趣实现**形状识别**功能，但不确定参数规模在 **0.5B** 左右的小型廉价**语言模型**是否能胜任此任务。
  - 该项目在 **Boox Palma** 设备上运行，使用 **ollama** 上的 **qwen2:0.5b**，后端使用 **bun**，前端使用 **handwriting.js**。用户建议可能需要升级到 **gemma2B** 或 **phi-3-mini** 模型，并讨论了不同设备上的 Token 生成速度。
  - 关于 **LLM** 手写界面的实用性引发了争论，一些人认为这与 **LLM** 的优势相悖。另一些人则认为这一概念是开放权重（open weights）与不同输入类型结合的创新尝试，并提出了潜在用途，例如将简短的手写笔记转化为更流畅的文本。
  - 用户将该项目与虚构的魔法物品联系起来，特别是《哈利·波特》中的 **Tom Riddle's diary**。此外，还有对 **Boox** 公司的批评，呼吁出现尊重开源协议并生产更耐用设备的竞争对手。

## AI Reddit 全面回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 图像生成进展**

- **Flux 模型展示了多功能的图像生成能力**：
  - [通过单个提示词生成网格图](https://www.reddit.com/r/StableDiffusion/comments/1ew23gd/psa_flux_is_able_to_generate_grids_of_images/)
  - [产品摄影应用](https://www.reddit.com/r/StableDiffusion/comments/1ew5l8j/flux_is_a_game_changer_for_product_photography/)
  - [塔罗牌 LoRA 创作](https://www.reddit.com/r/StableDiffusion/comments/1ewkvl6/this_flux_tarot_card_lora_is_so_much_fun/)
  - [3D 立体图像生成](https://www.reddit.com/r/StableDiffusion/comments/1ew4avp/flux_has_the_capability_to_create_3d_stereo/)
  - [潜空间（Latent Space）随机漫步](https://www.reddit.com/r/StableDiffusion/comments/1ew2r1r/a_random_walk_through_flux_latent_space/)

- **Flux 模型的优势与局限性**：
  - [在几何和排版方面令人印象深刻的线条绘制能力](https://www.reddit.com/r/StableDiffusion/comments/1ewkvl6/this_flux_tarot_card_lora_is_so_much_fun/lizh6d4/)
  - [复杂场景中可能存在的问题](https://www.reddit.com/r/StableDiffusion/comments/1evyqu2/flux_is_fun_until/)

**AI 行业动态**

- **AMD 挑战 Nvidia 在 AI 基础设施领域的领先地位**：[AMD 签署 49 亿美元协议](https://www.reddit.com/r/singularity/comments/1ew1zgp/amd_signs_49bn_deal_to_challenge_nvidias_ai/)，旨在 AI 硬件市场展开竞争。

**AI 伦理与哲学讨论**

- **关于 AI 意识与智能的辩论**：
  - [关于 AI 永恒争论的梗图](https://www.reddit.com/r/singularity/comments/1evwsd5/a_meme_about_the_eternal_debate_about_ai/)
  - [关于人类认知的预测性和生成性本质的讨论](https://www.reddit.com/r/singularity/comments/1evwsd5/a_meme_about_the_eternal_debate_about_ai/liuh4n2/)
  - [对 AI 权利运动的批评](https://www.reddit.com/r/singularity/comments/1ewl51b/it_has_begun/)

**梗图与幽默**

- [关于 AI 辩论的梗图](https://www.reddit.com/r/singularity/comments/1evwsd5/a_meme_about_the_eternal_debate_about_ai/)
- [“这不是真正的思考，只是闪烁的推理”梗图](https://www.reddit.com/r/singularity/comments/1ew4vns/its_not_really_thinking_its_just_sparkling/)
- [AI 权利运动恶搞视频](https://www.reddit.com/r/singularity/comments/1ewl51b/it_has_begun/)


---

# AI Discord 回顾

> 由 Claude 3.5 Sonnet 生成的摘要之摘要的摘要

**1. LLM 进展与基准测试**

- **Hermes 3 挑战巨头**：**[Hermes 3](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b)**，一个拥有 70B 参数的模型已在 OpenRouter 上发布，具备先进的 Agent 能力和改进的角色扮演能力。
   - 用户渴望将 **Hermes 3** 的性能与 **Meta-Llama 405b** 等模型进行对比，尽管它尚未列入 [LLM Arena 排行榜](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)。
- **LLaMA 3.1 在 SQL 任务中表现不佳**：一位用户报告称，**[LLaMA 3.1 70B](https://ai.google.com/research/pubs/pub49727.html)** 无法使用 [LangChain 的 SQL Agent](https://langchain.readthedocs.io/en/latest/modules/agents/agents.html#sql-agent) 查询数据库，而 **GPT 3.5** 在相同配置下却能成功。
   - 尽管尝试了自定义解析器，问题依然存在，引发了关于 LLaMA 在某些任务中相较于其他模型局限性的推测。
  


**2. 模型性能优化**

- **Torch.compile 重新编译挑战**：用户讨论了由于生成过程中的输入形状变化以及在训练和推理模式之间切换而导致的 **torch.compile** 重新编译问题。
   - 讨论强调了 torch.compile 在处理动态场景（如传递 RNG 生成器对象）时的局限性，这些场景会导致图中断（graph breaks）。
- **自定义 Mask 与 KV-Cache 兼容性**：开发者探讨了自定义 Mask 与语言模型中 **KV-Cache** 的兼容性，指出直接使用可能不兼容。
   - 一个潜在的解决方案包括利用自定义 Mask 并移除 `self.causal_mask`，但这需要进一步的调查和测试。
- **用于本地内存的 AI 芯片设计**：讨论集中在 **AI 芯片**如何设计大量的本地内存以将模型放入缓存中，从而减少频繁向 RAM 传输数据的惩罚。
   - 辩论了片上网络（NoC）设计与缓存管理之间的权衡，指出虽然 NoC 提供了跨核心的高效数据传输，但也会引入延迟。
  


**3. 开源 AI 发展**

- **Whisperfile 简化音频转录**：由 Justine Tunney 创建的 **[Whisperfile](https://simonwillison.net/2024/Aug/19/whisperfile/)** 提供了一种使用 OpenAI 的 Whisper 模型在本地轻松转录音频的方法，支持 100% 本地运行和翻译功能。
   - 该工具甚至可以在转录过程中将非英语音频翻译成英语，使其成为音频处理任务的多功能解决方案。
- **LlamaIndex 扩展学习资源**：**LlamaIndex** 推出了 [O'Reilly Media 课程](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)，内容涵盖检索增强生成 (RAG) 的组件、评估、摄取管道、可观测性、Agents 和多模态。
   - 此外，LlamaIndex 正在举办一场名为 "LLMs in Production" 的 AI 产品见面会，重点关注使用 RAG 和 Vector DB 构建上下文增强型 LLMs，以及针对生产级 LLMs 的高性能推理。
- **Aider v0.51.0 增强开发工作流**：**[Aider v0.51.0](https://aider.chat/HISTORY.html#v0510)** 发布，改进了 Anthropic 模型的提示词缓存 (prompt caching)，优化了大型仓库的仓库映射 (repo mapping)，并增强了 Jupyter Notebook .ipynb 文件的编辑功能。
   - 此版本包含多项错误修复和改进，Aider 贡献了该版本 56% 的代码，展示了该工具在 AI 辅助开发方面的能力。
  


**4. 多模态 AI 和视觉模型**

- **LM Studio 的视觉模型限制**：用户询问了 **LM Studio** 是否具备处理照片或视频作为输入的能力，以便在编程任务中提供视觉上下文。
   - 经确认，LM Studio 中的本地模型无法处理此类任务，目前只有 **GPT4o** 和 **Claude** 等云端模型提供此功能。
- **Qdrant 1.10 提升多向量表示能力**：**[Qdrant 1.10](https://qdrant.tech/articles/late-interaction-models/)** 引入了对多向量表示的支持，提升了检索质量，并支持 **ColBERT** 等延迟交互模型。
   - 该更新允许通过移除池化步骤并使用 Token 级嵌入进行检索和重排序，从而将常规的稠密嵌入模型适配为延迟交互模型。
  


**5. LLM 训练和微调技术**

- **MiniPile：模型训练的精简替代方案**：**[MiniPile 数据集](https://huggingface.co/datasets/JeanKaddour/minipile)** 是 Pile 语料库的一个 6GB 子集，由于完整的 Pile 数据集体积庞大且成本高昂，它被推荐作为训练小规模模型的切实可行的替代方案。
   - MiniPile 通过过滤掉低质量簇进行筛选，确保了预训练数据集的多样性，对于学术预算和较小规模的实验来说更易于管理。
- **模型合并与扩展策略**：讨论中出现了一些新颖的模型合并策略，例如将 **UltraChat** 与基础 **Mistral** 之间的差异应用于 **Mistral-Yarn**，引发了关于“诅咒模型合并” (cursed model merging) 技术潜力的辩论。
   - 用户还探索了将 **Mistral** 等模型的 Token 限制扩展到初始范围之外的选项，建议在 *mergekit* 和 *frankenMoE finetuning* 方面做进一步工作作为潜在解决方案。
  

---

# PART 1: High level Discord summaries

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 对微调 Llama-3.1-405B 的限制**：一位用户询问是否可以在带有 **H100** 的 **Hugging Face Space GPU** 上微调 **Llama-3.1-405B**，但被告知 **Unsloth 目前不支持此操作**，因为该模型对显存要求极高。
   - 用户被告知需要 **至少 360 GB 的 GPU 显存** 和 **8 张 H100 GPU**，而 Unsloth 目前不提供此类支持。
- **Lambda 的免费模型访问和微调限制**：一位用户询问 **Lambda 是否提供免费微调** **Llama-3.1-405B** 的服务。
   - 他们被告知 **Lambda 仅提供免费的模型执行（推理）**，**不提供免费微调**，但类似的功能可以在 **Hugging Face、Meta 和 Groq** 等平台上找到。
- **Google Colab 上的训练损失问题与故障排除**：一位用户在 **Google Colab A100 运行时**微调模型时，难以将 **Training Loss** 保持在 **1.000** 以下。
   - 他们尝试调整 Learning Rate 和 Batch Size，但最终得出结论：由于 GPU 显存要求极高，**Colab A100 运行时可能不是一个可行的长期解决方案**。
- **Unsloth Premium 与合作伙伴关系**：一位用户询问了 **Unsloth Premium** 的定价以及潜在的 **Unsloth 合作伙伴关系**。
   - 他们被告知 **Unsloth Premium 不支持直接购买**，其更快速的版本仅限世界 500 强公司使用。建议用户联系 **Mike 或 Daniel** 以获取更多信息。
- **PPL 作为模型评估指标**：**PPL** (perplexity) 是比较量化效果的有用指标，但如果 Base 模型与量化模型之间的差异过大，该指标可能会产生误导。
   - **PPL** 在 Token 级别比较模型以识别观察到的主题时也很有价值，但其绝对值没有意义，模型之间的 Delta（差异）才是关注的重点。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Llama2 模型加载问题**：一位用户报告称，运行 Llama2 Eval 时在模型加载阶段崩溃，仅打印 "killed" 并退出。
   - 该用户在运行 Llama2 评估时还遇到了 Out-of-memory (OOM) 错误，尽管其系统应该有足够的 RAM 和 GPU 显存。
- **GPT-Fast 与 HF_eval 脚本对比**：讨论集中在不同评估脚本的使用上，特别是将 GPT-Fast 评估脚本与 HF_eval 进行对比。
   - 用户报告称，在运行 HF_eval 脚本评估 Llama2 时遇到问题，导致错误消息显示 `zero_point_domain` 参数的默认值不受支持。
- **初学者的 Triton Kernel 优化**：一位用户在尝试于 `triton.jit` Kernel 中对非 constexpr 值 `seqlen` 使用 `tl.arange` 时遇到了 `ValueError`。
   - 出现此问题是因为 `seqlen` 未声明为 `tl.constexpr` 类型，而这是 Triton 中 `tl.arange` 函数所必需的，这突显了 Triton 与常规 Python 代码之间的关键区别。
- **Comfy 的 FP16 与 FP8**：一位成员原以为 Comfy 默认支持 FP16 累加器，但实际上它需要自定义的 Torch C++ 扩展。
   - Comfy 的 FP8 实现实际上并不使用 FP8 Matmul 进行计算；它仅将其作为中间数据类型使用。Stable-fast 是另一种选择，它虽然不支持 Flux，但有一些有趣的优化思路。
- **扩散模型量化技术**：一位成员讨论了如何通过保持 Self-attention 和累加在 FP16 格式来有效地量化扩散模型。
   - Oneflow/Onediff 是扩散模型的一个封装器，使用 Oneflow 进行推理和图构建，但它与 Flux 不兼容，因为 Flux 的体积太大了。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 3 与 Meta-Llama 的比较**：一位成员询问了 **Hermes 3/405** 与其他模型（特别是 **Meta-Llama 405b**）的比较，因为他们在 [LLM Arena 排行榜](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)上找不到 **Hermes**。
   - 另一位成员确认，在一份技术报告中，**Hermes 3** 使用 15 个基准测试套件与 **Llama 3.1-instruct-405** 进行了基准测试，但他们也在寻找与 **Meta-Llama 405b** 的直接对比。
- **Hermes 3：文本到文本模型**：已确认 **Hermes 3** 是一个文本到文本（text-to-text）模型，这意味着它无法生成图像。
   - 虽然你可以在 [Discord](https://discord.com/channels/1053877538025386074/1149866614590816256) 中与 **H3-405B** 交互，但机器人无法通过命令触发图像生成，它们只能通过互相 @ 提及来进行交互。
- **Llama 3.1 Minitron 4B：剪枝后的文本到文本模型**：**Llama-3.1-Minitron-4B-Width-Base** 是一个文本到文本模型，可用于各种自然语言生成任务。
   - 它是通过对 **Llama-3.1-8B** 的 Embedding 大小、Attention Heads 和 MLP 中间维度进行剪枝（pruning）获得的，随后使用来自 Nemotron-4 15B 持续预训练数据集中的 940 亿个 Token 进行蒸馏（distillation）训练。
- **Hermes 3 Amnesia 模式：仅适用于 8B**：**Amnesia 模式** 是 **Hermes 3 8b** 的一项功能，可以通过在没有 System Prompts 的情况下输入 "Hi" 来触发。
   - 然而，此模式在 Discord 上不可用，因为机器人会记住所有聊天记录。
- **PyDantic-XML：序列化与反序列化**：**pydantic-xml** 扩展允许在 Pydantic 模型和 XML 之间进行数据的序列化和反序列化。
   - 你可以在 [https://pydantic-xml.readthedocs.io/en/latest/](https://pydantic-xml.readthedocs.io/en/latest/) 找到该扩展的文档。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **DeepMind OPRO 论文问题**：一位成员询问了关于基于 OPRO 的 Prompt Tuner 的信息来源。
   - 该成员正在寻求关于如何实现该技术的澄清，可能参考了 [OPRO 论文](https://arxiv.org/abs/2203.11824)。
- **C4AI Discord 服务器邀请**：一位成员请求 C4AI Discord 服务器的邀请。
   - 该成员被建议加入 Cohere Discord 并联系特定用户，但不确定合适的沟通渠道（私信或公开频道）。
- **Cohere API `response_format` 问题**：一位成员在使用 Cohere API 的 `response_format` 参数时遇到错误。
   - 他们正在寻求关于如何在 API 请求中正确使用 `response_format` 参数的指导。
- **Cohere Classify 端点停用**：一位成员询问了 Cohere Classify 端点的潜在替代方案。
   - 该成员正在寻求类似分类服务的建议，重点关注功能和可用性。
- **大语言数据集的 Reranker API 效率**：一位成员询问，将大型数据集分块并在每个块上独立运行 Reranker API 是否会产生准确的整体相关性分数。
   - 该成员正在探索以分块方式将 Reranker API 应用于大型数据集的潜在局限性和优势。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Hermes 3 发布**：**Hermes 3**，一个 70B 参数模型，已在 [OpenRouter](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b) 上发布，具有先进的 Agent 能力和更好的角色扮演（roleplaying）表现。
   - 发布公告中还包含了 OpenRouter, LLC 的版权声明，注明 © 2023 - 2024 OpenRouter, LLC。
- **GPT Function Calls 仍然支持吗？**：一位用户询问 OpenRouter 是否仍支持 GPT 函数，因为即使停止原因是 'functioncall'，他们收到的却是 'function_call=None'。
   - OpenRouter 团队确认更好的 tool call 路由即将推出，但目前除非使用 OpenAI、Anthropic 或 Google 模型，否则结果可能会有所不同。
- **用于德语预训练的 Mistral Large Instruct 2407**：一位用户询问是否有德语预训练效果良好的模型，得到的建议是尝试 Mistral-Large-Instruct-2407，该模型采用多语言设计并支持德语。
   - 用户测试了该模型，发现效果“还可以”但不是特别出色，并进一步建议在 Hugging Face 上查找其他模型。
- **OpenRouter 非免费模型的错误**：用户报告在尝试访问 OpenRouter 上的非免费模型时遇到错误，具体表现为“客户端异常（client-side exception）”，需要强制刷新浏览器。
   - OpenRouter 团队进行了调查，确定该问题与 access token 过期以及潜在的 CORS 错误有关，并最终解决了该问题。
- **OpenRouter 上的无审查模型？**：一位用户询问 OpenRouter 上的无审查（uncensored）模型，得到的建议是“开源”和“角色扮演”标签是可能产生 NSFW 内容的模型良好指标。
   - 无审查模型的热门选择包括 Dolphin、Stheno、Euryale 和 MythoMax。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **无审查模型：探索现状**：一位用户寻求用于非编程任务的无审查 LLM 模型建议，并获得了一个指向 [llm.extractum.io](https://llm.extractum.io/list/?uncensored) 的链接，该网站重点介绍了针对法律分析、医学研究和创意写作等多种用途的无审查 LLM。
- **LM Studio 服务器在 Llama 3.1 上遇到困难**：一位用户报告了 LM Studio 本地推理服务器的问题，特别是在使用 Llama 3.1 时，停止模式（stop pattern）被忽略了。
   - 用户指出该问题在聊天模式下不存在，并建议在相关频道进行讨论以进一步排查故障。
- **LM Studio 中的语音转文本和文本转语音**：一位用户询问了在 LM Studio 中与 Llama 2/3 模型进行语音交互的可能性，特别是是否集成了语音转文本（speech-to-text）和文本转语音（text-to-speech）功能。
   - 对方澄清 LM Studio 目前缺乏此类支持，促使用户探索外部解决方案，如用于文本转语音的 [Parler-TTS](https://github.com/huggingface/parler-tts) 和用于语音转文本的 [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) 为。
- **LM Studio 中的视觉模型：云端事务**：一位用户询问 LM Studio 中是否有能够处理照片或视频输入以提供编程任务视觉上下文的模型。
   - 经确认， LM Studio 中的本地模型无法处理此类任务；只有像 GPT-4o 和 Claude 这样的云端模型才提供此功能。
- **M2 Ultra：对 AI 性能寄予厚望**：一位用户对即将推出的 M2 Ultra 表示兴奋，指出其在 AI 任务中的性能备受期待。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT-4 神经元解释被推翻了？**: 一位成员对 GPT-4 神经元解释的有用性提出质疑，引用了一篇声称这些解释并不优于基准 (baselines) 的论文。
   - 另一位成员提供了一篇题为 "Language Models can explain neurons in language models" 的论文链接，但未能找到标题相似且声称 GPT-4 解释无效的论文，尽管内容可能相似。
- **在有限数据上训练模型 - 警惕胡言乱语！**: 由于随机初始化的影响，在单个小文件上训练模型可能会导致输出内容毫无意义。
   - 一位成员将其与文本压缩基准进行了比较，在这些基准中，模型被训练以记忆特定的文本块，并强调了多样化预训练数据的重要性。
- **用于高效训练的 MiniPile 数据集**: MiniPile 是 Pile 语料库的一个 6GB 子集，由于完整 Pile 数据集庞大的体积和高昂的成本，它被推荐作为训练较小规模模型的可行替代方案。
   - MiniPile 通过过滤掉低质量的集群进行策划，确保了预训练数据集的多样性，且对于学术预算来说更易于管理。
- **Frankenmerging - 组合来自不同模型的层**: 一位成员询问了将两个不同模型的层进行组合的可行性，这种技术被称为 “frankenmerging”。
   - 他们对这种方法的潜在风险表示困惑，质疑这是否会导致模型的内部表示变得混乱，并寻求关于潜在收益和挑战的澄清。
- **使用优化器进行模型合并**: 一位成员建议在将两个不同模型的层堆叠在一起之前，使用优化器来寻找层间通道的最佳排列 (permutation)。
   - 他们承认了潜在的挑战，并指出此类方法尚未在大规模 GPT 模型上得到验证。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Discord 访问令人困惑**: 用户无法加入 Perplexity Pro Discord 服务器，即使在退出并使用 Perplexity 设置中的链接重新加入后也是如此。
   - 问题似乎在于缺乏关于如何访问主 Discord 服务器内 Pro 栏目的清晰说明。
- **Perplexity 的搜索功能需要修复**: 用户报告了 Perplexity 搜索功能的问题，包括无法访问在线资源以及使用过时信息。
   - 一些用户认为这是一个后端问题，但 Perplexity 团队尚未承认或解决该问题。
- **Perplexity Pro 模型面临限制**: 用户正在讨论 Perplexity Pro 模型在编码和博客文章创建等任务中的局限性。
   - 一些用户发现 Perplexity Pro 在某些任务上不如其他模型有效，特别是在生成复杂代码或避免博客文章中的幻觉 (hallucinations) 方面。
- **Perplexity 对前端与后端的优先级排序**: 关于 Perplexity 是否优先考虑前端开发而非后端开发存在争议，一些用户报告了后端功能（如搜索和模型选择）的问题。
   - 一些用户认为这些问题表明缺乏对核心后端功能的关注，而这些功能对于平台的整体性能至关重要。
- **Perplexity Pro 功能升级讨论**: 发生了一场关于升级到 [Perplexity Pro](https://www.perplexity.ai/pro) 的讨论，该版本提供图像上传、更智能的 AI 和更多 Pro Search 等功能。
   - 其他用户还讨论了使用 [LMSYS Arena](https://www.youtube.com/embed/EQxYdx79vUg) 的潜在好处，以及即将推出的据称已准备好大规模生产的 [G1 Humanoid Robot](https://www.youtube.com/embed/EQxYdx79vUg)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex: 构建自然语言查询系统**：学习如何使用 LlamaIndex 和 Amazon Neptune 为图数据库构建自然语言查询系统！
   - 由 @bechbd 编写的综合指南展示了如何将自然语言问题转换为 openCypher 查询，并在 Amazon Neptune 图数据库上执行查询。
- **O'Reilly Media 的 RAG 课程**：LlamaIndex 推出了由 @ravithejads 编写的关于检索增强生成 (RAG) 的 O'Reilly Media 课程。
   - 这门 2 小时的课程涵盖了 LlamaIndex 的组件、RAG 系统的评估、摄取流水线 (ingestion pipeline)、可观测性、Agents、多模态等内容。
- **LlamaIndex: LLMs in Production 见面会**：参加由 @vesslai 和 @pinecone 在旧金山举办的 AI 产品见面会 "LLMs in Production"。
   - 向行业领袖学习如何使用 RAG 和 Vector DB 构建上下文增强的 LLMs，如何通过自定义 LLMs 实现更智能、更快、更便宜的解决方案，以及如何为生产级 LLMs 提供高性能推理。
- **Hierarchical Node Parser: 不进行分块？**：一位用户询问 LlamaIndex 的 Hierarchical Node Parser 是否可以在不进行分块 (Chunking) 的情况下创建层级，而是使用预定义的节点。
   - 该用户希望保留与节点关联的页面 ID 等元数据，但在当前的实现中无法实现。
- **使用 LlamaIndex 检索处理复杂问题**：一位用户讨论了在 LlamaIndex 中对简单和复杂问题进行检索能力的需求。
   - 他们设想了一种层级化方法，可以递归地总结节点并创建更高层级的数据表示，以获得细致且具有上下文的响应。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Jeremy Howard 畅谈 Latent Space**：最新的 Latent Space 播客邀请了 Jeremy Howard，讨论了 Encoder-Decoder 模型、Fast.html、保存/更新状态、fine-tuning vs RAG vs KV caching，以及他正在进行的一个新项目。
   - 在联合主持人 Swyx 俏皮地说了句“给我们尝一小口”之后，该播客被描述为“五道菜的大餐”。
- **Encoder-Decoder 模型崛起**：讨论强调了 Encoder-Decoder 模型相对于仅 Encoder 模型的优势，特别是在处理复杂上下文和错综复杂的关系方面。
   - 受访者（可能受 AI Paper Club 电话会议的影响）已经了解这种方法，表明 AI 社区对此的认识正在提高。
- **Whisperfile 让转录变得轻而易举**：Whisperfile 是一款新工具，允许用户利用 OpenAI 的 Whisper 模型轻松地在本地转录音频。
   - 由 Justine Tunney 创建，Whisperfile 提供 100% 本地操作，甚至可以在转录过程中将非英语音频翻译成英语。
- **Claude 3.5 Sonnet 获得 Token 提升**：Anthropic AI 已将 Claude 3.5 Sonnet 的最大输出 Token 限制翻倍，从 4096 增加到 8192。
   - 此更新现已在 Anthropic API 和 Vertex AI 中可用，使开发者更容易使用 Claude 3.5 Sonnet。
- **GPT-4 Fine-Tuning 挑战 Composer**：OpenAI 发布了 GPT-4 fine-tuning，这是一项允许用户自定义 GPT-4 行为和性能的新功能。
   - 这一更新可能会与 Cursor 的 Composer 功能展开竞争，因为两者都提供了类似的大型语言模型定制和使用方法。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 与 MAX 更新节奏同步**：此前，**Mojo** 和 **MAX** 拥有独立的更新周期，但现在它们已同步。
   - 这意味着你可以安装 **MAX+mojo main** 或 **MAX+mojo nightlies**，但不能分别安装 **MAX main** 和 **mojo nightlies**。
- **带有标签的孪生网络 (Siamese Networks)？**：一位用户询问如何将孪生网络的输出从 sigmoid 切换为标签（例如 "dog" 或 "cat"）。
   - 另一位用户建议，如果你想切换到打标签任务，使用该任务的标准模型可能比尝试适配孪生网络更有效率。
- **使用 Slice 自定义算子 (Custom Op)**：一位用户请求一个演示使用 **slice custom op** ([https://docs.modular.com/max/api/mojo/graph/ops/slicing/slice](https://docs.modular.com/max/api/mojo/graph/ops/slicing/slice)) 的代码示例。
   - 他们表示难以理解该算子的参数。
- **Mojo 的 `List` 赋值使用 `ref`**：一位用户惊讶地发现 Mojo 的 `List` 实现中没有 `__setitem__` 方法用于赋值，但被告知 `__getitem__` 返回一个 `ref[lifetime] T`，其行为类似于 `__setitem__`。
   - 这就是向 Mojo `List` 分配元素的方式。
- **Mojo 的 `ref` 和 `__lifetime_of` 函数**：函数返回类型中的 `ref` 关键字是最近（在 Mojo v244 中）作为新语言特性引入的。
   - Mojo 的 `__lifetime_of` 函数允许你确定引用的生命周期，这对于内存管理非常有用。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 在简单任务上表现挣扎**：一位用户指出，ChatGPT 在处理诸如计算单词 "strawberry" 中 "R" 的数量等简单任务时表现挣扎，这暗示 AI 可能并不像某些人想象的那么先进。
   - 这引发了关于 AI 当前局限性的讨论，以及它是真正的智能还是仅仅是一个可以执行特定任务的工具。
- **Grok2 采取了不同的方法**：一位用户提到 Grok2 在处理问题时有一种有趣的方法。
   - 另一位用户指出，Grok2 的方法涉及将每个问题分解并逐步解决，这与人类解决问题的方式类似。
- **AI 热潮——是否言过其实？**：一位用户表示，由于 AI 目前的局限性，“AI 爱好者”一词已经失去了意义。
   - 这种情绪源于对 ChatGPT 在简单任务上的挣扎以及 Grok2 解决问题方法的讨论。
- **构建智能食谱**：一位用户寻求关于创建“智能食谱”的建议，该食谱可以在他们喜欢的食谱上进行训练并提供个性化建议。
   - 该用户认为这种模型可以应用于任何“入门指南”类书籍，并请求有关现有解决方案或项目的信息。
- **Strawberry 发布推测**：一位用户询问 "Strawberry" 的发布日期，这可能是一个新的 AI 模型或功能。
   - 另一位用户开玩笑地回应称 "Strawberry" 仍处于“来源不可靠的泄露”阶段，并对它的发布表示怀疑。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torch.compile 在重新编译方面表现挣扎**：当输入形状改变（如在生成过程中）或在训练和推理模式之间切换时，会发生 Torch.compile 重新编译。
   - 这是由于 `grad_mode` 的变化引起的，可以通过实现 `torch.compile` 优化来改进。
- **Torch.compile 缓存大小限制**：`torch._dynamo hit config.cache_size_limit (8)` 消息表明已达到缓存大小限制。
   - 这暗示了 Torch.compile 友好性方面可能存在问题。可能需要增加缓存的大小。
- **RNG 对象与 Torch.compile 不兼容**：将 RNG 生成器对象传递到模型中会导致图中断 (graph breaks)，这表明 Torch.compile 目前不支持此类对象。
   - 这可能是一个挑战，但可以通过更新 `torch.compile` 以处理这些对象来解决。
- **自定义掩码 (Custom masks) vs kv-cache**：自定义掩码可能无法直接与 kv-cache 兼容，但使用你自己的掩码并移除 `self.causal_mask` 可能会有所帮助。
   - 这个问题值得进一步调查。
- **Torchtune 发布日期**：社区渴望知道 Torchtune 的发布日期，据报道它已经完成了 99%。
   - 讨论表明发布日期尚未最终确认。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LLaMA 3.1 70B 在 SQL 方面遇到困难**：[LLaMA 3.1 70B](https://ai.google.com/research/pubs/pub49727.html) 在使用 [LangChain 的 SQL agent](https://langchain.readthedocs.io/en/latest/modules/agents/agents.html#sql-agent) 查询数据库时遇到困难，而 [GPT 3.5](https://platform.openai.com/docs/models/gpt-3-5) 在相同配置下表现成功。
   - 尽管尝试了自定义解析器，问题仍然存在，这表明 LLaMA 的能力可能存在局限性。
- **Mistral 在扩展超过 8k 时面临挑战**：一位用户指出 [Mistral](https://www.mistral.ai/) 在没有进一步预训练的情况下无法扩展到 8k 以上。
   - 他们建议探索 *mergekit* 和 *frankenMoE finetuning* 来解决这一限制。
- **模型合并策略引发讨论**：一位用户提议将 **UltraChat** 和基础 **Mistral** 合并为 **Mistral-Yarn**，作为一种潜在的模型合并策略。
   - 虽然一些人表示怀疑，但该用户保持乐观，并引用了他们在所谓的“诅咒模型合并（cursed model merging）”中取得的过往成功。
- **Open Empathic 项目寻求协助**：一位用户请求支持扩展 **Open Empathic** 项目中的类别，特别是低端类别。
   - 他们分享了一个 [YouTube 视频](https://youtu.be/GZqYr8_Q7DE)，展示了项目的启动和教程，鼓励用户贡献来自 YouTube 视频中喜欢的电影场景，并提供了 [OpenEmpathic 项目](https://dct.openempathic.ai/)的链接。
- **LangChain 推出实验性的 SQLDatabaseChain**：一位用户介绍了 [LangChain 的 SQLDatabaseChain](https://langchain.readthedocs.io/en/latest/modules/chains/sql_database_chain.html)，这是一个旨在根据用户提示词生成 SQL 查询的实验性功能。
   - 他们提供了一个使用该功能的函数代码示例，概述了用于 SQL 查询生成的 Prompt 模板以及如何处理来自 Chain 的响应。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Ollama 与 OpenInterpreter 的集成**：一位用户寻求在远程机器上将 Ollama 与 OpenInterpreter 集成的指导，特别是配置 profile YAML 并使用该 profile 启动 Interpreter。
   - 他们询问了如何在 OpenInterpreter 的配置中使用正确的 IP 地址和端口连接到其 Ollama 实例，然而，OpenInterpreter 仍然拒绝连接。
- **Deepseek API：OpenAI 和本地 LLMs 的替代方案**：一位用户询问关于使用 Deepseek API 作为 OpenAI 或本地 LLMs 替代方案的指南。
   - 该用户表示有兴趣将 Deepseek 作为访问和利用大语言模型（LLMs）的潜在解决方案。
- **解决 Mac 上 Poetry 和 Pytorch 的安装问题**：一位用户报告在 Mac 上安装 Poetry 和 Pytorch 2.3.0 时遇到问题，并提到一个尚未解决的公开 issue。
   - 他们寻求解决此安装问题的指导，可能涉及替代安装方法或排查特定的配置设置。
- **OpenInterpreter 更新发布**：最新的 OpenInterpreter 更新已在 #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1271135268807905384) 频道宣布。
   - 未提供关于更新性质或范围的更多细节。
- **无障碍圆桌会议提醒**：#[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1275167433082146848) 频道发布了无障碍圆桌会议（Accessibility Roundtable）的提醒。
   - 提醒中包含了一个活动链接，表明这是一个虚拟或在线会议。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **dspy-ai 安装困扰**：一位用户注意到 `requirements.txt` 文件列出了 `dspy==2.0.5`，但质疑是否应该实际上是 `dspy-ai`。
   - 他们还指出 `pickle5==0.0.12` 存在潜在的兼容性问题，该版本兼容 Python 3.8 以下版本，而 `dspy-ai` 需要 Python 3.9 或更高版本。
- **ADAS 能否发明新的构建模块？**：一位用户询问 ADAS 是否可以发明新的构建模块，例如集成系统的 Function Calling。
   - 他们还询问是否已经有人尝试过类似的实验。
- **用于 DSPy 微调的 Multi-Lora 设置**：一位用户建议在 DSPy 微调中使用 Multi-Lora 设置，认为这可能是一种有价值的方法。
   - 关于如何实现这一点，目前没有提供更多细节。
- **DSPy vs. Langchain/LLamaindex：选择你的武器**：一位用户询问 DSPy 与 Langchain 和 LLamaindex 的对比。
   - 他们被引导至 DSPy 文档，以获取选择合适工具的指导。
- **Aider v0.51.0：Prompt Caching 和 Repo Mapping 改进**：Aider 发布了 0.51.0 版本，其特点是改进了 Anthropic 模型的 Prompt Caching，优化了大型仓库的 Repo Mapping，并增强了 Jupyter Notebook .ipynb 文件的编辑功能。
   - 该版本包含各种错误修复和改进，正如 [Release history](https://aider.chat/HISTORY.html#v0510>>>) 中所述，Aider 为该版本贡献了 56% 的代码。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LTXStudio 发布五项新功能**：LTXStudio 为用户发布了五项新功能，旨在将他们的项目提升到新的水平。
   - 这些功能现在可以访问和测试，LTXStudio 发布了一条推文宣布发布并鼓励用户尝试：[来自 LTX Studio (@LTXStudio) 的推文](https://x.com/LTXStudio/status/1825909655207383308?t=5Wk2X8i_lQ5R5HAJxcerlg&s=19)。
- **JPEG 编码：一种不确定的图像 Tokenization 方法**：一篇研究论文提出将 JPEG 编码作为一种可行的图像 Tokenization 方法，但目前基于 AR 的方法在信息丢失严重的情况下表现不佳，导致图像质量较低。
   - 该论文使用了 25 的 JPEG 质量设置，这在理论上阻碍了从 Token 生成高质量图像，并将 256*256 的图像压缩为 5,000 个 Token，使得训练和推理速度比传统的 VQ-VAE 更慢。
- **关于图像压缩极限的问题**：鉴于论文在 Tokenization 中使用了 25 的 JPEG 质量设置，作者对图像可能的最大压缩率提出了疑问。
   - 这引发了对该方法在实现最佳图像压缩方面潜在局限性的担忧。
- **在 H.265 或 AV1 帧上训练模型**：作者建议探索在 H.265 帧甚至 AV1 帧上训练模型的可能性，作为图像 Tokenization 中 JPEG 编码的潜在替代方案。
   - 这种方法可能解决当前 JPEG 编码方法的局限性，并带来更好的性能。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Leo 模型公开**：一位成员在 Hugging Face 上公开了其 [Leo 模型的量化版本](https://huggingface.co/GPT4All-Community)。
   - 他们很乐意接受反馈，并在需要时向用户转达信息，如果需要，还可以将信息添加到 Model Card 中。
- **通过 Model Card 进行反馈和更新**：该成员提议在 Model Card 中添加信息，以便进行反馈或向用户转达信息。
   - 这样，任何人都可以看到最新的信息、反馈或更新。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Xeophon 的推文**：Xeophon 发布了 Bilawal Sidhu 关于 Deep Learning 中 Interconnects 力量的推文链接。
   - 该推文强调了 Interconnects 对于模型大规模分布式训练（Distributed Training）至关重要，且该领域正在不断演进。
- **占位符**：这是一个占位摘要，以满足至少 2 条摘要的最低要求。
   - 如果你有其他话题要讨论，可以用真实的摘要替换此内容。



---


**Alignment Lab AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。


---

# 第 2 部分：频道详细摘要与链接


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1275214173705338955)** (91 messages🔥🔥): 

> - `Fine-tuning Llama-3.1-405B`
> - `Unsloth limitations`
> - `Hugging Face Space GPU`
> - `Free model access`
> - `Training Loss` 


- **使用 Hugging Face Space GPU 微调 Llama-3.1-405B？**：一名成员询问是否可以使用 **H100 Hugging Face Space GPU** 微调 **Llama-3.1-405B**。
   - 他们被告知至少需要 **360 GB 的 GPU 显存**，而 **Unsloth 目前不支持此操作。**
- **Unsloth 不支持微调 Llama-3.1-405B**：Unsloth 目前不支持微调 **Llama-3.1-405B**，因为这需要至少 **360 GB 的 GPU 显存。**
   - 据指出，这项任务需要 **8 块 H100 GPU**，但 Unsloth 并不提供此功能。
- **免费模型访问与训练**：一名成员询问 **Lambda** 是否为 **Llama-3.1-405B** 提供免费微调。
   - 他们获知 **Lambda 不提供免费微调**，仅提供免费的模型推理执行，这在 **Hugging Face、Meta 和 Groq** 等平台上也可以实现。
- **训练损失（Training Loss）问题与排查**：一名成员在 **Google Colab A100** 环境下微调模型时，难以将 **Training Loss** 维持在 **1.000** 以下。
   - 他们尝试了将 Learning Rate 减半和调整 Batch Size 等方法，但结论是由于高 GPU 显存需求，使用 **Colab A100** 可能不是一个可行的长期解决方案。
- **Unsloth Premium 与合作伙伴关系**：一名用户询问了 **Unsloth Premium** 的价格以及与 **Unsloth** 建立合作伙伴关系的可能性。
   - 官方表示 **Unsloth Premium** 不支持直接购买，其更快的版本仅限于财富 500 强公司。建议他们联系 **Mike 或 Daniel** 以获取更多信息。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1275202744889770045)** (103 messages🔥🔥): 

> - `Perplexity (PPL)`
> - `Model Fine-tuning`
> - `Javascript AST Walking` 


- **PPL 是廉价的测试，但并非完美的指标**：**PPL** 是衡量 Quantization（量化）效果的一个好指标，但如果基座模型与量化模型之间的差异很大，它可能会产生误导。
   - 它在 Token 级别的模型比较中也很有用，可以查看某个主题是否被观察到，但绝对值没有意义——重要的是 Delta（增量）。
- **公司虽然淡化微调，但它是一项强大的技术**：尽管像 Anthropic 和 Google 这样的公司淡化了 **Fine-tuning** 的重要性，但与从头开始训练相比，它能显著提高模型性能且更具成本效益。
   - OpenAI 和 Google 现在正在推进 Fine-tuning，可能是为了扩大市场并获取新客户。
- **Javascript AST walking 非常令人头疼**：一名成员表达了对 **Javascript AST walking** 的强烈厌恶，认为它既困难又耗时，特别是在处理混淆代码（Obfuscated Code）时。
   - 他们将这种经历描述为“一种痛苦”，并感叹需要处理复杂的调用栈、混淆以及大量的位移操作（Bit Shifting）。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1275213997389512796)** (12 messages🔥): 

> - `Llama 3.1 Fine-Tuning`
> - `WSL Anaconda Installation`
> - `Mini Conda` 


- **在 HuggingFace Space GPU 上进行 Llama 3.1 Fine-Tuning**：一位成员询问是否可以使用 **H100** 在 **HuggingFace Space GPU** 上对 **Llama-3.1-405B** 进行 Fine-Tuning。
- **LoRA_MLP Backward Pass 的困惑**：一位成员对 **LoRA_MLP** 在 Backward Pass 中的手动导数感到困惑，并正在寻求关于方程正确性的澄清。
- **WSL 上安装 Anaconda 的问题**：一位成员报告称，他们成功安装了 **WSL**，但在按照教程视频安装 **Anaconda** 时遇到了问题。
- **安装 Mini Conda 的建议**：另一位成员建议安装 **Mini Conda** 作为 Anaconda 安装问题的解决方案。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

etherl: <@488399737884639242> 请勿自我推广 <:slothhug:1257540335438008343>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1275399968814927924)** (3 messages): 

> - `Lottery Ticket Adaptation`
> - `LoRAs`
> - `Finetuning`
> - `Catastrophic Forgetting` 


- **Lottery Ticket Adaptation：一种新的 Fine-tuning 方法**：一种名为 [Lottery Ticket Adaptation](https://github.com/kiddyboots216/lottery-ticket-adaptation) 的新方法是 LoRAs 和 Finetuning 的替代方案，旨在避免 Catastrophic Forgetting。
   - 该方法识别新任务中重要的权重并仅训练这些权重，从而可能保留原始模型的知识。
- **Lottery Ticket Adaptation：一种新的 Fine-tuning 方法**：一种名为 [Lottery Ticket Adaptation](https://github.com/kiddyboots216/lottery-ticket-adaptation) 的新方法是 LoRAs 和 Finetuning 的替代方案，旨在避免 Catastrophic Forgetting。
   - 该方法识别新任务中重要的权重并仅训练这些权重，从而可能保留原始模型的知识。


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1275180850392072367)** (4 messages): 

> - `Llama Model on Mac` 


- **在 Mac 上设置 Llama**：一位成员询问是否可以在配备 M3 Air 和 24GB RAM 的 Mac 上设置新的 Llama 模型。
- **Hugging Face Discord**：提供链接的成员建议在 Hugging Face 的 Discord 中提出该问题。


  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1275497831398117467)** (8 messages🔥): 

> - `Triton Error`
> - `Triton kernel optimization`
> - `constexpr type` 


- **Triton kernel 优化**：一位用户在尝试于 `triton.jit` kernel 中对非 constexpr 值 `seqlen` 使用 `tl.arange` 时遇到了 `ValueError`。
   - 该用户是 Triton 初学者，并就该错误寻求帮助，问题被指向导致错误的特定行：`base_idx_hidden_states = base_idx + tl.arange(0, seqlen)[:, None] * head_dim`。 
- **Triton 中的 constexpr 类型**：出现该问题是因为 `seqlen` 未被声明为 `tl.constexpr` 类型，而这是 Triton 中 `tl.arange` 函数所要求的。
   - 建议用户在函数定义本身中显式指定变量 `seqlen` 的类型，以解决该错误。 
- **Triton 编译器行为**：用户注意到，在通常处理自动类型推断的 Python 中一般不需要这种类型指定。
   - 然而，这个例子突显了 Triton 与常规 Python 代码之间的关键区别，强调了在 Triton kernel 中进行显式类型声明的重要性。


  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1275242807958700155)** (73 messages🔥🔥): 

> - `Comfy FP16`
> - `FP8 Matmul`
> - `Stable-fast`
> - `Oneflow/Onediff`
> - `Flux` 


- **Comfy 的 FP16 实现**：一位成员原以为 Comfy 默认支持 FP16 累加器，但实际上它需要自定义的 Torch C++ 扩展。
- **Comfy 中的 FP8 Matmul**：Comfy 的 FP8 实现实际上并未在计算中使用 FP8 matmul；它仅将其作为中间数据类型使用。
- **Stable-fast：Comfy 的替代方案**：一位成员推荐使用 Stable-fast，虽然它不支持 Flux（因为已停止维护），但在优化方面有一些有趣的思路。
- **Oneflow/Onediff 与 Flux**：Oneflow/Onediff 是扩散模型的封装器，使用 Oneflow 进行推理和图构建，但它与 Flux 不兼容，因为 Flux 的体积太大了。
- **扩散模型量化**：一位成员讨论了如何通过将 self-attention 和累加保持在 FP16 中来有效地量化扩散模型。



**提及的链接**：<a href="https://github.com/mobiusml/hqq/blob/master/hqq/kernels/hqq_aten_cuda_kernel.cu#L14-L22">hqq/hqq/kernels/hqq_aten_cuda_kernel.cu at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1275258599052017765)** (2 messages): 

> - `CUDA matrix transpose`
> - `CUTLASS tutorial`
> - `4090D with 48GB`
> - `FP8 support`
> - `bf16 testing` 


- **CUTLASS 矩阵转置教程**：来自 Colfax International 的新教程，重点介绍了使用 [CUTLASS](https://github.com/NVIDIA/cutlass/) 和 CuTe 在 NVIDIA GPU 上进行内存复制的技术，并以矩阵转置为例。
   - 该教程灵感源自 [Mark Harris 的高效矩阵转置教程](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)，但侧重于 CuTe 抽象。
- **拥有 48GB 显存和 FP8 支持的 4090D**：Twitter 用户 @bdsqlsz 晒出了拥有一块 48GB 显存的 4090D 并能使用 Torch 2.4，并展示了对 FP8 的支持。
   - 他们还分享了与某云平台的对话，对方认出他是中国 AI 领域的知名人物。
- **bf16 测试结果**：该用户还提到了参与 bf16 测试的情况，进一步彰显了其在中国 AI 社区的活跃度。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/bdsqlsz/status/1821838464108917123">青龍聖者 (@bdsqlsz) 的推文</a>：48g 4090d with torch 2.4 fp8。引用青龍聖者 (@bdsqlsz) 的话：我刚刚联系了云平台进行测试，负责人对我说：噢，bdsqlsz，是你啊~ 给你免费算力，...</li><li><a href="https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/">教程：CUTLASS 中的矩阵转置</a>：本教程的目标是阐明在使用 CUTLASS 及其核心后端库 CuTe 在 NVIDIA® GPU 上编程时涉及内存复制的概念和技术。具体来说，我们将研究……
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1275339113628962816)** (1 messages): 

> - `Krish's Skillset`
> - `Krish's Job Search`
> - `Krish's Experience` 


- **Krish：具备机器学习专业知识的计算机科学毕业生**：Krish 是加州大学圣地亚哥分校（UC San Diego）的计算机科学硕士毕业生，拥有 2 年专业的软件工程和机器学习经验，专注于深度学习模型，包括语言模型和生成模型。
   - Krish 拥有深厚的 C++ 软件开发背景，曾将语言模型集成到企业软件中，并创建了点云可视化工具。
- **Krish 的求职意向：全职与实习机会**：Krish 正在尽快寻求全职和实习机会。
   - 作为一名面临期限压力的国际学生，他欢迎任何线索，并感谢在这个困难时期的任何帮助。
- **Krish 极强的工作职业道德**：Krish 强调了他的勤奋和对学习的热忱，并坚信自己能成为任何团队的宝贵财富。
   - Krish 的简历可以在 [https://shorturl.at/YESMq](https://shorturl.at/YESMq) 找到。



**提及的链接**：<a href="https://shorturl.at/YESMq">Krish_Rewanth_Resume.pdf</a>：未找到描述

  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1275395727715078239)** (9 messages🔥): 

> - `CUDA Setup for VS Code`
> - `PyTorch CUDA errors`
> - `C++ CUDA code issues`
> - `VS Code C_CPP_Properties.json`
> - `PyTorch Cpp Extension` 


- **为 VS Code 设置 CUDA**：一位用户询问了如何为 VS Code 设置 CUDA，一位热心的成员引导他们参考官方的 [NVIDIA Nsight™ Visual Studio Code Edition](https://developer.nvidia.com/nsight-visual-studio-code-edition) 文档以获取详细说明。
   - 该文档强调，被调试的 GPU 必须位于 Linux 或 QNX 目标系统上，且本地调试只能在 Linux 系统上进行。
- **解决 PyTorch CUDA 错误**：另一位用户在使用 PyTorch `cpp_extension` 运行 `.cu` 文件时遇到错误。
   - 一位用户建议在 VS Code 工作区的 `c_cpp_properties.json` 文件中添加特定的包含路径（include paths），包括与 PyTorch、THC、CUDA 和 Python 相关的路径。
- **C++ CUDA 代码错误与修复**：在添加了推荐的包含路径后，用户仍然面临错误。他们发现将 `torch::cuda` 替换为 `c10:cuda` 解决了该问题。
   - 用户还注意到，首次运行时间接近一分钟，但后续运行速度快得多，仅需几秒钟。



**提到的链接**：<a href="https://developer.nvidia.com/nsight-visual-studio-code-edition">Nsight Visual Studio Code Edition</a>：集成到 Microsoft Visual Studio Code 中的 NVIDIA 平台 CUDA 开发工具。

  

---


### **CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1275266091173548084)** (1 messages): 

> - `Composable Kernel`
> - `ROCm`
> - `Tile Programs`
> - `GPU Computing` 


- **Composable Kernel：性能可移植的 GPU 计算**：**ROCm** 内部 **Composable Kernel** 项目的一个 [GitHub 仓库](https://github.com/ROCm/composable_kernel/tree/ck_tile_toy/example/91_tile_program)展示了一个 **tile program** 示例，旨在实现 GPU 计算中的性能可移植性。
- **Tile Program 示例：91_tile_program**：名为 **91_tile_program** 的特定示例演示了在 Composable Kernel 框架内 tile programming 的实际实现，突显了其在各种 GPU 架构上优化性能的潜力。



**提到的链接**：<a href="https://github.com/ROCm/composable_kernel/tree/ck_tile_toy/example/91_tile_program">composable_kernel/example/91_tile_program at ck_tile_toy · ROCm/composable_kernel</a>：Composable Kernel：用于机器学习张量算子的性能可移植编程模型 - ROCm/composable_kernel

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1275295714561232989)** (81 messages🔥🔥): 

> - `Llama2 eval crashing`
> - `GPT-Fast eval script`
> - `HF_eval script`
> - `OOM for Llama2`
> - `Int4wo vs bf16 performance` 


- **Llama2 评估在模型加载期间崩溃**：一位用户报告称，运行 Llama2 评估在模型加载阶段崩溃，仅打印 'killed' 并退出。
- **GPT-Fast 与 HF_eval 脚本对比**：讨论集中在不同评估脚本的使用上，特别是对比了 GPT-Fast 评估脚本与 HF_eval。
- **HF_eval 脚本限制与性能问题**：用户报告在运行 HF_eval 脚本评估 Llama2 时遇到问题，导致错误消息提示 `zero_point_domain` 参数的默认值不受支持。
- **Llama2 的 OOM 问题**：用户在运行 Llama2 评估时遇到了显存溢出（OOM）错误，尽管他们的系统应该有足够的 RAM 和 GPU 显存。
- **Torch 与 Transformers 之间的模型加载差异**：讨论强调了 Transformers 库和 Torch 在加载模型方式上的差异，HF_eval 可能在正确使用指定精度进行模型加载方面存在问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/blob/main/torchao/_models/llama/generate.py#L146-L155">ao/torchao/_models/llama/generate.py at main · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化与稀疏化工具 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/scripts/hf_eval.py">ao/scripts/hf_eval.py at main · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化与稀疏化工具 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/_models/llama/generate.py">ao/torchao/_models/llama/generate.py at main · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化与稀疏化工具 - pytorch/ao
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1275262822044925994)** (3 条消息): 

> - `H100 L2 cache optimization`
> - `memcpy optimization`
> - `cuMemAddressReserve`
> - `deterministic memory allocation` 


- **H100 L2 cache 优化：逆向哈希函数**：作者试图对 H100 GPU 上的 L2 cache 哈希函数进行逆向工程，并发现唯一的动态数据似乎是每 2 MiB (MMU page) 一个比特位。
   - 该比特位以 4KiB 的粒度变化，在每 64KiB 中始终保持 50/50 的平衡，并且可以通过 persistent threads 高效处理。
- **在 memcpy 上与 NVIDIA 竞争**：虽然作者尚未尝试在矩阵乘法上超越 NVIDIA，但他们正考虑先优化内存复制（memory copy）操作。
   - 目标是实现比 NVIDIA 实现更高的能效，并可能简化 llm.c 的复杂性。
- **使用 cuMemAddressReserve 消除动态内存分配**：作者意识到 cuMemAddressReserve 结合 cuMemMap/cuMemUnmap 可以完全消除内存分配的动态特性。
   - 这允许基于简单哈希对物理页进行确定性分配，从而可能在 llm.c 和 PyTorch 中实现优化的 elementwise kernels。


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/)** (1 条消息): 

evil666man: 很乐意在这里合作！
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1275249617029562420)** (6 条消息): 

> - `Hermes 3/405`
> - `Llama 3.1-instruct-405`
> - `Meta-Llama 405b`
> - `LLM Arena`
> - `Hermes 3 Launch` 


- **Hermes 3/405 对比 Meta-Llama 405b**：一位成员询问是否有人将 **Hermes 3/405** 与其他模型进行过对比，特别是它是否与 **Meta-Llama 405b** 旗鼓相当。
   - 他们在 [LLM Arena leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) 中没有找到 **Hermes**。
- **Hermes 3 基准测试**：另一位成员表示，技术报告中使用了约 15 个基准测试套件将 **Hermes 3** 与 **Llama 3.1-instruct-405** 进行了对比。
   - 该成员正在寻找与其他 LLM（包括 **Meta-Llama 405b**）的性能对比。
- **Hermes 3 发布视频**：一位成员分享了一个讨论 **Hermes 3** 发布的 [YouTube 视频](https://www.youtube.com/watch?v=uAo513GIwoU)。
   - 他们还提到将在即将发布的视频中讨论 **Hermes 3**。



**提到的链接**：<a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - lmsys 提供的 Hugging Face Space</a>：未找到描述

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1275340024115757178)** (2 条消息): 

> - `pydantic-xml extension`
> - `Nous Aesthetics` 


- **PyDantic-XML 扩展**：**pydantic-xml** 扩展提供了在 Pydantic 模型和 XML 之间进行数据序列化和反序列化的功能。
- **Nous Aesthetics 视频**：分享了一个关于 **Nous Aesthetics** 的视频，附带 YouTube 链接：[https://youtu.be/qGQ5U3dkZzk?si=MPLh7XEd1NrskX5g](https://youtu.be/qGQ5U3dkZzk?si=MPLh7XEd1NrskX5g)。
- **PyDantic-XML 文档**：**pydantic-xml** 扩展文档可以在 [https://pydantic-xml.readthedocs.io/en/latest/](https://pydantic-xml.readthedocs.io/en/latest/) 找到。



**提到的链接**：<a href="https://pydantic-xml.readthedocs.io/en/latest/">pydantic-xml</a>：未找到描述

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1275188271013695661)** (93 messages🔥🔥): 

> - `Llama 3.1 Minitron 4B Width Base`
> - `Hermes 3 Image Generation`
> - `Hermes 3 Amnesia Mode`
> - `Hermes 3 405B Function Calling`
> - `Hermes 3 Performance Differences` 


- **Llama 3.1 Minitron 4B Width Base**: Llama-3.1-Minitron-4B-Width-Base 是一个文本到文本模型，可用于各种自然语言生成任务。它是通过对 Llama-3.1-8B 进行剪枝（pruning）获得的。
   - 具体而言，它剪枝了模型 embedding 大小、attention heads 数量以及 MLP 中间维度。剪枝后，我们使用 940 亿个 token 进行蒸馏（distillation）持续训练以得到最终模型，并使用了 Nemotron-4 15B 中使用的持续预训练数据集。
- **Hermes 3 无法生成图像**: Hermes 3 是一个文本到文本模型。你可以在 https://discord.com/channels/1053877538025386074/1149866614590816256 与 H3-405B 进行交互。
   - 机器人无法通过 / 命令触发图像生成，它们只能通过互相 @ 提及来进行交互。
- **Hermes 3 健忘模式 (Amnesia Mode)**: 健忘模式是 Hermes 3 8b 的一个功能，可以通过在没有 system prompts 的情况下输入 "Hi" 来触发。
   - 健忘模式在 Discord 上不可用，因为它会记录所有聊天内容。
- **Hermes 3 405B 函数调用 (Function Calling)**: 有人询问是否有其他 API 提供商能提供比 Lambda API 更快的 405b 模型速度，但目前尚无所知。
   - 用户试图说服 Together 和 Octo 等其他 LLM 服务器的 Discord 频道付费以换取更快的速度，但未获回应。
- **Hermes 3 性能差异**: 观察到 Hermes 3 8b 在 LM Studio 和 KoboldCPP 中的运行表现不同，尽管采样器（sampler）设置相似。
   - 虽然两者都只使用了 minp (0.1) 和 temp，但 KoboldCPP 的 temp 是 1.25，而 LM Studio 是 0.7。



**提及的链接**: <a href="https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Width-Base">nvidia/Llama-3.1-Minitron-4B-Width-Base · Hugging Face</a>: 未找到描述

  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1275505791809355787)** (7 messages): 

> - `Roleplay benchmark`
> - `Knowledge cutoff updates`
> - `Model pretraining` 


- **自动化角色扮演基准测试：一项具有挑战性的任务**: 一名成员思考了创建自动化角色扮演基准测试的可能性，建议将形容词计数作为潜在指标。
   - 该成员还询问了算法化评分创意写作的方法，强调了客观评估创意的难度。
- **知识截止日期 (Knowledge cutoff) 更新之谜**: 一名成员询问了关于知识截止日期更新的问题，寻求资源以了解新知识如何在不显著改变核心功能的情况下融入模型。
   - 另一名成员提到，知识更新以前更频繁，但最近变得较少，暗示最近的更新与特定的模型变更有关。
- **持续预训练 (Continued pretraining)：知识更新的关键**: 一名成员对在不进行持续预训练的情况下更新知识表示怀疑，认为持续预训练可能是将新知识融入模型的必要组件。
   - 该成员还指出，最近的知识更新通常伴随着新的模型变更，暗示模型更新与知识更新之间存在联系。


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1275174237983998126)** (78 messages🔥🔥): 

> - `OPRO Paper`
> - `C4AI Discord Invite`
> - `Cohere API Response Format`
> - `Cohere Classify Sunset`
> - `Reranker API on 10k Docs` 


- **DeepMind 的 OPRO 论文**: 一名成员询问了关于基于 OPRO 的 prompt tuner 的信息来源。
- **C4AI Discord 邀请**: 一名成员请求加入 C4AI Discord 服务器的邀请。
- **Cohere API 响应格式 - `response_format` 问题**: 一名成员在使用 Cohere API 的 `response_format` 参数时遇到错误。
- **Cohere Classify 停用 (Sunset)**: 一名成员询问了 Cohere Classify 端点的潜在替代方案。
- **Reranker API 在大型数据集上的效率**: 一名成员询问，如果将大型数据集分块并在每个块上独立运行 Reranker API，是否能产生准确的整体相关性评分。



**提及的链接**: <a href="https://jobs.lever.co/cohere/bb3df91e-bef0-43b0-9e69-d8efa5ec1c8b">Cohere - Research Scholar</a>: 为什么选择这个职位？Cohere For AI 是 Cohere 专门的研究部门。Cohere For AI 研究实验室旨在通过支持探索基础研究来解决复杂的机器学习问题...

  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1275204831069470823)** (16 messages🔥): 

> - `Cohere Sponsorship` (Cohere 赞助)
> - `PDF Abstract Extraction` (PDF 摘要提取)
> - `Cohere API SSL Verification` (Cohere API SSL 验证)
> - `Cohere API LangChain`
> - `Freelance Developer Team` (自由职业开发团队)


- **Hack49 Global 寻求 Cohere 赞助**：Rishi Shah，国际黑客松 Hack49 Global 的联合创始人，正在寻求关于如何推进 Cohere 赞助请求的指导。
   - 他们被建议加入 Cohere Discord 并联系特定用户，但不确定合适的沟通渠道（私信或公开频道）。
- **不使用 GPT-4 提取 PDF 摘要**：一位社区成员在使用 py_pdf_parser 和 embedding 模型时，难以在不依赖 GPT-4 的情况下从 PDF 中提取摘要。
   - 他们正在寻求社区帮助以寻找解决方案，并探索有尺寸限制（2GB 以下）的替代 LLM 模型。
- **解决 Cohere API SSL 验证错误**：一位用户在使用 LangChain 调用 Cohere API 时遇到 SSL 验证错误，希望通过将 API URL 加入白名单来解决此问题。
   - 社区建议在 LangChain 客户端中设置 `verify=False`，并提供了 API URL (`https://api.cohere.com`) 以方便加入白名单。
- **自由职业开发团队提供服务**：一位拥有超过 8 年经验的开发者组建了一个专家团队，正在为项目寻找客户。
   - 他们在项目范围和预算方面提供灵活性，以适应高要求和注重成本的客户，但强调不接受过分的要求。
- **在大型文档中查找敏感字段**：一位用户正在开发一个在大型文档中查找敏感字段的工具，并寻求高效处理的建议。
   - 他们考虑过使用 Cohere 的 `documents` 字段，但正在探索替代方案（如向量数据库），以便在处理大文件时获得更快的速度。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1275536642970751120)** (1 messages): 

> - `Hermes 3` 


- **Hermes 3 发布**：**Hermes 3**，一个 70B 参数模型，已经发布。
   - 你可以在 [OpenRouter](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b) 上体验它。
- **Hermes 3 发布公告**：Hermes 3 的公告还包含了 OpenRouter, LLC 的版权声明。
   - 版权声明为 © 2023 - 2024 OpenRouter, LLC



**提到的链接**：<a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b">Hermes 3 70B Instruct - API, Providers, Stats</a>：Hermes 3 是一款通用语言模型，相比 [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo) 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理...

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1275213908554022922)** (84 条消息🔥🔥): 

> - `GPT Functions`
> - `OpenRouter Model Support`
> - `German Pretraining`
> - `Mistral`
> - `Multilingual Models` 


- **OpenRouter 上的 GPT Functions**：一位用户询问 OpenRouter 是否仍支持 GPT functions，因为尽管停止原因是 'functioncall'，但他们收到的却是 'function_call=None'。
   - OpenRouter 团队确认更好的 tool call 路由即将推出，但目前除非使用 OpenAI、Anthropic 或 Google 模型，否则结果可能会有所不同。
- **适用于德语的 Mistral Large Instruct 2407**：一位用户咨询是否有德语预训练效果良好的模型，建议尝试 Mistral-Large-Instruct-2407，该模型原生支持多语言（包括德语）。
   - 用户测试了该模型，发现效果“还可以”但不是特别出色，并进一步建议在 Hugging Face 上查找其他模型。
- **在 WordPress 上嵌入 OpenAI Assistant**：一位用户寻求在 WordPress 网站上嵌入 OpenAI assistant 的指导，包括文档和说明。
   - 用户提到 WordPress 支持直接的 API，但不支 Assistant API，并征求关于嵌入的首选服务或开源选项的建议。
- **OpenRouter 非免费模型错误**：用户报告在尝试访问 OpenRouter 上的非免费模型时遇到错误，具体表现为收到“客户端异常 (client-side exception)”并需要强制刷新浏览器。
   - OpenRouter 团队进行了调查，确定该问题与 access token 过期以及潜在的 CORS 错误有关，并最终解决了该问题。
- **OpenRouter 无审查模型**：一位用户咨询了 OpenRouter 上的无审查模型。
   - 建议将 “open source” 和 “roleplay” 标签作为可能产生 NSFW 内容的模型的良好指标，热门选项包括 Dolphin、Stheno、Euryale 和 MythoMax。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://js.stripe.com')">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai')."">未找到标题</a>：未找到描述</li><li><a href="https://ai.azure.com/explore/models/Phi-3.5-MoE-instruct/version/1/registry/azureml">Azure AI Studio</a>：未找到描述</li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>：未找到描述</li><li><a href="https://aka.ms/Phi-3.5-mini-instruct-pricing,">信息</a>：令人惊叹的美，就像今天照片中描绘的那样</li><li><a href="https://huggingface.co/microsoft/Phi-3.5-vision-instruct">microsoft/Phi-3.5-vision-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b">Hermes 3 70B Instruct - API, Providers, Stats</a>：Hermes 3 是一个通用语言模型，相比 [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo) 有许多改进，包括先进的 agentic 能力、更好的角色扮演、推理...</li><li><a href="https://huggingface.co/microsoft/Phi-3.5-MoE-instruct">microsoft/Phi-3.5-MoE-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://openrouter.ai/models?modality=text%2Bimage-%3Etext">Models | OpenRouter</a>：在 OpenRouter 上浏览模型
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1275234886642438154)** (45 条消息🔥): 

> - `Uncensored models`
> - `LM Studio server issues`
> - `Llama 3.1`
> - `Speech to Text and Text to Speech`
> - `Vision models` 


- **探索无审查 LLM 模型**：一位用户请求推荐最适合非编程任务的无审查 LLM 模型。
   - 提供了 [llm.extractum.io](https://llm.extractum.io/list/?uncensored) 的链接，强调了该平台对无审查 LLM 的关注，以及它们在法律分析、医学研究和创意写作等多样化应用中的潜力。
- **LM Studio 服务器在 Llama 3.1 上的问题**：一位成员报告在使用 LM Studio 的本地推理服务器时遇到问题，特别是在使用 Llama 3.1 时，停止符 (stop pattern) 被忽略。
   - 他们注意到该问题在聊天模式下不存在，并建议在相关频道发起讨论，以进一步排查并寻找潜在解决方案。
- **LM Studio 中的语音转文本和文本转语音**：一位用户询问是否可以与 Llama 2/3 模型进行语音交互，特别是 LM Studio 是否集成了 Speech-to-Text 和 Text-to-Speech 功能。
   - 澄清了 LM Studio 目前缺乏此支持，建议用户探索外部解决方案，如用于文本转语音的 [Parler-TTS](https://github.com/huggingface/parler-tts) 和用于语音转文本的 [Whisper.cpp](https://github.com/ggerganov/whisper.cpp)。
- **LM Studio 中的视觉模型**：一位用户询问能够处理照片或视频输入以提供编程任务视觉上下文的模型。
   - 确认了 LM Studio 中的本地模型无法处理此类任务；目前只有像 GPT4o 和 Claude 这样的云端模型提供此功能。
- **自动化 LM Studio 服务器启动和模型加载**：一位用户寻求自动化启动 LM Studio 服务器并加载特定 LLM 模型的帮助。
   - 推荐了 LM Studio SDK，通过其[文档](https://lmstudio.ai/docs/lmstudio-sdk/quick-start)和 [GitHub 仓库](https://github.com/lmstudio-ai/lmstudio.js)提供管理和自动化这些任务的方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://llm.extractum.io/list/?uncensored">Best Uncensored LLMs (Large Language Models): Explore the Curated List of the Best Uncensored LLMs</a>: 在我们的平台上发现顶级无审查 LLM 模型的高级功能。探索无审查 LLM 的功能、基准测试和内部机制，是处理复杂数据和敏感内容的理想选择...</li><li><a href="https://x.com/hellokillian/status/1723106008061587651)">来自 killian (@hellokillian) 的推文</a>: 直到现在我们简直是在盲目飞行 $ interpreter --vision &gt; 在 Tailwind CSS 中重构此组件（这是实时的）</li><li><a href="https://lmstudio.ai/docs/lmstudio-sdk/quick-start">快速入门指南 | LM Studio</a>: 开始使用 LM Studio SDK 的最小化设置</li><li><a href="https://github.com/lmstudio-ai/lmstudio.js">GitHub - lmstudio-ai/lmstudio.js: LM Studio TypeScript SDK (pre-release public alpha)</a>: LM Studio TypeScript SDK (预发布公开 Alpha 版) - lmstudio-ai/lmstudio.js</li><li><a href="https://github.com/huggingface/parler-tts">GitHub - huggingface/parler-tts: Inference and training library for high-quality TTS models.</a>: 高质量 TTS 模型的推理和训练库。- huggingface/parler-tts</li><li><a href="https://github.com/ggerganov/whisper.cpp">GitHub - ggerganov/whisper.cpp: Port of OpenAI&#39;s Whisper model in C/C++</a>: OpenAI Whisper 模型的 C/C++ 移植版本。欢迎在 GitHub 上为 ggerganov/whisper.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1275222028013797528)** (36 messages🔥): 

> - `M2 Ultra`
> - `GPU performance`
> - `Nvidia 4090 vs 4060 Ti`
> - `Nvidia 48GB Card`
> - `LLM speed` 


- **M2 Ultra 用于 AI 任务**：一位用户对即将推出的 M2 Ultra 表示兴奋，指出其在 AI 任务中的性能备受期待。
- **Nvidia 4090 vs 4060 Ti 用于 AI**：讨论转向了在 AI 应用中对比 Nvidia 4090 和两张 4060 Ti 的方案。
- **Nvidia 48GB 显卡传闻**：一位用户询问了 Nvidia 发布 48GB 消费级显卡的可能性。
- **AMD 7900GRE 在 LLMs 上的性能**：一位用户报告了 AMD 7900GRE 在 AI 任务中的性能缓慢，特别是运行 6B、7B 和 8B 模型时。
- **推荐用于 AI 的 Nvidia 显卡**：多位用户推荐使用 Nvidia 显卡，特别是 3090，以获得更快的 AI 性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/skeletor-laughs-in-evil-laughing-myah-myaah-dasmemeistgut-gif-5356566587527840753">Skeletor Laughs In Evil Laughing Myah Myaah Dasmemeistgut GIF - Skeletor laughs in evil laughing myah myaah dasmemeistgut - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1275410724910272576)** (6 messages): 

> - `GPT-4 Neuron Explanations`
> - `BlackboxNLP Paper`
> - `Language Models Explain Neurons` 


- **GPT-4 神经元解释被证伪？**：一位成员回忆起一篇论文，该论文声称 GPT-4 对神经元的解释并不比基准线（baselines）更有用或更好。 
   - 另一位成员提供了一篇题为 "Language Models can explain neurons in language models" 的论文链接，但未能找到标题类似且声称 GPT-4 解释无用的论文，尽管内容与该成员的回忆相似。
- **BlackboxNLP 论文探索**：一位成员试图寻找一篇声称 GPT-4 对神经元的解释并不比基准线更有用或更好的论文。 
   - 另一位成员搜索了 "Language Models can explain neurons in language models" 论文在 Google Scholar 上的引用，没有发现标题类似的此类论文。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1275178946245300268)** (48 条消息🔥): 

> - `Model Training`
> - `MiniPile Dataset`
> - `Frankenmerging`
> - `Model Merging`
> - `KANs` 


- **在有限数据上训练模型**：一位成员指出，由于随机初始化的影响，在单个小文件上训练模型可能会导致输出无意义的结果。
   - 他们进一步将其与文本压缩基准进行了比较，在这些基准中，模型被训练以尽可能有效地记忆特定的文本块。
- **用于高效训练的 MiniPile 数据集**：一位成员推荐了 MiniPile 数据集，作为完整 Pile 语料库的良好替代方案，因为后者对于学术预算来说通常过于庞大。
   - MiniPile 是 Pile 语料库的一个 6GB 子集，通过过滤掉低质量聚类进行精选，旨在为较小规模的模型提供多样化的预训练数据集。
- **Frankenmerging - 组合来自不同模型的层**：一位成员询问了组合来自两个不同模型的层的可行性，这种技术被称为“frankenmerging”。
   - 他们对为什么这不会导致模型内部表示完全错乱表示困惑，并寻求关于该方法的潜在收益和挑战的澄清。
- **使用优化器进行模型合并**：一位成员建议在将两个不同模型的层堆叠在一起之前，使用优化器寻找层间通道的最佳排列（permutation）。
   - 他们承认了潜在的挑战，并指出此类方法尚未在大规模 GPT 模型上得到验证。
- **Kolmogorov-Arnold Networks (KANs) - 一种新方法？**：讨论了一篇关于 KANs 的论文，该论文主张通过利用其识别相关特征、揭示模块化结构和发现符号公式的能力，实现与科学发现的无缝协同。
   - 讨论集中在 KANs 在特定任务上是否比 Transformer 或 CNN 具有优势，以及除了将其与 MLP 进行比较的单一实验外，是否缺乏支持其主张的有力证据。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2309.16039">Effective Long-Context Scaling of Foundation Models</a>：我们提出了一系列支持高达 32,768 个 token 有效上下文窗口的长上下文 LLM。我们的模型系列是通过对 Llama 2 进行更长训练序列的持续预训练而构建的...</li><li><a href="https://arxiv.org/abs/2402.01032">Repeat After Me: Transformers are Better than State Space Models at Copying</a>：Transformer 是序列建模的主流架构，但人们对使用不依赖于序列长度的固定大小潜在状态的模型（我们称之为...）的兴趣日益浓厚。</li><li><a href="https://arxiv.org/abs/2406.07887">An Empirical Study of Mamba-based Language Models</a>：像 Mamba 这样的选择性状态空间模型（SSM）克服了 Transformer 的一些缺点，例如随序列长度呈二次方增长的计算复杂度和推理时巨大的内存需求...</li><li><a href="https://arxiv.org/abs/2408.10205">KAN 2.0: Kolmogorov-Arnold Networks Meet Science</a>：AI + Science 的一个主要挑战在于它们固有的不兼容性：今天的 AI 主要基于联结主义，而科学依赖于符号主义。为了架起这两个世界的桥梁，我们提出了一个...</li><li><a href="https://huggingface.co/datasets/JeanKaddour/minipile">JeanKaddour/minipile · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1275457161304084551)** (4 条消息): 

> - `Chinchilla vs Gopher Data Filtering` 


- **Gopher 论文提到了数据过滤**：Gopher 论文讨论了数据去重和内容过滤，但 Chinchilla 尽管引用了 Gopher 论文，似乎并未特别提到这一点。
   - 两篇论文都使用了 MassiveText 数据集，但 Chinchilla 在 1.4T tokens 上进行了预训练，而 Gopher 在 300B tokens 上进行了预训练。
- **数据过滤策略存在差异**：一位用户指出，尽管都使用 MassiveText 数据集，Chinchilla 和 Gopher 在数据过滤技术上可能有所不同。
   - 他们对 Chinchilla 使用的具体数据过滤方法感到好奇，但该论文可能没有提供详细的见解。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1275169687810080808)** (21 条消息🔥): 

> - `Llama 3.1 System Prompt`
> - `Llama Eval Chat Template`
> - `Huggingface Chat Template`
> - `System Prompt in Huggingface`
> - `YAML Parameters` 


- **Llama 3.1 自动添加 System Prompt**：一位用户注意到 Llama 3.1 会自动添加 System Prompt，而 Llama 3 则不会。
   - Llama 3.1 自动添加的 System Prompt 内容如下：

*Cutting Knowledge Date: December 2023*
*Today Date: 26 Jul 2024*
- **System Prompt 覆盖**：用户询问当为 Llama 3.1 提供自定义 System Prompt 时会发生什么。
   - 已确认自定义 System Prompt 会与默认 System Prompt 拼接在一起。
- **Llama 3.1 Chat Template 中的 Bug**：该问题被怀疑是 Llama 3.1 Chat Template 或 `tokenizer.apply_chat_template` 方法中的一个 Bug。
   - 有建议认为此问题理想情况下应在 Huggingface 的上游进行修复。
- **使用自定义 Jinja Template 作为变通方案**：正在考虑使用自定义 Jinja Template 的变通方案。
   - 用户不确定是否存在更规范的方法来处理此问题及类似问题。
- **在 doc_to_text 函数中访问 YAML 参数**：用户询问如何在 `doc_to_text` 函数中访问 YAML 参数。
   - 建议使用 `process_docs` 函数添加一个所有文档都能访问的字段，然后在运行 `doc_to_text` 时访问给定文档的任何字段。


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1275187344814772244)** (56 条消息🔥🔥): 

> - `Perplexity Pro Discord`
> - `Perplexity Pro Features`
> - `Perplexity Pro Search Bugs`
> - `Perplexity AI Models`
> - `Perplexity's Future` 


- **Perplexity Pro Discord 访问困惑**：一些用户无法加入 Perplexity Pro Discord 服务器，即使退出并使用 Perplexity 设置中的链接重新加入也是如此。
   - 问题似乎在于缺乏关于如何在主 Discord 服务器内访问 Pro 栏目的清晰说明。
- **Perplexity 的搜索功能出现故障**：用户报告 Perplexity 的搜索功能存在问题，包括无法访问在线资源以及使用过时信息。
   - 一些用户认为这是一个后端问题，但 Perplexity 团队尚未承认或解决该问题。
- **Perplexity Pro 模型限制**：用户正在讨论 Perplexity Pro 模型在代码编写和博客文章创作等任务中的局限性。
   - 一些用户发现 Perplexity Pro 在某些任务上不如其他模型有效，特别是在生成复杂代码或避免博客文章中的 Hallucination（幻觉）方面。
- **Perplexity 对前端与后端的投入**：关于 Perplexity 是否优先考虑前端开发而非后端开发存在争论，一些用户报告了后端功能（如搜索和模型选择）的问题。
   - 一些用户认为这些问题表明缺乏对核心后端功能的关注，而这些功能对平台的整体性能至关重要。
- **Perplexity AI 图像生成**：用户正在询问 Perplexity Pro 生成 AI 图像的能力。
   - 然而，目前看来 Perplexity Pro 并不提供图像生成功能，尽管有关于未来潜力的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/hub/faq">Perplexity 常见问题解答</a>：如果您对 Perplexity 有疑问，我们的 FAQ 页面是寻找答案的绝佳去处。我们的 FAQ 页面按类别组织，提供清晰简洁的回答。</li><li><a href="https://www.perplexity.ai/hub/blog/pro-search-upgraded-for-more-advanced-problem-solving">Pro Search：升级以解决更高级的问题</a>：研究塑造了我们的日常生活。我们利用它来做出明智的决策并解决问题——去创新、学习和成长。</li><li><a href="https://x.com/rauchg/status/1825158716821320071?s=61">Guillermo Rauch (@rauchg) 的推文</a>：这非常令人兴奋，因为 App Router 和 RSC 的灵感之一就是让 Google Search 和 Facebook 等系统极其动态的渲染能力民主化。当你搜索“w...”时</li><li><a href="https://monnef.gitlab.io/by-ai">[By AI] 项目索引</a>：未找到描述</li><li><a href="https://www.perplexity.ai/search/see-uploaded-files-then-write-ZpLJtgqzRa6oO2byfwnY1Q">查看上传的文件。然后请根据此页面编写一个 WordPress 主题...</a>：根据上传的文件，我可以描述应用样式后的页面可能的样子：页面看起来具有复古、像素化的视频游戏风格...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1275307693459902517)** (5 条消息): 

> - `Perplexity Pro`
> - `LMSYS Arena`
> - `G1 Humanoid Robot` 


- **Perplexity Pro 功能升级**：讨论了升级到 [Perplexity Pro](https://www.perplexity.ai/pro) 的相关内容，该版本提供图片上传、更智能的 AI 以及更多 Pro Search 次数等功能。
- **LMSYS Arena Elo**：一位用户正在寻找关于 **LMSYS Arena** Elo 评分的统计数据。
- **G1 Humanoid Robot 准备量产**：据报道，[G1 Humanoid Robot](https://www.youtube.com/embed/EQxYdx79vUg) 已准备好投入大规模生产。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.perplexity.ai/search/in-baseball-if-a-batter-swing-UAK3snvlQjmIXxnVTM8N0Q">在棒球比赛中，如果击球手在第三次好球时挥棒落空，且投球是...</a>：是的，在棒球比赛中，如果击球手在第三次好球时挥棒落空且投球失控，跑垒员可以进垒。这种情况受“未捕获...”规则限制。</li><li><a href="https://www.perplexity.ai/search/how-to-use-claude-to-do-some-i-XgTfxNeARnS4GS0vjIXsZA">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，能为任何问题提供准确、可信且实时的答案。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1275431970662514709)** (4 条消息): 

> - `Camera quality`
> - `Discord issues` 


- **高端相机质量**：一位用户请求一张在雨中路灯下学习的贫穷男孩的照片，要求使用高端相机拍摄。
   - 这一请求暗示了对高图像质量、细腻对焦以及艺术构图的需求。
- **Discord 链接问题**：一位用户报告了 Discord 链接出现的问题，并分享了发生问题的特定频道链接。
   - 另一位用户询问是否有人遇到同样的问题，表明这可能是一个普遍存在的问题。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1275182877113323671)** (3 条消息): 

> - `LlamaIndex`
> - `RAG`
> - `Retrieval-Augmented Generation`
> - `LLMs in Production`
> - `Amazon Neptune` 


- **LlamaIndex：构建自然语言查询系统**：了解如何使用 LlamaIndex 和 Amazon Neptune 为图数据库构建自然语言查询系统！
   - 这份由 @bechbd 编写的综合指南将展示如何将自然语言问题转换为 openCypher 查询，并在 Amazon Neptune 图数据库上执行。
- **O'Reilly Media 的 RAG 课程**：LlamaIndex 推出了由 @ravithejads 编写的 O'Reilly Media 关于 Retrieval-Augmented Generation (RAG) 的课程。
   - 这门 2 小时的课程涵盖了 LlamaIndex 的组件、RAG 系统的评估、Ingestion Pipeline、可观测性、Agent、多模态等内容。
- **LLMs in Production：AI 产品见面会**：加入 LlamaIndex 参加由 @vesslai 和 @pinecone 在旧金山举办的 “LLMs in Production” AI 产品见面会。
   - 向行业领导者学习如何通过 RAG 和 Vector DB 构建上下文增强的 LLM，如何利用自定义 LLM 实现更智能、更快速、更廉价的解决方案，以及如何为生产级 LLM 实现高性能推理。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1275191931248443434)** (36 条消息🔥): 

> - `LlamaIndex Hierarchical Node Parser`
> - `LlamaIndex Retrieval`
> - `LlamaIndex ChromaDB Vector Store`
> - `Rag Application with LlamaIndex`
> - `Connecting LlamaIndex to Private LLMs` 


- **不进行分块的 Hierarchical Node Parser**：一位用户询问如何在不执行分块（chunking）的情况下使用 LlamaIndex 的 Hierarchical Node Parser，而是希望利用预定义节点创建层级结构和知识图谱。
   - 用户还希望保留与节点关联的元数据，例如页面 ID。在 Hierarchical Node Parser 的当前实现中，该场景被认为无法实现。
- **针对复杂问题的 LlamaIndex Retrieval**：用户讨论了针对简单和复杂问题的检索能力需求。
   - 他们设想了一种层级化方法，可以递归地总结节点并创建更高层级的数据表示，从而为复杂的查询提供更细致、更具上下文的响应。
- **ChromaDB 向量存储与 Top-K 检索**：一位用户询问如何在 LlamaIndex 中从 ChromaDB 向量存储中检索与查询最接近的前 K 个匹配项，类似于 LangChain 中的 `search_by_vector` 功能。
   - 用户解释了他们的使用场景：处理与数据库表相关的问题，然后使用向量识别最相关的表，以便后续执行 SQL 查询。用户发现 LlamaIndex 目前并未直接提供此功能。
- **将 LlamaIndex 连接到私有 LLM**：一位用户寻求关于将 LlamaIndex 连接到通过 HTTPS 访问的私有 LLM 的指导。
   - 他们遇到了 SSL 方面的挑战，并就如何使用 HTTPS 和 API token 连接到外部 LLM 实例请求建议。用户被引导至 LlamaIndex 中自定义 LLM 设置的指南以获取进一步帮助。
- **RAG 流水线设置：Semantic Chunking 与 Llama-parse**：用户讨论了构建稳健 RAG 流水线设置的各种方法，强调了 Semantic Chunking 和 Llama-parse 作为数据摄取（ingestion）潜在增强功能的重要性。
   - 他们询问了关于增强 SimpleDirectoryReader 和 VectorStoreIndex 等基础摄取方法的常见做法和建议，其中 Semantic Chunking 和 Llama-parse 分别被考虑用于生成空间文本和 Markdown 文本。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-using-a-custom-llm-model-advanced">Customizing LLMs - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/17f23014953e07eb8f8e7690d4cca7fb26c2109c/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py#L378">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py at 17f23014953e07eb8f8e7690d4cca7fb26c2109c · run-llama/llama_index</a>：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1275194209032470590)** (22 messages🔥): 

> - `Latent Space podcast`
> - `Encoder-Decoder models`
> - `Fast.html`
> - `Saving state/Updating state`
> - `Fine-tuning vs RAG vs KV caching` 


- **Jeremy Howard 最新的 Latent Space 访谈**：最新一期 Latent Space 播客采访了 Jeremy Howard，亮点包括对 Encoder-Decoder 模型、Fast.html、保存/更新状态、Fine-tuning vs RAG vs KV caching 的讨论，以及 Howard 正在进行的一个新项目。
   - 播客被描述为“五道菜的大餐”，这是在联合主持人 Swyx 俏皮地说出“给我们尝一小口”之后的评价。
- **Encoder-Decoder 模型 vs Encoder Only**：讨论强调了 Encoder-Decoder 模型优于 Encoder-only 模型的益处，特别是在详细上下文和复杂关系至关重要的场景中。
   - 受访者（可能通过 AI Paper Club 电话会议的潜移默化）已经了解了这种方法，这表明 AI 社区内的相关意识正在增强。
- **Whisperfile：音频转录新工具**：Whisperfile 是一款新工具，允许用户在本地轻松地将音频转换为文本。
   - 由 Justine Tunney 创建，Whisperfile 嵌入了 OpenAI 的 Whisper 模型，提供 100% 的本地运行，甚至可以在转录过程中将非英语音频翻译成英语。
- **Anthropic AI 提升 Claude 3.5 Sonnet 的输出限制**：Anthropic AI 将 Claude 3.5 Sonnet 的最大输出 Token 限制翻倍，从 4096 扩展到 8192。
   - 此更新现已在 Anthropic API 和 Vertex AI 中提供，使开发者能够更轻松地利用 Claude 3.5 Sonnet 的能力。
- **GPT-4 Fine-Tuning：新前沿**：OpenAI 发布了 GPT-4 Fine-tuning，这是一项允许用户自定义 GPT-4 行为和性能的新功能。
   - 这一更新为 Cursor 的 Composer 功能提供了潜在的竞争，因为它提供了一种类似的自定义和使用大型语言模型（LLM）的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://simple-bench.com]">未找到标题</a>：未找到描述</li><li><a href="https://x.com/simonw/status/1825626551398052180?s=46">来自 Simon Willison (@simonw) 的推文</a>：这是我尝试 whisperfile 的笔记，分别针对 tiny 和 medium 尺寸的 Whisper 模型 https://simonwillison.net/2024/Aug/19/whisperfile/ 引用 Justine Tunney (@JustineTunney) I j...</li><li><a href="https://x.com/ParallaxAngle/status/1825633740933955929">来自 JediCat (@ParallaxAngle) 的推文</a>：嗨 Howard 博士 @jeremyphoward，我非常喜欢你在最新一期 Latent Space 播客中与 @swyx 的讨论。28:30 之后的部分我听了两遍。我最喜欢的课题：1) Encoder-Decode...</li><li><a href="https://x.com/alexalbert__/status/1825920737326281184">来自 Alex Albert (@alexalbert__) 的推文</a>：我们已将其移出 Beta 阶段，因此你不再需要使用 header 了！现在已在 Anthropic API 和 Vertex AI 中为 Claude 3.5 Sonnet 提供。引用 Alex Albert (@alexalbert__) 好消息 f...</li><li><a href="https://x.com/justinetunney/status/1825594600528162818?s=46">来自 Justine Tunney (@JustineTunney) 的推文</a>：我刚刚发布了 whisperfile，这是将音频转换为文本最简单的方法。你只需下载一个嵌入了 OpenAI Whisper 模型的文件，即可 100% 在本地运行。它甚至可以翻译非英语内容...
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1275237689771425853)** (8 messages🔥): 

> - `Mojo & MAX Update Cadence`
> - `Siamese Networks with Labels`
> - `Slice Custom Op Usage` 


- **Mojo 与 MAX 版本同步**：此前，**Mojo** 和 **MAX** 拥有独立的更新周期，但现在它们已同步。
   - 这意味着你可以安装 **MAX+mojo main** 或 **MAX+mojo nightlies**，但不能分别安装 **MAX main** 和 **mojo nightlies**。
- **带标签的 Siamese Network？**：一位用户询问如何将 Siamese Network（孪生网络）的输出从 sigmoid 切换为标签（例如“狗”或“猫”）。
   - 另一位用户建议，如果你想切换到打标签，使用标准模型来完成该任务可能比尝试适配 Siamese Network 更有效。
- **Slice Custom Op 用法**：一位用户请求一个演示使用 **slice custom op** 的代码示例 (https://docs.modular.com/max/api/mojo/graph/ops/slicing/slice)。
   - 他们表示难以理解该算子的参数。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1275201216175013929)** (12 messages🔥): 

> - `Mojo's List implementation` (Mojo 的 List 实现)
> - `Mojo's ref keyword` (Mojo 的 ref 关键字)
> - `Mojo's __lifetime_of function` (Mojo 的 __lifetime_of 函数)
> - `AI Chip Performance` (AI 芯片性能)
> - `Network on Chip (NoC)` (片上网络)


- **Mojo 的 `List` 使用 `ref` 进行赋值**：一位用户惊讶地发现 Mojo 的 `List` 实现中没有用于赋值的 `__setitem__` 方法，但随后获知 `__getitem__` 会返回一个 `ref[lifetime] T`，其行为类似于 `__setitem__`。
- **Mojo 的 `ref` 和 `__lifetime_of` 是新特性**：函数返回类型中的 `ref` 关键字是最近（在 Mojo v244 中）作为新语言特性引入的。
- **AI 芯片通过本地内存提升性能**：AI 芯片设计有大量本地内存，以便将模型放入缓存中，从而减少频繁向 RAM 传输数据的开销。
- **NoC 与缓存管理的权衡**：虽然 NoC 在多核之间提供了高效的数据传输，但它也会在核心之间引入延迟。
- **Mojo 的生产就绪性**：有人提出了 Mojo 何时能达到生产就绪状态的问题。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1275349492060786749)** (2 messages): 

> - `Modular installation issues` (Modular 安装问题)
> - `Modular Manifest Error` (Modular Manifest 错误)
> - `Modular Expiration` (Modular 过期)


- **Modular 安装错误**：一位用户报告在尝试使用 `modular install max` 命令安装 "max" 模块时遇到了一系列错误。
- **故障排除步骤**：该用户尝试通过先使用 `modular clean` 命令清理 Modular 环境，然后重新安装 "max" 模块来解决问题。
- **请求 Modular 版本信息**：另一位用户建议使用 `modular -v` 命令检查 Modular 版本，以潜在地识别错误原因。


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1275176727647092809)** (19 messages🔥): 

> - `ChatGPT capabilities` (ChatGPT 能力)
> - `AI Enthusiasm` (AI 热情)
> - `Grok2`
> - `Smart Cookbook` (智能食谱)
> - `Strawberry Release` (Strawberry 发布)


- **ChatGPT 无法数清 Strawberry 中的 'R'**：一位用户指出 ChatGPT 在处理诸如计算单词 "strawberry" 中 'R' 的数量等简单任务时表现挣扎，暗示 AI 并不像某些人认为的那样先进。
   - 这引发了关于 AI 当前局限性以及它是真正的智能还是仅仅是一个可以执行特定任务的工具的讨论。
- **Grok2 有趣的方法**：一位用户提到 Grok2 在处理问题时有一种有趣的方法。
   - 另一位用户指出 Grok2 的方法涉及分解每个问题并逐步解决，这与人类解决问题的方式类似。
- **AI 热情被高估了？**：一位用户表示，由于 AI 当前的局限性，“AI 爱好者”一词已经失去了意义。
   - 这种情绪源于关于 ChatGPT 在简单任务上的挣扎以及 Grok2 解决问题方法的讨论。
- **构建智能食谱**：一位用户寻求关于创建一个“智能食谱”的建议，该食谱可以根据他们最喜欢的食谱进行训练并提供个性化建议。
   - 该用户认为这种模型可以应用于任何“操作指南”类书籍，并请求有关现有解决方案或项目的信息。
- **Strawberry 发布推测**：一位用户询问了 "Strawberry" 的发布日期，这可能是一个新的 AI 模型或功能。
   - 另一位用户开玩笑地回应说 "Strawberry" 仍处于“来源不可靠的泄露”阶段，并对其发布表示怀疑。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1275553419423383552)** (1 messages): 

> - `Structured Output` (结构化输出)
> - `JSON Output` (JSON 输出)
> - `Model Performance` (模型性能)
> - `Prompt Engineering`


- **结构化输出 vs. JSON 输出**：一位用户注意到结构化输出有时给出的响应比常规 JSON 模式更差。
- **理解差异**：讨论集中在探索结构化输出和 JSON 模式之间响应质量差异的潜在原因。
- **对 Prompt Engineering 的影响**：对话强调了理解这些差异对于进行有效的 Prompt Engineering 的重要性。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1275553419423383552)** (1 messages): 

> - `structured output`
> - `JSON mode` 


- **Structured Output vs. JSON Mode**: 用户注意到 `structured output` 有时生成的响应比常规的 `JSON mode` 更差。
   - 用户未就此话题提供更多细节或见解。
- **调查差异**: 了解为什么 `structured output` 可能会产生比 `JSON mode` 逊色的响应非常重要。
   - 需要进一步分析以确定此问题的根本原因。


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

wiiiktor.: 请问你们计划什么时候发布？我看它已经完成了 99%。
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1275477856486686804)** (19 messages🔥): 

> - `torch.compile recompilations`
> - `torch.compile optimization`
> - `kv-cache for generation`
> - `rng generator object in torch.compile`
> - `torch.compile and custom masks` 


- **输入形状变化导致 Torch.compile 重新编译**: 当输入形状发生变化时（例如在生成过程中拼接新 token），会触发重新编译。
- **`grad_mode` 变化导致 Torch.compile 重新编译**: 在训练和推理模式之间切换时，由于 `grad_mode` 发生变化，会发生重新编译。
- **Torch.compile 缓存大小限制**: 触发 `torch._dynamo hit config.cache_size_limit (8)` 消息表明可能存在 `torch.compile` 友好性问题。
- **Torch.compile 对 RNG 对象的限制**: 将 RNG generator 对象传入模型会导致 graph breaks，这表明 `torch.compile` 目前不支持此类对象。
- **自定义 mask 与 kv-cache**: 使用自定义 mask 可能无法直接与 `kv-cache` 兼容，但利用你自己的 mask 并移除 `self.causal_mask` 可能会解决该问题。


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1275236543803424769)** (18 messages🔥): 

> - `LLaMA 3.1 70B for SQL`
> - `Mistral 8k Limits`
> - `Model Merging Tactics`
> - `Open Empathic Project`
> - `LangChain SQLDatabaseChain` 


- **LLaMA 3.1 70B 在 SQL 任务上表现不佳**: 有用户报告 [LLaMA 3.1 70B](https://ai.google.com/research/pubs/pub49727.html) 无法使用 [LangChain 的 SQL agent](https://langchain.readthedocs.io/en/latest/modules/agents/agents.html#sql-agent) 查询数据库，而 [GPT 3.5](https://platform.openai.com/docs/models/gpt-3-5) 在相同配置下能成功完成任务。
   - 他们尝试了各种解决方案，如使用自定义解析器，但问题仍然存在，这让他们认为问题在于 LLaMA 的能力。
- **Mistral 难以扩展超过 8k**: 用户报告 [Mistral](https://www.mistral.ai/) 在没有持续预训练的情况下无法扩展到 8k 以上。
   - 用户建议进一步研究 *mergekit* 和 *frankenMoE finetuning* 作为潜在的解决方案。
- **模型合并策略讨论**: 用户建议了一种潜在的模型合并策略，即将 **UltraChat** 和基础 **Mistral** 之间的差异应用到 **Mistral-Yarn** 上。
   - 其他成员表示怀疑，但该用户保持乐观，并引用了过去成功的“诡异模型合并 (cursed model merging)”尝试。
- **Open Empathic 项目寻求协助**: 用户请求帮助扩展 **Open Empathic** 项目的类别，特别是低端类别。
   - 他们分享了一个 [YouTube 视频](https://youtu.be/GZqYr8_Q7DE) 展示了项目的启动和教程，引导用户从 YouTube 视频中贡献首选的电影场景，并附上了 [OpenEmpathic 项目](https://dct.openempathic.ai/) 的链接。
- **LangChain 的 SQLDatabaseChain**: 用户介绍了 [LangChain 的 SQLDatabaseChain](https://langchain.readthedocs.io/en/latest/modules/chains/sql_database_chain.html)，这是一个根据用户提示词生成 SQL 查询的实验性功能。
   - 他们分享了一个使用该功能的函数代码示例，提供了一个用于生成 SQL 查询和处理链响应的提示词模板。



**Link mentioned**: <a href="https://mydomain.com']">no title found</a>: no description found

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1275167433082146848)** (16 条消息🔥): 

> - `Accessibility Roundtable`
> - `Deepseek API`
> - `OI 与 Ollama`
> - `Mac 上的 Poetry 和 Pytorch`
> - `在不同机器上运行 Ollama` 


- **Accessibility Roundtable 提醒**：关于本周四 Accessibility Roundtable 的提醒，并附带了活动链接。
- **Deepseek API 对比 OpenAI 和本地 LLM**：一位用户询问是否有关于使用 Deepseek API 替代 OpenAI 或本地 LLM 的指南。
- **在不同机器上将 Ollama 与 OI 集成**：一位用户寻求关于在 Ollama 未托管在 localhost 时如何将其与 OI 配合使用的指导。
   - 具体来说，他们询问了如何配置 profile YAML 以及如何使用该 profile 启动 interpreter。
- **Mac 上的 Poetry 和 Pytorch 安装问题**：一位用户报告了在 Mac 上安装 Poetry 和 Pytorch 2.3.0 时遇到的麻烦，并提到有一个未得到回复的公开 issue。
   - 他们请求指导以寻找该问题的解决方案。
- **OpenInterpreter 的 API_BASE 配置**：随后讨论了如何为 OpenInterpreter 配置 API_BASE，以便与另一台机器上的 Ollama 配合工作。
   - 一位用户确认他们尝试使用了 Ollama 实例的正确 IP 地址和端口，但 OpenInterpreter 仍然拒绝连接。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://host:port">未找到标题</a>：未找到描述</li><li><a href="http://10.0.0.4:11434)">未找到标题</a>：未找到描述</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/tree/main/interpreter/core/computer">GitHub - OpenInterpreter/open-interpreter/interpreter/core/computer</a>：一个计算机的自然语言接口。通过在 GitHub 上创建一个账户来为 OpenInterpreter/open-interpreter 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1275314289904975943)** (1 条消息): 

> - `OpenInterpreter 更新` 


- **OpenInterpreter 更新**：最新的 OpenInterpreter 更新可通过[此链接](https://discord.com/channels/1146610656779440188/1194880263122075688/1271135268807905384)获取。
   - 未提供更多细节。
- **OpenInterpreter 更新 2**：未提供更多细节。
   - 未提供更多细节。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 条消息): 

notnaton：来自 Tool Use 的最新一集 🚀：https://www.youtube.com/watch?v=uAo513GIwoU
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1275292714803662960)** (2 条消息): 

> - `dspy-ai 安装`
> - `ADAS 和 Function Calling`
> - `pickle5 兼容性`
> - `Python 版本` 


- **dspy-ai 安装困惑**：一位用户注意到 `requirements.txt` 文件中列出了 `dspy==2.0.5`，但质疑是否实际上应该是 `dspy-ai`。
   - 他们还指出 `pickle5==0.0.12` 存在潜在的兼容性问题，该版本兼容 3.8 以下的 Python 版本，而 `dspy-ai` 需要 Python 3.9 或更高版本。
- **ADAS 能否发明新的构建块？**：一位用户询问 ADAS 是否可以发明新的构建块，例如向集成系统进行 Function Calling。
   - 他们还询问是否已经有人进行过类似的实验。

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1275171945666711603)** (9 messages🔥): 

> - `DSPy Finetuning`
> - `DSPy vs. Langchain/LLamaindex`
> - `Aider v0.51.0 更新日志`
> - `为 DSPy 文档提供反馈`
> - `Multi-Lora 设置` 


- **使用 Multi-Lora 进行 DSPy Finetuning**：一位用户询问了在 DSPy Finetuning 中使用 Multi-Lora 设置的潜力，认为这可能是一个有价值的方法。
- **DSPy vs. Langchain/LLamaindex**：一位用户咨询了 DSPy 与 Langchain 和 LLamaindex 的对比，并被引导至 DSPy 文档以获取选择合适工具的指导。
- **Aider v0.51.0：增强的 Prompt Caching 和 Repo Mapping**：Aider 发布了 v0.51.0 版本，其特点包括针对 Anthropic 模型的改进版 Prompt Caching、针对大型仓库优化的 Repo Mapping，以及增强的 Jupyter Notebook .ipynb 文件编辑功能。
   - 该版本包含多项 Bug 修复和改进，Aider 为该版本贡献了 56% 的代码。
- **为 DSPy 文档提供反馈**：一位用户询问了为 DSPy 文档提供反馈的最佳方式。
   - 建议是提交 Pull Request 或 Issue，并在标题中引用 Roadmap。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/paulgauthier/status/1825934199465119803">Paul Gauthier (@paulgauthier) 的推文</a>：Aider v0.51.0 - 针对 Anthropic 模型的 Prompt caching（使用 --cache-prompts）。- 大型/单体仓库中的 Repo map 加速。- 改进的 Jupyter Notebook .ipynb 文件编辑。- Aider 编写了该版本中 56% 的代码...</li><li><a href="https://aider.chat/HISTORY.html#v0510>>>">发布历史</a>：关于 Aider 编写自身代码的发布说明和统计数据。
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1275506387602116650)** (1 messages): 

> - `Late Interaction 模型`
> - `Dense Embedding 模型`
> - `Qdrant 1.10`
> - `ColBERT` 


- **Qdrant 1.10 新增对 Multi-Vector 表示的支持**：Qdrant 1.10 引入了对 Multi-Vector 表示的支持，Late Interaction 是该模型的一个显著示例。 
   - 识别相关文档涉及根据对应的 Query 和文档 Embedding 之间的相似度计算得分。
- **深入了解 Late Interaction**：Late Interaction 模型（如 ColBERT）根据对应的 Query 和文档 Embedding 之间的相似度计算得分。 
- **将 Dense Embedding 模型适配为 Late Interaction**：常规的 Dense Embedding 模型可以通过移除 Pooling 步骤并使用 Token 级别的 Embedding 进行检索/重排序（Reranking），从而适配 Late Interaction。
- **Hybrid Search 详解**：更新后的 [Hybrid Search](https://qdrant.tech/articles/hybrid-search/) 文章解释了 Multi-Vector 表示如何增强检索质量。



**提及的链接**：<a href="https://qdrant.tech/articles/late-interaction-models/">任何 Embedding 模型都可以成为 Late Interaction 模型 - 只要你给它机会！- Qdrant</a>：我们发现了一些有趣的事情。标准的 Dense Embedding 模型在 Late Interaction 场景中表现得惊人地好。

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1275495655615234088)** (1 messages): 

> - `LTXStudio 新功能` 


- **LTXStudio 推出五项新功能**：LTXStudio 发布了五项新功能，旨在帮助用户将项目提升到新的水平。
   - 这些功能现在即可访问和测试。
- **LTXStudio 用于增强项目的新功能**：LTXStudio 推出了五项新功能，旨在将项目提升到新的水平。
   - 这些功能现在可供用户探索和利用。



**提及的链接**：<a href="https://x.com/LTXStudio/status/1825909655207383308?t=5Wk2X8i_lQ5R5HAJxcerlg&s=19">LTX Studio (@LTXStudio) 的推文</a>：🎉 等待结束了 🎉 为了庆祝，我们推出了五项新功能，将您的项目提升到新的水平。现在就亲自尝试吧 🔥

  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1275204038178111710)** (5 条消息): 

> - `JPEG Encoding for Images`
> - `AR-Based Image Tokenization`
> - `VQ-VAE`
> - `Image Compression Limits`
> - `H.265/AV1 for Training` 


- **JPEG Encoding：一种可行的图像 Tokenization 方法？**：一篇研究论文指出 JPEG 编码可能是一种很好的图像 Tokenization 方法，但目前的 AR-based 方法面临严重的信息丢失，导致图像质量较差。
   - 该论文使用了 25 的 JPEG 质量设置，这使得从 Token 生成高质量图像在理论上是不可能的，并将一张 256*256 的图像压缩到约 5,000 个 Token，导致与传统的 VQ-VAE 相比，训练和推理时间更长。
- **关于图像压缩极限的不确定性**：作者对图像可实现的最大压缩率表示怀疑，因为该论文在 Tokenization 时使用了 25 的 JPEG 质量设置。
- **在 H.265 或 AV1 帧上训练模型**：作者建议探索在 H.265 帧甚至 AV1 帧上训练模型的可能性，作为 JPEG 编码的替代方案。


  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1275491282713579521)** (1 条消息): 

> - `GPT4All-Community`
> - `Leo models`
> - `Hugging Face`
> - `Model Card` 


- **Leo 模型公开可用**：一名成员在 Hugging Face 上公开了其 [Leo 模型的量化版本](https://huggingface.co/GPT4All-Community)。
   - 如果需要，他们愿意接受反馈并向用户转达信息，并根据需要将其添加到 Model Card 中。
- **通过 Model Card 进行反馈和更新**：该成员提议在 Model Card 中添加信息，以便进行反馈或向用户传递信息。


  

---



### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 条消息): 

xeophon.: https://x.com/bilawalsidhu/status/1825548322687574410?s=46
  

---



---



---



---



---



{% else %}


> 完整的逐频道详情已在邮件中截断。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}