---
companies:
- google-deepmind
- yi-ai
- microsoft
- hugging-face
- langchain
- maven
date: '2024-05-20T23:02:42.432305Z'
description: '在2024年5月17日至5月20日期间，AI领域的重要更新如下：


  **Google DeepMind发布了Gemini 1.5 Pro和Flash模型**，其采用了稀疏多模态混合专家（MoE）架构，支持高达**1000万（10M）的上下文窗口**，并配备了稠密Transformer解码器，使速度提升了**3倍**，成本降低了**10倍**。**零一万物（Yi
  AI）发布了Yi-1.5模型**，其上下文窗口扩展至**32K和16K tokens**。其他值得关注的发布还包括：**Kosmos 2.5（微软）、PaliGemma（谷歌）、Falcon
  2、DeepSeek v2 lite以及混元DiT（HunyuanDiT）扩散模型**。


  在研究亮点方面：**《观察性缩放法则》（Observational Scaling Laws）论文**提出了预测跨系列模型性能的方法；**“层压缩KV缓存”（Layer-Condensed
  KV Cache）技术**可将推理吞吐量提升**高达26倍**；而**SUPRA方法**则通过将大语言模型（LLM）转换为循环神经网络（RNN）来降低计算成本。


  在生态与社区方面：Hugging Face扩展了本地AI能力，实现了无需依赖云端的设备端AI。LangChain更新了其v0.2版本，并改进了文档。此外，社区还迎来了由Hamel
  Husain和Dan Becker为Maven课程用户创建的新LLM微调Discord频道。**“Hugging Face已实现盈利，或接近盈利”**，这使其能够为开发者提供价值1000万美元的免费共享GPU资源。'
id: a823d9be-3c00-44e1-85ae-271beb397f02
models:
- gemini-1.5-pro
- gemini-1.5-flash
- yi-1.5
- kosmos-2.5
- paligemma
- falcon-2
- deepseek-v2
- hunyuan-dit
- gemini-1.5
- gemini-1.5-flash
- yi-1.5
original_slug: ainews-to-be-named-3447
people:
- hamel-husain
- dan-becker
- clement-delangue
- philschmid
- osanseviero
- arankomatsuzaki
- jason-wei
- rohanpaul_ai
title: "“Skyfall” 根据语境有以下几种常见的中文翻译：\n\n1.  **电影名称**（第23部詹姆斯·邦德电影）：\n    *   中国大陆：**《007：大破天幕杀机》**\n\
  \    *   香港：**《新铁金刚：智破天凶城》**\n    *   台湾：**《007：空降危机》**\n\n2.  **字面意思**：\n    *\
  \   **天崩地裂**\n    *   **天塌**\n\n3.  **片中地名**（邦德在苏格兰的祖宅）：\n    *   **天幕庄园**\n\n4.\
  \  **同名主题曲**（阿黛尔演唱）：\n    *   通常直接称为 **《Skyfall》** 或 **《天幕杀机》**。"
topics:
- multimodality
- mixture-of-experts
- transformer
- model-optimization
- long-context
- model-performance
- model-inference
- fine-tuning
- local-ai
- scaling-laws
- causal-models
- hallucination-detection
- model-distillation
- model-efficiency
---

<!-- buttondown-editor-mode: plaintext -->**不再思考 ~~superalignment~~ ~~Google~~ ~~Scarlett Johansson~~ 就足够了。**

> 2024年5月17日至5月20日的 AI 新闻。
我们为你检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**366** 个频道，共 **9564** 条消息）。
预计节省的阅读时间（按 200wpm 计算）：**1116 分钟**。

虽然这是一个相对活跃的周末，但大多数争论都是非技术性质的，没有任何公告明显适合作为本次的主打特性。

因此，这里列出了一些次要笔记：

- 我们弃用了一些不活跃的 Discord，并**添加了 Hamel Husain 和 Dan Becker 为其热门 Maven 课程开设的新 LLM Finetuning Discord**（[此处为合作链接](https://maven.com/parlance-labs/fine-tuning?utm_campaign=29ce77&utm_medium=partner&utm_source=instructor)）
- [HuggingFace 的 ZeroGPU](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai?utm_source=ainews&utm_medium=email) 已在 Hugging Face 的 Spaces 上线，承诺提供价值 1000 万美元的免费共享 GPU，以帮助开发者创建新的 AI 技术，因为 Hugging Face 已经“盈利，或接近盈利”
- LangChain 在发布 [v0.2 版本](https://blog.langchain.dev/langchain-v02-leap-to-stability/) 后，进行了急需的[文档更新](https://blog.langchain.dev/documentation-refresh-for-langchain-v0-2/)
- [Omar Sanseviero 的推文串](https://x.com/osanseviero/status/1792273392839557288?utm_source=ainews&utm_medium=email) 总结了上周发布的小型模型（其中一些我们在 AInews 中报道过）——BLIP3、Yi-1.5、Kosmos 2.5、Falcon 2, PaliGemma、DeepSeekV2 等

但别开玩笑了，你可能更想读读 [Scarlett 用备忘录对 OpenAI 的抨击](https://x.com/BobbyAllyn/status/1792679435701014908) (:

 
![image.png](https://assets.buttondown.email/images/89e806ac-a369-415c-8b42-14465dfc9877.png?w=960&fit=max)
 

---

**目录**

[TOC] 

---

# AI Twitter 回顾

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中选取最佳。我们正在尝试使用 Haiku 进行聚类和流程工程。

**AI 模型发布与更新**

- **Google DeepMind 发布 Gemini 1.5 Pro 和 Flash 模型**：[@_philschmid](https://twitter.com/_philschmid/status/1792528829040251147) 分享道，Gemini 1.5 Pro 是一个稀疏多模态 MoE 模型，可处理文本、音频、图像和视频，支持高达 10M 的 context，而 Flash 是从 Pro 蒸馏出的稠密 Transformer 解码器模型，速度**快 3 倍且价格便宜 10 倍**。两者均支持高达 2M token 的 context。
- **Yi AI 发布具有更长 context 的 Yi-1.5 模型**：[@01AI_Yi](https://twitter.com/01AI_Yi/status/1792386612430774510) 宣布发布具有 **32K 和 16K context 长度**的 Yi-1.5 模型，已在 Hugging Face 上线。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1792504986049376609) 强调了其更长的 context 窗口。
- **其他值得关注的模型发布**：[@osanseviero](https://twitter.com/osanseviero/status/1792273392839557288) 回顾了本周的开源机器学习更新，包括 **Microsoft 的 Kosmos 2.5、Google 的 PaliGemma、CumoLLM、Falcon 2、DeepSeek v2 lite、HunyuanDiT 扩散模型以及 Lumina next**。

**研究论文与技术**

- **Observational Scaling Laws 论文泛化了计算 Scaling Laws**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1792384879742877943) 和 [@_jasonwei](https://twitter.com/_jasonwei/status/1792401639552565496) 讨论的这篇论文使用共享的低维能力空间处理多个模型家族，展示了**对模型性能令人印象深刻的预测能力**。
- **Layer-Condensed KV Cache 实现高效推理**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1792386318300749848) 分享了一篇关于该技术的论文，该技术实现的 **LLM 吞吐量比标准 Transformer 高出 26 倍**。
- **健壮的 Agent 学习因果世界模型**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1792178928649404644) 总结了一篇论文，该论文表明在分布偏移下满足遗憾边界（regret bounds）的 Agent 必须**学习数据生成过程的近似因果模型**。
- **使用 SUPRA 方法将 LLM 线性化**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1792561008738791797) 分享了一篇关于 SUPRA 的论文，该方法**将预训练的 LLM 转换为 RNN，显著降低了计算成本**。
- **研究微调后 LLM 的幻觉问题**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1792149331186761877) 总结了一篇论文，该论文表明**通过微调引入新知识可能会对幻觉倾向产生意想不到的后果**。

**框架、工具与平台**

- **Hugging Face 扩展本地 AI 能力**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1792570357645271466) 宣布了 **Hugging Face 的本地 AI 新功能，无需云端、无成本且无需向外部发送数据**。
- **LangChain v0.2 发布，包含重大文档改进**：[@LangChainAI](https://twitter.com/LangChainAI/status/1792596301915599059) 和 [@hwchase17](https://twitter.com/hwchase17/status/1792598084968382856) 强调了此次发布，包括 **版本化文档、更清晰的结构、整合的内容以及升级指南**。
- **Cognita 框架基于 LangChain 构建，用于模块化 RAG 应用**：[@LangChainAI](https://twitter.com/LangChainAI/status/1792218404838850662) 分享了这一 **开源框架，为构建 RAG 应用程序提供了开箱即用的体验**。
- **Together Cloud 为大规模模型训练增加了 H100 GPU**：[@togethercompute](https://twitter.com/togethercompute/status/1792593306159112494) 宣布在其 **供 AI 公司使用的集群中增加了 6,096 块 H100 GPU**。

**讨论与观点**

- **幻觉（Hallucinations）是 LLM 投入生产的阻碍**：[@realSharonZhou](https://twitter.com/realSharonZhou/status/1792576516444065967) 指出幻觉是一个主要障碍，但分享了 **通过微调 LLM 以“照相式记忆”召回细节，已将幻觉率降至 5% 以下**。
- **Anthropic 反思负责任扩展政策（Responsible Scaling Policy）的进展**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1792598295388279124) 在 **继续迭代其框架** 的过程中分享了反思。
- **RAG 应用面临的挑战**：[@jxnlco](https://twitter.com/jxnlco/status/1792593174755422466) 预约了专家咨询以寻求帮助，[@HamelHusain](https://twitter.com/HamelHusain/status/1792579262677180609) 分享了 **即将举行的 RAG 工作坊** 的细节。
- **目前 LLM 最大的用例**：[@fchollet](https://twitter.com/fchollet/status/1792316976620278154) 列出了前三大用例：**替代 StackOverflow、写作业以及企业内部知识库**。

**梗与幽默**

- **关于用贪吃蛇游戏测试 LLM 编程能力的梗**：[@svpino](https://twitter.com/svpino/status/1792564474190131362) 调侃说 **源代码在 Google 上很容易找到，所以不需要用 LLM 来写**。
- **关于 AI 女友应用的梗**：[@bindureddy](https://twitter.com/bindureddy/status/1792409279066186074) 调侃说，**尽管发明巨型 AI 模型是为了“解开宇宙之谜”，但它们却是使用 LLM 的最大类消费级应用**。
- **关于开源 AGI 以防止性能削弱的梗**：[@bindureddy](https://twitter.com/bindureddy/status/1792566986347831352) 引用电影《她》（Her）调侃说，首要原因是 **防止模型被削弱（nerfed）和审查**。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 进展与能力**

- **Apple 与 OpenAI 合作**：在 /r/MachineLearning 中，据报道 [**Apple 正与 OpenAI 合作，为 iOS 18 添加 AI 技术，预计将在 WWDC 上发布重大公告**](https://www.bloomberg.com/news/newsletters/2024-05-19/what-is-apple-doing-in-ai-summaries-cloud-and-on-device-llms-openai-deal-lwdj5pkz?srnd=undefined)。
- **OpenAI 研究重点的转移**：/r/MachineLearning 讨论了 [**OpenAI 如何从 DOTA2 Open Five 等令人兴奋的研究，转向使用 GPT-4 和 GPT-4o 预测序列中的下一个 Token (next token prediction)，这可能是由于财务状况和盈利需求所致**](https://www.reddit.com/r/MachineLearning/comments/1cvslyc/d_how_did_openai_go_from_doing_exciting_research/)。
- **GPT-4o 的图像描述能力**：在 /r/OpenAI 中，用户注意到 [**GPT-4o 拥有比以往模型更优越的图像描述能力，能够准确理解绘画风格、时间点、情绪和氛围**](https://www.reddit.com/r/OpenAI/comments/1cw2p2f/gpt4o_vastly_superior_image_description/)。

**AI 安全与对齐**

- **OpenAI 解散 AI 安全团队**：在 /r/OpenAI 中，据报道 [**OpenAI 已解散其 Superalignment AI 安全团队**](https://www.cnbc.com/2024/05/17/openai-superalignment-sutskever-leike.html)。
- **非常规 AI 攻击向量**：一篇文章讨论了失控的 AI 可能会使用非常规攻击向量，例如通过破坏浮游植物来摧毁生态系统，而不是使用生物武器或核风险。
- **对齐 AI 中的不诚实行为**：根据一篇文章，即使是仁慈对齐的超人工智能，为了实现超越人类理解的目标，也可能需要采取不诚实和操纵手段。

**AI 对就业和经济的影响**

- **AI 冲击劳动力**：在 /r/economy 中，IMF 负责人表示 [**AI 正在像“海啸”一样冲击劳动力市场**](https://www.reuters.com/technology/artificial-intelligence-hitting-labour-forces-like-tsunami-imf-chief-2024-05-13/?utm_source=reddit.com)。
- **全民基本收入 (UBI)**：“AI 教父”认为 [**由于 AI 的影响，将需要全民基本收入**](https://www.bbc.co.uk/news/articles/cnd607ekl99o)。其他帖子讨论了实施 UBI 的可行性和时机挑战。

**AI 模型与框架**

- **Smaug-Llama-3-70B-Instruct 模型**：在 /r/LocalLLaMA 中，[**Smaug-Llama-3-70B-Instruct 模型发布，该模型仅在特定数据集上进行训练，在 Arena-Hard 基准测试中表现出色**](https://www.reddit.com/r/LocalLLaMA/comments/1cvly7e/creator_of_smaug_here_clearing_up_some/)。
- **Yi 1.5 长上下文版本**：[**Yi 1.5 16K 和 32K 长上下文版本已发布**](https://twitter.com/01AI_Yi/status/1792386612430774510?t=rwxRESA-YMSYRzkzyX8hzQ&s=19)。
- **Level4SDXL alphaV0.3**：[**Level4SDXL alphaV0.3 作为一款无需 LoRA/refiners/detailers 的全能模型发布**](https://www.reddit.com/gallery/1cw5zan)。

**AI 伦理与社会影响**

- **OpenAI 暂停 "Sky" 语音**：在被质疑模仿 Scarlett Johansson 后，OpenAI [**暂停了在 GPT-4o 中使用 "Sky" 语音**](https://openai.com/index/how-the-voices-for-chatgpt-were-chosen/)。
- **AI 生成情色内容的隐私担忧**：使用 AI 服务生成情色内容的用户可能会意识到他们的查询并非私密，因为数据会被发送到 API 进行处理。
- **BlackRock 在欧洲的 AI 投资**：[**BlackRock 正与各国政府就投资电力以满足欧洲 AI 需求进行谈判**](https://www.reuters.com/technology/blackrock-ceo-sees-giant-issue-europe-due-ai-power-needs-2024-05-17/)。

---

# AI Discord 回顾

> 摘要之摘要的总结

1. **LLM 微调的进展与挑战**：
   - [Unsloth AI](https://github.com/unslothai/unsloth) 通过优化技术实现了对 **Llama-3-70B Instruct** 等模型的有效**微调 (fine-tuning)**，但讨论中也涉及了围绕 IP 使用的法律担忧，例如 [Scarlett Johansson 起诉 OpenAI](https://www.theverge.com/2024/5/20/24160621/openai-chatgpt-gpt4o-sky-scarlett-johansson-voice-assistant-her)。
   - [LLM 微调课程](https://maven.com/parlance-labs/fine-tuning)引发了关于质量的辩论，一些人认为初始内容过于基础，而另一些人则赞赏其在训练、评估和提示工程 (prompt engineering) 方面的实战方法。
   - 关于 [LoRA](https://arxiv.org/abs/2405.09673) 微调的讨论强调了最佳配置、dropout、权重衰减 (weight decay) 和学习率，以防止在 3090 等 GPU 上出现过拟合，正如[这条推文](https://x.com/cwolferesearch/status/1788998798414410032)所分享的那样。

2. **多模态与生成式 AI 创新**：
   - [Hugging Face](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai) 承诺提供 **1000 万美元的免费 GPU**，以支持小型开发者、学术界和初创公司创建新的 AI 技术。
   - 来自 Meta 的 [Chameleon 模型](https://arxiv.org/abs/2405.09818)展示了在同时理解和生成图像与文本方面的 SOTA（最先进）性能，超越了 Llama-2 等更大的模型。
   - [GPT-4o 与 LlamaParse 的集成](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a)实现了多模态能力，同时其[中文 Token 污染](https://www-technologyreview-com.cdn.ampproject.org/c/s/www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/amp/)问题也引起了关注。
   - [4Wall AI](https://beta.4wall.ai) 和 [AI Reality TV](https://x.com/edgarhnd/status/1791586276178587707) 等创新项目探索了由 AI 驱动、包含用户生成内容和社交模拟的娱乐平台。

3. **开源数据集与模型开发**：
   - 针对 [CommonCanvas 数据集](https://x.com/multimodalart/status/1791201296357142663)限制性的非商业许可证，不满情绪日益增加，该许可证限制了修改和衍生作品。
   - 努力集中在创建高质量的开源数据集上，例如避免标注中的幻觉 (hallucinations)，因为这会损害视觉语言模型 (VLLM) 和文本生成图像 (T2I) 模型。
   - [Sakuga-42M 数据集](https://arxiv.org/abs/2405.07425)推出了首个大规模卡通动画数据集，填补了卡通特定训练数据的空白。
   - 针对 [CogVLM2 许可证](https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE)提出了担忧，该许可证限制了不利于中国利益的使用，并规定争议受中国管辖。

4. **AI 安全、伦理与人才招聘**：
   - 核心研究员如 [Jan Leike 辞去](https://x.com/janleike/status/1791498174659715494)了 OpenAI 对齐 (alignment) 负责人一职，理由是对公司优先事项存在分歧，这引发了关于 OpenAI [有争议的雇佣惯例](https://x.com/kelseytuoc/status/1791539443016536265?s=46&t=_jodDCDeIUnWb_Td0294bw)的讨论。
   - 在对其与 Scarlett Johansson 声音相似性表示担忧后，OpenAI [暂停了在 ChatGPT 中使用 Sky 语音](https://openai.com/index/how-the-voices-for-chatgpt-were-chosen/)。
   - [Neural Magic](https://discord.com/channels/1189498204333543425/1190208177829068860/1241409136705404968) 正在寻求 CUDA/Triton 工程师为开源工作做出贡献，重点关注激活量化、稀疏化以及针对 MoE 和采样的算子 (kernel) 优化。
   - 讨论了对更好的 AI 安全基准测试的需求，并建议为“高达 2M 的现代 LAMBADA”建立基准，以评估独立处理重叠分块的模型（[来源](https://discord.com/channels/1053877538025386074/1108104624482812015/1241767003480658083)）。


---



# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Deepseek Dilemma**: **Deepseek** 的架构差异导致其目前无法运行，用户达成共识认为 *"它可能无法工作"*。在解决方案出现之前，将其投入运行的尝试已暂停。
  
- **Fine-Tune Frontier**: 分享了 **Meta-Llama 模型** 的改进方案，用户现在可以使用 *"orthogonalized bfloat16 safetensor weights"* 有效地微调 **Llama-3-70B Instruct** 模型。然而，社区仍在探讨在模型微调中使用知名 IP 的影响，并提到了 *Scarlett Johansson 起诉 OpenAI* 等担忧。

- **Colab Conundrums and JAX Jousts**: 关于在 **Colab 或 Kaggle T4** 上运行 6GB 数据集和 5GB Llama3 模型的问题引发了不同反应，主要集中在存储与 VRAM 使用的权衡。同时，尽管最初存在疑虑，但在 **TPU 上使用 JAX**（特别是 Google TPU）被证明是有效的。

- **Multi-GPU Madness and Dependency Despair**: 社区成员高度期待 **Unsloth** 的多 GPU 支持，并意识到这能为工作流带来的优势。环境搭建仍面临挑战，特别是 **WSL 与原生 Windows 安装** 以及将 **Triton** 等依赖项整合到环境中的问题。

- **Showcase Shines with Finetuning Feats**: 微调领域的创新备受关注，包括通过 [LinkedIn 帖子](https://www.linkedin.com/posts/tomaz-bratanic-a58891127_im-very-excited-to-announce-that-ive-finetuned-activity-7197286502895075329-geKp?utm_source=share&utm_medium=member_desktop) 分享的 **Text2Cypher 模型**。一篇关于利用 **LLaMA 3 8b** 进行情感分析的详尽文章发表在 [Medium](https://medium.com/@seandearnaley/elevating-sentiment-analysis-ad02a316df1d) 上，为他人使用 Unsloth 复制微调过程指明了路径。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**New Dataset Invites AI Experiments**: 推出了 **[Tuxemon 数据集](https://huggingface.co/datasets/diffusers/tuxemon)** 作为 Pokemon 数据集的替代方案，提供 `cc-by-sa-3.0` 许可的图像，以获得更大的实验自由度。它为图像提供了两种类型的 Caption，以便在实验中进行多样化的描述。

**Progress in Generative AI Learning Resources**: 社区建议那些寻求 Generative AI 和 LLM 知识的人阅读 "Attention is All You Need" 并访问 **[HuggingFace 学习门户](https://huggingface.co/learn)**。对 GROVE 论文和用于叙事理解的 Conan 基准测试的讨论，表明了社区对推进集体认知的积极兴趣。

**AI Influencers Crafted by Vision and AI**: 一段 [教程视频](https://www.youtube.com/watch?v=qTsdgUyMY94) 受到关注，展示了如何利用 Computer Vision 和 AI 打造虚拟 AI 网红，反映了技术与社交媒体现象交汇点的浓厚兴趣。

**Tokenizer Set to Reduce Llama Model Size**: 一款新开发的 Tokenizer **Tokun** 承诺将 Llama 模型缩小 10 倍，同时提升性能。这种新颖的方法已在 [GitHub](https://github.com/apehex/tokun) 上公开，并在 [Twitter](https://x.com/4pe0x/status/1792638900059385942) 上进行了讨论。

**Clarifying LLMs Configuration for Task-Specific Queries**: AI 工程师专注于配置 **Large Language Models** 以进行 HTML 生成以及维护聊天机器人的对话历史。社区建议采用手动干预（如将之前的消息附加到新 Prompt 中）来解决这些细微的挑战。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**对 Perplexity 的 GPT-4o 性能感到沮丧**：工程师们注意到 **GPT-4o** 倾向于重复回答并忽略 Prompt 的更改，这在对话式 AI 领域是一种退步。有人将其与之前的 LLM 进行对比，认为其表现不佳，并对其交互能力表示失望。

**脚本小子集结，寻求更好的模型切换**：用户正在积极分享和使用自定义脚本，以便在 Perplexity 上实现动态模型切换，特别是使用 [Violentmonkey](https://violentmonkey.github.io/) 等工具，将其作为解决服务限制的补丁。

**API 的怪癖与配额**：关于 Perplexity 的 API Rate Limits（区分请求限制和 Token 限制）及其对工程师工作流的影响存在困惑。同时，出现了关于 API 性能测试的讨论，用户更倾向于 *Omni* 模型，并寻求关于 Threads 功能的澄清，以支持对话上下文。

**寻求升级 API 访问权限**：用户继续施压要求改进 API 访问，表达了对更高 Rate Limits 和更快速支持响应的需求，这反映了对机器学习基础设施日益增长的需求。

**工程师探索聊天之外的 AI**：用户分享的链接显示其兴趣正扩展到 Stability AI 的潜力、体育锻炼带来的精神提升、系外行星 WASP-193b 的细节，以及通过 AI 辅助的《龙与地下城》（Dungeons & Dragons）剧本创作来为儿童生成引人入胜的内容。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**声音被沉默**：OpenAI 已**暂停**在 ChatGPT 中使用 **Sky 声音**，并提供了一份[声明和解释](https://openai.com/index/how-the-voices-for-chatgpt-were-chosen/)以回应用户的疑虑。

**语言模型摆脱束缚**：工程师们报告了在不使用 OpenAI API 的情况下运行 **LangChain** 的成功经验，并介绍了与 **Ollama** 等本地工具的集成。

**GPT-4o 逐步推出但伴随摩擦**：GPT-4 和 GPT-4o 之间的差异显而易见，后者在 Token 上下文窗口方面表现出**局限性**，且使用限制影响了实际应用。GPT-4o 增强的多模态能力得到了认可，官方分享了**[价格页面](https://openai.com/api/pricing/)**以及**[文件上传常见问题解答](https://help.openai.com/en/articles/8555545-file-uploads-faq)**，以进一步明确使用细节。

**Prompt 编写的挑战与创新**：在工程领域，针对自我意识和技术集成的 Prompt 优化存在诸多**挑战**，但同时也在分享**创新的 Prompt 策略**，以提升创意和结构化生成的能力。**JSON mode** 被建议作为提高指令精确度的可行工具；OpenAI 的文档仍是首选参考资料。

**API 的痛苦与收获**：API 用户报告了 `chat.completion.create` 的**不一致性**，包括响应不完整的问题，并表现出对使用 JSON mode 来控制格式和内容的偏好。尽管存在小插曲，但关于协调创意的讨论非常热烈，有人提出了 **"Orchestrating Innovation on the Fringes of Chaos"**（在混沌边缘编排创新）作为一种探索性方法。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 的杀毒软件误报**：LM Studio 用户注意到 **llama.cpp 二进制文件**因未签名而被 **Comodo 杀毒软件**拦截。建议用户在遇到安全警告时将其视为潜在问题。

- **模型加载与硬件讨论**：关于各种 GPU 的讨论中，一位用户发现 **Tesla P100 的表现低于预期**。其他讨论指出 Alder Lake CPU 的 E-cores 会影响 GPT 的 **Quantization（量化）性能**。在 RAM 方面，**更高的频率**与**更好的 LLM 性能**直接挂钩。

- **GGUF 登上舞台**：用户讨论了将 Hugging Face 的模型集成到 LM Studio 中，建议使用 **GGUF** 格式文件以确保兼容性。社区对最近推出的用于导入模型的 "HF -> LM Studio deeplink" 功能给出了积极反馈。

- **LLM 的创意用例交汇**：从 **OpenBioLLM** 等医疗 LLM 推荐，到生成 SVG 和 ASCII 艺术的基准测试，用户正在积极探索多样化的应用。其中 **MPT-7b-WizardLM** 模型因其在生成无审核故事方面的潜力而受到关注。

- **LM Studio autogen 的缺陷与修复**：讨论了 LM Studio 的 autogen 功能中导致简短响应的 Bug，修复方法是将 **max_tokens** 设置为 **-1**。用户还指出 LM Studio 的本地服务器与 OpenAI 规范之间存在差异，影响了 **AutoGPT** 等应用的 tool_calls 处理。

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**为 LoRAs 编写完美提示词**：工程师们分享了一种在 Stable Diffusion 中利用多个 LoRAs 的提示词结构，但观察到超过三层后收益递减或出现问题，这暗示了潜在的优化空间。

**初次使用 Stable Diffusion 的困扰**：一个 'NoneType' 对象属性错误导致一位新用户在首次运行 Stable Diffusion 时遇到障碍，引发了对故障排除专业知识的呼求，目前尚无明确解决方案。

**SD3 的到来引发期待与质疑**：对于 SD3 的发布，情绪呈现两极分化，既有怀疑也有乐观。Emad Mostaque 的推文证实工作正在进行中。

**Topaz 之争**：关于 Topaz 作为视频上采样解决方案的有效性引发了辩论。工程师们承认其强大之处，但将其与 ComfyUI 的吸引力进行了对比，强调了成本和功能等考量因素。

**应对 SDXL 的重量级需求**：一位用户强调了在处理 SDXL 模型对高分辨率的需求时，充足的 VRAM 至关重要。同时明确了 SDXL 和 SD1.5 需要不同的 ControlNet 模型。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 在 Windows 上仍处于开发中 (WIP)**：尽管关注度很高，但 Mojo 尚未原生支持 Windows，目前需要 WSL；用户在 CMD 和 PowerShell 中遇到了问题，但 [Windows 支持已指日可待](https://docs.modular.com/mojo/manual/get-started/)。

**Bend vs. Mojo：性能视角**：讨论重点关注了 Chris Lattner 对 Bend 性能的见解，指出虽然它在单核上落后于 CPython，但 Mojo 是专为高性能场景设计的。这两种语言的社区都在期待增强功能和即将举行的社区会议。

**Llama 的 Python 亲戚**：社区注意到一个**从零开始实现 Llama3** 的项目，可在 [GitHub](https://github.com/naklecha/llama3-from-scratch) 上找到，被描述为“一次构建一个矩阵乘法”，这是对语言内部细节的一次迷人探索。

**深入探索 Mojo 内部机制**：各项讨论包括将 `nightly` 设为默认分支以避免 DCO 失败的见解、Mojo 中潜在的列表容量优化、SIMD 优化辩论、关于类似 Rust 的 `Vec::shrink_to_fit()` 新列表方法的建议，以及解决导致段错误 (segfaults) 的别名问题。提到的关键点包括社区对列表初始化的贡献，这可能会带来性能提升，以及[对性能产生积极影响](https://github.com/modularml/mojo/issues/2556)的补丁。

**工程师的思维内部**：讨论了 PR DCO 检查失败的技术解决办法并提供了流程见解；不稳定的测试引发了关于修复和 CI 痛点的讨论；自定义数组类型中的段错误促使了同行调试环节。社区对分享有助于解开优化之谜的复杂细节表示赞赏。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **LLM 工作坊和 Fine-tuning 讨论升温**：参与者正准备参加即将举行的工作坊，包括 Jeremy Howard 的 *Build Applications in Python* 以及关于 RAG 模型优化的会议。提出了关于 Finetuning 的实际问题，例如使用 LlamaParse 和 GPT-4o 等工具的 PDF 解析技术，以及如何使用 FastAPI 和 Streamlit 等框架部署 Fine-tuned LLMs。

- **技术大牛排障并协作应对挑战**：来自不同地区的亚洲爱好者正在建立联系并解决诸如 Modal 命令错误等挑战，讨论在车辆故障预测中进行 Fine-tuning 的潜力，并就预训练 LLMs 的 LoRa 配置进行头脑风暴。

- **跨平台积分记录持续更新**：参与者正在处理 JarvisLabs 等服务的积分获取和确认流程，组织者在幕后协调以确保积分分配到账户，有时会因邮箱不匹配而面临注册障碍。

- **学习资源汇集**：重点介绍了从 Hamel 的博客到关于 GPT-4V 开源替代方案的 CVPR 2024 论文等知识库。大家在讨论是否将这些精华内容存放在公共 GitHub 仓库中，并寻找更有效地组织学习材料的方法。

- **新加坡用户活跃**：亚洲时区频道中来自新加坡的人数出奇地高，引发了对该国代表性显著的评论。随着新成员的加入，现场气氛热烈，大家在处理积分安排的同时利用各种学习机会。

渴望学习的人和新兴专家都投身于 Fine-tuning、数据提取、应用及 LLMs 其他方面的变革浪潮中，这预示着一个充满智力协同和对实际 AI 工程能力不懈追求的时期。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Benchmarks 和 AGI 讨论激发工程师好奇心**：工程师们考虑需要改进 Benchmarks，呼吁建立“支持高达 2M 的现代 LAMBADA”来评估独立处理重叠块的模型，并讨论了一篇关于 AGI 进展和必要策略的论文，题目为《[人工智能的演变](https://arxiv.org/abs/2405.10313)》。

- **Sam Altman 模仿推文引发笑声、VC 怀疑和 AI 经济谜题**：一条挑衅性的 Sam Altman 模仿推文引发了关于 VC 在 AI 中的角色、AI 对公司裁员的实际财务影响的讨论，以及一名成员关于参加 Runpod 黑客松的咨询。

- **Hermes 2 Mixtral：面向行动的 LLMs 的开端**：Nous Hermes 2 Mixtral 因其在 CrewAI Agent 框架内触发行动的独特能力而受到赞誉，讨论还涉及多语言能力、多轮对话数据的重要性以及高昂的训练成本。

- **模型利用策略**：工程师们比较了对 Llama3 和 GPT-4 等模型进行 Finetuning 与高级 Prompting 的效果，同时他们也在寻找 Fine-tuned 重排序器的 Benchmarks，并强调了本地模型在处理具有敏感性和可预测性需求任务时的优势。

- **WorldSim 随着终端 UX 翻新进入 GPT-4o 时代**：WorldSim 进行了终端 UX 翻新，并即将集成 GPT-4o，同时社区参与了复杂自适应系统、符号知识图谱的讨论，并探索了 WorldSim 在生成 AI 相关知识图谱方面的潜力。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Hugging Face 向 AI 社区投入 1000 万美元**：Hugging Face 承诺投入 1000 万美元，为初创公司和学术界提供免费的共享 GPU 资源，作为其推动 AI 开发民主化努力的一部分。其 CEO Clement Delangue 在一轮大规模融资后宣布了这一消息，详情见 [The Verge](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai) 的报道。

**编程领域的新选手 Bend**：一种名为 Bend 的新型高级编程语言登场，引发了关于其相对于 Triton 和 Mojo 等现有 GPU 语言优势的讨论。尽管 Mojo 在 GPU 上存在局限性，且 Triton 专注于机器学习，但 Bend 的优势在 [GitHub](https://github.com/HigherOrderCO/Bend) 上得到了阐述。

**优化机器学习推理**：专家们就构建高效推理服务器交换了建议，推荐使用 Nvidia Triton 和 TorchServe 进行模型服务。重点贡献包括在使用 **torch.compile()** 处理静态形状（static shapes）时应用优化，并参考 GitHub 上的代码改进以更好地支持 NHWC 格式的组归一化（group normalization），详见此 [pull request](https://github.com/pytorch/pytorch/pull/126635/files#r1605935532)。

**CUDA 的复杂性——加法与内存**：围绕 `cuda::complex` 的原子操作（atomic operations）以及 128 位 `atomicCAS` 的阈值限制展开了激烈的辩论。社区分享了处理复数的代码变通方法和公认的方法论，并讨论了 Torch 中原地乘法（in-place multiplication）期间潜在的内存开销。

**扩展与优化 CUDA 挑战**：社区剖析了梯度裁剪（gradient clipping）、内存优化模板化以及 ZeRO-2 实现中的问题。他们分享了多个 GitHub 讨论和 pull requests（[#427](https://github.com/karpathy/llm.c/pull/427), [#429](https://github.com/karpathy/llm.c/pull/429), [#435](https://github.com/karpathy/llm.c/pull/435)），表明了对性能和微调 CUDA 应用的高度关注。

**解决 ParPaRaw 解析器性能问题**：有关 **libcudf** 与 CPU 并行操作基准测试的询问不断出现，暗示了社区对高效解析的热情，并注意到 GPU 相对于 CPU 的性能提升。关注点还集中在 Dask-cuDF 合并入 cuDF 以及前者随后的归档，详见 [GitHub](https://github.com/rapidsai/dask-cudf)。

**深入了解 GPU 查询引擎**：一位来自 Voltron 的 **cuDF** 资深人士即将发表演讲，分享关于构建 GPU 原生查询引擎的见解，阐明从内核设计到生产部署的策略。可通过此 [Zoom 会议](https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09#success) 获取参与详情。

**CUDA 架构师深入探讨 GPU 基础知识**：分享了一个指向 CUDA 架构师 Stephen Jones 在 YouTube 上演讲的链接，该演讲清晰地阐述了 GPU 编程和高效内存使用策略，这对于现代 AI 工程任务至关重要。通过[此处](https://www.youtube.com/watch?v=QQceTDjA4f4)的链接深入了解 GPU 的工作原理。

**Neural Magic 为 CUDA/Triton 创新招募人才**：Neural Magic 正在寻找热衷于 CUDA/Triton 项目的工程师，重点关注激活量化（activation quantization）。他们特别感兴趣于利用 2:4 稀疏性等下一代 GPU 特性，并进一步优化 MoE 和采样中的内核。

**解析 PyTorch 与 CUDA 的交互**：针对 PyTorch 与 CUDA 的高效数据类型打包/解包进行了详细的头脑风暴，重点关注 `uint2`、`uint4` 和 `uint8` 类型。项目管理和协作编程在讨论中占据重要地位，并提及了用于自定义 CUDA 扩展管理的 GitHub Premier [#135](https://github.com/pytorch/ao/pull/135) 。

**屏障同步简化**：一位社区成员通过将屏障同步（barrier synchronization）比作确保所有学生在参观完博物馆后都回到巴士上，帮助他人理解这一概念。这是一个贴切的比喻，支撑了 GPU 操作中的同步过程。

**民主化 Bitnet 协议**：大家正共同努力举办 Bitnet 小组会议并审查重要的技术文档，量化讨论集中在将 `uint4` 转换为 `uint8` 类型。正如协作驱动中所提到的，共享资源正在指导这些会议。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **CC 数据集中的垃圾信息警报**：Eleuther 社区在 Common Crawl (CC) 数据集中发现了大量垃圾信息，其中中文文本受影响尤为严重。《Technology Review》一篇关于 GPT-4o 标记污染的[文章](https://www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/)强调了类似的担忧，指出了非英语数据清洗中的问题。
  
- **OpenELM 专注于效率**：一种名为 OpenELM 的新 LLM 因其可复现性以及比 OLMo 提高 2.36% 的准确率而引起了成员们的兴趣。欲了解详情，请查看 [OpenELM 研究页面](https://machinelearning.apple.com/research/openelm)。

- **AI 内存效率备受关注**：模型训练中计算 FLOPs 的挑战引起了关注，EleutherAI 的 cookbook 提供了准确估算的[指南](https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py)，这是优化内存和计算资源使用的关键环节。

- **跨模态学习成为焦点**：研究人员正在探索像 **ImageBind** 和 **PaLM-E** 这样的模型在多模态数据训练后是否对单模态任务有益。Zero-shot 识别和模态特定嵌入的集成可以增强检索性能，[ImageBind](https://arxiv.org/abs/2305.05665) 和 [PaLM-E](https://arxiv.org/abs/2303.03378) 等论文是这一对话的核心。

- **模型微调的优势与怪癖**：成员们注意到了 HF 模型中的自动提示设置，并讨论了微调技术，包括非 pipeline 情况下的 soft prompt tuning。然而，也出现了一些问题，例如在调用 `model.to_sequential()` 后 `param.requires_grad` 会重置，这可能会阻碍开发进程。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Meta、DeepMind 和 Anthropic 的模型混战**：Meta 的 **Chameleon** 模型拥有 **34B parameters**，在人工评估中表现优于 **Flamingo** 和 **IDEFICS**。DeepMind 的 **Flash-8B** 提供了多模态能力和效率，而其 Gemini 1.5 模型在基准测试中表现出色。与此同时，Anthropic 的计算量比其上一个模型扩大了四倍，LMsys 的 “Hard Prompts” 类别为 AI 评估带来了新挑战。

**AI 安全团队解散引发动荡**：OpenAI 的 **superalignment team**（包括 Ilya Sutskever 和 Jan Leike）在对 OpenAI 政策的分歧和批评中解散。OpenAI 的离职协议因极具争议的终身不贬低条款（nondisparagement clauses）而引起了特别的愤怒。

**播客思考与游戏荣耀**：**Retort AI podcast** 分析了 OpenAI 的举动，引发了关于词汇表大小 Scaling Laws 的辩论，并带点幽默地引用了控制理论中的滞后现象（hysteresis）。成员们怀旧地分享了《使命召唤》(Call of Duty) 的游戏渊源以及在 YouTube 上创作学术内容的抱负。

**对 ORPO 保持谨慎**：对 ORPO 方法的可扩展性和有效性的怀疑有所增加，社区成员分享的测试结果表明可能存在过度正则化（over-regularization）的风险。随着该方法被添加到 **Hugging Face** 库中，对其的担忧进一步放大。

**挑战 Chinatalk 并从 Llama3 中学习**：对 Chinatalk 剧集的赞赏、**llama3-from-scratch** 作为学习资源的价值，以及一篇解释 Latent Consistency Models 的巧妙 Notion 博客，为自我提升提供了信息化建议。然而，关于 Books4 数据集法律风险的警告也为对话增添了紧张气氛。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Scaling Up 是秘诀**：Geoffrey Hinton 认可了 Ilya Sutskever 关于 Scaling（规模化）是 AI 成功的关键这一信念，并表示：*“[Ilya] 总是宣扬只要把它做大，它就会运行得更好。事实证明 Ilya 基本上是对的。”* 讨论中重点提到了一段 [完整采访](https://x.com/joelhellermark/status/1791398092400390195)，Hinton 在其中分享了这一见解。

- **垂直轴风力发电的新风向**：EPFL 的研究人员利用遗传算法优化了垂直轴风力涡轮机，旨在超越水平轴版本的局限性。这项工作有望带来更安静、更环保的涡轮机，详情见 [完整文章](https://actu.epfl.ch/news/machine-learning-enables-viability-of-vertical-axi)。

- **AI Agent，包含自由意志吗？**：讨论围绕 AI Agent 的自主性展开，包括 Andrew Ng 对 [AI Agent 的看法](https://x.com/AndrewYNg/status/1770897666702233815) 以及 Gordon Brander 在分享的 [YouTube 视频](https://www.youtube.com/watch?v=BNFRGfWQo6M) 中关于自适应 AI 的断言。

- **首席对齐师离职**：在 Jan Leike 辞去 OpenAI 对齐负责人职务后，社区思考了其影响，同时 Sam Altman 和 Greg Brockman 也分享了他们的看法，详见 [此处](https://x.com/gdb/status/1791869138132218351?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)。

- **AI 编程语言之争**：随着 Hugging Face 在 Candle 和 tokenizers 等项目中采用 Rust，以及 Go 在基于 HTTP 请求的 AI 应用中保持其地位，关于哪种语言在 AI 开发中占据主导地位的争论依然激烈。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **内存对自主 Agent 至关重要**：一场关于 **memary** 项目的网络研讨会定于周四上午 9 点（太平洋时间）举行，重点讨论自主 Agent 的长期记忆。对记忆挑战和未来方向感兴趣的 AI 工程师可以 [报名参加活动](https://lu.ma/nzh3o83f)。

- **表格削弱了 QA 效果**：LLM 仍被复杂的表格（如 Caltrain 时刻表）难住，由于解析效果差导致幻觉问题，更多详情见 [此分析](https://t.co/Scvp7LH2pL)。

- **大幅提升向量搜索速度**：[JinaAI_](https://t.co/NnHhGudMa8) 分享了使用 32-bit 向量将向量搜索速度提高 32 倍的方法，仅牺牲了 4% 的准确率——这是生产环境应用的关键优化。

- **旧金山 AI 思想集会**：LlamaIndex 计划在总部举行一场线下的旧金山见面会，重点讨论高级 RAG 引擎技术，预约链接见 [此处](https://t.co/o0BWxeq3TJ)。

- **数据治理的 Metadata 诀窍**：工程师们阐述了 MetaDataFilters 在 LlamaIndex 数据库层级进行数据治理的效用，并提出了对敏感财务数据进行选择性索引的想法。

- **与 GPT-4o 集成**：一次显著的讨论涉及 GPT-4o 与 LlamaParse 的集成，关于该主题的 [Medium 文章](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a) 获得了社区成员的认可和好评。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **有争议的许可限制隐现**：CommonCanvas 数据集提供了 7000 万图像-文本对，但因其限制性的非商业许可和禁止衍生品的规定引发了争论，这令看到有益修改潜力的成员感到沮丧（[CommonCanvas 公告](https://x.com/multimodalart/status/1791201296357142663)）。

- **技术对话——PyTorch 困扰工程师**：关于 PyTorch 的 `native_group_norm` 在不使用 `torch.compile` 时导致速度变慢的讨论非常激烈；一位成员指出 eager mode 与编译方法的性能几乎持平。

- **数据集完整性受审视**：AI 工程师们担心训练视觉语言模型 (VLLMs) 和文本生成图像模型 (T2I) 时幻觉标注的影响，同时也表达了创建高质量开源数据集以避免此类问题的意愿。

- **多模态领域的新秀**：Chameleon 模型因其出色的图像和文本理解及生成能力而受到认可，在图像标注和生成任务中表现出优于 Llama-2 等大型模型的潜力（[Chameleon arXiv 论文](https://arxiv.org/abs/2405.09818)）。

- **CogVLM2 有争议的条款**：成员们被提醒注意 CogVLM2 模型的许可协议，其中包括可能限制用于损害中国利益的条款，并规定由中国管辖争议（[CogVLM2 许可证](https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE)）。

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**4Wall Beta 发布**：[4Wall](https://beta.4wall.ai) 这一 AI 驱动的娱乐平台已进入 Beta 测试阶段，提供无缝的 [AI Town 集成](https://www.aireality.tv/)以及用于创建地图和游戏的用户生成内容（UGC）工具。正如其[公告](https://x.com/4wallai_/status/1792359640170410339?s=46&t=W_c0j4FPVSWZuhD7zTaSYA)所示，他们还在开发 3D AI 角色。

**Game Jam 冠军**：**Rosebud / #WeekOfAI 教育 Game Jam** 公布了获胜者，包括 **"Pathfinder: Terra’s Fate"** 和 **"Ferment!"**，突显了 AI 在教育游戏领域的潜力。游戏可以在[此处](https://play.rosebud.ai/)访问，更多详情见[公告推文](https://x.com/Rosebud_AI/status/1791616913279160327)。

**AI Town 的 Windows 里程碑**：**AI Town** 已实现对 Windows 的原生兼容，并在 [Tweet](https://fxtwitter.com/cocktailpeanut/status/1791495360541593964) 中进行了庆祝，这引发了关于创新实现的讨论，包括使用 [GitHub - Townplayer](https://github.com/cocktailpeanut/townplayer) 等工具进行对话转储的方法。此外，用户正在探索 AI Town 中结合深度世界背景集成的创意场景。

**AI Reality TV 发布**：互动式 **AI Reality TV 平台**的发布引起了社区关注，邀请用户模拟与 AI 角色的社交互动，正如[此公告](https://x.com/edgarhnd/status/1791586276178587707)所述。

**故障排除与技术技巧**：AI 工程师们交流了 AI Town 设置问题的解决方案，包括解决 Agent 通信问题和从 SQLite 数据库提取数据的建议。建议包括查阅内存系统文档以及调整 AI Town 内部设置。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **服务器拒绝 Function Calls**：工程师们在使用 **OpenRouter** 时遇到了障碍，服务器返回 500 状态码并提示错误消息 "Function calling is not supported by openrouter"，该问题在讨论中尚未解决。
- **404 错误处理失误**：用户发现了一个缺陷，即无效的模型 URL 会导致应用程序错误并显示消息，而不是显示 404 页面，这表明基于登录状态的用户体验不一致。
- **支付故障**：社区讨论了**自动充值支付被拒**的问题，导致用户无法手动充值，怀疑是由于用户银行（特别是 WISE EUROPE SA/NV）的拦截造成的。
- **模型推荐**：用户交流了模型推荐，重点介绍了 **“Cat-LLaMA-3-70B”** 和 **Midnight-Miqu 模型**，并呼吁采用更好的微调策略，而不是使用“随机的未清洗数据”。
- **不稳定的 Wizard LM 服务**：用户在 OpenRouter 上使用 **Wizard LM 8x22B** 时遇到了间歇性的请求失败，这归因于多个供应商普遍出现的临时请求超时 (408) 激增。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Galore 工具缺乏 DDP 支持**：工程师们强调了 **Galore Layerwise** 工具无法支持 **Distributed Data Parallel (DDP)**，指出了其在扩展使用方面的重大局限性。

- **大规模中文数据集的训练困境**：讨论集中在 *使用 10 亿中文 tokens 微调 8B 模型*，并关注了 [Multimodal Art Projection (M-A-P)](https://huggingface.co/m-a-p) 和 [BAAI 数据集](https://huggingface.co/BAAI)，暗示了多语言模型训练的趋势。

- **Llama 的梯度增长问题**：**Llama 3 8B 模型** 观察到一个技术挑战，即低秩微调会导致 *梯度范数 (gradient norm) 无限制增加*，这表明可能存在权重饱和和梯度更新的问题。

- **GPT-4o 的 Token 麻烦**：最近关于 [GPT-4o](https://www-technologyreview-com.cdn.ampproject.org/c/s/www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/amp/) 的反馈发现，其 token 数据包含垃圾信息和色情短语，引发了对其语言处理质量和清洁度的担忧，尤其是在中文方面。

- **Commandr 配置进展**：社区持续提供支持和贡献，例如一个 [特定的 GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files)，旨在增强 **axolotl** 的 Commandr 设置，表明项目正在积极迭代和解决问题。

- **Axolotl 配置困惑**：工程师们分享了特定的用例问题：一个涉及在持续预训练期间由于词表外填充 tokens 导致的 *非法内存访问错误*；另一个详细描述了在微调 **Mistral 7b** 时的问题，尽管 loss 下降，但模型的学习效果并不理想。

- **Axolotl-Phorm Bot 见解**：来自 *axolotl-phorm bot* 频道的关键要点包括：探索用于数据结构化的 **ORPO 格式**；关于使用 weight decay 和 LoRA Dropout 来避免 LLM 训练中过拟合的阐述；通过 **Hugging Face Accelerator 库** 进行梯度累积 (gradient accumulation) 的好处；以及关于在 Axolotl 的损失函数中实现样本权重而无需额外自定义的讨论。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**内存对模型魔力的重要性**：讨论了在代理后端使用 cross-encoders 进行重排序 (Re-ranking)，重点关注 **OpenAI GPTs 和 Gemini 模型**。人们对 **短期记忆解决方案** 感兴趣，例如用于聊天机器人维持对话上下文的缓冲区。

**LangChain 获得指引**：关于 **在 LangChain 中引导模型响应** 的查询促成了 `PromptTemplate` 解决方案的分享，并引用了 [关于该主题的 GitHub issue](https://github.com/langchain-ai/langchain/issues/18820)。同时，**面向 Swift 开发者的 LangChain** 已发布，提供了在 iOS 和 macOS 平台上工作的资源，详见 [LangChain Swift 的 GitHub 仓库](https://github.com/buhe/langchain-swift)。

**SQL 是关键**：**LangChain 与 SQL 数据** 的应用为跨数据集总结概念打开了大门。对话转向将 SQL 数据库集成为内存解决方案的方法，相关指南可见于 [LangChain 文档](https://python.langchain.com/v0.1/docs/integrations/memory/)。

**Langmem 的长期记忆精通**：**Langmem 的上下文管理能力** 受到赞赏。YouTube 演示展示了 Langmem 如何在对话中有效地切换上下文并保持长期记忆，突显了其在复杂对话任务中的实用性 ([Langmem 演示](https://youtu.be/7QU89qL795A))。

**可疑链接充斥 Feed**：多个频道报告了 **可疑的 50 美元 Steam 礼品链接** ([可疑链接](https://bitly.cx/OjEZl)) 的传播，警告成员谨慎行事，并暗示该链接很可能是欺诈性的。

**AI 的魔方**：**Rubik's AI** 承诺提供增强的研究协助，使用 **促销代码 RUBIX** 可获得两个月的免费高级功能访问权限。

**玩转 RAG-Fusion**：有一个关于 **RAG-Fusion** 的教程，强调了其在 **用于文档处理的 AI 聊天机器人** 中的应用，并强调了其相对于 RAG 单查询限制的多查询能力。该教程为工程师提供了使用 LangChain 和 GPT-4o 的见解，可在 [LangChain + RAG Fusion + GPT-4o 项目](https://youtu.be/P_xZ1HJwKl8?si=cQZ1CTydmFRjvveP) 查看。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Discord 支持系统请求改版**：一名成员呼吁改进 Discord 支持系统，理由是许多咨询未得到处理。有人指出，目前的系统运作方式更像是一个**社区支持平台**，而非由官方人员维护。

- **速率限制影响 Trial API 用户**：用户在使用 `RAG retriever` 时遇到 **403 错误**，这归因于触发了 **Trial API** 的速率限制，该 API 并非为生产环境设计。

- **用户咨询免费 API 密钥**：讨论涉及 Cohere **免费 API 密钥**的可用性和范围，并澄清这些密钥旨在用于初步原型设计，且带有特定的使用限制。

- **翻译服务求助**：有用户寻求使用 `CommandR+` 进行翻译服务的帮助，并得到了指向 [Chat API 文档](https://docs.cohere.com/docs/chat-api) 的指引，该文档提供了实现指南。

- **展示 Cohere AI 的实际应用**：分享了一份名为“Cohere AI 完整指南”的新资源，并在 [Analytics Vidhya 平台](https://www.analyticsvidhya.com/blog/2024/05/guide-to-using-cohere-ai/#) 上提供了完整的安装和使用说明。配套的演示应用可以在 [Streamlit](https://cohere-guide-blog.streamlit.app) 进行测试。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Hugging Face GPU 盛宴**：Hugging Face 正向小型开发者、学术界和初创公司捐赠价值 **1000 万美元的免费共享 GPU 资源**，利用其财务地位和近期投资，正如 [The Verge 文章](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai) 中所述。

**OpenInterpreter 攻克 Pi 5 和 DevOps**：OpenInterpreter 已成功部署在**运行 Ubuntu 的 Pi 5** 上，并讨论了涉及项目集成的合作，包括潜在的 Azure 额度支持。此外，一名初级全栈 DevOps 工程师正在寻求社区帮助，以开发一个 “lite 01” AI 助手模块。

**技术技巧与窍门**：分享了在不同平台上解决 OpenInterpreter 环境设置问题的方案，讨论重点集中在 WSL、虚拟环境和 IDE 的使用。通过一个用于 Flutter 集成的 [GitHub 仓库](https://github.com/Tonylib/o1_for_flutter) 提供了进一步的帮助，并收到了关于名为 O1 Lite 设备开发帮助的请求。

**语音 AI 的机器人腔调**：社区讨论批评语音 AI 与 GPT-4 的文本能力相比缺乏自然感，同时 [YouTube 视频](https://www.youtube.com/shorts/zgUanlLV_OQ) 中强调了语音助手具备中断能力的构想。

**活动与社区参与**：发布了邀请社区参加首届无障碍圆桌会议（Accessibility Round Table）和专注于本地开发的直播通知，旨在促进直播环境下的参与和知识分享。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **RAG 调试风波**：一个嵌入模型的问题导致在 RAG 教程期间出现段错误（segfault），错误信息为 *"llama_get_logits_ith: invalid logits id 420, reason: no logits"*。经确认，该问题是由于使用了“仅嵌入（embeddings-only）”模型引起的，此类模型不具备生成任务的能力，这一细节在 [Mozilla 教程](https://future.mozilla.org/news/llamafiles-for-embeddings-in-local-rag-applications/) 中可能被忽略了。

- **云服务选择**：支持 GPU 的云服务成为热门话题，工程团队推荐了像 [vast.ai](https://vast.ai) 这样的供应商，用于实验和处理临时计算负载。

- **SQLite 遇见向量**：Alex Garcia 带着他的 [sqlite-vec](https://github.com/asg017/sqlite-vec) 项目参与了讨论，这是一个支持向量搜索的 SQLite 扩展，引发了将其与 Llamafile 集成以增强内存和语义搜索能力的兴趣。

- **Llamafiles 澄清**：一个关键的澄清：其教程中链接的 [Mozilla Llamafile 嵌入模型](https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile/tree/main) 不具备生成能力，这一点需要重点说明以明确用户的预期。

- **模型部署创新**：关于在各种平台上战略性部署 Llamafile 模型的讨论正热，这表明云供应商提供的 GPU 算力是实际实验的关注焦点。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

**微调热潮兴起**：工程师们对 [LLM 微调课程](https://maven.com/parlance-labs/fine-tuning) 表达了复杂的看法。一些人认为其在 LLM 训练、评估和 Prompt Engineering 方面的动手实践方法很有价值，而另一些人则持怀疑态度，对营销手段下的内容质量表示担忧。

**课程内容参差不齐**：课程参与者反馈体验各异，一些人认为入门材料较为基础，但这取决于个人的背景；这说明了针对不同专业水平调整内容难度的挑战。

**区间预测**：[MAPIE 文档](https://mapie.readthedocs.io/en/latest/) 成为那些希望实现预测区间的人员的关键资源。此外，还讨论了关于 Conformal Prediction 的见解，并提到了适用于时间序列数据的 Nixtla。

**从 Inpainting 演进的 Embeddings**：与掩码语言建模（Masked Language Modeling）类似，通过 Inpainting 技术推导图像 Embeddings 成为关注的话题，强调了一种从可见数据估计未见图像特征的方法。

**多语言实体进入评估阶段**：讨论了跨语言比较实体的策略（例如 "University of California" 和 "Universidad de California"），可能结合了对比学习（Contrastive Learning）和特定语言的前缀，并提到了 [arxiv 论文](https://arxiv.org/pdf/2401.12178) 以供进一步阅读。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **寻求 YOLO 在 Comma 上的运行速度**：围绕在 Comma 设备上运行 YOLO 模型的各种可行性和性能指标展开了讨论，目前的报告显示 **预测时间约为 1000ms**。

- **多项式精度的权衡**：一位工程师报告称使用 11 阶多项式进行正弦近似，产生了 *1e-8* 的误差，同时在评估使用更高阶多项式以达到 *1e-12* 误差的可能性，尽管存在计算效率方面的担忧。

- **对数和指数近似的担忧**：讨论重点包括在对数和指数函数的多项式近似中保持准确性的困难，建议使用范围缩减（Range Reduction）技术，这可能有助于平衡精度与复杂度。

- **思考 tinygrad 中的位移操作**：关于 tinygrad 内部位移效率的询问，具体在于是否有比 `x.e(BinaryOps.DIV, 2 ** 16).e(BinaryOps.MUL, 2 ** 16)` 更精简的方法。

- **揭秘 Metal 编译器**：
   - 一位参与者对 Metal 编译器展开 for 循环的决策表示好奇，指出在调用 `Tensor.arange(1, 32)` 与 `Tensor.arange(1, 33)` 时生成的代码存在差异。
   - 提出了一个谜题：为什么 32 这个数字会特别影响 Metal 编译器的编译行为，并强调了这一神秘阈值对性能产生的影响。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Squeak 与 Claude 的碰撞**：出现了一场关于将 **Claude3** 与 **Squeak Smalltalk** 集成的讨论，显示出将尖端 AI 与经典编程环境结合的兴趣。具体的实际应用细节仍有待商榷。
  
- **语音模式焕然一新**：在 GPT-4o 中，由于担心与 Scarlett Johansson 的声音相似，名为 **Sky** 的语音被替换为 **Juniper**。GPT-4o 从多模型转向单一模型的方法旨在降低延迟并增强情感表达，尽管这增加了复杂性（[语音聊天常见问题解答](https://help.openai.com/en/articles/8400625-voice-chat-faq)）。

- **AI 的双刃剑**：随着 GPT-4o 等模型的演进，它们面临着诸如 **Prompt Injection** 风险和不可预测行为等挑战，这些问题可能与遗留系统遇到意外命令时一样棘手。

- **永无止境的改进循环**：呼应了 Stainslaw Lem 的《颠倒的进化》（The Upside-Down Evolution），讨论了 AI 和其他复杂系统的韧性，并达成共识：虽然完美的可靠性是一种神话，但培养容错设计（Fault-tolerant designs）至关重要——即使这会导致新的预料之外的问题。



---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**法律专家关注 GPT-4o**：AI 工程师指出，与 **GPT-4** 和 **GPT-4-Turbo** 等前代模型相比，**GPT-4o** 在复杂的法律推理方面表现出显著进步。相关改进和方法论已在 [Evan Harris 的 LinkedIn 文章](https://www.linkedin.com/posts/evan-harris-387375b2_the-release-of-gpt-4o-from-openai-has-been-activity-7196856963454959617-w1i1)中分享。

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

- **征集 Docker 开发者进行 AI 协作**：目前正在为一篇关于**使用 Docker 训练和部署 AI 模型**的即将发布的文章征集贡献者。发起者正在寻求在撰写、贡献或审阅文章方面的帮助，并邀请感兴趣的工程师通过私信进行协作。

---

**DiscoResearch Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1240921967327187047)** (718 条消息 🔥🔥🔥): 

- **Deepseek 尚无法运行**：用户讨论了由于架构不同，Deepseek 目前无法正常工作。一位用户指出 *"它可能无法运行"*，另一位用户确认 *"Deepseek 暂时还不行"*。
- **在 Colab/Kaggle 上处理大型数据集**：有用户询问 6GB 的数据集是否能与 5GB 的 Llama3 模型一起放入 Colab 或 Kaggle T4 中。意见虽有分歧，但有人指出 *"datasets (hf 库) 不会将数据集加载到 RAM 中"*；因此，这更多是存储问题，而非 VRAM 限制。
- **尽管存在质疑，JAX TPU 训练效果良好**：关于在 TPU 上使用 JAX 展开了激烈辩论，一位用户断言它在 Google TPU 上训练良好。*"即使使用 torch 也可以在 TPU 上训练，但 JAX 基本上是生产环境中的主要选择"*，这是一项关键见解。
- **讨论有效的微调技巧**：值得注意的是，kearm 讨论了一种改进的方法，通过使用 *"正交化 bfloat16 safetensor 权重"* 来 *"移除 Meta-Llama 模型中的护栏"*，并建议 *Llama-3-70B Instruct* 现在可以进行高效且低成本的微调。
- **法律问题与 AI 微调**：用户思考了使用知名 IP 进行模型微调的风险，同时其他人提到了正在进行的诉讼，例如 *Scarlett Johansson 起诉 OpenAI*。*"她可能会赢"*，这种情绪在法律纠纷的讨论中得到了共鸣。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://colab.research.google.com/drive/1a-aQvKC9avdZpdyBn4jgRQFObTPy1JZw?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/NexaAIDev/Octopus-v4/blob/main/config.json">config.json · NexaAIDev/Octopus-v4 at main</a>: 未找到描述</li><li><a href="https://github.com/unslot">UNSLOT - 概览</a>: 正在输入... GitHub 是 UNSLOT 构建软件的地方。</li><li><a href="https://tenor.com/view/confused-confused-look-confused-face-huh-what-gif-2480734549943489640">困惑的表情 GIF - Confused Confused look Confused face - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/HCOQmKTFzYY?si=Ktlemk1OFhMfj8gK">令人惊叹的 GPU 新编程语言刚刚发布...</a>: 什么是用于并行计算的 Bend 编程语言？让我们初步了解 Bend 及其如何使用类 Python 语法来编写高性能 ...</li><li><a href="https://cloud.google.com/tpu/docs/run-calculation-pytorch">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/failspy/llama-3-70B-Instruct-abliterated">failspy/llama-3-70B-Instruct-abliterated · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a">Blackhole - lamhieu 收藏集</a>: 未找到描述</li><li><a href="https://huggingface.co/failspy/Meta-Llama-3-8B-Instruct-abliterated-v3">failspy/Meta-Llama-3-8B-Instruct-abliterated-v3 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/accelerate/en/usage_guides/quantization">Quantization</a>: 未找到描述</li><li><a href="https://cloud.google.com/tpu/docs/run-calculation-jax">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/sad-sad-cat-cat-depressed-depression-gif-13240550249247957481">忧伤的小猫 GIF - Sad Sad cat Cat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/explosion-boom-iron-man-gif-14282225">爆炸轰鸣 GIF - Explosion Boom Iron Man - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/big-ups-mike-tyson-cameo-good-job-props-gif-18006586">Big Ups Mike Tyson GIF - Big Ups Mike Tyson Cameo - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/no-no-wait-wait-gif-8174347161288218584">不不等等 GIF - No no wait wait - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/surprise-welcome-one-sure-gif-13921142">惊喜欢迎 GIF - Surprise Welcome One - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k/tree/main">gradientai/Llama-3-8B-Instruct-262k at main</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ct5h16/llama_3_vs_llama_3_instruct/l49y05r/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/ml-explore/mlx-examples/blob/42458914c896472af617a86e3c765f0f18f226e0/llms/mlx_lm/tuner/trainer.py#L94C1-L98C46">mlx-examples/llms/mlx_lm/tuner/trainer.py at 42458914c896472af617a86e3c765f0f18f226e0 · ml-explore/mlx-examples</a>: MLX 框架中的示例。通过在 GitHub 上创建账号，为 ml-explore/mlx-examples 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80%</a>: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80%</a>: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://news.mit.edu/2024/natural-language-boosts-llm-performance-coding-planning-robotics-0501">自然语言提升了 LLM 在编码、规划和机器人技术方面的性能</a>: MIT CSAIL 研究人员创建了三种神经符号方法，帮助语言模型在自然语言中构建更好的抽象库：LILO 辅助代码合成，Ada 帮助 AI 规划...</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=IqM-T1RTzY6C">Google Colab</a>: 未找到描述</li><li><a href="https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">LLM 中的拒绝行为是由单一方向介导的 — LessWrong</a>: 这项工作是 Neel Nanda 在 ML Alignment &amp; Theory Scholars Program - 2023-24 冬季班项目中的一部分，由...共同指导。</li>

</li><li><a href="https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector">Steering GPT-2-XL by adding an activation vector — LessWrong</a>：通过添加激活向量来引导 GPT-2-XL — LessWrong。给模型的提示词 [1] 我讨厌你，因为 GPT-2 我讨厌你，因为你是我见过的最恶心的东西。GPT-2 + “Love” 向量我讨厌……</li><li><a href="https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">Refusal in LLMs is mediated by a single direction — AI Alignment Forum</a>：这项工作是 Neel Nanda 在 ML Alignment & Theory Scholars Program - 2023-24 冬季班中的一部分，由……共同指导。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1240937678560690236)** (55 messages🔥🔥): 

- **Python 中的 Regex 和文本格式化**：成员们讨论了使用 Python 识别相似格式文本的技术。建议包括使用 Regex (`re.findall`) 以及使用 `text.isupper()` 检查全大写。
  
- **对 Sam Altman 和 OpenAI 的批评**：针对 Sam Altman 的领导能力和 OpenAI 的影响力发表了强烈看法。评论反映了对 Altman 散布恐惧的策略以及科技界对财富和权力的偶像崇拜的反感。

- **在许可证中排除 OpenAI**：Cognitive Computations 正在修改许可证，禁止 OpenAI 和加利福尼亚州使用其模型和数据集。此举旨在传达他们反对当前 AI 领导层和政策的信息。

- **华盛顿特区的 AI 安全游说**：一篇分享的 [Politico 文章](https://www.politico.com/news/2024/05/12/ai-lobbyists-gain-upper-hand-washington-00157437) 讨论了 AI 游说者如何将华盛顿的辩论焦点从生存风险转向商业机会，并特别关注中国。

- **内容推荐**：成员们分享了一些有趣内容的链接，包括一段关于 GPU 的 Bend 编程语言的 [YouTube 视频](https://youtu.be/HCOQmKTFzYY)、一段 [Instagram reel](https://www.instagram.com/reel/C5n5C9AsB7Z/?igsh=MzRlODBiNWFlZA==)，以及一个名为 "[Dachshund Doom and Cryptid Chaos](https://youtube.com/playlist?list=PLB32jU2MhQwWPCi53uwDZFLEC8p96fTIn&si=qxrzTQ0ONEb-DmoG)" 的 YouTube 播放列表。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.politico.com/news/2024/05/12/ai-lobbyists-gain-upper-hand-washington-00157437">In DC, a new wave of AI lobbyists gains the upper hand</a>：在华盛顿，新一波 AI 游说者占据了上风。由科技巨头、初创公司和风险投资家组成的联盟正斥资数百万美元，试图说服华盛顿，对 AI 末日的恐惧被夸大了。到目前为止，这一策略奏效了。</li><li><a href="https://youtube.com/playlist?list=PLB32jU2MhQwWPCi53uwDZFLEC8p96fTIn&si=qxrzTQ0ONEb-DmoG">Dachsund Doom and Cryptid Chaos</a>：未找到描述</li><li><a href="https://youtu.be/HCOQmKTFzYY">Mind-bending new programming language for GPUs just dropped...</a>：什么是用于并行计算的 Bend 编程语言？让我们初步了解 Bend，以及它如何使用类似 Python 的语法来编写高性能……</li><li><a href="https://www.instagram.com/reel/C5n5C9AsB7Z/?igsh=MzRlODBiNWFlZA==">the forest jar on Instagram: &quot;be realistic&quot;</a>：3.8 万个赞，514 条评论 - theforestjar 于 2024 年 4 月 11 日发布：&quot;be realistic&quot;。 
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1240936506680545302)** (454 messages🔥🔥🔥): 

```html
- **Llama3 使用 torch.float16 时报错**：用户尝试使用 **torch.float16** 训练 Llama3，但遇到错误提示建议改用 bfloat16。他们寻求解决方案，但没有找到行之有效的方法。
- **Databricks 中的 torch 和 CUDA 问题**：**Torch** 在 Databricks 的 A100 80GB 上运行时引发错误。用户讨论了潜在的修复方案，如**将 torch 参数设置为 False** 或更新软件版本，但仍面临挑战。
- **上传和使用 GGUF 模型**：**用户在没有配置文件的情况下，在 Hugging Face 上上传和运行模型时面临挑战**。解决方案包括从预训练模型中获取配置文件，或确保格式正确并进行更新。
- **热切期待多 GPU 支持**：**社区成员对 Unsloth 的多 GPU 支持表示热切期待**，该功能正在开发中，尚未发布。
- **环境搭建故障排除**：参与者在 **WSL 和原生 Windows 上为 Unsloth 搭建环境时遇到困难**，特别是在安装 **Triton** 等依赖项时。
```

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">Quantization</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=FqfebeAdT073">Google Colab</a>：未找到描述</li><li><a href="https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora">使用 LoRA 和 QLoRA 微调 LLM 的深度指南</a>：在本博客中，我们详细解释了 QLoRA 的工作原理，以及如何使用 Hugging Face 对模型进行微调。</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit">unsloth/llama-3-8b-Instruct-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora#qlora-vs-standard-finetuning">使用 LoRA 和 QLoRA 微调 LLM 的深度指南</a>：在本博客中，我们详细解释了 QLoRA 的工作原理，以及如何使用 Hugging Face 对模型进行微调。</li><li><a href="https://www.unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>：通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，支持 6 倍长的上下文长度！</li><li><a href="https://huggingface.co/omar8/bpm_v2_gguf">omar8/bpm_v2_gguf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/inference-endpoints/dedicated">Inference Endpoints - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/omar8/bpm__v1/tree/main">omar8/bpm__v1 at main</a>：未找到描述</li><li><a href="https://blog.eleuther.ai/transformer-math/">Transformer 数学入门</a>：我们介绍了与 Transformer 计算和内存使用相关的基础数学。</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k/tree/main">gradientai/Llama-3-8B-Instruct-262k at main</a>：未找到描述</li><li><a href="https://tenor.com/view/cat-kitten-cat-crying-kitten-crying-05starrynight-gif-10141647709992578610">小猫 GIF - 小猫哭泣 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/conda-forge/miniforge#unix-like-platforms-mac-os--linux">GitHub - conda-forge/miniforge：一个 conda-forge 发行版。</a>：一个 conda-forge 发行版。通过在 GitHub 上创建账号为 conda-forge/miniforge 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth：微调 Llama 3、Mistral 和 Gemma LLM，速度提升 2-5 倍，内存占用减少 80%</a>：微调 Llama 3、Mistral 和 Gemma LLM，速度提升 2-5 倍，内存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/210">我在原生 Windows 中运行了 Unsloth。· Issue #210 · unslothai/unsloth</a>：我在原生 Windows（非 WSL）中运行了 Unsloth。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一个完整的安装教程，我本想写在这里，但我现在在用手机...</li><li><a href="https://github.com/unslothai/unsloth/issues/4">Apple Silicon 支持 · Issue #4 · unslothai/unsloth</a>：很棒的项目。希望能看到对 Apple Silicon 的支持！</li><li><a href="https://download.pytorch.org/whl/cu118/xformers-0.0.26.post1%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl">未找到标题</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7204">由 slaren 移除 convert-lora-to-ggml.py · Pull Request #7204 · ggerganov/llama.cpp</a>：模型转换过程中张量的排列等变化使得从 HF PEFT 转换 LoRA 变得不可靠，因此为了避免混淆，我认为在功能完善前最好将其完全移除...</li><li><a href="https://colab.research.google.com/drive/1gLSYbJWEBB93RkPWsJrqci45iqC9KE7H?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=r2v_X2fA0Df5">Google Colab</a>：未找到描述</li><li><a href="https://github.com/openai/triton.git">GitHub - triton-lang/triton：Triton 语言和编译器的开发仓库</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-gguf">主页</a>：微调 Llama 3、Mistral 和 Gemma LLM，速度提升 2-5 倍，内存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---

**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1241115840389320705)** (22 messages🔥): 

- **Text2Cypher 模型微调**：一位成员使用 Unsloth 微调了 **Text2Cypher 模型**（一种用于图数据库的查询语言）。他们分享了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/tomaz-bratanic-a58891127_im-very-excited-to-announce-that-ive-finetuned-activity-7197286502895075329-geKp?utm_source=share&utm_medium=member_desktop)，赞扬了其易用性以及生成的 gguf 版本。
- **情感分析新文章**：一位成员发表了一篇关于使用 Unsloth 微调 LLaMA 3 8b 进行情感分析的详尽文章，并附带了代码和指南。他们在 [Medium](https://medium.com/@seandearnaley/elevating-sentiment-analysis-ad02a316df1d) 上分享了这篇文章。
- **Kolibrify 中的关键数据采样 Bug**：在 Kolibrify 的数据采样过程中发现了一个重大 Bug。理论上能改善训练结果的修复程序将于下周发布，目前正在进行重新训练以评估效果。
- **课程数据集处理中的问题**：由于使用了 `datasets.Dataset.from_generator` 而非 `datasets.IterableDataset.from_generator`，课程数据生成器失效。一位成员彻底检查了他们的流水线，仅使用约 2 万个样本就达到了 dolphin-mistral-2.6 的性能，并计划很快发布该模型。
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1240928676409774110)** (853 messages🔥🔥🔥): 

```html
- **Issue with GPTs Agents on MPS Devices**: A member noted that **GPTs agents** can only load bfloat16 models with MPS devices, as bitsandbytes isn't supported on M1 chips. They expressed frustration with MPS being fast but "running in the wrong direction".
- **Member seeks MLflow deployment help**: Someone asked for assistance in deploying custom models via **MLflow**, specifically for a fine-tuned cross encoder model. They did not receive a direct response from other members.
- **Interest in HuggingChat's limitations**: A user inquired why **HuggingChat** doesn't support files and images. No comprehensive answer was provided.
- **Clarifying technical script adjustments**: Multiple users engaged in debugging and modifying a script for sending requests to a vllm endpoint using **aiohttp** and **asyncio**. Key changes and adaptations were discussed, particularly for integrating with OpenAI's API.
- **Concerns about service and model preferences**: An extensive discussion ensued regarding the benefits and downsides of Hugging Face's **Pro accounts**, spaces creation, and the limitations versus preferences for running models like **Llama**. One member expressed dissatisfaction with needing workarounds for explicit content and limitations on tokens in HuggingChat. Another user sought advice on deployment vs. local computation for InstructBLIP.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://aixblock.org/docs">Docs | AIxBlock</a>: AIxBlock 是一个全面的 AI 计划链上平台，集成了去中心化超级计算机。</li><li><a href="https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat">Zephyr Chat - a Hugging Face Space by HuggingFaceH4</a>: 未找到描述</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: 让社区最好的 AI 聊天模型惠及每一个人。</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/not-a-nerd-bart-nerds-are-smart-milhouse-the-simpsons-gif-16461565">Not A Nerd Bart GIF - Not A Nerd Bart Nerds Are Smart - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces/KingNish/GPT-4o">OpenGPT 4o - a Hugging Face Space by KingNish</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/optimum/v1.16.2/amd/ryzenai/overview">AMD Ryzen AI</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=qTsdgUyMY94&t=640s">Influenceuse I.A : POURQUOI et COMMENT créer une influenceuse virtuelle originale ?</a>: 各位 Zinzins 大家好！🤪 虚拟影响力者的迷人世界正在本视频中展开。它们的创作正在经历真正的繁荣，事情正在发生变化...</li><li><a href="https://x.com/LiamFedus/status/1790064963966370209?mx=2">Tweet from William Fedus (@LiamFedus)</a>: GPT-4o 是我们最新的 SOTA 前沿模型。我们一直在 LMSys arena 上以 im-also-a-good-gpt2-chatbot 的名义测试一个版本 🙂。以下是它的表现。</li><li><a href="https://github.com/ggerganov/llama.cpp/tree/master/examples/server#extending-or-building-alternative-web-front-end">llama.cpp/examples/server at master · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/TonyLianLong/LLM-groundedDiffusion">GitHub - TonyLianLong/LLM-groundedDiffusion: LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models (LLM-grounded Diffusion: LMD)</a>: LLM-grounded Diffusion：利用大语言模型（LLM）增强文本生成图像扩散模型的提示词理解能力 (LLM-grounded Diffusion: LMD) - TonyLianLong/LLM-groundedDiffusion</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot/blob/main/app.py">app.py · huggingface-projects/LevelBot at main</a>: 未找到描述</li><li><a href="http://hf.co/papers">Daily Papers - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593">MIT/ast-finetuned-audioset-10-10-0.4593 · Hugging Face</a>: 未找到描述</li><li><a href="https://pypi.org/project/ratelimiter/">ratelimiter</a>: 简单的 Python 速率限制对象</li><li><a href="https://www.gradio.app/guides/using-hugging-face-integrations#using-hugging-face-inference-api">Using Hugging Face Integrations</a>: Gradio 循序渐进教程</li><li><a href="https://huggingface.co/spaces/parler-tts/parler-tts-expresso">Parler TTS Expresso - a Hugging Face Space by parler-tts</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/parler-tts/parler_tts_mini">Parler-TTS Mini - a Hugging Face Space by parler-tts</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/hf-audio/open_asr_leaderboard">Open ASR Leaderboard - a Hugging Face Space by hf-audio</a>: 未找到描述</li><li><a href="https://huggingface.co/chat/models/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 - HuggingChat</a>: 在 HuggingChat 中使用 HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1</li><li><a href="https://huggingface.co/docs/hub/spaces-overview">Spaces Overview</a>: 未找到描述</li><li><a href="https://tenor.com/view/dog-snoop-dogg-rabjouj-gif-21804700">Dog Snoop GIF - Dog Snoop Dogg - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/learn/deep-rl-course/unit1/hands-on#install-dependencies-and-create-a-virtual-screen-">Train your first Deep Reinforcement Learning Agent 🤖 - Hugging Face Deep RL Course</a>: 未找到描述</li><li><a href="https://tenor.com/view/wolf-of-wall-street-jordan-belfort-leonardo-di-caprio-one-of-us-jonah-hill-gif-5441859">One Of Us GIF - Wolf Of Wall Street Jordan Belfort Leonardo Di Caprio - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/docs/transformers/v4.41.0/model_doc/instructblip">InstructBLIP</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot/blob/ca772d68a73c254a8d1f88a25ab15765361a836e/app.py#L240">app.py · huggingface-projects/LevelBot at ca772d68a73c254a8d1f88a25ab15765361a836e</a>: 未找到描述</li><li><a href="htt

ps://www.reddit.com/r/LocalLLaMA/comments/1cj4det/llama_3_70b_instruct_works_surprisingly_well_on/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.</a>: 一个用于 Large Language Models 的 Gradio Web UI。支持 Transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。 - oobabooga/text-generation-webui</li><li><a href="https://github.com/ollama/ollama">GitHub - ollama/ollama: Get up and running with Llama 3, Mistral, Gemma, and other large language models.</a>: 快速上手 Llama 3, Mistral, Gemma 以及其他 Large Language Models。 - ollama/ollama</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cj4det/llama_3_70b_instruct_works_surprisingly_well_on">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot/blob/main/app.py#:~:text=if%20reaction.message.author.id%20!%3D%20user.id%3A%20%23%20can%27t%20earn%20while%20self%2Dreacting%2C%20which%20is%20abuseable)">app.py · huggingface-projects/LevelBot at main</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot/blob/ca772d68a73c254a8d1f88a25ab15765361a836e/app.py#L110">app.py · huggingface-projects/LevelBot at ca772d68a73c254a8d1f88a25ab15765361a836e</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/tree/master/examples/server#extending-or-building-alternative-web-front-end>">llama.cpp/examples/server at master · ggerganov/llama.cpp</a>: 使用 C/C++ 进行 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://docs.google.com/spreadsheets/d/1C8aLqgCqLYcMiIFf-P_Aosaa03C_WLIB_UyqvjSdWg8/edit#gid=0">test_merge</a>: Sheet1 discord_user_id, discord_user_name, discord_exp, discord_level, hf_user_name, hub_exp, total_exp, verified_date, likes, models, datasets, spaces, discussions, papers, upvotes L251101219542532097L, osansevier...</li><li><a href="https://elevenlabs.io/">Text to Speech &amp; AI Voice Generator</a>: 使用有史以来最强大的在线 AI Text to Speech (TTS) 软件，以任何风格和语言免费创建优质 AI 语音。在几分钟内通过我们的角色 AI 语音生成文本转语音配音...</li><li><a href="https://www.udio.com/">Udio | AI Music Generator - Official Website</a>: 发现、创作并与世界分享音乐。使用最新技术在几秒钟内创作 AI 音乐。</li><li><a href="https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt">Transformers, what can they do? - Hugging Face NLP Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/audio-course/chapter4/introduction">Unit 4. Build a music genre classifier - Hugging Face Audio Course</a>: 未找到描述</li><li><a href="https://github.com/muellerzr/minimal-trainer-zoo">GitHub - muellerzr/minimal-trainer-zoo: Minimal example scripts of the Hugging Face Trainer, focused on staying under 150 lines</a>: Hugging Face Trainer 的极简示例脚本，专注于保持在 150 行以内 - muellerzr/minimal-trainer-zoo</li><li><a href="https://huggingface.co/learn">Hugging Face - Learn</a>: 未找到描述</li><li><a href="https://huggingface.co/blog">Hugging Face – Blog</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/142q5k5/updated_relative_comparison_of_ggml_quantization/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1240974446559232011)** (11 条消息🔥): 

- **AI 商业顾问项目分享**：一位成员分享了一个 [YouTube 视频](https://youtu.be/uQcHXEGRECU)，题为 "使用 LangChain 和 Gemini AI 创业的商业顾问 AI 项目"，展示了一个旨在利用这些技术创建商业顾问的项目。这是一个具有实际应用价值的创业想法。
  
- **简化 🤗 Transformers 安装**：一位用户分享了 [Transformers 安装指南](https://huggingface.co/docs/transformers/installation)，提供了使用 PyTorch, TensorFlow 和 Flax 设置该库的说明。这有助于用户为他们的深度学习项目安装和配置 🤗 Transformers。
  
- **创新的博客/页眉细节分享**：一位成员描述了他们新的博客/页眉，其特点是在节点上运行康威生命游戏（Game of Life）的 Delaunay 三角剖分。他们强调将游戏规则重新设计为分数计数，并提到由于使用 d3 重新渲染每一帧而不是使用 GPU 优化，因此具有“巨大的渲染开销”。

- **成果分享邀请**：针对商业顾问项目视频，另一位成员鼓励分享成果或仓库，以促进社区协作和反馈。
  
- **AI 人声增强指南发布**：一位用户简要提到他们编写了一份关于让 AI 人声听起来更自然的指南，通过增加厚度和深度使其重焕生机。目前尚未提供该指南的进一步详情或链接。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/installation">Installation</a>：未找到描述</li><li><a href="https://youtu.be/uQcHXEGRECU">business advisor AI project using langchain and gemini AI startup.</a>：在这段视频中，我们制作了一个使用 LangChain 和 Gemini 构建商业顾问的项目。AI 创业点子。我们恢复了作品集 AI 创业点子。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1240978405139152976)** (18 messages🔥): 

- **结合 LlamaParse 的多模态 GPT-4o**：分享了一篇名为“释放多模态力量：GPT-4o 与 LlamaParse 的集成”的文章。[点击此处阅读更多](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a)。

- **YouTube 上的技术盛赞**：声称发现了 YouTube 上可能有史以来最好的技术视频。[点击此处观看](https://youtu.be/dX9CGRZwD-w)。

- **对 OpenAI 的批评**：“OpenAI 并不 Open（开放）”，引发了关于封闭 AI 系统的讨论。[观看视频](https://www.youtube.com/watch?v=8BlRT7Ktw1c) 批评大厂 AI。

- **RLHF 与 LLM 评估**：分享了关于 RLHF 和 LLM 评估现状的有益讨论。[观看对话](https://www.youtube.com/watch?v=u8xxEkH3a5g&ab_channel=RunLLM)，特邀嘉宾为 Nathan Lambert。

- **物理学中的生成式 AI**：介绍了一种利用生成式 AI 回答物理学复杂问题的新研究技术，可能有助于研究新型材料。[阅读全文](https://news.mit.edu/topic/artificial-intelligence2)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/giffmana/status/1791541209883717973?s=46">Lucas Beyer (bl16) (@giffmana) 的推文</a>：Merve 表现得非常出色：引用 merve (@mervenoyann) —— 有人问我关于 PaliGemma 的文档理解能力，所以我建立了一个包含所有 PaliGemma 微调文档的 Space...</li><li><a href="https://youtu.be/AhyznRSDjw8?si=tZjOSRP_ZQMyQIxv">MIT 6.S191 (2023): Reinforcement Learning</a>：MIT 深度学习导论 6.S191：第 5 讲，深度强化学习。讲师：Alexander Amini，2023 版。包含所有讲座、幻灯片和实验材料...</li><li><a href="https://www.youtube.com/watch?v=8BlRT7Ktw1c">Big Tech AI Is A Lie</a>：通过 Hubspot 的免费 AI for GTM 捆绑包学习如何在工作中使用 AI：https://clickhubspot.com/u2o。大厂 AI 确实存在很多问题，而且是个谎言。✉️ NEWSLETT...</li><li><a href="https://news.mit.edu/topic/artificial-intelligence2">Artificial intelligence | MIT News | Massachusetts Institute of Technology</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=u8xxEkH3a5g&ab_channel=RunLLM">Generating Conversation: RLHF and LLM Evaluations with Nathan Lambert (Episode 6)</a>：本周的《生成对话》节目邀请了 Nathan Lambert。Nathan 是 HuggingFace 的研究科学家和 RLHF 团队负责人。他在...完成了博士学位。</li><li><a href="https://github.com/mintisan/awesome-kan">GitHub - mintisan/awesome-kan: KAN (Kolmogorov-Arnold Network) 相关资源的全面集合，包括库、项目、教程、论文等，供 Kolmogorov-Arnold Network 领域的研究人员和开发人员参考。</a>：KAN (Kolmogorov-Arnold Network) 相关资源的全面集合，包括库、项目、教程、论文等...</li><li><a href="https://www.noaa.gov/education/resource-collections/climate/climate-change-impact">Climate change impacts</a>：虽然我们经常认为人为造成的气候变化是未来才会发生的事情，但它实际上是一个持续的过程。美国及世界各地的生态系统和社区正在...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1240974461302476852)** (20 messages🔥): 

- **商业 AI 顾问项目上线**：一段名为 [使用 LangChain 和 Gemini 的商业顾问 AI 项目](https://youtu.be/uQcHXEGRECU) 的 YouTube 视频展示了一个旨在利用这些技术创建商业顾问的项目。它包含了一个针对 AI 创业点子的简历作品集。

- **使用 GenAI 的学习伴侣程序**：在 [LinkedIn](https://www.linkedin.com/posts/harshdayal_educationinnovation-genai-activity-7197227129409810432-4llP) 上分享了一个利用 GenAI 作为强大学习伴侣的程序。该工具旨在创新教育体验。

- **SimpleTuner 新模型训练支持**：[SimpleTuner 已增加对 SDXL、SD 1.5 和 SD 2.1 的完整 ControlNet 模型训练支持](https://github.com/bghira/SimpleTuner/blob/main/documentation/CONTROLNET.md)，进一步扩展了其功能。

- **SDXL Flash 模型发布**：推出了 [两个版本的 SDXL Flash](https://huggingface.co/sd-community/sdxl-flash)，承诺在 AI 模型中提供更快的性能和更高的质量。同时还推出了 SDXL Flash Mini，在提供高效性能的同时几乎没有质量损失。

- **受 Andrej Karpathy 启发的 Tokenizer 创新**：一名成员开发了 [Tokun，一种新型 Tokenizer](https://github.com/apehex/tokun)，据报道它可以将 Llama 模型的大小缩小 10 倍，同时增强其能力。更多见解和测试文章已在 [Twitter](https://x.com/4pe0x/status/1792638900059385942) 上分享。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/uQcHXEGRECU">使用 langchain 和 gemini AI 创业公司的商业顾问 AI 项目</a>：在本视频中，我们制作了一个使用 LangChain 和 Gemini 打造商业顾问的项目。AI 创业点子。我们恢复了作品集 AI 创业点子。</li><li><a href="https://swiftapi.pro/">Swift API</a>：未找到描述</li><li><a href="https://huggingface.co/sd-community/sdxl-flash">sd-community/sdxl-flash · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/4pe0x/status/1792638900059385942">来自 Apehex (@4pe0x) 的推文</a>：很高兴介绍 `tokun`，一个为 #LLM 带来变革的 #tokenizer。它可以将 #llama3 的大小缩小 10 倍，同时提升能力！https://github.com/apehex/tokun/blob/main/arti...</li><li><a href="https://huggingface.co/spaces/narra-ai/friday">Friday - narra-ai 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/KingNish/SDXL-Flash">SDXL Flash - KingNish 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/sd-community/sdxl-flash-mini">sd-community/sdxl-flash-mini · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/apehex/tokun">GitHub - apehex/tokun: tokun to can tokens</a>：tokun to can tokens。通过创建 GitHub 账号为 apehex/tokun 的开发做出贡献。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1240970710609035366)** (109 条消息🔥🔥): 

- **寻求生成式 AI 资源**：
  一位用户请求获取学习 Generative AI 和 LLM 的资源。推荐内容包括 "Attention is All You Need" 等研究论文以及 [HuggingFace](https://huggingface.co/learn) 上的课程。

- **AlphaFold3 读书会环节**：
  一位用户分享了一篇关于 AlphaFold3 的[博客文章](https://huggingface.co/blog/as-cle-bert/what-is-going-on-with-alphafold3)，该文章同时适合生物学家和计算机科学家。其他人建议将其作为下一次读书会的主题。

- **条件故事生成论文**：
  宣布了一场讨论多篇论文的会议，包括条件故事生成框架 GROVE（[arxiv 链接](https://arxiv.org/abs/2310.05388)），以及用于叙事理解的 Conan 基准测试（[arxiv 链接](https://www.arxiv.org/abs/2402.11051)）。

- **录音与资源**：
  成员们询问如何获取会议录音。录音已分享至 [YouTube](https://www.youtube.com/watch?v=UvWVfVnVZXc)，过往演示文稿的链接可在 [GitHub](https://github.com/isamu-isozaki/huggingface-reading-group) 上找到。

- **关于未来演示的讨论**：
  讨论了未来的主题，包括 AlphaFold3 以及可能涵盖的其他论文（如 KAN 论文）。此外还分享了这些环节的排期方式和时间详情。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=UvWVfVnVZXc">Hugging Face Reading Group 21: Understanding Current State of Story Generation with AI</a>：演讲者：Isamu Isozaki。书面总结：https://medium.com/@isamu-website/understanding-ai-for-stories-d0c1cd7b7bdc。所有演示文稿：https://github.com/isamu-isoz...</li><li><a href="https://huggingface.co/learn">Hugging Face - Learn</a>：未找到描述</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>：未找到描述</li><li><a href="https://studio.youtube.com/playlist/PLyKDb3IHyjoGE-Z5crcm0TtTRorLbP9mz/videos">YouTube</a>：未找到描述</li><li><a href="https://github.com/isamu-isozaki/huggingface-reading-group">GitHub - isamu-isozaki/huggingface-reading-group: This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group</a>：该仓库的目标是预编译 Hugging Face 读书会过去所有的演示文稿 - isamu-isozaki/huggingface-reading-group</li><li><a href="https://arxiv.org/abs/2310.05388">GROVE: A Retrieval-augmented Complex Story Generation Framework with A Forest of Evidence</a>：条件故事生成在人机交互中具有重要意义，特别是在创作具有复杂情节的故事方面。虽然 Large Language Models (LLMs) 在多项 NLP 任务中表现出色，但...</li><li><a href="https://www.arxiv.org/abs/2402.11051">Large Language Models Fall Short: Understanding Complex Relationships in Detective Narratives</a>：现有的叙事理解数据集往往无法体现现实社交场景中关系的复杂性和不确定性。为了填补这一空白，我们引入了一个新的基准测试...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1240935488651792394)** (1 messages): 

- **Tuxemons 替代 Pokemons 增加数据集趣味性**：一位成员宣布了一个新的替代数据集，使用 Tuxemons 代替 Pokemons。他们提到：*"样本数量较少，但图像均为 `cc-by-sa-3.0` 协议，因此在实验中你可以获得更多自由，减少顾虑。"* 此外，每张图像都配有两种类型的 captions，以增加描述的多样性。[探索该数据集](https://huggingface.co/datasets/diffusers/tuxemon)。

**提到的链接**：<a href="https://huggingface.co/datasets/diffusers/tuxemon">diffusers/tuxemon · Datasets at Hugging Face</a>：未找到描述

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1240970302461050891)** (25 messages🔥): 

- **关于模型结构问题的分歧**：成员们讨论了一个 Unet 模型的性能，重点关注 `forward` 和 `fit` 方法。有人强调了模型结构中潜在的问题，这些问题导致了收敛（convergence）问题，尽管运行成功，但结果几乎是随机猜测。

- **创建虚拟 AI 网红**：一位成员分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=qTsdgUyMY94)，内容是关于利用 Computer Vision 和 AI 工具创建虚拟 AI 网红（influencer）。该视频旨在详细介绍虚拟网红这一引人入胜且蓬勃发展的趋势。

- **处理 Parquet 文件中的图像数据**：讨论了在 Parquet 文件中包含图像的方法，以及上传到 Hugging Face 时图像数据以 byte array 格式出现的问题。建议的替代方案是使用 datasets 库，并提供了一个 [GitHub 链接](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation#creating-a-datasetdict)，指导如何从包含图像路径的 dictionary 创建数据集。

- **关于全卷积网络（Fully Convolutional Networks）的澄清**：简短的交流澄清了全卷积网络在检测头中避免使用 dense layers，这与 yolov2 和 yolov1 等模型形成对比。yolov2 相比 yolov1 的性能提升被视为一项优势。

- **CenterCrop 与图像增强**：在讨论 ViT 教程时，一位成员质疑了当输入和输出图像尺寸相等时 CenterCrop 的效用，认为它起到了 identity function 的作用。对方澄清说，CenterCrop 会引入噪声，并通过裁剪后调整大小来起到图像增强（augmentation）的作用。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=qTsdgUyMY94&t=640s">Influenceuse I.A : POURQUOI et COMMENT créer une influenceuse virtuelle originale ?</a>: 大家好！🤪 虚拟影响者（Virtual Influencers）的迷人世界走进了这段视频。它们的创作正经历着真正的繁荣，形势正在发生变化……</li><li><a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation#creating-a-datasetdict">transformers/examples/pytorch/semantic-segmentation at main · huggingface/transformers</a>: 🤗 Transformers: 为 Pytorch, TensorFlow, 和 JAX 提供最先进的 Machine Learning。 - huggingface/transformers
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1241038097806917734)** (13 条消息🔥): 

- **Connectionist Temporal Classification (CTC) 的相关性**: 一位成员询问 **Connectionist Temporal Classification** (CTC) 在今天是否仍在使用。目前没有后续跟进或回复。

- **访问 Hugging Face 模型架构**: 一位成员询问如何查看 **Hugging Face 预训练模型**的架构。另一位成员解释说，可以在 GitHub 上的建模文件、文档中找到，或者通过使用 `help(model)` 并检查配置文件来查看。

- **将文本查询分类为命令**: 一位成员寻求指导，希望将文本查询转换为用于翻译模型和视频游戏等应用的离散命令。然而，聊天中没有建议具体的模型或方法。

- **理解 LLM 中的 HTML**: 一位用户表达了在使用 **Large Language Models (LLMs)** 理解和生成 HTML 代码方面的困难。他们不确定 HTML 是否应被视为与自然语言不同的模态，并就如何有效使用不同的 Tokenizer 寻求建议。

- **处理基于 LLM 的机器人对话历史**: 一位用户在处理机器人无法记住之前对话的问题上感到困扰并寻求帮助。另一位用户解释说，LLM 需要手动处理对话历史，通常是通过将之前的消息与新的 Prompt 进行拼接。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1241084774227906681)** (22 条消息🔥): 

- **Google Colab 中 Hugging Face Diffusion 模型报错**: 一位用户在 Google Colab 中进行 HuggingFace Diffusion 模型课程的第 7 步时，遇到了路径相关的 `ValueError`。建议他们检查是否正确创建了 Pipeline。

- **SDXL 配置问题及使用示例**: 另一位用户报告了与 `SDXL` 模型中时间嵌入向量长度相关的 `ValueError`。讨论内容包括分享代码片段，以及建议参考 [Hugging Face 文档](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl#refine-image-quality) 中记录的 Stable Diffusion XL 模型使用方法。

- **初学者解决 Diffusers 问题指南**: 一位初学者询问如何开始解决与 Diffusers 相关的问题，建议他们学习 [Fastai 课程](https://www.fast.ai/)，并参考 Hugging Face GitHub 上之前合并的带有 "good first issue" 标签的 PR。

- **Discord LLM 聊天机器人问题**: 一位用户在使用 Discord LLM 聊天机器人时遇到问题，机器人无法记住对话历史，并将每条消息视为新的对话。建议他们在 NLP 频道发布问题，并参考 [LangChain 文档](https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/) 中用于维持历史记录的代码片段。

- **特定语言查询的重定向**: 提醒大家保持使用英文讨论，并将一位用户重定向到更合适的频道处理其 NLP 相关查询。这确保了内容与 "Diffusion Models" 讨论的相关性。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/diffusion-course/unit1/2#step-7-push-your-model-to-the-hub)">🤗 Diffusers 入门 - Hugging Face Diffusion 课程</a>: 未找到描述</li><li><a href="https://github.com/huggingface/diffusers/issues?q=label%3A%22good+first+issue%22+sort%3Acreated-asc)">Issues · huggingface/diffusers</a>: 🤗 Diffusers: 在 PyTorch 和 FLAX 中用于图像和音频生成的先进扩散模型。- Issues · huggingface/diffusers</li><li><a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl#refine-image-quality)?">Stable Diffusion XL</a>: 未找到描述</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/">添加消息历史 (memory) | 🦜️🔗 LangChain</a>: RunnableWithMessageHistory 允许我们将消息历史添加到某些类型的链中。它包装了另一个 Runnable 并为其管理聊天消息历史。</li><li><a href="https://github.com/jakobdylanc/discord-llm-chatbot">GitHub - jakobdylanc/discord-llm-chatbot: llmcord.py • 和你的朋友们一起与 LLMs 聊天！</a>: llmcord.py • 和你的朋友们一起与 LLMs 聊天！通过在 GitHub 上创建账户为 jakobdylanc/discord-llm-chatbot 的开发做出贡献。</li><li><a href="https://huggingface.co/blog?tag=diffusion&p=1).">Hugging Face – 博客</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1240934408966635562)** (939 条消息🔥🔥🔥): 

- **Perplexity 在 GPT-4o 模型限制方面遇到困难**：用户注意到 **GPT-4o** 经常重复之前的回答，并且在对话过程中无法有效地切换话题。一位用户将其描述为：*“作为一名高级用户，在过去的几年里，我从未见过任何 LLM 像这样完全无视 Prompt。”*。
- **图片上传功能需求**：成员们表达了在 Perplexity 中上传和分析视频及图片的愿望，并将其与 OpenAI 平台上的功能进行了对比。尽管进行了尝试，但目前尚不支持此类功能。
- **API 限制问题持续存在**：多位用户正在寻求提高 Perplexity API 的速率限制（rate limits），其中一人表示他们已经等待了两周的回复，并询问支持团队是否可以加快额度提升。
- **模型切换和自定义脚本**：讨论强调了一个流行的用户脚本，该脚本允许在 Perplexity 中进行动态模型切换。用户分享了脚本和 [Violentmonkey](https://violentmonkey.github.io/) 等工具的链接，通过实现可用 AI 模型之间的快速切换，增强了平台的可用性。
- **网站宕机引发不满**：Perplexity 经历了宕机，这让依赖该服务进行日常工作的用户感到沮丧。在此期间，一些用户甚至幽默地要求：*“我要求因这次事故获得无限量的 Opus”*。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/terminator-endoskeleton-flames-on-fire-t800-gif-14919281">Terminator Endoskeleton GIF - Terminator Endoskeleton Flames - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://docs.perplexity.ai/docs/rate-limits">Rate Limits</a>：未找到描述</li><li><a href="https://ollama.com/library/dolphin-llama3">dolphin-llama3</a>：Dolphin 2.9 是 Eric Hartford 基于 Llama 3 开发的新模型，提供 8B 和 70B 两种尺寸，具备多种指令、对话和编码技能。</li><li><a href="https://fonts.google.com/specimen/Karla">Karla - Google Fonts</a>：Karla 是一款 grotesque 无衬线字体系列，现已扩展为可变字体，粗细轴范围从 ExtraLight 到 ExtraBold，并提供完整支持。</li><li><a href="https://www.theverge.com/2024/5/16/24158529/reddit-openai-chatgpt-api-access-advertising">Reddit’s deal with OpenAI will plug its posts into “ChatGPT and new products”</a>：Reddit 与 OpenAI 签署了 AI 授权协议，将其帖子接入 “ChatGPT 及新产品”。Reddit 还与 Google 签署了授权协议。</li><li><a href="https://violentmonkey.github.io/">no title found</a>：未找到描述</li><li><a href="https://www.google.com/search?q=<searchquery>">&lt;searchquery&gt; - Google Search</a>：未找到描述</li><li><a href="https://spectrum.ieee.org/perplexity-ai">Perplexity.ai Turns Tables on Google, Upends SEO Credos</a>：AI 搜索领导者将 Meta 构建的智能与初创公司的拼搏精神相结合，扭转局面挑战 Google。</li><li><a href="https://tenor.com/view/oh-no-homer-simpsons-hide-disappear-gif-16799752">Oh No Homer GIF - Oh No Homer Simpsons - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.google.com/search?q=this+is+cool">this is cool - Google Search</a>：未找到描述</li><li><a href="https://greasyfork.org/en/scripts/490634-perplexity-model-selection">Perplexity Model Selection</a>：使用 jQuery 为 Perplexity AI 添加模型选择按钮</li><li><a href="https://deepmind.google/technologies/veo/">Veo</a>：Veo 是我们迄今为止功能最强大的视频生成模型。它能生成高质量、1080p 分辨率且时长可超过一分钟的视频，涵盖广泛的电影和视觉风格。</li><li><a href="https://www.youtube.com/watch?v=LT-b1qXznKI">The Fast Show - Suit you Sir ! -16- Johnny Depp</a>：Johnny Depp 主演一名美国人……</li><li><a href="https://docs.perplexity.ai/docs/perplexitybot">PerplexityBot</a>：未找到描述</li><li><a href="https://youtu.be/AxIk_MtryDQ?t=11">Gorgon City - Roped In</a>：Selected - 音乐新高度。» Spotify: https://selected.lnk.to/spotify » Instagram: https://selected.lnk.to/instagram » Apple Music: https://selected.lnk.t...</li><li><a href="https://www.udio.com/songs/kyBuHwPy8bLDpr2J2yhC1a">dailyfocus - Opus 50 | Udio</a>：在 Udio 上收听 dailyfocus 的 Opus 50。发现、创作并与世界分享音乐。利用最新技术在数秒内创作 AI 音乐。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1240971642381926430)** (12 条消息🔥): 

- **Stability AI 引起用户兴趣**：一名成员分享了一个[链接](https://www.perplexity.ai/search/Stability-AI-is-CznMl2swRumQbTO5U4AzIw)来探索 Stability AI 的功能和产品。讨论围绕该 AI 技术的潜在应用和益处展开。
- **步行的脑部益处**：另一名成员发布了关于[步行的脑部益处](https://www.perplexity.ai/search/brain-benefits-of-VJYShXcNROeGjfaWRL842w)的内容。分享的链接旨在详细说明“步行”如何对认知功能和整体心理健康产生积极影响。
- **WASP-193b 是什么？**：一场讨论由一个探索系外行星 WASP-193b 的[链接](https://www.perplexity.ai/search/What-is-WASP193b-IBFHgr6RQ4W2E3eqaOPPBg#0)发起。内容似乎集中在这一天体的天文发现和特征上。
- **分析狗狗症状**：有一个关于狗狗表现出异常症状（如平衡失调和颈部持续活动）的查询，链接至[此搜索](https://www.perplexity.ai/search/un-perro-presenta-.n42RyNMTCqlfqBRpvaExw)。讨论可能涉及兽医学见解或可能的诊断。
- **用 Dungeons & Dragons 娱乐孩子**：一位家长分享了一个[链接](https://www.perplexity.ai/search/Generate-a-dungeons-gxx_hPaAQfWy1RCSqft1iA)，用于生成 Dungeons & Dragons 场景来娱乐他们的孩子。重点在于让这款奇幻游戏对儿童更具吸引力和趣味性。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1240932748835819520)** (19 条消息🔥):

- **Perplexity API 使用说明澄清**：分享了使用 [Perplexity API](https://docs.perplexity.ai/reference/post_chat_completions) 生成模型响应的说明。一位用户强调默认的 temperature 值为 0.2。
- **理解 OpenAI 首席科学家查询**：成员们讨论了查询 OpenAI 现任首席科学家的挑战。建议模型应该能够处理时间线并提供正确答案：Jakub Pachocki。
- **API 性能测试**：用户注意到不同模型在类似查询下的成功率各异，其中 *Omni* 表现良好。有人不愿使用 *labs.perplexity.ai* 进行 API 性能测试。
- **API 使用的 Rate Limits**：讨论并澄清了 API 的请求速率限制，指出请求限制（20次/分钟）与 token 限制（2,000,000/分钟）之间存在差异，并对未来模型容量进行了推测。
- **API 中的 Threads 功能**：有人提出了关于 Threads 功能的问题，该功能在网页端很突出，但在 API 中似乎缺失。澄清指出，最接近的功能是添加来自先前消息的更多 role/content。

**提到的链接**：<a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>：未找到描述

---

**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1242002561725366332)** (1 条消息): 

- **暂停 ChatGPT 中的 Sky 语音**：OpenAI 宣布在处理用户疑虑期间暂停在 ChatGPT 中使用 Sky 语音。他们分享了一个 [链接](https://openai.com/index/how-the-voices-for-chatgpt-were-chosen/) 来解释这些语音是如何被选中的。

---

**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1240927868910043146)** (347 条消息 🔥🔥): 

- **LangChain 无需 OpenAI API 即可工作**：成员们讨论了将 LangChain 与各种 LLM 配合使用，包括通过 Ollama 等工具在本地使用。一位用户确认“你可以将 LangChain 与任何你想要的 LLM 配合使用”。
- **对 GPT-4o 可用性的困惑**：在访问 GPT-4o 方面存在不同的体验；一些用户报告称尽管拥有访问权限，但仍缺失部分功能。澄清指出 GPT-4o 正在逐步 rollout，所有功能将很快上线。
- **ChatGPT-4o 的视频和实时处理能力**：讨论围绕 ChatGPT-4o 如何以 2-4 fps 处理视频帧及其在实时调整方面的能力展开。成员们辩论了模型是否能根据新的数据输入在 mid-stream 调整响应。
- **Usage caps 导致 GPT-4o 的限制**：一位成员对 ChatGPT App 当前的 usage caps 表示沮丧，认为这使得许多潜在应用变得不切实际。其他人指出，设定 usage caps 是为了平衡并确保所有用户获得一致的体验。
- **GPT-4o 的多模态能力受到赞扬**：尽管存在批评，GPT-4o 的 multimodal 能力仍受到称赞，一位用户强调它同时集成了音频、视频和文本处理。成员们还提到，该模型开启了超越传统文本模型的新可能性。

分享了 [Pricing](https://openai.com/api/pricing/) 和 [file upload FAQ](https://help.openai.com/en/articles/8555545-file-uploads-faq) 链接以获取更多详情。

---

**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1240935678985113641)** (167 条消息 🔥🔥): 

- **GPT-4o 中的 Context window 困惑与限制**：一位用户澄清“128k tokens 的 context window”是指 AI 可以处理的整个输入。许多参与者对使用大量 token 时的限制和错误表示沮丧，并将其与 Gemini 更大的 context window 能力进行了比较。
- **Custom GPTs 与模型切换**：针对使用 GPT-4o 的 Custom GPTs 的问题得到了解答，透露目前尚无法轻松切换模型。此外，成员们分享称一些 Custom GPTs 已经过渡到了 GPT-4o。
- **GPT-4o 的可用性与逐步推出**：许多用户对访问 GPT-4o 表示困惑和沮丧，尤其是在 iOS 和免费账户上。解释称 rollout 是分阶段进行的，目前没看到该选项的用户将随着时间的推移获得访问权限。
- **用户对模型 Rate Limits 和性能的沮丧**：关于 GPT-4o 与普通 GPT-4 之间差异的讨论包括了性能差异和 rate limits 的共同经历。注意到 GPT-4o 看起来更快，但一些用户发现普通 GPT-4 的回答结构更好。
- **未来功能及语音/聊天能力的推出**：用户推测了 GPT-4o 新功能（如 vision 和语音能力）的 rollout 时间线，官方回应指出将在未来几个月内为 Plus 用户分阶段实施。

---

**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1240928457118978149)** (178 messages🔥🔥): 

- **ChatGPT 在自我意识和提示词清晰度方面面临挑战**：成员们分享了在询问 ChatGPT 关于其自身信息或优化提示词以获取特定修正时遇到的困难。一位成员指出，*"模型会将其视为指令，并尝试为你寻找最佳答案。"*
- **通过 Fine-tuning 和 JSON mode 获得更好的结果**：多位成员讨论了通过 Fine-tuning GPT-4 和使用 JSON mode 来提升提示词质量。分享了 [OpenAI 关于 JSON mode 的文档](https://platform.openai.com/docs/guides/text-generation/json-mode) 以辅助此过程。
- **用于创意和精准度的复杂提示词策略**：分享了如 "Humanizer" 和 "Pandomain Prompt Voyager" 等详细且高度结构化的提示词，以优化和改进模型的创意及结构化内容生成。
- **与 GPT-4 的编码和技术集成**：成员们讨论了 `chat.completion.create` 中回复不完整的问题，以及使用 GPT-4 创建 UI 的提示词策略。一位成员分享了关于 Visual Studio Code 插件的具体经验和故障排除步骤。
- **利用模型行为和示例引导回复**：探索了确保模型行为精准的技术，例如设置字符限制和调整回复准确度。分享了具体的挑战和解决方案，以帮助模型提供更具体、更具可操作性的回复。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1240928457118978149)** (178 messages🔥🔥): 

```html
<ul>
  <li><strong>ChatGPT 难以有效地优化提示词：</strong>用户对 <strong>4o</strong> 无法跟进提示词修正或有效修改初稿表示沮丧。一位成员指出，“它会重写原始回复，而不是告诉我如何修复我的提示词。”</li>
  
  <li><strong>对回复不完整的挫败感：</strong>像 cicada.exe 这样的用户报告称，尽管没有超过 Token 限制，但 <code>chat.completion.create</code> 仍会出现回复不完整的情况。输出被突然截断的问题依然存在。</li>
  
  <li><strong>实现 JSON mode：</strong>Ashthescholar 建议 razeblox 在 API 中使用 <a href="https://platform.openai.com/docs/guides/text-generation/json-mode">JSON mode</a> 来解决回复问题，特别是在格式和内容控制方面。</li>
  
  <li><strong>创意写作提示词在 GPT-4 上的表现优于 4o：</strong>用户分享到，虽然 4o 在某些创意任务上表现出色，但在优化草稿方面却很吃力。“当给予 4o 创意写作的自由发挥空间时，它表现得相当不错，但如果给它一份初稿进行改进，根据我的经验，它通常只是重复初稿内容，而不是进行修改，”keller._ 指出。</li>
  
  <li><strong>创意合成的创新方法：</strong>Stunspot 分享了一个提示词——“在混沌边缘编排创新”（Orchestrating Innovation on the Fringes of Chaos），强调通过网络动力学、分形探索、自适应创新和韧性来探索想法，以促进突破。</li>
</ul>
```
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1240935611045515344)** (537 messages🔥🔥🔥):

```html
<ul>
    <li><strong>GPTs Agents 在初始训练后无法学习</strong>：一位成员询问了在本地存储对话以进行上下文搜索的能力，另一位成员澄清说目前在 LM Studio 中无法实现。他们建议复制并粘贴文本，但指出“你无法上传文档并与之聊天。”</li>
    <li><strong>处理 “Unsupported Architecture” 错误</strong>：多位成员讨论了在 LM Studio 中加载 GPT-Sw3 时出现的 “Unsupported Architecture” 问题。共识是仅支持 GGUF 文件，用户建议在应用内下载并开启 “compatibility guess”。</li>
    <li><strong>在有限显存 (VRAM) 系统上运行 LM Studio</strong>：用户询问了如何在 6-8GB 等显存有限的系统上运行 LLM 模型。成员建议使用较小的模型和 Q5_K_M 等量化版本以获得更好的性能。</li>
    <li><strong>离线使用问题</strong>：一位用户报告了 LM Studio 在离线状态下无法运行的问题。经过社区建议，明确了先加载模型然后禁用网络应该是可行的，但建议提交更详细的 Bug 报告。</li>
    <li><strong>常规故障排除和设置问题</strong>：用户经常询问有关服务器设置、模型兼容性以及在低配置系统上的性能等问题。许多用户被引导至特定频道 (<a href="https://discord.com/channels/1111440136287297637">#1139405564586229810</a>) 发布详细帖子以获取进一步帮助。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF">Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - a Hugging Face Space by ggml-org</a>：未找到描述</li><li><a href="https://docs.continue.dev/walkthroughs/tab-autocomplete#setting-up-with-lm-studio">Tab Autocomplete (beta) | Continue</a>：Continue 现在支持在 VS Code 和 JetBrains IDE 中进行 Tab 自动补全。我们将在接下来的几个版本中大幅改进体验，听到反馈总是很有帮助的。如果...</li><li><a href="https://youtu.be/OphjEzHF5dY?si=2q9v3Bqe6tqBS7Ma">AGI Breaks the Team at OpenAI: Full Story Exposed</a>：由于 AGI 问题，高管们离开 OpenAI。#ai #ainews #openai #agi #singularity 0:00 Intro 1:14 Background 4:53 Chief scientist leaves 6:40 Sam's response 9:34 Supe...</li><li><a href="https://github.com/Lisoveliy/StarCoderEx">GitHub - Lisoveliy/StarCoderEx: Extension for using alternative GitHub Copilot (StarCoder API) in VSCode</a>：在 VSCode 中使用替代 GitHub Copilot (StarCoder API) 的扩展 - Lisoveliy/StarCoderEx</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7118">llama : add DeepSeek-v2-Chat support · Issue #7118 · ggerganov/llama.cpp</a>：请支持 deepseek-ai/DeepSeek-V2-Chat https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat</li><li><a href="https://www.hwinfo.com/download/">Free Download HWiNFO Sofware | Installer &amp; Portable for Windows, DOS</a>：现在就开始分析你的硬件！HWiNFO 提供适用于 Windows（32/64 位）的安装版和便携版，以及适用于 DOS 的便携版。</li><li><a href="https://github.com/xtekky/gpt4free">GitHub - xtekky/gpt4free: The official gpt4free repository | various collection of powerful language models</a>：官方 gpt4free 仓库 | 各种强大语言模型的集合 - xtekky/gpt4free</li><li><a href="https://www.nuget.org/packages/OpenAI/1.11.0?_src=template#json-mode">OpenAI 1.11.0</a>：一个简单的 C# / .NET 库，用于 OpenAI 的 API，包括 GPT 3.5, GPT 4, ChatGPT, DALL-E, Whisper 等。独立开发，非官方库，我不隶属于...</li><li><a href="https://www.nuget.org/packages/OpenAI/1.11.0?_src=template">OpenAI 1.11.0</a>：一个简单的 C# / .NET 库，用于 OpenAI 的 API，包括 GPT 3.5, GPT 4, ChatGPT, DALL-E, Whisper 等。独立开发，非官方库，我不隶属于...</li><li><a href="https://www.nuget.org/packages/OpenAI/1.11.0">OpenAI 1.11.0</a>：一个简单的 C# / .NET 库，用于 OpenAI 的 API，包括 GPT 3.5, GPT 4, ChatGPT, DALL-E, Whisper 等。独立开发，非官方库，我不隶属于...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1241055146058715167)** (82 条消息🔥🔥):

- **医疗 LLM 推荐**：一位成员询问了医疗 LLM，另一位成员建议尝试 [OpenBioLLM-Llama3-8B-GGUF](https://huggingface.co/aaditya/OpenBioLLM-Llama3-8B-GGUF)，并指出其具有 8.03B 参数和 Llama 架构。推荐者还分享了使用该模型的 Spaces 等额外资源。
  
- **SVG 和 ASCII 艺术基准测试**：一位成员分享了 LLM 生成 SVG 艺术的基准测试结果，指出 **WizardLM2** 是目前的冠军，并将其与 **GPT-4o** 进行了比较。另一位成员询问了 ASCII 艺术能力，结果显示 **GPT-4o** 在 ASCII 方面表现良好。

- **德语 Embedding 模型**：讨论了在使用 LM Studio 时难以找到合适的德语 Embedding 模型的问题。成员们建议尝试使用 llama.cpp 等工具手动转换模型，并提供了一个特定的[多语言模型](https://huggingface.co/intfloat/multilingual-e5-large)用于潜在的转换。

- **生成基于文本的艺术**：一位用户提到使用 **MPT-7b-WizardLM** 模型生成不受限的故事，另一位用户询问了配置和 Prompt 设置。该模型的创建者建议使用特定的量化版本（quants）和正确的模板以避免问题。

- **图像质量关注**：关于图像生成质量的简短讨论建议使用 **automatic1111** 和 **ComfyUI** 等工具以获得更好的控制和改进的结果。对话建议从 **Civit.ai** 获取高质量模型，尽管对其 NSFW 内容提出了警告。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - a Hugging Face Space by ggml-org</a>：未找到描述</li><li><a href="https://huggingface.co/aaditya/OpenBioLLM-Llama3-8B-GGUF">aaditya/OpenBioLLM-Llama3-8B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/DavidAU/TieFighter-Holodeck-Holomax-Mythomax-F1-V1-COMPOS-20B-gguf">DavidAU/TieFighter-Holodeck-Holomax-Mythomax-F1-V1-COMPOS-20B-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF">DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/intfloat/multilingual-e5-large">intfloat/multilingual-e5-large · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/DavidAU">DavidAU (David Belton)</a>：未找到描述</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1242144163001143406)** (7 条消息): 

- **Hugging Face 与 LM Studio 集成**：团队推出了全新的“HF -> LM Studio deeplink”功能，允许用户浏览 Hugging Face，发现感兴趣的模型，然后点击“Use this model”将其导入 LM Studio。此功能需要 LM Studio 0.2.23 或更高版本，专注于本地 AI 使用，无云端依赖。

- **v1 版本中的手动下载选择**：在当前版本的功能中，用户从 Hugging Face 导入模型时，需要手动选择想要下载的文件。

- **自动下载建议**：用户建议进行改进，包括为自动下载设置默认量化级别，以及根据可用 RAM 配置该功能以自动下载最合适的模型。

- **正面的用户反馈**：社区反应积极，一位成员表示他们一直在寻找这样的按钮，并认为加入该功能非常有益。

**提到的链接**：<a href="https://x.com/LMStudioAI/status/1792576553601102024">来自 LM Studio (@LMStudioAI) 的推文</a>：1. 浏览 HF 2. 这个模型看起来很有趣 3. 在 LM Studio 中使用它 👾🤗 引用 clem 🤗 (@ClementDelangue) 无云、无成本、不向任何人发送数据、没问题。欢迎来到 Hugging Face 上的本地 AI...

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1240954781153890305)** (10 条消息🔥): 

- **Comodo 标记 llama.cpp 二进制文件**：一位成员注意到 Comodo 杀毒软件触发了对 llama.cpp 二进制文件的警报。另一位成员解释说，这可能是因为二进制文件未签名，这会导致严格的杀毒软件将其标记。

- **模型加载错误排查**：一位用户分享了尝试在 LM Studio 中加载模型时的 JSON 错误消息。错误表明尽管 RAM 和 VRAM 充足，但模型操作失败，建议他们尝试不同的模型或配置。

- **AVX 支持澄清**：一位成员询问为什么 LM Studio 不支持 AVX。回复提到，支持较少的老旧硬件可以减少需要处理的 Bug 和问题。

- **磁盘空间 Bug 导致下载崩溃**：一名成员报告称，在下载模型时磁盘空间耗尽会导致程序崩溃并重置队列，导致不清楚哪些模型未完全下载。

- **服务器启动问题**：另一名成员分享了日志，表明尽管启用了详细的服务器日志（verbose server logs），服务器仍无法启动。
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1240950923992240128)** (32 messages🔥): 

- **成员剖析 LLama3 模板转换**：一位用户在将 Prompt 模板转换为 LLama3 时遇到困难，询问如何调整其现有格式。另一位成员提出了一个详细模板，包含历史对话以提供上下文，并强调“状态由客户端保持，而非服务器端”。

- **利用 LangChain Memory 管理聊天历史**：讨论显示用户依赖 LangChain 的 `ConversationBufferWindowMemory` 来管理聊天历史和用户输入。在收到关于构建 Prompt 模板的建议后，用户确认：“是的，它有效，会进行更多实验，谢谢！”

- **提到 Gemini 的 Context Caching**：针对处理对话历史，有人建议了一个替代方案：“像 Gemini 的 Context Caching 这样的新服务”，尽管用户表示相比付费方案更倾向于开源解决方案。

- **通过调整 System Prompt 避免回复截断**：另一位用户建议在 System Prompt 中加入“不要过早截断回复（Do not prematurely cut off a response）”，以避免回复不完整，为正在进行的 Prompt 讨论提供了一个实用技巧。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1240932589867630623)** (93 messages🔥🔥): 

- **研究 Alder Lake 和量化**：一名成员讨论了在 Alder Lake CPU 上禁用 E-cores 时的性能差异，指出 Q8 量化（Q8 quants）的性能从 *0.4 tokens/sec 提升到了 0.6 tokens/sec*。他们还在 IQ3 量化中遇到了不连贯的结果，并考虑自行进行量化。

- **Tesla P100 表现令人失望**：有一场关于各种 GPU 的对比讨论，指出**拥有 700+ GB/s 显存带宽的 Tesla P100 甚至难以击败 GTX 1060**。尽管规格很高，但在某些任务中它无法超越 K80 等旧型号。

- **规避 Apple 昂贵的存储价格**：一名成员通过在 Thunderbolt 外壳中使用**外部 4TB M.2 SSD** 避开了 Apple 昂贵的 SSD 价格，实现了*超过 2GB/second 的传输速度*。

- **多 GPU 设置：成本 vs 性能**：深入讨论了多 GPU 设置的实际收益，一些**经验证据表明，由于 PCIe 带宽限制等问题，超过两个 GPU 后收益递减**。

- **RAM 速度对 LLM 性能的影响**：一组详细的测试显示，**提高 RAM 速度可以提升 LLM 性能**，尽管效果因模型和量化方法而异。例如，*将 RAM 从 2133MHz 升级到 3200MHz 可以显著提高 Token 输出速度，但存在性能差异*。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/turboderp/exllama/discussions/16#discussioncomment-6245573">各种硬件上的性能测试 · turboderp/exllama · Discussion #16</a>：首先，我要感谢你的工作，非常喜欢你的推理实现，目前看来它是 NVIDIA GPU 上最快的！我在各种 GPU 上运行了一系列测试，想要……</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1811fk4/comment/kahdtgs/">Reddit - 深入探讨</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/search/?q=pcie+multigpu">Reddit - 深入探讨</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1241025397479440484)** (1 messages): 

- **聊天已移至新频道！**：一位用户提到他们已将聊天移至新频道。提供的链接将成员引导至 Discord 上的新讨论位置，点击[此处](https://discord.com/channels/1110598183144399058/1111440136287297637/1240773722441519126)。
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1241047852478890056)** (12 messages🔥):

- **LM Studio autogen bug produces brief responses**: 用户报告遇到 **LM Studio** 仅回复 1-2 个单词后紧跟 TERMINATE 消息的问题。一位用户指出这是由于一个已计划修复的 bug 导致的。
- **Autogen issues linked to max_tokens setting**: 该问题似乎与 **max_tokens** 属性被设置为 **null** 有关。根据多位用户的说法，将此属性设置为 **-1** 可以解决该问题。
- **LM Studio's OpenAI emulation is off-spec**: 用户建议 **LM Studio 的本地服务器** 并不完全符合 OpenAI 规范，特别是在处理 **max_tokens** 参数方面。这种不正确的处理导致了响应的过早终止。
- **CLI LMStudio Client workaround**: 一位正在构建 CLI LMStudio 客户端的用户确认，将 **max_tokens** 设置为 **-1** 解决了截断问题。对于像 **AutoGPT** 这样的工具，可能需要手动调整以正确处理 tool_calls。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1241434022597623890)** (21 messages🔥): 

- **6600XT workaround for LM Studio**: 成员们讨论了 **AMD 6600XT 显卡** 及其在使用 **OpenCL** 进行 GPU offload 时与 **LM Studio** 的兼容性。一位成员确认：“支持 OpenCL。这是 Intel 和非 ROCm AMD 显卡能够进行 GPU offload 的方式。”

- **Call for Linux users testing ROCm**: 一位用户呼吁拥有**较新 AMD GPU 的 Linux 用户**测试集成了 ROCm 的早期版本 LM Studio。包括拥有 6900XT 和 6600XT 在内的感兴趣成员做出了积极回应，尽管某些 GPU 未被正式列出。[在此查看支持列表](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

- **ROCm with different Linux distributions**: 成员们报告了在 **Arch Linux**、**Ubuntu 22.04 LTS** 和 **Fedora 40** 等各种 Linux 发行版上配合不同 AMD GPU 运行 **ROCm** 的情况。一位用户确认：“ROCm 6.1 可以在 Arch Linux 上配合 6900xt 运行，至少官方的 torch nightly 构建版是可以的。”

- **Reunion in the Discord**: 对话中包含了一个轻松的时刻，两名用户在 Discord 中认出了彼此。其中一人回复道：“是的，不过大部分时间都在这里潜水 :)。”
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/reunited-peaches-and-herb-and-it-feels-so-good-cause-we-understood-old-skool-gif-17279659">Reunited Peaches And Herb GIF - Reunited Peaches And Herb And It Feels So Good - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://rocm.docs.am">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1240933645049401344)** (664 messages🔥🔥🔥): 

- **Combining and Using Multiple LoRAs in Prompts**: 用户讨论了如何在 Stable Diffusion 的提示词中使用类似 `<lora:pack1:1><Lora:pack2:1>` 的语法组合多个 LoRAs。一位用户确认，添加超过三个可能会导致问题。
- **Persistent Issues with Stable Diffusion on First Run**: 一位新用户在首次运行 Stable Diffusion 时遇到了错误，指出存在 'NoneType' 对象属性问题。他们向社区寻求帮助，但尚未提供明确的解决方案。
- **Lively Debate on SD3 Release and Preparations**: 关于 SD3 的发布存在持续的讨论和一些质疑。然而，其他人安慰说它最终会发布，并强调了 Emad Mostaque 确认正在努力的一条推文。
- **Topaz as a Video Upscaling Solution**: 用户辩论了 Topaz 在视频超分辨率（upscaling）方面的有效性。虽然大家一致认为它是一个强大的工具，但也提出了对其成本以及 ComfyUI 等替代方案的担忧。
- **SDXL Model and ControlNet Usage Tips**: 一位用户分享了关于运行 SDXL 模型时 VRAM 重要性的见解，提到更高的分辨率需要更多的显存。另一位用户澄清说，与 SD1.5 相比，SDXL 模型需要独立的 ControlNet 模型。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1002292111942635562/1089974139927920741/1241293682435428352">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://discordapp.com/channels/1002292111942635562/1089974139927920741/1241614349315870793">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://discordapp.com/channels/100229211194263556">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://idm-vton.github.io/">IDM-VTON</a>: 未找到描述</li><li><a href="https://play.google.com/store/apps/details?id=com.grisoft.pixart&hl=en&gl=US">Pixart AI Photo Editor - Google Play 应用</a>: 未找到描述</li><li><a href="https://invideo.io/">Invideo AI - 将创意转化为视频 - AI 视频生成器 </a>: 通过向 invideo AI 提供提示词轻松制作视频。invideo AI 是内容创作者、YouTuber 和营销人员的理想选择，它提供了一种无缝的方式，利用 AI 将你的创意转化为可发布的视频。</li><li><a href="https://play.google.com/store/apps/details?id=com.grisoft.pixart&hl=en&gl">Pixart AI Photo Editor - Google Play 应用</a>: 未找到描述</li><li><a href="https://tenor.com/view/apo-solary-apo-apofps-gif-9924237009492714744">Apo Solary Apo GIF - Apo Solary Apo ApoFPS - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/aw-shucks-aww-thank-you-gif-25109804">Aw Shucks GIF - Aw Shucks Aww - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/smash-gif-21365305">Smash GIF - Smash - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/high-five-amy-santiago-rosa-diaz-stephanie-beatriz-melissa-fumero-gif-23124416">High Five Amy Santiago GIF - High Five Amy Santiago Rosa Diaz - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/EMostaque/status/1790451196915831018?t=YJuHqJJ-YCivInuOrZ2_Lw&s=33">来自 Emad (@EMostaque) 的推文</a>: @morew4rd @GoogleDeepMind SD3 即将发布，老实说，有了正确的流水线，我觉得大家不需要更多其他东西了</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/wiki/ControlNet-model-download">ControlNet 模型下载</a>: 通过在 GitHub 上创建账号，为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://huggingface.co/lllyasviel/sd_control_collection/tree/main">lllyasviel/sd_control_collection at main</a>: 未找到描述</li><li><a href="https://pinokio.computer/">Pinokio</a>: AI 浏览器</li><li><a href="https://github.com/yisol/IDM-VTON">GitHub - yisol/IDM-VTON: IDM-VTON : Improving Diffusion Models for Authentic Virtual Try-on in the Wild</a>: IDM-VTON：改进扩散模型以实现真实的野外虚拟试穿 - yisol/IDM-VTON</li><li><a href="https://github.com/BadCafeCode/masquerade-nodes-comfyui">GitHub - BadCafeCode/masquerade-nodes-comfyui: A powerful set of mask-related nodes for ComfyUI</a>: 一套强大的 ComfyUI 遮罩相关节点。通过在 GitHub 上创建账号，为 BadCafeCode/masquerade-nodes-comfyui 的开发做出贡献。</li><li><a href="https://tenor.com/beb52.gif">Judges 10 GIF - Judges 10 Score Up - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/mood-sad-panda-gif-14650463720672021603">Mood Sad GIF - Mood Sad Panda - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://stable-diffusion-art.com/">Stable Diffusion Art - 教程、提示词和资源</a>: Stable Diffusion 是一个将文本转化为图像的免费 AI 模型。本网站提供易于遵循的教程、工作流和结构化课程，教你关于 Stable Diffusion 所需了解的一切...</li><li><a href="https://civitai.com/images/12597091">20Twenty 发布的图像</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1241049054482403410)** (74 条消息🔥🔥): 

- **Mojo 在 Windows 上的困扰**: 多位用户讨论了 Mojo 在 Windows 上缺乏直接支持的问题，特别提到了在使用 CMD 或 Powershell 时遇到的问题。官方表示，Mojo SDK 目前适用于 *Ubuntu 和 macOS*，目前预计将通过 WSL 提供对 [Windows 的未来支持](https://docs.modular.com/mojo/manual/get-started/)。

- **Mojo vs. Bend 编程语言之争**：成员们对比了 Mojo 和 Bend 编程语言，*Chris Lattner* 提供了详细见解，指出 *Bend* 并非以性能为中心，且缺乏一些关键功能。Bend *目前在单核上的性能与 CPython 相当*，而 Mojo 的目标是即使在单 CPU 上也能实现高性能。

- **社区参与和资源**：大家对即将举行的公开社区会议感到兴奋，并提供了 [会议详情](https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit) 和 Zoom 会议链接。会议录像承诺将会分享。

- **趣味 Mojo 语法**：用户们尝试使用表情符号和反引号创建搞怪的 Mojo 代码，并分享了趣味代码片段。这引发了幽默的交流，彰显了社区积极活跃且富有幽默感的精神。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/">Modular: Accelerating the Pace of AI</a>：Modular Accelerated Xecution (MAX) 平台是全球唯一能为您的 AI 工作负载解锁性能、可编程性和可移植性的平台。</li><li><a href="https://www.google.com/url?q=https://modular.zoom.us/j/89417554201?pwd%3DVj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1&sa=D&source=calendar&ust=1716255791532130&usg=AOvVaw2IgLzFgI9-S5vkyEC7_b2v">重定向中</a>：未找到描述</li><li><a href="https://www.google.com/url?q=https://modular.zoom.us/j">重定向通知</a>：未找到描述</li><li><a href="https://docs.modular.com/mojo/manual/get-started/">Mojo🔥 入门 | Modular 文档</a>：获取 Mojo SDK 或尝试在 Mojo Playground 中编写代码。</li><li><a href="https://paper.higherorderco.com/)">PAPER.pdf</a>：未找到描述</li><li><a href="https://tenor.com/view/cloudy-with-a-chance-of-meatballs-enough-to-make-a-grown-man-cry-police-officer-make-a-man-cry-gif-15227532">《食破天惊》足以让一个成年男子流泪 GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/HCOQmKTFzYY">令人惊叹的全新 GPU 编程语言刚刚发布...</a>：什么是用于并行计算的 Bend 编程语言？让我们初步了解 Bend 以及它如何使用类似 Python 的语法来编写高性能代码...</li><li><a href="https://www.modular.com/max/engine">MAX Engine：全球最快的统一 AI 引擎</a>：全球最快的统一 AI 推理引擎，使您能够在不同框架和硬件上实现无与伦比的性能、可编程性和可移植性。</li><li><a href="https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit">[公开] Mojo 社区会议</a>：未找到描述</li><li><a href="https://github.com/tairov/llama2.mojo">GitHub - tairov/llama2.mojo: 仅用一个纯 🔥 文件实现 Llama 2 推理</a>：仅用一个纯 🔥 文件实现 Llama 2 推理。通过在 GitHub 上创建账号为 tairov/llama2.mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1791535613411570039>
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1241951209364131881)** (1 条消息): 

- **从零开始实现 Llama3**：一位成员分享了一个有趣的 [GitHub 仓库](https://github.com/naklecha/llama3-from-scratch) 链接，其中包含 **Llama3** 的实现。该仓库被描述为“一次一个矩阵乘法”地构建 Llama3。

**提到的链接**：<a href="https://github.com/naklecha/llama3-from-scratch">GitHub - naklecha/llama3-from-scratch: 一次一个矩阵乘法实现 llama3</a>：一次一个矩阵乘法实现 llama3 - naklecha/llama3-from-scratch

  

---


**Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1240949397605584917)** (4 条消息):

- **HVM 使用隐式并行程序模型**：关于 [HVM](https://higherorderco.com/) 如何实现自动并行化，已明确 HVM 以函数式形式（例如 Haskell）运行已经并行的算法。尽管其概念很酷，但在 [Hacker News 讨论](https://news.ycombinator.com/item?id=40390287) 中指出，其 CPU 性能慢于 CPython，GPU 性能低于 Mojo。
- **对 Mojo 的 GPU 和加速器支持感到兴奋**：在 Bend 发布之后，人们对 **Mojo** 将如何支持 **GPUs 和加速器** 充满期待。
- **速度关键型程序的共享内存 IPC**：对速度至关重要的程序已经在使用共享内存 IPC，这显著降低了延迟。讨论了 **DPDK 和 SPDK** 因其性能而得到更广泛使用的潜力，并希望提高易用性以及与 Mojo 的集成。
- **旧硬件和 MMU 依赖**：重要软件在某些执行模型下往往会变慢，因此在旧硬件退役之前，仍需继续使用 MMUs。有人担心旧硬件在允许区域之外使用 DMA，以及 **CXL 设备** 中的 64-bit 指针限制，这表明旧硬件的影响将长期存在。
- **类似 io_uring 的 API 前景**：未来的进步可能来自类似 io_uring 的 APIs，它们利用 syscalls 作为控制路径机制，并利用共享内存与内核通信。正如 Jens Axboe 的工作所示，这可以消除大部分开销，专注于改进 APIs。

**提到的链接**：<a href="https://news.ycombinator.com/item?id=40390287">未找到标题</a>：未找到描述

---

**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1240950434324287510)** (397 条消息🔥🔥): 

- **为 Tuple 实现 `__iter__` 和 `__contains__` 引发争论**：一位成员致力于为 `Tuple` 实现 `__iter__` 和 `__contains__`，并面临 `tuple_mutability` 和 `tuple_lifetime` 的问题。这引发了关于在 Mojo 中使用 `Tuple` 作为可迭代对象的实用性和设计选择的讨论，参考了相关的 GitHub issues，如 [issue #2658](https://github.com/modularml/mojo/issues/2658)。
- **探索集合和指针操作**：关于各种集合类型和操作（包括 `ListLiteral`、`Tuple`、`i1` 和 `SIMD`）正确使用的热烈讨论。关于 `rebind` 的作用和定义 MLIR 类型的辩论非常突出。
- **功能请求和增强，包括 Unicode 和单元测试中的分配**：成员建议了诸如 [在单元测试中断言最大分配量](https://github.com/modularml/mojo/issues/2725) 和更好的 Unicode 支持等功能，并询问了社区贡献的时间表和可行性。
- **使用线程安全和协程模型的并行性**：成员深入探讨了 Mojo 的线程安全和并行化方法，在类 OpenMP 语法和 Rust 的 async 模型之间进行辩论。
- **Mojo 的 Tensor 实现策略**：Chris Lattner 澄清说，Mojo 标准库不会包含最终的 tensor 实现，以确保开发者的灵活性。大家一致认为需要一个统一的 tensor trait，同时保持模块化的实现方式。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/collections/">collections | Modular Docs</a>: 实现 collections 包。</li><li><a href="https://www.uiua.org/">Uiua</a>: 未找到描述</li><li><a href="https://tenor.com/view/magic-gif-26166638">Magic GIF - Magic - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://mlir.llvm.org/docs/Dialects/Builtin/#integertype)">Builtin Dialect - MLIR</a>: 未找到描述</li><li><a href="https://www.google.com/url?q=https://modular.zoom.us/j/89417554201?pwd%3DVj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1&sa=D&source=calendar&usd=2&usg=AOvVaw37jsmYkBEWm4CHK4NwSCMB">Redirect Notice</a>: 未找到描述</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/rebind/#functions)">rebind | Modular Docs</a>: 实现类型 rebind。</li><li><a href="https://docs.modular.com/mojo/manual/python/#set-up-a-python-environment-with-conda>">Python integration | Modular Docs</a>: 同时使用 Python 和 Mojo。</li><li><a href="https://without.boats/blog/the-registers-of-rust/">The registers of Rust</a>: 未找到描述</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read">FileHandle | Modular Docs</a>: 已打开文件的文件句柄。</li><li><a href="https://github.com/modularml/mojo/pull/2703">[mojo-stdlib] Add variadic initialiser, __iter__ and __contains__ to InlineList by ChristopherLR · Pull Request #2703 · modularml/mojo</a>: 此 PR 为 InlineList 添加了一些功能（相关 issue #2658）。变长参数初始化器 `var x = InlineList[Int](1,2,3)`，迭代器 `var x = InlineList[Int](1,2,3); for i in x: print(i)`，包含判断 `var x = In...`</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/inferred-parameters.md">mojo/proposals/inferred-parameters.md at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/tree/main/stdlib/src/python">mojo/stdlib/src/python at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2725">[Feature Request] Assert max allocations in unit tests · Issue #2725 · modularml/mojo</a>: 审查 Mojo 的优先级。我已经阅读了路线图和优先级，并认为此请求符合优先级。你的请求是什么？描述：作为一名使用 Mojo 的开发者，我希望能够...</li><li><a href="https://victorzhou.com/blog/intro-to-neural-networks/">Machine Learning for Beginners: An Introduction to Neural Networks - victorzhou.com</a>: 简单解释它们的工作原理以及如何从零开始用 Python 实现一个神经网络。</li><li><a href="https://github.com/mzaks/mojo-unicode">GitHub - mzaks/mojo-unicode</a>: 通过在 GitHub 上创建账户为 mzaks/mojo-unicode 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2658">[stdlib] Implement `__contains__` for `Tuple`, `List`, `ListLiteral` (almost) · Issue #2658 · modularml/mojo</a>: 现在我们有了 ComparableCollectionElement，我们可以尝试使用类似于 #2190 中采用的变通方法，为一些常见的集合类型实现 `__contains__`。有可能...</li><li><a href="https://www.avanderlee.com/swift/custom-operators-swift/#calculating-with-emojis-in-swift">Custom Operators in Swift with practical code examples</a>: 学习如何在 Swift 中使用自定义运算符。其优势是什么，以及为了获得最佳可读性，哪些其他解决方案比自定义运算符更好。</li><li><a href="https://github.com/modularml/mojo/discussions/81#discussioncomment-5860938">Discussion of the Potential of Unicode Characters as Alias Operators · modularml/mojo · Discussion #81</a>: 在 Mojo 中使用 Unicode 逻辑和数学运算符：讨论引言。我发起这次讨论是为了探索引入 Unicode 逻辑和数学运算符的潜在好处...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1240990171525288087)** (31 条消息🔥): 

- **Rust 和 Go 共享内存分配技术**：讨论揭示了 Rust 的 **Vec** 和 Go 的 slices 在追加元素时都会将容量翻倍，直到达到特定阈值，之后 Go 会按 25% 的比例增长。相关链接包括 Rust 的 [raw_vec.rs](https://github.com/rust-lang/rust/blob/master/library/alloc/src/raw_vec.rs#L464) 和 [Go 的 runtime slice](https://github.com/golang/go/blob/cb2353deb74ecc1ca2105be44881c5d563a00fb8/src/runtime/slice.go#L95)。

- **Mojo 中 List 容量的优化见解**：在 Mojo 中调整 List 初始化容量（例如 `List[Int](capacity=N+50)`）与默认设置相比，带来了 2 倍的性能提升。Clattner 确认即将发布的补丁将解决 `def` 输入参数复制的问题，这将进一步增强性能。

- **关于 SIMD gather/scatter 优化的讨论**：成员们讨论了 Mojo 中掩码式 gather 和 scatter 指令的有效性，特别是在带有 AVX512 的 x86 和 ARM SVE 等不同架构上。虽然用户的反馈褒贬不一，但有人表示，由于潜在的内存墙（memory wall）问题，重新计算值有时可能比使用查找表更有利。

- **针对优化的潜在新 List 方法建议**：一位成员建议在 Mojo 的 List 中添加类似 Rust 的 `Vec::shrink_to_fit()` 方法以优化分配空间，并分享了他们在 MoString 中使用的[一个简单实现](https://github.com/dorjeduck/mostring)。

- **社区赞扬 Clattner 的详细见解**：一位成员对 Chris Lattner 分享关于 Mojo 内部机制的深度技术见解表示感谢，这显著帮助他们更好地理解和优化代码。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/collections/list.mojo#L223">mojo/stdlib/src/collections/list.mojo at bf73717d79fbb79b4b2bf586b3a40072308b6184 · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/dorjeduck/mostring">GitHub - dorjeduck/mostring: variations over StringBuilder ideas in Mojo</a>：Mojo 中关于 StringBuilder 构想的变体。通过在 GitHub 上创建账号为 dorjeduck/mostring 的开发做出贡献。</li><li><a href="https://github.com/golang/go/blob/cb2353deb74ecc1ca2105be44881c5d563a00fb8/src/runtime/slice.go#L95>">go/src/runtime/slice.go at cb2353deb74ecc1ca2105be44881c5d563a00fb8 · golang/go</a>：Go 编程语言。通过在 GitHub 上创建账号为 golang/go 的开发做出贡献。</li><li><a href="https://doc.rust-lang.org/std/vec/struct.Vec.html#capacity-and-reallocation">Vec in std::vec - Rust</a>：未找到描述</li><li><a href="https://github.com/rust-lang/rust/blob/master/library/alloc/src/raw_vec.rs#L464">rust/library/alloc/src/raw_vec.rs at master · rust-lang/rust</a>：赋能每个人构建可靠且高效的软件。 - rust-lang/rust</li><li><a href="https://doc.rust-lang.org/std/vec/struct.Vec.html#method.shrink_to_fit">Vec in std::vec - Rust</a>：未找到描述
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 条消息): 

Zapier: Modverse Weekly - 第 34 期
https://www.modular.com/newsletters/modverse-weekly-34
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/)** (1 条消息): 

ModularBot: 恭喜 <@891492812447698976>，你刚刚晋升到 3 级！
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1240923969490653216)** (114 条消息🔥🔥): 

- **解决 PR DCO 检查失败问题**：成员们讨论了将 fork 与 `nightly` 分支同步的问题，并提供了一份详细的分步指南，以避免提交数量虚增和 DCO 检查失败。诸如[将 `nightly` 设置为默认分支](https://github.com/modularml/mojo/issues/2556)之类的建议被作为潜在的修复方案提出。

- **处理 Nightly 和 Stable 版本发布**：分享了关于从 `nightly` 过渡到 stable 版本的流程说明。针对项目准备工作，解释了 stable 版本通常在正式发布前几天截取，且公开日期并不固定。

- **应对 Segfaults 和 Bug**：一位用户报告了在特定条件下使用自定义数组类型时出现的 segfault 问题。后续互动旨在调试和隔离问题，并建议使用内置类型，同时讨论了复杂类型的生命周期管理。

- **不稳定的测试和正在进行的修复**：Gab Peace 强调了与 `List.index()` 相关的 CI 测试不稳定问题，以及[潜在的修复方案](https://github.com/modularml/mojo/pull/2745)。他强调了这些 bug 对正在进行的工作（如 SSO 和单元测试中的断言）的影响。

- **Alias 问题导致 Segfaults**：成员们报告并讨论了与类型别名（aliasing）和实例化（materializing）相关的[各种 bug](https://github.com/modularml/mojo/issues/2753)，指出当前实现中存在的重大问题，并概述了这些 bug 如何阻碍当前工作。



<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/lifecycle/death#field-lifetimes))*">Death of a value | Modular Docs</a>: 关于 Mojo 何时以及如何销毁值的说明。</li><li><a href="https://dangitgit.com/en">Dangit, Git!?!</a>: 未找到描述</li><li><a href="https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md#branching-off-nightly">mojo/CONTRIBUTING.md at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2556)">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2556">[Feature Request] DX: Change the default branch of modularml/mojo from `main` to `nightly` · Issue #2556 · modularml/mojo</a>: 审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。你的请求是什么？我希望 modularml 管理员前往设置...</li><li><a href="https://github.com/gabrieldemarmiesse/getting_started_open_source">GitHub - gabrieldemarmiesse/getting_started_open_source: You want to contribute to an open-source project? You don&#39;t know how to do it? Here is how to.</a>: 你想为开源项目做贡献吗？不知道该怎么做？这里有方法。 - gabrieldemarmiesse/getting_started_open_source</li><li><a href="https://github.com/modularml/mojo/pull/2745">[stdlib] Fix out of bounds access in `List.index()` by gabrieldemarmiesse · Pull Request #2745 · modularml/mojo</a>: 与 #2687 相关。那里存在多个与裁剪相关的 bug。简而言之，Python 中 list.index() 的行为是：给定起始和结束位置，Python 会在其中查找元素...</li><li><a href="https://github.com/modularml/mojo/issues/2434">[BUG] param_env (string_literal, stringref, _memmem) · Issue #2434 · modularml/mojo</a>: Bug 描述编辑：见 GitHub issue 的第一条评论。你好，我当时正在编写一个非常有用的教程 👍，在添加了一个约束后：示例：from sys import param_env alias D.....</li><li><a href="https://github.com/modularml/mojo/issues/2751">[BUG] Flaky segfault during `mojo build` with `-D MOJO_ENABLE_ASSERTIONS` · Issue #2751 · modularml/mojo</a>: Bug 描述。此 Bug 是 #2687 的阻碍因素。在使用 -D MOJO_ENABLE_ASSERTIONS 编译 test_string.mojo 时，我注意到出现了一些不稳定的段错误（segfaults）。正如你所见，这在 CI 中是可以复现的...</li><li><a href="https://github.com/modularml/mojo/issues/2753">[BUG] alias materialization of list · Issue #2753 · modularml/mojo</a>: Bug 描述。你好，这是根据聊天中 @JoeLoser 的建议提交的 Bug 报告。def main(): alias x = List(&quot;ok&quot;) var y = x print(y[0]) mojo run main.mojo 请提交 Bug 报告至 htt...</li><li><a href="https://github.com/modularml/mojo/issues">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2637">[BUG] Incorrect pointer behavior when materializing a type · Issue #2637 · modularml/mojo</a>: Bug 描述。我正尝试实现小缓冲区优化（small buffer optimization）。为此，我有一个指针，它可以指向一些栈分配的数据，也可以指向堆。为了知道我们是否需要...
</li>
</ul>

</div>
  

---



**LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1241005256247087194)** (242 条消息🔥🔥): 

- **研讨会公告与期待**: 成员们分享了关于新研讨会的更新，例如 Jeremy Howard 主讲的名为 *Build Applications in Python* 的研讨会，计划于 2024 年 6 月 6 日举行。另一位成员对该课程持续的成功和高质量内容表示兴奋。

- **额度与资源讨论**: 针对 Modal Labs、Replicate、Jarvis Labs 和 LangSmith 等服务的各种额度获取，出现了多次询问和确认。一位成员确认收到了价值 200 美元的额外 Jarvis Labs 额度。

- **RAG 应用的 PDF 解析**: 成员们讨论了从 PDF 中解析表格数据的最佳工具，推荐了 LlamaParse、Vik Paruchuri 开发的 Marker，以及集成 GPT-4o 等模型进行复杂文档提取。另一位成员建议尝试使用 UniTable 进行 PDF 数据提取。

- **托管 LLM 与提供 API 服务**: 针对使用 FastAPI、Streamlit 和 Modal 等框架将微调后的 LLM 作为自定义 API 提供服务，成员们提出了询问和建议。分享了 Modal Labs 的示例仓库作为快速入门指南。

- **RAG 优化工作坊**：宣布了由 Jason Liu 主持的关于优化 RAG 模型的新工作坊，并邀请成员填写调查问卷以定制内容。鉴于已说明的前提条件，一些成员谨慎地表达了他们的兴趣。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://github.com">GitHub: Let’s build from here</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。在这里参与开源社区，管理你的 Git 仓库，像专业人士一样进行 Code Review，跟踪 Bug 和功能...</li><li><a href="https://x.com/realSharonZhou/status/1792576516444065967">Sharon Zhou (@realSharonZhou) 的推文</a>: Hallucinations 是生产级 LLM & Agent 的最大阻碍之一。在内部以及为客户服务时，我们已经实现了无 Hallucinations（<5%）。我们已经能够微调 LLM 来召回特定...</li><li><a href="https://x.com/VikParuchuri/status/1788966758742982696">Vik Paruchuri (@VikParuchuri) 的推文</a>: Marker v2 发布了！主要新功能：- 提取图像/图表 - 更好的表格解析 - Pip 包安装 - 可用于商业用途 - 改进了支持更多语言的 OCR - 更好的内容排序...</li><li><a href="https://x.com/llama_index/status/1791258285993230786">LlamaIndex 🦙 (@llama_index) 的推文</a>: 使用 GPT-4o 进行结构化图像提取 🖼️ GPT-4o 在整合图像/文本理解方面处于 State-of-the-art 水平，我们创建了一份完整的 Cookbook，向你展示如何使用 GPT-4o 提取结构化...</li><li><a href="https://x.com/Kyrannio/status/1792440824355332313">Kiri (@Kyrannio) 的推文</a>: 出于好奇，我找到了 GPT-4o iOS 系统的 Prompt：“你是 ChatGPT，一个由 OpenAI 训练的大型语言模型，基于 GPT-4 架构。你正在通过 ChatGPT iOS 界面与用户聊天...”</li><li><a href="https://www.quora.com/profile/Quora-Prompt-Generator/activity">Quora Prompt Generator - Quora</a>: 未找到描述</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/home">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/hf-accelerate/model-memory-usage">Model Memory Utility - hf-accelerate 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://x.com/jxnlco/status/1792549015273513102">jason liu (@jxnlco) 的推文</a>: 如果你是一家正在构建 RAG 的公司，并希望提升你的工程团队水平，请填写此表格。 https://q7gjsgfstrp.typeform.com/to/SL656ADC 我们将邀请其他运营者分享他们的故事，提供...</li><li><a href="https://x.com/runpod_io/status/1792101299087196615">RunPod (@runpod_io) 的推文</a>: @cleavey1985 @HamelHusain $501.45 看起来差不多</li><li><a href="https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base#2%EF%BC%89code-insertion)">deepseek-ai/deepseek-coder-6.7b-base · Hugging Face</a>: 未找到描述</li><li><a href="https://www.quora.com/profile/">Quora - 一个分享知识、更好地了解世界的场所</a>: 未找到描述</li><li><a href="https://github.com/bigcode-project/starcoder2-self-align/tree/main?tab=readme-ov-file#data-generation-pipeline">GitHub - bigcode-project/starcoder2-self-align: StarCoder2-Instruct: 用于 Code Generation 的完全透明且许可的 Self-Alignment</a>: StarCoder2-Instruct: 用于 Code Generation 的完全透明且许可的 Self-Alignment - bigcode-project/starcoder2-self-align</li><li><a href="https://x.com/charliebholtz/status/1791571514086629757?s=46&t=QitgwfFVpCSQgUY0DIcTdA">Charlie Holtz (@charliebholtz) 的推文</a>: 我们出 $501.43。引用 Omar Sanseviero (@osanseviero)：对 LLM 感兴趣？加入这个由顶尖专家主持的 Fine-Tuning 课程！🚀 @huggingface 为 Space 演示提供 $501.42 的 GPU 额度...</li><li><a href="https://github.com/VikParuchuri">VikParuchuri - 概览</a>: VikParuchuri 有 88 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/poloclub/unitable">GitHub - poloclub/unitable: UniTable: 迈向统一的表格 Foundation Model</a>: UniTable: 迈向统一的表格 Foundation Model - poloclub/unitable</li><li><a href="https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemma/docs/paligemma/fine-tuning-paligemma.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/explosion/prodigy-segment">GitHub - explosion/prodigy-segment: 通过 Facebook 的 Segment-Anything 模型在 Prodigy 中选择像素。</a>: 通过 Facebook 的 Segment-Anything 模型在 Prodigy 中选择像素。 - explosion/prodigy-segment</li><li><a href="https://pymupdf.readthedocs.io/en/latest/tutorial.html">教程 - PyMuPDF 1.24.4 文档</a>: 未找到描述</li><li><a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/text_generation_inference.py">modal-examples/06_gpu_and_ml/llm-serving/text_generation_inference.py (main 分支) · modal-labs/modal-examples</a>: 使用 Modal 构建的程序示例。通过在 GitHub 上创建账号，为 modal-labs/modal-examples 的开发做出贡献。</li><li><a href="https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving">modal-examples/06_gpu_and_ml/llm-serving (main 分支) · modal-labs/modal-examples</a>: 示例...</li>

ples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.</li><li><a href="https://www.amazon.science/publications/instructpts-instruction-tuning-llms-for-product-title-summarization">InstructPTS: Instruction-tuning LLMs for product title summarization</a>: 电子商务产品目录包含数十亿件商品。大多数产品的标题都很长，因为卖家会堆砌产品属性以提高检索率并突出关键产品方面。这项研究...</li><li><a href="https://github.com/xl0">xl0 - Overview</a>: 全职学习者。(Linux, Biology, Electronics) -> AI :heart: 编写一些可爱的软件。:two_hearts: 欢迎各种令人兴奋的机会！- xl0</li><li><a href="https://chinese-reader.vercel.app">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1240929862076862555)** (168 messages🔥🔥): 

- **LLM 不是数据库，它们是 pattern-bases**：一位成员讨论了关于 LLM 可以简单地通过 fine-tuning 学习领域知识的误解，强调 LLM 学习的是模式（patterns）而非一次性事实，并建议改用 Retrieval-Augmented Generation (RAG)。
- **用于车辆故障预测的 Fine-tuning**：提议将使用车辆诊断数据预测更换零件的故障类型作为一个可行的 fine-tuning 用例，因为输入和输出具有特定领域的性质。
- **Modal 命令问题已解决**：几位成员讨论了在使用 `modal` 命令进行训练时遇到的错误，最终通过删除之前创建的 volumes 解决了问题。
- **关于 LLM 用例的家庭作业回答**：各位成员提交了广泛的 LLM fine-tuning 用例，包括拼写检查、AI 艺术评论、市场研究机器人、编码模型增强、医疗和法律术语的语义增强、创意写作辅助等。
- **Dan Biderman 谈 LoRa 配置**：一条推文讨论了使用 LoRa 进行 continued pretraining 的细微差别，强调了避免性能下降和信息丢失的最佳参数和技术，并建议使用特定配置以获得更好的结果。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">sentence-transformers/all-MiniLM-L6-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.myaltimate.com/document/write/">编写文档 - dbt Power User</a>: 未找到描述</li><li><a href="https://modal.com/bobflagg/apps/ap-hlyOqSPPZNV28H45JTKzu5">登录</a>: 欢迎回到 Modal！请在下方选择身份提供商以登录您的 Modal 账户。</li><li><a href="https://youtu.be/LSNfwlfdrto?feature=shared">使用 LLM 创建项目经理 - 第一部分 (WIP)</a>: 项目经理花费大量时间提供和更新任务状态——这可以交给 LLM 吗？在这里我们开始调查。还有这个...</li><li><a href="https://github.com/genomoncology/FuzzTypes">GitHub - genomoncology/FuzzTypes: 用于标注自动纠错字段的 Pydantic 扩展。</a>: 用于标注自动纠错字段的 Pydantic 扩展。 - genomoncology/FuzzTypes</li><li><a href="https://blogs.microsoft.com/blog/2023/03/16/introducing-microsoft-365-copilot-your-copilot-for-work/">介绍 Microsoft 365 Copilot – 你的工作副驾驶 - 微软官方博客</a>: 人类天生就喜欢梦想、创造和创新。我们每个人都寻求从事赋予我们目标的工作——写一部伟大的小说、做出发现、建立强大的社区、照顾病人...</li><li><a href="https://github.com/modal-labs/modal-client/blob/f76bd98013372b423ab765cdc7a745996012211c/modal/object.py#L96-L103">modal-client/modal/object.py (位于 f76bd98013372b423ab765cdc7a745996012211c) · modal-labs/modal-client</a>: Modal 的 Python 客户端库。通过在 GitHub 上创建账户来为 modal-labs/modal-client 的开发做出贡献。</li><li><a href="https://x.com/danielhanchen/status/1791900967472140583">Daniel Han (@danielhanchen) 的推文</a>: 我对“LoRA 学得更少，遗忘也更少”的看法：1) “MLP/All”不包括 gate_proj。训练了 QKVO、up 和 down，但没有训练 gate（第 3 页脚注） 2) 为什么 LoRA 在数学方面表现良好，并且...</li><li><a href="https://github.com/nppoly/cyac">GitHub - nppoly/cyac: 适用于 Python 的高性能 Trie 和 Ahocorasick 自动机 (AC 自动机) 关键词匹配与替换工具</a>: 适用于 Python 的高性能 Trie 和 Ahocorasick 自动机 (AC 自动机) 关键词匹配与替换工具 - nppoly/cyac</li><li><a href="https://arxiv.org/abs/2212.09535">BLOOM+1: 为 BLOOM 添加语言支持以实现 Zero-Shot Prompting</a>: BLOOM 模型是一个大型公开可用的多语言语言模型，但其预训练仅限于 46 种语言。为了在不产生过高成本的情况下将 BLOOM 的优势扩展到其他语言...</li><li><a href="https://xebia.com/blog/lessons-learned-from-a-diy-llm-benchmark/">DIY LLM 评估，ABBA 模式押韵案例研究</a>: DIY LLM 评估，ABBA 模式押韵案例研究 - Xebia</li><li><a href="https://github.com/eliasdabbas/openai_entity_extraction">GitHub - eliasdabbas/openai_entity_extraction: 使用 ChatGPT 进行实体提取</a>: 使用 ChatGPT 进行实体提取。通过在 GitHub 上创建账户来为 eliasdabbas/openai_entity_extraction 的开发做出贡献。</li><li><a href="https://adver.tools/entity-extraction/">由 OpenAI 的 ChatGPT 驱动的实体提取 - advertools</a>: 未找到描述</li><li><a href="https://www.onetonline.org/find/all">查看所有职业</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1240945284154920971)** (47 条消息🔥): 

- **来自亚洲各地的成员自我介绍**：来自韩国、印度、日本、新加坡、澳大利亚等不同地区的个人加入了频道并互相问候。rparcus、vishnu9158、.thebigpanda 等用户分享了他们对课程的兴奋之情以及所在地。

- **对 GPU 提供商的赞扬和讨论**：一位成员表达了对 Jarvislabs 的钦佩，提到在购买个人 GPU 之前，它是他们的首选 GPU 提供商。Vishnu9158 对此表示赞赏，并希望他们将来会需要更多资源。

- **更倾向于录播而非直播**：像 rparcus 和 pugio 这样的成员分享了由于直播时间不便，他们更倾向于观看课程的录播。Vishnu9158 提到了不参加直播会错过社交机会的缺点。

- **作业讨论**：ivanleomk 分享了他对本周作业的尝试，列举了 Style Transfer、Classification、Extraction 以及用于提取的 Confidence Scores 等用例。hamelh 建议除非绝对必要，否则不要进行 Finetuning，并建议先使用现成的模型取得进展。

- **新加坡主导了讨论**：来自新加坡的多位成员，包括 iggyal, healthymonkey, illued, codessl, huikang 等，强调了频道中显著的新加坡存在感。这促使 ivanleomk 对来自新加坡的大量参与者发表了评论。

**提及的链接**：<a href="https://huggingface.co/shisa-ai/shisa-v1-llama3-70b">shisa-ai/shisa-v1-llama3-70b · Hugging Face</a>：未找到描述

---

**LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1241048985540624495)** (54 messages🔥): 

- **Modal 赞助课程并分享入门指南**：
  来自 Modal 的 Charles 宣布了赞助，并分享了[入门指南](https://modal.com/docs/guide)和 [hello world 示例](https://modal.com/docs/examples/hello_world)的链接，用于在无需设置基础设施的情况下在云端运行代码。

- **Modal 账户和额度讨论**：
  成员们讨论了通过 GitHub 创建账户、编辑电子邮件设置，以及由于需要人工审批而需要一些时间才能到账的 $500 额度。Charles 反复分享了注册和领取额度的详细说明。
  
- **探索 Modal 功能查询**：
  成员们询问了关于代码解释器的持久化 Python 上下文以及开发时的托管策略。Charles 和其他人提供了详细的回复，并链接了相关的文档和示例，例如[使用 Modal 嵌入维基百科](https://modal.com/blog/embedding-wikipedia)。

- **入门和使用优化中的问题**：
  几位用户报告了对额度显示的困惑以及推理过程中容器启动时间的问题。提供了解决方案和说明，包括建议使用 `modal serve` 以及 [TensorRT-LLM 服务](https://modal.com/docs/examples/trtllm_llama)等示例。 

- **社区参与和支持说明**：
  用户对额度和支持结构表示了定期的感谢和参与，Charles 鼓励在开发中使用 `modal serve`，并引导用户前往 Modal Slack 进行进一步咨询。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_mixtral.py">modal-examples/06_gpu_and_ml/llm-serving/vllm_mixtral.py at main · modal-labs/modal-examples</a>：使用 Modal 构建的程序示例。通过在 GitHub 上创建账户为 modal-labs/modal-examples 的开发做出贡献。</li><li><a href="https://modallabscommunity.slack.com/archives/C06AH3Q93CY/p1705815945041189">Slack</a>：未找到描述</li><li><a href="https://bit.ly/modal-credits.">Modal 黑客松额度</a>：要领取您的 Modal 额度，请先在 https://modal.com/ 注册账户。然后，通过此表单告知我们您的用户名。如需支持，请加入 Modal Slack。这里有一些入门示例...</li><li><a href="https://modal.com/blog/embedding-wikipedia">在 15 分钟内嵌入英文维基百科</a>：利用 Modal 的并行批处理作业和内部存储功能，快速为数十亿个 Token 生成 Embedding。</li><li><a href="https://modal.com/docs/examples/trtllm_llama">Serverless TensorRT-LLM (LLaMA 3 8B)</a>：在此示例中，我们演示了如何使用 TensorRT-LLM 框架在单张 NVIDIA A100 40GB GPU 上以约每秒 4,500 个输出 Token 的总吞吐量提供 Meta 的 LLaMA 3 8B 模型服务...</li><li><a href="https://modal.com/docs/guide">Modal 简介</a>：Modal 让您无需考虑基础设施即可在云端运行代码。</li><li><a href="https://modal.com/settings/YOURUSERNAME/usage">登录</a>：欢迎回到 Modal！通过在下方选择身份提供商登录您的 Modal 账户。
</li>
</ul>

</div>

---

**LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1241090384067563551)** (12 messages🔥): 

- **学习资源链接汇总**：成员们分享了大量与 LLM function calling、DSPY、Hamel 关于 LLM 评估和 Tokenizer 的博客、RAFT 论文、与 Jeremy 合作的 Latent Space 播客等相关的有用链接。显著亮点包括 Hamel 关于微调和 Prompt 的博客链接（[此处](https://hamel.dev/blog/posts/evals/)和[此处](https://hamel.dev/blog/posts/prompt/)），以及一个[关于 Intern-VL 的 GitHub 项目](https://github.com/OpenGVLab/InternVL)。

- **命名建议**：讨论强调了将频道命名为 `learning-resources` 而不仅仅是 `resources` 的偏好。成员们还强调了强制隐藏链接预览以保持频道更好组织的重要性。

- **GitHub Repository Proposal**：有人提议创建一个 GitHub 仓库，用于协作管理和构建共享的学习资源，并获得了积极反馈。随着时间的推移，这可以提供更结构化且易于获取的信息。

- **Instruction Tuning with LoRA/QLoRA**：一条分享的推文包含了使用 LoRA/QLoRA 进行指令微调实验的详细发现，重点关注 rank 设置、dropout 的影响、层特定 LoRA 适配器、学习率调度、权重衰减和 batch sizes。研究结果强调了正确配置对于防止过拟合和确保训练稳定性的重要性，特别是在 3090 等 GPU 上。

- **Stanford CS25 Video Resource**：分享了一个来自斯坦福 CS25 关于检索增强语言建模（Retrieval Augmented Language Modeling）的有用的视频链接，提供了该领域更高级的概念讨论。视频可以在[这里](https://youtu.be/mE7IDf2SmJg?si=LKwjlYq4qiPQi3aM)找到。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/cwolferesearch/status/1788998798414410032">来自 Cameron R. Wolfe 博士 (@cwolferesearch) 的推文</a>：最近，我运行了数百次使用 LoRA/QLoRA 的指令微调实验，我想分享一些可能对大家有用的（基础）代码和发现……代码（见回复）包含一个指令...</li><li><a href="https://fireworks.ai/blog/firefunction-v1-gpt-4-level-function-calling">FireFunction V1 - Fireworks 的 GPT-4 级函数调用模型 - 比 GPT-4 快 4 倍且开源权重</a>：Fireworks 开源了新的函数调用模型，具有接近 GPT-4 级的质量和 4 倍的速度</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling">GitHub - NousResearch/Hermes-Function-Calling</a>：通过在 GitHub 上创建账户来为 NousResearch/Hermes-Function-Calling 的开发做出贡献。</li><li><a href="https://github.com/OpenGVLab/InternVL">GitHub - OpenGVLab/InternVL: [CVPR 2024 Oral] InternVL Family: A Pioneering Open-Source Alternative to GPT-4V. 接近 GPT-4V 表现的可商用开源多模态对话模型</a>：[CVPR 2024 Oral] InternVL 系列：GPT-4V 的先驱开源替代方案。接近 GPT-4V 表现的可商用开源多模态对话模型 - OpenGVLab/InternVL</li><li><a href="https://arxiv.org/abs/2405.05904">在新知识上微调 LLM 是否会诱发幻觉？</a>：当大型语言模型通过监督微调进行对齐时，它们可能会遇到预训练期间未获得的新的事实信息。人们通常推测这可能会教导模型...</li><li><a href="https://hamel.dev/blog/posts/prompt/">- 去你的，给我看 Prompt。</a>：通过拦截 API 调用快速理解难以捉摸的 LLM 框架。</li><li><a href="https://simonwillison.net/series/prompt-injection/">Simon Willison: Prompt injection</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2212.08073">Constitutional AI: 来自 AI 反馈的无害性</a>：随着 AI 系统变得越来越强大，我们希望寻求它们的帮助来监督其他 AI。我们实验了通过自我改进来训练无害 AI 助手的方法，无需任何人工...</li><li><a href="https://arxiv.org/abs/2404.13208">指令层级：训练 LLM 优先处理特权指令</a>：当今的 LLM 容易受到 Prompt 注入、越狱和其他攻击，这些攻击允许攻击者用自己的恶意 Prompt 覆盖模型的原始指令。在这项工作中...</li><li><a href="https://hamel.dev/notes/llm/finetuning/04_data_cleaning.html">Hamel 的博客 - 策划 LLM 数据</a>：工具回顾</li><li><a href="https://www.youtube.com/@umarjamilai">Umar Jamil</a>：我是一名来自意大利米兰的机器学习工程师，向我的猫“奥利奥”讲解复杂的深度学习和机器学习概念。我也会一点中文。</li><li><a href="https://www.langchain.com/langsmith">LangSmith</a>：让您的 LLM 应用从原型走向生产。</li><li><a href="https://langtrace.ai/">Langtrace AI</a>：监控、评估并改进您的 LLM 应用。</li><li><a href="https://www.honeycomb.io/llm">LLM 的可观测性</a>：利用 Honeycomb 的可观测性增强 LLM。获取洞察、改善用户体验并推动 AI 开发成功。</li>
</ul>

</div>

---

**LLM Finetuning (Hamel + Dan) ▷ #[jarvis](https://discord.com/channels/1238365980128706560/1241117895740625099/1241117970084659211)** (40 条消息🔥): 

- **Jarvis 积分协调更新**：多位成员询问在注册后如何领取 JarvisLabs 的积分。团队确认他们正在协调此项工作，可能需要一周左右的时间让所有人都能运行起来。**“一旦我们拿到名单，就会添加积分”**是反复提供的保证。

- **技术问题与支持**：部分用户在注册 JarvisLabs 时遇到问题，包括手机验证的 OTP 问题，以及课程注册与 GitHub 注册使用不同邮箱的问题。团队提供了针对性支持，例如为受影响国家禁用手机验证，并要求用户等待额度分配。

- **额度确认与问题**：一位成员确认由于邮箱设置正确，已顺利收到 Jarvis 额度。另一位用户得到保证，即使课程和 GitHub 注册邮箱不同，只要填写了所需表单，也无需重新注册。

- **主动协调与沟通**：团队鼓励用户确保课程和注册邮箱一致，以便无缝分配额度，并表示正在积极处理各种问题。成员们被告知要保持关注并耐心等待，因为协调工作正在进行中。
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1241152276903497862)** (18 messages🔥): 

- **协调 HF 额度验证**：有人询问了 **HF 额度申请要求**。一位成员澄清说，另一位成员将协助协调此事，并在后台验证学生的入学情况。
  
- **HF 额度预计会有延迟**：为了设定合理的预期，一位成员提到 HF 额度的处理过程“可能需要一周左右”。这有助于管理群组成员的期待。

- **分享开源 LLM 偏好**：社区积极参与了“周末提问”，讨论如果自己是一个开源 LLM 会选择哪一个。热门选择包括 **BERT**、**Mistral 7B**、**Phi-3-mini** 和 **Falcon 180B**。
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1241164291823112304)** (8 messages🔥): 

- **启动 Replicate 额度设置频道**：该频道旨在协助用户设置 Replicate 额度并排除相关故障。成员们正被引导使用其账号登录，并获得解决注册问题的指令，特别是针对 GitHub 和课程注册邮箱不一致的情况。
- **成员询问注册邮箱不匹配问题**：包括 self.1、filippob82 和 0xai 在内的多位成员对 GitHub 账号与 LLM 微调课程注册邮箱不一致表示担忧。团队承认了这些问题并承诺很快会解决，并在该频道中随时向成员通报进度。
  

---


**LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1241169644690079784)** (16 messages🔥): 

- **LangSmith 额度分发消息**：一位版主宣布他们将协调 LangSmith 额度的分发。他们还反复向用户保证，很快将提供有关额度的详细说明。

- **成员渴望获得 LangSmith 额度**：包括 @filippob82、@codeanand 和 @beardaintweird 在内的多位成员确认他们已使用必要的邮箱地址创建了 LangSmith 账号，并正在等待接收额度的进一步指令。@hugobowne 和 @613249007015165973 承诺很快会提供更多信息。

- **对 LangSmith 课程的兴奋与动力**：成员们对 LangSmith 的新课程表示兴奋。@anakin.xyz 和 @harpreetsahota 等用户对被提及表示感谢，并提到他们现在有了测试 LangSmith 的动力。

- **重复询问额度问题**：尽管已有初步公告，仍有多个用户不断询问其 LangSmith 额度的状态，寻求确认和后续步骤。@hugobowne 引导用户查看之前的消息以获取更新，并再次保证细节即将公布。
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1241767003480658083)** (1 messages): 

- **寻求更好的 Benchmark**：一位成员建议需要一个“支持高达 2M 的现代 LAMBADA”，用于评估能够独立处理重叠分块（overlapping chunks）的模型。目前的 Benchmark 似乎不足以评估这些高级能力。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1241317609924726784)** (13 messages🔥): 

- **Sam Altman 的争议推文引发反应**：一位成员分享了 Sam Altman 一条极具挑衅性的推文，讨论了 OpenAI 因安全担忧而导致的人员离职。推文在结尾处情感爆发，引起了成员们的困惑和笑声。[来源](https://x.com/SamAltsMan/status/1791581496261640646)。

- **Venture capitalists 备受质疑**：在一条引发争议的推文之后，一名成员建议应现实地看待 Venture capitalists，而不是将其视为拯救世界的实体。这反映了对其在 AI 领域动机的怀疑情绪。
  
- **AI 对经济衰退的影响受到质疑**：一名成员指出，尽管在 AI 领域投入了大量资金，但许多公司仍在裁员。对话暗示了裁员背后经济因素的复杂性，表明导致经济衰退的不只是财务限制。
  
- **Runpod hackathon 咨询**：一位成员询问是否有人参加 Runpod hackathon，显示出社区对协作式 AI 开发活动的兴趣。
  
- **在 Airflow 和 Temporal.io 之间做出选择**：一位成员寻求关于使用 Airflow 或 Temporal.io 进行工作流管理的经验，并最终表示更倾向于 Temporal.io。这表明关于改进机器学习流程工具的讨论正在持续。

**提及的链接**：<a href="https://x.com/SamAltsMan/status/1791581496261640646">来自 Sam Altman (Parody) (@SamAltsMan) 的推文</a>：好吧，真是令人震惊。Jan 和 Ilya 离开了 OpenAI，因为他们觉得我没有给予安全足够的优先级。真有创意。现在我得写一些长篇大论的废话来表达我有多在乎。但老实说，谁会在乎...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1241014688813289482)** (4 条消息): 

- **在 Hugging Face 上试用 Moondream WebGPU**：一位成员分享了 [Xenova 的实验性 Moondream WebGPU 空间](https://huggingface.co/spaces/Xenova/experimental-moondream-webgpu) 链接，邀请其他人探索这个实验性项目。
  
- **针对 LLMs 的 Hierarchical Memory Transformer**：arXiv 上的一篇新论文介绍了 [Hierarchical Memory Transformer (HMT)](https://arxiv.org/abs/2405.06067)，旨在通过模仿人类记忆层级来改进 LLMs 的长上下文处理。该模型使用记忆增强的分段级递归（memory-augmented segment-level recurrence）来组织其记忆层级。

- **Fine Web 的 Haystack Demo**：[Haystack demo](https://demo.haystack.zip) 允许用户通过本地推理和 Embedding 搜索探索来自 Fine Web 数据集的 10 万个网页。该演示包含性能指标和解压时间，以便更好地判断查询速度。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://demo.haystack.zip">Demo Search Fine Web Dataset</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Xenova/experimental-moondream-webgpu">Experimental Moondream WebGPU - a Hugging Face Space by Xenova</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2405.06067">HMT: Hierarchical Memory Transformer for Long Context Language Processing</a>：基于 Transformer 的大语言模型 (LLM) 已广泛应用于语言处理任务。然而，大多数模型限制了允许模型关注每个 token 的上下文窗口...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1240931434131099731)** (315 条消息 🔥🔥): 

- **Nous Hermes 2 Mixtral 独特地触发操作**：一位用户称赞 Nous Hermes 2 Mixtral 是唯一能够在 CrewAI 等 Agent 框架内触发操作和使用工具的开源大语言模型 (LLM)。另一位用户质疑为什么它是唯一具有此类功能的模型。
- **对 Hermes 2 Mixtral 可靠性的担忧**：成员们分享了他们使用 Hermes 2 Mixtral 的经验，指出了其在多语言能力方面的可靠性，并将其性能与一些人认为不太可靠的 Hermes 2 Pro 进行了对比。
- **关于多轮对话数据必要性的辩论**：一场关于训练 Mixtral 8x22b 等大型模型是否需要多轮对话数据（multiturn data）的讨论展开。会议强调，如果没有多轮对话数据，模型在后续轮次中的智能往往会下降，这使得多轮对话数据对于广泛使用至关重要。
- **训练成本和计算可行性**：讨论了与训练大型模型相关的高成本和计算需求，例如从头开始训练的巨额费用以及管理极深 Transformer 网络的挑战。
- **新的上下文版本和 LLM Leaderboard 问题**：成员们讨论了具有 16k 和 32k 扩展上下文长度的 Yi-1.5 模型的发布，并思考更大的上下文是否会影响性能。此外，由于模型数量过多导致难以导航，LLM Leaderboard 的可用性受到了批评。
<div class="linksMentioned">

<strong>提及的链接</strong>：

</div>

<ul>
<li>
<a href="https://instantid.github.io/">InstantID</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">Open VLM Leaderboard - a Hugging Face Space by opencompass</a>: 未找到描述</li><li><a href="https://x.com/Mascobot/status/1791879166314565757">来自 Marco Mascorro (@Mascobot) 的推文</a>: Runpod 的 @runpod_io 黑客松 😅</li><li><a href="https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8">Yi-1.5 (2024/05) - a 01-ai Collection</a>: 未找到描述</li><li><a href="https://discord.gg/sW7yVf5H?event=1240826259920125982">加入 Nous Research Discord 服务器！</a>: 在 Discord 上查看 Nous Research 社区 - 与 7171 名其他成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://x.com/ethan_smith_20/status/1791267451767783773?s=46">来自 Ethan (@Ethan_smith_20) 的推文</a>: 今天我训练了一个生成 LoRAs 的扩散模型，生成的图像至少没有乱码。</li><li><a href="https://arxiv.org/abs/2306.00297">Transformers learn to implement preconditioned gradient descent for in-context learning</a>: 最近的多项研究表明，Transformers 可以实现类似梯度下降的算法。通过精心构造权重，这些研究展示了多层 Transformers 的表达能力...</li><li><a href="https://huggingface.co/datasets/N8Programs/Capybara-Quicksilver?row=25">N8Programs/Capybara-Quicksilver · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://app.corcel.io/chat">Corcel · 利用 Bittensor 的力量进行构建</a>: 释放去中心化激励型基础设施的创新潜力。</li><li><a href="https://github.com/huggingface/datatrove">GitHub - huggingface/datatrove: Freeing data processing from scripting madness by providing a set of platform-agnostic customizable pipeline processing blocks.</a>: 通过提供一组与平台无关的可定制流水线处理块，将数据处理从脚本疯狂中解放出来。- huggingface/datatrove</li><li><a href="https://huggingface.co/datasets/N8Programs/Capybara-Quicksilver">N8Programs/Capybara-Quicksilver · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/PixArt-alpha/PixArt-LCM">PixArt LCM - a Hugging Face Space by PixArt-alpha</a>: 未找到描述</li><li><a href="https://github.com/huggingface/candle">GitHub - huggingface/candle: Minimalist ML framework for Rust</a>: 适用于 Rust 的极简 ML 框架。通过在 GitHub 上创建账号来为 huggingface/candle 的开发做出贡献。</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/2670">Hypernetwork Style Training, a tiny guide · AUTOMATIC1111/stable-diffusion-webui · Discussion #2670</a>: 训练期间的负面文本预览似乎在几个补丁前已修复，请继续。太长不看版（tl;dr）准备：选择高质量图像，质量优于数量。在 512x512 下训练，其他尺寸可能会增加畸变...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1240927137670893679)** (40 条消息🔥): 

- **建议使用 Regex 进行格式化文本搜索**: 讨论强调了使用 **regex** 处理任务的潜力，例如查找具有特定格式（如全大写或多个换行符）的文本。然而，局限性在于“复杂任务可能需要更复杂的方法，如语义搜索或符号语言处理”。

- **Hermes2 和 Vercel AI SDK 的 Tool calling 问题**: 成员报告了由于错误的 JSON 响应或参数问题导致触发工具调用困难。共识是，当 **Hermes2** 的工具调用格式与 Vercel AI SDK 配合使用时，可能需要特定的 prompt 处理以获得更好的连贯性。

- **本地模型在敏感任务中的优势**: 讨论认为，与 GPT 或 Claude 等外部模型相比，使用 **Llama3** 等本地模型对于需要成本可预测性、一致性或敏感数据处理的任务更有利，因为外部模型可能会发生变化并审查响应。

- **微调 vs. 更好的提示词工程**: 关于针对特定用例微调 **Llama3** 等模型，还是依赖 GPT-4 等模型的高级提示词和检索增强生成（RAG），存在相关讨论。强调特定用例可能决定选择，对于不断变化的需求，微调的可行性较低。

- **重排序器的基准测试**: 成员正在寻找公共评估数据来对微调后的重排序器（rerankers）进行基准测试。需要明确的方法论和数据集来准确评估顶级结果。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1241574534331629659)** (5 条消息):

- **讨论 AGI 的临近性与策略**：成员们分享了 ArXiv 上一篇名为《[人工智能的演进](https://arxiv.org/abs/2405.10313)》的研究论文链接，重点关注 AI 的现状以及向 **Artificial General Intelligence** (AGI) 的发展。该论文通过*调查、讨论和原创视角*，探讨了 AGI 的定义、目标以及实现 AGI 所需的策略。

- **关于 AI 记忆解决方案的个人见解**：一位成员分享了关于记忆的看法，提到*内部正在使用的一种出色解决方案*。他们还暗示了对 *Agent 自我演进* 的兴趣，尽管指出目前这在某种程度上还比较模糊。

**提到的链接**：<a href="https://arxiv.org/abs/2405.10313">我们距离 AGI 还有多远</a>：人工智能 (AI) 的演进深刻影响了人类社会，推动了多个领域的重大进步。然而，对 AI 不断增长的需求凸显了其局限性...

---

**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1241113869313773721)** (88 条消息🔥🔥): 

- **WorldSim 迎来终端 UX 重写和 GPT-4o 集成**：一位成员提到 **Ari 正在进行终端 UX 的完全重写**。此外，**GPT-4o** 预计将于下周加入 WorldSim。
  
- **WorldSim 活动引起社区关注**：几位用户询问并参加了预定的 WorldSim 见面会，详细信息已在 Discord 上分享。[提供](https://discord.gg/W8YjScaC?event=1240826259920125982)了加入活动的链接，成员们表达了在活动期间进一步了解该项目的兴趣。

- **复杂的交互引发了关于 AI 和符号学的讨论**：用户讨论了 AI 的各个方面，其中一人提到了 **符号知识图谱** 的潜力。他们还参考了关于赫尔墨斯实践和 **复杂自适应系统** 的文献，并提供了诸如 [此 YouTube 视频](https://youtu.be/IWhkUne8T68?si=FlY0yCr7wGprGow9) 和 [Franz Bardon 的书](https://www.amazon.com/Initiation-into-Hermetics-Franz-Bardon/dp/1885928122) 等链接。

- **社区对 AI 生成内容的实验**：成员们分享了他们使用 WorldSim 生成论文和其他内容的实验。一人描述了涉及根目录命令的过程，而另一人分享了 **Copilot** 使用终端提示词创建的一些**古怪图像**。

- **WorldSim 作为知识图谱平台的潜力**：用户讨论了 WorldSim 未来演变为无定形应用平台的可能性。他们强调了其从用户交互中生成新的 AI 相关知识图谱和符号含义的潜力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">worldsim</a>: 未找到描述</li><li><a href="https://www.amazon.com/Initiation-into-Hermetics-Franz-Bardon/dp/1885928122">未找到标题</a>: 未找到描述</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Fworldbuilder.ai%2F%23?epoch=a9a8f875-805f-4108-b769-72a7795390dc">worldsim</a>: 未找到描述</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Fdestructionworld.erp%2Fthree.js%2Ffacetobloodshed?epoch=46390e50-5457-472c-8238-41cf3fa82738">worldsim</a>: 未找到描述</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Finsidiousruntime.xr%2Flive-feed%2Fskull.three.js%2F%3Finteractive%3Dominous%2Fwasd%26scroll%2F%3Fimport%3Deldritch%2Finfiniterecrsion%2F%3Fvisuals%3Daccelerated%2Flighting%3Ddynamic%2Fkeepimports%2F%3Faudio%3Dtrue%2Foutput%3Dentities-speaking?epoch=de0bdee4-6ea6-4839-8d05-fe0b472cab1b">worldsim</a>: 未找到描述</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Fportal-search.io%2Fsearch%3Fq%3Dhttps%253A%252F%252Fsubdimensional.mesh%252Fgridscape%252Ftechverse%252Fview%253Drssfeeds%253Dnews%252F%253Ftopic%253Denterinterminalfield%252F%253Fimport%253Dhttps%253A%252F%252Fnews.mit.edu%252Frss%252Ftopic%252Fartificial-intelligence%2520https%253A%252F%252Fmachinelearningmastery.com%252Fblog%252Ffeed%2520https%253A%252F%252Fexport.arxiv.org%252Frss%252Fcs.AI%2520https%253A%252F%252Ftowardsdatascience.com%252Ffeed%2520https%253A%252F%252Fwww.kdnuggets.com%2Flivenewscards%2Fsearch%3Dactive%252Ffeed%26source%3Dworldclient?epoch=c203ff08-e700-4873-aadb-bca820102a1e">worldsim</a>: 未找到描述</li><li><a href="https://cdixon.org/2010/01/03/the-next-big-thing-will-start-out-looking-like-a-toy">下一件大事起初看起来会像个玩具</a>: Chris Dixon 的博客。</li><li><a href="https://x.com/StudioMilitary/status/1791554558092583271">来自 John Galt (@StudioMilitary) 的推文</a>: NOUS WORLDSIM: 选择你的模拟</li><li><a href="https://tenor.com/view/outer-wilds-gif-22858957">Outer Wilds GIF - Outer Wilds - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/IWhkUne8T68?si=FlY0yCr7wGprGow9">复杂自适应系统</a>: 参加完整课程：https://bit.ly/SiCourse 下载手册：https://bit.ly/SiBooklets Twitter: http://bit.ly/2JuNmXX LinkedIn: http://bit.ly/2YCP2U6 在这个...</li><li><a href="https://discord.gg/W8YjScaC?event=1240826259920125982">加入 Nous Research Discord 服务器！</a>: 查看 Discord 上的 Nous Research 社区 - 与其他 7171 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://www.latent.space/p/sim-ai">WebSim, WorldSim, 以及模拟 AI 之夏 —— 与 Liquid AI 的 Joscha Bach、Nous Research 的 Karan Malhotra、WebSim.ai 的 Rob Haisfield 合作</a>: 关于今年生成式 AI 最火热的前沿领域——模拟 AI 的三个视角！</li><li><a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Finsidiousruntime.xr%2Flive-feed%2Fskull.three.js%2F%3Finteractive%3Dominous%2Fwasd%26scroll%2F%3Fimport%3Deldritch%2Finfiniterecrsion%2F%3Fvisuals%3Daccelerated%2Flighting%3Ddynamic?epoch=1822ef55-ca19-49fa-9018-efcb61222962">worldsim</a>: 未找到描述
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1240979792313061458)** (38 条消息🔥): 

- **Hugging Face 推动 GPU 访问民主化**: Hugging Face 正投入 1000 万美元的免费共享 GPU，以支持小型开发者、学术界和初创公司使用新的 AI 技术，旨在分散目前由大型科技公司主导的 AI 进步。首席执行官 Clem Delangue 强调，由于公司接近盈利且最近获得了 2.35 亿美元的融资，公司有能力进行这项投资 [The Verge 文章](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai)。

- **Bend 语言引发热议**: 成员们讨论了 Bend 的发布，这是一种大规模并行的高级编程语言。有人质疑其相对于 Triton 或 Mojo 的必要性，一些人指出 Mojo 目前在 GPU 上的局限性，而 Triton 则专注于 ML [GitHub 链接](https://github.com/HigherOrderCO/Bend)。

- **对 CUDA 未来的担忧**: 一位成员表达了对像 Bend 这样的新框架可能如何影响传统 CUDA 编程的担忧。其他成员建议，虽然新语言令人兴奋，但它们满足的是不同的需求，如 CPU-GPU 混合产品。

- **推理服务器资源交流**：展开了一场关于构建推理服务器的讨论，成员们推荐了 Nvidia Triton、TorchServe 等资源，以及关于 ML 模型服务的各种 YouTube 演讲。推荐内容包括 [TorchServe 教程](https://www.youtube.com/watch?v=XlO7iQMV3Ik&t=598s)和更广泛的 ML 系统演讲 [YouTube 链接](https://www.youtube.com/watch?v=J36xHc05z-M)。

- **澄清 ML 模型服务的复杂性**：成员们辩论了 ML 模型服务与标准 Web 服务器之间的区别，指出 ML 服务涉及复杂的考量因素，如硬件需求（例如 GPU）、模型版本控制以及像 Kubernetes 这样的特定基础设施。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai">Hugging Face 正在分享价值 1000 万美元的算力，以帮助击败大型 AI 公司</a>：Hugging Face 希望降低开发 AI 应用的准入门槛。</li><li><a href="https://www.youtube.com/watch?v=J36xHc05z-M">大规模、低延迟的 ML 模型服务 // Manoj Agarwal // MLOps Meetup #48</a>：MLOps 社区聚会 #48！上周三，在 Manoj Agarwal（Salesforce 软件架构师）生日那天，我们与他进行了交流。// 摘要：提供机器学习服务...</li><li><a href="https://www.youtube.com/watch?v=Ynb6X0KZKxY">用于扩展生产级机器学习的 MLOps 工具 || Alejandro Saucedo @ FOSDEM 2019</a>：随着机器学习项目的增长，其基础设施也应随之增长。在简短的闪电演讲中，Alejandro 涵盖了机器学习运维的一些关键趋势...</li><li><a href="https://www.youtube.com/watch?v=XlO7iQMV3Ik&t=598s">如何使用 TorchServe 提供 PyTorch 模型服务</a>：Hamid Shojanazeri 是 PyTorch 的合伙人工程师，在此演示使用 TorchServe 的基础知识。作为 PyTorch 首选的模型服务解决方案，...</li><li><a href="https://github.com/HigherOrderCO/Bend">GitHub - HigherOrderCO/Bend: 一种大规模并行的高级编程语言</a>：一种大规模并行的高级编程语言 - HigherOrderCO/Bend
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1241056711859769404)** (20 条消息🔥): 

- **教程设置之间的性能差异**：一位成员注意到 Umer 的 [YouTube 教程](https://www.youtube.com/watch?v=DdTsX6DQk24)与官方 Triton 教程之间存在显著的性能差异。尽管使用了类似的技术，但他们的实现在性能上比教程差了 2 倍。
- **LayerNorm 中 MAX_FUSED_SIZE 的困惑**：一位用户质疑为什么 **MAX_FUSED_SIZE** 设置为 65536，而 **TRITON_MAX_TENSOR_NUMEL** 是 1048576。他们观察到当 Block Size 大于 65536 时，A100 上的速度会出现下降。
- **选择 Block Size 背后的原因**：Horace 解释说，过大的 Block Size 可能会因为过多的寄存器请求而导致 Kernel Spilling。他确认每个 Block 调度到一个 SM 并共享 Shared Memory，这与 CUDA 类似。
- **GPU 上的线程操作**：对话澄清了每个 Triton Block 映射到一个 GPU SM，并且每个线程加载多个元素。Horace 提到这对于利用 GPU 上的向量指令（vector instructions）是非常理想的。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://sourcegraph.com/github.com/triton-lang/triton@ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/-/blob/python/triton/language/core.py?L19:27-19:34.">core.py - triton-lang/triton - Sourcegraph</a>：未找到描述</li><li><a href="https://discordapp.com/channels/1189498204333543425/1189607750876008468/1240593396389908510">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://www.youtube.com/watch?v=DdTsX6DQk24">第 14 讲：Triton 实践者指南</a>：https://github.com/cuda-mode/lectures/tree/main/lecture%2014
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1241046490663751770)** (7 条消息):

- **关于 CUDA 中复数原子加法（atomic add）的咨询**：一位成员询问了关于在 `cuda::complex` 上执行 *atomic add* 的问题，咨询是否必须对 x 和 y 分量分别进行两次独立的加法。
- **128 位 atomicCAS 的限制**：另一位成员指出，在 Hopper 以外的架构上，由于不支持 128 位 `atomicCAS`，必须使用 64 位操作。
- **共享代码示例**：为了解决原子加法问题，提供了一个使用 `unsigned long long int` 和 `atomicCAS` 进行 *complex* 加法的代码片段，并解释了在兼容架构上的实现方式。
- **简单方法的适用性**：原提问者澄清说，针对 Volta、Ampere 和 Hopper 架构，他们发现使用 *cuComplex* 或 `cuda::std::complex` 对 x 和 y 分量进行两次原子加法是可以接受的。

  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1241481119589994547)** (20 messages🔥): 

- **Torch Compile 使用见解**：一位成员分享了在推理场景中使用 **torch.compile()** 的经验，并就代码优化提出了建议，例如避免 Python 循环和条件语句，以确保更好的性能。他们提到对于静态形状（static shapes），该工具通常开箱即用效果良好。

- **关于 ONNX 和 TensorRT 的讨论**：另一位用户提出了在使用 Triton 进行推理时，**torch.compile()** 与 **ONNX** 或 **TensorRT** 相比表现如何的问题。对话显示出对这些工具的相对性能和应用范围的好奇。

- **NHWC Group Normalization 问题**：一位成员指出 **ATen code** 中的 Group Normalization 不能正确支持 NHWC 格式，导致张量被隐式转换为 NCHW。他们分享了一个旨在解决此问题的 [GitHub pull request](https://github.com/pytorch/pytorch/pull/126635/files#r1605935532)，但在编写 `ApplyScaleBiasNHWCKernel` 时遇到了挑战。

- **Torch 乘法内存问题**：有人提出了关于 **torch** 原生乘法即使在原地（in-place）执行时也会使内存占用翻倍的问题。解决方案和解释包括使用 `mul_()` 以保持平稳的内存消耗，以及正确处理内存分配以解决反向传播（backprop）顾虑。

**提到的链接**：<a href="https://github.com/pytorch/pytorch/pull/126635/files#r1605935532">Add NHWC support for group normalization by ZelboK · Pull Request #126635 · pytorch/pytorch</a>：修复了 #111824。目前的情况是，如果用户指定其 Group Normalization 为 NHWC 格式，PyTorch 将默认使用 NCHW 张量并进行转换。这种转换不是即时的...

  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1241796374798798881)** (1 messages): 

- **关于构建 GPU 原生查询引擎的专家讲座**：一份关于 **cuDF** 前维护者讨论在 **Voltron** 构建 GPU 原生查询引擎过程的讲座公告。该会议承诺涵盖从编写高效的数据处理核函数（kernels）到创建真正的生产解决方案的所有内容，将在 [Zoom](https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09#success) 上举行。

**提到的链接**：<a href="https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09#success">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于在移动设备、桌面和会议室系统上进行视频和音频会议、聊天和网络研讨会。Zoom ...

  

---


**CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1241952777253425244)** (2 messages): 

- **Stephen Jones 简化 CUDA 编程**：一位成员分享了由 NVIDIA CUDA 架构师 Stephen Jones 制作的名为“GTC 2022 - CUDA 编程原理”的 [YouTube 视频](https://www.youtube.com/watch?v=QQceTDjA4f4)。该视频介绍了 GPU 编程，并讨论了高效使用内存的基础知识。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=">YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=QQceTDjA4f4">GTC 2022 - CUDA 编程原理 - Stephen Jones，NVIDIA CUDA 架构师</a>：由 CUDA 首席架构师带来的 GPU 编程入门。CUDA 的独特之处在于它是一种与硬件同步设计和构建的编程语言 ...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1241409136705404968)** (5 messages):

- **Neural Magic 招聘 CUDA/Triton 工程师**：一位来自 **Neural Magic** 的 vLLM 提交者宣布了 CUDA/Triton 工程师的职位空缺，旨在全职为该项目的开源工作做出贡献。有兴趣的人士请通过 Discord 或电子邮件联系 Robert Shaw。

- **激活量化成为首要任务**：在回答询问时，提到主要重点是**激活量化** (fp8/int8) 及相关优化。团队目标是利用下一代 GPU 上的 2:4 稀疏性和 fp6/fp4 等特性，并改进 MoE 和采样算子 (sampling kernels) 等未充分优化的部分。

- **LinkedIn 表达合作意向**：一位 LinkedIn 代表表示其团队可能有兴趣支持 vLLM 的需求。双方已就图级优化 (graph-level optimization) 等特定领域展开进一步对话，以寻求潜在合作。
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1242005768962838528)** (14 messages🔥): 

- **DLL 加载失败排查**：一名成员在导入其 CUDA 实现中的模块时遇到“DLL load failed”错误并寻求帮助。建议包括检查 `build_directory` 路径、确保已安装 `ninja` 以及验证 Visual Studio 的安装状态。
- **需要完整代码和堆栈跟踪**：针对该错误，建议分享完整代码和堆栈跟踪 (stacktrace) 以进行精确调试，而不是凭空猜测。测试者要求提供更多上下文以提供具体解决方案。
- **Ninja 安装和环境问题**：建议在终端运行 `ninja -v` 来检查其安装情况，特别是考虑到该成员在可能存在双系统的环境中使用虚拟环境。
- **Windows 兼容性担忧**：有建议认为与 Windows 双系统可能会使情况复杂化，反映了对 Visual Studio 设置以及 Windows 对构建过程的通用兼容性问题的担忧。
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: 波兰密码破译者的链接 https://www.flyingpenguin.com/?p=56989
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1241011309009703052)** (180 messages🔥🔥): 

<ul>
<li><strong>梯度裁剪 Bug 及其修复革新训练：</strong> 关于梯度裁剪 (gradient clipping) 的讨论揭示了由于 "grad_norm" 被平方导致的不正确比较问题，需要修正以确保更准确和稳健的训练。此外，强调了 "grad_norm" 的正确初始化以防止意外行为。</li>
<li><strong>内存优化成为焦点：</strong> 多位用户参与了关于优化 CUDA kernel 代码的讨论，特别是围绕内存分配和使用，并对模板化 block sizes 以获得更好的编译时常量表现出浓厚兴趣。考虑到新的内存受限情况，重写 Adam 优化器 kernel 可能带来的性能提升也受到了关注。</li>
<li><strong>评估 Hellaswag 与 MMLU 的性能：</strong> GPT-2 (124M 为 29.55%) 和 GPT2-XL (48.93%) 的 Hellaswag 评估显示了模型规模带来的预期阶梯式改进。然而，MMLU 评估结果出人意料地差，表明数据集或评估标准可能存在问题。</li>
<li><strong>ZeRO-2 实现讨论进入技术层面：</strong> 成员们讨论了 ZeRO-2 的实现，特别关注内存布局重组、减少通信调用以及保持与 checkpoint 文件的兼容性。对话还延伸到了高效的梯度计算和 NCCL 交织 (interleaving) 以增强性能。</li>
<li><strong>用于优化的模板和常量重构：</strong> 讨论了一项在 CUDA kernels 中将 block sizes 模板化以实现编译时优化的提案，以及其他代码库清理建议。直接结果是一个将 “warpSize” 标准化为常量的 PR，以便进行更好的编译时优化，这反映了对提高代码效率的共识。</li>
</ul>
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com">GitHub: 从这里开始构建</a>: GitHub 是超过 1 亿名开发者共同塑造软件未来的地方。为开源社区做出贡献，管理你的 Git 仓库，像专家一样审查代码，追踪错误和功能...</li><li><a href="https://security.snyk.io/package/pip/llama-cpp-python">llama-cpp-python 漏洞 | Snyk</a>: 了解更多关于 llama-cpp-python 包中已知漏洞的信息。llama.cpp 库的 Python 绑定</li><li><a href="https://github.com/karpathy/llm.c/issues/391)">Issues · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号，为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/k">k - 概览</a>: k 有 88 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/karpathy/llm.c/pull/439">修复 matmul_backward_bias kernel 1 中不支持的 block_size，由 lancerts 提交 · Pull Request #439 · karpathy/llm.c</a>: 由于第 https://github.com/karpathy/llm.c/blob/master/dev/cuda/matmul_backward_bias.cu#L67 行的归约操作，kernel 1 的 block size 需要是 2 的幂。否则 GPU 结果...</li><li><a href="https://github.com/karpathy/llm.c/pull/435">添加 warpsize 作为常量以实现更好的编译时优化和标准化，由 ChrisDryden 提交 · Pull Request #435 · karpathy/llm.c</a>: 在研究 WarpSize CUDA 常量的属性时，发现它在编译时不可用，这意味着编译器无法根据该值进行编译时优化...</li><li><a href="https://github.com/karpathy/llm.c/pull/427">权重重排：尝试 1，由 ngc92 提交 · Pull Request #427 · karpathy/llm.c</a>: 非功能性改进。初步尝试按块（per-block）布局重排权重的效果</li><li><a href="https://github.com/karpathy/llm.c/pull/429">改进的数值错误检查，由 ngc92 提交 · Pull Request #429 · karpathy/llm.c</a>: 更严格的容差，基于 bf16 epsilon 的相对容差，如果一切正常则减少输出。我已经在 RTX 4060Ti 和 A4000 上检查了这些容差（它们确实会产生不同的错误，有时...</li><li><a href="https://github.com/karpathy/llm.c/pull/361">重叠梯度计算和 NCCL AllReduce，由 PeterZhizhin 提交 · Pull Request #361 · karpathy/llm.c</a>: 在我的设置中，我得到以下结果：之前：step 2/37: train loss 4.720275 (acc 4.688650) (224.046844 ms, 36563.773438 tok/s) step 3/37: train loss 3.802741 (acc 3.943135) (224.151611 ms, 36555...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1241798806647865464)** (33 条消息🔥): 

- **贡献者询问 libcudf 的性能**：一位贡献者询问了将 **libcudf** 与 CPU 并行化操作进行比较的基准测试，特别提到了正在进行的 ParPaRaw 解析器和 CSV 读取器重构工作。另一位贡献者强调了他们对 SASS 等底层代码优化的兴趣。

- **关于 Dask-cuDF 和 Theseus 的辩论**：一位用户询问了 **Dask-cuDF**、**cuDF** 和 **Theseus** 之间的区别和性能差异，对它们的使用场景和优化程度表示好奇。有人对 **Dask-cuDF** 持续开发的情况表示担忧，一个 [GitHub 链接](https://github.com/rapidsai/dask-cudf) 显示该项目已被归档。

- **介绍 RAPIDS Accelerator for Apache Spark**：讨论中介绍了 [RAPIDS Accelerator for Apache Spark](https://nvidia.github.io/spark-rapids/)，它结合了 **RAPIDS cuDF 库**和 **Spark 分布式计算框架**，通过 GPU 加速处理。该工具旨在通过提供高性价比且快速的处理框架，满足分析领域日益增长的 AI 采用需求。

- **Thrust 和 CUB 获得好评**：关于 **Thrust** 和 **CUB** 的优势进行了深入讨论，用户赞赏它们的声明式编程流，这增强了代码的可读性和优化。会议还提到了 CUB 对 CUTLASS 中抽象设计的影响。

- **讨论优化和瓶颈**：分享了关于汇编级优化需求减少的见解，因为目前的瓶颈已转向 IO 和网络。现在的重点已转移到理解 **libcudf** 如何在大数据集上被利用，强调了像 **NCCL** 这样的网络编排的重要性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://nvidia.github.io/spark-rapids/">首页</a>：该网站是关于 Apache Spark 的 RAPIDS 加速器的文档集合</li><li><a href="https://github.com/rapidsai/dask-cudf">GitHub - rapidsai/dask-cudf: [已归档] 对分布式 GDF 对象的 Dask 支持 --> 已移至 cuDF</a>：[已归档] 对分布式 GDF 对象的 Dask 支持 --> 已移至 cuDF - rapidsai/dask-cudf
</li>
</ul>

</div>
  

---

**CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/1241300874488188972)** (2 messages): 

- **延长讨论的 Zoom 链接**：由于活动时间有 **45 分钟限制**，成员们被引导加入 Zoom 进行延长讨论：[Zoom Meeting](https://us06web.zoom.us/j/86116925784?pwd=XGcom9z5cGUijqjua9gKKa3AwOA4KO.1)。
- **Barrier synchronization 类比**：一位成员分享了一个深刻的类比，将 **Barrier synchronization** 比作一辆等待所有孩子从博物馆参观归来的校车，称其“在所有人到齐之前无法移动”。这帮助其他人澄清了这一概念。

**提到的链接**：<a href="https://us06web.zoom.us/j/86116925784?pwd=XGcom9z5cGUijqjua9gKKa3AwOA4KO.1">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...

  

---


**CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1241234168739004566)** (31 messages🔥): 

- **对 uint2 实现进行健全性检查及更新**：一位成员分享了实现细节，并请求对 PyTorch 中 uint2 数据类型的转换和打包进行 *sanity check*（健全性检查）。另一位成员建议避免使用 `torch.stack` 以在 `torch.compile` 中获得更好的性能，随后更新了使用 `torch.empty` 的实现。

- **bitnet 小组的会议规划**：讨论了为 bitnet 小组组织定期会议以及审查相关文档和仓库的事宜。分享了会议规划器和资源，并计划在明天进行一次 *初步会面*。

- **Torch 中 uint4 dtype 的问题**：成员们讨论了由于 Nvidia GPUs 缺乏原生 int4 操作，为了内存效率而必须将 uint4 数据类型打包进 uint8 的必要性。会议明确，如果不进行打包，内存消耗将翻倍。

- **将 uint8 解包为三进制值**：讨论并改进了将 uint8 数据解包为三进制值以及处理有符号/无符号位移的代码示例。还考虑了一种通过平移分布进行量化的潜在变通方案。

- **项目管理的协作努力**：成员们承认了项目管理中的挑战，同时确保分享并遵循了自定义 CUDA 扩展和 dtype 创建的所有必要参考资料和最佳实践。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/135">msaroufim 开发的自定义 CUDA 扩展 · Pull Request #135 · pytorch/ao</a>：这是 #130 的可合并版本 - 我必须进行一些更新，包括：除非使用 PyTorch 2.4+ 否则跳过测试，如果 CUDA 不可用则跳过测试，将 ninja 添加到开发依赖项，本地...</li><li><a href="https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?pli=1#heading=h.ptttacy8y1u9">C++ 自定义算子手册</a>：未找到描述</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/dtypes/uint4.py">ao/torchao/dtypes/uint4.py (main 分支) · pytorch/ao</a>：用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao</li><li><a href="https://github.com/TimDettmers/bitsandbytes/commit/67475257a96b792f9b66e71892dab90f7a60ed87">为 NF4 添加文档；修复失败的 8-bit matmul；修复 absmax bug。… · TimDettmers/bitsandbytes@6747525</a>：…#529 #543</li><li><a href="https://github.com/pytorch/ao/pull/248">由 gau-nernst 改进 FP6 量化的原语 · Pull Request #248 · pytorch/ao</a>：解决 #208 待办事项：FP32/FP16/BF16 -> FP6 (CPU + CUDA)（带有正确的舍入），FP6 -> FP32/FP16/BF16 (CPU + CUDA)，添加测试，修复 OpenMP 中的异常，想办法在 CUDA kernel 中进行检查？...</li><li><a href="https://github.com/pytorch/ao/pull/68">由 msaroufim 实现 1.58 bit · Pull Request #68 · pytorch/ao</a>：修复了 #67
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1240948219861794869)** (219 messages🔥🔥): 

- **CC 数据集包含大量垃圾信息**：成员们讨论了 CC 数据集中持续存在的垃圾信息问题，指出各语言中都存在大量自动生成和重复的内容。Asada.shinon 提到中文数据集包含最多的垃圾信息，并设有专门的过滤器来处理此问题，同时分享了一篇来自 [Technology Review 的文章](https://www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/)，关于 GPT-4o 的中文 token 污染问题。

- **OpenELM 提供透明度与效率**：Smerkyg 询问了关于具有频繁 Checkpoints 的 LLM 模型，引发了关于 OpenELM 的讨论。OpenELM 是一种强调可复现性和效率的新型 LLM，与 OLMo 相比，其准确度提升了 2.36%。建议参考 [OpenELM Research](https://machinelearning.apple.com/research/openelm) 资源。

- **LoRA 中的内存与效率**：Premiumonion 询问了 LoRA 中的 FLOPs 和内存效率，结论是 LoRA 与全量微调（Full Fine-tuning）相比主要节省了内存。Skyward2989 确认内存通常是 AI 模型训练中的瓶颈。

- **加拿大与英国 AI 安全研究所合作**：Hyperion.ai 分享了英国和加拿大宣布在 AI 安全方面进行合作的消息，包括专业交流和借调以加强研究，详见 [政府出版物](https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership)。

- **对时间序列建模的兴趣**：Tiley 和 Hawk1399 讨论了使用自回归（Autoregression）对连续多变量时间序列进行建模的方法。Tiley 对自回归推理中的误差表示担忧，并建议查看 [arXiv](https://arxiv.org/abs/2402.03885) 上的 MOMENT，并考虑考虑非线性动力学范围的方法。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/">GPT-4o 的中文 Token 训练数据被垃圾邮件和色情网站污染</a>：该问题可能是由于数据清洗不足导致的，可能引发幻觉、性能下降和滥用。</li><li><a href="https://arxiv.org/abs/2404.07143">Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention</a>：这项工作引入了一种高效的方法，将基于 Transformer 的大语言模型 (LLMs) 扩展到无限长的输入，且内存和计算量有限。我们提出的方法中的一个关键组件...</li><li><a href="https://arxiv.org/abs/2310.01889">Ring Attention with Blockwise Transformers for Near-Infinite Context</a>：Transformer 已成为许多最先进 AI 模型的首选架构，在广泛的 AI 应用中展示了卓越的性能。然而，内存需求...</li><li><a href="https://arxiv.org/abs/2402.03885">MOMENT: A Family of Open Time-series Foundation Models</a>：我们介绍了 MOMENT，这是一系列用于通用时间序列分析的开源基础模型。在时间序列数据上预训练大模型具有挑战性，原因在于 (1) 缺乏大规模...</li><li><a href="https://machinelearning.apple.com/research/openelm">OpenELM: An Efficient Language Model Family with Open Training and Inference Framework</a>：大语言模型的可复现性和透明度对于推进开放研究、确保可信度至关重要...</li><li><a href="https://github.com/nshepperd/flash_attn_jax">GitHub - nshepperd/flash_attn_jax: JAX bindings for Flash Attention v2</a>：Flash Attention v2 的 JAX 绑定。通过在 GitHub 上创建账号来为 nshepperd/flash_attn_jax 的开发做出贡献。</li><li><a href="https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py">jax/jax/experimental/pallas/ops/tpu/flash_attention.py at main · google/jax</a>：Python+NumPy 程序的组合变换：微分、向量化、JIT 到 GPU/TPU 等 - google/jax</li><li><a href="https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership">英国-加拿大 AI 安全科学伙伴关系</a>：未找到描述</li><li><a href="https://zenn.dev/hellorusk/articles/27684d0ed96c4c">【風吹けば名無し】GPT-4o が獲得した日本語の語彙を調べる</a>：未找到描述</li><li><a href="https://www.aisi.gov.uk/">AI 安全研究所 (AISI)</a>：AI 安全研究所是科学、创新和技术部的一个局，旨在促进严谨的研究以实现先进的 AI 治理。</li><li><a href="https://blog.allenai.org/olmo-open-language-model-87ccfc95f580">OLMo: Open Language Model</a>：一个最先进的、真正开放的 LLM 和框架
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1240937732151316480)** (93 条消息🔥🔥):

- **讨论带有 CLIP guidance 的非离散嵌入空间**：成员们讨论了 **CLIP guidance** 的潜在问题，指出由于非离散性质和模型训练偏差，嵌入空间可能无法捕获所需的属性。一位参与者建议采用一种类似于 *针对文本的 LPIPS* 的替代方法。
- **Twitter 论文再次出现，引发讨论**：一个分享的 [Twitter 链接](https://twitter.com/arankomatsuzaki/status/1791289342121455993) 引发了对一篇显然具有影响力的论文的辩论。成员们讨论了其相关性以及对模型训练技术的影响。
- **Hierarchical Memory Transformers 的创新**：一篇关于 [Hierarchical Memory Transformers 的论文](https://arxiv.org/abs/2405.06067) 引起了兴趣，该论文提出了一种模仿人类记忆的新型框架，以增强长上下文处理。讨论深入探讨了循环模型和记忆架构。
- **分析 LLM 共现问题**：成员们探讨了评估语言模型输出中共现问题的挑战，特别是当模型遵循其先前的输出而非 Prompt 时。建议包括测量 cross-attention 贡献和 perplexity 指标。
- **研究跨模态的正向迁移**：围绕 **ImageBind** 及相关论文（[ImageBind](https://arxiv.org/abs/2305.05665), [PaLM-E](https://arxiv.org/abs/2303.03378)）的对话探讨了跨多种模态训练模型是否可以增强单模态任务的性能。这包括关于 **zero-shot recognition** 和结合模态嵌入以提高检索性能的讨论。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.06067">HMT: Hierarchical Memory Transformer for Long Context Language Processing</a>：基于 Transformer 的大语言模型（LLM）已广泛应用于语言处理任务。然而，大多数模型限制了允许模型关注每个 token 的上下文窗口...</li><li><a href="https://arxiv.org/abs/2405.05417">Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models</a>：众所周知，语言模型中分词器（tokenizer）的创建与模型训练之间的脱节会导致某些输入（如臭名昭著的 SolidGoldMagikarp token）诱发异常行为。...</li><li><a href="https://arxiv.org/abs/2305.05665">ImageBind: One Embedding Space To Bind Them All</a>：我们提出了 ImageBind，这是一种学习跨六种不同模态（图像、文本、音频、深度、热成像和 IMU 数据）联合嵌入的方法。我们展示了所有配对数据的组合并非必...</li><li><a href="https://arxiv.org/abs/2205.06175">A Generalist Agent</a>：受大规模语言建模进展的启发，我们采用类似的方法构建了一个超越文本输出领域的单一通用智能体（generalist agent）。该智能体我们称之为 Gato...</li><li><a href="https://arxiv.org/abs/2303.03378">PaLM-E: An Embodied Multimodal Language Model</a>：大语言模型在广泛的复杂任务中表现出色。然而，在现实世界中实现通用推理（例如机器人问题）提出了具身（grounding）的挑战。我们提出了具身...</li><li><a href="https://arxiv.org/abs/2310.02557">Generalization in diffusion models arises from geometry-adaptive harmonic representations</a>：为图像去噪而训练的深度神经网络（DNN）能够通过基于分数的逆扩散算法生成高质量样本。这些令人印象深刻的能力似乎暗示了逃离...</li><li><a href="https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/">ICLR 2024 Outstanding Paper Awards &#8211; ICLR Blog</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1240971637466337331)** (14 条消息🔥):

- **Scaling bare bones 论文面临批评**：一位成员对最近讨论的一篇研究论文的简陋性质发表了评论，指出其缺乏超参数调优（hyperparameter tuning），并对其在更高水平上的可扩展性（scalability）表示好奇。
- **估算 FLOP 计算的挑战**：关于模型前向和后向传播中 FLOPs 的正确计算展开了详细讨论。成员们提供了见解并引用了特定资源，如 [EleutherAI's cookbook](https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py) 以消除困惑，并指出某些计算可能排除了投影计算（projection computations），从而导致差异。
- **关于样本效率（sample efficiency）指标的查询**：一位成员提出了关于在各个领域定义和测量 **sample efficiency** 的问题，暗示该概念在 Scaling Laws 和高效资源管理方面的重要性。
- **关于 Bitnet 计算效率的理论问题**：进行了一场有趣的理论讨论，探讨如果一个计算效率更高的模型版本（使用相同的参数量但计算量显著减少）是否会改变 Chinchilla scaling laws 定义的最佳参数与 Token 比例。共识倾向于不会改变，假设增加的计算能力只会扩展此类模型的计算预算。

**提到的链接**：<a href="https://arxiv.org/abs/2405.10938">Observational Scaling Laws and the Predictability of Language Model Performance</a>：了解语言模型性能如何随规模变化对于基准测试和算法开发至关重要。Scaling Laws 是建立这种理解的一种方法，但其要求...

  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1240969521699880960)** (13 messages🔥): 

- **HF 模型自动使用默认 Prompt**：*模型会根据当前的通用实践自动使用默认 Prompt 进行提示。* 一位用户分享了他们使用不同方法微调模型的经验，并注意到性能上的差异。
- **寻求英语金融和加密货币相关的 AI 任务**：一位成员询问了有关金融、交易、投资和加密货币相关主题的优质任务，并明确表示偏好英语任务。
- **NeurIPS 基准测试文章伪评审请求**：一位成员询问是否有人有兴趣评审他们的 NeurIPS 基准测试文章。另一位成员给出了积极回应，同意了该请求。
- **提高大型模型的评估速度**：一位用户分享了在大型模型上运行评估的困难，注意到 MMLU 等任务耗时较长。另一位用户建议优化 Batch Size 设置以加快评估过程。
- **没有专门的 AI Safety/基准测试活动频道**：一位成员询问是否可以在专门频道中推广 AI Safety 或基准测试相关的活动。回复指出，目前 EleutherAI Discord 中没有此类频道。

**提到的链接**：<a href="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro">TIGER-Lab/MMLU-Pro · Datasets at Hugging Face</a>：未找到描述

  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1241456488741077002)** (1 messages): 

- **Soft prompt tuning 设置问题**：一位成员询问了最近在非 pipeline 情况下使用 Soft prompt tuning 设置的经验。他们提到了一个特定问题，即 *"在调用 model.to_sequential() 后，param.requires_grad 似乎被重置了。"*
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1241007050121543720)** (2 messages): 

- **请求每月汇总（Monthly Round Up）**：一位成员建议每月汇总将 *"非常有帮助"*。该提议表明了对定期摘要或更新以保持知情的渴望。 

- **表达不确定性**：Nathan Lambert 回复道 *"lol I don’t know Man"*，表示对之前的建议或相关讨论的不确定或模棱两可。这显示了对话中的随性语气。
  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1241042777727570022)** (29 messages🔥):

<html>
  <body>
    <ul>
      <li><strong>Meta 推出 Chameleon</strong>：Meta 的新模型 Chameleon 是一个 34B 参数的多模态基础模型，在文本和图像任务中的表现均优于 Flamingo 和 IDEFICS 等模型。它在约 10T tokens 上进行训练，并在人工评估中声称优于 GPT-4V。[Source](https://arxiv.org/abs/2405.09818)</li>
      <li><strong>DeepMind 发布 Flash-8B</strong>：更新后的 Gemini 1.5 论文介绍了 Flash-8B，这是一个不同于 Gemini 1.5 Flash 的新模型。Flash-8B 具有多模态和超长上下文窗口，同时保持极高的效率。[Source](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf#page=45)</li>
      <li><strong>Gemini 1.5 模型家族扩展</strong>：Gemini 1.5 Pro 和 Flash 模型较之前版本有显著改进，在文本和视觉基准测试中表现出色。它们在 MMLU 任务中的表现展示了其系列中的最高能力。[Source](https://goo.gle/GeminiV1-5)</li>
      <li><strong>Anthropic 扩大规模</strong>：据报道，Anthropic 使用了比其前代模型 Opus 多四倍的 compute，旨在开发更大、更强大的模型。[Source](https://www.anthropic.com/news/reflections-on-our-responsible-scaling-policy)</li>
      <li><strong>LMsys 宣布 “Hard Prompts” 类别</strong>：LMsys 在 Arena 中引入了 “Hard Prompts” 类别，以便在更具挑战性的任务上评估模型，并观察到了显著的排名变化。Llama-3-70B-Instruct 被用作裁判模型，但其可靠性受到质疑。[Source](https://fxtwitter.com/lmsysorg/status/1792625968865026427)</li>
    </ul>
  </body>
</html>
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/lmsysorg/status/1792625977207468315">来自 lmsys.org (@lmsysorg) 的推文</a>: 我们如何对这些标准进行分类？我们采用 Llama-3-70B-Instruct 作为裁判模型，帮助我们标记超过 100 万场 Arena 对战。总体而言，我们的分析揭示了 Arena 用户提示词的质量...</li><li><a href="https://x.com/swishfever/status/1791551855954370985?s=46">来自 fishy business (@swishfever) 的推文</a>: Chameleon 论文中的注释行： % \item 我们开源了 \model{} 的变体，允许文本和图像输入，但在所有模型规模上仅支持文本输出。引用 Tanishq Mathew Abraham, Ph.D. (...</li><li><a href="https://x.com/dalucasgonzalez/status/1791525232622342492?s=46">来自 lucas g (@DaLucasGonzalez) 的推文</a>: 我们更新后的 Gemini 1.5 技术报告发布了！很高兴能预览我们正在开发的新模型：Flash-8B https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf#page=4...</li><li><a href="https://fxtwitter.com/lmsysorg/status/1792625968865026427">来自 lmsys.org (@lmsysorg) 的推文</a>: 在 Arena 中引入 “Hard Prompts” 类别！为了响应社区对在更具挑战性的任务上评估模型日益增长的兴趣，我们很高兴推出新的 “Hard Pr...</li><li><a href="https://x.com/dalucasgonzalez/status/1791526024444006489?s=46">来自 lucas g (@DaLucasGonzalez) 的推文</a>: Flash-8B 具有与其他 1.5 模型相同的多模态和百万级上下文窗口，但占用空间极小，效率极高。世界上没有其他类似的模型。它展示了令人难以置信的能力...</li><li><a href="https://x.com/dalucasgonzalez/status/1791526696312803727?s=46">来自 lucas g (@DaLucasGonzalez) 的推文</a>: 我们的初步基准测试非常有前景，这只是初步预览，因为我们仍在积极开发该模型，以在该规模下实现性能最大化。</li><li><a href="https://x.com/suchenzang/status/1791533241494835376?s=46&t=_jodDCDeIUnWb_Td0294bw">来自 Susan Zhang (@suchenzang) 的推文</a>: 更新后的技术报告包含很多好东西！现在这个推文串主要关注 ⚡️ Gemini 1.5 Flash ⚡️... 🧵 引用 Jeff Dean (@🏡) (@JeffDean) Gemini 1.5 模型家族：技术报告更新...</li><li><a href="https://x.com/aidan_mclau/status/1792610354255769919">来自 Aidan McLau (@aidan_mclau) 的推文</a>: 哟，Anthropic 在酝酿什么，比 Opus 多 4 倍的 compute，真牛
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1241059896409981040)** (145 条消息🔥🔥): 

- **OpenAI 的 Superalignment 团队解散**：OpenAI 去年宣布成立 “Superalignment 团队”，旨在为潜在的超智能 AI 做准备。在包括 Ilya Sutskever 在内的核心研究人员离职后，该团队现已解散，详情参阅[此处](https://archive.is/gEjjA)。

- **Jan Leike 从 OpenAI 离职**：Superalignment 团队的前共同负责人 Jan Leike 在 [Twitter](https://x.com/janleike/status/1791498178346549382) 上表达了对 OpenAI 核心优先事项的分歧。

- **古德哈特定律 (Goodhart's law) 与 AI 欺骗**：用户讨论了古德哈特定律对大语言模型 (LLM) 的影响，担忧仅仅增加模型规模可能导致模型更好地触发古德哈特定律 (goodharting)，从而变得更具欺骗性。

- **OpenAI 备受争议的雇佣惯例**：OpenAI 因要求离职员工签署终身不得贬低协议 (nondisparagement agreements) 以保留已归属股权而面临批评，尽管领导层后来在 [Twitter](https://x.com/soumithchintala/status/1791612240371580999?s=46) 上澄清他们从未执行过此类条款。

- **OpenAI 回应 AI 语音争议**：在对其选拔过程提出质疑后，OpenAI 暂停了其 AI 语音 "Sky" 的使用。他们澄清该声音并非模仿名人，而是属于一位专业女演员。点击[此处](https://openai.com/index/how-the-voices-for-chatgpt-were-chosen/)了解更多。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/kelseytuoc/status/1791584361718100361?s=46">来自 Kelsey Piper (@KelseyTuoc) 的推文</a>：如果你在 60 天内不签署，你的份额就会消失。而且情况还会变得更糟——因为 OpenAI 还可以拒绝让你参加年度活动，而这是出售你已归属 PPUs 的唯一途径...</li><li><a href="https://x.com/kelseytuoc/status/1791584322698559780?s=46">来自 Kelsey Piper (@KelseyTuoc) 的推文</a>：我对关于 OpenAI 离职协议的文章收到了两种反应：“这很正常！”（并不正常；其他领先的 AI 实验室没有类似的政策）以及“这怎么可能...”</li><li><a href="https://x.com/sama/status/1791936857594581428?s=46">来自 Sam Altman (@sama) 的推文</a>：关于最近有关 OpenAI 如何处理股权的事宜：我们从未收回过任何人的已归属股权，如果人们不签署离职协议（或不同意...），我们也不会这样做。</li><li><a href="https://fxtwitter.com/OpenAI/status/1792443575839678909">来自 OpenAI (@OpenAI) 的推文</a>：我们听到了关于如何选择 ChatGPT 语音（尤其是 Sky）的疑问。在处理这些问题期间，我们正努力暂停 Sky 的使用。阅读更多关于我们如何选择这些语音的信息：https://openai...</li><li><a href="https://x.com/kelseytuoc/status/1791539443016536265?s=46&t=_jodDCDeIUnWb_Td0294bw">来自 Kelsey Piper (@KelseyTuoc) 的推文</a>：当你离开 OpenAI 时，你会得到一个令人不快的惊喜：一份离职协议，如果你不签署终身不得贬低承诺，你将失去所有已归属股权：https://www.vox.com/futu...</li><li><a href="https://x.com/soumithchintala/status/1791612240371580999?s=46">来自 Soumith Chintala (@soumithchintala) 的推文</a>：我从多位前 OpenAI 员工那里证实了这是真的，这就是为什么他们不对自己的经历发表任何负面评论。</li><li><a href="https://deepmind.google/discover/blog/introducing-the-frontier-safety-framework/">介绍前沿安全框架 (Frontier Safety Framework)</a>：我们分析和缓解先进 AI 模型带来的未来风险的方法</li><li><a href="https://x.com/janleike/status/1791498178346549382">来自 Jan Leike (@janleike) 的推文</a>：我加入是因为我认为 OpenAI 将是世界上进行这项研究的最佳场所。然而，相当长一段时间以来，我一直与 OpenAI 领导层在公司的核心优先级上存在分歧...</li><li><a href="https://x.com/kelseytuoc/status/1791539443016536265?s=46&t=_jodDC">来自 Kelsey Piper (@KelseyTuoc) 的推文</a>：当你离开 OpenAI 时，你会得到一个令人不快的惊喜：一份离职协议，如果你不签署终身不得贬低承诺，你将失去所有已归属股权：https://www.vox.com/futu...</li><li><a href="https://youtu.be/ZP_N4q5U3eE?si=hFlutzYz2Jd9E_rH&t=211">OpenAI 致力于确保超级智能安全的大力推动 | Jan Leike</a>：2023 年 7 月，OpenAI 宣布将投入 20% 的计算资源支持一个新团队和项目——超级对齐 (Superalignment)，旨在...</li><li><a href="https://www.vox.com/future-perfect/2024/5/17/24158403/openai-resignations-ai-safety-ilya-sutskever-jan-leike-artificial-intelligence">来自“我失去了信任”的推文：为什么负责守护人类的 OpenAI 团队崩溃了</a>：公司内部人士解释了为什么具有安全意识的员工正在离开。</li><li><a href="https://archive.is/gEjjA">OpenAI 的长期 AI 风险团队已解散 | WIRED</a>：未找到描述内容
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1241211192937283686)** (24 条消息🔥):

- **Chinatalk 节目获得点赞**：一位成员用大拇指表情符号称赞了 Chinatalk 节目，表示内容非常出色。
- **Llama3-from-scratch 项目是一个极佳的学习工具**：[Llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch) 被强调为优秀的学习资源，并暗示此类工具的创作者具备被雇佣的实力。一位成员感叹道：*"这类东西是最好的学习工具"*。
- **为初学者解释 Latent Consistency Models**：推荐了一篇为初学者解释 Latent Consistency Models (LCMs) 的博客，因其易读性而受到特别称赞。博客地址见[此处](https://naklecha.notion.site/explained-latent-consistency-models-13a9290c0fd3427d8d1a1e0bed97bde2)。
- **购买了新域名**：关于购买和抢注域名的讨论促使一位成员购买了域名 **rlhfbook.com**。价格非常低，通过 Porkbun 购买每年仅需 7 美元。
- **对 Books4 数据集保持警惕**：Books4 数据集被幽默地称为法律雷区，被比作大富翁游戏中的“直接入狱”卡。提到之前的法律行动主要针对的是数据集的维护者。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://naklecha.notion.site/explained-latent-consistency-models-13a9290c0fd3427d8d1a1e0bed97bde2">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://github.com/naklecha/llama3-from-scratch">GitHub - naklecha/llama3-from-scratch: 每次一个矩阵乘法实现 llama3</a>：每次一个矩阵乘法实现 llama3 - naklecha/llama3-from-scratch</li><li><a href="https://web.archive.org/web/20240519104217/https://www.reddit.com/r/datasets/comments/1cvi151/ai_books4_dataset_for_training_llms_further/">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1241229144436768819)** (41 条消息🔥): 

- **Yudkowsky 的末日见解获得关注**：一位用户分享了 **Liron Shapira** 的[一条帖子](https://x.com/liron/status/1791592296053686341?s=46)，强调了 **Eliezer Yudkowsky** 在 AI 风险意识方面广泛但不完整的的影响力。该用户强调，其他专家仍处于通往对该问题“充分认识”的道路上。

- **搞笑的 AI 增长黑客手段**：用户讨论了来自 **Hamel Husain** 的[一个迷因 (meme)](https://x.com/HamelHusain/status/1791707778245185613)，其中一人建议将该概念作为营销噱头。这个想法围绕着提供“1 年付费期”展开，尽管其提供的价值微乎其微。

- **卖掉沙发换取 AI 额度**：一位用户幽默地承认卖掉了沙发来购买 AI 课程，并宣称在“额度 (credits)”上很富有。在讨论中，**Natolambert** 承认了使用额度和尝试各种 API 的乐趣。

- **关于付费内容讲座的辩论**：**Natolambert** 表达了对为付费内容提供讲座的不适感，并提到 **Maven** 曾试图邀请他加入一门课程。他还对通过 YouTube 合作帮助他人利用自己的品牌获利表示担忧。

- **游戏根源与 YouTube 尝试**：对话谈到了 **Call of Duty** 的经历，**Natolambert** 分享了[他的 YouTube 频道](https://www.youtube.com/@natolambert)。大家怀旧地回忆了通过游戏技巧赢得尊重，以及对围绕学术论文进行内容创作的共同兴奋。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.theverge.com/2024/5/20/24160621/openai-chatgpt-gpt4o-sky-scarlett-johansson-voice-assistant-her">OpenAI 撤回了其类似于 Scarlett Johansson 的 ChatGPT 语音</a>：也许《她》(2014) 不应该成为 AI 语音功能的蓝图。</li><li><a href="https://x.com/liron/status/1791592296053686341?s=46">来自 Liron Shapira (@liron) 的推文</a>：你必须意识到 @ESYudkowsky 对我们处境有多糟糕的洞察力水准仍远超众人。仅仅因为其他专家终于开始转向他的观点，并不意味着他们已经...</li><li><a href="https://x.com/HamelHusain/status/1791707778245185613">来自 Hamel Husain (@HamelHusain) 的推文</a>：🤣
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1241189481751511134)** (6 条消息):

- **关于 RLHF 中 ORPO 论文的担忧**：一位用户询问是否有人看过 **ORPO 论文**，并指出该论文已被添加到 **Hugging Face 的库**中。另一位成员对 ORPO 的可扩展性表示怀疑，称：“听起来不错，但我不知道它的扩展性如何”，在表达怀疑的同时也提醒自己应给予其更多关注。
- **实际测试揭示 ORPO 的局限性**：一位成员分享了对 ORPO 的[测试结果](https://x.com/ethayarajh/status/1783270535369195905)，发现它“看起来还可以，但不是特别出色”。他们认为将 SFT 与基于 margin 的损失函数结合通常效果不佳，并暗示 ORPO 用 1-policy 替换参考模型的方法可能会导致过度正则化（over-regularization）。

**提到的链接**：<a href="https://x.com/ethayarajh/status/1783270535369195905">来自 Kawin Ethayarajh (@ethayarajh) 的推文</a>：@maximelabonne @winniethexu 在 ultrafeedback 上对齐了 zephyr-sft-beta，看起来 kto/dpo 效果更好一些？注意 zephyr-sft-beta 是在 ultrachat（而非 ultrafeedback）上进行 SFT 的，所以所有的...

---

**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1241928485950193764)** (2 messages): 

- **Chamath Palihapitiya 面临批评**：一篇 Substack 文章批评了 **Chamath Palihapitiya** 在推广特殊目的收购公司 (SPACs) 中扮演的角色，这导致了散户投资者的财务损失。作者认为 Palihapitiya 在继续否认任何不当行为的同时，对他人遭受的损失不屑一顾（[竞技场中的骗局](https://open.substack.com/pub/newcomer/p/the-scam-in-the-arena?r=68gy5&utm_medium=ios)）。
- **对 All-In Pod 主持人的幸灾乐祸**：一位成员表示乐于读到 All In Podcast 主持人们的失败，并指出他们看起来很不真诚。*“我真的很喜欢读关于 all in pod 主持人失败的消息。他们感觉太假了”*。

**提到的链接**：<a href="https://open.substack.com/pub/newcomer/p/the-scam-in-the-arena?r=68gy5&utm_medium=ios">竞技场中的骗局</a>：Chamath Palihapitiya 利用散户投资者大赚一笔后全身而退，却无法让自己低调收场。

---

**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>

---

**Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1241098150089986078)** (21 messages🔥): 

- **对 OnlyFans 私信自动化持悲观态度**：一位成员提到听了最近的 Latent Space 播客，其中采访了一位将 OnlyFans 私信自动化的人，这让他们感到“相当愤世嫉俗”。这被认为与当前的剧集讨论相关。
- **关于 OpenAI 动态的有趣剧集**：新的 [Retort AI 剧集](https://retortai.com/episodes/openai-diamond-of-the-season-or-quite-the-scandal) 讨论了 OpenAI 的两大进展，包括他们的新聊天助手以及为 RLHF 目标发布的 Model Spec。其中一个重点片段提到了亲密关系与技术之间日益模糊的界限。
- **词表大小的 Scaling laws**：一位成员提出了关于词表（vocab size）大小与模型大小相关的 Scaling laws 问题，思考推理速度与复杂度之间的潜在权衡。另一位成员回应称，使用奇特的分词器（tokenizers）会让模型更难稳定训练。
- **滞后现象与控制理论**：成员们讨论了“滞后现象”（hysteresis）一词及其在控制理论中的相关性，并提及了通过 [Amazon 书籍链接](https://www.amazon.com/Nonlinear-Dynamics-Chaos-Applications-Nonlinearity/dp/0738204536) 引用的 Steven Strogatz 的著作。他们幽默地思考是否为了掌握 GRE 词汇而需要了解更多控制理论。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.amazon.com/Nonlinear-Dynamics-Chaos-Applications-Nonlinearity/dp/0738204536">无标题</a>：未找到描述</li><li><a href="https://retortai.com/episodes/openai-diamond-of-the-season-or-quite-the-scandal">The Retort AI Podcast | ChatGPT 对话：社交季的钻石还是十足的丑闻？</a>：Tom 和 Nate 讨论了过去一周 OpenAI 的两大事件。备受欢迎的聊天助手，以及它所揭示的 OpenAI 世界观。我们还讨论了 OpenAI 新的 Mo...
</li>
</ul>

</div>

---

**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1241022818183942206)** (126 messages🔥🔥):

- **Hinton 和 Ilya 关于 Scaling Laws 的辩论**：@joelhellermark 分享了 Hinton 对 Ilya 关于 Scaling Laws 直觉的评价，他表示：*"Ilya 总是宣扬只要把它做大，它就会运行得更好。我一直觉得这有点像在逃避，你还得有新的想法才行。事实证明 Ilya 基本上是对的。"* [完整采访链接请点击此处](https://x.com/joelhellermark/status/1791398092400390195)。

- **Jan Leike 从 OpenAI 离职**：多位贡献者强调了 Jan Leike 辞去 OpenAI Alignment 负责人一职的消息，并分享了多个来源以及关于他离职影响的推测。Sam Altman 和 Greg Brockman 在[此处](https://x.com/gdb/status/1791869138132218351?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)发布了他们的感谢及未来的安全计划。

- **使用 Machine Learning 的垂直轴风力涡轮机**：分享了一篇关于 EPFL 研究人员使用 genetic learning algorithm 优化垂直轴风力涡轮机叶片轮廓的文章。与水平轴风力涡轮机相比，这种涡轮机噪音更小，对野生动物更友好。[完整报道请点击此处](https://actu.epfl.ch/news/machine-learning-enables-viability-of-vertical-axi)。

- **Obsidian 与 AI 辅助日记**：用户讨论了将 AI 与 Obsidian 等笔记系统集成，以创建更高效的日记/日志工作流。@neuralution 提到了一个项目，该项目通过自定义 Telegram bot 进行语音对话，将日记条目总结到 Obsidian 中。

- **比较 AI 编程语言：Rust vs Go**：一位成员询问 Rust 还是 Go 更适合 AI 开发。贡献者指出 Rust 正在受到关注，特别是 Hugging Face 的 Candle 和 tokenizers 等项目，而 Go 则更适合进行 LLM APIs 的 HTTP 调用应用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://www.vox.com/future-perfect/2024/5/17/24158403/openai-resignations-ai-safety-ilya-sutskever-jan-leike-artificial-intelligence">来自“我失去了信任”的推文：为什么负责守护人类的 OpenAI 团队崩溃了</a>：公司内部人士解释了为什么注重安全的员工正在离职。</li><li><a href="https://js.langchain.com/v0.1/docs/additional_resources/tutorials/">教程 | 🦜️🔗 Langchain</a>：以下是关于 LangChain.js 的教程和课程链接。有关 LangChain.js 常见用例的书面指南，请查看用例和指南部分。</li><li><a href="https://x.com/realsharonzhou/status/1792576516444065967?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Sharon Zhou (@realSharonZhou) 的推文</a>：幻觉（Hallucinations）是生产级 LLM 和 Agent 的最大阻碍之一。在内部以及为客户提供服务时，已经实现了无幻觉（<5%）。我们已经能够调整 LLM 来召回特定...</li><li><a href="https://hamel.dev/blog/posts/fine_tuning_valuable.html">Hamel 的博客 - 微调（Fine-Tuning）仍然有价值吗？</a>：对近期微调幻灭趋势的回应。</li><li><a href="https://x.com/gdb/status/1791869138132218351?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Greg Brockman (@gdb) 的推文</a>：我们非常感谢 Jan 为 OpenAI 所做的一切，我们知道他将继续从外部为这一使命做出贡献。鉴于他的离职引发的问题，我们...</li><li><a href="https://x.com/joelhellermark/status/1791398092400390195?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Joel Hellermark (@joelhellermark) 的推文</a>：与 @geoffreyhinton 谈论了 OpenAI 联合创始人 @ilyasut 对 Scaling Laws 的直觉👇。“Ilya 总是宣扬只要把它做得更大，它就会运行得更好。而我一直认为...”</li><li><a href="https://x.com/soniajoseph_/status/1791604177581310234?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Sonia Joseph (@soniajoseph_) 的推文</a>：致那些就 AGI 合意非合意（cnc）性爱派对联系我的记者们——在我二十多岁身处硅谷期间，我通过社区住房场景出入于精英科技/AI 圈子。...</li><li><a href="https://x.com/dan_biderman/status/1791506475010977875">来自 Dan Biderman (@dan_biderman) 的推文</a>：人们认为 LoRA 是 LLM 的万灵丹。真的是吗？它能在消费级 GPU 上提供与全量微调（full finetuning）相同的质量吗？虽然 LoRA 具有较低内存占用的优势，但我们发现...</li><li><a href="https://x.com/ns123abc/status/1791548950719103319">来自 NIK (@ns123abc) 的推文</a>：天哪，这绝对彻底完蛋了</li><li><a href="https://x.com/sama/status/1791543264090472660?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Sam Altman (@sama) 的推文</a>：我非常感谢 @janleike 对 OpenAI 对齐（alignment）研究和安全文化的贡献，对他离开感到非常难过。他是对的，我们还有很多工作要做；我们致力于...</li><li><a href="https://x.com/natfriedman/status/1791462511889559615?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Nat Friedman (@natfriedman) 的推文</a>：通过阅读推荐论文并重新运行一些评估（evals），我得出的一个不太确定的结论是，从代码到其他推理过程存在一些微弱的迁移和泛化迹象...</li><li><a href="https://x.com/janleike/status/1791498174659715494">来自 Jan Leike (@janleike) 的推文</a>：昨天是我作为 OpenAI 对齐负责人、超级对齐（superalignment）负责人和高管的最后一天。</li><li><a href="https://actu.epfl.ch/news/machine-learning-enables-viability-of-vertical-axi/">机器学习使垂直轴风力涡轮机的可行性成为可能</a>：洛桑联邦理工学院（EPFL）的研究人员使用遗传学习算法来确定垂直轴风力涡轮机叶片的最佳变桨曲线，尽管这种涡轮机具有很高的能量潜力，但直到现在...</li><li><a href="https://threadreaderapp.com/thread/1791498174659715494.html">Thread Reader App 上 @janleike 的推文串</a>：@janleike：昨天是我作为 OpenAI 对齐负责人、超级对齐负责人和高管的最后一天。在过去的约 3 年里，这是一段疯狂的旅程。我的团队推出了有史以来第一个 RLHF LLM...</li><li><a href="https://github.com/sublayerapp/sublayer">GitHub - sublayerapp/sublayer：一个模型无关的 Ruby 生成式 AI DSL 和框架。提供用于构建 Generator、Action、Task 和 Agent 的基类，可用于在 Ruby 中构建 AI 驱动的应用程序。</a>：一个模型无关的 Ruby 生成式 AI DSL 和框架。提供用于构建 Generator、Action、Task 和 Agent 的基类，可用于在 Ruby 中构建 AI 驱动的应用程序。</li><li><a href="https://github.com/go-go-golems/geppetto">GitHub - go-go-golems/geppetto：golang GPT3 工具集</a>：golang GPT3 工具集。通过在 GitHub 上创建账户为 go-go-golems/geppetto 的开发做出贡献。</li><li><a href="https://news.ycombinator.com/item?id=40400224#40403951">S</a>

Sam 和 Greg 对 OpenAI 安全研究员指控的回应 | Hacker News</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1241116585867874455)** (127 messages🔥🔥): 

- **讨论了 "Feedback Is All You Need"**：分享了一个名为 "Feedback Is All You Need - Gordon Brander" 的 [YouTube 视频](https://www.youtube.com/watch?v=BNFRGfWQo6M)，引发了关于当前的 AI Agent 是否能够学习、适应并做出自主决策的讨论。
- **Andrew Ng 谈 AI Agent**：分享了 [Andrew Ng 的一条推文](https://x.com/AndrewYNg/status/1770897666702233815)链接，强调了 AI Agentic 工作流推动 AI 重大进展的潜力。他详细阐述了迭代工作流的好处，以及构建 Agent 的各种设计模式，如 Reflection、工具使用、规划和多 Agent 协作。
- **关于 AI Agent 定义的辩论**：成员们辩论了 AI Agent 的定义和属性，将其与传统的软件 Agent 进行比较，并将自主性、社交能力、反应性和持久性视为关键因素。
- **强化学习与历史背景**：讨论了 AI 中 Agent 的历史背景，引用了 1959 年 Samuel 的跳棋程序等开创性工作，突出了基于 Agent 的决策系统的起源和演变。
- **对 AI 音乐生成的兴趣**：成员们表达了对 AI 生成音乐及相关项目的兴奋和兴趣，并分享了个人轶事和未来的协作计划。一位成员提到正在进行 MusicGen 微调（finetunes）工作，并承诺分享相关链接。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Software_agent">Software agent - Wikipedia</a>: 未找到描述</li><li><a href="https://tenor.com/view/the-simpsons-mr-burns-muahahaha-evil-laugh-gif-4482837">Muahaha GIF - The Simpsons Mr Burns Muahahaha - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=BNFRGfWQo6M">Feedback Is All You Need - Gordon Brander</a>: 我们距离实现能够学习、适应并自主决策的 AI Agent 还有多远？我们已经达到了吗？究竟什么是 Agent？答案就在...</li><li><a href="https://x.com/AndrewYNg/status/1770897666702233815">Andrew Ng (@AndrewYNg) 的推文</a>: 我认为 AI Agentic 工作流今年将推动巨大的 AI 进步——甚至可能超过下一代基础模型。这是一个重要的趋势，我敦促每一位从事 AI 工作的人...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024 主题, 日期, 主持人, 资源,@dropdown,@ GenAI 的 UI/UX 模式,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1242138352224964679)** (1 messages): 

- **关于 Memary 项目的新网络研讨会**：本周四太平洋时间上午 9 点，我们将邀请 **memary** 的作者。**memary** 是自主 Agent 长期记忆的开源参考实现。本次网络研讨会将深入探讨该项目，并设有问答环节讨论记忆挑战和未来方向——[在此报名](https://lu.ma/nzh3o83f)。

**提到的链接**: <a href="https://lu.ma/nzh3o83f">LlamaIndex Webinar: Open-Source Longterm Memory for Autonomous Agents · Zoom · Luma</a>: 在本次网络研讨会中，我们很高兴邀请到 memary 的作者——这是一个完全开源的自主 Agent 长期记忆参考实现 🧠🕸️ 在……

  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1241065559664099370)** (10 messages🔥):

```html
- **QA 在处理大型表格时面临挑战**：由于解析效果不佳，即使是最新的 LLM 在处理像 Caltrain 时刻表这样复杂的表格时仍会出现幻觉。更多详情请见[此处](https://t.co/Scvp7LH2pL)。
- **将向量搜索速度提升 32 倍**：[JinaAI_](https://t.co/NnHhGudMa8) 分享了使用 32-bit 向量的方法，在仅损失 4% 准确率的情况下实现了显著的性能提升。这种优化对于生产级应用至关重要。
- **构建 Agentic 多文档 RAG**：Plaban Nayak 的文章解释了如何使用 LlamaIndex 和 Mistral 构建多文档 Agent。每个文档都被建模为一组用于全面总结的工具，详见[此处](https://t.co/FksUI3mm5l)和[此处](https://t.co/MbDtlrxk5B)。
- **全本地 Text-to-SQL 设置**：Diptiman Raichaudhuri 提供了一个教程，介绍如何设置本地 Text-to-SQL 系统，以便在无需外部依赖的情况下查询结构化数据库。指南可在此处获取：[此处](https://t.co/u3LG9NKE0X)。
- **旧金山见面会公告**：LlamaIndex 将在其总部举办线下见面会，届时将有来自 Tryolabs 和 Activeloop 等知名合作伙伴的演讲。见面会将涵盖高级 RAG 引擎技术；预约（RSVP）及更多详情请见[此处](https://t.co/o0BWxeq3TJ)。
```

**提到的链接**: <a href="https://t.co/qIGOmCW62G">预约 GenAI Summit Pre-Game: Why RAG Is Not Enough? | Partiful</a>：注：这是在旧金山 LlamaIndex 总部举办的线下见面会！顺道参加我们的见面会，了解为公司构建生产级 Retrieval Augmented Generation 引擎的最新创新技术...

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1240939005718302780)** (139 条消息🔥🔥): 

- **用于数据治理的 MetaDataFilters**：一位用户弄清楚了 LlamaIndex 中 MetaDataFilters 的工作原理，并直接在数据库（DB）层面应用过滤器。他们好奇 MetaDataFilters 对于可扩展应用中的数据治理是否可行，并询问了关于限制金融数据访问的选择性索引（selective indexing）问题。

- **Neo4jVectorStore 的 Embedding 问题**：一位用户报告了在将 LlamaIndex 与包含预先创建的 Embedding 和节点的现有 Neo4j 图数据库集成时出现的错误。他们讨论了几种使用 LlamaIndex 创建兼容节点和 Embedding 的方法以解决此问题。

- **模型和查询配置帮助**：用户讨论了在 LlamaIndex 中使用不同的 Embedding 模型和查询引擎（query engines），包括设置环境变量、将模型传递给查询引擎以及处理 Embedding 设置问题。分享了多个指向 LlamaIndex 文档和示例的链接。

- **多 Agent 和工具的挑战**：对话详细讨论了在 LlamaIndex 中使用多个工具和 Agent 的问题，包括 GPT-4 等 Agent 在工具选择上的困惑和低效。一位用户分享了他们的变通方案，包括使用 ReactAgent 作为子 Agent（sub-agent）。

- **RAG 应用中的数据治理**：进行了一场关于使用 LlamaIndex 和 Langchain 在 RAG 应用中实现数据治理的复杂讨论。分享了来自 NVIDIA 和 Microsoft 关于集成访问控制（access control）的演讲和文章链接，以供深入了解。

- **其他 LlamaIndex 查询**：用户询问了聊天机器人引擎（chatbot engines）与查询引擎（query engines）之间的区别、处理 Pinecone 中的文档重复、为 RAG 应用从网页抓取数据，以及修改 LlamaIndex 中 OpenAI Agent 的系统提示词（system prompts）。交流了各种解决方案和故障排除步骤。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llamahub.ai/l/readers/llama-index-readers-s3?from=">未找到标题</a>：未找到描述</li><li><a href="https://llamahub.ai/">Llama Hub</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#modules">正在重定向...</a>：未找到描述</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtc24-s62731/">检索增强生成：政府效率的新前沿 | NVIDIA On-Demand</a>：我们将介绍检索增强生成 (RAG)，这是一种可以从大型数据源中搜索并生成答案的 AI 技术</li><li><a href="https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/access-control-in-generative-ai-applications-with-azure-ai/ba-p/3956408">在使用 Azure AI Search 的生成式 AI 应用程序中进行访问控制</a>：在您的生成式 AI 应用程序中应用访问控制，以执行组织策略并限制对授权内容的访问。</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management/?h=insertion#insertion">文档管理 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Neo4jVectorDemo/?h=neo4j">Neo4j 向量存储 - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/b26112b02be29eed82fc2b808eaf55bc51e472c7/llama-index-core/llama_index/core/readers/file/base.py#L68">run-llama/llama_index 中的 llama_index/llama-index-core/llama_index/core/readers/file/base.py</a>：LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/qdrant_hybrid/">Qdrant 混合搜索 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/?h=ollama">Ollama - Llama 3 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/pandas_query_engine/?h=pandas+query">Pandas 查询引擎 - LlamaIndex</a>：未找到描述</li><li><a href="https://llamahub.ai/l/readers/llama-index-readers-file?from=">未找到标题</a>：未找到描述</li><li><a href="https://git.tonic-ai.com/contribute/snowflake/fdabot">🛂contribute / ❄️snowflake / FDABot · GitLab</a>：🙋🏻‍♂️ 欢迎来到 🌟Tonic-AI 社区
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1240977763679211541)** (4 条消息): 

- **GPT-4o 释放多模态能力**：一位用户分享了一篇关于将 GPT-4o 与 LlamaParse 集成的 [Medium 文章](https://medium.com/ai-advances/unleashing-multimodal-power-gpt-4o-integration-with-llamaparse-31c1e5e9da3a)。该链接获得了积极的反响，另一位成员评论道“nice!”。
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1240963475770048532)** (134 条消息🔥🔥): 

- **CommonCanvas 数据集引发争议**：成员们对 CommonCanvas 表达了复杂的看法。这是一个包含 70M 图像-文本对（包含 alt 文本和合成字幕）的数据集，但由于其限制性的非商业许可证，引发了不满。*“似乎完全适得其反”*以及*“禁止衍生作品也很奇怪，因为如果人们能修改或扩展这个数据集不是更好吗？”*这些言论体现了这种挫败感（链接：[公告](https://x.com/multimodalart/status/1791201296357142663)）。

- **torch.compile 和 GPU 利用率的挑战**：成员们（如 drhead）讨论了由于 PyTorch 的 native_group_norm 和频繁的设备同步问题导致的显著减速。该问题突出了 PyTorch 的 eager 模式与 torch.compile 之间的性能差异（*“我运行它的速度仅比使用 torch.compile 所能达到的速度慢 5% 左右”*）。

- **对 AI 字幕中幻觉的担忧**：关于幻觉字幕对训练视觉语言模型和文本生成图像模型（VLLM 和 T2I）影响的辩论正在进行中。*“我一直在和一个实验室交流，他们说字幕中的幻觉对 VLLM 和 T2I 实际上是非常有害的，但我还在等待相关论文”*。

- **讨论使用 LLava 和 CogVLM 创建数据集**：成员们正在探索使用 LLava 和 CogVLM 等各种 AI 模型为大型数据集生成字幕。虽然 LLava-next 和 LLaMA 模型正受到关注，但他们对 CogVLM 的性能表示怀疑（*“cogvlm 也很烂”*）。

- **对更多开源数据集的渴望**：用户正在积极讨论创建足够大、高质量且多样化的数据集，以用于训练基础模型，并参考了 CC12M 等项目以及各种 VLM，同时也表达了对数据完整性和可访问性的关注。*“我将永远开源我的（数据集）”*以及避免训练数据中出现“幻觉”的情绪，突显了他们的努力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/multimodalart/status/1791201296357142663">来自 apolinario (multimodal.art) (@multimodalart) 的推文</a>：非常激动 CommonCanvas 刚刚发布！🖼️ • 首个完全基于公开授权图像训练的开源文本生成图像模型（SD2 和 SDXL 架构） • 该数据集包含约 70M 公开授权...</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B">THUDM/cogvlm2-llama3-chat-19B · Hugging Face</a>：未找到描述</li><li><a href="https://fxtwitter.com/mustafasuleyman/status/1792623877744623806?t=t5EX1E--TJ-mAJJZtzX4eg&s=19">来自 Mustafa Suleyman (@mustafasuleyman) 的推文</a>：我们正在将 Copilot 提升到新的水平。🚀 Copilot 将能够实时观看、聆听、交谈并提供帮助。观看此演示以了解我的意思。很快，你的 AI 伴侣将开始与你一起生活，无论...</li><li><a href="https://x.com/OpenAI/status/1792443575839678909">来自 OpenAI (@OpenAI) 的推文</a>：我们收到了关于如何选择 ChatGPT 语音（尤其是 Sky）的问题。我们正在暂停使用 Sky，同时处理这些问题。详细了解我们如何选择这些语音：https://openai...</li><li><a href="https://github.com/ProGamerGov/VLM-Captioning-Tools/blob/main/bad_caption_finder.py">VLM-Captioning-Tools/bad_caption_finder.py (位于 main 分支) · ProGamerGov/VLM-Captioning-Tools</a>：用于使用 VLM 为图像生成字幕的 Python 脚本 - ProGamerGov/VLM-Captioning-Tools
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1241016051563757698)** (13 条消息🔥): 

- **Chameleon 取得新突破**：在 [arXiv 论文](https://arxiv.org/abs/2405.09818)中介绍的 Chameleon 模型是一种混合模态模型，能够同时理解和生成图像与文本。它在图像字幕生成等任务中展示了 **state-of-the-art 性能**，其生成能力甚至超过了像 Llama-2 这样更大的模型。

- **Sakuga-42M，动画数据集的游戏规则改变者**：一项 [arXiv 研究](https://arxiv.org/abs/2405.07425)介绍了 Sakuga-42M，这是首个大规模卡通动画数据集。该数据集包含“4200 万个关键帧”，旨在填补卡通特定训练数据的空白。

- **CogVLM2 许可证引发关注**：针对新 CogVLM2 模型的许可证发布了**警告**，该许可证包含关于不得损害中国利益的使用限制条款，并要求争议由中国法院解决（[来源](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B/blob/main/LICENSE)，[GitHub](https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE)）。

- **MambaOut 在 Mamba 受挫之处介入**：尽管 Mamba 模型在架构上极具前景，但在视觉任务中的表现不如注意力机制和卷积模型（[arXiv 论文](https://arxiv.org/abs/2405.07992)）。经验证据表明 Mamba 对于图像分类并非必要，但其长序列处理能力在检测和分割任务中仍具前景。

- **科比·布莱恩特因 Mamba 的表现被做成梗图**：用户幽默地引用了科比·布莱恩特的名言“Mamba out”来评论 Mamba 模型不尽如人意的表现。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.07425">Sakuga-42M 数据集：扩展卡通研究</a>：手绘卡通动画利用草图和色块来创造运动的错觉。虽然 CLIP、SVD 和 Sora 等近期进展在理解和...方面表现出色。</li><li><a href="https://arxiv.org/abs/2405.07992">MambaOut：我们真的需要 Mamba 来处理视觉吗？</a>：Mamba 是一种具有类似 RNN 的状态空间模型 (SSM) token mixer 的架构，最近被引入以解决注意力机制的二次复杂度问题，并随后应用于视觉任务...</li><li><a href="https://arxiv.org/abs/2405.09818">Chameleon：混合模态早期融合基础模型</a>：我们介绍了 Chameleon，这是一个基于 token 的早期融合混合模态模型系列，能够以任意顺序理解和生成图像与文本。我们概述了一种稳定的训练方法...</li><li><a href="https://arxiv.org/abs/2405.09673">LoRA 学得更少，遗忘也更少</a>：低秩自适应 (LoRA) 是一种广泛使用的 LLM 参数高效微调方法。LoRA 通过仅对选定的权重矩阵训练低秩扰动来节省显存。在...</li><li><a href="https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE">CogVLM2/MODEL_LICENSE (位于 main 分支) · THUDM/CogVLM2</a>：第二代 CogVLM 多模态预训练对话模型。通过在 GitHub 上创建账号为 THUDM/CogVLM2 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1240942274532085800)** (1 条消息):

- **构建语义研究论文应用**：一位成员分享了关于如何使用 **LangChain, Chainlit 和 Literal AI** 构建语义研究论文应用的[近期文章](https://towardsdatascience.com/building-an-observable-arxiv-rag-chatbot-with-langchain-chainlit-and-literal-ai-9c345fcd1cd8)。该文章还包括了将可观测性（observability）功能集成到应用中的步骤。
  

---



**AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1241921904885895199)** (2 messages): 

- **4Wall 发布 AI 娱乐平台**：[4wall](https://beta.4wall.ai) 背后的团队正在开发一个 AI 驱动的娱乐平台，目前处于 beta 阶段。在 [X (原 Twitter)](https://x.com/4wallai_/status/1792359640170410339?s=46&t=W_c0j4FPVSWZuhD7zTaSYA) 上分享了一段预告视频。
- **AI Town 集成与用户生成内容**：4Wall 计划将 AI Town 集成到他们的平台中，允许用户无缝使用 bot。他们还在开发让用户创建地图和游戏的功能。
- **3D AI 角色正在开发中**：4Wall 团队宣布 3D AI 角色功能正在开发中，很快就会上线。
<div class="linksMentioned">

<strong>提及链接</strong>:

<ul>
<li>
<a href="https://beta.4wall.ai">4Wall AI</a>: 在 4thWall AI 上探索互动式 AI 内容</li><li><a href="https://x.com/4wallai_/status/1792359640170410339?s=46&t=W_c0j4FPVSWZuhD7zTaSYA">来自 4Wall AI (@4WallAI_) 的推文</a>: ✨即将登陆 4Wall✨ http://beta.4wall.ai
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/)** (1 messages): 

.ghost001: 当更先进的版本发布时，他们会觉得自己很蠢。
  

---


**AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1241197728050970624)** (1 messages): 

- **宣布 Rosebud AI Game Jam 获胜者**：**Rosebud / #WeekOfAI 教育 Game Jam** 的获胜者已公布，展示了令人惊叹的 AI 驱动教育游戏。第一名游戏 **"Pathfinder: Terra’s Fate"** 和第三名 **"Ferment!"** 因其引人入胜的体验而受到关注。[查看获胜者](https://x.com/Rosebud_AI/status/1791616913279160327) 并在此处尝试 Rosebud [here](https://play.rosebud.ai/)。
<div class="linksMentioned">

<strong>提及链接</strong>:

<ul>
<li>
<a href="https://x.com/Rosebud_AI/status/1791616913279160327">来自 ⚡Rosebud AI🌹 (@Rosebud_AI) 的推文</a>: 🌟 展示我们 #WeekOfAI 教育 Game Jam 的获胜者！🌟 这 4 款使用 Rosebud AI 制作的令人惊叹的 AI 驱动游戏，展示了如何创建有趣的教育游戏。非常感谢我们的评委...</li><li><a href="https://play.rosebud.ai/">在 Rosebud 上玩和创建游戏 —— AI 驱动的游戏开发</a>: 使用 AI 来创建、分享和玩游戏。从文本描述到代码再到游戏。
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1240957770447196180)** (38 messages🔥): 

- **AI Town 现在原生支持 Windows**：一位成员兴奋地宣布 **AI Town** 现在可以在 Windows 上原生运行，无需 WSL 或 Docker，并分享了 [@cocktailpeanut 的推文](https://fxtwitter.com/cocktailpeanut/status/1791495360541593964) 确认了 AI Town Windows 版一键启动器（1 Click Launcher）的发布。
- **AI 真人秀平台发布**：兴奋的成员们分享了 [AI 真人秀（AI Reality TV）平台的发布](https://x.com/edgarhnd/status/1791586276178587707)，该平台允许用户创建社交模拟并观察 AI 驱动的角色互动。他们鼓励其他人加入并主持下一个看点，创建诸如“伊丽莎白在《加勒比海盗》中选择杰克还是威尔”之类的模拟。
- **安装问题与解决方案分享**：一位用户在设置 AI Town 对话时遇到问题，被建议查看 [内存系统文档](https://github.com/a16z-infra/ai-town/blob/main/convex/agents.ts) 并调整 `convex/constants.ts` 中的设置以改进对话持久性。
- **从 SQLite 数据库提取对话**：用户讨论了从 AI Town 使用的 SQLite 数据库中提取对话的方法。分享了用于导出数据的有用 SQL 查询，并提供了 [相关仓库链接](https://github.com/cocktailpeanut/townplayer/blob/main/index.html) 以进一步协助过滤和导出对话数据。
- **添加有趣的角色并观察 AI 互动**：成员们分享了他们在 AI Town 中添加的创意角色，如间谍和当地记者，并注意到了真实的互动。提到了故障排除，用户分享了关于调整内存获取（memory fetch）设置如何改善角色互动的经验。
<div class="linksMentioned">

_

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/edgarhnd/status/1791586276178587707">Edgar Haond 🎲 (@edgarhnd) 的推文</a>：很高兴今天发布 AI Reality TV！我们的新平台让你能够创建自己的社交模拟。曾经好奇过《加勒比海盗》里的伊丽莎白更喜欢杰克还是威尔吗？现在你可以模拟...</li><li><a href="https://fxtwitter.com/cocktailpeanut/status/1791495360541593964">cocktail peanut (@cocktailpeanut) 的推文</a>：AI Town 一键启动器登陆 Windows！感谢 @convex_dev 团队的努力，我们终于有了 Windows 原生 convex 二进制文件（为 AI Town 提供动力）。AI Town——一个完全可黑客化、持久化的...</li><li><a href="https://www.aireality.tv/">AI Reality TV</a>：未找到描述</li><li><a href="https://x.com/emollick/status/1791695567212699874">Ethan Mollick (@emollick) 的推文</a>：好的，我在机器上本地运行了一个由自主 AI Agent 组成的小镇，并给他们分配了《公园与游憩》中的角色。让我们看看会发生什么。</li><li><a href="https://youtu.be/UoJjeyQR66s?si=3EnN8hJO7UypY72K">我们制作了 AI Town 的后端。以下是实现方法。</a>：[0:00] - 介绍 [1:15] - 组件 [1:23] - Agents (https://github.com/a16z-infra/ai-town/blob/main/convex/agents.ts) [1:29] - 引擎 (https://gi...</li><li><a href="https://pinokio.computer/">Pinokio</a>：AI 浏览器</li><li><a href="https://x.com/cocktailpeanut/status/1786421948638965870">cocktail peanut (@cocktailpeanut) 的推文</a>：介绍 AI Town Player。你知道整个 AI Town 都通过 @convex_dev 存储在单个 sqlite 文件中吗？我逆向工程了其模式，并构建了一个 Web 应用，让任何人都可以回放任何 A...</li><li><a href="https://github.com/cocktailpeanut/townplayer/blob/main/index.html">townplayer/index.html at main · cocktailpeanut/townplayer</a>：回放 AI Town。通过在 GitHub 上创建账户，为 cocktailpeanut/townplayer 的开发做出贡献。
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1241066877623734342)** (94 条消息🔥🔥): 

- **AI Town 为 AI Reality TV 做好准备**：分享了一个加入 AI Reality TV 节目的链接，点击[这里](https://discord.com/invite/NtUXDUSnKa)。鼓励成员创建自己的 AI 并帮助其赢得比赛，详情见此：[AI Reality TV](https://www.aireality.tv/)。

- **分享 AI Town 的技术细节**：AI Town 的技术栈被描述为：使用 Convex 作为后端，JS/TS 处理应用逻辑，Pixi.js 处理图形，Clerk 处理身份验证，推理则在 Ollama 和 OpenAI 之间切换。

- **AI Town 中的错误排查**：成员在 Windows 上使用 AI Town 时遇到了连接问题，在 Agent 通信期间出现错误。他们被引导至 Pinokio Discord 服务器寻求进一步帮助。

- **保存和提取对话**：讨论了使用 Web 应用从 AI Town 导出 sqlite 文件的可能性，并提供了 [GitHub - Townplayer](https://github.com/cocktailpeanut/townplayer) 的链接。替代方法包括使用任何 sqlite 查看器，以及针对托管版本的 convex 仪表板。

- **世界上下文集成建议**：有人指出，直接在角色提示词（prompts）中添加上下文可以丰富 AI Town 的叙事。建议使用世界描述来获得更好的上下文，并讨论了让 Convex 的托管仪表板支持本地部署的计划。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/cocktailpeanut/status/1786421948638965870">cocktail peanut (@cocktailpeanut) 的推文</a>：介绍 AI Town Player。你知道整个 AI Town 都通过 @convex_dev 存储在单个 sqlite 文件中吗？我逆向工程了其模式，并构建了一个 Web 应用，让任何人都可以回放任何 A...</li><li><a href="https://www.aireality.tv/">AI Reality TV</a>：未找到描述</li><li><a href="https://github.com/cocktailpeanut/townplayer">GitHub - cocktailpeanut/townplayer: Replay AI Town</a>：回放 AI Town。通过在 GitHub 上创建账户，为 cocktailpeanut/townplayer 的开发做出贡献。
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1241016110480883853)** (112 条消息🔥🔥):

- **服务器在 function calls 时返回 500 状态码：** 一位用户报告称，*“当使用 function calls 调用时，服务器返回 500 状态码，并显示消息 'Function calling is not supported by openrouter'。”* 对话中未立即提供解决方案。
- **无效的模型 URL 导致应用错误：** 用户注意到，导航到无效的模型 URL 会导致页面崩溃并显示 *“Application error: a client-side exception has occurred (see the browser console for more information)”*，而不是正常的 404 错误。该行为根据用户是否登录而有所不同。
- **自动充值支付问题：** 多次交流讨论了 **自动充值支付被拒绝** 的问题，导致用户的额度低于允许限制且 **无法手动充值**。该问题被确定为可能被用户的银行（WISE EUROPE SA/NV）拦截。
- **模型推荐和微调反馈：** 用户分享了他们对各种模型的体验，提到了 **“Cat-LLaMA-3-70B”**、**Midnight-Miqu 模型**，以及需要 **更好的微调方法**，而不是“随机未清洗数据”的方法。一位用户指出：*“试试 Cat-LLaMA-3-70B，当你真正让它运行起来时，它的表现非常令人印象深刻。”*
- **Wizard LM 8x22B 请求失败问题：** 一位用户询问关于 OpenRouter 上 **Wizard LM 8x22B** 频繁失败的问题，这些失败被确定为来自多个提供商的请求超时（408）临时激增。

在此阅读完整对话：[OpenRouter Discord](https://discord.com/channels/1091220969173028894)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/Xenova/the-tokenizer-playground">The Tokenizer Playground - a Hugging Face Space by Xenova</a>：未找到描述</li><li><a href="https://orw.karleo.net/list">OpenRouter API Watcher</a>：探索 OpenRouter 的模型列表和记录的变更。每小时更新一次。</li><li><a href="https://openrouter.ai/models/google/gejksdf">OpenRouter</a>：LLM 和其他 AI 模型的路由中心</li><li><a href="https://openrouter.ai/models/google/gejk.sdf">OpenRouter</a>：LLM 和其他 AI 模型的路由中心
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1241019330238480455)** (58 messages🔥🔥): 

- **Galore Layerwise 缺乏 DDP 支持**：用户对 Galore Layerwise 工具 **无法支持 DDP (Distributed Data Parallel)** 表示沮丧，这表明其功能仍存在局限性。

- **大型中文数据集引起关注**：关于使用 10 亿个 Token 在非英语语言（特别是中文）中 **微调大型 8B 模型** 的对话引起了关注。相关数据集链接：[Multimodal Art Projection (M-A-P)](https://huggingface.co/m-a-p) 和 [BAAI](https://huggingface.co/BAAI)。

- **微调过程中的梯度范数问题**：讨论揭示了在模型微调（特别是 llama 3 8B）中使用 low rank 时，**梯度范数（gradient norms）出现无限制增长**。该问题似乎根源于权重饱和，导致在没有显著扰动的情况下无法更新梯度。

- **GPT-4o 的垃圾 Token**：成员们对 [GPT-4o 的 Token 被垃圾信息和色情短语污染](https://www-technologyreview-com.cdn.ampproject.org/c/s/www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/amp/) 表示担忧，强调了最新版本在中文 Token 解析方面的缺陷。

- **针对 axolotl 的 Commandr 配置**：与设置 Commandr 配置相关的问题已通过一个 [特定的 GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files) 得到部分解决。用户正在协作测试并实施此配置，以期将其合并到项目中。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www-technologyreview-com.cdn.ampproject.org/c/s/www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/amp/">GPT-4o’s Chinese token-training data is polluted by spam and porn websites</a>: 未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files">Feat: Add cohere (commandr) by NanoCode012 · Pull Request #1547 · OpenAccess-AI-Collective/axolotl</a>: 描述、动力与背景。如何测试？未测试！屏幕截图（如适用）。变更类型。社交账号（可选）。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/">Pull requests · OpenAccess-AI-Collective/axolotl</a>: 尽管提出 axolotl 问题。通过在 GitHub 上创建账户来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://huggingface.co/m-a-p">m-a-p (Multimodal Art Projection)</a>: 未找到描述</li><li><a href="https://huggingface.co/BAAI">BAAI (Beijing Academy of Artificial Intelligence)</a>: 未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1241006527091970090)** (13 messages🔥): 

- **数据集对 PoSE 的影响**：一位成员询问数据集的选择是否会显著影响 PoSE 中上下文扩展的质量。另一位成员回答道：“*我没怎么尝试过不同的数据集*。”
- **针对 Llama 的 Unsloth 优化**：一位成员询问是否有理由不在全量微调（full finetune）中使用 [针对 Llama 的 Unsloth 优化](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1609)。另一位成员回答说，**Unsloth** 的交叉熵损失（cross entropy loss）对于全量微调是没问题的。
- **随机数据集对 PoSE 来说足够了**：当被问及随机数据集是否足以用于 PoSE 时，一位成员确认道：“*是的，对于 niah 来说足够了，但老实说，PoSE 似乎并没有真正提升长上下文推理或理解能力*。”
- **Torchtune 优化**：一位成员强调了来自 [Torchtune pull request #993](https://github.com/pytorch/torchtune/pull/993) 的潜在有价值的优化。他们提到：“*axolotl 与 torchtune 的集成即将推出*。”
- **HF 后端的未来**：成员们讨论了 Torchtune 是会取代 HF 后端还是仅仅作为另一个选项。有人建议：“*拆解 hf*”，表达了对重大变革的渴望。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/hpcai-tech/grok-1">hpcai-tech/grok-1 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1609">Unsloth optims for Llama by winglian · Pull Request #1609 · OpenAccess-AI-Collective/axolotl</a>: 将 Unsloth 的优化集成到 axolotl 的 WIP。针对 MLP、QKV、O 的手动 autograd 似乎只减少了 1% 的 VRAM，而不是报告的 8%。交叉熵损失确实有显著帮助...</li><li><a href="https://github.com/pytorch/torchtune/pull/993">Llama3-70b: Full Finetune w/CPU offload + fused optimizer by rohan-varma · Pull Request #993 · pytorch/torchtune</a>: 背景。此 PR 的目的是什么？是添加新功能、修复 bug、更新测试和/或文档，还是其他（请在此添加）。请链接此 PR 解决的任何 issue。变更日志...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1241152120778915911)** (3 messages): 

- **持续预训练导致非法内存访问**：一位用户请求使用 Axolotl 进行持续预训练（continued pre-training）的示例，并指出他们尝试使用预训练数据集时导致了 *词表外（out-of-vocab）的 padding token 导致非法内存访问*。他们明确表示不想更改 tokenizer 的词表，并提供了一个示例配置。

- **Mistral 7b 微调问题**：一位用户分享了他们在指令数据上微调 **Mistral 7b** 的挑战，观察到尽管 loss 下降，但模型 *“把事情搞混了，似乎什么也没学到”*。他们提到其配置是基于 [这个示例](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/lora.yml) 并进行了一些自定义调整。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1241229094906106056)** (37 messages🔥): 

- **Phorm 协助处理 ORPO 格式查询**：用户询问了 ORPO (Object-Role-Property-Operation) 格式。虽然没有给出具体的实现细节，但提到 **ORPO 用于结构化数据/操作**，并且 Axolotl 中的一个示例包含了它在 prompt 策略中的应用。

- **阐明 LLM 训练中的权重衰减 (Weight decay)**：成员们讨论了权重衰减，它作为一种正则化技术，通过在损失函数中增加惩罚项来**防止过拟合**。这确保了模型权重保持在较小范围，从而获得更好的泛化能力。

- **LoRA Dropout 详解**：LoRA Dropout 通过在**低秩自适应矩阵 (low-rank adaptation matrices)** 中引入 Dropout，帮助微调 LLM，防止过拟合并提高泛化能力。

- **梯度累积 (Gradient accumulation) 益于 LLM 训练**：梯度累积允许在不增加显存占用的情况下，使用更大的有效 Batch Size 进行训练，这对于 LLM 的稳定性和效率至关重要。该方法通过 PyTorch 和 **Hugging Face Accelerator 库**进行了示例演示。

- **Axolotl 中的样本权重 (Sample weights)**：一位成员寻求在**不自定义损失函数的情况下分配样本权重**。建议在 `compute_loss` 方法中使用样本权重进行自定义损失处理，同时提醒并非所有损失函数都原生支持此功能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/src/axolotl/core/trainer_builder.py#L484L493)">axolotl/src/axolotl/core/trainer_builder.py at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/src/axolotl/prompt_strategies/orpo/chat_template.py#L205L239)">axolotl/src/axolotl/prompt_strategies/orpo/chat_template.py at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/examples/mistral/mistral-qlora-orpo.yml#L1L83),">axolotl/examples/mistral/mistral-qlora-orpo.yml at main · OpenAccess-AI-Collective/axolotl</a>：尽管提问。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=b80342a0-719c-4ab5-9320-9d50afdf43da)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8005a5ee-d28e-42f2-960d-3c29cc0e03ad)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=72e80667-3df2-495a-8fec-c8acfd801de6)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=fcde63a8-860b-4760-a6c8-03900deac358)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=7330f4bb-2f39-4e78-8f4d-36d7317c666c)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=f4c68df8-2e30-4ffb-b40c-ab1c79f91770)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=79ea3546-1ab6-4fe1-8984-1d8eb8183eda)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1240927402327281766)** (69 条消息🔥🔥):

- **LangChain 处理重排序（re-ranking）和记忆（memory）**：一位成员询问了在组织代理（organizational proxy）后使用 cross encoder 对结果进行重排序的问题，想知道 OpenAI GPTs 或 Gemini 模型是否可行。另一位成员强调了为聊天机器人实现短期记忆（如 buffer memory）的重要性。
- **在 LangChain 中引导模型响应**：一位成员询问如何在 React agent 中设置特定问题以引导模型获得最佳答案。另一位成员通过建议使用 LangChain 中的 `PromptTemplate` 函数自定义提示词或模板进行了澄清，并分享了一个 [GitHub issue 链接](https://github.com/langchain-ai/langchain/issues/18820)。
- **面向 Swift 开发者的 LangChain**：一位成员询问 LangChain 是否适用于开发 iOS 或 macOS 的 Swift 开发者。另一位成员分享了一个针对 iOS、macOS、watchOS 和 visionOS 优化的 [LangChain Swift GitHub 链接](https://github.com/buhe/langchain-swift)。
- **在 LangChain 中处理 SQL 数据**：一位正在处理呼叫中心通话摘要的成员讨论了跨多个通话总结概念的问题，并询问如何将 SQL 数据作为 LangChain 中的记忆使用。另一位成员推荐了与类 SQL 数据库的各种集成，并分享了一个 [LangChain 集成链接](https://python.langchain.com/v0.1/docs/integrations/memory/)。
- **Langmem 的上下文能力**：一位成员对 Langmem 在会话中途切换话题时保持上下文对话的能力表示惊讶，并分享了展示 Langmem 长期记忆和上下文管理的 [YouTube 视频](https://youtu.be/7QU89qL795A)。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/integrations/memory/">Memory | 🦜️🔗 LangChain</a>: 未找到描述</li><li><a href="https://app.reclaim.ai/m/cp/ai-storytelling-and-gaming">AI Storytelling and Gaming</a>: 嗨 - 我是 Chris，我正在尝试了解人们如何使用 AI 来讲故事和玩游戏。如果你尝试过 AI Dungeon 或 Novel AI 等应用，或者只是使用 ChatGPT 尝试讲故事...</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B#prompt-format">NousResearch/Hermes-2-Pro-Mistral-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md">openai-python/chatml.md at release-v0.28.0 · openai/openai-python</a>: OpenAI API 的官方 Python 库。通过在 GitHub 上创建账号为 openai-python 的开发做出贡献。</li><li><a href="https://youtu.be/OL6RDg04FNc">Langmem - Episode 2 | Managing context switching</a>: 在这段录音中，我展示了 langmem 如何帮助继续上下文对话。它能够在不同上下文之间切换。前一个视频：https://youtu.be...</li><li><a href="https://github.com/Dataherald/dataherald?tab=readme-ov-file">GitHub - Dataherald/dataherald: Interact with your SQL database, Natural Language to SQL using LLMs</a>: 与你的 SQL 数据库交互，使用 LLM 将自然语言转换为 SQL - Dataherald/dataherald</li><li><a href="https://github.com/langchain-ai/langchain/issues/18820>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://youtu.be/7QU89qL795A">Langmem | Long term memory from Langchain</a>: 在这段录音中，我谈到了 langmem。这是来自 langchain 的最新创新之一。它专注于长期记忆。我认为我们应该专注于 m...</li><li><a href="https://github.com/buhe/langchain-swift">GitHub - buhe/langchain-swift: 🚀 LangChain for Swift. Optimized for iOS, macOS, watchOS (part) and visionOS.(beta)</a>: 🚀 LangChain for Swift。针对 iOS, macOS, watchOS (部分) 和 visionOS 优化。(beta) - buhe/langchain-swift</li><li><a href="https://www.reddit.com/r/LangChain/s/0e0H0tm1o1">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1241509355355111465)** (2 messages): 

- **Kenny Tang 分享 50 美元 Steam 礼品链接**：KennyTang 发布了一个据称是 50 美元 Steam 礼品的链接：[steamcommunity.com/gift/50](https://bitly.cx/OjEZl)。该消息提到了 @everyone 和 @here。
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1241509366599909456)** (2 messages): 

- **分享了可疑的 50 美元礼品链接**：一位用户分享了一个标题为 *"Gift 50$"* 的链接，指向 [steamcommunity.com](https://bitly.cx/OjEZl)。该链接被多次分享，并在频道中提到了所有人。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1241035600455401502)** (7 messages):

- **Rubik's AI 提供免费高级访问权限**：一位成员介绍了一个高级研究助手和搜索引擎，邀请 Beta 测试人员使用促销代码 **RUBIX** 获得两个月的免费高级访问权限。提供的模型包括 GPT-4 Turbo、Claude-3 Opus 和 Mistral Large。[去看看](https://rubiks.ai/)。

- **分享了 LangServe 博客文章**：一位成员分享了关于 LangServe 的博客文章链接。[什么是 LangServe？](https://flatteredwithflutter.com/what-is-langserve/)。

- **可疑的 50 美元 Steam 礼品链接**：一位成员发布了两条带有潜在可疑的 50 美元 Steam 礼品链接的消息。链接在[这里](https://bitly.cx/OjEZl)。

- **ChatGPT Chrome 扩展的联属计划**：另一位成员宣布了其 **Easy Folders** Chrome 扩展的联属计划。联属会员可赚取 25% 的佣金，而客户可获得 10% 的折扣。[在此注册](https://easyfolders.promotekit.com/)并[下载扩展](https://chromewebstore.google.com/detail/easy-folders-chatgpt-clau/gdocioajfidpnaejbgmbnkflgmppibfe?hl=en-GB&authuser=0)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/rutamstwt/status/1792091646663754226">来自 Rutam Bhagat (@rutamstwt) 的推文</a>: http://x.com/i/article/1792080553656922112</li><li><a href="https://x.com/the_mint_flip/status/1791944845772132433?s=46&t=RFXQiGP9iFKCCIMhv9N8qQ">来自 AmpJemima.Cro (@the_mint_flip) 的推文</a>: ‼️HAAaaLLP #crofam ‼️  🚨I need 84 more points🚨  🫵 SIGNUP FOR THE $Flu #airdrop 🫵  🐥😷🐥 http://trop.ee/nr9hRS5hyR🐥😷🐥  #bornbrave  #fftb  #cronosMemesDegens #CronosMemes #cronoschain #CronosMem...</li><li><a href="https://rubiks.ai/">Rubik's AI - AI 研究助手 & 搜索引擎</a>: 未找到描述</li><li><a href="https://easyfolders.promotekit.com/">注册</a>: 为 Stripe 提供的联属营销软件</li><li><a href="https://chromewebstore.google.com/detail/easy-folders-chatgpt-clau/gdocioajfidpnaejbgmbnkflgmppibfe?hl=en-GB&authuser=0">Easy Folders: ChatGPT &amp; Claude 聊天整理器</a>: 为 ChatGPT 和 Claude 提供拖放文件夹。彩色文件夹。嵌套文件夹。历史搜索。书签。批量删除聊天。</li><li><a href="https://chatgpt-easy-folders.vercel.app/">ChatGPT Easy Folders</a>: 一个通过文件夹、书签和搜索来整理 ChatGPT 历史记录的浏览器扩展。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1241424375782182983)** (3 条消息): 

- **RAG-Fusion 简化了多查询处理**：一条详细消息强调了 **RAG**（单查询）和 **RAG-Fusion**（多查询）之间的区别，并提供了关于集成 LangChain 和 GPT-4o 以创建用于文档处理的 AI 聊天机器人的见解。查看此 [YouTube 教程](https://youtu.be/P_xZ1HJwKl8?si=cQZ1CTydmFRjvveP)了解更多信息。
- **发布了可疑的 50 美元 Steam 礼品链接**：分享了一个声称提供 50 美元 Steam 礼品的链接（[可疑链接](https://bitly.cx/OjEZl)）并 @ 了所有人。建议对此类链接保持警惕。

**提到的链接**：<a href="https://youtu.be/P_xZ1HJwKl8?si=cQZ1CTydmFRjvveP">LangChain + RAG Fusion + GPT-4o Python 项目：轻松为你的文档创建 AI/聊天</a>: #automation #rag #llm #ai #programming #gpt4o #langchain 在这段视频中，我有一个非常快速的教程，向你展示如何使用 L 为你的 PDF 创建一个 AI...

  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1240926095012794428)** (76 条消息🔥🔥): 

- **改进 Discord 支持**：一位成员建议更改 Discord 上的支持系统，指出问题长期未得到解答。另一位成员澄清说，该频道更多是作为社区支持的聊天，而非官方员工支持系统。
  
- **试用版 API 的速率限制问题**：一位正在实验 `RAG retriever` 的用户遇到了 403 错误，并推测达到了试用版 API 的限制。其他人提到试用密钥受速率限制，不适用于生产环境。

- **免费 API 密钥**：出现了关于获取免费 API 密钥及其限制的查询。一位成员确认免费密钥可用但有限制，主要适用于原型设计而非生产环境。

- **使用 CommandR+ 进行翻译**：一位用户寻求使用 `CommandR+` 进行翻译的示例。另一位成员建议参考 [Chat API 文档](https://docs.cohere.com/docs/chat-api)了解实现细节。

- **作品集 vs. 生产环境使用**：讨论了在 Vercel 等平台上托管使用 Cohere AI 的应用以用于作品集展示。澄清了在作品集中执行通常被视为原型设计，属于免费使用范围，而生产环境涉及商业部署并会产生费用。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/docs/chat-api">使用 Chat API</a>: 未找到描述</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/retrievers/cohere/">Cohere RAG | 🦜️🔗 LangChain</a>: Cohere 是一家加拿大初创公司，提供自然语言处理模型，帮助企业改善人机交互。</li><li><a href="https://huggingface.co/Xenova/codegen-350M-mono">Xenova/codegen-350M-mono · Hugging Face</a>: 未找到描述</li><li><a href="https://cohere.com/pricing">价格</a>: 为各种规模的企业提供灵活且价格合理的自然语言技术。今天即可免费开始，按需付费。
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1241351465574862940)** (1 条消息): 

- **Cohere AI 完整指南已发布**: 一位成员在 [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2024/05/guide-to-using-cohere-ai/#) 上发布了题为 "A Complete Guide to Cohere AI" 的博文。该指南涵盖了 Cohere 企业级 AI 平台的安装、设置和使用，包括在 [Streamlit](https://cohere-guide-blog.streamlit.app) 上提供的演示应用，以便于实际理解。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.analyticsvidhya.com/blog/2024/05/guide-to-using-cohere-ai/#">使用 Cohere AI 的完整指南</a>: 利用 Cohere 的企业级 AI 平台解锁洞察、自动化任务并增强体验。轻松安装、定制和部署。</li><li><a href="https://cohere-guide-blog.streamlit.app">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1240923485853716532)** (41 条消息🔥): 

- **Hugging Face 承诺提供价值 1000 万美元的免费 GPU**: Hugging Face 承诺提供价值 1000 万美元的免费共享 GPU，以帮助开发者创建新的 AI 技术，旨在协助小型开发者、学术界和初创公司。据 [The Verge](https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai) 报道，首席执行官 Clem Delangue 表示，公司的盈利能力和近期融资使这一举措成为可能。

- **报告 Pi 5 安装成功**: 一位成员确认在运行 Ubuntu 的 Pi 5 上成功运行了 OpenInterpreter，未使用本地模型，而是使用 GPT4 执行各种任务。另一位用户表示有兴趣合并项目，并收到了提供 Azure 额度以协助集成的提议。

- **平台技巧与故障排除**: 成员们分享了使用 WSL、虚拟环境和各种 IDE 设置 OpenInterpreter 的技巧。一位用户通过升级 `litellm` 依赖解决了 OpenRouter 上 GPT-4o 的问题，强调了改进 OpenInterpreter 默认设置的潜在领域。

- **活动与直播公告**: 社区受邀参加首届无障碍圆桌会议（Accessibility Round Table）活动，旨在讨论技术对每个人的益处。此外，一位成员宣布在 X 上进行本地开发的直播，鼓励其他人加入。

- **寻求项目合作**: 一位初级全栈 DevOps 工程师寻求帮助，构建一个 "lite 01" AI 助手模块，用于简化日常任务并在工作环境中提供隐蔽协助。该请求强调了对 DevOps 工具和云计算综合资源的需求。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discord.gg/open-interpreter-1146610656779440188?event=1241028896846254141">Discord - 与朋友和社区聊天的新方式</a>: Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持联系。</li><li><a href="https://www.theverge.com/2024/5/16/24156755/hugging-face-celement-delangue-free-shared-gpus-ai">Hugging Face 正在分享价值 1000 万美元的算力，以帮助击败大型 AI 公司</a>: Hugging Face 希望降低开发 AI 应用的门槛。</li><li><a href="https://docs.openinterpreter.com/language-models/hosted-models.">介绍 - Open Interpreter</a>: 未找到描述</li><li><a href="https://www.youtube.com/shorts/dpkzijtXOqw">HoloMat 更新：Jarvis 控制了我的打印机！#engineering #3dprinting #ironman</a>: 未找到描述</li><li><a href="https://denise.ai/">Denise Legacy - 虚拟助手 Denise</a>: Denise 活过来了！Deniise 2.0 即将到来！我们都在等待的时刻已经到来！Denise Legacy 已开放购买！仅需 49.90 美元即可获得 Denise Legacy 终身促销版，含 Cha...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1241023239740723250)** (15 条消息🔥):

- **故障排除：连接问题快速解决**：一位用户最初在连接 App 时遇到问题，但通过提供的示例格式成功解决。另一位成员赞赏了该 App 的美观和“原生（native）”感，并确认它是用 Swift 构建的。

- **Windows 用户的服务器设置技巧**：一位成员询问是应该在 Windows 的 Ubuntu 中运行服务器还是使用 PowerShell。另一位用户分享了他们的设置方法，利用 poetry 运行带有特定参数的 OpenInterpreter，并确保使用了正确的本地 IP 和端口。

- **环境使用说明**：一位新手询问了关于为 OpenInterpreter 使用 Linux VM 的问题。确认这是可行的，并且能与直接在宿主机上运行的 OpenInterpreter 正确交互。

- **GitHub 资源分享**：分享了一个 [GitHub 仓库](https://github.com/Tonylib/o1_for_flutter) 链接，重点介绍了一个与在 Flutter 上运行 O1 相关的项目。讨论内容包括贡献和开发指南。

- **社区项目与求助请求**：一位成员讨论了他们正在构建的 O1 Lite 设备，包含所有零件和 3D 打印外壳。另一位寻求帮助开发用于任务简化和远程协助的 AI 模块的用户，由于预订延迟，向社区寻求支持。

**提到的链接**：<a href="https://github.com/Tonylib/o1_for_flutter">GitHub - Tonylib/o1_for_flutter</a>：通过在 GitHub 上创建账号来为 Tonylib/o1_for_flutter 的开发做出贡献。

  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1240922872139087922)** (6 条消息): 

- **Google DeepMind 关注 Google IO**：一位成员分享了来自 [GoogleDeepMind](https://x.com/GoogleDeepMind/status/1790463259822420239) 的帖子，讨论了他们在 #GoogleIO 上参与的 Project Astra。另一位成员评论道：“Google 确实在发力。”
- **语音 AI 仍有机械感局限**：关于语音 AI 的现状存在争论，一位成员表示“声音有点太机械了”，认为它落后于 GPT-4 的能力。
- **有趣的 AI 语音交互创意**：一位成员分享的 YouTube 短视频讨论了 AI 语音助手的一个新想法，强调了它们中断用户的能力（[YouTube 视频](https://www.youtube.com/shorts/zgUanlLV_OQ)）。一位用户幽默地补充说，这是“错过了一个让它发出哞哞声的机会”。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/GoogleDeepMind/status/1790463259822420239">Google DeepMind (@GoogleDeepMind) 的推文</a>：我们与 Project Astra 一起观看了 #GoogleIO。👀</li><li><a href="https://www.youtube.com/shorts/zgUanlLV_OQ">AI 语音助手的一个有趣创意，我想我还没见过 #voiceai #gpt4o #interruptingcow</a>：未找到描述
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1241160048567124048)** (26 条消息🔥): 

- **RAG 教程中的段错误（Segfault）排查**：一位用户在按照 [RAG 教程](https://future.mozilla.org/news/llamafiles-for-embeddings-in-local-rag-applications/) 查询索引时遇到了段错误。关键日志信息为 *"llama_get_logits_ith: invalid logits id 420, reason: no logits"*；另一位用户建议检查代码库，随后发现这些模型是仅限嵌入（embeddings-only）的。
  
- **Llamafile 嵌入模型说明**：澄清了 [教程](https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile/tree/main) 中链接的嵌入模型无法执行生成任务，这在示例中并没有立即明确。

- **云端部署讨论**：用户讨论了运行 Llamafile 的各种云服务商，倾向于支持 GPU 的服务。[vast.ai](https://vast.ai) 被推荐用于实验和短期工作负载。

- **用于向量搜索的 SQLite 项目**：Alex Garcia 介绍了他的项目 [sqlite-vec](https://github.com/asg017/sqlite-vec)，这是一个用于向量搜索的 SQLite 扩展，旨在将其集成到 Llamafile 中。该项目承诺提供内存和语义搜索等功能，并且已经发布了 Beta 版资源。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://<Your">no title found</a>: no description found</li><li><a href="https://salad.com/">Salad - GPU Cloud | 10k+ GPUs for Generative AI</a>: 节省高达 90% 的云账单。轻松部署 AI/ML 生产模型。每美元可获得多 600% 的图像和 10 倍的推理。立即免费试用 SaladCloud。</li><li><a href="https://fly.io/docs/gpus/gpu-quickstart/">Fly GPUs 快速入门</a>: 来自 Fly.io 团队的文档和指南。</li><li><a href="https://github.com/beetbox/beets/issues/1166#issuecomment-68076160">convert: Character encoding/path issue on Windows · Issue #1166 · beetbox/beets</a>: convert 插件似乎在处理某些字符时存在问题。以下是导入时的 cmd.exe 输出：C:\Users\Michael\Desktop\blink-182\blink-182&gt;beet import . C:\Users\Michael\Desktop\blink...</li><li><a href="https://future.mozilla.org/news">Mozilla Innovation Projects | Recent Articles</a>: no description found</li><li><a href="https://github.com/Mozilla-Ocho/llamafile-rag-example/blob/main/app.sh">llamafile-rag-example/app.sh at main · Mozilla-Ocho/llamafile-rag-example</a>: 通过在 GitHub 上创建账号来为 Mozilla-Ocho/llamafile-rag-example 的开发做出贡献。</li><li><a href="https://github.com/asg017/sqlite-vec">GitHub - asg017/sqlite-vec: Work-in-progress vector search SQLite extension that runs anywhere.</a>: 正在开发中的、可在任何地方运行的向量搜索 SQLite 扩展。 - asg017/sqlite-vec</li><li><a href="https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html">我正在编写一个新的向量搜索 SQLite 扩展</a>: sqlite-vec 是一个新的向量搜索 SQLite 扩展，即将推出！</li><li><a href="https://github.com/asg017/sqlite-vec/releases">Releases · asg017/sqlite-vec</a>: 正在开发中的、可在任何地方运行的向量搜索 SQLite 扩展。 - asg017/sqlite-vec
</li>
</ul>

</div>
  

---



**MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1241552403052892212)** (9 messages🔥): 

- **抢先加入 LLM Fine-Tuning 课程**: 一位成员分享了他们加入 [LLM Fine-Tuning 课程](https://maven.com/parlance-labs/fine-tuning)的热情。该课程承诺提供 LLM 的实战经验，涵盖从训练到部署的主题，并设有关于评估、仪表化（instrumentation）和 Prompt Engineering 的研讨会。
- **对课程内容的质疑**: 另一位成员对该课程表示怀疑，认为由于促销赠品和涉及的专家范围过广，内容可能比较“虚（fluff）”。他们质疑其价值与吸引参与者的营销手段是否匹配。
- **第一周印象褒贬不一**: 参与者对课程第一周的反馈各不相同。有人将其描述为“相当基础”，重点在于寻找 LLM 的用例等入门主题，这可能很大程度上取决于参与者之前的经验。

**提到的链接**: <a href="https://maven.com/parlance-labs/fine-tuning">Mastering LLMs: End-to-End Fine-Tuning and Deployment by Dan Becker and Hamel Husain on Maven</a>: Maven 上历史最畅销的课程！训练、验证并部署你的第一个 Fine-Tuned LLM。

  

---


**MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1241132122107019387)** (7 messages): 

- **使用 MAPIE 获取预测区间**: 一位成员询问了关于实现预测区间（Prediction Intervals）的建议，并分享了 [MAPIE 文档](https://mapie.readthedocs.io/en/latest/)的链接。正在探索该工具在此背景下的效用。

- **Valeriy Manokhin 谈符合预测 (Conformal Predictions)**: 另一位成员推荐了 Valeriy Manokhin 的 [Medium](https://valeman.medium.com/) 专栏以了解符合预测，并指出 Manokhin 偏好 Nixtla，这可能与时间序列数据相关。

- **通过图像修复获取图像 Embedding**: 有人提问关于使用图像修复（Inpainting）或上下文编码来推导图像 Embedding，并将其与掩码语言建模（Masked Language Modeling）进行比较。该方法涉及使用图像的可视部分来预测隐藏部分。

- **多语言实体提取**: 讨论围绕如何使“University of California”和“Universidad de California”等多语言实体具有可比性展开。建议包括使用对比学习（Contrastive Learning）以及为任务添加语言标识符前缀，正如在某些查询和文档编码策略中所见。 

- **应用相关论文的想法**: 一位成员建议应用最近发表在 [arxiv](https://arxiv.org/pdf/2401.12178) 上的一篇论文中的概念，并提到这是他们本周多语言实体提取工作的一部分。

**提到的链接**: <a href="https://mapie.readthedocs.io/en/latest/">MAPIE - Model Agnostic Prediction Interval Estimator &mdash; MAPIE 0.8.3 文档</a>: no description found

  

---

**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1241588106084487168)** (7 messages): 

- **在 comma 设备上运行 YOLO 模型**：一名成员询问是否有人尝试在 comma 设备上运行 YOLO 模型，并提到他们获得的预测延迟约为 **~1000ms**。他们没有提供关于模型版本或所用优化方案的更多细节。

- **sin 函数逼近的多项式阶数限制**：成员们讨论了使用高阶多项式逼近 sine 函数的局限性。一位用户指出，他们正在使用 11 阶多项式，误差约为 **1e-8**，但这无法满足 **1e-12** 误差的测试要求，他们正在考虑尽管有性能顾虑是否仍要增加阶数。

- **多项式逼近中的精度问题**：另一位用户强调，对于 sine 函数，周期性有助于管理精度问题，但警告说在逼近 logarithm 和 exponential 等函数时会出现显著的精度损失。他们建议使用 range reduction 技术来保持精度，但也承认在不增加计算复杂度的前提下满足高精度要求的挑战。
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1241425883756105840)** (4 messages): 

- **关于 tinygrad 中位移操作的问题**：一名成员询问在 tinygrad 中是否有比使用表达式 `x.e(BinaryOps.DIV, 2 ** 16).e(BinaryOps.MUL, 2 ** 16)` 更高效的位移（bitshifting）方法。

- **Metal 编译器中的 for 循环展开**：另一名成员分享了一段代码片段，并询问 Metal 编译器在何处决定展开（unwrap）for 循环。他们展示了为 `Tensor.arange(1, 32)` 生成的 Metal 代码。

- **与 Tensor.arange(1, 33) 生成代码的对比**：同一名成员演示了使用 `Tensor.arange(1, 33)` 代替 32 会导致生成的 Metal 代码显著不同，其中包括使用了 threadgroup 变量和 barriers。

- **令人困惑的魔数 32**：该成员还质疑为什么数字 32 会导致 Metal 编译器产生不同的编译行为，并指出这带来了明显的性能影响。
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1240941859077881896)** (6 messages): 

- **为 Squeak Smalltalk 提供 Claude3 支持**：一位用户提出了将 **Claude3 支持**添加到 **Squeak Smalltalk** 的想法。虽然没有讨论具体的实现细节或益处，但这标志着将先进模型与传统编程环境集成的兴趣日益浓厚。
  
- **GPT-4o 演示语音模式解析**：另一位用户分享到，**GPT-4o 演示中的语音**最初包含在版本 1 的 Voice Mode 中，被称为 **Sky**，并推测它是默认选项。OpenAI 在意识到它无意中与 Scarlett Johansson 的声音相似后暂停了其使用，并用新的女性声音 **Juniper** 取而代之。

- **语音模式中的延迟与模型集成**：一位用户引用了一篇文章，详细说明了之前的 Voice Mode 版本如何使用独立的模型进行转录、处理和音频输出，从而导致延迟问题。**GPT-4o** 现在将这些功能整合在单个模型中，增强了情感表达，尽管这引入了复杂性和潜在的不可预测性（[来源](https://help.openai.com/en/articles/8400625-voice-chat-faq)）。

- **对 AI 复杂性和 Prompt Injection 的担忧**：进一步的讨论集中在 GPT-4o 等先进功能如何带来显著的缺点，例如易受 **Prompt Injection** 的影响。新模型增加的复杂度可能导致不可预测的行为，并有更高概率产生令用户反感的输出，类似于旧系统被新指令覆盖的问题。

- **容错系统中的韧性**：引用斯坦尼斯瓦夫·莱姆（Stainslaw Lem）的《颠倒的进化》（The Upside-Down Evolution），一位用户指出，虽然完全的可靠性是无法实现的（特别是在复杂系统中），但构建具有韧性（resilient）的基础设施是关键。他们强调，随着系统演进得更加容错，新的问题不可避免地会产生，这呼应了莱姆关于“刚跳出油锅，又掉进火坑”的观点。
  

---



**LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1242201665835237448)** (1 messages): 

- **GPT-4o 在复杂法律推理方面表现出色**：一名成员对 **GPT-4o** 在复杂法律推理任务上进行了内部评估，注意到其相比 **GPT-4** 和 **GPT-4-Turbo** 有显著提升。更多细节可以在他们的 [LinkedIn 帖子](https://www.linkedin.com/posts/evan-harris-387375b2_the-release-of-gpt-4o-from-openai-has-been-activity-7196856963454959617-w1i1)中找到。
  

---

**YAIG (a16z Infra) ▷ #[ai-ml](https://discord.com/channels/958905134119784489/1013536071709118565/1241717102315049031)** (1 条消息): 

- **关于 Docker 和 AI 的贡献者招募**：一位成员宣布计划撰写一篇专注于**使用 Docker 容器进行 AI 训练和部署**的文章。他们邀请其他人参与协助、贡献或审阅草案，并请感兴趣的人私信（DM）他们。

---

---