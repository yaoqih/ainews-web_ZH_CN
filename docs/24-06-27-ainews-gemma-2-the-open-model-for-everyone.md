---
companies:
- google-deepmind
- alibaba
- mistral-ai
- anthropic
date: '2024-06-28T06:21:39.390033Z'
description: 由 **Google DeepMind** 推出的 **27B** 参数模型 **Gemma 2** 正式发布。该模型引入了 1:1 局部-全局注意力交替（local-global
  attention alternation）和 Logit 软截断（logit soft-capping）等创新技术，并利用**知识蒸馏**（knowledge
  distillation）在超过计算最优（compute-optimal）Token 数量 50 倍的数据上对较小模型进行了训练。该模型支持多语言和多模态能力，并已在
  200 多种印度语系变体上成功完成了微调。**Open LLM 排行榜**显示，**阿里巴巴的 Qwen 72B** 位居榜首，**Mistral AI 的 Mixtral-8x22B-Instruct**
  同样名列前茅。**Anthropic** 推出了 **Claude 3.5 Sonnet**，在中端成本和速度下提升了智能水平。此外，关于在大语言模型（LLM）中消除矩阵乘法的研究有望在不损失性能的前提下显著节省内存。*Kathleen
  Kenealy* 和 *Daniel Han* 分别就 Gemma 2 的分词器（tokenizer）和注意力缩放（attention scaling）分享了见解。
id: 2c5b54c8-2b22-4462-8e1d-fc3c428eecab
models:
- gemma-2
- qwen-72b
- mixtral-8x22b-instruct
- claude-3.5-sonnet
original_slug: ainews-gemma-2-the-open-model-for-everyone
people:
- kathleen-kenealy
- daniel-han
title: Gemma 2：面向所有人的开放模型
topics:
- knowledge-distillation
- attention-mechanisms
- multilingual-models
- multimodality
- model-training
- model-optimization
- memory-optimization
- fine-tuning
---

<!-- buttondown-editor-mode: plaintext -->**Knowledge Distillation 是解决 Token 危机所需的一切吗？**

> 2024/6/26-2024/6/27 的 AI 新闻。
我们为您检查了 7 个 subreddit、[384 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 30 个 Discord（416 个频道和 2698 条消息）。
预计节省阅读时间（按 200wpm 计算）：317 分钟。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

[Gemma 2 发布了！](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) 在 I/O 大会上进行了预览（[我们的报告](https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/)），现在正式发布了，包含了他们讨论过的 27B 模型，但奇怪的是没有 2B 模型。无论如何，就其规模而言，它当然很出色——在评估中低于 Phi-3，但在 LMSys 的评分中表现更好，仅次于 yi-large（后者也于[周一在 World's Fair Hackathon 上发布](https://x.com/01AI_Yi/status/1805431304999022812)）：

 
![image.png](https://assets.buttondown.email/images/b7f2a5de-6997-4f87-8529-fa486a002b6d.png?w=960&fit=max)
 

关于其驱动因素，我们有一些小提示：

- 局部注意力和全局注意力之间的 1:1 交替（类似于 [Shazeer et al 2024](https://buttondown.email/ainews/archive/ainews-shazeer-et-al-2024/)）
- 参考 Gemini 1.5 和 Grok 的 Logit soft-capping
- GQA, Post/pre rmsnorm

但当然，数据才是关键问题；而这里的故事一直是 KD：

> 特别是，我们将精力集中在 Knowledge Distillation（Hinton 等人，2015 年）上，它**将每个 token 处看到的 one-hot 向量替换为从大模型计算出的潜在下一个 token 的分布**。
>
> 这种方法通常用于通过为较小模型提供更丰富的梯度来缩短其训练时间。在这项工作中，我们转而使用蒸馏技术对大量 token 进行训练，以模拟超出可用 token 数量的训练。具体来说，我们使用一个大型语言模型作为教师，在**超过理论预测的计算最优数量（Hoffmann 等人，2022 年）50 倍以上的 token 量**上训练小模型，即 9B 和 2.6B 模型。除了通过蒸馏训练的模型外，我们还发布了一个为这项工作从头开始训练的 27B 模型。

在她的 [World's Fair 关于 Gemma 2 的演讲](https://youtubetranscript.com/?v=JVSKlEmUr0k&t=12381)中，Gemma 研究员 Kathleen Kenealy 还强调了 Gemini/Gemma 分词器（tokenizer）：

> “虽然 Gemma 主要在英语数据上进行训练，但 Gemini 模型是多模态且多语言的，这意味着 Gemma 模型非常容易适应不同的语言。我们看到的最喜欢的项目之一（在 I/O 大会上也有所强调）是一个印度研究团队对 Gemma 进行了微调，在 200 多种印度语言变体上实现了前所未有的 SOTA 性能。”

同为 World's Fair 演讲者的 Daniel Han 也指出了只有在代码中才能发现的 [attention-scaling](https://x.com/danielhanchen/status/1806372357684220308)：

 
![image.png](https://assets.buttondown.email/images/e54c2aaf-b43c-4a8a-858b-575fd5ccf8c1.png?w=960&fit=max)
 

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要回顾

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中择优。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**AI 模型与架构**

- **新的 Open LLM Leaderboard 发布**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1805989925080219927) 指出新的 Open LLM Leaderboard 评估了**所有主要的开源 LLM**，其中 **Qwen 72B 位居榜首**。之前的评估对于近期的模型来说变得过于简单，这表明 AI 开发者可能过于关注主要评估指标，而牺牲了模型在其他方面的性能。
- **阿里巴巴的 Qwen 模型主导 Open LLM Leaderboard**：[@clefourrier](https://twitter.com/clefourrier/status/1806016524496322950) 强调 **阿里巴巴的 Qwen 模型占据了前 10 名中的 4 个席位**，拥有**最佳的 instruct 和 base 模型**。Mistral AI 的 Mixtral-8x22B-Instruct 位列第 4 名。
- **Anthropic 发布 Claude 3.5 Sonnet**：[@dl_weekly](https://twitter.com/dl_weekly/status/1806094847901659256) 报道称 Anthropic 发布了 **Claude 3.5 Sonnet，以其中端模型速度和成本，提升了智能水平的标准**。
- **消除 LLM 中的矩阵乘法**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1806108390231331260) 分享了一篇关于 **'Scalable MatMul-free Language Modeling'** 的论文，该研究在保持十亿参数规模强劲性能的同时，消除了昂贵的矩阵乘法（matrix multiplications）。与未优化的模型相比，内存消耗可降低 10 倍以上。
- **NV-Embed：改进将 LLM 训练为通用嵌入模型（embedding models）的技术**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1806034882855875029) 重点介绍了 **NVIDIA 的 NV-Embed 模型**，该模型引入了新设计，例如让 LLM 关注潜在向量（latent vectors）以获得更好的池化嵌入输出，以及一种两阶段指令微调方法，以增强在检索和非检索任务上的准确性。

**工具、框架与平台**

- **LangChain 在 LangSmith 中发布自改进评估器**：[@hwchase17](https://twitter.com/hwchase17/status/1806016844266197439) 介绍了一项新的 LangSmith 功能，用于 **从人类反馈中学习的自改进 LLM 评估器**，灵感来自 @sh_reya 的工作。当用户审查和调整 AI 判断时，系统会将这些存储为 few-shot 示例，以自动改进未来的评估。
- **Anthropic 启动 Build with Claude 竞赛**：[@alexalbert__](https://twitter.com/alexalbert__/status/1806040271672766756) 宣布了一项 **3 万美元的竞赛，鼓励通过 Anthropic API 使用 Claude 构建应用**。提交的作品将根据创意、影响力、实用性和实现情况进行评审。
- **Mozilla 发布新的 AI 产品**：[@swyx](https://twitter.com/swyx/status/1806008516597146098) 指出 **Mozilla 带着新的 AI 产品强势回归**，暗示他们可能在浏览器之后成为“AI OS”。
- **Meta 开放 Llama Impact Innovation Awards 申请**：[@AIatMeta](https://twitter.com/AIatMeta/status/1806048204452159848) 宣布开放 **Meta Llama Impact Innovation Awards 的申请，以表彰在各地区使用 Llama 产生社会影响的组织**。
- **Hugging Face Tasksource-DPO-pairs 数据集发布**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1806100283853779166) 分享了 **Hugging Face 上 Tasksource-DPO-pairs 数据集**的发布，其中包含 600 万个经过人工标注或人工验证的 DPO 对，涵盖了许多之前集合中未包含的数据集。

**迷因与幽默**

- [@svpino](https://twitter.com/svpino/status/1806024708410015761) 调侃了**他们期待 AI 取代的事物**，包括 Jira、Scrum、软件估算、“Velocity”暴行、非技术背景的软件经理、Stack Overflow 以及“10 个你不想错过的疯狂 AI 演示”。
- [@nearcyan](https://twitter.com/nearcyan/status/1806106875764801623) 对**日本麦当劳的“薯条小说”**（ポテト小説。。。😋）发表了幽默评论。
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1806060587107164349) 分享了一个关于 **“Perplexity 在 Figma config 2024，由设计负责人 @henrymodis 演讲”** 的迷因。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 进展与能力**

- **低功耗 LLMs**：研究人员开发出一种高性能大语言模型，其运行所需的[**能量仅相当于一个灯泡**](https://news.ucsc.edu/2024/06/matmul-free-llm.html)。这是通过[在 LLM 中消除矩阵乘法](https://arstechnica.com/information-technology/2024/06/researchers-upend-ai-status-quo-by-eliminating-matrix-multiplication-in-llms/)实现的，颠覆了 AI 的现状。

- **AI 自我意识辩论**：Claude 3.5 [通过了镜像测试](https://www.reddit.com/gallery/1dpj0fw)，这是一项经典的动物自我意识测试，引发了关于这是否真实展示了 AI 自我意识的辩论。另一篇关于同一话题的[帖子](https://www.reddit.com/gallery/1dpj4a2)中，评论者对这是否代表真正的自我意识持怀疑态度。

- **AI 表现优于人类**：在一项现实世界的“图灵测试”案例研究中，[**AI 在 83.4% 的情况下表现优于大学生**](https://i.redd.it/eux68vd2b19d1.png)，且 94% 的 AI 提交内容未被检测出非人类创作。然而，根据 [Hugging Face 归一化得分](https://i.redd.it/f9q9a4h2819d1.png)，人类在 MuSR 基准测试中仍然优于 LLM。

- **模型进展迅速**：过去 16 个月 [LLaMA 模型家族的时间线](https://i.redd.it/35e3at3wr19d1.png)展示了正在取得的飞速进展。[Gemma V2 模型在 Lmsys arena 的测试](https://www.reddit.com/r/LocalLLaMA/comments/1dovvbd/gemma_v2_in_the_lmsys_arena/)根据以往模式暗示即将发布。对 [llama.cpp bitnet 的持续改进](https://github.com/ggerganov/llama.cpp/pull/8103)也在进行中。

- **对当前 LLM 智能的怀疑**：尽管取得了进展，Google AI 研究员 Francois Chollet 在最近的播客中[认为当前的 LLM 几乎没有智能](https://www.preposterousuniverse.com/podcast/2024/06/24/280-francois-chollet-on-deep-learning-and-the-meaning-of-intelligence/)，并且是通往 AGI 道路上的一个“出口”。一张《人工智能的神话》（The Myth of Artificial Intelligence）书籍的照片[引发了关于 AI 现状的讨论](https://i.redd.it/vv9by2jrpw8d1.jpeg)。

**迷因与幽默**

- **AI 的挣扎与怪癖**：迷因嘲讽了 AI 的怪癖，例如一个 AI 模型[难以生成一张连贯的图像](https://v.redd.it/5fohm4ft5w8d1)（女孩躺在草地上），以及[冗长的 AI 输出](https://i.redd.it/gj5c7gpsfv8d1.png)。一个迷因[开玩笑说](https://i.redd.it/2lcebugjvx8d1.jpeg)与 AI 进行了最有意义的对话。

- **调侃公司、人物和趋势**：迷因幽默地抨击了 [Anthropic](https://i.redd.it/4m9vvdeuix8d1.png)、[特定的 Reddit 板块](https://i.redd.it/9oqilymzjw8d1.png)以及人们对 AI [谨慎的乐观态度](https://i.redd.it/3haw02qs0x8d1.png)。一首[诗幽默地赞美了](https://i.redd.it/ctuno8r9mz8d1.jpeg)“机器之神”。

**其他 AI 与技术新闻**

- **AI 版权问题**：主要音乐厂牌 [Sony, Universal 和 Warner 正在起诉 AI 音乐初创公司](https://nypost.com/2024/06/24/business/sony-universal-warner-sue-ai-startups-suno-udio-for-infringement/) Suno 和 Udio 侵犯版权。

- **AI 新功能**：OpenAI 已[确认语音模式](https://www.reddit.com/r/OpenAI/comments/1dp9rkl/openai_confirms_voice_mode_will_roll_out_starting/)将从 7 月底开始为其模型推出。一位 Redditor 简要地[展示了访问](https://v.redd.it/n61ymct8qx8d1) GPT-4o 实时语音模式的过程。

- **图像生成进展**：推出了一种名为 [AuraSR（基于 GigaGAN）](https://blog.fal.ai/introducing-aurasr-an-open-reproduction-of-the-gigagan-upscaler-2/) 的新型开源超分辨率放大器。[ResMaster 方法](https://i.redd.it/x45h2la2ty8d1.png)允许扩散模型生成超出其训练分辨率限制的高分辨率图像。

- **生物技术突破**：两篇关于“桥接编辑”（bridge editing）的 [Nature 论文](https://i.redd.it/vaj4yhrmty8d1.jpeg)引发了关注，这是一种新的基因组工程技术。还宣布了一种[实现可编程基因组设计的新机制](https://x.com/pdhsu/status/1805981296276955571)。

- **硬件开发**：一位开发者令人印象深刻地以个人之力为 BitNet LLM [设计了他们自己的微型 ASIC](https://github.com/rejunity/tiny-asic-1_58bit-matrix-mul)。

---

# AI Discord 回顾

> 摘要之摘要的摘要

## Claude 3.5 Sonnet

1. **Google 的 Gemma 2 引起轰动**：

   - **Gemma 2 亮相**：Google 在 [Kaggle 上发布了 Gemma 2](https://www.kaggle.com/models/google/gemma-2)，包含 9B 和 27B 两种尺寸，具有 sliding window attention 和 soft-capping logits 特性。据报道，27B 版本[接近 Llama 3 70B 的性能](https://x.com/reach_vb/status/1806343018640781675)。

   - **评价褒贬不一**：虽然 9B 模型在[初步测试](https://youtu.be/6SLuneidHYw)中表现出色，但 27B 版本[让部分用户感到失望](https://youtu.be/vIKNRiVxWeo)，凸显了模型性能的变动性。

2. **Meta 发布 LLM Compiler**：

   - **面向代码任务的新模型**：Meta [推出了基于 Meta Code Llama 构建的 LLM Compiler 模型](https://x.com/aiatmeta/status/1806361623831171318)，专注于代码优化和编译器能力。这些模型在宽松的许可证下提供，可用于研究和商业用途。

3. **基准测试与排行榜讨论**：

   - **出人意料的排名**：[Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) 上出现了一些知名度较低的模型（如 Yi）名列前茅的情况，引发了多个 Discord 社区关于基准测试饱和以及评估指标的讨论。

4. **AI 开发框架与工具**：

   - **LlamaIndex 的 Multi-Agent 框架**：LlamaIndex [宣布推出 llama-agents](https://twitter.com/llama_index/status/1806116419995844947)，这是一个用于在生产环境中部署 Multi-Agent AI 系统的新框架，具有分布式架构和 HTTP API 通信功能。

   - **Figma AI 免费试用**：[Figma AI 提供一年的免费试用](https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)，允许用户在无需立即付费的情况下探索 AI 驱动的设计工具。

5. **AI 开发的硬件争论**：

   - **GPU 对比**：Discord 服务器上的讨论对比了拥有 48GB VRAM 的 NVIDIA A6000 GPU 与使用多块 RTX 3090 配置的优劣，考虑了 NVLink 连接性和性价比等因素。

   - **散热挑战**：多个社区的用户分享了高功率 GPU 配置的散热经验，报告称即使使用了大量的散热解决方案，仍存在散热问题。

6. **伦理与法律考量**：

   - **对 AI 生成内容的担忧**：一篇关于 [Perplexity AI](https://www.forbes.com/sites/rashishrivastava/2024/06/26/search-startup-perplexity-increasingly-cites-ai-generated-sources/) 引用 AI 生成来源的文章，引发了不同 Discord 服务器上关于信息可靠性和归属权的讨论。

   - **数据排除的伦理**：多个社区辩论了从 AI 训练中排除某些数据类型（例如与儿童相关的）以防止滥用的伦理问题，并与模型多样性和能力的需求之间进行了权衡。

## Claude 3 Opus

**1. LLM 性能与能力的进展**

- **Google 的 Gemma 2** 模型（9B 和 27B）已发布，展示了与 [Meta 的 Llama 3 70B](https://x.com/_philschmid/status/1806343336292229234?s=46) 等更大型模型相比的强劲性能。这些模型具有 sliding window attention 和 logit soft-capping 特性。

- **Meta 的 LLM Compiler** 模型基于 Meta Code Llama 构建，专注于代码优化和编译器任务。这些模型在宽松的许可证下发布，可用于 [研究和商业用途](https://x.com/aiatmeta/status/1806361623831171318?s=46)。

- **Stheno 8B** 是来自 Sao10k 的创意写作和角色扮演模型，现已在 [OpenRouter](https://openrouter.ai/models/sao10k/l3-stheno-8b) 上线，支持 32K context window。

**2. 开源 AI 框架与社区努力**

- **LlamaIndex** 推出了 [llama-agents](https://twitter.com/llama_index/status/1806116419995844947)，这是一个用于在生产环境中部署 multi-agent AI 系统的新框架，并为其全托管的 ingestion 服务 LlamaCloud 开启了候补名单。

- **Axolotl** 项目遇到了 [Transformers 代码影响 Gemma 2 样本打包 (sample packing)](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718) 的问题，引发了 pull request 以及关于 Hugging Face 典型 bug 的讨论。

- **Rig** 是一个 [用于构建 LLM 驱动应用程序的 Rust 库](https://discord.com/channels/954421988141711382/1218409701339828245/1255618654142202156)，现已发布并为开发者提供了激励性的反馈计划。

**3. 优化 LLM 训练与推理**

- 工程师们讨论了 **infinigram ensemble** 技术在提高 LLM 域外（OOD）检测方面的潜力，并引用了一篇关于 [神经网络学习低阶矩 (low-order moments)](https://arxiv.org/abs/2402.04362) 的论文。

- [新论文](https://arxiv.org/abs/2406.16747) 中介绍了 **SPARSEK Attention** 机制，旨在通过稀疏选择机制克服 autoregressive Transformers 中的计算和内存限制。

- **Adam-mini** 是一种声称性能与 AdamW 相当但内存占用显著降低的优化器，在一次 [详细讨论](https://x.com/ericzhang0410/status/1805814432595165567) 中被拿来与 NovoGrad 进行对比。

**4. 多模态 AI 与生成模型创新**

- **Character.AI** 推出了 [Character Calls](https://blog.character.ai/introducing-character-calls/)，允许用户与 AI 角色进行语音通话，尽管该功能在性能和流畅度方面的评价褒贬不一。

- Stability AI 的 Discord 机器人 **Stable Artisan** 集成了 Stable Diffusion 3、Stable Video Diffusion 和 Stable Image Core 等模型，用于 [直接在 Discord 内进行媒体生成和编辑](https://discord.com/channels/1002292111942635562/1002292112739549196/1255599689118777468)。

- [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/) 中提到的 **Phi 3** 模型通过 WebGPU 将强大的 AI 聊天机器人带到了浏览器端。

## GPT4O (gpt-4o-2024-05-13)

1. **LLM 部署与训练优化**：

   - **AI 部署中的障碍让工程师感到沮丧**：工程师们分享了在高效部署自定义模型方面面临的挑战，讨论集中在如何避免权重错误，以及如何使用 [Koboldcpp](https://github.com/LostRuins/koboldcpp) 等工具为 RTX 4090 等硬件优化参数。

   - **深入探讨 Flash Attention**：成员们请求关于 [Flash Attention](https://towardsdatascience.com/flash-attention-fast-and-memory-efficient-exact-attention-with-io-awareness-a-deep-dive-724af489997b) 的教程，这是一种模型内存管理的高效技术，强调了对这种优化方式加深理解的需求。

2. **基准测试与性能评估**：

   - **Yi 席卷 LLM 排行榜**：[Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) 引起了广泛关注，因为像 Yi 这样的模型出人意料地升至前列，促使工程师们重新评估其模型的性能。

   - **Gemma 2 的反响褒贬不一**：围绕 **Gemma 2** 的兴奋与怀疑并存——虽然一些人称赞其创新，但另一些人不确定它是否标志着重大飞跃。与现有模型的比较受到了 [基准测试分析](https://x.com/danielhanchen/status/1806372357684220308) 的推动。

3. **开源 AI 框架与工具**：

   - **LlamaIndex 推出 llama-agents**：[LlamaIndex](https://twitter.com/llama_index/status/1806116419995844947) 宣布了 **llama-agents**，这是一个旨在简化生产部署的多 Agent AI 框架；它包含分布式架构和 HTTP API 通信。

   - **LangChain AI 讨论端点构建**：工程师们分享了构建 LangChain 端点的示例，[文档](https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events) 展示了 `load_qa_chain()` 的正确用法以及如何处理高并发请求。

4. **AI 许可与伦理考量**：

   - **AI 训练伦理引发激烈辩论**：**LAION** 的工程师们审议了伦理训练实践，辩论是否应排除与儿童相关的数据以防止滥用，同时平衡这对模型多样性和正常场景生成的影响。

   - **对 AI 许可模式的怀疑**：围绕通过 [OpenRouter](https://openrouter.ai/models/cohere/command-r/status) 提供的独占 **Command-R** 模型产生了法律和实际层面的担忧，探讨了潜在的许可滥用和合规执行问题。

5. **前沿 AI 模型与创新**：

   - **Meta 发布 LLM Compiler 模型**：[Meta](https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/) 推出了 **Meta LLM Compiler**，专注于代码优化，这些模型构建在广泛的 Token 语料库之上，用于高级编译器任务。

   - **创新的 SPARSEK Attention 机制**：**SPARSEK Attention** 机制承诺以线性复杂度实现高效的长序列处理，正如一份新 [论文](https://arxiv.org/abs/2406.16747) 中详述的那样，旨在克服典型的 Self-Attention 局限性。

6. **杂项**

   - **Mojo 轻松编译并执行模型**：社区成员讨论了 [Mojo 语言的挑战](https://www.modular.com/blog/mojo-vs-rust-is-mojo-faster-than-rust)，强调了对象标识和自引用类型问题，以及对详尽 GitHub 文档的需求。

   - **大模型存储需求揭晓**：在 **Nous Research AI** 中分享的见解讨论了运行 [DeepCoder V2](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF) 等模型所需的硬件，指出高效运行需要大量的 RAM 和 VRAM。

---

# 第 1 部分：Discord 高层摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Yi 登顶 LLM 排行榜**：[新的基准测试](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)将像 Yi 这样此前知名度较低的模型排在了 LLM 排行榜中令人惊讶的高位，引起了 AI 社区的浓厚兴趣。

**Gemma 2 的发布引发兴奋与质疑**：**Gemma 2** 的发布激发了热情和好奇心，特别是围绕它与 Grok 的相似之处。值得注意的是，一篇[剖析 Gemma 2 创新的推文](https://x.com/danielhanchen/status/1806372357684220308)成为了焦点，尽管一些用户质疑这些进步是否标志着相对于之前模型的重大飞跃。

**AI 部署与训练中的障碍**：讨论指出了部署自定义模型时的挑战和解决方案，重点在于避免权重错误。AI 工程师分享了关于使用 **Ollama** 保存和提供模型的见解，并建议针对 RTX 4090 等硬件进行优化的参数调整，提到了 [Koboldcpp](https://github.com/LostRuins/koboldcpp) 等特定工具。

**AI World's Fair 前夕讨论的 Bug 与支持**：Unsloth AI 团队正准备参加 AI World's Fair，计划讨论开源模型问题以及新加入的 **@ollama** 支持，正如在[这条推文](https://x.com/danielhanchen/status/1806051465544536535)中所宣布的那样。

**针对 ChatGPT 的激烈讨论**：ChatGPT 成为一个有争议的话题，一些社区成员称其“简直是彻头彻尾的垃圾”，而另一些人则承认了它在铺平 AI 道路方面的作用，尽管 **ChatGPT 3.5** 存在准确性问题。此外，人们还幽默地感叹了 AI 硬件过热的问题。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**多模态 RAG 即将到来**：围绕一篇多模态 RAG 文章的开发展开了兴奋的讨论，期待能有突破性的成果；然而，文中并未讨论模型或结果等具体细节。

**实体提取工具评估**：技术讨论确定了 BERT 在 NER 方面的缺点，成员们建议使用 GLiNER 和 NuExtract 等替代方案，这些方案因在提取非预定义实体方面的灵活性而受到推崇，并指向了 [ZeroGPU Spaces](https://huggingface.co/spaces/enzostvs/zero-gpu-spaces) 等社区资源。

**对 Sohu AI 芯片持怀疑态度**：社区对 Sohu 新型 AI 芯片声称的性能持谨慎怀疑态度，成员们考虑在 Sohu 广告宣传的服务上进行实验，尽管目前还没有人分享直接的使用经验。

**高效的动态 Diffusion 交付**：成员们热烈交流了增强 Stable Diffusion 模型性能的策略，特别是包括 "torch compile" 以及利用 Accelerate 和 [stable-fast](https://github.com/chengzeyi/stable-fast) 等库来缩短推理时间。

**对 AI 排行榜的反思**：[Open LLM Leaderboard 博客](https://huggingfile.co/spaces/open-llm-leaderboard/blog)引发了对 AI 基准测试饱和的担忧，反映了社区对持续改进和新基准测试的驱动力。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**GPT 兄弟之争**：**CriticGPT** 出现并用于修复 GPT-4 代码中的 Bug，它被集成到 OpenAI 的 RLHF 流水线中以增强 AI 监督，[官方公告详情](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/)。

**Claude vs. GPT-4o - 上下文窗口大对决**：**Claude 3.5 Sonnet** 因其编程能力和广阔的上下文窗口而受到赞誉，使 **GPT-4o** 显得逊色，一些人声称后者缺乏真正的全模态（omnimodal）能力且面临响应速度慢的问题。

**超越传统的文本聊天**：创新者利用 **3.5 Sonnet API** 和 **ElevenLabs API** 来驱动实时对话，挑战了在某些场景下使用 **ChatGPT** 的必要性。

**Prompt Engineering 的难题与陷阱**：用户交流了 Few-shot Prompting 和 Prompt 压缩的方法，关注于使用 YAML/XML 结构化 Prompt 以提高精确度，并尝试使用“Unicode 符号学”来创建 Token 高效的 Prompt。

**探索 API 迷宫**：讨论集中在计算 Prompt 成本、寻找模型训练的知识库示例、使用 GPT 创建 GIF 的挑战、已弃用插件的替代方案，以及 API 在处理某些文字游戏时的吃力表现。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Tensor Cores 倾向于 Transformers**：工程师们注意到，虽然 GPU 上的 **tensor cores** 是通用的，但它们有一种更“专用于 **transformers**”的趋势。成员们对此表示赞同，并讨论了 **tensor cores** 在特定架构之外的广泛适用性。

- **深入探讨 Flash Attention**：有人寻求关于 [Flash Attention](https://towardsdatascience.com/flash-attention-fast-and-memory-efficient-exact-attention-with-io-awareness-a-deep-dive-724af489997b) 的教程，这是一种在模型中实现快速且内存高效的 **attention** 的技术。分享了一篇文章以帮助成员更好地理解这种优化。

- **Triton 中的幂函数（Power Functions）**：关于 **Triton** 语言的讨论集中在实现 *pow functions* 上，最终使用 `libdevice.pow()` 作为权宜之计。建议检查 **Triton** 是否为 pow 实现生成了最优的 PTX 代码，以确保性能效率。

- **PyTorch 优化解析**：新的 [TorchAO 0.3.0 版本](https://github.com/pytorch/ao/releases/tag/v0.3.0) 凭借其 **quantize API** 和 **FP6 dtype** 引起了关注，旨在为 **PyTorch** 用户提供更好的优化选项。同时，澄清了 `choose_qparams_affine()` 函数的行为，并鼓励社区贡献以加强平台。

- **稀疏性（Sparsity）提升训练速度**：在 [xFormers](https://pytorch.org/blog/accelerating-neural-network-training/) 的项目中使用 2:4 稀疏性集成，使得推理速度提升了 10%，训练速度提升了 1.3 倍，这在 NVIDIA A100 上针对 [DINOv2 ViT-L](https://github.com/facebookresearch/dinov2) 等模型得到了验证。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Infinigram 与低阶动量**：讨论强调了使用 **infinigram ensemble** 技术来增强 LLMs 的分布外（OOD）检测的潜力，引用了《神经网络学习日益复杂的统计数据》[论文](https://arxiv.org/abs/2402.04362)，并考虑在神经 LM 训练中集成 **n-gram** 或 **bag of words**。

- **Attention 效率革命**：提出了一种新的 **SPARSEK Attention** 机制，承诺具有线性时间复杂度的更精简计算需求，详见[这篇论文](https://arxiv.org/abs/2406.16747)；同时，根据另一项[最新研究](https://arxiv.org/abs/2406.16793)，**Adam-mini** 被宣传为 AdamW 的一种内存高效替代方案。

- **论文、优化器与 Transformers**：研究人员辩论了 **Transformers** 的最佳层排序，引用了多篇 arXiv 论文，并分享了关于流形假设检验（manifold hypothesis testing）的见解，尽管后者没有提到具体的代码资源。

- **Mozilla 的本地 AI 计划**：更新了 Mozilla 关于**本地 AI** 资助呼吁的消息，并通过快速在线搜索解决了 Discord 邀请链接过期的问题。

- **重构神经元的舞动**：利用 Zipf 定律和 Monte Carlo 方法直接在神经元排列分布上进行训练，具有潜在的效率提升，这为观察神经元权重排序提供了一种新颖视角。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **GPU 对决：A6000 vs. 3090s**：工程师们对比了 **拥有 48GB VRAM 的 NVIDIA A6000 GPU** 与 **四路 3090 配置**。他们指出 A6000 的 NVLink 可以实现 96GB 的合并 VRAM，而一些人则因价格和多 GPU 配置下的性能而更青睐 3090。

- **高性价比 GPU 选择**：讨论了预算型 GPU，建议将 **认证翻新的 P40 和 K80** 作为处理大型模型的切入方案，指出这比 3090 等高端 GPU 能节省大量成本。

- **专用 AI 芯片的局限性**：**Etched 开发的 Sohu 芯片** 因其过于专注于 Transformer 而受到批评，引发了对其适应性的担忧；同时，Nvidia 即将推出的 Transformer 核心被视为潜在的竞争对手。

- **AI 训练伦理与数据范围**：关于是否应在 AI 训练中排除儿童相关数据以防止滥用，展开了激烈的辩论。一些人担心此类排除可能会降低模型的多样性，并阻碍生成家庭场景等非 NSFW 内容的能力。

- **NSFW 数据在基础 AI 中的作用**：NSFW 数据对于基础 AI 模型的必要性受到质疑，结论是它对于预训练并非至关重要，后期训练可以使模型适应特定任务，但在如何伦理地管理这些数据方面存在不同意见。

- **AIW+ 问题复杂性解析**：探讨了解决 AIW+ 问题的挑战（与常识性 AIW 相比），计算堂表亲等家庭关系的复杂性以及细微的可能性导致结论认为，此事仍存在模糊性。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **寻求 AI 模型的内存预测公式**：工程师们正在寻找一种可靠的方法，根据 **Context Window Size** 来预测模型的内存占用，并考虑了 *GGUF 元数据* 以及不同模型在 Attention 机制上的差异。有人提议通过经验测试来进行准确测量，但对现有公式的包容性仍持怀疑态度。

- **Chat GPT API 前端展示**：社区分享了新的 GPT API 前端，包括 [Teknium 的 Prompt-Engineering-Toolkit](https://github.com/teknium1/Prompt-Engineering-Toolkit) 和 [FreeAIChatbot.org](https://freeaichatbot.org)，同时也表达了对使用 Big-AGI 等平台的安全担忧。此外还讨论了 **LibreChat** 和 **HuggingChat** 等替代方案的使用。

- **Meta 和 JinaAI 提升 LLM 能力**：Meta 新推出的模型优化了编译器优化中的代码大小，JinaAI 的 [PE-Rank](https://x.com/JinaAI_/status/1806331488247402524) 降低了重排序（Reranking）延迟，这些都表明了技术的快速进步，部分模型现已在宽松的许可证下发布，可用于实际研究和开发。

- **AI 模型中的布尔值混淆**：强调了 JSON 格式化问题，其中 **Hermes Pro** 返回了 `True` 而不是 `true`，引发了关于数据集完整性以及训练合并（Training Merges）对不同 AI 模型中布尔值有效性潜在影响的辩论。

- **RAG 数据集扩展**：**Glaive-RAG-v1 数据集** 的发布标志着向针对特定用例微调模型迈进。用户讨论了 Hermes RAG 的格式适应性，并考虑在新的领域生成数据以增强数据集多样性，目标理想规模为 5k-20k 个样本。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **MacBook Air：是否能胜任 AI 工作？**：成员们正在辩论配备 6GB 或 8GB RAM 的 MacBook Air 是否适合 AI 任务，并指出对于 Apple 硬件在此类应用中的性能尚未达成共识。
- **LoRA 训练技术探讨**：为了获得更好的 LoRA 模型性能，改变 Batch Size 和 Epoch 是关键；一位成员举例说明，通过 16 Batch Size 和 200 Epoch 的组合，可以在减少细节的同时获得良好的轮廓。
- **Stable Diffusion 许可困扰**：SD3 和 Civitai 模型的许可困境依然存在；成员们讨论了在当前 SD3 许可下禁止此类模型的情况，特别是在 Civit 等商业项目中的应用。
- **Kaggle：研究者的 GPU 避风港**：Kaggle 正在提供两个具有 32GB VRAM 的 T4 GPU，有利于模型训练；GitHub 上一个有用的 [Stable Diffusion Web UI Notebook](https://github.com/DEX-1101/sd-webui-notebook) 已被分享。
- **拯救频道：对过去的诉求**：AI 社区成员表达了恢复充满生成式 AI 讨论的存档频道的愿望，他们非常看重这些频道所提供的专业对话深度和战友情谊。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **图表库权衡导航**：工程师们就最优图表库展开了激烈辩论，考虑了静态与交互式图表、原生与浏览器渲染以及数据输入格式；讨论集中在确定图表库的核心需求上。

- **使用 Docker 进行创意容器化**：针对 Mojo nightly 构建的 Docker 容器引发了对话，社区成员交换了技巧和修正建议，例如使用 `modular install nightly/mojo` 进行安装。此外还宣传了即将举行的 Mojo 社区会议，并提供了视频会议链接。

- **关于 Mojo 语言挑战的见解**：Mojo 讨论中的话题强调了在 GitHub 上报告问题的必要性，解决了来自 Mojo 与 Rust 博客对比中关于对象标识（object identity）的问题，并观察到 Mojo 运行时意外的网络活动，促使建议开启 GitHub issue 以进行进一步调查。

- **Tensor 动荡与 Changelog 澄清**：Mojo 编译器的 nightly 构建版本 `2024.6.2705` 引入了重大变化，例如重新定位了 `tensor` 模块，引发了关于代码依赖影响的讨论。参与者呼吁提供更明确的 changelog，从而得到了改进文档的承诺。

- **关于心灵与机器的哲学思考**：AI 频道的一条独立消息提出了人类心灵的二元性概念，将其分为代表创意部分的“魔法”和代表神经网络方面的“认知”，认为智能驱动行为，而行为在与现实世界交互前会经过认知过程的路由。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Perplexity API：故障排除还是令人困惑？**：用户在调用 Perplexity AI 的 API 时遇到了 **5xx 和 401 错误**，引发了关于需要 *status page*（状态页）和身份验证故障排除的讨论。

**Perplexity 功能愿望清单**：爱好者们剖析了 Perplexity AI 当前的功能，如图像解读，并建议增加 **artifact implementation** 以更好地管理文件。

**顶级 AI 对比**：社区分析并对比了各种 AI 模型，特别是 **GPT-4 Turbo、GPT-4o** 和 **Claude 3.5 Sonnet**；大家表达了各自的偏好，但未达成共识。

**Perplexity 的相关性搜索**：分享的 **Perplexity AI 页面** 显示了对从心理健康到最新操作系统（如 **Android 14** 的性能提升）等各种话题的兴趣。

**处于新闻伦理风口浪尖的 AI**：一篇文章批评 Perplexity 越来越多地引用 **AI 生成的内容**，引发了关于 AI 生成来源的可靠性和隐私的讨论。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**趁热领取 Figma AI**：*Figma AI* 目前免费提供一年，由 [@AustinTByrd](https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 分享；详情可见 [Config2024 讨论串](https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)。

**AI Engineer World Fair 的烦恼**：成员们提到了在 *AI Engineer World Fair* 活动期间遇到的技术困难，从音频问题到屏幕共享不等，并建议通过退出舞台并重新加入等策略来解决问题。

**LangGraph Cloud 正式起飞**：[LangChainAI 发布了 LangGraph Cloud](http://bit.ly/langgraph-cloud-beta-1)，这是一项为弹性 Agent 提供强大基础设施的新服务，但一些工程师质疑此类 Agent 是否需要专门的基础设施。

**会议内容关注**：*AI Engineer YouTube 频道* 是观看 *AI Engineer World Fair* 直播和回顾的首选，包含面向 AI 爱好者的关键研讨会和技术讨论，会议转录内容可在 [Compass 转录网站](https://aie.compasswearable.com)上找到。

**Bee 带来的可穿戴设备更新**：可穿戴技术讨论包括了像 [Bee.computer](https://bee.computer/) 这样的创新产品，它可以执行记录和转录等任务，甚至提供 Apple Watch 应用，预示着流线型、多功能设备的发展趋势。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**LM Studio 缺乏关键功能**：LM Studio 被指出缺乏对**基于文档的训练**或 **RAG 能力**的支持，这强调了社区内对“训练 (train)”一词普遍存在的误解。

**代码模型蓄势待发**：**Claude 3.5 Sonnet** 在 **Poe** 和 **Anthropic** 框架中的代码辅助表现受到称赞，同时人们对 LM Studio 和 llama.cpp 即将支持 **Gemma 2** 充满期待。

**硬件依赖性凸显**：用户讨论了在具有高 RAM 配置的设备上运行 **DeepCoder V2** 的良好性能，但指出由于内存限制，在 **M2 Ultra Mac Studio** 上会出现崩溃。此外，**服务器冷却**和 LM Studio 的 **AVX2 处理器要求**也是硬件相关讨论的话题。

**内存瓶颈与修复**：成员们分享了在 LM Studio 中加载模型时遇到 VRAM 限制的经验，并提供了诸如禁用 GPU offload 和升级到更高 VRAM 的 GPU 以获得更好支持等建议。

**新兴 AI 工具与技术**：关于 [Meta 的新 LLM Compiler 模型](https://go.fb.me/tdd3dw) 以及将 **Mamba-2** 集成到 llama.cpp 的讨论非常热烈，展示了 AI 工具和技术在效率和优化方面的进展。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**无法在 Python 中直接打印流 (Streams)**：一位用户强调，你不能在 Python 中直接打印流对象，并提供了一个显示正确方法的代码片段：遍历流并打印每个 token 的内容。

**正确使用 LangChain 处理相关的用户查询**：讨论了如何使用 LangChain 提高用户查询中的向量相关性，潜在的解决方案包括在聊天记录中保留之前的检索结果，以及使用 `query_knowledge_base("Green light printer problem")` 函数。

**将 LangChain 与 FastAPI 集成并增强检索**：社区成员分享了关于使用 FastAPI 中的 `add_routes` 构建 LangChain 端点的文档和示例，以及优化使用 `load_qa_chain()` 进行服务器端文档提供的经验。

**LangChain Expression Language 的尖端特性**：提供了对 LangChain Expression Language (LCEL) 的见解，强调了异步支持、流式传输、并行执行和重试机制，指出需要全面的文档才能完全理解。

**LangChain 的新工具和案例研究**：值得注意的提到包括引入了 [Merlinn](https://github.com/merlinn-co/merlinn)（一个用于排查生产事故的 AI 机器人）、一个 [ML 系统设计案例研究的 Airtable](https://www.evidentlyai.com/ml-system-design)，以及通过 [ZenGuard AI](https://python.langchain.com/v0.2/docs/integrations/tools/zenguard) 将安全功能集成到 LangChain 中。还重点介绍了一个 YouTube 教程，展示了如何使用 Visual LangChain 创建无代码 Chrome 扩展聊天机器人。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 的新 AI 战士**：[LlamaIndex 宣布](https://twitter.com/llama_index/status/1806116419995844947)推出 **llama-agents**，这是一个新的多 Agent AI 框架，主打分布式架构和 HTTP API 通信。新兴的 LlamaCloud 服务已开始为寻求全托管摄取服务的用户开放候补名单注册。

- **LlamaIndex 的 JsonGate**：工程师们就 LlamaIndex 默认 Readers 映射中排除 JSONReader 一事展开了激烈辩论，最终以[提交添加它的 pull request](https://github.com/run-llama/llama_index/pull/14419) 告终。

- **当 AI 想象力过剩时**：LlamaParse 因其在处理财务文档方面的卓越表现而受到关注，但也因产生幻觉数据而受到审查，这促使官方请求提交文档以进行调试和改进模型。

- **BM25 的重新索引困境**：用户讨论指出，在集成新文档时，BM25 算法需要频繁重新索引，效率较低，因此建议采用替代的稀疏嵌入 (sparse embedding) 方法并关注优化。

- **摄取管道 (Ingestion Pipeline) 变慢**：在 LlamaIndex 的摄取管道中处理大型文档时出现了性能下降，对此提出了一个很有前景的批量节点删除方案以减轻负载。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **API 收入超过 Azure 销售额**：OpenAI 的 API 收入现已超过微软在 Azure 上对其进行的转售收入，正如 Aaron P. Holmes 揭示的一项重大市场转变。详情见 [Aaron 的推文](https://x.com/aaronpholmes/status/1806312654505443347?s=46)。

- **Meta 的新编译器工具**：发布了 **Meta Large Language Model Compiler**，旨在通过基础模型改进编译器优化，该模型处理来自 5460 亿 token 海量语料库的 LLVM-IR 和汇编代码。该工具的介绍和研究可以在 [Meta 的出版物](https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/)中探索。

- **Character Calls - AI 电话功能**：Character.AI 推出了 **Character Calls**，这是一项支持与 AI 角色进行语音交互的新功能。虽然旨在提升用户体验，但首次亮相收到的反馈褒贬不一，详见 [Character.AI 的博客文章](https://blog.character.ai/introducing-character-calls/)。

- **编程面试困境**：工程师们分享了对极具挑战性的面试问题和不明确预期的烦恼，以及一个有趣的案例，涉及 [AndrewCurran 在 Twitter 上](https://x.com/AndrewCurran_/status/1806178276001329373)提到的关于声称在 ChatGPT 中获得了带有音效的高级语音功能的说法。

- **专利论述 - 创新还是抑制？**：社区辩论了专利技术的含义，从 Chain of Thought 提示策略到 Google 未强制执行的 Transformer 架构专利，引发了关于科技领域专利性及法律复杂性的讨论。参考资料包括 [Andrew White 关于提示专利的推文](https://x.com/andrewwhite01/status/1806347002126446736?s=46)。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Stheno 8B 在 OpenRouter 上备受关注**：OpenRouter 推出了由 Sao10k 开发的 **[Stheno 8B 32K](https://openrouter.ai/models/sao10k/l3-stheno-8b)** 作为当前特色模型，为 2023-2024 年提供具有 32K 上下文窗口的创意写作和角色扮演新能力。

- **选择 NVIDIA Nemotron 的技术故障**：用户在不同设备上选择 **NVIDIA Nemotron** 时遇到了不稳定的情况，一些人报告“页面无法工作”错误，而另一些人则体验顺畅。

- **API Key 兼容性查询与无审查 AI 模型讨论**：工程师们探讨了 **OpenRouter API keys** 与期望 OpenAI keys 的应用程序的兼容性，并深入研究了无审查 AI 模型的替代方案，包括 **Cmd-r**、**Euryale 2.1** 以及即将推出的 **Magnum**。

- **Google Gemini API 提供 2M Token 窗口**：开发者们欢迎 **[Gemini 1.5 Pro](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/)** 的消息，它现在提供巨大的 200 万 token 上下文窗口和代码执行能力，旨在优化输入成本管理。

- **在 OpenRouter 中寻求 Anthropic Artifacts 的替代方案**：用户对 Anthropic 的 Artifacts 功能感到好奇，引发了关于 **Sonnet-3.5** 是否有潜力在 OpenRouter 中通过典型的 prompt 方法提供类似代码生成能力的讨论。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**创新 API 策略**：使用 **Cohere API**，OpenRouter 可以在不违反许可协议的情况下进行非商业用途；社区确认 API 的使用规避了非商业限制。

**Command-R 模型引发独占热议**：以其先进的指令遵循能力而闻名的 **Command-R** 模型，目前仅通过 [OpenRouter 向 'I'm All In' 订阅者](https://openrouter.ai/models/cohere/command-r/status)开放，引发了围绕模型可访问性和许可的讨论。

**险些陷入的许可陷阱**：关于 SpicyChat 可能滥用 **Command-R** 许可的辩论随之展开，但成员们得出结论，向 Cohere 支付费用应该可以解决任何许可问题。

**技术故障排除成功**：一位成员在参考了官方 [Cohere 多步工具文档](https://docs.cohere.com/docs/multi-step-tool-use#step-2-ask-model-for-tool-calls-and-send-back-tool-results)后，解决了 Colab 和 PyCharm 上的 **Cohere API 脚本错误**。

**Rust 库发布及奖励计划**：**Rig**，一个旨在构建 LLM 驱动应用程序的新 Rust 库正式发布，同时推出的还有反馈计划，奖励开发者的贡献和想法，并特别提到其与 Cohere 模型的兼容性。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **解码神经网络**：工程师可以参加在旧金山 Block's Mission 办公室举办的为期 **4周** 的免费学习小组，重点研究基于 Andrej Karpathy 系列课程的神经网络。通过此 [Google Form](https://forms.gle/L4u3TMfTs5TjqWpt7) 报名；更多详情请见 [活动页面](https://lu.ma/yzzespyu)。

- **开源模型吸引 Interpreter 爱好者**：Discord 社区讨论了最适合本地部署的开源模型，特别是 **GPT-4o**。讨论内容包括在 **Ollama** 或 **Groq** 硬件支持下的潜在用途。

- **GitHub 政策合规对话**：成员们对一个可能与 GitHub 政策冲突的项目表示担忧，强调了在采取 DMCA 通知等正式行动之前进行公开对话的重要性。

- **Meta 推进 LLM Compiler**：Meta 基于 **Meta Code Llama** 构建的新型 **LLM Compiler** 旨在优化和反汇编代码。详情可见 [研究论文](https://go.fb.me/85zwgy) 和相应的 [HuggingFace 仓库](https://go.fb.me/tdd3dw)。

- **O1 的变化**：最新版本的 O1 不再包含 `--local` 选项，社区正在寻求关于可用模型以及在不同语言（如西班牙语）中使用订阅实用性的明确说明。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **调试警示：NCCL Watchdog 遭遇 CUDA 错误**：工程师指出遇到了涉及 **NCCL watchdog 线程终止** 的 **CUDA error**，并建议启用 `CUDA_LAUNCH_BLOCKING=1` 进行调试，并使用 `TORCH_USE_CUDA_DSA` 进行编译以激活设备端断言。

- **Gemma2 备受关注，Google 表现出色**：社区正在评估 **Google 的 Gemma 2**（9B 和 27B 版本），该模型实现了滑动窗口注意力（sliding window attention）和软截断 Logits（soft-capped logits）等特性，其评分与 **Meta 的 Llama 3 70B** 相当。虽然 **Gemma2:9B** 模型在一次 [早期测试](https://youtu.be/6SLuneidHYw) 中获得了积极反馈，但 **Gemma2:27B** 在初始测试中表现令人失望，正如 [另一个视频](https://youtu.be/vIKNRiVxWeo) 中所讨论的。

- **Meta 发布 LLM Compiler**：**Meta 宣布** 了其基于 **Meta Code Llama** 的 **LLM Compiler** 模型，专为代码优化和编译器任务设计，因其宽松的许可和据报道的最先进结果而引起了关注。

- **Gemma2 对阵 Transformers：第一轮较量**：影响 **Gemma 2** 样本打包（sample packing）的 Transformers 代码技术问题浮出水面，建议通过一个 [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718) 进行修复，并等待 **Hugging Face 团队** 的上游修复。

- **跟我重复，Mistral7B**：据报道，**Mistral7B** 在全量指令微调（full instruction-tuning）期间出现了重复句子或段落的运行异常；鉴于训练数据集中不存在此类模式，该问题令人困惑。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**PyTorch 的崛起被拍成电影**：分享了一部 [官方 PyTorch 纪录片](https://www.youtube.com/watch?v=rgP_LBtaUEc)，记录了 **PyTorch 的开发历程**及其成功背后的工程师们，为 AI 爱好者和专业人士提供了深入见解。

**通用的 Transformer FPGA 设计**：一位频道成员澄清说，他们的 FPGA 设计不针对特定品牌，可以随时 **加载 Huggingface 库中的任何 Transformer 模型**，这对于那些正在评估模型部署硬件选项的人来说是一个值得注意的进展。

**tinygrad 的迭代改进**：将 **SDXL** 与 **tinygrad** 集成的工作正在取得进展，一位贡献者计划在开启 pull request 之前优化功能和性能，这是协作者们关注的焦点。

**Hotz 参加演讲活动**：George Hotz 预定进行一次 **8 分钟的演讲**，具体细节尚未披露，他的追随者或潜在协作者可能会对此感兴趣。

**tinygrad 征集代码优化器**：宣布了一项 500 美元的现金奖励，用于提升 tinygrad [匹配引擎的速度](https://github.com/tinygrad/tinygrad/issues/4878)，这是对开发者参与贡献并协作提高项目效率的公开邀请。

**深入探讨 tinygrad 内部机制**：讨论内容包括请求将 PyTorch 的 MultiheadAttention 移植到 tinygrad 的示例、通过创建 **NOOP backend** 来估算模型训练的 VRAM 需求策略，以及参考 [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes/) 对 **Shapetracker** 高效数据表示能力的解释。这些技术交流对于那些寻求理解或贡献于 tinygrad 内部运作的人至关重要。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Anthropic 宣布 Build-With-Claude 竞赛**：重点介绍了围绕 **Claude** 构建应用程序的竞赛，并引用了 [竞赛详情](https://docs.anthropic.com/en/build-with-claude-contest/overview)。

- **LLM 求职信生成查询**：成员们讨论了如何微调 **语言模型** 以根据简历和职位描述生成求职信，并寻求关于使用测试数据有效衡量模型性能的建议。

- **通过 LLM 模仿社交媒体风格**：有人正在创建一个机器人，使用 Flask 和 Tweepy 进行 Twitter API 交互，以其独特的风格回答查询，并正在寻求关于使用其推文训练模型的指导。

- **Cursor 在学生中获得青睐**：关于使用 **OpenAI 的 Cursor** 还是 **Copilot** 的辩论和建议不断涌现，包括在 Cursor 中集成 Copilot 的新颖想法，并提供了 [在 Cursor 中安装 VSCode 扩展的指南](https://www.cursor.com/how-to-install-extension)。

- **额度分配与协作协助**：用户请求有关账户 **额度分配** 的协助和更新，这暗示了持续的社区支持动态，但未提供具体细节。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1255600222344839341)** (269 条消息🔥🔥): 

- **LLM 排行榜引起轰动**：LLM 排行榜已更新 [新基准测试](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)，在社区中引发了兴奋，包括 Yi 等模型出人意料的高排名。
- **Unsloth 团队前往 AI 世界博览会**：他们计划在活动中讨论“OSS 模型中的 bug”，并展示 Unsloth AI 中新的 @ollama 支持。公告链接可以在 [这里](https://x.com/danielhanchen/status/1806051465544536535) 找到。
- **Apple Silicon 支持不会立即推出**：虽然对 Mac 和 AMD 支持有需求，但 theyruinedelise 澄清说，由于缺乏 Mac 设备，Mac 支持进展缓慢。
- **Gemma 2 引发热议**：在 Google 的 [Gemma 2](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) 在 Kaggle 发布后，引发了广泛讨论。[Hugging Face 4-bit 模型](https://huggingface.co/unsloth/gemma-2-9b-bnb-4bit) 已被迅速上传，以便进行高效微调。
- **Meta 发布 LLM Compiler**：Meta 宣布了一个新的模型家族，提供 [代码优化和编译器能力](https://x.com/aiatmeta/status/1806361623831171318?s=46)。这些模型可以模拟编译器，并已在 [Hugging Face](https://go.fb.me/tdd3dw) 上提供。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: 我们提出了 Adam-mini，这是一种优化器，其性能与 AdamW 相当或更好，但内存占用减少了 45% 到 50%。Adam-mini 通过减少学习率资源来降低内存...</li><li><a href="https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315">Gemma 2 Release - a google Collection</a>: 未找到描述</li><li><a href="https://ai.google.dev/gemma/?utm_source=agd&utm_medium=referral&utm_campaign=blog-june&utm_content=gemma2">未找到标题</a>: 未找到描述</li><li><a href="https://ollama.com/library/gemma2">gemma2</a>: Google Gemma 2 现已推出 9B 和 27B 两个尺寸。</li><li><a href="https://x.com/OfficialLoganK/status/1806342850637918288">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: Gemma 2 来了 ❗️ 现在可以在 AI Studio 中进行测试：https://aistudio.google.com/</li><li><a href="https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3">UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/gemma-2-9b-bnb-4bit">unsloth/gemma-2-9b-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/">Gemini 1.5 Pro 2M context window, code execution capabilities, and Gemma 2 are available today</a>: 未找到描述</li><li><a href="https://www.kaggle.com/models/google/gemma-2">Google | Gemma 2 | Kaggle</a>: Gemma 是来自 Google 的一系列轻量级、最先进的开放模型，基于创建 Gemini 模型时使用的相同研究和技术构建。</li><li><a href="https://arxiv.org/abs/2406.02528">Scalable MatMul-free Language Modeling</a>: 矩阵乘法 (MatMul) 通常在大型语言模型 (LLMs) 的整体计算成本中占据主导地位。随着 LLMs 扩展到更大的嵌入维度和上下文长度，这种成本只会不断增加...</li><li><a href="https://x.com/danielhanchen/status/1806051465544536535">来自 Daniel Han (@danielhanchen) 的推文</a>: 我们今天旧金山时间下午 2:20 将在 @aiDotEngineer 世界博览会 YBB Salon 8 讨论 OSS 模型中的 bug。我们在工作坊中讨论了 Gemma 和 Phi-3 —— 今天是 Llama-3！我们还将展示...</li><li><a href="https://x.com/danielhanchen/status/1806410668285030530">来自 Daniel Han (@danielhanchen) 的推文</a>: 已将预量化的 4bit bitsandbytes 版本上传到 http://huggingface.co/unsloth。下载速度快 4 倍，且在 QLoRA 微调中减少了超过 1GB 的 VRAM 碎片。同时请安装开发版 HF pip...</li><li><a href="https://x.com/aiatmeta/status/1806361623831171318?s=46">来自 AI at Meta (@AIatMeta) 的推文</a>: 今天我们宣布推出 Meta LLM Compiler，这是一系列基于 Meta Code Llama 构建的模型，具有额外的代码优化和编译器能力。这些模型可以模拟编译器，预测最优...</li><li><a href="https://x.com/mindbranches/status/1806370172506091843?s=46">来自 MindBranches (@MindBranches) 的推文</a>: @AIatMeta 完整研究论文摘要：“Meta Large Language Model Compiler: Foundation Models of Compiler Optimization”</li><li><a href="https://github.com/albertan017/LLM4Decompile">GitHub - albertan017/LLM4Decompile: Reverse Engineering: Decompiling Binary Code with Large Language Models</a>: 逆向工程：使用大型语言模型反编译二进制代码 - albertan017/LLM4Decompile</li><li><a href="https://x.com/danielhanchen/status/1806372357684220308">来自 Daniel Han (@danielhanchen) 的推文</a>: 刚刚分析了 Google 新发布的 Gemma 2！9B 和 27B 的 Base 和 Instruct 版本都在这里！1. Pre & Post Layernorms = 像 Grok 一样有 2 倍的 LN。2. 使用了 Grok 的 softcapping！Attn logits 被截断为 (-30, 3...</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - 由 open-llm-leaderboard 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1dpr487/gemma_2_is_live_on_kaggle_27b_9b/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/L">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1255634352457646154)** (17 messages🔥): 

- **ChatGPT 引发分歧意见**：一位成员严厉批评了 ChatGPT，将其描述为 *“简直是垃圾”*。另一位成员表示反对，指出虽然 **ChatGPT 3.5** 缺乏准确性，但它仍然为现代 AI 的进步 *“铺平了道路”*。
- **AI 硬件与散热困扰**：一位用户幽默地分享说，他们的系统由 2x4090 GPU 驱动，功耗约 1000W，让空调不堪重负。另一位成员深有感触，提到尽管使用了多个散热器，仍然会遇到热失控（thermal runaway）。
- **对 Gemma 2 创新的赞赏与困惑**：讨论了一篇分析 Google **Gemma 2 发布**的 [推文](https://x.com/danielhanchen/status/1806372357684220308)，重点介绍了借鉴自 Grok 的几项创新特性。一位用户发现 prepost ln 和 logit softcap 等技巧非常迷人，但质疑是否真的有来自 Grok 的重大突破。
- **传统与现代蒸馏方法**：另一位用户指出，Gemma 2 中使用的 *Knowledge Distillation (KD)* 似乎已经过时，更倾向于“现代蒸馏”方法。他们对 **2 个 perplexity 的差异**印象深刻，称之为 *“😍”*。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=R0X7mPagRiE">AI Engineer World’s Fair 2024 - Open Models track</a>: https://twitter.com/aidotengineer</li><li><a href="https://x.com/danielhanchen/status/1806372357684220308">Daniel Han (@danielhanchen) 的推文</a>: 刚刚分析了 Google 新发布的 Gemma 2！9B 和 27B 的基础版和指令版都在这里！1. Pre & Post Layernorms = 像 Grok 一样多出 2 倍的 LN；2. 使用了 Grok 的 softcapping！Attn logits 被截断至 (-30, 3...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1255625481232056390)** (83 messages🔥🔥): 

- **模型部署：Colab 对比 Ollama**：用户讨论了训练后部署自定义模型的各种选项，并注意到端点上权重错误等未解决问题。一位成员建议使用 [Ollama](https://discord.com/channels/1179035537009545276/1249414095359312087) 进行部署，正如现有 notebook 中演示的那样。
- **本地模型保存与部署问题**：成员们提供了在 Google Colab 中保存微调模型的指导，包括将其传输到 Google Drive，以及使用 [Koboldcpp](https://github.com/LostRuins/koboldcpp) 等工具在本地运行 GGUF 模型。有人指出，仍在 RAM 中的模型可以使用 `model.save_pretrained("lora_model")` 进行保存。
- **性能优化的训练细节**：一位用户寻求在 RTX 4090 上运行 Unsloth 的最佳训练设置，建议包括调整 batch sizes、learning rates，并考虑像 Qwen 32b 这样更高参数的模型。为了更好地处理瑞典语等语言，建议在训练配置中使用 embed tokens 和 lm_head。
- **VRAM 与微调讨论**：讨论涵盖了微调 lm_head 和 embed tokens 时显著的 VRAM 占用，并论证了其在特定语言训练中的必要性。一位用户链接到了一个[为韩语训练 Mistral 的 notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)，强调了其复杂性和要求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1xg3xz0J6BZPkUh8sor03_JrLHLKVGp-U?usp=sharing">Google Colab</a>: 无描述</li><li><a href="https://github.com/LostRuins/koboldcpp">GitHub - LostRuins/koboldcpp: 使用 KoboldAI UI 运行各种 GGML 和 GGUF 模型的简单单文件方法</a>: 使用 KoboldAI UI 运行各种 GGML 和 GGUF 模型的简单单文件方法 - LostRuins/koboldcpp</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: 无描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1255816018543120414)** (2 messages): 

- **请求上传 4bit 模型**：一位成员请求上传 [GLM-4-9B](https://huggingface.co/THUDM/glm-4-9b/blob/main/README_en.md) 的 **4bit bnb 版本**。他们强调了该模型在各种 benchmarks 中的卓越表现，以及多语言支持和扩展上下文长度等高级特性。
- **用户对新模型感兴趣**：另一位成员对 **GLM-4-9B** 模型表示出兴趣，说道：*“噢，这是什么？看起来挺有意思的。”*



**提到的链接**：<a href="https://huggingface.co/THUDM/glm-4-9b/blob/main/README_en.md">README_en.md · THUDM/glm-4-9b at main</a>: 无描述

  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1255598480194408561)** (238 messages🔥🔥): 

- **对多模态 RAG 文章的期待**：一位成员透露他们正在撰写一篇多模态 RAG 文章，引发了其他人的期待，认为这将是“史诗级”的作品。*"it's gonna be epic ✨"*

- **对实体提取和 GraphDB 的兴趣**：成员们讨论了各种实体提取工具，fulx69 分享了他使用 GraphDB 的方法，并表达了对用于 NER 的 BERT 的失望。Cursorop 推荐并讨论了使用 GLiNER 和 NuExtract，因为它们在提取非预定义实体方面具有灵活性。

- **HuggingFace 免费 GPU 访问资源**：一场关于与 HuggingFace 合作以获取免费 GPU 访问权限的讨论展开，vipitis 提到了他们的 ZeroGPU 计划并链接到了 [ZeroGPU Spaces](https://huggingface.co/spaces/enzostvs/zero-gpu-spaces)。Fulx69 还指出，通过 Colab, Kaggle, AWS 等平台获得资金支持可能是必要的。

- **对 Sohu AI 芯片性能的质疑**：用户对一篇详细的 [Etched 博客文章](https://www.etched.com/announcing-etched) 中描述的新型 Sohu AI 芯片令人印象深刻的性能主张表示怀疑。尽管存在疑虑，一些人仍表现出申请其云服务的兴趣。

- **加速 Stable Diffusion 推理的技巧**：社区成员分享了提高 Stable Diffusion 模型推理时间的各种策略，推荐了 Accelerate 和 stable-fast 等库。Welltoobado 建议使用 “torch compile” 方法，并链接到了 [stable-fast GitHub 仓库](https://github.com/chengzeyi/stable-fast)。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/models/google/gemma-2">Google | Gemma 2 | Kaggle</a>: Gemma 是来自 Google 的一系列轻量级、最先进的开放模型，基于用于创建 Gemini 模型的相同研究和技术构建。</li><li><a href="https://hexdocs.pm/fss/0.1.1/FSS.html">FSS — fss v0.1.1</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/numind/NuExtract">NuExtract - numind 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/urchade/gliner_multiv2.1">GLiNER-Multiv2.1 - urchade 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/cat-look-cat-look-at-camera-silly-cat-in-a-cage-gif-889392959852579879">Cat Look GIF - 猫看镜头 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://filesystem-spec.readthedocs.io/en/latest/">fsspec: Python 的文件系统接口 — fsspec 2024.6.0.post1+g8be9763.d20240613 文档</a>: 未找到描述</li><li><a href="https://huggingface.co/numind/NuExtract-large">numind/NuExtract-large · Hugging Face</a>: 未找到描述</li><li><a href="https://www.etched.com/announcing-etched">Etched 正在进行 AI 领域最大的豪赌</a>: 未找到描述</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - 疑惑猫 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU 探索者)</a>: 未找到描述</li><li><a href="https://tenor.com/view/beach-vacation-artem-gif-26266521">Beach Vacation GIF - 海滩度假 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://lu.ma/yzzespyu">AI Study Group @ Block: Andrej Karpathy 的 Zero to GPT Hero · Luma</a>: ______ 报名参加此活动时，系统将通过电子邮件要求您通过以下方式加入学习小组……</li><li><a href="https://tenor.com/view/joke-missed-over-your-head-gif-8604199">Joke Missed GIF - 没接住梗 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/thanos-memoji-gif-23490017">Thanos Memoji GIF - 灭霸 Memoji - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://fxtwitter.com/etched/status/1805625693113663834?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">来自 Etched (@Etched) 的推文</a>: 认识一下 Sohu，有史以来最快的 AI 芯片。在运行 Llama 70B 时，Sohu 每秒可处理超过 500,000 个 Token，让您能够构建在 GPU 上无法实现的产品。一台 8xSohu 服务器可替代 160 台 H100。Soh...</li><li><a href="https://huggingface.co/docs/hub/en/api#get-apidatasetsrepoidparquetconfigsplitnparquet">Hub API 端点</a>: 未找到描述</li><li><a href="https://huggingface.co/urchade/gliner_base">urchade/gliner_base · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/aiatmeta/status/1806361623831171318?s=46">来自 AI at Meta (@AIatMeta) 的推文</a>: 今天我们发布了 Meta LLM Compiler，这是一系列基于 Meta Code Llama 构建的模型，具有额外的代码优化和编译器能力。这些模型可以模拟编译器，预测最优...</li><li><a href="https://x.com/mindbranches/status/1806370172506091843?s=46">来自 MindBranches (@MindBranches) 的推文</a>: @AIatMeta 完整研究论文摘要：“Meta Large Language Model Compiler: 编译器优化的基础模型”</li><li><a href="https://github.com/chengzeyi/stable-fast">GitHub - chengzeyi/stable-fast: 针对 NVIDIA GPU 上 HuggingFace Diffusers 的最佳推理性能优化框架。</a>: 针对 NVIDIA GPU 上 HuggingFace Diffusers 的最佳推理性能优化框架。 - chengzeyi/stable-fast</li><li><a href="https://tenor.com/view/omg-wat-dafuq-huh-wth-gif-9101314">Omg Wat GIF - 我的天哪 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://forms.gle/u397YGMioNFjvWXq6">Sohu 开发者云申请</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

neuralink: 不，我没在休息，我只是没发帖
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1255638555766821041)** (3 条消息): 

- **深入探讨 ML 中的自相似性**：一位成员分享了来自 [Papers We Love](https://github.com/papers-we-love/papers-we-love/blob/main/machine_learning/General-self-similarity--an-overview.pdf) 仓库的概述 PDF。该文档探讨了 Machine Learning 概念和应用中的通用自相似性。

- **令人惊叹的液态神经网络 (LNNs)**：另一位成员重点介绍了关于 [液态神经网络 (LNNs) 的 Medium 文章](https://medium.com/@hession520/liquid-neural-nets-lnns-32ce1bfb045a)，将其描述为适用于时间序列预测的动态自适应神经网络。LNNs 以其在噪声环境下的鲁棒性以及在初始训练后仍能持续适应的能力而闻名。

- **深入的预算报告分析**：一位成员分享了托管在 GitHub 上的 [预算报告论文](https://github.com/alidenewade/Publications/blob/main/Budget%20Speech%20Essay%20Final.pdf) 链接。该论文对预算报告进行了详细分析，为公共财政管理的广泛讨论做出了贡献。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@hession520/liquid-neural-nets-lnns-32ce1bfb045a">Liquid Neural Nets (LNNs)</a>：深入探讨液态神经网络，这是时间序列预测领域近期最令人兴奋的发展之一。</li><li><a href="https://github.com/papers-we-love/papers-we-love/blob/main/machine_learning/General-self-similarity--an-overview.pdf">papers-we-love/machine_learning/General-self-similarity--an-overview.pdf at main · papers-we-love/papers-we-love</a>：来自计算机科学社区的论文，供阅读和讨论。 - papers-we-love/papers-we-love</li><li><a href="https://github.com/alidenewade/Publications/blob/main/Budget%20Speech%20Essay%20Final.pdf">Publications/Budget Speech Essay Final.pdf at main · alidenewade/Publications</a>：通过在 GitHub 上创建账号来为 alidenewade/Publications 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1255633499558383738)** (12 条消息🔥): 

- **PixUP Upscale 加速图像增强**：[PixUP-Upscale](https://github.com/U-C4N/PixUP-Upscale) 项目采用超快且 CPU 友好的设计，用于快速提升图像质量。该项目托管在 GitHub 上，包含详细说明并欢迎贡献。

- **VoiceChat-AI 实现本地和云端 AI 对话**：由 bigsk1 开发的 [voice-chat-ai](https://github.com/bigsk1/voice-chat-ai) 项目允许用户与 AI 对话，既可以使用 ollama 在本地运行，也可以通过 OpenAI 和 ElevenLabs 等云服务运行。

- **Fast Whisper Server 提升转录速度**：一个利用 [Fast Whisper](https://github.com/SYSTRAN/faster-whisper) 的 Space 已部署，它提供了 Whisper 的更快变体用于音频转录，并使用与 OpenAI 相同的 API。详情请查看 [Faster Whisper Server](https://github.com/fedirz/faster-whisper-server)。

- **SimpleTuner 获得压缩升级**：SimpleTuner 的 [v0.9.7.5](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.7.5) 版本包含了 EMA offload 改进和 T5 embeds 的磁盘压缩等更新，提升了效率和存储表现。

- **法语深度学习课程重大更新**：Ricto3092 更新了法语 Deep Learning 课程，增加了关于 YOLO、对比训练、RNNs、GPT 实现等新内容。该课程可在 [GitHub](https://github.com/SimonThomine/CoursDeepLearning) 上获取，并欢迎反馈和贡献。

- **Gemma2 模型测试发布在 YouTube**：Volko76 分享了测试 [Gemma2:9B](https://youtu.be/6SLuneidHYw) 和 [Gemma2:27B](https://youtu.be/vIKNRiVxWeo) 模型的 YouTube 视频，展示了它们强大的性能。

- **开源设备端转录应用**：Hugo Duprez 使用 Ratchet 开发的 [设备端转录应用](https://github.com/Hugo-Dz/on-device-transcription) 现已开源。该应用基于 Svelte 和 Electron 构建，可实现极简且高效的语音转文本。

- **BLAST-SummarAIzer 助力生物信息学研究**：Astrabert 介绍了一个名为 [BLAST-SummarAIzer](https://huggingface.co/spaces/as-cle-bert/BLAST-SummarAIzer) 的新 Space，用于对 16S rRNA 细菌序列运行 BLAST 并使用 LLMs 总结结果。其旨在为研究人员简化复杂的 BLAST 搜索结果解读。

- **Flight-Radar 多语言追踪航班**：Deuz_ai_80619 分享了一个名为 [Flight-Radar](https://github.com/U-C4N/Flight-Radar) 的多语言实时航班追踪应用，使用 Flask 和 JavaScript 构建，并调用了 OpenSky Network API。功能包括地理定位、可调节的搜索半径以及将航班数据下载为 JPG。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/as-cle-bert/BLAST-SummarAIzer">BLAST SummarAIzer - as-cle-bert 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://youtu.be/6SLuneidHYw">Gemma2 首次测试！令人惊叹的结果</a>: 今天，我们将使用 ollama 安装并测试 Gemma2</li><li><a href="https://youtu.be/vIKNRiVxWeo">Gemma2:27B 首次测试！令人惊叹的结果</a>: 让我们用 ollama 测试 Google 一小时前发布的 gemma2 最大版本 (27B)</li><li><a href="https://github.com/bghira/SimpleTuner/releases/tag/v0.9.7.5">Release v0.9.7.5 - embed 缓存压缩 · bghira/SimpleTuner</a>: 更新内容：ema: 卸载至 cpu，每 n 步更新一次（由 @bghira 在 #517 提交）；ema: 正确移动（由 @bghira 在 #520 提交）；EMA: 重构以支持 CPU 卸载、跳步和 DiT 模型；pixart: 减少...</li><li><a href="https://github.com/U-C4N/PixUP-Upscale/">GitHub - U-C4N/PixUP-Upscale</a>: 通过在 GitHub 上创建账号来为 U-C4N/PixUP-Upscale 的开发做出贡献。</li><li><a href="https://github.com/Hugo-Dz/on-device-transcription">GitHub - Hugo-Dz/on-device-transcription: 一个开箱即用的极简应用，可将任何语音转换为文本。</a>: 一个开箱即用的极简应用，可将任何语音转换为文本。 - Hugo-Dz/on-device-transcription</li><li><a href="https://github.com/U-C4N/Flight-Radar">GitHub - U-C4N/Flight-Radar: 一个使用 OpenSky Network API 的多语言实时航班追踪 Web 应用程序。使用 Flask 和 JavaScript 构建，允许用户查看附近的航班、调整搜索半径，并支持六种语言。功能包括地理定位以及将航班数据下载为 JPG 的能力。</a>: 一个使用 OpenSky Network API 的多语言实时航班追踪 Web 应用程序。使用 Flask 和 JavaScript 构建，允许用户查看附近的航班、调整搜索半径，并支持六...</li><li><a href="https://github.com/bigsk1/voice-chat-ai">GitHub - bigsk1/voice-chat-ai: 🎙️ 与 AI 对话 - 使用 ollama 本地运行或使用 OpenAI - 支持 XTTS、OpenAI Speech 或 ElevenLabs</a>: 🎙️ 与 AI 对话 - 使用 ollama 本地运行或使用 OpenAI - 支持 XTTS、OpenAI Speech 或 ElevenLabs - bigsk1/voice-chat-ai</li><li><a href="https://huggingface.co/spaces/Iatalking/fast-whisper-server">Fastwhisper - Iatalking 开发的 Hugging Face Space</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1255616368703701102)** (1 条消息): 

- **排行榜对饱和状态的担忧**：正如链接的博客文章所强调的，排行榜正面临饱和问题。讨论引用了 [Open LLM Leaderboard 博客](https://huggingface.co/spaces/open-llm-leaderboard/blog)。
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1255604983022489650)** (2 条消息): 

```html
<ul>
    <li><strong>充满热情的公开小组启动</strong>：一位成员以充满活力的“**Ghost!**”开启了讨论。这似乎鼓励了频道内的进一步互动。</li>
    <li><strong>对使用效果的好奇</strong>：另一位成员 hayden_85058 问道：“*你觉得使用它的效果如何？*”。这表明了对使用特定工具或方法的实际结果或经验的兴趣。</li>
</ul>
```
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1255931160765268049)** (1 条消息): 

- **OpenAI 推出用于 Bug 检测的 CriticGPT**：一项公告披露了新模型 **CriticGPT** 的训练情况，该模型旨在发现 GPT-4 代码中的 Bug。他们正在将这些模型集成到其 RLHF 对齐流水线中，旨在协助人类在具有挑战性的任务上监督 AI。更多详情请参阅[此处](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/)。
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1255602967319416842)** (97 条消息🔥🔥): 

- **Claude 3.5 Sonnet 在编程方面表现出色，GPT-4o 稍显逊色**：成员们分享了对 **Claude 3.5 Sonnet** 在编程中表现的兴奋之情，有人表示与需要多次修正的 **GPT-4o** 相比，它能“一次就写对”。另一位成员指出，**Claude** 提供了显著更大的 Context Window（上下文窗口），使其更适合大型项目。

- **GPT-4o 全模态（Omnimodal）混淆引发争论**：关于 **GPT-4o** 是否是真正的全模态存在分歧，一些成员认为它是，但禁用了一些功能，而另一些人则认为它仅在“纸面上”是全模态。一位成员提到，一名 **OpenAI** 员工在 Twitter 上澄清过该模型禁用了某些输入/输出。

- **实时对话绕过 ChatGPT**：一位成员声称，利用 **3.5 Sonnet API** 和 **ElevenLabs API** 等 API 可以实现实时对话，并暗示在某些用例中可能不再需要 **ChatGPT**。

- **OpenAI 严格的过滤器影响功能**：用户对 **OpenAI** 严格的过滤器和监管表示沮丧，认为这减慢了处理速度并影响了功能。一位成员提到，“他们似乎不断地用严格的过滤器和监管搬起石头砸自己的脚”。

- **测试 AI 边界导致封号**：讨论了测试 AI 规避手段极限的话题，并警告这样做可能会导致封号。引用了来自 [OpenAI](https://openai.com/policies/usage-policies) 的官方政策，声明规避安全防护措施可能会导致账号暂停或终止。
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1255615035405766748)** (25 条消息🔥): 

- **GPT-4 响应速度问题**：多位用户对 GPT-4 响应缓慢表示担忧。一位用户明确问道：*“GPT-4 出问题了吗？因为它的响应非常慢！”*。

- **准确计算 Prompt 成本**：一位用户询问了在 LLM 应用中计算 Prompt 成本的行业标准，提到他们目前是根据单个用户的输入来汇总所有 Token 成本。他们对是否应该对多个用户进行采样并平均成本表示不确定。

- **寻求 GPTs 的知识库示例**：一位用户请求提供可上传用于 GPT 训练的综合知识库示例，提到他们想为客户创建一个。另一位用户回答说，每个 GPT 都会根据创建者的需求使用不同的知识库。

- **使用 GPT 生成 GIF**：用户讨论了使用 GPT 生成合适的动画 GIF 的困难，一位用户表示很难在不同帧之间保持相同的角色。另一位用户表示在进一步实验后会分享 Prompt 思路。

- **已弃用的插件被 GPTs 取代**：关于在单个对话中使用多个插件功能的查询得到了澄清，即插件已被弃用并由 GPTs 取代。用户不再是在一个对话中启动多个 GPT，而是可以使用 `@mention` 功能在同一个对话中调用不同的 GPTs。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1255676134884900874)** (53 条消息🔥): 

- **寻求关于带工具的 ChatGPT Few-shot Prompting 建议**：一位用户询问了改进 ChatGPT Few-shot Prompting 的方法，特别是针对自定义工具使用的示例。他们目前将工具使用描述为 Function Calls（函数调用），并正在寻找更好的方法。

- **剖析 LLM Prompt Engineering 的内部机制**：一位用户表达了对学习 Prompt Engineering 内部机制以及使用纯数学创建 Prompt 的好奇。他们强调从数字 Token 开始，并表示难以理解基于语言的 Prompt。

- **提供 Meta-prompting 指导**：另一位成员提供了一个结构化模板，使用 YAML, XML 或 JSON 构建 Prompt 以管理层级注意力（Hierarchical Attention），建议用户引导 AI 填充这些模板。

- **语言方法与数学方法的博弈**：随后进行了关于用数字生成 Prompt 的广泛讨论，一位用户坚持要理解数学方法。另一位用户则建议，与 AI 的有效沟通从根本上需要使用自然语言。

- **Prompt 压缩的探索**：一位用户引入了 Prompt 压缩的话题，询问经验和参考资料。讨论重点提到了 “Unicode Semiotics”（Unicode 符号学），它虽然增加了 Token 数量但使用了更少的字符，尽管缺乏广泛的文档，但在 In-context Learning（上下文学习）中很有用。
  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1255676134884900874)** (53 条消息🔥): 

- **ChatGPT 工具使用的 Few-shot 提示策略**：一位用户寻求关于通过使用类似 `func(key1=value1, key2=value2)` 的函数调用来进行 Few-shot 提示，从而提高 ChatGPT 性能的建议。他们想知道是否有更好的方法来实现这一目标。
- **从基础理解 Prompt Engineering**：一位用户请求指导如何从纯数学和逻辑的角度从零开始学习 Prompt Engineering，表达了对复杂结构的困惑，并寻求确定性过程的示例。
- **高效提示的模板**：一位用户分享了一个详细的 YAML/XML 提示模板，用于在提示中实现更好的层级和控制，强调了其对 ChatGPT 的有效性，并建议让 AI 协助构建提示。
- **提示压缩与 Unicode 符号学**：讨论了通过 Unicode 符号学进行提示压缩的方法，指出它虽然使用的字符更少，但消耗的 Token 更多，目前尚无相关论文对此进行解释。尽管 Token 成本较高，该方法仍有助于 In-context learning。
- **API 在单词拼图上的困境**：一位用户分享了他们在让 API 解决乱序单词游戏（如将 "edtPto lumAli" 转换为 "Potted Allium"）时的困难，并提到多次尝试提示均告失败。
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1255651086874972261)** (11 条消息🔥): 

- **关于 Tensor Core 专业化的辩论**：一场关于 Tensor Core 通用性的讨论展开了，一位成员指出“*他们指的可能是‘比这更专注于 Transformer’*”。其他人表示赞同，强调 GPU Tensor Core 仍然是通用的。
  
- **PyTorch 与 NVIDIA 在结构化稀疏性上的不一致**：分享的一条推文强调了 PyTorch 关于结构化稀疏性的新帖子与 NVIDIA 的 PTX 文档之间的一致性问题。用户对 *潜在的 4:8 稀疏性支持* 表达了兴趣。

- **寻求 PyTorch 会议折扣**：一位成员询问是否有人能提供 Meta 的 PyTorch 会议折扣，因为 *600 美元有点太贵了*。

- **Flash Attention 教程请求**：一位成员询问如何从零开始学习 Flash Attention；另一位成员推荐了一篇 [Towards Data Science 的文章](https://towardsdatascience.com/flash-attention-fast-and-memory-efficient-exact-attention-with-io-awareness-a-deep-dive-724af489997b)，该文章深入探讨了这种功耗优化技术。

- **GPU 在 AI 与通用用途中的专一性**：成员们讨论了 GPU 的多功能性，其中一人评论道：“*MM（矩阵乘法）足够通用，可以在其他地方发挥作用*”，另一人指出，由于晶体管限制， GPU 的复杂性并不总是能转化为效率的提升。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/trevorycai/status/1806025579222958317">Trevor Cai (@trevorycai) 的推文</a>：PyTorch 关于结构化稀疏性的新帖子与 NVIDIA 的 PTX 文档之间存在奇怪的不一致。如果能支持 4:8 稀疏性就太酷了！</li><li><a href="https://towardsdatascience.com/flash-attention-fast-and-memory-efficient-exact-attention-with-io-awareness-a-deep-dive-724af489997b">Flash attention (快速且内存高效的具有 IO 感知能力的精确注意力机制)：深度解析</a>：Flash attention 是一种功耗优化的 Transformer 注意力机制，可提供 15% 的效率提升。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1255625274389954710)** (46 messages🔥): 

- **新成员为 pow 实现找到解决方案**：一位新成员分享了一个关于在 Triton 中添加 pow 函数的 [issue](https://github.com/triton-lang/triton/issues/4190)，随后找到了使用 `libdevice.pow(x,x+1)` 的变通方法。他们对社区的帮助表示感谢，并表示该 issue 现已关闭。

- **Triton 缺少原生 pow 但有替代方案**：成员们讨论了 Triton 缺失 pow 函数的问题，建议使用 `exp` 和 `log` 函数来模拟它。其中一个建议是：“你可能可以通过乘法来搞定”。

- **CUDA pow 核函数会被编译为 exp 和 log**：有人指出，在使用 fast math 时，CUDA pow 核函数会编译为包含 `exp` 和 `log` 指令的序列。在没有 fast math 的情况下，为 pow 生成的代码更复杂，这意味着在精度和性能之间存在权衡。

- **深度学习中 pow 的精度考量**：对话强调，虽然 pow 的 exp+log 模拟精度较低，但这种不精确性在深度学习语境下通常是可以接受的。“exp + log 的方式虽然不精确，但对于深度学习来说并不重要” 总结了这一观点。

- **建议验证 Triton 生成的 PTX 代码**：为了确保最佳性能，建议验证 Triton 是否生成了“快速”版本的 pow 代码。原帖作者同意进行检查，并指出 Triton 目前使用的是 CUDA 较慢的 pow 实现。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developer.download.nvidia.com/cg/pow.html">pow</a>: 未找到描述</li><li><a href="https://triton-lang.org/main/python-api/triton.language.html">triton.language &mdash; Triton 文档</a>: 未找到描述</li><li><a href="https://github.com/triton-lang/triton/issues/4190">如何在 python.triton.language.core 中添加 pow 函数？ · Issue #4190 · triton-lang/triton</a>: 我尝试在 triton.jitted 函数中使用 pow 操作：output = x + y**3 ^ 但得到了 AttributeError(&quot;&#39;tensor&#39; object has no attribute &#39;__pow__&#39;&quot;)。在文件 python/trit...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

gau.nernst: https://github.com/efeslab/Atom
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1255630374743052378)** (16 messages🔥): 

- **TorchAO 0.3.0 发布**：查看新的 [TorchAO 0.3.0 版本](https://github.com/pytorch/ao/releases/tag/v0.3.0)，其中包含大量新功能，如新的 quantize API、MX 格式和 FP6 dtype。此版本旨在为 PyTorch 提供更好的性能和优化。

- **讨论 `choose_qparams_affine()` 的行为**：有一个关于当 block size 等于维度时，`choose_qparams_affine()` 返回较少维度的问题。正如 [源代码](https://github.com/pytorch/ao/blob/c2f9b84604536a72804787001c1b63daae792ee9/torchao/quantization/quant_primitives.py#L335) 中详细说明的那样，这被确认是故意的。

- **欢迎新成员贡献**：鼓励社区新成员从使用项目、识别潜在改进和建议更改开始。例如 [danielpatrickhug](https://github.com/pytorch/pytorch/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen) 计划使用自定义静态分析工具。

- **`TORCH_LOGS="output_code"` 的问题**：有人提出了一个问题，即在某些情况下（特别是在添加量化之后）使用 `TORCH_LOGS="output_code"` 不会输出代码。敦促用户报告这些问题并提供最小可复现片段以协助排查。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/blob/c2f9b84604536a72804787001c1b63daae792ee9/torchao/quantization/quant_primitives.py#L335">ao/torchao/quantization/quant_primitives.py · pytorch/ao</a>: 创建并集成自定义数据类型、布局和核函数，推理速度提升高达 2 倍，VRAM 占用减少 65%，并支持训练 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/releases/tag/v0.3.0">Release v0.3.0 · pytorch/ao</a>: v0.3.0 亮点 我们很高兴地宣布 torchao 0.3 版本发布！此版本增加了对新 quantize API、MX 格式、FP6 dtype 和 bitpacking、2:4 稀疏加速训练的支持...</li><li><a href="https://github.com/pytorch/pytorch/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen">Issues · pytorch/pytorch</a>: Python 中具有强 GPU 加速的 Tensor 和动态神经网络 - Issues · pytorch/pytorch
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1255624486976225290)** (131 条消息🔥🔥): 

- **CUDA 中的创新指针处理**：在关于优化 CUDA 编程的讨论中，有人给出了深刻的解释：“将……视为仅对该‘Parameter Pack’的每个元素应用操作”（指使用 Parameter Pack 进行指针初始化）。它强调了减少指定类型时的冗余，并探讨了诸如 `dtype_of` 之类实用程序的替代设计。

- **16 GPU 训练的胜利**：一位成员分享了在 **Lambda 上使用 16 个 GPU** 进行训练的进展，称其过程非常“壮观”，并描述了显著的加速效果，几乎实现了 1.97 倍的提升。尽管设置过程充满挑战，涉及 MPI 错误和 SSH 密钥问题，但最终取得了成功。

- **关于 CUDA 分配与性能的辩论**：成员们讨论了各种内存分配方法的权衡，特别是关于 `cudaMallocHost()`，并分享了其对性能影响的经验。一个建议是“`尽量接近仅分配一次的状态，这样可以保证在第一步之后不会出现 OOM`”。

- **异步 Checkpointing PR 审查**：一个关于“异步状态和模型 Checkpointing”的 PR 受到严格审查，人们担心增加的复杂性和对内存分配的影响。一位成员认为，“*我不确定现在这是否值得*”，暗示倾向于将此类更新推迟到 1.0 版本之后。

- **Gemma 2 AI 模型发布备受关注**：Google 发布 **Gemma 2** 引发了热议，因其击败了 Llama3 70B 和 Qwen 72B 等更大规模的模型而受到称赞。亮点包括其在更少 Token 下的高效性能、对局部和全局 Attention 层的使用，以及诸如“Soft attention capping”和“WARP model merging”等创新训练技术。提供的链接：[Reach_vb Gemma](https://x.com/reach_vb/status/1806343018640781675), [Danielhanchen Gemma](https://x.com/danielhanchen/status/1806372357684220308)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1806372357684220308">来自 Daniel Han (@danielhanchen) 的推文</a>：刚刚分析了 Google 新发布的 Gemma 2！9B 和 27B 的 Base 和 Instruct 版本已经发布！1. Pre &amp; Post Layernorms = 像 Grok 一样有 2 倍的 LNs。2. 使用了 Grok 的 softcapping！Attn logits 被截断到 (-30, 3...</li><li><a href="https://x.com/reach_vb/status/1806343018640781675">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：冲啊！Google 刚刚发布了 Gemma 2 27B &amp; 9B 🔥 &gt; 在 LMSYS Chat arena 中击败了 Llama3 70B/ Qwen 72B/ Command R+，且 9B 是目前最好的 15B 以下模型。&gt; 比 Llama 小 2.5 倍...</li><li><a href="https://github.com/karpathy/llm.c/pull/652">由 ademeure 提交的 Pull Request #652 · karpathy/llm.c：为 Flash Attention backward 使 cuDNN 确定性化</a>：6 月 13 日发布的 cuDNN Frontend 1.5 添加了一个新设置，使其 backward 算法具有确定性，该设置默认禁用：NVIDIA/cudnn-frontend@47d800c https://github.com/NVIDIA/cu...</li><li><a href="https://github.com/karpathy/llm.c/pull/644">由 ngc92 提交的 Pull Request #644 · karpathy/llm.c：混合 dtypes</a>：这目前基于 #635，因为在那里我们最终需要在 bf16 中减少 loss。所以这个 PR 首先进行了一些 malloc-and-point 泛化以允许多种 dtypes，然后更改...</li><li><a href="https://github.com/NVIDIA/cudnn-frontend/commit/47d800ccd9449e1bbc255d64d794ae88d99b043d">cudnn-frontend 1.5.0 发布说明：(#81) · NVIDIA/cudnn-frontend@47d800c</a>：[新功能] 使用 cudnn backend 9.2.0 及以上版本，`Graph::check_support` 可以在不调用 nvrtc 编译器的情况下确定运行时引擎的支持检查。这允许用户检查支持...</li><li><a href="https://github.com/karpathy/llm.c/pull/653">由 ademeure 提交的 Pull Request #653 · karpathy/llm.c：仅使用 cuBLASLt + GELU Fusion 的 Matmul 重构</a>：为了准备 FP8，此 PR 将所有 cuBLAS 调用替换为 cuBLASLt，现在由单个 matmul_cublaslt() 函数包装。它还增加了对 GELU fusion 的支持，可以在 c...</li><li><a href="https://github.com/karpathy/llm.c/actions/runs/9699238303/job/26767716987?pr=653">仅使用 cuBLASLt + GELU Fusion 的 Matmul 重构 · karpathy/llm.c@7082ab6</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/microsoft/mutransformers/tree/ed0e4af9700247e2067a131c2757a85133ab7d09">GitHub - microsoft/mutransformers at ed0e4af9700247e2067a131c2757a85133ab7d09</a>：一些使用最大更新参数化 (µP) 的常用 Huggingface transformers - GitHub - microsoft/mutransformers at ed0e4af9700247e2067a131c2757a85133ab7d09</li><li><a href="https://github.com/karpathy/llm.c/pull/651">由 chinthysl 提交的 Pull Request #651 · karpathy/llm.c：异步优化器状态和模型 Checkpointing</a>：使用非阻塞后台线程 Checkpoint 优化器状态和模型参数的附加功能。一次性将 device buffers 内存拷贝到 pinned host buffer，并让后台线程执行...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[sparsity](https://discord.com/channels/1189498204333543425/1247663759434977453/1255923599504707594)** (1 条消息): 

- **2:4 Sparsity 加速神经网络训练**：最近在 [xFormers](https://pytorch.org/blog/accelerating-neural-network-training/) 中使用 2:4 sparsity 的工作显示，[Segment Anything](https://github.com/pytorch/ao/tree/main/torchao/sparsity#segment-anything) 项目的推理速度提升了 10%。通过扩展这种方法，他们在 NVIDIA A100 上使用 `SemiSparseLinear` 层实现了 1.3 倍的模型训练加速，使 [DINOv2 ViT-L](https://github.com/facebookresearch/dinov2) 训练的实际耗时缩短了 6%。

**提到的链接**：<a href="https://pytorch.org/blog/accelerating-neural-network-training/">使用半结构化 (2:4) Sparsity 加速神经网络训练</a>：在过去的一年里，我们为 PyTorch 增加了对半结构化 (2:4) sparsity 的支持。只需几行代码，我们就能够在 segment-anything 上展示 10% 的端到端推理加速...

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1255599091279466687)** (27 messages🔥): 

- **LLM OOD 检测机制**：成员们讨论了 **infinigram ensemble** 技术是否能提升 LLM 性能。一位用户分享了 [Extrapolates Predictably](https://arxiv.org/abs/2402.04362) 论文，解释说神经语言模型（neural LMs）在训练早期学习低阶矩（low-order moments），但在后期会失去这种能力。
- **N-Gram 与神经语言模型（neural LMs）**：关于 **n-gram / bag of words** 特征在改进神经语言模型训练中的有效性存在争论。一位用户澄清说 infinigram 通常不用于特征生成，但对支持这一点的文献感兴趣。
- **分享新研究论文**：一位用户分享了关于 [分布简单性偏置（distributional simplicity bias）](https://arxiv.org/abs/2402.04362) 和 [Efron-Stein decomposition](https://arxiv.org/abs/2111.09375) 的论文链接，引发了对其影响的进一步讨论。
- **Mozilla 针对本地 AI 项目的资助征集**：一位成员向频道通报了 Mozilla 新的资助征集，重点关注 **local AI**，包括微调技术和新的 UI 范式。另一位用户注意到 PDF 中的 Discord 邀请链接已失效，建议直接给联系人发邮件。
- **寻找 Mozilla AI Discord 邀请链接**：在发现 PDF 中的链接过期后，一位成员正在寻找有效的 Mozilla AI Discord 邀请链接。另一位用户通过快速在线搜索找到了相关链接，解决了问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.04362">Neural Networks Learn Statistics of Increasing Complexity</a>：分布简单性偏置（DSB）假设神经网络首先学习数据分布的低阶矩，然后再转向高阶相关性。在这项工作中，我们展示了 com...</li><li><a href="https://arxiv.org/abs/2111.09375">Hypercontractivity on High Dimensional Expanders: Approximate Efron-Stein Decompositions for $\varepsilon$-Product Spaces</a>：我们证明了高维扩展图（high dimensional expanders）上的超收缩不等式。正如在 p-biased 超立方体、对称群和 Grassmann 方案的设置中一样，我们的不等式对于 gl...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1255611941532209282)** (161 messages🔥🔥): 

- **理论中的神经元与权重排列**：讨论围绕神经元激活形成可预测模式的观点展开，提出了直接在排列分布（permutation distributions）上进行训练的可能性，这可能会利用 Zipf 启发式和 Monte Carlo 搜索进行优化。该假设建议通过重新构建神经元权重的顺序来获得潜在的效率提升。

- **SPARSEK Attention 降低复杂度**：一篇新论文介绍了 **SPARSEK Attention**，旨在通过使用稀疏选择机制克服自回归 Transformer 中 self-attention 的计算和内存限制。该方法承诺具有线性时间复杂度，并能显著提升速度（[arxiv 链接](https://arxiv.org/abs/2406.16747)）。

- **流形假设（Manifold Hypothesis）测试咨询**：一位成员询问是否有可用于在数据集上测试流形假设的代码，寻求最佳方法的建议。目前未提供具体的链接或资源。

- **优化器的进展与比较**：介绍了 **Adam-mini**，这是一种声称性能与 AdamW 相当但内存占用显著降低的优化器。与 NovoGrad 的详细对比讨论突出了架构选择和实际考量（[arxiv 链接](https://arxiv.org/abs/2406.16793)）。

- **关于 Transformer 中最佳层顺序的辩论**：关于在混合模型（hybrid models）中应优先选择线性注意力（linear attention）还是滑动窗口注意力（sliding window attention）的激烈辩论。对话引用了几篇最近的论文和演示文稿，这些文献对大规模模型中的最佳层排序策略持有不同观点（[额外的 arxiv 链接](https://arxiv.org/abs/2405.05254v2)）。
<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: 我们提出了 Adam-mini，这是一种优化器，其性能与 AdamW 相当或更好，但内存占用减少了 45% 到 50%。Adam-mini 通过削减学习率资源来减少内存...</li><li><a href="https://arxiv.org/abs/2406.17711">Data curation via joint example selection further accelerates multimodal learning</a>: 数据策展（Data curation）是大规模预训练的重要组成部分。在这项工作中，我们证明了联合选择数据批次比独立选择样本对学习更有效...</li><li><a href="https://arxiv.org/abs/1911.03864">Improving Transformer Models by Reordering their Sublayers</a>: 多层 Transformer 网络由交错的 self-attention 和 feedforward 子层组成。以不同的模式排列子层是否能带来更好的性能？我们生成了随机或...</li><li><a href="https://arxiv.org/abs/2406.13155">Convolutional Kolmogorov-Arnold Networks</a>: 在本文中，我们介绍了卷积 Kolmogorov-Arnold 网络 (Convolutional KANs)，这是对标准卷积神经网络 (CNNs) 的一种创新替代方案，后者曾彻底改变了...</li><li><a href="https://arxiv.org/abs/2406.18532">Symbolic Learning Enables Self-Evolving Agents</a>: AI 社区一直在通过开发 "language agents" 来探索通往通用人工智能 (AGI) 的路径，这些 Agent 是复杂的 LLM 流水线，涉及...</li><li><a href="https://x.com/ericzhang0410/status/1805814432595165567">Tweet from YushunZhang (@ericzhang0410)</a>: 非常感谢提到 NovoGrad！我们更新了论文 https://arxiv.org/pdf/2406.16793 并讨论了它们与 Adam-mini 的区别。简而言之，至少存在两个方面的重大差异...</li><li><a href="https://arxiv.org/abs/2406.17245">Unlocking Continual Learning Abilities in Language Models</a>: 语言模型 (LMs) 表现出令人印象深刻的性能和泛化能力。然而，LMs 面临着灾难性遗忘的持久挑战，这损害了它们的长期...</li><li><a href="https://arxiv.org/abs/2406.07887">An Empirical Study of Mamba-based Language Models</a>: 像 Mamba 这样的选择性状态空间模型 (SSMs) 克服了 Transformers 的一些缺点，例如随序列长度增加的二次计算复杂度和推理时巨大的内存需求...</li><li><a href="https://arxiv.org/abs/2406.14596">ICAL: Continual Learning of Multimodal Agents by Transforming Trajectories into Actionable Insights</a>: 大规模生成式语言和视觉语言模型 (LLMs 和 VLMs) 在决策和指令遵循的 few-shot in-context learning 方面表现出色。然而，它们需要高质量的样本...</li><li><a href="https://arxiv.org/abs/2406.16747">Sparser is Faster and Less is More: Efficient Sparse Attention for Long-Range Transformers</a>: 在自回归 Transformer 中高效处理长序列，特别是在扩展的上下文窗口内，由于二次计算复杂度和...</li><li><a href="https://x.com/BlancheMinerva/status/1741855005601141091">Tweet from Stella Biderman (@BlancheMinerva)</a>: 许多人似乎认为在大型实验室之外无法进行有趣的 LLM 研究，或者被迫进入拥挤的主题。实际上，有很多完全开放的高价值问题。为了证明...</li><li><a href="https://arxiv.org/abs/2405.05254v2">You Only Cache Once: Decoder-Decoder Architectures for Language Models</a>: 我们为大型语言模型引入了一种解码器-解码器架构 YOCO，它仅缓存一次键值对 (key-value pairs)。它由两个组件组成，即堆叠在 self-decoder 之上的 cross-decoder。...</li><li><a href="https://github.com/google/gemma_pytorch">GitHub - google/gemma_pytorch: The official PyTorch implementation of Google's Gemma models</a>: Google Gemma 模型的官方 PyTorch 实现 - google/gemma_pytorch</li><li><a href="https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_novograd.cu#L108>).">apex/csrc/multi_tensor_novograd.cu at master · NVIDIA/apex</a>: 一个 PyTorch 扩展：在 PyTorch 中实现轻松混合精度和分布式训练的工具 - NVIDIA/apex
</li>
</ul>

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1255602423666446419)** (7 messages): 

- **实践中的 Hopfield Networks 理解**：一位成员对 **Hopfield layers** 如何在神经网络中作为 attention 机制使用表示困惑。另一位成员澄清说，记忆发生在 pre-training 阶段，而检索发生在 forward pass 过程中，类似于 Transformer 中的 self-attention。
- **Hopfield 与 Self-Attention 的计算对比**：有成员对 **Hopfield networks** 和 **self-attention** 机制之间的计算差异提出了疑问。解释称，当 Hopfield layer 用作 self-attention 时，它仅向训练好的模式更新一步，其行为类似于 Hopfield network 的单步操作。
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1255599616657723513)** (193 messages🔥🔥): 

- **"A6000 vs. 3090s GPU 对决"**：成员们讨论了拥有 **48GB VRAM** 的 **NVIDIA A6000** GPU 以及通过 NVLink 连接实现 **96GB VRAM** 的优缺点。相比之下，考虑到价格和 **multi-GPU setups** 的计算能力，一些人更倾向于 **quad 3090s**。
  
- **"探索预算友好的 GPU 选项"**：讨论包括推荐更便宜的 GPU，如 **certified refurbished P40s** 或 **K80s**，认为它们在预算大幅降低的情况下仍能处理大型模型。一位成员指出：*"你可以在两块 P40/P100 上运行合理 quant 的 L3 70B，其成本仅为单块 3090 的 1/4 到 1/2"*。
  
- **"专用 AI 芯片的灵活性不足"**：针对 Etched 公司推出的 **Sohu chip** 存在质疑，称其仅专注于 Transformer，预测会有局限性。对此有人反驳道：*"噢天哪，我没想到它会那么不灵活"*，并进一步讨论了 Nvidia 预期推出的 **transformer cores** 将如何与此类芯片竞争。
  
- **"AI 模型训练中的伦理考量"**：成员们辩论了是否应排除与儿童相关的数据以防止在 NSFW 场景中被滥用，同时也提出了维持模型多样性的论点。有人对模型在普通任务中的可用性表示担忧：*"如果模型无法生成普通的家庭场景，那它就是无用的"*。

- **"NSFW 在 AI 模型中的角色"**：关于 foundational AI models 是否需要 NSFW 数据进行了讨论，结论是这并非必不可少，因为 *"在 pre-training 之后，模型可以被训练成几乎任何你想要的样貌"*。然而，在平衡伦理考量与实际 AI 应用的最佳实践上，意见分歧显著。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fastvoiceagent.cerebrium.ai">全球最快的语音机器人演示</a>：一个展示了在优化并部署以最小化网络和模型延迟后，语音驱动的 AI 聊天机器人潜在能力的演示。</li><li><a href="https://www.etched.com/announcing-etched">Etched 正在进行 AI 领域最大的豪赌</a>：未找到描述</li><li><a href="https://tenor.com/view/jim-halpert-the-office-confused-gif-25227530">Jim Halpert GIF - Jim Halpert The - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/james-crashes-into-tar-hit-cargo-thomas-the-train-gif-17279386">James Crashes Into Tar Hit GIF - James Crashes Into Tar Hit Cargo - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://arxiv.org/abs/2405.07992">MambaOut: 视觉领域真的需要 Mamba 吗？</a>：Mamba 是一种具有类似 RNN 的状态空间模型 (SSM) token mixer 的架构，最近被提出以解决 attention 机制的二次复杂度问题，并随后应用于视觉任务...</li><li><a href="https://x.com/bryan_johnson/status/1805629207374086490">Bryan Johnson /dd (@bryan_johnson) 的推文</a>：很高兴投资 @Etched 的 1.2 亿美元 A 轮融资。便宜 10 倍的 AI 模型将让我们解决衰老问题的速度提高 100 倍。引用 Etched (@Etched)：介绍 Sohu，史上最快的 AI 芯片...</li><li><a href="https://www.reddit.com/r/singularity/comments/1dpxocg/gege_ai_a_new_music_maker_ai_that_allows_you_to/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/tenstorrent/tt-firmware">GitHub - tenstorrent/tt-firmware: Tenstorrent 固件仓库</a>：Tenstorrent 固件仓库。通过在 GitHub 上创建账号为 tenstorrent/tt-firmware 开发做出贡献。
</li>
</ul>

</div>

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1255607529866526827)** (2 messages): 

- **AIW+ 问题仍然模糊**：一名成员讨论了与常识性 AIW 相比，解决 AIW+ 问题的复杂性。他们强调了计算表亲的问题以及潜在的歧义，例如减去 Alice 及其姐妹以及 Alice 父亲兄弟姐妹的影响，并总结道 *“这并不能证明不存在歧义。”*
  

---



### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1255710442966225018)** (12 messages🔥): 

- **寻求预测模型内存需求的公式**：一位用户询问了一个公式，用于预测模型在 X 大小的上下文窗口下所需的内存，旨在根据 gguf 元数据估算上下文长度设置。
- **内存使用的模型特定差异**：另一位用户指出，由于 attention heads 的差异，内存使用量可能因模型而异，并强调需要一个特定的公式。
- **用于上下文设置的元数据见解**：一位用户分享了来自 llama 模型的元数据细节，这有助于估算内存需求，并指出超过某些上下文数值（例如 8192）可能会导致性能下降。
- **建议采用“笨拙的”经验方法**：一位用户建议采用经验方法，提倡编写一个脚本来加载并测量不同上下文长度下的 RAM 使用情况，并收集数据绘图以找出变化率。
- **Claude 令人信服但尚不确定的回答**：一位用户提到收到了来自 Claude 令人信服的回复，但仍持怀疑态度，由于涉及的复杂性和众多变量，更倾向于采用经验方法。
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1255619972663742566)** (16 messages🔥): 

- **给你的狗做一个星座 App**：成员们讨论了为狗创建一个星座 App，重点是让它变得“有点搞怪”。一位成员特别感兴趣这个 App 是否已经被开发出来。
- **为超级富豪提供的“换血少年”版 Tinder**：一位成员分享了一个[链接](https://x.com/yaya_labs_/status/1806252865628860494?t=YEplRGt5YkcKxbA1bQMCPw&s=19)，关于一个幽默的提议，即开发一个类似 Tinder 的 App，将超级富豪与提供输血的“blood boys”进行匹配。讨论涉及了这一想法的实用性和伦理问题。
- **血液即服务 (BAAS)**：成员们讨论了“血液即服务 (BAAS)”的概念，思考了直接输血以吸收活力的物流和潜在益处。他们注意到，虽然 Bryan 从他儿子的血液中获益不多，但他的父亲（Bryan 儿子的祖父）看到了更显著的益处。

**提到的链接**：<a href="https://x.com/yaya_labs_/status/1806252865628860494?t=YEplRGt5YkcKxbA1bQMCPw&s=19">来自 Yaya Labs (@yaya_labs_) 的推文</a>：开发一个类似 Tinder 的 App 来匹配超级富豪和他们的 blood boys 怎么样。你会安装吗？

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1255902714064273500)** (5 条消息): 

- **Rakis 系统彻底改变了 LLM 推理**：[Rakis GitHub 仓库](https://github.com/hrishioa/rakis)提供了一个 **100% 基于浏览器的去中心化 LLM 推理**解决方案。一位成员认为这种方法“非常酷”，并可能为去中心化 AI 应用带来变革。

- **Meta 发布 LLM Compiler**：[Meta 的 LLM Compiler](https://x.com/AIatMeta/status/1806361623831171318) 集成了先进的代码优化和编译器功能，提升了代码大小优化和反汇编能力。他们发布了 7B 和 13B 模型，采用宽松的许可证以支持研究和商业用途，展示了 AI 在优化代码方面的潜力。

- **JinaAI 推出 PE-Rank 以实现高效重排序**：JinaAI 的 [PE-Rank](https://x.com/JinaAI_/status/1806331488247402524) 利用段落嵌入 (passage embeddings) 通过 LLM 实现高效的 listwise 重排序。通过将段落编码为特殊 token 并限制输出空间，该方法将 100 个文档的重排序延迟从 21 秒降低到 3 秒。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/AIatMeta/status/1806361623831171318">来自 AI at Meta (@AIatMeta) 的推文</a>：今天我们宣布推出 Meta LLM Compiler，这是一个基于 Meta Code Llama 构建的模型家族，具有额外的代码优化和编译器功能。这些模型可以模拟编译器，预测最佳方案...</li><li><a href="https://x.com/JinaAI_/status/1806331488247402524">来自 Jina AI (@JinaAI_) 的推文</a>：我们不能直接用 LLM 进行重排序吗？🤔只需将查询、doc1、doc2、...docN 放入上下文窗口，让 LLM 找出 top-K？事实证明我们可以，而且几乎就是你所想的那样...</li><li><a href="https://github.com/hrishioa/rakis">GitHub - hrishioa/rakis</a>：通过在 GitHub 上创建账号来为 hrishioa/rakis 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1255974809028792350)** (1 条消息): 

- **Hermes 2 Pro 70B 发布**：“我们刚刚发布了 Hermes 2 Pro 70B！这是一个纯粹的 Hermes 版本，没有与 Llama-3 Instruct 进行合并。” 此次发布承诺解决函数调用 (function call) 问题或拒绝回答的情况，尽管会有轻微的性能代价。请在 [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B) 上查看。

- **模型描述亮点**：Hermes 2 Pro 是 Nous Hermes 2 的升级版，采用了“更新且清洗过的 OpenHermes 2.5 数据集”以及全新的 **Function Calling 和 JSON Mode 数据集**。在与 Fireworks.AI 合作进行的评估中，它在函数调用评估中得分 **90%**，在结构化 JSON 输出评估中得分 **84%**。

**提到的链接**：<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B">NousResearch/Hermes-2-Pro-Llama-3-70B · Hugging Face</a>：未找到描述

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1255624397843337340)** (96 条消息🔥🔥): 

- **LLM 重复问题与采样设置相关**：成员们讨论了 **instruction-tuned LLMs（指令微调 LLM）重复文本** 的原因。有人建议，“通常是由于缺乏重复惩罚（repetition penalty）或糟糕的采样设置”，而另一位成员表示赞同，称其为“重复惩罚问题或采样问题”。

- **Big-AGI 前端与安全顾虑**：一位成员质疑使用 **Big-AGI** 托管其 ChatGPT 密钥的安全性。其他人推荐了替代方案，如 [librechat](https://librechat.github.io/) 和 [huggingchat](https://huggingface.co/huggingchat)，并指出大多数选项需要自行托管（self-hosting）。

- **GPT API 使用的可用前端**：Teknium 分享了一个开源 Web 应用 [Prompt-Engineering-Toolkit](https://github.com/teknium1/Prompt-Engineering-Toolkit)，作为 GPT APIs 的前端。另一位成员介绍了他们自己的高性价比平台 [FreeAIChatbot.org](https://freeaichatbot.org)，该平台支持多种功能并本地存储数据。

- **Meta 发布先进的 LLM Compiler 模型**：Meta 宣布发布针对代码大小和反汇编任务优化的 **LLM Compiler 模型**。公告中包含了 [Hugging Face 仓库](https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3) 和 [研究论文](https://arxiv.org/abs/2405.00675) 的链接。

- **野外出现的新型先进模型**：用户讨论了新发布的 **Llama-3-Instruct-8B-SPPO**，认为作为一个 8B 模型，它的智能程度和上下文感知能力令人印象深刻。分享的链接：[Meta Llama 3-8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://get.big-agi.com/">big-AGI</a>: 启动 big-AGI 以释放 AI 的全部潜力，精确控制您的数据和模型。具备语音界面、AI 角色、高级功能和有趣的 UX。</li><li><a href="https://freeaichatbot.org">FreeAIChatbot.org</a>: 一个用于编程、学习等的免费 AI 聊天机器人。使用 Claude、GPT-4 等生成图像、处理 CSV、与图像聊天等。</li><li><a href="https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3">UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/dancing-duck-dance-duck-duck-ooontz-dance-gif-10943740227711557279">Dancing Duck Dance Duck GIF - Dancing duck Dance duck Duck - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/peakcooper/status/1804867319350394912>">来自 Cooper (@peakcooper) 的推文</a>: 是的，Claude 3.5 Sonnet 强得离谱</li><li><a href="https://x.com/AIatMeta/status/1806361623831171318">来自 AI at Meta (@AIatMeta) 的推文</a>: 今天我们宣布推出 Meta LLM Compiler，这是一个基于 Meta Code Llama 构建的模型家族，具有额外的代码优化和编译器能力。这些模型可以模拟编译器，预测最优...</li><li><a href="https://lluminous.chat">lluminous</a>: 未找到描述</li><li><a href="https://github.com/teknium1/Prompt-Engineering-Toolkit">GitHub - teknium1/Prompt-Engineering-Toolkit</a>: 通过在 GitHub 上创建账户来为 Prompt-Engineering-Toolkit 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1255618178470383800)** (25 条消息🔥): 

- **Hermes Pro 中的布尔值混淆**：成员们注意到 **Hermes Pro** 中的函数调用（function calls）存在一个问题，即布尔值在 JSON 中返回的是 `True` 而不是 `true`。**Teknium** 认为这可能是由于 **Python** 格式渗入到了 Schema 构建中，而 **.interstellarninja** 确认这两种格式都经过了训练并被接受。
- **讨论数据集完整性**：**Teknium** 提出了一个问题，即是否应该修复数据集以仅输出有效的 JSON。**.interstellarninja** 提到函数调用是由 OAI API 生成的，引发了关于数据中是否存在无效布尔标签的进一步讨论。
- **特定模型的问题？**：**Teknium** 推测合并问题可能影响了 **Theta** 模型，暗示这可能会影响布尔值的有效性。然而，**craigsdenniscf** 澄清该问题是在 **Hermes Pro** 中观察到的，而不是 **Theta**。
  

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1255825144639918141)** (8 条消息🔥): 

- **Glaive-RAG-v1 数据集发布**：一位成员表示：“我们本周发布了一个 RAG 数据集和模型”，并指向了 [Glaive-RAG-v1](https://huggingface.co/datasets/glaiveai/RAG-v1) 数据集。该数据集包含约 5 万个样本，是使用 Glaive 平台构建的，旨在为 RAG 使用场景微调模型。数据包括上下文文档、问题、回答模式以及带有引用的回答。

- **澄清引用标签的来源**：一位用户询问了数据集系统提示词中 `<co:` 标签的来源，另一位成员澄清说，虽然他们使用了与 Cohere 相同的引用方法，但数据并非使用 Cohere 模型生成的。

- **将 Glaive-RAG-v1 格式适配到 Hermes RAG**：用户 interstellarninja 指出，Glaive-RAG-v1 中的“输入-上下文-输出-引用”格式可以适配到 Hermes RAG。他们还表示愿意在生成特定领域数据集方面进行协作。

- **为新数据集生成选择领域**：sahilch 确定了网页搜索和 Wikipedia 是生成新数据的潜在领域，并思考了成员间讨论的 Google 表格中列出的领域。

- **理想数据集大小的讨论**：在关于数据集理想大小的讨论中，有人提到对于完整的 RAG 样本库，目标大小约为 5k-20k。目前 Glaive-RAG-v1 数据集已经涵盖了这个范围，但可以进一步扩展以增加多样性或填补缺失领域。

**提到的链接**：<a href="https://huggingface.co/datasets/glaiveai/RAG-v1">glaiveai/RAG-v1 · Datasets at Hugging Face</a>：未找到描述

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1255885748243140659)** (11 条消息🔥): 

- **恐怖 Gif 大放送**：一位用户分享了来自 Tenor 的一系列 GIF 链接，包括与 *《惊声尖笑》矩阵椅子*、*万圣节幽灵*、*奇异博士* 和 *《黑洞》* 相关的片段。这些 GIF 涵盖了电影、动漫和太空等各种主题。

- **问候交流**：进行了一段简短的对话，用户 @teknium 带着自定义表情向小组问候 *"Hows it going"*，随后 @rezonaut 回复了 *"hello"*。互动最后以 @teknium 回复一个挥手表情结束。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/halloween-ghost-ghosts-spooky-gif-15339491">Halloween Ghost GIF - 万圣节幽灵 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/the-black-hole-space-galaxy-gif-14315492">The Black Hole Space GIF - 黑洞太空星系 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/plato-cave-rave-platonic-solids-gif-13990069">Plato Cave GIF - 柏拉图洞穴狂欢 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/lain-iwakura-lain-serial-experiments-lain-the-wired-computer-gif-22576121">Lain Iwakura Lain GIF - 玲音 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/scary-movie-matrix-chair-gif-12964507">Scary Movie Matrix GIF - 惊声尖笑矩阵椅子 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/doctor-strange-ancient-one-marvel-gif-14594882">Doctor Strange Ancient One GIF - 奇异博士古一法师 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/lain-lain-iwakura-serial-experiments-lain-wires-wired-gif-1481475804337586659">Lain Lain Iwakura GIF - 玲音连线 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/fmab-edward-elric-fullmetal-alchemist-fullmetal-alchemist-brotherhood-fma-gif-19554771">Fmab Edward Elric GIF - 钢之炼金术师爱德华 - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1255599689118777468)** (163 messages🔥🔥): 

- **关于 Apple 硬件与 AI 任务的辩论**：一名成员考虑使用 6GB 或 8GB 的 MacBook Air 处理 AI 任务，但缺乏足够的他人反馈。另一名成员承认不清楚 Apple 是否适合 AI 工作：*“不确定这里有多少人了解 Apple 电脑... 找不到相关信息。”*
- **LoRA 训练中的质量与参数**：一名成员解释了在 LoRA 训练中使用不同 batch sizes 和 epochs 产生的结果。他们寻求关于在训练模型时平衡细节与质量的建议，并表示：*"16 batches 和 200 epochs... 图像细节较少，但形状良好。"*
- **SD3 与 civitai 许可协议的冲突**：一名成员询问了关于 SD3 和 civitai 的更新情况，得知由于许可协议问题，相关模型仍被禁止。另一人对许可限制评论道：*“在当前的 SD3 许可协议下”... 由于 civit 属于商业运营，发布模型可能不被允许*。
- **Kaggle 提供免费 GPU 资源**：一位用户强调 Kaggle 免费提供两个 T4 GPU（32GB VRAM），这对训练模型的人很有帮助。他们分享了 Kaggle 的链接以及 GitHub 上的一个 Stable Diffusion notebook 以帮助他人：[Kaggle](https://www.kaggle.com) 和 [SD WebUI Notebook](https://github.com/DEX-1101/sd-webui-notebook)。
- **对恢复旧频道的兴趣**：几位成员讨论了对包含生成式 AI 内容的旧频道进行存档的问题，并有兴趣可能恢复它们。一位成员回忆了特定主题讨论的价值：*“遗憾，能有特定主题的讨论频道并建立联系是很棒的。”*
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/etched/status/1805625693113663834?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">来自 Etched (@Etched) 的推文</a>：认识 Sohu，史上最快的 AI 芯片。在运行 Llama 70B 时每秒超过 500,000 tokens，Sohu 让你能够构建在 GPU 上无法实现的产品。一台 8xSohu 服务器可替代 160 块 H100。Soh...</li><li><a href="https://www.kaggle.com/">Kaggle：您的机器学习与数据科学社区</a>：Kaggle 是全球最大的数据科学社区，拥有强大的工具和资源，帮助您实现数据科学目标。</li><li><a href="https://github.com/DEX-1101/sd-webui-notebook">GitHub - DEX-1101/sd-webui-notebook: Stable Diffusion Web UI Notebook</a>：Stable Diffusion Web UI Notebook。通过在 GitHub 上创建账户为 DEX-1101/sd-webui-notebook 的开发做出贡献。</li><li><a href="https://opendata.blender.org/">Blender - 开放数据</a>：Blender Open Data 是一个收集、展示和查询硬件及软件性能测试结果的平台，由公众提供。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1255612709626708078)** (102 条消息🔥🔥): 

- **图表库讨论凸显权衡**：关于图表库的讨论强调了诸如静态 vs 交互式图表、原生 vs 浏览器渲染以及数据输入格式等决策。例如，*"社区需要决定他们想要从图表库中得到什么。"*

- **渲染大型数据集的访问与性能**：关于浏览器 vs 原生解决方案等渲染选项的辩论集中在可访问性和效率上。*"有时你只需要绘制 2 亿个数据点，而 Chrome 会因此卡死崩溃。"*

- **用于 Mojo Nightly 构建的 Docker 容器**：存在关于为不同 Mojo 构建设置 Docker 容器的问题，并分享了具体的建议和示例。*"不过你的安装行写错了……你可以通过使用 `modular install nightly/mojo` 来演示基础操作。"*

- **Mojo K-Means 示例问题**：一名成员在运行 Mojo 的 K-Means 示例时遇到错误，讨论围绕过时的代码和可能的修复方案展开。*"这是其中一条错误信息：error: use of unknown declaration 'mod'。"*

- **Mojo 社区会议公告**：宣布了一场 Mojo 社区会议，并分享了日期、时间和访问链接的详细信息。*"我们将举行下一次 Mojo 社区会议…… Zoom: [链接]。"*

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://microsoft.github.io/SandDance/">SandDance 主页</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/examples/docker/Dockerfile.mojosdk#L54">modularml/mojo 中的 Dockerfile.mojosdk</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/1887#issuecomment-1998929184).">[BUG]: modular install --install-version 的回归问题 · Issue #1887 · modularml/mojo</a>：Bug 描述 --install-version 被添加用于支持 Mojo 的版本固定，例如：在 CI/CD 环境中。我一直在 mojo-pytest 项目中成功使用它。在 24.1 版本中，这个版本固定...</li><li><a href="https://github.com/modularml/devrel-extras/tree/main/blogs/mojo-kmeans-from-python">modularml/devrel-extras 中的 mojo-kmeans-from-python</a>：包含开发者关系博客文章、视频和研讨会的支持材料 - modularml/devrel-extras</li><li><a href="https://github.com/modularml/mojo/branches">Branches · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://lobste.rs/s/wnff6n/python_still_surprises">Python 依然令人惊讶 | Lobsters</a>：未找到描述</li><li><a href="https://idl.uw.edu/mosaic/examples/linear-regression-10m.html">线性回归 10M | Mosaic</a>：未找到描述</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder/encodeInto">TextEncoder: encodeInto() 方法 - Web APIs | MDN</a>：TextEncoder.encodeInto() 方法接受一个要编码的字符串和一个用于存放生成的 UTF-8 编码文本的目标 Uint8Array，并返回一个表示进度的字典对象...</li><li><a href="https://modul.ar/community-meeting-zoom.">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。</li><li><a href="https://modul.ar/community-meeting-doc">[公开] Mojo 社区会议</a>：Mojo 社区会议 文档链接：https://modul.ar/community-meeting-doc 这是一个公开文档；欢迎所有人查看并评论/建议。所有会议参与者必须遵守...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1255631651497709568)** (2 条消息): 

- **Modular 分享 Twitter 更新**：ModularBot 发布了来自 [Modular Twitter 账号](https://twitter.com/Modular/status/1806070670293692594)的推文。更新内容包括[此处](https://twitter.com/Modular/status/1806356878282371398)链接的另一条推文。
  

---

### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1255770692662726767)** (1 条消息): 

- **人类思维：魔法与神经网络**：一位成员提出，人类思维由一个“魔法事物”组成，它负责创建、使用和编辑神经网络。这种二分法被称为“智能”（即魔法）和“认知”（即神经网络）。
- **上下文迭代与熵**：该假设指出，特定的人类神经网络是“魔法事物”的上下文迭代，在近距离范围内有效，但超出该范围后会受到“不可名状的熵”的影响。
- **智能向认知的转录**：大多数智能行为被描述为认知过程的结果。这种转录通常在现实世界显现之前先转化为认知层，即使在采取捷径时，通常也能找到认知等效物。
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1255631646187585576)** (23 条消息🔥): 

- **鼓励在 GitHub 上报告问题**：成员们讨论了将 Mojo 中的任何问题报告给 GitHub 的重要性，一位成员表示：*“他们一直鼓励在 GitHub 上提交问题。”* 用户被告知即使使用 nightly 构建版本，也不要犹豫提出疑虑。

- **寻求 Mojo 版本兼容性的自动化方案**：一位成员询问如何自动识别 Mojo 代码版本，建议采用从低版本编译器开始的试错法。另一位成员提出，如果需要，可以帮助将代码适配到 main 分支。

- **对 Mojo 对象标识（Object Identity）的好奇**：一位用户询问了关于 [Mojo vs Rust 博客文章](https://www.modular.com/blog/mojo-vs-rust-is-mojo-faster-than-rust) 中“无需 Pin 要求”部分的更多信息，特别是关于对象标识和自引用类型（self-referential types）的内容。

- **对运行 Mojo 时网络活动的担忧**：多位用户观察到在终端运行 Mojo 时会尝试连接互联网。一位成员提到这可能与账户身份验证有关，并建议在 GitHub 上提交 issue 以进一步调查。

**提到的链接**：<a href="https://youtu.be/UOTAzCYQjHs">Mojo - First Impression [Programming Languages Episode 29]</a>：►Full First Look Series Playlist: https://www.youtube.com/playlist?list=PLvv0ScY6vfd-5hJ47DNAOKKLLIHjz1Tzq►Find full courses on: https://courses.mshah.io/►Jo...

  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1255601754675089478)** (31 条消息🔥): 

- **Mojo 编译器 Nightly 版本发布**：Nightly 更新宣布了新的 Mojo 编译器版本 `2024.6.2705`，包含多项更改，包括将 `tensor` 模块移动到 `max`，以及将 `print()` 的要求更改为 `Formattable`。提供了 [原始差异和变更日志链接](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。
- **Tensor 模块移动引发问题**：成员们讨论了将 `tensor` 模块移动到 `max` 所引起的问题，这破坏了像 BlazeSeq 这样的依赖项。建议使用 `Buffer` 和 `NDBuffer` 等替代方案，相关原因已在 [此处](https://docs.modular.com/mojo/stdlib/buffer/buffer/) 说明。
- **Static 生命周期讨论**：对话深入探讨了 `ImmutableStaticLifetime`，澄清了它属于“静态”项的生命周期，例如存储在 alias 中的内容。这允许获取 `alias` 项的引用，类似于 `let`。
- **Graph API 和索引更新**：对 Graph API 进行了改进，特别是整数面值切片和跨所有维度的切片，并为了优化设置了一些使用限制。建议使用 `ops.unsqueeze` 作为某些尚未支持的索引模式的变通方案。
- **呼吁更详细的变更日志**：成员们表示需要更详细的变更日志，特别是与 `max` 中的 API 更改相关的日志。开发者承认了这一疏忽，并承诺在未来的版本中提供更好的文档。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/buffer/buffer/">buffer | Modular 文档</a>：实现 Buffer 类。</li><li><a href="https://github.com/modularml/mojo/issues/3098">[BUG] 从具有错误类型的列表初始化的 `Tensor` 显示异常行为 · Issue #3098 · modularml/mojo</a>：错误描述：具体来说，使用 List[UIn8] 初始化的 Tensor[DType.int8] 无法正确计算其元素总数。我认为这又与隐式转换有关...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md (nightly 分支) · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md#-removed">mojo/docs/changelog.md (nightly 分支) · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modular">Modular Inc</a>：Modular 是一套集成的、可组合的工具套件，可简化您的 AI 基础设施，让您的团队能够更快地开发、部署和创新。- Modular Inc</li><li><a href="https://github.com/modul">modul - 概览</a>：modul 有 20 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/modularml/mojo/issues/3126">[BUG] `List` 在编译时无法工作。 · Issue #3126 · modularml/mojo</a>：错误描述：如题。至少 List.__getitem__ 无法工作。复现步骤 fn main(): alias l = List[Int](1, 2, 3) print(l[0]) # 打印 0 系统信息 Mojo 2024.6.2614 (366c690a) o...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1255604538715541585)** (137 条消息🔥🔥): 

- **寻找 Perplexity 的 Prompt 目录**：一名成员询问是否有与最新的基于 Agent 的 Perplexity Pro 搜索兼容的公开 Prompt 目录。他们正在寻求能够增强使用体验的资源。
- **API 问题和状态页缺失困扰用户**：几位成员抱怨收到 5xx 错误，并对 Perplexity API 缺少状态页（Status Page）表示沮丧。一位用户提到：“为什么没有关于 API 的状态页。伙计们，这可是非常基础的东西。”
- **使用限制及 AI 工具对比**：讨论集中在 Perplexity 与 Claude.ai 的区别、Sonnet 的限制，以及用户在 GPT-4 Turbo、GPT-4o 和 Claude 3.5 Sonnet 等不同模型之间的偏好。一位成员指出：“我认为他们删除了它，因为应用中已经有更好的模型了。”
- **生成式 AI 与隐私担忧**：一位用户强调了一篇文章，讨论了 Perplexity 越来越多地引用 AI 生成的来源，引发了对信息可靠性的担忧。另一个讨论集中在隐私友好性上，用户指出：“由于它使用企业级 API，似乎比直接使用 ChatGPT 更好。”
- **功能与偏好**：成员们思考了 Perplexity 功能的有效性和局限性，例如图像解读与 Google Lens 的对比，并讨论了未来的潜在增强功能，如用于更好处理文件的 Artifact 实现。有人指出：“Perplexity 实现 Artifacts 功能……将是一个轻松的胜利。”

**提到的链接**：<a href="https://www.forbes.com/sites/rashishrivastava/2024/06/26/search-startup-perplexity-increasingly-cites-ai-generated-sources/">垃圾进，垃圾出：Perplexity 传播来自垃圾 AI 博客文章的错误信息</a>：随着 Perplexity 因涉嫌剽窃新闻作品并像媒体公司一样分发而面临批评，它越来越多地引用充斥着错误信息的 AI 生成博客和 LinkedIn 帖子……

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1255613146547355728)** (8 条消息🔥): 

- **利用 Perplexity AI 战胜创伤**：一位成员分享了一个关于克服创伤的 [Perplexity AI 页面](https://www.perplexity.ai/page/Overcoming-Trauma-and-9_3ox12FRFaMON3Zk8lezQ)，暗示了其在心理健康讨论中的潜在用途。
- **朱利安·阿桑奇（Julian Assange）获释新闻**：另一位用户发布了一个涵盖朱利安·阿桑奇获释的 [Perplexity AI 页面](https://www.perplexity.ai/page/Julian-Assange-Released-cLtbci_iSxW32Xve2NgKGA)，表明了对时事和法律事务的关注。
- **探索重力的影响**：一位成员分享了一个探索重力影响的链接，你可以在[这里](https://www.perplexity.ai/search/If-Gravity-affects-20bEiugFSnudRflOAGYlnA)查看，指向了科学好奇心。
- **统一理论（Unified Theory）见解**：通过链接到 [统一理论的搜索](https://www.perplexity.ai/search/What-is-Unified-TbPeDD79TTKQPDPRpQXVFA) 展示了对理论框架的兴趣，这对从事科学研究或物理学的人很有价值。
- **Android 14 性能提升分析**：在此[页面](https://www.perplexity.ai/page/Android-14-boosts-WxN8GGdgRQSKPPn7DftxtQ)中了解 Android 14 的性能提升，重点介绍了移动操作系统开发的进展。

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1255650427219873792)** (5 条消息): 

- **用户报告 Perplexity API 错误**：一位用户询问了 **5xx 错误**，并询问是否有状态页可以检查 API 服务器的状态。另一位用户报告收到 **401 错误**，并询问其他人是否面临同样的问题。
- **讨论身份验证问题**：一位成员澄清说 **401 错误**通常与 API Key 的身份验证问题有关，但提到他们自己的使用未受影响。遇到问题的原始用户指出他们的 Key 没有更改，促使他们联系支持部门寻求帮助。

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1255607370642358332)** (63 条消息🔥🔥): 

```html
<ul>
  <li><strong>Figma AI 免费一年</strong>：根据 <a href="https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">@AustinTByrd</a> 的说法，<em>“Figma AI 在开始向所有人收费之前将免费提供一年。”</em> 完整详情请访问链接：<a href="https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Config2024 线程</a>。</li>
  
  <li><strong>会议演讲现已通过直播提供</strong>：非直播环节（如 RAG 演讲）的录像仍在等待中。同时，成员可以在 <a href="https://youtube.com/@aidotengineer">AI Engineer YouTube 频道</a>观看精选直播。</li>
  
  <li><strong>分享了 Compass 转录网站</strong>：分享了 <a href="https://aie.compasswearable.com">Compass 转录网站</a>用于查看会议转录。这些资源被认为非常有用且扎实。</li>
  
  <li><strong>LangGraph Cloud 发布</strong>：<a href="https://x.com/LangChainAI/status/1806371717084025165?t=15TNW0RaIb6EoIJ">@LangChainAI</a> 发布了 <a href="http://bit.ly/langgraph-cloud-beta-1">LangGraph Cloud</a>，为容错 Agent 提供可扩展的基础设施，并集成了追踪与监控功能。然而，一些成员对状态机是否需要专门的基础设施表示质疑。</li>
  
  <li><strong>大量可穿戴技术涌现</strong>：讨论包括了像 <a href="https://bee.computer/">Bee.computer</a> 这样的新型可穿戴设备及其录音、转录和任务执行等功能。该服务甚至提供 Apple Watch 应用，使得额外设备成为可选。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/@aidotengineer?si=KfTkCwPDCRU7jY3t">AI Engineer</a>：为 AI Engineer 提供的演讲、工作坊、活动和培训。 </li><li><a href="https://x.com/llama_index/status/1806116419995844947?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">LlamaIndex 🦙 (@llama_index) 的推文</a>：✨ 刚刚在 @aiDotEngineer 世界博览会舞台上宣布！✨ 一个用于将多 Agent AI 系统投入生产的全新框架！目前为 Alpha 版本，llama-agents 提供：⭐️ 分布式...</li><li><a href="https://x.com/LangChainAI/status/1806371717084025165?t=15TNW0RaIb6EoIJ">LangChain (@LangChainAI) 的推文</a>：🚀 介绍 LangGraph Cloud 🚀 LangGraph 帮助你构建真正起作用的可靠 Agent。今天，我们发布了 LangGraph Cloud，这是我们用于运行容错 LangGraph Agent 的新基础设施...</li><li><a href="https://www.youtube.com/watch?v=ziGNnhNABqA&t=1334s">David Luan：为什么 Nvidia 将进入模型领域 & 模型将进入芯片领域 | E1169</a>：David Luan 是 Adept 的 CEO 兼联合创始人，该公司正在为知识工作者构建 AI Agent。迄今为止，David 已为公司筹集了超过 4 亿美元...</li><li><a href="https://x.com/LangChainAI/status/1806371717084025165?t=15TNW0RaIb6EoIJKPq_IjA&s=19">LangChain (@LangChainAI) 的推文</a>：🚀 介绍 LangGraph Cloud 🚀 LangGraph 帮助你构建真正起作用的可靠 Agent。今天，我们发布了 LangGraph Cloud，这是我们用于运行容错 LangGraph Agent 的新基础设施...</li><li><a href="https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/">Gemini 1.5 Pro 2M 上下文窗口、代码执行能力以及 Gemma 2 今日可用</a>：未找到描述</li><li><a href="https://x.com/DavidKPiano/status/1806417216914817514?t=99I0TJJfrKHHDQYeiizv8A&s=19">David K 🎹 (@DavidKPiano) 的推文</a>：我喜欢 AI 初创公司如何逐渐（重新）发现用于 Agent 行为和系统的状态机和 Actor 模型。仍然不确定为什么需要专门的基础设施；这一切都只是...</li><li><a href="https://bee.computer/">Bee AI</a>：未找到描述</li><li><a href="https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Austin Byrd (@AustinTByrd) 的推文</a>：Figma AI 在开始向所有人收费之前将免费提供一年</li><li><a href="https://aie.compasswearable.com">AI Engineers 世界博览会回顾 - 由 Compass 提供支持</a>：通过实时转录和 AI 生成的摘要体验最大的技术性 AI 会议。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1255600076898963468)** (70 条消息🔥🔥): 

```html
- **成员讨论 AGI 时代的信息处理**：一位成员开玩笑说 *“如果 AGI 到来时你一次只能处理一个信息流，那就 ngmi（没戏了）”*，表达了未来多任务处理的重要性。另一位成员幽默地将正在进行的讨论称为 *“巅峰精神分裂 (PEAK SCHIZO)”*。

- **活动演示期间的技术困难**：多名成员报告了直播活动期间的听觉和屏幕共享问题。一位成员建议尝试 *“离开舞台并重新进入”*，而另一位成员则建议直接分享幻灯片以使演示更顺畅。

- **AI Engineer World Fair 的规划与协调**：成员们讨论了活动的物流和协调工作，包括确保主持人拥有必要的指南和特殊说明。分享了一个 YouTube 链接 [AI Engineer](https://www.youtube.com/@aiDotEngineer)，重点介绍了面向 AI Engineer 的演讲、工作坊和活动。

- **AI Engineer Conference 的回顾请求**：有人请求提供周日的回顾或 AI Engineer Conference 的总结。回复中强调了同时管理多个会议和活动的挑战。

- **管理活动资源和后勤**：成员们协调了海报板等资源的可用性，并确保创始人为他们的环节做好准备。发布了特殊说明以确保演讲者和嘉宾获得无缝体验，并更新了团队对转录文本和可穿戴技术的关注。
```

**提到的链接**：<a href="https://www.youtube.com/@aiDotEngineer">AI Engineer</a>：面向 AI Engineer 的演讲、工作坊、活动和培训。 

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1255599781514973315)** (75 条消息🔥🔥): 

```html
- **LM Studio 缺乏文档训练功能**：成员们澄清了 LM Studio 不支持基于文档的训练或 RAG 功能。一位成员强调：“当大多数人说‘训练’时，他们指的是将文档喂给现有模型。”
  
- **AnythingLLM 与 LM Studio 集成以实现文档摘要**：AnythingLLM 支持各种文档类型并生成简洁的摘要，与 LM Studio 无缝集成。一位用户分享道：“它完全免费且开源，无需订阅。”

- **Claude 3.5 Sonnet 被誉为顶级代码模型**：社区成员对在 Poe 和 Anthropic 上可用的 Claude 3.5 Sonnet 给予了高度评价，称其为他们进行编程辅助的“新常用工具 (daily driver)”。

- **训练 Llama 3 的要求**：关于训练 Llama 3 的讨论强调需要大量的硬件投资，特别是对于 70B 模型。一位用户解释说：“你会发现他们中的大多数是在租用的 8xH100 GPU 集群上训练的。”

- **Gemma 2 支持正在进行中**：成员们分享了关于 LM Studio 和 llama.cpp 即将支持 Gemma 2 的更新。一位用户提到：“我知道 LM Studio 的开发者正在努力尽快发布支持 Gemma 2 的版本。”
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF">bartowski/gemma-2-9b-it-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://lu.ma/yzzespyu">AI Study Group @ Block: Andrej Karpathy's Zero to GPT Hero · Luma</a>：______ 报名参加此活动时，您将通过电子邮件收到加入学习小组的邀请……</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8156">Add support for Gemma2ForCausalLM by pculliton · Pull Request #8156 · ggerganov/llama.cpp</a>：添加了对 Gemma 2 系列模型的推理支持。包括对以下模型的支持：Gemma 2 27B、Gemma 2 9B。更新了 Gemma 架构以包含 post-norm 等特性。</li><li><a href="https://llamaimodel.com/requirements">Llama 3 Requirements [What you Need to Use It] 💻</a>：Llama 3 是 AI 领域的强大力量，同时迎合了开发者和研究人员的需求。为了充分利用 Llama 3 的功能，满足特定的硬件和软件要求至关重要……
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1255604163992092775)** (23 messages🔥): 

- **DeepCoder V2 在高 RAM 配置下运行良好**：一位成员在 LM Studio 上成功运行了 **Bartkowski 的 Q4KM DeepCoder V2 230B**，使用了 8K 和 16K context，并指出使用了 **160GB RAM** 和 **64GB VRAM**。在达到 **2.68-2.8 tokens/sec** 的速度时，他们提到在尝试 32K context 时遇到了内存限制。

- **Mac Studio 在运行 DeepCoder V2 时表现挣扎**：另一位成员报告在配备 **192GB RAM 的 Mac Studio M2 Ultra** 上运行相同模型，由于内存不足导致**频繁崩溃**。在 **8K context** 下，性能表现为 **9 tokens/sec**。

- **由于内存问题导致模型加载失败**：一篇带有 **"Failed to load model"** 错误的帖子表明 GPU memory 不足以进行 offloading，建议**关闭 GPU offload** 以解决此问题。

- **Gemma 2 有限的 context window 令人失望**：**Gemma 2 在 Kaggle 上的发布**反应不一，原因是其 **4K context 限制**。尽管有人建议实际上是 **8K**，但成员们对以当今标准来看较小的 context window 表示不满。

- **Meta 发布新的 LLM Compiler 模型**：**Meta 推出了新的 LLM Compiler 模型**，具有代码优化和 compiler 能力。[更多细节和资源](https://go.fb.me/tdd3dw)已分享给感兴趣的开发者和研究人员。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/models/google/gemma-2">Google | Gemma 2 | Kaggle</a>: Gemma 是来自 Google 的一系列轻量级、先进的开放模型，基于与创建 Gemini 模型相同的研究和技术构建。</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard</a>: 未找到描述</li><li><a href="https://x.com/AIatMeta/status/1806361623831171318">来自 AI at Meta (@AIatMeta) 的推文</a>: 今天我们宣布推出 Meta LLM Compiler，这是一个基于 Meta Code Llama 构建的模型家族，具有额外的代码优化和编译器能力。这些模型可以模拟编译器，预测最优...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1dpr487/gemma_2_is_live_on_kaggle_27b_9b/">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1255858134472196136)** (1 messages): 

- **平衡对不同 AI 模型的预期**：一位成员强调需要平衡对 **GPT 和 Bard** 等不同 AI 模型的预期。他们建议针对特定需求（如 coding、讲故事和写笑话）采用不同的模型，但警告由于硬件限制，需要在**速度和质量**之间进行权衡。
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1255615156239728730)** (17 messages🔥): 

- **定制 3D 打印气流设计获得好评**：一位成员分享道，“我不知道内部看起来如何，但你可以设计一个定制的 3D 打印气流方案，”他展示了一个包含 2xP40 显卡和多个风扇的配置，在负载下保持温度在 66 度。
- **服务器的静音冷却方案**：关于如何静音冷却服务器的讨论中，有人指出，“我可以换成 Noctua 服务器风扇，这样就能很好地降低噪音水平。”
- **双 GPU 功耗见解**：在关于功耗的对话中，有人分享道，“我的 2xP40 在待机时每个功耗为 18W，”强调了不同负载下的具体功耗。
- **LM Studio 的 AVX2 要求**：有人对 “LM Studio 现在需要 AVX2” 表示惊讶并得到了确认，一位成员澄清说自 0.2.10 版本左右开始就要求支持 AVX2。
- **排查 VRAM 问题**：一位用户由于 VRAM 限制遇到了 “Failed to load model” 错误，在澄清其 GPU 共享 VRAM 后，建议升级到 3060 12GB 以更好地支持大型模型。
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1255685222679445505)** (1 messages): 

- **llama.cpp 对 Mamba-2 的支持**：关于在 llama.cpp 中跟踪 **Mamba-2** 支持的讨论。查找更新的最佳方法是搜索 [GitHub issues 页面](https://github.com/ggerganov/llama.cpp/issues/7727)。

**提及的链接**: <a href="https://github.com/ggerganov/llama.cpp/issues/7727">llama : support Mamba-2 · Issue #7727 · ggerganov/llama.cpp</a>: Mamba-2 是 Mamba 架构的新版本：博客: https://tridao.me/blog/2024/mamba2-part1-model/ 论文: https://arxiv.org/abs/2405.21060

### **LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1255598967190982757)** (1 条消息): 

- **Interpreter 禁止直接移动文件**：一位用户询问为什么他们无法直接将文档或图像拖入 Interpreter 的终端。终端似乎拒绝此类操作，实际上是发出了“禁令”，且不授予这些操作的权限。
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1255622188237852682)** (1 条消息): 

```html
- **使用 Python 提取 Token 生成数据**：一位成员询问如何利用 Python 从本地 LM Studio 服务器检索数据。他们特别关注 **Token 速度** 以及 **生成首个 Token 所需的时间**。
```
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1255694219734159420)** (28 条消息🔥): 

- **在 Python 中直接打印 stream 对象失败**：一位成员指出“不能直接打印 stream 对象”，并建议对其进行迭代。他们提供了一个 Python 代码片段来演示正确用法：*"for token in llm.stream(input_text): print(token.content,end='')"*。
- **针对用户查询检索相关向量**：一位成员描述了一个问题：当用户从列表中选择一个选项时，尽管最初检索到了相关向量，但检索系统却获取了不相关的向量。他们讨论了潜在的解决方案，例如“在聊天历史中保留之前的检索结果”，或者使用另一位成员建议的 `query_knowledge_base("Green light printer problem")` 函数。
- **在 Streamlit 中使用 `astream_events`**：当被问及如何将 `astream_events` 与 Streamlit 集成时，回复强调了提供的知识库中缺乏具体示例。建议参考 LangChain [文档](https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events)以获取更多详情。
- **将 LLM 响应转换为移动应用的音频**：一位成员分享了他们的方法，即使用 Google Text-to-Speech 将 LLM 文本响应转换为音频文件并发送回设备。他们询问 Gemini 是否支持直接以音频格式流式传输文本响应。
- **使用 MemGPT 与 LangGraph 的示例代码**：有人请求示例代码，展示如何使用 MemGPT 配合 LangGraph 等 Agent 实现无限上下文记忆。他们对涉及开源 LLM 和 OpenAI 的实现都感兴趣。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://192.168.1.70:11434")`">未找到标题</a>：未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/17703>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/streaming/#filtering-events>)">如何流式传输 runnable | 🦜️🔗 LangChain</a>：本指南假设你已熟悉以下概念：</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/agents/#streaming-tokens>).">构建一个 Agent | 🦜️🔗 LangChain</a>：本指南假设你已熟悉以下概念：
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1255814096322957312)** (70 条消息🔥🔥): 

- **使用并行处理构建 LangChain 端点**：成员们讨论了如何使用 `add_routes` 构建端点，并通过 `RunnableParallel` 启用并行处理。分享了大量的文档和示例，包括 FastAPI 和语言模型的代码片段。

- **在 LangChain 服务端提供文档**：社区探索了如何使用 `load_qa_chain()` 在服务端提供文档，并给出了 `run()` 和基于字典的方法的示例。这包括使用 `map_reduce` 链类型处理并行处理。

- **处理 LangChain 中的高并发请求**：提出了关于管理 RAG 端点 100 个并发请求的问题，并就使用 Flask 或 FastAPI 等现代 Web 服务器进行并发请求处理提供了指导。建议通过配置增加 worker 进程或线程以获得更好的性能。

- **LangChain Expression Language (LCEL) 的功能**：详细讨论了 LCEL 的特性，如一流的流式传输支持、异步支持、优化的并行执行和重试机制。提供了文档链接以供更全面的理解。

- **LangChain chains 中的并发处理**：社区寻求关于 `RetrievalQA` 等 chains 中的 `invoke()` 方法是否支持并发处理的澄清。经澄清，虽然 `invoke()` 处理单个请求，但整体并发能力取决于服务器配置，以独立处理多个并发请求。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://js.langchain.com/v0.2/docs/how_to/sequence/#coercion>).">如何链接 Runnable | 🦜️🔗 Langchain</a>: 关于 [LangChain Expression</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/functions/#next-steps>).">如何运行自定义函数 | 🦜️🔗 Langchain</a>: 本指南假设您熟悉以下概念：</li><li><a href="https://python.langchain.com/v0.2/docs/langserve/#endpoints>)">🦜️🏓 LangServe | 🦜️🔗 LangChain</a>: 发行说明</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/lcel_cheatsheet/#runnableparallel>)">LangChain Expression Language 备忘单 | 🦜️🔗 Langchain</a>: 这是所有最重要的 LCEL 原语的快速参考。</li><li><a href="https://github.com/langchain-ai/langchain/issues/11433>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/12423>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/7876>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13696>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/1145>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/8399>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/16980>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/20492>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/retrievers/flashrank-reranker/#qa-reranking-with-flashrank>)">FlashRank 重排序器 | 🦜️🔗 LangChain</a>: FlashRank 是一个超轻量且超快速的 Python 库，用于为您现有的搜索和检索流水线添加重排序功能。它基于 SoTA cross-encoders，感谢所有模型所有者...</li><li><a href="https://github.com/langchain-ai/langchain/issues/9865>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/4950>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/#langchain-expression-language-lcel>)">操作指南 | 🦜️🔗 LangChain</a>: 在这里，您可以找到“我该如何……？”这类问题的答案。</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel>)">概念指南 | 🦜️🔗 LangChain</a>: 本节包含对 LangChain 关键部分的介绍。</li><li><a href="https://js.langchain.com/v0.2/docs/concepts/#langchain-expression-language>)">概念指南 | 🦜️🔗 Langchain</a>: 本节包含对 LangChain 关键部分的介绍。</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/rag/#built-in-chains>)">构建检索增强生成 (RAG) 应用 | 🦜️🔗 LangChain</a>: LLM 实现的最强大应用之一是复杂的问答 (Q&amp;A) 聊天机器人。这些应用可以回答有关特定源信息的问题。这些 ...</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/providers/dspy/#normal-lcel>)">DSPy | 🦜️🔗 LangChain</a>: DSPy 是一个出色的 LLM 框架，它引入了一个自动编译器，教导 LM 如何执行程序中的声明式步骤。具体来说，DSPy 编译器将在内部跟踪...</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/rag/#retrieval-and-generation-generate>)">构建检索增强生成 (RAG) 应用 | 🦜️🔗 Langchain</a>: 最强大的应用之一

由 LLM 启用的交互非常复杂
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1255818992447258685)** (4 条消息): 

- **Merlinn 作为 Slack 值班开发机器人首次亮相**：介绍 [Merlinn](https://github.com/merlinn-co/merlinn)，这是一个开源 AI 工具，通过连接各种工具并提供根因分析（root cause analysis）来协助**排查生产事故**。开发者强调其通过利用 LangChain 工作流来提高值班效率的能力，并邀请大家在 GitHub 上提供反馈和点赞（stars）。
  
- **Evidently AI 分享 ML 系统设计案例研究**：一个包含 [450 个案例研究的综合 Airtable](https://www.evidentlyai.com/ml-system-design) 详细介绍了来自 100 多家公司的 ML 和 LLM 系统的实际应用和设计心得。该数据库包含按行业和 ML 使用场景分类的过滤器，并带有标签以帮助用户快速找到相关的研究。

- **ZenGuard AI 为 LangChain 增加安全功能**：与 [ZenGuard AI](https://python.langchain.com/v0.2/docs/integrations/tools/zenguard) 的新集成包括 Prompt 注入保护、越狱（jailbreak）预防和数据泄露预防等功能。ZenGuard AI 旨在保护应用程序免受恶意活动和未经授权的访问，并邀请在 GitHub 上提供反馈。

- **关于无代码 Chrome 扩展聊天机器人的 YouTube 教程**：一段 [YouTube 视频](https://www.youtube.com/watch?v=-OKC7CY2bbQ) 展示了如何使用 Visual LangChain 创建无代码 Chrome 扩展聊天机器人。该演示展示了如何设计一个具有交互式聊天功能的 LangChain RAG 应用程序。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.evidentlyai.com/ml-system-design">Evidently AI - ML 系统设计：450 个案例研究</a>：顶尖公司如何应用 ML？我们建立了一个包含来自 100 多家公司的 450 个案例研究的数据库，涵盖了实际的 ML 使用场景和设计 ML 系统的经验教训。</li><li><a href="https://www.youtube.com/watch?v=-OKC7CY2bbQ">使用 Visual LangChain 的无代码 Chrome 扩展聊天机器人</a>：在此演示中，我展示了 Visual Agents 的一个令人兴奋的新功能，你可以设计你的 LangChain RAG 应用程序，包括一个交互式聊天功能...</li><li><a href="https://github.com/merlinn-co/merlinn">GitHub - merlinn-co/merlinn: 开源 AI 值班开发人员 🧙‍♂️ 在几秒钟内获取有关生产事故的相关上下文和根因分析，让值班工程师效率提升 10 倍 🏎️</a>：开源 AI 值班开发人员 🧙‍♂️ 在几秒钟内获取有关生产事故的相关上下文和根因分析，让值班工程师效率提升 10 倍 🏎️ - merlinn-co/merlinn</li><li><a href="https://zenguard.ai)">无标题</a>：未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/new?assignees=&labels=03+-+Documentation&projects=&template=documentation.yml&title=DOC%3A+%3CIssue+related+to+/v0.2/docs/integrations/tools/zenguard/%3E&url=https://python.langchain.com/v0.2/docs/integrations/tools/zenguard/">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、分叉并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 条消息): 

emarco: https://www.youtube.com/watch?v=Q_yKRLACx78&t=1s
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1255675507643519060)** (2 条消息): 

- **LlamaIndex 发布新的多 Agent AI 框架**：在 AI Engineer World's Fair 上宣布，LlamaIndex 推出了 **llama-agents**，这是一个用于在生产环境中部署多 Agent AI 系统的新框架。Alpha 版本提供了一个分布式的、面向服务的架构，通过标准 HTTP APIs 进行通信。[Twitter 公告](https://twitter.com/llama_index/status/1806116419995844947)。

- **LlamaCloud 开启候补名单**：LlamaIndex 已为 **LlamaCloud**（其全托管的数据摄取服务）开启了候补名单。感兴趣的用户可以[注册](https://cloud.llamaindex.ai/)，并受邀分享他们的电子邮件地址，以便以受控的速度获得访问权限。[更多详情](https://twitter.com/llama_index/status/1806117132956299497)。

**提到的链接**：<a href="https://t.co/kAY9YEmOkx">LlamaCloud 候补名单</a>：感谢您对 LlamaCloud 的关注！请在下方注册并告诉我们您使用的电子邮件地址，我们将以受控的速度允许人员进入。

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1255599153879453798)** (56 条消息🔥🔥): 

- **用户质疑 Readers 映射中缺少 JSONReader**：多位用户讨论了为什么 LlamaIndex 的默认文件扩展名 Readers 映射中不包含 JSONReader。一位成员建议提交 PR 来添加该映射，另一位成员指出可以通过传递自定义的 `file_extractor` 来进行覆盖。

- **LlamaParse 存在幻觉问题**：一位用户报告称 LlamaParse 在处理财务文档时表现优于 GPT-4，但在基础信息上仍会出现幻觉。该用户被要求发送文件以便进行调试。

- **新 AI 栈发布公告**：用户讨论了 LlamaIndex 新 AI 栈的发布，并分享了[博客文章链接](https://www.llamaindex.ai/blog/introducing-llama-agents-a-powerful-framework-for-building-production-multi-agent-ai-systems)。

- **BM25 需要频繁重新索引**：讨论围绕添加新文档时需要重新索引 BM25 的必要性展开，有建议指出这一过程效率较低。推荐了如 Splade 等替代的稀疏嵌入（sparse embedding）方法。

- **数据摄取流水线（ingestion pipelines）的性能问题**：一位用户指出 LlamaIndex 数据摄取流水线中的大型文档管理会导致显著的性能下降。他们建议在更新参考文档信息之前批量删除节点，这一想法被视为一种潜在的增强功能而受到欢迎。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.llamaindex.ai/blog/introducing-llama-agents-a-powerful-framework-for-building-production-multi-agent-ai-systems">Introducing llama-agents: A Powerful Framework for Building Production Multi-Agent AI Systems — LlamaIndex, Data Framework for LLM Applications</a>：LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。</li><li><a href="https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb">RetrievalTutorials/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb at main · FullStackRetrieval-com/RetrievalTutorials</a>：通过在 GitHub 上创建账号来为 FullStackRetrieval-com/RetrievalTutorials 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_index/blob/a24292c79424affeeb47920b327c20eca5ba85ff/llama-index-core/llama_index/core/storage/docstore/keyval_docstore.py#L485),">llama_index/llama-index-core/llama_index/core/storage/docstore/keyval_docstore.py at a24292c79424affeeb47920b327c20eca5ba85ff · run-llama/llama_index</a>：LlamaIndex 是一个用于 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/pull/14419">added JSONReader to file mappings by denen99 · Pull Request #14419 · run-llama/llama_index</a>：描述：将文件扩展名映射到相应 XReader 的 default_file_reader_cls 字典中不包含新 JSONReader 的映射。此 PR 为 .json 文件添加了映射...</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/response_synthesizers/#llama_index.core.response_synthesizers.type.ResponseMode.REFINE>).">Index - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/response_synthesizers/#llama_index.core.response_synthesizers.type.ResponseMode>).">Index - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/json.py#L51">llama_index/llama-index-core/llama_index/core/readers/json.py at main · run-llama/llama_index</a>：LlamaIndex 是一个用于 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/file/base.py#L69">llama_index/llama-index-core/llama_index/core/readers/file/base.py at main · run-llama/llama_index</a>：LlamaIndex 是一个用于 LLM 应用的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1255656405982253197)** (13 条消息🔥): 

```html
- **OpenAI API 销售额超过 Microsoft Azure**: OpenAI 目前通过 API 销售产生的收入已超过 Microsoft 在 Azure 上转售所获得的收入。这一消息由 Aaron P. Holmes 通过推文分享，凸显了市场动态中令人惊讶的转变。[来源](https://x.com/aaronpholmes/status/1806312654505443347?s=46)
- **Meta 发布用于代码优化的 LLM Compiler**: Meta 推出了 **Meta Large Language Model Compiler**，旨在利用预训练模型处理编译器优化任务。该套件专注于 LLVM-IR 和汇编代码，利用了包含 5460 亿个 token 的庞大语料库。[研究出版物](https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/)
- **Character.AI 推出 Character Calls**: Character.AI 推出了 **Character Calls**，允许用户与 AI 角色进行语音对话。该功能可通过其应用程序访问，旨在创造更具沉浸感的 AI 体验，但在性能和流畅度方面收到了褒贬不一的评价。[博客文章](https://blog.character.ai/introducing-character-calls/)
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.character.ai/introducing-character-calls/">Introducing Character Calls</a>: 0:00 / 0:07 1× 呼叫 Character.AI 社区！我们很高兴能推出一项令人兴奋的新功能，它将重新定义您的 Character.AI 体验：Character Calls！...</li><li><a href="https://x.com/giffmana/status/1806411302190915603?s=46">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>: @fouriergalois @character_ai 刚试了一下，不幸的是它无法相提并论，本来会非常令人印象深刻的！一点也不流畅。我说话结束时有 5 秒的延迟。我无法打断...</li><li><a href="https://huggingface.co/collections/facebook/llm-compiler-667c5b05557fe99a9edd25cb">LLM Compiler - facebook 集合</a>: 未发现描述</li><li><a href="https://x.com/aaronpholmes/status/1806312654505443347?s=46">来自 aaron holmes (@aaronpholmes) 的推文</a>: 新消息：OpenAI 现在从其 API 销售中获得的收入超过了 Microsoft 在 Azure 上转售所获得的收入 https://www.theinformation.com/articles/in-a-surprise-openai-is-selling-more-of-its-ai-models-than-...</li><li><a href="https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/">未发现标题</a>: 未发现描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1255630429231517767)** (15 条消息🔥): 

- **面试挑战难度被认为过高**：一位成员分享了一位朋友的经历，其面试的在线评估包含“Codeforces 最高难度的题目”，这看起来“有点过分”。该成员还提到，由于面试过程中要求进行“有偿劳动”，可能存在违反合同的情况。
  
- **编程面试的随意门槛**：另一位成员表达了挫败感，尽管在编程面试中表现良好，但未能通过面试官设定的“随意门槛”。这种情绪引起了另一位成员的共鸣，他很高兴自己不是唯一面临这个问题的人。

- **ChatGPT Subreddit 中的高级语音功能**：一条推文 ([AndrewCurran](https://x.com/AndrewCurran_/status/1806178276001329373)) 报道称，ChatGPT Subreddit 上的某人声称获得了高级语音功能的访问权限，该功能可以在生成语音的同时生成音效。这个案例被分享为“帖子中包含有趣的音频”。

- **Chain of Thought 提示策略专利**：成员们在看到一条推文 ([andrewwhite01](https://x.com/andrewwhite01/status/1806347002126446736?s=46)) 后，讨论了发现的“Chain of Thought 提示策略”专利。他们质疑该专利是已授予还是仅在申请中，并辩论了此类专利在实际中的可执行性。

- **Google 对 Transformer 架构专利的非强制执行**：有人指出 Google 拥有 Transformer 架构的专利，但“并未行使该权利”——这引发了人们对为何不追究法律责任的好奇，猜测倾向于认为在法庭上可能会失败。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/james-bond-007-tomorrow-never-dies-elliot-carver-stamper-gif-24934311">James Bond 007 GIF - James Bond 007 Tomorrow Never Dies - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/andrewwhite01/status/1806347002126446736?s=46">来自 Andrew White 🐦‍⬛/acc (@andrewwhite01) 的推文</a>：长见识了，Chain of Thought 提示竟然有专利。没意识到提示策略也是可以申请专利的。</li><li><a href="https://x.com/AndrewCurran_/status/1806178276001329373">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：Reddit ChatGPT 版块的某人声称今天早上获得了高级语音权限，在失去权限之前，他们录下了 4o 讲故事。有趣的部分是 4o 生成了配套的...</li><li><a href="https://patents.google.com/patent/US20230394328A1/en?oq=US+2023/0394328+A1">US20230394328A1 - Prompting Machine-Learned Models Using Chains of Thought 
      - Google Patents</a>：未发现描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1255684248225189961)** (23 messages🔥): 

- **猜鞋游戏难倒成员**：一位成员分享了一个有趣的挑战，让大家*“通过鞋子猜著名的 ML 人物”*。讨论中辨认出了一些错误的猜测，有人称 *“那肯定不是 Sam。Sam 穿的是 Nike”*，并分享了一个推文链接，猜测其发型变化等造型更新：[Twitter](https://x.com/1vnzh/status/1802093900993073611)。

- **Cohere 创始人的酷范儿**：成员们讨论了 Cohere 的创始人，分享了一段视频采访和一条推文，赞扬了他们对 AI 解决方案的务实关注，并强调了他们像摇滚明星般的地位。分享了完整视频链接：[YouTube](https://www.youtube.com/watch?v=4JF1V2hzGKE)。

- **关于采访的趣味调侃**：一位成员幽默地剖析了 OpenAI 的 Sam Altman 和 Airbnb 的 Brian Chesky 的 YouTube 采访，指出提问似乎有些偏心，针对 Sam 的问题更严肃，而针对 Brian 的问题则更轻松随意。分享链接：[YouTube](https://www.youtube.com/watch?v=8e8RpbO2lNU)。

- **各种幽默链接**：分享了各种好玩且离题的链接，包括一段引用了 Cohere 创始人的梗视频，以及《谍影重重 2》（Bourne Supremacy）的电影片段：[YouTube](https://youtu.be/I3znSbbu9IU?si=EbbsoUgHAFS1wuMY&t=65)。另一条推文幽默地将 Cohere 的 CTO 描绘成一名在压力下表现出色的高水平玩家：[Twitter](https://x.com/internetvin/status/1800019341343084662)。

- **Emoji 带来的语境统一**：一位成员理论化了某个帖子中 Emoji 使用的隐含意义，认为它传达了某种心照不宣的共识或语境：*“你猜对了我们没明说的事”*。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=ok5XRYhgH9Q">imagine if ninja got a low taper fade</a>：未找到描述</li><li><a href="https://x.com/internetvin/status/1800019341343084662">来自 internetVin (@internetvin) 的推文</a>：以第一视角对抗 Cohere 的 CTO @1vnzh，当他拿着 AWP 背水一战时，脑海中浮现出关于 Cohere 成员在演示中垫底的 BetaKit 头条新闻...</li><li><a href="https://x.com/youraimarketer/status/1805629336973688853">来自 Muratcan Koylan (@youraimarketer) 的推文</a>：我看好 Cohere，因为他们的创始人不仅谈吐像摇滚明星，而且确实是摇滚明星。引用 Cohere (@cohere) —— 在这段片段中，@MLStreetTalk 与 Cohere 联合创始人 Nick Fro... 进行了交流。</li><li><a href="https://www.youtube.com/watch?v=8e8RpbO2lNU">Lester Holt 采访 OpenAI 的 Sam Altman 和 Airbnb 的 Brian Chesky</a>：OpenAI 首席执行官 Sam Altman 和 Airbnb 联合创始人兼首席执行官 Brian Chesky 加入 NBC 新闻的 Lester Holt，共同探讨人工智能的益处...</li><li><a href="https://youtu.be/I3znSbbu9IU?si=EbbsoUgHAFS1wuMY&t=65">《谍影重重 2》(9/9) 电影片段 - 给 Pamela 的最后一通电话 (2004) HD</a>：电影片段链接...</li><li><a href="https://x.com/aidangomez/status/1797900448776822948">来自 Aidan Gomez (@aidangomez) 的推文</a>：和 Cohere 的 xrisk 安全团队度过了一天，密切关注 Command R++ 的训练。</li><li><a href="https://x.com/1vnzh/status/1802093900993073611">来自 Ivan Zhang (@1vnzh) 的推文</a>：如果 Aidan Gomez 剪了个低渐变发型会怎样。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1255766574044938240)** (3 messages): 

- **关于“bases”术语的澄清**：一位成员询问了最近一篇合成数据文章中“bases”一词的含义。另一位成员澄清说，它指的是 **base models**（基础模型）。
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1255855049050161172)** (1 条消息): 

- **Stheno 8B 在 OpenRouter 首次亮相**：**[L3 Stheno 8B 32K](https://openrouter.ai/models/sao10k/l3-stheno-8b)** 现已在 OpenRouter 上线。该模型由 **OpenRouter, LLC** 为 2023-2024 年度推出。
- **本周风味 (Flavor of the Week)**：**Stheno 8B** 被选为 OpenRouter 的 **[本周风味模型](https://openrouter.ai/models/openrouter/flavor-of-the-week)**。在 OpenRouter, LLC 2023-2024 年度的推广下，该模型持续吸引着用户的关注。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/sao10k/l3-stheno-8b)">sao10k 开发的 Llama 3 Stheno 8B v3.3 32K</a>：Stheno 8B 32K 是来自 [Sao10k](https://ko-fi.com/sao10k) 的创意写作/角色扮演模型。它在 8K 上下文下进行训练，随后扩展至 32K 上下文。与旧版 Stheno 相比，该模型……</li><li><a href="https://openrouter.ai/models/openrouter/flavor-of-the-week>)!">sao10k 开发的 Flavor of The Week</a>：这是一个每周轮换底层模型的路由模型。它旨在提供一种简单的方式，在保持相同模型 ID 的同时探索新模型的能力。当前的底层模型是 [L...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1255767355569475686)** (39 条消息🔥): 

- **NVIDIA Nemotron 页面问题困扰用户**：一名用户报告称，在手机上尝试选择 **NVIDIA Nemotron** 时出现“页面无法运行”的错误，尽管另一名用户似乎可以正常使用。他们指出，如果这只是手机的个别问题，那就不是什么大碍。

- **OpenRouter API key 兼容性咨询**：一位用户询问原本需要 OpenAI key 的应用程序是否可以使用 **OpenRouter API key**。建议他们尝试覆盖基础 API URL，不过具体解决方案可能因应用程序而异。

- **审查担忧下的模型推荐**：一名用户请求推荐无审查模型。建议使用 **Cmd-r** 和 **Euryale 2.1**（微调后的 Llama 3），并提到 **Magnum** 正等待加入 OpenRouter，同时还强调了**越狱版 Claude 3**。

- **Google Gemini API 更新令开发者兴奋**：分享的一篇 [Google 博客文章](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/) 宣布为 **Gemini 1.5 Pro** 提供 **200 万 token 上下文窗口**，以及代码执行能力。此次更新旨在通过上下文缓存（context caching）帮助开发者管理输入成本。

- **Claude 3 网页版的 Artifacts 功能引发关注**：用户希望在 OpenRouter 上拥有类似 Anthropic 的 Artifacts 功能。建议称 **Sonnet-3.5** 可能会通过常规提示词生成代码的方式提供部分替代方案。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ai.google.dev/gemini-api/docs/code-execution?lang=node">未找到标题</a>：未找到描述</li><li><a href="https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/">Gemini 1.5 Pro 2M 上下文窗口、代码执行能力以及 Gemma 2 今日上线</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1255728961040678942)** (23 条消息🔥): 

- **OpenRouter 利用 Cohere API**：成员们讨论了 OpenRouter 使用 **Cohere API** 来避免违反许可证中的 NC（非商业）条款。一位成员确认道：*"如果他们使用 Cohere API，NC 限制就不适用。"*

- **Command-R 模型在 OpenRouter 上的独特性**：指向 OpenRouter 页面的链接强调了 **Command-R** 模型可通过 [OpenRouter](https://openrouter.ai/models/cohere/command-r/status) 获取，并且在一篇 Patreon 帖子中提到它是为 *"I'm All In Subscribers"* 提供的。该模型因其创造力和指令遵循（prompt-following）能力而受到认可。

- **关于 Command-R 许可问题的争议**：有人担心 SpicyChat 可能在滥用许可证，并就 **Command-R** 的使用是否符合 NC 许可证进行了深入讨论。尽管如此，成员们澄清向 Cohere 支付费用应该可以解决任何许可问题。

- **Colab 脚本上的 Tool Use 错误已解决**：一位成员解决了在 Colab 和本地 PyCharm 上运行 Cohere API 脚本时遇到的问题。通过参考 [关于多步工具使用的 Cohere 文档](https://docs.cohere.com/docs/multi-step-tool-use#step-2-ask-model-for-tool-calls-and-send-back-tool-results)，该用户修正了错误并使脚本完美运行。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/cohere/command-r/activity">Cohere: Command R – 最近活动</a>：查看 Cohere: Command R 的最近活动和使用统计数据 - Command-R 是一个 35B 参数模型，能以更高质量、更可靠且更长的上下文执行对话语言任务...</li><li><a href="https://openrouter.ai/models/cohere/command-r/status">Cohere: Command R – 提供商状态和负载均衡</a>：查看提供商状态并向 Cohere: Command R 发起负载均衡请求 - Command-R 是一个 35B 参数模型，能以更高质量、更可靠且...</li><li><a href="https://docs.cohere.com/docs/multi-step-tool-use#step-2-ask-model-for-tool-calls-and-send-back-tool-results">多步工具使用 (Agents)</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1255618654142202156)** (2 条消息): 

- **Rig 激励开发者反馈**：一位成员宣布发布 **Rig**，这是一个用于构建 LLM 驱动应用程序的 Rust 库，并推出了一个激励性反馈计划，开发者在构建用例并提供关于该库的反馈时会获得奖励。他们询问在频道中发布详细信息是否合适，并被建议确保该库支持 Cohere 的模型。
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1255657271048798269)** (15 条消息🔥): 

- **加入 Block 的 Mission 办公室参加 GPT 深度探讨**：在旧金山 Block 的 Mission 办公室举办的一场活动，提供为期 4 周的免费学习小组，基于 Andrej Karpathy 的 [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) YouTube 系列。在此处[报名](https://lu.ma/yzzespyu)，并通过此 [Google 表单](https://forms.gle/L4u3TMfTs5TjqWpt7)加入学习小组。

- **探索适用于 Interpreter 的开源模型**：一位用户询问了适用于 Interpreter 的最佳开源模型，提到了 **GPT-4o**，并寻求关于使用 **Ollama** 或 **Groq** 进行本地部署的建议。该问题反映了利用开源模型优化 Interpreter 的普遍兴趣。

- **对 GitHub 可接受使用政策的担忧**：一位用户对某个项目可能违反 GitHub 的可接受使用政策（Acceptable Use Policies）和 DMCA 移除政策（DMCA Takedown Policy）表示担忧。他们强调在提交正式通知之前，需要公开讨论这些问题。

- **Meta 发布 LLM Compiler**：Meta 发布了 **LLM Compiler**，其特点是基于 **Meta Code Llama** 构建的模型，用于优化和反汇编代码，并可针对各种任务进行微调。公告包含了 [HuggingFace 仓库](https://go.fb.me/tdd3dw)和[研究论文](https://go.fb.me/85zwgy)的链接，模型在允许研究和商业使用的许可协议下发布。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/aiatmeta/status/1806361623831171318?s=46">来自 AI at Meta (@AIatMeta) 的推文</a>: 今天我们发布了 Meta LLM Compiler，这是一个基于 Meta Code Llama 构建的模型家族，具有额外的代码优化和编译器能力。这些模型可以模拟编译器，预测最优方案...</li><li><a href="https://x.com/mindbranches/status/1806370172506091843?s=46">来自 MindBranches (@MindBranches) 的推文</a>: @AIatMeta 完整研究论文摘要："Meta Large Language Model Compiler: Foundation Models of Compiler Optimization"</li><li><a href="https://lu.ma/yzzespyu">Block 的 AI 学习小组：Andrej Karpathy 的 Zero to GPT Hero · Luma</a>: ______ 报名参加此活动时，您将收到电子邮件邀请，通过以下方式加入学习小组……
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1255609762825900043)** (5 条消息): 

- **01 不再提供 `--local` 选项**：一位成员指出，在最新版本中，01 不再提供 `--local` 选项。这引发了关于目前可用模型的关注。
- **Interpreter 购买物流**：一位用户询问是否可以使用朋友的地址购买 01 Interpreter 然后寄往西班牙。他们还询问了其在西班牙语环境下的功能表现。
- **模型可用性与使用**：针对最新版本中可用的模型提出了疑问，特别是询问是否仅支持 GPT-4。他们还质疑使用是否绑定到 OpenAI API keys，或者使用 20 欧元的订阅账号登录是否足够。
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1255598742732800100)** (13 条消息🔥): 

- **发现 NCCL Watchdog CUDA 错误**：一名成员报告称，他们遇到了与非法内存访问相关的 **CUDA error** 导致的 **NCCL watchdog thread termination**。他们建议通过传递 `CUDA_LAUNCH_BLOCKING=1` 进行调试，并通过编译时使用 `TORCH_USE_CUDA_DSA` 来启用设备端断言。

- **Gemma 2 发布详情**：[Gemma 2 发布了！](https://x.com/_philschmid/status/1806343336292229234?s=46) **Google 的 Gemma 2** 模型提供 9B 和 27B 两种尺寸，具有 sliding window attention 和 logit soft-capping 等先进功能。该模型的评分与 **Meta 的 Llama 3 70B** 相当，显示出强劲的性能指标。

- **Meta LLM Compiler 发布公告**：[Meta 宣布了 LLM Compiler 模型](https://x.com/aiatmeta/status/1806361623831171318?s=46)，该系列基于 **Meta Code Llama** 构建，专注于代码优化和编译器任务。这些模型取得了 state-of-the-art 的结果，并以允许研究和商业使用的许可协议发布。

- **Gemma 2 Transformer 代码问题**：讨论强调了 **Transformers 代码中的问题** 影响了 Gemma 2 的 sample packing。目前已提交了一个 [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718) 来解决这些问题，正等待 Hugging Face 的上游修复。

- **社区对 HF Bug 的调侃**：社区对典型的 **Hugging Face bugs** 表示无奈，特别提到了 **Gemma2DecoderLayer.forward** 在 sliding window 操作中损坏了 attention mask。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/_philschmid/status/1806343336292229234?s=46">Philipp Schmid (@_philschmid) 的推文</a>: Gemma 2 发布了！Google 刚刚发布了其开源 LLM 的下一次迭代！Gemma 2 有 9B 和 27B 两个版本，在 13T token 上训练。Gemma 2 27B 的性能接近 @AIatMeta 的 Llama 3 70B！...</li><li><a href="https://x.com/aiatmeta/status/1806361623831171318?s=46">AI at Meta (@AIatMeta) 的推文</a>: 今天我们发布了 Meta LLM Compiler，这是一个基于 Meta Code Llama 构建的模型家族，具有额外的代码优化和编译器能力。这些模型可以模拟编译器，预测最佳...</li><li><a href="https://x.com/mindbranches/status/1806370172506091843?s=46">MindBranches (@MindBranches) 的推文</a>: @AIatMeta 完整研究论文摘要："Meta Large Language Model Compiler: Foundation Models of Compiler Optimization"</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718">winglian 支持 gemma2 的 sample packing · Pull Request #1718 · OpenAccess-AI-Collective/axolotl</a>: 描述、动机和背景、如何测试、截图（如果适用）、变更类型、社交账号（可选）
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1255625370359828540)** (1 条消息): 

- **Mistral7B 喜欢重复自己**：一位成员分享了在进行全量指令微调（full instruction-tuning）时 **Mistral7B** 出现的问题，即模型有时会重复句子或段落，即使在高 temperature 设置下也是如此。他们澄清数据集并不包含此类实例，并寻求关于潜在原因的建议。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1255930674641113149)** (2 条消息): 

- **Gemma2:9B 模型在首次测试中表现出色**：一段名为 ["Gemma2 First Test ! Incredible Results"](https://youtu.be/6SLuneidHYw) 的视频展示了使用 **ollama** 安装和测试 **Gemma2:9B 模型** 的过程。描述强调了这次初步测试中令人难以置信的结果。
- **Gemma2:27B 差强人意**：另一段名为 ["Gemma2:27B First Test ! How Can it be THAT Bad ?!"](https://youtu.be/vIKNRiVxBeo) 的视频展示了对 **Gemma2:27B 模型** 的测试，该模型是 **Google** 在一小时前发布的。视频指出了这个较大模型在性能上的显著令人失望之处。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/6SLuneidHYw">Gemma2 First Test ! Incredible Results</a>: 今天，我们将使用 ollama 安装并测试 Gemma2</li><li><a href="https://youtu.be/vIKNRiVxBeo">Gemma2:27B First Test ! How Can it be THAT Bad ?!</a>: 让我们用 ollama 测试一下 Google 一小时前发布的 gemma2 的最大版本 (27B)
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1255607970172112896)** (8 messages🔥): 

- **观看 PyTorch 纪录片**：分享了一个名为 [Official PyTorch Documentary: Powering the AI Revolution](https://www.youtube.com/watch?v=rgP_LBtaUEc) 的 YouTube 视频。该纪录片讲述了 **PyTorch 诞生**的真实故事，以及由一群敬业的工程师进行的开发历程。

- **FPGA 设计选择澄清**：一位成员澄清说，他们没有使用 **Xilinx/AMD FPGAs**，并强调在 FPGA 上实例化的设计对于 **所有 Transformer 模型都是通用的**。他们表示，该 FPGA 可以加载 Huggingface Transformer 库中的任何模型，而无需专门的 RTL。

- **tinygrad 项目贡献**：一位成员提到在 **tinygrad 中使用 SDXL** 进行生成，并指出仍需进行一些清理和性能增强工作。他们计划在准备就绪后开启一个 PR，将其合并到 upstream 的 examples 中。

- **今日演讲**：George Hotz 宣布今天预定有一个 **8 分钟的演讲**，但未说明具体细节。

- **tinygrad 匹配引擎悬赏问题**：重点介绍了一个旨在提高 [匹配引擎速度 (matching engine's speed)](https://github.com/tinygrad/tinygrad/issues/4878) 的 500 美元悬赏 Issue。鼓励成员阅读 PR 以确定是否已有其他人在处理该问题。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=rgP_LBtaUEc">Official PyTorch Documentary: Powering the AI Revolution</a>：这部影片揭示了 PyTorch 诞生的真实叙事，将其存在归功于一群推动技术创新的无名英雄...</li><li><a href="https://github.com/tinygrad/tinygrad/issues/4878">Matching engine is slow ($500 bounty) · Issue #4878 · tinygrad/tinygrad</a>：重写匹配引擎，使我的机器（Intel Core Ultra 7 155h）在 "model lower" 时获得 2 倍的加速。jesse@x1:~/tinygrad$ PROFILE=0 python3 test/external/external_benchmark_schedule.py *...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1255701453864042549)** (5 messages): 

- **tinygrad 上的 MultiheadAttention 移植问题**：一位成员询问有关将 **PyTorch 的 MultiheadAttention 移植到 tinygrad** 的示例，其中 `in_proj_bias` 和 `in_proj_weight` 作为单个参数。他们随后发现 `pytorch.nn.functional.linear` **期望权重是一个预转置矩阵 (pre-transposed matrix)**。

- **估算模型训练的 VRAM**：另一位成员询问是否可以创建一个 **NOOP backend** 来轻松估算总内存分配。他们引用了一条推文，讨论了如何**估算给定参数规模的模型训练所需的 VRAM** [来源](https://x.com/skalskip92/status/1806293661014958330)。

- **理解 tinygrad 中的 shapetracker**：通过学习 [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes/)，一位成员解释了 **Shapetracker 如何通过在不改变底层内存的情况下表示各种多维结构，从而实现零成本的移动操作 (movement operations)**。他们寻求关于 **shapetracker 中 mask** 作用的进一步澄清。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html">How ShapeTracker works</a>：tinygrad 教程</li><li><a href="https://x.com/skalskip92/status/1806293661014958330">SkalskiP (@skalskip92) 的推文</a>：有没有人有好的方法来估算训练具有 X 数量参数的模型需要多少 VRAM？
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1255600251440730267)** (2 messages): 

- **Anthropic 的 Claude 竞赛备受关注**：分享了 [Anthropic 文档](https://docs.anthropic.com/en/build-with-claude-contest/overview) 的链接。该文档概述了围绕使用 Claude 进行构建的竞赛详情。
- **寻求关于微调 LLM 生成求职信的指导**：一位用户正在开发一个项目，旨在利用简历文本和职位描述微调 LLM 以起草求职信。他们询问如何有效地使用测试数据来评估模型在此类文本生成任务中的表现。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/)** (1 messages): 

__dchx: <@610008277714993152> 你的问题有答案了吗？
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1255878416109142070)** (1 messages): 

- **用户等待额度分配**：一名成员就其账户详情和额度分配状态向另一名成员发送了私信。他们尚未收到额度。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1255750647853350954)** (1 messages): 

- **评估用于求职信生成的 LLM**：一位成员询问关于使用简历文本和职位描述来微调 **LLM 模型以编写求职信** 的事宜。他们询问如何利用测试数据来评估模型在此 **文本生成任务** 中的性能。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1255795071438295151)** (1 messages): 

- **使用 Flask 构建基于推文的机器人**：有人正在开发一个项目，创建一个可以使用其推文训练模型的端点。他们已经使用 Flask 和 Tweepy 集成了 Twitter API，并正在寻求关于如何引入模型、使用推文对其进行训练并使其能够 *“以我的风格”* 回答问题的建议。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1255876604278607882)** (1 messages): 

- **Ajay 请求协助**：一位用户联系了另一位成员 **@466291653154439169** 寻求帮助。未提供具体的背景或请求详情。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/)** (1 messages): 

lalithnarayan: 私信你了.. 请查看一下 🙏
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1255710861746638932)** (3 messages): 

- **学生纠结于 Copilot 还是 Cursor**：一名学生正在讨论是否要使用提供的 OpenAI 额度从 **Copilot** 切换到 **Cursor**。他们还好奇付费版的 **Cursor** 是否比这两个选项提供更好的功能。
- **在 Cursor 中混合搭配 Copilot**：有人建议使用提供的 OpenAI 额度通过免费版测试 **Cursor**，并可能在 **Cursor** 中安装 **Copilot**。分享了 [在 Cursor 中安装 VSCode 扩展的指南](https://www.cursor.com/how-to-install-extension) 以帮助完成安装过程。
- **职业频道变成了 Cursor 传教现场**：有人幽默地指出，职业频道已被倡导使用 **Cursor** 而非其他工具的讨论所占据。

**提到的链接**：<a href="https://www.cursor.com/how-to-install-extension">Extension | Cursor - AI 优先的代码编辑器</a>：未找到描述

  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1255604696694128782)** (4 messages): 

- **angry.penguin 挺身而出防止未来的垃圾信息**：一名成员自荐成为版主（mod），以帮助防止未来的垃圾信息事件。管理员表示感谢并迅速授予了版主权限。
- **垃圾信息清理完成**：在获得版主权限后，该成员报告称他们已经清理了现有的垃圾信息，确保今后有更整洁的聊天体验。
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1255946358796582983)** (2 messages): 

- **Gemma 2 支持即将到来**：一位用户询问了关于添加 **Gemma 2 支持** 的事宜。回复强调这已在计划中，但如果有人想立即尝试使用该模型，也欢迎社区贡献。
  

---



### **DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/)** (1 messages): 

le_mess: 干得好 🙂 你介意分享一下训练代码吗？
  

---



---



---



---



---



---



{% else %}


> 完整的逐频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整细分，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}