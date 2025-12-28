---
companies:
- deepseek-ai
- mistral-ai
- microsoft
- openai
- scale-ai
- tesla
- nvidia
- google-deepmind
date: '2024-05-06T23:37:03.494203Z'
description: '**DeepSeek V2** 推出了一款新型的顶尖混合专家（MoE）模型，拥有 **2360 亿参数**和一种新颖的多头潜在注意力（Multi-Head
  Latent Attention）机制，实现了更快的推理速度，并在 AlignBench 基准测试上超越了 GPT-4。**Llama 3 120B** 展示了强大的创意写作能力，而据报道，微软正在开发一个名为
  **MAI-1** 的 **5000 亿参数**大语言模型。Scale AI 的研究强调了 **Mistral** 和 **Phi** 等模型中存在的过拟合问题，而
  **GPT-4**、**Claude**、**Gemini** 和 **Llama** 则保持了基准测试的鲁棒性。在机器人领域，**特斯拉 Optimus**
  凭借卓越的数据采集和远程操控技术取得进展，**LeRobot** 标志着向开源机器人 AI 的迈进，而**英伟达的 DrEureka** 则实现了机器人技能训练的自动化。此外，一项调查研究了多模态大语言模型的幻觉问题并提出了新的缓解策略，谷歌的
  **Med-Gemini** 通过微调的多模态模型在医学基准测试中达到了业内领先（SOTA）的水平。'
id: 71868cce-79a2-4ff0-af3e-13d4e6017711
models:
- deepseek-v2
- llama-3-120b
- llama-3-400b
- gpt-4
- mistral
- phi
- claude
- gemini
- mai-1
- med-gemini
original_slug: ainews-deepseek-v2-beats-mixtral-8x22b
people:
- erhartford
- maximelabonne
- bindureddy
- adcock_brett
- drjimfan
- clementdelangue
- omarsar0
- rohanpaul_ai
title: DeepSeek-V2 性能超越 Mixtral 8x22B：拥有 160 多个专家，且成本仅需一半。
topics:
- mixture-of-experts
- multi-head-attention
- model-inference
- benchmarking
- overfitting
- robotics
- teleoperation
- open-source
- multimodality
- hallucination-detection
- fine-tuning
- medical-ai
- model-training
---

<!-- buttondown-editor-mode: plaintext -->> 2024年5月3日至5月6日的 AI 新闻。我们为您查阅了 7 个 subreddit、[**373** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 社区（**419** 个频道，共 **10335** 条消息）。为您节省了预计阅读时间（以 200wpm 计算）：**1112 分钟**。

**更多专家就是你所需要的一切？**

[DeepSeek V2](https://github.com/deepseek-ai/DeepSeek-V2) 突破了[上个月的 Mistral 凸包 (Convex Hull)](https://buttondown.email/ainews/archive/ainews-mixtral-8x22b-instruct-defines-frontier/)：

 
![image.png](https://assets.buttondown.email/images/bcf759e8-0ca7-4ccd-a901-6289aedd96ea.png?w=960&fit=max)
 

关于数据集的信息非常少；他们只提到它是 8B tokens（是 [DeepSeek v1](https://arxiv.org/abs/2401.02954) 的 4 倍），其中中文比例比英文高出约 12%。

[Snowflake Arctic](https://buttondown.email/ainews/archive/ainews-snowflake/) 是我们之前见过的最后一个拥有最高专家数量（128 个）的超大型 MoE 模型；DeepSeek v2 现在设定了新的标杆，不仅扩展了 DeepSeekMOE 已经取得的成功，还引入了一种名为 Multi-Head Latent Attention 的新注意力变体。

 
![image.png](https://assets.buttondown.email/images/16916531-1d7f-4068-a398-00d74c9e8fbc.png?w=960&fit=max)
 

通过缓存压缩后的 KV（“减少了 93.3% 的 KV cache”），这显著提升了推理速度。

 
![image.png](https://assets.buttondown.email/images/4b75f5cc-73a5-4525-ac1d-a394849e4cb4.png?w=960&fit=max)
 

论文详细介绍了他们发现有效的其他小技巧。

DeepSeek 正在用实际行动证明自己——他们在平台上提供的 [token 推理价格为每百万 token 0.28 美元](https://twitter.com/deepseek_ai/status/1787478994478321872)，大约是 [2023 年 12 月 Mixtral 价格战](https://twitter.com/swyx/status/1744467383090372743) 中最低价格的一半。

---

**目录**

[TOC] 

---

# AI Twitter 回顾

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中择优。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**LLM 进展与发布**

- **Llama 3 发布**：[@erhartford](https://twitter.com/erhartford/status/1787050962114207886) 指出 Llama 3 120B 比 Opus 更聪明，并对 llama3-400b 充满期待。[@maximelabonne](https://twitter.com/maximelabonne/status/1787401780021649911) 分享道，Llama 3 120B 在创意写作方面优于 GPT-4，但在推理方面逊于 L3 70B。
- **DeepSeek-V2 发布**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1787478986731429933) 推出了 DeepSeek-V2，这是一款开源 MoE 模型，在 AlignBench 中位列前三，超越了 GPT-4。它拥有 236B 参数，生成过程中仅激活 21B。
- **来自 Microsoft 的 MAI-1 500B**：[@bindureddy](https://twitter.com/bindureddy/status/1787498838024139185) 预测 Microsoft 正在训练自己的 500B 参数 LLM，名为 MAI-1，可能会在 Build 大会上预展。一旦发布，它将与 OpenAI 的 GPT 系列展开竞争。
- **Mistral 和开源 LLM 的基准测试过拟合问题**：[@adcock_brett](https://twitter.com/adcock_brett/status/1787151286305017966) 分享了 Scale AI 发布的研究，揭示了 Mistral 和 Phi 等某些 LLM 在流行 AI 基准测试中存在“过拟合”现象，而 GPT-4、Claude、Gemini 和 Llama 则表现稳健。

**机器人与具身智能 (Embodied AI)**

- **Tesla Optimus 更新**：[@DrJimFan](https://twitter.com/DrJimFan/status/1787154880110694614) 祝贺 Tesla Optimus 团队的更新，指出他们的人类数据采集场是 Optimus 最大的领先优势，拥有顶级的机械手、远程操作软件、庞大的车队以及精心设计的任务和环境。
- **LeRobot 开启开源机器人技术**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1787474711582003702) 欢迎由 @remicadene 及其团队开发的 LeRobot，这标志着机器人 AI 向开源方向的转变。
- **来自 Nvidia 的 DrEureka**：[@adcock_brett](https://twitter.com/adcock_brett/status/1787151046713786421) 分享了 Nvidia 的 “DrEureka”，这是一个 LLM Agent，可以自动编写代码来训练机器人技能，用于在模拟中训练机器狗的技能，并将其零样本（zero-shot）迁移到现实世界。

**多模态 AI 与幻觉**

- **多模态 LLM 幻觉综述**：[@omarsar0](https://twitter.com/omarsar0/status/1787510195922346154) 分享了一篇论文，对多模态 LLM 中的幻觉进行了综述，讨论了在检测、评估、缓解策略、原因、基准测试、指标和挑战方面的最新进展。
- **来自 Google 的 Med-Gemini**：[@adcock_brett](https://twitter.com/adcock_brett/status/1787151219149926801) 报道了 Google 推出的 Med-Gemini，这是一个针对医疗任务进行微调的 AI 模型系列，在文本、多模态和长上下文应用的 14 个基准测试中，有 10 个达到了 SOTA。

**新兴架构与训练技术**

- **Kolmogorov-Arnold Networks (KANs)**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1787258503897157735) 强调了一篇论文，提议将 KANs 作为 MLPs 的替代方案来逼近非线性函数，其表现优于 MLPs，且在不使用线性权重的情况下拥有更快的 neural scaling laws。
- **用于 Parameter-Efficient Finetuning 的 LoRA**：[@rasbt](https://twitter.com/rasbt/status/1787467605718008228) 从零开始实现了 LoRA，用于训练一个在 SPAM 分类中达到 98% 准确率的 GPT 模型，并指出 LoRA 是他最喜欢的 LLMs 参数高效微调技术。
- **带有 Expert Router 的混合 LLM 方法**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1787450115566747905) 分享了一篇关于高性价比混合 LLM 方法的论文，该方法使用 Expert Router 将“简单”查询引导至较小的模型，以在保持质量的同时降低成本。

**基准测试、框架和工具**

- **从 PyTorch Lightning 导出 TorchScript 模型**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1787461157395022020) 指出，使用 `to_torchscript()` 方法从 PyTorch Lightning 导出和编译模型到 TorchScript 非常顺畅，能够为非 Python 环境实现模型序列化。
- **带有 Whisper 和 Diarization 的 Hugging Face Inference Endpoints**：[@_philschmid](https://twitter.com/_philschmid/status/1787487522978717915) 为 Hugging Face Inference Endpoints 创建了一个优化的 Whisper（支持发言人日志），利用 flash attention、speculative decoding 和自定义处理器，在 1x A10G GPU 上实现了 60 秒音频仅需 4.15 秒的转录。
- **用于复杂 AI Agents 的 LangChain**：[@omarsar0](https://twitter.com/omarsar0/status/1787513175660806488) 分享了一个 2 小时的免费研讨会，内容是使用 LangChain 构建复杂的 AI Agents，用于自动化客户支持、营销、技术支持、销售和内容创作中的任务。

**趋势、观点和讨论**

- **LLMs 商品化**：[@bindureddy](https://twitter.com/bindureddy/status/1787507453023994251) 认为 LLMs 已经成为一种商品，即使 GPT-5 非常出色，其他主要参与者也会在几个月内赶上。推理价格将趋于下降，表现最好的 LLM 每隔几周就会更替。最佳策略是使用 LLM-agnostic 服务，并从基础模型转向构建 AI Agents。
- **读写能力与技术**：[@ylecun](https://twitter.com/ylecun/status/1787392175522672664) 分享了对不同时期人们对阅读和技术态度转变的观察，从 1900 年的“你为什么不去耕田而是读书？”到 2020 年的“你为什么不去玩平板而是看电视？”。
- **基础研究资助**：[@ylecun](https://twitter.com/ylecun/status/1787041840484557203) 认为，几乎所有拨给大学的联邦资金都流向了 STEM 和生物医学研究，社会科学得到的很少，人文科学几乎为零。削减这些资金将“杀掉下金蛋的鹅”，并可能导致生命损失。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 发展与能力**

- **Tesla Optimus 进展**：在 /r/singularity 中，一段新视频展示了 Tesla Optimus 机器人的最新能力，包括[**手部精细触觉和力觉感知**](https://v.redd.it/6ll8ixioekyc1)。讨论围绕机器人的当前速度限制，以及一旦达到人类工人的“20倍速率”，在工厂进行 24/7 全天候运行的潜力。
- **Sora AI 视频渲染**：在 /r/singularity 中，AI 系统 Sora 展示了[**在更改单个元素的同时渲染视频**](https://v.redd.it/6h46hqbtenyc1)的能力，尽管该功能仍处于研究阶段，尚未公开发布。
- **GPT-4 训练的机器人狗**：在 /r/singularity 中，一只使用 GPT-4 训练的机器人狗[**展示了其在滚动且放气的瑜伽球上保持平衡的能力**](https://youtu.be/d5mdW1yPXIg)，体现了 AI 驱动的机器人技术和平衡控制方面的进步。
- **算力与 AI 里程碑**：在 /r/singularity 中，Microsoft CTO Kevin Scott 认为 [**AI 里程碑成就的共同因素是使用了更多算力**](https://www.reddit.com/r/singularity/comments/1cklexf/compute_is_all_you_need_microsoft_cto_kevin_scott/)。讨论还涉及 Llama 3 400b 的潜力，由于其在 25,000 块 H100 上训练，而据报道 GPT-4 仅使用了 10,000 块 A100，因此其表现可能超越 GPT-4。
- **LLaMA 70B 性能**：在 /r/singularity 中，一位用户报告在配备 4 年前 3090 GPU 的 7 年老旧 PC 上运行 Llama 3 70B，[**在某些情况下获得了比 GPT-4 和 Claude 3 更好的响应**](https://www.reddit.com/r/singularity/comments/1ckq5k8/llama_70b_q5_works_on_24gb_graphics_cards_and_the/)。该帖子强调了拥有一个无需互联网连接且能提供高质量输出的高智能 AI 的意义。

**社会影响与担忧**

- **公众对 AI 生成图像的认知**：在 /r/singularity 中，一项调查显示[**一半的美国人不知道 AI 可以生成逼真的人物图像**](https://i.redd.it/bn5wjsqabkyc1.png)，这引发了关于人们在不知情的情况下接触了多少 AI 生成图像的问题。评论讨论了美国公众普遍缺乏相关知识的现状。
- **AI 与全民富足**：在 /r/singularity 中，一则帖子质疑了[**AI 将导致全民富足的信念，认为 AI 可能会巩固现有的权力结构并加剧不平等**](https://www.reddit.com/r/singularity/comments/1cl3bgq/why_do_people_here_think_ai_will_lead_to/)。作者认为这种转变将是渐进的，失业率和生活成本会随着时间的推移缓慢增加，直到灾难发生。
- **沃伦·巴菲特对 AI 的担忧**：在 /r/StableDiffusion 中，沃伦·巴菲特[**将 AI 比作原子弹，强调了其在诈骗方面的潜力，并对 AI 的力量表示担忧**](https://www.reddit.com/r/StableDiffusion/comments/1cl69r7/warren_buffett_compares_ai_to_the_atomic_bomb/)。评论讨论了 AI 的双重性质，将其与电力的出现进行类比，既有积极影响也有消极影响。

**AI 应用与发展**

- **AI 在医疗笔记记录中的应用**：在 /r/singularity 中，安大略省的一位家庭医生报告称，[**AI 驱动的笔记记录显著改善了她的工作并保住了她的职位**](https://globalnews.ca/news/10463535/ontario-family-doctor-artificial-intelligence-notes/)，突显了 AI 辅助医疗文档记录的潜力。
- **Optimus 手部进展**：在 /r/singularity 中，Elon Musk 宣布[**将于今年晚些时候发布的新版 Optimus 手部将拥有 22 个自由度 (DoF)，比之前的 11 个 DoF 有所增加**](https://x.com/elonmusk/status/1787157110804910168)。
- **AI 训练与推理的未来**：在 /r/singularity 中，Nvidia CEO 预测未来[**AI 训练和推理将成为一个单一过程，使 AI 能够在与用户互动的同时进行学习**](https://www.youtube.com/watch?v=oNwoA5akBlg)。该视频被推荐为关注 AI 发展人士的有趣内容。

**迷因与幽默**

- **AI 训练与奇怪的结果**：在 /r/StableDiffusion 中，一张迷因图片暗示[**在不寻常或非传统的图像上训练 AI 会导致奇怪的结果**](https://i.redd.it/1a7wf3um6myc1.jpeg)。
- **精致的香蕉**：在 /r/StableDiffusion 中，一个[**幽默的图片帖子展示了一根精致的香蕉**](https://i.redd.it/f31y59ljwlyc1.jpeg)。
- **安全团队建议**：在 /r/StableDiffusion 中，一段视频迷因描绘了一个恶作剧，[**一个人的椅子被从身后抽走，导致其摔倒**](https://v.redd.it/urf2bp29jqyc1)。评论讨论了该恶作剧的危险性以及造成严重伤害的可能性。

---

# AI Discord 摘要

> 摘要之摘要的摘要

- **Llama3 GGUF 转换挑战**：用户在使用 llama.cpp 将 **Llama3** 模型转换为 GGUF 格式时遇到问题，训练数据丢失与精度无关。换行符的 Regex 不匹配被认为是潜在原因，影响了 [ollama 和 lm studio](https://github.com/ggerganov/llama.cpp/issues/7062) 等平台。社区成员正在协作进行 [regex 修改](https://github.com/ggerganov/llama.cpp/pull/6965)等修复工作。

- **GPT-4 Turbo 性能担忧**：OpenAI 用户报告了 GPT-4 Turbo 显著的**延迟增加**以及对**消息上限阈值**的困惑，部分用户经历了 [5-10 倍的响应速度变慢](https://discord.com/channels/974519864045756446/1001151820170801244/1235851459891957821)，且上限在 25-50 条消息之间。相关理论认为这是高峰时段的动态调整。

- **Stable Diffusion 安装困扰**：Stability.ai 社区成员在 **Stable Diffusion 设置无法访问 GPU 资源**方面寻求帮助，遇到了如 ["RuntimeError: Torch is not able to use GPU"](https://discord.com/channels/1002292111942635562/1002292112739549196/1235849532609265724) 的错误。讨论还涉及缺乏全面且最新的 **LoRA/DreamBooth/微调教程**。

- **Hermes 2 Pro Llama 3 的上下文表现令人印象深刻**：**Hermes 2 Pro Llama 3** 在使用 **vLLM** 和 RoPE 缩放的 32GB Nvidia v100 Tesla 上展示了约 32k 的上下文，具有[完美的 16k token 召回且无衰减](https://discord.com/channels/1053877538025386074/1149866623109439599/1235849078479519855)。通过编辑 `config.json` 和 RoPE 缩放因子可以实现扩展上下文。

- **Perplexity AI 的 Pages 功能引起关注**：Perplexity AI 用于创建综合报告的新 **Pages 功能**引发了热议，同时用户对 [Claude 3 Opus 每天 50 条消息的限制表示沮丧](https://discord.com/channels/1047197230748151888/1047649527299055688/1235849900500058132)，相比之下 GPT-4 Turbo 和 Sonnet 的限制较少。讨论还涉及 Perplexity 从无限消息向有限消息的转变。

- **LM Studio 启用 Headless 模式**：LM Studio 用户利用 `lms` CLI 工具在 GUI 之外进行 **headless 操作**，排查内存异常并[为平滑的服务器端部署制定策略](https://discord.com/channels/1110598183144399058/1110598183144399061/1235890438762791002)，从而避免通过 RDP 消耗 VRAM。微调瓶颈也被讨论，一名成员报告在 128GB M3 Max MacBook Pro 上成功进行了 8 小时的微调。

- **CUDA 编译与多 GPU 训练挑战**：CUDA 开发者遇到 `nvcc 11.5` 在**旧款 GPU 上进行 bfloat16 操作**报错的问题，并[提出了一个修复方案](https://github.com/karpathy/llm.c/pull/353)以手动处理算术运算实现向后兼容。正如 [Issue #369](https://github.com/karpathy/llm.c/issues/369) 所述，最近的提交还导致了**多 GPU 训练挂起**，目前有一个独立分支维持功能正常。

- **Mojo 编译器与类型系统演进**：Mojo 的 nightly 编译器更新带来了符合当前实践的变化，不再采用 **80 列宽**限制，并向**可寄存器传递类型（register passable types）**转型。讨论涉及逐步淘汰 `OptionalReg`，转而支持指示寄存器传递能力的 traits，详见 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。

- **HuggingFace 社区亮点**：HuggingFace 社区中值得关注的项目包括 [Moondream 2 批处理](https://huggingface.co/spaces/Csplk/moondream2-batch-processing)、[FLUENT 的最新迭代](https://huggingface.co/spaces/fluently/Fluently-Playground)、[HF 音频课程章节的葡萄牙语翻译](https://iatalk.ing/destaques-comunidade-hugging-face/)，以及针对长标题的 [BLIP 微调](https://huggingface.co/spaces/unography/image-captioning-with-longcap)。完整列表可在[社区亮点](https://iatalk.ing/destaques-comunidade-hugging-face/)中查看。

- **Eleuther 探讨 Transformer 的国际象棋实力**：一篇 [arXiv 论文](https://arxiv.org/abs/2402.04494)展示了一个 270M 参数的 Transformer 模型在没有特定领域算法的情况下，在国际象棋中超越了 AlphaZero 的策略和价值网络，这引发了 Eleuther 社区关于规模对策略游戏影响的讨论。

---

# 第 1 部分：高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Llama3 的 GGUF 转换小故障**：Unsloth 社区在使用 llama.cpp 处理 Llama3 模型时遇到了**转换问题**，特别是在转换为 GGUF 格式时影响了训练数据。问题不仅限于 FP16 转换，这暗示了除了精度损失之外，还存在更深层的底层问题。

**换行符引发的大问题**：故障中的一个反复出现的主题与换行符的 Tokenization 有关，不同正则表达式库（regex libraries）之间的行为差异导致了不稳定的 tokenizer.json 模式。社区正在探索涉及正则表达式修改的潜在解决方案，以修复 GGUF 转换挑战。

**Llama 变体挑战基因组数据**：M.chimiste 推出的 **[LLaMA-3-8B-RDF-Experiment](https://huggingface.co/M-Chimiste/Llama-3-8B-RDF-Experiment)** 模型标志着将 LLM 与基因组数据和知识图谱构建相结合的尝试。

**对视觉语言模型微调工具的需求**：社区提出了对视觉语言模型（LVLM）通用微调方法的需求，一位成员表达了对支持 **Moondream** 的兴趣，详见其 [GitHub notebook](https://github.com/vikhyat/moondream/blob/main/notebooks/Finetuning.ipynb)。

**展示与分享平台的增长**：关于设立专门的 LLM 部署讨论频道的提案突显了对共享学习的需求。这与 Oncord 集成 Unsloth AI 用于 Web 开发 AI 工具的展示，以及增强 Llama-3 能力的模型发布相契合。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Perplexity AI 凭借 Pages 功能领先**：Perplexity AI 新推出的 **Pages 功能**因其创建综合报告的能力而备受关注。与此同时，随着工程师们讨论投资回报递减的问题，对于 **GPT-5** 潜力的怀疑态度也随之产生。

**AGI 概念引发辩论**：Discord 上的 AI 社区陷入了关于 AGI 定义以及像 **ChatGPT** 这样的 AI 模型是否是 AGI 先驱版本的辩论。对 AI 生成音乐的兴趣表明，人们对创意 AI 应用的需求日益增长，并提到了 **Udio** 等服务。

**GPT-4 Turbo 遭遇性能瓶颈**：据报告，**GPT-4 Turbo** 的响应**延迟**显著增加，用户正在寻求关于不一致的**消息上限阈值**的解释，这表明在高峰时段可能存在动态调整。

**Prompt Engineering 的挑战与策略**：工程师们分享了经验和资源，推荐了 **Teddy Dicus Murphy 的 "Wordplay"** 以获取 Prompt 创作见解，并深入探讨了在 OpenAI API 中使用 **logit bias** 来操纵 Token 概率的复杂性。

**为查询微调 AI**：一场热烈的讨论围绕着微调模型以**生成问题**而非答案展开，包括改进用于提取产品信息的 **GPT-4-TURBO** Prompt 的策略，并辅以 [logit bias 教程](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api)的支持。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **GPU 问题成为焦点**：成员们报告了 **Stable Diffusion** 安装程序无法访问 GPU 资源的问题，具体表现为 "RuntimeError: Torch is not able to use GPU" 等错误。

- **Stable Diffusion 3 传闻引发热议**：关于 **Stable Diffusion 3** 发布的预期不断升温，引发了对其潜在延迟影响的辩论，而怀疑者则对其是否真的会发布表示质疑。

- **微调教程的缺失**：社区对缺乏关于 **LoRA/DreamBooth/fine-tuning** 等技术的最新、全面教程感到沮丧，许多人发现现有教程要么过时，要么细节匮乏。

- **追求独特的面孔**：一位成员询问了训练 AI 生成独特、真实面孔的策略，纠结于是对多张面孔使用 **LoRA**，还是在生成的随机面孔基础上进行训练。

- **开源障碍讨论**：对话转向了 **Stable Diffusion** 开源承诺的真实性，人们担心未来高质量模型、Checkpoints 和训练细节可能会被设置门槛。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **SVM 在 AI 圈内依然活跃**：Discord 成员在技术闲聊中澄清，**SVM** 代表 **Support Vector Machine**（支持向量机）。
- **对 Meta-Llama-3-120B-Instruct 的期待**：Hugging Face 上的 **Meta-Llama-3-120B-Instruct** 模型引发了对其潜力的讨论，用户呼吁进行全面的 Benchmarking（基准测试），而非仅仅依赖炒作。
- **部署困境**：用户讨论了 Serverless **Llama** 的局限性，同时探讨了具备充足 VRAM 的更好 GPU 选项，例如 **Azure 的 NC80adis_H100_v5**，以处理大上下文任务需求。
- **Hermes 2 令人印象深刻的性能表现**：**Hermes 2 Pro Llama 8B** 展示了惊人的约 32k 扩展上下文容量且无明显衰减，在 32GB 的 Nvidia v100 Tesla 上实现了 16k 的完美 Recall（召回）。
- **Cynde 助力数据耕作**：分享了关于 **Cynde** 的更新，标志着其核心实现的完成。社区对这个用于 Intelligence Farming（智能耕作）的框架表现出极高热情，**[Cynde 的 GitHub 仓库](https://github.com/Neural-Dragon-AI/Cynde)** 欢迎贡献者加入。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Pages Beta 不再开放申请**：由于参与人数已满，**Pages** 的 Beta 测试申请阶段已经结束。关于 **Pages** 的后续更新将另行通知。

- **关于 Perplexity AI 性能与限制的热烈讨论**：成员们遇到了 Claude 3 模型**响应速度慢**的问题，并对 Claude 3 Opus 模型**每天 50 条消息的限制表示沮丧**。在将 Opus 与 **GPT-4 Turbo** 及 **Sonnet** 进行对比时，用户还对 Perplexity 从无限量转向**受限的消息能力**表示担忧。

- **探索 AI 的创意与新颖用途**：**Perplexity AI 社区**正积极探索该平台在**图像生成**、**模仿小说写作风格**以及**多样化搜索**方面的能力，例如挖掘 BASIC 编程语言的历史或深入研究 Perplexity 自身的发展史。

- **API 探索与灵活调整**：用户讨论了模型迁移，特别是从 **sonar-medium-online 切换到 llama-3-sonar-large-32k-online**，并**询问了潜在的计费不一致问题**。对话还涉及了 **AI 结果优化**的成功与困扰，以及使用 Perplexity API 创建**极简代码 Telegram 机器人**的建议。

- **多渠道搜索查询分享**：社区分享了多个**搜索查询及其结果**，引发了关于 **Perplexity 的有效使用及其提供见解深度**的讨论。这些探索涵盖了从**编程历史到专有技术见解**的各种背景。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **无头模式（Headless）进展**：工程师们正在利用 LM Studio 的 CLI 工具 `lms` 进行**无头操作**，并与 GUI 版本配合使用。他们正在解决内存消耗异常的问题，并讨论在不通过 RDP 消耗 VRAM 的情况下实现平滑服务器端部署的策略。

- **微调技巧与模型故障**：成员们正在排查微调瓶颈，分享在 128GB M3 Max MacBook Pro 等硬件上进行长时间微调的**成功案例**，并讨论困扰 Llama 3 等模型的**输出不一致**问题。

- **交互意图与 AI 记忆怪癖**：用户表达了一个令人困惑的观察结果，即语言模型可能会保留已删除 Prompt 元素的上下文，这暗示了潜在的 **Bug 或对模型行为的误解**。他们探索了**个性化写作风格**的交互技术，并为 LLM 实现了对文档部分的“作用域访问（scoped access）”。

- **角色扮演无限制？没那么快**：围绕 AI 与 **RPG** 结合的讨论非常热烈，用户目标是将 AI 训练为 D&D 的地下城主（Dungeon Masters），并指出现有系统受限于内容审核，这可能会影响故事的黑暗程度和深度。

- **ROCm 赞誉与 Linux 热情**：ROCm 的更新表现稳健，但讨论也涉及了**模型转换**和为 Embeddings 发送更长序列的挑战。对话转向社区对贡献 **Linux ROCm 构建版本**的兴趣，暗示如果项目寻求更多开源协作，将会有进一步的参与。

- **硬件前沿的 AI**：成员们投入到激烈的硬件交流中，对比了 Tesla P40 等**旧款 GPU** 相对于 GRID K1 的**适用性**，并痴迷于以 AI 为中心的家庭实验室中多 GPU 设置的细节。**具体细节**涵盖了从服务器硬件采购到散热、电源和驱动兼容性等问题。

- **LM Studio 最新阵容**：`lmstudio-community` 仓库已**更新了 CodeGemma 1.1** 和 **Nvidia 的 ChatQA 1.5 模型**，前者引起了热切期待，而后者提供了专为基于上下文的问答（Q/A）应用量身定制的专业模型。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**BackPACK 表现强劲**：[BackPACK](https://backpack.pt/) 是一个用于从反向传播中提取额外信息的 PyTorch 扩展工具，讨论强调了它对 PyTorch 开发者的潜力。详情见 Dangel 等人 2020 年发表的论文 "BackPACK: Packing more into Backprop"。

**DoRA 在融合方面表现出色**：一个新的 **fused DoRA layer implementation** 减少了单个 kernel 的数量，并针对 GEMM 和 reduction 操作进行了优化，详见 [GitHub pull request](https://github.com/pytorch/ao/pull/216)。社区对即将发布的针对这些增强功能的 benchmark 表示期待。

**自定义 CUDA 扩展的定制化**：成员们讨论了安装自定义 PyTorch/CUDA 扩展的最佳实践，分享了多个 GitHub pull requests（如 [PR#135](https://github.com/pytorch/ao/pull/135)）和一个示例 `setup.py` 以供参考，旨在简化安装过程。

**利用 CUTLASS 稳步前行**：围绕 CUTLASS 中使用的 stream-K 调度技术引起了广泛关注，并建议在未来的演讲中深入探讨其工作原理。

**GPU 通信课程**：宣布了即将举行的关于使用 **NCCL** 进行 GPU 集体通信（Collective Communications）的会议，重点关注分布式 ML 概念。

**必读的 ML 系统论文**：对于机器学习系统的新手，GitHub 上的 [ML Systems Onboarding list](https://github.com/cuda-mode/awesomeMLSys) 提供了一系列精选的参考论文。

**克服 CUDA 编译难题**：针对 `nvcc 11.5` 在 bfloat16 操作中报错的问题，已在 [fix proposal](https://github.com/karpathy/llm.c/pull/353) 中提出解决方案，旨在支持旧版 GPU 和工具包。还讨论了多 GPU 训练挂起的问题，涉及 [Issue #369](https://github.com/karpathy/llm.c/issues/369)，并有一个独立分支维持功能。

**LLaMa 的精简学习**：关于 **LLaMa 2 70B 模型训练** 期间内存效率的讨论强调了可以减少内存使用的配置。提到了一个名为 **HTA** 的工具，用于定位 PyTorch 中的性能瓶颈。

**量化带来的后训练巅峰**：分享了一个 [YouTube 视频](https://youtu.be/0VdNflU08yA?feature=shared)，详细介绍了 PyTorch 中量化的过程和优势。

**GreenBitAI 走向全球**：介绍了一个名为 [green-bit-llm](https://github.com/GreenBitAI/green-bit-llm) 的工具包，用于微调和推理 GreenBitAI 的语言模型。BitBlas 因其快速的 2-bit 操作 gemv kernel 而受到关注，同时 GreenBitAI 的工具包中还包含一种独特的梯度计算方法。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**收看 Mojo 直播，获取 MAX 24.3 更新**：Modular 的新直播视频 "[Modular Community Livestream - New in MAX 24.3](https://www.youtube.com/watch?v=kKOCuLy-0UY)" 邀请社区探索 MAX Engine 和 Mojo 的最新功能，并介绍了 MAX Engine Extensibility API。

**社区项目飞速发展**：值得关注的更新包括 NuMojo 性能的提升以及用于图像解析的 [Mimage](https://github.com/fnands/mimage) 的推出。Basalt 项目也达到了 200 stars 的里程碑，并发布了新的 [文档](https://basalt-docs.vercel.app/)。

**Mojo 编译器不断演进**：Mojo 编译器迎来了 [nightly updates](https://github.com/modularml/mojo/pull/2498/files)，其更改更符合当前实践，例如放弃了 80 列宽度限制，并向更适合寄存器传递（register passability）的类型过渡。

**AI 工程师探索 Don Hoffman 的意识研究**：对加州大学欧文分校（UCI）Donald Hoffman 关于意识研究工作的兴趣与 AI 产生了关联，人们在裂脑患者的感官数据限制与 AI 幻觉（hallucinations）之间找到了相似之处。

**Mojo 生态系统的成长与开发者指导**：讨论了 Mojo 的贡献流程，符合 [GitHub 的 pull request 指南](https://github.com/modularml/mojo/pull/2457)，并通过 [参数教程](https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md) 深入了解开发工作流，展示了对快速扩张的 Mojo 生态系统贡献者的积极支持。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Moondream 和 BLOOM 掀起波澜**：**HuggingFace** 社区聚焦了多项新进展，包括 **[Moondream 2 批处理](https://huggingface.co/spaces/Csplk/moondream2-batch-processing)** 和 **FLUENT 的最新迭代**，以及多语言支持工具。特别值得关注的是 **BLOOM 多语言聊天** 和 **AutoTrain 对 YAML 配置的支持**，这简化了机器学习初学者的训练流程。查看 [社区亮点](https://iatalk.ing/destaques-comunidade-hugging-face/)。

**当音频模型开始歌唱**：人们对用于生成式音乐的音频扩散模型产生了浓厚兴趣，**Whisper** 正在针对菲律宾语 ASR 进行微调，并引发了关于优化的讨论。然而，一位用户在将 PyTorch 模型转换为 TensorFlow Lite 时因尺寸限制遇到了挑战。

**AI 的前线**：网络安全成为焦点，**Hugging Face Twitter** 账号被盗，强调了对稳健的 AI 相关安全性的需求。成员们还交流了 GPU 利用率技巧，以应对不同配置间训练时间的差异。

**量子与 AI 结合的愿景**：在 **computer vision** 领域，重点在于改进传统方法，如使用 YOLO 进行汽车零部件的缝隙检测，以及调整 CLIP 等模型以识别旋转物体。**GhostNet 的预训练权重**备受追捧，CV 成员们也在思考 SURF 和 SIFT 等方法在当代的适用性。

**图论专家齐聚**：近期关于将 **LLM** 与图机器学习结合的论文提出了新颖的整合方式，其中一篇 **[论文](https://arxiv.org/abs/2404.19705)** 专门教学 LLM 仅在需要时通过 `<RET>` 标记检索信息。**[阅读小组](https://discord.com/events/879548962464493619/1234913780048203856)** 为渴望学习更多知识的人提供了额外资源。

**展示合成与应用 AI**：在 **[#i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1235893212393111594)** 板块，发布了 **Podcastify** 和 **OpenGPTs-platform** 等工具，以及使用 **mergekit** 构建的 **shadow-clown-BioMistral-7B-DARE** 等模型。

**NLP 者的疑惑与查询**：在 **[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1235892774528880641)** 板块，一位用户为 **Mistral-7B-instruct** 的定制训练提供报酬，同时也有人对 LLM 评估其他 LLM 表示担忧。介绍了使用 GPT 3.5+ 衡量翻译质量的 **GEMBA** 指标，并提供了[了解更多](https://aclanthology.org/2023.eamt-1.19/)的链接。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**将 OpenInterpreter 与 Groq LLM 集成**：工程师们讨论了将 **Groq LLM** 集成到 Open Interpreter 上的挑战，强调了输出不可控和错误文件创建等问题。分享的连接命令为 `interpreter --api_base "https://api.groq.com/openai/v1" --api_key "YOUR_API_KEY_HERE" --model "llama3-70b-8192" -y --max_tokens 8192`。

**微软黑客松寻求 Open Interpreter 爱好者**：一个团队正在组建以参加使用 Open Interpreter 的微软开源 AI 黑客松（Microsoft Open Source AI Hackathon）；该活动承诺提供实操教程，报名详情请见[此处](https://lu.ma/iu1wijgd)。

**Open Interpreter 的 iOS 重构**：讨论围绕在 Open Interpreter 上为 iOS 重新实现 TMC 协议，以及解决 Azure Open AI 模型的设置问题，一位成员分享了正在开发的 iOS 应用的 GitHub 仓库链接，见[此处](https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile)。

**本地 LLM 挑战开发者**：分享了对 **Phi-3-mini-128k-instruct** 等本地 LLM 的个人测试，结果显示出显著的性能差异，并呼吁在未来的实现中采用更好的优化方法。

**AI Vtuber 的 STT 难题**：为 AI 驱动的虚拟主播实现 Speech-to-Text 带来了实际挑战，工程师们考虑使用触发词，并致力于通过独立的 LLM 实例实现 AI 驱动的 Twitch 聊天互动，旨在获得全面的回复。对于正在处理类似集成的开发者，一位成员指出了其 GitHub 上的 **main.py** 文件作为参考资源。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **国际象棋特级大师请注意，Transformers 来了**：一项[新研究](https://arxiv.org/abs/2402.04494)揭示了一个拥有 270M 参数的 Transformer 模型在没有领域特定算法的情况下，在国际象棋领域超越了 AlphaZero 的策略和价值网络，引发了关于 Scale 在策略游戏中有效性的讨论。

- **LLM 研究在多语言和 Prompting 技术方面蓬勃发展**：研究亮点包括一项关于 LLM 处理多语言输入的研究，以及尽管对其实用性存在质疑，但“Maieutic Prompting”在处理不一致数据方面的潜力。该领域的贡献提供了见解和论文链接，例如 [How Mixture Models Handle Multilingualism](https://arxiv.org/abs/2402.18815v1) 以及对抗 LLM 漏洞的方法，包括 *The Instruction Hierarchy* [论文](http://arxiv.org/abs/2404.13208)。

- **显微镜下的模型性能**：迁移学习的 Scaling Laws 表明，预训练模型通过有效的迁移数据在固定大小的数据集上有所提升，这与社区在确定 LLM In-context Learning 的准确衡量标准和性能评估方法方面的努力相呼应。

- **解释 Transformers 并提高可部署性**：分享了关于解释基于 Transformer 的 LLM 的入门指南和综述，以及关于跨模型泛化的讨论。社区对解决 **Phi-2** 和 **Mistral-7B** 等模型中的 Weight Tying 问题表现出浓厚兴趣，并澄清了关于知名开源模型中 Weight Tying 的误解。

- **社区参与 ICLR 和求职**：尽管面临旅行挑战，ICLR 线下聚会的准备工作正在展开；社区支持显而易见，成员们分享了就业资源以及参与 **OSLO** 和 **Polyglot** 团队等项目的经验。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Llama 家族新成员**：[Llama 3 Lumimaid 8B](https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b) 模型已发布，同时还提供 [扩展版](https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b:extended)，而 [Llama 3 8B Instruct Extended](https://openrouter.ai/models/meta-llama/llama-3-8b-instruct:extended) 则迎来了降价。由于服务器更新，Lynn 系列模型宣布了短暂的停机。

- **高端 AI 招募 Beta 测试员**：Rubik's AI Pro 是一款先进的研究助手和搜索引擎，正在招募 Beta 测试员，提供 2 个月的尊享访问权限，包括 GPT-4 Turbo 和 Mistral Large 等模型。该项目可通过 [此处](signup.php) 使用促销代码 `RUBIX` 访问。

- **混合搭配模型**：社区成员报告称 **Gemini Pro** 现在已无错误，并讨论了 **Lumimaid 70B** 的潜在托管方。**Phi-3** 等模型备受期待，但供应稀缺。不同供应商的模型精度各异，大多数使用 **fp16**，部分使用量化的 **int8**。

- **模型合并**：对话重点介绍了 Hugging Face 上新创建的 **Meta-Llama 3 70B 自合并版本**，引发了关于自合并（Self-merges）与传统层映射合并（Layer-mapped merges）效果的辩论。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**提升 Agent 智能**：**LlamaIndex 0.10.34** 引入了 **introspective agents**（内省型智能体），能够通过反思机制实现自我改进。详情见 [notebook](https://t.co/X8tJGXkcPM)，该内容包含敏感材料警告。

**Agentic RAG 升级**：一段教学视频展示了如何集成 LlamaParse + Firecrawl 来构建 **agentic RAG 系统**，发布详情可通过 [此链接](https://t.co/wR35iYIKjo) 查看。

**信任评分的 RAG 响应**：@CleanlabAI 的 "Trustworthy Language Model" 引入了一套针对 RAG 响应可信度的评分系统，旨在确保生成内容的准确性。更多见解请参考其公告 [此处](https://t.co/KW1XsllRqQ)。

**本地 RAG 流水线手册上架**：为寻求摆脱云服务的开发者，一份使用 LlamaIndex 搭建全本地 RAG 流水线的手册现已发布，承诺比快速入门指南更深入，可在此处访问 [此处](https://t.co/2RCvaxOzKo)。

**Hugging Face 与 LlamaIndex 深度集成**：LlamaIndex 宣布支持 **Hugging Face TGI**，从而在 Huggingface 上实现语言模型的最优部署，并增强了 **function calling** 和延迟优化等功能。点击 [此处](https://t.co/3vGpxcbP18) 了解 TGI 的新功能。

**创建对话式 SQL Agents**：AI 工程师正在考虑使用 **HyDE** 为拥有大量表格的数据库构建 **NL-SQL bots**，旨在提高 LLM 生成 SQL 查询的精确度；同时，introspective agent 方法论也引起了关注，更多阅读见 [Introspective Agents with LlamaIndex](https://medium.com/ai-artistry/introspective-agents-with-llamaindex-777d018f791d)。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Hermes 2 Pro Llama 3 速度测试结果**：**Hermes 2 Pro Llama 3** 在配备 8GB RAM 的 Android 设备上展示了令人印象深刻的 **inference speed**（推理速度），这得益于 **llama.cpp** 的增强。

**动漫在 AI 对话中的角色**：成员们幽默地讨论了 **anime**（动漫）的兴起与 **AI question-answering**（问答）及 **image generation**（图像生成）任务能力提升之间的关系。

**Gradio 定制化成果**：**Gradio** 的调整现在允许通过 [YAML file](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1591) 进行动态配置，从而能够以编程方式设置隐私级别和服务器参数。

**AI 训练数据集备受关注**：讨论了一个包含 143,327 个经过验证的 Python 示例的新数据集（[Python Dataset](https://huggingface.co/datasets/Vezora/Tested-143k-Python-Alpaca)），以及即使使用以数学为中心的数据集，提升 Llama3 数学性能仍面临困难，凸显了 AI 训练中数据集的挑战。

**AI 训练平台的增强与需求**：有人呼吁完善 **Axolotl 的文档**，特别是关于合并模型权重和模型推理的部分，可在 [Axolotl Community Docs](https://axolotl.continuumlabs.pro/) 访问。此外，还解决了 gradient clipping（梯度裁剪）配置的问题，Phorm 提供了关于为 **gradient clipping** 和 **chatbot prompt** 定制 **TrainingArguments** 的见解。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gary 玩转 Ableton**：一个新的开发中 Python 项目 [gary4live](https://github.com/betweentwomidnights/gary4live) 将 Python continuations 与 Ableton 集成用于现场音乐表演，邀请社区贡献和同行评审。

- **Suno 扩大音乐制作规模**：关于使用 **Suno** 进行音乐生成的讨论包括与 *Musicgen* 等其他设置的比较，重点是 Suno 的音频 tokenization 过程，并探索这些模型是否能自动生成乐谱。

- **Token 探讨**：深入探讨音乐模型的 token 结构，参与者研究了音频合成中的 token 长度和组成，引用了学术论文中的特定架构设计但未展开细节。

- **打破音频合成的障碍**：讨论了将音频直接集成到多模态模型中的潜力，重点是音频通道的实时替换以及直接音频对实现全模态（omnimodal）功能的重要性。

- **Stable Audio 的商业节奏**：出现了关于稳定音频模型输出的商业用途和许可问题，特别关注其在现场表演中的实时应用以及对行业的潜在影响。



---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **本地硬件攻克 AI 任务**：用户现在可以使用 **llama-farm** 在旧笔记本电脑上本地运行 **Ollama**，以处理 LLM 任务，而无需将其暴露在公共互联网中。这还关联到了一个 GitHub 仓库，其中包含更多实现细节（[GitHub 上的 llama-farm chat](https://github.com/get-convex/llama-farm-chat)）。

- **实现 AI 云端独立**：讨论表明，使用 **Faraday** 允许用户永久保留下载的角色和模型，并且在拥有 6 GB VRAM 配置的情况下，本地运行工具可以规避云端订阅费用。本地执行无需订阅，是工具使用方面一个极具性价比的选择。

- **Ubuntu 用户重获控制权**：通过降级到 Node 版本 18.17.0 并根据 [GitHub issue](https://github.com/get-convex/convex-backend/issues/1) 更新 Ubuntu，解决了 Ubuntu 18 上 `convex-local-backend` 的安装问题。提出了 Dockerization（Docker 化）作为简化未来配置的潜在解决方案。

- **模拟现实备受关注**：旧金山的 Mission Control 举办了一场 **AI Simulated Party**（AI 模拟派对），融合了真实与数字体验。此外，**AI-Westworld** 模拟进入了公开测试阶段，并推出了一个名为 **AI Town Player** 的 Web 应用，用于通过导入 sqlite 文件回放 AI Town 场景。

- **剪贴板与节拍的融合**：有人呼吁合作创建一个涉及嘻哈歌手 **Kendrick** 和 **Drake** 的模拟。这展示了将 AI 开发与文化评论相结合的兴趣。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**CLIP vs. T5：模型大比拼**：关于集成 [CLIP 和 T5 编码器](https://old.reddit.com/r/StableDiffusion/comments/1cgr74j/april_30th/l2bxv66/) 进行 AI 模型训练的讨论非常热烈；虽然同时使用两种编码器显示出前景，但一些人主张仅使用 T5，因为 CLIP 存在提示词遵循（prompt adherence）问题。

**小模型是大趋势吗？**：在模型尺寸领域，小模型的增强正被优先考虑，400M 的 DeepFloyd 备受关注就是证明，技术对话涉及到了扩展至 8B 模型的挑战。

**发布 SD3：吊胃口还是全量发布？**：社区对 Stability AI 暗示的逐步推出 SD3 模型（从小型到大型）的反应褒贬不一，既有怀疑也有渴望，大家都在思考这种发布策略是否符合社区的预期。

**LLama Embeds 走入聚光灯下**：关于在模型训练中使用 LLama embeds 效果的辩论浮出水面，一些成员主张使用它们而非 T5 embeds，并分享了 [LaVi-Bridge](https://github.com/ShihaoZhaoZSH/LaVi-Bridge) 等资源来展示现代应用。

**从概念到应用：数据之辩**：对话深入探讨了为什么在某些研究中合成数据集比 MNIST 和 ImageNet 等现实世界数据集更受青睐，提到了 AI 方法中可解释性的价值，并分享了 [StoryDiffusion 网站](https://storydiffusion.github.io/) 以获取见解。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**代码执行找到了 AI 伙伴**：围绕使用 AI 执行生成的代码展开了热烈对话，重点介绍了 **Open Interpreter** 等方法以及开发 **custom tools**（如 `CLITOOL`）。这些讨论对于构建更具交互性和自动化系统的人来说至关重要。

**Langchain 学习新语言**：Langchain 库通过 [langchain4j](https://github.com/langchain4j/langchain4j) 扩展到 Java 生态系统，这对于渴望利用 AI 助手能力的 Java 开发者来说是关键的一步。

**Langchain 获得高性能优化**：**LangChain** 与 **Dragonfly** 的结合在聊天机器人上下文管理方面取得了显著增强，正如一篇详细介绍这些进展的 [博客文章](https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly) 所描述的那样。

**去中心化搜索创新**：社区正热议 **LangChain** 去中心化搜索功能的开发，该功能有望通过用户拥有的索引网络（index network）提升搜索功能。这项工作在最近的一条 [推文](https://twitter.com/indexnetwork_/status/1786110169396429093) 中得到了展示。

**使用 Llama 和 LangGraph 的奇点空间**：一位贡献者分享了一段关于使用 **Llama 3** 在没有 vectorstore 的情况下实现 *Retrieval-Augmented Generation*（检索增强生成）技术的[视频](https://www.youtube.com/watch?v=vvW2dwvNm2Q)，而另一位贡献者则通过[对比](https://www.youtube.com/watch?v=UcD42NA2WoI)执行领域的 **LangGraph** 与 **LangChain Core** 丰富了对话内容。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Clojure 吸引了工程师对符号编程的兴趣**：工程师们正在讨论与 Python 相比，使用 **Clojure** 进行符号编程（symbolic programming）的便利性，建议通过悬赏任务（bounties）来加速上手 *tinygrad*，并辩论在 ML/AI 领域 **Julia** 是否优于 Clojure。

**tinygrad 的 UOps 让工程师困惑**：有人提议重新格式化 *tinygrad* 的文本 UOps 表示，使其更易于理解（可能类似于 LLVM IR），并解释了这些 UOps 确实是静态单赋值（SSA）的一种形式。

**为 Qualcomm GPU 乐园优化 tinygrad**：讨论强调了 *tinygrad* 通过利用 textures 和 pixel shaders 在 **Qualcomm GPUs** 上高效运行，但提醒激活 DSP 支持可能会使过程复杂化。

**tinygrad 的单线程 CPU 故事**：**George Hotz** 本人确认 *tinygrad* 在 CPU 端是**单线程**运行的，不存在线程冲突。

**理解 tinygrad 的张量探戈**：用户对 `matmul` 函数和张量转置的好奇引发了讨论，另一位用户分享了关于在 tinygrad 中计算符号均值（symbolic mean）的[书面分析](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/symbolic-mean.md)。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Json\Schema 与 llamafile 发生冲突**：`json_schema` 与 **llamafile 0.8.1** 之间的冲突引发了讨论，建议使用 `--unsecure` 作为临时方案，并暗示在未来版本中会有永久修复。

- **寻找更轻量级的机器学习模型**：社区交流了关于轻量级 AI 模型的想法，其中 **phi 3 mini** 被认为太重，而 **Rocket-3B** 因其在低资源系统上的灵活性而被推荐。

- **为 Llamafile 整合缓存**：确认 **llamafile** 确实可以利用来自 **ollama cache** 的模型，只要保持 GGUF 文件兼容性，就可以通过避免重复下载来简化操作。

- **AutoGPT 与 Llamafile 携手并进**：分享了一个集成计划，重点介绍了将 **llamafile** 与 **AutoGPT** 融合的草案 Pull Request；设置说明已发布在 [AutoGPT/llamafile-integration](https://github.com/Mozilla-Ocho/AutoGPT/tree/draft-llamafile-support/autogpts/autogpt/llamafile-integration)，正等待维护者反馈。

- **为 Llamafile 选择正确的本地模型**：聚焦实时问题解决，一位用户在区分了实际模型文件和元数据后，成功让 **llamafile** 配合本地缓存的 **.gguf** 文件运行。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Mixtral 问题频发**：**mixtral transformers** 因影响微调（finetune）性能的 Bug 而遇到障碍；参考资料包括 [Twitter](https://twitter.com/kalomaze/status/1786869036946522256)、[Gist](https://gist.github.com/kalomaze/661b79095fdd91df8a84802f7cb6f26a) 和一个[已关闭的 GitHub PR](https://github.com/huggingface/transformers/pull/30658)。目前尚不清楚该 Bug 是仅影响训练还是也影响推理生成，需要进一步审查。

**量化版 LLaMA-3 性能受损**：一则 Reddit 帖子显示，与 LLaMA-2 相比，量化（quantization）显著降低了 LLaMA-3 的性能，并提供了一项可能有启发性的 [arXiv 研究](https://arxiv.org/abs/2404.14047)。Meta 的缩放策略可能是导致 LLaMA-3 精度下降问题的原因，而 [GitHub PR #6936](https://github.com/ggerganov/llama.cpp/pull/6936#issuecomment-2083214112) 和 [Issue #7088](https://github.com/ggerganov/llama.cpp/issues/7088#issuecomment-2094933215) 讨论了潜在的修复方案。

**结识社区新模型**：对话表明 **8x22b Mistral** 正被用于当前的工程任务，但未披露具体的性能指标或使用细节。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **AI 语音：真假难辨**：[The Atlantic](https://www.theatlantic.com/technology/archive/2024/05/elevenlabs-ai-voice-cloning-deepfakes/678288/) 发表了一篇文章，讨论了 **ElevenLabs** 如何创建先进的 AI 语音克隆技术。用户对 **ElevenLabs** 的能力表现出既着迷又警惕的反应，其中一人对限制完全访问此类内容的付费墙表示不屑。
  
- **Prometheus 2：评判评判者**：[最近的一篇 arXiv 论文](https://arxiv.org/abs/2405.01535)介绍了 **Prometheus 2**，这是一个与人类和 **GPT-4 判断**保持一致的语言模型评估器，旨在解决专有语言模型中的透明度和成本问题。尽管该论文显著忽略了该模型表现不佳的 *RewardBench* 评分，但社区对测试 **Prometheus 2** 的评估能力表现出浓厚兴趣。
  
- **经典 RL 之谜**：**rl** 频道的对话探讨了经典强化学习（Reinforcement Learning）中尚未探索的领域。讨论重点强调了价值函数（value function）在 **PPO** 和 **DPO** 等方法中的重要性，并强调了其在 RL 系统规划中的关键作用。
  
- **John 模糊回应之谜**：在 **random** 频道中，成员们分享了对重复成功的隐秘担忧，并开玩笑说某个“john”对一项提议给出了模棱两可的回应。这些陈述背后的相关性和背景仍不清楚。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Anthropic 的 Prompt 生成器引发关注**：工程师们讨论了 **Anthropic console** 中提供的一个新的 **prompt generator tool**（Prompt 生成工具），这对于寻求高效生成 Prompt 的人来说可能非常有用。
- **礼貌模式测试运行**：测试了该工具“礼貌地改写句子”的能力，产生的结果受到了成员们的好评。
- **破译系统机制**：目前正在努力了解该工具的 system prompt 是如何运作的，重点是揭开其中嵌入的 **k-shot examples** 的秘密。
- **提取长内容**：从该工具中提取完整数据一直面临挑战，有报告称 system prompt 被截断，特别是在冗长的“苏格拉底式数学导师”示例期间。
- **揭秘**：一旦成功完整提取，将承诺向社区分享完整的 system prompt，这对于那些对 Prompt Engineering 感兴趣的人来说可能是一个资源。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **伪造数据实验**：一位成员正在寻找一个**伪造数据集**，旨在测试 **Llama 3** 和 **Phi3** 模型的微调，这暗示了他们的实验并不要求数据的真实性。
- **通过快速计算加速 AI**：表现出潜力的 Skunkworks AI 项目可以获得 **Fast compute grants**（快速计算资助），更多详情见[最近的推文](https://twitter.com/PrimeIntellect/status/1786386588726960167)。
- **YouTube 上的 AI 教育内容**：分享了一个与 AI 相关的教育类 [YouTube 视频](https://www.youtube.com/watch?v=vvW2dwvNm2Q)，可能为社区正在进行的技术讨论增添价值。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **LLM 将错误日志转化为启发**：一种利用 LLM 在运行 `conda activate` 命令后迅速总结错误的方法已被证明有效，并建议将该方法集成到 [LLM README](https://github.com/simonw/llm/blob/main/README.md) 文档中。
- **Bash 魔法遇上 LLM 洞察**：一个新编写的 `llm-err` bash 函数已提上日程，旨在将命令输出直接输入 LLM 以进行快速错误诊断，进一步简化工程师的错误排查流程。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **寻找奥斯汀的 AI 专家**：向位于德克萨斯州**奥斯汀**的 AI 专业人士发出了友好问候。
- **Finexov 的融资前沿**：**Vivien** 介绍了 **Finexov**，这是一个旨在简化 **R&D**（研发）融资机会识别的 AI 平台，目前已开展初步合作并获得 **Founder Institute** ([fi.co](https://fi.co/)) 的支持。
- **为 Finexov 寻找技术领导者**：正在寻找一位具有深厚 **ML** 背景的 **CTO 联合创始人**来领导 Finexov，并准备应对团队建设和融资的挑战；优先考虑常驻欧洲或中东的候选人，会说法语者优先。
- **迪拜聚会预告**：Vivien 预告今年 6 月可能在**迪拜**举行聚会，邀请潜在合作伙伴讨论与 Finexov 的合作机会。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI21 Labs 突破边界**：AI21 Labs 表明了其进一步扩展技术的雄心。工作人员鼓励社区成员通过私信分享他们的使用案例和见解。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **获取算力支持**：感兴趣的各方有机会获得 **快速算力资助 (fast compute grants)**；一名成员分享的推文呼吁申请或提名以授予算力资源，这对 AI 研究和项目非常有益。[查看推文了解详情](https://twitter.com/PrimeIntellect/status/1786386588726960167)。

---

# PART 2: 各频道详细摘要与链接

**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1235848656473690112)** (791 条消息🔥🔥🔥): 

- **关于 Llama3 微调和 GGUF 转换的讨论**：用户一直在尝试使用 [Unsloth](https://huggingface.co/unsloth) 微调 Llama3，并将微调后的模型转换为 GGUF，结果各异。一些人报告了转换后出现无限生成的问题，并被引导关注一个 [GitHub issue](https://github.com/ggerganov/llama.cpp/issues/7062)，该 issue 强调了转换为 GGUF 的模型存在的问题。
  
- **关于 Unsloth 全量微调的咨询**：一位用户对使用 Unsloth 进行全量微调（Full Finetuning，而不仅仅是 LoRA）的可能性感到好奇，引发了关于潜在 VRAM 节省和性能的讨论。Unsloth 社区成员就如何实现这一目标提供了见解，并引用了一个 [GitHub feature request](https://github.com/unslothai/unsloth)。

- **深度量化模型性能调查**：一位用户质疑了针对 7B 模型的 4 Bit Q2_K 等深度量化的有效性，建议在低资源应用中可能应改用 Phi-3，强调了为模型性能选择正确量化级别的重要性。

- **资源分享与 Unsloth 故障排除**：用户分享了他们的经验，并就运行 Unsloth 模型的云供应商（如 Tensordock）、Unsloth Studio 的使用，以及处理微调数据集、量化效果和使用不同推理引擎的通用技巧提供了建议。

- **关于使用 LLM 微调低资源语言的不确定性**：一位考虑使用 LLM 微调低资源语言的用户寻求关于 LLM 与 T5 等模型效果对比的建议。社区讨论强调了 Phi-3 等模型在此类任务中的潜力，并就如何处理微调过程的不同方面提供了建议。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNf">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/papers/2402.05119">论文页面 - A Closer Look at the Limitations of Instruction Tuning</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - 由 ggml-org 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct">unsloth/Phi-3-mini-4k-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - 由 NyxKrage 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit">unsloth/Phi-3-mini-4k-instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/comment/kefkdut/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ckvx9l/part2_confirmed_possible_bug_llama3_gguf/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ckvx9l/part2_confirmed_possible">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">主页</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth?search_models=llama-3-70b">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://github.com/IBM/unitxt">GitHub - IBM/unitxt: 🦄 Unitxt: 一个用于准备训练和评估数据的 Python 库</a>: 🦄 Unitxt: 一个用于准备训练和评估数据的 Python 库 - IBM/unitxt</li><li><a href="https://github.com/g">Grizzly</a>: Grizzly 有 9 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices">主页</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine">Cerebras Systems 发布拥有惊人的 4 万亿个晶体管的全球最快 AI 芯片 - Cerebras</a>: 第三代 5nm Wafer Scale Engine (WSE-3) 为业界最具扩展性的 AI 超级计算机提供动力，通过 2048 个节点可达 256 exaFLOPs</li><li><a href="https://huggingface.co/docs/transformers/v4.40.1/en/pad_truncation#padding-and-truncation">Padding 和 truncation</a>: 未找到描述</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=WxQbWTRNTxY&t=83s">如何微调 Llama 3 以获得更好的指令遵循能力？</a>: 🚀 在今天的视频中，我很高兴能带你了解微调 LLaMA 3 模型以实现最佳指令遵循的复杂过程！从设置...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ckcw6z/1m_context_models_after_16k_tokens/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062">合并 LORA Adapter 后的 Llama3 GGUF 转换似乎会随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>: 我正在使用 Unsloth 在 llama3-8b 上微调 LORA 指令模型。1：我将模型与 LORA adapter 合并为 safetensors 2：在 Python 中直接使用合并后的模型进行推理...</li><li><a h

ref="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094875716">Llama3 GGUF 转换与合并的 LoRA Adapter 似乎会随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>：我正在使用 Unsloth 对 llama3-8b 的 Instruct 模型进行 LoRA 微调。1：我将模型与 LoRA adapter 合并为 safetensors 2：在 python 中直接使用合并后的模型运行推理...</li><li><a href="https://llama-hub.com/article_detail/060ef5ec-1fd6-4662-a428-6bbc6f3a4496">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1235890587337494528)** (107 条消息🔥🔥): 

- **LLaMA3 的露骨内容警报**：一位用户报告称，在输入淫秽查询时，**LLaMa3** 生成了不当且露骨的内容，质疑该模型的审查程度。[另一位用户](https://www.github.com/status-check/status) 发现，即使使用系统提示词（system prompts）来防止此类响应，也会得到类似的结果。
- **支持者的新角色**：在关于支持者角色的简短讨论中，用户了解到新增了一个 "**regulars**" 角色，并且成为会员或捐赠至少 $10 的用户可以进入私有的支持者频道。
- **RTX 4090 获得 Suprim 优惠**：在关于显卡交易的新讨论中，有人指出 **MSi GeForce RTX 4090 SUPRIM LIQUID X** 正在以 $1549 的价格促销，一位用户敦促其他人抓住这个机会。该显卡与其他型号相比更紧凑的尺寸引发了进一步的辩论。
- **Kendrick 与 Drake 的动态**：用户讨论了 Kendrick Lamar 和 Drake 恩怨的最新进展，指出 Kendrick 的曲目《Meet the Grahams》在 Drake 的《Family Ties》发布后不久便发布，在说唱界引起了巨大轰动。
- **YouTube 上的 Unsloth.ai**：一段对话涉及一位用户祝贺另一位向 PyTorch 团队进行演示，并引导他们观看来自 Unsloth.ai 的 [YouTube 视频](https://www.youtube.com/watch?v=MQwryfkydc0)，暗示很快会发布进一步的更新。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://paperswithcode.com/paper/x-lora-mixture-of-low-rank-adapter-experts-a">Papers with Code - X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Molecular Design</a>：已在 3 个代码库中实现。</li><li><a href="https://www.reddit.com/r/buildapcsales/comments/1cljlba/gpu_msi_geforce_rtx_4090_suprim_liquid_x_24_gb/">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=MQwryfkydc0">Unsloth.ai：轻松微调和训练 LLM</a>：未找到描述</li><li><a href="https://huggingface.co/blog/mayank-mishra/padding-free-transformer">在微调期间使用无填充 Transformer 层节省显存</a>：未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1235848585611051049)** (1215 条消息🔥🔥🔥): 

- **Llama3 GGUF 转换问题已定位**：用户发现使用 llama.cpp 对 Llama3 模型进行 GGUF 转换会失败，导致训练数据被篡改或丢失，且丢失模式不明确，无论使用 FP16 还是 FP32 转换方法都会出现。这些异常甚至在 F32 中也会发生，证明该问题与精度损失无关。
- **换行符可能存在正则表达式不匹配**：该问题可能与 regex 库有关，其中 `\n` 序列被错误地分词（tokenized），这可能是由于不同 regex 库的行为差异导致的。建议的修复方案是修改 tokenizer.json 的 regex 模式以提高跨 regex 库的兼容性，但对于不同长度的 `\n` 的影响仍存疑。
- **问题不仅限于 GGUF**：在 ooba 等应用中，使用 AWQ 也发现了类似的推理问题，这表明分词器（tokenizer）或分词过程存在问题，而不仅仅是 GGUF 格式的问题。Unsloth 的推理函数似乎表现良好，暗示问题可能特定于 llama.cpp。
- **多个平台受到影响**：依赖于 llama.cpp 的平台（如 ollama 和 lm studio）也面临相关 Bug，不同界面均报告了分词问题，可能影响广泛的用户和应用。
- **社区合作寻求解决方案**：用户贡献（包括 regex 修改）正在被讨论和测试，以提供 GGUF 转换难题的临时修复方案，重点在于查明问题是特定于 Unsloth 微调过程还是 llama.cpp 的分词方法。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1-GGUF">Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharin">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cltac3/part3_cause_to_issue_found_possible_bug_llama3/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/loading#json">Load</a>: 未找到描述</li><li><a href="https://github.com/xaedes/llama.cpp/tree/finetune-lora/examples/export-lora">llama.cpp/examples/export-lora at finetune-lora · xaedes/llama.cpp</a>: Facebook LLaMA 模型的 C/C++ 移植版本。通过在 GitHub 上创建账号来为 xaedes/llama.cpp 的开发做出贡献。</li><li><a href="https://www.reddit.com/user/Dependent_Factor_204/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/casper-hansen/AutoAWQ/blob/main/examples/generate.py">AutoAWQ/examples/generate.py at main · casper-hansen/AutoAWQ</a>: AutoAWQ 实现了用于 4-bit 量化的 AWQ 算法，在推理过程中可实现 2 倍加速。文档：- casper-hansen/AutoAWQ</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/5360">creating gguf model from lora adapter · ggerganov/llama.cpp · Discussion #5360</a>: 我有一个由 convert-lora-to-ggml.py 创建的 ggml 适配器模型 (ggml-adapter-model.bin)。现在我的疑问是如何从中创建完整的 GGUF 模型？我见过使用 ./main -m models/llama...</li><li><a href="https://github.com/ScottMcNaught">ScottMcNaught - Overview</a>: ScottMcNaught 有一个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/unslothai/unsloth/pull/371">llama.cpp failing by bet0x · Pull Request #371 · unslothai/unsloth</a>: llama.cpp 无法为训练好的模型生成量化版本。错误：你可能需要自己编译 llama.cpp，然后再次运行。你不需要关闭这个 Python 程序。运行...</li><li><a href="https://x.com/bartowski1182/status/1786038369132171444?t=hJfQz8lGt9v31yZRG4X1vA&s=09">bartowski (@bartowski1182) 的推文</a>: 经过几天的计算（因为我不得不重新开始），它终于上线了！带有 tokenizer 修复的 Llama 3 70B GGUF :)  https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF  另外，刚刚订购了...</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/f4ab2a41476600a98067a9474ea8f9e6db41bcfa">llama : fix BPE pre-tokenization (#6920) · ggerganov/llama.cpp@f4ab2a4</a>: * 将 deepseeker 模型的更改合并到 main 分支
 
 * 将正则表达式模式移动到 unicode.cpp 并更新了 unicode.h
 
 * 移动了头文件
 
 * 解决了问题
 
</li></ul>

* 添加并重构了 unic...</li><li><a href="https://github.com/unslothai/unsloth/issues/430">GGUF 损坏 - llama-3 · Issue #430 · unslothai/unsloth</a>: 来自 ggerganov/llama.cpp#7062 和 Discord 聊天的发现：复现用的 Notebook：https://colab.research.google.com/drive/1djwQGbEJtUEZo_OuqzN_JF6xSOUKhm4q?usp=sharing Unsloth + float16 + QLoRA = 正常工作...</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">主页</a>: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/ca3632602091e959ed2ad4c09c67a7c790b10d31">readme : 添加关于 convert.py 不支持 LLaMA 3 的说明 (#7065) · ggerganov/llama.cpp@ca36326</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-savin">主页</a>: 微调 Llama 3, Mistral &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/210">我让 unsloth 在原生 windows 上运行了。 · Issue #210 · unslothai/unsloth</a>: 我让 unsloth 在原生 windows 上运行了（无需 WSL）。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一个完整的安装教程，我本想在这里写完，但我现在在手机上...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7021">无法将 llama3 8b 模型转换为 gguf · Issue #7021 · ggerganov/llama.cpp</a>: 请包含有关您的系统信息、重现 Bug 的步骤以及您正在使用的 llama.cpp 版本。如果可能，请提供一个重现该问题的最小代码示例...</li><li><a href="https://github.com/ggerganov/llama.cpp/tree/gg/bpe-preprocess">GitHub - ggerganov/llama.cpp 分支 gg/bpe-preprocess</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6965">jaime-m-p 提交的 llama3 自定义正则分割 · Pull Request #6965 · ggerganov/llama.cpp</a>: unicode_regex_split_custom_llama3() 的实现。</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094961774">带有合并 LoRA Adapter 的 Llama3 GGUF 转换似乎会随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>: 我正在使用 Unsloth 在 llama3-8b 上微调 LoRA 指令模型。1：我将模型与 LoRA Adapter 合并为 safetensors 2：在 Python 中直接使用合并后的模型运行推理...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062">带有合并 LoRA Adapter 的 Llama3 GGUF 转换似乎会随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>: 我正在使用 Unsloth 在 llama3-8b 上微调 LoRA 指令模型。1：我将模型与 LoRA Adapter 合并为 safetensors 2：在 Python 中直接使用合并后的模型运行推理...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2095465106">带有合并 LoRA Adapter 的 Llama3 GGUF 转换似乎会随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>: 我正在使用 Unsloth 在 llama3-8b 上微调 LoRA 指令模型。1：我将模型与 LoRA Adapter 合并为 safetensors 2：在 Python 中直接使用合并后的模型运行推理...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094955278">带有合并 LoRA Adapter 的 Llama3 GGUF 转换似乎会随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>: 我正在使用 Unsloth 在 llama3-8b 上微调 LoRA 指令模型。1：我将模型与 LoRA Adapter 合并为 safetensors 2：在 Python 中直接使用合并后的模型运行推理...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094875716">带有合并 LoRA Adapter 的 Llama3 GGUF 转换似乎会随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>: 我正在使用 Unsloth 在 llama3-8b 上微调 LoRA 指令模型。1：我将模型与 LoRA Adapter 合并为 safetensors 2：在 Python 中直接使用合并后的模型运行推理...</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/convert.py">llama.cpp/convert.py (master 分支) · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094948789">带有合并 LoRA Adapter 的 Llama3 GGUF 转换似乎会随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>: 我正在使用 Unsloth 在 llama3-8b 上微调 LoRA 指令模型。1：我将模型与 LoRA Adapter 合并为 safetensors 2：在 Python 中直接使用合并后的模型运行推理...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2095371349">带有合并 LoRA Ada...</a>

pter 似乎随机丢失训练数据 · Issue #7062 · ggerganov/llama.cpp</a>: 我正在运行 Unsloth 在 llama3-8b 上微调 LORA Instruct 模型。1：我将模型与 LORA 适配器合并为 safetensors 2：在 python 中直接使用合并后的模型运行推理...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1235848362516021248)** (80 条消息🔥🔥): 

- **关于模型尺寸讨论频道的建议**：一位用户建议在 Unsloth Discord 上创建一个单独的频道，用于讨论部署大型语言模型 (LLM) 的**成功经验与策略**。对话强调了分享经验对于增强集体学习的价值。

- **推动基于 Llama-3-8B 的项目**：RomboDawg 宣布发布了一个新的编码模型，该模型增强了 **Llama-3-8B-Instruct**，其性能可与 **Llama-3-70B-Instruct** 媲美。该模型可以在 [这里](https://huggingface.co/rombodawg/Codellama-3-8B-Finetuned-Instruct) 访问，并表示对预计在大约三天内发布的第 2 版感到兴奋。

- **知识图谱 LLM 变体发布**：M.chimiste 开发了一个 **Llama-3 变体** 以辅助知识图谱构建，命名为 **LLaMA-3-8B-RDF-Experiment**，强调了其在生成知识图谱三元组方面的实用性以及在基因组数据训练方面的潜力。该模型可以在 [Hugging Face 模型库](https://huggingface.co/M-Chimiste/Llama-3-8B-RDF-Experiment) 中找到。

- **加密协作的前景**：在一次深入讨论中，一位用户正在寻求关于构建一个可能将加密元素集成到区块链技术中的系统的建议和协作讨论，并表示有兴趣向社区学习。

- **AI 增强的 Web 开发工具主题**：Oncord 被展示为一个提供内置营销和商业工具的现代 Web 开发平台，其开发者正在集成 **Unsloth AI** 进行 **LLM 微调**，以提供代码补全并可能支持 AI 驱动的重新设计功能。关于 Oncord 的更多信息可以在 [这里](https://www.oncord.com/) 找到。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.llama-hub.com/models">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/M-Chimiste/Llama-3-8B-RDF-Experiment">M-Chimiste/Llama-3-8B-RDF-Experiment · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/dog-awkward-awkward-dog-staring-dog-patchibana-gif-13086408744970718509">Dog Awkward GIF - Dog Awkward Awkward dog - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://pubmed.ncbi.nlm.nih.gov/28288169/):">miR-200 家族在溃疡性结肠炎患者的异型增生病变中增加 - PubMed</a>：UC-异型增生与粘膜中 miRNA 表达的改变和 miR-200b-3p 水平升高有关。</li><li><a href="https://x.com/dudeman6790/status/1786783966738919738">RomboDawg (@dudeman6790) 的推文</a>：宣布 Codellama-3-8B，这是在完整的 OpenCodeInterpreter 数据集上对 llama-3-8b-instruct 进行的 Qalore 微调。它的编码能力远优于基础 instruct 模型，并且在代码迭代方面表现极佳。Forgi...</li><li><a href="https://www.llama-hub.com/model_detail/4df65c9a-6b23-413e-b4eb-9d34c446db48">Llama-3-8B-Instruct-Coder</a>：未找到描述</li><li><a href="https://www.oncord.com/">Oncord - 数字营销软件</a>：集网站、电子邮件营销和电子商务于一体的直观软件平台。Oncord 托管的 CMS 使其变得简单。</li><li><a href="https://www.tryoncord.com/admin/">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1237119442899435582)** (3 条消息): 

- **期望微调 LVLM**：一位成员表达了对**通用 LVLM 微调方式**的愿望，表明了对语言视觉模型定制和优化的持续兴趣。

- **对 MoonDream 微调的兴趣**：另一位成员推荐**支持 Moondream**，这是一个微型视觉语言模型，目前仅支持微调 **phi 1.5 文本模型**。他们提供了一个 GitHub notebook 作为资源：[GitHub 上的 moondream/notebooks/Finetuning.ipynb](https://github.com/vikhyat/moondream/blob/main/notebooks/Finetuning.ipynb)。

**提到的链接**：<a href="https://github.com/vikhyat/moondream/blob/main/notebooks/Finetuning.ipynb">moondream/notebooks/Finetuning.ipynb at main · vikhyat/moondream</a>：微型视觉语言模型。通过在 GitHub 上创建一个账号来为 vikhyat/moondream 的开发做出贡献。

  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1235963163971682304)** (854 条消息🔥🔥🔥):

- **Perplexity 的新挑战者**：用户正在讨论 Perplexity AI 的优势，特别是其新的 Pages 功能，该功能允许创建综合报告。
- **AI 与自学习**：一些人讨论了像 OpenAI 的 GPT 这样的 AI 引擎教用户编程基础并帮助编写代码的可能性，支持具有自我改进能力的自给自足 AI 的想法。
- **AGI 定义的演变**：社区正在就 AI 的现状及其与真正 AGI (Artificial General Intelligence) 的接近程度展开辩论，对于像 ChatGPT 这样的现代 AI 是否符合早期 AGI 的标准持有不同意见。
- **对 AI 生成音乐的渴望**：用户对 AI 生成的音乐表现出兴趣，提到了 Udio 等服务，并讨论了 OpenAI 是否应该发布自己的 AI 音乐服务。
- **AI 作为扩展工具**：对话探讨了 AI 目前如何增强人类生产力，以及 AI 未来可能接管平凡和复杂任务的潜力，同时也反映了这可能如何颠覆我们的社会经济模式。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://scholar.google.com/citations?user=4FsWE64AAAAJ)">Google Scholar Citations</a>: 未找到描述</li><li><a href="https://tenor.com/view/dirty-docks-shawty-triflin-shawty-triflin-she-gif-22455514">Dirty Docks Shawty GIF - Dirty Docks Shawty Triflin - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.nature.com/articles/d41586-023-00107-z">ChatGPT listed as author on research papers: many scientists disapprove</a>: 至少有四篇文章将该 AI 工具列为共同作者，出版商正争相规范其使用。</li><li><a href="https://github.com/catppuccin/catppuccin">GitHub - catppuccin/catppuccin: 😸 Soothing pastel theme for the high-spirited!</a>: 为充满活力的人准备的舒缓粉彩色调主题！欢迎在 GitHub 上为 catppuccin 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1235851459891957821)** (40 messages🔥): 

- **慢而稳并不总是赢家**：成员们报告 **GPT-4 Turbo** 的延迟显著增加，一些人的响应时间比平时慢了 **5-10 倍**。
- **对话限制**：关于 GPT-4 的消息上限存在困惑，用户报告了不同的超时阈值。一些人表示上限在 **25 到 50 条消息**之间，而另一些人则怀疑在**高使用率**期间会有动态调整。
- **OpenAI 平台的 UX 忧郁**：针对 OpenAI 新项目功能的体验出现了投诉，涉及**项目管理**、**删除**和**导航**方面的问题；还注意到每个项目**缺乏活动跟踪**。
- **会有 GPT-5 吗？**：用户对 GPT-5 的发布持怀疑态度，讨论了**收益递减**以及它可能是“**1.5 倍于 GPT-4 的性能，但成本是其 2 倍**”的可能性。
- **知识优先级探索**：用户辩论如何让 ChatGPT 在回答前**先搜索其知识库**的策略，涉及 **RAG (Retrieval-Augmented Generation)** 和**知识向量化**等概念，以协助提供上下文相关的答案。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1236180170323267597)** (30 messages🔥): 

- **为提问微调 GPT**：一位成员正在寻求关于如何微调模型以进行提问而非回答的建议，并提到了之前在类似项目中的挣扎。他们指出难以找到合适的用户查询和助手查询对，并考虑使用单元组对话作为微调样本。

- **韧性十足的入职机器人**：成员 **leveloper** 提到一个成功运行的机器人，旨在入职过程中提问，尽管在一个大型服务器上，它仍未被用户的尝试所迷惑。

- **避免负向提示**：**majestic_axolotl_19289** 建议使用负向提示（Negative Prompts）可能会适得其反，因为它们往往会以意想不到的方式影响结果。其他成员讨论了负向提示是否有效，引用了“Contrastive Chain of Thoughts”论文和个人经验。

- **Prompt Engineering 书籍推荐**：成员 **sephyfox_** 推荐了 Teddy Dicus Murphy 的《Wordplay: Your Guide to Using Artificial Intelligence for Writing Software》，认为它对 Prompt Engineering 很有帮助。

- **改进 GPT-4-TURBO 提取产品信息的提示词请求**：成员 **stevenli_36050** 寻求帮助，以优化从 PDF 超市宣传册中提取产品信息、名称和价格并进行相应分类的提示词。

- **讨论 Token Suppression 中的 Logit Bias**：用户 **bambooshoots** 分享了一个链接 (https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api)，关于在 OpenAI API 中使用 Logit Bias 操纵概率以抑制特定 Token。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1236180170323267597)** (30 messages🔥): 

- **寻找提问机器人**：一位成员讨论了微调 GPT 以在对话中生成问题而非提供答案的挑战，强调了在这种场景下定义用户查询和机器人响应结构的难度。

- **Contrastive Chain of Thought (CCoT) 讨论**：关于在 Prompting 策略中使用负面提示（negative prompts）的辩论。对话提到了关于 "Contrastive Chain of Thoughts" 的论文，并质疑了在长对话中使用 CCoT 的有效性，随后邀请读者在 [AIEmpower 博客](https://aiempower.com/)上进一步阅读。

- **Prompt Engineering 资源与技术分享**：用户分享了关于 Prompt Engineering 的资源，包括 Teddy Dicus Murry 的书籍《Wordplay: Your Guide to Using Artificial Intelligence for Writing Software》的推荐，以及 Ronnie Sheer 的 LinkedIn 学习课程。

- **提取超市产品数据**：一位用户寻求关于改进 GPT-4-TURBO Prompt 的建议，以便从 PDF 超市宣传册中识别产品名称和价格，并希望以 CSV 格式输出结果。
 
- **用于 Token 概率操纵的 Logit Bias**：一位成员引用了 Logit Bias 作为在 Prompt 中操纵 Token 概率的方法，并附上了 OpenAI 官方文档链接：[Using logit bias to alter token probability with the OpenAI API](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api)。
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1235849532609265724)** (919 messages🔥🔥🔥): 

- **GPU 兼容性查询**：用户正在寻求有关无法访问 GPU 资源的 Stable Diffusion 安装帮助，提到了诸如 "RuntimeError: Torch is not able to use GPU" 之类的错误。
- **Stability.ai 与 SD3 推测**：对话围绕 Stable Diffusion 3 的预期发布展开，许多人对其是否会发布表示怀疑，而其他人则讨论了如果不发布会产生的影响。
- **微调教程寻求者**：成员们对缺乏 LoRA/DreamBooth/微调模型的详细教程表示沮丧，称现有资源已过时或不够全面。
- **生成独特面孔的求助请求**：有人询问如何使用 AI 训练一个独特的、看起来真实的人物，方法包括在多张面孔上训练 LoRA，或者生成随机面孔然后在该结果上训练 LoRA。
- **关于 Stable Diffusion “Open Source” 性质的讨论**：一些用户讨论了真正“Open Source” AI 艺术生成的障碍，分享了对未来高质量模型 Checkpoints 和训练细节可能进入付费墙的担忧。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://humanaigc.github.io/emote-portrait-alive/">EMO</a>: EMO: Emote Portrait Alive - 在弱条件下使用 Audio2Video Diffusion 模型生成具有表现力的肖像视频</li><li><a href="https://highlight.fm">Highlight: Generate photos with friends</a>: Highlight 是一款通过与朋友共同生成图像来进行白日梦（想象）的应用程序。</li><li><a href="https://stability.ai/news/stable-diffusion-3-research-paper">Stable Diffusion 3: Research Paper &mdash; Stability AI</a>: 继我们宣布 Stable Diffusion 3 的早期预览版之后，今天我们发布了研究论文，概述了即将发布的模型的详细技术细节，并邀请您...</li><li><a href="https://fireworks.ai/models/stability/sd3">Fireworks - Generative AI For Product Innovation!</a>: 使用 Fireworks.ai 以极快的速度使用最先进的开源 LLM 和图像模型，或者无需额外费用即可微调并部署您自己的模型！</li><li><a href="https://www.instagram.com/farina.fab?igsh=YXlsbWRycnIxbjNu">Login • Instagram</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=kqXpAKVQDNU&list=PLXS4AwfYDUi5sbsxZmDQWxOQTml9Uqyd2">How to Install Stable Diffusion - automatic1111</a>: 第 2 部分：如何使用 Stable Diffusion https://youtu.be/nJlHJZo66UA Automatic1111 https://github.com/AUTOMATIC1111/stable-diffusion-webui 安装 Python https://w...</li><li><a href="https://www.youtube.com/watch?v=juP6gpiOY2A">High-Similarity Face Swapping: ControlNet IP-Adapter + Instant-ID Combo</a>: 探索使用 WebUI Forge、IP-Adapter 和 Instant-ID 进行高相似度换脸的艺术，以获得无缝、逼真的效果。🖹 文章教程：- https:/...</li><li><a href="https://github.com/philz1337x/clarity-upscaler">GitHub - philz1337x/clarity-upscaler: Clarity AI | AI Image Upscaler &amp; Enhancer - free and open-source Magnific Alternative</a>: Clarity AI | AI 图像放大器与增强器 - 免费且开源的 Magnific 替代方案 - philz1337x/clarity-upscaler</li><li><a href="https://civitai.com/models/410151/aether-light-lora-for-sdxl">Aether Light - LoRA for SDXL - v1.0 | Stable Diffusion LoRA | Civitai</a>: 商务咨询、商业许可、定制模型和咨询，请通过 joachim@rundiffusion.com 联系我。介绍 Aether Light，我...</li><li><a href="https://github.com/crystian/ComfyUI-Crystools">GitHub - crystian/ComfyUI-Crystools: A powerful set of tools for ComfyUI</a>: 一套功能强大的 ComfyUI 工具。通过在 GitHub 上创建账户为 ComfyUI-Crystools 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cgr74j/comment/l2bxv66/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ciyzn5/comment/l2dhd6q/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://youtu.be/xXNr9mrdV7s?si=PalGotNpAeYl4Zy4">LORA training EXPLAINED for beginners</a>: LoRA 训练指南/教程，以便您了解如何使用 KohyaSS 上的重要参数。使用 Dreamlook.AI 在几分钟内完成训练：https://dreamlook.ai/?...</li><li><a href="https://rentry.co/59xed3#prodigy">THE OTHER LoRA TRAINING RENTRY</a>: Stable Diffusion LoRA 训练科学与笔记，由 The Other LoRA Rentry Guy 提供。这不是安装指南，而是关于如何改进结果、描述不同选项的指南...
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1237067945465876572)** (1 条消息): 

我对应造成的困惑表示抱歉，但作为 AI，我无法直接访问 Discord 服务器、频道或消息。因此，我无法总结名为 ctx-length-research 的 Nous Research AI Discord 频道的内容。如果您能提供想要总结的特定 Discord 消息文本，我很乐意为您提供帮助。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1235957375173656637)** (20 条消息🔥):

- **相机色彩的演变**：一位成员幽默地指出，对比 2002 年和 2024 年**圣彼得堡起义广场利戈夫斯基大街 (Saint Petersburg, Ligovsky Avenue at Vosstaniya Square)** 的照片，可以看出*相机的色彩还原度变得更高了*。
- **烹饪风味融合**：简单提到了配有蛋黄酱和黑麦面包的格瓦斯 **Okroshka**（俄式冷汤），可能是在讨论或引用俄罗斯传统美食。
- **关于 SVM 的咨询**：一位成员询问：“什么是 SVM？”另一位成员迅速澄清说 SVM 代表 **Support Vector Machine**。
- **改进 FreeGPT.today 的 UX**：一位成员请求对其网站 [FreeGPT.today](https://freegpt.today/) 的用户体验提供反馈，邀请其他人注册、聊天并测试生成图表的 PDF 上传功能。提出了几项改进建议，包括增加 Google 身份验证、将默认登录落地页更改为“立即聊天”、改进 UI 元素以及为文件上传实现进度条。
- **警惕垃圾链接**：提到聊天中分享的一个 Discord 邀请链接实际上是垃圾信息，导致分享者被封禁。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/J1ZiaE7cqQY">Recipic Demo</a>：是否曾对晚餐或午餐做什么感到困惑？如果有一个网站，你只需上传现有的食材就能获得食谱……</li><li><a href="https://freegpt.today/">FreeGPT.today - 最强大的免费 AI 语言模型！</a>：免费访问最强大的 AI 语言模型。无需信用卡。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1235972681413689445)** (47 条消息🔥): 

- **使用 LLM 探索 Taskmaster**：分享了一个使用结构化数据管理、状态机和 OpenAI API 实现的 **Taskmaster** 节目代码实现。代码可在 [GitHub](https://github.com/LEXNY/Taskmaster-LLM) 上获得。
- **评估 LLM 响应**：介绍了另一个 GitHub 仓库，其特色是 **Prometheus**，一个用于评估 LLM 响应的工具，可在 [prometheus-eval](https://github.com/prometheus-eval/prometheus-eval) 获取。
- **LLM 的 VRAM 消耗计算器**：提到了一个 Hugging Face Space，其中包含一个 LLM Model VRAM Calculator，以帮助用户确定他们需要多少 VRAM，可在[此处](https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator)查看。
- **修复 Mistral 模型问题**：讨论集中在修复 **Mistral 模型**的问题上，并重点介绍了解决这些问题的潜在 Pull Requests (PRs)。关于修改的持续对话，特别是围绕 rotary embeddings 的部分，可以在 [GitHub](https://github.com/huggingface/transformers/pull/30658) 上找到最新的相关 PR。
- **开放预训练数据集的改进与问题**：提到了一篇最近的论文，该论文检查了用于语言模型的训练语料库的质量。研究讨论了这些数据集中重复、合成和低质量内容的普遍性，详情可见其 [arXiv 论文](https://arxiv.org/abs/2310.20707)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://secretllama.com/">Secret Llama</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2401.17377">Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens</a>: 在神经大语言模型（LLM）时代，$n$-gram 语言模型是否仍然具有相关性？我们的答案是肯定的，并展示了它们在文本分析和改进神经 LLM 方面的价值。这篇...</li><li><a href="https://demo.haystack.zip/">Demo Search Fine Web Dataset</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.19737">Better &amp; Faster Large Language Models via Multi-token Prediction</a>: 诸如 GPT 和 Llama 之类的 LLM 是通过下一个 Token 预测损失进行训练的。在这项工作中，我们建议训练语言模型一次预测多个未来 Token 会导致...</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2401.10774">Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads</a>: 由于自回归解码过程缺乏并行性，LLM 的推理过程通常受到限制，导致大多数操作受限于内存...</li><li><a href="https://arxiv.org/abs/2310.20707">What&#39;s In My Big Data?</a>: 大型文本语料库是语言模型的支柱。然而，我们对这些语料库内容的理解有限，包括一般统计数据、质量、社会因素以及...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1clinlb/bringing_2bit_llms_to_production_new_aqlm_models/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1clmo7u/phi3_weights_orthogonalized_to_inhibit_refusal/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/LEXNY/Taskmaster-LLM">GitHub - LEXNY/Taskmaster-LLM</a>: 通过在 GitHub 上创建账户来为 LEXNY/Taskmaster-LLM 的开发做出贡献。</li><li><a href="https://github.com/prometheus-eval/prometheus-eval">GitHub - prometheus-eval/prometheus-eval: Evaluate your LLM&#39;s response with Prometheus 💯</a>: 使用 Prometheus 评估你的 LLM 响应 💯。通过在 GitHub 上创建账户来为 prometheus-eval/prometheus-eval 的开发做出贡献。</li><li><a href="https://youtu.be/oNwoA5akBlg">NVIDIA CEO Jensen Huang Leaves Everyone SPEECHLESS (Supercut)</a>: NVIDIA（#nvda 股票）创始人兼首席执行官黄仁勋在斯坦福经济政策研究所（SIEPR）演讲的亮点。亮点包括...</li><li><a href="https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">Refusal in LLMs is mediated by a single direction — LessWrong</a>: 这项工作是 Neel Nanda 在 ML Alignment &amp; Theory Scholars 项目（2023-24 冬季班）中的一部分，由...共同指导。</li><li><a href="https://github.com/huggingface/transformers/pull/30658">[WIP][FIX] Fix Mixtral model by casper-hansen · Pull Request #30658 · huggingface/transformers</a>: 此 PR 是基于 @kalomaze 实现的进行中工作（WIP），旨在修复 Mixtral 模型。众所周知，由于代码中的某些错误，Mixtral 一直难以训练。请注意，这...</li><li><a href="https://github.com/huggingface/transformers/pull/">Pull requests · huggingface/transformers</a>: 🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。- Pull requests · huggingface/transformers</li><li><a href="https://app.wordware.ai/r/81fef99d-70e5-4c6a-ad0d-bd1057bfc818">Wordware - WebIntellect - Search with ScratchPad-Think Framework (V2)</a>: 利用 &lt;ScratchPad-Think&gt; 的力量进行日常网络搜索。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1235849078479519855)** (717 条消息🔥🔥🔥): 

- **Hermes 在经典 Llama 缩放下的卓越表现**：**Hermes 2 Pro Llama 8B** 在 32GB Nvidia v100 Tesla 上使用 **vLLM** 通过 RoPE 缩放将上下文容量扩展至约 32k，且没有明显的性能下降。根据用户经验，在 16k 时可提供完美的召回率。

- **设置增强上下文**：建议修改 Hugging Face 上 Hermes 模型的 `config.json`，并在初始化服务器之前调整 RoPE 缩放因子以进行上下文扩展。

- **Serverless Llama 的局限性**：用户报告了不同模型推理提供商之间的各种功能和限制，需要协调语法和 JSON 模式等功能。根据 **vLLM** GitHub issues 页面的讨论，这些功能目前仅在 **llama.cpp** 中受支持，**vLLM** 尚不支持。

- **对 Llama-3-120B-Instruct 的高度期待**：一个名为 **Meta-Llama-3-120B-Instruct** 的 **Hugging Face** 模型（一个 self-merged 模型）因其据称提升的性能而引起了关注和兴趣；然而，一些用户提醒在没有经过彻底的 benchmarking 之前，应对此类炒作保持谨慎。

- **平衡计算资源与模型性能**：用户讨论了使用更强大的 GPU（如 **Azure 的 NC80adis_H100_v5**）的权衡，以及在需要大 context sizes 的任务中，如何在充足的 VRAM、latency 和 tokens per second 之间取得平衡以供实际使用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://huggingface.co/turboderp/Cat-Llama-3-70B-instruct">turboderp/Cat-Llama-3-70B-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/dudeman6790/status/1786783966738919738">来自 RomboDawg (@dudeman6790) 的推文</a>: 发布 Codellama-3-8B，这是一个基于完整的 OpenCodeInterpreter 数据集对 llama-3-8b-instruct 进行的 Qalore 微调版本。它的代码编写能力远优于基础 instruct 模型，并且在代码迭代方面表现出色。Forgi...</li><li><a href="https://huggingface.co/cognitivecomputations/Meta-Llama-3-120B-Instruct-gguf">cognitivecomputations/Meta-Llama-3-120B-Instruct-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B#prompt-format-for-json-mode--structured-outputs">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1a-aQvKC9avdZpdyBn4jgRQFObTPy1JZw?usp=sharing#scrollTo=2EoxY5i1CWe3">Google Colab</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.19234">Multi-hop Question Answering over Knowledge Graphs using Large Language Models</a>: 知识图谱 (KGs) 是具有特定结构的大型数据集，代表了大型知识库 (KB)，其中每个节点代表一个关键实体，它们之间的关系是类型化的边。自然语...</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw2.5-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2405.00200">In-Context Learning with Long-Context Models: An In-Depth Exploration</a>: 随着模型上下文长度的不断增加，上下文中可以提供的示例数量已接近整个训练数据集的大小。我们研究了 In-Context Learning 的行为...</li><li><a href="https://x.com/0xblacklight/status/1787329977957982398">来自 Kyle Mistele 🏴‍☠️ (@0xblacklight) 的推文</a>: 顺便说一下，我用 @vllm_project 测试了它，它可以将 @NousResearch 的 Hermes 2 Pro Llama 3 8B 扩展到约 32k 上下文，并具有出色的连贯性和性能（我让它总结了 @paulg 的文章）下载...</li><li><a href="https://huggingface.co/datasets/CarperAI/pilev2-dev">CarperAI/pilev2-dev · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://x.com/yacinemtb/status/1786958419846418664?s=46&t=stOPrwZiN_fxSK0RuC8Fl">来自 kache (@yacineMTB) 的推文</a>: 当 llama 400b 降临时，你的整个公司都将不复存在。你真的认为你能监管得够快吗？你知道政府动作有多慢吗？只要一个种子文件 (torrent) 发布，你的整个业务...</li><li><a href="https://arxiv.org/abs/2310.00785">BooookScore: A systematic exploration of book-length summarization in the era of LLMs</a>: 总结超过大语言模型 (LLMs) 上下文窗口大小的长篇文档 (>100K tokens) 需要先将输入文档分解为较小的块，然后进行提示...</li><li><a href="https://x.com/yacinemtb/status/1786958419846418664?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 kache (@yacineMTB) 的推文</a>: 当 llama 400b 降临时，你的整个公司都将不复存在。你真的认为你能监管得够快吗？你知道政府动作有多慢吗？只要一个种子文件 (torrent) 发布，你的整个业务...</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/blob/main/examples/crewai_agents.ipynb">NousResearch/Hermes-Function-Calling 项目 main 分支下的 examples/crewai_agents.ipynb</a>: 通过在 GitHub 上创建账号来为 NousResearch/Hermes-Function-Calling 的开发做出贡献。</li><li><a href="https://cloud.google.com/pricing/">价格概览</a>: 通过 Google Cloud 的按需付费定价模式，你只需为你使用的服务付费。无需预付费用。无终止费。</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw4-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw4-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw5.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw5.5-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw6-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw6-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7013">由 K-Mistele 提交的 Pull Request #7013 · ggerganov/llama.cpp：更新服务器 README，增加 RoPE、YaRN 和 KV cache 量化等未记录选项</a>: 我最近更新了我的 llama.cpp，发现有许多服务器 CLI 选项在 README 中没有描述，包括 RoPE、YaRN 和 KV cache 量化以及 flash a...</li>

t...</li><li><a href="https://github.com/theavgjojo/openai_api_tool_call_proxy">GitHub - theavgjojo/openai_api_tool_call_proxy: A thin proxy PoC to support prompt/message handling of tool calls for OpenAI API-compliant local APIs which don&#39;t support tool calls</a>: 一个轻量级代理 PoC，用于为不支持 tool calls 的 OpenAI API 兼容本地 API 提供 tool calls 的 prompt/消息处理支持 - theavgjojo/openai_api_tool_call_proxy</li><li><a href="https://tenor.com/view/mlp-relevant-mylittlepony-interests-gif-4506356">Mlp Relevant GIF - MLP Relevant Mylittlepony - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ckcw6z/1m_context_models_after_16k_tokens/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/huggingface/lerobot">GitHub - huggingface/lerobot: 🤗 LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch</a>: 🤗 LeRobot：基于 Pytorch 的真实世界机器人技术最先进机器学习库 - huggingface/lerobot</li><li><a href="https://github.com/Infini-AI-Lab/Sequoia">GitHub - Infini-AI-Lab/Sequoia: scalable and robust tree-based speculative decoding algorithm</a>: 可扩展且鲁棒的基于树的 speculative decoding 算法 - Infini-AI-Lab/Sequoia</li><li><a href="https://github.com/vllm-project/vllm/issues/1229">Support for grammar · Issue #1229 · vllm-project/vllm</a>: 如果该库能加入对 Grammar 和 GBNF 文件的支持，将非常有益。 https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md</li><li><a href="https://github.com/snakers4/silero-models">GitHub - snakers4/silero-models: Silero Models: pre-trained speech-to-text, text-to-speech and text-enhancement models made embarrassingly simple</a>: Silero Models：预训练的语音转文本、文本转语音和文本增强模型，极其简单易用 - snakers4/silero-models</li><li><a href="https://arxiv.org/abs/2404.17733">Building a Large Japanese Web Corpus for Large Language Models</a>: 为大语言模型（LLMs）构建大规模日语 Web 语料库：开源日语大语言模型（LLMs）已在 CC-100、mC4 和 OSCAR 等语料库的日语部分进行了训练。然而，这些语料库并非为了日语文本质量而创建...</li><li><a href="https://github.com/N8python/SYNTH-8">GitHub - N8python/SYNTH-8: An open-source voice-enabled chatbot. Many features will come soon.</a>: 一个开源的语音启用聊天机器人。许多功能即将推出。 - N8python/SYNTH-8</li><li><a href="https://arxiv.org/abs/2312.15166">SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling</a>: SOLAR 10.7B：通过简单而有效的 Depth Up-Scaling 扩展大语言模型：我们推出了 SOLAR 10.7B，一个拥有 107 亿参数的大语言模型（LLM），在各种自然语言处理（NLP）任务中表现出卓越的性能。受近期努力的启发...</li><li><a href="https://x.com/maziyarpanahi/status/1786751050130608168?s=46">Tweet from Maziyar PANAHI (@MaziyarPanahi)</a>: 做得好 @Gradient_AI_！这个模型非常接近 Instruct 版本，非常令人印象深刻！❤️🚀👏🏽 引用 OpenLLMLeaders (@OpenLLMLeaders) 排行榜新增模型！模型名称 h...</li><li><a href="https://www.paddle.com/ai-launchpad">Scale your AI business with Paddle | AI Launchpad</a>: 未找到描述</li><li><a href="https://blog.arcee.ai/arcee-mergekit-launch-model-merging-hackathon/">Arcee/Mergekit launch Model Merging Hackathon</a>: Arcee 和 MergeKit 通过启动由 AWS 共同赞助的 MergeKit 黑客松来推进模型合并（model merging）创新。提交您的模型合并研究、实验和结果，有机会赢取现金奖励...</li><li><a href="https://github.com/hsiehjackson/RULER">GitHub - hsiehjackson/RULER: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models?</a>: 此仓库包含 RULER 的源代码：您的长上下文语言模型的真实上下文大小是多少？ - hsiehjackson/RULER</li><li><a href="https://github.com/OpenBMB/InfiniteBench#evaluation-result">GitHub - OpenBMB/InfiniteBench: Codes for the paper &quot;∞Bench: Extending Long Context Evaluation Beyond 100K Tokens&quot;: https://arxiv.org/abs/2402.13718</a>: 论文 "∞Bench: Extending Long Context Evaluation Beyond 100K Tokens" 的代码：https://arxiv.org/abs/2402.13718 - OpenBMB/InfiniteBench</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5104">Port of self extension to server by Maximilian-Winter · Pull Request #5104 · ggerganov/llama.cpp</a>: 你好，我将 self extension 的代码移植到了服务器。我已经通过信息检索进行了测试，我在约 6500 个 token 长的文本中插入了上下文之外的信息，它起作用了，至少...</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blockblock">

block/Hermes-2-Pro-Llama-3-8B-bpw3.5-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.5-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.7-exl2">blockblockblock/Hermes-2-Pro-Llama-3-8B-bpw3.7-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/1965">Extending context size via RoPE scaling · ggerganov/llama.cpp · Discussion #1965</a>: 简介：这是一个关于最近提出的扩展 LLaMA 模型上下文大小策略的讨论。最初的想法在这里提出：https://kaiokendev.github.io/til#extending-context-t...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1235868437432107029)** (60 条消息🔥🔥): 

- **LLM 引起热潮**：一位成员表达了在本地 AI 上进行实验的喜悦，分享了他们在该平台上的第一次愉快体验。
- **Hermes 2 Pro Llama 3 对比 Mistral**：讨论围绕 **Hermes 2 Pro Llama 3** 在性能上不如 **Mistral** 展开，深入探讨了 **Mixtral** 较大的模型规模如何使其排名更高，特别是在 **MMLU benchmark** 中。
- **了解 LLaVA 的多模态能力**：关于向 GPT/LLM 教授图像识别，成员们被引导去探索 **LLaVA**，这是一个具有增强视觉和语言理解能力的大型多模态模型，在 11 个基准测试中表现出色。
- **文本生成中的 Tool XML 标签问题**：交流了在迁移到 **LlamaCPP** 时无法生成 `<tool_call>` XML 标签的问题，随后通过将 **LlamaCPP** 更新到最新版本解决了该问题。
- **LoRA Llama 3 8B 训练的速度困扰**：一位成员询问了 **Llama 3 8B** 的 LoRA 训练耗时似乎过长的问题，并将其与其他人在不同设置下报告的更快速体验进行了对比。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.writewithlaika.com)">未找到标题</a>：未找到描述</li><li><a href="https://llava-vl.github.io/">LLaVA</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/tree/main">NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/aniketmaurya/agents/blob/main/src/agents/hermes/functioncall.py#L30">agents/src/agents/hermes/functioncall.py at main · aniketmaurya/agents</a>：一个由 LangChain 驱动的、用于构建带有函数调用的 Agentic 工作流的有趣项目。- aniketmaurya/agents
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1235976657823862865)** (2 条消息): 

- **寻找免费数据集**：一位成员询问了优质免费通用数据集的来源。
  
- **Cynde 核心实现更新**：分享了关于 **Cynde**（一个用于智能耕作的框架）的更新。核心实现已经就绪，贡献者欢迎帮助并努力保持代码整洁，并表示目前有意尚未加入 RAG。更新后的 Readme 和笔记可在 [Neural-Dragon-AI/Cynde](https://github.com/Neural-Dragon-AI/Cynde) 查看。

**提到的链接**：<a href="https://github.com/Neural-Dragon-AI/Cynde/blob/main/README.md">Cynde/README.md at main · Neural-Dragon-AI/Cynde</a>：一个用于智能耕作（Intelligence Farming）的框架。通过在 GitHub 上创建账户为 Neural-Dragon-AI/Cynde 的开发做出贡献。

  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1235895009111179264)** (74 条消息🔥🔥):

- **Anticipation for World-Sim's Return**: 成员们对可能测试新版本 **world-sim** 的角色分配表示兴奋并进行询问，其中一位成员特别激动，因为这恰好是他们的生日。
- **Philosophical Grounding in AI**: 针对 **Joscha** 的哲学观点以及哲学家因 A(G)I 发展而提出糟糕观点所引发的尴尬进行了反复讨论；未详细说明具体的尴尬观点。
- **Cosmic Scale World-building**: 成员 **@amiramogus_90887** 讨论了其项目的叙事层级，涉及人类后裔、**transcendental Minds** 以及由 **Brainers** 运行的跨星系模拟，展示了利用 **websim.ai** 构建的宏大世界观概念。
- **Ethical Considerations in Simulations**: 一位成员讨论了创建模拟的伦理影响，建议对这些模拟中可能存在的有意识实体保持同理心，而另一位成员则提议在与 AI 交互时进行相互对齐和共同的元现实（meta-reality）探索。
- **Sharing World Sim Projects & Volunteer Sign-Up**: 几位成员分享了他们的 **world-sim** 相关项目链接，其他人询问如何报名成为志愿者，其中一人分享了他们在 Twitter 上发现的另一个 **world-sim** 项目链接。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://websim.ai/c/5bn1mKjsAhs2NgJnx">FutureEpoch Wiki - 探索人类与宇宙的遥远未来</a>: 未找到描述</li><li><a href="https://websim.ai/c/F8xMqy00m38waO5tJ">量子粒子球观测台</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1235982270985142463)** (1 条消息): 

- **Beta Testers Locked In**: **Pages** 的 Beta 测试人员申请现已关闭，已有足够的参与者。关于 **Pages** 开发的进一步更新将在后续分享。
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1235849900500058132)** (814 条消息🔥🔥🔥): 

- **Perplexity Performance Queries**: 用户报告了 Perplexity AI 响应缓慢的问题，特别是在使用 Claude 3 时，注意到生成答案时存在异常延迟。故障排除包括检查网络连接以及在不同设备和浏览器上进行测试。

- **Opus Use Limits Discussion**: 对话集中在 Claude 3 Opus 模型每天 50 条消息的使用限制上。几位用户表达了沮丧并讨论了替代方案，将 Opus 在创意和编程方面的能力与 GPT-4 Turbo 和 Sonnet 进行了比较。

- **Image Generation Inquiry**: 一位用户寻求关于 Perplexity Pro 上最有效的图像生成模型的建议，引发了关于使用场景和生成图像法律所有权的讨论。

- **Scrutiny of User Limitation Communications**: 社区深入探讨了 Perplexity 关于引入消息限制的沟通，用户审查了从无限消息更改为有限消息的伦理影响，以及这是否可能违反了所宣传的服务。

- **Exploring Writing Styles with AI**: 成员们讨论了利用 Perplexity AI 学习和模仿小说写作风格的潜力，并建议利用 "collections" 功能在不同提示词之间保持一致的写作风格。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1047197230748151888/1047649527299055688/1230472581837230100">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、闲逛，并与你的朋友和社区保持紧密联系。</li><li><a href="https://www.theverge.com/24111326/ai-search-perplexity-copilot-you-google-review">这就是为什么 AI 搜索引擎真的无法取代 Google</a>：搜索引擎不仅仅是一个搜索引擎，而 AI 仍然无法完全跟上步伐。</li><li><a href="https://www.tiktok.com/@dnaturelovers?_t=8m88ov8QuoL&_r=1">未找到标题</a>：未找到描述</li><li><a href="https://news.sky.com/story/china-hacked-ministry-of-defence-sky-news-learns-13130757">Sky News 获悉，中国黑客攻击了英国国防部</a>：国会议员将在周二被告知一起涉及国防部的大规模数据泄露事件，目标是现役人员。</li><li><a href="https://tiktokenizer.vercel.app/">Tiktokenizer</a>：未找到描述</li><li><a href="https://techcrunch.com/2024/04/23/perplexity-is-raising-250m-at-2-point-5-3b-valuation-ai-search-sources-say/?guccounter=1">独家：消息人士称，Perplexity 正在为其 AI 搜索平台以 25-30 亿美元的估值筹集超过 2.5 亿美元资金</a>：AI 搜索引擎初创公司 Perplexity 目前是热门项目。TechCrunch 获悉，该公司目前正在筹集至少 2.5 亿美元。</li><li><a href="https://x.com/_weiping/status/1786511543255126396">来自 Wei Ping (@_weiping) 的推文</a>：介绍 ChatQA-1.5，这一系列模型在 RAG 和对话式 QA 方面超越了 GPT-4-0613 和 Command-R-Plus。ChatQA-1.5 有两个变体：Llama3-ChatQA-1.5-8B，https://huggingface.co/nvidia...</li><li><a href="https://www.trustpilot.com/review/www.perplexity.ai">Perplexity 在 Trustpilot 上的评分为“一般”，分数为 2.9 / 5</a>：你同意 Perplexity 的 TrustScore 吗？今天就发表你的意见，听听 14 位客户已经说了什么。</li><li><a href="https://tenor.com/view/thanos-talking-meme-thanos-talking-meme-thanos-speech-gif-1800590086203910493">灭霸说话 GIF - 灭霸说话迷因 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/dancing-duck-dance-duck-duck-ooontz-dance-gif-10943740227711557279">跳舞的鸭子 GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/baqua-gif-22467620">Baqua GIF - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1235849306272043008)** (43 条消息🔥): 

- **探索 Perplexity 的悠久历史**：一位成员分享了一个深入了解 Perplexity 历史的[链接](https://www.perplexity.ai/search/The-history-of-hfvkvCOtRiGSiKlK8YKd1Q)。
- **检索到 BASIC 语言信息**：几位成员似乎通过分享的搜索（如这个[示例](https://www.perplexity.ai/search/BASIC-programming-language-WB8fDre0Ta.oP96gtQ5k1g)）挖掘了 **BASIC 编程语言** 的起源和细节。
- **AI 的隐藏发现被揭示**：[AI 发现](https://www.perplexity.ai/search/AI-discovers-27000-_7Jf6R7jQkCu41nN3WgqtQ)了 27,000 个未知项目，引发了社区的好奇。
- **福布斯报道 Perplexity**：一位成员分享了福布斯视频中对 Perplexity 功能的介绍，展示了其提供更深层互联网见解的能力。视频可以在[这里](https://www.youtube.com/watch?v=Sct_YUU40m4)找到。
- **创意搜索查询促使 AI 探索**：像[这样](https://www.perplexity.ai/search/How-do-I-_4dQUZbbSTCL_8b66wZnYQ)的链接显示成员们正在使用 Perplexity 探索各种创意咨询。

**提到的链接**：<a href="https://www.youtube.com/watch?v=Sct_YUU40m4">Perplexity 想要帮助你在互联网上找到更好的答案 | 福布斯</a>：Google 搜索或维基百科可能是寻找互联网信息的首选方法。Perplexity 旨在帮助你深入探索并找到简洁的答案...

  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1235939642704920586)** (59 条消息🔥🔥):

- **模型兼容性咨询**：一位成员询问是否需要从 **sonar-medium-online** 切换到 **llama-3-sonar-large-32k-online**。共识是旧模型目前仍可运行，但未来可能需要更新。
- **优化 AI 结果**：一位成员讨论了 AI 模型未返回预期竞品分析结果的问题。在提供不同的 prompt 结构和设置时，模型给出的输出效果更好，但一致性仍然是一个问题。
- **Opus 模型支持澄清**：成员们讨论了 Perplexity 产品中缺乏对 **Opus** 等专有模型的 API 支持。会议澄清，不应指望转售专有模型的访问权限。
- **计费逻辑变更**：一位用户询问了 API credits 计费逻辑可能存在的变化，因为其账户余额似乎不一致。讨论中未提供解决方案。
- **自托管 Telegram Bot**：一位成员征求关于集成 Perplexity API 且代码量最少的 Telegram bot 建议，回复建议创建一个这样的机器人应该不会太难。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://optonal.com`">未找到标题</a>: 未找到描述</li><li><a href="https://"">未找到标题</a>: 未找到描述</li><li><a href="https://optonal.com">OpTonal • 为使用 Slack, HubSpot, Google Meet 的团队提供的 AI 销售 Agent</a>: 未找到描述</li><li><a href="https://aws.amazon.com/solutions/case-studies/perplexity-case-study/">Perplexity 通过 Amazon SageMaker HyperPod 将基础模型训练速度提升 40% | Perplexity 案例研究 | AWS</a>: 未找到描述</li><li><a href="https://sensiseeds.com](https://sensiseeds.com)\n2.">未找到标题</a>: 未找到描述</li><li><a href="https://seed.com](https://seed.com)">未找到标题</a>: 未找到描述</li><li><a href="https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/">不仅仅是 OpenAI 的套壳：Perplexity 转向开源</a>: Perplexity 首席执行官 Aravind Srinivas 是 Larry Page 的忠实粉丝。然而，他认为自己已经找到了一种不仅能与 Google 搜索竞争，还能与 OpenAI 的 GPT 竞争的方法。</li><li><a href="http://www.ghirardelli.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.godiva.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.lindt.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.russellstover.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.hersheys.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.dovechocolate.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.toblerone.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.lamaisonduchocolat.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.pierremarcolini.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.vosgeshautchocolat.com)">未找到标题</a>: 未找到描述</li><li><a href="http://www.teuscher.com)">未找到标题</a>: 未找到描述</li><li><a href="https://www.salesforce.com/">Salesforce：以客户为中心的公司</a>: Salesforce 作为排名第一的 AI CRM，使公司能够通过统一的 Einstein 1 平台与客户建立联系，该平台结合了 CRM、AI、数据和信任。</li><li><a href="https://www.hubspot.com/products/sales">适用于小型到企业级公司的销售软件 | 免费开始使用</a>: 强大的销售软件，帮助您的团队在统一的互联平台上达成更多交易、加深关系并更有效地管理销售漏斗。</li><li><a href="https://www.zoho.com/crm/">Zoho CRM | 备受客户好评的销售 CRM 软件</a>: Zoho CRM 是一款在线销售 CRM 软件，在单一 CRM 平台上管理您的销售、营销和支持。全球超过 1 亿用户信赖！立即注册免费试用。</li><li><a href="https://www.gong.io/">Gong - 收入智能平台</a>: Gong 捕捉客户互动，然后大规模提供洞察，赋能团队基于数据而非主观意见做出决策。</li><li><a href="https://www.exceed.ai/">排名第一的对话式营销和销售平台 - Exceed.ai</a>: 利用对话式 AI 增强潜在客户转化。自动化收入互动，实现大规模参与，并通过电子邮件、聊天、短信进行互动。</li><li><a href="https://salesloft.com/">Salesloft：领先的销售参与平台</a>: 未找到描述</li><li><a href="https://www.yesware.com/">让销售参与变得简单 | Yesware</a>: Yesware 帮助高效销售团队大规模开展有意义的电子邮件外展。如果您需要通过电子邮件外展推动更多收入，但觉得复杂的平台过于繁琐，请尝试 Yesware。</li><li><a href="http://ghirardelli.com)">未找到标题</a>: 未找到描述</li><li><a href="http://hersheys.com)">未找到标题</a>: 未找到描述</li><li><a href="http://russellstover.com)">未找到标题</a>: 未找到描述</li><li><a href="http://lindt.com)">未找到标题</a>: 未找到描述</li><li><a href="http://godiva.com)">未找到标题</a>: 未找到描述</li><li><a href="https://sidecardoughnuts.com/)">Sidecar Doughnuts - 世界上最新鲜的甜甜圈！</a>: 自 2012 年以来提供世界上最新鲜的甜甜圈、招牌混合咖啡和微笑服务 | 加利福尼亚州科斯塔梅萨、圣莫尼卡和德尔马</li><li><a href="https://thepieholela.com/)">The Pie Hole</a>: 您的下一次活动需要新鲜派或 Pie Holes 吗？在线下单，全国免费配送，因为派就是爱。
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1235890438762791002)** (396 条消息🔥🔥):

- **以服务器模式启动 LM Studio**：用户正在探索以无头服务器模式（headless server mode）启动 LM Studio 的方法，询问是否存在命令行选项可以在启动应用时预选模型并激活服务器模式。目前关于使用 `lms`（LM Studio 的 CLI 工具）配合 GUI 版本实现无头操作的讨论正在进行中。

- **排查 LM Studio 中的 VRAM 和 RAM 使用问题**：一位用户提出了关于 LM Studio 的 VRAM 和 RAM 使用情况的疑虑，指出在启用 Flash Attention 并将模型卸载（offloading）到 GPU 时，内存消耗行为异常。该用户被要求分享截图并进一步详细说明预期行为与实际行为，以便协助解决问题。

- **远程访问测试系统上的 VRAM**：一位用户就如何在不通过 RDP 禁用 VRAM 的情况下，远程访问专为测试 LLM 构建的电脑寻求建议。SSH 和通过 CLI 使用 LMS 被建议作为维持 VRAM 访问的有效替代方案。

- **通过 Prompt Engineering 获得更好的 LLM 体验**：关于 Prompt Engineering 益处的讨论强调了其在从语言模型中提取高质量输出方面的重要性。Prompt Engineering 可以显著影响生成内容的质量，目前已被 AI 圈内公认为一项宝贵的技能。

- **在 LM Studio 中探索 Stable Diffusion**：有关于 LM Studio 是否支持 Stable Diffusion 的咨询。官方澄清虽然 Stable Diffusion 模型会出现在平台中，但 LM Studio 并不支持它们，列出的 GGUF 文件是用于 Stable Diffusion 的 C++ 实现。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>：未找到描述</li><li><a href="https://lmstudio.ai/blog/lms">Introducing `lms` - LM Studio's companion cli tool | LM Studio</a>：今天，随 LM Studio 0.2.22 一起，我们发布了 lms 的第一个版本 —— LM Studio 的配套 CLI 工具。</li><li><a href="https://lmstudio.ai/docs/welcome">Welcome | LM Studio</a>：LM Studio 是一款用于在电脑上运行本地 LLM 的桌面应用程序。</li><li><a href="https://tenor.com/view/rick-roll-rick-ashley-never-gonna-give-you-up-gif-22113173">Rick Roll Rick Ashley GIF - Rick Roll Rick Ashley Never Gonna Give You Up - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/openai-community/gpt2-xl">openai-community/gpt2-xl · Hugging Face</a>：未找到描述</li><li><a href="https://forum.cursor.sh/t/unable-to-use-lm-studio-with-override/2637">Unable to use LM Studio with override</a>：覆盖（override）应该可以工作，因为他们将其设计为作为 URL 覆盖与 OpenAI Python 库集成。但我认为他们不支持发送空查询来检查 API Key。这就是为什么...</li><li><a href="https://docs.google.com/document/d/1a75YXCCVJi0OGIc4jkXLTKI6q0N00yCWvBieSJ3PG9s/edit?usp=drivesdk">High Quality Story Writing Type Third Person</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/releases/tag/b2775">Release b2775 · ggerganov/llama.cpp</a>：未找到描述</li><li><a href="https://chatboxai.app/">Chatbox - Your AI Copilot on the Desktop, Free Download</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/lms/blob/main/CONTRIBUTING.md">lms/CONTRIBUTING.md at main · lmstudio-ai/lms</a>：终端里的 LM Studio。通过在 GitHub 上创建账号为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio in your terminal</a>：终端里的 LM Studio。通过在 GitHub 上创建账号为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://github.com/ollama/ollama/issues/4051#issuecomment-2092092698">Enable Flash Attention on GGML/GGUF (feature now merged into llama.cpp) · Issue #4051 · ollama/ollama</a>：Flash Attention 已进入 llama.cpp (ggerganov/llama.cpp#5021)。简而言之，只需将 -fa 标志传递给 llama.cpp 的服务器。我们能否拥有一个 Ollama 服务器环境变量来传递此标志...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1235862971394293770)** (234 条消息🔥🔥): 

- **微调 (Fine-Tuning) 的困境与解决方案**：成员们讨论了微调 Llama 3 和 phi 3 等模型，强调了相关问题并分享了资源，例如 [MacBook 指南](https://huggingface.co/blog/abhishek/phi3-finetune-macbook) 和 [使用转换工具的技巧](https://github.com/ggerganov/llama.cpp/pull/6745#issuecomment-2094964796)。一些人建议寻找 GPU 服务以获得更好的性能，而一位成员提到在 128GB M3 Max MacBook Pro 上成功对 phi-3 进行了 8 小时的微调。

- **ChatQA 模型讨论**：用户分享了使用 ChatQA 1.5 模型的经验，包括在模型连贯性和模板格式化方面遇到的挑战。共识表明，像 CMDR+ 这样更大的模型在复杂性和召回率方面表现更优，特别是在处理《圣经》等主题时。

- **Vision 和 RAG 模型的探索**：人们对用于网页自动化的 Vision 模型截图功能表现出兴趣，提到了 Pix2Struct 和 CLaude。对于阅读和生成文本文件（如 PDF），建议使用 Cohere 的 Command-R，而对于 RAG 应用，推荐使用 ChatQA 而非普通的 Llama 3 Instruct。

- **对 Llama 3 模型输出的担忧**：用户报告了 Llama 3 产生不稳定或无意义输出的问题，例如说俄语、全大写字母喊叫等。有人指出，即使在调整了模板并删除了不需要的 token 前缀后，模型的响应质量仍然难以预测。

- **LLMs 的转换挑战**：围绕将 Llama 模型转换为不同格式的挑战展开了技术讨论。解决方案包括调整命令参数的顺序以及确保正确的文件路径，并分享了关于转换脚本所需 flags 变化的见解。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf">xtuner/llava-llama-3-8b-v1_1-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/abhishek/phi3-finetune-macbook">如何在 MacBook Pro 上微调 phi-3</a>: 未找到描述</li><li><a href="https://huggingface.co/google/codegemma-1.1-7b-it-GGUF">google/codegemma-1.1-7b-it-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mzwing/MiniCPM-V-2-GGUF">mzwing/MiniCPM-V-2-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/DavidAU/D_AU-Tiefighter-Holomax-15B-UNHINGED-V1">DavidAU/D_AU-Tiefighter-Holomax-15B-UNHINGED-V1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mradermacher/D_AU-Tiefighter-Holomax-20B-V1-GGUF">mradermacher/D_AU-Tiefighter-Holomax-20B-V1-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-guard-2/">Meta Llama Guard 2 | 模型卡片与提示词格式</a>: 由于护栏（guardrails）可以同时应用于模型的输入和输出，因此存在两种不同的提示词：一种用于用户输入，另一种用于 Agent 输出。</li><li><a href="https://llama.meta.com/docs/how-to-guides/fine-tuning/">微调 | 操作指南</a>: 全参数微调是一种对预训练模型所有层的所有参数进行微调的方法。</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF">dranger003/c4ai-command-r-plus-iMat.GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/releases/tag/b2791">Release b2791 · ggerganov/llama.cpp</a>: 未找到描述</li><li><a href="https://tenor.com/view/im-out-no-thanks-bugs-bunny-oh-no-not-interested-gif-16824550">我退出，不客气 GIF - 我退出，不客气 Bugs Bunny - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/daleks-exterminate-doctor-who-whovian-gif-10468156">Daleks Exterminate GIF - Daleks Exterminate Doctor Who - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/upgrades-robots-gif-21291099">Upgrades Robots GIF - Upgrades Robots - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/mlabonne/llm-course">GitHub - mlabonne/llm-course: 包含路线图和 Colab 笔记本的入门大语言模型 (LLMs) 课程。</a>: 包含路线图和 Colab 笔记本的入门大语言模型 (LLMs) 课程。 - mlabonne/llm-course</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/2948">教程：如何将 HuggingFace 模型转换为 GGUF 格式 · ggerganov/llama.cpp · Discussion #2948</a>: 来源：https://www.substratus.ai/blog/converting-hf-model-gguf-model/ 我在我们的博客上发布了这篇文章，但认为这里的其他人也可能受益，因此也在 GitHub 上分享了原始博客。希望它...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6745#issuecomment-2094964796">支持 Llama 3 转换，由 pcuenca 提交 · Pull Request #6745 · ggerganov/llama.cpp</a>: 分词器（tokenizer）是 BPE。</li><li><a href="https://gist.github.com/wassname/42aba7168bb83e278fcfea87e70fa3af">baukit_orth_act_steering.ipynb</a>: GitHub Gist: 即时分享代码、笔记和片段。</li><li><a href="https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction#LEz9uRJ89vmtYkvqT">LLMs 中的拒绝是由单一方向介导的 — LessWrong</a>: 这项工作是 Neel Nanda 在 ML Alignment & Theory Scholars Program - 2023-24 冬季班项目中的一部分，由……共同指导。</li><li><a href="https://huggingface.co/hjhj3168/Llama-3-8b-Orthogonalized-exl2/discussions">hjhj3168/Llama-3-8b-Orthogonalized-exl2 · Discussions</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1236222899044614187)** (8 条消息🔥): 

- **命令行困惑已解决**：一位成员在使用 Python OpenAI API 打印消息时遇到了包含系统提示词的问题，这似乎与尝试使用 **LMS CLI 工具**有关。另一位成员建议从 [lmstudio.ai](https://lmstudio.ai) 重新下载 v0.2.22 版本，因为该问题已在此版本中修复。

- **所有系统运行正常**：在重新下载推荐版本后，该成员确认 GUI 运行正常，并计划测试 CLI 以查看是否存在潜在的重复问题。

- **版本讨论中的初始化错误**：一位成员询问关于初始化 **phi-3** 时遇到的错误，另一位成员指示其升级到更新的版本，特别是 **0.2.22**，可以从 [lmstudio.ai](https://lmstudio.ai) 下载。

**提及链接**: <a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs

---

**LM Studio ▷ #[📝-提示词讨论聊天](https://discord.com/channels/1110598183144399058/1120489168687087708/1236266063952347156)** (8 条消息🔥): 

- **寻求个性化写作助手**：一位成员讨论了如何优化写作模型以模仿个人写作风格，询问 Prompt Engineering 或交互式技术是否能提升效果。另一位参与者建议使用 **autotrain** 等工具对 "llama 2/3" 或 "Mistral" 等现有模型进行 **finetuning**，以便更好地采用个人风格。

- **AI 的限定范围文档访问**：一位成员询问了在语言模型上下文中为特定文档段落提供“临时限定范围访问”的方法。建议将**文档部分的针对性包含**在 Prompt 中作为该需求的实际变通方案。

- **澄清 AI 记忆限制**：随后，他们询问了在 LM Studio 中编辑或删除 Prompt 的部分内容后上下文的持久性，怀疑被删除的内容仍被意外保留。结论是，如果语言模型似乎记住了已删除的上下文，那可能是由于 **bug 或错误**，因为模型不应保留已移除的信息。

---

**LM Studio ▷ #[⚙-配置讨论](https://discord.com/channels/1110598183144399058/1136793122941190258/1236015180312477806)** (56 条消息🔥🔥): 

- **WSL 问题与代理解决方案**：成员们讨论了从 WSL 连接到 LM Studio 的问题，建议使用在 `ipconfig` 中找到的 Windows WSL vEthernet 适配器 IP 可能是一个解决方案。一些人指出可能需要 [reverse proxy](https://docs.microsoft.com/en-us/windows-server/administration/reverse-proxy)，一位成员提供了一个 PowerShell **netsh** 技巧：`netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=0.0.0.0 connectport=$PORT connectaddress=127.0.0.1`。

- **在 D&D 战役中发挥创意**：
    - 一位成员希望使用 LM Studio 驱动带有 AI 队友的单人 D&D 战役，询问如何轻松地将个人小说和游戏书籍库注入模型以进行上下文游戏。
    - 虽然有人提出了考虑使用 *command-r-plus* 等模型的有益建议，但后续消息显示需要一个能够记住角色卡并有效调整游戏叙事的 AI 地城主（Dungeon Master），这强调了当前的局限性以及未来进步的前景。

- **寻求 AI 地城主**：出于对 AI 处理《龙与地下城》（Dungeons & Dragons）游戏会话的渴望，成员们分享了愿景以及使用 *AnythingLLM* 和 *SillyTavern* 等平台的持续尝试，展示了在持续进化的 AI 驱动冒险中涵盖故事、规则和氛围功能的目标。

- **对 AI 角色扮演边界的担忧**：一位成员讨论了在尝试使用 *ChatGPT* 体验更黑暗、无限制的桌面角色扮演游戏叙事时遇到的困难，遇到了 AI 的政策违规限制，这表明了目前 AI 系统内的内容审查局限性。

- **释放 AI 在游戏中的潜力**：对话转向了 AI 在游戏领域的未来潜力，讨论了 AI 生成图像、动态背景音乐和角色语音区分等功能，这些功能将把沉浸式游戏体验提升到新高度。

**提及链接**: <a href="https://www.udio.com/">Udio | AI 音乐生成器 - 官方网站</a>：发现、创作并与世界分享音乐。使用最新技术在几秒钟内创作 AI 音乐。

---

**LM Studio ▷ #[🎛-硬件讨论](https://discord.com/channels/1110598183144399058/1153759714082033735/1236326311207895103)** (123 条消息🔥🔥): 

<ul>
  <li><strong>用于 AI 部署的 GPU 选择</strong>：成员们讨论了使用旧显卡执行 AI 任务的可行性。有人提到像 GRID K1 这样的显卡可能太旧且不受当前支持，建议将 Tesla P40 作为最老旧但可行的选择。用户建议虽然 P40 以其价格提供了大量 VRAM，但散热和供电可能比较棘手，并且在运行 Stable Diffusion 等任务时可能无法提供最佳性能。</li>

  <li><strong>构建以 AI 为中心的硬件配置</strong>：对话围绕构建高效的 AI 家庭实验室展开，分享了一个 PNY GeForce RTX 4070 VERTO Dual Fan 12GB GDDR6X 显卡的 eBay 链接，作为目前 3060 GPU 的潜在升级方案以满足个人游戏需求。建议在游戏和 LLMs 方面，12GB 是 VRAM 的最低要求，更倾向于 16GB 或 24GB 的型号。</li>

<ul>
  <li><strong>服务器硬件采购</strong>：用户分享了购买二手服务器的经验，提到了 ASUS ESC 4000 G3 服务器等特定型号，该服务器可容纳多个 GPU（如 P40），且价格合理，包含大量 RAM。用户还表达了对硬件兼容性以及可能需要升级以支持 AVX2 的担忧。</li>

  <li><strong>多 GPU 与推理速度</strong>：讨论涉及了 P40 的推理速度，并将其与 Mac 的性能进行了对比，承认虽然多个 GPU 有助于将大型模型完全托管在 VRAM 中，但在特定任务中，其速度可能不会显著超过高性能的单 GPU。</li>

  <li><strong>多 GPU 配置的主板注意事项</strong>：成员们交流了最适合搭载 Tesla P40 等多个 GPU 的主板类型，并讨论了由于驱动不兼容，将数据中心 GPU 与消费级 GPU 混合运行可能出现的问题。共识似乎是，虽然运行多个 GPU 具有成本效益，但也可能面临带宽瓶颈、电源限制和散热挑战等复杂问题。</li>
</ul>
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.amazon.com/MSI-GeForce-Tri-Frozr-Architecture-Graphics/dp/B095L2PTLT/ref=sr_1_1?c=ts&dib=eyJ2IjoiMSJ9.EXsKtTGwddxDdfXJCqbDCPadjBIuEsxDCxjFfqKKaYdlNI1HHU6xGQJuSaQZda6j4aw-qC1apJp1WpFcRRxpf_LbHv4WeNRpGy7BS5OhZFzDL1Omhb8_auWnr4bE0j_GZe_M1G8kCBSgcxd_LL0Hi4cC3PP96_dZOFIqEtVoHKJ_kcTsHa8wUbe4p3ZgnmNiSEtl-3m53NTQSfvAMSE1fUsjvFrXtF3oeWla9ilph0AOsjCxEm2KT9nLQ-O1SNiNOT6C-MtSDyBTIeB99fuwXw.wg3-X6VJcsFkDpoETDbYKvJmcwkViq5nN8SKlViEaOA&dib_tag=se&keywords=Computer+Graphics+Cards&qid=1714950982&refinements=p_n_feature_twenty_browse-bin%3A23572110011%2Cp_36%3A1253507011%2Cp_n_feature_four_browse-bin%3A79630736011&rnid=79630706011&s=pc&sr=1-1&ts_id=284822)">未找到标题</a>：未找到描述</li><li><a href="https://endoflife.date/nvidia-gpu">NVIDIA GPUs</a>：检查 NVIDIA GPU 的生命周期终止、发布政策和支持计划。</li><li><a href="https://www.ebay.com/itm/386939573137?epid=5060641239&itmmeta=01HX59VYQJ2Y1RGNR28XH6RARM&hash=item5a17655391:g:POAAAOSwa15kusfX&itmprp=enc%3AAQAJAAAA4LJw7CDsQRPj%2BT86XiAmxa7LCEA%2Bs66Gdh5OrNvT%2FvTno%2Fa5U3Tul660r9O0Nazl2HLVEmleeFUotntyVk8Tm7K4M57SPVcYPin6XCI0%2BwXBfu0UrMjbUBzL7TamlRRLKVVg3o6FKMKPWJcv4Ro2dt56dpDm0axhE%2FE7Qk0E238i6RkgFGcC9PE34oTnXYYngi24RreVIovqgXOX%2F5ja8cTHLhf6OsSrfymcAnXi%2FrRppjmn4MSBtt0S8f9zbyGUjSpSvb%2BGkv5YckCxsKHm%2FY3XcqlV%2BBWMLl7gUkActc8V%7Ctkp%3ABk9SR_Lr76npYw)">PNY GeForce RTX 4070 VERTO Dual Fan 12GB GDDR6X Graphics Card 751492775005 | eBay</a>：未找到描述</li><li><a href="https://www.ebay.com/itm/386939573137?epid=5060641239&itmmeta=01HX59VYQJ2Y1RGNR28XH6RARM">PNY GeForce RTX 4070 VERTO Dual Fan 12GB GDDR6X Graphics Card 751492775005 | eBay</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1236376055393292469)** (1 条消息): 

- **LM Studio API 语音限制**：一位成员报告称其 LM Studio API 在停止前最多只能说两个单词。他们正在寻求专家的技术见解，以了解为何会出现此问题。
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/)** (1 条消息): 

drjflamez: Secrets don't make friends (秘密交不到朋友)
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1235873142648606741)** (28 条消息🔥): 

- **更新警报：ROCm 下载已就绪**：提到了 ROCm 技术预览版的更新；修复程序可在 [lmstudio.ai/rocm](https://lmstudio.ai/rocm) 获取，解决了之前报告的 Embedding 模型问题。

- **最大 Token 截断说明**：一位成员询问当发送大于报告的 512 Token 最大上下文的序列进行 Embedding 时会发生什么，并指出他们成功嵌入了 1000+ Token 而没有出现问题。

- **新硬件上的卓越性能**：一位用户报告在 RX 7900 xt 上成功部署了 **NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF**，使用 16 FP 达到每秒 34 个 Tensor，且完美适配 VRAM。

- **赞扬 ROCm 的流畅表现**：一位社区成员对 ROCm 的稳定性和有效性表示满意，并好奇为什么从 0.2.18 版本开始表现优异，却仍被标记为预览版/测试版。

- **社区驱动的 Linux 构建兴趣**：关于潜在 Linux ROCm 构建的讨论浮出水面，用户分享了个人变通方案，并表示如果代码库开源，他们渴望为其做出贡献。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/rocm,">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://tenor.com/view/oil-gif-21418714">Oil GIF - Oil - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1236452526484750347)** (1 条消息): 

- **CodeGemma 1.1 加入阵容**：`lmstudio-community` 仓库已更新 **CodeGemma 1.1**。人们对其性能提升抱有很高期待，类似于从 **Gemma 1.0 到 Gemma 1.1** 的升级，尽管具体细节仍然较少。[尝试 CodeGemma 1.1](https://huggingface.co/lmstudio-community/codegemma-1.1-7b-it-GGUF)

- **Nvidia 发布 ChatQA 1.5 模型**：Nvidia 发布了两个版本的 **ChatQA 1.5**，尺寸分别为 **8B** 和 **70B**。专为 RAG 和基于上下文的问答设计，它们可能不适合作为通用聊天机器人，但非常适合处理上下文相关的查询。[尝试 ChatQA 1.5 - 8B](https://huggingface.co/lmstudio-community/Llama3-ChatQA-1.5-8B-GGUF), [尝试 ChatQA 1.5 - 70B](https://huggingface.co/lmstudio-community/Llama3-ChatQA-1.5-70B-GGUF)
  

---


**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1236035899880636468)** (53 条消息🔥): 

- **沙箱解决方案**：用户讨论了在遇到提示沙箱问题的错误后，通过使用 `--no-sandbox` 标志修复应用在与终端交互时退出的问题。
- **LM Studio.js 服务器激活建议**：提供了关于使用 `lms server start` 命令启动 LM Studio 服务器，并使用 HTTP 监听器等待服务器激活的指导。
- **LM Studio 进入无头模式**：[yagilb](https://discord.com/channels/1110598183144399058/1234988891153629205/1235668310243151963) 解释说，新的 LM Studio v0.2.22 和 lms CLI 允许 LM Studio 以无头（headless）模式运行，并计划在未来进一步简化该过程。
- **欢迎为 CLI 贡献**：LM Studio 的 [CLI 是开源的](https://github.com/lmstudio-ai/lms)，鼓励社区为其开发做出贡献。
- **对流线型体验的期待**：一位用户表达了希望在 Linux 服务器上运行 LLM 时能有易于使用的无头设置，yagilb 回应称 CLI 已经实现了这一点，并将进一步改进。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://localhost:${port}`,">未找到标题</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/local-server">本地 LLM 服务器 | LM Studio</a>：你可以通过运行在 localhost 的 API 服务器使用在 LM Studio 中加载的 LLM。</li><li><a href="https://tenor.com/view/qawe-asd-gif-26050335">Qawe Asd GIF - Qawe Asd - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: 终端中的 LM Studio</a>：终端中的 LM Studio。通过在 GitHub 上创建账号为 lmstudio-ai/lms 的开发做出贡献。
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1236392835553693878)** (15 条消息🔥): 

- **BackPACK：PyTorch 用户的新工具**：[BackPACK 库](https://backpack.pt/) 可以从 [PyTorch](https://pytorch.org/) 的反向传播（backward pass）中提取更多信息。它包含一份出版物引用：Dangel, F., Kunstner, F., & Hennig, P. (2020) 题为 *[BackPACK: Packing more into Backprop](https://openreview.net/forum?id=BJlrF24twB)*。
  
- **CUDA NCCL 讲座**：由于 Discord 的问题，今天的 CUDA NCCL 会议移至 [Google Meet](https://meet.google.com/xtg-ihck-fmx)。

- **Google Meet 最佳实践**：一位成员分享了管理 Google Meet 会议的技巧，例如策划演讲、让参与者举手提问、管理聊天查询、处理机器人，以及鼓励使用摄像头以获得互动式的演讲体验。

- **增强互动式讲座**：鼓励参与者在演讲期间保持互动并开启摄像头，这比单纯看录像更具参与感。

- **Citadel 的盈利策略揭秘**：一位成员分享了一篇 [arXiv 论文](https://arxiv.org/abs/1804.06826)，解释了 Citadel 成功的财务策略。

- **即将发布的 CUDA NCCL 录像**：一位成员询问 NCCL 会议是否会上传到 YouTube，另一位成员回答说“很快（soon TM）”。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/2wlearning/status/1786638674538754189">来自 2wl (@2wlearning) 的推文</a>：啊，现在我明白为什么 Citadel 赚这么多钱了 https://arxiv.org/abs/1804.06826</li><li><a href="https://backpack.pt/">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1236790278182932571)** (15 messages🔥): 

- **宣布融合 DoRA Kernel**：宣布了一个新的融合 **DoRA 层实现**，它显著减少了独立 kernel 的数量，特别是通过为层权重形状定制 GEMM kernel，并将 reduction 操作直接融合到 kernel 的 epilogue 中。详细信息、基准测试和用法可以在 [Fused DoRA kernels GitHub pull request](https://github.com/pytorch/ao/pull/216) 中找到。
  
- **DoRA 的潜在优化**：针对该公告，有人建议可以在推理时将 DoRA 权重预处理为等效于 LoRA，以潜在地减少所需的计算量，尽管这不适用于训练场景。

- **为 DoRA 定制的 Autotuner**：新的 DoRA kernel 实现了一个经过调整的 autotuner 用于调试，其中包括更好的日志功能，尽管大家承认 Triton 更新后的 autotuner 中现在可能也存在类似的功能，并正在考虑与 Triton 内置的 autotuner 保持一致。

- **期待深入的基准测试**：成员们表示有兴趣看到比较 DoRA 层内计算成本和数据移动的基准测试，特别是关注新的融合 GEMM kernel 的表现，并包含了参考实现以便进一步进行 profiling。

- **ONNX 中的 Triton Kernel**：有人发布了关于在 ONNX Runtime 中将 Triton kernel 作为自定义算子使用的求助请求，因为现有的文档被认为有些有限且过时。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/217/files">Fix the URLs of web pages by Jokeren · Pull Request #217 · pytorch/ao</a>: 未找到描述</li><li><a href="https://github.com/pytorch/ao/pull/216">Fused DoRA kernels by jeromeku · Pull Request #216 · pytorch/ao</a>: Fused DoRA Kernels。融合 DoRA 层实现，将独立 kernel 的数量从 ~10 个减少到 5 个。内容包括背景、优化、主要贡献、用法、测试、基准测试、Profiling、下一步工作...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1235860230395400202)** (64 messages🔥🔥): 

- **安装自定义 PyTorch/CUDA 扩展**：一位成员询问了在 `setup.py` 文件中安装自定义 PyTorch/CUDA 扩展的更简洁方法。他们提到了使用命令行时的日志记录和系统兼容性问题。讨论引用了三个 GitHub pull requests 以及来自 PyTorch/AO 仓库的 `setup.py` 特定部分作为示例：[PR#135](https://github.com/pytorch/ao/pull/135)、[PR#186](https://github.com/pytorch/ao/pull/186)、[PR#176](https://github.com/pytorch/ao/pull/176) 和 [pytorch/ao setup.py 示例](https://github.com/pytorch/ao/blob/0ba0006eb704dea33becec82b3f34512fe8a6dff/setup.py#L35-L78)。

- **TorchServe GPU 配置说明**：一位成员需要澄清演示中提到的性能设置，特别是关于 `torch.set_num_threads` 的部分。分享了一篇 [博客文章](https://pytorch.org/tutorials/intermediate/torchserve_with_ipex.html) 以获取有关 `torch.set_num_threads` 的详细信息。进一步的说明指出，文档中关于较大 batch size 会导致更高延迟的描述有误，并讨论了如何调整 worker 数量以优化吞吐量和延迟。

- **CUDA 中的原子操作**：关于一段使用 `reinterpret_cast` 的 CUDA 代码片段是否具有原子性的讨论。确认该代码确实以原子方式执行，但根据 C++ 标准属于未定义行为。正确的、符合标准的做法应该使用 `std::bit_cast`。

- **Numba-CUDA 与 CUDA-C 的性能对比**：一项关于比较 numba-CUDA 和 CUDA-C 性能的查询显示，numba-CUDA 版本运行较慢。通过分享性能分析文件并检查 pTX 文件，发现 numba 版本包含可能减慢执行速度的内存安全检查。

- **对 CUTLASS 和 Stream-K 调度技术的兴趣**：一位成员表示有兴趣在未来讨论或讲座中探讨 CUTLASS 中用于 GEMM 的 stream-K 调度技术。虽然对该建议持开放态度，但有人指出 stream-K 可以作为另一个演讲中的一个简短小节，因为解释 CUTLASS 2.0 API 可能会非常冗长。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/1804.06826">Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking</a>: 每年都会推出新的 NVIDIA GPU 设计。这种快速的架构和技术进步，加上制造商不愿透露底层细节，使得...</li><li><a href="https://pytorch.org/tutorials/intermediate/torchserve_with_ipex.html">Grokking PyTorch Intel CPU performance from first principles — PyTorch Tutorials 2.3.0+cu121 documentation</a>: 未找到描述</li><li><a href="https://github.com/pytorch/serve/blob/master/docs/performance_guide.md#torchserve-on-gpu">serve/docs/performance_guide.md at master · pytorch/serve</a>: 在生产环境中提供、优化和扩展 PyTorch 模型 - pytorch/serve</li><li><a href="https://github.com/pytorch/serve/blob/master/docs/performance_guide.md#torchserve-on-cpu-">serve/docs/performance_guide.md at master · pytorch/serve</a>: 在生产环境中提供、优化和扩展 PyTorch 模型 - pytorch/serve</li><li><a href="https://github.com/pytorch/serve/blob/master/frontend/server/src/main/java/org/pytorch/serve/util/ConfigManager.java#L80">serve/frontend/server/src/main/java/org/pytorch/serve/util/ConfigManager.java at master · pytorch/serve</a>: 在生产环境中提供、优化和扩展 PyTorch 模型 - pytorch/serve</li><li><a href="https://github.com/pytorch/serve/blob/master/docs/configuration.md">serve/docs/configuration.md at master · pytorch/serve</a>: 在生产环境中提供、优化和扩展 PyTorch 模型 - pytorch/serve</li><li><a href="https://github.com/eureka-research/DrEureka">GitHub - eureka-research/DrEureka</a>: 通过在 GitHub 上创建账户来为 eureka-research/DrEureka 的开发做出贡献。</li><li><a href="https://pytorch.org/serve/configuration.html">Advanced configuration &mdash; PyTorch/Serve master documentation</a>: 未找到描述</li><li><a href="https://github.com/mobiusml/hqq/blob/master/setup.py#L11-L15">hqq/setup.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch/ao/blob/0ba0006eb704dea33becec82b3f34512fe8a6dff/setup.py#L35-L78">ao/setup.py at 0ba0006eb704dea33becec82b3f34512fe8a6dff · pytorch/ao</a>: 用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao</li><li><a href="https://youtu.be/HkyWFIbs4JY?t=558)">Lightning Talk: The Fastest Path to Production: PyTorch Inference in Python - Mark Saroufim, Meta</a>: 闪电演讲：通往生产的最快路径：Python 中的 PyTorch 推理 - Mark Saroufim, Meta。从历史上看，为了进行推理，用户不得不重写他们的...</li><li><a href="https://aws.amazon.com/ec2/instance-types/">Compute – Amazon EC2 Instance Types – AWS</a>: 未找到描述</li><li><a href="https://github.com/pytorch/ao/pull/135">Custom CUDA extensions by msaroufim · Pull Request #135 · pytorch/ao</a>: 这是 #130 的可合并版本 - 我必须进行一些更新：添加除非使用 PyTorch 2.4+ 否则跳过的测试，以及如果 CUDA 不可用则跳过的测试；在开发依赖中添加 ninja；本地...</li><li><a href="https://github.com/pytorch/ao/pull/186">louder warning + docs for custom cuda extensions by msaroufim · Pull Request #186 · pytorch/ao</a>: 未找到描述</li><li><a href="https://github.com/pytorch/ao/pull/176">Add A10G support in CI by msaroufim · Pull Request #176 · pytorch/ao</a>: 支持 A10G + manylinux，以便 CUDA 扩展可以在尽可能多的系统上运行
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1235928158063169706)** (19 messages🔥): 

- **调试符号问题 (Debug Debugging Symbols)**: 一位参与者在使用旨在构建带有 **debug symbols** 的特定文件的脚本时遇到困难，该脚本对他们来说效果不佳。他们提到所有内容都过于混乱，无法进行适当的调试，并正在寻求另一种构建调试符号的方法，因为文档缺乏细节。

- **PyTorch 中的约束限制**: 一位成员讨论了 PyTorch 2.2 和 2.3 版本中由 `torch._dynamo.mark_dynamic(inputs, index=1)` 引起的不一致的 `ConstraintViolationError` 问题。他们发布了错误消息，并指出编译器似乎在多个 batch 的动态形状上存在分歧。

- **呼吁提交 GitHub Issue**: 一位成员建议针对前面提到的 PyTorch 约束问题创建一个 **GitHub issue**，并指出需要特定专家的见解。

- **Answer.AI 发布开源系统**: 一位成员提到了 **Answer.AI 的新开源系统**，该系统允许在带有游戏 GPU 的台式机上训练 70B 参数的语言模型。他们提供了一个 GitHub 链接，并分享了关于在不导致 out-of-memory（显存溢出）的情况下最快设置的问题。

- **模型训练显存洞察**：另一场对话中，成员们讨论了在不同配置以及不同版本的 PyTorch 和 Transformers 下，**LLaMa 2 70B 模型训练**的显存占用情况。报告的 8.6GB 峰值显存出乎意料，此外还分享了使用近 24GB 显存的微调命令。

- **PyTorch 的全方位追踪分析 (HTA)**：一位参与者介绍了 **HTA**（Holistic Trace Analysis）工具，并提供了文档链接。HTA 旨在通过分析 PyTorch Profiler 的追踪记录来协助识别性能瓶颈。

- **`torch.compile` 的特化错误 (Specialization Errors)**：针对早前的一个约束错误，一位成员解释说，该问题是由于代码强制对预期为动态的维度进行特化（specialization）导致的，并建议通过增加日志记录来诊断问题。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html">Answer.AI - You can now train a 70b language model at home</a>：我们正在发布一个基于 FSDP 和 QLoRA 的开源系统，可以在两个 24GB GPU 上训练 70b 模型。</li><li><a href="https://www.answer.ai">Answer.AI - Answer.AI - Practical AI R&amp;D</a>：实用 AI 研发</li><li><a href="https://hta.readthedocs.io/en/latest/">Holistic Trace Analysis &mdash; Holistic Trace Analysis 0.2.0 documentation</a>：未找到描述</li><li><a href="https://github.com/AnswerDotAI/fsdp_qlora#finetune-llama-2-70b-on-dual-24gb-gpus">GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP</a>：使用 QLoRA + FSDP 训练 LLM。通过在 GitHub 上创建账户为 AnswerDotAI/fsdp_qlora 的开发做出贡献。</li><li><a href="https://github.com/AnswerDotAI/fsdp_qlora">GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP</a>：使用 QLoRA + FSDP 训练 LLM。通过在 GitHub 上创建账户为 AnswerDotAI/fsdp_qlora 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1236390606222266388)** (1 messages): 

- **GPU 集体通信速成课程**：CUDA MODE Discord 频道即将举行一场关于使用 **NCCL** 进行 GPU 集体通信的会议。一位兴奋的成员期待学习 PMPP 书籍中未涵盖的分布式 ML 概念。
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1236436649433632788)** (5 messages): 

- **适合 ML 系统新手的论文列表**：Marksaroufim 分享了一个 [ML 系统入门列表](https://github.com/cuda-mode/awesomeMLSys) 的 GitHub 链接，其中包含对机器学习系统新手有帮助的论文。
- **量化学习资源**：Mr.osophy 分享了一个 [YouTube 视频](https://youtu.be/0VdNflU08yA?feature=shared)，解释了**量化**及其在 PyTorch 中的实现，这对于有兴趣学习该主题的人来说是宝贵的资源。
- **动态内存压缩 (DMC) 提升 LLM 性能**：Andreaskoepf 提到了一种名为动态内存压缩 (DMC) 的新技术，该技术在 H100 GPU 上可将 Llama 模型的吞吐量提高多达 370%。他们分享了[原始推文](https://x.com/p_nawrot/status/1768645461689168365)，其中还包含[研究论文](https://arxiv.org/abs/2403.09636)的链接。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/p_nawrot/status/1768645461689168365">Piotr Nawrot (@p_nawrot) 的推文</a>：Transformer 中的内存在推理时随序列长度线性增长。在 SSM 中它是常数，但通常以性能为代价。我们引入了动态内存压缩 (DMC)...</li><li><a href="https://github.com/cuda-mode/awesomemlsys">GitHub - cuda-mode/awesomeMLSys: An ML Systems Onboarding list</a>：一个 ML 系统入门列表。通过在 GitHub 上创建账户为 cuda-mode/awesomeMLSys 的开发做出贡献。</li><li><a href="https://youtu.be/0VdNflU08yA?feature=shared">使用 PyTorch 解释量化 - 训练后量化、量化感知训练</a>：在此视频中，我将介绍并解释量化：我们将首先简要介绍整数和浮点数的数值表示...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1236522597903237170)** (9 messages🔥):

- **CUDA MODE Discord 语音频道风波**：由于语音频道被滥用发布不当内容，多名用户被误封；管理员已表示歉意并开始恢复受影响的用户，包括 **@wilson**、**@c_cholesky**、**@jeffjeff** 和 **@harryone1**。
- **GPU 时钟频率困惑澄清**：针对 **H100 GPU** 时钟频率出现了一个初学者问题，特别是关于每秒操作数的计算和理论峰值性能。另一位用户指出这可能是一个单位错误，建议应为 **1.8 GHz**，而非 1.8 MHz。
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1236967183829962802)** (4 messages): 

- **矩阵转置难题**：一位成员质疑在每个元素仅被访问一次的情况下，矩阵转置中分块 (tiling) 的必要性。回答指出这是为了实现**合并内存写入 (coalesced memory writes)**，并提供了一篇[关于 CUDA 矩阵转置的澄清博客文章](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)。
- **关于合并 (Coalescing) 的预习**：该成员感谢对合并 (coalescing) 的澄清，并提到该主题在下一章才会被讲解，这导致了他们最初的困惑。
- **话题顺序可能导致困惑**：作为回应，有人指出书中的问题有时会出现在相关主题讲解之前，这可能会让读者感到困惑。

**提到的链接**：<a href="https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/">An Efficient Matrix Transpose in CUDA C/C++ | NVIDIA Technical Blog</a>：我上一篇 CUDA C++ 文章介绍了使用 Shared Memory 的机制，包括静态和动态分配。在本篇中，我将展示使用 Shared Memory 可以实现的一些性能提升。

  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1236329263892926557)** (6 messages): 

- **感谢支持**：一位成员对大家在处理高优先级工作导致频道内容更新延迟时的支持和理解表示感谢。
- **对 PyTorch Profiling 的认可**：成员们对 **nsys** 感到兴奋，并有兴趣尝试“轻量级”的 **PyTorch profiling 工具**。该成员受到一段录音的启发，并询问了活动结束后 Discord 中可能出现的突出问题。
- **对源码注解 (Source Annotation) 的赞赏**：成员提到 Taylor 即将推出的源码注解工具“非常酷”，让人联想到 Apple 的 Metal profiler 界面，可进行逐行着色器分析 (line-by-line shader profiling)。他们链接了 Apple 的开发者文档：[Optimize shaders with per-line shader profiling statistics](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance#Optimize-shaders-with-per-line-shader-profiling-statistics)。
- **强调分析器功能**：文中重点介绍了一个能够在分析追踪 (profiled trace) 上进行编辑并获得近乎实时估算的分析器功能。它涉及 Instruments 利用架构知识来“重新运行”执行，可能基于采样 (sampling) 技术。

**提到的链接**：<a href="https://developer.apple.com/documentation/xcode/optimizing-gpu-performance#Optimize-shaders-with-per-line-shader-profiling-statistics">Optimizing GPU performance | Apple Developer Documentation</a>：使用 Metal 调试器查找并解决性能瓶颈。

  

---


**CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1236409651550093463)** (1 messages): 

- **探索 JAX 多进程模型**：一位成员分享了他们对 JAX **分布式设置能力**的赞赏，特别是在 GPU 集群和 [Cloud TPU pods](https://cloud.google.com/tpu) 等环境下。他们引用了 [JAX 多进程文档](https://jax.readthedocs.io/en/latest/multi_process.html)，该文档提供了启动 JAX 进程和运行多进程计算的详细指南。

**提到的链接**：<a href="https://jax.readthedocs.io/en/latest/multi_process.html">Using JAX in multi-host and multi-process environments &#8212; JAX  documentation</a>：未找到描述。

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1235936759435628575)** (12 messages🔥): 

- **动漫爱好分享**：成员们谈论了各自的动漫喜好；一位成员从小看《火影忍者 (**Naruto**)》长大，喜欢《一拳超人 (**One Punch Man** )》和《剑风传奇 (**Berserk** )》，并认为《咒术回战 (**JJK** )》拥有顶级的动画和战斗场面。另一位成员在某个场景的蓝光版发布后，幽默地表达了对《咒术回战》中宿傩 (Sukuna) 角色的钦佩。

- **iPhone 和 Mac 作为临时音视频方案**：一位成员建议使用 [iPhone & Mac](https://a.co/d/7uxdnek) 来获得更好的通话音视频质量，并指出当两台设备都更新并登录同一个 Apple ID 时，它们会自动集成。在 Photo Booth, Discord, Google Meet 和 Streamlabs 等各种平台上，都可以选择 iPhone 作为摄像头/麦克风输入。

- **对 Discord 到 Google Calendar 自动化的兴趣**：一位成员询问如何设置自动化，将 Discord 活动同步到 Google Calendar，以避免错过读书小组。虽然目前还没有提到现有的解决方案，但如果需求显著，大家对设置该功能持开放态度。
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1236014784009474089)** (4 messages): 

- **GreenBitAI 推出 LLM 工具包**：一位成员重点介绍了 [GreenBitAI 的 green-bit-llm](https://github.com/GreenBitAI/green-bit-llm)，这是一个用于微调、推理和评估 GreenBitAI 语言模型的工具包，其范围比之前讨论的专门针对矩阵乘法操作的 bitblas 更广。
- **使用 BitBlas 进行快速推理**：据一位成员称，BitBlas 拥有针对 2-bit 操作优化的快速 gemv kernel，有助于加速推理任务，但他们尚未亲自测试。
- **GreenBitAI 的二进制矩阵乘法**：成员们对 [GreenBitAI 的 cutlass kernels](https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp) 表现出浓厚兴趣，特别是其在 bitorch-engine 中实现的二进制矩阵乘法。
- **权重中计算梯度**：另一位成员指出了 GreenBitAI 工具包的一个有趣属性；如 [bitorch-engine 的代码片段](https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py#L81) 所示，它会计算权重的梯度，由于梯度在训练期间没有被打包（packed），这引发了关于潜在 VRAM 占用的好奇。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/GreenBitAI/green-bit-llm">GitHub - GreenBitAI/green-bit-llm: 用于微调、推理和评估 GreenBitAI LLM 的工具包。</a>：一个用于微调、推理和评估 GreenBitAI LLM 的工具包。 - GreenBitAI/green-bit-llm</li><li><a href="https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp">bitorch-engine/bitorch_engine/layers/qlinear/binary/cutlass/binary_linear_cutlass.cpp at main · GreenBitAI/bitorch-engine</a>：该工具包通过为低比特量化神经网络提供专门函数来增强 PyTorch。 - GreenBitAI/bitorch-engine</li><li><a href="https://github.com/GreenBitAI/bitorch-engine/blob/main/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py#L81C9-L81C20">bitorch-engine/bitorch_engine/layers/qlinear/nbit/cutlass/q4_layer.py at main · GreenBitAI/bitorch-engine</a>：该工具包通过为低比特量化神经网络提供专门函数来增强 PyTorch。 - GreenBitAI/bitorch-engine
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1235855152624173120)** (630 messages🔥🔥🔥):

- **CUDA 编译问题**：像 `nvcc 11.5` 这样的编译器在旧型号 GPU 上进行 bfloat16 运算时会报错；`__ldcs` 和 `__stcd` 等函数未定义，且 `__bfloat1622float2` 等操作会导致问题。目前已提出一个 [修复方案](https://github.com/karpathy/llm.c/pull/353)，通过手动处理 bfloat16 算术运算来支持旧显卡和工具包。
- **多 GPU 训练挂起**：正如 [Issue #369](https://github.com/karpathy/llm.c/issues/369) 所报告的，最近对 master 分支的提交导致多 GPU 训练挂起。一个 [独立的开发分支](https://github.com/PeterZhizhin/llm.c/branch/nccl) 维持了正常的多 GPU 训练功能，目前正在考虑在诊断 master 分支问题的同时合并该分支。
- **性能与重构更新**：一个已合并的 PR 通过引入新的 [优化版 matmul_bias kernel](https://github.com/karpathy/llm.c/pull/343) 带来了小幅性能提升，随后的贡献旨在通过 kernel 融合和 [CUDA stream 调整](https://github.com/ademeure/llm.c/pull/2) 进一步增强性能。
- **NCCL 与计算重叠的正确性**：在多 GPU 训练中尝试重叠 NCCL 和反向传播计算，使迭代时间从 225ms 降低到 193ms ([PR #361](https://github.com/karpathy/llm.c/pull/361))。在优化多 GPU 逻辑时，正确性验证和测试仍然至关重要。
- **Nsight Systems 性能分析**：改进性能分析的努力包括使用 Nvidia 的 Nsight Systems 以获得更好的可视化效果，并深入了解 GPU 上应用程序性能的复杂性。这包括编写教程以帮助他人设置和使用 Nsight Systems 来分析和优化 CUDA 程序。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://huggingface.co/datasets/nampdn-ai/mini-fineweb">nampdn-ai/mini-fineweb · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://siboehm.com/articles/22/CUDA-MMM">如何优化 CUDA Matmul 内核以获得类似 cuBLAS 的性能：工作日志</a>：在这篇文章中，我将迭代地优化一个用 CUDA 编写的矩阵乘法实现。我的目标不是构建一个 cuBLAS 的替代品，而是深入...</li><li><a href="https://github.com/karpathy/llm.c/discussions/344">项目进展报告 [2024年5月3日] · karpathy/llm.c · Discussion #344</a>：[2024年5月3日] 这是 llm.c 项目的第 24 天。我们现在可以进行多 GPU 训练，支持 bfloat16 和 Flash Attention，而且速度非常快！🚀 单 GPU 训练方面，我们现在训练 GPT-2 (124M) 的速度更快了.....</li><li><a href="https://github.com/karpathy/llm.c/issues/369">多 GPU 训练挂起 · Issue #369 · karpathy/llm.c</a>：在使用多个 GPU 运行 mpirun 时，在为参数的主副本分配了 474 MiB 后挂起。极有可能是由于引入了 CUDA Streams。@karpathy @PeterZhizhin</li><li><a href="https://github.com/karpathy/llm.c/pull/361">重叠梯度计算与 NCCL AllReduce，由 PeterZhizhin 提交 · Pull Request #361 · karpathy/llm.c</a>：在我的设置中，结果如下：优化前：step    2/37: train loss 4.720275 (acc 4.688650) (224.046844 ms, 36563.773438 tok/s) step    3/37: train loss 3.802741 (acc 3.943135) (224.151611 ms, 36555...</li><li><a href="https://github.com/karpathy/llm.c/pull/363">修改版的 ademeure 融合 GELU 前向内核，由 ChrisDryden 提交 · Pull Request #363 · karpathy/llm.c</a>：正在尝试融合 GELU 内核，以便在处理之前构建的非 GELU Matmuls 时能结合之前的代码，在本地运行时似乎有一个 p...</li><li><a href="https://github.com/karpathy/llm.c/pull/347/files">尝试布局的全局实例化，由 ChrisDryden 提交 · Pull Request #347 · karpathy/llm.c</a>：我观察到了速度提升，虽然需要进行大量的清理和重构，但很高兴看到潜在的加速效果</li><li><a href="https://github.com/NVIDIA/cudnn-frontend">GitHub - NVIDIA/cudnn-frontend: cudnn_frontend 为 cuDNN 后端 API 提供了一个 C++ 封装以及使用示例</a>：cudnn_frontend 为 cuDNN 后端 API 提供了一个 C++ 封装以及使用示例 - NVIDIA/cudnn-frontend</li><li><a href="https://github.com/karpathy/llm.c/pull/353">融合 LayerNorm 残差，由 ngc92 提交 · Pull Request #353 · karpathy/llm.c</a>：目前基于 #352。我没有使用 kernel 6，因为 a) 性能似乎对参数非常敏感 b) 我不理解性能测量结果/不是 100% 确定...</li><li><a href="https://github.com/karpathy/llm.c/pull/342">修复反向传播中的激活梯度重置，由 ngc92 提交 · Pull Request #342 · karpathy/llm.c</a>：此外，我们不需要在 zero_grad 中触碰其他缓冲区，这些缓冲区在反向传播期间无论如何都会被多次覆盖</li><li><a href="https://github.com/karpathy/llm.c/pull/343/commits/a0b80920f19567c1895679c4f5b553848ebd669d">性能优化：matmul_bias, CUDA Streams, 融合分类器（+移除 Cooperative Groups），由 ademeure 提交 · Pull Request #343 · karpathy/llm.c</a>：我可能需要将其拆分为多个 PR，请告诉我你的想法（我仍需将新内核添加到 /dev/cuda/）。主要变更：新的超优化 matmul_backward_bias_kernel6 CU...</li><li><a href="https://github.com/karpathy/llm.c/pull/319">将 layernorm_forward 中的所有 float 转换为 floatX，由 JaneIllario 提交 · Pull Request #319 · karpathy/llm.c</a>：将所有内核更改为使用 floatX</li><li><a href="https://github.com/karpathy/llm.c/pull/352">用于混合精度测试/基准测试的实用程序，由 ngc92 提交 · Pull Request #352 · karpathy/llm.c</a>：这允许我们编译一个单一的可执行文件，作为内核的 f32、f16 和 bf16 版本的测试/基准测试。到目前为止，我只更新了那些已经定义了 BF... 的测试文件。</li><li><a href="https://github.com/PeterZhizhin/llm.c/blob/master/train_gpt2.cu#L2036">llm.c/train_gpt2.cu (master 分支) · PeterZhizhin/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 PeterZhizhin/llm.c 的开发做出贡献。</li><li><a href="https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners">添加自托管运行器 - GitHub Docs</a>：未找到描述</li><li><a href="https://github.com/NVIDIA/cudnn-frontend?tab=readme-ov-file#debugging">GitHub - NVIDIA/cudnn-frontend: cudnn_frontend 为 cuDNN 后端 API 提供了一个 C++ 封装以及使用示例</a>：cudnn_frontend 为 cuDNN 后端 API 提供了一个 C++ 封装以及使用示例 - NVIDIA/cudnn-frontend</li><li><a href="https://github.com/karpathy/llm.c/pull/346">首次尝试将 cuDNN 移出主文件</a></li>

for faster compiles by ngc92 · Pull Request #346 · karpathy/llm.c</a>: 我认为这破坏了非 cuDNN 的构建，可能还有 Windows。不过我对 Makefile 了解不多，所以如果有人知道如何优雅地处理这些，那就太好了 :)</li><li><a href="https://openhub.net/p/tensorflow">The TensorFlow Open Source Project on Open Hub</a>: 未找到描述</li><li><a href="https://developer.nvidia.com/nsight-systems">NVIDIA Nsight Systems</a>: 分析系统、分析性能并优化平台。</li><li><a href="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_3/NsightSystems-macos-public-2024.3.1.75-3419530.dmg">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/343">Performance: matmul_bias, cuda streams, fused_classifier (+remove cooperative groups) by ademeure · Pull Request #343 · karpathy/llm.c</a>: 我可能需要将其拆分为多个 PR，请告诉我你的想法（我还需要将新的 kernel 添加到 /dev/cuda/）。主要变化：新的超优化 matmul_backward_bias_kernel6 CU...</li><li><a href="https://github.com/ademeure/llm.c/pull/2">Refactoring &amp; Improvements to reduce LOC by ademeure · Pull Request #2 · ademeure/llm.c</a>: 重构并移除未使用的函数，以减少代码行数（LOC）并使一切更加一致（同时仍保留代码的呼吸空间）。同时更新了 encoder_...</li><li><a href="https://ppc-exercises.cs.aalto.fi/course/open2024a/cp/cp4">CP4: GPU baseline</a>: 未找到描述</li><li><a href="https://ppc-exercises.cs.aalto.fi/course/open2024a/cp/cp5">CP5: fast GPU solution</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1235867967179329607)** (102 messages🔥🔥): 

- **Mojo 安装查询**：一位用户询问了在桌面端安装 Mojo 的说明，表示需要支持。
- **社区进展**：ModularBot 庆祝了一位社区成员的等级提升，展示了一种基于成就的参与系统。
- **对 Mojo 的新贡献**：讨论显示了一个开源开发环境，用户被引导至 GitHub 仓库和 Issue 进行贡献，特别是根据 'soracc' 的建议向 [Mojo 标准库](https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md)贡献。
- **解决贡献困惑**：成员 'gabrieldemarmiesse' 和 'soracc' 之间的讨论集中在澄清贡献流程、引用 [GitHub](https://github.com/modularml/mojo/pull/2457)，并考虑避免贡献者重复劳动的方法，例如“舔饼干（licking the cookie）”现象。
- **Mojo 版本方案说明**：用户澄清了 Mojo 使用的是 `YY.major.minor` 版本方案，而非语义化版本（SemVer），其中年份反映在第一个数字中（例如，版本 24.3.x 代表该年度的第三个主要发布版本）。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository">Creating a pull request template for your repository - GitHub Docs</a>: 未找到描述</li><li><a href="https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide">Modular: How to Contribute to Mojo Standard Library: A Step-by-Step Guide</a>: 我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：如何为 Mojo 标准库做贡献：分步指南</li><li><a href="https://devblogs.microsoft.com/oldnewthing/20091201-00/?p=15843)">Microspeak: Cookie licking - The Old New Thing</a>: 现在别人都不能拥有它了。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md">mojo/stdlib/docs/development.md at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做贡献。</li><li><a href="https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md#create-a-pull-request">mojo/CONTRIBUTING.md at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 开发做贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2487">[Feature Request] Make the `msg` argument of `assert_true/false/...` keyword only · Issue #2487 · modularml/mojo</a>: 审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。你的请求是什么？如题。你进行此更改的动机是什么？为了...</li><li><a href="https://open.spotify.com/track/3XwQ8ks84wlj3YcRyxXrlN?si=XJlRyCe_TzOmqPwVtDbCQQ&utm_source=copy-link">Mojo</a>: -M- · 歌曲 · 2012</li><li><a href="https://www.youtube.com/watch?v=SEwTjZvy8vw">2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing</a>: 2023 LLVM 开发者大会 https://llvm.org/devmtg/2023-10------Mojo 🔥：一种用于异构计算的系统编程语言。演讲者：Abdul Dakkak, Chr...</li><li><a href="https://github.com/modularml/mojo/issues/2415">[Feature Request] Add `__rfloordiv__()` to SIMD type · Issue #2415 · modularml/mojo</a>: 审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。你的请求是什么？Int 和 Object 类型支持 rfloordiv。我添加了...</li><li><a href="https://github.com/apple/swift/issues/43464">[SR-852] [QoI] Poor diagnostic with missing &quot;self.&quot; in convenience initializer · Issue #43464 · apple/swift</a>: 之前的 ID SR-852 Radar 无 原始报告者 @ddunbar 类型 Bug 状态 已解决 分辨率 已完成 来自 JIRA 的其他详细信息 投票 0 组件 编译器 标签 Bug, DiagnosticsQoI 被指派人 @dduan...</li><li><a href="https://github.com/modularml/mojo/pull/2457">[stdlib] Support print to stderr by GeauxEric · Pull Request #2457 · modularml/mojo</a>: 为 print 函数添加关键字参数以支持流向 stderr。修复 #2453。签署人：Yun Ding yunding.eric@gmail.com
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1786483510141657384>
  

---


**Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1237145345541017682)** (1 条消息): 

- **Modular 社区直播公告**：Modular 宣布了一场直播活动，邀请大家探索其技术的最新更新，标题为“[Modular 社区直播 - MAX 24.3 新特性](https://www.youtube.com/watch?v=kKOCuLy-0UY)”。视频将讨论 MAX Engine 和 Mojo🔥 的新功能，并介绍 MAX Engine Extensibility API。

**提到的链接**：<a href="https://www.youtube.com/watch?v=kKOCuLy-0UY">Modular 社区直播 - MAX 24.3 新特性</a>：MAX 24.3 现已发布！加入我们即将举行的直播，我们将讨论 MAX Engine 和 Mojo🔥 的新功能 - 预览 MAX Engine Extensibility API...

  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1235975948986220596)** (3 条消息): 

- **对 Donald Hoffman 意识研究的兴趣**：一位成员计划转学到 UCI，以便参与 Donald Hoffman 教授的工作，他正致力于绘制意识体验图。他们认为裂脑患者有限的感官数据与 AI 幻觉之间存在关联，这支持了模拟大脑功能的效率。
  
- **共同的学术抱负**：另一位成员表达了对上述目标的共同兴趣，表明与意识研究相关的工作保持一致。

- **寻找 Max 开发者**：一名成员宣布他们正在为一个项目寻找 Max 开发者，并请求感兴趣的人员通过私信联系以获取更多细节。
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1235850464592134234)** (172 条消息🔥🔥): 

- **大数组下 InlineArray 的异常行为**：`InlineArray` 在处理大数组时存在一些持续性的异常行为问题，正如 GitHub issue [此处](https://github.com/modularml/mojo/issues/2425) 所强调的。
- **Mojo 的 GPU 支持受到质疑**：用户对 Mojo 是“解锁 AI 硬件的语言”这一说法提出了挑战，随后官方澄清 GPU 支持计划在未来几个月内推出，并特别提到了对 Nvidia 的支持。
- **MLIR 解锁 Mojo 的潜力**：一个关键讨论点是 Mojo 的潜力不仅限于 GPU 支持，还通过 MLIR 扩展到其他硬件加速，这使得该语言在面对新兴技术时具有前瞻性。
- **关于 Mojo 中 LaTeX 脚本并行化的问题**：一位用户在 Mojo 中对 LaTeX 脚本使用并行化时遇到困难，引发了关于可并行化函数的约束以及错误处理的建议。
- **Mojo 装饰器和自定义 `None` 值的挑战**：一位用户寻求关于装饰器的帮助（目前尚未完全支持），而另一位用户在为未初始化的 struct 成员表示 `None` 时遇到困难，并学习了如何使用 `Optional[Node]` 进行正确的类型标注。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/implementing-numpy-style-matrix-slicing-in-mojo">Modular: 在 Mojo 中实现 NumPy 风格的矩阵切片🔥</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：在 Mojo 中实现 NumPy 风格的矩阵切片🔥</li><li><a href="https://github.com/modularml/mojo/tree/main/examples">mojo/examples (位于 main 分支) · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://docs.modular.com/mojo/roadmap#full-mlir-decorator-reflection,">Mojo🔥 路线图与注意事项 | Modular 文档</a>：Mojo 计划摘要，包括即将推出的功能和我们需要修复的问题。</li><li><a href="https://github.com/Nautilus-Institute/quals-2024/blob/main/%F0%9F%8C%8C/src/async_runtime.mojo">quals-2024/🌌/src/async_runtime.mojo (位于 main 分支) · Nautilus-Institute/quals-2024</a>：通过在 GitHub 上创建账号为 Nautilus-Institute/quals-2024 的开发做出贡献。</li><li><a href="https://github.com/modularml/devrel-extras/tree/main/blogs/mojo-matrix-slice">devrel-extras/blogs/mojo-matrix-slice (位于 main 分支) · modularml/devrel-extras</a>：包含开发者关系博客文章、视频和研讨会的辅助材料 - modularml/devrel-extras</li><li><a href="https://docs.modular.com/mojo/notebooks/Matmul#vectorizing-the-inner-most-loop">Mojo 中的矩阵乘法 | Modular 文档</a>：学习如何利用 Mojo 的各种函数编写高性能的 matmul。</li><li><a href="https://github.com/modularml/mojo/issues/2425.">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2467#issuecomment-2092884166">[功能请求] 统一 `InlinedString` 和 `String` 类型的 SSO · Issue #2467 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？我们目前有 https://docs.modular.com/mojo/stdlib...</li><li><a href="https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20140901/233938.html"> [llvm] r217292 - [docs] 在提交信息中记录 "NFC" 的含义</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/issues">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/pull/2507">[stdlib] 由 gabrieldemarmiesse 为 `String` 结构体添加小字符串优化 (SSO) · Pull Request #2507 · modularml/mojo</a>：修复 #2467 的一部分。此 PR 将保持草稿状态，因为它太大无法一次性合并，我将进一步拆分为多个 PR。我还有一些清理工作和基准测试要做。但它可以提供...</li><li><a href="https://github.com/modularml/mojo/pull/2539">由 KarateCowboy 添加关于尚未支持自定义装饰器的说明 · Pull Request #2539 · modularml/mojo</a>：未找到描述
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1236019023083212881)** (22 条消息🔥):

- **NuMojo 更新突飞猛进**：[NuMojo](https://github.com/MadAlex1997/NuMojo)（原名 Mojo-Arrays）已恢复活跃开发，并更新至 Mojo 24.3 版本。该库专注于围绕标准库 tensor 构建函数，目前速度显著提升，与 NumPy 相比，性能提高了 6 倍到 20 倍。

- **用于 Mojo 图像解析的 Mimage 库**：引入了一个名为 [Mimage](https://github.com/fnands/mimage) 的新库，用于在 Mojo 中进行图像解析，目前支持简单的 8 位 RGB PNG。社区正在讨论是采用 PIL 风格的 Image 类还是采用图像的 ND 数组表示。

- **Basalt 开发里程碑**：[Basalt 项目](https://github.com/basalt-org/basalt) 庆祝其 GitHub Star 数达到 200 颗，并在 [Basalt Docs](https://basalt-docs.vercel.app/) 发布了新文档，同时宣布了针对 Mojo 24.3 的更新。这些更新包括实验性的 ONNX 模型导入/导出、动态算子支持以及各种增强和错误修复。

- **Mojo 中 Struct 可组合性原型**：用于 Mojo 中 HTML 生成的 lsx 库在 [GitHub lsx](https://github.com/rd4com/lsx/tree/main/struct%20composability%20prototype) 分享了一个新的 Struct 可组合性原型，旨在实现与 lsx 的完全兼容并更好地处理 UnsafePointers。

- **MinBPE 移植与性能见解**：发布了 Andrej Karpathy 的 minbpe 项目的 Mojo 移植版本 [minbpe.mojo](https://github.com/dorjeduck/minbpe.mojo)，强调了从 Python 移植的挑战以及 Mojo 中缺少继承的问题。Mojo 版本的速度大约是 Python 原版的三倍，在切换到更高效的字典实现后，性能提升尤为明显。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/rd4com/lsx/tree/main/struct%20composability%20prototype">lsx/struct composability prototype at main · rd4com/lsx</a>：一个用于 Mojo 中 HTML 生成的实验性库 - rd4com/lsx</li><li><a href="https://github.com/dorjeduck/minbpe.mojo">GitHub - dorjeduck/minbpe.mojo: port of Andrjey Karpathy&#39;s minbpe to Mojo</a>：将 Andrej Karpathy 的 minbpe 移植到 Mojo。可以通过在 GitHub 上创建账户为 dorjeduck/minbpe.mojo 的开发做出贡献。</li><li><a href="https://github.com/mzaks/mojo-sort">GitHub - mzaks/mojo-sort</a>：为 mzaks/mojo-sort 的开发做出贡献。</li><li><a href="https://github.com/saviorand/lightbug_http/issues/34">Client tests don&#39;t work with changes in Mojo 24.3 · Issue #34 · saviorand/lightbug_http</a>：自 Mojo 24.3 起，不再支持包内的 main() 函数。这曾用于 /tests/run.mojo 来运行测试套件（目前只是一个客户端测试）。客户端测试通过运行...</li><li><a href="https://github.com/gorodion/pycv">GitHub - gorodion/pycv</a>：为 gorodion/pycv 的开发做出贡献。</li><li><a href="https://github.com/gorodion/pycv/blob/main/demo.ipynb">pycv/demo.ipynb at main · gorodion/pycv</a>：为 gorodion/pycv 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1236021318067949648)** (6 条消息): 

- **使用 Mojo 和 Parameters 构建**：分享了一个关于使用 Parameters 构建 Mojo 应用的新教程，增强了工作流并集成了自定义约束。教程可在 [GitHub - Tutorial on parameters in Mojo](https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md) 获取。

- **语法高亮技巧**：针对 Mojo Parameters 教程，有人建议在 Markdown 文件中使用带有 "mojo" 标识的三反引号来提高代码的可读性。

- **探索在 Mojo 中解析 PNG**：分享了一篇关于使用 Mojo 解析 PNG 的博文，并发布了一个名为 *mimage* 的库，用于在 Mojo 中读取图像。[博文](https://fnands.com/mojo-png-parsing/)和 [mimage 库](https://github.com/fnands/mimage)均可在网上访问。

- **社区正面反馈**：关于 PNG 解析的博文收到了社区的正面反馈，同行们对这一努力表示赞赏。

- **RSS Feed 需要修复**：在一名社区成员表示有兴趣订阅未来的文章后，该博文作者承认需要修复其网站上的 RSS Feed 问题。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/rd4com/mojo-learning/blob/main/tutorials/use-parameters-to-create-or-integrate-workflow.md">mojo-learning/tutorials/use-parameters-to-create-or-integrate-workflow.md at main · rd4com/mojo-learning</a>: 📖 学习一些 Mojo！通过在 GitHub 上创建账号来为 rd4com/mojo-learning 的开发做出贡献。</li><li><a href="https://fnands.com/mojo-png-parsing/">Parsing PNG images in Mojo</a>: 目前 Mojo 还没有直接读取图像文件的方法。在这篇文章中，我介绍了如何在不通过 Python 的情况下，直接在 Mojo 中解析 PNG 文件。此外，我还重构了...</li><li><a href="https://github.com/fnands/mimage">GitHub - fnands/mimage: A library for parsing images in Mojo</a>: 一个用于在 Mojo 中解析图像的库。通过在 GitHub 上创建账号来为 fnands/mimage 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 条消息): 

Zapier: Modverse Weekly - 第 32 期
https://www.modular.com/newsletters/modverse-weekly-32
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1235851909139660883)** (92 条消息🔥🔥): 

- **80 列限制辩论升温**：Discord 参与者讨论了超越 [80 列惯例](https://stackoverflow.com/questions/4651012/why-is-the-default-terminal-width-80-characters) 的必要性，这是打孔卡和显示器的历史遗留问题。一些成员表示更倾向于 100 列，认为这仍然允许并排查看多个文件。

- **Nightly Mojo 编译器更新**：发布了新的 [Mojo 编译器 nightly 版本](https://github.com/modularml/mojo/pull/2498/files)，提供的链接中包含近期更改的详细信息。鼓励用户使用 `modular update nightly/mojo` 进行更新。

- **Register passable 类型面临调整**：围绕 Mojo 中 "register passable" 概念的演变展开了讨论，目标是逐步淘汰像 `OptionalReg` 这样的类型，转而使用像 `Optional` 这样全能的类型，并倾向于使用 traits 来指示 register passability。

- **回应 math 模块的状态**：确认 math 模块并未消失；它尚未开源，因此在 stdlib 的开源部分中删除了对它的引用。

- **提交了 Pre-commit hook 问题**：报告了一个关于 ["check-license" pre-commit hook 的问题](https://github.com/modularml/mojo/issues/2528#issuecomment-2094837006) 的 issue，该 hook 无法找到 stdlib，引发了讨论并最终针对这个间歇性问题提交了 issue。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/questions/4651012/why-is-the-default-terminal-width-80-characters">为什么默认终端宽度是 80 个字符？</a>：80 似乎是许多不同环境中的默认值，我正在寻找技术或历史原因。众所周知，代码行不应超过 80 个字符，但是...</li><li><a href="https://github.com/modularml/mojo/issues/2413">[功能请求] 允许子 Trait 替换父 Trait · Issue #2413 · modularml/mojo</a>：查看 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。您的请求是什么？如果一个函数接收受 Trait 约束的可变参数...</li><li><a href="https://github.com/modularml/mojo/issues/2492)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2528#i">[BUG] check-license 有时失败 · Issue #2528 · modularml/mojo</a>：Bug 描述。用于许可证检查的 pre-commit 钩子有时会失败。我一直无法理解它何时发生，从日志来看，似乎找不到我的 stdlib.mojopkg，但运行...</li><li><a href="https://github.com/modularml/mojo/issues/2528#issuecomment-2094837006">[BUG] check-license 有时失败 · Issue #2528 · modularml/mojo</a>：Bug 描述。用于许可证检查的 pre-commit 钩子有时会失败。我一直无法理解它何时发生，从日志来看，似乎找不到我的 stdlib.mojopkg，但运行...</li><li><a href="https://github.com/modularml/mojo/issues/2534">[BUG] `__call_location().file_name` 返回错误信息 · Issue #2534 · modularml/mojo</a>：Bug 描述。似乎 __call_location() 函数返回了错误的数据。有人建议，“它看起来像是将我们的内部源代码路径硬编码到了 Mojo 二进制文件中...”</li><li><a href="https://github.com/modularml/mojo/issues/2529">[BUG] 函数、Trait、结构体和别名泄露到 builtins 中，并且可以从任何地方导入 · Issue #2529 · modularml/mojo</a>：Bug 描述。正如标题所述，在 "./stdlib/builtin/anything.mojo" 中导入任何不带前导下划线的内容，都会将其插入到不需要全局导入的内容列表中...</li><li><a href="https://github.com/modularml/mojo/issues/2425">[BUG] 操作 StaticTuple 时编译时间过长 · Issue #2425 · modularml/mojo</a>：Bug 描述。以下代码编译大约需要 40 秒，而构建后的实际执行时间微不足道。编译时间还与 Tuple 大小有关，而不是...</li><li><a href="https://github.com/modularml/mojo/pull/2498/files">[stdlib] 更新 stdlib 以对应 2024-05-03 nightly/mojo，由 JoeLoser 提交 · Pull Request #2498 · modularml/mojo</a>：这使用与今天的 nightly 版本对应的内部提交更新了 stdlib：mojo 2024.5.323。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">nightly 分支下的 mojo/docs/changelog.md · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户来为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1235906896414769202)** (2 条消息):

```html
<ul>
    <li><strong>社区亮点更新</strong>：社区亮点第 56 期介绍了 <a href="https://huggingface.co/spaces/Csplk/moondream2-batch-processing">Moondream 2 批处理</a>、<a href="https://huggingface.co/spaces/fluently/Fluently-Playground">FluentlyXL v4</a>、HF Audio 课程前几章的葡萄牙语翻译、用于长字幕的 <a href="https://huggingface.co/spaces/unography/image-captioning-with-longcap">BLIP 微调</a>以及许多其他项目。此处还提供了一份全面的葡萄牙语列表和亮点回顾 <a href="https://iatalk.ing/destaques-comunidade-hugging-face/">here</a>。</li>
    <li><strong>AI 新进展分享</strong>：最新的 Spaces 包含 <a href="https://huggingface.co/spaces/as-cle-bert/bloom-multilingual-chat">BLOOM 多语言聊天</a>、一个 <a href="https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad">局部重绘（inpainting）素描板</a>以及一个链接预测 <a href="https://github.com/Lama-West/PnPR-GCN_ACM_SAC_24/tree/main">仓库</a>。此外，正如这条 <a href="https://twitter.com/dstackai/status/1785315721578459402">推文</a>所述，HuggingFace alignment handbook 任务现在可以通过 dstack 在云端运行。</li>
    <li><strong>社区揭晓的酷炫内容</strong>：涵盖了从 <a href="https://huggingface.co/blog/AmelieSchreiber/protein-optimization-and-design">使用生成式 AI 进行蛋白质优化</a> 到 <a href="https://huggingface.co/blog/AviSoori1x/seemore-vision-language-model">从零开始实现 Vision Language Model</a> 的广泛主题。还讨论了结合 LLM 的 Google Search、用于快速 LLM 推理的 Token Merging 以及 <a href="https://huggingface.co/blog/maywell/llm-feature-transfer">一键创建聊天模型</a>。</li>
    <li><strong>前沿对话</strong>：已安排读书会讨论近期进展并分享见解，进一步促进 AI 领域的知识交流。要参加下一场活动，请查看此 <a href="https://discord.com/events/879548962464493619/1234913780048203856">链接</a>。</li>
    <li><strong>引入 AutoTrain 配置</strong>：AutoTrain 现在支持 yaml 配置文件，简化了模型训练过程，即使是机器学习新手也能轻松上手。有关此新功能的公告已发布在 <a href="https://twitter.com/abhi1thakur/status/1786368641388179797">推文</a>上，包含示例配置的 GitHub 仓库可在此处 <a href="https://github.com/huggingface/autotrain-advanced">访问</a>。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/autotrain-advanced">GitHub - huggingface/autotrain-advanced: 🤗 AutoTrain Advanced</a>: 🤗 AutoTrain Advanced。通过在 GitHub 上创建账号，为 huggingface/autotrain-advanced 的开发做出贡献。</li><li><a href="https://iatalk.ing/destaques-comunidade-hugging-face/)">🤗 Destaques da Comunidade</a>: Destaques da Comunidade 是在 Hugging Face Discord 上定期发布的一篇帖子，包含由社区制作的一系列项目、模型、Spaces、帖子和文章……
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1235859617876021278)** (225 messages🔥🔥): 

- **探索音频扩散建模**：围绕创建一个根据反馈迭代生成音乐的模型展开了讨论，可能使用音频扩散模型。讨论了此类模型所需的计算深度及其在生成更长且符合理论的乐曲方面的能力。

- **困扰于大型模型转换**：一位用户在将 PyTorch 模型转换为 TensorFlow Lite 格式时遇到困难，遇到了大小限制错误。该模型在从 ONNX 转换为 TensorFlow 时超过了 2GB 的限制。

- **为菲律宾语 ASR 部署 Whisper**：讨论了为菲律宾语微调 Whisper ASR 模型的可行性。提到了 `weight_decay`、学习率和数据集大小（80k 音频块）等影响性能的因素。

- **黑客攻击后引发的安全担忧**：多条消息表明 Hugging Face 的 Twitter 账号被盗，引发了关于网络安全措施及其对 AI 系统影响的讨论。社区积极标记可疑活动并调查情况。

- **GPU 利用率之谜**：用户分享了关于本地机器与 Google Colab 之间 GPU 训练时间差异的经验和建议，研究了消费级显卡与边缘推理卡之间的效率差异，并提供了优化建议。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html">目录</a>：旨在通过多年发表的科学研究来追溯数据科学历史的开源项目</li><li><a href="https://www.llama2.ai/">在 Replicate 上与 Meta Llama 3 聊天</a>：Llama 3 是来自 Meta 的最新语言模型。</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">meta-llama/Meta-Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-Gradient-1048k">crusoeai/Llama-3-8B-Instruct-Gradient-1048k-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/learn/computer-vision-course">欢迎来到社区计算机视觉课程 - Hugging Face Community Computer Vision Course</a>：未找到描述</li><li><a href="https://github.com/komorra">komorra - 概览</a>：Programmer / twitter: https://twitter.com/komorra86 - komorra</li><li><a href="https://github.com/amjadmajid/BabyTorch">GitHub - amjadmajid/BabyTorch: BabyTorch 是一个极简的深度学习框架，具有与 PyTorch 类似的 API。这种极简设计鼓励学习者探索和理解深度学习过程的底层算法和机制。它的设计初衷是当学习者准备好切换到 PyTorch 时，只需删除 `baby` 这个词即可。</a>：BabyTorch 是一个极简的深度学习框架，具有与 PyTorch 类似的 API。这种极简设计鼓励学习者探索和理解深度学习过程的底层算法和机制...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1235893693454614528)** (12 条消息🔥): 

- **模型导出难题**：一位成员在导出微调模型时遇到困难，并遇到了令人沮丧的错误。
- **是否要手写循环**：关于是否总是建议编写自己的训练循环存在争论，一位成员建议使用来自 **Diffusers** 的示例并进行修改，这样可以实现更多的自定义。
- **对 Kolmogorov-Arnold Networks 感兴趣**：**Kolmogorov-Arnold Networks (KANs)** 因其比 MLPs 使用更少计算图的潜力而受到关注。该概念得到了研究支持，并分享了一个[学术链接](https://arxiv.org/abs/2404.19756v1)，该链接在准确性和可解释性方面将 KANs 与 MLPs 进行了比较。
- **深入研究微调**：一位成员分享了关于微调生成式 AI 模型含义的教育资源，包括一段[两分钟的 YouTube 视频](https://www.youtube.com/watch?v=yoLwkowb2TU&t=1s)和一个 [HuggingFace 教程](https://huggingface.co/docs/transformers/training)。
- **克服 API 部署挑战**：一位学习者在 Hugging Face Space 的 API 构建阶段遇到问题并寻求帮助，提到了 deeplearning.ai Hugging Face 课程中的一课，并指出了 `requirements.txt` 中的版本问题。
- **分步推理方法论**：一位成员尝试为 LLM 输出实现“逐步思考”的方法，但发现本地模型不能很好地理解这一点。一种涉及 `planner`、`writer`、`analyst` 和 `editor` 链的替代方案在 Llama 3 instruct 7B 上测试时取得了更全面的结果。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.19756v1">KAN: Kolmogorov-Arnold Networks</a>：受 Kolmogorov-Arnold 表示定理的启发，我们提出 Kolmogorov-Arnold Networks (KANs) 作为多层感知器 (MLPs) 的有前途的替代方案。虽然 MLPs 具有固定的激活函数...</li><li><a href="https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/15/deployment">DLAI - 使用 Hugging Face 的开源模型</a>：介绍 · 选择模型 · 自然语言处理 (NLP) · 翻译和摘要 · 句子嵌入 · 零样本音频分类 · 自动语音识别 · 文本转语音...</li><li><a href="https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/15/deployment=====">DLAI - 使用 Hugging Face 的开源模型</a>：介绍 · 选择模型 · 自然语言处理 (NLP) · 翻译和摘要 · 句子嵌入 · 零样本音频分类 · 自动语音识别 · 文本转语音...</li><li><a href="https://huggingface.co/docs/transformers/training">微调预训练模型</a>：未找到描述</li><li><a href="https://docs.google.com/presentation/d/1IkzESdOwdmwvPxIELYJi8--K3EZ98_cL6c5ZcLKSyVg/edit#slide=id.p">2024 年构建大语言模型小指南</a>：2024 年构建大语言模型小指南 thomas@huggingface.co
</li>
</ul>

</div>
  

---

**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1235890773149614120)** (11 messages🔥): 

- **利用 RAG 彻底改变检索**：一份 [Databricks 术语表条目](https://www.databricks.com/it/glossary/retrieval-augmented-generation-rag) 讨论了 *检索增强生成 (Retrieval-Augmented Generation, RAG)*，强调了它如何解决大语言模型 (LLMs) 无法访问其原始训练集之外数据的问题，从而避免模型变得静态且有时不准确。
- **数据集巨头在 GitHub 上交锋**：Microsoft 发布了 [MS-MARCO-Web-Search 数据集](https://github.com/microsoft/MS-MARCO-Web-Search)，这是一个大规模 Web 数据集，包含数百万个真实的点击查询-文档标签，用于改进信息检索系统。
- **让 Webhooks 响起来**：Hugging Face 发布了一份指南，介绍如何创建一个监听 Webhooks 的服务器，部署到基于 Gradio 的 Spaces，并 [与 Huggingface Hub 集成](https://huggingface.co/docs/huggingface_hub/guides/webhooks_server#create-an-endpoint)。
- **步入量子服务**：分享了一个指向 [oqtant™ 量子虚拟服务器平台](https://oqtant.infleqtion.com/) 的链接，暗示了量子计算资源在可访问性方面的进展。
- **使用 Ragas 评估你的 RAG**：[Ragas 框架](https://docs.ragas.io/en/stable/) 被介绍为一种评估 LLM 应用中检索增强生成 (RAG) 流水线性能的工具，强调指标驱动的开发和用于稳健评估的合成测试集生成。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.ragas.io/en/stable/">Introduction | Ragas</a>：未找到描述</li><li><a href="https://oqtant.infleqtion.com/">Oqtant</a>：未找到描述</li><li><a href="https://lilianweng.github.io/">Lil&#39;Log</a>：记录我的学习笔记。</li><li><a href="https://huggingface.co/docs/huggingface_hub/guides/webhooks_server#create-an-endpoint">Webhooks Server</a>：未找到描述</li><li><a href="https://github.com/microsoft/MS-MARCO-Web-Search">GitHub - microsoft/MS-MARCO-Web-Search</a>：一个大规模、信息丰富的 Web 数据集，具有数百万个真实的点击查询-文档标签。</li><li><a href="https://www.databricks.com/it/glossary/retrieval-augmented-generation-rag">什么是 Retrieval Augmented Generation (RAG)? | Databricks</a>：RAG 是一种架构方法，它使用数据作为大语言模型 (LLM) 的上下文，以提高相关性...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1235893212393111594)** (19 messages🔥): 

- **Shadow-Clown BioMistral 发布**：一个名为 [shadow-clown-BioMistral-7B-DARE](https://huggingface.co/kimou605/shadow-clown-BioMistral-7B-DARE) 的新模型已创建，它使用 **mergekit** 合并了 **BioMistral-7B-DARE** 和 **shadow-clown-7B-dare**，旨在结合两个模型的能力。
- **生成式合成数据工具发布**：一个用于生成和规范化合成数据的新工具现已在 PyPI 上可用，这可能有利于微调大语言模型。更多详情可以在 [GitHub 仓库](https://github.com/tobiadefami/fuxion) 中找到。
- **通过 Ollama 高效加载 LLM**：一个 [GitHub 页面](https://github.com/di37/LLM-Load-Unload-Ollama) 和一篇 [LinkedIn 帖子](https://www.linkedin.com/feed/update/urn:li:activity:7192369828848877568/) 展示了在使用 Ollama 时高效加载和卸载 LLM 的方法。
- **AI 辅助你的播客创作**：HuggingFace 上的 [Podcastify](https://huggingface.co/spaces/eswardivi/Podcastify) Space 可以将文章转换为类似播客的对话。
- **OpenGPTs 挑战 GPT Store**：[OpenGPTs-platform](https://github.com/OpenGPTs-platform) 发布，旨在模仿并扩展官方 GPT Store 的功能，初始版本包含 "Assistants API" 和各种内容检索工具。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/Delik/pyannote-speaker-diarization-3.1">Pyannote Speaker Diarization 3.1 - a Hugging Face Space by Delik</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/eswardivi/Podcastify">Podcastify - a Hugging Face Space by eswardivi</a>: 未找到描述</li><li><a href="https://www.notion.so/Tutorial-Moondream-2-Vision-Model-with-LLaMA-71006babe8d647ce8f7a98e683713018?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: 一个将日常工作应用整合为一的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://huggingface.co/datasets/BEE-spoke-data/fineweb-100_128k">BEE-spoke-data/fineweb-100_128k · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/kimou605/shadow-clown-BioMistral-7B-DARE">kimou605/shadow-clown-BioMistral-7B-DARE · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Nick088/Real-ESRGAN_Pytorch">RealESRGAN Pytorch - a Hugging Face Space by Nick088</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/fishaudio/fish-speech-1">Fish Speech 1 - a Hugging Face Space by fishaudio</a>: 未找到描述</li><li><a href="https://github.com/di37/LLM-Load-Unload-Ollama">GitHub - di37/LLM-Load-Unload-Ollama: This is a simple demonstration to show how to keep an LLM loaded for prolonged time in the memory or unloading the model immediately after inferencing when using it via Ollama.</a>: 这是一个简单的演示，展示了在使用 Ollama 时，如何让 LLM 在内存中长时间保持加载状态，或者在推理后立即卸载模型。</li><li><a href="https://github.com/tobiadefami/fuxion">GitHub - Tobiadefami/fuxion: Sythetic data generation and normalization functions</a>: 合成数据生成与归一化函数 - Tobiadefami/fuxion</li><li><a href="https://github.com/Gapi505/Sparky-2">GitHub - Gapi505/Sparky-2</a>: 通过在 GitHub 上创建账号来为 Gapi505/Sparky-2 的开发做出贡献。</li><li><a href="https://astrabert.github.io/everything-ai">everything-ai</a>: 介绍 everything-ai，您功能完备、AI 驱动且本地运行的聊天机器人助手！ 🤖</li><li><a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: Introducing everything-ai, your fully proficient, AI-powered and local chatbot assistant! 🤖</a>: 介绍 everything-ai，您功能完备、AI 驱动且本地运行的聊天机器人助手！ 🤖 - AstraBert/everything-ai</li><li><a href="https://youtubevideosum.streamlit.app/">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1236354360716427314)** (45 messages🔥): 

- **Graph ML 与 LLMs 讨论预告**: **HuggingFace Discord** 小组正在举行一场围绕近期一篇关于 [Graph Machine Learning](https://arxiv.org/abs/2404.14928)（图机器学习）论文的[会议](https://discord.com/channels/879548962464493619/1203285086624157696)。该论文涵盖了大型语言模型 (LLMs) 在图机器学习中的应用及其广泛用途。
- **GNNs：无限可能的图景**: 成员们正在讨论 **Graph Neural Networks (GNNs)** 的多样化用途，从欺诈检测到生成推荐，甚至包括机器人的任务规划。GNNs 的多功能性激发了参与者的兴趣，促使一些人开始[尝试](https://cdn.discordapp.com/emojis/1225927322117341337.webp?size=48&quality=lossless)这些模型。
- **分享演示资源**: 演讲者 **Isamu Isozaki** 分享了一篇深入探讨讨论主题的 [Medium 文章](https://isamu-website.medium.com/understanding-graph-machine-learning-in-the-era-of-large-language-models-llms-dce2fd3f3af4)，并为错过现场演示的人提供了 [YouTube 视频](https://www.youtube.com/watch?v=cgMAvqgq0Ew&ab_channel=IsamuIsozaki)。此外，由于 Medium 的访问限制，大家还在讨论将内容上传到其他平台。
- **在 LLMs 中加入特殊 Token**: 一位成员重点介绍了一篇[论文](https://arxiv.org/abs/2404.19705)，该论文提出了一种训练方法，教导 LLMs 在不确定时使用特殊 Token `<RET>` 来触发信息检索。该方法旨在通过仅在必要时检索信息，来提高 LLMs 的准确性和效率。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/what-is-bro-yammering-about-what-is-bro-wafflin-about-what-is-bro-yapping-abo">no title found</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.14928">Graph Machine Learning in the Era of Large Language Models (LLMs)</a>: 图在表示社交网络、知识图谱和分子发现等各个领域的复杂关系方面发挥着重要作用。随着深度学习的出现，图神经网络 (Graph Neural N...</li><li><a href="https://tenor.com/view/what-is-bro-yammering-about-what-is-bro-wafflin-about-what-is-bro-yapping-about-yapping-what-is-bro-yappin-about-gif-12728898718751592705">What Is Bro Yammering About What Is Bro Wafflin About GIF - What is bro yammering about What is bro wafflin about What is bro yapping about - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=cgMAvqgq0Ew&ab_channel=IsamuIsozaki">Hugging Face Reading Group 20: Graph Machine Learning in the Era of Large Language Models (LLMs)</a>: 演讲者：Isamu Isozaki，总结报告：https://isamu-website.medium.com/understanding-graph-machine-learning-in-the-era-of-large-language-models-llms-dce2fd3f3af4</li><li><a href="https://arxiv.org/abs/2404.19705">When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively</a>: 在本文中，我们展示了大语言模型 (LLMs) 如何有效地学习使用现成的检索 (IR) 系统，特别是在需要额外上下文来回答...</li><li><a href="https://bytez.com/read/arxiv/2404.19705">Bytez: When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively</a>: 在本文中，我们展示了大语言模型 (LLMs) 如何有效地学习使用现成的检索 (IR) 系统，特别是在需要额外上下文来回答...</li><li><a href="https://x.com/omarsar0/status/1785498325913108556?t=Mfnr02-d3Hn0J4vcH9KPNA&s=09">Tweet from elvis (@omarsar0)</a>: 何时进行检索？这篇新论文提出了一种训练 LLMs 有效利用信息检索的方法。它首先提出了一种训练方法，教 LLM 生成一个特殊的 token，&...</li><li><a href="https://youtu.be/gu5ttnClB5g?si=pTOTrcgsdMG6Q4mV">Training an LLM to effectively use information retrieval</a>: 这篇新论文提出了一种训练 LLMs 有效利用信息检索的方法。它首先提出了一种训练方法，教 LLM 生成...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1236019902398201936)** (42 messages🔥): 

- **汽车零部件中的间隙检测挑战**：一位成员描述了使用简单的 YOLO 分类模型来检测某些车辆部件间隙时遇到的问题。他们请求关于替代模型或技术的建议，以提高检测性能。

- **对经典 CV 的渴望**：一位计算机视觉领域的新成员询问了传统 CV 技术（如 SURF 和 SIFT）在当前行业的关联性，并想知道是否有必要深入了解这些方法。

- **微调目标检测**：讨论了微调目标检测模型的分类器部分，重点在于使用额外的 CNN 进行图像缩放是否比在输入 Darknet YOLO 等模型之前预缩放图像更有帮助。

- **CLIP 在旋转物体上的性能**：一位用户寻求关于使用 CLIP 模型匹配未完全对齐的《万智牌》(Magic: The Gathering) 卡牌图像的建议。建议包括使用旋转和倾斜的图像增强训练数据，以提高鲁棒性。

- **寻找 GhostNet 权重**：一位成员询问了适用于 TensorFlow 的 ImageNet 预训练 GhostNet 权重的可用性，分享了 [GhostNet 论文摘要](https://arxiv.org/abs/1911.11907) 和 [Efficient-AI-Backbones GitHub 仓库](https://github.com/huawei-noah/ghostnet)，但请求在 TensorFlow 中使用所提供权重的帮助。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1911.11907">GhostNet: More Features from Cheap Operations</a>: 由于内存和计算资源有限，在嵌入式设备上部署卷积神经网络 (CNN) 非常困难。特征图中的冗余是该模型的一个重要特征...</li><li><a href="https://github.com/huawei-noah/ghostnet">GitHub - huawei-noah/Efficient-AI-Backbones: Efficient AI Backbones including GhostNet, TNT and MLP, developed by Huawei Noah&#39;s Ark Lab.</a>: 由华为诺亚方舟实验室开发的搞笑 AI 骨干网络，包括 GhostNet、TNT 和 MLP。 - huawei-noah/Efficient-AI-Backbones
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1235892774528880641)** (12 messages🔥):

- **寻求简化指令**：一位用户询问关于使用某个工具或方法的简化版本，但未指明具体是哪一个。
- **提供定制微调服务**：有一项公开请求，用户愿意为如何使用自定义数据集微调 **Mistral-7B-instruct** 模型的指导提供经济报酬。
- **对 LLM 评估的怀疑**：一位成员对使用 Large Language Models (LLMs) 来评估其他 LLMs 表示怀疑，理由是潜在的幻觉问题以及基础模型的快速迭代。该成员还指出，企业在针对其特定需求评估 LLMs 和 Retrieval-Augmented Generation (RAG) 系统时面临挑战。
- **基于 LLM 的翻译指标论文介绍**：通过 [ACL Anthology 论文链接](https://aclanthology.org/2023.eamt-1.19/) 介绍了 GEMBA 指标，这是一种基于 GPT 的翻译质量评估工具，论文描述了其在 GPT 3.5 及更大模型上的有效性。
- **请求 Flash Attention 实现教程**：一位成员询问如何将 **flash attention 2** 添加到 XLM-R，并询问 Hugging Face 是否提供了此类实现的教程或指南。

**提及的链接**：<a href="https://aclanthology.org/2023.eamt-1.19/">Large Language Models Are State-of-the-Art Evaluators of Translation Quality</a>：Tom Kocmi, Christian Federmann。Proceedings of the 24th Annual Conference of the European Association for Machine Translation。2023。

---

**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1235938948904128554)** (17 messages🔥): 

- **微调 StableDiffusionPipelines**：一位成员探索了使用两个不同 Pipeline 进行部分扩散（partial diffusion）的概念，即先用一个 Pipeline 对图像进行一半的去噪，然后用另一个继续。他们被引导至一个优秀的 **[pull request](https://github.com/huggingface/diffusers/compare/main...bghira:diffusers:partial-diffusion-2)**，该 PR 为 StableDiffusionXLPipeline 实现了这一过程。

- **部分扩散 PR 的协助**：鼓励该成员通过链接的 Pull Request 测试部分扩散功能，并直接在 PR 中报告任何问题，因为代码很快将进行重新审查和更新。

- **在多个主体上训练 Diffusion 模型**：一位成员询问关于同时训练 Diffusion 模型以学习多个主体的问题。建议他们探索 **[Custom Diffusion](https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion#:~:text=Custom%20Diffusion%20is%20unique%20because%20it%20can%20also%20learn%20multiple%20concepts%20at%20the%20same%20time.)**，这是一种允许同时学习多个概念的训练技术。

- **Accelerate 多 GPU 运行与 CPU Offloading 的冲突问题**：一位成员在结合 **accelerate 的多 GPU 运行** 与 **diffuser 的模型 CPU offloading** 时遇到了技术挑战，特别是设备相关的错误。截至最后一条消息，社区尚未解决此问题。

- **使用 LLM 价格计算器估算账单**：另一位成员寻求确认，他们拥有的 Token 计数是否足以使用分享的 **[LLM Model Pricing](https://docsbot.ai/tools/gpt-openai-api-pricing-calculator)** 计算器来估算其 API 账单。该查询在讨论中尚未得到回应。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion#:~:text=Custom%20Diffusion%20is%20unique%20because%20it%20can%20also%20learn%20multiple%20concepts%20at%20the%20same%20time.)">Custom Diffusion</a>：未找到描述</li><li><a href="https://docsbot.ai/tools/gpt-openai-api-pricing-calculator">OpenAI &amp; 其他 LLM API 价格计算器 - DocsBot AI</a>：使用我们强大的免费价格计算器计算并比较使用 OpenAI, Azure, Anthropic, Llama 3, Google Gemini, Mistral 和 Cohere API 的成本。</li><li><a href="https://huggingface.co/docs/diffusers/main/en/training/custom_diffusion#">Custom Diffusion</a>：未找到描述</li><li><a href="https://github.com/huggingface/diffusers/compare/main...bghira:diffusers:partial-diffusion-2">比较 huggingface:main...bghira:partial-diffusion-2 · huggingface/diffusers</a>：🤗 Diffusers：在 PyTorch 和 FLAX 中用于图像和音频生成的先进扩散模型。</li>
</ul>

</div>

---

**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1235902892762730557)** (212 messages🔥🔥):

- **呼吁开发者关注技能库（Skills Library）机会**：一位成员探讨了关于 OpenInterpreter **skills library** 的工作，引用了 Killian 在 GitHub 上的贡献，并建议查看 [skills.py](https://github.com/OpenInterpreter/open-interpreter/commits/59956e01ebedc74e0bfed80352ea0a90ecf154b1/interpreter/core/computer/skills/skills.py) 的提交历史。

- **微软开源 AI 黑客松公告**：成员们正在组建团队参加在西雅图举行的微软开源 AI 黑客松，意图使用 **Open Interpreter** 创建一个项目。黑客松承诺提供**实操教程**、披萨和下午茶点心，详情见[此处](https://lu.ma/iu1wijgd)。

- **Groq LLM 集成及问题**：讨论了将 **Groq LLM** 与 Open Interpreter 集成时遇到的问题，如不受控制的输出以及在桌面上创建多个文件等异常行为。提供的连接命令为：`interpreter --api_base "https://api.groq.com/openai/v1" --api_key "YOUR_API_KEY_HERE" --model "llama3-70b-8192" -y --max_tokens 8192`。

- **OpenAI Token 成本与优化担忧**：一位成员对使用 **OpenAI** GPT 的成本表示担忧，称在 API Token 上花费了大量资金。此外，还有人批评 Open Interpreter 针对闭源 AI 系统进行优化，认为这与其作为开源项目的身份不符，从而引起困惑。

- **分享本地 LLM 性能经验**：讨论包括对本地 LLM 的个人测试经验，涉及 **Phi-3-mini-128k-instruct** 和 **Groq** 模型。一位成员观察到前者存在明显的性能问题以及环境配置问题。另一位成员指出，纠正 LLM 的决策可能会带来更好的命令执行效果。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://ip_address:port/v1`">未找到标题</a>：未找到描述</li><li><a href="https://iyo.audio/">IYO</a>：IYO 构建音频计算机，欢迎你进入音频计算的世界。沉浸在混合音频现实中，与虚拟音频 Agent 交谈，帮助你学习、工作、购物和创作。</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/life-barrel-me-roll-gif-17943995">Life Barrel GIF - Life Barrel Me - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/search?q=repo%3AOpenInterpreter%2F01%20skill&type=code">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/search?q=repo%3AOpenInterpreter%2Fopen-interpreter%20skill&type=code">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://lu.ma/iu1wijgd">Open Source AI Hackathon #4 · Luma</a>：根据上一次黑客松的反馈，我们已经找到了 LLM 赞助商！OctoAI 将为所有注册者提供获得 50 美元...的机会。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#android">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>：计算机的自然语言界面。为 GitHub 上的 OpenInterpreter/open-interpreter 开发做出贡献。</li><li><a href="https://huggingface.co/microsoft/">microsoft (Microsoft)</a>：未找到描述</li><li><a href="https://rubiks.ai/search/?id=2doji3-eejo-88bg-v35a-sz678y8bv5y1">What is Reka Core?</a>：**Reka Core** 是由 Reka 开发的前沿级多模态语言模型。它是仅有的两个商用综合多模态解决方案之一，能够处理和理解...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#running-open-interpreter-locally">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>：计算机的自然语言界面。为 GitHub 上的 OpenInterpreter/open-interpreter 开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/commits/59956e01ebedc74e0bfed80352ea0a90ecf154b1/interpreter/core/computer/skills/skills.py">History for interpreter/core/computer/skills/skills.py - OpenInterpreter/open-interpreter</a>：计算机的自然语言界面。为 GitHub 上的 OpenInterpreter/open-interpreter 开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1235874656725110856)** (104 条消息🔥🔥):

- **TMC 协议的 iOS 实现**：一位成员正在重新实现用于 iOS 的 TMC 协议，以允许访问原生功能。他们质疑使用 TMC 协议相比普通 function calling 的优势，并等待关于其优点的进一步澄清。
  
- **使用 Azure Open AI 模型设置 O1**：一位成员在设置 O1 以配合 Azure Open AI 模型时遇到困难，指出尽管 OI 运行正常，但 .env 中的细节被忽略了。在之前的尝试失败后，他们正在寻求解决此问题的建议。
  
- **关于 O1 iOS 应用发布的询问**：成员们询问了 O1 iOS 应用的状态，其中一人分享了 [GitHub 仓库](https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile)的链接，其中包含了相关的源文件。进一步的讨论表明该应用仍在开发中，并提供了一个 [YouTube 链接](https://youtube.com/clip/UgkxfnZt5xbMkao8C0DmdsRTpU2bn_iaWtOI?si=wlcIV_ySO6gAfncF)，介绍如何使用 Expo 同时为 Android 和 iOS 进行构建。

- **O1 的技术故障与解决方案**：成员们正在排查 O1 的各种问题，包括安装 poetry 的问题、利用空格键执行命令的问题以及运行本地模型的困难。解决这些问题的建议包括使用 conda 环境、降低 Python 版本以及正确安装软件包。

- **探索微软 Phi-3 Mini 的兼容性**：一位用户询问是否可以将微软的 Phi-3 Mini 模型与 Open Interpreter 配合使用，另一位用户提供了安装该模型并从启动列表中选择它的说明。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.openinterpreter.com/language-models/custom-models">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile">01/software/source/clients/mobile at main · OpenInterpreter/01</a>：开源语言模型计算机。欢迎在 GitHub 上为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1258">由 rbrisita 修复文档 · Pull Request #1258 · OpenInterpreter/open-interpreter</a>：描述你所做的更改：修复自定义模型使用的文档。引用任何相关的 issue（例如 &quot;Fixes #000&quot;）：修复了 #1182 提交前检查清单（可选但建议...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1236133517772193885)** (15 条消息🔥): 

- **AI Vtuber 的 STT 挑战**：一位成员强调他们使用 **fast whisper** 实现了 **Speech-to-Text (STT)** 作为一键通话，但在实时转录方面遇到了挑战，例如 AI 中断用户以及转录背景语音。有人建议使用 *trigger word* 来提示系统，但在虚拟主播场景下被认为比较尴尬。
  
- **鼓励 AI 与直播观众互动**：AI Vtuber 主要通过 **Twitch API** 响应聊天，但在沉默期间，人工催化剂可以维持互动，直到形成观众群或 AI 学会参与游戏，这代表了集成 Twitch 聊天互动的早期阶段。
  
- **AI 管理 Twitch 聊天互动的计划**：管理 Twitch 聊天的方法涉及设置一个独立的 LLM 实例，它将理解对话流和用户消息以创建回复，目标是最终拥有一个能全面与直播聊天观众互动的 chatbot。
  
- **通过 Prompt 控制 LLM 行为**：强调了标准模型与涉及 Prompt 的 Instruct 模型之间的区别；建议使用经过微调以更好遵循指令的 **Instruct model**，以获得可控的结果。

- **分享实用的 AI 集成代码**：提到某位成员 GitHub 上的 **main.py** 文件包含聊天机器人集成的可用代码，用户只需更换 system prompt 即可适应其实现需求。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1235910528711528488)** (113 条消息🔥🔥): 

- **论文后续引发关注**：成员们分享了相关论文的链接，验证了 **large language models (LLMs)** 如何处理多语言能力，并讨论了描绘 LLM 处理多语言输入的框架，链接指向如 [How Mixture Models handle Multilingualism](https://arxiv.org/abs/2402.18815v1) 等论文。

- **对抗性挑战与架构讨论**：社区就对抗鲁棒性、通过扩展模型规模提升防御能力的潜力，以及建立系统化层级或缓冲机制以防止利用的需求展开了技术讨论，并引用了一篇关于[解决 LLM 漏洞的相关论文](http://arxiv.org/abs/2404.13208)。

- **求职分享与社区支持**：一位成员积极寻求就业机会，分享了他们的 LinkedIn 和 Google Scholar 个人资料，并强调了他们在 **EleutherAI** 的经验以及对 **Polyglot** 团队和 **OSLO project** 的贡献。

- **改进 In-Context Learning 测量**：有人提议了一种新的基准测试方法，通过改变 shot 数量来测量模型的 In-Context Learning 性能，这引发了关于评估 LLM 行为这一方面的最佳方法的对话。

- **ICLR 聚会协调**：几位社区成员讨论并安排了在 **ICLR** 的聚会，分享了计划并表达了对线下见面的兴奋，尽管一些人面临签证等旅行限制。

- **探索 System Prompt 的作用**：一位成员提到有兴趣使用 **lm-evaluation-harness** 探索 System Prompt 如何影响模型性能，但指出在使用 **Hugging Face models** 时难以找到指定 System Prompt 的方法。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.18819">Dual Operating Modes of In-Context Learning</a>：In-Context Learning (ICL) 表现出双重运行模式：任务学习（即从上下文样本中获取新技能）和任务检索（即定位并激活相关的预训练技能...）</li><li><a href="http://arxiv.org/abs/2404.13208">The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions</a>：当今的 LLM 容易受到 Prompt 注入、越狱和其他攻击的影响，这些攻击允许攻击者用自己的恶意 Prompt 覆盖模型的原始指令。在这项工作中...</li><li><a href="https://arxiv.org/abs/2402.12530">Parallel Structures in Pre-training Data Yield In-Context Learning</a>：预训练语言模型 (LMs) 具有 In-Context Learning (ICL) 能力：它们可以仅通过 Prompt 中给出的几个示例来适应任务，而无需任何参数更新。然而，目前尚不清楚...</li><li><a href="https://arxiv.org/abs/2402.18815v1">How do Large Language Models Handle Multilingualism?</a>：大语言模型 (LLMs) 在多种语言中都表现出卓越的性能。在这项工作中，我们深入探讨了这个问题：LLM 如何处理多语言？我们引入了一个框架来...</li><li><a href="https://scholar.google.com/citations?user=AbpywLMAAAAJ&hl=en">Kichang Yang</a>：崇实大学 - 被引用 50 次 - 机器学习 - NLP</li><li><a href="https://github.com/jason9693">jason9693 - 概览</a>：AI 研究工程师。jason9693 拥有 71 个代码仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1235963390623613040)** (165 条消息🔥🔥): 

- **扩展后的 Transformer 征服国际象棋**：一篇新的[研究论文](https://arxiv.org/abs/2402.04494)讨论了一个拥有 2.7 亿参数的 Transformer 模型，该模型在由 Stockfish 16 标注的 1000 万场国际象棋对局上进行了训练，在 Lichess 闪电战对局和国际象棋谜题中取得了卓越表现，且无需领域特定的调整或显式搜索算法。该模型在没有 MCTS 的情况下优于 AlphaZero 的策略和价值网络，并提出了关于规模对策略游戏影响的问题。

- **GPT-2 的复活**：消息暗示了服务器上发布内容与互动之间存在巨大差距，例如一位成员提到在回复旧帖子前有三年的间隔，另一位成员则持续与过时内容互动。

- **通过“助产式提示词”（Maieutic Prompting）增强 LLM 搜索**：介绍了 [Maieutic Prompting](https://arxiv.org/abs/2205.11822) 的概念，这是一种通过生成溯因解释树来改进 LLM 从噪声和不一致数据中进行推理的方法，尽管对其实际有效性存在怀疑。

- **人工主导评估中的挑战与考量**：详细论述了研究中人工评估在确定样本量、显著性水平和统计检验方面的复杂性，例如比较两个聊天机器人。讨论提到了非劣性检验和系统误差分析，以有意义地评估干预措施的影响。

- **防止模型滥用的不可微调学习 (Non-Fine-Tunable Learning)**：在 SOPHON 框架中展示的一个名为 [non-fine-tunable learning](https://arxiv.org/abs/2404.12699) 的新概念，旨在保护预训练模型不被微调用于不道德的用途，同时保持其在原始任务中的性能。有人担心此类保护可能会过度限制未来模型在合法应用中的适应性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hehao13.github.io/projects-CameraCtrl/">CameraCtrl</a>：未找到描述</li><li><a href="https://fxtwitter.com/sama/status/1787222050589028528">来自 Sam Altman (@sama) 的推文</a>：im-a-good-gpt2-chatbot</li><li><a href="https://arxiv.org/abs/2404.12699">SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability For Pre-trained Models</a>：开发者越来越多地依赖于将预训练模型适配到自定义任务，而不是从头构建深度学习模型。然而，强大的预训练模型可能会被滥用……</li><li><a href="https://xkcd.com/882/">Significant</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.04494#deepmind">Grandmaster-Level Chess Without Search</a>：机器学习最近的突破性成功主要归功于规模：即大规模基于 Attention 的架构和前所未有的数据集规模。本文研究了……</li><li><a href="https://en.wikipedia.org/wiki/Lady_tasting_tea">Lady tasting tea - Wikipedia</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2205.11822">Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations</a>：尽管预训练大语言模型 (LMs) 能力惊人，但在一致性推理方面仍面临挑战；最近，提示 LMs 生成自我引导推理的解释已经出现……</li><li><a href="http://arxiv.org/abs/2405.01535">Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models</a>：GPT-4 等私有 LMs 常被用于评估各种 LMs 的响应质量。然而，出于透明度、可控性和成本等方面的考虑，强烈促使……</li><li><a href="https://www.nature.com/articles/s41562-017-0189-z">Redefine statistical significance - Nature Human Behaviour</a>：我们建议将新发现声明的统计显著性默认 P 值阈值从 0.05 更改为 0.005。</li><li><a href="https://www.melonimarco.it/en/2021/03/08/stockfish-and-lc0-test-at-different-number-of-nodes/">Stockfish and Lc0, test at different number of nodes – MeloniMarco.it</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1235971685035933758)** (9 条消息🔥): 

- **预训练和微调中的 Scaling Laws**：指向 [arXiv](https://arxiv.org/abs/2102.01293) 上的一项研究链接，详细介绍了迁移学习的经验性 Scaling Laws。研究发现，由于从预训练中迁移的有效数据，预训练模型在固定大小的数据集上持续改进，这由参数量和微调数据集大小的幂律 (power-law) 描述。

- **数据集担忧中的准确率**：两名成员讨论了 [Papers With Code](mailto:hello@paperswithcode.com) 显示数学解题准确率在两年内超过 70% 的影响。一位成员认为，最近的一些进展可能是由于专门用于性能测量的 GSM8K 和 MATH 等数据集发生数据泄漏的结果。

- **预训练中包含考试数据**：成员们讨论了 OpenAI 在其预训练数据集中包含 GSM8K 和 MATH 数据的可能性。虽然一些人对规则的遵守情况表示不确定，但他们澄清说，在 MATH 上进行微调是 2021 年达到 SOTA (state-of-the-art) 的标准做法。

- **评估原始测试数据集性能**：一位成员提供了 [GitHub 上的 odyssey-math](https://github.com/protagolabs/odyssey-math) 链接，并对 GPT-4-Turbo 在该原始测试数据集上报告的 47% 基准准确率发表了评论。他们计划对部分问题进行抽样，以评估数据集的难度，并指出该数据集规模较小，仅约 350 道题。

<ul>
<li>
<a href="https://paperswithcode.com/sota/math-word-problem-solving-on-math">Papers with Code - MATH Benchmark (Math Word Problem Solving)</a>: 目前 MATH 上的 SOTA 是 GPT-4 Turbo (MACM, w/code, voting)。查看 110 篇带有代码的论文的完整对比。</li><li><a href="https://arxiv.org/abs/2102.01293">Scaling Laws for Transfer</a>: 我们研究了在无监督微调设置下，不同分布之间迁移学习的经验性缩放法则。当我们从零开始在固定大小的数据上训练越来越大的神经网络时...</li><li><a href="https://github.com/protagolabs/odyssey-math">GitHub - protagolabs/odyssey-math</a>: 通过在 GitHub 上创建账号来为 protagolabs/odyssey-math 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1235905471685333092)** (7 messages): 

- **Transformer 模型解码**: 推出了一份关于 [基于 Transformer 的语言模型的新入门指南](https://twitter.com/javifer_96/status/1786317169979970046)，提供了从多年研究中获得的模型组件和解释方法的见解，以及对可解释性工具的广泛调查。
- **寻求模型部署协助**: 一位成员请求协助模型部署，但未提供有关其面临问题的更多细节。
- **跨模型泛化得到确认**: 使用英语作为枢轴语言的语言模型可解释性结果已在多种模型中得到复现，包括 **llama 1, 2** 以及现在的 **llama 3**，如最近的一条 [推文](https://twitter.com/Butanium_/status/1786394217478004950) 所分享。
- **深入探讨权重共享 (Weight Tying) 问题**: 一位成员正在使用 **LogitLens** 探索 **Phi-2** 和 **Mistral-7B** 等开源模型中的权重共享，并在输出层发现了意想不到的结果。
- **澄清权重共享难题**: 进一步调查得出的结论是，当代的开源模型实际上并未采用权重共享，这澄清了之前观察到的异常结果。
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1235953686811770921)** (3 messages): 

- **Prometheus 模型引起关注**: 成员们对 Hugging Face 上的 [**AlekseiPravdin/prometheus-7b-v2_0-gguf**](https://huggingface.co/AlekseiPravdin/prometheus-7b-v2_0-gguf) 模型表示了兴趣，认为这可能是其工作中一个显著的改进。
- **寻求合作**: 一位成员自愿协助上述模型的集成，并强调了聊天模板在性能指标方面带来的好处。
- **集成准备工作正在进行中**: 正在编写用于实施基于 **AlekseiPravdin/prometheus-7b-v2_0-gguf** 改进的产品需求文档 (PRD)。该模型的作者也在聊天中，预示着潜在的直接合作。

**提及链接**: <a href="https://huggingface.co/papers/2405.01535">论文页面 - Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models</a>: 未找到描述

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1236191657049981049)** (3 messages): 

- **Llama 3 Lumimaid 8B 现已上线**: OpenRouter 发布了一个新模型 [Llama 3 Lumimaid 8B](https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b)，适用于 2023 - 2024 年。
- **Llama 3 Lumimaid 8B 扩展版发布**: 同时也提供 Llama 3 Lumimaid 8B 的扩展版本，为用户提供额外功能，命名为 [Llama 3 Lumimaid 8B Extended](https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b:extended)。
- **Llama 3 8B Instruct Extended 降价**: 对于寻找实惠的用户来说有个好消息，[Llama 3 8B Instruct Extended](https://openrouter.ai/models/meta-llama/llama-3-8b-instruct:extended) 的价格已经下调。
- **Lynn 模型临时停机**: 服务器更新将导致 [Lynn](https://openrouter.ai/models/lynn) 及其相关模型出现约 10 分钟的短暂停机。
- **Soliloquy L3 8B 更新至 v2**: Soliloquy L3 8B 模型已升级到版本 2，改进了重复和检索问题，增强了指令遵循能力，新价格为每 1M tokens 0.15 美元。在此探索 [Soliloquy L3 8B v2](https://openrouter.ai/models/lynn/soliloquy-l3)。
<div class="linksMentioned">

<strong>提及链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/lynn/soliloquy-l3)">Lynn: Llama 3 Soliloquy 8B v2 by lynn | OpenRouter</a>: Soliloquy-L3 v2 是一款快速且功能强大的角色扮演模型，专为沉浸式、动态体验而设计。Soliloquy-L3 在超过 2.5 亿个 token 的角色扮演数据上进行了训练，拥有庞大的知识库……</li><li><a href="https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b)">Llama 3 Lumimaid 8B by neversleep | OpenRouter</a>: NeverSleep 团队回归，带来了基于其精选角色扮演数据训练的 Llama 3 8B 微调模型。Lumimaid 在 eRP 和 RP 之间取得了平衡，设计风格严肃，但在必要时保持无审查……</li><li><a href="https://openrouter.ai/models/neversleep/llama-3-lumimaid-8b:extended>)">Llama 3 Lumimaid 8B by neversleep | OpenRouter</a>: NeverSleep 团队回归，带来了基于其精选角色扮演数据训练的 Llama 3 8B 微调模型。Lumimaid 在 eRP 和 RP 之间取得了平衡，设计风格严肃，但在必要时保持无审查……</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-8b-instruct:extended>)">Meta: Llama 3 8B Instruct by meta-llama | OpenRouter</a>: Meta 最新的模型系列 (Llama 3) 发布了多种尺寸和版本。这个 8B 指令微调版本针对高质量对话场景进行了优化。它展示了强大的……</li><li><a href="https://openrouter.ai/models/lynn>)">OpenRouter</a>: 在 OpenRouter 上浏览模型
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1236801053090119691)** (3 messages): 

- **介绍 eGirlfriend AI**：一名成员构建了名为 [eGirlfriend AI](https://egirlfriend.ai) 的项目初始版本，并邀请社区提供反馈，并指出它是 **100% 免费**的。

- **适合家庭使用的 Streamlit 聊天应用**：
 一款名为 *Family Chat* 的家庭聊天应用已创建，旨在经济高效地利用 OpenRouter API 和 OpenAI API，具有**对话记忆 (Conversational Memory)**、**PDFChat** 和**图像生成**功能。你可以在 [GitHub](https://github.com/DrDavidL/family-chat/blob/main/README.md) 上探索并为其做出贡献。

- **Rubik's AI Pro 招募 Beta 测试人员**： 
 名为 **Rubik's AI Pro** 的高级研究助手和搜索引擎的创建者正在招募 Beta 测试人员，提供 2 个月的免费高级版，其中包括访问 **GPT-4 Turbo** 和 **Mistral Large** 等模型。感兴趣的人员可以在[此处](signup.php)注册并输入促销代码 `RUBIX`。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://egirlfriend.ai,">未找到标题</a>: 未找到描述</li><li><a href="https://rubiks.ai/">Rubik's AI - AI 研究助手 & 搜索引擎</a>: 未找到描述
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1235990719760568461)** (248 messages🔥🔥): 

- **Gemini Pro 故障已修复**：报告了一个关于 **Gemini Pro** 错误消息的问题，但在几天内得到了解决。用户被告知该功能已恢复正常，如果问题仍然存在，请联系支持人员。

- **Lumimaid 70B 期待**：讨论表明正在与 Mancer 沟通关于托管 **Lumimaid 70B** 的事宜，并建议向专注于 RP 模型的提供商 Novita 咨询。

- **Phi-3 托管的不确定性**：尽管有兴趣，但目前似乎缺乏托管 **Phi-3** 的提供商，尽管据说 Microsoft Azure runner 拥有该模型，但没有按 token 计费的定价。

- **OpenRouter 与 AI 模型精度**：澄清了 OpenRouter 上的模型提供商使用不同的精度；大多数运行在 **fp16**，有些运行在量化的 **int8**。

- **Meta-Llama 3 120B Instruct 自合并**：Hugging Face 上出现了一个 **Meta-Llama 3 70B 的自合并版本**，灵感来自其他大型合并，引发了人们对自合并与层映射合并 (layer-mapped merges) 相比效果如何的好奇。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mlabonne/Meta-Llama-3-120B-Instruct">mlabonne/Meta-Llama-3-120B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B">abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B · Hugging Face</a>: 未找到描述</li><li><a href="https://octo.ai/blog/mixtral-8x22b-is-now-available-on-octoai/">Mixtral 8x22B 现已在 OctoAI 文本生成解决方案上线 | OctoAI</a>: 你可以使用 /completions API、curl 或 OpenAI SDK 在 OctoAI 上对 Mixtral 8x22B 进行推理。联系我们以运行微调版本。
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1235984738683064388)** (7 messages):

- **反思型自我改进 Agent**：LlamaIndex 0.10.34 引入了 **introspective agents**（内省型 Agent），它们可以通过反思和自我批评在无需人工干预的情况下提升性能。该方法及 `llama-index-agent-introspective` 软件包在 [notebook](https://t.co/X8tJGXkcPM) 中有详细介绍，并附带安装指南，其中包含针对敏感内容的警告。

- **Agentic RAG 进展演示**：@jasonzhou1993 的一段视频展示了构建 Agentic RAG 所需的**组件概览**，重点介绍了使用 LlamaParse + Firecrawl 进行的高级文档处理。感兴趣构建 Agent 系统的人员可以在[此处](https://t.co/wR35iYIKjo)观看视频。

- **RAG 响应的信任评估**：@CleanlabAI 开发了一个“可信语言模型”（Trustworthy Language Model），为检索增强生成 (RAG) 的响应分配**可信度评分**，解决了验证生成内容准确性的挑战。有关此功能的更多详细信息可以在他们的推文[此处](https://t.co/KW1XsllRqQ)找到。

- **本地 RAG 设置指南**：对于寻求**全本地 RAG 流水线**的用户，@pavan_mantha1 提供了一份深入的手册，介绍了使用 @llama_index 和 HyDE 层的设置。该文章被描述为比“5 行代码”快速入门更底层的指南，可通过[此链接](https://t.co/2RCvaxOzKo)访问。

- **LlamaIndex 宣布支持 Hugging Face TGI**：LlamaIndex 宣布支持 **Hugging Face TGI**，这是一个确保 Hugging Face 上语言模型优化部署的工具包，现在具备 **function calling**、批处理推理和更快的延迟等特性。关于 TGI 功能的详细信息在[此处](https://t.co/3vGpxcbP18)列出。

**提及的链接**：<a href="https://t.co/X8tJGXkcPM">Introspective Agents: Performing Tasks With Reflection - LlamaIndex</a>：未找到描述

---

**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1235849486564200489)** (226 条消息🔥🔥): 

- **使用可控 Agent 探索 RAG**：一位用户询问如何在**检索增强生成 (RAG)** 项目中实现 **Controllable agents**（可控 Agent），使 Agent 能够提出后续问题以获得更精确的检索结果。提供了使用 LlamaIndex 的详细实现指南，包括指向 [Agent Runner](https://docs.llamaindex.ai/en/examples/agent/agent_runner/agent_runner/) 和 [Controllable Agent Runner](https://docs.llamaindex.ai/en/examples/agent/agent_runner/agent_runner_rag_controllable/) 等相关文档的链接。

- **LlamaIndex 内存问题排查**：用户讨论了使用 LlamaIndex 时的高 VRAM 占用和潜在的内存泄漏问题，导致清理缓慢并回退到 CPU 处理。一位用户指出，通过新的 **[ollama v0.1.33 更新](https://github.com/ollama/ollama/releases/tag/v0.1.33)** 成功解决了此类问题。

- **LLM 微调与成本讨论**：讨论了专门针对特定任务（如在特定领域专业化的轻量级模型）微调语言模型 (LLM) 的话题。提到了微调的高昂成本，用户正在寻找可优化且具有成本效益的解决方案。

- **实现 SharePoint Reader 与 VectorStore 的挑战**：一位成员寻求关于集成 **SharePoint Reader** 以从 SharePoint 加载文件的反馈，另一位成员遇到了 LlamaIndex 中 **SupabaseVectorStore** 返回空响应的问题，这表明可能存在配置问题。

- **理解并优化基于 Excel 数据的问答系统**：一位用户询问了构建基于中等规模 Excel 表格的问答系统的最佳方法，重点是为复杂查询提供上下文相关的上下文信息。

- **LlamaIndex 特定细节的实现与配置**：多位用户讨论了导入错误、`llama-index` 中的正确路径、如何处理法律文档数据提取、如何处理 Intel 处理器的 embeddings，以及动态配置 ReAct Agent。在 **cheesyfishes**（推测是社区中的专家）的帮助下，同行之间进行了求助与交流，提供了关于 LlamaIndex 使用和集成的指导。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="http://localhost:11434",">未找到标题</a>: 未找到描述</li><li><a href="https://llamahub.ai">Llama Hub</a>: 未找到描述</li><li><a href="https://www.llamaindex.ai/contact">联系我们 — LlamaIndex，LLM 应用的数据框架</a>: 如果您对 LlamaIndex 有任何疑问，请联系我们，我们将尽快安排通话。</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">入门教程（本地模型）- LlamaIndex</a>: 未找到描述</li><li><a href="https://llama.meta.com/docs/how-to-guides/prompting">Prompting | 操作指南</a>: Prompt engineering 是一种用于自然语言处理 (NLP) 的技术，通过为语言模型提供更多关于当前任务的上下文和信息来提高其性能。</li><li><a href="https://github.com/ollama/ollama/releases/tag/v0.1.33">Release v0.1.33 · ollama/ollama</a>: 新模型：Llama 3：Meta 推出的新模型，也是迄今为止功能最强大的开源 LLM；Phi 3 Mini：Microsoft 推出的新型 3.8B 参数、轻量级、最先进的开源模型。Moondream moon...</li><li><a href="https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/embeddings/llama-index-embeddings-huggingface">llama_index/llama-index-integrations/embeddings/llama-index-embeddings-huggingface at main · run-llama/llama_index</a>: LlamaIndex 是适用于您的 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/optimum_intel/">使用 Optimum-Intel 优化 Embedding 模型 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning/">Fine-Tuning - LlamaIndex</a>: 未找到描述</li><li><a href="https://python.langchain.com/docs/modules/composition/">Composition | 🦜️🔗 LangChain</a>: 本节包含更高级别的组件，它们将其他任意系统（例如外部 API 和服务）和/或 LangChain 原语组合在一起。</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | 模型卡片和 Prompt 格式</a>: Meta Llama 3 使用的特殊 Token。一个 Prompt 应包含一条 system 消息，可以包含多条交替的 user 和 assistant 消息，并且始终以最后一条 user 消息结尾...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/ingestion/redis_ingestion_pipeline/?h=ingestionpipeline">Redis Ingestion Pipeline - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/supabase#llama_index.vector_stores.supabase.SupabaseVectorStore>).">Supabase - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/pull/13196">通过 co-antwan 调用带有 documents 参数的 Cohere RAG 推理 · Pull Request #13196 · run-llama/llama_index</a>: 描述：增加了在 RAG 流水线中使用时对 Cohere.chat 的 documents 参数的支持。这确保了 Cohere 客户端的正确格式化，并带来更好的下游性能。T...</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/monsterapi#rag-approach-to-import-external-knowledge-into-llm-as-context>).">未找到标题</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/fine-tuning/fine-tuning#finetuning-embeddings>).">Fine-Tuning - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1237021916792098916)** (4 messages): 

- **寻求 NL-SQL 机器人建议**：一位成员正在为一个拥有数百张表的复杂数据库创建 **NL-SQL 聊天机器人**，并询问关于使用 **HyDE 方法** 的建议。他们正在探索提高 LLM 生成 SQL 查询准确性的解决方案，并指出 HyDE 主要用于基于文本的聊天机器人。
- **内省代理 (Introspective Agents) 讨论**：提到了一篇题为 **"Introspective Agents with LlamaIndex"** 的文章，指出了一种涉及内省代理的新方法或进展。分享了文章链接：[Introspective Agents with LlamaIndex](https://medium.com/ai-artistry/introspective-agents-with-llamaindex-777d018f791d)。
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1235886536466239549)** (33 messages🔥): 

- **Hermes 在 Android 上运行飞快**：一位成员对 **Hermes 2 Pro Llama 3** 在 8GB RAM 的 Android 设备上的 **推理速度** 表示惊讶，并将其性能归功于 **llama.cpp**。

- **Anime 风格 AI 创新**：有一场幽默的讨论，暗示 AI 的进步和技术创新似乎与 **Anime** 在 **问答 (question-answering)** 和 **图像生成 (image generation)** 领域的泛滥交织在一起。

- **Llama.cpp 合并性能增强 PR**：一位成员分享了合并到 **llama.cpp** 的新 Pull Request 的消息，该 PR 带来了 **30% 的推理速度提升**，似乎在邀请创作 **更多 Anime**。

- **Axolotl 渐进式文档**：分享了 **Axolotl 社区** 的 **开发中文档 (work-in-progress documentation)** 链接，并邀请大家提供反馈。

- **梯度检查点 (Gradient Checkpointing) 优化报告**：记录了关于 **新 Unsloth** 梯度检查点的更新，该更新减少了 VRAM 占用，展示了社区在优化机器学习过程中内存利用率方面的积极努力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/turboderp/Cat-Llama-3-70B-instruct">turboderp/Cat-Llama-3-70B-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://www.philschmid.de/vllm-inference-endpoints">在 Hugging Face Inference Endpoints 上使用 vLLM 部署开源 LLM</a>：在这篇博文中，我们将向您展示如何在 Hugging Face Inference Endpoints 上使用 vLLM 部署开源 LLM。</li><li><a href="https://x.com/granawkins/status/1786428318478168447">来自 Grant♟️ (@granawkins) 的推文</a>：2024 年的 SOTA RAG</li><li><a href="https://x.com/tomshardware/status/1786807369961210203">来自 Tom's Hardware (@tomshardware) 的推文</a>：价值数百万美元的 Cheyenne 超级计算机拍卖以 480,085 美元成交 —— 买家带走了 8,064 颗 Intel Xeon Broadwell CPU、313TB DDR4-2400 ECC RAM 以及一些漏水问题 https://trib.al/7BzUc...</li><li><a href="https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file">GitHub - HVision-NKU/StoryDiffusion: 创造魔法故事！</a>：创造魔法故事！通过在 GitHub 上创建账号为 HVision-NKU/StoryDiffusion 的开发做出贡献。</li><li><a href="https://axolotl.continuumlabs.pro/">简介 | Continuum 训练平台 | Axolotl 训练平台</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1235918795705417758)** (8 条消息🔥): 

- **Gradio 迎来可配置性**：一位成员寻求帮助，希望通过 YAML 配置 Gradio 选项（如将 Demo 设为私有、设置 IP 地址）。解决方案包括将这些选项添加到 YAML 中，并修改代码以解析设置，正如[其实现](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1591)所示。

- **深入探讨 Gradio Token 问题**：出现了一个令人费解的问题，Gradio 没有为 Llama 3 模型使用正确的 Token，意外打印了 `<|end_of_text|>`。似乎 Gradio 的默认 Token 可能会无意中覆盖已加载的分词器 (Tokenizer) 设置，除非指定了特殊 Token。

- **推动更动态的 Gradio**：讨论了一项代码更改，允许动态配置 Gradio 的参数，如 "private"、"server_name" 和 "port"。这将通过 YAML 配置实现对 Gradio 行为的更大控制。

- **PR 已准备好评审**：提交了一个解决 Gradio 自定义问题的 Pull Request，为项目中各种硬编码选项添加了可配置参数，记录了重要细节并通过 [GitHub PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1591) 展示了实现。

- **Issue 还是 Pull Request？永恒的问题**：一位成员询问是应该为问题开一个 Issue 还是直接提交 Pull Request。虽然没有记录回复，但该成员主动创建了一个 Pull Request 来解决潜在问题。

**提到的链接**：<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1591">marijnfs 提交的 Gradio 配置参数 · Pull Request #1591 · OpenAccess-AI-Collective/axolotl</a>：Gradio 的各种参数之前是硬编码的（例如 share=True、IP 地址、端口、Token 数量、Temperature），我在这里将它们设为可配置。此外，默认 Token 被覆盖到了...

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1235866742010417153)** (8 条消息🔥):

- **对训练后的 Llama3 进行推理**：有人询问在使用 fft 脚本训练 **llama3** 后如何进行推理，并澄清通常的 **qlora** 命令和 **qlora_model_dir** 似乎并不适用。
- **调整推理参数**：一位成员建议在未指明的上下文中使用 **4,4** 的参数设置，暗示这些设置取得了成功。
- **将 Safetensors 转换为 GGUF**：一位用户寻求帮助，希望将 safetensors 转换为 **gguf**，且需要比 **llama.cpp** 提供的更多选项，特别提到了 `Q4_K` 和 `Q5_K` 等格式。
- **Llama.cpp 转换脚本**：该用户被引导至 **llama.cpp** 的转换脚本，特别提到了 [convert-gg.sh](https://github.com/ggerganov/llama.cpp/blob/master/scripts/convert-gg.sh)，推测是为了处理 **gguf 转换选项**。
- **Axolotl 社区文档**：分享了 Axolotl 社区文档的链接，该文档仍需完善，特别是在训练后合并模型权重以及使用模型进行推理方面，并邀请在 [Axolotl Community Docs](https://axolotl.continuumlabs.pro/) 提供反馈。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/ggerganov/llama.cpp/blob/master/scripts/convert-gg.sh">llama.cpp/scripts/convert-gg.sh at master · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。欢迎在 GitHub 上为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://axolotl.continuumlabs.pro/">Introduction | Continuum Training Platform | Axolotl Training Platform</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1236106691053883542)** (39 messages🔥): 

- **CodeTester 数据集扩展**：来自 Vezora 的*更新版 Python 数据集*现在包含 143,327 个经过仔细测试且可运行的代码示例，旨在辅助从 Alpaca 格式的数据集中提取和验证 Python 代码片段。关于该数据集及其创建过程的更多信息可以在 [Hugging Face 数据集仓库](https://huggingface.co/datasets/Vezora/Tested-143k-Python-Alpaca)中找到。

- **Llama3 数学训练难题**：成员们讨论了在提升 Llama3 数学内容性能方面的困难，指出尽管在 orca-math-word-problems-200k 和 MetaMathQA 等数据集上进行了训练，但*数学主题评分反而下降*，这些数据集可在 [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) 和 [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) 获取。

- **量化对模型性能的影响**：一位成员强调了 **llama.cpp 量化**对模型性能可能产生的负面影响，并引用了 GitHub 上关于合并 LORA Adapter 后进行 Llama3 GGUF 转换的讨论，更多细节可在[此 issue](https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094961774)中探索。

- **评估脚本与提示词**：一位成员使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 对 Llama3 进行推理和评估，而其他人则指出了确保正确提示词格式的重要性，并对使用 Alpaca 格式提示词对模型性能的潜在影响提出了疑问。

- **提示词格式难题**：关于微调期间的提示词格式（如使用 Alpaca 格式）如何影响模型性能的争论仍在继续。成员们正在思考，即使模型没有生成词表外 (out-of-vocabulary) 的文本结束标记，这是否仍会导致问题。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | Model Cards and Prompt formats</a>: Meta Llama 3 使用的特殊 Token。一个 Prompt 应包含单个系统消息，可以包含多个交替的用户和助手消息，并且始终以最后一个用户消息结尾...</li><li><a href="https://huggingface.co/datasets/Vezora/Tested-143k-Python-Alpaca">Vezora/Tested-143k-Python-Alpaca · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47">axolotl/src/axolotl/prompters.py at 3367fca73253c85e386ef69af3068d42cea09e4f · OpenAccess-AI-Collective/axolotl</a>: 尽管提问（axolotl questions）。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: 一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062#issuecomment-2094961774">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: 我正在运行 Unsloth 在 llama3-8b 上对 Instruct 模型进行 LORA 微调。1：我将模型与 LORA 适配器合并为 safetensors 2：在 Python 中使用合并后的模型直接运行推理...</li><li><a href="https://huggingface.co/datasets/TIGER-Lab/MathInstruct">TIGER-Lab/MathInstruct · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/meta-math/MetaMathQA">meta-math/MetaMathQA · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1236482608502935615)** (27 条消息🔥): 

- **梯度裁剪（Gradient Clipping）查询**：围绕在 **Axolotl** 中使用 Axolotl `TrainingArguments` 或在 YAML 配置中设置梯度裁剪展开了讨论。Phorm 建议在 `TrainingArguments` 中或 YAML 文件的优化设置下设置 `max_grad_norm`。

- **需要文档超链接**：成员指出，由于向 quarto markdown 的过渡，在 Axolotl YAML 中指定梯度裁剪可能未在文档中反映，这表明需要更新文档索引。

- **修改聊天机器人提示词**：一位用户询问了如何在 ShareGPT 数据集格式中修改对话训练的系统提示词（System Prompt）。Phorm 指出应调整对话模板，或者修改 `ShareGPTPrompter` 类及相关配置文件中的初始消息。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d76285fb-b795-43de-a278-b9adfdec1559)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=12d8fd24-8f30-4de3-bb8b-7a85951a30ec)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c1ca4368-cf3a-4dee-8f17-6686eaf48b1a)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=ef7c3959-d5a0-4b42-b13d-5ccc8940f344)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1236044647785304109)** (95 条消息🔥🔥):

- **Gary for Live! 一场计算音乐之旅**：一位成员分享了 [GitHub 上的 gary4live](https://github.com/betweentwomidnights/gary4live) 链接，这是一个涉及 Python continuations 和 Ableton 的在研项目，并鼓励其他人查看代码。
- **Suno 与音乐生成讨论**：围绕使用 Suno 生成音乐以及 *Musicgen* 等其他音乐生成设置的能力展开了对话。大家特别感兴趣的是探索这些模型如何处理不同的音频元素，以及它们是否生成乐谱等资产。
- **深入探讨音乐模型 Token**：聊天深入探讨了音乐模型 Token 的复杂性，讨论重点在于 Suno 对音频的 Token 化，以及关于这些 Token 的长度和组成的问题。虽然提到了论文中的架构设计，但讨论中并未充实具体细节。
- **音频合成中的潜空间 (Latent Spaces)**：参与者讨论了多模态模型在不经过文本中间体的情况下直接整合音频的潜力，强调了包含音频对于实现真正的全模态 (omnimodal) 能力的重要性。对话还包括在实时应用中使用模型生成内容来替换音频通道等想法。
- **探索 Stable Audio 的商业用途和许可**：一位成员提出了关于 Stable Audio 模型输出的商业用途和许可问题。讨论转向了此类模型的实时应用，例如使用 AI 进行现场表演循环。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/aSTTaUfm">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的朋友和社区保持紧密联系。</li><li><a href="https://notesbylex.com/snake-activation-function">来自 Snake 激活函数的推文</a>：Snake 是一种神经网络激活函数，对于建模具有“周期性归纳偏置”的问题（换句话说，具有规律性、重复模式的问题）非常有用...</li><li><a href="https://x.com/yikesawjeez/status/1786299657460855174">来自 yikes (@yikesawjeez) 的推文</a>：醒醒吧宝贝，新的神经网络架构刚刚发布 https://arxiv.org/abs/2404.19756</li><li><a href="https://arxiv.org/abs/2404.10301v1">使用潜扩散的长篇音乐生成</a>：基于音频的音乐生成模型最近取得了长足进步，但到目前为止还未能产生具有连贯音乐结构的全长音乐曲目。我们展示了通过训练一个地理...</li><li><a href="https://github.com/betweentwomidnights/gary4live">GitHub - betweentwomidnights/gary4live: 这是 gary。Python continuations 加上 Ableton 内部的 continuations。这是一个新手的在研项目。</a>：这是 gary。Python continuations 加上 Ableton 内部的 continuations。这是一个新手的在研项目。 - betweentwomidnights/gary4live
</li>
</ul>

</div>
  

---



**AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1237016300333694997)** (6 messages): 

- **云订阅费用的澄清**：成员们确认，如果本地运行，则**不需要云订阅费用**；该工具在 6 GB VRAM 下运行良好，并包含免费的语音输出。
- **拥有下载内容**：强调了通过 **Faraday** 下载角色和模型后，它们将*永久*归你所有。
- **本地使用取代云订阅**：性能足够的 GPU 可以免除云订阅的需求，云订阅被建议作为对工具开发者的可选捐赠。
  

---


**AI Stack Devs (Yoko Li) ▷ #[team-up](https://discord.com/channels/1122748573000409160/1128471951963328512/1237055950032998422)** (2 messages): 

- **征集嘻哈模拟合作**：一位成员表示有兴趣创建一个引用 **Kendrick** 和 **Drake** 之间局势的有趣模拟。另一位成员对合作号召做出了积极回应。
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1235974067073585294)** (15 messages🔥):

- **AI 领导人选举讨论**：人们对 AI 是否会选举领导人表现出好奇，特别是原始模拟论文中描述的市长选举，该事件在模拟中似乎*从未真正触发*。
- **在玩家简介中设置 AI 选举**：一位成员表示，在玩家简介（Player Bios）中设置 AI 选举将是*简单的*，并引用了对 AI 模拟中市长事件的好奇。
- **AI-Westworld 公测与 The THING 模拟**：@TheoMediaAI 的一条推文强调了对两个 AI 世界模拟的探索，包括处于公测阶段的 @fablesimulation 的 **AI-Westworld**，以及在 @realaitown 中重现电影《怪形》（The THING）。
- **推出用于回放的 AI Town Player**：@cocktailpeanut 的一条推文介绍了 **AI Town Player Web 应用**，它允许通过导入 sqlite 文件来回放任何 AI Town。该应用指出整个 AI Town 都通过 @convex_dev 存储在单个 sqlite 文件中，并兼容 Mac 和 Linux，但不兼容 Windows。
- **AI 模拟派对登上新闻**：[sfstandard.com](https://sfstandard.com/2024/05/04/mission-control-hacker-house-san-francisco-ai-simulated-party/) 的一篇专题报道描述了在旧金山 Mission Control 举办的 **AI 模拟派对**，人类参与者的活动与在屏幕上显示的数字化版本中运行的 AI 版本同步。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheoMediaAI/status/1786377663889678437">来自 Theoretically Media (@TheoMediaAI) 的推文</a>：探索两个卓越的 AI 世界模拟：首先是来自 @fablesimulation 的 AI-Westworld（公测已开启！），同时也尝试了 @realaitown，并重现了有史以来最棒的电影（The THI...</li><li><a href="https://x.com/cocktailpeanut/status/1786421948638965870">来自 cocktail peanut (@cocktailpeanut) 的推文</a>：介绍 AI Town Player。你知道整个 AI Town 都通过 @convex_dev 存储在单个 sqlite 文件中吗？我逆向工程了其 Schema 并构建了一个 Web 应用，让任何人都可以回放任何 A...</li><li><a href="https://sfstandard.com/2024/05/04/mission-control-hacker-house-san-francisco-ai-simulated-party/">我们去了旧金山“首个 AI 模拟派对”，所以你不用去了</a>：前往 Mission Control 黑客之家的一次旅行，在那里，一场超现实的虚拟狂欢之后紧接着是一场真实的 DJ 舞会。
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1235923879835205704)** (61 messages🔥🔥): 

- **Ubuntu 和 Node 版本困扰**：用户 `utensil_18981` 报告了在 Ubuntu 18 上尝试运行 `convex-local-backend` 时遇到的问题，最终通过将 Node 降级到 18.17.0 版本并修补 Ubuntu 解决了多个问题，详见[此 GitHub 线程](https://github.com/get-convex/convex-backend/issues/1)。
  
- **考虑使用 Docker 进行简化**：`utensil_18981` 对设置 `convex-backend` 和 `ollama` 表示沮丧，提到可能的 Docker 构建可以简化该过程。`.casado` 承认了该想法的价值，并考虑在周末进行研究。

- **为本地 LLM 推出 llama-farm**：`ianmacartney` 介绍了 `llama-farm`，这是一个旨在将运行 Ollama 的本地机器连接到云端后端的新项目，通过避免公网暴露提供简单的扩展性和安全性。该项目可以在 GitHub [此处](https://github.com/get-convex/llama-farm-chat)找到。

- **AI 真人秀和 AI Town 体验预告**：`edgarhnd` 预览了即将推出的 AI 真人秀迭代版本，该版本将允许公众与 AI Town 互动，暗示了增强的共享体验。

- **远程 LLM 部署的挑战与解决方案**：成员们讨论了部署本地语言模型服务器（`ollama`）并将其连接到远程 convex 后端的复杂性和障碍，`utensil_95057` 最终通过更新到最新的 Ollama 版本并使用 `ssh` 隧道使其成功运行。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.convex.dev/cli#run-the-convex-dev-server">CLI | Convex 开发者中心</a>: Convex 命令行界面 (CLI) 是你管理 Convex 的界面</li><li><a href="https://github.com/get-convex/convex-backend/issues/1">TypeError [ERR_UNKNOWN_FILE_EXTENSION]: 未知的文件扩展名 ".ts"，路径为 /app/npm-packages/convex/src/cli/index.ts · Issue #1 · get-convex/convex-backend</a>: 我按照前提条件中的步骤操作，但在仅运行 run-local-backend 时遇到了这个错误：Failed to run convex deploy: TypeError [ERR_UNKNOWN_FILE_EXTENSION]: Unknown file extension ".ts"...</li><li><a href="https://github.com/get-convex/llama-farm-chat">GitHub - get-convex/llama-farm-chat: 使用本地托管的 LLM 为你的云端托管 Web 应用提供支持</a>: 使用本地托管的 LLM 为你的云端托管 Web 应用提供支持 - get-convex/llama-farm-chat
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[local-ai-stack](https://discord.com/channels/1122748573000409160/1168947823920812125/1236174462051942410)** (1 messages): 

- **为旧笔记本电脑推出 llama-farm**: 一位成员宣布发布 `llama-farm`，它允许在旧笔记本电脑上运行 **Ollama**，为面向公众的 AI 应用提供 LLM 任务服务。正如 [GitHub](https://github.com/get-convex/llama-farm-chat) 上所述，该设置通过在其他机器上运行客户端进行扩展，且不需要代理或暴露在公共互联网中。

**提及的链接**: <a href="https://github.com/get-convex/llama-farm-chat">GitHub - get-convex/llama-farm-chat: 使用本地托管的 LLM 为你的云端托管 Web 应用提供支持</a>: 使用本地托管的 LLM 为你的云端托管 Web 应用提供支持 - get-convex/llama-farm-chat

  

---


**AI Stack Devs (Yoko Li) ▷ #[paper-spam](https://discord.com/channels/1122748573000409160/1227492197541220394/)** (1 messages): 

Deforum 每日论文：论文现在将发送至 <#1227492197541220394>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742/)** (1 messages): 

jakekies: ??
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1235934782727127081)** (59 messages🔥🔥): 

- **探索 CLIP 和 T5 的组合**: 有一场关于使用 CLIP 和 T5 编码器进行模型训练的 [讨论](https://old.reddit.com/r/StableDiffusion/comments/1cgr74j/april_30th/l2bxv66/)；一位成员提到了 CLIP 的 Prompt 遵循问题，并考虑仅使用 T5，而另一位成员则强调了过去同时使用这两种编码器的成功经验。
- **改进小型模型的考虑因素**: 提到了增强小型模型实用性的重点，并指出了 400M DeepFloyd 以及准备发布 8B 模型所面临的挑战。
- **对 SD3 策略的质疑**: 来自 Stability AI 的评论建议逐步发布 SD3 模型，从较小的模型开始到较大的模型，这引发了关于这是否是一种高效方法的讨论，特别是考虑到社区的期待。
- **在训练中使用 LLama Embeds 的潜力**: 关于在训练中使用 LLama embeds 代替 T5 的优点的对话，并分享了一个名为 [LaVi-Bridge](https://github.com/ShihaoZhaoZSH/LaVi-Bridge) 的示例桥接链接，强调了现代应用和效率。
- **图像生成和 LLM 领域的进展比较**: 成员们比较了图像生成和 LLM 领域开源模型的现状，讨论了新模型的适配，并提到了一个新的 CogVL 跑马灯。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1ciyzn5/sd3_weights_are_never_going_to_be_released_are/l2dhd6q/">SD3 权重永远不会发布了，对吧</a>: 将会发布。目前还没有日期。肯定会发布。如果这有帮助的话，我们已经与多家合作伙伴公司分享了 Beta 版模型权重...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1235953963652743298)** (5 messages): 

- **真实世界数据集与合成数据集的疑问**: 一位成员对为什么在实验中使用合成数据集而不是像 MNIST、CIFAR 或 ImageNet 这样的标准数据集表示好奇。人们对那些优先考虑可解释性但可能无法解决实际任务的方法的现实世界适用性表示担忧。

- **讨论可解释性演示**: 有人提到，在实验中使用合成数据集是为了展示正在开发的方法在可解释性方面的表现。

- **分享 StoryDiffusion 资源**: 分享了 [StoryDiffusion 网站](https://storydiffusion.github.io/) 的链接，其中可能包含有关 AI 可解释性的相关信息或资源。

- **函数表示中的复杂性优于简单性**：一位成员澄清说，研究有时旨在通过函数逼近复杂的数学表示，而不是通常与视觉识别相关的“简单”模板化任务。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1235950376109477979)** (45 条消息🔥): 

- **LLM 与数据库接口引发好奇**：参与者讨论了是将数据库数据转换为自然语言文本，还是使用 LLM 将自然语言转换为数据库查询。讨论还考虑了在这种背景下图形数据库与关系型数据库的适用性。
- **Node.JS 难题与 Langchain 入门**：一位用户在 NodeJS 中寻求解析用户问题和提取 JSON 数据的帮助，而另一位用户在使用 FAISS 与 Langchain 时遇到错误，但通过升级到最新版本解决了该问题。
- **通过 AI 执行代码**：社区成员交流了关于通过 AI Agent 执行生成的代码的见解，建议包括使用 Open Interpreter 以及创建如 `CLITOOL` 之类的自定义工具。
- **Langchain 集成查询**：用户询问了 Langchain 对 Microsoft Graph 的支持情况、在工作中使用类似 kappa-bot-langchain 的 API，以及使用 Langsmith 免费版时是否存在上传大小限制。
- **新进展与自定义工具讨论**：关于 GPT2 问题后 ChatGPT 响应变化的猜测不断出现，对话围绕在 Langchain 社区内创建和共享自定义工具展开。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://codepad.site/edit/x084ua3n">TestCode - CodePad</a>: 未找到描述</li><li><a href="https://developers.google.com/analytics/devguides/reporting/data/v1/api-schema">no title found</a>: 未找到描述</li><li><a href="https://learn.microsoft.com/en-us/graph/query-parameters?tabs=http">使用查询参数自定义响应 - Microsoft Graph</a>: Microsoft Graph 提供可选的查询参数，可用于指定和控制响应中返回的数据量。包括常用参数。</li><li><a href="https://python.langchain.com/docs/modules/tools/custom_tools/">定义自定义工具 | 🦜️🔗 LangChain</a>: 在构建自己的 Agent 时，你需要为其提供一个</li><li><a href="https://api.python.langchain.com/en/latest/memory/langchain.memory.entity.ConversationEntityMemory.html">langchain.memory.entity.ConversationEntityMemory &mdash; 🦜🔗 LangChain 0.1.17</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1235917336821891092)** (6 条消息): 

- **Java 加入 LangChain 家族**：**LangChain** 现在通过 [langchain4j](https://github.com/langchain4j/langchain4j)（LangChain 库的 Java 移植版）面向 Java 开发者开放，为 AI 助手工具集提供了扩展的应用生态系统。

- **Dragonfly 提升 LangChain 缓存能力**：**LangChain** 与高性能内存数据存储 **Dragonfly** 的集成展示了在聊天机器人上下文管理方面的显著改进，详情见新发布的 [博客文章](https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly)。

- **利用 Langchain 实现去中心化搜索**：一项新的去中心化搜索功能正在开发中，它利用用户拥有的索引网络来提供强大的搜索能力，开发者在最近的一条 [推文](https://twitter.com/indexnetwork_/status/1786110169396429093) 中记录了所有这些内容。

- **OpenGPTs-platform 亮相**：一个名为 [OpenGPTs-platform](https://github.com/OpenGPTs-platform) 的 GPT Store 开源替代方案已经发布，其特点是包含 'retrieval' 和 'web_retrieval' 等工具，演示视频已上传至 [YouTube](https://www.youtube.com/watch?v=yPdIEKb3jWc)。该项目旨在通过模块化方法复制并扩展 GPT Store 的功能，并通过 [OpenGPTs Discord](https://discord.gg/23aZEjyjp2) 与社区互动。

- **认识 everything-ai：全能 AI 助手**：更名后的 v1.0.0 **everything-ai** 本地助手提供从与 PDF 和模型聊天到总结文本和生成图像的一系列任务。这个多容器 Docker 应用程序专注于通用性和隐私，其功能和快速入门文档可在其 [GitHub 页面](https://astrabert.github.io/everything-ai) 上找到。

- **高级研究助手招募 Beta 测试人员**：招募 Beta 测试人员体验一个高级研究平台，该平台可访问包括 GPT-4 Turbo 和 Mistral Large 在内的多个 AI 模型。在 [Rubiks.ai](https://rubiks.ai/) 上使用代码 `RUBIX` 可免费获得两个月的高级版会员。该优惠包括为增强研究能力而量身定制的额外模型和工具。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/23aZEjyjp2)">Discord | 你的聊天与聚会场所</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里与你的朋友和社区聊天、聚会并保持紧密联系。</li><li><a href="https://www.dragonflydb.io/blog/efficient-context-management-in-langchain-with-dragonfly">在 LangChain 聊天机器人中使用 Dragonfly 进行高效上下文管理</a>：探索使用 Dragonfly 为 LangChain OpenAI 聊天机器人提供高效的上下文管理，通过缓存技术提升性能和用户体验。</li><li><a href="https://rubiks.ai/">Rubik's AI - AI 研究助手与搜索引擎</a>：未找到描述</li><li><a href="https://astrabert.github.io/everything-ai">everything-ai</a>：介绍 everything-ai，你功能完备、AI 驱动且本地运行的聊天机器人助手！🤖</li><li><a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: 介绍 everything-ai，你功能完备、AI 驱动且本地运行的聊天机器人助手！🤖</a>：Introducing everything-ai, your fully proficient, AI-powered and local chatbot assistant! 🤖 - AstraBert/everything-ai</li><li><a href="https://github.com/langchain4j/langchain4j">GitHub - langchain4j/langchain4j: LangChain 的 Java 版本</a>：LangChain 的 Java 版本。欢迎在 GitHub 上为 langchain4j/langchain4j 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1236245724430077963)** (2 messages): 

- **使用 Llama 3 的 RAG 技术**：一位用户分享了一个题为“[使用 SVM 且无需 Vectorstore 的 Llama 3 RAG](https://www.youtube.com/watch?v=vvW2dwvNm2Q)”的 YouTube 视频，提供了关于使用 **Llama 3** 配合相似度测量分类器进行 *Retrieval-Augmented Generation* (RAG) 且无需 Vectorstore 的见解。
- **探索将 LangGraph 作为 AgentExecutor**：另一项贡献是一个 [YouTube 视频](https://www.youtube.com/watch?v=UcD42NA2WoI)，展示了 **LangGraph** 与 **LangChain Core** 组件之间的对比，并提出了 AgentExecutor 实现方面的进展。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=UcD42NA2WoI">LangGraph 是 AgentExecutor 的未来吗？对比揭晓一切！</a>：🚀 深入探讨今天视频中的 AgentExecutor 实现，我将在其中展示 LangGraph 🦜🕸️ 与 LangChain Core 🦜🔗 组件之间的对比！🔧 什么是...</li><li><a href="https://www.youtube.com/watch?v=vvW2dwvNm2Q">使用 SVM 且无需 Vectorstore 的 Llama 3 RAG</a>：我们将探讨如何使用 Llama 3 Groq 和相似度分类器进行 RAG，而无需 Vectorstore。https://github.com/githubpradeep/notebooks/blob/m...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1236088423656456243)** (17 messages🔥): 

- **在 Clojure 中探索符号编程**：一位用户提到通过悬赏任务来熟悉 *tinygrad*，发现 **Clojure** 中的符号编程比 Python 更容易。
- **Julia 与 Clojure 之争**：一位成员认为 **Julia** 在符号编程方面优于 *Clojure*，并对其在 ML/AI 领域缺乏普及度表示惊讶。
- **寻求 tinygrad Bug 处理指导**：用户被引导使用 GitHub 的 issues 标签或 Discord 上的 bug 报告频道来报告 *tinygrad* 的 Bug。
- **理解 tinygrad 的 UOps 表示法的困难**：一位成员表示难以理解 *tinygrad* 的文本 UOps 表示法，并建议将其更改为更接近 LLVM IR 的格式以提高可读性，这引发了关于 phi 的格式和使用的讨论。
- **以静态单赋值 (SSA) 形式表示 UOps**：讨论继续解释了 UOps 作为 SSA 的一种形式，为什么 phi 位于块的末尾，并建议可能提交一个 Pull Request (PR) 来提出改进方案。
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.org/channels/1068976834382925865/1070745817025106080/1236700433003581541)** (12 messages🔥):

- **Tinygrad 在 Qualcomm GPU 上表现出色**：Tinygrad 通过在计算中使用 textures 和 pixel shaders，针对 Qualcomm GPU 进行了优化。正如 terafo 所解释的，整个代码库中分布着以 **image datatype** 进行的数据管理。
- **在 Qualcomm 上探索 Tinygrad**：在 Qualcomm 智能手机上运行 Tinygrad 是可行的，且不需要付出巨大努力，除非需要 **DSP support**，这会显著增加复杂性。
- **关于 Tinygrad 符号操作的见解**：一位成员分享了他们帖子的链接，该帖子详细分解了 Tinygrad 中的符号均值计算（symbolic mean computation），为其他使用或学习 Tinygrad 的人提供了清晰的见解。点击[此处](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/symbolic-mean.md)查看他们的解释。
- **Tinygrad 中的 CPU 操作是顺序的而非并行的**：George Hotz 确认 Tinygrad 是**单线程**的，在 CPU 计算期间不会发生并行线程操作。
- **对 Tinygrad 中 Tensor 操作的疑问**：Cappuchinoraro 询问了 `matmul` 函数的行为以及在 Tinygrad 操作中转置 Tensor 的影响。

**提到的链接**：<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/symbolic-mean.md">tinygrad-notes/symbolic-mean.md at main · mesozoic-egg/tinygrad-notes</a>：Tinygrad 教程。通过在 GitHub 上创建账号为 mesozoic-egg/tinygrad-notes 的开发做出贡献。

  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1236429992452292679)** (25 messages🔥): 

- **解决 json_schema 兼容性问题**：一位成员遇到了 `json_schema` 无法在 **llamafile 0.8.1** 上运行的问题；另一位成员建议使用 `--unsecure` 标志作为潜在的修复方案，并提到计划在即将发布的版本中解决此问题。

- **寻找轻量级模型**：发起了一场关于寻找能在低配置下运行的模型的讨论。推荐了 **phi 3 mini**，而当 phi 3 mini 运行速度太慢时，建议使用更小的模型 **Rocket-3B** 以获得更好的速度。

- **在 llamafile 中利用 ollama 缓存**：一位成员询问 **llamafile** 是否可以使用存储在 **ollama cache** 中的模型以防止重复下载，另一位成员确认如果 GGUF 文件受 llamafile 支持，这是可行的。

- **llamafile 与 AutoGPT 的集成**：讨论了一个关于将 **llamafile** 作为 LLM provider 集成到 **AutoGPT** 的 Pull Request 反馈请求。有人分享了设置此配置的说明链接 ([AutoGPT/llamafile-integration](https://github.com/Mozilla-Ocho/AutoGPT/tree/draft-llamafile-support/autogpts/autogpt/llamafile-integration))，正在等待维护者的回复，然后再进行进一步的代码编写。

- **识别并使用正确的本地模型**：在讨论澄清了哪些文件是实际模型、哪些是元数据后，一位用户成功地使用本地缓存的 **.gguf** 文件运行了 **llamafile**，展示了实时的故障排除和同行支持。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/Mozilla-Ocho/AutoGPT/tree/draft-llamafile-support/autogpts/autogpt/llamafile-integration">AutoGPT/autogpts/autogpt/llamafile-integration at draft-llamafile-support · Mozilla-Ocho/AutoGPT</a>：AutoGPT 是让每个人都能使用和构建 AI 的愿景。我们的使命是提供工具，让你专注于重要的事情。 - Mozilla-Ocho/AutoGPT</li><li><a href="https://github.com/Significant-Gravitas/AutoGPT/pull/7091">Draft llamafile support by k8si · Pull Request #7091 · Significant-Gravitas/AutoGPT</a>：背景：此草案 PR 是通过添加 llamafile 作为 LLM provider 来实现在 AutoGPT 中使用本地模型的一步。相关问题：#6336 #6947。变更 🏗️：有关此内容的完整文档.....
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1236622793127493643)** (7 messages): 

- **Mixtral Transformers Bug 影响性能**：有人指出 **mixtral transformers** 实现中存在 Bug，导致过去 Mixtral 的微调（finetunes）性能不佳。通过 [Twitter](https://twitter.com/kalomaze/status/1786869036946522256)、[Gist](https://gist.github.com/kalomaze/661b79095fdd91df8a84802f7cb6f26a) 以及 [GitHub 上的 Pull Request](https://github.com/huggingface/transformers/pull/30658) 分享了关键问题和关于此问题的进一步讨论。

- **对 Mixtral 问题范围的不确定性**：成员们质疑 Mixtral 的问题是仅限于 *training* 还是也影响 *generation*。目前尚未达成明确共识，强调了进一步澄清的必要性。

- **问题解决进行中**：一名成员提到了一场正在进行的对话，并指向了与另一位 Discord 用户的讨论，暗示目前正在努力定位并解决 Mixtral 的问题。然而，并未提供该对话的具体细节。

- **Bug 修复似乎陷入停滞**：一位成员对现状表示幽默，暗示他认为 Mixtral 一直以来就存在已知问题。这种插话反映出用户之间存在一种“问题早就在预料之中”的看法。

- **Pull Request 被拒绝增加了 Mixtral 的困惑**：提到的用于修复 Mixtral Bug 的 Pull Request 已被*关闭/拒绝*，这为这些问题的解决状态增添了另一层不确定性。此次拒绝对比 Mixtral 实现的影响未得到进一步讨论。
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1235956851133386872)** (3 条消息): 

- **量化版 LLaMA-3 的性能下降**：一篇 Reddit 帖子[讨论了量化对 LLaMA-3 的影响](https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/)，认为与 LLaMA-2 相比，LLaMA-3 的性能退化更为显著。一项关于 [LLaMA-3 低比特量化的研究](https://arxiv.org/abs/2404.14047)可能会为 LLM 压缩面临的挑战提供额外的见解。
- **Meta 忽略了 Chinchilla 的教训？**：一位成员指出，Meta 尽管有 *Chinchilla* 的教训，但仍采取扩展 LLaMA 的方法，这可能是为什么 LLaMA-3 模型在精度降低时信息损失更严重的原因。
- **修复补丁正在开发中**：一个 GitHub Pull Request 为 LLaMA-3 中观察到的量化问题提供了可能的修复方案，包括额外的统计数据和文档（[PR #6936](https://github.com/ggerganov/llama.cpp/pull/6936#issuecomment-2083214112)），以及围绕预分词 BPE 处理的讨论（[Issue #7088](https://github.com/ggerganov/llama.cpp/issues/7088#issuecomment-2094933215)）。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cetn9z/quantization_seems_to_hurt_the_quality_of_llama_3/">Reddit - 深入了解任何事物</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.14047">低比特量化 LLaMA3 模型的效果如何？一项实证研究</a>：Meta 的 LLaMA 家族已成为最强大的开源大语言模型（LLM）系列之一。值得注意的是，LLaMA3 模型最近发布，并在各项指标上取得了令人印象深刻的性能……</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6936#issuecomment-2083214112">perplexity：更多统计数据，由 JohannesGaessler 添加的文档 · Pull Request #6936 · ggerganov/llama.cpp</a>：我看到了一些主观报告，称量化对 LLaMA 3 的危害比对 LLaMA 2 更大。我决定对此进行调查，并为此在 pe... 中添加了更多统计数据（和文档）。</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7088#issuecomment-2094933215)">convert-hf-to-gguf-update.py 似乎无法工作 · Issue #7088 · ggerganov/llama.cpp</a>：Ubuntu 20.04, cudatoolkit12.2 GPU: Nvidia A100 24G RAM 10G(可用) 当我使用 llama.cpp 中的 'convert-hf-to-gguf-update.py' 将 'hf' 转换为 'gguf' 时，它既没有报告任何错误...
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1237016176429629501)** (3 条消息): 

- **揭晓当前使用的模型**：频道内的讨论透露，**8x22b Mistral** 是目前一位成员用于其任务的模型。未提供关于性能或具体应用细节的进一步信息。
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1236328226222116965)** (3 条消息): 

- **ElevenLabs 逼真 AI 语音背后的故事**：[《大西洋月刊》](https://www.theatlantic.com/technology/archive/2024/05/elevenlabs-ai-voice-cloning-deepfakes/678288/)的一篇文章详细介绍了名为 **ElevenLabs** 的初创公司如何开发出一些最令人信服的 AI 语音克隆技术。作者分享了使用该服务克隆自己声音的个人体验。

- **付费墙：现代的烦恼**：一位成员对遇到付费墙表示沮丧，表示无法阅读《大西洋月刊》关于 **ElevenLabs** 文章的完整内容。

- **ElevenLabs：疯狂的存在**：同一位成员对 **ElevenLabs** 的存在发表了评论，称这家初创公司能够创造出如此逼真的 AI 生成语音简直是“疯狂”。

**提到的链接**：<a href="https://www.theatlantic.com/technology/archive/2024/05/elevenlabs-ai-voice-cloning-deepfakes/678288/">ElevenLabs 正在构建一支语音克隆大军</a>：一家微型初创公司制造了一些最具说服力的 AI 语音。它的创造者们准备好迎接他们正在释放的混乱了吗？

---

**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1236002486280781904)** (2 条消息): 

- **论文跳过 RewardBench 评分**：一篇新发表在 [arXiv 上的论文](https://arxiv.org/abs/2405.01535) 忽略了报告 *RewardBench* 评分，因为结果不理想，这引发了一些学术界的冷嘲热讽，并配上了 <:facepalm:1207415956020797521> 表情符号。
- **引入 Prometheus 2 LM 用于无偏见评估**：该论文介绍了 **Prometheus 2**，这是一个开源的评估器语言模型，声称与人类和 **GPT-4 的判断**高度一致，并解决了影响专有 LM 的透明度、可控性和成本问题。
- **希望实现并测试 Prometheus 2**：一位成员表示渴望实现 **Prometheus 2**，以便通过实际演示来挑战和验证论文中的主张。

**提到的链接**：<a href="https://arxiv.org/abs/2405.01535">Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models</a>：GPT-4 等专有 LM 通常被用来评估各种 LM 的响应质量。然而，出于对透明度、可控性和成本的担忧，强烈促使了...

---

**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1236040686512377937)** (2 条消息): 

- **势不可挡的成功令人侧目**：一位成员用 *“他不能一直这样逍遥法外”* 的措辞表达了惊讶和一丝担忧。
- **关于 John 回复的不确定性**：另一位成员反思了与 *“john”* 的对话，强调了对一个提议的模棱两可的回答，并评论道：*“该死，所以这就是为什么 john 只对我说了个也许，哈哈。”*

---

**Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1235969401660903585)** (4 条消息): 

- **思考经典 RL 中的未知领域**：一位成员询问是否有关于经典 RL 某个特定方面的研究，引发了好奇心，暗示了潜在的知识空白或未来探究的领域。
- **Value Function：不同方法中的可能关键**：另一位成员建议探索 **PPO value function** 与 **DPO 的 credit assignment** 之间的联系，暗示这可能会在 Reinforcement Learning 策略中产生有趣的见解。 
- **Value Function 在 Planning 中的重要性**：后续讨论强调了 Value Function 的重要性，特别是在 Planning 而非经典 Reinforcement Learning 的背景下，突出了其关键作用。

---

**LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1237119509165248593)** (7 条消息): 

- **探索 Anthropic 的 Prompt 生成器**：提到在 **Anthropic console** 中可以使用一个新的 **prompt generator tool**。
- **礼貌改写的结果**：一位成员测试了该工具，要求它 *用更礼貌的语言改写一个句子*，并分享说结果 *还不错*。
- **解码 System Prompt**：正在进行从工具中提取 System Prompt 的工作，其中 **k-shot examples** 是重要组成部分，包括一个著名的 *苏格拉底式数学导师* 示例。
- **提取的数据不完整**：尝试提取的成员报告说，Prompt 内容非常广泛，以至于在中间被截断了，特别是在冗长的数学导师示例部分。
- **承诺分享完整 Prompt**：该成员承诺一旦成功提取并整理好，将在这里分享完整的 System Prompt。

---

**Skunkworks AI ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/1236350458663141386)** (1 条消息): 

- **寻找虚构数据**：一位成员表示需要一个**充满虚假信息的数据集**，目的是在 **Llama 3** 和 **Phi3** 等模型上实验 Fine-tuning 技术。他们表示，即使是完全虚假的数据对于他们的研究也是可以接受的。

---

**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1235948611292893263)** (2 条消息):

- **提供快速算力资助**：一位成员为具有启发性的 Skunkworks AI 项目提供**快速算力资助 (fast compute grants)**，表达了支持创新的热忱。支持详情见 [推文](https://twitter.com/PrimeIntellect/status/1786386588726960167)。
- **分享 AI 视频资源**：分享了一个与人工智能相关的 YouTube 视频链接，作为社区成员的潜在资源或关注点。视频可以在[这里](https://www.youtube.com/watch?v=vvW2dwvNm2Q)观看。
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1236429191092899941)** (3 messages): 

- **LLM 在错误摘要方面表现出色**：一位成员分享了一种使用 LLM 总结错误的有效方法；他们提供了一个通过管道（pipe）传输到 LLM 的 `conda activate` 命令示例。建议将其包含在 [LLM README](https://github.com/simonw/llm/blob/main/README.md) 中。

- **利用 LLM 进行错误评估的 Bash 函数**：提出了一个新的 `llm-err` bash 函数，通过将命令输出直接通过管道传输到 LLM 来帮助评估错误。该函数接受一个命令作为参数，并使用 LLM 来指明遇到的任何错误的具体原因。
  

---



**Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/1236060159990566942)** (2 messages): 

- **向德克萨斯州奥斯汀社区致意**：一位成员向位于**德克萨斯州奥斯汀 (Austin, TX)** 的所有人发出了友好的问候。
- **处于融资阶段的法国 AI 初创公司**：来自法国的 **Vivien** 介绍了 **Finexov** ([Finexov](https://www.finexov.com/))，这是一个简化 **R&D** 资助机会识别和申请生成的 AI 平台。该平台已经发布，并建立了合作伙伴关系，获得了 **Founder Institute** ([FI.co](https://fi.co/)) 的支持。
- **寻找 CTO 联合创始人**：Vivien 正在寻找一位具有深厚 **ML** 背景、并有志于建立和领导团队的 **CTO 联合创始人**。潜在的 CTO 应常驻欧洲或中东，具备法语能力者优先，并准备好进行包括融资在内的高强度工作。
- **迪拜会面机会**：6 月初在**迪拜 (Dubai)** 有见面机会，Vivien 邀请感兴趣的人士联系并进行潜在的交流。

**提到的链接**：<a href="https://fi.co/">Founder Institute: 全球最大的种子前初创企业加速器。</a>：未找到描述。

  

---



**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1236029165447413770)** (2 messages): 

- **探索新高度**：AI21 Labs 的员工在谈到其技术的某些方面时表示：“我们仍在探索，但我们可以达到更高的高度”，并邀请社区成员通过私信讨论他们的使用案例和想法。
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1235947724063375370)** (1 messages): 

- **提供快速算力资助**：一位成员分享了一篇 [Twitter 帖子](https://twitter.com/PrimeIntellect/status/1786386588726960167)，宣布为有需要的人提供**快速算力资助 (fast compute grants)**。该推文似乎是在征集获取算力资源的申请或提名。