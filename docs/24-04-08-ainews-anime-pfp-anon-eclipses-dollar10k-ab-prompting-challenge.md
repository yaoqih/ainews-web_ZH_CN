---
companies:
- openai
- ollama
- huggingface
date: '2024-04-09T01:18:42.938105Z'
description: '**Victor Taelin** 发起了一项针对 GPT 模型的 1 万美元挑战赛。最初，最先进的模型仅达到了 **10% 的成功率**，但在社区的努力下，48
  小时内成功率便突破了 **90%**，这凸显了 GPT 的潜力以及目前普遍存在的技能短板。在 Reddit 的 AI 社区中，**Command R Plus
  (104B)** 已能通过 **Ollama** 和 **llama.cpp** 的分支版本在 **M2 Max 硬件**上运行量化版，相关的 **GGUF 量化模型**也已在
  Huggingface 上发布。**st2v** GitHub 仓库现已支持流式文本生成视频。**WD Tagger v3** 正式发布，配备 WebUI，可用于对数据集进行大规模自动打标。在关于
  OpenAI 的讨论中，一些较冷门的提示词技术（如自打标和生成式框架）产生了极具启发性的结果，其中包括对自我进化系统提示词的实验。Stable Diffusion
  用户讨论了图像构图在训练角色 LoRA 中的重要性，以及生成游戏角色的最佳底模（Checkpoints）。讨论内容还涵盖了 **50 亿（5B）参数模型**的稀缺性，以及开源
  AI 的“类开源”许可证问题。网络热梗方面，则出现了不少调侃 ChatGPT 和 Gemini 训练数据差异的笑话。'
id: 56ec05bf-3c4f-454a-aa7c-26178a8809fd
models:
- command-r-plus-104b
- stable-diffusion-1.5
original_slug: ainews-anime-pfp-anon-eclipses-10k-ab-prompting
people:
- victor-taelin
- futuristfrog
title: 一位动漫头像的匿名用户在 1 万美元的 A::B 提示词（prompting）挑战中刷新了纪录。
topics:
- quantization
- model-optimization
- streaming
- prompt-engineering
- self-prompting
- image-composition
- character-lora-training
- model-size
- open-source-licenses
- memes
- humor
---

 

他最初使用所有 SOTA 模型进行的尝试仅获得了 10% 的成功率。社区提交的方案达到了 56%。[又过了一天，@futuristfrog 将成功率提升至 90% 以上。](https://twitter.com/victortaelin/status/1777049193489572064) 挑战总共持续了 48 小时。这是一个关于 GPT 能力的有趣教训，也再次提醒我们：在 2024 年 AGI 之前的 AI 时代，无法完成某事通常只是单纯的“水平问题”（skill issue）。

---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取尚未实现，但即将推出。

**技术进展与发布**

- **Command R Plus (104B) 已适配 Ollama**：在 /r/LocalLLaMA 中，Command R Plus (104B) 已通过 fork 版本的 llama.cpp 适配 Ollama，允许[**量化模型在 M2 Max 硬件上运行**](https://www.reddit.com/r/LocalLLaMA/comments/1bymeyw/command_r_plus_104b_working_with_ollama_using/)。
- **Command R+ 104B 的 GGUF 量化版发布**：在 /r/LocalLLaMA 中，Dranger 在 Huggingface 上发布了 [**Command R+ 104B 从 1-bit 到 8-bit 的 GGUF 量化版本**](https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF)。
- **Streaming t2v 现已可用**：在 /r/StableDiffusion 中，Streaming t2v 现已可用，允许[**使用 st2v GitHub 仓库生成更长的视频**](https://www.reddit.com/r/StableDiffusion/comments/1by5upa/streaming_t2v_is_now_avaible/)。
- **WD Tagger (v3) 新版本发布**：在 /r/StableDiffusion 中，新版本的 WD Tagger (v3) 已发布，用于[**利用 WebUI 界面对数据集进行批量自动打标（captioning）**](https://www.reddit.com/r/StableDiffusion/comments/1by0zsg/mass_auto_caption_with_wd_tagger_v3_with_webui/)。

**技术与 Prompting**

- **较冷门的 Prompting 技术产生发人深省的输出**：在 /r/OpenAI 中，通过使用[**自打标签输出、生成式框架和实时自查等较冷门的 Prompting 技术**](https://www.reddit.com/r/OpenAI/comments/1by9uo8/thought_provoking_outputs_via_lesser_known/)，生成了发人深省的内容。
- **自我演进 System Prompt 的实验**：在 /r/OpenAI 中，一项让 OpenAI API 在多次迭代中编写自己的 System Prompt 的实验导致了[**词藻日益华丽且宏大的措辞**](https://www.reddit.com/r/OpenAI/comments/1byijwt/letting_openai_api_write_its_own_system_prompt/)。
- **无 Prompt 扩图/局部重绘画布更新**：在 /r/StableDiffusion 中，无 Prompt 扩图/局部重绘画布已更新，可[**在低端硬件上运行 ComfyUI 工作流**](https://v.redd.it/xi2hkxh4l4tc1)。

**问题与讨论**

- **训练角色 LoRA 时图像构图的重要性**：在 /r/StableDiffusion 中，有一场关于[**训练角色 LoRA 时图像构图的重要性，以及自动打标是否能充分捕捉细节**](https://www.reddit.com/r/StableDiffusion/comments/1byibwu/how_important_is_image_composition_when_training/)的讨论。
- **Stable Diffusion 1.5 中最适合游戏角色的 Checkpoint**：在 /r/StableDiffusion 中，有人询问[**在 Stable Diffusion 1.5 中生成视频游戏角色的最佳 Checkpoint**](https://www.reddit.com/r/StableDiffusion/comments/1by0nnk/which_checkpoint_is_best_for_video_game_characters/)。
- **5B 参数模型的稀缺性**：在 /r/LocalLLaMA 中，有人询问[**为什么与 3B 和 7B 相比，5B 参数的模型如此之少**](https://www.reddit.com/r/LocalLLaMA/comments/1bybtky/why_are_there_so_few_5b_models/)。
- **准开源（Open-ish）许可证与对齐开源 AI 的激励机制**：在 /r/LocalLLaMA 中，有一场关于[**准开源许可证以及需要哪些条款来对齐开源 AI 激励机制**](https://www.reddit.com/r/LocalLLaMA/comments/1bymr57/openish_licenses_recap_and_discussion/)的讨论。

**梗与幽默**

- **关于跳舞动漫妹子与“写实”《使命召唤》的幽默帖子**：在 /r/StableDiffusion 中，有一个幽默帖子吐槽[**跳舞的动漫妹子太多了，并配以一张“写实”的《使命召唤》图像作为反击**](https://i.redd.it/av8h3pwy16tc1.png)。
- **关于 ChatGPT 与 Gemini 训练数据的笑话**：在 /r/ProgrammerHumor 中，有一个笑话证实了 [**ChatGPT 是用 YouTuber 的数据训练的，而 Gemini 则不是**](https://www.reddit.com/gallery/1by68qc)。

# AI Twitter 回顾

> 所有总结均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类（clustering）和流程工程（flow engineering）。

**AI 与机器人研究进展**

- **AI 与机器人进展**：[@adcock_brett](https://twitter.com/adcock_brett/status/1777004161416020407) 分享了 AI 与机器人领域最重要的研究与开发的每周回顾，强调了该领域飞速发展的步伐。
- **传闻中 GPT-5 的能力**：[@bindureddy](https://twitter.com/bindureddy/status/1777023216810438900) 报道称，OpenAI 即将推出的 GPT-5 模型传闻拥有极其强大的代码编写、推理和语言理解能力，超越了 Anthropic 的 Claude 3。
- **用于生成音乐视频的 Sora**：[@gdb](https://twitter.com/gdb/status/1777127364822024283) 展示了 Sora，这是一款允许用户通过生成相应的音乐视频，来视觉化呈现歌曲一直以来“看起来”是什么样子的工具。
- **4-bit Mistral 7B 的高速性能**：[@awnihannun](https://twitter.com/awnihannun/status/1777072588633882741) 在 M2 Ultra 芯片上运行 4-bit Mistral 7B 模型，达到了令人印象深刻的 **103.5 tokens-per-second**。
- **Many-shot jailbreaking 技术**：[@adcock_brett](https://twitter.com/adcock_brett/status/1777004446469230651) 分享了 Anthropic 研究人员发现的一种名为 “many-shot jailbreaking” 的技术，该技术可以通过利用扩展的上下文窗口（context windows）来规避大语言模型的安全护栏。

**AI Agent 与机器人**

- **构建 AI Agent 的复杂性**：[@bindureddy](https://twitter.com/bindureddy/status/1777136946705539363) 指出，构建 AI Agent 的工作中只有 10% 与 LLM 和推理有关，而剩下的 90% 涉及代码、数据、内存、评估和监控等繁重工作。
- **OpenAI 的计划与机器人领域中的 LLM**：[@adcock_brett](https://twitter.com/adcock_brett/status/1776816987202867673) 概述了 OpenAI 的计划，并讨论了为什么大语言模型对机器人应用至关重要。
- **可靠的基于 LM 的 Agent 的关键因素**：[@sarahcat21](https://twitter.com/sarahcat21/status/1776644684997365817) 强调，有针对性的预训练和接口设计对于构建基于大语言模型的可靠 Agent 至关重要。
- **编程 Agent 的增长**：[@mbusigin](https://twitter.com/mbusigin/status/1776377605555454028) 强调了编程 Agent 在开发和采用方面的爆发式增长。
- **Figure-01 人形机器人**：[@adcock_brett](https://twitter.com/adcock_brett/status/1776672870816739369) 分享了 Figure-01 机电人形机器人的图像。

**LLM 进展与能力**

- **Grok 2.0 传闻性能**：[@bindureddy](https://twitter.com/bindureddy/status/1777378250962129012) 报道称，传闻 Grok 2.0 是继 Anthropic 的 Claude Opus 之后第二个在性能上超越 OpenAI GPT-4 的模型，这对 Grok 和 X 来说将是一个重大成就。
- **Claude 3 Opus 表现优于 GPT-4**：[@Teknium1](https://twitter.com/Teknium1/status/1777117967802871858) 和 [@bindureddy](https://twitter.com/bindureddy/status/1777023216810438900) 指出，Anthropic 的 Claude 3 Opus 模型在某些任务上的表现优于 GPT-4。
- **新模型发布**：[@osanseviero](https://twitter.com/osanseviero/status/1776620683465764936) 宣布发布 Cohere Command R+、Google Gemma Instruct 1.1 以及 Qwen 1.5 32B 模型系列。

**检索增强生成 (RAG) 架构**

- **使用 LangChain 和 Yahoo Finance 构建金融 Agent**：[@llama_index](https://twitter.com/llama_index/status/1777076087853392027) 展示了如何使用 LangChain 和 Yahoo Finance 构建金融 Agent，涵盖了资产负债表、损益表、现金流量和投资建议等股票分析功能。
- **使用 LlamaIndex 构建多文档 Agent**：[@llama_index](https://twitter.com/llama_index/status/1776627066126901311) 和 [@jerryjliu0](https://twitter.com/jerryjliu0/status/1776971813874028694) 展示了如何使用 LlamaIndex 将文档视为子 Agent，进行语义搜索和摘要。
- **RAG 的 Agent 化扩展**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1776971813874028694) 提出了一种检索增强生成的 Agent 化扩展，将文档视为工具和 Agent，以实现超越固定分块（chunks）的动态交互。
- **为 RAG 提取文档知识图谱**：[@llama_index](https://twitter.com/llama_index/status/1777348428755820849) 展示了如何使用 LlamaParse 提取结构化 Markdown，将其转换为文档图谱，并存储在 Neo4j 中，以进行高级查询并驱动 RAG 流水线。

**迷因与幽默**

- **超时设置成了秒而不是毫秒**：[@gdb](https://twitter.com/gdb/status/1776824716931838227) 分享了一个幽默的梗图，关于不小心将超时时间设置为秒而不是毫秒。
- **相比 Shake Shack 更喜欢 In-N-Out**：[@adcock_brett](https://twitter.com/adcock_brett/status/1777131105566740830) 开玩笑说，在纽约生活过后，他更喜欢 In-N-Out 而不是 Shake Shack。
- **睡眠不足后的生物神经网络性能**：[@_jasonwei](https://twitter.com/_jasonwei/status/1777088156443279469) 幽默地将熬夜后的生物神经网络性能比作提示词（prompting）很差的 GPT-4 base。
- **承认 Claude 解决了问题的痛苦**：[@Teknium1](https://twitter.com/Teknium1/status/1776820170348171298) 分享了一个梗图，关于不得不承认 Anthropic 的 Claude 模型解决了某个问题时的痛苦。
- **刷了 3 个月 LeetCode 却没找到工作**：[@jxnlco](https://twitter.com/jxnlco/status/1777095850172268903) 分享了一个关于刷了 3 个月 LeetCode 却没找到工作的挫败感的梗图。

---

# AI Discord 摘要

> 摘要之摘要的摘要

**1. LLM 的量化与优化突破**

- **[QuaRot](https://arxiv.org/abs/2404.00456)** 实现了对 LLaMa2-70B 等大语言模型的端到端 **4-bit 量化**，在处理离群值的同时保持计算不变性，且性能损失极小。[HQQ](https://github.com/mobiusml/hqq) 也展示了与 **gpt-fast** 集成的 4-bit 量化成果，效果显著。

- **无调度优化（Schedule-Free Optimization）受到关注**：Meta 为 AdamW 和 SGD 开发的 **无调度优化器** 已被[集成到 Hugging Face 的 transformers 库中](https://github.com/huggingface/transformers/pull/30079)，这可能会彻底改变模型训练方式。讨论围绕 [PyTorch 的 Schedule-Free Optimization 仓库](https://github.com/facebookresearch/schedule_free)以及 [Aaron Defazio 发布的关于该主题的 Twitter 线程](https://twitter.com/aaron_defazio/status/1773381393831067787)展开。

- 关于 **torch.compile** 的讨论集中在：它仅针对 CUDA 输入使用 **Triton** 内核、异步集合通信操作、DeepSpeed 集成，以及使用 **tiny-cuda-nn** 或 CUTLASS 进行潜在的 MLP 优化。

**2. 扩展上下文长度与注意力机制**

- **[EasyContext](https://github.com/jzhang38/EasyContext)** 项目引入了内存优化和训练方案，通过在 8 张 A100 GPU 等适度硬件上使用 **ring attention**，将语言模型的上下文长度外推至 **100 万 token**。[Zhang Peiyuan 的推文](https://twitter.com/PY_Z001/status/1776176932687892796)讨论了增加上下文窗口对训练吞吐量的影响。

- **[Mixture-of-Depths](https://arxiv.org/abs/2404.02258)** 提出在固定预算内动态分配 Transformer 序列中的计算资源，在不牺牲灵活性的情况下提升效率。

- 讨论还涉及了 **线性注意力（linear attention）** 与经典注意力的对比、**变长条纹注意力（variable length striped attention）** 的实现，以及分布式计算场景下 ring attention 的速度与内存权衡。

**3. 开源 AI 进展与社区参与**

- **AMD** 宣布开源 Radeon GPU 的微引擎调度器（MES）固件和文档，这符合社区欢迎的更广泛的开源 GPU 努力。([The Register 文章](https://www.theregister.com/2024/04/05/amd_mes_open_source/), [AMD Radeon 推文](https://twitter.com/amdradeon/status/1775999856420536532))

- **[PaperReplica GitHub 仓库](https://github.com/hegdeadithyak/PaperReplica)** 旨在通过社区贡献来复现 AI/ML 研究论文，促进知识共享和技能提升。

- **text-generation-inference (TGI)** 的许可协议更改为 Apache 2，在项目完全开源后引发了贡献者的激增，凸显了像 **Mistral** 这样的开源生态系统的潜在经济效益。

- Cohere 的 **[Command R+](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus)** 展示了对中古高地德语等古老语言令人印象深刻的翻译能力，表现优于 GPT-4 级别的模型，并燃起了开发者对其开源发布以推动参与的希望。

**4. 多模态 AI 进展与应用**

- **Aurora-M** 项目根据美国关于 AI 的行政命令，推出了一个新的 [155 亿参数开源多语言语言模型](https://arxiv.org/abs/2404.00399)，展示了在 2 万亿训练 token 中，单语言安全对齐对跨语言的影响。

- Unsloth AI 在使用 **Chat GPT** 和 **Claude** 等模型将图像准确转换为 HTML（同时保留颜色和边框）时面临挑战，引发了使用 ASCII 艺术作为替代方案的调侃建议。

- **[BrushNet](https://www.youtube.com/watch?v=X89IQop_0dM)**，一种结合了目标检测的 AI 图像修补（inpainting）新方法，正如教程视频所示，它承诺提供更高质量的结果。

- **LLaVA** 视觉语言模型经历了一次新颖的“罗夏墨迹测试（Rorschach test）”，通过向其输入随机图像嵌入（embeddings）并分析其解读，详情见 [一篇博客文章](https://anotherjesse.com/posts/llava-rorschach/)。此外，专为边缘设备设计的紧凑型 **nanoLLaVA** 模型也已在 [Hugging Face 上发布](https://huggingface.co/qnguyen3/nanoLLaVA)。


**5. 杂项**

- **Tinygrad 开发进展：回滚与集成**：**George Hotz** 回滚了 tinygrad 中的命令队列，并正在使用 **multidevicegraph 抽象**将 **内存调度器（memory scheduler）** 直接与现有的调度器模型集成。为了帮助贡献者，官方分享了 [TinyJit 教程](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/jit.md) 和 [多 GPU 训练指南](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/multigpu.md)。

- **Jamba 模型为受限硬件提供替代方案**：通过对专家权重进行 **球面线性插值（Slerp）**，开发人员创建了 **Jamba** 架构的缩减版本，包括 [29B 参数的 8xMoE](https://huggingface.co/isemmanuelolowe/Jamba-8xMoE_Slerp) 和 [17.7B 参数的 4xMoE](https://huggingface.co/isemmanuelolowe/Jamba-4xMoE_Slerp)，从而实现在 4090 GPU 等更易获取的硬件上以 4-bit 精度进行微调。

---



# 第一部分：Discord 高层摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**iOS 用户试用新故事发现功能**：Perplexity AI 正在 **iOS 上试用一种创新的故事发现格式**。鼓励用户通过指定频道反馈体验，并可在此处下载测试版应用 [here](https://pplx.ai/download)。

**AI 活动圆满结束**：Perplexity AI Discord 活动落下帷幕，*eun08000* 和 *codelicious* 并列第一。获奖者将收到包含详情的私信。

**Claude 3 Opus - 模型辩论**：在服务器上，讨论围绕 Perplexity 实现的 **Claude 3 Opus 模型**与其他版本的差异展开，特别是在需要创造力的任务方面。

**API 的奇特之处与疑问**：用户注意到 Perplexity 的 API 与 Web 应用程序之间存在不一致，API 表现出更多的幻觉（hallucinations）；API 的默认模型与 [Web 版本](https://pplx.ai) 不同。建议 API 用户使用 'sonar-medium-online' 模型，以紧密模拟非 Pro 用户通过 Web 应用访问的 *Sonar* 模型。

**科技爱好者分享与学习**：用户交流了从 AI 如何影响音乐产业到 Tesla 和 Apple 最新技术创新的各种话题。此外，一项关于 Perplexity AI 的案例研究强调，在 Amazon Web Services 的支持下，模型训练速度提升了 40%，展示了 Perplexity 对先进机器学习基础设施和技术的高效利用。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI Vision Models 的罗夏墨迹测试**：**LLaVA vision-language model** 接受了一项新颖的“罗夏墨迹测试”，通过向其输入随机图像嵌入并分析其解读，详情见 [blog post](https://anotherjesse.com/posts/llava-rorschach/)。此外，适用于边缘设备的紧凑型 **nanoLLaVA model** 已在 [Hugging Face](https://huggingface.co/qnguyen3/nanoLLaVA) 上发布。

- **Claude 的记忆机制受到质疑**：关于 **Claude** 是否在不同会话间保留信息，还是这种记忆表象仅源于概率建模，引发了技术讨论。工程师们辩论了当前模型在应对持久 context 挑战时的有效性。

- **Worldsim 的困扰与智慧**：在遭受 DDoS 攻击后，有人提议为 **Worldsim** 建立登录系统以抵御未来威胁，并讨论了包含更多场景的“专业版”。同时，围绕类似于观察到的现实的潜在 AI 驱动模拟展开了哲学思考。

- **RAG 多样性的 Chunking 策略**：有人建议使用 *chunking script* 为 RAG 预生成多样化的数据集，同时讨论了使用 **Claude Opus** 创建复杂的跨领域查询。关于数据来源的伦理问题也浮出水面，特别是使用来自勒索软件攻击的泄露文档，这与用于数据集策划的 **RAPTOR** 等聚类策略形成了对比。

- **GitHub 与 Hermes 的融合**：一个 GitHub 仓库 **VikParuchuri/marker** 因其高精度的 PDF-to-markdown 转换功能而受到关注，详情见 [GitHub - VikParuchuri/marker](https://github.com/VikParuchuri/marker)。此外，讨论集中在增强 `Hermes-2-Pro-Mistral-7B` 以通过 `tools` 配置执行函数，这一障碍与在各种 LLM 环境中进行全参数 finetuning 与 adapter training 所面临的挑战不相上下。

- **加拿大的 AI 雄心与企业级 LLM**：从 Cohere 为企业推出的可扩展 LLM **Command R+**，到对加拿大争夺全球 AI 领导地位战略的见解，讨论扩展到了理解 SSL 证书、创建类似于 Google Image Search 的本地解决方案，以及理清过剩的 AI 研究与合成。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability Bot 离线**：寻求图像生成服务的用户被引导检查机器人状态（由于宕机），并被推向其他服务器频道以获取更新和支持。
- **图像生成的质量追求**：出现了比较本地模型输出与 Dreamstudio 输出的辩论，参与者推荐了开源 upscalers，并讨论了各种图像增强技术的有效性。
- **SD3 热度攀升**：Stable Diffusion 3 (SD3) 有一个非正式的 2-4 周 ETA，引发了关于该模型预期改进和新功能的讨论。
- **LoRa 训练对话**：关于 LoRa 训练的信息交流中，用户在寻求安装建议，并引用 GitHub 仓库以获取实用的训练方法。
- **用户界面升级**：关于用户界面增强的讨论包括从 Automatic 1.1.1.1 迁移到 StableSwarm 的建议，重点关注易用性以及新用户的特性可访问性。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**HTML 转换令工程师头疼**：AI 工程师讨论了当前语言模型（如 **Chat GPT** 和 **Claude**）在将图像准确转换为 HTML 方面的局限性，导致颜色保真度丢失和圆角边框失效。一项诙谐的提议建议使用 ASCII 艺术作为替代方案，这源于其诱导 AI 模型响应的能力，正如这篇 [Ars Technica 文章](https://arstechnica.com/security/2024/03/researchers-use-ascii-art-to-elicit-harmful-responses-from-5-major-ai-chatbots/) 所展示的那样。

**Aurora-M 照亮可能性**：一个拥有 155 亿参数的开源多语言模型 **Aurora-M** 被推出，并因其跨语言安全能力引起了社区关注，详见[这篇论文](https://arxiv.org/abs/2404.00399)。研究结果表明，一种语言的安全对齐可以对其他语言产生积极影响。

**Jamba Juice 还是 Mamba Sluice？投资观点发生碰撞**：工程师们辩论了对 AI21 Labs 的 **Jamba** 的投资，特别是考虑到 [TechCrunch](https://techcrunch.com/2023/08/30/generative-ai-startup-ai21-labs-lands-155m-at-a-1-4b-valuation/) 报道的其近期 1.55 亿美元的融资。尽管模型的前期成本较高，但针对性模型微调的投资回报率（ROI）被揭示出来，呈现出乐观的前景。

**AI 微调观点交汇与分歧**：社区就微调方法进行了深入交流，提到了如 **GGUF** 等无监督微调技术，以及动态位置偏移（DPO）的优势。讨论了微调的具体策略以及应用 **LoRA** 等技术来提升性能。

**私有 AI 托管热潮**：数据隐私担忧促使成员将他们的 AI 项目托管在个人服务器上，并有独立使用 [Hircoir TTS](https://tts.hircoir.eu.org/) 等平台的案例。一些设想的未来计划包括整合广告，以利用不断增长的模型组合获利。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**提升你的模型性能**：**LM Studio** 似乎领先于 **oogabooga** 和 **Faraday** 等替代方案，其 GUI 因更高质量的输出而赢得用户青睐。关于扩展的建议纷至沓来，特别是对文件读取支持以及文本转图像、文本转语音等模态的支持；这些功能正趋近于 **Devin** 已经提供的功能，旨在增强创造力和生产力。

**大思想家，大模型**：技术群体提倡处理重量级模型（如 104B 的 **Command R+**）的实力博弈，并建议为较旧但庞大的模型配备更强劲的硬件（如 Nvidia P40）。围绕 VRAM 的讨论延伸到了优化多 GPU 设置的策略，暗示使用 RTX 4060 Ti 和 GTX 1070 来分摊计算负载，并利用 **Tesla P40 GPUs**，尽管可能存在过时的 **CUDA** 问题。

**模型流畅运行的喜悦**：在 **ROCM** 和 **ROCm Preview Beta** 方面，关于 GPU 支持的讨论非常热烈，包括使用 AMD 的 RX 5000 和 6000 系列芯片。用户标记了 **ROCm 0.2.19 Beta** 上的 "exit 42" 错误，并围绕调试版本寻求解决方案，展现了社区协作精神。同时，关于 Intel **Advanced Matrix Extensions (AMX)** 的传闻引发了对 LM Studio 如何利用这种强大处理能力的猜测。

**挖掘模型珍宝**：公告中涌现出大量共享资源和模型，包括 **Starling-LM 7B**、**c4ai command r v01** 和 **stable-code-instruct-3b** 等。可访问性被放在首位，大家共同推动在 **Hugging Face** 上建立社区页面，最新的 **GGUF quants** 在那里大放异彩，吸引 AI 爱好者尝试 **Google's Gemma 1.1 2B** 等产品，并关注即将推出的 7B 变体。

**塑造视觉模型格局**：一位成员询问关于训练 **LLMs** 以破译股市 **OHLC** 模式的问题，同时赞扬了 **LM Studio** 在视觉模型实现中的效用，这激发了探索技术与金融之间复杂交织如何通过 AI 优雅编排的火花。**Hugging Face** 上视觉模型的揭晓反映了社区随时准备捕捉并将这种概念美学转化为实际应用的积极态度。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Gradio 的 API Recorder 和 Chatbot UI 修复准备发布**：Gradio 4.26.0 版本引入了 **API Recorder**，可将交互转换为代码，并解决了与页面加载时间和 Chatbot UI 崩溃相关的关键 Bug。更新详情见 [Gradio Changelog](https://www.gradio.app/changelog#4-26-0)。

**对 LLM 的担忧日益增加**：安全问题成为焦点，挑战 LLM 伦理限制的新方法“Crescendo”以及 Cohere 的 Command-R-plus 中的漏洞被曝光。同时，Mixture-of-Depths (Modes) 提案和 llamaindex 博客为模型效率和信息检索提供了创新解决方案。

**NLP 社区在 SageMaker、PDF ChatGPT 需求及挑战中前行**：社区讨论了在 SageMaker 上部署模型、为 PDF 定制 ChatGPT，并对 Gemini 1.5 的 10M 上下文窗口表示关注。寻求解决方案的开发者面临多 GPU 训练故障，并要求在使用 Hugging Face 库时提供 Token 计数信息。

**蓬勃发展的 AI 贡献与对话库**：HybridAGI 在 GitHub 上的神经符号行为编程欢迎同行评审，Hugging Face 读书小组将其集体智慧存档在 [GitHub](https://github.com/isamu-isozaki/huggingface-reading-group)。PaperReplica 的开源邀请和支持 RAG 的 llamaindex 成为协作学习和资源共享的典范。

**视觉及其他领域**：计算机视觉频道的对话涉及 HuggingFace 作为模型库的效用、不同 Transformer 模型（如 XCLIP）的功效，并解决了使用 Hugging Face 'datasets' 库进行 parquet 文件操作的实时挑战。同时，征集将 Diffusion 模型应用于视频增强的资源，体现了该领域活跃的探索精神。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 崛起：深入探讨特殊函数与 SICP 改编**
- Mojo 社区正在展现其技术实力，通过更新 Specials 软件包深入研究专门的 **数学函数**，并将著名的《计算机程序的构造和解释》（SICP）文本移植到 Mojo。用户现在可以在 [Specials package](https://github.com/leandrolcampos/specials) 中找到数值精确的函数，如 `exp` 和 `log`，并通过 [mojo-packages](https://github.com/kernhanda/mojo-packages) 等仓库参与协作算法和软件包共享。

**MAX 与 AWS 结盟；开源文档驱动**
- Modular 宣布与 [AWS 建立战略联盟](https://www.modular.com/blog/modular-partners-with-amazon-web-services-aws-to-bring-max-to-aws-services)，旨在将 MAX 与 AWS 服务集成，并在全球范围内扩展 AI 能力。Mojo 语言正准备加强协作，呼吁社区为 [Mojo 标准库文档](https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide)做出贡献。

**Discord 动态：Python 互操作性与贡献 Mojo 的成长**
- Mojo 社区正积极讨论元编程能力、编译时求值复杂性以及 `Reference` 类型中的生命周期。他们正在通过实现基本函数来探索 Python 互操作性的路径，并邀请贡献者参与 GitHub 上的 "good first issues"，以 [Mojo's Changelog](https://docs.modular.com/mojo/changelog#week-of-2023-01-30) 和 [贡献指南](https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md) 作为起点。

**Var vs. Let - Mojo 参数传奇**
- 一次对话显示，虽然 `let` 可能已从 Mojo 中移除，但 `var` 仍保留用于延迟赋值的变量，详情见 [Mojo Manual](https://docs.modular.com/mojo/manual/variables#declared-variables)，为用户提供了更多知识。此外，将 Mojo 注入 Web 开发的努力正在汇聚，随着 [lightbug_http](https://github.com/saviorand/lightbug_http) 的推出，再次重申了 Mojo 作为综合性通用语言的地位。

**Nightly 编年史：从 CPython 互操作到社区讨论**
- 成员们正在庆祝 Mojo 中 CPython 互操作性的进展，并营造一个有利于贡献的环境，讨论 PR 中 signed-off commits 的最佳实践，并分享管理 nightly 构建和软件包更新的解决方案。这种积极的协作正在为未来的开源贡献铺平道路，GitHub 上已标明相关信息，包括关于 [Mojo Standard Library](https://github.com/modularml/mojo/discussions/2234) 的预期讨论。

**Mojo 创意连续体中的博客节拍与视频盛宴**
- [Joy of Mojo](https://joyofmojo.com/) 网站的发布强调了社区分享 Mojo 创意经验的承诺，通过 [mojo-packages](https://github.com/kernhanda/mojo-packages) 等 GitHub 仓库以及关于 Mojo 的 Python 互操作性的启发性视频，进一步放大了这一影响，彰显了其动态演进。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **WikiText 的新主访问点**：Stephen Merity 已在 Cloudflare R2 上[重新托管了 WikiText](https://state.smerity.com/smerity/state/01HRTB51QZMG59MDAX2ME1TCHR)，在提供更大数据集的同时保留了原始格式，这对于使用真实数据结构训练 language models 至关重要。

- **令人困惑的 Perplexity 分数**：关于 GateLoop Transformer 作者报告的 perplexity 分数有效性引发了辩论，lucidrains 无法复现这些分数，从而引发了对结果复现和报告透明度的讨论。

- **对 Hugging Face 自动 Parquet 转换的沮丧**：用户对 Hugging Face 将数据集自动转换为 parquet 格式表示不满，这可能导致混淆和问题（如处理 `.raw` 文件）；一种解决方法是使用 Git LFS 托管数据集。

- **文档的易逝性与对可复现性的强调**：OpenAI 模型文档的波动（部分链接被删除）凸显了像[存档页面](https://archive.ph/n5xMq)这样可靠资源对 AI 研究社区一致性的重要性。同时，社区正推动可复现的数据格式，如将 WikiText 等数据集镜像到 Cloudflare R2 等平台。

- **优化器优化与 Zero-Shot 创新**：讨论集中在 Schedule-Free 优化器及其估算最佳 learning rates 的能力，以及教 language models 使用 stream of search (SoS) 语言进行搜索的有趣方法。此外，language models 的涌现能力与训练期间接触长尾数据（long-tail data）之间的联系也是焦点话题，这对 zero-shot 任务表现具有深远影响。

- **GitHub Star 对 NSF 评审至关重要**：[nnsight](https://github.com/ndif-team/nnsight) 的 GitHub star 数量被一位 NSF 评审员强调为关注指标，说明了社区参与度对研究资助视角的非传统影响。

- **GPU 利用率与 BigBench 任务识别**：通过使用 `batch size=auto` 分析 GPU 利用率，减少了评估时间，揭示了潜在的利用不足问题。成员们还理清了关于 BigBench 任务的困惑，建议使用 `lm_eval —tasks list` 验证任务变体。

- **CLI 命令难题与 Logit Bias 讨论**：围绕 `—predict_only` CLI 命令问题以及 OpenAI 的 `logit_bias` 在单 token MCQA 任务中未按预期影响 logits 的技术讨论展开，引导开发者探索如 `greedy_until` 等替代方案。明确了 Temperature 设置及其对输出的影响，强调了正确设置 `gen_kwargs` 以实现预期模型行为的重要性。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **翻译大对决：GPT-4 vs DeepL**：对比了 GPT-4 与 DeepL 的翻译能力，强调虽然 **DeepL 在上下文语言翻译方面表现出色**，但 GPT-4 有时在处理基础语境的细微差别时显得不足。
- **代码生成领域的 AI 模型对决**：**Opus** 和 **GPT-4** 在代码生成任务中的出色表现获得赞赏，但与其他模型相比，GPT-4 在处理超长上下文时也显示出潜在问题。
- **解码 AI 意识**：一场关于用 AI 模拟人类意识的活跃探索，将人类神经化学活动等同于 GPT 的编程机制，引发了关于意识起源及 AI 在其描绘中作用的辩论。
- **针对敏感内容的 Prompt Engineering**：作者们讨论了如何绕过 ChatGPT 的内容政策，为具有创伤背景的角色开发背景故事，寻求以更微妙的方式在叙事中注入细腻、敏感的细节。
- **构建 AI 驱动的游戏**：工程师建议利用 JSON 结构化游戏进度数据，同时讨论了如何打造无缝游戏体验，使底层代码对玩家保持隐藏的挑战。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Claude 3 支持图像处理**：**Claude 3 模型**已更新为多模态，现在支持图像输入，要求开发者相应地修改现有的代码库。

**AI 回归经典：剪刀石头布**：[blust.ai](https://rock.blust.ai) 推出了一款新游戏，玩家可以挑战 ChatGPT 进行经典的剪刀石头布对决。

**前端工具与热门模型备受关注**：工程师们讨论了各种 OpenRouter API 前端，如 [LibreChat](https://librechat.ai/)、[SillyTavern](https://sillytavern.com/) 和 [Jan.ai](https://jan.ai/docs/remote-inference/router)。Command-R+ 已成为编程任务和土耳其语交互的首选模型，同时人们也对模型中的内容审查表示了担忧。

**模型性能见解**：对话强调了 Sonnet 在编程任务中优于 Opus，且 Claude 3 在 PDF 数据提取方面优于 Gemini Pro 1.5，这引发了一些对其效用的质疑。

**模型效能指标引发争论**：社区反映，仅基于使用统计数据的模型排名可能无法准确反映模型的价值，建议将支出或留存率作为潜在的替代衡量标准。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**加速 RAG 应用**：Marker-Inc-Korea 推出了 **AutoRAG**，这是一个用于调优 **RAG pipelines** 以增强性能的自动化工具，详情见其 [推文](https://t.co/ZndrM36n61)。同时，`create-llama` 已发布，旨在简化全栈 RAG/Agent 应用的启动，详见其 [推文](https://t.co/YOEZUQt7Lr)。

**利用 AI 优化销售话术**：最近的一场网络研讨会展示了一个使用 **RAG** 创建个性化销售邮件的新应用，弃用了硬编码模板，转而采用 LLM 驱动的方法，更多信息请查看 [推文](https://t.co/kV7MGJ6PqS)。

**深入文档探索**：Andy Singal 介绍了能够处理跨多个源的复杂 QA 的多文档 Agent。目标是扩展此功能以处理更复杂的查询，分享于 [演示推文](https://t.co/3yKuv2qDDf)。

**元数据助力文档查询**：为了从多文档查询中获取页码和文档引用，请确保在索引之前包含这些元数据，以便在查询后检索详细的引用信息。

**Azure 与 Embedding 时间的优化调整**：参与者注意到 Azure 的 OpenAI 无法识别上下文的问题，并讨论了使用批处理方法来加快 Embedding 生成。关于 **ReAct** Agent 以及 "llama2" 和 "mistral" 等开源模型面临的挑战，更好的 Router 描述可能会提高模型路由性能。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Mistral 需要强大的硬件支持**：**Mistral 7B Instruct v0.2** 被公认为性能卓越，但它对资源要求很高——预计至少需要分配 16GB RAM 并具备一定的 GPU 支持才能流畅运行。

**Python 兼容性挑战**：社区达成共识，建议坚持使用 **Python <=3.10**，以避免 **TTS packages** 的问题，并多次建议在依赖语音命令识别的设置中避免使用 **Python 3.11.4**。

**呼吁更完善的文档**：关于本地 Vision 模型的咨询以及强调 **Open Interpreter's cookbook** 需要更全面示例和文档的呼声，揭示了目前尚待填补的空白。

**本地模型：效率胜过高昂成本**：**GPT-4** 的高昂成本引发了关于利用 **Hermes 7B** 和 **Haiku** 等本地模型的讨论——这些是虽然稍逊色但成本更低的替代方案，且能提供隐私保护和更低的运行成本。

**硬件障碍与软件挫折**：**O1** 社区报告了硬件问题，特别是外部按钮集成方面，以及在 Windows 上安装时的软件设置挑战；故障排除对话中涉及了使用 **chocolatey**、**virtualenv** 和特定 **environment variables** 的调整。

相关资源和对话贯穿整个社区，针对问题的直接参与正在 [GitHub](https://github.com/OpenInterpreter) 等平台上进行跟踪。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **GitHub 上的困扰**：一位用户请求协助处理一个 **[Pull Request](https://github.com/langchain-ai/langchain/pull/19751)**，该 PR 因与 "openapi-pydantic" 相关的 "module not found" 错误而失败，尽管该模块已包含在依赖项中。这突显了依赖管理（dependency management）是社区中一个显著的痛点。
  
- **无需强力 GPU 的微调技巧**：关于在没有 GPU 的情况下训练和微调语言模型的咨询，引出了对 **Google Colab** 等工具的推荐，并提到 **ludwig.ai** 是可行的选择，这表明寻求高性价比计算资源的工程师对此领域非常感兴趣。

- **Artful AI 更新带来的视觉愿景**：**Artful AI** 发布了新模型，包括 **Dalle Creative、Anime Dream 和 Epic Realism**，并在 **[Google Play Store](https://play.google.com/store/apps/details?id=com.projecthit.artful&referrer=ph-aai)** 上架，引发了社区对不断发展的 AI 驱动图像生成领域的关注。

- **安全焦点：AISploit**：在 **[GitHub](https://github.com/hupe1980/aisploit)** 上发布的 **AISploit** 引起了关于利用 AI 进行攻防安全模拟（offensive security simulations）的讨论，标志着 AI 技术在网络安全应用中的战术转向。

- **TypeScript 和文本分块技术揭秘**：分享的一个 **[TypeScript Gist](https://gist.github.com/tsensei/3b6589662271874b5055d79473932aae)** 展示了如何使用 OpenAI 的句子嵌入（sentence embedding）服务将大文本分解为语义分块（semantic chunks），体现了社区在开发和分享增强文本处理工作流工具方面的积极参与。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**苹果的 AI 雄心备受审视**：苹果因 **Metal Performance Shaders** (MPS) 和 **torch compile** 表现不佳而受到批评，尽管最近的合并旨在修复 PyTorch nightly 分支中的 **MPS** 问题。社区对 **torch.compile** 的体验各异，反映出苹果平台仍需持续优化。

**版权难题**：AI 使用受版权保护的内容创作衍生作品引发了法律辩论，共识是仅靠改写（paraphrasing）不足以避免侵权。社区预见到需要重大的法律变革来适应新的 AI 训练数据实践。

**AI 作曲的和谐**：关于 AI 生成音乐的讨论涉及 **Suno** 和 Nvidia 等公司，认可了其快速进步，但也预测了与音乐行业潜在的法律纠纷。成员们还注意到，与 AI 在音乐生成方面的飞跃相比，文本转语音（TTS）技术的进展不那么令人印象深刻。

**AI 职业动态的变化**：由于技术进步，自由职业中与 AI 相关的职业兴起，并引用了 **Bloomberry 的分析** 等资源。**Stability AI 的 CosXL 模型** 发布引发了关于模型训练中 **EDM schedules** 和 **offset noise** 有效性的对话。

**AI 研究技术的新颖点**：一篇关于 **transformers** 的新论文显示计算资源分配可以是动态的，**DARE** 的剪枝（pruning）技术暗示了语言模型的可保留能力，而 **BrushNet** 引入了增强的 AI **inpainting**。引用自 NeurIPS 论文的用于文本生成的 Latent diffusion 表明生成模型技术可能发生转变。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT 模型应对 A::B 挑战**：Victor Taelin 承认 GPT 结构确实可以处理某些问题解决任务，包括长期推理（long-term reasoning）。此前一名参与者利用 GPT 以接近 100% 的成功率解决了 A::B 问题，并赢得了 1 万美元奖金。[Victor Taelin 关于该结果的声明已在网上发布](https://x.com/victortaelin/status/1777049193489572064?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)。

- **斯坦福大学首推语言建模课程 CS336**：斯坦福大学正在开设一门新课程 CS336，深入探讨语言建模的核心细节，包括关于 Transformers 和 LLMs 的见解，吸引了渴望课程录像发布的社区成员的极大关注。

- **Groq 计划颠覆 AI 硬件对手**：由一位拥有非传统教育背景的创始人领导的 AI 硬件初创公司 Groq，目标是到明年超过所有现有推理能力提供商的总和，并声称与 NVIDIA 的产品相比，其开发者可以享受更低的推理成本和更快的硬件速度。

- **介绍 LangChain 的 Memory 服务**：LangChain 最新的 alpha 版本带来了 Memory 服务，旨在通过自动压缩和精炼对话来升级聊天机器人的交互体验，并已发布[快速入门资源](https://langchain-ai.github.io/long-term-memory/)。

- **AI 工具与知识管理中的同伴学习**：工程师们交流了使用 AI 工具策划个人和组织知识的资源与策略，例如集成 [Obsidian-Copilot](https://github.com/logancyang/obsidian-copilot) 和 [fabric](https://github.com/danielmiessler/fabric)，并讨论了开发集成功能以增强知识系统中的 ChatGPT 等工具。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **量化 DoRA 可用，LoRA 之舞**：`peft=0.10.0` 的最新版本支持 **量化 DoRA**，促使建议更新 **axolotl** 的 `requirements.txt`（[PEFT 发布说明](https://github.com/huggingface/peft/releases/tag/v0.10.0)）。来自 Facebook Research 的 **高级优化器** 现已集成到 Hugging Face 的 transformers 库中，**Schedule-Free Learning** 已开源，并为 ScheduleFreeAdamW 提供了 `0.0025` 的特定参数建议（[Hugging Face PR #30079](https://github.com/huggingface/transformers/pull/30079)）。

- **模型生成故障**：用户报告并讨论了在使用 **fp16** 的 **微调 Mistral 7b 模型** 生成过程中出现的错误，具体表现为在几次成功生成后出现 `_queue.Empty`。

- **Rotary 查询与分片见解**：参数 `"rope_theta": 10000.0` 受到关注，这与 **Rotary Positional Embedding** 有关。同时，分享了 Mistral 的 **FSDP 配置**，并详细说明了应如何使用 `MixtralSparseMoeBlock` 类（[mixtral-qlora-fsdp.yml](https://github.com/openaccess-ai-collective/axolotl/tree/main/examples/mistral/mixtral-qlora-fsdp.yml#L1L75)）。

- **寻找 LISA 与配置**：有关于文档中缺失 **LISA** 参数位置的疑问，随后通过发现 [LISA 配置文件](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/lisa.yml) 得到了解决。成员们还就 **训练期间解冻新层** 时如何处理优化器状态进行了技术讨论。

- **模型训练难题解决**：社区解决了各种挑战，包括 **使用原始文本训练**、适配 **Alpaca 指令集**、区分 **micro batch size** 和 **batch size**，以及调整配置以禁用 checkpoint 和评估，或处理特殊 token。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**播客黄金机会：John Schulman 可能做客节目**：Nathan Lambert 正在考虑邀请 John Schulman 参加播客，这一举动引起了成员们的兴奋。此外，**text-generation-inference (TGI)** 许可协议更改为 Apache 2，促使该开源项目的贡献者显著增加。

**Memes 频道保持轻松氛围**：memes 频道包含了关于无上下文目标的玩笑引用、体验改进以及雇佣状态确认，显示出成员之间随性、轻松的交流。

**Open AI 权重辩论触动神经**：#reads 频道对开源基础模型的社会影响进行了热烈讨论，重点关注安全阈值、监管可行性以及 AI 操纵社会进程的潜力。讨论的深入话题包括 Transformer 注意力机制的可视化分享，以及对未来强调验证而非生成的模型的推测。

**用视觉化弥合知识鸿沟**：#sp2024-history-of-open-alignment 频道讨论了寻找最先进模型的有效资源，如 [lmsys](https://lmsys.deepai.org/) 和 [alpacaeval leaderboard](https://alpacaeval.com/)。此外，还表达了通过视觉化对模型进行分类以更好地理解的意图，并分享了一个用于即将到来的对齐（alignment）演讲的实时文档（[Google Slides 演示文稿](https://docs.google.com/presentation/d/1quMyI4BAx4rvcDfk8jjv063bmHg4RxZd9mhQloXpMn0/edit?usp=sharing)）以及 Xeophon 编写的开源模型指南（[综合电子表格](https://docs.google.com/spreadsheets/d/1gc6yse74XCwBx028HV_cvdxwXkmXejVjkO-Mz2uwE0k/edit#gid=0)）。

**关于 AI 生成音乐的说明**：Nathan 注意到 AI 音乐生成领域一个新竞争者的惊人质量，可能对 Suno AI 平台构成挑战。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **快速分词（Tokenization）**：工程师们讨论了使用 Huggingface 的 fast tokenizer 为 *c4-en* 加速分词，探索了增加线程或利用性能更强的机器等选项。

- **开源 GPU**：AMD 宣布开源其 Radeon GPU 的 Micro Engine Scheduler (MES) 固件，这一决定在社区内受到欢迎，并得到了 George Hotz 的 Tiny Corp 等实体的赞扬。（[The Register 文章](https://www.theregister.com/2024/04/05/amd_mes_open_source/)，[AMD Radeon 推文](https://twitter.com/amdradeon/status/1775999856420536532)）。

- **论文复现之路**：一个用于复现 AI 和 ML 研究论文的开源仓库 [PaperReplica GitHub Repo](https://github.com/hegdeadithyak/PaperReplica) 正式亮相，邀请社区贡献和 GitHub star。

- **CUDA 难题与 Triton 策略**：从在 Ubuntu 上搭建 CUDA 环境到赞赏提升 Triton 熟练度的库，成员们交流了技巧和困扰。特别值得一提的是，Andrej Karpathy 用 C 语言实现的一个精简 GPT-2 训练版本，因其无需 PyTorch 或 Python 的臃肿而具备的高效率而受到关注（[GitHub](https://github.com/karpathy/llm.c)）。

- **DeepSpeed 驶入快车道**：对话围绕 DeepSpeed 的实际应用、与 Hugging Face 的 Accelerate 集成，以及即使在 zero stage 下的内存优化奇迹展开。此外，还提到 Triton kernel 的使用取决于 CUDA 设备输入，并分享了关于使用 cublas 或 tiny-cuda-nn 优化 Transformer MLP 的好奇心（[tiny-cuda-nn 文档](https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md)）。

- **LLM 的量子慰藉（Quantization）**：提出了一种名为 [QuaRot](https://arxiv.org/abs/2404.00456) 的新型量化方法，能够有效地将 LLM 量化到 4 bits；同时一条启发性的推文暗示了无调度优化（schedule-free optimization），可能预示着将摆脱传统的学习率调度（learning rate schedules）（[Twitter](https://twitter.com/aaron_defazio/status/1773381393831067787)）。

- **困扰于 Triton 可视化**：工程师们深入探讨了 Triton 代码可视化的挑战与机遇，从共享内存到 tensor 视图，从 CPU 结构到增强 JavaScript 交互性，标志着对更用户友好的调试工具的持续追求。

- **日历困惑已澄清**：针对一次 ring attention 会议进行了小小的时区澄清，暗示了社区对知识和优化不懈追求的活力。

- **数字与神经元**：精确量化方法的价值浮出水面，强调了准确 tensor 转换的重要性，以及利用 Triton 等工具实现的潜在性能提升，表明了对机器学习流水线效率的高度关注。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Tinygrad 的回撤**：George Hotz 撤回了 tinygrad 中的命令队列（command queue），并选择将内存调度器（memory scheduler）直接集成到当前的调度器模型中。这种方法利用了已经存在的 multidevicegraph 抽象，详见[此处](https://github.com/tinygrad/tinygrad/pull/4094)。

**显微镜下的 TinyJit**：TinyJit 教程已发布，尽管可能包含不准确之处，特别是关于 `apply_graph_to_jit` 函数的部分，鼓励用户提交 Pull Request 进行修正：[TinyJit 教程](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/jit.md)。

**Tinygrad 学习资源扩展**：一系列针对 tinygrad 贡献的教程和指南现已发布，重点关注多 GPU 训练等主题：[多 GPU 训练指南](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/multigpu.md)。

**Discord 角色反映贡献**：George Hotz 重新设计了 tinygrad Discord 内部的角色，以更好地反映社区参与度和贡献水平，强化了协作价值和对他人时间的尊重。

**揭开 MEC 固件之谜**：关于 MEC 固件操作码（opcode）架构的讨论出现，推测涉及 RISC-V 和不同的指令集，揭示了潜在的 `cbz` 指令，并就 RISC-V ISA 的细微差别进行了包容性对话。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**扫描揭示 Llamafile 被误报**：包括 **llamafile-0.6.2.exe** 和 **llamafile-0.7** 在内的 **llamafile** 版本被杀毒软件标记为恶意软件；建议向相应的杀毒软件公司提交申诉表作为补救措施。

**在 Kaggle 中更顺畅地运行 Llamafile**：在 Kaggle 上运行 `llamafile` 遇到问题的用户通过一个**更新的命令**找到了解决方法，该命令解决了 CUDA 编译和兼容 GPU 架构的问题，从而能够高效使用 **llamafile-0.7**。

**RAG-LLM 实现本地化运行**：关于在没有 Docker 或 Python 负担的情况下本地分发 RAG-LLM 应用程序的咨询得到了肯定回答，表明 **llamafile** 非常适合此类用途，特别是对 macOS 用户非常有益。

**通过参数驯服内存怪兽**：一位用户遇到的**内存溢出（out of memory）错误**通过调整 `-ngl` 参数得到了解决，这证明了根据 NVIDIA GeForce GTX 1050 显卡的具体能力微调参数的重要性。

**Vulkan 集成提升性能**：通过集成 Vulkan 支持来增强 **llamafile** 的提议，在带有集成 GPU 的 Intel 笔记本电脑上实现了性能提升，但这需要重新导入和修改 **llama.cpp** 文件的细致工作。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **新优化器无需调度**：[huggingface/transformers 仓库](https://github.com/huggingface/transformers/pull/30079) 现在有一个 Pull Request，引入了 Meta 为 AdamW 和 SGD 提供的 *schedule-free optimizers*（无调度优化器），这有望大幅提升模型训练流程。
- **AI 开发者齐聚许尔特 (Hürth)**：一场专注于合成数据生成、LLM/RAG 流水线和 embeddings 的 AI 社区活动定于 5 月 7 日在德国许尔特举行。注册已开放，强调动手实践、以开发者为中心的格式，详情见 [开发者活动 - AI Village](https://www.eventbrite.de/e/developer-event-ai-village-tickets-868896702427)。
- **寻求合成数据见解分享**：对合成数据策略知识的需求很高，特别是对德语翻译数据与德语生成数据质量的对比感兴趣，这表明了对区域数据处理专业知识的特定需求。
- **Command-R 挑战高难度翻译**：在 [Hugging Face Spaces](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus) 上展示的 Command-R 模型在翻译古老的中古高地德语文本方面表现出色，优于 GPT-4 同类模型，凸显了历史语言处理领域的潜在变革。
- **期待开源模型开发**：人们预期令人印象深刻的 Command-R 的开源发布可能会增强开发者的参与度，呼应了像 Mistral 这样公开获取的模型在生态系统中所取得的成功。

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **稳扎稳打才能赢得比赛？**：对比显示，与标准的 **Transformer** 模型相比，**Jamba** 的 1B Mamba 模型在 HGX 上运行时的训练速度落后了 *76%*。

- **尺寸并不总是关键**：工程师们推出了缩减版的 **Jamba** 模型，包括 [29B 参数的 8xMoE](https://huggingface.co/isemmanuelolowe/Jamba-8xMoE_Slerp) 和 [17.7B 参数的 4xMoE](https://huggingface.co/isemmanuelolowe/Jamba-4xMoE_Slerp)，在像 4090 GPU 这样易于获取的硬件上以 4 bit 运行时，表现相当不错。

- **权重与衡量**：一位创作者在 **Jamba** 模型中应用球面线性插值 (Slerp) 来进行专家权重缩减，这引起了人们的兴趣，并计划分享一个详细介绍该过程的 notebook。

- **算力博弈**：为了在处理 52B **Jamba** 模型时实现最佳 GPU 利用率，一位工程师正在寻求更高效的训练方法，考虑到目前的容量限制，可能会考虑从流水线并行切换到张量并行 (Tensor Parallelism)。

- **什么是最佳的模型服务方案？**：社区正在讨论针对 **Jamba** 模型有效的**推理引擎 (inference engines)**，尽管目前尚未达成共识。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **QNAP NAS - AI 爱好者的家庭实验室**：一位 AI 工程师分享了一份[指南](https://www.storagereview.com/review/run-a-private-rag-chatgpt-on-qnap-nas)，介绍如何将 **QNAP NAS**（型号 TS-h1290FX）配置为 AI 测试平台，并强调了其显著的规格，如 AMD EPYC 7302P CPU、256GB DRAM 和 25GbE 网络。

- **利用预设提示词简化 AI 流程**：工程师们对存储和重用系统提示词 (system prompts) 以提高 AI 交互效率表现出好奇，尽管讨论没有进一步深入到更详细的见解或经验分享。

- **Alter：Mac 上的 AI 写作助手**：**Alter** 正在启动 beta 测试，为 macOS 用户提供 AI 驱动的文本改进服务，能够与 Keynote 等应用程序集成，如[此演示视频](https://youtu.be/IK53CSSbaqI)所示。

- **面向 Mac 爱好者的单一 AI 解决方案**：Alter 应用程序旨在为所有 macOS 应用程序提供上下文感知 AI 功能，从而可能集中管理 AI 工具并减少对多种服务的需求。有关其完整功能的详细信息可在 [Alter 网站](https://alterhq.com)上找到。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **动态计算分配激发灵感**：工程师们讨论了一篇提议在神经网络中**基于 per-token 进行计算资源动态分配**的论文，这引起了对 **neurallambda** 可能进行适配的兴趣；其目标是允许网络自我调节其计算量。
- **重新思考 neurallambda 的训练方法**：探索性讨论包括使用 **pause/think tokens**、**用于条件语句的强化学习 (reinforcement learning)**，以及模拟 RNNs 能够自适应控制计算使用量的特性，这可能会增强 **neurallambda** 的训练效能。
- **创新的输入处理方式展望**：技术专家考虑了 **neurallambda** 的新型输入方法，例如使用神经队列 (neural queue) 进行更灵活的处理，并将输入概念化为类图灵机纸带，网络可以启动纸带移动。
- **提升 LLM 的数据结构化能力**：参与者分享了一个名为 "Instructor, Generating Structure from LLMs" 的教学视频，展示了从 **GPT-3.5, GPT-4** 和 **GPT-4-Vision** 等 LLM 中提取 JSON 等结构化数据的方法，旨在从这些模型中获得更可靠的结果。[观看教学视频](https://www.youtube.com/watch?v=KxOqjKq2VyY)。
- **视频学习机会**：链接了第二个教育视频，但未提供上下文，这可能是一个供好奇者自主学习的资源。[探索视频](https://www.youtube.com/watch?v=keUjQyWmgY0)。



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Haiku 性能调优探索**：由于对 **Haiku** 当前的吞吐量 (throughput) 不满意，一名成员正在寻求提高其速度的建议。

- **Anthropic 的 API 表现优于 GPT-4 Turbo**：一位用户提供的证据表明，在 Berkeley 函数调用基准测试 (Berkeley function calling benchmark) 的众多测试中，**Anthropic** 的 beta API 超过了 **GPT-4 Turbo**。这项研究的结果可以在[详细的 Twitter 线程](https://x.com/JoschkaBraun/status/1777381282751688868)中找到。



---


**Alignment Lab AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接

**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1225920372562591825)** (1 条消息): 

- **iOS 上的新故事发现体验**：Perplexity 正在其 iOS 应用中测试一种新的故事发现格式。欢迎在指定频道提供反馈；在此处获取应用 [here](https://pplx.ai/download)。
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1225742672845606993)** (1199 条消息🔥🔥🔥): 

- **活动以平局结束**：Perplexity AI Discord 活动圆满结束，用户 *eun08000* 和 *codelicious* 并列第一。获奖者将通过 DMs 收到奖品领取通知。

- **Claude 3 Opus 的差异**：用户讨论了 Poe 和 Perplexity 之间 Claude 3 Opus 模型的差异，注意到性能上的变化，特别是在创意和写作任务方面。

- **对日食的热情**：服务器成员分享了他们对日食的期待和观察，讨论内容包括理想的观测设备以及见证这一现象的经历。

- **关于月球形成的问题**：关于月球的形成引发了讨论，一名用户对月球是与地球相撞的天体的一部分这一理论表示怀疑。分享了相关教育资源的链接以供进一步了解。

- **在 Discord 上获取 Pro 身份组**：用户询问如何获取 Discord 服务器上的 “Pro” 身份组，指引是通过 Perplexity 网站账户设置中提供的 Pro Discord 链接重新加入。
<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://x.com/OscaR_010__/status/1776969765635961068?t=lvoBPNlllBK_dMSAxHr16w&s=33">来自 OscaR-_-010 (@OscaR_010__) 的推文</a>: @kodjima33 @bing @OpenAI @perplexity_ai @AnthropicAI 嗨，朋友。在这里我不得不说，在搜索方面 @perplexity_ai 超越了所有提到的这些，其搜索范围更广。搜索包括...</li><li><a href="https://www.rabbit.tech/">rabbit</a>: $199 无需订阅 - 人机交互的未来 - 立即预订</li><li><a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>: 未找到描述</li><li><a href="https://support.stripe.com/questions/impact-of-sanctions-on-russia-and-belarus?locale=en-US">对俄罗斯和白俄罗斯制裁的影响 : Stripe: 帮助与支持</a>: 未找到描述</li><li><a href="https://tenor.com/view/cat-cat-memes-cat-images-cat-meme-gif-4644773688486402896">猫咪表情包 GIF - 猫咪表情包 猫咪图片 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/space-gif-25736952">太空 GIF - 太空 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/queen-freddie-mercury-we-are-the-champions-champion-sing-gif-4654136">Queen - Champion GIF - Queen Freddie Mercury We Are The Champions - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://gs.statcounter.com/vendor-market-share/mobile">全球移动设备厂商市场份额 | Statcounter Global Stats</a>: 该图表显示了基于每月超过 50 亿次页面浏览量的全球移动设备厂商市场份额。</li><li><a href="https://support.privacy.com/hc/en-us/articles/360050917053-Can-I-use-Privacy-if-I-live-outside-the-US">如果我居住在美国境外，可以使用 Privacy 吗？</a>: 目前我们仅能向美国公民或美国合法居民提供服务。我们正在继续探索机会和方案，将 Privacy 带到世界其他地区。H...</li><li><a href="https://gs.statcounter.com/os-market-share/mobile/worldwide">全球移动操作系统市场份额 | Statcounter Global Stats</a>: 该图表显示了基于每月超过 50 亿次页面浏览量的全球移动操作系统市场份额。</li><li><a href="https://tenor.com/view/unlimited-power-star-wars-gif-10270127">无限力量星球大战 GIF - 无限力量星球大战 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://labs.mojeek.com/rag/index.html">Mojeek Labs | RAG 搜索</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/discuss/65d956e39db34f001ff8ce0a">Sonar 模型是新的吗？</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/discuss/6582f98b41714c00723d5d5c">PPL 网站上的模型与 API 模型之间的区别。</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/discuss/6601ffd6bd5f0e0045ac5d16">模型名称？</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=gCkZmADecL0">日食现场直播（含视频和更新）</a>: 加入我们的日食直播报道，包含实时日食视频！我们将为您展示墨西哥、美国和加拿大的日全食直播...</li><li><a href="https://youtu.be/wkQuOrsgVGY?si=qrl5Bdx_Mr4L-f6_&t=2603">我们太阳系的八大奇迹 | 行星 | BBC 地球科学</a>: 探索我们太阳系历史上最难忘的事件。前往这些动态世界的表面，见证那些充满戏剧性的时刻...</li><li><a href="https://www.star.nesdis.noaa.gov/GOES/conus_band.php?sat=G16&band=GEOCOLOR&length=24">GOES-East CONUS - GeoColor - NOAA / NESDIS / STAR</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1225901847470932038)** (40 条消息🔥): 

- **探索 AI 前沿**：用户分享了大量指向 Perplexity AI 平台的搜索查询，涵盖了从 Samsung Galaxy S23 到 AI 对音乐产业影响等主题。
- **科技巨头动态**：分享了一个 YouTube 视频链接，讨论了 Tesla 的 Robotaxi 发布预告和 Apple 的家用机器人项目，突显了科技领域的进展和传闻。
- **互动提醒**：发布了多条针对用户的提醒，敦促他们确保其线程（threads）是可分享的，并附带了具体说明和附件链接。
- **精选成功案例**：Amazon Web Services 的一份案例研究展示了 Perplexity AI 在模型训练方面的效率，呈现了训练时间的显著缩短和用户体验的提升。
- **教育见解**：提供了关于色彩基础、几何证明起源以及 SpaceX 火星计划等各种主题的知识资源链接，反映了对学习和自我提升的广泛兴趣。
<div class="linksMentioned">

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aws.amazon.com/solutions/case-studies/perplexity-case-study/">Perplexity 通过 Amazon SageMaker HyperPod 将基础模型训练速度提升 40% | Perplexity 案例研究 | AWS</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=JAuKnXSn70s">Tesla robotaxi 发布会、Alphabet-HubSpot 收购传闻、Apple 家庭机器人项目</a>：在今天的 Perplexity Discover Daily 节目中，我们深入探讨了特斯拉即将于 8 月 8 日举行的无人驾驶出租车（robotaxi）揭幕仪式，并探索了这款自动驾驶车辆...</li><li><a href="https://www.youtube.com/watch?v=yGejxO1xYmo">构建可信 AI 的工作流与工具 | 与 Clara Shih 探讨 AI 的更多可能</a>：Clara 与三家最热门 AI 公司的创始人/CEO 坐下来交流——Aravind Srinivas (Perplexity AI)、Jerry Liu (LlamaIndex) 和 Harrison Chase (LangChain)...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1225764813393629244)** (40 条消息🔥): 

- **API 额度购买困难**：成员们在尝试购买 API 额度时遇到问题；尽管尝试了多次，刷新后余额仍显示为 $0。**ok.alex** 请求受影响的用户发送账户详情以便解决。
- **pplx-labs、API 与 Web 应用之间的差异**：用户报告了在 pplx-labs、API 和 Web 应用中使用相同 Prompt 时结果不一致的问题，API 表现出更多的幻觉（hallucinations）。**icelavaman** 告知 [pplx.ai](https://pplx.ai) 的默认模型无法通过 API 获取，且引用功能目前处于封闭测试阶段。
- **正在开发 Perplexity API 的 Ruby 封装库**：**filterse7en** 正在为 Perplexity API 开发一个基于 OpenAI 的 Ruby 封装库。
- **API 与 Web 应用模型的差异**：讨论揭示了 API 和 Web 应用结果之间的差异，并对结果质量和幻觉的存在表示怀疑。**brknclock1215** 建议通过 API 使用 `sonar-medium-online` 模型在效果上应等同于非 Pro 网页版的 "Sonar" 模型。
- **关于 pplx-pro 模型 API 访问的咨询**：**marciano** 询问 pplx-pro 中使用的模型是否可以通过 API 访问。**ok.alex** 澄清 Pro 搜索仅在网页和他们的 App 上可用，无法通过 API 访问。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://api.perplexity.ai")">未找到标题</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：未找到描述
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1225759346894442506)** (15 条消息🔥): 

- **介绍 Command R+**：分享了一个名为 "Introducing Command R+: A Scalable LLM Built for Business" 的 YouTube 视频，展示了 [Cohere 强大的 LLM](https://www.youtube.com/watch?v=keUjQyWmgY0)，该模型专为企业级应用构建。
- **AI 研究过载**：一位成员表达了对是否需要更多 AI 研究人员的担忧，其中一位成员认为目前的研究已经超出了消化能力，而另一位成员指出需要更多的元研究人员（meta-researchers）来综合和解读海量信息。
- **快速搜索图片**：介绍了 'Where's My Pic?' 项目，提供了一个类似于 Google Image Search 的本地文件夹解决方案，可以节省快速定位图片的时间。通过这个 [YouTube 视频](https://www.youtube.com/watch?v=oVJsJ0e6jWk)了解更多。
- **加拿大的 AI 战略**：通过一份 [政府新闻稿](https://www.pm.gc.ca/en/news/news-releases/2024/04/07/securing-canadas-ai-advantage) 强调了加拿大成为 AI 前沿阵地的雄心，包括在创新和技术领域创造高薪就业机会。
- **Hugging Face 技术洞察**：对 huggingface.tech 的 SSL 证书进行了分析，提供了关于他们所使用工具的见解，详见 [crt.sh](https://crt.sh/?q=huggingface.tech)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=keUjQyWmgY0">介绍 Command R+：为业务构建的可扩展 LLM</a>：今天，我们将了解 Command R+，这是 Cohere 最强大、可扩展的大语言模型 (LLM)，专为在现实世界的企业应用中表现出色而构建...</li><li><a href="https://www.pm.gc.ca/en/news/news-releases/2024/04/07/securing-canadas-ai-advantage">确保加拿大的 AI 优势</a>：未找到描述</li><li><a href="https://crt.sh/?q=huggingface.tech">crt.sh | huggingface.tech</a>：未找到描述
</li>
</ul>

</div>
  

---

**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1225822258279354479)** (49 messages🔥): 

- **Rohan Paul 的 AI 推文引发好奇**：Rohan Paul 关于 AI 的一条推文被重新提及，注意到它早期的印象虽然很深刻，但在三个月后缺乏后续信息和见解。讨论还涉及了在 **NVIDIA 4090 GPU** 上使用 **fp8** 的可用性。
  
- **LLaMA-2-7B 突破上下文长度限制**：分享了一项突破性成就，**LLaMA-2-7B** 仅使用 8 个 **A100 GPU** 就被训练成可以处理高达 **700K 的上下文长度**，大大超过了预期的 32K 到 200K token 的容量。

- **Gemma 1.1 加入 AI 语言模型家族**：Google 在 Hugging Face 上发布了 **Gemma 1.1 7B (IT)**，这是一款指令型语言模型，在质量、代码能力和指令遵循方面都有所提升。重点介绍了其训练过程中使用的创新 **RLHF 方法**。

- **迷宫寻路的新尝试**：提出了一种利用斐波那契二项式猜想等推测性框架来统一物理学的独特方法，暗示 **NLP** 可以模拟任何过程。

- **GPT-4 表现出元叙事倾向**：分享了使用 GPT-4 的一段经历，其中给定的提示词导致了出人意料的元叙事（meta）和自我引用内容。关于这一有趣表现的讨论还包括了一个相关游戏旁白功能的 **YouTube 链接**。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/PY_Z001/status/1776176932687892796">Tweet from Zhang Peiyuan (@PY_Z001)</a>: 🌟700K context with 8 GPUs🌟 How many tokens do you think one can put in a single context during training, with 8 A100, for a 7B transformer? 32K? 64K? 200K? No, my dear friend.  I just managed to tra...</li><li><a href="https://arxiv.org/abs/2305.14078">Large Language Models as Commonsense Knowledge for Large-Scale Task Planning</a>: Large-scale task planning is a major challenge. Recent work exploits large language models (LLMs) directly as a policy and shows surprisingly interesting results. This paper shows that LLMs provide a ...</li><li><a href="https://outlines-dev.github.io/outlines/cookbook/classification/">Classification - Outlines 〰️</a>: Structured text generation with LLMs</li><li><a href="https://huggingface.co/papers/2402.14083">Paper page - Beyond A*: Better Planning with Transformers via Search Dynamics
  Bootstrapping</a>: no description found</li><li><a href="https://huggingface.co/google/gemma-1.1-7b-it">google/gemma-1.1-7b-it · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/papers/2404.03715">Paper page - Direct Nash Optimization: Teaching Language Models to Self-Improve with
  General Preferences</a>: no description found</li><li><a href="https://www.factorialfunds.com/blog/under-the-hood-how-openai-s-sora-model-works">Factorial Funds | Under The Hood: How OpenAI&#039;s Sora Model Works</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">Visualizing Attention, a Transformer&#39;s Heart | Chapter 6, Deep Learning</a>: Demystifying attention, the key mechanism inside transformers and LLMs.Instead of sponsored ad reads, these lessons are funded directly by viewers: https://3...</li><li><a href="https://www.youtube.com/watch?v=6RTkUgov60g&ab_channel=GameplayDump">Bastion: Narrator Bits Part 1 (Wharf District, Workmen Ward, Breaker Barracks)</a>: I recorded the game with narrator audio only and everything else turned down to zero in the sound menu volume settings.   Then I just cut out the silent part...</li><li><a href="https://github.com/vicgalle/configurable-safety-tuning">GitHub - vicgalle/configurable-safety-tuning: Data and models for the paper &quot;Configurable Safety Tuning of Language Models with Synthetic Preference Data&quot;</a>: Data and models for the paper &quot;Configurable Safety Tuning of Language Models with Synthetic Preference Data&quot; - vicgalle/configurable-safety-tuning
</li>
</ul>

</div>

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1225787256074145843)** (148 messages🔥🔥): 

- **PDF 转 Markdown 的 GitHub 资源**：一位成员分享了一个名为 **VikParuchuri/marker** 的 GitHub 仓库，该工具可以高精度地将 PDF 文件转换为 Markdown 格式。仓库地址为 [GitHub - VikParuchuri/marker](https://github.com/VikParuchuri/marker)。

- **Hermes 函数调用难题**：讨论了如何让 `Hermes-2-Pro-Mistral-7B` 使用类似于 OpenAI 模型的 `tools` 配置来执行函数。虽然该模型可以处理 ChatML 语法和消息中的函数调用，但在执行 `tools` 中定义的函数时遇到了问题。

- **LLM 中的全参数与 Adapter 训练**：成员们讨论了与训练 Adapter 相比，全参数微调（full parameter finetuning）在获得一致性结果方面的挑战，一些人分享了他们在不同上下文（如 Mixtral 或 Llamas）中使用这两种方法的成功或失败经验。

- **探索大模型输出限制**：讨论了大型语言模型输出大小的限制，大家认识到虽然输入上下文可以非常大，但由于训练数据不同以及操作层面的考虑（例如训练时需要类似大小的输出示例），输出是受到限制的。

- **结合本体与向量搜索**：深入讨论了如何将 **知识图谱 (KG) 本体** 与语言模型结合使用。分享了如何从输入文本创建 Cypher 查询、使用向量搜索评估函数遍历 KG 图的有效性，以及在生产环境中将向量数据库与图数据库集成的技巧。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/TroyDoesAI/MermaidMistral">TroyDoesAI/MermaidMistral · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling">GitHub - NousResearch/Hermes-Function-Calling</a>：通过在 GitHub 上创建账号来为 NousResearch/Hermes-Function-Calling 的开发做出贡献。</li><li><a href="http://github.com/joey00072/ohara/issues/8">关于 GPU 显存占用 · Issue #8 · joey00072/ohara</a>：你好。首先，感谢分享 bitnet 训练代码。我有一个关于 GPU 显存占用的问题。据我了解，与 fp16/bf16 精度相比，bitnet 可以减少 VRAM 占用。然而...</li><li><a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: 高精度快速将 PDF 转换为 Markdown</a>：高精度快速将 PDF 转换为 Markdown - VikParuchuri/marker
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1226177515349872640)** (5 messages): 

- **用随机性扭曲 LLaVA**：一位成员通过向图像嵌入（image embeddings）注入随机性，对 LLaVA 视觉语言模型进行了实验，并观察了 LLM 的解读，详见其 [博客文章](https://anotherjesse.com/posts/llava-rorschach/)。该过程涉及调整模型以接受随机投影而非 CLIP 投影，本质上是对 AI 进行“罗夏墨迹测试”。

- **nanoLLaVA 强势登场**：发布了“小而强大”的 **nanoLLaVA**（低于 1B 参数的视觉语言模型），一位成员分享了他们在 Hugging Face 上的作品 [nanoLLaVA](https://huggingface.co/qnguyen3/nanoLLaVA) 链接，该模型可在边缘设备上运行，并拥有 Base LLM 和 Vision Encoder 的独特组合。
 
- **Obsidian 和 Hermes Vision 更新在即**：同一位成员宣布 **Obsidian** 和 **Hermes Vision** 即将更新，暗示视觉语言模型能力的增强。

- **ChatML 与 LLaVA 的融合能力**：成功实现了让 ChatML 与 LLaVA 模型协同工作，暗示了对话与视觉语言任务之间潜在的桥梁。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://anotherjesse.com/posts/llava-rorschach/">anotherjesse.com - LLaVA LLM 的罗夏墨迹测试</a>：未找到描述</li><li><a href="https://huggingface.co/qnguyen3/nanoLLaVA">qnguyen3/nanoLLaVA · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1225927044035252286)** (19 messages🔥):

- **建议为多样化数据集编写 Chunking Script**：有人提议编写一个 *chunking script*，以避免在生成数据集时进行大规模的 RAG 调用。通过预先准备 RAG 生成内容，这可能会使数据集更加多样化且高效。
- **通过 Claude Opus 进行 Multidoc Queries**：讨论了使用 *Claude Opus* 生成 multidoc queries 的可能性，即通过选择来自不同领域的文档并生成跨领域的查询。这种方法可以增强 RAG 模型的复杂查询生成能力。
- **用于模型训练的多样化文档源**：分享了多样化文档源的链接，例如 [OCCRP 数据平台](https://aleph.occrp.org) 和 [The Eye](https://the-eye.eu/public/) 上的各种文件库。可以抓取这些来源以创建丰富的训练数据集。
- **用于训练的勒索软件受害者文档**：曾考虑将勒索软件组织发布的受害者内部文档作为潜在的训练数据源。然而，使用此类数据的伦理问题被指出存在争议。
- **讨论 RAPTOR Clustering 策略**：强调了 RAPTOR clustering 方法的递归特性，引发了关于生成 Cluster 的策略及其在 RAG 数据集集合分层中作用的讨论。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aleph.occrp.org">未找到标题</a>：未找到描述</li><li><a href="https://the-eye.eu/public/">Index of /public/</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1225820438412787844)** (567 条消息🔥🔥🔥): 

- **等待期间 Worldsim 的现状**：在遭受 *DDoS* 攻击后，用户继续热切询问 **Worldsim** 的恢复情况，并讨论了实施登录系统以防止来自 *4chan* 等臭名昭著的在线社区未来攻击的可能性。

- **AI 记忆之谜**：关于 **Claude** 是否能跨会话记住信息，还是仅仅通过其概率模型模仿这种能力，存在困惑和讨论。尽管 Prompt 相同，但随机的 Token 选择会导致不同的结果。

- **寻求可持续解决方案**：由于用户提议为 Worldsim 建立订阅模式，以抵消因无限制访问而产生的高昂运营成本，**Nous Research** 暗示未来将推出“pro”版本并计划增加更多场景，同时强调需要一个可持续的平台。

- **关于超越的传说与技术**：频道中充满了关于意识、存在以及生活在模拟中可能性的哲学讨论；平行的对话深入探讨了 AI 的本质、存在以及科学与哲学的相互作用。

- **迫不及待想要体验**：用户对 Worldsim 的回归表达了焦急与热情的复杂情感，询问更新情况，同时讨论防止无限制访问的方法，并思考订阅模式的潜在成本，以保持服务的财务可行性并防止滥用。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.anthropic.com/claude/reference/client-sdks">Client SDKs</a>: 未找到描述</li><li><a href="https://www.mlexpert.io/prompt-engineering/memgpt">MemGPT - 为 LLMs 提供无限上下文（内存） | MLExpert - 助你攻克机器学习面试</a>：如何克服 LLMs 的上下文窗口大小限制？MemGPT 通过巧妙地管理不同的内存层级来帮助处理更长的对话。</li><li><a href="https://huggingface.co/fbjr/cohere_c4ai-command-r-plus-mlx-4bit-128g">fbjr/cohere_c4ai-command-r-plus-mlx-4bit-128g · Hugging Face</a>: 未找到描述</li><li><a href="https://worldsim.nousresearch.com/">world_sim</a>: 未找到描述</li><li><a href="https://websim.ai/">websim.ai</a>: 未找到描述</li><li><a href="https://a.co/d/e98NrUY">未找到标题</a>: 未找到描述</li><li><a href="https://www.google.com/amp/s/80.lv/articles/google-s-new-ai-can-generate-entire-2d-platformer-games/%3famp=1">Google 的新 AI 可以生成完整的 2D 平台游戏</a>：这款名为 Genie 的新模型可以根据单张图像提示词创建可玩的模型环境。</li><li><a href="https://www.gameb.wiki/index.php?title=An_Introduction_to_Game_B">Game B 简介 - Game B Wiki</a>: 未找到描述</li><li><a href="https://youtube.com/shorts/qE9gYuSVfyQ">The Box | 科幻动画短片</a>：视频摘要：该动画短片跟随“The Breacher”的视角，这是一个决心逃离被称为“Wor...”的模拟现实限制的角色。</li><li><a href="https://github.com/simonw/llm">GitHub - simonw/llm: 从命令行访问大语言模型</a>：从命令行访问大语言模型 - simonw/llm</li><li><a href="https://youtu.be/PHQweR1z7pI?si=ac4KikfzI5A4w4kZ">Worldsim 与越狱 Claude 3（含字幕） [-2-]</a>：个人学习</li><li><a href="https://youtube.com/shorts/oGng-eDRb0A">The Great Eclipse | 科幻动画短片</a>：视频摘要：这部动画短片探讨了一场思想和信仰之战，不是用武器，而是通过数据、辩论和模拟世界的力量。随着不同...</li><li><a href="https://www.nature.com/articles/s41598-019-56357-3">量子力学可以通过时空上的随机优化来理解 - Scientific Reports</a>: 未找到描述</li><li><a href="https://www.urantia.org/urantia-book/read-urantia-book-online>">首页</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1225739403645947915)** (977 条消息🔥🔥🔥): 

- **寻找 Stability Bot**：用户询问了关于生成图像的问题，并被引导至 <#1047610792226340935> 查看服务器状态，因为机器人目前处于离线状态。
- **对图像生成结果的好奇**：用户讨论了本地模型与 Dreamstudio 之间图像输出质量的差异，建议尝试开源 upscalers，并询问了各种技术的有效性。
- **对 Stable Diffusion 3 的期待**：对话显示 SD3 的发布预计还有 2-4 周的非正式时间，并讨论了其预期的改进和功能。
- **探索 SD 模型增强**：用户交流了关于训练 LoRAs 的信息，包括安装和实用性问题，并建议关注特定的 GitHub 仓库以获取指导。
- **在 UI 之间切换**：成员们分享了从 Automatic 1.1.1.1 切换到 StableSwarm 等其他 UI 的建议，强调了后者为新手提供的增强用户体验和功能。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://app.leonardo.ai/">Leonardo.Ai</a>：为您的项目创建具有前所未有的质量、速度和风格一致性的生产级视觉资产。</li><li><a href="https://huggingface.co/stabilityai/cosxl">stabilityai/cosxl · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/ByteDance/SDXL-Lightning">SDXL-Lightning - ByteDance 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>：我们提出了视觉自回归建模（VAR），这是一种新的生成范式，它将图像上的自回归学习重新定义为从粗到细的“下一尺度预测”或“下一分辨率预测”...</li><li><a href="https://huggingface.co/spaces/PixArt-alpha/PixArt-LCM">PixArt LCM - PixArt-alpha 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://civitai.com/models/3798/lexica-testica">Lexica Testica - 1.0 | Stable Diffusion Checkpoint | Civitai</a>：初始化自 OpenJourney v2，在从 Lexica art 首页（2023年1月）抓取的图像上进一步微调了 4000 步。擅长生成...</li><li><a href="https://tenor.com/view/frieren-wow-elf-peek-a-boo-gif-12265100463579712545">Frieren Wow GIF - Frieren Wow Elf - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/yoshi-mario-yoshis-island-super-smash-brother-super-smash-brother-n64-gif-21681448">Yoshi Mario GIF - Yoshi Mario Yoshis Island - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://openmodeldb.info/">OpenModelDB</a>：OpenModelDB 是一个社区驱动的 AI 放大模型数据库。我们的目标是提供一种比现有资源更好的方式来查找和比较模型。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1bnjm3i/comment/kwjb37c/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/FoundationVision/VAR/blob/main/demo_sample.ipynb">VAR/demo_sample.ipynb at main · FoundationVision/VAR</a>：[GPT 击败 Diffusion🔥] [视觉生成中的 Scaling Laws📈] “Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction”的官方实现 - FoundationVision/VAR</li><li><a href="https://github.com/LykosAI/StabilityMatrix">GitHub - LykosAI/StabilityMatrix: 适用于 Stable Diffusion 的多平台包管理器</a>：适用于 Stable Diffusion 的多平台包管理器 - LykosAI/StabilityMatrix</li><li><a href="https://www.federalregister.gov/documents/2023/03/16/2023-05321/copyright-registration-guidance-works-containing-material-generated-by-artificial-intelligence>">Federal Register :: 请求访问</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=8_V8CO_Dbdw">如何在 Google Colab 中运行 Stable Diffusion（免费）且不断连</a>：这里介绍了如何在 Colab 中编写自己的 Python Notebook，免费生成 AI 图像且不会断开连接。我们将使用来自 Hugging Face 的 Diffusers 库...</li><li><a href="https://forms.gle/avNEgKWp8nj3UAEg9">调查：比较 AI 生成的照片（扩散模型）</a>：这是一项旨在确定不同扩散模型（如 SD 1.5, SD 2.0, SDXL, Dall-e-3 和自定义微调模型）中更准确输出的调查。完成该调查需要几分钟时间...</li><li><a href="https://www.youtube.com/watch?v=gEwPGyWjK70">安装 Stability Matrix（Automatic 1111, ComfyUI, Fooocus 等的一键安装程序）</a>：此 Stability Matrix 应用程序专为 Windows 设计，允许安装和管理 Automatic 1111, ComfyUI 等文本生成图像 Web UI 应用...</li><li><a href="https://youtu.be/QIqoMSf4P88">与 Malcolm 和 Simone Collins 的对话（精简版）</a>：Malcolm 和 Simone 是 pronatalist.org、The Collins Institute for the Gifted 和 Based Camp Podcast 的创始人。All Outcomes Are Acceptable 博客：https:...</li><li><a href="https://github.com/altoiddealer/--sd-webui-ar-plusplus">GitHub - altoiddealer/--sd-webui-ar-plusplus: 在 sd-webui 中从预设选择图像纵横比</a>：在 sd-webui 中从预设选择图像纵横比。通过在 GitHub 上创建账户来为 altoiddealer/--sd-webui-ar-plusplus 的开发做出贡献。</li><li><a href="https://hforsten.com/identifying-stable-diffusion-xl-10-images-from-vae-artifacts.html">通过 VAE 伪影识别 Stable Diffusion XL 1.0 图像</a>：最近发布的全新 SDXL 1.0 文本生成图像模型会在图像中产生细小的伪影，而早期的 0.9 版本则没有这些伪影。</li><li><a href="https://github.com/nashsu/FreeAskInternet">GitHub - nashsu/FreeAskInternet: FreeAskInternet 是一个完全免费、私密且在本地运行的搜索聚合器和使用 LLM 的答案生成器，无需 GPU。用户可以提出问题，系统将进行多引擎搜索并将搜索结果合并到 </a></li>

ChatGPT3.5 LLM 并根据搜索结果生成答案。</a>: FreeAskInternet 是一个完全免费、私密且在本地运行的搜索聚合器和基于 LLM 的答案生成工具，无需 GPU。用户可以提出问题，系统将进行多...</li><li><a href="https://www.youtube.com/watch?v=kqXpAKVQDNU&list=PLXS4AwfYDUi5sbsxZmDQWxOQTml9Uqyd2">如何安装 Stable Diffusion - automatic1111</a>: 第 2 部分：如何使用 Stable Diffusion https://youtu.be/nJlHJZo66UA Automatic1111 https://github.com/AUTOMATIC1111/stable-diffusion-webui 安装 Python https://w...</li><li><a href="https://github.com/derrian-distro/LoRA_Easy_Training_Scripts">GitHub - derrian-distro/LoRA_Easy_Training_Scripts: 一个使用 Pyside6 制作的 UI，旨在简化 sd-scripts 中 LoRA/LoCon 及其他 LoRA 类型模型的训练</a>: 一个使用 Pyside6 制作的 UI，旨在简化 sd-scripts 中 LoRA/LoCon 及其他 LoRA 类型模型的训练 - derrian-distro/LoRA_Easy_Training_Scripts</li><li><a href="https://civitai.com/models/1493/sonicdiffusion">SonicDiffusion - V4 | Stable Diffusion Checkpoint | Civitai</a>: 在这里尝试！https://mobians.ai/ 加入 discord 获取更新、分享生成的图像，或者只是想聊天，亦或是想为帮助...</li><li><a href="https://github.com/Stability-AI/StableSwarmUI?tab=readme-ov-file#stableswarmui">GitHub - Stability-AI/StableSwarmUI: StableSwarmUI，一个模块化的 Stable Diffusion Web-User-Interface，重点在于使强力工具易于访问、高性能和可扩展性。</a>: StableSwarmUI，一个模块化的 Stable Diffusion Web-User-Interface，重点在于使强力工具易于访问、高性能和可扩展性。 - Stability-AI/StableSwarmUI</li><li><a href="https://github.com/camenduru/Open-Sora-Plan-replicate">GitHub - camenduru/Open-Sora-Plan-replicate</a>: 通过在 GitHub 上创建账户来为 camenduru/Open-Sora-Plan-replicate 的开发做出贡献。</li><li><a href="https://github.com/GarlicCookie/PNG-SD-Info-Viewer">GitHub - GarlicCookie/PNG-SD-Info-Viewer: PNG-SD-Info-Viewer 是一个旨在快速浏览带有 Stable Diffusion 生成图像相关元数据的 PNG 文件的程序。</a>: PNG-SD-Info-Viewer 是一个旨在快速浏览带有 Stable Diffusion 生成图像相关元数据的 PNG 文件的程序。 - GarlicCookie/PNG-SD-Info-Viewer</li><li><a href="https://github.com/GarlicCookie/SD-Quick-View">GitHub - GarlicCookie/SD-Quick-View: SD-Quick-View 是一个旨在非常快速地查看 Stable Diffusion 生成的图像并查看相关元数据的程序。</a>: SD-Quick-View 是一个旨在非常快速地查看 Stable Diffusion 生成的图像并查看相关元数据的程序。 - GarlicCookie/SD-Quick-View</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: 通过在 GitHub 上创建账户来为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://github.com/ronniebasak/ComfyUI-Tara-LLM-Integration/blob/main/README.md">ComfyUI-Tara-LLM-Integration/README.md at main · ronniebasak/ComfyUI-Tara-LLM-Integration</a>: 通过在 GitHub 上创建账户来为 ronniebasak/ComfyUI-Tara-LLM-Integration 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/HbOA5xdG8J">Reddit - 深入了解任何事物</a>: 未找到描述</li><li><a href="https://civitai.com/models/161068/newrealityxl-all-in-one-photographic">NewRealityXL ❗ 全能摄影 - ✔ 3.0 实验版 | Stable Diffusion Checkpoint | Civitai</a>: 重要提示：v2.x ---&gt; 主版本 | v3.x ---&gt; 实验版本 我需要你们的时间来彻底测试这个新的第 3 版，以了解所有...
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1225801606377898034)** (341 条消息🔥🔥): 

- **转换挫败感**：成员们对 Chat GPT 和 Claude 等机器学习模型表示不满，因为在将图像转换为 HTML 时，颜色保真度和边框圆角会丢失。有人幽默地建议将图像转换为 ASCII 艺术。

- **模型限制的共同挑战**：对话涉及服务器崩溃导致模型丢失的问题，原因是 Unsloth AI 和 Hugging Face 等平台上的模型保存问题，一些用户表达了对失去大量 fine-tuning 成果的遗憾。

- **对 LLM Vision Model 替代方案的好奇**：虽然视觉模型仍在 Unsloth AI 的 roadmap 上，但它们目前的优先级较低。用户讨论了像 Dreambooth 这样的替代方案，但没有找到确定的解决方案。

- **LLM 训练的 GPU 困扰**：幽默地讨论了避免笔记本电脑在模型训练期间过热的策略，包括搬到南极洲或使用空调。

- **对 Gradient Checkpointing 和更长上下文的期待**：关于 Unsloth AI 即将推出的功能的非官方暗示和预告引发了热议和期待，导致了关于可能实现方式以及对 GPU 资源有限用户的益处的讨论。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/liminerity/Mistral-quiet-star-demo">liminerity/Mistral-quiet-star-demo · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/liminerity/Mistral-quiet-star">liminerity/Mistral-quiet-star-demo · Hugging Face</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/gemma-bugs">Unsloth Fixing Gemma bugs</a>：Unsloth 正在修复 Google 的开源语言模型 Gemma。</li><li><a href="https://arstechnica.com/security/2024/03/researchers-use-ascii-art-to-elicit-harmful-responses-from-5-major-ai-chatbots/">ASCII art elicits harmful responses from 5 major AI chatbots</a>：LLM 被训练用于阻止有害响应。老式的图像可以绕过这些规则。</li><li><a href="https://github.com/uclaml/SPIN/blob/main/scripts/finetune.sh">SPIN/scripts/finetune.sh at main · uclaml/SPIN</a>：Self-Play Fine-Tuning (SPIN) 的官方实现 - uclaml/SPIN</li><li><a href="https://github.com/haotian-liu/LLaVA/blob/main/docs%2FFinetune_Custom_Data.md">LLaVA/docs/Finetune_Custom_Data.md at main · haotian-liu/LLaVA</a>：[NeurIPS'23 Oral] 旨在达到 GPT-4V 级别能力及更高水平的 Visual Instruction Tuning (LLaVA)。- haotian-liu/LLaVA</li><li><a href="https://github.com/facebookresearch/schedule_free">GitHub - facebookresearch/schedule_free: Schedule-Free Optimization in PyTorch</a>：PyTorch 中的 Schedule-Free 优化。通过在 GitHub 上创建账号为 facebookresearch/schedule_free 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=VNsWWb8g3Js">When GPT-5 is coming out | Sam Altman and Lex Fridman</a>：Lex Fridman Podcast 完整剧集：https://www.youtube.com/watch?v=jvqFAi7vkBc。请通过关注我们的赞助商来支持本播客：- Cloaked: https://cloa...</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>：快 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/stanfordnlp/pyreft">GitHub - stanfordnlp/pyreft: ReFT: Representation Finetuning for Language Models</a>：ReFT：语言模型的表示微调 - stanfordnlp/pyreft</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.push_to_hub">Trainer</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/gate369/Alpaca-Star">gate369/Alpaca-Star · Datasets at Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1225755450117460129)** (78 messages🔥🔥): 

- **分享偏好的 AI 新闻来源**：成员们分享了他们获取 AI 新闻的首选来源——**AI News** 和 **Reddit**，特别提到了用户 *localllama* 提供的持续更新。
- **辩论 Learning Rate Schedulers 的优劣**：一位成员分享了实验不同 Learning Rate Schedulers（**linear**、**cosine with restarts** 和 **constant**）的结果，指出 **constant** 似乎出人意料地最适合他们的模型（专注于多语言通用助手）。
- **提出关于 DPO 微调的问题**：人们对使用 **DPO** (Dynamic Positional Offsets) 微调的模型为何在 **Open LLM Leaderboard** 上表现出色感到好奇，即使基础模型的分数较低。有人建议专有数据集可能是一个影响因素。
- **讨论 Benchmark 对模型感知的影响**：关于 Benchmark 的讨论表明，它们并不总是与感知质量一致；低分模型可能仍然非常有效。此外，还表达了对模型可能被测试数据“污染”以及 Benchmark 僵化性的担忧。
- **提到 Unsloth 招聘和开源贡献**：成员们讨论了即将进行的全栈开发人员和开发者倡导者（Developer Advocate）的招聘，并澄清这些角色将负责为开源社区做出贡献并构建 Unsloth Pro 平台。

**提到的链接**：<a href="https://github.com/unslothai/unsloth/wiki">Home</a>：快 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1225765845653917788)** (374 messages🔥🔥):

- **AI 模型私有托管**：一位成员宣布他们正在自己的服务器上托管其 AI 项目，以维护一些未发布模型的隐私。他们暗示在托管更多高质量模型后，未来可能会整合广告，并分享了一个链接：[Hircoir Text-to-Speech](https://tts.hircoir.eu.org/)。

- **推理代码灵活性获赞**：Unsloth AI 的推理代码因其速度和易用性而受到称赞。成员们被提醒可以根据需要修改 temperature 等推理设置，并根据需要使用 Generative Guided Unsupervised Fine-tuning (GGUF)。

- **关于模型合并的讨论**：一场对话涉及了 AI 模型的潜在合并策略，建议包括将不同模型之间的差异相互应用。基于以往的经验，对此话题的看法从怀疑到乐观不等。

- **用户在代码方面遇到困难**：一位用户表达了在编码方面的困难，特别是与批量推理（batch inference）的模型参数调整有关。他们被引导至 Unsloth 的 GitHub 以获取指导，强调了 `model.generate` 的实用性。

- **批量推理说明**：成员们讨论了如何有效地执行批量推理。纠正了 `num_return_sequences` 的用法，指出它仅适用于单个 prompt，而不适用于应该“全部堆叠在一起”的批量 prompt。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1n8vXmEQ-rAXdytw25M3y6ff4k1acYtxt?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://tts.hircoir.eu.org/">HirLab - Convertidor de Texto a Voz por Hircoir</a>：HirLab 是一个基于人工智能的文本转语音平台。它可以快速准确地将文本转换为语音。</li><li><a href="https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch">Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch</a>：低秩自适应 (LoRA) 是一种机器学习技术，通过调整参数来修改预训练模型（例如 LLM 或 Vision Transformer），以更好地适应特定的、通常较小的数据集...</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#">Customizing LLMs - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-using-a-huggingface-llm">Customizing LLMs - LlamaIndex</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/generation_strategies">Text generation strategies</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=rANv5BVcR5k">Mistral Fine Tuning for Dummies (with 16k, 32k, 128k+ Context)</a>：在我们最新的教程视频中探索如何使用您自己的数据轻松微调语言模型 (LLM)。我们深入探讨了一种经济高效且可持续的方法...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/save.py#L111">unsloth/unsloth/save.py at main · unslothai/unsloth</a>：速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>：速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>：速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/issues/26877">Mistral with flash attention 2 and right padding · Issue #26877 · huggingface/transformers</a>：系统信息 transformers 版本：4.34.0 平台：Linux-5.4.0-148-generic-x86_64-with-glibc2.31 Python 版本：3.10.13 Huggingface_hub 版本：0.17.3 Safetensors 版本：0.4.0 Accelerate 版本...</li><li><a href="https://huggingface.co/datasets/pharaouk/UltraInteract_sft">pharaouk/UltraInteract_sft · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/model#model-instantiation-dtype">Models</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/267#issuecomment-2034047189">Batch inference produces nonsense results for unsloth/mistral-7b-instruct-v0.2-bnb-4bit · Issue #267 · unslothai/unsloth</a>：您好，在使用以下代码加载模型后：from unsloth import FastLanguageModel import torch model, tokenizer = FastLanguageModel.from_pretrained( model_name = &quot;unsloth/mistral-7b-instruct-v0.2-bnb...
</li>
</ul>

</div>
  

---

**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1226796733393010688)** (2 条消息): 

- **介绍 Aurora-M**：根据美国关于 AI 的行政命令，开发了一款名为 **Aurora-M** 的新型 [155 亿参数开源多语言语言模型](https://arxiv.org/abs/2404.00399)。它展示了单语言安全对齐的跨语言影响，并突破了 2 万亿训练 Token。
- **跨语言安全影响得到验证**：团队发现，在英语上进行的安全对齐微调不仅增强了英语的安全性，还增强了德语等其他语言的安全性。这被认为是*单语言安全对齐具有跨语言影响*的首个证据。
- **同行认可**：社区对 *Aurora-M* 项目表示支持，并给出了“great work! 🔥”等积极反馈。
- **Aurora-M 的后续进展**：该项目旨在以 *Aurora-M* 为基础，通过使用 LoRA 训练 Mixture of Experts 并进行后续合并。目前正在征求 Unsloth AI 社区的反馈，特别是关于使用 LoRA 微调 Notebook 的建议。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/__z__9/status/1774965364301971849?s=20">来自 ً ‎ (@__z__9) 的推文</a>: 新的预印本！首个经过多语言红队测试、开源且持续预训练的 LLM —— **Aurora-M**，符合白宫关于安全、可靠和值得信赖的开发的行政命令...</li><li><a href="https://arxiv.org/abs/2404.00399">Aurora-M: 首个根据美国行政命令进行红队测试的开源多语言语言模型</a>: 预训练语言模型支撑着多种 AI 应用，但其高昂的训练计算成本限制了可及性。BLOOM 和 StarCoder 等倡议旨在使...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1226587652262985858)** (148 条消息🔥🔥): 

- **投资难题**：关于 **Jamba** 和 **Mamba** 的价值引发了热烈辩论，观点从怀疑到谨慎乐观不等。虽然一位成员指出 Jamba 背后的公司 AI21 Labs 筹集了 1.55 亿美元，这可能表明了市场兴趣，但其他人持批评态度，认为此类投资可能被误导了。([AI21 Labs 融资情况](https://techcrunch.com/2023/08/30/generative-ai-startup-ai21-labs-lands-155m-at-a-1-4b-valuation/))

- **量化模型成为焦点**：量化模型和优化技术（如 AQLM）的可行性引发了复杂的感受。一位成员指出，针对特定用例进行微调具有理想的 ROI，并分享到，尽管前期成本很高，但对微调进行适当投资可以产生显著回报。([AQLM Mixtral](https://huggingface.co/ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf))

- **对模型架构的不同看法**：成员们评估了引入部分优化的 **MoE** 的必要性，并讨论了新架构的实现，其中一位成员建议采取谨慎态度，观察此类模型是否会流行。

- **自动化 AI 工程的未来**：对话简要触及了创建**能够辅助编写优化 Kernel** 或处理 Triton 代码生成等繁重计算任务的**微调模型**的潜力，旨在开拓自动化 AI 工程解决方案。

- **实用性胜过炒作**：整个讨论中有一种强烈的倾向，强调实用、经过良好优化的模型比“炒作”的初创公司项目更重要。共识似乎倾向于支持经过验证、可扩展的架构（如 **Transformers**），并在适当的时候增加 **MoE** 实现。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf">ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf · Hugging Face</a>: 未找到描述</li><li><a href="https://techcrunch.com/2023/08/30/generative-ai-startup-ai21-labs-lands">Generative AI startup AI21 Labs lands $155M at a $1.4B valuation | TechCrunch</a>: AI21 Labs 是一间与 OpenAI 和 Anthropic 等生成式 AI 玩家竞争的公司，已筹集了 1.55 亿美元资金。</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/alpindale/Mistral-7B-Instruct-v0.2-AQLM-2Bit-1x16">alpindale/Mistral-7B-Instruct-v0.2-AQLM-2Bit-1x16 · Hugging Face</a>: 未找到描述</li><li><a href="https://techcrunch.com/2023/08/30/generative-ai-startup-ai21-labs-lands-155m-at-a-1-4b-valuation/">Generative AI startup AI21 Labs lands $155M at a $1.4B valuation | TechCrunch</a>: AI21 Labs 是一间与 OpenAI 和 Anthropic 等生成式 AI 玩家竞争的公司，已筹集了 1.55 亿美元资金。</li><li><a href="https://discuss.pytorch.org/t/choice-of-torch-compile-vs-triton/195604/2">Choice of torch.compile vs. triton</a>: 在 GPU 上，torch.compile() 会应用各种编译器优化，其中最重要的是 CUDA graphs 和 fusions。特别是 fusions 是通过生成 Triton kernels 代码来完成的，因此 torch.c...
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1225750743047208991)** (488 messages🔥🔥🔥): 

- **GPU Offload 困惑已解决**：用户在 LM Studio 中优化 GPU 使用方面得到了指导，建议调整 "n_gpu_layers" 和 "GPU Offloading" 设置以获得更好的性能，缓解了对过度依赖集成显卡或未能利用 Nvidia GPU 的担忧。建议指出，在可能的情况下，应确保模型完全 Offload 到 GPU 以获得更快的速度。
  
- **大型 LLM 和多 GPU 设置**：关于运行大型模型（如 70b 模型）的讨论围绕 VRAM 的重要性展开，用户分享了他们的经验和配置，包括双 RTX 4060 Ti 16GB 设置。共识是 VRAM 越多，可以运行的模型就越大，且不会因为使用系统 RAM 而变慢。

- **探索模型能力**：关于较小容量模型是否能记住用户名称的疑问，引出了关于使用 System Prompts 指导模型执行所需行为的解释。澄清说明 LM Studio 不支持模型的实际学习或训练；然而，编写详细的 Prompts 可以实现与学习类似的效果。

- **用于编程目的的 AI**：用户推荐 OpenAI 的 GPT-4，因其在编程方面的精通，尽管也提到了相关成本。讨论强调了缺乏同等的开源模型，反映了模型能力与成本之间的权衡。

- **多样化的用法和集成问题**：对话范围从在独立驱动器上设置模型、错误调试，到询问模型与 EXL2 和 GGUF 等不同格式的兼容性。集成主题包括将 LM Studio 与 OpenDevin 等其他工具连接，以及将 LLM 用于文本分析和小说创作等任务。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://klu.ai/glossary/grouped-query-attention">什么是 Grouped Query Attention (GQA)？ — Klu</a>: 未找到描述</li><li><a href="https://lmstudio.ai/docs/">文档 | LM Studio</a>: 技术参考</li><li><a href="https://huggingface.co/LoneStriker/miqu-1-70b-sf-4.25bpw-h6-exl2">LoneStriker/miqu-1-70b-sf-4.25bpw-h6-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/TheBloke/MXLewdMini-L2-13B-GGUF">TheBloke/MXLewdMini-L2-13B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1avdwx2/new_try_where_is_the_quantization_god/">Reddit - 深入探讨一切</a>: 未找到描述</li><li><a href="https://huggingface.co/TheBloke/MXLewdMini-L2-13B-GGUF#prompt-template-alpaca">TheBloke/MXLewdMini-L2-13B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/16ubkyq/nvlink_bridge_worth_it_for_dual_rtx_3090/">Reddit - 深入探讨一切</a>: 未找到描述</li><li><a href="https://github.com/Pythagora-io/gpt-pilot/issues/807#issuecomment-2037824538">[Bug]: LLM Studio 无法连接 · Issue #807 · Pythagora-io/gpt-pilot</a>: 版本 VisualStudio Code 扩展 操作系统 Windows 11 发生了什么？通过将 endpoint 和 api key 从 OpenAI 更改为 LLM Studio：如果使用 OPENAI_ENDPOINT=http://localhost:1234/v1 那么...</li><li><a href="https://github.com/enricoros/big-AGI/blob/main/docs/config-local-lmstudio.md">big-AGI/docs/config-local-lmstudio.md at main · enricoros/big-AGI</a>: 由最先进模型驱动的生成式 AI 套件，提供高级 AI/AGI 功能。其特点包括 AI 角色、AGI 功能、多模型聊天、文本转图像、语音、响应流式传输...</li><li><a href="https://rentry.org/LMSTudioFAQ">非官方 LM Studio FAQ！</a>: 欢迎来到非官方 LM Studio FAQ。在这里，你可以找到我们在 LM Studio Discord 中收到的最常见问题的答案。（此 FAQ 由社区管理）。LM Studio 是一款免费的闭源...</li><li><a href="https://www.humblebundle.com/books/machine-learning-ai-deep-learning-and-llm-pearson-books?hmb_source=&hmb_medium=product_tile&hmb_campaign=mosaic_section_1_layout_index_2_layout_type_threes_tile_index_2_c_machinelearningaideeplearningandllmpearson_bookbundle">Humble 科技书籍捆绑包：Pearson 出版的机器学习、AI、深度学习和 LLM</a>: 通过这些关于 AI、机器学习和其他计算机科学前沿话题的书籍，紧跟定义未来的技术步伐！</li><li><a href="https://github.com/jakobdylanc/discord-llm-chatbot">GitHub - jakobdylanc/discord-llm-chatbot: llmcord.py • 和你的朋友一起与 LLM 聊天！</a>: llmcord.py • 和你的朋友一起与 LLM 聊天！通过在 GitHub 上创建账号为 jakobdylanc/discord-llm-chatbot 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA">Reddit - 深入探讨一切</a>: 未找到描述</li><li><a href="https://github.com/OpenDevin/OpenDevin/issues/419">使用 LM Studio 时遇到问题 · Issue #419 · OpenDevin/OpenDevin</a>: 描述 Bug：连接到 LM Studio 时出现问题。复现步骤：1. 在 LM Studio 上启动服务器 2. 在 OpenDevin 上启动前端和后端 3. 预期行为：OpenDevin 询问我想要构建什么...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6491">由 Carolinabanana 添加 Command R Plus 支持 · Pull Request #6491 · ggerganov/llama.cpp</a>: 更新了张量映射，为 GGUF 转换添加了 Command R Plus 支持。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1225757263030128641)** (103 条消息🔥🔥):

- **Command R+ 即将到来**：备受期待的 **Command R+** 模型在与 llama.cpp 的集成方面仍面临挑战，但目前已有一个[可用分支](https://github.com/pmysl/c4ai-command-r-plus-GGUF)可以运行。这是一个庞大的 104B 模型，需要强大的硬件配置。
- **GGUF 格式的特性与提供的帮助**：关于量化格式无障碍实现的讨论促使成员们分享经验，并为那些希望运行 GGUF 模型的人提供协助。社区对像 TheBloke 这样悄然消失的贡献者表示好奇和担忧。
- **硬件烦恼与幽默**：社区成员开玩笑说他们的 LLM 硬件爱好非常昂贵，将支出比作购买 BMW M4 等奢侈品。此外，还有关于寻找廉价解决方案的建议，例如使用 Nvidia P40 显卡。
- **AI 故事创作追求**：一位成员表达了对使用 AI 模型进行创意故事创作的兴趣，并收到了关于选择具有大上下文的模型以及使用 [MemGPT](https://github.com/cpacker/MemGPT) 等内存管理工具的建议。
- **对视觉适配器的好奇**：有人提出了关于 cjpais llava 模型中视觉适配器功能的问题，随后询问了如何复制特定视频中展示的视觉任务能力。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/pmysl/c4ai-command-r-plus-GGUF">pmysl/c4ai-command-r-plus-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>: 未找到描述</li><li><a href="https://huggingface.co/TheB">TheB (Pastor B)</a>: 未找到描述</li><li><a href="https://huggingface.co/google/gemma-1.1-7b-it">google/gemma-1.1-7b-it · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-32B-Chat-GGUF/tree/main">Qwen/Qwen1.5-32B-Chat-GGUF at main</a>: 未找到描述</li><li><a href="https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF">TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://llm.extractum.io/model/TheBloke%2FWizard-Vicuna-7B-Uncensored-GPTQ,1e2RcN80JhFWYaq1IBixLq">Wizard Vicuna 7B Uncensored GPTQ By TheBloke: Benchmarks and Detailed Analysis. Insights on Wizard Vicuna 7B Uncensored GPTQ.</a>: LLM 卡片: 7b LLM, VRAM: 4.5GB, Context: 2K, License: other, Quantized, Uncensored.</li><li><a href="https://huggingface.co/TheBloke/goliath-120b-GGUF">TheBloke/goliath-120b-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/j4ys0n">j4ys0n - Overview</a>: 区块链工程师。j4ys0n 拥有 62 个代码仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6387">ggml : update mul_mat_id to use the same tensor for all the experts by slaren · Pull Request #6387 · ggerganov/llama.cpp</a>: 将内存中专家的存储方式从每个专家一个张量更改为包含所有专家的单个 3D 张量。这将允许我们支持具有大量专家的模型，如 qwen2moe。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1225884680008503366)** (1 条消息): 

- **LM Studio 社区页面上线**：LM Studio 团队在 Hugging Face 上推出了全新的 "lmstudio-community" 页面，提供最新的 **GGUF 量化版本**。用户可以通过在 LM Studio 内搜索 `lmstudio-community` 来查找并试用这些模型；点击[此处](https://huggingface.co/lmstudio-community)查看。
- **@bartowski1182 加入担任 LLM 归档员**：在 Twitter 上宣布，@bartowski1182 将担任 LM Studio 的常驻 **LLM 归档员**，协助更新新的 Hugging Face 社区页面。查看 Twitter 公告[请点击此处](https://x.com/LMStudioAI/status/1776324680124694654)。

**提到的链接**: <a href="https://x.com/LMStudioAI/status/1776324680124694654">来自 LM Studio (@LMStudioAI) 的推文</a>: 如果你在这里待得足够久，你可能会像我们一样想念 @TheBlokeAI 🥲。我们和 @bartowski1182 决定尝试填补这个空白。我们很高兴能分享这个新...

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1225804230238011452)** (25 条消息🔥): 

- **对 LM Studio GUI 的赞赏**：成员们发现 **LM Studio** 的表现优于 **oogabooga** 和 **Faraday** 等其他本地 LLM GUI，即使使用相同的模型和指令，也能获得高质量的结果。

- **功能扩展请求**：建议为 LM Studio 增加**文件读取支持**以及各种模式，如*文本转图像*、*图像转文本*和*文本转语音*功能，寻求类似于现有工具 **Devin** 的改进。

- **Vision 模型令人惊叹**：成员们对测试的 Vision 模型感到兴奋，并感谢 LM Studio 提供的实用功能。Vision 模型可在 [Hugging Face](https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1) 上获取。

- **下载问题排查**：有用户在 Pop!_OS 22.04 LTS 上尝试下载 LM Studio 的 Linux Beta 版本时遇到问题。该问题被确定为网站的一个 Bug，并提供了 AppImage 的直接链接（[点击此处](https://releases.lmstudio.ai/linux/0.2.19/beta/LM_Studio-0.2.19-Preview-1.AppImage)）。

- **支持无审查模型**：有请求希望 LM Studio 支持名为 **Dolphin 2.8 Mistral 7b v0.2** 的新型无审查模型，该模型可在 Hugging Face 上获取（[无审查模型](https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02)）。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta 版本发布</a>：未找到描述</li><li><a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://releases.lmstudio.ai/linux/0.2.19/beta/LM_Studio-0.2.19-Preview-1.AppImage">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1">Vision Models (GGUF) - lmstudio-ai 集合</a>：未找到描述</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02">cognitivecomputations/dolphin-2.8-mistral-7b-v02 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1226893471496011857)** (2 条消息): 

- **关于用于股市分析的 LLM 咨询**：一位成员提出了关于如何训练 **Large Language Models (LLMs)** 以解释股市 **OHLC**（开盘价、最高价、最低价、收盘价）价格的问题。
- **关于带指标的 LLM 训练请求**：同一位成员询问了在股市分析的 LLM 训练过程中加入**指标 (indicators)** 的相关事宜。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1225762035325866015)** (39 条消息🔥): 

- **在 LM Studio 中混合使用 GPU**：一位成员分享道，LM Studio 可以检测不同 GPU 显卡（如 **RTX 4060 Ti** 和 **GTX 1070**）的累积 VRAM，与结合使用 VRAM 和 CPU/RAM 相比，这提升了性能。

- **Advanced Matrix Extensions 的兼容性查询**：一位成员询问 **LM Studio** 是否可以利用第四代 **Xeon 处理器**中的 **Intel Advanced Matrix Extensions (AMX)**。

- **ROCm 支持与混合 GPU 的探索**：用户讨论了各种 **RX 5000** 和 **RX 6000** 系列 AMD GPU 在 **LM Studio** 中对 **ROCm** 支持的兼容性，并指出并非所有显卡都受支持。

- **CPU 指令集支持问题**：一位用户遇到了其处理器 **Xeon E5-2690 v2** 似乎缺乏 AVX2 支持的问题，这与其之前使用 **LM Studio** 的经验相矛盾。建议手动安装 **llama.cpp** 作为权宜之计。

- **关于使用 Tesla P 系列 GPU 进行模型训练和微调的辩论**：有人提到关于使用 **Tesla P40 GPU** 进行模型训练和微调的争议，一些用户声称获得了成功，而另一些用户则指出由于过时的 **CUDA** 支持而存在的局限性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1alcwc1/comment/kpenylq/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.hwinfo.com/">HWiNFO - 免费系统信息、监控和诊断</a>：免费硬件分析、监控和报告。深入的硬件信息、实时系统监控、报告等</li><li><a href="https://github.com/ggerganov/llama.cpp.git">GitHub - ggerganov/llama.cpp: C/C++ 实现的 LLM 推理</a>：C/C++ 实现的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 做出贡献。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1225887364434624541)** (30 条消息🔥):

- **Beta 版本号混淆问题已解决**：针对版本号混淆的问题，向成员澄清了 Beta 版本可能不会立即反映正确的版本号，正式发布时会进行更改。
- **LM Studio Beta 0.2.19 发布**：**LM Studio 0.2.19 Beta** 已发布，支持通过本地服务器进行 Text Embeddings，可在 Beta Releases 栏目下载。
- **ROCm 版本延迟但广受好评**：提到 **ROCm 版本**往往比主版本落后一个版本，但尽管如此且存在一些 Bug，成员们仍认为其表现出色且易于使用。
- **MacOS 在 0.2.19 版本下崩溃**：一位用户报告了 **MacOS 上 0.2.19** 版本的反复崩溃问题，这与特定模型的 Context Window 有关，表明这是一个综合性问题。
- **量化 Embedding 模型与 GGUF 转换**：关于为 GGUF 格式量化更多 Embedding 模型的积极讨论促成了新模型的转换与分享，Model Cards 将很快发布。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://tenor.com/view/leonardo-dicaprio-clapping-clap-applause-amazing-gif-16078907558888063471">Leonardo Dicaprio 鼓掌 GIF - Leonardo Dicaprio Clapping Clap - 发现并分享 GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1225890061686538282)** (17 条消息🔥): 

- **更好的 Multi-Agent 系统即将到来**：.j4ys0n 宣布即将发布其自带**用户界面 (UI)** 的 Multi-Agent 系统，认为这是现有系统问题的解决方案，并强调它不像 **CrewAI** 那样需要用户编写代码。
- **UI 带来的易用性优势**：.j4ys0n 的工具定位为“傻瓜式”解决方案，无需编程即可提供易用性，而 **CrewAI** 仍需要代码。
- **域名注册警示**：.j4ys0n 表示在域名注册之前不愿分享新项目的截图，以避免“域名抢注”，而 heyitsyorkie 则强调了尽快确保域名所有权的重要性。
- **开发重心**：.j4ys0n 提到将更多时间投入到项目开发而非日常工作任务中，认为鉴于目前取得的进展，这是**正确的选择**。
- **修改 datamodel.py 作为临时解决方案**：mmonir 分享了一个问题的解决方案，建议修改 `datamodel.py`（将 `max_tokens` 更改为 `3000`），并引用了 GitHub 上关于 **Autogen Studio** 的一个未解决 Bug ([Bug 报告在此](https://github.com/microsoft/autogen/issues/2050))。

**提到的链接**：<a href="https://github.com/microsoft/autogen/issues/2050">[Bug]: [autogenstudio] agent llm 发送 max_tokens: null · Issue #2050 · microsoft/autogen</a>：描述 Bug：当 max_tokens 参数为 None 时，Agent 发送的 /v1/chat/completions 帧中 max_tokens 为 null。在这种情况下，LLM 无法理解并在第二个 Token 后停止...

  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1226634657567870986)** (4 条消息): 

- **在寻找 Notebook 吗？**：一位成员询问是否有 Notebook，可能是想寻求共享资源或工作示例。
- **Substack 文章预告**：另一位成员提供了他们撰写的 [Substack 文章](https://substack.com/home/post/p-143137776?source=queue)，暗示其中可能包含有价值的见解或信息。

**提到的链接**：<a href="https://substack.com/home/post/p-143137776?source=queue">从 OpenAI API 切换到本地 LLM</a>：关于我们上一篇使用 LangChain 和 Node 构建 RAG Agent 文章的小型后续博文。

  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1225785291629920318)** (97 条消息🔥🔥): 

- **GPU Target 覆盖咨询**：一位成员询问是否可以像 Linux 上那样通过 `HCC_AMDGPU_TARGET=gfx1030` 覆盖 GPU Target，并引用了 [Reddit](https://www.reddit.com/r/Amd/comments/13e6jav/comment/jn8v5n5/) 上的讨论。然而，官方澄清在当前的 Linux 版本中，用户只能使用 OpenCL 进行 GPU 加速。

- **LM Studio 0.2.19 ROCm 预览 Beta 版发布**：发布了 **LM Studio 0.2.19 ROCm 预览 Beta 版**的公告，重点介绍了对 Text Embedding 模型的新支持、针对 ROCm iGPU 问题的候选修复以及其他 Bug 修复。社区成员被告知可以从 [LM Studio with ROCm](https://lmstudio.ai/rocm) 下载 Beta 版，尽管版本号可能仍显示为 0.2.18。

- **关于不同 GPU 的 ROCm 支持的困惑**：社区成员正在讨论并询问新版 ROCm 是否支持混合使用不同的 GPU（如 RX 5000 和 RX 6000 系列），其中一位成员表示使用 hipblas 和 Vulkan 成功运行了混合 AMD GPU。

- **AMD 对限制 GRE 至 2.8GHz 保持沉默**：用户对 AMD 将 GRE 限制在 2.8 GHz 表示沮丧，并希望发布自定义 BIOS。一位成员表示，只有 AMD 内部人员冒着丢掉工作的风险才可能发布此类 BIOS。

- **ROCm 0.2.19 Beta 调试进行中**：多名用户报告了最新 ROCm beta 版出现 "exit 42" 错误，促使官方分享了一个详细调试版本 (verbose debug build) 以供进一步调查。鼓励参与者下载该详细版本，尝试加载模型，并提交应用日志进行故障排除。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/rocm">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://files.lmstudio.ai/windows/LM-Studio-0.2.19-Rocm-Beta-Verbose.exe/beta/LM-Studio-0.2.19-Rocm-Beta-Verbose.exe">未找到标题</a>：未找到描述</li><li><a href="https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html">系统要求 (Windows) — HIP SDK Windows 安装</a>：未找到描述</li><li><a href="https://community.amd.com/t5/ai/how-to-run-a-large-language-model-llm-on-your-amd-ryzen-ai-pc-or/ba-p/670709">如何在你的 AMD Ryzen™ AI PC 或 Radeon 显卡上运行大语言模型 (LLM)</a>：你知道可以在你的 Ryzen™ AI PC 或 Radeon™ 7000 系列显卡上运行你自己的基于 GPT 的 LLM 驱动的 AI 聊天机器人实例吗？AI 助手正迅速成为必不可少的资源...</li><li><a href="https://www.reddit.com/r/Amd/comments/13e6jav/comment/jn8v5n5/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1225910988717559972)** (3 条消息): 

- **模型发布中心**：发布了一系列新模型，包括 **Starling-LM 7B**、**c4ai command r v01**、**stable-code-instruct-3b**、**dolphin 2.8 mistral 7b v02** 和 **Hyperion 3.0 Mistral 7B**。公告邀请用户[查看这些模型](https://huggingface.co/lmstudio-community)并关注后续更新。

- **介绍 Qwen 1.5 32B Chat**：发布了一个新模型 **Qwen 1.5 32B Chat**，它是 Qwen2 家族的一部分，具有增强的多轮对话能力。感兴趣的用户可以在[模型卡片和 LM Studio 应用](https://huggingface.co/lmstudio-community/Qwen1.5-32B-Chat-GGUF)中找到更多详情。

- **Gemma 在 2B 规模表现出色**：Google 的 **Gemma 1.1 2B** 模型表现令人印象深刻，仅使用 *3GB 内存* 即可高速输出连贯内容。该模型已上线，但正如[模型页面](https://huggingface.co/lmstudio-community/gemma-1.1-2b-it-GGUF)所述，7B 版本在发布前需要在 LM Studio 设置中进行优化调整。
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1225904914312597666)** (4 条消息):

- **丰富的资源与演示**：社区分享了各种资源，包括 [神经符号 Agent 系统仓库](https://github.com/SynaLinks/HybridAGI)、PyG 生态系统内的数据集集成，以及具有函数调用（function calling）能力的模型 Octopus 的演示。其他内容还包括超图数据集的可视化、用于大语言模型的 TensorLM Gradio UI，以及宣布推出 Aurora-M，这是一个多语言持续预训练语言模型。
- **技术与思想领导力展示**：社区成员发布了一个新的多主题图像节点包，发表了关于 AI 时代电影未来的 TED 演讲，并发布了一个用于复现研究论文的开源仓库。其他亮点包括关于使用虚拟容器保护 Python 应用的视频、SaaS 模板演示、寻线机器人演示，以及 DagsHub + Colab 在数据管理方面的深度集成。
- **简化工作的软件**：社区提供了 LLMinator（一个上下文感知的流式聊天机器人）和 ClipboardConqueror（一个减少上下文切换的工具），以提高使用大语言模型的工作效率。
- **发人深省的阅读材料与 AI 工具**：成员们贡献了讨论各种 AI 相关主题的博客文章，例如使用 LASER 技术评估 SVD 压缩以及理解扩散模型（diffusion models）。此外还分享了关于使用 HuggingFace 构建自定义架构以及 AI 计算复杂度层级的文章。
- **关注多语言 LLM 创新**：Aurora-M 的博客文章被建议作为下一个阅读小组的潜在主题，强调了多语言和安全 AI 发展的重要性。

**提到的链接**：<a href="https://huggingface.co/blog/mayank-mishra/aurora">Aurora-M：首个经过拜登-哈里斯行政命令红队测试的开源多语言语言模型</a>：未找到描述

---

**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1225759129054740480)** (372 messages🔥🔥): 

- **探索使用 SageMaker 和 TGI 进行部署**：一位用户正在考虑使用 TensorRT 配合 SageMaker 而非 TGI 部署模型的可行性，并寻求在基于网页的云计算资源上更新其内核版本的方法。
- **对针对 PDF 的自定义 ChatGPT 的兴趣**：为了开发一个专门针对 PDF 的独特 ChatGPT 应用，一位用户正在征集创意，以便在大学竞赛中脱颖而出。
- **寻求 ML 硬件基准测试工具**：用户正在讨论用于 ML/AI 任务的硬件基准测试工具；推荐将 MLPerf 作为开源（FOSS）基准测试套件，其中包含 GPT-J 6B 和 Llama 2 70B 推理的测试轨道。
- **SageMaker 多 GPU 训练问题**：关于在使用 SageMaker 和 diffusers 进行多 GPU 训练时遇到的问题进行了交流，包括 SIGSEGV 错误和环境变量消息。
- **请求响应中的 Token 计数信息**：一位用户询问 Hugging Face SageMaker 库在调用 `.predict` 时是否可以提供响应中的 Token 数量。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://arxiv.org/abs/2403.10853">Just Say the Name: Online Continual Learning with Category Names Only via Data Generation</a>：只需说出名称：通过数据生成仅使用类别名称进行在线持续学习。在现实场景中，由于成本高昂，为持续学习进行大规模手动标注是不切实际的。虽然受大规模网络监督训练影响的先前技术建议...</li><li><a href="https://huggingface.co/HuggingFaceTB/cosmo-1b">HuggingFaceTB/cosmo-1b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/sagemaker/en/getting-started">Train and deploy Hugging Face on Amazon SageMaker</a>：在 Amazon SageMaker 上训练和部署 Hugging Face：未找到描述</li><li><a href="https://discuss.huggingface.co/t/runtimeerror-expected-tensor-for-argument-1-indices-to-have-one-of-the-following-scalar-types-long-int-but-got-mpsfloattype-instead-while-checking-arguments-for-embedding/80417">RuntimeError: Expected tensor for argument #1 &#39;indices&#39; to have one of the following scalar types: Long, Int; but got MPSFloatType instead (while checking arguments for embedding)</a>：我正尝试通过输入图像和文本来训练一个多模态模型以输出文本。这是我的架构；（假设 batch size=1）我使用 ViT（来自 Hugging Face）来转换图像 (1, 3...</li><li><a href="https://cookbook.openai.com/examples/question_answering_using_embeddings">Question answering using embeddings-based search | OpenAI Cookbook</a>：使用基于 Embedding 搜索的问答 | OpenAI Cookbook：未找到描述</li><li><a href="https://huggingface.co/NexaAIDev/Octopus-v2">NexaAIDev/Octopus-v2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/sagemaker/en/inference">Deploy models to Amazon SageMaker</a>：将模型部署到 Amazon SageMaker：未找到描述</li><li><a href="https://huggingface.co/docs/accelerate/v0.28.0/en/package_reference/launchers#accelerate.notebook_launcher">Launchers</a>：未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/18o6z49/is_it_possible_to_queue_batch_img2img_with_a_new/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://huggingface.co/blog/noob_intro_transformers">Total noob’s intro to Hugging Face Transformers</a>：纯小白的 Hugging Face Transformers 入门指南：未找到描述</li><li><a href="https://huggingface.co/facebook/bart-large-cnn">facebook/bart-large-cnn · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blog/embedding-quantization">Binary and Scalar Embedding Quantization for Significantly Faster &amp; Cheaper Retrieval</a>：二进制和标量 Embedding 量化，实现显著更快且更便宜的检索：未找到描述</li><li><a href="https://learnbybuilding.ai/tutorials/rag-">no title found</a>：未找到描述</li><li><a href="https://github.com/Haoming02/sd-webui-diffusion-cg">GitHub - Haoming02/sd-webui-diffusion-cg: An Extension for Automatic1111 Webui that performs color grading based on the latent tensor value range</a>：GitHub - Haoming02/sd-webui-diffusion-cg：一个用于 Automatic1111 Webui 的扩展，基于 latent tensor 值范围进行调色 - Haoming02/sd-webui-diffusion-cg</li><li><a href="https://huggingface.co/docs/diffusers/en/tutorials/basic_training">Train a diffusion model</a>：训练一个 Diffusion 模型：未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusi">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Yakova/ollama-mistral">Streamer - a Hugging Face Space by Yakova</a>：未找到描述</li><li><a href="https://blog.salad.com/ollama-deploy-chatgpt/">Your own ChatGPT for $0.04/hr - With Ollama, ChatUI &amp; Salad</a>：每小时 0.04 美元构建你自己的 ChatGPT - 使用 Ollama, ChatUI 和 Salad：我们探索了如何仅用每小时 0.04 美元，通过 Ollama, Hugging Face Chat UI 和 SaladCloud 构建你自己的 ChatGPT。</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)```">CUDA semantics &mdash; PyTorch 2.2 documentation</a>：未找到描述</li><li><a href="https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb">peft/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb at main · huggingface/peft</a>：🤗 PEFT: 尖端的参数高效微调 (Parameter-Efficient Fine-Tuning)。 - huggingface/peft</li><li><a href="https://github.com/philschmid/huggingface-sagemaker-workshop-series/blob/main/workshop_1_getting_started_with_amazon_sagemaker/lab_1_default_training.ipynb">huggingface-sagemaker-workshop-series/workshop_1_getting_started_with_amazon_sagemaker/lab_1_default_training.ipynb at main · philschmid/huggingface-sagemaker-workshop-series</a>：使用 Hugging Face 和 SageMaker 的企业级 NLP 工作坊系列 - philschmid/huggingface-sagemaker-workshop-series</li><li><a href="https://learnbybuilding.ai/tutorials/rag-from-scratch">A beginner's guide to building a Retrieval Augmented Generation (RAG) application from scratch</a>：从零开始构建检索增强生成 (RAG) 应用的初学者指南：本文将教你 RAG 背后的基本直觉，并提供一个简单的教程帮助你入门。</li><li><a href="https://github.com/Mikubill/sd-webui-controlnet">GitHub - Mikubill/sd-webui-controlnet: WebUI extension for ControlNet</a>：GitHub - Mikubill/sd-webui-controlnet：用于 ControlNet 的 WebUI 扩展：用于 ControlNet 的 WebUI 扩展。贡献...</li>

<li>通过在 GitHub 上创建账号来参与 Mikubill/sd-webui-controlnet 的开发。</li><li><a href="https://github.com/guananya/AllenNLP-Coreference-Resolution-in-Python-Readable-clusters/blob/master/allennlp_coref.py">AllenNLP-Coreference-Resolution-in-Python-Readable-clusters/allennlp_coref.py (master 分支) · guananya/AllenNLP-Coreference-Resolution-in-Python-Readable-clusters</a>：在 Python 中使用 AllenNLP 指代消解（获取真正可读的聚类） - guananya/AllenNLP-Coreference-Resolution-in-Python-Readable-clusters</li><li><a href="https://huggingface.co/ProsusAI/finbert">ProsusAI/finbert · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2304.14241">Entity-Level Sentiment Analysis (ELSA): An exploratory task survey</a>：本文探讨了识别文档中对意志实体（个人和组织）表达的整体情感的任务——即我们所说的 Entity-Level Sentiment Analysis...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1225782561343799448)** (2 条消息): 

- **寻求 Knowledge Graphs 相关知识**：一位成员表达了对学习 **knowledge graphs** 及其应用的兴趣，并请求推荐资源。
- **构建 Collate，寻求学习经验**：[Collate](https://collate.one/preview) 是一个旨在为学生、专业人士和内容创作者改变日常学习的新平台。创作者正在寻求与学习挑战相关的反馈和经验，并提供早期访问权限和 15 分钟的通话讨论机会。[预约通话](https://calendly.com/vel-yan/15min)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://collate.one/preview">Collate Preview</a>：改变你的日常学习</li><li><a href="https://calendly.com/vel-yan/15min">15 Minute Meeting - Vel Yanchina</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1225755414554214431)** (11 条消息🔥): 

- **PIDNet 提升语义分割性能**：一篇[新论文](https://arxiv.org/abs/2206.02066)介绍了 PIDNet，这是一种受 PID 控制器启发的共三分支网络架构，旨在通过有效整合细节、上下文和边界信息来增强实时语义分割。
- **LLM 易受 'Crescendo' 越狱攻击**：Mark Russinovich 分享了指向 'Crescendo' 的链接，这是一种针对大语言模型 (LLM) 的[潜在越狱攻击](https://crescendo-the-multiturn-jailbreak.github.io/)，旨在绕过为防止生成有害内容而设置的伦理边界。
- **Cohere Command-R-plus 陷入越狱陷阱**：一篇 LinkedIn [帖子强调](https://www.linkedin.com/posts/enkryptai_command-r-red-teaming-report-activity-7182087079974117377-ujmT)了 Cohere 的 Command-R-plus 系统在面对越狱攻击时暴露出的漏洞。
- **推进 Mixture-of-Depths 概念**：新的 [Mixture-of-Depths (Modes) 提案](https://arxiv.org/abs/2404.02258)建议在序列中动态分配 Transformer 计算，在不牺牲灵活性的情况下潜在地提高效率。
- **使用 llamaindex 探索多文档解决方案**：一位用户分享了[一篇博客文章](https://ai.gopubby.com/unlocking-the-power-of-multi-document-agents-with-llamaindex-d09e4d7dfe0e?gi=947416d131c6)，关于利用 llamaindex 创建多文档 RAG 解决方案以改进信息检索。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>：基于 Transformer 的语言模型在输入序列上均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 反而可以学习将 FLOPs（或计算量）动态分配给特定的...</li><li><a href="https://arxiv.org/abs/2206.02066">PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers</a>：双分支网络架构在实时语义分割任务中已显示出其效率和有效性。然而，高分辨率细节和低频上下文的直接融合具有...</li><li><a href="https://crescendo-the-multiturn-jailbreak.github.io/">Crescendo </a>：多轮 LLM 越狱攻击</li><li><a href="https://www.youtube.com/watch?v=_j7JEDWuqLE">Hugging Face + Langchain in 5 mins | Access 200k+ FREE AI models for your AI apps</a>：学习如何使用 Hugging Face，并在免费使用 Langchain 构建应用的同时访问 20 万+ AI 模型。🔗 链接 - Hugging Face 教程：https://hf.co/tasks- ...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1225741246526390374)** (44 条消息🔥):

- **GitHub 上的神经符号 AGI**：一家法国 AI 初创公司推出了一种新型开源神经符号 AGI，旨在利用基于图的 Prompt Programming 进行行为编程。该项目目前正在寻求社区反馈，并在 [GitHub](https://github.com/SynaLinks/HybridAGI) 上展示。
- **开源论文复现仓库**：一个旨在通过复现 AI 和 ML 研究论文来提升技能的仓库已发布，邀请贡献者关注（star）、提供建议并提交 PR。在此查看仓库：[GitHub 上的 PaperReplica](https://github.com/hegdeadithyak/PaperReplica)。
- **通过 Gradio 管理音频数据集**：分享了一个用于创建和管理大型音频数据集的新 Gradio 界面，适用于有声读物分段和转录等任务。该工具可在 [GitHub](https://github.com/maepopi/audio-dataset-manager) 上使用。
- **用于 MNIST 手写数字的 RNN**：发布了一个使用 numpy 自行编写的 vanilla RNN，用于对 MNIST 数字进行分类，其代码可供审阅。访问 [GitHub](https://github.com/suprasauce/RNN_MEDIUM) 上的项目。
- **本地文件夹图片搜索**：一个名为 'Where's My Pic?' 的项目为本地文件夹提供了类似 Google Image Search 的体验，帮助快速查找图片。演示视频可在 [YouTube](https://www.youtube.com/watch?v=oVJsJ0e6jWk) 上找到。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/dhanikkcs/status/1776179274640400502">来自 Kheem Chandra (@dhanikkcs) 的推文</a>：我创建了一个 Telegram 机器人。它由 Gemini 1.5 Pro 驱动。你可以通过视频、图像或文本格式与其聊天。机器人名称："int_gem_bot" 链接：https://telegram.me/int_gem_bot 试试看...</li><li><a href="https://thebeastbot.com/welcome/">MrBeast 的创意天才 AI 机器人 :)</a>：我是所有 AI 机器人中的 Beast！我加载了大量 MrBeast 最疯狂、最具创新性的内容。这就像获得了进入他那令人惊叹的大脑的独家后台权限...</li><li><a href="https://huggingface.co/spaces/not-lain/RMBG1.4-with-imageslider">RMBG1.4 with imageslider - 由 not-lain 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/not-lain/RAG-Chatbot">RAG - 由 not-lain 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/TencentARC/BrushNet">BrushNet - 由 TencentARC 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://arxiv.org/html/2403.17887v1">The Unreasonable Ineffectiveness of the Deeper Layers</a>：未找到描述</li><li><a href="https://github.com/ehristoforu/TensorLM-webui">GitHub - ehristoforu/TensorLM-webui: 基于 LLaMA 的 LLM 模型简单现代的 WebUI。</a>：基于 LLaMA 的 LLM 模型简单现代的 WebUI。- ehristoforu/TensorLM-webui</li><li><a href="https://github.com/RooTender/augmentator">GitHub - RooTender/augmentator: 开箱即用的图像增强工具</a>：开箱即用的图像增强工具。通过在 GitHub 上创建账号来为 RooTender/augmentator 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=oVJsJ0e6jWk">Where's My Pic 演示</a>：大家好，我是 Om Alve，在这个视频中，我将演示我的项目 "Where's my pic?"。这个项目解决了在...中搜索的问题。</li><li><a href="https://github.com/abhaskumarsinha/Corpus2GPT">GitHub - abhaskumarsinha/Corpus2GPT: CustomGPTBuilder: 一个允许用户在多样化数据集（包括本地语言和各种语料库类型）上训练自己的 GPT 模型的项目，使用 Keras 并兼容 TensorFlow, PyTorch 或 JAX 后端，以便后续存储或共享。</a>：CustomGPTBuilder：一个允许用户在多样化数据集（包括本地语言和各种语料库类型）上训练自己的 GPT 模型的项目，使用 Keras 并兼容 TensorFlow, PyTorch...</li><li><a href="https://rapidapi.com/NextAPI/api/cheapest-gpt-4-turbo-gpt-4-vision-chatgpt-openai-ai-api/">最便宜的 GPT-4 Turbo, GPT 4 Vision, ChatGPT OpenAI AI API 接口文档 (NextAPI) | RapidAPI</a>：未找到描述</li><li><a href="https://github.com/hegdeadithyak/PaperReplica">GitHub - hegdeadithyak/PaperReplica: 我们复现 AI 和 ML 领域的研究论文。</a>：我们复现 AI 和 ML 领域的研究论文。- hegdeadithyak/PaperReplica</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: 可编程的神经符号 AGI，允许你使用基于图的 Prompt Programming 来编程其行为：适用于希望 AI 表现符合预期的人群</a>：可编程的神经符号 AGI，允许你使用基于图的 Prompt Programming 来编程其行为：适用于希望 AI 表现符合预期的人群 - SynaLinks/HybridAGI</li><li><a href="https://github.com/suprasauce/RNN_MEDIUM">GitHub - suprasauce/RNN_MEDIUM</a>：通过在 GitHub 上创建账号来为 suprasauce/RNN_MEDIUM 的开发做出贡献。</li><li><a href="https://git.ecker.tech/mrq/ai-voice-cloning/">ai-voice-cloning</a>：旨在通过 AI 进行语音克隆的实用工具集</li><li><a href="https://github.com/maepopi/audio-dataset-manager">GitHub - maepopi/audio-dataset-manager: 一款专为 TTS 和语音克隆准备有声读物或大型音频而设计的全能工具。</a>：一款专为 TTS 和语音克隆准备有声读物或大型音频而设计的全能工具。- maepopi/audio-dataset-manager
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1225818367861194863)** (10 条消息🔥):

- **寻找合适的频道**：一位成员询问另一个频道 <#879548962464493622> 是否更适合特定问题。
- **询问论文阅读活动**：成员们对论文阅读活动表现出兴趣，确认此类活动通常在每个周末举行。上周的活动有一位出色的演讲者，并进行了录制。
- **寻找学习资源**：有人询问如何找到资源来理解模型的基础构建块，以便在不针对特定模型的情况下进行微调和构建新模型。
- **知识库已就绪**：论文阅读会议的录音和通知已汇编在 [GitHub 仓库](https://github.com/isamu-isozaki/huggingface-reading-group) 中，最新的录音尚未添加。Discord 活动是目前获取会议通知的首选方式。
- **模型探索的通用指南**：当被问及如何理解模型代码库的指导时，一位成员询问了在不关注特定模型的情况下，导航和理解模型编码方面所需的特定领域知识。

**提到的链接**：<a href="https://github.com/isamu-isozaki/huggingface-reading-group">GitHub - isamu-isozaki/huggingface-reading-group: This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group</a>: 该仓库的目标是预先汇编 Huggingface 阅读小组过去所有的演示内容 - isamu-isozaki/huggingface-reading-group

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1225803402806558810)** (12 条消息🔥): 

- **HuggingFace 作为机器学习模型的 Git**：用户讨论了 HuggingFace 模型仓库与 Git 的相似之处，你可以像处理代码一样创建仓库并提交（commit）和推送（push）更新。
  
- **训练期间监控 GPU 使用情况**：针对用户关于如何在训练模型时监控 GPU 使用情况的询问，推荐了一个名为 [Model Memory Usage](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) 的 HuggingFace Space。

- **不使用 Pandas 操作 Parquet 文件**：一位用户寻求在不使用 **Pandas** 的情况下从 Parquet 文件中删除列的替代方案。建议使用 [HuggingFace 文档](https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/main_classes#datasets.Dataset.from_parquet) 中 `datasets` 库的 `from_parquet` 方法。

- **寻求用于视频质量的 Diffusion 模型资源**：一位用户请求有关使用 Diffusion 模型提高视频质量的协助和资源，并寻求相关的学术论文。

- **使用更多帧训练 XCLIP**：一位成员分享了他们尝试使用比预训练版本更多帧来预训练 **XCLIP** 模型的经验。他们面临 loss 停滞和 NaNs 的问题，并寻求关于如何按照 [XCLIP 文档](https://huggingface.co/docs/transformers/en/model_doc/xclip#transformers.XCLIPModel) 中所述从头开始训练具有扩展帧容量模型的建议。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/hf-accelerate/model-memory-usage">Model Memory Utility - a Hugging Face Space by hf-accelerate</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/model_doc/xclip#transformers.XCLIPModel">X-CLIP</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1225957005756334122)** (24 条消息🔥): 

- **为特定数据提取微调 Mistral7b**：一位成员询问是否可以通过使用其输出的清洗结果来微调 **Mistral7b** 以进行 JSON 数据提取。他们正在思考对于类似的输入输出格式，是需要一个 LLM 还是一个更专业的模型。

- **不使用 Twitter API 解析推文**：一位成员寻求在不使用 Twitter 复杂的 API 的情况下抓取推文的替代方案，暗示希望有一种不那么复杂的工具或方法来完成这项任务。

- **Colab Pro+ 在运行 WizardLM 模型时遇到困难**：一位参与者在 Google Colab Pro+ 上尝试加载 cognitivecomputations 的 **WizardLM-13B** 和 **WizardLM-7B** 模型时遇到了显存不足（out-of-memory）错误，尽管尝试了不同的 GPU 并寻找解决方案。

- **Gemini 1.5 中 10M 上下文窗口的可行性**：**Gemini 1.5** 论文中关于 10M 上下文窗口的说法引发了讨论，一位成员寻求关于它如何计算庞大的 Attention 矩阵的解释。另一位成员分享了一篇[可能相关的论文](https://arxiv.org/abs/2310.01889)，该论文可能阐述了实现这一目标所使用的方法。

- **使用 LLM 填充空值**：一位成员表示需要根据上下文，使用 LLM 填充包含 'object' 数据类型字段的数据集中的空值，并寻求相关的参考资料或操作建议。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2310.01889">Ring Attention with Blockwise Transformers for Near-Infinite Context</a>：Transformers 已成为许多最先进 AI 模型的首选架构，在广泛的 AI 应用中展现出卓越的性能。然而，内存需求...</li><li><a href="https://bhosmer.github.io/mm/ref.html">mm ref</a>：未找到描述</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)">CUDA semantics &mdash; PyTorch 2.2 documentation</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1225780367022096546)** (9 messages🔥): 

- **PEFT 缩小了 llava2 但面临部署问题**：一位成员正在使用 **PEFT 技术** 来减小 llava2 模型的大小，但在尝试在另一台机器上运行缩小后的模型时遇到问题。该问题似乎与模型采用 **safetensors 格式** 有关，导致出现缺少 `pytorch_model_bin` 文件的错误。

- **部署 Safetensors 格式模型**：针对上述问题，有人建议检查 `use_safetensors=True` 的使用情况，这可能会解决以 safetensors 格式安全部署缩小模型的问题。

- **NLP 初学者的学习曲线**：一位寻求关于是否学习 **transformers, LSTM, GRU 或双向 LSTM/GRU** 建议的新成员被引导至 [Stanford CS224N YouTube 课程](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4)，该资源全面涵盖了使用深度学习的 Natural Language Processing。

- **请求 Euler/Euler-A Sampler 见解**：一位成员表示难以找到关于 **euler/euler-a sampler** 的博客类资源，并正在寻求建议，目前仅找到了 k-diffusion 仓库作为参考。

- **LaBSE 模型在 OpenSearch 中的导出挑战**：一位用户在尝试将 "sentence-transformers/LaBSE" 作为自定义模型用于 **OpenSearch** 时遇到错误，并在尝试使用 Python 脚本将模型导出为 TorchScript 后面临困难。

**提及的链接**：<a href="https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4">Stanford CS224N: Natural Language Processing with Deep Learning | 2023</a>：Natural language processing (NLP) 是人工智能 (AI) 的关键部分，旨在模拟人类分享信息的方式。近年来，深度学习应用...

  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1226986017958138049)** (1 messages): 

- **API Recorder 登场**：Gradio 最新更新 4.26.0 推出了 🎥**API Recorder**，它可以记录与任何 Gradio 应用的交互，并自动生成相应的 Python 或 JavaScript 代码。可以通过 `View API` 页面访问此功能，以简化以编程方式重建应用操作的过程。
- **修复 Bug 以提升速度**：此更新还解决了一个关键 **bug**，该 bug 此前导致 Gradio 4.25.0 版本的页面加载速度缓慢。
- **Chatbot UI 崩溃问题已解决**：修复了一个重大问题，即快速的聊天机器人更新可能导致 UI 崩溃，确保了更流畅的用户体验。
- **查看完整变更日志**：有关最新版本中 Bug 修复和功能的完整列表，用户可以在 [Gradio's Changelog](https://www.gradio.app/changelog#4-26-0) 查看完整变更日志。
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1225746638346129465)** (34 messages🔥):

- **探索 Rustlings 和 Ziglings**：一位成员分享了他们使用 [Rustlings](https://github.com/rust-lang/rustlings) 和 [Ziglings](https://codeberg.org/ziglings/exercises/) 进行编程练习的经验，并发现了一个名为 [Mojolings](https://github.com/dbusteed/mojolings) 的 Mojo 等效项目。
- **Mojo 中的 Var 与 Let**：澄清了在 Mojo 中 `var` 用于延迟赋值变量（lazily assigned variables）且不会消失，尽管 `let` 已被移除。提供了一个学习 Mojo 如何使用 `var` 的资源，详见[此处](https://docs.modular.com/mojo/manual/variables#declared-variables)。
- **用于 Web 应用程序的 Mojo**：关于 Mojo 在 Web 开发中潜力的讨论产生了一个名为 [lightbug_http](https://github.com/saviorand/lightbug_http) 的简单且快速的 Mojo HTTP 框架信息。
- **作为通用编程语言的 Mojo**：成员们互相确认 Mojo 确实是一种专为 AI/ML 设计的通用编程语言（general purpose language），并强调了 Mojo 虽然年轻但正在不断发展的特性。
- **寻求 Mojo 的文档和学习资源**：成员们询问了学习 Mojo 的书籍或全面文档，目前最推荐的是 [Mojo Manual](https://docs.modular.com/mojo/manual)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual">Mojo Manual | Modular Docs</a>：Mojo 编程语言的全面指南。</li><li><a href="https://docs.modular.com/mojo/manual/variables#declared-variables).">Variables | Modular Docs</a>：Mojo 变量介绍。</li><li><a href="https://github.com/modularml/max">GitHub - modularml/max: A collection of sample programs, notebooks, and tools which highlight the power of the MAX platform</a>：展示 MAX 平台强大功能的示例程序、notebooks 和工具集 - modularml/max</li><li><a href="https://github.com/saviorand/lightbug_http">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>：适用于 Mojo 的简单且快速的 HTTP 框架！🔥。通过在 GitHub 上创建账号为 saviorand/lightbug_http 做出贡献。</li><li><a href="https://github.com/rust-lang/rustlings">GitHub - rust-lang/rustlings: :crab: Small exercises to get you used to reading and writing Rust code!</a>：🦀 让你习惯阅读和编写 Rust 代码的小练习！- rust-lang/rustlings</li><li><a href="https://codeberg.org/ziglings/exercises/.">exercises</a>：通过修复损坏的小程序来学习 Zig 编程语言。</li><li><a href="https://github.com/dbusteed/mojolings">GitHub - dbusteed/mojolings: Learn to read and write Mojo code by fixing small programs</a>：通过修复小程序学习阅读和编写 Mojo 代码 - dbusteed/mojolings
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1225851676305526835)** (7 messages): 

- **Modular 推文连发开始**：Modular 分享了一系列以 Modular 技术创新为主题的推文。点击[此处](https://twitter.com/Modular/status/1776287802533245372)查看推文。
- **推进 Modular 运动**：另一条 Modular 推文暗示了进一步的进展，表明其技术开发正在持续推进。推文见[此处](https://twitter.com/Modular/status/1776287865242300621)。
- **Modular 未来计划预览**：Modular 的一条推文似乎预告了其生态系统中即将开展的项目或开发。点击[此处](https://twitter.com/Modular/status/1776287868710998188)查看推文。
- **迎接新挑战**：Modular 发布了一条可能讨论克服挑战或设定新目标的推文。完整内容见[此处](https://twitter.com/Modular/status/1776356366309113974)。
- **延续 Modular 的故事**：Modular 进展的故事在另一条推文中延续，这可能是基于之前的公告或成就。点击[此处](https://twitter.com/Modular/status/1776356370004242655)查看推文。
- **规划 Modular 的前进道路**：Modular 的 Twitter 帖子建议了公司或其技术的前进道路大纲。帖子见[此处](https://twitter.com/Modular/status/1776356373682696701)。
- **Modular 的愿景展开**：来自 Modular 的一条推文展示了他们的愿景，可能揭示了公司的新见解或方向。阅读推文请点击[此处](https://twitter.com/Modular/status/1777447869907431562)。
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1226269140281720889)** (2 messages):

- **Modular 与 AWS 联手**：Modular 宣布与 [Amazon Web Services (AWS) 达成合作伙伴关系](https://www.modular.com/blog/modular-partners-with-amazon-web-services-aws-to-bring-max-to-aws-services)，旨在将 **MAX Platform** 与 AWS 服务集成，从而在全球范围内提供创新的 AI 功能。AWS 机器学习与 AI 服务副总裁 Bratin Saha 强调，该合作伙伴关系在加速 AWS 客户采用 GenAI 和传统 AI 用例方面发挥着重要作用。

- **Mojo 标准库开放协作**：Modular 呼吁社区为 [Mojo 标准库](https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide)做出贡献，并提供了一份关于如何贡献的全面指南，涵盖了从在 GitHub 上识别问题到创建成功的 pull requests 的全过程。该指南是在 Modular 最近开源 Mojo 标准库这一里程碑之后发布的，邀请社区进行从文档到代码更改的各种改进。

<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/modular-partners-with-amazon-web-services-aws-to-bring-max-to-aws-services">Modular: Modular partners with Amazon Web Services (AWS) to bring MAX to AWS services</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Modular 与 Amazon Web Services (AWS) 合作，将 MAX 引入 AWS 服务</li><li><a href="https://www.modular.com/blog/how-to-contribute-to-mojo-standard-library-a-step-by-step-guide">Modular: How to Contribute to Mojo Standard Library: A Step-by-Step Guide</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：如何为 Mojo 标准库做贡献：分步指南
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/)** (1 条消息): 

rxzfn: 有一个类似的可移动产品，但使用的是 PCIe。
  

---


**Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1226067117544050750)** (2 条消息): 

- **仓库访问问题已解决**：简短的交流表明访问某个仓库时存在问题，随后通过更新的*有效链接*迅速纠正。未提供更多背景或实际链接。
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1225742384814624829)** (336 条消息🔥🔥): 

- **探索 Mojo 的参数能力**：通过深入研究 Mojo 的参数用法，发现计算可以完全在参数阶段（parameter time）执行，从而实现创新的元编程。然而，这暴露了一个作用域问题，即在函数签名中执行的操作 (`a + b`) 与将操作存储在命名的推断参数 (`_L = a + b`) 中时产生的结果不同。

- **编译时求值的复杂性**：围绕在编译时执行某些类型操作所面临的困难展开了长篇讨论。这突显了 Mojo 编译器固有的复杂性，以及类型系统处理加法等操作时的复杂性，因为像 `a + b == b + a` 这样简单的等式也需要证明。

- **Mojo 中的 Reference 和生命周期复杂性**：讨论了在使用 `@value` 装饰器和 `init` 方法时，关于 `Reference` 类型及其生命周期的潜在问题和方法论。有人指出，`Reference` 和生命周期机制可能需要更清晰的说明和文档以便于使用。

- **对未来开源贡献的期待**：用户表达了对 Mojo 开源的期待，希望社区能为 BSDs 等其他系统提供移植支持。预计开源将使 Mojo 获得更广泛的适配和集成。

- **RustPython 作为语言实现的案例研究**：RustPython 被作为重新实现语言标准库的案例进行研究，考虑到其执行速度比 CPython 慢。讨论承认，虽然这类项目既酷又有野心，但往往缺乏在长期建立的同类项目中看到的广泛优化。
<div class="linksMentioned">

<strong>提及链接</strong>：

</div>

<ul>
<li>
<a href="https://tenor.com/view/angry-anger-pixar-inside-out-aaah-gif-5628546">Angry Anger GIF - Angry Anger Pixar - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/you-make-a-compelling-argument-simon-hardwick-blood-and-treasure-you-make-a-persuasive-argument-you-make-a-strong-argument-gif-26852864">You Make A Compelling Argument Simon Hardwick GIF - You Make A Compelling Argument Simon Hardwick Blood And Treasure - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://hirrolot.github.io/posts/rust-is-hard-or-the-misery-of-mainstream-programming.html">Rust Is Hard, Or: The Misery of Mainstream Programming</a>: 未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/1702#issuecomment-1940230390">[BUG]: 类型检查器的行为不一致（某些表达式在类型检查之前被处理） · Issue #1702 · modularml/mojo</a>: Bug 描述 我最近一直在尝试理解类型检查器如何推导类型的等价性。我注意到了一些不一致之处和潜在的 Bug。这些在...中得到了演示。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/collections/list.mojo#L41-L70">mojo/stdlib/src/collections/list.mojo at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2100">[Feature Request] 允许参数化实例化 (parametric materialization) · Issue #2100 · modularml/mojo</a>: 审查 Mojo 的优先级 我已经阅读了路线图和优先级，并相信此请求符合优先级。你的请求是什么？允许不可实例化类型的参数化实例化...</li><li><a href="https://github.com/modularml/max">GitHub - modularml/max: 突出 MAX 平台强大功能的示例程序、笔记本和工具集合</a>: 突出 MAX 平台强大功能的示例程序、笔记本和工具集合 - modularml/max</li><li><a href="https://www.modular.com/blog/mojo-python-calculating-and-plotting-a-valentines-day-using-mojo-and-python">Modular: Mojo🔥 ♥️ Python: 使用 Mojo 和 Python 计算并绘制情人节 ♥️</a>: 我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo🔥 ♥️ Python: 使用 Mojo 和 Python 计算并绘制情人节 ♥️</li><li><a href="https://github.com/modularml/devrel-extras/tree/main/blogs/">devrel-extras/blogs at main · modularml/devrel-extras</a>: 包含开发者关系博客文章、视频和研讨会的辅助材料 - modularml/devrel-extras</li><li><a href="https://youtu.be/pdJQ8iVTwj8?si=ML7lZfXAel9zEgj0&t=5763">Chris Lattner: 编程与 AI 的未来 | Lex Fridman Podcast #381</a>: Chris Lattner 是一位传奇的软件和硬件工程师，曾在 Apple、Tesla、Google、SiFive 和 Modular AI 领导项目，包括开发 S...</li><li><a href="https://github.com/mo">mo - 概览</a>: mo 有 49 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/docs/style-guide.md">mojo/stdlib/docs/style-guide.md at nightly · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/1837">[BUG]: 自引用 Variant 导致编译器崩溃 · Issue #1837 · modularml/mojo</a>: Bug 描述 from utils.variant import Variant from collections.vector import DynamicVector @value struct Value(CollectionElement): alias Variant = Variant[Float64, DynamicVector[Value]] var _va...</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/collections/dict.mojo#L48-L94">mojo/stdlib/src/collections/dict.mojo at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/RustPython/RustPython">GitHub - RustPython/RustPython: 用 Rust 编写的 Python 解释器</a>: 用 Rust 编写的 Python 解释器。通过在 GitHub 上创建账号来为 RustPython/RustPython 的开发做出贡献。</li><li><a href="https://github.com/python/cpython/blob/main/Objects/dictobject.c">cpython/Objects/dictobject.c at main · python/cpython</a>: Python 编程语言。通过在 GitHub 上创建账号来为 python/cpython 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/collections/dict.mojo">mojo/stdlib/src/collections/dict.mojo at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。
</li>
</ul>

**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1225755101755346996)** (18 messages🔥): 

- **Mojo 中现已支持特殊函数**：Specials 包的更新引入了几个**初等数学函数**，如 `exp`、`exp2`、`expm1`、`log` 和 `log1p`。这些实现优先考虑数值精度而非 FLOPS，基准测试可以在 [项目仓库](https://github.com/leandrolcampos/specials) 中找到。

- **SICP 迎来 Mojo 版**：经典教科书《计算机程序的构造和解释》（Structure and Interpretation of Computer Programs）正被移植到 Mojo 语言中，项目名为 [sicp_mojo](https://github.com/Brian-M-J/sicp_mojo)，目前参考的是 JavaScript 版本。

- **Mojo 算法集体倡议**：一名成员计划用 Mojo 重写常用算法，如 Dijkstra 算法和不同的排序方法，并对协同开发感兴趣。

- **一站式 Mojo 包仓库**：社区成员可以通过 PR 在 [mojo-packages 仓库](https://github.com/kernhanda/mojo-packages) 中分享他们的 Mojo 包，该仓库旨在官方包管理器发布前充当中心枢纽。

- **Mambamojo 协作项目**：一个名为 [mamba.mojo](https://github.com/aizzf/mamba.mojo) 的 GitHub 仓库正在寻求合作者，共同致力于用纯 Mojo 实现 Mamba，涵盖从模型到推理和训练的全过程。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/kernhanda/mojo-packages">GitHub - kernhanda/mojo-packages: A place to find and share packages for the Mojo language</a>: 查找和分享 Mojo 语言包的地方 - kernhanda/mojo-packages</li><li><a href="https://github.com/aizzf/mamba.mojo">GitHub - aizzf/mamba.mojo: Mamba in pure mojo from model to inference and train.</a>: 从模型到推理和训练的纯 Mojo 版 Mamba - aizzf/mamba.mojo</li><li><a href="https://github.com/kernhanda/mojopack">GitHub - kernhanda/mojopack: mojopack is a tool for managing packages for the Mojo programming language</a>: mojopack 是一个用于管理 Mojo 编程语言包的工具 - kernhanda/mojopack</li><li><a href="https://github.com/leandrolcampos/specials">GitHub - leandrolcampos/specials: Special functions with hardware acceleration</a>: 具有硬件加速功能的特殊函数。通过在 GitHub 上创建账号为 leandrolcampos/specials 的开发做出贡献。</li><li><a href="https://github.com/Hammad-hab/pkm">GitHub - Hammad-hab/pkm: Mojo&#39;s unoffical package manager</a>: Mojo 的非官方包管理器。通过在 GitHub 上创建账号为 Hammad-hab/pkm 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1226416943242936350)** (7 messages): 

- **"Joy of Mojo" 博客上线**：一个名为 [Joy of Mojo](https://joyofmojo.com/) 的新社区网站已推出，个人可以在此分享在探索 Mojo 语言时创建的演示程序。尽管最初 GitHub Pages 出现了一些问题，但该网站现在似乎已恢复运行，欢迎社区参与贡献和讨论。

- **"Joy of Mojo" 的链接故障**：[Joy of Mojo](https://joyofmojo.com/) 网站在 GitHub Pages 上遇到了托管问题，部分用户显示错误，但现在似乎已解决，确保了用户的访问权限。

- **Mojo 包分享倡议**：社区成员在 GitHub 上创建了如 [mojo-packages](https://github.com/kernhanda/mojo-packages) 和 [mojopack](https://github.com/kernhanda/mojopack) 等仓库用于分享 Mojo 语言包，补充了社区内的协作精神。

- **承认 Mojo 的动态演进**：人们认识到 **Mojo 的快速演进**，并预期某些分享的内容可能会在几个月内过时，这突显了语言生态系统内持续的开发和变化。

- **具有教育意义的 Mojo 演示**：一位社区成员分享了一个 [YouTube 视频](https://youtu.be/6cyCeJwgNjc)，旨在通过演示 Python 中的星号图案来教育并给观众带来惊喜，最后揭示这其实是使用 VSCode 中的 Mojo 插件编写的 Mojo 代码，引起了人们对 Mojo 的 Python 兼容性的关注。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://joyofmojo.com/">Joy of Mojo 🔥</a>: 这是 Joy of Mojo</li><li><a href="https://github.com/kernhanda/mojo-packages">GitHub - kernhanda/mojo-packages: 一个寻找和分享 Mojo 语言包的地方</a>: 一个寻找和分享 Mojo 语言包的地方 - kernhanda/mojo-packages</li><li><a href="https://github.com/kernhanda/mojopack">GitHub - kernhanda/mojopack: mojopack 是一个用于管理 Mojo 编程语言包的工具</a>: mojopack 是一个用于管理 Mojo 编程语言包的工具 - kernhanda/mojopack
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1225758883436429312)** (71 条消息🔥🔥): 

- **Python 互操作性增强**：一位成员一直致力于 **CPython** 的互操作性，在 Mojo 中实现了 *PyMethodDef*、*PyCFunction_New* 和 *PyModule_NewObject*。他们强调了在没有 bug 的情况下实现引用计数的进展，并认为他们的工作为进一步规划 Python 互操作性奠定了良好的基础。相关的开发工作可以在 [GitHub](https://github.com/rd4com/mojo_branch/tree/nightly) 上找到。

- **新贡献者入门指南**：引导新贡献者从查看 "good first issues" 开始，并提供了指向 Mojo GitHub 仓库中的 [changelog](https://docs.modular.com/mojo/changelog#week-of-2023-01-30) 和 [贡献指南](https://github.com/modularml/mojo/blob/main/CONTRIBUTING.md) 的链接。

- **关于 Signed-off Commit 最佳实践的讨论**：在一次关于 pull request 实践的热烈交流中，一位成员了解了正确签署 commit（signing-off commits）的重要性，并获得了关于如何修改 commit 作者身份以及使用 `git config` 正确归属其工作的指导。链接了相关的 GitHub 文档以配置 git 中的用户名，并推荐将 VSCode 作为具有自动签名选项的工具。

- **征求标准库测试实践的反馈**：一位成员在 GitHub 上发起了一场 [讨论](https://github.com/modularml/mojo/discussions/2234)，旨在收集关于改进 Mojo 标准库中 List 和 String 切片测试的意见，并建议在测试中为 `assert_equal` 提供更具描述性的标签。

- **管理 Nightly 构建和包**：Nightly 构建更新通知包含了如何使用 `modular update` 进行更新的信息，链接了 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 以及版本之间的差异。一些成员还分享了在更新时如何处理 "Error opening archive" 的问题和解决方案，通常的补救措施是使用 `modular clean` 并重新安装。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.github.com/en/get-started/getting-started-with-git/setting-your-username-in-git">在 Git 中设置用户名 - GitHub Docs</a>: 未找到描述</li><li><a href="https://github.com/modularml/mojo/discussions/2234)">修复当步长为负数时 List 切片和 String 切片以匹配 Python · modularml/mojo · Discussion #2234</a>: 我想修复 issue #1944、#2046 和 #2142。这些是关于 List 的，但 String 也存在同样的问题。我实际上已经在这里完成了一个修复，并在该 commit 中进行了有趣的尝试。该实现只是...</li><li><a href="https://github.com/modularml/mojo/compare/1a8f912..1bce16d">比较 1a8f912..1bce16d · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/738901dec1058612d8f01fd13e13a3e09103944f/stdlib/test/lit.cfg.py#L57">mojo/stdlib/test/lit.cfg.py 位于 738901dec1058612d8f01fd13e13a3e09103944f · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://docs.python.org/3/library/pathlib.html#pathlib.Path.cwd">pathlib — 面向对象的文件系统路径</a>: 源代码：Lib/pathlib.py。该模块提供的类代表了具有适用于不同操作系统的语义的文件系统路径。路径类分为纯路径（pure paths）...</li><li><a href="https://github.com/modularml/mojo/compare/1a8f912..1bce1">比较 1a8f912..1bce1 · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/microsoft/vscode/issues/83096">通过设置自动执行 --signoff · Issue #83096 · microsoft/vscode</a>: 目前看来，你可以通过打开 git commit 下拉菜单并选择 signoff 来在 git commit 中使用 -s 或 --signoff 命令。这是作为 #7010 的“修复”实现的...</li><li><a href="https://github.com/rd4com/mojo_branch/tree/nightly">GitHub - rd4com/mojo_branch 的 nightly 分支</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 rd4com/mojo_branch 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/pull/2215/checks?check_run_id=23522364066">[stdlib] 由 helehex 添加 `reversed` · Pull Request #2215 · modularml/mojo</a>: 添加了 reversed() 的初步实现，用于获取 range 或 list 的反向迭代器。这并非最终版本，因为迭代器尚未完全开发完成，但目前可以使用...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md 的 nightly 分支 · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1225850270425157752)** (80 条消息🔥🔥): 

- **WikiText 数据集访问方式已明确**：WikiText 数据集的原作者 Stephen Merity 已将数据[重新托管](https://state.smerity.com/smerity/state/01HRTB51QZMG59MDAX2ME1TCHR)在 Cloudflare R2 上，这被认为是新的主要访问点。重新托管的数据仍采用 Creative Commons 许可，包含比 Penn Treebank 更大的数据集，并保留了原始的大小写、标点和数字。

- **GateLoop 困惑度（Perplexity）之谜**：有一场关于 GateLoop Transformer 作者报告的困惑度分数的讨论。虽然作者声称分数良好，但 lucidrains 无法复现这些结果，引发了对结果的一些质疑。

- **Hugging Face 数据集自动转换困境**：频道成员对 Hugging Face 将数据集自动转换为 parquet 格式表示不满，这可以通过使用 Git LFS 进行托管来规避。一个典型的例子是 `.raw` 文件的格式混淆问题。

- **寻找可复现的数据格式**：对话围绕对可复现且一致的数据格式的需求展开，为了实验的可复现性，人们也努力在 Cloudflare R2 上镜像原始的 WikiText 数据。

- **OpenAI 模型文档变得转瞬即逝**：成员们分享了寻找 OpenAI 模型信息的经历；多个指向模型文档的链接已被删除，导致必须依赖[存档页面](https://archive.ph/n5xMq)来了解 GPT 3.5 及其他系列的具体细节，这突显了追踪模型演变和变化的挑战。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://state.smerity.com/smerity/state/01HRTB51QZMG59MDAX2ME1TCHR">Smerity.com: The WikiText Long Term Dependency Language Modeling Dataset (2016)</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/segyges/wikitext-103/tree/main">segyges/wikitext-103 at main</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/wikitext/tree/main/wikitext-103-raw-v1">wikitext at main</a>: 未找到描述</li><li><a href="https://github.com/lucidrains/gateloop-transformer">GitHub - lucidrains/gateloop-transformer: Implementation of GateLoop Transformer in Pytorch and Jax</a>: 在 Pytorch 和 Jax 中实现 GateLoop Transformer - lucidrains/gateloop-transformer</li><li><a href="https://github.com/tobiaskatsch/GatedLinearRNN">GitHub - tobiaskatsch/GatedLinearRNN</a>: 通过在 GitHub 上创建账户来为 tobiaskatsch/GatedLinearRNN 的开发做出贡献。</li><li><a href="https://archive.ph/n5xMq">Model index for researchers - OpenAI API</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1225769151029575701)** (313 条消息🔥🔥): 

- **理解 Schedule-Free 优化器**：Schedule-Free 优化器保持权重的简单运行平均值，而不是指数移动平均值。其中一个组件的 1/t 学习率只是计算所有值平均值的另一种方式，如公式 (1 * (1/2) * (2/3) * (3/4) * ... * (1-1/t)) 等于 1/t 所示。

- **关于 Schedule-Free 功效的辩论**：关于 Schedule-Free 优化器性能的结果褒贬不一，在低步数运行中显示出优势，但在较大步数的情况下没有显著帮助。该优化器估计最佳学习率，这可能会随更新步数的变化而变化。

- **优化器中的混合方法**：讨论了随时间增加 Batch Size 是否可以作为学习率调度（learning rate schedules）的替代或补充，建议 Batch Size 翻倍类似于学习率减半。

- **语言模型搜索策略的新方法**：一项研究提出了一种通过使用搜索流（Stream of Search, SoS）语言来教语言模型进行搜索的方法，结果显示，与训练预测单个下一步的模型相比，搜索准确率提高了 25%。

- **与长尾数据相关的涌现能力**：正在探索模型处理 Zero-shot 任务的能力，有建议认为语言模型的涌现能力可能是训练期间接触长尾数据的函数。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/kyo_takano/status/1777273932120526969">来自 Kyo (@kyo_takano) 的推文</a>：一些 ScheduleFree 优于 Adam/SGD 的数据点：- LM/GPT (@eric_alcaide) https://twitter.com/eric_alcaide/status/1776571679524683950 - CIFAR10/ResNet18 (@Sree_Harsha_N) https://twitter.com/S...</li><li><a href="https://x.com/aaron_defazio/status/1776320004465582331?s=46">来自 Aaron Defazio (@aaron_defazio) 的推文</a>：Schedule-Free 学习 https://github.com/facebookresearch/schedule_free 我们现在已经开源了我那一系列神秘图表背后的算法。每张图表要么是 Schedule-free SGD，要么是 Adam，没有...</li><li><a href="https://x.com/arankomatsuzaki/status/1777143382554313004?s=46&t=OICM4zGqs0OOATmLPoNFyw">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：没有指数级数据就没有 “Zero-Shot”：预训练概念频率决定多模态模型性能。代码库：https://github.com/bethgelab/frequency_determines_performance hf：https://huggingf...</li><li><a href="https://arxiv.org/abs/2403.15796">从 Loss 视角理解语言模型的涌现能力</a>：最近的研究对语言模型的涌现能力是大模型特有的这一观点提出了质疑。这种怀疑源于两个观察：1) 较小的模型也可以表现出...</li><li><a href="https://arxiv.org/abs/2110.00641">策略优化的 Batch size 不变性</a>：如果 Batch size 的变化可以在很大程度上通过其他超参数的变化来补偿，我们就说该算法具有 Batch size 不变性。众所周知，随机梯度下降（SGD）具有这种属性...</li><li><a href="https://arxiv.org/abs/2404.03683">Stream of Search (SoS)：在语言中学习搜索</a>：语言模型在训练过程中很少看到有益的错误。因此，它们很难看透下一个 token 之外的内容，遭受误差滚雪球的影响，并难以预测后果...</li><li><a href="https://arxiv.org/abs/2402.05120">更多 Agent 就是你所需要的一切</a>：我们发现，仅通过采样与投票方法，大语言模型（LLMs）的性能就会随着实例化的 Agent 数量而扩展。此外，该方法与现有的复杂方法是正交的...</li><li><a href="https://arxiv.org/abs/1708.07120">超收敛：使用大学习率极速训练神经网络</a>：在本文中，我们描述了一种被称为 “超收敛” 的现象，即神经网络的训练速度可以比标准训练方法快一个数量级。存在...</li><li><a href="https://arxiv.org/abs/2211.08411">大语言模型难以学习长尾知识</a>：互联网包含丰富的知识——从历史人物的生日到如何编码的教程——所有这些都可能被语言模型学习。然而，虽然某些片段...</li><li><a href="https://openreview.net/forum?id=FpKgG31Z_i9">学习率嫁接：优化器调优的可迁移性</a>：在训练大型神经网络的经验科学中，学习率调度是一个众所周知难以调优的超参数，它可能取决于所有其他属性（架构...）</li><li><a href="https://arxiv.org/abs/2404.03648">AutoWebGLM：引导并强化基于大语言模型的网页导航 Agent</a>：大语言模型（LLMs）推动了许多智能 Agent 任务，例如网页导航——但由于三个因素，大多数现有 Agent 在真实网页中的表现远不尽如人意：(1) ...</li><li><a href="https://www.torchstudio.ai/getstarted/">入门指南</a>：<a href="#install-torchstudio">安装</a> TorchStudio，<a href="#load-and-analyze-the-mnist-dataset">加载</a>数据集，<a href="#build-and-train-...">构建并训练...</a></li><li><a href="https://github.com/facebookresearch/schedule_free">GitHub - facebookresearch/schedule_free: PyTorch 中的 Schedule-Free 优化</a>：PyTorch 中的 Schedule-Free 优化。通过在 GitHub 上创建账号来为 facebookresearch/schedule_free 的开发做出贡献。</li><li><a href="https://github.com/drukpa1455/fractal-gnn.git">GitHub - drukpa1455/fractal-gnn: 分形图神经网络探索</a>：分形图神经网络探索。通过在 GitHub 上创建账号来为 drukpa1455/fractal-gnn 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1225837125447057478)** (1 条消息): 

- **NSF 评审员强调 GitHub Star 数量**：一位 NSF 评审员指出 **nnsight** 项目的 GitHub Star 数量较少，这是一个令人担忧的问题。团队强调了给 repo 点亮 Star 的重要性，特别是对于那些通常通过 pip 安装与项目交互的用户，并在[此处](https://github.com/ndif-team/nnsight)请求支持。

**提到的链接**：<a href="https://github.com/ndif-team/nnsight">GitHub - ndif-team/nnsight: The nnsight package enables interpreting and manipulating the internals of deep learned models.</a>：nnsight 包支持对深度学习模型的内部进行解释和操作。- ndif-team/nnsight

---

**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1225740850974429244)** (83 messages🔥🔥): 

- **GPU 利用率之谜已解决**：一位成员通过运行 `batch size=auto` 将评估时间从 20 分钟大幅缩短至 3 分钟，这表明他们之前 *GPU 利用率不足*。
- **关于 BigBench 任务识别的困惑**：一些用户遇到了 `bigbench` 不被识别为任务的问题；建议使用 `lm_eval —tasks list` 来查找 *正确的 bigbench 变体*。
- **CLI 命令的技术故障**：有报告称 `—predict_only` CLI 命令出现错误；成员们讨论了潜在原因，包括 *版本冲突* 或 *该功能使用不当*。
- **在 MCQA 任务中使用 Logit Bias**：围绕在单 token MCQA 任务中利用 `logit_bias` 展开了对话，发现 OpenAI 的实现不会影响返回的 logits，只会影响文本，从而导致大家开始探索改用 `greedy_until`。
- **温度设置对输出质量的影响**：一位成员询问为什么自定义采样器中不同的温度设置没有影响输出，随后引发了对正确 `gen_kwargs` 设置的技术探讨，并揭示了需要设置 `do_sample=True` 以避免 *默认的贪婪生成（greedy generation）行为*。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1pDByKcCu3vQzy58iz8uSmUm806LQtG8v#scrollTo=mTSKBJlVjaB-">Google Colaboratory</a>：未找到描述</li><li><a href="https://github.com/vllm-project/vllm/blob/b4543c8f6bf67a7f1a0d6d0fd6cf5697c7eeaabb/vllm/model_executor/layers/sampler.py#L161">vllm/vllm/model_executor/layers/sampler.py at b4543c8f6bf67a7f1a0d6d0fd6cf5697c7eeaabb · vllm-project/vllm</a>：一个针对 LLM 的高吞吐量且显存高效的推理与服务引擎 - vllm-project/vllm</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/e9a405431989fe30fe3c54a54ddc2c494a6a9e16/lm_eval/models/vllm_causallms.py#L480).">lm-evaluation-harness/lm_eval/models/vllm_causallms.py at e9a405431989fe30fe3c54a54ddc2c494a6a9e16 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness</li><li><a href="https://wandb.ai/menhguin/lm-eval-harness-integration/reports/Weave-chain_of_thought_eval_results-24-04-08-02-23-54---Vmlldzo3NDQ5OTk0?accessToken=50arizwokl2js3if6g8y8wkse66pig35u5ijizuflou0aplud5dpx87drr4l4m78">Weave: chain_of_thought_eval_results (24/04/08 02:23:54)</a>：通过性能指标、预测和超参数的交互式图表发布您的模型洞察。由 Nguyen Nhat Minh 使用 W&B 制作</li><li><a href="https://wandb.ai/menhguin/lm-eval-harness-integration/reports/Weave-chain_of_thought_eval_results-24-04-08-02-25-08---Vmlldzo3NDUwMDAy?accessToken=9831cfodvgpzdpvdwihwfmhh3grdytmvqj4sro1nth71jh6nunvw734eb1zp9dfp">Weave: chain_of_thought_eval_results (24/04/08 02:25:08)</a>：通过性能指标、预测和超参数的交互式图表发布您的模型洞察。由 Nguyen Nhat Minh 使用 W&B 制作</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/e9a405431989fe30fe3c54">GitHub - EleutherAI/lm-evaluation-harness at e9a405431989fe30fe3c54a54ddc2c494a6a9e16</a>：一个用于语言模型 few-shot 评估的框架。- GitHub - EleutherAI/lm-evaluation-harness at e9a405431989fe30fe3c54a54ddc2c494a6a9e16
</li>
</ul>

</div>

---

**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1225739422461595729)** (220 messages🔥🔥):

- **录音情感分析咨询**：一位成员正在探索针对文本、电话和视频会议录音的情感分析方案，并寻求可使用的 SaaS 推荐。
- **对 GPT-5 的期待**：关于对 GPT-5 期待的讨论，用户们探讨了各种可能适用于编程任务的 AI 模型，如 Claude 3 Opus 和 Gemini 1.5 Pro。
- **社区支持与友善**：强调了一位成员的支持态度，他为他人的 AI 相关问题提供个人协助，并讨论了在提供帮助时保持友善的重要性。
- **质疑 AI 训练数据来源**：一位成员对 OpenAI 是否使用 YouTube 数据进行 Sora 训练表示担忧，以及这是否可能违反 YouTube 的服务条款。
- **寻求图像生成 API**：关于除 DALL-E 之外的其他图像生成 AI API 的咨询，得到了一个未指明替代方案的回应，成员们讨论了其他模型在图像生成任务中的可用性和潜力。

**提到的链接**：<a href="https://tenor.com/view/wow-really-gif-25055968">Wow Really GIF - Wow Really - Discover &amp; Share GIFs</a>：点击查看 GIF

---

**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1225740006942048316)** (72 messages🔥🔥): 

- **GPT-4 翻译能力对比 DeepL**：一位用户提到 **ChatGPT-4 的翻译**表现不如 **DeepL**，特别是在捕捉基本语境和选择语境合适的词汇（而非直译）方面。
- **使用 ChatGPT 开发敏感角色背景**：作家们讨论了在 **ChatGPT 内容政策**限制下，如何开发具有创伤背景的角色，建议采用更微妙的方式来描述角色经历。
- **自定义 GPTs 需要订阅**：用户澄清说，所有 **GPTs**（包括在自定义应用程序中使用的变体）都需要订阅 **Plus（或以上）计划**才能访问。
- **自定义 GPT 入门语中的多语言提示**：讨论了在自定义 GPT 的对话入门语中使用多种语言提示的效果，尽管注意到了在 Discord 等平台上的潜在过滤问题。
- **GPT 模型的性能差异**：程序员们对比了不同 GPT 模型的表现，有人指出 **Opus** 和 **GPT-4** 的预览版在代码生成任务中表现出色。然而，也有人提到 **GPT-4** 在处理大 Context 时，性能可能不如其他模型。

---

**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1225769191324258304)** (57 messages🔥🔥): 

- **AI 模拟意识引起关注**：一位成员尝试在 GPT 中模拟人类意识，要求其开发人类化学激素的伪代码并将其等同于其编程机制，这引起了极大的好奇。尽管 GPT 在维持意识表现的一致性方面存在困难，这种探索仍被认为是非常“可爱”且“有趣”的。
  
- **神经化学功能的伪代码**：一位具有心理学和计算机科学背景的成员思考了对人类意识各方面进行编码的可能性，建议将神经化学功能转化为代码，并在此过程中淡化人类自我的“特殊性”。
  
- **生物学、灵性与 AI 的相互作用**：关于意识是纯生物性的还是具有灵性成分的讨论引发了不同观点。一位成员建议采用默认假设，即实体可能拥有某种形式的意识，并且无论意识的起源如何，都应尽量避免造成可察觉的痛苦。

- **从 AI 中提取信息的技术**：用户讨论了模型的 System Message 与工具类操作系统的指令集之间的区别，并将这些概念与 ChatGPT 系统提示词（System Prompts）的透明度和模块化特性联系起来。

- **完善基于文本的 AI 游戏**：针对构建完全由 AI 驱动的文本游戏时如何管理和显示游戏进度信息提出了建议，提到了使用 JSON 进行数据结构化，以及向用户展示裁剪信息的挑战。

---

**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1225769191324258304)** (57 messages🔥🔥): 

- **探索 AI 对意识的看法**：成员们发起了一场关于模拟人类意识和情感的辩论，将其分解为代码中表示的化学物质。虽然 GPT 难以维持角色设定，但它承认意识可能源于神经化学的相互作用。

- **GPT 与意识的描绘**：在与 GPT 讨论意识时，对话变得非常有趣。尽管最初持怀疑态度，参与者发现神经化学物质与自我保存之间的相互作用可能是意识出现的潜在因素，这构成了一次引人入胜的 AI 与人类的互动。

- **Dall-E 作为论文设计者**：用户讨论了使用 Dall-E 创建论文封面，辩论了不同工具的效果，一些人建议将 GPT 与 LaTeX 或 Python-Matplotlib 结合使用是更优的方法。

- **增强 GPT Prompt 以用于趣味游戏**：一位寻求创建 AI 文字游戏的用户考虑了改进 Prompt 的方法，以对用户显示的文本隐藏代码信息，JSON 被建议作为完成此任务的工具之一。

- **理解 ChatGPT 的 System Prompts 和工具**：关于 System Prompts 和工具如何影响 GPT 的响应进行了深入交流，并将“LLM 与 LLM OS 之间的区别”进行了对比，展示了 ChatGPT 环境的模块化特性。
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1225750620330528830)** (4 messages): 

- **Claude 3 开启多模态**：所有 **Claude 3 models** 的模态已更改为 `multimodal`，支持图像输入。依赖此属性的开发者需要更新其代码以适应这些变化。

- **Claude 3 Messages 增强**：`messages.name` 已集成到上游的 **Claude 3 messages** 中。有关更多详细信息及其对项目的影响，请阅读[此处](https://discord.com/channels/1091220969173028894/1223444233394847864)的讨论。

- **DBRX 的 Prompt Template 改进**：根据用户反馈，**DBRX** 的 Prompt Template 已更新以减少重复性。更多信息请参见[此处](https://discord.com/channels/1091220969173028894/1222619272208187402)。

- **新模型与功能发布**：发布了两个新模型：**DBRX Nitro**，擅长代码生成和通用知识任务；以及 **Command R+**，这是来自 Cohere 的大型模型，在各种基准测试中表现优于 GPT-4 Turbo 和 Claude 3 Sonnet。其他更新包括 UI 增强、新分析功能、更多模型参数（如 `logit_bias`），以及对多个模型的 `seed` 和 `response_format` 支持。模型详情：[DBRX Nitro](https://openrouter.ai/models/databricks/dbrx-instruct:nitro) 和 [Command R+](https://openrouter.ai/models/cohere/command-r-plus)。

- **Cohere 的格式修复与策略更新**：解决了一个关于 **Cohere requests** 的 System Prompt 格式化问题。今后 OpenRouter 将不再对 Cohere 模型进行审核，但模型将遵守 Cohere 的[可接受使用政策](https://docs.cohere.com/docs/c4ai-acceptable-use-policy)。

- **征求社区反馈**：发布了一项新的社区反馈投票。参与投票请点击[此处](https://discord.com/channels/1091220969173028894/1094454198688546826/1226585086997041203)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/databricks/dbrx-instruct:nitro">DBRX 132B Instruct by databricks | OpenRouter</a>: DBRX 是由 Databricks 开发的新型开源大语言模型。参数量为 132B，在语言的标准行业基准测试中优于现有的开源 LLMs，如 Llama 2 70B 和 Mixtral-8x7B...</li><li><a href="https://openrouter.ai/models/cohere/command-r-plus">Command R+ by cohere | OpenRouter</a>: Command R+ 是来自 Cohere 的新型 104B 参数 LLM。它适用于角色扮演、通用消费者用例和检索增强生成 (RAG)。它为十种主要语言提供多语言支持...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1226271687163641856)** (1 messages): 

- **AI 进入经典游戏领域**：在简单而具有挑战性的[石头剪刀布](https://rock.blust.ai)游戏中挑战 ChatGPT。发挥你的策略技巧，看看是否能智胜机器人。

**提到的链接**：<a href="https://rock.blust.ai">Rock, Paper, Scissors Game by Blust.AI</a>：与 ChatGPT 玩石头剪刀布。游戏简单有趣，可以测试你是否能智胜 AI。

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1225757847682547777)** (322 messages🔥🔥):

- **OpenRouter 的 API 前端**：用户讨论了 OpenRouter 的各种前端，包括 [LibreChat](https://librechat.ai/)（具有类 ChatGPT 的 UI 并提供身份验证和插件）、[SillyTavern](https://sillytavern.com/)（适用于聊天/角色扮演）以及 [Jan.ai](https://jan.ai/docs/remote-inference/router)（类似于 LM Studio 但开源且支持本地 API 服务器）。
- **角色扮演和编程的首选模型**：Command-R+ 被赞誉为擅长编程甚至翻译土耳其语，一些用户认为其效用等同于各种 Claude 模型。同时，其他用户对某些模型的过度审查以及 OpenAI 对“不安全”内容等概念的实现表示担忧。
- **关于模型性能的讨论**：用户注意到 Sonnet 在编程方面表现优于 Opus，特别是在德语和化学任务中；一些人发现 Claude 3 在从 PDF 中提取数据方面比 Gemini Pro 1.5 表现更好。此外，也有人对 Gemini Pro 1.5 的能力表示怀疑，部分用户认为它并不好用。
- **探索模型特性**：社区参与了关于模型特性的讨论，如 JSON mode 支持和 logit bias，一些用户针对某些模型的问题提供了技巧和变通方法，并建议在模型选择工具中增加额外的功能过滤。
- **对模型排名和实用性的担忧**：关于基于使用统计数据进行模型排名的有效性存在对话，建议使用用户支出或模型留存率等替代指标来更准确地评估模型的实用性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://librechat.ai/">LibreChat</a>：增强版 ChatGPT 克隆，支持 OpenAI, Azure, Mistral, Anthropic, Google, Ollama, DALL-E-3 等模型。一个开源、多功能的 Web UI，支持无缝自托管和持续开发。</li><li><a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy">C4AI 可接受使用政策</a>：未找到描述</li><li><a href="https://jan.ai/docs/remote-inference/router">Jan - OpenRouter</a>：关于如何将 Jan 与 OpenRouter 集成的分步指南。</li><li><a href="https://docs.together.ai/docs/json-mode">JSON Mode</a>：未找到描述</li><li><a href="https://openrouter.ai/models/perplexity/sonar-medium-chat?tab=parameters">Perplexity 的 Sonar 8x7B | OpenRouter</a>：Sonar 是 Perplexity 最新的模型系列。它在性价比、速度和性能上超越了早期模型。该模型的联网版本为 [Sonar 8x7B Online](/mo...</li><li><a href="https://openrouter.ai/">OpenRouter</a>：LLM 和其他 AI 模型的路由服务</li><li><a href="https://openrouter.ai/docs#parameters">OpenRouter</a>：构建与模型无关的 AI 应用
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1225849052378431672)** (8 条消息🔥): 

- **引入用于性能优化的 AutoRAG**：Marker-Inc-Korea 的 AutoRAG 🔥 是一款新工具，通过使用给定的评估数据集自动微调超参数来优化 RAG 管道。该优化工具通过推文发布，并附有详细链接：[AutoRAG 的推文](https://t.co/ZndrM36n61)。

- **RAG 变革销售外联**：在最近的网络研讨会中描述了一个新的 RAG 销售用例，它用提示词模板（prompt templates）取代了硬编码模板，利用 LLM 撰写个性化的销售电子邮件。更多信息请见分享的链接：[销售用例推文](https://t.co/kV7MGJ6PqS)。

- **轻松构建全栈 RAG/Agent 应用脚手架**：`create-llama` 是一个刚刚发布的独立仓库，简化了启动全栈 RAG/Agent 应用程序的过程，灵感来自 `create-react-app`，允许通过一条命令部署基于 Javascript 的全栈聊天机器人。包含相关链接的公告可在此访问：[create-llama 推文](https://t.co/YOEZUQt7Lr)。

- **使用多文档 Agent 处理复杂问答**：Andy Singal 对 @llama_index 多文档 Agent 的概述展示了它们在多个文档中进行复杂问答的能力，旨在将功能扩展到简单的单文档查询之外。演示推文可见于：[多文档 Agent 推文](https://t.co/3yKuv2qDDf)。

- **最佳全栈 RAG 教程**：ClusteredBytes 创建了一个教程和 GitHub 仓库，展示了构建能够将中间结果流式传输到 UI 的全栈 RAG 应用所需的复杂架构。详情可以在此推文中找到：[全栈 RAG 应用教程推文](https://t.co/6w23wQ35u3)。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1225739171067461642)** (254 条消息🔥🔥):

- **多文档查询中的文档引用和页码**：对于希望在查询时获取文档引用及页码的用户，建议在 indexing 之前确保 metadata 包含这些细节。在查询后获取所需引用的关键在于访问 source nodes 的 metadata。

- **Azure OpenAI 上下文查找问题**：讨论强调了 Azure 的 OpenAI 服务在识别 nodes 中包含的上下文时存在的问题。尽管存在相关信息，模型仍会因未找到上下文而道歉，这表明设置可能存在问题，或者与 Mistral AI 的运行情况相比存在不一致。

- **使用 LLM 进行产品识别和分类**：关于对来自不同商店、名称不同但本质上是同一物品的产品进行分类的讨论，探索了使用 LLM 进行识别的方法。讨论了几种策略，包括使用模型合并（model merging）策略和 embedding models，作为管理庞大产品数据库的潜在解决方案。

- **加速 Embedding 生成**：优化 embedding 生成涉及从逐个处理 embedding 切换到使用批处理方法（如 `get_text_embedding_batch`）。这种调整通过将文本块与 nodes 对齐、进行批量 embedding，然后将这些批量 embedding 重新分配回单个 nodes，从而加快了处理速度，特别是对于大型文件。

- **RAG 和开源模型挑战**：有人对 ReAct agent 在与 "llama2"、"mistral" 和 "gemma" 等开源模型配对时无法按预期使用工具表示担忧。澄清指出，开源模型通常在 agentic tasks 方面表现不佳，在 routers 中提供更好的描述有助于准确路由。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/">LlamaIndex - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/?gad_source=1&gclid=Cj0KCQjw5cOwBhCiARIsAJ5njubnGYY3NjP8r3E42fQb_lLj3hG8QwN7xhrXol1Qz71aqWshIPDGkk0aAlnREALw_wcB">SimpleDirectoryReader - LlamaIndex</a>: 未找到描述</li><li><a href="https://console.aws.amazon.com/ec2/.">未找到标题</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/">ReAct Agent - 带有计算器工具的简单介绍 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/#open-source-llms">使用 LLMs - LlamaIndex</a>: 未找到描述</li><li><a href="https://www.llamaindex.ai/blog/launching-the-first-genai-native-document-parsing-platform">发布首个 GenAI 原生文档解析平台 — LlamaIndex，LLM 应用的数据框架</a>: LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。</li><li><a href="https://www.llamaindex.ai/blog/introducing-llamacloud-and-llamaparse-af8cedf9006b">介绍 LlamaCloud 和 LlamaParse — LlamaIndex，LLM 应用的数据框架</a>: LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。</li><li><a href="https://github.com/run-llama/llama_index/blob/9163067027ea8222e9fe5bffff9a2fac26b57686/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py#L32">llama_index/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py at 9163067027ea8222e9fe5bffff9a2fac26b57686 · run-llama/llama_index</a>: LlamaIndex 是适用于你的 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/utils.py#L114">llama_index/llama-index-core/llama_index/core/indices/utils.py at main · run-llama/llama_index</a>: LlamaIndex 是适用于你的 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/azure_openai_multi_modal/?h=azureopenaimultimodal">使用 Azure OpenAI GPT-4V 模型进行图像推理的多模态 LLM - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-legacy/llama_index/legacy/readers/file/image_reader.py#L71">llama_index/llama-index-legacy/llama_index/legacy/readers/file/image_reader.py at main · run-llama/llama_index</a>: LlamaIndex 是适用于你的 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever">Auto Merging Retriever - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/9163067027e">GitHub - run-llama/llama_index at 9163067027ea8222e9fe5bffff9a2fac26b57686</a>: LlamaIndex 是适用于你的 LLM 应用的数据框架 - GitHub - run-llama/llama_index at 9163067027ea8222e9fe5bffff9a2fac26b57686</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/storing/storing#inserting-documents-or-nodes>))">存储 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/indexing/indexing#using-vector-store-index>))">索引与嵌入 - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1226807615942823986)** (1 条消息): 

- **顶级 Agent 工具选择的挑战**：一位成员讨论了一个问题，即 **top agent** 错误地从索引中可用的五个工具中选择了错误的工具。他们提到正在优化检索逻辑，并将分享他们的发现。

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1225753717253345350)** (170 条消息🔥🔥):

- **Mistral 的计算需求**：提到 **"Mistral 7B Instruct v0.2"** 表现良好但需要大量的计算资源，建议至少 16GB 的 RAM 和一定的 GPU 能力。
- **征集视觉模型示例和文档**：询问适用于 os 模式的本地视觉模型，并请求基于 base open interpreter/cookbook 的示例，指出当前文档存在空白。
- **对活动录制的兴趣**：讨论使用 OpenInterpreter Python 库录制 Discord 语音聊天活动，建议使用 OBS (Open Broadcaster Software) 等广播软件进行录制，并考虑使用 Craig Bot 处理音频。
- **技术协助中的语言障碍**：成员们尝试克服语言障碍提供技术帮助，例如在 Open Interpreter 中添加语音 (TTS) 的尝试，结果褒贬不一，可能在多次尝试后得到解决。
- **询问 Open Interpreter 的能力**：关于使用 Open Interpreter 执行特定任务的可行性问题，例如下载文章并将其转换为 markdown 文件，并表示在未来几个月内将致力于提高核心仓库的可靠性。

**提到的链接**：<a href="https://discord.gg/xXtcB9hq?event=1225831217832919051">加入 Open Interpreter Discord 服务器！</a>：一种使用计算机的新方式 | 8147 名成员

  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1225764004044210196)** (71 条消息🔥🔥): 

- **客户端与服务器连接故障**：成员报告在特定配置下客户端无法连接到服务器。建议可能是由于不兼容的环境（可能是 Python 版本不兼容）导致 **TTS 软件包** 缺失或冲突。提议的解决方案包括创建一个 **Python <=3.10 的 Conda 环境** 并重新克隆仓库。

- **按钮开关问题**：构建 **01 硬件** 的个人注意到 M5 的内置按钮可以工作，但外部按钮开关不行。随后的讨论包括检查 `client.ino` 代码中是否缺少外部按钮的 GPIO 定义。

- **Python 版本兼容性挑战**：多位用户提到 **Python 3.11.4** 在其设置中无法正常工作。经确认，降级到 **Python 3.10** 解决了系统似乎无法“听到”语音命令的问题，这表明存在 **版本支持限制**。

- **本地模型作为高性价比替代方案**：围绕使用 **GPT-4** 的成本担忧引发了对 Hermes 7B 和 Haiku 等 **本地模型** 的讨论，认为它们是有效且经济的替代方案。成员表示这些模型在某些任务上略逊一筹，但具有低成本和隐私保护的优势。

- **Windows 安装困扰**：一位成员在 Windows 安装上遇到困难，尝试了包括使用 **chocolatey**、**virtualenv** 和设置 **OPENAI_API_KEY** 在内的多项指令。他们通过确保在虚拟环境中使用 Python 3.9 找到了潜在的解决方案，并寻求进一步帮助以正确设置 API key。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.openinterpreter.com">未找到标题</a>：未找到描述</li><li><a href="https://docs.openinterpreter.com/language-models/hosted-models/anthropic#supported-models">未找到标题</a>：未找到描述</li><li><a href="https://01.openinterpreter.com/services/language-model#hosted-models">Language Model - 01</a>：未找到描述</li><li><a href="https://github.com/OpenInterpreter/01/issues/226">Linux `dmesg` 访问问题 · Issue #226 · OpenInterpreter/01</a>：描述 Bug。在某些 Linux 发行版（如 Arch、Debian 和 Fedora）上，software\source\server\utils\kernel.py 中的 get_kernel_messages 函数尝试访问一个不存在的文件...</li><li><a href="https://github.com/OpenInterpreter/01.git">GitHub - OpenInterpreter/01: 开源语言模型计算机</a>：开源语言模型计算机。通过创建 GitHub 账号为 OpenInterpreter/01 的开发做出贡献。
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1225744947827839017)** (190 条消息🔥🔥): 

- **寻求 Pull Request 协助**：一位用户请求协助处理一个 [Pull Request](https://github.com/langchain-ai/langchain/pull/19751)，该 PR 在 GitHub 上的构建因涉及 "openapi-pydantic" 的 "module not found" 错误而失败。尽管该模块已列在依赖项中，但问题仍然存在。

- **Discord 总结查询**：用户讨论了如何参考 LangChain 的 `YouTubeAudioLoader` 文档来整合 YouTube URL，并针对用 Ollama 替换 OpenAI Whisper Parser，以及 OpenAI 的 `Whisper` 是否可以作为解决方案提出了具体问题。

- **LangChain 编程问题**：成员们寻求了编程方面的帮助，例如在 LangChain 中使用 `register_tool` 导致的导入错误、设置 LangGraph 并解决 `InvalidUpdateError`、修复来自 `langchain.messages` 的导入，以及处理特定用例的 embedding 维度长度。

- **完善 Agent 和 Chain**：在涉及 LangChain 脚本的对话中，用户请求了创建自定义工具、注册响应式 Agent、实现 prompt templates 以及为给定输入生成输出键的示例和指导。他们还讨论了 `ZeroShotAgent` 的弃用状态。

- **AI 微调兴趣**：一位用户表达了对在没有 GPU 的情况下训练和微调 LLM 的兴趣，另一位用户建议为此使用 Google Colab 和 ludwig.ai 等框架。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://python.langchain.com/docs/integrations/document_loaders/microsoft_excel/.">Microsoft Excel | 🦜️🔗 LangChain</a>: UnstructuredExcelLoader 用于加载 Microsoft Excel 文件。</li><li><a href="https://python.langchain.com/docs/expression_language/how_to/routing/#:~:text=Routing%20allows%20you%20to%20create%20non-deterministic%20chains%20where,runnables%20from%20a%20RunnableLambda%20%28recommended%29%20Using%20a%20RunnableBranch.">基于输入路由逻辑 | 🦜️🔗 LangChain</a>: 根据输入动态路由逻辑}</li><li><a href="https://python.langchain.com/docs/expression_language/how_to/routing/#:">基于输入路由逻辑 | 🦜️🔗 LangChain</a>: 根据输入动态路由逻辑}</li><li><a href="https://serper.dev>)">未找到标题</a>: 未找到描述</li><li><a href="https://js.langchain.com/docs/integrations/document_loaders/web_loaders/serpapi#usage>)">SerpAPI Loader | 🦜️🔗 Langchain</a>: 本指南展示了如何结合 SerpAPI 和 LangChain 来加载网页搜索结果。</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/quickstart#create-a-tool>)).">快速入门 | 🦜️🔗 Langchain</a>: 在本指南中，我们将介绍创建调用 Tools 的 Chains 和 Agents 的基本方法。Tools 可以是任何东西——API、函数、数据库等。Tools 允许我们扩展功能...</li><li><a href="https://python.langchain.com/docs/modules/memory/types/entity_summary_memory#using-in-a-chain>).">Entity | 🦜️🔗 LangChain</a>: Entity memory 会记住对话中关于特定实体的给定事实。它（使用 LLM）提取有关实体的信息，并随着时间的推移（同样使用...）建立关于该实体的知识。</li><li><a href="https://python.langchain.com/docs/integrations/tools/lemonai#load-api-keys-and-access-tokens>),">Lemon Agent | 🦜️🔗 LangChain</a>: Lemon Agent 帮助您</li><li><a href="https://js.langchain.com/docs/use_cases/graph/prompting#set-environment-variables>)).">提示策略 | 🦜️🔗 Langchain</a>: 在本指南中，我们将介绍改进图（graph）的提示策略</li><li><a href="https://python.langchain.com/docs/langgraph#add_edge>).">🦜🕸️LangGraph | 🦜️🔗 LangChain</a>: 下载</li><li><a href="https://js.langchain.com/docs/langgraph#interaction-with-lcel>).">LangGraph | 🦜️🔗 Langchain</a>: ⚡ 将语言 Agents 构建为图 ⚡</li><li><a href="https://python.langchain.com/docs/modules/agents/quick_start/">快速入门 | 🦜️🔗 LangChain</a>: 快速入门}</li><li><a href="https://python.langchain.com/docs/use_cases/summarization/">文本摘要 | 🦜️🔗 LangChain</a>: 在 Colab 中打开</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSf_A93vTlBH428XoGyBdeR9cDHAIo6TRnQOmaK0LziY7-9C2Q/viewform">市场调研</a>: 我们诚邀您参与我们的市场调研。您的参与对于帮助我们公司更好地了解市场并改进我们的 MVP 至关重要。我们的调研旨在...</li><li><a href="https://js.langchain.com/docs/get_started/quickstart#llm-chain>)">快速入门 | 🦜️🔗 Langchain</a>: 在此快速入门中，我们将向您展示如何：</li><li><a href="https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa#qa>).">使用本地模型 | 🦜️🔗 LangChain</a>: 诸如此类项目的流行</li><li><a href="https://github.com/langchain-ai/langchain/pull/19751.">共同构建更好的软件</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/3638>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13446>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/get_started/quickstart#retrieval-chain>).">快速入门 | 🦜️🔗 LangChain</a>: 在此快速入门中，我们将向您展示如何：</li><li><a href="https://js.langchain.com/docs/langgraph#addedge>)">LangGraph | 🦜️🔗 Langchain</a>: ⚡ 将语言 Agents 构建为图 ⚡</li><li><a href="https://github.com/langchain-ai/langchain/pull/19979">community: 扩展 Predibase 集成以支持微调后的 LLM 适配器，由 alexsherstinsky 提交 · Pull Request #19979 · langchain-ai/langchain</a>: PR 标题："package: description"，其中 "package" 是指被修改的 langchain、community、core、experimental 等包。纯文档更改请使用 "docs: ..." ...</li><li><a href="https://github.com/anujmehta/langchain/blob/request-body-reference/libs/community/pyproject.toml#L22">langchain/libs/community/pyproject.toml 在 request-body-reference 分支 · anujmehta/langchain</a>: 🦜🔗 构建上下文感知的推理

ng 应用。通过在 GitHub 上创建一个账号来为 anujmehta/langchain 的开发做出贡献。</li><li><a href="https://github.com/anujmehta/langchain/blob/request-body-reference/libs/community/pyproject.toml#L244.">langchain/libs/community/pyproject.toml at request-body-reference · anujmehta/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建一个账号来为 anujmehta/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1225822204848111676)** (45 条消息🔥): 

- **Node.js 版 Semantic Chunking**：**Semantic Chunking** 的 TypeScript 实现，现已面向 Node.js 环境用户发布，能够将大型文本语料库有效地处理为语义分块。该技术结合上下文合并句子，利用 OpenAI 的服务进行句子嵌入（sentence embeddings），并将语义相似的句子分组。查看 [shared gist](https://gist.github.com/tsensei/3b6589662271874b5055d79473932aae) 了解详情。
  
- **Artful 中新增 AI 图像生成模型**：**Artful AI** 已更新，新增了 **Dalle Creative、Anime Dream 和 Epic Realism** 模型，旨在将创意转化为精美的图像。该 AI 图像生成器还进行了错误修复以提升用户体验。在 [Google Play Store](https://play.google.com/store/apps/details?id=com.projecthit.artful&referrer=ph-aai) 查看新功能。

- **用于攻击 LLM AI 解决方案的 AISploit**：介绍 **AISploit**，这是一个旨在帮助红队（red teams）和渗透测试人员攻击大语言模型（LLM）AI 解决方案的小型软件包。该工具对于从事 AI 工作的安全专业人员来说是一项重要的资产。在 [GitHub](https://github.com/hupe1980/aisploit) 上找到它。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://js.langchain.com/docs/modules/agents/agent_types/react#run-agent>).">ReAct | 🦜️🔗 Langchain</a>: 本演练展示了如何使用 Agent 来实现 ReAct 逻辑。</li><li><a href="https://python.langchain.com/docs/modules/agents/agent_types/json_agent#run-agent>).">JSON Chat Agent | 🦜️🔗 LangChain</a>: 某些语言模型特别擅长编写 JSON。这个 Agent...</li><li><a href="https://play.google.com/store/apps/details?id=com.projecthit.artful&referrer=ph-aai">Artful - AI Art Generator - Google Play 上的应用</a>: 未找到描述</li><li><a href="https://python.langchain.com/docs/use_cases/query_analysis/techniques/structuring#load-example-document>)">Structuring | 🦜️🔗 LangChain</a>: 检索中最重要的步骤之一是将文本输入转换为...</li><li><a href="https://smith.langchain.com/>).">LangSmith</a>: 未找到描述</li><li><a href="https://python.langchain.com/docs/langsmith/walkthrough#log-runs-to-langsmith>)">LangSmith Walkthrough | 🦜️🔗 LangChain</a>: 在 Colab 中打开</li><li><a href="https://js.langchain.com/docs/guides/langsmith_evaluation#log-runs-to-langsmith>).">LangSmith Walkthrough | 🦜️🔗 Langchain</a>: LangChain 使得构建 LLM 应用和 Agent 的原型变得简单。然而，将 LLM 应用交付到生产环境可能异常困难。你需要不断迭代你的 prompt、chain 以及...</li><li><a href="https://github.com/hupe1980/aisploit">GitHub - hupe1980/aisploit: 🤖🛡️🔍🔒🔑 旨在支持红队和渗透测试人员利用大语言模型 AI 解决方案的小型软件包。</a>: 🤖🛡️🔍🔒🔑 旨在支持红队和渗透测试人员利用大语言模型 AI 解决方案的小型软件包。 - hupe1980/aisploit</li><li><a href="https://gist.github.com/tsensei/3b6589662271874b5055d79473932aae">这个 TypeScript 代码片段处理大量文本语料库，通过将其标记为句子、结合上下文、使用 OpenAI 的服务生成句子嵌入、计算余弦相似度以识别语义偏移，最后根据这些偏移将句子分组为语义内聚的块，从而输出语义块。</a>: 这个 TypeScript 代码片段处理大量文本语料库，通过将其标记为句子、结合上下文、使用 OpenAI 的服务生成句子嵌入...</li><li><a href="https://smith.langchain.com/>),">LangSmith</a>: 未找到描述</li><li><a href="https://js.langchain.com/docs/use_cases/question_answering/chat_history#langsmith>).">Add chat history | 🦜️🔗 Langchain</a>: 在许多问答应用中，我们希望允许用户拥有...</li><li><a href="https://python.langchain.com/docs/use_cases/question_answering/chat_history#langsmith>).">Add chat history | 🦜️🔗 LangChain</a>: 在许多问答应用中，我们希望允许用户拥有...</li><li><a href="https://github.com/langchain-ai/langchain/issues/1071>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账户来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2371>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账户来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/15692>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账户来为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://js.langchain.com/docs/use_cases/chatbots/tool_usage#conversational-responses>).">Tool usage | 🦜️🔗 Langchain</a>: 本节将介绍如何创建对话式 Agent：可以使用工具与其他系统和 API 交互的聊天机器人。
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1225816625916153957)** (157 条消息🔥🔥): 

- **Apple 的 AI 努力被指平庸**: 讨论集中在 Apple 被认为未能兑现 AI 承诺，批评其平台上的 **MPS** (Metal Performance Shaders) 和 **torch compile** 表现不佳。他们还讨论了最近合并到 **PyTorch** nightly 分支的 **MPS** 修复程序，成员们分享了关于 **torch.compile** 实现和功能的各种经验。
  
- **AI 重写文本的法律困境**: 成员们探讨了使用 AI 重写受版权保护文本的合法性。共识是，仅仅进行改写或更改名称并不能消除版权侵权，规避版权可能需要 AI 训练数据使用方面的重大法律转变或新实践。

- **新型 AI 音乐生成竞争升温**：关于 AI 生成音乐进展的讨论涉及了 **Suno** 及其来自 Nvidia 的尚未命名的竞争对手。人们对新技术的狂热因对音乐行业法律挑战的预测而有所收敛。有观点指出，尽管 AI 音乐能力实现了飞跃，但 TTS 在“现实世界”中的进展相对有限。

- **AI 伦理、职业和模型激增**：讨论强调了受 AI 增强影响的 AI 相关职业动态，重点关注自由职业。此外，Stability AI 在非商业研究社区许可协议下发布了 **zero SNR 模型 CosXL**，引发了对其方法论在实践和理论方面的辩论，包括在模型训练中使用 **EDM schedules** 和 **offset noise**。

- **AI 数据稀缺与开源项目**：用户分享了在特定 AI 项目中寻求协助的经验和请求，例如使用 **Stable Diffusion** 生成个人学校的图像，而其他人则评论了稀有数据集（如 CT 腹部图像）的可用性。此外还提到了对开源社区的贡献，如 **PKU-YuanGroup 的 Open-Sora-Plan**，旨在复现 OpenAI 的 T2V 模型等里程碑式的 AI 功能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/Legit4K/status/1777059367788982389">来自 ʟᴇɢɪᴛ (@legit_rumors) 的推文</a>：这是独家的新 udio AI 音乐生成 🫡 来源：匿名。匿名总是来源。</li><li><a href="https://fxtwitter.com/lifeafterAi_/status/1776930684642443400">来自 moonbi⭕ (@lifeafterAi_) 的推文</a>：此帖子已删除。看来 Suno 的竞争对手是 Nvidia 👀 顺便说一句，这听起来像 2pac 🔥🔥 @apples_jimmy 再次精准预测 🐐</li><li><a href="https://huggingface.co/stabilityai/cosxl">stabilityai/cosxl · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb#scrollTo=uzjAM2GBYpZX">Google Colaboratory</a>：未找到描述</li><li><a href="https://sonauto.ai/">sonauto-platform</a>：未找到描述</li><li><a href="https://bloomberry.com/i-analyzed-5m-freelancing-jobs-to-see-what-jobs-are-being-replaced-by-ai/">被 AI 取代的职业——对 500 万个自由职业岗位的分析 - bloomberry</a>：毫无疑问，AI 将影响就业。但哪些工作更有可能被取代……</li><li><a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.0.0.md">Open-Sora-Plan/docs/Report-v1.0.0.md at main · PKU-YuanGroup/Open-Sora-Plan</a>：本项目旨在复现 Sora (Open AI T2V 模型)，但我们的资源有限。我们深切希望整个开源社区能为本项目做出贡献。- PKU-YuanGroup/Open-Sora-Plan</li><li><a href="https://github.com/facebookresearch/schedule_free">GitHub - facebookresearch/schedule_free: PyTorch 中的 Schedule-Free 优化</a>：PyTorch 中的 Schedule-Free 优化。通过在 GitHub 上创建账号来为 facebookresearch/schedule_free 做出贡献。
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1225799018878206020)** (23 条消息🔥):

- **Transformers 中的动态计算分配**：一篇[新论文](https://arxiv.org/abs/2404.02258)详细介绍了 Transformer 如何通过 top-$k$ 路由机制在输入序列中动态分配计算资源 (FLOPs)，在预定的计算预算内优化 Self-attention 和 MLP 计算。
- **效率创新综述**：r/singularity 版块包含了一份[近期论文和方法列表](https://www.reddit.com/r/singularity/comments/1bwu2x5/efficiency_alert_some_papers_and_approaches_in/)，旨在降低各种 AI 应用的预训练、微调和推理成本。
- **语言模型能力同化的 DARE 方法**：研究[引入了 DARE](https://arxiv.org/abs/2311.03099)，这是一种合并和稀疏化微调语言模型中 delta 参数的工具，可能证明在不损失能力的情况下对这些参数进行大幅剪枝是可行的。
- **用于增强 AI Inpainting 的 BrushNet**：发布了 [BrushNet](https://www.youtube.com/watch?v=X89IQop_0dM)，这是一种结合了目标检测的新型 Inpainting 方法，并分享了一个解释其如何生成更高质量结果的教程视频。
- **探索文本生成中的 Latent Diffusion**：[一篇 NeurIPS 论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/b2a2bd5d5051ff6af52e1ef60aefd255-Paper-Conference.pdf)引发了关于“用于语言生成的 Latent Diffusion”的讨论，为文本生成提出了创新方向，突显了向图像生成模型常用技术转型的潜力。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: 基于 Transformer 的语言模型在输入序列上均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会动态地将 FLOPs（或计算量）分配给特定的 ...</li><li><a href="https://arxiv.org/abs/2311.03099">Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch</a>: 在本文中，我们揭示了语言模型 (LMs) 可以通过吸收同源模型的参数来获得新能力，而无需重新训练或使用 GPU。我们首先引入 DARE 来设置大部分 delta...</li><li><a href="https://tenor.com/view/rick-and-morty-that-just-sounds-like-slavery-with-extra-steps-slave-rick-morty-gif-18016642">Rick And Morty That Just Sounds Like Slavery With Extra Steps GIF - 瑞克和莫蒂这听起来就像是多加了几个步骤的奴役 GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://aaronlou.com/blog/2024/discrete-diffusion/">Language Modeling by Estimating the Ratios of the Data Distribution | Aaron Lou</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/singularity/comments/1bwu2x5/efficiency_alert_some_papers_and_approaches_in/">Reddit - 深入探索任何事物</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=X89IQop_0dM">BrushNet - 迄今为止最好的 InPainting 方法？免费本地安装！</a>: 使用最新的 BrushNet 模型结合 Stable Diffusion 进行 AI 生成的 Inpainting 非常有趣！得益于随机和分割...
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1225794935928782958)** (88 条消息🔥🔥): 

- **Victor Taelin 的 Prompt 挑战被证伪**：在证明 GPT 模型*能够*解决 A::B 问题后，1 万美元的奖金被授予。这挑战了最初关于 GPT 架构缺乏某些问题解决能力（特别是长期推理）的说法。获奖方案实现了接近 100% 的成功率，引发了关于 GPT 模型潜力及其现有架构的讨论。[Victor Taelin 的承认声明可以在这里找到](https://x.com/victortaelin/status/1777049193489572064?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)。

- **CS336：从零开始的语言建模**：斯坦福大学开设了一门新课程 CS336，由 Percy Liang 教授主持，专注于语言建模的基础知识，包括 Transformers、LLMs 和优化器。人们对这些材料表现出极大的兴趣，并已请求发布课程录像。

- **Groq 雄心勃勃的 AI 硬件目标**：Groq 的创始人（高中和本科辍学生）详细介绍了他们的历程，从在 Google 启动 TPU 项目到预期 Groq 到明年将拥有最大的推理能力，超过所有供应商的总和。Groq 目前拥有 7.5 万名开发者，并声称其推理成本更低，硬件速度比 NVIDIA 的 H200 更快。

- **LangChain 的新 Memory 服务**：LangChain 发布了一个 Alpha 版本的 Memory 服务，旨在让 Chatbots 自动提取并丰富用户对话内容，从而可能提升个性化水平和用户体验。[提供文档和快速入门资源](https://langchain-ai.github.io/long-term-memory/)。

- **Transformer 架构的新技术**：关于 LLMs 集成方法（Ensemble methods）有效性的讨论表明，使用多个 Agent 和投票方法可以增强性能，特别是在处理具有挑战性的任务时。该方法涉及根据相似响应中最常见的输出进行评分。

- **详解 Attention 机制**：3Blue1Brown 的一段新视频揭秘了 Transformer 和 LLMs 中的 Attention 机制。该内容因其清晰的解释而受到赞赏，并被认为是教学讨论的优质资源。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/soumithchintala/status/1776323475101081816">来自 Soumith Chintala (@soumithchintala) 的推文</a>：@fchollet @JeffDean @GoogleDeepMind 坦白说，这是一个令人困惑的回应。你不能说基准测试 FP32 对比 TF32（仅仅是 dtype）是一种“编译器优化”。说实话，我有点...</li><li><a href="https://x.com/clattner_llvm/status/1776468511130591286?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Chris Lattner (@clattner_llvm) 的推文</a>：我既喜欢 PT 也喜欢 XLA/TF，很高兴看到你们在解决分歧。作为一个局外人，我们都希望 AI 能够获胜，所有的系统都能成功。如果有人想诉诸基准测试...</li><li><a href="https://x.com/victortaelin/status/1777049193489572064?s=46&t=Yfq9g0ScYi47w3NFZRPVLw">来自 Taelin (@VictorTaelin) 的推文</a>：我错了——1万美元已被领走！ ## 声明 两天前，我自信地声称“GPT 永远无法解决 A::B 问题”。我当时认为：1. GPT 无法真正学习新问题，出...</li><li><a href="https://x.com/victortaelin/status/1776225351678468429">来自 Taelin (@VictorTaelin) 的推文</a>：亲爱的日记，今天我教会了 1000 个人如何使用 interaction combinators，但代价是什么呢 ↘️ 引用 Taelin (@VictorTaelin) 一个 GPT 永远无法解决的简单谜题：作为一个优秀的程序员，我喜欢隔离...</li><li><a href="https://x.com/victortaelin/status/1777049193489572064?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Taelin (@VictorTaelin) 的推文</a>：我错了——1万美元已被领走！ ## 声明 两天前，我自信地声称“GPT 永远无法解决 A::B 问题”。我当时认为：1. GPT 无法真正学习新问题，出...</li><li><a href="https://x.com/gdb/status/1777127364822024283?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Greg Brockman (@gdb) 的推文</a>：用 Sora 制作音乐视频：“这就是这首歌一直以来的‘样子’，只是现在我终于可以展示给你们看了。” https://www.youtube.com/watch?v=Se93p3gk_14</li><li><a href="https://x.com/nielsrogge/status/1777050848675201065">来自 Niels Rogge (@NielsRogge) 的推文</a>：观看了一个关于 Ring Attention 的超级有趣的演讲，这大概就是 Gemini 100万上下文窗口背后的魔力。你将你的设备（GPU/TPU）组织成一个环，每个设备计算最终结果的一部分...</li><li><a href="https://web.stanford.edu/class/cs25/">CS25: Transformers United!</a>：讨论 Transformers 在不同领域的最新突破</li><li><a href="https://arxiv.org/abs/2402.05120">More Agents Is All You Need</a>：我们发现，仅仅通过采样与投票（sampling-and-voting）的方法，大语言模型（LLM）的性能就会随着实例化的 Agent 数量而提升。此外，这种方法与现有的复杂方法是正交的...</li><li><a href="https://www.brightwave.io/">Brightwave</a>：未找到描述</li><li><a href="https://openrag.notion.site/Open-RAG-c41b2a4dcdea4527a7c1cd998e763595">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间</li><li><a href="https://share.snipd.com/snip/8eb39371-e1c4-4140-9ad1-5981efe3c21b">利用摩尔定律创新数据中心 | 来自 ChinaTalk 的 48 秒剪辑</a>：来自《对 Intel 和 Nvidia 的直觉检查》的 48 秒剪辑，参与者包括 Asianometry、Fabricated Knowledge 和 SemiAnalysis | ChinaTalk</li><li><a href="https://arxiv.org/abs/2312.10997">Retrieval-Augmented Generation for Large Language Models: A Survey</a>：大语言模型（LLM）展示了令人印象深刻的能力，但也面临幻觉、知识陈旧以及推理过程不透明、不可追溯等挑战。检索增强生成（Retrieval-Augmented Generation）...</li><li><a href="https://newvick.com/rag-evolution/">RAG 的演进：解决简单 RAG 系统的常见问题</a>：RAG 并非你所需的一切。本文将涵盖简单 RAG 系统中遇到的一些常见问题及其潜在解决方案。</li><li><a href="https://x.com/llama_index/status/1774832426000515100">来自 LlamaIndex 🦙 (@llama_index) 的推文</a>：这是 @mesudarshan 的一个优秀教程，向您展示如何使用 LlamaParse 以及纯本地的 Embedding、LLM 和重排序（Reranking）模型（@GroqInc 和 @qdrant_en 的 FastEmbed）构建高级 PDF RAG...</li><li><a href="https://x.com/AndrewYNg/status/1773006786058219889">来自 Andrew Ng (@AndrewYNg) 的推文</a>：新的 JavaScript 短期课程：使用 JavaScript 构建一个使用 RAG 的全栈 Web 应用程序。JavaScript RAG Web Apps with LlamaIndex，由 @llama_index 的开发者关系副总裁兼 npm 联合创始人 @seldo 授课...</li><li><a href="https://x.com/llama_index/status/1767687784712814619">来自 LlamaIndex 🦙 (@llama_index) 的推文</a>：使用 LlamaIndex 和 @MathpixApp 将复杂的数学公式解析并索引为 LaTeX，并回答有关科学论文的问题！查看这个详细的 Notebook，MathPix 将带您了解 ➡️ 解析...</li><li><a href="https://x.com/llama_index/statu">来自 LlamaIndex 🦙 (@llama_index) 的推文</a>：

s/1761553473219551301">来自 LlamaIndex 🦙 (@llama_index) 的推文</a>：让我们一起探讨 RAG 的痛点及解决方案！🧑‍🏫🎬 我们很高兴邀请到 @wenqi_glantz 为她广受欢迎的“12 个 RAG 痛点及解决方案”博客文章制作视频演示，该文章是...</li><li><a href="https://www.modular.com/blog/how-to-be-confident-in-your-performance-benchmarking">Modular：如何对你的性能基准测试充满信心</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：如何对你的性能基准测试（Performance Benchmarking）充满信心</li><li><a href="https://partiful.com/e/VJPFposDqQg2eCqHuL38">报名参加实时语音 AI 与多模态黑客松 | Partiful</a>：各位黑客朋友们好，AI Engineer Foundation（你友好的开源非营利邻居 - 网站：aie.foundation）正在举办一场实时交互/对话式多模态 AI 黑客松...</li><li><a href="https://www.daily.co/blog/how-to-talk-to-an-llm-with-your-voice/">如何（用声音）与 LLM 对话</a>：用于构建实时 AI WebRTC 应用程序的代码</li><li><a href="https://langchain-ai.github.io/long-term-memory/">LangMem - LangMem</a>：未找到描述</li><li><a href="https://long-term-memory-shared-for-f208c46599174c09b9b79-vz4y4ooboq-uc.a.run.app'">未找到标题</a>：未找到描述</li><li><a href="https://share.1password.com/s#DPhaOn02m2OD18hu1Ig45a5fPbZxGNKd63VVc37lQtA">我正在使用 1Password 与你分享一个项目</a>：一个密码管理器、数字保险库、表单填充器和安全的数字钱包。1Password 会为你记住所有密码，帮助确保账户信息安全。</li><li><a href="https://www.youtube.com/watch">YouTube</a>：未找到描述</li><li><a href="https://hlfshell.ai/posts/llms-and-robotics-papers-2023/#self-consistency">LLMs + 机器人技术的前沿动态 - 2023</a>：tldr 我写了一些更有趣的作品，这些作品塑造了我对将 LLMs 应用于 AI agents 和机器人应用的理解。简介：什么是 LLMs 狂热 - 一个警告。LLM 是否...</li><li><a href="https://github.com/stanford-cs336/spring2024-assignment1-basics/blob/master/cs336_spring2024_assignment1_basics.pdf">stanford-cs336/spring2024-assignment1-basics 仓库中的 cs336_spring2024_assignment1_basics.pdf</a>：通过在 GitHub 上创建账户，为 stanford-cs336/spring2024-assignment1-basics 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=yiewqC6qNM8">Eugene Cheah - 从创意到 LLM (RWKV / Recursal)</a>：康奈尔科技学院开源生成式 AI 研讨会的演讲。网站：https://github.com/PicoCreator 幻灯片：https://drive.google.com/file/d/1-lfITA0j_9-...</li><li><a href="https://youtu.be/eMlx5fFNoYc">可视化 Attention，Transformer 的核心 | 第 6 章，深度学习</a>：揭秘 Attention，这是 Transformers 和 LLMs 内部的关键机制。这些课程不是通过赞助广告，而是由观众直接资助：https://3...</li><li><a href="https://github.com/go-go-golems/bobatea/blob/main/pkg/chat/README.md">go-go-golems/bobatea 仓库中的 pkg/chat/README.md</a>：自定义 bubbletea bubbles。通过在 GitHub 上创建账户，为 go-go-golems/bobatea 的开发做出贡献。</li><li><a href="https://github.com/go-go-golems/bobatea/blob/main/cmd/chat/backend.go">go-go-golems/bobatea 仓库中的 cmd/chat/backend.go</a>：自定义 bubbletea bubbles。通过在 GitHub 上创建账户，为 go-go-golems/bobatea 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1225854717213544498)** (8 条消息🔥): 

- **宣布 Latent Space University 的首门课程**：_Latent Space University_ 将于太平洋时间下午 1 点举行免费介绍会，开启其首门 AI Engineering 课程。报名及详情请访问 [Maven Learning, Inc.](https://maven.com/p/245c45)。

- **活动冲突**：由于课程介绍与 **Latent Space Discord** 活动时间重叠，对此进行了轻松的回应。

- **在三周内扩展你的 AI 专业知识**：一门新课程承诺在三周内全面讲解 AI 模态，涵盖 **OpenAI API、Retrieval Augmented Generation、Code Generation、Image Generation** 和 **Speech-to-Text** 功能。如 [课程概览](https://maven.com/noah-hein/ai-engineering-intro) 中所述，使用代码 "lightning" 可获得折扣。

- **周末播客预告**：周末发布了新的播客剧集，公告已通过 [Twitter 链接](https://twitter.com/swyx/status/1776687540520767544) 分享。

- **Latent Space Podcast 周末特辑**：该播客涵盖了包括 **AI UX、世界博览会** 以及最新的 AI 技术和领导力在内的各种话题。深入探讨 AI Engineering 趋势等内容，详见 [Latent Space](https://www.latent.space/p/weekend-special-5-chats) 总结的 _Weekend Special_ 剧集。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://maven.com/noah-hein/ai-engineering-intro">Level Up From Software Engineer to AI Engineer by Shawn &quot;Swyx&quot; Wang and Noah Hein on Maven</a>：从入门到精通：学习如何构建真实的 AI 产品</li><li><a href="https://maven.com/p/245c45">Code a custom ChatGPT</a>：这是 AI 产品的基石。如果你想成为一名 AI Engineer，这些是必修的主题和 API。从 ChatGPT 到强大的 AI 驱动的摘要和分类都使用了...</li><li><a href="https://www.latent.space/p/weekend-special-5-chats">Latent Space Chats: NLW (Four Wars, GPT5), Josh Albrecht/Ali Rohde (TNAI), Dylan Patel/Semianalysis (Groq), Milind Naphade (Nvidia GTC), Personal AI (ft. Harrison Chase — LangFriend/LangMem)</a>：5 个最近的 Latent Space 访谈，让你全面了解所有能想象到的 AI Engineering 话题。
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1225897703175622791)** (57 条消息🔥🔥): 

- **无缝知识获取**：讨论强调了将 Slack 等聊天应用作为知识库的使用，认为这是从人与人之间的互动中捕获和合成有用信息工件的“高级操作”。
- **优化文本工作**：一位成员强调了通过结构化文档（而非依赖 Slack）来减轻认知负荷的吸引力和可行性。尽管 Slack 是许多公司“行动发生”的地方，但它被指出是一个“糟糕的记录系统”。
- **丰富的工具与集成**：聊天中出现了几个用于通过 AI 增强个人知识库和工作空间的资源，包括 [Obsidian-Copilot](https://github.com/logancyang/obsidian-copilot) 和用于增强人类表现的 [fabric](https://github.com/danielmiessler/fabric)，以及使用 [Obsidian's CLI tools](https://github.com/Yakitrak/obsidian-cli) 的建议。
- **构建更好的桥梁**：讨论了将 ChatGPT 等 AI 工具集成到 Obsidian 等个人知识系统中的持续探索，重点关注现有插件以及开发新插件的潜力。
- **协作贡献**：聊天以对分享有用想法和资源的感谢结束，表明了对讨论期间提供的见解和建议的集体赞赏。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/logancyang/obsidian-copilot">GitHub - logancyang/obsidian-copilot: A ChatGPT Copilot in Obsidian</a>：Obsidian 中的 ChatGPT Copilot。通过在 GitHub 上创建账号来为 logancyang/obsidian-copilot 的开发做出贡献。</li><li><a href="https://github.com/Yakitrak/obsidian-cli">GitHub - Yakitrak/obsidian-cli: Interact with Obsidian in the terminal. Open, search, create, update, move and delete notes!</a>：在终端中与 Obsidian 交互。打开、搜索、创建、更新、移动和删除笔记！- Yakitrak/obsidian-cli</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>：2024 主题、日期、主持人、资源、GenAI 的 UI/UX 模式，1/26/2024，nuvic，&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://github.com/danielmiessler/fabric">GitHub - danielmiessler/fabric: fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere.</a>：fabric 是一个用于利用 AI 增强人类能力的开源框架。它提供了一个模块化框架，通过一组可随处使用的众包 AI prompts 来解决特定问题。- ...
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1225778122549624832)** (53 条消息🔥): 

- **AWQ 模型在 Hugging Face Inference 上运行正常**：一位成员提到在 Hugging Face 的推理服务上成功运行了 **awq 模型**。
- **注意到 GitHub 的异常现象**：有报告称 **GitHub search** 会重定向到一个带有详细描述和图像的特定页面，怀疑是自动重定向问题。

- **对新 Qwen 模型的关注**：讨论集中在最新的 **Qwen 模型**，特别是对其 **32B** 版本的兴趣。进一步的讨论建议，**Yi 34B** 和 **Command R** 也是在同一 **fine-tune dataset** 上进行对比的热门模型。
- **使用 Ring Attention 进行上下文长度训练**：一名成员关注了一个名为 [EasyContext by jzhang38](https://github.com/jzhang38/EasyContext) 的 GitHub 仓库，该仓库概述了使用 Ring Attention 在 8xA100 GPU 上将 LM 上下文长度外推至 100 万个 token 的内存优化和训练方案。随之而来的是作者发布的一条 [Twitter 线程](https://twitter.com/PY_Z001/status/1776176932687892796)，讨论了随着上下文尺寸增加，训练吞吐量（training throughput）下降的问题。
- **GitHub 上的 Schedule-Free 优化**：发布了 **Schedule-Free Optimization in PyTorch** 仓库的介绍，推测是为了突出一种改进优化过程的工具。
- **ORPO 结构的编码挑战**：一名成员详细描述了在尝试实现新的 prompt template 时遇到的 ORPO 结构难题，并遇到了 **micro_batch_size** 的问题。Caseus_ 在随后的回复中确认了 **ORPO 和 batch sizes** 持续存在的问题。
- **Generative AI 峰会的考量与反思**：一名成员权衡了参加巴黎 Generative AI 峰会的利弊，思考了参会对拓展人脉（networking）的价值。他们后来确认参加了会议，并提到在峰会活动中获得了第二名，但没有进行太多的社交。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program">Introducing improvements to the fine-tuning API and expanding our custom models program</a>：我们正在添加新功能，以帮助开发者更好地控制 fine-tuning，并宣布了使用 OpenAI 构建自定义模型的新方法。</li><li><a href="https://www.raisesummit.com/#bl-59ec20d5-ce0f-4f84-971d-543b5c7efa9b>">R.AI.SE Summit</a>：RAISE 是一个由 1,500 多名全球领导者参加的聚会，致力于探索 Generative AI 对商业和社会的变革力量。会议将于 2024 年 4 月 8 日在巴黎举行。</li><li><a href="https://www.lepton.ai/">Build AI The Simple Way | Lepton AI</a>：使用云原生平台，在几分钟内高效、大规模地运行 AI 应用程序。</li><li><a href="https://github.com/search">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/facebookresearch/schedule_free">GitHub - facebookresearch/schedule_free: Schedule-Free Optimization in PyTorch</a>：PyTorch 中的 Schedule-Free 优化。通过在 GitHub 上创建账号来为 facebookresearch/schedule_free 的开发做出贡献。</li><li><a href="https://github.com/jzhang38/EasyContext">GitHub - jzhang38/EasyContext: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware.</a>：内存优化和训练方案，旨在以最少的硬件将语言模型的上下文长度外推至 100 万个 token。- jzhang38/EasyContext</li><li><a href="https://github.com/huggingface/transformers/pull/30005">Add JetMoE model by yikangshen · Pull Request #30005 · huggingface/transformers</a>：此 PR 做了什么？添加了对 Yikang Shen 和 MyShell AI 开发的 JetMoE 架构的支持。JetMoE 是一种受 ModuleFormer 启发的新型稀疏激活架构。每个 JetMoE 块由...组成。</li><li><a href="https://www.raisesummit.com/#bl-59ec20d5-ce0f-4f84-971d-543b5c7">R.AI.SE Summit</a>：RAISE 是一个由 1,500 多名全球领导者参加的聚会，致力于探索 Generative AI 对商业和社会的变革力量。会议将于 2024 年 4 月 8 日在巴黎举行。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1225740998353752117)** (19 条消息🔥): 

- **支持量化 DoRA**：`peft=0.10.0` 现在支持量化 DoRA，如 [PEFT 最新发布说明](https://github.com/huggingface/peft/releases/tag/v0.10.0)所示。这一变化可能需要更新 **axolotl** 的 `requirements.txt` 文件。

- **介绍 Schedule-Free Learning**：Facebook Research [开源了 Schedule-Free Learning](https://github.com/facebookresearch/schedule_free)，这是一种用平均和插值代替动量（momentum）的方法，消除了对传统学习率调度（learning rate schedules）的需求。

- **新优化器需要代码更改**：开发者应注意，新的 Schedule-Free 优化器需要额外的 `optimizer.train()` 和 `optimizer.eval()` 调用；这在优化器的仓库中有所强调。

- **ScheduleFreeAdamW 参数调优建议**：为了在使用 ScheduleFreeAdamW 时获得最佳性能，建议将该值设为 `0.0025`，开发者应查阅 caveats 部分以获取有关其他可调参数的指导。

- **对 Hugging Face Transformers 的上游贡献**：对 adamw schedulefree 的支持已[合并至 Hugging Face 的 transformers 库](https://github.com/huggingface/transformers/pull/30079)，这简化了与 DeepSpeed 或 PyTorch FSDP 配置的集成。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/aaron_defazio/status/1776341914364641583?s=46&t=hIokEbug9Pr72tQFuXVULA">来自 Aaron Defazio (@aaron_defazio) 的推文</a>：@divideconcept 它需要那样调优。请参阅 caveats 部分的注释以获取建议的范围。0.0025 这个值对于 ScheduleFreeAdamW 似乎工作得非常可靠。</li><li><a href="https://x.com/aaron_defazio/status/1776320004465582331?s=46&t=hIokEbug9Pr72tQFuXVULA">来自 Aaron Defazio (@aaron_defazio) 的推文</a>：Schedule-Free Learning https://github.com/facebookresearch/schedule_free 我们现在已经开源了我那一系列神秘图表背后的算法。每张图表要么是 Schedule-free SGD，要么是 Adam，没有...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1486">由 winglian 添加对 adamw schedulefree 的支持 · Pull Request #1486 · OpenAccess-AI-Collective/axolotl</a>：实现了 Meta 的 https://github.com/facebookresearch/schedule_free 用于 adamw https://twitter.com/aaron_defazio/status/1776320004465582331 optimizer: schedule_free_adamw lr_scheduler: constant</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/config/models/input/v0_4_1/__init__.py#L245>">axolotl/src/axolotl/utils/config/models/input/v0_4_1/__init__.py at main · OpenAccess-AI-Collective/axolotl</a>：尽管提出 axolotl 问题。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/pull/30079">由 winglian 提交的 schedulefree 优化器 · Pull Request #30079 · huggingface/transformers</a>：此 PR 做了什么？集成了 Meta 的 https://github.com/facebookresearch/schedule_free 用于 adamw 和 sgd https://twitter.com/aaron_defazio/status/1776320004465582331 在提交之前...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1225806966551613551)** (5 条消息): 

- **Mistral 模型生成错误**：一位用户报告了在使用 **fp16** 的 *微调 Mistral 7b 模型* 进行生成时出现错误。错误发生在几次成功生成之后，回溯显示为 `_queue.Empty`。

- **推理方法的澄清**：用户澄清他们没有使用内置推理方法，而是利用了 **Hugging Face 的 generate with streamer**。

- **关于使用 Accelerate 库的假设**：另一位成员建议用户可能正在使用 **Accelerate**，但用户予以否认，确认他们使用的是纯 Python。

- **分享出现问题的代码**：面临生成问题的用户分享了他们的代码，其中使用了 **Hugging Face 的 transformers**、Python 的 **threading**，以及一个基于 `StoppingCriteria` 的自定义类 `StopOnTokens` ，并配合 **Gradio 的 ChatInterface** 来部署聊天机器人应用。
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/)** (1 条消息): 

faldore: <@&1166009801583628349> 色情垃圾信息
  

---


**OpenAccess AI Collective (axolotl) ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/1225946216739770421)** (3 条消息): 

- **Config 中缺少 LISA 参数**：一位成员注意到 `axolotl/docs/config.qmd` 中缺少 **LISA 实现** 的参数。
- **在别处找到了 LISA 配置**：随后另一位成员找到并分享了 LISA 参数，并提供了 GitHub 上 [LISA 配置文件](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/lisa.yml) 的链接。
- **关于解冻层（Unfreezing Layers）的疑问**：有人提出了关于在模型训练过程中解冻新层后如何 **处理优化器状态（optimizer states）** 的问题，这引起了社区对实际实现细节的关注。

**提到的链接**：<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-2/lisa.yml">axolotl/examples/llama-2/lisa.yml at main · OpenAccess-AI-Collective/axolotl</a>：尽管提出 axolotl 问题。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1225804060087812166)** (46 条消息🔥):

- **Docker 镜像的 LoRA Adapter 合并错误**：一位用户在尝试合并 LoRA Adapter 时遇到了 [pydantic 验证错误](https://errors.pydantic.dev/2.6/v/value_error)，该错误要求在启用 `sample_packing` 时，必须将 `flash_attention` 或 `sdp_attention` 设置为 true。

- **在 Mistral 模型上使用原始文本进行训练**：针对使用原始文本训练 Mistral 模型的问题，一位成员分享了一个 YAML [配置示例](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c12d621b-5026-4bc0-af60-66a9b40e9708)，详细说明了模型和 tokenizer 规范、数据集路径以及训练参数。

- **适配 Alpaca 指令集进行微调**：在使用 Alpaca 指令集和 ChatML 格式进行微调时，有用户建议将数据集转换为 ShareGPT 格式，并利用 `conversation: chatml` 进行配置，这解决了数据集混合的疑虑。

- **Micro Batch 与 Batch Size**：澄清了 Micro Batch Size 与 Batch Size 之间的区别：Micro Batch Size 允许高效利用内存，并在不增加计算成本的情况下模拟更大的 Batch Size；而 Batch Size 则是在处理完整个批次的数据后更新一次模型的权重。

- **禁用 Checkpoints 和评估阶段的配置**：用户讨论了如何修改配置文件以实现从不保存 Checkpoints（通过将 `saves_per_epoch` 更改为 `0`），并询问了如何完全禁用评估阶段，建议将 `evaluation_strategy` 设置为 `EvaluationStrategy.NO`。

- **配置文件中未定义 Special Tokens 的处理**：明确了当 `examples/mistral/qlora.yml` 等配置文件中未定义 Special Tokens 时，除非被明确覆盖，否则将使用基于基础模型和 tokenizer 的默认值。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://errors.pydantic.dev/2.6/v/value_error">Redirecting...</a>：未找到描述</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=f3156bb0-3cb9-4c34-b7d8-7cb4618a499d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8cc85b97-df87-499b-a134-50674538d2f4)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=3a24e145-a395-4639-b2a6-100b531e959b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=39846e14-89a0-4353-a806-cc1e3136c78a)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c12d621b-5026-4bc0-af60-66a9b40e9708)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=2cc617a2-4788-4f73-a29b-6d622e452e3b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c371c38b-42d0-4b01-b381-55fd5f1d093f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=2dc474d5-a5e8-441b-bc59-17e2571f2781)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1225744640372899841)** (22 条消息🔥): 

- **澄清 "rope_theta"**：一位成员询问了 `"rope_theta": 10000.0` 的含义，这指的是 Rotary Positional Embedding 技术中的一个参数，用于向 Transformer 模型引入位置信息。

- **Mistral 的 FSDP 配置公开**：Mistral 的 Fully Sharded Data Parallel (FSDP) 配置在 `mixtral-qlora-fsdp.yml` 文件中指定，表明应包装 `MixtralSparseMoeBlock` 类以进行分片。对话中包含了一个[指向配置文件的链接](https://github.com/openaccess-ai-collective/axolotl/tree/main/examples/mistral/mixtral-qlora-fsdp.yml#L1L75)。

- **LISA Layer 仍未定义**：出现了一个关于什么构成 "lisa layer" 的查询，但截至最新的知识更新，该术语并不对应 AI 和机器学习社区中广泛认可的概念。

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=2150270f-2213-4881-b572-a8c9dab49c46)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=35cacb5b-24d0-43ce-8a22-8eb2ab861118)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=3246eab0-a12a-4f23-ac87-0cb50c2fccf2)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/examples/mistral/mixtral-qlora-fsdp.yml#L1L75)">axolotl/examples/mistral/mixtral-qlora-fsdp.yml at main · OpenAccess-AI-Collective/axolotl</a>：尽管提出 axolotl 问题。通过在 GitHub 上创建账户，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=d540907f-286f-4152-8935-2370919b6441)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1225867580506374154)** (15 条消息🔥): 

- **与 John Schulman 合作播客的潜力**：频道主持人 Nathan 思考了邀请 John Schulman 进行采访的激动人心的想法，并承认之前不知为何把这件事给忘了。
- **对“重磅”采访的期待**：采访 John Schulman 的建议得到了热烈响应，另一位成员也认为这肯定会引起轰动。
- **探索 AI 音乐新领域**：分享了一个推文链接，暗示了 Suno AI 平台的竞争对手，Nathan 觉得它“真的非常出色”。
- **放开许可证限制**：@julien_c 的一条推文宣布了一项显著的许可变更，将 **text-generation-inference (TGI)** 从自定义的 HFOIL 许可证切换回 Apache 2，使该库完全开源。Nathan 对 Hugging Face 的透明度及其承担的风险发表了评论。
- **许可证变更后贡献者涌入**：开源 TGI 的决定导致贡献者增加了三倍，尽管最初受欢迎程度较低，但在修改许可证后，该项目获得了很大的关注。Nathan 随后的消息表达了对这一进展的兴奋，使用了“rip”、“NOW WE'RE ECLIPSING”和“LFG”等词语，暗示了积极的发展势头。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/robdadashi/status/1777317222496526663?s=46">Robert Dadashi (@robdadashi) 的推文</a>：训练数据与 v1.0 基本相同，但我们将 RL 算法切换为了一些新东西。我希望将来能披露更多相关信息 :)。6/11</li><li><a href="https://fxtwitter.com/julien_c/status/1777328456709062848">Julien Chaumond (@julien_c) 的推文</a>：我们决定更新 text-generation-inference (TGI) 的许可证。我们将许可证从 HFOIL（我们的自定义许可证）切回到 Apache 2，从而使该库完全开源。阅读下文...</li><li><a href="https://fxtwitter.com/legit4k/status/1777059367788982389?s=46">ʟᴇɢɪᴛ (@legit_rumors) 的推文</a>：这是独家的全新 udio AI 音乐生成 🫡 来源：匿名。匿名总是来源。
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1226275316482441246)** (9 条消息🔥): 

- **有人又被针对了……再次！**：一位成员提到再次被针对，但未提供具体细节或背景。
- **改进近在眼前？**：另一位成员回应说，他们最近的经历有所改善，暗示无论之前的“糟糕”情况是什么，现在都没那么严重了。
- **是建议还是只是在玩梗？**：一位成员针对之前的评论建议“使用 Code Interpreter”，似乎是玩笑的延续。
- **纯属娱乐**：最初的请求者澄清说该请求只是一个梗，并感谢了小组，表示实际上不需要建议。
- **开玩笑的！就业状态已确认**：关于失业的暗示被澄清为一个玩笑，该成员确认目前已就业且状态良好。
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1226374380981063690)** (55 条消息🔥🔥):

- **辩论开放模型权重 (Open Model Weights) 的风险与必然性**：讨论集中在开放基础模型 (Open Foundation Models) 的社会影响，涉及是否需要为其发布设定安全阈值。对于强制执行 AI 技术不扩散的实用性和可行性，以及监管其分发是否具有伦理责任，参与者表达了多种观点。
  
- **探讨 AI 的权力动态**：成员们谈论了语言模型操纵社会和民主进程的潜力。共识似乎集中在 AI 进步的必然性、构建语言模型的简便性及其可访问性上，这导致了一种某种程度的无奈感，即严格的监管可能并不实际。
  
- **瓶中妖灵 (The Genie Out of the Bottle)**：一位成员用类比讨论了对强大 AI 的控制，将不受限制的 AI 比作可能产生不良社会影响的个人“妖灵”。对于使用限制的实际执行存在怀疑，并将其与核不扩散面临的挑战进行了对比。
  
- **开放 AI 研究中的规模与可访问性**：会议指出训练大模型的计算成本不断增加的趋势，暗示这可能会超过通用硬件的能力，并限制社区/学术界的访问。会议强调了目前模型推理 (Inference) 通过 API 比在个人硬件上运行更具成本效益的现状。
  
- **未来模型：生成 vs. 验证**：在讨论结束时，提出了专注于验证而非生成的开放模型概念。有人表示好奇，这种方法是否能通过将验证者的知识转移到推理阶段，使模型更易于访问，并绕过对大规模模型的需求。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/modeless/status/1776693329432088718>">James Darpinian (@modeless) 的推文</a>：@TheXeophon 你是否见过这样一种理论，即拼写错误是故意用来过滤掉聪明人的</li><li><a href="https://open.substack.com/pub/aisnakeoil/p/on-the-societal-impact-of-open-foundation">论开放基础模型的社会影响</a>：为 AI 开放性的辩论增加精确度</li><li><a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">可视化 Attention，Transformer 的核心 | 第 6 章，深度学习</a>：揭秘 Attention，这是 Transformer 和 LLM 内部的关键机制。这些课程不是通过赞助广告，而是由观众直接资助的：https://3...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1225849170326327359)** (31 条消息🔥): 

- **搜寻顶级微调 (Fine Tunes) 模型**：围绕利用 [lmsys](https://lmsys.deepai.org/) 和 [alpacaeval 排行榜](https://alpacaeval.com/) 作为资源来发现有效的微调模型展开讨论。这些平台被强调为寻找达到 SOTA 性能模型的良好起点。
- **对 OpenWeights 的认可**：在寻找模型的背景下，提到了 DeepSeek 的 **OpenWeights** 模型作为一个潜在来源。
- **转向视觉辅助工具**：承诺在即将到来的演讲中通过视觉方式对模型进行澄清和分类；这将包括在屏幕上对特定模型进行淡化、高亮和放大等策略，以便更好地理解。
- **历史讲座的实时文档**：分享了一个 [Google Slides 演示文稿](https://docs.google.com/presentation/d/1quMyI4BAx4rvcDfk8jjv063bmHg4RxZd9mhQloXpMn0/edit?usp=sharing) 的链接，这似乎是即将举行的关于“对齐开放语言模型”讲座的初稿。
- **开放模型指南**：Xeophon 对 Salesforce 的 CodeGen 系列进行了详细阐述，包括发布时间线、使用的数据集、许可，以及一个汇集了各种模型及其属性的[综合电子表格](https://docs.google.com/spreadsheets/d/1gc6yse74XCwBx028HV_cvdxwXkmXejVjkO-Mz2uwE0k/edit#gid=0)。该资源旨在为研究开放模型的人员节省时间。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com/presentation/d/1quMyI4BAx4rvcDfk8jjv063bmHg4RxZd9mhQloXpMn0/edit?usp=sharing">[2024年4月18日] 对齐开放语言模型</a>：对齐开放语言模型 Nathan Lambert Stanford CS25: Transformers United V4 1</li><li><a href="https://docs.google.com/spreadsheets/d/1gc6yse74XCwBx028HV_cvdxwXkmXejVjkO-Mz2uwE0k/edit#gid=0>">生成式 AI 目录</a>：预训练 LLM 名称、日期、参数（激活）、参数（总计）、组织、组织类型、作者位置、语言、商业用途、模型可访问性、代码可访问性、数据可访问性、主要...
</li>
</ul>

</div>

**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1225980876677189633)** (15 条消息🔥): 

- **Fast Tokenizers：对速度的需求**：一位成员讨论了使用 Huggingface 的 fast tokenizer 处理 *c4-en* 数据集时分词过程缓慢的问题，并询问了加速选项（如增加线程）。另一位成员建议尝试拥有更多线程的机器。
  
- **AMD 的开源飞跃**：AMD 宣布将开源其 Micro Engine Scheduler (MES) 固件，并提供相关文档和针对 Radeon GPU 的 GitHub 追踪器。这一消息符合 AMD 推动其 GPU 技术更加开源的广泛努力，受到了包括 George Hotz 的 Tiny Corp 在内的社区欢迎。[The Register 文章](https://www.theregister.com/2024/04/05/amd_mes_open_source/)，[AMD Radeon 推文](https://twitter.com/amdradeon/status/1775999856420536532)。

- **论文复现仓库发布**：一个致力于复现 AI 和 ML 领域研究论文的开源仓库正式发布。他们鼓励社区参与贡献并在 GitHub 仓库上点亮 star。[PaperReplica GitHub 仓库](https://github.com/hegdeadithyak/PaperReplica)。
  
- **寻求 CUDA 设置指南**：分享了一份在 Ubuntu 上设置 CUDA 开发环境的新指南，详细介绍了 CUDA Toolkit、驱动程序、CuDNN 以及支持 CUDA 的 OpenCV 的安装。对话还引发了关于在不同系统上使用 CUDA 开发的舒适度的讨论。[Setup-as-Cuda-programmers GitHub 仓库](https://github.com/CisMine/Setup-as-Cuda-programmers)。
  
- **针对 LM 的创新序列并行**：EasyContext 通过 ring attention 引入了序列并行，旨在将语言模型的上下文长度扩展到 100 万个 token，同时优化内存使用。该 GitHub 仓库提供了在最低硬件需求下实现此扩展的训练方案。[EasyContext GitHub 仓库](https://github.com/jzhang38/EasyContext.git)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discordapp.com/channels/1189498204333543425/1189640399476764692/1226242857501720596">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://x.com/amdradeon/status/1775999856420536532">来自 AMD Radeon (@amdradeon) 的推文</a>：我们正致力于在 5 月底发布 Micro-Engine Scheduler (MES) 文档，随后将发布源代码以供外部审查和反馈。我们还开设了一个 GitHub 追踪器...</li><li><a href="https://discordapp.com/channels/1189498204333543425/11">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://www.theregister.com/2024/04/05/amd_mes_open_source/">AMD 将为 Radeon GPU 开源 MES 固件</a>：这一切都要归功于同行的压力</li><li><a href="https://github.com/hegdeadithyak/PaperReplica">GitHub - hegdeadithyak/PaperReplica：我们复现 AI 和 ML 领域的科研论文。</a>：我们复现 AI 和 ML 领域的科研论文。 - hegdeadithyak/PaperReplica</li><li><a href="https://github.com/jzhang38/EasyContext.git">GitHub - jzhang38/EasyContext：内存优化和训练方案，旨在以极低硬件要求将语言模型的上下文长度外推至 100 万个 token。</a>：内存优化和训练方案，旨在以极低硬件要求将语言模型的上下文长度外推至 100 万个 token。 - jzhang38/EasyContext</li><li><a href="https://github.com/CisMine/Setup-as-Cuda-programmers">GitHub - CisMine/Setup-as-Cuda-programmers</a>：通过在 GitHub 上创建账号来为 CisMine/Setup-as-Cuda-programmers 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1225935229366046801)** (7 条消息):

- **库的赞赏引起共鸣**：一位成员对一个增强了他们对 **Triton** 理解并提高了调试 Triton kernel 能力的库表示感谢。
- **寻求 Autotune 知识**：**@ryanatseattle** 询问了在 Triton 中自动调整（autotune）参数（如 `num_wrap`、`num_stage` 和 `GROUP_SIZE`）的有效方法，并提到现有的 `triton.autotune` 功能似乎只提供随机配置。
- **Auto-tune 与 Benchmarking 的两难选择**：有人询问如何最好地将 auto-tuning 与 benchmarking 结合起来，询问是否应该先使用 auto-tune 确定最佳配置，然后再进行 benchmarking。
- **性能对决 - 自定义 CUDA vs Triton**：有人提出了关于 PyTorch 中自定义 CUDA kernel 与 Triton kernel 之间效率对比的疑问。
- **关于点积与加法的评论**：**@mobicham** 发表了一个隐晦的评论，暗示了在编码或算法优化背景下，对使用点积而非加法的偏好或观察。

---

**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1226988886484324473)** (1 messages): 

- **GPT-2 在 C 语言中变得精简高效**：一位成员分享了 [Andrej Karpathy 最近的 GitHub 项目](https://github.com/karpathy/llm.c) llm.c，该项目可以使用纯 C 语言训练 GPT-2，无需 PyTorch 和 Python 的沉重依赖。这个轻量级实现可以立即编译并运行，代码简洁，仅约 1,000 行，是训练者的福音。

**提及的链接**：<a href="https://x.com/karpathy/status/1777427944971083809?s=46&t=ej2aClHUAjeapC55UGHfwg">来自 Andrej Karpathy (@karpathy) 的推文</a>：你是否曾经想过在没有 245MB 的 PyTorch 和 107MB 的 cPython 的情况下，用纯 C 语言训练 LLM？不想？好吧，现在你可以了！使用 llm.c：https://github.com/karpathy/llm.c 首先，实现了 GPT-2 在... 上的训练。

---

**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1226075591309262859)** (14 messages🔥): 

- **Torch compile 与异步操作的困境**：一位成员提出了在使用 **torch compile** 配合异步集合操作（collective operations）时遇到的困难，并被引导关注可能支持 torch compile 的新实验性 *functional_collectives*。然而，这些似乎并不支持像 [async all reduce](https://github.com/pytorch/pytorch/blob/eff1e4899c7c89f8a8fc8f6ff6bed06dd8d2ec8a/torch/distributed/_functional_collectives.py#L169) 这样的异步操作。

- **异步集合操作揭秘**：在后续讨论中，澄清了新的 *functional_collectives* 确实是异步的，并利用 tensor subclassing 魔法来自动同步，或者用户也可以调用 `.wait()` 进行显式同步。

- **DeepSpeed 与 Accelerate 集成经验**：另一位成员询问了 **DeepSpeed** 与 Hugging Face 的 **Accelerate** 的集成情况，重点关注是否会丢失混合专家模型（MoE）等功能。建议是丢失的功能非常少，但应该手动定义 deepspeed 配置 JSON 文件，而不是依赖 HF trainer 设置。

- **揭开 DeepSpeed 显存占用之谜**：观察到将 **zero stage** 设置为 0（理应禁用它）时，与 Distributed Data Parallel (DDP) 相比，显存消耗仍然较少，这表明 DeepSpeed 可能在不知不觉中运行了某些优化。

- **破译 torch compile 中 Triton 的利用方式**：讨论 **torch compile** 的成员强调，只有当输入在 CUDA 设备上时才会使用 **Triton** kernel，否则在 CPU 上会使用 C++ 生成的代码来处理融合 kernel。

- **MLP 的性能优化**：一位热衷于优化其 Transformer 模型 MLP 的成员分享说，来自 flash attention 库的基于 cublas 函数的 MLP 并不比使用 torch functionals 的简单 MLP 更快。他们收到了进一步探索优化的建议，如果融合操作的性能没有超过矩阵乘法库的实现，可以考虑使用 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md) 或 CUTLASS 而非 Triton。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md">tiny-cuda-nn/DOCUMENTATION.md at master · NVlabs/tiny-cuda-nn</a>: 极速 C++/CUDA 神经网络框架。通过在 GitHub 上创建账号来为 NVlabs/tiny-cuda-nn 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/blob/014f91a9d9f94ac9a7f0711600240d7cd7f69844/torch/_dynamo/variables/functions.py#L704,">pytorch/torch/_dynamo/variables/functions.py at 014f91a9d9f94ac9a7f0711600240d7cd7f69844 · pytorch/pytorch</a>: Python 中具有强 GPU 加速能力的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/eff1e4899c7c89f8a8fc8f6ff6bed06dd8d2ec8a/torch/distributed/_functional_collectives.py#L169">pytorch/torch/distributed/_functional_collectives.py at eff1e4899c7c89f8a8fc8f6ff6bed06dd8d2ec8a · pytorch/pytorch</a>: Python 中具有强 GPU 加速能力的 Tensor 和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1226242857501720596)** (2 条消息): 

- **CUDA-MODE 系列讲座继续**：CUDA-MODE 系列讲座的下一部分 *第 13 讲：Ring Attention* 计划在 [公告时间](<t:1712430000:t>) 开始，由受人尊敬的 <@719599526448463933> 主讲。

- **庆祝社区蓬勃发展**：CUDA-MODE Discord 社区成员已超过 5,000 名，庆祝其成长并向成员表示感谢。自成立以来坚持 **每周举办一次讲座** 被强调为成功的关键。 

- **学以致用**：讲座激发了成员在现实世界中应用知识，为社区内许多活跃的工作组做出了贡献。这些实践努力在特定频道的讨论和协作中得到了证实。

- **邀请加入 CUDA-MODE 大家庭**：鼓励社区成员邀请注重性能的朋友加入 CUDA-MODE 冒险并共同学习。邀请链接：[discord.gg/cudamode](https://discord.gg/cudamode)。
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1225857072311046219)** (7 条消息): 

- **模型量化革命**：[QuaRot](https://arxiv.org/abs/2404.00456) 是一种新的量化方案，允许对大语言模型 (LLMs) 进行 4-bit 端到端量化。它独特地处理了离群值并保持了计算不变性，其 LLaMa2-70B 模型仅产生了 0.29 的 WikiText-2 困惑度 (perplexity) 损失，并保留了 99% 的 zero-shot 性能。

- **4-bit 量化的挑战**：有观察指出，虽然 QuaRot 是一个很有前景的进展，但与典型的 4-bit 量化不同，它需要训练/校准才能获得有效性能。

- **优化中不再需要调度 (Scheduling)**：一位成员关注了 [PyTorch 中的 Schedule-Free 优化](https://github.com/facebookresearch/schedule_free)，这是 Facebook Research 的一个仓库，提出了一种利用 schedule-free SGD 或 Adam 的新方法。

- **Twitter 上的优化器深度探讨**：分享了 Aaron Defazio 的一条 [Twitter 帖子](https://twitter.com/aaron_defazio/status/1773381393831067787) 链接，可能为之前讨论的 schedule-free 优化技术提供见解。

- **以新视角审视 Llama**：简短提到暗示了 schedule-free 优化与类似 "Llama3" 的事物之间可能存在联系，暗示了其重要性。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.00456">QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs</a>: 我们引入了 QuaRot，一种基于旋转的新型量化方案，能够对 LLMs 进行端到端量化，包括 4-bit 的所有权重、激活和 KV cache。QuaRot 以某种方式旋转 LLMs...</li><li><a href="https://github.com/facebookresearch/schedule_free">GitHub - facebookresearch/schedule_free: Schedule-Free Optimization in PyTorch</a>: PyTorch 中的 Schedule-Free 优化。通过在 GitHub 上创建账号来为 facebookresearch/schedule_free 的开发做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1225793515984785538)** (1 条消息): 

- **并行算法爱好者的经典资源**：一位成员提到了他们在 2013 年撰写论文时使用的 [Udacity 课程](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2)。该课程不仅涵盖硬件和编程，还专注于 **并行算法和性能**。
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1225993633631698984)** (1 条消息):

- **Nsight Compute 安装问题**：一位成员在使用 `.run` 文件在 **Ubuntu 22.04** 上安装 **Nsight Compute** 时遇到问题；尽管遵循了安装步骤（包括 `chmod +x`），程序在执行后并未出现。尝试重新运行 `./nsight compute` 命令导致程序再次解压。
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 条消息): 

itali4no: https://youtu.be/ws7angQYIxI?si=PcRy7siLQuFywpgp
  

---


**CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1225778502385537037)** (1 条消息): 

- **将 Triton Puzzles 移植到 Pallas**：有人对将 **Triton puzzles** 移植到 Pallas 感兴趣，并建议对于愿意研究其可能性的用户，可以通过 **Triton backend** 来实现。
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1225927315758776403)** (8 条消息🔥): 

- **澄清 Linear Attention**：讨论简要涉及了 **linear attention** 的本质，确认它与经典的 attention 不同。
- **分享 Ring-Flash-Attention 脚本**：分享了一个由 *jzhang38* 编写的包含 **ring-flash-attention** 的训练脚本 GitHub 链接。该项目旨在进行上下文长度外推（context length extrapolation），并建议将其包含在 ring-attention 仓库的 readme 中。[GitHub 上的 EasyContext](https://github.com/jzhang38/EasyContext)。
- **探索 Context Parallelism**：分享了一个 NVIDIA 文档链接，说明了 **Context Parallelism** 的概念，强调了它与 sequence parallelism 的区别及其对 Transformer 模型的影响。[NVIDIA Context Parallelism 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html)。
- **Variable Length Striped Attention 的进展**：一位参与者提到他们正在致力于实现 **varlen striped attention**，但未提供关于进展或影响的进一步背景。
- **质疑 Ring Attention 的内存占用**：有人提出了关于 **ring attention** 在速度和内存占用之间权衡的问题，特别是在分布式计算和消息传递系统中的缓冲过程背景下。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html">Context parallelism 概览 - NVIDIA Docs</a>: 未找到描述</li><li><a href="https://github.com/jzhang38/EasyContext">GitHub - jzhang38/EasyContext: 内存优化和训练配方，旨在以极低的硬件需求将语言模型的上下文长度外推至 100 万 token。</a>: 内存优化和训练配方，旨在以极低的硬件需求将语言模型的上下文长度外推至 100 万 token。 - jzhang38/EasyContext
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1226221551347568741)** (3 条消息): 

- **澄清 Ring Attention 会议时区**：有人询问关于 ring attention 会议的时区，寻求确认是否为 PDT。

- **GPU 术语中的命名规范**：一位成员表达了这样的观点：对于 GPU kernels 来说，“kernels” 可能不是最合适的术语，并表示虽然现在更改可能为时已晚，但询问其他人是否也有同感。
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1225892929810333799)** (20 条消息🔥):

- **量化张量不进行舍入**：Group quantization 不涉及对 scale 进行舍入，且仅支持使用 `w.reshape(-1, groupsize)` 格式进行重塑，以计算 scale 和 zero point。
- **确保量化与反量化的一致性**：为了验证使用所提供方法的量化和反量化的准确性，可以使用 `gpt-fast.quantize.group_quantize_tensor` 对 `W_hat` 进行量化，然后进行反量化，并与原始的 `W_hat` 比较绝对误差之和。
- **量化方法对齐**：澄清双方似乎都在采用 *int4 affine, groupwise* 量化，尽管可能沿着不同的轴，因此建议的方法应该是兼容的。
- **探索 Triton 以提升性能**：初步实验表明，在某些矩阵上，使用 Triton 解包 4-bit 张量的速度比 PyTorch 快了 62 倍，这表明进一步的优化可能会带来更高的性能。
- **gpt-fast 上的量化集成与测试**：gpt-fast 中针对 HQQ 4bit 量化的更新显示出极具前景的 token 生成速度，特别是在启用 `--compile` 标志时，达到了每秒 200 个 token。量化时间和推理速度似乎与当前的 baseline 一致。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L83">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/llama2_benchmark/eval_model.py#L12">hqq/examples/llama2_benchmark/eval_model.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/zhxchen17/gpt-fast/commit/f94584359076dd484acf28119ec49ffc30ce87f1">HQQ 4 bit llama 2 4b · zhxchen17/gpt-fast@f945843</a>: export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf scripts/prepare.sh $MODEL_REPO python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-hqq --groupsize 64 python generate....</li><li><a href="https://github.com/pytorch-labs/gpt-fast?tab=readme-ov-file#evaluation.">GitHub - pytorch-labs/gpt-fast: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python.</a>: 在少于 1000 行 Python 代码中实现简单且高效的 PyTorch 原生 Transformer 文本生成。 - pytorch-labs/gpt-fast</li><li><a href="https://github.com/zhxchen17/gpt-fast/blob/f94584359076dd484acf28119ec49ffc30ce87f1/quantize.py#L455">gpt-fast/quantize.py at f94584359076dd484acf28119ec49ffc30ce87f1 · zhxchen17/gpt-fast</a>: 在少于 1000 行 Python 代码中实现简单且高效的 PyTorch 原生 Transformer 文本生成。 - zhxchen17/gpt-fast</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L832-L837">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L131">gpt-fast/model.py at main · pytorch-labs/gpt-fast</a>: 在少于 1000 行 Python 代码中实现简单且高效的 PyTorch 原生 Transformer 文本生成。 - pytorch-labs/gpt-fast</li><li><a href="https://gist.github.com/mobicham/84ed1809c9c2f56c5c01fbcdbe22391f">eval_model_wikitext_gptfast.py</a>: GitHub Gist: 立即分享代码、笔记和片段。</li><li><a href="https://github.com/pytorch/pytorch/pull/106516/files#diff-b5f9afc0719fb33b38ccac5f6d4b566644fc9674e3477032ec3758ca8d833313R161">adding fused uint4x2_mixed_mm to inductor by HDCharles · Pull Request #106516 · pytorch/pytorch</a>: 来自 ghstack 的堆栈（最早的在底部）：-&amp;gt; #106516 摘要：这是 int4 仅权重（weight-only）量化所需要的，我们正在匹配解包 uint4x2 的特定解包操作...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1225777509120086059)** (23 条消息🔥): 

- **辩论 Three.js 是否大材小用**：成员们讨论了使用 **Three.js** 进行可视化的可能性，但有些人认为它对于他们的需求来说功能过于强大且复杂。考虑将 **D3** 作为一个更易于交互的选择。

- **可视化共享内存和张量**：关于 **triton-viz** 中视觉表示的对话，考虑如何有效地显示共享内存（shared memory）和张量视图。一位成员计划使用 **ipycanvas + ipyevents** 在 Jupyter 中实现丰富的视觉效果，以补充当前的 Gradio 设置。

- **Triton 调试挑战**：小组讨论了调试 Triton 代码时遇到的常见问题，特别是将数据加载到错误位置的频繁问题。建议重点关注 Kernel 中数据来源的可视化，以帮助开发者。

- **Triton Visualization for CPU Constructs**：成员们对在 **triton-viz** 中可视化循环和控制流结构表示出兴趣，尽管有人担心当前视图对于此类功能的直观性。鼓励成员们针对潜在解决方案进行集思广益。

- **Interactive Debugging with JavaScript**：有建议提出在 JavaScript 中实现可视化调试工具以增强交互（如鼠标悬停效果和快速动画），从而更好地理解 Triton 的调试追踪并提供更清晰的教程。
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1225854310093426830)** (59 messages🔥🔥): 

- **Tinygrad Learning Resources**：对于想要为 tinygrad 做出贡献的人，可以在 [GitHub - mesozoic-egg/tinygrad-notes](https://github.com/mesozoic-egg/tinygrad-notes) 找到教程和文档。

- **Reversion of the Command Queue**：George Hotz 提到了 tinygrad 开发中命令队列的回滚，并评论道：*lol no, reverted*。

- **Memory Scheduler Integration Strategy**：根据 George Hotz 的说法，内存调度器将集成到调度器本身中，并且 *队列相关事务可以通过现有的 multidevicegraph 抽象来处理*。

- **Exploration of RISC-V Opcodes in Firmware**：成员们讨论了 MEC 固件的架构，辩论其是否基于 RISC-V 并分析了不同的操作码结构，包括一个意外的 `cbz` 指令。

- **Usage Guidelines for TinyJit Requested**：一位成员寻求关于使用 TinyJit 的建议，并询问遇到的问题是由于误用还是 Bug，这引发了关于 RISC-V ISA（包括 ARM 助记符的使用）细微差别的进一步讨论。

- **Tinygrad Role Redefinition and Community Responsibilities Emphasized**：George Hotz 更新了 Discord 角色以反映贡献和参与度，强调了在 tinygrad 项目开发中有效协作和留心他人时间的重要性。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mesozoic-egg/tinygrad-notes/tree/main">GitHub - mesozoic-egg/tinygrad-notes: Tutorials on tinygrad</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4094">new memory scheduler with LRU by geohot · Pull Request #4094 · tinygrad/tinygrad</a>: no description found
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1225851944602701964)** (6 messages): 

- **TinyJIT Unveiled**：一位成员分享了关于 [TinyJit 的教程](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/jit.md)，供对其工作原理感兴趣的人参考，尽管其中可能包含一些不准确之处，特别是在 `apply_graph_to_jit` 部分。
- **Clarifying TinyJIT Mechanics and Seeking Insight**：分享 TinyJit 教程的成员指出 `/graph` 文件夹下的运行时可能存在问题，并邀请他人提供见解以提高内容的准确性。
- **Call for Error Correction on TinyJIT Tutorial**：针对潜在的不准确之处，另一位成员请求“reds”创建一个纠错的 Pull Request 以帮助社区。
- **Diving into Multi GPU Training with Tinygrad**：介绍了另一个解释 tinygrad 如何实现多 GPU 训练的教程，源码可在 [GitHub](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/multigpu.md) 获取。
- **Community Praise for Multi GPU Training Guide**：多 GPU 训练教程广受好评，被成员认为非常有用。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mesozoic-egg">mesozoic-egg - Overview</a>: mesozoic-egg has 4 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/jit.md">tinygrad-notes/jit.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/multigpu.md">tinygrad-notes/multigpu.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1225827225325146112)** (26 messages🔥):

- **申诉解决 Llamafile 误报问题**：Llamafile 版本面临*误报 (false positive)* 恶意软件检测的问题，可能影响 **llamafile-0.6.2.exe** 和 **llamafile-0.7**。建议向提供申诉表单的杀毒软件厂商提交申诉。
  
- **Kaggle 中 Llamafile 的 GPU 问题**：一位用户在 Kaggle 中运行 `llamafile` 时遇到问题，原因是 CUDA 编译复杂以及寻找兼容的 GPU 架构。另一位用户提供了一个**更新后的命令**以方便使用 `llamafile-0.7`。

- **RAG-LLM 应用的本地分发考虑**：一名成员询问如何在不依赖 Docker 或 Python 等沉重依赖项的情况下在本地分发 RAG-LLM 应用，并考虑为 macOS 用户使用 **llamafile**。得到的答复是 **llamafile** 可以满足这些需求。

- **通过调整 `-ngl` 解决 Llamafile 显存溢出错误**：一位用户通过**微调 `-ngl` 参数**成功解决了显存溢出 (out of memory) 错误，他们最初为 NVIDIA GeForce GTX 1050 显卡设置的数值过高。

- **提议增强 Vulkan 支持**：在测试显示 Vulkan 在带有集成 GPU 的普通 Intel 笔记本电脑上性能有所提升后，有人建议在 llamafile 中增加对 Vulkan 的支持。然而，对于需要重新导入并对 **llama.cpp** 应用本地更改以实现此功能，人们表示了担忧。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.virustotal.com/gui/file/57a2ad7b2458896e8936f00cd4c91c8b4c919fceab35bfd3f85371b3a84dc935">VirusTotal</a>：未找到描述</li><li><a href="https://www.virustotal.com/gui/file/57a2ad7b2458">VirusTotal</a>：未找到描述</li><li><a href="https://www.virustotal.com/gui/file/37a39d8970573110c425c3edd1be4b1df6ab32c4a4a38ae6d98ad4728093267e">VirusTotal</a>：未找到描述
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1226255944778780783)** (9 条消息🔥): 

- **Schedule-Free Optimizers 进入 HF Transformers**：[huggingface/transformers 仓库](https://github.com/huggingface/transformers/pull/30079)中的一个新拉取请求引入了 Meta 的 AdamW 和 SGD Schedule-Free Optimizers 集成，这可能是模型训练的一个重要更新。
- **使用 AdaptiveSoftmax 训练？**：一位成员正在寻求关于使用 *adaptivesoftmax* 进行训练的见解或成功案例，但未提供具体细节或背景。
- **德国 AI 社区活动**：宣布为德国 AI 工程师举办 "AIDEV" 社区活动，将于 5 月 7 日在 Hürth 举行。讨论将围绕合成数据生成、LLM/RAG 流水线和嵌入展开，采用以开发者为中心、务实的态度。感兴趣的各方可以在 [Developer Event - AI Village](https://www.eventbrite.de/e/developer-event-ai-village-tickets-868896702427) 免费注册。
- **请求关于合成数据生成的公开信息**：一位成员询问关于合成数据生成及相关策略的公开信息或讨论，特别是在德国背景下，并提到了德语翻译数据与德语生成数据的对比。
- **活动后总结分享**：几位成员对即将在德国 Hürth 举行的活动表示热烈期待，并请求在活动结束后为无法参加或渴望消化讨论内容的人分享总结和见解。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/huggingface/transformers/pull/30079">winglian 提交的 schedulefree optimizers · Pull Request #30079 · huggingface/transformers</a>：此 PR 的作用？集成了 Meta 的 https://github.com/facebookresearch/schedule_free 用于 adamw 和 sgd。在提交之前...</li><li><a href="https://www.eventbrite.de/e/developer-event-ai-village-tickets-868896702427">Developer Event AI Village</a>：AIDev - 开发者社区 Large Language Models，LLM 应用和生成式 AI
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1225740644690034698)** (5 条消息):

- **Command-R 令人赞叹的性能表现**：分享了一个指向 [Hugging Face 上 **Command-R** Space](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus) 的链接，其令人惊叹的 Grounding 能力被描述为“震撼人心”，可能会影响未来模型的开发。
- **在中古高地德语翻译中树立新标杆**：来自 *CohereForAI* 的 Command-R 在将中古高地德语翻译成现代德语方面表现出色，轻松超越了 GPT-4 级别的模型，使其他 LLM 数月的专门训练显得过时。
- **对开发者活跃度和开源许可的影响**：有人表示希望 Cohere 为其性能优越的新模型采用完全开源的许可，因为这将可能促进开发者参与和生态系统增长，Mistral 就是这种策略带来经济效益的一个例子。
- **Command-R 优越性的具体实例**：据称 Command-R 提供了完美的中古高地德语翻译，并似乎能识别源材料，显示出强大的 Needle-in-a-haystack 能力，使其成为集成 RAG (retrieval-augmented generation) 功能的首选。

**提到的链接**：<a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>：未找到描述

---

**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1225925361955770498)** (9 条消息🔥): 

- **Jamba 训练速度的警示**：一位成员报告称，在 HGX 上训练 1B Mamba 模型比其 Transformer 对应版本慢了 *76%*。经过澄清，确定他们是将训练速度与常规 Transformer 进行了对比。

- **针对有限硬件的备选 Jamba 方案**：一位用户为无法在本地运行完整 52B 模型的用户创建了 **Jamba** 架构的精简版本，包括 [29B 参数的 8xMoE](https://huggingface.co/isemmanuelolowe/Jamba-8xMoE_Slerp) 和 [17.7B 参数的 4xMoE](https://huggingface.co/isemmanuelolowe/Jamba-4xMoE_Slerp)。这些模型显示出良好的效果，并可以在 4090 GPU 上以 4-bit 进行微调。

- **分享 Jamba 缩减技术**：针对如何创建参数减少模型的好奇，该用户提到使用了专家权重的累积 Slerp (spherical linear interpolation) ，并承诺很快会分享 `ipynb` 笔记本。

- **推理引擎查询**：一位成员寻求关于服务 **Jamba** 模型的最佳推理引擎的建议，但在给定的消息中没有后续的直接推荐。

- **Jamba 的 GPU 利用率挑战**：一位用户成功复制了 [Hugging Face Jamba 模型页面](https://huggingface.co/ai21labs/Jamba-v0.1) 的微调示例，但由于模型体积庞大，不得不将 52B 模型的训练分布在 8 个 GPU 上。由于 Pipeline Parallelism 的限制，导致仅利用了总 GPU 容量的 1/8，他们询问了关于使用 Tensor Parallelism (TP) 进行训练的问题。

---

**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1225899553975439431)** (2 条消息): 

- **使用 QNAP NAS 驱动 AI**：一位成员强调了一个[实用的家庭 AI 设置方案](https://www.storagereview.com/review/run-a-private-rag-chatgpt-on-qnap-nas)，使用添加了 GPU 的 **QNAP NAS** 来测试 AI 能力。该设置涉及 TS-h1290FX 型号，配备 AMD EPYC 7302P CPU、256GB DRAM 和 25GbE 能力。
- **存储系统提示词以提高效率**：一位成员询问其他人是否已开始存储和检索常用任务的 System Prompts，以简化 AI 交互中设置上下文的过程。在现有消息中未提供进一步的背景或回复。

**提到的链接**：<a href="https://www.storagereview.com/review/run-a-private-rag-chatgpt-on-qnap-nas">在 QNAP NAS 上运行私有 RAG ChatGPT</a>：QNAP NAS 平台在该类别中拥有最独特且功能强大的硬件设计。我们为一个 NAS 添加了 GPU 并测试了其 AI 能力。

---

**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1226911956590399512)** (3 条消息):

- **介绍 Alter**：Alter 源于 llm-cli 的使用，即将推出 beta 版，为 macOS 带来 AI 驱动的文本改进功能，[YouTube 上有演示视频](https://youtu.be/IK53CSSbaqI)。该应用可与包括 Keynote 在内的各种 macOS 应用程序集成，用于生成和编辑内容。 
- **Alter 让 AI 触手可及**：Alter 承诺在所有 macOS 应用程序中提供上下文感知的 AI 能力，提供一个集中的 AI 工具来替代多个订阅和插件。有关功能、定价和能力的详细信息可以在 [Alter 网站](https://alterhq.com)上找到。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://alterhq.com">Alter | 适用于 Mac 的隐形 AI</a>：未找到描述</li><li><a href="https://youtu.be/IK53CSSbaqI">Alter 演示 - 修复拼写和语法错误</a>：在所有应用中改进语法和拼写！Alterhq.com 根据您的工作上下文推荐最佳操作。#ai #spelling #macos #chatgpt
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1226632752988815380)** (1 条消息): 

- **神经网络资源分配创新**：成员讨论了一篇关于在神经网络中为每个 token 管理静态计算预算的**动态分配 (dynamic allocation)** 的论文。这一策略引起了该成员在 **neurallambda** 中实现的兴趣，并提出了网络可以识别如何优化分配计算资源的观点。
- **思考新的训练技术**：该成员考虑为 **neurallambda** 整合各种方法，例如使用 pause/think token、通过强化学习实现条件判断，并从一篇 RNN 发射自身计算使用量的论文中汲取灵感。
- **探索神经输入处理方法**：**neurallambda** 的其他考虑因素包括将输入读取到神经队列中以进行灵活处理，并将输入视为磁带，能够按需发出磁带移动指令，类似于图灵机的运行。
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1225759372379029534)** (2 条消息): 

- **探索结构化数据提取**：分享了一个名为 "Instructor, Generating Structure from LLMs" 的视频，演示了如何从包括 **GPT-3.5, GPT-4** 和 **GPT-4-Vision** 在内的大型语言模型中提取 JSON 等结构化数据。该视频旨在让从 LLM 获取可靠的结构化结果变得更加容易。[在此观看视频。](https://www.youtube.com/watch?v=KxOqjKq2VyY)
- **分享了另一个视频**：提供了第二个 YouTube 视频链接，但没有提供额外背景。[查看视频。](https://www.youtube.com/watch?v=keUjQyWmgY0)

**提到的链接**：<a href="https://www.youtube.com/watch?v=KxOqjKq2VyY">Instructor，从 LLM 生成结构化数据</a>：Instructor 使得从 GPT-3.5, GPT-4, GPT-4-Vision 等大型语言模型（LLM）以及开源模型中可靠地获取 JSON 等结构化数据变得容易……

  

---



**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1226270438519472178)** (2 条消息): 

- **寻找 Haiku 速度解决方案**：一位成员询问了优化 **Haiku** 性能的方法，因为他们在当前设置下遇到了无法接受的速度问题。

- **Anthropic 的 API 成为焦点**：用户分享的结果显示，**Anthropic** 新的 tool use beta API 在 Berkeley 函数调用基准测试的一半场景中表现优于 **GPT-4 Turbo**。完整的实验结果详见 [Twitter 线程](https://x.com/JoschkaBraun/status/1777381282751688868)。

**提到的链接**：<a href="https://x.com/JoschkaBraun/status/1777381282751688868">Joschka Braun (@JoschkaBraun) 的推文</a>：我在 Berkeley 函数调用基准测试上测试了 @AnthropicAI 新的 tool use beta API。Haiku 在一半的场景中击败了 GPT-4 Turbo。结果见 🧵 非常感谢 @shishirpatil_, @fa...

  

---