---
companies:
- google
- cuda-mode
- nvidia
- polymind
- deepseek
- ollama
- runpod
- lmstudio
date: '2024-02-23T00:51:56.427034Z'
description: '**Google Gemini Pro** 重新引发了人们对长上下文能力的关注。**CUDA MODE Discord** 社区目前正致力于实现由
  Liu、Zaharia 和 Abbeel 发表的 **RingAttention** 论文，包括《世界模型 RingAttention》论文中的扩展内容，并提供了
  PyTorch 和 CUDA 的具体实现。


  **TheBloke Discord** 频道讨论了多个话题，包括 **LLM 猜谜游戏评估**、**英伟达（Nvidia）的 Chat with RTX**
  与 **Polymind** 之间的聊天机器人用户体验（UX）对比、**检索增强生成（RAG）** 集成的挑战、显存（VRAM）优化、使用**直接偏好优化（DPO）**进行角色扮演微调，以及
  **deepseek-coder-6.7B-instruct** 等模型选择。


  此外，还有关于 Mac Studio 上机器学习工作流的讨论，用户更倾向于使用 **llama.cpp** 而非 **ollama**，并探讨了如何利用 Runpod
  上的 **4090** 等 GPU 实现高性价比的推理扩展。**LM Studio** 用户需要手动更新至 **0.2.16** 版本，该版本包含对 **Gemma
  模型** 的支持以及针对 MacOS 的错误修复。在模型表现方面，**Gemma 7B** 模型存在性能问题，而 **Gemma 2B** 则获得了积极的反馈。'
id: cdeb5f79-6d6e-4d7a-a8eb-91e099ca1625
models:
- gemini-pro
- gemma-7b
- gemma-2b
- deepseek-coder-6.7b-instruct
- llama-cpp
original_slug: ainews-ring-attention-for-1m-context
people:
- liu
- zaharia
- abbeel
title: '**Ring Attention：支持超过 100 万上下文**'
topics:
- long-context
- ringattention
- pytorch
- cuda
- llm-guessing-game
- chatbots
- retrieval-augmented-generation
- vram-optimization
- fine-tuning
- dynamic-prompt-optimization
- ml-workflows
- gpu-scaling
- model-updates
---

<!-- buttondown-editor-mode: plaintext -->> 2024年2月21日的 AI Discord 动态。我们为您检查了 **20** 个公会、**317** 个频道和 **8751** 条消息。预计节省阅读时间（以 200wpm 计算）：**796 分钟**。

> **昨日更新**：抱歉发了一封空白邮件——有人在 langchain Discord 中发了一个违规链接，导致 buttondown 渲染过程出错。我们已经修复了它，您可以在这里查看 [昨日的 Google Gemini 简报](https://buttondown.email/ainews/archive/ainews-google-ai-win-some-gemma-15-pro-lose-some/)。

Gemini Pro 让大家意识到了长上下文（long context）的好处。CUDA MODE Discord 启动了一个实现 RingAttention 论文（[Liu, Zaharia, Abbeel](https://arxiv.org/abs/2310.01889)，以及扩展的 [World Model RingAttention 论文](https://arxiv.org/abs/2402.08268)）的项目。

 
![image.png](https://assets.buttondown.email/images/fbc80a0b-ad5b-43f0-a0eb-1731b1ee2cdb.png?w=960&fit=max)
 

论文当然附带了 [pytorch 实现](https://github.com/LargeWorldModel/LWM/blob/main/lwm/ring_attention.py#L3?)，[lucidrains](https://github.com/lucidrains/ring-attention-pytorch) 也有一个版本。但您可以在这里查看 CUDA 实现：[https://github.com/cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention)

---

**目录**

[TOC] 


# 第一部分：Discord 高层摘要




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

**LLM 猜谜游戏评估**：对语言模型的实验展示了它们在理解指令方面的潜力，特别是在互动猜谜游戏中，准确的数字选择和用户参与度是关键。

**UX 战场：聊天机器人**：围绕聊天机器人界面的激烈辩论，将笨重的 Nvidia 的 Chat with RTX 与灵活的 Polymind 进行了对比，强调了用户友好配置的重要性。

**RAG 的严苛实现之路**：检索与生成（RAG）功能的集成引发了讨论，重点关注将此类功能整洁且有效地融入项目的复杂性。

**Discord 机器人 CSS 难题**：用户表达了在自定义 Discord 机器人时遇到的 CSS 挑战，突显了 UI 设计与机器人功能之间无缝集成的困难。

**VRAM：隐形的计算货币**：专注于资源优化，讨论集中在协调 VRAM 容量与模型需求上，强调了性能与计算开销之间的平衡。

**角色扮演微调技巧**：像 `@superking__` 和 `@netrve` 这样的用户分享了为角色扮演微调 AI 的见解，策略围绕全面的基础知识和通过 Dynamic Prompt Optimization (DPO) 进行的针对性训练。

**AI 故事与角色扮演热潮**：针对故事写作和角色扮演的新模型发布，这些模型在人类生成的内容上进行训练，以改进 ChatML 中可控的交互，引发了实测的浓厚兴趣。

**代码分类难题**：寻找理想的 LLM 来对 RAG 流水线中的代码相关性进行分类，导致了对 `deepseek-coder-6.7B-instruct` 的考量，社区成员正在寻求进一步指导。

**Mistral 模型下载荒**：出现了一个关于本地 Mistral 可访问性的未详细说明的请求，但由于信息太少，无法获得建设性的社区支持。

**Mac Studio 上的工作流困扰**：阐述了 Mac Studio 上的 ML 工作流挣扎，包括可能从 ollama 切换到 `llama.cpp`，赞扬其简单性并质疑行业向 ollama 推进的趋势。

**VSCode 地位被 Zed 取代**：像 `@dirtytigerx` 这样的用户推崇 Zed 优于 Visual Studio Code，强调其极简设计和速度。基于 Atom 的开源文本编辑器 Pulsar 也受到了关注。

**通过战术性 GPU 部署扩展推理**：讨论了扩展推理服务器的成本效益方法，建议在全面部署之前，先在 runpod 上使用 4090 等实惠的 GPU 进行初步原型设计，同时注意云服务商服务协议的可靠性。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 摘要

- **LM Studio 更新需要手动操作**：由于应用内更新功能目前无法正常运行，用户必须从 [LM Studio v0.2.16](https://lmstudio.ai) 手动下载最新功能和错误修复。更新内容包括 Gemma 模型支持、改进的下载管理和 UI 增强，并在 v0.2.16 中解决了关键 bug，特别是针对 MacOS 用户遇到的高 CPU 占用问题。

- **社区解决 Gemma 故障**：持续的讨论显示 Gemma 7B 模型存在问题，伴有性能故障和错误；然而，Gemma 2B 模型收到了积极反馈。在调整 GPU 滑块后，M1 Mac 上的 Gemma 7B 表现有所改善。可在 [Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF) 获取可用的 Gemma 2B 模型。

- **Stable Diffusion 3 引起关注**：Stability.ai 宣布了 Stable Diffusion 3 的早期预览，引发了用户对其改进的多主体图像质量的讨论。爱好者们考虑申请预览，并讨论了如 AUTOMATIC1111 等 Web UI 工具，用于处理独立于 LM Studio 关注点之外的图像处理任务。

- **探索大模型的硬件障碍**：社区深入探讨了运行 Goliath 120B Q6 等大模型的挑战，交流了关于 Tesla P40 等旧款 GPU 可行性的见解，并辩论了 AI 任务中 VRAM 容量与 GPU 性能之间的平衡。

- **Gemma 模型排查持续进行**：用户在 Gemma 的不同量化版本上取得了参差不齐的成功，7B 模型经常产生乱码，而 2B 模型运行更可靠。LM Studio 下载面临关键问题，建议在 [LM Studio 官网](https://lmstudio.ai) 和 [GitHub](https://github.com/ggerganov/llama.cpp/issues/5635) 上解决。一个已确认适用于 LM Studio 的稳定量化版 Gemma 2B 模型可以在此 [Hugging Face 链接](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF) 找到。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

**将 LLM 扩展到新高度**：`@gabriel_syme` 强调了一个专注于数据工程的仓库，旨在将语言模型扩展到 128K context，这是该领域的一项重大进展。`@teknium` 指出，此类模型在 7B 规模下的 VRAM 需求超过 **600GB**，对资源的需求巨大。

**Google 进入 LLM 领域**：Google 推出了 **Gemma** 系列轻量级开源模型，`@sundarpichai` 进行了热烈报道，社区反馈褒贬不一，将其与 **Mistral** 和 **LLaMA** 等现有模型进行了比较。用户 `@big_ol_tender` 和 `@mihai4256` 参与了各种讨论，从指令放置的影响到不同服务间的 VM 性能。

**开源开发与支持**：`@pradeep1148` 分享了一个视频，建议自我反思可以改进 **RAG 模型**，`@blackblize` 寻求关于使用 AI 处理显微镜照片进行艺术图像生成的指导。同时，`@afterhoursbilly` 和 `@_3sphere` 批评了 AI 生成的 Minecraft 物品栏 UI 图像。

**新兴 AI 基础设施讨论**：关于 *Nous-Hermes-2-Mistral-7B-DPO-GGUF* 的对话反映了其与其他模型对比的查询，`@iamcoming5084` 谈到了 *Mixtral 8x7b models* 的显存溢出（out-of-memory）错误。还探讨了托管 *Mixtral 8x7b* 等大模型的策略，用户对不同工具进行了辩论，并指出了推理代码中的错误（[修正后的 Nous-Hermes-2-Mistral-7B-DPO 推理代码](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO)）。

**协作项目挑战**：在 **#project-obsidian** 频道中，`@qnguyen3` 通知由于个人原因项目有所延期，并建议通过私信进行项目方面的协调。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **澄清模型评估与 lm eval 的混淆**：`@lee0099` 对在 runpod 上设置 `lm eval` 的困惑促使 `@hailey_schoelkopf` 澄清了 `lm eval` 与 `llm-autoeval` 之间的区别，并引用了 [Open LLM Leaderboard 的 HF spaces 页面](https://huggingface.co/spaces) 以获取说明和参数。关于 `@gaindrew` 提出的按净碳排放量对模型进行排名的建议，由于准确性方面的挑战，尚未达成明确共识。

- **Gemma 的成长阵痛与技术磨合**：`@sundarpichai` 推出的 **Google Gemma** 引发了关于其相对于 **Mistral** 等模型改进程度的辩论。会议强调了 Gemma 模型中参数数量的误报问题（“gemma-7b” 实际上拥有 85 亿参数）。[Groq](https://www.semianalysis.com/p/groq-inference-tokenomics-speed-but) 声称在 Mistral Mixtral 8x7b 模型上实现了 4 倍的吞吐量，并大幅降低了成本。讨论中还涉及了对模型环境足迹的担忧以及 `@philpax` 对 Groq 声明的报告，同时研究人员深入探讨了模型效率以及 [PGI 在解决数据丢失中的应用](https://arxiv.org/abs/2402.13616)。

- **探索多语言模型的奥秘**：一篇 [Twitter 帖子](https://twitter.com/cervisiarius/status/1759989584371298554?t=fDN-bfsJDhWP4lfAjSOnHA&s=19) 及其配套的 [GitHub 仓库](https://github.com/epfl-dlab/llm-latent-language) 引发了关于模型是否“用英语思考”以及在 **Llama** 等模型上使用调优透镜（tuned lens）的实用性的辩论。`@mrgonao` 对多语言能力的讨论促使了对创建中文透镜的思考。

- **LM Thunderdome 的技术深潜**：在众多的内存问题中，`@pminervini` 面临 Colab 中 OOM 错误后 GPU 显存持续占用的问题，需要重启运行时，该问题在 [Colab 的 Evaluate OOM Issue 环境](https://colab.research.google.com/drive/1u5MoN-QUfdNJXilFJAyJaGY1HlYWnfwX?usp=sharing) 中重现。此外，还有关于评估 Gemma-7b 模型的报告，需要 `@hailey_schoelkopf` 的干预，他提供了修复方法和使用 `flash_attention_2` 的优化技巧。

- **解决假阴性问题并推进 CLIP**：在多模态对话中，`@tz6352` 和 `@_.hrafn._` 讨论了 **CLIP 模型** 中的批次内假阴性（in-batch false negative）问题，详细说明了涉及单模态嵌入的解决方案，以及在模型训练期间利用相似度得分进行负样本排除的策略。

- **预训练序列组合的重要性**：**gpt-neox-dev** 频道仅记录了来自 `@pminervini` 的一条消息，分享了一篇 [arXiv 论文](https://arxiv.org/abs/2402.13991)，该论文指出文档内因果掩码（intra-document causal masking）有助于消除前一个文档的干扰内容，从而可能提高语言模型在各种任务中的性能。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 总结

- **Google 发布 Gemma 模型，转向开放 AI**：Google 推出了 [Gemma](https://blog.google/technology/developers/gemma-open-models/)，这代表了其在 Gemini 模型基础上的进步，并暗示其正向更开放的 AI 开发转变。社区对 Google 发布实际开源权重的动机表现出兴趣，因为他们传统上对此持谨慎态度。

- **Stable Diffusion 3 的关注与担忧**：[Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3) 的早期预览版已经发布，重点是改进多主体提示词处理和图像质量，但其与早期版本的差异化正受到审查。此外，关于 SD3 的商业利用，以及开源究竟是作为一种宣传策略还是收入策略，也产生了一些疑问。

- **AI 领域的中心化引发关注**：讨论反映了对 AI 开发和资源中心化的日益担忧，例如 Stable Diffusion 3 的开放程度降低，这可能会使算力超出终端用户的承受范围。

- **扩散模型作为神经网络生成器**：一篇 [Arxiv 论文](https://arxiv.org/abs/2402.13144) 分享了关于如何使用扩散模型生成高效神经网络参数的见解，指出了一种构建新模型的全新且可能具有变革性的方法。

- **AnyGPT：统一多模态 LLM 的黎明**：[AnyGPT](https://junzhan2000.github.io/AnyGPT.github.io/) 的推出（演示视频可在 [YouTube](https://youtu.be/oW3E3pIsaRg) 观看）突显了语言学习模型（LLM）处理语音、文本、图像和音乐等多种数据类型的能力。

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **Mistral 的图像文本提取能力受到审视**：**Mistral AI** 在处理复杂图像文本检索方面的能力受到质疑。对于需要更高灵活性的任务，建议使用 **gpt4-vision**、**gemini-vision** 和 **blip2**，而非 **copyfish** 和 **google lens** 等简单工具。

- **Mistral API 与微调探索**：用户交流了关于各种 **Mistral models** 的信息，包括 **Mistral API** 指南、**Mistral 7B** 和 **Mistral 8x7b** 模型的微调，以及在 **Hugging Face** 和 **Vertex AI** 等平台上的部署。集成公司数据时引用了 **Basic RAG guide** ([Basic RAG | Mistral AI](https://docs.mistral.ai/guides/basic-RAG/))。

- **部署讨论聚焦关注点与成本评估**：关于 **AWS hosting costs** 和 vLLM 的 **GPU selection**（GPU 选择）的咨询引发了对部署方案的讨论。部署 vLLM 时参考了相关文档 ([vLLM | Mistral AI](https://docs.mistral.ai/self-deployment/vllm/))。

- **对未发布的 Mistral Next 的期待**：**Mistral-Next** 已确认为即将推出的模型，目前尚无 API 访问权限。**Mistral Next** 卓越的数学性能引发了与 **GPT-4** 的比较。详情备受期待但尚未发布。

- **展示 Mistral 的多功能性与潜力**：一段 YouTube 视频展示了通过自我反思增强 **RAG** ([Self RAG using LangGraph](https://www.youtube.com/watch?v=Eb7QF1nDWGU))，另一段视频讨论了微调的益处 ([BitDelta: Your Fine-Tune May Only Be Worth One Bit](https://www.youtube.com/watch?v=T_dYzuv4N70))。**Jay9265** 在 **Twitch** 上对 **Mistral-Next** 的测试 ([Twitch](https://www.twitch.tv/jay9265/)) 以及提示词能力指南 ([Prompting Capabilities | Mistral AI](https://docs.mistral.ai/guides/prompting-capabilities/)) 也被重点提及，以展示 Mistral 的能力和用途。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **Google 的 AI 持续进化**：Google 发布了一个具有更新功能的**新模型**；然而，关于其名称和能力的细节尚未完全明确。关于 OpenAI，讨论中确认 ChatGPT 的移动版缺乏插件支持，导致用户在移动浏览器上尝试桌面版以获取完整功能集。

- **OpenAI 定义 GPT-4 访问限制**：关于 GPT-4 的使用上限（usage cap）展开了辩论，成员们澄清该上限是根据需求和算力可用性进行**动态调整**的。显然，自发布以来 GPT-4 的模型性能并未下降，平息了关于其能力减弱的传闻。

- **AI 模型的稳定性与多样性**：**Stability.ai** 发布了 **Stable Diffusion 3** 的早期预览版，承诺在图像质量和提示词处理方面有所增强。而围绕 Google Gemini 模型的讨论则对其处理多样性的方法提出了疑问。

- **精通提示工程 (Prompt Engineering)**：对于旨在提高 AI 角色扮演能力的 AI 工程师来说，关键是构建具有**清晰、具体且逻辑一致的指令**的提示词，使用开放变量和正向强化方法。该领域的进一步学习资源可以在 **arXiv** 和 **Hugging Face** 等平台上找到。

- **探索 API 与模型能力**：API 交互采用按需付费（pay-as-you-go）模式，与 Plus 订阅分开。此外，最近增加了**文件上传限制**，现在支持 20 个 512MB 的文件。讨论还涉及了使用 HTML/CSS 文件训练模型的细微差别，帮助工程师优化 GPT 对 Web 开发语言的理解和输出。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 总结

- **404 账户之谜与 Diffusion Model 深度探讨**：用户报告了 HuggingFace 的各种问题，例如某个账户出现 404 错误（可能由于虚报库统计数据），以及在 NixOS 上配置 `huggingface-vscode` 扩展的挑战。此外，还分享了关于 SDXL 等 Diffusion Model 使用傅里叶变换增强微调节（microconditioning）输入的深度讨论，同时用户也对用于大学项目的基于中间语言（interlingua）的翻译器，以及运行具有扩展类别的 [BART-large-mnli 模型](https://huggingface.co/facebook/bart-large-mnli) 表现出兴趣。

- **AI 工程实践**：
    - 一位用户分享了一个[用于管理投资组合的 Web 应用](https://huggingface.co/spaces/luisotorres/portfolio-management)，并附带了 [Kaggle Notebook](https://www.kaggle.com/code/lusfernandotorres/building-an-investment-portfolio-management-app)。
    - 介绍了一个使用 SigLIP 的多标签图像分类[教程 Notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SigLIP/Fine_tuning_SigLIP_and_friends_for_multi_label_image_classification.ipynb)。
    - 通过重新安装 `2.15 版本` 解决了 TensorFlow 问题。
    - 探讨了生物医学领域的句子相似度挑战，并推荐使用对比学习以及 sentence transformers 和 setfit 等工具进行微调。

- **挑战 AI 范式**：
    - 讨论了 PEFT 在没有自动配置的模型中无法保存正确 head 的问题，并引用了一种使用 Reformer 架构在边缘设备上实现内存高效模型的新方法。
    - 关于模型基准测试工作的讨论包括分享的[排行榜](https://lnkd.in/gxUHqwNp)和[仓库链接](https://lnkd.in/dwhXQ_Bm)，并邀请贡献和见解。

- **新兴 AI 技术预警**：
    - 展示了一个用于单目深度估计的 Android 应用和一个使用 Selenium 的非官方 ChatGPT API，引发了对服务条款（TOS）和绕过保护机制的担忧。
    - 公告包括 Stable Diffusion 3 的早期预览，以及对 nanotron 在 [GitHub](https://github.com/huggingface/nanotron/tree/main/examples/doremi) 上开源的兴奋，这标志着 AI 领域的持续改进和社区努力。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **Google 发布 Gemma 语言模型**：Google 推出了名为 **Gemma** 的新语言模型系列，目前 7B 和 2B 尺寸已在 Hugging Face 上可用。分享的发布条款强调了对分发模型衍生品的限制（[Hugging Face 博客文章](https://huggingface.co/blog/gemma)，[发布条款](https://ai.google.dev/gemma/terms)）。

- **破解 Tokenizer 差异**：对 Gemma 的 Tokenizer 与 Llama 2 的 Tokenizer 进行了深入对比分析，揭示了 Gemma 具有更大的词汇表和特殊 Token。该分析得到了 Tokenizer 模型文件链接和 diffchecker 对比的支持（[Tokenizer 模型文件](https://github.com/google/gemma_pytorch/blob/main/tokenizer/tokenizer.model)，[diffchecker 对比](https://www.diffchecker.com/TRnbKRMH/)）。

- **Stable Diffusion 3 登场**：Stability AI 在早期预览中宣布了 Stable Diffusion 3，相比之前的版本，它在多主体提示词处理和图像质量方面有所提升（[Stability AI 公告](https://stability.ai/news/stable-diffusion-3)）。

- **ChatGPT 异常行为已修复**：据 OpenAI 状态页面显示，此前报告的 ChatGPT 异常行为事件已得到解决。成员们分享了推文链接和事件报告以提供背景信息（[OpenAI 状态页面](https://status.openai.com/incidents/ssg8fh7sfyz3)）。

- **探索 AI 驱动的生产力**：对话围绕 Google 的 **Gemini AI** 集成到 Workspace 和 Google One 服务展开，讨论了其新特性，如 1,000,000 Token 的上下文窗口和视频输入功能（[Google One Gemini AI](https://blog.google/products/google-one/google-one-gemini-ai-gmail-docs-sheets/)，[Google Workspace Gemini](https://blog.google/products/workspace/google-gemini-workspace/)）。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 总结

- **简化 RAG 构建**：`@IFTTT` 讨论了构建高级 RAG 系统的复杂性，并建议采用 [@jerryjliu0 的演示](https://t.co/FhwU6tA73o)中的方法来简化流程，该方法精准定位了每个 Pipeline 组件中的痛点。
  
- **RAG 前端创建变得简单**：对于缺乏 React 知识的 LLM/RAG 专家，Marco Bertelli 的教程（由 `@IFTTT` 推荐）展示了如何为他们的 RAG 后端制作精美的前端，资源可从 [@llama_index](https://t.co/35UeUCrKWg) 获取。

- **将 RAG Notebooks 提升为应用程序**：`@wenqi_glantz` 提供了一份将 RAG Notebooks 转换为包含摄取（ingestion）和推理（inference）微服务的全栈应用程序的指南，该指南由 `@IFTTT` 在推文中分享，完整教程[可在此访问](https://t.co/S86B38YZQ1)。

- **LlamaIndex 中的 QueryPipeline 设置和导入错误**：讨论了诸如使用 QueryPipeline 设置简单 RAG、从 `llama_index` 导入 `VectorStoreIndex` 的困难以及导入 `LangchainEmbedding` 等问题，并建议参考 [QueryPipeline 文档](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline.html#rag-pipeline-without-query-rewriting)以及从 `llama_index.core` 导入作为潜在的修复方案。

- **LlamaIndex 资源故障排除**：讨论的主题包括下载 CorrectiveRAGPack 时的 `ValueError`，相关的 [PR #11272](https://github.com/run-llama/llama_index/pull/11272) 可能会提供解决方案；以及影响 `@andaldana` 等用户的文档链接失效问题，这些用户正在寻求 LlamaIndex 中用于处理 SQL 数据库条目数据的更新方法或 Reader。

- **AI 讨论中的参与和咨询**：`@behanzin777` 对社区中建议的解决方案表示感谢，`@dadabit.` 寻求有关 LlamaIndex 内摘要指标和工具的建议，`@.dheemanth` 请求推荐一个用户友好的平台，用于评估具有类似于 **MT-Bench** 和 **MMLU** 能力的 LLM。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- **Google 的 Gemma 发布**：Google 新的 **Gemma** 模型系列引发了积极讨论，其许可协议被发现比 **LLaMA 2** 限制更少，其模型现在可以通过 [Hugging Face](https://huggingface.co/blog/gemma/?utm_source=agd&utm_medium=referral&utm_campaign=view-on-huggingface&utm_content=) 访问。一个 7B **Gemma** 模型被重新上传供公众使用，绕过了 Google 的访问请求协议。然而，微调（finetuning）**Gemma** 出现了一些问题，参考 GitHub 上的潜在早期停止回调（early stopping callback）问题。

- **Axolotl 开发深入研究 Gemma**：**axolotl** 代码库的工作正在进行中，集成了 readme、val 和示例修复。强调了在非开发版本的 *transformers* 上训练 **Gemma** 模型，并分享了更新后的 **gemma config file** 以简化设置。关于合适的超参数（如 **Gemma 模型** 的学习率和权重衰减）存在争论。还在探索优化 **Mixtral 模型** 的方法，有望通过 **AutoAWQ** 提升预填充（prefilling）和解码速度。

- **Alpaca 美学加入 Axolotl**：正在寻求用于 **alpaca** 的 jinja 模板，以增强 [axolotl 仓库](https://github.com/oobabooga/text-generation-webui/blob/main/instruction-templates/Alpaca.yaml)。对使用 DeepSpeed 的训练技巧和微调模型后的正确推理格式有需求，同时还在排除 **FlashAttention** 的故障。重复的查询促使人们呼吁更好的文档，引起了对全面指南必要性的关注。

- **Opus V1 模型助力极具吸引力的故事创作**：**Opus V1 模型** 已发布，在大量用于*故事写作*和*角色扮演*的语料库上进行了训练，可在 [Hugging Face](https://huggingface.co/collections/dreamgen/opus-v1-story-writing-and-role-playing-models-65d092a6f8ab7fc669111b31) 上访问。这些模型受益于先进的 *ChatML* 提示机制以实现受控输出，并有一份[指导指南](https://dub.sh/opus-v1-guide)详细说明了如何引导叙事。

- **RunPod 资源需要检索**：一位用户遇到了 **RunPod 镜像** 消失的问题，建议去 [Docker Hub](https://hub.docker.com/r/winglian/axolotl-runpod/tags) 查找现有的标签。**GitHub readme** 中的错误重定向表明需要更新文档，以正确引导用户找到合适的资源。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord 总结

- **Groq 的 LPU 表现优于竞争对手**：Groq 的 Language Processing Unit 在大型语言模型（LLM）上达到了每秒 241 个 token，创下了新的 AI 基准测试记录。关于 Groq 技术的进一步见解可以参考 Andrew Bitar 的演讲 "Software Defined Hardware for Dataflow Compute"，该视频已上传至 [YouTube](https://youtu.be/PKJYU9ecvWc?si=9BKG75HsaEGTVgMH)。

- **Docker 中的 NVIDIA Nsight 问题**：工程师们正在寻求在 Docker 容器中安装 NVIDIA Nsight 进行调试的帮助，一些人指出在不同云服务商处也遇到了类似困难，并提到 lighting.ai studios 有一个可行的解决方案。

- **新的 BnB FP4 仓库承诺提速**：一个针对 bnb fp4 代码的新 [GitHub 仓库](https://github.com/aredden/torch-bnb-fp4) 已发布，据报道其速度比 bitsandbytes 更快，但需要 CUDA 计算能力 >= 8.0 以及大量的 VRAM。

- **torch.compile 受到审视**：torch.compile 的局限性正受到讨论，特别是它未能捕获通过 Triton/CUDA 可获得的性能提升，以及无法有效处理动态控制流和 kernel fusion 带来的收益。

- **Gemini 1.5 讨论开启**：邀请所有人通过 [Discord 邀请链接](https://discord.gg/F4FfcQw3?event=1209440306404139008) 参加关于 Gemini 1.5 的讨论。此外，还分享了一个展示 AI 从音频文件中解锁语义知识能力的视频，提供了关于 **AI 从音频中学习** 的见解，详见 [此处](https://youtu.be/FgcN62LFzIU)。

- **SIXT 的 ML Engineer 职位**：位于慕尼黑的 SIXT 正在招聘 ML Engineer，重点关注 NLP 和 Generative AI。感兴趣者可以通过 [职业链接](https://www.sixt.jobs/en/job/feb00784-a96f-430b-b105-6116b993b472) 申请。

- **CUDA 在 Groq AI 崛起中依然坚挺**：关于 CUDA 是否会随着 Groq AI 的出现而过时的讨论，最终重申了 CUDA 的基础知识仍然具有价值，且不受先进编译器和架构的影响。

- **TPU 兼容性与 ROCm 的 GPU 困境**：将代码从 TPU 迁移到 GPU 时面临的形状维度（shape dimension）错误，以及 ROCm 对 AMD GPU 的有限支持是热门话题。分享的用于在 AMD GPU 上进行推理的 [GitHub 仓库](https://github.com/ROCm/flash-attention/tree/howiejay/navi_support/) 缺少必要的 backward 函数/kernel。

- **Ring-Attention 凝聚协作动力**：社区正积极参与调试和增强基于 flash-attention 的 ring-attention 实现，并计划进行现场编程会议（live hacking sessions）来解决诸如 FP32 累加必要性等问题。相关的讨论和代码可以在 [此仓库](https://github.com/zhuzilin/ring-flash-attention/) 中找到。

- **YouTube 录制频道的日常维护**：发布了一项维护频道秩序的提醒，要求用户仅发布与 youtube-recordings 相关的内容，并将无关内容重定向到指定的建议频道。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

**Gemini 揭秘**：`@brknclock1215` 帮助消除了关于 **Google Gemini 模型系列** 的困惑，分享了诸如 [Gemini Advanced (Ultra 1.0) 两个月免费试用](https://gemini.google.com/advanced)、[Gemini Pro 1.5 私测版](https://developers.googleblog.com/2024/02/gemini-15-available-for-private-preview-in-google-ai-studio.html?m=1) 等资源，并引导用户阅读一篇详述差异的 [博客文章](https://code.iaflw.com/2024/02/gemini-versus-gemini-understanding.html)。

**寻找 Bot 专家**：用户对 *Perplexity AI bot* 表现出戏谑的兴趣，讨论了其离线状态及使用方法。对于 *Perplexity Pro 版本* 和计费感到困惑的用户，其他人分享了 [FAQ](https://blog.perplexity.ai/faq/billing-and-subscription) 链接以供参考。

**API 难题与代码**：贡献者报告了 **Perplexity API** 与网站内容之间的差异，寻求提高准确性。建议指出应使用更简单的查询，同时承认了 **pplx-70b-online** 模型输出乱码的持续问题，并期待解决。此外，还有关于将 Google 的 [GEMMA](https://ai.google.dev/gemma) 集成到 Perplexity API 的咨询。

**加密货币与健康搜索成为焦点**：好奇的用户进行了 [Perplexity AI 搜索](https://www.perplexity.ai/search/what-does-dydx-Vo_6.U1XQg.eDbP_lg0FHQ?s=c)，主题涵盖从 **cryptocurrency** 交易术语到 **天然口腔健康** 疗法，突显了社区参与话题的多样性。

**金融工具查询**：对知识的探索引导了一次关于金融工具的 [搜索查询](https://www.perplexity.ai/search/What-is-a-fDAg8dSNRhmEeKU.SoY6Fg?s=c)，这表明在围绕金融的讨论中，技术专指性是关键趋势。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **动态类创建难题**：`@deltz_81780` 在尝试为 **PydanticOutputFunctionsParser** 动态生成类时遇到了 **ValidationError**，并在 [general](https://discord.com/channels/1038097195422978059/1038097196224086148/1209767513404084234) 频道寻求帮助。

- **AI 教育扩展**：`@mjoeldub` 宣布了一门专注于 **LangChain 和 LCEL** 的 **LinkedIn Learning 课程**，并分享了[课程链接](https://www.linkedin.com/learning/introduction-to-ai-orchestration-with-langchain-and-llamaindex)，同时重点介绍了由 `@a404.eth` 制作的全新“与你的 PDF 聊天” **LangChain AI 教程**。

- **支持与不满**：围绕 LangChain 支持展开了讨论，`@mysterious_avocado_98353` 表达了失望，而 `@renlo.` 则指出在 [定价页面](https://www.langchain.com/pricing) 上有付费支持选项。

- **LangSmith API 报错**：`@jacobito15` 在 [langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1210069768959434802) 频道的批量摄取测试中，因 `ChannelWrite` 名称超过 128 个字符而遇到了来自 LangSmith API 的 HTTP 422 错误。

- **创新邀请**：`@pk_penguin` 在 [share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1209943158230876190) 中发出了一个未命名的试用邀请，`@gokusan8896` 在 LinkedIn 上发布了关于 **在任何 LLM 模型中进行并行函数调用** 的内容，`@rogesmith` 则征求对潜在聚合查询平台/库的反馈。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **Google 开源 Gemma 模型引发语言多样性查询**：[@sebastian.bodza](https://discord.com/channels/sebastian.bodza) 关注到 Google 的 **Gemma 模型** 已开源，并询问了语言支持情况（特别是德语），引发了关于其在 [Kaggle](https://www.kaggle.com/models/google/gemma) 上的列表以及在 [Hugging Face](https://huggingface.co/google/gemma-7b-it) 上的指令版本可用性的讨论。对话还涉及了商业方面和词汇量大小。

- **对 Aleph Alpha 模型更新的反应褒贬不一**：[@sebastian.bodza](https://discord.com/channels/sebastian.bodza) 对 **Aleph Alpha 模型** 的更新表示怀疑，强调其缺乏指令微调（instruction tuning），[@devnull0](https://discord.com/channels/devnull0) 随后提到最近的招聘可能会影响未来的模型质量。批评意见指出更新未包含其 [变更日志](https://docs.aleph-alpha.com/changelog/) 中所示的基准测试或示例。

- **推文对模型性能的审视**：**Gemma** 和 **Aleph Alpha** 模型的有效性引发了批判性讨论，*@ivanfioravanti* 和 *@rohanpaul_ai* 发布的推文指出模型存在性能问题，特别是在德语等语言中，以及与 phi-2 等其他模型相比时。

- **Batch Size 影响模型评分**：[@calytrix](https://discord.com/channels/calytrix) 提出了关于 **Batch Size** 对模型性能影响的问题，特别是 Batch Size 不为 1 可能会导致评分降低，正如 [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/82) 讨论中所指出的。

- **模型测试公平性受到审视**：[@calytrix](https://discord.com/channels/calytrix) 发起了关于模型测试公平性的讨论，提议公平的测试应该是现实的、明确的、不靠运气的且易于理解的，并请求获取一个脚本以重新生成特定博客文章中的指标，深入探讨了可能扭曲模型评估公平性的细微差别。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 总结

- **Neuralink 面试内幕技巧**：公会成员 `@xilo0` 正在寻求有关即将到来的 **Neuralink** 面试的建议，特别是如何应对“卓越能力的证据”这一问题，以及应该突出哪些项目来给 Elon Musk 的团队留下深刻印象。

- **探索 AI 增强的深度**：`@pradeep1148` 分享了一系列教育类 [YouTube 视频](https://www.youtube.com/playlist?list=PL_kd4Kg6gOnz4BaAeGyI5n8c9VW6r5X6q)，涵盖了通过自我反思（self-reflection）改进 RAG 以及微调 LLM 的价值存疑等主题，并介绍了 Google 的开源模型 **Gemma**。

- **关于 KTO 引用的谜团**：在 papers 频道中，`nagaraj_arvind` 神秘地讨论了 KTO，但未透露细节，使得讨论背景不完整，且 KTO 对 AI 工程师的意义也未得到解释。

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 总结

- **Google 的 Gemini Pro 1.5 重新定义边界**：Google 新发布的 [Gemini Pro 1.5](https://simonwillison.net/2024/Feb/21/gemini-pro-video/) 提供了 **1,000,000 token 的上下文窗口**，并通过引入 *视频输入* 能力进一步创新。Simon W 对这些功能表示了极大的热情，认为这使其区别于 Claude 2.1 和 gpt-4-turbo 等其他模型。

- **Google ML 产品的新文档**：Google 机器学习产品的最新文档现已可在 [Google AI Developer Site](https://ai.google.dev/gemma/docs) 访问，不过目前尚未提供关于文档内容的具体细节。

- **寻求 LLM 集成故障的支持**：在处理系统集成挑战时，@simonw 建议将任何未解决的问题报告给 gpt4all 团队以寻求帮助。

- **GPT-Vision 的愿景**：针对在大型语言模型 (LLMs) 中加入文件支持的问题，@simonw 建议为 **GPT-Vision** 添加图像支持。

- **Gemma 模型初期问题**：有报告称新的 **Gemma 模型** 输出的是占位符文本而非预期结果，建议通过 `llm python` 命令更新依赖项来尝试修复此问题。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

- **寻找 Token？**：Scopexbt 询问是否存在与社区相关的 **token**，并指出目前缺乏相关信息。
- **GLAN 讨论启动**：`.benxh` 通过发布 [GLAN 论文](https://arxiv.org/pdf/2402.13064.pdf) 分享了对 **Gradient Layerwise Adaptive-Norms (GLAN)** 的兴趣，引发了积极的反响。



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **Google 发布 Gemma**：在 #opensource 频道中，potrock 分享了一篇 [博客文章](https://blog.google/technology/developers/gemma-open-models/)，宣布了 **Google** 新的 **Gemma 开放模型** 计划。
- **对比方法获得认可**：在 #embeddings 频道中，一位用户表达了对 **ContrastiveLoss** 的支持，强调了其在微调 embedding 方面的功效，并提到 *MultipleNegativesRankingLoss* 是另一个常用的损失函数。
- **警惕 Salesforce 实现**：在 #general 频道中，res6969 警告不要采用 Salesforce，暗示对于组织来说这可能是一个灾难性的选择。



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord 总结

- **技术畅谈：Gemini 1.5 等你来！**：`@shashank.f1` 邀请大家参加关于 **Gemini 1.5** 的实时讨论，并介绍了之前的会议，包括关于从音频中提取语义知识的 *A-JEPA AI 模型* 的演讲。之前的见解可在 [YouTube](https://youtu.be/FgcN62LFzIU) 上查看。
- **周末研讨会奇思妙想**：`@yikesawjeez` 考虑将原定的活动移至周末，旨在获得更好的参与机会和潜在的赞助合作，其中可能包括在 Twitter 上与 `@llamaindex` 建立联系以及设置 Devpost 页面。



---

# 第二部分：频道详细总结与链接



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1209767041435963442) (1132 条消息🔥🔥🔥): 

- **探索 LLM 游戏动态**：用户尝试使用语言模型来评估其准确理解指令的能力，特别是在猜数字游戏的语境下，模型需要选择一个数字并根据用户的猜测进行互动。

- **聊天机器人用户体验讨论**：用户对比了不同的聊天机器人 UI，重点关注其设置和使用的便捷性。对话中包含了对 Nvidia 的 Chat with RTX 的尖锐批评，以及对 Polymind 等更小、更高效设置的赞赏。

- **函数调用挑战与 RAG 实现**：讨论内容包括实现检索与生成 (RAG) 功能的复杂性以及用户的自定义实现，批评了现有实现的复杂性，并赞扬了更精简的版本。

- **Discord 机器人与 CSS 难题**：用户分享了对 CSS 实现难度的挫败感，并讨论了如何定制 Discord 机器人以实现更好的用户交互和任务处理。

- **优化与模型偏好**：硬件限制和优化是一个重要话题，用户针对各种硬件配置建议了合适的模型。对话强调了显存 (VRAM) 的重要性以及性能与模型复杂度之间的平衡。

**提到的链接**：

- [来自 Alex Cohen (@anothercohen) 的推文](https://fixupx.com/anothercohen/status/1760500433733165226)：遗憾地告诉大家，我今天被 Google 解雇了。我之前负责让 Gemini 的算法尽可能地“觉醒”（woke）。在今天 Twitter 上出现投诉后，我突然失去了访问权限...
- [Bloomberg - 你是机器人吗？](https://www.bloomberg.com/news/articles/2024-02-22/google-to-pause-gemini-image-generation-of-people)：未找到描述
- [Bloomberg - 你是机器人吗？](https://www.bloomberg.com/news/articles/2024-02-22/google-to-pause-gemini-image-generation-of-people-after-issues-lsx286rh)：未找到描述
- [试用全新 NVIDIA App Beta：PC 游戏玩家与创作者的必备伴侣](https://www.nvidia.com/en-us/geforce/news/nvidia-app-beta-download)：NVIDIA app 是保持驱动程序更新、发现 NVIDIA 应用程序、捕捉精彩瞬间以及配置 GPU 设置的最简单方式。
- [一无所知 GIF - No Idea IDK I Dunno - 发现并分享 GIF](https://tenor.com/view/no-idea-idk-i-dunno-i-dont-know-no-clue-gif-5178996)：点击查看 GIF
- [Chris Pratt Andy Dwyer GIF - Chris Pratt Andy Dwyer Omg - 发现并分享 GIF](https://tenor.com/view/chris-pratt-andy-dwyer-omg-shocked-face-meme-gif-25585329)：点击查看 GIF
- [Wyaking GIF - Wyaking - 发现并分享 GIF](https://tenor.com/view/wyaking-gif-9712475764034023502)：点击查看 GIF
- [不睡觉 GIF - No Sleep Love - 发现并分享 GIF](https://tenor.com/view/no-sleep-love-you-gif-22253477)：点击查看 GIF
- [https://i.redd.it/1v6hjhd86vj31.png](https://www.reddit.com/media?url=https%3A%2F%2Fi.redd.it%2F1v6hjhd86vj31.png)：未找到描述
- [我不知道但我喜欢它 Idk GIF - I Dont Know But I Like It I Dont Know Idk - 发现并分享 GIF](https://tenor.com/view/i-dont-know-but-i-like-it-i-dont-know-idk-no-idea-m-not-sure-gif-15770390)：点击查看 GIF
- [ASP.NET Core 有多快？](https://dusted.codes/how-fast-is-really-aspnet-core)：编程冒险
- [机器准备中 GIF - Machine Preparing Old Man - 发现并分享 GIF](https://tenor.com/view/machine-preparing-old-man-gif-17184195)：点击查看 GIF
- [绘画 GIF - Painting Bob Ross - 发现并分享 GIF](https://tenor.com/view/painting-bob-ross-gif-5675661)：点击查看 GIF
- [3rd Rock GIF - 3rd Rock From - 发现并分享 GIF](https://tenor.com/view/3rd-rock-from-the-sun-gif-5973311)：点击查看 GIF
- [Reddit - 深入探索一切](https://www.reddit.com/r/PygmalionAI/comments/19ai9hs/rpygmalionai_is_back_open/)：未找到描述
- [LLM (结合 RAG) 需要一个新的逻辑层 (斯坦福)](https://www.youtube.com/watch?v=42gHxqLu0Kk&ab_channel=code_your_own_AI)：Google DeepMind 和斯坦福大学关于当前 LLM (Gemini Pro, GPT-4 TURBO) 在因果推理和逻辑方面局限性的新见解。
- [VRAM 计算器](https://vram.asmirnov.xyz/)：未找到描述
- [GitHub - itsme2417/PolyMind: 一个多模态、支持 function calling 的 LLM webui。](https://github.com/itsme2417/PolyMind)：一个多模态、支持 function calling 的 LLM webui。 - GitHub - itsme2417/PolyMind: 一个多模态、支持 function calling 的 LLM webui。
- [Reddit - 深入探索一切](https://www.reddit.com/user/DreamGenAI/)：未找到描述
- [反推 DeepBooru：AUTOMATIC1111 中用于分析和标记图像的功能](https://www.andyhtu.com/post/interrogate-deepbooru)：了解 DeepBooru 图像分析和标记的强大功能。了解此功能如何增强 AUTOMATIC1111 以进行动漫风格的艺术创作。立即反推 DeepBooru！
- [GitHub - Malisius/booru2prompt: 一个用于 stable-diffusion-webui 的扩展，将图像 booru 帖子转换为提示词](https://github.com/Malisius/booru2prompt)：一个用于 stable-diffusion-webui 的扩展，将图像 booru 帖子转换为提示词 - Malisius/booru2prompt

---

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1209770438020956170) (299 messages🔥🔥): 

- **角色扮演微调技巧**：`@superking__` 和 `@netrve` 探讨了角色扮演模型的微调细节；关于让 **base model 了解一切**，然后通过微调使角色**仅编写其应该知道的内容**。还提到了使用 **DPO** (Dynamic Prompt Optimization) 来缩小训练范围，并询问训练数据集中科学论文的格式化方式。

- **通过 AI 头脑风暴获得更好的回复**：`@superking__` 观察到，在给出答案之前**让模型进行头脑风暴**通常会让它显得更聪明。相反，由于硬件资源有限，强制模型使用 grammars 回答可能会让它显得更笨。

- **探索模型中的科学论文格式**：`@kaltcit` 分享了他们对**科学论文进行 DPO** 的过程，从学术论文中为 DPO 创建了一个**折叠数据集 (collapsed dataset)**，并与 `@c.gato` 讨论了训练过程中模型 **loss spikes**（损失尖峰）的问题。

- **角色扮演与 ChatML 提示词策略**：`@superking__` 和 `@euchale` 讨论了角色扮演的 **prompt structures**（提示词结构）以及如何防止不希望出现的视角切换，而 `@netrve` 分享了使用 **MiquMaid v2** 进行角色扮演的经验，并指出它有时对**色情内容表现得过于积极**。

- **发布新的 AI 故事写作与角色扮演模型**：`@dreamgen` 宣布发布专门为**故事写作和角色扮演**设计的新 AI 模型。这些模型基于人工生成的数据进行训练，可以在 ChatML 的扩展版本中使用提示词，旨在实现可控的交互。`@splice0001` 和 `@superking__` 等用户对测试这些模型表现出了极大的热情。

**提到的链接**：

- [LoneStriker/miqu-1-70b-sf-5.5bpw-h6-exl2 · Hugging Face](https://huggingface.co/LoneStriker/miqu-1-70b-sf-5.5bpw-h6-exl2?text=My+name+is+Merve+and+my+favorite): 未找到描述
- [Viralhog Grandpa GIF - Viralhog Grandpa Grandpa Kiki Dance - Discover &amp; Share GIFs](https://tenor.com/view/viralhog-grandpa-grandpa-kiki-dance-kiki-dance-dance-party-gif-12380914): 点击查看 GIF
- [Sheeeeeit GIF - Sheeeeeit - Discover &amp; Share GIFs](https://tenor.com/view/sheeeeeit-gif-14618048145949655995): 点击查看 GIF
- [dreamgen/opus-v1-34b · Hugging Face](https://huggingface.co/dreamgen/opus-v1-34b): 未找到描述
- [Opus V1: Story-writing &amp; role-playing models - a dreamgen Collection](https://huggingface.co/collections/dreamgen/opus-v1-story-writing-and-role-playing-models-65d092a6f8ab7fc669111b31): 未找到描述
- [DreamGen: AI role-play and story-writing without limits](https://dub.sh/opus-v1-guide): 未找到描述
- [Models - Hugging Face](https://huggingface.co/models?search=LoneStriker/opus-v1>): 未找到描述
- [Models - Hugging Face](https://huggingface.co/models?search=LoneStriker/opus-v): 未找到描述

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1209831924646936597) (3 messages): 

- **为代码分类选择合适的模型**：用户 `@yustee.` 正在寻求建议，以选择一个 LLM 来为 RAG 流水线中的查询分类代码相关性。`@yustee.` 正在考虑 **deepseek-coder-6.7B-instruct**，但也欢迎其他推荐。

- **Mistral 下载困境**：用户 `@aamir_70931` 正在寻求在本地下载 **Mistral** 的帮助，但未提供进一步的上下文或后续信息。
  

---

### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1209935716587671612) (163 条消息🔥🔥): 

- **ML 工作流难题与 Mac 之谜**：`@fred.bliss` 讨论了在使用 Mac Studio 建立机器学习项目工作流时面临的挑战，并考虑因架构更简单而使用 `llama.cpp` 代替 **ollama**。尽管他们已经在非 GPU 的 PC 上使用 `llama.cpp` 有一段时间了，但他们对市场推崇 **ollama** 的趋势表示担忧。

- **探索 MLX 和 Zed 作为 VSCode 的替代方案**：`@dirtytigerx` 推荐在处理 TensorFlow/Keras 任务时使用 **MLX**，并称赞了由 Atom 团队开发的文本编辑器 Zed，认为其性能优异且配置极简，优于 Visual Studio Code。此外，他们还对从 Atom 分叉出来的开源项目 [Pulsar](https://github.com/pulsar-edit/pulsar) 表现出了一定兴趣。

- **VsCode 与 Zed 之争**：`@dirtytigerx` 向 `@wbsch` 详细说明了他们更倾向于 Zed 而非 Visual Studio Code 的原因，强调了 Zed 的极简设计和速度。他们还讨论了使用 **Neovim** 作为替代方案的经验，以及 Zed 像 **VSCode** 一样支持远程开发的潜力。

- **微软向开发者导向的转型**：`@dirtytigerx` 和 `@wbsch` 讨论了微软在迎合开发者方面的转型策略，特别提到收购 GitHub 带来的积极进展，以及集成了 **Copilot** 等工具的 VSCode 的普及。

- **扩展推理服务器与 GPU 利用率**：在与 `@etron711` 的对话中，`@dirtytigerx` 就扩展推理服务器以处理大量用户的策略提供了建议，建议先使用较便宜的资源进行原型设计（例如在 runpod 上以 0.80 美元/小时的价格租用 **4090**），作为成本分析的第一步。他们还提醒在与 AWS 等供应商合作时，要注意对 GPU 可用性和 SLA 的依赖。

**提到的链接**：

- [GitHub - raphamorim/rio: A hardware-accelerated GPU terminal emulator focusing to run in desktops and browsers.](https://github.com/raphamorim/rio): 一个专注于在桌面和浏览器中运行的硬件加速 GPU 终端模拟器。- raphamorim/rio
- [GitHub - pulsar-edit/pulsar: A Community-led Hyper-Hackable Text Editor](https://github.com/pulsar-edit/pulsar): 一个由社区主导的、具有高度可定制性的文本编辑器。可以通过在 GitHub 上创建账户来为 pulsar-edit/pulsar 的开发做出贡献。

---

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1209771357957521430) (598 messages🔥🔥🔥): 

- **LM Studio 发布更新**：建议用户从官网手动下载最新的 LM Studio 更新，因为应用内的“Check for Updates”功能目前无法正常工作。 
- **Gemma 模型讨论**：许多用户报告了 Gemma 7B 模型的问题，部分用户称即使在更新后仍存在性能问题。Gemma 2B 模型获得了一些正面反馈，并分享了 Hugging Face 上可用的 Gemma 2B 链接。
- **新版本 LM Studio 的性能问题**：几位用户描述了在 MacOS 上使用最新版本 LM Studio 时出现的性能下降和高 CPU 占用问题，特别是影响到了 Mixtral 7B 模型。
- **在 M1 Mac 上运行 Gemma 7B 需要调整 GPU 滑块**：在 M1 Mac 上运行 Gemma 7B 的用户注意到，将 GPU 滑块调整至“max”后性能有显著提升，尽管部分用户仍感到响应速度较慢。
- **Stable Diffusion 3 发布公告**：Stability.ai 宣布 Stable Diffusion 3 进入早期预览阶段，承诺将提升性能和多主体图像质量。用户表现出浓厚兴趣并讨论了如何申请预览。

**提到的链接**：

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/)：查找、下载并实验本地 LLM。
- [lmstudio-ai/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF)：未找到描述。
- [MSN](https://www.msn.com/en-us/news/technology/chatgpt-has-meltdown-and-starts-sending-alarming-messages-to-users/ar-BB1iDtsE?ocid=entnewsntp&pc=U531&cvid=c72557e2b33e491998be5116a12d196a&ei=31)：未找到描述。
- [Gemma: Introducing new state-of-the-art open models](https://blog.google/technology/developers/gemma-open-models/)：Gemma 是一个轻量级、先进的开源模型系列，基于与 Gemini 模型相同的研究和技术构建。
- [ Stable Diffusion 3 &mdash; Stability AI](https://stability.ai/news/stable-diffusion-3)：宣布 Stable Diffusion 3 进入早期预览版，这是我们功能最强大的文本生成图像模型，在多主体提示词、图像质量和拼写能力方面有显著提升。
- [google/gemma-7b · Hugging Face](https://huggingface.co/google/gemma-7b)：未找到描述。
- [LoneStriker/gemma-2b-GGUF · Hugging Face](https://huggingface.co/LoneStriker/gemma-2b-GGUF)：未找到描述。
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18oi2vd/question_about_system_ram_and_gpu_vram/)：未找到描述。
- [How To Run Stable Diffusion WebUI on AMD Radeon RX 7000 Series Graphics](https://www.youtube.com/watch?v=kw0WT5sDBIY)：你知道可以在 Automatic1111 下通过 Microsoft Olive 启用 Stable Diffusion，从而在 Windows 上通过 Microsoft DirectML 获得显著加速吗？
- [جربت ذكاء إصطناعي غير خاضع للرقابة، وجاوبني على اسئلة خطيرة](https://www.youtube.com/watch?v=to6FI5BseEc&t=61s&ab_channel=marouane53)：YouTube 视频链接，涉及不受限制的 AI 讨论。
- [GitHub - lllyasviel/Fooocus: Focus on prompting and generating](https://github.com/lllyasviel/Fooocus)：专注于提示和生成。通过在 GitHub 上创建账号为 Fooocus 的开发做贡献。
- [Mistral&#039;s next LLM could rival GPT-4, and you can try it now in chatbot arena](https://the-decoder.com/mistrals-next-llm-could-rival-gpt-4-and-you-can-try-it-now-in-chatbot-arena/)：法国 LLM 奇迹 Mistral 正准备发布其下一个语言模型。你现在已经可以在 chat arena 中进行测试。
- [Need support for GemmaForCausalLM · Issue #5635 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/5635)：llama.cpp 的 GitHub Issue，关于支持 GemmaForCausalLM 的讨论。

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1209858429183594556) (149 条消息🔥🔥): 

- **Gemma 模型混淆**：用户在使用 Gemma 模型时遇到问题。`@macaulj` 报告在 GPU 上运行 7b Gemma 模型时出错，而 `@nullt3r` 提到量化模型目前已损坏，正在等待 llama.cpp 的修复。`@yagilb` 建议检查 2B 版本，因为目前流传着许多错误的量化版本，`@heyitsyorkie` 澄清说 LM Studio 需要更新后 Gemma 模型才能正常运行。

- **LM Studio 模型兼容性与错误**：包括 `@swiftyos` 和 `@thorax7835` 在内的多位用户讨论了寻找最适合编程和无审查对话的模型，而 `@bambalejo` 在使用 Nous-Hermes-2-Yi-34B.Q5_K_M.gguf 模型时遇到了故障。LM Studio 0.2.15 版本中一个导致重新生成时输出乱码的已知 Bug 已得到处理，`@heyitsyorkie` 提出了修复建议。

- **图像生成模型讨论**：`@antonsosnicev` 询问是否有类似于 Adobe 生成式填充（generative fill）的图片生成功能，`@swight709` 建议使用 AUTOMATIC1111 的 Stable Diffusion web UI，因为它具备局部重绘（inpainting）和扩图（outpainting）等功能，并强调了其丰富的插件系统，以及它与专注于文本生成的 LM Studio 的区别。

- **硬件和配置挑战**：包括 `@goldensun3ds` 和 `@wildcat_aurora` 在内的用户分享了他们的配置以及运行 Goliath 120B Q6 等大型模型的挑战，讨论了性能与硬件限制（如 VRAM 和系统内存带宽）之间的权衡。

- **多模态 AI 期待**：对话涉及了对能够处理超出当前能力任务的模型的希望。`@drawingthesun` 表达了希望 LLM 和 Stable Diffusion 模型能够交互的愿望，而 `@heyitsyorkie` 暗示未来将出现功能更广泛的多模态模型。

**提到的链接**：

- [👾 LM Studio - 发现并运行本地 LLMs](https://lmstudio.ai/)：查找、下载并实验本地 LLM。
- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html)：未找到描述。
- [掌握生成式 AI 开发栈：实践手册](https://medium.com/@Naykafication/master-modern-generative-ai-stack-practical-handbook-393f446a706c?sk=731eb4d03418970b47143d1818f8c492)：又一篇 AI 文章。有时可能会让人应接不暇。在这份综合指南中，我将简化复杂的生成式 AI 世界……
- [lmstudio-ai/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF)：未找到描述。
- [ImportError: libcuda.so.1: cannot open shared object file](https://stackoverflow.com/questions/54249577/importerror-libcuda-so-1-cannot-open-shared-object-file)：当我直接用 TensorFlow 运行代码时，一切正常。然而，当我在 screen 窗口中运行时，出现了以下错误。ImportError: libcuda.so.1: cannot open shared object fi...
- [macaulj@macaulj-HP-Pavilion-Gaming-Laptop-15-cx0xxx:~$ sudo '/home/macaulj/Downl - Pastebin.com](https://pastebin.com/MVZmiH2Y)：Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储一段时间文本的网站。
- [Big Code Models Leaderboard - Hugging Face Space (由 bigcode 提供)](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)：未找到描述。
- [```json{  "cause": "(Exit code: 1). Please check settings and try loading th - Pastebin.com](https://pastebin.com/2RrDRx3e)：Pastebin.com 是自 2002 年以来排名第一的文本存储工具。
- [Models - Hugging Face](https://huggingface.co/models?search=fitness)：未找到描述。
- [wavymulder/Analog-Diffusion · Hugging Face](https://huggingface.co/wavymulder/Analog-Diffusion)：未找到描述。
- [使用 LM Studio LLMs (AI Chatbot) 测试 Shadow PC Pro (Cloud PC) 并与我的 RTX 4060 Ti PC 进行对比](https://youtu.be/Eaz-H-3FkZg)：自 ChatGPT 发布约一年以来我一直在使用它，并且已经熟练掌握了提示词工程，但我对于“本地”运行 LLM 还是个新手。当...
- [GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)：Stable Diffusion web UI。通过在 GitHub 上创建账号为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。
- [Reddit - 深入探索](https://www.reddit.com/r/LocalLLaMA/comments/189uauo/failed_to_load_model_running_lmstudio/)：未找到描述。

---

### LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1209948313957175357) (4 条消息): 

- **LM Studio v0.2.15 发布公告**：`@yagilb` 发布了 **LM Studio v0.2.15**，带来了令人兴奋的新功能，包括对 **Google Gemma 模型** 的支持、改进的下载管理、对话分支（conversation branching）、GPU 配置工具、全新的 UI 以及多项 Bug 修复。该更新适用于 Mac、Windows 和 Linux，可从 [LM Studio 官网](https://lmstudio.ai)下载，Linux 版本请点击[此处](https://releases.lmstudio.ai/linux/0.2.15/beta/LM_Studio-0.2.15-beta-1.AppImage)。

- **关键 Bug 修复更新**：`@yagilb` 敦促用户从 [LM Studio 官网](https://lmstudio.ai)重新下载 **LM Studio v0.2.15**，因为原始版本中遗漏了一些关键的 Bug 修复。

- **Gemma 模型集成技巧**：`@yagilb` 为 LM Studio 用户分享了推荐的 **Gemma 2b Instruct quant**（量化版）链接，可在 [Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF) 获取，并提醒用户注意 Google 关于 Gemma 服务的相关使用条款。

- **LM Studio v0.2.16 现已上线**：继之前的公告之后，`@yagilb` 通知用户 **LM Studio v0.2.16** 已立即发布。该版本包含了 v0.2.15 更新的所有内容，并额外修复了下载期间不稳定的重新生成（regenerations）和聊天滚动问题。建议已更新至 v0.2.15 的用户尽快更新到 v0.2.16。

**相关链接**：

- [👾 LM Studio - 发现并运行本地 LLMs](https://lmstudio.ai)：查找、下载并实验本地 LLMs
- [lmstudio-ai/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF)：未找到描述

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1209855699446927362) (30 条消息🔥): 

- **本地 LLM 安装问题**：用户 `@maaxport` 询问在获取 LM Studio 后如何配合 AutoGPT 安装本地 LLM，并表示希望将其托管在租用的服务器上。`@senecalouck` 提供了建议，指出设置本地 API endpoint 并更新 `base_url` 即可满足本地运行需求。

- **客户端更新困惑**：`@msz_mgs` 对客户端版本感到困惑，指出尽管有更新的版本，但 0.2.14 仍被识别为最新版。`@heyitsyorkie` 澄清说目前尚不支持应用内更新（in-app updating），需要手动下载并安装。

- **Gemma 模型错误及解决方案**：`@richardchinnis` 在使用 Gemma 模型时遇到问题，经过讨论，`@yagilb` 分享了一个 Hugging Face 上的 [2B 量化模型链接](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF) 以解决这些错误。

- **Gemma 7b 下载可见性排障**：用户 `@adtigerning` 和 `@thebest6337` 讨论了 Gemma 7b 下载文件的可见性问题，指出在 LM Studio 中查看 Google Files 时存在问题。`@heyitsyorkie` 就手动下载及预期的文件存放位置提供了指导。

- **滚动问题的 Bug 报告**：`@drawingthesun` 报告了聊天中的滚动问题，随后 `@heyitsyorkie` 确认这是一个已知 Bug。`@yagilb` 随后宣布在 0.2.16 版本中修复了该 Bug，`@heyitsyorkie` 也确认了该问题已解决。

**相关链接**：

- [👾 LM Studio - 发现并运行本地 LLMs](https://lmstudio.ai)：查找、下载并实验本地 LLMs
- [lmstudio-ai/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF)：未找到描述

  

---

### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1209773656364818452) (130 条消息🔥🔥): 

- **财报引发 Nvidia 焦虑**：`@nink1` 分享了他们对 Nvidia 财报的焦急期待，因为他们将毕生积蓄投资在了 Nvidia 产品上，特别是 3090 显卡。尽管 `@heyitsyorkie` 调侃可能会有“大牛股 (big stonks)”收益，但 `@nink1` 澄清说他们的投资是在硬件上，而不是股票。

- **解读闪存阵列的价值**：`@wolfspyre` 思考了三个 30Tb 闪存阵列（每个支持 1M iops）的潜在应用，引发了与 `@heyitsyorkie` 关于盗版以及海盗生活的弊端（如坏血病和恶劣的工作条件）的幽默交流。

- **VRAM 与用于 AI 渲染的新 GPU 之争**：`@freethepublicdebt` 询问在运行 Mixtral8x7 等大模型时，使用多个廉价 GPU 来增加 VRAM 的价值。`@heyitsyorkie` 提供了 GPU 规格链接，并建议虽然更多的 VRAM 是关键，但 GPU 性能也不容忽视，有时像 RTX 3090 这样单张强大的显卡就足够了。

- **Tesla P40 在预算受限的情况下受到关注**：`@wilsonkeebs` 和 `@krypt_lynx` 等参与者讨论了 Tesla P40 等旧款 GPU 用于 AI 任务的可行性，权衡了它们的易获得性与相比 RTX 3090 等新型替代品较慢的性能。

- **旧款 Nvidia 显卡的 AI 能力受到质疑**：`@exio4` 和 `@bobzdar` 等几位用户分享了他们关于使用旧款 Nvidia GPU 执行 AI 任务的经验和测试结果，显示出新显卡的进步对 AI 建模和推理的性能提升贡献巨大。

**提到的链接**：

- [未找到标题](https://www.amazon.com/Dell-Tesla-K80-Accelerator-Refurbished/dp/B07GJ45V3D/ref=asc_df_B07GJ45V3D/?tag=hyprod-20&linkCode=df0&hvadid=309751315916&hvpos=&hvnetw=g&hvrand=15721617830425222448&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1024339&hvtargid=pla-624228729967&psc=1&mcid=d6b00d04180c3502bc1b76aa12665646&tag=&ref=&adgrpid=67183599252&hvpone=&hvptwo=&hvadid=309751315916&hvpos=&hvnetw=g&hvrand=15721617830425222448&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1024339&hvtargid=pla-624228729967)：未找到描述
- [在 Apple Neural Engine 上部署 Transformers](https://machinelearning.apple.com/research/neural-engine-transformers)：我们在 Apple 每年构建的机器学习 (ML) 模型中，有越来越多的模型正在部分或全部采用 [Transformer…
- [MAG Z690 TOMAHAWK WIFI](https://www.msi.com/Motherboard/MAG-Z690-TOMAHAWK-WIFI)：微星 MAG Z690 TOMAHAWK WIFI 搭载 Intel 第 12 代 Core 处理器，具备性能核心规格，经久耐用。通过 Core boost、Memory B... 进行了更好的性能调优。
- [Have You GIF - Have You Ever - Discover &amp; Share GIFs](https://tenor.com/view/have-you-ever-condidered-piracy-gif-10055735)：点击查看 GIF
- [2023 年深度学习最佳 GPU —— 深度分析](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/.): 在这里，我提供了用于深度学习/机器学习的 GPU 深度分析，并解释了适合您的使用场景和预算的最佳 GPU。
- [2023 年深度学习最佳 GPU —— 深度分析](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/)): 在这里，我提供了用于深度学习/机器学习的 GPU 深度分析，并解释了适合您的使用场景和预算的最佳 GPU。
- [NVIDIA GeForce RTX 2060 SUPER 规格](https://www.techpowerup.com/gpu-specs/geforce-rtx-2060-super.c3441): NVIDIA TU106, 1650 MHz, 2176 Cores, 136 TMUs, 64 ROPs, 8192 MB GDDR6, 1750 MHz, 256 bit
- [NVIDIA GeForce RTX 3090 规格](https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622): NVIDIA GA102, 1695 MHz, 10496 Cores, 328 TMUs, 112 ROPs, 24576 MB GDDR6X, 1219 MHz, 384 bit

  

---

### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1209773154210283570) (266 messages🔥🔥): 

- **Gemma Quants 表现存疑**：用户报告 Google 的 Gemma 模型表现参差不齐，发现 [`7b-it` 量化版本经常输出乱码](https://twitter.com/ggerganov/status/1760418864418934922)，而 [`2b-it` 量化版本似乎很稳定且运行良好](https://huggingface.co/LoneStriker/gemma-2b-it-GGUF)。`@drawless111` 强调全精度模型对于达到基准测试结果是必要的，并建议较小的 (1-3B) 模型需要更精确的 Prompt 和设置。
- **LM Studio 持续改进**：`@yagilb` 发布了新的 LM Studio 下载版本，包含重大 Bug 修复，特别是解决了[此处](https://lmstudio.ai)提到的重新生成（regenerate）功能和多轮对话的问题。`@yagilb` 还澄清了重新生成问题与模型无关，而是由于糟糕的量化版本导致的；团队正在研究如何简化功能完备模型的下载流程。
- **Gemma 7B 体积问题说明**：用户讨论了 Google Gemma `7b-it` 模型体积庞大的原因，指出其缺乏量化且内存需求巨大。值得注意的是，llama.cpp 目前与 [Gemma 存在兼容性问题](https://github.com/ggerganov/llama.cpp/issues/5635)，预计很快会得到解决。
- **提升性能的用户友好预设**：用户一致认为需要正确的 Preset（预设）才能从 Gemma 模型中获得良好结果，`@pandora_box_open` 强调了特定预设的必要性，以避免输出质量低下。
- **LM Studio 确认可用的 GGUF**：`@yagilb` 推荐了一个他们为 LM Studio 量化并测试过的 [2B IT Gemma](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF) 模型，并计划上传 7B 版本。`@issaminu` 确认该 2B 模型可以工作，但智能程度不如功能更完备的 7B 模型。

**提到的链接**：

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/)：发现、下载并实验本地 LLM
- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai)：发现、下载并实验本地 LLM
- [asedmammad/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/asedmammad/gemma-2b-it-GGUF)：未找到描述
- [Thats What She Said Dirty Joke GIF - Thats What She Said What She Said Dirty Joke - Discover &amp; Share GIFs](https://tenor.com/view/thats-what-she-said-what-she-said-dirty-joke-joke-laugh-gif-15661968)：点击查看 GIF
- [HuggingChat](https://huggingface.co/chat/)：让社区最好的 AI 聊天模型对所有人可用。
- [```json{  &quot;cause&quot;: &quot;(Exit code: 1). Please check settings and try loading th - Pastebin.com](https://pastebin.com/2RrDRx3e)：Pastebin.com 自 2002 年以来是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。
- [google/gemma-2b-it · Hugging Face](https://huggingface.co/google/gemma-2b-it)：未找到描述
- [lmstudio-ai/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF)：未找到描述
- [LoneStriker/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/LoneStriker/gemma-2b-it-GGUF)：未找到描述
- [google/gemma-7b · Why the original GGUF is quite large ?](https://huggingface.co/google/gemma-7b/discussions/11)：未找到描述
- [google/gemma-7b-it · Hugging Face](https://huggingface.co/google/gemma-7b-it)：未找到描述
- [Need support for GemmaForCausalLM · Issue #5635 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/5635)：前提条件 在提交 Issue 之前，请先自行回答以下问题。我正在运行最新的代码。由于开发非常迅速，目前还没有标记版本。我...
- [Add `gemma` model by postmasters · Pull Request #5631 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5631)：该架构中有几点：共享输入和输出 Embedding 参数。Key 长度和 Value 长度不是由 n_embd 派生的。关于模型的更多信息可以在...找到。

  

---

### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1209778697045807144) (97 条消息🔥🔥): 

- **通过长上下文数据工程进行扩展**：`@gabriel_syme` 对名为 ["Long-Context Data Engineering"](https://github.com/FranxYao/Long-Context-Data-Engineering) 的 GitHub 仓库表示兴奋，提到了将语言模型扩展到 128K 上下文的数据工程技术实现。
- **128K 上下文模型的 VRAM 需求**：在关于 7B 模型在 128K 上下文下的 VRAM 需求查询中，`@teknium` 澄清说需要超过 600GB。
- **Tokenization 查询与考量**：`@vatsadev` 提到 GPT-3 和 GPT-4 的 Tokenizer 可以在 tiktoken 中找到，并引用了 Andrej Karpathy 的相关视频，但未提供直接链接。
- **Token 压缩挑战**：`@elder_plinius` 提出了在尝试将《谋杀绿脚趾》（Big Lebowski）剧本放入上下文限制内时的 Token 压缩问题，引发了与 `@vatsadev` 和 `@blackl1ght` 关于 Tokenizer 以及 [OpenAI tokenizer playground](https://gpt-tokenizer.dev) 服务器行为的讨论，最终观察到为什么压缩文本能被 ChatGPT 接受而原始文本却不行的原因。
- **在较低 VRAM 上进行长上下文推理**：`@blackl1ght` 分享了他们在 V100 32GB 上仅用 28GB VRAM 就在 64K 上下文下对 Mistral 7B 和 Solar 10.7B 进行了推理，这引发了与 `@teknium` 和 `@bloc97` 关于该方法可行性以及大型模型中 KV Cache 容量和 Offloading 的讨论。

**提到的链接**：

- [gpt-tokenizer playground](https://gpt-tokenizer.dev)：未找到描述
- [Aran Komatsuzaki (@arankomatsuzaki) 的推文](https://x.com/arankomatsuzaki/status/1760495656014405900?s=20)：Microsoft Research 发布 LongRoPE：将 LLM 上下文窗口扩展至 200 万 Token 以上 https://arxiv.org/abs/2402.13753
- [Pliny the Prompter 🐉 (@elder_plinius) 的推文](https://x.com/elder_plinius/status/1756436779056742863?s=46&t=Nf3Zw7IH6o_5y_YpAL5gew)：《谋杀绿脚趾》剧本通常无法完全放入 GPT-4 的上下文限制中，但在通过 myln 处理文本后，它可以了！
- [GitHub - FranxYao/Long-Context-Data-Engineering: Implementation of paper Data Engineering for Scaling Language Models to 128K Context](https://github.com/FranxYao/Long-Context-Data-Engineering)：论文《Data Engineering for Scaling Language Models to 128K Context》的实现 - FranxYao/Long-Context-Data-Engineering

  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1209768306899288087) (16 条消息🔥): 

- **神秘的 Minecraft 生物查询**：用户 `@teknium` 询问了 Minecraft 中某种特定生物的存在。`@nonameusr` 的回答简短有力：“一直都有”。

- **在 RAG 中探索自我反思 AI**：`@pradeep1148` 分享了一个名为“使用 LangGraph 进行 Self RAG”的 [YouTube 视频](https://www.youtube.com/watch?v=Eb7QF1nDWGU)链接，该视频建议自我反思可以增强检索增强生成（RAG）模型。

- **艺术图像生成的初学者指南请求**：用户 `@blackblize` 询问非专业人士是否可行在显微镜照片上训练模型以用于艺术目的，并寻求相关指导。

- **AI 生成 Minecraft 视频的进展**：`@afterhoursbilly` 分析了 AI 如何理解 Minecraft 视频中的物品栏 UI，而 `@_3sphere` 补充说，虽然 AI 生成的图像乍看之下很正常，但在仔细观察时会发现不准确之处。

- **讨论 Nous 模型的头像生成**：针对 `@stoicbatman` 对 Nous 模型头像生成的好奇，`@teknium` 提到使用了 DALL-E，随后通过 Midjourney 进行 img2img 处理。

**提到的链接**：

- [Gemma Google 的开源 SOTA 模型](https://www.youtube.com/watch?v=953U3FxHF-Q)：Gemma 是一个轻量级、最先进的开放模型系列，基于创建 Gemini 模型所使用的相同研究和技术构建。由 Google 开发...
- [使用 LangGraph 进行 Self RAG](https://www.youtube.com/watch?v=Eb7QF1nDWGU)：自我反思可以增强 RAG，从而纠正低质量的检索或生成。最近的几篇论文都聚焦于这一主题，但实现起来...
- [BitDelta: Your Fine-Tune May Only Be Worth One Bit](https://www.youtube.com/watch?v=T_dYzuv4N70)：大语言模型（LLM）通常分两个阶段训练：在大规模互联网数据集上进行预训练，以及针对下游任务进行微调。鉴于...

  

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1209850993693425684) (38 条消息🔥): 

- **Google 发布 Gemma**：`@burnytech` 分享了 `@sundarpichai` 的一条推文链接，宣布推出 **Gemma**，这是一个轻量级且开源的模型系列，提供 2B 和 7B 两种尺寸。[Sundar Pichai 的推文](https://fxtwitter.com/sundarpichai/status/1760288967352598843?t=dOvFXh4oPnnAZxjouwfMyQ&s=19) 表达了对全球可用性的兴奋，并鼓励在从开发者笔记本电脑到 Google Cloud 的各种平台上使用 Gemma 进行创作。
  
- **Gemini 1.5 讨论进行中**：`@shashank.f1` 邀请用户参加关于 **Gemini 1.5** 的讨论，并提到了之前关于 A-JEPA AI 模型的会议。`@ldj` 指出，该模型与 Meta 或 Yann Lecun 无关。

- **OpenAI 的 LLama 被复现版本超越**：`@euclaise` 和 `@teknium` 讨论了 OpenAI LLama 的一个复现版本如何表现优于原版，这增加了人们对模仿模型能力的兴趣。

- **人类知识导航**：`@.benxh` 提供了一种导航人类知识和能力分类的方法，建议建立一个包含所有可能领域的结构化列表，并引导用户参考 [美国国会图书馆](https://id.loc.gov/authorities/subjects.html) 以获取详尽示例。

- **Microsoft 将 LLM 推向新长度**：`@main.ai` 链接了 `@_akhaliq` 的一条推文，内容关于 Microsoft 的 LongRoPE。这是一种将 LLM 上下文窗口扩展到 200 万个 token 以上的技术，可以说彻底改变了 LLaMA 和 Mistral 等模型的长文本处理能力。该推文强调了这一进步，同时也没有忽视在原始上下文窗口大小下的性能。

**提到的链接**：

- [来自 Sundar Pichai (@sundarpichai) 的推文](https://fxtwitter.com/sundarpichai/status/1760288967352598843?t=dOvFXh4oPnnAZxjouwfMyQ&s=19)：介绍 Gemma - 一个轻量级、同类领先的开源模型系列，采用与构建 Gemini 模型相同的研究和技术。展示了强大的性能...
- [加入 hedwigAI Discord 服务器！](https://discord.gg/F4FfcQw3?event=1209440306404139008)：查看 Discord 上的 hedwigAI 社区 - 与其他 50 名成员一起交流，享受免费的语音和文字聊天。
- [来自 AK (@_akhaliq) 的推文](https://x.com/_akhaliq/status/1760499638056910955)：Microsoft 推出 LongRoPE，将 LLM 上下文窗口扩展至 200 万个 token 以上。大上下文窗口是大语言模型 (LLM) 中一个理想的特性。然而，由于高昂的微调成本...
- [来自 Emad (@EMostaque) 的推文](https://x.com/EMostaque/status/1760660709308846135?s=20)：@StabilityAI 一些说明：- 这使用了一种新型的 Diffusion Transformer（类似于 Sora），并结合了 Flow Matching 和其他改进。- 这利用了 Transformer 的改进，并且可以...
- [Library of Congress Subject Headings - LC Linked Data Service: Authorities and Vocabularies | Library of Congress](https://id.loc.gov/authorities/subjects.html)：未找到描述
- [benxh/us-library-of-congress-subjects · Hugging Face 数据集](https://huggingface.co/datasets/benxh/us-library-of-congress-subjects)：未找到描述
- [A-JEPA AI 模型：从 .wav / .mp3 文件或音频频谱图中解锁语义知识](https://youtu.be/FgcN62LFzIU)：🌟 解锁 AI 从音频中学习的力量！🔊 观看与 Oliver, Nevil, Ojasvita, Shashank, Srikanth 和 N... 深入讨论 A-JEPA 方法。
- [JARVIS/taskbench at main · microsoft/JARVIS](https://github.com/microsoft/JARVIS/tree/main/taskbench)：JARVIS，一个将 LLM 与 ML 社区连接的系统。论文：https://arxiv.org/pdf/2303.17580.pdf - microsoft/JARVIS
- [JARVIS/easytool at main · microsoft/JARVIS](https://github.com/microsoft/JARVIS/tree/main/easytool)：JARVIS，一个将 LLM 与 ML 社区连接的系统。论文：https://arxiv.org/pdf/2303.17580.pdf - microsoft/JARVIS

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1209768012928778241) (419 messages🔥🔥🔥): 

- **Gemma vs Mistral 对决**：推文正在流传，比较 [Google Gemma](https://fxtwitter.com/jxmnop/status/1760487700145041749) 与 Mistral 的 LLM，声称即使经过*几个小时的测试*，Gemma 的表现也没有超过 Mistral 的 7B 模型，尽管它比 *Llama 2* 更好。
- **关于 Gemma 指令遵循的讨论**：`@big_ol_tender` 注意到，对于 Nous-Mixtral 模型，**将指令放在命令末尾似乎比放在开头更有效**，这引发了关于命令格式的讨论。
- **不同服务上 VM 的速度**：建议 `@mihai4256` 尝试 VAST，以获得比 Runpod 更快、更具性价比的 VM，而另一位用户指出，尽管存在速度问题，Runpod 的 UX 更好。`@lightvector_` 随后报告说，今天*所有*供应商似乎都很慢。
- **对使用加密货币支付 GPU 租用时间的关注**：`@protofeather` 询问哪些平台提供使用加密货币购买 GPU 时间的服务，得到的建议是 **Runpod 和 VAST**，不过有人澄清说 Runpod 需要 *Crypto.com* 的 KYC 注册。
- **Gemma 可能获得的 Axolotl 支持**：`@gryphepadar` 使用 Axolotl 对 Gemma 进行了全量微调（full finetune），指出 Gemma 的大小似乎是 *10.5B*，因此比 Mistral 需要*多得多的 VRAM*。此外，用户分享了他们在各种设置和 DPO 数据集上的困难和成功经验。

**提到的链接**：

- [来自 Aaditya Ura (Ankit) (@aadityaura) 的推文](https://x.com/aadityaura/status/1760305308927426903?s=20)：来自 @GoogleDeepMind @GoogleAI 的新模型 Gemma 在医疗/保健领域的基准测试中没有表现出强大的性能。@GoogleDeepMind 的 Gemma 与 Mistral 的侧向对比...
- [来自 anton (@abacaj) 的推文](https://fxtwitter.com/abacaj/status/1760393505153679369?s=20)：在尝试了 Gemma 几个小时后，我可以说明它不会取代我的 Mistral 7B 模型。它比 Llama 2 更好，但令人惊讶的是并不比 Mistral 强。Mistral 团队确实做出了一个甚至连 Google 都...
- [伤心猫咪 GIF - Sad Cat - 发现并分享 GIF](https://tenor.com/view/sad-cat-gif-26527456)：点击查看 GIF
- [LMSys Chatbot Arena 排行榜 - lmsys 的 Hugging Face Space](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)：未找到描述
- [indischepartij/MiniCPM-3B-Hercules-v2.0 · Hugging Face](https://huggingface.co/indischepartij/MiniCPM-3B-Hercules-v2.0)：未找到描述
- [来自 TokenBender (e/xperiments) (@4evaBehindSOTA) 的推文](https://fxtwitter.com/4evaBehindSOTA/status/1760512560238109167?s=20)：根据我目前的测试，在通用微调或推理方面可以忽略 Gemma。然而，稍后可能会探索印度语系（indic language）的探索和特定用例测试。现在回到构建中...
- [未找到标题](https://ai.google.dev/gemma/prohibited_use_policy)：未找到描述
- [Models - Hugging Face](https://huggingface.co/models?other=gemma&sort=trending&search=google)：未找到描述
- [Reddit - 深入了解任何事物](https://www.reddit.com/r/OpenAI/comments/1avwdi4/wtf_chat_gpt_starts_talking_crazy_out_of_nowhere/)：未找到描述
- [eleutherai](https://wandb.ai/eleutherai/rnn-hermes/runs/rptfh8c7)：Weights & Biases，机器学习开发者工具
- [新手 LLM 训练指南](https://rentry.org/llm-training)：由 Alpin 编写，灵感来自 /hdg/ 的 LoRA 训练 rentry。本指南正在缓慢更新中。我们已经迁移到了 Axolotl 训练器。基础知识、Transformer 架构、训练基础、预训练...
- [GitHub - facebookresearch/diplomacy_cicero: Cicero 的代码，这是一个通过开放域自然语言协商玩《外交》（Diplomacy）游戏的 AI Agent。](https://github.com/facebookresearch/diplomacy_cicero?tab=readme-ov-file)：Cicero 的代码，这是一个通过开放域自然语言协商玩《外交》（Diplomacy）游戏的 AI Agent。 - facebookresearch/diplomacy_cicero
- [Neuranest/Nous-Hermes-2-Mistral-7B-DPO-BitDelta 在 main 分支](https://huggingface.co/Neuranest/Nous-Hermes-2-Mistral-7B-DPO-BitDelta/tree/main)：未找到描述
- [BitDelta](https://fasterdecoding.github.io/BitDelta/)：未找到描述
- [GitHub - FasterDecoding/BitDelta](https://github.com/FasterDecoding/BitDelta/tree/main)：通过创建一个账户为 FasterDecoding/BitDelta 的开发做出贡献。
- [由 monk1337 添加 Google 的 Gemma 模型 · Pull Request #1312 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1312)：添加 Gemma 模型配置 https://huggingface.co/google/gemma-7b 测试通过并正常工作！

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1209780678795726868) (9 messages🔥): 

- **自定义 Tokenizer 训练查询**：`@ex3ndr` 询问了训练完全自定义 Tokenizer 的可能性及其存储方式。`@nanobitz` 进行了回复，并要求澄清该任务的最终目标。
- **Nous-Hermes-2-Mistral-7B-DPO-GGUF 性能查询**：`@natefyi_30842` 询问了新款 Nous-Hermes-2-Mistral-7B-DPO-GGUF 与其 Solar 版本之间的性能对比，`@emraza110` 对其在准确回答特定测试问题方面的能力发表了评论。
- **Mixtral 模型显存溢出错误**：`@iamcoming5084` 提出了在处理 Mixtral 8x7b 模型时遇到的显存溢出 (Out-of-Memory) 错误问题。
- **影响准确性的微调参数**：`@iamcoming5084` 寻求关于在微调 Mixtral 8x7b 和 Mistral 7B 过程中可能影响准确性的参数建议，并艾特了 `@688549153751826432` 和 `@470599096487510016` 以获取反馈。
- **大型模型的托管与推理**：`@jacobi` 讨论了使用 OpenAI API 端点托管 Mixtral 8x7b 模型所面临的挑战并寻求策略，提到了 tabbyAPI 和 llama-cpp 等工具。
- **Nous-Hermes-2-Mistral-7B-DPO 推理代码错误**：`@qtnx` 指出了 Huggingface 上 Nous-Hermes-2-Mistral-7B-DPO 推理代码部分的错误，并提供了修正后的代码版本。[Nous-Hermes-2-Mistral-7B-DPO 的推理代码](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO)。

**提到的链接**：

- [Welcome Gemma - Google’s new open LLM](https://huggingface.co/blog/gemma)：未找到描述
- [NousResearch/Nous-Hermes-2-Mistral-7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO)：未找到描述

  

---


### Nous Research AI ▷ #[collective-cognition](https://discord.com/channels/1053877538025386074/1154961277748256831/1209995589236822117) (3 messages): 

- **对 Heroku 的简短差评**：`@bfpill` 对 Heroku 表达了负面情绪，直言 "screw heroku"。其挫败感表达得很简练，但未作详细说明。
- **亲切的回应**：`@adjectiveallison` 进行了回复，似乎认可了这种情绪，但指出 "我不认为那是重点，不过也行"。具体的争议点仍不明确。
- **共识还是巧合？**：`@bfpill` 回复道 "很高兴我们达成共识"，但在缺乏上下文的情况下，不确定是否真的达成了共识，还是这只是句玩笑话。
  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1209791740513624094) (3 messages): 

- **因宠物生病导致模型更新延迟**：`@qnguyen3` 为模型更新和完成进度较慢表示道歉，将延迟归因于他们的猫生病了。
- **邀请通过私信进行协作**：`@qnguyen3` 邀请成员如果需要就项目相关事宜进行联系，可以直接发送私信。

### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1209773000115752972) (101 messages🔥🔥): 

- **对 lm eval 的困惑**：`@lee0099` 疑惑为什么 lm eval 似乎只为 runpod 设置，促使 `@hailey_schoelkopf` 澄清 lm eval 与 llm-autoeval 不同，并指向 Open LLM Leaderboard 的 HF spaces 页面以获取详细说明和命令行参数。

- **关于模型环境影响的讨论**：`@gaindrew` 推测根据模型防止或贡献的净碳排放量对模型进行排名。承认准确性将是一个挑战，对话在没有进一步探索或链接的情况下结束。

- **loubb 的优化器问题**：`@loubb` 展示了基于 Whisper 模型训练时异常的 loss 曲线，并与 `@ai_waifu` 和 `@lucaslingle` 等人讨论了与优化器参数相关的潜在原因。

- **Google 发布 Gemma**：`@sundarpichai` 宣布了 Gemma，一个新的模型家族，引发了 `@lee0099` 和 `@.undeleted` 等用户关于 Gemma 是否比 Mistral 等现有模型有显著改进的辩论。

- **关于模拟人类体验的理论讨论**：`@rallio.` 与 `@sparetime.` 和 `@fern.bear` 就模拟人类认知的理论可能性进行了详细讨论。对话范围从建模人类情感和记忆的复杂性，到如何利用 GPT-4 创建一致的合成人类体验。

**提到的链接**：

- [Sundar Pichai (@sundarpichai) 的推文](https://x.com/sundarpichai/status/1760288967352598843?s=46)：介绍 Gemma - 一个轻量级、同类领先的开放模型家族，采用与创建 Gemini 模型相同的研究和技术构建。展示了强大的性能...
- [PropSegmEnt: A Large-Scale Corpus for Proposition-Level Segmentation and Entailment Recognition](https://arxiv.org/abs/2212.10750)：广泛研究的自然语言推理 (NLI) 任务要求系统识别一段文本是否在文本上蕴含另一段文本，即其全部含义是否可以被...
- [Everything WRONG with LLM Benchmarks (ft. MMLU)!!!](https://youtu.be/74Uo2HU8HBo?si=D9bHCZZrnIRX9skj)：🔗 链接 🔗 当 Benchmark 成为目标：揭示大语言模型排行榜的敏感性 https://arxiv.org/pdf/2402.01781.pdf ❤️ 如果你想...

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1209816741421654077) (305 messages🔥🔥): 

- **Groq 试图超越 Mistral**：`@philpax` 分享了一篇文章，强调 AI 硬件初创公司 [Groq](https://www.semianalysis.com/p/groq-inference-tokenomics-speed-but) 在其推理 API 上展示了 **Mistral Mixtral 8x7b** 模型的惊人演示，实现了高达 4 倍的吞吐量，且收费不到 Mistral 价格的三分之一。性能提升有利于 Chain of Thought 的实际可用性，以及代码生成和实时模型应用对低延迟的需求。

- **关于 Gemma 模型参数量误导的担忧**：频道中的讨论提出了参数量误导的问题，例如 "gemma-7b" 实际上包含 85 亿个参数，并建议 "7b" 等模型分类应严格意味着最多 79.9 亿个参数。

- **探索 LLM 数据和计算效率**：`@jckwind` 发起了关于 LLM 数据和计算效率的对话，指出它们需要大量数据且构建的世界模型不一致。一张分享的图表暗示 LLM 可能在双向学习方面存在困难，引发了辩论，并激发了关于大 context windows 或好奇心驱动的学习机制是否能解决这些低效问题的思考。

- **讨论新论文和研究方向**：分享了各种论文和研究课题，包括一篇关于 LLM 对抗性攻击的论文 [`@0x_paws`](https://arxiv.org/abs/2402.14020)，以及另一篇提出可编程梯度信息 (PGI) 概念以应对深度网络中数据丢失的论文 [`@jckwind`](https://arxiv.org/abs/2402.13616)。

- **模型优化和攻击面的更新**：`@benjamin_w` 提到 [PyTorch 2.2 的 SDPA](https://x.com/tri_dao/status/1760458183066472556) 和 FlashAttention v2.5.5 现在支持特定的 head dimensions，允许在消费级 GPU 上微调 Gemma 模型，扩大了优化和使用这些 LLM 的可及性。此外，分享了一篇讨论 LLM 广泛对抗性攻击面的论文，包括具有编码能力的模型预训练以及词汇表中 "glitch" token 的存在 [`@0x_paws`](https://arxiv.org/abs/2402.14020)。

**提到的链接**：

- [Coercing LLMs to do and reveal (almost) anything](https://arxiv.org/abs/2402.14020)：最近的研究表明，针对大语言模型（LLMs）的对抗性攻击可以“越狱”模型，使其发表有害言论。在这项工作中，我们认为对抗性攻击的频谱...
- [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)：当今的深度学习方法专注于如何设计最合适的目标函数，使模型的预测结果能够最接近真实值（ground truth）。同时，一个合适的...
- [Groq Inference Tokenomics: Speed, But At What Cost?](https://www.semianalysis.com/p/groq-inference-tokenomics-speed-but)：比 Nvidia 更快？剖析其背后的经济学
- [Spectral State Space Models](https://arxiv.org/abs/2312.06837)：本文研究了具有长距离依赖关系的预测任务的序列建模。我们提出了一种基于学习线性动力系统的新型状态空间模型（SSMs）公式，该公式具有...
- [Feist Publications, Inc., v. Rural Telephone Service Co. - Wikipedia](https://en.wikipedia.org/wiki/Feist_Publications,_Inc.,_v._Rural_Telephone_Service_Co.)：未找到描述
- [Gemma: Introducing new state-of-the-art open models](https://blog.google/technology/developers/gemma-open-models/)：Gemma 是一个轻量级、最先进的开放模型系列，采用与构建 Gemini 模型相同的研究和技术开发而成。
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)：Transformer 神经序列模型中使用的多头注意力层（Multi-head attention layers）是 RNN 的强大替代方案，用于在序列内部和序列之间传递信息。虽然训练这些层通常...
- [Paper page - LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://huggingface.co/papers/2402.13753)：未找到描述
- [Tweet from NVIDIA (@nvidia)](https://x.com/nvidia/status/1760331965994020946?s=20)：今天宣布，我们将作为发布合作伙伴与 @Google 合作交付 Gemma，这是一个经过优化的模型系列，使用户能够仅使用桌面级 RTX GPU 即可开发 #LLMs...
- [Tweet from Tri Dao (@tri_dao)](https://x.com/tri_dao/status/1760458183066472556)：FlashAttention v2.5.5 现在支持在消费级 GPU 上进行 head dim 256 的反向传播。希望这能让微调 Gemma 模型变得更容易
- [Lecture 20 - Efficient Transformers | MIT 6.S965](https://youtu.be/RGUCmX1fvOE?si=wcs1MDNbon1URKsO)：第 20 讲介绍了高效 Transformer。关键词：Transformer。幻灯片：https://efficientml.ai/schedule/---------------------------------------------------...

  

---

### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1209826662418751528) (43 messages🔥): 

- **多语言模型的内部语言受到质疑**：`@butanium` 分享了一条 [Twitter 帖子](https://twitter.com/cervisiarius/status/1759989584371298554?t=fDN-bfsJDhWP4lfAjSOnHA&s=19)，暗示模型在执行非英语任务时会“用英语思考”。他们提供了一篇论文和 [GitHub 仓库](https://github.com/epfl-dlab/llm-latent-language) 的见解，指出在分析模型内部语言使用时，Logit Lens 与 Tuned Lens 的不同之处。
- **Llama 模型的 Tuned Lens 可用性**：`@mrgonao` 澄清了关于 Llama 模型内部是否使用英语的调查，通过使用在其上训练的 Tuned Lens，并提供了一个 [Hugging Face Space](https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens/meta-llama) 来检查可用资源。
- **探索 Llama 模型的多语言能力**：`@mrgonao` 报告称，由于 13b 规模模型的不完整性以及提供的仓库中缺少某些任务的 Notebook，在所有语言中运行实验存在困难。他们表示一旦问题解决，愿意运行更多实验。
- **正在考虑为 Llama 模型开发中文 Lens**：响应 `@stellaathena` 的建议，`@mrgonao` 考虑使用易于获取的中文数据集创建一个用于中文分析的 Lens，随后表示该 Lens 的训练已经开始。
- **模型 Unlearning 技术讨论**：`@millander` 分享了一篇关于 [LLM Unlearning 的新综述论文](https://arxiv.org/abs/2402.08787) 的链接，频道内未对论文内容进行进一步讨论。

**Links mentioned**:

- [Rethinking Machine Unlearning for Large Language Models](https://arxiv.org/abs/2402.08787): 我们探索了大语言模型（LLM）领域的 Machine Unlearning (MU)，即 LLM Unlearning。该计划旨在消除不良数据的影响（例如敏感或非法数据...）
- [phoeniwwx/tuned_lens_q · Hugging Face](https://huggingface.co/phoeniwwx/tuned_lens_q): 未找到描述
- [AlignmentResearch/tuned-lens at main](https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens/meta-llama): 未找到描述
- [shjwudp/chinese-c4 · Datasets at Hugging Face](https://huggingface.co/datasets/shjwudp/chinese-c4): 未找到描述
- [GitHub - epfl-dlab/llm-latent-language: Repo accompanying our paper &quot;Do Llamas Work in English? On the Latent Language of Multilingual Transformers&quot;.](https://github.com/epfl-dlab/llm-latent-language): 论文 &quot;Do Llamas Work in English? On the Latent Language of Multilingual Transformers&quot; 的配套仓库。 - epfl-dlab/llm-latent-language
- [srgo - Overview](https://github.com/SrGo): srgo 有一个可用仓库。在 GitHub 上关注其代码。

---

### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1209852424102215741) (64 messages🔥🔥): 

- **探索 Few-Shots Context 实验**：`@baber_` 提到，如果将 Few-Shots Context 和后续内容格式化为交替的 "user" 和 "assistant" 轮次，Instruct tuned 模型可能会表现得更好，尽管他们尚未对此进行测试。

- **OOM 后 GPU 显存未释放**：`@pminervini` 在 Colab 上使用 `evaluator.simple_evaluate` 时遇到了显存溢出 (OOM) 问题，即使尝试了垃圾回收 (`gc.collect()`) 也无法解决，GPU 显存显示仍被占用。该问题需要重启运行时才能解决，`@hailey_schoelkopf` 和 `@baber_` 讨论并提出了潜在的修复建议，并提供了一个用于复现的 Colab 链接：[Evaluate OOM Issue](https://colab.research.google.com/drive/1u5MoN-QUfdNJXilFJAyJaGY1HlYWnfwX?usp=sharing)。

- **LM-Harness Logits 支持障碍**：`@dsajlkdasdsakl` 遇到了一个问题：在 LM-Harness 中本地运行带有 Log Likelihood 的任务时正常，但像 GPT 这样基于 API 的模型会产生 "No support for logits" 错误，而像 gsm8k 这样的预定义任务则运行顺畅。`@hailey_schoelkopf` 澄清这是因为大多数 API 提供商不支持 Logits，建议将任务转换为 Generative 格式，并更新错误消息以提高清晰度。

- **Gemma 模型评估问题**：用户 `@vraychev`、`.rand0mm` 和 `@ilovescience` 报告了在 lm-evaluation-harness 中评估 Gemma-7b 模型时的问题。`@hailey_schoelkopf` 承认存在 Bug，并提供了修复步骤（包括添加 BOS token），并指导用户如何让 Gemma 7b 配合 Flash Attention (`attn_implementation="flash_attention_2"`) 工作。此外还提到了 Transformers 4.38 中可能存在的问题以及升级 Torch 版本的必要性。

**提到的链接**：

- [Google Colaboratory](https://colab.research.google.com/drive/1u5MoN-QUfdNJXilFJAyJaGY1HlYWnfwX?usp=)：无描述
- [Google Colaboratory](https://colab.research.google.com/drive/1u5MoN-QUfdNJXilFJAyJaGY1HlYWnfwX?usp=sharing)：无描述
- [lm-evaluation-harness/lm_eval/evaluator.py at c26a6ac77bca2801a429fbd403e9606fd06e29c9 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c26a6ac77bca2801a429fbd403e9606fd06e29c9/lm_eval/evaluator.py#L190)：一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [lm-evaluation-harness/lm_eval/api/model.py at ba5cdf0f537e829e0150cee8050e07c2ada6b612 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/ba5cdf0f537e829e0150cee8050e07c2ada6b612/lm_eval/api/model.py#L277)：一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

  

---


### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1210092229738102804) (6 messages): 

- **CLIP 模型中的 Batch 内假阴性 (False Negative) 困境**：`@tz6352` 最初提出了关于如何解决 CLIP 模型中 Batch 内假阴性问题的方法。
- **寻求关于假阴性的澄清**：`@_.hrafn._` 询问是否担心 Batch 内存在潜在的假阴性。
- **确认假阴性问题**：`@tz6352` 确认该查询确实是关于处理 Batch 中的假阴性，且不局限于 Image-Text 对，这表明了不同的应用场景。
- **缓解假阴性的可能方案**：`@_.hrafn._` 建议使用来自独立文本和图像模型的 Unimodal Embeddings 来计算相似度得分，从而排除假阴性。
- **优化负样本排除策略**：此外，`@_.hrafn._` 提出在训练期间利用自己的模型计算相似度得分，以便更有效地筛选出 Hard Negatives。
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1210167779106357279) (1 messages): 

- **探索序列组合策略**：`@pminervini` 分享了一篇最近的 [arXiv 论文](https://arxiv.org/abs/2402.13991)，讨论了预训练序列组合 (Sequence Composition) 对语言模型的影响。研究表明，文档内因果掩码 (Intra-document Causal Masking) 可以通过消除来自前序文档的干扰信息，显著提高模型在各种任务上的性能。

**提到的链接**：

[Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://arxiv.org/abs/2402.13991)：大多数语言模型预训练框架将多个文档拼接成固定长度的序列，并使用 Causal Masking 来计算给定上下文时每个 Token 的似然概率；这种策略...

  

---

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1209773226176290847) (346 条消息🔥🔥): 

- **Google 新 Gemma 模型讨论**：`@itali4no` 分享了一个关于 Google 发布 Gemma 的[链接](https://blog.google/technology/developers/gemma-open-models/)。Gemma 基于 Gemini 模型的技术构建，强调负责任的 AI 开发。社区对此非常感兴趣，并询问 Google 是否会真正转向开源权重，因为他们在这些方面传统上比较保守。

- **Stable Diffusion 3 早期预览版发布**：`@thejonasbrothers` 引起了大家对 [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3) 的关注，讨论了其作为早期预览候补名单的一部分，在处理多主体提示词（prompts）能力的增强和图像质量的提升。围绕它的讨论包括对其新颖性以及与之前模型实际差异的怀疑。

- **关于使用 CogVL 进行照片标注的讨论**：`@pseudoterminalx` 报告了在 12 小时内使用 CogVL 为 2.88 万张图像生成标注的情况，并分享了关于图像标注所涉及的计算基础设施和成本的见解。这些成本相当可观，通常依赖于租用的多 GPU 服务器。

- **AI 开发中的主导地位与中心化**：一场关于模型和资源（如 SD3）如何变得不那么开放且日益中心化的对话。`@nodja` 等人对算力变得更加中心化表示担忧，以及这种转变如何使技术进一步脱离终端用户的触及范围。

- **关于 SD3 商业用途的推测**：随着 Stability.AI 宣布 SD3，一场关于此类模型是否会用于商业用途的辩论展开了。`@thejonasbrothers` 注意到封闭式开发的趋势，而 `@chad_in_the_house` 则认为开源主要是一种广告手段，而非收入策略。

**提到的链接**：

- [Gemma: Introducing new state-of-the-art open models](https://blog.google/technology/developers/gemma-open-models/)：Gemma 是一个轻量级、最先进的开源模型系列，采用与创建 Gemini 模型相同的研究和技术构建。
- [ Stable Diffusion 3 &mdash; Stability AI](https://stability.ai/news/stable-diffusion-3)：宣布 Stable Diffusion 3 进入早期预览版，这是我们功能最强大的文本生成图像模型，在多主体提示词、图像质量和拼写能力方面有显著提升。
- [no title found](https://ai.google.dev/gemma/docs/model_card)：未找到描述
- [ptx0/photo-concept-bucket · Datasets at Hugging Face](https://huggingface.co/datasets/ptx0/photo-concept-bucket)：未找到描述

---

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1209772678932734012) (65 messages🔥🔥): 

- **合成数据辩论继续**：`@unjay.` 对 OpenAI 的模型利用了大量合成数据表示强烈怀疑，因为存在某些类似 CGI 的伪影，尽管尚未看到 OpenAI 官方对此事的确认。特定 3D 风格的准确复制以及步态周期动画等异常现象是这一论点的关键点。
- **Diffusion 模型生成高性能模型**：`@jordo45` 分享了一篇[有趣的 Arxiv 论文](https://arxiv.org/abs/2402.13144)，证明了 Diffusion 模型可以生成有效的神经网络参数，提供了一种无需大规模架构更改或训练范式即可创建模型的新方法。
- **推出新型多模态 LLM**：`@helium__` 介绍了 [AnyGPT](https://junzhan2000.github.io/AnyGPT.github.io/)，这是一个统一的多模态语言模型，能够使用离散表示处理语音、文本、图像和音乐，突显了 LLM 在处理多种数据格式方面的多功能能力。
- **公共数据集动态讨论**：`@top_walk_town` 建议，由于链接失效和数据投毒等问题， LAION 5B 数据集可能应该退役，这引发了关于社区努力开发具有更好标注的新型高质量公共数据集的讨论。
- **探讨 OpenAI 的收购与结构**：围绕 OpenAI 的收购策略展开了讨论，用户们讨论了像 OpenAI 这样的非营利组织收购公司是否典型。分享的链接阐明了 OpenAI 的混合结构，包括投资者的 100 倍回报上限以及营利性子公司对非营利组织使命的承诺，展示了复杂的业务框架。

**提到的链接**：

- [Neural Network Diffusion](https://arxiv.org/abs/2402.13144)：Diffusion 模型在图像和视频生成方面取得了显著成功。在这项工作中，我们证明了 Diffusion 模型也可以 \textit{生成高性能的神经网络参数}...
- [OpenAI acquires Global Illumination](https://openai.com/blog/openai-acquires-global-illumination)：整个团队已加入 OpenAI。
- [apf1/datafilteringnetworks_2b · Datasets at Hugging Face](https://huggingface.co/datasets/apf1/datafilteringnetworks_2b)：未找到描述
- [Our structure](https://openai.com/our-structure)：我们设计了 OpenAI 的结构——由我们最初的非营利组织和新的上限利润部门组成的合伙关系——作为 OpenAI 使命的底盘：构建安全且...的通用人工智能 (AGI)。
- [AnyGPT](https://junzhan2000.github.io/AnyGPT.github.io/)：未找到描述
- [Demo for &quot;AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling&quot;](https://youtu.be/oW3E3pIsaRg)："AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling" 的演示视频

  

---


### LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/) (1 messages): 

said2000: https://arxiv.org/abs/2402.05608
  

---

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1209804533732212766) (296 条消息🔥🔥): 

- **Mistral AI 的图像文本能力受到质疑**：`@oweowe` 询问 **Mistral AI** 是否可以从 JPEG 格式的表格等复杂图像中提取并处理文本。`@i_am_dom` 建议使用 **gpt4-vision**、**gemini-vision** 或 **blip2** 以获得灵活性，并建议对于小规模数据使用 **copyfish** 和 **google lens** 等更简单的工具。

- **开源期望与权宜之计**：用户讨论了 **Mistral AI** 权重向公众发布的可能性及其影响。`@9faez` 推测如果权重发布，免费版本将迅速出现，而 `@i_am_dom` 则怀疑除非再次发生泄露，否则这不会发生。

- **关于 Mistral API 和 UI 开发的问题**：新手程序员 `@distrorodeo` 寻求使用 **Mistral AI API** 制作 Chat UI 的帮助。`@ethux` 提供了一个指向 **Huggingface ChatUI** 的 GitHub 链接以提供协助。

- **Mistral AI 的性能和微调讨论**：像 `@daroche` 这样的用户对小型 **Mistral 7b** 模型的强大性能表示惊讶，而 `@paul.martrenchar_pro` 建议使用 **RAG** (Retrieval-Augmented Generation) 将公司数据集成到 **Mistral** 中。可以通过 https://docs.mistral.ai/guides/basic-RAG/ 的文档详细了解该技术。

- **对 Mistral 下一代模型迭代的高度关注**：`@egalitaristen` 和 `@sapphics` 等用户报告了 **Mistral Next** 令人印象深刻的性能，特别是在数学方面，在评估中其准确率接近 **GPT-4**。用户还讨论了 **Mistral Next** 相比 **MiQU** 等先前版本可能需要的改进。

**提到的链接**：

- [Chat with Open Large Language Models](https://chat.lmsys.org/)：未找到描述
- [Aaditya Ura (Ankit) (@aadityaura) 的推文](https://x.com/aadityaura/status/1760305308927426903?s=20)：来自 @GoogleDeepMind @GoogleAI 的新模型 Gemma 在医疗/保健领域的基准测试中表现不佳。@GoogleDeepMind 的 Gemma 与 Mistral 的横向对比...
- [Chat with Open Large Language Models](https://chat.lmsys.org)：未找到描述
- [Pretraining on the Test Set Is All You Need](https://arxiv.org/abs/2309.08632)：受近期展示了在精心策划的数据上预训练的小型基于 Transformer 的语言模型前景的工作启发，我们通过投入大量精力策划...
- [gist:c9b5b603f38334c25659efe157ffc51c](https://gist.github.com/sublimator/c9b5b603f38334c25659efe157ffc51c)：GitHub Gist：即时分享代码、笔记和代码片段。
- [Basic RAG | Mistral AI Large Language Models](https://docs.mistral.ai/guides/basic-RAG/)：检索增强生成 (RAG) 是一种 AI 框架，它协同了 LLM 和信息检索系统的能力。它对于利用...回答问题或生成内容非常有用。
- [Mistral](https://huggingface.co/docs/transformers/main/en/model_doc/mistral)：未找到描述
- [GitHub - MeNicefellow/DrNiceFellow-s_Chat_WebUI](https://github.com/MeNicefellow/DrNiceFellow-s_Chat_WebUI)：通过在 GitHub 上创建账号，为 MeNicefellow/DrNiceFellow-s_Chat_WebUI 的开发做出贡献。
- [GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)：多个 NVIDIA GPU 还是 Apple Silicon 用于大语言模型推理？ - XiongjieDai/GPU-Benchmarks-on-LLM-Inference

  

---

### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1209802699445370890) (20 messages🔥): 

- **Mistral-tiny 困惑已消除**：`@hojjat_22712` 询问了 **Mistral-tiny** 与原始 7B 模型之间的可用性和区别，质疑是什么具体细节让 tiny 版本更好。`@akshay_1` 澄清说 API 使用的是 Mistral 7B instruct V2。
- **Mixtral 中出人意料的语言支持**：`@illorca_21005` 讨论了测试 **Mixtral** 的情况，报告其在荷兰语和希腊语中表现尚可，尽管官方文档仅声称支持英语、法语、意大利语、德语和西班牙语。尽管有人询问预训练数据集的文档，但 `@mrdragonfox` 未提供更多信息。
- **Mistral-Next 确认存在**：`@paul16307` 寻求确认 **Mistral-Next** 是否存在，认为它优于 Mistral-Medium，并指向了一个标记为 *null* 的链接。`@ethux` 确认了其真实性，但指出目前还没有 API 访问权限，细节将在未来发布。
- **对 Mistral 细节的期待**：`@ethux` 还提到他们不隶属于 Mistral，但推测有关 API 访问的细节即将公布。
- **Mistral 以价格和创新吸引用户**：`@mrdragonfox` 表示 Mistral 的定价对许多人非常有吸引力，且 Mistral 正在挑战 OpenAI 等公司之外的现有边界。

**提及的链接**：

[Chat with Open Large Language Models](https://chat.lmsys.org/)：未找到描述

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1209787976125513769) (54 messages🔥): 

- **Hugging Face 集成**：用户 `@sa_code` 提到在某些任务中使用 Hugging Face 的 `text-generation-inference`，但未提供进一步的背景或链接。
- **成本评估咨询**：`@ambre3024` 请求协助估算 Mistral 的 **AWS 托管成本**，`@ethux` 随后跟进以澄清正在考虑的是哪种模型（Mistral 7b 或 Mixtral）。
- **Mistral Next 的 API 可用性**：`@rantash68` 询问 **Mistral next** 是否可以通过 API 使用，`@sophiamyang` 简单地回答“不”。
- **在 Vertex AI 上部署 Mistral 的选项**：`@louis2567` 询问了在 Vertex AI 上部署 **Mistral 7b** 和 **Mixtral 8x7b** 模型进行批量预测的问题，并与多位社区成员（特别是 `@mrdragonfox`）讨论了文档缺失和部署效率问题，后者提供了使用 Docker 和 GPU 扩展的详细指导和命令示例。
- **vLLM 的 GPU 选择指南**：`@buttercookie6265` 请求一份为托管 vLLM 选择合适 GPU 的指南，收到了来自 `@mrdragonfox` 关于显存需求以及默认占用大部分 GPU 资源的建议。

**提及的链接**：

[vLLM | Mistral AI Large Language Models](https://docs.mistral.ai/self-deployment/vllm/)：vLLM 可以使用我们提供的 Docker 镜像部署，或者直接从 Python 包部署。

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1209850693075341402) (7 messages): 

- **咨询 Mistral 的微调参数**：`@iamcoming5084` 询问了在微调 **Mistral 8x7b 和 Mistral 7B** 期间可能影响准确性的参数。关于此话题的讨论未提供进一步的信息或建议。
  
- **关于在非结构化数据集上进行微调的咨询**：`@mohammedbelkaid.` 正在寻求在**非结构化电子邮件数据集上微调 Mistral 7B** 的帮助，并询问简单的预处理和分词（Tokenization）是否足以完成摘要和回答问题等任务。

- **请求在 Google Colab 上运行 Mistral 的指导**：`@_logan8_` 请求协助如何使用自己的数据集在 **Google Colab 上微调 Mistral 7B**，但聊天记录中未提供直接的说明或链接。

- **Unsloth 为初学者揭秘微调**：`_._pandora_._` 推荐使用 Unsloth 的演示/笔记本在 **Mistral 模型上进行 LoRA 微调**，强调该资源对初学者非常友好。

- **提升微调效果的技术技巧**：针对有关微调参数的问题，`_._pandora_._` 提到调整 **epoch/steps、batch size 以及 LoRA 超参数 r** 是需要尝试的基础要素。

### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1209805803993702410) (13 messages🔥): 

- **通过 LangGraph 为 RAG 增强自我反思能力**：`@pradeep1148` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=Eb7QF1nDWGU)，展示了如何使用 LangGraph 通过自我反思来增强检索增强生成 (RAG)，该方法可能与 Mistral 的应用相关。
- **讨论面向创意人士的 AI 支持**：`@distrorodeo` 表达了为艺术家创建 AI 创意决策支持系统的兴趣，并询问了如何启动此类项目以及独自完成是否可行。
- **大语言模型微调的复杂性**：`@pradeep1148` 推广了另一个 [YouTube 片段](https://www.youtube.com/watch?v=T_dYzuv4N70)，讨论了 BitDelta 并暗示微调 Large Language Models 可能只能带来边际收益。
- **Twitch 频道测试 Mistral-Next**：`@jay9265` 提到在他们的 Twitch 频道上测试 Mistral-Next 的数据工程用例，提供了直播 [链接](https://www.twitch.tv/jay9265/)，并表示如果这被视为自我推广请予以删除。
- **Mistral 提示词能力指南**：`@mrdragonfox` 建议通过指南进一步探索 Mistral 的 Prompting 能力，并提供了一个 [链接](https://docs.mistral.ai/guides/prompting-capabilities/)，其中包含使用 Mistral 模型进行分类、摘要、个性化和评估的示例。

**提到的链接**：

- [Twitch](https://www.twitch.tv/jay9265/)：未找到描述
- [Prompting Capabilities | Mistral AI Large Language Models](https://docs.mistral.ai/guides/prompting-capabilities/)：当你开始使用 Mistral 模型时，你的第一次交互将围绕 Prompt 展开。编写有效 Prompt 的艺术对于从 Mistral 模型生成理想响应至关重要...
- [Self RAG using LangGraph](https://www.youtube.com/watch?v=Eb7QF1nDWGU)：自我反思可以增强 RAG，从而纠正低质量的检索或生成。最近的几篇论文都聚焦于这一主题，但实现...
- [BitDelta: Your Fine-Tune May Only Be Worth One Bit](https://www.youtube.com/watch?v=T_dYzuv4N70)：Large Language Models (LLMs) 通常分为两个阶段进行训练：在大规模互联网数据集上进行预训练，以及针对下游任务进行微调。鉴于 ...

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1209896268932055171) (12 messages🔥): 

- **Mistral-Next 访问咨询**：用户 `@superseethat` 询问了 **Mistral-Next** 的访问权限，他目前拥有 Mistral Medium 的访问权限。`@ethux` 澄清说 **Mistral Next 尚未发布**，目前只能通过 lymsys 的聊天界面进行测试。
- **了解 API 计费阈值**：用户 `@sapphics` 询问了 **超过 API 计费阈值** 的具体含义。`@mrdragonfox` 确认了该阈值并建议联系支持团队：**support@mistral.ai**。
- **支持响应方面的问题**：`@ginterhauser` 对联系 Mistral 支持以提高限额后未收到回复表示沮丧。`@mrdragonfox` 询问请求中是否包含了 ID，`@nicolas_mistral` 表示如果他们通过 DM 发送 ID 或电子邮件，他可以提供帮助。
- **提供解决支持问题的方案**：来自 Mistral 的 `@nicolas_mistral` 和 `@lerela` 为 `@ginterhauser` 提供了计费问题方面的协助，承诺会解决问题，并要求如果问题仍然存在请发送私信。

**提到的链接**：

[no title found](https://console.mistral.ai/billing/limits/)：未找到描述

  

---

### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1209852346088034384) (57 条消息🔥🔥): 

- **Google 模型更新**：`@oleksandrshr` 提到 Google 发布了一个具有 **新名称** 的新模型，并提到了其可用性。尽管 `@eredon_144` 也提到他在 ChatGPT 移动版上没有看到 Plugins 选项。
- **GPT-4 使用上限争议**：`@7_vit_7` 和 `@solbus` 讨论了 GPT-4 的 **使用上限**，`@solbus` 提供了关于上限及其根据需求和算力可用性动态调整的官方解释 [链接](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4)。
- **对 GPT-4 模型性能的困惑**：用户讨论了 GPT-4 能力随时间变化的潜在可能，`@lugui` 表示关于 **GPT-4 比发布时更弱** 的传言并非事实。
- **Stability.ai 发布 Stable Diffusion 3**：`@pierrunoyt` 分享了 [新闻链接](https://stability.ai/news/stable-diffusion-3)，宣布 **Stable Diffusion 3** 进入早期预览阶段，旨在改进多主体提示词、图像质量和拼写能力。
- **Gemini 模型讨论**：`@ertagon` 强调了一个 [YouTube 视频](https://www.youtube.com/watch?v=Fr6Teh_ox-8)，讨论了与 Google 的 Gemini 模型相关的问题，特别是关于多样性的问题。

**提到的链接**：

- [Introducing ChatGPT Plus](https://openai.com/blog/chatgpt-plus)：我们正在推出 ChatGPT 的试点订阅计划，这是一款对话式 AI，可以与你聊天、回答后续问题并挑战不正确的假设。
- [Stable Diffusion 3 &mdash; Stability AI](https://stability.ai/news/stable-diffusion-3)：宣布 Stable Diffusion 3 进入早期预览版，这是我们能力最强的文本生成图像模型，在多主体提示词、图像质量和拼写能力方面有显著提升。
- [Gemini has a Diversity Problem](https://www.youtube.com/watch?v=Fr6Teh_ox-8)：Google 在其新的 Gemini Pro 模型上将反偏见拨盘调到了最大。参考资料：https://developers.googleblog.com/2024/02/gemini-15-available-for-private-...

---

### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1209843514813194292) (51 条消息🔥): 

- **API 访问说明**：`@solbus` 为 `@phil4246` 解惑，说明 **OpenAI API** 采用按需付费模式，与 Plus 订阅是 **分开的**。他们提到 Token 用于 DALL·E 2 等特定服务，但也与 Plus 订阅无关。

- **文件上传上限澄清**：针对 `@my5042` 的查询，`@solbus` 提供信息称 **文件上传限制** 已更新为 **20 个 512MB** 文件，达到每个终端用户 10GB 的限制，并建议查看最新的 FAQ 以获取准确详情。

- **GPT 写作风格挑战**：`@darthgustav.` 建议 `@thermaltf` 在尝试训练 GPT 模仿其写作风格时，使用 **模板示例** 和 **仅包含正面指令**。

- **神秘的 ChatGPT 模型失误**：`@Makeshift` 评论了 AI 需要 **增强批判性思维**，而 `@darthgustav.` 暗示此类请求可能涉及生成 **剽窃提示词**。

- **从访谈中提取见解**：`@darthgustav.` 向在创建 GPT 寻找访谈记录中精彩瞬间时遇到困难的 `@col.bean` 提供了广泛建议。建议包括在指令中使用 **正面框架** 和 **输出模板**，处理数据块大小，并可能为 **每份访谈记录创建一个新的 GPT** 以避免检索错误。

- **移动版 ChatGPT 无插件**：针对 `@eren_1444` 询问在移动版 ChatGPT 中使用 Plugins 的问题，`@thedreamakeem` 确认 **移动端不支持 Plugins**，并建议在移动浏览器上尝试桌面版。

- **Vector Database 差异**：`@thirawat_z` 表达了在使用 OpenAI embeddings 和 Qdrant 时，结果与教程相差甚远的担忧，分享了其输出与预期结果的显著差异。

- **讨论用 HTML/CSS 训练 ChatGPT**：`@ls_chicha` 询问关于 **使用 HTML 和 CSS 文件训练 ChatGPT** 的问题，引发了 `@_jonpo` 对其必要性的质疑（考虑到 ChatGPT 已有的广泛训练），而 `@toror` 则对 `@ls_chicha` 想要在 ChatGPT 现有能力之外实现的目标表示兴趣。

- **AI 模型对话构想**：`@link12313` 建议创建一个让 GPT-4 和 Google Gemini Ultra1.5 对话的应用，`@toror` 指出其他模型也曾尝试过此类做法，但如果没有引人入胜的切入点，交流往往会变得单调乏味。

---

### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1209874311708414042) (91 条消息🔥🔥): 

- **与 GPT-4 进行角色扮演**：`@shokkunn` 询问如何改进 AI 角色扮演，使其听起来更像角色本身，而不是扮演该角色的演员。`@darthgustav.` 建议在 Prompt 中明确指定自定义指令，包括简洁的指导和带有开放变量的输出模板，这些变量可以概括性地编码指令以保持逻辑一致性。

- **Prompt 中的正向强化被证明更有效**：`@darthgustav.` 强调了在对 AI 进行 Prompt 提示时使用正向指令的重要性，因为负面指令可能会导致不合规（不听从指令）。 

- **Turbo 与普通 GPT-4 在角色扮演中的对比**：`@shokkunn` 观察到标准版 GPT-4 在角色扮演方面的表现似乎优于 Turbo Preview 模型。`@darthgustav.` 建议继续尝试不同的 Prompt 以获得最佳效果，并为旧模型弃用（deprecated）后的过渡做好准备。

- **解决 ReAct Prompting 中的 Agent 循环问题**：`@tawsif2781` 在使用 ReAct Prompting 时遇到了 Agent 陷入逻辑循环的问题。`@darthgustav.` 建议避免在 Prompt 中出现逻辑不一致和负面指令，并建议增加冗余以确保 AI 能够通过中间上下文继续进行有效的操作。

- **Prompt Engineering 学习资源**：`@loamy_` 询问了学习更多关于 Prompt Engineering 的资源，`@darthgustav.` 建议从搜索 arXiv 和 Hugging Face 开始，按最早排序以了解基础知识，或按最新排序以获取高级策略。
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1209874311708414042) (91 条消息🔥🔥): 

- **角色扮演技巧与窍门**：`@shokkunn` 寻求使用 AI 扮演角色的建议，`@darthgustav.` 建议使用**具体**、**简洁**且**逻辑一致**的指令，并配合一个能强化指令的输出模板。强调了**正向指令优于负面指令**的重要性，因为前者能带来更好的合规性。
- **修改角色扮演 Prompt 以获得更好性能**：`@darthgustav.` 暗示旧模型可能会被**弃用（deprecated）**，并建议通过调整当前模型的 Prompt 来为升级做准备。角色扮演模板应包含**开放变量**，且命名规范应能概括指令内容。
- **应用时间戳以获得唯一的 AI 输出**：在关于打破 AI 循环和 **ReAct Prompting** 的讨论中，`@darthgustav.` 提到由于不同的**时间戳 Token**，每个 Prompt 都是唯一的，并建议在 Prompt 中增加冗余可以帮助弥补上下文中的断层。
- **讨论 Prompt Engineering 资源**：`@loamy_` 和 `@droggerhd` 询问了 **Prompt Engineering 资源**，对此 `@darthgustav.` 建议在 **arXiv** 和 **Hugging Face** 上搜索与 Prompt 策略和技术相关的特定关键词。
- **为获得一致的概率输出而调整 Prompt**：`@deb3009` 试图在比较 RCA 与对照数据集时，在输出中获得一致的概率值。他们讨论了通过 Prompt Engineering 产生一致概率的挑战，并收到了关于构建有效 Prompt 的建议。
  

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1209785601637748807) (186 条消息🔥🔥): 

<ul>
<li><strong>theamanstark 的 AI 乐园遇到麻烦</strong>：`@theamanstark` 在发现其 HuggingFace 账号出现 404 错误后感到困惑。`@lunarflu` 建议这可能与滥用 Space 来虚增库统计数据有关，并建议联系 HuggingFace 支持部门寻求解决。</li>
<li><strong>Diffusion Pipeline 讨论</strong>：`@_bootesvoid` 寻求关于使用 Diffusion Pipeline 和 ControlNet 的建议，而 `@thtslunar` 在将权重加载到 'PixArtAlphaPipeline' 时遇到问题，并在 `@not_lain` 的指导下，通过使用不同版本的 diffusers 库找到了解决方案。</li>
<li><strong>HuggingFace VSCode 扩展难题</strong>：`@industrial` 在 NixOS 上配置 `huggingface-vscode` 时面临挑战并寻求社区帮助。`@not_lain` 建议对照默认配置检查设置，并保证在即将发布的 transformers 库版本中会针对自定义架构进行增强。</li>
<li><strong>AI 创新火花揭晓</strong>：`@pierrunoyt` 分享了关于 Stable Diffusion 3 早期预览的激动人心消息，预告了在图像质量和功能方面的重大进步。</li>
<li><strong>寻求 Gradio & FastAPI 性能优化</strong>：`@akin8941` 紧急寻求帮助，以提高利用 Gradio 和 FastAPI 开发的应用的性能。</li>
</ul>

**提到的链接**:

- [3rd Rock GIF - 3rd Rock From - 发现并分享 GIF](https://tenor.com/view/3rd-rock-from-the-sun-gif-5973311)：点击查看 GIF
- [Deer GIF - Deer - 发现并分享 GIF](https://tenor.com/view/deer-gif-22652112)：点击查看 GIF
- [使用自定义模型](https://huggingface.co/docs/transformers.js/custom_usage)：未找到描述
- [Conrad 网站](https://www.catloverdev.com/)：未找到描述
- [ Stable Diffusion 3 &mdash; Stability AI](https://stability.ai/news/stable-diffusion-3)：宣布 Stable Diffusion 3 进入早期预览版，这是我们功能最强大的文本生成图像模型，在多主题提示词、图像质量和拼写能力方面有显著提升。
- [TensorFlow Lite 中的设备端训练 — TensorFlow 博客](https://blog.tensorflow.org/2021/11/on-device-training-in-tensorflow-lite.html)：未找到描述
- [Google Colaboratory](https://colab.research.google.com/drive/11OMSb4XBuOAWaKNEl9Ay7MPnF4rGEf9H#scrollTo=eT6IMdhG2n2u)：未找到描述
- [mayacinka/ramonda-7b-dpo-ties · 不错。太棒了。在你发布这个之前，我也为我的模型选择了同样的名字。](https://huggingface.co/mayacinka/ramonda-7b-dpo-ties/discussions/1)：未找到描述
- [thomas-c-reid/ppo-LunarLander-v2 · Hugging Face](https://huggingface.co/thomas-c-reid/ppo-LunarLander-v2)：未找到描述
- [深度强化学习排行榜 - 由 huggingface-projects 提供的 Hugging Face Space](https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard)：未找到描述
- [AWS Innovate - AI/ML 与数据版](https://aws.amazon.com/events/aws-innovate/apj/aiml-data/)：未找到描述
- [GitHub - kuangliu/pytorch-cifar: 使用 PyTorch 在 CIFAR10 上达到 95.47%](https://github.com/kuangliu/pytorch-cifar)：使用 PyTorch 在 CIFAR10 上达到 95.47%。通过在 GitHub 上创建账号为 kuangliu/pytorch-cifar 的开发做出贡献。
- [GitHub - SYSTRAN/faster-whisper: 使用 CTranslate2 实现更快的 Whisper 转录](https://github.com/SYSTRAN/faster-whisper)：使用 CTranslate2 实现更快的 Whisper 转录。通过在 GitHub 上创建账号为 SYSTRAN/faster-whisper 的开发做出贡献。
- [ptx0/photo-concept-bucket · Hugging Face 数据集](https://huggingface.co/datasets/ptx0/photo-concept-bucket)：未找到描述
- [Mistral 的下一个 LLM 可能与 GPT-4 竞争，你现在可以在 Chatbot Arena 中体验它](https://the-decoder.com/mistrals-next-llm-could-rival-gpt-4-and-you-can-try-it-now-in-chatbot-arena/)：法国 LLM 奇迹 Mistral 正准备发布其下一个语言模型。你已经可以在聊天中对其进行测试。

---

### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1209797458511659048) (7 条消息): 

- **新成员寻求 AI 帮助**：用户 `@mfd000m` 询问关于生成**电子商务产品的英雄图 (hero images)** 的事宜，并请求在 Hugging Face 上推荐适合该任务的模型。
- **寻找合适的模型**：`@jamorphy` 回复询问，以澄清 `@parvpareek` 在提到 "A Neural Probabilistic Language Model" 时具体指的是哪个模型。
- **发布了神秘的 Discord 链接**：用户 `@lightyisu` 发布了一个 Discord 链接 `https://discord.com/channels/879548962464493619/1106008166422028319/1106008166422028319`，但未提供任何上下文或内容。
- **Flutter 游戏查询**：用户 `.konoh` 询问了一个 **Flutter 游戏**，但在对话中没有给出进一步的上下文或回复。
- **Nanotron 开源公告**：`@neuralink` 分享了一个名为 **nanotron** 的项目现已开源，并提供了 GitHub 仓库链接 [`huggingface/nanotron`](https://github.com/huggingface/nanotron/tree/main/examples/doremi)，同时提到他们刚刚完成了合并。

**提到的链接**：

[nanotron/examples/doremi at main · huggingface/nanotron](https://github.com/huggingface/nanotron/tree/main/examples/doremi)：极简的大语言模型 (LLM) 3D-parallelism 训练 - huggingface/nanotron

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1209841382961848390) (8 条消息🔥): 

- **机器人角色扮演开发**：用户 `@ainerd777` 提到正在开发**角色扮演聊天机器人 (roleplay chatbots)**，但未提供更多细节。
- **宏大的合作伙伴计划**：`@aaaliahmad.` 期待与一家市值 1 亿美金的公司建立**合作伙伴关系**。未提供关于此类合作性质的具体细节。
- **对活动价格感到震惊**：`@lucifer_is_back_` 对一个定价为 **1000 美元/座** 的活动做出了反应，评论说有这笔钱他们宁愿投资训练一个 **70B model**。
- **ryzxl 公布模型基准测试结果**：`@ryzxl` 发布了他们的**综合模型基准测试计划 (Comprehensive Model Benchmarking Initiative)** 结果，邀请社区查看对列出的行业领先模型在数据集上进行的广泛测试，并提供了他们的排行榜和仓库链接（[Leaderboard](https://lnkd.in/gxUHqwNp) 和 [Repo](https://lnkd.in/dwhXQ_Bm)）。
- **呼吁发帖礼仪**：`@cakiki` 提醒社区不要跨频道发帖 (cross-post)，将多次发帖的情况标记为垃圾信息 (spam)。
  

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1209913507311657031) (22 条消息🔥): 

<ul>
    <li><strong>投资追踪变得简单</strong>：用户 `@luuisotorres` 介绍了一个用于管理投资组合的 <a href="https://huggingface.co/spaces/luisotorres/portfolio-management">Web 应用</a>，并附带了一个方便的 <a href="https://www.kaggle.com/code/lusfernandotorres/building-an-investment-portfolio-management-app">Kaggle Notebook</a> 来演示其创建过程。</li>
    <li><strong>Android 上的单目深度估计</strong>：`@shubhamx0204` 分享了一个使用转换后的 ONNX 模型进行单目深度估计的 Android 应用，可在 <a href="https://github.com/shubham0204/Depth-Anything-Android">GitHub</a> 上获取。</li>
    <li><strong>文档摘要难题</strong>：`@joethedataguy` 在使用 map reduce 链进行 PDF 文档摘要时遇到问题，并在 <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/document-summarization/summarization_large_documents_langchain.ipynb">GitHub</a> 上咨询了如何将 Vertex AI notebook 适配到 Hugging Face 模型。</li>
    <li><strong>基于 Selenium 的非官方 ChatGPT API</strong>：`@.infinityhawk` 介绍了一个使用 Selenium 和 Python 实现的非官方 ChatGPT API，可在 <a href="https://github.com/Priyanshu-hawk/ChatGPT-unofficial-api-selenium">GitHub</a> 上获取。讨论中涉及了可能违反 OpenAI TOS 的风险，以及使用 undetected drivers 绕过 Cloudflare 防护的方法。</li>
    <li><strong>优化 Stable Diffusion XL</strong>：用户 `@felixsanz` 发布了一篇关于优化 Stable Diffusion XL 的详尽文章，提供了提升性能和减少显存（Memory）占用的策略，详情见其 <a href="https://www.felixsanz.dev/articles/ultimate-guide-to-optimizing-stable-diffusion-xl">网站</a>。</li>
</ul>

**提到的链接**:

- [Prompt Magic v0.0.1](https://c6548e7f4c4e5a6d00.gradio.live/): 未找到描述
- [Proteus V0.4 - FumesAI 开发的 Hugging Face Space](https://huggingface.co/spaces/FumesAI/Proteus-V0.4): 未找到描述
- [Portfolio Management - luisotorres 开发的 Hugging Face Space](https://huggingface.co/spaces/luisotorres/portfolio-management): 未找到描述
- [构建投资组合管理应用 &#x1F4B0;](https://www.kaggle.com/code/lusfernandotorres/building-an-investment-portfolio-management-app): 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自无附加数据源的数据
- [GitHub - shubham0204/Depth-Anything-Android: 在 Depth-Anything 上运行推理的 Android 应用](https://github.com/shubham0204/Depth-Anything-Android): 一个在 Depth-Anything 上运行推理的 Android 应用 - GitHub - shubham0204/Depth-Anything-Android: An Android app running inference on Depth-Anything
- [GitHub - Priyanshu-hawk/ChatGPT-unofficial-api-selenium: 这是一个完全由我用 Python 和 Selenium 编写的非官方 ChatGPT API](https://github.com/Priyanshu-hawk/ChatGPT-unofficial-api-selenium): 这是一个完全由我用 Python 和 Selenium 编写的非官方 ChatGPT API - Priyanshu-hawk/ChatGPT-unofficial-api-selenium
- [generative-ai/language/use-cases/document-summarization/summarization_large_documents_langchain.ipynb at main · GoogleCloudPlatform/generative-ai](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/document-summarization/summarization_large_documents_langchain.ipynb): Google Cloud 上 Generative AI 的示例代码和 Notebooks - GoogleCloudPlatform/generative-ai
- [最便宜的 GPT-4 Turbo, GPT 4 Vision, ChatGPT OpenAI AI API 文档 (NextAPI) | RapidAPI](https://rapidapi.com/NextAPI/api/cheapest-gpt-4-turbo-gpt-4-vision-chatgpt-openai-ai-api): 未找到描述
- [优化 Stable Diffusion XL 的终极指南](https://www.felixsanz.dev/articles/ultimate-guide-to-optimizing-stable-diffusion-xl): 探索如何在任何显卡上获得 SDXL 的最佳质量和性能。

---

### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1209897641580826634) (10 messages🔥): 

- **Stable Diffusion 中的 Timestep embedding**: `@pseudoterminalx` 讨论了在 Stable Diffusion 中，timestep embed 是如何拼接到文本嵌入（text embedding）隐藏状态的，它可能不是简单的整数，而是通过 Fourier transform 创建的向量。
- **SDXL 微调节（microconditioning）输入增强**: `@pseudoterminalx` 解释了 SDXL 使用 Fourier transform 来增强微调节输入，将 6 元素的输入扩展为 256 元素，并特别提到它涉及“3 组双元素元组（two element tuples）”。
- **对 Diffusion 讨论的认可**: `@mr.osophy` 认可了 `@pseudoterminalx` 对 Diffusion 话题的回复，并表示打算稍后深入研究该主题。
- **对基于 Interlingua 翻译器的兴趣**: `@hobojesus6250a` 表达了在 Hugging Face 上为大学项目开发或寻找基于 Interlingua 的翻译器的兴趣，由于时间限制，希望扩展现有的模型或 LLM 来处理翻译任务。
- **针对更多类别的模型扩展**: `@agusschmidt` 询问如何运行超过 10 个类别的 [BART-large-mnli 模型](https://huggingface.co/facebook/bart-large-mnli)，引用了一项讨论建议在本地运行模型是可行的，并寻求指导或允许更多类别的替代模型。
  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1210215877396668457) (1 messages): 

- **多标签图像分类教程**: 用户 `@nielsr_` 分享了一个用于多标签图像分类的 [教程 Notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SigLIP/Fine_tuning_SigLIP_and_friends_for_multi_label_image_classification.ipynb)，演示了使用 Transformers 库中强大的视觉骨干网络 **SigLIP** 的过程，同时指出库中的任何视觉模型都可以使用。
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1209776565496451132) (36 messages🔥): 

- **解决 TensorFlow 难题**: 用户 `@diegot8170` 在使用 TensorFlow 加载模型时遇到问题，`@cursorop` 建议使用 pip 命令重新安装特定版本（`2.15`）的 TensorFlow 来解决。
- **生物医学领域的自定义句子相似度**: `@joshpopelka20` 在生物医学术语的句子相似度预训练嵌入模型方面面临挑战，`@lavi_39761` 建议探索 contrastive learning 以及用于微调的工具如 sentence transformers 和 setfit。
- **PEFT 持久化问题**: 参与者 `@grimsqueaker` 和 `@kingpoki` 讨论了一个反复出现的问题，即 PEFT 无法为自动配置未涵盖的模型保存正确的 heads，导致尝试通过参数调整进行变通。
- **探索 Reformer 架构**: `@devbravo` 提到正在研究 Reformer 架构，以开发适用于边缘设备（edge devices）的更小、更节省内存的模型。
- **Bert 训练数据困境未获解答**: `@jldevtech` 询问社区关于训练用于多标签分类的 Bert perf adapter 所需的最少数据量，但在交流中未收到反馈。
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1209897641580826634) (10 messages🔥): 

- **Stable Diffusion 嵌入讨论**: `@pseudoterminalx` 指出在 Stable Diffusion 中，timestep embed 被拼接到文本嵌入隐藏状态，可能使用 **Fourier transform** 来创建向量。
- **SDXL 微调节详解**: `@pseudoterminalx` 进一步解释了 SDXL 如何对附加到 time embed 的微调节输入使用 Fourier transform，将 6 元素输入扩展为 256 元素输出。
- **Time Embed 中的元组扩展**: 在澄清维度时，`@pseudoterminalx` 提到 Stable Diffusion 中的 time embeds 是 3 组双元素元组。
- **mr.osophy 认可讨论点**: `@mr.osophy` 感谢了 `@636706883859906562` 之前的回复，并计划稍后探索该话题。
- **寻找 Interlingua 翻译器项目**: `@hobojesus6250a` 询问是否有人在 Hugging Face 上研究过基于 Interlingua 的翻译器项目，希望因时间限制扩展一个用于大学项目。
- **BART 模型多类别查询**: `@agusschmidt` 寻求关于运行超过 10 个类别的 [BART-large-mnli 模型](https://huggingface.co/facebook/bart-large-mnli) 的指导，想知道如何在本地执行此操作，或者是否有其他模型支持更多类别。
  

---

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1209877158743384175) (78 messages🔥🔥): 

- **Gemma 隆重登场**：Google 在 Hugging Face 上发布了新的语言模型系列 **Gemma**，包含 7B 和 2B 两种尺寸。用户 `@mjng93` 链接了 [Hugging Face 博客](https://huggingface.co/blog/gemma)，`@coffeebean6887` 分享了[发布条款](https://ai.google.dev/gemma/terms)，重点介绍了分发模型衍生品的限制。

- **显微镜下的 Gemma**：`@guardiang` 对比了 Gemma 和 Llama 2 的 tokenizer，指出 Gemma 拥有更大的词表（vocab）并包含大量特殊 token；该分析的详细内容通过 [tokenizer 模型文件](https://github.com/google/gemma_pytorch/blob/main/tokenizer/tokenizer.model)和 [diffchecker 对比](https://www.diffchecker.com/TRnbKRMH/)的链接进行了分享。

- **Stable Diffusion 3 问世**：`@rubenartus` 宣布了 Stable Diffusion 3 的早期预览版，并提供了 [Stability AI 公告](https://stability.ai/news/stable-diffusion-3)以及 EMostaque 发布的包含更多细节的 [Twitter 线程](https://twitter.com/EMostaque/status/1760660709308846135)链接。

- **Google Gemini Pro 1.5 探索**：`@nuvic_` 对 Gemini Pro 1.5 新的 1,000,000 token 上下文窗口及其将视频作为输入的能力非常感兴趣，并引用了 Simon Willison 在其[个人博客](https://simonwillison.net/2024/Feb/21/gemini-pro-video/)上概述的相关技术实验。

- **ChatGPT 出现异常后修复**：`@swyxio` 分享了一个关于 [ChatGPT 异常行为的 Twitter 链接](https://twitter.com/E0M/status/1760476148763644166)，而 `@dimfeld` 指向了 [OpenAI 状态页面](https://status.openai.com/incidents/ssg8fh7sfyz3)，确认该问题已解决。


**提到的链接**：

- [未找到标题](https://news.ycombinator.com/item?id=39463470)：未找到描述
- [One Year of Latent Space](https://www.alessiofanelli.com/posts/latent-space)：Latent Space 在一年内从 0 增长到 100 万读者的经验（与回忆）。
- [ Stable Diffusion 3 &mdash; Stability AI](https://stability.ai/news/stable-diffusion-3)：宣布 Stable Diffusion 3 开启早期预览，这是我们最强大的文本生成图像模型，在多主体提示词、图像质量和拼写能力方面有显著提升。
- [The killer app of Gemini Pro 1.5 is video](https://simonwillison.net/2024/Feb/21/gemini-pro-video/)：上周 Google 推出了 Gemini Pro 1.5，这是对其 Gemini 系列 AI 模型的一次巨大升级。Gemini Pro 1.5 拥有 1,000,000 token 的上下文窗口。这非常惊人——此前……
- [Unexpected responses from ChatGPT](https://status.openai.com/incidents/ssg8fh7sfyz3)：未找到描述
- [Andrej Karpathy (@karpathy) 的推文](https://x.com/karpathy/status/1760350892317098371?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：鉴于我昨天发布了关于 Tokenizer 的视频，我想深入研究一下 Gemma 的 tokenizer 应该会很有趣。首先是 Gemma 技术报告 [pdf]：https://storage.googleapis.com/de...
- [Dana Woodman  (@DanaWoodman) 的推文](https://x.com/DanaWoodman/status/1760109214469607859?s=20)：这到底是怎么回事 @ChatGPTapp 😂 我不是网络专家，但我很确定这纯粹是胡言乱语……
- [Hamilton Ulmer (@hamiltonulmer) 的推文](https://x.com/hamiltonulmer/status/1760081097298444341?s=20)：我正处于最奇怪的 ChatGPT 实验分支中
- [Welcome Gemma - Google’s new open LLM](https://huggingface.co/blog/gemma)：未找到描述
- [未找到标题](https://ai.google.dev/gemma/terms)：未找到描述
- [Scaling ChatGPT: Five Real-World Engineering Challenges](https://newsletter.pragmaticengineer.com/p/scaling-chatgpt)：在发布仅一年后，ChatGPT 的周活跃用户就超过了 1 亿。为了满足这种爆发式需求，OpenAI 团队必须克服几个扩展挑战。独家深度解析。
- [Launch HN: Retell AI (YC W24) – Conversational Speech API for Your LLM | Hacker News](https://news.ycombinator.com/item?id=39453402)：未找到描述
- [Rise of the AI Engineer (with Build Club ANZ)](https://www.youtube.com/watch?v=ezhSIGKFtOc)：幻灯片：https://docs.google.com/presentation/d/157hX7F-9Y0kwCych4MyKuFfkm_SKPTN__BLOfmRh4xU/edit?usp=sharing🎯 Build Club 中的要点/亮点线程……
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1awbo84/google_publishes_open_source_2b_and_7b_model/)：未找到描述
- [未找到标题](https://ai.google.dev/gemma/prohibited_use_policy)：未找到描述

### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1209953727373377596) (3 条消息): 

- **深入探讨 'Building Your Own Product Copilot'**: `@swyxio` 宣布 `@451508585147400209` 正在主持关于 [Building Your Own Product Copilot](https://arxiv.org/abs/2312.14231) 论文的讨论。可以通过特定的 [Discord channel](https://discord.com/channels/822583790773862470/1197350122112168006) 参与该环节。
- **通过 Latent.Space 随时了解未来活动**: `@swyxio` 分享了 [Latent.Space events](http://Latent.Space) 的链接，用户可以点击 RSS 标志将活动日历添加到个人日历并接收通知。操作说明包括在悬停时点击 "Add iCal Subscription" 以实现自动更新。

**提到的链接**:

[Latent Space (Paper Club &amp; Other Events) · Luma](https://lu.ma/ls): 在 Luma 上查看并订阅来自 Latent Space (Paper Club &amp; Other Events) 的活动。Latent.Space 活动。请点击日历右上方正上方的 RSS 标志以添加到您的日历。"Ad...

  

---

### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1209952591669370940) (173 条消息🔥🔥): 

- **搞笑的人机交互**：`@_bassboost` 强调了论文中的一个古怪案例：在询问建议的对话中，用户会回复一些个人问题，比如没有朋友。工程师们试图引导模型避开可能导致敏感领域的议题。
- **论文俱乐部投票热潮**：`@eugeneyan`、`@henriqueln7` 和 `@amgadoz` 等成员讨论并投票决定深入研究哪篇论文，备选项包括 Copilot 研究和 Sora。文中提供了论文链接，包括 [arxiv](https://arxiv.org/abs/2312.14231) 上关于 Copilot 研究的摘要。
- **Google Gemini 正式起飞**：`@coffeebean6887` 讨论了 Google Gemini AI 与 Workspace 和 Google One 服务的集成，并提供了展示其先进功能的视觉效果和博客文章链接（[Google One](https://blog.google/products/google-one/google-one-gemini-ai-gmail-docs-sheets/)，[Workspace](https://blog.google/products/workspace/google-gemini-workspace/)）。
- **AI 评估裁判 (Judges)**：讨论转向了评估 AI 的回答，`@henriqueln7`、`@swyxio` 和 `@_bassboost` 等成员讨论了使用 Langsmith、GPT4 以及更小的模型作为对话式聊天机器人和学习平台的裁判。此外还分享了 [predibase.com Lora Land](https://predibase.com/lora-land) 等工具用于微调 (finetuning) 对比。
- **ML 和 GenAI 人才的未来**：在一个前瞻性话题中，`@lightningralf` 和 `@eugeneyan` 辩论了 ML/GenAI 人才以及采用 AI 的公司的发展格局。他们推测了快速改进的工具和 AI 进步可能带来的影响，这些进步可能会在几年内改变对某些技能组合的需求。

**提到的链接**：

- [Building Your Own Product Copilot: Challenges, Opportunities, and Needs](https://arxiv.org/abs/2312.14231)：一场将先进 AI 能力嵌入产品的竞赛正在进行。这些产品 Copilot 允许用户使用自然语言提问，并获得针对特定用途的相关回答……
- [Boost your productivity: Use Gemini in Gmail, Docs and more with the new Google One plan](https://blog.google/products/google-one/google-one-gemini-ai-gmail-docs-sheets/)：我们通过在 Gmail、Docs、Slides、Sheets 和 Meet（原 Duet AI）中引入 Gemini，为 Google One AI Premium 计划带来更多价值。
- [LoRA Land: Fine-Tuned Open-Source LLMs](https://predibase.com/lora-land)：性能超越 GPT-4 的微调开源 LLM，可在单张 GPU 上运行。
- [Tweet from Amazon Web Services (@awscloud)](https://x.com/awscloud/status/1752051165200601299?s=46&t=90xQ8sGy63D2OtiaoGJuww)：由 #AWS 发起的 PartyRock #generativeAI 黑客松现在开始！📣 了解如何无需编程即可构建有趣且直观的应用，有机会赢取现金奖励和 AWS 额度。🏆 #AI 别忘了你的马克杯……
- [SPQA: The AI-based Architecture That’ll Replace Most Existing Software](https://danielmiessler.com/p/spqa-ai-architecture-replace-existing-software/)：2023 年 3 月 10 日。得益于 GPT 带来的爆发，AI 在未来几个月和几年内将完成许多有趣的事情。但其中最重要的之一是……
- [no title found](https://open-vsx.org/extension/Continue/continue)：未找到描述
- [- Fuck You, Show Me The Prompt.](https://hamel.dev/blog/posts/prompt/)：通过拦截 API 调用，快速理解难以捉摸的 LLM 框架。
- [New ways Google Workspace customers can use Gemini](https://blog.google/products/workspace/google-gemini-workspace/)：我们正在推出一项新方案，帮助组织开始使用生成式 AI，并提供与 Gemini 聊天的独立体验。
- [Founder’s Guide to Basic Startup Infrastructure](https://www.flexport.com/blog/founders-guide-to-basic-startup-infrastructure/)：未找到描述
- [GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models](https://github.com/stanfordnlp/dspy)：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy

### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1209901544745472111) (3 messages): 

- **简化 RAG 的复杂性**：`@IFTTT` 强调了由于选项众多，构建高级 RAG 系统的复杂性。他们建议通过精确找出每个 Pipeline 组件中的痛点及相应的解决方案来简化流程，并分享了 [来自 @jerryjliu0 演讲的幻灯片](https://t.co/FhwU6tA73o)。
- **为 LLM/RAG 专家准备的前端**：由 `@IFTTT` 推荐的 Marco Bertelli 教程，教授没有 React 知识的 LLM/RAG 专家如何为他们的 RAG 后端创建一个美观的前端，资源来自 [@llama_index](https://t.co/35UeUCrKWg)。
- **从 RAG Notebooks 到全栈应用**：`@wenqi_glantz` 提供了一个教程，介绍如何将 RAG Notebooks 转换为包含摄取（Ingestion）和推理（Inference）微服务的综合应用程序，由 `@IFTTT` 在其推文中分享，包含教程链接和进一步步骤。[点击此处查看完整教程。](https://t.co/S86B38YZQ1)
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1209800397485772811) (246 messages🔥🔥): 

- **寻求 QueryPipeline RAG 的澄清**：用户 `@lapexer` 好奇如何在包含 Prompt、Retriever 和 LLM 的 DAG QueryPipeline 中编写一个简单的 RAG。文档 [RAG Pipeline Without Query Rewriting](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline.html#rag-pipeline-without-query-rewriting) 被提供用于指导如何设置 Pipeline。
- **LlamaIndex ImportError 问题**：用户 `@emmepra` 和 `@pymangekyo` 讨论了从 `llama_index` 导入 `VectorStoreIndex` 时遇到的问题。`@emmepra` 建议从 `llama_index.core` 而不是 `llama_index.legacy` 导入以尝试解决问题，而 `@whitefang_jr` 建议在卸载并重新安装后使用全新的环境。
- **LangchainEmbedding 导入问题**：尽管参考了文档，用户 `@pymangekyo` 仍无法从 `llama_index.embeddings` 导入 `LangchainEmbedding`。`@emmepra` 建议尝试从 `llama_index.core.indices` 导入，但 `@pymangekyo` 仍然面临问题。
- **CRAG Pack 下载问题**：用户 `@lapexer` 报告在尝试使用 `llamaindex-cli` 下载 CorrectiveRAGPack 时出现 `ValueError`。`@whitefang_jr` 指出一个修复 llama-pack 下载的相关 Pull Request 可能会解决此问题。链接了 [PR #11272](https://github.com/run-llama/llama_index/pull/11272) 以供参考。
- **LlamaIndex 文档和 LlamaHub Reader 链接失效**：用户 `@andaldana` 询问如何使用 `DatabaseReader` 和 `CSVreader` 处理 SQL 数据库中每条记录作为一个 Document 的数据，但发现文档链接已失效。他们正在寻求 LlamaIndex 中更新的方法或 Reader 来实现其目标。

**提到的链接**：

- [未找到标题](http://localhost:8001',): 未找到描述
- [T-RAG = RAG + Fine-Tuning + Entity Detection](https://cobusgreyling.medium.com/t-rag-rag-fine-tuning-entity-detection-9a5aaa01e437): T-RAG 方法的前提是将 RAG 架构与开源的 Fine-Tuned LLM 和实体树向量数据库相结合。
- [Fine-tuning - LlamaIndex 🦙 v0.10.11.post1](https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning.html#fine-tuning-llama-2-for-better-text-to-sql): 未找到描述
- [Google Colaboratory](https://colab.research.google.com/drive/1uJ2qXJ-laFIEweDWNKXqa2gLt765PPRD?usp=sharing): 未找到描述
- [Building Your Own Evals - Phoenix](https://docs.arize.com/phoenix/llm-evals/building-your-own-evals): 未找到描述
- [LangChain Embeddings - LlamaIndex 🦙 v0.10.11.post1](https://docs.llamaindex.ai/en/stable/examples/embeddings/Langchain.html): 未找到描述
- [RAG CLI - LlamaIndex 🦙 v0.10.11.post1](https://docs.llamaindex.ai/en/stable/use_cases/q_and_a/rag_cli.html#): 未找到描述
- [Loading Data (Ingestion) - LlamaIndex 🦙 v0.10.11.post1](https://docs.llamaindex.ai/en/stable/understanding/loading/loading.html#using-readers-from-llamahub),): 未找到描述
- [未找到标题](https://llamahub.ai/l/readers/llama-index-readers-database): 未找到描述
- [未找到标题](https://llamahub.ai/l/readers/llama-index-readers-file?from=readers): 未找到描述
- [llama_index/llama-index-core/llama_index/core/question_gen/llm_generators.py at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/question_gen/llm_generators.py): LlamaIndex（前身为 GPT Index）是为您 LLM 应用程序提供的数据框架 - run-llama/llama_index

- [llama_index/llama-index-core/llama_index/core/question_gen/llm_generators.py at da5f941662b65d2e3fe2100f2b58c3ba98d49e90 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/da5f941662b65d2e3fe2100f2b58c3ba98d49e90/llama-index-core/llama_index/core/question_gen/llm_generators.py#L10C5-L10C37): LlamaIndex（前身为 GPT Index）是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index
- [llama_index/llama-index-core/llama_index/core/callbacks/token_counting.py at 6fb1fa814fc274fe7b4747c047e64c9164d2042e · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/6fb1fa814fc274fe7b4747c047e64c9164d2042e/llama-index-core/llama_index/core/callbacks/token_counting.py#L53): LlamaIndex（前身为 GPT Index）是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index
- [An Introduction to LlamaIndex Query Pipelines - LlamaIndex 🦙 v0.10.11.post1](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline.html#rag-pipeline-without-query-rewriting): 未找到描述
- [no title found](https://cloud.google.com/docs/authentication/external/set-up-adc): 未找到描述
- [llama_parse/examples/demo_advanced_astradb.ipynb at main · run-llama/llama_parse](https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced_astradb.ipynb): 为实现最佳 RAG 解析文件。通过在 GitHub 上创建账号，为 run-llama/llama_parse 的开发做出贡献。
- [llama_parse/examples/demo_astradb.ipynb at main · run-llama/llama_parse](https://github.com/run-llama/llama_parse/blob/main/examples/demo_astradb.ipynb): 为实现最佳 RAG 解析文件。通过在 GitHub 上创建账号，为 run-llama/llama_parse 的开发做出贡献。
- [[FIX] download_llama_pack for python packages containing multiple packs by nerdai · Pull Request #11272 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/11272): 描述：之前的 download_llama_pack 逻辑对 GitHub 树的遍历不够深入，这给包含多个 pack 的包带来了问题（即，由于存在更多包含 s... 的文件夹）
- [Survey on your Research Journey](https://forms.gle/8N4DsuCWtCXKxLSv6): 为了彻底改变学术和商业研究，EurekAI 正在征求您的见解，以便根据您的需求定制我们的工具。无论您是沉浸在研究中还是偶尔参与，您的...
- [Custom Embeddings - LlamaIndex 🦙 v0.10.11.post1](https://docs.llamaindex.ai/en/stable/examples/embeddings/custom_embeddings.html#custom-embeddings-implementation): 未找到描述
- [llama_index/llama-index-integrations/embeddings/llama-index-embeddings-ollama/llama_index/embeddings/ollama/base.py at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-ollama/llama_index/embeddings/ollama/base.py): LlamaIndex（前身为 GPT Index）是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index
- [OpenAI API compatibility · Issue #305 · ollama/ollama](https://github.com/ollama/ollama/issues/305): 有没有可能考虑镜像 OpenAI 的 API 规范和输出？例如 /completions 和 /chat/completions。这样，通过更改... 就可以作为 Python openai 包的直接替代品。

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1209936622687363123) (3 条消息): 

- **表达感谢**：用户 `@behanzin777` 表示打算尝试建议的解决方案，并以 **"Thanks. I will give it a try 🙏🏾"** 表达了谢意。
- **寻求 LlamaIndex 的摘要评估指标**：`@dadabit.` 询问了在 LlamaIndex 中评估摘要的有效**指标和工具**。他们对基于社区经验的推荐感兴趣。
- **寻找 LLM 评估平台**：`@.dheemanth` 正在寻找一个**易于使用的平台**来评估 Large Language Models (LLMs)，该平台需包含类似于 **MT-Bench** 和 **MMLU** 的分析、追踪和评分功能。

### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1209848489668902913) (149 条消息🔥🔥): 

- **Google Gemma AI 模型讨论**：OpenAccess AI Collective 的用户正在积极讨论 Google 新推出的 **Gemma** 模型系列。`@nafnlaus00` [检查了许可证详情](https://ai.google.dev/gemma/prohibited_use_policy)，指出其限制比 **LLaMA 2** 更少。`@le_mess` 提供了 [Gemma 在 Hugging Face 上集成](https://huggingface.co/blog/gemma/?utm_source=agd&utm_medium=referral&utm_campaign=view-on-huggingface&utm_content=)的更新，并附带了模型和技术文档的链接。

- **Gemma 模型属性揭晓**：`@le_mess` 获得了 **Gemma** 仓库的访问权限，透露了诸如 `max_position_embeddings: 8192` 和 `vocab_size: 256000` 等特征。讨论集中在大词表大小的影响以及这可能如何影响推理时间。

- **Gemma 模型的公开访问**：`@le_mess` 报告称重新上传了 **Gemma** 的 7B 模型，使其可以在 Hugging Face 上公开使用，绕过了 Google 最初要求的访问请求。

- **Gemma 的微调挑战**：多位用户报告了微调 **Gemma** 时遇到的问题，特别是 `@stoicbatman` 在训练结束时遇到了错误。`@nanobitz` 引用了相关的 GitHub issues，指出可能存在 early stopping callback（早停回调）问题。

- **云算力成本分析**：`@yamashi` 在与 Google 的讨论中提出了云算力资源的高昂成本，并将其与物理拥有服务器的价格进行了比较。DreamGen 讨论了可能使云选项更具吸引力的折扣，特别是针对研究人员。

**提到的链接**：

- [未找到标题](https://ai.google.dev/gemma)：未找到描述
- [HuggingChat](https://huggingface.co/chat)：让每个人都能使用社区最好的 AI 聊天模型。
- [欢迎 Gemma - Google 的新开源 LLM](https://huggingface.co/blog/gemma/?utm_source=agd&utm_medium=referral&utm_campaign=view-on-huggingface&utm_content=)：未找到描述
- [mhenrichsen/gemma-7b · Hugging Face](https://huggingface.co/mhenrichsen/gemma-7b)：未找到描述
- [Google 推出名为 Gemma 的轻量级开源 AI 模型](https://www.engadget.com/google-introduces-a-lightweight-open-ai-model-called-gemma-130053289.html)：Google 表示 Gemma 是其对开源社区的贡献，旨在帮助开发者“负责任地构建 AI”。
- [mhenrichsen/gemma-7b-it · Hugging Face](https://huggingface.co/mhenrichsen/gemma-7b-it)：未找到描述
- [来自 Tri Dao (@tri_dao) 的推文](https://x.com/tri_dao/status/1760458183066472556?s=20)：FlashAttention v2.5.5 现在支持在消费级 GPU 上进行 head dim 256 的反向传播。希望这能让微调 Gemma 模型变得更容易。
- [使用 EarlyStoppingCallback 保存时出错 · Issue #29157 · huggingface/transformers](https://github.com/huggingface/transformers/issues/29157)：系统信息 transformers 版本：4.38.0.dev0（在 4.38.0 和 4.39.0.dev0 中也存在）平台：Linux-5.15.0-78-generic-x86_64-with-glibc2.35 Python 版本：3.10.12 Huggingface_hub 版本：0.20.3 Safete...
- [llm-foundry/scripts/train/README.md at main · mosaicml/llm-foundry](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#howmandygpus)：用于 MosaicML 基础模型的 LLM 训练代码。通过在 GitHub 上创建账号来为 mosaicml/llm-foundry 的开发做出贡献。
- [在消费级 GPU (Ampere, Ada) 上启用 headdim 256 反向传播 · Dao-AILab/flash-attention@2406f28](https://github.com/Dao-AILab/flash-attention/commit/2406f28805e2a3623427f48f38fc533a5d1f2c32)：未找到描述
- [GitHub - Dao-AILab/flash-attention: 快速且内存高效的精确注意力机制](https://github.com/Dao-AILab/flash-attention)：快速且内存高效的精确注意力机制。通过在 GitHub 上创建账号来为 Dao-AILab/flash-attention 的开发做出贡献。

---

### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1209830355037724682) (26 messages🔥): 

- **Merge Ready for Fixes**: `@nanobitz` 寻求确认以开始将包括 readme 修复、val 修复和示例修复在内的小型 PR 合并到 **axolotl** 代码库中。
- **Gemma 训练要求**: `@giftedgummybee` 强调了训练 **gemma** 模型需要非开发版本的 *transformers*，并提到开发版本不支持 "gemma" 类型的模型。`@stoicbatman` 证实了这一点，他在 **axolotl** 的 Docker 镜像上使用开发版本时遇到了问题。
- **Gemma 的配置说明**: `@stoicbatman` 分享了一个更新后的 **gemma config file** 以解决设置过程中的问题。同时，`@nanobitz` 指出 *sample packing* 在该模型上尚无法运行。
- **Gemma 微调中的超参数困惑**: `@faldore` 和 `@nanobitz` 讨论了 **Gemma models** 合适的学习率和权重衰减，参考了 Google 在不同文档中给出的 5e-5 和 2e-4 等多种建议。
- **Mixtral 的优化建议**: `@casper_ai` 分享了关于优化 **Mixtral model** 的见解，并讨论了提升速度的潜力，尽管提到自己缺乏编写 CUDA backward passes 的专业知识。他们还提到了使用 **AutoAWQ** 在 prefilling 和解码速度方面取得的成功。

**提到的链接**:

- [Welcome Gemma - Google’s new open LLM](https://huggingface.co/blog/gemma/?utm_source=agd&utm_medium=referral&utm_campaign=view-on-huggingface&utm_content=#fine-tuning-with-%F0%9F%A4%97-trl): 未找到描述
- [Google Colaboratory](https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemma/docs/lora_tuning.ipynb#scrollTo=_Peq7TnLtHse&line=1&uniqifier=1): 未找到描述
- [gemma_config_axolotl.yml](https://gist.github.com/monk1337/b7ee08781d62e351db7fc7c6fe0645e0): GitHub Gist：即时分享代码、笔记和片段。

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1209782855555616829) (51 messages🔥): 

- **寻找 Alpaca 模板**: `@yamashi` 正在寻找 alpaca 的 jinja template，`@rtyax` 分享了一个可能的模板，`@yamashi` 打算将其添加到 [axolotl repository](https://github.com/oobabooga/text-generation-webui/blob/main/instruction-templates/Alpaca.yaml)。
- **训练辅助**: `@napuh` 探讨了如何利用 DeepSpeed 和多 GPU 提高训练速度，而 `@nanobitz` 澄清了 micro batch size 乘以 gradient accumulation 是针对每个 GPU 的，这意味着更多 GPU 应该会导致更少的步数（steps）。
- **微调推理格式**: `@timisbister` 和 `@nani1149` 询问模型微调后的正确推理格式，`@nanobitz` 和 `@yamashi` 提供了模板和格式指导，`@yamashi` 指出需要完善文档以减少重复提问。
- **FlashAttention 困扰**: `@rakesh_46298` 遇到了与 FlashAttention 和 GPU 相关的运行时错误，`@nanobitz` 建议关闭该功能，但仍需进一步说明。
- **文档需求**: 鉴于重复出现的问题，`@yamashi` 和 `@nanobitz` 讨论了通过 read-the-docs 或 gitbooks 为 axolotl 提供更好文档的需求，并提到这是之前讨论过的话题。

**提到的链接**:

- [tokenizer_config.json · teknium/OpenHermes-2.5-Mistral-7B at main](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/tokenizer_config.json): 未找到描述
- [GitHub - tatsu-lab/stanford_alpaca: Code and documentation to train Stanford's Alpaca models, and generate the data.](https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release): 用于训练 Stanford Alpaca 模型并生成数据的代码和文档。 - tatsu-lab/stanford_alpaca
- [text-generation-webui/instruction-templates/Alpaca.yaml at main · oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui/blob/main/instruction-templates/Alpaca.yaml): 一个用于大语言模型的 Gradio Web UI。支持 transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。 - oobabooga/text-generation-webui

  

---

### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1210262728535711784) (1 messages): 

- **AI 叙事的新冒险**：`@dreamgen` 宣布发布了用于 AI 驱动的*故事写作*和*角色扮演*的新模型，现已在 [Hugging Face](https://huggingface.co/collections/dreamgen/opus-v1-story-writing-and-role-playing-models-65d092a6f8ab7fc669111b31) 上线。这些 **Opus V1 模型**是在约 100M tokens 的人类生成文本上训练的，并基于 ChatML 的扩展版本。
- **使用 ChatML+ 引导叙事**：包含的模型利用了改进版的 *ChatML* 进行 **prompting**，为更受控的输出增加了灵活性。模型的详细用法以及提示指令可以在此处的 **Opus V1 指南** [here](https://dub.sh/opus-v1-guide) 中找到。
- **引导对话的秘诀**：`@dreamgen` 解释了**可引导提示词 (steerable prompts)** 的概念，它涉及一种结构化输入：一个定义故事或角色扮演场景的 system prompt，随后是随着故事展开的文本轮次，以及引导接下来发生什么的指令。这允许用户更直接地影响生成内容的方向。

**提到的链接**：

- [Opus V1: Story-writing &amp; role-playing models - a dreamgen Collection](https://huggingface.co/collections/dreamgen/opus-v1-story-writing-and-role-playing-models-65d092a6f8ab7fc669111b31)：未找到描述
- [DreamGen: AI role-play and story-writing without limits](https://dub.sh/opus-v1-guide)：未找到描述

  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1210177857465229363) (6 messages): 

- **RunPod 镜像神秘消失**：`@stoicbatman` 报告了一个关于 **RunPod 镜像**似乎被删除的问题，并提到难以定位该镜像。
- **Docker 标签的有用指引**：针对这一困惑，`@nanobitz` 提供了一个指向 [Docker Hub](https://hub.docker.com/r/winglian/axolotl-runpod/tags) 的有用链接，可以在那里找到 RunPod 镜像的标签。
- **GitHub Readme 重定向问题**：`@stoicbatman` 指出 **GitHub readme** 没有正确地将用户重定向到实际的 RunPod 镜像，这表明 GitHub 文档可能存在问题。
- **最新链接困境**：`@nanobitz` 询问 `@stoicbatman` 是否拥有最新链接，这表明资源可能存在更新或更改，从而指向正确的 RunPod 镜像位置。

**提到的链接**：

[Docker](https://hub.docker.com/r/winglian/axolotl-runpod/tags)：未找到描述

  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1209851768507342901) (2 messages): 

- **Groq LPU 树立 AI 新基准**：`@srns27` 强调了 [Groq LPU 推理引擎](https://cryptoslate.com/groq-20000-lpu-card-breaks-ai-performance-records-to-rival-gpu-led-industry/) 在大语言模型方面的性能突破，它在最近的基准测试中超越了竞争对手，达到了每秒 241 个 tokens。基准测试详情可在 [Groq 官网](https://wow.groq.com/news_press/groq-lpu-inference-engine-leads-in-first-independent-llm-benchmark/) 和 [ArtificialAnalysis.ai](https://artificialanalysis.ai/models/llama-2-chat-70b) 查看。
  
- **深入探讨 Groq 架构**：`@dpearson` 分享了 Groq 编译器技术主管 Andrew Bitar 的 [YouTube 视频](https://youtu.be/PKJYU9ecvWc?si=9BKG75HsaEGTVgMH)，解释了 Groq 高速背后的架构。这份题为“用于数据流计算的软件定义硬件”的演讲是在 Intel/VMware Crossroads 3D-FPGA 学术研究中心发表的。

**提到的链接**：

- [Groq&#039;s $20,000 LPU chip breaks AI performance records to rival GPU-led industry](https://cryptoslate.com/groq-20000-lpu-card-breaks-ai-performance-records-to-rival-gpu-led-industry/)：Groq 的 LPU 推理引擎（一种专用的语言处理单元）在大语言模型的处理效率方面创下了新纪录。在 ArtificialAnalysis 最近进行的基准测试中……
- [Software Defined Hardware for Dataflow Compute / Crossroads 3D-FPGA Invited Lecture by Andrew Bitar](https://youtu.be/PKJYU9ecvWc?si=9BKG75HsaEGTVgMH)：Groq 编译器技术主管 Andrew Bitar 于 2022 年 12 月 11 日受邀为 Intel/VMware Crossroads 3D-FPGA 学术研究中心做的讲座。摘要：随着……

  

---

### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1209848103360790579) (3 messages): 

- **工具的简洁性**：`@srush1301` 提到在处理简单任务时使用 **Excalidraw**，并强调 **gpu puzzles** 可以与 **chalk-diagrams** 配合使用。
- **发现 Excalidraw**：`@morgangiraud` 表示他们之前不熟悉 `@srush1301` 提到的这个工具。
- **质疑 Triton 的优势**：`@_hazler` 询问 `@745353422043087000`，在 **Triton** 中实现某些功能是否能带来显著的速度提升或新的部署平台支持，还是主要用于教学目的。
  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1209864248767811634) (18 messages🔥): 

- **CUDA 函数指针查询**：用户 `@carrot007.` 询问在 `cudaMemcpyFromSymbol` 过程中遇到警告时，如何在 global 函数中调用 device 函数指针。`@morousg` 建议不要这样做，因为这可能导致效率低下以及类似 `cudaErrorInvalidPc` 的 Bug，并推荐使用 C++ templates 作为替代方案，以保持编译优化。
- **在 Docker 中安装 NVIDIA Nsight**：`@dvruette` 询问在 vast.ai 的 Docker 容器中安装 NVIDIA Nsight 进行调试的经验。`@marksaroufim` 提到在不同云服务商中都遇到过类似问题，并指出 lighting.ai studios 有可行的解决方案。
- **NVIDIA `ncu` 工具在 Docker 中运行正常**：在关于 CUDA profiling 的讨论中，`@lntg` 确认 `ncu` 在 Docker 容器中可以按预期工作，并为 CUDA mode 成员提供支持，包括在其平台上提供快速验证和免费额度。
- **NVIDIA Profiling 工具的性能困扰**：`@complexfilterr` 在尝试对其 CUDA 代码进行 profile 时遇到警告：`==WARNING== No kernels were profiled`。他们提供了所使用的命令：`ncu -o profile --set full ./add_cuda`。
- **发布新的 BnB FP4 仓库**：`@zippika` 为其 bnb fp4 代码创建了一个 [GitHub repository](https://github.com/aredden/torch-bnb-fp4)，并报告其速度比 bitsandbytes 更快。该代码要求 CUDA compute capability >= 8.0。他们还提供了一个详细的 Python 脚本来测试速度对比，并强调了特定模型对 VRAM 的高要求。

**提到的链接**：

[GitHub - aredden/torch-bnb-fp4](https://github.com/aredden/torch-bnb-fp4)：通过在 GitHub 上创建账号来为 aredden/torch-bnb-fp4 的开发做出贡献。

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1209950616219295844) (5 messages): 

- **寻求关于 torch.compile 局限性的澄清**：`@ardywibowo` 询问 *torch.compile 不具备哪些功能*，并对通过 Triton/CUDA 可获得但 torch.compile 可能无法实现的加速类型感到好奇。
- **关于将混合类型 Matmul 公开的查询**：`@jeremyhoward` 寻求关于是否有计划将混合类型矩阵乘法（matmul）公开的信息，以及是否存在任何安全性或实现细节（如 nf4 的使用）。
- **自定义 Kernel vs. PyTorch 原生 Kernel**：`@gogators.` 讨论了有时 PyTorch 的原生 kernel 性能较低，并举例说明在 batch size 为 1 的 1D 卷积中，自定义 kernel 实现了 6 倍的速度提升。然而，对于非研究用例，*常用算子* 的原生 kernel 是高效的。
- **torch.compile 与动态控制流**：`@gogators.` 提到 torch.compile 不能很好地处理 *动态控制流（dynamic control flow）*，但这在神经网络中通常是极少数情况。
- **torch.compile 错失的融合收益**：`@gogators.` 对 torch.compile 复制 *flash-attention* 中所见的 kernel fusion 收益的能力表示怀疑，并强调它可能无法像自定义 kernel 那样针对所有网络架构进行优化。

### CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1210135471230033990) (1 条消息): 

- **Gemini 1.5 讨论邀请**：`@shashank.f1` 邀请大家参加关于 **Gemini 1.5** 的直播讨论。感兴趣的参与者可以通过提供的 [Discord 邀请链接](https://discord.gg/F4FfcQw3?event=1209440306404139008)加入。
- **A-JEPA AI 探索音频语义知识**：同一位用户分享了一个 [YouTube 视频](https://youtu.be/FgcN62LFzIU)，标题为 "A-JEPA AI model: Unlock semantic knowledge from .wav / .mp3 file or audio spectrograms"。该视频承诺深入探讨 **AI 从音频中学习** 的见解，并展示了与多位专家的讨论。

**提到的链接**：

- [加入 hedwigAI Discord 服务器！](https://discord.gg/F4FfcQw3?event=1209440306404139008)：查看 Discord 上的 hedwigAI 社区 —— 与其他 50 名成员一起交流，享受免费的语音和文字聊天。
- [A-JEPA AI 模型：从 .wav / .mp3 文件或音频频谱图中解锁语义知识](https://youtu.be/FgcN62LFzIU)：🌟 解锁 AI 从音频中学习的力量！🔊 观看与 Oliver、Nevil、Ojasvita、Shashank、Srikanth 和 N... 针对 A-JEPA 方法的深度讨论。

  

---


### CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1210136760097439784) (1 条消息): 

- **慕尼黑 SIXT 的 ML Engineer 职位机会**：`@ppeter0480` 发布了慕尼黑 **SIXT 的 ML Engineer** 招聘信息，重点要求 **NLP** 和 **Generative AI** 技能，以及扎实的工程背景。感兴趣的候选人可以通过提供的 [职业链接](https://www.sixt.jobs/en/job/feb00784-a96f-430b-b105-6116b993b472) 申请。该角色包括将业务问题转化为技术解决方案，并利用先进算法提升客户体验。

**提到的链接**：

[立即申请：高级机器学习工程师 (m/f/d) | 慕尼黑](https://www.sixt.jobs/en/job/feb00784-a96f-430b-b105-6116b993b472)：你在慕尼黑的梦想工作：高级机器学习工程师 (m/f/d)。加入 SIXT 团队！我们期待你的申请！

  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1210002654751629362) (12 条消息🔥): 

- **CUDA 编译时间受到质疑**：`@0ut0f0rder` 对简单 CUDA kernel 的缓慢编译时间表示担忧，在使用 **torch_inline** 编译一个 x² kernel 时经历了约 1 分钟的编译时间。
- **在 Numba 中寻求速度**：针对 `@0ut0f0rder` 提出的编译缓慢问题，`@jeremyhoward` 提到虽然 CUDA 的编译时间确实较慢，但 **numba** 是一个更快的替代方案。
- **面对 Groq AI 质疑 CUDA 的持久性**：`@dpearson` 分享了一个 [YouTube 视频](https://youtu.be/PKJYU9ecvWc?t=1906)，讨论了 Groq AI 的新硬件和编译器，引发了关于随着编译器在资源利用方面变得更加高效和自动化，学习 CUDA 是否会过时的辩论。
- **学习 CUDA 仍然有价值**：用户 `@telepath8401` 反驳了 `@dpearson` 关于 CUDA 过时的担忧，强调了从学习 CUDA 中获得的基础知识及其在特定架构或平台之外的价值。
- **PyTorch 'torch_inline' 故障**：`@jrp0` 报告了一个使用 **torch_inline** 生成 `.so` 文件的技术问题，他无法在通过 runpod 启动的 Jupyter notebook 中生成预期文件，而使用 Colab 时则没有问题。

**提到的链接**：

- [torch.cuda.jiterator._create_jit_fn &mdash; PyTorch 2.2 文档](https://pytorch.org/docs/stable/generated/torch.cuda.jiterator._create_jit_fn.html)：未找到描述
- [数据流计算的软件定义硬件 / Andrew Bitar 的 Crossroads 3D-FPGA 特邀讲座](https://youtu.be/PKJYU9ecvWc?t=1906)：Groq 编译器技术负责人 Andrew Bitar 于 2022 年 12 月 11 日在 Intel/VMware Crossroads 3D-FPGA 学术研究中心的特邀讲座。摘要：随着 t...

  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1210128754437914665) (1 条消息): 

- **频道维护提醒**：用户 `andreaskoepf` 提醒所有用户保持 **youtube-recordings** 频道专注于其预期用途，并将无关内容移至相应的频道 *<#1189868872887705671>*。
  

---

### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1209970972472377426) (11 条消息🔥): 

- **CUDA 与 TPU 兼容性查询**：`@drexalt` 正在思考移除针对 TPU 的重复调用是否能使代码与 GPU 兼容。该用户正考虑尝试这种方法。

- **GPU 上的形状维度（Shape Dimension）困扰**：`@iron_bound` 在 GPU 上运行进程时遇到了典型的形状维度错误，但确认程序在崩溃前确实已经启动。

- **AMD GPU 的兼容性问题**：`@mrrational` 报告称在 AMD GPU 上的测试未成功，`@iron_bound` 也证实了这一点，他从未能在其 7900xtx 上成功运行 FA2 训练，即使是使用 Triton 版本。

- **ROCm 的 Flash-Attention 缺少反向传播算子（Backwards Kernel）**：`@iron_bound` 分享了一个 [GitHub 仓库](https://github.com/ROCm/flash-attention/tree/howiejay/navi_support/)，该仓库可能用于在 AMD GPU 上进行推理，但提到它缺少反向传播函数/算子（backwards function/kernel）。

- **排查 AMD 上的 Flash-Attention 问题**：`@drisspg` 告知 PyTorch 中对 Flash Attention v2 的支持有限，可能可以在 AMD GPU 上运行。`@iron_bound` 随后发布了尝试在版本 `2.3.0.dev20240118+rocm6.0` 下使用 `7900xtx` GPU 时收到的错误消息。`@drisspg` 表示如果创建了 issue，他愿意将其转发给 AMD 的代表。

**提到的链接**：

- [GitHub - srush/triton-autodiff: Experiment of using Tangent to autodiff triton](https://github.com/srush/triton-autodiff)：使用 Tangent 对 Triton 进行自动微分（autodiff）的实验。通过在 GitHub 上创建账号来为 srush/triton-autodiff 的开发做出贡献。
- [GitHub - ROCm/flash-attention at howiejay/navi_support](https://github.com/ROCm/flash-attention/tree/howiejay/navi_support/)：快速且内存高效的精确注意力机制。通过在 GitHub 上创建账号来为 ROCm/flash-attention 的开发做出贡献。

---

### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1209820850010722355) (39 条消息🔥): 

- **探索 Flash Attention 机制**：`@nshepperd` 讨论了 Flash Attention 的机制，明确了在 forward pass 期间需要累加器（accumulators），并提到对于 backward pass，现有的 `lse` 使得不再需要 online softmax。详细介绍了算法的工作原理，并提出了节点间梯度和数据的流动方式。
  
- **寻求 Attention 分布示例的贡献**：`@andreaskoepf` 表示对模拟多个 dummy GPUs 之间 Attention 分布的示例 notebook 感兴趣，促使 `@ericauld` 分享了一个正在开发中的算法 dummy 版本，他们指出该版本存在明显的数值不准确问题，这可能源于他们作为参考的 FlashAttention2 论文中的拼写错误。

- **寻找 Attention 算法中的拼写错误**：`@lancerts` 承认并确认了 FlashAttention2 论文中存在的拼写错误（由 `@ericauld` 指出），并提供了修正建议。他们还通过一个 [Pull Request](https://github.com/cuda-mode/ring-attention/pull/8) 为所讨论算法的高亮部分提出了修复方案。

- **快速 PyTorch 转换与调试**：`@iron_bound` 和 `@andreaskoepf` 分别分享了他们在将代码转换为 PyTorch 以及调试现有实现方面的进展，展示了社区驱动的开发模式。Iron Bound 寻求关于 torch distributed 集成的帮助。

- **规划协作 Live Hacking 会议**：`@andreaskoepf` 组织了一场 live hacking 会议，并鼓励大家参与以改进基于 Flash Attention 的 Ring Attention 实现。会议中提出了一个潜在的浮点精度问题，涉及在处理长上下文（long contexts）时 Flash Attention 中使用 FP32 累加的必要性。

**提到的链接**：

- [Google Colaboratory](https://colab.research.google.com/drive/1B9oD4oeuYqK5szEHfrS0VlnVbmLCO9HA#scrollTo=M-lh5Fk7rSLY)：未找到描述
- [Is there an equivalent of jax.lax.scan (eg in torch.func)?](https://discuss.pytorch.org/t/is-there-an-equivalent-of-jax-lax-scan-eg-in-torch-func/177088)：我想将以下 jax 代码（实现 Kalman filter）翻译为 torch。def kf(params, emissions, return_covs=False): F, Q, R = params[&#39;F&#39;], params[&#39;Q&#39;], para...
- [Andreas Köpf (@neurosp1ke) 的推文](https://x.com/neurosp1ke/status/1760558683136589983)：我们今天 19:00 UTC 将在 CUDA MODE Discord 对这个优秀的基于 Flash Attention 的 Ring Attention 实现（>1M 上下文长度）进行 live hack —— 感谢 Zilin Zhu：https://github.com/zhuzilin/ring-flash-atte...
- [ring-attention/notebooks/DummyRingAttentionImpl.ipynb at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/blob/main/notebooks/DummyRingAttentionImpl.ipynb)：Ring Attention 的优化内核 [进行中]。通过在 GitHub 上创建账号为 cuda-mode/ring-attention 的开发做出贡献。
- [GitHub - zhuzilin/ring-flash-attention: Ring attention implementation with flash attention](https://github.com/zhuzilin/ring-flash-attention/)：结合 Flash Attention 的 Ring Attention 实现 - zhuzilin/ring-flash-attention
- [xformers/xformers/ops/fmha/__init__.py at 99ad1723b0b80fb21c5e4dc45446e93752f41656 · facebookresearch/xformers](https://github.com/facebookresearch/xformers/blob/99ad1723b0b80fb21c5e4dc45446e93752f41656/xformers/ops/fmha/__init__.py#L417)：可黑客化且优化的 Transformer 构建块，支持组合式构建。 - facebookresearch/xformers
- [ring-attention/ring_attn/ring_attention.py at tests · Iron-Bound/ring-attention](https://github.com/Iron-Bound/ring-attention/blob/tests/ring_attn/ring_attention.py)：Ring Attention 的优化内核 [进行中]。通过在 GitHub 上创建账号为 Iron-Bound/ring-attention 的开发做出贡献。
- [fix the dummy-nb by lancerts · Pull Request #8 · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/pull/8)：@ericauld
- [ir - 概览](https://github.com/Ir)：ir 有 4 个可用的代码库。在 GitHub 上关注他们的代码。

  

---

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1209792002913210378) (58 messages🔥🔥): 

- **Google Gemini 模型混淆**：`@brknclock1215` 澄清了 *Gemini 模型系列*，分享了 *Gemini Advanced (Ultra 1.0)* 的 [两个月免费试用链接](https://gemini.google.com/advanced) 以及 [Gemini Pro 1.5 的私人预览申请](https://developers.googleblog.com/2024/02/gemini-15-available-for-private-preview-in-google-ai-studio.html?m=1)。此外，他们建议观看 Sam Witteveen 的 YouTube 视频进行实测，并指向了一篇 [解释 Gemini 系列模型的博客文章](https://code.iaflw.com/2024/02/gemini-versus-gemini-understanding.html)。
- **Perplexity AI Discord 机器人查询**：多位用户询问如何在 Discord 中使用或定位 *Perplexity AI 机器人*。`@icelavaman` 和 `@mares1317` 引导用户前往相应频道，`@nocind` 提到一个机器人目前处于离线状态，并对该机器人的“生死”进行了调侃。
- **Pro 版本访问及订阅问题**：用户在访问 *Perplexity Pro 版本* 相关功能时遇到困惑。`@me.lk` 建议重新加入服务器，而 `@mares1317` 提供了 [账单和订阅常见问题页面](https://blog.perplexity.ai/faq/billing-and-subscription) 的链接。`@tree.ai` 和 `@ok.alex` 回答了关于添加团队成员以及 *Gemini Ultra 模型* 可用性的问题。
- **对 API 响应不一致的担忧**：用户反映在使用 *Perplexity AI API* 时遇到响应不一致的问题。`@ok.alex` 承认了该问题，并建议暂时切换到其他模型。
- **申请 Perplexity Pro 访问权限及 AI 功能**：用户就获取 *Perplexity Pro 频道* 访问权限进行交流，并询问新发布的功能和模型。`@gooddawg10` 焦急等待 *GPT vision 连接到网络* 的更新，`@ok.alex` 承诺会及时通知社区。

**提到的链接**：

- [Code is a Four Letter Word: Gemini Versus Gemini: Understanding Google's Latest... Thing](https://code.iaflw.com/2024/02/gemini-versus-gemini-understanding.html)：未找到描述
- [Discover Daily by Perplexity](https://www.youtube.com/playlist?list=PLKwRkjCH760ObtANfb0-Kat2XlvB5dKxf)：我们希望将世界上的故事带到您的耳边，每日融合科技、科学和文化。每一集都基于我们的 Discover 提要精心制作...
- [Gemini 1.5: Our next-generation model, now available for Private Preview in Google AI Studio - Google for Developers](https://developers.googleblog.com/2024/02/gemini-15-available-for-private-preview-in-google-ai-studio.html?m=1)：未找到描述
- [Sam Witteveen](https://www.youtube.com/@samwitteveenai)：大家好，我是 Sam Witteveen，我从事深度学习工作 9 年，研究 Transformer 和 LLM 超过 5 年。我在 2017 年被任命为 Google 机器学习开发者专家，目前...
- [Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1760451622537158921?s=20)：介绍 Perplexity Labs 的新成员：体验以轻量级且性能卓越著称的 Gemma 2B 和 7B 模型。立即在 http://labs.pplx.ai 尝试。
- [Billing and Subscription](https://blog.perplexity.ai/faq/billing-and-subscription)：浏览 Perplexity 博客以获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1209884302393872394) (3 messages): 

- **探索加密货币机制**：`@ivanrykovski` 分享了一个关于 **`dy/dx`** 具体细节的 [Perplexity AI 搜索](https://www.perplexity.ai/search/what-does-dydx-Vo_6.U1XQg.eDbP_lg0FHQ?s=c)，这是一个与加密货币和衍生品交易相关的术语。
- **天然口腔健康方案**：受 **Andrew Huberman 和 Paul Saladino** 内容的启发，`@uberkoolsound` 讨论了在口腔护理中减少使用加工化学品的转变。他们分享了一个关于盐水作为潜在天然疗法益处的 [Perplexity AI 搜索](https://www.perplexity.ai/search/Does-salt-water-muDDr.Z9RHy_EvMKEhurPg?s=c)。
- **查询金融工具定义**：`@swordfish01` 发布了一个没有上下文的 [Perplexity AI 搜索](https://www.perplexity.ai/search/What-is-a-fDAg8dSNRhmEeKU.SoY6Fg?s=c)，推测是在询问特定的金融工具或概念。
  

---

### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1209818216394915860) (20 messages🔥): 

- **API 与网站响应不一致**：`@iflypper` 表达了 API 提供的答案与过时的网站相比存在差异的困扰。他们分享了一段代码，以寻求更准确的实现方式。

- **简化查询以获得更好的响应**：`@brknclock1215` 建议保持 API 查询简单以获得更好的性能，因为复杂或多方面的查询往往处理起来比较吃力。

- **API 模型行为困扰用户**：在 `@iflypper` 删除了 system prompt 并收到无关响应后，`@brknclock1215` 讨论了这一想法，但回想起系统消息可能不再被忽略，并引用了[更新后的文档](https://docs.perplexity.ai/docs/model-cards)。

- **pplx-70b-online 的乱码响应**：`@useful_tom` 报告称从 pplx-70b-online 模型收到了乱码响应，并指出其他人也遇到了类似问题。`@icelavaman` 提到团队正在调查此事，而 `@brknclock1215` 建议尝试其他 online 模型作为变通方法。

- **支付问题与潜在新功能**：`@jenish_79522` 提到在完成 API 额度支付时遇到问题，`@karan01993` 询问了关于在 Perplexity AI API 中集成 Google [GEMMA](https://ai.google.dev/gemma) 的支持情况。

**提到的链接**：

- [Supported Models](https://docs.perplexity.ai/docs/model-cards)：未找到描述
- [no title found](https://ai.google.dev/gemma)：未找到描述

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1209767513404084234) (38 messages🔥): 

- **动态类生成问题**：`@deltz_81780` 在尝试动态生成用于 **PydanticOutputFunctionsParser** 的类时遇到了 **ValidationError**。他们分享了代码片段和错误消息，寻求帮助。

- **关于 Agent 类型和用途的讨论**：`@problem9069` 询问了不同类型的 Agent，例如 **OpenAITools** 和 **OpenAIFunctions**，并详细说明了预期的模型类型和功能。他们质疑是否有必要学习所有类型，或者其中是否有一种首选类型。

- **LinkedIn Learning 课程亮点**：`@mjoeldub` 分享了一个新的 **LinkedIn Learning 课程**信息，该课程重点关注 **LangChain 和 LCEL**，并附带了[课程链接](https://www.linkedin.com/learning/introduction-to-ai-orchestration-with-langchain-and-llamaindex)。

- **新的 LangChain AI 教程提醒**：`@a404.eth` 发布了一个新教程 "Chat with your PDF"，这是一个**使用 LangChain AI 从零开始构建 RAG** 的教程，提到了 LangSmith 的使用和对话历史记录的改进，并呼吁在 Twitter 帖子链接中提供反馈。

- **支持模式讨论**：`@mysterious_avocado_98353` 对频道中的 LangChain 支持表示失望，随后 `@renlo.` 回应并强调了通过其[定价页面](https://www.langchain.com/pricing)提供的付费支持选项。

**提到的链接**：

- [Agent Types | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/agents/agent_types/.)：该页面从几个维度对所有可用的 Agent 进行了分类。
- [Austin Vance (@austinbv) 的推文](https://x.com/austinbv/status/1760320228725309951?s=46)：🚨 新教程 🚨 我的 "Chat with your PDF" 使用 @LangChainAI 从零构建 RAG 教程的终结篇！在第 4 部分中，我们：- 使用 LangSmith 处理一切 - 实现 Multi Query 以增加检索...
- [Pricing](https://www.langchain.com/pricing)：适用于任何规模团队的方案。
- [langgraph/examples/multi_agent/agent_supervisor.ipynb at main · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/agent_supervisor.ipynb)：通过在 GitHub 上创建账号为 langchain-ai/langgraph 的开发做出贡献。
- [Survey on your Research Journey](https://forms.gle/8N4DsuCWtCXKxLSv6)：为了彻底改变学术和商业研究，EurekAI 寻求您的见解，以便根据您的需求定制我们的工具。无论您是沉浸在研究中还是偶尔参与，您的...

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1210069768959434802) (1 messages): 

- **LangSmith API 批量摄取失败**：`@jacobito15` 遇到了一个警告，指出由于 `LangSmithError` 导致批量摄取运行失败。错误提示 `ChannelWrite` 名称超过 128 个字符，导致端点 `https://api.smith.langchain.com/runs/batch` 出现 HTTP 422 错误。
  

---

### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1209943158230876190) (3 messages): 

- **征求意见与测试者**：用户 `@pk_penguin` 就一个未指明的话题公开征求意见并提供试用。感兴趣的用户请私信了解详情。

- **释放并行函数调用能力**：`@gokusan8896` 分享了一个关于在**任何 LLM 模型中启用并行函数调用 (Parallel Function Calls)** 的 LinkedIn 帖子链接。该功能可以显著提升效率和能力，帖子包含更多详情：[探索并行函数调用](https://www.linkedin.com/feed/update/urn:li:activity:7166408137002962944/)。

- **聚合查询平台/库咨询**：`@rogesmith` 正在考虑是否继续开发一个允许用户聚合查询文档数据而非单独查询的平台/库。该消息旨在征求关于该项目潜在公共实用性的反馈。
  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=Eb7QF1nDWGU
  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1209856987861352469) (29 messages🔥): 

- **Google 的开源模型**：用户 `@sebastian.bodza` 分享了一个关于 Google 名为 Gemma 的开源模型的 Kaggle 链接，引发了另一位用户 `@philipmay` 对这些模型语言多样性（特别是德语）的询问。链接指向 [Kaggle 上的 Google Gemma 模型](https://www.kaggle.com/models/google/gemma)。
  
- **Hugging Face 托管 Gemma 模型**：用户 `@bjoernp` 提供了 Hugging Face 上 Gemma 指令微调版本 (Instruct version) 的链接，指出其许可证具有商业可行性，并提到其拥有 256k 的巨大词表大小。在此查看 [Hugging Face 上的 Gemma 模型](https://huggingface.co/google/gemma-7b-it)。

- **对 Aleph Alpha 模型更新的质疑**：用户 `@sebastian.bodza` 强调了 Aleph Alpha 模型的更新，但对其质量表示不确定。用户 `@devnull0` 指出 Andreas Köpf 已加入 Aleph Alpha，这可能会提高未来对该公司模型的期望。

- **Aleph Alpha 的更新日志与批评**：`@devnull0` 根据其 [更新日志 (changelog)](https://docs.aleph-alpha.com/changelog/) 分享了 Aleph Alpha 模型的变更，但遭到 `_jp1_` 的批评，认为其缺乏基准测试或示例，`@sebastian.bodza` 随后评论提到新模型缺乏指令微调 (Instruction tuning)。

- **性能担忧**：关于 Gemma 和 Aleph Alpha 模型在各种语言和语境下的性能讨论不断。`@bjoernp` 发布了 Gemma 令人失望的德语评估结果，而 `@devnull0` 分享了一条暗示 Llama 模型存在性能问题的推文，链接了 @ivanfioravanti 确认问题的推文 ([推文](https://fxtwitter.com/ivanfioravanti/status/1760423676376211673?t=OBZ02Et7P_B4oZYOjgJYpA&s=19))，以及另一条由 @rohanpaul_ai 发布的推文，在基准测试套件中将 Gemma-2b 与 phi-2 进行对比，结果不尽如人意 ([推文](https://fxtwitter.com/rohanpaul_ai/status/1760566473859408276?t=QZLSGE7d50DIlwyhW8bx3w&s=19))。

**提到的链接**：

- [Gemma](https://www.kaggle.com/models/google/gemma): Gemma 是一个轻量级、开放模型系列，基于 Google 用于创建 Gemini 模型的相同研究和技术构建。
- [Rohan Paul (@rohanpaul_ai) 的推文](https://fxtwitter.com/rohanpaul_ai/status/1760566473859408276?t=QZLSGE7d50DIlwyhW8bx3w&s=19): 在 Nous 的基准测试套件上，Gemma-2b 的表现大幅落后于 phi-2 https://eqbench.com/
- [HuggingChat](https://huggingface.co/chat): 让每个人都能使用社区最好的 AI 聊天模型。
- [博客 | Aleph Alpha API](https://docs.aleph-alpha.com/changelog/): 博客
- [ifioravanti (@ivanfioravanti) 的推文](https://fxtwitter.com/ivanfioravanti/status/1760423676376211673?t=OBZ02Et7P_B4oZYOjgJYpA&s=19): 在 @ollama 和 Apple MLX 上对 gemma 模型进行初步测试后，我可以肯定地说：- llama.cpp 存在一些问题，修复正在进行中 https://github.com/ggerganov/llama.cpp/pull/5631 - temperat...
- [google/gemma-7b-it · Hugging Face](https://huggingface.co/google/gemma-7b-it): 未找到描述
- [Chibb - German-English False Friends in Multilingual Transformer Models- An Evaluation on Robustness and Word-to-Word Fine-Tuning.pdf](https://drive.google.com/file/d/1jgq0nBnV-UiYNxbKNrrr2gxDEHm-DMKH/view?usp=share_link): 未找到描述
- [flozi00/dibt-0.1-german · Hugging Face 数据集](https://huggingface.co/datasets/flozi00/dibt-0.1-german): 未找到描述
- [DIBT/10k-prompt-collective · Hugging Face 数据集](https://huggingface.co/datasets/DIBT/10k-prompt-collective): 未找到描述

  

---

### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1210153792528715836) (1 messages): 

- **Batch Size 影响性能**：用户 `@calytrix` 强调了一个潜在问题，即使用 **Batch Size 大于 1** 可能会对模型评分产生负面影响，并参考了 [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/82) 上的讨论。
- **寻求指标重新生成代码**：`@calytrix` 还询问是否有可用的 **脚本或代码** 来重新生成特定博客文章中的所有指标。
- **模型的测试公平性标准**：`@calytrix` 分享了关于什么是公平测试的看法，指出测试应该是 **现实的 (realistic)**、**无歧义的 (unambiguous)**、**无运气成分的 (luckless)** 且 **易于理解的 (easy to understand)**。他们通过示例详细说明了如何识别不公平的测试。

**提到的链接**：

[HuggingFaceH4/open_llm_leaderboard · MMLU blog post discussion](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/82)：未找到描述

  

---



### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1210013169993383956) (1 messages): 

- **寻求 Neuralink 面试建议**：`@xilo0` 正处于 **Neuralink** 面试的后期阶段，正在寻求关于如何回答“卓越能力的证据 (evidence of exceptional ability)”这一问题的建议。他们正在考虑展示哪些项目，并向其他申请过 Elon Musk 旗下公司的人寻求见解。
  

---


### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1209805611643047967) (4 messages): 

- **通过自我反思增强 Self RAG**：`@pradeep1148` 分享了一个 [名为“使用 LangGraph 的 Self RAG”的 YouTube 视频](https://www.youtube.com/watch?v=Eb7QF1nDWGU)，讨论了自我反思如何通过纠正低质量的检索或生成来改进检索增强生成 (RAG)。

- **评估大语言模型中的微调**：`@pradeep1148` 发布了另一个 [名为“BitDelta：你的微调可能只值 1 bit”的 YouTube 视频](https://www.youtube.com/watch?v=T_dYzuv4N70)，该视频质疑了在实际影响可能微乎其微的情况下，对大语言模型 (LLM) 进行微调的价值。

- **谷歌开源 Gemma 模型介绍**：在持续的资源分享中，`@pradeep1148` 展示了一个 [视频](https://www.youtube.com/watch?v=953U3FxHF-Q)，详细介绍了 “Gemma”，这是谷歌的开源模型，与最先进的 Gemini 模型属于同一家族。

**提到的链接**：

- [Gemma Google's open source SOTA model](https://www.youtube.com/watch?v=953U3FxHF-Q)：Gemma 是一个轻量级、最先进的开源模型家族，基于与创建 Gemini 模型相同的研究和技术构建。由 Goo... 开发。
- [Self RAG using LangGraph](https://www.youtube.com/watch?v=Eb7QF1nDWGU)：自我反思可以增强 RAG，从而能够纠正低质量的检索或生成。最近的几篇论文都集中在这个主题上，但实现...
- [BitDelta: Your Fine-Tune May Only Be Worth One Bit](https://www.youtube.com/watch?v=T_dYzuv4N70)：大语言模型 (LLM) 通常分为两个阶段进行训练：在互联网规模的大型数据集上进行预训练，以及针对下游任务进行微调。鉴于...

  

---


### Skunkworks AI ▷ #[papers](https://discord.com/channels/1131084849432768614/1156310031768232007/) (1 messages): 

nagaraj_arvind: 我在最后提到了 KTO。但没有深入探讨细节。
  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1209925830365683773) (2 messages): 

- **谷歌发布 Gemini Pro 1.5**：`@simonw` 强调了最近发布的 [Google Gemini Pro 1.5](https://simonwillison.net/2024/Feb/21/gemini-pro-video/)，称赞其 *1,000,000 token 的 Context Size*，这使 Claude 2.1 和 GPT-4-Turbo 等竞争对手相形见绌。更值得注意的是，他对 **模型使用视频作为输入的能力** 感到兴奋，这是通过 [Google AI Studio](https://aistudio.google.com/app/prompts/new_chat) 探索的一项功能。

- **谷歌新的机器学习文档**：正如 `@derekpwillis` 所分享的，谷歌在其 [Google AI 开发者网站](https://ai.google.dev/gemma/docs) 上发布了其机器学习产品的新文档。没有讨论关于该文档或其内容的进一步细节。

**提到的链接**：

- [The killer app of Gemini Pro 1.5 is video](https://simonwillison.net/2024/Feb/21/gemini-pro-video/)：上周谷歌推出了 Gemini Pro 1.5，这是对其 Gemini 系列 AI 模型的重大升级。Gemini Pro 1.5 拥有 1,000,000 token 的 Context Size。这是巨大的——此前那个……
- [无标题](https://ai.google.dev/gemma/docs)：未找到描述

  

---

### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1210110851663798283) (4 messages): 

- **故障排除集成问题**：`@simonw` 就 `@887493957607645184` 的系统集成问题进行了沟通，并建议如果问题尚未解决，请向 gpt4all 团队报告。
- **探索 LLM 的文件支持**：`@simonw` 回应了 `@314900216124014623` 关于为 LLM 添加文件支持的查询，并建议从 **GPT-Vision** 的图像支持开始。对于 PDF，他建议使用工具提取文本后再输入 LLM。
- **Gemma 模型实现障碍**：`@simonw` 尝试运行 Google 的新 **Gemma model**，但遇到了输出问题，只收到了占位符文本而非预期结果。他还指出需要使用 `llm python` 命令更新 `llama-cpp-python`。
  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (1 messages): 

scopexbt: 大家好，我找不到任何关于 token 的信息，我们有吗？
  

---


### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1210174238657486928) (2 messages): 

- **分享 GLAN 论文**：`@.benxh` 询问是否有人正在研究 **Gradient Layerwise Adaptive-Norms (GLAN)**，并分享了 [GLAN 论文](https://arxiv.org/pdf/2402.13064.pdf)。
- **对 GLAN 表示兴趣**：`@entropi` 用简洁的 "Whoa, nice" 对 GLAN 概念表示了兴趣。
  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (1 messages): 

res6969: 远离 Salesforce，这将是你作为一家公司犯下的最大错误。
  

---


### LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (1 messages): 

potrock: https://blog.google/technology/developers/gemma-open-models/
  

---


### LLM Perf Enthusiasts AI ▷ #[embeddings](https://discord.com/channels/1168579740391710851/1168744166138859580/1209869748431491204) (1 messages): 

- **ContrastiveLoss 赢得 dartpain 的青睐**：`@dartpain` 在调整 embeddings 时表达了对 **ContrastiveLoss** 的偏好，强调了它对调整的影响。他们还提到 *MultipleNegativesRankingLoss* 是其青睐的损失函数。
  

---



### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1209913190599622706) (3 messages): 

- **加入关于 Gemini 1.5 的讨论**：`@shashank.f1` 邀请大家参加关于 **Gemini 1.5** 的实时讨论，同时回顾了上一次关于 *A-JEPA AI model* 的会议，该会议讨论了如何从音频文件中解锁语义知识。查看 [YouTube](https://youtu.be/FgcN62LFzIU) 上的往期会议。
- **Yikesawjeez 充满风格的规划**：`@yikesawjeez` 正在考虑将他们的活动移至周末，以便有更多时间在 Twitter 上与 `@llamaindex` 联系并寻找赞助商。他们还提到需要着手启动他们的 Devpost 页面。

**提及的链接**：

- [加入 hedwigAI Discord 服务器！](https://discord.gg/F4FfcQw3?event=1209440306404139008)：查看 Discord 上的 hedwigAI 社区 - 与其他 50 名成员一起交流，享受免费的语音和文字聊天。
- [A-JEPA AI 模型：从 .wav / .mp3 文件或音频频谱图中解锁语义知识](https://youtu.be/FgcN62LFzIU)：🌟 解锁音频 AI 学习的力量！🔊 观看与 Oliver, Nevil, Ojasvita, Shashank, Srikanth 和 N... 深入讨论 A-JEPA 方法。