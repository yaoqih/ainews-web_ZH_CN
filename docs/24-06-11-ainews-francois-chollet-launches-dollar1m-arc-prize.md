---
companies:
- openai
- apple
- togethercompute
date: '2024-06-11T23:42:03.241872Z'
description: '**弗朗索瓦·肖莱（François Chollet）** 批评了当前通往**通用人工智能（AGI）**的路径，强调了建立不易饱和、且侧重于技能习得和开放式问题解决的基准测试的重要性。**ARC-AGI**
  谜题体现了“人类易做，AI 难解”的挑战，旨在衡量通往 AGI 的进展。


  与此同时，**苹果**宣布通过与 **OpenAI** 合作，将 **ChatGPT** 集成到 iOS、iPadOS 和 macOS 中，在采取隐私保护措施的前提下，实现文档摘要和照片分析等
  AI 驱动的功能。相关讨论凸显了苹果对深度 AI 集成和端侧模型的关注，这些模型通过混合精度量化等技术进行了优化，尽管人们对其 AI 能力与 **GPT-4**
  相比仍存有一些疑虑。此外，**Together Compute** 推出了一种“智能体混合”（Mixture of Agents）方法，在 **AlpacaEval
  2.0** 上表现强劲。'
id: 011d8801-60a5-4eab-a7c5-330020cf23b3
models:
- gpt-4
- chatgpt
original_slug: ainews-francois-chollet-launches-1m-arc-prize
people:
- francois-chollet
- karpathy
- svpino
- philschmid
- clementdelangue
- sama
- gdb
- miramurati
- kevin-weil
- sarah-friar
title: 弗朗索瓦·肖莱（Francois Chollet）发起 100 万美元 ARC 奖金。
topics:
- benchmarking
- agi
- pattern-recognition
- skill-acquisition
- privacy
- on-device-ai
- mixed-precision-quantization
- mixture-of-experts
- multimodality
- agentic-ai
---

<!-- buttondown-editor-mode: plaintext -->**不可记忆的 Benchmark 才是你所需要的。**

> 2024年6月10日至6月11日的 AI 新闻。
我们为你检查了 7 个 subreddit、[**384** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 以及 **30** 个 Discord（**412** 个频道和 **2774** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**313 分钟**。

在[本周末的 Latent Space 播客中，我们讨论了测试集污染和 Benchmarking 的科学](https://www.latent.space/p/iclr-2024-benchmarks-agents)，今天该领域的一位元老带着解决方案回来了——生成一系列模式识别与补全的 Benchmark：

 
![image.png](https://assets.buttondown.email/images/17cec8b3-b977-41dd-a6c5-f5aad99b812a.png?w=960&fit=max)
 

你可以亲自尝试 ARC-AGI 谜题，感受一下什么是“对人类简单但对 AI 困难”的谜题：

 
![image.png](https://assets.buttondown.email/images/30ef397c-2045-45aa-a3b2-b6a10d41e64b.png?w=960&fit=max)
 

这一切都基于对 AGI 的一种带有鲜明观点的定义，该团队优雅地提供了这一观点：

> 定义 AGI
> 
> 共识但错误：
> **AGI 是一个能够自动化大部分具有经济价值的工作的系统。**
> 
> 正确：
> **AGI 是一个能够高效获取新技能并解决开放式问题的系统。**
> 
> 定义很重要。我们将它们转化为 Benchmark 来衡量 AGI 的进展。
> 没有 AGI，我们将永远无法拥有能够与人类一起发明和发现的系统。

这个 Benchmark 经过特殊设计，旨在抵御其他 Benchmark 所面临的经典的 1-2 年饱和周期：

 
![image.png](https://assets.buttondown.email/images/aacc028a-1899-4c79-a6c9-1321690a668f.png?w=960&fit=max)
 

[解决方案指南](https://arcprize.org/guide) 提供了 François 对极具前景的方向的思考，包括 Discrete program search、技能获取以及混合方法。

上周 Dwarkesh 播客因预测 [2027 年实现 AGI](https://www.dwarkeshpatel.com/p/leopold-aschenbrenner) 而引起轰动，今天 [它又回来了](https://www.dwarkeshpatel.com/p/francois-chollet)，François Chollet 断言我们目前所走的道路不会通向 AGI。AGI 观察者们，你们怎么看？



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 简报

> 所有简报均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**Apple 将 ChatGPT 集成至 iOS, iPadOS 和 macOS**

- **OpenAI 合作伙伴关系**：[@sama](https://twitter.com/sama/status/1800237314360127905) 和 [@gdb](https://twitter.com/gdb/status/1800237897871921435) 宣布 Apple 正与 OpenAI 合作，计划于今年晚些时候将 ChatGPT 集成到 Apple 设备中。[@miramurati](https://twitter.com/miramurati/status/1800371566464880663) 和 [@sama](https://twitter.com/sama/status/1800240506318037208) 欢迎 Kevin Weil 和 Sarah Friar 加入 OpenAI 团队以支持此项工作。
- **AI 功能**：Apple Intelligence 将在各类 App 中实现 AI 驱动的功能，如**总结文档、分析照片以及与屏幕内容交互**。[@karpathy](https://twitter.com/karpathy/status/1800242310116262150) 指出了 AI 逐步集成到操作系统中的路径，从多模态 I/O 到 Agent 能力。
- **隐私担忧**：尽管 Apple 提供了“私有云计算”（Private Cloud Compute）保证，但一些人对 Apple 与 OpenAI 共享用户数据表示怀疑。[@svpino](https://twitter.com/svpino/status/1800449867384258702) 详细介绍了 **Apple 采取的安全措施**，例如端侧处理和差分隐私（differential privacy）。

**对 Apple WWDC AI 发布内容的反应**

- **反应不一**：虽然一些人对 Apple 的 AI 集成印象深刻，但另一些人认为 Apple 已经落后或过于依赖 OpenAI。[@karpathy](https://twitter.com/karpathy/status/1800223553989886447) 和 [@far__el](https://twitter.com/far__el/status/1800237517649678598) 质疑 Apple 是否能**独立交付高性能的 AI**。
- **与其他模型对比**：Apple 的端侧模型似乎**优于其他小模型**，但其服务器端模型仍落后于 GPT-4。[@_philschmid](https://twitter._philschmid/status/1800414656000938439) 指出 Apple 正在使用 Adapter 和混合精度量化（mixed-precision quantization）来优化性能。
- **侧重集成**：许多人注意到 Apple 关注的是**深度、无缝的 AI 集成**，而非模型规模。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1800231734262337936) 赞扬了推动端侧 AI 以提升用户体验和隐私的做法。

**AI 研究与应用的进展**

- **Mixture of Agents (MoA)**：[@togethercompute](https://twitter.com/togethercompute/status/1800536106729157054) 推出了 MoA，它利用多个开源 LLM **在 AlpacaEval 2.0 上获得了 65.1% 的评分**，超越了 GPT-4。
- **AI 推理挑战赛 (ARC)**：[@fchollet](https://twitter.com/fchollet/status/1800577019979411560) 和 @mikeknoop 启动了 **100 万美元的 ARC 奖金**，旨在创建能够适应新奇事物并解决推理问题的 AI，引导该领域重新向 AGI 迈进。
- **语音与视觉进展**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1800496540672508261) 展示了 **Imagen 3 生成具有复杂纹理的丰富图像的能力**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1800365825703972946) 分享了 Microsoft 的 VALL-E 2，它在零样本（zero-shot）文本转语音方面达到了人类水平。
- **AI 应用**：示例包括 @adcock_brett 关于 **Figure 机器人制造**的更新，@vagabondjack 为 @brightwaveio 的 **AI 驱动财务分析筹集了 600 万美元种子轮融资**，以及 @AravSrinivas 提到 **Perplexity 已成为出版商的主要推荐流量来源**。

**梗与幽默**

- [@jxmnop](https://twitter.com/jxmnop/status/1800220386711470249) 用一张幽默的图片开玩笑说 **LLM 已经触及了人类知识的边界**。
- [@nearcyan](https://twitter.com/nearcyan/status/1800338495036383357) 用一个梗嘲讽了那些反复声称 **“NVIDIA 完蛋了”** 的言论。
- [@far__el](https://twitter.com/far__el/status/1800245080563011611) 调侃道：“Apple Intelligence 将成为最大规模的工具使用型 AI 部署，我希望有人能在 @aidotengineer 上分享其设计考量！”以此回应 @swyx 对 Apple 工程师分享见解的呼吁。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 进展**

- **苹果与 OpenAI 合作，将 GPT-4o 集成到 iOS、iPadOS 和 macOS 中**：在 /r/OpenAI 中，苹果发布了 "[Apple Intelligence](https://www.reddit.com/r/OpenAI/comments/1dcqp4m/apple_unveils_apple_intelligence_at_wwdc_live/)"，这是一款内置于其操作系统的个人 AI 系统。它集成了 OpenAI 的 GPT-4o 并支持端侧运行，增强了 Siri、写作工具并实现了图像生成功能。
- **苹果端侧 LLM 架构细节披露**：在 /r/LocalLLaMA 中，分享了更多关于 [Apple Intelligence 底层工作原理](https://www.reddit.com/r/LocalLLaMA/comments/1dcyo80/apple_intelligence_on_device_llm_details/) 的细节，包括使用名为 "adapters" 的量化任务特定 LoRAs，并针对推理性能进行了优化，同时利用“语义索引”来获取个人上下文。
- **AMD 为 AI 处理器发布开源 LLVM 编译器**：AMD 推出了 [Peano](https://videocardz.com/newz/amd-launches-peano-an-open-source-llvm-compiler-for-ryzen-ai-xdna-and-xdna2-npus)，这是一个针对其 Ryzen AI 处理器中 XDNA 和 XDNA2 神经网络处理单元（NPUs）的开源 LLVM 编译器。
- **微软为 Visual Studio Code 推出 AI Toolkit**：在 /r/LocalLLaMA 中，讨论了微软新的 [VS Code AI Toolkit 扩展](https://www.reddit.com/r/LocalLLaMA/comments/1dd0k9y/microsoft_ai_toolkit_for_visual_studio_code/)，它为各种模型提供了 Playground 和微调功能，并支持在本地或 Azure 上运行。

**研究与基准测试**

- **研究发现 RLHF 会降低 LLM 的创造力和输出多样性**：发布在 /r/LocalLLaMA 的一篇[新研究论文](https://www.reddit.com/r/LocalLLaMA/comments/1dd3z73/new_research_shows_rlhf_heavily_reduces_llm/)显示，虽然像 RLHF 这样的对齐技术减少了有害和偏见内容，但它们也限制了大语言模型的创造力，甚至在与安全无关的语境中也是如此。
- **针对 LLM 推理的廉价 AWS 实例基准测试**：在 /r/LocalLLaMA 中，对运行 Dolphin-Llama3 的[各种 AWS 实例进行的基准测试](https://www.reddit.com/r/LocalLLaMA/comments/1dclmwt/benchmarking_inexpensive_aws_instances/)显示，g4dn.xlarge 以 0.58 美元/小时的价格提供了最佳性价比，其中 GPU 速度是关键因素。更大的内存允许在输出中使用更高的 Token 数量。

**Stable Diffusion 3 及更多内容**

- **Stable Diffusion 3 的重要性及突出特性解析**：/r/StableDiffusion 的一个帖子分析了[为什么 SD3 是一个重大进步](https://www.reddit.com/r/StableDiffusion/comments/1dcuval/the_importance_of_stable_diffusion_3_its_standout/)，其全新的 16 通道 VAE 可以捕捉更多细节，实现更快的训练和更好的低分辨率效果。其多模态架构符合 LLM 研究趋势，预计将推动 ControlNets 和 adapters 等技术的发展。

**其他**

- **提升 CPU+RAM 推理速度约 40% 的技巧**：在 /r/LocalLLaMA 中，一位用户分享了一个[提高 tokens/sec 的技巧](https://www.reddit.com/r/LocalLLaMA/comments/1dcpdoc/trick_to_increase_inference_on_cpuram_by_40/)：在 BIOS 中启用 XMP，使 RAM 以规格带宽运行，而不是 JEDEC 默认值。内存超频可能会带来进一步提升，但存在不稳定的风险。
- **使用 BeyondLLM 0.2.1 简化 RAG 的可观测性**：/r/LocalLLaMA 的一个帖子解释了 [BeyondLLM 0.2.1 如何更轻松地为 LLM 和 RAG 应用添加可观测性](https://www.reddit.com/r/LocalLLaMA/comments/1dcljqk/observability_in_rag/)，从而允许跟踪响应时间、Token 使用情况和 API 调用类型等指标。

**迷因与幽默**

- **AI 理想与现实对比梗图**：在子版块中分享的一张[有趣的图片](https://i.redd.it/pcfrnkmrdp5d1.jpeg)，对比了 AI 系统被夸大的预期与实际能力。

---

# AI Discord 回顾

1. **Apple 首次亮相重大 AI 创新**：
   - 在 WWDC 2024 上，Apple 发布了 **[Apple Intelligence](https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/)**，这是一个深度集成于 iPhone、iPad 和 Mac 的 AI 系统。主要功能包括 **ChatGPT 集成到 Siri**、AI 写作工具，以及用于安全处理复杂任务的新型“**Private Cloud Compute**”。基准测试显示 Apple 的 **[on-device and server models](https://x.com/ldjconfirmed/status/1800355063120151031)** 在指令遵循和写作方面表现出色。然而，围绕 **用户隐私** 的担忧以及埃隆·马斯克因 OpenAI 集成而发出的在公司内禁用 Apple 设备的警告引发了辩论。

2. **模型压缩与优化策略**：
   - 工程师们积极讨论了对 **LLaMA 3** 等大语言模型进行 **quantizing**（量化）、**pruning**（剪枝）和优化的技术，以减小模型体积并提高效率。分享了 **[LLM-Pruner](https://github.com/horseee/LLM-Pruner)** 和 **[Sconce](https://github.com/satabios/sconce)** 等资源，并就 **FP8** 等低精度格式的稳定性展开了辩论。探索了 **LoRA**、**8-bit casting** 和 **offloading to CPU** 等优化手段，以解决训练过程中的 Out-of-Memory (OOM) 错误。
   - 工程师们讨论了使用 **offloading optimizer state to CPU** 和 **bnb 8bit casting** ([VRAM Calculator](https://vram.asmirnov.xyz/)) 等策略来 **克服 Out of Memory (OOM) 错误**，并重点介绍了 **Low-Rank Adapters (LoRA)** 等技术。
   - 社区交流分享了关于 **fine-tuning 挑战** 的见解，并提供了实际案例和资源，如 [YouTube tutorial](https://www.youtube.com/watch?v=jyOqtw4ry2w&themeRefresh=1)。

3. **令人兴奋的开源和基准测试新闻**：
   - **Stable Diffusion 3 (SD3)** 令成员们感到兴奋，其目标是更好的体素艺术（voxel art），而对 Huggingface 和 Civitai 等模型平台的比较引发了关于最佳上采样方法和可用性的辩论 ([SD3 Announcement](https://glif.app/@Oliveira/glifs/clw44qfbl0000m0zztwqk2tnf))。
   - **Hugging Face** 扩展了 **AutoTrain** 并增加了 **Unsloth support** ([Announcement](https://x.com/abhi1thakur/status/1800511251145015393))，通过增强的内存管理简化了大模型 fine-tuning。
   - **语言和多模态模型的进展**：AI 社区见证了令人兴奋的突破，包括用于自回归图像生成的 **[LlamaGen](https://arxiv.org/abs/2406.06525)**、在零样本（zero-shot）文本转语音合成中达到人类水平的 **[VALL-E 2](https://arxiv.org/abs/2406.05370)**，以及来自 CAMB AI、承诺在语音克隆中实现更高真实度的 **[MARS5 TTS](https://github.com/camb-ai/mars5-tts)**。讨论探索了用于高效模型部署的 **IQ4_xs** 和 **HQQ** 等量化技术，以及 **federated learning** 在隐私保护训练中的潜力。

4. **社区协作应对 AI 挑战**：
   - 在引人入胜的讨论帖中，重点讨论了医疗应用中的 **Chain of Thought retrieval** 以及核心模型 prompt engineering 技术 ([YouTube tutorial](https://www.youtube.com/watch?v=3eq84KrdTWY&t=665s))。
   - **OpenAccess AI Collective** 分享了一个适合初学者的 [RunPod Axolotl tutorial](https://docs.runpod.io/tutorials/pods/fine-tune-llm-axolotl)，简化了模型训练流程。

5. **量化与模型部署见解**：
   - 关于 Llama 3 的 **4-bit quantization** 交流以及使用 **Tensor Parallelism** 的建议展示了来自 AI 社区的实践经验 ([Quantization Blog](https://stephenpanaro.com/blog/llm-quantization-for-iphone))。
   - 讨论了 **DeepSeek-LLM-7B** 模型的 **LLaMA-based structure** 及其可解释性 ([DeepSeek Project](https://github.com/Butanium/llm-latent-language/blob/1054015066a4fa20386765d72601d03aa7ef5887/utils.py#L334))。


---

# PART 1: 高层级 Discord 摘要

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**LLM 微调，化繁为简**：工程师们分享了针对 **Out of Memory (OOM) 错误**的解决方案，并讨论了微调流程。大家在将优化器状态卸载到 CPU 或使用 CUDA managed memory 的益处上达成共识，并提到了 **bnb 8bit casting** 和 **Low-Rank Adapters (LoRA)** 等技术，以在训练期间节省显存并提升性能。有价值的资源包括关于 [8-bit Deep Learning 的 YouTube 视频](https://www.youtube.com/watch?v=jyOqtw4ry2w&themeRefresh=1) 以及基准测试工具 [VRAM Calculator](https://vram.asmirnov.xyz/)。

**对额度困惑的共鸣**：多位社区成员表示在获取承诺的额度（credits）时遇到困难。从 Modal、OpenAI 到 Replicate，多个平台都出现了额度缺失的情况，成员们在各自频道发布了寻求解决的请求。为了加快支持进度，成员们主动提供了用户 ID 和组织 ID 等信息。

**模型训练的困扰与成就**：成员们在不同平台上排查微调和推理的挑战，重点关注 **dataset preparation**（数据集准备）、使用 TRL 或 Axolotl 等现有框架，以及在有限硬件上处理大型模型训练等实际问题。另一方面，也有成员分享了在 Modal 上部署 **Mistral** 的正面体验，并对其热重载（hot-reload）功能表示赞赏。

**深入现实世界的 ML 讨论**：对话深入探讨了实际的 Machine Learning (ML) 应用，例如 **dynamically swapping LoRAs**（动态切换 LoRA）以及用于音频处理的 Google Gemini API。还研究了 **Llama-3 8B** 等模型使用 *Chain of Thought*（思维链）推理进行诊断的情况，同时也承认了模型结论中存在的缺陷。

**快速工程的资源积累**：社区一直积极分享资源，包括 Jeremy Howard 在 YouTube 上的 [《A Hackers' Guide to Language Models》](https://www.youtube.com/watch?v=jkrNMKz9pWU) 以及用于制作图表的 [Excalidraw](https://excalidraw.com/)。推荐使用 [Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/training_overview.html) 等工具来微调 Transformer，彰显了不断提升技艺的协作精神。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **等待“圣杯” SD3**：大家对 **Stable Diffusion 3 (SD3)** 的发布充满期待，希望其在体素艺术（voxel art）方面有功能改进，而一位用户幽默地表示担心在发布前不得不去睡觉。
- **缓慢的网络连接考验耐心**：由于达到数据上限，一位成员经历了长达 12 小时的漫长过程来下载 **Lora Maker**，忍受着来自 Pytorch.org 低至 "50kb/s" 的下载速度。
- **模型平台大比拼**：关于 AI 模型和 Checkpoints 可用性的讨论兴起，**Huggingface** 和 **Civitai** 成为关注焦点；Civitai 凭借丰富的 LoRA 和 Checkpoints 选择占据领先地位。
- **辩论：放大技术**：一场关于将 **SD1.5** 图像放大到 1024 是否能与直接在 1024x1024 分辨率下训练的 **SDXL** 效果相媲美的技术辩论被引发，随后有人建议测试 SDXL 放大到更高分辨率的能力。
- **AMD，为什么你不能运行 SD？**：一位努力在 **AMD GPU** 上运行 **Stable Diffusion** 的成员表达了挫败感，最终社区建议其重新查阅安装指南并寻求进一步的技术支持。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**2024年7月：Unsloth AI 备受期待的 MultiGPU 支持**  
Unsloth AI 的 MultiGPU 支持预计将于 2024 年 7 月初发布，以企业级 Unsloth Pro 为首；这将有望实现更高效的微调和模型训练。

**Llama 3 尝试多样化微调**  
用户探索了 Llama 模型的各种 Tokenizer 选项，讨论确认了来自 llama.cpp 和 Hugging Face 等服务的 Tokenizer 是互通的，并为寻求精确指令的用户参考了 [YouTube 上的微调指南](https://www.youtube.com/watch?v=3eq84KrdTWY&t=665s)。

**Hugging Face AutoTrain 扩展支持 Unsloth**  
[Hugging Face AutoTrain 现在包含 Unsloth 支持](https://x.com/abhi1thakur/status/1800511251145015393)，为更高效的大语言模型 (LLM) 微调铺平了道路，AI 社区对这些节省时间并减少内存占用的进展感到兴奋。

**AI 创新展示：心理治疗 AI 与 MARS5 TTS**  
新兴工具如[使用 Unsloth 在 Llama 3 8b 上微调的心理治疗 AI](https://xtherapy.streamlit.app/)，以及新开源的 [CAMB AI 的 MARS5 TTS 模型](https://github.com/camb-ai/mars5-tts)（该模型承诺在语音克隆中实现更高的真实感），正在社区中引起轰动。

**苹果招聘：AI 集成引发辩论**  
苹果最新的个性化 AI 计划被称为 "Apple Intelligence"，成为激烈讨论的话题。根据 WWDC 的报道，社区正在权衡其在语言支持和集成更大型模型方面的潜力。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**深度学习对效率的追求**：成员们辩论了 **Llama 3** 进行 **4-bit quantization** 的益处与障碍，并建议 **Tensor Parallelism** 尽管具有实验性质，但提供了可能的路径。讨论强调了包括 **IQ4_xs** 和 **HQQ** 在内的各种量化方法的适用性，并参考了一篇展示它们在 Apple Silicon 上性能的博客 [LLMs for your iPhone](https://stephenpanaro.com/blog/llm-quantization-for-iphone)。

**寻求更智能的 Transformer**：关于改进 **Transformer** 模型的讨论浮出水面，参考了如 "How Far Can Transformers Reason?" 等论文中强调的学习能力挑战，该论文提倡使用 *supervised scratchpads*。此外，还出现了关于模型中 *influence functions* 有用性的辩论，引用了如 [Koh and Liang 的 influence functions 论文](https://arxiv.org/pdf/1703.04730)等开创性著作。

**攻克文本转语音 (TTS) 合成**：*VALL-E 2* 因其卓越的 *zero-shot TTS* 能力被提及，尽管研究人员在访问[项目页面](https://web.archive.org/web/20240529183033/https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-2/)时遇到了问题。与此同时，**LlamaGen** 在视觉 Tokenization 方面的进展有望增强自回归模型，并引发了关于整合来自 "Stay on topic with Classifier-Free Guidance" 等相关工作方法的讨论。

**解释多模态转换**：讨论了 **DeepSeek-LLM-7B** 模型的集成挑战，其基于 **LLaMA** 的结构是焦点。分享的资源包括一个 [GitHub 仓库](https://github.com/Butanium/llm-latent-language/blob/1054015066a4fa20386765d72601d03aa7ef5887/utils.py#L334)，以协助社区进行解释性工作并克服模型集成复杂性。

**LLM 交互的优化策略**：Eleuther 通过 **--apply_chat_template** 标志引入了聊天模板功能，展示了增强用户与语言模型交互的持续工作。社区还在推动优化本地和 **OpenAI Batch API 应用**的 Batch API 实现，讨论了高层实现步骤，并计划在未来开发一个用于在 Batch 结果上重新运行指标的实用工具。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU 引发的热水浴缸幻想**：一个关于将 **GPU 废热** 重新利用来加热热水浴缸的有趣提议，引发了关于利用数据中心热输出的更广泛讨论。这个玩笑揭示了将废热转化为社区供暖解决方案的潜力，同时提供了可持续的数据中心运营模式。
  
- **穿越 Triton 丛林**：提供了关于提升 **Triton** 性能的导航建议，常见的困扰包括速度逊于 **cuDNN** 以及在 kernel 内部打印变量的复杂性。用户表达了对更简单语法的偏好，建议避免使用元组以减少开发困惑。

- **跨越从 Torch 到 C++ 的频谱**：展示了技术实力，参与者讨论了使用 `torch.compile` 进行全图编译的优点，而其他人则考虑为 PyTorch 编写 **HIP kernels**，两者都预示着即将到来的优化浪潮。这些对话的交汇点还思考了 **C++20** 的 concepts 是否能在不退回 C++17 的情况下解开代码的复杂性。

- **Bitnet 的 0 与 1 成为焦点**：围绕训练 **1-bit Large Language Models (LLMs)** 展开了深入交流，分享了来自 [Microsoft 的 Unilm GitHub](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf) 的资源，概述了其效率潜力，同时也承认了与 **FP16** 相比存在的稳定性问题。

- **LLM 的高度与压缩技术**：从分析 **ThunderKitten** 乏善可陈的 **TFLOPS** 表现，到探索使用 [Sconce](https://github.com/satabios/sconce) 的模型压缩策略，社区汇集智慧来应对这些复杂的领域。此外，[PyTorch 的 AO 仓库](https://github.com/pytorch/ao/pull/276) 中增加了一个基准测试 Pull Request，承诺为 **Llama** 模型提供准确的性能评估。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 中的并发难题与 GPU 困境**：工程师们在担心 Mojo 异步能力的同时，讨论了在其库中采用结构化并发的问题，并强调了 TPU 支持等异构硬件的重要性。一种强烈的观点认为，Mojo 在执行速度上取得了成功，但与 TPU 等高性价比方案相比，在硬件加速方面仍有不足。

- **Mojo 上的快速 RNG 与数学精通**：移植 xoshiro PRNG 的工作在笔记本电脑和使用 SIMD 时都带来了显著的速度提升，同时正努力通过 NuMojo 项目为 Mojo 带来等同于 numpy 的功能。趋势显示社区正在推动扩展 Mojo 的数值计算能力和效率。

- **解决 Mojo 的内存狂热**：关于 Mojo 内存管理实践引发了争议，讨论集中在对上下文管理器的需求、对 RAII 的依赖以及 UnsafePointers 的复杂性。这场辩论强调了社区致力于完善 Mojo 的所有权（ownership）和生命周期（lifetimes）范式的决心。

- **攻克 TPU 领地**：MAX 引擎与 TPU 的潜在兼容性成为亮点，社区成员探索了 OpenXLA 等资源以获取机器学习编译器的指导。前瞻性讨论涉及了 MAX 引擎路线图的更新，包括不可避免的 Nvidia GPU 支持。

- **Mojo Nightly 更新日志的细微差别**：新发布的 Nightly Mojo 编译器版本 `2024.6.1105` 带来了多项变化，包括移除 `SliceNew` 和 `SIMD.splat`，以及引入 `NamedTemporaryFile`。社区内这种持续集成的文化体现了对迭代和快速开发周期的偏好。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **苹果涉足个人 AI 领域**：苹果宣布了 "Apple Intelligence"，将 ChatGPT-4o 集成到 Siri 和写作工具中，以增强系统级的用户体验，同时优先考虑隐私，正如一篇 [Reddit 帖子](https://www.reddit.com/r/ChatGPT/s/KrhcqUpEuq)中所分享的那样。

- **iOS 18 和 WWDC 2024 拥抱 AI**：随着 WWDC 2024 设定的愿景，iOS 18 中由机器学习驱动的新照片应用可以更智能地对媒体进行分类，并伴随着整个苹果生态系统的重大 AI 集成和软件进步。

- **Rabbit R1 是一个失败的设备吗？**：成员们就 Rabbit R1 设备的合法性交换了意见，提到了其可疑的加密货币联系，并推测了其在 Android OS 上的功能，正如一段 Coffeezilla 视频中所讨论的那样。

- **Perplexity AI - 承诺与怀疑并存**：围绕 Perplexity 的 Pages 和 Pro 功能在桌面/网页端的限制存在困惑；同时，Perplexity 的学术来源准确性面临审查，用户指出 Google 的 NotebookLM 可能更胜一筹。

- **集成难题与隐藏的密钥**：在将 Perplexity AI 引入自定义 GPT 应用时遇到了障碍，引发了关于模型名称更新以及在 API 密钥误暴露后的安全 API 实践的讨论，相关内容记录在 [Perplexity API 指南](https://docs.perplexity.ai/discuss/65edc94038fa40001045873c)中。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **PDF 解析的探索**：工程师们正在探索用于解析 PDF 中结构化表单的**本地工具**，并建议使用 **Langchain** 结合本地 LLM 来高效提取字段。

- **WebUI 的困扰与变通方法**：由于 **LMStudio** 官方缺乏 **WebUI** 支持，导致用户使用 **llama.cpp server** 和 *[text-generation-webui](https://github.com/oobabooga/text-generation-webui/)* 从远程 PC 与该工具进行交互。

- **加州 AI 法案引发争议**：**SB 1047** 法案引发了关于其对开源 AI 影响的激烈辩论，人们担心这可能会使 AI 开发集中在少数大公司手中，并无限期地增加模型创建者的法律责任。[Dan Jeffries 的推文](https://x.com/dan_jeffries1/status/1794740447052525609?s=46)为这一讨论提供了见解。

- **GPU 升级与 ROCm 见解**：工程师们讨论了升级到具有更高 VRAM 的 GPU 以运行大型 AI 模型，并推荐 **AMD 的 ROCm** 作为计算任务中比 **OpenCL** 更快的替代方案。由于对 LMStudio 中**多 GPU 性能**的担忧，一些人转向了 *stable.cpp* 和 Zluda 等替代方案，以便在 AMD 上兼容 **CUDA**。

- **模型合并大师**：社区在模型合并方面非常活跃（例如 **Boptruth-NeuralMonarch-7B**），评估新配置（如 **Llama3-FiditeNemini-70B**），并处理 **AutogenStudio** 中的 token 限制 bug 等运行问题，相关修复已在 [GitHub](https://github.com/microsoft/autogen/issues/2050) 上跟踪。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Apple 在 AI 领域引起轰动**：Apple 宣布将 AI 集成到其生态系统中，社区对其对竞争和设备性能的影响议论纷纷。WWDC 2024 的参与者热烈讨论了 "Apple Intelligence"，这是一个深度集成到 Apple 设备中的系统，并正在研究现有的 [Apple Foundation Models 概览](https://machinelearning.apple.com/research/introducing-apple-foundation-models)。

- **关于 AI 与隐私的担忧与辩论**：随着 AI 的进步，隐私担忧也随之增加。用户对潜在的数据滥用表示担忧，主张使用更安全的设备端 AI 功能，而不是仅仅依赖云计算。这种讨论反映了技术爱好者对云端和本地解决方案都持怀疑态度的二分法。

- **GPT-4：高期望遭遇实际小问题**：OpenAI 关于即将推出的 ChatGPT 更新的承诺引起了兴奋，但用户报告了应用冻结以及对新语音模式延迟发布的困惑。此外，开发者对 GPT Store 中明显的政策违规行为感到沮丧，这阻碍了他们发布或编辑 GPTs 的能力。

- **跨时区的时间管理**：AI 工程师正在制定策略，以应对 Completions API 的时区挑战，权衡使用外部库进行时间戳转换或使用合成数据来降低风险并提高精度的方案。共识倾向于将 UTC 作为一致模型输出的基准，并在输出后进行针对特定用户的时区调整。

- **认识 Hana AI：你的 Google Chat 新队友**：Hana AI 被介绍为一款用于 Google Chat 的 AI 机器人，旨在通过处理各种生产力任务来提高团队效率，目前可免费使用。工程师可以试用并对该机器人提供反馈，它承诺为经理和高管提供帮助，可通过 [Hana AI 网站](https://hana.hanabitech.com)访问。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **寻求最佳医疗诊断 AI 的进程停滞**：在讨论中，对于用于医疗诊断的最佳大语言模型（LLM）尚未达成共识。
  
- **CVPR 论文获取的语义化飞跃**：一个索引了 **CVPR 2024 论文摘要**并具备**语义搜索**功能的新应用已被分享，可以点击[这里](https://huggingface.co/spaces/pedrogengo/CVPR2024_search_papers)访问。

- **Civitai 文件的技术故障**：一名成员在将 `diffusers.StableDiffusionPipeline.from_single_file()` 与来自 Civitai 的 safetensors 文件配合使用时，遇到了 `TypeError: argument of type 'NoneCycle' is not iterable` 错误。

- **AI 立法对开源领域产生重大影响**：一条推文线程批评了加州 AI 控制法案（California AI Control Bill），认为其可能会阻碍开源 AI 的发展，并对模型创建者的严格责任制提出了警示。

- **动漫遇上 Wuerstchen3 扩散模型**：一位用户发布了经过动漫微调的 SoteDiffusion Wuerstchen3 版本，并提供了指向 [Fal.AI 文档](https://fal.ai/models/fal-ai/stable-cascade/sote-diffusion)的有用链接，以获取 API 实现细节。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Character Codex 发布**：Nous Research 发布了 **[Character Codex 数据集](https://huggingface.co/datasets/NousResearch/CharacterCodex)**，包含来自动漫、历史档案和流行偶像等不同来源的 15,939 个角色数据，现已开放下载。

**技术讨论热火朝天**：引人入胜的对话包括 LLM 中 **RLHF** 可能对创造力造成的抑制，并与 Anthropic 等公司的成功进行了对比。辩论还涵盖了**模型量化和剪枝方法**，其中一种针对 [LLaMA 3 10b 的策略](https://github.com/horseee/LLM-Pruner)旨在巧妙地缩减模型尺寸。

**知识同步**：成员们讨论了 CoHere 用于多步输出构建的 **Chain of Thought** (CoT) 检索技术，并提出了一种混合检索方法，可能将 **elastic search** 与 **bm25 + embedding** 以及网络搜索结合起来。

**代码遭遇立法**：有一篇针对 **CA SB 1047** 的出色评论，认为它对开源 AI 构成了风险，同时一位成员分享了 **[Dan Jeffries 对此事的见解](https://x.com/dan_jeffries1/status/1794740447052525609?s=46)**。此外还提到了旨在保护 AI 创新的反向提案 **SB 1048**。

**新的 Rust 库 Rig 登场**：用于创建 LLM 驱动应用的开源 Rust 库 **'Rig'** 的发布引起了关注；其 [GitHub 仓库](https://github.com/0xPlaygrounds/rig)是 AI 开发者获取示例和工具的宝库。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Apple 的 AI 变革者**：Apple 在 WWDC 2024 上发布了 'Apple Intelligence'，将 ChatGPT 集成到 Siri 中，以提升 iPhone, iPad 和 Mac 的用户界面体验，这引发了关于安全性的担忧和讨论。公告详情已在[这篇文章](https://asknews.app/en/stories/Apples-AI-Leap-Sparks-Controversy-Amid-Musks-Security-Concerns)中分享。

- **求职现状反思**：一位向往 Cohere 团队的成员分享了尽管在黑客松中取得显著成绩并拥有 ML 经验，却依然遭遇求职被拒的挫折，这引发了关于个人推荐是否优于资历的讨论。

- **Cohere 的开发者对话**：Cohere 推出了 Developer Office Hours，这是一个供开发者解决疑虑并直接与 Cohere 团队互动的论坛。官方发布了[下一场活动的提醒](https://discord.gg/7zjrJmKtBB?event=1248300806600392766)并邀请大家参与。

- **反馈热烈**：成员们对 Cohere 提供的这种新的 Developer Office Hours 形式表示高度满意，称赞团队营造了一个极具参与感且轻松的环境。

- **与专家交流**：Cohere 鼓励成员积极参与，并为开发者提供了通过 Developer Office Hours 扩展知识并与团队一起排查问题的机会。下一场活动定于 6 月 11 日下午 1:00 (ET)，可通过此 [Discord Event](https://discord.gg/7zjrJmKtBB?event=1248300806600392766) 参加。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Apple 全力投入 AI 集成**：Apple 宣布在其整个 OS 中集成 AI，重点关注多模态 I/O 和用户体验，同时保持隐私标准。针对 AI 任务，他们推出了 "Private Cloud Compute"，这是一个安全的系统，可以在不损害用户隐私的情况下将计算卸载到云端。

- **ChatGPT 找到新归宿**：Apple 与 OpenAI 宣布达成合作伙伴关系，将 ChatGPT 引入 iOS, iPadOS 和 macOS，标志着向 AI 驱动的操作系统的重大迈进。这将使 Apple 用户在今年晚些时候能直接使用对话式 AI。

- **Mistral 掀起融资浪潮**：AI 初创公司 Mistral 获得了 6 亿欧元的 Series B 融资用于全球扩张，这证明了投资者对人工智能未来的信心。此轮融资紧随 AI 领域投资激增的趋势，凸显了市场的增长潜力。

- **PostgreSQL 的 AI 性能超越 Pinecone**：PostgreSQL 的新开源扩展 "pgvectorscale" 因在 AI 应用中表现优于 Pinecone 而受到赞誉，承诺提供更好的性能和成本效率。这标志着支持 AI 工作负载的数据库技术取得了重大进展。

- **LLM 在现实世界中的应用**：Mike Conover 和 Vagabond Jack 做客 Latent Space 播客，分享了他们在生产环境中部署 Large Language Models (LLM) 的经验以及金融领域的 AI Engineering 策略。讨论集中在行业背景下有效利用 LLM 的实际考量和策略。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **进阶知识图谱专题**：一场聚焦于“进阶知识图谱 RAG”的特别研讨会已与来自 Neo4j 的 Tomaz Bratanic 共同安排，旨在探索 LlamaIndex 属性图（property graph）抽象。鼓励工程师们[注册参加](https://lu.ma/kqxmbuou)将于太平洋时间周四上午 9 点举行的活动。

- **巴黎 AI 聚会**：@hexapode 将在 [Paris Local & Open-Source AI Developer meetup](https://t.co/5GLV08cGFa) 上进行现场演示，该活动将于 6 月 20 日下午 6:00 在巴黎 Station F 举行，届时将有包括 Koyeb、Giskard、Red Hat 和 Docker 在内的多家知名公司参加，其他人也可以通过[此处](https://forms.gle/YMXvYCVhuuppTWTp7)申请演示自己的作品。

- **LlamaIndex 的问题与变通方案**：用户正在寻求集成各种查询引擎和 LLM 流水线的帮助，例如使用 LlamaIndex 结合 SQL、Vector Search 和 Image Search，以及在查询向量数据库时使用 OpenAI Chat Completion 作为备选方案。对于涉及使用 Llama 3 进行 SQL 数据库检索和分析的项目，建议探索 text-to-SQL 流水线并参考 [LlamaIndex 进阶指南](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/?h=text2)。

- **伯克利头脑风暴**：加州大学伯克利分校（UC Berkeley）的一个研究团队正在探索自定义 RAG 系统的领域，寻求资深工程师的意见，以应对构建、部署和维护此类系统的复杂性。

- **稀疏向量生成的提速需求**：在使用 Qdrant 和 LlamaIndex 的混合模式下，生成和上传稀疏向量（sparse vectors）对某些用户来说太慢了，有建议提示利用本地 GPU 或使用 API 来加速这一过程。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LAION 陷入争议**：LAION 数据集在巴西电视节目中受到批评；问题源于人权观察组织（Human Rights Watch）的一项指控，称 AI 工具正在滥用儿童的在线个人照片，详情见[此处](https://www.hrw.org/news/2024/06/10/brazil-childrens-personal-photos-misused-power-ai-tools)。

- **隐私与互联网素养辩论**：工程师们对互联网数据隐私的普遍误解表示担忧，触及了数十亿用户缺乏相关知识所导致的严重问题。

- **LlamaGen 推动图像生成发展**：发布的 LlamaGen 模型展示了图像生成领域的重大进展，利用语言模型技术进行视觉内容创作，详见其[研究论文](https://arxiv.org/abs/2406.06525)。

- **CAMB AI 的 MARS5 宣布开源**：由 CAMB AI 开发的 TTS 模型 MARS5 已向社区开源，Reddit 上的一个帖子邀请大家提供反馈，更多技术讨论可在[此线程](https://www.reddit.com/r/CAMB_AI/comments/1day7ta/introducing_mars5_opensource_insanely_prosodic/)中找到。

- **视觉数据集的安全性**：LlavaGuard 项目（详见[此处](https://ml-research.github.io/human-centered-genai/projects/llavaguard/index.html)）提出了一个模型，旨在提高视觉数据集标注的安全性和伦理合规性。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Apple Intelligence 评价褒贬不一**：工程师们对 OpenAI 与 Apple 的合作反馈不一，认为集成到 **Apple Intelligence** 中的程度可能较浅；然而，尽管存在传闻和质疑，官方公告中仍强调了用户隐私（[阅读更多](https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/)）。Apple 设备端和服务器端模型的对比基准测试引发了人们对其相对于同类产品性能的好奇。

- **建立清晰区分**：Apple 将 **Apple Intelligence** 与 **Siri** 分开的战略方法引发了关于其对用户采用率以及对新系统功能认知潜在影响的对话。

- **科技界期待重磅访谈**：**Dwarkesh Patel** 即将对 **François Chollet** 进行的采访让工程师们对 AGI 时间线辩论可能出现的转向充满期待，强调了基于 Chollet 对智力衡量研究进行深度提问的重要性。

- **TRL 实现引发争议**：有人对实现 TRL 提出了警告，称该技术“未经证实”。一位成员计划为 TRL 提交 Pull Request (PR)，得到了另一位社区成员的积极鼓励和 Review 提议。

- **社区贡献中的支持**：协作精神显而易见，一名成员计划为 **TRL** 做出贡献并收到了 Review 承诺，展示了该组织相互支持和知识共享的文化。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Apple Intelligence 备受关注**：社区对 Open Interpreter 与 Apple 在 [Apple Intelligence 页面](https://www.apple.com/apple-intelligence/)上概述的以隐私为中心的 AI 功能进行潜在集成的可能性表现出浓厚兴趣。这可能会促使利用开发者 API 来增强 Apple 设备上的 AI 功能。

- **SB 1047 成为众矢之的**：[Dan Jeffries 批评了](https://x.com/dan_jeffries1/status/1794740447052525609?s=46)由 Dan Hendyrcks 提出的加州 AI 控制与中心化法案 (SB 1047)，理由是该法案对 AI 实施中心化控制，并对开源 AI 创新构成威胁。

- **Mac M1 上的 Arduino IDE 问题已解决**：通过在 [GitHub pull request](https://github.com/lacamera/ESPAsyncWebServer/pull/2/files) 中找到的修复方案，解决了 Mac M1 芯片上 Arduino IDE 的一个问题，但该修复导致设备重启时 Wi-Fi 设置出现了额外的问题。

- **Linux 成为 Open Interpreter 的避风港**：成员间的讨论强调了在未来的 Open Interpreter 开发中优先考虑 Linux 的想法，旨在提供独立于 Apple 和 Microsoft 等主流操作系统的 AI 辅助工具。

- **具有记忆功能的个人助手**：分享了增强 Open Interpreter 的工作，通过一个熟练的 Prompting 系统，使其能够像个人助手一样存储、搜索和检索信息，突出了在为 AI 系统创建记忆保留方面的创新。

- **Killian 的见解被记录**：在 Killian 最近的演讲之后进行了一场值得关注的讨论，该演讲对于在社区成员中聚焦相关的 AI 话题起到了重要作用。录音可以在[此处查看](https://discord.com/channels/1146610656779440188/1147665339266650133/1248858812761509938)。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 的标签功能（Tagging）出现问题**：工程师们注意到 LangChain 中的 `create_tagging_chain()` 函数会忽略 Prompt，由于目前尚未提供解决方案，这引起了开发者的不满。
- **征集 RAG 开发见解的协作邀请**：加州大学伯克利分校（UC Berkeley）团队成员正积极寻求与在 **Retrieval-Augmented Generation (RAG)** 系统方面有经验的工程师进行交流，以分享在开发和部署过程中面临的挑战。
- **LangGraph 与 LangChain 的对比**：社区表现出对理解使用 **LangGraph** 优于经典 **LangChain** 配置的优势的兴趣，特别是关于在 LangGraph 中执行受控脚本方面。
- **期待 ONNX 与 LangChain 的结合**：有人对 ONNX 和 LangChain 之间的潜在兼容性感到好奇；然而，对话并未深入展开。
- **通过 OpenAI 简化大型数据集处理**：分享了一份使用 OpenAI API 处理大型数据集的综合指南，重点介绍了设置环境变量、数据匿名化以及使用 Elasticsearch 和 Milvus 进行高效数据检索的最佳实践。文中提供了相关文档和 GitHub issue 链接供参考。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **新手遭遇权限难题**：一位渴望参与 **tinygrad** 开发的新成员发现自己被 bounties 频道锁定权限，无法参与 **AMX support bounty** 的工作。George Hotz 解决了这一困惑，表示必须“成为紫色（Become a purple）”才能获得必要的贡献权限。
  
- **George 扮演“守门人”角色**：在回答有关 **tinygrad** 中 **AMX support** 的问题时，George Hotz 暗示在处理此类任务之前，需要更深入地了解社区文档，并提到需要阅读一份特定的问题文档。

- **一个经典的误会**：当一名新成员引用了错误的指南——*["How To Ask Questions The Smart Way"](http://www.catb.org/~esr/faqs/smart-questions.html)* 时，发生了一次文档引用失误，导致与 George Hotz 之间出现了一个幽默的“先有鸡还是先有蛋的问题”时刻。

- **回到起点**：在一番反复沟通后，这位新贡献者决定退后一步，在带着更精确的问题回来之前，先深入研究 **tinygrad** 的代码库，这展示了为此类项目做贡献所需的复杂性和专注度。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 的高速服务**：OpenRouter 通过利用 **Vercel Edge** 和 **Cloudflare Edge** 网络解决了延迟问题，确保服务器节点在地理位置上靠近用户，从而实现更快的响应速度。
- **计划中的供应商偏好设置**：虽然 OpenRouter playground 目前缺少用户选择首选 API 供应商的功能，但已确认计划实现该功能。
- **技术极客的 API 供应商选择**：用户可以通过使用 API 来绕过 OpenRouter playground 中缺乏直接供应商选择的问题；该变通方法的指南可在 [OpenRouter 文档](https://openrouter.ai/docs/provider-routing)中找到。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **ShareGPT 的训练面纱**：在训练时，**ShareGPT** 不会“看到”其自身转换后的 prompt 格式，从而确保训练过程的纯净。
- **Apple AI 展示实力**：**Apple 新的端侧和服务器模型**的[基准测试结果已出炉](https://x.com/ldjconfirmed/status/1800355063120151031)，展示了它们在指令遵循和写作方面的实力，并与其他领先模型进行了对比。
- **乐天 (Rakuten) 模型席卷而来**：**Rakuten AI 团队**发布了一系列在日语方面表现出色的 LLM，这些模型基于 **[Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/)** 并提供商业许可，在社区成员中引发了乐观的反响。
- **JSON 响应带来的喜悦**：工程师们在交流中愉快地赞赏了模型以 JSON 格式响应的能力，表达了对模型这一技术能力的认可与喜爱。
- **使用 Axolotl 让微调更简单**：AI 从业者可以参考在 **RunPod** 上进行[微调的新教程](https://docs.runpod.io/tutorials/pods/fine-tune-llm-axolotl)，该教程概述了微调 LLM 的简化流程，并提供了涵盖各种模型系列的 YAML 示例。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **编码风暴前的宁静**：Vincent Warmerdam 推荐使用 [calmcode.io](https://calmcode.io) 进行模型训练，用户们也认可该网站在模型训练策略和技术方面的实用内容。
  
- **RAG 的正确姿势**：一篇 [Stack Overflow 博客文章](https://stackoverflow.blog/2024/06/06/breaking-up-is-hard-to-do-chunking-in-rag-applications/) 详细介绍了 RAG (retrieval-augmented generation) 实现中的分块 (chunking) 策略，强调了 text embeddings 在将源文本准确映射到 LLM 语义结构中的作用，从而增强了对源数据的可靠性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **关于 TRL 中 DPO 的 KL 图表说明**：Dominant Policy Optimization (DPO) 的实现中没有直接绘制 Kullback–Leibler (KL) 散度图表，但在 **Trust Region Learning (TRL)** 的 Proximal Policy Optimization (PPO) 训练器中确实存在此类 KL 图表。正如 [TRL 的 GitHub 仓库](https://github.com/huggingface/trl/blob/34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb/trl/trainer/ppo_trainer.py#L1133)中指出的，KL 图表可以在 PPO 训练器的代码中找到。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

**AI 社区齐聚 Mosaic 活动**：在 [Databricks Summit 的 Mosaic 活动](https://mosaicx.events/events/june-10-2024-san-francisco-ca?events=main)上与 **Chip Huyen** 面对面交流，与 AI 和 ML 专家建立联系。该聚会定于 **2024 年 6 月 10 日**在旧金山举行。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

鉴于提供的片段中缺乏实质性的讨论点且上下文不足，目前没有重大的技术或详细讨论可供工程师读者总结。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：各频道详细摘要与链接

{% if medium == 'web' %}

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1249830188628578495)** (37 条消息🔥): 

- **微调中模型大小的启发式方法**：一名成员提出了一个关于如何根据任务复杂度选择微调模型大小的通用问题，并提到了在快速原型设计中进行广泛评估的困难。他们询问资深用户是否会随着时间的推移培养出对模型能力的直觉。
  
- **Karpathy 极具影响力的视频**：讨论强调了 Andrej Karpathy 视频的教育价值，其中一名成员分享了一个[完整实现仓库](https://github.com/gao-hongnan/omniverse/tree/main/omnivault/transformer)以及基于 Karpathy 早期教程的 GPT 补充笔记。

- **NCCL 超时问题**：一位用户分享了一个显示 NCCL 工作超时的错误日志，寻求社区建议。该日志突显了 ProcessGroupNCCL 操作中的复杂问题。

- **Gorilla 项目在工具使用和 API 生成方面表现出色**：**Gorilla** 项目被提及为通过微调模型来改进工具使用和 API 生成的一个有趣案例。他们强调了该项目的资源，包括 [GoEx 运行时](https://goex.gorilla-llm.com/index)和[排行榜](https://gorilla.cs.berkeley.edu/leaderboard.html)，并分享了一个概述该项目的 [YouTube 视频](https://www.youtube.com/live/WAvO8FTDJ8M?si=dR_9-Q5hLxMPvRCS)。

- **LLM 与传统 ML/DL 的对比**：关于从传统 ML/DL 向 LLM 转型的讨论指出，在从零开始之前利用现有模型的重要性。来自 ML/DL 的核心原则（如数据准备、EDA 和模型流水线）在 LLM 生命周期中仍然非常重要。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gist.github.com/sroecker/feaa61ea69182cb7ae1c9328b755786a">使用 Moondream 为 datikz 图表生成字幕的脚本</a>：使用 Moondream 为 datikz 图表生成字幕的脚本。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/writer/writer-framework">GitHub - writer/writer-framework: 前端无代码，后端 Python。一个用于创建数据应用的开源框架。</a>：前端无代码，后端 Python。一个用于创建数据应用的开源框架。 - writer/writer-framework</li><li><a href="https://brain.nehiljain.com/posts/how-i-improved-my-prompting-for-budget-categorization/">我如何改进预算分类的 Prompt 编写</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/sroecker/datikz-v2-moondream-caption-test2/viewer?row=18">sroecker/datikz-v2-moondream-caption-test2 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/sroecker/datikz-v2-moondream-caption-test">sroecker/datikz-v2-moondream-caption-test · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://Gorilla.cs.berkeley.edu">Gorilla</a>：未找到描述</li><li><a href="https://www.youtube.com/live/WAvO8FTDJ8M?si=dR_9-Q5hLxMPvRCS">大规模教学 LLM 使用工具 - Shishir Patil | Stanford MLSys #98</a>：Stanford MLSys 研讨会系列第 98 集！大规模教学 LLM 使用工具。演讲者：Shishir Patil。简介：Shishir G. Patil 是加州大学伯克利分校的计算机科学博士生...</li><li><a href="https://www.gaohongnan.com/influential/generative_pretrained_transformer/03_concept.html#autoregressive-self-supervised-learning-paradigm)">Generative Pre-trained Transformers (GPT) 的概念 &#8212; Omniverse</a>：未找到描述</li><li><a href="https://archive.ph/v8lN0">Slop 是 AI 对垃圾邮件的回应吗？一个针对糟糕搜索结果的短语出现了。 - The…</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1249829935250673784)** (44 messages🔥): 

- **关于奖励额度（Bonus Credits）的说明**：成员们询问了使用 Modal 的奖励额度发放情况。据澄清，第二批额度将在周二 UTC 时间午夜过后不久发放。

- **“简单可扩展的 Serverless 服务”幻灯片分享**：一位用户分享了 Charles 关于“Mastering LLMs - Simple Scalable Serverless Services”演讲的 [Google Slides 链接](https://docs.google.com/presentation/d/14uDnzd06j9i0zAQ3lTmB7QHBSO45BIsVGUZBZ3HKxGo/edit#slide=id.g2c7588f453b_0_272)。

- **GitHub 项目与仓库**：分享了多个 GitHub 仓库，包括 [charlesfrye/minimodal](https://github.com/charlesfrye/minimodal) 和 [awesome-modal](https://github.com/modal-labs/awesome-modal)，并附带了额外的贡献指南和项目链接。

- **关于成本管理的讨论**：用户讨论了防止 AWS S3 或 Vercel 等 Serverless 服务成本激增的最佳实践。建议包括设置较高的预估成本上限和负载均衡器限制，以防止意外支出。

- **Modal 中的 Notebooks 功能**：在询问 Modal 内部 Notebooks 功能的开发进展时，Charles 确认该功能可以使用但存在一些限制，并建议用户在遇到问题时提交 issue。分享了一个 [mistral-finetune-modal](https://github.com/andresckamilo/mistral-finetune-modal/blob/main/src/main.py) 的示例项目链接以说明其用法。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modal-labs/awesome-modal/pull/1">Add indic-subtitler project by kurianbenoy · Pull Request #1 · modal-labs/awesome-modal</a>：IndicSubtitler GitHub 项目网站</li><li><a href="https://github.com/andresckamilo/mistral-finetune-modal/blob/main/src/main.py">mistral-finetune-modal/src/main.py at main · andresckamilo/mistral-finetune-modal</a>：通过在 GitHub 上创建账号来为 andresckamilo/mistral-finetune-modal 的开发做出贡献。</li><li><a href="https://modal.com/docs/guide/notebooks">Jupyter notebooks</a>：你可以在 Jupyter 等 notebook 环境中使用 Modal 客户端库！只需导入 modal 并正常使用即可。不过，在 notebook 中使用 Modal 时存在一些限制。</li><li><a href="https://github.com/modal-labs/awesome-modal">GitHub - modal-labs/awesome-modal: A curated list of amazingly awesome Modal applications, demos, and shiny things. Inspired by awesome-php.</a>：一份精选的、令人惊叹的 Modal 应用、演示和闪光点列表。灵感来自 awesome-php。</li><li><a href="https://github.com/charlesfrye/minimodal">GitHub - charlesfrye/minimodal: A miniature version of Modal</a>：Modal 的微型版本。通过在 GitHub 上创建账号来为 charlesfrye/minimodal 的开发做出贡献。</li><li><a href="https://docs.google.com/presentation/d/14uDnzd06j9i0zAQ3lTmB7QHBSO45BIsVGUZBZ3HKxGo/edit#slide=id.g2c7588f453b_0_272">Mastering LLMs - Simple Scalable Serverless Services</a>：简单可扩展的 Serverless 服务 bit.ly/mastering-llms-ssss
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1250111760795439156)** (4 messages): 

- **使用 Hugging Face 额度**：一位成员询问了除了推理端点（inference endpoints）之外，有效使用 Hugging Face 额度的方法。其他成员建议将额度用于 **Spaces with GPUs** 和 **AutoTrain**。AutoTrain 支持通过简单上传数据来实现自定义机器学习模型的**自动训练**和快速部署，涵盖 LLM finetuning、图像分类和文本分类等任务。
- **新表单咨询**：另一位成员询问新表单是否已经可用，但目前没有相关回复的记录。

**提及的链接**：<a href="https://hf.co/autotrain">AutoTrain – Hugging Face</a>：未找到描述

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1250129165429243986)** (10 messages🔥): 

- **Replicate 额度故障排查咨询**：多位用户反映尽管按照说明操作，但仍未收到 Replicate 额度。支持团队的一名成员做出了回应，要求用户私信用户名和电子邮件以解决问题。
  
- **无需设置账单即可获取 Replicate 额度**：一位用户询问是否需要在 Replicate 上设置账单信息才能接收额度。回复澄清说，获取额度不需要设置账单。

- **关于训练和部署 OSS Tool Calling 模型的反馈**：一位成员分享了他们使用入门仓库训练和部署 OSS Tool Calling 模型的经验。他们寻求关于其设置是否正确的反馈，并表示有兴趣进行进一步讨论。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1249810351667675266)** (7 messages): 

- **关于额度的邮件沟通**：一位用户提到他们发送了一封关于额度（credits）的邮件。随后，另一位用户确认了针对此事的邮件回复。
- **最初不需要添加支付方式**：解释提到 *"访问您的额度需要在档案中有一个有效的支付方式"*，但用户在填写额度表单时不需要设置账单。[提供的参考链接](https://discord.com/channels/1238365980128706560/1241167367040405544/1247687054826012693)。
- **表单缺失 Org ID**：一位用户在表单提交时漏填了 Org ID，但仍获得了额度方面的协助。*"我已经为您添加了这些额度。"*
- **通过邮件解决额度问题**：用户被要求通过私信或发送邮件至 jess@langchain.dev 提供用于额度表单的邮箱。这是解决额度添加问题的一部分。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/1250157367056662728)** (3 messages): 

- **Chain of Thought 导致错误的结论**：一位成员讨论了使用 **Chain of Thought (CoT)** 推理进行诊断步骤，但他们的模型（**Llama-3 8B**）有时会得出错误的结论。给出的一个例子是，模型在诊断区间内的患者年龄时错误地表示 "*这违反了规则*"。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[whitaker_napkin_math](https://discord.com/channels/1238365980128706560/1242223332695478332/1250123249715646585)** (2 messages): 

- **估算 DPO 训练的 VRAM 消耗很棘手**：一位成员对 **DPO** 训练期间波动的 **VRAM** 消耗表示担忧，寻求估算方法以避免显存溢出（**OOM**）错误。另一位成员建议从最长的序列开始，以防止训练中期出现意外的 **VRAM** 峰值。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-4](https://discord.com/channels/1238365980128706560/1242223495673286737/1250020611938324510)** (9 messages🔥): 

- **Apple 加入 LoRA 切换领域**：一位成员指出 Apple 推出了一种“动态切换 **LoRA**”的新技术，引起了同行对它与 **S-LoRA** 相似性的兴奋和好奇。另一位成员询问了根据查询类型实现**动态专业化 LoRA 适配器**的资源，得到了诸如 "Lorax" 的建议，并提到了相关工作，如 **LoRA 的 CBTM** 以及**针对每个任务的 prompt 和数据集的语义相似性**。 
- **Workshop 的见解早于 Apple 的公告**：另一位成员强调，Apple 讨论的这种动态 **LoRA** 切换概念在 Workshop 中已经涵盖过（尽管是针对云端应用的），这使得设备端适配变得令人兴奋且具有前瞻性。回顾来自 "Travis" 的见解，他们赞赏 Workshop 提供的远见和详细理解。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1249873584231551076)** (129 条消息🔥🔥): 

- **Jeremy Howard 分享《黑客语言模型指南》**：Jeremy Howard 推荐了一个名为 [“A Hackers' Guide to Language Models”](https://www.youtube.com/watch?v=jkrNMKz9pWU) 的 YouTube 视频。他的视频被描述为“极具启发性”，涵盖了现代语言模型的全面用途。
- **Ben Clavié 关于 Reranking 和 NER 的资源**：Ben 分享了多个宝贵资源，包括 [rerankers 的 GitHub 链接](https://github.com/AnswerDotAI/rerankers) 以及对 GLiNER 的详细解释。GLiNER 是一个用于 zero-shot 实体识别的强大模型。他强调了该模型处理内部术语和特定类别的能力。
- **关于余弦距离（Cosine Distance）与 L2 距离的讨论**：成员们讨论了在 RAG 应用的向量搜索中，使用余弦距离与归一化欧几里得距离（L2）的优劣。他们得出的结论是“余弦距离等于归一化的欧几里得距离”。
- **分享其他工具和库**：成员们分享了用于制作流程图的 [Excalidraw](https://excalidraw.com/) 等工具，以及用于微调 Transformer 的各种资源，如 [Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/training_overview.html)。 
- **Maven 上的视频托管挑战**：Ben Clavié 在 Maven 上的演讲视频播放出现问题，显然是由于 Zoom 的转录过程导致的。目前正在努力解决此问题，旨在让这些资料重新可供访问。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://stats.stackexchange.com/questions/71614/distance-measure-of-angles-between-two-vectors-taking-magnitude-into-account">两个向量之间角度的距离测量，考虑量级</a>：假设我有两个向量 v1 和 v2，我可以利用 arccos 函数计算这两个向量之间的角度，作为它们“距离”的度量。例如：&#xA;&#xA;...</li><li><a href="https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html">Dense vector 字段类型 | Elasticsearch 指南 [8.14] | Elastic</a>：未找到描述</li><li><a href="https://github.com/urchade/GLiNER">GitHub - urchade/GLiNER: 通用且轻量级的命名实体识别模型（从文本中提取任何实体类型）@ NAACL 2024</a>：通用且轻量级的命名实体识别模型（从文本中提取任何实体类型）@ NAACL 2024 - urchade/GLiNER</li><li><a href="https://sbert.net/">SentenceTransformers 文档 &mdash; Sentence Transformers 文档</a>：未找到描述</li><li><a href="https://excalidraw.com/">Excalidraw — 让协作白板变得简单</a>：Excalidraw 是一款虚拟协作白板工具，可让你轻松绘制具有手绘感的图表。</li><li><a href="https://www.youtube.com/watch?v=jkrNMKz9pWU">A Hackers&#39; Guide to Language Models</a>：在这段极具启发性的视频中，fast.ai 联合创始人、ULMFiT 方法（所有现代语言模型的基础）的创建者 Jeremy Howard...</li><li><a href="https://tenor.com/view/clem-fandango-steven-toast-toast-of-london-yes-i-can-hear-you-clem-fandango-gif-9211791307522605321">Clem Fandango Steven Toast GIF - Clem Fandango Steven Toast Toast of London - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/AnswerDotAI/rerankers">GitHub - AnswerDotAI/rerankers</a>：通过在 GitHub 上创建账号来为 AnswerDotAI/rerankers 做出贡献。</li><li><a href="https://gist.github.com/bclavie/f7b041328615d52cf5c0a9caaf03fd5e">rag_mvp.py</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://www.sbert.net/docs/sentence_transformer/training_overview.html">训练概览 &mdash; Sentence Transformers 文档</a>：未找到描述</li><li><a href="https://x.com/bclavie">来自未定义用户的推文</a>：未找到描述</li><li><a href="https://github.com/bclavie/RAGatouille">GitHub - bclavie/RAGatouille: 在任何 RAG 流水线中轻松使用和训练最先进的后期交互检索方法 (ColBERT)。专为模块化和易用性设计，并有研究支持。</a>：在任何 RAG 流水线中轻松使用和训练最先进的后期交互检索方法 (ColBERT)。专为模块化和易用性设计，并有研究支持。 - bclavie/RAGatouille</li><li><a href="https://arxiv.org/abs/2311.08526">GLiNER: 使用双向 Transformer 的通用命名实体识别模型</a>：命名实体识别 (NER) 在各种自然语言处理 (NLP) 应用中至关重要。传统的 NER 模型虽然有效，但局限于一组预定义的实体类型。相比之下...
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1249815782934974605)** (1 messages): 

- **探索比 function calling 更快的实体提取**：用户讨论了将类别集成为 metadata 以潜在地改进 reranker。他们选择了实体提取 + router model 的方法，而不是 function calling，因为前者在复杂性和速度上更具优势。
- **寻求模型训练细节的见解**：另一位用户寻求关于使用 function calling 的模型的复杂性和训练细节。他们询问了所需的样本量、准备的函数数量，以及产品数据的复杂性细节，例如 “is_accessory_of” 或 “bought_together” 等关系。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1249869859828011071)** (1 messages): 

- **对 Fasthtml 的兴奋**：一位用户表达了他们对 **fasthtml** 的兴奋，强调了他们在将 Streamlit 应用扩展到更复杂的应用程序时遇到的困难。他们提到 fasthtml 可能会让他们免于学习 Typescript。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[saroufimxu_slaying_ooms](https://discord.com/channels/1238365980128706560/1242224552415596554/1249805912080650241)** (148 messages🔥🔥): 

- **卸载 Optimizer State 以避免 OOM**：用户讨论了将 **optimizer state 卸载到 CPU** 或 CUDA 托管内存的策略，以防止模型训练中的 Out of Memory (OOM) 错误。他们强调了性能不可预测性的权衡，以及 fused optimizers 对加速 `optimizer.step` 操作的重要性。

- **分享高效深度学习技术的见解**：几位用户交流了关于 **bnb 8bit casting** 和 **LoRA** 等高级优化技术的见解。他们探讨了这些技术如何在训练期间节省内存并增强模型性能。

- **广泛的资源分享**：成员们分享了大量关于模型训练优化的资源，包括 **Profetto UI**、**torch profiler** 的链接以及各种 GitHub 仓库。具体的 URL 包括一个[关于 8-bit 深度学习的 YouTube 视频](https://www.youtube.com/watch?v=jyOqtw4ry2w&themeRefresh=1)和一个包含相关幻灯片和 trace 的 [Google Drive](https://drive.google.com/drive/u/3/folders/1HmGNC4v4L5nXhtdDMVCpUBrme1ELp-2C)。

- **关于模型量化和 FSDP 的热烈讨论**：用户积极讨论了量化的好处和复杂性，特别是使用 FSDP2 等工具时，**强调了高效的内存管理**。对话突出了处理 NF4 tensors 和大规模模型训练的实际实现与挑战。

- **互动且充满感激的社区参与**：聊天中充满了支持性的互动、幽默和赞扬，特别是针对特定成员分享的富有启发性的演讲和材料。社区对详细的演示表示感谢，一位成员幽默地补充道：*“对于我们这样的人群来说，Meme 是最好的信息传播方式。”*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://perfetto.dev/">Perfetto - 系统性能分析、应用追踪与追踪分析</a>: 未找到描述</li><li><a href="https://vast.ai/docs/autoscaler/introduction">概览 | Vast.ai</a>: 未找到描述</li><li><a href="https://tenor.com/view/im-pretending-i-know-what-youre-talking-about-ahmed-aldoori-i-have-no-idea-faking-it-pretending-gif-18453815">我在假装听懂你在说什么 Ahmed Aldoori GIF - 我在假装听懂你在说什么 Ahmed Aldoori 我完全不知道 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/xHx8.gif">我认识其中一些词 Mhmm GIF - 我认识其中一些词 Mhmm 毫无头绪 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-tracing-functionality">PyTorch Profiler &mdash; PyTorch 教程 2.3.0+cu121 文档</a>: 未找到描述</li><li><a href="https://tenor.com/view/thanks-bow-thank-you-sign-of-respect-gif-4807966236937524301">谢谢鞠躬 GIF - 谢谢鞠躬 谢谢你 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=SKV6kDk1s94">第 16 讲：上手 Profiling</a>: 未找到描述</li><li><a href="https://drive.google.com/">Google Drive：登录</a>: 未找到描述</li><li><a href="https://discord.gg/RfcRWeNs">加入 llm-fine-tuning Discord 服务器！</a>: 查看 Discord 上的 llm-fine-tuning 社区 - 与其他 1895 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://www.youtube.com/watch?v=jyOqtw4ry2w&themeRefresh=1">与 Tim Dettmers 探讨高效深度学习的 8-bit 方法</a>: Tim Dettmers（华盛顿大学博士候选人）在本次 Cohere For AI 技术演讲中展示了“高效深度学习的 8-bit 方法”。摘要：La...</li><li><a href="https://drive.google.com/drive/u/3/folders/1HmGNC4v4L5nXhtdDMVCpUBrme1ELp-2C">解决 OOMs 追踪 – Google Drive</a>: 未找到描述</li><li><a href="https://github.com/yandex/YaFSDP">GitHub - yandex/YaFSDP: YaFSDP: 又一个 Fully Sharded Data Parallel</a>: YaFSDP: 又一个 Fully Sharded Data Parallel。通过在 GitHub 上创建账号来为 yandex/YaFSDP 的开发做出贡献。</li><li><a href="https://asmirnov.xyz/vram">拆解 GPU VRAM 消耗</a>: 未找到描述</li><li><a href="https://drive.google.com/drive/u/0/folders/1HmGNC4v4L5nXhtdDMVCpUBrme1ELp-2C">解决 OOMs 追踪 – Google Drive</a>: 未找到描述</li><li><a href="https://github.com/pytorch/torchtune">GitHub - pytorch/torchtune: 一个用于 LLM 微调的原生 PyTorch 库</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html">Answer.AI - 你现在可以在家训练 70b 语言模型了</a>: 我们正在发布一个基于 FSDP 和 QLoRA 的开源系统，可以在两块 24GB GPU 上训练 70b 模型。</li><li><a href="https://vram.asmirnov.xyz/">VRAM 计算器</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/pytorch/torchtitan">GitHub - pytorch/torchtitan: 一个用于大模型训练的原生 PyTorch 库</a>: 一个用于大模型训练的原生 PyTorch 库。通过在 GitHub 上创建账号来为 pytorch/torchtitan 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/909">由 weifengpy 开启 QLoRA + FSDP2 · Pull Request #909 · pytorch/torchtune</a>: 此 PR 构建在包含 NF4Tensor FSDP2 算子 PR1 PR2 的 TorchAO nightly 以及包含 meta init + cpu offloading PR 的 Pytorch nightly 之上。单元测试：pytest -s tests/torchtune/utils/test_di...</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/dtypes/nf4tensor.py#L801">ao/torchao/dtypes/nf4tensor.py 位于 main 分支 · pytorch/ao</a>: 用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao</li><li><a href="https://drive.google.com/file/d/1yJ176PyAyiMJLI07PL5Mhq-_E7-q2z-B/view">FINAL_torchtune_BIG_wrapping_policy_fused_adamw_llama2_7b_dummy_bs8_cpu_offload_ns48_threads8.json</a>: 未找到描述</li><li><a href="https://vast.ai/">租用 GPU | Vast.ai</a>: 通过最优质的云端 GPU 租赁服务，将您的云计算成本降低 3-5 倍。Vast.ai 简单的搜索界面让您可以公平地比较来自所有供应商的 GPU 租赁。</li><li><a href="https://mlflow.org/">MLflow | MLflow</a>: 描述将放入 &lt;head /&gt; 中的 meta 标签</li><li><a href="https://github.com/janeyx99">janeyx99 - 概览</a>: janeyx99 拥有 32 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://x.com/marksaroufim">来自未定义用户的推文</a>: 未找到描述</li><li><a href="https://github.com/msaroufim">msaroufim - 概览</a>: CUDA 卸载失败。请联系支持。</li>

ørt før åššīštåñćē - msaroufim</li><li><a href="https://github.com/drisspg">drisspg - 概览</a>：@pytorch 核心成员。drisspg 拥有 37 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/awgu">awgu - 概览</a>：awgu 拥有 10 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/weifengpy">weifengpy - 概览</a>：PyTorch Distributed。weifengpy 拥有 7 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/rohan-varma">rohan-varma - 概览</a>：PyTorch @facebook | UCLA。rohan-varma 拥有 83 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/ebsmothers">ebsmothers - 概览</a>：ebsmothers 拥有 8 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/pytorch/torchtune/blob/1fa1f04baf124c074dcd93831fa38c8b657af1e9/recipes/configs/dev/llama2/7B_qlora_fsdp2.yaml">pytorch/torchtune 中的 torchtune/recipes/configs/dev/llama2/7B_qlora_fsdp2.yaml</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486">FSDP &amp; CUDACachingAllocator：一个局外新手的视角</a>：大家好。这次讨论的主要动力是：FSDP 令人质疑的 profile 结果，这促使 Ke W. + Alban D. + Andrew G. 讨论解决方案，我也从中受益匪浅...</li><li><a href="https://github.com/pytorch/pytorch/blob/f600faf2480ddd6600ad88fbfc5dd28da132d61d/torch/distributed/_composable/fsdp/_fsdp_param.py#L515">pytorch/pytorch 中的 torch/distributed/_composable/fsdp/_fsdp_param.py</a>：Python 中具有强 GPU 加速能力的张量和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[paige_when_finetune](https://discord.com/channels/1238365980128706560/1242224662142779530/1249860352565448834)** (166 条消息🔥🔥): 

- **爆米花分类变得极客化**：频道内幽默地讨论了使用合成的爆米花爆裂时间数据来微调 LLM，并对爆米花核心进行生存分析。一位成员调侃道：“谁要是根据 ftcourse 仓库对爆米花核心做一个案例研究，那绝对是传奇。”

- **逆泊松分布引发数学讨论**：围绕逆泊松分布话题展开了详细讨论，一位用户分享了 [math stack exchange 链接](https://math.stackexchange.com/questions/1195566/inverse-of-a-poisson-distribution-function) 来解释公式和随机性。

- **Gemini API 引起关注**：成员们讨论了 Google 的 Gemini Flash 支持音频输入的功能，并引用了 [文档](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Audio.ipynb)。另一位成员询问了将音频摄取到 Gemini API 的过程。

- **Prompt 工程技巧**：一个关键讨论围绕着使用模型为自己创建 Prompt，包括 meta-prompt 策略和自我改进 Prompt 技术的示例。一位用户分享道：“*你可以要求模型为自己（或其他模型）编写 Prompt*”。

- **感谢致辞与资源分享**：聊天以对 Paige 演讲的感谢结束，并分享了一些重要资源，如电子邮件联系方式和补充阅读材料。[Paige 的个人网站](https://webpaige.dev/) 和 [context caching](https://ai.google.dev/gemini-api/docs/caching) 文档被作为有用的后续链接分享。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://ai.google.dev/gemini-api/docs/caching">未找到标题</a>: 未找到描述</li><li><a href="https://ai.google.dev/pricing">未找到标题</a>: 未找到描述</li><li><a href="https://math.stackexchange.com/questions/1195566/inverse-of-a-poisson-distribution-function">泊松分布函数的逆函数</a>: 我有两个独立同分布（i.i.d）的随机变量 $X_{1}$ 和 $X_{2}$，服从连续泊松分布函数&#xA;&#xA;$P(x) = \lambda e^{-\lambda\cdot x}$。&#xA;&#xA;我希望获得一个分布函数...</li><li><a href="https://ai.google.dev/gemini-api/docs/get-started/android_aicore">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/googledevs/status/1800565067324195032">来自 Google for Developers (@googledevs) 的推文</a>: 📣 🧠 对于挑战高效深度学习极限的研究人员来说，这是一个激动人心的消息！我们已将 RecurrentGemma 扩展到了 90 亿参数。🧵↓</li><li><a href="https://huggingface.co/blog/paligemma">PaliGemma – Google 前沿的开源视觉语言模型</a>: 未找到描述</li><li><a href="https://tenor.com/view/spongebob-patrick-star-noted-notes-gif-17474838830648097856">海绵宝宝派大星 GIF - 海绵宝宝派大星 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/google-gemini/cookbook/blob/main/quickstarts/PDF_Files.ipynb">google-gemini/cookbook 仓库中的 cookbook/quickstarts/PDF_Files.ipynb</a>: Gemini API 的指南和示例集合。- google-gemini/cookbook</li><li><a href="https://tenor.com/view/so-excited-cant-wait-gif-24703188">太兴奋了 GIF - 太兴奋了等不及了 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: 结构化文本生成</a>: 结构化文本生成。通过在 GitHub 上创建账号来为 outlines-dev/outlines 的开发做出贡献。</li><li><a href="https://github.com/google-gemini/cookbook/blob/main/quickstarts/Audio.ipynb">google-gemini/cookbook 仓库中的 cookbook/quickstarts/Audio.ipynb</a>: Gemini API 的指南和示例集合。- google-gemini/cookbook</li><li><a href="https://github.com/google-research/t5x">GitHub - google-research/t5x</a>: 通过在 GitHub 上创建账号来为 google-research/t5x 的开发做出贡献。</li><li><a href="https://simonwillison.net/2024/Feb/21/gemini-pro-video/#images-vs-video">Gemini Pro 1.5 的杀手级应用是视频</a>: 上周 Google 推出了 Gemini Pro 1.5，这是对其 Gemini 系列 AI 模型的重大升级。Gemini Pro 1.5 拥有 1,000,000 token 的上下文窗口。这非常惊人——在此之前……</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-token-count-multimodal">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/dynamicwebpaige">来自未定义用户的推文</a>: 未找到描述</li><li><a href="https://webpaige.dev/">webpaige.dev</a>: 未找到描述</li><li><a href="https://cloud.google.com/vertex-ai?hl=en">配备 Gemini 1.5 Pro 和 Gemini 1.5 Flash 的 Vertex AI</a>: 尝试 Vertex AI，这是一个用于构建生成式 AI 应用的全托管 AI 开发平台，可访问包括 Gemini 1.5 模型在内的 130 多个基础模型。</li><li><a href="https://www.youtube.com/watch?v=wa0MT8OwHuk">使用 44 分钟电影进行多模态提示 | Gemini 1.5 Pro 演示</a>: 这是长上下文理解的演示，它是我们最新模型 Gemini 1.5 Pro 中的一项实验性功能，使用了 44 分钟的 Buster Keaton 默片《福尔摩斯二世》...</li><li><a href="https://www.youtube.com/watch?v=LHKL_210CcU">在 402 页的转录文本中进行推理 | Gemini 1.5 Pro 演示</a>: 这是长上下文理解的演示，它是我们最新模型 Gemini 1.5 Pro 中的一项实验性功能，使用了 402 页的 PDF 转录文本和一系列多模态...</li><li><a href="https://www.youtube.com/watch?v=SSnsmqIj1MI">在 100,633 行代码中解决问题 | Gemini 1.5 Pro 演示</a>: 这是长上下文理解的演示，它是我们最新模型 Gemini 1.5 Pro 中的一项实验性功能，使用了 100,633 行代码和一系列多模态...</li><li><a href="https://aistudio.google.com/">未找到标题</a>: 未找到描述</li><li><a href="https://discuss.ai.google.dev/">使用 Google AI 进行构建</a>: 在 Google 的 Gemini API 和 Google AI Studio 上提问并获取支持
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1250089153475903580)** (2 messages): 

- **LLAMA3 LORA 训练在 RTX4090 上遇到 OOM 问题**：一位用户报告在 RTX4090 上尝试将 LORA 权重与基础模型合并时遇到 OOM (Out of Memory) 错误，尽管尝试了 `lora_on_cpu: true` 和 `gpu_memory_limit` 等建议的解决方案。他们引用了 [Axolotl GitHub README](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base) 获取详细信息。
- **分享 Axolotl 数据集格式和资源**：同一位用户分享了多个链接，以帮助理解 Axolotl 支持的数据集格式和 HuggingFace chat templates。其中包括 [Axolotl 文档](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)、[HuggingFace chat templating](https://huggingface.co/docs/transformers/en/chat_templating) 以及一个相关的 GitHub 仓库 [Chat Templates for HuggingFace](https://github.com/chujiezheng/chat_templates)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base).">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>：尽管去问 Axolotl 问题。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - 数据集格式</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/chat_templating">Chat Models 的模板</a>：未找到描述</li><li><a href="https://github.com/chujiezheng/chat_templates">GitHub - chujiezheng/chat_templates: Chat Templates for 🤗 HuggingFace Large Language Models</a>：🤗 HuggingFace 大语言模型的 Chat Templates - chujiezheng/chat_templates
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1249818485127643187)** (8 messages🔥): 

- **Codingwitcher 喜欢 Modal 的独特方法**：Codingwitcher 分享了他们在 Modal 上部署 **Mistral 进行 inference** 的兴奋之情。他们将这种体验描述为“神奇”，特别赞赏远程机器上的 hot-reload 功能。
- **Ed157 寻求微调设置方面的帮助**：Ed157 请求指导在指令微调的 config YAML 文件的 datasets 和 tokens 部分中应输入什么内容。他们提供了一个模板来阐明自己的需求。
- **DamonCrockett 面临技术挑战**：DamonCrockett 在 Modal 上运行 llm-finetuning 示例时遇到了与缺失 volume ID 相关的错误消息。尽管之前运行成功，他们仍在寻求协助以解决此问题。
- **Charles 转发支持查询**：Charles_irl 建议就 DamonCrockett 的问题联系 [Slack 上的 Modal 团队](https://modal.com/slack)，并建议 Ed157 关于 axolotl 的问题去 <#1242542198008975430> 频道。
- **Danbecker 整合演示讨论**：Danbecker 指示使用 <#1241044231829848125> 频道进行即将到来的关于 Charles 演示的讨论，以保持内容条理性。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1249845739723554838)** (7 messages): 

- **用户请求添加额度**：多位成员请求向其账户添加 credits。分享了如 `i-00dda2`、`dimitry-611a0a`、`tanmaygupta9-70b723`、`contact-ff3a2c`、`ferdousbd-24e887`、`yorick-van-zweeden-e9b5c2` 和 `ashiqur-cd00ce` 等成员账户 ID，希望能解决额度问题。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[emmanuel_finetuning_dead](https://discord.com/channels/1238365980128706560/1245129595749925086/1250127543621779489)** (4 条消息): 

```html
- **Request for Fine-Tuning Example**: One user asked for an example that illustrates the **fine-tuning process**, such as a notebook, GitHub repo, or blog post. They inquired whether this process can be done with **existing frameworks like TRL or Axolotl**.

- **Dataset Preparation Standard**: Another member shared a [link](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) to the OpenAI guidelines for preparing datasets for fine-tuning, establishing it as a standard reference.

- **Two-Step Fine-Tuning Process**: Clarification was made on a two-step process for fine-tuning, which includes pretraining and alignment during finetuning. The discussion emphasized *"adding a 'head' layer on the pre-trained model's transformer stack for NLP tasks"* and using QLora to mitigate OOM errors.

- **Technical Breakdown of Mistral Model**: The member provided an example with detailed code illustrating a **MistralForCausalLM** model. The explanation detailed how the last layer `lm_head` functions and how **QLora** replaces linear layers with low-rank matrices to handle out-of-memory errors.
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[west-coast-usa](https://discord.com/channels/1238365980128706560/1245410680065097738/)** (1 条消息): 

jonbiz: 如果时间允许，我们可以聚聚？看看还有谁感兴趣？
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1249802975136977067)** (2 条消息): 

- **用户报告收到 25 美元额度**：一位名叫 David 的用户报告称注册并收到了 25 美元的额度，并分享了他的租户 ID (c4697a91)。Michael Ortega 回应称，他会为 David 跟进此事。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/)** (1 条消息): 

_iw3: 嗨，我也仍然看到 100 美元的额度而不是 222 美元，我应该找谁核实？谢谢
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1249808714853122070)** (16 条消息🔥): 

- **Prompt 评分工具包建议**：一位用户请求推荐用于对大量 Prompt 列表进行评分的工具，要求具备错误处理和断点续传等功能。推荐的工具包括 Promptlayer 和 Promptfoo，其中一位用户特别寻求 CLI 解决方案。
- **使用 OpenAI 额度**：用户讨论了使用 OpenAI 额度的各种方式。一位用户提到将它们用于 embedding 任务并尝试最近 RAG 演讲中的技术，另一位用户提到根据他们的书生成内容。
- **额度接收问题**：一位用户报告称填写了表格并多次给支持部门发邮件，但仍未收到 OpenAI 额度。他们提供了自己的 Org ID 和 User ID，试图解决该问题。
- **Tier 2 和 GPT-4 访问权限**：讨论了关于通过 Tier 2 使用计划访问 GPT-4/4o 的问题。一位用户分享说，他们在添加付款方式和额度后才能访问。
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1249800510568136775)** (402 条消息🔥🔥): 

- **成员受困于慢速网络**：一位成员报告称，由于达到数据流量上限导致限速，下载 Lora Maker 花费了 12 小时。他们提到，即使速度慢到“从 Plytorch.org 下载只有 50kb/s”，但“YT 还能用”。
- **SD3 发布前的紧张气氛**：随着成员们讨论 Stable Diffusion 3 的即将发布，热情和期待不断高涨。一位成员幽默地问道：“你是说在它发布之前我还得再上床睡一觉？”而另一位成员则期待更好的像素艺术生成效果：“我希望 sd3 在体素艺术（voxel art）方面也很出色”。
- **AI 模型和平台探索**：用户比较了 Huggingface 和 Civitai 等平台的模型可用性。一位成员提到在 Civitai 上可以找到大多数 Loras 和 checkpoints，但也指出“有很多合规内容是以 torrent 格式提供的”。
- **SDXL 与传统放大（upscaling）之争**：讨论围绕着将 SD1.5 图像放大到 1024 是否能达到与针对 1024x1024 训练的 SDXL 类似的效果。一位用户提出了一个实际的解决方案：“当你用 SDXL 放大到 2048 时再试一下。”
- **Stable Diffusion 安装挑战**：一位成员在尝试使用 AMD GPU 运行 Stable Diffusion 时遇到困难，并表达了沮丧：“问题出在哪？……它没在用我的 GPU”。建议他们重新查看安装指南并寻求技术支持频道的帮助。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://glif.app/@Oliveira/glifs/clw44qfbl0000m0zztwqk2tnf">glif - StableDiffusion 3 + GPT4 Helper + SDXL 1.5x Upscale (CopyGenius)，作者 Yuri Oliveira COPYGENIUS </a>：未找到描述</li><li><a href="https://sk2visual.gumroad.com/l/spsjsz">VISION Preset Pack #1 - @visualsk2</a>：VisualSK2 的预设包系列（PC-移动端）。这是我日常使用的最佳 Lightroom 预设集合，用于为我的拍摄提供电影感和一致的外观。内含 20 个预设...</li><li><a href="https://www.youtube.com/watch?v=KyLqUf4cdwc">Microsoft Vista 语音识别测试 - Perl 脚本</a>：感谢 scrubadub（查看用户 scrubadub1 获取更多此类视频！）首先分享了这段视频，直到他被封禁... 我们又开始了... 请不要...</li><li><a href="https://www.instagram.com/p/C6p8KgSSzo3/">madhav kohli 在 Instagram 上：“NCR 的恐惧与厌恶……”</a>：1.4 万次点赞，73 条评论 - mvdhav 于 2024 年 5 月 6 日发布：“NCR 的恐惧与厌恶……” </li><li><a href="https://youtu.be/ScPp2nhowgA">唐纳德·特朗普微积分之歌（他唱得很糟糕）</a>：唐纳德·特朗普微积分之歌（他唱得很糟糕）。这是一个学校项目。请点赞、评论并订阅。我的成绩全靠它了。🙏免责声明：这...</li><li><a href="https://www.instagram.com/p/C6_kd_hoNGb/">Samuele “SK2” Poggi 在 Instagram 上："[Vision III/Part. 4] ✨🤍 SK2• 快速的一天 •

#photography #longexposure #explore #trending #explorepage"</a>：3.3 万次点赞，265 条评论 - visualsk2 于 2024 年 5 月 15 日发布："[Vision III/Part. 4] ✨🤍 SK2• 快速的一天 • #photography #longexposure #explore #trending #explorepage"。 </li><li><a href="https://www.seaart.ai/models/detail/0e5b32eb19562e304d29771ad3898af5">Hard Muscle - SeaArt AI 模型</a>：未找到描述</li><li><a href="https://www.instagram.com/p/C781eUDoJ2h/">Samuele “SK2” Poggi 在 Instagram 上："[Vision IV/Part.6] 非常感谢 170,000 名粉丝 ✨🙏🏼🏼
距离教程发布仅剩几天。

#grainisgood #idea #reels #framebyframe #photography #blurry #explorepage"</a>：1.6 万次点赞，130 条评论 - visualsk2 于 2024 年 6 月 8 日发布："[Vision IV/Part.6] 非常感谢 170,000 名粉丝 ✨🙏🏼🏼 距离教程发布仅剩几天。 #gra...</li><li><a href="https://github.com/lks-ai/ComfyUI-StableAudioSampler">GitHub - lks-ai/ComfyUI-StableAudioSampler: ComfyUI 节点中的新 Stable Diffusion Audio Sampler 1.0。来做点节奏吧！</a>：ComfyUI 节点中的新 Stable Diffusion Audio Sampler 1.0。来做点节奏吧！ - lks-ai/ComfyUI-StableAudioSampler</li><li><a href="https://aitracker.art/">主页 :: AiTracker</a>：未找到描述</li><li><a href="https://tensor.art/models/654286272942196700">Hard Muscle - v1.0 | Stable Diffusion Checkpoint</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1249804822316843111)** (164 条消息🔥🔥): 

- **预计 2024 年 7 月初支持 Multigpu**：成员们热切期待 Multigpu 支持的发布，初步定于 2024 年 7 月初。一位成员幽默地提到：“2025 年，但不，说认真的，是 2024 年 7 月初。”
- **LORA 推理接口与 vLLM**：大家对在推理过程中允许启用/禁用 LORA 的推理接口很感兴趣。一位用户发现 vLLM 支持此功能，并思考它是否可以与 exl2 或 TabbyAPI 配合使用。
- **训练中的过拟合问题**：一位成员在模型训练中遇到了 Overfitting（过拟合）问题，导致在简单任务上的表现较差。建议包括尝试数据增强、利用 weight decay（权重衰减），以及确保训练数据的多样性和全面性。
- **Fine-Tuning 与 EOS Token 讨论**：成员们讨论了在通用文本上训练 instruct 模型时 EOS token 的重要性。有人建议在持续预训练中使用 `BOS_token + entire text + EOS_token`。
- **Hugging Face AutoTrain 新增 Unsloth 支持**：[Hugging Face AutoTrain 现在支持 Unsloth](https://x.com/abhi1thakur/status/1800511251145015393)，使用户能够更高效地 Fine-tune LLMs。这一新功能受到了热烈欢迎和赞赏。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.05587">Creativity Has Left the Chat: The Price of Debiasing Language Models</a>: Large Language Models (LLMs) 彻底改变了自然语言处理，但可能会表现出偏见并可能生成有害内容。虽然像 Reinforcement Learning from Human Fe... 等对齐技术</li><li><a href="https://huggingface.co/bartowski/Qwen2-72B-Instruct-GGUF">bartowski/Qwen2-72B-Instruct-GGUF · Hugging Face</a>: 暂无描述</li><li><a href="https://x.com/abhi1thakur/status/1800511251145015393">来自 abhishek (@abhi1thakur) 的推文</a>: AutoTrain + Unsloth = 🚀🚀🚀 AutoTrain 现在增加了对 Unsloth 的支持，这意味着你可以使用 Unsloth 的优化来超快速且以更少的显存 Fine-tune LLMs 💥 你只需要...</li><li><a href="https://github.com/unslothai/unsloth/pull/609">在调用 internal_model.save_pretr… 之前清除所有 selected_adapters，由 neph1 提交 · Pull Request #609 · unslothai/unsloth</a>: …ained 我有一个脚本从 HF 下载 adapter，将其与基础模型合并并上传结果。它在一个月前（左右）还可以工作，但现在失败了。在 Colab、Kaggle 和本地都试过...</li><li><a href="https://github.com/unslothai/unsloth/issues/611">save_pretrained_merged 未合并模型 · Issue #611 · unslothai/unsloth</a>: 问题：我的目标是将合并后的模型保存为 GGUF 文件，但我遇到了各种错误。更深层的问题似乎是合并 LORA + 基础模型并没有保存合并后的文件。我认为...</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: 暂无描述</li><li><a href="https://xtherapy.streamlit.app/">无标题</a>: 暂无描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1249821272008032337)** (72 条消息🔥🔥): 

- **重塑设计配色**：几位用户讨论了聊天机器人界面的改进，建议将红色背景改为白色，将正方形改为菱形，并降低色彩饱和度。一位用户评论道：*"好了，现在这个版本看起来没那么蠢了。"*
- **用于可扩展图像生成的 Llama**：对话涉及了 [LlamaGen](https://github.com/FoundationVision/LlamaGen)，这是一个用于可扩展图像生成的自回归模型。大家对该项目讨论热烈，并分享了 GitHub 链接。
- **Apple 在 WWDC 上的新 AI 集成**：Apple 宣布的个性化 AI（"Apple Intelligence"）引发了关于其效率和集成大模型潜力的讨论。用户们辩论了其实现方式和潜在的语言支持，有评论称：*"Apple 完美地将各项功能集成到了他们的 App 中。"* 
- **即时训练 (Training on the Fly)**：讨论了即时模型训练的可行性和益处。人们对训练成本和质量表示担忧，而一些人则看到了每日 finetuning 的潜力，将实时训练比作金融应用。
- **在线机器学习的局限性**：讨论强调了在线机器学习（online machine learning）的潜在问题，如灾难性遗忘（catastrophic forgetting）和人工数据输入的质量。一位用户提到：*"它之所以没能成功可能有几个原因。我猜灾难性遗忘是其中之一。"*
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/FoundationVision/LlamaGen">GitHub - FoundationVision/LlamaGen: Autoregressive Model Beats Diffusion: 🦙 Llama for Scalable Image Generation</a>: 自回归模型击败扩散模型：🦙 用于可扩展图像生成的 Llama - FoundationVision/LlamaGen</li><li><a href="https://github.com/ml-explore">ml-explore</a>: 在笔记本电脑或数据中心进行机器学习研究 - 由 Apple 提供 - ml-explore</li><li><a href="https://www.macrumors.com/2024/06/10/apple-intelligence-generative-personal-ai-unveiled-for-iphone-ipad-and-mac/">'Apple Intelligence' Personal AI Unveiled for iPhone, iPad, and Mac</a>: Apple 在今天的 WWDC 上宣布了 Apple Intelligence，这是一种为 Apple 设备深度集成、个性化的 AI 体验，使用了尖端的生成式...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1249817994024980500)** (67 条消息🔥🔥): 

- **Tokenizer 并非 Unsloth 模型特有**："任何服务的 Tokenizer 都可以（llama.cpp tokenizer, Hugging Face tokenizer）。事实上，除了 llama.cpp 之外，可能其他工具都在使用 Hugging Face tokenizer（包括 Unsloth）"。
- **Unsloth 模型大小混淆已解决**：一位用户询问为什么保存模型后只得到一个 100MB 的文件。澄清如下："Save_pretrained_merged 应该会保存整个模型"。
- **微调所需的示例 CSV 格式已确认**：一位用户询问他们的 CSV 格式对于微调是否正确；讨论了 "question,answer" 格式。他们被引导至一段 [YouTube 视频](https://www.youtube.com/watch?v=3eq84KrdTWY&t=665s) 以获取详细指导。
- **多 GPU 支持将于 7 月推出**："目前 Unsloth 仅支持单 GPU。我们将在 7 月初推出多 GPU 支持"，并澄清 Unsloth Pro 主要面向企业。
- **无法直接微调 GGUF 模型**：尝试微调 GGUF 模型时得到的解释是目前不支持，并建议使用 [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/gguf#support-within-transformers) 中新的实验性互操作功能。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=3e"> - YouTube</a>: 未找到描述</li><li><a href="https://unsloth.ai/contact">Contact</a>: 未找到描述</li><li><a href="https://huggingface.co/Bibekananda/bk_gguf_Chat_model">Bibekananda/bk_gguf_Chat_model · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=3eq84KrdTWY&t=665s">Llama 3 Fine Tuning for Dummies (with 16k, 32k,... Context)</a>: 在本分步教程中，学习如何使用 Unsloth 轻松微调 Meta 强大的新 Llama 3 语言模型。我们涵盖了：Llama 3 8B 概述以及...</li><li><a href="https://huggingface.co/docs/transformers/en/gguf#support-within-transformers">GGUF and interaction with Transformers</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 微调 Llama 3, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/pull/609">clears any selected_adapters before calling internal_model.save_pretr… by neph1 · Pull Request #609 · unslothai/unsloth</a>: …ained 我有一个脚本从 HF 下载 adapter，将其与基础模型合并并上传结果。一个月前（左右）还能用，但现在失败了。在 Colab, Kaggle 和本地都试过...</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing)">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1250136212250034277)** (3 条消息): 

- **尝试基于 Llama 3 8B 微调的 xTherapy AI**：[查看这个心理治疗 AI](https://xtherapy.streamlit.app/)，它是使用 **Unsloth** 在 **Llama 3 8B** 上微调的。欢迎提供改进反馈。
  
- **CAMB AI 发布 MARS5 TTS**：[CAMB AI](https://github.com/camb-ai/mars5-tts) 已在 GitHub 上开源了他们的第 5 代 MARS TTS 模型。他们还被 [VentureBeat 报道](https://venturebeat.com/ai/exclusive-camb-takes-on-elevenlabs-with-open-voice-cloning-ai-model-mars5-offering-higher-realism-support-for-140-languages/)，并邀请社区提供反馈。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/camb-ai/mars5-tts">GitHub - Camb-ai/MARS5-TTS: MARS5 speech model (TTS) from CAMB.AI</a>: 来自 CAMB.AI 的 MARS5 语音模型 (TTS)。通过在 GitHub 上创建账号为 Camb-ai/MARS5-TTS 的开发做出贡献。</li><li><a href="https://xtherapy.streamlit.app/">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1249807677840953415)** (62 messages🔥🔥): 

- **Llama 3 的 4-bit 量化面临挑战**：一位成员寻求关于 **Llama 3** 8b 进行 **4-bit 量化**且性能损耗微乎其微的建议，用于训练 SAEs。其他成员建议尝试 **sharding**（分片）或使用 **Tensor Parallelism**（张量并行），但指出了这些方法的潜在挑战和实验性质 [PyTorch 中的 Tensor Parallelism](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)。

- **关于量化方法的辩论**：成员们讨论了最有效的量化方法，涉及 **IQ4_xs** 以及 **HQQ, AWQ, EXLv2**。分享了一个 [博客文章](https://stephenpanaro.com/blog/llm-quantization-for-iphone) 链接，展示了适用于 Apple Silicon 的各种量化技术，并声称其性能优于传统方法。

- **联邦学习（Federated Learning）的考量**：一位用户提出了 Apple 在不将数据移出设备的情况下利用个人数据进行训练的想法，引发了关于 **Federated Learning** 的讨论。讨论涉及对隐私和梯度数据潜在滥用的担忧，阐明了联邦学习实现的复杂性。

- **地理时间序列数据预测**：一位成员询问关于 **地理时间序列数据** 和预测特定地点事件的经验。另一位成员分享了他们使用 **Google Earth Engine + LSTM** 进行类似预测的经验。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://stephenpanaro.com/blog/llm-quantization-for-iphone">LLMs for your iPhone: Whole-Tensor 4 Bit Quantization</a>：介绍一种与 Apple Silicon 完全兼容的新型 4-bit 量化方案……</li><li><a href="https://pytorch.org/docs/stable/distributed.tensor.parallel.html">Tensor Parallelism - torch.distributed.tensor.parallel &mdash; PyTorch 2.3 documentation</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cst400/result_llama_3_mmlu_score_vs_quantization_for/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1249835887819231313)** (118 messages🔥🔥): 

- **寻求常规规模 LLMs 的 Online RL 研究**：一位成员正在寻找关于常规规模 LLMs 的 *Online RL* 研究论文，但发现理论假设在实践中往往难以证实。
  
- **VALL-E 2 推动 Zero-Shot TTS 进步**：[VALL-E 2](https://arxiv.org/abs/2406.05370) 在 Zero-Shot TTS 领域达到了人类水平，在 Repetition Aware Sampling 和 Grouped Code Modeling 方面有所改进。然而，由于过早泄露，[项目页面](https://web.archive.org/web/20240529183033/https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-2/) 的可用性处于波动状态。
  
- **LlamaGen 探索视觉 Tokenization**：[LlamaGen](https://arxiv.org/abs/2406.06525) 的新模型将自回归 Next-Token 预测应用于视觉领域，性能显著优于流行的 Diffusion 模型。讨论中提到了它在自回归解码中对 CFG 的新颖实现，让人联想到 [之前研究](https://arxiv.org/abs/2306.17806) 中的方法。

- **Transformer 学习能力的挑战**：诸如 [这篇论文](https://arxiv.org/abs/2406.06467) 概述了标准 Transformer 在不实现有监督 *scratchpads* 的情况下难以学习的任务。讨论触及了由于复杂 Token 交互中梯度下降的低效，导致 *无监督 scratchpads* 缺乏实用性的问题。

- **影响函数（Influence Functions）的有效性**：探讨了影响函数的效用和局限性，并链接到 [Koh 和 Liang 的论文](https://arxiv.org/pdf/1703.04730) 等基础解释，重点关注近似方法和实际适用性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.05370">VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers</a>: 本文介绍了 VALL-E 2，这是神经编解码语言模型的最新进展，标志着零样本（Zero-Shot）文本转语音合成（TTS）的一个里程碑，首次达到了人类水平。Ba...</li><li><a href="https://arxiv.org/abs/2406.06525">Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation</a>: 我们推出了 LlamaGen，这是一个新的图像生成模型系列，将大型语言模型原始的“下一个 Token 预测（next-token prediction）”范式应用于视觉生成领域。这是一个肯定的...</li><li><a href="https://arxiv.org/abs/2406.06248">Compute Better Spent: Replacing Dense Layers with Structured Matrices</a>: 稠密线性层是基础模型（foundation models）中的主要计算瓶颈。寻找稠密矩阵的高效替代方案对于构建更具计算效率的模型具有巨大潜力...</li><li><a href="http://arxiv.org/abs/2406.06248">Compute Better Spent: Replacing Dense Layers with Structured Matrices</a>: 稠密线性层是基础模型中的主要计算瓶颈。寻找稠密矩阵的高效替代方案对于构建更具计算效率的模型具有巨大潜力...</li><li><a href="http://arxiv.org/abs/2406.06467">How Far Can Transformers Reason? The Locality Barrier and Inductive Scratchpad</a>: Transformer 能通过组合已有的三段论来预测新的三段论吗？更广泛地说，这类模型从零开始可以学习到什么类型的目标？最近的研究表明 Transformer 可以是 Turin...</li><li><a href="https://arxiv.org/abs/2210.02671">A Logic for Expressing Log-Precision Transformers</a>: 解释基于 Transformer 的语言模型推理能力的一种方法是描述它们在输入文本上可以解析的逻辑规则类型。最近，Chiang 等人 (2023) 表明...</li><li><a href="https://arxiv.org/abs/2406.06484">Parallelizing Linear Transformers with the Delta Rule over Sequence Length</a>: 具有线性注意力（即线性 Transformer）和状态空间模型（state-space models）最近被认为是具有 Softmax 注意力的 Transformer 的可行线性时间替代方案。然而，...</li><li><a href="https://ai.stackexchange.com/q/45949/68078">Is a small transformer model able to effectively handle any input length provided it is fine-tuned on it?</a>: 假设我们有一个可以执行摘要任务的 Transformer LLM。我知道 Transformer 从技术上可以处理任何输入长度（假设我们没有使用学习到的位置嵌入），因为...</li><li><a href="https://arxiv.org/abs/2306.17806">Stay on topic with Classifier-Free Guidance</a>: 无分类器指导（Classifier-Free Guidance, CFG）最近在文本到图像生成中出现，作为一种轻量级技术来增强生成内容对提示词的遵循度。在这项工作中，我们证明了 CFG 可以被用于...</li><li><a href="https://web.archive.org/web/20240529183033/https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-2/">VALL-E</a>: VALL-E 是一种神经编解码语言模型，使用源自现成神经音频编解码模型的离散代码，并将 TTS 视为一种条件语言建模任务。VALL-E 展现出了上下文内（in-context）...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1249818695421788201)** (6 messages): 

- **DeepSeek 模型集成挑战**：一名成员询问了关于解释 DeepSeek-LLM-7B 模型及其在 Transformerlens 中的添加。另一名成员确认了难度，提到他们在尝试过程中遇到了短路问题。
- **DeepSeek 模型基于 LLaMA**：一条有用的评论指出 DeepSeek-LLM-7B 模型的架构基于 LLaMA，建议通过一些 hack 手段可以很容易地集成到 Transformerlens 中。他们还建议仔细检查输出概率以避免意外。
- **多语言 Transformer 仓库**：一名成员分享了一个 [GitHub 链接](https://github.com/Butanium/llm-latent-language/blob/1054015066a4fa20386765d72601d03aa7ef5887/utils.py#L334)，该仓库配套论文 "Do Llamas Work in English? On the Latent Language of Multilingual Transformers"。该资源可能为 DeepSeek 模型集成提供见解或方法。

**提到的链接**：<a href="https://github.com/Butanium/llm-latent-language/blob/1054015066a4fa20386765d72601d03aa7ef5887/utils.py#L334">llm-latent-language/utils.py at 1054015066a4fa20386765d72601d03aa7ef5887 · Butanium/llm-latent-language</a>：配套我们论文 "Do Llamas Work in English? On the Latent Language of Multilingual Transformers" 的仓库。- Butanium/llm-latent-language

  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1249817176869965937)** (6 messages): 

- **使用 --apply_chat_template 标志启用聊天模板**：Eleuther 现在支持使用 **--apply_chat_template** 标志对 HF 模型进行聊天模板处理。然而，*此功能默认不开启*。
- **指定停止序列以解决任务问题**：一些用户发现手动指定停止序列有助于解决特定任务的问题。然而，在 `doc_to_choices` 中打乱选项并未如预期般影响模型的回答。
- **Batch API 需要改进，欢迎贡献**：目前在 API 或本地服务器模型中的批处理实现并非最优，特别是对于 **OpenAI Batch API** 的集成。欢迎通过更好的批处理方法来改进此功能的贡献。
- **讨论了实现 Batch API 的步骤**：Batch API 的高层实现包括创建一个 JSONL 文件、通过 API 上传、运行选定模型，并返回运行和文件 ID 以进行状态检查。建议使用 **Async evaluation calls**（异步评估调用）来简化流程。
- **计划添加在 Batch API 结果文件上重新运行指标的工具**：该提案包括添加一个工具，将来自 **OpenAI** 的响应转换为逐样本输出。这将有助于在 harness 中对保存的结果文件重新运行指标。
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1250174056477491361)** (1 messages): 

- **寻找 LTIP 数据集替代方案**：一位用户询问了 Deepmind 用于预训练 **Flamingo 和 GATO** 的 **LTIP 数据集** 的 **开源替代方案**。他们指出 **LTIP 数据集的数据表** 可以在 [Flamingo 论文的附录中](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf) 找到，并提到目前已被撤回的 **LAION 数据集** 曾是一个选项。
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1249906243984953416)** (128 messages🔥🔥): 

- **评估用于装机的 7950x3D 和 GeForce RTX 4090**：一位成员考虑使用 Ryzen 7950x 搭配 2 个 GeForce RTX 4090 进行装机，但对 4090 的尺寸、功耗以及缺乏 NVLink 通信表示担忧。另一位成员推荐使用 7950x3D，因为它具有更大的 L3 缓存且价格增幅较小。
- **用于 Llama-3 推理的 GPU**：成员们讨论了用于 Llama-3 推理的最佳 GPU，在单块 RTX 4090 和双块 3090 设置之间进行了权衡。后者在需要更多 VRAM 的应用中更受青睐，尽管强调了某些型号（如 4060Ti）缺乏 NVLink 以及潜在的 PCI 瓶颈。
- **Triton Kernel 开发中的挑战**：一位新用户分享了他们在基于 Triton 的 Conv2d/Linear 层实现中遇到的问题，发现它们比 cuDNN 对应版本慢，并且在 Kernel 内部调试时感到困难。他们寻求关于 Triton Kernel 内打印变量的资源和方法建议。
- **关于 HPC 设置的 CPU/GPU 配置讨论**：对话深入探讨了各种配置，包括旧款 Threadripper 型号、现代 Epyc CPU 及其 PCIe 通道限制。还讨论了重型 GPU 设置的功耗考虑和冷却需求。
- **GPU 废热的创新利用**：一位成员开玩笑地建议使用 GPU 代替热水浴缸加热器，这引发了关于利用废热造福社区的可持续数据中心的讨论。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.msi.com/Graphics-Card/GeForce-RTX-4090-SUPRIM-LIQUID-24G">MSI GeForce RTX™ 4090 SUPRIM LIQUID 24G</a>：GeForce RTX™ 4090 SUPRIM LIQUID 24G 采用 GPU 液冷和 VRM 风冷设计，并配有坚固的拉丝金属背板提供被动冷却。MSI SUPRIM LIQUID 易于安装...</li><li><a href="https://huggingface.co/Mozilla/Meta-Llama-3-70B-Instruct-llamafile/tree/main">Mozilla/Meta-Llama-3-70B-Instruct-llamafile at main</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1249838457581469830)** (2 messages): 

- **For 循环是必要的**：一位用户强调了在他们的代码中需要 for 循环，简单地表示：“不，你需要执行 for 循环。”
- **在 Triton 中更倾向于非元组语法**：另一位用户指出他们更喜欢 Triton 中 `load_2d` 函数的非元组版本。他们解释道：“在我看来，元组只会增加括号。所以我维持原样。”
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1249868128058605679)** (10 条消息🔥): 

- **结合 torch.compile 的自定义 C++/CUDA**: 一位成员询问与 torch.compile 兼容的自定义 C++/CUDA 算子是否允许全图编译（full graph compilation），以及是否可以通过 torch.export 进行 AOT 编译。另一位成员回答说，它应该允许全图编译，但不完全确定 export 的情况。他们提供了一个参考示例：[Custom CUDA extensions by msaroufim · Pull Request #135 · pytorch/ao](https://github.com/pytorch/ao/pull/135)。
- **PyTorch 中的 HIP kernel**: 一位用户询问是否可以编写 HIP kernel 并在 PyTorch 中使用。另一位成员建议使用 `load_inline` 应该可以正常工作。
- **AWQ 的推理优化**: 一位成员质疑为什么缺乏针对 AWQ 优化的推理 Triton/CUDA kernel，并推测其权重层量化方式的异构性是否构成了挑战。另一位成员引用了 [PyTorch int4 matmul 文档](https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html)作为回应。
- **CUDA 库预热 (warmup)**: 一位用户提到，他们在某些 CUDA 库上遇到了问题，这些库对于 torch 内部使用的某些算法需要一个预热期。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html">函数 at::_weight_int4pack_mm &mdash; PyTorch 主文档</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1-LdJZBzlxiF0Tm-8NfbyFvRJaofdwRgLcycXGmlIpS0/edit">[教程] 自定义 C++ 和 CUDA 算子</a>: 自定义 C++ 和 CUDA 算子。PyTorch 提供了大量的 Tensor 算子库（例如 torch.add, torch.sum 等）。然而，你可能希望向 PyTorch 引入新的自定义算子。本教程...</li><li><a href="https://github.com/pytorch/ao/pull/135">msaroufim 的自定义 CUDA 扩展 · Pull Request #135 · pytorch/ao</a>: 这是 #130 的可合并版本 - 我必须进行一些更新：除非使用 PyTorch 2.4+，否则添加跳过测试；如果 CUDA 不可用，则添加跳过测试；将 ninja 添加到开发依赖项中...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1250151507890798613)** (1 条消息): 

- **Satabios 展示新的模型压缩包**: 一位用户介绍了他们自建的模型压缩和推理包 [Sconce](https://github.com/satabios/sconce)。他们邀请其他成员如果喜欢该项目可以给个 star，并欢迎提出改进建议。
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1249832156151152784)** (1 条消息): 

- **分享 Iron Bound 访谈**: 一位成员发布了 [The Amp Hour 播客访谈](https://theamphour.com/the-amp-hour-84-bunnies-bibelot-bonification/)的链接。该播客集名为 "Bunnies Bibelot Bonification"。
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1249821968941711491)** (2 条消息): 

- **Charles 的 PR 改进了基准测试**: Charles 提交了一个 [PyTorch AO 仓库的 pull request](https://github.com/pytorch/ao/pull/276)，增加了对 Llama 模型基准测试的支持。这旨在为 TorchAO 代码库提供“稳定的评估/基准测试”功能。
- **较大的 N 可能不需要更改**: 一位成员指出，如果 N（样本量）足够大，可能不需要额外的修改。这意味着在样本量较大时，对结果的影响可能微乎其微。

**提到的链接**: <a href="https://github.com/pytorch/ao/pull/276">HDCharles 将 Llama 添加到 TorchAO · Pull Request #276 · pytorch/ao</a>: 摘要：此 PR 为 torchao 代码库增加了 Llama 模型的稳定评估/基准测试功能。模型相关内容位于 torchao/_models/llama，评估部分已移至 _models/_eval.py...

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1250022396031795200)** (42 条消息🔥): 

```html
- **ThunderKitten 性能令人失望**：成员们讨论了 **ThunderKitten** 的性能，指出在 **A100** 上进行基础 matmul 时，它仅达到约 75 TFLOPS，而 **cuBLAS** 则达到约 400 TFLOPS。一种解释是 ThunderKitten 可能过度关注 **TMA**，导致非 TMA 路径受到严重的 L1/load-store 限制。
- **ThunderKitten 中的 C++20**：对话强调了 **ThunderKitten** 需要 C++20，尽管该语言在处理 concepts 方面具有优势，但一些成员认为这很繁琐。关于是否可以使用 C++17 实现类似功能存在争论，尽管那会导致更复杂且可读性更差的模板代码。
- **FP8 训练稳定性担忧**：一位成员提到，尽管 FP8 训练被认为能提供性能提升，但由于稳定性问题，许多团队仍倾向于使用 **FP16**。他们指出 **FP8** 尚未被完全理解或稳定，使得 **FP16** 目前是更可预测的训练选择。
- **使用 Thrust 进行逐元素变换**：一位成员询问了如何使用 **Thrust** 优化 **Hopper/Blackwell** GPU 上的逐元素变换（elementwise transformations）性能。他们寻求关于利用对齐数据进行更高效计算的建议，并比较了包括 **manual TMA** 在内的不同方法的性能。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://devblogs.microsoft.com/cppblog/cpp23-deducing-this/#crtp)">C++23&#039;s Deducing this: what it is, why it is, how to use it - C++ Team Blog</a>: 了解 C++23 的 Deducing this 特性如何帮助改进您的代码。</li><li><a href="https://www.modernescpp.com/index.php/c23-deducing-this/).">C++23: Deducing This &#8211; MC++ BLOG</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1249848448711262268)** (1 条消息): 

- **分享 1-bit LLMs 训练指南**：分享了一个训练 **1-bit LLMs** 的重要资源，其中包括技巧、代码和常见问题解答（FAQs）。请查看 [Microsoft 的 Unilm GitHub](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf) 上的完整指南。

**提到的链接**: <a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>: 跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm

  

---


### **CUDA MODE ▷ #[sparsity](https://discord.com/channels/1189498204333543425/1247663759434977453/)** (1 条消息): 

satabios: 模型压缩/推理包：https://github.com/satabios/sconce

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1249831070916939997)** (134 条消息🔥🔥): 

- **辩论 Mojo 的最佳并发模型**：关于结构化并发（使用 async/await 等原语）应该是 Mojo 标准库的一部分还是存在于生态系统中的广泛讨论。**“Rust 是第一个实现完整 async 到 io_uring 的语言”**，这引发了关于如何在避免危险陷阱的同时实现最佳性能的辩论。

- **Mojo 的异步能力和 GPU 支持**：对话围绕 Mojo 目前仅限于 CPU 的局限性、预期的 GPU 支持以及 TPU 实现的潜力展开。一位用户表达了担忧：*“如果 Mojo 比 Python 快 3 倍，但不能在便宜 6-8 倍的 TPU 上运行……”*，强调了异构硬件兼容性的重要性。

- **Modular 的方法和资金**：成员们分享了关于 Modular 资金和开发策略的见解，指出尽管获得了 1.3 亿美元的融资，**“Mojo 实际上是在与 C, C++, Rust 和 CUDA 竞争，而不是 Python”**。还强调了未来支持各种硬件架构的承诺。

- **AI 并发示例和结构化范式**：在 Mojo 长期目标的背景下，讨论了不同 AI 并发范式（如结构化并发）的有效性。有人提出了编译器支持对于分解任务的重要性，以确保跨设备的**“编程模型将保持一致”**。

- **Mojo 的潜力和社区情绪**：用户讨论了对 Mojo 未来能力以及支持 TPU 等额外硬件的时间表的希望和疑虑。一位用户提醒道：**“请记住 Mojo 及其基础设施还相当新”**，呼吁在生态系统演进过程中保持耐心。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/.">Modular: Accelerating the Pace of AI</a>：Modular Accelerated Xecution (MAX) 平台是全球唯一能为您的 AI 工作负载解锁性能、可编程性和可移植性的平台。</li><li><a href="https://mlir.llvm.org/docs/Dialects/AsyncDialect/">'async' Dialect - MLIR</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1800580309156847626>
  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1249803277030395937)** (21 条消息🔥): 

- **Xoshiro PRNG 加速模拟**：一位成员将 xoshiro PRNG 移植到了 Mojo，在笔记本电脑上实现了 *64 Gbps* 的速度，并利用 SIMD 和 4 个并行流达到了 *180 Gbps*。他们询问是否有计划在标准库中增加更多的生成器。讨论了 [Numerics for Mojo Repo](https://github.com/thk686/numojo)。
- **NuMojo 和数学库**：另一位成员提到一个正在进行的将 NumPy 库移植到 Mojo 的项目，并链接了 [NuMojo 仓库](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo)。此外还有关于高能 Python 数学移植的个人工作。
- **Mojo 中的循环向量化**：关于在 Mojo 中有效进行循环向量化的讨论，重点介绍了通过 SIMD 创建递增序列的 `math.iota` 函数。性能测试显示，向量化循环的执行速度（0.032032 秒）快于普通循环（0.059314 秒）。
- **性能测试的预热循环**：建议在开始性能测试前加入预热循环以确保准确性。有人指出，这在 Python 等解释型语言中可能更相关，而在 Mojo 这种编译型语言中重要性较低。
- **编译时解释器推测**：一位成员对 Mojo 潜在的编译时解释器表示好奇，该解释器可能像 C++ 的 `constexpr` 一样防止未定义行为。他们强调了这对于测试使用 unsafe 特性的代码安全性非常重要。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/max/api/mojo/graph/quantization/quantization_encoding/QuantizationEncoding">QuantizationEncoding | Modular 文档</a>：描述了可量化数据类型的编码。</li><li><a href="https://docs.modular.com/mojo/roadmap#no-python-style-generator-functions">Mojo🔥 路线图与注意事项 | Modular 文档</a>：Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://docs.modular.com/mojo/stdlib/math/math/iota">iota | Modular 文档</a>：iotatype Int -&gt; SIMD[$0, $1]</li><li><a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo">GitHub - Mojo-Numerics-and-Algorithms-group/NuMojo</a>：一个用于 Mojo 编程语言的数值计算库。</li><li><a href="https://github.com/thk686/numojo">GitHub - thk686/numojo: Numerics for Mojo</a>：Mojo 数值计算库。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1249903046075482145)** (9 条消息🔥): 

- **博客文章中的链接失效**：一位成员报告说 [博客文章](https://www.modular.com/blog/max-24-4-introducing-quantization-apis-and-max-on-macos) 上的 [链接](https://docs.modular.com/max/reference/mojo/graph/quantization/) 导致 404 错误。建议将其调整为 [此链接](https://docs.modular.com/max/api/mojo/graph/quantization/)。

- **关于 TPU 兼容性的咨询**：一位用户咨询了将 Mojo (MAX engine) 与 Google TPU 加速器配合运行的可能性。他们强调了 TPU 相比 A100x8 加速器的性价比。

- **MAX Engine 路线图和 TPU 开发**：据分享，Nvidia GPU 支持已列入夏季发布的路线图。一位用户对开发 TPU 实现的可行性表示关注，理由是潜在的时间和成本优势。

- **探索 TPU 资源**：一位成员引用了 [OpenXLA GitHub 仓库](https://github.com/openxla/xla) 作为开发 TPU 实现的潜在资源。他们考虑为此利用现有的机器学习编译器功能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/max/api/mojo/graph/quantization/quantization_encoding/QuantizationEncoding">QuantizationEncoding | Modular 文档</a>：描述了可量化数据类型的编码。</li><li><a href="https://github.com/openxla/xla">GitHub - openxla/xla</a>：一个用于 GPU、CPU 和 ML 加速器的机器学习编译器。</li><li><a href="https://docs.modular.com/max/api/mojo/graph/quantization/">quantization | Modular 文档</a>：用于量化图张量（graph tensors）的 API。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1249821272917934160)** (13 条消息🔥): 

- **关于 RAII 在 Mojo 内存管理中作用的辩论**：讨论集中在 Mojo 中是否需要上下文管理器（context manager），一名成员表示 “RAII 清理” 可以处理文件清理。另一名成员指出考虑到 Mojo 的内存模型，可能存在潜在问题，并对 UnsafePointers 和生命周期（lifetimes）发表了评论。
- **UnsafePointers 与所有权（Ownership）**：成员们辩论了 UnsafePointers 在 Mojo 中的角色，强调由于这些指针具有非所有权性质且缺乏生命周期，因此不使用 RAII。有人建议 Mojo 可以采用类似 Rust 的 `Box` 那样的所有权指针类型。
- **Mojo nightly 编译器更新发布**：发布了新的 nightly Mojo 编译器，版本号为 `2024.6.1105`。关键更新包括移除了 `SliceNew` 和 `SIMD.splat`，并在 `tempfile` 模块中实现了 `NamedTemporaryFile`。提供了 [原始差异 (Raw diff)](https://github.com/modularml/mojo/compare/f8c229b856795f2782e77db6d125fda1f8d753d4...76eda306af929d9576d7190a7f8f3aa1df83baf6) 和 [当前变更日志 (changelog)](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1249805037304025230)** (157 条消息🔥🔥): 

- **苹果新的 “个人智能” 系统登场**：讨论重点介绍了苹果推出的 “Apple Intelligence”，其特点是将 ChatGPT-4o 免费集成到 Siri 和系统范围的写作工具中。在 [Reddit 帖子](https://www.reddit.com/r/ChatGPT/s/KrhcqUpEuq) 中有详细介绍，该系统专注于用户隐私，且在不损害个人上下文的情况下运行。

- **Rabbit R1 设备：是骗局吗？**：针对 Rabbit R1 展开了辩论，一些人表示它可能是一个与加密货币历史有关的骗局，Coffeezilla 的视频中对此进行了进一步讨论。用户对其实用性进行了辩论，并建议安装 Android 以获得更好的可用性。

- **Perplexity Pro 访问权限与功能**：关于访问 Perplexity Pages 以及面向 Pro 用户的推广存在困惑。目前，页面生成功能仅限 Pro 用户使用，且仅在桌面/网页端设置下工作。

- **使用问题及学术准确性反馈**：用户报告了 Perplexity 在引用学术内容时的准确性问题，更倾向于使用 Google 的 NotebookLM 来获得可靠的来源和更好的学术结果。用户对 Perplexity Pages 缺乏编辑能力和对来源的控制表示担忧。

- **请求集成 SAML SSO**：一位用户请求协助将 Perplexity.ai 集成到其现有的基于 SAML 的单点登录（Single Sign-On）基础设施中，寻求特定的服务提供商元数据和指令。该请求强调了对企业集成更好的支持和文档的需求。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/nwmsrocks-northwest-motorsport-pnw-pacific-northwest-toyota-gif-20681791">Nwmsrocks Northwest Motorsport GIF - Nwmsrocks Northwest Motorsport Pnw - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/you-know-it-wink-the-office-michael-scott-steve-carell-gif-17547389">You Know It Wink GIF - You Know It Wink The Office - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/ChatGPT/s/KrhcqUpEuq">Reddit - 深入探索任何事物</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/Fr-SAMLSSO-bei-Cf6NgplsT8.QRLumYk3BNA">对于 perplexity.ai 的 SAML-SSO，我需要 'Service Provider Metadata'...</a>: 根据提供的搜索结果，我找不到关于 perplexity.ai 的 SAML-SSO 配置的具体信息。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1249814174788489279)** (9 条消息🔥): 

- **iOS 18 包含海量功能**：iOS 18 中重新设计的 Photos 应用使用先进的 **machine learning algorithms** 自动根据人物、地点和事件对照片进行组织和分类。前所未有的自定义选项和自定义表情符号设计为消息传递增添了趣味和创意，是其中的亮点功能 ([来源](https://www.perplexity.ai/page/What-Was-Introduced-9M8PxU85Tg2kA.6ubApgJw))。

- **WWDC 2024 以 AI 为核心**：Apple 的 WWDC 2024 主旨演讲介绍了新的软件版本，强调了跨设备的 **AI integration** 和软件增强。亮点包括 iOS 18 的 Photos 应用重新设计、iPadOS 18 的生产力增强以及 **macOS 15** 的统一任务管理 ([来源](https://www.perplexity.ai/page/What-Was-Introduced-0laYmO7vS2mGryU354PYaA))。

- **Image-er 是正确读音**：“Imgur”的正确读音是“image-er”，这一点已得到 **Imgur staff** 的确认，并有多个 Reddit 讨论支持，尽管用户有各种不同的读音 ([来源](https://www.perplexity.ai/search/Whats-the-correct-2XyomPRVT0a3uF9C_ZDq0g))。

- **MARS5 树立了 TTS 新标准**：由 CAMB.AI 开发的 **MARS5 TTS model** 使用两阶段流水线生成高质量的韵律和情感，适用于配音和翻译等应用。它擅长在体育解说和动漫等各种场景中应用细腻的情感表演 ([来源](https://www.perplexity.ai/search/MARS-5-TTS-mLBhVSs_RRWWnypF.Aiv2Q))。
  
- **Lenny 关于产品管理见解的通讯**：Lenny 提供了关于 **building products and growth strategies** 的有价值见解，涵盖了成功的转型和为功能团队 PM 辩护等主题。订阅者可以阅读深度文章并收听他的播客 ([来源](https://www.lennysnewsletter.com/p/how-to-use-perplexity-in-your-pm))。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/hhX9EKtInok">YouTube</a>：未找到描述</li><li><a href="https://www.lennysnewsletter.com/p/how-to-use-perplexity-in-your-pm">如何在 PM 工作中使用 Perplexity</a>：27 个产品经理目前使用 Perplexity 的示例（包含实际 prompt）</li><li><a href="https://www.perplexity.ai/search/What-is-Perplexity-OC0LtdykRoC7dmzTcHXWVw#1">什么是 Perplexity AI？</a>：Perplexity AI 是一款由 AI 驱动的搜索引擎，旨在通过利用大语言模型为用户的查询提供简洁、准确的答案...</li><li><a href="https://www.perplexity.ai/page/What-Was-Introduced-0laYmO7vS2mGryU354PYaA">WWDC 2024 推出了什么？</a>：Apple 的 WWDC 2024 主旨演讲揭晓了一波 AI 驱动的功能和软件更新，旨在跨平台提供更智能、更个性化的体验...</li><li><a href="https://www.perplexity.ai/search/MARS-5-TTS-mLBhVSs_RRWWnypF.Aiv2Q">MARS 5，来自 Camb.ai 的 TTS 模型</a>：MARS5 是由 CAMB.AI 开发的文本转语音 (TTS) 模型。它旨在生成具有高质量韵律和情感的语音，使其适用于...</li><li><a href="https://www.perplexity.ai/page/What-Was-Introduced-9M8PxU85Tg2kA.6ubApgJw">WWDC 2024 推出了什么？</a>：Apple 的 WWDC 2024 主旨演讲揭晓了一系列令人兴奋的更新和功能，重点强调了人工智能集成和软件...</li><li><a href="https://www.perplexity.ai/search/Revise-into-a-t4taOXiIRU.nnMM4GwkMzQ#0">修改为更稳健的文章。使用包含的 &lt;scratchpad-think&gt;...</a>：这是我尝试使用提供的 scratchpad 框架将文章修改为更稳健版本的尝试：&lt;scratchpad&gt; 来自...的关键信息</li><li><a href="https://www.perplexity.ai/search/How-does-singing-3icDgrnMSAixgVGK9r3TbQ">声调语言中的唱歌是如何运作的？</a>：在中文、越南语和泰语等声调语言中唱歌面临着独特的挑战，因为在说话中传达意义的声调需要与...</li><li><a href="https://www.perplexity.ai/search/Whats-the-correct-2XyomPRVT0a3uF9C_ZDq0g">Imgur 的正确读音是什么？</a>：根据搜索结果，“Imgur”的正确读音是“image-er”或 /ˈɪm.ɪdʒ.ər/。虽然许多人最初将其读作“im-grr”或...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1249840516997320827)** (6 messages): 

- **在自定义 GPTs 中集成 Perplexity API 遇到困难**：一名成员对 Chat GPT 为自定义 GPTs 提供的网页搜索功能表示失望，寻求帮助以集成 Serper、Tavily 或 Perplexity API 等替代方案。他们分享了 [Perplexity API 文档链接](https://docs.perplexity.ai/discuss/65edc94038fa40001045873c)，并询问当 GPT 无法找到答案时如何实现该功能的建议。
- **更新模型名称可能有所帮助**：另一位成员建议更新模型名称（例如，从 pplx-70b-online 切换到 llama-3-sonar-large-32k-online）可能会解决集成问题，但承认仍需进一步审查。
- **API key 泄露在共享代码中**：有成员分享了使用 Perplexity API 的代码，但另一位成员指出，出于安全考虑，应删除泄露的 API key 并创建一个新 key。

**提到的链接**：<a href="https://docs.perplexity.ai/discuss/65edc94038fa40001045873c">在自定义 GPT 中使用 Perplexity API</a>：未找到描述

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1249800442808893451)** (85 messages🔥🔥): 

- **寻求本地解析 PDF 的工具**：一位开发者正在寻找在本地解析 PDF 中结构化表单的工具，寻求推荐 **LLMs 或脚本** 来提取字段和答案。有人建议使用 **Langchain** 集成本地 LLMs，以最少的配置从 PDF 中提取字段。

- **WebUI 与 LMStudio 的问题**：一位成员希望通过 WebUI 从另一台 PC 与 LMStudio 交互，但没有找到官方包。另一位成员建议使用内置的 **llama.cpp server** 获取简单界面，并推荐查看 *[text-generation-webui](https://github.com/oobabooga/text-generation-webui/)* 进行基于 Web 的交互。

- **对 SB 1047 和开源 AI 的担忧**：关于 **SB 1047** 及其对开源 AI 影响的讨论非常激烈。该法案被认为阴险地旨在将 AI 发展限制在少数几家公司手中，摧毁开源 AI，并要求模型制作者承担无限期的责任。链接的 [推文](https://x.com/dan_jeffries1/status/1794740447052525609?s=46) 提供了更多细节。

- **针对 RAM 和 GPU 不足的适配**：一位用户讨论了升级 RAM 和 GPU 以运行像 70B 这样的大型 AI 模型。共识是购买一块至少拥有 24GB VRAM 的 **更强大的 GPU**，因为他们目前的 6700 XT 缺乏 ROCm 支持，且使用 **OpenCL** 的性能会慢得多。

- **比较用于 AI 任务的 GPU**：成员们讨论了 GPU 推荐，建议预算有限的配置可以使用像 **P40** 这样的旧服务器卡。为了获得更好的性能，建议选择 **7900XT(X)** 和 **二手 3090**，并指出对于 AMD 显卡，**ROCm** 比 OpenCL 快得多。[ROCm 信息](https://www.amd.com/en/products/software/rocm.html)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.useanything.com/setup/llm-configuration/local/lmstudio">LMStudio LLM ~ AnythingLLM</a>：LMStudio 是一个流行的用户界面、API 和 LLM 引擎，允许你从 HuggingFace 下载任何 GGUF 模型并在 CPU 或 GPU 上运行。</li><li><a href="https://x.com/dan_jeffries1/status/1794740447052525609?s=46">Daniel Jeffries (@Dan_Jeffries1) 的推文</a>：我花了几小时听 Dan Hendyrcks 的发言，他是 SB 1047（又称加州 AI 控制与中心化法案）背后的非营利 AI 安全组织的负责人。我觉得他很有魅力、稳重且聪明...</li><li><a href="https://github.com/oobabooga/text-generation-webui/">GitHub - oobabooga/text-generation-webui: 用于大语言模型的 Gradio Web UI。支持 transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。</a>：用于大语言模型的 Gradio Web UI。支持 transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。- oobabooga/text-generation-webui
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1249871256224792677)** (5 条消息): 

- **Boptruth-NeuralMonarch-7B 亮相**：一位成员分享了他们使用 [LazyMergekit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing) 成功合并的模型 **Boptruth-NeuralMonarch-7B**，并敦促其他人尝试。该模型在配合 Alpaca 聊天模板时表现最佳，可以在 [Huggingface](https://huggingface.co/theprint/Boptruth-NeuralMonarch-7B) 上找到。
- **驯服 Qwen2 72B 模型**：另一位成员报告成功在 128MB M3 Max 上运行 **Dolphin 2.9.2 Qwen2 72B Q8** 模型。他们评论说，尽管过去在处理大型多分片模型时遇到过困难，但这个模型的表现相当不错。
- **Llama3 微调版令人印象深刻**：一位成员正在测试 **Llama3-FiditeNemini-70B-Source.i1-Q6_K.gguf**，发现它优于基础版 Llama3 instruct，并称赞其文笔巧妙。他们提供了各种[量化版本](https://huggingface.co/mradermacher/Llama3-FiditeNemini-70B-Source-i1-GGUF)的链接及使用资源。
- **关于 Prompt 问题的提问**：有人提问该 Llama3 微调版是否会出现回复中随机大喊大叫的常见问题。最初的测试者确认他们尚未观察到此问题，并将其用于角色扮演活动，结果令人满意。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mradermacher/Llama3-FiditeNemini-70B-Source-i1-GGUF">mradermacher/Llama3-FiditeNemini-70B-Source-i1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/theprint/Boptruth-NeuralMonarch-7B">theprint/Boptruth-NeuralMonarch-7B · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1249936796905640036)** (3 条消息): 

- **虚拟内存与物理内存的混淆**：一位用户询问如何解决内存相关问题，另一位成员澄清说，看到的巨大“提交大小 (commit size)”是虚拟内存预留，而非实际物理内存占用。他们强调，在完全卸载（offloading）到 GPU 的情况下，RAM 使用率保持在较低水平，并举例说明 10GB 的提交大小仅导致 160MB 的实际（私有）占用。
- **页面错误与 GPU VRAM 使用**：进一步解释说明，由于在物理 RAM 中找不到数据，页面错误（page faults）会增加，通常这会导致磁盘读取。然而，通过 GPU 映射内存，Windows 可以将其重定向到 GPU VRAM，或使用直接访问 VRAM 的方式来完全避免页面错误。
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1249815556249616466)** (38 条消息🔥): 

- **P40 温度与散热讨论**：成员们讨论了 P40 显卡的温度和散热解决方案，并指出 *“我找到了 P40 的‘产品规格’，上面说气流方向并不重要”*，热节流（thermal-throttling）温度为 90C。分享了多个 [Aliexpress 风扇](https://aliexpress.ru/item/1005002259578351.html) 链接作为潜在的散热方案。 

- **8700g 性能引起关注**：一位成员提到测试了 8700g，它 *“现在在 LM Studio 中可以达到 11 toks，并能寻址 32GB RAM”*，认为它是运行大型模型的一个经济实惠的选择。 

- **对 LM Studio 多 GPU 性能的担忧**：有用户抱怨 LM Studio 无法高效处理跨多个 GPU 的模型，*“瓶颈在于 PCIe 吞吐量”*，且在部分 GPU 卸载场景下表现不佳。与 *llama.cpp* 和 *ollama* 等其他工具的对比凸显了这些平台更好的多 GPU 支持。

- **Tesla V100 兼容性咨询**：一位用户询问在 LM Studio 上运行 Tesla V100 的情况，表达了对潜在兼容性或驱动问题的担忧。另一位成员分享了一个[链接](https://github.com/l4rz/running-nvidia-sxm-gpus-in-consumer-pcs)以供进一步阅读。

- **LM Studio 的操作系统偏好**：成员们辩论了 LM Studio 在 Windows 还是 Linux 上表现更好，一位成员指出 Windows *“开箱即用”*，而 Linux 需要特定的发行版和驱动，且处于“测试版”。另一位用户评论道：*“两个操作系统上的 GPU 干扰大致相同，Linux 上的 CPU 干扰处理可能更快”*。
  

---

### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1249803467208265788)** (1 messages): 

- **AutogenStudio token 限制 Bug 已修复**：一位成员遇到了 AutogenStudio 与 **TheBloke/Llama-2-7B-Chat-GGUF** 模型配合使用时，生成 token 被限制在 2 个的问题。该问题已通过 [GitHub](https://github.com/microsoft/autogen/issues/2050) 上分享的涉及 `max_tokens` 参数的临时解决方案得到解决。

**提到的链接**：<a href="https://github.com/microsoft/autogen/issues/2050">[Bug]: [autogenstudio] agent llm send max_tokens: null · Issue #2050 · microsoft/autogen</a>：描述该 Bug：当 `max_tokens` 参数为 `None` 时，Agent 发送的 `/v1/chat/completions` 帧中 `max_tokens` 为 `null`。在这种情况下，LLM 无法理解并在第二个 token 后停止。

  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1249805942632222752)** (11 messages🔥): 

- **不同工具的 ROCm 实现**：关于在 **LM Studio** 中使用 **ROCm** 并将其支持扩展到 **Auto1111** 或 **Comfy** 的可行性讨论正在进行中。一位成员指出，“A1111 上的 ROCm 实现非常 hacky（粗糙）。”
- **Stable.cpp 和 Zluda 让 AMD 挂钩 CUDA**：成员们讨论了 **stable.cpp 项目** 和一个名为 **Zluda** 的工具的潜力，后者可能允许 AMD 挂钩到 CUDA。一位成员将这种集成描述为一项重大的“苦差事”和“真正的挑战”，目前尚未解决。
- **GPU 加速工具对比**：在讨论不同 GPU 时，成员们分享了使用 **CUDA**、**OpenCL** 甚至 **Metal** 构建 GPU 加速应用程序的经验。
- **对 SD.next 界面的不满**：尽管一些用户在 **automatic1111** 上使用 **Zluda** 获得了更好的性能，但人们对 **SD.next 界面** 明显反感。一位成员表示：“我就是讨厌 SD.next 的界面。”
- **LLMStudio 的最佳操作系统**：关于在 **Windows 还是 Linux (Ubuntu)** 上运行 **LLMStudio** 更好存在争论。各种用户偏好和技术考量（如 GPU 兼容性和软件支持）塑造了这些讨论。
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1249951378823188590)** (1 messages): 

- **尊重频道礼仪**：一位成员对另一位用户在多个频道发布相同问题的行为表示不满。他们强调，在多个频道刷屏同一个查询被视为“缺乏礼仪”。
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1249800400857600184)** (95 messages🔥🔥): 

- **苹果进入 AI 领域加剧竞争**：成员们对苹果最近进入 AI 领域感到兴奋，一位成员指出：“既然苹果已经进入该领域，竞争将变得异常激烈。”另一位成员评论道：“凭借苹果的生态系统，集成 AI 将会非常棒。”
- **新款苹果设备的 AI 功能**：关于哪些苹果设备将支持新的 AI 功能存在讨论，一位评论者表示：“就 iPhone 而言，端侧 AI 需要 8GB RAM 以及 A17 Pro 或更高版本。”这引起了使用旧设备用户的担忧，因为他们可能不得不折价贴换或升级。
- **苹果 vs 云计算之争**：围绕苹果同时使用端侧 AI 功能和云计算展开了激烈的辩论，一位成员批评道：“所以他们在引入 PCC 概念后立即将其抛之脑后。”其他人则讨论了云计算在安全处理更复杂任务方面的必要性。
- **对 AI 和数据隐私的担忧**：存在一种强烈的反对 AI 滥用数据的观点，一位成员认为：“这些数据 100% 会被滥用……它会被出售，你的隐私会被侵犯，你的安全会受到威胁。”另一位成员指出了技术爱好者中既反云又反本地部署方案的矛盾现象。
- **WWDC 2024 上的苹果集成 AI**：成员们正在分享关于 WWDC 2024 上推出的新“Apple Intelligence”的兴奋点和资源，一位成员分享了 [Apple Foundation Models 概览](https://machinelearning.apple.com/research/introducing-apple-foundation-models) 的链接。这展示了为苹果操作系统生态系统中各种用户任务量身定制的多个生成式模型。

**提到的链接**：<a href="https://machinelearning.apple.com/research/introducing-apple-foundation-models">Introducing Apple’s On-Device and Server Foundation Models</a>：在 2024 年全球开发者大会上，我们推出了 Apple Intelligence，这是一个深度集成的个人智能系统……

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1249801913398984885)** (22 messages🔥): 

- **新版 Voice Mode 迟迟未到**：用户对 OpenAI 推迟发布新版语音模式表示沮丧，指出“未来几周内推出”的承诺已经拖得太久。有人对时间线的不确定性发表了评论，暗示考虑到他们的资金投入，这感觉像是空头支票。

- **ChatGPT 更新带来的兴奋感消退**：一位用户提到在应用商店看到 ChatGPT 更新时很兴奋，但在文本生成过程中遇到了卡顿问题。另一位用户指出，由于需要频繁刷新，再加上使用限制，导致浪费了大量的 Prompt。

- **对 GPT Store 政策违规的困惑**：一位用户报告称，尽管最近没有进行任何更改，但由于所谓的政策违规，无法在 GPT Store 中编辑或发布 GPTs。另一位用户敦促保持耐心，但对 OpenAI 缺乏有效的客户支持表示遗憾。

- **GPTs 与互联网访问**：关于 GPT 联网能力的问题得到了肯定的回答，即 GPTs 可以调用外部 API。后续建议用户在配置 GPTs 时查看“Capabilities”部分以获取更多信息。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1250052401180971029)** (10 messages🔥): 

- **时区导致 GPT Completions 出现混乱**：一位用户提出了在使用 Completions API 处理全球提醒设置时的时区处理问题。他们表示，尽管添加了时区上下文，返回的时间通常仍然是错误的。

- **将 UTC 转换为用户所在时区**：另一位成员建议在 GPT 之外处理时区转换。他们推荐使用 Function Call（函数调用）将 UTC 时间转换为用户的本地时区，以避免浪费 Token 和产生 Hallucination（幻觉）风险。

- **合成数据可能有所帮助**：同一位成员提到，使用合成数据对模型进行 Fine-tuning（微调）以遵循 ISO 8601 标准可能带来好处。这种方法被分享为过去处理日期时间转换的一个成功策略。

- **使用 Google 主日历**：为了获取时区上下文，用户提到他们使用用户的 Google Primary Calendar，旨在通过让模型以 UTC 形式提供决策来保持一致性。

- **关于一致性的建议**：该成员指出，虽然 GPT-4 表现更好，但 GPT-3.5 在处理时间戳方面并不可靠。他们建议将决策维持在 UTC 格式以获得更好的一致性。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1250052401180971029)** (10 messages🔥): 

- **Completions API 中的时区挑战**：用户 nav_archer_23316 提出了 **Completions API** 以 UTC 格式错误返回时间戳的问题，导致不同时区的提醒不匹配。他们正在考虑在聊天历史中添加时区上下文和时间戳来解决此问题。
- **使用库转换时间戳**：用户 zaki_1052 建议让 GPT 以 UTC 格式返回时间，然后使用 *Library*（库）将其转换为用户的时区。他们强调通过使用函数调用进行转换来最小化 Token 使用并防止 Hallucination。
- **时间管理方面的 GPT 对比**：nav_archer_23316 指出，GPT-4 在管理时间戳方面的表现明显优于 GPT-3.5，后者在处理此类任务时被描述为“非常糟糕”。
- **一致的 UTC 模型决策**：为了确保一致性，nav_archer_23316 旨在让模型始终以 UTC 格式给出决策。这一决定是他们更有效地处理时区差异方法的一部分。
- **使用 Google 日历获取时区上下文**：为了保持准确的时区，nav_archer_23316 提到利用用户的 Google Primary Calendar 来获取必要的时区上下文。
  

---

### **OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1250050588763226206)** (3 条消息): 

- **通过 Hana AI 提升团队效率**：想象一下，在 Google Chat 上拥有一位无所不知、永不遗忘、全能专家、24/7 全天候在线、不知疲倦、礼貌且乐于助人的 AI 团队成员。Hana AI 机器人旨在通过无缝集成到 Google Chat 中来“增强您的团队”，从而提升生产力和管理能力。
- **利用 Hana AI 释放潜力**：详细了解 Hana AI 及其如何简化经理和高管的日常任务，从而让他们能够专注于真正重要的事情。该机器人承诺通过其多功能特性减轻负担并提高生产力。
- **免费体验 Hana AI**：Hana AI 提供永久免费计划，并寻求用户对其产品进行试用和反馈。立即通过[此处](https://hana.hanabitech.com)注册，免费开始使用 Hana AI 提升您的团队生产力。

**提及的链接**：<a href="https://hana.hanabitech.com">Hana：您的 AI 驱动型 Google Chat 助手</a>：通过 Hana 提升团队生产力，这是由 Hanabi Technologies 开发的 AI 驱动助手，专为与 Google Chat 无缝集成而设计。

  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1249801940615827567)** (71 条消息🔥🔥): 

- **寻求最佳医学诊断模型**：一位用户询问了用于医学诊断的最佳大语言模型 (LLM)，但回复中未提到具体的模型。
  
- **搜索编程书籍数据集**：一位用户询问是否有编程书籍和文章的数据集，有人分享了一个包含某些文本的 GitHub 仓库 [amephraim/nlp](https://github.com/amephraim/nlp/tree/master/texts)，尽管其中包含《哈利·波特》书籍。

- **safetensors 文件的技术问题**：一位用户在使用 Civitai 的 safetensors 文件时遇到问题，在使用 `diffusers.StableDiffusionPipeline.from_single_file()` 时报错 `TypeError: argument of type 'NoneType' is not iterable`。

- **对 AI 管制立法的担忧**：一位用户分享了一个批评加州 AI 管制与中心化法案（California AI Control and Centralization Bill）的 Twitter 线程，表达了对该法案旨在限制开源 AI 发展并对模型制作者施加严厉责任的担忧。

- **模型的 Checkpoint 使用**：有一段关于使用训练会话中的 checkpoint 的对话，澄清了每个 checkpoint 都可以作为独立模型加载，并且格式（例如 `.pt` 或 safetensors）可以根据需要进行转换。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/dan_jeffries1/status/1794740447052525609?s=46">来自 Daniel Jeffries (@Dan_Jeffries1) 的推文</a>：我花了几个小时听 Dan Hendyrcks 的发言，他是 SB 1047（即加州 AI 管制与中心化法案）背后的非营利 AI 安全组织的负责人。我觉得他很有魅力、稳重、聪明...</li><li><a href="https://huggingface.co/spaces/vinthony/SadTalker">SadTalker - 由 vinthony 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/nroggendorff/dolphin">Dolphin - 由 nroggendorff 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/amephraim/nlp/tree/master/texts">master 分支下的 nlp/texts · amephraim/nlp</a>：通过在 GitHub 上创建账号来为 amephraim/nlp 的开发做出贡献。</li><li><a href="https://github.com/0xPlaygrounds/rig">GitHub - 0xPlaygrounds/rig: 一个用于开发 Rust 驱动的 LLM 应用程序的库。</a>：一个用于开发 Rust 驱动的 LLM 应用程序的库。 - 0xPlaygrounds/rig</li><li><a href="https://civitai.com/api/download/models/348913?type=Model&format=SafeTensor&size=full&fp=fp16&token=urtoken>">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2406.05587">创意已离开聊天室：语言模型去偏见的代价</a>：大语言模型 (LLM) 彻底改变了自然语言处理，但可能会表现出偏见并可能生成有毒内容。虽然像来自人类反馈的强化学习 (RLHF) 等对齐技术...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1250060952309600257)** (4 messages): 

- **黑白漫画模型展示多种艺术风格**：一位用户在 HuggingFace 上分享了 [B&W Manga 模型](https://huggingface.co/alvdansen/BandW-Manga)，展示了诸如“*穿水手服皱眉的男孩*”和“*戴花环的女孩*”等插画。其他人对该模型表示赞赏，称赞作品“可爱”，并用爱心表情符号表达了喜爱。

- **Supermemory GitHub 项目旨在帮助健忘的用户**：一名成员发布了 [GitHub 上的 Supermemory 项目](https://github.com/Dhravya/supermemory)链接。该项目旨在成为“书签版的 ChatGPT”，允许用户通过 Chrome extension 导入推文或使用插件保存网页内容。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/alvdansen/BandW-Manga">alvdansen/BandW-Manga · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/Dhravya/supermemory">GitHub - Dhravya/supermemory: Build your own second brain with supermemory. It&#39;s a ChatGPT for your bookmarks. Import tweets or save websites and content using the chrome extension.</a>：使用 supermemory 构建你自己的第二大脑。它是书签版的 ChatGPT。通过 Chrome extension 导入推文或保存网站及内容。 - Dhravya/supermemory
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1249827785577595051)** (9 messages🔥): 

- **SoteDiffusion Wuerstchen3 上线**：分享了一个 Würstchen V3 的 anime finetune 版本，该版本基于 6M 图像并经过 3 个 epochs 训练。关于 API 使用，他们提供了 [Fal.AI 文档](https://fal.ai/models/fal-ai/stable-cascade/sote-diffusion)的链接。
- **在 Hugging Face Spaces 上与多个模型聊天**：介绍了 [Chat With 'Em](https://huggingface.co/spaces/as-cle-bert/chat-with-em)，这是一个可自定义的聊天模型空间。用户可以通过提供 API key，在 Claude、GPT-3.5、GPT-4 和 Llama-3 系列等模型之间切换。
- **预测一级方程式赛车 (Formula 1) 单圈时间**：分享了一个使用历史遥测数据预测单圈时间的项目，并在详细的 [Kaggle notebook](https://www.kaggle.com/code/lucasdraichi/hamilton-lap-time-prediction) 中进行了展示。他们正在寻求社区的反馈。
- **CAMB AI 发布 MARS5 TTS 模型**：宣布发布 MARS5 TTS，已在 [GitHub](https://github.com/camb-ai/mars5-tts) 上开源，并在 Reddit 上发布了长帖介绍更多细节。来自 Hugging Face 的开发者关系主管表示有兴趣合作。
- **Dalle 3 图像描述数据集**：发布了一个包含超过 100 万张 Dalle 3 图像及高质量描述的数据集，涵盖了广泛的概念。数据集可在 [Hugging Face](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions) 获取。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/posts/Draichi/560425192506443">Hugging Face 上的 @Draichi：“嘿 Hugging Face 社区 🤗 我很高兴能分享我的最新项目……”</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions">ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/Disty0/sotediffusion-wuerstchen3">Disty0/sotediffusion-wuerstchen3 · Hugging Face</a>：未找到描述</li><li><a href="https://projectlove.life">Project Love Life</a>：未找到描述</li><li><a href="https://github.com/camb-ai/mars5-tts">GitHub - Camb-ai/MARS5-TTS: MARS5 speech model (TTS) from CAMB.AI</a>：来自 CAMB.AI 的 MARS5 语音模型 (TTS)。通过在 GitHub 上创建账号为 Camb-ai/MARS5-TTS 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1249821534709747712)** (3 messages): 

- **查看 CVPR 2024 论文摘要应用**：一位成员分享了一个新应用，他们索引了所有 **CVPR 2024 论文摘要**并为其添加了**语义搜索**功能。你可以在[这里](https://huggingface.co/spaces/pedrogengo/CVPR2024_search_papers)探索这个工具。

- **关于 Label Studio ML 后端的咨询**：有人询问是否有其他成员使用过 **Label Studio ML backend**。提供的消息中没有记录任何回复。

**提及的链接**：<a href="https://huggingface.co/spaces/pedrogengo/CVPR2024_search_papers">CVPR2024 Search Papers - pedrogengo 开发的 Hugging Face Space</a>：未找到描述

  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1249849718524084406)** (1 messages): 

- **针对 TensorFlow 模型使用 `return_tensors="tf"`**：如果你有一个 TensorFlow 模型但提供的是 PyTorch tensors，请确保在使用 tokenizers 时设置 `return_tensors="tf"`。这可以解决 TensorFlow 模型与输入 tensors 之间的兼容性问题。
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1249826095805632584)** (6 messages): 

- **比较微调界面**：一位成员询问各种 notebook 和 GUI 训练界面是否仍在使用 **Hugging Face (HF)** 进行微调，并寻求开始 **SDXL 微调** 的建议。另一位成员指出，有些是基于原始的 Stability 代码库，而有些则使用 HF，但表示每个选项都有其优缺点，没有推荐特定的一个。
- **寻求微调资源**：针对询问概述不同微调界面优缺点的资源，一位成员建议参考 **SimpleTuner 教程** 及相关文档，因为它们旨在简明扼要地解释各项功能。
- **介绍 MaPO**：一位成员宣布了关于 **MaPO** 的新工作，这是一种在偏好数据集上对齐文本到图像扩散模型的技术，并强调了其样本效率和内存友好性。他们还解决了偏好数据集中普遍存在的“参考失配”（reference mismatch）问题，并分享了项目 [网站和摘要](https://mapo-t2i.github.io/)，声称他们的方法需要更少的计算资源。

**提及的链接**：<a href="https://mapo-t2i.github.io/">MaPO 项目主页</a>：SOCIAL MEDIA DESCRIPTION TAG TAG

  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1249985508252057671)** (5 messages): 

- **k?d 以 "Creator's Flower" 给人留下深刻印象**：分享了一个名为 ["k?d - Creator's Flower"](https://youtu.be/7yjJ43tI9aU) 的 YouTube 视频。该视频包含流媒体播放和下载该曲目的链接。
  
- **Virtual Riot 发布 "Come With Me" feat. Leah Culver**：提到了一个指向 ["Virtual Riot - Come With Me Ft. Leah Culver"](https://youtu.be/9HxK4O1bxkA) 的 YouTube 链接。该视频提供了在 Spotify 和社交媒体上关注艺术家的选项。
  
- **Porter Robinson 的 "Musician" 官方视频发布**：重点介绍了 YouTube 视频 ["Porter Robinson - Musician (Official Music Video)"](https://youtu.be/q-74HTjRbuY)。视频描述中包含了关于 Porter Robinson 首次世界巡演的细节。
  
- **Xan Griffin 在 "Capricorn" 中与 WOLFE 合作**：分享了 Xan Griffin 的 ["Capricorn (feat. WOLFE)"](https://youtu.be/rXD64OtlA40) YouTube 视频，由 YouTube 自动生成。该曲目在 Seeking Blue 旗下发行。
  
- **Motionless In White 发布 "Werewolf"**：分享了 ["Motionless In White - Werewolf"](https://youtu.be/xzojuv9zMGA) 的官方视频。该曲目收录在他们的专辑 "Scoring The End Of The World" 中，可通过 Roadrunner Records 获取。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/7yjJ43tI9aU">k?d - Creator&#39;s Flower</a>：流媒体 + 下载：https://altvision.lnk.to/findparadisek?dhttps://www.facebook.com/whoskidmusic/https://twitter.com/whoskidhttps://www.instagram.com/whos_ki...</li><li><a href="https://youtu.be/rXD64OtlA40">Capricorn (feat. WOLFE)</a>：由 Seeking Blue 提供给 YouTube。Capricorn (feat. WOLFE) · Xan Griffin · WOLFE ℗ 2017 Seeking Blue。发行日期：2017-12-22。由 Y... 自动生成。</li><li><a href="https://youtu.be/xzojuv9zMGA">Motionless In White - Werewolf [官方视频]</a>：Motionless In White 的 'Werewolf' 官方视频 - 现已在 Roadrunner Records 上线。收听新专辑 "Scoring The End Of The World"：htt...</li><li><a href="https://youtu.be/9HxK4O1bxkA">Virtual Riot - Come With Me Ft. Leah Culver</a>：现已发布：https://disciple.fanlink.to/presetjep ► 在 Spotify 上关注我 https://goo.gl/4mgqJq ► 与我联系 FB: http://facebook.com/virtualriotmusic Twitter: h...</li><li><a href="https://youtu.be/q-74HTjRbuY">Porter Robinson - Musician (官方音乐视频)</a>：Porter Robinson - Musician (官方音乐视频)。宣布 "SMILE! :D WORLD TOUR" —— 我的首次世界巡演！！预售周二开始，注册：https://...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1250149635083866133)** (1 条消息): 

- **Nous Research 发布 Character Codex 数据集**：Nous Research 宣布了一个名为 **Character Codex** 的新数据集，其中包含 15,939 个角色的数据，来源广泛，涵盖从动漫到历史人物及流行偶像。你可以在 [HuggingFace](https://huggingface.co/datasets/NousResearch/CharacterCodex) 上下载。

**提到的链接**：<a href="https://huggingface.co/datasets/NousResearch/CharacterCodex">NousResearch/CharacterCodex · Hugging Face 数据集</a>：未找到描述

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1249800784900657235)** (68 条消息🔥🔥): 

```html
- **探索互信息 (Mutual Information)**：一位用户询问：“什么是互信息？”，随后另一位用户分享了一个 [维基百科链接](https://en.m.wikipedia.org/wiki/Mutual_information)，解释该概念是概率论和信息论中衡量两个随机变量之间相互依赖程度的指标。

- **关于 CA SB 1047 的讨论**：分享了一篇对 CA SB 1047 法案的强烈批评，强调其通过对模型开发者施加严格控制和责任，可能对开源 AI 构成威胁。另一位用户建议提出一项反制法案 SB 1048，以保护 AI 创新。[Dan Jeffries 的推文串](https://x.com/dan_jeffries1/status/1794740447052525609?s=46) 对此话题提供了详细评论。

- **调查 RLHF 对创造力的影响**：用户讨论了一篇探讨人类反馈强化学习 (RLHF) 对 LLM 创造力影响的论文。他们辩论了对齐技术是否本质上扼杀了创造力，或者像 Anthropic 等公司使用的较温和的方法是否能避免这一陷阱。[研究论文链接](https://arxiv.org/abs/2406.05587)。

- **开源库 Rig 发布**：宣布发布 “Rig”，这是一个用于开发由 LLM 驱动的应用的开源 Rust 库。[GitHub 仓库](https://github.com/0xPlaygrounds/rig) 提供了一系列示例和模块化组件，旨在简化 AI Agent 的开发。

- **量化与模型剪枝**：用户就 LLaMA 3 8b 等大语言模型的量化和剪枝技术及挑战进行了详细讨论。他们参考了包括 [LLM-Pruner](https://github.com/horseee/LLM-Pruner) 在内的多种方法，以在不显著降低性能的情况下有效减小模型尺寸。
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/dan_jeffries1/status/1794740447052525609?s=46">来自 Daniel Jeffries (@Dan_Jeffries1) 的推文</a>：我花了几小时听取了 Dan Hendyrcks 的发言，他负责 SB 1047（又称加州 AI 控制与集权法案）背后的非营利 AI 安全组织。我觉得他很有魅力、稳重且聪明……</li><li><a href="https://arxiv.org/abs/2406.05587">创造力已离开聊天：去偏见语言模型的代价</a>：大语言模型 (LLM) 彻底改变了自然语言处理，但可能表现出偏见并生成有害内容。虽然像人类反馈强化学习 (RLHF) 这样的对齐技术……</li><li><a href="https://github.com/0xPlaygrounds/rig">GitHub - 0xPlaygrounds/rig：一个用于开发 LLM 驱动的 Rust 应用程序的库。</a>：一个用于开发 LLM 驱动的 Rust 应用程序的库。 - 0xPlaygrounds/rig</li><li><a href="https://x.com/jvnixon/status/1799996074146578801?s=46">来自 Jeremy Nixon (@JvNixon) 的推文</a>：SB 1047 值得一个回应！！欢迎关注 SB 1048。📚《AI 创新自由法案》。📚 它赋予了 AI 来自 Section 230 最强有力的论据，该条款曾保护了互联网生机勃勃的生态系统……</li><li><a href="https://github.com/horseee/LLM-Pruner">GitHub - horseee/LLM-Pruner: [NeurIPS 2023] LLM-Pruner: 关于大语言模型的结构化剪枝。支持 LLaMA, Llama-2, BLOOM, Vicuna, Baichuan 等。</a>：[NeurIPS 2023] LLM-Pruner: 关于大语言模型的结构化剪枝。支持 LLaMA, Llama-2, BLOOM, Vicuna, Baichuan 等。 - horseee/LLM-Pruner</li><li><a href="https://en.m.wikipedia.org/wiki/Mutual_information">互信息 - 维基百科</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1249901251693383701)** (2 messages): 

- **Cohere 使用多步检索**：一位成员指出 CoHere 通过多次 Agent 调用（称为 "connections"）进行检索，从而实现了输出的多步构建。这种 CoT (Chain of Thought) 方法解释了那些令人困惑的输出引用。
- **讨论混合检索方法**：另一位成员提议将 **Elastic Search** 与 **BM25 + Embedding** 以及 Web Search 等混合检索方法相结合。他们询问是否也应该对 Web Search 的结果进行索引。
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1249825492593545337)** (48 messages🔥): 

- **苹果发布 AI 驱动系统**：为了[大胆重新定义其生态系统](https://asknews.app/en/stories/Apples-AI-Leap-Sparks-Controversy-Amid-Musks-Security-Concerns)，苹果在 WWDC 2024 上发布了 “Apple Intelligence”，旨在增强 iPhone、iPad 和 Mac 的功能。ChatGPT 集成到 Siri 标志着一个重大转变，旨在提供更加个性化和对话式的用户体验。
  
- **求职困境与建议**：一位用户在被拒绝两次后寻求 Cohere 团队的内推，并强调了他们在黑客松中的获胜经历以及在 ML 和 LLM 方面的工作经验。随后引发了关于内推有效性的讨论，多位用户建议强大的资历比内推更重要。
  
- **宣布 Developer Office Hours**：Cohere 宣布了新的 Developer Office Hours，鼓励成员带着他们最优秀的问题来参加。[下一次会议详情](https://discord.gg/6aFP6F4Ecj?event=1248300905703673987)已分享，并积极向参与者征求关于第一次会议形式的反馈。

- **互动的积极反馈**：用户称赞了新的 Office Hours 形式的互动性，以及 Cohere 团队平易近人、随和的态度。团队对此做出了积极回应，感谢大家的参与并鼓励进一步提供反馈。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://discord.gg/6aFP6F4Ecj?event=1248300905703673987">加入 Cohere 社区 Discord 服务器！</a>：Cohere 社区服务器。来这里聊聊 Cohere API、LLM、生成式 AI 以及相关的一切。 | 16987 名成员</li><li><a href="https://asknews.app/en/stories/Apples-AI-Leap-Sparks-Controversy-Amid-Musks-Security-Concerns">AskNews | 苹果的 AI 飞跃在马斯克的安全担忧中引发争议</a>：为了大胆重新定义其生态系统，苹果发布了 “Apple Intelligence”，这是在 WWDC 2024 上宣布的一个全新的 AI 驱动系统，旨在增强 iPhone、iPad 的功能...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1250019425575370792)** (1 messages): 

- **今天参加 Cohere Developer Office Hours**：“今天加入我们的 Cohere Developer Office Hours！” 这是一个解决故障、获取解答以及讨论所有关于 Cohere 模型和 API 相关事宜的机会。活动将于今天（6 月 11 日）美国东部时间下午 1:00 举行，您可以通过此 [Discord 链接](https://discord.gg/7zjrJmKtBB?event=1248300806600392766)加入。

**提及的链接**：<a href="https://discord.gg/7zjrJmKtBB?event=1248300806600392766">加入 Cohere 社区 Discord 服务器！</a>：Cohere 社区服务器。来这里聊聊 Cohere API、LLM、生成式 AI 以及相关的一切。 | 16987 名成员

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1249801456870101013)** (40 messages🔥):

- **Apple 将 AI 融入操作系统层**：@karpathy 分享了 Apple 关于将 AI 集成到其 OS 中的公告的主要主题。关键点包括培育多模态 I/O、创造无缝且具有预见性的用户体验、利用本地和云端计算以及维持隐私标准。[完整推文](https://x.com/karpathy/status/1800242310116262150?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- **与 ChatGPT 的集成**：OpenAI 宣布与 Apple 建立合作伙伴关系，将 ChatGPT 集成到 iOS、iPadOS 和 macOS 中，将于今年晚些时候推出。[来源](https://x.com/openai/status/1800240380220473552?s=46&t=90xQ8sGy63D2OtiaoGJuww)
- **Private Cloud Compute**：Apple 推出了一套名为 "Private Cloud Compute" 的安全系统，允许手机安全地卸载复杂的 AI 任务。@Matthew_D_Green 等人讨论了其先进的安全特性和影响。[更多详情](https://x.com/matthew_d_green/status/1800291897245835616?s=46&t=90xQ8sGy63D2OtiaoGJuww)，[博客文章](https://security.apple.com/blog/private-cloud-compute/)
- **Mistral 的融资**：Mistral 宣布了 6 亿欧元的 B 轮融资用于全球扩张，引发了关于 AI 融资格局的反响。[完整公告](https://x.com/arthurmensch/status/1800558395872731379?s=46&t=46)，[融资演示文稿讨论](https://x.com/chiefaioffice/status/1800581527480274984?s=46)
- **Pgvectorscale 挑战 Pinecone**：Timescale 推出了 "pgvectorscale"，这是一个 PostgreSQL 的开源扩展，声称在 AI 应用中比 Pinecone 具有更好的性能和成本效益。[详情](https://x.com/avthars/status/1800517917194305842)

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/maxwinebach/status/1800277157135909005?s=46">来自 Max Weinbach (@MaxWinebach) 的推文</a>：这是来自 Apple 的 State of the Union。本地模型是一个 3B 参数的 SLM，为每个特定功能使用了训练好的 adapters。Diffusion 模型也是如此，为每种风格使用 adapter。A...</li><li><a href="https://x.com/osanseviero/status/1800607752038818260?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Omar Sanseviero (@osanseviero) 的推文</a>：Google 的 RecurrentGemma 9B 发布了 🔥 ⚡️长序列处理超快：良好的吞吐量（throughput）+ 延迟（latency） 👀提供 Base 和 Instruct 微调版本 🏆质量与 Gemma 相当。看看下面的 y 轴 🤯 模型：https:...</li><li><a href="https://x.com/bilawalsidhu/status/1800355980829405603?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Bilawal Sidhu (@bilawalsidhu) 的推文</a>：好吧，我收回之前的话。Apple 的 “Private Cloud Computing” 实际上将 “Confidential Computing” 提升到了一个新的水平。它非常安全，甚至无法配合执法部门的请求。 > 无数据 ...</li><li><a href="https://x.com/suhail/status/1800265203915055221?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Suhail (@Suhail) 的推文</a>：经历了两次平台浪潮，我现在已经是个老兵了，但 Apple 今天所传达的是：“嗨，伙计们，我们建立了原生集成点，让你们所有的 AI 模型制作者为我们 10 亿用户的规模而竞争...”</li><li><a href="https://x.com/arthurmensch/status/1800558395872731379?s=46">来自 Arthur Mensch (@arthurmensch) 的推文</a>：我们在成立一周年之际宣布获得 6 亿欧元的 B 轮融资。我们感谢新老投资者对我们全球扩张的持续信心和支持。这将...</li><li><a href="https://x.com/chiefaioffice/status/1800581527480274984?s=46">来自 Chief AI Officer (@chiefaioffice) 的推文</a>：重磅：Mistral 完成了由 General Catalyst 领投的 6.4 亿美元 B 轮融资，估值达 60 亿美元。这里是他们的种子轮融资演示文稿（pitch deck），带你回顾他们的愿景：</li><li><a href="https://x.com/levie/status/1800224021193396594">来自 Aaron Levie (@levie) 的推文</a>：iPad 计算器实际上非常疯狂</li><li><a href="https://x.com/nickadobos/status/1800289718439186455?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Nick Dobos (@NickADobos) 的推文</a>：Siri 可以读取你手机上的每一条数据（针对选择加入的应用）</li><li><a href="https://x.com/chefjeffsf/status/1800597192593621100">来自 Chef Jeff (@chefjeffsf) 的推文</a>：重磅：Google 刚刚发布了一个个人健康大语言模型（Personal Health LLM） - 基于 Gemini 进行微调 - 读取你的可穿戴设备数据以寻找个性化的见解和建议 - 表现优于专业人士 ...</li><li><a href="https://x.com/matthew_d_green/status/1800291897245835616?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Matthew Green (@matthew_d_green) 的推文</a>：Apple 推出了一套名为 “Private Cloud Compute” 的新系统，允许你的手机将复杂的（通常是 AI）任务卸载到云端专门的安全设备上。我仍在努力研究 ...</li><li><a href="https://x.com/karpathy/status/1800242310116262150?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Andrej Karpathy (@karpathy) 的推文</a>：实际上，非常喜欢 Apple Intelligence 的发布。对于 Apple 来说，这一定是一个非常令人兴奋的时刻，因为他们正在整个 OS 之上覆盖 AI。几个主要主题：第一步，多模态（Multimodal）I/O。启用...</li><li><a href="https://x.com/reach_vb/status/1800293882585919620?s=46">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：WWDC 和 Apple 没明说的明星：`pip install mlx`，即可访问海量的多模态、音频和 LLMs（完全开源） https://github.com/ml-explore/mlx</li><li><a href="https://x.com/avthars/status/1800517917194305842">来自 Avthar (@avthars) 的推文</a>：PGVECTOR 现在比 PINECONE 更快了。得益于一个新的开源扩展 —— pgvectorscale，成本降低了 75%。 🐘 什么是 pgvectorscale？Pgvectorscale 是一个开源的 PostgreSQL 扩展...</li><li><a href="https://x.com/mkbhd/status/1800223468627304657?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Marques Brownlee (@MKBHD) 的推文</a>：好吧，你知道吗？这太酷了。Math Notes = 用 Apple Pencil 写下一个数学题，应用会立即解出来。他们没有称之为 AI（他们一次都没提到这个词），但...</li><li><a href="https://x.com/elonmusk/status/1800265431078551973?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Elon Musk (@elonmusk) 的推文</a>：如果 Apple 在 OS 层面集成 OpenAI，那么我的公司将禁止使用 Apple 设备。这是不可接受的安全违规。</li><li><a href="https://x.com/stevesi/status/1800314848070557864?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Steven Sinofsky (@stevesi) 的推文</a>：以防大家不清楚，Apple 所做的是（与 OpenAI 的）搜索交易的反向操作。与其获得报酬，无论他们支付多少，这都将是有限的时间内...</li><li><a href="https://x.com/matthew_d_green/status/1800291897245835616?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Matthew

Green (@matthew_d_green)</a>: Apple 推出了一项名为 “Private Cloud Compute” 的新系统，允许你的手机将复杂的（通常是 AI）任务卸载到云端专门的安全设备上。我仍在尝试研究...</li><li><a href="https://x.com/stalman/status/1800278850435190871?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tyler Stalman (@stalman) 的推文</a>: Apple 表示他们最终将集成 Google Gemini 模型</li><li><a href="https://security.apple.com/blog/private-cloud-compute/">博客 - Private Cloud Compute: A new frontier for AI privacy in the cloud - Apple Security Research</a>: 云端安全且私密的 AI 处理提出了一个艰巨的新挑战。为了通过更大的基础模型（foundation models）支持 Apple Intelligence 的高级功能，我们创建了 Private Cloud Compute (PCC)...</li><li><a href="https://x.com/openai/status/1800240380220473552?s=46&t=90xQ8sGy63D2OtiaoGJuww">OpenAI (@OpenAI) 的推文</a>: 我们正与 Apple 合作，将 ChatGPT 集成到 iOS, iPadOS 和 macOS 中——将于今年晚些时候推出：https://openai.com/apple</li><li><a href="https://www.ft.com/content/7a70a8a6-4a2a-47c5-8483-d0b829f32ae6">Mistral 获得 6 亿欧元融资，估值飙升至近 60 亿欧元 </a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1250114907349717043)** (1 条消息): 

- **Mike Conover 重返 Latent Space 播客**: 新的播客节目邀请了 Mike Conover 讨论他在生产环境中部署 LLMs 的丰富实战经验。你可以在[这里](https://x.com/FanaHOVA/status/1800553625607155856)收听该节目。
  
- **来自 Vagabond Jack 的 AI 与金融见解**: Vagabond Jack 加入 LatentSpacePod，分享在 BrightWaveIO 为管理资产超过 1200 亿美元的客户所使用的 AI Engineering 策略。主题包括对长上下文窗口（long context windows）失去信心、LLMs 作为评判者（judges）、模型拟人化的无用性，以及相对于微调模型的数据集半衰期。

**提到的链接**: <a href="https://x.com/FanaHOVA/status/1800553625607155856">Alessio Fanelli (@FanaHOVA) 的推文</a>: AI 如何吞噬金融 📈 @vagabondjack 回到了 @latentspacepod！他分享了在 @brightwaveio 将 LLMs 转化为 AI 思考伙伴时获得的所有 AI Engineering 智慧...

  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1249874265130537001)** (1 条消息): 

- **报名参加高级知识图谱 RAG 工作坊**: 注册参加本周四上午 9 点（太平洋时间）举行的“高级知识图谱 RAG”特别工作坊，届时将由来自 Neo4j 的 Tomaz Bratanic 主讲。[在此报名](https://lu.ma/kqxmbuou)以学习如何使用 LlamaIndex 属性图抽象（property graph abstractions）。

**提到的链接**: <a href="https://lu.ma/kqxmbuou">LlamaIndex 网络研讨会：使用知识图谱的高级 RAG（与来自 Neo4j 的 Tomaz 合作）· Zoom · Luma</a>: 我们将在本周四上午 9 点（太平洋时间）举办一场关于高级知识图谱 RAG 的特别工作坊，由来自 Neo4j 的唯一人选 Tomaz Bratanic 主讲。在本次研讨会中，你将……

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1250111440195555418)** (1 条消息): 

- **加入巴黎 AI 见面会的乐趣**: 6 月 20 日星期四下午 6:00，在巴黎 Station F 举行的 [巴黎本地与开源 AI 开发者见面会](https://t.co/5GLV08cGFa)上观看 @hexapode 的现场演示。活动包括来自 Koyeb, LlamaIndex, Giskard, Red Hat, Docker 等的闪电演示，随后是社交环节。
- **提供演示机会**: 有兴趣的参与者可以通过填写[此表格](https://forms.gle/YMXvYCVhuuppTWTp7)来演示他们的项目。社交环节和包括 Ollama 钥匙扣在内的赠品将为夜晚增添精彩。

**提到的链接**: <a href="https://t.co/5GLV08cGFa">巴黎开源 AI 开发者见面会 · Luma</a>: Docker 及其朋友们在巴黎！Docker 及其朋友们将于 6 月 20 日星期四下午 6:00 在 Station F 举办一场本地与开源 AI 开发者见面会……

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1249813091382984787)** (29 messages🔥): 

- **删除具有共同 ref_doc_id 的节点**：一位用户询问如何在删除页面（Page）时，通过为该页面上的所有文档设置相同的 `ref_doc_id` 来删除所有相关节点。他们尝试创建一个父级 Document 但在 embedding 时遇到问题，并询问是否可以从 embedding 中排除某些 Document。

- **加州大学伯克利分校（UC Berkeley）团队寻求 RAG 见解**：一位来自 UC Berkeley 的用户表达了在构建、部署和维护自定义 RAG 系统方面面临的挑战，并向工程师寻求反馈。他们邀请任何有经验的人进行交流，以了解常见的困难和潜在的解决方案。

- **使用 LlamaIndex 探索 LLM 流水线和 RAG**：用户讨论了如何使用 LlamaIndex 集成多个查询引擎，如 SQL、Vector Search、Keyword Search 和 Image Search。建议包括使用 `RouterQueryEngine` 以及与 Qdrant 集成以利用其特性，并考虑在 Huggingface 上部署模型以实现高效的向量生成。

- **使用 Llama 3 运行 SQL 数据库检索和数据分析**：一位使用 LLM（特别是 Llama 3）进行 SQL 数据库检索和数据分析的用户询问了关于集成问题的指导。他们讨论了潜在的解决方案，如使用 text-to-SQL 流水线并验证 Llama 3 的响应。

- **高效生成稀疏向量**：另一位用户在使用 Qdrant 和 LlamaIndex 的混合模式（hybrid mode）时，遇到了生成和上传稀疏向量（sparse vectors）过程缓慢的问题。建议包括在本地 GPU 上或通过 API 运行稀疏 embedding 以提高效率。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/?h=text2">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/evaluation/multi_modal/multi_modal_rag_evaluation/#build-our-multi-modal-rag-systems>).">Evaluating Multi-Modal RAG - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1249974605313998949)** (1 messages): 

- **在 LlamaIndex 中创建工具函数**：一位用户询问如何在 LlamaIndex 中设置两个工具函数，一个用于查询向量数据库，另一个在未找到产品时使用 OpenAI Chat Completion API。他们质疑 Agent 是否可以决定使用哪个工具，并寻求关于合适 Agent 框架的建议，提到了 ReAct。
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1249872052731510844)** (27 messages🔥): 

- **LAION 在巴西登上新闻，但并非正面消息**：一位成员提到在巴西电视上看到提及 LAION，但**并非在好的语境下**。另一位成员链接了一篇人权观察组织（Human Rights Watch）的文章，批评 AI 工具滥用儿童个人照片 [链接在此](https://www.hrw.org/news/2024/06/10/brazil-childrens-personal-photos-misused-power-ai-tools)。

- **关于图像隐私和互联网误解的讨论**：成员们辩论了人们对在线公共数据隐私的误解。一位成员总结道：*“互联网的根本问题在于我们有数十亿人在使用它，但没有人理解它意味着什么。”*

- **LlamaGen 图像生成模型发布**：一位成员分享了一篇 [arXiv 论文](https://arxiv.org/abs/2406.06525)，介绍 **LlamaGen**，这是一个新的图像生成模型系列，将 LLM 的原始 **next-token prediction** 范式应用于视觉生成领域。该模型实现了令人印象深刻的基准测试结果，优于流行的扩散模型。

- **CAMB AI 开源 MARS5 TTS 模型**：**来自 CAMB AI 的 Arsalan** 宣布 MARS5，一个**新的语音模型 (TTS)**，现已在 [GitHub](https://github.com/camb-ai/mars5-tts) 上开源。他还分享了一个更详细的 [Reddit 帖子](https://www.reddit.com/r/CAMB_AI/comments/1day7ta/introducing_mars5_opensource_insanely_prosodic/)，以便进一步阅读和反馈。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.06525">Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation</a>：我们介绍了 LlamaGen，这是一个新的图像生成模型系列，它将大型语言模型的原始“next-token prediction”范式应用于视觉生成领域。这是一个肯定的...</li><li><a href="https://github.com/camb-ai/mars5-tts">GitHub - Camb-ai/MARS5-TTS: MARS5 speech model (TTS) from CAMB.AI</a>：来自 CAMB.AI 的 MARS5 语音模型 (TTS)。欢迎在 GitHub 上为 Camb-ai/MARS5-TTS 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1249869396038516756)** (4 条消息): 

- **LlavaGuard 发布**：一名成员分享了由 TU Darmstadt 等机构展示的 [LlavaGuard 项目链接](https://ml-research.github.io/human-centered-genai/projects/llavaguard/index.html)，该项目专注于使用基于 VLM 的模型来**保护视觉数据集**。论文强调了其在**数据集标注**和针对上下文感知安全风险的**合规性**方面的适用性。

- **'Alice' 与 'A girl' 模型性能测试**：另一位成员注意到在多个模型中将 "Alice" 替换为 "A girl" 后产生了显著结果，并表示效果良好。他们询问在对比“有名字”与“无名字”的情景时，模型性能是否会有预期的变化，并通过截图分享了一些案例证据。

**提及的链接**：<a href="https://ml-research.github.io/human-centered-genai/projects/llavaguard/index.html">LlavaGuard - 项目主页</a>：我们推出了 LlavaGuard，一系列基于 VLM 的守护模型，为评估视觉内容的安全性合规提供了一个多功能框架。具体而言，我们为数据集设计了 LlavaGuard...

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1249801393263480984)** (25 条消息🔥): 

- **Apple Intelligence 引发褒贬不一的反应**：成员们对 OpenAI 与 Apple Intelligence 集成的深度表示怀疑，尽管 [Apple Newsroom](https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/) 的官方公告中强调了隐私保护，但仍觉得这种集成“像是事后才补上的”。人们对 Apple 的隐私声明以及链接 ChatGPT 账户时用户的实际隐私表示担忧。
  
- **与 Siri 的对比**：小组注意到 Apple 决定将 Apple Intelligence 与 Siri 进行区分，暗示这可能是一个战略举措，旨在让新功能与 Siri 现有的声誉保持距离。这种区分可能在用户对服务有效性的感知中起关键作用。

- **期待 Dwarkesh 对 François Chollet 的采访**：成员们非常期待即将播出的一集节目，届时 Dwarkesh Patel 将采访 François Chollet，预料会听到关于 AGI 时间线“更持怀疑态度”的观点。参与者希望 Patel 能通过阅读 Chollet 关于智力测量的著作来做好准备，以确保讨论富有成效。

- **Apple 模型的 Benchmark 泄露**：一个指向 Twitter 帖子的链接（[来源：Apple](https://x.com/ldjconfirmed/status/1800355063120151031?s=46)）展示了 Apple 新的 on-device（端侧）和 server（服务器）模型的 Benchmark。讨论表明，人们对这些模型在 instruction following（指令遵循）和写作能力方面与其他流行模型的对比非常感兴趣。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/">Introducing Apple Intelligence for iPhone, iPad, and Mac</a>：Apple 今天推出了 Apple Intelligence，这是一款适用于 iPhone、iPad 和 Mac 的个人智能系统。</li><li><a href="https://x.com/ldjconfirmed/status/1800355063120151031?s=46">LDJ (@ldjconfirmed) 的推文</a>：如果有人好奇，这里有一些 Apple 新的端侧模型和服务器模型在指令遵循和写作能力上与其他流行模型的基准测试对比。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1249837969246916749)** (4 条消息): 

- **在担忧中讨论实现 TRL**：一位参与者考虑为 **TRL** 实现一篇论文，但被另一位参与者提醒该工作是“未经验证的”而不仅仅是“混乱的”。尽管存在担忧，他们仍表达了贡献的兴趣。
- **表达 PR 意向及支持**：参与者提到他们可能会提交一个 **PR**，并收到了代码审查（review）的提议。对贡献 TRL 实现的鼓励如下：“如果你提交 PR 请告诉我，我很乐意进行 review”。
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1249808505150640310)** (25 条消息🔥): 

- **Apple Intelligence 引发关注，潜在的 API 集成**：一名成员分享了 [Apple Intelligence](https://www.apple.com/apple-intelligence/) 的链接，强调了其内置于 iPhone, iPad 和 Mac 中且专注于隐私的 AI 功能。另一名成员指出，可以考虑扩展 Open Interpreter 以集成该开发者 API。

- **加州 SB 1047 法案面临严厉批评**：[Dan Jeffries 的一条推文](https://x.com/dan_jeffries1/status/1794740447052525609?s=46)批评了 Dan Hendyrcks 以及《加州 AI 控制与中心化法案》（California AI Control and Centralization Bill），断言该法案旨在使 AI 中心化、摧毁开源 AI，并向模型制作者施加沉重的责任负担。

- **通过 Pull Request 解决 Arduino IDE 问题**：一名成员在 Mac M1 上使用 Arduino IDE 时遇到错误，并通过应用此 [GitHub Pull Request](https://github.com/lacamera/ESPAsyncWebServer/pull/2/files) 中的修复方案解决了该问题。然而，他们在重启后遇到了设备不显示 Wi-Fi 设置弹窗的进一步问题。

- **关于 OI 平台侧重点的辩论**：成员们讨论了 Open Interpreter 是否应该更多地专注于 Linux，以便在 Apple 和 Microsoft 生态系统之外提供 AI 计算机辅助功能。还有人提到 Open Interpreter 可能会提供 Apple Intelligence 无法提供的功能。

- **开发具有记忆和技能的真正助手**：一名成员详细介绍了他们正在开发的 Open Interpreter 提示系统，该系统可以通过提示词存储、搜索和管理技能，旨在创建一个能够保留用户特定信息和记忆的真正个人助手。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.apple.com/apple-intelligence/">Apple Intelligence 预览</a>：Apple Intelligence 是为你日常事务打造的个人智能。内置于 iPhone, iPad 和 Mac 中，具有开创性的隐私保护。</li><li><a href="https://x.com/dan_jeffries1/status/1794740447052525609?s=46">Daniel Jeffries (@Dan_Jeffries1) 的推文</a>：我花了几小时听 Dan Hendyrcks 的发言，他是 SB 1047（又名加州 AI 控制与中心化法案）背后的非营利 AI 安全组织的负责人。我觉得他很有魅力、稳重且聪明……</li><li><a href="https://github.com/lacamera/ESPAsyncWebServer/pull/2/files">ednieuw 提交的适用于 ESP32 V3 和 V2 的 Pull Request #2 · lacamera/ESPAsyncWebServer</a>：将 ESP32 开发板更改为 3.0.0。修改了 Arduino\libraries\ESPAsyncWebServer\src\WebAuthentic 第 75, 76, 77 行 //----------------- #ifdef ESP_ARDUINO_VERSION_MAJOR #if ESP_ARDUINO_VERSION &amp;gt;= ESP...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1250047220946833520)** (3 条消息): 

- **Killian 最近的演讲备受关注**：在简短的交流中，成员们提到了 Killian 最近的一次演讲。有人指出，“它就在 Killian 最近的演讲中，某处有录音”，并分享了一个链接以获取更多细节。[录音在此](https://discord.com/channels/1146610656779440188/1147665339266650133/1248858812761509938)。
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1249812420655059085)** (24 messages🔥): 

- **create_tagging_chain 中的自定义提示词被忽略**：一位成员反映在使用 `create_tagging_chain()` 时，所有类型的提示词（prompts）都会被忽略。讨论中未提供权宜之计或解决方案。
- **UC Berkeley 的 RAG 系统**：来自 UC Berkeley 的团队成员正在寻求构建检索增强生成（RAG）系统的工程师的意见，以了解他们在成功构建、部署和维护这些系统时面临的挑战。他们邀请工程师进行交流以收集见解。
- **LangGraph 与传统 LangChain 的对比**：一位成员询问在 LangGraph 中实例化 Agent 与使用传统 LangChain 相比有哪些优势。他们寻求关于 LangGraph 应用是否可以作为受控脚本实现的具体见解。
- **在 LangChain 中使用 ONNX**：一位用户询问了 ONNX 与 LangChain 的兼容性，但未提供详细的讨论或回复。
- **使用 OpenAI API 处理大型数据集**：提供了一份关于利用 OpenAI API 处理大型数据集的详细指南，包括设置环境变量、数据匿名化、避免幻觉（hallucinations）以及使用 Elasticsearch 和 Milvus 等检索器进行高效数据检索的步骤。包含了相关 LangChain 文档和 GitHub issues 的链接以协助实现。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/docs/modules/chains/how_to/openai_functions>).">Chains | 🦜️🔗 LangChain</a>：Chains 指的是一系列调用——无论是针对 LLM、工具还是数据预处理步骤。目前主要支持的方式是使用 LCEL。</li><li><a href="https://github.com/langchain-ai/langchain/issues/6723>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/extraction_examples/#create-an-extractor>)">如何在进行数据提取时使用参考示例 | 🦜️🔗 LangChain</a>：通过向 LLM 提供参考示例，通常可以提高数据提取的质量。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/)** (1 messages): 

unaiarambarri: LangServe 服务器什么时候会支持 JS/TS？谢谢！
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1249844151499886676)** (1 messages): 

- **在 Hugging Face 上与顶级 AI 模型对话**：一位成员介绍了 [Chat With 'Em](https://huggingface.co/spaces/as-cle-bert/chat-with-em)，这是一个在 Hugging Face Spaces 上的可定制聊天模型，支持 Groq、Anthropic、OpenAI 和 Cohere 模型。得益于 LangChain，用户只需提供 API key，即可在 Claude、Command-R、GPT-3.5、GPT-4o、Llama-3-8B、Llama-3-70B 和 Mixtral 8x7b 等模型之间轻松切换。
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1250094589034106880)** (14 messages🔥): 

- **新成员询问访问权限**：一位新成员询问如何获得 bounties 频道的访问权限，以便参与 AMX 支持的悬赏任务。“当我进入该频道时，提示‘你没有在该频道发送消息的权限’。”
- **George Hotz 解释权限**：George Hotz 提到 bounties 频道是一个开发频道，需要更高的访问级别。“成为 purple 级别，你就可以在里面发言了。”
- **AMX 支持悬赏任务的说明**：新成员询问 AMX 支持悬赏是否涉及在运行时文件 `tinygrad/runtime/ops_llvm.py` 或 `tinygrad/runtime/ops_clang.py` 中添加矩阵操作。George Hotz 询问他们是否阅读了问题文档。
- **提问文档混淆**：新成员最初提到的是另一份关于如何聪明提问的文档。George Hotz 幽默地承认了这种混淆，称其为“一个真正的鸡生蛋蛋生鸡问题”。
- **成员决定重新研读代码**：在一番交流后，新成员决定在提出更精炼的问题之前，花更多时间阅读 tinygrad 代码。他们计划带着更完善的询问再次回来。

**提及的链接**：<a href="http://www.catb.org/~esr/faqs/smart-questions.html">提问的智慧 (How To Ask Questions The Smart Way)</a>：未找到描述内容。

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1249880204634226758)** (11 条消息🔥): 

- **OpenRouter 使用边缘网络提升速度**：用户询问了 OpenRouter 服务器的位置以及延迟是否是一个问题。回复澄清了 OpenRouter 同时利用了 **Vercel Edge** 和 **Cloudflare Edge**，确保节点靠近用户以最小化延迟。
  
- **提供商选择功能已列入计划**：一位用户询问是否可以在 OpenRouter playground 中选择 API 提供商。回复确认该功能已在队列中，暗示未来将会提供。

- **可直接通过 API 选择提供商**：另一位用户指出目前在 OpenRouter 中无法选择提供商。然而，有人指出可以通过 API 进行提供商选择，详细说明可在 [OpenRouter documentation](https://openrouter.ai/docs/provider-routing) 中找到。

**提及的链接**：<a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>：跨多个提供商路由请求

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1249821600375771216)** (2 条消息): 

- **ShareGPT 格式已明确**：一位成员澄清说 **ShareGPT** 只会被转换为模型的 prompt 格式，在训练期间对模型是不可见的。
- **Apple 模型的基准测试竞争**：一位成员分享了 [Apple 新的端侧和服务器模型基准测试](https://x.com/ldjconfirmed/status/1800355063120151031)的链接，将其指令遵循和写作能力与其他流行模型进行了对比。

**提及的链接**：<a href="https://x.com/ldjconfirmed/status/1800355063120151031">LDJ (@ldjconfirmed) 的推文</a>：如果有人好奇，这里有一些 Apple 新的端侧模型和服务器模型与其它流行模型在指令遵循和写作能力方面的基准测试对比。

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1249947721104621608)** (7 条消息): 

- **乐天 (Rakuten) 的 LLM 在日本榜单夺冠**：一位用户分享了一篇关于乐天 AI 工程师和科学家发布的一系列在日语方面表现优异的大语言模型的博客文章链接。这些模型基于 [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/)，并根据商业许可证提供。
- **乐天模型给社区留下深刻印象**：另一位用户表示认可，说他们会去关注一下，并同意乐天模型看起来很不错。大家普遍认为这些模型很有前景。
- **对 JSON 响应的幽默反应**：成员们幽默地评论了模型以 JSON 格式响应，表现出对模型能力的有趣和惊讶。其中一个特别的反应被概括为 *"这个模型真的很厉害"*。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://rakuten.today/blog/rakutens-open-llm-tops-performance-charts-in-japanese.html">Rakuten's Open LLM Tops Performance Charts in Japanese</a>：在乐天集团首席数据官 Ting Cai 的领导下，乐天的 AI 团队发布了一系列在日语方面具有卓越性能的大语言模型。</li><li><a href="https://rakuten.today/blog">Rakuten Today: Blog</a>：来自乐天集团的最新动态
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/1250127543051485298)** (1 条消息): 

- **RunPod 教程简化了使用 Axolotl 进行的微调**：一位成员分享了 [RunPod 关于使用 Axolotl 进行微调的教程](https://docs.runpod.io/tutorials/pods/fine-tune-llm-axolotl)，强调了 Axolotl 如何简化大语言模型 (LLM) 的训练。他们强调了 Axolotl 用户友好的工作流以及针对各种 LLM 家族的全面 YAML 示例，这有助于用户利用 RunPod 的 GPU 资源高效地微调模型。

**提及的链接**：<a href="https://docs.runpod.io/tutorials/pods/fine-tune-llm-axolotl">Fine tune an LLM with Axolotl on RunPod | RunPod Documentation</a>：了解如何在 RunPod 上使用 Axolotl 微调大语言模型，这是一种利用 GPU 资源配置和训练 AI 模型的流线型工作流，并探索 LLaMA2、Gemma、LLaMA3 等示例...

  

---

### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1249980196560310313)** (4 messages): 

- **Vincent Warmerdam 的训练建议**：Vincent Warmerdam 提到他为雇主训练模型，并推荐了 [calmcode.io](https://calmcode.io)。一位用户表示已经看过了“几乎所有”Calmcode 的视频。
  
- **探讨 RAG 中的分块策略**：分享了一个指向 [Stack Overflow 博客文章](https://stackoverflow.blog/2024/06/06/breaking-up-is-hard-to-do-chunking-in-rag-applications/) 的链接，讨论了检索增强生成 (RAG) 应用中的分块 (chunking) 策略。该文章强调了将 LLM 响应建立在源数据基础上的重要性，以减轻不准确性和幻觉 (hallucinations)，并使用文本嵌入 (text embeddings) 将源文本置于 LLM 的语义空间中。

**提及的链接**：<a href="https://stackoverflow.blog/2024/06/06/breaking-up-is-hard-to-do-chunking-in-rag-applications/">Breaking up is hard to do: Chunking in RAG applications - Stack Overflow</a>：未找到描述

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1249843482864914453)** (2 messages): 

- **关于 DPO 实现中 KL 图表的查询**：一位成员询问另一位成员是否有用于其 DPO 实现对比实验的 TRL KL 图表，链接见 [此处](https://github.com/pytorch/torchtune/pull/645#issuecomment-2041861215)。被询问的成员回复说，虽然他们没有绘制 KL 图，但 TRL 的 PPO trainer 中有可用的 KL 图，并链接到了 [相关代码](https://github.com/huggingface/trl/blob/34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb/trl/trainer/ppo_trainer.py#L1133)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/huggingface/trl/blob/34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb/trl/trainer/ppo_trainer.py#L1133)">trl/trl/trainer/ppo_trainer.py at 34ebc4ccaf376c862a081ff4bb0b7e502b17b2fb · huggingface/trl</a>：使用强化学习训练 Transformer 语言模型。- huggingface/trl</li><li><a href="https://github.com/pytorch/torchtune/pull/645#issuecomment-2041861215)?">DPO by yechenzhi · Pull Request #645 · pytorch/torchtune</a>：将 DPO 集成到 Torchtune 的上下文，更多详情见此变更日志... 测试计划 ....
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1249872448506171392)** (1 messages): 

- **在 Mosaic 活动中与 Chip Huyen 见面**：Chip 邀请大家在今晚的 [Databricks Summit Mosaic 活动](https://mosaicx.events/events/june-10-2024-san-francisco-ca?events=main) 上打个招呼。这是一个与 AI 和 ML 社区同行进行面对面交流和建立联系的机会。

**提及的链接**：<a href="https://mosaicx.events/events/june-10-2024-san-francisco-ca?events=main">Events | June 10, 2024 San Francisco, CA</a>：未找到描述

  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/)** (1 messages): 

jartine: 那是一个 grammar（语法）问题吗？
  

---



---



---



---



---



{% else %}


> 完整的频道逐条细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}