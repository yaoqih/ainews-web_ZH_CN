---
companies:
- openai
- microsoft
- google
- amazon
- github
- xiaomi
- openai-devs
- vllm_project
- kimi-moonshot
date: '2026-04-27T05:44:39.731046Z'
description: '**OpenAI** 放宽了其与 **Azure** 的独家合作限制，允许模型在 **Google TPU**、**AWS Trainium**
  以及 **Bedrock** 上进行分发，相关承诺将持续至 **2032 年**，营收分成协议则持续至 **2030 年**。**GPT-5.5** 在基准测试中有所提升，但并非在所有领域都占据绝对主导地位，在编程、文档处理、数学和视觉任务中的排名各有高低。GitHub
  的 **Copilot** 将从 6 月 1 日起转向**按量计费（usage-based billing）**模式，这反映了运行成本的增加。


  **OpenAI** 开源了 **Symphony**，这是一个用于问题追踪（issue tracking）和 Codex 智能体的编排层。**小米**发布了
  **MiMo-V2.5** 和 **MiMo-V2.5-Pro**，这两款长上下文模型支持高达 **100 万 token 的上下文窗口**，训练量达数万亿 token，重点强调了复杂的智能体和全模态能力。**Kimi
  K2.6** 在 OpenRouter 排行榜上名列前茅，其在编程以及具备大规模子智能体协同能力的长程（long-horizon）智能体任务中的表现尤为突出。'
id: MjAyNS0x
models:
- gpt-5.5
- gpt-5.4
- opus-4.7
- mimo-v2.5-pro
- mimo-v2.5
- kimi-k2.6
- codex
- copilot
people:
- sama
- scaling01
- kimmonismus
- ajassy
- simonw
- htihle
- arena
- gdb
- hangsiin
- eliebakouch
- _luofuli
- teortaxestex
title: 今天没发生什么特别的事。
topics:
- model-distribution
- cloud-computing
- benchmarking
- usage-based-billing
- model-orchestration
- open-source
- large-context-models
- agent-scaling
- coding
- model-training
- fp8
- attention-mechanisms
- multi-agent-systems
---

**平静的一天。**

> 2026/4/26-2026/4/27 的 AI 新闻。我们检查了 12 个 subreddits、[544 个 Twitters](https://twitter.com/i/lists/1585430245762441216)，没有发现更多 Discord 动态。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现已成为 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件发送频率！

---

# AI Twitter 综述

**OpenAI 分发策略转变、GPT-5.5 基准测试以及 Codex/Copilot 定价信号**

- **OpenAI 放宽了 Azure 的独占性**：[@sama](https://x.com/sama/status/2048755148361707946) 表示 OpenAI 更新了与 Microsoft 的合作伙伴关系，Microsoft 仍是其**主要云平台**，但 OpenAI 现在可以**在所有云平台上**提供产品，其产品/模型承诺延长至 **2032 年**，收入分成持续到 **2030 年**。[@scaling01](https://x.com/scaling01/status/2048752418305769473) 和 [@kimmonismus](https://x.com/kimmonismus/status/2048759615500804395) 迅速得出了结论：OpenAI 现在可以通过 **Google TPU / AWS Trainium / Bedrock** 进行分发，而 Microsoft 对 OpenAI IP 的授权变成了**非独占的**。[@ajassy](https://x.com/ajassy/status/2048806022253609115) 确认 **OpenAI 模型将在未来几周内登陆 AWS Bedrock**。[@simonw](https://x.com/simonw/status/2048834476323823983) 指出，新的表述可能意味着旧的 **AGI 条款实际上已经失效**。
- **GPT-5.5 是全面升级，但并非在所有领域都占据绝对优势**：来自 [@htihle](https://x.com/htihle/status/2048717753394090274) 的社区评估显示，**GPT-5.5 no-thinking 在 WeirdML 上的得分为 67.1%**，高于 **GPT-5.4 的 57.4%**，但仍落后于 **Opus 4.7 no-thinking 的 76.4%**，且后者使用的 tokens 更少。来自 [@arena](https://x.com/arena/status/2048794479646388732) 的 LMSYS Arena 结果显示，GPT-5.5 在 **Code Arena 排名第 9**、**Document 第 6**、**Text 第 7**、**Math 第 3**、**Search 第 2**、**Vision 第 5**，[Expert Arena 排名第 5](https://x.com/arena/status/2048808366810800259)。Arena 还澄清目前的评估涵盖了**中等/高难度推理**，而 **xHigh 仍待定** ([1](https://x.com/arena/status/2048820224938631492), [2](https://x.com/arena/status/2048846896744247468))。[@gdb](https://x.com/gdb/status/2048777802586149331) 等开发者对硬核编程任务（如 GPU kernels）的反馈是积极的，但 [@htihle](https://x.com/htihle/status/2048741770125603304) 也报告了在 no-thinking 模式下出现了“压缩后的 CoT 泄露”或输出格式错误的情况。
- **开发者经济学变得更加明确**：GitHub 宣布 [Copilot 将于 6 月 1 日转向基于用量的计费模式](https://x.com/github/status/2048794729274278258)，这是一个显著的转变，因为 Agentic 工作流会消耗更多的运行时资源。与此同时，[@Hangsiin](https://x.com/Hangsiin/status/2048719057885818902) 记录了 Codex 的使用倍率：**GPT-5.4 fast = 2x**，**GPT-5.5 fast = 2.5x**，而 5.4-mini 和 GPT-5.3-Codex 则明显更便宜。[@sama](https://x.com/sama/status/2048913887614115857) 认为 **Codex 定价 20 美元** 依然极具价值。此外，OpenAI 还通过 [@OpenAIDevs](https://x.com/OpenAIDevs/status/2048825010371039648) 开源了 **Symphony**，这是一个将问题追踪器连接到 Codex Agent 的编排层，实现了“开启问题 → Agent → PR → 人工审核”的工作流。

**小米 MiMo-V2.5、Kimi K2.6 以及中国面向 Agent 的开源权重推动**

- **MiMo-V2.5 是当日最重大的开源发布之一**：[@XiaomiMiMo](https://x.com/XiaomiMiMo/status/2048821516079661561) 在 **MIT** 协议下开源了 **MiMo‑V2.5-Pro** 和 **MiMo‑V2.5**，两者均具备 **1M-token context**。Pro 模型被定位为**复杂 Agent/编程**模型，较小的模型则定位为**原生全模态 Agent**。来自 [@eliebakouch](https://x.com/eliebakouch/status/2048845602633433258) 的社区总结补充了有用的技术细节：**MiMo‑V2.5-Pro** 总参数约为 **1T / 激活参数 42B**，在 **27T tokens** 上以 **FP8** 精度训练；而 **MiMo‑V2.5** 总参数约为 **310B / 激活参数 15B**，训练量为 **48T tokens**，采用了激进的**交替 SWA/全局注意力（interleaved SWA/global attention）**且无共享专家。小米还通过 [@_LuoFuli](https://x.com/_LuoFuli/status/2048851054662762618) 宣布为开发者提供 **100T token 资助**。首日推理支持已迅速在 [vLLM](https://x.com/vllm_project/status/2048825703244972375) 和 [SGLang/vLLM](https://x.com/XiaomiMiMo/status/2048821520798302409) 中上线。
- **Kimi K2.6 在关注度和部署方面继续领先**：[@Kimi_Moonshot](https://x.com/Kimi_Moonshot/status/2048693682329776223) 表示 **Kimi K2.6** 目前在 **OpenRouter 周榜上排名第一**。相关报道将其描述为一款适用于**编程和长程 Agent**的模型，包括在 **4,000 个协调步骤中扩展到 300 个并发子 Agent** ([dl_weekly](https://x.com/dl_weekly/status/2048764506105348129))。从业者对速度与质量的权衡仍持有分歧：[@teortaxesTex](https://x.com/teortaxesTex/status/2048820805258059837) 发现 Hermes 中的 Kimi 比 DeepSeek V4 慢得多，但有时能修复 V4 无法解决的 Bug。
- **更广泛的中国模型趋势**：多篇帖子将中国实验室描绘为正积极推进**类开源、面向 Agent、长上下文系统**：如 [Qwen 3.6 Flash](https://x.com/scaling01/status/2048730112636473792)、DeepSeek V4/Flash、GLM-5.1 推广（[三倍用量扩展](https://x.com/Zai_org/status/2048784274523148750)）以及小米的 MIT 协议发布。一个反复出现的主题是，在实际的 Agent 基准测试中，更小/更廉价的变体往往优于其体量更大的同门模型。

**Agent 运行时、编排和本地优先工具**

- **Sakana 的 Conductor 是一项显著的多 Agent 成果**：[@SakanaAILabs](https://x.com/SakanaAILabs/status/2048777689763639741) 推出了一个经过 RL 训练的 **7B Conductor**，用于以自然语言编排一组前沿模型，而非直接解决任务。它能动态决定**调用哪个 Agent、分配什么子任务以及暴露哪些上下文**，据报道在 **LiveCodeBench 上达到 83.9%**，在 **GPQA-Diamond 上达到 87.5%**，超越了其模型池中的任何单个模型。[@hardmaru](https://x.com/hardmaru/status/2048778095935795338) 强调“**AI 管理 AI**”和递归自选择是 **推理时扩展（test-time scaling）** 的一个新维度。
- **本地和混合 Agent 持续进化**：多篇帖子展示了在本地运行的编程/助手栈。[@patloeber](https://x.com/patloeber/status/2048715918541558075) 和 [@_philschmid](https://x.com/_philschmid/status/2048719354905108623) 记录了通过 LM Studio/Ollama/llama.cpp 本地运行 **Pi agent + Gemma 4 26B A4B**。[@googlegemma](https://x.com/googlegemma/status/2048805789788413984) 演示了一个使用 **Gemma 4 + WebGPU** 的**完全本地浏览器 Agent**，具备用于浏览历史、标签页管理和页面总结的原生工具调用功能。[@cognition](https://x.com/cognition/status/2048821234281181302) 发布了 **Devin for Terminal**，这是一个本地 Shell Agent，稍后可以**移交给云端**处理。
- **Agent 易用性与框架演进**：Hermes 表现强劲：[@Teknium](https://x.com/Teknium/status/2048710115885523444) 指出 **Hermes Agent 的仓库已超过 Claude Code**，同时[原生视觉在支持的情况下已成为默认配置](https://x.com/Teknium/status/2048766822766547451)。更广泛的生态系统在不断填补空白：[Cline Kanban](https://x.com/cline/status/2048814649513275448) 现在支持**每个任务卡片使用不同的 Agent/模型**；[Future AGI](https://x.com/omarsar0/status/2048759865007591615) 开源了一个用于自我进化 Agent 的评估/优化栈；[@_philschmid](https://x.com/_philschmid/status/2048781492914885079) 则认为 MCP 的最佳工作方式是通过**显式的 @mention 加载**或**子 Agent 范围内的工具分配**，而非无差别的服务器连接。

**推理基础设施、Attention/KV 工程及系统工作**

- **Google 的 TPU 拆分是一个重要的架构信号**：多篇文章剖析了 Google Cloud Next 的公告，即 **TPU v8 被拆分为用于训练（training）的 8t 和用于推理（inference）的 8i**，据称其训练速度比上一代快约 **2.8 倍**，推理性价比（performance/$）提升 **80%**。[@kimmonismus](https://x.com/kimmonismus/status/2048745304007299230) 强调这是 Google 首次按工作负载拆分定制芯片，据报道 OpenAI、Anthropic 和 Meta 都在购买 TPU 算力。
- **DeepSeek V4 在基础设施栈中的支持正迅速成熟**：[@vllm_project](https://x.com/vllm_project/status/2048769886483329525) 表示即将支持 **DeepSeek V4 基础模型**，这需要一个 `expert_dtype` 配置字段来区分 **FP4 指令微调版 vs FP8 基础版**。在 [vLLM 0.20.0 版本](https://x.com/vllm_project/status/2048918629144805619)中，亮点包括 **DeepSeek V4 支持**、**FA4 作为默认 MLA prefill**、**TurboQuant 2-bit KV**，以及在 Blackwell 上针对 DeepSeek 特有的 **MegaMoE** 路径。
- **KV cache 优化仍是激烈的竞争领域**：围绕长上下文瓶颈和 KV 策略展开了密集讨论。[@cHHillee](https://x.com/cHHillee/status/2048756662845022655) 总结了长上下文的三个主要杠杆：**局部/滑动窗口注意力（local/sliding attention）**、**交替的局部-全局注意力**，以及通过 **GQA/MLA/KV tying/量化** 实现**更小的每层全局 KV**。在实现方面，[@vllm_project](https://x.com/vllm_project/status/2048796304508330462) 与 Red Hat/AWS 发布了关于 FP8 KV-cache 的深度探讨，其中对 **FA3 两级累加** 的修复将 **128k 大海捞针（needle-in-a-haystack）的准确率从 13% 提升至 89%**，同时保留了 FP8 的解码加速。社区批评者还质疑了 DeepSeek V4 相对于 HiSparse 等重度依赖 offloading 方案的具体 KV 权衡（[讨论](https://x.com/Grad62304977/status/2048785005216723072)）。

**基准测试、评估与开放研究方向**

- **开放世界评估（Open-world evaluation）势头正盛**：[@sarahookr](https://x.com/sarahookr/status/2048731841759428935) 认为，大多数 Agent 基准测试都过度拟合于**可自动验证**的任务，而真正的技术前沿是**开放世界、不确定且非完全可验证**的工作。相关讨论将其与**持续学习（continual learning）**、记忆存储和自适应数据系统联系起来（[1](https://x.com/sarahookr/status/2048759884125233453), [2](https://x.com/adaption_ai/status/2048771654008877400)）。
- **具备成本意识的 Agent 评估正成为一等公民**：[@dair_ai](https://x.com/dair_ai/status/2048784506635878644) 强调了一项关于在 SWE-bench Verified 上进行代码 Agent 支出的新研究：Agent 化编程消耗的 **Token 数量可能比聊天/代码推理多约 1000 倍**，相同任务的不同运行之间使用量差异可达 **30 倍**，且增加支出并**不**能单调地提高准确率。这与 Copilot 定价模型的变更以及对不受控的 Agent 运行经济效益日益增长的担忧相吻合。
- **新的基准测试和特定领域的评估**：来自 LlamaIndex 的 [ParseBench](https://x.com/osanseviero/status/2048777802015535189) 为解析 Agent 增加了 **2000 页经过验证的企业文档**。[AgentIR](https://x.com/CShorten30/status/2048764263196500002) 通过将**推理轨迹与查询嵌入在一起**，重新定义了研究型 Agent 的检索，**AgentIR-4B 在 BrowseComp-Plus 上达到了 68% 的准确率，而更大规模的传统嵌入模型仅为 52%**。还有一些前沿模型的基准测试快照——例如 [Opus 4.7 以 42.2% 的成绩领跑 GSO](https://x.com/scaling01/status/2048853227211251891)，以及关于 WeirdML / ALE-Bench / PencilPuzzleBench 的讨论——但更强的信号在于方法论：越来越多的人开始测量**运行时成本、检索质量和开放世界行为**，而不仅仅是最终答案的准确率。

**热门推文（按互动量排序）**

- **OpenAI–Microsoft 合作伙伴关系重置**：[@sama](https://x.com/sama/status/2048755148361707946) 谈论跨云可用性及与 Microsoft 的持续合作。
- **OpenAI 接入 AWS**：[@ajassy](https://x.com/ajassy/status/2048806022253609115) 确认 OpenAI 模型即将登陆 **Bedrock**。
- **GitHub Copilot 定价变更**：[@github](https://x.com/github/status/2048794729274278258) 宣布从 6 月 1 日起开始**基于用量的计费**。
- **小米 MiMo-V2.5 开源发布**：[@XiaomiMiMo](https://x.com/XiaomiMiMo/status/2048821516079661561) 采用 **MIT 协议**并支持 **1M 上下文**。
- **Codex 的开源编排工具**：[@OpenAIDevs](https://x.com/OpenAIDevs/status/2048825010371039648) 发布 **Symphony**。
- **Gemma 本地浏览器 Agent**：[@googlegemma](https://x.com/googlegemma/status/2048805789788413984) 展示了一个通过 WebGPU 实现的 **100% 驻留浏览器的本地 Agent**。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Qwen3.6 模型性能与优化

- **[Luce DFlash: Qwen3.6-27B 在单张 RTX 3090 上实现高达 2x 吞吐量](https://www.reddit.com/r/LocalLLaMA/comments/1sx8uok/luce_dflash_qwen3627b_at_up_to_2x_throughput_on_a/)** (Activity: 743): **Luce DFlash** 是针对 Qwen3.6-27B 模型的一种新型投机采样解码（speculative decoding）实现，它针对单张 RTX 3090 GPU 进行了优化，使用了基于 ggml 构建的独立 C++/CUDA 技术栈。该方案在 HumanEval、GSM8K 和 Math500 等基准测试中，与自回归解码（autoregressive decoding）相比，实现了高达 `1.98x` 的吞吐量提升，且无需重新训练。系统利用压缩 KV cache 和滑动窗口 flash attention 来高效处理大上下文，并支持通过兼容 OpenAI 的 HTTP 端口提供服务。该实现仅限于 CUDA 环境，不支持多 GPU 配置。评论者们对这一创新充满热情，有人指出其在本地 AI 推理方面的潜力，另一位则因其相比现有配置的速度优势而对其表现出浓厚兴趣。

    - Tiny_Arugula_5648 提出了关于量化（quantization）对模型准确性影响的关键点。他们强调，虽然吞吐量的提升令人印象深刻，但涉及的重度量化可能会显著影响模型在某些应用中的性能，例如编码或工具调用（tool calling），在这些场景中精度至关重要。这突显了在部署此类模型时，理解速度与准确性权衡的重要性。
    - drrck82 表达了对该方案的兴趣，尤其是作为双 RTX 3090 的持有者。他们提到目前使用 Q6_K_XL 模型以获得更高的智能水平，但发现 Qwen3.6-27B 模型翻倍的速度潜力非常有吸引力。这表明人们关注点在于计算效率与模型复杂程度之间的平衡。
    - DeepV 询问了该方案 Docker 化的可能性，表明了对容器化解决方案的需求，这些方案可以简化部署和扩展。Docker 化可以促进其更容易地集成到现有工作流中，并增强在不同环境下的可复现性。

  - **[致 16GB VRAM 用户：插上你的旧显卡吧](https://www.reddit.com/r/LocalLLaMA/comments/1swzjnu/to_16gb_vram_users_plug_in_your_old_gpu/)** (Activity: 666): **该帖子讨论了如何利用一块显存至少为 `6GB VRAM` 的旧显卡，配合一块主 `16GB VRAM` 显卡，以更高效地运行 `30b` 等稠密模型（dense models）。作者使用了一张 `5070Ti 16GB` 和一张旧的 `2060 6GB`，总计获得了 `22GB VRAM`，接近 `24GB` 级别显卡的容量。设置过程涉及使用带有特定配置的 `llama-server` 来优化 GPU 利用率，例如使用 `dev=Vulkan1,Vulkan2` 来启用多张 GPU，以及使用 `no-mmap` 将模型保持在显存而非 RAM 中。基准测试结果显示了显著的性能提升，在 `128k` 最大上下文下，提示词处理速度为 `186 tokens/s`，生成速度为 `19 tokens/s`，而单卡仅为 `4 tokens/s`。帖子还提供了使用 CUDA 的详细 `llama-bench` 结果，强调了将模型放入 GPU VRAM 以提高速度的重要性，尤其是在处理长上下文时。评论者讨论了 Vulkan 与 CUDA 的优劣，一些人建议使用 CUDA 以获得更好的性能。另一些人指出，虽然性能较弱的 GPU 提供的额外 VRAM 有所帮助，但它可能会成为强力 GPU 的瓶颈，例如在 `3090 Ti` 和 `2070` 的配置中，跨 GPU 拆分任务反而比单用 `3090 Ti` 的速度更慢。

    - **Mysterious_Role_8852** 讨论了同时使用 3090 Ti 和 2070 时的性能瓶颈。他们指出 2070 拖了 3090 的后腿，导致在 GPU 之间拆分任务时性能从 `30t/s` 下降到 `20t/s`。这强调了匹配 GPU 能力以避免性能下降的重要性，尤其是在处理像 Qwen 3.6 27b Q6 Quant 这样的大模型时。
    - **mac1e2** 详细描述了在硬件受限的系统（GTX 1650 4GB 和 62GB RAM）上运行 Qwen3.6-35B-A3B 的情况。他们强调了理解硬件限制和优化配置的重要性，例如使用 `--cpu-moe`、`--mlock` 以及特定的缓存设置。该帖子强调了严谨资源管理的价值，将其与现代通常依赖丰富的硬件资源而不去优化效率的做法形成了鲜明对比。
    - **jacek2023** 提到在三张 3090 之外，还使用了一张 3060 作为辅助 GPU，但仅用于最大的模型。这展示了一种战略性利用现有硬件的方法，即有选择地利用额外的 GPU 来最大化特定任务的性能，而不是盲目地不加区分地使用所有可用资源。

- **[在编写代码过程中从 Qwen3.6 35b-a3b 切换到 Qwen3.6 27b，效果明显更好！](https://www.reddit.com/r/LocalLLaMA/comments/1swifke/switched_from_qwen36_35ba3b_to_qwen36_27b_mid/)** (活跃度: 437): **该图片是一个使用 Qwen3.6 模型开发的塔防游戏的截图，特别强调了从 35b-a3b 版本到 27b 版本的过渡。尽管由于 VRAM 限制使用了压缩程度更高的 IQ3_M 版本，用户在使用 27b 模型时仍体验到了性能提升。这表明像 Qwen3.6-27B-i1-GGUF 这样的 Dense 模型比 MoE 模型能更好地处理压缩，证据是该模型能够识别出较大模型无法发现的疑难 Bug。用户报告 27b 模型的 Token 处理速度为每秒 40 tokens，相比 35b-a3b 模型波动的性能，它保持了稳定的速度。** 评论者普遍认为，对于 VRAM 有限的用户，像 Qwen3.6-27B 这样的 Dense 模型更高效、更可靠，并且他们赞赏这种无需依赖云端即可在本地使用的模型的可用性。一些用户已成功将这些模型用于正式工作，并指出其速度和 Context Window 令人满意。

    - 用户 'ridablellama' 强调了 Qwen3.6 27b 模型的实用价值，指出虽然它可能无法与 Claude Code 的性能相媲美，但仍能有效处理正式工作任务。他们强调该模型在速度和 Context Window 方面表现出色，并建议通过进一步的 Fine-tuning，其性能可能会有显著提升。这突显了该模型作为 16-24 GB VRAM 用户可靠基准的潜力，为云端解决方案提供了一个具有成本效益的替代方案。
    - 'YairHairNow' 对不同的 Qwen3.6 模型配置进行了详细对比，重点关注 Token 生成速度和 Context Length。27B IQ4_XS 模型因其支持最大 196K 的长上下文模式和每秒 48 tokens 的生成速度而受到关注。相比之下，35B-A3B Q3_K_S 配置提供了更快的 Token 生成速度（最高达每秒 149 tokens），但 Context Length 较短，约为 65K。这种对比对于在速度和上下文能力之间权衡的用户非常有价值。
    - 'KillerX629' 提到切换模型时有明显的降速，由于每秒生成的 tokens 减少，影响了他们的工作流。这突显了在模型选择中，性能速度与其他因素（如模型大小或上下文能力）之间常见的权衡。该评论建议用户需要根据具体需求和硬件能力来平衡这些方面。

  - **[Qwen3.6-27B-INT4 在单张 RTX 5090 上通过 vLLM 0.19 达到 100 tps 并支持 256k 上下文长度](https://www.reddit.com/r/LocalLLaMA/comments/1sw21op/qwen3627bint4_clocking_100_tps_with_256k_context/)** (活跃度: 426): **该帖子讨论了 **Qwen3.6-27B-INT4** 模型的性能，在单张 **RTX 5090** 上使用 **vLLM 0.19** 达到了 `105-108 tokens per second (tps)` 以及 `256k context length`。该模型可在 [Hugging Face](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) 上获取，受益于 **MTP 支持**和更小的体积，允许充分利用原生上下文窗口。该设置使用 `auto_round` 量化和 `fp8_e4m3` 作为 KV cache 数据类型，重点关注交互性和 Speculative Decoding。配置包括 `flashinfer` 注意力后端和带有 `3 speculative tokens` 的 `mtp` 推测配置。** 一个值得注意的评论提到了使用 **Turboquant 3-bit NC KV Cache** 来压缩 KV 状态，从而在 `24GB VRAM` 内实现 `125K context window`。**MTP n=3 Speculative Decoding** 因其吞吐量倍增器而受到称赞，而 **Cudagraph PIECEWISE Mode** 则因消除重复循环而被提及。通过 **Chunked Prefill** 和 **Prefix Caching**，该设置的效率得到进一步增强，在初始 Cudagraph 编译后稳定了请求时间。

- Qwen3.6-27B 模型在单张 RTX 5090 上实现了令人印象深刻的性能，叙事任务的持续吞吐量为 120-124 tokens per second (TPS)，代码任务为 156-159 TPS。这得益于 INT4 AutoRound 量化和 BF16 MTP head 保留，运行在带有 Genesis v7.0 补丁的 vLLM 0.19.2rc1 上。该配置利用了 258,048 token 的上下文窗口，接近 262,144 token 的架构最大值，并保持了 93% 的高 GPU 利用率，功耗为 400-426W。
- 使用 MTP n=3 的 speculative decoding（投机解码）显著增强了吞吐量，实现了 2.65-3.46 的平均接受长度和 55-82% 之间的接受率。该方法涉及使用三个辅助头在每次前向传递中生成草案 token，然后根据主头进行验证，与非投机基准相比，提供了约 3 倍的吞吐量乘数。这种技术对于在本地推理场景中保持高性能至关重要。
- Turboquant 的 3-bit NC KV Cache 是一个显著特征，允许将 KV 状态压缩到 3-bit 非均匀量化。这使得在 24GB VRAM 内实现 125K 上下文窗口而不会出现 OOM（显存溢出）。此外，使用仅捕获 attention-op 边界的 Cudagraph PIECEWISE 模式，有助于消除在多 GPU 主机上由于 FULL_AND_PIECEWISE 模式下过时的 MTP 状态引起的退化重复循环。

- **[在 Claude Code 中运行 Qwen 3.6](https://www.reddit.com/r/LocalLLM/comments/1swbh68/running_qwen_36_in_claude_code/)** (活跃度: 194): **用户尝试在配备 `RTX 4070` GPU (`8GB VRAM`) 和 `32GB RAM` 的系统上运行大型本地模型，如 *Qwen 3.6 27B* 和 *Gemma 4 26B*，但面临执行缓慢和无限循环等性能问题。像 *Qwen 2.5-coder 7B* 和 *Gemma 4 e4b* 这样较小的模型不足以胜任编码任务。有人建议使用 **Qwen3.6-35B-A3B**，这是一个 Mixture of Experts (MoE) 模型，允许将模型的一部分卸载到 GPU，其余部分保留在 RAM 中，在保持强劲性能的同时，有可能将速度提高 `2-3倍`。用户还考虑在 VSCode 中使用 Roo Code 作为 Claude Code 的替代方案，因为 Qwen 在后者中无法完成任务。** 评论者指出，本地 MoE 模型在处理涉及多个工具的复杂任务时可能会很吃力，原因是初始 Prompt 巨大且处理速度较慢 (`20tps`)。这可能导致超时和任务无法完成，这表明有效的本地模型执行需要充足的 VRAM (`48GB+`)。

- ghgi_ 建议使用 Qwen3.6-35B-A3B 模型，这是一种 Mixture of Experts (MoE) 模型。这允许将 3B 部分卸载到 GPU，同时将剩余权重保留在 RAM 中，与 Qwen 3.6 27B 模型相比，可能提供 2-3 倍的速度提升。尽管其性能略低于 27B 模型，但在大多数指标上仍具竞争力。
- OneSlash137 讨论了使用本地 MoE 模型的挑战，特别是在涉及多个工具的复杂任务中。他们指出，虽然模型可以处理简单的交互，但在处理大型 Prompt (10-20k tokens) 时会感到吃力，导致处理时间过长 (3-5 分钟) 且频繁超时。由于模型倾向于重新发送整个 Prompt 而不是增量更新，这一问题被进一步放大，导致了严重的延迟和低效。
- Plane-Pause-469 询问了在 Claude Code 中运行本地模型的情况，并寻求关于在 128 GB RAM 和 16GB VRAM 配置下使用哪些最佳模型的建议。这突显了用户在优化硬件资源以有效在本地运行大语言模型方面的共同关注。

### 2. 新模型与基准测试发布

  - **[微软发布 "TRELLIS.2"：一款开源、4b-Parameter、Image-To-3D 模型，可生成高达 1536³ 的 PBR 纹理资产，基于具有 16× 空间压缩的原生 3D VAEs 构建，提供高效、可扩展、高保真的资产生成。](https://www.reddit.com/r/LocalLLaMA/comments/1sxf2u0/microsoft_presents_trellis2_an_opensource/)** (热度: 376): **微软 (Microsoft)** 推出了 "TRELLIS.2"，这是一款尖端的 4b-Parameter 模型，用于从图像生成高保真 3D 资产。该模型利用了一种新颖的 'O-Voxel' 结构，能够创建具有锐利特征和完整 PBR 材质的复杂 3D 拓扑，通过 `16×` 空间压缩实现高达 `1536³` 的分辨率。该模型是开源的，代码可在 [GitHub](https://github.com/microsoft/TRELLIS.2) 获取，实时 Demo 托管在 [Hugging Face](https://huggingface.co/spaces/microsoft/TRELLIS.2)。关于 ROCm 支持存在技术争议，因为目前的文档主要提到 CUDA。一位用户报告了在 AMD 7800XT GPU 上运行该模型时出现的问题，遇到了可能由于依赖冲突和 ROCm overrides 导致的段错误 (segmentation faults)。

    - DeedleDumbDee 讨论了使用 ROCm 在 AMD 硬件上运行 TRELLIS.2 的挑战，指出模型文档主要支持 CUDA。他们提到在尝试于 7800XT GPU 上运行模型时遇到了段错误，而该模型目前仅在具有 24GB VRAM 的 NVIDIA GPUs 上进行过测试。这些问题可能是由于依赖冲突和使用 ROCm overrides 造成的。

  - **[AMD Hipfire - 一款针对 AMD GPUs 优化的新型推理引擎](https://www.reddit.com/r/LocalLLaMA/comments/1swpsv0/amd_hipfire_a_new_inference_engine_optimized_for/)** (热度: 426): **AMD Hipfire** 是一款针对 AMD GPUs 优化的新型推理引擎，采用独特的 `mq4 quantization` 方法。它并非由 AMD 官方授权，但显示出显著的性能提升，尤其是在 RDNA3 架构上。[Localmaxxing](https://www.localmaxxing.com/) 上的基准测试显示了大幅加速，一位用户报告在 RX 7900 XTX 上相比基准线实现了 `2.86×` 的加速。该引擎目前正在积极测试中，模型可在 [Hugging Face](https://huggingface.co/schuttdev) 获取。一些用户表示更倾向于使用 GGUF 等行业标准量化格式，认为这会简化采用流程。基准测试表明，虽然 **Hipfire** 在 AR 解码方面表现出色，但在 prefill 方面落后于 **llama.cpp**，其性能高度依赖于工作负载类型，尤其在结构化/代码生成任务中获益显著。

    - 用户 alphatrad 报告了在 RX 7900 XTX 上使用 AMD Hipfire 带来的显著性能提升，与 AR 基准相比实现了 `2.86×` 的加速且输出连贯。然而，他们指出实际应用可能与速度测试有所不同，特别是在编码任务中。
    - Own_Suspect5343 提供了 Hipfire 与 llama.cpp 在 AMD 硬件上的详细基准测试对比。Hipfire 的 AR 解码比 llama.cpp 快 `30%`，但 llama.cpp 在 prefill 任务中表现更佳。DFlash（一种投机解码方法）在结构化/代码生成任务中提供了大幅提速，在 merge_sort 提示词下实现了 `3.45x` 的加速，凸显了其性能对工作负载的依赖性。
    - FullstackSensei 表示倾向于在全行业采用 GGUF 进行模型量化，认为这将简化跨不同平台和模型的兼容性问题。这反映了对 AI 模型部署标准化的普遍需求。

## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude and GPT Image 2 创新

  - **[ChatGPT Image 2 的回归非常惊人](https://www.reddit.com/r/singularity/comments/1sw6q9j/the_comeback_chatgpt_did_with_image_2_is_insane/)** (热度: 1035): **该帖子对比了两张使用相同提示词生成的孟加拉国达卡布加迪威龙（Bugatti Chiron）的 AI 生成图像。第一张图由 **Nano Banana pro** 生成，第二张则由 **ChatGPT Image 2** 生成。后者因其真实感而受到关注，一位评论者称其“看起来像一张真实的照片”。然而，AI 的局限性在标牌文字上显而易见，文字呈现为孟加拉语和印地语脚本的混合体，突显了在准确渲染本地化文本方面的挑战。** 评论者们强调了 **ChatGPT Image 2** 输出结果的现实感，并注意到了它的照片级质量。然而，他们也指出 AI 未能准确描绘本地脚本，这降低了场景的真实性。


  - **[斯坦福大学研究人员向语言模型喂入 DNA 序列并要求其创建一种新病毒。它编写了数百种，其中 16 种有效。其中一种使用了地球上任何已知生物都不存在的蛋白质。](https://www.reddit.com/r/OpenAI/comments/1sw0vcf/stanford_researchers_fed_a_language_model_a_dna/)** (热度: 1080): **图像是一篇名为“利用基因组语言模型进行新型噬菌体的生成式设计”的研究论文，详细介绍了斯坦福大学研究人员使用 AI 设计新型噬菌体的研究。AI 生成的序列产生了 16 种具有活性的噬菌体，其中一种利用了在任何已知地球生物中都未发现的蛋白质。这项研究强调了 AI 在合成生物学中的潜力，特别是在设计可针对耐药细菌的噬菌体方面，同时也凸显了 AI 在生物工程中的前景与风险。** 评论者对 AI 在生物信息学中的双重用途性质表示担忧，指出它既有益处（如针对耐药细菌的靶向疗法），也存在潜在的危害，因为制造新型病毒的准入门槛相对较低。

    - Saotik 强调了 AI 在生物信息学中的双重用途，指出虽然 AI 生成的噬菌体可能带来针对耐药细菌的靶向疗法，但同样的技术也可能被用于制造危害人类的病毒。这凸显了在部署此类技术时进行谨慎监管和伦理考虑的必要性。
    - AI 驱动的生物信息学在潜在影响方面被比作核技术。评论者强调，一旦利用 AI 生成序列创建新型病毒的知识广泛传播，准入门槛就会变低，引发对滥用的担忧。这指向了在传播和应用该技术时进行严格控制和监督的迫切需求。
    - yamankara 澄清说，在这种语境下的“语言模型”是指“基因组语言模型”，而非通用的 Large Language Model (LLM)。这一区分对于理解用于生成新型病毒序列的模型的特定能力和应用至关重要。

  - **[Claude 4.7 通过一段 125 字的未发表作品识别出了一名记者](https://www.reddit.com/r/ClaudeAI/comments/1sw8npc/claude_47_named_a_journalist_from_125_words_of/)** (热度: 800): ****Kelsey Piper** 使用 **Claude 4.7** 进行了一项实验，她输入了 125 字未发表的文字，模型识别出了她的名字。她通过退出登录、使用 API 以及在朋友的笔记本电脑上进行测试来确保匿名性，排除了账号、浏览器和 IP 识别的可能性。该实验表明，**Claude 4.7** 可以从一小段文本样本中识别出作者独特的“声音（voice）”，这是 **ChatGPT** 或 **Gemini** 无法匹敌的能力。这表明模型有潜力将写作风格作为一种独特的“指纹”进行衡量和识别，突显了 **Claude 4.7** 卓越的阅读能力，尽管由于其对散文模式的深度编码，它在生成散文时可能不够灵活。[了解更多](https://www.theargumentmag.com/p/i-can-never-talk-to-an-ai-anonymously)。** 一些评论者质疑 Piper 方法论的技术严谨性，指出了潜在的疏忽，如使用 API 时需要账号登录等。其他人则将其与过去的语言分析实例联系起来，例如通过独特的词汇选择识别作者身份，认为写作风格可以像声纹一样具有辨识度。

### 2. DeepSeek 与 Qwen 模型讨论

  - **[DeepSeek 再次降价，输入 Token 缓存命中的价格降至当前水平的 1/10](https://www.reddit.com/r/DeepSeek/comments/1sw6y3c/deepseek_reduces_prices_again_the_price_for_input/)** (热度: 356): **DeepSeek** 宣布大幅降低输入 Token 缓存命中的价格，成本从之前的 `$0.145` 降至 `1/10`，即 `$0.0145`。此次降价是永久性的，与前一天提供的临时折扣不同。此举预计将使 **DeepSeek** 的服务更具成本效益，特别是对于需要 `1M` 上下文长度（context length）的应用，从而增强其在市场中的竞争优势。评论者对此次降价表示惊讶和赞赏，一些人将其归因于中国公司的战略性大方，而另一些人则强调了其对超长上下文应用的潜在影响。

    - DeepSeek 最近将输入 Token 缓存命中的价格从 $0.145 降至 $0.0145，标志着成本效率的显著提升，特别是在处理 1M 上下文长度时。这一举措可能使大规模模型的使用对于开发者和研究人员来说变得更加经济实惠。
    - DeepSeek Flash 因其性能而受到关注，在各种任务中与当前最先进（SOTA）的模型并驾齐驱。这表明 DeepSeek 不仅在价格上具有竞争力，在技术能力上也不逊色，使其成为高性能应用的可行选择。
    - DeepSeek 的大幅降价被视为一种战略举措，旨在使其服务近乎免费，通过以极低的成本提供高质量模型，可能会颠覆市场，从而鼓励更广泛的采用和实验。

  - **[DeepSeek V4 Pro（无折扣版）贵吗？](https://www.reddit.com/r/DeepSeek/comments/1sw80vi/is_deepseek_v4_pro_expensive_without_discounts/)** (热度: 183): **该图片是一个条形图，对比了运行各种 AI 模型的成本，突出显示 **DeepSeek V4 Pro** 的总成本为 `$1071`，与 **Claude Sonnet 4.6**（`$3959`）和 **GPT-5.4**（`$2851`）等其他模型相比处于中等水平。成本被细分为输入、推理和输出成本，其中 DeepSeek V4 Pro 的输入成本为 `$614`，推理成本为 `$420`。这表明虽然 DeepSeek V4 Pro 不是最贵的选项，但与 Claude 和 GPT 等订阅模式（可能便宜约 `50%`）相比，成本仍然很高。讨论还涉及订阅模式的不可预测性以及本地模型的潜在优势。** 一位评论者指出，DeepSeek V4 Pro 大约比 Mimo v2.5 Pro 贵 `2.4` 倍，而 Mimo 在智能基准测试中领先，且由于回复更简洁（lower verbosity）而具有成本优势。另一位评论者对“订阅费用仅为 API 使用费的 10%”这一说法表示怀疑，认为本地模型可能是更稳定的投资。

    - zoser69 提供了 DeepSeek v4 Pro 与 Mimo v2.5 Pro 之间的详细成本对比，指出 DeepSeek 贵了约 2.4 倍。在性能方面，Mimo 在智能（Intelligence）方面领先两个百分点，而 DeepSeek 在编程（Coding）方面略胜一筹。Mimo 较低的冗余度分数有助于其提高成本效益，且两种模型在 OpenRouter 上的响应速度相似，使得 Mimo 的时间效率更高。
    - Old_Stretch_3045 强调了订阅模式的不可预测性，认为由于服务限制和成本的潜在变化，依赖云端模型可能不可持续。他们提倡探索本地模型并升级硬件，作为一种更稳定的替代方案。
    - zoser69 列出了基于假设的前沿模型 10% 定价标准得出的各种模型预计成本。他们估算 DeepSeek v4 Pro 为 $661.2，而 GPT 5.5 和 Mimo v2.5 Pro 等其他模型的定价分别为 $225 和 $276。MiniMax m2.7 被认为是智能水平高于平均水平且最实惠的模型，即使没有开发者折扣，价格也仅为 $104.4。

### 3. 面向开发者的 AI 工具与框架


  - **[LTX2.3 在 Ostris AI toolkit 中使用 5090 训练 7 小时完成 ... 我学灭霸说：好吧，我亲自动手](https://www.reddit.com/r/StableDiffusion/comments/1swrs76/ltx23_in_ostris_ai_toolkit_on_a_5090_training/)** (Activity: 616): **该帖子详细介绍了在 NVIDIA 5090 GPU 上使用 Ostris AI toolkit 对 LTX2.3 模型进行自定义训练的过程，耗时 7 小时完成。关键调整包括初始将 `lora rank` 设置为 48，第一阶段使用 `600 steps`，并采用为 2 的 `gradient accumulation`。整个过程分为多个阶段，每个阶段针对 `differential guidance`、`learning rate` 和数据集配置（如 `512x512` 分辨率和每个片段 `25 frames`）进行了特定设置。作者强调了准确的 prompt 和 trigger words 对有效训练的重要性。** 评论者对仅通过 `1-second` 的片段就能捕捉神似的能力感到好奇，并索要训练中使用的 prompt 和 caption 示例。此外，还讨论了使用短视频片段训练 Loras 的可行性和效果。

    - **DateOk9511** 概述了使用 5090 GPU 训练 LTX2.3 的详细多阶段过程，强调了数据集组成和训练参数的重要性。该过程使用了 25 个视频片段，每个片段长 1 秒、帧率为 25 fps，并采用了 'low VRAM'、'lora rank' 和 'differential guidance' 等特定设置。训练分为四个阶段，每个阶段有不同的步数和设置，例如调整 'lora rank' 和 'differential guidance' 以优化性能和准确性。
    - **Disastrous-Agency675** 分享了一种在 3090 GPU 上训练 LTX2.3 时优化 VRAM 使用的技巧，即通过禁用 sampling，从而显著加快速度。通过开启 VRAM offload 和其他 low VRAM 设置，该方法可以在 6-7 小时内完成 7000 steps，为有效管理资源和提高训练速度提供了参考。
    - **Upper-Reflection7997** 报告了在配备 64GB RAM 的 5090 GPU 上进行 LTX2.3 LORA 训练的失败尝试，指出了设置中可能存在的挑战或配置错误。这突显了训练结果随硬件和配置变化的差异性，以及为了获得成功结果而进行精确参数微调的必要性。





# AI Discords

不幸的是，Discord 今天关闭了我们的访问权限。我们不会再以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢阅读到这里，这是一段美好的历程。