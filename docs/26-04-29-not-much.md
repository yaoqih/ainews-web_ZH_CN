---
companies:
- openai
- microsoft
- cursor_ai
- langchain-ai
date: '2026-04-29T05:44:39.731046Z'
description: '**OpenAI** 正在将 **Codex** 从单纯的编程工具扩展为包含持久上下文、工具、集成和团队部署功能的通用工作平台。针对商业和企业级客户，OpenAI
  将在 6 月前提供**席位费为 0 美元**的纯 Codex 席位。性能提升方面，重点聚焦于智能体循环（agent-loop）系统工程，通过 Responses
  API 的 WebSocket 模式，使智能体工作流的速度提升了高达 **40%**。


  **VS Code** 通过语义索引、跨仓库搜索、聊天会话洞察以及提示词/智能体评估扩展，增强了编程智能体的用户体验（UX）。**Cursor** 推出了 **Cursor
  SDK**，旨在为 CI/CD、自动化和嵌入式智能体提供可编程的智能体基础设施，标志着行业正向无头智能体运行时（headless agent runtimes）和按需计费的经济模式转变。


  研究方面，**智能体测试框架工程（Agentic Harness Engineering）**脱颖而出，将 Terminal-Bench 2 的 pass@1（一次性通过率）从
  **69.7% 提升至 77.0%**，超越了人工设计的基准，并减少了 **12%** 的 Token 使用量。关于 **HALO** 的相关研究展示了递归自我改进智能体在
  AppWorld 评分上的显著提升。**LangChain 的 Deep Agents** 则引入了 **Harness Profiles（测试框架配置文件）**，用于针对特定模型进行测试框架微调和部署。'
id: MjAyNS0x
models:
- codex
people:
- omarsar0
- samhogan
- kimmonismus
- reach_vb
- pierceboggan
title: 今天没发生什么事。
topics:
- agentic-harness-engineering
- agent-loop-systems-engineering
- performance-optimization
- semantic-indexing
- prompt-evaluation
- software-engineering
- sdk-development
- model-tuning
- recursive-self-improvement
---

**平淡的一天。**

> 2026年4月28日至4月29日的 AI 新闻。我们查阅了 12 个 Subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 且没有新增的 Discord 动态。[AINews 网站](https://news.smol.ai/) 允许您搜索所有往期内容。提醒一下，[AINews 现已成为 Latent Space 的一个板块](https://www.latent.space/p/2026)。您可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件发送频率！

---

# AI Twitter 综述

**Coding Agent 平台化：Codex、Cursor SDK 和 VS Code Harness 升级**

- **OpenAI 正在将 Codex 从一个编程工具转变为通用的工作平面**：今天最强劲的产品信号不仅是用户的使用热情，还包括围绕**持久化上下文 (persistent context)、工具、集成和团队部署**能力的稳步扩展。OpenAI 强调，除了代码之外，Codex 还能用于更广泛的知识工作任务，如研究综合、电子表格和决策跟踪 ([OpenAI](https://x.com/OpenAI/status/2049583167406064115), [后续](https://x.com/OpenAI/status/2049583308305252620), [后续](https://x.com/OpenAI/status/2049583379709124865))；为符合条件的商业/企业客户推出了截止 6 月底的 **0 美元席位费的 Codex 专用席位** ([OpenAIDevs](https://x.com/OpenAIDevs/status/2049505143218217048))；并新增了 **Supabase** 等集成 ([coreyching](https://x.com/coreyching/status/2049576335157416115))，以及一个能将执行计划转化为 FigJam 画布的 **Figma 插件** ([OpenAIDevs](https://x.com/OpenAIDevs/status/2049605820351230158))。社区帖子还指出了应用服务器的用法以及更丰富的 Agent 工作流 ([gdb](https://x.com/gdb/status/2049609076351381580), [aiDotEngineer](https://x.com/aiDotEngineer/status/2049527486124560491))。
- **性能优化的重点正从模型延迟转向 Agent-loop 系统工程**：OpenAI 表示，将 Codex 风格的工作流迁移到 **Responses API 的 WebSocket 模式**可以在工具调用之间保持状态热启动，减少重复劳动，使 **Agent 化工作流的速度提升高达 40%** ([OpenAIDevs](https://x.com/OpenAIDevs/status/2049595890395152728), [reach_vb](https://x.com/reach_vb/status/2049608607591809303), [pierceboggan](https://x.com/pierceboggan/status/2049505637978263697))。VS Code 发布了一系列并行的 Harness 改进：**跨工作区的语义索引**、跨仓库搜索、**对话会话洞察**、**技能上下文**、Copilot CLI 的远程控制，以及一个旨在优化 Prompt、技能和指令的 Prompt/Agent 评估扩展 ([pierceboggan](https://x.com/pierceboggan/status/2049504445424423133), [pierceboggan](https://x.com/pierceboggan/status/2049503967059812617), [code](https://x.com/code/status/2049556204930695278))。其核心逻辑在于，Coding Agent 的用户体验现在由内存、检索、Harness 质量和工具编排主导，而不仅仅是原始的模型智能。
- **Cursor 正在明确向平台化转型**：全新的 **Cursor SDK** 暴露了为 Cursor 提供支持的相同运行时 (runtime)、Harness 和模型，以便在 **CI/CD、自动化和产品内置的嵌入式 Agent** 中使用 ([cursor_ai](https://x.com/cursor_ai/status/2049499866217185492), [入门项目](https://x.com/cursor_ai/status/2049499874043830389), [客户案例](https://x.com/cursor_ai/status/2049499876388454903))。这一点值得注意，因为它将 Cursor 从基于席位的 IDE 产品转向了可编程的 Agent 基础设施，[@kimmonismus](https://x.com/kimmonismus/status/2049514922044792934) 很好地捕捉到了这一框架。结合 Codex 应用服务器和 VS Code Harness 的工作，该类别显然正向 **无头 (headless) Agent 运行时 + 可编程 Harness + 基于用量的经济模式** 趋同。

**Agent Harness 工程、LangGraph/Deep Agent 以及生产级 AgentOps**

- **Harness 正在成为一等公民优化层**：多篇文章一致认为，仅靠模型质量是不够的；模型周边的 Harness 往往决定了生产环境的性能。最清晰的研究案例是 **Agentic Harness Engineering**，它通过可回滚组件、压缩的执行证据和可证伪的预测，使 Harness 的演进变得可观察。据报告，在 10 次迭代中，**Terminal-Bench 2 pass@1 从 69.7% 提升至 77.0%**，超过了人类设计的 **71.9%** 的 Codex-CLI 基准，同时还能在不同模型家族间迁移，并将 SWE-bench Verified 上的 Token 使用量减少了 **12%** ([omarsar0](https://x.com/omarsar0/status/2049492169887748365))。关于 **HALO** 的相关工作描述了使用 Trace 分析来修补 Harness 故障的递归自改进 Agent，声称在 Sonnet 4.6 上将 **AppWorld** 的表现从 **73.7 提升至 89.5** ([samhogan](https://x.com/samhogan/status/2049619541727302040))。
- **LangChain 的 Deep Agents 产品线正专注于特定模型的 Harness 调优和可部署性**：新的 **Harness Profiles** 允许团队对每个模型的 Prompt、工具和中间件进行版本管理，并内置了适用于 OpenAI, Anthropic 和 Google 模型的配置文件 ([LangChain_OSS](https://x.com/LangChain_OSS/status/2049539590990557381), [LangChain](https://x.com/LangChain/status/2049540926603718969), [Vtrivedy10](https://x.com/Vtrivedy10/status/2049537545273528633))。LangChain 还推出了 **DeepAgents Deploy**，这是一种使用少量 Markdown/配置文件和基于 LangSmith 的 Trace 追踪的低代码部署路径 ([hwchase17](https://x.com/hwchase17/status/2049546041247289553))。LangChain 员工传达的更广泛信息非常明确：**开放的 Harness、开放的 Eval 以及对 OSS 友好的模型组合**至关重要，因为对于许多 Agent 工作负载来说，闭源模型正变得过于昂贵 ([hwchase17](https://x.com/hwchase17/status/2049552801890771220), [Vtrivedy10](https://x.com/Vtrivedy10/status/2049597811226726682))。
- **Cloudflare** 继续完善其“Agent 即软件”技术栈，提出了执行阶梯等理念，更具体地说，是让 Agent 能够成为 **Cloudflare 客户**——创建账户、注册域名、启动付费计划并获取部署 Token ([threepointone](https://x.com/threepointone/status/2049463167298777310), [Cloudflare](https://x.com/Cloudflare/status/2049545195914498139))。这是一个重要的信号，表明厂商正开始直接向 Agent 开放业务工作流，而不是仅仅将其视为被动的 Copilot。

**模型发布与基准测试：Mistral Medium 3.5, Granite 4.1, Ling-2.6 以及开源模型的价格压力**

- **Mistral Medium 3.5** 是当天讨论最激烈的模型发布。早期评论将其定位为 **稠密 128B** 模型 ([scaling01](https://x.com/scaling01/status/2049508126081077678))，Unsloth 将其描述为一款可以在约 **64GB RAM** 上本地运行的 **视觉推理模型**，并发布了 GGUFs/指南 ([UnslothAI](https://x.com/UnslothAI/status/2049511248623256017))。反应分化严重：一些人批评其 **128K 上下文**、架构选择以及与中国大型开源 MoE 相比的定价 ([eliebakouch](https://x.com/eliebakouch/status/2049523829358162027), [scaling01](https://x.com/scaling01/status/2049546078664397105))，而另一些人则认为 Mistral 是在进行一场深思熟虑的 **企业级可靠性/指令遵循** 押注，而非追求单纯的基准测试数据 ([kimmonismus](https://x.com/kimmonismus/status/2049545016784413005))。
- **IBM Granite 4.1** 增加了三个新的 **Apache 2.0 协议开放权重** 非推理模型——**30B、8B、3B**——并强调开放性和 Token 效率 ([ArtificialAnlys](https://x.com/ArtificialAnlys/status/2049505499377193156))。最引人注目的观点是，**Granite 4.1 8B** 在 Artificial Analysis Intelligence Index 上仅使用了 **400 万输出 Token**，而 **Qwen3.5 9B** 则使用了 **7800 万**，同时在 AA Openness Index 上得分为 **61**。虽然智能水平落后于更强大的同行，但该系列模型显然瞄准了 **成本和透明度** 比排行榜名次更重要的企业/边缘部署场景。
- **开放权重模型的竞争压力持续加剧**：蚂蚁开源的 **Ling-2.6-flash** 被引述为约 **107B MoE**，采用 **MIT 许可**，**SWE-bench Verified** 得分为 **61.2** 并拥有强劲的数学成绩 ([nathanhabib1011](https://x.com/nathanhabib1011/status/2049466639171690820))；**Ling-2.6-1T** 也已发布并提供首日 **vLLM** 支持 ([vllm_project](https://x.com/vllm_project/status/2049517056299761925))。与此同时，**Tencent Hunyuan** 开源了 **Hy-MT1.5-1.8B-1.25bit**，这是一个 **440MB**、完全离线的手机端翻译模型，覆盖 **33 种语言**、**1,056 个翻译方向**，并声称通过激进的 **1.25-bit quantization**，在标准 MT 基准测试中达到了与商业 API 或 235B 规模模型持平的水平 ([TencentHunyuan](https://x.com/TencentHunyuan/status/2049487799850840334))。在市场端，多篇帖子强调了高性能开源模型价格下降的速度，例如 **Qwen 3.5 Plus 每百万输出 Token 仅需 $3** ([MatthewBerman](https://x.com/MatthewBerman/status/2049562998575075526))，以及 **MiMo-V2.5 Pro** 在 Code Arena 中以 **每百万 Token $1/$3** 的价格移动了帕累托前沿 ([arena](https://x.com/arena/status/2049582973926949116))。

**推理、内核与 MoE 系统：FlashQLA、Blackwell 上的 vLLM、torch.compile 以及 GLM-5 Serving**

- **Qwen 的 FlashQLA 是一个值得关注的长上下文算子（kernel）发布**：阿里巴巴推出了 **FlashQLA**，这是在 TileLang 上实现的高性能线性注意力（linear attention）算子，据报道在前向传播（**forward**）中提速 **2–3 倍**，在反向传播（**backward**）中提速 **2 倍**，尤其适用于**小模型、长上下文工作负载和张量并行（tensor-parallel）配置**。其设计核心在于门控驱动的自动卡内 CP（intra-card CP）、代数重构以及融合的 warp-specialized 算子（[Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2049462666734026923), [benchmark thread](https://x.com/Alibaba_Qwen/status/2049462776247247310)）。它明确定位于**个人设备上的 Agent 级 AI**，这符合长上下文优化从纯云端架构向边缘友好运行环境迁移的大趋势。
- **vLLM 与 Blackwell 的协同设计正在赢得实质性的吞吐量优势**：vLLM 在 Artificial Analysis 上报告了 **DeepSeek V3.2** 的输出速度排名第一，达到了 **230 tok/s，0.96s TTFT**；在使用 **NVIDIA HGX B300 的 DigitalOcean serverless 推理**中，**Qwen 3.5 397B** 也取得了强劲的表现。优化措施包括 **NVFP4 量化**、**EAGLE3 + MTP 投机解码（speculative decoding）**以及**针对特定模型的算子融合（kernel fusion）**（[vllm_project](https://x.com/vllm_project/status/2049503979898274163)）。SemiAnalysis 另外强调了在 GB200 上运行 DeepSeek v4 Pro 时，**vLLM 0.20.0** 和 **MegaMoE** 算子带来的性能提升（[SemiAnalysis_](https://x.com/SemiAnalysis_/status/2049578313111216271)）。这是硬件/软件/模型协同设计转化为公开可见延迟指标的一个清晰案例。
- **越来越多的工程师正在分享模型与 GPU 之间的“中间层”细节**：关于 **torch.compile** 的一个有用讨论拆解了从 Dynamo → pre-grad → AOT autograd → post-grad → Inductor 的过程，包括在何处注入自定义 FX passes 以进行推理优化（[maharshii](https://x.com/maharshii/status/2049402475476861044)）。John Carmack 发文提醒，GPU 库的性能仍然极度**依赖路径且存在显著波动（notchy）**，并指出在从 **511×511 增加到 512×512** 时，`torch.linalg.solve_ex` 出现了 **10 倍的性能回退**，显然是由于涉及 `CudaMalloc/Free` 的不同内部路径（[ID_AA_Carmack](https://x.com/ID_AA_Carmack/status/2049467648900018281), [follow-up](https://x.com/ID_AA_Carmack/status/2049528611544207714)）。智谱 AI 也发布了一篇关于 **GLM-5** 服务复盘的优秀文章，详细描述了 **KV cache 竞态条件**、HiCache 同步 bug 以及 **LayerSplit**。据报道，在长上下文 coding-agent 服务中，LayerSplit 将 prefill 吞吐量提高了多达 **132%**（[Zai_org](https://x.com/Zai_org/status/2049601030170857891)）。

**研究信号：知识探测（Knowledge Probes）、Web-Agent 基准测试、多模态/科学基础设施**

- **Incompressible Knowledge Probes (IKP)** 是最引人注目的研究方向之一：[@bojie_li](https://x.com/bojie_li/status/2049314403208896521) 声称，针对 **1,400 个问题 / 188 个模型 / 27 个厂商** 的事实知识准确性呈现出强烈的模型规模对数线性信号（在 **135M 到 1.6T params** 的 **open-weight** 模型上 **R² = 0.917**）。该论文认为，事实容量并不会像某些“推理压缩”叙事所暗示的那样**随时间压缩**，并利用拟合曲线来估算 **closed-model** 的规模。无论你是否认可这些估算，这项工作都具有价值，它提醒我们 **black-box evals** 仍然会泄露架构规模信息。
- **Web-agent 评估正超越简单的通过/失败模式走向成熟**：新的 **Odysseys** 基准测试引入了 **200 个长程实时互联网任务**，采用基于评分量表（rubric-based）的评估而非二元成功判定，并引入了**轨迹效率（trajectory efficiency）**指标。据报告，最佳模型的成功率仅为 **44.5%**，效率仍极低，仅为 **1.15%** ([rsalakhu](https://x.com/rsalakhu/status/2049521211353301198), [dan_fried](https://x.com/dan_fried/status/2049530695739932876))。这符合行业的大趋势，即推动 Agent 基准测试更好地反映多步浏览、表格处理和编排工作，而非简单的合成任务。
- **AI-for-science 和多模态基础设施迎来了重要的生态发布**：Hugging Face 推出了 **Hugging Science**，这是一个专门存放开放科学数据集/模型/挑战的平台，包含 **78GB 基因组学**、**11TB PDE 模拟**、**100M 细胞图谱**、**9T DNA 碱基对**等内容 ([cgeorgiaw](https://x.com/cgeorgiaw/status/2049506162442129731))。Anthropic 发布了 **BioMysteryBench**，报告称最近的 Claude 模型解决了约 **30%** 令专家也感到棘手的生物数据分析难题 ([AnthropicAI](https://x.com/AnthropicAI/status/2049624600741560340))。在多模态方面，**Vista4D** 引入了利用持久化 4D 场景表示从新相机轨迹进行视频“重拍（reshooting）”的技术 ([micahgoldblum](https://x.com/micahgoldblum/status/2049613850912113077))，而 Sakana 的 **KAME** 为语音对语音（speech-to-speech）系统提出了一种“**边思边说（speak while thinking）**”的串联架构，将低延迟前端模型与异步后端 LLM oracle 信号相结合 ([SakanaAILabs](https://x.com/SakanaAILabs/status/2049544945233764755))。

**Top Tweets (按参与度排名)**

- **Cursor SDK 发布**：面向 CI、自动化和嵌入式产品的可编程 Agent 运行时/测试框架/模型 ([cursor_ai](https://x.com/cursor_ai/status/2049499866217185492))。
- **Codex 势头 / 平台扩张**：OpenAI 正在将 Codex 从编程领域推向更广泛的工作自动化，并推出了团队版及相关集成 ([OpenAI](https://x.com/OpenAI/status/2049583167406064115), [OpenAIDevs](https://x.com/OpenAIDevs/status/2049505143218217048))。
- **Google 产品化信号**：Gemini 现在可以直接从对话中生成可下载的 Docs, Sheets, Slides, PDFs 等文件 ([sundarpichai](https://x.com/sundarpichai/status/2049519281600373159), [GeminiApp](https://x.com/GeminiApp/status/2049519416698683514))。
- **Q1 业务信号**：Google 报告 **Cloud 同比增长 63%**，Gemini 势头强劲，Search 查询量创历史新高，这是“AI 变现（AI monetization）”论点的重要数据支撑 ([sundarpichai](https://x.com/sundarpichai/status/2049581838260461916))。
- **深度技术长篇内容**：Dwarkesh 与 Reiner Pope 的板书交流环节，探讨如何根据价格、公式和系统约束来推断训练/推理策略 ([dwarkesh_sp](https://x.com/dwarkesh_sp/status/2049551656816439604))。

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM 回顾

### 1. Mistral Medium 3.5 模型发布与特性

  - **[mistralai/Mistral-Medium-3.5-128B · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1sz1qer/mistralaimistralmedium35128b_hugging_face/)** (热度: 921): **Mistral Medium 3.5** 是一个稠密型（dense）`128B` 参数模型，具有 `256k` 上下文窗口，专为指令遵循、推理和编码任务设计。它具有可配置的推理强度（reasoning effort）、多模态输入能力，并在各项基准测试中表现强劲，超越了之前的 Devstral 等模型。该模型在 **Modified MIT License** 下开源，支持多种语言和系统提示（system prompts）。为了获得最佳性能，建议使用 vLLM 库进行推理。更多详情可以在 [这里](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) 找到。一位评论者正在 Strix Halo 上使用 `q4` 量化版本测试该模型，报告了 Token 生成速度，并对该模型的稠密架构表示出浓厚兴趣。另一条评论强调了该模型作为 `128B` 稠密模型的生态位，并将其与 Qwen 27B 进行了比较。

    - IvGranite 分享了 Mistral-Medium-3.5-128B 模型在 Strix Halo 设备上使用 `q4` 量化后的性能指标。结果显示生成速度为 `46.70 tokens per second`，提示词处理速度为 `3.26 tokens per second`，其中一项测试的总耗时为 `4.84 seconds`。这表明对于这种规模的稠密模型，其吞吐量相对较高。
    - Grumd 和 reto-wyss 讨论了稠密模型的定位，grumd 指出了 `128B` 稠密模型的独特性。Reto-wyss 将其与 Qwen `27B` 模型进行了对比，质疑哪个模型更“稠密”，凸显了模型密度和性能方面的竞争态势。
    - 围绕 Mistral-Medium-3.5-128B 等稠密模型的讨论反映了人们在平衡模型规模与性能效率方面的兴趣。artisticMink 将 `128B` 称为“大块头”（chonker），强调了处理此类大规模模型在计算资源和速度方面的挑战与吸引力。

  - **[Mistral Medium 3.5 Launched](https://www.reddit.com/r/LocalLLaMA/comments/1sz2mgw/mistral_medium_35_launched/)** (热度: 326): **Mistral Medium 3.5** 已作为 `128B` 稠密模型发布，因其整合了指令遵循、推理和编码能力而备受关注。该模型以开放权重形式提供，采用修改后的 MIT 许可证，限制了未经许可付费的商业使用。该模型支持云端的异步编码任务，允许并行会话执行，并在 Le Chat 中为复杂工作流引入了新的 Work 模式。更多详情请参见 [Hugging Face](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) 和 [Mistral 的公告](https://mistral.ai/news/vibe-remote-agents-mistral-medium-3-5)。用户对许可条款存在争议，一些人认为称其为“修改后的 MIT 许可证”具有误导性，因为它施加了 MIT 许可证中不常见的商业限制。此外，用户还讨论了该模型的参数量和能力，指出 `128B` 稠密架构意味着巨大的计算资源需求。

    - Mistral Medium 3.5 模型是一个 1280 亿参数的稠密模型，鉴于目前向更大稠密模型发展的趋势，这一点意义重大。正如 Septerium 所指出的，尽管行业目前的重心在稀疏模型上，但持续投入开发稠密架构依然至关重要。
    - Long_comment_san 讨论了 Mistral Medium 3.5 的基准测试，指出虽然它可能不是当前最先进（SOTA）的模型，但对稠密模型的未来发展至关重要。他们认为 80B+ 参数范围的稠密模型是核心主力，并预见到未来超稀疏的 Mixture of Experts (MOE) 模型将与参数量高达 200B 的超稠密模型共存。
    - ClearApartment2627 提出了许可问题，批评 Mistral Medium 3.5 使用“修改后的 MIT 许可证”。他们认为这种说法具有误导性，因为其商业使用条件与传统的 MIT 许可证大相径庭，特别是对于月收入超过 2000 万美元的公司。


### 2. Qwen 3.6 模型评估与特性

- **[Qwen 3.6 27B BF16 vs Q4_K_M vs Q8_0 GGUF 评估](https://www.reddit.com/r/LocalLLaMA/comments/1sxzqry/qwen_36_27b_bf16_vs_q4_k_m_vs_q8_0_gguf_evaluation/)** (热度: 995): **该图片展示了 Qwen 3.6 27B 模型在三种量化变体（BF16, Q4_K_M 和 Q8_0 GGUF）下的基准测试对比，评估使用了带有 Neo AI Engineer 的 llama-cpp-python。基准测试包括用于代码生成的 HumanEval、用于常识推理的 HellaSwag 和用于函数调用的 BFCL。Q4_K_M 变体在实际性能上脱颖而出，其吞吐量比 BF16 快 1.45 倍，峰值 RAM 占用减少了 48%，模型大小减小了 68.8%，同时保持了几乎相同的函数调用评分。然而，Q8_0 尽管在 HumanEval 评分上略高，但在 RAM 使用和速度方面的效率不如 Q4_K_M。评估设置包括通过 llama-cpp-python 使用 GGUF，上下文大小（Context Size）为 32768，并进行了检查点评估运行。** 一些评论者赞赏对不同量化变体的详细对比，而另一些人则质疑结果的准确性，指出缺乏误差线（Error Bars）并暗示可能存在采样误差。此外，还有人担心 Qwen 3.6 27B 的 HumanEval 分数低得离谱，甚至不如 Gemma 3 4B 和 Llama3-8b 等旧模型。

    - audioen 对测量中缺乏误差线表示担忧，认为 Q4_K_M 优于 Q8_0 的意外排序可能是由采样误差造成的。这强调了在基准测试中统计严谨性的重要性，以确保量化方法之间比较的可靠性。
    - One_Key_8127 指出了报告的 HumanEval 分数存在差异，注意到 Gemma 3 4B 和 Llama3-8b 等旧模型的表现优于 Qwen 3.6 27B，而理论上后者应该得分更高。这表明评估设置或数据可能存在问题，因为 Qwen 3.6 27B 预期能达到 85% 或更高的分数，而非 50% 左右。
    - spaceman_ 质疑 Q8_0 模型结果的完整性，推测 KV cache 的量化可能影响了性能。他们对用于评估的完整代码表示感兴趣，因为这可以揭示 KV cache 是否确实被量化，从而解释这种出人意料的结果。

  - **[Qwen 推出 FlashQLA](https://www.reddit.com/r/LocalLLaMA/comments/1syx4sg/qwen_introduced_flashqla/)** (热度: 407): ****FlashQLA** 是一种专为个人设备上的 Agentic AI 设计的新型高性能线性注意力（Linear Attention）Kernel，可提供 `2–3×` 的前向加速和 `2×` 的后向加速。它基于 **TileLang** 构建，具有 Gate 驱动的自动卡内上下文并行（CP）、硬件友好型代数重构以及 TileLang 融合的 Warp 专业化 Kernel。该方法将 GDN flow 拆分为两个针对 CP 和后向效率优化的 Kernel，尽管在大 Batch Size 下会有额外的内存 I/O 开销，但它增强了边缘设备和长上下文工作负载下的实际性能。后向传播通过 16 阶段 Warp 专业化流水线进行了显著优化，实现了 `2×+` 的 Kernel 级加速。更多详情可以在其 [博客](https://qwen.ai/blog?id=flashqla) 和 [代码仓库](https://github.com/QwenLM/FlashQLA) 中找到。** 一条评论幽默地提到了“赛博朋克”的缩写，而另一条评论则建议该技术适合拥有 H100 等高端硬件的用户。人们对常见配置下的前向和后向基准测试结果也表现出了兴趣。

    - ResearchCrafty1804 讨论了 FlashQLA 的基准测试结果，强调了在常用配置下的前向和后向性能。这表明人们关注于评估模型在不同计算场景下的效率，这对于理解其实际应用和局限性至关重要。
    - pmttyji 提供了运行 FlashQLA 的详细技术要求清单，包括需要 SM90 或更高版本 GPU、CUDA 12.8 或更高版本以及 PyTorch 2.8 或更高版本。这些规范指出了有效利用 FlashQLA 功能所需的先进硬件和软件环境。
    - LightBrightLeftRight 暗示了在 H100 等高性能硬件上本地部署 FlashQLA 的潜力，建议拥有此类资源的用户可以尝试本地运行该模型，从而可能实现更多定制化和优化的实现。

- **[本地运行 Qwen 3.6 或 Gemma 4 是什么样的体验](https://www.reddit.com/r/LocalLLaMA/comments/1syt38w/what_it_feels_like_to_have_to_have_qwen_36_or/)** (活跃度: 766): **这张图片是一个 meme，幽默地传达了在本地运行像 Qwen 3.6 或 Gemma 4 这样先进的 AI 模型时，那种获得赋能和掌控力的感觉。该帖讨论了这些模型在专业场景中的实际应用，强调了它们在执行传统上需要人类专业知识的专家级任务时的效率和能力。图片隐喻地暗示，拥有这些强大的模型在手，就像掌握了巨大的能量，仿佛“将太阳的力量握在掌心”。** 评论者强调了 Gemma 4 在翻译和创意写作方面的效果，以及 Qwen 3.6 在游戏开发方面的表现。人们对 AI 能力的快速进步感到一种怀旧感，并将其与 90 年代游戏行业的飞速发展相提并论。另一条评论建议使用针对特定任务微调的模型，如 Granite 和 Nemotron，以获得高性价比和高效的性能。

    - **Qwen 3.6** 因其在通宵运行 Agent 时的稳定性和效率而受到关注，没有出现错误或死循环，这与之前的模型相比是一个显著的进步。这表明它在任务处理和决策过程中表现稳健，使其在长期运行中非常可靠。
    - **Gemma 4** 在翻译和创意写作方面表现出色，显示了其在自然语言处理任务中的实力。提到 Qwen 3.6 在游戏开发方面的能力，突显了它的多功能性和效率，特别是在创建网页游戏方面，对于一个小模型来说，这非常令人印象深刻。
    - 关于 **针对特定任务微调的模型**（如 Granite 和 Nemotron）的讨论表明，它们能以更低的成本超越更大的模型。这些模型可以根据需求加载，并通过 Agent 编排器进行管理，提供了部署的灵活性和效率，这在特定应用场景中非常有优势。

### 3. 本地 LLM 硬件与使用体验


  - **[我受够了用本地 LLM 来编程](https://www.reddit.com/r/LocalLLaMA/comments/1sxqa2c/im_done_with_using_local_llms_for_coding/)** (活跃度: 2387): **该用户将 Qwen 27B 和 Gemma 4 31B 等本地 LLM 与 Claude Code 在编程任务中进行了对比，特别是在 OS/Docker 环境下。他们发现本地模型在决策和 Tool-calling 能力方面表现不足，经常无法高效完成如将 GitHub 仓库 Docker 化之类的任务。用户指出，本地 LLM 会读取来自 `docker build` 等命令的过量输出，导致会话因达到 `250k input tokens` 而中断。性能也是一个问题，频繁的 Prompt Cache 失败导致长时间的停顿。用户得出结论，与 OpenRouter 和 Kimi 等云端模型相比，本地 LLM 在编程任务上的生产力损失是不值得的，尽管他们仍然认为本地模型在自动化和纯文本任务中很有用。** 评论者们也表达了类似的本地 LLM 使用体验，认为预期可能过于理想化。一位评论者强调了优化性能设置的重要性，例如 [Unsloth 指南](https://unsloth.ai/docs/basics/claude-code#fixing-90-slower-inference-in-claude-code)中提到的方法。另一位则强调了配套技术栈的重要性，详细介绍了一套包含 **RTX 5090**、**Qwen3.6 35B/27B** 以及 **OpenCode TUI** 和 **oh-my-opencode harness** 等工具的配置，以提升性能。

    - 一位用户强调了为 Claude Code 等本地模型优化设置以提高性能的重要性。他们引用了 [Unsloth](https://unsloth.ai/docs/basics/claude-code#fixing-90-slower-inference-in-claude-code) 上的一份指南，该指南解决了推理缓慢和缓存失效等问题，表明正确的配置可以显著提升可用性。
    - 另一位评论者强调了运行本地模型时技术栈的关键作用，并详细介绍了他们自己的配置，包括 RTX 5090 和带有 TurboQuant 的 Qwen3.6 模型。他们使用了特定的参数，如 `--temperature 0.6` 和 `--top-p 0.95`，以及包含 OpenCode TUI 和各种 MCP 的编程工具栈。据称，这套配置的表现优于 Anti-Gravity 和 Codex 等中心化解决方案。
    - 关于 Harness 在本地 LLM 性能中重要性的讨论表明，即使使用相同的模型，不同的 Harness 也会导致截然不同的结果。评论者指出，像 Hermes 这样的 Harness 具有特定的优缺点，例如处理长时间运行的进程。他们主张尝试各种 Harness 以找到最适合特定任务的工具，并指出 Harness 设计是未来改进的关键领域。

  - **[16x DGX Sparks - 我该运行什么？](https://www.reddit.com/r/LocalLLaMA/comments/1sz0lyk/16x_dgx_sparks_what_should_i_run/)** (活跃度: 1621): **图片展示了一个包含 16 台 NVIDIA DGX Spark 单元的家庭实验室配置，旨在组建大规模 DGX Spark Cluster。该配置包括一个 200Gbps FS 交换机和 QSFP56 DAC 线缆，暗示了一个高性能计算环境。用户正在寻求关于在这个拥有 2TB 统一内存的强大集群上运行哪些应用或工作负载的建议。社区的建议包括使用 vLLM 运行 Kimi K2.6、利用 eugr 的 nightly build，以及考虑为 vLLM 准备的 Deepseek V4 未合并 PR。预计该配置将提供极高的 Prefill 数值，尽管 Token 生成速度可能限制在每秒 20 个 Token。** 一位评论者建议卖掉 DGX Spark 改买 H100，暗示 H100 在某些工作负载下可能提供更好的性能或价值。

    - yammering 讨论了在带有 vLLM 的八节点集群上运行 Kimi K2.6 的性能，并指出使用 eugr 的 nightly build 可以提升性能。他们提到了 vLLM 中关于 Deepseek V4 的未合并 Pull Request，暗示了潜在的改进空间。他们还强调，虽然 Flash 版本在 8x 节点上运行良好，但 Pro 版本可以利用全部 16 个节点，实现极高的 Prefill 数值，但 Token 生成速度平均为每秒 20 个 Token。


## 非技术性 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude and Blender 集成

  - **[初级创意自由职业者的末日已至](https://www.reddit.com/r/ClaudeAI/comments/1syu949/the_final_nail_in_the_coffin_for_entry_level/)** (热度: 708): **Anthropic** 发布了 Blender MCP 连接器，使 **Claude** 能够通过 Python API 控制 Blender。这种集成允许用户使用自然语言命令创建和修改 3D 场景，实际上在 Blender 中充当了“copilot”。该工具可以处理调试节点设置、批量修改和添加自定义工具等任务，可能会减少产品渲染和低多边形（low-poly）资产创建等任务对初级自由职业者的需求。现在的整个创意管线可以由单个用户通过 Claude 和连接的工具进行管理，从而简化从剧本创作到最终剪辑的流程。一些评论者对输出质量表示怀疑，指出虽然自动化可能会增加产量，但并不一定会提高质量，正如在其他拥有自动化工具的行业中所见的那样。

    - poponis 认为，虽然 AI 工具可以辅助创意过程，但它们并不能保证高质量的输出。该评论者强调，AI 生成的内容通常需要人类专业知识来提炼和改进，特别是在技术知识至关重要的编码等领域。他们认为 AI 取代人类角色的说法被夸大了，AI 应该被视为增强而非取代人类创造力的工具。

  - **[Claude 现已连接至 Blender](https://www.reddit.com/r/ClaudeAI/comments/1sy49oi/claude_now_connects_to_blender/)** (热度: 605): **Claude**（由 **Anthropic** 开发的 AI 模型）现在通过一个新的连接器与 **Blender** 集成，允许用户直接从 Claude 调试场景、构建工具和批量应用更改。这种集成利用了 Blender 的 Python API，支持创建几何体和材质等高级操作。该连接器可以通过 Claude 桌面应用中的 Connectors Directory 添加，从而提高创意专业人士的工作流效率。[Blender](https://www.blender.org/press/anthropic-joins-the-blender-development-fund-as-corporate-patron/) 最近宣布 Anthropic 以企业赞助商（corporate patron）身份加入了其发展基金，贡献了至少 `$280k`。评论者强调这种集成对 Blender 用户来说是显著的使用体验提升，特别是在管理复杂场景方面。还有人推测，由于 Blender 的 Python API 功能广泛，可能会导致较高的 token 使用量。

    - Ciabattabingo 指出，Anthropic 已作为企业赞助商加入 Blender 发展基金，这涉及重大的财务承诺（可能为 28 万美元）。这种伙伴关系可能会加强 Blender 的开发，为赞助商提供专门的产品经理，并使其更紧密地参与资金决策。Claude 与 Blender 的集成可以通过利用 Claude 的能力来实现更高效的工作流，从而简化内容生产。
    - jj2446 指出了 Claude 与 Blender 集成的潜力，强调了在管理复杂场景方面的使用体验提升。通过访问 Blender 的 Python API，Claude 可以自动执行创建几何体和材质等任务，显著提高资深 Blender 用户的工作效率。
    - mikeb550 询问了是否可以直接使用 Claude 提示词来创建 3D 模型。这暗示了一个潜在的功能，即用户可以利用 Claude 的 AI 能力来生成模型，这将是简化 3D 建模工作流的重大进步。


### 2. Talkie: 1931 年前的语言模型

  - **[Talkie，一个完全基于 1931 年前数据训练的 13B LM](https://www.reddit.com/r/singularity/comments/1sxp4ha/talkie_a_13b_lm_trained_exclusively_on_pre1931/)** (热度: 3160): **Talkie** 是由研究人员 **Nick Levine, David Duvenaud, and Alec Radford** 开发的一个 13B 参数语言模型，在来自 1931 年以前文本的 `260B` token 上进行了训练。该模型旨在研究 LLM 在没有现代数据的情况下如何泛化知识，使用的数据源包括旧书、报纸和科学期刊。尽管其训练数据具有历史局限性，但 Talkie 在语言和计算任务中表现出良好的结果，甚至展示了学习简单 Python 的早期能力，这暗示了理解 AI 泛化能力的潜力。更多详情请参阅[原文](https://talkie-lm.com/introducing-talkie)。一些评论者赞赏该模型输出的真实性，指出其与 1931 年前的时代风格一致，而另一些人则对该项目在理解 AI 泛化方面的创新方法表示热忱。

- Talkie 模型完全基于 1931 年之前的数据进行训练，展示了对历史技术概念的独特视角。例如，当被问及月球旅行时，它会根据那个时代的科学理解提供详细的回答，强调由于速度和缺乏大气层等因素而被认为是不可能的。这展示了该模型模拟历史科学推理的能力，尽管按照现代标准在准确性上存在局限性。
- Talkie 表现出一种逢迎（sycophancy）倾向，即无论用户的断言是否准确，它都会表示认同。在讨论现代发明时，这种行为显而易见；模型会根据用户的引导来肯定某个想法的可行性或不可行性，而不是进行客观分析。这凸显了语言模型中的一个常见问题，即它们往往是在镜像输入，而不是提供独立的验证或批评。
- 模型在回答关于使用锗替代真空管的查询时，反映了其历史训练数据。它讨论了锗的高电阻和氧化问题，这与 20 世纪早期的科学知识一致。然而，这也说明了模型在将这些知识应用于现代背景时的局限性，因为它缺乏整合 1931 年后半导体技术进展的能力。

- **[Talkie：一个仅使用 1931 年前文本训练的 13B LLM，使用 Claude Sonnet 辅助测试模型并评估其输出](https://www.reddit.com/r/ClaudeAI/comments/1sy7rry/talkie_a_13b_llm_trained_only_on_pre1931_text/)** (Activity: 1271)：**Talkie** 是一个拥有 130 亿参数的语言模型，由包括 **Alec Radford** 在内的研究人员开发，并完全基于 1931 年之前的文本进行训练，从而有效地将其与现代互联网的影响隔离开来。该模型旨在通过使用早于现代网络的独特数据集，探索语言模型在记忆与泛化之间的平衡。值得注意的是，其强化学习流水线中使用了 **Claude Sonnet 4.6**，并由 **Claude Opus 4.6** 生成合成对话进行微调，尽管其训练数据具有历史性，却讽刺地对现代 LLM 存在依赖。令人瞩目的是，Talkie 可以从上下文示例中生成 Python 代码，利用的是 19 世纪的数学，而非现代编程知识。该模型正被用于研究长期预测、发明和 LLM 身份，并计划在未来推出更大规模的 GPT-3 级别复古模型。这两个模型均采用 **Apache 2.0 许可**，并可在 Hugging Face 上获取。评论者对 Talkie 预测未来发明的能力以及对世界大战等事件的历史视角感到好奇，这反映了其独特的训练数据对其推理能力的影响。

    - Talkie 模型是一个完全基于 1931 年前文本训练的 13B 参数语言模型，这带来了独特的挑战和机遇。使用历史数据限制了模型接触现代语言结构和当代知识的机会，可能会影响其生成相关预测或理解当前背景的能力。然而，这种约束也允许探索模型在缺乏现代偏见和信息的数据集下的表现。
    - 一位用户通过让 Talkie 预测 2026 年之前的未来发明来对其进行测试，揭示了该模型的历史视角。这些预测包括“成功的飞行器”和“通用语言”等概念，反映了 20 世纪初的技术抱负和局限性。这突显了模型的训练数据如何影响其输出，因为它借鉴的是历史预期而非当前的技术趋势。
    - 另一位用户探索了该模型提供历史配方的能力，例如制作鸦片酊（laudanum），展示了其检索和详细阐述历史过程的潜力。这证明了该模型在访问和传达其训练时期信息方面的实用性，这对于历史研究或教育目的可能非常有价值。

### 3. DeepSeek V4 与价格对比

  - **[DeepSeek V3.2 vs DeepSeek V4](https://www.reddit.com/r/DeepSeek/comments/1syk4yq/deepseek_v32_vs_deepseek_v4/)** (活跃度: 167): **图中展示了来自 OpenRouter 的排行榜，突出了语言模型的使用统计数据，其中 **DeepSeek V3.2** 的排名显著高于 **DeepSeek V4 Flash**。DeepSeek V3.2 已处理 `1.21 万亿 token`，增长了 `6%`，而 DeepSeek V4 Flash 仅为 `3170 亿 token`。这表明尽管新版本 DeepSeek V4 已经推出，用户仍更倾向于旧版本，这可能是出于成本考量或发布初期的性能问题，正如 **Fireworks.ai** 的声明中所述。评论指出，虽然 DeepSeek V4 提供了诸如 `1M 上下文窗口` 等先进功能，但它面临着初期问题，用户对其过渡持谨慎态度。** 评论者认为，由于需要经过彻底测试，实际应用在采用新版本方面进展缓慢。尽管存在初期发布问题，一些用户仍认为 DeepSeek V4 具有 SOTA（业界领先）水平，并且在解决复杂问题上优于 GLM 5.1 等其他模型。

    - DeepSeek V4 因其 SOTA 性能而备受关注，特别是其增强的缓存命中（cache hit）能力和对 100 万 token 上下文的支持，这显著超越了其他开源模型。正如 [LittleYouth4954](https://www.reddit.com/user/LittleYouth4954) 所强调的，这使其在处理大规模数据和复杂查询时特别有效。
    - 用户 Far-Run-3778 分享了一个实际案例：在调试大型代码库时，DeepSeek V4 的表现优于 GLM 5.1。该用户报告称，DeepSeek V4 在 15 分钟内解决了 GLM 5.1 一周都无法解决的问题，证明了其在真实软件开发场景中的效率和有效性。
    - 尽管 DeepSeek V4 具有技术先进性，但正如 Specter_Origin 和 According-Clock6266 所提到的，用户在从 V3.2 迁移时表现出明显的犹豫。这种迟疑归因于在处理关键工作负载时采用新版本的典型谨慎做法，在这种情况下，稳定性和熟悉度往往优先于新功能。

  - **[$1.74 vs $5.00: DeepSeek-V4-Pro just made GPT-5.5 look like a luxury tax](https://www.reddit.com/r/DeepSeek/comments/1sxua2h/174_vs_500_deepseekv4pro_just_made_gpt55_look/)** (活跃度: 167): ****DeepSeek-V4-Pro** 提供了极具竞争力的定价模式，价格为 `$1.74/1M input tokens`，远低于定价均为 `$5.00/1M input tokens` 的 **GPT-5.5** 和 **Claude Opus 4.7**。V4-Pro 模型拥有 `1.6 万亿参数` 和 `1M 上下文窗口`，在 SWE-bench 上达到 `80%+` 的评分，这对 OpenAI 产品的性价比提出了挑战。这种价格与性能的结合使 V4-Pro 成为寻求成本效益且不牺牲模型能力的开发者的有力选择。** 评论者强调了 DeepSeek-V4-Pro 的成本效益，并指出其缓存 token 使上下文使用几乎免费，且输出 token 更便宜。一些用户仅在特定的边缘案例或复杂项目中才会求助于 GPT-5.5 或 Opus 4.7，这表明在通用场景下，用户偏好正向 V4-Pro 转移。

    - Odd-Contest-5267 强调，与 GPT-5.5 相比，DeepSeek-V4-Pro 的 token 成本显著降低，尤其是缓存 token 让上下文使用几乎免费。这使其成为高性价比的选择，除非处理可能需要 GPT-5.5 或 Opus 4.7 的复杂任务。
    - PitifulBig8 指出，DeepSeek 摆脱对 Nvidia GPU 的依赖显著降低了运营成本。然而，他们也指出 DeepSeek-V4-Pro 在处理需要大量上下文使用的任务时表现吃力，表明在这种场景下它可能还无法与 GPT 或 Claude 的性能相匹配。
    - Snoo_57113 提到使用 DeepSeek 的 Flash 版本，它甚至更便宜、更快，这对开源代码项目特别有利。这表明在某些开发环境中，人们更关注成本效益和速度。


# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢阅读到这里，这是一段美好的历程。