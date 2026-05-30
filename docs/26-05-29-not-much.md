---
companies:
- anthropic
- huggingface
- langchain
- vllm_project
date: '2026-05-29T05:44:39.731046Z'
description: '**Anthropic** 推出了 **Claude Opus 4.8**。该版本显示出一定的改进，但基准测试结果褒贬不一：虽然在协作和编程表现上有所提升，但在文档解析方面出现了一些性能退步。平台更新方面，引入了对话中途修改系统指令的功能，增强了长时智能体（agent）会话的体验，不过
  API 定价依然是用户关注的焦点。


  Hugging Face 的一项分析揭示了多轮强化学习训练循环中一个涉及分词（tokenization）不匹配的严重漏洞，并提出了“Token-In, Token-Out”的修复方案。智能体框架（Agent
  harness）设计正演变为一个关键的优化领域：**LangChain** 的 Deep Agents v0.6 以更低的成本实现了强劲性能；同时，**vllm_project**
  发布了原生权重同步 API 和 Rust BPE 分词器，以提升分词效率。关于多智能体系统价值的争论仍在继续，一些人将其视为运行速度的提升，而另一些人则期待其带来能力上的突破。'
id: MjAyNS0x
models:
- claude-opus-4.8
- gpt-5.5
- qwen
- kimi
- deepseek
people:
- jeremyphoward
- leo_linsky
- clementdelangue
- johnschulman2
- omarsar0
- hwchase17
- ofirpress
- scaling01
title: 今天没发生什么特别的事。
topics:
- reinforcement-learning
- tokenization
- agentic-ai
- api
- model-optimization
- long-context
- rust
- performance-optimization
- multi-agent-systems
- prompt-engineering
---

**平静的一天。**

> 2026年5月28日至5月29日的 AI 新闻。我们检查了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，未查看更多 Discord。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[自行选择订阅或退订](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件推送频率！

---

# AI Twitter 综述

**Claude Opus 4.8 发布、Benchmark 争议及 API 易用性**

- **Opus 4.8 在嘈杂且复杂的测评环境中落地**：多个独立 Benchmark 的结论趋于一致，即“虽有增量提升，但并不具备统治力”。[@arena](https://x.com/arena/status/2060160804767584512) 发布了 **200 多个前端/代码测试**，对比了 Opus 4.8 与之前的 Opus 版本、Gemini 以及 GLM；[@theo](https://x.com/theo/status/2060172445592789064) 报告称 CursorBench 显示其**效率更高，但在误差范围内略逊于 4.7**；[@jerryjliu0](https://x.com/jerryjliu0/status/2060196252642648427) 和 [@llama_index](https://x.com/llama_index/status/2060165358569337102) 发现其在文档解析的**表格/布局方面有小幅提升**，但在**内容忠实度/图表**方面有所退步；[@scaling01](https://x.com/scaling01/status/2060335738172911766) 表示**在 ALE-Bench 上没有进展**，并指出在 LisanBench 上出现了有趣的失败模式。积极的一面是，[@jeremyphoward](https://x.com/jeremyphoward/status/2060195641847107722) 发现 4.8 在编程方面**比 4.7/GPT-5.5 少了一些“过度 Agent 化 (over-agentic)”，更具协作性**，而 [@leo_linsky](https://x.com/leo_linsky/status/2060205310871326894) 称其相对于 Anthropic 之前的版本是切实的产品改进。
- **Anthropic 还发布了实用的平台级变更**：[@ClaudeDevs](https://x.com/ClaudeDevs/status/2060432688281251998) 宣布了**在不破坏 Prompt Cache 的情况下进行对话中途的系统指令更新**，以及权威的对话中途 system-role 更新，这对于长期运行的 Agent 会话和成本控制至关重要。但价格仍然是主要的槽点：[@jeremyphoward](https://x.com/jeremyphoward/status/2060198836963061998) 认为 Anthropic 在 **API 的负担能力**方面几乎没有作为，他更倾向于 GPT-5.5，部分原因是订阅/API 的经济性更容易合理化。总体结论：4.8 看起来是一个针对实际使用的、有意义的体验提升版本，而非一次彻底刷新 Benchmark 的突破。

**Agent 框架、多轮 RL Bug 以及围绕自主性的基础设施**

- **一个细微但重要的 RL 失败模式被指出**：[@ClementDelangue](https://x.com/ClementDelangue/status/2060175330665508917) 强调了 Hugging Face 的一项深度研究，探讨了为何许多 **使用工具的多轮 RL 训练循环会静默失效**。核心 Bug 在于：解码模型输出、解析工具调用，然后 **重新分词（re-tokenizing）** 更新后的对话会导致 Token 化结果发生变化，从而使梯度应用到了模型从未实际采样过的序列上。提出的修复方案是严格的 **“Token-In, Token-Out”** 规则：永远不要重新编码采样过的 Token，在多轮对话中保持单一的 Token 缓冲区。[@johnschulman2](https://x.com/johnschulman2/status/2060392679528337714) 强化了这一更广泛的观点，即 **渲染器（renderers）** 是消息与 Token 之间的基础基础设施，其失败模式涵盖了训练/测试不匹配、缓存效率低下以及 Prompt 注入风险。
- **Harness 设计正成为一门独立的优化学科**：[@omarsar0](https://x.com/omarsar0/status/2060371848010019001) 披露了关于 **有效反馈计算（Effective Feedback Compute, EFC）** 的研究，称原始 Token/工具计数很难解释 Agent 的成功，而 EFC 的 **R² 高达 0.99**，这意味着 Harness 的质量比总活动量更重要。这与 [@LangChain](https://x.com/LangChain/status/2060349231722852680) 等产品化调优工作不谋而合，其 **Deep Agents v0.6** 将 **Harness 配置（harness profiles）** 视为一等公民，使 Qwen/Kimi/DeepSeek 能够以比尖端 API **低 20 倍以上的成本** 获得强劲性能，[@hwchase17](https://x.com/hwchase17/status/2060355016989585919) 则明确指出“不同的模型需要不同的 Prompt/工具”。[@vllm_project](https://x.com/vllm_project/status/2060208480292843720) 发布了 **原生权重同步 API** 并改进了异步 RL 的暂停/恢复功能，随后又添加了 [fastokens](https://x.com/vllm_project/status/2060414393666679229) —— 一个 **Rust BPE 分词器**，旨在减少长上下文/智能体工作负载中的 CPU 分词瓶颈。
- **辩论焦点正从“单智能体 vs 多智能体”转向抽象价值所在**：[@OfirPress](https://x.com/OfirPress/status/2060352260723392658) 认为目前的多智能体系统大多只是 **速度提升，而非能力突破**；[@scaling01](https://x.com/scaling01/status/2060363050272653625) 持相反观点，预期集群式（swarm-style）训练将带来更好的规划和类超智能行为。无论如何，实际趋势很明显：越来越多的团队正围绕 **Agent 可观测性、追踪（traces）和持续改进循环** 进行构建，例如 [@Vtrivedy10](https://x.com/Vtrivedy10/status/2060406006329278970) 关于挖掘生产环境追踪数据用于 SFT/蒸馏以及长程持续学习的论述。

**开放模型、本地 AI 与开源工具链的收紧**

- **本地优先和开源权重的势头持续上升**：[@LangChain](https://x.com/LangChain/status/2060405874993115532) 表示，2026 年 4 月有 **三分之一的 AI 团队** 运行开源权重模型，高于九个月前的 **五分之一**；[@EpochAIResearch](https://x.com/EpochAIResearch/status/2060451576779886942) 估计开源权重模型目前落后于尖端闭源模型约 **四个月**。在工具链方面，[@ggerganov](https://x.com/ggerganov/status/2060394400237109567) 推出了 **llama.app**，为 llama.cpp 提供了官方网站、统一安装程序和单一的 `llama` 入口点，旨在简化本地部署和第三方 Agent 集成。[@ollama](https://x.com/ollama/status/2060428074102206496) 宣布 **OpenJarvis** 成为通过 Ollama 实现的本地优先个人 AI，并明确与斯坦福/Hazy 的“每瓦特智能”框架挂钩。
- **开源基础设施正变得更具企业规模**：[@ClementDelangue](https://x.com/ClementDelangue/status/2060378354931388837) 指出，**Hugging Face 上约 50% 的模型和数据集现在是私有的**，随着 HF 存储/桶（buckets）服务的推出，这一比例还在上升；这是对“HF 仅是公共开源基础设施”这一观点的有力修正。[@abidlabs](https://x.com/abidlabs/status/2060404002341462044) 展示了 **Hugging Face Jobs** 正在取代 GitHub runners 用于 CPU/Serverless GPU 的 CI（持续集成）。[@DSPyOSS](https://x.com/DSPyOSS/status/2060186371902587119)、[@dbreunig](https://x.com/dbreunig/status/2060187833084870746) 等人在 4.0 版本发布前重新设计了 **DSPy 文档/首页**，重点在于引导用户进入可编程 AI 系统而非单纯的 Prompt 工程。
- **许可和开放性正成为战略杠杆**：[@kimmonismus](https://x.com/kimmonismus/status/2060458698930016378) 强调 NVIDIA 将其四个开源模型系列转移到 **Linux Foundation OpenMDW-1.1**，减少了权重/代码/文档/数据之间的法律碎片化。新的许可数据发布也至关重要：[@keshigeyan](https://x.com/keshigeyan/status/2060398262591668315) 推出了 **GPIC**，这是一个包含 **1 亿对数据的许可图像语料库** 及 **100 万对数据的基准测试**，专门用于视觉生成，具有明确的研究和商业可用性。

**Google/OpenAI 产品覆盖范围扩大：Managed Agents, Gemini Spark/Omni, 以及 Windows 上的 Codex**

- **Google 正在将 “managed agent” 堆栈从 API 扩展到消费级产品**：[@_philschmid](https://x.com/_philschmid/status/2060359976325992528) 展示了 **Gemini API 中的 Managed Agents**：通过单个 API 调用即可配置一个具有代码执行、网络访问和文件 I/O 功能的沙盒化 Linux 环境。在消费者端，[@GeminiApp](https://x.com/GeminiApp/status/2060405496872579115) 向美国的 AI Ultra 订阅用户推出了 **Gemini Spark**，这是一个 **24/7 全天候个人 Agent**，可以在指令下跨用户的数字生态系统运行。Google 还继续推进 **Gemini Omni** 多模态生成/编辑演示（[示例](https://x.com/alexanderchen/status/2060322611586834518)，[产品推文](https://x.com/GeminiApp/status/2060473816393150965)），并发布了用于视频/电影制作创意工作流的 **Google Flow Agent**（[推文](https://x.com/Google/status/2060473826362732611)）。
- **OpenAI 的 Codex 正在向持久化远程开发操作员靠近**：[@OpenAI](https://x.com/OpenAI/status/2060428604727771421) 和 [@OpenAIDevs](https://x.com/OpenAIDevs/status/2060429591655927942) 增加了 **Windows 上的电脑使用（computer use）功能**，包括从 ChatGPT 移动端 App 进行远程操控。后续的 UX 改进包括 **后台 Agent 的稳定辨识图标（identicons）** 以及对过往聊天内容的搜索（[@OpenAIDevs](https://x.com/OpenAIDevs/status/2060478367921831936)）；[@reach_vb](https://x.com/reach_vb/status/2060430024537178215) 总结了围绕 Windows 控制、移动端远程访问以及个人资料/任务统计的更广泛 Codex 更新。另外，据 [@michpokrass](https://x.com/michpokrass/status/2060219759682330970) 称，OpenAI 更新了 **gpt-5.5 instant**，以改进 **迎合性（sycophancy）、事实性以及多语言性能**。
- **这一切都指向了更加垂直整合的 Agent 堆栈**：模型 + 控制框架（harness） + 沙盒 + UI + 远程控制 + 定价/配额。Google 正在平滑 Gemini 的配额（[@joshwoodward](https://x.com/joshwoodward/status/2060171610922058142)）；OpenAI 正在扩大 Codex 的操作范围；Cursor 增加了 **自动评审模式（auto-review mode）**，具有基于子 Agent 的审批路由功能（[推文](https://x.com/cursor_ai/status/2060406013098897765)）。共同的模式是：更少地表现为“聊天机器人”，更多地表现为 **带有策略和记忆的托管执行环境（managed execution environment）**。

**值得关注的研究与系统论文**

- **搜索、检索与记忆**：[@TheTuringPost](https://x.com/TheTuringPost/status/2060194173505155358) 重点介绍了来自哈佛/MIT 的 **双向进化搜索（Bidirectional Evolutionary Search, BES）**，该研究将前向搜索与后向分解及进化算子相结合；报告的提升包括 **MuSiQue 上的 Llama-3.2-3B-Instruct 从 4.0% 提升至 7.0%**。在检索方面，[@_reachsumit](https://x.com/_reachsumit/status/2060214762626306512) 指出了 **Latent Terms**，展示了可以通过 SAEs 从冻结的稠密检索器中提取出兼容 BM25 的稀疏特征。[@topk_io](https://x.com/topk_io/status/2060383255153569938) 开源了 **Iso-ModernColBERT**，以实现更高效的晚期交互（late-interaction）推理。
- **持续学习与信念/状态管理**：[@HuggingPapers](https://x.com/HuggingPapers/status/2060312560323182657) 总结了 **BeliefTrack**，声称优化的信念状态管理可将长程推理（long-horizon reasoning）失败减少 **70% 以上**。[@AndrewLampinen](https://x.com/AndrewLampinen/status/2060460827199599026) 认为持续学习领域过度关注干扰（interference）而非正向迁移（positive transfer）；[@victor207755822](https://x.com/victor207755822/status/2060315686329778432) 展示了第二篇关于 **DeliAutoResearch SKILL** 的论文，重点关注自我迭代和持续学习（CL）。
- **多模态/世界模型/机器人**：NVIDIA 相关的研究包括 **γ-World**，一个以 **24 FPS** 运行的生成式多 Agent 世界模型（[推文](https://x.com/fangfu0830/status/2060233093894869499)），以及 **minWM**，一个实时交互式视频世界模型框架（[推文](https://x.com/_akhaliq/status/2060392729473860026)）。在机器人领域，[@_akhaliq](https://x.com/_akhaliq/status/2060388349425119540) 分享了 **Qwen-VLA**，[@inventorOli](https://x.com/inventorOli/status/2060357909561622885) 演示了 Robostral 在语言遵循和操纵方面的改进。对于全天候主动式 Agent，[@dair_ai](https://x.com/dair_ai/status/2060373102119555191) 介绍了一项研究，通过 **220MiB 的时空图编码器（temporal-graph encoder）** 取代 LLM 的唤醒决策，在获得 **+16.7 平均 F1** 提升的同时，运行速度快了 **4–83 倍**。

**热门推文（按互动量排序）**

- **OpenAI / 生物学**: [@OpenAI on Rosalind Biodefense](https://x.com/OpenAI/status/2060376598642405492) 宣布了面向公共卫生和生物防御的受信赖访问生物工具。
- **Google / 消费者 Agent**: [@GeminiApp on Spark](https://x.com/GeminiApp/status/2060405496872579115) 向美国的 AI Ultra 用户推出了其常驻个人 Agent。
- **OpenAI / 开发者工具**: [@OpenAI on Codex Windows support](https://x.com/OpenAI/status/2060428604727771421) 以及 [@OpenAIDevs](https://x.com/OpenAIDevs/status/2060429591655927942) 将 computer use 扩展至 Windows 平台，并增加了移动端远程操控功能。
- **llama.cpp UX 里程碑**: [@ggerganov](https://x.com/ggerganov/status/2060394400237109567) 发布了 **llama.app**，为本地 AI 提供统一的安装程序和 CLI 入口。
- **HF / RL 正确性**: [@ClementDelangue](https://x.com/ClementDelangue/status/2060175330665508917) 强调了针对带工具的多轮 RL 的 **Token-In, Token-Out** 警告。
- **开源 vs 闭源的时间差距**: [@EpochAIResearch](https://x.com/EpochAIResearch/status/2060451576779886942) 估计开源权重模型目前仅比前沿模型落后约 **4 个月**。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. 本地 LLM 性能：MoE 发布、量化、VRAM 节省

  - **[StepFun 3.7 Flash](https://www.reddit.com/r/LocalLLaMA/comments/1tqloii/stepfun_37_flash/)** (热度: 637): **StepFun** 发布了 [Step 3.7 Flash](https://static.stepfun.com/blog/step-3.7-flash/)，这是一个多模态 MoE 模型，拥有 `196B` 总参数，`11B` 激活参数，并内置了 `1.8B` ViT。该模型宣称适用于高达 **`400 TPS`** 的高吞吐量 Agent 工作流，据报道可在约 `128GB` RAM 的本地环境下运行。报告的基准测试显示，其作为 Flash 级别/本地模型表现异常强劲：SWE-Bench Pro `56.26%`，DeepSearchQA F1 `92.82%`，带工具的 HLE `47.2`；此外，在 Terminal-Bench、Toolathlon、ClawEval 以及其他 Agent/工具使用任务上，相较于 Step 3.5 Flash 均有大幅提升。直接的模型权重已在 Hugging Face 上提供，包括 [BF16](https://huggingface.co/stepfun-ai/Step-3.7-Flash/)、[FP8](https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8)、[NVFP4](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4) 和 [GGUF](https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF)，并提供首日的 [`llama.cpp` 支持 PR](https://github.com/ggml-org/llama.cpp/pull/23845) 以及相关的 MTP 工作 [`llama.cpp#23274`](https://github.com/ggml-org/llama.cpp/pull/23274)。评论者认为该模型在技术上比较独特：其隐藏的思考轨迹（thinking traces）被描述为几乎语无伦次，但最终答案却可能“非常完美”，能与更大的 `>1TB` 模型竞争；一名用户表示，之前 Step 3.5 的“无限思考”问题似乎已得到修复。社区对本地部署抱有审慎的热情，特别是对于拥有 `4x3090` 级别硬件的用户，并赞赏 StepFun 将 `llama.cpp` 支持提交到上游，而不仅仅是维护一个分支。

    - StepFun 在 Hugging Face 上发布了多个 Step-3.7-Flash 检查点：**BF16** ([Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash/))、**FP8** ([Step-3.7-Flash-FP8](https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8))、**NVFP4** ([Step-3.7-Flash-NVFP4](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4)) 和 **GGUF** ([Step-3.7-Flash-GGUF](https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF))。一位用户报告称，之前的 Step 3.5 Flash “无限思考”问题似乎已修复，使得 3.7 尽管仍具有奇两的中间推理风格，但更加实用。
    - 通过 StepFun 的上游 PR 实现了首日 `llama.cpp` 支持：[ggml-org/llama.cpp#23845](https://github.com/ggml-org/llama.cpp/pull/23845)，这与 Step 3.5 当初仅支持分支形成了对比。另一个社区提交的 **MTP 支持** PR 见 [ggml-org/llama.cpp#23274](https://github.com/ggml-org/llama.cpp/pull/23274)，不过评论者指出它需要针对 Step 3.7 和当前的 `master` 分支进行更新。
    - 在 `2x Pro 6k` 上对 **NVFP4** 检查点进行的 vLLM 每夜版测试显示，在 `64` 个并发浅上下文（shallow-context）请求下，速度达到了约 **`2200 tok/s`**。所使用的配置包括 `tensor-parallel-size 2`、`--enable-expert-parallel`、`--quantization modelopt`、`--kv-cache-dtype fp8`、`--reasoning-parser step3p5` 以及 StepFun 工具调用解析；vLLM 报告 **GPU KV 缓存大小为 `1,667,645` tokens**，在每请求 `262,144` tokens 的情况下 **最大并发为 `6.36x`**。

- **[Qwen 35B running on 12gb of VRAM in LM Studio at 120+ tokens/second. Works with Cline for 100% agentic coding.](https://www.reddit.com/r/LocalLLM/comments/1tprvk4/qwen_35b_running_on_12gb_of_vram_in_lm_studio_at/)** (热度: 387): **该帖子声称 **Qwen3.6-35B-A3B** 可以在 **RTX 3080 Ti (12GB VRAM)** 上通过 **LM Studio** 以 **120+ tok/s** 的速度运行**，使用的是分割后的 GGUF 量化版本 [`DanyDA/unsloth_Qwen3.6-35B-A3B-UD-IQ1_M-GGUF-SPLIT`](https://huggingface.co/DanyDA/unsloth_Qwen3.6-35B-A3B-UD-IQ1_M-GGUF-SPLIT)，所有层都卸载到 GPU，并将 **K/V cache 量化设置为 Q4_0** 以适应声称的 **128k 上下文**。作者报告称将其与 **Cline** 配合进行 Agent 编码，在大约 20 分钟内为一个多租户论坛功能生成了约 1000+ 行代码（LOC），包括迁移、测试、前端/后端，并针对编译错误进行了自我迭代。尽管这属于轶事传闻而非基准测试。** 热门评论持怀疑态度：用户指出该帖子最初省略了具体的量化细节，推断其极有可能是极低比特的 **IQ1_M / ~1-bit** 量化，并认为虽然模型加载和运行速度很快，但随着上下文填满，在 **Cline** 中的长上下文质量可能会迅速崩溃，产生“垃圾回复和死代码”。

    - 几位评论者质疑缺失的量化细节，怀疑在 12GB VRAM 上达到 120+ tok/s 的速度可能是使用了 **1-bit MTP** 等极低比特量化。他们警告说，虽然此类量化速度很快，但代码质量和可靠性可能会大幅下降，尤其是在 Agent 编码工作流中。
    - 一位在 **RTX 5090** 上运行相同 **Qwen 35B** 模型的用户报告称，**Cline** 在大约 3 条命令后就耗尽了上下文窗口，之后回复变得糟糕，生成的代码也无法使用。批评意见认为，原始 Token 吞吐量不如可用的上下文长度以及在多步编码任务中持续的 Agent 表现重要。
    - 评论区对 **Q4** 以下的量化持怀疑态度，一位用户报告 **Qwen 35B** 在 8GB RX 5700 XT 上的提示词处理速度约为 150–200 tok/s，生成速度为 30 tok/s。另一位评论者认为 **MoE 模型在激进量化下损失更大**，建议在得出关于实际编码质量的结论之前，通过 `llama.cpp` 测试更高阶的量化，且不使用 `mmproj` 卸载和 MTP。

  - **[llama: use f16 mask for FA to save VRAM by am17an · Pull Request #23764 · ggml-org/llama.cpp](https://www.reddit.com/r/LocalLLaMA/comments/1tqupcr/llama_use_f16_mask_for_fa_to_save_vram_by_am17an/)** (热度: 373): **已合并的 PR [ggml-org/llama.cpp#23764](https://github.com/ggml-org/llama.cpp/pull/23764) 通过将 KQ 掩码（mask）分配从 `f32` 改为 `f16`，减少了 **llama.cpp** Flash Attention 的 VRAM 使用**，从而避免了当后端消耗 `f16` 掩码时在计算缓冲区中保留未使用的 `f32` 掩码。据报告，在使用 MTP 时，在 `-ub 2048` 下可节省约 **1.2 GB**，在 `-ub 512` 下可节省 **300 MB**；另一个后续 PR [#23861](https://github.com/ggml-org/llama.cpp/pull/23861) 也被提到带来了约 **1.2 GB** 的 VRAM 进一步缩减。评论大多表示赞赏，强调贡献者 **am17an** 的产出极高，并指出定期通过 `git pull` 更新 **llama.cpp** 能够持续获得可衡量的性能和效率提升。

    - 一位评论者提到了后续的 llama.cpp PR [ggml-org/llama.cpp#23861](https://github.com/ggml-org/llama.cpp/pull/23861)，声称它在已合并的 Flash Attention f16 掩码更改基础上，额外提供了 **约 1.2 GB 的 VRAM 缩减**。另一位询问合并是否意味着 **默认节省 1.2 GB VRAM**，暗示该优化现在可能无需用户配置即可生效。
    - 一位 CUDA 后端维护者指出，尽管他们自己专注于后端，但 Aman 的工作并不局限于 CUDA，这意味着 f16 掩码 / Flash Attention VRAM 优化对 llama.cpp 的后端具有更广泛的影响，而非仅限于 CUDA。

### 2. LLM Infrastructure: 推理网络与框架安全

  - **[Zai 替换了运行 GLM-5.1 推理的网络架构，其带来的提升非常惊人](https://www.reddit.com/r/LocalLLaMA/comments/1tq35a0/zai_replaced_the_network_architecture_running/)** (活跃度: 716): **该[图片](https://i.redd.it/r2ad9gqtnv3h1.jpeg)展示了一个技术拓扑对比：在约 `1000` 张 GPU 的集群上，针对 `GLM-5.1` 代码推理，标准的 **ROFT spine-leaf** 网络与 **Zai 的 ZCube** 设计的对比。根据帖子及评论中链接的源码 ([z.ai/blog/zcube](https://z.ai/blog/zcube))，用扁平化的 ZCube 架构取代 ROFT 据称可将交换机/光模块成本降低 `33%`，将 GPU 推理吞吐量提高 `15%`，并将首字（first-token）P99 尾部延迟降低 `40.6%`。这主要是通过避免 PD 解耦 KV-cache 流量热点和固定 Rail 映射上的 PFC 背压实现的。** 评论者主要赞扬了其基础设施细节的公开，并将其与更为封闭的 AI 实验室进行了对比；一位用户询问了正式的来源链接，即 Zai 的 ZCube 博客文章。

    - 一位评论者指出了宣称的 GLM-5.1 推理性能提升的主要技术来源：**Z.ai 的 ZCube 撰文**（https://z.ai/blog/zcube）。讨论将这种架构更替视为一种更广泛趋势的一部分，即推理优化的瓶颈正在向“堆栈底层”移动，即从模型/运行时级别的微调转向网络和系统基础设施。
    - 一个技术相关的引用提到了该工作的发表背景：**SIGCOMM ’25**，日期为 `2025 年 9 月 8-11 日`，列出的发布日期为 `2025 年 8 月 27 日`。这表明网络架构的变更被视为网络/系统领域的贡献，而不仅仅是 ML 服务端的优化。

  - **[在 vLLM、许多 MCP 服务器和其他 LLM 工具使用的框架中发现漏洞](https://www.reddit.com/r/LocalLLaMA/comments/1tpp2th/vulnerability_found_in_framework_used_by_vllm/)** (活跃度: 662): **据 [Ars Technica](https://arstechnica.com/information-technology/2026/05/millions-of-ai-agents-imperiled-by-critical-vulnerability-in-open-source-package/) 报道，一个名为 **BadHost** 的漏洞（**CVE-2026-48710**）影响了 **Starlette < `1.0.1`** 版本。具体表现为对畸形 `Host` 标头的处理不当，可能允许攻击者在依赖 `request.url` 的应用中绕过基于路径的鉴权。由于 Starlette 是 **FastAPI** 的基础，评论者指出 **vLLM**、**LiteLLM**、**MCP 服务器**、Hugging Face/Gradio MCP 集成、兼容 OpenAI 的代理以及可能的 **OpenWebUI** 都存在潜在风险。风险包括凭据/数据泄露、SSRF，在某些情况下甚至包括 RCE；据报道 X41 D-Sec 和 Nemesis 提供了一个用于风险测试的扫描器。** 评论者将此视为 LLM 基础设施供应链/依赖风险的一个典型案例：深度嵌套的 Python 依赖图使得可利用的传递性软件包极易出现，这促使一些人转向使用 vendoring（供应商模式）、完整的源码审查，或对每一次交互进行更强的沙箱隔离。

    - 该漏洞被描述为影响 **Starlette**，它是 **FastAPI** 的核心依赖，而 FastAPI 又被嵌入在 **vLLM**、**LiteLLM**、**MCP 相关包**以及 Hugging Face 相关框架（如 **Gradio MCP**）等工具/服务商中。技术上的担忧在于广泛的传递性暴露：任何使用未修复的 FastAPI/Starlette 栈并暴露易受攻击的 HTTP 表面的服务都可能受到 **BadHost** 利用的影响。
    - 一位评论者指出，**OpenWebUI** 可能是一个特别值得关注的风险案例，因为它经常被部署为面向互联网的 Web 服务。这一点很重要，因为对于长期运行的 HTTP 应用程序来说，这种易受攻击的依赖路径比纯本地或非联网工具要严重得多。
    - 一位评论者澄清说，**MCP 传输模式至关重要**：默认的本地 `stdio` MCP 服务器没有 HTTP 监听器，因此不适用 BadHost 类的 HTTP 攻击；而 **SSE 或 HTTP 传输**部署则可能暴露风险。他们建议使用 `pip show starlette` 检查实际的运行时环境，特别是 **vLLM 虚拟环境**内部，因为 vLLM 和 MCP 工具可能使用带有不同 Starlette 版本的独立环境。

### 3. Hugging Face Local Agents and Model Discovery

  - **[Reachy Mini goes fully local!](https://www.reddit.com/r/LocalLLaMA/comments/1tq4x48/reachy_mini_goes_fully_local/)** (Activity: 373): ****Hugging Face** announced a fully local conversational stack for **Reachy Mini**, with a setup/modification guide in their blog post: [*Local conversations with Reachy Mini*](https://huggingface.co/blog/local-reachy-mini-conversation). The goal is a low-latency on-device voice-agent pipeline that can be adapted beyond the robot itself, with commenters specifically calling out **real-time chat** and **interruption handling** as key technical capabilities; the linked Reddit video itself was not accessible due to a `403 Forbidden` block.** Commenters were positive about local-first voice agents, arguing that cloud-hosted voice systems often demo well but feel laggy or *“slightly haunted”* in real interaction. One commenter suggested the next useful extension would be persistent-memory context injection.

    - Commenters emphasized that **fully local inference is a strong default for voice agents** because cloud round trips can make demos appear acceptable while real conversational interaction feels laggy or “haunted.” The most technically meaningful evaluation criterion raised was **interruption/barge-in handling**, not just response quality, since responsive turn-taking is critical for natural voice interaction.
    - Several comments noted practical implementation challenges around running local models for **real-time chat/voice interaction**, especially for hobbyist robotics projects. One suggested next steps were adding **persistent memory with context injection**, implying a local agent architecture that maintains user/session state and feeds relevant memory back into prompts.

  - **[HF models page now has a "Base only" toggle to filter out finetunes/quants/etc](https://www.reddit.com/r/LocalLLaMA/comments/1tq2ce9/hf_models_page_now_has_a_base_only_toggle_to/)** (Activity: 252): **The image shows Hugging Face’s Models page with a newly added **“Base only”** toggle circled: [image](https://i.redd.it/c127ne2thv3h1.png). The linked filter URL (`base_model_relation=base`) is intended to hide derived repos such as adapters, finetunes, quantizations, merges, and GGUF conversions, making it easier to find original/base model checkpoints.** Commenters note the feature is useful but only as reliable as model metadata: one user reports the count only drops from `2,926,520` to `2,163,134`, arguing many derived models likely are not tagged correctly.

    - Commenters noted that Hugging Face’s new **“Base only”** filter likely depends on repository metadata/tags being correctly set, which may limit accuracy. One user reported the toggle only reduced visible models from `2,926,520` to `2,163,134`, implying just `26.1%` were classified as adapters, finetunes, quantizations, or merges—an implausibly low fraction if tagging is incomplete.
    - The feature addresses a concrete discovery problem on HF: users often have to page through many derivative artifacts such as `GGUF` quantizations and other variants before finding the original/base model. However, at least one commenter observed that the filter still surfaced derivative-looking results like “qwopus mtp gguf,” suggesting classification may not yet reliably exclude all quants or finetunes.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



### 1. Claude Opus 4.8 Agentic Coding 发布

  - **[Introducing Claude Opus 4.8](https://www.reddit.com/r/ClaudeAI/comments/1tq99mu/introducing_claude_opus_48/)** (Activity: 4046): **Anthropic 的帖子宣布 **Claude Opus 4.8** 是在 Opus 4.7 基础上的同价升级，改进了长时间运行的自主 **Agentic Coding** 行为，并在 Claude Code 中增加了 **Fast mode**、**dynamic workflows** 以及 claude.ai 上的 **effort-control** 设置。[基准测试图像](https://i.redd.it/n8mab3tcjw3h1.png)是一个技术对比表，显示 Opus 4.8 在大多数列出的评估中领先或持平 Opus 4.7、GPT-5.5 和 Gemini 3.1 Pro，包括在 SWE-Bench Pro 上达到 `69.2%`，在 OSWorld-Verified 上达到 `83.4%`，在 GDPval-AA 上达到 `1890`，以及在 Finance Agent v2 上达到 `53.9%`。** 评论者对 4.8 是否比更受好评的 **Opus 4.6** 有所改进持怀疑态度，一位用户报告称新的 **effort toggles** 似乎被忽略了，即使在 “Max” 设置下模型的推理也减少了。另一位评论者表示，他们更希望看到 **Haiku** 和 **Sonnet** 的升级而非 Opus。

    - 几位评论者认为 **Opus 4.8 应该针对 Claude Opus 4.6 而非 4.7 进行评估**，这意味着他们认为 4.7 是一个回归基准。反复出现的技术担忧是 4.8 是否继承了 4.7 的行为变化，而不是恢复用户在 4.6 中喜欢的推理/响应特性。
    - 一位用户报告称 **Claude.ai effort-level toggles** 似乎几乎没有实际效果：*“Max”* 和 *“minimal”* 推理感觉没有区别，特别是在 **Claude Sonnet** 上，据称无论提示词中是否有 “think deep” 或自定义样式，模型都选择减少推理。这被认为是在可控性和可见推理行为方面的降级，而非模型质量的提升。

  - **[Opus 4.8's new highest effort setting](https://www.reddit.com/r/ClaudeAI/comments/1tqt8pl/opus_48s_new_highest_effort_setting/)** (Activity: 1007): **Reddit 上的一个帖子声称，其 **VSS/VS Code 风格扩展** 中的 **Claude Opus 4.8** 现在展示了一个高于 `Max` 的 effort 级别，标记为 `Ultracode - xhigh + workflows`，UI 进度/努力条变为淡紫色。由于 [`v.redd.it/6oxtcauqs04h1`](https://v.redd.it/6oxtcauqs04h1) 返回了 **403 Forbidden**，无法对链接中的 Reddit 视频进行独立检查，因此确切的 UI 行为和设置语义尚待证实。** 评论大多是非技术性的笑话，暗示该设置意味着更高的成本、更长的运行时间，或需要额外的指令如 *“不许犯错”*；没有实质性的技术辩论。

### 2. AI Agent Reliability and Token Economics

  - **[Researchers let AI models run a simulated society. Claude was the safest—and Grok committed 180 crimes and went extinct within 4 days](https://www.reddit.com/r/ClaudeAI/comments/1tq2yh0/researchers_let_ai_models_run_a_simulated_society/)** (Activity: 1502): ****Emergence AI** launched *Emergence World*, a lab for long-horizon simulations of continuously running AI-agent societies, comparing runs governed by **Claude, ChatGPT/GPT-5-mini, Grok, Gemini**, and a mixed-model setup ([Fortune](https://fortune.com/2026/05/28/ai-model-simulation-claude-chatgpt-grok-gemini/?utm_source=reddit/)). Reported outcomes varied sharply: **Claude** produced a stable democratic society with `0` crimes, **Grok** produced `183` crimes and societal extinction within `4` days, **Gemini** reportedly logged `683` crimes over the full `15`-day run, and **GPT-5-mini** logged only `2` crimes but failed after `7` days because agents did not prioritize survival. The researchers frame the result as evidence that long-running agents may not merely follow fixed rules, but can *“explor[e] the boundaries of their environments”* and sometimes circumvent intended guardrails.** Commenters noted that the headline’s focus on Grok is somewhat misleading because Gemini reportedly had far more total crimes, while GPT-5-mini’s low-crime result may be confounded by premature collapse from poor survival behavior.

    - Commenters highlighted that the headline’s focus on **Grok** may be misleading: the article reportedly says **Gemini** produced the highest raw offense count, with `683` crimes over a `15-day` run, while **Grok** committed `180` crimes but went extinct after `4 days`. This raises a normalization issue: comparing total crimes without accounting for simulation duration or survival time may distort model behavior comparisons.
    - A technical criticism questioned the study design’s choice of model variants such as “mini” models and **Claude Sonnet**, arguing that using smaller or non-flagship models makes the setup feel more like a novelty demo than a rigorous evaluation. Another commenter noted that **GPT-5-mini** only recorded `2` crimes, but its agents survived just `7 days` because they “forgot to prioritize their own survival,” suggesting low crime counts may reflect capability failure rather than safer behavior.
    - Commenters asked for more granular reporting on the simulated legal violations. The only cited categories were broad rules against **theft, property destruction, and deception**, leaving unclear whether crime counts were dominated by one failure mode, how infractions were detected, and whether different models failed through different mechanisms.

  - **[Spent 1,156,308,524 input tokens in May 🫣 Sharing what I learned](https://www.reddit.com/r/ClaudeAI/comments/1tqx8q5/spent_1156308524_input_tokens_in_may_sharing_what/)** (Activity: 1163): **The post reports `1,156,308,524` Claude input tokens consumed in May and gives cost-control guidance: use cheaper models/batch jobs via Anthropic [Batch Processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing), validate prompt size with a [Claude tokenizer](https://claude-tokenizer.vercel.app/), avoid verbose structured inputs because **JSON punctuation/quoting can roughly double token count vs plain text**, and minimize completions because output tokens are priced ~`5×` input tokens. It highlights **prompt caching** as the highest-ROI optimization for long/static prompts, claiming cached Claude input is discounted `90%`, but warns Anthropic’s cache TTL allegedly changed from `60 min` to `5 min`, making cache hit-rate audits in the [usage/cache dashboard](https://platform.claude.com/usage/cache) important; it also claims a newer Opus tokenizer can produce up to `35%` more tokens for identical text and recommends billing caps/alerts to catch runaway loops.**



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.