---
companies:
- cohere
- deepseek
- intel
- huggingface
- baseten
- vllm-project
- chutes-ai
- anycoder
date: '2025-08-21T05:44:39.731046Z'
description: 'Cohere 的 **Command A 推理模型**在开放深度研究能力方面超越了 GPT-OSS，重点强调 2025 年的智能体（agentic）应用场景。


  **DeepSeek-V3.1** 引入了混合推理架构，可在推理和非推理模式之间切换，并针对智能体工作流和编程进行了优化。该模型具备广泛的长上下文预训练（32k
  上下文约 6300 亿 token，128k 约 2090 亿 token），采用 FP8 训练，并拥有庞大的 MoE 专家规模（约 370 亿）。基准测试显示其性能极具竞争力，尤其在
  SWE-Bench 和其他推理任务中有显著提升。


  在定价方面，DeepSeek API 的收费标准为输入 $0.56/百万 token，输出 $1.68/百万 token。该模型已实现快速的生态集成，包括 Hugging
  Face 权重、Intel 的 INT4 量化以及 vLLM 推理切换支持。社区反馈强调了这种混合设计在智能体和软件工程工作流中的务实性，不过也有人指出其推理模式下尚不支持工具调用（tool
  use）。'
id: MjAyNS0w
models:
- command-a-reasoning
- deepseek-v3.1
people:
- artificialanlys
- reach_vb
- scaling01
- cline
- ben_burtenshaw
- haihaoshen
- jon_durbin
- _akhaliq
- willccbb
- teortaxestex
title: Cohere Command A Reasoning 击败了 GPT-OSS-120B 和 DeepSeek R1 0528。
topics:
- agentic-ai
- hybrid-models
- long-context
- fp8-training
- mixture-of-experts
- benchmarking
- quantization
- reasoning
- coding-workflows
- model-pricing
---

**一个新的 SOTA 开源模型。**

> 2025/8/20-2025/8/21 的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 和 29 个 Discord（229 个频道，7429 条消息）。预计节省阅读时间（以 200wpm 计算）：605 分钟。我们的新网站现已上线，具有完整的元数据搜索和美观的 vibe coded 呈现方式。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

我们上次关注 [Cohere 的 Command A 是在三月份](https://news.smol.ai/issues/25-03-17-ainews-coheres-command-a-claims-3-open-model-spot-after-deepseek-and-gemma)，然后是上周他们的 [70 亿美元 D 轮融资](https://news.smol.ai/issues/25-08-14-cohere-ai2)，但我们没想到这么快就会再次谈论 Cohere —— 根据 Cohere 自己的评估，[Command A Reasoning](https://x.com/cohere/status/1958542682890047511) 让 GPT-OSS 感到羞愧：


![](https://resend-attachments.s3.amazonaws.com/1Ynv4Tjwe5C6Y1k)


对于 2025 年杀手级的 Agent 应用场景来说，重要的是，它是一个非常出色的开源深度研究模型：


![](https://resend-attachments.s3.amazonaws.com/f2mfFNddYUeLEjL)


---

# AI Twitter 回顾

**DeepSeek V3.1：混合推理发布、聚焦 Agent 以及早期结果**

- **DeepSeek-V3.1（思考/非思考混合）**：DeepSeek 推出了一款统一模型，可以通过 `<think></think>` 标记在“推理”和“非推理”模式之间切换，并明确推动 Agent 应用场景和编码工作流。官方公告和演示链接：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1958417062008918312)。社区汇总的显著细节：
    - 训练/后训练：扩展的长上下文预训练（据报道 32k 为 ~630B tokens，128k 为 ~209B tokens），针对下一代加速器调整的 FP8 训练（“UE8M0 FP8”）。架构保持 671B 总参数，~37B 激活 MoE 专家 ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1958432118562041983), [@Anonyous_FPS](https://twitter.com/Anonyous_FPS/status/1958437047359995914), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1958437815710089697))。
    - 能力/限制：推理模式禁用工具/函数调用（tool/function-calling）；非思考模式支持工具使用 ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1958432118562041983))。强调了新的“搜索 Agent”能力 ([@reach_vb](https://twitter.com/reach_vb/status/1958430639595864378))。
    - 基准测试（精选）：GPQA 80.1；AIME 2024 93.1；LiveCodeBench 74.8；开启思考时 Aider Polyglot 为 76.3%；SWE-Bench Verified 从 44.6% 升至 66% ([@reach_vb](https://twitter.com/reach_vb/status/1958430639595864378), [@scaling01](https://twitter.com/scaling01/status/1958438863279681824), [@scaling01](https://twitter.com/scaling01/status/1958438007104549243), [@cline](https://twitter.com/cline/status/1958580433979154720))。Artificial Analysis 的综合“AAI 指数”将 V3.1（推理）定为 60，而 R1 为 59，仍落后于 Qwen3 235B 2507（推理）([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1958432118562041983))。
    - 定价/上下文：DeepSeek API 输入 $0.56/M，输出 $1.68/M；deepseek-chat-v3.1 具有 164k 上下文窗口 ([@scaling01](https://twitter.com/scaling01/status/1958438863279681824), [@cline](https://twitter.com/cline/status/1958580433979154720))。
    - 生态系统支持迅速落地：HF 权重和推理提供商 ([@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1958449429511352549))，Intel 的 INT4 量化 ([@HaihaoShen](https://twitter.com/HaihaoShen/status/1958507863749325197))，vLLM 推理切换 ([@vllm_project](https://twitter.com/vllm_project/status/1958580047658491947))，SGLang 工具调用 + 思考标志解析器 ([@jon_durbin](https://twitter.com/jon_durbin/status/1958488353478758599))，Chutes 托管/定价 ([@chutes_ai](https://twitter.com/chutes_ai/status/1958507978476106196))，Baseten 延迟追踪 ([@basetenco](https://twitter.com/basetenco/status/1958515897972232526))，anycoder 集成 ([@_akhaliq](https://twitter.com/_akhaliq/status/1958488877024362966))。
    - 社区总结：强大的 Agent/编码提升和效率；在某些综合指数中，与 R1/V3 相比整体能力差异较小；混合设计符合务实的 Agent 和 SWE 工作流；对推理模式下缺失工具使用的担忧；认知分歧在于它是“小版本更新”还是“有意义的 Agent 进步” ([@willccbb](https://twitter.com/willccbb/status/1958420877537849801), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1958437173948023127), [@reach_vb](https://twitter.com/reach_vb/status/1958430639595864378), [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1958432118562041983))。

**Cohere 的 Command A Reasoning 及其他新推理模型**

- **Cohere – Command A Reasoning (权重开放)**：Cohere 发布了其面向企业的推理模型，并开放了用于研究/私有部署的权重；商业用途需要 Cohere 许可。重点在于安全性与实用性的平衡、减少过度拒绝，以及强大的 Tool-use/Agentic 基准测试表现。可在 Cohere 平台和 Hugging Face 上获取；在 anycoder 和 Inference Providers 中实现了首日集成 ([@cohere](https://twitter.com/cohere/status/1958542682890047511), [@Cohere_Labs](https://twitter.com/Cohere_Labs/status/1958576284763611322), [@reach_vb](https://twitter.com/reach_vb/status/1958563034810446169), [@_akhaliq](https://twitter.com/_akhaliq/status/1958602589681197494), 关于许可的讨论: [@scaling01](https://twitter.com/scaling01/status/1958561844810903708))。
- **NVIDIA Nemotron Nano 2**：作为一款混合 Mamba-Transformer 推理模型发布；推文中公开的细节有限，但随着 NVIDIA 小算力推理产品线的演进，值得持续关注 ([@_akhaliq](https://twitter.com/_akhaliq/status/1958545622618788174))。

**Google AI：Gemini 效率论文、Agentic 搜索、Veo 访问权限及政府平台**

- **Gemini 推理效率实测**：Google 分享了详细的方法论和结果：Gemini Apps 的中位数文本 Prompt 消耗约 0.24 Wh 电量和 0.26 ml 水；从 2024 年 5 月到 2025 年 5 月，由于模型/系统效率提升和更清洁的能源，每个中位数 Prompt 的能耗下降了 33 倍，碳足迹下降了 44 倍。推文中附有技术论文和博客链接 ([@JeffDean](https://twitter.com/JeffDean/status/1958525015722434945))。另有评论指出资料中提到了 Gemini 的 Hybrid-reasoning 架构 ([@eliebakouch](https://twitter.com/eliebakouch/status/1958603730951029157))。
- **AI Mode 具备 Agentic 能力**：Search 中的 AI Mode 可以规划并执行多步任务（例如，跨网站进行具有实时可用性的餐厅预订）、个性化结果并共享会话上下文；正以英文版向 180 多个国家/地区推广 ([@Google](https://twitter.com/Google/status/1958530316072534323), [@GoogleAI](https://twitter.com/GoogleAI/status/1958561117833228705), [@rmstein](https://twitter.com/rmstein/status/1958552694626607616))。
- **Veo 3 访问权限与演示**：Google 预告将通过 Gemini 应用开放更广泛的 Veo 3 访问权限，并为此准备了 TPU 算力 ([@GeminiApp](https://twitter.com/GeminiApp/status/1958558340163699008), [@joshwoodward](https://twitter.com/joshwoodward/status/1958555951344345461))。Google Devs 发布了一个 Next.js 模板，用于使用 Veo 3 和 Imagen 4 构建浏览器内的 AI 视频工作室 ([@googleaidevs](https://twitter.com/googleaidevs/status/1958599306472206349))。
- **Gemini for Government**：与 @USGSA 合作，为美国联邦政府用途扩展了安全 AI 平台（包括 NotebookLM 和 Veo）；宣传对符合条件的联邦雇员“几乎无成本” ([@sundarpichai](https://twitter.com/sundarpichai/status/1958538684208476611))。

**推理、RL 与评估：新方法与基准测试**

- **用于 LLM 推理的 RL（综述）**：实证总结“第一部分：技巧还是陷阱？”系统地探讨了用于推理的 RL 算法改进，强调了规模/算力限制，以及即使在 8B 模型上收益也各不相同 ([@nrehiew_](https://twitter.com/nrehiew_/status/1958521596492071411))。
- **DuPO – Dual Preference Optimization**：通过对偶性生成自监督反馈，从而在无需外部标注的情况下实现可靠的自我验证；框架可逆性示例（例如，反向推导数学解以恢复隐藏变量）。论文 + 讨论：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1958467512296939593), [@teortaxesTex](https://twitter.com/teortaxesTex/status/1958415225952035274)。
- **PRM 与 RL/搜索的统一**：“Your Reward Function for RL is Your Best PRM for Search-Based TTS” 提出通过 AIRL+GRPO 从正确轨迹中学习密集动态 PRM，既可用作 RL Critic，也可用作搜索启发式算法 ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1958470481050534069))。
- **KV cache 争论**：一个流传甚广的提醒指出，在某些情况下重新计算 KV 可能优于存储它，这引发了将其与类 MLA 权衡（而非“无 KV”）进行对比的讨论 ([@dylan522p](https://twitter.com/dylan522p/status/1958367037773824095), 回复: [@giffmana](https://twitter.com/giffmana/status/1958575876540428379))。
- **新评估资产**：
    - MM-BrowseComp：224 个面向 Agent 的多模态 Web 任务（文本+图像+视频）；包含代码+GitHub 数据集+arXiv ([@GeZhang86038849](https://twitter.com/GeZhang86038849/status/1958381269617955165))。
    - Kaggle Game Arena（纯文本国际象棋）：类 Elo 机制的模型排名 ([@kaggle](https://twitter.com/kaggle/status/1958546786081030206))。
    - ARC-AGI-3 预览：新增 3 个用于 Agent 评估的公开留存游戏 ([@arcprize](https://twitter.com/arcprize/status/1958597816823202216))。

**系统与工具：API、推理服务与开发基础设施**

- **OpenAI Responses API 更新**：新的 “Connectors” 可在一次调用中从 Gmail/Calendar/Dropbox 等提取上下文；“Conversations” 增加了持久化线程存储（消息、工具调用、输出），因此你无需运行自己的聊天数据库；包含演示应用和文档 ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1958660207745409120), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1958660216624751097), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1958660224019247176)；概览：[@gdb](https://twitter.com/gdb/status/1958691151139283454))。
- **Agent/开发工具**：
    - Cursor + Linear：直接从 issue/评论启动 Agent ([@cursor_ai](https://twitter.com/cursor_ai/status/1958627514852811034))。
    - vLLM 和 SGLang：提供一流的 DeepSeek-V3.1 Think/Non-Think 支持 ([@vllm_project](https://twitter.com/vllm_project/status/1958580047658491947), [@jon_durbin](https://twitter.com/jon_durbin/status/1958488353478758599))。
    - MLX 生态系统：mlx-vlm 0.3.3 新增 GLM-4.5V, Command-A-Vision；JinaAI mlx-retrieval 支持在 M3 Ultra 上以约 4000 tok/s 的速度运行本地 Gemma3-270m embeddings/rerankers ([@Prince_Canuma](https://twitter.com/Prince_Canuma/status/1958469233622327785), [@JinaAI_](https://twitter.com/JinaAI_/status/1958547803489415195))。
    - 解析/RAG：LlamaParse 新增引用 + 模式（cost-effective/Agentic/Agentic+）；Weaviate 的 Elysia 发布了带有实时推理可视化功能的决策树 Agentic RAG ([@VikParuchuri](https://twitter.com/VikParuchuri/status/1958520215844655576), [@weaviate_io](https://twitter.com/weaviate_io/status/1958568536420299184))。
    - LlamaIndex：“vibe-llama” CLI 为多种编程 Agent 构建上下文感知规则脚手架 ([@llama_index](https://twitter.com/llama_index/status/1958656414295237014))。
    - 托管：W&B Inference 新增 DeepSeek V3.1（$0.55/$1.65/M tok），Chutes 定价托管，Baseten 库入口 ([@weave_wb](https://twitter.com/weave_wb/status/1958681269484880026), [@chutes_ai](https://twitter.com/chutes_ai/status/1958507978476106196), [@basetenco](https://twitter.com/basetenco/status/1958515897972232526))。
- **应用产品**：Perplexity Finance 在各平台推出针对印度股票的自然语言选股功能 ([@AravSrinivas](https://twitter.com/AravSrinivas/status/1958385027185877066), [@jeffgrimes9](https://twitter.com/jeffgrimes9/status/1958364311178674232))。

**研究亮点 (视觉, 多模态, 3D, 具身智能)**

- **MeshCoder**：LLM 驱动的从点云生成结构化网格代码 ([@_akhaliq](https://twitter.com/_akhaliq/status/1958379365147775414))。
- **RynnEC**：连接多模态 LLM 与具身控制 ([@_akhaliq](https://twitter.com/_akhaliq/status/1958380616711299561))。
- **Tinker Diffusion**：无需逐场景优化的多视图一致 3D 编辑 ([@_akhaliq](https://twitter.com/_akhaliq/status/1958380980000981208))。
- **RotBench**：诊断 MLLM 在图像旋转识别上的表现 ([@_akhaliq](https://twitter.com/_akhaliq/status/1958635243197325625))。
- 其他社区笔记：Goodfire AI 关于如何有效发现稀有、不良的训练后行为，并通过行为引导/可解释性进行缓解 ([@GoodfireAI](https://twitter.com/GoodfireAI/status/1958567217089716334))。

**热门推文 (按互动量排序)**

- DeepSeek-V3.1 发布：混合推理、聚焦 Agent、公开演示 ([@deepseek_ai](https://twitter.com/deepseek_ai/status/1958417062008918312), 16.3k+)
- Google 的 Gemini 推理效率论文：每条 prompt 的能源消耗同比减少 33 倍，碳排放减少 44 倍，中位数 prompt 消耗 0.24 Wh 电量和 0.26 ml 水 ([@JeffDean](https://twitter.com/JeffDean/status/1958525015722434945), 3.8k+)
- Ernest Ryu 关于近期凸优化相关结果的讨论（细致的数学观点推文） ([@ErnestRyu](https://twitter.com/ErnestRyu/status/1958408925864403068), 3.1k+)
- Perplexity Finance 印度股票选股功能向所有用户开放 ([@AravSrinivas](https://twitter.com/AravSrinivas/status/1958385027185877066), 2.3k+)
- Alex Wang 谈 Meta Superintelligence Labs 的投资轨迹 ([@alexandr_wang](https://twitter.com/alexandr_wang/status/1958599969151361126), 2.6k+)
- François Chollet 澄清其长期立场：看好深度学习 Scaling 对实用性的提升，但不认为仅靠 Scaling 就能实现 AGI ([@fchollet](https://twitter.com/fchollet/status/1958410017683681698), 900–1000+)

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. DeepSeek V3.1：Anthropic API 兼容性 + Thinking 模式基准测试

- [**DeepSeek-V3.1 实现了 Anthropic API 兼容性**](https://i.redd.it/0pp8mwjkkbkf1.jpeg) ([Score: 265, Comments: 31](https://www.reddit.com/r/LocalLLaMA/comments/1mw3nat/deepseekv31_implements_anthropic_api_compatibility/))：根据官方指南 (https://api-docs.deepseek.com/guides/anthropic_api)，**DeepSeek-V3.1 增加了与 Anthropic API 的即插即用兼容性。截图显示通过 npm 安装 Anthropic/Claude 客户端，设置环境变量（API key/base URL）指向 DeepSeek，然后通过 Anthropic 的 messages 接口调用 DeepSeek 模型——使现有的集成 Anthropic 的应用能够以极少的代码改动切换后端。** 评论指出文档中存在拼写错误（“Anthoripic”），并询问这是否能让他们在将 DeepSeek 作为模型后端的同时，利用兼容 Claude 的工具链，作为应对近期价格变动的省钱方案。
    - Anthropic API 兼容性意味着你可以将现有的 Claude 客户端（messages + tool-use）指向 DeepSeek-V3.1，并保持相同的 tools JSON schema 和调用流程；唯一的改动是 base URL 和模型名称。这使得在 DeepSeek 作为后端时能直接使用 Claude Tools，如果应用能容忍模型差异，这将是成本/性能优化的利器。文档：[Messages API](https://docs.anthropic.com/claude/reference/messages_post), [Tool use](https://docs.anthropic.com/en/docs/tool-use)。
    - 一位用户报告了 404 错误：`{"error_msg":"Not Found. Please check the configuration."}`，这通常表示在 Anthropic 兼容适配层中端点或 Header 不匹配。请检查准确的 base URL/路径（例如 `/v1/messages`）、必需的 Header（`anthropic-version`、`x-api-key`、`content-type`）以及有效的 `model` ID；某些提供商在使用工具时还需要 `anthropic-beta` tools Header。在不覆盖 host 的情况下对非默认 base URL 使用 Anthropic SDK 通常会导致此 404 错误。
    - 几位用户指出这反映了一个更广泛的趋势（例如 kimi-k2、GLM-4.5），即厂商提供兼容层以降低切换成本。与 Anthropic/OpenAI API 的兼容性让团队能够以极少的代码改动复用客户端、提示词工具和函数调用规范，从而实现后端的快速 A/B 测试。
- [**DeepSeek V3.1 (Thinking) 聚合基准测试（对比 gpt-oss-120b）**](https://www.reddit.com/gallery/1mwexgd) ([Score: 150, Comments: 61](https://www.reddit.com/r/LocalLLaMA/comments/1mwexgd/deepseek_v31_thinking_aggregated_benchmarks_vs/))：**帖子对比了 DeepSeek V3.1 (Thinking) 与 gpt-oss-120b (High)：两者都提供约 128–131K 上下文，但 DeepSeek 是大型 MoE（总参数** `671B` **，激活参数** `37B` **），而 gpt-oss-120b 较小（总参数** `120B` **，激活参数** `5.1B` **）。报告的“智能指数”综合得分相似（** `60` **对** `61` **），DeepSeek 在“编程指数”上领先（** `59` **对** `50` **），数学得分未指明；然而，延迟/吞吐量和成本差异巨大：DeepSeek 在“500 token + 思考”时需要约** `~127.8s` **，速度约** `~20 tok/s` **，价格为** `$0.32 / $1.15` **（输入/输出），而 gpt-oss-120b 约为** `~11.5s` **、** `~228 tok/s` **且价格为** `$0.072 / $0.28` **。** 评论者质疑基准测试的有效性：引用的“Artificial Analysis Coding Index”中 gpt-oss-20b 得分为 `54`，超过了 gpt-oss-120b (`50`) 和 Claude Sonnet 4 thinking (`53`)，引发了“这里肯定有问题……”的质疑，有人总结道 *现在的基准测试几乎没用了*。另一种观点认为，尽管 gpt-oss 性能与体积比很高，但在绝对性能上仍无法与约 700B 参数的 DeepSeek 竞争。
    - 几位用户注意到 “Artificial Analysis Coding Index” 中的异常：**gpt-oss 20B (high)** 得分为 `54`，略高于 **Claude Sonnet 4 Thinking** 的 `53`，并击败了 **gpt-oss 120B (high)** 的 `50`。较小的 20B 模型超越 120B 模型和 Claude 的思考模式，表明元基准测试方法中可能存在聚合偏差、任务选择偏倚或归一化处理不当。
    - 实践者报告了不同的真实测试结果：据称 **DeepSeek (V3.1/“Whale” ~700B 级别)** 在 SWE 风格的任务上表现优于 gpt-oss，具有“接近 Sonnet 级别”的解决方案质量、强大的指令遵循能力和可靠的工具调用。这与聚合指数形成对比，暗示交互式、工具增强的 SWE 评估可能会揭示基准测试组合未能捕捉到的优势。
    - 批评集中在该基准测试是一个元聚合（“零独立思考”），即现有测试的加权混合，缺乏全新的任务设计或透明的校准。评论者认为这种汇总可能不可靠——会产生对包含的基准测试的过拟合，掩盖不同任务类型间的差异，并产生违反直觉的排名倒挂（例如 20B > 120B）。

- [**Deepseek V3.1 终究没那么糟...**](https://www.reddit.com/gallery/1mw3j7l) ([Score: 154, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1mw3j7l/deepseek_v31_is_not_so_bad_after_all/)): **楼主（OP）认为 DeepSeek V3.1 被误判了，因为它针对的是不同的优先级——即速度和 Agentic/任务执行用例——在这些方面它表现良好。提到的一个显著的实际更新是 DeepSeek 现在支持兼容 Anthropic 的 API 格式，从而能够与为 Claude/Claude Code 构建的客户端进行无缝集成（参见 Anthropic API 文档：https://docs.anthropic.com/en/api/），这可能为其他提供商提供了一个更低成本的替代方案。** 评论大多指出，并没有广泛的共识认为它“很糟”，并敦促在评判模型质量之前，先等待几周让工作流、Prompting 和集成趋于成熟。
    - DeepSeek 对 **Anthropic API 格式**的新支持意味着它可以作为期望 Claude 风格请求（Messages、Tool Use/Function Calling 语义）的客户端/工具的近乎无缝替代品，这可能让团队以极小的衔接成本将其接入 **Claude Code**。这降低了集成摩擦，并可能为目前绑定在 Anthropic/OpenAI 上的工作负载提供更便宜的替换方案，具体取决于延迟/吞吐量/成本的权衡；参考 Anthropic 的 Messages API 规范：https://docs.anthropic.com/claude/reference/messages_post 以及 Claude Code：https://www.anthropic.com/news/claude-code。
    - 一位评论者指出了 Benchmark 的泛滥，含蓄地呼吁建立“Benchmark 的 Benchmark”——即标准化我们如何评估各个评估套件的可靠性、方差和污染。实际上，这强调了报告方法论（Prompting、Context Windows、Temperature、Seed Control）的必要性，并倾向于稳健的多维度评估，而不是挑选出来的排行榜，尤其是当像 DeepSeek V3.1 这样的模型在早期被零散的指标评判时。
- [**热爱 DeepSeek 这支小而强大的团队**](https://i.redd.it/38d427vmpdkf1.png) ([Score: 801, Comments: 39](https://www.reddit.com/r/LocalLLaMA/comments/1mwbsww/love_small_but_mighty_team_of_deepseek/)): **该帖子是一个梗图，指出了 DeepSeek API 文档中的一个拼写错误，将 “Anthropic API” 错拼为 “Anthoripic API”（见高亮截图：https://i.redd.it/38d427vmpdkf1.png）。标题/正文将其描述为一个快速行动的小团队在快速发布文档，含蓄地强调了文档 QA/打磨与速度之间的权衡，同时引用了文档中与 Anthropic 相关的部分。** 热门评论认为这个拼写错误表明文档是由人类编写的（真实性的标志），而不是由 LLM 生成的，并开玩笑说 LLM “永远不会犯那个拼写错误”，称其为“真正的软件工程师”的标志。
    - 

### 2. 模型发布/移植：DeepSeek-V3.1 HF Card 和 Kimi-VL-A3B-Thinking GGUF (llama.cpp PR #15458)

- [**deepseek-ai/DeepSeek-V3.1 · Hugging Face**](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) ([Score: 508, Comments: 78](https://www.reddit.com/r/LocalLLaMA/comments/1mw3c7s/deepseekaideepseekv31_hugging_face/)): [**DeepSeek‑V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) 是一个** `671B` **参数的 MoE（激活参数约** `37B`**）混合模型，通过 Chat Templates 支持“思考（Thinking）”和“非思考”模式，其 Post-training 专注于 Tool/Agent 使用，且在同等质量下推理速度比 R1-0528 更快。长上下文在 V3 Base 的基础上通过两个阶段进行了扩展：32K（扩展 10 倍至** `630B` **Tokens）和 128K（扩展 3.3 倍至** `209B` **Tokens），采用 UE8M0 FP8 Microscaling 训练；MIT 许可的权重（Base/Instruct）已在 HF 上发布。报告的 Benchmark 包括：HLE（带 Search+Python）为** `29.8%` **（对比 R1-0528 的** `24.8`**，GPT-5 Thinking 的** `35.2`**，o3 的** `24.3`**，Grok 4 的** `38.6`**，Gemini Deep Research 的** `26.9`**）；SWE-Bench Verified（不带思考模式）为** `66.0%` **（对比 R1-0528 的** `44.6`**，GPT-5 Thinking 的** `74.9`**，o3 的** `69.1`**，Claude 4.1 Opus 的** `74.5`**，Kimi K2 的** `65.8`**）；以及 Terminal Bench (Terminus-1) 为** `31.3%` **（对比 o3 的** `30.2`**，GPT-5 的** `30.0`**，Gemini 2.5 Pro 的** `25.3`**）。Model Card 强调了改进的 Tool Calling、混合思考模式以及定义的 Prompt/Tool/Search-Agent 格式；更广泛的评估引用了强劲的得分（例如 MMLU-Redux** `93.7`**，LiveCodeBench** `74.8`**，AIME’24** `93.1`**）。** 热门评论指出了 Agentic 部署的注意事项：Prompt/Framework 驱动用户体验，DeepSeek 缺乏品牌化的 Search/Code Agent（不像 OpenAI/Anthropic/Google），且 Serverless 托管商可能会通过低精度、专家剪枝或糟糕的采样降低质量。观察者推断 R1+V3 的合并和默认的 `128K` 上下文优化了 Agentic Coding 的 Token TCO；一些退步（GPQA, 离线 HLE）表明 V3 家族已接近极限，引发了对 V4 的呼声。此外还澄清了：此次发布是在 Base 之上的 Post-trained 模型，而不仅仅是 Base Checkpoint。

- Agentic/工具使用性能有明显提升：在 HLE（带搜索+Python）上达到 `29.8%`（对比 R1-0528 `24.8%`，GPT-5 Thinking `35.2%`，o3 `24.3%`，Grok 4 `38.6%`，Gemini Deep Research `26.9%`）；在不带思考模式的 SWE-bench Verified 上达到 `66.0%`（对比 R1-0528 `44.6%`，GPT-5 Thinking `74.9%`，o3 `69.1%`，Claude 4.1 Opus `74.5%`，Kimi K2 `65.8%`）；在 TerminalBench (Terminus 1) 上达到 `31.3%`（对比 o3 `30.2%`，GPT-5 `30.0%`，Gemini 2.5 Pro `25.3%`）。注意事项：DeepSeek HLE 运行使用的是纯文本子集；OpenAI SWE-bench 使用了 `477/500` 个问题；Grok 4 可能缺乏网页过滤（可能存在数据污染）。更广泛的表格显示 V3.1-Thinking 在 MMLU-Pro 为 `84.8`，GPQA Diamond 为 `80.1`（参考数据集：https://huggingface.co/datasets/Idavidrein/gpqa），AIME 2025 为 `88.4`，LiveCodeBench 为 `74.8`（https://livecodebench.github.io/），以及 Aider Polyglot 为 `76.3`（https://github.com/Aider-AI/aider），在多个维度上通常落后于顶尖闭源模型。
- 训练/架构更新：V3.1 是一个混合体，通过 chat template 切换思考（thinking）与非思考模式，并声称通过 post-training 实现了更智能的 tool-calling。长上下文扩展（Long-context extension）得到了大幅提升——32K 阶段扩大约 10 倍至 `630B` tokens，128K 阶段扩大 `3.3×` 至 `209B`——并采用 `UE8M0` FP8 规模训练以支持 microscaling 格式；V3.1-Think 的目标是在保持 R1-0528 级别回答质量的同时提供更快的响应。
- 部署/Agent 考量：将 R1 合并到具有 `128K` 默认窗口的 V3 中，旨在降低 Agentic coding 的 TCO（这类场景通常是 token 密集型，而非长 CoT 密集型）。如果服务商以较低精度运行、剪枝专家（prune experts）或采样调优不当，实际的 UX 可能会下降；此外，提示词/Agent 框架（例如来自 OpenAI/Anthropic/Google 的品牌搜索/代码 Agent）会关键性地影响结果，而 DeepSeek 自己的框架尚未公开。在 GPQA 和线下 HLE 上观察到一些退化，且多个公开基准测试强调了不带思考轨迹（thinking traces）的 Agentic 使用。
- [**Kimi-VL-A3B-Thinking-2506-GGUF 终于发布**](https://huggingface.co/ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF) ([评分: 174, 评论: 11](https://www.reddit.com/r/LocalLLaMA/comments/1mw0tc4/finally_kimivla3bthinking2506gguf_is_available/)): **16B 参数的 Kimi-VL-A3B-Thinking-2506 VLM 的 GGUF 格式构建版本现已托管在 Hugging Face ([原始模型](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506), [GGUF 量化版本](https://huggingface.co/ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF))，[llama.cpp PR #15458](https://github.com/ggml-org/llama.cpp/pull/15458) 增加了后端支持。该版本涵盖了** `4/8/16-bit` **(12) 变体；然而，PR 的最新说明报告了一个未解决的问题，即“输出 token 数量仍不正确”，表明目前的推理可能不稳定或不完整。** 评论者庆祝“VLM 终于得到了一些关注”并报告了良好的早期测试结果，而其他人则认为由于 token 计数 bug，它“目前还不可用/无法工作”；另一位用户强调了 Kimi 较低的谄媚性（sycophancy），并希望这种行为能延续下来。
    - 可用性/状态：一位评论者指出最新的 PR 说明称：*“嗯，事实证明输出 token 的数量仍然不正确。但从另一方面看，我没有破坏其他模型”*，这意味着由于输出 token 计数 bug，GGUF 构建版本尚未完全正常工作，在修复落地前可能无法可靠使用。
    - 模态差距：另一位用户报告该模型无法访问视频中的音轨。这与许多开源 VLM/MLLM 仅支持视觉+文本的情况一致；真正的视听理解需要 ASR 前端或端到端的音视频（AV）训练，而 Kimi-VL-A3B 目前显然缺乏这些。
    - 竞品兴趣：用户询问它与 **Qwen3-30B-A3B** 相比如何，但讨论帖中未提供基准测试或正面交锋的结果（例如 MMBench/MMMU 准确率、A3B 解码吞吐量/延迟）。妥善的比较需要在相同的推理设置下进行标准化的视觉 QA 和多图/视频评估。

### 3. 效率与缩放：1–8 bit 量化指南，100k H100 欠缩放，以及 160GB VRAM 本地构建

- [**为什么低比特模型并非完全“脑残”：从 1-bit 梗图到 FP16 研究指南**](https://i.redd.it/5t58iz5u9ekf1.jpeg) ([Score: 299, Comments: 36](https://www.reddit.com/r/LocalLLaMA/comments/1mwevt4/why_lowbit_models_arent_totally_braindead_a_guide/)): **这篇以梗图为类比的文章将有损图像压缩 (JPEG) 与 LLM 量化进行了等同：通过降低权重精度来缩小模型，同时通过混合精度、校准驱动的舍入以及专为超低精度设计的架构来保留显著行为。它重点介绍了低比特训练/推理，如 Microsoft 的 BitNet——1-bit 和三元 (~1.58-bit) Transformer ([BitNet 1-bit](https://arxiv.org/abs/2310.11453), [1.58-bit 后续研究](https://arxiv.org/abs/2402.17764))——以及实用的方案，如 [Unsloth Dynamic 2.0 GGUFs](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)，并指出较大的模型（例如** `~70B`**）能容忍更激进的量化。资源包括一个视觉解释器和一段关于量化感知训练的视频，强调量化并非千篇一律，可以针对每个层/块进行定制，以在压缩次要知识的同时保留语法/推理能力。** 评论建议使用特定任务的校准数据集（例如 Q4-coding vs Q4-writing）来定制量化，并提到一个轶事，即 JPEG 的表现可能优于 Stable Diffusion 1.5 的 VAE 压缩；另一条评论链接了一个戏谑的“0.5-bit”梗图。
    - Stable Diffusion 1.5 的原始 VAE 因重建保真度差而受到指责——一位评论者声称普通 JPEG 在某些图像上的表现优于它，强调了 SD1.5 KL-f8 VAE 相对于传统编解码器会引入明显的伪影和信息丢失。实际意义：在图像生成流水线中，VAE 质量可能是比模型量化更大的瓶颈，因此更换 VAE 或使用更高保真度的解码器可能比单纯调整比特宽度获得更大的收益。
    - 量化依赖于校准数据集，因此领域特定的校准（例如 Q4-coding vs Q4-writing）可以通过使激活/统计数据适应任务分布来实质性地影响准确性。这反映了像 **AWQ** 和 **GPTQ** 这样感知离群值的 PTQ 方法，其中代表性样本驱动每通道/每组的缩放，在不进行重新训练的情况下提高相同比特宽度下关键特征的保留。
    - 实现资源和格式：**ggml** 项目中的官方 **GGUF** 规范 ([文档](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)) 和 **Hugging Face** 对 **ggml** 的技术介绍 ([文章](https://huggingface.co/blog/introduction-to-ggml))，涵盖了低比特推理的细节；1-bit LLM/BitNet 的简明解释 ([视频](https://www.youtube.com/watch?v=7hMoz9q4zv0))；以及实用的 **llama.cpp** 设置/使用指南 ([博客](https://blog.steelph0enix.dev/posts/llama-cpp-guide/))。
- [**前沿 AI 实验室宣传的 100k-H100 训练任务表现不及预期，原因是软件和系统无法高效扩展，浪费了庞大的 GPU 集群**](https://www.reddit.com/gallery/1mw2lme) ([Score: 335, Comments: 81](https://www.reddit.com/r/LocalLLaMA/comments/1mw2lme/frontier_ai_labs_publicized_100kh100_training/)): **文章认为，前沿实验室大肆宣传的约 100k–H100 训练任务在集群级扩展效率方面表现糟糕，因此尽管 GPU 数量巨大，有效吞吐量（例如 MFU/步时）却不及预期，这是由于分布式技术栈（数据/张量/流水线并行编排、ZeRO/FSDP 等优化器分片、NCCL 集合通信、存储 I/O 和调度器拓扑）中的软件和系统瓶颈造成的。该观点指出，如果没有精细的计算/通信重叠、调优的高吞吐量互连 (NVLink/InfiniBand) 和 NCCL 参数，利用率在大规模下会崩溃——浪费了集群的大部分算力——这意味着“更多 GPU”并不等于线性加速；链接的来源 ([Reddit 图集](https://www.reddit.com/gallery/1mw2lme)) 有访问限制，因此细节无法在此直接验证。** 评论者将此与 DeepSeek 等项目进行了对比，后者报告称通过激进的底层优化在约 2k GPU 上取得了强劲成果，暗示驱动吞吐量的是软件成熟度而非集群规模；其他人指出，多 GPU/多节点推理在开源领域仍然困难，真正的超大规模性能通常需要定制的、内部调优的框架，而不是“业余”或盲目模仿的架构。
    - 评论者强调，100k H100 训练任务无法在现成的 PyTorch 技术栈上扩展；在那种节点数量下，你需要拓扑感知的分片/并行、高性能集合通信、精细的通信/计算重叠以及自定义调度器，以避免 all-reduce 和织物网络 (fabric) 瓶颈。如果没有定制的系统工程，同步和互连开销将占据主导地位，GPU 利用率会崩溃，导致集群显得被“浪费”。他们指出，在几十到几百个 GPU 上表现良好的技术栈，在超过几千个节点时往往会因为延迟和协调成本而瓦解。

- DeepSeek 报道的 2k-GPU 训练运行被引用为一个软件效率击败暴力扩展（brute-force scaling）的反例；该团队描述了为最大化单 GPU 吞吐量而进行的广泛优化。核心结论是：算子融合（kernel fusion）、内存/布局调优、减少同步以及严谨的并行策略，可以让较小的集群表现优于工程设计糟糕的 100k-GPU 任务。最初对 DeepSeek 的怀疑凸显了系统工作相对于头条新闻中的 GPU 数量是多么容易被低估。
- 开源推理生态系统说明了分布式系统仍然面临巨大挑战：很少有项目能在单台主机上实现稳健的多 GPU 推理，能处理好多节点（multi-node）的更是凤毛麟角。评论者认为，可扩展的软件通常需要针对具体需求量身定制的框架，而不是盲目模仿（cargo-cult）模式；此外，缺乏接触真正大规模部署的机会限制了学习和验证扩展方法的机会。当组织在没有专业基础设施的情况下尝试 100k-GPU 规模时，这种技能和工具链的差距会导致交付不足。
- [**Pewdiepie 惊人的 160GB VRAM 配置**](https://youtu.be/2JzOe1Hs26Q?si=9Ck53vK9hja3BZD7) ([得分: 165, 评论: 38](https://www.reddit.com/r/LocalLLaMA/comments/1mwme5c/pewdiepies_monstrous_160gb_vram_build/)): **PewDiePie 展示了一个拥有 8× NVIDIA “RTX 4000” 级别 GPU 的工作站（每张 20 GB），总计约** `160 GB` **VRAM 和** `192 GB` **系统 RAM，并声称他可以用一半的 GPU 运行 [Llama 3 70B](https://ai.meta.com/blog/meta-llama-3/)。在 4×20 GB（约** `80 GB`**）上，只有通过低比特量化（≤8-bit，通常为 4-bit）加上张量并行（tensor parallelism），以及可能针对 KV cache/激活值进行一些 CPU offload，70B 的推理才具有可行性；全精度权重（FP16）在不计开销前约为** `140+ GB` **且无法装下。视频：[“意外建造了一台核能超级计算机”](https://youtu.be/2JzOe1Hs26Q)。** 评论者称这种 8×20 GB 的配置非同寻常，并指出它仍然无法托管超大型 MLLM（例如 DeepSeek/Kimi 变体），除非进行极端量化（Q1）或大量 offload；尽管有 `192 GB` RAM，内存仍可能成为瓶颈。其他人认为这是本地 LLM 走向主流的证据，并建议电力效率可能是选择这些配件的原因。
    - 组装细节：非同寻常的 8× RTX 4000 Ada 配置（每张 `20 GB`），总计 `160 GB` VRAM，配以 `192 GB` 系统 RAM（从 96 GB 修正而来）。评论者推测该选择旨在追求能效比，同时依赖 CPU offload；随着更多内存通道的可用，如果需要，RAM 可以进一步扩展，从而提高处理大上下文时的 offload 空间。
    - 模型适配限制：尽管有 160 GB VRAM，评论者指出它可能仍无法端到端地托管像 “Kimi-K2” 或 DeepSeek 这样的前沿模型，除非使用极端量化（如 “Q1”）。一位用户报告称，即使约 `300 GB` 的 VRAM+RAM 组合仍难以容纳 Kimi，这凸显了许多近期 SOTA 模型的内存占用超过了此配置在不进行重度分片（sharding）/offload 及其导致的带宽瓶颈下所能承载的极限。

## 技术性较低的 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

---

# AI Discord 汇总

> 由 X.ai Grok-4 提供的总结之总结
> 

**主题 1. DeepSeek V3.1 发布，评价褒贬不一**

- **DeepSeek V3.1 亮相，引发热议与抱怨**：工程师们称赞 **DeepSeek V3.1** 在非思考模式下 SWE-bench 评分达到 **66**，但对其创意写作和角色扮演方面的失败表示不满，称其尽管在编程方面有潜力，但仍是 *Gemini 2.5 Pro 的略逊版本*。Cursor 用户报告称其在 TypeScript/JavaScript 上的表现稳健，且成本低于 Sonnet，尽管一些人对 *中国 LLM* 持怀疑态度并面临连接故障。
- **DeepSeek V3.1 价格上涨，API 集成扩展**：DeepSeek 将从 2025 年 9 月 5 日起将输入价格上调至 **$0.25-$0.27**，与推理模型（reasoner）成本持平，同时增加 **Anthropic API** 支持以扩大生态系统使用，正如 [DeepSeek 的 X 帖子](https://x.com/deepseek_ai/status/1958417062008918312)所宣布的那样。社区热切期待 9 月份的免费公共访问，并指出付费版 OpenRouter 的响应速度比免费版快。
- **DeepSeek V3.1 进入竞技场，面临封禁**：LMArena 加入了 **DeepSeek V3.1** 和 **deepseek-v3.1-thinking** 进行对战，但 Gemini 的大规模封禁迫使用户转向替代方案，有人调侃道 *我们正被送回 2023 年*。Cursor 开发者在兴奋与怀疑中进行测试，根据 [DeepSeek 的 Hugging Face 页面](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) 的报告，在 *增量改进* 和性能退化之间进行权衡。

**主题 2. 字节跳动的 Seed-OSS 模型引发关注**

- **字节跳动发布 Seed-OSS-36B 猛兽**：字节跳动推出了 **Seed-OSS-36B-Base-woSyn**，这是一个拥有 **36B** 参数、**512K** 上下文的稠密模型，在 **12T tokens** 上训练且不含合成数据。尽管其 *vanilla* 架构缺乏 MHLA 或 MoE，但仍令渴望进行 GPT-ASS 实验的微调者们感到兴奋。Latent Space 和 Nous Research 的用户邀请社区在 [字节跳动的 GitHub 仓库](https://github.com/orgs/bytedance/repositories) 和 Hugging Face 上进行测试，并对其长上下文能力表示赞赏。
- **Seed-OSS GGUF 延迟引发 ASIC 争议**：Nous Research 讨论了 **Seed-OSS-36B** 缺失 GGUF 量化的问题，将其归咎于定制的 vLLM 以及 llama.cpp 中不支持的 *SeedOssForCausalLM* 架构，其中有人链接了 [Aditya Tomar 的 X 帖子](https://x.com/adityastomar_/status/1958048129275805867)，质疑其对 ASIC 的影响。工程师们注意到 qkv head 中的 dropout 和偏置项，以及类似于 Llama 的定制 MLP，推测其用于正则化，但排除了简单重命名为 Llama 的可能性。
- **字节跳动 SEED Prover 斩获 IMO 银牌**：Eleuther 庆祝 **字节跳动 SEED Prover** 在 IMO 2025 中获得银牌，但根据 [字节跳动的博客](https://seed.bytedance.com/en/blog/bytedance-seed-prover-achieves-silver-medal-score-in-imo-2025) 质疑其现实世界的数学能力。Unsloth AI 用户对其 **512K** 上下文和无 QK norm 的设计感到痴迷，期待能有深度解析论文。

**Theme 3. 硬件障碍与基准测试升温**

- **RTX 5090 引发钱包大战**：Unsloth AI 讨论了 **RTX 5090** 为了显存升级而定价 **$2000** 的问题，抱怨 NVIDIA 不支持 P2P/NVLink，同时关注在 4090-5090 机架上通过 Infiniband 进行分布式训练。GPU MODE 用户为家庭 Infiniband 构建了定制 PyTorch 库和 mini-NCCL 后端，称其为绝佳的分布式计算学习技巧。
- **Apple M4 Max 在基准测试中碾压 GGUF**：LM Studio 在 M4 Max 上对 **GPT-OSS-20b** 进行了基准测试，在使用 4-bit 量化和 4K 上下文时，MLX GPU 在 **32W** 功耗下达到 **76.6 t/s**，而 GGUF CPU 在 **43W** 下仅为 **26.2 t/s**。用户通过 ctrl+shift+r 在 4070 TI Super 上调整 CUDA 以支持 flash attention 和 batch 2048 下的 q8_0 KV 量化，修复了 *0 GPUs detected* 错误。
- **AMD 调试器 Alpha 版发布，支持 Wave Stepping**：GPU MODE 发布了无需 amdkfd 依赖的 alpha 版 **AMD GPU 调试器**，在[此演示视频](https://cdn.discordapp.com/attachments/1233704710389764236/1407932291912695949/2025-08-21_06-20-14.mp4?ex=68a88f60&is=68a73de0&hm=6e9ca7ceed29674c6943e989940a6c8e144707797c5abd571e1cfe60d3f3210d)中展示了反汇编和 wave stepping 功能。MI300 以 **3.50 ms** 占据 trimul 排行榜首位，而 B200 达到 **2.15 ms**，H100 以 **3.80 ms** 位居第二。

**Theme 4. 训练技巧与数据集占据主导**

- **GRPO 要求游戏数据集具备智能**：Unsloth AI 建议将多步游戏数据集拆分为独立的提示词以适配 GRPO，并警告全量 PPO 更适合游戏，因为 GRPO 在 LLM *大致知道该做什么* 时表现更佳。成员们推动多元化的 imatrix 校准，例如 [Ed Addario 的数据集](https://huggingface.co/datasets/eaddario/imatrix-calibration)，而非使用 WikiText-raw 来保留多语言量化效果。
- **WildChat 数据集对英文提示词进行去重**：Unsloth AI 发布了 **WildChat-4M-English-Semantic-Deduplicated**，包含 **2000 tokens** 以下的提示词，根据 [Hugging Face 详情](https://huggingface.co/datasets/MasonMac/WildChat-4M-English-Semantic-Deduplicated)，该数据集使用 Qwen-4B-Embedding 和 HNSW 进行语义去重。Qwen3-30B-A3B 通过 llama-bench 在 CPU 上达到 **10 t/s**，RoPE 缩放使 30B/235B 模型支持 **512K** 上下文。
- **R-Zero 无需人工即可进化 LLM**：Moonshot AI 分享了关于无需人工数据自进化 LLM 训练方法的 [R-Zero PDF 研究](https://cdn.discordapp.com/attachments/1371757564005711973/1408153973751545896/R-Zero__A_Comprehensive_Study_of_a_Self-Evolving_LLM_Training_Method.pdf?ex=68a8b515&is=68a76395&hm=dcba93436f636eeec364d08d08e1603131147faaa595637065f2e226772005f2&)，大幅降低了对数据集的依赖。根据 [缩放论文](https://arxiv.org/abs/2508.12104)，Eleuther 的 CoMET 模型在来自 **1.18 亿患者** 的 **115B 医学事件** 上进行了预训练，表现优于监督模型。

**Theme 5. 行业转变与安全障碍**

- **Meta 在 Wang 的统治下重组 AI**：Latent Space 报道称 Meta 将 AI 拆分为由 **Alexandr Wang** 领导的四个团队，解散了 AGI Foundations，Nat Friedman 和 Yann LeCun 向其汇报，据 [Business Insider](https://www.businessinsider.com/meta-ai-superintelligence-labs-reorg-alexandr-wang-memo-2025-8) 称其目标是开发“omni”模型。Yannick Kilcher 推测 Yann LeCun 在 FAIR 的降职预示着 Meta 将退出开源。
- **生成式 AI 为 95% 的机构带来零 ROI**：OpenRouter 引用 [AFR Chanticleer 报告](https://archive.md/IlP7F) 显示，由于缺乏对业务细微差别的学习，**95% 的组织**未能从定制化 AI 中获得回报。Moonshot AI 对比了中国数据中心将“能源视为既定条件”与美国关于电网的争论，详见 [Fortune 文章](https://fortune.com/2025/08/14/data-centers-china-grid-us-infrastructure/)。
- **API 泄露和封禁困扰用户**：一名 OpenRouter 用户因 API key 泄露损失了 **$300**，代理掩盖了 IP；Gemini 的封禁引发了对 AI Dungeon 清洗事件的回忆。LlamaIndex 分享了 [AI safety 调查](https://mukullight.pythonanywhere.com/form)，征集社区对关键问题的看法。


---

# Discord: 高层级 Discord 摘要




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nano-Banana 沦为 McLau's Law 的牺牲品**：成员们开玩笑说 **Nano-Banana** 模型的表现通常低于预期，幽默地将这一现象称为“**McLau's Law**”（引用自一位 **OpenAI** 研究员），并引发了关于[附图](https://cdn.discordapp.com/attachments/1340554757827461211/1407957527987097642/RDT_20250821_0619535468074318472918433.jpg?ex=68a8a6e1&is=68a75561&hm=bfcfd37e574ea905e84f2ff1c9e6ffee1855681ad903050cfe30a367ea5d96f3&)中描绘的 **AI** 当前能力的讨论。
   - 一位用户建议 **Nano-Banana** 产生的结果往往*远低于 nano-banana*。
- **Video Arena 饱受 Bot 宕机困扰**：用户报告 **Video Arena Bot** 离线，导致指令失败且无法生成视频，实际上锁定了 prompt 频道 <#1397655695150682194>、<#1400148557427904664> 和 <#1400148597768720384>。
   - 管理员确认了宕机情况并正在修复，引导用户关注公告频道以获取更新，并表示很快将推出登录功能以防止未来的服务中断。
- **DeepSeek V3.1 登场**：**DeepSeek V3.1** 和 **deepseek-v3.1-thinking** 模型已添加到 LMArena 并可供使用。
   - 共识是 **v3.1** 模型是 *Gemini 2.5 pro 的略逊版本*，尽管它作为 coding 模型很有前景，但在通用能力方面仍需增强。
- **LMArena 用户遭遇数据丢失**：站点故障导致大规模数据丢失，包括聊天记录缺失和无法接受服务条款。
   - 管理员承认了该问题并向用户保证修复工作正在进行中。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **字节跳动发布 Seed-OSS 36B Base 模型**：字节跳动在 Hugging Face 上发布了 **Seed-OSS-36B-Base-woSyn** 模型，这是一个具有 **512K** context window 的 **36B** 稠密模型，在 **12T tokens** 上训练。
   - 成员们渴望尝试用该模型微调 GPT-ASS，认为缺乏合成数据非常有吸引力。
- **GRPO 需要巧妙的数据集设计**：为了将 **GRPO** 用于多步游戏动作，成员建议为每一步设计带有独立 prompt 的数据集。
   - Full PPO 可能更适合游戏，因为 GRPO 对 LLM 主要有效的原因是 *它们起初就大致知道该做什么*。
- **DeepSeek V3.1 的思考能力**：**DeepSeek V3.1** 模型在非思考模式下在 SWE-bench verified 上获得了 **66** 分，引发了成员的热议。
   - 然而，随后有人对其创意写作和角色扮演表现表示担忧，指出 *混合模型在非思考模式下缺乏指令遵循能力和创造力*。
- **RTX 5090 价格引发升级争论**：**RTX 5090** 定价约 **$2000**，引发了关于是否升级的讨论，尤其是考虑到其 **VRAM** 能力对于训练的意义。
   - 一些成员对 **NVIDIA** 的限制表示不满，特别是缺乏 **P2P 或 NVLink**。
- **WildChat-4M-English 发布**：**WildChat-4M-English-Semantic-Deduplicated 数据集**已在 Hugging Face 上线，包含来自 WildChat-4M 数据集的英文 prompt，并使用了多种方法去重。
   - 当前版本包含 **<= ~2000 tokens** 的 prompt，更大的 prompt 将在稍后添加，更多信息见[此处](https://huggingface.co/datasets/MasonMac/WildChat-4M-English-Semantic-Deduplicated)。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Deepseek V3.1 热潮即将来临！**：用户正热切期待 **Deepseek v3.1** 的公开发布，预计从 9 月份开始将免费提供。
   - 用户确认，在 **OpenRouter** 上为 **Deepseek** 模型付费比使用免费模型响应速度更快。
- **OpenRouter API Key 泄露风险！**：一名用户报告因 **OpenRouter API key** 泄露损失了 **$300**，并寻求关于如何识别未经授权使用来源的建议。
   - 用户需对任何泄露的 Key 负责，且攻击者可以使用代理来掩盖其原始 IP。
- **Gemini 面临大规模封号潮！**：用户报告 **Gemini** 正在发生大规模封号，导致许多人寻找替代方案，并回想起由 OpenAI 引发的 AI Dungeon 清洗事件。
   - 用户表示 *我们正被送回 2023 年*。
- **Gemini 输入 Token 触发异常计数！**：一位仪表盘开发者注意到，当输入中包含图像时，**OpenRouter** 对 **Gemini 模型** 的 **input tokens** 计算会出现异常计数，并引用了 [Google AI Developers 论坛](https://discuss.ai.google.dev/t/token-counts-mismatch-9x-discrepancy/72633/2) 上的相关讨论。
   - 该开发者正考虑就此问题向 OpenRouter 团队寻求澄清。
- **大多数机构在生成式 AI 上零回报！**：根据 [AFR Chanticleer 报告](https://archive.md/IlP7F)，**95% 的机构在部署生成式 AI 时获得了零回报**，该报告重点关注了部署 **定制化 AI 模型** 的公司。
   - 报告指出，关键问题在于公司及其技术供应商没有投入足够的时间来确保其定制化 AI 模型能够持续学习业务中的细微差别。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Claude 缓存的不稳定性导致昂贵的难题**：用户报告称 **Claude** 在 *cache reads* 方面遇到问题，导致与受益于可持续缓存的 **Auto** 相比费用增加。
   - 有推测认为 **Auto** 和 **Claude** 秘密地使用了同一个模型，并将 Token 使用量的减少归因于 *安慰剂效应*。
- **Sonic 极速模型在 Cursor 中大放异彩**：社区目前正在 Cursor 中测试新的 **Sonic** 模型，因其速度极快，初步印象相当不错。
   - 虽然在处理新项目时受到好评，但一些用户警告其在大型代码库中的效果可能会下降，并确认 **Sonic 不是 Grok 模型**，其起源仍来自一家 *隐身公司*。
- **Agentwise 作为开源项目觉醒**：**Agentwise** 已开源，支持网站副本、图像/文档上传，并支持超过 100 个 Agent，且承诺将提供 [Cursor CLI 支持](https://discord.com/channels/1074847526655643750/1408047562019049523)。
   - 邀请用户在该项目的专用 Discord 频道中提供反馈，以帮助进一步开发。
- **Cursor 成本确认：API 费用明晰**：关于 Auto Agent 成本的困惑已得到澄清，*pro* 订阅包含了不同供应商的 API 使用成本。
   - 几位用户确认了成本说明，其中一位表示相比 Sonic Agent 更倾向于使用 Auto Agent。
- **DeepSeek 亮相，开发者反应不一**：新的 **DeepSeek V3.1** 模型出现在 Cursor 的选项中，引起了褒贬不一的反应；一些用户遇到了连接问题，而另一些用户则表达了对 *中国 LLM* 的不信任。
   - 尽管存在担忧，但一些人报告说 DeepSeek V3.1 在 **TypeScript** 和 **JavaScript** 方面表现良好，性能出色且比 Sonnet 更便宜。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **CUDA 修复解决了 4070 检测问题**：用户发现通过 **ctrl+shift+r** 将运行时更改为 **CUDA llama.cpp**，可能会解决 LM Studio 在 **4070 TI Super** 显卡上出现的 *"0 GPUs detected with CUDA"* 错误。
   - 他们讨论了通过 `-fa -ub 2048 -ctv q8_0 -ctk q8_0` 等命令来启用 **Flash Attention**、**KV Cache 量化**以及 **Batch Size 为 2048** 的各种配置。
- **GPT-OSS 在 Prompt Eval 上完胜 Qwen**：成员们观察到 **GPT-OSS** 在使用 **3080ti** 进行 Prompt Eval 时达到了 *2k tokens/s*，在 LM Studio 中超越了 **Qwen** 的 *1000 tokens/s*。
   - 一位用户报告称 LM Studio 的 API 调用比聊天界面慢得多（30倍），但在使用 curl 命令 `curl.exe http://localhost:11434/v1/chat/completions -d {"model":"gpt-oss:20b","messages":[{"role":"system","content":"Why is the sun hot?\n"}]}` 时，该问题因未知原因自行解决。
- **Qwen3-30B CPU 配置表现令人惊喜**：使用 [llama-bench](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench)，一位用户在纯 CPU 配置下运行 **Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf** 达到了 **10 tokens/s**。
   - 他们指出性能随线程数而变化，由于扩展性和开销问题，超过一定阈值后收益会递减。
- **MLX 在 M4 Max 上的表现碾压 GGUF**：在 Apple M4 Max 上对 **GPT-OSS-20b** 进行基准测试显示，**MLX (GPU)** 在 **32W** 功耗下达到了 **76.6 t/s (2.39 t/W)**，而 **GGUF (CPU)** 在 **43W** 功耗下仅达到 **26.2 t/s (0.61 t/W)**。
   - 在 **4bit 量化**和 **4k 上下文**下，MLX 证明了其比 GGUF 更快且能效更高，尽管 GGUF 的性能也给他们留下了深刻印象。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Agent 深入研究 M2M 经济**：成员们探讨了**机器对机器 (M2M) 经济**，即 AI Agent 自主交换价值，重点关注*身份与信任、智能合约逻辑和自主性*等挑战。
   - **支出上限、审计日志和保险**等保障措施可能会加速 AI 在交易中的应用，但*真正的信任建立仍需时日*。
- **去中心化 AI 项目的 BOINC 悬赏**：一位成员寻找类似 **BOINC** 的**去中心化 AI 项目**，并指出 [Petals network](https://petals.ml/) 在贡献和模型更新方面面临挑战。
   - 贡献者建议，**财务或活动驱动的激励措施**可以加强去中心化 AI 的发展。
- **Few-Shot 健身提示词展示**：成员们剖析了在针对健身房的 **29,000 token 提示词**中有效使用 **Few-Shot 示例**的最佳策略，强调了 **Prompt Engineering** 的重要性。
   - 建议包括在提示词中提供直接示例，并反复测试较小的分块以提升性能。
- **GPT-5 的思考模式变笨**：一位用户报告称 **GPT-5** 的*思考 (thinking)* 模式产生了直接且**低质量的回复**，类似于旧版本的模型，令人感到沮丧。
   - 另一位成员推测该用户可能超过了*思考配额限制，系统设置为回退 (fallback) 模式而非置灰*。
- **AI 测验生成器产生琐碎问题**：一位成员强调了 **AI 测验生成器**在测验中产生明显错误选项的问题。
   - 另一位成员建议确保*所有选项必须具有合理性*，以改进 AI 的输出并产生更真实的响应。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **PileT5-XL 发声**：来自 **PileT5-XL** 的嵌入张量（embedding tensor）既可以作为 **pile-t5-xl-flan**（生成文本）的指令，也可以作为 **AuraFlow**（生成图像）的提示词（prompt），这表明这些嵌入像语言中的单词一样具有意义。
   - 一位成员对文本反转（textual inversion）感兴趣，尝试将一张黑狗图片配合应用了 pile-t5-xl-flan 的 auraflow 使用，以观察文本是否会将该狗描述为黑色。
- **Cosmos 医疗模型规模化！**：**Cosmos Medical Event Transformer (CoMET)** 模型系列是经过 **1.18 亿患者**、代表 **1150 亿离散医疗事件**（1510 亿 tokens）预训练的仅解码器 Transformer 模型，其表现通常优于或等同于特定任务的监督模型。
   - 这项研究在 [Generative Medical Event Models Improve with Scale](https://arxiv.org/abs/2508.12104) 中进行了讨论，使用了 **Epic Cosmos** 数据集，该数据集包含来自 **310 个医疗系统**、超过 **3 亿份独特患者记录**、**163 亿次就诊**的去标识化纵向健康记录。
- **字节跳动 Prover 获奖**：**字节跳动（Bytedance）的 SEED Prover** 在 [IMO 2025 中获得了银牌分数](https://seed.bytedance.com/en/blog/bytedance-seed-prover-achieves-silver-medal-score-in-imo-2025)。
   - 然而，目前尚不清楚这如何转化为现实世界的数学问题解决能力。
- **隔离 Llama3.2 注意力头**：一位成员隔离了一种特定的 *head*（注意力头），发现 **Llama 3.2-1b instruct** 和 **Qwen3-4B-Instruct-2507** 在不同输出之间的解码结果向量非常相似。
   - 该成员表示，*这两个注意力头似乎促进了非常相似的内容*。
- **寻求 Muon 内核支持**：一位成员表示有兴趣添加 **muon 支持**，并提到了潜在的 **内核（kernel）优化机会**。
   - 他们认为，一旦实现了基础支持，就有空间就这些优化进行协作。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Wang 晋升后 Meta 拆分**：根据 [Business Insider](https://www.businessinsider.com/meta-ai-superintelligence-labs-reorg-alexandr-wang-memo-2025-8) 的报道，Meta 正在将其 AI 业务重组为新任 MSL 负责人 **Alexandr Wang** 领导下的**四个团队**（TBD Lab、FAIR、产品/应用研究、基础设施），**AGI Foundations** 小组将被解散。
   - **Nat Friedman** 和 **Yann LeCun** 现在向 Wang 汇报，**FAIR** 将直接支持模型训练，并且正在考虑开发一个“omni”模型。
- **GPT-5-pro 静默吞掉提示词**：根据[此报告](https://x.com/pvncher/status/1958193631250072024?s=46)，**GPT-5-pro** 正在静默截断超过 **60k tokens** 的提示词，且没有任何警告或错误消息，这使得大型代码库的提示词变得不可靠。
   - 一些用户还报告说，**Cursor** 中的 **GPT-5** 表现得比平时笨得多，有人怀疑正在进行负载卸载（load shedding）。
- **Dropout 灵感来自银行柜员**：一条热门推文声称 **Geoffrey Hinton** 在注意到**轮换的银行柜员**能防止勾结后构思出了 *dropout* ([来源](https://x.com/eigenron/status/1958181550987632927?s=46))。
   - 反应从对这种偶然洞察力的钦佩，到怀疑以及关于从家庭聚会中产生注意力机制的笑话不等。
- **字节跳动发布 Seed-OSS 模型**：字节跳动的 Seed 团队宣布了 **Seed-OSS**，这是一个新的开源大语言模型系列，可在 [GitHub](https://github.com/orgs/bytedance/repositories) 和 [Hugging Face](https://huggingface.co/models) 上获取。
   - 该团队邀请社区对模型、代码和权重进行测试并提供反馈。
- **Wonda 承诺视频革命**：Dimi Nikolaou 介绍了 **Wonda**，这是一个旨在彻底改变视频/音频创作的 AI Agent，称其为“Lovable 之于网站，正如 Wonda 之于内容” ([推文链接](https://xcancel.com/dimireadsthings/status/1957805267799740571))。
   - 早期访问将通过候补名单授予，大约在 **3 周**内发放邀请。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 困扰 ChatGPT**：一位成员发现 **ChatGPT** 在 **CUDA float3 对齐**和**大小**方面给出了言之凿凿但错误的回答，并随后将该话题的难度归因于 **OpenCL** 和 **OpenGL** 实现的复杂性。
   - 该成员已验证 **CUDA** 中不存在填充（padding）。
- **黑客松将于周六上午开始**：**GPU Hackathon** *很可能*在周六上午 **9:30** 左右拉开帷幕，并有暗示称参与者将使用较新的 **Nvidia 芯片**。
   - 有人询问了黑客松的先决条件，但频道内无人回答。
- **AMD GPU 调试器发布首个 Alpha 版本**：一位工程师在[这段视频](https://cdn.discordapp.com/attachments/1233704710389764236/1407932291912695949/2025-08-21_06-20-14.mp4?ex=68a88f60&is=68a73de0&hm=6e9ca7ceed29674c6943e989940a6c8e144707797c5abd571e1cfe60d3f3210d)中展示了其新款 **AMD GPU 调试器**的 Alpha 版本，目前已支持反汇编和 wave 步进。
   - 该调试器不依赖于 **amdkfd KMD**，而是使用微型 UMD 驱动程序和 Linux 内核的 debugfs 接口，旨在成为 **rocdbgapi** 的替代方案。
- **DIY 分布式训练框架出现**：一位成员正在构建自己的 **PyTorch 分布式训练库**和微型 **NCCL** 作为后端，用于在家中的 **4090** 和 **5090** 之间通过 **Infiniband** 进行连接。
   - 另一位成员表示了兴趣，认为这是研究分布式计算细节的好方法。
- **MI300 霸榜 Trimul 排行榜**：`trimul` 排行榜现在显示 **MI300** 的提交分数为 **3.50 ms**，另一个 **MI300** 的提交以 **5.83 ms** 的成绩获得第二名。
   - 一位成员在 **B200** 上以 **8.86 ms** 的成绩获得第 6 名，随后在 `trimul` 排行榜上提升至第 4 名（**7.29 ms**）；另一位成员在 **H100** 上以 **3.80 ms** 的成绩获得第二名。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **福布斯发现缺陷，引发纷争！**：[Forbes](https://www.forbes.com/sites/iainmartin/2025/08/20/elon-musks-xai-published-hundreds-of-thousands-of-grok-chatbot-conversations/) 透露 **Elon Musk 的 xAI** 发布了数十万条 **Grok** 聊天机器人的对话。
   - 当被问及这是否属实时，*@grok* 的回答闪烁其辞，引发了进一步的猜测。
- **LeCun 是要离开、失败还是闲逛？！**：一位用户根据 [Zuckerberg 的帖子](https://www.threads.com/@zuck/post/DMiwjXJSYCd?xmt=AQF06RR3brSSoKTMcQrLDyX6WGtg9UdpubK94sSm8FIPsg)猜测 **Yann LeCun** 可能会离开 **FAIR**。
   - 另一位成员暗示 **LeCun** 可能已被降职，且 **Meta** 正在退出开源模型领域。
- **无限内存是机器强大的必要条件！**：一位成员认为图灵完备性需要无限内存，因此由于内存不足，宇宙无法创造出图灵完备的机器。
   - 另一位成员开玩笑地建议，让计算机足够慢，就可以利用宇宙的膨胀来解决空间问题。
- **新名称，新麻烦：针对 AI 的侮辱性词汇出现！**：一位用户分享了[一篇《滚石》杂志的文章](https://www.rollingstone.com/culture/culture-features/clanker-cogsucker-robot-ai-slurs-viral-1235401262/)，讨论了诸如 *clanker* 和 *cogsucker* 等针对 **AI** 的新侮辱性词汇的出现。
   - 频道内的反应很平淡，但似乎大家都一致认为这些词确实非常不雅。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face Pro 用户遭遇支付问题**：一名用户报告称在未获得服务的情况下被收取了两次 **Pro version** 费用，建议其他用户发送邮件至 website@huggingface.co 并在指定的 [MCP channel](https://discord.com/channels/879548962464493619/1389546106970701865) 寻求帮助。
   - 尽管账户被反复扣费，该用户仍无法获得 **Pro** 服务。
- **AgentX 承诺更智能的 AI 交易**：新的 [**AgentX** 平台](https://www.linkedin.com/posts/alaa-salamah-96167b227_agentx-agentx-api-activity-7364245216851050498-BfRO?utm_source=share&utm_medium=member_desktop&rcm=ACoAADjfvpkBwpCXn_8Pmby7ixjV93Dje5TcmgUHi) 旨在提供一个汇集了最顶尖 AI 大脑——**ChatGPT**、**Gemini**、**LLaMA**、**Grok**——的交易台，它们将共同辩论直到就最佳操作达成一致。
   - 该平台通过让 **LLMs** 辩论最佳决策，力求为交易者提供一个可以完全信赖的系统。
- **成员辩论 SFT 与 DPO**：成员们讨论了 **DPO** (Direct Preference Optimization) 与 **SFT** (Supervised Fine-Tuning) 的有效性，其中一名成员指出 *DPO 与推理（reasoning）没有关系*，但在 **SFT** 之后进行 **DPO** 比单纯使用 **SFT** 能提升结果。
   - 讨论涉及利用 **DPO** 提升性能，然而，其与推理的关系在成员间存在争议。
- **HF Learn 课程受 422 错误困扰**：一名成员报告称 [Hugging Face LLM 课程的一个页面](https://huggingface.co/learn/llm-course/en/chapter12/3a) 宕机并显示 **422 error**。
   - 用户目前无法访问该学习课程中损坏的页面。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **用户发现利用 Gems 优化播客生成的技巧**：用户正在开发工作流，例如[这个示例](https://github.com/para-droid-ai/scratchpad/blob/main/assistant-workflows-tasks-personas/deeper_podcast_synthetic-082025.txt)，利用 **Gems**、**Gemini**、**PPLX** 或 **ChatGPT** 创建更深层的研究框架来生成播客。
   - 关键在于设置 Prompt 来逐段规划整个文稿，从而根据较长的 **YouTube** 视频生成播客。
- **自定义界面允许用户配置播客长度**：用户可以通过 **Customize** 选项（三个点）调整 NotebookLM 中的播客长度，将其延长至 **45-60 分钟**。
   - 指定主题可以让 Bot *专注于特定话题*，而不是指望它将所有重要内容都塞进一个播客中。
- **隐私政策担忧依然存在**：用户正在使用 **Gemini** 和 **NotebookLM** 分析医疗公司的隐私政策和使用条款。
   - 用户对 *向这些公司泄露了多少信息* 感到惊讶，并认为这种理解 **Terms of Use**（使用条款）和 **Privacy policies**（隐私政策）的方法非常有用。
- **Android 应用功能对齐延迟**：用户请求在 NotebookLM Web 端应用和 **Android app** 之间实现更多 **feature parity**（功能对齐），特别是针对学习指南功能。
   - 一位用户表示目前的原生应用 *几乎没用*，因为学习指南依赖于笔记功能，而原生应用中缺失了该功能。
- **NotebookLM API 仍未发布**：虽然 NotebookLM 的官方 API 尚未提供，但用户建议使用 **Gemini API** 作为替代方案。
   - 另一位用户分享了结合 **GPT4-Vision** 和 **NotebookLM** 的策略，以 *快速消化带有标注的复杂 PDF 原理图*。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **字节跳动发布长上下文模型**：根据[这张图片](https://cdn.discordapp.com/attachments/1149866623109439599/1407959284280459305/image.png?ex=68a8a883&is=68a75703&hm=b8a5430da1f445204c76334cef07358c8d9b815da989424483a5ecf3bc65c790)，字节跳动发布了一个具有极长上下文的基础模型，其特点是没有 **MHLA**、没有 **MoE**，甚至没有 **QK** norm。
   - 该模型的架构被描述为 *vanilla*（原生），人们希望即将发布的论文能提供更多见解。
- **Seed-OSS-36B 缺失 GGUF 引发猜测**：用户询问为何 **Seed-OSS-36B** 缺失 **GGUF** 版本，并指出这类版本通常出现得很快，参考[此链接](https://x.com/adityastomar_/status/1958048129275805867)质疑其对 **ASICs** 的影响。
   - 有建议认为延迟可能源于自定义的 **vllm** 实现，由于 `architectures: ["SeedOssForCausalLM"]`，该架构目前尚未被 **llama.cpp** 支持。
- **Seed 模型采用 Dropout 和 Bias**：**Seed** 模型结合了自定义 **MLP** 和类似于 **LLaMA** 的 attention 机制，但具有 dropout、输出偏置项（bias term）以及 **qkv** 头的偏置项。
   - 这些新增项被推测用作正则化技术；然而，该模型经过的训练轮数（epochs）仍不得而知，并已确认仅将其重命名为 **LLaMA** 将无法运行。
- **Qwen 通过 RoPE 扩展至 512k 上下文**：根据 [Hugging Face 数据集](https://huggingface.co/datasets/eaddario/imatrix-calibration)，**30B** 和 **235B** 的 **Qwen 2507** 模型可以使用 **RoPE** scaling 达到 **512k** 上下文。
   - 这些数据集用于生成重要性矩阵（**imatrix**），有助于在量化过程中最大限度地减少误差。
- **Cursor 的 Kernel 博客赢得喝彩**：成员们分享了 [Cursor kernel 博客](https://x.com/stuart_sul/status/1957927497351467372)的链接。
   - 许多人一致认为 Cursor 在这方面做得非常出色（*cursor cooked*）。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **DeepSeek V3.1 亮相，小幅提升**：新的 **DeepSeek V3.1** 模型已发布，一些成员指出它就像是一个“增量改进”，并伴随一些退步，参考 [DeepSeek 官方页面](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)。
   - 社区正密切关注其性能，以观察细微的提升和潜在的缺点。
- **DeepSeek 引入 Anthropic API 集成**：正如 [X 平台](https://x.com/deepseek_ai/status/1958417062008918312)上宣布的那样，**DeepSeek** 现在支持 **Anthropic API**，扩展了其功能和覆盖范围。
   - 这种集成使用户能够在 **Anthropic** 生态系统中使用 **DeepSeek**，为 AI 解决方案的开发提供了灵活性。
- **R-Zero LLM 无需人工数据即可进化**：一份关于 **R-Zero** 的综合研究 [PDF](https://cdn.discordapp.com/attachments/1371757564005711973/1408153973751545896/R-Zero__A_Comprehensive_Study_of_a_Self-Evolving_LLM_Training_Method.pdf?ex=68a8b515&is=68a76395&hm=dcba93436f636eeec364d08d08e1603131147faaa595637065f2e226772005f2&) 被分享，这是一种自进化的 **LLM 训练方法**，从零人工数据开始并独立改进。
   - 该方法标志着与传统 **LLM 训练**的背离，有可能减少对人工标注数据集的依赖。
- **中国避开数据中心能源困境**：一位成员指出，在中国，“能源供应被视为理所当然”，这与美国关于数据中心功耗和电网限制的辩论形成对比，参考 [Fortune 文章](https://fortune.com/2025/08/14/data-centers-china-grid-us-infrastructure/)。
   - 这种方法的差异可能会使中国 AI 公司在扩展能源密集型模型方面获得竞争优势。
- **Kimi K2 期待更好的图像生成**：一位成员指出，如果 **Kimi K2** 能结合“比 GPT-5 更好的图像生成”，它将变得更加强大（OP），并分享了 [Reddit 链接](https://www.reddit.com/r/ChatGPT/s/vUrGedSwY5)。
   - 集成增强的图像生成功能将使 **Kimi K2** 成为一个更全面、更具竞争力的 AI 助手。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro 遇挫而 Flash 表现出色**：有用户报告 **Gemini 2.5 Flash** 功能正常，而 **Gemini 2.5 Pro** 则持续失败；不过，在配置计费后，`gemini/gemini-2.5-pro-preview-06-05` 可以正常运行。
   - 另一位用户报告了一个 **qwen-cli** 进程产生了 **$25** 的费用并申请退款，这突显了模型性能和计费方面可能存在的不一致性。
- **用户遭遇意外的 Qwen CLI 扣费**：一名用户在进行 Google OAuth 认证后使用 **qwen-cli** 产生了 **$25** 的费用，而该用户原本预期使用的是来自阿里云的免费额度。
   - 该用户提交了支持工单，引用了控制台中记录的 *一次 $23 且无输出的调用* 来对这笔意外费用提出申诉。
- **社区对 GPT-5 Mini 模型进行基准测试**：由于完整版 **gpt-5** 的速率限制，社区成员正积极对 **gpt-5-mini** 和 **gpt-5-nano** 进行基准测试，一位用户声称 *gpt-5-mini 非常出色且价格低廉*。
   - 目前已有 **gpt-5-mini** 的基准测试结果和 PR，反映了社区对评估更小、更易获取的模型的浓厚兴趣。
- **DeepSeek v3.1 价格上涨**：从 2025 年 9 月 5 日起，DeepSeek 将把两个模型的输入价格从 **$0.25** 上调至 **$0.27**，以匹配 reasoner 模型的价格。
   - 价格上调以匹配 **deepseek 3.1** 模型反映了定价策略的变化。
- **OpenRouter 需要“思考”模式**：用户注意到 **OpenRouter** 缺乏用于增强推理的原生“思考”模式，但可以通过命令行启用：`aider --model openrouter/deepseek/deepseek-chat-v3.1 --reasoning-effort high`。
   - 社区成员建议更新模型配置以解决这一功能缺失。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Marimo Notebooks 作为 Jupyter 替代方案兴起**：一位成员发布了 [**marimo notebooks** 教程](https://www.youtube.com/watch?v=2aepn9uRVOM)，强调了它在 **DSPy Graph RAG** 想法迭代中的应用，它能同时作为 notebook、脚本和 App 使用。
   - 接下来的视频将探索 **DSPy modules** 的优化，并在当前向新用户介绍 **marimo** 的教程基础上进行扩展。
- **可读性辩论：DSPy 代码先遭抨击后获支持**：在一位成员驳斥了 **IBM AutoPDL** 关于不可读性的指控后，他们辩护称 **DSPy 的代码** 和 **prompts** 具有极高的人类可读性和清晰度。
   - 辩护强调了代码的可访问性，使其易于理解和操作。
- **GEPA 登陆 DSPy v3.0.1**：成员们确认 **GEPA** 已在 **dspy** 版本 **3.0.1** 中可用，如附带的 [截图](https://cdn.discordapp.com/attachments/1161519469319946286/1407936409615990904/image.png?ex=68a89336&is=68a741b6&hm=72219936a525599fc3faca9d127d106a09e0639e9eeae1564a8bbfc196b07ffa&) 所示。
   - 在微调期间，一位成员询问在 **dspy.InputField()** 和 **dspy.OutputField()** 中使用 *“基础描述 (vanilla descriptions)”* 是否常见，以便让优化器自由思考。
- **Pickle 问题：DSPy 程序未保存**：一位用户报告了保存优化后程序的问题，指出即使使用了 `optimized_agent.save("./optimized_2", save_program=True)`，元数据也仅包含依赖版本而未包含程序本身。
   - 当另一位用户为 **GEPA** 设置了 **32k** 的最大上下文长度但仍收到被截断的响应时，成员们讨论了长推理的复杂性以及多模态设置中潜在的问题。
- **RAG vs 拼接：百万文档之争**：成员们辩论了对于处理税法或农作物保险文档等任务，**RAG** (Retrieval-Augmented Generation) 还是简单的**拼接 (concatenation)** 更合适。
   - 辩论承认，虽然 **RAG** 常被视为大材小用，但数百万份文档的规模有时足以证明其使用的合理性。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command A Reasoning 发布**：Cohere 推出了专为企业设计的 **Command A Reasoning**，在 Agent 和多语言基准测试中表现优于其他模型；可通过 [Cohere Platform](https://dashboard.cohere.com/playground/chat?model=command-a-reasoning-08-2025) 和 [Hugging Face](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025) 获取。
   - 根据 [Cohere blog](https://cohere.com/blog/command-a-reasoning)，它可以在单张 **H100** 或 **A100** 上运行，上下文长度为 **128k**，在多 GPU 上可扩展至 **256k**。
- **Command 的 Token 预算功能解决难题**：**Command A Reasoning** 具备 **token budget**（Token 预算）设置，能够直接管理计算资源使用并控制成本，从而无需区分推理模型和非推理模型。
   - 它也是驱动 **North** 的核心生成模型，**North** 是 Cohere 的安全 Agentic AI 平台，支持自定义 AI Agent 和本地自动化。
- **Command-a-03-2025 引用功能不稳定**：`command-a-03-2025` 仅间歇性地返回引用，即使将 maxTokens 设置为 8K 也是如此，这在生产环境中引发了信任问题。
   - 一位 Cohere 成员澄清说，它在引用时使用的是 *"fast"* 模式（根据 [API reference](https://docs.cohere.com/reference/chat#request.body.citation_options.mode)），且不保证一定提供引用；建议改用 **command-a-reasoning**。
- **Langchain RAG 开发中**：一位成员正在学习 Langchain 以构建 RAG（Retrieval-Augmented Generation）应用，并打算使用 **command-a-reasoning**。
   - 他们期待 **command-a-omni** 的发布，并对未来名为 **Command Raz** 的模型表示期待。



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 客户端无视指令字段**：成员们报告称 **MCP 客户端**（特别是 **Claude**）忽略了 **instructions field**（指令字段），仅考虑 **tool descriptions**（工具描述）。
   - 一位成员建议 *添加指令、上下文然后重复指令会产生更好的效果*，但这在集成 API 中无法实现；另一位成员则建议 **MCP server** 应优先处理 **tool descriptions**。
- **多样化的 MCP Server 实践**：成员们分享了他们首选的 **MCP server** 配置和工具，包括用于版本控制的 GitHub、用于后端开发的 Python 和 FastAPI，以及用于机器学习的 PyTorch。
   - 一位用户就如何让 Agent 遵循特定的 **generate_test_prompt.md** 文件寻求建议，并链接了其配置的 [截图](https://cdn.discordapp.com/attachments/1312302100125843476/1408171379236409354/Screenshot_3.png?ex=68a8c54b&is=68a773cb&hm=1fbd862963889b97ef764b1599c2960340dab7ce357b3717f577e2b1491ffdc2)。
- **Web-curl 释放 LLM Agent 威力**：**Web-curl** 是一个使用 Node.js 和 TypeScript 构建的开源 **MCP server**，它使 LLM Agent 能够获取、探索并与网页及 API 交互，源代码可在 [GitHub](https://github.com/rayss868/MCP-Web-Curl) 获取。
   - 在功能上，**Web-curl** 允许 LLM Agent 以结构化的方式获取、探索并与网页及 API 交互。
- **MCP-Boss 集中化密钥管理**：一位成员介绍了 **MCP Boss**，用于集中管理密钥，为所有服务提供统一的网关 URL，并支持多用户身份验证以及通过 OAuth2.1 或静态 HTTP 标头进行的 MCP 授权。
   - 更多信息请访问 [mcp-boss.com](https://mcp-boss.com/)。
- **AI 路由功能赋能 MCP Gateway**：一位成员介绍了一个带有 **AI 驱动路由** 功能的轻量级网关，旨在解决 Agent 需要知道哪个特定服务器拥有正确工具的问题，代码可在 [GitHub](https://github.com/oliverye7/mcp-gateway) 获取。
   - 通过使用该网关，可以利用 AI 来解决 **MCP routing** 问题。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 庆祝 Modverse 里程碑**：Modular 发布了 [Modverse #50](https://www.modular.com/blog/modverse-50) 并宣布了自定义服务器标签，如 [Screenshot_2025-08-21_at_5.22.15_PM.png](https://cdn.discordapp.com/attachments/1098713601386233997/1408199603861323878/Screenshot_2025-08-21_at_5.22.15_PM.png?ex=68a8df94&is=68a78e14&hm=2991584cc0b81449dbc278d1d8302e55aabc54c58a6620e7041deb9dbd20e951&) 所示。
   - 自定义服务器标签已部署。
- **文档匮乏困扰 kgen 和 pop**：成员反映 **kgen** 和 **pop** 缺乏文档，特别是关于操作和参数的部分，有人指出*目前还没有关于内部 MLIR dialects 的全面文档*。
   - 共享了 GitHub 上 [pop_dialect.md](https://github.com/modular/modular/blob/main/mojo/stdlib/docs/internal/pop_dialect.md) 的链接，并澄清这些是 stdlib 与编译器之间协议的一部分，*因此在 stdlib 之外使用它们需自担风险*。
- **POP Union 面临对齐问题质疑**：由于在使用 `sizeof` 时出现意外的大小差异，人们怀疑 **pop.union** 存在对齐（alignment）Bug。
   - 一名成员在 GitHub 上创建了 [issue 5202](https://github.com/modular/modular/issues/5202) 以调查 **pop.union** 中疑似的对齐 Bug，同时观察到 **pop.union** 似乎没有在任何地方被使用。
- **TextGenerationPipeline 的 Execute 方法被发现**：一名成员找到了 `TextGenerationPipeline` 上的 `execute` 方法，并链接到了 [Modular 仓库中的相关代码行](https://github.com/modular/modular/blob/fe7ba5d5f8b08ccba7356c45b90afcf495421469/max/pipelines/lib/pipeline.py#L977)。
   - 他们建议检查 MAX 版本。
- **内存分配器备受关注**：有成员建议在将内存分配器集成到语言之前，可能需要健壮的分配器支持，因为大多数用户不想手动处理内存溢出（**OOM**）错误。
   - 这些评论是在讨论其他困难时提出的，其中一名成员报告在创建自定义推理循环时，难以在获取下一个 Token 的同时检索 **logits**，并链接了一份 [Google Docs 文档](https://docs.google.com/document/d/1Hd6xZnf0bmg9SMQU1h10cd4Cwd9HDOPzPHZNqiPnrpg/edit?tab=t.0) 以提供背景信息。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 首次展示企业级文档 AI**：LlamaIndex 产品副总裁将于 **9 月 30 日** **PST 时间上午 9 点** 预告关于 [文档](https://t.co/x70xjEQaFs) 解析、提取和索引的企业级经验。
   - 重点在于 LlamaIndex 如何解决现实世界中的文档挑战。
- **vibe-llama CLI 工具配置 Coding Agents**：LlamaIndex 推出了 **vibe-llama**，这是一个 CLI 工具，可为 **LlamaIndex framework** 和 **LlamaCloud** 自动配置带有上下文和最佳实践的 Coding Agents，详情见 [此处](https://t.co/G1gINq9kge)。
   - 目标是简化开发工作流。
- **CrossEncoder 类：Core vs Integrations**：一名成员询问了 `llama-index` 中重复的 **CrossEncoder class** 实现，具体位于 `.core` 和 `.integrations` 下（[代码链接](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/sbert_rerank.py)）。
   - 官方澄清 `.core` 版本是 v0.10.x 迁移的遗留物，建议通过 `pip install llama-index-postprocessor-sbert-rerank` 使用 `llama_index.postprocessor.sbert_rerank`。
- **寻求 Agent 创建网关**：一名成员正在寻找现有的 **gateway** 项目，该项目能将 **model, memory, and tools** 结合在一起，并暴露一个 **OpenAI 兼容端点**。
   - 他们希望在 Agent 探索中避免重复造轮子。
- **AI 安全调查收集社区意见**：一名成员分享了一份 [AI 安全调查](https://mukullight.pythonanywhere.com/form)，以收集社区对重要 **AI 安全问题** 的看法。
   - 该调查旨在了解 **AI 安全社区** 最感兴趣的内容。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **用户报告缺少积分购买选项**：成员们报告称购买额外积分的选项消失了，用户只能看到 *upgrade package*（升级包）选项。
   - 已确认该选项目前处于 *down right now*（下线状态）。
- **支持工单无人回复**：一位用户报告了一个任务问题并创建了工单 **#1318**，但尚未收到回复或获得查看工单的权限。
   - 他们请求团队协助，并标记了一名特定成员。
- **比赛获胜者引发操纵指控**：一位用户指称比赛的第二名获得者 *不配获胜*，并声称比赛 *看起来像被操纵了*。
   - 目前没有提供进一步的证据或细节来支持这一说法。
- **每日免费积分已停止？**：一位回归用户注意到他们没有收到往常的 **每日 300 免费积分**。
   - 他们询问 Manus 是否已停止提供这些积分。
- **推荐积分代码困惑**：一位用户询问如何领取推荐积分，并指出系统要求输入代码。
   - 该用户表示不知道在哪里可以找到所需的代码。

---

## [tinygrad (George Hotz) Discord](https://discord.com/channels/1068976834382925865) Discord

- **探索 Overworld 常量折叠 (Const Folding)**：一位成员探索了 **overworld const folding** 和潜在的 **view(const) refactor**，在[此 Discord 线程](https://discord.com/channels/1068976834382925865/1255400554012741683/1407782654958506004)中重新定义了 `UPat.cvar` 和 `UPat.const_like` 以匹配 `CONST` 和 `VIEW(CONST)`。
   - 目标是折叠像 `x * 0` 这样的表达式，然而，有人对符号计算中有效性和 `.base` 扩散表示担忧。
- **ALU View Pushing 作为替代方案**：建议了一种替代方法，包括在 kernelize 中添加一个 upat，直接将 view 推送到 **ALU** 上，效仿 **S-Lykles's method**。
   - 鉴于 `* 0` 在计算上的无关性，这种方法和针对 `x * 0` 的特殊规则将允许未经修改的符号匹配。
- **提倡移除 base**：一位成员强烈建议不要采用提议的方法，认为它 *“非常丑陋”*，并主张 **移除 `.base`**。
   - 讨论还质疑了在此背景下如何处理 **PAD** 操作。
- **RANGEIFY=1 简化实现**：有人建议设置 **RANGEIFY=1** 可以带来更整洁的实现。
   - 然而，该项目目前正处于旧引擎和 rangeify 共存的过渡阶段，处于一种悬而未决的状态。

---

## [Nomic.ai (GPT4All) Discord](https://discord.com/channels/1076964370942267462) Discord

- **GPT4ALL 免费层级支持私有 AI**：一位用户询问是否可以为希望 **私密且安全地使用 AI 模型** 的公司使用 **GPT4ALL**。
   - 另一位成员澄清说，如果公司已经拥有自己的 **AI 模型**，那么 **免费版本** 就足够了。
- **用户寻求 LocalDocs 模型**：一位用户正在寻求模型推荐，以便利用 **GPT4All 的 LocalDocs 功能**，从数百篇 **PDF 格式的科学论文** 中构建个人知识库。
   - 该用户说明他们拥有 **Nvidia RTX 5090**，配备 **24 GB VRAM** 和 **64 GB RAM**，并希望所选模型具备 **reasoning capabilities**（推理能力）。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

**Torchtune Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：按频道分类的详细摘要和链接

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1407801395884720330)** (951 条消息🔥🔥🔥): 

> `nano-banana 模型, Video Arena 问题, DeepSeek V3.1, Gemini 3` 


- **Nano-Banana 的 McLau's Law 揭晓**：一位成员开玩笑说 **Nano-Banana** 的结果往往 *远低于 nano-banana* 水准，并将这一现象幽默地称为“**McLau's Law**”，以此致敬 **OpenAI** 的一位研究员。
   - 附带了一张[幽默图片](https://cdn.discordapp.com/attachments/1340554757827461211/1407957527987097642/RDT_20250821_0619535468074318472918433.jpg?ex=68a8a6e1&is=68a75561&hm=bfcfd37e574ea905e84f2ff1c9e6ffee1855681ad903050cfe30a367ea5d96f3&)，引发了关于 **AI** 当前能力的讨论。
- **Video Arena 饱受 Bot 宕机困扰**：成员们报告了 **Video Arena** 的问题，称无法使用命令或生成视频，管理员确认了 **Bot** 的宕机情况并正在进行修复。
   - 针对视频创建权限的重复询问，得到的解释是 **Bot** 暂时不可用，并引导用户关注公告频道以获取更新。
- **DeepSeek V3.1 进入竞技场**：用户讨论了该平台引入 **DeepSeek V3.1** 的情况，一位用户将其描述为 *Gemini 2.5 pro 的略逊版本*。
   - 然而，共识是它作为编程模型具有潜力，但需要进一步提升通用能力。
- **用户声称 Gemini 3 即将到来**：虽然未经证实，但一位用户暗示 **Gemini 3** 即将发布，推测发布日期将与 **Google Pixel event** 同步，引发了成员们的期待。
   - 该用户未引用任何来源，此说法很快被其他社区成员否定。 
- **网站故障导致聊天记录清空**：用户报告在网站故障后出现大规模数据丢失，包括聊天记录缺失和无法接受服务条款，管理员对此表示知晓并保证会进行修复。
   - 管理员还表示，很快将推出登录功能，以防止此类情况再次发生。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1408069950391980122)** (2 条消息): 

> `Video Arena Bot, Deepseek v3.1, LMArena 模型` 


- **Video Arena Bot 宕机，频道已锁定**：**Video Arena Bot** 目前无法工作，锁定了对 prompt 频道 <#1397655695150682194>、<#1400148557427904664> 和 <#1400148597768720384> 的访问权限。
   - **Bot** 必须在线才能在这些特定频道中输入 prompt。
- **DeepSeek v3.1 已添加到 LMArena**：两个新模型已添加到 LMArena：**deepseek-v3.1** 和 **deepseek-v3.1-thinking**。
   - 这些模型现在可以在竞技场中使用。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1407802780516614178)** (887 条消息🔥🔥🔥): 

> `字节跳动 Seed 模型, GRPO 训练, DeepSeek V3.1 量化, Nvidia GPU 与定价, GLM-4.5 Cline 集成` 


- **字节跳动发布 Seed-OSS 36B 基础模型**：字节跳动在 Hugging Face 上发布了 **Seed-OSS-36B-Base-woSyn** 模型，这是一个 **36B** 参数的稠密模型，具有 **512K** 上下文窗口，并明确声称*无合成指令数据*，这使其成为进一步微调的理想基础模型。
   - 成员们表示非常兴奋，指出它与 **Qwen3** 等模型不同，一些人渴望在数据集完成后尝试用它来微调 GPT-ASS，尽管该模型“仅”在 **12T tokens** 上进行了训练。
- **GRPO 训练需要巧妙的数据集设计**：为了将 GRPO 用于多步游戏动作，成员建议设计数据集时为每一步设置单独的提示词，例如 **[['step1 instruct'], ['step1 instruct', 'step1 output', 'step2 instruct']]**，并实现一个奖励函数来匹配输出。
   - 有人指出 Full PPO 可能更适合游戏，因为 GRPO 对 LLM 主要有效的原因是*它们起初就大致知道该怎么做*。
- **DeepSeek V3.1 在思考和非思考模式下横扫排行榜**：**DeepSeek V3.1** 模型表现出了极具竞争力的结果，在非思考模式下的 SWE-bench verified 测试中达到了 **66** 分，成员们对此表示期待，并将其与 **GPT5** 的中等推理能力进行比较。
   - 尽管最初备受推崇，但随后的讨论提到了对其在创意写作和角色扮演中表现的担忧，一些人指出*混合模型在非思考模式下缺乏指令遵循能力和创造力*。
- **Nvidia RTX 5090 价格尘埃落定，引发升级争论**：**RTX 5090** 目前定价在 **$2000** 左右，引发了是否升级的讨论，特别是考虑到其 **VRAM** 容量对训练的价值，而其他人则建议坚持使用 **3090s** 或等待 **RTX 6000**。
   - 一些成员对 **NVIDIA** 的限制表示沮丧，特别是缺乏 **P2P 或 NVLink**，一位成员开玩笑说：*如果你坐在一台 5090 上，你肯定会用它玩游戏*。
- **高质量 Imatrix 校准数据是关键**：成员们指出 WikiText-raw 被认为是校准 imatrix 的*糟糕*数据集，因为 imatrix 需要充分多样化，并在模型原生的 chat-template 格式示例上进行训练。
   - 相反，[Ed Addorio 最新的校准数据](https://huggingface.co/datasets/eaddario/imatrix-calibration)包含数学、代码和语言提示词，如果操作得当，可以改善并帮助保留模型对多种语言的理解。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/)** (1 条消息): 

.zackmorris: Hello
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1407836226488111114)** (27 条消息🔥): 

> `GRPO 20mb 分配失败, ChatGPT 深度研究, Grok-4, 重复惩罚, RAG` 


- ****GRPO 20MB 分配失败困扰 Gemma 模型！****：一位用户报告在处理 [gemma-3-4b-it-unslop-GRPO-v3](https://huggingface.co/electroglyph/gemma-3-4b-it-unslop-GRPO-v3) 时，使用 **GRPO** 频繁出现 **20MB 分配失败**。
- ****ChatGPT 的深度思考模式提升性能！****：一位用户建议通过启用网页搜索并在提示词中添加 *“尽可能使用深度思考 (deep thought)”* 来增强 **ChatGPT** 的性能，即使没有完整的深度研究功能。
- ****Grok-4 表现出色！****：一位用户对 **Grok-4** 印象深刻，暗示他们可能一直在秘密使用 **Grok-4-Heavy**。
- ****重复惩罚引发的趣事****：一位用户分享了一张图片来展示 **repetition penalty**（重复惩罚）参数的重要性。
- ****RAG 协助****：一位用户请求在处理 **RAG** 时获得帮助。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1407822574725107743)** (101 messages🔥🔥): 

> `视网膜照片训练策略, GPT-OSS 20B 在 Sagemaker 上的部署, Unsloth Zoo 问题, 使用 Unsloth 加载 GGUF, Gemma 3 Vision Encoder 训练损失` 


- **为视网膜照片微调 Vision-Text Encoder**：一位用户询问是应该为视网膜照片训练自定义的 Vision-Text Encoder，还是使用 Unsloth 配合主流模型，并指出**视网膜照片在训练数据集中代表性不足**。
   - 建议尝试计算机视觉模型、在相似数据集上进行迁移学习以及多模态方法，并利用 Prompt Engineering 和 Personas 生成合成临床笔记。
- **排除 GPT-OSS 20B Sagemaker 部署故障**：一位用户在 Sagemaker 上部署 **unsloth/gpt-oss-20b-unsloth-bnb-4bit** 时遇到 `ModelError`，收到 **400 错误**和带有 `\u0027gpt_oss\u0027` 消息的 InternalServerException。
   - 有回复指出该模型无法在 AWS Sagemaker 上运行，建议部署 GGUF 或普通版本，使用 LMI 容器并参考 [AWS 文档](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-container-docs.html)。
- **Unsloth Zoo 安装问题**：一位用户在 Sagemaker 实例中安装 **unsloth-zoo** 后仍遇到导入错误。
   - 用户通过删除所有包，然后重新安装 Unsloth、Unsloth Zoo 以及 JupyterLab 解决了该问题，同时还需要更新 Unsloth 并刷新 Notebook。
- **Apple Silicon Mac 的量化考量**：一位用户寻求关于 M 系列 Apple Silicon 最适合哪种 **GGUF 量化**的指导，并指出 Mac 针对 **4-bit** 和 **8-bit** 计算进行了优化。
   - 建议用户选择 **Q3_K_XL**，如果 Context 无法容纳在内存中则选择 **IQ3_XXS**；Q3-4 量化性能较好，但如果使用 GGUF，差异则没那么大。
- **GPT-OSS 通过 LLaVA 获得多模态能力**：一位用户询问为什么 vision llama13b notebook 无法用于 gpt-oss-20b，并想知道是否有人成功实现过。
   - 解释称 GPT-OSS 是纯文本模型而非视觉模型，因此无法直接运行；若要添加视觉支持，用户必须像 LLaVA 那样附加自己的 **ViT 模块**，可参考 [LLaVA 指南](https://github.com/haotian-liu/LLaVA)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1407927838123888651)** (11 messages🔥): 

> `WildChat-4M-English-Semantic-Deduplicated 数据集, Behemoth-R1-123B-v2 模型, GPU Rich 炫耀` 


- **WildChat-4M 英文 Prompt 数据集发布**：**WildChat-4M-English-Semantic-Deduplicated 数据集**已在 Hugging Face 上线，包含来自 WildChat-4M 数据集的英文 Prompt，并使用了包括 **Qwen-4B-Embedding** 和 **HNSW** 语义去重在内的多种方法进行去重。
   - 当前版本包含 **<= ~2000 Token** 的 Prompt，后续将添加更长的 Prompt，更多信息请见[此处](https://huggingface.co/datasets/MasonMac/WildChat-4M-English-Semantic-Deduplicated)。
- **TheDrummer 发布 Behemoth-R1-123B-v2**：由 TheDrummer 创建的 **Behemoth-R1-123B-v2** 模型已发布，可以在[此处](https://huggingface.co/TheDrummer/Behemoth-R1-123B-v2)找到。
   - 一位成员提到，能在 Hugging Face 中配置自己的硬件真是太疯狂了。
- **GPU Rich 是新的炫耀方式**：一位成员分享了一张图片，描绘了对贫穷的嘲讽，但炫耀了 **GPU Rich**（GPU 富有）。
   - 看到以 **TFLOPS** 为单位的 GPU 性能是一种炫耀。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1407840310024995026)** (7 条消息): 

> `Qwen3-4B finetuning, TTS with Gemini 270m, Mixture Models, JetMoE, BAM` 


- ****Unsloth** + **Qwen3-4B**：强强联手？**：一位成员正在使用 **Unsloth** 对 **Qwen3-4B** 进行 finetuning，并将在完成后分享包括评估在内的结果；目前微调进展顺利。
   - 另一位成员祝其好运！
- **从零开始训练模型**：一位成员正在从零开始训练一个概念验证模型，进度已达 **22%**。该模型使用自建的 6 年级数学数据集，包含 **500k** 样本数据。
   - 如果成功，他们将把数据集扩展到其他学科。
- **使用 Gemini 270M 实现 TTS 的构想**：一位成员想尝试使用 **Gemini 270m** 实现 **TTS** 概念，并希望在月底前开始。
   - 他们的灵感来自关于 mixture model 的论文。
- **专家讨论合并模型在 HumanEval 上的弱点**：一位成员引用了关于从零训练的 mixture models 的 [JetMoE 论文](https://arxiv.org/pdf/2404.07413#page=9.56)，指出尽管它们在其他方面的表现优于基准模型，但在 **HumanEval** 上的表现较差。
   - 他们还提到了 [BAM](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2408.08274)，其中预训练模型被复制并在不同领域进行训练后合并，但在编程方面也损失了百分点。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1408170025436844156)** (1 条消息): 

> `Cloudflare outage, Generations API stability` 


- **Generations API 受 Cloudflare 波动影响**：由于上游基础设施提供商的问题，**Generations API endpoint** 经历了暂时性中断，导致部分调用出现 **404 错误**。
   - 公告指出，该问题与 **Cloudflare** 的间歇性故障有关，但 **Generations API** 现已恢复到健康状态。
- **可重试的恢复**：对该终端的调用可能会出现 **404**，但应该很快就能 **重试**。
   - 公告向用户保证服务将很快恢复，并建议他们重试任何失败的调用。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1408135423468765276)** (4 条消息): 

> `OpenRouter Cost Dashboard, Average Request Size, Gemini Input Token Calculation` 


- ****费用报告实现可视化！****：一位成员开发了一个免费的仪表盘，用于可视化来自 [OpenRouter](https://openrouter.ai/) 的 `.csv` 费用报告，旨在分析共享账户的数据。
   - 该仪表盘可在 [openroutercosts.lorenzozane.com](https://openroutercosts.lorenzozane.com/) 访问，计划加入更多 **KPI** 和增强型图表，欢迎反馈。
- ****仪表盘请求添加平均请求大小指标！****：一位成员请求在 OpenRouter 费用仪表盘中增加 **average request size** 指标，特别是 **average input tokens** 和 **average output tokens**。
   - 仪表盘开发者承诺很快会添加此功能。
- ****Gemini Input Tokens 触发异常计数！****：仪表盘开发者注意到，当输入中包含图像时，**OpenRouter** 对 **Gemini 模型** 的 **input tokens** 计算似乎产生了异常数值。
   - 他们正在考虑就此问题寻求 OpenRouter 团队的澄清，并参考了 [Google AI Developers 论坛](https://discuss.ai.google.dev/t/token-counts-mismatch-9x-discrepancy/72633/2)上的相关讨论。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1407830899223036106)** (528 条消息🔥🔥🔥): 

> `Deepseek pricing, OpenRouter rate limits, Gemini banning, Using OpenRouter with RAG systems, 4.6T parameter model` 


- **Deepseek V3.1 公开发布在即！**：许多用户正急切等待 **Deepseek v3.1** 的公开发布，对其极度渴求，并预计它将从 9 月开始免费。
- **付费版 Deepseek 提供更快的响应**：用户确认在 OpenRouter 上为 **Deepseek** 模型付费比使用免费模型响应更快。由于 **Chutes** 导致响应变慢，一名用户选择了切换，且由于不断的速率限制 (rate limits)，免费模型上的用户体验并不理想。
   - 一位用户表示：*自从 Chutes 导致响应变慢后，我就直接决定付费了*。
- **OpenRouter API 密钥易受泄露和利用**：一名用户报告因 OpenRouter API 密钥泄露损失了 **$300**，并寻求关于识别未经授权使用来源的建议。但攻击者可能会使用代理来掩盖其原始 IP，用户需对任何泄露的密钥负责。
- **Gemini 正在进行封号大清洗吗？**：用户报告 **Gemini** 正在发生大规模封号，导致许多人寻找替代方案，并回想起由 OpenAI 引起的 AI Dungeon 清洗事件。
   - 一位用户哀叹道：*我们正被送回 2023 年*。
- **OpenRouter API 密钥可以用于 RAG 吗？**：用户讨论了在 **RAG 系统** 中使用 **OpenRouter LLM API 密钥** 的可能性，配合由 Milvus 创建的本地存储向量数据库。
   - 共识是可行的，但 OpenRouter 并不直接支持 embeddings，因此你必须使用 Milvus 检索文档，并将其与你的提示词问题一起发送给 OpenRouter LLM API。


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1407869061840506900)** (3 条消息): 

> `` 


- **Readybot.io 宣布 OpenRouter 新模型**：Readybot.io 发布了关于 **OpenRouter** 平台上可用**新模型**的更新和信息。
- **OpenRouter 新模型更新**：**OpenRouter** 平台重点介绍了其 **AI 模型** 选择的最新增加和变化，如 Readybot.io 所宣布。


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1407806939878129774)** (16 条消息🔥): 

> `Qwen3 coder 480b, DeepSeek v3 0324, Zero return from generative AI, Google Gemini 400 Error, Cohere reasoning model` 


- **LLM 难以正确格式化输出**：用户发现 [像 **Qwen3 coder 480b** 和 **DeepSeek v3 0324** 这样的 LLM](https://link.to.example) 难以遵循正确格式化输出的指令，经常导致 Bug 和提示词被忽略。
   - 一位用户发现它们*没用*且*相当让人分心*，经常创建井字游戏网站而不是预期的应用程序。
- **大多数组织在生成式 AI 上看到零回报**：根据 [AFR Chanticleer 报告](https://archive.md/IlP7F)，**95% 的组织在部署生成式 AI 时没有获得任何回报**。
   - 报告指出，这主要集中在部署了**定制化 AI 模型**的公司，关键问题在于公司及其技术供应商没有投入足够的时间来确保其定制化 AI 模型能够持续学习业务的细微差别。
- **Google Gemini 模型触发 400 错误**：当带有工具调用 (tool calls) 的助手消息使用 **OpenAI 标准的复杂内容格式** `[{"type": "text", "text": "..."}]` 而非简单的字符串格式时，**Google Gemini** 模型会返回 **HTTP 400 错误**。
   - 此问题影响所有 `google/gemini-*` 模型，且仅在消息链中存在工具调用和工具结果时发生。
- **Cohere 发布推理模型**：[Cohere 刚刚发布了一个推理模型](https://cohere.com/blog/command-a-reasoning)，更多细节可在 [Discord](https://discord.com/channels/954421988141711382/996880279224451154/1408103056800874497) 上查看。
   - 目前没有更多细节。
- **功能请求：自动折叠冗长的用户消息**：一位用户请求是否可以在聊天室中自动折叠冗长的用户消息。
   - 该用户对聊天室和聊天管理表示了赞赏。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1407803160356982795)** (432 条消息🔥🔥🔥): 

> `Claude Cache Reads, Sonic Model origin, Open Sourcing Agentwise, Cursor API costs with Auto agent, DeepSeek V3.1` 


- **缓存问题困扰 Claude**：用户报告称 **Claude** 目前在*缓存读取（cache reads）*方面存在故障，导致其成本高于具有可持续缓存机制的 **Auto**。
   - 一位用户猜测 **Auto** 和 **Claude** 是否秘密地使用了相同的模型，并将 Token 使用量的减少归因于安慰剂效应。
- **Sonic 进驻 Cursor IDE**：社区正在测试 Cursor 中新的 **Sonic** 模型，一位用户报告称它*非常不错*且速度极快，而另一位用户则认为它适用于新项目，但不适用于具有大型代码库的项目。
   - 该模型的来源是一家*隐身公司（stealth company）*，一位成员确认 **Sonic 并非 Grok 模型**。
- **Agentwise 宣布开源**：一位成员宣布 **Agentwise** 开源，该工具支持网站副本、图片/文档上传，并支持超过 100 个 Agent，并承诺将提供 [Cursor CLI 支持](https://discord.com/channels/1074847526655643750/1408047562019049523)。
   - 鼓励成员在项目的 Discord 频道中提供反馈。
- **Cursor API 成本说明**：用户对 Auto agent 成本的困惑得到了澄清，确认在拥有 "Pro" 订阅的情况下，**没有额外费用**，只有由订阅涵盖的不同提供商的 API 使用成本。
   - 一位用户发现 Auto agent 比 Sonic agent 更好用。
- **DeepSeek V3.1 进入竞技场**：用户注意到 Cursor 选项中出现了新的 **DeepSeek V3.1** 模型，但部分用户在连接提供商时遇到困难，其中一人表示*他们不信任中国的 LLM*。
   - 然而，一位成员报告称 DeepSeek V3.1 在 **TypeScript** 和 **JavaScript** 上表现良好，甚至表现*出色*，且价格比 Sonnet 更便宜。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1407802650908688424)** (11 条消息🔥): 

> `Agent Auditing, MySQL Installation in Background Agents, Background Task Errors, Remote IDE connection to Background Agent` 


- **Agent 自我审计修复问题**：一位用户报告称，通过要求 Agent 执行 commit 和 push 新分支的操作修复了一个问题，并指出这似乎是一个内部反复出现的问题。
   - 另一位用户确认这是一种审计行为，并解释说这是 Agent 使用 **AI-GPL 授权的审计 PDCA 流程框架**进行自我审计。
- **Agent 中的 MySQL 配置说明**：一位用户询问在 Background Agents 中安装 **MySQL** 的事宜，质疑它是预装的还是像 Codex 一样仅限于 **SQLite**。
   - 另一位用户澄清说 **MySQL** 默认未安装，但可以通过 `environment.json` 或 **Dockerfile** 添加到 Agent 的环境中。
- **Background Task 错误排查**：一位用户报告称，在启动 Background Task 后立即持续报错（即使是从 Web 端启动），并提供了一张[截图](https://cdn.discordapp.com/attachments/1367213641027551352/1408202779096383550/Screenshot_2025-08-21_at_4.34.24_PM.png?ex=68a8e289&is=68a79109&hm=313d4bdb3a6bb89b6beeb5e9ffb22927afd3259ca9dc351a930226cbb122227c&)。
- **远程 IDE 连接存在困惑**：一位用户寻求关于将**远程 IDE** 实例连接到远程机器的明确说明，虽然参考了文档但发现指令不清晰。
   - 他们询问是否需要一个虚拟的 Background Agent 来辅助建立此连接。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1407801641675260104)** (141 条消息🔥🔥): 

> `4070 TI Super 的 CUDA 错误，LM Studio 多 GPU 性能，SerpAPI 与 LM Studio 的集成，GPT-OSS 性能，显存（VRAM）使用的模型参数配置` 


- **修复 4070 识别问题需要 CUDA 驱动**：一位使用 **4070 TI Super** 的用户报告在 LM Studio 中出现 *"0 GPUs detected with CUDA"* 错误，另一位用户建议通过按下 **ctrl+shift+r** 将运行时更改为 **CUDA llama.cpp** 来尝试解决该问题。
- **Flash Attention 加上 KV 量化可显著降低显存（VRAM）**：一位成员建议使用命令 `-fa -ub 2048 -ctv q8_0 -ctk q8_0` 来启用 **flash attention**、**KV 缓存量化**以及 **2048 的 batch size**。
   - 同时建议增加 `-n-cpu-moe` 的值以管理显存使用，并指出这仅影响速度。
- **GPT-OSS 在 Prompt Eval 上远超 Qwen**：成员们注意到 **GPT-OSS** 在 **3080ti** 上的 prompt eval 达到了 *2k tokens/s*，而 **Qwen** 约为 *1000 tokens/s*。
- **Bolt.new 仅限云端**：一位用户询问如何将 Bolt.new 与 LM Studio 配合使用，但另一位用户澄清说 [Bolt 仅限云端](https://github.com/stackblitz-labs/bolt.diy)，不支持本地模型。
- **LM Studio API 调用慢如蜗牛**：一位用户报告 LM Studio API 调用比聊天界面慢得多（30 倍），后来该问题因不明原因自行解决——此问题可能无法配置。
   - 他们使用了 curl 命令 `curl.exe http://localhost:11434/v1/chat/completions -d {"model":"gpt-oss:20b","messages":[{"role":"system","content":"Why is the sun hot?\n"}]}`


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1407827727985152000)** (54 条消息🔥): 

> `Z390 Designare 对比 Threadripper/Epyc，Qwen3-30B-A3B-Instruct-2507-GGUF 基准测试，Model M 屈伸弹簧键盘，Apple M4 Max 上的 GGUF 对比 MLX，在 Apple M1 上运行 GPT-OSS-20b` 


- **旧款 Z390 Designare 受限于 PCIe 带宽**：在旧款 Z390 Designare 上运行 RTX PRO 6000 可能会因为与 Threadripper 或 Epyc 系统相比有限的 PCIe 带宽而导致**性能轻微下降**。
   - 旧主板限制了 PCIe 带宽，导致了瓶颈。
- **Qwen3-30B 在 CPU 上达到 10 tok/sec！**：一位用户在 **Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf** 上运行了 [llama-bench](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama-bench)，在纯 CPU 配置下获得了约 **10 tokens per second** 的速度。
   - 性能随线程数变化，由于扩展性和开销，超过一定阈值后收益递减。
- **Unicomp Model M 屈伸弹簧键盘：依然出色**：用户建议购买 **Unicomp Model M 屈伸弹簧键盘**用于快速测试机，并指出 Unicomp 已获得生产权。
   - 一位用户提到他们不得不*寻找有库存的英国供应商*。
- **M4 Max 上的 MLX 击败 GGUF**：一位用户在 Apple M4 Max 上对 **GPT-OSS-20b** 进行了基准测试，发现 **MLX (GPU)** 在 **32W (2.39 t/W)** 功率下达到了 **76.6 t/s**，而 **GGUF (CPU)** 在 **43W (0.61 t/W)** 功率下为 **26.2 t/s**。
   - 测试使用了 **4bit 量化**和 **4k 上下文**，结果显示 MLX 比 GGUF 稍快且能效更高，用户对 GGUF 的性能印象深刻。
- **GPT-OSS-20b 勉强适配 Apple M1**：用户讨论了在 16GB 内存的 Apple M1 上运行 **GPT-OSS-20b** 的挑战，指出它需要大约 **32GB 的 RAM**。
   - 一位用户建议尝试 [Hugging Face 上的 4-bit MLX 版本](https://huggingface.co/InferenceIllusionist/gpt-oss-20b-MLX-4bit)，并指出*它勉强能装下*。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1407807741900230718)** (167 条消息🔥🔥): 

> `Machine-to-Machine Economies, AI safeguards, Decentralized AI projects, Few-shot examples for Large Prompts, GPT-5's Direct Responses` 


- **机器人接入 M2M 经济**：成员们讨论了 AI Agent 或机器人如何自主交换价值或服务，触及了 **machine-to-machine (M2M) 经济**的概念。
   - 最困难的部分在于*机器人之间的身份与信任、智能合约逻辑、支付基础设施、自主性与安全性，以及法律和伦理挑战。*
- **智能安全保障可加速 AI 普及**：成员们讨论了如**支出上限、审计日志和保险**等安全保障措施，这些措施可能会加速能够进行价值交易的 AI Agent 的普及。
   - 然而，普遍观点是，尽管有安全保障，*真正的信任建立仍需时日。*
- **征集开源去中心化 AI 项目**：一位成员询问为什么还没有建立 **BOINC 风格的去中心化 AI 项目**，并提到 [Petals network](https://petals.ml/) 在贡献和保持模型更新方面存在问题。
   - 有建议认为，**经济激励**或**活动驱动的激励**可能会有所帮助。
- **深入探讨长 Prompt 的 Few-shot 示例**：一位成员询问了在为逻辑复杂的健身工作室编写的 **29,000 token Prompt** 中使用 **few-shot 示例**的最佳实践。
   - 建议包括直接在 Prompt 中提供示例，并将 Prompt 拆分为更小的块，以测试单个组件的性能。
- **GPT-5 的直接回答引发挫败感**：一位用户抱怨 **GPT-5** 的“思考”模式给出的回答非常直接且**质量极低**，仿佛回退到了旧的模型版本。
   - 另一位成员建议，该用户可能达到了*思考配额限制，并且设置了回退（fallback）而不是置灰？*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1407853430252376064)** (9 条消息🔥): 

> `GPT-4 projects UI files, AI court legal case, Android app development with GPT, Token usage for uploaded content, GPT server issues` 


- **GPT Projects UI 文件上传**：一位用户正在寻求关于上传到 **Projects UI** 的文件如何工作的确切信息，并指出 **ChatGPT** 告知他们*目前 Project Files 中的 PDF 不对搜索或检索开放*。
   - Bot 指出唯一活跃的连接器是用于会议记录的 **recording_knowledge**，且不支持 **source_filter**。
- **GPT 扮演法庭：AI 法律专家屹立不倒**：一位用户模拟了一个 **AI 法庭法律案件**，发现 **GPT-5** 坚持自己的立场，而不是接受基于现实世界 TRAIGA 法律的法律规则。
   - 在面对“每周 9 亿用户不可能都在幻觉，称你为退化而非真正的更新”这一指责后，该 AI 表示接受*保持现状会更好*。
- **Token 使用成本曝光**：一位用户发现，即使是上传的内容（如 **PDF 页面**）也会计入 Token 使用量。
   - 他们指出，*196k Token 大约相当于 300 页 PDF 的用户上下文*，并强调在考虑上下文时，提问和 GPT 的回复都会消耗 Token。
- **Android 应用末日：GPT 的 APK 梦想破灭**：一位用户在尝试将 **Canvas** 应用转换为 Android 就绪版本时遇到了困难，询问 **GPT** 是否能构建 **Android 应用**并使用 **Android Studio** 生成 **APK**。
   - 修复一个问题后又会出现另一个问题，得出的结论是*它还没准备好进行应用开发*，尽管 Bot 在一天后建议将 PWA 或 JSX 文件封装在 APK 壳中。
- **GPT 服务器在追踪中途崩溃**：一位用户在追踪每日数据时遇到了**服务器问题**，该问题从前一天晚上开始出现。
   - 其他人评论说，这些工具让编码变得*更容易*，但不会为你完成所有工作。你必须具备一定程度的编程知识。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1407886141813821440)** (6 条消息): 

> `AI Quiz generation, GPT models quitting` 


- **AI 测验生成明显的错误答案**：一位成员尝试使用 AI 生成测验，但面临 AI 提供的错误选项*过于明显*的问题。
   - 另一位成员建议确保*所有选项必须具有合理性*。
- **LLM 可能会随机退出**：一位成员询问如何防止 **GPT 模型**在推理一段时间后随机退出。
   - 另一位成员回答说，减少难以处理的查询以及关于其自身推理的查询会有所帮助，但归根结底 **LLM** 是**随机性的（stochastic）**，没有保证能阻止它们以特定方式响应的方法。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1407886141813821440)** (6 messages): 

> `AI Generated Quizzes, GPT-5 Random Quitting, Plausible Response Options, LLM Stochasticity` 


- **AI 测验生成器使选项变得平庸**：一位成员正面临 AI 测验生成器产生明显错误答案选项的问题，例如在多选题中出现 *1029384*。
   - 另一位成员建议确保 *所有响应选项必须具有合理性 (plausible)* 以避免此类问题。
- **GPT-5 意外退出**：一位用户询问是否有办法防止 **GPT-5** 在推理一段时间后随机退出。
   - 一位成员回答说，虽然有减少频率的方法，例如避免难以处理的查询或关于其自身推理的问题，但由于 **LLM 的随机性 (stochastic nature)**，完全消除这种情况是不可能的。
- **LLM 是随机的，需要 Guardrails**：由于 Large Language Models 的随机性，*实际上无法阻止它们在足够大的样本量中至少一次以任何给定的方式做出响应。*
   - 由于 LLM 的非确定性，Guardrails 是必要的。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1407813276863168583)** (96 messages🔥🔥): 

> `PileT5-XL embeddings as instructions, Networks that process in latent space, Multimodal generative models, image editing models, Latent space editing` 


- **PileT5-XL Embeddings 包含丰富信息**：来自 **PileT5-XL** 的 Embedding Tensor 既可以作为 **pile-t5-xl-flan**（生成文本）的指令，也可以作为 **AuraFlow**（生成图像）的 Prompt，这表明这些 Embedding 像语言中的单词一样具有意义。
   - 一位成员对如何使用黑狗图片的 Textual Inversion 配合 AuraFlow 并应用于 pile-t5-xl-flan 感兴趣，想知道 pile-t5-xl-flan 生成的文本是否会将狗描述为黑色。
- **深入探索 Latent Space**：一位成员有兴趣探索在 Latent Space 中处理、且仅在必要时以模块化方式转换为文本/图像/音频的网络。
   - 有人指出，这个想法与人们构建多模态生成模型和 VQGAN-CLIP 的方式相似，并指出让不同的 AI 研究人员 *同意使用相同的 Latent Space* 是一个挑战。
- **精细编辑图像**：围绕专为图像编辑设计的模型（如 FLUX.kontext）展开了讨论，以及它们是否编辑 Conditioning Latent 并输出相同空间中的新 Conditioning Latent。
   - 一种方法是获取一堆包含鸟的图像，将鸟编辑掉，然后将两者都通过 Encoder 运行，最后平均它们之间的差异以获得 *Latent Space 鸟类* 向量。
- **调整 Transformer 的透镜**：关于 **Tuned Lens** ([https://arxiv.org/abs/2303.08112](https://arxiv.org/abs/2303.08112)) 的工作从 Transformer 中提取了 *模型在第 k 层后的最佳猜测*，这反驳了关于 Decoder Transformer 中 Latent Space 处理的一些假设。
   - 还提到了关于从图像空间到文本空间线性映射 ([https://arxiv.org/abs/2209.15162](https://arxiv.org/abs/2209.15162)) 的进一步研究。
- **解码音频的秘密**：一个备受关注的模型是 Decoder-only 音频模型 ([https://huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M))，它可能为训练开启新的可能性。
   - 据称，预训练期间看到的音频数据量从 1 分钟到 100 小时不等，也许你可以用 0 分钟的音频进行训练？


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1407829390640939050)** (54 messages🔥): 

> `SSL objectives, Medical event pretraining, Noise-data trajectories, ByteDance's Prover, Unfriendly Activation Steering` 


- **SSL objectives 与 Maximal Coding Rate 相关研究**：一名成员将近期关于 **SSL objectives** 的观点与 [maximal coding rate stuff](https://arxiv.org/abs/2005.10242)、[contrastive learning](https://arxiv.org/abs/2406.10743) 以及 [neural collapse](https://arxiv.org/abs/2303.06484) 联系起来。
- **字节跳动的 SEED Prover 获得银牌成绩**：**Bytedance's SEED Prover** 在 [IMO 2025 中获得了银牌分数](https://seed.bytedance.com/en/blog/bytedance-seed-prover-achieves-silver-medal-score-in-imo-2025)，但目前尚不清楚这如何转化为现实世界的数学问题解决能力。
- **生成式医疗事件模型的 Scaling Laws**：**Cosmos Medical Event Transformer (CoMET)** 模型系列是仅含解码器的 Transformer 模型，在代表 **1150 亿个离散医疗事件**（1510 亿个 tokens）的 **1.18 亿名患者** 数据上进行了预训练。研究发现，这些模型在相关任务上的表现通常优于或等同于特定任务的监督模型。
   - 该研究在 [Generative Medical Event Models Improve with Scale](https://arxiv.org/abs/2508.12104) 中进行了讨论，使用了 **Epic Cosmos** 数据集，该数据集包含来自 **310 个医疗系统**、**3 亿条唯一患者记录**中 **163 亿次就诊** 的去标识化纵向健康记录。
- **可视化噪声-数据轨迹 (Noise-Data Trajectories)**：成员们讨论了可视化 Flow Model 中 **noise-data trajectories** 的方法，包括在预计算的中间体上使用 **UMAP**，但发现其信息量不足。
   - 假设存在截然不同的轨迹簇，他们希望有一种方法能将这些轨迹挑选出来并单独观察，并确定完全不同类型的输入或两种不同形式的 conditioning 是否遵循 *相同的* 轨迹。
- **训练期间的不友好激活引导 (Unfriendly Activation Steering)**：一名成员提到在训练期间使用 **unfriendly activation steering** 来影响模型权重的工作，并附上了相关 [tweet](https://fxtwitter.com/Dorialexander/status/1958269223320613241) 的链接。


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1407853408211177494)** (1 messages): 

> `Model Overtraining, Token Repetition in Models` 


- **在 Chinchilla 之后继续过度训练模型！**：即使在 **Chinchilla** scaling laws 之后，你仍然应该 **overtrain 你的模型**。
   - 显然，*甚至重复 tokens 也不是坏事*。
- **Token 重复可能无害**：在训练期间重复 tokens 可能并不像之前认为的那样有害。
   - 持续训练带来的收益似乎超过了 token 重复带来的潜在弊端。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1407804201567912107)** (11 messages🔥): 

> `Qwen3 Training, Weight lifting from llama series, Head isolation` 


- **Qwen3：从零训练还是借鉴了 Llama？**：一名成员询问 **Qwen3** 是从零开始训练的，还是从 **Llama** 系列中提取了权重（weight lifting）。
   - 另一名成员指出，类似的训练数据混合比例可能会导致类似的结果。
- **发现相同的 Head！**：一名成员发现并隔离了一种特定的 *head*，发现 **Llama 3.2-1b instruct** 和 **Qwen3-4B-Instruct-2507** 之间的解码结果向量在不同输出中表现出惊人的相似性。
   - 该成员表示，*这两个 head 似乎促进的内容非常相似*。
- **方法论论文发布**：一名成员链接了 [一篇论文](https://arxiv.org/abs/2502.12292)，详细介绍了确定 **Qwen3** 是否从零开始训练的方法论。
   - 另一名成员称该用户为“简直是降临人间派发礼物的神”。
- **潜意识学习 (Subliminal Learning) 案例**：一名成员分享了 [一篇论文](https://aclanthology.org/2025.acl-long.407.pdf)，将其视为 *潜意识学习的一个明确案例*。
   - 另一名成员对此分享表示感谢。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1407927947200827462)** (2 messages): 

> `Muon Support, Slurm Script for NeoX Job with Docker` 


- **寻求 Muon 支持**：一名成员表示有兴趣添加 **muon 支持**，并提到了潜在的 **kernel 优化机会**。
   - 他们认为一旦实现了基础支持，就有协作进行这些优化的空间。
- **请求用于 NeoX Docker 任务的 Slurm 脚本**：一名成员请求一个使用 **Docker** 启动 **NeoX 任务** 的 **Slurm 脚本** 示例。
   - 拥有一个参考点对他们来说非常有价值。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1407805054215262350)** (83 messages🔥🔥): 

> `Meta AI 重组, GPT-5-pro 截断, 受银行柜员轮换启发的 Dropout, Meta AI 招聘冻结, 字节跳动 Seed-OSS LLMs` 


- **Wang 晋升后 Meta 拆分为四个团队**：据 [Business Insider](https://www.businessinsider.com/meta-ai-superintelligence-labs-reorg-alexandr-wang-memo-2025-8) 报道，Meta 正在将其 AI 业务重组为新 MSL 负责人 **Alexandr Wang** 领导下的**四个团队**（TBD Lab, FAIR, Product/Applied Research, Infra），同时 **AGI Foundations** 小组将被解散。
   - **Nat Friedman** 和 **Yann LeCun** 现在向 Wang 汇报，**FAIR** 将直接支持模型训练，并且正在考虑开发一个 "omni" 模型。
- **GPT-5-pro 迅速截断提示词**：据[此报告](https://x.com/pvncher/status/1958193631250072024?s=46)显示，**GPT-5-pro** 会在没有任何警告或错误消息的情况下，静默截断超过 **60k tokens** 的提示词，这使得大代码库的提示词变得不可靠。
   - 一些用户还反映 **Cursor** 中的 **GPT-5** 表现得比平时笨得多，有人怀疑正在进行负载卸载（load shedding）。
- **银行柜员 Dropout！**：一条疯传的推文声称 **Geoffrey Hinton** 在注意到**轮换银行柜员**可以防止勾结后构思了 *dropout* ([来源](https://x.com/eigenron/status/1958181550987632927?s=46))。
   - 反应从对这种偶然洞察的钦佩，到对从家庭聚会中诞生 Attention 机制的怀疑和调侃不等。
- **字节跳动发布新 LLMs**：字节跳动的 Seed 团队宣布了 **Seed-OSS**，这是一个新的开源大语言模型系列，可在 [GitHub](https://github.com/orgs/bytedance/repositories) 和 [Hugging Face](https://huggingface.co/models) 上获取。
   - 该团队正邀请社区对模型、代码和权重进行测试并提供反馈。
- **OpenAI 觊觎 AWS 宝座**：OpenAI 的 CFO 表示，公司计划在“未来”出租算力，目标是像一个小型 AWS 那样运作 ([来源](https://x.com/ns123abc/status/1958268338582265948?s=46))。
   - 反应从对 OpenAI 所谓算力短缺的怀疑，到对利润模式转变以及与 Google 和 Microsoft 等现有超大规模云服务商（hyperscalers）冲突的分析不等。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1407823946979741806)** (13 messages🔥): 

> `Wonda AI, 亿万富翁搏击俱乐部, Qwen 图像编辑` 


- **Wonda AI Agent 承诺带来革命**：Dimi Nikolaou 推出了 **Wonda**，这是一个旨在彻底改变视频/音频创作的 AI Agent，称其为“Lovable 为网站做了什么，Wonda 就为内容创作做什么” ([推文链接](https://xcancel.com/dimireadsthings/status/1957805267799740571))。
   - 此次发布引发了对预告媒体质量的热烈反应，早期访问权限通过候补名单授予，邀请函预计在约 **3 周**内发放。
- **黑客帝国翻拍版：小扎对阵奥特曼**：AIST 发布了 ["Billionaires Fight Club Vol.2"](https://xcancel.com/aist_digital/status/1954905895025942918?s=46)，这是一部使用 AI 重新创作的短片，展示了 **Mark Zuckerberg** (Neo) 与 **Sam Altman** (Agent Smith) 在《黑客帝国》中的对决。
   - 该视频获得了积极反馈，促使 AIST 鼓励观众艾特 Sam 和 Zuck，敦促他们转发该片以获得更广泛的曝光。
- **Qwen 图像编辑成功**：Luis C 展示了使用 **qwen-image-edit** 将两张不同的图像合成一张女人抱着娃娃的照片的成功案例 ([推文链接](https://xcancel.com/lucataco93/status/1958581409141944635))。
   - 作为回应，Jay Sensei 声称在 lmarena 进行的测试中，**nano banana** 的表现优于 **Qwen**。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1407829749526565056)** (25 条消息🔥): 

> `Hackathon 开始时间, ChatGPT CUDA 谎言, Hackathon 先决条件, 单个超大 Epoch vs 多个较小 Epoch, CUDA vs Triton` 


- **Hackathon 将于周六上午 9:30 开始**：据一名成员透露，Hackathon *很可能*在周六上午 **9:30** 左右开始。
- **ChatGPT 编造 CUDA 谎言**：一位成员报告称，**ChatGPT** 在 **CUDA** 中的 **float3 对齐**和 **大小**问题上公然撒了两次谎，但对 **ChatGPT** 表示理解，因为从 **OpenCL** 和 **OpenGL** 的实现来看，这是一个很难处理正确的问题。
   - 该成员证实 **CUDA** 中不存在填充（padding）。
- **关于 Hackathon 先决条件和申请的疑问**：一位成员询问了 **GPU hackathon** 的先决条件以及申请通道是否仍然开放。
   - 聊天中没有明确回答这个问题。
- **关于单次 vs 多次 Epoch 的辩论**：一位成员询问，对于 **CLM**（因果语言模型），是在超大数据集上进行 **1 epoch** 训练更好，还是在较小数据集上进行多次 Epoch 更好，以及目前最新的 Scaling Law 是什么。
   - 另一位成员回应称，他们处理的是较小的模型，在规模较大时，两倍数据的 1 epoch 与一半数据的 2 epoch 性能相同。
- **CUDA 与 Triton 正面交锋！**：一位成员询问 Hackathon 会使用 **CUDA**、**Triton** 还是其他工具。
   - 有人提到两者都可以，**Triton** 可能会帮助参与者提高开发速度；并暗示参与者将使用较新的 **Nvidia 芯片**。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1408081843097571348)** (1 条消息): 

> `Triton, AMD, NVIDIA, GPU, 数据布局` 


- **通过 Triton 处理 AMD 与 NVIDIA GPU 的数据布局差异？**：一位用户询问在使用 **Triton** 时，**AMD** 和 **NVIDIA** GPU 之间的数据布局差异是否需要调整代码，特别是关于行优先（row-wise）与列优先（column-wise）数据读取的问题。
   - 用户澄清他们询问的不是 **tile sizes** 或 **grid layouts**，而是由 **Triton AMD 后端**自动处理的底层数据转置。
- **AMD vs NVIDIA**：对比了消费级 GPU 对消费级 GPU，或服务器级 GPU 对服务器级 GPU 的架构。
   - 对 **AMD** 和 **NVIDIA** 架构进行了比较。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1408113668868018246)** (10 条消息🔥): 

> `CUDA 部署, CudaWrangler, 动态链接` 


- **在没有 CUDA toolkit 的机器上运行 CUDA 程序**：一位用户寻求关于在没有安装 CUDA toolkit 但配备了 NVIDIA GPU 的机器上部署 CUDA 程序的建议。
   - 一位成员建议利用 **Driver API** 和 **CudaWrangler** 库 ([CudaWrangler/cuew](https://github.com/CudaWrangler/cuew)) 来查询驱动程序，而不会导致程序崩溃。
- **动态链接与 PTX 烘焙简化 CUDA 部署**：原帖作者报告称，通过从“动态加载”切换到“动态链接”并禁用 *runtime/cudart* 依赖，取得了成功。
   - 他们还能够将 **PTX** 直接嵌入到二进制文件中，从而消除了对独立 **PTX** 文件的需求。
- **ldd 辅助识别和打包 Linux 上的 CUDA 程序依赖**：一位成员建议使用 **ldd** 来识别依赖项，设置 **rpath**，并将它们随二进制文件一起发布，类似于 Linux 上的“Windows 方式”。
   - 原帖作者指出该程序在 Windows 和 Linux 之间具有跨平台兼容性，但 macOS 尚未测试。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1408177180583792731)** (1 条消息): 

> `2025 PyTorch 贡献者奖, 表彰 PyTorch 创新` 


- **PyTorch 奖项截止日期临近！**：**2025 PyTorch 贡献者奖**的提名将于 **8 月 22 日**截止，不要错过表彰在 **PyTorch 生态系统**中推动创新和影响力的个人的机会。
   - 立即通过此[链接](https://linuxfoundation.research.net/r/8XD5T8N)提交您的提名，并查看[优秀提名技巧](https://pytorch.org/blog/nominations-open-for-the-2025-pytorch-contributor-awards/)。
- **通过提名推动创新**：表彰 **PyTorch 生态系统**中不断创新的贡献者。
   - 在 **8 月 22 日**之前提交提名。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 条消息): 

honeyspoon: 与 sglang 之类的工具相比，infinity server 的 embedding 速度有多差？
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

snektron: 我更喜欢 Stolwijker
  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1407932292470542387)** (11 条消息🔥): 

> `AMD GPU debugger, rocGDB, SPIRV parser, libspirv` 


- **AMD GPU debugger 获得反汇编和 Wave Stepping 功能**: 一名成员正在开发一个 **AMD GPU debugger**，并添加了反汇编和 wave stepping 功能，展示在[这段视频](https://cdn.discordapp.com/attachments/1233704710389764236/1407932291912695949/2025-08-21_06-20-14.mp4?ex=68a88f60&is=68a73de0&hm=6e9ca7ceed29674c6943e989940a6c8e144707797c5abd571e1cfe60d3f3210d)中。
   - 该调试器不依赖于 **amdkfd KMD**，而是使用一个 mini UMD 驱动程序和 linux kernel debugfs 接口，旨在成为 **rocdbgapi** 的等效替代方案。
- **放弃 rocGDB 转而使用自定义驱动程序**: 一名成员正在构建一个不依赖于 **rocGDB** 的 AMD GPU debugger，而是使用 mini UMD 驱动程序加上 linux kernel debugfs 接口来读写 GPU 寄存器。
   - 其目标是主要面向图形开发人员，旨在实现一个 **rocdbgapi** 的等效工具，至少目前是这样。
- **自己编写 SPIRV Parser？**: 一名成员询问关于构建自己的 **SPIRV parser** 以进行反汇编、反射和调试信息提取的事宜，并提到 **SPIRV spec** 似乎非常直观。
   - 他们注意到缺乏处理调试信息的合适库，因此考虑进行完整实现。
- **libspirv 相当简单**: 一名成员建议使用 **libspirv**，并指出 **SPIRV spec** 包含了自己动手实现所需的所有信息。
   - 原帖作者在认可该建议的同时，决定实现一个自定义解决方案以获得更好的集成效果。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1408106371680960602)** (2 条消息): 

> `C=AB matmul, ALU utilization, buffer read bandwidth, float4x4 matmul, float4 / metal::dot kernel` 


- **分块 C=AB Matmul 中的 GPU ALU 受限**: 一名成员编写了一个分块（tiled）**C=AB matmul** kernel，其中每个线程使用 **float4x4 matmul** 来计算 C 的 4x4 分块，并观察到 **ALU utilization/limiter** 为 **55/75%**，而 **buffer read bandwidth** 为 **35%**。
   - 他对此感到惊讶，想知道 **float4x4 matmul** 是否在专用硬件中执行，并分享了 [kernel 的 gist](https://gist.github.com/0xekez/c94ba3d5b43df10d17c98581e91280e3)。
- **朴素 Kernel 性能优于分块 Matmul**: 同一位成员指出，一个使用 **float4 / metal::dot** 的更朴素的 kernel 比分块 kernel 快 **2 倍以上**。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/)** (1 条消息): 

miserlou1241: 非常酷！
  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1408081014441377833)** (12 条消息🔥): 

> `torch.compile errors, local evaluation issues` 


- ****Torch.compile** 抛出意外错误**: 一名成员报告了使用 **torch.compile** 时出现的 *unexpected error*，并分享了两个解决方案：一个使用了 **torch.compile**（提交编号 34166），另一个则没有（提交编号 34160）。
   - 尽管出现了错误，提交仍然成功注册，使该成员排名第 2，并注明使用的 GPU 是 **B200**。
- **解决本地评估工具问题**: 一名成员询问了关于本地代码评估的问题，表示 **eval.py** 无法工作，特别是询问了关于 `POPCORN_FD` 的情况。
   - 另一名成员澄清说 `POPCORN_FD` 是输出文件的文件描述符，并建议将其设置为 `1` 以指向 stdout。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1407815994784747571)** (11 条消息🔥): 

> `Trimul Leaderboard Updates, B200 Performance, H100 Performance, MI300 Performance` 


- **MI300 在 Trimul 取得成功成绩**: 一名成员成功向 `trimul` 排行榜提交了在 **MI300** 上 **3.50 ms** 的成绩。
   - 另一个在 **MI300** 上的提交以 **5.83 ms** 的成绩获得第二名。
- **B200 霸榜 Trimul 排行榜**: 一名成员在 **B200** 上以 **8.86 ms** 的时间获得 `trimul` 排行榜第 6 名，随后进步到 **7.29 ms** 获得第 4 名。
   - 同一位成员在 **B200** 上多次获得第 3 名，最佳时间达到 **4.54 ms**，随后又实现了一次 **2.15 ms** 的成功运行。
- **H100 稳居第二**: 一名成员在 **H100** 上以 **3.80 ms** 的时间获得 `trimul` 排行榜第二名。
   - 这次提交突显了 **H100** 平台极具竞争力的性能。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1407992161051475978)** (3 messages): 

> `Opus 4.1, Steel Plate Production, Task Emphasis, Red Science Production` 


- **Opus 4.1 发现财富，助力工厂**：在对 **Opus 4.1** 进行钢板生产测试时，它意外地开始开采铜矿并提取石油。
   - 这表明其对*当前任务的重视程度不够*，促使开发团队转向观察设置，以研究 **Opus 4.1** 如何提高其专注度。
- **AI 自动化 Red Science**：如截图所示，AI 系统成功实现了 **Red Science** 生产的自动化。
   - 该系统正确识别并生产了自动化创建科技包（science packs）所需的组件。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1407954456745873438)** (3 messages): 

> `ND Layouts, colex` 


- **通过 Colex 访问 ND Layouts 中的元素**：一位成员询问在使用整数作为 **ND layout** 的索引时，元素的访问顺序是怎样的。
   - 另一位成员澄清该顺序是 **colex**（列优先/左优先）。
- **确认 Colex 顺序**：一位用户确认，在 ND layouts 中使用整数索引时，元素访问顺序确实是 **colex**。
   - 这再次强调了 **colex**（即列优先顺序）是此类索引的标准方法。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1408129525929345044)** (10 messages🔥): 

> `Infiniband at home, Distributed training library, NCCL backend, IBGDA requirements` 


- **寻求家庭实验室 Infiniband 方案**：一位成员正尝试在家的 **4090** 和 **5090** 之间设置 **Infiniband**，以进行分布式训练/推理。
   - 他们在 eBay 上以 25 美元的价格购买了一些 **ConnectX-3 卡**，但发现驱动程序仅适用于 Ubuntu 20.04 及更早版本。
- **DIY 分布式训练框架兴起**：一位成员正在构建自己的 **PyTorch 分布式训练库**，并使用迷你版 **NCCL** 作为后端。
   - 另一位成员对此表示兴趣，认为这是学习细节的一种方式。
- **深入研究 NVIDIA Networking 文档**：一位成员建议查看 Internet Archive 上的旧版 [NVIDIA networking 文档](https://docs.nvidia.com/networking/index.html) 以寻找相关驱动程序。
   - 该成员希望这能提供更多细节。
- **CX4 或 CX5 卡具备 GPU-Aware 特性**：一位成员指出，许多 GPU-aware 功能依赖于 **ConnectX-4 (CX4)** 或 **ConnectX-5 (CX5)** 及更新型号的网卡。
   - 他们举例说明 **IBGDA** 需要 **CX5** 或更新型号。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1407883262126456913)** (33 messages🔥): 

> `Infinite Memory, Arxiv paper guide, LLMs for Legal Field, HRM Models Analysis, Message Passing Approaches` 


- **福布斯曝光 Grok 聊天记录**：[Forbes](https://www.forbes.com/sites/iainmartin/2025/08/20/elon-musks-xai-published-hundreds-of-thousands-of-grok-chatbot-conversations/) 的一篇文章透露，**Elon Musk 的 xAI** 发布了数十万条 **Grok** 聊天机器人的对话。
   - 一位成员向 *@grok* 询问这是否属实。
- **图灵完备性需要无限内存**：一位成员认为图灵完备性需要无限内存，因此由于内存不足，宇宙无法创造出图灵完备机。
   - 另一位成员开玩笑地建议，让计算机足够慢，或许可以利用宇宙的膨胀来解决空间问题；而另一位成员补充道：*真实的内存需要被检索，距离越远，检索所需的时间就越长*。
- **牛津指南助力初露头角的 Arxiv 作者**：一位成员分享了一份由牛津大学教授编写的 [Google Docs 指南](https://docs.google.com/document/d/16R1E2ExKUCP5SlXWHr-KzbVDx9DBUclra-EbU8IB-iE/edit?tab=t.0#heading=h.16t67gkeu9dx)，旨在帮助程序员撰写关于 LLM 训练的 Arxiv 论文。
   - 该用户想分享见解，但不知从何下手。
- **ARC Prize 分析 HRM 模型**：一位成员分享了 [fxtwitter 帖子](https://fxtwitter.com/arcprize/status/1956431617951740044) 和 [ARC Prize 博客文章](https://arcprize.org/blog/hrm-analysis) 的链接，其中分析了 HRM 模型。
   - 这是为了回应另一位用户关于 HRM 模型是否值得花时间学习的问题。
- **图片展示消息传递方法**：一位成员分享了一张插图，展示了神经网络中消息传递（message passing）的不同方法。
   - 该图片源自一本书，可通过 [arXiv 上的 PDF](https://arxiv.org/pdf/2104.13478) 获取。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1407812166702207027)** (46 messages🔥): 

> `Personality GAN, AI Welfare, Genome Conscious?, Super Weight, LLM Preferences` 


- ****SpongeBob GAN** 亮相！**: 一名成员提议了一种 Personality GAN，其 Generator = LLM 且 Discriminator = LLM，通过 LoRA 进行 fine-tuning，直到判别器无法区分真假 **Sponge Bob**。
   - 难点在于寻找一个尚未在 **Sponge Bob** 数据上进行过大量训练的 LLM。
- ****AI Welfare** 被严肃对待！**: 讨论了一篇关于 *Taking AI Welfare Seriously* 的论文 [arxiv link](https://arxiv.org/abs/2411.00986)，内容涉及 Anthropic 关于 *Exploring Model Welfare* 的文章 [Anthropic link](https://www.anthropic.com/news/exploring-model-welfare)。
   - 这也与 [Anthropic 的另一篇](https://www.anthropic.com/research/end-subset-conversations) 关于 end-subset conversations 的文章有关。
- ****LLM Weight** 的古怪现象！**: **Llama 3 7B** 权重矩阵中单个数字的变化就导致其输出乱码，引发了关于意识/身份的疑问 [Apple link](https://machinelearning.apple.com/research/the-super-weight)。
   - 一位成员问道：*“他们是否仅通过调整一个数字就抹除了它的‘意识’/‘身份’？”*
- ****LLM Preferences** 显现！**: 有人指出模型在 pre-training 过程中会形成类似人类的表征，且 LLM 确实存在偏好，并引用了 [这篇 LessWrong 文章](https://www.lesswrong.com/posts/eWdzuHXzRdBkg49R9/favorite-colors-of-some-llms)。
   - 一位成员评论道：*“在我那个年代，我们管这叫类别不平衡偏差 (class imbalance bias)。”*
- ****AI Duality** 引发辩论！**: 讨论涉及 AI 作为一种双重用途技术，因其普适性而适用于所有领域 [QuantaMagazine link](https://www.quantamagazine.org/the-ai-was-fed-sloppy-code-it-turned-into-something-evil-20250813/)。
   - 一位成员表示 *“聪明是相对的”*，并且 [恒温器也具有 Agency](https://www.youtube.com/watch?v=PiJwIUGJGmw&t=19s)，因为它们会对自身及其外部环境建模。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1407827073749221577)** (8 messages🔥): 

> `Yann LeCun's position at FAIR, Thermodynamic computing chip, AI Slurs, Energy Efficiency in AI` 


- ****Zuckerberg** 可能解雇了 **LeCun**？！**: 一名用户根据 [Zuckerberg 的一条动态](https://www.threads.com/@zuck/post/DMiwjXJSYCd?xmt=AQF06RR3brSSoKTMcQrLDyX6WGtg9UdpubK94sSm8FIPsg) 推测 **Yann LeCun** 可能离开 **FAIR**。
   - 另一位成员暗示 **LeCun** 可能被降职，且 **Meta** 正在从开源模型领域撤退。
- **Clanker Cogsucker 机器人 AI 侮辱词汇走红！**: 一名用户分享了 [一篇 Rolling Stone 的文章](https://www.rollingstone.com/culture/culture-features/clanker-cogsucker-robot-ai-slurs-viral-1235401262/)，讨论了如 *clanker* 和 *cogsucker* 等新型 **AI 侮辱词汇 (AI slurs)** 的出现。
- **首款热力学计算芯片完成 Tape-out**: 一名成员发布了来自 [Tom's Hardware 的文章](https://www.tomshardware.com/tech-industry/semiconductors/worlds-first-thermodynamic-computing-chip-)，报道了 *全球首款热力学计算芯片* 已达到 Tape-out 阶段。
- **AI 行业并不关心能效**: 一名用户分享了 [一段 YouTube 视频](https://www.youtube.com/watch?v=LTCbx5KdqpU)，认为 **AI 行业** 普遍不优先考虑 **能效 (Energy Efficiency)**。
   - 他们指出，另一家具有类似价值主张的公司已经破产，这表明行业并不关心能效。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1407849425656746066)** (67 条消息🔥🔥): 

> `max_steps 困惑, levelbot space 访问, 高 token 下的模型幻觉, Pro 版本支付问题, root mean square norm 量化误差` 


- **关于 max_steps 参数的困惑**：一名成员对 **max_steps** 参数及其在 **5090** GPU 上使用 **vllm** 的实现感到困惑，并询问 [LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B) 模型是否合适。
- **Token 限制触发幻觉**：一名成员询问了模型开始产生幻觉的 token 限制，并对任何模型能在 **1 million tokens** 下有效运行表示怀疑。
   - 另一名成员分享了 [Hugging Face 的 Agents 课程](https://huggingface.co/learn/agents-course/unit0/introduction) 链接和一个 Discord 频道，建议将这些资源作为潜在的解决方案。
- **用户报告 Pro 版本支付问题**：一名用户报告称被收取了两次 **Pro 版本** 费用却未获得服务，被建议发送邮件至 website@huggingface.co 并在指定的 [MCP 频道](https://discord.com/channels/879548962464493619/1389546106970701865) 寻求帮助。
- **自定义损失函数微调 SFTTrainer**：一名成员分享了在 **ChatGPT** 帮助下创建的自定义损失函数，旨在与 **SFTTrainer** 配合使用，以增强模型对医学文本中特定**否定词（negation words）**的关注。
   - 另一名成员建议改用带有偏好对（preference pairs）的 **DPO**，而另一位成员则强调了在医学领域挖掘难负样本（hard negatives）后使用 triplet loss 的实用性。
- **LLM 训练中 SFT 与 DPO 的比较**：成员们讨论了 **DPO** (Direct Preference Optimization) 与 **SFT** (Supervised Fine-Tuning) 的有效性，一名成员指出 *DPO 与推理（reasoning）没有关系*，但在 **SFT** 之后进行 **DPO** 比仅进行 **SFT** 能产生更好的结果。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1408040029137142094)** (3 条消息): 

> `AgentX 交易平台, 语言扩散模型, 本地 AI 工作区 PDF 阅读器` 


- ****AgentX** 承诺打造 AI 交易智囊团**：全新的 [**AgentX**](https://www.linkedin.com/posts/alaa-salamah-96167b227_agentx-agentx-api-activity-7364245216851050498-BfRO?utm_source=share&utm_medium=member_desktop&rcm=ACoAADjfvpkBwpCXn_8Pmby7ixjV93Dje5TcmgUHi) 平台旨在提供一个汇集了最聪明 AI 大脑——**ChatGPT**、**Gemini**、**LLaMA**、**Grok**——共同协作的交易台。
   - 目标是让这些模型进行辩论，直到它们对最佳操作达成一致，为交易者提供一个可以完全信任的系统。
- **不到 80 行代码复现扩散语言模型**：一名成员使用 🤗 Transformers 在不到 80 行代码内复现了 Nie 等人 (2025) 的论文 *Large Language Diffusion Models* 的部分内容。
   - 该[项目](https://github.com/gumran/language-diffusion)在 **TinyStories** 数据集上微调了 **DistilBERT**，结果好于预期，目前正在寻求反馈和 Star。
- **本地优先的 PDF 阅读 AI 工作区亮相**：一名成员在 Product Hunt 上发布了一个本地优先（local-first）的 AI 工作区 PDF 阅读器，并分享了[链接](https://www.producthunt.com/products/collate-2?launch=collate-4)。
   - 他们请求社区的支持。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1408102264597385228)** (1 条消息): 

> `Hugging Face Learn 课程, 422 错误` 


- **Hugging Face Learn 课程页面宕机**：一名成员报告称 [Hugging Face LLM 课程的一个页面](https://huggingface.co/learn/llm-course/en/chapter12/3a) 无法访问。
   - 该页面显示 **422 错误**。
- **Hugging Face Learn 课程需要修复**：一名用户报告 Hugging Face Learn 课程页面宕机并显示 **422 错误**。
   - 该问题需要解决，以便用户可以访问内容。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1407997140890026077)** (4 messages): 

> `Hugging Face Certificates, Agents vs MCP Course, Agent tool, LLM tasks` 


- **Hugging Face 证书位置难倒用户**：一位用户询问在哪里可以找到他们的 **Hugging Face 证书**，以便将其发布到 LinkedIn。
   - 他们提到在平台或电子邮件中都找不到这些证书。
- **Agents 课程与 MCP 课程引发辩论**：一位用户正在纠结是在完成 Agents 课程的 Unit 1 后转向 **MCP Course**，还是先完成 **Agents Course**。
   - 由于时间限制，他们想知道应该优先考虑哪门课程。
- **Agent 工具功能揭秘**：一位用户寻求关于 **Agent Unit 1** 成功运行的解释。
   - 他们理解 Agent 使用工具（函数），并触发这些工具来执行任务，而不是直接调用 **LLM**。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1407887543743283231)** (19 messages🔥): 

> `Gems for podcast generation, NotebookLM podcast length, Customizing NotebookLM podcasts, Analyzing Terms of Use and Privacy Policies, South Park episode on Terms and Conditions` 


- **AI 大师分享生成长播客的 Gems**：一位用户询问如何在 NotebookLM 中从 3-4 小时的 YouTube 视频生成更长的播客，对此一位用户建议使用设定的提示词（prompts）来逐段规划整个文案。
   - 一位用户分享了[一个工作流](https://github.com/para-droid-ai/scratchpad/blob/main/assistant-workflows-tasks-personas/deeper_podcast_synthetic-082025.txt)，用于创建一个“深度研究报告框架”，然后可以使用 Gems, Gemini, PPLX 或 ChatGPT 生成播客。
- **通过自定义解锁更长的 NotebookLM 播客**：一位用户询问 NotebookLM 中的播客长度限制，另一位用户指出在 **Customize** 选项（三个点）中，可以将播客长度设置为 45-60 分钟。
   - 另一位用户补充说，指定主题可以让 Bot *集中讨论特定话题*，而不是指望它将所有重要内容都塞进一个播客中。
- **隐私政策偏执：医疗保健网站的妥协被曝光**：一位用户在想起*有人使用 AI 工具分析这两份文件并大有发现*后，使用 Gemini 和 NotebookLM 分析了一家医疗保健公司的隐私政策和使用条款。
   - 该用户对*你向这些公司泄露了多少信息*感到惊讶，并认为这种方法对于理解使用条款（Terms of Use）和隐私政策非常有用。
- **《南方公园》预言了接受条款与条件的痛苦**：一位用户建议去看看那集关于接受条款与条件的经典 **South Park** 剧集。
   - 另一位用户回想起一个游戏，其 EULA/隐私/条款中隐藏了一个竞赛：第一个拨打特定电话号码的人可以赢得一千美元，而这个奖项在六个月内都无人认领。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1407818234690011138)** (51 messages🔥): 

> `Video Length Limits, Study guide on android app, Audio Language Change, Public Sharing Issue, Notebook LM API` 


- **Android 应用功能对齐（Feature Parity）延迟**：用户要求 NotebookLM Web 端和 Android 应用之间实现更多的**功能对齐**，特别是学习指南功能。
   - 一位用户表示，目前的原生应用*几乎没用*，因为学习指南依赖于笔记功能，而原生应用中缺少该功能。
- **自定义屏幕上提供语言更改选项**：一位用户询问如何更改 iOS 应用中生成的音频概览（Audio Overview）的语言。
   - 另一位用户回答说，语言设置可以在 **Customize** 菜单中找到。
- **尚不支持向公众公开分享 Notebook**：一位用户报告称，尽管拥有 Pro 账户，但仍无法公开或向外部分享 Notebook。
   - 该功能目前尚未开放。
- **NotebookLM 缺乏官方 API，但存在变通方法**：一位用户询问 NotebookLM 的 API。
   - 另一位用户建议使用 **Gemini API** 作为变通方案。
- **NotebookLM 中的 OCR 操作**：用户讨论了 NotebookLM 是否对多模态 PDF 执行 OCR 操作。
   - NotebookLM 支持 PDF 并且正在改进图像处理，但 OCR 识别尚不完美，用户可能需要重新上传 PDF 或使用**外部 OCR 工具**。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1407807040277053510)** (65 messages🔥🔥): 

> `Base Model Release, Ideal 30B Model, FA2 and Context, Qwen Scaling, Importance Matrix Calibration Datasets` 


- **字节跳动发布长上下文模型**：字节跳动发布了一个具有极长上下文的 Base Model，其特点是没有 MHLA，没有 MoE，甚至没有 QK norm，详见[这张图片](https://cdn.discordapp.com/attachments/1149866623109439599/1407959284280459305/image.png?ex=68a8a883&is=68a75703&hm=b8a5430da1f445204c76334cef07358c8d9b815da989424483a5ecf3bc65c790)。
   - 该模型在架构上被描述为 *vanilla*（原生），人们希望他们能发布一篇包含更多解释的论文。
- **Seed-OSS-36B 缺失 GGUF 引发关注**：用户想知道为什么还没有 **Seed-OSS-36B** 的 **GGUF** 版本，因为这类版本通常出现得很快。他们引用了[这个链接](https://x.com/adityastomar_/status/1958048129275805867)，询问这是否意味着对 ASIC 持悲观态度。
   - 据指出，延迟可能是由于自定义的 **vllm** 实现，且由于其架构名为 ["SeedOssForCausalLM"]，目前 **llama.cpp** 尚未支持该架构。
- **Seed 模型实现了 Dropout 和 Bias**：**Seed** 模型拥有类似于 **LLaMA** 的自定义 MLP 和 Attention 机制，但增加了 Dropout、输出偏置项（bias term）以及 **qkv** heads 的偏置项，这些被解读为正则化技术。
   - 成员们想知道该模型训练了多少个 epoch，但确认将其重命名为 **LLaMA** 是行不通的。
- **Qwen 通过 RoPE Scaling 实现 512k 上下文**：正如[这个 Hugging Face 数据集](https://huggingface.co/datasets/eaddario/imatrix-calibration)中所讨论的，**30B** 和 **235B Qwen 2507** 模型可以通过 **RoPE** scaling 实现 **512k** 的上下文。
   - 这些数据集用于生成重要性矩阵（imatrix），有助于在量化过程中减少误差。
- **Cursor 的 Kernel 博客获得好评**：成员们分享了 [Cursor kernel 博客](https://x.com/stuart_sul/status/1957927497351467372)的链接。
   - 有人评价说 *Cursor 在这方面表现出色（cooked）*。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1407950357379809300)** (47 messages🔥): 

> `DeepSeek V3.1, R-Zero LLM Training Method, Energy availability in China vs US, Kimi K2 combined with Better image gen than gpt 5` 


- **DeepSeek V3.1 发布：增量式进步**：新的 **DeepSeek V3.1** 模型已发布，一些成员指出这更像是一个带有某些退步的*增量改进*，参考了 [DeepSeek 官方页面](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)。
- **DeepSeek 支持 Anthropic API**：**DeepSeek** 现在支持 **Anthropic API**，扩展了其功能和覆盖范围，正如 [X 上的公告](https://x.com/deepseek_ai/status/1958417062008918312)所述。
- **R-Zero：自我进化的 LLM**：一份关于 **R-Zero** 的综合研究被分享在 [PDF](https://cdn.discordapp.com/attachments/1371757564005711973/1408153973751545896/R-Zero__A_Comprehensive_Study_of_a_Self-Evolving_LLM_Training_Method.pdf?ex=68a8b515&is=68a76395&hm=dcba93436f636eeec364d08d08e1603131147faaa595637065f2e226772005f2&) 中，这是一种从零人类数据开始并独立改进的自我进化 **LLM 训练方法**。
- **中国优先保障能源可用性**：一位成员指出，在中国，*能源可用性被视为理所当然*，这与美国关于数据中心功耗和电网限制的辩论形成对比，参考了[这篇《财富》杂志的文章](https://fortune.com/2025/08/14/data-centers-china-grid-us-infrastructure/)。
- **更好的图像生成 + Kimi K2**：一位成员指出，如果 **Kimi K2** 能结合**比 GPT-5 更好的图像生成能力**，将会更加强大（OP），并分享了[这个 Reddit 链接](https://www.reddit.com/r/ChatGPT/s/vUrGedSwY5)。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1407819836352106507)** (36 messages🔥): 

> `Gemini 2.5 Pro Failure, Qwen CLI Charging, GPT-5 Benchmarks, DeepSeek v3.1 Pricing, OpenRouter Think Mode` 


- ****Gemini 2.5 Pro 失败而 Flash 成功****：一位成员报告称 **Gemini 2.5 Flash** 可以工作，但 **Gemini 2.5 Pro** 持续失败，而如果设置了计费，`gemini/gemini-2.5-pro-preview-06-05` 则可以工作。
   - 另一位成员报告因 **qwen-cli** 进程被扣除 **$25**，正在寻求退款。
- ****用户因使用 Qwen CLI 被意外扣费****：一名用户在通过 OAuth 使用 Google 身份验证后，因使用 **qwen-cli** 被扣除 **$25**，尽管其目标是获取 Alibaba Cloud 的免费额度。
   - 他们提交了一个工单，展示了控制台记录中 **一次调用花费 $23 且没有输出** 的情况。
- ****社区渴望对 GPT-5 低推理模型进行基准测试****：成员们正在对 **gpt-5-mini** 和 **gpt-5-nano** 进行基准测试，因为他们在完整版 **gpt-5** 上受到了速率限制，尽管一位用户声称 *gpt-5-mini 非常出色且便宜*。
   - 频道中已经发布了 **gpt-5-mini** 的测试结果和 PR。
- ****DeepSeek v3.1 价格显著上涨****：用户报告称，从 2025 年 9 月 5 日开始，DeepSeek 将提高两种模型的价格，以匹配 reasoner 模型的价格。
   - 与新的 **deepseek 3.1** 相比，输入价格从 **$0.25** 上涨至 **$0.27**。
- ****OpenRouter 需要 Think 模式****：一位用户报告 **OpenRouter** 似乎没有 "think" 模式，但可以通过命令行使用以下代码片段来调用：`aider --model openrouter/deepseek/deepseek-chat-v3.1 --reasoning-effort high`。
   - 社区建议更新模型配置以解决此问题。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1407817255621754893)** (3 messages): 

> `aider stdout issue, polyglot benchmark on llama cpp` 


- **Aider 的标准输出（stdout）难题**：一位用户报告了 **program output/stdout** 无法在 **aider** 中显示的问题，并发布了一张 [图片](https://cdn.discordapp.com/attachments/1133060505792159755/1407817255433277440/image.png?ex=68a8ccfd&is=68a77b7d&hm=c93b6e3d3d4d1b0dc321355cd459dbd4e8371fd5bfe1c43c82d2701b9b6cd831&)。
- **破解 Polyglot 基准测试结果**：一位在本地 **llama cpp model** 上运行 **polyglot benchmark** 的用户询问如何获取每个语言的结果。
   - 该用户随后找到了 [解决方案](https://discord.com/channels/1131200896827654144/1400603686350360678/1400993983999770694) 并分享了链接，供其他寻求特定语言基准测试结果的人参考。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

end4749: <@293486003245809664> 垃圾信息？ ^
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1408187482075299851)** (1 messages): 

> `marimo notebooks, Graph RAG with DSPy, DSPy modules optimization` 


- **Marimo Notebooks：Jupyter 的精神继任者**：一位成员一直在发布关于 [**marimo notebooks** 的教程](https://www.youtube.com/watch?v=2aepn9uRVOM)，它可以同时作为 notebook、Python 脚本和应用运行。
   - 该教程强调了在迭代 **Graph RAG with DSPy** 的想法时 **marimo** 的实用性。
- **未经优化的 DSPy 流水线**：展示的 **DSPy pipeline** 故意没有进行优化，以强调仅通过 signatures 和 modules 就能实现多少功能。
   - 这种方法专注于在深入优化之前，通过以各种方式组合 **DSPy modules** 来进行快速迭代。
- **深入优化**：即将推出的视频和博客文章将深入探讨 **DSPy modules** 优化的主题。
   - 当前的教程为那些想要开始使用的人提供了 **marimo** 的入门介绍。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1408079463996199084)** (5 messages): 

> `IBM AutoPDL paper, DSPy code readability, Justification of work` 


- **IBM 的 AutoPDL 主张被驳回**：一位成员认为没有必要回应每一个主张，暗示每个人都在寻找一个角度来证明自己工作的合理性，并且关于不可读性的主张是错误的。
   - 他们表示 *DSPy 代码和 prompt 在任何意义上都极其具有人类可读性，甚至可以说非常优美。*
- **捍卫 DSPy 代码可读性**：一位成员辩称 **DSPy** 的代码和 **prompts** 极其易于人类阅读、易于访问且清晰，并对相反的主张提出了挑战。
   - 该成员强调，代码的可读性使其易于理解和使用。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1407849483231825921)** (28 messages🔥): 

> `dspy.GEPA 版本，微调 dspy 描述，保存优化后的程序，GEPA 的上下文长度，KPMG 入职` 


- **DSPy 的 GEPA 在 v3.0.1 中现身**：一位成员询问包含 **GEPA** 的 **dspy** 库版本，另一位成员确认其在版本 **3.0.1** 中可用，如附带的[截图](https://cdn.discordapp.com/attachments/1161519469319946286/1407936409615990904/image.png?ex=68a89336&is=68a741b6&hm=72219936a525599fc3faca9d127d106a09e0639e9eeae1564a8bbfc196b07ffa&)所示。
- **DSPy 微调：描述性还是 vanilla？**：在微调过程中，一位成员询问是否通常为 **dspy.InputField()** 和 **dspy.OutputField()** 使用 *"vanilla 描述"*，以便让优化器自由思考。
- **DSPy 将优化后的程序保存在 Pickle 中**：一位用户报告了保存优化程序时的问题，指出元数据仅包含有关 **dependency versions**（依赖版本）的信息，而不包含程序本身，即使使用了 `optimized_agent.save("./optimized_2", save_program=True)`。
- **GEPA 被截断**：当用户为 **GEPA** 设置了 **32k** 的最大上下文长度但仍收到被截断的回复时，成员们讨论了长推理的复杂性以及多模态设置的潜在问题。
   - 一位成员引用一个复杂的提示词示例开玩笑说：*"想象一下必须维护那个东西"*。
- **RAG 是大材小用，直接拼接即可（或者不）**：成员们开玩笑地争论 **RAG** (Retrieval-Augmented Generation) 还是简单的**拼接**更适合处理税法或农作物保险文件等任务，并承认数百万份文件的规模有时确实需要 **RAG**。
   - 一位成员调侃道：*"RAG 是大材小用。直接把税法拼接起来就行，"* 而另一位反驳道：*"哦，我猜那超过 100 页了。好吧，那 RAG 挺好的。"*


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1407880904814366720)** (13 messages🔥): 

> `command-a-03-2025 的引用问题，保证引用，command-a-reasoning 发布，使用 Langchain 构建 RAG，Cohere 对比 Qwen3-coder 30B` 


- **`command-a-03-2025` 间歇性引用引发的提示词挫败感**：一位用户报告称 `command-a-03-2025` 仅间歇性地返回引用，即使 maxTokens 设置为 8K，这导致了生产环境中的信任问题，并寻求某种保证。
   - 一位 Cohere 成员澄清说 `command-a-03-2025` 在引用方面使用 "fast" 模式（根据 [API 参考](https://docs.cohere.com/reference/chat#request.body.citation_options.mode)），引用并不保证一定会生成，但可以通过系统提示词引导模型，并且最新发布的 SOTA 模型 **command-a-reasoning** 可能也会有所帮助（参见 [博客](https://cohere.com/blog/command-a-reasoning)）。
- **Langchain RAG 探索开启**：一位成员正在学习 Langchain 以构建 **RAG** (Retrieval-Augmented Generation) 应用。
   - 他们提到打算使用 **command-a-reasoning**，期待 **command-a-omni** 的发布，并对未来名为 **Command Raz** 的模型表示期待。
- **Cohere 与 Qwen 争夺本地 LLM 地位**：一位用户正在寻找 Cohere 的替代方案来替代 **Qwen3-coder 30B** 模型，目标是使其能够运行在 **64GB M4 Max** 配置上。
   - 该用户 *非常想尝试 Cohere 的方案来替代本地强力模型 Qwen3-coder 30B*，以便能适配其 64GB M4 Max。


  

---

### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1408103056800874497)** (1 条消息): 

> `Command A Reasoning Model, Enterprise AI, Agentic AI Platform` 


- **Cohere 发布 Command A Reasoning 模型**：Cohere 发布了 **Command A Reasoning**，这是其最新的用于推理任务的企业级模型，在 Agentic 和多语言基准测试中表现优于其他私有部署模型；该模型可通过 [Cohere Platform](https://dashboard.cohere.com/playground/chat?model=command-a-reasoning-08-2025) 和 [Hugging Face](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025) 获取。
- **Command A Reasoning 规格与特性公开**：新模型专为企业需求设计，提供高度安全、高效且可扩展的部署选项，可在单张 **H100** 或 **A100** 上运行，支持 **128k** 上下文长度，在多 GPU 上可扩展至 **256k**；更多信息请参阅 [Cohere 博客](https://cohere.com/blog/command-a-reasoning)。
- **Token Budget 功能控制成本与计算资源使用**：Cohere 的 Command A Reasoning 具备 **token budget** 设置，用于直接管理计算资源使用和成本控制，无需区分推理和非推理模型，同时满足准确性和吞吐量需求。
- **Command A Reasoning 驱动 North**：**Command A Reasoning** 是驱动 **North** 的核心生成模型，North 是 Cohere 的安全 Agentic AI 平台，支持自定义 AI Agent 和本地自动化。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1408009102625341461)** (4 条消息): 

> `Cohere Embed-v4 on Azure AI Foundry, Cohere Python Library Document Object` 


- **Cohere Embed-v4 输入类型映射**：一位成员正在 .NET 应用程序中使用部署在 **Azure AI Foundry** 上的 **Cohere Embed-v4**（通过 Azure AI Inference API），并寻求关于 **Microsoft 的 `EmbeddingInputType`** 如何映射到 **Cohere API** 文本嵌入的澄清。
   - 具体而言，由于 Cohere 的 `input_type` 参数中缺乏显式的文本选项，他们不确定 `EmbeddingInputType.Text` 是否应该映射到 Cohere API 中的 `search_document`。
- **Cohere Python 库的 Document 对象**：一位成员对 Cohere Python 库中的 **`Document` 对象**提出疑问，其中 `data` 字段预期为一个字典 (`typing.Dict[str, typing.Optional[typing.Any]]`)。
   - 他们指出 Tool Use 快速入门示例在该字段中使用了一个字符串（`json.dumps` 调用的输出），并想知道 Python 绑定是否正确处理了这种情况，参考文档为 [Tool Use 快速入门文档](https://docs.cohere.com/v2/docs/tool-use-quickstart)。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1407811130512113815)** (7 条消息): 

> `MLE Research, Independent Interpretability Research, AI Innovation and Value Creation, Enterprise Workflows` 


- **MLE 寻求研究团队联系**：一位拥有 **MLE** 经验的计算机科学硕士毕业生，正寻求与研究团队或组织建立联系。
   - 该成员表达了合作并为研究工作做出贡献的兴趣。
- **可解释性研究员渴望合作**：一位在印度班加罗尔、拥有 **8 年** 应用 ML 经验的独立可解释性研究员，正在转向 AI 研究，重点关注机械可解释性 (Mechanistic Interpretability)。
   - 该研究员对模型评估、去偏见和 RL 表现出兴趣，寻求在可解释性相关话题上的合作与讨论。
- **执行顾问连接 AI 创新与价值**：一位拥有 **25 年以上** 经验的独立顾问和执行顾问加入了社区，擅长将技术和 AI 创新与价值创造联系起来。
   - 凭借在埃森哲 (Accenture)、IBM 和德勤 (Deloitte) 等公司的经验，他们现在帮助客户从 AI 中创造可持续的、组织范围内的价值，公司网站为 [Mantha Advisory](https://www.manthaadvisory.com/own)。
- **CTO 探索 Cohere 以优化产品**：一位拥有 **25 年以上** 经验的 CTO 最近发现了 Cohere，并有兴趣探索其在改进产品方面的能力。
   - 他们关注数据质量、规模、性能、工作流、数据完整性和多语言支持，并热衷于向社区学习。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1407802615718482010)** (12 messages🔥): 

> `C# client library, MCP server's instructions field, MCP servers, generate_test_prompt.md, GitHub` 


- **MCP 客户端忽略 Instructions 字段**：成员们在使用 **MCP 客户端**（尤其是 **Claude**）时遇到问题，**instructions 字段**似乎被忽略了，而更倾向于使用 **tool descriptions**（工具描述）。
   - 一位成员建议，*添加指令、上下文，然后重复指令会产生更好的效果，但对于集成到 API 的工具来说，这是不可能的*。
- **MCP Server 选项评估**：一位成员询问开发者们正在使用哪些 **MCP servers**，以及哪些工具在这些服务器中效率更高。
   - 另一位成员强调了 **GitHub** 用于版本控制、**Python** 配合 **FastAPI** 进行后端开发，以及 **PyTorch** 用于机器学习的实用性。
- **让 Agent 遵循指令**：一位用户询问如何让 Agent 遵循特定的 **generate_test_prompt.md** 文件，并表达了对 Agent 在开启新对话时未遵循项目设计模式的沮丧。
   - 他们在消息中附带了一张 [截图](https://cdn.discordapp.com/attachments/1312302100125843479/1408171379236409354/Screenshot_3.png?ex=68a8c54b&is=68a773cb&hm=1fbd862963889b97ef764b1599c2960340dab7ce357b3717f577e2b1491ffdc2)。
- **MCP Server 解析优先处理工具描述**：一位成员指出，**MCP server** 内部的解析逻辑可以结构化为在 **instructions 字段**之前处理 **tool descriptions**。
   - 建议*审查服务器文档、检查客户端配置、分析服务端逻辑*并*进行受控实验*。
- **提及的指令遵循模型**：成员们讨论了哪些模型能够遵循指令并生成结构化输出，推荐了 **Mistral-7B-Instruct**、**DeepSeek-Coder** 和 **Phi-3**。
   - 他们还提到了 **OpenHermes-2.5-Mistral-7B**、**WizardLM-2** 和 **Gorilla-LLM** 作为专门针对 Function Calling 的模型。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1407927339345772656)** (10 messages🔥): 

> `Web-curl, MCP-Boss, MCP Explained Video, SWAG-MCP, MCP Routing` 


- ****Web-curl** 为 LLM Agent 赋能 Web 与 API 交互**：一位成员介绍了 **Web-curl**，这是一个使用 Node.js 和 TypeScript 构建的开源 **MCP server**，使 LLM Agent 能够以结构化的方式获取、探索并与 Web 及 API 交互，完整代码可在 [GitHub](https://github.com/rayss868/MCP-Web-Curl) 获取。
- ****MCP Boss** 集中化管理 MCP 服务的密钥**：一位成员构建了 **MCP Boss** 来集中管理密钥，提供单一 URL 来网关化所有服务，具有多用户认证和通过 OAuth2.1 或静态 HTTP 请求头进行 MCP 授权等功能 ([mcp-boss.com](https://mcp-boss.com/))。
- **视频揭秘 MCP**：一位成员发布了名为《MCP Explained: The Ultimate Deep Dive》的视频，[可在 YouTube 观看](https://youtu.be/xPq53oQi2tY)，并邀请大家就 Elicitation、roots 和 sampling 等客户端能力进行反馈和讨论。
- ****SWAG-MCP** 为可流式传输的 HTTP MCP server 生成反向代理配置**：一位成员分享了 **SWAG-MCP**，这是一个旨在为 SWAG 生成反向代理配置的 MCP server，支持自托管服务和可流式传输的 HTTP MCP server ([github.com/jmagar/swag-mcp](https://github.com/jmagar/swag-mcp))。
- ****MCP Gateway** 使用 AI 路由请求**：一位成员开发了一个带有 **AI 驱动路由**功能的轻量级网关，以解决 Agent 需要知道哪个特定服务器拥有正确工具的问题，代码可在 [GitHub](https://github.com/oliverye7/mcp-gateway) 获取。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1408147314702286910)** (2 messages): 

> `Modverse #50, Custom Server Tag` 


- **Modular 发布 Modverse #50**：Modular 发布了 [Modverse #50](https://www.modular.com/blog/modverse-50)，介绍了多位成员。
   - 公告还提到他们现在拥有了自定义服务器标签。
- **自定义服务器标签上线**：Modular 团队宣布自定义服务器标签上线，并在附件图片中展示。
   - 链接的图片 ([Screenshot_2025-08-21_at_5.22.15_PM.png](https://cdn.discordapp.com/attachments/1098713601386233997/1408199603861323878/Screenshot_2025-08-21_at_5.22.15_PM.png?ex=68a8df94&is=68a78e14&hm=2991584cc0b81449dbc278d1d8302e55aabc54c58a6620e7041deb9dbd20e951&)) 显示了新标签。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1407812660845871204)** (10 messages🔥): 

> `kgen and pop documentation, MLIR dialects, pop.union alignment bug, Github issue 5202` 


- **kgen 和 pop 的文档非常稀缺**：一名成员询问关于 **kgen** 和 **pop** 的文档，特别是操作和参数，但另一名成员表示 *目前没有关于内部 MLIR dialects 的全面文档*。
   - 分享了 GitHub 上的 [pop_dialect.md](https://github.com/modular/modular/blob/main/mojo/stdlib/docs/internal/pop_dialect.md) 链接，并澄清这些是 stdlib 与 compiler 之间约定的一部分，*因此在 stdlib 之外使用它们需自行承担风险*。
- **怀疑 pop.union 存在对齐 Bug**：一名成员询问了 **pop.union** 中元素的对齐问题，指出在使用 `sizeof` 时出现了意料之外的大小。
   - 他们分享的代码显示 `union_type_simple_8_bit_stdlib` 的大小为 **16 bytes**，而 `union_type_simple_8_bit` 和 `union_type_simple_multi_bit` 的大小均为 **8 bytes**，另一名成员建议 *对齐问题可能是一个 bug*。
- **已创建 Issue 以调查对齐 Bug**：一名成员在 GitHub 上创建了 [issue 5202](https://github.com/modular/modular/issues/5202)，以调查 **pop.union** 中疑似存在的对齐 bug。
   - 该成员指出他们不确定这是操作失误还是 bug，同时也观察到 **pop.union** 似乎没有在任何地方被使用。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1407837356937187378)** (7 messages): 

> `TextGenerationPipeline 'execute' method, Custom inference loops for retrieving logits, Language allocators and OOM handling` 


- **TextGenerationPipeline 的 `execute` 方法出现**：一名成员正在寻找 `TextGenerationPipeline` 上的 `execute` 方法但未能找到。
   - 另一名成员指向了 [Modular 仓库中的相关代码行](https://github.com/modular/modular/blob/fe7ba5d5f8b08ccba7356c45b90afcf495421469/max/pipelines/lib/pipeline.py#L977)，并建议检查 MAX 版本。
- **为 Logit 爱好者准备的自定义推理循环？**：一名成员报告在创建自定义推理循环时，难以在获取下一个 token 的同时检索 **logits**，感觉过程有些繁琐。
   - 该成员链接了一个 [Google Docs 文档](https://docs.google.com/document/d/1Hd6xZnf0bmg9SMQU1h10cd4Cwd9HDOPzPHZNqiPnrpg/edit?tab=t.0) 以提供背景信息，并确认该选项目前仍然可用，但其未来尚不确定。
- **内存分配器是必备项吗？**：一名成员建议，在将内存分配器集成到语言之前，可能需要健壮的分配器支持。
   - 他们认为大多数用户不想手动处理内存不足（**OOM**）错误。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1408123828470677533)** (2 messages): 

> `Enterprise document AI, vibe-llama` 


- **LlamaIndex 揭秘企业级文档 AI**：LlamaIndex 的产品副总裁将于 **PST 时间 9 月 30 日上午 9 点**分享关于[文档](https://t.co/x70xjEQaFs)解析、提取和索引的一年期企业级实践经验。
- **使用 vibe-llama 简化开发**：LlamaIndex 发布了 **vibe-llama**，这是一个命令行工具，可以自动为阁下喜爱的 coding agents 配置有关 **LlamaIndex framework** 和 **LlamaCloud** 的最新上下文和最佳实践。
   - 它还包含[更多信息](https://t.co/G1gINq9kge)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1407815234013364325)** (13 messages🔥): 

> `HuggingFace CrossEncoder 重复问题、Agent 创建项目、AI 安全调查` 


- ****CrossEncoder 类**：Core 与 Integrations 的对比**：一名成员询问了 `llama-index` 中重复的 **CrossEncoder 类**实现，具体位于 `.core` 和 `.integrations` 下（[代码链接](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/sbert_rerank.py)）。
   - 另一名成员澄清说，`.core` 中的实现是 v0.10.x 迁移的遗留物，应该被删除，建议改用 `llama_index.postprocessor.sbert_rerank` 并通过 `pip install llama-index-postprocessor-sbert-rerank` 安装。
- **寻求 **Agent 创建网关****：一名成员询问是否有现有的项目可以作为**网关**，将 **model、memory 和 tools** 结合在一起，并暴露一个 **OpenAI 兼容端点**。
   - 该成员想知道是否有现成的项目可以利用，以避免在他们的 Agent 探索中重复造轮子。
- ****AI 安全调查**：需要社区意见！**：一名成员分享了一个 [AI 安全调查链接](https://mukullight.pythonanywhere.com/form)，以收集社区对重要 **AI 安全问题**的看法。
   - 该成员请求大家填写表单，以帮助他们了解 **AI 安全社区**最感兴趣的内容，并请大家对可能的加载延迟保持耐心。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1407840535074439358)** (13 messages🔥): 

> `积分购买、工单问题、比赛操纵指控、每日免费积分、推荐积分` 


- **积分购买选项缺失**：成员们报告说购买额外积分的选项消失了，其中一人指出他们只能看到*升级包（upgrade package）*选项。
   - 另一名成员确认该选项*目前已下线*。
- **未解决的支持工单困扰用户**：一名用户报告在一个任务中遇到问题并创建了工单 **#1318**，但尚未收到回复或获得查看工单的权限。
   - 他们请求团队协助，并标记了一名特定成员。
- **比赛获胜者引发操纵指控**：一名用户指称比赛的第二名*不配获胜*，并声称比赛*似乎被操纵了*。
   - 目前没有提供进一步的证据或细节来支持这一说法。
- **每日免费积分已停止？**：一名在一个月后返回 Manus 的用户注意到，他们没有收到通常的 **300 每日免费积分**。
   - 他们询问 Manus 是否已经停止提供这些积分。
- **推荐积分代码难题**：一名用户询问如何领取推荐积分，提到系统要求输入代码。
   - 该用户表示他们不知道在哪里可以找到所需的代码。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1407818167493066922)** (7 messages): 

> `Overworld 常量折叠、View(const) 重构、UPat cvar 与 UPat.const_like 重新定义、RANGEIFY=1 的影响、base 移除` 


- **探索 Overworld 常量折叠策略**：一名成员正在探索 Overworld 常量折叠，可能涉及 **view(const) 重构**，并提议重新定义 `UPat.cvar` 和 `UPat.const_like` 以匹配 `CONST` 和 `VIEW(CONST)`。
   - 目标是折叠像 `x * 0` 这样的表达式，但有人担心符号计算中可能出现的有效性问题和 `.base` 扩散问题，如[此 Discord 线程](https://discord.com/channels/1068976834382925865/1255400554012741683/1407782654958506004)所述。
- **替代方案：ALU View Pushing**：有人建议了一种替代方案，模仿 **S-Lykles 的方法**，即在 kernelize 中添加一个 upat，将 view 直接推送到 **ALU** 上。
   - 这种方法配合 `x * 0` 的特殊规则（理由是 `* 0` 在计算上无关紧要），将允许未经修改的符号匹配。
- **主张移除 base**：一名成员强烈建议不要采用提议的方法，认为它“非常丑陋”，并主张**移除 `.base`**。
   - 讨论还质疑了在此背景下如何处理 **PAD** 操作。
- **RANGEIFY=1 作为潜在的简化方案**：有人建议设置 **RANGEIFY=1** 可能会带来更简洁的实现。
   - 然而，项目目前处于旧引擎和 rangeify 并存的过渡阶段，处于一种悬而未决的状态。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1408057198164049941)** (3 条消息): 

> `GPT4ALL 企业版 vs 免费版，LocalDocs 的模型选择` 


- **GPT4ALL 免费版用于私有模型使用**：一位用户咨询了关于公司希望**私密且安全地**使用其 **AI 模型**时，使用 **GPT4ALL** 的相关事宜。
   - 另一位成员澄清说，如果公司已经有了**准备就绪的 AI 模型**，那么**免费版**就足够了。
- **LocalDocs 的模型选择**：一位用户正在寻求模型推荐，以便利用 **GPT4All 的 LocalDocs 功能**，从数百篇 **PDF 格式的科学论文**中构建个人知识库。
   - 该用户说明其拥有配备 **24 GB VRAM** 的 **Nvidia RTX 5090** 和 **64 GB RAM**，并希望所选模型具备**推理能力**。