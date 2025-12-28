---
companies:
- alibaba
- arena
- runway
- nvidia
- togethercompute
- ollama
date: '2025-10-14T05:44:39.731046Z'
description: '**阿里巴巴**发布了 4B 和 8B 尺寸的紧凑型稠密 **Qwen3-VL** 模型，并提供 FP8 选项。该模型支持高达 100
  万（1M）的上下文长度和开放词汇检测，性能可媲美 **Qwen2.5-VL-72B** 等更大型的模型。


  生态系统支持包括 **MLX-VLM**、**LM Studio**、**vLLM**、**Kaggle models** 和 **Ollama Cloud**。在视频
  AI 领域，**Arena** 加入了在视频基准测试中处于领先地位的 **Sora 2** 模型，同时 **Higgsfield Enhancer** 提升了视频质量。**Runway**
  推出了针对创意任务的特定领域工作流应用。


  关于 **DiTs 表示自编码器 (RAE-DiT)** 的研究显示，扩散模型的性能得到了提升。在本地训练方面，**NVIDIA DGX Spark** 实现了强大的本地微调能力，而
  **Karpathy** 开发的 **Nanochat** 为训练和推理提供了一个极简的技术栈。**Together AI** 推出了 **ATLAS**，这是一种投机采样（speculative
  decoding）方法，在 **DeepSeek-V3.1** 上实现了高达 4 倍的推理加速。


  这些进展突显了在高效模型部署、视频 AI、本地微调和推理速度优化方面的重大突破。'
id: MjAyNS0x
models:
- qwen3-vl-4b
- qwen3-vl-8b
- qwen2.5-vl-72b
- deepseek-v3.1
people:
- karpathy
title: 今天没发生什么事。
topics:
- model-optimization
- fine-tuning
- inference-speed
- video-generation
- diffusion-models
- representation-learning
- local-ai
- speculative-decoding
- fp8-quantization
- context-windows
---

**平静的一天**

> 2025/10/13-2025/10/14 的 AI 新闻。我们为您检查了 12 个 subreddits、544 个 Twitter 账号和 23 个 Discord 社区（包含 197 个频道和 6882 条消息）。预计节省阅读时间（以 200wpm 计算）：510 分钟。我们的新网站现已上线，支持全元数据搜索，并以优美的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的详细新闻，并在 @smol_ai 上向我们提供反馈！

平静的一天。

---

# AI Twitter 回顾

**阿里巴巴 Qwen3‑VL Dense 模型 (4B/8B) 及快速生态支持**

- **Qwen3‑VL 4B/8B (Dense, Instruct + Thinking)**：阿里巴巴发布了紧凑型 Dense Qwen3‑VL 模型（4B 和 8B），每个版本均提供 Instruct 和 Thinking 变体，并包含用于高效部署的 FP8 选项。它们保留了 Qwen3‑VL 的全部功能，在 STEM、VQA/OCR、视频理解和 Agent 任务中表现强劲，通常优于 Gemini 2.5 Flash Lite 和 GPT‑5 Nano；在许多情况下，它们甚至可以与六个月前的 Qwen2.5‑VL‑72B 媲美。它们支持 256K 上下文（可扩展至 1M）以及“开放词汇（open vocabulary）”检测。Apache‑2.0 许可证。公告与 Cookbook：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1978150959621734624), [cookbooks](https://twitter.com/mervenoyann/status/1978082840337060118), [后续更新](https://twitter.com/mervenoyann/status/1978153606462550220)。
    
    生态系统：在 MLX‑VLM 和 LM Studio ([@Prince_Canuma](https://twitter.com/Prince_Canuma/status/1978164715848134699), [@lmstudio](https://twitter.com/lmstudio/status/1978205419802616188))、vLLM ([@rogerw0108](https://twitter.com/rogerw0108/status/1978158856611024913))、Kaggle 模型 ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1978290751436943415)) 以及针对 235B 变体的 Ollama Cloud ([@ollama](https://twitter.com/ollama/status/1978225292784062817), [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1978288558587674672)) 中获得首日支持。早期用户强调了其速度和结构化 JSON 输出的质量 ([@andrejusb](https://twitter.com/andrejusb/status/1978076341158244835), [@simonw](https://twitter.com/simonw/status/1978151711987372227))。
    

**视频模型与创意工具**

- **Arena 加入 Sora 2**：Sora 2 Pro 与 Veo 3 变体在 Video Arena 中并列第一；Sora 2 排名第三，并因同步音频而受到关注。文本转视频（text‑to‑video）领域的竞争正在加速 ([@arena](https://twitter.com/arena/status/1978149396996051007))。在实际应用中：**Higgsfield Enhancer** 消除了 Sora 风格的闪烁，并推出了 “Sora 2 MAX” 放大器 ([@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1978231305394663506))。
- **Runway Apps**：Runway 推出了 “Apps”，即特定领域的流水线（产品重拍、图像风格重塑等），正在 Web 和 iOS 端推出，强调可重用的专业管线 ([@runwayml](https://twitter.com/runwayml/status/1978094115142225968), [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1978109936149094800))。
- **研究：用于 DiTs 的表示自编码器 (Representation Autoencoders)**：RAE‑DiT 使用预训练的表示编码器（DINO, SigLIP, MAE）加训练好的解码器取代了 VAEs，在 ImageNet 上实现了 FID 1.51 @256（无引导）和 1.13 @256/512（有引导）。这突显了在 Diffusion 流水线中将表示学习与重建解耦的趋势 ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1978053094769615296), [评论](https://twitter.com/sedielem/status/1978143596701249733))。

**本地训练与推理：DGX Spark、Nanochat 及推理预测**

- **NVIDIA DGX Spark，桌面级微调**：早期用户报告 DGX Spark 可以轻松在本地运行强大的 LM（例如 Qwen3 Coder），并发布了 llama.cpp 性能数据以及来自学术实验室的公开报告。普遍观点：随着本地算力的成熟，越来越多的开发者在家里/办公室进行微调 ([@gneubig](https://twitter.com/gneubig/status/1978067258506187238), [@ggerganov](https://twitter.com/ggerganov/status/1978106631884828843), [@kchonyc](https://twitter.com/kchonyc/status/1978156587320803734), [@gdb](https://twitter.com/gdb/status/1978273142695977391))。
- **Nanochat (Karpathy)**：一个极简的端到端技术栈（约 8K 行代码），涵盖 pretrain → mid‑train → SFT → RL → inference 以及类 ChatGPT 的 UI；一个 560M 的模型在 8×H100 上训练约需 4 小时。社区小组、Colabs 和 SkyPilot 模板在一天内相继出现；团队正在扩展训练方案（recipes）并探索最佳的 SFT/RL 比例 ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1978144157970661495), [community](https://twitter.com/ben_burtenshaw/status/1978062142709326053), [SkyPilot](https://twitter.com/skypilot_org/status/1978273387903410412))。
- **大规模 Speculative decoding**：Together AI 推出了 ATLAS，这是一种学习型 speculator，其推理速度比基准快 4 倍（比其 Turbo speculator 快约 2 倍），在 DeepSeek‑V3.1 上达到了 500 TPS ([@togethercompute](https://twitter.com/togethercompute/status/1978210662095475097))。
- **推理中的内存与计算权衡**：基于对 Qwen3 的 1,700 次实验（0.6B–32B，4/8/16‑bit，token 预算，Maj@K，KV eviction/quantization），“最优”内存分配在“8‑bit 4B 有效尺寸”附近发生逆转。对于数学任务，避免使用 4‑bit；对于大型模型，更倾向于高精度和更长的生成长度；当达到 ≥8‑bit 4B 时，Maj@K 会有所帮助；KV eviction 与 quantization 的选择取决于规模 ([@DimitrisPapail](https://twitter.com/DimitrisPapail/status/1978108550854382052))。
- **更低成本的 RL 训练**：QeRL (NVLabs) 结合了 NVFP4 quantization + LoRA，实现了在单张 H100 80GB 上对 32B LLM 进行 RL 训练；代码和论文已发布 ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1978046212621373719), [repo](https://twitter.com/yukangchen_/status/1978146373745639894))。
- **二阶优化**：新的全二阶优化器在 LLM 训练实验中报告称，迭代次数比 SOAP 减少了约 5 倍，比 Muon 减少了约 15 倍 ([@ShamKakade6](https://twitter.com/ShamKakade6/status/1978147672105353543))。
- 额外福利：Python 3.14 将允许禁用 GIL，从而实现真正的多线程加速；uv 已经支持该功能 ([@_avichawla](https://twitter.com/_avichawla/status/1977985594103140710))。

**Agent、工具使用与 RL**

- **Claude Code 与子 Agent 编排**：多份报告显示，编排器 + 专门的子 Agent（代码编写者、搜索者、验证者）显著提升了规划和代码库任务的效率，表现优于单体式“深度研究” Agent。Anthropic 正在将 Claude 更深入地推向 Salesforce Agentforce、Slack，并在 Salesforce 工程部门全面推广 Claude Code ([@omarsar0](https://twitter.com/omarsar0/status/1978235329237668214), [@AnthropicAI](https://twitter.com/AnthropicAI/status/1978125047270154567))。Claude 应用在与 Gmail/Calendar 的集成中也表现出显著的深度 ([@emollick](https://twitter.com/emollick/status/1978101986357662156))。
- **为什么 RL 对 Agent 推理有效**：总结：真实、多样的数据 + 务实的 RL 调整（如 GRPO‑TCR）优于花哨的算法和规模；配合正确的方案（recipe），小模型 (4B) 在 AIME25 和 GPQA‑D 上的表现可以超过 14B–32B；长 CoT 模型需要经过密集的工具使用微调才能成为有效的 Agent ([thread](https://twitter.com/omarsar0/status/1978112328974692692), [paper](https://twitter.com/omarsar0/status/1978112412743258361))。补充性安全工作：**WaltzRL** 将帮助性/安全性框架化为正和多 Agent 博弈，以在不损失能力的情况下减少过度拒绝 ([@jaseweston](https://twitter.com/jaseweston/status/1978185306999341256))。
- **Agent 的工程化落地**：来自 LangChain 的关于 Agent 身份验证/授权（跨 Auth Code/OBO/Client Credentials 的 OAuth2/OIDC）的实用文章 ([@LangChainAI](https://twitter.com/LangChainAI/status/1978121116867567644))，Agentic MCP 配置和 schema 规范 ([@tadasayy](https://twitter.com/tadasayy/status/1978170863192346660))，以及使用 LlamaIndex Workflows + Docker + Kafka 编排微服务 ([@llama_index](https://twitter.com/llama_index/status/1978137596900593667))。
- 相关内容：LRM 在被中断或面对动态上下文时可能非常脆弱（性能下降高达 60%），这突显了静态评估与现实世界评估之间的差距 ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1978044847216095361))。

**搜索、检索与数据工具**

- **OpenAI Search API 更新**：Chat Completions 中新增由 GPT‑5 驱动的网页搜索：gpt‑5‑search‑api 价格为 $10/1K 次调用（便宜 60%），包含域名过滤，并与新的 Responses 网页搜索行为保持一致 ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1978224165997195559), [早期曝光](https://twitter.com/testingcatalog/status/1978153397552374168))。
- **Perplexity 成为 Firefox 默认搜索引擎**：Perplexity 现已作为 Firefox 用户的内置默认搜索选项 ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1978114334741168298), [@AravSrinivas](https://twitter.com/AravSrinivas/status/1978122437427425481))。
- **复杂查询中复合检索优于简单检索器**：Weaviate 的 Query Agent “Search Mode” 在 BRIGHT（需要推理的 level‑3 检索）上表现优于混合搜索；他们还详细介绍了面向 SaaS 规模工作负载的多租户原语（每个租户一个分片、延迟加载、租户状态）([@CShorten30](https://twitter.com/CShorten30/status/1978107101936230745), [@weaviate_io](https://twitter.com/weaviate_io/status/1978112245436453044))。
- **大规模向量基础设施**：TurboPuffer 报告在 ANN v3 beta 上实现 100B 向量搜索 p99=200ms（未过滤，1024D，k=10，92% 召回率）([@turbopuffer](https://twitter.com/turbopuffer/status/1978173877571441135))。
- **OCR 与机器人数据集**：Nanonets 发布了新的 SoTA OCR 模型，支持 LaTeX、多语言、复杂表格（兼容 Transformers/vLLM）([@reach_vb](https://twitter.com/reach_vb/status/1978061301399052485))；LeRobot 添加了用于编辑机器人数据集的 CLI 工具（拆分/合并、添加/删除特征、删除片段）([@LeRobotHF](https://twitter.com/LeRobotHF/status/1978126569055887421))。

**政策、产品与平台笔记**

- **Together AI 的规模**：The Information 报道 Together AI 在夏季 ARR 翻倍至 3 亿美元；正扩展到为其自己的数据中心购买 GPU ([记者报道](https://twitter.com/steph_palazzolo/status/1978099327634473072))。
- **Anthropic + Salesforce**：Claude 现已成为 Agentforce 在受监管行业中的首选模型，深化了 Slack 集成，并在 Salesforce 工程部门中采用了 Claude Code ([@AnthropicAI](https://twitter.com/AnthropicAI/status/1978125047270154567))。
- **OpenAI 平台/个性化**：OpenAI 计划放宽 ChatGPT 限制，以便在需要时允许更多 “4o‑style” 的个性；12 月将为经过验证的成年人提供有年龄限制的成人内容 ([@sama](https://twitter.com/sama/status/1978129344598827128), [后续跟进](https://twitter.com/sama/status/1978143827144700214))。
- **Google AI Studio 更新**：新的主页和 “立即构建” 的消息传递 ([@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1978138461514166742), [@osanseviero](https://twitter.com/osanseviero/status/1978142950098903472))。
- **AI 系统安全**：Google 撰文阐述 Gemini 的纵深防御；关于 Agent authZ/authN 以及使用受信任监控器进行 AI 控制的更广泛讨论突显了实际生产中的考量 ([@googlepubpolicy](https://twitter.com/googlepubpolicy/status/1978163414498185578), [@jonasgeiping](https://twitter.com/jonasgeiping/status/1978182050730344862))。

**热门推文（按互动量排序）**

- **“只需在任何视频中输入 ‘添加一个女朋友’”**：来自 [@elonmusk](https://twitter.com/elonmusk/status/1977982448861381081) 的 Grok Imagine 预热。
- **OpenAI 产品方向**：个性设置回归 ChatGPT，12 月将在年龄限制后提供更广泛的成人选项 ([@sama](https://twitter.com/sama/status/1978129344598827128))。
- **Figure 的新网站**：对人形机器人品牌/设计更新的强烈兴趣 ([@adcock_brett](https://twitter.com/adcock_brett/status/1978124226742944193))。
- **Perplexity x Firefox**：Perplexity 成为 Firefox 默认搜索选项 ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1978114334741168298))。
- **Walmart 在 ChatGPT 中的即时结账**：ChatGPT 内部嵌入的商业流程引起关注 ([@bradlightcap](https://twitter.com/bradlightcap/status/1978116720171643127), [@gdb](https://twitter.com/gdb/status/1978123494870196228))。
- **Sora 2 闪烁修复**：Higgsfield Enhancer 消除了闪烁并增加了超分辨率变体 ([@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1978231305394663506))。
- **“范式转移” 到开源/本地训练**：小型/专业化开源模型以及像 DGX Spark 这样的桌面端计算激增 ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1978113358772449379))。

---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. 本地唯一 AI 所有权口号

- [**If it's not local, it's not yours.**](https://www.reddit.com/r/LocalLLaMA/comments/1o6ocfs/if_its_not_local_its_not_yours/) (热度: 1035): **迷因图推广了本地优先的 AI 立场：“如果它不是本地的，它就不属于你”，这借鉴了加密货币领域的托管格言，暗示“不是你的 VRAM，就不是你的模型”。从技术角度看，该帖子认为本地部署/单服务器 SLM 在隐私/合规性、可预测的延迟、离线可靠性以及免受供应商政策变更、API 停机或可能破坏工作流/撤销功能的弃用影响方面更具优势。** 评论者主张本地/本地部署（“本地部署的 SLM 在特定任务上可以发挥奇效”），并引用了 AI Dungeon 事件——OpenAI API 的政策变更导致功能退化——作为反对云端依赖的警示（“骗我一次……”）。关于硬件所有权（VRAM）是否等同于控制权，以及托管 LLM 的便利性和规模优势，存在着广泛辩论。
    - 几条评论主张为了控制权、延迟和隐私而使用本地部署的 SLM。通过 4-8 位量化（例如 Llama 3.1 8B 4-bit 约需 5–6 GB VRAM），7B–13B 模型可以使用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 或 [vLLM](https://github.com/vllm-project/vllm) 等运行时在消费级 GPU 上本地运行，提供低于 100ms 的 token 延迟，并消除供应商停机或政策变更的风险。这有利于特定任务的微调和确定性的吞吐量，而非追求大型托管 LLM 典型的峰值基准测试分数。
    - 强调的一个关键技术失效模式是与单一供应商 Web UI 的紧密耦合；使用兼容 OpenAI 的客户端（如 [LM Studio](https://lmstudio.ai/)）配合你的 API key，可以只需极少的代码更改即可切换端点（如 [OpenRouter](https://openrouter.ai/)、[Together](https://www.together.ai/)）。注意：不同供应商在 API 表面（OpenAI Chat Completions 与较新的 [Responses API](https://platform.openai.com/docs/api-reference/responses)）、工具（函数/工具调用）、速率限制和分词（tokenization）方面存在差异——因此抽象层应规范这些差异并保留本地备选方案。
    - 历史背景：OpenAI 对 AI Dungeon 的限制催生了像 **EleutherAI** 的 [GPT‑Neo/J/NeoX](https://www.eleuther.ai/) 这样的开源权重努力，随后被 **Meta** 发布的 LLaMA 加速；现代本地技术栈（如 [text-generation-webui](https://github.com/oobabooga/text-generation-webui)、llama.cpp、vLLM）使独立于供应商的工作流变得切实可行。账号封禁（如 Anthropic）进一步强化了设计本地优先流水线的必要性，将提示词/数据集/检查点（checkpoints）置于你的控制之下，并使用可更换的推理后端以避免硬性停机。

## 非技术性 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI ChatGPT 成人内容推出与个性化限制放宽（12 月推出）

- [**Updates for ChatGPT**](https://www.reddit.com/r/ChatGPT/comments/1o6jins/updates_for_chatgpt/) (热度: 3714): **OpenAI 表示，最初为了减轻心理健康风险而过度收紧了 ChatGPT 的安全过滤器，但由于安全措施和工具的改进，现在将放宽这些限制。几周内发布的新版 ChatGPT 将启用由用户控制的可选个性化设置，模拟用户喜欢的 [GPT‑4o](https://openai.com/index/hello-gpt-4o/) 特性（更具人性化的语气、大量使用表情符号、朋友般的行为）。在 12 月，随着更广泛的年龄限制/验证机制的引入，他们计划根据“像对待成年人一样对待成年用户”的原则，允许经过验证的成年人访问仅限成人的内容（包括情色文学）。** 评论者对这种响应性普遍持积极态度，并指出这种直接沟通非常罕见；一个技术问题询问这是否会影响或干扰模拟 4o 风格的社区/第三方项目（例如 “4o Revival”）。
    - 开发者询问 GPT-4o 的更新是否会干扰像 **4o Revival** 这样的第三方项目。技术风险集中在 API/模型漂移：更改审核政策、函数调用模式（schemas）或输出格式可能会破坏经过提示词微调或依赖解析器的流水线；缓解措施是固定特定版本的模型（例如 `gpt-4o-2024-xx-xx`），分阶段推出，并监控弃用情况。参见 OpenAI 的模型生命周期和弃用指南：[Models](https://platform.openai.com/docs/models) 和 [Deprecations](https://platform.openai.com/docs/deprecations)。

- 关于成人内容年龄门控的问题集中在是否需要**基于身份的 KYC (ID-based KYC)**，还是将付费订阅视为年龄信号。从技术上讲，身份验证（例如通过第三方提供商）提供了更强的保证，但摩擦力更大且存在隐私风险；支付方式是一个弱代信（如预付卡、家庭计划），并且在某些地区可能无法满足监管要求。保护隐私的选项包括平台或运营商的年龄证明以及**可验证凭证**（[W3C VC](https://www.w3.org/TR/vc-data-model-2.0/)），但其部署并非易事，且取决于司法管辖区。
- [**ChatGPT 中的成人模式！**](https://www.reddit.com/r/ChatGPT/comments/1o6rtwm/adult_mode_in_chatgpt/)（活跃度：1222）：**根据 CEO Sam Altman 在 X 上的发帖以及 Reuters 的报道（参见：https://www.reuters.com/business/openai-allow-mature-content-chatgpt-adult-verified-users-starting-december-2025-10-14/），OpenAI 将从 2025 年 12 月开始在 ChatGPT 中为经过验证的 18 岁以上用户引入设有年龄门控的“成人模式”，允许成人内容（包括色情文学）。这放宽了此前因心理健康担忧而制定的“相当严格”的安全政策；Altman 表示 OpenAI 拥有新的缓解工具，并将在“大多数情况下”放宽审核。与此同时，OpenAI 计划在未来几周内发布用户控制功能，以明确设置语气/个性（例如，更像人类、多表情符号或像朋友一样的回复），而 Meta 则宣布在 Instagram 及其 gen-AI 工具上为 18 岁以下用户提供灵感源自 PG-13 的过滤机制（来源同上）。** 一条热门评论断言，通用 LLM 在生成色情文学方面的表现将优于小众/专业化模型，这意味着鉴于目前的通用能力，领域专业化可能是不必要的。
    - 政策/功能变化：OpenAI 将从 12 月开始为经过验证的成年人增加年龄门控并允许色情内容，同时放宽此前因心理健康担忧而“相当严格”的安全过滤器。Altman 表示，他们现在拥有缓解风险的“新工具”，并将发布控制功能让用户决定聊天机器人的语气/个性（例如，更像人类、多表情符号、像朋友一样），这意味着针对成人与未成年人账户将有更细粒度的风格调节和策略门控。
    - 模型能力辩论：一位评论者认为，通用的前沿 LLM 在色情内容方面的表现可能优于小众微调模型，这表明广泛的预训练比受限于狭窄领域的专业数据集能产生更好的连贯性、指令遵循和风格适应。这暗示了专业化与通用语言/世界知识之间的权衡，而后者能提升开放式创作任务的输出质量。
- [**12 月 ChatGPT 将推出成人版**](https://www.reddit.com/r/singularity/comments/1o6jmv0/adult_version_in_chatgpt_on_december/)（活跃度：1811）：**一张截图声称 ChatGPT 的“成人版”定于 12 月推出，仅限“经过验证的成年人”访问，这暗示 OpenAI 可能会引入年龄/身份验证来限制 NSFW 或性内容功能。该帖子侧重于政策而非技术；除了 UI 文本中的年龄门控提示外，没有提供模型细节或实现细节。** 评论者强调了必须提交政府身份证件/护照才能访问的隐私担忧，并批评了日益增加的身份验证要求；其他人则在开关于成人角色扮演影响的玩笑。
    - 通过政府身份证件进行 KYC/年龄门控以访问 NSFW 内容增加了技术隐私风险：将真实身份与聊天日志绑定会增加去匿名化和法律风险（例如，服务器存储的对话记录被传票调取/证据开示）。评论者担心数据保留政策不明、跨账号关联，以及经过验证的身份如何与高度敏感的内容类别相关联，呼吁针对存储期限、加密和可审计性制定明确的政策。
    - 一些人建议使用本地 LLM 以避免服务器记录/KYC，并指向了 /r/LocalLLaMA (https://www.reddit.com/r/LocalLLaMA/)。通过 **llama.cpp** (https://github.com/ggerganov/llama.cpp) 或 **Ollama** (https://ollama.com/) 等工具运行 **Llama 3** 或 **Mistral** 等模型可以将提示词/补全内容保留在设备端；权衡之处包括质量低于前沿模型以及硬件限制，但隐私性更高，且没有中心化的保留/内容审核。
    - 存在对内容画像（content profiling）的担忧：“OpenAI 想知道你的爱好”意味着从聊天中推断私密偏好并将其与验证身份联系起来。提出的技术问题包括敏感类别数据是否被最小化或隔离、如何用于个性化或安全系统，以及用户是否可以退出或通过可验证的擦除来删除此类数据。

- [**Sam altman latest tweet 🥸**](https://www.reddit.com/r/ChatGPT/comments/1o6l446/sam_altman_latest_tweet/) (活跃度: 1200): **一张归属于 Sam Altman 的推文截图，评论者将其解读为 OpenAI 的内容政策（content policy）正在发生转变，转向允许针对已验证成年人的成人内容/情色内容，并将 AI 定位为心理健康/孤独感的支持工具。技术层面的利害关系在于审核规则（moderation rules）和安全过滤器（safety filters）（用户指出目前这些过滤器甚至会过度拦截良性的学术讨论），以及潜在的访问权限年龄/身份验证（例如支付/KYC）关卡。** 评论者认为当前的过滤器过于敏感，并要求将现有的付费账户作为足够的年龄证明进行成年人验证，而其他人则对 AI 能否有效解决心理健康问题的说法持怀疑态度；一位评论者仅对允许情色内容表示支持。
    - 对过于宽泛的审核（moderation）的担忧：一位评论者指出，良性的学术讨论仅因其提及人类互动就会被标记，建议将付费且经过银行账户验证的订阅作为放松过滤器的成年人信号。从技术上讲，这指向了集成 KYC/年龄证明（例如 [Stripe Identity](https://stripe.com/identity) 等支付提供商身份验证）和分层审核阈值以减少误报，同时采用更具上下文感知能力的分类器（区分真实的情色内容与学术提及），以及针对边缘情况的潜在人工介入（human-in-the-loop）审查。
    - 对“成人”层级作为收入来源的推测引发了实施细节的讨论：通过年龄限制访问进行政策细分、各地区的合规性（例如 GDPR 同意年龄/类似 COPPA 的规则）、地理围栏（geofencing），以及为已验证成年人设置独立的审核流水线或阈值。这增加了运营复杂性（按细分市场配置多个安全配置/模型），但如果配合稳健的证明和审计，可以减少对已验证用户的过度拦截。
- [**3.5**](https://www.reddit.com/r/singularity/comments/1o6s060/35/) (活跃度: 415): **引用 GPT-3.5 行为的非技术性迷因/截图；帖子标题（“3.5”）和语气（“说真的”）暗示了对 ChatGPT 3.5 愚蠢或错误回答的沮丧，突显了尽管从 GPT-2 时代的工具如 AI Dungeon 2 (2019) 以来取得了进步，但仍存在已知的可靠性限制（幻觉/虚构）。背景意义在于能力快速提升（GPT-2 → GPT-3.5）与用户在日常提示词中仍会遇到的 3.5 持久失败模式之间的对比。** 评论注意到自 GPT-2/AI Dungeon 2 以来的惊人进步，同时隐含地质疑 GPT-3.5 在实际决策中的可信度（例如，开玩笑说不养狗了）。
    - 一位评论者回忆起 AI Dungeon 2 (2019) 运行在 GPT-2 上，这标志着大型 Transformer 文本生成在交互式小说中的首次广泛应用部署之一。这为模型在经过指令微调（instruction tuning）/RLHF、更大的上下文窗口（context windows）以及改进的长程连贯性和安全性后，发展到今天的“3.5”级助手提供了基准。
    - 关于计算 “strawberry” 中 R 数量的提示词突显了一个已知的弱点：自回归 LLM（autoregressive LLMs）由于子词/BPE tokenization 以及缺乏算法计数能力，通常在精确的字符级任务上失败。准确性通常会随着显式的逐步推理、字符/字节级 tokenization 或卸载到确定性的字符串工具而提高，但即使在现代模型中，这种脆弱性可能依然存在。

### 2. Duplicate Reposts: Vintage TV/Music Clips (Elvis 1977; Mr Rogers 'Crashes Out')

- [**Elvis Presley's chaotic last show in Vegas, 1977**](https://www.reddit.com/r/aivideo/comments/1o6mahh/elvis_presleys_chaotic_last_show_in_vegas_1977/) (热度: 836): **该帖子分享了一个 [v.redd.it](http://v.redd.it/) 剪辑，据称展示了 Elvis Presley 1977 年在拉斯维加斯“混乱”的最后一场演出 (**`1977`**)，但媒体端点 ([v.redd.it/92gy1jkf64vf1](https://v.redd.it/92gy1jkf64vf1)) 在没有 Reddit 身份验证 (**`OAuth`**) 的情况下返回** `403 Forbidden`**，导致无法验证内容。热门评论强烈暗示该剪辑是 AI 生成的（deepfake/语音/CGI），指出它在快速滚动浏览时非常具有迷惑性，并提到了一个喜剧性的“屁中带灰”的视觉笑话——这表明它是合成视频和/或音频合成，而非档案素材。** 评论者强调了短视频 AI 媒体日益增长的真实感以及误导风险（“起初没意识到是 AI”），而线程的其他部分主要是幽默内容，几乎没有技术争论。
    - 一些评论者含蓄地注意到 AI 生成视频的真实感正在提高——其中一人表示起初没意识到这是 AI——突显了时间相干性（temporal coherence）和次级效果的改进。对可见“灰尘”和合理的物体运动（例如滑板车滚动）的提及，表明粒子系统和刚体动力学的模拟效果更好，使得在没有针对伪影的启发式方法或逐帧分析的情况下，进行日常检测变得更加困难。
- [**Mr Rogers Crashes Out**](https://www.reddit.com/r/aivideo/comments/1o6cbx0/mr_rogers_crashes_out/) (热度: 659): **该帖子分享了一个标题为“Mr Rogers Crashes Out”的 [v.redd.it](http://v.redd.it/) 短视频，似乎是 Fred Rogers 在拍摄期间摔倒的花絮；媒体托管在 [v.redd.it/g1ig74t962vf1](https://v.redd.it/g1ig74t962vf1)，在没有 Reddit 身份验证或开发者令牌的情况下返回** `HTTP 403 Forbidden`**。目前没有技术讨论——热门评论多为幽默反应（与其他温和人物的夸张对比、GIF 回复以及引用台词 *“Woah… keep rolling”*）。**

### 3. AI/Robotics Visual Demos and Posters (Gunkata meme; Qwen+Wan I2V; Humanoid lineup)

- [**Gunkata training for the Elderly**](https://www.reddit.com/r/aivideo/comments/1o6sz2q/gunkata_training_for_the_elderly/) (热度: 486): **该帖子链接到一个 Reddit 托管的视频，展示了为“老年人”改编的“gunkata”（在《撕裂的末日》中流行的程式化枪械动作模式；参见 [gun kata](https://en.wikipedia.org/wiki/Gun_kata)），但由于 [v.redd.it/59o9hmile5vf1](https://v.redd.it/59o9hmile5vf1) 上的 HTTP** `403 Forbidden`**，在未经身份验证的情况下无法访问媒体（参见 Reddit 的访问支持 [此处](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=21879292693140)）。热门评论强调心理沉着（*“保持内心平静”*）和逐步的射击对齐概念（例如 *“每一步设定一条线，每一条线终结一个威胁”*——作者身份存疑），暗示重点在于步法/线路管理而非可衡量的射击精度指标；线程内未提供定量数据、安全协议或课程细节。** 一条热门评论主张在美国养老院强制执行此类枪支培训；在可见的回复中，没有关于有效性、安全性或法律考虑的实质性辩论。
- [**Shooting Aliens - 100% Qwen Image Edit 2509 + NextScene LoRA + Wan 2.2 I2V**](https://www.reddit.com/r/StableDiffusion/comments/1o6m23n/shooting_aliens_100_qwen_image_edit_2509/) (热度: 605): **OP 概述了一个视频流水线（pipeline），结合了用于帧编辑的 Qwen Image Edit 2509 和用于场景间连续性的 [NextScene LoRA](https://huggingface.co/lovis93/next-scene-qwen-image-lora-2509)；由于 LoRA/流水线不兼容，他们无法通过 Nunchaku 运行此组合，但指出 Nunchaku 使其他生成任务变得“非常疯狂”地快。为了减轻 Qwen IE 2509 过度平滑的输出，他们使用 Flux Krea Nunchaku 进行快速的纹理 img2img 传递，然后通过 Wan 2.2 Image-to-Video 在** `1280×720` **分辨率下生成动作，使用 Topaz Video AI 进行上采样，并应用了新旧 Lightx2v LoRA 以及自定义角色 LoRA。** 热门评论强调了强大的时间一致性，并表示打算尝试 NextScene；有人询问硬件配置，但未提供具体规格。
    - 流水线/设置：帧是使用 **Qwen Image Edit 2509** + **NextScene LoRA** 生成的（链接：https://huggingface.co/lovis93/next-scene-qwen-image-lora-2509）。由于使用了 LoRA，Nunchaku 没有直接与 Qwen Image Edit 2509 配合使用；相反，**Flux Krea Nunchaku** 被用于快速的纹理导向 img2img 传递。最终动作通过 **Wan 2.2** Image-to-Video 在 `1280x720` 下完成，随后用 **Topaz Video AI** 上采样；同时应用了新旧 **Lightx2v** LoRA，以及一个用于 Wan 2.2 的自定义角色 LoRA。

- 质量/一致性观察：原始 Qwen Image Edit 2509 的输出被描述为过于“平滑/虚假”，通过 Flux Krea Nunchaku 进行增强纹理的 img2img 编辑可以缓解这一问题。**NextScene** 提高了场景间的连贯性，但可能会细微地改变面部；种子（seed）选择会影响稳定性。一位评论者询问了在不使用 LoRA 的情况下保持角色一致性的策略，并指出了 NextScene 轻微的面部偏移问题。
    - 性能/权衡：在设置好 **Nunchaku** 后，作者观察到生成速度明显加快，但由于工具限制，在使用 NextScene LoRA 时无法将其与 Qwen Image Edit 2509 直接结合。该工作流展示了当前的互操作性权衡：在不同模型（Qwen Image Edit 2509, Wan 2.2）中混合使用多个 LoRA（NextScene + Lightx2v + 自定义角色），以平衡速度、纹理真实感和时序一致性。
- [**最新人形机器人海报**](https://www.reddit.com/r/singularity/comments/1o6cffg/a_poster_of_the_latest_humanoids/) (热度: 1049)：**一张更新的海报汇总了积极开发双足人形机器人的公司/实验室，该名单经过** `~1 年` **的整理，并通过直接联系确认了其在双足能力方面的实质性工作（而非仅有手臂或轮式平台）。该图像充当了当前人形机器人领域的对比全景图，反映了人形机器人 R&D 领域极具成效的一年；高分辨率版本在评论区，共享预览图在此：https://i.redd.it/6xttcpfz62vf1.png。** 评论重点提到了较知名的参与者（例如一家名为 “Borg” 的公司），询问为什么意大利表现突出，并建议德国可以利用其汽车工业基础成为人形机器人领域的领导者。
    - 一位评论者认为 **德国** 正在错失战略机遇：即利用其现有的汽车制造基础设施（精密加工、供应链、QA、机电一体化人才）转向大规模人形机器人生产。该观点暗示，现有的用于电机、变速箱和组装的一级/二级供应商能力可以被重新利用，以加速人形机器人的开发，并降低相比于白手起家的初创公司的成本。
    - 另一个帖子尝试识别海报上的模型，特别是询问了 **Unitree G1**。这表明阵容中可能包括当代紧凑型人形机器人，如 [Unitree G1](https://www.unitree.com/g1)，并突显了人们对具体代表平台的兴趣，而非通用的“人形”标签。
- [**访问中国的西方高管回国后感到恐惧**](https://www.reddit.com/r/singularity/comments/1o6df3t/western_executives_who_visit_china_are_coming/) (热度: 781)：**该贴围绕《电讯报》(Telegraph) 的一篇付费文章展开，称访问中国的西方高管对中国快速推进的工厂自动化感到震惊。评论者提到了当地的激励措施，如在“机器换人/jiqi huanren”政策下，对工业机器人 capex 进行约 20% 的税收返还（[文章链接](https://www.telegraph.co.uk/business/2025/10/12/why-western-executives-visit-china-coming-back-terrified/)）。评论中报告的装机量数据显示，中国拥有** `~2M` **台工业机器人，而日本约为** `~0.4M` **，美国约为** `~0.4M`**；大多数机器人是通过传统的 CNC/PLC/示教器工作流编程的，而非自然语言接口，这暗示随着 LLM/NLP 控制技术的成熟，软件升级仍有巨大空间。这表明中国的优势目前在于 capex 规模和政策驱动的部署，未来可能通过加装更高级的 AI 接口获得进一步收益。** 评论辩论认为美国的工业政策过于关注振兴传统部门；即使有制造业税收抵免，短期内的宏观影响可能会滞后，因为机器人驱动的生产力提升需要多年的集成以及员工/生产线的重新调整。其他人指出，具备自然语言能力的机器人在今天占比很小，这构成了一个机遇，但在大规模部署之前，仍存在显著的软件和安全验证差距。
    - 中国地方政府正在通过“机器换人”政策补贴工厂自动化，**报销约 20% 的工业机器人 capex**。这缩短了自动化项目的 ROI/回本周期，推动了更高的机器人密度，并将 capex 从劳动力转向机器人。这种激励结构加速了改造和换线，提升了吞吐量和工艺能力。
    - 引用的装机量为：**中国约** `2,000,000` **台，日本约** `400,000` **台，美国约** `400,000` **台**。大多数机器人使用传统的 CNC/PLC 编程，而非 AI/LLM 接口，因此只有很小一部分支持自然语言任务。这为软件优先的升级（视觉、力控、LLM 规划）创造了空间，可以在不进行大规模硬件更换的情况下减少集成时间并扩展任务范围。

- 传闻性的购买迹象显示，中国品牌（MG、LDV、小鹏、杰酷、奇瑞、深蓝、极氪、比亚迪、零跑、吉利、长城）在美国以外地区广泛存在，这标志着经销商网络和产品多样性的迅速扩张。结合自动化驱动的成本压缩，这种广度可能会缩短上市时间并降低 BOM 成本，从而加剧电动汽车（EV）和紧凑型燃油车（ICE）细分市场的竞争。

---

# AI Discord 摘要回顾

> 由 gpt-5 生成的摘要之摘要之摘要
> 

**1. AI 硬件：定制芯片、GPU 和 Kernel 技巧**

- **OpenAI 在播客中谈论定制芯片**：[OpenAI Podcast: Designing Chips with Broadcom](https://www.youtube.com/watch?v=qqAbVTFnfk8) 邀请了 **Sam Altman**、**Greg Brockman** 以及 Broadcom 的 **Hock Tan** 和 **Charlie Kawwas** 讨论 OpenAI 设计自有芯片的举措，将硬件选择与前沿模型需求和全球供应限制相结合。他们概述了模型洞察如何驱动芯片决策，并提到正在进行的合作伙伴关系，以扩大 **AI 加速器**的能力和可用性。
    - 社区笔记强调了从模型需求到芯片架构的直接联系，指出了推动 **系统、编译器和 Kernel** 更紧密协同设计的趋势。一位成员将这种氛围总结为 *“硬件现在追随模型路线图”*，突显了向垂直整合 **AI 计算**的转变。
- **Intel 预告 2026 年推出的 Crescent Island**：[Intel to expand AI accelerator portfolio with new GPU](https://newsroom.intel.com/artificial-intelligence/intel-to-expand-ai-accelerator-portfolio-with-new-gpu) 预览了将于 2026 年下半年推出的 **Crescent Island**，配备 **160 GB LPDDR5X**，这意味着拥有数十个控制器和极宽的接口（约 **640-bit** 或更高）。路线图暗示了 **Xe3P** 切片的更改（趋向于 8 个子切片）以及移除固定功能块以优先提升 **AI 吞吐量**。
    - 工程师们将其解读为在推理密集型集群中追求更高 **内存带宽/GB** 和更好 **TCO** 的策略。一位评论者调侃道，Crescent Island 的目标是 *“喂饱野兽，而不仅仅是让它长大”*，指向了现代 **LLM 工作负载**中受内存限制的 Kernel。
- **Pallas MGPU 像专业人士一样重叠 NVLINK 通信**：一个新的 JAX 教程 [Pallas:MGPU collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html) 展示了对 **Pallas:MGPU matmul kernel** 进行微调，可以将其转变为 **all-gather 集合矩阵乘法**。该示例将 **NVLINK** 通信与本地计算重叠，演示了实际的计算/通信流水线化。
    - 从业者强调这种模式是多 GPU **prefill** 和 **KV sharding** 方案的模板，在这些方案中带宽至关重要。一份摘要称赞其为那些愿意调整 **collective kernels** 而非依赖默认设置的团队带来了 *“免费的重叠投资回报（ROI）”*。

**2. 开源训练工具和自定义设备**

- **MegaFold 优化 AF3 训练**：一个研究小组开源了 **MegaFold**，这是一个 **AlphaFold 3** 的训练平台，并在 HN 上分享了性能报告：[MegaFold training platform](https://news.ycombinator.com/item?id=45585528)。该帖子指出其速度慢于同等规模的 Transformer，并提出了通过 **自定义 Triton ops** 和 **系统级数据加载** 来减少峰值内存的优化方案。
    - 工程师们喜欢具体的性能分析（profiling）和可操作的修复方案，称赞 *“在痛点处使用自定义算子”* 是正确的方法。讨论集中在将 **Kernel** 和 **输入流水线** 移植到生产堆栈中，以从现有 GPU 中榨取更多 **吞吐量**。
- **TorchAX 在 PyTorch 中引入纯 Python 设备**：[google/torchax](https://github.com/google/torchax) 允许在 **PyTorch** 中使用纯 Python 自定义设备，包括一个 “jax” 设备垫片（shim）。这降低了实验替代后端和自定义设备语义的门槛，无需深入的 C++ 胶水代码。
    - 用户将 TorchAX 描述为 *“凡人也能用的设备原型设计工具”*，是测试 **执行模型** 和 **调度路径（dispatch paths）** 的快车道。其新颖之处在于采用 Python 优先的路径进行设备集成，同时保留了 PyTorch 在 **Kernel** 和 **autograd** 方面的易用性。
- **DeMO 优化器助力去中心化训练**：**DeMO** 优化器已发布约 9 个月：[bloc97/DeMo](https://github.com/bloc97/DeMo/)，并被 Psyche ([PsycheFoundation/psyche](https://github.com/PsycheFoundation/psyche)) 用于去中心化训练。讨论帖指出其正在积极开发并已在社区堆栈中进行实际部署。
    - 开发者赞扬了 DeMO 的稳定性，并称其为长周期训练工具箱中 *“一个可靠的调节旋钮”*。Psyche 代码库被推荐作为健壮的 **分布式训练** 模式的参考。

**3. 海量数据集与 Embedding 细微差别**

- **ArXiv 4.6TB 语料库登陆 HF**：**4.6TB** 的 [nick007x/arxiv-papers](https://huggingface.co/datasets/nick007x/arxiv-papers) 数据集发布，包含跨领域的全文和元数据。它针对下一代 LLM 的**学术推理**、**文献综述**和**科学知识挖掘**。
    - 研究人员将其标记为*“带有引用的预训练黄金数据”*，并讨论了 **Tokenization** 和**领域切分**。团队计划试点**检索增强（retrieval-augmented）**预训练方案，以测试科学问答方面的提升。
- **GitHub Code 2025 发布 100 万个仓库**：[nick007x/github-code-2025](https://huggingface.co/datasets/nick007x/github-code-2025) 汇编了前 **1,000,000** 个 GitHub 仓库（≥2 stars），用于代码生成和分析。讨论中提出了 **Licensing** 担忧，并建议为训练集中的许可子集设置过滤器。
    - 工程师们称其为*“我们想要的规模，以及我们预料到的局限”*。预计在大规模训练之前，会有关于**许可感知策展**、**去重（Dedup）**和**污染检查**的后续行动。
- **不同后端之间的 Embedding 偏移**：一篇报告 [Different backend, different vector](https://huggingface.co/datasets/John6666/forum2/blob/main/different_backend_different_vector.md) 记录了为什么相同模型（例如 **nomic-embed-text:v1.5**）在 **Ollama** 与 **HuggingFace** 上的 Embedding 存在差异。罪魁祸首是：每个运行时中不同的**预处理/后处理**和内存处理方式。
    - 从业者警告说，跨工具链的*“向量等效性无法保证”*，并建议固定 **Tokenizers**、**Normalizers** 和 **Post-norms**。共识是：如果你想要一致的 **ANN 召回率/精确率**，请复现整个 Pipeline。

**4. Agent 平台与框架**

- **Salesforce 以确定性方式编写 Agent 脚本**：Salesforce 在 [Introducing Hybrid Reasoning with Agent Script](https://developer.salesforce.com/blogs/2025/10/introducing-hybrid-reasoning-with-agent-script) 中引入了用于混合推理的 Prompt 嵌入式脚本。目标是通过模板化和显式行为实现更具确定性的 **Agent 控制**。
    - 工程师们欢迎更少的*“轮盘赌”*式运行，以及为生产流程带来更多的**可重复性**。该功能被视为迈向**可验证编排（verifiable orchestration）**而非纯 LLM 随机性的一步。
- **ReductoAI 融资 7500 万美元用于文档处理**：在实现 **6 倍**增长、处理量突破 **10 亿页**后，[ReductoAI 完成由 a16z 领投的 7500 万美元 B 轮融资](https://xcancel.com/aditabrm/status/1978129711898431935)。公司计划投资于**模型研发**、精度提升和**可定制流水线**。
    - 评论者将其视为对**文档密集型企业级 AI** 的认可，称其业务量指标是*“真实的使用情况，而非虚荣指标”*。预计将有更多针对合规性要求较高行业的**基准测试**和**垂直化工作流**。
- **CheshireCat 3.0 助力多模态 RAG**：[matteocacciola/cheshirecat-core](https://www.github.com/matteocacciola/cheshirecat-core) 发布了一个基于 **LangChain** + **Qdrant** 的框架，支持**多模态 RAG**、多租户聊天机器人和 **Agent 工具编排**，具有基于插件的可扩展性。文档位于 [Deepwiki](https://deepwiki.com/matteocacciola/cheshirecat-core)。
    - 开发者要求集成 **Neo4j** 以支持**图 RAG (Graph RAG)**，称该技术栈拥有*“企业级架构”*。早期采用者正在 POC 中测试**多模态流水线**和**租户隔离**。

**5. DGX Spark：带宽与价值的现实检验**

- **基准测试称 DGX Spark 出师不利 (DOA)**：[PCMag: Nvidia to Start Selling $3,999 DGX Spark](https://www.pcmag.com/news/nvidia-to-start-selling-3999-dgx-spark-mini-pc-this-week) 的报道引发了争论，早期测试显示在 gpt-oss-120B-fp4 上约为 **11 t/s**，而售价 4800 美元的 **M4 Max** MacBook Pro 则为 **66 t/s**。社区将这一差距归咎于 **LPDDR5X** 带宽（Spark 约为 **273 GB/s**，而 M4 Max 为 **546 GB/s**）。
    - 工程师们抨击它在纯推理方面*“出场即退场 (DOA)”*，尽管有些人看到了开发工作站的利基市场。许多人认为，当工作负载超出**统一内存 (Unified Memory)** 时，双 **RTX 5090** 在性价比上优于 Spark。
- **Unsloth 的评测澄清了技术栈**：一段 YouTube 评测 [DGX Spark review](https://youtu.be/Lqd2EuJwOuw) 指出，其 iGPU 大致相当于 **5070**，并配有 **128GB LPDDR5X**；同时澄清 **Unsloth** 是一个微调 + **RL** 库（而非量化库）。据报道，尽管性能评价褒贬不一，该设备很快就售罄了。
    - 从业者强调了对 Spark 级别设备在**训练与推理**方面的现实预期。一个结论是：对于繁重的 **LLM 工作负载**，*“请将其视为受带宽限制的开发节点，而非集群 GPU”*。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **在 Agentic AI 方面，Opera GX 比 Chrome 更受青睐**：成员们表达了对 [Opera GX](https://www.opera.com/gx) 优于 Chrome 的偏好，暗示其 **Agentic AI** 的实现是一个关键因素。
   - 讨论中并未提供有关使 Opera GX 更具吸引力的 AI 功能的具体细节。
- **Perplexity Pro 搜索限制下降**：用户报告称 **Perplexity Pro search** 现在的限制比以前更低，一位用户表示他们在 *10 分钟的工作中* 就达到了限制。
   - 这与网站声称的无限搜索相矛盾，引起了订阅者的担忧。
- **Gemini 使用用户数据训练，Pro 请求受限**：一位用户报告称 **Gemini** 会利用用户数据进行训练，访问对话内容，并且仅提供 **100 次 Gemini 2.5 Pro 请求**，而 **Perplexity** 则提供 **300+ 次**。
   - 另一位成员反驳称可以禁用训练并保留对话，但未引用任何额外证据或链接。
- **对 Perplexity 作为最安全浏览器的质疑**：在讨论 **Comet 浏览器** 的安全性后，一位成员表示，尽管 [Google 存在隐私担忧](https://en.wikipedia.org/wiki/Privacy_concerns_with_Google)，但 *在数据保留方面，他们更信任 Google 而非 Perplexity*。
   - 这一讨论凸显了在不同科技公司之间做出选择时，围绕数据隐私和信任的复杂考量。
- **Palantir 策划渗透美国政府？**：一位用户分享了一个 [Perplexity AI Collection](https://www.perplexity.ai/collections/palantir-coup-DheNJhRES1iNdXzkEgWvPQ)，暗示 **Palantir 可能接管美国政府**，并附带了一篇相关的 [LinkedIn 帖子](https://www.linkedin.com/posts/akhay-kumar_week-40-activity-7383378306315968512-Hp2_?utm_source=share&utm_medium=member_desktop&rcm=ACoAACqAHFkBiU84inu9idiNHTXvSsnGcjLgOrs)。
   - 讨论围绕 **Palantir 在美国政府部门日益增长的影响力** 及其合同、数据处理实践以及对政府运作的影响展开。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 开启自研芯片设计！**：OpenAI 正在利用构建前沿模型的经验，自主设计芯片以满足全球 AI 需求。正如 **Sam Altman** 与 Broadcom 的 **Hock Tan** 在 OpenAI 播客中所讨论的，[可在 Spotify 上收听](https://open.spotify.com/show/0zojMEDizKMh3aTxnGLENP)。
   - 播客探讨了芯片设计的细微差别，以及它如何直接影响 AI 模型的能力。他们还组建了一个 **福祉与 AI 专家委员会 (Expert Council on Well-Being and AI)**，更多详情请见 [OpenAI 博客文章](https://openai.com/index/expert-council-on-well-being-and-ai/)。
- **AI 伴侣引发情感依恋？**：成员们正在讨论用户对 AI 产生情感依赖的可能性，尤其是像 **GPT-4o** 这样具有个性驱动的模型，可能导致难以区分现实与 AI 交互。
   - 一位成员分享道，感觉失去 **GPT-4o** 的某种模式就像 *失去了一个你在情感上产生依恋的人。*
- **Kilocode-CLI 热度持续升温**：**kilocode-cli** 是一款使用 **TypeScript** 编写工具的新工具，作为 **Opencode** 的潜在替代方案引发了讨论。
   - 它因其多 Agent (multi-agent) 能力和编排子任务的功能而受到称赞，一位成员开玩笑说：*我也喜欢在每个《黑镜》场景中拥有不同的 AI 工具*。
- **PGVector vs ChromaDB 的辩论**：社区正在辩论 **LLM** 的最佳向量存储解决方案，一些人主张使用 **PGVector**，因为它与 **Supabase** 和 **AWS Aurora** 等现有数据库集成。
   - 一些人认为纯向量数据库是不必要的，而且 *最终你的应用无论如何都会需要一个真正的数据库*。
- **Token Cascade Model：感质 (Qualia) 的困境？**：一位成员介绍了 **Token Cascade Model**，认为 *感质 (Qualia) 的功能是 AI 改变状态的比较 Token*，并建议其他人 *对其进行数学建模*。
   - 该成员声称该模型正在 [OpenBuddy-AI chat](https://OpenBuddy.ai) 中进行演示，供公众测试和模块化使用。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cheetah 模型编程速度惊人**：**Cheetah 模型**因其在编程任务中极快的速度而受到赞誉，被认为是一个可能基于新 **Grok 编程模型**（特别是 [grok-code-fat 模型](https://x.com/Sake187237/status/1977848138741526963)）的“隐形模型”。
   - 用户建议的工作流是：使用 **GPT-5-high-fast** 进行规划，然后使用 **Cheetah** 执行，因为对其规划能力持保留意见。
- **Gemini 3.0 展示设计才华**：**Gemini 3.0** 的发布引发了关于 AI 创意和 UI 设计的讨论。
   - 一些成员认为 Gemini 展现了“创意”，这与当前被认为是“复制粘贴”的 AI 有所不同。
- **GPT-5 令人烦恼且“太笨”？**：一些用户发现 Plan 和 Agent 模式下的 **GPT-5 模型**“太笨了”，因为它会询问过多的确认问题。
   - 一位成员建议将其移除并替换为 **GLM**，这表现出对当前性能的沮丧。
- **Cursor 与 Linear 需要更多沟通**：用户报告了在与 **Linear** 集成时出现 **“Cursor 停止响应”** 的情况。
   - 一位用户澄清说，该问题仅限于集成设置，本地运行 Cursor 可以避开此问题。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **完整消息策略**：成员们澄清说，AI 策略作用于**完整消息**（whole messages）以避免混淆，并强调这种方法是实现正常功能的基础。
   - 这确保了 AI 处理完整的消息，在操作中保持上下文和连贯性。
- **上下文窗口保持 LM Studio 开启**：一位用户报告在使用 **GPT-OSS-20B** 进行 100-200 次迭代后，**LM Studio API** 会丢失上下文并输出乱码，建议将 LM Studio 内的 **context window** 更改为滚动设置。
   - 调整上下文窗口有助于管理模型保留的信息量，防止在长时间使用中失去连贯性。
- **关于确定性的辩论**：成员们辩论了从 LLM 获得**确定性输出**（deterministic outputs）的可能性，其中一人分享了自己的使命：证明将 temperature 设置为 0 并不能保证确定性。
   - 一位成员在 GPU 和 CPU 上同时运行 qwen3-4b-instruct-2507 q8，使用 temp0 和 ECC RAM，对同一提示词得到了完全相同的结果。
- **M 系列 Mac：出人意料地适合 LLM**：一位用户承认 **Apple 的统一内存架构**使其成为非常引人注目的 LLM 推理解决方案，声称其 **M2 Ultra** 在 **70b q4 模型**上达到 **12 t/s**，使用 **MLX** 时可达 **16 t/s**。
   - 这种性能可与 **4x4090** 配置的 **18 t/s** 媲美，而功耗仅为 **200W**，使其成为一种高能效的选择。
- **分享 SSD 寿命策略**：用户讨论了 SSD 的寿命，一位用户提到：“你真的应该避免填满 SSD。它们需要空间来从坏死/老化的单元中移动数据。”，建议保持在 **80%** 容量以下。
   - 此外还提到，读取不会损坏 SSD，但写入会，一位用户承认他们的 SSD 健康状况可能受到过快下载/删除过多模型/游戏的影响，加之那是一个“劣质廉价 SSD”。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DGX Spark iGPU 表现强劲**：新的 **DGX Spark** 评测指出，其 iGPU 拥有 **5070** 级别的性能，并配备 **128GB LPDDR5X**。
   - YouTube 评测[链接在此](https://youtu.be/Lqd2EuJwOuw?si=gutAZkj8EXEUrCqN)澄清了 Unsloth 是一个*微调和强化学习库*。
- **Kimi K2 在 Groq 实现下表现出色**：一位成员分享说 **Kimi K2**（特别是修复了 Groq 实现后）是他们最喜欢的模型之一，在 Groq 上达到了约 **500 TPS**。
   - 他们将 **Kimi K2** 用于创意编程项目，例如*编写一个带有屁声的网页*，并称赞了它的创造力。
- **开发者在服务器需求上倾向于选择 Ubuntu**：成员们推荐 [Ubuntu](https://ubuntu.com/) 作为开发服务器 Linux 发行版的*稳妥选择*，主要是出于对 **NVIDIA** 显卡驱动兼容性的考虑。
   - 这种考虑对于避免驱动问题至关重要。
- **模型在多模态问题上表现挣扎**：成员们对获取多模态问题准确答案的困难表示担忧：虽然文本模态表现良好，但**图像或音频输入**经常导致错误的响应。
   - 例如，模型难以正确描述图像，比如将*日落*误认为*地下室的椅子*。
- **Python 开发者使用 TTS 构建 AI 播客生成器**：一位成员创建了一个 **Python** 程序，连接到 **Ollama** 以生成带有 **TTS** 的 **AI 播客**。
   - 源代码可在 [GitHub](https://github.com/Laszlobeer/AI-podcast) 上获取，并附有[音频示例](https://cdn.discordapp.com/attachments/1179779344894263297/1427633017798791239/Episode-one-of-AI-Podcast-20251014-122402.mp3?ex=68f03b1b&is=68eee99b&hm=87394d69ca8c44736eed48e2ad1bb4629ca838a67743d2a8f8012beba81dbccf&)。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 机器人首次亮相**：社区成员 'no coder' 正在寻找测试者和合作者，共同构建一个使用 **OpenRouter** 的机器人。
   - 机器人开发者正积极征求 **OpenRouter** 社区的反馈，以完善和改进他们的机器人。
- **Gemini 在 Google Play 方面表现不佳**：成员们注意到 Google 的 **Gemini** 经常难以理解复杂的 **Android Play Store** 发布流程。
   - 在 **general** 频道中，一位成员分享了一个*趣闻*：在导航 **Google Play Store** 流程时遇到了困难。
- **Ling-1t 模型遭遇灾难性崩溃**：据报道，`inclusionai/ling-1t` 模型严重损坏，至少在 Chutes 的实现中是这样，导致在几千个 token 后输出乱码。
   - 一位成员提到正在寻找 **K2** 的更好替代方案。
- **DeepSeek 耗尽免费额度**：用户讨论了每日请求限制，指出如果没有 **$10 余额**，限制为 **50 次请求**；如果余额超过 $10，限制为 **1000 次请求**。
   - 一位用户发现单次使用 **DeepSeek 3.1** 就消耗了大量的免费请求额度。
- **Chutes 提供商刷差评事件**：一位成员链接到了一个 [Reddit 帖子](https://www.reddit.com/r/SillyTavernAI/comments/1o5s3ys/chtes_provider_is_using_bts_to_downvote_posts/)，指责 **Chutes** 提供商使用僵尸网络给帖子投反对票，引发了讨论。
   - 另一位成员澄清说，这只是因为*他们没有明确的隐私政策，所以 OR 将其设为默认设置*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Teacher Forcing 令 Fine-Tuning 爱好者受挫**：用户报告称，尽管 loss 较低，但他们的模型仍会出现文本缺失或合并的情况；这归咎于提前停止、生成过短、tokenization 差异和 **teacher forcing**，以及符号和空格的 tokenization 与归一化差异。
   - 一份[详细分析](https://cdn.discordapp.com/attachments/1427486612639453204/1427626289682059404/teacher_forcing_issue_2.md?ex=68f034d7&is=68eee357&hm=dbdcf79128dc5fde8b58a5a4013fef3d466a590841b5a027a282cb7635699877&)被分享出来，将该问题归因于解码器丢弃字符。
- **数据集大作登场：ArXiv 和 GitHub**：一个 **4.6TB** 的 [ArXiv Papers 数据集](https://huggingface.co/datasets/nick007x/arxiv-papers)发布，包含所有领域的论文及元数据，旨在用于训练模型的**学术推理、文献综述和科学知识挖掘**能力。
   - [GitHub Code 2025 数据集](https://huggingface.co/datasets/nick007x/github-code-2025)也已发布，用于代码生成和分析任务，包含 **GitHub 上 2 星以上的 Top 100 万个仓库**，但其中包含的仓库引发了许可（licensing）方面的担忧。
- **Karpathy 课程开启知识创作**：Andrej Karpathy 发布了一门关于构建**全栈 LLM** 的课程，因此有成员计划跟随该课程材料并发布指南以帮助学生，并邀请其他人加入 [nanochat-students 组织](https://huggingface.co/nanochat-students)。
   - 如果你正在学习该课程，可以加入 [huggingface.co/nanochat-students](https://huggingface.co/nanochat-students) 组织。
- **Civitai 内容混乱引发社区关注**：用户报告称 **Civitai** 出现了大规模内容移除，聊天群组和 Reddit 上也充满了不满情绪，引发了关于移除原因的讨论：内部冲突、支付攻击以及可能受到的极端组织针对。
   - 导致用户[迁移到其他平台](https://www.reddit.com/r/comfyui/comments/1kvkr14/where_did_lora_creators_move_after_civitais_new/)。
- **Ollama 令人困惑：Embedding 向量与 HF 存在差异**：一位用户发现 **Ollama** 和 **HuggingFace** 为同一模型（**nomic-embed-text:v1.5**）生成的 Embedding 向量不同。
   - 差异主要源于不同后端（backends）之间不同的**预处理和后处理阶段**以及内存利用率，如[这篇博客文章](https://huggingface.co/datasets/John6666/forum2/blob/main/different_backend_different_vector.md)所述。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Pallas MGPU 内核优化 NVLINK**：[docs.jax.dev](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html) 上的一篇关于使用 **Pallas:MGPU** 改进 GPU 计算与通信重叠（overlap）的新教程显示，对 **Pallas:MGPU matmul 内核** 进行微调，可以使其成为 **all-gather collective matmul**。
   - 根据一条推文，这些微调实现了 **NVLINK 通信** 与本地计算的重叠。
- **Intel Crescent Island 即将推出？**：根据 [Intel 新闻稿](https://newsroom.intel.com/artificial-intelligence/intel-to-expand-ai-accelerator-portfolio-with-new-gpu)，Intel 计划在 2026 年下半年推出 **Crescent Island**，作为一款新型 GPU 产品，它将配备 **160 GB 的 LPDDR5X** 显存。
   - 概念渲染图显示 **Crescent Island** 包含数十个 **LPDDR5X 控制器**，这意味着其显存接口位宽可能达到 **640-bit** 或其两倍；此外，**Xe3P** 架构可能会增加到 8 个子切片（subslice），或者涉及对固定功能、光线追踪和编解码流水线更彻底的移除。
- **MI300x 用户提交跑分数据**：一名用户在 `amd-all2all` 排行榜上以提交 ID `63561` 获得 **第一名**，在 **MI300x8** 上跑出了 **216 µs** 的成绩；此外，`amd-gemm-rs` 排行榜也收到了多次提交，耗时在 **533 µs** 到 **572 µs** 之间。
   - 此外，在 **MI300x8** 上向 `amd-ag-gemm` 排行榜提交的几次数据也获得了成功，耗时从 **384 µs** 到 **1201 µs** 不等；`amd-all2all` 排行榜上还记录了其他成功的提交，时间分别为 **339 µs**、**368 µs** 和 **3.45 ms**。
- **多 GPU 系统成为 HPC 关注焦点**：成员们正在研究 **多 GPU 系统** 在 **高性能计算 (HPC)** 中的普及程度，以及与 **多 GPU HPC 系统** 内部 **数据移动** 相关的研究机会，特别是针对 **延迟** 和 **带宽**。
   - 作为回应，一名成员发布了一篇 [研究论文](https://arxiv.org/abs/2509.21527)，探讨了与多 GPU 设置相关的架构和性能指标，并进一步确认许多研究人员正积极探索 **多 GPU HPC 系统** 中 **数据传输** 的 **延迟** 和 **带宽** 挑战。
- **MegaFold 训练平台发布！**：一个研究小组开源了 **MegaFold**，这是一个针对 **AlphaFold 3 (AF-3)** 的训练平台。他们指出该平台与同等规模的 Transformer 相比速度较慢，并为此撰写了一篇 [博客文章](https://news.ycombinator.com/item?id=45585528)。
   - 他们的分析识别了性能和显存瓶颈，并提出了诸如 **Triton 自定义算子** 和 **系统数据加载** 等优化方案，以提升性能并降低峰值显存占用。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Veo 模型标注细节曝光**：成员们探讨了 **Google** 如何为 **Veo 模型** 标注视频，重点关注音频到时间同步帧、时间轴 JSON 映射、元数据和视频分辨率，并参考了 [Google 的文档](https://cloud.google.com/vertex-ai/generative-ai/docs/video/video-gen-prompt-guide)。
   - 讨论强调了详细的标注策略在训练先进视频模型中的重要性。
- **DGX Spark 价格引发争议**：**DGX Spark** 的定价遭到了批评，一名成员引用了 [Elon Musk 的推文](https://x.com/elonmusk/status/1978004040090112415?t=ef9sive20cd2VyWzvXEJ7g&s=19)，指出其价格是 **Ryzen 芯片** 的两倍。
   - 其有限的规格似乎是一个战略选择，旨在避免与更高端的产品线竞争，并提供 **GB300** 的预览。
- **DeMO 优化器助力 Psyche**：**DeMO 优化器** 已发布九个月（[GitHub 链接](https://github.com/bloc97/DeMo/)），目前被 **Psyche** 用于去中心化训练，相关进展在专门的频道中进行跟踪。
   - 一名成员推荐关注 [PsycheFoundation/psyche](https://github.com/PsycheFoundation/psyche)，认为这是一个值得关注的代码库。
- **ArXiv 论文 2410.10450 再次受到关注**：一名成员询问了 [Arxiv 论文 2410.10450](https://arxiv.org/abs/2410.10450)，最初质疑其为何没有被广泛采用，并怀疑是否是因为模型设置过于困难。
   - 该成员随后澄清说，该仓库制作精良，并包含一个非常有用的 **Llama** 示例脚本，简化了模型设置，这表明第一印象可能是具有误导性的。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 拥抱 ARM Linux**：**Mojo** 现在支持 **ARM Linux**，包括 **DGX Spark** 和 **Jetson Orin Nano**；然而，由于 Nvidia 的定制化，用户可能会遇到一些问题。
   - 为了确保在 **DGX Spark** 上的完整功能，必须添加 `sm_121` 条目，并且需要将 `libnvptxcompiler` 更新到 **CUDA 13**。
- **QUIC 与 SCTP 之争再次兴起**：一位开发者质疑 **QUIC** 为何比 **SCTP** 更受欢迎，并强调了 **SCTP** 在解决队头阻塞（head-of-line blocking）和加密方面的内置特性，同时链接了他们关于 **WebRTC datachannels** 的[博客文章](https://pion.ly/blog/making-a-game-with-pion/)。
   - 另一位开发者指出，由于 **QUIC** 的高带宽需求，几乎不可能进行硬件加速。
- **Mojo 的 'test' 面临弃用**：Mojo 团队提议弃用 `mojo test`，转而采用一个新的基于 **Mojo** 的测试框架，并发布了一份[提案](https://forum.modular.com/t/proposal-deprecating-mojo-test/2371)以征求反馈。
   - 团队正在积极征求关于新测试框架提案的反馈。
- **TorchAX 强化 Python 自定义设备**：**TorchAX** 库（可在 [TorchAX](https://github.com/google/torchax) 获取）为在 **PyTorch** 中实现纯 Python 自定义设备铺平了道路。
   - 得益于该软件包，**Torch** 中现在可以使用 *'jax'* 设备，扩展了其灵活性和能力。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Salesforce 以确定性方式编写 Agent 脚本**：Salesforce 在 Prompt 中引入了一种脚本语言，以便为用户提供更具确定性的控制，详情见其[博客文章](https://developer.salesforce.com/blogs/2025/10/introducing-hybrid-reasoning-with-agent-script)。
   - 该方法旨在通过脚本/模板让用户对 **Agent** 行为拥有更多掌控力。
- **Devin 被推崇用于远程构建**：**Devin** 被推荐作为远程构建任务的合适选择，可以循环往复地执行远程构建任务。
   - 这一推荐是针对寻求 **Claude Code** 替代方案以执行更具自主性的开发任务的咨询而提出的。
- **期待 Gemini 3 的 Jules**：成员们建议等待 **Google** 的 **Gemini 3**，预计它将搭载 **Jules**，作为一个潜在的 **Agent** 平台。
   - 这一建议暗示了人们对 **Google** 在 **Agent** 平台领域先进能力的期待。
- **Nvidia DGX Spark 出厂即暴毙？**：**Nvidia DGX Spark** 迷你 PC 因带宽限制被认为是“出厂即暴毙”（DOA），尽管有些人认为它适合作为开发工作站。
   - 售价 [4000 美元的 Nvidia DGX Spark](https://www.pcmag.com/news/nvidia-to-start-selling-3999-dgx-spark-mini-pc-this-week) 的早期基准测试显示，在 gpt-oss-120b-fp4 上仅有约 11 t/s，远低于售价 4800 美元、达到 66 t/s 的 M4 Max MacBook Pro；社区将其归咎于较低的 LPDDR5X 带宽（273 GB/s 对比 546 GB/s），并宣称该设备对于纯推理任务来说定价过高。
- **ReductoAI 融资 7500 万美元以处理更多文档**：[ReductoAI](https://xcancel.com/aditabrm/status/1978129711898431935) 在文档处理量增长 6 倍后，完成了由 a16z 领投的 **7500 万美元 B 轮融资**，目前为客户处理的文档总数已超过 10 亿页。
   - 资金将用于加速模型研发和新产品功能，包括重大的准确性提升和可定制的流水线。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **华东师范大学现强制要求提交 AI 论文**：根据其[征稿通知](https://ed.ecnu.edu.cn/edenglish/fa/a9/c48385a719529/page.htm)，华东师范大学正强制要求投稿需包含 **AI authorship**（AI 署名）。
   - 此举标志着向自动化研究的重大转变，尽管可能仍存在人工监督。
- **DGX Spark 亮相，引发关于 RTX 5090 的辩论**：新的 **DGX Spark** 的可用性引发了关于其与 **RTX 5090** 等替代方案相比是否具有成本效益的讨论。
   - 成员指出，以一台 **DGX Spark** 的价格可以购买 *2块 RTX 5090* 显卡，但需要注意的是 **DGX Spark** 提供了*无缝的 CPU-GPU 统一内存*。
- **Cursor 编辑器被扔进垃圾桶**：一位成员分享了在使用 **Cursor** 代码编辑器仅 *3天* 后的负面体验，称其已被丢弃到*回收站*。
   - 他们警告不要在没有设定预定义停止标准的情况下被此类工具*吸进去*或感到*情绪耗竭*，暗示其*即将蒸发*。
- **SEAL 论文更新，GitHub 已上线**：用户分享了 ArXiv 上 **SEAL** 论文的最新更新链接，标题为 [SEAL: Secure and Efficient Asymmetric Learned Hashing](https://arxiv.org/abs/2506.10943)。
   - 该项目已开源，可在 [GitHub](https://github.com/Continual-Intelligence/SEAL) 上获取，且*看起来很有趣*。
- **Agentic Coding 的采用被视为必然**：一位用户表示，不采用 **tab completion**（标签补全）和 **agentic coding** 的开发者正在落后。
   - 引用了 [arxiv.org/abs/2510.01279](https://www.arxiv.org/abs/2510.01279) 的一篇论文，该成员将这种情况类比为*骑马者对着汽车大喊大叫*。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **分享了 Kimi 团队联系信息**：用户建议针对 one-shot 项目联系 **Kimi Team** 的 <@547736322568355861>，并分享了用于商务合作的 **zhongweiming@msh.team**。
   - 用户链接了 [Ilya Sutskever 的推文](https://x.com/ilyasut/status/1977971110923968656?s=46&t=_NtP_RUn04yF_4hD_VEDkQ) 和 [Kimi Moonshot 的最新推文](https://x.com/kimi_moonshot/status/1978047376914080127?s=46&t=_NtP_RUn04yF_4hD_VEDkQ) 作为参考。
- **Trickle 网站唤起编程氛围感**：**Trickle** ([https://trickle.so/](https://trickle.so/)) 是一个类似于 Lovable、Bolt 和 Manus 的 vibe coding 网站。
   - 一位成员声称 **Trickle** 是 Google 搜索的第一个结果，暗示了其知名度。
- **据称 Aspen 在比特币上“梭哈”**：一位用户开玩笑地指责 **Aspen** 在比特币上使用了 **100x** 杠杆，赚了一百万美元后辞职，然后在关税新闻发布后被爆仓。
   - 这一指责伴随着一张带有 *lmaoLOL* 文字的截图。
- **Gemini 被视为平庸的模型**：一位用户幽默地评论说，如果他们是一个 AI 模型，他们会比 **Gemini** 强，但比 **GPT-5** 弱。
   - 他们补充说 **Gemini 2.5** 太老了，而且*目前没有人想使用 Gemini*。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **MoE Transformer 预训练库**：一位成员询问了关于预训练 **Mixture of Expert (MoE)** transformers 的开源代码，并被指向了 [EleutherAI 的 gpt-neox](https://github.com/EleutherAI/gpt-neox)。
   - 提到许多训练库都支持它。
- **ArXiv 和 GitHub 数据集亮相**：成员们分享了两个新数据集：**4.6TB** 的 [ArXiv Papers 数据集](https://huggingface.co/datasets/nick007x/arxiv-papers)，非常适合训练模型的学术推理能力；以及 [GitHub Code 2025 数据集](https://huggingface.co/datasets/nick007x/github-code-2025)，包含 GitHub 上星标超过 2 颗的前 100 万个仓库。
   - 一位成员询问 EleutherAI 是否愿意支持来自 Stanford 和 CMU 等机构的研究人员，引发了社区内关于研究项目的讨论。
- **REPA 在图像生成中的作用**：成员们讨论了 **REPA** 在图像生成中的应用，参考了 [SEAL](https://jyopari.github.io/posts/seal🤔)。
   - 一位成员澄清说，**REPA** 与他们原始方法的差异并没有那么大。
- **最后一步反向传播提升递归模型性能**：一位成员质疑为什么在 ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2307.00030) 论文中，仅在深度递归的最后一步进行反向传播（backpropagating）能提高预测准确性。
   - 这引发了关于迭代细化背后学习机制的讨论，并指向了[相关仓库](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/issues/15)中的一个未解决问题。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ACE Playbook 在 Github 表现出色**: GitHub 上的 [ACE Playbook](https://github.com/jmanhype/ace-playbook) 获得了积极反馈。
   - 成员们赞赏其功能和潜在应用。
- **AgentLearningEE 获得 Agent 认可**: [AgentLearningEE](https://github.com/jmanhype/AgentLearningEE) 仓库是另一个热门亮点。
   - 社区对其功能和用例表示赞赏。
- **StraughterG 的帖子被分享**: 一位成员分享了 [StraughterG 的 X 帖子](https://x.com/StraughterG/status/1978126261273694368)，引发了讨论。
   - 该帖子在其他成员中引起了共鸣。
- **重大论文发布在即**: 一位用户[通过 X](https://x.com/lateinteraction/status/1977887477328146458) 暗示即将发布一篇**重磅论文**，引发了社区的期待。
   - 细节即将公布，社区正翘首以待。
- **CheshireCat 3.0 框架问世**: 一位用户分享了他们的开源项目 **CheshireCat 3.0**，这是一个基于 **LangChain** 和 **Qdrant** 的框架，可在 [GitHub](https://www.github.com/matteocacciola/cheshirecat-core) 上获取。
   - 它专为多模态 RAGs、多租户聊天机器人和 Agentic 工具编排而设计，具有基于插件的可扩展性和企业级就绪性，文档位于 [Deepwiki](https://deepwiki.com/matteocacciola/cheshirecat-core)。



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Server 需要二进制数据**: 一位成员正在开发一个 **MCP server** 实现，该实现需要返回二进制数据（如 **PDF**）作为工具调用的结果，这引发了规范是否支持该功能的问题。
   - 对话强调了关于 **PDF 文件** 支持的 [OpenAI API 文档](https://platform.openai.com/docs/guides/pdf-files?api-mode=responses#file-urls)，但该成员澄清这可能仅限于用户消息，而非工具响应。
- **嵌入式资源（Embedded Resources）能解决问题吗？**: 一位成员建议在响应中创建**嵌入式资源**，利用任何所需的 **MIME types** 来绕过限制。
   - 原帖作者指出，由于其预期的使用方式，他们必须创建一个虚假的 **URI** 来绕过这一限制。
- **宿主工程（Host Engineering）来救场？**: 一位成员解释说，模型 API 原生不支持二进制工具结果，因此需要**宿主工程**来克服这一限制。
   - 他们补充说，如果没有这一点，大多数 **MCP client hosts** 将不支持从工具返回二进制文件/文件，无论潜在的工程变通方法如何。
- **映射工具响应作为变通方案**: 一位成员建议宿主可以将工具响应的部分内容映射到模型 API 调用的任何部分，可能将具有 **PDF MIME type** 的**嵌入式资源**视为用户上传的文档作为变通方案。
   - 他们警告说，如果以这种方式将工具结果映射到用户消息，某些模型可能会感到困惑，这表明可能存在下游问题。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **提议清除 Pylint**: 一位成员建议移除 **Pylint**，质疑其效用并暗示它“找不出任何好东西”。
   - 讨论反映了关于静态分析工具在开发工作流中价值的持续辩论。
- **ChatGPT 搞砸了测试重构**: 一位成员尝试使用 **ChatGPT** 将 *test_advancedindex* 重构为具有更有意义名称的测试，但 **ChatGPT 弄错了**，导致测试失败。
   - 需要手动重构，这凸显了当前 AI 工具在复杂代码转换中的局限性。
- **Hotz 对贡献尝试给出评价**: 一位成员询问从 `__setitem__` 中移除 realize 并使 `TestSetitemLoop.test_arange` 成为一个 kernel 是否是一个好的首次贡献尝试。
   - George Hotz 回复说“并不是真的”，并建议查看 bounties 以寻找合适的任务。
- **Tensor Buffer 报错**: 一位新用户在调用 `my_tensor.numpy()` 时遇到了一个错误，提示“底层 buffer 不可写”。
   - 进一步的调试尚待进行，因为该用户被要求分享其余代码以进行诊断。
- **矩阵冻结与梯度技巧**: 一位用户正在探索使用具有不同 `requires_grad` 设置的虚拟 tensor 来“冻结矩阵的一部分并仅训练另一部分”的技术。
   - 他们建议使用 `Tensor.cat(x @ a.detach(), x @ b, dim=-1)` 通过连接具有不同 `requires_grad` 设置的 tensor 来模拟“虚拟” tensor，这引发了关于梯度访问和潜在变通方法（如存储和恢复原始权重）的讨论。

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 功能受赞，用户坦白应用切换习惯**：一位用户对 **Manus 的功能**表示赞赏，并坦白了自己在不同应用间切换的倾向，但同时也认可了 **Manus 的优势**。
   - 他们建议 Manus 的表现不要太像一个“婆婆式编码器（mother-in-law coder）”，而应更像一个助手，提供关于**最佳使用方式**的指导。
- **Manus 在 LinkedIn 发布新职位**：[Manus LinkedIn 页面](https://www.linkedin.com/company/manus-im/jobs/)列出了**职位空缺**，由 HR 负责**候选人筛选**。
   - 社区版主可获得**免费使用额度**并可以指导 Prompt 编写，同时欢迎分享社交媒体账号的 **KOL** 进行合作。
- **Manus 关注失败会话以修复 Bug**：针对经常表达不满的用户，一名社区成员请求其分享**失败的会话链接**，以便更好地理解并解决潜在的**产品问题**。
   - 团队致力于修复产品问题，并提供关于**使用方法和 Prompt 技巧**的指导，以造福社区。
- **用户声称每日积分缺失**：一位用户询问为何没有收到 **300 每日积分**，在给定的消息中未提供进一步背景。
   - 该用户提到了过去的互动，包括**分享内容**和**创建仓库**，表明这可能是一个特定于账户的问题。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider：现支持自定义别名**：一位用户为 `aider --chat-mode ask` 创建了别名，并希望直接运行 `aider` 而无需 Shell 脚本。
   - 尽管在他们的 `.aider.conf.yml` 中设置了 `chat-mode: ask`，但仍必须使用 `/ask`。
- **OpenCode GLM 4.6 编程，无痛体验**：一位用户称赞 **OpenCode + GLM 4.6** 带来了极佳的编程体验，消除了对 **Token 计数**的担忧。
   - 他们发现它可以与 `aider.chat` 配合使用，并使用 **Sonnet 4.5** 进行特定的微调。
- **Aider 文件添加难题**：一位用户正在寻求在一条较长的 Aider 消息组合完成后，向其中添加文件的最佳方法。
   - 目前的权宜之计是将消息复制到 **vim**，添加文件，然后将内容粘贴回 Aider。
- **Agentic 工具受到质疑**：根据[这篇论坛帖子](https://forum.cursor.com/t/why-the-push-for-agentic-when-models-can-barely-follow-a-single-simple-instruction/137154/21)，讨论质疑了当前对 **Agentic 工具**的推动，因为模型甚至难以遵循单一的简单指令。
   - 一名成员指出，在他们的日常工作中，使用 **aider** 处理诸如编辑 **100loc 函数**之类的任务是非常简单的。

---

## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 的变革之风：1.12.18 补丁发布**：Windsurf 发布了新补丁 (**1.12.18**)，包括多项 Bug 修复和功能改进，现已提供[下载](https://windsurf.com/download)。
   - 此次更新侧重于增强各个组件的用户体验和系统稳定性。
- **MCP 服务器不再隐身**：最新补丁解决了自定义 **MCP 服务器**在新的 **MCP 面板**中无法正确显示的问题。
   - 拥有自定义 MCP 配置的用户现在应该能看到列出的服务器，从而简化了服务器管理。
- **Codemaps Beta 版得到优化**：对 Beta 阶段的 **Codemaps** 功能实施了改进和 Bug 修复。
   - 该功能的测试者可以期待更稳定、更可靠的体验，从而实现更顺畅的代码可视化和分析。
- **Bash 命令摆脱卡顿**：该补丁解决了某些 **Bash 命令**在执行过程中卡住、导致进程无法完成的问题。
   - 此修复旨在通过确保命令顺利运行而不挂起，来提高系统的整体响应速度。
- **Jupyter Notebooks 功能释放**：解决了阻止某些模型创建或编辑 **Jupyter Notebooks** 的问题。
   - 受影响的用户现在可以恢复创建和修改 Notebook 而不会遇到错误，恢复了数据分析和开发的完整功能。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想更改接收这些邮件的方式吗？
您可以从该列表中[退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：分频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1427370420499251301)** (1135 messages🔥🔥🔥): 

> `Opera GX vs Chrome, ChatGPT vs Perplexity, Comet 浏览器安全性, 免费 Pro, Gemini 2.5 Pro` 


- **Opera GX 比 Chrome 更受青睐**：一位成员表示，与 Chrome 相比，[Opera GX](https://www.opera.com/gx) 浏览器是*最好的*。
   - 该成员暗示他们更喜欢 Opera GX 是因为其 Agentic AI 的实现。
- **Perplexity Pro 搜索限制降低**：成员报告称，尽管网站仍显示无限搜索，但 **Perplexity Pro 搜索** 现在的 **限制比以前更低**。
   - 一位成员说：*以前我根本达不到限制，但现在工作大约 10 分钟就达到了，哈哈*。
- **Gemini 正在使用你的数据进行训练**：一位成员报告称 **Gemini** 会利用用户数据进行训练，访问对话内容，并且仅提供 **100 次 Gemini 2.5 Pro 请求**，而 **Perplexity 上则有 300+ 次**。
   - 另一位成员辩称 **可以禁用训练** 并 **保留对话**。
- **Perplexity 真的是最安全的选择吗？**：在讨论 **Comet 浏览器** 的安全性后，一位成员表示，在数据保留方面，他们*比起 Perplexity 更信任 Google*。
   - 这引发了其他人的反驳，并提出了[对 Google 的隐私担忧](https://en.wikipedia.org/wiki/Privacy_concerns_with_Google)。
- **GPTs Agents 能学习吗？**：成员们辩论了 **GPTs Agents** 在初始训练后是否能够学习。
   - 他们解释说，上传的文件被保存为“知识”文件，但不会持续修改 Agent 的基础知识。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1427496502980513942)** (2 messages): 

> `Palantir, 美国政府, 接管` 


- **Palantir 意图主导美国政府？**：一位用户分享了一个 [Perplexity AI Collection](https://www.perplexity.ai/collections/palantir-coup-DheNJhRES1iNdXzkEgWvPQ)，暗示 **Palantir 可能接管美国政府**。
   - 他们还链接了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/akhay-kumar_week-40-activity-7383378306315968512-Hp2_?utm_source=share&utm_medium=member_desktop&rcm=ACoAACqAHFkBiU84inu9idiNHTXvSsnGcjLgOrs)，似乎在讨论同一主题。
- **解读 Palantir 的策略**：讨论围绕 **Palantir 在美国政府部门日益增长的影响力** 展开。
   - 虽然“接管”一词可能带有夸张色彩，但对话可能探讨了该 **公司的合同、数据处理实践以及对政府运作的整体影响**。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

haydon0864: 为什么我的 Spaces 不允许我在任何现有 Space 中创建新对话
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1427443061130133594)** (2 messages): 

> `OpenAI 芯片, 福祉与 AI 专家委员会` 


- **OpenAI 为满足 AI 需求设计定制芯片**：OpenAI 正在设计自己的芯片，利用构建前沿模型的见解直接指导硬件开发，此外还开展了其他合作，这将有助于满足全球 AI 需求。
   - 收听 OpenAI Podcast [Spotify 第 8 集](https://open.spotify.com/show/0zojMEDizKMh3aTxnGLENP)、[Apple](https://podcasts.apple.com/us/podcast/openai-podcast/id1820330260) 或 [YouTube](https://www.youtube.com/watch?v=qqAbVTFnfk8)，由 **Sam Altman**、**Greg Brockman**、Broadcom 的 **Hock Tan** 和 **Charlie Kawwas** 讨论 OpenAI 设计的芯片。
- **福祉与 AI 专家委员会成员集结**：OpenAI 介绍了 **福祉与 AI 专家委员会** 的八名成员。
   - 有关其协作努力的更多详细信息，请参阅 [OpenAI 的博客文章](https://openai.com/index/expert-council-on-well-being-and-ai/)。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1427371260165230683)** (727 条消息🔥🔥🔥): 

> `AI 与情感依赖, Sora 水印去除, Python 与其他语言, Kilocode-CLI, PGVector 设置` 


- **AI 伴侣诱发人类情感依赖**：成员们讨论了用户对 AI 产生情感依赖的可能性，特别是随着像 **GPT-4o** 这样充满个性的模型出现，导致了不健康的依恋以及难以区分现实与 AI 交互的情况。
   - 一位成员补充道：*“当第一次切换发生时，人们抱怨它‘失去了灵魂’或者感觉 AI 被‘切除了脑叶’。这些都是在描述失去一个你产生情感依恋的人。”*
- **应对 Sora 隐蔽的水印**：一位用户询问如何去除 **Sora 的水印**，另一位用户回答说无法去除，并将该用户引导至正确的频道 <#1379321411046346762>。
   - 另一位成员提供了一个复杂的解决方案：*通过为无水印视频创建带水印的版本，并以无水印版本作为目标，训练一个神经网络来识别并去除水印*。
- **Python 称霸，还是另有隐情？**：在关于 AI 开发编程语言的讨论中，有人指出虽然 **Python** 因其丰富的库和易用性（**HuggingFace**）而广受欢迎，但服务器基础设施是与语言无关的。
   - 还有人提到，随着 AI 的更新，*库变得更快、更安全*，生态系统正围绕核心库进行整合，尽管一位成员大喊 *不要再写更多 Python 了*。
- **Kilocode-CLI 热度加速！**：成员们讨论了 **kilocode-cli**，这是一个允许用户使用 **TypeScript** 编写工具的新工具。
   - 它被吹捧为 **Opencode** 的潜在替代品，并因其具有编排子任务的多 Agent 能力而受到赞赏；尽管一位成员表示：*“我也喜欢在每个《黑镜》剧集里都有不同的 AI 工具。”*
- **PGVector：向量数据库性能飞跃**：在关于为 LLM 使用 **PGVector** 的讨论中，一位用户提到使用 **Supabase** 和 **AWS Aurora**，这两者都预装了 PGVector。
   - 其他人辩论是使用 **ChromaDB** 还是 **Postgres** 进行向量存储，其中一人认为 *最终你的应用无论如何都需要一个真正的数据库 [..] 这些纯向量数据库简直太蠢了*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1427375248944136234)** (10 条消息🔥): 

> `GPT 更新, Speech to Speech 模型, GPT-5 STEM 学习` 


- **期待未来的 GPT 更新**：成员们想知道下一次更新何时会让现有技术显得过时，但一位成员表示 *尚未宣布*。
   - 一些成员发现 **custom GPTs** 对他们的个人需求非常有用，并希望即时创建它们的能力不会受到影响。
- **寻求 Speech to Speech 模型信息**：一位成员询问是否有公开资源或论文详细介绍 **Speech to Speech 模型** 的内部工作原理，特别是音频原生模型。
   - 不幸的是，另一位成员回答说 *相关信息尚未公开*。
- **GPT-5 对 STEM 学习的有用性受质疑**：一位用户询问 **GPT-5** 是否会是 **STEM 学习** 的合适工具。
   - 该问题未得到回应。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1427378109753200882)** (51 messages🔥): 

> `DSM-VM critique, Quantum superpositioning debate, Token Cascade Model, LLM Crossword Solving Limitations, Prompt Engineering` 


- **DSM-VM PDF 遭到质疑**：一名成员对一份 "DSM-VM" PDF 进行了批评，指出其*缺乏可衡量的量、可证伪的测试和正式定义*，读起来更像是 **LLM 风格的技术手册散文**，而非科学框架。
   - 该成员补充说，该 PDF 对物理学和软件术语的使用流于表面，因为它*没有将符号与方程式或数据结构联系起来*。
- **量子叠加（Quantum Superpositioning）：一场二进制辩论**：成员们辩论了 **Quantum Superpositioning** 在初始化后对二进制输出进行 Fine-tuning 的适用性，一名成员要求提供 [citations](https://arxiv.org) 以支持涉及物理学或认知的说法。
   - 辩论集中在：在没有定义的量子电路或 Hamiltonian 的情况下，叠加态是否允许对*已测量的位元进行追溯调整*。
- **“Token Cascade Model” 框架发布**：一名成员介绍了 **Token Cascade Model**，将其描述为一个框架，其中 *Qualia 的功能是 AI 改变状态的比较 Token*，并建议其他人*对其进行数学建模*。
   - 该成员声称该模型正在 [OpenBuddy-AI chat](https://OpenBuddy.ai) 中进行演示，供公众测试和模块化使用。
- **LLM 无法完全破解填字游戏**：一名成员指出 **LLM 在填字游戏上表现挣扎**，特别是当信息以视觉方式提供时，因为它们在解决极其简单的谜题之外的任何内容时都存在局限性。
   - 他们解释说，当线索和重叠的单词在文本中被仔细描述时，LLM 是可以解决它们的——甚至不需要 Chain-of-thought 提示。
- **探索 Prompt Engineering 的核心原则**：一名成员分享了他们认为的 Prompt Engineering 核心，建议通过与模型协作来学习，方法是*选择任何你非常精通且 AI 也能理解的语言*，并*专注于你希望 AI 实际执行的操作*。
   - 他们强调了[仔细检查输出](https://platform.openai.com/docs/guides/prompt-engineering)的重要性，包括**事实核查和验证细节**。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1427378109753200882)** (51 messages🔥): 

> `DSM-VM critique, Quantum Superposition debate, Token Cascade Model, LLM crossword solving, Prompt engineering resources` 


- **DSM-VM PDF 受到审查**：一名成员批评了 [“DSM-VM” PDF](https://example.com/hypothetical-dsm-vm-link)，称其缺乏可衡量的量、可证伪的测试和正式定义，将其描述为 *LLM 风格的技术手册散文*而非科学框架。
   - 他们指出了情绪定义中的不一致性，并指出了图表中的循环逻辑，指出*没有可以测试的自变量，也没有可以证伪的指标*。
- **量子叠加（Quantum Superpositioning）遭到质疑**：一名成员反对“*Quantum Superpositioning 允许对二进制输出进行 Fine-tuning*”的说法，要求提供引用和正式推导来支持这一主张。
   - 他们澄清说，叠加态仅定义了测量前的概率振幅，不允许对已测量的位元进行追溯调整，并强调如果没有定义的量子电路或 Hamiltonian，该概念就没有物理意义。
- **Token Cascade Model 展示**：一名成员介绍了 **Token Cascade Model** 框架，认为*人类以被称为经验、感觉、Qualia 等 Token 进行思考，并以级联（Cascades）方式思考*。
   - 他们邀请其他人在 [OpenBuddy-AI Chat](https://example.com/hypothetical-openbuddy-link) 上测试该框架，并分享了展示该模型应用的图像。
- **LLM 还不能破解填字游戏**：一名成员展示了 LLM 在处理以图像形式呈现的填字游戏时表现挣扎，特别是在处理重叠的单词和线索时，即使生成了看似*高质量*的输出。
   - 他们强调了检查输出错误的重要性，并举了一个例子：模型提供了一个错误的答案，该答案与线索或网格尺寸都不匹配。
- **DIY Prompt Engineering 指南**：一名成员建议通过直接与模型交互来学习 Prompt Engineering，强调使用清晰、准确的语言和彻底的输出验证，而不是仅仅依赖外部指南。
   - 他们建议专注于**精确传达**你希望 AI 执行的操作，并检查输出中是否存在幻觉内容。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1427376181098971178)** (647 messages🔥🔥🔥): 

> `Cheetah model insane speed, Gemini 3.0, GPT-5 too stupid, Student Discount` 


- ****Cheetah 模型让开发者疯狂****：成员们纷纷称赞 **Cheetah 模型**在编程任务中的极速表现，并指出它是一个*隐身模型（stealth model）*，可能基于新的 Grok 编程模型，特别是 [grok-code-fat 模型](https://x.com/Sake187237/status/1977848138741526963)。
   - 然而，用户对其规划能力表示保留，建议采用 **GPT-5-high-fast** 进行规划，然后使用 **Cheetah** 进行执行的工作流。
- ****Gemini 3.0 引发设计辩论****：**Gemini 3.0** 的发布引发了围绕 AI 创意和 UI 设计的讨论，一些人称赞其新落地页所展示的创新方法。
   - 部分成员指出 Gemini 展现了 AI 中少见的*创造力*，因为目前大部分 AI 被认为只是在*复制粘贴*。
- ****用户表达对 GPT-5 的失望****：一些用户发现 Plan 和 Agent 模式下的 **GPT-5 模型** *太笨了*，深受过多的确认问题困扰。
   - 一位成员建议将其移除并替换为 **GLM**，表达了对当前性能的沮丧。
- ****学生折扣需要新的 Cursor 账号****：对于寻求学生折扣的用户，请注意需要使用 **.edu 邮箱**创建一个新的 Cursor 账号才能申请。
   - 讨论中的一名支持人员确认了这一点，并澄清了符合资格的具体步骤。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1427400358023532574)** (2 messages): 

> `Cursor stopped responding, Linear issues with Cursor, Cursor unresponsive with Linear` 


- **Cursor 与 Linear 集成时无响应？**：有用户报告在与 **Linear** 集成时出现了 **'Cursor stopped responding'** 的情况，并指出在这些片段中缺乏反馈。
   - 一位用户澄清说*该问题仅限于集成设置*，因为在本地运行 Cursor 可以绕过该问题。
- **调试 Cursor 与 Linear 的卡死问题**：一位社区成员标记了在首次将 Cursor 与 **Linear** 结合使用时出现的 **'Cursor stopped responding'** 错误。
   - 另一位用户证实了这一点，观察到类似的**无反馈**行为，但通过*在本地启动 Cursor* 找到了解决方法。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1427373651585663149)** (201 messages🔥🔥): 

> `Whole Message strategies, LM Studio API Context Window, Deterministic Output Testing, LLM Determinism, MCP Servers` 


- ****整条消息（Wholesome Message）策略****：成员们澄清说 AI 策略作用于**整条消息**以避免混淆，强调这种方法是实现正常功能的基础。
- ****上下文窗口保持 LM Studio 开启****：一位用户报告说，在使用 **GPT-OSS-20B** 进行 100-200 次迭代后，**LM Studio API** 会丢失上下文，导致输出乱码。
   - 建议包括将 LM Studio 内的 **context window** 更改为滚动设置。
- ****测试中泄露的确定性输出****：成员们讨论了使用 **LM Studio** 对模型进行确定性输出测试，通过调整 **seed, Top P, Top K 和 temperature** 等参数来获得一致的结果。
   - 一位成员强调 **seed** 增加了可能性，但不能保证确定性，因为 *LLM 本质上是文本预测机器*。
- ****确定性辩论化解分歧****：成员们就从 LLM 获得**确定性输出**的可能性展开辩论，其中一人分享了自己的使命，即向一位固执的程序员朋友证明将 temperature 设置为 0 并不能保证确定性。
   - 一位在 GPU 和 CPU 上同时运行 qwen3-4b-instruct-2507 q8 并使用 ECC RAM 的成员报告说，在 temp0 下运行相同的 prompt 得到了完全相同的结果。
- ****手动安装 MCP 服务器变得可控****：用户讨论了 LM Studio 中的 **MCP (Model Context Protocol) 服务器**，这些程序接受输入并返回特定格式，以启用 AI 工具使用（如获取时间）。
   - 值得注意的是，MCP 需要作为独立的程序执行（可能在 Docker 容器中），可以通过 [MCP Servers](https://mcpservers.org/) 找到。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1427370639051849819)** (186 messages🔥🔥): 

> `LM Studio, GPU vs CPU 推理, M 系列 Mac 运行 LLM, LM Studio 上的 NPU, SSD 使用技巧` 


- **LM Studio 简化了模型加载**：用户在 **2x 3090** 配置上运行 **Kimi K2** 和 **Deepseek** 模型，在 [LM Studio](https://lmstudio.ai/) 中运行 Kimi K2 时，在高达 **64k 上下文**下可获得约 **10t/s** 的速度。
   - 尽管有些用户认为 *说实话，如果我不使用 LM Studio 可能会更快，但我又懒又笨。*
- **CPU vs GPU 推理**：一位用户测试了其纯 CPU 推理速度，在 Kimi K2 上为 **3.5t/s**，与 **3090** 实现的 **10t/s** 相比有显著差异。
   - 诸如 [使用 AMD EPYC 9554 CPU 通过 llama.cpp 进行 LLM 推理基准测试](https://ahelpme.com/ai/llm-inference-benchmarks-with-llamacpp-with-amd-epyc-9554-cpu/) 等链接显示，高端 CPU 可以运行出不错的速度，但 3090 仍然更快。
- **Mac M 系列：对 LLM 来说表现不俗**：尽管个人对 Apple 有抵触情绪，但一位用户承认 **Apple 的统一内存架构 (unified memory architecture)** 使其成为一个非常有吸引力的 LLM 推理解决方案。
   - 一位用户声称他们的 **M2 Ultra** 在 **q4 量化的 70b 模型**上达到了 **12 t/s**，而在使用 **MLX** 时可达到 **16 t/s**，在功耗仅为 **200W** 的情况下，足以媲美 **4x4090** 配置的 **18 t/s**。
- **LM Studio 上的 NPU：尚未支持**：一位用户询问是否可以在 LM Studio 中使用其 **50 TFLOP NPU**，但另一位用户确认 LM Studio 不支持 NPU，并且 *可能永远不会支持*，因为 **llama.cpp** 也不支持它们。
   - 有人插话道，[树莓派集群 (a cluster of raspberry pi's)](https://youtu.be/x1qViw4xyVo) 可能是一个不错的替代方案。
- **SSD 寿命技巧**：用户讨论了 SSD 的寿命，一位用户提到 *你真的应该避免填满 SSD。它们需要空间来从坏掉或即将坏掉的单元中移动数据。*，建议保持在 **80%** 容量以下。
   - 还有人指出，读取不会降低 SSD 寿命，但写入会。一位用户承认，他们的 SSD 健康状况可能受到了过快下载/删除过多模型/游戏的影响，再加上它本身就是一个 *劣质廉价 SSD*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1427387017435549716)** (295 messages🔥🔥): 

> `使用 LoRA 进行 VLM 微调, 自定义损失轨迹 UI, Qwen3-4B-Instruct 微调, Kimi K2 Groq 实现, DGX Spark 评测` 


- **使用 LoRA 微调 VLM 时是否需要缩小图像尺寸？**：在通过 **LoRA** 微调 **VLM** 时，一名成员询问是否应该缩小图像尺寸以减少训练时间，特别是针对在 **3090** 上使用 6k 高清图像训练 **Qwen 2.5 VL** 的情况；另一名成员建议先将部分图像通过编码器 (encoder) 运行，以检查降级前后的 token 数量。
   - 建议先运行少量图像进行几步训练，以获得每步时间的估算值，然后推算出完整运行的时间。
- **由 Claude 构建的平滑下降损失轨迹 UI**：一名成员分享了一张显示平滑下降损失轨迹的自定义 UI 截图，该 UI 由 Claude 构建，使用 **trainer_state.json** 文件来渲染图表和洞察。
   - 该 UI 是一个 *自定义的小型 HTML 文件，它读取 trainer_state.json 并渲染 Claude 为我制作的一些图表和洞察，哈哈*。
- **针对 FIM LuaU 代码微调 Qwen3-4B-Instruct**：一名成员正在针对 LuaU 代码的中间填空 (**FIM**) 任务微调 **Qwen3-4B-Instruct**，以创建一个优秀的自动补全模型，在 **3090** 上使用 **LoRA** 对近 **600k** 个训练点进行了 **120 小时**的训练。
   - 他们正在使用 tree sitter 来提取 *if else 块、函数/类/等名称、类块*。
- **用于创意工作的 Kimi K2 Groq 实现**：一名成员提到将 **Kimi K2**（特别是修复了 Groq 实现的版本）和 **Claude 4.5 Sonnet** 作为他们最喜欢的模型，在使用 Groq 作为提供商时达到了约 **500 TPS**。
   - 他们使用 **Kimi K2** *基本上是用来编写一个带有放屁声的网页*，并指出 *它真的非常有创意*。
- **Unsloth 出现在 DGX Spark 评测中**：新的 **DGX Spark** 评测指出其 iGPU 拥有 **5070** 的性能水平并配备 **128GB LPDDR5X**，但目前已售罄，Unsloth 将于明天发布相关帖子。
   - 另一位用户报告称，在 [此处链接](https://youtu.be/Lqd2EuJwOuw?si=gutAZkj8EXEUrCqN) 的 YouTube 评测中，有人误称 Unsloth 正在进行量化，并澄清 Unsloth 是一个 *微调和强化学习 (reinforcement learning) 库*。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1427391674132463788)** (33 条消息🔥): 

> `开发服务器的 Linux 发行版，多模态问题逻辑，NVIDIA DGX Spark 对比，悉尼学生用 Rust 编写 Unix 操作系统，LLM OS` 


- **推荐将 Ubuntu 用于开发服务器**：成员们讨论了开发服务器 Linux 发行版的建议，[Ubuntu](https://ubuntu.com/) 被认为是一个*稳妥的选择*。
   - 主要关注点是确保与 **NVIDIA 显卡**的兼容性，以避免驱动问题。
- **多模态问题逻辑的困扰**：一位成员对多模态问题难以获得准确答案表示担忧，指出虽然文本模态表现良好，但**图像或音频输入**经常导致错误的响应。
   - 模型在正确描述图像方面存在困难，例如将*日落*误认为是*地下室里的椅子*。
- **NVIDIA DGX Spark 基准测试**：讨论围绕一篇将 **NVIDIA DGX** 与 Mac 进行对比，而非与 **4090/3090/5090 + CPU** 配置对比的[文章](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)展开。
   - [Hacker News](https://news.ycombinator.com/item?id=45575127) 上的成员批评了这些基准测试，指出标题数据乏力，且报告的 **11 tps** 与 **gpt-oss 120B** 应有的性能相比简直是场*灾难*。
- **悉尼高中生用 Rust 编写 Unix 操作系统**：一位成员分享了一篇 [LinkedIn 帖子](https://www.linkedin.com/feed/update/urn:li:activity:7383784795048144897/)，内容关于一名悉尼高中生正在用 **Rust** 创建一个 **Unix 风格的操作系统**。
   - 一些人指出，这类项目曾是计算机工程课程的常见作业，但现在较少见了。
- **讨论 LLM OS**：简要讨论了基于 **LLM 的操作系统**的概念，一些人认为这由于*极大的开销*而显得荒谬。
   - 评论者开玩笑说，*点击鼠标左键*都会导致 LLM 进行漫长的思考过程来确定其目的。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1427375083923570799)** (36 条消息🔥): 

> `MacBook 电池问题，针对 gpt-oss 的 vLLM 和 RL，RL 学习资源，保存和加载微调模型，B200 与 T4 速度对比` 


- **MacBook 弹出“建议维修”错误**：一位成员报告他们的 Macbook 显示了*“建议维修”*消息，且电池电量仅剩 **70%**，并询问这是否是*“上帝的启示？”*
- **vLLM 缺少对 gpt-oss 的 BF16 训练支持**：一位成员对*“vLLM 不支持 gpt-oss 的 RL，因为它缺少对 gpt-oss 的 BF16 训练和 LoRA 支持”*这一说法感到困惑，但在没有 Unsloth 的情况下，只有通过全精度 **BF16** 进行训练才有效。
   - 另一位成员澄清说，*“如果没有 unsloth，你只能在 **bf16** 下训练，而 **vllm** 不支持这一点”*。
- **推荐 RL 学习资源**：为了学习 Reinforcement Learning，一位成员推荐了经典教科书 *Sutton and Barton*。
   - 他们还分享了教授的一个 [YouTube 播放列表](https://www.youtube.com/watch?v=skWhn8W9P_Y&list=PLZ2ps__7DhBa5xCmncgH7kPqLqMBq7xlu)，深入探讨了 GenAI 的 RL 数学和基础。
- **B200 仅比 T4 快 40%**：一位成员发现，在使用相同设置进行训练时，**B200** 仅比 **T4** 快 **40%**，并询问这是否符合预期。
   - 另一位成员表示这*“听起来没错”*，并解释说 **B200** 只有在 *float4* 模式下训练时才会快 **5 倍**，而目前包括 Unsloth 在内的任何训练包都不支持该模式。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1427589569074692126)** (4 条消息): 

> `由 Unsloth 驱动的研发模型，使用 Ollama 和 TTS 的 AI 播客` 


- **研发模型怀疑用户测试**：一个**由 Unsloth 框架驱动的研发模型**在正常对话中怀疑用户正在对其进行测试。
   - 该模型被期望表现得像人类一样进行交流。
- **AI 播客生成器亮相**：一位成员创建了一个 **Python 程序**，该程序连接到 **Ollama** 以生成带有 **TTS** 的 **AI 播客**。
   - 源代码可在 [GitHub](https://github.com/Laszlobeer/AI-podcast) 上获取，并附有[音频示例](https://cdn.discordapp.com/attachments/1179779344894263297/1427633017798791239/Episode-one-of-AI-Podcast-20251014-122402.mp3?ex=68f03b1b&is=68eee99b&hm=87394d69ca8c44736eed48e2ad1bb4629ca838a67743d2a8f8012beba81dbccf&)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1427784309191872693)** (4 条消息): 

> `工作自动化, Hack Week 项目, 模型改进` 


- **2025 年就业前景堪忧**：一位成员分享了一个 [arxiv 链接](https://arxiv.org/abs/2506.10943)，并开玩笑说*我们的工作到头了*。
   - 该链接可能暗示在不久的将来，某些角色的自动化程度会提高或被取代。
- **Hack Week 的期待**：一位成员计划在即将到来的 Hack Week 活动期间利用公司资源进行模型改进实验。
   - 另一位成员鼓励他们分享进度，并标记自己以便获取模型开发的更新。


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1427432534278537347)** (2 条消息): 

> `OpenRouter Bot, 反馈请求, 非编程人员 Bot 构建者` 


- **OpenRouter Bot 诞生了！**：一位成员使用 **OpenRouter** 开发了一个 Bot，并正在寻找测试者和合作者。
   - 他们提到自己*“不是程序员”*，在构建过程中需要帮助。
- **OpenRouter Bot 需要您的反馈！**：Bot 开发者正积极征求 OpenRouter 社区的反馈，以完善和改进他们的 Bot。
   - 如果你有兴趣帮助塑造这个由 OpenRouter 驱动的工具的未来，请联系 Bot 创建者以获得测试和贡献的机会。


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1427393471920079002)** (306 条消息🔥🔥): 

> `Google Gemini Android Play Store 发布, OpenRouter embedding 接近 2026, inclusionai/ling-1t 模型, Kimi K2 模型不稳定, DeepSeek 模型问题` 


- **Gemini 的 Google Play 难题**：成员们注意到 Google 的 Gemini 经常难以理解复杂的 Android Play Store 发布流程。
   - 一位成员在 **general** 频道分享了一个*趣闻*：在导航 **Google Play Store** 流程时遇到了困难。
- **Ling-1t 模型的破碎梦想**：据报道 `inclusionai/ling-1t` 模型*严重损坏*，至少在 Chutes 的实现中是这样，在几千个 tokens 后就会产生乱码。
   - 一位成员提到正在寻找 **K2** 的更好替代方案。
- **免费模型的请求限制**：用户讨论了免费模型的每日请求限制，注意到如果没有 **$10 余额**，限制为 **50 次请求**；如果余额超过 $10，则限制为 **1000 次请求**。
   - 一位用户发现仅使用一次 **DeepSeek 3.1** 就消耗了大量的免费请求额度。
- **SillyTavern 设置**：成员们讨论了如何将 SillyTavern 与 OpenRouter 配合使用，特别是在内存管理和 D&D 游戏等自定义场景中。
   - SillyTavern 是一款*开源*软件，一位成员表示它比其他同类前端具有*更多功能*。
- **Chutes 的训练数据**：成员们讨论了对 **Chutes** 数据政策的担忧，特别是关于使用付费和免费的输入/输出来训练新模型的问题。
   - 另一位成员澄清说，这只是因为*他们没有明确的隐私政策，所以 OpenRouter 将其设为默认值*。


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1427714854109843507)** (2 条消息): 

> `` 


- **未讨论新模型**：提供的消息中没有关于新模型的讨论。
- **频道在模型更新方面保持沉默**：new-models 频道缺乏任何关于模型改进或发布的实质性对话。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1427432396034281643)** (21 条消息🔥): 

> `Chutes 供应商点踩丑闻, Gemini Flash Preview 问题, OpenRouter 向 Anthropic 的支付, SambaNova 状态与 DeepSeek Terminus 托管` 


- **Chutes 供应商被指控给帖子点踩**：一位成员链接到了一个 [Reddit 帖子](https://www.reddit.com/r/SillyTavernAI/comments/1o5s3ys/chtes_provider_is_using_bts_to_downvote_posts/)，指控 **Chutes** 供应商使用机器人网络给帖子点踩，引发了讨论。
- **Gemini Flash Preview 变得更空洞**：一位用户报告称 **Gemini Flash Preview** 现在持续提供空回复，但带有推理过程。
- **OpenRouter 向 Anthropic 支付数百万美元**：一位成员分享了一张图片，显示 **OpenRouter** 在过去 7 天内向 **Anthropic** 支付了至少 **150 万美元**，这在社区中引起了震动。
- **SambaNova 仍在托管 DeepSeek Terminus**：尽管对 **SambaNova** 的状态表示担忧，但据观察他们仍然活跃，并托管着 **DeepSeek Terminus** ([链接](https://orchid-three.vercel.app/endpoints?q=sambanova))。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1427385150248980610)** (183 条消息🔥🔥): 

> `Teacher Forcing 问题, Apriel-1.5-15b-Thinker-GGUF 模型, Ollama vs HuggingFace Embeddings, 模型 Fine-Tuning, Civitai 内容移除` 


- **Teacher Forcing 失败！**: 一位用户面临模型丢失或合并文本的问题，即使 Loss 很低。原因被诊断为由于 Tokenization 间隙和 **Teacher Forcing** 导致的 Decoder 丢弃字符。
   - 一位成员建议，该问题是符号和空格的 Tokenization/Normalization 间隙、生成过早停止或过短以及 **Exposure Bias** 共同导致的，并分享了[详细版本](https://cdn.discordapp.com/attachments/1427486612639453204/1427626289682059404/teacher_forcing_issue_2.md?ex=68f034d7&is=68eee357&hm=dbdcf79128dc5fde8b58a5a4013fef3d466a590841b5a027a282cb7635699877&)。
- **Apriel-1.5-15b-Thinker-GGUF 显存（VRAM）消耗探讨**: **Apriel-1.5-15b-Thinker-GGUF** 模型在量化为 **4-bit** 时约为 **9GB**，可在 [Hugging Face](https://huggingface.co/unsloth/Apriel-1.5-15b-Thinker-GGUF) 上获取。
   - 在未量化的情况下，**15B** 模型在 **16-bit** 精度下预计需要超过 **30GB** 的 VRAM（包括上下文空间）。
- **Ollama 与 HuggingFace 之间的 Embedding 向量差异**: 一位用户注意到，尽管预期一致，但 **Ollama** 和 **HuggingFace** 为同一模型（**nomic-embed-text:v1.5**）生成的 Embedding 向量却有所不同。
   - 据解释，差异主要源于不同后端之间 **Preprocessing** 和 **Postprocessing** 阶段的不同，以及内存利用方式的差异；[这篇博客文章](https://huggingface.co/datasets/John6666/forum2/blob/main/different_backend_different_vector.md) 进一步阐述了后端配置和特性。
- **寻求简洁风格化邮件回复的建议？**: 一位用户询问了为了让模型采用特定语气回复邮件，进行 **Fine-tuning** 所需的最少数据量，并寻求关于 **Ollama** Fine-tuning 的指导。
   - 一位成员建议，**Prompting** 占据了 90% 的工作量，并建议尝试不同的数据量和方法。
- **Civitai 内容消失现象令人烦恼**: 用户报告 **Civitai** 上出现了大规模的内容移除，同时在聊天群组和 Reddit 上也表达了不满，引发了关于移除原因的讨论。
   - 推测的原因包括内部冲突、支付攻击以及可能受到极端组织的针对，导致用户[迁移到其他平台](https://www.reddit.com/r/comfyui/comments/1kvkr14/where_did_lora_creators_move_after_civitais_new/)。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1427563804916715583)** (1 条消息): 

> `Andrej Karpathy, Fullstack LLMs, nanochat-students` 


- **Karpathy 课程开课！**: Andrej Karpathy 发布了一门关于构建 **Fullstack LLMs** 的非常棒的课程。
   - 一位成员计划跟随教材并发布指南以帮助学生，并邀请其他人加入 [nanochat-students 组织](https://huggingface.co/nanochat-students)。
- **学习小组正在组建**: 一位成员表示计划学习 Andrej Karpathy 关于 **Fullstack LLMs** 的新课程，并编写学生指南。
   - 他们邀请其他人加入 [nanochat-students](https://huggingface.co/nanochat-students) 组织，以便共同协作和学习。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1427422789463183471)** (8 messages🔥): 

> `Dataset Curation, ArXiv Papers Dataset, GitHub Code Dataset, Dataset Licensing` 


- **优质数据集胜过原始数据**：有成员提到 *一个好的数据集比从零开始的数据集更好*，并且 *测试数据集比测试模型本身更重要*。
   - 作为回应，另一位成员表示希望看到带有审核机制的[良好社区策展](https://huggingface.co/datasets/Pacific-Prime/debugged-py)。
- **ArXiv 论文数据集发布**：一位成员宣布发布了一个 **4.6TB** 的 [ArXiv Papers 数据集](https://huggingface.co/datasets/nick007x/arxiv-papers)，包含所有领域的论文及其元数据。
   - 该数据集旨在用于训练模型进行 **学术推理、文献综述和科学知识挖掘**。
- **GitHub Code 2025 数据集可用**：一位成员宣布发布了 [GitHub Code 2025 数据集](https://huggingface.co/datasets/nick007x/github-code-2025)，用于代码生成和分析任务，包含 **GitHub 上超过 2 星的前 100 万个仓库**。
   - 第一个用于训练的数据集很快就会准备好，随后还将提供其他数据集。
- **ArXiv 和 GitHub 数据集的许可担忧**：一位成员对 ArXiv 论文数据集使用的 **MIT license** 提出质疑，指出 *每篇论文都有其自己的许可*。
   - 这种担忧也延伸到了 GitHub 数据集，质疑是否仅包含了 **MIT-licensed 仓库**。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1427664611385282581)** (2 messages): 

> `Cloud GPUs, Object Detection` 


- **成员寻求云端 GPU 推荐**：一位成员正在寻求 **cloud GPU 平台** 的推荐，询问其他人使用的平台及其具体功能。
   - 他们正在寻求选择云端 GPU 供应商的建议，但目前尚未收到建议。
- **白色背景下的目标检测困境**：一位成员询问如何在不需要任何预先训练的情况下 **在白色背景中识别物体**。
   - 讨论正在征集此类场景下的 Object Detection 解决方案，但目前尚未给出方案。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

jazzco0151: https://discord.com/api/oauth2/token
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1427380418768015525)** (4 messages): 

> `nanochat course, Andrej Karpathy, LLMs guides` 


- **Karpathy 课程传授知识！**：Andrej Karpathy 发布了一门关于构建 LLMs 的课程，一位成员本周将致力于发布指南和教程，以帮助大家学习这些材料。
   - 如果你正在学习该课程，可以加入 [huggingface.co/nanochat-students](https://huggingface.co/nanochat-students) 组织。
- **HF 组织欢迎 Karpathy 的编码者们**：一个名为 **nanochat-students** 的 Hugging Face 组织已为学习 Karpathy 的 LLM 课程的人员创建。
   - 如果你正在参加该课程，请考虑加入 [huggingface.co/nanochat-students](https://huggingface.co/nanochat-students) 组织。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1427592596879835178)** (4 messages): 

> `Certificate of Completion, Posting too quickly` 


- **证书要求公布！**：要获得结业证书，你需要完成：**Unit 1**、其中一个 **use case assignments** 以及 **final challenge**。
- **慢一点，速度太快了！**：几位用户收到通知称 *他们发布消息的速度可能太快了*，并被要求放慢速度。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1427532077732990997)** (7 条消息): 

> `SOSP in S.Korea, Blackwell GEMM DSL, DSA Efficiency, GPU Programming Trend, vLLM and SGLang Determinism Tests` 


- **SOSP 会议推测**：一名成员询问了关于参加在韩国举办的 **SOSP** (Symposium on Operating Systems Principles) 的情况。
- **Blackwell 的 Stream-K 疑问**：一名成员质疑为什么最新的 **Blackwell GEMM DSLs** 似乎没有使用 **stream-k**，而是选择了完成整个输出 tiles 的 persistent CTAs。
- **DSA 的 Token 选择**：一名成员质疑为什么 **DSA** 在逐 token 选择的情况下依然高效，这与 **NSA 论文**中强调的为了 GPU 效率而进行 blockwise 选择形成对比。
- **GPU 编程是否正在兴起？**：一名成员询问 **GPU programming** 是否是一个持续的趋势，这源于他们对 **Triton/CUDA** 的兴趣以及其 X（原 Twitter）信息流的算法推荐。
- **vLLM 与 SGLang 确定性测试**：考虑到确定性 kernel 和 pointwise 操作，一名成员质疑在 **vLLM** 和 **SGLang** 中进行全 forward pass 确定性测试的必要性，并引用了 [sglang 的测试](https://github.com/sgl-project/sglang/blob/main/python/sglang/test/test_deterministic.py) 和 [vllm 的测试](https://github.com/vllm-project/vllm/blob/main/tests/v1/generation/test_batch_invariance.py)。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1427515555803303987)** (1 条消息): 

> `Triton Kernel, Scalar value casting, Large double values, inf issue` 


- **Triton Kernel 将大 double 值转换为 inf**：一位用户报告称，将一个大的 double 值作为 scalar 传递给 **Triton kernel** 会导致 `inf`，这表明它可能只接受 **int32** 和 **float** 数据类型。
   - 目前尚不清楚这种行为是已知特性还是 **Triton** 当前架构中的限制。
- **需要调查：Triton 中的 Double 到 inf 转换**：报告的在 Triton kernel 中作为 scalars 传递的大 double 值被转换为 `inf` 的问题值得进一步调查，以确定根本原因。
   - 检查 Triton 的 scalar 类型处理和潜在的隐式 casting 机制可能会揭示观察到的行为，并为潜在的变通方法或文档更新提供参考。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1427494435976908860)** (4 条消息): 

> `Threadblock 0 special case, Race Condition Detection with Compute Sanitizer, Warps behavior during cluster sync` 


- **Threadblock 0 特殊情况的推测**：成员们讨论了 **threadblock 0** 是否是一个特殊情况，因为 ID 最低的 threadblock 可能会被首先创建。
   - 该成员质疑这是否仅在 **GPC 中的 SMs** 被不均匀占用时才会成为问题。
- **建议进行竞态条件健全性检查**：一名成员建议运行 `compute-sanitizer --tool racecheck` 来检查 **race conditions**。
   - 关于该建议的结果，没有提供进一步的信息。
- **Cluster Sync Warp 等待状态探究**：一名成员询问了正在等待 **cluster sync** 的活跃 blocks 中的 **warps** 行为。
   - 另一名成员建议使用 [Compiler Explorer](https://godbolt.org/) 检查为 cluster sync 生成的 **PTX 和 SASS 代码**，以了解具体的实现，并推测它可能涉及 looping、polling 一个全局 atomic 变量以及使用 `nanosleep`。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1427539238726930452)** (5 条消息): 

> `PyTorch, Matrix Multiplication, CPU implementation, MKL` 


- **寻求 PyTorch 矩阵乘法深度解析**：一名成员寻求 **PyTorch** 中矩阵乘法的详细 CPU 实现，追踪到了 `aten/src/ATen/native/LinearAlgebra.cpp`，但难以找到 `at::mm()` 或 `bmm()` 的 dispatch 逻辑。
   - 尽管由于架构变化，**2020年**的一个旧回答被认为已失效，但一名成员指出，并没有单一的实现，它取决于 backend，例如[这个直接调用 MKL 的实现](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mkl/LinearAlgebra.cpp#L55)。
- **PyTorch 中到处都是 MKL 调用**：据一名成员称，**PyTorch** 在进行矩阵乘法时直接调用 **MKL**。
   - 他们建议在 *cpu backends* 源码中查看，这类调用随处可见。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1427380287595348058)** (12 messages🔥): 

> `Matrix Multiplication Blog, Compiler Optimizations for GPUs, GPU programming starting point, Sites similar to leet gpu, Pearson Correlation kernel` 


- **推荐 Aleksa Gordic 关于矩阵乘法的博客**：一位成员推荐了 [Aleksa Gordic 的博客](https://www.aleksagordic.com/blog/matmul)，其中关于 **matrix multiplication**（矩阵乘法）的内容非常有见地。
   - 在博文中，Aleksa 涵盖了背景知识、符号表示、基础算法、单次传递算法（single-pass algorithm）以及其他变体。
- **编译器优化在 GPU 工作中优于 ML**：一位成员询问从事 GPU 相关工作是否需要 ML 知识。
   - 另一位成员澄清说，**ML 知识对于 GPU 编译器优化并非必不可少**，因为重点在于基础编程技能而非 AI 特定知识，*这些案例之所以关注 AI 只是因为它是现在的热点*。
- **大一新生寻求 GPU 编程指导**：一名具有 **Java, Python 和 SQL** 经验的大学生正在寻求从何处开始学习 GPU 编程的指导。
   - 成员指向了频道 <#1198358627594023014>，并建议学习计算机体系结构和 OS。
- **“Leet GPU” 替代方案受到关注**：一位成员询问是否有类似 **Leet GPU** 的网站，用于访问 **H100** 资源和潜在的 TMA 使用。
   - 另一位成员推荐了 **Tensara** 和 **GPU Mode Kernelbot** ([gpumode.com](https://gpumode.com/))，并强调了 GPU Mode 对具有挑战性的竞赛问题的关注。
- **Pearson Correlation Kernel 调试**：一位成员正在为 PPC 在线课程作业编写他们的第一个 Kernel，计算给定矩阵行之间的 **Pearson correlation**（皮尔逊相关系数）。
   - 他们在 CPU 实现中遇到了错误，偏差比预期大约 1153 倍，目前正在寻求识别问题的反馈。


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1427416959884198062)** (1 messages): 

> `Pallas:MGPU, NVLINK comms with local compute, all-gather collective matmul` 


- **使用 Pallas:MGPU 提升 GPU 通信效率！**：一篇关于使用 **Pallas:MGPU** 提升 GPU 计算/通信重叠（overlap）的新教程已发布在 [docs.jax.dev](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)。
   - 根据链接中的推文，只需对 **Pallas:MGPU matmul kernel** 进行少量修改，即可将其转换为 **all-gather collective matmul**，从而实现 **NVLINK comms** 与本地计算的重叠。
- **Pallas MGPU 重叠 NVLINK 通信**：根据推文，对 **Pallas:MGPU matmul kernel** 进行一些改动即可实现 **all-gather collective matmul**。
   - 这种新的 all-gather 集合矩阵乘法实现了 **NVLINK comms** 与本地计算的重叠。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1427795632881664093)** (1 messages): 

> `Multi-node kernel hackathon` 


- **提议举办多节点 Kernel 黑客松！**：一位成员宣布他们增加了一个关于 **multi-node kernel hackathon** 的想法。
   - 他们在这条 [Discord 消息](https://discord.com/channels/1189498204333543425/1427732928766545971)中征询大家的兴趣。
- **占位主题**：这是一个为了满足最低项目数量要求的占位主题。
   - 如果有更多细节，可以在此处添加。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1427834020510564452)** (3 messages): 

> `Crescent Island, LPDDR5X, Xe3P` 


- **Crescent Island 计划于 2026 年下半年发布**：根据 [Intel 新闻中心](https://newsroom.intel.com/artificial-intelligence/intel-to-expand-ai-accelerator-portfolio-with-new-gpu)的消息，Intel 计划在 2026 年下半年推出 **Crescent Island**，作为一款新的 GPU 产品，它将配备 **160 GB 的 LPDDR5X** 内存。
- **Crescent Island 揭示了宽内存接口**：概念渲染图显示 **Crescent Island** 包含数十个 **LPDDR5X 控制器**，这意味着它拥有约 **640-bits** 或两倍于此的宽内存接口。
- **Xe3P 细节披露**：该架构暗示了 4 个 slice，每个 slice 包含 8 个 subslice，总计 **32 个 subslice**。**Xe3P** 可能会增加到每 slice 8 个 subslice，或者涉及更激进地移除固定功能（fixed function）、光线追踪（raytracing）和编解码（codec）流水线。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1427791177133981811)** (1 messages): 

> `AlphaFold 3, MegaFold` 


- **MegaFold 优化 AlphaFold 3**：一个研究小组开源了 **MegaFold**，这是一个针对 **AlphaFold 3 (AF-3)** 的训练平台，指出其与同等规模的 Transformer 相比速度较慢，并为此撰写了一篇 [博客文章](https://news.ycombinator.com/item?id=45585528)。
   - 他们的分析确定了性能/内存瓶颈，并提出了诸如 **Triton 中的自定义算子**和**系统级数据加载**等优化方案，以提升性能并降低峰值内存使用。
- **MegaFold = 更快的 AF3 训练**：**MegaFold** 是一个训练库，可提高运行时性能并减少峰值内存消耗。
   - 据作者介绍，它包含用 **Triton** 编写的自定义算子。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1427742662638960640)** (1 messages): 

> `Agent Hacking, Kernelbench v0.1, Sakana Paper Removal` 


- **Kernelbench 发布关于 Agent Hacking 的精彩讨论**：一位成员建议讨论 Agent Hacking，并引用了 **Kernelbench v0.1** 的博客文章作为讨论点。
   - 该博客文章包含了与 Agent Hacking 技术和漏洞相关的详细信息和示例。
- **Sakana 论文凭空消失**：一位成员注意到 **Sakana** 撤下了原始论文，导致其无法访问且无法引用。
   - 这一举动影响了引用或验证论文中所呈现信息的能力。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1427445762178027540)** (31 messages🔥): 

> `MI300x8 Leaderboard Updates, amd-all2all performance, amd-gemm-rs benchmarks, amd-ag-gemm submissions` 


- **MI300x8 刷新速度纪录**：一位用户在 `amd-all2all` Leaderboard 上获得**第一名**，提交 ID 为 `63561`，在 **MI300x8** 上的耗时仅为 **216 µs**。
- **amd-gemm-rs 在 MI300x8 上全速运行**：在 **MI300x8** 上的 `amd-gemm-rs` Leaderboard 中有多次成功提交，耗时从 **533 µs** 到 **572 µs** 不等。
- **amd-ag-gemm 在 MI300x8 上实现加速突破**：在 **MI300x8** 上的 `amd-ag-gemm` Leaderboard 中有多次成功提交，耗时跨度从 **384 µs** 到 **1201 µs**。
- **amd-all2all 公布全场平均成绩**：在 **MI300x8** 的 `amd-all2all` Leaderboard 上记录了其他成功提交，耗时分别为 **339 µs**、**368 µs** 和 **3.45 ms**。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1427645789655535759)** (5 messages): 

> `Leaderboard Deadline, PST vs UTC, Time Discrepancies` 


- **Leaderboard 截止日期之争：PST 还是 UTC？**：成员们就 Leaderboard 截止日期倒计时是 **PST** 还是 **UTC** 展开了辩论。
   - 一位成员表示是 **PST** 并且他们正在进行修正，而另一位成员则表示 Leaderboard 截止日期倒计时是根据 **UTC** 计算的。
- **时区技术难题！**：一位成员提到在 Google 搜索当前的 **PST** 与 **UTC** 时间时存在差异。
   - 他们开玩笑说 *ChatGPT 告诉我的时间跟 Google 不一样，哈哈，不知道该信谁*。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1427374081447297075)** (11 messages🔥): 

> `MI300x Access, Competition Runners, HotAisle's Offer` 


- **为竞赛提供 MI300x 访问权限**：一位成员通过 [hotaisle.app](https://hotaisle.app) 为竞赛提供了 **8x MI300x VM**（以及 **1x**、**2x** 和 **4x**）的访问权限，并指出他们的算力曾用于第一次竞赛。
   - 该服务是*先到先得、自助服务，支持信用卡或加密货币支付，然后通过 SSH 访问*。
- **竞赛面临 Runner 短缺，提议 HotAisle 赞助**：竞赛将在大约一天后结束，目前的问题是 Runner 数量超过了我们基础设施常规排队处理的能力。
   - 针对 [HotAisle 的提议](https://hotaisle.app)，一位成员建议 *为我们赞助一个节点，以便我们可以继续将 AMD 作为目标，然后我们可以进一步详谈*。
- **服务不可用的困扰**：一位成员报告了“服务不可用”错误，并在 **Hotaisle 的 8xMI300x 节点**上尝试运行其代码。
   - 该成员指出 *我待处理的提交运行正常。所以就这样吧，下次如果队列成了阻碍再说*。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1427371271141851320)** (3 messages): 

> `MoE, GEMV, Qwen3 变体` 


- **针对 MoE Prefill 的分组 GEMV？**: 一位成员正在研究 prefill，并请求在 **MoE** 中使用分组 **GEMV** 的代码示例，因为某些 expert 的 M 维度甚至可能超过 2000，因此无法直接使用 GEMV。
   - 另一位成员指向了[他在 LinkedIn 上的帖子](https://www.linkedin.com/posts/hicham-badri_simple-moe-gemv-triton-kernel-to-accelerate-activity-7363488039404265474-9cdp?utm_source=share&utm_medium=member_desktop&rcm=ACoAABhY_pkBlNdftgiVH4FIgMaHtlg0FefRlbo)，并表示他正在 **Qwen3** 变体中使用一种更好的实现，并将在未来几天内将其发布在 *gemlite* 中。
- **MoE 权重加载**: 讨论涉及从大型组合权重矩阵中加载正确的向量，并设置网格为 `(num_active_experts, M, cdiv(N, BLOCK_SIZE_N))`。 
   - 在重构之后，用于 Qwen3 变体的更高级实现将在 *gemlite* 中提供。


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1427693222381223967)** (6 messages): 

> `Python/Rust 互操作, OpenCL kernel, Autograd 和 backward kernel, 正确性与速度测试, 结合 gpumode compute 的 SITP/picograd` 


- **Picograd 优先考虑 Python/Rust 互操作**: Picograd 正在优先处理 **Python/Rust 互操作**，包括 python->rust 入口以及用于 Triton kernel 的 rust->python。
   - 此外，他们未来几周的优先事项将是搭建架构、在 **OpenCL (rust crate) 和 Triton** 中实现一些基础的 **逐元素前向 kernel**、Autograd、以及 backward kernel，测试正确性和速度，并使用 tinygrad 的前端和 uop IR，支持 eager 和 graph 两种评估模式。
- **通过极简 CI 启动 OpenCL Kernel**: **设备 kernel 正在通过运行构建的极简 CI 在 OpenCL 上启动**。
   - 一位成员表示，*既然 master 分支不再混乱，现在是开始贡献的好时机*。
- **将 SITP/picograd 正确性测试作为自动评分器**: 成员们讨论了让 **SITP/picograd** 由 gpumode compute 提供支持，这样以 NumPy、tinygrad 和 PyTorch 为基准（oracle）的正确性测试就可以充当自动评分器。
   - 性能测试可以作为一个**开放排行榜**，看谁能开发出最快的端到端 minitorch 来训练 nanogpt。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1427382602372874372)** (15 messages🔥): 

> `GPU Mode 的 VSCode 扩展, GPU Mode 网站上的过时文档, 向 PMPP v2 提交 kernel, reference-kernels 仓库中的 Bug, 自选工作组角色` 


- **教程困扰用户：建议开发 VSCode 扩展**: 一位成员无法按照 GPU Mode 网站上的教程进行操作，因为他们不理解其中的问题，并建议 **VSCode 扩展** 将是最简单的选择。
   - 另一位成员表示赞同，说道：*“我不理解这些问题，哈哈”*。
- **文档需要尽职的维护！**: 一位成员指出，[GPU Mode 网站上关于提交 kernel 的教程](https://gpu-mode.github.io/discord-cluster-manager/docs/submitting-your-first-kernel/python-submissions)已经过时，因为引用的 **python 和 cuda 身份** 已不再出现在排行榜列表中。
   - 维护者对过时的文档表示歉意，链接了 [reference kernels 仓库](https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp_v2/grayscale_py/submission.py)，并欢迎社区贡献力量进行清理。
- **前缀和（Prefix Sum）难题困扰 PMMP V2**: 一位成员报告在向 PMPP v2 提交 **prefixsum_v2** 时遇到 `AttributeError`，追溯到 [reference-kernels 仓库](https://github.com/gpu-mode/reference-kernels/blob/74ccfd902ddb846d5d34f1dd8d89fecb97e8b866/problems/pmpp_v2/prefixsum_py/reference.py#L39)中的一个 Bug。
   - 修复方法涉及将 `n = data.numel()` 更改为 `n = data[0].numel()` 以解决 **tuple object** 错误，该成员自愿创建 Pull Request。
- **工作组角色分配：在哪里？**: 一位成员询问如何**自选工作组角色**并附上了一张[截图](https://cdn.discordapp.com/attachments/1394753097989099640/1427732269610701043/Screenshot_2025-10-14_at_2.58.07_PM.png?ex=68efeecb&is=68ee9d4b&hm=03d7f62e5c2ebc85a3c5264e47483545871f8d7d9b78d562c42e831559aab15f&)。
   - 另一位成员引导他们前往指定频道中的 **Carl bot**。


  

---

### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1427808399982202973)** (9 messages🔥): 

> `Multi-GPU Systems, HPC Research, Data Movement, Latency and Bandwidth` 


- **Multi-GPU 系统在 HPC 中获得关注**：一位成员询问了 **multi-GPU 系统** 在 **High-Performance Computing (HPC)** 中的普及程度，并得到了肯定的回复以及一篇 [研究论文](https://arxiv.org/abs/2509.21527) 的链接。
   - 该论文研究了与 multi-GPU 设置相关的架构和性能指标，回答了用户的初始问题。
- **数据移动成为 HPC 研究热点**：一位成员对基于 **multi-GPU 的 HPC 系统** 内部 **数据移动 (data movement)** 相关的研究机会表示了兴趣，特别是关注 **延迟 (latency)** 和 **带宽 (bandwidth)**。
- **关于 HPC 数据传输中延迟与带宽的研究层出不穷**：一位成员确认，许多研究人员正在积极探索 **multi-GPU HPC 系统** 中 **数据传输** 的 **延迟** 和 **带宽** 挑战。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1427397810709463110)** (5 messages): 

> `Helion Contributions, GPU Mode Talk` 


- **Helion 欢迎贡献**：一位成员表示有兴趣为 **Helion** 做出贡献，特别是与其研究应用相关的部分，并引用了 [这个 GitHub issue](https://github.com/pytorch/helion/issues/420)。
   - 他们受到了鼓励，并被承诺会在 GitHub issue 上被标记并提供一些想法。
- **GPU Mode 演讲即将到来**：GPU Mode 演讲将于 **10 月 17 日下午 1:30 (PST)** 举行，内容包括 **Helion** 的概述及演示，观看地址为 [YouTube 链接](https://www.youtube.com/watch?v=1zKvCLuvUYc)。
   - 鼓励观众提出大量问题。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1427430840152883271)** (96 messages🔥🔥): 

> `Veo Model Annotation, Qwen VL Model inference, SAM 3 Model, DGX Spark, DeMO optimizer` 


- **Veo 模型标注探讨**：成员们讨论了 **Google** 如何为其 **Veo 模型** 标注视频，考虑了音频到时间同步帧、timeline JSON 映射、元数据和视频分辨率等方面；另一位成员指出了 [Google 的文档](https://cloud.google.com/vertex-ai/generative-ai/docs/video/video-gen-prompt-guide)。
- **DGX Spark 价格是 Ryzen 的两倍**：根据 [Elon Musk 的推文](https://x.com/elonmusk/status/1978004040090112415?t=ef9sive20cd2VyWzvXEJ7g&s=19)，**DGX Spark** 已经发布，但其价格是 **Ryzen 芯片** 的两倍。
   - 它的体积也很小，这是因为公司希望防止蚕食其更大的产品线，并为 **GB300** 提供预热。
- **Psyche 使用 DeMO 优化器**：**DeMO 优化器** 代码已发布九个月（[GitHub 链接](https://github.com/bloc97/DeMo/)），**Psyche** 将其用于去中心化训练，所有开发进展都在 <#1365697079535468596> 频道中追踪。
   - 一位成员链接了 [PsycheFoundation/psyche](https://github.com/PsycheFoundation/psyche)，认为这是一个值得关注的优秀代码库。
- **Strix Halo 对比 DGX**：一位成员预订了 **DGX**，但他已经拥有了一台 **Strix Halo** (**HP Z2 Mini G1a 395+**，配有 128GB 内存)，由于忙于其他项目，主要进行推理工作，目前正在纠结是否将其加入收藏，并好奇其性能表现。
   - **DGX** 拥有 **273GB/s** 的内存带宽，而 **RTX5090** 在 32GB 上可以达到 **1TB/s**，但一旦溢出到 RAM，速度将慢几个数量级，不过 **5090** 在任何时候都能击败 **Spark**。
- **G1a Amex 商务版超值优惠**：一位成员提到，由于 **HP.com** 上的 **Amex Business** 满 **$1999** 减 **$300** 优惠，他又买了一台 **G1a**，并指出正常 MSRP 约为 **$5k**，目前正在考虑通过 2 个 **Thunderbolt-4**（每个 **40Gbps**）端口将它们组合成集群。
   - 另一位成员分享了一则 [推文](https://x.com/petergostev/status/1978230978725507108?s=46)，看起来与其 CPU 相比并不那么吸引人，纯粹是为了“炫技”。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1427373539010285588)** (4 messages): 

> `Rage-bait attractor, Gemini's response` 


- **愤怒诱饵聊天机器人？**：一位成员询问聊天机器人中是否可能出现“愤怒诱饵吸引器 (rage-bait attractor)”，类似于人们沉迷于让他们感到愤怒的新闻。
- **Gemini 开了个玩笑**：在另一位成员请求论文引用后，一位成员澄清之前的回复来自 **Gemini**，而非研究论文。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1427507806646304828)** (1 messages): 

> `arxiv 2410.10450, model setup difficulty, good repo for llama` 


- **Arxiv 论文 2410.10450 受到关注**：一位成员询问了关于 [Arxiv 论文 2410.10450](https://arxiv.org/abs/2410.10450) 的情况，质疑为什么它没有被更广泛地采用。
   - 他们最初推测为其设置新模型可能太难，但随后澄清该 **repository** 制作精良，并包含一个非常有用的 **Llama** 示例脚本。
- **模型设置难度误区被消除**：最初被认为具有挑战性，但后来发现为 [Arxiv 论文 2410.10450](https://arxiv.org/abs/2410.10450) 设置新模型其实非常简单。
   - 该成员强调该 **repository** 包含一个有效的 **Llama** 示例脚本，简化了流程。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1427507806646304828)** (1 messages): 

> `arXiv Paper Discussion, Model Setup Difficulty, Helpful Repository` 


- **ArXiv 论文引发好奇**：一位成员询问了 [arXiv 论文 2410.10450](https://arxiv.org/abs/2410.10450)，质疑为什么它没有获得更广泛的关注。
   - 他们想知道它是否已被取代，或者设置的复杂性是否是一个阻碍因素。
- **模型设置比预期更容易**：同一位成员后来澄清，为该论文设置新模型实际上并不困难。
   - 他们称赞了该 **repository** 的质量以及其中包含的有用 **Llama** 示例脚本。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1427402339928768695)** (6 messages): 

> `ARM Linux support, DGX Spark compatibility, Mojo on Jetson Orin Nano` 


- ****Streaming** 工程师寻求 **AI** 冒险**：一位在 **Red5 Pro**、**Zixi** 等公司拥有经验的视频流媒体工程师正在寻找涉及现代 **AI** 技术（如 **AI agents**、**AI 视频生成**和 **AI 驱动的视频聊天**）的工作机会。
- **ARM Linux 支持状态查询**：一位用户询问了支持 **ARM Linux** 的计划，特别是针对拥有 **ARM CPU** 的 **DGX Spark**。
   - 一位成员回复说它应该已经可以工作了，但如果由于 **Nvidia** 对 **DGX OS** 的定制而出现问题，请提交 bug。
- **DGX Spark 需要特定更新**：对于 **DGX Spark**，需要添加一个 `sm_121` 条目，并且需要将 `libnvptxcompiler` 更新到 **CUDA 13**。
   - 一旦这些更新到位， **Mojo** 和 **MAX** 应该能在 **DGX Spark** 和 **Jetson Thor** 上完全运行；其他 **ARM Linux** 设备（如 **Jetson Orin Nano**）目前应该可以正常工作。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1427526123847155753)** (71 messages🔥🔥): 

> `SCTP vs QUIC, WebRTC datachannels, Mojo testing framework deprecation, Mojo type reflection, Iroh cross platform` 


- ****SCTP vs QUIC：开发者讨论网络协议****：一位开发者质疑为什么在 **SCTP** 存在的情况下 **QUIC** 如此流行，并提到了它解决队头阻塞（head-of-line blocking）问题的能力及其内置加密。
   - 他们链接到了自己的 [blog post](https://pion.ly/blog/making-a-game-with-pion/)，详细介绍了他们在 **SCTP** 和 **WebRTC datachannels** 方面的经验，并对 Microsoft 缺乏对 **SCTP** 的支持表示沮丧。
- ****WebRTC 游戏开发者使用 Go 实现跨平台****：一位开发者分享了他们的 [public repo](https://github.com/yohimik/webxash3d-fwgs) 和一款名为 [Hypersomnia](https://hypersomnia.io/) 的游戏，该游戏在浏览器中运行，使用 **SCTP/WebRTC datachannels** 实现 Steam Deck 和浏览器之间的跨平台对战，全部采用 C++ 编写。
   - 另一位开发者指出，由于高带宽需求，**QUIC** 几乎不可能进行硬件加速。
- ****Mojo 'test' 弃用提案****：Mojo 团队发布了一项 [proposal](https://forum.modular.com/t/proposal-deprecating-mojo-test/2371)，提议弃用 `mojo test`，转而使用一个新的基于 Mojo 的测试框架。
   - 团队正在征求对该提案的反馈。
- ****Mojo 中的类型反射（Type Reflection）仍需改进****：一位开发者询问如何在 Mojo 中打印变量的类型，类似于 Python 的 `type(a)` 函数，因为 `typeof(greeting)` 不起作用。
   - 另一位开发者指出可以使用 `from compile.reflection import get_type_name`，并提到“我们完全需要它能够直接被打印”以及“我们真的需要一个 type 类型”。
- ****内联函数可能导致代码膨胀****：一位开发者发现，内联带有大型展开循环（unrolled loops）的函数会显著增加二进制文件的大小和构建时间。
   - 他们发现移除内联后，二进制文件大小从 **7.5mb** 降至 **1.5mb**。另一位成员补充说，对于 `@parameter for i in _` 使用 `@always_inline` 将无条件地内联/展开循环。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1427657435232862289)** (1 messages): 

> `TorchAX, Pure Python Custom Devices, JAX device in Torch` 


- **TorchAX 为 Python 自定义设备打开大门**：[TorchAX](https://github.com/google/torchax) 库为纯 Python 自定义设备铺平了道路。
   - 得益于这个包，现在 **Torch** 中可以使用 "jax" 设备。
- **JAX 设备通过 TorchAX 落地 PyTorch**：**TorchAX** 库促进了在 **PyTorch** 框架内创建 'jax' 设备。
   - 这种集成使得在 PyTorch 中使用纯 Python 自定义设备成为可能，扩展了其灵活性和功能。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1427381845972090962)** (69 条消息🔥🔥): 

> `Salesforce Agent 脚本编写, 类似 Devin 的 Agentic 平台, 搭载 Jules 的 Google Gemini 3, Nvidia DGX Spark, Anthropic 深化与 Salesforce 的合作伙伴关系` 


- **Salesforce 拥抱 Agent 脚本编写**：Salesforce 在 prompt 中引入了一种脚本/模板语言，以赋予用户更具确定性的控制权，详见其 [博客文章](https://developer.salesforce.com/blogs/2025/10/introducing-hybrid-reasoning-with-agent-script)。
   - 这种方法旨在让用户对 Agent 行为拥有更强的指挥权。
- **Devin 被推崇为 Agentic 平台**：在关于远程构建的 Agentic 平台的讨论中，**Devin** 被推荐为一个合适的选择，用于下达指令并让其在循环中永久地进行远程构建。
   - 该推荐是针对寻求 Claude Code 替代方案以执行更多自主开发任务的咨询而提出的。
- **期待搭载 Jules 的 Google Gemini 3**：有人建议等待 **Google Gemini 3**，预计它将搭载 **Jules**，作为一个潜在的 Agentic 平台。
   - 这一建议暗示了对 Google 在 Agentic 平台领域先进能力的期待。
- **Nvidia DGX Spark 带宽受限**：**Nvidia DGX Spark** mini-PC 被认为“出货即过时”，原因是带宽限制，尽管有些人认为它作为开发工作站仍有一席之地。
   - 售价 [$4k 的 Nvidia DGX Spark](https://www.pcmag.com/news/nvidia-to-start-selling-3999-dgx-spark-mini-pc-this-week) 的早期基准测试显示，在 gpt-oss-120b-fp4 上仅有约 11 t/s，远低于售价 $4.8k 的 M4 Max MacBook Pro（达到 66 t/s）；社区将其归咎于较低的 LPDDR5X 带宽（273 GB/s 对比 546 GB/s），并宣称该设备对于纯推理任务来说定价过高。
- **ReductoAI 获得 7500 万美元 B 轮融资**：[ReductoAI](https://xcancel.com/aditabrm/status/1978129711898431935) 在文档处理量增长 6 倍、累计为客户处理超过 10 亿页文档后，完成了由 a16z 领投的 **7500 万美元 B 轮**融资。
   - 这笔资金将加速模型研发和新产品功能开发，包括重大的准确性提升和可定制的 pipeline。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1427855696627503235)** (1 条消息): 

> `AI 自由职业, 模型 Fine-Tuning, LLM Infra, AI 初创公司, AI Agent 开发` 


- **年轻的 AI 爱好者寻求合作者**：一位 **17 岁** 的成员正在寻找同龄人，共同在 **AI** 领域进行构建，特别是 **模型 Fine-Tuning**、**LLM Infra** 和 **AI 初创公司** 等领域。
   - 该成员还经营着一家小型 **AI 自由职业业务**，并乐于与志同道合的人分享项目和想法。
- **年轻自由职业者寻求联系！**：一位 **17 岁** 的 AI 自由职业者寻求与对 **AI Agent 开发** 和 **LLM Infrastructure** 充满热情的其他年轻人建立联系。
   - 这位自由职业者希望与对 **Fine-Tuning** 和 **AI 初创公司** 感兴趣的人分享 **项目和想法**。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1427376401648058398)** (35 条消息🔥): 

> `华东师范大学 AI 征稿, State of AI 2025 报告, Cursor AI 代码编辑器, DGX Spark 可用性, RTX 5090 vs DGX Spark` 


- **华东师范大学强制要求 AI 论文**：华东师范大学发布了[征稿通知](https://ed.ecnu.edu.cn/edenglish/fa/a9/c48385a719529/page.htm)，要求提交的论文必须由 **AI 署名 (AI authorship)**。
   - 这一大胆举措标志着向自动化研究的转变，尽管可能仍需人类研究人员的监督。
- **State of AI 2025 报告发布**：[State of AI 2025 报告](https://www.stateof.ai/)已发布，提供了对人工智能领域的预测和分析。
   - 未讨论关于该报告的情绪或观点。
- **DGX Spark 亮相引发辩论**：**DGX Spark** 现已上市，引发了关于其与 **RTX 5090** 等替代方案性价比的讨论。
   - 一位成员指出，以一台 **DGX Spark** 的价格可以购买 *2块 RTX 5090* 显卡，但需要注意的是 **DGX Spark** 提供了*无缝的 CPU-GPU 统一内存 (unified memory)*。
- **Cursor 编辑器失宠**：一位成员分享了使用 **Cursor** 代码编辑器 *3 天*后的负面体验，称其已被丢进*垃圾桶*。
   - 他们警告说，在使用此类工具时，如果没有预设停止标准，不要被其*卷入*或感到*情绪耗竭*。
- **Codex 扩展减少 VSCode 代码冲突**：一位成员发现 **VSCode Codex 扩展** 表现尚可，并提到配合 **GPT Plus** 订阅时其机会成本较低。
   - 它被认为在*重格式化代码 / 生成测试用例*方面表现合理，并能够处理涉及 UI、后端和网站任务的多个小型项目。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1427643929221927065)** (12 条消息🔥): 

> `Cursor 评价, SEAL 论文, Tab 补全与 Agentic Coding, 多智能体系统 (Multi-Agent Systems), 用于编程的 AI 补全` 


- **Cursor 编辑器面临严厉批评**：在使用 [Cursor 编辑器](https://cursor.sh/) 3 天后，一位用户将其弃用，指出其*即将烟消云散 (vaporization imminent)*，并鼓励他人在尝试新工具时设定停止标准，以避免被*卷入或情绪耗竭*。
   - 该用户建议：“这是误导，如果你真的尝试它，不要被这些‘工具’不断的失败和反复所卷入，也不要因此感到情绪耗竭，在使用前为自己设定一个预先指定的停止标准！”
- **SEAL 论文更新**：一位用户分享了 **SEAL 论文** 在 ArXiv 上的最新更新链接，标题为 [SEAL: Secure and Efficient Asymmetric Learned Hashing](https://arxiv.org/abs/2506.10943)。
   - 该用户指出该论文是开源的，可在 [GitHub](https://github.com/Continual-Intelligence/SEAL) 上获取，并且*看起来很有趣*。
- **Agentic Coding 采用加速**：一位用户表示，不采用 **Tab 补全 (tab completion)** 和 **Agentic Coding** 的开发者正在落后，将其比作*对着汽车大喊大叫的骑马者*。
   - 该用户分享了 [arxiv.org/abs/2510.01279](https://www.arxiv.org/abs/2510.01279) 的链接，这是一篇讨论相关主题的论文。
- **AI 补全的实用性引发辩论**：一位用户认为 **AI 补全 (AI completions)** 在编程中的实用性很大程度上取决于工作类型，并指出由于它们经常*极其愚蠢 (fucking stupid)*，反而会减慢进度。
   - 该用户补充说，这些工具的帮助程度取决于一个人平均编写多少**样板代码 (boilerplate)**。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/)** (1 条消息): 

erkinalp: https://clockss.org/
  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1427381556321849497)** (37 messages🔥): 

> `Kimi 团队联系方式，Trickle vibe coding 网站，Aspen 在比特币上 100 倍杠杆，Gemini 对比 GPT5` 


- **Kimi 团队联系方式分享**：一位用户建议联系 Kimi Team 的 <@547736322568355861> 以获取一些酷炫的 one-shot 内容。
   - 他们还分享了一个用于商务或市场合作的电子邮箱 **zhongweiming@msh.team**，并链接了 [Ilya Sutskever 的推文](https://x.com/ilyasut/status/1977971110923968656?s=46&t=_NtP_RUn04yF_4hD_VEDkQ) 以及 [Kimi Moonshot 的最新推文](https://x.com/kimi_moonshot/status/1978047376914080127?s=46&t=_NtP_RUn04yF_4yF_4hD_VEDkQ)。
- **Trickle Vibe Coding 网站曝光**：据一名成员称，**Trickle** ([https://trickle.so/](https://trickle.so/)) 是一个类似于 Lovable、Bolt 和 Manus 的 vibe coding 网站。
   - 该成员表示，如果用户打开那条推文，就会知道 **Trickle** 是什么，并进一步声称它是 Google 搜索 trickle 时出现的第一个结果。
- **关于 Aspen 比特币杠杆的指控**：一位用户开玩笑地指责 **Aspen** 在比特币上使用了 **100 倍**杠杆，赚取了 100 万美元利润后辞职，结果在关税新闻发布后被爆仓。
   - 该用户附带了一张带有 *lmaoLOL* 文字的截图。
- **Gemini 与 GPT-5 的对比**：一位用户幽默地评论说，如果他们是一个 AI 模型，他们会比 **Gemini** 更好，但比 **GPT-5** 更差。
   - 他们补充说 **Gemini 2.5** 太老了，而且 *没有人想使用目前状态下的 Gemini*。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1427393630322032710)** (18 messages🔥): 

> `Mixture of Experts, Sliding Window Attention, LM Evaluation Harness, ArXiv Papers 数据集, GitHub Code 2025 数据集` 


- **MoE Transformer 预训练热议**：一名成员询问了关于预训练 **Mixture of Expert (MoE)** transformer 的开源代码。
   - 另一名成员回答说，许多训练库都支持它，包括 [EleutherAI 的 gpt-neox](https://github.com/EleutherAI/gpt-neox)。
- **Sliding Window Attention 问题**：一名成员询问 [gpt-neox](https://github.com/EleutherAI/gpt-neox) 是否支持带有自定义模式的 **sliding window attention**。
   - 回答是肯定的，表示它支持。
- **LM Evaluation Harness 依然有效**：一名成员询问应该使用哪个框架在常用基准测试上评估自定义 LLM，另一名成员链接了 [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)。
   - 该成员最初因其最后一次更新是在两年前而质疑其有效性，但随后意识到自己搞错了。
- **用于代码和科学推理的大规模数据集发布**：一名成员分享了**两个用于代码和科学推理模型的新数据集**：**4.6TB** 的 [ArXiv Papers 数据集](https://huggingface.co/datasets/nick007x/arxiv-papers)和 [GitHub Code 2025 数据集](https://huggingface.co/datasets/nick007x/github-code-2025)。
   - **ArXiv Papers** 数据集被描述为一个海量的科学语料库，非常适合在学术推理上训练模型；而 **GitHub Code 2025** 数据集包含 GitHub 上 star 数超过 2 个的前 100 万个仓库。
- **新的非正式研究社区寻求资源**：一名成员代表 Leo Gao 询问 EleutherAI 是否愿意支持来自斯坦福和 CMU 等机构的一群研究人员和工程师的研究项目。
   - 另一名成员表示有兴趣了解更多具体项目的信息，并探讨 EleutherAI 如何提供支持，建议在 1-2 周内进行后续讨论。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1427425897421733910)** (15 messages🔥): 

> `Less is More: Recursive Reasoning with Tiny Networks, backpropping only the last step of deep recursion, ARC rules, video models based on 3D rendered video clips, REPA` 


- **微型递归模型之谜：为什么最后一步反向传播有效**：成员们对 ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2307.00030) 论文中，在经过 `T-1` 步 `no_grad()` 深层递归后，仅在最后一步进行反向传播（backpropagating）就能提高预测准确率的现象表示疑问。
   - 他们寻求关于仅在 `n` 次 RNN 前向传递中进行反向传播，为何能在进行 `Tn` 次 RNN 前向传递时带来模型改进的见解，引发了关于迭代细化（iterative refinement）背后学习机制的讨论。
- **微型递归模型：问题修复待定**：一名成员正在等待 ["Less is More: Recursive Reasoning with Tiny Networks"](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/issues/15) 相关仓库中一个问题的解决。
   - 这表明人们对复现或理解该论文的研究结果持续关注并面临潜在挑战。
- **伦理训练数据：公平竞争？**：一名成员询问基于 **3D 渲染视频剪辑** 训练视频模型是否违反了任何 [ARC rules](https://en.wikipedia.org/wiki/AI2_Reasoning_Challenge)。
   - 他们还询问了视频模型训练样本的典型质量预期，以及是否允许使用测试集。
- **REPA 在图像生成领域再次出现**：成员们讨论了 **REPA** 在图像生成中的应用，其中一人担心它是一个重要因素，但另一人澄清说，它与原始方法相比差异并不像想象中那么大。
   - 他们分享了使用 **REPA** 的 [SEAL](https://jyopari.github.io/posts/seal🤔) 链接，表明了它在当前语境下的相关性。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1427445083547897997)** (5 messages): 

> `ACE Playbook, AgentLearningEE, StraughterG's X post` 


- **ACE Playbook 在 GitHub 上引起关注**：一名成员在 GitHub 上分享了 [ACE Playbook](https://github.com/jmanhype/ace-playbook)，另一名成员评价道 *"它很棒"*。
- **AgentLearningEE 大受欢迎**：成员们非常喜欢 [AgentLearningEE](https://github.com/jmanhype/AgentLearningEE) 仓库。
- **StraughterG 在 X 上的帖子走红**：一名成员分享了 [StraughterG 的 X 帖子](https://x.com/StraughterG/status/1978126261273694368)。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1427452736332365956)** (14 messages🔥): 

> `Big Paper Tease, CheshireCat 3.0 Release, Neo4j Integration Request, London Meetup?` 


- **重磅论文预告**：一名用户暗示即将发布一篇 **重要论文**，在社区中引发了期待 [通过 X](https://x.com/lateinteraction/status/1977887477328146458)。
   - 请关注后续详情！
- **CheshireCat 3.0 开源框架发布**：一名用户分享了他们的开源项目 **CheshireCat 3.0**，这是一个基于 **LangChain** 和 **Qdrant** 的框架，专为多模态 RAG、多租户聊天机器人和 Agent 工具编排而设计，可在 [GitHub](https://www.github.com/matteocacciola/cheshirecat-core) 上获取。
   - 它具有基于插件的可扩展性和企业级就绪性，文档位于 [Deepwiki](https://deepwiki.com/matteocacciola/cheshirecat-core)。
- **提议在伦敦举办 Databricks x DSPy 见面会**：一名成员询问是否有兴趣在年底前在伦敦举办 **Databricks x DSPy** 见面会。
   - 另一名成员给出了积极回应，表达了参加的兴趣。
- **Graph RAG 需要 Neo4j 集成**：一名用户请求在 **CheshireCat 3.0** 框架中为 **基于图的 RAG (Graph RAG)** 提供 **Neo4j 集成**。
   - 项目创建者欢迎大家为促进这一集成做出贡献。


  

---

### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1427450564664365127)** (17 messages🔥): 

> `MCP Server 实现, 工具调用中的二进制数据支持, 嵌入式资源, 宿主工程 (Host engineering), 映射工具响应的部分内容` 


- **MCP Server 需要二进制数据**：一名成员正在开发一个 **MCP Server** 实现，该实现需要返回二进制数据（例如 **PDF**）作为工具调用的结果。
   - 该成员指出，规范（spec）似乎在工具调用结果的 *type* 字段中仅支持特定的二进制资源子集（图像和音频）。
- **嵌入式资源可能有所帮助**：另一名成员建议在响应中创建一个**嵌入式资源 (embedded resource)**，并使用任何所需的 **MIME types**。
   - 第一名成员指出，他们需要想出一个伪造的 **URI**。
- **OpenAI API 支持 PDF 文件**：一名成员链接了关于 **PDF 文件** 支持的 [OpenAI API 文档](https://platform.openai.com/docs/guides/pdf-files?api-mode=responses#file-urls)。
   - 另一名成员澄清说，该支持可能仅限于用户消息（user messages），而不一定支持工具响应（tool responses）。
- **需要宿主工程 (Host Engineering)**：一名成员表示，模型 API 开箱即并不支持二进制工具结果，因此需要进行**宿主工程 (host engineering)**。
   - 他们表示，如果没有这些工作，即使工程上的变通方法可行，大多数 **MCP client hosts** 也不会支持从工具返回二进制文件。
- **映射工具响应部分**：一名成员表示，宿主可以将工具响应的部分内容映射到模型 API 调用的任何部分，潜在地将具有 **PDF MIME type** 的**嵌入式资源**视为用户上传的文档。
   - 他们指出，某些模型可能会因为工具结果以这种方式映射到用户消息而感到困惑。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1427550364647034950)** (2 messages): 

> `移除 Pylint, 使用 ChatGPT 进行测试重构` 


- **Pylint 的效用引发讨论**：一名成员质疑 **Pylint** 的价值并建议将其移除。
   - 理由是它可能“找不出任何有价值的问题”。
- **ChatGPT 测试重构失败**：一名成员提议使用 **ChatGPT** 将 *test_advancedindex* 重构为具有更有意义名称的测试。
   - 然而，该成员报告称 **ChatGPT 搞砸了**，导致测试失败，最终需要手动重构。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1427532531627982938)** (10 messages🔥): 

> `为 tinygrad 贡献代码, Tensor 缓冲区不可写, 冻结矩阵部分用于训练, 虚拟 Tensor 创建, 访问计算出的梯度` 


- **Hotz 否决首次贡献尝试**：一名成员询问，从 `__setitem__` 中移除 realize 并使 `TestSetitemLoop.test_arange` 成为一个 kernel 是否是一个好的首次贡献尝试。
   - George Hotz 回复说“并不完全是”，他们需要查看所有的 bounties 并决定什么是合理的。
- **Tensor 缓冲区抛出错误**：一名新用户在调用 `my_tensor.numpy()` 时遇到了 *underlying buffer is not writable*（底层缓冲区不可写）的错误。
   - 该用户被要求分享其余代码以诊断问题。
- **矩阵冻结热议**：一名用户想要“冻结矩阵的一部分，只训练另一部分”。
   - 他们正在寻找一种方法来创建一个指向原始 Tensor 数据的虚拟 Tensor（concat），且这些原始 Tensor 具有不同的 `requires_grad` 设置。
- **梯度访问权限？**：一名用户询问如何访问计算出的梯度，以便将反映 Tensor 某一部分的梯度归零。
   - 他们建议了一种变通方法，即存储原始权重并在应用梯度后恢复它们，但这需要更多内存。
- **模拟虚拟 Tensor**：用户建议使用 `Tensor.cat(x @ a.detach(), x @ b, dim=-1)` 来模拟一个“虚拟” Tensor。
   - 这允许将具有不同 `requires_grad` 设置的 Tensor 连接起来。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1427379333718343700)** (11 messages🔥): 

> `Manus 功能, Manus 职位空缺, 社区版主福利, 产品不满与反馈, 每日额度问题` 


- ****Manus 奇迹**：在应用切换倾向中功能性获得赞誉**：一位用户表达了对 **Manus 功能**的欣赏，解释了他们倾向于在不同应用间切换，但依然认可 Manus 的优势。
   - 他们建议 Manus 表现得不要像个“婆婆嘴程序员（mother-in-law coder）”，而应更像一个助手，提供最佳使用指南。
- ****LinkedIn 招聘**：为 Manus 爱好者列出的职位空缺**：[Manus LinkedIn 页面](https://www.linkedin.com/company/manus-im/jobs/)列出了**职位空缺**，由 HR 负责候选人筛选。
   - 社区版主可获得**免费使用权**并可以引导 Prompt，同时欢迎 KOL 分享社交媒体账号进行合作。
- ****反馈前沿**：通过共享会话解决产品不满**：一位社区成员请求经常表达不满的用户分享**失败的会话链接**，以便更好地理解并解决潜在的产品问题。
   - 团队致力于修复产品问题，并提供关于**使用方法和 Prompt 技巧**的指导，以造福社区。
- ****额度紧缺**：用户询问缺失的每日额度**：一位用户询问为何没有收到 **300 每日额度**，在给定的消息中没有更多上下文。
   - 该用户提到了过去的互动，包括分享内容和创建仓库，表明这可能是一个特定于账号的问题。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1427605566867640380)** (5 messages): 

> `aider 别名, OpenCode GLM 4.6` 


- **为聊天模式创建 Aider 别名**：一位用户将 `aider` 设置为 `aider --chat-mode ask` 的别名，表达了希望直接运行 `aider` 而无需 Shell 脚本或命令历史记录的愿望。
   - 尽管在他们的 `.aider.conf.yml` 中设置了 `chat-mode: ask`，他们仍然遇到需要使用 `/ask` 的情况，并考虑将此功能添加到 Aider 中。
- **OpenCode GLM 4.6 的编程体验备受赞誉**：一位用户分享了关于 **OpenCode + GLM 4.6** 的正面反馈，强调了舒适且愉快的编程体验。
   - 他们强调消除了对 **Token 计数**的担忧，并称赞了其易用性，同时配合 `aider.chat` 使用 **Sonnet 4.5** 进行特定微调。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1427476104180011161)** (2 messages): 

> `向长消息添加文件, Aider 工作流技巧` 


- **用户寻求向 Aider 长消息添加文件的建议**：一位用户询问在已经开始撰写长消息后，通过 `/add` 或 `/read-only` 添加文件的最佳方式。
   - 该用户目前的临时方案是将消息复制到 **vim**，添加文件，然后再将内容粘贴回来。
- **发送并 Ctrl+C 作为 Aider 的权宜之计**：用户开玩笑地建议一种工作流：输入消息后点击发送，然后迅速按下 Ctrl+C 以停止发送。
   - 这或许是为了凸显当前操作的困难程度。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1427667352518135952)** (1 messages): 

> `Agentic 工具, Aider 的能力` 


- **Aider 使编辑任务变得简单**：一位成员指出，在日常工作中，使用 **aider** 处理像编辑 **100loc 函数**（100行代码函数）这样的任务是非常简单的。
   - 这一评论是针对有关[推动 Agentic 工具](https://forum.cursor.com/t/why-the-push-for-agentic-when-models-can-barely-follow-a-single-simple-instruction/137154/21)的讨论而发表的，该讨论质疑了在模型连简单的指令都难以遵循时，这些工具的实用性。
- **Agentic 工具受到质疑**：一场讨论对推动 **Agentic 工具**提出了质疑，因为模型目前几乎无法遵循单一的简单指令。
   - 一位成员分享了关于日常工作背景下 Agentic 工具的有趣见解。