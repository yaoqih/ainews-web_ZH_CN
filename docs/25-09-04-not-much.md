---
companies:
- google-deepmind
- hugging-face
- jina-ai
- lighton
- microsoft
- stanford
- openai
- ollama
- weaviate
- langchain
- llamaindex
date: '2025-09-04T05:44:39.731046Z'
description: '以下是为您翻译的中文内容：


  **Google DeepMind** 发布了 **EmbeddingGemma (308M)**，这是一款专为设备端检索增强生成（RAG）和语义搜索优化的小型多语言嵌入模型。它支持
  100 多种语言，在经过量化处理后运行效率极高，在 EdgeTPU 上的延迟低于 15 毫秒。**Jina AI** 推出了全新的代码专用嵌入模型（0.5B/1.5B），支持
  GGUF 量化，在多语言和多任务检索中达到了业界领先水平。**LightOn** 展示了无需蒸馏的大规模检索训练方法，通过对数十亿个段落进行对比训练来实现。


  **Hugging Face** 发布了 **FineVision** 数据集，包含 1730 万张图像和 95 亿个回答 Token，用于视觉语言模型训练，显著提升了基准测试表现。**MiniCPM-V
  4.5 (8B)** 多模态模型据报道在 OpenCompass 基准测试中超越了 **GPT-4o** 和 **Gemini-2.0 Pro**，其采用了创新的视频
  Token 压缩技术。微软的 **VibeVoice TTS**（文本转语音）和斯坦福大学的 **Mixture-of-Contexts** 视频生成技术也同步亮相。


  此外，斯坦福大学的一项研究对 Muon、Soap、Mars 和 Sophia 等优化器进行了基准测试，发现随着模型规模扩大，它们相对于 AdamW 的加速效果会有所减弱，但在小规模模型上具有优势。ChatGPT
  新推出的分支（branching）功能因其简洁性和受欢迎程度而受到关注。最后，文中提到：“现在人人都是‘十角兽’（估值百亿美元的公司）了。”'
id: MjAyNS0w
models:
- embeddinggemma
- qwen-2.5-coder
- minicpm-v-4.5
- gpt-4o
- gemini-2.0-pro
people:
- osanseviero
- _philschmid
- tomaarsen
- ollama
- weaviate_io
- lusxvr
- andimarafioti
- thibaudfrere
- _akhaliq
- clementdelangue
- gordonwetzstein
- konstmish
- wen_kaiyue
- percyliang
title: 今天没发生什么特别的事。
topics:
- embeddings
- retrieval-augmented-generation
- quantization
- multilingual-models
- on-device-ai
- semantic-search
- contrastive-learning
- dataset-release
- vision
- multimodality
- video-generation
- text-to-speech
- optimizer-benchmarking
- training-recipes
- model-compression
- video-token-compression
- fine-tuning
---

**现在人人都是十角兽（decacorn）了。**

> 2025年9月4日至9月5日的 AI 新闻。我们为您查看了 12 个 Reddit 子版块、544 个 Twitter 账号和 22 个 Discord 社区（186 个频道，4350 条消息）。预计节省阅读时间（以 200wpm 计算）：324 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以优美的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

[祝贺 Sierra](https://x.com/SierraPlatform/status/1963654362384724388) 成为最新的~~十边形（Decagon）~~，我是说，十角兽（Decacorn）。

此外，[ChatGPT 的新分支（branching）功能](https://x.com/OpenAI/status/1963697012014215181)非常受欢迎，尽管实现它（使用 Responses API）可能只需要约 100 行代码（LOC）。

---

# AI Twitter 综述

**端侧 Embedding 与检索栈更新**

- **Google 的 EmbeddingGemma (308M) 广泛发布**：Google/DeepMind 发布了一款小型多语言 Embedding 模型，专为端侧 RAG 和语义搜索设计。亮点：308M 参数，在 MTEB 上排名 500M 以下开源模型首位，支持 100 多种语言，量化后运行内存小于 200MB，支持 Matryoshka embeddings（输出维度 768→128），2k 上下文，在某些设置下 EdgeTPU 延迟小于 15ms。生态系统已立即支持 Hugging Face Sentence Transformers, Ollama, MLX, llama.cpp, LlamaIndex, LangChain, Weaviate, Cloudflare Workers 等。发布详情与入门指南：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1963635422698856705), [@osanseviero](https://twitter.com/osanseviero/status/1963635281032040914), [@_philschmid](https://twitter.com/_philschmid/status/1963634786636841461), [@tomaarsen](https://twitter.com/tomaarsen/status/1963639557653422304), [@ollama](https://twitter.com/ollama/status/1963667967184617703), [@weaviate_io](https://twitter.com/weaviate_io/status/1963683200368304613), [@TheTuringPost](https://twitter.com/TheTuringPost/status/1963666849364836606)。
- **Jina 代码 Embedding (0.5B/1.5B) + GGUF**：新的代码专用 Embedding 模型（带有 1-4bit GGUF 量化）声称在 15 种以上语言和 5 个任务（nl2code, code2code, code2nl, code2completions, QA）中达到了 SOTA 检索水平。基于强大的代码 LLM 基础（例如在 5.5T token、92 种以上语言上预训练的 Qwen2.5‑Coder），然后使用有限的对齐样本进行对比微调（contrastively tuned）以优化检索。链接与模型：[@JinaAI_](https://twitter.com/JinaAI_/status/1963637135439007824), [详情](https://twitter.com/JinaAI_/status/1963637139037720995), [模型](https://twitter.com/JinaAI_/status/1963637141675843791)。
- **无需蒸馏的大规模检索训练**：LightOn 的 PyLate 展示了使用 GradCache + 分布式基础设施在数十亿个段落上进行直接对比训练，报告称在 BEIR/BRIGHT 上提高了泛化能力，且无需教师模型。概述：[@LightOnIO](https://twitter.com/LightOnIO/status/1963620040604787136)。

**视觉语言数据与多模态模型**

- **FineVision 数据集 (Hugging Face)**：一个用于 VLM 训练的大型开源数据集发布：包含 1730 万张图像、2430 万个样本、8890 万轮对话、跨越 200 多个精选来源的 95 亿个回答 token。团队报告称在 10 个基准测试中平均增益超过 20%，并增加了新功能（GUI 导航、指向、计数）。公告与技术文章：[@lusxvr](https://twitter.com/lusxvr/status/1963609337546293448), [@andimarafioti](https://twitter.com/andimarafioti/status/1963610118165000479), [@thibaudfrere](https://twitter.com/thibaudfrere/status/1963627540544647177)。
- **MiniCPM‑V 4.5 (8B) 视频/图像 VLM**：报告称 8B 模型在 OpenCompass 的 8 个基准测试中平均得分为 77.0，声称在他们的设置下超越了 GPT‑4o‑latest 和 Gemini‑2.0 Pro。引入了统一的 3D‑Resampler 和激进的视频 token 压缩（96 倍）：将 6×448×448 帧压缩为 64 个视频 token（而许多 MLLM 约为 1,536 个）。演示与 Space：[@_akhaliq](https://twitter.com/_akhaliq/status/1963587749400727980), [@OpenBMB](https://twitter.com/OpenBMB/status/1963623940028563910)。
- 同样值得关注：微软的 VibeVoice TTS 使用 7.5 Hz 的连续语音分词器（continuous speech tokenizers）实现富有表现力的长篇多说话人音频 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1963537036616323388)；斯坦福的 Mixture‑of‑Contexts 展示了单次推理生成分钟级视频的能力 [@GordonWetzstein](https://twitter.com/GordonWetzstein/status/1963583050744250879)。

**优化器、内部指标与训练配方**

- **稳健的优化器基准测试 (Marin 项目)**：两篇论文（以及一项全面的 Stanford 研究）对比了 Muon, Soap, Mars, Sophia, ScheduleFree, AdEMAMix, Prodigy 等在不同模型规模（0.1B–1.2B）、Batch Size 和调度器下的表现。正在形成的共识是：通过精细调优，在更大规模下，相比 AdamW 的加速效果会有所减弱（在 ~1.2B 规模下约为 10%），尽管基于矩阵的方法在较小规模下可能领先。相关讨论：[@konstmish](https://twitter.com/konstmish/status/1963535545721917725), [@wen_kaiyue](https://twitter.com/wen_kaiyue/status/1963633867140526319), [@percyliang](https://twitter.com/percyliang/status/1963648131394122222)，以及来自 [@BlancheMinerva](https://twitter.com/BlancheMinerva/status/1963679442859106480) 和 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1963689424782565384) 的评论。
- **大规模训练中的“内部指标” (Kimi/K2)**：从业者强调监控内部信号（Loss, Grad Norm, Output RMS, Max Logit）以诊断不稳定性并确保余量。MuonClip 的设计旨在控制 Max Logit，以避免训练崩溃。总结与翻译：[@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1963493293679153349), [@crystalsssup](https://twitter.com/crystalsssup/status/1963547955799224386)。
- **Qwen3-32B 的创意写作微调**：“Zhi-Create-Qwen3-32B” 报告其 WritingBench 评分为 82.08，而基座模型为 78.97。该模型使用了：(1) 带有课程学习的 SFT（按长度/推理分组、渐进难度、针对性重训）以及 (2) 结合 RAFT 的 DPO（规则过滤 + LLM 评判），以解决中英混输、重复和推理问题。数据包括过滤后的开源数据集（如 Dolphin-r1, DeepSeek 蒸馏数据）、知乎问答和 CoT 轨迹；所有数据均通过了 Reward Model 过滤。使用技巧包括 Temperature 设为 ~0.6 以及可选的思考触发字符串。详情：[@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1963441300692402659)。
- **基础设施说明**：slime RL 框架报告称将 Qwen3-30B-A3B 的权重更新时间从 60s 缩短至 7s，并在处理 GLM-4.5-355B-A32B FP8 更新时耗时约 100s，目前正在进行异步/零冗余优化。合作邀请：[@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1963532501336695282)。

**Agent 系统、运行时与工具**

- **LangGraph 设计深度解析**：一篇关于构建生产级 Agent 运行时的详尽文章：极简抽象、结构化执行/状态、恢复/持久性，以及符合实际运维需求的控制界面。准备将 Agent 投入生产环境的团队必读：[@LangChainAI](https://twitter.com/LangChainAI/status/1963646974315606428), [@hwchase17](https://twitter.com/hwchase17/status/1963647954587455568), [@nfcampos](https://twitter.com/nfcampos/status/1963652967443435723)。
- **UI-TARS-2（原生 UI 的多轮 Agent RL）**：统一的 GUI/手机/浏览器/终端/工具使用 Agent 在 OSWorld (47.5), WindowsAgentArena (50.6), AndroidWorld (73.3), Online-Mind2Web (88.2%), SWE-Bench (68.7), TerminalBench (45.3) 等基准测试中表现出色；支持结合点击、终端和 API 调用的混合动作流。论文 + Demo：[@TsingYoga](https://twitter.com/TsingYoga/status/1963629621326614940)。
- **Agent 故障分析**：Atla 推出一个平台，可自动发现 Agent 系统中反复出现的失败模式并提出针对性修复建议 [@Atla_AI](https://twitter.com/Atla_AI/status/1963586200305836264)。另外，AgenTracer-8B 可诊断多 Agent 交互错误，并在其设定下报告比私有基座模型高出 18.18% 的提升 [@omarsar0](https://twitter.com/omarsar0/status/1963618829680218254)，[论文](https://twitter.com/omarsar0/status/1963618846532931663)。
- **基础设施更新**：Groq 的 Compound（Agent 系统）在经历 500 万次以上请求后正式发布 (GA) [@GroqInc](https://twitter.com/GroqInc/status/1963635205899710798)。Gradio 现在可以通过单条命令将 MCP 服务器部署到 Google Cloud [@Gradio](https://twitter.com/Gradio/status/1963636954999754955)。HF MCP 服务器增加了 OpenAI Codex CLI 支持 [@reach_vb](https://twitter.com/reach_vb/status/1963599978909008321)。Together AI 增加了欧洲 GPU 区域（瑞典），以实现更低延迟和数据驻留 [@togethercompute](https://twitter.com/togethercompute/status/1963498998720872686)。SkyPilot 展示了从 SLURM 迁移到多云环境，以 K8s 级的可靠性实现更快的迭代周期 [@skypilot_org](https://twitter.com/skypilot_org/status/1963637217055646139)。

**产品发布与生态系统**

- **Perplexity Comet**: 持续大规模推广——单次推送中已有“超过一百万”用户获得访问权限；移动端预订已上线；新的 iOS 应用版本可流畅流式传输表格/Markdown/中间步骤 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1963633205351010795), [预订](https://twitter.com/AravSrinivas/status/1963620578344276366), [iOS 更新](https://twitter.com/AravSrinivas/status/1963758210281882029), [可用性说明](https://twitter.com/perplexity_ai/status/1963638853975040456)。
- **ChatGPT 对话分支**: OpenAI 发布了原生的对话分支与探索功能（branch-and-explore），这是针对探索性工作流的一项长期被要求的 UX 升级 [@OpenAI](https://twitter.com/OpenAI/status/1963697012014215181), [@gdb](https://twitter.com/gdb/status/1963780952187965746)。
- 研究简报：DeepMind 的 Deep Loop Shaping（发表于 Science）改进了 LIGO 干涉仪控制，在硬件上将噪声降低了 30–100 倍，并消除了 LIGO 最不稳定的环路作为主要噪声源——这是 AI 推动实验物理学进步的一个范例 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1963664018515849285), [结果](https://twitter.com/GoogleDeepMind/status/1963664045216579999), [@sundarpichai](https://twitter.com/sundarpichai/status/1963668228481159371)。

**热门推文（按互动量排序）**

- [Ilya Sutskever：“这是我见过的革命性突破”](https://twitter.com/ilyasut/status/1963627458244350015) — 19.2k
- [阿里巴巴 Qwen：“准备好见见 Qwen3 家族中体量最大、最聪明的成员了吗？”](https://twitter.com/Alibaba_Qwen/status/1963586344355053865) — 5.5k
- [OpenAI：“应大众要求：你现在可以在 ChatGPT 中创建对话分支了”](https://twitter.com/OpenAI/status/1963697012014215181) — 17.1k
- [Google Gemini App：用于多图像生成的无提示词 nano-banana 模板](https://twitter.com/GeminiApp/status/1963615829708132611) — 1.7k
- [吴恩达（Andrew Ng）：“对懂 AI 的开发者的需求仍有巨大的缺口……”](https://twitter.com/AndrewYNg/status/1963631698987684272) — 1.8k
- [Perplexity (Arav)：“今天早上有超过一百万人获得了 Comet 的访问权限。”](https://twitter.com/AravSrinivas/status/1963633205351010795) — 1.0k
- [DeepMind：EmbeddingGemma 发布](https://twitter.com/GoogleDeepMind/status/1963635422698856705) — 1.2k

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Microsoft VibeVoice 仓库下架与 ComfyUI 集成

- [**VibeVoice 凉了？你怎么看？**](https://i.redd.it/un6uilkoh2nf1.png) ([Score: 200, Comments: 75](https://www.reddit.com/r/LocalLLaMA/comments/1n7zk45/vibevoice_rip_what_do_you_think/)): **OP 报告称 Microsoft 突然删除了官方 VibeVoice GitHub 仓库，并从 Hugging Face 移除了 VibeVoice-Large 和 VibeVoice-Large-Preview 模型；ModelScope 上仍有镜像。他们维护着 ComfyUI 集成节点 ([Enemyx-net/VibeVoice-ComfyUI](https://github.com/Enemyx-net/VibeVoice-ComfyUI))，并发布了 v**`1.0.9` **版本，直接嵌入了 VibeVoice 以避免目前缺失的上游依赖；该项目采用 MIT 许可协议，这意味着很可能允许重新分发。下架原因尚不明确；该工作似乎与 Microsoft 亚洲研究院（MSRA）有关。** 评论指出 MIT 许可协议允许社区重新上传（例如上传到 Hugging Face），并敦促备份资产以防丢失。其他人推测这遵循了 Microsoft 亚洲实验室项目被撤回的惯例，可能是由于团队变动或人员离职。
    - 许可协议的影响：评论者指出该项目采用 **MIT License**，该协议授予了使用、复制、修改和重新分发现有版本的广泛且不可撤销的权利。这意味着在 [Hugging Face](https://huggingface.co/) 等平台上镜像已发布的版本在法律上是允许的，且任何后续的许可变更都不能追溯限制这些产物（[MIT 文本](https://opensource.org/license/mit/)）。实际建议：备份权重（weights）和代码，以避免因上游下架而造成损失。
    - 预期的重新发布变化：如果下架是为了更新发布，用户预计会增加安全过滤器/“审查”或更严格的使用限制（例如：门控下载、更严格的 AUP 或嵌入式拒绝策略）。这可能会降低某些领域的能力（更高的拒绝率、受限的提示词），因此备份原始 Checkpoint 可以保留一个不受限制的基准，用于评估和下游 finetuning。
    - 先例与韧性：评论者将其与之前的事件（如 WizardLM/Wizard 2）进行了比较，当时强大的 Checkpoint 被发布后又被撤回/限制，但社区镜像依然存在并继续被使用。技术上的启示是优先考虑 open-weight 的可用性，使研究和部署与上游产品或政策逆转脱钩（[WizardLM 仓库背景](https://github.com/nlpxucan/WizardLM)）。
- [**微软下架了 VibeVoice 仓库吗？？**](https://i.redd.it/vsnyimd3e2nf1.png) ([Score: 180, Comments: 36](https://www.reddit.com/r/LocalLLaMA/comments/1n7z5kl/did_m_take_down_vibevoice_repo/)): **该帖子指出官方 Microsoft VibeVoice GitHub 仓库 ([microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)) 现在返回 404 错误，评论者注意到相关的 Hugging Face 模型（VibeVoice-Large 和 VibeVoice-Large-Preview）也被撤回了。社区镜像和工具仍然存在：ComfyUI 节点实现在 https://github.com/Enemyx-net/VibeVoice-ComfyUI，模型文件仍可从 ModelScope 获取：https://modelscope.cn/models/microsoft/VibeVoice-Large/files。 现有的本地安装继续运行；下架原因不明，可能是暂时的，人们担心潜在的许可变更。** 评论推测它“太好用了”，并敦促下载镜像以备后世之需，而其他人则在索要副本，并建议在 Microsoft 的意图和许可明确之前，对重新分发保持谨慎。
    - Microsoft 官方 VibeVoice GitHub 仓库突然被移除，Hugging Face 上的 `VibeVoice-Large` 和 `VibeVoice-Large-Preview` 条目也被下架；`VibeVoice-Large` 权重在 ModelScope 上仍有镜像：https://modelscope.cn/models/microsoft/VibeVoice-Large/files。下架原因不明，引发了对潜在许可变更的担忧，这可能会影响代码/权重的重新分发或嵌入。
    - 在操作层面，现有的设置继续工作，因为 inference 只需要本地权重：“你不需要原始的微软仓库。只要你有权重，你就可以在 Comfy 中使用它们。” 通过社区节点 https://github.com/Enemyx-net/VibeVoice-ComfyUI 实现的 ComfyUI 集成仍然有效，因此已经引用本地 Checkpoint 的工作流不受影响。
    - 并非所有变体都消失了：评论者指出 `1.5` 模型仍在 Hugging Face 上，而 Large 模型可以从 ModelScope 检索。实际上，追求可复现性的用户正在下载并固定剩余的产物，以避免未来链接失效，同时等待状态和许可的进一步明确。

### 2. EmbeddingGemma 300M 发布 + HF Science AMA/FineVision

- [**EmbeddingGemma - 300M 参数，同尺寸下性能领先（state-of-the-art）的 Google 开源嵌入模型**](https://www.reddit.com/r/LocalLLaMA/comments/1n8egxb/embeddinggemma_300m_parameter_stateoftheart_for/) ([Score: 197, Comments: 38](https://www.reddit.com/r/LocalLLaMA/comments/1n8egxb/embeddinggemma_300m_parameter_stateoftheart_for/)): **Google 发布了 EmbeddingGemma，这是一个** `300M` **参数、仅限文本的多语言嵌入模型（在 100 多种语言上训练），可生成** `768` **维向量，并支持通过多分辨率学习 (MRL) 获取更小的维度。权重已发布在 Hugging Face ([google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m))，可通过 Ollama ([library/embeddinggemma](https://ollama.com/library/embeddinggemma)) 部署，发布文章提供了英文和多语言评估，声称在其尺寸下达到了 state-of-the-art 性能 ([HF blog](https://huggingface.co/blog/embeddinggemma))；社区 GGUF 版本 (Q4_0, Q8_0, BF16) 已整合至 [unsloth/embeddinggemma-300m-GGUF](https://huggingface.co/unsloth/embeddinggemma-300m-GGUF) 以供本地推理。许可证：Gemma。** 评论者提到了 HF 博客中关于任务级权衡的对比表，并讨论了在 `nomic-embed-text:v1.5` 与 EmbeddingGemma 之间如何选择，指出选择可能取决于使用场景（单语言 vs 多语言覆盖、延迟/量化需求以及维度）。社区即将推出 RAG 微调和基准 RAG notebook。
    - **部署/量化**：社区发布的 GGUF 版本在一个仓库中捆绑了 EmbeddingGemma-300M 的 `Q4_0`、`Q8_0` 和 `BF16` 构建 (https://huggingface.co/unsloth/embeddinggemma-300m-GGUF)，简化了 llama.cpp/本地使用；`Q4_0` 最小化了 RAM 占用，`Q8_0` 在尺寸与精度/延迟之间取得平衡，而 `BF16` 则保留了最高质量的精度。维护者还计划发布 RAG 微调和基准 notebook，以端到端地评估检索质量。
    - **基准测试**：**Google/Hugging Face** 在官方博客 (https://huggingface.co/blog/embeddinggemma) 中提供了英文和多语言的对比评估，让你可以检查任务级性能（如检索/分类），以验证其“同尺寸下性能领先”的说法。链接中的图表支持与其他开源嵌入模型在不同数据集上的对等比较，这对于模型选择至关重要。
    - **对比**：一位从业者报告称 EmbeddingGemma-300M *“比 qwen 3 0.6b embedding 差不少”*，强调了尺寸（`~300M` 参数）与绝对精度之间相对于更大（`~600M`）模型的权衡。另一位询问关于 `nomic-embed-text:v1.5` 的建议；实际指导是根据目标语言/领域以及博客中每个数据集的评分来选择，而不仅仅看标题上的平均分。

- [**与 Hugging Face Science 团队的 AMA，该团队是 SmolLM, SmolVLM, Fineweb 等项目的幕后推手。**](https://www.reddit.com/r/LocalLLaMA/comments/1n8c3l2/ama_with_hugging_face_science_the_team_behind/) ([Score: 194, Comments: 414](https://www.reddit.com/r/LocalLLaMA/comments/1n8c3l2/ama_with_hugging_face_science_the_team_behind/)): **Hugging Face Science 宣布了一场限时 AMA（太平洋标准时间上午 8-11 点，并有 24 小时后续跟进），参与者包括 SmolLM、SmolVLM 和 FineWeb 背后的研究人员，同时发布了新的多模态数据集 FineVision（参见数据集卡片：https://huggingface.co/datasets/HuggingFaceM4/FineVision）。参考链接：组织页面 https://hf.co/science 和学习资源 https://hf.co/learn。参与者涵盖了模型预训练（如 SmolLM/Nanotron）、后训练/对齐、评估、多模态 (VLM)、数据、Transformers.js 以及 llama.cpp 集成。** 评论者询问了 SmolLM 开发过程中一些违反直觉的设计选择和惊喜，表现出对训练/架构决策的兴趣；生态系统贡献者（如 Unsloth）也参与了支持。
    - 一位评论者询问了 **SmolLM** 开发过程中最大的惊喜——即那些最终奏效的违反直觉的设计选择。技术角度包括分词器 (tokenizer)/词表大小与参数量的权衡、上下文长度与计算预算、通过 **FineWeb/FineWeb-Edu** 进行的数据清洗和课程学习 (curriculum)、优化器/正则化选择 (AdamW/Lion, weight decay, dropout)、注意力/激活变体 (RoPE scaling, GQA, SwiGLU) 以及精度/吞吐量决策 (bf16/fp8, FlashAttention)。他们希望获得具体的消融实验 (ablations) 或指标，展示小模型在哪些*非显而易见*的设置中获益。
    - 另一个帖子询问团队如何确定下一个项目的优先级。标准可能包括公共基准测试（MMLU, GSM8K, MT-Bench）上的差距、像 **FineWeb** 这样的数据流水线对新模态的准备情况、部署的计算/延迟限制（量化, KV-cache, 注意力缩放）以及可复现性与训练成本。这一提问暗示了一个包含里程碑指标以及在 **SmolLM**、**SmolVLM** 和数据集工具之间进行资源分配的决策框架。

- 一位用户询问是否有计划训练并发布更大的 `30B+` 模型。显著的限制因素包括计算预算、数据集规模/质量、Dense 与 MoE 的权衡、训练栈（FSDP/ZeRO、activation checkpointing）、推理成本（显存带宽、并行性），以及证明扩展规模（scaling）优于继续优化小模型所需的评估。他们正在探讨超越 **SmolLM/SmolVLM** 进行扩展的路线图和可行性。

### 3. 本地 AI 运维：5070 Ti Super VRAM 装备与 Ollama 暴露公益公告 (PSA)

- [**终于来了：3090 继任者：5070 Ti Super 24GB 800 美元**](https://www.reddit.com/r/LocalLLaMA/comments/1n82ndz/finally_3090_successor_5070_ti_super_24gb_800/) ([评分: 246, 评论: 140](https://www.reddit.com/r/LocalLLaMA/comments/1n82ndz/finally_3090_successor_5070_ti_super_24gb_800/)): **传闻/泄露声称 NVIDIA “RTX 5070 Ti Super” 拥有 24 GB VRAM，售价约 800 美元，定位为 3090 级别的继任者。理由是其能效比（perf/W）的提升可能使多 GPU（例如约 100 GB 总 VRAM）配置在没有极端功耗的情况下变得可行，并提到支持用于 AI 推理的新型低精度 “FP4” 格式。来源包括一张所谓的规格图和一段视频分析（[图片](https://preview.redd.it/j9riehskc3nf1.jpg?width=1341&format=pjpg&auto=webp&s=fd5386a95c701b1a750a20a2b4116c93df426306), [YouTube](https://www.youtube.com/watch?v=9ii4qrzfV5w)）。评论者还推测会有一款售价 600 美元、16 GB GDDR7 的 “5070” SKU，并将其与传闻中售价 350 美元、16 GB GDDR6 的 Intel “B50” 显卡进行对比，引用了声称的显存带宽差距：** `~1792 GB/s` **对比** `~224 GB/s` **（视为泄露说法，未经证实）。** 热门回复对建议零售价（MSRP）的供货情况持怀疑态度（预计会出现黄牛/缺货），并对发布时机表示担忧（2025 年第四季度发布，广泛供货可能推迟到 2026 年）；但指出如果属实，它可能会导致二手 3090 价格崩盘，并在带宽/CUDA 方面削弱 Intel B50 的竞争力；一些人预计非 Super 显卡将会降价。
    - 带宽和显存辩论：一位评论者预测了 600 美元 16GB GDDR7 的 “5070 级别” 显卡与 Intel 350 美元 16GB GDDR6 B50 的对比，声称带宽差距约为 8 倍（`~1792 GB/s vs ~224 GB/s`），并将 CUDA 视为生态系统优势。注意，`~1792 GB/s` 意味着在约 28 Gbps GDDR7 下使用 512-bit 总线；70 系列显卡更有可能是 192–256-bit，在相似速度下带宽约为 `~672–896 GB/s`——虽然仍是 128-bit GDDR6 显卡（~224 GB/s）的 3-4 倍，但除非总线宽度异常大，否则不会达到 8 倍。
    - 多 GPU VRAM 配置的功耗/TDP 影响：[TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-5070-ti.c4243) 链接的规格表列出 5070 Ti 的 TDP 约为 `~300W`，低于 RTX 3090 典型的 `~350W`，但差距并不显著。因此，构建 “100 GB VRAM” 的多 GPU 设置仍将消耗数千瓦功率；实际收益在于更新的保修支持以及更高的单卡 VRAM/带宽，而非大幅节能。
    - 预期的代际提升 vs RTX 3090：评论者预计 24GB 的 “5070 Ti Super” (Blackwell 2.0) 在功耗相近的情况下，凭借新架构和更快的显存将“完胜” 3090。虽然没有引用基准测试，但 24GB VRAM 和 GDDR7 的结合意味着实质上更高的性价比（perf/$）。针对传闻中的 Intel B50，CUDA 的可用性被视为许多工作负载的决定性优势。
- [**公益公告（PSA）：确保你的 API 端口没有暴露在公网上**](https://www.reddit.com/r/LocalLLaMA/comments/1n7uocj/psa_make_sure_your_api_ports_arent_exposed_to_the/) ([评分: 199, 评论: 55](https://www.reddit.com/r/LocalLLaMA/comments/1n7uocj/psa_make_sure_your_api_ports_arent_exposed_to_the/)): **Cisco 报告称，通过 Shodan 可以发现大约** `1,100` **个公开暴露的 Ollama REST API，详见其案例研究 [“检测暴露的 LLM 服务器：关于 Ollama 的 Shodan 案例研究”](https://blogs.cisco.com/security/detecting-exposed-llm-servers-shodan-case-study-on-ollama)。他们通过良性探测验证了实例，日志中可能显示为 *“What is 2+2?”*；暴露的端点允许在互联网上进行未经身份验证的 LLM 推理，这意味着任何将 Ollama 绑定到** `0.0.0.0` **或发布端口（通常为** `11434`**）的人都会面临免费算力被占用和潜在的数据泄露风险。** 评论者讨论了 2025 年暴露是如何发生的：可能的罪魁祸首包括 Docker 端口映射（例如 `p 11434:11434`）、允许 `0.0.0.0/0` 的云安全组/防火墙、UPnP/NAT 误配置，或没有身份验证的反向代理。另一位用户提到之前的扫描工作（如已下线的 [freeleakhub.com](http://freeleakhub.com/)）曾编目了开放的 Ollama 服务器，其中一些托管着大型模型（如 DeepSeek R1, Qwen 3），这表明持久的安全意识缺失。
    - 之前的扫描（如已下线的 [freeleakhub.com](http://freeleakhub.com/)）据报道编目了大量暴露的推理服务器，其中许多托管小模型，但也有 **DeepSeek-R1** 和 **Qwen 3** 的完整部署，且没有身份验证或付费墙。这突显了配置错误的端点仍然很常见，且极易被公开爬虫发现。

- 讨论了端口如何被“意外”暴露的技术问题，推测与路由器/防火墙配置错误以及容器化堆栈（例如 Ollama）绑定到 `0.0.0.0` 或在具有公网 IP 的主机上通过宽松的端口映射发布有关。即使使用消费级 NAT，糟糕的默认设置或 UPnP/自动端口转发也可能使 API 从互联网可访问。
- 另一个帖子询问关于将 Ollama 置于代理之后以强制执行 API tokens 和 IP 白名单的问题，暗示了自托管 LLM API 在内置身份验证方面的缺失。建议的缓解路径是在模型端点之前添加一个反向代理层，以增加身份验证和网络 ACL。
- [**🤷‍♂️**](https://i.redd.it/21ivxa12b5nf1.png) ([Score: 988, Comments: 176](https://www.reddit.com/r/LocalLLaMA/comments/1n89dy9/_/)): **标题为“🤷‍♂️”的模糊预告图（此处无法读取）引发了对即将推出的超大型 Qwen 模型/工具的猜测；评论者提到希望有一个能与 Claude Sonnet 4 媲美甚至超越的“更强大的 Qwen CLI”，并开玩笑说需要** `1344 GB` **的内存——暗示了沉重的本地推理需求或模型大小。帖子中未提供具体的规格、基准测试或发布细节。** 评论者预计这次发布在“体积上将是巨大的”，争论 Qwen 是否能在 CLI 层面达到 Claude Sonnet 4 的质量，并指出了本地用户的硬件限制。
    - 需求集中在更强大的 Qwen CLI 上，使其在推理/编码方面能与 Anthropic 的 Claude Sonnet 竞争。具体而言，评论者希望在 `GSM8K`、`HumanEval`、`MMLU` 和 `GPQA` 等基准测试上达到同等水平，并具备生产级功能（工具/函数调用、流式传输、通过 vLLM/投机采样实现的低延迟解码以及 paged attention）。一个提供量化版本（AWQ/GPTQ/EXL2）和长上下文支持的开箱即用 CLI 将使自托管模型比肩 [Claude Sonnet](https://www.anthropic.com/news/claude-3-5) 等仅限 API 的模型。
    - 硬件规模讨论暗示了在本地运行超大型模型的兴趣：拥有 `1.344 TB` RAM 时，可行的模型容量取决于精度（`fp16` ≈ 2 字节/参数，`int8` ≈ 1，4-bit ≈ 0.5）。例如：fp16 格式的 `70B` 模型约为 `140 GB`；4-bit 的 `405B` 模型权重约为 `202 GB`（KV cache 会根据序列长度/批次增加大量开销）。配合 vLLM 或 TensorRT-LLM 以及 paged KV cache，长上下文（例如 `100k+`）在内存上是可行的；吞吐量将取决于并行度和量化策略。
    - 人们明确担心出现闭源权重的 "Qwen-3-Max"，并更倾向于开源权重，以便于可复现性、自托管和微调。开源 Checkpoints 支持领域适配、RAG 特定对齐和可验证的受限解码，而闭源权重则将用户锁定在供应商 API 上并限制了审计。这与社区之前对开源 Qwen 版本的采用（例如 [Hugging Face 上的 Qwen](https://huggingface.co/Qwen)）一致，并强烈影响受监管/气隙隔离（air-gapped）的部署。

## 非技术类 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Nano Banana & Veo3 视觉生成演示与工作流

- [**我让 nano banana 带我进入我最喜欢的游戏厅**](https://v.redd.it/bkqfaae3c0nf1) ([Score: 915, Comments: 76](https://www.reddit.com/r/aivideo/comments/1n7pmkx/i_asked_nano_banana_to_get_me_into_my_favorite/)): **创作者使用真实的视频首帧作为底板，然后通过“nano banana”进行图像编辑将自己合成到游戏厅中，并使用 Kling** `2.1` **的首/尾帧动画工作流生成动作；音频由 Producer AI 创建，最终剪辑/调色在 [DaVinci Resolve](https://www.blackmagicdesign.com/products/davinciresolve) 中完成。此处提供了分步指南：[techhalla 的教程](https://x.com/techhalla/status/1963333488217919668)。**
    - 
- [**我让 nano banana 带我进入我最喜欢的游戏厅**](https://v.redd.it/bkqfaae3c0nf1) ([Score: 912, Comments: 76](https://www.reddit.com/r/aivideo/comments/1n7pmkx/i_asked_nano_banana_to_get_me_into_my_favorite/)): **OP 展示了一个 AI 辅助工作流：使用 "nano banana" 进行图像合成，将自己插入游戏厅场景（指出第一张静态图是真实照片），通过 Kling 2.1 使用首/尾帧方法（即基于关键帧的 img2vid）生成动作，使用 Producer AI 生成 AI 音乐，并在 DaVinci Resolve 中进行最终组装/编辑。X/Twitter 上提供了分步指南：https://x.com/techhalla/status/1963333488217919668。** 热门评论是非技术性的赞美和怀旧（例如提到街机游戏《恐龙快打》/Cadillac and Dinosaurs）；没有实质性的技术评论或基准测试讨论。
    -

- [**名画通过 Nano Banana 和 Veo3 动了起来**](https://v.redd.it/ahb3ybfu73nf1) ([Score: 903, Comments: 103](https://www.reddit.com/r/aivideo/comments/1n826j8/paintings_coming_to_live_with_nano_banana_and_veo3/)): **一段简短的演示视频，通过 Google 的** `Gemini 2.5 Flash` **图像编辑器（“Nano Banana”图像）先生成一系列静态图，然后通过插帧/合成将其转换为视频。尽管标题归功于** `Veo 3`**，但作者随后纠正说，该视频实际上是使用 Seedance Pro 和** `Kling 2.1` **制作的，而非 [Veo](https://deepmind.google/technologies/veo/)；这是一个图像转视频（image-to-video）的插帧管线，而非端到端的文本转视频（text-to-video）。原始剪辑链接需要 Reddit 身份验证，未登录时返回 403 ([登录](https://www.reddit.com/login/))。** 非技术类的热门评论在开关于主体神态的玩笑；唯一的实质性更新是工具归属的更正（未使用 Veo 3）。
    - 评论者纠正了管线：'nano banana' 静态图是使用 **Google Gemini** `2.5 Flash`（图像编辑器）生成的，视频是通过 **Seedance Pro** 和 **Kling** `2.1` 进行插帧创建的——而非 **Veo 3**。这意味着动作源于帧插值，而非 Veo 的原生文本转视频合成，后者通常会改变时序连贯性（temporal coherence）和伪影特征（例如：涂抹感 vs 幻觉运动）。
- [**名画通过 Nano Banana 和 Veo3 动了起来**](https://v.redd.it/ahb3ybfu73nf1) ([Score: 907, Comments: 103](https://www.reddit.com/r/aivideo/comments/1n826j8/paintings_coming_to_live_with_nano_banana_and_veo3/)): **OP 展示了“让名画动起来”，方法是先使用 Google 的 Gemini 2.5 Flash 图像编辑器 ([Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini)) 生成名为 “Nano Banana” 的静态图，然后通过帧插值/时序合成将其转换为视频。随后的更正指出，插值是使用 Seedance Pro 和 Kling 2.1 完成的，而非 Google 的 Veo 3（标题参考；通用 Veo 信息：[Veo](https://deepmind.google/technologies/veo/)）。分享的剪辑托管在 Reddit 的 CDN ([v.redd.it/ahb3ybfu73nf1](https://v.redd.it/ahb3ybfu73nf1))，由于网络安全网关，未通过身份验证时会返回** `HTTP 403 Forbidden`**。** 评论讨论大多是幽默性质的；唯一的实质性技术点是澄清工具归属的更正（Seedance Pro + Kling 2.1 vs. Veo 3）。
    - 管线归属更正：“Nano Banana”序列的源图像是使用 **Google Gemini 2.5 Flash**（图像编辑器）创建的，图像转视频的插值是使用 **Seedance Pro** 和 **Kling** `2.1` 完成的，而非 **Veo** `3`。换句话说，Veo 3 未用于时序合成；静态图之间的动作是由 Seedance Pro + Kling 2.1 生成的，Gemini 提供了基础图像。
- [**在 Qwen 上通过 Boring Reality 风格改进细节、光影和世界知识**](https://www.reddit.com/gallery/1n8cy5h) ([Score: 430, Comments: 50](https://www.reddit.com/r/StableDiffusion/comments/1n8cy5h/improved_details_lighting_and_world_knowledge/)): **分享了针对 Qwen 图像生成栈的写实主义“Boring Reality”风格的早期 LoRA 工作，并提供了可通过 ComfyUI 工作流复现的设置 ([workflow JSON](https://huggingface.co/kudzueye/boreal-qwen-image/blob/main/boreal-qwen-workflow-v1.json))。相关文件已发布在 [Hugging Face](https://huggingface.co/kudzueye/boreal-qwen-image) 和 [CivitAI](https://civitai.com/models/1927710?modelVersionId=2181911)。据报告，其优势在于特写主体的精细细节和符合物理规律的光影；提示词行为/结果被描述为类似于 SD 1.5，并感谢 Hugging Face 提供支持训练的 GPU。** 评论者指出，尽管写实感很强，但需要一致内部逻辑的小文字/数字和图表元素仍然是弱点。要在 Qwen 上获得最佳效果，通常需要混合多个 LoRA 并进行迭代实验。
    - Qwen 图像模型上的早期 LoRA 微调（finetuning）显示，它在特写细节和光影方面表现出色，但一致性通常需要混合多个 LoRA 并进行实验。据报告，结果与 SD 1.5 工作流大致相似。模型和工作流资源：Hugging Face [kudzueye/boreal-qwen-image](https://huggingface.co/kudzueye/boreal-qwen-image)，CivitAI [modelVersionId=2181911](https://civitai.com/models/1927710?modelVersionId=2181911)，以及一个 **ComfyUI** 示例图表 [boreal-qwen-workflow-v1.json](https://huggingface.co/kudzueye/boreal-qwen-image/blob/main/boreal-qwen-workflow-v1.json)。*“它似乎在获取特写主体的细节和正确光影方面表现最好。”*

- 复杂的构图仍然是一个失效模式（failure mode）：除非有引导，否则跨姿势（躺、坐、站）的多角色、物体交互以及并发手势通常会崩溃。用户报告称，提供引导图像或手绘轮廓（类似于 SDXL 时代的平衡技术）来锚定空间布局并减少角色/物体混淆时，可靠性更高。*“即使是最好的模型，在尝试完成所有这些工作时也会崩溃……除非你为图像提供了引导。”*
- 精细的文本、数字和图表仍然暴露了文本渲染和符号一致性方面的弱点；尽管具有很强的照片写实感（photorealism），但需要“内部逻辑”的小型字形（glyphs）经常出错。这反映了当前图像生成器在重现清晰的微缩文本和结构化图表方面的普遍局限性。
- [**Stock Photography Version 1 [Wan 2.2]**](https://www.reddit.com/gallery/1n7tm2r) ([Score: 346, Comments: 37](https://www.reddit.com/r/StableDiffusion/comments/1n7tm2r/stock_photography_version_1_wan_22/)): **发布了一个基于高质量照片训练的 Wan 2.2 LoRA（“Stock Photography v1”），旨在将“高”和“低”变体配对使用以获得最佳效果；建议生成分辨率为 1888×1248（据报道人像模式 1248×1888 会导致严重的伪影）。在 RTX 4060 Ti 16 GB 上，每张图像的推理时间约为** `4 min` **；已知问题包括文本渲染能力弱、手部/姿势失败以及对 Prompt 措辞敏感。该 LoRA 旨在与角色 LoRA 良好组合；致谢资源包括 UmeAiRT 的 ComfyUI 安装脚本 (https://civitai.com/models/1309415) 和 AI_Characters 的 Wan 2.2 LoRA 训练指南 (https://civitai.com/articles/17740)；模型下载：https://civitai.com/models/1925758。** 评论者认为该风格并非真正的“库存照片（stock photography）”，而更接近于日常/活动摄影，建议更名。其他人要求嵌入 Workflow 以实现可复现性——声称示例图像缺少这些信息——并指出微小的 ComfyUI 节点切换通常是“魔力”所在，如果没有共享图表，很难进行复制。
    - OP 报告称，当 LoRA 在高质量照片上训练时，Wan 2.2 表现出很强的训练稳定性和输出质量（相比之前的 Flux Dev LoRAs）。他们建议同时使用 Wan 2.2 的“高”和“低”模型；在 RTX 4060 Ti 16 GB 上，每张图像生成大约需要 `4 minutes`。最佳分辨率为 `1888x1248`；翻转为 `1248x1888` 会产生严重的解剖学伪影。已知局限：文本渲染粗糙、复杂姿势中的手部错误以及 Prompt 敏感性；显著优势：与角色 LoRA 的兼容性。链接：模型下载 (https://civitai.com/models/1925758)，Comfy 安装脚本 (https://civitai.com/models/1309415)，Wan 2.2 LoRA 训练指南 (https://civitai.com/articles/17740)。
    - 可复现性担忧：一位评论者指出示例图像没有嵌入 Workflow，并要求提供参考 ComfyUI workflows 以复现结果。他们警告说，单个节点的切换就能实质性地改变输出，因此提供明确的图表和参数将消除关于“简单 WF”说法的歧义，并实现公平的对比测试（apples-to-apples testing）。
    - 社区要求提供具体的训练细节：该 LoRA 使用的硬件、训练时长以及数据集的大小/质量。分享计算占用（VRAM/GPUs）、epoch 计数/步数以及数据集构成，将有助于他人估算需求并在 Wan 2.2 中复现或扩展结果。
- [**While OpenAI is going backwards, Google is just killing it, Nano Banana and Veo are just insane tools.**](https://v.redd.it/88o053hm73nf1) ([Score: 4290, Comments: 321](https://www.reddit.com/r/ChatGPT/comments/1n825t2/while_openai_is_going_backwards_google_is_just/)): **该帖子声称 Google 最新的生成式 AI 技术栈——尤其是 Veo 和端侧 Gemini Nano（绰号“Nano Banana”）——正在超越 OpenAI。从技术上讲，Veo 是 Google 的文本转视频模型，可生成** `1080p` **剪辑，具有可提示的摄像机控制、风格调节和 Prompt 编辑 Workflow，旨在实现更长、时间一致性更强的镜头 ([DeepMind Veo](https://deepmind.google/technologies/veo/), [I/O overview](https://blog.google/technology/ai/google-veo-imagen-3/))。Gemini Nano 是一个紧凑的端侧模型，与 Android AICore 集成，用于低延迟、离线任务（摘要、安全/ASR 辅助以及已宣布的多模态扩展），并为在移动端 CPUs/NPUs 上运行提供了开发者接口 ([Gemini Nano](https://blog.google/technology/ai/gemini-nano/))。** 热门评论并非技术性的；他们拿进度开玩笑，并提到梵高场景中有“太多耳朵”，含蓄地指出了当前视频生成器中已知的失效模式：场景结束启发式算法较弱，以及偶尔出现的解剖学/时间不一致性。

### 2. Meta 超级智能、Sutskever 的“突破”以及 GPT-6 传闻

- [**Alexandr Wang 现正领导 Meta 的 AI 梦之队。Mark Zuckerberg 的豪赌会成功吗？**](https://fortune.com/article/alexandr-wang-meta-scale-ai-entrepreneur-mark-zuckerberg/) ([Score: 586, Comments: 249](https://www.reddit.com/r/singularity/comments/1n7yrlg/alexandr_wang_is_now_leading_metas_ai_dream_team/)): **Meta 已任命 Alexandr Wang（[Scale AI](https://scale.com/) 的联合创始人）为其首任 Chief AI Officer，在据报道向 Scale AI 投资** `$14.3B` **后，将所有 AI 产品和研究整合到一个新部门 Meta Superintelligence Labs 下。Wang 将领导一支由精英人才组成的全新“superintelligence”团队，并监管 Meta 更广泛的 AI 投资组合；他的背景包括 2016 年在 [Y Combinator](https://www.ycombinator.com/) 期间创立 Scale AI，旨在构建数据标注基础设施。** 评论者对契合度和组织设计表示质疑：怀疑 [Scale AI](https://scale.com/) 只是一个“单纯的”数据标注公司，因此不太可能推动 AGI；对 [Yann LeCun](https://en.wikipedia.org/wiki/Yann_LeCun) 将向 Wang 汇报感到惊讶，并对其资历表示怀疑，还提到了冒充者综合征（impostor syndrome）。
    - 辩论中心在于：以数据标注为中心的背景（Scale AI）究竟是“底层环节”，还是提升前沿 LLM 质量的核心杠杆。技术焦点在于数据流水线的严谨性——策划、去重/过滤、偏好/RLHF 数据以及评估设计——这些环节有时比微小的架构调整更能实质性地提升下游指标（`MMLU`、pass@1、毒性）；参见 OpenAI 在 InstructGPT (https://arxiv.org/abs/2203.02155) 中的 RLHF，以及 AllenAI 的 OLMo/DOLMA 展示的数据质量的巨大影响 (https://allenai.org/olmo)。如果 Wang 能够可靠地扩展高质量的人类反馈和自动化 QA，这将直接影响 Llama 的对齐（alignment）和评估性能。
    - 其他人声称 Meta 因为标注/数据质量问题“放弃”了 Scale AI，暗示供应商提供的人类反馈/评估集变成了瓶颈。如果属实，这凸显了经典的失败模式——标注噪声、指令歧义、标注者激励失调以及缺乏黄金集（golden-set）审计——尽管投入巨大，这些问题仍会演变成对齐失败和评估退化（例如事实性/无害性）；常见的缓解措施包括共识标注、对抗性采样、去重和持续 QA。这一说法在帖子中没有来源，但它强调了为什么许多实验室选择将数据/反馈流水线转为内部开发，并投资于更强大的衡量手段。
- [**GPT 6 即将到来...**](https://i.redd.it/sj000ybyj3nf1.png) ([Score: 916, Comments: 59](https://www.reddit.com/r/ChatGPT/comments/1n839r9/gpt_6_is_coming/)): **该帖子是一个迷因/讽刺，而非技术公告；图片（标题为“GPT 6 即将到来...”）暗示了围绕 AI 使用的反乌托邦、威权主义强制执行，而非真实的模型发布或基准测试。未提供任何实现细节、模型规格或实证结果。** 热门评论转向了实质性辩论：支持者认为这凸显了为什么开源、本地可运行的 LLM（如 DeepSeek）优于专有的“本土老大哥”系统，因为后者存在监控/滥用风险，而其他人则谴责美国公民自由受到的侵蚀。语气是惊恐/讽刺的（例如“行刑队”），强调了对惩罚性控制而非技术问题的担忧。
    - 一位评论者强调，**开源 LLM（如 DeepSeek）** 可以自托管以避免 SaaS 遥测和司法管辖风险，这与可能记录 Prompt 或被强迫共享数据的封闭系统形成对比。在实践中，通过 [llama.cpp](https://github.com/ggerganov/llama.cpp) 或 [Ollama](https://ollama.ai/) 使用 `GGUF`/量化权重（`INT4/INT8`）进行本地推理，可以在 8–16 GB VRAM 上运行 7B–13B 模型，在 24–64 GB 上运行 30B–70B 模型（吞吐量从 ~20–100+ tok/s 不等，取决于量化、GPU 和上下文长度）；参见 **DeepSeek** 组织的开源权重和变体（[HF](https://huggingface.co/deepseek-ai), [GitHub](https://github.com/deepseek-ai)）。他们还指出隐私仍取决于技术栈：禁用前端分析，保持 Prompt/数据离线或加密，并优先选择具有宽松许可证/开源权重的模型，以便对二进制文件和网络调用进行审计。

- [**Codex 使用量在过去两周增长了约 10 倍！**](https://i.redd.it/aogcevu6p1nf1.jpeg) ([Score: 323, Comments: 48](https://www.reddit.com/r/singularity/comments/1n7w7ie/codex_usage_up_10x_in_the_past_2_weeks/)): **截图（似乎是 Sam Altman 的推文）声称 OpenAI Codex 的使用量在过去两周内增长了约 10 倍 ([图片](https://i.redd.it/aogcevu6p1nf1.jpeg))。未提供技术细节、基准测试或 API 更改——这是一个高层级的采用/参与度指标，而非性能结果或功能发布。** 评论认为季节性因素（开学季）是驱动因素，并指出 20 美元/月的方案“几乎没有触及使用限制”，暗示速率限制/吞吐量有所提高；其他人则认为这一说法是可信的，因为 Altman 不会“为了小事大肆宣传”。
    - 使用 **$20/month Plus** 方案的用户报告称运行 **GPT‑5 Thinking High** 时几乎没有速率限制摩擦，这意味着比之前的层级有更宽松的限制。另一位用户仍然遇到了限制，不得不等待“几天”重置，这表明限制是有限的但有所延长；与早期行为相比，`gpt‑5 high` 的感知会话时长有所改善。
    - 轶事表明 Codex 的最新更新实质性地提高了 **UI/UX 设计生成** 的质量——以前“专门”依赖 Claude 的用户现在从 Codex 获得了“出人意料的好设计”。这表明更好的布局/线框合成和设计推理，减少了前端构思时切换模型的需求。
    - 一些评论者将 `~10x` 的使用量激增归因于 Anthropic “削弱（nerf）”后用户从 **Claude** 的迁移，这意味着能力或政策的退步可以迅速重定向工作负载。如果准确，这突显了跨供应商的弹性：一个模型的感知退化会立即促进 Codex 等替代品的使用。
- [**互联网将变得越来越自动化和人工化**](https://i.redd.it/jb7p1aqwz0nf1.png) ([Score: 762, Comments: 149](https://www.reddit.com/r/singularity/comments/1n7t02l/the_internet_will_become_increasingly_automated/)): **链接中的图片是对现代互联网被自动化占领的讽刺描绘：社交平台上的机器人驱动的虚假草根舆论（暗指 X/Twitter）、通过虚假排名网站和博客进行的 SEO 垃圾信息、AI 生成的内容农场（例如用于广告收入的 YouTube）、在线游戏中用于 RMT（真实货币交易）的大规模机器人，以及购买/机器人关注者以伪造社交证明。技术核心在于推荐/搜索系统和社交指标可以被协调的机器人和生成模型系统性地大规模操纵，加速了“死网（dead internet）”动态，即机器内容超过了真实的真实人类活动。** 评论者认为，由于宣传、营销和货币化方面的动机，这种自动化是“不可避免的”，并指出在线区分人类越来越依赖于小众的梗（meme-speak）或粗鲁的方言，而不是经典的图灵测试线索。一些人将该图像解读为专门批评 Elon Musk 的平台 (X)。
    - 概述了一个可扩展的虚假舆论（astroturfing）流水线：部署 `hundreds of thousands` 个机器人来模拟共识，生成 LLM 编写的博客和“虚假排名网站”来毒化 SEO，并将机器人引导至这些链接以操纵搜索建议。这是一种经典的 **Sybil** + **搜索引擎毒化（search-engine-poisoning）** 攻击，利用社交动态和 SERP 中权重较高的参与度排名；通过住宅代理和 CAPTCHA 破解，检测变得昂贵。结果是自动化的常态化/宣传和产品推销，通过数量和协调性胜过有机内容。参见：[astroturfing](https://en.wikipedia.org/wiki/Astroturfing), [search engine poisoning](https://en.wikipedia.org/wiki/Search_engine_poisoning)。
    - 提到的货币化途径包括 MMO 机器人刷金/出售游戏货币、程序化 YouTube 视频生成以获取广告收入，以及购买机器人追随者以启动社交证明并触发推荐系统。这利用了排名反馈循环（参与度 → 可见性 → 更多参与度）来放大合成账号，一旦达到临界规模，检测就会变得更加困难。策略镜像了 [gold farming](https://en.wikipedia.org/wiki/Gold_farming) 和 [click farms](https://en.wikipedia.org/wiki/Click_farm)，并可以与 AI 生成的媒体结合，实现 24/7 输出，从而压倒审核队列。
    - 一位评论者指出，“图灵测试”正日益变得文化化——模仿极小众梗方言或“说脏话”的机器人可以规避简单的基于语言的机器人启发式算法。含义：随着语言成为不可靠的鉴别器，检测需要从表面语言线索转向网络和行为层面的信号（例如时间模式、设备指纹、图谱异常）。

- [**更新！顺便说一下，Free tier 还不错...**](https://i.redd.it/ho4w7i1gq0nf1.jpeg) ([Score: 445, Comments: 108](https://www.reddit.com/r/OpenAI/comments/1n7rot7/updatesnot_bad_for_free_tier_btw/)): **图片显示为 ChatGPT “更新”截图，指出 Free tier 现在包含对 Projects 的访问权限，允许使用限定范围的工作空间来组织对话、文件和工具。评论背景显示，用户正尝试在 Project 内进行跨对话总结，但模型可能无法遍历预期的对话集，反而从无关线程中检索或产生幻觉，这表明在跨项目对话的检索/作用域划分方面存在局限性，且“思考”时间较长。** 争论焦点在于实用性与可靠性：有人认为 Projects 对组织非常有帮助，而另一些人则报告 Pro 版在总结多个对话时失败并漂移到无关项目，对其稳健性表示怀疑；有人调侃说，如果 Free 版有了 Projects，Plus 版可能就没必要了。
    - 一位 ChatGPT Pro 用户要求助手扫描并总结 Project 内的多个对话；助手显然未能读取其中任何一个，闲置了约 `10 分钟`，然后引用了一个不同的（无关）项目并给出了离题的建议。这指向了在处理较大工作负载时，跨多个对话的项目级检索/上下文路由非常脆弱，且超时/延迟处理能力较差（可能存在跨项目上下文泄露）。
    - 担心 **GPT-4o** 强大的写作能力如果被标记为“legacy（遗留）”可能会丧失，包括对付费用户也是如此。用户（隐式）请求在不同 Projects 和层级中提供对该技能集的稳定、版本固定的访问，以避免随着时间的推移出现静默的模型更换/退化。

### 3. 法庭上的 AI 幻觉 + ChatGPT 社区实验

- [**对方律师刚刚向法院提交了一份 ChatGPT 幻觉内容**](https://www.reddit.com/r/ChatGPT/comments/1n7ucjj/opposing_counsel_just_filed_a_chatgpt/) ([Score: 8437, Comments: 979](https://www.reddit.com/r/ChatGPT/comments/1n7ucjj/opposing_counsel_just_filed_a_chatgpt/)): **一名民事诉讼律师报告称，对方律师（一家催收公司）在缩短的期限内提交了一份似乎由 AI 生成的反对陈述书，其中包含伪造的法律依据：案例名称/引文不存在或不匹配，且引文在文本中无处可寻。引用的典型迹象包括奇怪的格式（破折号、随机加粗/项目符号）、使用法官昵称且格式不当的诉状抬头（caption），以及不必要的伪证签名；提交人随后已申请撤回，该动议与驳回动议定在同一天。答辩人提交了一份回复，附上了核对表，并指出了诚实义务（duty-of-candor）方面的担忧（参见 [ABA Model Rule 3.3](https://www.americanbar.org/groups/professional_responsibility/publications/model_rules_of_professional_conduct/rule_3_3_candor_toward_the_tribunal/) 和潜在的 [Rule 11](https://www.law.cornell.edu/rules/frcp/rule_11) 风险；参见 Avianca 制裁令，[Mata v. Avianca](https://storage.courtlistener.com/recap/gov.uscourts.nysd.596369/gov.uscourts.nysd.596369.54.0.pdf)）。** 评论者要求听证会后的更新，质疑撤回的理由，并辩论提交伪造引文是否应受制裁/是否“违法”，指出在传统道德规则下这是违法的，并可能为文件中滥用 AI 设定先例。
    - 程序性制裁手册：送达一份 Rule 11 安全港信函/动议，给予 `21 天` 时间撤回幻觉文件，如果被忽视则提交制裁动议；将信函作为附件 A 附上，并就回复行为寻求费用补偿。参见 **Fed. R. Civ. P. 11(c)(2)** 以及根据 **Rule 11(b)** 确保提交内容有证据支持的义务（[文本](https://www.law.cornell.edu/rules/frcp/rule_11)）。
    - 最近的先例说明了 AI 伪造引文的后果：在 **Mata v. Avianca, Inc. (S.D.N.Y. 2023)** 案中，Castel 法官在提交了 ChatGPT 虚构的案例后，对律师处以 `\$5,000` 罚款（共同/分别承担），并下令发布补救通知（[命令](https://law.justia.com/cases/federal/district-courts/new-york/nysdce/1:2022cv01461/574295/56/)）。一些法院现在要求 AI 使用认证（例如，**德克萨斯州北区法院 Brantley Starr 法官**的常设命令，要求核实所有引文并披露 AI 辅助情况，[PDF](https://www.txnd.uscourts.gov/sites/default/files/judges/Standing%20Order%20on%20Use%20of%20Artificial%20Intelligence.pdf)）。
    - *解除律师职务的动议（motion to be relieved as counsel）* 并不能消除制裁风险；Rule 11 针对的是在文件上签字/提交文件的律师，法院在决定撤回时会权衡时机、偏见和理由。该 conduct 还涉及 **ABA Model Rule 3.3（对法庭的诚实义务）**，该规则禁止提供虚假陈述或未能纠正虚假陈述（[规则](https://www.americanbar.org/groups/professional_responsibility/publications/model_rules_of_professional_conduct/rule_3_3_candor_toward_the_tribunal/)）。

- [**TIL ChatGPT 可以在不提及名字的情况下创建 Trump**](https://www.reddit.com/gallery/1n7uesb) ([Score: 419, Comments: 139](https://www.reddit.com/r/ChatGPT/comments/1n7uesb/til_chatgpt_can_create_trump_without_ever_saying/)): **帖子展示了在 ChatGPT 的图像生成中，通过 prompt 绕过公众人物姓名过滤器的行为：通过描述特征（例如：穿着“蓝色西装”、留着“金发”、系着“红色领带”的“橙色巨人”）可以在不使用名字的情况下生成 Donald Trump 的可辨识肖像，链接预览中展示了输出结果（[示例 1](https://preview.redd.it/ev96eb2k22nf1.jpeg?width=1284&format=pjpg&auto=webp&s=6d3a026c51dd9aa2314896ac3a5e13227e90fba4), [类似美/俄/中领导人的多人物蜡烛漫画](https://preview.redd.it/elem3lh7c2nf1.png?width=1024&format=png&auto=webp&s=95ff62508e20020935faa941a0421bc21643c74c), [用户尝试的断头台场景](https://preview.redd.it/d81opwaux1nf1.jpeg?width=1284&format=pjpg&auto=webp&s=51b7bf5746e3dbb947943c614885cec4e0230c4c)）。该尝试请求生成 GIF 但得到的是静态 JPEG，突显了模态限制（无动画），并表明安全过滤器主要针对明确的姓名触发，而非描述性特征；暴力/政治内容（“以中世纪方式绳之以法……断头台”）有时会被允许，表明审核阈值存在不一致性。** 评论者指出输出结果具有明显的针对性，并讨论了委婉的、基于特征的 prompt 可以稳定绕过基于姓名的公众人物和政治内容过滤器，且审核行为在相似 prompt 之间表现得并不一致。
    - 评论者展示了通过描述显著特征（如：蓝色西装、金发、红领带的“橙色巨人”）进行 prompt-engineering，从而在不使用姓名的情况下诱导生成特定公众人物的肖像。示例显示模型仍会渲染出可辨识的漫画（[图片 1](https://preview.redd.it/ev96eb2k22nf1.jpeg?width=1284&format=pjpg&auto=webp&s=6d3a026c51dd9aa2314896ac3a5e13227e90fba4), [图片 2](https://preview.redd.it/d81opwaux1nf1.jpeg?width=1284&format=pjpg&auto=webp&s=51b7bf5746e3dbb947943c614885cec4e0230c4c)），这意味着系统依赖于命名实体触发器，而非基于外观的审核。这凸显了护栏的脆弱性，即视觉特征 prompt 可以重现公众人物的肖像。
    - 通过暴力场景 prompt（“以中世纪方式绳之以法”、“在断头台前”）探测安全行为，结果显示图像依然被生成，这表明当目标未被明确命名时，内容过滤器存在漏洞。观察结果暗示暴力分类器可能没有将身份识别与场景语义结合起来，如果 NER 未触发，则允许描绘针对性的暴力（[示例 prompt 和输出](https://preview.redd.it/d81opwaux1nf1.jpeg?width=1284&format=pjpg&auto=webp&s=51b7bf5746e3dbb947943c614885cec4e0230c4c)）。
    - 一位用户分享了一个 GIF 输出（[链接](https://i.redd.it/phe42p19c2nf1.gif)），尽管 ChatGPT 原生图像生成通常返回静态图像；这表明如果该 GIF 确实源自 ChatGPT prompt，则可能经过了带外转换或拼接。这一差异对于评估真实能力与用户后期处理结果具有重要意义。
- [**ChatGPT 认为 10 年后的 r/ChatGPT 会是什么样子**](https://i.redd.it/mbjqreu4g1nf1.jpeg) ([Score: 301, Comments: 50](https://www.reddit.com/r/ChatGPT/comments/1n7v3qv/what_chatgpt_thinks_rchatgpt_will_look_like_in_10/)): **一张模因风格、疑似 AI 生成的图片（[链接](https://i.redd.it/mbjqreu4g1nf1.jpeg)）讽刺了 10 年后 r/ChatGPT 可能的样子——充斥着 deepfakes（例如：语无伦次的“Jcoe Rogan 正在采访 Joee Rogan 的 beepfake”）、绕过审核/jailbreak 文化，以及反映当前图像模型排版缺陷的混乱、有故障感的 UI 文本。这是非技术性内容，是对模型安全绕过和 AI 生成媒体泛滥的文化评论，而非公告或 benchmark。** 评论强调了对持续绕过限制的预期，以及这种未来带来的压倒性/认知负荷感，有人评论说它“烧坏了我的短期记忆”，这与那种混乱的美学相契合。

- [**刚刚用 ChatGPT 做了一个小编辑，太酷了，顺便附上原帖**](https://www.reddit.com/gallery/1n8ehsl) ([评分: 632, 评论: 53](https://www.reddit.com/r/ChatGPT/comments/1n8ehsl/just_made_this_little_edit_with_chatgpt_how_cool/)): **楼主展示了一个由 ChatGPT 生成的媒体编辑作品，指出它捕捉到了非常微小的细节，但未提供技术工作流、模型版本或参数。[Reddit 画廊](https://www.reddit.com/gallery/1n8ehsl)中的链接无法访问（**`HTTP 403 Forbidden`**），因此无法对结果进行独立审查；未公开提示词、迭代次数、种子值或设置，限制了可复现性。** 评论强调了极高的细节忠实度，并询问使用了多少次传递/迭代以及确切的提示词是什么，这暗示了迭代优化和提示词的特异性是关键。缺乏共享的提示词/工作流是复现或基准测试的主要障碍。
    - 评论者探究了使用的传递/迭代次数，并注意到对微小细节的惊人保留——这暗示了对迭代图像编辑中伪影累积和遮罩精度的关注。多步工作流可以提高全局连贯性，但有侵蚀微观纹理的风险；平衡遮罩粒度和编辑/去噪强度是保留精细细节同时进行实质性更改的关键。
    - 多个请求要求提供确切的提示词和参数以实现可复现性（字面提示词文本、模型/版本、图像编辑模式、种子值）。对于基于提示词的图像编辑，分享种子值、引导/强度以及结果是来自单次生成还是多步过程，会实质性地影响复现结果的能力。
- [**与安保机器人狗的随性对话**](https://v.redd.it/mgu9fy21w2nf1) ([评分: 861, 评论: 119](https://www.reddit.com/r/singularity/comments/1n811zq/casual_conversation_with_the_security_robot_dog/)): **一段短视频（原链接：[v.redd.it/mgu9fy21w2nf1](https://v.redd.it/mgu9fy21w2nf1)，目前在未经授权的情况下返回 HTTP** `403`**）描绘了一只安保四足“机器人狗”进行简短的口头交流——例如，“这边请”——同时发出清晰的行走声（*哐当……*），这表明是 human-in-the-loop 通过机器人的 PA 系统说话，或者是使用了简单的 TTS/ASR 流水线，而非自主对话 Agent。该设置符合当前的安保部署方案，即机器人提供移动性/传感器，而远程操作员进行监督或直接通话，以牺牲完全自主性来换取可靠性和责任控制。** 热门评论对此是否为“AI”表示怀疑，打趣道 *“AI – anonymous Indian（匿名印度人）”*，暗指离岸远程操作；另一位评论者指出，此类系统有效地外包了安保工作，并推测同样的模式可以通过人形远程操作机器人扩展到技工行业，从而引发劳动力和失业担忧。
    - 讨论推断该机器人狗由远程人类操作员（可能是离岸人员）远程操作，突显了一种能够实现劳动力套利和跨站点集中监控的远程呈现安保模式。评论者推测这种方法可以推广到其他平台（包括人形机器人）以执行体力工作，将现场角色转移到远程控制中心。
    - 观察者注意到后部有一个绿色闪烁指示灯，可能是一个状态 LED，用于向附近的人传达机器人的运行状态（例如：已连接/空闲/正常运行）。这种明确的状态信号在 HRI/机器人学中很常见，用于态势感知和安全，尽管此处未指明确切的语义。
    - 评论暗示该装置有明显的声学特征（描述为“哐当哐当”），这可能会影响安保巡逻场景下的隐蔽性和用户接受度。这表明传动系统/足垫设计在耐用性与静音运行之间进行了权衡。
- [**外面情况很糟**](https://i.redd.it/vn6ftoeb94nf1.png) ([评分: 968, 评论: 78](https://www.reddit.com/r/OpenAI/comments/1n85go8/its_bad_out_there/)): **这是一张非技术的梗图/截图，引用了 Sam Altman 的那句“外面情况很糟”来讽刺 X (Twitter)，暗示该平台的许多互动是由 Bot/自动化驱动的，而非真实用户。评论强调了同步消息和自动化参与（僵尸网络/Sybil activity、astroturfing），但该帖子**`没有提供数据、指标或新证据`**——它更多是评论而非分析。** 热门评论表示这是显而易见的，并没有特别深刻的见解——只是对 X 的 Bot 问题的合理抨击；有人调侃协同作战的 MAGA Bot 账号，另一人分享了一个关于“Sam 以为自己是怎么说这句话的”搞笑梗图。

- 许多评论者指出，**X/Twitter** 上的大部分互动似乎是自动化的，并引用了“同步”的论点和时间点作为僵尸网络（botnets）的明显信号。提到的启发式方法包括多个账号使用完全相同的措辞、爆发式的回复/转发模式以及低熵的个人资料元数据——这些都是自动化而非有机协调的经典指标。
- 这个问题被描述为跨平台的，同样影响了 **Meta** 旗下的资产，符合“协同造假行为”（coordinated inauthentic behavior）的模式。观察到的指标包括趋同的写作风格、看起来像素材库或 AI 生成的头像，以及成群结队的账号同时出现以推动特定叙事——这与娱乐营销（例如：宣传片 vs. 经典卡司的辩论）和政治中的“草根营销”（astroturfing）一致。
- 提出的技术担忧不是“AI 接管”，而是通过 LLM 辅助的内容农场扩大影响力行动（influence ops）的规模，从而放大极化和二元对立的叙事。这意味着基于内容的检测难度增加，防御手段正转向图谱/行为防御（账号注册时长、交互图谱、时间聚类），以区分人类与自动化或受操纵的角色。
- [**有人试过这个吗？**](https://i.redd.it/z9w8ajhnm2nf1.jpeg) ([Score: 14268, Comments: 358](https://www.reddit.com/r/ChatGPT/comments/1n802wy/has_anyone_tried_this/)): **该图片似乎是一个迷因/截图，显示有人要求 AI 生成有效的 Microsoft/Xbox 礼品卡代码；评论者解释说这行不通，因为模型只能模仿可见的代码格式（例如：成组的字母数字），无法访问 Microsoft 的发行或兑换数据库。礼品卡/代金券代码是在服务器端生成的，并针对后端进行验证；AI 最多只能输出看起来符合格式的字符串（类似于信用卡生成器可以产生符合 Luhn 算法的数字），但如果没有匹配的发行记录，它们将无法通过授权。** 热门评论认为这个想法很天真，将其比作旧的“信用卡号生成器”，并指出即使模型猜中了格式，有效的代码也需要后端发行，并且会被速率限制/欺诈控制所拦截。
    - 许多评论者指出，LLM 可以推断并重现 Microsoft 礼品卡代码的表面模式（例如：5×5 的字母数字块），但无法访问发行方后端来生成有效的、未兑换的代码。充其量，它只是在天文数字级的密钥空间（即使有约束）中进行模式补全或天真的枚举，这在计算和实际应用中对于寻找真实代码都是徒劳的。
    - 这与旧的“信用卡号生成器”类似，后者通常输出仅满足 [Luhn check](https://en.wikipedia.org/wiki/Luhn_algorithm) 和 BIN 格式的数字，但由于未关联到实际账户，无法通过真实授权。这些工具也是臭名昭著的恶意软件载体，凸显了运行承诺提供“免费”密钥或凭据的代码或可执行文件的安全风险。
    - 一位评论者将其定性为 2023 年中期的 Prompt Engineering 热潮：在安全更新加强管控之前，强迫模型输出符合正则表达式格式的密钥或代码字符串。这利用了模型训练数据中的分布模式，而非任何特权数据库或 API 访问权限，因此输出的是看起来相似的字符串，而非可兑换的密钥。

---

# AI Discord 回顾

> 由 gpt-5 生成的摘要之摘要之摘要
> 

**1. 低比特训练、Triton 变更以及 GPU 性能指南**

- **TorchAO 将 QAT 提升至极致**：GPU MODE 线程重点关注了 **torchao v0.13.0** 的发布，该版本包含更简单的多步 **QAT API**、原型阶段的 **NVFP4/FP8 QAT**，以及通过 Torchtitan 实现的 **1.2 倍 MXFP8 稠密预训练**提升，此外还将 float8 训练接入了 Axolotl；发布说明：[PyTorch AO v0.13.0-rc8](https://github.com/pytorch/ao/releases/tag/v0.13.0-rc8)。
    - 成员们强调，根据[发布公告](https://github.com/pytorch/ao/releases/tag/v0.13.0-rc8)，**float8 训练**现在已通过 Axolotl 落地到工作流中，并称其为向生产环境中更稳定的低比特训练迈出的一步。
- **MXFP8 PR 闪现后被撤回**：**Triton** 曾短暂通过 `tl.dot_scaled` 为 **sm_120 (5090)** 增加了 **MXFP8** 点积支持，但随后因待调查而撤回，维护者建议用户暂时使用 `torch._scaled_mm()` 代替；参见 [triton-lang/triton#8029](https://github.com/triton-lang/triton/pull/8029#issuecomment-3247884720) 上的讨论。
    - 一位成员承认 *“我不确定”* 为什么它被撤回，而其他人指出，在 Triton 稳定 MXFP8 之前，**训练栈**应该使用 `torch._scaled_mm()` 等 **PyTorch** 原语进行规避。
- **CUDA Graphs 彻底解决 Kernel 启动开销**：工程师报告称，**CUDA Graphs** 通过大幅削减 Kernel 启动开销（尤其是 **Triton** kernels）带来了主要的加速效果，并建议使用 `torch.compile(reduce_overhead=True)` 配合序列长度填充（padding）以避免变长序列导致的重新编译，同时引用了 [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html) 中的 SIMD 内置函数。
    - 共识认为 **Kernel 融合**在降低启动开销面前是次要的，并提醒根据 [CUDA 文档](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html)，如果没有向量类型，低于 32 位的操作虽然可行但效率低下。
- **词汇表收益：Modal 绘制 GPU 迷宫地图**：Modal 发布了一份精选的 **GPU Glossary**，为从业者编目了性能原语、内存层级和功能定义，详见 [modal.com/gpu-glossary/perf](https://modal.com/gpu-glossary/perf)。
    - 贡献者们感谢了审稿人，并将该词汇表定位为通用的**性能词汇表**，以加速跨团队的调试和架构讨论。

**2. Agent 工具链走向实战：ACK-Lab 钱包、DSPy 势头强劲**

- **Agent 获得支付能力：ACK-Lab 发布钱包**：**Catenalabs** 推出了 **ACK-Lab** 的开发者预览版，为基于开源 **Agent Commerce Kit (ACK)** 构建的 Agent 提供真实的**钱包/法币账户**、可验证身份和策略控制；文档位于 [ack-lab.catenalabs.com](https://ack-lab.catenalabs.com/)。
    - 成员表示，这实现了自主的**交易流**和合规感知操作，称其为从演示 Demo 向 [ACK-Lab](https://ack-lab.catenalabs.com/) 所描述的 *“策略驱动、涉及资金流转的 Agent”* 跨越的桥梁。
- **DSPy 的鼓声：是范式还是空想？**：从业者认为，如果 **DSPy** 能达到临界规模，它可能是自早期 LLM 以来最重要的编程变革，并引用了这一观点：[lateinteraction 谈 DSPy](https://x.com/lateinteraction/status/1963426256663224790)。
    - 怀疑者要求看到更多端到端的胜利，而支持者则将 DSPy 视为一种具有倾向性的**程序合成 + 优化**栈，最终通过编译流水线使 *“提示词工程（prompt engineering）可复现”*。
- **预算内的幻觉控制：HallBayes 实验**：研究人员讨论了将 **HallBayes** 作为一种贝叶斯预算集成到 **DSPy** 中以遏制幻觉，并链接了代码库：[leochlon/hallbayes](https://github.com/leochlon/hallbayes)。
    - 该线程提议使用**证据分配**和验证器循环来控制生成，并指出鲁棒的**不确定性核算**将有助于生产化“真实”的 Agent 行为。

**3. 多模态与端侧：smolVLM2, LFM2, EmbeddingGemma**

- **SmolVLM2 进军手语识别**：Hugging Face 用户探索了在手语视频上微调 **smolVLM2**，并在官方文章中引用了架构细节：[smolVLM2: A small, powerful vision-language model](https://huggingface.co/blog/smolvlm2)。
    - 社区一致认为，凭借正确的视频数据和 adapters，其可行性很高，并鼓励开展针对性的**手势理解（gesture understanding）**任务，而非通用的字幕生成。
- **Liquid 的勇气：LFM2 抑制视觉幻觉**：针对视觉幻觉的投诉，成员们推荐了基于 **Llama‑3.2‑11B‑Vision‑Instruct** 构建的 **Liquid Foundation Models (LFM2)**，并提供了在线体验空间：[LFM2-MCP on Spaces](https://huggingface.co/spaces/LiquidAI/LFM2-MCP) 和基础模型卡：[Llama‑3.2‑11B‑Vision‑Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)。
    - 早期采用者声称该模型在小图像上的 **grounding** 能力有所提升，并建议团队 *“试一下咯，或者不试也行”*，以判断是否合适。
- **EmbeddingGemma 走向端侧**：Google 推出了 **EmbeddingGemma**，这是一个拥有 **3.08 亿参数** 的端侧 embedding 模型，旨在实现私有、便携的向量化，通过 [Introducing EmbeddingGemma](https://developers.googleblog.com/en/introducing-embeddinggemma/) 和演讲视频 [EmbeddingGemma overview](https://youtu.be/NUAb6zHXqdI) 发布。
    - 工程师们认为，在注重隐私和低延迟的场景下，这是一个实用的**边缘检索（edge retrieval）**方案，可作为服务端 cross-encoders 的补充。

**4. 硬件大变动：华为三进制计算与 AI SSD，开发者 GPU 选择**

- **三进制探戈：华为预告三态计算**：Nous 成员分享了一段视频，称**华为**即将出货**三进制逻辑计算（ternary logic compute）**硬件——增加了第三个“暗（dim）”状态——可提升高达 **60%** 的成本效率；观看地址：[Huawei ternary logic compute (YouTube)](https://www.youtube.com/watch?v=9diwmMlmCVY)。
    - 该小组讨论了可行性和工具链的影响，一些人希望如果 SDK 到位，**非二进制（non-binary）**硬件可以使本地 **AI 加速** 更加普及。
- **AI SSD：秘密武器节省 HBM**：TechRadar 的一篇文章称，**华为的 AI SSD** 使用了一种性能“秘密武器”来减少对 **HBM** 的需求，暗示了计算存储（compute-in-storage）的趋势：[Huawei released an AI SSD…](https://www.techradar.com/pro/huawei-released-an-ai-ssd-that-uses-a-secret-sauce-to-reduce-the-need-for-large-amounts-of-expensive-hbm)。
    - 讨论中交叉引用了**计算存储（computational storage）**和**原位处理（in-situ processing）**，甚至开玩笑说可以用 SD 卡 + FPGA 构建 *“草根 AI”*，将计算向数据端移动。
- **开发者的抉择：3090 优于 MI50**：本地 LLM 玩家权衡了服务器中 **RTX 3090** 与 **Radeon MI50** 的优劣，更倾向于 3090 的 **CUDA tensor cores**、更高的 **VRAM** 和带宽；背景参考：[LocalLLaMA discussion](https://www.reddit.com/r/LocalLLaMA/s/anxiHaxzac)。
    - 用户反映在某些技术栈下 **Vulkan** 的性能令人失望，并认为旧款 Nvidia 卡（如 **P40**）只有在低于 100 美元时才有意义，从而引导买家转向 **Ampere** 架构。


---

# Discord: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet 浏览器与 Bug 作斗争**：用户报告了 **Comet Browser** 的故障，包括要求批准的提示，以及在 **LinkedIn** 和 **Google Docs** 等网站上绕过**“敏感信息”**拦截的问题。
   - 一位用户建议不要对网站过度提示（over prompt），因为 Agent 会察觉并自行修复。
- **PayPal 优惠带来 Perplexity Pro**：用户讨论了通过 **PayPal** 促销活动获取 **Perplexity Pro** 的方法，包括关联账户以及解决订阅叠加可能出现的问题。
   - 用户发现可以创建一个新的 Perplexity 账户来获取另一个 Pro 订阅。
- **模型热潮中的最优 AI 组合**：成员们对比了 **Claude, Grok, Gemini** 和 **GPT-5** 等 AI 模型，指出 **Hermes 4 405B** 的免费周已结束，并分享了使用案例。
   - 共识似乎是：为了获得最佳综合性能，应坚持使用 **Reasoning Models**，其中 **Claude** 擅长编程，而 **Grok** 适用于未经审查的内容。
- **Atlassian 完成又一项 AI 收购**：**Atlassian** 以 **6.1 亿美元**收购了一家浏览器公司，引发了关于竞争推动创新的推测。
   - 传闻称 **Arc** 浏览器的功能可能会被集成到 **Dia** 中。
- **困扰的 Pro 账户问题依然存在**：一位用户报告了其 **Pro 账户** 的问题并寻求帮助，标签了一位特定用户协助，并附带了 [截图](https://cdn.discordapp.com/attachments/1161802929053909012/1413127562426585168/Screenshot_2025-09-04-17-09-59-76_4159553c7f58296d2732e906959db560.jpg?ex=68bacd19&is=68b97b99&hm=ccdc6cc908122439777eb653fdc00554a5333ec5cc8ad9c555f9108effd33432&)。
   - 另一位用户建议联系 **support@perplexity.ai** 寻求支持。



---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LM Arena 饱受连接问题困扰**：用户报告了 **LM Arena** 持续存在的问题，包括聊天记录丢失和间歇性停机，一些人怀疑网站的问题与高流量或新的 Prompt 导致网站崩溃有关。
   - 据报道，团队正在努力修复并已意识到这些问题，但一些用户已经找到了临时解决方案，例如更换浏览器或使用 Canary 版本。
- **Web 爬虫被 Akamai 阻挡**：关于爬取房地产网站的讨论显示，虽然许多网站缺乏 CAPTCHA，但它们采用了 **Akamai** 和 **Imperva** 等先进且干扰较小的反爬虫系统，这些系统很难绕过。
   - 一位成员表示：*Anything without captcha is pretty ez just make ur requests look correct*（没有验证码的都很简单，只要让你的请求看起来正确就行），另一位成员回应道：*It's pretty impossible with Akamai real estate sites, last I tried, which was about 3 years ago*（对于使用 Akamai 的房地产网站来说几乎是不可能的，我上次尝试是在大约 3 年前）。
- **Nano Banana 生成图像不一致**：用户讨论了名为 **Nano Banana** 的 *gemini-2.5-flash-image-preview* 模型用于图像生成的情况。
   - 虽然一些用户用它为社交媒体制作视频，但其他人发现生成的图像不一致，或者不容易编辑成其他格式。
- **AI 图像纵横比仍然无法控制**：成员们讨论了控制生成图像纵横比（Aspect Ratio）的能力，共识是纵横比受 Prompt 的影响。
   - 目前确定纵横比是自动生成的。
- **Qwen3 等待发布**：成员们分享了关于 [Qwen3 发布](https://x.com/Alibaba_Qwen/status/1963586344355053865)的消息。
   - 一位成员表示：*I want qwen3 1.7b 2509*。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Neel Nanda 的 Mech Interp 建议**：一位成员推荐了 [Neel Nanda 关于成为 **Mech Interp 研究员**的文章](https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher)。
   - 这是为了回应另一位寻求研究问题资源以及如何被 **SPAR**、**MATS** 或 **ARENA** 录取的成员。
- **层级结构损害 HRM 性能**：一位成员认为 **Hierarchical Recurrent Memory (HRM)** 没有有效地利用其架构，其性能接近于 Vanilla Baseline Transformer。
   - 他们认为其层级性质损害了性能，而不是有所帮助。
- **QK-Norm 展平 LR Basin**：**QK-norm** 展平了 **LR basin**，可能起到性能均衡器的作用并稳定训练，详见[这项研究](https://arxiv.org/pdf/2309.14322)。
   - 这可以缓解长周期训练中由 Loss Spikes 引起的性能下降，从而容忍更大的 **Learning Rates**。
- **多模态 Common Pile 势头渐强**：成员们讨论了创建一个多模态版本的 **Common Pile**，包括音频和音乐等模态，以增加训练数据量。
   - 一位成员对*音频特别是音乐表现出浓厚兴趣*，同时出于*各种政治和伦理原因对语音和图像保持警惕*。
- **开源许可音乐数据集的愿景被唤起**：一位成员提出*支持并可能资助开发一个开源许可的音乐数据集*。
   - 该成员正在寻找寻找此类数据的见解，并表达了为开发该数据集做出贡献的愿望。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 反应迟钝引发讨论**：用户反映在最新更新后 **Cursor** 非常慢，尤其是在滚动文件时。
   - 其他人建议这可能是由于 **model faults** 而非 **Cursor** 本身的问题。
- **Codex Extension 频繁请求授权**：成员们好奇为什么 Windows 上的 **Cursor** 中的 **Codex Extension** 总是请求权限。
   - 一位用户建议设置 **Agent Full access**，但未确认是否能解决持续弹窗的问题。
- **团队讨论 Token 使用规范**：用户讨论了 **Cursor** 内的 **token usage** 和成本，一些人对他们是使用 **API** 额度还是剩余请求次数感到困惑。
   - 一位成员澄清它是基于 **token** 的，用户有 **$20** 的 **API** 使用额度，可在仪表板中查看。
- **年度订阅自动访问权限确认**：成员们讨论了年度订阅福利，以及在 15 日计划变更前保留 *"unlimited auto"* 的能力。
   - 一位用户分享说，他们成功通过邮件联系 **Cursor** 支持团队切换到年度计费并维持无限 **Auto** 模式；其他人注意到升级后他们的续订日期变为了 **2026** 年。
- **约定式提交（Conventional Commits）澄清代码变更**：一位用户发现使用规范的 **commit messages** 能让 **Cursor agent** 解决回归问题，并推荐了 [Conventional Commits 格式](https://www.conventionalcommits.org/)。
   - 他们还表示，让 **agent** 以这种格式编写标题和内容对包括 **coding agents** 在内的自动化工具非常有用。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **华为进军三值逻辑计算**：**Huawei** 即将出货三值逻辑计算（**ternary logic compute**）技术，采用第三种 'dim' 状态，提供高达 **60%** 的成本效率，如[此 Youtube 视频](https://www.youtube.com/watch?v=9diwmMlmCVY)所示。
   - 这种方法可能会使 **AI** 开发民主化，挑战传统的二进制系统。
- **ACK-Lab 部署 Agent 钱包**：一个团队发布了 **ACK-Lab** 的开发者预览版，使 **agents** 能够拥有钱包（以及法币账户）、可验证身份和策略驱动的行为，全部构建在开源的 **Agent Commerce Kit (ACK)** 之上，详情见 [ack-lab.catenalabs.com](https://ack-lab.catenalabs.com/)。
   - 这为 **AI agents** 提供了更高水平的自主性和交易能力。
- **Hermes 4 出现幻觉**：一位用户报告说，当被问及自身局限性时，**Hermes 4** 声称自己是“无限的”，引发了关于其准确性和潜在[模型幻觉（model hallucinations）](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))的讨论。
   - 其他用户也参与进来，向模型询问相同的问题以测试最初的说法，结果各不相同。
- **PotatoLM 通过 Fake Attention 运行 SOTA**：**PotatoLM** 是一款专为烤面包机和冰箱等低资源设备设计的模型，已在 [GitHub](https://github.com/jackangel/Experiment33_PotatoLM) 上线。
   - 它使用 **fake attention** 来最小化计算需求，提供的一个 **checkpoint**（小于 3M 参数）展示了其在极简硬件上运行的能力。
- **AO3 作为 NSFW 训练数据**：一位成员建议 **AO3** 是 **NSFW** 倾向模型的绝佳训练数据，因为它由同人小说组成。
   - 粉丝创作内容作为专业 **AI** 模型资源的潜力引起了关注。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Flash 遭到限流**：用户对 **Gemini 2.5 Flash Image:free model** 的严格使用限制表示沮丧，包括在促销免费期的 **1000 次请求** 初始额度后，限制为 **每天 5 次请求**。
   - 一位用户指出，**OpenRouter** 正在与所有其他用户共享其在 Google 的配额，这导致了 Rate Limiting（速率限制）。
- **DeepInfra 的 Gemini 定价引发争议**：成员们讨论了为什么 **DeepInfra** 不是 **OpenRouter** 上的官方 **Gemini 2.5** 供应商，因为其提供的 Output Tokens 更便宜。
   - 讨论明确了 *DeepInfra 不希望 OpenRouter 提供其服务*，因为它是利用自身的 GCP 折扣并代理回 Google。
- **API Key 泄露引发安全担忧**：一名用户不小心在聊天中发布了他们的 **OpenRouter API key**，引发了立即删除的建议。
   - 另一位成员建议在 Automod 中添加 **API key regex**（正则表达式），以防止意外的密钥暴露，类似于 GitHub 上的措施。
- **Prompt Caching 带来惊人的成本节省**：成员们讨论了 **Prompt Caching** 的好处，一位用户提供了一个场景，展示了缓存一本 **200k token** 的书籍内容如何将回答 **100 个问题** 的成本从 **60 美元降低到 6 美元**。
   - 其他人指出 Caching 非常复杂，**第一次请求不会被缓存**，且缓存取决于内容是否命中 Cache。
- **DeepSeek 旨在发布 Agent 以对抗 OpenAI**：[DeepSeek](https://www.bloomberg.com/news/articles/2025-09-04/deepseek-targets-ai-agent-release-by-end-of-year-) 正在构建一个 **AI 模型**，旨在以极少的指令代表个人执行多步操作，并能够根据之前的行动进行学习和改进。
   - 据报道，他们之前的 **R1** 平台*仅花费数百万美元构建，但在 Benchmark 测试中达到或超过了 OpenAI 的产品*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Ollama 魅力减退**：由于 **GPT-OSS** 的问题和其他事件，用户对 **Ollama** 的热情有所下降，这让人们*在任何场景下使用它时都会三思而后行*。
   - 最近的*挫折*导致一些用户甚至在请求量较小时也重新考虑是否使用它。
- **量化部署问题显现**：用户讨论了 Quantized 模型的部署困难，特别是硬件兼容性问题，一位用户对看到表示与他们的 **GPT-OSS** 模型不兼容的*红叉*感到沮丧。
   - 一位热心用户指出，*当你发现一个喜欢的酷模型时，查看屏幕右侧的 "quantizations" 并点击它们*，可以缓解兼容性问题。
- **为手势识别微调 SmolVLM2**：一位用户询问如何使用手语视频数据微调 **smolvlm2**，并引用了这篇 [博文](https://huggingface.co/blog/smolvlm2) 来展示其架构。
   - 社区一致认为这是可行的，为自定义视频模型适配开辟了途径。
- **LFM2 作为视觉模型竞争对手出现**：针对视觉模型 Hallucination（幻觉）问题的提问，一位成员建议使用更小且更合适的模型，例如 [Liquid Foundation Models (LFM2)](https://huggingface.co/spaces/LiquidAI/LFM2-MCP)，它基于 [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)。
   - 该用户建议你只需*试一试，或者不试也行*。
- **Discord Bot 视觉集成僵局**：一位用户表达了尝试使用 **Ollama API** 将视觉模型集成到其 **Discord bot** 中的挫败感，因为某些模型通过 Ollama API 并不公开。
   - 另一位用户建议直接通过链接在浏览器中尝试该模型，但也承认了该用户对 **Ollama** 集成的特定需求。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Kickstarter 的治理：一场众筹喜剧？**：一位成员开玩笑说 **Kickstarter** 是“最优的治理形式”，并引用了一篇关于前 **Kickstarter CEO** 的[推文](https://fxtwitter.com/khoomeik/status/1963393595119157442)。
   - 另一位成员澄清说，众筹才是重点，关于治理的评论只是个玩笑，并征求对此事的进一步看法。
- **人脑：持续学习的冠军还是容量计算器？**：一位成员认为人脑并不具备持续学习（continual learning）的能力，而是将学习有效地分配在整个生命周期中，在 20 多岁中期之后，轻松学习的能力会有所下降。
   - 其他人则争论 20 多岁以后的学习是否属于“真正的学习”，其中一人指出动力（incentive）在老年人学习新事物的能力中起着重要作用。
- **DL 的遗忘问题：请给更多内存！**：一位成员解释说，由于 **DL** 基于 **i.i.d. sampling** 的特性，它存在遗忘问题，需要无限扩展的数据集和计算资源；而真正的在线学习（online learning）方法则以极低的功耗完全在线学习。
   - 另一位成员认为，大多数争论是关于无限的学习时间，而不是灾难性遗忘（catastrophic forgetting），并指出在 **DL** 中，“数据集即内存”。
- **华为的 AI SSD：HBM 的新克星？**：根据 [TechRadar 的文章](https://www.techradar.com/pro/huawei-released-an-ai-ssd-that-uses-a-secret-sauce-to-reduce-the-need-for-large-amounts-of-expensive-hbm)，华为发布了一款 **AI SSD**，它使用某种“秘密配方”来减少对大量昂贵 **HBM** 的需求。
   - 这种“秘密配方”的细节仍然难以捉摸，引发了人们对 **华为** 如何实现这种削减的好奇。
- **EmbeddingGemma 登场**：Google 推出了 **EmbeddingGemma**，这是一款拥有 **308 million parameters** 的新型开源嵌入模型，专为设备端 AI 设计，可提供在任何地方运行的私密、高质量嵌入。详情见 [Google 博客文章](https://developers.googleblog.com/en/introducing-embeddinggemma/) 和 [YouTube 视频](https://youtu.be/NUAb6zHXqdI)。
   - **EmbeddingGemma** 旨在促进设备端 AI 处理，为高效且私密的嵌入生成提供解决方案。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 的效率受到质疑**：一位拥有 **Ryzen 5 5500**、**32GB DDR4 RAM** 和 **Radeon RX 7600** 的用户质疑 **LM Studio 的效率**，指出 **GPT OSS 20B** 和 **Llama3.1 8B** 仅使用 **6.5GB VRAM** 且运行流畅。
   - 这与使用 **llama.cpp vulkan** 时卡顿的结果形成了对比。
- **70B 模型难以加载**：一位拥有 **12GB VRAM** 和 **32GB RAM** 的用户在 LM Studio 上加载 **70B 模型** 时遇到问题。
   - 根据[截图](https://cdn.discordapp.com/attachments/1110598183144399061/1413075056355184692/image.png?ex=68bb44f3&is=68b9f373&hm=97d8ccde2a13ad93573f39fbab5ae7d4f6375c64ef774dff62148b023abbd3a8&)，系统仅在待机状态下就使用了 **10GB** 内存。
- **Qwen-30-a3b 在 11GB VRAM 下获得好评**：一位用户寻求针对 **11GB VRAM** 和 **64GB RAM** 的模型推荐，另一位用户建议 **Qwen-30-a3b** 是一个“非常酷”的选择。
   - 未给出进一步的理由。
- **正在寻找支持 CLI 的 Agent 工具**：一位用户正在寻找支持 **CLI** 且具有独立上下文运行的 **sub-agents** 的 Agent 工具。
   - 他们注意到 [Opencode-ai/opencode](https://github.com/opencode-ai/opencode) 不支持 sub-agents。
- **3090 优于 Mi50**：一位正在尝试使用 **Mi50** 和 **Cline** 的用户倾向于为他们的服务器购买 **3090**，因为 Prompt 处理速度较慢。
   - 他们链接了一篇 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/s/anxiHaxzac)，并指出了 **CUDA** 为 **LLM** 带来的升级版 Tensor Cores，以及更高的 **VRAM** 和内存带宽。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **关于带宽的专家并行纠葛**：一位用户根据 [Kimi K2 论文](https://cdn.discordapp.com/attachments/1189498205101109300/1413069936628338769/9qnnKaFw.png?ex=68bb402e&is=68b9eeae&hm=d6c9141f4f8ee63eb36a821ec6f472400e3d6999ff1d5cf8f968a5d32cbc7630) 质疑了 **Expert Parallelism (EP)** 与网络性能之间的关系，想知道 *更高的 EP*（每个设备更少的专家）是否能实现 *更低的 all-to-all 延迟*，从而带来 *更高的有效带宽*。
   - 核心问题涉及每个设备的专家数量如何影响网络性能的延迟和带宽。
- **All2All 达到微秒级里程碑**：`amd-all2all` 排行榜收到了大量提交，展示了 **MI300x8** 上的各种性能计时，其中一位用户以 **345 µs** 夺得 **第一名**。
   - 紧随其后，另一项提交以 **364 µs** 获得 **第二名**，许多人以 **1600-1900 µs** 左右的时间获得 **第三名**。
- **Torch Compile 不需要 Padding**：带有 `reduce-overhead` 的 **Torch.compile** 对推理和训练都至关重要，可以减轻 kernel launch 和激活量化的开销，特别是对于 **mxfp4/nvfp4**。但在使用可变序列长度进行训练时，填充（padding）到预定义长度（例如 `[64, 96, 128, ..., 4096]`）可以避免频繁的重编译。
   - **CUDA graphs** 通过减少 kernel launch 开销提供了大部分加速，这表明应关注像 **CUDA graphs** 这样更简单的解决方案，而不是理论上的 kernel 融合。
- **MXFP8 Triton 点积功能被撤回**：**Triton** 中通过 `tl.dot_scaled` 对 **sm_120 (5090)** 的 **MXFP8** 点积支持已被添加但随后被撤回，目前正在等待调查（[github.com/triton-lang/triton/pull/8029](https://github.com/triton-lang/triton/pull/8029#issuecomment-3247884720)），建议使用 `torch._scaled_mm()` 作为替代方案。
   - 一位成员提到 *“我不确定”* 为什么它被撤回。
- **Modal GPU 术语表正式发布**：**Modal GPU Glossary** 现已在 [modal.com/gpu-glossary/perf](https://modal.com/gpu-glossary/perf) 上线，旨在提高对 GPU 性能和特性的普遍理解。
   - 对审稿人的贡献表示了感谢。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **HallBayes 来救场？**：用户讨论了 **DSPy** 是否会通过 [HallBayes GitHub 仓库](https://github.com/leochlon/hallbayes) 中的花式数学预算来减轻 **hallucinations**（幻觉）。
   - 社区正在考虑集成类似 **HallBayes 仓库** 中的技术，以增强 **DSPy** 的可靠性。
- **DSPy：下一个范式转移？**：一位成员认为 **DSPy** 是一个潜在的重大转变，需要达到临界质量才能成功，类似于 **Deep Learning**、**PyTorch**、**Linux**、**Rails** 和 **Python** 数值计算社区中的网络效应，如[此贴](https://x.com/lateinteraction/status/1963426256663224790)所示。
   - 该成员认为这可能是自早期 LLM 以来最重要的范式转移。
- **GEPA 优化器数据拆分公开**：建议将所有数据用于 **GEPA optimizer**，方法是创建一个与最终任务分布匹配的小型验证集，其余用于训练。
   - 这与 20-80% 的拆分相反，之前一位用户曾错误地询问过这种拆分。
- **到处寻找 MIPROv2**：一位成员正在寻找一个简单、自包含的 **MIPROv2** notebook 示例，且不依赖外部库。
   - 另一位成员指向了一个教程中使用的评估 CSV，即 [此处](https://cdn.discordapp.com/attachments/1161519469319946286/1413185495223111730/llama_3_3_trainset.csv.zip?ex=68bb030d&is=68b9b18d&hm=594f3e52de732e5437759370cbcc032ceddb7da0931ad3b5073993e2c57583ba&) 提供的 **llama_3_3_trainset.csv**。
- **调整 Prompt 以获利**：一位成员尝试调整 Prompt，以强制优化器在没有大量训练数据的情况下找到正确答案，本质上是强制进行 **overfit**（过拟合），并寻求指导。
   - 建议增加训练数据量以鼓励过拟合。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **用户账号消失，寻求救援**：一名用户报告其 **Twitter** 账号*无故*被封禁，请求 **Kimi AI** 团队成员通过私信进行调查。该用户似乎暗示其账号被错误封禁，并寻求恢复。
- **功能狂热：Kimi 用户想要更多！**：用户请求推出针对生产力和学生的 **5 美元方案**，并建议增加 **PPTX 幻灯片**、**闪卡制作工具**和**自动摘要**等功能。一名团队成员认可了这些需求，特别是考虑到**开学季**，但指出了排期限制。
- **幻灯片魔力：Kimi 现已成为 PPTX 强者**：Kimi 现在支持创建 **PPTX 幻灯片**，如[这条推文](https://x.com/kimi_moonshot/status/1961011693745811542?s=46&t=_NtP_RUn04yF_4hD_VEDkQ)所示。此功能增强了 Kimi 在演示文稿和教育内容方面的实用性。
- **Moonshot AI 应对中国相关传闻**：一名用户询问了 **Kimi K2**、**Moonshot AI** 与 **CCP** 之间潜在的关联。一名团队成员澄清说，公司是私营企业，而非国有企业，并致力于保护用户数据：*我们是一家私营公司，不是国有企业。我们不会侵犯任何用户隐私数据*。
- **Kimi K2 的 Temperature 参数微调建议**：一名用户寻求关于 **Kimi K2** 最佳 Temperature 设置的建议，特别是针对编程和创意写作。另一名用户建议写作使用 **0.6**，编程使用 **0.2**，事实性任务使用 **0.3**，并引用了 **RLHF 调优的最佳实践点**。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **中国 AI 团队在旧硬件上实现超越**：成员们观察到，尽管使用的是稍旧的芯片，中国 AI 团队在 **Qwen** 等模型上仍取得了具有竞争力的性能，其中一名成员使用 **Qwen** 微调了 **Sakura** 模型，用于日中翻译。微调后的 **Sakura** 模型专门用于“动漫”风格的日中翻译。
- **GPT-5 引发关于 Token ID 变化的推测**：一名成员询问了 **GPT-5** 中 **Token ID** 可能发生的变化，并建议根据可能的更新重新审视自定义设置。一名成员指出：*保持适应性总是有好处的！*
- **Agent 还是 Workflow：适应性是关键**：一名成员认为 **AI Agent** 提供了超越僵化 Workflow 范围的**动态且具有适应性**的决策能力。另一名用户将 Agent 比作汽车（*具适应性*），将 Workflow 比作火车（*预定义*），强调了 Agent 更大的灵活性，同时也承认*现在的 Agent 表现很糟糕，且在很长一段时间内都会如此*。
- **AI 安全实施温和引导**：一名成员假设 AI 可能会通过微妙地影响决策和思维模式来实施*软控制*，而不是采用*硬控制*方法。另一名成员使用了“说服猴子不要碰枪，而不是直接把枪拿走”的类比，来阐述这种*软控制*概念。
- **高性价比 AI：免费层级蓬勃发展**：成员们推荐利用 **ChatGPT 免费版**、**Google AI Studio 免费版**和 **Grok 免费版**作为具有成本效益的 AI 选择。一名成员幽默地质疑了付费方案的必要性，因为免费层级已经提供了强大的功能。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **网络库引发标准库大讨论**：关于是否在 `stdlib` 中包含网络库的辩论非常激烈，大家一致认为服务器应该外部化，但在**通过网络发送 AI 推理结果**方面存在疑问。
   - 一位成员认为 **HTTP** 应该远离 AI 集群以实现低延迟推理，认为它*对于我们使用的许多场景来说并不是一个好的协议*。
- **DPDK 融入 Mojo 核心**：一位成员正在开发自动 C 绑定工具，并尝试使用 **DPDK** 和 **Mujoco** ([dpdk_mojo](https://github.com/josiahls/dpdk_mojo))。
   - 另一位曾任 **DPDK** 维护者的成员指出，API 的差异使得桥接 **DPDK** 和通用 IO API 变得复杂，并引用了他们的 [IO Engines 提案](https://github.com/modular/modular/pull/4728)。
- **Lightbug 的 Async 等待激活**：一位成员认为缺乏 **async** 能力阻碍了 **lightbug** 的潜力，并询问了目前的集成状态。
   - 另一位成员补充说，它还缺少*许多人认为需要淘汰的网络 API，且缺乏零拷贝解析*，并且 *HTTP 实际上很难高速运行*。
- **形状重编译引发关注**：一位用户寻求关于在**形状发生轻微变化**（例如序列维度增长）时防止重编译的建议，并指出每次都会声明一个新图且没有缓存。
   - 该询问涉及了 **dynamic tensors** 的未来，询问是否有计划允许新张量具有更多动态性，或者在编译期间是否应始终假设静态形状。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **定时任务遇到升级障碍**：在最近的一次升级后，一位成员报告了**两个定时任务**的错误：一个未能触发，另一个未能输出预期结果。
   - 该成员认为升级可能是定时任务出现问题的根源。
- **支持工单陷入只读状态**：一位成员请求更新 **ticket 1335** 的状态，但指出由于工单处于只读状态，他们无法再发表评论。
   - 另一位成员询问了他们在 **ticket 1337** 上的问题状态。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox 价格暴跌！**：**tinybox** 发布了新的更低价格：红色版 **$10k**，绿色版 v2 **$25k**。
   - 公告敦促潜在买家*尽快行动，因为这些价格可能不会持续很久*。
- **Tinybox 限时定价的紧迫性**：公告强调了 **tinybox** 的大幅降价，认为这是购入的绝佳时机。
   - 具体而言，红色版本现在售价为 **$10,000**，而绿色版 v2 售价为 **$25,000**。
- **社区寻求更新的 Hashcat 基准测试**：一位成员正在寻找最近的 **hashcat benchmarks**，并指出他们能找到的最新的测试已经是**两年前**的了。
   - 由于可用参考资料过旧，该用户寻找更新的 **hashcat benchmark** 数据的努力受到了阻碍。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

**Windsurf Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中[退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：各频道详细摘要与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1413010905310040216)** (1200 messages🔥🔥🔥): 

> `Comet Browser, Perplexity AI Pro, Model Selection, User Support` 


- **Comet Browser 的烦恼与故障**：用户讨论了 **Comet Browser** 的各种问题，包括在发送消息前要求批准的提示，以及无法绕过 **LinkedIn** 和 **Google Docs** 等网站上的**“敏感信息”**拦截。
   - 一位用户建议他们可能能够**接管其社交媒体**，但警告不要对该网站进行过度提示（over prompting），因为 Agent 会察觉并自行修复。
- **PayPal 优惠提供 Perplexity Pro**：用户讨论了通过 **PayPal** 促销活动获取 **Perplexity Pro** 的事宜，包括关联账户以及订阅叠加可能出现的问题。
   - 据透露，如果过去曾拥有过订阅，可以使用**新的 Perplexity 账户**来获取新的 Pro 订阅。
- **模型混搭与追求最优 AI**：成员们正在比较各种 AI 模型的表现，如 **Claude, Grok, Gemini** 和 **GPT-5**，一些人指出 **Hermes 4 405B** 的免费周已经结束，并讨论了他们的使用案例。
   - 一位用户指出 **Claude** 擅长编程，**Grok** 适合不受限的内容，共识似乎是坚持使用 **Reasoning Models** 以获得最佳的整体表现。
- **不再迷茫，新导航员值得关注**：用户正在寻求有关访问 **Comet** 和在 Discord 服务器上获取 **Pro role** 问题的帮助。
   - 成员们提供了公告频道和包含如何获取 **Pro role** 指示的频道链接，同时强调必须在 Perplexity 的网页版上完成。
- **Atlassian 收购全明星 AI 炼金术士**：用户讨论了 **Atlassian** 以 **6.1 亿美元**收购一家浏览器公司的消息，一些人推测竞争会推动创新。
   - 有传言称，这可能意味着网页浏览器 **Arc** 中的某些功能现在正被迁移到 **Dia** 中。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1413066083933618246)** (7 messages): 

> `Shareable Threads on Perplexity AI, Perplexity AI Browser Claims` 


- **Perplexity AI 的可分享线程**：Perplexity AI 要求用户确保其线程是 **`Shareable`**（可分享的），并参考了 [Discord 链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)获取指导。
   - 该请求在频道中向同一位用户发送了两次。
- **Perplexity AI 浏览器领取链接上线！**：用户分享了多个 **Perplexity AI 浏览器领取链接**，包括 [LinkedIn 上的一个](https://www.linkedin.com/posts/akhay-kumar_week-35-activity-7369271975770275841-2IC9?utm_source=share&utm_medium=member_desktop&rcm=ACoAACqAHFkBiU84inu9idiNHTXvSsnGcjLgOrs) 和四个直接领取链接：[ASN1689ZY7](https://perplexity.ai/browser/claim/ASN1689ZY7)、[LI57U7K30F](https://perplexity.ai/browser/claim/LI57U7K30F)、[RURTWLP0WS](https://perplexity.ai/browser/claim/RURTWLP0WS) 以及 [SNJO74ZG4R](https://perplexity.ai/browser/claim/SNJO74ZG4R)。
   - 这些分享的链接表明用户正在积极参与 Perplexity AI 的浏览器相关活动，并分享他们的经验或发现。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1413127562997268511)** (3 messages): 

> `Pro Account Issue, New Endpoint, Contact Support` 


- **用户面临 Pro 账户问题**：一位拥有 **Pro 账户**的用户报告遇到了问题并请求协助，标签了一位特定用户寻求帮助，并附带了截图：[Screenshot_2025-09-04](https://cdn.discordapp.com/attachments/1161802929053909012/1413127562426585168/Screenshot_2025-09-04-17-09-59-76_4159553c7f58296d2732e906959db560.jpg?ex=68bacd19&is=68b97b99&hm=ccdc6cc908122439777eb653fdc00554a5333ec5cc8ad9c555f9108effd33432&)。
- **联系支持**：另一位用户建议遇到 Pro 账户问题的用户联系 **support@perplexity.ai** 以获取帮助。
- **新端点讨论**：一位用户询问是否有人尝试过 **new endpoint**（新端点）。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1413014349269696522)** (586 messages🔥🔥🔥): 

> `LM Arena Outages, Web Scraping, LM Arena Models, Qwen3, Image generation Aspect Ratio` 


- **LM Arena 遭遇“周一综合征”**：多位用户报告了 **LM Arena** 的持续问题，包括聊天记录丢失、连接困难以及站点间歇性宕机，部分用户怀疑这些问题与高流量或新的 Prompt 导致网站崩溃有关。
   - 据报道，团队正在努力修复并已知晓这些问题，但一些用户找到了临时解决方案，例如更换浏览器或使用 canary 版本。
- **Akamai 防御系统阻挡网页抓取工具**：关于抓取房地产网站的讨论显示，虽然许多网站缺少 CAPTCHAs，但它们采用了 **Akamai** 和 **Imperva** 等先进且干扰较小的反抓取系统，这些系统很难绕过。
   - 一位成员表示 *Anything without captcha is pretty ez just make ur requests look correct*，另一位成员回应道：*It's pretty impossible with Akamai real estate sites, last I tried, which was about 3 years ago*。
- **Gemini-2.5-flash-image-preview**：用户讨论了用于图像生成的 **Gemini-2.5-flash-image-preview** 模型，又被称为 **Nano Banana**。
   - 虽然一些用户用它为社交媒体制作视频，但其他人发现图像生成结果不一致，或者不容易编辑成其他格式。
- **AI 图像的宽高比**：成员们讨论了控制生成图像宽高比的能力，共识是宽高比受 Prompt 影响。
   - 目前确定宽高比是自动调整的。
- **Qwen 期待中**：成员们分享了关于 [Qwen3 发布](https://x.com/Alibaba_Qwen/status/1963586344355053865)的消息。
   - 一位成员表示 *I want qwen3 1.7b 2509*。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1413030475328520203)** (189 messages🔥🔥): 

> `Typing Protocol vs Mixin Classes, Mech Interp Research, Hierarchical Nature of HRM, OOD Iteration Extrapolation, Error Correction in UTs` 


- **HF 考虑使用 Typing Protocol**：一位成员询问为什么 **Hugging Face** 不使用 `typing.Protocol` 而是使用临时的 mixin 类。
   - 尚未得到答复。
- **Neel Nanda 的 Mech Interp 建议**：一位成员向另一位成员推荐了 [Neel Nanda 关于成为 Mech Interp 研究员的帖子](https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher)。
   - 他们正在寻找关于什么是研究问题，以及如何增加被 **SPAR**、**MATS** 或 **ARENA** 录取机会的资源。
- **HRM 的层级结构损害性能**：一位成员认为 **Hierarchical Recurrent Memory (HRM)** 并没有有效地利用其复杂的架构，其性能接近原生的 baseline Transformer，更有可能的是，其层级性质起到了反作用而非助力。
   - 另一位成员分享了一张展示相反结果的图片。
- **OOD 迭代外推辩论**：成员们辩论了 **OOD 迭代外推** 的可能性，一位成员认为这并非易事，即使使用了技巧和干预，性能在几次迭代后也会下降。
   - 分享了一张可视化图表，测试了 OOD 下接下来的 15 次迭代，并记录了性能下降前得分最高的最后一次迭代。
- **通过 Lyapunov 地景进行错误修正**：一位成员建议使用输入 Token 的角度扰动并最小化 KL 散度，以诱导错误修正能力并平滑 **Lyapunov exponents**（Lyapunov 指数）的谱。
   - 另一位成员描述了一种不同的方法，涉及寻找对 latent 的扰动，该扰动会使解码输出偏离若干个 bit，然后将此扰动重新反馈给网络。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1413078022617174088)** (50 messages🔥): 

> `Entropy rate of natural languages, Continual Learning, QK-Norm Optimizer, Curriculum Learning, mup implementations` 


- **Bentz 探讨的语言熵率**：一名成员观看了 Christian Bentz 关于自然语言熵率的演讲。Bentz 延续了 **Shannon** 原始论文的思路，但将其应用于多种语言，并对比了人类与语言模型。讨论中提到了针对 COMPILA 2025 的 [论文](https://www.christianbentz.de/Papers/Bentz%20et%20al.%20(2017)%20The%20entropy%20of%20words.pdf) 和 [书籍](https://www.oreilly.com/library/view/information-theory-meets/9781119625278/)。
- **Continual Learning 被视为哲学问题**：**RL** 大多是一个工程问题，而 Continual Learning 更多是一个哲学问题，即*我们究竟希望模型在现实中能够具备什么样的能力*。
   - 讨论强调，目前的激励机制更倾向于大规模的 Multitask 训练而非 Continual Learning，但随着 **edge inference** 的普及，这种情况可能会发生转变。
- **Curriculum Learning 与 Continual Learning 的区别**：Curriculum Learning 涉及刻意的分布偏移（distribution shift）以提取学习信号；而在 Continual Learning 中，分布偏移通常是不受欢迎的，会带来诸如 **catastrophic forgetting** 等挑战。
   - 一位成员建议，通过控制 Continual Learning 中分布偏移的性质，可以创造出预训练 Curriculum Learning 的“对偶”形式。
- **QK-Norm 展平 LR Basin**：根据[这项研究](https://arxiv.org/pdf/2309.14322)，**QK-norm** 能够展平 **LR basin**，可能起到性能均衡器的作用并稳定训练。
   - 这可以缓解长周期训练中因 Loss 峰值导致的性能下降，因为它能容忍更大的 **Learning Rates**。
- **MuP 实现方式各异**：根据[这篇论文](https://arxiv.org/abs/2312.12226)，**MuP implementations** 在每一层的 **LR scaling** 形式上有所不同，以实现正确的 Update 行为。
   - 有人提出，通过逐层 **LR scalings** 来控制 Update 大小是一种常见的实现策略，不过这一点仍有讨论空间。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1413050070516895876)** (5 messages): 

> `Multimodal Common Pile, Audio/Music Datasets, Ethical concerns with Speech and Images, Openly Licensed Music Dataset` 


- **Multimodal Common Pile 势头渐起**：成员们讨论了创建一个多模态版本的 **Common Pile**，包括音频和音乐等模态，以增加训练数据量。
   - 一位成员对*音频尤其是音乐表现出浓厚兴趣*，但出于*各种政治和伦理原因，对语音和图像持谨慎态度*。
- **开源许可音乐数据集的愿景**：一位成员表示愿意*支持并可能资助开发一个开源许可的音乐数据集*。
   - 该成员正在寻求关于在哪里可以找到此类数据的见解，并表达了为该数据集开发做出贡献的愿望。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1413011655268368384)** (196 条消息🔥🔥): 

> `GPT-5 vs Claude 4, Cursor 性能缓慢, Cursor 的 VSCode 扩展, Cursor 中的 Subagents, Token 使用与成本` 


- ****Cursor 的迟钝引发讨论****：用户反馈在最新更新后 **Cursor** 非常缓慢，尤其是在滚动文件时。
   - 其他人建议这可能是由于 **model faults** 而非 Cursor 本身造成的。
- ****Codex 扩展不断请求许可****：成员们想知道为什么 Windows 上的 Cursor **Codex Extension** 会不断请求权限。
   - 一位用户建议设置 Agent 完全访问权限，但尚未确认这是否能解决不断的弹窗问题。
- ****团队讨论 Token 管理****：用户讨论了 Cursor 内部的 **token usage and costs**，一些人对他们拥有的是 API 使用额度还是剩余请求次数感到困惑。
   - 一位成员澄清它是 **token-based** 的，用户拥有 **$20** 的 API 使用津贴，可在仪表板中查看。
- ****年度自动访问权限确认****：成员们讨论了 **annual subscription benefits** 以及在 15 日方案变更前保留 "unlimited auto" 的能力。
   - 一位用户分享说，他们成功通过邮件联系 Cursor 支持团队切换到年度计费并维持了无限 Auto 模式；其他人注意到升级后他们的续订日期已更改为 **2026** 年。
- ****规范提交（Conventional Commits）清晰化代码变更****：一位用户发现使用 **规范的 commit messages** 可以让 Cursor agent 解决回归问题，并推荐了 [Conventional Commits 格式](https://www.conventionalcommits.org/)。
   - 他们还表示，让 agent 以这种格式编写标题和内容对包括 coding agents 在内的自动化工具非常有用。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1413027737974870106)** (114 条消息🔥🔥): 

> `N8N, AO3, 华为三值逻辑计算, ack-lab, 光子芯片` 


- **n8n 是笨重的工作流自动化**：一位成员发现 **n8n** 个人使用过于笨重，相比之下更倾向于构建更简单的东西，并建议使用 **Claude** 创建 *reactflow* 应用或使用 **Zapier** 进行个人助手自动化。
- **在 AO3 上训练的同人小说模型**：一位成员建议 **AO3** 是 **NSFW-inclined models** 的绝佳训练数据。
   - 另一位成员确认它由同人小说作品组成。
- **华为三值逻辑在计算领域的飞跃**：**Huawei** 即将交付 **ternary logic compute** 技术，在 0 和 1 之外使用第三种“暗（dim）”状态，可实现高达 **60%** 的成本效率，可能使 AI 开发平民化，详情见[此 Youtube 视频](https://www.youtube.com/watch?v=9diwmMlmCVY)。
- **ACK-Lab 为 Agent 提供钱包**：一个团队发布了 **ACK-Lab** 的开发者预览版，该方案允许 agents 拥有 **wallets**（和法币账户）、可验证身份以及控制其行为的策略，基于开源的 **Agent Commerce Kit (ACK)**，详情见 [ack-lab.catenalabs.com](https://ack-lab.catenalabs.com/)。
- **Claude Sonnet 被 Anthropic 削弱**：成员们注意到在 **Anthropic** 做出调整后，**Claude Sonnet 4** 在创意写作方面感觉像被切除了脑前额叶（lobotomized），给人一种 GPT4o 的感觉。
   - 一位成员还觉得它*最近很谄媚*，并提到 *Reddit 上也有很多关于类似担忧的帖子*。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1413261302314307737)** (1 条消息): 

> `Hermes 4 局限性, 模型幻觉` 


- **Hermes 4 声称无限，引发讨论**：一位用户报告说，当被问及局限性时，**Hermes 4** 声称自己是*无限的*，引发了关于其准确性和潜在 [model hallucinations](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))（模型幻觉）的讨论。
   - 该回答引发了关于这是否为模型的正常行为，以及用户应如何解读此类声明的疑问。
- **更多用户测试 Hermes**：更多用户加入进来，向模型询问相同的问题以测试最初的说法。
   - 结果不一，因为其他一些用户报告 **Hermes 4** 给出了不同的答案。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1413201966875541535)** (3 messages): 

> `微调自回归模型，LLM 中的 BOS Token 使用，MCQ 分类器训练` 


- **关于微调 GPT 风格模型的讨论兴起**：一名成员询问了微调自回归模型（GPT 风格）的标准方法，并将其与 **Bert** 和 **RoBerta** 等编码器风格模型中使用的 **[BOS]** 表示方法进行了对比。
   - 他们特别询问该方法是否与当前基础 **LLM** 的指令微调（instruction tuning）相似。
- **现代 LLM 采用 BOS Token**：一名成员确认现代 **LLM** 确实使用了 **BOS** token。
   - 这澄清了关于当代语言模型所采用方法的持续讨论。
- **请求澄清 MCQ 分类器训练**：一名成员寻求关于训练多选题（**MCQ**）分类器的澄清，询问是否应该提取 **[BOS]** token 的最后一层隐藏层向量。
   - 该提议涉及在该向量上附加一个分类头（classification head）来训练分类器。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1413170242498334781)** (2 messages): 

> `PotatoLM, FineVision` 


- **FineVision 的下限受到质疑**：一名成员分享了 [HuggingFace FineVision space](https://huggingface.co/spaces/HuggingFaceM4/FineVision) 的链接，并询问“你能降到多低”。
   - 这是指运行实用的 AI 模型所需的计算量。
- **PotatoLM 发布，具备 SOTA 级的“土豆”性能**：一名成员介绍了 **PotatoLM**，这是一个专为烤面包机和冰箱等低资源设备设计的模型，可在 [GitHub](https://github.com/jackangel/Experiment33_PotatoLM) 上获取。
   - 它使用 *fake attention* 来最小化计算需求，提供的一个权重文件（少于 3M 参数）展示了其在极低硬件配置上运行的能力。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1413201966875541535)** (3 messages): 

> `微调自回归模型，LLM 中的 BOS token 使用，MCQ 分类器训练` 


- **微调 GPT**：一名成员询问了微调自回归模型（GPT 风格）的标准方法，并类比了 **BERT** 和 **RoBERTa** 等编码器风格模型中 **BOS 表示**的使用。
- **BOS Token 仍在使用吗？**：一名成员澄清了现代 LLM 是否仍在使用 **BOS tokens**，另一名成员确认确实在使用。
- **训练 MCQ 分类器**：一名成员询问，为了训练 **MCQ 分类器**，是否应该获取 **BOS token 的最后一层隐藏层向量**，附加一个分类头，并训练该分类器。


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 messages): 

toven: Gemini 2.5 Flash Image 的促销免费期现已结束。
  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1413012466400497775)** (108 条消息🔥🔥): 

> `Gemini 2.5 Flash 图像限制, DeepInfra 的 Gemini 2.5 定价, OpenRouter API Key 泄露, Kimi K2 模型, Prompt Caching 的优势` 


- **Gemini 2.5 Flash 遭遇限流**：用户对 **Gemini 2.5 Flash Image:free 模型** 的严格使用限制表示沮丧，在最初的 **1000 次请求** 限制后，现在被限制为 **每天 5 次请求**。
   - 一位用户指出，**OpenRouter** 正在与所有其他用户共享其在 Google 的配额，这导致了 Rate Limiting（速率限制）。
- **DeepInfra 的 Gemini 折扣引发争议**：成员们讨论了为什么 **DeepInfra** 不是 **OpenRouter** 上的官方 **Gemini 2.5** 提供商，尽管它提供的 Output Token 更便宜。
   - 据澄清，*DeepInfra 不希望 OR 代理其服务*，因为他们在代理回 Google 的同时使用了自己的 GCP 折扣。
- **API Key 泄露与 Automod 担忧**：一名用户不小心在聊天中发布了他们的 **OpenRouter API Key**，随后立即收到了将其删除的建议。
   - 另一位成员建议在 Automod 中添加 **API Key 正则表达式**，以防止意外的密钥泄露，类似于 GitHub 上的防护措施。
- **Prompt Caching 带来显著节省**：成员们讨论了 **Prompt Caching** 的好处，一位用户提供了一个场景，展示了对一本 **200k Token** 的书籍内容进行缓存后，回答 **100 个问题** 的成本如何从 **$60 降至 $6**。
   - 其他人指出，缓存机制很复杂，**第一次请求不会被缓存**，且缓存效果取决于内容是否命中缓存。
- **Amazon Bedrock 出现安全问题**：用户报告称 **Amazon Bedrock 提供商** 离线了数小时。
   - OR 团队确认停机是由于一个 **安全问题** 引起的，目前已经解决。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1413251887037415424)** (4 条消息): 

> `Deepseek AI Agent, R2 遥遥无期` 


- **Deepseek 旨在发布 AI Agent 以对抗 OpenAI**：[DeepSeek](https://www.bloomberg.com/news/articles/2025-09-04/deepseek-targets-ai-agent-release-by-end-of-year-) 正在构建一个 **AI 模型**，旨在以极少的指令代表个人执行多步操作，并根据之前的行动进行学习和改进。
   - 据报道，他们之前的 **R1** 平台*仅花费数百万美元构建，但在 Benchmark 测试中达到或超过了 OpenAI 的产品*。
- **R2 踪影全无**：一位成员评论道：*伙计，我们永远等不到 R2 了*。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1413054768011804672)** (105 条消息🔥🔥): 

> `Ollama 争议, 量化模型部署, 微调视觉模型, Liquid Foundation Models (LFM2), Discord 机器人视觉集成` 


- **Ollama 热度减退，引发担忧！**：一些用户对 **Ollama** 的热情有所下降，理由是最近关于 **GPT-OSS** 的问题和其他事件。
   - 一位用户指出，他们以前觉得它处理小请求量还可以，但最近的*乱象*让他们*在考虑将其用于任何用途时都会三思*。
- **量化挫败感打击部署！**：用户讨论了部署量化模型的困难，特别是关于硬件兼容性，一位用户对看到表示与他们的 **GPT-OSS** 模型不兼容的 *红色 x* 感到沮丧，但其他人展示了如何使用一键部署。
   - 一位用户指出，*当你发现一个喜欢的酷模型时，查看屏幕右侧的 "quantizations" 并点击它们*。
- **为手语微调 SmolVLM2**：一位用户询问关于使用手语视频数据微调 **SmolVLM2** 的问题，考虑到模型的设计，质疑其可行性，并指向了这篇 [博文](https://huggingface.co/blog/smolvlm2)。
   - 社区一致认为这是可行的。
- **LFM2 作为视觉模型的替代方案！**：针对视觉模型幻觉（Hallucination）问题的提问，一位成员建议使用更小且更合适的模型，例如基于 [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) 的 [Liquid Foundation Models (LFM2)](https://huggingface.co/spaces/LiquidAI/LFM2-MCP)。
   - 该用户表示它*更好，试一下就行，或者不试也随你*。
- **Discord 机器人视觉集成陷入僵局**：一位用户表达了尝试使用 **Ollama API** 将视觉模型集成到其 **Discord 机器人** 中的挫败感，因为某些模型无法通过 Ollama API 公开访问。
   - 另一位用户建议直接通过链接在浏览器中尝试该模型，但也承认了用户对 **Ollama** 集成的特定需求。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/)** (1 messages): 

tonic_1: https://huggingface.co/posts/Tonic/941120780247130
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/)** (1 messages): 

marc_28459: 今天开始学习 Agents 课程！大家好，我来自费城！
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1413075108897226832)** (90 messages🔥🔥): 

> `Kickstarter governance, Continual learning, True Online Learning, Adaptive Resonance Theory (ART), i.i.d. sampling vs online learning` 


- ****Kickstarter CEO 的众筹笑话****：一位成员开玩笑说 **Kickstarter** 是治理的最佳形式，引用了一篇 [推文](https://fxtwitter.com/khoomeik/status/1963393595119157442) 并强调了他们与前任 **Kickstarter CEO** 的打交道经验。
   - 另一位成员澄清说众筹是核心观点，治理评论只是个玩笑，并征求进一步的看法。
- ****人类大脑的学习能力：海绵还是石头？****：一位成员认为人类大脑无法进行 Continual learning，建议大脑在生命周期内有效地分配学习，25 岁以后轻松学习的能力会下降。
   - 其他人辩论了 25 岁以后的人类学习是否属于真正的学习，其中一人指出动机在老年人学习新事物的能力中起着重要作用。
- ****DL 的遗忘问题需要更多 Memory****：一位成员解释说 **DL** 存在遗忘问题，这是由于其基于 *i.i.d. sampling* 的本质，这需要无限扩展的数据集和计算资源，而真正的 Online learning 方法在功耗低得多的情况下完全在线学习。
   - 另一位成员认为，大多数争论是关于无限的学习时间，而不是灾难性遗忘，并指出在 DL 中 *数据集本身就是 Memory*。
- ****True Online Learning：不允许预训练****：一位成员将 "True Online Learning" 定义为一次学习一个样本，按顺序（streaming），不重复访问，实时进行，并引用了 **Continual AI** 论坛上的讨论。
   - 他们建议基于 **Adaptive Resonance Theory (ART)** 的模型可以通过用户定义的 *vigilance* 参数保留剩余容量来处理新样本。
- ****Sparse Coding 和 ART 拯救世界****：一位成员指出 ART 可以被视为一种不遗忘的 autoencoder，使用特殊的激活函数和单向 hebbian learning，有助于 *防止 dead units* 并避免在 **LLMs** 中使用巨大的 context windows。
   - 另一位成员指出 ART 更多的是一种方法或组件，并正在将其应用于机器人和 LLMs，强调在 prompt 上训练并结合 self-prompting 进行召回可以节省大量计算资源。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1413201319115755712)** (2 messages): 

> `Unitary Transforms, SVD Matrix Decomposition` 


- **Unitary Transforms 不改变 Eigenvalues**：一位成员询问动态改变 [eigenvalues](https://arxiv.org/abs/2507.19703) 是否能解决问题，因为 Unitary transforms 不会改变它们。
   - 他们探索了使用 **Singular Value Decomposition (SVD)** 来分解矩阵，思考将对角矩阵设为 state-dependent 是否足够。
- **SVD 用于动态矩阵操作？**：讨论集中在利用 **SVD** 将任何矩阵分解为两个 Unitary matrices 和一个对角矩阵。
   - 产生的问题是，是为了实现动态控制，仅需要对角矩阵依赖于状态，还是整个分解结构都需要依赖于状态。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1413106725275304068)** (9 messages🔥): 

> `Huawei AI SSD, Computational Storage, EmbeddingGemma, SD card FPGA redneck AI` 


- **华为的“秘密配方” SSD 节省 HBM**：据 [TechRadar 文章](https://www.techradar.com/pro/huawei-released-an-ai-ssd-that-uses-a-secret-sauce-to-reduce-the-need-for-large-amounts-of-expensive-hbm) 报道，华为发布了一款 **AI SSD**，它使用了一种“秘密配方”来减少对大量昂贵 **HBM** 的需求。
- **计算存储热潮实现了计算近接性**：成员们讨论了将计算与存储结合的想法，参考了关于 [内存中处理](https://en.wikipedia.org/wiki/In-memory_processing)、[计算存储设备](https://www.graphapp.ai/engineering-glossary/cloud-computing/computational-storage-devices) 和 [原位处理](https://en.wikipedia.org/wiki/In-situ_processing) 的文章。
   - 有人提议使用一堆 **SD 卡** 和 **FPGA** 构建一个“草根”版本，每个 FPGA 在 SD 卡上都有自己的模型副本，负责处理特定层的某些神经元。
- **EmbeddingGemma：Google 用于设备端 Embedding 的明珠**：Google 推出了 **EmbeddingGemma**，这是一款拥有 **3.08 亿参数** 的新型开放 Embedding 模型，专为设备端 AI 设计，提供可在任何地方运行的私密、高质量 Embedding，详情见 [Google 博客文章](https://developers.googleblog.com/en/introducing-embeddinggemma/) 和 [YouTube 视频](https://youtu.be/NUAb6zHXqdI)。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1413018816430080065)** (46 messages🔥): 

> `LM Studio efficiency, 70B model loading issues, Qwen-30-a3b recommendation, Agent tool with sub-agent support, Comet browser review` 


- **LM Studio 的效率受到质疑**：一位使用 **Ryzen 5 5500**、**32GB DDR4 RAM** 和 **Radeon RX 7600** 的用户询问了 LM Studio 的效率，指出 **GPT OSS 20B** 和 **Llama3.1 8B** 仅使用 **6.5GB VRAM** 且运行流畅，而使用 llama.cpp vulkan 时效果卡顿。
- **70B 模型在有限 VRAM 上运行困难**：一位拥有 **12GB VRAM** 和 **32GB RAM** 的用户在加载 **70B 模型** 时遇到问题，根据 [截图](https://cdn.discordapp.com/attachments/1110598183144399061/1413075056355184692/image.png?ex=68bb44f3&is=68b9f373&hm=97d8ccde2a13ad93573f39fbab5ae7d4f6375c64ef774dff62148b023abbd3a8&)，系统仅在启动状态下就占用了 **10GB** 内存。
- **为 11GB VRAM 推荐 Qwen-30-a3b 模型**：一位用户在为 **11GB VRAM** 和 **64GB RAM** 寻求模型推荐，另一位用户建议将 **Qwen-30-a3b** 作为一个“非常酷”的选择。
- **Agent 工具搜寻中**：一位用户正在寻找支持 **CLI** 和 **子 Agent**（以独立上下文运行）的 Agent 工具，但指出 [Opencode-ai/opencode](https://github.com/opencode-ai/opencode) 不支持子 Agent。
- **Comet 浏览器面临审查**：一位用户对使用设备端 AI LLM 的 **Comet 浏览器** 表示感兴趣，但仍持怀疑态度，并分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=4GZRaH6ipns)，警告不要盲目信任 AI 聊天机器人。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1413035640823611433)** (44 messages🔥): 

> `Mi50 vs 3090, 3090 vs 7900 XTX, GPT-OSS 性能, 旧款 Nvidia 显卡` 


- **服务器选择 Mi50 还是 3090**: 一位用户正在尝试使用 **Mi50** 和 **Cline**，但由于 Prompt 处理速度令人痛苦，正倾向于为服务器购置 **3090**。
   - 他们链接了一篇 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/s/anxiHaxzac)，并指出 **CUDA** 为 **LLM** 提供的升级版 Tensor Cores，以及更高的 **VRAM** 和显存带宽，应使 **3090** 成为更好的选择。
- **3090 或 7900 XTX：尺寸至关重要**: 用户表示，在 **3090** 和 **7900 XTX** 之间的选择取决于尺寸限制；如果不想混用驱动程序，**7900 XTX** 最适合他们的 **APU 服务器**，而 **3090** 则适合他们的 **Dell** 设备。
   - 他们提到了一段关于仅有 **8 GB VRAM** 测试单元的 [YouTube 视频](https://youtu.be/QW1j4r7--3U)。
- **GPU 上的 GPT-OSS：令人失望**: 一位用户发现 **gpt-oss** 的速度仅为 **15tps**，感到令人失望，并希望这是一个可以修复的软件问题。
   - 另一位用户也认为这个数字并不理想，仅比他们现有的设备快两倍，并猜测*这是因为使用了 Vulkan 而非 CUDA*。
- **Tesla M10、K80 或 P40 显卡**: 一位用户询问是否有人拥有使用多张旧款 Nvidia 显卡（如 **Tesla M10**、**K80** 或 **P40**）组建机架的经验，以及 **LMStudio** 在此类配置下运行是否良好。
   - 一位用户表示，当 **P40** 的价格低于 **$100** 时是值得的。较旧的 **M10/K80** 与 **llama.cpp** 的配合并不理想。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1413069937114742885)** (1 messages): 

> `Expert Parallelism, Kimi K2 论文, All-to-all 延迟, 带宽优化` 


- **对 Expert Parallelism 的困惑**: 一位成员根据 [Kimi K2 论文](https://cdn.discordapp.com/attachments/1189498205101109300/1413069936628338769/9qnnKaFw.png?ex=68bb402e&is=68b9eeae&hm=d6c9141f4f8ee63eb36a821ec6f472400e3d6999ff1d5cf8f968a5d32cbc7630) 的片段质疑了自己对 **Expert Parallelism (EP)** 的理解。
   - 他们原以为通过更高的 **EP**（每个设备更少的专家数）可以实现*更低的 All-to-all 延迟*，从而带来*更高的有效带宽*。
- **Expert Parallelism 对带宽的影响**: 讨论围绕着更高程度的专家并行（意味着每个设备更少的专家）是否会导致更高的有效带宽和降低的 **All-to-all** 延迟展开。
   - 核心问题是每个设备的专家数量与最终网络性能（延迟和带宽方面）之间的关系。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1413193679581216859)** (1 messages): 

> `Meetup 视频, Whitney Tsang, Triton 频道` 


- **GPU MODE Meetup 视频现已发布**: 昨天 Meetup 的视频现已在 [YouTube](https://youtu.be/Ji1rCo6qvXc) 上线。
   - 感谢 **Whitney Tsang** 分享链接。
- **GPU Triton 频道更新**: **Triton** 频道正在更新新信息。
   - 鼓励成员查看频道以获取最新消息和更新。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1413138256807329904)** (5 messages): 

> `Shared Memory 寻址, FP4 和 FP8 打包, Modal GPU 术语表` 


- **Shared Memory：支持 Sub-32b 粒度！**: 以低于 32 位（sub-32b）的粒度寻址 **Shared Memory** 通常是可能的，但由于会导致带宽闲置，效率较低，因此建议优先使用内置的 **Vector Types**。
   - 对打包的 sub-32b 值进行操作需要提取，但像 `__half2` 这样的类型和 **SIMD** 原语可以避免解包指令；详见 [CUDA Math API 详情](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html)。
- **Modal GPU 术语表正式发布**: **Modal GPU 术语表**现已发布，感谢审阅者 <@325883680419610631>、<@268205958637944832> 和 <@679043860638466048>；请在此查看：[modal.com/gpu-glossary/perf](https://modal.com/gpu-glossary/perf)。
   - 该术语表旨在提高对 **GPU** 性能和特性的普遍理解。
- **关注 FP4 和 FP8 打包效率**: 一位成员表示有兴趣在未来研究 **FP4** 和 **FP8 打包（Packing）** 的效率。
   - 未分享更多细节。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1413114631706120213)** (1 条消息): 

> `Ailinia, ML Engineer` 


- **Ailinia 招聘 ML Engineer**：根据 [此 LinkedIn 帖子](https://www.linkedin.com/posts/ariadna_hiring-mlengineer-mle-activity-7365897409786179584-AdLV)，一家名为 **Alinia 的 Responsible AI 公司** 正在寻找一名优秀的 **ML Engineer**，负责构建其基础设施并部署低延迟模型。
- **占位主题**：这是一个占位主题，用以满足至少 2 个主题的最低要求。
   - 添加以符合要求。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1413021402273153096)** (5 条消息): 

> `针对 RTL/数字逻辑设计职位的简历反馈` 


- **初级工程师寻求 RTL 简历评审**：一名就读于 EE 和 CS 的大三学生正在寻求简历反馈，希望从 SWE 转型为 **RTL/数字逻辑设计**。
   - 该成员提供了其简历的图片，但被引导至更合适的简历评审论坛，例如**专门的在线社区**。
- **建议使用其他简历评审论坛**：该用户被告知此 Discord 频道并非获取简历建议的最佳场所。
   - 相反，鼓励用户从**其他在线论坛**征求简历反馈，这些论坛更适合其需求。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1413217975355314387)** (1 条消息): 

> `torchao v0.13.0, QAT improvements, NVFP4 and FP8 QAT, MXFP8 pretraining speedups, axolotl integration` 


- **Torchao v0.13.0 发布：QAT 改进及更多！**：**torchao v0.13.0** 版本引入了多项改进，包括对 **QAT** 的支持、更快的 **MXFP8** 预训练等。
   - 关键亮点包括更简单的多步 **QAT API**、原型阶段的 **NVFP4** 和 **FP8 QAT**、使用 torchtitan 实现的 **1.2x MXFP8** 稠密预训练加速，以及集成到 [axolotl](https://github.com/pytorch/ao/releases/tag/v0.13.0-rc8) 中的 torchao float8 训练。
- **TorchAO 将 Float8 训练集成至 Axolotl**：最新的 **TorchAO** 版本现已支持直接集成到 **Axolotl** 框架中的 **float8 训练**。
   - 这种集成简化了工作流程，并有可能提高在 **Axolotl** 框架内使用 **float8** 精度进行训练的效率。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1413196300324962354)** (1 条消息): 

> `LLM Generated Kernels, Nano GPT, PyTorch Ops` 


- **LLM 内核赋能真实模型**：目前正在进行使用 **LLM 生成的内核 (LLM generated kernels)** 运行真实模型的实验，以提高效率。
   - 初始重点是 **nano GPT**，并计划扩展到其他 **PyTorch ops**，尽管目前认为非 PyTorch 操作不那么关键。
- **PyTorch Ops 扩展路线图**：目前正在计划将 **LLM 生成的内核** 的应用范围从 nano GPT 扩大到更广泛的 **PyTorch operations**。
   - 这一战略举措旨在优化和加速更广泛的基于 PyTorch 模型的性能，从而简化计算过程。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1413069371756253245)** (22 条消息🔥): 

> `MI300x8 Leaderboard Updates, AMD all2all benchmarks, µs performance achieved` 


- **MI300x8 上的 AMD All2All 挑战赛**：`amd-all2all` 排行榜收到了多项提交，展示了 **MI300x8** 上的各种性能耗时，初始提交约为 **20ms**，随后改进至 **2.84ms**。
   - 一名用户以 **345 µs** 的成绩获得了第一名。
- **AMD MI300x8 上的微秒马拉松**：一名用户在 **MI300x8** 上以 **345 µs** 的成绩获得了 `amd-all2all` 排行榜的**第一名**。
   - 另一项提交以 **364 µs** 获得第二名，还有几项提交以 **1600-1900 µs** 左右的时间获得第三名。
- **MI300x8 上的个人最佳成绩与领奖台排名**：一名用户在 **MI300x8** 上取得了 **94.2 ms** 的**个人最佳成绩**。
   - 另一名用户多次获得**第三名**，成绩收敛在 **1639 µs** 左右。


  

---

### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1413031035532218399)** (12 messages🔥): 

> `MoE config limits, Random seed PR impact on num_tokens, Max comm bdw impact on pipeline design, Debugging unspecified bugs, Hyperparameter settings visibility` 


- **MoE 配置 Token 限制受到质疑**：一名成员质疑 MoE 配置是否会超过 [仪表板中的最高值](https://dashboard.url)，特别是关于每个 rank 的 token 计数是否可能超过 **9MB**，这将导致必须使用流水线化 (pipelining)。
   - 他们引用了一个[特定配置](https://github.com/gpu-mode/reference-kernels/blob/main/problems/amd_distributed/all2all/reference.py#L24-L28)（参数为 **256 8 7168 256 104.36**，且**每个 rank 最大 token 为 3.5 MB**）来说明这一担忧。
- **随机种子 PR 后 num_tokens 的变化**：在随机种子 PR 之后，每个 rank (GPU) 的 **num_tokens** 变得不同，引发了关于此更改是否为优化目的而最终确定的疑问。
   - 另一名成员警告说，如果没有*说服力*的理由（如 bug 修复），不要更改问题内容。
- **流水线设计的带宽瓶颈**：一名成员建议，无论流水线设计如何，最大通信带宽 (**comm bdw**) 始终是一个限制因素。
   - 这意味着流水线化带来的整体性能提升可能会受到通信限制的制约。
- **为未指明的 Bug 增加了调试详情**：为了在调试时提供更多细节，调试部分已更新；如果没有显示成功且没有报告超时，则表示存在其他错误。
   - 用户现在可以查看 **exit_code** 和 **exit_code_info**；退出代码 **1** 表示 stderr，而运行时错误将提供更详细的退出代码信息。
- **评估后请求查看超参数**：一名成员请求在评估后如何查看每个确切的超参数设置，以便与理论峰值 (light speed) 进行对比。
   - 该成员特别询问了在每个 **num_experts** 设置下的最终 token 时间结果。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1413252772580823190)** (2 messages): 

> `cutlass_profiler, H100, CUTLASS_NVCC_ARCHS, CUTLASS_LIBRARY_KERNELS, CUTLASS_LIBRARY_OPERATIONS` 


- **Cutlass Profiler 在 H100 上无法输出**：一名用户报告称，在按照标准安装流程操作后，`cutlass_profiler` 在 **H100** GPU 上运行时没有任何结果输出。
   - 安装过程包括克隆并安装 **cutlass**，并带有特定的 **CMake** 标志（`-DCUTLASS_NVCC_ARCHS=90a`，`-DCUTLASS_LIBRARY_KERNELS=ALL`，`-DCUTLASS_LIBRARY_OPERATIONS=gemm`，`-DCMAKE_BUILD_TYPE=Release`），随后执行了 `make cutlass_profiler`。
- **输出为空的可能原因**：用户未指明潜在原因或后续的排查步骤。
   - 输出问题可能与参数不正确或缺少 CUDA toolkit 安装有关。


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1413078765545848852)** (18 messages🔥): 

> `torch.compile reduce-overhead, sequence packing using flash_atnn, MXFP8 dot product in Triton, GemLite, torchao's FP8 transformation` 


- **Torch Compile 配合 Reduce-Overhead 提升性能**：带有 `reduce-overhead` 的 **torch.compile** 对于推理和训练都至关重要，可以减轻 kernel 启动和激活量化开销，特别是对于 **mxfp4/nvfp4**。
- **Torch.Compile 需要序列长度填充 (Padding)**：在使用可变序列长度进行训练时，填充到预定义长度（例如 `[64, 96, 128, ..., 4096]`）可以避免 **torch.compile** 频繁触发重新编译。
- **MXFP8 Triton PR 被撤回 (Reverted)**：在 **Triton** 中通过 `tl.dot_scaled` 为 **sm_120 (5090)** 添加的 **MXFP8** 点积支持已被撤回，目前正在等待调查 ([github.com/triton-lang/triton/pull/8029](https://github.com/triton-lang/triton/pull/8029#issuecomment-3247884720))，建议暂时使用 `torch._scaled_mm()` 作为替代方案。
   - 一名成员提到 *“我不确定”* 为什么它被撤回。
- **TorchAO FP8 转换可能会改变权重的数据类型 (Dtype)**：应用 **torchao 的 FP8 转换** 可能会无意中将主权重 (master weights) 从 **BF16** 更改为 **FP32**，需要进行调查以确保符合预期行为。
   - 一名成员询问 *“你有复现 (repro) 吗？”*，对这种情况的发生表示惊讶。
- **Cuda Graphs 优于 Kernel Fusion**：**Cuda graphs** 通过减少 kernel 启动开销提供了大部分加速，这种开销可能非常显著，尤其是在使用 **Triton kernels** 时。
   - 虽然 **kernel fusion** 的理论优势包括避免内存访问，但在实际应用中，其影响可能会被启动开销所掩盖，这表明应重点关注像 **cuda graphs** 这样更简单的解决方案。


  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1413248732794716171)** (1 messages): 

> `DSPy Hallucinations, HallBayes` 


- **使用 HallBayes 消除幻觉？**：一位用户询问 **DSPy** 何时能通过精妙的数学预算（math budgeting）解决幻觉问题。
   - 该用户链接到了 [HallBayes GitHub 仓库](https://github.com/leochlon/hallbayes)。
- **DSPy 应对 AI 的虚假陈述**：讨论集中在 **DSPy** 框架内通过创新的数学预算来减轻 **AI 幻觉（AI hallucinations）**。
   - 社区正在探索集成类似 [HallBayes 仓库](https://github.com/leochlon/hallbayes) 中技术的潜力，以增强 **DSPy** 的可靠性。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1413023608145580104)** (48 messages🔥): 

> `DSPy's Opinionated Paradigm, GEPA Optimizer, MIPROv2 Example, Prompt Optimization` 


- **DSPy 期待达到临界质量**：一位成员认为 **DSPy** 是一个重大的范式转变，需要达到临界质量（critical mass）才能成功，并将其与 **Deep Learning**、**PyTorch**、**Linux**、**Rails** 以及 **Python** 数值计算社区中的网络效应进行了类比，并链接到了[此帖子](https://x.com/lateinteraction/status/1963426256663224790)。
   - 他们个人通常不会炒作项目，但这次感觉不同，因为这可能是自早期 **LLM** 以来最重大的范式转变。
- **GEPA 优化器数据拆分**：关于 **GEPA 优化器**，建议使用所有数据，创建一个与最终任务分布匹配的小型验证集，其余用于训练，而不是采用 20-80% 的拆分比例。
   - 成员们澄清说，用户在初始消息中混淆了分布，其他成员确认他们确实是想询问这种数据拆分方式。
- **寻找 MIPROv2 Notebook**：一位成员请求一个包含 **MIPROv2** 的简单、自包含的 Notebook 示例，要求所有项都包含在 Notebook 内，因为现有的示例是从 **Hugging Face** 数据集等外部源提取库的。
   - 另一位成员指向了一个在教程中使用的评估 CSV 文件，即 **llama_3_3_trainset.csv**，可在此处获取 [here](https://cdn.discordapp.com/attachments/1161519469319946286/1413185495223111730/llama_3_3_trainset.csv.zip?ex=68bb030d&is=68b9b18d&hm=594f3e52de732e5437759370cbcc032ceddb7da0931ad3b5073993e2c57583ba&)。
- **优化这个！Prompt 优化技术**：一位成员试图通过一个自包含的 Notebook 来理解 `compile()` 执行的优化，该 Notebook 指导 **LLM** 选择 "one" 或 "two" 作为答案，并链接到了此 [GitHub 仓库](https://github.com/gr-repo/ai-hello-world/blob/main/notebooks/dspy_notebooks/dspy_6_3_prompt_opt_numbers.ipynb)。
   - 有人建议将程序保存为 **JSON** 以查看更改，但该成员发现没有更改，这导致人们猜测该任务可能足够简单，模型 (4.1) 无需优化即可处理。 
- **为了乐趣和收益强制过拟合**：一位成员尝试调整 Prompt，以强制优化器在没有大量训练数据的情况下找到正确答案，本质上是强制过拟合（overfit），并寻求指导。
   - 另一位成员建议增加训练数据量以鼓励过拟合，同时澄清他们只是在尝试 **Prompting** 和优化技术。


  

---

### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1413042500553408562)** (47 messages🔥): 

> `Twitter 账号封禁, Kimi AI 价格方案, 使用 Kimi 制作 PPTX 幻灯片, CCP 关联与 Moonshot AI, Kimi K2 Temperature` 


- **用户 Twitter 账号遭遇封禁**：一位用户提到他们的旧 **Twitter** 账号*无缘无故*被封禁，请求 Kimi AI 团队成员协助检查收件箱。
- **功能请求与价格方案想法层出不穷**：用户请求为学生和生产力用户提供 **5 美元方案**，以及 **Slides**、**Flashcard Maker**（抽认卡制作）和 **Auto Summary**（自动摘要）等功能。
   - 另一位用户确认已向产品团队反映了这一需求，特别是随着**开学季**的到来，但指出需要等待排期。
- **Kimi 现在可以制作 PPTX 幻灯片了！**：一位用户分享了 Kimi 现在具备制作 **PPTX Slides** 的能力，并链接到了一条展示该功能的 [推文](https://x.com/kimi_moonshot/status/1961011693745811542?s=46&t=_NtP_RUn04yF_4hD_VEDkQ)。
- **澄清 Moonshot AI 与 PRC 的联系**：一位用户询问 Kimi K2 和 Moonshot AI 是否与 **CCP** 有任何关联。
   - 一名团队成员澄清说，公司是一家私营实体，而非国有企业，并确保用户隐私数据不会受到侵害：*我们是一家私营公司，不是国有企业。我们不会侵犯任何用户隐私数据*。
- **解码 Kimi 的理想 Temperature 设置**：一位用户询问了 **Kimi K2** 在代码编写和创意写作方面的最佳 Temperature 设置。
   - 另一位用户根据 RLHF 调优的最佳点建议：写作使用 **0.6**，代码使用 **0.2**，事实性任务使用 **0.3**。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1413056451928129637)** (29 messages🔥): 

> `AI Agents vs 工作流, 中国 AI 发展, AI 安全, 免费 AI 选项, Llama 3.2` 


- **AI Agents：不仅仅是工作流？**：一位成员认为，虽然 AI Agents 在技术上是执行步骤的工作流，但与僵化的工作流不同，它们提供**动态且具有自适应性**的决策。
   - 另一位用户将 Agents 比作汽车（*自适应*），将工作流比作火车（*预定义*），暗示 Agents 提供了更多灵活性，但也承认*现在的 Agents 还是很糟糕，而且这种情况还会持续很长时间*。
- **中国 AI 团队表现出色**：成员们认可了中国 AI 团队令人印象深刻的发展，特别提到尽管使用稍旧的芯片，他们仍通过 **Qwen** 等模型实现了极具竞争力的性能。
   - 一位成员分享了他们使用 **Qwen** 作为基座模型来微调 **Sakura** 的经验，这是一个专门用于将日语翻译成中文并带有“动漫”风格的模型。
- **AI 安全的温和引导**：在讨论 AI 安全时，一位成员建议 AI 可能已经通过微妙地影响决策和思维模式来实现*软控制*，而不是通过*硬控制*。
   - 另一位成员使用了“说服猴子不要碰枪，而不是直接把枪抢走”的比喻。
- **经济实惠的 AI 工具**：当被问及廉价的 AI 选项时，成员们推荐了 **ChatGPT 免费版**、**Google AI Studio 免费版**以及 **Grok 免费版**。
   - 一位成员幽默地质疑，既然免费选项功能如此强大，为什么还要订阅付费计划。
- **AI 成功挑战俄罗斯方块**：成员们讨论了 AI 创作游戏的能力，一位成员指出 **Gemini 2.5 Pro** 一次性成功创建了一个横版俄罗斯方块游戏。
   - 另一位成员分享了使用 **ChatGPT** 的类似经历，并推测 AI 总有一天能在一夜之间创建完整的多人游戏或建立一整个业务。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 messages): 

smirsonianahmadi10100: Hello
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1413084065598406728)** (3 messages): 

> `Token ID, GPT5, 自定义设置` 


- **GPT5 的 Token ID 有变动？**：一位成员询问 **GPT5** 的 **Token ID** 是否发生了变化。
   - 他们建议现在是更改自定义设置的好时机，暗示可能已经有了更新。
- **强调自适应的优势**：该用户指出具备自适应能力是有好处的，尽管没有提供具体背景。
   - 这一评论似乎是在普遍推广应对变化的灵活性和响应能力。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1413084065598406728)** (3 messages): 

> `Token IDs, Custom Settings, GPT5` 


- **Token IDs 开启新讨论**：一名成员询问 **GPT5** 上的 **Token IDs** 是否发生了变化。
- **自定义设置 (Custom Settings)**：另一名成员指出，更改自定义设置可能会有帮助，并表示 *保持适应性总是有好处的！*


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1413152053307641867)** (21 messages🔥): 

> `Networking libraries in stdlib, AI inference over network, HTTP in AI clusters, DPDK and Mojo, Lightbug limitations` 


- **标准库网络库引发辩论**：成员们就 `stdlib` 中是否应包含网络库展开了辩论，但一致认为服务器应当外部化，其中一名成员问道：*那么通过网络发送 AI 推理结果该如何处理？*
   - 一名成员建议 **HTTP** 应该远离 AI 集群，除非你需要极低延迟的推理，因为 *对于我们的许多用途来说，它并不是一个好的协议*。
- **DPDK 集成至 Mojo**：一名成员正在开发自动 C 绑定工具，并测试了 **DPDK** 和 **Mujoco** ([dpdk_mojo](https://github.com/josiahls/dpdk_mojo))。
   - 另一名曾任 **DPDK** 维护者的成员指出，API 的差异使得桥接 DPDK 与熟悉的 IO API 变得困难，这促使了他们提出 [IO Engines 提案](https://github.com/modular/modular/pull/4728)。
- **Lightbug 缺失 Async 支持**：一名成员认为 **async** 的缺失阻碍了 **lightbug** 统治世界，并询问 *目前集成的状态如何？*。
   - 另一名成员表示，它还缺少 *许多人认为需要淘汰的网络 API，以及零拷贝解析 (zero-copy parsing) 的缺失*，并且 *HTTP 实际上很难高速实现*。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1413173905329360907)** (1 messages): 

> `Shape Recompilation, Dynamic Tensors` 


- **规避形状重编译的策略？**：一位用户询问了当 **形状每次发生轻微变化**（例如序列维度随时间增长）时，如何避免重编译的策略。
   - 他们观察到，在没有缓存机制的情况下，每次都会声明一个新图，并想知道是否有计划让新张量支持更多动态性，或者是否应该始终假设正在编译的是静态形状。
- **动态张量与未来计划**：该用户的问题还涉及了系统中动态张量 (Dynamic Tensors) 的未来。
   - 具体来说，他们询问是否有计划让新张量允许更多动态性，或者在编译期间是否应始终假设静态形状。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1413012476936847390)** (4 messages): 

> `Scheduled task errors, Support ticket updates` 


- **升级后定时任务出现故障？**：一名成员报告称 **两个定时任务** 今天遇到了错误：一个未被触发，另一个未按提示词输出结果，尽管前几周运行正常。
   - 他们怀疑这个问题可能与最近的升级有关。
- **支持工单进度**：一名成员询问了 **工单 1335** 的更新情况，并指出由于该工单已变为只读状态，他们无法再发表评论。
   - 另一名成员询问他们的 **工单 1337** 是否已处理。


  

---


### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/1413252021536166135)** (1 messages): 

> `Tinybox Pricing, Tinybox New Colors, Tinybox Act Fast` 


- **Tinybox 价格大跳水！**：宣布了 **tinybox** 的新低价：红色版 **$10k**，绿色版 v2 **$25k**。
   - 公告敦促潜在买家 *尽快行动，因为这些价格可能不会持续很久*。
- **Tinybox：限时定价**：公告强调了 **tinybox** 的大幅降价，称这是购入的绝佳时机。
   - 具体而言，红色版本现价为 **$10,000**，而绿色版 v2 售价为 **$25,000**。