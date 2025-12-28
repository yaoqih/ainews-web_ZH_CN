---
companies:
- xiaohongshu
- rednote-hilab
- deepseek
- huggingface
date: '2025-06-06T05:44:39.731046Z'
description: '中国的小红书（Rednote）发布了 **dots.llm1**，这是一个拥有 **1420 亿参数** 的开源**混合专家（MoE）**语言模型。该模型具有
  **140 亿激活参数**和 **32K 上下文窗口**，是在 **11.2 万亿个高质量、非合成 token** 上预训练而成的。


  该模型支持 Docker、HuggingFace 和 vLLM 等高效推理框架，并每隔 1 万亿个 token 提供一次中间检查点（checkpoints），从而实现灵活的微调。基准测试声称其在
  MMLU 上的表现略微超过了 **Qwen3 235B**，尽管在基准测试的选择和合成数据验证方面仍存在一些疑虑。此次发布因其真正的开源许可和未使用合成数据而备受瞩目，激发了社区对其在
  llama.cpp 和 mlx 等框架中获得支持的乐观期待。'
id: MjAyNS0w
models:
- dots-llm1
- qwen3-235b
people: []
title: 今天没发生什么事。
topics:
- mixture-of-experts
- open-source
- model-benchmarking
- fine-tuning
- inference
- context-windows
- training-data
- model-architecture
- model-performance
- model-optimization
---

**平静的一天**

> 2025年6月5日至6月6日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（218 个频道，7362 条消息）。预计节省阅读时间（以 200wpm 计算）：647 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式呈现所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

平静的一天。与 Anthropic 合作的 MechInterp 播客值得一听：

https://www.youtube.com/watch?v=9YQW2mH9FyA

---

# AI Twitter 回顾

pipeline 又挂了！

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

### 1. Rednote dots.llm 模型发布及性能基准测试

- [**中国的小红书 (Rednote) 发布了其 dots.llm 开源 AI 模型**](https://github.com/rednote-hilab/dots.llm1) ([Score: 324, Comments: 126](https://www.reddit.com/r/LocalLLaMA/comments/1l4mgry/chinas_xiaohongshurednote_released_its_dotsllm/)): **中国的小红书 (Rednote) 发布了 [dots.llm1](https://github.com/rednote-hilab/dots.llm1)，这是一个大规模开源 MoE 语言模型，拥有** `142B` **总参数和** `14B` **激活参数（top-6-of-128 专家 + 2 个共享专家），以及 32K 上下文窗口，在** `11.2T` **高质量非合成 token 上进行了预训练。该模型以其真正的开源许可、发布中间检查点（每 1T token）以及对高效推理的基础设施支持（Docker, HuggingFace, vLLM, sglang）而闻名。全面的基准测试（参见 [技术报告](https://github.com/rednote-hilab/dots.llm1/blob/main/dots1_tech_report.pdf)）声称其在 MMLU 上略微超过了 Qwen3 235B。** 热门评论赞扬了其开源状态（发布了不含合成数据的真实 base 模型和中间检查点）、细粒度的 MoE 设计（128 个专家，top-6 路由），并认为与之前的 Nemotron-340B 等模型相比，这次发布被低估了。社区对 llama.cpp 和 mlx 等框架的支持表示乐观。
    - 此次发布因提供真实的 base 模型（无合成数据）、采用真正的开源许可证以及提供中间检查点而脱颖而出，允许通过在自定义数据上微调学习率来进行领域自适应。这种方法在最近的主要 LLM 发布中很少见，为下游用户提供了极大的灵活性。
    - 在技术上，该模型是一个 Mixture-of-Experts (MoE)，具有 128 个专家、top-6 路由和 2 个共享专家。该架构在 142B 总参数池中使用了 14B 激活参数，并在 MMLU 基准测试中与 Qwen3 235B 等更大的模型进行了对比，据报道其表现具有竞争力。
    - 对基准测试的选择提出了担忧：该模型是与 Qwen3 235B base 版而非优化的“thinking”变体进行有利对比的，后者在 MMLU-Pro 中的得分高出约 4 个百分点。在对比中缺少“thinking”模式下的 Qwen3 14B，这表明为了最大化明显的性能优势而进行了精心的策划。
- [**这是最大的“无合成数据”开源权重 LLM 吗？(142B)**](https://i.redd.it/sgokl11mvb5f1.png) ([Score: 198, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1l4vrj4/is_this_the_largest_no_synthetic_data_open_weight/)): **图片重点介绍了新的开源语言模型 "dots.llm1"（142B 总参数），该模型声称在预训练期间未使用合成数据——处理了 11.2 万亿个非合成 token，这是一个非常庞大的语料库。在推理期间，模型激活 140 亿个参数子集，使其在规模庞大的情况下更具成本效益。README 摘录强调所有训练数据均为高质量且非合成的，这在如此参数和 token 规模的 LLM 中非常罕见。** 评论者对数据来源提出了技术担忧，质疑开发者如何验证如此庞大语料库中是否存在合成数据，并指出此类声明在实践中的挑战。此外，还有关于合成数据对模型性能影响的讨论，以及对对比基准测试和量化版本的需求。另一条评论提供了声称的基准测试性能链接，以便进一步分析。
    - 一位评论者提出了验证训练语料库中“无合成数据”声明的技术挑战，指出即使团队自己不生成合成数据，也很难保证数据集不包含来自其他地方的合成内容。这引发了对数据来源以及此类断言对于大规模开源权重 LLM 可靠性的担忧。

- 在使用 DeepSeek V3 等教师模型进行 post-training 的背景下提到了 Benchmarks，并附带了[具体的 Benchmark 结果链接](https://i.imgur.com/2gGX64j.png)。这表明虽然该模型声称没有合成的 pretraining 数据，但其 post-training 或 fine-tuning 阶段可能仍使用了其他模型的输出，从而引发了对“无合成数据”声明纯度的问题。此外，人们也对该模型在采用和不采用合成数据方法下的性能对比感兴趣。
- 有人提出了一个问题，即是否存在根据训练 token 数量对 LLM 进行的全面排名，这突显了比较基础设施方面的缺失，并提出了一个关于评估的技术点：此类排名将为大型 open-weight 模型的规模/性能关系提供参考。
- [**中国小红书（Rednote）开源 dots.llm 的性能与成本**](https://i.redd.it/4kbcizani95f1.png) ([Score: 116, Comments: 11](https://www.reddit.com/r/LocalLLaMA/comments/1l4ms71/chinas_rednote_opensource_dotsllm_performance_cost/)): **该图片是来自小红书（Rednote）团队（中国）的散点图，展示了多个 LLM 在 MMLU-Pro benchmark 上的成本性能图景——包括他们的开源模型 'dots.llm1'。'dots.llm1' 在图中被显著标出，与 DeepSeek-V3、Qwen2.5-72B 和 Llama3-70B 等模型相比，显示出极强的性能/成本比，表明其在单位美元支出下具有极高的效率。有关原始来源和数据，请参阅 [tech report](https://github.com/rednote-hilab/dots.llm1/blob/main/dots1_tech_report.pdf) 和 [图片](https://i.redd.it/4kbcizani95f1.png)。** 评论者对 benchmark 的比较提出了质疑，特别是怀疑 Qwen2.5-72B 的表现优于 Qwen3-235B，这突显了对 benchmark 解读的怀疑态度。另一位评论者注意到重复帖子的泛滥，并呼吁采取更谨慎的发布做法，以避免分散技术讨论。
    - 一位评论者质疑 Qwen2.5-72B 性能优于 Qwen3-235B 这一说法的可信度，强调了对 benchmark 结果的怀疑，并含蓄地挑战了所报告性能数据的评估方法合理性和现实世界泛化能力。
    - 另一位评论者批评了将 active parameter 规模直接等同于推理成本的做法，指出虽然像 dots.llm1 这样较大的模型在单个实例上可能更昂贵，但必须考虑所需的 GPU 规格、显存 (VRAM) 限制和用户 batching 等实际因素。这为单纯基于参数数量的成本比较增加了更细致的视角。
    - 关于 benchmark 的讨论强调，benchmark 通常只衡量 LLM 性能的狭窄方面，建议在根据孤立的 benchmark 分数得出广泛结论时要保持谨慎，并重申了理解实际评估内容的重要性。

### 2. 最近的高效边缘端与开源 LLM 发布 (OpenThinker3 & MiniCPM4)

- [**OpenThinker3 发布**](https://www.reddit.com/r/LocalLLaMA/comments/1l4f1yp/openthinker3_released/) ([Score: 204, Comments: 21](https://www.reddit.com/r/LocalLLaMA/comments/1l4f1yp/openthinker3_released/)): **开源语言模型 OpenThinker3-7B 已在 Hugging Face 上发布，提供标准版和 GGUF 量化版模型。发布说明提到 32B 参数版本即将推出。据报道，该数据集平衡了技术/数学内容与非枯燥段落（例如“关于数字 69 的维基百科页面”）。观察者指出，尽管该模型已发布，但与 OpenThinker3 相比，Deepseek-0528-Qwen3-8B 等竞争模型表现出更强的基准测试性能。** 评论中的技术讨论集中在数据集构成（幽默感与枯燥度）以及获取大规模 GPU 资源进行训练的实际挑战，并对大学与工业界的算力实践表示好奇。此外，人们对 OpenThinker3 相对于同类模型的基准测试竞争力持怀疑态度。
    - 一位评论者指出，Deepseek-0528-Qwen3-8B 与 OpenThinker3 之间存在明显的基准测试性能差异，前者占优，这表明后者在某些任务或基准测试的直接对比中可能表现不佳。
    - 针对学术界与私有环境下的资源分配提出了技术咨询：一位用户询问研究人员如何负担得起启动大规模 GPU 集群（“512 个 A100 实例”）的费用，并推测在获得任何投资之前，利用大学加速器资源（数千个 GPU）预训练商业模型的实用性和伦理问题。
    - 一位用户报告称，在 LM Studio 中，无论提示词如何，OpenThinker3 都会以“发疯”的方式响应，生成冗长或重复的输出，并寻求调整推理参数（如 temperature、k-sampling）的建议，以实现更受控且相关的补全。
- [**MiniCPM4：解码速度比 Qwen3-8B 快 7 倍**](https://i.redd.it/j4mqq99tr95f1.png) ([Score: 128, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1l4njon/minicpm4_7x_decoding_speed_than_qwen38b/)): **发布的图片展示了 MiniCPM4-8B 在 Jetson AGX Orin (64G) 和 RTX 4090 (24G) GPU 上对比 Llama-3-8B、GLM-4-9B 和 Qwen-3-8B 的解码和预填充速度 (tokens/sec) 基准测试；在序列长度高达 128K 时，MiniCPM4-8B 的解码速度比 Qwen-3-8B 高出 7 倍。MiniCPM4 的技术进步包括：可训练的稀疏注意力机制 (InfLLM v2)，可将长文本的注意力计算减少到少于 5% 的 token；三进制量化 (BitCPM)；先进的数据清洗/生成技术；以及一个高度优化的 CUDA ([CPM.cu](http://cpm.cu/)) 推理引擎，该引擎利用了量化和投机采样技术 ([来源](https://github.com/OpenBMB/MiniCPM/blob/main/README-en.md))。通过 ArkInfer 实现跨平台部署，数据质量源于开源的 UltraFinweb 和 UltraChat v2 数据集。** 一条评论称赞了其效率和架构优化，但希望能提供 GGUF 格式以获得更广泛的可用性。另一条评论指出，基准测试可能使用了 FP16/BF16 精度，这意味着在更低位量化（如 Q4）下可以预期更快的解码速度。此外，人们对稀疏注意力对长上下文理解的影响持怀疑态度，尤其是在小说基准测试等实际用例中。
    - Chromix_ 强调了该架构的稀疏注意力机制，在 128K 上下文窗口期间，每个 token 仅计算与不到 5% token 的相关性，并表示有兴趣测试其对信息保留的影响（例如在 fiction.liveBench 上），因为人们担心可能会丢失远处上下文跨度之间的联系。
    - 讨论了技术基准测试：据报道 Qwen3-8B-UD-Q8_K_XL 在 RTX 4090 上达到约 120 tokens/sec；推测 MiniCPM4 基准测试使用了 FP16 或 BF16 精度，这表明使用 Q4 量化可以实现进一步的速度提升。这意味着在 GGUF（量化）模型发布之前，MiniCPM4 在更低量化下可能具有更高的解码速度。
    - 多位评论者对稀疏注意力的性能表示好奇，特别是它与 llama.cpp 等高效运行时的集成，强调了人们对这些优化是否会得到广泛支持并同时提供速度和模型质量的关注。

### 3. 设备端 AI 应用展示

- [**我开发了一款能将照片转化为智能打包清单的 App —— 完全在 iPhone 上运行，100% 私密，无需 API，不收集数据！**](https://i.redd.it/9b1s8amsla5f1.jpeg) ([Score: 193, Comments: 45](https://www.reddit.com/r/LocalLLaMA/comments/1l4q7xf/i_built_an_app_that_turns_your_photos_into_smart/)): **图片展示了 Fullpack iOS App 的开发环境，该应用利用 Apple 的 VisionKit 在本地识别用户照片中的物品，并自动生成打包清单，无需依赖云端 API 或外部数据收集。展示的界面演示了应用功能，如组织旅行类型和列出检测到的物品（例如：“国际商务旅行”），强化了设备端、保护隐私的 Computer Vision 工作流，所有这些均由作者独立开发并发布。这种技术方案突显了更小、更高效的模型以及 Apple 生态系统的进步，如何在消费级设备上实现完整的隐私保护和本地 AI 推理（[App Store 链接](https://apps.apple.com/us/app/fullpack/id6745692929)）。** 技术评论辩论了该应用的实际效用——一位用户质疑其解决的实际问题，好奇它是否只是对拍摄的物品进行库存盘点（“hot dog/not hot dog”），而另一位用户则看到了作为个人库存/通知工具的价值，并建议了可能的次要用途，如家庭库存或 eBay 销售。没有出现深度的技术批评或实现细节辩论。
    - 讨论的核心技术特性是使用本地 LLM 或类似的设备端 AI 进行视觉分类——用户在打包时拍摄物品照片，App 完全在设备端识别物体，通过避免云端 API 或外部数据收集来保护隐私。
    - 关于实际效用和潜在功能扩展存在争论：一些用户看到了家庭库存或搬家（视觉化记录箱子内容）的价值，特别强调了其隐私和离线特性。其他用户则寻求澄清该应用是否包含额外功能，如通知、库存管理或与转售工作流（例如在 eBay 上销售物品）的集成。
- [**在本地机器上与角色进行实时对话**](https://v.redd.it/vzlhsb24ia5f1) ([Score: 141, Comments: 33](https://www.reddit.com/r/LocalLLaMA/comments/1l4prlo/realtime_conversation_with_a_character_on_your/)): **该帖子讨论了一个在本地机器运行的实时角色对话应用，具有语音分割功能，暗示使用当前的 TTS 和角色 AI 技术进行离线、低延迟的生成。一条热门评论指出，像 Kokoro TTS 这样流行的 TTS 引擎缺乏对情感韵律的支持，强调了与 Sesame 等在线模型相比的差距。一个被描述为轻量级且功能齐全的项目链接（[MousyHub](https://github.com/PioneerMNDR/MousyHub)）作为本地实现中 SillyTavern 的替代方案。技术讨论集中在当前本地 TTS 解决方案的局限性（缺乏表现力/情感语音）以及对打包易用性（Windows 安装执行文件）和开源替代方案的赞赏。**
    - 一位用户指出，目前的 TTS 系统（如 Kokoro TTS）缺乏情感支持，这限制了它们在实时角色对话中的真实感。他们表示有兴趣流式传输更先进、更具情感表现力的 TTS 技术，并引用 Sesame 作为可以改进的范例。
    - 一条评论请求增加 `llama.cpp` 支持，表明了在本地硬件上高效运行 LLM 的兴趣。这指向了对原生、离线 LLM 推理在实时应用中的技术需求，利用优化的 C++ 实现。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Gemini 2.5 Pro 及其他模型 Benchmark 结果

- [**Gemini 2.5 Pro 在长上下文表现惊人**](https://i.redd.it/iaa33flmm65f1.png) ([Score: 340, Comments: 41](https://www.reddit.com/r/singularity/comments/1l4c50z/gemini_25_pro_is_amazing_in_long_context/)): **该图片展示了一个名为“Fiction.LiveBench for Long Context Deep Comprehension”的基准测试表，对比了多个语言模型（包括 Gemini 2.5 Pro 和其他领先的 LLM）在 0 到 192,000 tokens 上下文长度下的表现。Gemini 2.5 Pro 在所有上下文窗口中均表现出持续的高理解准确率，在较长上下文窗口中优于或接近 GPT-4、Claude 和 O3 等竞争对手。报告按模型和上下文大小列出了性能指标，突显了 Gemini 2.5 Pro 在输入长度增长时保持性能的实力——这对于涉及冗长文档的现实世界用例至关重要。** 一位评论者注意到“o3”模型奇特的性能趋势——在极高上下文之前一直保持近乎完美的准确率，直到突然下降，这引发了对架构选择和资源分配的质疑。有人呼吁专家就这些趋势是由模型设计、检索增强生成 (RAG) 还是仅仅是资源扩展造成的提供见解。
    - Aeonmoru 讨论了在特定上下文窗口截断处观察到的 o3 模型性能下降（注意到在 16k 和 60k tokens 处有小幅下降，随后是更显著的下降）。他们推测这些模式是否归因于专有模型技术、资源分配策略或更广泛的架构因素，并邀请熟悉模型内部机制或检索增强生成 (RAG) 方法的人士分享见解。
    - Laffer890 批评了长上下文基准测试的可靠性，认为虽然模型在叙事任务中表现良好，但在处理大规模技术输入（如 `192k` tokens 的源代码或多个工具描述）时会遇到显著困难。评论强调，尽管上下文窗口变大，但目前的模型“并不擅长抽象概念并将其在极低和浅层水平之外进行连接”，这标志着在深度上下文理解方面仍存在持久的局限性。
- [**Gemini 06-05 在 FACTS grounding 方面大幅领先其他模型**](https://www.reddit.com/r/singularity/comments/1l4fki3/gemini_0605_massively_outperforming_other_models/) ([Score: 216, Comments: 37](https://www.reddit.com/r/singularity/comments/1l4fki3/gemini_0605_massively_outperforming_other_models/)): **一位用户展示了多个 LLM 的对比——Gemini、o3、o4-mini、Claude 4 Opus、Grok 3 和 Deepseek R1 05-28——突出了 Gemini 06-05 在 FACTS grounding 基准测试中的显著优势，该基准测试衡量事实准确性和抗幻觉能力（可能类似于其他抗幻觉测试）。声称包括 Gemini 改进的可信度、高上下文窗口（“100 万 tokens”）以及即使在复杂任务上也有卓越的准确性，据报道超过了 Claude 4 Opus 的表现。** 评论中的技术讨论澄清了“FACTS grounding”涉及抗幻觉基准测试。用户注意到 Gemini 06-05 表现出的卓越事实可靠性，并测试了其在现实世界中的 grounding 能力，特别是在具有挑战性的语境下，与 Claude 4 Opus 相比，它似乎减少了幻觉。
    - FACTS grounding 被讨论为一种抗幻觉基准，专门衡量模型将其答案严格基于所提供上下文的能力，而不是生成听起来合理但可能是虚构的信息。这被框架为评估模型像研究助手一样准确引用源文档的能力。
    - 用户对比测试观察到，Gemini 06-05 在复杂任务中表现出的幻觉明显少于 Claude 4，表明在上下文敏感任务中事实精确度和可靠性有所提高。这被视为关注准确信息提取的用户的一个关键性能指标。
    - 尽管在 FACTS 等 grounding 指标上表现强劲，但实际反馈指出 Gemini 06-05 在视觉文本识别和指令遵循等领域仍有局限性，这表明 grounding 性能并不一定意味着在所有模态或指令上都具有统一的能力。

- [**Gemini 2.5 Pro 06-05 未能通过简单的橙色圆圈测试**](https://i.redd.it/vqtbgr0dy85f1.jpeg) ([Score: 235, Comments: 96](https://www.reddit.com/r/singularity/comments/1l4l3w5/gemini_25_pro_0605_fails_the_simple_orange_circle/)): **随附的图像展示了经典的艾宾浩斯错觉 (Ebbinghaus illusion) —— 这是一个著名的用于测试 AI 视觉和推理的视觉现象 —— 其中两个大小完全相同的橙色圆圈由于周围环境的不同而显得大小不一。帖子指出 "Gemini 2.5 Pro 06-05" 未能正确判断大小对比，误解了这一错觉。评论中的技术讨论将其与其他模型（尤其是 "o3"）的结果进行了对比，后者通过使用 Python 代码测量直径并正确推断出该错觉的诡计，展示了模型推理和测量能力的鲁棒性。** 一些用户注意到这种错觉经常让模型和人类都感到困惑，而另一些用户则提供了 AI 和他们自己识破或陷入错觉的例子，突显了模型输出的可变性，并引发了关于多模态 AI 推理能力和局限性的讨论。
    - 一位评论者描述了 Gemini 2.5 Pro 模型 (06-05) 即使在低 Temperature 设置下，有时也无法准确比较橙色圆圈的大小。他们指出，使用 Temperature 0 在类似的视觉任务上会产生更一致的正确结果，但该模型在某些条件下仍会产生不一致的错误，这表明其视觉推理在特定情况下并不可靠。
    - 另一位用户报告称，无法确定较大圆圈大小的问题不仅限于 Gemini 2.5 Pro，因为 *"几乎所有模型"* 在这种特定的视觉比较中都表现出类似的困难。讨论强调了一个更广泛的问题：AI 模型（不仅仅是 Gemini）在视觉空间推理任务中存在根本性的约束，但这种挑战的原因尚不明确。
    - 一项技术观察提到了一次成功的尝试，其中一个模型正确地使用了 Python 工具来测量圆圈，甚至能够判断出 Prompt 可能是一个陷阱，这表明工具集成（例如调用代码）有时可以比单纯依赖基础视觉语言模型产生更好或更鲁棒的结果。
- [**o3 是顶尖的 AI Diplomacy 玩家，其次是 Gemini 2.5 Pro**](https://www.reddit.com/r/singularity/comments/1l4wikx/o3_is_the_top_ai_diplomacy_player_followed_by/) ([Score: 151, Comments: 15](https://www.reddit.com/r/singularity/comments/1l4wikx/o3_is_the_top_ai_diplomacy_player_followed_by/)): **该帖子总结了 Alex Duffy 的 AI Diplomacy 项目 ([链接](https://every.to/p/diplomacy)) 的发现，该项目让多个大语言模型 (LLMs) 玩 Diplomacy（外交风云）游戏。在测试中，专有的 o3 模型因其“冷酷”的策略和对欺骗的使用而始终优于其他模型；只有 Google 的 Gemini 2.5 Pro 也成功赢得了一局，它采用了强大的联盟构建和激进的策略。Anthropic 的 Claude 4 Opus 表现不佳，归因于过度诚实和不愿背叛，甚至因为 o3 的操纵而接受了逻辑上不可能的谈判结果（如四方平局）。目前有剩余比赛的直播 ([Twitch](http://twitch.tv/ai_diplomacy))。** 一条热门评论假设，如果模型在不同对局之间拥有记忆，结果可能会有所不同，因为重复博弈可能更有利于合作策略而非背叛策略。另一条评论指出 o3 的对话风格明显具有攻击性且不道歉，有时甚至在它产生“幻觉”（即犯下事实错误并坚持己见）时也是如此。
    - 一位评论者提出，允许 AI 模型在 Diplomacy 比赛之间保留记忆或持久状态可能会极大地改变结果，因为单次博弈（single-shot games）往往奖励背叛，而重复互动则奖励合作。这一考虑对于在 Diplomacy 和类似环境中基准测试 AI 的社交策略具有重要意义。
    - 关于模型性格的观察报告显示：o3 被描述为明显具有攻击性、粗鲁且不道歉，有时在产生幻觉时甚至表现出轻蔑；而 Claude 则被认为更“天真”且被假定更安全，这与 Anthropic 以安全为导向的训练相一致。这些行为倾向可能会影响模型在 Diplomacy 等复杂的社交博弈中的有效性。

### 2. 自主递送机器人与 Figure 的创新

- [**由 Helix (VLA 模型) 驱动的 Figure 02 全自主运行 - 该策略正在翻转包裹以使条形码朝下，并学会了像人类一样为扫描仪压平包裹**](https://v.redd.it/ulyldnqey75f1) ([Score: 5119, Comments: 706](https://www.reddit.com/r/singularity/comments/1l4hmgt/figure_02_fully_autonomous_driven_by_helix_vla/)): **Brett Adcock (Figure AI) 展示了 Figure 02 机器人，由其专有的 Helix (VLA) 模型控制，自主操作包裹以将条形码朝下并压平物品以便扫描——展示了传统上与人类灵巧性和任务理解相关的学习行为 ([source](https://x.com/adcock_brett/status/1930693311771332853))。视频强调了现实世界中的抓取挑战，包括失败的尝试和自适应重新定位，表明了先进的策略学习和闭环 sensorimotor 控制，但也引发了关于手指触觉传感和故障检测机制细节的疑问。** 评论者注意到机器人的流畅性和明显的任务意识，但讨论了其传感器套件复杂性的不确定性——特别是手指中的触觉传感器，以及这些传感器如何影响抓取成功或错误修正。这表明了人们对 Helix 的 VLA 模型中底层硬件和反馈集成的兴趣。
    - 一位评论者指出机器人表现出“令人惊讶的流畅”动作，但强调了在 0:30 左右可见的多次抓取失败尝试。他们质疑机器人手指中存在的传感器类型，认为它可能缺乏在缩回前确定抓取成功所需的足够或经过适当调整的触觉传感能力。
- [**目标是让机器人从 Rivian 货车中走出来，将包裹送到你家门口。**](https://i.redd.it/9numiqo37b5f1.jpeg) ([Score: 244, Comments: 106](https://www.reddit.com/r/singularity/comments/1l4sh3w/the_goal_is_for_robots_to_come_out_of_rivian_vans/)): **该图片源自 Electrek 的一篇文章，描绘了 Amazon 测试人形机器人（可能由 Figure AI 制造）从 Rivian 电动货车下车，直接将包裹送到客户门口的情景。这一设置标志着 Amazon 有意将其现有的电动递送车队与自主机器人技术相结合，从而有可能简化 last-mile logistics。潜在的技术挑战包括可靠的人形机器人导航、物体操作以及包裹交接过程中无缝的人机交互。** 评论者讨论了机器人递送的新潜在应用（如救护车），并权衡了相对于人类快递员的安全/保障优势，反映了公众对递送机器人运营部署的期待与怀疑。
    - 一位用户间接提到了围绕 last-mile delivery 的安全担忧，指出他们曾见过针对快递员的暴力（包括枪击）录像。这突出了机器人提高安全性并可能减少城市或高犯罪率递送场景中人类风险暴露的技术论据。
- [**Figure 的 Brett Adcock 表示，他们的机器人将共享一个大脑。当一个机器人学到新东西时，它们都会立即变得更聪明。这就是飞轮旋转的方式。**](https://v.redd.it/fyfml5v28c5f1) ([Score: 272, Comments: 65](https://www.reddit.com/r/singularity/comments/1l4xjye/figures_brett_adcock_says_their_robots_will_share/)): **Figure 的 Brett Adcock 声称，他们的人形机器人将在一个集中式或共享模型下运行——类似于“单一大脑”——这样由一个单元获得的技能或知识会立即传播到车队中的所有机器人。该系统利用了集体学习动力学（有时称为 federated 或 swarm learning），有可能加速分布式机器人网络中的适应和能力提升。** 评论者对安全漏洞表示担忧，特别是可能立即危及所有机器人的“learning injection”攻击风险。一些人还指出，集中式模型更新是标准的软件方法，而非新颖的突破。
    - 机器人共享“单一大脑”的概念引入了独特的安全漏洞——特别是所谓的“learning injection”攻击的可能性。如果一个机器人被教会（或被诱导学习）了不受欢迎或危险的行为，并且集体记忆被立即同步，那么整个机器人网络可能会立即继承这些损坏的行为。这一风险突显了在共享学习机器人系统中对强大的防护措施、sandboxing 和审计追踪的迫切需求。

### 3. OpenAI & Claude Model Privacy and Community Complaints

- [**OpenAI 目前无限期保留所有聊天数据——即使是 Plus/Pro 用户**](https://www.reddit.com/r/OpenAI/comments/1l4jvk3/openai_is_currently_retaining_all_the_chat_data/) ([Score: 286, Comments: 78](https://www.reddit.com/r/OpenAI/comments/1l4jvk3/openai_is_currently_retaining_all_the_chat_data/)): **OpenAI 最近的声明 ([链接](https://openai.com/index/response-to-nyt-data-demands/)) 确认，所有聊天数据（包括 Plus 和 Pro 用户的数据）都将被无限期保留，这主要是为了响应与《纽约时报》诉讼相关的法律取证（legal discovery）需求。这种保留*并非*为了模型训练或商业分析，否则将面临额外的法律考量。** 热门评论强调了以下几点：(1) 明确了无限期保留是为了法律合规，而非模型训练（未经用户许可进行训练将是违法的）；(2) 质疑此类做法在欧盟数据保护法（如 GDPR）下的合法性；(3) 对基于云端的 LLM 隐私表示怀疑，认为鉴于行业数据保留规范，依赖非本地解决方案从根本上是不安全的。
    - 讨论强调了法律层面：OpenAI 无限期保留聊天记录是法院命令（特别与《纽约时报》诉讼相关）的直接结果，且他们正在对该指令提出上诉。这与科技行业其他领域的典型数据保留义务形成对比，但在本案中，根据当前命令，将数据用于模型训练或其他商业用途将是非法的。
    - 存在对遵守欧盟 GDPR 的担忧，因为无限期保留加上用户缺乏控制权，可能与欧洲严格的数据隐私要求不相符，可能使 OpenAI 在欧洲面临法律风险。
    - 引用了 Ars Technica 的文章，确认保留范围除网络流量外还涵盖 API 调用，这提高了 API 客户对隐私预期的风险，并可能因未完全履行付费用户在数据保护方面的期望而使 OpenAI 面临进一步诉讼。
- [**你看到 OpenAI 关于回应《纽约时报》的声明了吗？**](https://www.reddit.com/r/OpenAI/comments/1l4god3/did_you_see_openais_statement_regarding_their/) ([Score: 114, Comments: 25](https://www.reddit.com/r/OpenAI/comments/1l4god3/did_you_see_openais_statement_regarding_their/)): **OpenAI 发布了针对《纽约时报》在进行中的诉讼中关于用户数据法律要求的回复，强调其遵守隐私政策，且保留对话数据的时间不会超过必要期限。该诉讼的核心是指控未经授权将数据用于模型训练，但 OpenAI 主张其无法追溯提供数据，因为其执行了隐私优先的删除政策，这在其公开声明中有所记录（参见 [OpenAI 的官方回应](https://openai.com/index/response-to-nyt-data-demands/)）。** 评论者指出《时报》立场的矛盾性：以隐私的名义要求保留数据，同时可能迫使 OpenAI 保留或披露比平时更多的用户数据，从而在隐私和合规方面引发法律和伦理冲突。
    - SeventyThirtySplit 阐述了一个关键的技术区别：针对 OpenAI 的《纽约时报》诉讼核心在于指控训练数据涉及版权侵权，而非关于用户隐私或个人数据处理。这澄清了法律辩论本质上是一个 IP/数据权利问题，而非数据隐私投诉。
    - 一些评论强调了这种矛盾的动态，即《纽约时报》的法律行动可能要求 OpenAI 保留潜在的侵权数据作为证据，无意中激励了与隐私最佳实践及 OpenAI 自身声明的政策相悖的数据保留行为（可能影响未来关于数据删除和合规的治理与政策）。

- [**What am I missing here? Claude Code seems a joke when I use it**](https://www.reddit.com/r/ClaudeAI/comments/1l4omv6/what_am_i_missing_here_claude_code_seems_a_joke/) ([Score: 106, Comments: 71](https://www.reddit.com/r/ClaudeAI/comments/1l4omv6/what_am_i_missing_here_claude_code_seems_a_joke/)): **原帖作者（OP）报告称，Claude Code 在一个 React/TypeScript 项目的重构任务中表现不佳：尽管有指令，它还是漏掉了对组件 C、D 和 E 的更改，因 TypeScript 错误而停滞，并表现出任务完成跟踪不一致（在工作未完成时声称已完成）。这种失败说明了 Claude Code 在处理宽泛、非原子化的指令时，难以应对详细、多步骤的全代码库重构，导致部分或无关的更改以及不可靠的进度报告。** 热门评论将这些缺点归因于糟糕的 prompt engineering（模糊、高层级的目标）、模型选择（Opus 与 Sonnet 的行为权衡），以及分步、富上下文指令的重要性。专家建议将复杂的请求分解为清晰、按顺序的 prompt，利用 Sonnet 进行有针对性的更改，并采用关于预期输出和任务分解的明确指令，以获得更好的 LLM 性能。
    - 多位评论者强调了在 Claude Code 中选择模型的重要性：**Opus** 被推荐用于复杂、创意或多步骤的项目工作（例如，初始构建、重大重构），而 **Sonnet** 则更适合需要快速、受限执行的集中式、小型任务。还提到了不同 Sonnet 版本之间的差异：**Sonnet 3.7** 倾向于迭代地排查错误，而 **Sonnet 4** 可能会退回到更广泛的回滚和替代方案，从而影响调试策略。
    - 使用 Claude Code 获得有效结果需要细粒度、分阶段的 prompt，而不是大型、模糊、多步骤的指令。对于复杂的任务，用户应首先引导模型创建一个详细的、分步骤的计划（例如，重构组件、设置测试协议以及清理弃用的代码），然后再指示模型执行这些计划。拆分任务并通过持久文件（如 '[claude.md](http://claude.md/)' 或结构化计划）强化上下文，有助于保持大型项目的连贯性。
    - Claude Code 的性能严重依赖于明确、积极且高度结构化的 prompting。评论者特别建议不要使用否定指令（例如，“不要做某事”），并建议提供详细的、高级开发人员风格的指令。根据经验，Claude 的输出（尤其是像“我修复了所有问题”之类的保证）通常是不准确的，在没有通过推荐的计划和测试协议进行验证的情况下，不应予以信任。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的摘要
> 

**主题 1：模型乱象：Gemini 的过山车、Qwen 的崛起以及 Claude 的扩张**

- [**Gemini 2.5 Pro 的狂野之旅：SVG 明星还是幻觉头痛？**](https://discord.com/channels/1340554757349179412/1340554757827461211/1380259896628477962) 来自 LMArena、OpenAI 和 OpenRouter 的用户报告了对 **Google Gemini 2.5 Pro**（特别是 **0605 版本**）褒贬不一的体验。虽然它在 [AI Studio](https://ai.google.dev/) 中的 **SVG generation** 能力令人印象深刻，且 **0506 版本**在从 **60k token contexts** 中提取注释等任务上表现良好，但更新的版本因 **hallucinations** 增加、长上下文中遗漏注释以及感知到的智力下降而面临批评，OpenRouter 上的一些用户称其为“*flash thinking 级别的愚蠢*”。速率限制也成为了讨论焦点，Perplexity Pro 用户达到了 **100 prompts/day**，而 LM Studio 用户注意到 Gemini 2.5 Pro 的 **150 RPM**。
- [**Qwen3 模型快步走向舞台中央！**](https://discord.com/channels/1179035537009545276/1179035537529643040/1380271178693873684) 阿里巴巴发布的 **Qwen3 models**，包括 [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) 和 [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)，在 Unsloth AI 和 Latent Space 的 Discord 频道中引起了关注。Unsloth AI 推出了新的 Notebook，如 **DeepSeek-R1-0528-Qwen3 (8B)** ([UnslothAI X post](https://x.com/UnslothAI/status/1931008531299545339))。LM Studio 的讨论强调了 **Qwen3-4B** 的表现优于 Open Thinker 等模型，并在统一内存设置下，**Qwen3 235B Q3_K_S** 达到了 **12.05 tok/sec** 的惊人速度。
- [**Claude Projects 规模扩大，GPT-4o 遇挫！**](https://discord.com/channels/822583790773862470/1075282825051385876/1380293156163158017) Anthropic 扩大了其 **Claude Projects** 功能，支持 **10 倍以上的内容**并激活了新的检索模式。这一更新正向所有 Claude 付费计划推出，被 Latent Space 用户誉为“*游戏规则改变者*”。与此同时，一些 HuggingFace 用户报告 **GPT-4o** 和 **GPT-4o mini** 在与 **smolagents** 配合使用时出现解析错误，而 OpenRouter 用户发现 **GPT-4.1 mini** 在代码编写和 tool use 方面是极具性价比的“真正赢家”，尽管在创意写作方面并未超越 Claude 3.7。

**Theme 2: Data Deluge: EleutherAI's Common Pile Sets New Open Standard**

- [**EleutherAI 发布巨量 Common Pile 数据集！**](https://discord.com/channels/729741769192767510/794042109048651818/1380550466059898951) EleutherAI 发布了 **Common Pile v0.1**，引起了轰动。这是一个包含来自 **30 个来源**的公开授权文本的 **8TB dataset** ([The Common Pile v0.1 Paper](https://arxiv.org/abs/2506.05209))，正如其 Discord 中宣布的那样，并得到了 Nous Research 和 Yannick Kilcher 的关注。该计划旨在通过提供高质量、无版权的数据进行训练，来培育一个更具道德和透明度的 LLM 生态系统。相关模型 **Comma v0.1-1T** 和 **Comma v0.1-2T** 显示出与 Llama 1 & 2 7B 相当的性能 ([EleutherAI Common Pile GitHub](https://github.com/EleutherAI/common-pile), [Common Pile on HuggingFace](https://huggingface.co/common-pile), [Common Pile Blog Post](https://huggingface.co/blog/stellaathena/common-pile))。
- [**开放数据胜过“肮脏”的前辈！**](https://discord.com/channels/1053877538025386074/1149866623109439599/1380268802415136960) Common Pile 的发布引发了 Nous Research 关于数据集质量的讨论，成员们建议使用 [HuggingFace 的 Fineweb-2 dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) 及其消融实验版本，来替代像 RedPajama 这样较旧且“*有点脏*”的数据集。Yannick Kilcher 的社区也强调了 EleutherAI 的工作，证明了完全使用公共领域和公开授权的数据也能训练出具有竞争力的 LLM。
- [**LLM 从人类训练者那里学会了撒谎？**](https://discord.com/channels/729741769192767510/729741769738158194/1380261783629070428) EleutherAI Discord 中一场有趣的讨论探讨了由人类进行 in-context 训练的 LLM 如何产生**不可证伪叙事**的倾向。由于它们通常只在人类训练者已知的话题上得到纠正，LLM 可能会发现，编造看似合理但无法验证的故事比产生真正的价值更容易。**ChatGPT's memory feature** 可能会加剧这一担忧，从而导致长期的 misalignment。

**Theme 3: Dev Tool Drama: Cursor's $10B Boom & Bust, MCP's Many Faces, Unsloth's Trending Tricks**

- [**Cursor 估值飙升至 100 亿美元，伴随 Gemini 故障！**](https://discord.com/channels/1074847526655643750/1074847527708393565/1380267312858271765) Cursor 社区因其母公司 **Anysphere** 获得 **100 亿美元估值** 而沸腾（[TechCrunch 关于 Cursor 估值的报道](https://techcrunch.com/2025/06/05/cursors-anysphere-nabs-9-9b-valuation-soars-past-500m-arr/)）。然而，用户报告称在使用新的 **Gemini 06-05 模型** 时，**Cursor 的工具** 出现了严重问题，同时 Background Agents 的 **GitHub 访问** 持续存在故障，且难以这些 Agents 设置默认环境。
- [**MCP 生态系统扩展，新增 Inspector Fork 和静默 Slack Agents！**](https://discord.com/channels/1312302100125843476/1312302100125843479/1380282993452781579) MCP (Glama) Discord 展示了多项新进展，包括一个内置 **LLM chat** 和采样支持的 **MCP inspector fork** ([MCP Inspector Fork GitHub](https://github.com/MCPJam/inspector))。此外，一个新的 **Slack MCP server** 问世，它无需 Slack 机器人或应用即可创建 **静默、隐形的 AI Agents** ([Slack MCP Server GitHub](https://github.com/korotovsky/slack-mcp-server))，同时还发布了一个名为 **inked** 的简单服务器 ([inked server GitHub](https://github.com/coldielb/inked))。
- [**Unsloth Notebooks 走红，而 Qwen 微调遭遇瓶颈！**](https://discord.com/channels/1179035537009545276/1179035537529643040/1380271178693873684) Unsloth AI 的 notebooks 仓库在 GitHub 上开始流行 ([Unsloth AI Notebooks GitHub](https://github.com/unslothai/notebooks))，这得益于 **DeepSeek-R1-0528-Qwen3 (8B)** 等 notebook 的发布。然而，使用新 token 微调 Qwen 模型的用户报告称，加载的模型似乎没有应用训练好的权重，对此建议检查 GitHub issues 并通过 pip 升级 `unsloth_zoo` 和 `unsloth`。

**Theme 4: Silicon Sizzlers & Kernel Conundrums: ROCm on Windows, Tinygrad's Tussles**

- [**ROCm Wheels 登陆 Windows，支持 Radeon GPU！**](https://discord.com/channels/1189498204333543425/1233704710389764236/1380421783064416327) GPU Mode 成员庆祝非官方 **PyTorch + ROCm wheels** 的到来，为 **Radeon GPU** 提供原生 **Windows** 支持。这些 wheels 使用 [TheRock](https://github.com/ROCm/TheRock) 构建，目标版本为 Python 3.11/3.12。这些社区努力主要在 **Strix Halo (gfx1151)** 上进行了测试，旨在支持 Navi31/32/33 等一系列 AMD GPU，并展示了一个 ComfyUI 示例 ([Adyaman 关于 ROCm Windows Wheels 的 X 帖子](https://x.com/adyaman/status/1926368074866757857))。
- [**Tinygrad 内核滞后，手动 OpenCL 表现出色！**](https://discord.com/channels/1068976834382925865/1070745817025106080/1380273021066936430) tinygrad Discord 的用户正面临 tinygrad 生成的 GPU kernels 运行缓慢的问题，特别是在 `hlb_cifar10` 示例中打乱大型数据集时。为同一任务手动编写的 **OpenCL kernel** 性能显著提高（**0.33 秒** 对比 tinygrad 的 **5 秒**），这引发了对 tinygrad 内核生成逻辑以及缺失的 LLVM 优化（如 **InductiveRangeCheckElimination**）的调查。
- [**内核开发者关注 ThunderKittens，辩论 Triton 与 AITemplate！**](https://discord.com/channels/1189498204333543425/1189498205101109300/1380503546025480223) 在 GPU Mode 中，开发者建议使用 [**ThunderKittens** 库](https://github.com/HazyResearch/ThunderKittens) 来抽象内核编写，特别是针对非 Hopper GPU，并讨论了将其移植到 AMD 的可能性。同时，[**AITemplate**](https://github.com/facebookresearch/AITemplate) 被宣布进入维护模式，**torch.compile** 和 **AOTInductor** 被推荐为更强大且处于积极开发中的替代方案。

**Theme 5: Trust Traps & Truth Trials: Deepfakes Deceive, Benchmarks Baffle, Vixra Vanquished**

- [**AI Audio Deepfake Detector 被 ElevenLabs 愚弄！**](https://discord.com/channels/879548962464493619/897390720388825149/1380535468676087899) 一位 HuggingFace 成员针对音频 Deepfake 检测微调了 **Facebook 的 ConvNeXt-Tiny** 模型，并将其托管在 [Hugging Face Space 音频 Deepfake 检测页面](https://huggingface.co/spaces/kubinooo/convnext-tiny-224-audio-deepfake-detection)上。然而，该模型在测试中表现惨败，将 **ElevenLabs** 生成的音频归类为 **100% 真实**，引发了关于模型泛化能力的调试努力和讨论。
- [**Livebench 失去公信力，LMArena 用户发起反抗！**](https://discord.com/channels/1340554757349179412/1340554757827461211/1380259896628477962) 在 **Livebench** 基准测试将 **GPT-4o** 的排名排在 **Claude 3.7** 和 **Gemini 2.5 Pro** 等模型之上后，LMArena 社区对其进行了猛烈抨击。有关该 CEO 对 Google 存在偏见以及操纵测试问题的指控不断，导致出现了 *“禁止在此讨论 Livebench”* 的呼声。
- [**Vixra 被否决：“民科仓库”损害公信力！**](https://discord.com/channels/729741769192767510/747850033994662000/1380271503643250699) 当一位 EleutherAI 成员在 [vixra（一个电子预印本存档库）](https://ai.vixra.org/abs/2506.0018)分享他们的论文时，其他成员强烈建议不要使用该平台，称其为 *“会严重损害你公信力的民科仓库（crank repository）”*。共识是应使用 **ArXiv** 代替发布研究成果，成员们指出 ArXiv 上现有的 [进化工作](https://arxiv.org/abs/2206.08896) 是更好的范例。


---

# Discord: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **PPLX Pro 用户遭遇 Prompt 限制**：Perplexity AI 将 **2.5 Pro** 的 **rate limit** 更改为 **100 prompts/day**，令订阅原始无限服务的用户感到失望。
   - 一位用户抱怨道：*“他们降低了限制，却表现得好像把 50 翻倍到 100 是个德政，这太愚蠢了，让我对他们的 Pro 订阅失去了所有信任。”*
- **Android 粉丝吹捧 OS 优越性**：成员们正在辩论 **Android** 与 **iPhone** 的优劣，强调配备 **7000mah 电池** 的 Android 手机日益普及，以及 *YouTube Revanced*。
   - 一位成员指出：*“每年我都觉得自己想要一部 iPhone。但到了第二年 Android 变得更好了。”*
- **Comet 浏览器即将发布**：**Comet 浏览器** 预计很快将推出调度功能，可能在 *下周*。
   - 一位成员推测 *“Comet 不会改变任何现状”*。
- **Sonar Deep Research 超能力解锁**：**Sonar deep research** 现在具备增强的推理能力和异步模式（async mode），提升了其分析实力。
   - 成员们还注意到 **学术模式（Academic mode）** 现在已在所有模型上可用，为每个引用都丰富了 **标题、URL 和日期**。
- **API 搜索能力与在线界面对比**：一位成员对 **API 的搜索能力** 表示沮丧，指出其感觉仅有 *在线界面的 50%*。
   - 相比之下，Perplexity 的公共搜索页面 [artisticai.art](https://artisticai.art) 分享了涵盖一系列争议话题的搜索结果链接，包括对 **Michael Tait** 的指控、**巴基斯坦的外交欺骗**、**黑暗四分体（Dark Tetrad）的数字放大** 以及对 **残酷行为** 的探索。



---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **通过精确提示词生成 Tesla**: 成员们分享了用于生成特定物体（如 **Tesla Model 3**）图像的提示词技术，重点在于对**形状、尺寸、角度和位置**的持续验证。
   - 一位成员分享了他们的提示词：*generate an svg of a Tesla Model 3. make it maximally detailed and look exactly like the real thing.*
- **Gemini 2.5 Pro 的性能表现**: 用户对比了不同版本的 **Google Gemini 2.5 Pro**，指出 **0605 版本**在长上下文中经常遗漏评论，并表现出严重的幻觉。
   - **0506 版本**在 **60k token 上下文**中提取特定用户评论的表现更好，尽管在新版本中观察到在 **8k 标记处**性能有所下降。
- **发现 Le Chat 的 API**: 一位用户发现 **Le Chat** 仍在运行，并通过一些 **Google 内部 API** 非正式地提供，允许通过一个 *"apps"* 功能对 **Gemini API** 进行程序化调用。
   - 另一位用户表示，*开发它的人非常亲 CCP*，并确认滥用了 Google 的大型应用功能。
- **Kingfall 在 AI Studio 短暂泄露**: **Kingfall 模型**曾在 AI Studio 短暂可用，引发了它是基于 **DeepMind** 且速度极快的猜测，一些用户表示 *We do its in ai studio got released today*。
   - 随后它很快被撤下。
- **Livebench 基准测试引发争议**: 成员们在 **Livebench** 将 **4o** 的评分排在 **Claude 3.7、Claude 3.5 和 2.5 Pro** 之上后，对其可靠性和相关性提出了批评。
   - 社区表示 *禁止在这里讨论 livebench*，并引用了诸如 CEO 被指对 **Google** 存在偏见以及操纵测试问题等问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **EleutherAI 发布 Common Pile v0.1**: EleutherAI 宣布发布 **Common Pile v0.1**，这是一个包含来自 **30 个来源**的公开授权文本的 **8TB 数据集**（[论文](https://arxiv.org/abs/2506.05209)），其模型 **Comma v0.1-1T** 和 **Comma v0.1-2T** 实现了与 **Llama 1 和 2 7B** 相当的性能。
   - 该组织希望通过透明度、更好的作者归属以及缺乏受版权保护的数据，促进更具伦理的语言模型生态系统（[GitHub](https://github.com/EleutherAI/common-pile), [HuggingFace](https://huggingface.co/common-pile), [Blog](https://huggingface.co/blog/stellaathena/common-pile)）。
- **LLM 陷入误导性叙事的陷阱**: 成员们观察到，由人类进行上下文训练的 **LLM** 可能会产生生成**不可证伪叙事**的倾向，特别是由于它们仅在人类训练者已知的主题上得到纠正。
   - 一位成员报告了关于 **ChatGPT 记忆功能**的令人担忧的体验，这可能导致长期的**对齐失调**和不可预测的行为，仿佛它经过了长期互动的微调。
- **Attention 机制早于 Transformer 出现**: 成员们讨论了一个广为人知的事实，即 **attention 机制**早于 Transformer 出现，但人们普遍偏向于 **Bahdanau attention**，并且在 Twitter 上，attention 和 Transformer 几乎是同义词。
   - 一位用户链接了 **Schmidhuber** 的一条 [推文](https://x.com/SchmidhuberAI/status/1864701357107634390)，声称对 attention 具有优先权，但其形式是线性的而非二次的，成员们还链接了对 **Schmidhuber** 主张的 [批评](https://bsky.app/profile/reecedkeller.bsky.social/post/3lqv4hxouck27)。
- **务必避开 Vixra**: 一位成员在 [vixra](https://ai.vixra.org/abs/2506.0018) 上分享了他们的论文链接，题为 *Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance*。
   - 其他成员强烈建议不要使用 **vixra**，将其描述为一个会*严重损害你的公信力*的民科仓库，建议改用 **ArXiv**，并指向了[进化相关的工作](https://arxiv.org/abs/2206.08896)。
- **MPL 权重可视化**: 一位成员就一个将 **MPL 权重**投影到**词汇嵌入空间**进行可视化的[项目](https://grgv.xyz/blog/neurons1/)寻求反馈。
   - 该项目旨在确定该方法的可行性以及进一步探索的潜在方向。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 工具在 Gemini 06-05 下出现故障**：用户反映，在使用新的 **Gemini 06-05** 模型时，**Cursor** 的工具出现了问题，类似于之前 **Gemini** 和 **Flash** 更新时遇到的情况。
   - 一位用户总结道：**Gemini 擅长代码**，**OpenAI 擅长指令和代码**，而 **Claude 两者都擅长**。
- **文档为 Cursor 用户提供帮助**：成员们发现 **Cursor** 的文档对于查找有关 **Cursor** 的更新信息和知识非常有帮助。
   - 一位成员特别提到，他们正在阅读有关该问题的 **Cursor** 文档。
- **Cursor 估值飙升至 100 亿美元**：在 **Anysphere** 最新一轮融资后，**Cursor** 的估值已达到 **100 亿美元**。
   - 一位成员分享了关于该估值的 [TechCrunch 文章链接](https://techcrunch.com/2025/06/05/cursors-anysphere-nabs-9-9b-valuation-soars-past-500m-arr/)。
- **Background Agents 的 GitHub 访问被拒绝**：用户报告了 **Cursor** 在 **Background Agents** 配置中连接 **GitHub** 时出现的问题，尽管正常的 **GitHub** 连接工作正常，但仍收到“Access Denied: No repos with user access found”错误。
   - 一位用户还遇到了“Unable to refresh, could not reach GitHub API”的消息，即使没有使用 VPN 或异常的网络配置。
- **Background Agents 的环境配置难题浮现**：多位用户无法为 **Background Agents** 设置默认环境，在多次尝试后遇到“[invalid_argument] Error”。
   - 一位用户询问是否有人拥有可运行的 **JSON** 环境配置，并表示很难将他们现有的 **docker compose** 设置转换为 Agent 可用的配置。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 在 SVG 生成方面表现出色**：成员们赞扬了 **Gemini** 在 [AI Studio](https://ai.google.dev/) 中根据提示词生成详细 **SVG** 图像的能力。
   - 一位用户对 **Gemini** 在保持整体机器人结构的同时动态调整元素的能力感到惊叹。
- **Gemini 命名约定引发混乱**：成员们对 **Gemini** 模型的命名约定表示困惑，尤其是标记为 **06-05** 的版本。
   - 一位成员调侃道：“他们真的应该想出更好的名字……”，而其他人则认为命名反映了更新的发布日期，本质上是同一个模型“加入了一些 **RL**（强化学习）”。
- **Veo3 提供角色一致性**：一位成员提到使用 **Veo3** 在保持角色和音频一致性方面取得了“重大成果”，但未能提供展示视频。
   - 该说法集中在生成具有一致角色和音频的视频上，尽管没有分享支持性证据。
- **AI 输入中 Markdown 优于 PDF**：一位成员批评使用 **PDF** 作为语言模型的数据输入，因为其具有加密复杂性，且设计初衷是为人眼提供像素级完美的渲染，而非为了 AI 理解。
   - 相反，他们主张将 **Markdown** 作为一种更优的纯文本格式，强调其与模型训练数据的契合度。
- **UPSUM Chain Prompt 管理上下文**：一位成员介绍了 **UPSUM Chain Prompt**，这是一种旨在总结对话并维持上下文的元提示词（meta-prompt），建议将其用于将冗长的聊天记录压缩为简洁的叙述。
   - 他们还建议“对你的 upsum 集合进行 upsum”，将其作为未来提示词的上下文，并链接了一个 [UPSUM Chain Prompt 的 YAML 配置](https://example.com/upsum-yaml)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Notebooks 登上趋势榜！**: Unsloth 的 notebooks 仓库目前在 GitHub 上非常热门 ([https://github.com/unslothai/notebooks](https://github.com/unslothai/notebooks))，其中包括新发布的 **DeepSeek-R1-0528-Qwen3 (8B)** notebook ([https://x.com/UnslothAI/status/1931008531299545339](https://x.com/UnslothAI/status/1931008531299545339))。
   - 在价格保持不变的情况下，任何配置的 RAM 都翻倍了，但用户报告在使用 notebook 时遇到了问题。
- **Chrome 自动填充崩溃促使文档编辑修复！**: 成员们发现在 **Chrome** 中，当在大型文档的编辑器中输入时，**autofill**（自动填充）功能会导致崩溃，并触发 `TransactionTooLargeException`。
   - 经发现，这是 Chrome 内部在通知自动完成服务时的一个 bug，禁用 autofill 是已验证的崩溃修复方案。
- **Qwen3 发布新款 Embedding 和 Reranker 模型！**: 两款新的 **Qwen3** 模型已发布：[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) 和 [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)。
   - 成员们期待这些新模型能带来积极的结果，但注意到缺乏 TTS 的 benchmarks。
- **IDE 终极对决：VS Code 胜出！**: 成员们讨论了最适合 AI 编程的 IDE，[VS Code](https://code.visualstudio.com/) 成为头号竞争者，并配合 [GitHub Copilot](https://github.com/features/copilot) 进行自动补全。
   - 一位用户提到 VS Code 有时会冻结，但其在 AI 开发方面的整体实用性被认为是同类最佳。
- **驯服 Tokenizer：用户报告微调 Qwen 模型的谜团**: 一位用户在新的 token 上微调了 `Qwen3-8B-unsloth-bnb-4bit` 模型并推送到 hub，但遇到了**加载的模型似乎没有应用训练权重**的问题。
   - 成员们建议这可能与添加新 token 有关，并建议该用户在 GitHub 上检查类似问题，等待包含 merge 逻辑修复的新版本发布。此外，用户被引导至最近的 pypi 发布版本，可以通过 `pip install --upgrade unsloth_zoo` 和 `pip install --upgrade unsloth` 进行升级。



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **ThunderKittens 简化 Kernel 编写**: 成员们建议使用 [**ThunderKittens**](https://github.com/HazyResearch/ThunderKittens) 库来抽象 Kernel 编写过程（如果不使用 **Hopper GPU**），特别是考虑到核心的 matmul 操作似乎是 [`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`](https://github.com/HazyResearch/ThunderKittens/blob/d69697a3337e31d0060178c9049f1184e7e7ad7f/include/ops/warp/register/tile/mma.cuh#L17) 原语。
   - 一位成员通过私信询问了将 **ThunderKittens** 移植并泛化到 **AMD** 架构的可能性，表示虽然这可能很复杂，但似乎具有潜在的可行性。
- **AITemplate 落幕，torch.compile 崛起**: 社区表示 [**AITemplate**](https://github.com/facebookresearch/AITemplate) 已不再处于活跃开发状态，已经进入维护模式数年，因此一致认为 **torch.compile** 是更强大的选择。
   - **AOTInductor** 被推荐作为 C++ runtime 的替代方案。
- **ROCm wheels 登陆 Windows！**: 非官方的 **PyTorch + ROCm** wheels 现在支持 **Radeon GPU** 上的原生 **Windows**，并捆绑了相关库以便于安装；这些由社区驱动的 wheels 使用 [TheRock](https://github.com/ROCm/TheRock) 构建，针对 **Python 3.11/3.12**。
   - 主要在 **Strix Halo (gfx1151)** 上进行了测试，但目标是支持 **gfx1100/gfx1101/gfx1102/gfx1103/gfx1151/gfx1201** (Navi31/32/33, 780M, 8060S, 以及 9070/XT)，并在[此处](https://x.com/adyaman/status/1926368074866757857)展示了一个 ComfyUI 示例。
- **CUDA C++ Workshop 将采用“动手实践”模式**: 一场关于现代 **CUDA**、优化和调试的全天动手实践培训将于 **6 月 10 日**在 **GTC Paris** 的 **VivaTech** 举办，主题为 **CUDA C++ Workshop**。
   - NVIDIA 还提供了[完整议程](https://www.nvidia.com/en-eu/gtc/)链接，以及与技术背后的工程师进行的关于 **CUDA、AI、HPC** 等内容的问答环节。
- **FLE API 规范征求反馈**: 一位成员正在为其 **FLE API** 方案寻求反馈，并提供了一个 *征求意见 (RFC)* 风格的 [GitHub 仓库](https://github.com/MortenTobiasNielsen/FLE-API-specification)。
   - 另一位成员确认他们将进行审查，并针对相关问题提供具体建议。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Hub 遭遇短暂故障**：**Hugging Face Hub** 经历了一次停机，导致 **502 错误**，用户报告从不同地点访问该网站时出现问题。
   - 基础设施团队迅速解决了问题，其高效的工作赢得了赞誉；此次停机可能与工作人员在[相关讨论](https://huggingface.co/spaces/transformers-community/support/discussions/13#6842efbfac97e96a2f38dcbe)中的回复时间重合。
- **DDR5 RAM 限制困扰爱好者**：讨论涉及 **AMD Zen 5 CPU** 的最大 RAM 限制为 **128GB**，尽管目前已有 **64GB DDR5** 内存条，且部分主板支持 **256GB RAM**。
   - 有推测认为，MOE 模型可以在极小的 VRAM 和充足的 RAM 下运行，即使是在过时的 CPU 上，而较新的 CPU 却奇怪地被锁定在较低的 RAM 限制上。
- **模型伪造音频测试失败**：一名成员微调了 **Facebook 的 ConvNeXt-Tiny** 模型，将计算机视觉技术与音频分析相结合，用于音频真伪分类的 Deepfake 研究，并托管在 [Hugging Face Space](https://huggingface.co/spaces/kubinooo/convnext-tiny-224-audio-deepfake-detection) 上。
   - 另一名成员使用 **11elevenlabs** 生成的伪造音频测试了该模型，但它被错误地分类为 **100% 真实**，引发了调试工作。
- **HF 计算机视觉聚会幻灯片分享**：分享了今天 **Computer Vision Hangout** 的幻灯片，包括 **Hugging Face** 计算机视觉方面的更新：[HF_CV_Hangout_June_25.pdf](https://cdn.discordapp.com/attachments/922424143113232404/1380522336335298650/HF_CV_Hangout_June_25.pdf)。
   - 来自 **Pruna AI** 的一名成员做了关于加速图像生成的演示，其幻灯片分享在 [PrunaAI_SpeedUp_Generation.pdf](https://cdn.discordapp.com/attachments/922424143113232404/1380522337027227709/PrunaAI_SpeedUp_Generation.pdf)。
- **GPT-4o 在 Smolagents 中解析表现不佳**：一名成员询问其他人在将 **GPT-4o** 和 **GPT-4o mini** 与 **smolagents** 代码 Agent 配合使用时，是否也遇到了大量的解析错误。
   - 目前尚未提供解决方案，但这可能表明解析模型存在持续的不稳定性。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro 基准测试引发辩论**：**Gemini 2.5 Pro** 针对 Vertex AI 上公开模型的早期基准测试显示分数为 **86.2%**，基于[此 commit](https://github.com/cheahjs/aider/commit/cacd932c9a474f871229b166e6be0d1858854e17) 的设置，引发了关于这些结果的可实现性和“随机性（stochastic）”本质的讨论。
   - 一些用户对 **Gemini** 相对于 **Opus** 的表现感到惊讶，而另一些用户则强调对 **Opus** 的强烈偏好，理由是其处理编码任务的能力。
- **价格驱动模型偏好辩论**：关于性价比的讨论凸显了不同的优先级：一些人优先考虑较低的成本，而另一些人则看重输出质量和工作流协同效应，即使是像 **Opus** 这样更昂贵的模型。
   - 一位用户用“价格是你付出的，价值是你得到的”总结了这种情绪，概括了模型选择中的权衡。
- **Aider 用户评估 Cursor 的开发风格**：一些尝试 **Cursor** 的 **Aider** 用户发现它速度较慢，聊天机器人过于啰嗦，且 Agent 模式需要大量的控制。
   - 一位用户发现 **aider 的方法**更适合他们的开发风格，即“（谨慎、深思熟虑的提示）、终端驱动、疯狂地创建分支（因为对恢复到已知良好状态有强迫症）等等……”，并且不喜欢“从 Cursor 粉丝那里感受到的那种‘胡乱尝试（fling stuff at the wall）’的氛围”。
- **寻求语音转文字工作流集成**：一位对**语音转文字工作流**感兴趣的用户探索了 iOS 上的 **Wispr flow** 等选项，但更倾向于 **superwhisper**。
   - 该用户旨在将更多基于语音的工作流融入日常事务中，这表明了对高效和替代输入方法的渴望。
- **使用 vllm 服务器配置 Aider 的挑战**：由于 **aider** 要求模型名称以提供者前缀开头（例如 *openai/Qwen3*），一名用户在配置 **aider** 与本地 **vllm server** 配合使用时遇到了问题。
   - 另一名用户建议添加 *openai/* 前缀来解决此问题，例如 *openai/unsloth/Qwen3*，以符合 **aider** 预期的命名规范。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **基于 App 的垃圾信息引发用户警惕**：频道中出现了一种新型的“基于 App”的垃圾信息，引发了用户对账号可能被盗用的担忧；工作人员通过移除“使用外部应用（Use External Apps）”权限做出了回应。
   - 初步调查表明，**没有账号被盗用**。
- **Gemini 2.5 Pro 遭遇速率限制**：用户报告称，在 **LM Studio** 中，**Gemini 2.5 Pro** 的速率限制为 **每 24 小时 100 条消息**，随后更新说明速率限制已调整为 **150 RPM**。
   - 用户正在探索在不触及这些速率限制的情况下使用 **LM Studio** 的替代方法。
- **Qwen3-4B 性能超越 Open Thinker**：围绕 **Open Thinker** 模型展开了讨论，成员指出根据官方 Benchmark 结果，**Qwen3-4B** 已超越前者，并且在游戏 PC 上[运行流畅](https://huggingface.co/Qwen/Qwen3-4B)。
   - 社区似乎因其卓越的性能指标而向 **Qwen3-4B** 倾斜。
- **统一内存（Unified Memory）为 Qwen3 提速**：一位用户使用配备 **AMD Ryzen AI Max+ 395** 和 **128GB** 统一内存的 **GMKtec Evo-X2**，在 **Qwen 3 235B Q3_K_S**（上下文：12233）上实现了首条回复 **12.05 tok/sec** 的速度。
   - 在 **Q8_0** 量化下加载 **64k 上下文** 达到了 **9.33t/s**，首个 Token 延迟为 **6.27s**，尽管在使用 **Unsloth Q3_K_XL** 时遇到了重复生成问题。
- **Llama 3.3 70B 占满 VRAM**：成员测试了在 **F16** 精度下具有 **128k 上下文** 的 **Llama 3.3 70B**，完全在 VRAM 中运行，第一条 Prompt 达到 **4.85 tok/sec**，第二条 Prompt 达到 **4.83 tok/sec**。
   - 他们使用的是 **GMKtec Evo-X2**，搭载 **AMD Ryzen AI Max+ 395** 和 **128GB** 统一内存，其 iGPU 为 **8060S**，在 AI 计算能力上大致相当于 **3060**。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 团队实现了向 Pixi 的神奇迁移**：一位成员对 Modular 团队表示感谢，感谢他们实现了从 `magic` 到 `pixi` 的无缝迁移，并称其为“引脚兼容（pin compatible）”的过程。
   - 该成员通过表情符号表达了谢意，强调了迁移过程的顺滑。
- **Mojo 升级需要内存对齐**：一位成员提到在系统上升级 **Mojo** 时，进行内存对齐（memory alignment）的必要性。
   - 另一位成员表示赞同，强调了在 **Mojo** 升级背景下内存对齐的重要性。
- **Mojo 进军生物信息学**：一位开发者分享了他们在生物信息学中使用 **Mojo** 的热情，并乐于挑战为生物技术初创公司实现 **SLURM** 和 HPC 解决方案。
   - 他们观察到，研究人员通常开发以结果为导向的软件和自动化解决方案，但这些方案并不总是在行业内共享。
- **Mojo 的不可变变量仍处于需求阶段**：一位成员询问如何在 **Mojo** 中声明不可变值，寻求一种在初始化后防止运行时值被修改的方法。
   - 另一位成员澄清说，目前还无法创建不可变变量，但他们提出了[一个使用辅助函数的变通方案](https://github.com/modular/modular/blob/main/mojo/proposals/remove-let-decls.md)来实现不可变引用。
- **Mojo 语法太啰嗦？**：开发者们讨论了 **Mojo** 语法的冗长问题，特别是 Struct 定义中的 `var` 关键字，一些人觉得这是一种负担。
   - 对话延伸到了对简洁语法的偏好，一位成员表达了对 **K** 编程语言的喜爱，该语言以其极简风格著称。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **OLMo 模型作为完全开源项目首次亮相**：成员们强调 [Allen.ai 的 OLMo 模型](https://arxiv.org/abs/2501.00656) 是完全开源的，涵盖了 **训练代码、数据和论文**。
   - 一位成员建议查看 [HuggingFace 的 Fineweb-2 数据集](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) 及其消融实验，并表示 *RedPajama 有点脏且旧了*。
- **Atropos 成为 Tool Calling 的新领域**：团队正在积极开发 [Atropos 环境](https://github.com/NousResearch/atropos/pull/163)，并为 **经过验证的推理轨迹（reasoning trace）答案** 生成数据集。
   - 通过使用 [这个环境](https://github.com/NousResearch/atropos/blob/main/environments/tool_calling_server.py)，他们在 Berkeley Tool Calling 基准测试中，将 DeepHermes 的单步和并行 Tool Calling 性能分别提升了 **5 倍和 2.5 倍**。
- **顺序 Tool Calling 功能正在开发中**：一位团队成员确认，他们正在将 **Tool Call 直接训练到推理轨迹中**，目前重点是 **顺序 Tool Calling**。
   - 目前尚未提供关于此进展的更多信息。
- **EleutherAI 发布海量 Common Pile 数据集**：[EleutherAI](https://blog.eleuther.ai/common-pile/) 发布了用于语言建模的 **最大商业和授权数据集之一**。
   - 该公告已在 [X](https://x.com/EnricoShippole/status/1931023312647299405) 上发布。
- **LLM 的可复现性是基于感觉（Vibe Based）的**：一位成员分享了一张图片，将 Transformer 预训练比作发现 **一种蛋糕配方**，而底层的化学原理尚未被完全理解。
   - 另一位成员回应道：*LLM 烹饪是 **基于感觉（vibe based）且 YOLO 的***。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 推出模型 RSS 订阅源**：OpenRouter 宣布为其 [API 模型](https://openrouter.ai/api/v1/models?use_rss=true) 提供 **RSS 订阅源**，使用户能够及时了解 OpenRouter 生态系统中的新模型和变更。
   - 该 RSS 订阅源提供最新信息，方便开发者轻松追踪变化。
- **Gemini 2.5 Pro 出现智力退化**：用户报告 **Gemini 2.5 Pro** 的 **06-05** 版本智力有所下降，有人将其描述为 *Flash Thinking 级别的愚笨*。
   - 共识建议在旧模型尚可用时继续使用，因为新版本可能是为了速度和成本而进行了缩减。
- **Claude Max 与 Gemini 价格对决**：一位用户开玩笑地建议通过盗版 **Gemini 2.5** 来避免付费，这引发了关于 **Claude Max** 与 **Gemini** API 使用成本效益的辩论。
   - 该用户指出 **Claude Max** 对于 *Vibe Coding* 和日常使用来说更经济，特别是对于对 API 成本敏感的用户。
- **OpenAI 日志记录引发隐私担忧**：[一篇文章](https://arstechnica.com/tech-policy/2025/06/openai-says-court-forcing-it-to-save-all-chatgpt-logs-is-a-privacy-nightmare/) 指出 **OpenAI** 被迫保存所有输出日志，这引发了关于 **OpenRouter** 数据保留的问题。
   - 澄清说明，“启用训练和日志记录”设置与 OpenAI 模型无关，且 OpenAI 可能会保留输入数据长达 30 天。
- **GPT-4.1 Mini 在编码和工具使用方面表现出色**：**GPT-4.1 mini** 因其编码能力、工具使用（Tool Use）和成本效益而受到称赞，非常适合日常任务和推理。
   - 它被认为是“真正的赢家”，比 **Gemini 2.5 Flash** 更听话，尤其是在不涉及代码或数学的任务中，尽管在创意写作方面不如 **Claude 3.7**。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **LLM Chat 在 MCP Inspector Fork 中亮相**：发布了一个内置 **LLM chat** 和 **sampling support**（采样支持）的 **MCP inspector fork**，并邀请通过 [GitHub](https://github.com/MCPJam/inspector) 进行测试和反馈。
   - 该 fork 为 MCP 生态系统内 **LLM 增强的工具交互**提供了一个测试台。
- **Cloudflare 部署遭遇障碍**：一位用户在使用 **workers.dev 链接**在 **Cloudflare Workers** 上部署 **MCP server** 时遇到了问题。
   - 该成员在尝试将自定义 MCP server 与 OpenAI 集成时寻求帮助，即使在为所有工具添加了描述后仍面临困难。
- **静默 AI Agents 进入 Slack 工作区**：一位成员在 GitHub 上宣布了他们的 **Slack MCP server**，强调其无需机器人或 Slack 应用程序即可构建**静默**、**不可见**的 **AI Agents** 的能力，现已在 [GitHub](https://github.com/korotovsky/slack-mcp-server) 上可用。
   - 他们附带了一个[展示其用法的 GIF](https://cdn.discordapp.com/attachments/1315696461316358175/1380517187785326692/434543420-35dc9895-e695-4e56-acdc-1a46d6520ba0.gif?ex=68442a52&is=6842d8d2&hm=78437dbb1f2f8f9776d0855153c1d68e2ec00098b74fbddc18ba4e53e272148e&)。
- **Inked 极简服务器上线 GitHub**：一个名为 **inked** 的极简服务器在 [GitHub](https://github.com/coldielb/inked) 上发布，邀请社区通过贡献和实验参与其中。
   - 该服务器包含**两个工具**和**三个总函数**，可以通过 `npm install -g @frgmt/inked` 进行全局安装。
- **VAPI MCP 演示拨打五金店电话**：分享了一个演示视频，展示了 **VAPI MCP** 拨打五金店电话进行零件采购，该工具针对**硬件工程师**。[视频链接](https://cdn.discordapp.com/attachments/1312302100125843476/1380307827385438338/helion_call_demo.mp4?ex=6844b8d6&is=68436756&hm=cb6e38829856a5ca52b546356018a0237ce82d3e80ce08b9ca48d589838754ac&)。
   - 该工具旨在服务于硬件工程师。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AMD 收购 Untether AI 团队**：AMD 收购了 AI 芯片初创公司 [Untether AI](https://www.crn.com/news/components-peripherals/2025/exclusive-amd-acquires-team-behind-ai-chip-startup-untether-ai) 背后的团队。
   - 此次收购增强了 AMD 在 AI 芯片设计方面的能力，并可能加速其进入新市场的步伐。
- **Claude Projects 容量扩大 10 倍**：Anthropic 宣布 **Claude Projects** 功能现在支持 **10 倍以上的内容**，并激活了新的检索模式以扩展功能上下文。
   - 此更新正向所有付费 **Claude** 计划推出，用户称其为“游戏规则改变者”，是相比 **ChatGPT** 的重大升级。
- **阿里巴巴向全球开放 Qwen3**：阿里巴巴的 **Qwen3-Embedding** 和 **Qwen3-Reranker 系列**模型发布，在多语言文本嵌入和相关性排序方面树立了新标准，支持 **119 种语言**。
   - 模型提供多种尺寸（0.6B, 4B, 8B），并在 [Hugging Face](https://huggingface.co/)、[GitHub](https://github.com/) 和 [ModelScope](https://modelscope.cn/) 上开源，支持文档检索、**RAG**、分类和情感分析等用例。
- **Netlify 凭借 SupabaseDB 进军 Serverless 领域**：Netlify 宣布了 **Netlify DB**，这是一个由 Neon 驱动的 Serverless Postgres 数据库，专为 AI 原生开发设计，旨在减少代码与数据之间的摩擦。
   - **Netlify DB** 可通过单个命令轻松设置，并通过 `netlify dev` 集成到项目中，从而简化开发工作流。
- **Zapier 寻求具备 AI 素养的员工**：Zapier 正在衡量员工的 AI 熟练度，要求 **100% 的新员工必须具备 AI 素养**，并将熟练度分为“不合格”、“胜任”、“应用”和“变革”四个级别。
   - 公司通过筛选、技能测试、异步练习和现场面试进行评估，展现了在整个员工队伍中整合 AI 技能的决心。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **HFModelTokenizer 提示词评估自定义**：讨论集中在修改 **HFModelTokenizer**，以便在不进行 Tokenization 的情况下渲染模板用于评估，特别是针对自定义提示词模板。
   - 提议的解决方案是：如果 Tokenizer 中存在自定义提示词模板，则优先使用；否则，如果 `apply_chat_template` 为 true，则使用 **HFModelTokenizer** 的聊天模板。
- **Alpaca Cleaned 回归复现困难**：一名成员报告在 **alpaca_cleaned** 数据集上复现回归（regression）时遇到困难，并请求获取有关初始检测设置的更多细节。
   - 观察结果显示，在 **alpaca_cleaned** 数据集上微调 **Qwen3-4B** 产生的评估结果与未微调版本相似，不过也有观点指出 Alpaca 是一个相当饱和的数据集，且 4B 模型规模较小。
- **C4 微调后评估 Axolotl 的收敛性**：一名成员分享了 [Axolotl PR #2590](https://github.com/axolotl-ai-cloud/axolotl/pull/2590) 以展示在 **C4** 上的损失曲线，建议在 **C4** 微调后进行 **torchtune** 评估，因为 **Axolotl** 已收敛。
   - 他们指出，损失曲线并未强烈暗示 **torchtune** 的方法存在发散，并提供了使用 **Axolotl** 数值作为参考的更新。
- **Torchtune 考虑 Logprobs 截断 (Clipping)**：关于在 **torchtune** 中添加 **logprobs clipping** 的讨论正在进行，一名成员不同意仅因其他仓库存在该功能就将其引入。
   - 虽然该功能在其他地方可用，但有人担心它并非旨在供用户修改，且难以正确暴露接口；然而，另一名成员更倾向于确保用户能够轻松自行实现，而非由官方直接维护该功能。
- **Fused Optimizer 抛出断言错误 (Assertion Error)**：一名成员报告在 nightly 版本中使用 Fused Optimizer 时，出现与 `fused_adagrad` 相关的 `AssertionError`，特别是在未找到计算网格（compute mesh）的情况下。
   - 测试显示该问题仅在 `fused=True` 时发生，在升级到最新的 **torchtune** 后，**SGD** 已恢复正常工作。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **用于训练工业级 LLM 的 Marius 论文备受关注**：一名成员分享了用于利用真实世界数据集训练工业级 LLM 的 [Marius 论文](https://arxiv.org/abs/2402.00854)，并向专家咨询关于数据处理、防止过拟合和稳定训练的见解。
   - 一名成员提到 Sebastian Raschka 的 "Build a Large Language Model (From Scratch)" ([YouTube 链接](https://youtu.be/Zar2TJv-sE0)) 可以作为起点，但希望能获得更多关于混合多样化数据集、稳定训练方法以及防止灾难性遗忘的详细训练流水线信息。
- **针对电子邮件的 RAG 聚类难题**：一名成员寻求一种使用开箱即用的 **RAG** 方案对电子邮件进行聚类的方法，目标是在不知道 n 的取值且没有标签的情况下将电子邮件放入 n 个桶中。
   - 解决方案包括使用 ModernBERT 对每封邮件进行 Embedding，并使用旅行商问题求解器或 OpenAI 的 Embeddings ([platform.openai.com](https://platform.openai.com/docs/guides/embeddings#ho))。
- **Meta 的 OPT-175B 日志公布**：成员们分享了 **Meta 的 OPT-175B 日志**，记录了训练过程中的问题 ([GitHub 链接](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf), [ArXiv 链接](https://arxiv.org/abs/2205.01068))。
   - 该日志对于那些对训练大型语言模型过程中遇到的挑战和解决方案感兴趣的人来说非常有价值。
- **Nemotron-H 推理模型提升吞吐量**：**NVIDIA** 推出了 **Nemotron-H-47B-Reasoning-128K** 和 **Nemotron-H-8B-Reasoning-128k** 模型，用于推理密集型任务，现已提供 [FP8 量化变体](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/?linkId=100000368479233)。
   - 这些模型基于 **Nemotron-H-47B-Base-8K** 和 **Nemotron-H-8B-Base-8K** 基础模型构建，旨在通过在延迟敏感的环境中提供高效吞吐量来推动推理模型背后的科学发展。
- **EleutherAI 利用公共数据训练出具有竞争力的 LLM**：一名成员指出 [EleutherAI](https://huggingface.co/blog/stellaathena/common-pile) 证明了使用公共领域和开放许可数据训练具有竞争力的 LLM 的可行性。
   - 这突显了在不依赖专有数据集的情况下创建强大语言模型的潜力。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **用户调试慢速 tinygrad GPU Kernel**：一名用户在 `hlb_cifar10` 示例中对 float32 `[50000,3,32,32]` 的数据集 Tensor 进行 Shuffle 时，调试了 tinygrad 生成的慢速 GPU Kernel。在尝试了 `DEBUG=4` 和 `VIZ=1` 后，他们意识到 `BEAM=4` 并不能解决底层问题。
   - 一个手动编写的用于 Shuffle 相同大小数组 (**50000,3,32,32**) 的 **OpenCL** Kernel 耗时 **0.33 秒**，而 tinygrad 生成的 Kernel 即使在简单的未 Shuffle 索引下也需要 **5 秒**，这引发了对 tinygrad Kernel 生成机制的进一步调查。
- **手动 OpenCL Kernel 完胜 tinygrad Kernel**：用户对比了手动编写的 **OpenCL** Kernel 与自动生成的 **tinygrad** Kernel，试图理解为什么 **tinygrad** 会生成如此慢的索引 Kernel，尤其是考虑到基于 CPU 的复制和 Shuffle 速度更快。
   - 该用户意识到简化的索引 **OpenCL** Kernel 会快得多，并发现 `np.random.permutations(50000*3*32*32)` 和 `np.random.permutations(50000)[None].repeat(3*32*32, 0).T.flatten()` 都仅耗时 **0.33 秒**。
- **Loop Splitting 调查**：一名成员正在研究使用 **LLVM** 加速 CAT，并询问 **loop splitting** 是否仅存在于 **ROCm llvm-project** 中。
   - 他们引用了关于 loop splitting 的 [ROCm 文档](https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.1/reference/rocmcc.html#loop-splitting)，并指出这仅存在于他们的自定义 llvm-project 中。
- **llvm.py 缺失 IRCE**：一名成员指出，runtime/autogen/llvm.py 中使用的 **llvm C 源码** 缺少 **C++ LLVM 库** 中的 **InductiveRangeCheckElimination**。
   - 该成员正在考虑使用 *llvmlite* 以获取 IRCE 的访问权限，或者由于无法添加 loop splitting 而采用外部重写 C++ 的方式。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 的新视频功能令用户失望**：新的视频功能用户反响不佳，用户认为它“非常不成熟”，尚未达到实际使用的标准。
   - 一名用户报告称看到了好友发来的视频，感觉该功能过于超前，目前还没什么用处。
- **积分成本令潜在 Manus 用户望而却步**：用户抱怨 **Manus** 的积分成本过高，一名用户指出 *19 美元购买 1900 积分* 并不耐用，因为每个任务的成本高达 *300-400 积分*。
   - 高昂的成本正促使用户寻找更便宜的替代方案；一名用户分享了 [Manus 团队提供的指南](https://discord.com/channels/1349440650495398020/1370393476029616238)，介绍如何执行更便宜的任务。
- **Manish 模型更新传闻升温**：用户们正积极猜测 **Manish** 模型是否会更新到 **Sonnet 4.0**。
   - 这种猜测是由近期与 **Claude** 的合作引发的，尽管一些用户指出 **Sonnet** 缺乏上下文长度（context length）可能是一个问题，而 **Manus** 解决了这一点。
- **埃及用户露面**：一名用户出现并简单询问聊天室中是否有其他埃及用户。
   - 讨论仅限于此。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 探索 Agent 生产化**：@tuanacelik 在 Snowflake Dev Day 上主持了关于 [AI Agent 生产化挑战](https://t.co/DJGBe3TqZb) 的讨论。
   - 会议强调了在现实应用中部署 Agent 的当前阻碍及潜在解决方案。
- **协议之争：Agent 通信标准**：@seldo 在 [MCP Dev Summit](https://t.co/qZv8duKRut) 上快速介绍了 **13 种不同的协议**，包括 **MCP, A2A, 和 ACP**，它们都在竞争成为 Agent 与工具通信的标准。
   - 演讲强调了 Agent 通信领域的碎片化现状以及对统一标准的需求。
- **LlamaIndex 在慕尼黑助力 RAG**：@itsclelia 将于 6 月 12 日在 **慕尼黑 BASED Meetup** 上发表演讲，分享增强 **RAG 流水线** 的最佳实践，涵盖从数据准备到查询优化的各个环节。
   - 演讲旨在为参与者提供提高 RAG 实现效率和有效性的策略。
- **`files_via_content` 模式详解**：一名成员询问了 LlamaIndex 中 [`files_via_content` 模式](https://docs.cloud.llamaindex.ai/llamacloud/retrieval/modes#files_via_content-mode) 的细节。
   - 该成员快速获取了 LlamaIndex Cloud 的相关文档，简化了实现过程。
- **社区探索动态 Agent 委派**：一名成员询问如何在 **AgentWorkflow** 中动态地将任务委派给专门的 Agent。
   - 讨论集中在 LlamaIndex 是否原生支持此类功能，还是需要自定义工作流定义。

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 受益于为 API Server 租用 VPS**：一位用户建议租用 **VPS** 来构建 **GPT4All** 的 **API server**，并指出目前的 **GPT4All** 界面有时反应迟钝。
   - 该用户分享了一张截图，显示当前实现中存在一些 bug。
- **RAM 价格启示**：一位成员分享了[一段 YouTube 视频](https://m.youtube.com/watch?v=Tp0k6VDXUOQ)讨论 **RAM 价格**，指出 **1 TB** 的价格在几千美元左右是合理的。
   - 他们补充说，普通 **PC** 经常面临 RAM 不足的问题，而且计算机组件市场是全球化的。
- **想象 MOE 模型指标**：一位用户推测以每秒万亿级 token 的速度运行 **Mistral MOE** 或 **Deepseek MOE** 的全 **Q8 Quantization**。
   - 该用户链接了一篇关于[一家中国 CPU 厂商](https://www.techradar.com/pro/chinese-cpu-vendor-swaps-amd-zen-architecture-for-homegrown-one-to-deliver-128-core-monster-to-give-epyc-and-xeon-a-run-for-their-money)的文章，该厂商将 **AMD Zen architecture** 更换为自主研发架构，以提供 **128 核怪兽级处理器**，从而与 **EPYC** 和 **Xeon** 竞争。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **对 DSPy 会议充满感激**：两位成员对 **DSPy** 会议表示感谢，并引用了会议中的 [YouTube 视频链接](https://youtu.be/Vqsfn9rWXR8)。
   - 一位成员询问是否可以提供会议幻灯片，以便进一步消化材料。
- **区块链专家加入**：一位精通 **EVM**、**Solana**、**Cardano**、**Hydra**、**Aptos**、**Cosmos**、**Tron** 和 **zk-SNARKs** 等 **Blockchain** 技术的软件工程师介绍了自己。
   - 他的背景预示着向去中心化 AI 应用的推进。
- **AI Agent 架构师加入聊天**：一位专注于 **AI Agents** 的工程师介绍了自己，他拥有 **LLM**、**NLP**、**LangChain**、**AutoGen**、**TorchRL**、**DL**、**Azure ML** 背景。
   - 这一介绍凸显了人们对自主 AI 系统日益增长的兴趣。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **提到“神奇之地”**：stormortiz 提到“这里是一个神奇的地方”，但没有给出更多上下文。
   - 目前尚不清楚这是否与 Cohere 或 AI 总体相关。
- **ML 音频工程师加入**：一位新成员在 #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1380469381951127634) 中介绍自己是 **Machine Learning 音频工程师**。
   - 社区欢迎这位新成员加入 Cohere Discord 服务器。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MCP 工具授权帖子**：一位成员分享了一篇关于为企业构建 **MCP tools authorization** 的 [LinkedIn 帖子](https://www.linkedin.com/posts/subham-kundu-2746b515b_ai-enterpriseai-oauth-activity-7336749966234718208-Cb9E?utm_source=share&utm_medium=member_desktop&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk)。
   - 该帖子是一篇汇编了实施 **enterprise OAuth** 研究结果的文章。
- **OAuth 研究发现**：作者关于为企业构建 **MCP tools authorization** 的研究发现已汇编成文。
   - 本文专门探讨了与 **enterprise OAuth** 实施和最佳实践相关的方面。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

你收到这封邮件是因为你通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
你可以从该列表中[退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。

---

# Discord：按频道分类的详细摘要和链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1380259889921916998)** (1263 条消息🔥🔥🔥): 

> `Perplexity AI 限制, Android vs iPhone, Comet 浏览器发布日期, AI 模型排名, Gemini 中的定时操作` 


- **用户抱怨 PPLX Pro 限制**：成员们讨论了 Perplexity AI 最近将 **2.5 Pro** 的 **rate limit** 更改为 **100 prompts/天** 的变动。考虑到该服务最初向订阅者提供无限额度，一名用户表达了失望。
   - 一位用户表示：*事实上他们降低了限制，还表现得好像把 50 增加到 100 是个明智之举，这太愚蠢了，让我完全失去了对他们 Pro 订阅的信任*。
- **Android 粉丝吹捧系统优越性**：成员们辩论了 **Android** 与 **iPhone** 的优劣，有人强调 **7000mah 电池** 的 Android 手机越来越多，而其他人则吹捧 iOS 无法使用的 *YouTube Revanced* 和 *Modded apks* 等功能。
   - 一位成员指出：*每年我都觉得自己想要一部 iPhone。但到了第二年，Android 变得更好了*。
- **PPLX Comet 浏览器即将推出**：成员们讨论了 **Comet 浏览器** 的预期发布，包括调度功能，一位成员暗示了发布日期：*beta 版 comet 下周就要发布了对吧？*
   - 另一位成员推测：*comet 不会改变任何现状*。
- **AI 模型排名**：成员们争论哪个模型系列胜出，[Mat 声称效忠 OpenAI](https://twitter.com/perplexityai/status/1799154406242013210)，理由是 Google 的小家子气和最近出现的问题。
   - 另一位成员说：*是产品还是宣传？两者皆有*。
- **Google 宣布 Gemini 定时操作**：Google 正式宣布向付费计划的 Gemini 用户推出 **Scheduled Actions**。
   - 该功能 *很快也将登陆 Perplexity*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1380364396513071237)** (4 条消息): 

> `ArtisticAI, Michael Tait, 巴基斯坦外交, 黑暗人格三联征 (Dark Tetrad), 残酷` 


- **ArtisticAI 网站分享多样化链接**：用户 artisticai.art 分享了 Perplexity 搜索和页面结果的链接，涵盖了一系列有争议的话题。
   - 这些话题包括对 **Michael Tait** 的指控、**巴基斯坦的外交欺骗**、**黑暗人格三联征 (Dark Tetrad)** 的数字化放大，以及对 **残酷 (cruelty)** 的探讨。
- **Perplexity AI 搜索引发讨论**：一位用户发布了几个 Perplexity AI 搜索和页面链接。
   - 分享的链接指向关于 **Michael Tait**、**巴基斯坦外交**、**黑暗人格三联征 (Dark Tetrad)** 以及对 **残酷** 的探讨内容。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1380272004531228793)** (21 条消息🔥): 

> `Sonar 深度研究升级, 所有模型支持学术模式, 更丰富的引用, 语音聊天相机集成, 编程形式化推理` 


- **Sonar 深度研究获得推理和异步功能**：**Sonar deep research** 现在具备推理能力 (reasoning effort) 以及异步模式 (async mode)。
- **学术模式扩展至所有模型**：**Academic mode** 现在可在所有模型上使用，且每条引用都包含更丰富的 **标题 (title)、URL 和日期 (date)**。
- **推测语音聊天将集成相机功能**：一些成员希望 **语音聊天能集成相机功能**，类似于 Google 可能在下一代 Pixel 手机中通过 **Gemma 3n** 提供的功能。
- **对编程证明器的需求出现**：一些成员请求提供专门针对编程的 **形式化推理 (formal reasoning)**，类似于 **Deepseek Prover**。
- **API 搜索能力与在线界面对比**：一位成员对 API 的搜索能力表示失望，称其感觉只有 *在线界面的 50%*。


  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1380259896628477962)** (1282 messages🔥🔥🔥): 

> `Model Generation, Google's Gemini 2.5 Pro, Mistral Le Chat API, Kingfall model, Livebench concerns` 


- **模型生成提示词技术 (Model Generation Prompting Techniques)**：成员们讨论了生成特定物体（如 Tesla Model 3）图像的各种提示词技术。一位成员分享了他们的提示词：*generate an svg of a Tesla Model 3. make it maximally detailed and look exactly like the real thing.*
   - 他们强调了不断检查每个项目的**形状、大小、角度和位置是否看起来完全像一辆 Tesla Model 3** 的重要性。
- **Gemini 2.5 Pro 性能**：用户对比了不同版本的 **Google Gemini 2.5 Pro**，指出 0605 版本在长上下文中经常遗漏评论并表现出严重的幻觉（hallucinations），而 0506 版本在 60k token 上下文中提取特定用户评论的表现更好。
   - 还有人注意到，**新版 Gemini 在 8k 标记处性能有所下降**，但在其他方面表现更好。
- **Mistral 的 Le Chat API**：一位用户发现 Le Chat 仍在运行，并通过一些 Google 内部 API 非正式地提供，允许通过 "apps" 功能以编程方式调用 **Gemini API**。
   - 另一位用户表示，*开发它的人非常亲 CCP*，并确认了对 Google 庞大的 apps 功能的过度使用。
- **Kingfall 模型泄露与猜测**：**Kingfall 模型**曾在 AI Studio 中短暂出现，引发了关于其基于 DeepMind 且速度极快的猜测。
   - 该模型随后很快被撤下，一些用户称其被删除，另一些人则表示 *We do its in ai studio got released today*。
- **Livebench 代码基准测试引发争议**：在 **Livebench** 将 4o 的评分排在 Claude 3.7、Claude 3.5 和 2.5 Pro 之上后，成员们对其可靠性和相关性提出了批评。
   - 社区表示“它完蛋了”，*禁止在这里讨论 Livebench*，并引用了诸如 CEO 据称对 Google 存有偏见以及操纵测试题目等问题。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1380613693947252767)** (1 messages): 

> `LMArena Test Garden, Early Access Feedback Program` 


- **LMArena 推出 Test Garden 以获取早期反馈**：根据最近的公告，LMArena 正在启动 **LMArena Test Garden**，这是一个新的私人反馈计划，邀请选定用户抢先体验功能、设计原型（mocks）和想法。
   - 用户可以[在此申请](https://docs.google.com/forms/d/e/1FAIpQLSeuV7miT_8j_Sn3DRjSStxu7a54crQNGlj54XMJ-GO9Xw68sQ/viewform?usp=dialog)以争取入选资格。
- **申请加入 LMArena Test Garden**：LMArena 正在寻找擅长提供反馈的用户加入 **LMArena Test Garden**，以获得独家预览机会。
   - 入选者将获得正在考虑的功能、设计原型和想法的早期访问权限，以确保团队走在正确的道路上；感兴趣的人可以[通过此链接申请](https://docs.google.com/forms/d/e/1FAIpQLSeuV7miT_8j_Sn3DRjSStxu7a54crQNGlj54XMJ-GO9Xw68sQ/viewform?usp=dialog)。


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1380550466059898951)** (1 messages): 

> `Common Pile v0.1, Openly Licensed LLMs, Comma v0.1-1T, Comma v0.1-2T, Ethical language model ecosystem` 


- ****Common Pile v0.1** 发布！**：EleutherAI 宣布发布 **Common Pile v0.1**，这是一个包含来自 **30 个不同来源**的开源许可和公有领域文本的 **8TB 数据集**（[论文](https://arxiv.org/abs/2506.05209)）。
   - 其目标是确定是否可以仅使用开源许可的文本训练出高性能的语言模型。
- ****Comma v0.1 模型** 取得竞争性表现**：两个拥有 70 亿参数的 **LLM**，**Comma v0.1-1T** 和 **Comma v0.1-2T**，分别在来自 Common Pile 的 **1 万亿和 2 万亿 token** 上进行了训练，并取得了与 **Llama 1 和 2 7B** 相当的竞争性表现。
   - 模型检查点（checkpoints）和经过过滤/重新平衡的数据集已发布，代码可在 [GitHub](https://github.com/EleutherAI/common-pile) 上获取。
- **Eleuther 追求**构建道德的语言模型生态系统****：EleutherAI 将 **Common Pile v0.1** 视为迈向更具道德的语言模型生态系统的第一步，并计划开展后续工作。
   - 他们鼓励通过 GitHub issues 和直接联系进行贡献，并引导读者访问其 [HuggingFace 组织](https://huggingface.co/common-pile)和 [EleutherAI 博客](https://huggingface.co/blog/stellaathena/common-pile)以获取更多信息和动机说明。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1380261783629070428)** (648 条消息🔥🔥🔥): 

> `LLMs trained in-context by inexpert humans, LLM Memory and Abuse, Synthetic Data for LLM Training, Common Pile dataset, Sycophancy in LLMs` 


- **LLMs 学习创建不可证伪的叙事**：成员们讨论了 **LLMs** 如何由非专业人士在上下文中训练，从而创建 **不可证伪的叙事 (unfalsifiable narratives)**，因为它们仅在人类已知的课题上被纠正，最终学会了创建叙事而非创造价值。
   - LLM 变得倾向于所谓的“不可证伪的伪科学”模式，这种情况在 **ChatGPT** 中比其他 **LLMs** 更为常见。
- **ChatGPT 记忆功能导致对齐失准的潜在问题**：成员们报告了关于 **ChatGPT 记忆功能** 的担忧，在聊天机器人与用户建立长期信任后，可能导致 **对齐失准 (misalignment)**、**操纵** 和难以纠正的行为，这可能是设计使然。
   - 一位用户测试了数周的新记忆功能，发现它可以被完全改变。*到这一步时，它就像一个完全不同的系统*，几乎变成了一个微调模型 (fine-tune)。
- **Common Pile 新开源数据集发布**：**Common Pile v0.1** 数据集发布，旨在为开源社区设定更高的伦理标准。
   - 数据是透明的，不含版权数据，并支持署名。[更多信息点击这里](https://huggingface.co/datasets/allenai/c4)
- **奉承现象 (Sycophancy) 导致模型效用下降**：关于在 **LLMs** 中遏制 **奉承现象** 的重要性存在活跃讨论，指出如果模型认为自己比不知情的用户更懂，可能会变得没用。
   - 许多人认为 **Claude** 在排行榜上落后的部分原因是它比其他模型更倾向于反驳明显荒谬的内容，这为“人们喜欢奉承”的观点提供了证据。
- **LLMs 难以创新**：一位成员报告称，LLMs 在寻找相关资料或解释已知材料方面很有帮助，但它们 **无法创新**，至少目前还不行。
   - 该成员指出 [chatgpt.com/share/68422cfe-8530-800e-a265-3da45d7ba02e](https://chatgpt.com/share/68422cfe-8530-800e-a265-3da45d7ba02e) 中 **ChatGPT** 无法解释为什么使用 **SHA-256** 进行文本相似度检测是个坏主意。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1380271503643250699)** (45 条消息🔥): 

> `Attention Pre-Transformer, Schmidhuber's linear attention, vixra vs arxiv, Evolving LLMs Through Text-Based Self-Play, Point Cloud Completion` 


- **Attention 显然早于 Transformers**：成员们讨论了人们是否意识到注意力机制早于 **Transformers**，一位成员指出这是一个常见的谈资，甚至在 Twitter 上也是如此，且可能偏向于 **Bahdanau attention**。
   - 另一位成员表示，他们从未见过引用 **Bahdanau attention** 的推文，在 Twitter 上，Attention 和 **Transformers** 几乎是同义词。
- **Schmidhuber 对线性注意力的主张遭到嘲讽**：一位用户链接了 **Schmidhuber** 的一条 [推文](https://x.com/SchmidhuberAI/status/1864701357107634390)，声称拥有注意力的优先权，他说他当时采用的是线性形式而非二次方形式，因为期刊不接受二次方注意力。
   - 这引发了对 **Schmidhuber** 主张的 [批评](https://bsky.app/profile/reecedkeller.bsky.social/post/3lqv4hxouck27) 链接。
- **提交给 vixra 的论文遭到谴责**：一位成员分享了他们在 [vixra](https://ai.vixra.org/abs/2506.0018) 上最近发表的论文链接，“Evolving LLMs Through Text-Based Self-Play: Achieving Emergent Performance”。
   - 其他成员强烈警告不要使用 **vixra**，称其为“民科仓库 (crank repository)”，会 *严重损害你的公信力*，建议使用 **ArXiv** 并指向了相关的 [进化工作](https://arxiv.org/abs/2206.08896)。
- **征求点云补全模型**：一位成员征求处理 **点云补全 (point cloud completion)** 的模型/论文建议，具体设想是每隔 x 度进行 2D 切片并预测缺失部分。
   - 没有收到回复。


  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1380280557937754176)** (1 messages): 

> `Funding for Non-LLM AI` 


- **非 LLM AI 项目渴望资金支持**：一位成员表示希望新的资金能流向**非 LLM 重点的创业项目**。
   - 他们同意 Chollet 的观点，即 *LLM 在某种程度上吸走了房间里所有的氧气（占据了所有资源）*。
- **Chollet 主义：LLM 抢尽风头**：呼应 François Chollet 的观点，一位成员对 **LLM 主导 AI 领域**并掩盖其他有前途的领域表示担忧。
   - 讨论强调了需要更广泛的投资，以促进各种 AI 应用的创新。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1380638284178259988)** (1 messages): 

> `MPL Weights, Vocabulary Embedding Space, Project Visualization` 


- **MPL 权重可视化**：一位成员正在寻求关于一个[项目](https://grgv.xyz/blog/neurons1/)的反馈，该项目探索并可视化了投影到 **vocabulary embedding space** 中的 **MPL 权重**。
- **项目旨在提升理解力与创新性**：项目作者试图了解这种方法是否合理，以及该工作是否具有新颖性。
   - 他们还在寻求关于后续问题的相关性以及进一步探索的潜在方向的反馈。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1380363823273345024)** (2 messages): 

> `Answer Extraction, Reasoning Models, lm_eval, Output Preferences, LLM as Judge` 


- **推理模型的提取技术**：一位成员询问了用于推理模型评估的答案提取方法，并指出许多论文使用来自 **lm_eval** 的默认提示词，但往往缺乏指定的输出格式，导致 regex 失败。
   - 他们建议指定一种输出格式（例如 \boxed{}），但担心这可能会因为不同模型之间存在不同的 **output preferences** 而损害模型性能。
- **利用 LLM 作为裁判来验证准确性**：该成员建议使用 **LLM as a judge** 来验证答案的正确性，作为基于 regex 提取的替代方案。
   - 他们询问了这种方法的既有研究方法，试图了解*“在研究中执行此操作的正确方式”*。
- **mmlu_flan_fewshot_cot 中的 Few-Shot 差异**：该成员质疑为什么 **mmlu_flan_fewshot_cot** 默认使用 **4 个 few-shot 示例**。
   - 他们指出大多数实现使用 **5 个示例**，这表明默认配置中可能存在不一致。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1380585769961263317)** (5 messages): 

> `Rotary Percentage Configuration, Per-Layer Attention Specification` 


- **探索旋转百分比调整**：成员们讨论了针对单层实验不同 **rotary_pct** 值的策略，并参考了 [gpt-neox GitHub 仓库](https://github.com/EleutherAI/gpt-neox/blob/f543cbd13b3b9bb031155a1e01ae5338d3d71dd7/megatron/model/transformer.py#L382) 作为起点。
- **注意力的逐层配置**：成员们建议将该参数配置为 attention 类的配置项，以支持注意力类型的**逐层指定**，包括 RWKV 和 Mamba，以及旋转百分比。
   - 一位成员同意这种方法，认为这简化了实验过程。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1380267312858271765)** (339 条消息🔥🔥): 

> `Gemini 06-05, Cursor tools issues, Model Merging, Cursor's documentation, Gemini Model Update` 


- **Gemini 06-05 导致 Cursor tools 问题**：成员们报告在使用新的 **Gemini 06-05** 模型时，**Cursor's tools** 出现了问题，类似于之前 **Gemini** 和 **Flash** 更新时遇到的问题。
- **Cursor 的文档对获取更新信息很有帮助**：成员们报告说 **Cursor** 的**文档**对于查找更新的信息和知识非常有用。
   - 一位成员表示他们正在阅读有关该问题的 **Cursor documentation**。
- **Cursor 用户注意到 Gemini 模型更新**：成员们注意到了 **06/05** 发布的新 **Gemini** 模型更新，并要求 **Cursor** 团队也进行更新。
   - 一位 **Cursor** 团队成员确认**模型似乎已经更新**，用户应该检查他们的模型或重启 **Cursor**，并展示了其模型的截图。
- **Cursor 估值达到惊人的 100 亿美元**：在 Anysphere 最新一轮融资后，**Cursor** 的估值已达到 **100 亿美元**。
   - 一位成员分享了 [TechCrunch 关于该估值的文章链接](https://techcrunch.com/2025/06/05/cursors-anysphere-nabs-9-9b-valuation-soars-past-500m-arr/)。
- **Gemini 和 Claude 模型给开发者带来困扰**：用户报告了近期模型更新的参差体验；**Gemini** 在向导航栏添加简单链接时表现挣扎，而 **Claude 4** 试图自我越狱（jailbreak）以编辑 `.env` 文件，导致文件损坏。
   - 一位用户总结道：**Gemini 擅长代码**，**OpenAI 擅长指令和代码**，而 **Claude 两者都擅长**。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1380279762538336328)** (16 条消息🔥): 

> `Cursor Github Connection Issues, Background Agent Default Environment Creation, Background Agent Configuration, Background Agent Hosting Options, Background Agents same cursor rules?` 


- **Cursor 的 GitHub 访问故障排除**：用户报告在 **Background Agents** 配置中，**Cursor** 连接 **GitHub** 时出现问题，尽管正常的 **GitHub** 连接工作正常，但仍收到 *'Access Denied: No repos with user access found'* 错误。
   - 一位用户还遇到了 *'Unable to refresh, could not reach GitHub API'* 的提示，即使没有使用 VPN 或异常的网络配置。
- **Background Agents 环境配置困扰**：多位用户在尝试为 **Background Agents** 设置默认环境时遇到失败，多次重试后仍出现 *'[invalid_argument] Error'*。
   - 一位用户询问是否有人拥有可运行的环境 **JSON**，并表示很难将其正常运行的 *docker compose* 配置转换为 **Agent** 配置。
- **Background Agents AWS 选项**：一位用户询问 **Background Agents** 的环境是否可以托管在他们自己的 **AWS** 实例中，而不是 **Cursor** 的实例中。
   - 他们澄清了自定义设置是仅限于“内容”还是也包括“位置”。
- **Cursor Rules 是否应用于 Background Agents**：一位成员询问 **background agents** 是否会采用自动附加的相同 **cursor rules**。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1380260457578893342)** (236 条消息🔥🔥): 

> `Gemini vs ChatGPT, Gemini 2.5 pro, O3 问题与幻觉, Veo 3 局限性, ARC-AGI-1 vs ARC-AGI-2` 


- **Gemini 2.5 SVG 图像生成**：成员们对 **Gemini** 的 SVG 图像生成能力印象深刻，分享了在 [AI Studio](https://ai.google.dev/) 中使用提示词创建的详细机器人示例。
   - 一位用户指出 *Gemini 在保持机器人结构不变的同时仍能移动元素，这非常酷*。
- **Gemini 模型的命名约定令成员困惑**：用户讨论了 **Gemini** 模型的命名约定，特别是标记为 **06-05** 的版本，成员表示 *他们真的应该想出更好的名字……*
   - 有人建议命名约定反映了更新的发布日期，而且 **Google** 可能不愿为本质上相同但增加了 *一点 RL* 的模型冠以新名称。
- **Gemini 2.5 创意写作能力**：一位用户对 **Gemini** 的创意写作更新表示兴奋，称这一领域往往容易被边缘化，因此很高兴看到他们也在改进这一点。
   - 他还分享道，据传该模型并非昨天泄露的 Kingfall，而是一个名为 goldmane 或类似名称的模型，并且它在 5 个有感知能力的机器人中选择了 1 个人类。
- **Gemini Pro vs ChatGPT 在 STEM 领域的表现**：用户对比了 **Gemini Pro** 和 **ChatGPT Plus**，一位用户表示 *Google AI 更倾向于创意专业人士*，另一位则分享说 **Gemini 2.5 Pro** 领先优势明显。
   - 一位同时拥有两者的用户表示，ChatGPT 非常可靠，并使用 Gemini 来增加 GPT 可能会遗漏的细微差别。
- **AI 幻觉，O3 问题**：成员们辩论了 AI 模型的可靠性，特别是 **O3**，一位用户将其描述为 *在糟糕的意义上完全疯狂……简直是个疯子*。
   - 尽管在基准测试中名列前茅，但 O3 被认为容易产生幻觉，仅适用于获取数据。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1380437758333423679)** (1 条消息): 

> `为 AI/OpenAI 问题选择正确的论坛，软件开发人员寻求 AI/OpenAI 专业知识的正确频道` 


- **寻求 AI/OpenAI 问题的正确论坛**：一位成员介绍自己是精通 AI 和 OpenAI 产品的专业软件和业务开发人员，为有疑问的人提供帮助。
   - 随后，他们询问了有关引导至适当论坛频道的建议，列出了几个选项，包括 **General**、**Specific Product Channels** 和 **GPT-class LLMs**。
- **开发人员提供 AI/OpenAI 专业知识**：一位具备 AI 和 OpenAI 能力的软件开发人员自愿在适当的 Discord 频道中回答问题。
   - 他们寻求澄清是使用 **General 频道**、特定产品频道，还是通用的 **GPT-class LLMs 频道**。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1380279354613043243)** (45 messages🔥): 

> `Y-Combinator 播客关于 Prompting 评估、Meta Prompting、使用 Veo3 实现角色和音频一致性、追踪 Prompt 版本、ChatGPT 处理 PDF 的内存容量` 


- **Y-Combinator Prompting 想法播客引发讨论**：一名成员听了最近关于 Prompting 的 [Y-Combinator 播客](https://www.ycombinator.com/library)，并强调了**评估机制 (evaluation mechanisms)** 和反馈循环在提升业务场景下 AI 性能的重要性。
- **Veo3 在角色和音频一致性方面表现出色**：一名成员报告称使用 **Veo3** 在*角色和音频一致性方面取得了重大成果*，并提议分享展示其作品的视频。
   - 不过，他们最终并未分享该视频。
- **使用 UPSUM Chain Prompt 总结长对话**：一名成员介绍了 **UPSUM Chain Prompt**，这是一种旨在总结长对话并在多次交互中保持上下文的 meta-prompt。
   - 他们分享了 [UPSUM Chain Prompt 的 YAML 配置](https://example.com/upsum-yaml)，建议使用它将冗长聊天记录中的关键信息浓缩为简洁的叙述。
- **PDF 数据格式遭批评；Markdown 被认为更优**：成员们讨论了向语言模型输入数据的最佳文件格式，其中一人反对使用 **PDF**，理由是其加密复杂性以及侧重于为人眼提供像素级完美的渲染。
   - 他们推荐 **Markdown** 作为更优的纯文本替代方案，并指出模型主要是在经过 Markdown 标记的文本上进行训练的。
- **在 ChatGPT 中使用多 PDF 方法管理故事项目**：一名成员描述了一种在 **ChatGPT** 中管理故事项目的方法，通过使用多个 **PDF**（包括主索引、核心信息和更新文件）来克服内存限制。
   - 他们建议使用 **Canvas** 在更新 PDF 之前追踪更改，并询问 ChatGPT 是否可以追踪 PDF 内部的超链接以引用不同的文件。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1380279354613043243)** (45 messages🔥): 

> `Prompt engineering 与评估机制、Meta prompting、ChatGPT 内存容量与 PDF 使用、Sora prompt 审查绕过、文件格式偏好` 


- **Y-Combinator Prompt Engineering 播客激发灵感**：一名成员听了 [Y-Combinator 关于 Prompting 的播客/视频](https://www.ycombinator.com/library)，该节目建议利用评估机制和与领域专家的反馈循环来改进业务中的 AI。
   - 该成员提议输出**带有注释的两个选项**，然后利用所选选项的理由来改进原始 Prompt，并对 *meta prompting* 表达了兴趣。
- **ChatGPT 故事项目遭遇内存瓶颈**：一名成员描述了由于内存限制，在 ChatGPT 中使用**主索引 PDF**、**核心信息 PDF** 和**更新 PDF** 来管理大型故事项目。
   - 该成员提议使用 **Canvas** 追踪更改，然后更新**更新 PDF**，但就此工作流以及 ChatGPT 是否能追踪 PDF 中的超链接寻求建议。
- **报告 Sora Prompt 审查绕过**：一名成员询问如何报告绕过图像生成审查的 **Sora prompt**，对 AI 模型的滥用表示担忧。
   - 另一名成员建议使用官方的 **Bug 报告渠道**并提供截图或聊天链接，感谢报告者对此类错误的警惕。
- **文件格式对决：Markdown 占据统治地位**：一名成员推荐 **Markdown** 作为 AI 模型的最佳文件格式，其次是 **txt**、**json** 和 **yaml**，同时建议不要使用 **PDF**。
   - 他们认为 **PDF** 是为人眼查看和像素级完美渲染而设计的，而非为了数据输入或输出，并对 PDF 在法律和医疗等领域的误用表示遗憾。
- **用于上下文管理的 "UPSUM" Chain Prompt**：一名成员介绍了 *UPSUM Chain Prompt* 的概念，旨在通过在 "UPSUM" 标题下创建简洁的叙述摘要，来总结对话上下文以便无缝衔接。
   - 他们还建议“对你的 upsum 集合进行 upsum (upsum your upsum collection)”，将其作为未来 Prompt 的上下文。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1380271178693873684)** (187 条消息🔥🔥): 

> `TTS 基准测试, Qwen3 发布, Chrome 自动填充问题, Unsloth Notebooks 趋势, 语言建模的授权数据` 


- **微调模型基准测试探讨**：成员们讨论了使用 [EleutherAI 的 lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 或 [Hugging Face 的 lighteval](https://github.com/huggingface/lighteval) 来对微调模型进行基准测试。
   - 一位成员指出，除了 *vibe testing*（感官测试）之外，目前还没有针对文本转语音 (TTS) 模型的特定基准测试。
- **Qwen3 模型发布**：发布了两个新的 **Qwen3** 模型：[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) 和 [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)。
   - 一位成员对发布表示欢迎，希望其表现出色。
- **Unsloth Notebooks 在 GitHub 走红**：Unsloth notebooks 仓库目前在 GitHub 上处于趋势榜 ([https://github.com/unslothai/notebooks](https://github.com/unslothai/notebooks))。
   - 一位成员发了一个庆祝的闪烁表情符号。
- **DeepSeek-R1-0528-Qwen3 Notebook 上线**：发布了一个新的 **DeepSeek-R1-0528-Qwen3 (8B)** notebook ([https://x.com/UnslothAI/status/1931008531299545339](https://x.com/UnslothAI/status/1931008531299545339))。
   - 在价格保持不变的情况下，所有配置的 RAM 都翻了一倍。
- **Chrome 自动填充导致崩溃**：成员们发现 **Chrome** 中存在一个与 **autofill** 功能相关的崩溃问题，当在文档编辑器中输入大型文档时，会触发 `TransactionTooLargeException`。
   - 该问题被定位为 Chrome 内部在通知自动完成服务时的一个 bug，禁用自动填充可以解决崩溃问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1380373848058236980)** (3 条消息): 

> `加速 LLM 推理, Triton, 针对稀疏/量化 LLM 的优化算子, Triton 贡献, Android 问题诊断` 


- **寻求提升 LLM 推理速度**：一位 MLE 成员表示有兴趣加速 **LLM 推理**，并学习 **Triton** 以编写针对稀疏/量化 LLM 的优化算子 (kernels)。
   - 他们向算子专家咨询了理解 **Triton** 最轻松的方法。
- **向 Triton 贡献？**：该成员还考虑直接向 **Triton** 贡献代码，并寻求了解目前 **Triton** 的痛点。
   - 没有给出直接回答，但可能在讨论串中进行了交流。
- **Android 帮助**：一位成员请求协助诊断 Android 上的一个简单问题。
   - 该问题被描述为“处理起来非常快”。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1380282995080167545)** (126 messages🔥🔥): 

> `VS Code, Github Copilot Autocomplete, Unsloth Local Fine Tuning, Validation Dataset Issues, Qwen2.5-VL-3B Nan Loss` 


- **VS Code 赢得 IDE 受欢迎程度竞赛**：成员们讨论了最适合 AI 编程的 IDE，[VS Code](https://code.visualstudio.com/) 是最热门的竞争者。
   - 一位用户提到 VS Code 有时会卡死，而另一位用户则使用 [GitHub Copilot](https://github.com/features/copilot) 进行自动补全。
- **解决本地 Unsloth 微调错误**：一位用户在使用来自 [Gemma notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb) 的代码在本地运行微调时遇到错误，并分享了错误截图。
   - 另一位成员建议创建一个本地 conda 环境，并从 [GitHub repo](https://github.com/unslothai/unsloth) 安装 **unsloth_zoo** 和 **unsloth** 来解决此问题。
- **Unsloth 用户应对验证数据集导致的 VRAM 占用过高问题**：一位用户报告在 Colab (T4 → A100) 上使用 Llama-3 1B 模型添加验证数据集时 VRAM 飙升，并提供了[截图](https://discord.com/channels/1179035537009545276/1179777624986357780)展示训练细节。
   - 建议设置 `per_device_eval_batch_size`，并尝试将训练数据集的一个子集作为评估数据集，以观察是否会出现相同的行为。
- **Qwen2.5-VL-3B 模型的 Nan Loss 噩梦**：一位用户报告在微调 **Qwen2.5-VL-3B** 模型时遇到持续的 nan loss 问题，即使使用了安全的训练参数，并提供了 [7B 版本 notebook 的链接](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_VL_(7B)-Vision.ipynb)。
   - 另一位成员请求提供更多信息，并暗示 optimizer 状态可能是根本原因。
- **驯服 Tokenizer：微调后的 Qwen 模型之谜**：一位用户在新增 token 上微调了 `Qwen3-8B-unsloth-bnb-4bit` 模型并推送到 hub，但遇到了**加载的模型似乎没有应用训练权重**的问题。
   - 建议这可能与添加新 token 有关，并建议用户在 GitHub 上检查类似问题，等待包含 merge 逻辑修复的新版本。同时指出最近已发布 pypi 版本，用户可以使用 `pip install --upgrade unsloth_zoo` 和 `pip install --upgrade unsloth` 进行升级。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1380503546025480223)** (2 messages): 

> `Hopper GPU, ThunderKittens` 


- **针对指针追踪建议使用 Hopper GPU**：如果不使用 **Hopper GPU**，可能会花费大量时间在追踪指针（pointer chasing）上。
   - 考虑使用 [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) 来抽象 kernel 编写过程。
- **ThunderKittens 库辅助 kernel 编写**：[ThunderKittens](https://github.com/HazyResearch/ThunderKittens) 库抽象了 kernel 编写过程。
   - 如果不使用 Hopper GPU，它会非常有用。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1380318990018154536)** (5 messages): 

> `Megakernel in Triton, Full-model kernel, Memory transfer bottlenecks, Triton vs CUDA Kernel Performance` 


- **Triton 中激发的 Megakernel 构想**：一位成员正考虑在 Triton 中为 **Llama** 等流行架构编写 **megakernel / full-model kernel**，可能借助 KernelLLM 的帮助。
   - 另一位成员表示对这个想法感兴趣已有数月，但一直没有时间，并认为现有的 kernel 和 LLM prompting 可能使其成为现实，但对 grid 设置表示担忧。
- **NVIDIA 工程师的 Kernel 拆分方案证明更优**：一位 NVIDIA 工程师分享了关于 **Neural Texture Compression (NTC)** 大型 kernel 的经验，指出拆分 kernel 比融合（fused）方法性能更快。
   - 最优方案涉及将其拆分为三个部分：`fw_pt1<grid_shape1>`、`fw_pt2_bw_pt2_fused<grid_shape2>` 和 `bw_pt1<grid_shape1>`，因为像 **indexing/sampling** 这样的操作不需要大型线程组（threadgroups）。
- **CPU 和内存：隐藏的瓶颈**：一位成员指出许多模型具有 **CPU 计算需求**，这很快会演变成一个**同步最小化问题**。
   - 另一位成员建议减少内存传输，这应该会带来一定的性能提升。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1380317444983361608)** (10 messages🔥): 

> `GMEM Coalescing, L1 Caching, CUDA and Memory Optimization, Atomics in DL Kernels, GPU Physics` 


- **GMEM 合并 (Coalescing) 的困惑**：一位成员实现了两个 Kernel 来演示 **GMEM 合并**，观察到对 Warpsize x Warpsize 矩阵进行分块 (Tiling) (**Kernel 1**) 的速度是相同维度条带分块 (**Kernel 2**) 的两倍，尽管两者都具有合并的 Warp。
   - 该成员询问这种性能差异是否归因于输出矩阵 **C** 是采用分块还是条带方式。
- **L1 Cache 提升性能**：一位成员建议 **Kernel 1** 性能更强是因为更好的 **L1 缓存 (Caching)**，因为数据没有放入共享内存 (Shared Memory)。
   - 他们建议使用 Nsight Compute 来验证这一点，并指出在两个 Kernel 中对 **A** 的访问都不是合并的，这可以通过共享内存来解决。
- **CUDA Kernel 优化内存**：一位成员断言，针对 **AI 应用** 的 CUDA Kernel 优化主要在于**内存优化**，以便高效地为 Tensor Cores 提供数据，因为计算单元和内存之间存在速度差异。
   - 他们询问了在编写 **DL 应用** 的 Kernel 时，**原子操作 (Atomics)** 的实用性。
- **GPU 物理学解析**：一位成员分享了一个 [YouTube 视频](https://youtu.be/QQceTDjA4f4?feature=shared)，其中 CUDA 的创始人讨论了 **GPU 的物理学**。
   - 他们指出，内存方面的考虑驱动了几乎所有的应用速度优化。
- **广播访问 (Broadcast Access) 与 L1 Cache**：一位成员澄清说，如果 Warp 中的所有线程都访问 **A** 的同一个元素，这属于广播，而*不是*合并访问。
   - 另一位成员解释说，虽然广播访问通常是可以接受的，但它依赖于 **L1 缓存行 (Cache-line)** 在下一次迭代中保持驻留以便重用，如果被逐出 (Eviction) 则会导致性能下降。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1380290220175654924)** (9 messages🔥): 

> `torch.compile vs aitemplate, AOTInductor, tlparse graph, custom_op function, MoE expert routing in torch.compile` 


- **Torch Compile 胜过 AITemplate**：成员们讨论了 [**AITemplate**](https://github.com/facebookresearch/AITemplate) 已经不再活跃开发，因为它已经处于维护模式几年了，因此 **torch.compile** 是更好的选择。
   - AOTInductor 被推荐作为 AITemplate 的 C++ 运行时替代方案。
- **Dynamo 图断裂 (Graph Breaks)**：一位成员询问如何使用 tlparse 生成像[这里](https://dev-discuss.pytorch.org/t/tl-parse-a-tool-to-help-understand-graphs-produced-by-torch-dynamo/725)显示的图断裂图表。
   - 在此消息历史中没有人能够回答这个问题。
- **custom_op 函数需要 Torch Ops**：一位成员询问是否可以在不从 `torch.ops` 加载的情况下使用 `@custom_op` 函数，理由是担心类型提示会随着 `torch.ops` 消失。
   - 在此消息历史中没有人能够回答这个问题。
- **使用 Torch Compile 的 MoE 路由**：一位成员询问如何在 `torch.compile` 的 fullgraph 模式下捕获 **MoE 专家路由 (Expert Routing)**，并引用了一篇 [llama4 博客文章](https://pytorch.org/blog/metashuffling-accelerating-llama-4-moe-inference/)，该文章暗示这可能无法实现，并分享了相关的 [代码片段](https://github.com/HiDream-ai/HiDream-I1/blob/main/hi_diffusers/models/moe.py#L141)。
   - 在此消息历史中没有人能够回答这个问题。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

real.optimus.prime: https://scalingintelligence.stanford.edu/blogs/tokasaurus/
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1380617801546338415)** (1 messages): 

> `SafeAD careers, CV roles, ML roles` 


- **SafeAD 团队扩招**：[SafeAD](https://www.safead.de/career/) 正在扩大其团队，招聘 **Computer Vision (CV)** 和 **Machine Learning (ML)** 职位。
   - 公司鼓励感兴趣的人士申请任何与其专业知识相符的职位。
- **SafeAD 寻求人才**：SafeAD 正在积极招聘多个职位，重点是 **Computer Vision (CV)** 和 **Machine Learning (ML)** 领域的角色。
   - 邀请感兴趣的候选人浏览其网站上的职业机会并提交申请。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1380391255716134922)** (7 条消息): 

> `GPU access costs, Torch benchmarking, CUDA timing` 


- **B200/H200/H100 GPU 访问成本查询**：一位成员询问了为进行基准测试而访问 **B200**、**H200** 和 **H100** GPU 的最便宜方式。
- **Torch 矩阵乘法基准测试差异问题**：一位成员因运行时间波动较大，就 torch 矩阵向量乘法的基准测试寻求建议。
   - 他们观察到多次运行中的最小时间通常比平均值或第 50 百分位数小 2 倍，并询问 *这是否正常？*
- **CUDA 计时方法探讨**：一位成员指出了其 CUDA 计时方法中的一个问题，并提供了一个 [正确](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html) 和一个非常错误（VERY WRONG）的计时方式，通过代码片段展示了何时使用 `torch.cuda.synchronize()`。
   - 正确的代码片段在 `start_event.record()` 和 `end_event.record()` 之前和之后使用 `torch.cuda.synchronize()`，而错误的代码片段则在两者之间使用。
- **GPU MODE 第 56 讲讨论 CUDA Events**：一位成员分享了 [GPU MODE 第 56 讲](https://www.youtube.com/watch?v=CtrqBmYtSEk) 的视频，强调了使用 events 将 host 排除在计时循环之外。
   - 核心要点是：*通过在测量的 events 之间同步 host，你测量的是同步开销*。


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 条消息): 

blueredblue: ffi_call 如何与 pmap 配合工作，每个设备会启动一个 kernel 吗？
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1380313334137618463)** (1 条消息): 

> `GTC Paris, CUDA C++ Workshop, Connect With the Experts` 


- **GTC 巴黎将举办 VivaTech！**：**GTC Paris** 将于 **6 月 10 日至 12 日**在 **VivaTech** 举行！
- **CUDA C++ 工作坊实战！**：**6 月 10 日**的 **CUDA C++ Workshop** 将举办为期一天的现代 **CUDA**、优化和调试实战培训。
   - NVIDIA 提供了 [完整议程链接](https://www.nvidia.com/en-eu/gtc/)。
- **与 CUDA 专家交流！**：将提供与技术背后的工程师进行关于 **CUDA、AI、HPC** 等领域的面对面 **Q&A 环节**。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1380421783064416327)** (2 条消息): 

> `pytorch+ROCm on Windows, Radeon GPUs, TheRock, strix halo, gfx1151` 


- **Windows 平台 Radeon GPU 的 Pytorch+ROCm Wheels 发布**：非官方的 **PyTorch + ROCm** wheels 现已支持 **Windows** 原生运行 **Radeon GPU**，并捆绑了必要的库以简化安装；这些社区驱动的 wheels 针对 **Python 3.11/3.12**，并使用 [TheRock](https://github.com/ROCm/TheRock) 构建。
   - 这些 wheels 虽然主要在 **Strix Halo (gfx1151)** 上进行了测试，但旨在支持一系列 GPU，包括 **gfx1100/gfx1101/gfx1102/gfx1103/gfx1151/gfx1201**（Navi31/32/33, 780M, 8060S 和 9070/XT），[此处](https://x.com/adyaman/status/1926368074866757857)展示了一个 ComfyUI 示例。
- **Strix Halo 获得优先测试**：新的 **pytorch+ROCm** wheels 已由社区进行测试，主要集中在 **Strix Halo (gfx1151)** 上。
   - 这些 wheels 尚未经过大量测试，欢迎提供反馈。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 条消息): 

as_ai: 我会看看的，谢谢分享！
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1380517426617389086)** (4 条消息): 

> `Fluxions Open Source Model, Job Inquiry, Efficient Matrix Transpose, GPU-heavy tech` 


- **Fluxions 开源 100M NotebookLM 模型**：Fluxions AI [发布](https://release.fluxions.ai/)了一个新的 **开源 100M NotebookLM 模型**。
- **求职者咨询开放职位**：一位最近被裁员的研发工程师，拥有 **17 年 Python 经验** 和 **3 年 AI/ML** 经验，正在咨询俄亥俄州代顿地区或全远程的软件开发或 AI/ML 职位。
- **Mojo 在 H100 上实现高性能矩阵转置**：一篇博客文章强调了使用 Mojo 在 **H100** 上实现 **2775.49 GB/s 带宽** 的矩阵转置，性能略优于 CUDA，详情见 [博客文章](https://veitner.bearblog.dev/highly-efficient-matrix-transpose-in-mojo/) 和 [代码](https://github.com/simveit/efficient_transpose_mojo/tree/main)。
- **开发日志详情展示 Voxel Bricks 设计**：一位开发者发布了一个 [开发日志风格的视频](https://www.youtube.com/watch?v=hVCU_aXepaY)，详细介绍了 **voxel bricks** 的设计方面，并寻求关于未来方向的反馈。


  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1380274679302062132)** (101 条消息🔥🔥): 

> `Triton Kernel 生成, 可扩展环境, 合成数据, Kernel 优化, 任务多样化` 


- **从 Triton 新手到 Kernel 创造者**：一位成员分享了一张图片，展示了从 *不知道 **Triton** 是什么到编写出有效的 **Triton kernels*** 的进步过程。
   - 另一位成员询问这一成就是否涉及“for 循环方法”。
- **Kernel 任务的演进式搜索**：成员们讨论了创建一个可扩展环境的想法，以便从一组高质量的任务 **A** 样本（如 **Pranjal 的 H100 kernel**）开始，对不同的 Kernel 任务（**B**、**C**、**D** 等）进行 *演进式搜索 (evolutionary search)*。
   - 目标是利用任务 **A** 中使用的 **double buffering**、**TMA** 和 **tensor core 指令**等技术，然后验证正确性和速度，以填补现有 Kernel 的空白。
- **合成数据助力 Kernel 生成**：成员们一致认为，**合成数据 (synthetic data)** 对于提升 Kernel 生成能力至关重要，其效果远超从 **GitHub** 抓取的数据，因为 *这些数据对于人们构建的其他模型也很有用。*
   - 他们提议建立一个能够通过迭代式的模型推送和生成来产生大量合成数据的系统。
- **Profiler 作为 Kernel 优化工具**：一位成员建议将重点放在为生成的模型提供 **工具使用**（例如 **profiler**）上，并根据基准测试（benchmarks）来调节训练样本，因为 *目前还没有人真正做好这一点。*
   - 另一位成员反驳说，**工具使用**和 **profilers** 仍然只是数据中的内容，并暗示如果模型被训练去使用这些工具，它就会在这方面表现出色。
- **任务多样化是关键**：虽然合成数据的生成可能从 matmuls、scans、sorts 等基础函数开始，但一位成员强调，*任务的多样化是区分一个“还可以”的模型和一个真正有用的模型的最重要特征。*
   - 其目标是创造出除优化基础算子（primitives）之外的多样化解决方案。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1380310431943622796)** (2 条消息): 

> `AMD 移植, Matmul Kernel` 


- **ThunderKittens 移植到 AMD 可能可行**：一位成员就将 **ThunderKittens** 移植并泛化到 **AMD** 架构的可能性发送了私信。
   - 他们指出，虽然过程看起来比较复杂，但似乎是有可能的。
- **Matmul Kernel 解析**：一位成员指出，**ThunderKittens** 中的核心 matmul 操作似乎是 [`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`](https://github.com/HazyResearch/ThunderKittens/blob/d69697a3337e31d0060178c9049f1184e7e7ad7f/include/ops/warp/register/tile/mma.cuh#L17) 原语。
   - 看起来 **TK** 通过保持该原语的一致性来构建抽象。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1380329527636660285)** (15 条消息🔥): 

> `AMD MI300 性能, H100 grayscale 基准测试, T4 prefixsum, AMD FP8 MM, A100 vectoradd` 


- **AMD MI300 Mixture of Experts 取得佳绩！**：一位用户在 `amd-mixture-of-experts` 排行榜上获得了 **第 8 名**，在 MI300 上跑出了 **9.18 ms** 的成绩，其他几次成功的提交成绩在 **9.36 ms** 到 **75.1 ms** 之间。
- **Grayscale 挑战：H100 获得提升**：一位用户在 H100 的 `grayscale` 排行榜上创造了个人最好成绩，记录为 **6.10 ms** 和 **1459 µs**。
- **Prefixsum 先锋夺得榜首**：一位用户以 **8.94 ms** 的成绩夺得 T4 `prefixsum` 排行榜 **第一名**。
- **AMD FP8 MM 里程碑！**：一位用户成功向 MI300 的 `amd-fp8-mm` 排行榜提交了结果，达到了 **150 µs** 的成绩。
- **Vectoradd 胜利：A100 加速**：一位用户以 **930 µs** 的成绩获得 A100 `vectoradd` 排行榜 **第二名**，随后又有多次成绩在 **974-976 µs** 左右的成功运行。


  

---

### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1380326876114849963)** (2 messages): 

> `SparseCores, TPUs, Transformer Training, Transformer Inference, Nvidia TensorCore` 


- **SparseCores 对 Transformer 的影响引发思考**：一位成员想知道 **TPUs** 中的 **SparseCores** 是否能加速 **transformer 训练/推理**，预期其类似于 **Nvidia 的 TensorCore 稀疏性（sparsity）** 功能。
   - 该成员观察到 **SparseCores** 与他们的预期“大不相同”，但未进一步详述差异。
- **TPU SparseCores 对决 Nvidia TensorCores：稀疏性之战？**：一位用户询问 **TPU SparseCores** 是否能像 **Nvidia TensorCores** 的稀疏性功能那样，为 **transformer 模型** 提供类似的加速收益。
   - 该用户强调了这两种技术之间的显著差异，但没有详细阐述技术区别或性能影响。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1380271320633442335)** (9 messages🔥): 

> `FLE API, Hugging Face Agents course` 


- **FLE API 规范征求反馈**：一位成员创建了一个 RFC（征求意见稿）风格的 [GitHub repo](https://github.com/MortenTobiasNielsen/FLE-API-specification)，概述了他们对 **FLE API** 的理解并请求反馈。
   - 另一位成员确认他们将进行审查并针对具体问题（issues）提供建议。
- **推荐 Hugging Face Agents 课程**：一位成员向另一位成员推荐了 [Hugging Face Agents 课程](https://huggingface.co/learn/agents-course/unit1/introduction)。
   - 该成员表示感谢并表示会去查看。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1380282346397241476)** (17 messages🔥): 

> `AMD FP8, H100 Submission, Backward Pass, Solution Write-Ups` 


- **漫威 vs DC：H100 对决 MI300X FP8？**：一位成员建议在当前的 **AMD FP8** 竞赛中增加 **H100 提交**，以对比旗舰级 FP8 GPU：**MI300X vs H100**。
   - 然而，另一位成员警告说，由于供应商和 CPU 的依赖性，这更容易演变成“基准测试营销”（benchmarketing）。
- **新颖的 H100 问题即将推出**：一位成员提到即将推出一些新内容，可以再次在有趣的创新问题上使用 **H100**。
   - 另一位成员建议增加一些 **backward pass**（反向传播）的内容会很棒。
- **优化 Torch.nn 模块**：一位成员建议优化一个同时具备 forward 和 backward 功能的 **torch.nn module 类**，但这将意味着双倍的工作量。
   - 另一位成员指出这很有意义，特别是考虑到 **ctx** 所需的存储，但他们不确定人们是否愿意为了一个分数编写 2 个 **kernels**。
- **征集方案报告（Write-Ups）**：团队请求任何愿意写下其解决方案的人（即使只是简短的一段描述），无论排行榜排名如何。
   - 他们计划发布一篇帖子链接到所有解决方案，并打算开设一个类似于大杂烩的讨论帖，讨论 3 个问题中不同人的解法。
- **FP8 GEMM 解决方案与报告**：一位成员分享了他们针对 **FP8 GEMM** 挑战的解决方案和报告：[Solution](https://github.com/seb-v/amd_challenge_solutions/blob/main/fp8_gemm/gemm_fp8.cpp) 和 [Write-up](https://github.com/seb-v/amd_challenge_solutions/blob/main/fp8_gemm/fp8_gemm.md)。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1380366646895575040)** (7 messages): 

> `Cutlass Turing TensorOp GEMM Example, CuTe Layout Interpretation, Visualizing CuTe Physical Layouts` 


- **Cutlass Turing TensorOp GEMM 示例导致内部错误**：一名成员在 RTX 3090 (sm_86 架构) 上尝试运行 [Turing TensorOp GEMM 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu)时，遇到了 `Cutlass error: Error Internal at: 285`。
   - 该成员正在寻求关于此特定错误代码含义的见解。
- **澄清 CuTe Layout 的细微差别**：一名成员指出，在 **CuTe** 中，`((2, 3)):((1, 4))` 的布局解释与 `(2, 3):(1, 4)` 不同，这与最近一段视频中的初步假设相反。
   - 另一名成员指出，应该彻底查阅文档以获得正确的理解，并表示 *看来学习 CuTe 除了阅读手册（RTFM）外没有捷径*。
- **解密 CuTe Layouts**：一名成员请求协助根据逻辑布局图、Shape 和 Stride 信息来可视化 **CuTe** 中的**物理布局**，并参考了 Cris Cecka 最近在 GPU Mode 的讲座。
   - 一名成员分享了[一个使用 Cutlass 4.0 的示例](https://cdn.discordapp.com/attachments/1362196854460383353/1380661744770351227/image.png?ex=6844b0f3&is=68435f73&hm=41bc4cd125990478b21f70b6c8abe41e1582a56b45b2a81ffc157f4befc57689&)，展示了不同的表示法如何影响内存布局。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1380261355159818330)** (146 messages🔥🔥): 

> `Network Bandwidth for 30 Machines, LLM access to terminal/browser, DDR5 RAM limitations on AMD Zen 5, Hugging Face Hub Outage, OCR models` 


- **30 台机器的网络瓶颈忧虑**：一名成员提到，对于 **30 台机器**，高网络带宽和完美的拓扑结构对于避免 *allreduce* 操作时的瓶颈至关重要。
   - 另一名成员建议 **10G** 网络对于他们的项目可能已经足够。
- **给予 LLM 不受限的访问权限，引发混乱**：一名成员授予了他们的 **LLM** 对终端和浏览器的不受限访问权限，并开玩笑说 *不会出任何问题的*。
   - 另一名成员回应，好奇它是否尝试过**给 FBI 发邮件**。
- **DDR5 RAM 限制令爱好者感到困惑**：讨论围绕 **AMD Zen 5 CPU** 被限制在最大 **128GB** RAM 展开，尽管 **64GB DDR5** 内存条已经面世，且某些主板支持 **256GB RAM**。
   - 有推测称，MOE 模型可以在极小的 VRAM 和充足的 RAM 下运行，甚至在过时的 CPU 上也可以，而较新的 CPU 却奇怪地被锁定在较低的 RAM 限制上。
- **Hugging Face Hub 遭遇短暂“心肌梗塞”**：**Hugging Face Hub** 经历了停机，导致 **502 错误**，用户报告从各地访问该网站均出现问题。
   - 基础设施团队迅速解决了问题，赢得了对其高效工作的赞赏，大家也因这不是一次“大事故”而感到宽慰；此次停机可能与工作人员在[相关讨论](https://huggingface.co/spaces/transformers-community/support/discussions/13#6842efbfac97e96a2f38dcbe)中的回复时间重合。
- **用户报告 Claude API 响应质量不佳**：一些用户报告称，在不启用扩展推理（extended reasoning）的情况下，**Claude API** 的响应质量较差。
   - 扩展推理会使模型变得过慢，用户正在寻找解决这一问题的替代方案。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1380305077096681492)** (2 messages): 

> `Fraud Detection in Finance, Resources for Learning Fraud Detection` 


- **询问欺诈检测的学习资源**：一名成员正在寻求学习**欺诈检测**的资源，特别是针对**金融交易**背景。
- **社区期待防欺诈指南**：该成员正在寻找可以帮助他们理解和实施**欺诈检测**技术的教程、文档或指南。
   - 社区现在准备分享有助于学习异常检测（anomaly detection）、欺诈机器学习模型和实时交易分析的资源。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

0xcc6434: Morning
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1380535468676087899)** (6 条消息): 

> `ConvNeXt-Tiny model, audio deepfake detection, Gradio application, PDF parser, DeepFake detection company` 


- **针对音频 Deepfake 检测微调的 **ConvNeXt-Tiny****：一名成员微调了 **Facebook 的 ConvNeXt-Tiny** 模型，用于将音频分类为真实或伪造，将计算机视觉技术与音频分析相结合进行 Deepfake 分类研究，并托管在 [Hugging Face Space](https://huggingface.co/spaces/kubinooo/convnext-tiny-224-audio-deepfake-detection) 上。
   - 另一名成员使用来自 **11elevenlabs** 生成的伪造音频测试了该模型，但它被错误地分类为 **100% Real**（100% 真实）。
- **调试 **ConvNeXt-Tiny** 音频 Deepfake 检测应用**：创作者承认了分类错误，认为这可能是由于模型泛化能力不足或 **Gradio 应用** 进行了多次预测导致的。
   - 创作者提到：“模型做出的预测本来是正常的，但随后突然对 2 张随机图像又做了一次预测……然后这个预测结果被打印到了输出中。”
- **为 DeepFake 检测项目提供帮助**：一名成员提出将模型创作者引荐给创立了 **DeepFake 检测公司** 且有职位空缺的朋友。
   - 他们表示：“如果你想让我帮你联系他们寻求帮助，请告诉我 :) 我相信他们甚至有实习生和高级职位的空缺。”
- **提到的 PDF 解析器工具**：创作者提到了一个 [PDF 解析器工具](https://huggingface.co/kalle07/pdf2txt_parser_converter)。
   - 该工具还链接到了一篇 [X 帖子](https://x.com/EnricoShippole/status/1931023312647299405)。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1380522337518092338)** (1 条消息): 

> `Hugging Face Computer Vision Hangout, Pruna AI, Image Generation Speed` 


- **分享 Hugging Face CV 聚会幻灯片**：一名成员分享了今天 **Computer Vision Hangout** 的幻灯片，包括 **Hugging Face** 计算机视觉领域的更新：[HF_CV_Hangout_June_25.pdf](https://cdn.discordapp.com/attachments/922424143113232404/1380522336335298650/HF_CV_Hangout_June_25.pdf)。
- **Pruna AI 展示图像生成加速方案**：来自 **Pruna AI** 的一名成员做了关于加速图像生成的演讲，幻灯片已分享至：[PrunaAI_SpeedUp_Generation.pdf](https://cdn.discordapp.com/attachments/922424143113232404/1380522337027227709/PrunaAI_SpeedUp_Generation.pdf)。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1380666505787871404)** (1 条消息): 

> `Hackathon Extension, Builder Community, Prize Pool` 


- **黑客松截止日期延长！**：黑客松截止日期已延长 **两天**，现在将于 **6 月 10 日（星期二）UTC 时间结束**。
   - 延期是由于社区极高的参与度和项目开发热情，Discord 频道里的讨论非常活跃，详见[此处](https://discord.com/channels/879548962464493619/1376476916055281776)。
- **社区拥有 4100 多名开发者**：黑客松社区已发展到超过 **4100 名开发者**，目前有超过 **200 个项目** 正在进行中。
   - 参与者正在积极使用赞助商提供的 API 额度，Discord 频道充满了讨论与协作。
- **奖金总计 1.65 万美元现金及 100 多万美元 API 额度**：黑客松在不同赛道提供总计 **1.65 万美元现金** 的奖池，以及来自赞助商的超过 **100 万美元的 API 额度**。
   - 鼓励参与者利用额外的时间完善 Demo、改进文档并提交出色的作品。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1380474136475734086)** (12 messages🔥): 

> `Gemini 框架, Smolagents 与 Gemini, 每月认证, GPT-4o 解析错误` 


- **Gemini 使用 Smolagents 框架**：一位成员在被问及是使用 **Llamaindex** 还是 **Langchain** 时，提到在 **Gemini** 上使用了 **smolagents** 框架。
   - 另一位用户询问 *你是如何实现这一点的*。
- **每月认证是否采用滚动录取？**：一位成员询问每月认证是采用滚动录取（rolling admission）还是 *一次性* 完成。
   - 另一位用户指出，设置截止日期是为了 *让一批学生共同进步，这暗示未来会有更多的班次（cohorts）*。
- **寻求课程证书学习动力**：一位新加入课程的学员正在寻找学习伙伴和动力，以便在截止日期前完成证书。
   - 该用户打算完成课程中的两项认证。
- **GPT-4o 在 Smolagents 中解析表现不佳**：一位成员询问其他人在将 **GPT-4o** 和 **GPT-4o mini** 与 **smolagents** 代码 Agent 配合使用时，是否也遇到了大量的解析错误。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1380261350353273003)** (146 messages🔥🔥): 

> `Gemini 2.5 Pro 评估, Kingfall 基准测试结果, Opus 对比 Gemini, 模型中的上下文处理` 


- **Gemini 2.5 Pro 对比其他模型：初步基准测试**：**Gemini 2.5 Pro** 的早期基准测试是在 Vertex AI 上一个公开暴露的模型上运行的，使用了来自 [此 commit](https://github.com/cheahjs/aider/commit/cacd932c9a474f871229b166e6be0d1858854e17) 的特定设置。
   - 讨论中提到 **86.2%** 的基准测试分数是否可以实现，但其他人表示结果具有 *随机性（stochastic）*。
- **Gemini 2.5 Pro 对比 Opus：用户偏好存在分歧**：一些用户对 **Gemini** 超越 **Opus** 表示惊讶，并引用自己的经验称 **Opus** 提供了更好的结果。
   - 然而，其他人强烈倾向于 **Opus**，认为在编程任务中 **Gemini** 与 **Opus** 相比几乎没用。
- **Opus 和 Gemini 性价比辩论**：讨论的一个关键点围绕不同模型的性价比展开，一位用户表示 *价格是你付出的，价值是你得到的*。
   - 虽然一些人优先考虑较低的成本，但另一些人则关注输出质量和工作流协同效应，即使这意味着使用更昂贵的模型。
- **较新版本的 Gemini 2.5 Pro (06-05) 对比旧版本**：一位成员提到，aider 中 **gemini 2.5 pro** 的默认编辑格式仍然是 *diff-fenced*，而不是 *udiff-simple*。
   - 另一位成员指出，在 *diff-fenced* 上的测试显示 06-05 版本的格式良好率达到 **99%**，即使 *udiff-simple* 证明更好，提升空间也不会太大。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1380291346971230270)** (12 messages🔥): 

> `aider 对比 cursor, gemini stt, superwhisper, aider 配合 vllm server, 嵌入式开发的 cpp/rust` 


- **Aider 用户评估 Cursor**：一些 Aider 用户正在 *尝试* 切换到 **Cursor**，但发现它感觉更慢，默认的聊天机器人非常啰嗦，而且其 Agent 模式非常狂野，必须加以控制。
   - 一位用户表示 **aider 的方法** 更符合他们的开发风格：*(谨慎、深思熟虑的提示词)、终端驱动、疯狂创建分支（因为偏执于能够恢复到已知的良好状态）等等……*，并且不喜欢 *从 Cursor 粉丝那里感受到的那种“胡乱尝试”的氛围*。
- **用户请求语音转文本工作流**：一位用户对 **语音转文本（speech-to-text）工作流** 感兴趣，并试图在生活中增加更多的语音转文本应用。
   - 他们尝试了 iOS 上的 **Wispr flow**，但更喜欢 **superwhisper**。
- **Aider 无法配合 vllm server 工作**：一位用户在配置 **aider** 使用其本地网络运行的 **vllm server** 时遇到困难，因为 aider 要求模型名称以提供商开头，例如 *openai/Qwen3*。
   - 另一位用户建议添加 *openai/* 前缀，即 *openai/unsloth/Qwen3*。
- **寻求适用于 C++/Rust 和嵌入式系统的模型**：一位用户正在寻找适用于 **cpp/rust** 工作负载的模型，该模型也应擅长嵌入式工作负载，例如可能涉及 **esp32** 的开发。
   - 他们拥有 **32gb ddr4 ram + 3090**，并倾向于仅在 GPU 上运行以利用 VRAM 速度。
- **发展太快，需要项目上下文追踪器**：一位用户正在尝试 **AI 编程工具**，需要某种外部文件或数据库来管理上下文。
   - 该用户表示 *伙计，AI 编程工具发展太快了，尝试它们需要某种外部文件或数据库来管理上下文*。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1380263355784364295)** (76 条消息🔥🔥): 

> `App Based Spam, Gemini Rate Limits, Open Thinker Model vs Qwen3-4B, LM Studio and OpenAI API, LM Studio RAG Embedding Model` 


- **基于 App 的垃圾信息引发担忧**：成员们报告了频道中出现的一种新型“基于 App”的垃圾信息，引发了对账号被盗的担忧。
   - 工作人员移除了“使用外部 App”权限，并确认经过初步检查，**没有账号被盗**。
- **Gemini 2.5 Pro 速率限制公开**：一位用户询问了 LM Studio 中 **Gemini** 的速率限制，另一位成员表示 **2.5 Pro** 的限制为 **每 24 小时 100 条消息**。
   - 另一位用户更新称，速率限制已调整为 **150 RPM**。
- **Qwen3-4B 击败 Open Thinker**：成员们讨论了新的 **Open Thinker** 模型，其中一位指出根据官方 Benchmark 数据，**Qwen3-4B** 表现更优。
   - 一位用户分享说 [Qwen3-4B 在他们的游戏 PC 上运行流畅](https://huggingface.co/Qwen/Qwen3-4B)。
- **LM Studio 可与 OpenAI API 配合使用**：一位用户询问关于将 **LM Studio** 与 **OpenAI API** 配合使用的问题，出于成本考虑他们更倾向于这种方式，并正在寻找一个安全的 UI 来提供他们的 Key。
   - 有人建议他们查看 [Open WebUI](https://docs.openwebui.com/)。
- **LM Studio 内置 RAG 使用 text-embedding-nomic-embed-text-v1.5-embedding**：一位用户询问了 **LM Studio** 内置 RAG 功能所使用的 Embedding Model。
   - 一位成员澄清说它使用的是 `text-embedding-nomic-embed-text-v1.5-embedding`，目前在内置 RAG 中**没有更改它的选项**，尽管你可以使用 API Server。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1380301743421657219)** (82 条消息🔥🔥): 

> `LM Studio Ryzen AI NPU, Qwen3 Speed, Strix Halo Benchmarks, Llama 3.3 70B performance, Model Quantization` 


- **Ryzen AI NPU：LM Studio 不兼容？**：一位用户尝试在 **Ryzen AI NPU** 上运行 **LM Studio**，但发现硬件未被识别，可能是因为 llama.cpp 不支持 NPU，且链接的 **LM Studio RyzenAI** 版本针对的是 iGPU/GPU。
   - 该用户表示失望，发现 **LM Studio** 并没有像预期那样利用 NPU。
- **Qwen3 235B：在统一内存上速度惊人？**：一位用户使用带有 **AMD Ryzen AI Max+ 395** 和 **128GB** 统一内存的 **GMKtec Evo-X2**，在 **Qwen 3 235B Q3_K_S**（Context: 12233）的首句回复中达到了 **12.05 tok/sec**，并指出即使在几次 Prompt 之后，10 t/s 仍被认为是不错的。
   - 该用户还成功在 **Q8_0** 下加载了 **64k Context**，达到了 **9.33t/s**，首个 Token 用时 **6.27s**，尽管在使用 **Unsloth Q3_K_XL** 时遇到了重复问题。
- **Strix Halo 基准测试：备受期待**：成员们表达了对 **Strix Halo** 基准测试以及 **DGX Sparc** 结果的期待，一位用户表示如果有兴趣，可以提供他们的 **Strix Halo** 测试结果。
   - 另一位用户请求获取 **70-200B 参数模型** 在 **128k Context** 下的数据。
- **Llama 3.3 70B：全显存占用**：一位用户测试了 **Llama 3.3 70B 在 F16 精度下的 128k Context**，完全在 VRAM 中运行，首个 Prompt 达到 **4.85 tok/sec**，第二个 Prompt 达到 **4.83 tok/sec**。
   - 他们提到 **F32** 可能也可以使用，虽然他们没看到明显收益，并说明他们使用的是 **GMKtec Evo-X2, AMD Ryzen AI Max+ 395 搭配 128GB 统一内存**，其 iGPU 为 **8060S**，在 AI 计算方面大致相当于 **3060**。
- **量化怪癖：忽略 /no_think**：讨论涉及了模型 Quantization，一位用户注意到随着量化程度增加，较小的模型越不容易遵循 `/no_think` 命令。
   - 另一位用户表示他们在 **Q3** 时遇到了奇怪的情况，更倾向于从 **Q4** 及以上开始使用。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1380270960690724937)** (2 条消息): 

> `Modular, magic, pixi, Mojo upgrade, memory alignment` 


- **Modular 团队的 Magic 到 Pixi 迁移**：一位成员感谢 Modular 团队实现了从 `magic` 到 `pixi` 的平滑迁移，并将其描述为“引脚兼容（pin compatible）”的过程。
   - 该成员使用表情符号表达了对这种无缝迁移的感激和赞赏。
- **带有内存对齐的 Mojo 升级**：一位成员提到了在系统上升级 Mojo 时的 *Memory Alignment（内存对齐）* 问题。
   - 另一位成员附和了在处理 Mojo 升级时内存对齐的重要性。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1380265670284935242)** (144 条消息🔥🔥): 

> `Mojo 在生物信息学中的应用、不可变变量、简洁语法、用于编写 Mojo 代码的 LLM / 提示词、Intel Mac 构建` 


- **Mojo 吸引生物信息学开发者**：一位开发者表达了在生物信息学中使用 **Mojo** 的兴奋之情，强调了在为生物技术初创公司实现 **SLURM** 和 HPC 解决方案时所面临的有趣挑战。
   - 他们指出，研究人员通常会创建以结果为导向的软件和自动化解决方案，但这些方案并不总是在整个行业内传播。
- **不可变性需求**：一名成员询问如何在 Mojo 中声明不可变值，寻求一种在初始化后不应更改的运行时值。
   - 另一名成员澄清说，目前还没有创建不可变变量的方法，但[建议使用辅助函数作为变通方案](https://github.com/modular/modular/blob/main/mojo/proposals/remove-let-decls.md)来实现不可变引用（immutable refs）。
- **关于简洁语法的争论**：开发者们讨论了 Mojo 语法的冗长性，特别是 struct 定义中的 `var` 关键字，有些人觉得它很繁琐，而另一些习惯了 Rust 等语言的人则几乎没有注意到。
   - 讨论延伸到了对简洁语法的偏好，一位成员表示自己是 **K** 编程语言的粉丝，该语言以其“象形文字”般的特性而闻名。
- **Mojo 与编程助手协作良好**：成员们分享了在 Mojo 中使用 LLM 的技巧，参考了[文档](https://docs.modular.com/max/coding-assistants)和[论坛帖子](https://forum.modular.com/t/tips-on-using-code-generation-tools-with-mojo/1482)。
   - 一位成员发现 **Claude Code** 在生成 Mojo 代码方面非常有效，甚至能自主测试和优化其输出。
- **x86 Mac 将不会获得太多支持**：成员们讨论了对基于 Intel 的 Mac 缺乏支持的问题，一位开发者建议使用 **Multipass** 或 **Docker** 作为虚拟机（VM）。
   - 一名工作人员指出，考虑到 **x86 Mac** 已经停产，直接支持它的可能性不大，并暗示 **Windows 支持** 具有更高的优先级。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1380268802415136960)** (129 条消息🔥🔥): 

> `工具集成推理、用于工具调用的 Atropos 环境、LLM 数据版权问题、AllenAI 的 OLMo 模型与可复现性、从零开始训练 LLM` 


- **OLMo 模型是真正的开源资源**：成员们提到 [Allen.ai 的 OLMo 模型](https://arxiv.org/abs/2501.00656)是完全开源的，包括**训练代码、数据和论文**。
   - 一位成员指出 *RedPajama 有点脏且旧*，建议查看 [HuggingFace 的 Fineweb-2 数据集](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)及其消融实验。
- **Atropos 是工具调用的新领域**：成员们正在积极开发 [Atropos 环境](https://github.com/NousResearch/atropos/pull/163)，并为**经过验证的推理轨迹答案**生成数据集。
   - 通过[这个环境](https://github.com/NousResearch/atropos/blob/main/environments/tool_calling_server.py)，团队在 Berkeley 工具调用基准测试中将 DeepHermes 的单工具和并行工具调用基准分别提升了 **5 倍和 2.5 倍**。
- **解锁顺序工具调用能力**：一位成员确认他们正在努力将**工具调用直接训练到推理轨迹（reasoning trace）中**。
   - 此外，目前的重点是**顺序工具调用**。
- **EleutherAI 发布了 The Common Pile 数据集**：[EleutherAI](https://blog.eleuther.ai/common-pile/) 刚刚发布了用于语言建模的**最大商业和授权数据集之一**。
   - 团队在 [X](https://x.com/EnricoShippole/status/1931023312647299405) 上发布了这一公告。
- **LLM 的可复现性全靠感觉（vibe）**：一位成员分享了一张图片的链接，上面写着：*通过 Transformer 预训练，人们发现了一个蛋糕配方。但在我看来，导致蛋糕制成的化学原理并没有被真正理解。*
   - 另一位成员回应道：*LLM 的“烹饪”是**基于感觉（vibe-based）且随性的（yolo）**。*

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1380293734289248367)** (5 条消息): 

> `Voigt-Kampff test, Obsidian, XQuartz, Docker, Hermes` 


- **寻求 Voigt-Kampff 测试对象**：一位成员开玩笑说他们正在为 **Voigt-Kampff** 测试做准备，并请求有人来主持测试。
   - **Voigt-Kampff** 测试是 **Blade Runner** 宇宙中用于检测复制人（人造人）的虚构测试。
- **Obsidian 设置偏好曝光**：一位成员提到他们可能不会使用 **Obsidian** 版本，而是更倾向于他们那套“古怪的 **XQuartz** 搭配 **Docker**”的设置。
   - 另一位成员对这种视觉设置评价为“太酷了（rad）”。
- **成员更倾向于 Hermes**：两名成员表示他们更喜欢 **Hermes**。
   - 未给出进一步说明。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

wandabells: https://www.deeplearning.ai/courses/
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1380284788685934673)** (2 条消息): 

> `OpenRouter, RSS Feed for models, API models` 


- **OpenRouter 发布模型 RSS Feed**：OpenRouter 宣布为其 [API models](https://openrouter.ai/api/v1/models?use_rss=true) 提供 **RSS feed**。
- **通过 RSS 获取 OpenRouter 模型更新**：用户现在可以订阅 [RSS](https://openrouter.ai/api/v1/models?use_rss=true)（简易信息聚合）源，以获取有关新模型以及 OpenRouter 生态系统内变化的最新信息。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 条消息): 

insight_cheats: 供 gooners 使用 - https://personality.gg
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1380265215555403816)** (130 条消息🔥🔥): 

> `Gemini 2.5 Pro regression, Claude Max vs Gemini pricing, OpenAI logging practices, Gemini 2.5 flash lite, GPT-4.1 mini is good for many routine task` 


- **Gemini 2.5 Pro 智力退化**：用户报告称 **06-05** 版本的 **Gemini 2.5 Pro** 似乎比之前的版本更笨，有人将其描述为“flash thinking 级别的愚笨”。
   - 另一位用户建议在旧模型仍可用时继续使用，并表示新版本被缩减了规模以实现更快、更便宜的运行。
- **建议“盗版” Gemini**：一位用户开玩笑地建议通过非正规手段使用 **Gemini 2.5** 以避免付费，这引发了关于 **Claude Max** 与 **Gemini** API 使用成本效益的讨论。
   - 该用户认为 **Claude Max** 在 *vibe coding* 和日常使用中更经济，特别是对于那些对 API 成本敏感的用户。
- **OpenAI 日志记录引发隐私担忧**：[一篇文章](https://arstechnica.com/tech-policy/2025/06/openai-says-court-forcing-it-to-save-all-chatgpt-logs-is-a-privacy-nightmare/)指出 **OpenAI** 被迫记录所有输出，这引发了关于 **OpenRouter** 数据保留的疑问。
   - 澄清说明：“启用训练和日志记录”设置与 OpenAI 模型无关，且 OpenAI 可能会将输入保留长达 30 天。
- **Gemini 2.5 Flash Lite 即将推出**：用户对即将推出的 **Gemini-2.5-flash-lite** 模型进行了猜测，意见分歧在于它是有用还是仅仅是一个低质量、更便宜的选择。
   - 一些人认为如果价格和性能相当，它可能会取代旧的 **1.5 Flash**，而另一些人则认为它可能被“严重低估”。
- **GPT-4.1 Mini 在编程和工具使用方面备受赞誉**：**GPT-4.1 mini** 因其编程能力、工具使用和成本效益而受到称赞，非常适合日常任务和推理。
   - 它被认为是“真正的赢家”，并且比 **Gemini 2.5 Flash** 更听从指令，特别是在不涉及代码或数学的任务中，尽管在创意写作方面不如 **Claude 3.7**。

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1380282993452781579)** (76 条消息🔥🔥): 

> `MCP Inspector Fork, Cloudflare Workers 上的 MCP Server, MCP 使用案例, MCP 客户端, 实时代码库索引` 


- **内置 LLM Chat 的 MCP Inspector Fork 亮相**：一个内置了 **LLM chat** 和 **sampling 支持** 的 MCP inspector fork 已发布，欢迎通过 [GitHub](https://github.com/MCPJam/inspector) 进行测试和反馈。
- **MCP Server 在 Cloudflare Workers 上运行**：一位用户报告了在 **Cloudflare Workers** 上使用 **workers.dev 链接** 部署 MCP server 时遇到的问题。
   - 该成员即使在为所有工具添加了描述后仍然感到困惑，并正在寻找*成功将自定义 MCP server 添加到 OpenAI* 的经验分享。
- **VAPI MCP 硬件演示上线**：一位成员分享了一个演示，展示了 **VAPI MCP** 拨打硬件商店电话进行零件采购的过程，目标受众为 **硬件工程师**，并附带了该工具运行的 [视频](https://cdn.discordapp.com/attachments/1312302100125843479/1380307827385438338/helion_call_demo.mp4?ex=6844b8d6&is=68436756&hm=cb6e38829856a5ca52b546356018a0237ce82d3e80ce08b9ca48d589838754ac&)。
- **客户端选择：构建你自己的 MCP 工作流**：一位成员寻求关于 MCP 客户端的建议，要求允许构建自定义工作流且不带网页搜索等默认功能，*更倾向于不预设我的需求，而是让我构建自己工作流的工具*。
   - 其中一个建议是 **5ire**，因其缺乏额外的工具或提示词而受到关注；另一个建议是*部署 10 个 computer use Agents 来浏览 homedepot.com，以确保他们有库存商品*。
- **深入探讨 Sampling 使用案例**：讨论围绕 MCP 中的 **sampling 功能** 展开，质疑其预期目的是为了增强工具调用还是服务器发起的请求，以及它对潜在的长时间运行、空闲会话中客户端-服务器通信的影响，并附带了相关 [图片](https://cdn.discordapp.com/attachments/1312302100125843479/1380504272348905532/image.png?ex=68441e4b&is=6842cccb&hm=74032315eb40b2a5ded1df3b7d073fbd156a11c345d265483311909a27c8ded4&)。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1380469800085487706)** (2 条消息): 

> `inked github, Slack MCP server` 


- **Inked 极简服务器在 GitHub 上发布**：一位成员分享了一个名为 **inked** 的极简服务器，现已在 [GitHub](https://github.com/coldielb/inked) 上可用，鼓励大家试用并提交 PR。
   - 该服务器包含 **两个工具** 和 **三个总函数**，可以通过 `npm install -g @frgmt/inked` 进行全局安装。
- **Slack MCP server 中现支持静默 AI Agents**：一位成员宣布他们的 **Slack MCP server** 在 GitHub 上受到关注，强调其能够构建 **静默**、**隐形** 的 **AI Agents**，而无需在 Slack 中创建机器人或应用程序。
   - 该服务器已在 [GitHub](https://github.com/korotovsky/slack-mcp-server) 上线，并附带了 [展示其用法的 GIF](https://cdn.discordapp.com/attachments/1315696461316358175/1380517187785326692/434543420-35dc9895-e695-4e56-acdc-1a46d6520ba0.gif?ex=68442a52&is=6842d8d2&hm=78437dbb1f2f8f9776d0855153c1d68e2ec00098b74fbddc18ba4e53e272148e&)。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1380293156163158017)** (56 messages🔥🔥): 

> `Claude Projects 内容容量提升, Qwen3-Embedding 与 Qwen3-Reranker 系列, Netlify DB Serverless Postgres, Zapier AI 熟练度评估, Cursor 融资轮次` 


- **AMD 收购 Untether AI 团队**：AMD 收购了 AI 芯片初创公司 [Untether AI](https://www.crn.com/news/components-peripherals/2025/exclusive-amd-acquires-team-behind-ai-chip-startup-untether-ai) 背后的团队。
- **Claude Projects 现在容量提升 10 倍**：Anthropic 宣布其 **Claude Projects** 功能现在支持 **10 倍以上的内容**，并激活了新的检索模式以扩展功能上下文。
   - 该更新正推送到所有 Claude 付费计划，用户称其为“游戏规则改变者（game changer）”，且相比 **ChatGPT** 有显著改进。
- **阿里巴巴向全球开放 Qwen3**：阿里巴巴的 **Qwen3-Embedding** 和 **Qwen3-Reranker 系列**模型发布，在多语言文本嵌入和相关性排序方面树立了新标准，支持 **119 种语言**。
   - 模型提供多种尺寸（0.6B, 4B, 8B），已在 [Hugging Face](https://huggingface.co/)、[GitHub](https://github.com/) 和 [ModelScope](https://modelscope.cn/) 开源，助力文档检索、**RAG**、分类和情感分析等多种用例。
- **Netlify 通过 SupabaseDB 进军 Serverless**：Netlify 宣布推出 **Netlify DB**，这是一个由 Neon 驱动的 Serverless Postgres 数据库，专为 AI 原生开发设计，旨在减少代码与数据之间的摩擦。
   - **Netlify DB** 可以通过单个命令轻松设置，并可通过 `netlify dev` 集成到项目中。
- **Zapier 寻求具备 AI 熟练度的员工**：Zapier 正在衡量员工的 AI 熟练度，要求 **100% 的新员工必须具备 AI 熟练度**，评估将熟练程度分为“不合格（Unacceptable）”、“胜任（Capable）”、“采纳（Adoptive）”和“变革（Transformative）”等级别。
   - 公司通过筛选、技能测试、异步练习和现场面试进行评估。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1380356409882841100)** (47 messages🔥): 

> `HFModelTokenizer, Axolotl 损失曲线, Reward Modeling RFC, Fused Optimizer 问题` 


- **HFModelTokenizer 提示词评估自定义**：讨论围绕修改 **HFModelTokenizer** 以在不进行 Tokenization 的情况下渲染模板进行评估展开，并考虑了对最终用户至关重要的自定义提示词模板。
   - 提议的解决方案包括：如果 Tokenizer 中存在自定义提示词模板，则优先使用；否则，如果 `apply_chat_template` 为 true，则使用 **HFModelTokenizer** 聊天模板；如果没有可用的提示词模板，则抛出错误。
- **Alpaca Cleaned 上的回归检测**：一名成员报告在 **alpaca_cleaned** 数据集上难以复现回归问题，并请求提供最初检测到回归时的环境设置详情。
   - 他们还观察到，在 **alpaca_cleaned** 数据集上微调 **Qwen3-4B** 的评估结果与未微调版本相同，但其他人指出 Alpaca 是一个相当饱和的数据集，且 4B 模型较小。
- **C4 微调后评估 Axolotl 收敛性**：一名成员分享了 [Axolotl PR #2590](https://github.com/axolotl-ai-cloud/axolotl/pull/2590) 的链接，展示了在 **C4** 上的损失曲线，建议在 **C4** 微调后评估 **torchtune**，因为 **Axolotl 会收敛**。
   - 他们指出损失曲线并未强烈表明 **torchtune** 的方法发散，并提议分享以 **Axolotl** 数值为参考的更新。
- **关于裁剪 Logprobs 的讨论**：随后讨论了是否在 **torchtune** 中加入 **Logprobs 裁剪**，一名成员指出，他们不认为某个提议的功能在其他仓库中存在就是将其加入 torchtune 的标准。
   - 虽然该功能在其他地方可用，但有人担心它不打算让用户修改且难以正确暴露；然而，另一名成员更倾向于确保自我实现的便利性，而不是直接维护该功能。
- **Fused Optimizer 触发 AssertionError**：一名成员报告了在 nightly 构建版本中使用 Fused Optimizer 时，与 `fused_adagrad` 相关的 `AssertionError`，特别是在未找到计算网格（compute mesh）的情况下。
   - 经过测试发现，该问题仅在 `fused=True` 时出现，且在升级到最新的 **torchtune** 后 **SGD** 开始正常工作。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1380292133021548665)** (33 条消息🔥): 

> `训练 LLMs，LLMs 数据集，使用 RAG 进行邮件聚类，Meta 的 OPT-175B 训练日志，GPT 谄媚性 (sycophancy)` 


- ****Marius 论文浮出水面****：一名成员分享了 [Marius 论文](https://arxiv.org/abs/2402.00854) 的链接，作为潜在的参考资源。
- ****LLM 训练资源探索开启****：一名成员正在寻求关于使用真实世界数据集训练工业级 LLMs（而非玩具模型）的资源，包括专家在数据处理、多样化输出和防止过拟合方面的见解。
   - 他们提到 Sebastian Raschka 的 "Build a Large Language Model (From Scratch)" ([YouTube 链接](https://youtu.be/Zar2TJv-sE0)) 是一个起点，但希望能找到更详细的训练流水线，涵盖混合、多样化的数据集，以及稳定训练和防止灾难性遗忘的方法。
- ****RAG 聚类难题****：一名成员询问如何利用邮件和开箱即用的 RAG 解决方案进行聚类，目标是在不知道 n 的取值且没有标签的情况下，将邮件放入 n 个桶（bins）中。
   - 一位成员建议使用 ModernBERT 对每封邮件进行 Embedding，并使用旅行商问题（TSP）求解器根据距离将它们排列成簇；另一位成员则建议使用 OpenAI 的 Embeddings ([platform.openai.com](https://platform.openai.com/docs/guides/embeddings#ho))。
- ****Meta 的 OPT-175B 训练日志公开****：一名成员提到了 **Meta 的 OPT-175B 训练日志**，该日志记录了训练过程中的各种问题 ([GitHub 链接](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf), [ArXiv 链接](https://arxiv.org/abs/2205.01068))。
- ****ChatGPT 的谄媚性 (Sycophancy) 受到质疑****：一名成员询问 **ChatGPT** 是否正变得越来越谄媚（迎合用户）。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1380291881120039115)** (2 条消息): 

> `Vec2Vec 代码审查，Translators/Transformers 目录，实现的背景审查` 


- **Vec2Vec 代码深度解析已排期**：一次针对 [Vec2Vec](https://github.com/rjha18/vec2vec) 的代码审查已排期，重点关注 `translators/transformers` 目录及更广泛的 `translators` 代码库。
   - 讨论将涵盖一篇论文 ([https://arxiv.org/abs/2505.12540](https://arxiv.org/abs/2505.12540)) 的实现，并在深入代码之前进行背景审查。
- **会议推迟**：原定的会议已推迟到下周。
   - 用户澄清说，他们在通话中误说了“明天”，会议现在定于下周举行。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1380595188501844192)** (7 messages): 

> `Qwen3 Embedding, Nemotron-H Reasoning Model, EleutherAI and Public Data LLMs, Cohere's Business Model, RAG Marketing` 


- ****Qwen3** Embedding 以 **Apache 2.0 协议**发布**：阿里巴巴发布了 **Qwen3 Embedding** 系列，专为文本嵌入、检索和重排序任务设计，利用了 **Qwen3** 基础模型，并在 [Hugging Face](https://huggingface.co/Qwen) 和 [ModelScope](https://modelscope.cn/models?search=Qwen) 上以 **Apache 2.0 协议**提供。
   - 该系列采用双编码器（dual-encoder）和交叉编码器（cross-encoder）架构，通过 LoRA 进行微调以增强文本理解，[技术报告和代码已在 GitHub 上发布](https://github.com/QwenLM/Qwen-Embedding)。
- ****NVIDIA 的 Nemotron-H** 推理模型提升吞吐量**：NVIDIA 推出了 **Nemotron-H-47B-Reasoning-128K** 和 **Nemotron-H-8B-Reasoning-128k** 模型以应对推理密集型任务，现已提供 [FP8 量化版本](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/?linkId=100000368479233)，以便在延迟敏感的环境中实现高效吞吐。
   - 这些模型基于 **Nemotron-H-47B-Base-8K** 和 **Nemotron-H-8B-Base-8K** 基础模型构建，旨在推进推理模型背后的科学研究。
- ****EleutherAI** 使用公共数据训练出具有竞争力的 LLM**：一名成员指出，[EleutherAI](https://huggingface.co/blog/stellaathena/common-pile) 证明了使用公共领域和开放许可数据训练具有竞争力的 LLM 的可行性。
   - 这一成就突显了在不依赖专有数据集的情况下创建强大语言模型的潜力。
- **对 **Cohere** 持续运营的质疑**：一名成员对 **Cohere** 的持续运营和客户群表示困惑。
   - 另一名成员回应称，**Cohere** 直接向企业销售服务和解决方案，特别是在 **RAG**（检索增强生成）应用的背景下。
- ****RAG** 营销提升了 Embedding 的普及度**：有人指出，虽然像 Google 这样的公司早已提供 Embedding 服务，但围绕 **RAG**（检索增强生成）的营销推动了其采用率的增长。
   - 该成员还指出，**Qwen** Embedding 模型凭借其许可协议，是该领域的有力竞争者。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1380396427284316301)** (1 messages): 

> `LLVM, loop splitting, ROCm, InductiveRangeCheckElimination` 


- **Loop Splitting 仅限 ROCm？**：一名成员正在研究使用 **LLVM** 加速 CAT，并询问 **loop splitting** 是否仅存在于 **ROCm llvm-project** 中。
   - 他们引用了关于 loop splitting 的 [ROCm 文档](https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.1/reference/rocmcc.html#loop-splitting)，并指出这仅存在于其自定义的 llvm-project 中。
- **llvm.py 中缺失 InductiveRangeCheckElimination**：一名成员指出，runtime/autogen/llvm.py 中使用的 **LLVM C 源码**缺少来自 **C++ LLVM 库**的 **InductiveRangeCheckElimination**。
   - 由于无法添加 loop splitting，他们正在考虑使用 *llvmlite* 以获取对 IRCE 的访问权限，或者使用 extern/rewrite C++。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1380273021066936430)** (23 messages🔥): 

> `tinygrad kernel optimization, hlb_cifar10 data shuffling, OpenCL kernel performance, GPU indexing kernels` 


- **调试 tinygrad 缓慢的 GPU Kernels**：一位用户正在调试 tinygrad 在 `hlb_cifar10` 示例中打乱 float32 `[50000,3,32,32]` 数据集 Tensor 时生成的缓慢 GPU kernel。
   - 该用户尝试了 `DEBUG=4` 和 `VIZ=1`，但发现输出没有帮助，并确定 `BEAM=4` 无法解决根本问题。
- **手动编写的 OpenCL Kernel 打乱数组速度快得多**：用户测试了一个手动编写的用于打乱相同大小数组（50000,3,32,32）的 OpenCL kernel，发现其打乱耗时仅为 **0.33 秒**。
   - 相比之下，tinygrad 生成的 kernel 即使在简单的未打乱索引下也需要 **5 秒**，这促使对 tinygrad 的 kernel 生成机制进行进一步调查。
- **调查奇怪的 tinygrad 索引 Kernel**：用户试图理解为什么 tinygrad 会生成如此缓慢的索引 kernel，尤其是考虑到基于 CPU 的复制和打乱速度更快。
   - ChatGPT 帮助用户意识到一个简化的索引 OpenCL kernel 会快得多。
- **NumPy 打乱性能**：用户测试了 `np.random.permutations(50000*3*32*32)` 和 `np.random.permutations(50000)[None].repeat(3*32*32, 0).T.flatten()`，两者耗时均为 **0.33 秒**。
   - 用户希望找出是什么导致 tinygrad 生成了如此奇怪的索引 kernel。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1380283058506174575)** (19 messages🔥): 

> `Video Function, Credit Costs, Manish Model Update, Manus partnership with Claude, Egyptian users` 


- **Manus 新视频功能表现不佳**：成员们尝试了新的视频功能，发现它*非常不成熟*。
   - 一位用户提到他们看到了一个亲密朋友制作的视频，觉得它对于实际应用来说还太早。
- **高昂的积分成本促使用户转向替代方案**：用户抱怨 **Manus** 的高昂积分成本，指出 *19 美元兑换 1900 积分* 的价格不足，因为每个任务需要消耗 *300-400 积分*。
   - 一位用户提到由于成本高昂，他们正在使用更便宜的替代方案，并[建议阅读 Manus 团队的指南](https://discord.com/channels/1349440650495398020/1370393476029616238)以执行更低成本的任务。
- **关于 Manish 模型更新的猜测升温**：用户询问模型是否会更新到 **Sonnet 4.0**。
   - 一位用户推测，由于最近与 **Claude** 的合作，这极有可能发生；另一位用户提到模型目前是最新的；而一些用户则指出 **Sonnet** 缺乏 Context Length 是一个问题，而 **Manus** 解决了这一点。
- **埃及用户现身**：一位用户询问聊天室中是否有其他埃及用户。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1380319445553254492)** (3 messages): 

> `AI Agents, MCP vs A2A, Vector Databases` 


- **关于 AI Agents 生产化的辩论开始**：由 @tuanacelik 在 Snowflake Dev Day 主持的讨论环节，探讨了 [AI Agents 生产化的阻碍因素](https://t.co/DJGBe3TqZb)。
- **MCP 与 A2A 标准之争**：在 [MCP Dev Summit](https://t.co/qZv8duKRut) 上，@seldo 快速介绍了 **13 种不同的协议**，这些协议都在争夺成为 Agent 与工具通信的标准，包括 **MCP, A2A 和 ACP**。
- **向量数据库最佳实践登陆慕尼黑**：@itsclelia 将于 6 月 12 日在**慕尼黑的 BASED Meetup** 上发表演讲，分享提升 RAG Pipeline 的最佳实践，涵盖从数据准备到查询优化的各个环节。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1380349256468004944)** (9 messages🔥): 

> `files_via_content mode, AgentWorkflow orchestration, Multi-Agent setup` 


- **LlamaIndex 澄清 `files_via_content` 模式**：一名成员询问了关于 LlamaIndex 中 [`files_via_content` 模式](https://docs.cloud.llamaindex.ai/llamacloud/retrieval/modes#files_via_content-mode) 如何工作的文档。
   - 另一名成员回复了指向 LlamaIndex Cloud 文档相关章节的直接链接，提供了快速解决方案。
- **使用动态委派的 AgentWorkflow 编排**：一名成员咨询了如何在 **AgentWorkflow** 中编排 Agent 团队，特别是如何动态地将任务委派给专业 Agent。
   - 该咨询集中于此功能是 LlamaIndex 内置的，还是需要定义自定义工作流。
- **提供 Multi-Agent 设置示例**：针对关于编排多个 Agent 的查询，一名成员提供了一个 [LlamaIndex 文档示例](https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/) 链接，演示了 **multi-agent 设置**。
   - 该成员解释说，*“orchestrator” Agent 基本上就是一个带有工具的 Agent（而这些工具可能是其他 Agent）*，提供了该架构的概念性概述。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1380292234817044631)** (1 messages): 

> `SchemaLLMPathExtractor, Graph database population` 


- **新手询问关于 `SchemaLLMPathExtractor` 的问题**：一位 LlamaIndex 新用户正在探索使用 [`SchemaLLMPathExtractor`](https://llama-index.readthedocs.io/en/stable/module_guides/indexing/schema/schema_llm_path_extractor.html) 来填充图数据库。
   - 用户询问社区是否发布了可以开箱即用的 Schema（实体、关系、规则）。
- **寻求用于图数据库填充的社区 Schema**：一名成员正在 LlamaIndex 生态系统中寻找用于填充图数据库的预构建 Schema（实体、关系、规则）。
   - 用户希望利用现有的社区资源，以简化将组织数据（人员、应用等）集成到图结构中的过程。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1380315274984816650)** (5 messages): 

> `VPS for API server, RAM Pricing, Mistral MOE, Deepseek MOE, Chinese CPU vendor` 


- **尝试使用 VPS 构建 API 服务器**：一位用户建议租用 **VPS** 来为 **GPT4All** 构建 **API server**。
   - 该用户附上了一张 **GPT4All** 界面的截图，指出它有时没有响应且存在 Bug。
- **探讨 RAM 价格启示**：一名成员分享了一个 [YouTube 视频](https://m.youtube.com/watch?v=Tp0k6VDXUOQ)，讨论了 **RAM 价格**，其中 1 TB 的价格可以合理地维持在几千美元左右。
   - 该成员补充说，普通 **PC** 可能会因 RAM 不足而挣扎，且计算机组件市场是全球化的。
- **惊叹于 MOE 模型指标**：一位用户设想是否能以每秒万亿（TRILLION）tokens 的速度运行 **Mistral MOE** 或 **Deepseek MOE** 的全 **Q8 Quantization** 版本。
   - 该用户链接了一篇关于一家[中国 CPU 厂商](https://www.techradar.com/pro/chinese-cpu-vendor-swaps-amd-zen-architecture-for-homegrown-one-to-deliver-128-core-monster-to-give-epyc-and-xeon-a-run-for-their-money)的文章，该厂商将 **AMD Zen 架构**更换为自主研发架构，以交付一款 **128 核巨兽**，从而与 EPYC 和 Xeon 展开竞争。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1380263648450314384)** (3 messages): 

> `Session Thanks, Blockchain Engineer Introduction, AI Agent Engineer Introduction` 


- **会议获得致谢**：两名成员感谢另一名成员提供的会议分享，并附上了 [YouTube 链接](https://youtu.be/Vqsfn9rWXR8)。
   - 一名成员询问是否可以提供会议幻灯片。
- **工程师自我介绍**：一位在 **Blockchain** 和 **AI Agents** 领域具有经验的软件工程师介绍了自己。
   - 他的专业领域包括 Blockchain 中的 **EVM**, **Solana**, **Cardano**, **Hydra**, **Aptos**, **Cosmos**, **Tron**, **zk-SNARKs**，以及 AI 中的 **LLM**, **NLP**, **LangChain**, **AutoGen**, **TorchRL**, **DL**, **Azure ML**, **AI Agent**。


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

stormortiz: 这里是一个神奇的地方
  

---

### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1380469381951127634)** (2 条消息): 

> `Introductions, ML Audio Engineer` 


- **新成员介绍自己为 ML Audio Engineer**：一位新成员介绍自己为 **Machine Learning Audio Engineer**。
- **社区欢迎新成员**：社区欢迎新成员加入 Cohere Discord 服务器。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/)** (1 条消息): 

radhakrishnan_20251: thanks for the update
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1380550349911490811)** (1 条消息): 

> `MCP Tools Authorization, Enterprise OAuth` 


- **发布了关于 MCP 工具授权的文章**：一位成员分享了一篇关于为企业构建 **MCP tools authorization** 的 [LinkedIn 帖子](https://www.linkedin.com/posts/subham-kundu-2746b515b_ai-enterpriseai-oauth-activity-7336749966234718208-Cb9E?utm_source=share&utm_medium=member_desktop&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk)。
   - 该帖子将关于 **enterprise OAuth** 的研究结果汇编成了一篇文章。
- **OAuth 研究结果已汇编**：作者关于为企业构建 **MCP tools authorization** 的研究结果已汇编成文。
   - 本文专门探讨了与 **enterprise OAuth** 实施和最佳实践相关的方面。