---
companies:
- openai
- google-deepmind
- meta-ai-fair
- hugging-face
date: '2024-12-19T05:17:10.096057Z'
description: '**OpenAI** 发布了 **o1 模型** API，支持函数调用（function calling）、结构化输出（structured
  outputs）、视觉支持和开发者消息，其推理 token 消耗比预览版减少了 **60%**。该模型在数学和编程领域表现优异，**LiveBench Coding
  评分达 0.76**，超越了 Sonnet 3.5。此外，OpenAI 还发布了 Go 和 Java 的 Beta 版 SDK，并推出了价格降低 **60%**
  的 WebRTC 支持。


  **Google Gemini 2.0 Pro (Gemini Exp 1206)** 的部署也在加速，展现出更强的编程、数学和推理能力。Meta AI FAIR
  介绍了关于利用动态熵补丁（dynamic entropy-based patching）技术直接在原始字节上训练 Transformer 的研究。某行业参与者成功部署了商用人形机器人。


  **Hugging Face** 的研究人员证明，通过搜索技术，其 **3B Llama 模型**在 MATH-500 准确率上可以超越 **70B Llama
  模型**，凸显了小模型的效率优势。同时，相关报告也提到了对可复现性和特定领域局限性的担忧。'
id: fc096fb8-c65f-415f-ac11-5edc3c431554
models:
- o1
- o1-preview
- gpt-4o
- claude-3.5-sonnet
- gemini-2.0-pro
- llama-3-3b
- llama-3-70b
original_slug: ainews-genesis-generative-physics-engine-for
people:
- aidan_mclau
- sundarpichai
- adcock_brett
title: Genesis：面向机器人技术的生成式物理引擎（o1-mini 版本）
topics:
- function-calling
- structured-outputs
- vision
- performance-benchmarks
- sdk
- webrtc
- reasoning
- math
- code-generation
- transformer-architecture
- model-training
- humanoid-robots
- search
- model-efficiency
- dataset-sharing
---

<!-- buttondown-editor-mode: plaintext -->**用于对比的旧版 o1-mini 版本**

> 2024/12/17-2024/12/18 的 AI 新闻。我们为您检查了 7 个 subreddits、[433 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 32 个 Discords（215 个频道，4542 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**497 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

**您正在阅读由 `o1-mini-2024-09-12` 生成的 AINews。按照新前沿模型发布日的传统，我们尝试发布多个版本进行 A/B 测试/自我评估。请查看我们的存档以获取 [o1-2024-12-17 版本](https://buttondown.com/ainews/archive/ainews-genesis-generative-physics-engine-for-6175/)。对于昨天的重复发送（平台 bug）我们深表歉意，但今天的发送是故意的。**


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有摘要由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

以下是按主题组织的重点讨论：

**OpenAI o1 API 发布与特性**

- **o1 模型已发布至 API**，支持 [function calling、structured outputs、vision 支持和 developer messages](https://twitter.com/OpenAIDevs/status/1869156065788715409)。该模型使用的 **reasoning tokens 比 o1-preview 少 60%**，并包含一个新的 "reasoning_effort" 参数。

- **性能基准测试**：[@aidan_mclau 指出](https://twitter.com/aidan_mclau/status/1869068880326635645) o1 在 **“数学/代码方面强得离谱”但在“其他方面表现平平”**。[基准测试结果显示](https://twitter.com/scaling01/status/1869083247554220353) o1 在 **LiveBench Coding 上得分 0.76**，而 Sonnet 3.5 为 0.67。

- **新 SDK**：发布了 [Go 和 Java 的测试版 SDK](https://twitter.com/OpenAIDevs/status/1869140165798821942)。还为 realtime API 添加了 **WebRTC 支持**，**价格降低了 60%**。

**Google Gemini 更新**

- [@sundarpichai 确认](https://twitter.com/scaling01/status/1869072489881747818) **Gemini Exp 1206 即为 Gemini 2.0 Pro**，在编程、数学和推理任务上表现出更强的性能。

- [Gemini 2.0 部署加速](https://twitter.com/asadovsky/status/1869114982971093316)，以响应 Advanced 用户的反馈。

**模型开发与架构**

- 围绕模型大小和训练的讨论——[关于](https://twitter.com/aidan_mclau/status/1869080913860251911) o1-preview 的大小是否与 o1 匹配以及与 GPT-4o 关系的辩论。

- Meta 关于[直接在原始字节（raw bytes）上训练 Transformer](https://twitter.com/LiorOnAI/status/1869409580192555015)的新研究，使用了基于熵（entropy）的动态补丁（dynamic patching）。

**行业与商业**

- [@adcock_brett 报告](https://twitter.com/adcock_brett/status/1869235067580764635)了商用人形机器人在客户现场的成功部署，并实现了从总部的快速迁移。

- [宣布了新的 LlamaReport 工具](https://twitter.com/llama_index/status/1869094544169677138)，用于使用 LLM 将文档数据库转换为人类可读的报告。

**迷因与幽默**

- [关于在 IMAX 观看《Attention Is All You Need》重映版的笑话](https://twitter.com/jxmnop/status/1869154293888258139)

---

# AI Reddit 回顾

## /r/LocalLlama 摘要

**主题 1. Hugging Face 的 3B Llama 模型：通过搜索超越 70B 模型**

- **[Hugging Face 研究人员通过搜索技术使 3B Llama 表现优于 70B](https://i.redd.it/kksacsh1sk7e1.png)** ([Score: 668, Comments: 123](https://reddit.com/r/LocalLLaMA/comments/1hgybhg/hugging_face_researchers_got_3b_llama_to/)): **Hugging Face** 研究人员取得了一项突破，通过使用搜索技术使 **3B Llama 模型**在 MATH-500 准确率上超过了 **70B Llama 模型**。图表显示，在特定条件下，**3B 模型**的表现超过了 **70B 模型**，准确率是根据每个问题的生成次数来衡量的，突显了该模型与大型模型相比的潜在效率和有效性。
  - **推理时间与模型大小优化**：用户讨论了在推理时间和模型大小之间寻找最佳平衡的潜力，认为如果小型模型在特定任务上表现足够好，特别是在知识嵌入 Prompt 或针对特定领域进行微调的情况下，它们会更有效率。
  - **可复现性与数据集引用**：由于 **Diverse Verifier Tree Search (DVTS)** 模型尚未发布，人们对结果的可复现性表示担忧，文中提供了所用数据集的链接 ([Hugging Face Dataset](https://huggingface.co/datasets/edbeeching/dvts_3b)) 以及 DVTS 的实现代码 ([GitHub](https://github.com/huggingface/search-and-learn/blob/main/src/sal/search/diverse_verifier_tree_search.py))。
  - **特定领域的局限性**：由于缺乏在其他领域训练的 **PRMs** 以及具有逐步标注的数据集，人们对该方法在数学和代码领域之外的适用性持怀疑态度，质疑该方法的泛化能力。


**Theme 2. Moonshine Web：比 Whisper 更快、更准确**

- **[Moonshine Web：实时浏览器内语音识别，比 Whisper 更快、更准确](https://v.redd.it/gqh3gg170n7e1)** ([Score: 193, Comments: 25](https://reddit.com/r/LocalLLaMA/comments/1hh5y87/moonshine_web_realtime_inbrowser_speech/)): **Moonshine Web** 声称提供**实时浏览器内语音识别**，比 **Whisper** 更快且更准确。
  - **Moonshine Web** 在 **MIT 许可证**下开源，目前正努力将其集成到 **transformers** 中，详见 [此 PR](https://github.com/huggingface/transformers/pull/34784)。**ONNX 模型**已在 [Hugging Face Hub](https://huggingface.co/models?library=transformers.js&other=moonshine&sort=trending) 上可用，尽管人们对 **ONNX web runtime** 的不透明性表示担忧。
  - 讨论重点包括对 Moonshine 与 **Whisper** 模型（特别是 **v3 large**）相比的**实时能力**和准确性声明的怀疑。用户对该模型执行 **speaker diarization** 的能力以及目前仅限于**英语**的局限性感到好奇。
  - **Moonshine** 针对实时、设备端应用进行了优化，**Transformers.js v3.2** 已添加支持。[演示源代码](https://github.com/huggingface/transformers.js-examples/tree/main/moonshine-web) 和 [在线演示](https://huggingface.co/spaces/webml-community/moonshine-web) 可供测试和探索。


**Theme 3. Granite 3.1 语言模型：128k 上下文与开源许可证**

- **[Granite 3.1 Language Models: 128k context length & Apache 2.0](https://huggingface.co/collections/ibm-granite/granite-31-language-models-6751dbbf2f3389bec5c6f02d)** ([Score: 144, Comments: 22](https://reddit.com/r/LocalLLaMA/comments/1hh403g/granite_31_language_models_128k_context_length/)): **Granite 3.1 Language Models** 现在具备 **128k context length**，并采用 **Apache 2.0 许可证**发布，这标志着在处理更大规模数据集的能力以及对开发者的可访问性方面取得了重大进展。
  - **Granite 模型性能**：据报告，**Granite 3.1 3B MoE 模型**在 Open LLM Leaderboard 上的平均得分高于 **Falcon 3 1B**，这反驳了 MoE 模型性能仅与具有等效激活参数（active parameters）的稠密模型相当的说法。尽管其激活参数比竞争对手**少 20%**，但表现依然出色。
  - **模型规格与许可**：**Granite 稠密模型**（2B 和 8B）以及 **MoE 模型**（1B 和 3B）分别在超过 **12 万亿**和 **10 万亿 tokens** 上进行了训练。稠密模型支持基于工具的使用场景，而 MoE 模型则专为低延迟应用设计。这些模型均以 **Apache 2.0 许可证**发布，其中 8B 模型在代码生成和翻译任务中的表现尤为突出。
  - **社区见解与对比**：**Granite Code 模型**因其被低估的性能而受到赞誉，特别是 **Granite 8BCode** 模型，可与 **Qwen2.5 Coder 7B** 竞争。讨论还强调了 MoE 模型促进各种检索策略的潜力，以及像 Red Hat 集成 Granite 模型这类熟悉的企业级解决方案的重要性。


**Theme 4. Moxin LLM 7B: A Fully Open-Source AI Model**

- **Moxin LLM 7B: A fully open-source LLM - Base and Chat + GGUF** ([Score: 131, Comments: 5](https://reddit.com/r/LocalLLaMA/comments/1hh067r/moxin_llm_7b_a_fully_opensource_llm_base_and_chat/)): **Moxin LLM 7B** 是一个完全开源的大语言模型，在来自 **SlimPajama**、**DCLM-BASELINE** 和 **the-stack-dedup** 的文本和代码数据上进行了训练，实现了优于其他 7B 模型的 zero-shot 性能。它具有 32k context size，支持通过 grouped-query attention、sliding window attention 和 Rolling Buffer Cache 进行长文本处理，所有开发资源均可在 [GitHub](https://github.com/moxin-org/Moxin-LLM) 和 [Hugging Face](https://huggingface.co/moxin-org/moxin-chat-7b) 上获取。
  - **Moxin LLM 7B** 被赞誉为模型训练的绝佳资源，正如 **Stepfunction** 所指出的，它拥有简洁且易于获取的代码和数据集。该模型全面的开发资源被视为一项重大优势。
  - **TheActualStudy** 称赞该模型集成了 **Qwen 级别的 context**、**Gemma 级别的技术**以及 **Mistral-7B-v0.1** 的性能。这种先进方法与数据的结合被认为令人印象深刻。
  - **Many_SuchCases** 提到在探索 GitHub 仓库时发现缺少一些组件（如中间 checkpoints），并猜测这些可能会在稍后上传。


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. Imagen v2 Quality Elevates Image Generation Benchmark**

- **[New Imagen v2 is insane](https://www.reddit.com/gallery/1hh5swo)** ([Score: 680, Comments: 119](https://reddit.com/r/OpenAI/comments/1hh5swo/new_imagen_v2_is_insane/)): **Imagen 3** 随着新版本的发布正在树立 **图像质量** 的新基准，该版本被称为 **Imagen v2**。帖子强调了该技术令人印象深刻的进步，但未提供额外的背景或细节。
  - **访问与使用**：用户讨论了通过 **Google Labs** 网站访问 **Imagen 3**，并建议在受限地区使用 **VPNs**。有人提到在 [labs.google/fx/tools/image-fx](https://labs.google/fx/tools/image-fx) 上可以免费访问，但有一定的每日使用配额。
  - **艺术领域的担忧**：艺术家们对 **Imagen 3** 对艺术行业的影响表示极大担忧，担心对人类艺术家的需求减少，以及传统艺术被 AI 生成的图像所掩盖。一些用户认为，这种转变可能会导致创意领域的私有化和艺术劳动的侵蚀。
  - **模型混淆与改进**：关于 **Imagen 3** 的命名和版本存在一些混淆，用户澄清其为 **Imagen3 v2**。用户注意到图像质量有显著提升，早期测试者对结果表示满意，认为其优于之前的版本。


**Theme 2. NotebookLM's Conversational Podcast Revolution**

- **OpenAI 应该开发自己的 NotebookLM 应用，这太令人震撼了！** ([Score: 299, Comments: 75](https://reddit.com/r/OpenAI/comments/1hgwvwt/openai_should_make_their_own_notebooklm/)): **NotebookLM** 生成的 AI 播客听起来非常自然，在对话质量上甚至超越了 **Huberman** 的播客。该帖子建议 **OpenAI** 应该开发类似的应用，因为这可能会对该领域产生重大影响。
  - **NotebookLM 的语音质量** 受到称赞，但与人类主持人相比仍被认为不够自然，而 **Gemini 2.0** 提供了与播客主持人的实时聊天功能，增强了其吸引力。用户注意到不同平台之间的功能集成问题，强调了在使用高级语音模式和自定义项目方面的局限性。
  - **对话式 AI 的价值**（如总结 PDF 任务）引发了争论，一些人认为它在节省时间和成人学习理论方面具有革命性，而另一些人则认为内容浅薄且缺乏深度。**Gemini** 模型因其巨大的上下文窗口（context window）而受到关注，使其非常适合处理大量信息。
  - **Google 的硬件优势** 被强调，他们在基础设施和能源解决方案上的投资使其能够提供比 **OpenAI** 更具成本效益的 AI 模型。这使得 Google 有可能在播客 AI 领域超越 OpenAI，利用其硬件能力显著降低成本。


**Theme 3. Gemini 2.0 在学术写作方面超越其他模型**

- **Gemini 2.0 Advanced 在学术写作方面表现得异常出色。** ([Score: 166, Comments: 39](https://reddit.com/r/OpenAI/comments/1hgva9g/gemini_20_advanced_is_insanely_good_for_academic/)): **Gemini 2.0 Advanced** 在学术写作方面表现卓越，与包括 **ChatGPT** 在内的其他模型相比，提供了更出色的理解力、结构和风格。作者考虑在 **OpenAI** 发布改进版本之前转向使用 Gemini 2.0。
  - **Gemini 2.0 Advanced** 在 [AI Studio](https://aistudio.google.com/) 上被标识为 **Gemini Experimental 1206**，目前无需付费版本即可使用，尽管用户需要通过交换数据来获得访问权限。**Google** 的命名惯例以及缺乏统一的 AI 服务在用户中引起了一些困惑。
  - **Gemini 2.0 Advanced** 在学术写作质量上表现出显著提升，在评估中优于 **GPT-4o** 和 **Claude**。它提供详细的反馈，经常带有幽默感地批评回复，用户认为这既有效又有趣。
  - 用户讨论了通过订阅获取 **Gemini 2.0 Advanced** 的可能性，对于其在 **Gemini web app** 中被列为 "2.0 Experimental Advanced, Preview gemini-exp-1206" 存在一些困惑。该模型在学术背景下的表现受到称赞，用户希望这将推动 **OpenAI** 解决 **ChatGPT** 中的问题。


**Theme 4. Veo 2 通过逼真的视频生成挑战 Sora**

- **[Google 正在通过其视频生成模型的最新版本 Veo 2 挑战 OpenAI 的 Sora，据称该模型能生成更逼真的视频。](https://v.redd.it/qok7o7rhsl7e1)** ([Score: 124, Comments: 34](https://reddit.com/r/OpenAI/comments/1hh0vwu/google_is_challenging_openals_sora_with_the/)): **Google** 正在与 **OpenAI** 的 **Sora** 竞争，发布了其视频生成模型的新版本 **Veo 2**，声称可以制作更逼真的视频。
  - **Veo 2 的可用性和性能**：几位评论者强调 **Veo 2** 仍处于早期测试阶段，尚未广泛使用，这与关于其已发布的说法形成对比。尽管如此，**Twitter** 等平台上的部分测试者报告了令人印象深刻的结果，特别是在物理特性和一致性等领域，表现优于 **Sora**。
  - **市场策略和可访问性**：人们怀疑这次发布是针对 **OpenAI** 的一种市场策略。对 **Veo 2** 和 **Sora** 缺乏公众访问权限和 API 可用性的担忧普遍存在，并注意到 **aistudio** 已确认将于 **1 月**发布。
  - **对视频真实性的信任**：讨论涉及由于 **Veo 2** 等先进生成模型的出现，视频真实性可能受到的侵蚀。一些人提出了一些解决方案，例如通过区块链注册表使用个人 AI 来验证媒体的真实性，以解决这一问题。


---

# AI Discord 回顾

> 由 o1-2024-12-17 生成的总结之总结的摘要

**主题 1. AI 扩展和项目中的挑战**  

- **Codeium 扩展在 VSCode 中短暂崩溃**：该扩展仅在瞬间显示自动补全建议，导致无法使用。根据多名用户的报告，回滚到 1.24.8 版本可恢复正常功能。  
- **Windsurf 在高负载下性能崩溃**：部分用户遇到了超过 10 分钟的加载时间，以及偶发的“代码消失”或 Cascade 功能损坏。在稳定修复方案出台前，提交支持工单是首选建议。  
- **Bolt 用户对浪费 Token 表示不满**：在收到导致额度耗尽的无关回复后，用户开玩笑地提议增加一个*“拳打 AI”*按钮。许多人呼吁在即将发布的版本中改进记忆控制。

**主题 2. 新模型与升级模型**  

- [**OpenAI o1 的 Function Calling 功能令人惊艳**](https://openrouter.ai/openai/o1-preview)：作为 o1-preview 的继任者，它引入了一个新的 *“reasoning_effort”* 参数，用于控制回复前的思考时长。通过 [OpenRouter](https://openrouter.ai) 使用时，其延迟明显降低。  
- [**EVA Llama 成为故事创作专家**](https://openrouter.ai/eva-unit-01/eva-llama-3.33-70b)：该模型针对角色扮演和叙事任务，据报道在多步骤故事讲述方面表现出色。早期采用者称赞其创意输出和用户友好的设计。  
- **受欢迎模型大幅降价**：MythoMax 13B 降价 12.5%，QwQ 推理模型暴跌 55%。这些折扣旨在扩大社区实验的准入门槛。

**主题 3. GPU 与推理陷阱**  

- **AMD 驱动更新导致性能大幅下降**：用户发现从驱动版本 24.10.1 升级到 24.12.1 后，每秒 Token 数（TPS）从 90+ 骤降至 20 左右。回滚驱动可解决减速问题，这再次提醒用户对新发布的 GPU 驱动保持谨慎。  
- **Stable Diffusion 在 Ubuntu 上遇到障碍**：ComfyUI 或 Forge UI 等工具通常需要深入的 Linux 知识来解决兼容性问题。许多人仍推荐将拥有 16GB VRAM 的 NVIDIA 3060 作为更顺畅的基准配置。  
- [**TinyGrad, Torch 和 CUDA 显存困惑**](https://discuss.pytorch.org/t/reduce-time-to-first-kernel-when-using-cuda-graphs/214310)：移除如 *IsDense(y) && IsSame(x, y)* 之类的检查解决了意外的推理失败，但引入了新的复杂性。这促使开发者参考官方 CUDA Graphs 讨论以寻求潜在解决方案。

**主题 4. 高级微调与 RAG 技术**  

- [**使用 4-bit 转换微调 Llama 3.2**](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama)：许多人依赖 *load_in_4bit=true* 来平衡 VRAM 占用和模型精度。通过部分精度设置，Checkpoint 可以重复使用，并最大限度地减少资源限制。  
- [**Depth AI 大规模索引代码库**](https://www.trydepth.ai/)：它在回答技术查询时达到了 *99% 的准确率*，尽管索引 18 万个 Token 可能需要 40 分钟。虽然存在像 [LightRAG](https://github.com/HKUDS/LightRAG) 这样的竞争方案，但 Depth AI 因设置更简单而受到称赞。  
- [**Gemini 2.0 增加 Google Search Grounding**](https://github.com/BerriAI/litellm/pull/7257)：新配置允许实时网络查询以优化答案。早期评论强调了其在编程和问答场景中事实精准度的提升。

**主题 5. NotebookLM 与 Agentic 工作流**  

- **NotebookLM 翻新其三栏式 UI**：此次更新因使用率低移除了*“建议操作”*，但开发者承诺将重新引入设计更佳的类似功能。计划包括根据用户反馈增强*“引用”*和*“回答准确性”*。  
- **多语言 Prompt 引发广泛参与**：用户尝试了巴西葡萄牙语和孟加拉语查询，发现明确*告知* NotebookLM 语言语境会使交互更流畅。这展示了其包容性全球通信的能力。  
- **控制播客长度依然难以实现**：即使在 Prompt 中指定了时间，最终输出往往仍会超出或忽略限制。大多数人依靠灵活的长度范围来在深度覆盖和听众参与度之间取得平衡。


---

# 第一部分：高层级 Discord 总结

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Codeium Extension AutoComplete Issues**：用户报告 VSCode 中的 **Codeium extension** 显示自动补全建议的时间极短，导致无法使用。回退到 **version 1.24.8** 可恢复功能。
   - 讨论了多种补救措施，重点是将版本回退作为潜在解决方案。
- **Windsurf Performance and Error Handling**：**Windsurf** 正在经历严重的性能滞后，实例加载时间超过 **10 分钟**，且频繁的错误消息中断了工作流。
   - 用户呼吁 Codeium 针对“代码消失”和 Cascade 功能失效等 Bug 提供更清晰的沟通。
- **Flex Credits Usage Concerns**：几位用户询问 **flex credits** 是否可以结转，并指出在服务中断期间积分被扣除的问题。
   - 用户对频繁的错误消息和服务停机对积分使用的影响表示担忧。
- **Connection Issues with Codeium Server**：成员们分享了连接 **Codeium server** 的困难，交流了经验并寻求帮助。
   - 建议提交 support tickets 以便进一步调查和潜在修复。
- **Prompting with o1 in AI Applications**：一位用户分享了一个关于 [o1 prompting 课程](https://www.deeplearning.ai/short-courses/reasoning-with-o1/) 的链接，该课程涵盖了其在编程和推理任务中的应用。
   - 另一位用户因课程内容的复杂性请求提供内容摘要。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 0.44.2 Update Stabilizes Editor**：**Cursor team** 在解决了 **v0.44** 中的 Bug 后，回退到了 [version 0.44.2](https://www.cursor.com/changelog)，从而增强了稳定性。
   - 用户强调了新功能，如 **terminal** 以及各种提升整体体验的 **bug fixes**。
- **PyQt/PySide6 Setup Hits Snags**：开发者在设置 **PySide6** 时遇到了缺失 '**QtWebEngineCore.dll**' 等文件的问题，导致应用程序运行失败。
   - 建议包括验证正确的 **Python version** 并遵循详细的 **installation steps** 来解决问题。
- **O1 Pro Boosts Bug Fix Efficiency**：**O1 Pro** 用户报告称，与早期版本相比，使用更少的 Prompt 即可成功实现 **bug resolutions**。
   - 尽管增加了 **cost**，许多人发现 **O1 Pro's performance** 对他们的工作流非常有益。
- **Kepler Browser Focuses on Privacy**：**Kepler Community** 浏览器的开发重点在于 **privacy** 和 **lightweight** 功能。
   - 开发者正在鼓励 **open-source collaboration**，邀请大家为增强用户隐私功能做出贡献。
- **Cursor's Copy-Paste Functionality Frustrates**：用户报告 **Cursor's copy-paste** 有时会将终端文本粘贴为纯文本而非代码。
   - 建议包括使用 **Ctrl + Shift + V** 并正确定位 **terminal outputs** 以提高易用性。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **o1 API 访问争议**：讨论强调了 **Tier 5 订阅者**对访问 **o1 API** 的不满，主要担忧集中在 **每百万 token 15 美元**的定价与 **200 美元 o1 pro 订阅**之间的差异。
   - 成员们就定价结构的合理性展开了辩论，指出虽然有些人认为这很合理，但另一些人认为对于他们的使用场景来说，价格昂贵得令人望而却步。
- **Aider vs. Sonnet 性能对比**：**Aider** 的最新更新在有效性上已超越 **Sonnet**，基准测试得分达到 **84.2**，与 Sonnet 的表现相当。
   - 用户观察到，虽然 **Aider** 在 **editor mode** 下表现出色，但 **Gemini** 模型在处理 JavaScript 任务时遇到困难，导致在某些编程场景中用户更倾向于选择 Aider。
- **即将推出的模型：Veo 2 和 R1**：成员们对 **Veo 2** 和 **R1** 的发布充满期待，讨论了这些模型在日益激烈的竞争中可能如何影响 **OpenAI 的市场地位**。
   - 对话表明，新模型的引入可能会使 **Sora** 等现有模型竞争力下降，引发了关于其持续有效性的辩论。
- **Gemini 2.0 Google Search 集成**：**Vertex AI** 上的 **Gemini 2.0 Flash Experimental** 模型现在支持 **Google Search grounding**，通过最近 [GitHub pull request](https://github.com/BerriAI/litellm/pull/7257) 中详细说明的特定配置启用。
   - 这一集成增强了模型执行接地搜索的能力，与 **Gemini** 功能的最新进展保持一致。
- **Depth AI 代码库理解**：**Depth AI** 能够生成代码库的全面**知识图谱 (knowledge graph)**，在回答技术查询方面达到了 **99% 的准确率**，给用户留下了深刻印象。
   - 虽然设置很简单，但索引 **200k 到 150 万 token** 的较大型项目可能需要相当长的时间，一位用户报告称索引一个 180k token 的仓库耗时 **40 分钟**。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 的 12 天更新活动**：OpenAI 正在庆祝 **12 Days of OpenAI**，鼓励成员在 <#customize> 中锁定角色以保持关注并参与庆典。该计划旨在让社区参与到持续的**更新**和活动中。
   - 在 **第 10 天**，一个链接的 [YouTube video](https://www.youtube.com/watch?v=LWa6OHeNK3s) 展示了当天的庆祝活动，促使成员们探索与活动相关的精彩**内容**。
- **OpenAI vs Google：AI 进展**：**ai-discussions** 频道引发了关于 **OpenAI** 和 **Google** 在 AI 领域竞争进展的辩论，许多成员断言 **Google** 目前在 AI 开发方面正超越 **OpenAI**。有人担心 **OpenAI** 可能会为了战略利益而**限制模型发布**。
   - 参与者推测，**Google** 迅速的创新轨迹可能会显著塑造未来的 **AI 格局**，影响技术的演进和采用方式。
- **DALL·E vs Midjourney：图像生成大对决**：成员们将 **OpenAI** 的 **DALL·E** 与 **Midjourney** 以及 **Google 的 Imagen** 进行了对比，尽管 **DALL·E** 免费开放，但常因其明显的“AI 生成”痕迹而受到批评。讨论强调了 **Midjourney** 的**定价**和卓越的**制作质量**是关键因素。
   - 用户对 **DALL·E** 的局限性表示沮丧，同时承认 **Midjourney** 的优势，反映出用户即使付费也更倾向于高质量的图像生成模型。
- **自定义 GPTs 功能**：在 **gpt-4-discussions** 频道中，成员们质疑通过指令 *'you are now a manager to train me'* 来提示 ChatGPT 的有效性，旨在提高回答质量。
   - 此外，用户对无法**编辑自定义 GPTs** 表示不满，引发了对用户自定义选项受限的担忧。
- **频道发帖礼仪执行**：**prompt-engineering** 和 **api-discussions** 频道的讨论集中在执行**频道发帖礼仪**上，成员们批评他人在多个频道重复发帖的 **spam** 行为，并建议从错误的频道中删除消息。
   - 成员们还强调了在寻找帮助时识别合适频道的挑战，强调了遵守特定**指南**以维持秩序和简化讨论的重要性。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Falcon 模型表现出潜力**：**Falcon3** 模型，特别是 **7B** 和 **10B** 变体，展现出强劲的性能。最近的 [更新](https://huggingface.co/blog/falcon3) 引入了 tool-use 支持，增强了它们处理复杂交互的能力。
   - 工程师们热衷于在各种应用中测试这些模型，并注意到更新后功能的提升。
- **创新的 Prompt Chaining 策略**：**Prompt chaining** 正被用于通过多个模型顺序处理响应来优化模型输出。**Structured output** 和 **tree structures** 等技术正在被探索，以增强故事创作等创意任务。
   - 这些策略旨在迭代提高响应质量，正如 [Langflow 文档](https://docs.langflow.org/) 中所讨论的。
- **OpenAI 的安全实践受到审查**：人们对 **OpenAI 的安全协议** 表示担忧，尤其是在一次演示揭示了其模型在 **GPT-4o vs o1 preview** 对比中的 jailbreak 情况后。这引发了关于 OpenAI 的安全声明与实际模型漏洞之间一致性的辩论。
   - 讨论强调了对更透明的安全评估的需求，正如 [Democratize Intelligence 的推文](https://x.com/demi_network/status/1869085748852662718) 所引用的。
- **探索本地模型上的 Function Calling**：关于在**小型本地模型上进行 function calling** 的最佳库和方法的咨询，表明了对优化本地 AI 性能的关注。这种兴趣指向了在不依赖外部 API 的情况下提高模型效率的持续努力。
   - 对话强调了合适的库对于有效的本地模型部署的重要性。
- **确保 LLM 输出的一致性**：讨论集中在 **LLM 输出的一致性**上，特别是对于长文本和超长文本生成。成员们正在寻求解决这些在维持长篇输出质量方面挑战的顶级论文推荐。
   - 这种兴趣反映了工程界对于在广泛应用中维持模型可靠性的普遍关注。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 中的 3 面板 UI 更改**：新的 **3 面板 UI** 移除了 NotebookLM 中的 **'suggested actions'** 功能，解决了因发现率和功能有限导致的使用率低的问题。
   - 开发团队计划重新引入具有改进设计的类似功能，重点是增强 **citations** 和**响应准确性**，并鼓励用户为即将发布的版本提供反馈。
- **多语言功能增强**：成员们正在利用 NotebookLM 的交互功能来促进**巴西葡萄牙语**和**孟加拉语**等语言的对话，通过多语言提示词提高参与度。
   - 一位用户强调，在提示词中表达多语言能力可以简化讨论，促进工具内更具包容性和多样性的交互。
- **交互模式推广挑战**：NotebookLM 中 **interactive mode** 的推广正经历延迟和访问不一致的问题，一些用户面临音频生成滞后和意外重置等问题。
   - 反馈表明需要更可靠的部署策略，以确保所有使用新 UI 的用户都能无缝访问交互功能。
- **播客长度自定义策略**：用户正在探索控制**播客剧集长度**的模板，旨在保持深度内容探索的同时不牺牲引人入胜的对话。
   - 讨论显示出对灵活时间范围而非固定时长的偏好，突显了实现精确播客长度控制的复杂性。
- **使用 NotebookLM 生成知识库**：成员们正在调查 NotebookLM 生成类似于**检索增强生成 (RAG)** 的**知识库**的能力，寻求见解和替代方案。
   - 一段分享的 [YouTube 视频](https://youtu.be/NXdUMyZPUi4) 演示了将 NotebookLM 用作知识库，符合用户对结构化信息检索的需求。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **使用 4-bit 转换微调 Llama 3.2**：一名成员正在探索如何通过添加数据集有效地微调 **Llama 3.2** 模型，并讨论了加载先前 **checkpoints** 的选项。另一名成员强调，对于非 Unsloth 上传的模型，设置 **load_in_4bit=true** 允许自动转换。
   - 这种方法旨在在管理资源限制的同时增强模型性能，详情见 [Unsloth Tutorial](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama)。
- **优化 Batch Size 和 VRAM 管理**：关于最佳 **batch size** 的讨论表明，较大的尺寸可能会提高训练稳定性和准确性，但需要更多 **VRAM**。成员们一致认为，对于 **VRAM** 有限的用户，增加 **gradient accumulation** 是一个可行的替代方案。
   - 这种平衡对于高效的训练工作流至关重要，确保模型性能和资源利用率都达到最大化。
- **关于 QwQ 等开源推理模型的辩论**：成员们辩论了 **QwQ** 等开源推理模型的有效性，指出虽然复现推理过程很直接，但创建一个成功的模型仍然具有挑战性。有人对当前模型设计中**强化学习 (RL)** 的必要性表示怀疑。
   - 有建议认为，使用高质量数据集进行纯**监督微调 (SFT)** 可能就足够了，这可能会简化模型开发过程。
- **Unsloth 中的多 GPU 和 Mac 支持**：**Unsloth Pro** 现在支持**多 GPU** 设置，增强了本地和云环境的模型训练体验。然而，Mac 上的 **M4 MAX GPUs** 支持仍不可用，预计时间表在 **2025 年第二季度**左右。
   - 鼓励社区贡献以加快 Mac 支持，解决没有 **NVIDIA** 硬件的用户所面临的限制。
- **DiLoCo 研究与分布式训练技术**：一名成员分享了他们关于 **DiLoCo**（Distributed Low-Communication Training of Language Models）的研究，并向小组展示了他们的发现。这引起了兴趣，并鼓励更广泛的传播以获取更多反馈。
   - 参考了 [DiLoCo Presentation](https://docs.google.com/presentation/d/18Twuq0q1H75BxUOgRc8ZTs2lWGvE2XtnAGZ7CWtq3cA/edit?usp=sharing) 和相关的 [ArXiv papers](http://arxiv.org/abs/2311.08105)，以深入了解分布式训练方法。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI o1 模型推出增强功能**：新的 OpenAI **o1 模型**现已上线，接替了 [o1-preview](https://openrouter.ai/openai/o1-preview)，具有 **function calling** 和降低延迟等功能。
   - 它引入了一个新的 `reasoning_effort` API 参数，用于控制模型在回答前的思考时间，增强了用户交互性。
- **结构化输出归一化范围扩大**：OpenRouter 现在为 **8 家公司**的 **46 个模型**实现了**结构化输出 (structured outputs)** 归一化，简化了结果格式化。
   - 分享了一个[教程](https://x.com/OpenRouterAI/status/1869077909438091485)来演示其实际用法。
- **EVA Llama 作为故事讲述专家发布**：**EVA Llama** 模型已发布，专注于角色扮演和故事讲述，同时还更新了 **Grok 2** 和 **Cohere** 模型。
   - 有关 **EVA Llama** 的详细信息可以在[这里](https://openrouter.ai/eva-unit-01/eva-llama-3.33-70b)查看。
- **热门模型大幅降价**：**MythoMax 13B** 降价 **12.5%**，而 **QwQ 推理模型**降价 **55%**，提高了性价比。
   - 这些降价旨在让社区更容易使用这些模型。
- **OpenRouter 推出供应商页面分析**：供应商页面现在提供详细的分析数据，用户可以通过点击供应商名称查看模型托管图表。
   - 可以通过 [DeepInfra 供应商页面](https://openrouter.ai/provider/deepinfra) 查看示例，提供全面的洞察。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **辩论 Warmup 阶段公式**：讨论集中在 Kevin 用于近似 **warmup phase** 的公式 *(1 - beta1^step)*，并强调了当前 **LR schedulers** 缺乏支持的问题。
   - 成员们分享了他们的 [实现方案](https://github.com/EleutherAI/gpt-neox/blob/f5325805678c2b9e35aae4528283e0132c5f5bbc/megatron/logging.py#L352-L361)，并对使用 **lambdaLR** 时可能出现的 off-by-one 错误表示担忧。
- **利用 Meta-Learning 缓解 Overfitting**：社区探讨了 **Meta-Learning** 策略是否能有效减少监督学习模型中的 overfitting，并寻求具体的应用案例。
   - 虽然存在支持该方法的理论框架，但参与者指出在当前模型中缺乏实际落地。
- **神经网络压缩的进展**：成员们深入研究了 **compression methods**，如 depthwise compression 以及集成稀疏和低秩矩阵的 **OATS** 等剪枝技术。
   - 参与者对潜在的性能下降和数据覆盖损失表示担忧，特别是对于在记忆任务上训练的模型。
- **探索 AI 中的 Grokking 现象**：**grokking** 现象是讨论的焦点，涉及其重要性以及目前缺乏在 AI 模型中诱导该现象的有效方法。
   - 参与者表示，虽然 grokking 已得到认可，但大多数研究工作仍集中在 **large language models** 上，限制了更广泛的探索。
- **质疑 Koopman 算子理论的集成**：对于 **Koopman operator theory** 在神经网络中的适用性存在怀疑，质疑将神经层建模为动力系统的益处。
   - 批评者认为，该理论主要是在重新表述残差连接的使用，而没有引入实质性的创新。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **高效 Lora 训练**：一位用户分享了创建 **Lora** 的实际步骤：从强大的数据集开始，选择合适的模型，训练 Lora，然后进行测试。他们强调了研究如何创建高质量数据集以获得最佳结果。
   - 该用户强调了数据集质量的重要性，指出深入研究对于实现最佳训练效果至关重要。
- **首选的 Stable Diffusion 模型**：用户讨论了他们偏好的 **Stable Diffusion** 模型，一些人青睐 'flux' 模型，而另一些人则因易用性推荐 'InvokeAI'。
   - 大家一致认为必须配备 NVIDIA GPU，并建议使用带有 16GB 显存的 3060 以获得更流畅的性能。
- **在 Ubuntu 上运行 SD 的挑战**：用户表达了在 **Ubuntu** 上运行 **SDXL** 的挫败感，理由是与 **ComfyUI** 和 **Forge UI** 存在兼容性问题。
   - 有效运行 **SDXL** 可能需要对 **Ubuntu** 系统有深入的了解，以应对这些兼容性挑战。
- **生成图像的最佳分辨率**：一位初学者询问了生成的最佳图像分辨率，寻求在质量和处理时间之间取得平衡。
   - 建议包括尝试 **1024x1024** 左右的分辨率，并利用 **hires.fix** 来增强输出质量。
- **AI 生成内容指标**：讨论了模型训练中使用的技术和指标，特别是 **Pony** 模型及其评分系统。
   - 用户注意到这种独特的方法如何影响图像生成并影响社区认知。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **自定义网页源增强 Perplexity**：Perplexity 现在在 **Perplexity Spaces** 中提供 [自定义网页源](https://www.perplexity.ai/spaces)，以便针对特定用例定制搜索查询。
   - [发布视频](https://cdn.discordapp.com/attachments/1047204950763122820/1318746209778929674/Custom_web_sources_launch_video_-_v6.mp4) 展示了新的自定义功能。
- **Perplexity Pro 订阅发布**：[Perplexity Pro 订阅](https://perplexity.supply/shop/perplexity-subscription) 现已上线，提供 1 到 12 个月的礼品选项，可访问 **3 倍以上的来源** 和 **最新的 AI 模型**。
   - 用户正在利用这些订阅来增强搜索能力，并紧跟最新的 AI 发展。
- **AI 模型性能备受关注**：社区成员正在评估 **Perplexity Pro** 中 **AI 模型** 的性能，试图提高搜索质量，并建议使用 **Claude 3.5 Sonnet** 等替代方案。
   - 针对 **GPT-4o** 等模型声称的进步提出了疑问，引发了关于选择最佳架构的讨论。
- **Meta 旨在阻止 OpenAI 的营利性尝试**：**Meta** 表示有意阻止 **OpenAI** 追求 **营利性商业模式**，这可能会显著影响行业内 **未来的 AI 发展**。
   - 此举引发了关于市场竞争和 AI 创新动态潜在重塑的辩论。
- **用户面临 Perplexity 的速率限制**：多位用户报告在使用 **Perplexity** 时遇到 **Rate Limits**，引发了关于个性化速率限制增强必要性的讨论。
   - 有推测认为更高级别的订阅层级在缓解这些限制方面具有优势，用户分享了他们的相关经验。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 内存复制问题**：一位成员报告称，从代码中移除 **IsDense(y) && IsSame(x, y)** 条件可以解决 LLM 模型推理过程中的异常行为，并强调 **CudaCopy** 会启动 CUDA kernels。更多详情请参考 [使用 CUDA graphs 时减少到第一个 kernel 的时间](https://discuss.pytorch.org/t/reduce-time-to-first-kernel-when-using-cuda-graphs/214310)。
   - 讨论还涉及 **CUDA graphs** 缺乏支持 **cudaMemcpyAsync** 的官方文档，引发了对在 CUDA 实现中处理异步内存操作的担忧。
- **Megatron-LM 的训练效率**：随着成员计划在分布式设置中提高 **训练吞吐量**，**Megatron-LM** 的效率仍处于审查之中。建议参考 **Gensyn** 和 Christine Yip 活跃社区的见解来优化分布式训练。
   - 对话强调了利用社区资源解决扩展性挑战并提高 **Megatron-LM** 整体训练性能的重要性。
- **自定义视觉编码器集成**：一位成员提议开发 **自定义视觉编码器**，以更好地处理现有语言模型中的小像素级图像，认为编码器配对的灵活性优于预训练 VLMs 的优势。
   - 讨论了将该编码器与各种 **LLMs** 集成的潜力，强调了在专门图像处理任务中的适应性和性能提升。
- **RTX 3090 微调实验**：分享了使用 **RTX 3090** 进行微调的实验，并讨论了采用 **bf16** 或 **QLora+int8** 精度的最佳设置。来自 **WandB** 的示例确认 **8bit Lora** 对于该 GPU 上的 **8B models** 是有效的。
   - 成员们探索了计算效率与模型性能之间的平衡，旨在确定在消费级硬件上进行大规模模型微调的最佳实践。
- **Axolotl Lora 配置成功**：[用于 llama-3-vision 的 Axolotl Lora 配置](https://github.com/axolotl-ai-cloud/axolotl/blob/effc4dc4097af212432c9ebaba7eb9677d768467/examples/llama-3-vision/lora-11b.yaml) 已验证可在 **2x A6000 GPUs** 上无缝运行，展示了在多 GPU 环境中的可靠性能。
   - 目前人们对寻求计算赞助以促进更大规模的实验持续关注，这取决于初始配置的成功。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 设置与兼容性**：用户分享了他们的 **LM Studio** 配置，包括 [RTX 4060 笔记本](https://lmstudio.ai/beta-releases) 和配备 **96GB RAM** 的 **M3 Max**，突显了该应用的多功能性。
   - 一名用户在 LM Studio 中加载 [Llama 3.2 11B Vision](https://huggingface.co/mlx-community/Llama-3.2-11B-Vision-Instruct-4bit) 时遇到了 'unknown model architecture' 错误。
- **Qwen QwQ 在 Roleplay 应用中表现出色**：讨论推荐 **Qwen QwQ** 作为 Roleplay LLM 任务的强力候选者，多位用户对其表现表示赞赏。
   - 一位成员指出 **Qwen2** 在 Python 编程场景中表现出卓越的性能。
- **AMD GPU 驱动导致 Llama 性能下降**：用户报告称，使用 **24.12.1 驱动** 的 **AMD GPU** 遇到了 'Safetensors header is unexpectedly large' 错误，导致有人回退到 **24.10.1**。
   - **Llama 3.2 3B 模型** 的性能从 **24.10.1** 驱动下的 **90+ tok/s** 下降到新驱动下的 **20 tok/s**。
- **LM Studio 缺乏移动端支持**：一位成员表达了在移动设备上使用 **LM Studio** 的需求，但发现目前没有可用的移动端 App。
   - 虽然有人建议了替代方案，但直接的移动端兼容性仍不可用。
- **大模型推理需要高 RAM**：用户讨论指出，运行 **70B 模型** 需要 **70GB** 的 VRAM 或主内存。
   - 建议在以 **q8** 精度运行时，额外准备 **10-20% 的 VRAM** 以保证操作灵活性。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **无缝切换：从 Firebase 迁移到 Supabase**：**#[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1318837235692474419)** 频道的一位用户正在寻求将整个网站从 **Firebase** 迁移到 **Supabase** 的最佳策略，强调了对全面迁移实践的需求。
   - 社区正在积极分享策略和最佳实践，以确保数据完整性并最大限度地减少迁移过程中的停机时间。
- **Bootstrap 与 create-mf-app 的博弈**：一位成员在 **#[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1318837235692474419)** 中讨论了将 **create-mf-app** 与 **Bootstrap** 集成时面临的挑战，指出与 **Tailwind** 的冲突会导致设置不稳定。
   - 提出的解决方案包括标准化的集成方法，以便在不损害项目稳定性的情况下协调使用这两个框架。
- **Bolt Pilot 寻求测试者**：在 **#[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1318837235692474419)** 中，一位成员介绍了 **Bolt Pilot**（一个为 **Bolt** 设计的新 GPT），并请求社区测试其功能以进行改进。
   - 早期测试者的反馈对于在广泛发布前优化 **Bolt Pilot** 的性能和功能集至关重要。
- **Bolt 的 Token 消耗令用户沮丧**：在 **#[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1318669185370296461)** 中，许多用户对 **Bolt** 过度的 Token 消耗表示不满，并提出了诸如添加 **'punch the AI'** 按钮来减少浪费等建议。
   - 成员们分享了收到无关回复的经历，引发了关于优化 Token 分配以提高效率的讨论。
- **为 Bolt 增强支付集成功能**：在 **#[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1318669185370296461)** 中有一场关于在 **Bolt** 中实现 **Stripe** 和 **PayPal** 等支付集成复杂性的对话。
   - 用户强调了动态计费功能的必要性，并对未来支持这些集成的更新表示关注。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Toolkit 部署问题**：一名成员根据 AWS 指南部署了 **Cohere Toolkit**，但遇到了间歇性的 `stream ended unexpectedly` 错误。
   - 另一名成员建议检查 **docker logs** 以诊断问题，并指出在应用日志中可能会发现更深入的见解。
- **Findr 应用在 Product Hunt 上线**：**Findr** 正式在 [Product Hunt](https://www.producthunt.com/posts/findr-remember-everything) 上线，旨在为人类提供**无限记忆**和**可搜索的数字大脑**。
   - 该团队正通过其宣传 [推文](https://x.com/Nish306/status/1868953328975261712) 寻求支持，并收到了来自社区的**积极反馈**。
- **Multimodal Embed-v3 速率限制提升**：响应社区反馈，针对生产密钥（production keys），**Multimodal Image Embed** 端点的速率限制从 **40 images/min** 提高到了 **400 images/min**。
   - 测试版（Trial）速率限制仍保持在 **5 images/min**，而像 **Chat** 这样的其他端点有其特定的速率限制，详见 [API Keys and Rate Limits — Cohere](https://docs.cohere.com/v2/docs/rate-limits) 文档。
- **Cohere Reranker 性能**：一位开发者报告称，配合 **ContextualCompressionRetriever** 使用的 **Cohere Reranker** 有时无法选择最相关的分块（chunks），导致回答错误。
   - 尽管其 RAG 应用中的分块非常准确，但 **reranking 行为** 显得随机，给用户带来了困惑。
- **嵌入模型维度挑战**：一位用户询问关于为来自 **text-3-embedding-large (3072 维度)** 和 **Cohere Embed v3 (1024 维度)** 的嵌入创建独立向量存储（vector stores）的问题。
   - 在整合文本、表格和图像的嵌入时，维度差异可能会影响存储策略。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Archcraft 上的 Mojo REPL 故障**：一位用户报告在 [Archcraft Linux](https://discord.com/channels/1087530497313357884/1098713601386233997/1318669970682679397) 上进入 **Mojo REPL** 时出现问题，提示缺少 **mojo-ldd** 库。
   - 社区讨论了与 **mojo-lld** 相关的潜在链接器错误以及解决该问题所需的安装步骤。
- **Mojo 文档中关于 Var 关键字的辩论**：[Mojo 文档](https://docs.modular.com/mojo/manual/basics#variables) 的更新引发了关于变量声明中 `var` 关键字必要性的辩论。
   - 成员们建议让 `var` 变为可选，同时讨论了它对 struct 定义和代码清晰度的影响。
- **澄清 Mojo Kernel 术语**：Mojo 中的“kernel”一词被澄清为是指在加速器（accelerators）上运行的函数，而非传统的 OS kernels。
   - 讨论强调了针对硬件的代码块优化，以及计算 kernel（compute kernels）与 OS kernel 之间的区别。
- **Max 中的 Custom Ops 加载问题**：在 [Max](https://github.com/modularml/max/issues/269) 中加载 **mandelbrot** 自定义算子（custom op）时报告了问题，特别是与未注册的 Mojo kernels 相关。
   - 成员们指出需要对自定义算子进行正确注册，以确保在 Mojo 中顺利执行。
- **自定义算子处理增强**：有人提交了一项 [功能请求](https://github.com/modularml/max/issues/269)，旨在改进 Max 中缺失自定义算子时的错误消息和处理机制。
   - 这包括在发生错误时引导用户查看相关文档，从而提升整体用户体验。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 的持续性问题**：多名用户报告了 **Open Interpreter** 的持续性问题，特别是与 `--conversations` 命令相关的错误，导致丢失了宝贵的对话。
   - 成员们正在积极寻求这些持续性错误的解决方案，强调了对可靠的 **conversation management**（对话管理）的需求。
- **升级到 Open Interpreter 1.x**：一位用户询问了从 **Open Interpreter 0.34** 升级到最新的 **1.x 版本** 的事宜，引发了关于新版本中 OS mode 可用性的讨论。
   - 成员们策划了潜在的改进方案，并分享了对 **Open Interpreter 1.0** 预期新功能的见解。
- **创新 AI 应用与模型**：讨论集中在利用 **AI** 进行 **Raspberry Pi** 设置等项目，以及集成语音转语音模型用于 **home automation**。
   - 用户探索了将较小模型与较大系统连接的方法，以增强整体 **functionality**（功能性）。
- **Truffle-1：新的 AI 动力源**：一名成员介绍了 **Truffle-1**，这是一个能够运行多个模型的个人计算堆栈，配备 **64GB unified memory**，押金为 500 美元，每月 115 美元。更多详情请访问 [Truffle 网站](https://itsalltruffles.com)。
   - **Truffle-1** 承诺无限的推理时间，并支持编写和分享应用，设备定于 **1月** 发货。
- **在本地使用 Open Interpreter 的 OS Mode**：一位用户询问了在本地使用 **Open Interpreter** 的 **OS mode** 的可行性，这引发了关于可用配置选项的讨论。
   - 成员们分享了配置技巧，以帮助在本地 **OS mode** 设置中遇到问题的用户。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **基准测试对决：TinyGrad OpenCL vs PyTorch CUDA**：一名成员请求提供 [benchmarks](https://github.com/tinygrad/tinygrad/issues/8194)，对比 **TinyGrad** 的 **OpenCL** 实现与 **PyTorch** 的 **CUDA** 在各种 **Llama models** 上的表现。
   - 这突显了社区内对不同 AI 框架之间 **performance comparisons**（性能比较）的持续关注。
- **可合并形状：解决 ShapeTracker 的复杂性**：关于在 Lean 中证明两个任意 **ShapeTrackers** 的 **mergeability**（可合并性）的复杂性展开了讨论，一名用户表示不可能有一个像矩阵行列式那样简单的判定标准。
   - 他们强调了步长 (strides) 和形状中存在的巧合，这使得 **mergeability checks** 变得复杂。
- **CuTe 中的布局代数揭秘**：成员们询问 **mergeability** 是否等同于 **CuTe's layout algebra** 中的复合 (composition)，并引用了 [关于 CuTe Layouts 代数的笔记](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/)。
   - 讨论涉及了 NVIDIA **CUTLASS** 库中的基本抽象以及布局操作的数学处理。
- **布局单射性中的 NP-Hard 挑战**：有人对证明与 **layout algebra** 中的 **injectivity**（单射性）相关的条件表示担忧，并建议此类检查可能是 **NP hard**。
   - 参与者强调了由于潜在的步长干扰，在布局代数中建立充分条件的困难。
- **符号优势：函数 vs 布局**：一名成员指出，在检查必要性和充分性方面，**symbolic integer functions**（符号整数函数）比 **layouts** 具有更强大的能力。
   - 这与关于合并视图的 **algorithm complexities**（算法复杂度）的讨论一致，并支持正在进行的 **research directions**（研究方向）。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **FSDP 归一化缩放**：讨论显示必须解决 **FSDP 按 `world_size` 进行的归一化**问题，通过 `world_size` 进行缩放可以修正平均操作的问题。
   - 一位成员建议提交 [PR #2172](https://github.com/pytorch/torchtune/pull/2172) 来实现此修复，重点关注 `scale_grads` 函数。
- **训练中的显式缩放**：社区强调了在训练配方（training recipe）中进行**显式损失缩放**的重要性，而不是将逻辑隐藏在别处，以简化理解。
   - 经过评估，成员们同意在训练和优化钩子（optimization hooks）中进一步明确缩放过程。
- **跨框架的 Bug 识别**：识别出影响 `1/world_size` 因子缩减的类似 Bug 可能存在于多个库中，包括 `trl` 和 Hugging Face 的 trainer。
   - 成员们赞扬了 **Hugging Face** 团队在其训练框架中识别并解决这些问题，详见相关的 [GitHub issues](https://github.com/huggingface/transformers/issues/34242)。
- **处理 Hugging Face 中的 No Sync**：成员们讨论了 **Hugging Face** 如何通过在正确计算损失的同时避免梯度累积归一化来处理 no sync 场景。
   - 具体实现细节可在 [trainer.py](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3662) 文件中找到。
- **机器学习中的进化算法**：**进化算法（Evolutionary algorithms）**在机器学习讨论中越来越受到关注，凸显了其潜在的应用价值。
   - 一位成员指出了它们的重要性，并建议在社区内进一步探索其使用案例。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **AI 重塑知识经济**：[AI and Knowledge Economy](https://arxiv.org/abs/2312.05481) 介绍了一个框架，分析 **AI** 如何通过在“工人”和“解决者”之间重新分配角色来转型知识经济。基础自主 AI 取代人类，而高级自主 AI 则使规模更大、生产力更高的公司受益。
   - 随着自主 Agent 获得关注，它们主要使知识最渊博的人受益，使其能够高效管理常规工作；而知识较少的人则从聊天机器人等非自主 AI 中受益。
- **Coconut - 连续思维范式**：来自 Meta 的论文 [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/html/2412.06769v1) 提出了 **Coconut**，这是一种新的推理范式，它使用 LLM 的最后一个隐藏状态（last hidden state）进行推理，而不是传统的语言空间。
   - 该方法旨在通过探索不受限制的潜空间（latent spaces）来克服基于语言推理的局限性，从而可能增强 LLM 在复杂推理任务上的性能。
- **TypedReAct 之谜已解决**：一位成员分享了 **TypedReAct** 的新实现，询问是否提交 PR，但指出在未来版本中 **TypedChainOfThought** 可能存在弃用问题。
   - 另一位成员建议移除“Typed”前缀将解决兼容性问题，并强调内置的 ReAct 在没有类型定义的情况下也非常有效。
- **RouteLLM 维护担忧**：一位成员对 **RouteLLM** 缺乏维护表示担忧，表示对潜在的 **DSPy** 集成感兴趣。
   - 对话强调了支持缺乏监管的模型开发的重要性。
- **DSPy 随推理模型的发展**：一位成员询问 **DSPy** 将如何随着推理模型的兴起而演进，强调在分支层面进行微调。
   - 这一观点将重点从传统的 Prompting 转向过程奖励机制（process reward mechanisms），预示着模型训练可能发生范式转移。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 在 Jinja 模板方面遇到困难**：用户报告称 **GPT4All** 在 **Jinja templates** 上遇到了严重问题，而这对于模型功能至关重要。目前的问题包括空格错误、换行错误以及不支持 'none' 和 '[1:]' 等函数。
   - 解决这些模板问题的努力正在进行中，但尚未实施详细的解决方案。
- **对 GPT4All Docker 部署的需求**：有用户请求提供带有 Web UI 的 **GPT4All Docker 版本**，旨在简化部署流程。
   - 截至目前，社区尚未提供满足该需求的特定资源或现有解决方案。
- **通过 CLI 访问 GPT4All 中的本地文档**：用户在使用 **GPT4All CLI** 访问本地文档时遇到困难，因为旧版 CLI 已不再正式支持该功能。
   - 然而，有人指出，如果在 GUI 中启用，**server API** 允许以编程方式访问本地文档。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AI SDR 使用 LlamaIndex 自动化线索生成**：一个使用 **LlamaIndex** 构建的 [Agentic AI SDR](https://t.co/tczv5ZDI4H) 展示了其在自动化线索生成方面的能力，并链接到了多个 [GitHub features](https://github.com/features)。
   - 该工具强调了 **LlamaIndex** 的集成能力，提高了线索生成工作流的效率。
- **速成课程教授使用 LlamaIndex 构建 Agent**：由 **LlamaIndex** 主导的[速成课程](https://twitter.com/llama_index/status/1869454248620269615)专注于构建具有 Function Calling 功能的 Agent，以管理实时数据查询。
   - 参与者还将学习创建能在向量工具和摘要工具之间智能路由的 **Agentic RAG**，以及如何实现 ReAct。
- **OpenAIAgent 面临并发执行限制**：一名成员报告称，即使在异步环境中进行了异步修改，`OpenAIAgent` 的函数执行仍然是非并发的。
   - 这凸显了 **OpenAIAgent** 执行模型中的一个局限性，影响了异步操作。
- **社区参与 RAG 评估策略讨论**：关于 **RAG evaluation** 的讨论非常活跃，一名成员邀请同行通过私信进行深入交流。
   - 参与者正在探索 AI 社区内有效的评估策略。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCL 排行榜功能失效**：一名用户报告称 [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard) 的函数调用演示卡在 “**Loading Model Response...**”。
   - 另一名成员确认是 **证书问题 (certificate issue)** 导致模型端点失效。
- **用于结构化输出的 Gorilla 基准测试**：一名用户询问如何使用 **Gorilla benchmark** 来评估模型的结构化输出，特别是关于根据提供的 **JSON schema** 或 **Pydantic model** 生成文本的子任务。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 频道的感谢**：一名成员在 [mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/) 频道中表达了谢意：*Thank you for that!*
   - 这种表达凸显了 **LLM Agents (Berkeley MOOC)** 讨论中的积极互动。
- **MOOC 讨论中的正面反馈**：在 [mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/) 中分享了一条感谢信息：*Thank you for that!*
   - 这些认可表明了该公会中 AI 工程师们的积极参与和满意度。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **新工程师加入负责强化学习**：一名新工程师将于 **1 月**加入，协助 **Reinforcement Learning** 相关工作。
   - 他们的专业知识将增强团队在 **Reinforcement Learning** 方面的能力，为正在进行的项目做出贡献。
- **增强对 KTO 项目的支持**：新工程师将从 **1 月**开始为 **kto** 项目提供支持。
   - 预计这一协助将对 **kto** 项目的开发产生积极影响。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Developer Hub 更新发布**：宣布了 **Developer Hub** 的重大更新，详细介绍了改进和新功能。您可以在[此处](https://discord.com/channels/1089876418936180786/1230938514955436242/1318638353503227935)查看完整公告。
   - 鼓励社区反馈以提升**用户体验**。
- **面向开源 AI 的 Blueprints 倡议**：**Blueprints 倡议**旨在协助开发者创建开源 AI 解决方案。更多详情请见[此讨论帖](https://discord.com/channels/1089876418936180786/1318689803021058158)。
   - 该倡议作为开发者的资源，帮助他们有效地启动项目。



---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。


---


**HuggingFace Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道分类的详细摘要和链接


{% if medium == 'web' %}




### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1318669267478253689)** (60 条消息🔥🔥): 

> `Codeium 扩展问题, Windsurf 性能问题, Flex Credits 担忧, 连接到 Codeium 服务器, 使用 o1 进行 Prompting` 


- **Codeium 扩展遭遇自动补全问题**：多位用户报告 VSCode 中的 **Codeium 扩展**显示的自动补全建议仅出现几分之一秒，导致无法使用。
   - 修复建议包括回退到 **1.24.8 版本**，该版本似乎可以恢复功能。
- **Windsurf 性能滞后**：用户对 **Windsurf** 变得极其缓慢或完全无法加载表示沮丧，一位用户等待了超过 **10 分钟**才打开。
   - 另一位用户报告了频繁的错误消息干扰了工作流，并寻求潜在的解决方案。
- **关于 Flex Credits 使用的担忧**：几位用户询问 **flex credits** 是否可以结转，因为频繁的错误消息和服务器停机影响了他们的使用。
   - 用户报告称，即使在遭遇服务停机时，积分仍被扣除。
- **与 Codeium 服务器的连接问题**：讨论集中在连接 **Codeium 服务器**的困难上，用户分享了他们的经历并请求协助解决该问题。
   - 有建议提出提交支持工单（support tickets）以便进一步调查和修复。
- **在 AI 应用中使用 o1 进行 Prompting**：一位用户分享了关于 **o1 prompting** 的链接，讨论了它如何有效地执行编码和推理任务，并敦促他人探索其能力。
   - 另一位用户因提供的信息过于复杂，请求对该课程内容进行总结。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.deeplearning.ai/short-courses/reasoning-with-o1/">Reasoning with o1</a>: 学习如何使用 OpenAI 的 o1 模型并为其编写提示词，以处理复杂的推理任务。</li><li><a href="https://tenor.com/view/hello-there-gif-5677380953331354485">Hello There GIF - Hello there - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1318669293239533671)** (678 条消息🔥🔥🔥): 

> `Windsurf vs Cursor, 模型性能对比, Windsurf 中的错误处理, 开发中的 AI 集成, 编程性能与工具` 


- **Windsurf vs Cursor**: 用户正在讨论 Windsurf 和 Cursor 之间的区别，强调 Cursor 的 20 美元方案通过无限请求等功能提供了更好的性价比，而 Windsurf 的定价更高且采用积分系统。
   - 一些用户倾向于同时保留两个选项进行对比，而另一些用户则因性价比而青睐 Cursor。
- **模型性能对比**: 讨论显示 Codeium 的 4o-mini 和 Haiku 模型通常被认为更高效且更具成本效益，同时还与其他模型（如 Llama 3.1 和 GPT）进行了对比。
   - 参与者提到 4o-mini 可以有效地执行类似任务，并且最近增加了接受图像的能力。
- **Windsurf 中的错误处理**: 用户报告了 Windsurf 的各种错误和 Bug，包括“代码消失”以及 Cascade 功能未按预期运行的问题。
   - 一些用户在文件操作期间遇到内部错误，并呼吁 Codeium 针对这些问题提供更清晰的沟通。
- **开发中的 AI 集成**: 参与者对包括 Copilot 和 Codeium 在内的各种 AI 工具如何集成到他们的编码工作流中表示关注，讨论了这些工具在自动补全和代码建议方面的有效性。
   - 对这些工具有效性的分析表明，大家普遍认为尝试不同模型以找到最适合的模型至关重要。
- **编程性能与工具**: 围绕使用 AI 编程的最佳实践的讨论反映出，用户需要明确何时在 Windsurf 等工具中使用 Chat 模式或 Write 模式。
   - 建议强调了使用绝对路径和为 Prompt 定义明确目标的重要性，以提高 AI 辅助编程的效率。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://livebench.ai/#/?Coding=a">LiveBench</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://codeium.canny.io/feature-requests/p/cannot-use-windsurf-as-git-editor">Cannot use windsurf as git editor | Feature Requests | Codeium</a>: git config --global core.editor &#x27;windsurf --wait &#x27; 在 rebase 时抛出错误提示：Waiting for your editor to close the file... [1119/144632.</li><li><a href="https://codeium.canny.io/feature-requests/p/windsurf-focus-follows-mouse-as-a-configuration-option">Windsurf - Focus Follows Mouse (as a configuration option) | Feature Requests | Codeium</a>: VSCode 有一个公开的 GitHub PR，表面上看已经有 4 年多了，但实际上比这还要久。</li><li><a href="https://www.youtube.com/watch?v=9jgR-Ih_wGs"> - YouTube</a>: 未找到描述</li><li><a href="https://www.ray.io">Productionizing and scaling Python ML workloads simply | Ray</a>: Ray 管理、执行并优化跨 AI 工作负载的计算需求。它统一了基础设施并支持任何 AI 工作负载。立即免费试用。</li><li><a href="https://www.reddit.com/r/ChatGPTCoding/s/vPI207Unh3">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/Codeium/s/v22PYhpKn1">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1318672809093763163)** (707 条消息🔥🔥🔥): 

> `Cursor 更新 0.44.2，Cursor 中的开发工具，PyQt 和 PySide6 问题，O1 Pro 使用，Kepler Community 浏览器` 


- **Cursor 更新 0.44.2 发布**：在修复了之前 0.44 版本的 Bug 后，Cursor 团队已回滚至 0.44.2 版本，用户反馈稳定性有所提升。
   - 用户讨论了此次更新的体验，包括终端（terminal）等新功能以及 Bug 修复。
- **PyQt 和 PySide6 的挑战**：用户在配置 PySide6 时遇到了诸如 'QtWebEngineCore.dll' 文件缺失的问题，导致应用程序出现故障。
   - 建议确保安装了正确的 Python 版本，并对安装步骤进行排查。
- **O1 Pro 增强功能**：用户讨论了使用 O1 Pro 的优势，报告称与早期版本相比，仅需极少量的 Prompt 即可成功解决 Bug。
   - 虽然提到了 O1 Pro 的成本，但一些用户认为尽管有额外支出，其性能表现物有所值。
- **Kepler Community 浏览器开发**：一位用户分享了开发 Kepler Community 浏览器的进展，强调其专注于隐私和轻量化功能。
   - 开发者表达了对开源协作的承诺，邀请他人为这个旨在增强用户隐私的项目做出贡献。
- **Cursor 的复制粘贴功能**：用户反馈了对 Cursor 处理终端复制文本方式的不满，有时文本会以纯文本而非代码形式粘贴。
   - 建议包括使用 Ctrl + Shift + V 进行粘贴，并有效针对终端输出以提高可用性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://download.todesktop.com/230313mzl4w4u92/cursor-0.44.0-build-2412187f9v0nffu-x86_64.AppImage">未找到标题</a>: 未找到描述</li><li><a href="https://www.cursor.com/settings">设置 | Cursor - AI 代码编辑器</a>: 你可以在此处管理你的账户、账单和团队设置。</li><li><a href="https://www.cursor.com/downloads">下载 | Cursor - AI 代码编辑器</a>: 选择你的平台以下载最新版本的 Cursor。</li><li><a href="https://python-poetry.org/">Poetry - 让 Python 依赖管理和打包变得简单</a>: 未找到描述</li><li><a href="https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-environment-manager">Python 环境管理器 - Visual Studio Marketplace</a>: Visual Studio Code 扩展 - 查看并管理 Python 环境和包。</li><li><a href="https://docs.astral.sh/uv/">uv</a>: 未找到描述</li><li><a href="https://github.com/ultrasev/cursor-reset">GitHub - ultrasev/cursor-reset: 用于重置 Cursor 编辑器设备识别系统的 Mac 工具。有助于解决账号限制和试用相关问题。</a>: 用于重置 Cursor 编辑器设备识别系统的 Mac 工具。有助于解决账号限制和试用相关问题。 - ultrasev/cursor-reset</li><li><a href="https://github.com/ZackPlauche/add-cursor-to-win-context-menu">GitHub - ZackPlauche/add-cursor-to-win-context-menu</a>: 通过在 GitHub 上创建账户来为 ZackPlauche/add-cursor-to-win-context-menu 的开发做出贡献。</li><li><a href="https://forum.cursor.com/t/warning-cursor-v0-44-breaks-all-devcontainers-v0-394-0/35747/7">警告：Cursor v0.44 破坏了所有 devcontainers v0.394.0</a>: 你是如何强制禁用 Cursor 更新的？我陷入了一个困境，每当重启 Cursor 时，它总是会自动更新到 v0.44.0。额外的问题是，即使我禁用了 “devcontainer” 扩展...</li><li><a href="https://tenor.com/view/danger-alert-siren-alarm-red-light-gif-16931369">危险警报 GIF - Danger Alert Siren - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>: 新的更新和改进。</li><li><a href="https://pastebin.com/Bgu7XD6C">index.html - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://pastebin.com/cqNdfphK">style.css - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://github.com/TheGalaxyStars/KEPLER-COMMUNITY">GitHub - TheGalaxyStars/KEPLER-COMMUNITY: 自由探索，不留痕迹。</a>: 自由探索，不留痕迹。通过在 GitHub 上创建账户来为 TheGalaxyStars/KEPLER-COMMUNITY 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1318696787279876159)** (264 条消息🔥🔥): 

> `o1 API 访问权限, Benchmark 性能, 退款与支持体验, Gemini vs. Sonnet, Aider 功能` 


- **关于 o1 API 访问权限和定价的争议**：讨论显示，关于 **o1 API** 访问权限的体验褒贬不一，一些用户表示尽管是 Tier 5 订阅者，但仍未获得访问权限，对此感到沮丧。
   - 一位成员提到了定价结构，强调 API 每 **100 万 tokens 15 美元**的价格与 **200 美元订阅**的 o1 pro 相比被认为偏高，但也有人认为这是合理的。
- **Aider 和 Sonnet 的性能比较**：用户比较了 **Aider** 和 **Sonnet** 的性能，报告称 Aider 的最新更新显示其更加有效，o1 的 Benchmark 达到了 **84.2**，与 Sonnet 旗鼓相当。
   - 其他人讨论了 o1 在 **editor 模式**下表现良好，而 Gemini 模型在处理 JavaScript 时比较吃力，这表明 Aider 在某些编码任务中表现更好。
- **订阅服务的退款流程**：几位成员分享了他们对 o1 pro 订阅**退款流程**的经验，指出回复可能会有延迟，但退款最终会到账。
   - 虽然有人报告退款等待时间较长，但也有人声称速度较快，特别是有一位成员在提交请求后几小时内就收到了退款。
- **对即将推出的模型的期待**：成员们表达了对 **Veo 2** 和 **R1** 等即将推出的模型的期待，指出竞争日益激烈，可能会影响 OpenAI 的市场地位。
   - 对话表明，随着新模型的推出，像 Sora 这样的现有模型可能会落后，从而引发了关于其有效性和性能的辩论。
- **Aider 改进的功能**：用户注意到 Aider 功能的改进，特别是讨论了无需手动 `/add` 即可**查看所有文件**的能力，并强调了对 `.aiderignore` 文件的潜在需求。
   - 一位成员讨论了使用 Aider 的 editor 功能的效率，特别是配合 Gemini 模型时，同时也对 JavaScript 的编辑限制表示了担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/AndrewYNg/status/1869421643925422166">来自 Andrew Ng (@AndrewYNg) 的推文</a>：OpenAI 昨天刚刚宣布了 o1（高级推理模型）的 API 访问权限。我很高兴今天宣布一门新的短课程《Reasoning with o1》，与 @OpenAI 合作开发，由 @colintjarvis 授课...</li><li><a href="https://x.com/CodeByPoonam/status/1869289412951220395">来自 Poonam Soni (@CodeByPoonam) 的推文</a>：Google 刚刚发布了 Veo 2，简直疯狂。剧透：OpenAI Sora 现在落后了。10 个展示其能力的疯狂示例：（不要错过第 5 个）</li><li><a href="https://aider.chat/docs/usage/lint-test.html">Linting 和测试</a>：自动修复 linting 和测试错误。</li><li><a href="https://openrouter.ai/openai/o1">o1 - API、提供商、统计数据</a>：OpenAI 最新且最强大的模型系列，o1 旨在在回答前花更多时间思考。o1 模型系列通过大规模强化学习进行训练，以使用...进行推理。</li><li><a href="https://aider.chat/docs/config/options.html">选项参考</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/config/options.html#fixing-and-committing">选项参考</a>：关于 aider 所有设置的详细信息。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1318706793655963689)** (18 条消息🔥): 

> `Aider 对 Gemini Flash 2 的支持、使用 /architect 和 /ask 模式、管理代码重构、文件上传问题、Gemini 2.0 中的 Google Search Grounding` 


- **Aider 不支持 Gemini Flash 2 的特殊功能**：一位成员询问了 Aider 是否支持 Gemini Flash 2 的 grounding 功能，但得到的澄清是 Aider 在 API 层面对此没有做特殊处理。
   - 另一位成员提到 **Gemini 模型支持 Google Search grounding**，但这涉及到特定的模型要求和定价。
- **使用 /architect 和 /ask 进行项目规划**：成员们讨论了如何有效地利用 /architect 和 /ask 模式来定义项目计划，进而通过 /code 模式进行实施。
   - 一位成员建议要求 Aider 创建一个 *todo.md* 文件来跟踪任务，从而增强工作流的组织性。
- **代码重构与任务管理的挑战**：一位成员表示，随着项目规模扩大，在使用 Claude 进行功能开发时，保持代码整洁变得困难。
   - 参与者分享道，如果没有监督，生成的代码可能会变得混乱，在开发过程中可能需要穿插重构步骤。
- **Aider 文件上传问题**：一位成员报告在尝试向 Aider 添加文件时没有出现文件下拉菜单，这引发了对新版本可用性的担忧。
   - 另一位用户确认该 bug 已在 main 分支中修复，并提供了更新说明。
- **Gemini 2.0 中 Google Search 的集成**：一位成员详细说明了 Vertex AI 上的 Gemini 2.0 Flash Experimental 模型支持通过特定配置启用 Google Search grounding。
   - 他们分享了一个相关的 GitHub pull request，该 PR 增强了对该功能的支持，并提供了设置所需的 YAML 配置。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/repomap.html">Repository map</a>: Aider 使用 git 仓库地图为 LLM 提供代码上下文。</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-t">FAQ</a>: 关于 aider 的常见问题解答。</li><li><a href="https://github.com/yamad">yamad - 概览</a>: yamad 有 85 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>: 关于 aider 的常见问题解答。</li><li><a href="https://github.com/yamadashy/repomix">GitHub - yamadashy/repomix: 📦 Repomix (原 Repopack) 是一个强大的工具，可将整个仓库打包成单个 AI 友好的文件。非常适合需要将代码库输入给 Large Language Models (LLMs) 或其他 AI 工具（如 Claude、ChatGPT 和 Gemini）的场景。</a></li><li><a href="https://github.com/BerriAI/litellm/pull/7257">Add support for Gemini 2.0 GoogleSearch tool by samling · Pull Request #7257 · BerriAI/litellm</a>: 标题：将 googleSearch() 工具添加到 Gemini/VertexAI 模型的有效工具列表中，以支持 Gemini 2.0 grounding。相关问题：增强了 #7188。类型：🆕 新功能。测试：已通过。更改：添加 googleSearch() 工具...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1318741968242737282)** (11 messages🔥): 

> `Depth AI, LightRAG, Codebase Indexing, AI Assistant Deployment` 


- **Depth AI 的代码理解能力令人印象深刻**：用户们非常喜欢 [Depth AI](https://www.trydepth.ai/)，因为它能够构建代码库的全面知识图谱，并以 **99% 的准确率**回答深层的技术问题。
   - 许多人发现设置非常简单，尽管对大型项目（20万 - 150万 tokens）进行索引可能需要一些时间，一位用户指出一个 18万 tokens 的 repo 花费了 **40 分钟**。
- **LightRAG 作为替代方案被讨论**：在关于 Depth AI 的讨论中，一位用户建议尝试 [LightRAG](https://github.com/HKUDS/LightRAG)，它被描述为一个简单且快速的 Retrieval-Augmented Generation（RAG）工具。
   - 然而，另一位用户表示更倾向于 Depth AI，认为它更容易设置且可能更有效。
- **索引时长引发褒贬不一的反应**：虽然一位用户报告成功完成了项目的索引，但另一位用户提到其中型项目已经索引了 **4 小时**。
   - 索引所需时间似乎根据 token 大小有很大差异，这强调了耐心的重要性。
- **对 Depth AI 输出结果的担忧**：一位用户表达了挫败感，因为 Depth AI 在完成索引后，针对其查询返回了“未生成任何输出（no output was generated）”。
   - 这引发了人们对即使索引成功，输出可靠性仍存疑的问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.trydepth.ai/">Depth AI - 深度理解你代码库的 AI</a>：与你的代码库聊天或构建自定义 AI Assistants。在任何你工作的地方部署它们 —— Slack, Github Copilot, Jira 等。</li><li><a href="https://github.com/HKUDS/LightRAG">GitHub - HKUDS/LightRAG: &quot;LightRAG: Simple and Fast Retrieval-Augmented Generation&quot;</a>: &quot;LightRAG: 简单快速的检索增强生成&quot; - HKUDS/LightRAG
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1318999359412506645)** (1 messages): 

> `12 Days of OpenAI, Stay Updated Role` 


- **加入 12 Days of OpenAI 活动**：OpenAI 鼓励成员通过在 <id:customize> 中领取 <@&1261377106890199132> 身份组，来关注 **12 Days of OpenAI** 期间的最新动态。
   - 这是接收更新并参与正在进行的庆祝活动的绝佳方式。
- **第 10 天庆祝活动亮点**：公告通过链接的 [YouTube 视频](https://www.youtube.com/watch?v=LWa6OHeNK3s) 展示了 **Day 10** 的庆祝活动，该视频展示了正在进行的各项环节。
   - 鼓励成员观看视频，了解与当天活动相关的精彩内容。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1318674710694989916)** (220 条消息🔥🔥): 

> `OpenAI 与 Google AI 的进展对比，不同 AI 模型的使用体验，AI 与安全担忧，个人辅助 AI，DALL·E 与 Midjourney 图像生成对比` 


- **OpenAI 与 Google 的竞争格局**：讨论集中在 [OpenAI](https://openai.com) 与 Google 之间的竞争，许多参与者认为 Google 目前在 AI 进展方面领先于 OpenAI。
   - 有人担心 OpenAI 可能会为了竞争策略而保留模型，而另一些人则推测 Google 的快速创新可能会定义未来的 AI 格局。
- **对不同 AI 模型的多元化体验**：成员们分享了对不同 AI 模型的看法，许多人倾向于使用 OpenAI 的 GPT 模型进行编程和数学运算，同时也对 Gemini 2.0 Flash 的表现表示了一些不满。
   - 用户表达了 Agent 如何通过自主执行任务来显著改善残障人士的生活，反映出对 AI 实际应用的渴望。
- **AI 安全与伦理担忧**：参与者辩论了当前 AI 安全措施的有效性和伦理，一些人声称当前的解决方案可能会限制创造力和实用性。
   - 重点在于寻找确保安全与允许探索 AI 能力之间的平衡，一些人指出过度审查是一个潜在问题。
- **对个人 AI 助手的兴趣**：一个重要的讨论点是渴望拥有能够自主管理任务并简化日常生活的个人 AI 助手，特别是针对正在康复的健康问题的老年用户。
   - 对话集中在这种技术如何提高生活质量，并提到了 Google 在该领域的持续进展。
- **图像生成模型对比**：用户将 OpenAI 的 DALL·E 与 Midjourney 以及 Google 的 Imagen 进行了对比，尽管 DALL·E 是免费使用的，但用户经常抱怨其局限性和质量。
   - 用户对 DALL·E 的输出很容易被识别为“AI 生成”表示不满，同时强调了 Midjourney 的价格和制作质量是需要考虑的因素。



**提到的链接**：<a href="https://github.com/AlignAGI/Alignment/">GitHub - AlignAGI/Alignment: Promoting global awareness and action for ethical AI alignment and safeguarding humanity against AI self-replication risks. Includes research, frameworks, and open-source resources.</a>：促进全球对伦理 AI 对齐（Alignment）的意识和行动，保护人类免受 AI 自我复制风险。包括研究、框架和开源资源。

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1318739662197624903)** (3 条消息): 

> `Custom GPTs 功能，训练中的经理角色` 


- **ChatGPT 训练角色澄清**：一名成员询问在提示 ChatGPT 承担特定角色时，“你现在是训练我的经理”这一指令是否能有效发挥作用。
   - *这是获得更好回复的关键吗？*
- **编辑 Custom GPTs 的限制**：另一名成员对无法编辑 Custom GPTs 表示沮丧，认为这是系统的一个潜在缺陷。
   - *我们是否别无选择？*


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1318796591389347902)** (4 条消息): 

> `频道发布礼仪，在合适的频道寻求帮助` 


- **频道发布礼仪受到批评**：一名成员批评另一名成员在多个频道发布信息，将其标记为 **垃圾信息 (spam)**，并指示其删除除正确频道 <#1047565374645870743> 以外的所有消息。
   - 这一评论强调了使用指定频道以维持秩序和避免混乱的重要性。
- **在正确的地方寻求帮助**：一名成员对合适的频道表示不确定，称他们只是想找到最佳的输入方式。
   - 这一询问凸显了用户在遵循频道指南寻求帮助时面临的挑战。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1318796591389347902)** (4 条消息): 

> `Channel Overposting, Seeking Help, Proper Channel Usage, Spam Concerns` 


- **频道过度发布引发辩论**：一名成员质疑为什么一个帖子被分享到了 **四个频道**，强调了对 Spam 的担忧。
   - 他们建议从其他频道删除该帖子，以简化讨论流程。
- **成员寻求指导**：另一名成员对询问所在的 **合适频道** 表示不确定，称他们只是在寻求帮助。
   - 这引发了关于频道组织和成员意识的问题。
- **呼吁正确使用频道**：作为回应，一名成员强调此类帖子有指定的 **正确频道**，建议遵守指南。
   - 他们表示在帖子从其他位置删除后将提供协助。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1318670110012997702)** (210 messages🔥🔥): 

> `Falcon Model Performance, Prompt Chaining Techniques, OpenAI Safety Discussions, Feedback and Evaluation Systems, API and Tool-Use Support in Models` 


- **Falcon 模型展现潜力**：Falcon3 模型，特别是 7B 和 10B 版本，展现出强劲的性能，用户表示有兴趣测试其在各种应用中的能力。
   - 最近的更新增加了 Tool-use 支持，增强了它们的功能，特别是在需要复杂交互的场景中。
- **创新的 Prompt Chaining 策略**：关于 Prompt Chaining 的讨论强调了其通过使用一系列模型迭代处理和优化响应来增强模型输出的效用。
   - 建议使用结构化输出和树状结构等技术来改进故事讲述和其他创意任务。
- **OpenAI 的安全可信度受到质疑**：人们对 OpenAI 对安全实践的关注表示担忧，特别是在 GPT-4o 和 o1 preview 的对比演示中展示了针对其模型的 Jailbreak。
   - 这引发了关于其安全声明与实际模型漏洞之间 Alignment（对齐）情况的持续对话。
- **反馈和评分系统**：用户正在实施评估框架，使用详细说明各种叙事元素的特定量表（rubrics）来评估模型生成的故事质量。
   - 这种系统化方法旨在通过迭代反馈和评估机制产生更高质量的输出。
- **API 和本地模型性能**：关于在本地模型中不使用 Batching 运行 Inference 的普遍性进行了讨论，用户主张通过请求排队来提高效率。
   - 目标是探索将其集成到包括市场模拟在内的各种应用中，在这些应用中，现实世界的测试变得至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.langflow.org/">Welcome to Langflow | Langflow Documentation</a>：Langflow 是一个用于构建 multi-agent 和 RAG 应用的新型可视化框架。它是开源的、由 Python 驱动、完全可定制，并且与 LLM 和向量数据库无关。</li><li><a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">Scaling test-time compute - a Hugging Face Space by HuggingFaceH4</a>：未找到描述</li><li><a href="https://x.com/demi_network/status/1869085748852662718">Democratize Intelligence (@demi_network) 的推文</a>：&#34;这不是公司与 AI 之间对齐的问题，而是公司与你之间对齐的问题。你的 AI 为谁工作将变得非常重要。如果你的 AI 是 ...</li><li><a href="https://huggingface.co/tiiuae/Falcon3-7B-Instruct-1.58bit">tiiuae/Falcon3-7B-Instruct-1.58bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/tiiuae/falcon-11B">tiiuae/falcon-11B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/tiiuae/falcon-7b-instruct">tiiuae/falcon-7b-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/tiiuae/Falcon3-10B-Instruct">tiiuae/Falcon3-10B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blog/falcon3">Welcome to the Falcon 3 Family of Open Models!</a>：未找到描述</li><li><a href="https://huggingface.co/tiiuae/falcon-40b-instruct">tiiuae/falcon-40b-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/tiiuae/Falcon3-10B-Instruct#benchmarks)">tiiuae/Falcon3-10B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/_xjdr/status/1869062741849461146">xjdr (@_xjdr) 的推文</a>：这是我在 NeurIPS 从 ~可靠来源多次听到的最有趣的事情之一（newsonnet 是 400B dense）引用 Aidan McLau (@aidan_mclau) @Heraklines1 @deedydas 不，不是 ...</li><li><a href="https://safepine.co/">Safepine</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/175ejvi/quick_start_example_for_llava_generate_image/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1318853936660221984)** (13 messages🔥): 

> `在本地模型上进行 Function calling，函数获取中的偏差，搜索引擎的有效性，Hermes 3 405B 模型问题，AI 回复中的粉红大象问题` 


- **探索 Function Calling 库**：有人询问了在**小型本地模型**上进行 **function calling** 的最佳库和方法。
   - 这表明人们对优化本地系统上的 AI 性能有着持续的兴趣。
- **模型召回导致的偏差**：讨论集中在使用语言模型进行数据召回的陷阱上，强调**正确性**是基于来源和目的的主观判断。
   - 有人担心，如果模型利用通用的网络搜索，可能会将**有偏差的信息**误认为真理。
- **搜索引擎质量辩论**：一位成员表达了沮丧，认为目前的搜索引擎充斥着 **SEO 垃圾信息**和不可信的新闻网站。
   - 他们渴望有一个能索引所有已著书籍和论文的高级搜索引擎。
- **Hermes 3 405B 模型反馈**：一位用户报告了 **Hermes 3 405B 模型**在回复时会还原 Prompt 的问题，尽管有指令要求不要这样做。
   - 他们指出与 GPT-4O 的对比显示后者问题较少，并询问重新组织 Prompt 是否会有所帮助。
- **粉红大象问题与模型回复**：讨论了“粉红大象问题”，说明了指示模型“不要做什么”反而会无意中触发该行为。
   - 提到了增强模型针对此类陷阱的鲁棒性的研究，促使了用户 Prompt 策略的转变。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1318732957287972976)** (2 messages): 

> `推理中的信号与噪声，LLM 输出的一致性，长输出挑战` 


- **信号 vs 噪声：清晰思考的关键**：**信噪比**的重要性被强调为连贯且清晰推理的关键，类似于它在**人脑**中的作用。
   - *“我们什么时候能听到关于这类事情的消息？”* 表示对该话题更深入讨论的期待。
- **寻求关于 LLM 一致性论文的推荐**：一位成员表示有兴趣寻找专注于 **LLM** 输出**一致性**的**优秀论文**，特别是针对长输出和超长输出。
   - 这引发了对 LLM 在长文本长度下维持输出质量所面临**挑战**的进一步探索。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1318732957287972976)** (2 messages): 

> `推理中的信号与噪声，LLM 输出的一致性` 


- **推理中信号与噪声的重要性**：一位成员强调，**信号与噪声**的比率对于连贯且清晰的推理至关重要，并将其与人脑中的作用进行了类比。
   - *“我们什么时候可以期待听到更多关于这方面的内容？”*
- **寻求关于 LLM 一致性论文的推荐**：另一位成员表示有兴趣听取关于 LLM 输出**一致性**的**最佳论文**推荐，特别是关注长到超长输出。
   - 他们征求了小组的意见，明确表示希望围绕该话题进行相关讨论。


  

---

### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1319005592332796045)** (1 条消息): 

> `3-panel UI 变更、建议操作移除、缺失功能的替代方案` 


- **3-panel UI 的推出移除了建议操作**：新的 **3-panel UI** 取消了 NotebookLM 原有的 **'suggested actions'**（建议操作）功能，其中包括 'Explain'（解释）和 'Critique'（评论）等提示词。由于之前的设置发现率和功能性有限，很少被用户利用。
   - 许多用户注意到了这一变化，此前反馈也强调了建议操作的使用率极低。
- **计划以更好的设计恢复功能**：开发团队计划在未来几个月内，以更直观的方式恢复建议操作中的大部分功能。他们旨在通过集成提高引用和回答准确性的新功能来增强用户体验。
   - 鼓励用户在后续版本的改进实施过程中分享更多反馈。
- **引入替代方案**：在此期间，用户可以通过从源文件中复制文本并在聊天中直接要求解释或总结来重新实现建议操作。“**将所有笔记转换为源文件**”（convert all notes to source）功能允许用户从笔记中创建新源，以便进行更结构化的查询。
   - 这种方法通过确保回答包含可点击的引用，同时直接关注用户笔记，从而维持了功能性。


  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1318687630849871952)** (27 条消息🔥): 

> `多语言功能、播客长度自定义、交互式 AI 使用案例、知识库生成、创意播客制作` 


- **多语言功能实验**：成员们对实验 NotebookLM 的交互功能以简化不同语言（特别是巴西语和孟加拉语）的对话感到兴奋。
   - 一位用户提到，在提示词中表达多语言需求，可以更轻松地在聊天中参与这些讨论。
- **播客长度自定义模板讨论**：有建议提出创建一个时间模板来控制单集长度，一位成员表示希望播客能更长一些，以便深入探讨内容而不跳过引人入胜的对话。
   - 另一位成员询问此类模板将如何运作，暗示需要一个范围而不是精确的时长。
- **交互式 AI 的创意用途**：多位用户讨论了利用 NotebookLM 及类似工具进行创意尝试，包括生成播客以及参与可能未被广泛覆盖的小众话题。
   - 一位用户分享了他们在为一个开源预测市场平台审阅学术材料时录制简短单集的方法。
- **使用 NotebookLM 生成知识库**：一位成员询问 NotebookLM 是否可以生成类似于检索增强生成（RAG）的知识库，并寻求见解或替代方案。
   - 另一位用户指向了一个展示将 NBLM 用作知识库的 YouTube 视频，暗示这可能正是询问者所寻找的。
- **AI 驱动的播客制作见解**：一位用户分享了他们制作 AI 生成播客的经验，强调需要添加个人评论以避免“AI 废话”（AI slop）并保持内容质量。
   - 他们计划通过不仅依赖 NotebookLM 的初稿，还通过参与交互模式来完善内容，从而增强他们的播客。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/NXdUMyZPUi4"> - YouTube</a>：未找到描述</li><li><a href="https://open.spotify.com/episode/6pjDfRqlfDGZY1KTpxD2iS?si=qEpiAJXiRPm67UPLDbedKw">Ask Gennie! Reverse Mortgage Q&amp;A - What is a Reverse Mortgage for Seniors? What are the benefits of the reverse mortgages for elder people and retirees?</a>：Ask Gennie! Mortgage Questions Answered with Experts from GenNext.Mortgage (NMLS #2326098) · Episode
</li>
</ul>

</div>

### **NotebookLM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1318669153707491358)** (194 条消息🔥🔥): 

> `NotebookLM 播客功能、交互模式（Interactive Mode）推出、音频概览（Audio Overview）功能、来源集成与更新、使用 NotebookLM 进行案例研究准备` 


- **播客长度控制的挑战**：用户在设置播客特定长度时遇到困难，即使在备注中加入音频长度要求也经常被忽略。
   - 一些人建议使用精确的 Prompting 技巧，但输出结果仍存在不一致性。
- **交互模式（Interactive Mode）推出问题**：交互模式功能的推送速度较慢且具有随机性；拥有新 UI 的用户可能仍无法访问该功能。
   - 反馈表明音频生成经常出现延迟或失败，部分用户通过重置来应对这些限制。
- **同步 Google Docs 作为来源**：用户不确定链接为来源的 Google Docs 是否会自动同步更新，还是需要手动刷新。
   - 目前，来源不会自动更新，这引发了关于未来路线图中自动同步文件计划的疑问。
- **合并与管理笔记**：新 UI 缺乏合并所选笔记的功能，操作仅限于单条笔记或同时处理所有笔记。
   - 这一限制引发了关于改进 UI 以促进更好笔记管理的讨论。
- **案例研究与学习辅助**：用户分享了利用 NotebookLM 作为学习辅助工具的经验，强调了该工具在整理演讲者备注方面的帮助。
   - 针对备考（尤其是案例研究）的深入建议强调了通过彻底的资源集成来应用概念的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/noob-gif-5274024">Noob GIF - Noob - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://blog.google/technology/google-labs/notebooklm-new-features-december-2024/">NotebookLM gets a new look, audio interactivity and a premium version</a>: NotebookLM 正在推出新功能以及名为 NotebookLM Plus 的高级版本。</li><li><a href="https://www.reddit.com/r/GooglePixel/comments/z5i6ns/a_hidden_gem_the_pixel_recorder/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://youtu.be/7mqciPtMfBI?si=IStj7r25df71U40Y"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1318675575484711043)** (66 条消息🔥🔥): 

> `Fine-tuning Llama 3.2, Batch Size and Training, Function Calling in Models, Multi-GPU Support in Unsloth, Overfitting in Machine Learning Models` 


- **Fine-tuning Llama 3.2 与 4-bit 转换**：一位成员正在探索如何通过添加数据集有效地对 **Llama 3.2** 模型进行 Fine-tune，并讨论了加载之前 Checkpoints 的选项。
   - 另一位成员强调，像 **load_in_4bit=true** 这样的设置允许对非 Unsloth 上传的模型进行自动转换。
- **优化训练的 Batch Size**：讨论了关于最佳 **Batch Size** 的问题，较大的尺寸可能会提高训练的稳定性和准确性，尽管这需要更多的 VRAM。
   - 成员们一致认为，对于 VRAM 有限的用户，增加 Gradient Accumulation 可以作为一种替代方案。
- **Function Calling 与模型理解**：用户寻求关于包含 Function Calling 的模型 Prompt 格式的澄清，一些成员指出直接包含 Special Tokens 是可行的。
   - 分享了一个资源链接，展示了 Llama 模型中 Function Calling 的 Prompt 格式。
- **Unsloth Pro 的 Multi-GPU 支持**：一位用户询问 Unsloth Pro 的 **Multi-GPU 支持** 是否已投入使用，特别是它是否适用于本地设置或仅通过云平台运行。
   - 回复确认 Multi-GPU 功能已可用，增强了模型训练体验。
- **解决 Fine-tuned 模型中的 Overfitting 问题**：一位成员报告其在 **Hugging Face** 上导出的 Fine-tuned 模型表现不佳，暗示可能存在 **Overfitting**。
   - 另一位成员建议，问题可能源于模型参数或数据集质量，而非 Fine-tuning 框架本身。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/chat-templates>)...">未找到标题</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">教程：如何 Fine-tune Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>: 为在 Ollama 上本地运行创建定制化个人助手（如 ChatGPT）的初学者指南</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#zero-shot-function-calling-e2e-format>)">meta-llama/llama-models 仓库 main 分支下的 llama-models/models/llama3_2/text_prompt_format.md</a>: 旨在用于 Llama 模型的实用工具。通过在 GitHub 上创建账户为 meta-llama/llama-models 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1318684381124100097)** (139 条消息🔥🔥): 

> `Open Source Reasoning Models, Unsloth Model Training, Fine-Tuning with QwQ, DiLoCo Presentation, LORA vs Model Architecture` 


- **开源推理模型辩论**：成员们讨论了像 **QwQ** 这样的开源推理模型优于传统模型的潜力，并指出虽然复现推理过程很容易，但创建一个成功的模型仍然具有挑战性。
   - 对于当前模型设计中强化学习 (RL) 的必要性存在怀疑，有建议认为纯监督微调 (SFT) 配合高质量数据集可能就足够了。
- **Unsloth 训练经验**：一位用户详细介绍了他们使用 **Unsloth** 训练模型的经验，由于对外部仓库的依赖，在以 GGUF 格式保存模型时遇到了问题。
   - 对话包括了故障排除方法，强调了正确安装的重要性以及成功执行所需特定文件的存在。
- **Adapter 与 Model 的区别解释**：用户得到了澄清，即模型由影响其参数的权重集合组成，而低秩自适应 (LoRAs) 仅修改这些参数的一个子集。
   - 讨论强调了 LoRAs 如何与模型结合以进行高效训练，而无需改变整个架构。
- **DiLoCo 研究分享**：一位成员分享了他们关于 **DiLoCo** (Distributed Low-Communication Training of Language Models) 的研究，并为他们的小组制作了演示报告，引发了频道内其他人的兴趣。
   - 该成员被鼓励在更广泛的背景下发布他们的发现，以获得额外的反馈。
- **使用 LORA 训练的输出大小疑问**：一位用户询问了使用 LoRA 训练模型时的预期输出大小，指出由于 Adapter 训练的特性，其输出明显小于预期。
   - 随后讨论了如何有效地结合模型和 Adapter，并参考了关于正确保存和量化模型的文档。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing#scrollTo=r2v_X2fA0Df5">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://medium.com/@yuxiaojian/fine-tuning-ollama-models-with-unsloth-a504ff9e8002">使用 Unsloth 微调 Ollama 模型</a>：在前两篇文章中，我们探讨了在云端 Kubernetes (K8s) 集群中托管您自己的 Ollama 服务，以及运行您自己的 OLLAMA...</li><li><a href="https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-gguf">保存为 GGUF | Unsloth 文档</a>：将模型保存为 16bit 的 GGUF，以便您可以将其用于 Ollama、Jan AI、Open WebUI 等！</li><li><a href="https://huggingface.co/collections/kaleinaNyan/eule-675ad4e60d8d2cd0d958b32a">Eule - 一个 kaleinaNyan 收藏集</a>：未找到描述</li><li><a href="https://huggingface.co/kaleinaNyan/eule-qwen2.5instruct-7b-111224">kaleinaNyan/eule-qwen2.5instruct-7b-111224 · Hugging Face</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：查看下方我们所有的 Notebook 列表：</li><li><a href="https://docs.google.com/presentation/d/18Twuq0q1H75BxUOgRc8ZTs2lWGvE2XtnAGZ7CWtq3cA/edit?usp=sharing">DiLoCo: Distributed Low-Communication Training of Language Models</a>：DiLoCo: Distributed Low-Communication Training of Language Models OpenDiLoCo: 一个用于全球分布式低通信训练的开源框架 INTELLECT-1 技术报告</li><li><a href="http://arxiv.org/abs/2311.08105">DiLoCo: Distributed Low-Communication Training of Language Models</a>：大语言模型 (LLM) 已成为机器学习许多应用中的关键组成部分。然而，训练 LLM 的标准方法需要大量紧密互连的加速器...</li><li><a href="http://arxiv.org/abs/2407.07852">OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training</a>：OpenDiLoCo 是大语言模型分布式低通信 (DiLoCo) 训练方法的开源实现和复现。我们提供了一个可复现的 DiLoCo 实现...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1318787357348204586)** (15 条消息🔥): 

> `Llama 3.2 训练 Loss，M4 MAX GPU 兼容性，Mac 上的 Unsloth 支持` 


- **Llama 3.2 的 Loss 差异**：一位用户报告称，在使用 **llama template** 训练 **Llama 3.2** **1bn instruct model** 时，其 Loss 比使用 **alpaca prompt** 时高出 **3 倍**，初始值从 **5.1** 开始。
   - 另一位用户寻求澄清数据集是否正确地与 llama template 配合使用。
- **M4 MAX GPU 仍处于未开发领域**：一位用户询问了对 **M4 MAX GPU** 的支持情况，并指出目前的 **conda install** 指南仅适用于 **CUDA**。
   - 回复指出 **Unsloth** 目前不支持 Mac。
- **Mac 支持时间线推测**：一位成员推测对 Mac 的支持应该会在 **2025 年第二季度** 左右落地，但这取决于可用于开发的时间。
   - 鼓励社区贡献以加速这一进程。
- **Mac 上有限的微调选项**：一位用户提到 Mac 上缺乏 **fast fine-tuning alternatives**，并询问目前是否仍然如此。
   - 回复确认了这种不确定性，因为该用户没有 **NVIDIA** 硬件。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1318791940686479383)** (1 条消息): 

> `OpenAI o1 model, Structured outputs, EVA Llama model, Price reductions, Provider pages` 


- **OpenAI o1 模型发布，带来酷炫功能**：全新的 OpenAI **o1 model** 已上线，接替了 [o1-preview](https://openrouter.ai/openai/o1-preview)，并具备 **function calling** 和更低延迟等特性。
   - 它引入了一个新的 `reasoning_effort` API 参数，用于控制模型在回答前的思考时间，增强了用户交互性。
- **Structured outputs 受到关注**：OpenRouter 现在为来自 **8 家不同公司**的 **46 个模型**标准化了 **structured outputs**，使用户更容易以首选格式获取结果。
   - [此处](https://x.com/OpenRouterAI/status/1869077909438091485)分享了关于这一技巧的教程，强调了其在实际使用中的相关性。
- **新的故事讲述模型 EVA Llama 加入阵容**：一款新的角色扮演和故事讲述模型 **EVA Llama** 已发布，同时还更新了 **Grok 2** 和 **Cohere** 模型。
   - 用户可以通过[此链接](https://openrouter.ai/eva-unit-01/eva-llama-3.33-70b)深入了解 **EVA Llama** 的详细信息。
- **热门模型迎来令人兴奋的价格下调**：**mythomax-l2-13b** 模型降价 **12.5%**，使其更易于使用。
   - 此外，备受追捧的 **QwQ reasoning model** 大幅降价 **55%**，其高性价比令社区印象深刻。
- **Provider 页面提供深入的分析数据**：用户现在可以点击提供商名称查看模型托管图表，提高了性能随时间变化的透明度。
   - 以 [DeepInfra 的 provider 页面](https://openrouter.ai/provider/deepinfra)为例，该页面提供了详细的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/openai/o1-preview>)">o1-preview - API, Providers, Stats</a>：OpenAI 最新且最强大的模型系列，o1 旨在响应前花费更多时间进行思考。o1 模型针对数学、科学、编程和其他 STEM 相关任务进行了优化...</li><li><a href="https://x.com/OpenRouterAI/status/1869077909438091485))">来自 OpenRouter (@OpenRouterAI) 的推文</a>：Structured outputs 被严重低估了。将 LLM 输出约束为 JSON schema 通常比请求 tool call 要容易得多。OpenRouter 现在为 46 个模型、8 个...标准化了 structured outputs。</li><li><a href="https://openrouter.ai/openai/o1>)">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格。</li><li><a href="https://openrouter.ai/eva-unit-01/eva-llama-3.33-70b>)">EVA Llama 3.33 70b - API, Providers, Stats</a>：EVA Llama 3.33 70b 是一款角色扮演和故事创作专业模型。通过 API 运行 EVA Llama 3.33 70b。</li><li><a href="https://openrouter.ai/models)">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格。</li><li><a href="https://openrouter.ai/provider/deepinfra)">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格。</li><li><a href="https://openrouter.ai/gryphe/mythomax-l2-13b>)">MythoMax 13B - API, Providers, Stats</a>：Llama 2 13B 最具性能且最受欢迎的微调版本之一，具有丰富的描述和角色扮演能力。#merge。通过 API 运行 MythoMax 13B。</li><li><a href="https://openrouter.ai/qwen/qwq-32b-preview)">QwQ 32B Preview - API, Providers, Stats</a>：QwQ-32B-Preview 是由 Qwen 团队开发的专注于 AI 推理能力的实验性研究模型。作为预览版，它展示了出色的分析能力，同时也存在一些...</li><li><a href="https://x.com/OpenRouterAI/status/1869237170952978935">来自 OpenRouter (@OpenRouterAI) 的推文</a>：OpenAI o1 现已对所有人开放！在以下方面尝试它的 🧠：图像输入、structured outputs、function calling、以及 "reasoning effort" 控制。下方的 Chatroom 链接有一些挑战，您可以尝试...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1318682928175382682)** (209 条消息🔥🔥): 

> `Exposed OpenRouter keys, Chat details in API, Using OpenRouter API keys with PKCE, OpenRouter pricing structure, Model performance comparisons` 


- **报告泄露的 OpenRouter 密钥**：一位用户在 GitHub 上发现了限额超过 100 美元的泄露 OpenRouter API 密钥，并询问在哪里报告，一名成员建议发送至 support@openrouter.ai。
   - 讨论了通过电子邮件发送这些受损密钥的安全性。
- **在 API 中检索聊天详情**：有人询问如何查看 API 调用的聊天详情，因为担心一旦访问了元数据就无法检索 Prompt 或响应。
   - 建议增加一个标志位，将对话视为聊天而非无状态请求。
- **结合 PKCE 使用 OpenRouter**：一位用户讨论了通过 PKCE 使用 OpenRouter API 密钥创建 Web 应用，权衡在客户端还是后端处理密钥的安全性。
   - 建议在保持近乎无状态架构的同时安全地管理 API 密钥。
- **OpenRouter 定价与成本**：用户寻求关于 OpenRouter 服务相关费用的澄清，特别是使用自己的 API 密钥是否会产生额外费用。
   - 注意到使用自定义密钥会在上游提供商成本的基础上额外收取 5% 的费用。
- **各种模型的性能**：一位用户注意到模型响应的不一致性，特别是 QwQ，引发了关于模型规模在指令遵循中作用的讨论。
   - 鼓励用户使用更高端的模型，如 Google Experimental 1206 或 DeepSeek-v2，以获得更一致的代码辅助。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/terms#_4_-payment">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格</li><li><a href="https://openrouter.ai/docs/integrations">Integrations | OpenRouter</a>: 在 OpenRouter 中使用您自己的提供商密钥</li><li><a href="https://openrouter.ai/docs/integrations#embeddings">Integrations | OpenRouter</a>: 在 OpenRouter 中使用您自己的提供商密钥</li><li><a href="https://cdn.openai.com/spec/model-spec-2024-05-08.html">Model Spec (2024/05/08)</a>: 无描述</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: 设置模型使用限制</li><li><a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>: 根据各应用的使用情况对语言模型进行排名和分析
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1318986871719858287)** (1 条消息): 

> `Retail/E-commerce ad models, Runway, OpenAI Sora, Veo 2` 


- **寻求有效的零售广告模型**：一位成员询问用于创建**零售/电子商务广告内容**（包括视频和文案格式）的有效模型。
   - 他们特别提到正在考虑 **Runway**、**OpenAI Sora** 和 **Veo 2**，同时征求其他选项的建议。
- **探索广告内容的替代方案**：讨论旨在识别除已提到的模型之外，专门为广告内容定制的**其他潜在模型**。
   - 该成员专注于收集多样化的选项，引发了关于市场上现有技术的更广泛对话。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1318673940763119689)** (123 条消息🔥🔥): 

> `Warmup phase for learning rates, Meta-Learning to reduce overfitting, Compression methods in neural networks, Grokking in large models, Koopman operator theory in neural networks` 


- **学习率的 Warmup 阶段**：讨论了在训练初期使用 Warmup 阶段来稳定学习率的重要性，特别是在处理大规模数据集和复杂模型架构时。
- **通过 Meta-Learning 减少过拟合**：成员们探讨了 Meta-Learning 技术如何通过使模型能够从较少的数据点中更有效地泛化，从而帮助减轻过拟合。
- **神经网络中的压缩方法**：对话涉及了各种神经网络压缩技术，旨在减少模型大小并提高推理速度，同时不显著牺牲准确性。
- **大型模型中的 Grokking 现象**：讨论了 Grokking 现象，即模型在长时间训练后突然从记忆转向泛化，以及它在 Transformer 架构中的表现。
- **神经网络中的 Koopman 算子理论**：研究人员讨论了 Koopman 算子理论在理解神经网络动力学方面的应用，旨在为模型行为提供更具解释性的框架。

- **辩论 Warmup 阶段公式**：Kevin 用于近似 Warmup 阶段的公式 (1 - beta1^step) 目前缺乏 LR schedulers 的支持，引发了关于其实现的讨论。
   - 成员们分享了各自的实现，并表达了在使用 lambdaLR 时与步数（step counts）相关的 off-by-one 错误的担忧。
- **利用 Meta-Learning 解决过拟合问题**：讨论了 **Meta-Learning** 是否能帮助缓解监督学习模型中的过拟合，并征集了具体案例。
   - 社区指出，虽然存在理论框架，但实际案例仍然稀缺。
- **探索神经网络压缩技术**：成员们探讨了神经网络压缩的相关想法，重点关注深度压缩（depthwise compression）和结合了稀疏与低秩矩阵的 OATS 等神经网络剪枝（pruning）方法。
   - 针对压缩可能导致的数据覆盖范围和性能损失提出了担忧，特别是对于针对记忆任务训练的模型。
- **Grokking 作为 AI 研究的核心主题**：讨论了 Grokking 现象，重点关注其重要性以及缺乏在 AI 模型中诱导该现象的有效方法。
   - 普遍认为，虽然 Grokking 得到了一定程度的研究，但主要的研究兴趣集中在 Large Language Models 上，掩盖了更广泛的探索。
- **对 Koopman 理论集成的怀疑**：辩论了 Koopman 算子理论在神经网络中的适用性，成员们对其益处以及将神经层构建为动力系统（dynamical systems）的合理性表示怀疑。
   - 批评者指出论文中可能存在故弄玄虚的情况，认为其主要转化为了对残差连接（residual connections）的利用，而非引入重大创新。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/BlinkDL_AI/status/1869433254425833487">来自 BlinkDL (@BlinkDL_AI) 的推文</a>: 在 ctx4k 下训练的 RWKV-7-World 0.1B (L12-D768) 完美解决了 NIAH ctx16k 🤯 100% RNN 且无 attention。RWKV is all you need。https://www.rwkv.com/ #RWKV 引用 BlinkDL (@BlinkDL_AI) RWKV-7 &#34;Go...</li><li><a href="https://openreview.net/forum?id=DLDuVbxORA">OATS: Outlier-Aware Pruning Through Sparse and Low Rank Decomposition</a>: 最近向大规模基础模型的范式转移为深度学习带来了新纪元，虽然在实践中取得了巨大成功，但也一直受到极高成本的困扰...</li><li><a href="https://arxiv.org/abs/2304.15004">Are Emergent Abilities of Large Language Models a Mirage?</a>: 最近的研究声称 Large Language Models 展示了涌现能力，即在较小规模模型中不存在但在较大规模模型中出现的能力。使涌现能力引人入胜的是...</li><li><a href="https://arxiv.org/abs/1810.01479">Time-Delay Observables for Koopman: Theory and Applications</a>: 非线性动力系统在科学和工程中无处不在，但对这些系统的分析和预测仍然是一个挑战。Koopman 算子理论通过将...规避了其中一些问题。</li><li><a href="https://x.com/BlinkDL_AI/status/1869368399849238727">来自 BlinkDL (@BlinkDL_AI) 的推文</a>: RWKV-7 &#34;Goose&#34; 🪿 0.4B 在 ctx4k 下训练，自动外推至 ctx32k+，并完美解决 NIAH ctx16k🤯 仅在 Pile 数据集上训练。无微调。可复现的训练运行。由我们的...测试</li><li><a href="https://www.jasonwei.net/blog/common-arguments-regarding-emergent-abilities">关于涌现能力的常见争论 &mdash; Jason Wei</a>: 本博客文章不代表我雇主（过去、现在或未来）的立场。我将回顾在讨论 Large Language Models 涌现能力时出现的一些常见争论...</li><li><a href="https://arxiv.org/abs/2409.01308">Representing Neural Network Layers as Linear Operations via Koopman Operator Theory</a>: 简单神经网络的强大性能通常归功于它们的非线性激活。然而，神经网络的线性视角使得理解和控制网络变得更加容易...</li><li><a href="https://distill.pub/2020/growing-ca/">Growing Neural Cellular Automata</a>: 训练一个端到端可微的、自组织的形态发生细胞自动机模型，能够生长和再生特定的模式。</li><li><a href="https://github.com/Jamba15/SpectralTools">GitHub - Jamba15/SpectralTools: Spectral analysis and training of dense layers</a>: 稠密层的频谱分析与训练。欢迎在 GitHub 上为 Jamba15/SpectralTools 的开发做出贡献。
</li>
</ul>

</div>

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1318956907389784074)** (6 条消息): 

> `doc_to_text function arguments, Creating new configs, Overloading config fields` 


- **doc_to_text 函数的额外参数**：一位用户询问是否可以在新任务中向 **doc_to_text** 函数传递额外参数。
   - 另一位成员澄清说，实现这一点的核心入口是通过 **configs**。
- **为 prompts 创建不同的 configs**：一位用户解释说他们有一个定义了函数的 **base config**，并正在考虑为不同的 prompts 使用独立的 configs。
   - 这将导致为每个 prompt 创建不同的子任务，从而增强任务的可定制性。
- **通过包含的任务重载 configs**：有人建议可以使用 `include: <other configs>` 基于另一个 config 创建新 config，以重载特定字段。
   - 然而，这种方法会将重载应用于该 group config 中所有包含的任务，这可能会限制灵活性。
- **MMLU config 示例链接**：一位成员分享说，用户也可以向 group config 添加内容，但这将整体重载所包含的任务。
   - 他们提供了 [MMLU config](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_mmlu.yaml) 的参考链接以获取更多细节。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1318925162933915749)** (9 条消息🔥): 

> `WANDB logging, Configuring WANDB run names, Pull Requests on features` 


- **针对 MFU 和性能指标的 WANDB 日志记录**：一位成员询问在 neox 进行 **pretraining** 期间，是否可以将 **MFU**、**batches/sec** 和 **tokens/sec** 记录到 WANDB，并暗示这将有利于直接绘图。
   - 另一位成员确认，虽然目前没有现成选项，但可以参考[现有的 logging method](https://github.com/EleutherAI/gpt-neox/blob/f5325805678c2b9e35aae4528283e0132c5f5bbc/megatron/logging.py#L352-L361) 来实现。
- **在 Config 中设置 WANDB 运行名称**：一位用户寻求关于如何从 config 设置 WANDB 运行名称的解答，但在尝试直接添加时遇到了错误。
   - 一位成员回复说该选项目前不可用，但承诺将在即将发布的 PR 中连同指标日志功能一起添加。
- **计划中的功能 Pull Requests**：一位成员表示打算在周末为 non-parametric layernorm 功能提交一个 pull request (PR)。
   - 另一位成员提出在日志改进方面提供帮助，但随后保证他们将亲自处理该 PR。



**提到的链接**：<a href="https://github.com/EleutherAI/gpt-neox/blob/f5325805678c2b9e35aae4528283e0132c5f5bbc/megatron/logging.py#L352-L361">gpt-neox/megatron/logging.py at f5325805678c2b9e35aae4528283e0132c5f5bbc · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 transformers 的实现 - EleutherAI/gpt-neox

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1318674059189424158)** (122 messages🔥🔥): 

> `Lora Training Techniques, Current Models in Use, Running Stable Diffusion on Linux, Navigating Image Resolution and Performance, Understanding AI Generated Content and Models` 


- **Lora 训练的有效步骤**：一位用户分享了创建 Lora 的实用步骤：从高质量的数据集开始，选择合适的模型，训练 Lora，然后进行测试。他们强调了研究并构建优质数据集对于获得最佳结果的重要性。
- **Stable Diffusion 的首选模型**：多位用户讨论了他们偏好的模型；一些人青睐 'flux' 模型，而另一些人则因其易用性推荐 'InvokeAI'。其他人指出了拥有 NVIDIA GPU 的重要性，建议使用带有 16GB VRAM 的 3060 以获得更流畅的性能。
- **在 Ubuntu 上运行 Stable Diffusion 的挑战**：用户表达了在 Ubuntu 上运行 SDXL 的挫败感，提到了 ComfyUI 和 Forge UI 的 Linux 兼容性问题。有人指出，想要有效运行 SDXL 可能需要对系统环境有一定了解。
- **为生成模型选择图像分辨率**：一位初学者询问了生成的最佳图像分辨率，试图在质量和处理时间之间寻找平衡。建议包括尝试 1024x1024 左右的分辨率，并使用 hires.fix 以获得更高质量的输出。
- **理解 AI 生成内容的指标**：讨论围绕模型训练中使用的技术和指标展开，特别是针对 Pony 模型及其评分系统。用户注意到这种独特的方法如何影响图像生成效果以及社区的看法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://civitai.com/models/1045828">Epoch Helper - v1.1 | Other Other | Civitai</a>：源代码 - https://github.com/Monkellie/epochcalc # Epoch Helper 工具 这是一个我（在 AI 辅助下）创建的工具，用于帮助自己进行计算...</li><li><a href="https://www.youtube.com/watch?v=AbB33AxrcZo"> - YouTube</a>：未找到描述</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/main/webui-user.sh">stable-diffusion-webui-forge/webui-user.sh at main · lllyasviel/stable-diffusion-webui-forge</a>：通过在 GitHub 上创建账户，为 stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://evermeet.cx/ffmpeg/">static FFmpeg binaries for macOS 64-bit Intel</a>：为 macOS 64 位 Intel 下载静态 FFmpeg 二进制文件。提供快照（snapshots）和发布版二进制文件。FFmpeg 开发人员强烈建议所有用户使用当前的快照构建版本，而不是发布版...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1318746208197414943)** (1 messages): 

> `Custom Web Sources, Perplexity Spaces` 


- **在 Perplexity Spaces 中引入自定义 Web 来源！**：Perplexity 现在允许用户为搜索**选择自定义 Web 来源**，从而能够针对对[你](https://www.perplexity.ai/spaces?utm_source=discord&utm_campaign=websourceslaunch122624)最重要的特定用例进一步定制查询。
   - 随此公告发布的是一段展示新功能的[发布视频](https://cdn.discordapp.com/attachments/1047204950763122820/1318746209778929674/Custom_web_sources_launch_video_-_v6.mp4?ex=67641a5d&is=6762c8dd&hm=df3d15393ffcbb4e7a4be49861d8a1530b60a9dde6a330974e1d1d5ec7789ad2&)。
- **定制化水平再创新高！**：此次更新为用户提供了**增强的定制**选项，使他们能够根据自己的特定需求更有效地策划其 Perplexity 体验。
   - 通过选择 Perplexity 搜索的网站，用户可以根据自己的偏好提高检索信息的针对性和质量。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1318671407638511719)** (108 条消息🔥🔥): 

> `Perplexity Pro Subscriptions, New Features and Updates, User Experience with AI Models, Rate Limits and Performance, User Interface Suggestions` 


- **Perplexity Pro 订阅上线**：用户讨论了 [Perplexity Pro 订阅](https://perplexity.supply/shop/perplexity-subscription) 的推出，该订阅允许赠送知识，提供 1 到 12 个月的时长选项，提升了用户体验。
   - 该订阅提供额外功能，例如搜索 3 倍数量的来源以及访问最新的 AI 模型。
- **用户期待新功能**：成员们表达了对 Perplexity **新功能**的渴望，特别是随着 Google 和 OpenAI 等竞争对手频繁发布新模型，用户感到 Perplexity 进度停滞。
   - 还分享了关于可能与 Meta 等公司合作以取得进展的想法，强调了创新的紧迫性。
- **对速率限制的担忧**：一位用户报告在使用 Perplexity 时遇到了 **rate limits**（速率限制），收到的消息提示他们需要注册更高的个性化速率限制以获得更好的访问权限。
   - 其他用户推测了更高级别在缓解这些限制方面的实际好处，并分享了关于速率限制的个人经历。
- **用户界面增强建议**：一位用户建议在 Perplexity UI 中添加 **下雪效果**，收到了褒贬不一的反馈；一些人认为视觉上很吸引人，而另一些人则更看重实用性。
   - 成员们继续讨论界面美学和可用性如何能更好地满足他们的专业需求。
- **关于 AI 模型性能的讨论**：围绕 **AI 模型性能**展开了对话，一些用户根据其经验认为 Pro Search 的质量有待提高。
   - 一位用户建议使用 Claude 3.5 Sonnet 以获得更好的结果，并质疑了 GPT-4o 等模型声称的进步。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/pplxsupply/status/1868738538231287816?s=46">来自 Perplexity Supply (@PPLXsupply) 的推文</a>: 赠送知识。Perplexity Pro 礼品订阅现已上线。</li><li><a href="https://perplexity.supply/shop/perplexity-subscription">Perplexity Pro 订阅 | Perplexity Supply</a>: Perplexity Supply 旨在通过精心设计的产品探索时尚与智慧之间的关系，激发对话并展示你对知识的无限追求。</li><li><a href="https://vm.tiktok.com/ZMkjhUDEa/">TikTok - Make Your Day</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1318733227187372183)** (4 条消息): 

> `Meta vs OpenAI Pro-Fit, Microbe Threat Warning, Plant Communication, Dopamine Precursors, Cell Revival` 


- **Meta 想要阻止 OpenAI 的营利性业务**：正如在各个论坛中所讨论的，Meta 表达了阻止 OpenAI 追求营利性商业模式的愿望。
   - 围绕这可能对行业内未来 AI 发展产生的影响展开了有趣的讨论。
- **微生物威胁警告出现**：社区关注了最近关于微生物潜在威胁的警告，这是一个可能影响生态平衡的问题。
   - 讨论包括对预防措施的引用，以及在应对这些微生物风险时提高意识的重要性。
- **植物表现出“哭泣”行为**：一篇分享的文章探讨了植物表现出“哭泣”行为的概念，暗示了植物群落中一种独特的交流形式。
   - 这些发现对理解植物对环境压力的反应具有重要意义，激发了人们对植物智能的好奇心。
- **了解多巴胺前体**：链接了一个关于多巴胺前体的资源，阐明了对心理健康至关重要的生化途径。
   - 这一话题引起了社区对其与神经科学研究及潜在治疗结果相关性的兴趣。
- **死细胞复活技术**：成员们讨论了允许死细胞复活的技术的迷人进展，这提出了重大的生物伦理问题。
   - 这项技术对医学和伦理的影响在小组内引发了热烈辩论。



**提到的链接**: <a href="https://www.youtube.com/embed/7PBvDi_aKbs">YouTube</a>: 未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1319010860332482601)** (1 messages): 

> `Perplexity API, Web Search Feature, Cost Overview` 


- **咨询 Perplexity API 中的网页搜索功能**：一位成员询问在他即将使用 Perplexity 开发的项目中，**web search feature**（网页搜索功能）是否包含在 **chat completion API call** 中。
   - 这引发了关于 Perplexity 与他们使用过的其他 API 相比在集成能力方面的重要问题。
- **寻求 Perplexity 的成本概览**：同一位成员表示有兴趣了解使用 Perplexity 服务的 **cost overview**（成本概览）。
   - 了解定价结构对于有效规划他们的项目至关重要。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1318676213417381979)** (41 messages🔥): 

> `6D Parallelism Article, PC Troubleshooting, GPU Performance and Coil Whine, Multi-GPU Instances with NVLink, Coil Whine and Audio Experimentation` 


- **深入了解 6D Parallelism**：一篇关于 [6D parallelism](https://main-horse.github.io/posts/visualizing-6d/) 的详细文章强调了训练中的 **collective communications**（集合通信），旨在提供比现有资源更清晰的视觉效果和解释。
   - 文章批评了其他著作缺乏深度，未能解决结合各种训练方法的复杂性。
- **等待后 PC 成功启动**：一位用户的新 PC 最初出现了问题，显示器没有信号，直到系统在大约一分钟后成功启动，此时 **LED 灯熄灭**。
   - 另一位成员建议尝试使用单根内存条来排除故障。
- **对 Radeon 显卡的沮丧**：一位用户表达了对他们 **Radeon card** 的不满，指出虽然它比 Nvidia 4060 多出约 **10 FPS**，但存在无法接受的 coil whine（电感啸叫）。
   - 他们得出结论，Radeon 和 Nvidia 显卡各有缺点，而 Radeon 的电感啸叫尤其令人困扰。
- **在 VastAI 上寻找多 GPU 实例**：一位用户询问如何在 VastAI 上找到带有 NVLink 的 **multi-GPU instances**，并对列表中显示的有限 **NVLink bandwidth** 表示担忧。
   - 他们推测另一位成员根据之前的对话可能在这一领域有经验。
- **电感啸叫音乐实验创意**：讨论中提到了创建一个利用 GPU 电感啸叫来播放音乐的程序的想法，成员们注意到其 **pitch**（音调）会根据功耗而变化。
   - 一位成员幽默地建议为这个音乐项目捐赠一块 GPU，并将其与他们处理电感啸叫的经历联系起来。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://main-horse.github.io/series/nccl-source-code-study/">NCCL Source Code Study</a>: 未找到描述</li><li><a href="https://main-horse.github.io/posts/visualizing-6d/">Visualizing 6D Mesh Parallelism</a>: 包含一些背景知识
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1318874718299885653)** (1 messages): 

> `Kernel Computation Optimization, Memory Management in GPU, Output Concatenation Techniques` 


- **GPU Kernel 内部的拼接**：一位用户询问在 GPU Kernel 的循环内部是否有高效拼接输出的方法，并引用了之前成功的求和方法。
   - 他们认为在循环期间写入 global memory（全局内存）可能很慢，并询问使用类似 `var[idx:idx+block_size] = value` 的语法是否是可行的替代方案。
- **寻求高效的内存技术**：关于在 GPU Kernel 中拼接输出时写入 global memory 的速度，出现了另一个讨论点。
   - 用户强调在循环内运行操作时需要一种非缓慢的解决方案，这反映了开发者之间的共同关注点。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1318751828472762399)** (8 条消息🔥): 

> `CUDA Memory Copy Issues, Comparing A100 and H100 GPUs, AMP Related Differences` 


- **CUDA Memory Copy 问题**：一位成员报告称，注释掉与 **IsDense(y) && IsSame(x, y)** 相关的特定代码段后功能恢复正常，而保留该代码段会导致 LLM 模型推理过程中出现异常行为。
   - 他们指出 **CudaCopy** 会触发 CUDA kernels，从而引发了关于在此背景下如何处理内存操作的疑问。
- **A100 vs H100 训练差异**：一位成员询问了 **A100** 和 **H100** GPU 在训练期间的差异，特别注意到在单 GPU 任务训练的第一步中存在 **0.3% loss** 差异。
   - 这一意外结果引发了讨论，并促使人们对这两种 GPU 型号的性能指标进行潜在对比。
- **关于 CUDA Graph 支持的疑问**：一位成员对缺乏官方文档说明为何 **CUDA graphs** 在其实现中无法支持 **cudaMemcpyAsync** 操作表示担忧。
   - 这引发了关于异步操作及其在 CUDA graphs 中局限性的进一步讨论。
- **AMP 的潜在影响**：一位成员推测 A100 和 H100 之间观察到的差异是否可能与训练中使用的 **Automatic Mixed Precision (AMP)** 设置有关。
   - 这开启了关于 AMP 如何影响训练结果以及是否需要针对不同 GPU 型号进行调整的对话。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1318671579131150367)** (33 条消息🔥): 

> `Megatron-LM efficiency, Torch.compile warnings handling, Distributed training community, FlexAttention development, Keras/PyTorch contributions` 


- **Megatron-LM 的训练效率受质疑**：一位成员询问 **Megatron-LM** 在训练方面是否仍然高效，因为他们计划在分布式设置中提高 **training throughput**。
   - 另一位成员建议联系 **Gensyn** 以获取见解，并提到了 Christine Yip 活跃的分布式训练社区。
- **智能处理 Torch.compile 警告**：一位用户寻求关于在支持各种 shapes 时管理 **torch.compile** 警告的指导，并强调了 `dynamic=True` 时 kernels 运行缓慢的问题。
   - 另一位成员建议使用 `fn_compiled = torch.compile(fn)` 来灵活处理函数调用，而不必受限于 decorator。
- **开发 FlexAttention 的挑战**：讨论强调了升级 **PyTorch** 时面临的困难，成员们分享了 image builds 涉及的复杂过程。
   - 一位成员承认，他们的目标是在应对自定义设置带来的升级挑战的同时，使 **FlexAttention** 更加健壮。
- **包装模型调用以获得更好的灵活性**：一位成员建议包装模型调用（model invocation）而不是 **torch.compile**，以便更好地处理 shape 变化。
   - 他们还指出，可以使用 Python 的 warnings 模块来过滤特定警告，而不是抑制所有日志。
- **对 Keras/PyTorch 贡献的兴趣**：有人询问了关于对 **Keras** 和 **PyTorch** 的持续贡献，强调了社区参与。
   - 这可能表明了对协作或参与进一步开发工作的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discuss.pytorch.org/t/reduce-time-to-first-kernel-when-using-cuda-graphs/214310">Reduce time to first kernel when using CUDA graphs</a>: 我一直在针对 vLLM 分析我使用的推理栈，我发现由于在调用 graph replay 后，他们的第一个 kernel 几乎立即执行（左侧），而在我的代码中...</li><li><a href="https://github.com/pytorch/torchtitan">GitHub - pytorch/torchtitan: A native PyTorch Library for large model training</a>: 一个用于大模型训练的原生 PyTorch 库。可以通过在 GitHub 上创建账号来为 pytorch/torchtitan 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1318674557439311932)** (4 messages): 

> `Raspberry Pi 5 Deployment, Edge Device Models, Esp32 / Xtensa LX7 Chips` 


- **配备 NVMe 的 Raspberry Pi 5 提升 LLM 性能**：**Raspberry Pi 5** 已超频至 **2.8GHz**，并配备 **256GB NVMe**，以增强部署较小的 **1.5B 参数模型**时的数据传输速度。
   - 使用通过 **OpenBlas** 编译的 **Ollama**，模型可以在 Pi5 上本地运行，从而简化边缘设备的操作。
- **对 Esp32 / Xtensa LX7 芯片的期待**：人们期待 **Esp32 / Xtensa LX7 芯片**能够实现通过 **API** 远程调用 LLM 的新场景。
   - 一位用户在探索不同的部署策略时表达了热情，称其看起来**很有趣！**


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1319053856751095948)** (1 messages): 

> `MatX LLM accelerator, Job openings in ML, ASIC roles` 


- **MatX 开发 LLM 加速器 ASIC**：MatX 正在积极构建 **LLM 加速器 ASIC**，旨在提高机器学习的性能和效率。
   - 他们目前正在为其团队寻找 **底层计算内核作者 (low-level compute kernel author)**、**编译器 (compiler)** 和 **ML 性能工程师**等职位的候选人。
- **MatX 的招聘机会**：MatX 在其网站上列出了多个职位空缺，包括对其 ASIC 技术开发至关重要的职位。
   - 感兴趣的候选人可以在 [MatX Careers](https://matx.com/jobs) 找到有关这些机会的更多详细信息。



**提到的链接**：<a href="https://matx.com/jobs">Tweet from MatX | Jobs</a>：未找到描述

  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1318929596292534343)** (5 messages): 

> `int4group scheme, Training process quantization, Tinygemm compute method` 


- **关于 int4group 流程的澄清**：一名成员询问在 **int4group 方案**中，是否权重保持量化 (int4) 而激活保持为 fp16，从而导致 fp16 x int4 = fp16 的 matmul。
   - 分享了一张图片来可视化该过程，确认其理解与所述流程一致。
- **训练期间激活不进行量化**：关于训练过程的讨论质疑是否会对激活进行任何**伪量化 (fake quantization)**。
   - 澄清了 **Tinygemm 使用 bf16 进行计算**，并且在 QAT 和推理阶段，激活都保持未量化状态。
- **内核中的即时反量化 (On-the-fly dequantization)**：一名成员确认 **int4 权重的反量化**是在 matmul 内核内部即时发生的。
   - 这符合对处理流程的预期，明确了 matmul 内核的运行方式。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1318820781857177630)** (1 messages): 

> `MigraphX in MI300X, ONNX frontend support, Opset compatibility` 


- **在 MI300X 上为 ONNX 构建 MigraphX**：讨论集中在为 **ONNX** 前端在 **MI300X** 上构建 **MigraphX** 的可能性，认为这应该是可行的。
   - 一位成员指出，*“我没有检查支持的 opset(11)，应该是最新的”*，这表明需要进一步验证兼容性。
- **关于 ONNX Opset 支持的查询**：有人提出了关于最新实现中是否支持 **opset(11)** 操作的问题。
   - 这表明当前知识库中可能存在空白，需要团队进一步探索。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 messages): 

kimishpatel: 这就是我来这里的目的 🙂
  

---

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1318672936063733890)** (18 messages🔥): 

> `Custom Vision Encoder, Chain of Thought Generation, Axolotl Configurations, Efficient Sampling Processes, Experimenting with Finetuning` 


- **Custom Vision Encoder 讨论**：一位成员建议创建一个 **custom vision encoder** 以集成到现有的语言模型中，因为目前的模型可能无法有效处理小像素规模的图像。
   - 强调了将 encoder 与各种 LLM 配对的灵活性带来的潜在收益，这超过了预训练 VLM 所带来的改进。
- **探索 Chain of Thought 生成**：讨论集中在 Chain of Thought (CoT) 实现的粒度上，质疑它仅仅是解释核心思想，还是真正提供了迭代思考。
   - 一位成员提出了**双重方法**：在输出前进行推理独白，以及根据谜题类型提供多个模板进行引导式探索。
- **Axolotl Lora 配置成功**：一位成员确认 [llama-3-vision 的 Axolotl Lora 配置](https://github.com/axolotl-ai-cloud/axolotl/blob/effc4dc4097af212432c9ebaba7eb9677d768467/examples/llama-3-vision/lora-11b.yaml) 示例在 2x A6000 GPU 上运行良好。
   - 一旦初始设置得到验证，人们有兴趣寻找计算资源赞助商来支持更大规模的实验。
- **CoT Prompt 的去中心化采样流程**：讨论了在不进行训练的情况下运行采样流程的可能性，旨在通过人工引导的探索来改进 **CoT prompts**。
   - 这种去中心化方法有助于为未来的研究高效收集数据集。
- **RTX 3090 微调实验**：一位成员提到他们有能力在 **RTX 3090** 上运行实验，同时询问使用 bf16 或 Qlora+int8 的最佳微调设置。
   - 引用 **WandB** 的一个示例，确认了 8bit Lora 确实可以在 RTX 3090 上用于 8B 模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://wandb.ai/augmxnt/train-bench/runs/zelehjsm/overview">augmxnt</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/blob/effc4dc4097af212432c9ebaba7eb9677d768467/examples/llama-3-vision/lora-11b.yaml">axolotl/examples/llama-3-vision/lora-11b.yaml at effc4dc4097af212432c9ebaba7eb9677d768467 · axolotl-ai-cloud/axolotl</a>：尽管问 Axolotl 问题。通过在 GitHub 上创建账号为 axolotl-ai-cloud/axolotl 开发做贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1318672102517248053)** (87 条消息🔥🔥): 

> `LM Studio 设置, Qwen QwQ 与角色扮演 LLM, 模型兼容性与错误, 在移动端使用 LM Studio, AI 模型的新进展` 


- **LM Studio 设置与兼容性**：用户讨论了他们的 LM Studio 配置，提到了 RTX 4060 笔记本电脑和拥有 96GB RAM 的 M3 Max 等各种硬件配置，展示了该应用程序的多功能性。
   - 一个具体案例强调了在 LM Studio 中加载 Llama 3.2 11b Vision 时的问题，一名用户遇到了 “unknown model architecture” 错误。
- **Qwen QwQ 作为角色扮演 LLM**：讨论建议将 Qwen QwQ 作为角色扮演类应用的合适选择，几位用户对其性能表示满意。
   - 一位成员指出 Qwen2 在 Python 方面表现出色，表明其在编程语境下的稳健性。
- **模型错误与下载问题**：一条关于 “Safetensors header is unexpectedly large” 的错误消息引发了关于下载过程中潜在文件损坏的讨论。
   - 提醒用户确保模型是从 LM Studio 内部正确下载的，一些用户报告在他们的系统上成功加载。
- **在移动设备上使用 LM Studio**：一位成员表示有兴趣在旅途中通过手机访问 LM Studio，但发现目前没有可用的移动端 App。
   - 虽然有人建议探索替代方案，但直接的移动端兼容性仍然是一个限制。
- **AI 模型的新进展**：用户询问了将类 o1 的 CoT 应用于开源模型的进展，并提到了 Falcon3 bitnet 模型。
   - 社区强调了对 AI 模型增强的持续关注和趋势，推测了未来的可能性和可访问性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mlx-community/Llama-3.2-11B-Vision-Instruct-4bit">mlx-community/Llama-3.2-11B-Vision-Instruct-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://fixupx.com/slow_developer/status/1869059311969661103">来自 Haider. (@slow_developer) 的推文</a>：🚨 NVIDIA 推出 Jetson Nano Super > 每秒可进行 70-T 次运算的紧凑型 AI 计算机 > 专为机器人设计，支持包括 LLM 在内的高级模型，售价 249 美元</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio Beta 版本</a>：LM Studio Beta Releases
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1318670170578751529)** (17 条消息🔥): 

> `3060ti 混淆, AMD 驱动问题, Llama 模型性能, 推理硬件需求, 大模型的 RAM 要求` 


- **3060ti 对比普通 3060**：关于是否存在 **11GB** 显存的 **3060ti** 变体引发了讨论，随后澄清这可能指的是拥有 **12GB** 显存的普通 **3060**。
   - 参与者对其规格表示困惑，其中一人注意到异常的性能表现。
- **AMD GPU 与驱动问题**：有人提到 **Radeon VII** 可能正面临与其他 **AMD GPU** 在 **24.12.1 驱动**下相同的问题，这导致一名用户回退到 **24.10.1**。
   - 问题包括加载模型强制 GPU 达到 **100% 占用率** 却无功耗使用，导致严重的延迟。
- **Llama 模型性能担忧**：一位用户报告称，在一个简单的 **Llama 3.2 3B 模型**上，性能从 **24.10.1** 驱动下的 **90+ tok/s** 骤降至新驱动下的 **20 tok/s**。
   - 另一位用户建议检查配置以确保 **llama.cpp** 设置为使用 **CUDA**，从而潜在地提高性能。
- **对高性能硬件的需求**：一位用户表达了对使用强大的 **M4** MacBook Pro 进行推理的渴望，并反思了他们使用 **M2 MBA** 仅作为入门的经历。
   - 另一位成员引用 **gddr6x** 幽默地评论了对更强硬件需求是一个“无底洞”。
- **大模型的 RAM 要求**：用户讨论了运行大模型的 RAM 需求，指出运行 **70B 模型** 需要在 VRAM 或主内存中准备 **70GB** 空间。
   - 强调了在以 **q8** 精度运行时，拥有 **10-20% 的额外 VRAM** 对于上下文和操作灵活性是理想的。


  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1318837235692474419)** (6 条消息): 

> `从 Firebase 迁移到 Supabase，在 create-mf-app 中使用 Bootstrap，Google reCAPTCHA 问题，测试 ChatGPT Bolt Pilot，Vite Pre-Transform 错误` 


- **从 Firebase 迁移到 Supabase**：一位用户询问了将基于 **Firebase** 构建的整个网站迁移到 **Supabase** 的最佳方法。
   - 关于此类迁移的策略和最佳实践的讨论仍在进行中。
- **Create-mf-app 与 Bootstrap 冲突**：一位成员正在寻求一种一致的方法，将 **create-mf-app** 与 **Bootstrap** 集成，且不与 **Tailwind** 产生冲突。
   - 他们指出，尝试将两者结合通常会导致配置混乱。
- **Google reCAPTCHA 故障排除**：一位用户报告在使用 **Google reCAPTCHA** 时出现了 “Invalid key type” 的初始错误，原因是选择了 v3 而非 **Bolt.new** 所实现的 v2。
   - 切换到 v2 后，他们仍面临验证计数问题，且无法收到来自联系表单的电子邮件。
- **Bolt Pilot 反馈请求**：一位成员宣布他们创建了 **Bolt Pilot**（一个针对 **Bolt** 的新 GPT），并请求用户测试其功能。
   - 他们鼓励用户就任何需要调整以改进的地方提供反馈。
- **Vite Pre-Transform 错误报告**：一位用户对开发过程中遇到大量重复的 “[vite] Pre-Transform” 错误表示担忧。
   - 这个问题似乎也影响了其他人，引发了关于潜在解决方案的进一步讨论。


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1318669185370296461)** (97 条消息🔥🔥): 

> `Token 浪费问题，项目协作，Bolt.diy 导入项目，支付集成讨论，Bolt 用户体验` 


- **用户对 Token 浪费感到沮丧**：许多用户对 **Bolt** 浪费 Token 表示沮丧，甚至有人因为其行为考虑暂停使用。
   - 建议包括添加一个 **“punch the AI”** 按钮来停止 Token 浪费，成员们分享了收到无关回复的经历。
- **项目协作机会**：一位用户正在寻求在 **Bolt** 项目上的协作，邀请他人加入并共同构建优秀的作品。
   - 这引发了关于共享资源和在即将开展的项目中进行团队合作的进一步讨论。
- **从 Bolt.new 导入项目**：用户对如何将项目从 **bolt.new** 导入到 **bolt.diy** 感到好奇，讨论了一种将项目下载为 zip 文件的方法。
   - 提供了关于使用导入文件夹功能以继续处理之前创建的项目的说明。
- **支付集成功能**：讨论了 **Bolt** 实现各种支付集成（如 **Stripe** 和 **PayPal**）的复杂程度。
   - 用户强调了对动态计费等功能的需求，并对该事项未来可能的更新表示关注。
- **用户体验与 Bug**：对重复出现的 Bug 和占位符的沮丧情绪显现，导致测试新功能的用户项目延期。
   - 解决问题的建议包括关闭 diffs 并专注于更好地管理项目文件。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aesthetic-sorbet-a2513b.netlify.app/">Vite + React + TS</a>: 未找到描述</li><li><a href="https://thinktank.ottomator.ai/">oTTomator Community</a>: 创新者和专家齐聚一堂，共同推动 AI 驱动自动化的未来</li><li><a href="https://github.com/RealSput/Wenode">GitHub - RealSput/Wenode: WebContainers, except it&#39;s a million times easier to use</a>: WebContainers，但使用起来要简单一百万倍 - RealSput/Wenode</li><li><a href="https://github.com/stackblitz-labs/bolt.diy#join-the-community-for-boltdiy">GitHub - stackblitz-labs/bolt.diy: Prompt, run, edit, and deploy full-stack web applications using any LLM you want!</a>: 使用任何你想要的 LLM 来提示、运行、编辑和部署全栈 Web 应用程序！ - stackblitz-labs/bolt.diy
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1318718521940770847)** (42 条消息🔥): 

> `Maya Tool Use, 模型集成挑战, 睡眠的重要性, 图像工具开发, 本地模型使用` 


- **推动 Maya Tool Use**：一位成员强调了对 *maya tool use* 的需求，表示“**我们需要 maya tool use**”来增强他们模型的能力。
   - 另一位成员鼓励大家**多休息**，提醒他们恢复精力能激发创造力。
- **模型集成挑战**：讨论了本地模型的集成问题，特别提到了*无原生 tool use*，这给一位成员带来了困难。
   - 在技术挑战面前，他们表达了对目前方法的迷茫，说道：“*我不知道自己在做什么*”。
- **讨论睡眠的重要性**：一位成员提出了一个问题：“**为什么睡眠很重要？**”，随后大家一致认为休息对心理健康是必要的。
   - 团队收到了一个轻松的提醒，鼓励成员们恢复精力，在奉献精神与身心健康之间取得平衡。
- **分享图像工具构思**：成员们交流了关于创建一个新的 image_tool 的想法，该工具可以与模型交互进行多步查询，从而最大限度地利用图像输出。
   - 这将允许模型直接与工具交互，增强处理图像时的响应生成过程。
- **技术故障阻碍进度**：一位成员幽默地报告说，他们的 IDE 在加载 **7.1 万行 JSON** 时崩溃了，导致工作流暂停。
   - 团队对在紧迫的时间节点（如圣诞节目标）下推进开发所面临的挑战付诸一笑。


  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1318696792207921213)** (1 条消息): 

> `Multimodal Embed-v3 图像的 Rate-limit 提升, 试用版与生产版 Rate-limit 对比, API Key 选项与定价` 


- **Multimodal Embed-v3 图像处理能力提升 10 倍！**：根据社区反馈，对于 production keys，**Multimodal Image Embed** 端点的 Rate-limit 已从 **40 images/min** 增加到 **400 images/min**。
   - 试用版 Rate-limit 将保持在 **5 images/min**，以供免费测试。
- **了解试用版和生产版 Rate-limit**：除了 **Embed (Images)** 端点的显著提升外，其他各种端点的具体 Rate-limit 也在提供的图表中进行了详细说明。
   - 例如，**Chat** 端点在试用版中允许 **20 images/min**，在生产版中允许 **500 images/min**，突显了付费购买 production keys 的优势。
- **探索 API Key 和定价详情**：Cohere 提供两种 API Key：免费的 **evaluation keys**，以及付费并提供更高限制的 **production keys**。
   - 开发者可以在 [API keys 页面](https://dashboard.cohere.com/api-keys) 创建和管理他们的 Key，并在 [定价文档](https://docs.cohere.com/v2/docs/how-does-cohere-pricing-work) 中查看定价详情。



**提到的链接**：<a href="https://docs.cohere.com/v2/docs/rate-limits">API Keys and Rate Limits — Cohere</a>：此页面描述了 Cohere API 针对生产版和评估版 Key 的 Rate-limit。

  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1318719316299874397)** (51 messages🔥): 

> `Cohere Reranker 问题, 使用不同的 Embedding 模型, Cohere 与 Nvidia 的依赖关系, AI 系统中的 TPU, 不同维度的 Vector Store` 


- **Cohere Reranker 并不总是能一致地选择相关的 chunk**：一位开发者报告称，与 **ContextualCompressionRetriever** 一起使用的 **Cohere Reranker** 有时无法从检索到的数据中选择最相关的 chunk，导致回答错误。
   - 尽管他们的 RAG 应用中 chunking 很准确，但 **reranking 行为** 是随机的，经常选择相关性较低的 chunk，造成了困扰。
- **关于存储不同维度 embedding 的问题**：一位用户询问，鉴于 **text-3-embedding-large** 和 **cohere embed v3** 的维度分别为 **3072** 和 **1024**，是否应该为它们生成的 embedding 创建独立的 vector stores。
   - 提出这一担忧是因为在整合文本、表格和图像的 embedding 时，维度差异可能会影响存储策略。
- **AI 系统对 Nvidia 产品的依赖**：一位参与者指出，由于 **CUDA** 和 **NCCL** 提供的强大生态系统，**Nvidia** 是大多数 AI 系统的核心组件。
   - 虽然 AMD 和 TPU 是替代方案，但与 Nvidia 在 AI 领域的广泛采用相比，它们被认为更加小众。
- **探索 TPU 在 AI 中的使用**：讨论了 **TPU** 及其作为快速向量处理器的有效性，特别是针对 AI 应用中的**矩阵乘法 (matrix multiplication)**。
   - 虽然 **Anthropic** 大量使用 TPU，但共识是大多数系统仍然主要依赖 **Nvidia**，因为它拥有强大的生态系统。
- **利用多样化的 AI 计算架构**：一位参与者分享了他们过去使用 **FPGA** 进行推理 (inference) 的经验，表明 AI 处理存在多种硬件选项。
   - 讨论强调了需要考虑解决方案的“开箱即用 (turn-key)”程度，权衡替代架构的灵活性与实现的简易性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/docs/structured-outputs">Structured Outputs — Cohere</a>：该页面描述了如何让 Cohere 模型以特定格式（如 JSON）创建输出。</li><li><a href="https://docs.cohere.com/reference/chat#request.body.response_format">Chat — Cohere</a>：对用户消息生成文本响应并逐个 token 进行流式传输。要了解如何使用带有流式传输的 Chat API，请遵循我们的 [Text Generation 指南](https://docs.cohere.com/v2/docs/cha...</li><li><a href="https://docs.cohere.com/reference/chat#request.body.strict_tools">Chat — Cohere</a>：对用户消息生成文本响应并逐个 token 进行流式传输。要了解如何使用带有流式传输的 Chat API，请遵循我们的 [Text Generation 指南](https://docs.cohere.com/v2/docs/cha...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/)** (1 messages): 

setupisanoun: 嘿，伙计
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1318889996584878081)** (2 messages): 

> `Product Hunt 发布, Findr 应用, 数字记忆` 


- **Findr 应用在 Product Hunt 上线**：[Findr](https://www.producthunt.com/posts/findr-remember-everything) 已正式在 Product Hunt 上线，旨在为人类提供**无限记忆**和**可搜索的数字大脑**。
   - 团队正在请求支持，如其宣传 [推文](https://x.com/Nish306/status/1868953328975261712) 所示。
- **来自社区的支持**：包括 @meor.amer 在内的社区成员对 Findr 的发布表示祝贺。
   - 这显示了对该应用创新概念感兴趣的用户的**积极反馈**。



**提到的链接**：<a href="https://x.com/Nish306/status/1868953328975261712">来自 Nishkarsh (@Nish306) 的推文</a>：我们已在 Product Hunt 上线。非常感谢您的支持 https://www.producthunt.com/posts/findr-remember-everything 我们正在赋予人类无限的记忆和可搜索的数字大脑...

  

---

### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1318945686573551689)** (3 messages): 

> `Cohere Toolkit 部署、AWS 流错误、Docker 日志检查` 


- **Cohere Toolkit 已部署但面临问题**：一名成员使用提供的 **AWS 指南** 成功部署了 **Cohere Toolkit**，但遇到了间歇性的 `stream ended unexpectedly` 错误。
   - 这个问题似乎是随机发生的，有时某些消息可以正常工作。
- **寻求流错误的见解**：该成员询问是否有人也遇到过 **stream ended unexpectedly** 错误，目前对于可能导致该问题的原因尚无具体线索。
   - 尽管其他功能有时表现正常，但该问题依然存在，因此请求分享相关经验。
- **建议检查 Docker 日志**：另一名成员建议检查 **docker logs** 以获取有关该错误的更多信息。
   - 这一建议表明，在与部署相关的应用程序日志中可能会发现更深入的见解。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1318669970682679397)** (22 messages🔥): 

> `Mojo 在 Archcraft Linux 上的问题、Max 和 Magic 的安装、使用 Mojo REPL、magic 环境中的 Python 需求` 


- **祝贺新版本发布！**：成员们庆祝新版本发布，并分享了 [Stable Diffusion 的 GitHub 仓库](https://github.com/modularml/max/tree/main/examples/inference/stable-diffusion-mojo-onnx) 中可用的示例。
   - 一名成员在表示祝贺的同时，提供了有助于进一步探索的链接。
- **用户在 Archcraft Linux 上使用 Mojo 遇到困难**：一名用户报告在 Archcraft Linux 上进入 Mojo REPL 时遇到问题，称其无法找到动态链接库，可能名为 **mojo-ldd**。
   - 随后讨论了与链接器 **mojo-lld** 相关的错误及其安装要求。
- **Max 安装过程问题**：另一名成员提到 Max 安装过程意外终止，阻碍了他们的进度。
   - 他们表示尽管可以使用 Max 和 Magic，但在访问 Mojo REPL 时遇到困难。
- **外部管理环境错误**：同一用户表示，在 magic 环境中尝试安装 Python 需求时，遇到了提示其处于 **externally managed environment** 的错误。
   - 他们寻求针对此问题的帮助，表明无法下载必要的依赖需求。
- **问题解决线程的建议**：一名成员建议为问题解决创建新线程，以协助遇到类似问题的其他人。
   - 鼓励采用这种方法，以确保协作解决方案并获得社区的持续协助。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1318676804692873278)** (57 messages🔥🔥): 

> `Mojo 文档更新, Mojo Kernel 术语, 计算 Kernel vs OS Kernel, 关于 var 关键字的讨论, Argmax 和 Argmin 的移除` 


- **澄清 Mojo 文档更新**：一名团队成员讨论了 [Mojo 文档](https://docs.modular.com/mojo/manual/basics#variables) 中关于变量声明的更新，特别是 `var` 关键字的使用。
   - 另一名成员确认他们正在处理关于 `var` 必要性的更新，该工作仍在进行中。
- **理解 Mojo Kernel 术语**：成员们讨论了 Mojo 语境下的 “kernel” 一词，澄清它指的是在加速器上运行的函数，而不是 OS Kernel（操作系统内核）。
   - 一名成员幽默地指出，使用这个术语是为了显得高端，而另一名成员则解释说它是针对硬件优化的特定代码块。
- **计算 Kernel 与 OS Kernel 的区别**：讨论重新定义了计算 Kernel 与 OS Kernel，强调 Mojo 虽然可以用于用户空间驱动程序，但仍需改进。
   - 成员们一致认为，虽然 Mojo 可以帮助解决编译和移植方面的问题，但在达到 OS Kernel 的能力之前还需要更多工作。
- **关于 Mojo 中 `var` 关键字的辩论**：成员们对 `var` 关键字的必要性表达了不同看法，有人建议将其设为可选，但在代码中予以标注。
   - 一名成员讨论了移除 `var` 可能如何影响 structs，而其他人则希望在使用上更加清晰。
- **对移除 Argmax 和 Argmin 函数的担忧**：一名成员询问为何 `argmax` 和 `argmin` 从 algorithm.reduction 中消失了，担心需要从头开始实现它们。
   - 这引发了关于 Mojo 库更新和变化的讨论，表明需要更清晰的 changelog（更新日志）。



**提及的链接**：<a href="https://docs.modular.com/mojo/manual/basics#variables)">Mojo 语言基础 | Modular 文档</a>：Mojo 基础语言特性介绍。

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1318740743908622337)** (13 messages🔥): 

> `Mojo 中的自定义算子 (Custom ops), 错误处理与文档, 自定义算子消息的功能请求, Max GitHub 仓库 Issue, 带有自定义算子的会话加载` 


- **Mojo 中的自定义算子 (Custom ops) 出现问题**：在 Mojo 中加载名为 [mandelbrot](https://github.com/modularml/max/issues/269) 的自定义算子时出现问题，特别是在尝试导入类型时。
   - 错误表明 mandelbrot 自定义算子的 Mojo Kernel 未注册，阻碍了执行。
- **需要更新文档**：成员们讨论了在 Max GitHub 仓库上提交关于错误消息清晰度的 Issue，特别是 “未找到自定义算子” 的消息。
   - 建议包括改进错误消息，并可能引导用户查阅相关文档。
- **改进自定义算子处理的功能请求**：一名成员发起了一项 [功能请求 (Feature Request)](https://github.com/modularml/max/issues/269)，旨在更好地处理未找到的自定义算子并提供更清晰的错误消息。
   - 该请求旨在通过在发生错误时引导用户查阅文档来解决用户体验问题。
- **确认 Bug 报告**：成员们对发现问题表示感谢，并确认他们将提交 Bug 报告，特别是针对自定义操作（Custom operations）。
   - 为了解决 Mojo 处理自定义算子的现有问题，关于 GitHub Issue 的清晰沟通是必要的。
- **会话加载困惑**：讨论围绕使用自定义算子加载会话展开，特别是使用指向 Kernel 目录的路径。
   - 一名成员提到，在调用 `session.load` 并使用 `custom_ops_paths` 参数时，应强调具体细节以确保清晰。



**提及的链接**：<a href="https://github.com/modularml/max/issues/269">[功能请求] 单一编译单元 Kernel 和/或改进的错误消息 · Issue #269 · modularml/max</a>：您的请求是什么？这是一个由两部分组成的请求，但捆绑在一起，因为它们都解决了同一个用户体验问题。第一部分是使 “未找到自定义算子” 错误消息引导用户查阅文档...

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1318671414433284116)** (28 条消息🔥): 

> `Open Interpreter 错误, 最新 OI 版本, AI 应用与模型, Truffle-1 计算栈, OI 中的长期记忆` 


- **持续的 Open Interpreter 错误**：多位用户报告了在使用 Open Interpreter 时遇到的持续问题，特别是涉及与 `--conversations` 命令相关的错误。
   - 一位成员对丢失宝贵的对话表示沮丧，并提出了如何解决这些持续存在的问题。
- **关于最新 OI 版本的咨询**：一位用户对升级到 Open Interpreter 1.x 版本感到好奇，提到他们仍在使用 0.34 版本，并听说有了更早的 1.0 发布版。
   - 讨论内容包括最新版本中是否提供 OS mode，成员们正在策划改进方案。
- **探索 AI 应用与模型**：讨论涉及将 AI 用于各种应用，包括 Raspberry Pi 设置以及用于家庭自动化的潜在语音转语音模型。
   - 用户们在思考如何将较小的模型与更大、功能更强的系统连接起来，以增强功能。
- **Truffle-1 AI 计算机介绍**：一位成员分享了关于 Truffle-1 的细节，这是一个运行多个模型的个人计算栈，拥有 64GB unified memory，价格为 500 美元定金加每月 115 美元。
   - 这款个人 Agentic 计算设备旨在提供无限的推理时间，并支持编写和分享应用，设备将于 1 月份发货。
- **本地使用 OS Mode**：一位用户询问是否可以在本地使用 Open Interpreter 的 OS mode。
   - 这引发了关于为遇到问题的用户提供可用配置选项的进一步讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://itsalltruffles.com">Truffle</a>: 一个个人 Agentic 计算栈 —— Truffle-1 在设备上运行混合模型，配备 64GB unified memory</li><li><a href="https://x.com/iamgingertrash/status/1869450385896751449">来自 simp 4 satoshi (@iamgingertrash) 的推文</a>: 简要回顾：&gt; 今日授权 500 美元定金 &gt; 12 个月每月 115 美元 &gt; 无限推理时间计算 &gt; 编写与分享你自己的应用 &gt; 一个带有 64GB Orin 的发光球体 &gt; 我们真的...</li><li><a href="https://tenor.com/view/first-of-all-all-things-are-possible-jot-that-down-pointing-serious-gif-14586817">First Of All All Things Are Possible GIF - First Of All All Things Are Possible Jot That Down - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1318864276387139694)** (27 条消息🔥): 

> `Llama 模型基准测试，ShapeTrackers 中的可合并性，CuTe 中的布局代数，合并中的算法复杂度，布局代数中的单射性` 


- **Llama 模型基准测试请求**：一名成员询问是否有人拥有 Llama 模型的基准测试数据，用于对比 **TinyGrad** 的 OpenCL 实现与 **PyTorch** 的 CUDA 实现。
   - 这突显了人们对 AI 框架之间**性能比较**的持续关注。
- **ShapeTrackers 可合并性的挑战**：一位用户讨论了在 Lean 中证明两个任意 **ShapeTrackers** 可合并性的复杂性，并表示不可能存在类似于矩阵行列式的简单判定标准。
   - 他们强调了步长（strides）和形状（shapes）中存在的巧合，这使得可合并性检查变得复杂。
- **关于 CuTe 布局代数的见解**：探讨了可合并性是否等同于 **CuTe 布局代数**中的复合（composition），并参考了一篇关于布局操作的学术笔记。
   - 此次讨论涉及了 NVIDIA **CUTLASS** 库中的基本抽象以及布局操作的数学处理。
- **布局代数的复杂度**：有人对证明布局代数中与单射性（injectivity）相关的条件表示担忧，并暗示此类检查可能是 **NP hard** 的。
   - 这强调了由于潜在的步长干扰，在布局代数中建立充分条件的困难。
- **符号函数 vs 布局**：一名成员指出，在检查必要性和充分性方面，**符号整数函数**（symbolic integer functions）严格来说比布局更强大。
   - 这与关于合并视图中算法复杂度的讨论相一致，并支持了当前的研究方向。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/">A note on the algebra of CuTe Layouts</a>: NVIDIA CUTLASS 库高性能线性代数的核心抽象是 CuTe Layout。在这篇技术笔记中，我们对这些布局的代数进行了严谨的数学处理……</li><li><a href="https://github.com/tinygrad/tinygrad/issues/8194),">Issues · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - Issues · tinygrad/tinygrad</li><li><a href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md">cutlass/media/docs/cute/02_layout_algebra.md at main · NVIDIA/cutlass</a>: 用于线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账号为 NVIDIA/cutlass 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1318727531725197412)** (25 messages🔥): 

> `FSDP 归一化, 损失计算中的缩放因子, trl 和 HF trainer 的 Bug 报告, 带有 weight decay 的优化器行为, PR 更新` 


- **FSDP 归一化需要缩放**：讨论揭示了必须处理 FSDP 按 `world_size` 进行的归一化问题，通过 `world_size` 进行缩放可以修正平均操作中的错误。
   - 一名成员建议开启一个 PR 来实现此修复，因为这不需要大规模改动，主要围绕 `scale_grads` 函数进行。
- **训练中倾向于显式缩放**：社区强调了在 Recipe 内部对 loss 进行显式缩放的重要性，而不是将逻辑隐藏在其他地方，以便于理解。
   - 经过评估，成员们同意在训练和优化钩子（optimization hooks）中进一步明确缩放过程。
- **识别跨框架的 Bug**：有人指出，影响 `1/world_size` 归约（reduction）因子的类似 Bug 可能存在于多个库中，包括 `trl` 和 Hugging Face 的 trainer。
   - 成员们赞扬了 HF 团队在其训练框架中识别并解决这些问题的做法，正如相关的 GitHub issues 中所述。
- **处理 No Sync 场景**：成员们讨论了 Hugging Face 如何通过避免梯度累积（grad accumulation）归一化同时正确计算 loss 来处理 no sync 场景。
   - 提供了具体的链接，详细说明了他们获取 batch 中 item 数量的方法，以促进准确的 loss 归一化。
- **PR 的更新内容**：一名成员确认在现有的 PR 中为 `optimizer_in_bwd` 情况添加了缩放因子，以解决潜在问题。
   - 该功能非常重要，因为它调整了优化器应用 weight decay 的方式，并确保在特定情况下能更好地处理梯度。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/3518492f43a8a5a462cbd604be4101268ff5bd52/recipes/full_finetune_distributed.py#L768">torchtune/recipes/full_finetune_distributed.py at 3518492f43a8a5a462cbd604be4101268ff5bd52 · pytorch/torchtune</a>: PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2172">Fix gradient scaling to account for world_size normalization by mirceamironenco · Pull Request #2172 · pytorch/torchtune</a>: 背景：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档还是其他（请在此添加）？请链接此 PR 解决的任何 issue。Changelog...</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/training/memory.py#L219">torchtune/torchtune/training/memory.py at main · pytorch/torchtune</a>: PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/issues/34242">Add DDP token averaging for equivalent non-parallel training similar to #34191 · Issue #34242 · huggingface/transformers</a>: 功能请求：梯度累积中的 Token 平均已在 #34191 中修复。但 DDP 中的 Token 平均似乎存在相同问题。预期行为：所有参与 loss 计算的 token...</li><li><a href="https://github.com/pytorch/torchtune/blob/3518492f4">GitHub - pytorch/torchtune at 3518492f43a8a5a462cbd604be4101268ff5bd52</a>: PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3662">transformers/src/transformers/trainer.py at 052e652d6d53c2b26ffde87e039b723949a53493 · huggingface/transformers</a>: 🤗 Transformers: 适用于 Pytorch, TensorFlow 和 JAX 的先进机器学习库。 - huggingface/transformers
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1318872136684933151)** (2 messages): 

> `进化算法 (Evolutionary Algorithms), 规模化进化 (Scale Up Evolution), 梯度技术 (Gradient Techniques)` 


- **进化算法成为焦点**：一名成员指出**进化算法**的有趣应用，展示了它们在当前讨论中的重要性。
   - 这种潜在的创新邀请人们进一步探索其在机器学习中的应用。
- **Sakana 旨在与梯度技术竞争**：**Sakana** 正尝试扩大其进化方法的规模，以保持与主流**梯度技术**的竞争力。
   - 这一举动表明社区对替代优化策略的兴趣日益浓厚。


  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

合作视频：https://youtu.be/BrvVheleOqc
  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1318687341027655800)** (4 messages): 

> `AI 与知识经济，Coconut - 连续思维链，自主 vs 非自主 AI` 


- **AI 重塑知识经济**：[这篇论文](https://arxiv.org/abs/2312.05481) 介绍了一个分析 AI 如何通过在“工人”和“解决者”之间重新分配角色来转变知识经济的框架。研究强调，基础的自主 AI 会取代人类，而先进的自主 AI 则会催生规模更大、生产力更高的公司。
   - 随着自主 Agent 的普及，它们主要使知识最渊博的个人受益，让他们能够高效管理常规工作；而知识较少的人则从聊天机器人等非自主 AI 中受益。
- **介绍 Coconut - 连续思维**：[Meta 最近的一篇论文](https://arxiv.org/html/2412.06769v1) 提出了一种名为 Coconut 的新推理范式，它使用 LLM 的最后一个隐藏状态（hidden state）进行推理，而不是传统的语言空间。作者认为传统方法可能无法有效捕捉推理过程，并引入了“连续思维（continuous thought）”的概念。
   - 这种方法试图通过探索不受限制的潜空间（latent spaces）来克服基于语言推理的局限性，从而增强 LLM 在复杂推理任务上的表现。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2312.05481">知识经济中的人工智能</a>：人工智能（AI）的兴起有潜力通过实现大规模问题解决来重塑知识经济。本文介绍了一个框架来分析这种转变...</li><li><a href="https://arxiv.org/html/2412.06769v1">训练大语言模型在连续潜空间中进行推理</a>：未找到描述
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1318971783793344573)** (11 messages🔥): 

> `TypedReAct 集成，RouteLLM 维护担忧，DSPy 随推理模型的发展` 


- **TypedReAct 谜题解决**：一位成员分享了 `TypedReAct` 的新实现，询问是否提交 PR，但注意到在未来版本中 `TypedChainOfThought` 可能存在弃用问题。
   - 另一位成员建议，只需移除“Typed”前缀即可解决兼容性问题，并强调内置的 ReAct 在没有类型定义的情况下也很有效。
- **RouteLLM 的长期停滞**：一位成员对 RouteLLM 缺乏维护表示担忧，并表示对潜在的 DSPy 集成感兴趣。
   - 对话强调了支持监管较少的模型开发是多么关键。
- **讨论推理模型背景下 DSPy 的未来**：一位成员询问了关于 DSPy 如何随着推理模型的兴起而演进的讨论，强调了在分支层级的微调（fine-tuning）。
   - 这一观点将焦点从传统的 prompting 方法转向过程奖励机制（process reward mechanisms），预示着模型训练可能发生范式转移。



**提到的链接**：<a href="https://dspy.ai/tutorials/agents/">Agents - DSPy</a>：用于编程（而非提示）语言模型的框架。

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1318701547609260185)** (12 messages🔥): 

> `GPT4All 问题，Jinja 模板功能，Docker 版 GPT4All，命令行界面担忧，CLI 中的本地文档` 


- **GPT4All 在 Jinja 模板上遇到困难**：用户对 GPT4All 在侧向加载（side loading）方面“完全崩溃”表示沮丧，理由是 Jinja 模板对模型功能至关重要，但目前存在问题。
   - 已确定的当前问题包括需要正确设置元素间距、修复换行符问题，以及不支持“none”和“[1:]”等函数。
- **对 Docker 版 GPT4All 的需求**：一位用户询问是否有可以从 Docker 容器运行并带有 Web UI 的 GPT4All 版本，表明对更简便部署方案的兴趣。
   - 社区目前尚未针对此请求提供具体的资源或现有解决方案。
- **不带 localdocs 的 CLI 交互**：一位用户尝试通过命令行访问 GPT4All 模型，但在当前设置下无法使用本地文档（localdocs）。
   - 另一位用户告知他们，旧的 CLI 已不再受官方支持，但如果 GUI 中启用了 localdocs，服务器 API 允许以编程方式访问。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1318705846204633099)** (2 条消息): 

> `AI SDR, Agent 构建速成课程, LlamaIndex Function Calling, Agentic RAG, ReAct` 


- **AI SDR 使用 LlamaIndex 生成潜在客户**: 看看这个为你生成线索的 [agentic AI SDR](https://t.co/tczv5ZDI4H)，它是使用 **LlamaIndex** 构建的。
   - 该工具因其在自动化潜在客户生成方面的能力而受到关注，并链接到了多个 [GitHub features](https://github.com/features)。
- **从零开始学习构建 Agent**: 来自 [@TRJ_0751](https://twitter.com/llama_index/status/1869454248620269615) 的速成课程教授如何使用 **LlamaIndex** 构建 Agent，重点是使用 Function Calling 来管理实时数据查询。
   - 参与者还将学习如何创建一个能在向量工具和摘要工具之间智能路由的 **agentic RAG**，以及如何创建 ReAct。



**提到的链接**: <a href="https://t.co/tczv5ZDI4H">composio/python/examples/quickstarters at master · ComposioHQ/composio</a>: Composio 通过 Function Calling 为你的 AI Agent 和 LLM 提供 100 多个高质量集成 - ComposioHQ/composio

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1318816823642292254)** (4 条消息): 

> `OpenAIAgent 并发, RAG 评估讨论` 


- **OpenAIAgent 执行可能不是并发的**: 一位成员询问 `OpenAIAgent` 的函数执行是否可以在异步环境中并发完成，并指出这与并行 Function Calling 不同。
   - 对此进行的调查发现，即使针对异步进行了修改，函数执行仍然是非并发的。
- **利用异步工具进行函数执行**: 另一位成员建议使用异步入口点和异步工具，并表示这种方法应该能确保正确的执行。
   - 他们提供了一个代码片段，演示了如何使用 `OpenAIAgent` 异步实现工具。
- **寻找 RAG 评估讨论**: 一位成员表示有兴趣讨论 RAG 评估，并邀请其他人如果想聊聊可以私信。
   - 这表明在 AI 社区中，人们正在持续努力与同行就评估策略进行交流。



**提到的链接**: <a href="https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_parallel_function_calling/#example-from-openai-docs">Single-Turn Multi-Function Calling OpenAI Agents - LlamaIndex</a>: 未找到描述

  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1318697465855086655)** (3 条消息): 

> `BFCL 排行榜, Function Call Demo 问题, 用于结构化输出的 Gorilla 基准测试` 


- **BFCL 排行榜功能受到质疑**: 一位用户指出 [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard) 上的函数调用 Demo 存在问题，称其卡在“Loading Model Response...”状态。
   - 作为回应，另一位成员确认存在 **证书问题**，导致模型端点宕机。
- **关于结构化输出评估的咨询**: 一位用户表示有兴趣使用 **Gorilla benchmark** 来评估模型结构化输出的质量。
   - 他们特别询问是否有专门用于根据提供的 **JSON schema** 或 **Pydantic model** 生成文本的子任务。



**提到的链接**: <a href="https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard">
        Berkeley Function Calling Leaderboard V3 (又名 Berkeley Tool Calling Leaderboard V3)
    </a>: 未找到描述

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/)** (1 条消息): 

kallemickelborg: 谢谢你！
  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1318904347194556490)** (1 条消息): 

> `新工程师入职, Reinforcement Learning 支持` 


- **新工程师将于 1 月加入**: 一位新工程师定于 **1 月**加入，负责协助通用的 **Reinforcement Learning** 工作。
   - 届时他们还将为 **kto** 项目提供支持。
- **对 RL 和 kto 的支持**: 新工程师的专业知识将增强团队在 **Reinforcement Learning** 方面的能力。
   - 预计他们的协助也将对 **kto** 的开发产生积极影响。

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1318691196566241371)** (1 条消息): 

> `Developer Hub, Blueprints Initiative` 


- **Developer Hub 更新发布**：发布了关于 **Developer Hub** 的重大更新，详细介绍了改进和新功能。您可以在[此处](https://discord.com/channels/1089876418936180786/1230938514955436242/1318638353503227935)查看完整公告。
   - 在努力提升用户体验的过程中，非常感谢来自社区的反馈。
- **开源 AI 解决方案的 Blueprints**：分享了一个讨论 **Blueprints initiative** 的帖子，旨在帮助开发者创建开源 AI 解决方案。更多详情可以在[该帖子](https://discord.com/channels/1089876418936180786/1318689803021058158)中找到。
   - 该倡议被定位为开发者有效启动项目的资源。


  

---


---


---


{% else %}


> 为了邮件展示，完整的频道细分内容已被截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}