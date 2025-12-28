---
companies:
- openai
- lmsys
- scale-ai
- cognition
- langchain
- qdrant
- rohanpaul_ai
date: '2024-09-18T00:27:31.736910Z'
description: '以下是该文本的中文翻译：


  **OpenAI 的 o1 模型**由于其极端的限制以及在思维链（CoT）上进行强化学习（RL）等独特的训练进展，在开源复制方面面临质疑。**ChatGPT-4o**
  在各项基准测试中表现出显著的性能提升。**Llama-3.1-405b** 的 fp8 和 bf16 版本性能表现相似，但 fp8 具有成本优势。一个新的开源基准测试
  **“人类最后的考试”（Humanity''s Last Exam）** 提供了 50 万美元的奖金来挑战大语言模型（LLM）。**模型合并（Model merging）**
  受益于神经网络的稀疏性和线性模式连接性（linear mode connectivity）。**基于嵌入（Embedding）的有毒提示词检测**以较低的计算量实现了高准确率。**InstantDrag**
  实现了快速、无需优化的拖拽式图像编辑。**LangChain v0.3** 发布，改进了依赖管理。自动化代码审查工具 **CodeRabbit** 能够适应团队的编码风格。**视觉搜索**的进展整合了多模态数据，以实现更好的产品搜索。专家预测，到
  2030 年，**AI 将成为软件的默认配置**。'
id: e94eef2f-48e7-4416-b4e9-429de3b66f45
models:
- o1
- chatgpt-4o
- llama-3-1-405b
original_slug: ainews-nothing-much-happened-today-7147
people:
- denny_zhou
- svpino
- alexandr_wang
- cwolferesearch
- rohanpaul_ai
- _akhaliq
- kylebrussell
title: 今天没发生什么特别的事。
topics:
- reinforcement-learning
- model-merging
- embedding-models
- toxicity-detection
- image-editing
- dependency-management
- automated-code-review
- visual-search
- benchmarking
---

<!-- buttondown-editor-mode: plaintext -->**宁静是你唯一需要的。**

> 2024/9/16-2024/9/17 的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**221** 个频道，**2197** 条消息）。预计节省阅读时间（以 200wpm 计算）：**225 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

鉴于围绕 o1 的[极端限制、成本以及缺乏透明度](https://news.ycombinator.com/item?id=41534474)，每个人对于 o1 是否能在开源领域或野外被复制都各有各的看法。正如 [/r/localLlama 中讨论的](https://www.reddit.com//r/LocalLLaMA/comments/1fiadsy/will_an_open_source_model_beat_o1_by_the_end_of/)，Manifold 预测市场目前认为开源版本出现的概率为 63%：


![image.png](https://assets.buttondown.email/images/666119c8-25fd-4bf0-8292-a74b238961ab.png?w=960&fit=max)


与此同时，以下情况都有可能：

- o1 的许多方面都可以在开源中复制，特别是如果拥有 OpenAssistant 级别的众包推理轨迹（reasoning trace）数据集。
- 也许人们一直在传阅的一些 MCST 论文是相关的，但也可能不相关。
- 在训练层面实现的真正的 [RL on CoT](https://x.com/wgussml/status/1834691198013129053) 进展，是任何程度的数据集修补（futzing）都无法企及的。

仅凭最后一个原因，模型开发中标准的“达到 OSS 等效水平的时间”曲线在此案例中可能并不适用。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型更新与进展**

- **OpenAI 的 o1 模型**：[@denny_zhou](https://twitter.com/denny_zhou/status/1835761801453306089) 强调，理论上只要有足够的中间推理 Token，即使深度恒定，Transformer 也可以解决任何问题。这表明扩展 LLM 推理性能具有巨大潜力。

- **性能提升**：[@lmsysorg](https://twitter.com/lmsysorg/status/1835825082280902829) 报告了 ChatGPT-4o (20240903) 在各项基准测试中的显著提升，包括整体性能、风格控制、困难提示词（hard prompts）以及多轮交互。

- **模型对比**：[@lmsysorg](https://twitter.com/lmsysorg/status/1835760196758728898) 对比了 Llama-3.1-405b 的 bf16 和 fp8 版本，发现在各类别中性能相近，fp8 在显著降低成本的同时，表现与 bf16 非常接近。

- **新兴能力**：[@svpino](https://twitter.com/svpino/status/1835740534729830800) 讨论了 GPT-4o 在系统 1（System 1）思维和 OpenAI o1 在系统 2（System 2）思维方面的专业化，并预见未来模型将在单一框架下融合两者。

**AI 开发与研究**

- **评估挑战**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1835738937719140440) 宣布 Scale 与 CAIS 合作推出“人类最后的考试”（Humanity's Last Exam），这是一个极具挑战性的 LLM 开源基准测试，为最佳问题提供 50 万美元奖金。

- **模型合并**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1835748003128193470) 解释了模型合并（model merging）的有效性，将其成功归功于神经网络中的线性模式连接（linear mode connectivity）和稀疏性。

- **AI 安全**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1835739851452510258) 分享了关于基于 Embedding 的有毒提示词检测的见解，以极低的计算开销实现了高准确率。

- **多模态能力**：[@_akhaliq](https://twitter.com/_akhaliq/status/1835677372344873377) 介绍了 InstantDrag，这是一个无需优化的拖拽式图像编辑流水线，增强了图像处理任务中的交互性和速度。

**AI 工具与应用**

- **LangChain 更新**：[@LangChainAI](https://twitter.com/LangChainAI/status/1835720923414184128) 宣布发布适用于 Python 和 JavaScript 的 LangChain v0.3，重点在于改进依赖项并转向对等依赖（peer dependencies）。

- **AI 在代码审查中的应用**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1835664732390351085) 讨论了使用 CodeRabbit 进行自动化代码审查，强调其能够适应团队编码习惯并提供定制化反馈。

- **AI 在产品搜索中的应用**：[@qdrant_engine](https://twitter.com/qdrant_engine/status/1835634931977892120) 分享了视觉搜索解决方案的进展，将图像、文本和其他数据集成到统一的向量表示中，以提升产品搜索体验。

**行业趋势与观察**

- **AI 集成**：[@kylebrussell](https://twitter.com/kylebrussell/status/1835706377785798694) 预测到 2030 年，AI 将成为默认配置，软件将实现自我生成，而 Agent 将成为新的应用形式。

- **开源进展**：[@far__el](https://twitter.com/far__el/status/1835791026034036845) 暗示了开源 AI 模型的最新进展，表明其可能与闭源模型展开竞争。

- **AI 在时尚界的应用**：[@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1835745610919706685) 展示了 AI 生成的时尚模特，暗示品牌营销策略可能发生转变。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 模型压缩与量化方面的进展**

- **[LMSYS 发现 Chatbot Arena 中 bf16 和 fp8 版本的 Llama-3.1-405b 差异极小](https://x.com/lmsysorg/status/1835760196758728898)** ([Score: 109, Comments: 34](https://reddit.com//r/LocalLLaMA/comments/1fil2an/lmsys_finds_minimal_differences_between_bf16_and/)): LMSYS 在其 **Chatbot Arena** 中对 **Llama-3.1-405b** 的 **bf16** 和 **fp8** 版本进行了对比，发现两者在性能上的差异微乎其微。**fp8** 模型的胜率仅比 **bf16** 版本下降了 **0.3%**，这表明 **fp8 quantization** 可以在显著减小模型体积和显存需求的同时，对质量的影响几乎可以忽略不计。
  - 用户反映不同量化版本在**编程性能方面存在显著差异**，一些人指出 **fp8** 在编程任务中的表现不如 **q8**。**Aidan McLau** 的一条推文批评了 **LMSYS** 的评估方法，认为 **bf16** 在特定提示词下表现更优。
  - 讨论强调了像 **LMSYS leaderboard** 这种**基于人类感知的评估方式的局限性**。一些用户观察到 **q8** 和 **fp16** 在编程方面的差异很小，而另一些用户则在基准测试中报告了相互矛盾的结果。
  - 几条评论称赞了 **quantization** 技术，一位用户成功地将 **Llama 3.1 70b** 的 **IQ2_M 版本** 用于编程任务。争论延伸到了各种量化级别（**q6_k**, **q4km**）之间的比较及其对模型性能的影响。


- **发布采用 AQLM-PV 压缩的 Llama3.1-70B 权重。** ([Score: 249, Comments: 81](https://reddit.com//r/LocalLLaMA/comments/1fiscnl/release_of_llama3170b_weights_with_aqlmpv/)): **Llama3.1-70B** 和 **Llama3.1-70B-Instruct** 模型已使用 **AQLM+PV-tuning** 进行压缩，将其体积减小至 **22GB**，从而能够在单张 **3090 GPU** 上运行。这种压缩导致 **MMLU 性能下降了 4-5 个百分点**，基础模型的得分从 **0.78 降至 0.73**，指令微调模型的得分从 **0.82 降至 0.78**。此外，还发布了一个压缩版的 **Llama3.1-8B** 模型，它可以作为 Android 应用运行，仅需 **2.5GB RAM**。
  - 压缩后的 **Llama3.1-70B** 模型与 **IQ_2M** 量化类似，具有相当的 **22GB** 体积和 **MMLU** 分数。用户讨论了运行方法，包括 **Transformers**, **vLLM** 和 **Aphrodite**，一些人在实现过程中遇到了挑战。
  - 人们对压缩更大的模型（如 **405B 版本** 和 **Gemma-2 27B**）表现出浓厚兴趣。用户推测了潜在的体积以及与特定硬件（如配备 **128GB** RAM 的 **M3 Max**）的兼容性。
  - **AQLM** 量化方法作为一个 [开源项目](https://github.com/Vahe1994/AQLM) 提供，但目前不支持 **GGUF** 格式。用户报告推理速度较慢，在 **3090 GPU** 上约为 **7 tokens/秒**。

- **[Hugging Face 优化了 Segment Anything 2 (SAM 2)，可在设备端（Mac/ iPhone）实现亚秒级推理！](https://v.redd.it/4ndo0w4sf7pd1)** ([Score: 83, Comments: 14](https://reddit.com//r/LocalLLaMA/comments/1fiab07/hugging_face_optimised_segment_anything_2_sam_2/)): **Hugging Face** 针对设备端推理优化了 **Segment Anything 2 (SAM 2)**，使其能够在 **Mac 和 iPhone** 上以**亚秒级**性能运行。这种优化实现了移动设备上的实时分割任务，可能为增强现实、图像编辑和边缘设备上的计算机视觉开启新的应用。
  - **Hugging Face** 正在发布各种尺寸的 SAM 2 **Apache 许可证优化模型权重检查点**，以及一个用于亚秒级图像标注的 [开源应用](https://github.com/huggingface/sam2-studio)。他们还为 **Medical SAM** 等 SAM2 微调模型提供转换指南。
  - 开发者计划增加**视频支持**，并对未来功能的建议持开放态度。这表明 SAM 2 优化项目正在持续开发中，并具有扩展能力的潜力。
  - 用户对 **Apple** 优化其他模型表示出兴趣，特别是提到了 **GroundingDino**。这表明对更多针对 Apple 硬件优化的端侧 AI 模型存在需求。


**主题 2. 开源 LLM 正在缩小与闭源模型的差距**

- **开源模型会在 2025 年 Q1 结束前击败 o1 吗？** ([Score: 111, Comments: 52](https://reddit.com//r/LocalLLaMA/comments/1fiadsy/will_an_open_source_model_beat_o1_by_the_end_of/)): 该帖子推测 **开源语言模型** 是否能通过使用 **"System 2" 风格** 的方法（如 **Monte Carlo Tree Search (MCTS)** 和 **reflection**），在 **2025 年 Q1** 之前超越 **OpenAI** 的 **GPT-4**（此处指代 "o1"）。作者引用了 **Noam Brown** 的工作，并创建了一个 [Manifold market](https://manifold.markets/JohnL/by-the-end-of-q1-2025-will-an-open?r=Sm9obkw) 来衡量公众对这一可能性的看法。
  - **开源模型** 有可能在 **2025 年 Q1** 达到 **GPT-4** 的性能水平，用户提到了 **Claude 3.5** 的显著进步，以及 **reflection** 和 **thinking magic** 进一步增强开源模型的潜力。
  - 对 **GPT-4** 架构的推测表明，它可能是一项 **工程成就** 而非一个新模型，可能使用了 **微调后的现有模型**、巧妙的 prompting 以及一个 **"critic" LLM** 来对回答进行评估。
  - 关于时间线的观点各不相同，一些人认为 **开源模型** 可能在 2025 年底超越 **GPT-4**，而另一些人指出 **OpenAI** 可能会进一步改进其模型，保持对开源替代方案的领先地位。


- **发布采用 AQLM-PV 压缩的 Llama3.1-70B 权重。** ([Score: 249, Comments: 81](https://reddit.com//r/LocalLLaMA/comments/1fiscnl/release_of_llama3170b_weights_with_aqlmpv/)): **Llama3.1-70B** 和 **Llama3.1-70B-Instruct** 模型已使用 **AQLM+PV-tuning** 进行压缩，将其大小减小至 **22GB**，使其能够在单个 **3090 GPU** 上运行。压缩导致 **MMLU 性能** 下降了 **4-5 个百分点**，基础模型的得分从 **0.78 降至 0.73**，指令模型的得分从 **0.82 降至 0.78**。此外，还发布了一个压缩后的 **Llama3.1-8B** 模型，该模型已作为 [Android app](https://blacksamorez.substack.com/p/aqlm-executorch-android?r=49hqp1&utm_campaign=post&utm_medium=web&triedRedirect=true) 运行，仅需 **2.5GB RAM**。
  - 用户将 **AQLM+PV-tuning** 与 **IQ_2M** 量化进行了比较，注意到两者在 **22GB** 大小和 **MMLU 得分** 上表现相似。该模型的 **chat template** 已修复，以提高与 **vLLM** 和 **Aphrodite** 的兼容性。
  - 由于尺寸限制，在 **16GB VRAM** 系统上运行该模型被证明具有挑战性。**70B 模型** 仅权重就需要至少 **17.5GB**，此外还需要额外的内存用于缓存和 embeddings。
  - 用户表示有兴趣将 AQLM 压缩应用于其他模型，如 **Gemma-2 27B** 和 **Mixtral**。[AQLM GitHub 仓库](https://github.com/Vahe1994/AQLM) 已分享给那些有兴趣量化自己模型的人。
- **创建开源 o1 模型似乎指日可待！** ([Score: 173, Comments: 55](https://reddit.com//r/LocalLLaMA/comments/1fim224/there_seems_to_be_promise_in_creating_an/)): 作者报告了在创建一个 **开源类 o1 模型** 方面取得的 **令人鼓舞的结果**，该模型使用了在 **370 行的小型数据集** 上微调的 **Q4_K_M 8B 模型**。他们提供了 [模型](https://huggingface.co/Lyte/Llama-3.1-8B-Instruct-Reasoner-1o1_v0.3)、[演示](https://huggingface.co/spaces/Lyte/Llama-3.1-8B-Instruct-Reasoner-1o1_v0.3-Q4_K_M) 和用于微调的 [数据集](https://huggingface.co/datasets/Lyte/Reasoner-1o1-v0.3-HQ) 的链接，强调了 **GPU 受限的用户** 很快就能获得类似模型的潜力。
  - 用户将该项目与 **Matt 的 o1 实验** 进行了比较，指出这次尝试确实产生了结果。作者澄清说他们并不是在声称这是一个 **SOTA 模型**，只是分享一个有趣的实验。
  - 讨论集中在需要实现 **reinforcement learning** 以完全复制 o1 的方法。一些人推测 o1 使用 RL 来为 **chain-of-thought** 过程寻找最佳措辞和句法结构。
  - 几条评论建议运行 **热门基准测试** 以证明可信度并比较结果。作者已将模型提交至 **open llm leaderboard** 进行评估，并承认由于数据集较小和 GPU 限制而存在的局限性。


**主题 3：LLM 推理（Reasoning）与推理（Inference）技术的发展**

- **o1-preview: 一个擅长数学和推理、编程表现平平、写作表现较差的模型** ([Score: 87, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1ficb0z/o1preview_a_model_great_at_math_and_reasonong/)): **o1-preview 模型**在**复杂推理**、**数学**和**科学**方面展现了卓越的能力，在处理挑战性提示词的 single-shot 响应中优于其他模型。然而，它在**创意写作**方面表现不佳，在**编程**方面表现平平；由于更好的**推理速度（inference speed）**和准确性权衡，作者在编程任务中更倾向于使用 **Sonnet 3.5**。尽管推理步骤有时不一致，该模型偶尔仍能提供正确答案。虽然它代表了重大进步，但在推理或数学方面尚未达到 **Ph.D. level**。

- **论文：Chain of Thought 赋能 Transformers 解决固有的串行问题** ([Score: 136, Comments: 27](https://reddit.com//r/LocalLLaMA/comments/1fiftvc/paper_chain_of_thought_empowers_transformers_to/)): 来自 **Google DeepMind** 的 **Denny Zhou** 声称，正如他们的论文所证明的那样，**Large Language Models (LLMs)** 在扩展推理时没有性能限制。研究表明，只要能生成足够的中间推理 token，**Transformers** 就可以解决任何具有**固定深度（constant depth）**的问题，详情见 [arXiv](https://arxiv.org/abs/2402.12875) 上的论文。

- **推理阶段 LLM “推理”策略的圣杯** ([Score: 39, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1fimvl6/the_holy_grail_of_llm_reasoning_tactics_during/)): 该帖子重点介绍了一个 **GitHub 仓库**，该仓库汇编了受近期 **Reflection models** 及其扩展启发的各种推理阶段使用的 **LLM “推理”策略**。该仓库由第三方创建，地址为 **[https://github.com/codelion/optillm](https://github.com/codelion/optillm)**，提供了一个**即插即用的 API（drop-in API）**，用于测试不同的推理或“思考”方法，并可适配各种本地模型提供商。
  - 用户对该仓库表示了兴趣，其中一位指出这些**进步超越了常规的微调算法（fine-tuning algorithms）**。讨论了该仓库与**本地服务器**的兼容性，并确认已成功与 **oobaboogas textgen** 集成。
  - 该仓库作为一个**透明的 OpenAI API 兼容代理**运行，允许与各种工具和框架集成。可以通过在本地服务器中设置 **base_url** 来使用该代理。
  - 与 **Patchwork** 的集成相比基础模型带来了**显著的性能提升**。有关此集成的详细信息可以在 [仓库的 README](https://github.com/codelion/optillm?tab=readme-ov-file#patchwork-with-optillm) 和 [wiki](https://github.com/codelion/optillm/wiki/Patchwork) 中找到。


**主题 4. LLM 评估与可靠性方面的挑战**



- **作为一个热衷于 LLM 工作流的人，我发现很难信任 o1 的输出** ([Score: 35, Comments: 9](https://reddit.com//r/LocalLLaMA/comments/1fid8z5/as_someone_who_is_passionate_about_workflows_in/)): 该帖子批评了 **o1 在处理复杂任务（尤其是编程场景）时的输出和工作流方法**。作者热衷于 **LLM workflows**，他观察到 o1 的输出更像是一种工作流结构，而非标准的 Chain of Thought，这可能导致一些问题，例如 LLM 在简单问题上**陷入逻辑死胡同**，或者通过多个处理步骤丢失功能从而**弄乱 Python 方法**。帖子认为针对不同类型的任务（如推理 vs 编程）采用**定制化工作流**至关重要，并暗示 o1 目前对所有任务使用单一工作流的方法可能会有问题，特别是对于复杂的开发工作，这导致作者在编程任务中仍然首选 **ChatGPT 4o**。

- **新模型可识别并移除数据集中的废话（Slop）** ([Score: 68, Comments: 18](https://reddit.com//r/LocalLLaMA/comments/1fidhib/new_model_identifies_and_removes_slop_from/)): **Exllama 社区**开发了一个模型，用于识别和移除公共数据集（包括 **HuggingFace** 上的数据集）中的**“废话（slop）”**和**说教（moralization）**。这一突破允许检测**企业废话（corporate slop）**、对废话类型进行分类以及分析低质量数据轨迹，从而可能提高 **LLM 的对话能力**并理解提示词拒绝模式。有关该项目的更多信息可以在 [Exllama Discord](https://discord.gg/m5yEPEwK) 服务器上找到，感兴趣的人可以与模型的创建者 **Kal'tsit** 交流。

- **博士级模型 GPT-o1 在初中数学“陷阱”题上失败，准确率仅为 24.3%** ([Score: 270, Comments: 78](https://reddit.com//r/LocalLLaMA/comments/1fipkus/phdlevel_model_gpto1_fails_on_middle_school_math/))：尽管声称具有博士级智能，**GPT-o1 模型**在 **MathTrap_Public** 数据集上的准确率仅为 **24.3%**，该数据集包含带有“陷阱”的初中数学题。研究人员通过修改 **GSM8K 和 MATH 数据集**中的问题创建了 **MathTrap 数据集**，引入了矛盾或不可解的元素，这需要同时理解原始问题和陷阱才能识别。**开源模型**在 **MathTrap_Private** 上的表现更差，**Reflection-70B** 的准确率为 **16.0%**，**Llama-3.1-8B** 为 **13.5%**，**Llama-3.1-70B** 为 **19.4%**。
  - **博士级数学家**和其他用户指出，他们也会犯和 AI 同样的错误，其中一人表示这个问题“**从根本上来说毫无趣味**”。许多人认为 **x=0 处的间断点**并非本质问题，极限方法是有效的。
  - 用户质疑了研究方法，有人指出该**预印本最后修订于 7 月 11 日**，并未提及 **o1**。他们测试了陷阱题，发现 **o1 在第一次尝试时就正确识别了所有陷阱**，暗示可能存在误导信息。
  - 几位评论者批评了 **Prompt 设计**，认为更合理的提问方式会产生更准确的结果。有人建议这样问：*"该函数是周期性的吗？如果是，计算周期；否则，证明不存在周期。请证明你的论点。"*

## 其他 AI Subreddit 摘要

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型进展与基准测试**

- **OpenAI 的新 GPT-4o1 模型在 IQ 测试中获得了 120 分**，超过了 90% 的人类。然而，在它从未见过的新问题上，其得分接近人类平均水平 100 分。[这仍然代表了 AI 推理能力的重大进步](https://www.reddit.com/r/OpenAI/comments/1fipe3b/openais_new_gpt_model_reaches_iq_120_beating_90/)。

- OpenAI 将 o1-mini 模型的速率限制提高了 7 倍，从[每周 50 条消息增加到每天 50 条消息](https://www.reddit.com/r/singularity/comments/1fim3hh/openai_weve_increased_rate_limits_for_o1mini_by_7x/)。o1-preview 模型也从每周 30 条增加到 50 条。

- o1 模型在编程基准测试中表现出比 o1-preview 的重大改进，[正确率从 62% 跃升至 89%](https://www.reddit.com/r/singularity/comments/1fi2hmb/are_big_jumps_in_reasoning_for_models_of_the_same/)。这代表了复杂代码生成的可靠性提升了 3.5 倍。

- 一些用户报告称 o1-mini 已经取代了 GPT-4 执行编程任务，因为它提供[完整的、无上限的响应，无需点击“继续”](https://www.reddit.com/r/singularity/comments/1fi2hmb/are_big_jumps_in_reasoning_for_models_of_the_same/)。

**AI 伦理与社会影响**

- 亿万富翁 Larry Ellison 建议 [AI 驱动的监控系统可以确保“公民表现出最佳行为”](https://www.reddit.com/r/singularity/comments/1fi0iuq/billionaire_larry_ellison_says_a_vast_aifueled/)，引发了关于隐私担忧和 AI 技术潜在滥用的辩论。

- 关于是该庆祝还是担忧 AI 的飞速进步，目前仍有持续讨论。一些人将其视为令人兴奋的技术进步，而另一些人则对失业和社会影响表示担忧。

**AI 开发与研究**

- o1 模型似乎使用了涉及[带有内置 Chain of Thought 过程的强化训练](https://www.reddit.com/r/singularity/comments/1fi2hmb/are_big_jumps_in_reasoning_for_models_of_the_same/)的突破，这可能允许能力的大规模扩展。

- 一些研究人员建议 o1 可以被视为一种“原型 AGI”架构，尽管在短期和长期记忆等领域可能仍需要额外的突破才能实现通用人工智能。

**AI 工具与应用**

- 像 FLUX 这样新的 AI 图像生成工具正在产生令人印象深刻的结果，展示了[受《半条命》(Half-Life) 启发的苏联时代场景](https://www.reddit.com/r/StableDiffusion/comments/1fi1e04/flux_halflife_but_soviet_era/)和[抽象超现实主义景观](https://www.reddit.com/r/StableDiffusion/comments/1fi8rmg/mirrorscapes_flux/)的示例。

- Quest 3 VR 头显结合 AI 视频生成工具正在开启[新型沉浸式内容创作](https://www.reddit.com/r/singularity/comments/1fihviw/quest_3_vr_headset_using_gravitysketch_runawaymls/)。


---

# AI Discord 摘要

> 摘要的摘要的摘要

## O1-mini

**主题 1. AI 模型：新发布与竞争**

- **Claude 3.5 对决 GPT-4o**：社区在 **Claude 3.5** 和 **GPT-4o** 之间摇摆不定，成员们通过测试来确定哪个模型在特定任务中表现更出色。[Claude vs GPT-4o 对决](https://discord.com/channels/1047197230748151888/1047649527299055688/1285314374591844523) 突显了这场持续的竞争。

- **Qwen 2.5 发布更严格的变体**：**Qwen 2.5** 推出了从 **0.5B** 到 **72B** 参数的新模型尺寸，全部具备增强的内容过滤功能。用户对 **knowledge retention**（知识保留）的担忧依然存在。

- **Mistral 的 Pixtral-12B 登场**：[Pixtral-12B](https://mistral.ai/news/pixtral-12b/) 标志着多模态模型的重大飞跃，提供了可与现有巨头媲美的强大视频和图像生成能力。

**主题 2. 创新工具与集成**

- **Superflex 实现 Figma 转代码**：**Superflex** 现在允许开发者直接从 [Figma 设计](https://www.producthunt.com/posts/superflex) 生成前端代码，无缝简化了从设计到开发的流程。

- **OpenRouter 为 Google Sheets 提供 AI 助力**：[GPT Unleashed for Sheets](https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147) 集成了 **OpenRouter** 的“jobs”和“contexts”等功能，实现在电子表格中进行高效的 **prompt engineering**。

- **Aider 联手 Sonnet 助力编程**：**Sonnet 3.5** 与 **O1 Mini** 的集成增强了 **Aider** 处理编程任务的可靠性，用户对其处理快速修复和任务的效率赞誉有加。

**主题 3. 训练、优化与技术挑战**

- **LM Studio 大幅缩短训练时间**：在 **LM Studio** 中调整 token 和 batch size 将模型训练时间从 **5 天** 缩短至仅 **1.3 小时**，展示了显著的 **optimization** 收益。

- **Tinygrad 面临 AMD 兼容性问题**：用户在 AMD 系统上更新 **tinygrad** 时遇到 **AttributeError**，引发了关于潜在 **kernel version** 不匹配和故障排除策略的讨论。

- **CUDA 模式应对存内计算**：SK Hynix 在 [Hot Chips 2024](https://www.servethehome.com/sk-hynix-ai-specific-computing-memory-solution-aimx-xpu-at-hot-chips-2024/) 上介绍了 **AiMX-xPU**，通过直接在内存中进行计算来增强 **LLM inference**，从而提高 **power efficiency**。

**主题 4. AI 安全与伦理担忧**

- **Cohere 推出可定制的安全模式**：[Cohere 的安全模式](https://discord.com/channels/954421988141711382/954421988783444043/1285331547032911883) 在其 Chat API 中允许用户定制模型输出以满足特定的 **safety requirements**，旨在减轻 **liability concerns**。

- **Unsloth AI 的审查引发争议**：**Phi-3.5** 模型因过度审查而面临抵制，用户分享了 [未审查版本](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored) 并讨论了 **safety** 与 **usability** 之间的平衡。

- **Jailbreaking Claude 3.5 打开了潘多拉魔盒**：针对 **Claude 3.5 Sonnet** 的成功 [越狱 (jailbreak)](https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d) 引发了关于 **model security** 以及 **bypassing safeguards** 伦理影响的讨论。

**主题 5. 社区热点与融资动态**

- **YOLO Vision 2024 邀请 AI 工程师**：由 **Ultralytics** 在马德里 Google 创业校园举办的 [YOLO Vision 2024](https://www.ultralytics.com/events/yolovision) 邀请 AI 工程师注册参加，通过为活动音乐投票等活动促进 **community interaction**。

- **11x AI 获得 2400 万美元 A 轮融资**：**11x AI** 从 **Benchmark** 筹集了 **2400 万美元 A 轮** 融资，使其年度经常性收入增长了 15 倍，并将客户群扩大到 **250 多家客户**。

- **Mistral 的战略举措引发讨论**：对 [微软战略](https://mistral.ai/news/september-24-release/)（将 AI 技术与 Mistral 的产品集成）的分析促使社区反思该公司的 **竞争方向** 以及与其 **历史目标** 的一致性。

---

## O1-preview

**主题 1. 新 AI 模型与发布点燃技术社区**

- [**Qwen 2.5 发布，提供新尺寸和更严格的过滤器**](https://huggingface.co/Qwen?sort_models=modified#models)：**Qwen 2.5** 推出了参数量从 **0.5B 到 72B** 不等的模型，与前代相比引入了更严格的内容过滤。初步测试显示其在主题知识方面存在局限性，引发了对 **knowledge retention**（知识保留）影响的担忧。
- [**Mistral-Small-Instruct-2409 隆重登场**](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409)：**Mistral-Small-Instruct-2409** 模型拥有 **22B parameters**，支持 function calls 以及高达 **128k tokens** 的序列长度。尽管潜力巨大，但它带有非商业使用限制，且建议搭配 [vLLM](https://github.com/vllm-project/vllm) 以获得最佳性能。
- [**LlamaCloud 展现多模态 RAG 魔法**](https://t.co/43eL8zvm7H)：**LlamaCloud** 推出了 **multimodal capabilities**（多模态能力），能够跨非结构化数据类型快速创建端到端的 **multimodal RAG pipelines**。这一飞跃增强了 **marketing decks**（营销演示文稿）、**legal contracts**（法律合同）和 **finance reports**（财务报告）的工作流。

**主题 2. AI 工具获得超能力：丰富的集成**

- [**Google Sheets 通过 OpenRouter 集成获得提升**](https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147)：**OpenRouter** 与 [GPT Unleashed for Sheets](https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147) 插件联手，提供对 **100+ 模型** 的免费访问。用户可以为 prompt 分配短代码，极大增强了电子表格内的 AI 输出管理。
- [**Aider 联手 Sonnet 打造代码魔法**](https://aider.chat/docs/)：开发者们欢呼雀跃，因为 **Aider** 集成了 **Sonnet 3.5** 与 **O1 mini**，通过可靠的编辑和修复增强了编码任务。用户称赞 Aider 在处理快速代码微调和任务分配方面的高效性。
- [**Superflex 将 Figma 设计转化为实时代码**](https://www.producthunt.com/posts/superflex)：**Superflex** 直接将 [Figma designs](https://www.producthunt.com/posts/superflex) 转换为前端代码，并无缝集成到现有项目中。该工具加速了开发进程，让设计师的梦想变为现实。

**主题 3. 技术故障与解决方案：克服 AI 障碍**

- **LM Studio 用户苦恼于 GPU 识别失效**：尽管设置正确，**LM Studio** 仍固执地忽略 GPU，转而让 CPU 和 RAM 过载。与抗锯齿设置相关的模糊屏幕促使用户调整配置以获得更平滑的体验。
- **Unsloth 微调热潮导致幻觉**：微调 'unsloth/llama-3-8b-bnb-4bit' 会导致模型产生幻觉，暗示在保存过程中可能存在数据损坏。社区正在讨论使用 `save_method = 'merged_4bit_forced'` 的影响。
- **BitNet 的三进制技巧引发讨论**：将 **5 个三进制值打包进 8-bit 空间** 的做法被证明是聪明但复杂的。围绕使用 **Lookup Tables**（查找表）来增强该方法的讨论不断升温，挑战着神经网络效率的极限。

**主题 4. AI 安全与研究成为焦点**

- **AI 安全奖学金助力新研究项目**：一位社区成员在获得 **Open Philanthropy fellowship** 后投身于 **AI safety**，热衷于解决 **interpretability**（可解释性）和对齐研究。他们正在寻找未来 **六个月** 的合作机会。
- [**傅里叶变换揭示隐藏状态的秘密**](https://sander.ai/2024/09/02/spectral-autoregression.html)：深入研究隐藏状态的 [Fourier transforms](https://sander.ai/2024/09/02/spectral-autoregression.html) 发现，随着层数加深，状态从均匀分布转向 **power law**（幂律）。人们对 attention 机制在这一频谱现象中的作用感到愈发好奇。
- [**LlamaIndex 通过多模态 RAG 处理视觉数据**](https://t.co/GOedcAdLqF)：由于其视觉特性，**产品手册**一直是一项挑战。**LlamaIndex** 引入了一个复杂的 [indexing pipeline](https://t.co/GOedcAdLqF)，帮助 LLM 有效地导航和理解包含大量图像的文档。

**主题 5. AI 进军商业与创意领域**

- [**Ultralytics 在 YOLO Vision 2024 举办活动**](https://www.ultralytics.com/events/yolovision)：**Ultralytics** 邀请 AI 爱好者参加 **10 月 28 日** 在马德里举行的 [YOLO Vision 2024](https://www.ultralytics.com/events/yolovision)。与会者可以在讨论小组期间为自己喜欢的曲目投票，将技术与乐趣融合。
- [**AdaletGPT 推出用于法律援助的 RAG 聊天机器人**](https://adaletgpt.com)：**AdaletGPT** 推出了基于 **OpenAI** 和 **LangChain** 构建的 **RAG 聊天机器人**，在 [adaletgpt.com](https://adaletgpt.com) 提供 AI 驱动的法律支持。用户可以通过友好的界面获取先进的援助。
- **Open Interpreter 的智能令用户惊叹**：**Open Interpreter** 因其聪明才智和强大能力而广受赞誉。随着用户探索其潜力，兴奋之情溢于言表，Beta 测试名额需求量极大。

# 第 1 部分：Discord 高层摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **O1 Mini 每日限制使用 10 次**：用户对 Perplexity 上 O1 Mini 最近限制为**每天 10 次**表示沮丧，认为与竞争对手相比限制了访问。
   - 有**推测**认为此限制旨在管理服务器成本和营销策略，引发了关于用户体验的质疑。
- **Claude 3.5 与 GPT-4o 的对决**：随着社区成员权衡在 **Claude 3.5 和 GPT-4o** 之间选择的优缺点，紧张局势升级，认为测试对于辨别差异至关重要。
   - 参与者指出 **GPT-4o** 可能在特定任务中表现出色，暗示其增强的能力。
- **Perplexity AI 的推理功能引发热议**：Perplexity 中推出的 **Reasoning focus** 功能引发了讨论，用户正在 **Pro Search** 环境中尝试增强的功能。
   - 反馈强调了输出质量和推理步骤的改进，展示了显著的升级。
- **Minecraft 封禁管理问题解析**：在专门页面上发起了一场由社区主导的 **Minecraft 封禁管理讨论**，征求用户对现有政策的意见。
   - 邀请成员分享想法，建议共同努力解决潜在的管理缺陷。
- **微软的策略引发辩论**：一篇对**微软策略**提出质疑的分析文章引起了关注，促使用户审视该公司的竞争方向。
   - 讨论鼓励反思微软最近的行动是否与其历史目标一致。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5 推出新模型变体**：Qwen 2.5 推出了新的模型尺寸，如 **0.5B, 1.5B, 3B, 7B, 14B, 32B 和 72B**，与前代相比，所有尺寸都有更严格的内容过滤。
   - 据报道，这些模型变体限制了某些主题的知识，引发了对**知识保留**潜在影响的担忧。
- **Mistral-Small-Instruct-2409 发布**：**Mistral-Small-Instruct-2409** 模型拥有 **22B 参数**，支持 function calls 和高达 **128k tokens** 的序列，但有非商业使用限制。
   - 建议与 [vLLM](https://github.com/vllm-project/vllm) 配合使用，以获得最佳的推理流水线性能。
- **微调模型中的幻觉问题**：在对模型 'unsloth/llama-3-8b-bnb-4bit' 进行微调后，用户报告从 Hugging Face 下载的版本存在幻觉，引发了对潜在数据损坏的担忧。
   - 这引发了围绕 `save_method = 'merged_4bit_forced'` 的使用及其对模型性能影响的讨论。
- **优先考虑应用知识而非死记硬背**：强调在 LeetCode 等平台中，**应用知识**胜过单纯对问题的死记硬背，这对于现实场景中的有效编码至关重要。
   - 扎实掌握算法和数据结构（如 **linked lists** 和 **hashmaps**）对于实际应用至关重要。
- **KTO 在 RLHF 圈子中占据主导地位**：在强化学习中，由于 **KTO** 作为一个“点赞、点踩”数据集的简单性，人们更倾向于选择它而非 **ORPO**。
   - 虽然认识到 **RLHF** 方法可以简化模型，但强调了*测试所有可用选项*的必要性。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O1 模型在实际应用中表现滞后**：用户对 **O1 模型**的表现表示失望，指出它们在 playground 场景中表现出色，但在 **Aider** 等实际应用中由于 system prompts 的限制而表现不佳。
   - 虽然 O1 模型展现了潜力，但其有效部署仍是一个问题，促使开发者寻找替代方案。
- **Sonnet 与 Aider 协作**：社区讨论显示，用户主张将 **Sonnet 3.5** 与 **O1 mini** 集成以增强编码任务，理由是其在编辑和修复方面的可靠性更高。
   - 许多人称赞 Aider 能高效处理快速的代码修复，展示了结合这些工具的优势。
- **关于编程 RAG 的辩论**：讨论强调了在编程中 **RAG** 方法相对于在特定代码库上进行 **fine-tuning** 的有效性，许多人主张采用定制化方法以获得更好的结果。
   - 人们对大型代码库中检索机制失效表示担忧，强调了改进策略的必要性。
- **使用 Aider 设置 Azure API Key**：一位用户详细介绍了将 **Aider** 与 **Azure OpenAI** 集成所需的配置步骤，强调了结构化 JSON 请求对功能实现的重要性。
   - 推荐了 LiteLLM 文档等额外资源，以有效处理 Azure API keys。
- **Superflex 将 Figma 转换为代码**：**Superflex** 的发布改变了游戏规则，允许开发者直接从 [Figma 设计](https://www.producthunt.com/posts/superflex)生成前端代码，简化了工作流程。
   - 该工具能将设计平滑地集成到现有项目中，使其成为现代 Web 开发中极具吸引力的选择。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPU 性能故障排除**：尽管在 **Settings -> System Resources** 中进行了正确设置，用户仍对 LM Studio 未利用其 GPU 表示沮丧。导致屏幕模糊的问题与 anti-aliasing 设置有关，从而引发了对配置调整的建议。
   - 活跃的对话强调了常见的故障排除步骤，这些步骤可以增强 GPU 利用率并减少用户界面中的模糊视觉效果。
- **训练时间大幅缩短**：一位用户训练了一个 100k 参数的模型，通过调整 tokens 和 batch size，时间从 **5 天** 缩短到了 **1.3 小时**。社区成员讨论了 data loader 中的瓶颈，强调了高效配置对训练效率的重要性。
   - 对话阐明了通过参数调整优化模型训练时长的实际解决方案。
- **LM Studio 的新功能引发热议**：LM Studio 最近新增的文档集成功能引发了积极反馈，证明了社区对该功能的长期需求。用户渴望测试更新版本并利用改进的功能。
   - 这一功能强调了设计上的简洁性如何吸引缺乏深厚 IT 背景的用户，使高级功能变得更加易于使用。
- **关于双 GPU 配置的讨论**：用户探索了双 **4060 Ti** 配置的优势，旨在不消耗过多功率的情况下最大化 VRAM。这种实用的配置引发了关于使用相同 GPU 以简化设置和管理能效优势的辩论。
   - 讨论表明，在 GPU 配置中优化成本效益和性能的趋势日益增长。
- **VRAM 对 LLM 性能至关重要**：针对处理强大的 LLM 时对 VRAM 的关键需求，人们提出了担忧，并深入探讨了各种 GPU 在 token 生成速率方面的能力。成员们分享的个人经验表明，许多强大的模型超出了当前可用显卡的 VRAM 限制。
   - 对 VRAM 的强调引发了关于 GPU 进步如何更好地支持 LLM 训练和推理需求的深入对话。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **API 文档得到提升**：[Hugging Face Inference API docs](https://huggingface.co/docs/api-inference) 进行了关键更新，现在包含更清晰的速率限制（rate limits）、增强的代码示例以及专门的 PRO 专区。
   - 此次改版旨在随着数据集供应的持续增加来优化用户体验，使部署更加直观。
- **100 万个模型倒计时**：社区推测很快将达到 **100 万个模型**，统计数据显示每周有 **40K 个新模型**上线。
   - 随着参与者对比不同模型仓库的增长率，预测即将迎来这一里程碑，兴奋之情溢于言表。
- **数据集创建新工具**：[DataCraft](https://x.com/dvilasuero/status/1835711765570630017) 作为一种使用自然语言生成合成数据集（synthetic datasets）的无代码工具被推出，旨在简化数据创建的挑战。
   - 该工具结合了最佳实践，为希望构建高效 AI 数据集的用户增强了易用性。
- **参与 Gradio Office Hours**：成员受邀参加正在进行的 [Gradio office hours](https://t.co/Dxeb0jaQ6e)，这是一个讨论功能、增强功能和社区反馈的开放论坛。
   - 该环节为直接与专家分享见解和解决 Gradio 相关问题提供了沃土。
- **LLaMA3 设置挑战**：一位用户在下载 **LLaMA3** 模型时寻求帮助，表达了他们在当前 **PyTorch** 设置中遇到的困难并请求指导。
   - 关于实现选择的困惑随之而来，揭示了大家对模型操作中异构工具有效性的共同需求。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o 在 GeoGuessr 中表现惊人**：成员们对 **GPT-4o** 在 **GeoGuessr** 中的表现感到惊讶，尽管它仍落后于专家级玩家。值得注意的是，它与 **o1-mini** 模型的预期速度有所偏差。
   - 这种表现引发了人们对其在游戏之外的潜在改进和应用的关注。
- **微调任务触及硬限制**：一位用户对他们的微调（fine-tuning）任务超过硬限制（hard limit）表示沮丧，在剩余 **$19.91** 配额的情况下产生了 **$24.31** 的费用。有人猜测这可能与折扣有关。
   - 讨论集中在微调操作中的成本管理策略。
- **高级语音模式（Advanced Voice Mode）可用性待定**：多位成员反映虽然使用了 Plus，但仍无法访问 **Advanced Voice Mode**，预计将在 **秋末** 开放。这引发了关于推送时间的疑问。
   - 这种期待反映了用户对语音功能进步的浓厚兴趣。
- **探索 Ideogram/Midjourney 的自动提示词**：一位成员分享了 **Ideogram/Midjourney 的自动提示词（auto prompt）**，鼓励大家反馈并评价其可用性，并强调这是免费分享的。
   - 这种资源交换的启动展示了社区的协作精神。
- **关于官方库的讨论**：提到 **官方库（official libraries）** 引起了兴趣，但随后没有进行深入交流。这为未来讨论潜在资源留下了空间。
   - 这种模糊性为寻求更多细节的用户留下了澄清的空间。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 与 Google Sheets 集成**：OpenRouter 已被整合进 [GPT Unleashed for Sheets](https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147) 插件中，根据用户需求，该功能现已免费开放。
   - *我个人也非常喜欢使用 OR*，并期待随着更多用户采用这一集成功能，能获得有益的反馈。
- **令人兴奋的功能提升 Google Sheets 性能**：Google Sheets 插件中新增的 'jobs'（任务）、'contexts'（上下文）和 'model presets'（模型预设）等功能简化了 Prompt Engineering。
   - 这些增强功能允许用户为提示词分配短代码，从而优化 AI 输出管理。
- **OpenRouter 遭遇 API 停机**：多名用户报告了访问 OpenRouter 时出现的间歇性问题，特别是 `o1` 模型，导致了对 Rate Limits（速率限制）的困惑。
   - 一位用户注意到瑞士地区出现了临时停机，但确认功能在不久后已恢复。
- **Gemini 在图像生成一致性方面表现不佳**：关于 Gemini 的图像生成能力存在争议，其官方网站与 OpenRouter 上的表现存在差异。
   - 据澄清，Gemini 的聊天机器人使用 Imagen 模型进行图像生成，而 OpenRouter 使用的是 Google Vertex AI。
- **Mistral API 价格大幅下调**：最新公告显示 Mistral API 大幅降价，Large 2 模型降至 **$2**，使其成为一个极具竞争力的选择。
   - 这一转变正在影响用户对于 API 调用所选模型的决策。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **探索 Metal Puzzles 与协作**：[Metal Puzzles GitHub 仓库](https://github.com/abeleinin/Metal-Puzzles)通过协作解谜促进 Metal 编程学习，鼓励社区参与。
   - 提议进行一场直播解谜活动，成员们的热情预示着新手对此的兴趣日益浓厚。
- **Triton LayerNorm 遭遇一致性瓶颈**：一名成员报告称，在 Triton LayerNorm 中使用 **Tensor Parallelism > 1** 会导致**非确定性梯度累积**，从而影响其 MoE 训练。
   - 他们正在联系 Liger 团队，以寻求潜在的见解和替代实现方案。
- **FP8 实现端到端功能修复**：最近的实现更新已成功恢复了前向和反向传播的 **FP8** 端到端能力，推进了 AI 工作流的功能性。
   - 未来的任务将包括多 GPU 支持和性能测试，以确保与现有技术的收敛。
- **SK Hynix 推动内存计算创新**：在 **Hot Chips 2024** 上，SK Hynix 展示了其专为高效 LLM 推理量身定制的内存计算（In-memory Computing）技术 **AiMX-xPU** 和 **LPDDR-AiM**。
   - 该方法通过直接在内存中进行计算，显著降低了功耗和延迟。
- **BitNet 的三元打包特性**：讨论揭示了将 **5 个三元值（ternary values）打包进 8 位空间**优于传统方法，尽管实现复杂，但提升了效率。
   - 成员们考虑将 Lookup Tables（查找表）作为打包方法的可能增强手段，推动进一步探索。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousCon 细节确认**：**NousCon** 的地点详情已确认于当晚发出，引发了关于未来活动举办地（包括 **NYC**）的讨论。
   - 一位用户询问了未来活动对社区参与的更广泛影响。
- **对 Hermes 3 Unleashed 的兴趣**：一位新成员表示希望将 **AI 模型 Hermes 3** 用于业务咨询，并寻求联系方式。
   - 另一位用户建议联系特定成员以获取建议。
- **InstantDrag 成为关注焦点**：**InstantDrag** 被强调为一种现代的基于拖拽的图像编辑解决方案，因其在无需掩码或文本提示的情况下提高速度而受到关注。
   - 开发者将其与 **DragGAN** 进行了对比，展示了更快工作流的潜力。
- **LLM 推理性能极限探索**：Denny Zhou 的一条推文指出，如果给予足够的中间推理 Token（Intermediate Reasoning Tokens），Transformer 理论上可以解决任何问题。
   - 这与一篇被 **ICLR 2024** 接收的论文相关联，强调了 **Constant Depth**（固定深度）在 Transformer 能力中的重要性。
- **Claude 3.5 越狱方法揭晓**：一名成员成功创建了针对 **Claude 3.5 Sonnet** 的 Jailbreak（越狱），据报道该模型特别难以攻破。
   - 虽然受到了之前作品的启发，但他们强调了自己的独特方法和功能性。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Luma Labs 发布 Dream Machine API**：Luma Labs 宣布发布 [Dream Machine API](https://lumalabs.ai/dream-machine/api)，使开发者能够以极少的工具投入利用领先的视频生成模型。
   - 这一举措旨在让视频创作变得触手可及，允许用户直接投入到创意开发中。
- **11x AI 筹集 2400 万美元 A 轮融资**：11x AI 成功从 **Benchmark** 获得了 **2400 万美元 A 轮**融资，其今年年度经常性收入增长了 **15 倍**，服务客户超过 **250 家**。
   - 该团队计划构建 **LLM-powered systems**，旨在变革数字市场进入（go-to-market）策略。
- **AI 对就业市场的冲击**：一份报告预测，明年美国和墨西哥将有 **6000 万个工作岗位**受到 AI 影响，未来十年的预测可能会增加到美国的 **7000 万个**和墨西哥的 **2600 万个**。
   - 虽然某些工作转型可能不会导致失业，但仍有大量职位面临相当大的风险，这凸显了劳动力适应的必要性。
- **Claude 3.5 系统提示词流传**：**Claude 3.5 Projects + Artifacts 系统提示词**通过一个 [gist](https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d) 被分享，在有兴趣探索 AI 应用的用户中获得了关注。
   - 该提示词的相关性因其在多个平台上的讨论而受到关注，表明了它在当前 AI 评估中的重要性。
- **Yann LeCun 展示基于 ZIG 的推理栈**：Yann LeCun 介绍了一个新的**基于 ZIG 的推理栈**，旨在优化高性能 AI 推理，能够在各种硬件上高效运行深度学习系统。
   - 这个开源项目标志着它脱离了隐身模式，展示了在 AI 性能方面的显著进步。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **基础模型在生物技术领域砥砺前行**：一位成员展示了他们在 **biotech** 领域 **foundation models** 的工作，重点关注序列和表格数据的 **large scale representation learning**，强调了 AI 与生物技术应用日益增长的交集。
   - 这突显了利用 **AI technologies** 彻底改变传统生物技术流程的兴趣日益浓厚。
- **AI Safety 奖学金引发关注**：一位成员分享了在获得 **Open Philanthropy 职业转型奖学金**后转向 **AI safety** 的经历，表达了参与 **interpretability** 和 alignment 研究的热情。
   - 他们邀请其他人分享研究项目，以便在接下来的**六个月**内进行潜在的合作。
- **解决 TensorRT-LLM 构建问题**：关于在 **T4** 显卡上构建 **TensorRT-LLM** 的问题浮出水面，特别是引用了与 workspace size 相关的错误，并寻求故障排除建议。
   - 解决该问题的一个建议是使用 `IBuilderConfig::setMemoryPoolLimit()` 来增加 workspace size。
- **通过傅里叶变换解释隐藏状态**：讨论重点关注隐藏状态的 [Fourier transforms](https://sander.ai/2024/09/02/spectral-autoregression.html)，揭示了随着层深度增加，从均匀性到**幂律 (power law)** 的趋势。
   - 有人提出疑问，attention 机制是否在最终隐藏状态的 **power spectrum** 形成中发挥了作用。
- **Pythia Checkpoints 受到关注**：社区成员强调 **Pythia suite** 是探测规模和架构对模型行为影响的强大资源，鼓励更广泛的探索。
   - 成员们表达了通过 [Pythia repository](https://github.com/EleutherAI/pythia) 分析不同架构的兴趣，以确认与模型训练效果相关的观察结果。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SSH 密钥更新后连接失败**：一名成员在更新 SSH 密钥后，其部署的 pods 遇到了 **SSH 连接问题**，询问是否有任何配置调整可以解决此问题。
   - *“我进不去了！”* 引发了关于通过详细配置检查来寻找可能修复方案和替代方案的讨论。
- **Stable Diffusion 模型无法加载**：另一位用户在按照 [安装指南](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides) 操作后仍遇到“模型加载失败”错误，陷入安装困境。
   - 社区建议通过分享具体的错误日志来寻求帮助，以便进行针对性的故障排除。
- **ComfyUI 面临白屏困境**：更新后，一位用户报告 ComfyUI 出现 **白屏** 问题，导致其 GUI 尝试中断。
   - 提出了一种修复方法：完全卸载 ComfyUI 并使用更新脚本重新启动。
- **Control Net 需要强大的数据集**：成员们讨论了训练有效 Control Net 的 **数据集要求**，强调需要高质量数据。
   - 建议包括探索 **新型数据集增强** 方法以提升训练效果。
- **CivitAI 悬赏包寻求建议**：一位成员询问关于发布一个包含 49 个项目、约 4000 张图像的角色包 **CivitAI 悬赏**，寻求合理的 Buzz 报酬建议。
   - *“什么样的报价才合理？”* 引发了关于悬赏定价策略的讨论。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud 发布多模态 RAG 功能**：最近发布的 **LlamaCloud 多模态功能** 使用户能够跨非结构化数据格式快速创建 **端到端多模态 RAG 流水线**，显著增强了工作流（[详情点击此处](https://t.co/43eL8zvm7H)）。
   - 该工具包支持各种应用，包括 **营销幻灯片**、 **法律合同** 和 **财务报告**，从而简化了复杂的数据处理。
- **LlamaIndex 与 Neo4j 无缝集成**：社区成员探索了如何使用 **LlamaIndex** 检索存储在 **Neo4j** 中作为节点属性的 embeddings，建议通过属性图索引进行连接以实现有效查询。
   - 讨论认为，一旦检索到节点，解析其属性以获取 embeddings 应该是一项简单的任务，并链接到了 [Neo4j Graph Store - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/Neo4jKGIndexDemo/)。
- **解决 LlamaIndex 包中的循环依赖问题**：在 `llama-index-agent-openai` 和 `llama-index-llms-openai` 之间检测到循环依赖问题，促使成员们集思广益潜在解决方案，包括创建一个 **openai-utils** 包。
   - 关于这些修复时间表的问题激增，需要社区贡献以迅速解决依赖问题。
- **使用 GPT-4o 导航图像坐标**：一位用户强调了使用 **GPT-4o** 进行 **图像坐标提取** 的挑战，特别是由于其网格叠加方法，在对齐标签和获取准确坐标方面存在困难。
   - 社区鼓励提供反馈，以提高检测实体进行图像裁剪的精度，强调了涉及空间识别的技术难度。
- **多模态 RAG 与产品手册挑战**：**产品手册** 已被证明对 RAG 技术来说非常困难，因为它们主要是视觉化的，需要复杂的 [索引流水线](https://t.co/GOedcAdLqF) 才能让 LLM 有效地导航。
   - 讨论强调需要处理产品手册中典型的分步视觉效果和图表的方法。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Mistral 发布新功能**：Mistral 推出了多项功能，包括在 [La Plateforme 上的免费层级](https://mistral.ai/news/september-24-release/)，旨在供开发者进行 API 实验。
   - 这些更新还包括**降价**以及对 **Mistral Small** 的增强，使其对用户更具吸引力。
- **Transformer 受益于中间生成**：研究表明，在 Transformer 中加入“思维链”（chain of thought）可以显著增强其计算能力。
   - 这种方法有望提高在标准 Transformer 难以应对的推理任务上的性能。
- **揭秘 Gemini 模型**：关于未发布的 **Gemini 模型**（如 **potter-v1** 和 **dumbledore-v1**）的令人兴奋的见解已经出现，暗示了包括 **gemini-test** 和 **qwen2.5-72b-instruct** 在内的强大阵容。
   - 社区对这些新模型议论纷纷，标志着模型开发的一个关键时刻。
- **共同庆祝 Newsletter 读者**：一位成员分享了“伟大的 Newsletter 读者派对”的邀请，通过分享阅读创造了社区参与的机会。
   - 这一举措旨在建立联系，并培养参与者对精选内容的喜爱。
- **对主流媒体依赖的批评**：一场讨论强调了仅依靠主流媒体获取新闻的弊端。
   - 成员们表达了对探索更多样化和替代性来源的愿望。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **探索 LangChain 中的聊天历史管理**：成员们讨论了 **LangChain** 中 **Chat Message History Management** 的复杂性，特别是关于在 **PostgresChatMessageHistory** 中存储 UI 消息的问题。
   - 大家一致认为，UI 特有的消息必须存放在单独的表中，因为现有系统缺乏组合事务支持。
- **设定开源贡献目标**：一位成员表达了为开源项目做出重大贡献的雄心，同时寻求赞助以保持独立。
   - 他们请求社区就实现这些有影响力的贡献的途径提供见解。
- **迁移到现代 LLMChain 实现**：反馈建议从旧版的 **LLMChain** 迁移到更新的模型，以获得更好的参数清晰度和流式传输（streaming）能力。
   - 更新的实现允许更轻松地访问原始消息输出，强调了保持更新的重要性。
- **AdaletGPT 推出 RAG 聊天机器人**：**adaletgpt.com** 的一名后端开发人员推出了一个利用 **OpenAI** 和 **LangChain** 的 **RAG 聊天机器人**，邀请用户在 [adaletgpt.com](https://adaletgpt.com) 进行体验。
   - 他们鼓励社区咨询，并承诺会以“我将竭尽全力为您服务”的态度提供支持。
- **针对本地业务集成的 AI 解决方案**：一位成员表示准备向本地企业推广 AI 解决方案，并询问有效的实施策略。
   - 他们专门寻求了关于如何吸引可能不熟悉 AI 的企业主的建议。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 遇到 AMD 相关问题**：用户在尝试将 **tinygrad** 从 **0.9.0** 升级到 **0.9.2** 时在 AMD 平台上遇到了 **AttributeError**，这表明 **struct_kfd_ioctl_criu_args** 可能存在内核版本问题。
   - 调查参考了 **tinygrad/extra/hip_gpu_driver/test_kfd_2.py** 文件以及解决该问题的 **[pull request #5917](https://github.com/tinygrad/tinygrad/pull/5917)**。
- **监控 VRAM 分配峰值**：用户寻求关于识别 **VRAM 分配峰值** 原因的建议，引发了关于有效内存使用监控工具的讨论。
   - 社区成员强调了理解这些峰值对于优化 Tinygrad 性能的重要性。
- **调查 Tinygrad Tensor 错误**：另一位成员报告了在 **Tinygrad** 中进行 Tensor 操作时遇到的错误，并链接到了一个 **[公开 issue](https://github.com/tinygrad/tinygrad/issues/6352)** 以获取更多细节。
   - 这突显了调试 Tinygrad 过程中持续存在的挑战以及社区协作的必要性。
- **Diffusers 分支集成 Tinygrad**：讨论围绕一个利用 Tinygrad 的 **Diffusers fork** 展开，该分支避开了 Torch，旨在采用一种不直接复制的新方法。
   - 社区成员对这一举措表示热烈欢迎，认为这是对 Tinygrad 生态系统的潜在增强。
- **NotebookLM 制作引人入胜的 Tinygrad 播客**：**NotebookLM** 团队发布了一个 **8 分钟的播客**，通过生动的比喻来阐明 Tinygrad 的概念，并有效地推介了 **tinybox**。
   - 这种方法展示了教育他人了解 Tinygrad 原理和应用的创新方式。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 推出测试版安全模式 (Safety Modes)**：Cohere 宣布在其 Chat API 中启动 **Safety Modes** 测试版，允许用户根据安全需求自定义模型输出。
   - *这可能允许用户实施安全检查并减轻责任顾虑。*
- **Cohere 优化市场策略**：**Cohere** 战略性地专注于特定的用例，以在拥挤的 LLM 市场中航行，避免过度饱和。
   - 成员们讨论了**务实的商业选择**的价值，这些选择强调了模型应用中的清晰度和实用性。
- **关于微调模型的咨询**：一位用户询问在微调期间是否可以跳过最后的 `<|END_OF_TURN_TOKEN|>`，以便更顺畅地继续推理。
   - 他们提出了一个训练数据的 POC 示例，强调了微调聊天模型的潜在好处。
- **Sagemaker Client 问题反馈**：一位用户报告在访问端点时，从 Sagemaker 客户端收到了 `input_tokens=-1.0` 和 `output_tokens=-1.0`。
   - 这引发了对端点设置过程中可能存在配置错误的担忧。
- **Sagemaker 查询的支持渠道**：有人建议原帖作者联系 [support@cohere.com](mailto:support@cohere.com) 以寻求有关 Sagemaker 计费问题的帮助。
   - 该用户表示他们将通过检查用户账户来进一步调查此事。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **GitHub 交流引发期待**：一位成员在 **GitHub** 上回复了 Prashant 关于正在进行的讨论，并可以关注后续的潜在反应。
   - *请继续关注这次互动可能产生的任何后续反应。*
- **展示 CodeBlueprint 与 Aider**：一位成员分享了一个链接，展示了他们新的编码模式 **CodeBlueprint with Aider**，展示了其集成潜力。
   - 这一展示可能为在编码实践中采用新工具提供见解。
- **Ruff 检查遇到错误**：Prashant 报告在执行 `ruff check . --fix-only` 时遇到 **TOML 解析错误**，提示未知字段 `indent-width`。
   - 此错误突显了需要解决的潜在配置不匹配问题。
- **引入 GPT-4 Vision API 封装器**：一个新的 [Pull Request](https://github.com/stanfordnlp/dspy/pull/682) 添加了 **GPT-4 Vision API wrapper**，简化了 DSPy 仓库中的图像分析请求。
   - 在 `visionopenai.py` 中引入 **GPT4Vision** 类应该会简化开发者的 API 交互。
- **社区渴望贡献和悬赏**：成员们表达了贡献的热情，其中一人询问是否有可参与的悬赏 (bounties)。
   - 尽管承认需要进行更改，但讨论期间未透露有关悬赏的具体细节。

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **图像合成技术表现出色**：成员们讨论了基础的 **compositing**（合成）技术是图像生成的有效选择，建议使用 **Pillow** 等库来增强效果。
   - *不建议使用带有集成文本的图像进行训练*，以实现海报级的视觉效果。
- **后期处理提升质量**：涉及 **GIMP** 等工具的有效工作流可以通过后期处理技术显著提高图像的准确性和效果。
   - *在后期处理中完成* 相比仅依赖初始方法能产生最佳效果。
- **Nouswise 增强创意流程**：**Nouswise** 被强调为一个个人搜索引擎，在从 **reading**（阅读）到 **curation**（策展）的各个创意阶段提供可信的答案。
   - 它的功能简化了 **searching**（搜索）和 **writing**（写作）的方法，提升了整体生产力。
- **寻求 Whisper speech 见解**：一位成员询问了关于 **Whisper speech** 技术的经验，引发了查看特定频道以获取进一步指导的建议。
   - 社区讨论允许分享见解和*集体知识*，并提供相关资源链接。
- **StyleTTS-ZS 项目资源征集**：一位成员为 **StyleTTS-ZS** 项目请求计算资源支持，该项目旨在实现高效的高质量 zero-shot 文本转语音合成。
   - 该项目的详细信息已发布在 [GitHub](https://github.com/yl4579/StyleTTS-ZS) 上，鼓励社区协作开发。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 给用户留下深刻印象**：**Open Interpreter** 因其巧妙的设计赢得了赞誉，增强了社区对其功能的兴奋感。
   - 成员们表达了探索其潜力的渴望，并围绕其功能展开了持续讨论。
- **Beta 测试兴趣高涨**：成员们询问了 **Open Interpreter** 的 **beta testers** 名额，表明了对贡献开发的持续热情。
   - 此类询问反映了对协助工具进步和改善用户体验的浓厚兴趣。
- **本周五 Human Device Discord 活动**：**Human Device** 即将举行的活动定于本周五，鼓励参与者通过 [Discord 链接](https://discord.gg/UmXdvf3v?event=1285618083448225813) 加入。
   - 该活动旨在让用户参与有关创新技术和产品的讨论。
- **Tool Use 播客聚焦语音智能**：[Tool Use](https://youtu.be/La9BfaFTsFU) 的最新一期节目展示了 **Killian Lucas** 讨论语音智能的进展以及 **01 Voices** 脚本的能力。
   - 听众可以深入了解语音 Agent 如何在群组对话中无缝交互。
- **Deepgram 走向开源**：一位成员宣布创建了 **Deepgram** 的开源和本地版本，激发了社区对更易用工具的热情。
   - 这一举措强调了社区在开发有效的语音智能解决方案方面的参与。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Eleuther Eval Recipe 的使用限制**：关于 **Eleuther eval recipe** 及其在 **generation**（生成）和 **multiple choice (mc)** 任务中的表现出现了担忧，特别是关于生成任务的 **cache**（缓存）对后续任务执行的影响。
   - 其他用户确认该 recipe 运行异常，暗示可能存在与 **cache management** 相关的潜在问题。
- **缓存重置的必要性**：用户讨论了缺乏适当的缓存重置可能是问题的根源，特别是在 **model generation** 之后切换任务时。
   - 一位成员指出他们在生成后重置缓存的习惯，但强调这仅是为新一轮生成做准备，并未实现完全重置。
- **MM 评估期间 Batch Size 不一致**：讨论指出在模型评估期间（特别是使用缓存时）存在未达到预期 Batch Size 的问题。
   - 预计当另一位用户尝试未来的多模型评估时，这一挑战将再次出现。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **社区对 RISC-V 支持的好奇**：成员们正在询问支持 **RISC-V** 的计划，但目前该架构**尚无计划**。
   - 这种兴趣可能会引发未来关于替代架构兼容性的讨论。
- **零拷贝互操作性缺乏 Mojo-Python 集成**：由于目前无法从 Python 导入或调用 Mojo 模块，实现**零拷贝数据互操作性**面临挑战。
   - 讨论中提到了 **Mandelbrot 示例** 如何通过 `numpy_array.itemset()` 低效地利用内存。
- **Mandelbrot 示例突显了 Mojo 的潜力**：关于 **Mandelbrot 集** 的教程展示了 Mojo 在集成 Python 可视化工具的同时，能够执行高性能代码。
   - 该教程说明了 Mojo 适合利用 Python 库为不规则应用构建快速解决方案。
- **LLVM Intrinsics 现在支持在 Comptime 使用**：Mojo 扩展了对 **comptime LLVM intrinsics** 的支持，重点针对整数的 `ctlz` 和 `popcount` 等函数。
   - 未来的发展取决于 LLVM 对这些 intrinsics 进行常量折叠（constant fold）的能力，从而为更广泛的类型支持开辟道路。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Shampoo 在 Transformers 中未受重视**：一位成员指出 **Transformers** 和 **Axolotl** 中都缺少 **Shampoo**，并认为它提供了被忽视的实质性好处。
   - *Shampoo 在大规模、可预测的方式下简直就是免费的午餐*，这表明其潜力可能值得进一步探索。
- **Shampoo 缩放定律 vs Adam**：关于 **语言模型的 Shampoo 缩放定律** 的讨论揭示了与 **Adam** 的对比分析，并引用了 **Kaplan et al** 的图表。
   - 该图表展示了 Shampoo 有效的缩放特性，表明对于大型模型，它是比 **Adam** 更优的选择。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Ultralytics 邀请社区参加 YOLO Vision 2024！**：Ultralytics 将于 <t:1727424000:F> - <t:1727458200:t> 在马德里的 Google Campus for Startups 举办 [YOLO Vision 2024](https://www.ultralytics.com/events/yolovision) 🇪🇸，并邀请 AI 工程师注册参加。
   - 与会者可以通过在讨论环节为音乐投票来参与互动，旨在增强社区交流！
- **为 YOLO Vision 2024 的音乐投票！**：[YOLO Vision 2024](https://www.ultralytics.com/events/yolovision) 的注册参与者可以对讨论期间播放的音乐进行投票，为活动增添独特的互动环节。
   - 这一举措鼓励与会者参与，旨在营造活跃的活动氛围。

---

**Alignment Lab AI Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

# PART 2: 按频道详细摘要及链接


{% if medium == 'web' %}

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1285314374591844523)** (389 条消息🔥🔥): 

> - `O1 Mini 限制`
> - `AI 模型对比`
> - `Perplexity 特性与功能`
> - `与其他服务的集成`
> - `Pro Search 增强功能的承诺` 


- **O1 Mini 的每日使用限制**：用户讨论了 Perplexity 上 O1 Mini 最近每人每天 **10 次使用** 的限制，对与其他平台相比更低的上限表示不满。
   - 有推测认为，限制 O1 Mini 的访问可能是为了防止干扰营销策略并管理服务器成本。
- **AI 模型对比：Claude vs. GPT-4o**：成员们不确定在 **Claude 3.5 和 GPT-4o** 之间该如何选择，强调了测试两者以找到更合适模型的重要性。
   - 讨论指出，GPT-4o 在某些任务上可能更胜一筹，主要是由于其更广泛的能力。
- **Perplexity 的新功能**：Perplexity 中引入的 **Reasoning focus** 引起了兴趣，用户注意到它似乎利用了 O1 Mini，并增强了 Pro Search 环境中的功能。
   - 用户正在尝试并分享新功能的体验，展示了在输出质量和推理步骤方面的进步。
- **将 Perplexity 与其他工具集成**：有关于如何将 Perplexity Pro 与 **VSCode extension** 集成以实现 autocomplete 功能的咨询，表明了对增强工作流集成的需求。
   - 用户指出，目前的功能存在于 Perplexity 平台内部，但与外部应用的集成仍然不够直接。
- **社区协作与资源**：鼓励用户探索 **Complexity extension**，该扩展显著增强了 Perplexity 的用户体验，提供了高级功能和组织特性。
   - 平台内的社区经理强调了用户反馈和协作在改进工具以及体验平台全部潜力方面的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.testingcatalog.com/perplexity-adopts-o1-model-amid-openais-new-message-limits/">Perplexity adopts o1 model amid OpenAI’s new message limits</a>：OpenAI 的 o1 模型现在允许每天 50 条消息，高于之前的每周限制。o1 mini 可能很快会免费。o1 具有先进的推理能力，4o 的知识截止日期现在是 2024 年 9 月。</li><li><a href="https://docs.openinterpreter.com/getting-started/introduction">Introduction - Open Interpreter</a>：未找到描述</li><li><a href="https://tenor.com/view/the-universe-tim-and-eric-mind-blown-mind-blown-meme-mind-explosion-mind-explosion-meme-gif-18002878">The Universe Tim And Eric Mind Blown GIF - The Universe Tim And Eric Mind Blown Mind Blown Meme - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/testingcatalog/status/1835799548284883148?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>：ChatGPT 正在实验 Forced Search 功能的新设计，使其对用户更加突出。“整个网络，现在都在 ChatGPT 中”。也被 @btibor91 发现。</li><li><a href="https://cplx.vercel.app/">Complexity</a>：每个人都想要的 Perplexity.ai 增强版。</li><li><a href="https://chromewebstore.google.com/detail/complexity/ffppmilmeaekegkpckebkeahjgmhggpj">Complexity - Chrome Web Store</a>：⚡ 为你的 Perplexity.ai 注入强劲动力</li><li><a href="https://addons.mozilla.org/en-US/firefox/addon/complexity/">Complexity – Get this Extension for 🦊 Firefox (en-US)</a>：下载 Firefox 版 Complexity。⚡ 为你的 Perplexity.ai 注入强劲动力
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1285339735585456219)** (14 条消息🔥): 

> - `Minecraft 审核封禁问题`
> - `Microsoft 的战略`
> - `研究课题讨论`
> - `全球 AI 中心实力`
> - `Bitcoin 的 66 位谜题` 


- **Minecraft 审核封禁问题讨论**：一个关于 **Minecraft** 审核封禁问题的[详细页面](https://www.perplexity.ai/page/minecraft-moderation-ban-issue-udsocXhbT8uu5egJmMjLFg)已开启供用户讨论。
   - 鼓励社区成员分享他们对当前审核政策的看法。
- **Microsoft 的业务战略受到审查**：一篇[质疑 Microsoft 在科技领域策略](https://www.perplexity.ai/search/why-does-microsoft-not-seem-to-yp8liVn9QP6FoBu15ueWIQ)的帖子引发了对其竞争立场和未来方向的关注。
   - 建议用户分析 Microsoft 的行动是否与其历史方法和目标一致。
- **探索新研究课题**：一位成员表达了讨论新研究课题的意向，并在搜索[帖子](https://www.perplexity.ai/search/i-have-a-research-topic-i-want-9LTYvWD5RNONSiW.mFBsmg)中详细说明了他们的兴趣。
   - 他们正在寻求反馈和资源以进一步发展他们的想法。
- **全球 AI 中心实力辩论**：一场关于[全球 AI 中心实力](https://www.perplexity.ai/page/global-ai-center-strength-trai-Bq16gnJgT8uDpzZPuWWSPg)的对话强调了各个中心在 AI 领域的能力和贡献。
   - 参与者正在评估这些中心在塑造未来 AI 进步方面的潜力。
- **Bitcoin 的 66 位谜题已破解！**：一个详细介绍 **Bitcoin** 中 **66-bit puzzle** 如何被破解的页面引起了爱好者的兴趣，可在[此处](https://www.perplexity.ai/page/bitcoin-s-66-bit-puzzle-solved-1fFxJ9Z8Ti6V83.DnGIU.Q)查看。
   - 讨论围绕该解决方案对加密货币安全性的影响展开。



**提及的链接**: <a href="https://www.youtube.com/embed/Eq4HMjeDj08">YouTube</a>: 未找到描述

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1285337201550299136)** (280 条消息🔥🔥): 

> - `Qwen 2.5 模型发布`
> - `Mistral-Small-Instruct-2409`
> - `Unsloth 安装问题`
> - `微调 LLM`
> - `Backus-Naur Form (BNF)` 


- **Qwen 2.5 推出新模型变体**：Qwen 2.5 具有新的模型尺寸，包括 **0.5B, 1.5B, 3B, 7B, 14B, 32B 和 72B**。根据初步测试，与前代相比，它具有更严格的内容过滤。
   - 据报告，模型变体限制了某些主题的知识，可能会影响从某些来源获取的知识保留。
- **Mistral-Small-Instruct-2409 发布**：**Mistral-Small-Instruct-2409** 模型拥有 **22B 参数**，能够支持函数调用（function calls），并可处理长达 **128k** token 的序列，尽管目前有非商业使用限制。
   - 建议将此模型与 [vLLM](https://github.com/vllm-project/vllm) 配合使用，以实现高效的推理流水线。
- **Unsloth 安装问题**：用户在安装 Unsloth 和管理 **xformers** 等依赖项时遇到困难，部分用户在 Windows 上收到“不支持的平台”错误。
   - 已提供建议使用 **WSL** 或安装特定版本的 CUDA 来解决这些安装问题。
- **针对特定 JSON 语法的微调策略**：为了针对特定 JSON 语法有效地微调模型，所需的训练数据量可能有所不同；根据模型的先验知识，500 到数千个示例可能就足够了。
   - 强调训练质量优于数量，并建议实施 **Backus-Naur Form (BNF)** 以确保输出的结构完整性。
- **关于 Backus-Naur Form (BNF) 的讨论**：使用 **BNF** 可以帮助限制语言模型遵循定义的结构，这可能会增强生成需要特定格式的输出时的性能。
   - 理解 BNF 对于解析输出并确保其保持所需的结构完整性至关重要。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mistral-Small-Instruct-2409">mistralai/Mistral-Small-Instruct-2409 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/SmolLM-1.7B-Instruct">unsloth/SmolLM-1.7B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/zhouwenmeng/status/1834899729165304198">来自 Wenmeng Zhou (@zhouwenmeng) 的推文</a>: Qwen-q1 ? ? 🍓🍓🍓🍓🍓</li><li><a href="https://huggingface.co/unsloth/SmolLM-1.7B-Instruct-bnb-4bit">unsloth/SmolLM-1.7B-Instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/google/datagemma-release-66df7636084d2b150a4e6643">DataGemma Release - google 集合</a>: 未找到描述</li><li><a href="https://download.pytorch.org/whl/cu124">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes/multilingual/README.md">llama-recipes/recipes/multilingual/README.md (位于 0efb8bd31e4359ba9e8f52e8d003d35ff038e081) · meta-llama/llama-recipes</a>: 使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama3 的脚本，涵盖单节点/多节点 GPU。支持用于摘要和问答等应用的默认及自定义数据集...</li><li><a href="https://huggingface.co/Qwen?sort_models=modified#models">Qwen (Qwen)</a>: 未找到描述</li><li><a href="https://github.com/unclemusclez/ollama-toolkit">GitHub - unclemusclez/ollama-toolkit: Ollama Toolkit 是一系列强大的工具集合，旨在增强您对 Ollama 项目的体验。Ollama 是一个用于部署和扩展机器学习模型的开源框架。可以将其视为简化工作流程并释放 Ollama 全部潜力的“一站式商店”！</a>: Ollama Toolkit 是一系列强大的工具集合，旨在增强您对 Ollama 项目的体验。Ollama 是一个用于部署和扩展机器学习模型的开源框架。可以将其视为...</li><li><a href="https://github.com/ACGNnsj/triton-windows-build/releases/">Releases · ACGNnsj/triton-windows-build</a>: Triton 语言和编译器的开发仓库 - ACGNnsj/triton-windows-build</li><li><a href="https://huggingface.co/google/gemma-7b">google/gemma-7b · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/KTVeTXPZD9">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/vllm-project/vllm/issues/3561">如何知道 generate 同时能处理的最大并发请求数/Token 数？· Issue #3561 · vllm-project/vllm</a>: 您当前的环境。我想知道如何了解或配置并发请求的数量（Token 数量）。我可以从日志中看到这些值：INFO 03-18 12:34:52 llm_engine.py:706] Avg p...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1285456514219245632)** (7 条消息): 

> - `求职策略`
> - `应用知识的重要性`
> - `代码风格合规性`
> - `代码中的基础理解`
> - `LeetCode 的局限性` 


- **优先考虑应用知识而非死记硬背**：一位成员强调，虽然涉及大量记忆，但**应用知识**比单纯背诵 LeetCode 题目更重要。
   - 理解算法和数据结构（如**链表**和**哈希表**）对于实际编程至关重要。
- **遵守现有代码风格**：有人指出，修改代码通常需要遵守**现有的代码风格**，以防止随意更改。
   - 这意味着即使有更好的想法，也可能因为**基础编程实践**的差异而不被接受。
- **处理遗留代码库**：在大多数公司中，开发者通常在可能未使用最新技术或方法的**遗留代码库**中工作。
   - 这种协作通常会影响新想法的接受程度，因为同事会与他人编写的代码进行交互。
- **LeetCode 准备工作的边际收益递减**：一位成员指出，通过 LeetCode 进行死记硬背并不一定能转化为实际场景中的**应用能力**。
   - 了解如何以及何时应用概念可能比背诵解决方案更重要。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1285314964973813883)** (15 messages🔥): 

> - `Model fine-tuning`
> - `Mac compatibility issues`
> - `Gratitude in the community` 


- **微调模型中的幻觉 (Hallucinations)**：一位用户报告称，在对 'unsloth/llama-3-8b-bnb-4bit' 模型进行微调后，从 Hugging Face 下载的版本开始出现幻觉，引发了对保存过程中可能出现损坏的担忧。
   - 他们分享了上传命令，其中包括 `save_method = 'merged_4bit_forced'`，引发了关于这是否会影响模型性能的讨论。
- **Mac M3 芯片运行问题**：一位用户在 Mac M3 芯片上运行 Kaggle notebook 时遇到问题，提示 'Torch not compiled with CUDA enabled' 错误，并指出 Mac OS 缺乏 CUDA 支持。
   - 另一位成员指出 'Unsloth 不支持 Mac'，关于潜在解决方案的问题仍悬而未决。
- **社区感谢**：一位成员对社区在训练生成 Python 代码的神经网络方面提供的帮助表示感谢，并以庆祝性的评论结束。
   - 随后出现了鼓励和认可的回应，增强了小组内互助的氛围。



**提到的链接**：<a href="https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-1-8b-conversational-unsloth/notebook"> Kaggle Llama 3.1 8b Conversational Unsloth</a>：使用 Kaggle Notebooks 探索和运行机器学习代码 | 使用来自无附加数据源的数据

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1285583026507616266)** (23 messages🔥): 

> - `KTO vs PPO`
> - `Domain Adaptation of Llama-3.1-8B`
> - `Continued Pre-Training vs Full Fine-Tuning`
> - `GPU Limitations`
> - `Unsloth Support` 


- **KTO 在 RLHF 领域占据主导地位**：一位成员表示在强化学习中，相比 **ORPO** 更倾向于 **KTO**，理由是其作为“点赞、点踩”数据集的简洁性。
   - 虽然承认 **RLHF** 方法可以简化模型，但他们强调需要 *测试所有可用选项*。
- **Llama-3.1-8B 的领域自适应 (Domain Adaptation) 难题**：另一位成员正在寻求 **Llama-3.1-8B** 领域自适应的帮助，目标是在不量化权重的情况下进行全量微调 (Full Fine-Tuning)，但在 **H100 160GB** GPU 上遇到了错误。
   - 他们成功执行了持续预训练 (Continued Pre-Training)，但渴望看到全量微调的表现，并正在研究提高模型精度的方法。
- **持续预训练可能足以替代微调**：讨论强调，**持续预训练**可以达到接近全量微调的效果，尤其是在逐步探索参数调整时。
   - 社区似乎支持尝试量化版本之外的高精度模型，正如韩国 notebook 示例所示。
- **GPU 资源限制影响研究**：对 **GPU 资源有限** 的挫败感显现，这减慢了强化学习中各种方法的测试速度。
   - 成员们讨论了需要谨慎地逐一进行评估，就像为长途旅行节省有限的燃料一样。
- **现有编码脚本中的潜在问题**：一位成员在使用多个 **H100-80GB GPU** 进行 FFT 处理时遇到了代码问题，尽管资源充足但仍产生错误。
   - 这引发了对其现有设置中可能存在的错误的询问，突显了在领域自适应过程中遇到的挑战。


  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1285318099926782023)** (238 messages🔥🔥): 

> - `O1 Models Performance`
> - `Aider and Sonnet Integration`
> - `RAG and Fine-tuning for Codebases`
> - `Use Cases for Different Models`
> - `Feedback on Flux Canvas Art`

- **O1 Models 性能**：讨论集中在 O1 Models 的性能上，一些用户对其因缺乏 system prompts 而在应用中受限表示沮丧。
   - 用户指出，虽然 O1 Models 在 playgrounds 中表现出色，但在 Aider 等其他应用中却表现不佳。
- **Aider 与 Sonnet 集成**：Aider 用户建议将 Sonnet 3.5 与 O1 mini 结合用于编程任务，事实证明 Sonnet 在编辑和编码方面更可靠。
   - 几位用户报告了利用 Aider 的功能成功进行快速修复和任务处理的经验。
- **针对代码库的 RAG 与 Fine-tuning**：关于 RAG 在编程任务中的有效性存在争论，一些用户主张针对特定 codebases 对模型进行 Fine-tuning。
   - 对话强调了在大型 codebases 中使用检索机制所面临的挑战及其涉及的过程。
- **不同模型的用例**：对 O1-mini 和 Claude 等模型进行了比较，重点关注它们独特的优势和应用场景。
   - 一些用户发现 O1 Models 适合从头开始生成代码，但在重构和代码编辑场景中表现不足。
- **对 Flux Canvas Art 的反馈**：一位用户请求对其网站 Flux Canvas Art 提供反馈，寻求社区的见解。
   - 这一反馈请求出现在关于开发中所使用的各种技术和工具的讨论中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/config/aider_conf.html">YAML 配置文件</a>：如何使用 YAML 配置文件配置 aider。</li><li><a href="https://aider.chat/docs/repomap.html">仓库地图 (Repository map)</a>：Aider 使用 Git 仓库地图为 LLM 提供代码上下文。</li><li><a href="https://aider.chat/docs/usage/modes.html">聊天模式</a>：使用 chat、ask 和 help 聊天模式。</li><li><a href="https://trypear.ai/">PearAI - 用于快速开发的开源 AI 代码编辑器</a>：PearAI 是一款开源的 AI 驱动代码编辑器，具有 AI 聊天、内联提示和调试等功能，可加速您的编码过程。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://x.com/openai/status/1835851762609357292?s=46">来自 OpenAI (@OpenAI) 的推文</a>：我们非常感谢您对 OpenAI o1 的热情，我们希望您能更多地使用它。我们将 o1-mini 的速率限制提高了 7 倍，从每周 50 条消息增加到每天 50 条消息。o1-preview i...</li><li><a href="https://fluxcanvas.art/">Flux Canvas Art</a>：未找到描述</li><li><a href="https://github.com/mckaywrigley/o1-ai-playground">GitHub - mckaywrigley/o1-ai-playground：加入互联网上学习 AI 技能的最佳场所。使用代码 "o1launchparty" 可享受额外 20% 的折扣。</a>：加入互联网上学习 AI 技能的最佳场所。使用代码 "o1launchparty" 可享受额外 20% 的折扣。 - mckaywrigley/o1-ai-playground</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/CONTRIBUTING.md#setting-up-a-development-environment">paul-gauthier/aider 项目 main 分支下的 CONTRIBUTING.md</a>：aider 是您终端中的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="http://fluxcanvas.art/">Flux Canvas Art</a>：未找到描述</li><li><a href="https://github.com/mckaywrigley/o1-ai-playground/pull/2">由 fry69 修复的深色系统主题注水 (hydration) 错误 · Pull Request #2 · mckaywrigley/o1-ai-playground</a>：修复当系统默认设置为深色主题时的注水错误。解决方案取自此处 -> facebook/react#17741 (comment)</li><li><a href="https://github.com/DoS007/big-AGI-2">GitHub - DoS007/big-AGI-2：由最先进模型驱动并提供高级 AI/AGI 功能的生成式 AI 套件。它具有 AI 角色、AGI 功能、多模型聊天、文本转图像、语音、响应流、代码高亮和执行、PDF 导入、开发者预设等更多功能。支持本地部署或云端部署。</a>：由最先进模型驱动并提供高级 AI/AGI 功能的生成式 AI 套件。它具有 AI 角色、AGI 功能、多模型聊天、文本转图像、语音、响应流...</li><li><a href="https://github.com/enricoros/big-AGI.git">GitHub - enricoros/big-AGI：由最先进模型驱动并提供高级 AI/AGI 功能的生成式 AI 套件。它具有 AI 角色、AGI 功能、多模型聊天、文本转图像、语音、响应流、代码高亮和执行、PDF 导入、开发者预设等更多功能。支持本地部署或云端部署。</a>：由最先进模型驱动并提供高级 AI/AGI 功能的生成式 AI 套件。它具有 AI 角色、AGI 功能、多模型聊天、文本转图像、语音、响应流...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1285319914000744470)** (45 messages🔥): 

> - `Aider configuration`
> - `Azure OpenAI integration`
> - `User story implementation`
> - `Streaming output metrics`
> - `OpenRouter outages` 


- **针对 Azure OpenAI 的 Aider 设置**：一位用户分享了配置 Aider 以配合 Azure OpenAI 使用的过程，需要为每个请求以 JSON 格式传递用户密钥。
   - 建议查阅 LiteLLM 文档，了解如何传递 Azure API 密钥以及处理特定的请求格式。
- **利用 user stories 增强应用功能**：一位用户将 Aider 的功能与 marblism.com 进行了对比，强调了两者如何通过任务管理来帮助创建应用功能。
   - 他们表示有兴趣使用 Aider 来实现 user stories，以改进其应用开发流程。
- **流式传输指标的挑战**：一位用户询问了在使用 Aider 的流式传输（streaming）功能时如何获取准确的指标，并提到评估完成状态存在困难。
   - 他们指出禁用流式传输会显著降低体验，强调了寻找平衡点的必要性。
- **报告 OpenRouter 故障**：几位用户讨论了 OpenRouter 出现的故障，并寻求其他用户对该平台状态的确认。
   - 有人提到服务应该是正常的，但部分用户的问题仍然存在。
- **用户界面工具对比**：一位用户介绍了 marblism.com，称其为与 Aider 类似的应用创建工具，并指出其专注于利用 user stories 进行功能开发。
   - 他们建议探索 Aider 如何以类似方式组织任务，以提升应用功能。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://some-azure-endpoit',">未找到标题</a>：未找到描述</li><li><a href="https://aider.chat/docs/llms/azure.html">Azure</a>：Aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML 配置文件</a>：如何使用 YAML 配置文件配置 Aider。</li><li><a href="https://aider.chat/docs/scripting.html">脚本化 Aider</a>：你可以通过命令行或 Python 对 Aider 进行脚本化操作。</li><li><a href="https://aider.chat/docs/usage/commands.html">聊天内命令</a>：使用 /add、/model 等聊天内命令控制 Aider。</li><li><a href="https://docs.litellm.ai/docs/providers/azure">Azure OpenAI | liteLLM</a>：API 密钥、参数
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1285361804431589397)** (8 条消息🔥): 

> - `Superflex AI Assistant`
> - `Claude 3.5 Artifacts`
> - `RethinkMCTS Algorithm`
> - `从 Figma 进行代码集成`
> - `用于推理代理的 Optillm` 


- **Superflex AI 转换 Figma 设计**：Superflex 允许用户直接从 [Figma 设计](https://www.producthunt.com/posts/superflex) 编写前端代码，平滑集成到现有项目中，并提供比以往专注于原型的迭代版本更强的功能。
   - 使用 Superflex 将线框图集成到代码库中是可行的，特别是对于使用 UI kit 制作的设计，这使其成为开发者的一个极具吸引力的选择。
- **关于 Claude 3.5 Artifacts Prompt 的见解**：一个分享的链接揭示了 **提取出的 Claude 3.5 系统提示词 (system prompt)**，提供了关于其 Artifacts 的详细见解；对于使用 AI 模型的开发者来说，这是一个非常有用的资源。
   - 该提示词可以通过一个 [GitHub gist](https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d) 查看，其中包含了增强自动生成能力所需的信息。
- **RethinkMCTS 增强 LLM 代码生成**：该论文介绍了 **RethinkMCTS**，这是一种利用 Monte Carlo Tree Search 的算法，通过详细的执行反馈来优化搜索策略，从而增强代码生成 ([查看论文](https://www.arxiv.org/abs/2409.09584))。
   - 该方法旨在解决以往代码生成任务中搜索质量受限的问题，可能会产生更相关的输出。
- **关于直接图像集成方法的讨论**：有人提出了关于 Superflex 与简单粘贴图像进行 UI 创建的对比，因为一些用户发现剪贴板方法在集成 UI 组件方面也很有效。
   - 这凸显了开发者社区内部关于将设计转化为功能代码的最佳实践的持续争论。
- **Optillm：优化 LLM 推理**：分享了一个 **Optillm GitHub 仓库** 的链接，强调这是一个专门为 LLM 设计的优化推理代理 ([GitHub 链接](https://github.com/codelion/optillm))。
   - 该工具旨在提高处理大型语言模型时的性能和可用性，对于希望简化工作流程的开发者来说至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.arxiv.org/abs/2409.09584">RethinkMCTS: Refining Erroneous Thoughts in Monte Carlo Tree Search for Code Generation</a>: 通过树搜索算法增强的 LLM Agent 在代码生成方面取得了显著的表现。然而，该领域目前的搜索算法由于几个原因导致搜索质量较低...</li><li><a href="https://github.com/codelion/optillm">GitHub - codelion/optillm: Optimizing inference proxy for LLMs</a>: LLM 优化推理代理。通过在 GitHub 上创建一个账户来为 codelion/optillm 的开发做出贡献。</li><li><a href="https://www.producthunt.com/posts/superflex"> Superflex - Write Front-End Code 10x Faster ⚡️ | Product Hunt</a>: Superflex 是一款 AI 助手，可将想法从 Figma 设计、图像或文本提示词转化为前端代码——匹配您的编码风格并利用您的 UI 组件。更快地构建更好的前端...</li><li><a href="https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d">Extracted Claude 3.5 Sonnet system prompt for artifacts</a>: 提取的 Claude 3.5 Sonnet Artifacts 系统提示词 - claude_35_artifacts_system_prompt.txt
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1285314911005577389)** (154 条消息🔥🔥): 

> - `GPU 与性能问题`
> - `模型训练挑战`
> - `LM Studio 的新功能`
> - `社区工具与扩展`
> - `用户体验与设置` 


- **GPU 性能故障排除**：一位用户对 LM Studio 未能利用其 GPU 表示沮丧，而是反复使用 CPU 和 RAM。另一位用户指出，GPU 设置位于 **Settings -> System Resources** 下，并确认可以通过 **Task Manager** 监控 GPU 使用情况。
   - 一个导致屏幕模糊的已知问题与抗锯齿 (anti-aliasing) 设置有关，建议禁用某些配置以进行改进。
- **训练小型语言模型**：一位用户报告称正在训练一个 100k 参数的模型，最初预估训练时间长达 5 天。通过调整 token 数量和 batch size，预估时间显著缩短至约 1.3 小时。
   - 有人对 data loader 中潜在的瓶颈表示担忧，这会导致训练期间漫长的等待，并讨论了如何有效地使用 PyTorch。
- **LM Studio 令人兴奋的新功能**：用户讨论了最近在 LM Studio 中引入的文档集成功能，这是社区的一个主要需求。热烈的反馈表明用户渴望测试该应用程序的更新版本。
   - 一位成员指出，LM Studio 的简洁性对于那些没有广泛 IT 知识的人来说是一个显著优势。
- **社区工具与扩展开发**：一位成员分享了他们为 LM Studio 开发的 Discord 应用，突显了社区驱动的开发努力。还有人请求提供扩展程序，反映了对增强功能的自定义工具的兴趣。
   - 回复显示社区兴趣日益增长，并有潜力在围绕 LM Studio 平台创建有用工具和扩展方面进行协作。
- **设置与功能的用户体验**：用户分享了关于 LM Studio 聊天中 system prompts 和功能集成的问题及说明。提供了访问系统设置的指导，以简化模型交互过程中的重复任务。
   - 对话强调了用户友好功能对于改善整体体验的重要性，特别是对于平台新手而言。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/datagemma-rag-27b-it-GGUF">lmstudio-community/datagemma-rag-27b-it-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/acting-nod-hmph-gif-18509831">Acting Nod GIF - Acting Nod Hmph - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1285387903249416192)** (46 条消息🔥): 

> - `LM Studio 的 GPU 推荐`
> - `双 GPU 配置`
> - `二手 GPU 市场见解`
> - `Intel ARC 性能`
> - `VRAM 在 LLM 中的重要性` 


- **为 VRAM 在 4090 和 4080 之间做出选择**：一位成员考虑购买 **4090** 以获得最大的 **VRAM**，但对其相对于 **4080** 的高昂价格感到纠结，质疑性能差距是否值得。
   - 另一位成员建议，两块 **4060 Ti** 可能会提供更多 VRAM，且更具成本效益，同时功耗更低。
- **双 GPU 配置的优点**：讨论表明，使用两块 **4060 Ti** 可以在不超出功耗限制的情况下最大化 VRAM，是一个实用的选择。
   - 参与者指出，使用相同的 GPU 可以简化配置，且精细的电源管理可以降低整体能源成本。
- **在市场上寻找二手 GPU**：成员们分享了在不同地区寻找 **二手 3090** 的见解，强调了价格和可用性等挑战。
   - 虽然有些人在 eBay 上找到了划算的交易，但其他人更喜欢像 **Kijiji** 这样的本地平台购买二手零件，提到的价格约为 **800-920 加元**。
- **Intel ARC 在 LLM 性能中的角色**：一位成员询问关于使用 **Intel ARC A770** 运行 LLM 的情况，引发了关于利用 SYCL 后端性能指标的讨论。
   - 有说法称 ARC 配置可以达到每秒 **34 tokens**，并有可能通过 **IPEX** 进一步提升。
- **VRAM 在 LLM 中的关键性**：围绕对充足 **VRAM** 的需求产生了担忧，强调大多数强大的模型可能需要比当前显卡提供的更多的显存。
   - 成员们讨论了他们在各种 GPU 上的 **token 生成** 速率体验，特别强调了 VRAM 容量的重要性。

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1285707527459180595)** (1 条消息): 

> - `Hugging Face API docs`
> - `TRL v0.10 release`
> - `Sentence Transformers v3.1`
> - `DataCraft for synthetic datasets`
> - `Core ML Segment Anything 2` 


- **翻新后的 Hugging Face API 文档**：Hugging Face [推出了全新的 API 文档](https://x.com/Wauplin/status/1835715850583564713)，重点介绍了更清晰的 **rate limits**、专门的 **PRO section**、增强的代码示例以及更详细的参数列表。
   - 此次更新旨在简化 **AI deployment** 并提升用户体验。
- **为 Vision-Language Models 引入 TRL v0.10**：[TRL v0.10](https://x.com/QGallouedec/status/1833893093793304950) 已发布新功能，仅需两行代码即可实现 **vision-language models** 的微调，恰逢 Mistral 发布 Pixtral。
   - 这种极简的方法使集成新模型变得更加容易和高效。
- **Sentence Transformers v3.1 发布**：最新的 [Sentence Transformers v3.1](https://x.com/tomaarsen/status/1833870859552928172) 包含一个 **hard negatives mining utility** 以改进模型训练，以及一个新的强力 Loss Function。
   - 它还支持使用 Streaming Datasets 和自定义模块进行训练，增强了模型开发的灵活性。
- **DataCraft 简化合成数据集创建**：[DataCraft](https://x.com/dvilasuero/status/1835711765570630017) 已推出，旨在帮助用户在无代码 UI 中使用自然语言创建 Synthetic Datasets，解决了生成高质量数据的挑战。
   - 它利用了数据集生成的最佳实践，使用户能够更轻松地构建有效的数据集。
- **Core ML Segment Anything 2 现已发布**：为 Core ML 推出的 [Segment Anything 2](https://x.com/pcuenq/status/1834616110475514343) 展示了 **on-device ML** 能力，并在 Mac 上提供了演示应用。
   - 这一进展指向了 **on-device AI** 应用充满希望的未来。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Wauplin/status/1835715850583564713)">来自 Wauplin (@Wauplin) 的推文</a>：我很高兴能揭晓我们翻新后的 Inference API 文档！我们正面解决了你们的反馈：更清晰的 rate limits，专门的 PRO section，更好的代码示例，以及详细的参数列表...</li><li><a href="https://x.com/QGallouedec/status/1833893093793304950)">来自 Quentin Gallouédec (@QGallouedec) 的推文</a>：时机完美！@MistralAI 发布了 Pixtral，这是他们的第一个多模态模型，而我们全新的 TRL 版本恰好增加了两行代码微调 vision-language models 的功能 🌟</li><li><a href="https://x.com/qlhoest/status/1829145570465722578)">来自 Quentin Lhoest 🤗 (@qlhoest) 的推文</a>：🤗Hugging Face Datasets 用户们请欢呼吧！我为 ✨PySpark✨ 编写了几行代码，用于从 HF Datasets 读取/写入。全部经过分布式优化！代码片段 / 文档和 JupyterLab 演示见下方 🧡</li><li><a href="https://x.com/tomaarsen/status/1833870859552928172)">来自 tomaarsen (@tomaarsen) 的推文</a>：Sentence Transformers v3.1 发布了！具有 hard negatives mining utility，可以从您的数据中获得更好的模型，一个新的强力 Loss Function，支持 Streaming Datasets 训练，自定义模块，Bug 修复...</li><li><a href="https://x.com/dvilasuero/status/1835711765570630017)">来自 Daniel Vila Suero (@dvilasuero) 的推文</a>：🧶 介绍 DataCraft：使用自然语言构建合成数据集！创建高质量的合成数据很困难。这是一个反复试验的过程，需要很多技巧。DataCraft 提供...</li><li><a href="https://x.com/pcuenq/status/1834616110475514343)">来自 Pedro Cuenca (@pcuenq) 的推文</a>：宣布 SAM 2 Studio 和 Core ML Segment Anything 2！我对 on-device ML 感到非常兴奋，并坚信它将成为 AI 未来的重要组成部分。我们将 Segment Anything 2 转换为了...</li><li><a href="https://x.com/OzzyGT/status/1834594141822406796)">来自 Alvaro Somoza (@OzzyGT) 的推文</a>：想知道如何使用 Diffusers 擦除/填充图像的部分内容吗？虽然花了一些时间，但我终于有了新的指南和 Space 供您尝试。您可以在这篇博客文章中阅读相关内容：https...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1285319311312814160)** (135 条消息🔥🔥): 

> - `短视频工具`
> - `Hugging Face Inference API 更新`
> - `FSDP GPU 占用`
> - `CogvideoX img2vid 能力`
> - `Hugging Face SQL Console 发布` 


- **短视频最佳工具**：一位成员建议使用 [visla.us](https://visla.us) 配合 OpenAI Visla 插件，认为它是创建 TikTok 等短视频的最佳工具。
   - 这引发了社区关于各种工具的效果和性能的讨论。
- **全新修订的 Inference API 文档亮相**：许多用户对新更新的 [Inference API 文档](https://huggingface.co/docs/api-inference) 表示兴奋，该文档包含更清晰的速率限制（rate limits）、更好的代码示例以及专门的 PRO 部分。
   - 随着 Hugging Face 上数据集的不断增长，此次更新旨在简化 AI 部署并提升用户体验。
- **关于 FSDP GPU 显存占用的困惑**：一位用户分享了在结合 FSDP 和 BF16 AMP 对 8B LLaMA 模型进行微调时，显存占用异常高的困惑，观察到 8 个 GPU 共使用了 29G 显存。
   - 建议包括使用原始 PyTorch 调用进行调试，以及探索 FSDP 中优化资源使用的潜力。
- **CogvideoX 的认知能力**：成员们讨论了新款 [CogvideoX img2vid](https://huggingface.co/spaces) 令人印象深刻的能力，指出其在生成视频时具有极高的效率和极低的 VRAM 占用。
   - 尽管最初有一些关于基础镜头的批评，但其他人称赞了它处理复杂场景（如行走或骑滑板车）的能力。
- **数据集 SQL Console 发布**：社区庆祝 Hugging Face 推出了 SQL Console 功能，允许用户直接在数据集上运行 SQL 查询，增强了可发现性和可用性。
   - 随着对数据集管理需求的增长，鼓励用户分享与此新功能相关的想法和 SQL 片段。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Wauplin/status/1835715850583564713">来自 Wauplin (@Wauplin) 的推文</a>: 我很高兴宣布我们全新修订的 Inference API 文档！我们正面解决了大家的反馈：更清晰的速率限制、专门的 PRO 部分、更好的代码示例以及详细的参数列表...</li><li><a href="https://huggingface.co/code-of-conduct#:~:text=Our%20Standards&text=Demonstrating%20empathy%20and%20kindness%20toward,and%20learning%20from%20the%20experience">行为准则 – Hugging Face</a>: 未找到描述</li><li><a href="https://docs.omniverse.nvidia.com/composer/latest/index.html">USD Composer 概览 &mdash; Omniverse USD Composer 最新文档</a>: 未找到描述</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/sql-console">在数据集上引入 SQL Console</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/sanchit-gandhi/whisper-jax-spaces">Whisper JAX - sanchit-gandhi 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html">使用 Fully Sharded Data Parallel (FSDP) 进行高级模型训练 — PyTorch 教程 2.4.0+cu121 文档</a>: 未找到描述</li><li><a href="https://github.com/unclemusclez/ollama-toolkit">GitHub - unclemusclez/ollama-toolkit: Ollama Toolkit 是一系列强大的工具集合，旨在增强您对 Ollama 项目（一个用于部署和扩展机器学习模型的开源框架）的使用体验。可以将其视为简化工作流程并释放 Ollama 全部潜力的一站式商店！</a>: Ollama Toolkit 是一系列强大的工具集合，旨在增强您对 Ollama 项目的使用体验...</li><li><a href="https://github.com/yt-dlp/yt-dlp/issues/10128">[youtube] 登录以确认你不是机器人。这有助于保护我们的社区 · Issue #10128 · yt-dlp/yt-dlp</a>: 请勿删除或跳过议题模板。我明白如果我故意删除或跳过任何必填项，我将被封禁。检查清单：我正在提问，而不是报告错误或请求功能...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1285363482815893621)** (5 messages): 

> - `学习 Manim`
> - `使用 PyTorch 构建 ML Data Pipelines`
> - `Hugging Face Dataset 问题` 


- **探索 ML Data Pipelines**：另一位用户分享了他们在学习使用 **PyTorch** 构建 **ML Data Pipelines** 的过程，重点是训练 **1D CNN** 分类器。
   - 这场讨论引发了关注，促使其他人寻求资源。
- **Hugging Face Dataset 中的图像问题**：一位成员在处理 **Hugging Face** 上的数据集时表示困惑，尽管确信是自己的失误，但仍无法看到图像。
   - 他们直接链接了 **synthetic drilling dataset**，邀请社区提供反馈。



**提及的链接**：<a href="https://huggingface.co/datasets/jonasmaltebecker/synthetic_drilling_dataset/viewer/default/validation?row=1">jonasmaltebecker/synthetic_drilling_dataset · Datasets at Hugging Face</a>：未找到描述

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1285339908197580810)** (11 messages🔥): 

> - `Inference API 文档改进`
> - `模型增长与下载讨论`
> - `AI 社区参与` 


- **翻新后的 Inference API 文档发布**：一位成员宣布推出了改进后的 [Inference API 文档](https://huggingface.co/docs/api-inference)，强调了更清晰的速率限制（rate limits）、专门的 PRO 部分、更好的代码示例以及详细的参数列表。
   - 另一位用户表达了兴奋之情，说道：*我太喜欢了 ❤️*。
- **冲向 100 万个模型**：围绕谁会先达到 100 万展开了讨论：是 flux 还是模型总数，并推测下周可能会实现 **1M models** 的目标。
   - 一位成员提到：*我正在获取一些统计数据，我们每周新增模型接近 4 万个 🤯*。
- **AI 爱好者 WhatsApp 群组**：有人请求建立一个专门讨论 AI 的 WhatsApp 群组。
   - 另一位成员针对此询问建议不要跨频道发帖（cross-posting）。



**提及的链接**：<a href="https://x.com/Wauplin/status/1835715850583564713>">Wauplin (@Wauplin) 的推文</a>：我非常激动地揭晓我们翻新后的 Inference API 文档！我们正面解决了大家的反馈：更清晰的速率限制、专门的 PRO 部分、更好的代码示例，以及详细的参数列表...

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1285328853073657949)** (5 messages): 

> - `Behavioral Biometric Recognition in Minecraft` (Minecraft 中的行为生物识别)
> - `PowershAI Multilingual Documentation` (PowershAI 多语言文档)
> - `Nvidia Mini-4B Model Release` (Nvidia Mini-4B 模型发布)
> - `HuggingFace Agent Registration` (HuggingFace Agent 注册)
> - `Continuous MFA and Ban Evasion Detection` (持续 MFA 与规避封禁检测)


- **Behavioral Biometric Recognition in Minecraft**: 一位成员展示了他们的模型，该模型通过 **Minecraft** 中的鼠标移动来识别玩家，旨在实现 **continuous MFA**（持续多因素认证）和检测 **ban evasion**（规避封禁）。
   - 该项目在 [GitHub](https://github.com/templateprotection/AimNet-Mouse-Dynamics) 上有一个开源仓库，强调了其在游戏之外的潜在应用。
- **PowershAI Documentation Translated**: 更新后的 **PowershAI 文档** 现已提供多种语言版本，全部使用该工具本身完成翻译，并托管在 [GitHub](https://github.com/rrg92/powershai) 上。
   - 欢迎成员们审阅翻译并提出新的建议，以增强可访问性。
- **Nvidia Releases Mini-4B Model**: Nvidia 推出了 **Mini-4B model**，以其紧凑的尺寸著称，但在超出设备限制运行时需要特定的 **Nvidia drivers**。
   - 该模型被宣传为其尺寸类别中性能最佳的模型，可在 [Hugging Face Space](https://huggingface.co/spaces/Tonic/Nemotron-Mini-4B) 上查看。
- **HuggingFace Agent Registration Suggestion**: 有人建议将最近发布的 Mini-4B 注册为 **HuggingFace agent**，这将使其能够查询 **SQL** 并与其他 Agent 集成。
   - 这种集成可以显著增强功能和用户交互性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Tonic/Nemotron-Mini-4B">Minitron - a Hugging Face Space by Tonic</a>: 未找到描述</li><li><a href="https://github.com/rrg92/powershai">GitHub - rrg92/powershai: Powershell + AI</a>: Powershell + AI。通过创建账户为 rrg92/powershai 的开发做出贡献。</li><li><a href="https://github.com/templateprotection/AimNet-Mouse-Dynamics">GitHub - templateprotection/AimNet-Mouse-Dynamics: An open sourced approach to One-Shot Learning for Mouse Dynamics recognition in PyTorch. This includes tools for data preprocessing, training both classification and embedding models, and evaluating model performance on a Minecraft dataset.</a>: 一个在 PyTorch 中实现鼠标动态识别 One-Shot Learning 的开源方法。包括数据预处理工具、分类和嵌入模型的训练，以及在 Minecraft 数据集上评估模型性能。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1285349051318931645)** (5 messages): 

> - `Downloading LLaMA3` (下载 LLaMA3)
> - `Using PyTorch` (使用 PyTorch)
> - `MLIR Conversion Tool` (MLIR 转换工具)


- **Need assistance with LLaMA3 download**: 一位成员请求帮助使用 **PyTorch** 下载并运行 **LLaMA3** 开源 **LLM**，并对任何指导表示感谢。
   - *“我没发现什么有用的信息，”* 反映了该模型入门的难度。
- **Clarification on PyTorch usage**: 另一位成员询问为什么选择 **PyTorch** 来实现 **LLaMA3**。
   - 这引起了初始用户的困惑，他寻求对该问题的进一步澄清。
- **Uploading LLaMA3 model locally**: 该成员澄清说，他们只是尝试在本地设置中加载 **LLaMA3** 模型。
   - 后续说明指出需要 **PyTorch** 是因为要兼容一个将 **PyTorch** 代码转换为 **MLIR** 的工具。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1285637296053551167)** (1 messages): 

> - `Gradio Office Hours` 


- **Join Gradio Office Hours Now!**: <@997168115802722304> 正在我们的 Discord 中主持 Gradio **Office Hours**，提供了一个讨论 **Gradio**、**HF** 和 **AI** 的机会。
   - 邀请所有人加入对话，更多信息请访问 [此链接](https://t.co/Dxeb0jaQ6e)。
- **Chat with Experts at Gradio**: 正在进行的 **Office Hours** 专为那些对 **Gradio** 话题感兴趣的人设计，例如新功能、来自 **HF** 的更新以及 **AI** 的进展。
   - 鼓励所有参与者加入并互动，这是一个提问和分享见解的好机会。



**提到的链接**: <a href="https://t.co/Dxeb0jaQ6e">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 非常适合玩游戏和与朋友放松，甚至可以建立全球社区。自定义你自己的空间来聊天、玩耍和聚会。

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1285364699139018852)** (99 条消息🔥🔥): 

> - `GPT-4o Performance`
> - `Alpha Rollouts`
> - `AI Implementation in Businesses`
> - `Custom GPTs for Code Snippets`
> - `LLM Benchmarks` 


- **GPT-4o 在 GeoGuessr 中表现惊人**：成员们对 **GPT-4o** 在 **GeoGuessr** 中的表现感到惊讶，尽管它尚未击败顶级专家选手。
   - 一位成员指出，它在响应时并没有遵循 **o1-mini** 模型预期的速度。
- **意外的 Alpha 推送引发讨论**：**Alpha** 功能可能存在的非预期推送引起了用户的好奇和猜测。
   - 用户感到沮丧，因为尽管功能看起来可用，但部分人遇到了故障。
- **向本地商家销售 AI 解决方案**：一位成员表达了向本地商家销售 **AI 解决方案** 的信心，并向有经验的人寻求建议。
   - 对话集中在成交策略和推动 AI 技术落地的方法上。
- **用于代码片段管理的 Custom GPTs**：一位用户询问了有效管理和重用代码片段的 AI 解决方案，强调了更好组织方式的需求。
   - 成员们建议使用 **Custom GPTs**，并强调了上传注释良好的知识库的重要性。
- **寻找 LLM 基准测试信息**：一位成员询问了提供各种 **LLM 模型** 全面基准测试的资源。
   - 其他人推荐使用 **lmsys.org** 并咨询 **GPT-4o** 以获取有用的选项。



**提及链接**：<a href="https://status.openai.com/">OpenAI Status</a>：未找到描述

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1285325298380050566)** (26 条消息🔥): 

> - `Fine-tuning Limitations`
> - `Advanced Voice Mode Availability`
> - `Custom GPT Sharing`
> - `Token Refresh Confusion` 


- **微调任务再次触及硬限制**：一位用户对微调任务超过其硬限制感到沮丧，成本为 **$24.31**，而配额仅剩 **$19.91**。
   - 另一位成员推测这可能是折扣而非配额问题，引发了关于是否继续该任务的讨论。
- **高级语音模式（Advanced Voice Mode）何时可用？**：多位成员表示他们正在使用 Plus，但尚未获得 **Advanced Voice Mode** 的访问权限。
   - 一位用户提到 **预计可用时间** 是在 **秋季末** 之前。
- **需要 Custom GPT 共享指南**：一位用户寻求帮助，希望在不透露完整账单姓名的情况下分享他们定制的 GPT，因为该选项显示为灰色。
   - 他们询问启用 builder profile 是否允许更改显示名称，并询问是否有用户愿意测试他们的 GPT。
- **Token 刷新时间引起关注**：一位用户对 **免费 Token** 的刷新时间表示不确定，并因担心产生意外费用而犹豫是否进行测试。
   - 他们提到 **ask-ai** 频道的一个建议指出刷新时间在 **UTC 时间午夜**。
- **关于页面加载问题的讨论**：一位用户报告在尝试加载页面时遇到 **404 错误**，引起了其他成员的关注。
   - 这一技术故障似乎引起了其他人的共鸣，但未作进一步阐述。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1285626364900413501)** (3 条消息): 

> - `Auto prompt for ideogram/midjourney`
> - `Prompt sharing practice`
> - `Library resources` 


- **分享 Ideogram/Midjourney 自动提示词**：一位成员创建了一个 **Ideogram/Midjourney 自动提示词**，详细说明了所有必要步骤，并鼓励其他人进行评价。
   - 他们表示愿意广泛分享，并表示欢迎反馈。
- **对创意提示词的兴趣**：该成员询问其他人是否对新创建的 Ideogram/Midjourney 提示词感兴趣。
   - 该提示词 *免费分享*，被定位为社区中其他人的资源。
- **关于官方库的讨论**：简要提到了 **官方库**，但随后没有进行详细讨论。
   - 这一提及的背景仍然模糊，需要在未来的对话中进一步探索。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1285626364900413501)** (3 messages): 

> - `Ideogram/Midjourney 的自动提示词 (Auto prompt)`
> - `官方库 (Official libraries)` 


- **为 Ideogram/Midjourney 创建自动提示词**：一位成员分享了一个用于 **Ideogram/Midjourney** 的自动提示词，包含了所有必要步骤，并提到该提示词可免费分享。
   - 该成员鼓励其他人对提示词进行**评分**，并询问是否有人感**兴趣**。
- **关于官方库的讨论**：提到了“**官方库 (official libraries)**”一词，暗示这可能是用户感兴趣的话题。
   - 未提供有关此提及的详细信息或背景，留待进一步讨论。


  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1285584253253455923)** (1 messages): 

> - `OpenRouter 集成`
> - `Google Sheets 插件功能`
> - `更新与改进`
> - `用户反馈`
> - `支持多种模型` 


- **OpenRouter 成功集成到 Google Sheets**：应用户要求，OpenRouter 已添加到 [GPT Unleashed for Sheets](https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147) 插件中，并免费提供。
   - *我个人也非常喜欢使用 OpenRouter*，希望在此过程中能收到宝贵的反馈并吸引更多用户。
- **创新功能增强 Google Sheets 性能**：该插件包含“任务 (jobs)”、“上下文 (contexts)”和“模型预设 (model presets)”等功能，以简化提示词工程 (Prompt engineering) 并提高生产力。
   - 用户可以为提示词分配短代码，从而更轻松地重用和优化 AI 输出。
- **九月更新提升插件功能**：最近的更新增加了对 Anthropic 的 **Claude** 的支持，增强了 UX/UI，并提升了整体性能。
   - 通过 OpenRouter 集成，用户现在可以在插件中访问 **100 多个模型**。
- **用户评价凸显插件优势**：用户赞赏该插件**永久免费**，支持众多流行的语言模型，并简化了 AI 工具的构建。
   - 主要优势包括大幅提升生产力，以及对结果和 API 调用进行有效跟踪。



**Link mentioned**: <a href="https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147">GPT Unleashed for Sheets™ - Google Workspace Marketplace</a>: 未找到描述

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1285324078953267313)** (117 条消息🔥🔥): 

> - `OpenRouter API 问题`
> - `Gemini 图像生成`
> - `Prompt Caching 使用`
> - `Mistral API 降价`
> - `模型性能与评分` 


- **OpenRouter API 出现问题**：多名用户报告了访问 OpenRouter 时遇到的问题，特别是关于 `o1` 模型，这导致了对速率限制（rate limits）和请求耗尽的困惑。
   - 一名用户注意到瑞士出现了暂时性停机，但随后确认在初始问题后功能已恢复。
- **Gemini 在图像生成方面的能力差异**：用户讨论了 Gemini 在其官网上的图像生成能力与其通过 OpenRouter 表现出的性能之间的差异。
   - 澄清指出，Gemini 聊天机器人集成了来自 Imagen 模型的图像生成功能，而 OpenRouter 则为 Gemini 模型调用 Google Vertex AI。
- **深入了解 Prompt Caching**：关于 Prompt Caching 的讨论阐明了其成本效益，允许重复使用 Prompt 以减少后续查询的费用。
   - 用户举例说明了可以缓存关键 Prompt 组件的场景，从而在多次相关查询中节省成本。
- **Mistral API 大幅降价**：公告显示 Mistral API 大幅降价，Large 2 模型的新价格定为 $2，与其他供应商相比具有优势。
   - 这一价格变动被视为具有竞争力，可能会影响用户对 API 请求所选模型的决策。
- **模型性能讨论**：用户对视觉模型的性能发表了不同看法，指出 Google 的 Flash 模型在某些方面似乎优于 Pixtral 12B。
   - 对话还包括了对持续测试和使用场景中常见的速率限制（rate limits）及性能问题的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/september-24-release/">AI in abundance</a>：推出免费 API，全面优化定价，新增企业级 Mistral Small，并在 le Chat 上提供免费视觉能力。</li><li><a href="https://openrouter.ai/settings/keys">Keys | OpenRouter</a>：管理您的密钥或创建新密钥</li><li><a href="https://gemini.google.com.">‎Gemini - 激发灵感的聊天</a>：Bard 现已更名为 Gemini。从 Google AI 获取写作、规划、学习等方面的帮助。</li><li><a href="https://openrouter.ai/credits">Credits | OpenRouter</a>：管理您的额度和支付历史</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>：查看您在 OpenRouter 上使用模型的情况。</li><li><a href="https://openrouter.ai/activity?api_key_id=359060">Activity | OpenRouter</a>：查看您在 OpenRouter 上使用模型的情况。</li><li><a href="https://mistral.ai/news/pixtral-12b/">Announcing Pixtral 12B</a>：Pixtral 12B - 首个多模态 Mistral 模型。Apache 2.0。</li><li><a href="https://openrouter.ai/models/mistralai/pixtral-12b:free">Pixtral 12B (free) - API, Providers, Stats</a>：来自 Mistral AI 的首个图像转文本模型。其权重按照其传统通过 torrent 发布：https://x。通过 API 运行 Pixtral 12B (免费)</li><li><a href="https://openrouter.ai/activity?api_key_id=496719">Activity | OpenRouter</a>：查看您在 OpenRouter 上使用模型的情况。</li><li><a href="https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb">anthropic-cookbook/misc/prompt_caching.ipynb at main · anthropics/anthropic-cookbook</a>：展示使用 Claude 的一些有趣且有效方式的 Notebook/Recipe 集合。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1285334283363291207)** (14 messages🔥): 

> - `Metal 讨论组`
> - `ZML 项目见解`
> - `Zig 编程语言`
> - `Zig 中的 ATen`
> - `Zig 对 CUDA 的支持` 


- **对创建 Metal 讨论组的兴趣**：一名成员建议创建一个 Metal 讨论组，想知道是否有人有兴趣一起探索 [Metal-Puzzles GitHub 仓库](https://github.com/abeleinin/Metal-Puzzles)。
   - 他们强调该项目专注于通过解决谜题，在协作环境中学习 Metal。
- **探索 ZML 高性能 AI 栈**：成员们分享了对 [ZML 项目](https://github.com/zml/zml)的兴趣，指出其在高性能 AI 推理方面的潜力，特别是对从事编程语言设计的人员很有吸引力。
   - 他们讨论了在像 PyTorch 这样复杂的框架中，Zig 是否能比 C++ 简化开发。
- **比较 Zig 与 C++ 的适用性**：讨论了底层 Zig 是否能在编程范式存在差异的情况下，保持与 Python 的兼容性。
   - 成员们反思了 PyTorch 内部面临的挑战，并辩论了 Zig 可能带来的潜在改进。
- **对 Zig 中 ATen 的好奇**：一名成员表达了对如果用 Zig 实现 ATen 库会是什么样子的兴趣，并展望了潜在的收益和优化。
   - 这引发了关于对 AI 框架及其底层基础设施影响的讨论。
- **Zig 对 CUDA 的支持**：一名成员提到在 Zig 编程环境中支持 CUDA 的重要性，认为这将增强该语言在 AI 领域的适用性。
   - 这反映了利用 Zig 进行高性能计算任务（包括涉及 GPU 加速的任务）的广泛兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/zml/zml/">GitHub - zml/zml: High performance AI inference stack. Built for production. @ziglang / @openxla / MLIR / @bazelbuild</a>：高性能 AI 推理栈。为生产环境构建。@ziglang / @openxla / MLIR / @bazelbuild - zml/zml</li><li><a href="https://github.com/abeleinin/Metal-Puzzles">GitHub - abeleinin/Metal-Puzzles: Solve Puzzles. Learn Metal 🤘</a>：解决谜题。学习 Metal 🤘。通过在 GitHub 上创建账号为 abeleinin/Metal-Puzzles 开发做贡献。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1285394688911605760)** (9 messages🔥): 

> - `Triton 开发者大会`
> - `Proton 教程`
> - `Triton CPU/ARM 开发`
> - `Keynote 提到了社区`
> - `CUDA 社区` 


- **关于 Triton 开发者大会出席情况的询问**：一名成员询问了关于参加明天 **Triton 开发者大会** 的情况，寻求与组织方人员建立联系。
   - *如果你参与了组织工作，请私信我！*
- **Proton 教程给与会者留下深刻印象**：一名参会者称赞了 **Proton 教程**，将其描述为一个非常棒的工具。
   - 他们链接了该教程的 [notebook](https://github.com/Deep-Learning-Profiling-Tools/triton-samples/blob/main/Triton_Tools_Tutorial.ipynb) 以供进一步探索。
- **Keynote 中对社区的致谢**：一名成员报告称，在 **Triton 大会 Keynote** 中受到了另一位参与者的点名致谢。
   - 现场有一些轻松的玩笑，鼓励更多人加入围绕大会的讨论。
- **关于 Triton CPU/ARM 开发的对话**：有人询问目前 **Triton CPU 和 ARM** 开发的性质，特别是它是开源还是闭源的。
   - 成员们似乎渴望了解该项正在进行的工作的具体细节。
- **赞扬 CUDA 社区服务器**：一名成员对该服务器表示赞赏，称其绝对是进行 CUDA 讨论的**最佳**场所。
   - 其他人也表达了同样的看法，增强了社区氛围。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/Deep-Learning-Profiling-Tools/triton-samples/blob/main/Triton_Tools_Tutorial.ipynb">triton-samples/Triton_Tools_Tutorial.ipynb at main · Deep-Learning-Profiling-Tools/triton-samples</a>：为 GitHub 上的 Deep-Learning-Profiling-Tools/triton-samples 开发做贡献。</li><li><a href="https://github.com/triton-lang/triton/blob/main/docs/meetups/02-20-2024/Proton.pdf">triton/docs/meetups/02-20-2024/Proton.pdf at main · triton-lang/triton</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1285509111160508447)** (2 messages): 

> - `Flash Attention v2 with learnable bias`
> - `BitBlas and Triton-like language` 


- **探索带有可学习偏置的 Flash Attention v2 实现**：一位成员询问了如何实现带有大小为 **[B, H, N, N]** 的可学习偏置（learnable bias）的 **Flash Attention v2**，该偏置在 softmax 操作前添加并需要计算梯度。
   - *关于如何开始着手解决这个问题有什么建议吗？*
- **BitBlas 作者创建了一种极具前景的类 Triton 语言**：**BitBlas** 的作者正在开发一种基于 **TVM** 的新型**类 Triton 语言**，展现出巨大的潜力。正如 [test_tilelang_dequantize_gemm.py](https://github.com/microsoft/BitBLAS/blob/main/testing/python/tilelang/test_tilelang_dequantize_gemm.py) 示例所示，如果成功，这可能会带来重大进展。
   - **BitBlas** 库专注于支持混合精度矩阵乘法，特别是针对**量化 LLM 部署**。



**提到的链接**：<a href="https://github.com/microsoft/BitBLAS/blob/main/testing/python/tilelang/test_tilelang_dequantize_gemm.py">BitBLAS/testing/python/tilelang/test_tilelang_dequantize_gemm.py at main · microsoft/BitBLAS</a>：BitBLAS 是一个支持混合精度矩阵乘法的库，特别适用于量化 LLM 部署。 - microsoft/BitBLAS

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1285552277444362302)** (1 messages): 

> - `SK Hynix AiMX-xPU`
> - `In-Memory Computing`
> - `LLM Inference`
> - `Power Efficiency` 


- **SK Hynix 在 Hot Chips 2024 上展示 AiMX-xPU**：在 **Hot Chips 2024** 期间，SK Hynix 介绍了 **AiMX-xPU** 和 **LPDDR-AiM**，展示了他们在针对 **LLM 推理**的存内计算（In-Memory Computing）方面的进展。
   - 该创新允许数据转换直接在内存中进行，通过减少互连传输来提高**能效**和速度。
- **存内计算助力内存受限的 LLM**：SK Hynix 强调了他们致力于支持 **LLM** 的承诺，由于这些模型严重依赖内存访问，其特性被定义为**内存受限（memory-bound）**。
   - 这一重点与他们旨在专门为 AI 模型简化操作的新型计算解决方案相契合。



**提到的链接**：<a href="https://www.servethehome.com/sk-hynix-ai-specific-computing-memory-solution-aimx-xpu-at-hot-chips-2024/">SK Hynix AI-Specific Computing Memory Solution AiMX-xPU at Hot Chips 2024</a>：SK Hynix 在 Hot Chips 2024 上展示了其 AiMX-xPU 概念，旨在实现更高效的存内 LLM 推理计算。

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1285326893717127192)** (2 messages): 

> - `Learning Custom CUDA Kernels`
> - `Neural Network Training` 


- **初学者对自定义 CUDA Kernel 知识的探索**：一位成员表达了在接下来的**六周**内学习并教导他人编写自定义 CUDA Kernel 的抱负。
   - *虽然感觉像个神经网络专家*，他们分享了自己的背景，但也承认自己在 CUDA 开发方面是**初学者状态**。
- **来自社区的鼓励**：另一位成员对最初的询问做出了积极回应，表示支持并愿意提供帮助。
   - 这种互动凸显了社区对新人的**欢迎**态度。


  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1285627961055252522)** (3 messages): 

> - `Implementation using Metal or WebGPU`
> - `CUDA Alternatives`
> - `FAQs on GPU Programming`
> - `Metal Channel in Discord` 


- **在 PMPP 中使用 Metal 或 WebGPU 的可行性**：.mattrix96 询问在没有 Nvidia GPU 的情况下，是否可以使用 **Metal** 或 **WebGPU** 代替 **CUDA** 来学习 PMPP 这本书。
   - 这一担忧凸显了在硬件受限时对 GPU 编程替代方案的需求。
- **学习 CUDA 的推荐方法**：mr.osophy 分享了来自 FAQ 的指导，指出学习者理想情况下应至少涵盖 PMPP 的**第 6 章**，以掌握 **CUDA** 的基础概念，因为这些技能可以迁移到其他平台。
   - 建议的策略包括边做边学、应对挑战，并通过相应的 Discord 频道寻求帮助。
- **关于 Metal 频道的讨论**：在收到关于学习替代方案的见解后，.mattrix96 表示有兴趣查看 **Metal 频道**。
   - 他们承认了合适硬件的必要性，并表示如果需要，愿意寻找解决方案。


  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1285509089576751166)** (4 条消息): 

> - `H100 purse`
> - `GH100 confusion`
> - `High pricing concerns` 


- **H100 purse 定价离谱**：一名成员指出一款 [H100 purse](https://gpupurse.com/products/h100-purse) 标价高达 **$65,536.00 USD**，且没有明显的促销迹象。
   - 另一位成员指出这很可疑，因为**该物品甚至不是 H100**，表明可能存在诈骗。
- **对天真买家的怀疑**：有人分享道 *“事实是他们只需要一个天真且被炒作冲昏头脑的人花 6.5 万美元买下它”*，强调了对剥削行为的担忧。
   - 这种情绪反映了成员们对所谓高价值产品定价策略的普遍怀疑。
- **GH100 品牌混淆**：讨论中一名成员透露，如果放大观察，硅片上实际上写着 **GH100**。
   - 这表明商品详情页可能存在误导，进一步加剧了对该市场的不信任。



**提到的链接**：<a href="https://gpupurse.com/products/h100-purse">H100 Purse</a>：装有稀有且独一无二的 GPT-4 训练 GPU 的手提包。该手提包受出口管制。

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1285316900548776070)** (37 条消息🔥): 

> - `RMSNorm Implementation`
> - `FP8 Stability Issues`
> - `Consistency between Python and C/CUDA`
> - `Llama 3 Token Support`
> - `Dynamic Threadgroup Sizing` 


- **RMSNorm 实现进度**：一名成员正在努力添加 RMSNorm 支持，最近相应地修改了 layer norm kernel，并重点审查 `rmsnorm_forward_kernel6`。
   - 他们最初观察到 Python 和 C/CUDA 之间存在 ~1e-3 的差异，但后来发现这是由于使用了 bf16 精度而非 fp32 导致的。
- **FP8 端到端功能已恢复**：新的基于 tensor 的方法已成功恢复了前向和后向功能的 FP8 端到端能力。
   - 未来的工作将包括清理实现、重新添加多 GPU 支持，以及测试与之前方法的性能收敛性。
- **Python 与 CUDA 之间的一致性检查**：正在两个终端上测试 Llama 3 分支，以确保 Python 和 C/CUDA 实现之间的激活值（activations）一致。
   - 成员们配置了他们的环境，以确保在训练过程的前向传播（forward pass）期间激活值匹配。
- **解决动态 Threadgroup 分配问题**：一名成员表示，通过新的更改，如果超过共享内存（shared memory）限制，动态 Threadgroup 调整可以轻松进行调整。
   - 因此，他们决定不为 kernel 内存限制实现回退方案，而是依赖动态调整功能。
- **Llama 3 Token 的实现**：dataloader 已更新以支持 Llama 3 token，采用了新的 uint32_t 数据类型，取代了之前的 uint16_t。
   - 此外，已添加 RMSNorm 前向计算，其输出与 Llama 3 Encoder 前向计算匹配，同时正在准备进一步的调整。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/757">RMSNorm - WIP by gordicaleksa · Pull Request #757 · karpathy/llm.c</a>：WIP - 添加 RMSNorm 支持。</li><li><a href="https://github.com/karpathy/llm.c/pull/757/files">RMSNorm - WIP by gordicaleksa · Pull Request #757 · karpathy/llm.c</a>：WIP - 添加 RMSNorm 支持。</li><li><a href="https://github.com/karpathy/llm.c/pull/754">add llama 3 support to llm.c by karpathy · Pull Request #754 · karpathy/llm.c</a>：此分支从复制 train_gpt2.cu 和 test_gpt2.cu 开始，但在合并回 master 之前，这两个文件（以及其他文件）将进行更改以整合 Llama 3.1 支持。</li><li><a href="https://github.com/ademeure/llm.c/blob/llmc_reorg2/llmc/layernorm.cuh">llm.c/llmc/layernorm.cuh at llmc_reorg2 · ademeure/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。欢迎在 GitHub 上为 ademeure/llm.c 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1285314372352213054)** (15 messages🔥): 

> - `BitNet efficiency`
> - `SK Hynix 的内存计算 (In-memory computing)`
> - `三进制打包方法`
> - `用于神经网络的定制芯片`
> - `用于打包的查找表 (Lookup tables)` 


- **BitNet 的打包策略受到审视**：讨论表明，在 **8-bit 空间中打包 5 个三进制值**比传统的 2-bit 打包方法更有效，尽管实现起来比较复杂。
   - 一位成员分享了演示打包和拆包过程的代码，并考虑避免使用取模和除法以优化性能。
- **SK Hynix 展示内存计算 (In-memory computing)**：在 **Hot Chips 2024** 上，SK Hynix 介绍了用于 LLM 推理的 **in-memory computing** 进展，利用了他们的 AiMX-xPU 和 LPDDR-AiM 技术。
   - 这种方法通过直接在内存中进行计算来降低功耗并提高效率，这对于通常受内存限制 (memory-bound) 的 LLM 至关重要。
- **探索查找表 (Lookup Tables) 的效用**：一位成员询问了使用 **Lookup Tables (LUT)** 来增强之前讨论的打包方法效率的潜在好处。
   - 将 LUT 与打包值结合的实用性正在考虑中，强调需要进一步检查。
- **定制芯片开发讨论**：成员们讨论了一家名为 **Deepsilicon** 的新公司，该公司专注于为 AI 计算构建定制硬件和软件，声称运行所需的 RAM 显著减少。
   - 针对其雄心勃勃目标的司行性提出了担忧，突显了人们对创新 AI 计算方法的持续关注。
- **对 BitNet 实现的困惑**：成员们辩论了 BitNet 论文中使用的 **2-bit 实现**，质疑其有效性以及与 GPU runtime 性能的相关性。
   - 他们承认需要深入研究论文中关于 embedding、LM-head 和量化 (quantization) 策略的细节。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.servethehome.com/sk-hynix-ai-specific-computing-memory-solution-aimx-xpu-at-hot-chips-2024/">SK Hynix AI-Specific Computing Memory Solution AiMX-xPU at Hot Chips 2024</a>：SK Hynix 在 Hot Chips 2024 上展示了其 AiMX-xPU 概念，旨在实现更高效的内存内 LLM 推理计算。</li><li><a href="https://www.deepsilicon.net">deepsilicon</a>：未找到描述</li><li><a href="https://x.com/sdianahu/status/1833186687369023550">Diana (@sdianahu) 的推文</a>：Deepsilicon 运行神经网络所需的 RAM 减少了 5 倍，速度提高了约 20 倍。他们正在为此构建软件和定制芯片。有趣的是，他们已经通过软件证明了这一点，你甚至可以尝试一下。在 w...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1285336181017411635)** (9 messages🔥): 

> - `Hack Ideas 讨论`
> - `点云配准 (Point Cloud Registration) Kernel`
> - `PyTorch Conference 聚会`
> - `学生票价咨询` 


- **优先确定 Hack Session 的想法**：鼓励成员查看 hack-ideas 线程中的想法，并对感兴趣的项目点赞，以帮助在 Hack Session 之前确定重点关注的领域。
   - 这将有助于简化规划并统一参与者的兴趣。
- **3D 计算机视觉的新想法**：一位新成员介绍了一个用于**点云配准 (ICP)** 的自定义 Kernel 想法，强调了它在 **3D 计算机视觉**中的作用，并表示愿意在其他项目上进行协作。
   - *“我的主要目标是学习和获得乐趣。”* 反映了他们的积极态度。
- **CUDA HACK 的聚会计划**：一位成员提议为同时参加 **PyTorch Conference** 和 **CUDA HACK** 的人员举办聚会，建议组建一个小组在活动期间建立联系。
   - 讨论包括确认出席情况并建立参与者之间的友谊。
- **PyTorch Conference 的学生票价**：一位成员询问了获得 **PyTorch Conference** 学生票价的可能性，分享了他们关于教育邮箱仍处于激活状态的情况。
   - 他们计划在探索 Hack Session 议程的同时，如果价格符合预算就参加。
- **对松鼠头像的赞赏**：发生了一段关于用户**松鼠头像**的轻松交流，称赞其可爱，该成员也给出了有趣的回复。
   - *“我会把这话转达给松鼠的。”* 为对话增添了幽默感。


  

---

### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1285358965269725235)** (1 条消息): 

> - `Triton LayerNorm 问题`
> - `Tensor Parallelism 与训练 MoEs` 


- **Triton LayerNorm 引入了不一致性**：一位成员报告了在使用 **Tensor Parallelism > 1** 时，Triton LayerNorm 和 RMSNorm 实现存在问题，指出 *“参数梯度以非确定性方式累积”*，导致结果不一致。
   - 此问题特别影响了他们训练 **Mixture of Experts (MoEs)** 的尝试，促使他们寻求替代实现方案。
- **向 Liger 团队寻求关于 Triton 的见解**：该成员希望向 Liger 团队确认，鉴于上述问题，他们是否测试过其 Kernel 的 Triton 实现。
   - 他们标记了另一位成员，认为该成员可能对这个问题及其影响有更深入的理解。


  

---


### **CUDA MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1285384858448691263)** (5 条消息): 

> - `Metal Puzzles GitHub 仓库`
> - `直播解题环节`
> - `会议` 


- **在 GitHub 上探索 Metal Puzzles**：[Metal Puzzles GitHub 仓库](https://github.com/abeleinin/Metal-Puzzles) 鼓励用户在学习 Metal 编程的同时解决谜题。
   - 该项目旨在促进 Metal 社区的协作和知识共享，同时增加趣味性。
- **关于直播解题环节的提议**：鉴于繁忙的会议日程，一位成员提议下周组织一次直播解题环节。
   - 另一位成员热情地表示同意，并对这个想法感到兴奋。
- **新人开始尝试谜题**：一位新人提到他们开始了谜题之旅，表明社区的兴趣正在增长。
   - 这反映了一个积极的趋势，越来越多的成员参与到解题活动中。



**提到的链接**：<a href="https://github.com/abeleinin/Metal-Puzzles">GitHub - abeleinin/Metal-Puzzles: Solve Puzzles. Learn Metal 🤘</a>：解决谜题。学习 Metal 🤘。通过创建 GitHub 账户为 abeleinin/Metal-Puzzles 的开发做出贡献。

  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1285319549813657630)** (66 条消息🔥🔥): 

> - `NousCon 咨询`
> - `AI 模型 Hermes 3 的使用`
> - `InstantDrag 开发`
> - `针对 Perplexity 的 CPLX 扩展`
> - `Claude 3.5 的越狱` 


- **分享 NousCon 地点详情**：一位成员询问 NousCon 的地点，另一位成员确认详情将在当晚发布。
   - 这引发了关于未来潜在活动的讨论，包括在 NYC 等地举办的想法。
- **对使用 AI 模型 Hermes 3 的兴趣**：一位新成员表示有兴趣使用 AI 模型 Hermes 3，并寻求商务咨询的联系方式。
   - 另一位用户建议联系特定成员以获取更多信息。
- **关于 InstantDrag 的讨论**：一位用户强调 InstantDrag 是一种现代的基于拖拽的图像编辑解决方案，在无需掩码或文本提示的情况下提高了交互性和速度。
   - 讨论中将其与 DragGAN 进行了比较，并指出在应用程序中实现更快编辑的潜力。
- **宣布针对 Perplexity 的 CPLX 扩展**：推出了 Perplexity 的 CPLX 扩展 Alpha 版本，其特点是拥有一个独立于主输出的 Scratchpad。
   - 进一步的讨论揭示了它与 “scrapthpad-think” 框架的集成，展示了新的功能。
- **成功实现 Claude 3.5 Sonnet 的越狱**：一位用户自豪地分享了他们成功为 Claude 3.5 Sonnet 创建越狱的消息，该模型被描述为最难攻破的模型之一。
   - 他们指出，虽然受到了他人工作的启发，但他们的方法是独特且有效的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Ffftdtd5dtft/Hermes-3-Llama-3.1-8B-IQ1_S-GGUF">Ffftdtd5dtft/Hermes-3-Llama-3.1-8B-IQ1_S-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/_akhaliq/status/1835677372344873377?t=Zkttn9BN3f0bv5lGZAfcZw&s=19">来自 AK (@_akhaliq) 的推文</a>：InstantDrag 提升基于拖拽图像编辑的交互性。讨论：https://huggingface.co/papers/2409.08857。基于拖拽的图像编辑最近因其交互性和效率而广受欢迎...</li><li><a href="https://github.com/XingangPan/DragGAN">GitHub - XingangPan/DragGAN: Official Code for DragGAN (SIGGRAPH 2023)</a>：DragGAN 官方代码 (SIGGRAPH 2023)。通过创建 GitHub 账户为 XingangPan/DragGAN 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1285377361587605514)** (18 条消息🔥): 

> - `参数与 RAM 估算`
> - `模型训练数据效率`
> - `缩放参数与 Token`
> - `LLM 中的最优算力使用`
> - `Llama 模型数据缩放` 


- **参数估算偏差**：围绕 **1B 参数模型**的内存需求展开了讨论，根据训练上下文的不同，估算值在 **14GB** 到 **40GB** 之间波动。
   - 这种差异引发了对不同训练类型影响的询问，强调了**所有参数都会影响内存需求**。
- **参数与数据相关性澄清**：成员们辩论了参数是否是训练数据量的直接代指，结论是**更多参数允许更好的模式提取**，但两者并非严格相关。
   - 这达成了共识：虽然它们通常同步缩放，但并非必须如此，这凸显了模型训练的复杂性。
- **参数与 Token 的独立控制**：指出可以独立控制参数和 Token，并建议以 **1:1 比例**进行缩放以实现最优算力（Compute）利用。
   - 然而，成员们指出像 **Llama** 这样的模型，其训练数据量往往远超其参数量所暗示的规模。
- **详细内存需求计算**：为了计算内存需求，建议的公式是将参数数量乘以精度和优化器类型的各种系数，从而得出所需 RAM 的粗略计数。
   - 最终的内存估算还需要根据**激活值需求（activation requirements）**进行进一步调整，而这可能会有很大差异。
- **开源 LLM 处理长回答的挑战**：一位成员询问了使用 **gpt4all** 等开源模型**发送和接收大型 Prompt** 的策略。
   - 这反映了社区对优化与大语言模型交互以获得更好性能的持续关注。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1285326215414026330)** (3 条消息): 

> - `缩放 LLM 推理`
> - `研究中的分块阶段 (Chunking Phases)` 


- **Transformer 随缩放的性能极限**：在最近的一条 [推文](https://x.com/denny_zhou/status/1835761801453306089?s=46&t=VBhI-dqaQfawcUDHNO0L9A) 中强调，从数学上讲，如果允许 Transformer 生成尽可能多的**中间推理 Token**，它可以解决任何问题，并断言**恒定深度已足够**。
   - 这一见解与一篇即将发表的论文相关，详见此 [arXiv 链接](http://arxiv.org/abs/2402.12875)，该论文将在 **ICLR 2024** 上展示。
- **关于分块阶段研究的咨询**：有人请求提供与 *分块阶段 (chunking phases)* 和 *近似 (approximation)* 技术相关的**顶尖**且**最新研究论文**。
   - 这反映了对理解该研究领域当前方法论的持续兴趣。



**提到的链接**：<a href="https://x.com/denny_zhou/status/1835761801453306089?s=46&t=VBhI-dqaQfawcUDHNO0L9A">来自 Denny Zhou (@denny_zhou) 的推文</a>：缩放 LLM 推理时的性能极限是什么？潜力无限。我们已经从数学上证明了 Transformer 可以解决任何问题，只要允许它们生成尽可能多的中间...

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1285321676741148674)** (3 messages): 

> - `ChatGPT o1-preview`
> - `开发环境中的 RL`
> - `iText2KG 和 SeekTopic 算法`
> - `LLM 生成研究想法` 


- **ChatGPT o1-preview 展示编程实力**：@realGeorgeHotz 宣称 **ChatGPT o1-preview** 是第一个具备编程能力的模型，估计其智商为 **120 IQ**。
   - 他对开发中的**强化学习（Reinforcement Learning）**表示强烈乐观，特别是在编写和测试代码方面，并分享了一个 ChatGPT 编写 **tinygrad 测试**的链接：[link](https://chatgpt.com/share/66e693ef-1a50-8000-81ff-899498f9d052)。
- **iText2KG 开发讨论**：一位成员分享了与 **iText2KG** 开发者的对话，内容涉及添加用于边缘提取的 **SeekTopic 算法**，表明开发工作正在推进。
   - 另一位成员确认了对该方法的兴趣，认为这是该研究中一个非常有前景的方向。
- **LLM 增强研究生成**：研究强调，根据在 [arXiv](https://arxiv.org/html/2409.04109) 上发现的论文，**LLM** 可以生成更好的研究想法和计划。
   - 这一见解支持了人们日益增长的认知，即 LLM 在贡献研究和创新方面具有强大能力。



**提到的链接**：<a href="https://x.com/realGeorgeHotz/status/1835228364837470398">来自 George Hotz 🌑 (@realGeorgeHotz) 的推文</a>：ChatGPT o1-preview 是第一个（完全）具备编程能力的模型。看到一个 120 IQ 的估值，感觉差不多。非常看好开发环境中的 RL。写代码，写测试……

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1285326215414026330)** (3 messages): 

> - `扩展 LLM 推理极限`
> - `研究中的分块阶段`
> - `Transformer 能力` 


- **探索 LLM 推理性能极限**：[Denny Zhou 最近的一条推文](https://x.com/denny_zhou/status/1835761801453306089?s=46&t=VBhI-dqaQfawcUDHNO0L9A)声称，只要给予足够的中间推理 Token，**Transformer** 可以解决任何问题，并断言恒定深度（constant depth）就足够了。
   - 这一数学证明出现在一篇题为《What is the performance limit when scaling LLM inference?》的论文中，该论文已被 ICLR 2024 接收，可以在[这里](http://arxiv.org/abs/2402.12875)查看。
- **征集关于分块阶段（Chunking Phases）的研究**：一位用户询问了近期专注于**分块阶段**和近似计算的研究论文，寻求顶尖且最新的研究成果。
   - 他们特别询问了是否有任何相关工作可以推进该领域的理解。



**提到的链接**：<a href="https://x.com/denny_zhou/status/1835761801453306089?s=46&t=VBhI-dqaQfawcUDHNO0L9A">来自 Denny Zhou (@denny_zhou) 的推文</a>：扩展 LLM 推理的性能极限在哪里？天空才是极限。我们已经从数学上证明，只要允许 Transformer 生成尽可能多的中间结果，它们就可以解决任何问题……

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1285321251430207568)** (77 messages🔥🔥): 

> - `Dream Machine API`
> - `11x AI A 轮融资`
> - `AI 对就业的影响`
> - `Claude 3.5 系统提示词`
> - `基于 ZIG 的推理栈`

- **Dream Machine API 发布**：Luma Labs 宣布推出 [Dream Machine API](https://lumalabs.ai/dream-machine/api)，允许开发者在无需复杂工具的情况下，利用领先的视频生成模型进行创作。
   - *即刻开始* 探索简化的视频创作功能。
- **11x AI 获得 2400 万美元 A 轮融资**：由 Alice 和 Jordan 共同创立的 11x AI 已从 **Benchmark** 等机构筹集了 **2400 万美元 A 轮融资**，其今年 ARR 增长了 **15 倍**，并为超过 **250 家客户** 提供支持。
   - 该团队计划为数字员工开发 **LLM 驱动的系统**，旨在重新定义现代 GTM 职能。
- **AI 对就业日益增长的影响**：一份报告估计，明年美国和墨西哥将有 **6000 万个工作岗位** 受到 AI 的影响，未来十年的预测显示，美国受影响岗位将增至 **7000 万个**，墨西哥将增至 **2600 万个**。
   - 虽然并非所有的职业变动都会导致失业，但仍有大量职业处于弱势地位，凸显了适应变化的紧迫性。
- **Claude 3.5 系统提示词分享**：一位用户在 gist 中分享了 **Claude 3.5 Projects + Artifacts 系统提示词**，这对探索 AI 应用的人士具有参考价值。
   - 该提示词目前已在多个 Discord 频道中引起讨论，反映了其在当前 AI 评估中的重要性。
- **由 Yann LeCun 支持的基于 ZIG 的推理栈**：一个由 **Yann LeCun** 支持的全新 **基于 ZIG 的推理栈** 已公开，旨在提供高性能的 AI 推理，能够在各种硬件上运行深度学习系统。
   - 该项目已开源并结束隐身模式，展示了 AI 性能能力的进步。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/11x_official/status/1835711787712582082?s=46">来自 11x (@11x_official) 的推文</a>：👋🏻 来自 Alice 和 Jordan 的问候 - 我们刚刚从 @benchmark 筹集了 2400 万美元的 A 轮融资！在此阅读我们的完整博客文章：https://www.11x.ai/blog/series-a 今年到目前为止的一些亮点：- ARR 增长了...</li><li><a href="https://www.amazon.com/Decline-American-Programmer-Edward-Yourdon/dp/013191958X">未找到标题</a>：未找到描述</li><li><a href="https://www.amazon.com/Resurrection-American-Programmer-Yourdon-Computing/dp/013121831X/ref=">未找到标题</a>：未找到描述</li><li><a href="https://www.amazon.com/Resurrection-American-Programmer-Yourdon-Computing/dp/013121831X/ref=sw_img_mw_ace_sim?_encoding=UTF8&pd_rd_i=013121831X&pd_rd_w=hIDWd&content-id=amzn1.sym.5a573fc7-d6aa-4c57-a9db-8a2220925981&pf_rd_p=5a573fc7-d6aa-4c57-a9db-8a2220925981&pf_rd_r=T08GE2EA6ZR6NJNC71CV&pd_rd_wg=jhWGS&pd_rd_r=c962c739-02de-4045-afb9-cdd0bacb017f">未找到标题</a>：未找到描述</li><li><a href="https://x.com/lmsysorg/status/1835825082280902829">来自 lmsys.org (@lmsysorg) 的推文</a>：Chatbot Arena 更新🔥 我们在过去 2 周内一直在测试最新的 ChatGPT-4o (20240903)，结果显示各方面都有显著提升：- 总分：1316 -> 1336 - ...</li><li><a href="https://x.com/maartengr/status/1835709176703508688?s=46">来自 Maarten Grootendorst (@MaartenGr) 的推文</a>：我很高兴地宣布《Hands-On Large Language Models》数字版发布了 🎉 本书包含 250 多个视觉图表（彩色！），帮助你理解内部原理...</li><li><a href="https://www.arcads.ai/">Arcads - 创建 AI 视频广告</a>：AI UGC 变得简单：编写脚本，挑选演员，在 2 分钟内生成你的 UGC 视频。</li><li><a href="https://x.com/sullyomarr/status/1836059834543734892?s=46">来自 Sully (@SullyOmarr) 的推文</a>：终于兴奋地发布了 Otto！！Otto 让你在表格中使用 AI Agent，在几分钟内自动化数小时的手动调研。这是我关于如何使用它的快速拆解，包含一些实际用例...</li><li><a href="https://x.com/lumalabsai/status/1835742651662139529?s=46">来自 Luma AI (@LumaLabsAI) 的推文</a>：🚀 推出 Dream Machine API。开发者现在可以使用全球最受欢迎且直观的视频生成模型来构建和扩展创意产品，而无需在自己的环境中构建复杂的工具...</li><li><a href="https://x.com/zml_ai/status/1835973073385685099?s">来自 ZML (@zml_ai) 的推文</a>：https://github.com/zml/zml</li><li><a href="https://x.com/zml_ai/status/1835973073385685099?s=46">来自 ZML (@zml_ai) 的推文</a>：https://github.com/zml/zml</li><li><a href="https://x.com/ylecun/status/1836030233796874244?s=46">来自 Yann LeCun (@ylecun) 的推文</a>：ZML：一个高性能 AI 推理栈，可以在许多不同的硬件上并行化并运行深度学习系统。它已结束隐身模式，令人印象深刻，并且是开源的。引用 ZML (@zml_a...</li><li><a href="https://english.elpais.com/economy-and-business/2024-09-15/artificial-intelligence-will-affect-60-million-us-and-mexican-jobs-within-the-year.html#">人工智能将在一年内影响 6000 万个美国和墨西哥的工作岗位</a>：IDB 的研究显示了 AI 对劳动力市场的影响。女性和低技能工人更容易被取代</li><li><a href="https://www.patreon.com/posts/super-panavision-109117838?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_fan&utm_content=web_share">Super Panavision 70 教程 | Abandoned Films</a>：在 Patreon 上获取更多来自 Abandoned Films 的内容</li><li><a href="https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d">提取的 Claude 3.5 Sonnet Artifacts 系统提示词</a>：提取的 Claude 3.5 Sonnet Artifacts 系统提示词 - claude_35_artifacts_system_prompt.txt</li><li><a href="https://www.when2meet.com/?26526365-sTYbJ">Evals Group - When2meet</a>：未找到描述
</li>
</ul>

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1285358690504802314)** (10 messages🔥): 

> - `Foundation Models in Biotech`
> - `AI Safety Transition`
> - `TensorRT-LLM Issues`
> - `Transformer Model Memory Profiling`
> - `Photo Upgrade Tools` 


- **Biotech 领域的基础模型分享**：一位成员介绍了他们在 **biotech** 领域 **foundation models** 的工作，重点关注序列和表格数据的 **large scale representation learning**。
   - 这一专业领域凸显了 AI 技术与生物技术应用之间日益增长的交集。
- **AI Safety 职业转型**：一位成员宣布，在从事了 9 年应用 AI/ML 模型开发后，他们获得了 **Open Philanthropy career transition fellowship**，将研究重点转向 **AI safety**。
   - 他们最初的兴趣在于 **interpretability** 以及与 AI 相关的更广泛的 **alignment** 讨论。
- **TensorRT-LLM 的构建问题**：一位用户询问了在与 **CUDA** 兼容的 **T4 显卡**上使用 **TensorRT-LLM** 构建模型时遇到的困难。
   - 这引起了那些在模型开发工作流中使用特定硬件配置的人员的关注。
- **调试 Transformer 模型内存**：一位成员正在调查训练小型 transformers 时的内存需求，注意到尽管他们的计算显示只需要 **16** 字节，但实际至少需要每个参数 **26 bytes per param**。
   - 他们正在探索 **pytorch profiling tools**，但由于 **FSDP** 和 **mixed precision** 层使分析复杂化，发现堆栈跟踪（stack traces）不够透明。
- **照片增强工具建议**：针对“升级”拍摄不佳的肖像工具的需求，一位成员建议使用带有 **stable diffusion plugin** 的 **Krita**。
   - 该咨询强调了对在保留相似度的同时增强摄影效果的开源解决方案的需求。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1285388437607682119)** (8 messages🔥): 

> - `AI Safety Fellowship`
> - `Token Embedding Variability`
> - `Multi-head Low-Rank Attention`
> - `Diagram of Thought`
> - `Hyper-graphs` 


- **AI Safety 探索与研究帮助**：一位成员分享了在获得 Open Philanthropy 奖学金后转向 **AI Safety** 的兴奋之情，并对 **interpretability** 和 **alignment** 研究表示了兴趣。
   - 他们鼓励大家分享正在进行的研究项目，他们可以利用自己丰富的经验提供志愿帮助，目标是在未来 **六个月** 内做出贡献。
- **Token Embedding Variability 作为稳定性代理**：最近的一篇论文介绍了 **Token Embedding Variability (TEV)** 作为评估语言模型预训练稳定性的有效方法，同时提出了 **Multi-head Low-Rank Attention (MLRA)** 以防止梯度爆炸。
   - 在 **GPT-2** 上的实验结果显示，稳定性得到了提高，困惑度（perplexity）更低，特别是在更深的网络中，尽管一些人对实验设置的细节提出了疑问。
- **用于迭代推理的 Diagram of Thought**：介绍了 **Diagram of Thought (DoT)** 框架，该框架将 **large language models** 中的迭代推理建模为单个模型内的有向无环图 (DAG)，通过复杂路径增强逻辑一致性。
   - 该模型通过利用 **natural language feedback** 和自回归 token 预测，可以实现更好的推理迭代改进。
- **在推理模型中调用超图 (Hyper-graphs)**：一位参与者建议在推理模型中使用 **hyper-graphs**，暗示这可能对近期文献中提出的 search-in-the-chain 方法有所增强。
   - 尽管对影响存在一些疑问，但他们承认在这种背景下探索 **hyper-graph** 结构的有趣性质。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.07146">Gated Slot Attention for Efficient Linear-Time Sequence Modeling</a>: 线性注意力 Transformers 及其门控变体因能够实现并行训练和高效的递归推理而受到推崇，但在召回密集型任务中与传统的相比仍有不足...</li><li><a href="https://arxiv.org/abs/2409.07787">Stable Language Model Pre-training by Reducing Embedding Variability</a>: 稳定的预训练对于实现性能更好的语言模型至关重要。然而，由于符号原因，通过在每一步计算梯度方差来跟踪预训练稳定性是不切实际的...</li><li><a href="https://arxiv.org/abs/2409.10038">On the Diagram of Thought</a>: 我们介绍了 Diagram of Thought (DoT)，这是一个在单个模型中将大型语言模型 (LLMs) 中的迭代推理建模为有向无环图 (DAG) 构建的框架。与...不同
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1285703238879805513)** (9 条消息🔥): 

> - `Hidden States 的 Fourier Transforms`
> - `Hidden States 中的 Power Law`
> - `Pythia Checkpoints 探索`
> - `未训练模型的行为`
> - `Attention Residuals 分析` 


- **探索 Hidden States 的 Fourier Transforms**：发起了关于如何像 Sander Dieleman 在其博客文章中所做的那样，解释 Hidden States 的 [Fourier transforms](https://sander.ai/2024/09/02/spectral-autoregression.html) 的讨论。主要发现表明，Hidden States 在初始阶段呈均匀分布，并随着层深度的增加演变为 Power Law。
   - 提出了关于 Attention 机制是否在最终的 Hidden States 中诱导了这种功率谱 Power Law 的疑问。
- **推荐使用 Pythia Checkpoints**：一位成员建议使用 [Pythia suite](https://github.com/EleutherAI/pythia) 来研究规模和训练对观察到的现象的影响。这种模型行为可能受到架构或初始化因素的影响。
   - 建议探索具有不同架构的不同模型以确认这些观察结果。
- **关于 Hidden States 和 Attention Residuals 的澄清**：澄清了 Hidden States 图表源自预训练的 OPT-125m，而 Attention Residuals 则来自未训练的模型。这表明基于训练状态的不同光谱特性。
   - 有人指出 Hidden States 的特性在训练过程中不断演变，这表明训练有可能使这些结果产生偏差。
- **Attention Residuals 和 MLP 分析**：预训练模型的 Attention Residuals 显示出与层结构一致的显著峰值，表明了光谱行为。对 MLP Residuals 的分析补充了这一理解，并对早期层行为进行了重要观察。
   - 新初始化的模型与预训练模型之间的对比行为，引发了关于训练和表示效率的进一步问题。
- **下一步重点是 Pythia 的利用**：对话以达成探索 Pythia 以更深入了解模型训练效果的共识结束。这与量化模型训练过程中变化的目标一致。



**提到的链接**：<a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)">interpreting GPT: the logit lens — LessWrong</a>：这篇文章涉及我在使用 GPT-2 时的一个观察，我还没有在其他地方看到过。…

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1285523748090286090)** (37 条消息🔥): 

> - `LM Evaluation Harness 相关问题`
> - `Torchtune 的集成`
> - `TensorRT-LLM 构建错误`
> - `在 Hugging Face 上部署数据集`
> - `Chain of Thought 提示词` 


- **LM Evaluation Harness 问题求助**：一位用户就 LM Evaluation Harness 的一个 [issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/2305) 寻求帮助。
   - 另一位用户指出，他们可以通过将数据集添加到 HF 来提供帮助。
- **Torchtune 与 LM Evaluation Harness 集成**：Torchtune 的一位维护者提到了他们与 eval harness 的集成，并建议针对使用静态 KV-caching 的生成任务进行增强。
   - 他们建议寻找最大的可整除 batch size，以避免缓存设置出现错误。
- **TensorRT-LLM 构建故障**：一位用户提出了构建 TRT-LLM 引擎时遇到的问题，引用了与 workspace 大小相关的错误。
   - 建议包括使用 IBuilderConfig::setMemoryPoolLimit() 来增加 workspace 大小。
- **在 Hugging Face 上部署数据集**：一位用户确认他们已将数据集添加到 [HF](https://huggingface.co/datasets/baber/multilingual_mmlu) 以协助评估。
   - 为了优化 splits 和 subsets，用户请求对数据集结构进行进一步说明。
- **Chain of Thought 提示词咨询**：一位用户询问了使用 LM Evaluation Harness 进行 Chain of Thought (思维链) 提示的经验。
   - 他们对将模型回答附加到后续提示词中并记录结果感兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/2305)">Issues · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - Issues · EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/pytorch/torchtune/blob/60a7e3dae0d43e841c5e0ee4c1622fc9b1c2c4ca/recipes/eleuther_eval.py#L37">torchtune/recipes/eleuther_eval.py at 60a7e3dae0d43e841c5e0ee4c1622fc9b1c2c4ca · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。在 GitHub 上通过创建账号为 pytorch/torchtune 开发做贡献。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/fb963f0f0a5b28b69763590bb59676072cf43a01/lm_eval/models/huggingface.py#L1236)">lm-evaluation-harness/lm_eval/models/huggingface.py at fb963f0f0a5b28b69763590bb59676072cf43a01 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/fb963f0f0a5b28b69763590bb59676072cf43a01/lm_eval/models/huggingface.py#L1295)">lm-evaluation-harness/lm_eval/models/huggingface.py at fb963f0f0a5b28b69763590bb59676072cf43a01 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/fb963f0f0a5b28b69763590bb59676072cf43a01/lm_eval/evaluator.py#L439)">lm-evaluation-harness/lm_eval/evaluator.py at fb963f0f0a5b28b69763590bb59676072cf43a01 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/fb963f0f0a5b28b69763590bb59676072cf43a01/lm_eval/evaluator_utils.py#L203)">lm-evaluation-harness/lm_eval/evaluator_utils.py at fb963f0f0a5b28b69763590bb59676072cf43a01 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1285680433249390594)** (1 条消息): 

> - `模型输出`
> - `库的利用` 


- **社区对模型输出的热情**：成员们对展示他们的模型以及使用这些库创建的输出表现出极大的热情，这表明了支持性的社区氛围。
   - *一定要让我们知道*你什么时候有了模型或其他输出，因为社区非常喜欢听到人们利用这些资源所取得的伟大成就。
- **库的参与度**：有强烈迹象表明，社区成员非常看重与其库实现相关的更新，强调了协作与分享的重要性。
   - *我们很乐意听到人们使用我们的库所做的伟大事情*，这突显了社区贡献的重要性。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1285317374853251083)** (53 messages🔥): 

> - `SSH 连接问题`
> - `Stable Diffusion 安装错误`
> - `ComfyUI 白屏`
> - `Control Net 训练挑战`
> - `CivitAI 悬赏报价` 


- **已部署 Pods 的 SSH 连接问题**：一名成员报告在更新 SSH 密钥后，连接到已部署的 Pods 时遇到困难。
   - 他们询问是否有任何配置更改可以允许他们通过 SSH 进行连接。
- **Stable Diffusion 安装故障**：尽管按照指示运行了设置脚本，一名成员仍遇到了“Stable Diffusion 模型加载失败”的错误。
   - 其他成员建议参考安装指南，并发布详细的错误日志以获取技术支持。
- **更新后 ComfyUI 出现白屏**：一名用户更新了 ComfyUI，在尝试加载 GUI 时遇到了白屏问题。
   - 一位成员建议在重启系统之前，完全卸载 ComfyUI 并再次运行更新脚本。
- **Control Net 训练中的挑战**：围绕使用 Control Net 进行有效训练是否需要大量数据集展开了讨论。
   - 成员们建议考虑新颖的数据集增强方法和工作流，以实现预期的结果。
- **CivitAI 悬赏包咨询**：一名成员正寻求发布一个 CivitAI 悬赏，旨在创建一个包含 49 个角色、图片数量约为 4000 张的资源包。
   - 他们正在就该悬赏应提供的适当 Buzz 数量寻求建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Gui">Home</a>: Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui 安装指南</a>: Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://github.com/fairy-root/Flux-Prompt-Generator?tab=readme-ov-file">GitHub - fairy-root/Flux-Prompt-Generator: Flux Prompt Generator 提供了一个灵活且可定制的提示词生成器，用于为图像生成模型生成详细且富有创意的提示词。</a>: Flux Prompt Generator 提供了一个灵活且可定制的提示词生成器，用于为图像生成模型生成详细且富有创意的提示词。- fairy-root/Flux-Prompt-Generator
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1285373740410081362)** (2 messages): 

> - `Multimodal RAG 技术`
> - `LlamaCloud 发布`
> - `产品手册挑战` 


- **利用 Multimodal RAG 处理产品手册**：由于产品手册具有以视觉为中心的特性，缺乏文本且主要包含**分步视觉效果**和图表，因此通常对 RAG 构成挑战。
   - 为了使 LLM 能够有效地浏览这些手册，需要一个复杂的 [indexing pipeline](https://t.co/GOedcAdLqF)。
- **LlamaCloud 发布多模态功能**：今天发布的 **LlamaCloud 多模态功能**允许用户跨各种非结构化数据格式快速创建端到端的 Multimodal RAG 流水线。
   - 该工具包支持多种应用，包括**营销幻灯片**、法律和保险合同以及财务报告——简化了用户的工作流程（[详情点击此处](https://t.co/43eL8zvm7H)）。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1285429789653733446)** (36 messages🔥): 

> - `LlamaIndex 与 Neo4j 集成`
> - `从 Neo4j 检索 Embedding`
> - `LlamaIndex 包中的循环依赖`
> - `GraphRAG 实现`
> - `使用 GPT-4o 提取图像坐标` 


- **LlamaIndex 与 Neo4j 的 Embedding 集成**：一位用户询问如何使用 LlamaIndex 检索存储在 Neo4j 节点属性中的 Embedding。其他人建议将 Neo4j 与 LlamaIndex 的 Property Graph Index 连接，以实现高效查询。
   - 提到一旦检索到节点，解析其属性以获取 Embedding 应该是很直接的。
- **LlamaIndex 包中检测到循环依赖**：在更新 Bazel Dependency Graph 期间，发现 `llama-index-agent-openai` 和 `llama-index-llms-openai` 包之间存在循环依赖。成员们讨论了潜在的解决方案，包括创建一个 `openai-utils` 包来解决此问题。
   - 提出了关于解决此依赖关系的时间表问题，并建议社区贡献以加快修复。
- **评估向 GraphRAG 的转型**：一位成员表示有兴趣从基础 RAG 转向 GraphRAG，并就使用 LlamaIndex 的抽象还是 Microsoft 的包寻求建议。还请求了关于构建评估集以比较不同方法的建议。
   - 几位成员表示有兴趣分享关于这种转型的最佳实践见解。
- **使用 GPT-4o 提取图像坐标的挑战**：一位用户描述了在使用 GPT-4o 时对齐标签和从图像中提取准确坐标的困难。他们请求关于改进其网格叠加（grid overlay）方法的建议，以确保准确的空间识别。
   - 目标是根据检测到的实体生成用于裁剪图像的精确坐标，欢迎社区提供反馈。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/Neo4jKGIndexDemo/">Neo4j Graph Store - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_neo4j/">Neo4j Property Graph Index - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/#texttocypherretriever">Property Graph Index - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/7e557890adac0f505be29d06a1eff60fc7dc629b/llama-index-integrations/agent/llama-index-agent-openai/pyproject.toml#L35)">llama_index/llama-index-integrations/agent/llama-index-agent-openai/pyproject.toml at 7e557890adac0f505be29d06a1eff60fc7dc629b · run-llama/llama_index</a>: LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/7e557890adac0f505be29d06a1eff60fc7dc629b/llama-index-integrations/llms/llama-index-llms-openai/pyproject.toml#L38)">llama_index/llama-index-integrations/llms/llama-index-llms-openai/pyproject.toml at 7e557890adac0f505be29d06a1eff60fc7dc629b · run-llama/llama_index</a>: LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1285644658353111070)** (4 messages): 

> - `Mistral 九月发布`
> - `La Plateforme 上的免费层级`
> - `价格更新`
> - `Mistral Small 改进`
> - `Pixtral 12B 的视觉能力` 


- **Mistral 发布新功能**：Mistral 发布了多项功能，包括在 [La Plateforme 上的免费层级](https://mistral.ai/news/september-24-release/)，供开发者免费实验 API 端点。
   - 新更新还带来了全面的**降价**以及对 **Mistral Small** 的改进。
- **Mistral 的免费层级引发反应**：针对新宣布的免费层级，一位成员的反应是 *“免费？呵呵 (lol)”*，表示对其影响持怀疑态度。
   - 这种情绪反映了社区中关于 **VC 渴望** 更好用户参与度的可行性的更广泛讨论。
- **用户数据请求以获取见解**：另一位成员敦促道，*“伙计们，也请给我们用户数据，”* 指出社区渴望从用户参与度方面获得更多见解和细节。
   - 这突显了在评估 Mistral 新产品过程中，对理解用户行为的持续兴趣。



**提到的链接**: <a href="https://mistral.ai/news/september-24-release/">AI in abundance</a>: 推出免费 API，全面优化价格，全新的企业级 Mistral Small，以及 le Chat 上的免费视觉能力。

  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1285645734540218480)** (12 条消息🔥): 

> - `Transformer 中的中间生成 (Intermediate Generation)`
> - `注意力矩阵可视化 (Visualizing Attention Matrices)`
> - `Alpha Code 网站功能`
> - `Attention Rollout 论文`
> - `基于梯度的 Token 关联` 


- **中间生成增强 Transformer 能力**：研究表明，允许 Transformer 使用“思维链 (Chain of Thought)”或“草稿本 (Scratchpad)”可以显著增强其计算能力，这取决于中间生成的数量。
   - 这一进展展示了解决标准 Transformer 难以应对的推理问题的潜力，并对有效利用中间步骤具有启发意义。
- **注意力矩阵可视化的最佳实践**：一位成员询问了在 QA 场景中可视化注意力矩阵的最佳实践，以展示问题与支持事实之间的关联。
   - 建议包括探索各种可视化表示连接强度的技术，这可以阐明答案是如何得出的。
- **Alpha Code 的交互式透明度功能**：讨论强调了 Alpha Code 网站的一个功能，即悬停在 Token 上会显示 Prompt 中被关注最多的 Token，并根据关联强度进行颜色编码。
   - 这种交互方式可以增强用户对生成响应中注意力关系的理解。
- **将 Attention Rollout 作为“最受关注”的参考**：有人建议参考 Attention Rollout 论文中关于“最受关注 (most attended)”的各种定义，以深入了解其实现方式。
   - 这可以为在 Transformer 模型中有效衡量注意力提供坚实的基础实践。
- **输出 Token 关联的梯度分析**：讨论了探索输出 Token 概率相对于输入 Token 的梯度，作为理解 Token 关联的另一种方法，尽管实现起来较为复杂。
   - 该方法可能计算量巨大，需要编写自定义代码来处理其 O(n^2) 的时间复杂度。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://poloclub.github.io/transformer-explainer/">Transformer Explainer: LLM Transformer Model Visually Explained</a>: 一个交互式可视化工具，向你展示 Transformer 模型在 GPT 等大语言模型 (LLM) 中是如何工作的。</li><li><a href="https://arxiv.org/abs/2402.12875">Chain of Thought Empowers Transformers to Solve Inherently Serial Problems</a>: 指导模型生成一系列中间步骤，即思维链 (CoT)，是提高大语言模型 (LLM) 在算术上准确率的一种非常有效的方法……</li><li><a href="https://arxiv.org/abs/2310.07923">The Expressive Power of Transformers with Chain of Thought</a>: 最近的理论工作发现了一些极其简单的推理问题，例如检查图中的两个节点是否连接或模拟有限状态机，这些问题被证明是无法解决的……
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1285448997787402270)** (15 messages🔥): 

> - `Gemini Models`
> - `NotebookLM Tweet`
> - `Podcast with Riley`
> - `Guest Lecture on LLMs` 


- **大量未发布的 Gemini 模型**: 关于未发布的 **Gemini models** 的消息流出，包括在 vision arena 中的 **potter-v1** 和 **dumbledore-v1**，以及 **gemini-test**（**Gemini 1.5 pro refresh**）。
   - 其他提到的模型包括 **zeus-flare-thunder** (v1 和 v2) 以及 **qwen2.5-72b-instruct**，预示着即将有强力的阵容发布。
- **NotebookLM 引起关注**: 一条关于 **NotebookLM** 的随机推文意外获得了关注，导致两位来自 **Google** 的人员发来了私信。
   - 这种兴奋感是显而易见的，因为这次互动为讨论开启了新的机会，并可能建立更多的联系。
- **与 Riley 的播客带来有趣的见解**: 与 **Riley** 的播客环节被描述为非常愉快，他的迷人个性和分享的见解受到了赞扬。
   - 鼓励听众加入讨论，并强调与 **Twitter** 等其他平台相比，这里的社区氛围要愉快得多。
- **在 McGill 举办首场关于 LLMs 的客座讲座**: 一位成员庆祝了他们在 **McGill** 作为兼职教授举办的首场关于 **LLMs** 的客座讲座，并分享了他们的幻灯片作为宝贵资源。
   - 他们旨在通过这些材料帮助他人，展示了在社区中分享知识的积极态度。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/phill__1/status/1835989093093617739?s=46">Tweet from Phil (@phill__1)</a>: lmarena 目前非常疯狂。未发布的模型：- potter-v1 和 -v2 以及 dumbledore-v1 和 -v2 (仅在 vision arena) - zeus-flare-thunder-v1 和 -v2 - sharp-game-player-v1 和 -v2 - qwen2.5-72b...</li><li><a href="https://x.com/agarwl_/status/1836119825216602548?s=46">Tweet from Rishabh Agarwal (@agarwl_)</a>: 我今天在 McGill 的一个 LLMs 研究生课程中作为（即将上任的）兼职教授进行了我的第一次客座讲座。把幻灯片放在这里，也许对某些人有用 ;) https://drive.google.com/file/d/1komQ7s9...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1285396690194403399)** (2 messages): 

> - `AI developers skipping Google's Gemini`
> - `Humorous AI article` 


- **AI 开发者对 Google 的 Gemini 嗤之以鼻**: 一位成员指出了一篇发表在 [The Information](https://www.theinformation.com/articles/why-ai-developers-are-skipping-googles-gemini) 上的文章，讨论了为什么 **AI 开发者正在跳过 Google 的 Gemini**。
   - *这篇文章的配图让我忍俊不禁。*
- **文章引发成员幽默感**: 同一篇文章因其内容的**幽默性**引起了成员们的笑声。
   - 有人特别评论了文章的配图，称其*非常滑稽*。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1285388361686716457)** (3 messages): 

> - `Newsletter Reader Party`
> - `Mainstream Media Critique` 


- **在 Newsletter 读者派对上庆祝**: 一位成员宣布了“伟大的 Newsletter 读者派对”，邀请大家加入并享受共同阅读的乐趣。
   - 该活动旨在促进 Newsletter 爱好者之间的社区建设和互动。
- **对主流媒体消费的批评**: 讨论中出现了一个关于仅依赖主流媒体获取新闻的弊端的观点。
   - 所表达的情绪突显了人们对替代来源和更多元化内容日益增长的渴望。


  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1285340237228150845)** (15 messages🔥): 

> - `Chat Message History Management` (聊天消息历史管理)
> - `UI Messages Storage` (UI 消息存储)
> - `Open Source Aspirations` (开源愿景)
> - `Migrating to LLMChain` (迁移到 LLMChain)
> - `Implementing AI in Business` (在业务中实施 AI)


- **澄清聊天消息历史管理**：一位成员寻求关于在 **LangChain** 中管理聊天历史的澄清，并指出在存储消息历史的同时存储额外 UI 数据的复杂性，目前 **PostgresChatMessageHistory** 是将这两者分开处理的。
   - 其他人确认需要将 UI 特定的消息存储在不同的表中，因为现有系统不支持组合事务。
- **发展开源贡献**：一位成员表达了成为开源领域重要贡献者的抱负，希望在保持独立性（通过赞助）的同时参与有影响力的项目。
   - 他们向社区寻求关于实现这一宏伟目标的路径见解。
- **迁移到新的 LLMChain 实现**：一位成员建议从旧版的 **LLMChain** 转换到较新的实现方式，因为后者具有更清晰的参数和增强的 Streaming 能力。
   - 文中强调了几个优势，包括更容易获取原始消息输出，突显了保持版本更新的好处。
- **在本地业务中集成 AI 解决方案**：一位成员表达了向本地商家销售 AI 解决方案的信心，并寻求关于成功实施策略和繁荣市场的建议。
   - 他们特别感兴趣于如何与对 AI 技术了解有限的业务负责人进行沟通的技巧。
- **加速结构化响应**：一位成员询问了在 **LangChain** 中加快结构化响应的方法，特别是结合使用 **Pydantic** 模型和 **JSONOutputParser**。
   - 这一询问反映了开发者在提高响应效率方面面临的持续挑战。



**相关链接**：<a href="https://python.langchain.com/docs/versions/migrating_chains/llm_chain/">Migrating from LLMChain | 🦜️🔗 LangChain</a>：LLMChain 将提示模板、LLM 和输出解析器组合成一个类。

  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/)** (1 messages): 

taixian0420: please dm me (请私信我)
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1285493431992324168)** (6 messages): 

> - `RAG Chatbot` (RAG 聊天机器人)
> - `AdaletGPT` 


- **AdaletGPT 发布 RAG 聊天机器人**：**adaletgpt.com** 的一名后端开发人员使用 OpenAI 和 LangChain 成功构建了一个 **RAG 聊天机器人**。
   - 您可以在其网站上查看：[adaletgpt.com](https://adaletgpt.com)。
- **公开提问邀请**：开发人员鼓励社区成员就聊天机器人相关的任何其他问题直接通过 DM（私信）联系。
   - 他们承诺会尽力提供支持。



**相关链接**：<a href="https://adaletgpt.com">no title found</a>: no description found

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1285493749467713566)** (1 messages): 

> - `RAG Chatbot` (RAG 聊天机器人)
> - `OpenAI Integration` (OpenAI 集成)
> - `LangChain Framework` (LangChain 框架) 


- **后端开发人员发布 RAG 聊天机器人**：[adaletgpt.com](https://adaletgpt.com/) 的一名后端开发人员分享了他们使用 **OpenAI** 和 **LangChain** 构建的新 RAG 聊天机器人。
   - 他们鼓励其他人就任何问题通过 DM 联系，并表示将尽力提供帮助。
- **提问邀请**：开发人员邀请社区就 RAG 聊天机器人进行进一步咨询。
   - 他们表达了支持用户的意愿，并表示：“我会为你尽我所能。”



**相关链接**：<a href="https://adaletgpt.com/">no title found</a>: no description found

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1285338481559601242)** (10 messages🔥): 

> - `tinygrad version bump`
> - `ROCm compatibility`
> - `CRIU feature in AMDKFD`
> - `pytest filtering`
> - `testing unnecessary files` 


- **Tinygrad 升级到 0.9.2 在 AMD 上失败**：一位用户在尝试将 **nixpkgs** 中的 **tinygrad** 从 **0.9.0 升级到 0.9.2** 时，遇到了与 `struct_kfd_ioctl_criu_args` 相关的 **AttributeError**。
   - 他们质疑这是否由错误的内核版本引起，因为该结构体存在于 **/usr/include/linux/kfd_ioctl.h** 中。
- **CRIU 支持是最近新增的**：社区成员指出，**amdgpu driver** 中的 **CRIU** 支持是一个相对较新的特性。
   - 这可能表明在升级 **tinygrad** 或相关库时存在兼容性担忧。
- **extra/ 目录中被误识别的测试**：用户指出失败源于 **extra/hip_gpu_driver/test_kfd_2.py**，这些不应该在实际测试中运行。
   - 还提到了 **extra/hip_gpu_driver/test_sdma_fun.py** 中的另一个失败，建议它也不应被视为有效的测试。
- **过滤掉无关的测试**：团队同意在运行 **pytest** 时过滤掉 **extra/** 文件夹，从而避免不必要的失败。
   - 这将确保只有 **test/** 目录中相关的测试被执行。
- **承认仓库杂物**：**George Hotz** 确认 **extra/** 文件夹中的文件被视为“仓库杂物 (repo junk)”，不应包含在测试中。
   - 为了获得准确的结果，应该只执行位于 **test/** 目录中的相关测试。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/518c022c29104d79c7a50ec41af5b7e6404da317/extra/hip_gpu_driver/test_kfd_2.py#L31)">tinygrad/extra/hip_gpu_driver/test_kfd_2.py at 518c022c29104d79c7a50ec41af5b7e6404da317 · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/pull/5917">wozeparrot 提交的 hip_ioctl 更改 · Pull Request #5917 · tinygrad/tinygrad</a>: feat: 允许将处理器指定为环境变量 feat: vendor kfd_ioctl.h
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1285322186370060482)** (8 messages🔥): 

> - `VRAM allocation spikes`
> - `Tinygrad Tensor error`
> - `Diffusers fork with Tinygrad`
> - `NotebookLM podcast`
> - `Fundamental operations in Tinygrad` 


- **调试 VRAM 峰值**：一位用户询问了识别 **VRAM 分配峰值** 原因的最佳方法。
   - 讨论暗示正在寻求潜在的工具或策略来有效地监控 **memory usage**。
- **Tinygrad Tensor 错误调查**：一位成员报告在运行涉及 Tensor 操作的 **Tinygrad** 代码片段时遇到错误。
   - 他们链接到了一个[公开的 issue](https://github.com/tinygrad/tinygrad/issues/6352)，可能为该问题提供背景。
- **Fork Diffusers 以使用 Tinygrad**：一位成员分享了关于协作开发 **Diffusers fork** 的更新，该分支集成了 Tinygrad 而不是 Torch。
   - 他们希望在不紧密复制现有实现的情况下采用一种新方法。
- **NotebookLM 将 Tinygrad 转化为播客**：一位成员分享说 **NotebookLM** 创建了一个 **8 分钟的播客**，使用引人入胜的比喻解释了 Tinygrad。
   - 他们强调该播客实际上起到了 **tinybox 推销** 的作用。
- **理解 Tinygrad 的简约方法**：一位用户通过说明“他们只使用 3 种基本操作类型”来展示 Tinygrad 的效率。
   - 他们将其比作一位只使用三把刀的大厨，强调了强大工具背后的简单性。



**提到的链接**: <a href="https://github.com/tinygrad/tinygrad/issues/6352)">Issues · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - Issues · tinygrad/tinygrad

  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1285331547032911883)** (10 messages🔥): 

> - `Cohere Chat API Safety Modes` (Cohere Chat API 安全模式)
> - `Cohere's market strategy` (Cohere 的市场策略)
> - `Training language models` (训练语言模型)
> - `Applying to Cohere` (申请 Cohere 职位)


- **Cohere 在 Chat API 中推出测试版安全模式 (Safety Modes)**：Cohere 宣布在 Cohere Chat API 中测试版发布 **Safety Modes**，这是一项新功能，允许客户自定义模型输出以满足安全要求。
   - *这可能允许用户实施安全检查并减轻责任风险*。
- **Cohere 专注的市场策略**：**Cohere** 通过专注于特定的用例，在拥挤的 LLM 市场中进行战略性导航，避开了过度饱和的领域。
   - 成员们讨论了**务实的商业选择**的价值，这些选择强调了模型应用中的清晰度和实用性。
- **安全仍需防护栏 (Guard rails)**：虽然新的 **Safety Modes** 提供了基础检查，但一些成员强调需要进一步的防护栏以确保全面的用户安全。
   - *这只是一个基础安全检查*，暗示用户应维持额外的安全措施。
- **教模型学习当地语言**：一位成员幽默地询问教 **Command-R** 理解多伦多俚语是否可行，并认可了其现有的能力。
   - 这引发了关于语言模型偏见的讨论，其中一人指出了潜在的**加拿大偏见**。
- **Cohere 讨论中的新候选人**：一位新成员介绍了自己，表示他们最近申请了 Cohere 专注于**日语**的职位。
   - 社区反应积极，欢迎新成员加入小组。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1285329176496312360)** (1 messages): 

> - `Fine-tuning models` (微调模型)
> - `Dataset management` (数据集管理)
> - `Cohere platform capabilities` (Cohere 平台功能)


- **关于在微调中跳过结束 Token 的咨询**：一位用户询问是否可以在微调过程中根据需求跳过最终的 `<|END_OF_TURN_TOKEN|>`，旨在实现更无缝的推理衔接。
   - 他们提出了一个训练数据的 POC 示例，并表示他们认为这个功能可能具有潜力，特别是对于微调聊天模型。
- **了解 Cohere 的数据集管理**：讨论涉及了 Cohere 平台上[列出的数据集类型](https://docs.cohere.com/docs/datasets#dataset-types)及其对微调数据集管理的影响。
   - 涵盖的关键点包括数据集限制、保留策略，以及通过 Dashboard 或 [Datasets API](https://docs.cohere.com/reference/create-dataset) 管理数据集的能力。



**提到的链接**：<a href="https://docs.cohere.com/docs/datasets#dataset-types">Datasets — Cohere</a>：该文档提供了 Dataset API 的概述，包括文件大小限制、数据保留策略、数据集创建、验证、元数据保留、使用数据集微调模型等。

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1285347997911355483)** (4 messages): 

> - `Sagemaker Client Issues` (Sagemaker 客户端问题)
> - `Cohere Support` (Cohere 支持)


- **Sagemaker 中计费单位显示为 -1.0**：一位用户报告称，在调用端点时，Sagemaker 客户端的响应中计费单位返回 `input_tokens=-1.0` 和 `output_tokens=-1.0`。
   - 这个问题引发了关于设置端点时潜在的输入配置错误或异常的疑问。
- **建议针对 Sagemaker 咨询寻求支持**：另一位用户建议原帖作者联系 [support@cohere.com](mailto:support@cohere.com) 以获得进一步帮助。
   - 他们提出可以查看用户的账户，以更好地解决计费单位问题。


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1285601778943135844)** (3 messages): 

> - `GitHub Responses` (GitHub 回复)
> - `CodeBlueprint with Aider` (结合 Aider 的 CodeBlueprint)
> - `Ruff Check Errors` (Ruff 检查错误)


- **GitHub 沟通更新**：@connorshorten 通知称，他已在 **GitHub** 上就正在进行的讨论回复了 Prashant。
   - *请关注* Prashant 对此次互动的后续反应。
- **展示 CodeBlueprint 模式**：一位成员分享了一个链接，展示他们的新模式：**结合 Aider 的 CodeBlueprint**。
   - 此次展示可以为编码实践中新工具的集成提供见解。
- **遇到 Ruff 检查错误**：Prashant 报告在运行 `ruff check . --fix-only` 时遇到错误，引用了 **TOML 解析错误**。
   - 该错误表明配置中存在**未知字段 `indent-width`**，与预期参数不匹配。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1285339587677257738)** (11 messages🔥): 

> - `GPT-4 Vision API 封装器`
> - `贡献与悬赏 (Bounties)`
> - `文档需求`
> - `DSPy 程序 API 的灵活性` 


- **介绍 GPT-4 Vision API 封装器**：一个新的 [Pull Request](https://github.com/stanfordnlp/dspy/pull/682) 在 DSPy 仓库中增加了 **GPT-4 Vision API 封装器**，简化了图像分析的请求流程。
   - 该更改在 `visionopenai.py` 中引入了一个 **GPT4Vision** 类，优化了 API 交互过程。
- **对贡献和悬赏的兴趣**：成员们表达了贡献意愿，其中一人问道：*“我很想贡献，有什么你们想好的悬赏（bounties）吗？”*
   - 虽然大家承认需要进行一些更改，但并未讨论具体的可用悬赏细节。
- **对简单文档的需求**：一名成员指出需要 **简单的文档**，并表示愿意协助编写。
   - 这反映了社区在改进用户资源和支持方面的持续努力。
- **DSPy 程序灵活性咨询**：有人提出了一个普遍问题，即是否可以从 Python 以外的其他编程语言**调用优化后的 DSPy 程序**，而不局限于 Python。
   - 另一名成员建议这可能需要一个 **Python VM**，并提到复杂的进程间通信可能是另一种替代方案。



**提到的链接**：<a href="https://github.com/stanfordnlp/dspy/pull/682">Add GPT-4 Vision API wrapper by jmanhype · Pull Request #682 · stanfordnlp/dspy</a>：在 visionopenai.py 中引入一个新的 GPT4Vision 类来封装 GPT-4 Vision API。该抽象层简化了向 API 发起图像分析请求的过程。关键功能...

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1285384200144556082)** (10 messages🔥): 

> - `图像合成技术`
> - `用于图像处理的 Pillow 库`
> - `图像中的文本集成`
> - `使用 Nouswise 的创作过程`
> - `Whisper 语音支持` 


- **合成技术：基础但有效**：成员们讨论了基础的 **合成 (Compositing)** 技术是创建图像的可行方案，并推荐了 **Pillow** 等特定库。
   - 一位成员强调，*不建议通过训练带有集成文本的图像*来获得海报级的视觉效果。
- **后期处理以获得更好的图像质量**：实现高质量图像的有效工作流涉及 **GIMP** 等工具，后期处理可以极大地增强准确性和效果。
   - 有人指出，与其他方法相比，*在后期处理中完成这些工作效果最好*。
- **Nouswise：创作过程的工具**：**Nouswise** 被重点介绍为一个个人搜索引擎，在从 **阅读** 到 **策展** 的创作过程各个阶段提供值得信赖的答案。
   - 其功能包括有效的 **搜索** 和 **写作** 方法，从而提高整体生产力。
- **寻求 Whisper 语音支持**：一位用户询问了关于 **Whisper speech** 技术的经验，随后有人建议查看特定频道以获取指导。
   - 社区参与提供了一条集体知识共享的途径，相关链接确保了对讨论内容的访问。
- **StyleTTS-ZS 的资源支持**：一名成员为讨论中链接的 **StyleTTS-ZS** 项目请求计算资源支持。
   - 该 **GitHub** 项目承诺使用先进技术实现高效、高质量的 Zero-shot 文本转语音合成，并鼓励社区协作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/yl4579/StyleTTS-ZS">GitHub - yl4579/StyleTTS-ZS: StyleTTS-ZS: Efficient High-Quality Zero-Shot Text-to-Speech Synthesis with Distilled Time-Varying Style Diffusion</a>：StyleTTS-ZS：通过蒸馏时变风格扩散实现高效高质量的零样本文本转语音合成 - yl4579/StyleTTS-ZS</li><li><a href="https://nouswise.com">Nouswise</a>：Nouswise 是你的个人搜索引擎，基于你最信任的个人信息。
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 messages): 

mkaic: https://mistral.ai/news/pixtral-12b/
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1285509730889891871)** (5 messages): 

> - `Open Interpreter updates`
> - `Beta testing inquiry`
> - `01 app functionality` 


- **Open Interpreter 的灵巧性**：一位成员称赞 **Open Interpreter** 非常**聪明（clever）**，强调了其潜在的能力。
   - *Geese* 表达了对该工具在社区内功能的兴奋。
- **关于 Beta 测试机会的咨询**：一位成员询问 **Open Interpreter** 是否仍有 **Beta 测试名额**，并表达了分享创新想法的兴奋之情。
   - 该咨询表明了用户对参与其开发的持续兴趣和意愿。
- **关于 01 App 功能的问题**：一位成员询问其他人是否成功在手机上运行 **01 app**，这释放了潜在的用户关注或兴趣信号。
   - 另一位成员确认他们成功运行了 **01 app**，表明了积极的用户体验。
- **关于 01 Light 功能的讨论**：一位成员提到了 **01 light**，可能指的是 **01 app** 背景下的某个功能或更新。
   - 这指向了关于该 App 相关各种功能的持续讨论。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1285631612490612748)** (2 messages): 

> - `Human Device Discord Event`
> - `Beta Availability Inquiry` 


- **Human Device 将于本周五举办 Discord 活动**：一条消息强调了 **Human Device** 计划于本周五举办的活动。感兴趣的参与者可以通过 [Discord 链接](https://discord.gg/UmXdvf3v?event=1285618083448225813)加入。
- **关于 Beta 可用性的咨询**：一位成员询问当前的 **Beta** 版本是否还有名额。这表明用户对参与 Beta 测试有着持续的兴趣。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1285668862330933309)** (2 messages): 

> - `Tool Use Podcast`
> - `01 Voices Script`
> - `Voice Agents in Group Conversations`
> - `Deepgram Local Version` 


- **Tool Use 播客精彩集数：对话 Killian**：本周的 [Tool Use](https://youtu.be/La9BfaFTsFU) 节目邀请了特别嘉宾 **Killian Lucas**，深入探讨语音智能领域。
   - 本集讨论了语音 Agent 的创新，Killian 作为创作者分享了见解。
- **展示可扩展的 01 Voices 脚本**：**mikebirdtech** 强调了一个展示 **01** 强大扩展性的精彩片段，其中包含由 **<@601600720864542741>** 编写的令人印象深刻的脚本。
   - 演示内容包括语音 Agent 如何积极参与群聊，而不会对无关陈述做出过度反应。
- **社区贡献：开源版 Deepgram**：一位成员宣布他们创建了一个开源且本地化的 **Deepgram** 版本，并对该项目表示兴奋。
   - 这突显了社区在开发语音智能工具方面的参与度。



**提到的链接**：<a href="https://youtu.be/La9BfaFTsFU">The Future of Voice Agents with Killian Lucas - Ep 5 - Tool Use</a>：加入本周的 Tool Use 节目，我们将深入探讨令人兴奋的语音智能世界。我们邀请到了特别嘉宾 Killian Lucas，他是...的创作者。

  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1285672763239170212)** (5 messages): 

> - `Eleuther Eval Recipe`
> - `Cache Management`
> - `Model Generation Issues` 


- **关于 Eleuther Eval 在 Generation 和 MC 任务中的疑虑**：一位用户质疑 Eleuther eval recipe 是否能有效处理 **generation** 和多项选择 (**mc**) 任务，怀疑来自 generation 任务的 **cache** 可能会影响后续任务。
   - 另一位用户确认该 recipe 确实存在问题，表明缓存管理可能存在底层问题。
- **需要重置缓存**：讨论了是否因为缺少缓存重置而导致问题，特别是在生成后切换任务时。
   - 一位成员评论说，他们在每次模型生成后都会重置缓存，但这仅是为新生成做准备，并不具备完整的重置功能。
- **MM Evals 中的 Batch Size 预期**：发现了一个并行问题，即在启用缓存的模型评估期间，并不总是能达到预期的 Batch Size。
   - 当另一位用户在未来尝试 MM evaluations 时，此问题可能会再次出现。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1285560688991014975)** (2 messages): 

> - `RISC-V support` 


- **RISC-V 支持的未来计划**：一位成员询问了是否有支持 **RISC-V** 的计划。
   - 另一位成员回应称，目前**尚无计划**支持 RISC-V。
- **社区对 RISC-V 的兴趣**：关于 RISC-V 的提问表明，**社区兴趣**在不断增长，希望更新对多样化架构的支持。
   - 现阶段缺乏计划可能会引发未来关于替代架构兼容性的进一步讨论。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1285318423169204289)** (2 messages): 

> - `Zero-copy data interoperability`
> - `Mandelbrot example`
> - `LLVM intrinsics in Mojo` 


- **零拷贝数据互操作性仍是一个挑战**：目前不支持从 Python 导入 Mojo 模块或调用其函数，这被视为零拷贝互操作性的先决条件。
   - 成员们好奇在 Mojo 和 Python 之间不进行拷贝而传输数据是否可行，并引用了 Mandelbrot 示例（如 `numpy_array.itemset((row, col), matrix[row, col])`）作为潜在的低效案例。
- **Mandelbrot 集合展示了 Mojo 的能力**：关于 Mandelbrot 集合的教程强调了 Mojo 如何在利用 Python 生态系统进行可视化的同时编写高性能代码。
   - 它强调了 Mojo 适用于开发快速程序，特别是针对可以从 Python 库中受益的非规则应用。
- **Mojo 在编译时扩展了 LLVM intrinsics 支持**：在 Mojo 社区会议之后，透露出 Mojo 现在在编译时（comptime）支持 LLVM intrinsics，特别是处理整数的函数如 `ctlz` 和 `popcount`。
   - 这一特性将简化未来支持其他类型的扩展，前提是 LLVM 能够对这些 intrinsics 进行常量折叠（constant fold）。



**提到的链接**：<a href="https://docs.modular.com/mojo/notebooks/Mandelbrot">Mandelbrot in Mojo with Python plots | Modular Docs</a>：学习如何编写高性能 Mojo 代码并导入 Python 包。

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1285450423636983920)** (2 messages): 

> - `Shampoo in Transformers`
> - `Liger usage`
> - `Shampoo Scaling Law`
> - `Performance of Shampoo`
> - `Shampoo vs Adam` 


- **Shampoo 在 Transformers 和 Axolotl 中的缺失**：一位成员对 **Transformers** 和 **Axolotl** 中缺乏 **Shampoo** 实现提出了疑问，引发了对其潜在益处的讨论。
   - 原作者指出，*Shampoo 在大规模、可预测的方式下简直就是免费的午餐*，但似乎被忽视了。
- **关于 Shampoo 缩放定律的讨论**：分享了一个讨论语言模型 **Shampoo 缩放定律（Scaling law）** 的链接，并将其性能与 **Adam** 进行了对比。
   - 该图表引用了 **Kaplan et al** 的研究，并强调了 Shampoo 在大型模型中有效的缩放特性。



**提到的链接**：<a href="https://x.com/cloneofsimo/status/1836003682141577418">Simo Ryu (@cloneofsimo) 的推文</a>：语言模型的 Shampoo 缩放定律图表，风格类似于 Kaplan et al，但对比了 Shampoo 和 Adam。Shampoo 在大规模、可预测的方式下简直就是免费的午餐。

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1285380011775230053)** (1 messages): 

> - `YOLO Vision 2024`
> - `Ultralytics Event`
> - `Google Campus for Startups` 


- **加入 Ultralytics 的 YOLO Vision 2024！**：Ultralytics 将于 <t:1727424000:F> - <t:1727458200:t> 在马德里的 Google Campus for Startups 举办 [YOLO Vision 2024](https://www.ultralytics.com/events/yolovision)。
   - 请务必注册参加，并参与决定社区讨论环节的音乐！
- **为 YOLO Vision 2024 的音乐投票！**：注册 [YOLO Vision 2024](https://www.ultralytics.com/events/yolovision) 后，与会者可以为讨论环节播放的音乐投票。
   - 这一互动环节旨在增强活动期间的参与度，鼓励社区投入！


  

---



---



---



---



---



---



{% else %}


> 完整的频道细分内容已为邮件格式截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}