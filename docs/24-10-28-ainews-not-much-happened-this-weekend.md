---
companies:
- moondream
- openai
- anthropic
- hugging-face
- mistral-ai
- google-deepmind
- langchain
- deepmind
- microsoft
date: '2024-10-28T22:27:43.875190Z'
description: '**Moondream**（一个 1.6b 参数的视觉语言模型）获得了种子轮融资，凸显了以“月亮”命名的微型模型趋势，与之并列的还有 **Moonshine**（27-61m
  参数的自动语音识别模型）。**Claude 3.5 Sonnet** 被用于 AI 推特摘要。


  讨论内容涉及大语言模型（LLM）中**模式识别**与**智能**的对比、用于提示词优化的**强化学习**，以及 **NotebookLlama**（一个开源的
  **NotebookLM** 变体，使用 **LLaMA 模型**处理**文本转语音**等任务）。在**模型优化**方面，提到了 **PyTorch** 中用于**张量并行**的
  **async-TP** 进展以及超参数调优。


  **Mini-Omni 2** 展示了涵盖**图像**、**音频**和**文本**的多模态能力，可用于语音对话，并强调了**模态对齐**和**多模态微调**。文中还介绍了一些
  AI 生产力工具，如 **AI 邮件撰写器**和基于 **LlamaCloud** 的研究助手。此外，还强调了实际技能开发以及使用 **Llama3-8B**
  进行注重隐私的 AI 工具应用。


  分享的生成式 AI 工具包括 **#AIPythonforBeginners** 以及使用 **LangGraph** 构建的 **GenAI 智能体**。商业洞察涵盖了
  AI 产品开发中的快速执行以及新兴的 AI 相关岗位。最后，讨论了企业级 text-to-SQL 和高级检索方法面临的挑战，并提供了使用 **LangChain**
  和 **MongoDB** 的 **RAG**（检索增强生成）应用教程。'
id: d98ed4a0-b5e7-47b2-8fa7-611b82789b8f
models:
- claude-3.5-sonnet
- llama-3
- llama-3-8b
- notebookllama
- min-omni-2
original_slug: ainews-not-much-happened-this-weekend-2670
people:
- amanda-askell
- philschmid
- stasbekman
- francois-fleuret
- mervenoyann
- reach_vb
- dzhng
- aravsrinivas
- sama
- lateinteraction
- andrew-y-ng
- bindureddy
- jerryjliu0
title: 这个周末没发生什么特别的事。
topics:
- pattern-recognition
- reinforcement-learning
- prompt-optimization
- text-to-speech
- model-optimization
- tensor-parallelism
- hyperparameters
- multimodal
- modal-alignment
- multimodal-fine-tuning
- ai-productivity
- privacy
- generative-ai
- rag
- retrieval-augmentation
- enterprise-text-to-sql
---

<!-- buttondown-editor-mode: plaintext -->**一个安静的周末正是你所需要的。**

> 2024/10/25-2024/10/28 的 AI News。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 以及 **32** 个 Discord（**230** 个频道和 **5833** 条消息）。预计节省阅读时间（以 200wpm 计算）：**601 分钟**。你现在可以在 AINews 讨论中标记 [@smol_ai](https://x.com/smol_ai) 了！

[恭喜 Moondream](https://x.com/VentureBeat/status/1850885273749532852)（一个 1.6b 的 vision language model）获得种子轮融资。随着 [Moonshine](https://github.com/usefulsensors/moonshine)（27-61m 的 ASR model）也引起了一些关注，以月亮为主题的小型模型似乎形成了一种小趋势。

https://youtu.be/T7sxvrJLJ14
 

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

**AI 研发**

- **高级语言模型与技术**：[@AmandaAskell](https://twitter.com/AmandaAskell/status/1850597117624283626) 提出了一个**真诚的疑问**：**LLM** 中的**模式识别**与**智能**有何区别。[@cwolferesearch](https://twitter.com/cwolferesearch/status/1850919005407662435) 讨论了使用 **RL** 为 **LLM** **优化提示词 (prompts)**，并强调了**离散 Token 优化**面临的挑战。[@ophilschmid](https://twitter.com/_philschmid/status/1850800978070564911) 介绍了 **NotebookLlama**，这是 **NotebookLM** 的开源版本，利用各种 **LLaMA 模型**处理**文本转语音 (text-to-speech)** 等任务。
  
- **模型优化与效率**：[@Philschmid](https://twitter.com/_philschmid/status/1850800980100542795) 分享了一个与 **NotebookLlama** 相关的**示例**。[@StasBekman](https://twitter.com/StasBekman/status/1850696223092850848) 重点介绍了 **PyTorch** 中用于**张量并行 (Tensor Parallelism)** 的 **async-TP** 实现，提高了**计算效率**。[@francoisfleuret](https://twitter.com/francoisfleuret/status/1850782782416674878) 讨论了**模型超参数**，特别是 **LLM** 中的 **d_model** 和 **n_heads**。

- **多模态机器学习**：[@mervenoyann](https://twitter.com/mervenoyann/status/1850922155740962831) 展示了 **Mini-Omni 2**，该模型能够理解**图像**、**音频**和**文本**输入，用于**语音对话**。[@Reach_vb](https://twitter.com/reach_vb/status/1850895844167286859) 详细介绍了 **Mini-Omni 2** 的**技术概览**，强调了**模态对齐**和**多模态微调**。

**AI 应用与工具**

- **AI 生产力工具**：[@dzhng](https://twitter.com/dzhng/status/1850598769848700966) 推广了一款旨在提高**效率**且没有 **AI 垃圾邮件感 (AI email slop)** 的 **AI 邮件撰写工具**，能够生成**经过充分研究的邮件序列**。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1850656330119573734) 介绍了一个**知识助手视频系列**，利用 **LlamaCloud** 构建 **AI 研究助手**。

- **AI 增强型软件开发**：[@sama](https://twitter.com/sama/status/1850610764467544558) 强调了在**技能提升**方面，**实际练习**比**复杂的先决计划**更重要。[@Lateinteraction](https://twitter.com/lateinteraction/status/1850920738955685957) 讨论了使用 **DSPy 优化器**训练 **Llama3-8B**，以实现**注重隐私的 AI 工具使用**。

- **生成式 AI 工具**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1850938085137342550) 强调了 **#AIPythonforBeginners** 在**自动化任务**和**集成 LLM** 方面的影响。[@LangChainAI](https://twitter.com/LangChainAI/status/1850930775589519633) 分享了 **GenAI Agents** 开发资源，重点关注 **LangGraph** 中的 **Agent 架构**。

**AI 商业与初创公司**

- **初创公司执行与 AI 集成**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1850912176896463328) 讨论了在 **AI 驱动的产品开发**中**快速执行的重要性**，并概述了增强**市场契合度**的**反馈循环策略**。[@Bindureddy](https://twitter.com/bindureddy/status/1850905700828115426) 预测了**后 AI 时代新工作岗位**的演变，例如 **AI Agent 主管**和**后备人员 (fallback humans)**。

- **软件行业中的 AI**：[@Jerryjliu0](https://twitter.com/jerryjliu0/status/1850655329333522487) 强调了**企业级 text-to-SQL** 的挑战以及**高级检索**方法的必要性。[@LangChainAI](https://twitter.com/LangChainAI/status/1850605886613422576) 提供了使用 **LangChain** 和 **MongoDB** 优化 **RAG 应用**的教程。

**软件工程与 ML 工程**

- **软件开发实践**：[@scottastevenson](https://twitter.com/scottastevenson/status/1850606747188396156) 批判了**软件工程的演变**，指出了诸如**软件设计**与**构建**之间**缺乏区分**，以及与传统工程学科相比，**软件设计**所需的**细节导向**等问题。

- **机器学习工程**：[@LangChainAI](https://twitter.com/LangChainAI/status/1850681382969774249) 讨论了在利用**小型本地 LM** **构建应用**中使用 **LangGraph.js**，推广了**开源模型**的优势。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 结合 RAG 的小型 LLM：1B-3B 模型令人惊讶的能力**

- **[glm-4-voice-9b 现在可以在 12GB GPU 上运行](https://v.redd.it/v01grshffbxd1)** ([Score: 109, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1gddatq/the_glm4voice9b_is_now_runnable_on_12gb_gpus/)): **glm-4-voice-9b** 模型现在能够在 **12GB GPU** 上运行，从而实现更高效的推理。这一进展使得该模型更易于获取和使用，并可能在配置较低的硬件上扩展其在语音相关 AI 任务中的应用。
  - 用户在 **RTX 3060 12GB GPU** 上测试了 **glm-4-voice-9b** 模型，报告称其可以运行，但在实时对话中并不流畅。一些用户在 [Runpod](https://www.runpod.io/) 上遇到了 **30-60 秒的延迟**和噪声生成问题。
  - 关于 AI 语音助手未来发展的讨论，预测从 **3 年**到短至 **6-12 个月**不等，即可达到与当前 ChatGPT 语音功能相当的能力。**Moshi** 被认为是该领域的潜在领导者。
  - 提示词 "*cry about your lost cat*"（为丢失的猫哭泣）引发了网友的兴趣，突显了该模型多样化且有时出人意料的使用场景。
- **我测试了小型 LLM (1B/3B) 在本地 RAG 中的实际表现 - 这是我的心得** ([Score: 542, Comments: 67](https://reddit.com//r/LocalLLaMA/comments/1gdqlw7/i_tested_what_small_llms_1b3b_can_actually_do/)): 在 **MacBook Pro M1 Pro** 上测试了 **Llama3.2 3B** 的本地 RAG 表现，使用了包括 **Nomic 的 embedding 模型**、**Langchain RAG 工作流**和 **Chroma DB** 在内的配置。该系统在 **英伟达 2025 年第二季度财报** 的基础问答中表现良好，**PDF 加载时间不到 2 秒**，简单信息检索的速度略快于 **Claude 3.5 Sonnet**。作者还尝试使用 **LoRA** 进行特定任务（如生成图表），并使用 **Octopus_v2 action 模型**作为任务路由，展示了小型基础模型配合任务特定“插件”在本地 RAG 系统中的潜力。
  - 用户讨论了本地 RAG 系统的潜在应用，包括 **桌游规则查询器** 和 **儿童教育工具**。后者使用 **Gemma2 27B** 来解释概念并生成有关学校科目的问题。
  - **小型基础模型配合可插拔 LoRA** 的概念被拿来与 **苹果的端侧智能 AI 方案** 进行比较。分享了一篇讨论苹果参考架构实现 GenAI 的 [博客文章](https://predibase.com/blog/breaking-down-apples-reference-architecture-for-genai-small-fine-tuned-and)。
  - 关于 **embedding 模型** 的讨论澄清了对于此类模型而言，**137M 参数** 并不算小。引用了 **Hugging Face MTEB 排行榜**，显示顶级模型使用的 **内存超过 1GB**。


**主题 2. 多模态模型：Llama 3.2 Vision 和 Pixtral 的进展**

- **有人注意到 Ollama 已经推出了 llama3.2-vision 测试版吗？** ([Score: 76, Comments: 35](https://reddit.com//r/LocalLLaMA/comments/1gd6df6/has_anyone_realized_that_ollama_has_launched/)): **Ollama** 发布了 **Llama 3.2 Vision** 的测试版，这是一个能够同时处理文本和图像的 **多模态模型**。这个新模型需要 **Ollama 0.4.0**（目前为 **预发布版本**），可以通过 [ollama.com/x/llama3.2-vision](https://ollama.com/x/llama3.2-vision) 获取。
  - 用户对 **Qwen-VL** 和 **Minicpm 2.6** 等其他模型表示了兴趣，一些人指出 **Qwen-VL 性能更优**，而 **Minicpm 发布更早** 且已兼容 Ollama。
  - 有人担心 **Ollama** 通过直接与 Meta 合作，可能会 **劫持来自 llama.cpp 生态系统** 的努力。然而，相关人员澄清该实现是由 **Ollama 贡献者** 完成的，且代码保持 **开源**。
  - Ollama 中的 **Llama 3.2 Vision 模型** 可以处理图像，但 **无法生成图像**。用户讨论了其性能，一些人觉得满意，而另一些人则在等待 **Pixtral** 等模型。

- **Pixtral 表现惊人。** ([Score: 167, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1gdr3al/pixtral_is_amazing/)): **Pixtral** 在**图像分析**和**文本到文本任务**中均展现出令人印象深刻的性能，在发布者的测试中超越了 **MiniCPM-V-2.6** 和 **Llama3.2 11B Vision**。该模型结合了与 **MiniCPM** 相当的强大视觉理解能力和卓越的文本生成能力，同时保持了极低的审查限制，使其成为多模态 AI 任务的通用选择。
  - **Qwen2-VL** 和 **Molmo-7B** 是推荐的 Pixtral 替代方案，其中 Molmo 提供了一个具有 **1B 激活参数**的 **7B MoE 变体**。用户报告其审查问题极少且性能强劲，尽管 Pixtral 在**头脑风暴/故事创作**方面表现更优。
  - Pixtral 可以使用 **vllm** 在本地运行，并针对 **12GB VRAM** 和 **64GB RAM** 进行了特定设置。该模型在个人工作中表现良好，在受限硬件上能以 **1.1 tokens/s** 的速度快速生成图像描述。
  - **Llama 3.2 70b vision** 被报告为 **OCR 领域的 SOTA**，**MiniCPM 2.6v** 紧随其后。Pixtral 的功能类似于 ChatGPT 的图像分析能力，允许用户针对给定图像提问。


**主题 3. 推理引擎之战：Llama.cpp vs MLC LLM vs vLLM**

- **[推理引擎之战：Llama.cpp vs MLC LLM vs vLLM。针对单卡 RTX 3090 和 4 卡 RTX 3090 的测试。](https://www.reddit.com/gallery/1gdccyr)** ([Score: 86, Comments: 41](https://reddit.com//r/LocalLLaMA/comments/1gdccyr/battle_of_the_inference_engines_llamacpp_vs_mlc/)): 该帖子比较了 **Llama.cpp**、**MLC LLM** 和 **vLLM** 推理引擎在基于 **RTX 3090** 显卡的**单 GPU** 和**多 GPU** 配置下的性能。测试采用了多种配置，包括**单卡 RTX 3090** 和 **四卡 RTX 3090** 的设置，以评估这些推理引擎在大语言模型上的效率和速度。
  - **MLC LLM** 的性能令用户印象深刻，其速度和量化选项包括 **q0f16, q3f16_0, q4f16_0** 等。一些人注意到它在**短上下文**场景中表现出色（**在 3090 上运行 Qwen 7b q4f16 可达 150t/s**），但在**长上下文（>8k）**下会变慢。
  - 用户讨论了**批处理推理（batch inference）**能力，其中一位报告称，在单块 **RTX 3090 Ti** 上使用 **vLLM** 配合 **W8A8 INT8** 模型和 eager 模式下的 **flashInfer engine**，每小时可处理 **81M 输入 token 和 5.5M 输出 token**。
  - 对未来基准测试的建议包括测试 **exllama v2**、比较 **PCIe 带宽需求**以及评估 GPU 之间使用 **NVLINK** 的性能。推荐使用 [MMLU Pro 测试](https://github.com/chigkim/Ollama-MMLU-Pro.git)进行批处理推理比较。

**主题 4. Meta 的开源版 NotebookLM：增强文档交互**

- **[Meta 发布 Google NotebookLM 的开源版本](https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/NotebookLlama)** ([Score: 76, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1gdk92b/meta_releases_an_open_version_of_googles/)): Meta 发布了 **Llama-Coder**，这是 Google NotebookLM 的开源版本，旨在辅助**代码生成与分析**。该工具基于 Meta 的 **Llama 2** 语言模型构建，提供与 NotebookLM 类似的功能，包括对代码的**上下文理解**以及跨多种编程语言**生成、解释和调试**代码的能力。

**主题 5. 顶尖编程模型：Qwen 2.5 32B 及其 70B 以下的替代方案**

- **在编程方面，是否有比 Mistral-Nemo 12b 更强，但体积仍小于 Llama 3.1 70b 量化版的模型？** ([Score: 30, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1gdpvoy/is_there_anything_that_beats_mistralnemo_12b_in/)): 这篇标题为“**在编程方面，是否有比 Mistral-Nemo 12b 更强，但体积仍小于 Llama 3.1 70b 量化版的模型？**”的帖子讨论了各种语言模型在编程任务中的表现。**Qwen 2.5 32B** 被提及在编程基准测试中优于更大的模型，这可能使其成为介于 **Mistral-Nemo 12B** 和 **Llama 3.1 70B** 之间的强力竞争者。
  - **Qwen 2.5 32B** 在编程基准测试中优于 **Llama 3 70B**，并接近 **Llama 3.1 70B**，在 [Aider benchmark](https://aider.chat/docs/leaderboards/) 上得分为 **54.1%**。对于内存受限的配置，它被推荐使用，因为它允许使用 **Q8** 量化，而不是 70B 模型所使用的 Q3/Q4 量化。
  - 建议了几款较小的模型作为 **Mistral-Nemo 12B** 的替代方案，包括 **Qwen Coder 2.5 7B**、**Yi Coder 9B** 和 **Codestral**。文中提到即将推出的 **32B 版本的 Qwen Coder** 可能是该尺寸范围内最大的代码专用模型。
  - 用户建议针对特定的编程用例测试 **Qwen2.5 14B**、**Mistral Small** 和 **Gemma 2 27B** 等模型，因为它们的实际表现可能与基准测试结果有所不同。**DeepSeek-Coder-V2 236B** 被指出是一款规模大得多的代码专用模型。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 研究与技术**

- **Google DeepMind 推进多模态学习**：来自 [Google DeepMind 的论文](https://arxiv.org/html/2406.17711v1) 展示了如何通过联合样本选择进行数据策展（data curation），从而加速多模态学习。(/r/MachineLearning)

- **Microsoft 的 MInference 加速长上下文推理**：[Microsoft 的 MInference 技术](https://arxiv.org/abs/2407.02490) 能够在保持准确性的同时，实现长上下文任务中高达数百万个 token 的推理。(/r/MachineLearning)

- **扩展合成数据生成**：一篇关于 [扩展合成数据生成](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/) 的论文利用 10 亿个从网络策展的角色（personas）来生成多样化的训练数据。(/r/MachineLearning)

**AI 模型发布与改进**

- **Salesforce 发布 xLAM-1b 模型**：Salesforce 的 10 亿参数 xLAM-1b 模型 [在函数调用（function calling）方面实现了 70% 的准确率，超越了 GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。(/r/LocalLLaMA)

- **更新版 Phi-3 Mini 支持函数调用**：Rubra AI 发布了更新后的 Phi-3 Mini 模型，[具备函数调用能力](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争。(/r/LocalLLaMA)

- **IC-Light V2 演示版发布**：基于 Flux 模型的 IC-Light V2 演示版已在 [Hugging Face 上发布](https://huggingface.co/spaces/lllyasviel/iclight-v2)。权重尚未发布，且该模型将是非商业用途。(/r/StableDiffusion)

**AI 训练与微调技术**

- **详细的 SDXL 微调过程分享**：一位开发者分享了 [针对 4000 万样本微调 SDXL](https://www.reddit.com/r/StableDiffusion/comments/1gdkpqp/the_gory_details_of_finetuning_sdxl_for_40m/) 的大量细节，包括数据集准备、质量建模、打标（captioning）和训练细节。(/r/StableDiffusion)

**AI 伦理与社会影响**

- **关于 AI 训练数据伦理的辩论**：围绕使用受版权保护的材料训练 AI 模型的伦理问题引发了讨论，一些人主张 [合理使用（fair use）](https://www.reddit.com/r/singularity/comments/1gd5yxq/i_think_we_could_have_a_problem_with_this_down/)，而另一些人则担心对内容创作者的潜在影响。(/r/singularity)

- **James Cameron 对 AGI 表示担忧**：电影导演 James Cameron [对 AGI 导致超智能](https://www.reddit.com/r/singularity/comments/1gdbec3/james_cameron_says_that_agi_will_inevitably_lead/) 及潜在冲突表示担忧，引发了关于 AI 安全和 AI 人格化的辩论。(/r/singularity)

**AI 应用与演示**

- **Neuralink 竞争对手的眼部植入物恢复视力**：一家 [Neuralink 竞争对手报告称](https://www.reddit.com/r/singularity/comments/1gdcpi7/a_neuralink_competitor_says_its_experimental_eye/)，其置于视网膜下的 2 毫米实验性眼部植入物在临床试验中使盲人恢复了视力。(/r/singularity)

- **AI 展示 Minecraft 建造能力**：[AI 模型 Sonnet 3.6 展示了在没有特定训练的情况下在 Minecraft 中建造复杂结构的能力](https://www.reddit.com/r/singularity/comments/1gd7jry/sonnet_36_is_told_to_build_a_massive_mansion_in/)，展示了涌现能力（emergent capabilities）。(/r/singularity)

- **为视频生成的 AI 音乐**：一个名为 MuVi 的新 AI 系统可以通过分析重要特征并使用节奏同步，[生成与视频画面匹配的音乐](https://www.reddit.com/r/singularity/comments/1gdhgkc/muvi_can_generate_music_that_matches_the_visuals/)。(/r/singularity)

**AI 发展与政策**

- **美国国家安全顾问敦促加速 AI 发展**：美国国家安全顾问 Jake Sullivan [呼吁加速 AI 的开发和部署](https://www.reddit.com/r/singularity/comments/1gdu5cz/us_national_security_advisor_jake_sullivan_the_us/)以保持美国的领先地位，并对其他国家的 AI 发展表示担忧。(/r/singularity)

- **Google 的 Project Jarvis 泄露**：关于 Google [Project Jarvis 的泄露凸显了 Gemini 2.0 的潜在进步](https://www.reddit.com/r/singularity/comments/1gdfz97/project_jarvis_leak_highlights_google_gemini_20s/)，暗示 AI 能力将有显著提升。(/r/singularity)

---

# AI Discord 回顾

> 由 O1-mini 总结的总结之总结

**主题 1：模型突破与困境**

- [**Llama 和 Phi 突破性能边界**](https://huggingface.co/spaces/KaliumPotas/Api_recommend)：**Llama-3.1-405B** 和 **Phi-3.5** 模型在**自动化渗透测试**和**图像生成**等任务中展示了令人印象深刻的进展。虽然 Llama 受益于**强化学习**，但 Phi-3.5 却在应对**过度审查**的问题，这限制了它的实际应用。
- [**Stable Diffusion 的起伏历程**](https://sandner.art/stable-diffusion-35-large-what-you-need-to-know/)：**Stable Diffusion 3.5** 引发了褒贬不一的反应，人们在争论其与 **1.5** 版本相比的**速度**与**质量**。包含 **120 位艺术家**和 **140 种风格**的画廊突显了其扩展的艺术能力。
- [**Dualformer 和 Grok 2 进军多模态**](https://openrouter.ai/models?q=google)：**Dualformer** 整合了**快速**（System 1）和**慢速**（System 2）推理，提升了 **Transformer** 的效率。同时，**Grok 2** 引入了**多模态理解**，使其能够同时处理图像和文本，扩大了应用范围。

**主题 2：工具探戈——构建、故障排除与集成**

- [**LM Studio 像专业人士一样驾驭多 GPU**](https://github.com/pytorch/torchtune/pull/1909)：**LM Studio** 现在可以有效地**识别和利用多个 GPU**，为高需求项目优化性能。工程师强调需要特定的**配置调整**来最大化计算效率。
- [**OpenRouter 连接混乱仍在持续**](https://openrouter.ai/inflection/inflection-3-productivity)：持续的 **Cloudflare 错误**困扰着 **OpenRouter**，尽管状态指示灯显示运行正常，但仍导致了中断。用户正在探索切换**浏览器**或**地理位置**等变通方法，以恢复稳定的连接。
- [**Tinygrad 对复数的探索**](https://github.com/tinygrad/tinygrad/pull/7107)：**Tinygrad** 在**复数集成**方面面临挑战，这对于**离散傅里叶变换**等任务至关重要。社区驱动的**仿真策略**和贡献旨在增强这一功能。

**主题 3：协作星群——聚会、学习小组与共享项目**

- [**多伦多科技巨头即将聚会**](https://discord.com/events/1089876418936180786/1299558554872582154)：一场计划在多伦多举行的关于 **NVIDIA GPU** 和 **CUDA** 编程的聚会引发了热议，首场活动定于 **11 月 15 日**举行。组织者邀请**演讲者和合作者**加入 AI 知识交流。
- [**学习小组在 LLM Agents MOOC 中激发同步**](https://forms.gle/QtQ2C6qzomeHDrC38)：热情的学习者提议组建**虚拟学习小组**，深入研究 **LLM Agent 讲座**，促进**同行协作**和**集体学习**，以有效应对复杂的课程材料。
- [**AdaletGPT - 土耳其法律 AI 助手**](https://adaletgpt.com)：**AdaletGPT**（一款**土耳其法律聊天机器人**）的推出，展示了社区对**特定领域 AI 工具**的推动。它使用 **RAG** 框架构建，邀请**协作投入**和**开源贡献**以提供法律援助。

**主题 4：隐私与政策——以伦理为念驾驭 AI**

- [**Phi-3.5 因审查制度遭到抵制**](https://huggingface.co/models?other=base_model:quantized:stabilityai/stable-diffusion-3.5-large)：**Phi-3.5 模型**因其**过度审查的回答**而面临抵制，这阻碍了它在**技术和编码任务**中的有效性。关于 **AI 审查**的适当程度及其对专业场景**可用性**影响的辩论随之而来。
- [**Meta 打造自己的 AI 搜索引擎**](https://github.com/Mark-Inc-Korea/AutoRAG)：为了减少对 **Google** 和 **Bing** 的依赖，**Meta** 正在开发**专有搜索引擎**，反映了行业向 **AI 驱动的信息检索**和**数据主权**的更广泛趋势。
- [**Apple 的百万美元 AI 安全悬赏**](https://x.com/varun_mathur/status/1850502044307562871?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：**Apple** 宣布悬赏 **100 万美元**奖励成功破解其 **AI 服务器**的人员，强调了 **AI 安全**和在 AI 部署中**主动识别漏洞**的关键重要性。

**主题 5：部署困境——配置、GPU 设置与性能调优**

- [**CUDA 配置僵局解决**](https://github.com/pytorch/torchtune/pull/1909)：经过多次**重装和升级**，工程师们解决了 **GPT-NeoX** 的 **CUDA 兼容性问题**，稳定了其**训练环境**以实现稳健的**模型部署**。
- [**Gorilla 的 Function Call 排行榜说明**](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d9f323715d7d2012a955d7f4114e/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json#L42C1-L42C2438)：关于 **Gorilla 排行榜**中 **'multiple' 功能**的说明，强调了其在评估 **LLMs** 的**多步推理**和**函数选择**中的作用，增强了性能评估的**透明度**。
- [**Torchtune 调整优化性能**](https://github.com/pytorch/torchtune/pull/1909)：**Torchtune** 最近的 **LoRA 错误修复**和**配置标志提案**简化了**模型微调**，改进了**单设备和多设备训练设置**，并确保在各种 **LLM 配置**中获得更**一致的性能**。

---

# PART 1: Discord 高层级摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **探索 Hugging Face Spaces**：用户讨论了 Hugging Face Spaces 上提供的各种模型，重点关注**图像生成**和**模型量化**模型，同时分享了特定模型的有用链接。
  
  - 建议包括在本地项目中使用更轻量级的模型，如 **Llama** 和 **Flux**，并强调了它们独特的功能。
- **自动化渗透测试的新基准**：最近的一篇论文引入了一个**基于 LLM 的自动化渗透测试**基准，展示了使用 **PentestGPT 工具**的 **GPT-4o** 和 **Llama 3.1-405B** 等模型。
  
  - 虽然 **Llama 3.1** 占据优势，但两种模型在渗透测试中都表现挣扎，引发了关于通过**强化学习**进行改进的讨论。
- **Stable Diffusion 3.5 画廊...**：一位用户展示了演示 **Stable Diffusion 3.5** 如何诠释艺术风格的画廊，涵盖了超过 **120 位艺术家**和 **140 种风格**。
  
  - 两个画廊均可通过 [**Artists Gallery**](https://enragedantelope.github.io/Artists-SD35L/) 和 [**Styles Gallery**](https://enragedantelope.github.io/Styles-SD35L/) 访问，并详细列出了所使用的提示词（prompts）。
- **讨论 AI 开发工具**：一名成员询问由于公司网络限制而需要的**离线 AI 开发工具**，收到了使用便携式**虚拟机**或 **Docker** 的建议。
  
  - 社区成员警告说，导出和导入环境可能会非常繁琐。
- **Bionic Reading Hub 仓库发布**：分享了一个名为 **Bionic Reading Hub** 的 GitHub 项目，该项目允许将 PDF 转换为 Bionic Reading 格式，以增强可读性。
  
  - 该工具可以辅助处理复杂材料，对网络安全领域的用户尤其有益。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 每日限制引发不满**：用户对新实施的 Audio Overview 生成**每日限制**表示不满，并推测未来可能采取订阅模式。
  
  - *不满情绪源于*许多人感到被有关这些限制的沟通缺乏所蒙蔽，强调了 Google 透明度的必要性。
- **加入 UXR 团队的深度研究**：UXR 团队将于 **10 月 31 日至 11 月 6 日**举行远程 1:1 访谈，以收集参与者对未来发展的反馈，并提供 **75 美元的感谢奖励**。
  
  - 该研究仅提供 **6 个名额**，敦促感兴趣的参与者填写 [资格调查问卷](https://forms.gle/TngndJHaPRdRsnhT7)。
- **介绍 PodCraft：个性化播客**：一位用户提议开发一个名为 **PodCraft** 的应用程序，该程序提供个性化的播客内容，无需筛选大量剧集。
  
  - 该应用旨在以喜爱创作者的声音即时访问内容，迎合那些难以找到相关见解的受挫听众。
- **成功集成 HeyGen 头像**：一位用户分享了一个项目，该项目增强了 **HeyGen 头像**，使其在万圣节特别视频中表现得更加逼真。
  
  - 用户对 AI 生成内容的能力以及取得的显著进步表示兴奋。
- **对开源 AI 模型的看法**：用户青睐各种**开源** AI 图像生成工具，尤其是那些使用限制较少的工具。
  
  - 用户对 **Google 的 Imagen** 表示不满，许多人表示存在更好的模型且没有使用限制。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 性能在修复 Bug 中得到提升**：最近对 **Unsloth** 的升级带来了速度提升，尽管根据社区反馈，`unsloth-zoo` 中由于可索引数据集假设导致了崩溃报告，但仍实现了更好的处理性能。
  
  - 用户在 GitHub 上积极参与解决模型 quantization 问题，体现了社区驱动的故障排除方法。
- **多模态模型集成讨论**：讨论强调了合并视觉和语言模型的复杂性，其中 **adapters** 起着关键作用，促使用户考虑增强兼容性的潜在解决方案。
  
  - **GLM-4** 作为一个支持音频和文本输入的强力示例出现，引发了对音频 adapters 以实现更好多模态交互的兴趣。
- **Gradient Accumulation 影响训练工作流**：成员们分享了关于 **gradient accumulation** 修复后改进的经验，强调了训练效率，但也指出了与 batch sizes 和内存管理相关的挑战。
  
  - 反馈表明，用户在适应 **Unsloth** 最新的 gradient accumulation 功能时存在学习曲线。
- **利用 3D 模型进行 AI 视频生成**：一位成员提议开发使用 **3D 模型** 的 **AI 视频生成器**，加入摄像机控制和一致环境等功能，并可能利用 **Unreal Engine** 物理引擎。
  
  - 这引发了关于将 AI 与视频生成相结合的现有项目的咨询，暗示了社区的协作兴趣。
- **引入 Dualformer 以提高推理效率**：**Dualformer** 模型提出了一种集成快速（System 1）和慢速（System 2）推理的新方法，以提高 Transformer 效率，超越了 **Searchformer** 等前作。
  
  - 它将认知系统理论与 AI 模型相结合，揭示了在迷宫导航和数学等复杂推理任务中的性能进步。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 支持多 GPU 运行**：用户确认 **LM Studio** 能够有效识别并利用多块 GPU，一位成员强调了他们成功配置了两块 **RTX 3060 Ti** 显卡。
  
  - 然而，为了优化两块 GPU 的性能，特定的配置是必要的。
- **NPU 面临功能短板**：一位用户对他们的 **NPU** 表示失望，因为与标准 PC 配置相比，它缺乏对 AI 任务的软件支持。
  
  - 讨论中包括了对 Intel 可能如何与 Microsoft 合作以提升 NPU 能力从而获得更好 AI 性能的推测。
- **对 Apple M3 及未来 M4 性能的质疑**：关于 **Apple M4** 的对话强调了对新 Mac 机型内存限制的担忧，导致人们怀疑它们能否高效处理大型 AI 模型。
  
  - 参与者批评了升级 RAM 的高昂成本，认为这是处理严肃工作负载的主要障碍。
- **高性价比 AI 系统构建技巧**：成员们强调，与购买缺乏足够资源的 Apple 硬件相比，构建具有更高 **RAM** 容量的定制系统更具性价比。
  
  - 共识是，构建一台强大的具备 AI 能力的机器仍然是一个更省钱的选择。
- **混合 GPU 配置性能问题**：针对 **RTX 3090 和 4090** GPU 的混合配置出现了担忧，用户在讨论是否出于兼容性原因卖掉性能更强的显卡。
  
  - 重点在于优化用于处理大型模型的设备，将兼容性置于推理速度之上。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Inflection 恢复在线**：**Inflection** 已解决最近的计费问题，恢复了用户访问最新功能 [Inflection 3 Pi](https://openrouter.ai/inflection/inflection-3-pi) 和 [Inflection 3 Productivity](https://openrouter.ai/inflection/inflection-3-productivity) 的权限。
  
  - 随着计费问题的修复，Inflection 明确了其旨在提高用户生产力的增强功能。
- **OpenRouter 连接混乱持续**：用户正面临持续的 **OpenRouter** 连接问题，尽管状态页显示一切正常，但仍有用户报告 Cloudflare 520 和 524 错误。
  
  - 一些用户怀疑欧洲地区的问题更为严重，并建议尝试使用不同的浏览器作为临时解决方案。
- **Sonnet 模型响应质量令人担忧**：许多用户指出 **Sonnet** 模型的响应质量明显下降，现在生成的通用后续问题比以前更多。
  
  - 这种下降似乎与限制免费版本后的调整有关，导致用户对模型交互性降低表示沮丧。
- **Grok 2 带来多模态理解**：社区对 **Grok 2** 的发布反响热烈，该模型现在具备同时处理图像和文本的能力，扩展了其潜在应用。
  
  - 用户非常期待探索这些多模态能力与市场上现有模型的对比情况。
- **对集成访问权限的需求增长**：大量用户正积极寻求 **Integrations** 的访问权限，表明社区对该功能有着浓厚兴趣。
  
  - 对集成权限的礼貌请求显示出用户群体的积极参与，他们渴望功能扩展，并不断发信息感谢社区成员可能提供的帮助。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **ASR 领域中 Whisper 与 Moonshine 的对比**：参与者分析了 **Whisper** 与 **Moonshine** 等新技术在 ASR 领域的对比，后者声称在边缘设备上具有更高的性能和更低的计算成本。
  
  - 虽然 **Moonshine** 在性能上超过了 Whisper 的小型模型，但批评者认为 Whisper 的大型模型仍保持着性能优势。
- **Apple 在同态加密方面迈出重要一步**：Apple 关于 **Homomorphic Encryption**（同态加密）的公告标志着一项显著创新，允许在不牺牲机密性的情况下在 AI 中使用私有数据，这被类比为 AI 的 **HTTPS 时刻**。
  
  - 专家讨论了潜在的实现方式，例如在不暴露隐私信息的情况下进行数据检索，尽管推理速度问题仍是一个挑战。
- **Moondream 获得 450 万美元融资**：**Moondream** 确认完成一轮成功融资，筹集了 **450 万美元**，用于测试小型 AI 模型在竞争环境中的有效性。
  
  - 这笔资金引发了关于小型模型在克服行业普遍障碍时能力局限性的辩论。
- **提升编程效率的 Cursor Pro 技巧**：成员们分享了 **Cursor Pro Tips**，强调了如 `ctrl+k` 进行局部编辑等快捷键，显著增强了编程工作流。
  
  - 鉴于目前仅探讨了一部分潜在实践，大家对后续深入探讨这些技巧的环节表现出浓厚兴趣。
- **Discord 中的音频问题**：用户报告在会议期间遇到 **音频问题**，这阻碍了他们有效参与和跟踪讨论。
  
  - 针对 **Discord** 服务器性能的担忧浮出水面，认为这可能是导致持续音频问题的因素之一。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 策展人计划 (Curators Program) 启动**：Perplexity 团队宣布了 **策展人计划 (Curators Program)**，旨在为 **Discover Feed** 创作引人入胜的内容。感兴趣的个人可以点击[此处](https://perplexity.ai/curators)申请或标记好友参与该计划。
  
  - 该计划邀请喜欢创建 **Pinterest 看板**、编辑 **Wikipedia 页面**以及深入研究 **YouTube 视频论文**的用户来启发全球观众。
- **MacOS 应用可用性评价褒贬不一**：用户报告了 Perplexity MacOS 应用的问题，提到崩溃和有问题的弹窗影响了性能。一些用户强调了与网页版相比，在**复制粘贴**图像方面的限制。
  
  - 由于缺乏足够的反馈渠道，用户的不满情绪日益增加，表明迫切需要改进可用性。
- **新功能引发辩论**：Perplexity 引入的购物功能在用户中引起了复杂的情绪，用户要求将这些功能更加模块化。关于这些发展的战略影响的推测仍在继续。
  
  - 用户表示渴望看到这些功能将如何影响他们与平台的日常互动。
- **对下一代 AI 模型的期待**：闲聊指出 **GPT-5** 可能在 2024 年 12 月发布，AI 开发者之间的竞争动态正在演变。Meta 建立自己搜索引擎的举动增加了这一竞争态势。
  
  - 用户对未来几个月这些进展将如何塑造功能感到好奇。
- **关于 Perplexity API 访问的澄清**：成员们讨论了如何为 API 结果**获取来源**，并链接到一条 [Discord 消息](https://discord.com/channels/1047197230748151888/1161802929053909012/1286596437768933376)以寻求帮助。用户正寻求复制与标准 Perplexity 聊天中类似的结果。
  
  - 用户对**引用功能封闭测试 (citations closed beta) 权限**表示担忧，一名用户强调需要就其申请状态进行更好的沟通。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **超声波设备销量激增**：一名开发者用于驱鼠的超声波设备在目标人群中的销量从 **15% 跃升至 28%**，这得益于逻辑增长曲线分析。
  
  - 通过适当重新定义时间变量，澄清了逻辑模型中 A 值和 B 值的混淆。
- **AI 蒸馏 (Distillation) 技术引发辩论**：围绕蒸馏的对话强调了 Arcee 的 **Llama-3.1** 模型，该模型利用来自大型框架的 Logits 高效训练较小的模型。
  
  - 用户对 Meta 缺乏足够的技术文档表示担忧，引发了对其训练方法的深入讨论。
- **Hermes 3 数据集保持闭源**：成员确认 **Hermes 3 SFT 数据集** **不是开源的**，这与其前身 Hermes 1 和 2 不同。
  
  - 尽管如此，还是提供了 [OpenHermes-2.5 数据集](https://huggingface.co/datasets/teknium/OpenHermes-2.5)的链接作为资源。
- **思维偏好优化 (TPO) 提升性能**：论文《Thinking LLMs》指出，**思维偏好优化 (Thought Preference Optimization, TPO)** 可以增强 LLM 的指令遵循能力，带来 **4%** 的性能提升。
  
  - 在 **Llama 3 8B Instruct** 模型上实施 TPO 显示，不恰当的提示词可能会降低性能。
- **苹果推出 Ferret-UI 以进行 iOS 集成**：苹果推出了 **Ferret-UI**，这是一款专为 **iPhone/iOS** 优化使用的多模态 LLM，在与 [Hugging Face Transformers](https://huggingface.co) 集成时可增强用户体验。
  
  - Ferret-UI 展示了令人印象深刻的移动 UI 理解能力，在图标识别和文本定位方面甚至超越了 **GPT-4V**。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **在有限资源下训练 LLM**：成员们讨论了在有限 GPU 资源上训练像 **LLaMA-2** 这样的大型语言模型的挑战，指出有效复现需要大量的硬件需求。
  
  - 建议将 *部署 nanoGPT* 作为初学者寻求更简单训练方式的轻量级模型。
- **开源 AI 项目的贡献受到关注**：一位用户表达了尽管主要拥有专有经验，但仍有兴趣参与 **EleutherAI** 的项目，强调开源贡献是宝贵的学习经历。
  
  - 回复强调，较小的项目可以提供重要的见解，并为软件工程师提供转型机会。
- **Stick-Breaking Attention 机制设计讨论**：如最近的一篇 [arXiv 论文](https://arxiv.org/abs/2410.17980)所述，一种新颖的 **stick-breaking attention** 机制通过解决位置嵌入和 **softmax** 限制，为 **Transformer** 模型提供了改进。
  
  - 社区反馈强调需要对此类机制进行更清晰的介绍，并提到了 [IBM 的 ModuleFormer](https://github.com/IBM/ModuleFormer) 等相关项目。
- **Python 3.10 兼容性问题解决**：在 **Python 3.10** 环境下设置 **GPT-NeoX** 需要将 **Torch** 版本覆盖为 `1.11.0` 以解决导入失败问题，用户在一个特定的 **Colab** 笔记本中记录了安装修复方案。
  
  - 警告提到了关于 **Torch** 版本的兼容性问题，指出 **torch 2.4** 会导致失败，而 **2.3** 可能是可行的。
- **分布式 GPU 训练的挑战探讨**：在为 **GPT-NeoX** 训练共享消费级 **GPU** 时，出现了对网络困难的担忧，促使用户建议查看 **INTELLECT-1** 以了解去中心化努力。
  
  - [PrimeIntellect](https://app.primeintellect.ai/intelligence) 分享的一个正在进行的工作链接强调了一项贡献计算资源的倡议。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3.5 反响不一**：用户报告了对 **Stable Diffusion 3.5** 的**褒贬不一的体验**，质疑其与 **1.5** 相比的**速度**和**质量**。建议在不同模型上运行相同的 prompt 以有效比较结果。
  
  - 一些成员分享了[这份指南](https://sandner.art/stable-diffusion-35-large-what-you-need-to-know/)，旨在最大限度地提高新模型的性能。
- **在 Runpod 上部署 Juggernaut 受到关注**：一位用户探索了在 **Runpod** 上部署名为 **Juggernaut** 的自定义模型，并注意到缺少 **Forge** 模板。其他人强调使用 **Auto1111** 可以提供更用户友好的方法。
  
  - 这一讨论指向了对自定义模型部署需要更清晰资源的需求。
- **AMD GPUs 在本地生成方面展现潜力**：社区讨论了使用 **AMD GPUs** 的**本地生成**能力，鼓励遵循置顶指南以获得最佳性能。用户分享了关于 **VRAM** 限制和模型测试的见解，特别提到了 **Gemma 2**。
  
  - 重点放在了尝试各种模型以找到最适合 **AMD** 配置的模型上。
- **建筑设计中的草图转渲染工作流**：利用 **Stable Diffusion** 进行针对**建筑设计**定制的“草图转渲染”流程的兴趣日益浓厚。成员建议利用 **ControlNet** 等工具来增强细节和准确性。
  
  - 这种方法旨在改进从简单草图到高保真渲染的转换。
- **用于 Flux Inpainting 开发的 Discord 机器人**：开发者们集思广益，计划创建一个 **Discord 机器人**以促进 **Flux** 中的 **inpainting**，并注意到该用例的模型可用性有限。一位参与者表现出为社区工具实现功能性 **inpainting** 特性的热忱。
  
  - 这次对话反映了将高级图像处理直接集成到社区平台中的兴趣日益增长。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 与 PearAI 功能对决**：成员们强调了 **PearAI** 与 **Aider** 之间的重叠，特别是在与开源工具的集成能力方面，引发了对功能复制的伦理担忧。
  
  - 他们引用了 [*Open Source Pledge*](https://www.theregister.com/2024/10/25/open_source_funding_ads/)，强调科技公司需要为开源开发贡献更多力量。
- **Claude 1022 提升生产力**：一位用户报告了结合使用 **Claude 1022** 和 **Aider** 开发 Flutter 应用程序的高效体验，生成 **4300 行代码** 仅花费了 **$18** 的额度。
  
  - 他们指出在该项目上花费了 **15 小时**，展示了通过有效的 Prompting 带来的显著生产力提升。
- **Nvidia Nemotron 设置故障排除**：用户在 Aider 中配置 **Nvidia Nemotron** 时面临挑战，特别是围绕自定义模型元数据设置和 exec 命令。
  
  - 一位成员鼓励在连接时忽略模型警告，并建议查阅 [故障排除指南](https://aider.chat/docs/troubleshooting/edit-errors.html) 以获取指导。
- **Sonnet 3.5 基准测试：请求文件**：用户表示需要 **Sonnet 3.5** 的基准测试数据文件，特别是关于代码编辑和重构的文件，以帮助避免昂贵的测试。
  
  - 有人专门请求获取 `.aider.chat.history.md` 和 `.aider.results.json` 文件进行实证评估。
- **Aider 中本地模型的隐私问题**：在 **Aider** 中使用本地模型时出现了数据隐私方面的担忧，特别是关于敏感信息的处理。
  
  - 用户得到保证，Aider 在使用本地模型时不会存储用户数据，从而维护隐私。

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo API 文档需要示例**：讨论强调了 Mojo API 文档中 **缺乏 Collections（集合）示例**，导致有人建议通过 [GitHub](https://github.com/modularml/mojo/tree/main) 为文档做贡献。
  
  - 成员们强调了社区参与的重要性，并将 **准备 Pull Requests** 作为改进文档的一步。
- **学习 Mojo 还是 C++**：一位正在考虑学习 Mojo 还是 C++ 的用户收到的建议是，Mojo 作为一种现代系统语言，可能更适合他们的探索，特别是在 **ML 和数据科学** 领域。
  
  - 社区成员分享了关于语言选择的见解，建议关注 **Rust 或在 Mojo 中构建库**。
- **可变张量将增强训练对象**：当前的 Nightly 版本正在引入 **可变张量 (Mutable Tensors)**，从而能够表示训练对象，如 **训练权重** 和 **KVCaches**。
  
  - 从 API 的角度来看，该功能仍在开发中，但预计将包含在 **下一个版本** 中。

 

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **高性能混合精度计算准备就绪**：一场关于 **高性能混合精度计算** 的即将举行的演讲在社区内引起了兴奋，该演讲计划于近期举行。
  
  - 成员们反应积极，表明对性能优化策略有浓厚兴趣。
- **H100 与 CUDA Profiling 的挑战**：用户讨论了在 **H100** 上进行 CUDA Profiling 时遇到的 “Command Buffer Full” 错误，这一问题在 **A100** 上并未出现。
  
  - 成员们正在寻求处理 CUDA 限制的建议，以及是否需要探索其他渠道来寻求解决方案。
- **FLUX 与 LLM.int8 重构**：关于 **Sayak** 在 Twitter 上的发现有了新的见解，指出 **FLUX** 的性能有所提高，引发了对 **LLM.int8 重构** 的好奇。
  
  - 协作讨论集中在优化模型和解锁更好的功能上。
- **聚焦 NVIDIA GPU 的多伦多聚会**：关于 **NVIDIA GPU** 和 **CUDA 编程** 的多伦多聚会计划正在制定中，第一场会议定于 **11 月 15 日**。
  
  - 组织者正号召演讲者为该活动做出贡献，旨在加强 AI 专业人士之间的协作。
- **解决复合 CUDA 问题**：一位用户分享了他们排除由最近 Ubuntu 更新引起的 CUDA 安装问题的波折经历，确认在升级到 CUDA **12.4** 后获得成功。
  
  - 这一混乱引发了幽默的反思，强调了开发者在建立稳定环境时面临的典型障碍。

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 中的 Connector 查询遇到障碍**：用户在通过 Cohere Connector 检索数据时遇到问题，在查询特定用户 ID 时收到诸如 *'I was unable to find any information'* 的消息。
  
  - 建议就这些问题联系 [support@cohere.com](mailto:support@cohere.com) 以获取帮助。
- **Playground 出现延迟**：讨论强调了 Cohere Playground 中持续存在的延迟问题，尤其是在多条消息之后，这影响了用户体验。
  
  - 建议通过 *开启新对话* 或清除缓存作为潜在的解决方法，这可能与设备限制和上下文过载有关。
- **算法交易讨论的点滴**：成员们交流了关于算法交易的见解，重点关注 AI 情绪对市场波动的影响以及媒体偏见的细微差别。
  
  - 有人指出，重要的交易见解最好来源于 EDGAR 等平台，而不是人类视角。
- **访问 Cohere 社区服务器**：关于加入 **Cohere For AI** 社区服务器的咨询引导分享了 [申请页面](https://cohere.com/research)。
  
  - 同时提供了关于一个旨在解决复杂机器学习挑战的研究实验室的信息。
- **配置 Cohere Connectors**：用户寻求关于为 Cohere chat 端点使用 Connector 的指导，促使分享了必要的 [文档](https://docs.cohere.com/v1/docs/creating-and-deploying-a-connector)。
  
  - 对于 Connector 设置，务必使用 v1 API，因为目前尚不支持 v2。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **探索 AI 研究资助经验**：一名成员询问了申请 **AI 研究** 资助的经验，反映出社区对资金机会的兴趣日益增长。
  
  - 这一交流凸显了获取资源以推进 AI 项目的多样化途径。
- **AI 定制化的挑战**：关于 **ChatGPT** 经常忽略定制化指令并导致不可预测输出的问题引起了关注。
  
  - 参与者分享了指令未被遵循的案例，引发了对 AI 推理能力的质疑。
- **了解 LLM 的局限性**：有人指出 **LLM 擅长语言生成但在数学方面表现不佳**，因此建议使用 Python 进行计算。
  
  - 一名成员强调了提供逐步引导（step-wise guidance）对提升 LLM 功能的重要性。
- **在 AI 解决方案中使用多个 LLM**：讨论强调了使用多个 **LLM** 来有效处理不同任务的必要性，因为单一模型可能不足以胜任。
  
  - 参与者探索了 “prompt chaining” 和 **agentic workflows** 对增强结果的好处。
- **AI 的一致性是一个神话**：成员们指出 **AI 并不具有一致性**，认为不可预测性是用户面临的一个根本挑战。
  
  - 参与 AI 任务被认为既复杂又有趣，呈现出兴奋与复杂性的交织。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 与 Google 的 12 月大决战**：OpenAI 计划在 12 月发布其下一个 AI 模型，而 Google 也在致力于发布 **Gemini 2.0**，加剧了 AI 领域的竞争。虽然 OpenAI 的推出是分阶段的，但 Google 寻求广泛发布，尽管**性能预期**可能无法完全达到。
  
  - 12 月正演变为一场 AI 发布会的对决之月，这使得工程师们及时了解这些进展变得至关重要。
- **Meta 构建自己的搜索引擎**：Meta 正在工程经理 **Xueyuan Su** 的领导下开发一种新的网络搜索引擎，以减少对 Google 和 Bing 数据源的依赖。该项目旨在为 Meta 的平台提供更独立的 AI 解决方案，避免再次出现**类似苹果公司的局面**。
  
  - 这一转变反映了 Meta 增强对其信息生态系统控制的战略，可能会影响数据获取实践。
- **生成式 AI 的采用速度缓慢**：最近的一篇论文称，虽然 **40% 的美国成年人**在使用生成式 AI，但实际上只有 **0.5% – 3.5%** 的工作时间涉及其辅助。采用率远低于预期，揭示了使用情况与预期的生产力影响之间的差距。
  
  - 这引发了关于如何改进工作流中的 AI 集成以实现效率最大化的问题。
- **对 Gemini 发布版本的担忧**：**Gemini 模型**的发布因性能较之前版本下降以及针对消费者的营销问题而面临批评。这次发布被认为是**最糟糕的发布之一**，严重的退化影响了用户体验。
  
  - 用户体验的转变引发了人们对高风险 AI 环境中产品开发遗留问题的担忧。
- **人工生成示例定价查询**：一位成员询问了关于**人工生成示例的价格**与将其标注为好或坏的价格信息。这个问题突显了在**手动与自动化**标注过程的价值主张中需要明确性。
  
  - 随着 AI 系统的持续激增，建立评估生成示例的明确标准至关重要。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **快速数学模式引发讨论**：成员们强调了 Metal 中的 **fast math mode** 如何自动执行代数变换，为了严格遵守浮点标准需要手动禁用。`-fassociative-math` 的使用被提及作为数学表达式的一种**优化**。
  
  - *Reassociation* 被引用为在数学设置中值得探索的潜在增强功能。
- **Tinygrad 在复数集成方面进展缓慢**：用户报告了 Tinygrad 中复数的问题，特别是在创建 DFT 时，由于支持不足而遇到 `AssertionError`。George 表达了对更简便的复数处理方式的渴望，建议使用 2D 轴进行潜在的模拟。
  
  - 对于旨在实现高级算法的用户来说，**复数支持**的需求至关重要。
- **Tinygrad 通过 OpenCL 在 Android 上推出**：一位用户询问在带有 OpenCL 的 Android 设备上使用 Tinygrad 进行模型编译，寻求设置指导。`compile_efficientnet.py` 等资源被分享作为建立必要的 OpenCL 内核和缓冲区的潜在路径。
  
  - 成员们强调，不依赖 Python 运行模型的能力是移动应用的一个显著优势。
- **需遵循严格的 PR 提交指南**：George Hotz 强调了在提交新 PR 之前审查现有 PR 的重要性，并指出理解不足的更改可能会面临拒绝。他敦促贡献者优先考虑 **bug 修复**，而不是重复提交包含类似信息的 PR。
  
  - 这种方法确保了 Tinygrad 开发过程的完整性和有意义的贡献。
- **Tinygrad 生态系统开发初具规模**：George 讨论了 Tinygrad 生态系统的演变，暗示将转向性能增强和更广泛的实现。社区对开发类似于 HuggingFace 提供的模型转换工具表现出兴趣，以简化**模型管理**。
  
  - 对话集中在这些工具随着 Tinygrad 成熟而变得重要，强化了对易用性和兼容性的关注。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **在 Ray Summit 探索智能知识助手**：Ray Summit 工作坊展示了构建**智能知识助手**的愿景，这些助手能够以多种方式处理复杂数据，视频现已上传至 [YouTube](https://t.co/9eOQf5pYYi)。
  
  - 会议讨论了*超越简单任务所需的所有组件*，相关内容可以在[这里](https://t.co/ETH05Y8JNG)找到。
- **需要 NVIDIA 案例研究 Cookbook**：多位成员对 [NVIDIA 案例研究](https://github.com/Chainlit/cookbook/pull/138)的 Cookbook 表示感兴趣，特别是关注使用 Chainlit 的流式传输（streaming）用例。
  
  - 一位成员强调，在追求自定义 Agent 工作流时，在 Chainlit 框架内嵌套父/子步骤遇到了困难。
- **掌握处理 500 张表的 Text-to-SQL**：一个可靠的 Text-to-SQL 教程演示了如何构建一个能够在 **500 张表**上运行的 SQL Agent，视频可在 [YouTube](https://t.co/nfRvlw3idI) 观看。
  
  - 该资源在处理复杂数据设置方面表现出色，更多信息可在此处[获取](https://t.co/Fakrwdh0Op)。
- **令人印象深刻的 Deepfake 语音生成**：一位用户体验了**令人印象深刻的 Deepfake 语音生成**，系统在 Teams Tier 方案中能够像用户本人一样自动预测并回复。
  
  - AI 不仅能用自己的声音提问，还能用用户的声音回答，展示了**现实生活中的自动预测能力**。
- **Retriever 问题报告**：一位成员报告了 Retriever 返回空节点的问题，尽管使用 Chat Engine 测试 Index 时是成功的。
  
  - 另一位成员建议分享代码以进一步排查故障，因为 Retriever 的配置似乎设置不当。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **使用 MIPROv2 进行自动提示词生成**：一位成员分享了在 gsm8k 数据集上使用 MIPROv2 优化器实现**自动提示词生成**技术的讨论串，该技术被结构化为演示、指令和输出三个清晰的模块。
  
  - 这种流线型的方法增强了提示词构建过程，详见这篇[推文](https://x.com/karthikkalyan90/status/1849902254373077196)。
- **瑞士公民协作制定法律**：一位成员正在开发一个**协作软件应用**，使瑞士公民能够通过全民动议流程直接参与立法，这是一个具有个人学术价值的主题。
  
  - 该项目展示了在公民参与方面的重大投入，并与关于参与式民主的更广泛讨论相关联。
- **DSPy 2.5 映射说明**：关于向 **DSPy 2.5** 过渡的讨论出现，成员们参考了[迁移文档](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb)以了解实现上的变化。
  
  - 预计*不会有重大差异*，这表明现有用户可以平滑过渡。
- **音频输入功能的开发**：成员们探索了 DSPy 中**音频输入功能**的持续开发，引用了一个潜在的 [GitHub Pull Request](https://github.com/stanfordnlp/dspy/tree/mm-adapter)，其中讨论了支持像 Ultravox 结合 LLaMa 这样的架构。
  
  - 这种集成可能会提升多模态能力，将 DSPy 转向更广泛的应用领域。
- **NER 和关系抽取的示例**：一位成员提供了 DSPy 中**命名实体识别 (NER)** 的代码片段，强调现代的 `dspy.ChainOfThought` 实现优于已弃用的方法。
  
  - 讨论还关注了关系抽取，并建议利用 Hugging Face 的相关数据集来增强项目洞察力。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 处理电子表格的性能**：在关于提升 **Open Interpreter** 性能的讨论中，用户尝试了来自 [YouTube 教程](https://www.youtube.com/watch?v=4X4rKqtmxJg) 的见解，但发现本地模型如 **qwen2.5:32b-instruct** 在执行任务时表现非常吃力。
  
  - 一位成员建议，提升性能的关键在于使用高质量的 **models** 和有效的 prompting 技巧，甚至建议为任务澄清创建一个 profile。
- **Open Interpreter 设置指南**：初学者在通过 Windows 终端设置 **Open Interpreter** 时面临挑战，促使另一位成员分享了包含 pip 安装命令的 [设置说明](https://docs.openinterpreter.com/getting-started/setup)。
  
  - 这一简化的设置指南旨在为开始使用该工具的新用户提供更轻松的入门体验。
- **Open Interpreter 中本地模型的限制**：关于本地模型视觉能力需求的咨询表明，目前没有本地模型能匹配 **Sonnet** 的性能，这削弱了本地操作的效果。
  
  - 一位精通技术的成员强调了正确导入 **computer API** 以使本地模型有效运行的重要性。
- **对 Markdown 与 Obsidian 的热爱**：成员们表达了对 **Markdown** 的热情，其中一人暗示即将发布与 **Obsidian** 工具相关的精彩演示，定会让人印象深刻。
  
  - 这反映了在 AI 编程环境中实施 Markdown 实践的日益增长的热情，推动了创造性的利用。
- **OpenAI 推出高级语音功能**：OpenAI 的公告显示，**Advanced Voice** 现已向欧盟（包括 **Switzerland** 和 **Norway**）的免费用户开放，增强了其移动应用的功能。
  
  - 这一可访问性里程碑标志着向更广泛用户群体民主化高级 AI 功能迈出了重要一步。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **对 Discord LLM 助手的需求**：成员们表达了对 **'Discord LLM helper'** 的强烈渴望，用于按需总结聊天内容和回答问题，同时指出了 Discord 当前 **beta feature** 的局限性。
  
  - *这是一个错失的机会*，特别是考虑到提供 **ephemeral responses**（临时响应）可以通过保持用户特定性来简化交互。
- **具有临时响应功能的自定义 Bot**：社区对开发一个能够利用 **ephemeral responses** 高效处理问答和总结的 **custom Discord bot** 表现出兴趣。
  
  - 这种方法可以通过使响应仅对执行命令的用户可见，从而显著提高聊天交互的清晰度。
- **Minecraft LLM 项目**：围绕将 **Minecraft** 与 **LLMs** 集成的积极讨论激发了社区对创意项目的热情。
  
  - 参与者评论说，这些结合项目不仅有趣，而且在实现上也提出了独特的挑战。
- **关于 Llama3-8B-1.58 模型的澄清**：讨论澄清了 **Llama3-8B-1.58** 模型的血统，指出它源自 **Llama-3-8B-Instruct**，而非之前假设的 **BitNet**。
  
  - 成员们引用了一篇关于 [extreme quantization](https://huggingface.co/blog/1_58_llm_extreme_quantization)（极端量化）的博客以获取更多细节和指导。
- **关于模型规格的困惑**：围绕 **Llama3-8B-1.58** 的模型规格进行了澄清，特别是关于它是一个 **100B** 模型的误解。
  
  - 成员们承认了这一误解，并一致认为在模型描述中需要对 **8B parameters** 进行更好的沟通。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mixtral AI 模型已过时**：一位成员幽默地建议将 **Mixtral AI** 模型升级到更新版本的 **MistralAI**，暗示其已过时。
  
  - *至少升级到更新的 MistralAI 模型。*\*
- **关于 SymNoise 实现代码的查询**：成员们讨论了关于 **SymNoise** fine-tuning 技术的代码实现需求，该技术利用对称噪声来增强语言模型。
  
  - *尝试过自己实现，但它似乎通过 concatenation 使 embeddings 的 batch size 翻倍了，我不知道该如何处理。*
- **SAT 阅读测试抓取不完整被揭露**：一位成员报告了 **SAT 阅读测试** 和几项 **AP 测试** 的 **抓取不完整**，引发了关于格式的讨论。
  
  - *感谢您让我注意到这一点！* 对关于抓取的反馈表示感谢。
- **关于包含多模态问题的担忧**：在观察到 SAT 数据集中的 `formatted_prompt` 和 `rewritten_answer` 字段后，成员们提出了关于问题是否应附带图像的疑问。
  
  - 原始抓取者确认，虽然 **完整集合确实包含** 某些问题的图像，但该数据集旨在保持单模态（unimodal）。
- **需要澄清 Qwen 模型配置**：在详细讨论中，成员们强调了为 **Qwen/Qwen2.5-32B** 指定确切模型类型的必要性，而不是使用像 `AutoModel` 这样的通用占位符。
  
  - 成员们还对与 `trust_remote_code` 设置相关的潜在安全问题表示了担忧。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **使用 HuggingFace 本地模型创建 ReAct Agent**：一位成员目前正在使用本地模型初始化 **ReAct Agent**，并在调用过程中遇到了 **parserException**。
  
  - 他们正在寻求帮助，因为在网上找不到针对该特定错误的解决方案。
- **探索高级 RAG 方法**：出现了关于 **Retrieval-Augmented Generation (RAG)** 最先进技术以及传统方法相关性的问题。
  
  - 提到的常见做法包括数据清洗和在 **Pinecone/vector databases** 中的存储，同时成员们也在寻找最近的参考资料。
- **使用 create_sql_agent 返回 Pandas DataFrame**：有人询问如何利用 **create_sql_agent** 生成 **Pandas DataFrame** 而不仅仅是文本字符串。
  
  - 该成员特别询问了在这种情况下 **SQLDatabaseToolkit** 的必要性。
- **介绍 AdaletGPT - 一个土耳其法律聊天机器人**：**AdaletGPT** 是一款基于 RAG 的 **土耳其法律聊天机器人**，使用 LangChain、Pinecone 和 OpenAI 构建，用于法律援助。
  
  - 该平台允许用户进行由 AI 驱动的法律咨询互动。
- **bootstrap-rag v0.0.11 发布并带来令人兴奋的更新**：新版本 **bootstrap-rag** v0.0.11 整合了 **LLM as Judge 模板**，并包含来自 Arize AI Phoenix 的增强功能。
  
  - 此次更新包括关键的错误修复和改进的文档，以提供更流畅的用户体验。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **第 8 讲今天开课！**：**第 8 讲** 于今天 **PST 时间下午 3:00** 开始，可通过[此处](https://www.youtube.com/live/wm9-7VBpdEo)观看直播，承诺涵盖 **LLM Agents** 的核心方面。
  
  - 与会者期待客座演讲者 **Yuandong Tian** 的见解，他将探讨 **神经与符号决策框架（neural and symbolic decision-making frameworks）** 的融合。
- **学习小组讨论热度持续高涨**：参与者热衷于组建 **学习小组** 进行协作讨论，并分享了一个 [Google 表单](https://forms.gle/QtQ2C6qzomeHDrC38) 用于收集时间安排偏好。
  
  - 响应非常积极，表明晚加入的成员也渴望一起剖析课程内容，参与度有所提高。
- **黑客松时间表发布**：即将举行的黑客松详情（包括 **Applications**、**Benchmarks** 和 **Safety** 等不同赛道）现已在 [黑客松网站](https://rdi.berkeley.edu/llm-agents-hackathon/) 上发布。
  
  - 该活动由 **Berkeley RDI** 主办，旨在汇聚多元人才以增强 **LLM agent 技术** 领域。
- **数据集讨论仍在继续**：成员们正在寻求适用于黑客松 **benchmarking 赛道** 的合适 **数据集** 指导，引发了关于资源的开放式对话。
  
  - 尽管大家很有兴趣，但尚未分享具体的数据集资源，表明需要进一步的探索和协作。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Embedding 配置标志提案**：一位成员提议在配置中公开**两个布尔标志**（`embedding_trainable=False` 和 `norms_trainable=False`），以减轻未来的配置问题，因为 **TransformerDecoder** 可能需要更重大的更改。
  
  - *这种方法旨在简化*从布尔标志到列表的过渡，防止大量的配置适配。
- **LoRA Bug 修复已提交**：通过 [pull request #1909](https://github.com/pytorch/torchtune/pull/1909) 提交了 **LoRA** bug 的修复，解决了在 `use_dora=True` 进行单设备微调时的 NaN 损失问题。
  
  - *然而，该修复在所有 recipe 中的兼容性仍存在不确定性*，特别是在分布式设置中。
- **超参数优化 Recipe 讨论**：一个 GitHub [issue](https://github.com/pytorch/torchtune/issues/1752) 提议了一个超参数优化 recipe，允许用户输入配置、数据集和参数，以对常用默认值进行 sweep。
  
  - *有趣的是，目前还没有人明确要求*这个功能，这表明用户需求可能存在缺口。
- **对 muP 实用性的质疑**：成员们质疑了 **muP** 在微调中的实用性，指出其主要提及点与预训练相关，而改进生成和早停（early stopping）应具有更高优先级。
  
  - *人们仍然担心*，相比解决现有问题，实施 muP 是否值得投入。
- **优先处理开发问题**：一位成员强调了积压的 **200 个待处理 issue** 过多，强调迫切需要解决更快的强化学习生成和改进的 LLM 分类问题。
  
  - *此外，对分布式 Shampoo 的支持*也被列为另一个高优先级项目。

 

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Human Native AI 市场上线**：新的 [Human Native AI Marketplace](https://www.humannative.ai/) 允许创作者授权其内容用于 AI 训练并获得报酬。
  
  - 联合创始人 **James Smith** 将在即将举行的 Mozilla **Data Futures Lab Speaker Series** 中讨论进展。
- **精彩的 11 月成员活动已排定**：11 月将举办一系列由成员组织的活动，包括关于 Sqlite-Vec 和 Refact.ai 的会议，以及远程会议和旧金山见面会。
  
  - 成员应 [RSVP](https://discord.com/channels/1089876418936180786/1254956456772505601) 以加入重要的讨论。
- **开源项目展示**：Mozilla AI 重点展示的项目包括 **Open Interpreter**、**Homebrew** 和 **Sentry.io 的开源自动修复 (auto fix)**。
  
  - 期待在 **Public AI** 上展示来自 **3300 名成员**社区的更多项目。
- **OSS4AI 见面会汇聚本地成员**：即将举行的 [OSS4AI 旧金山线下见面会 (IRL Meetup)](https://discord.com/events/1089876418936180786/1299558554872582154) 邀请成员进行交流与协作。
  
  - 这是本地爱好者参与有意义的项目讨论的绝佳机会。
- **Sqlite-Vec 元数据过滤技术讨论**：关于 [Sqlite-Vec 中的元数据过滤](https://discord.events/1089876418936180786/1300483739872399411) 的活动将探讨高效数据管理的关键策略。
  
  - 该倡议强调在支持 AI 训练的同时保持数据完整性。

 

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **关于排行榜“Multiple”功能的澄清**：用户询问了排行榜语境下“multiple”的含义，怀疑它表示从多个可用选项中选择合适函数的能力。
  
  - 有人指出，虽然这一方面很明确，但对真正的多步（multi-step）功能的评估仍然**不明确**。
- **Function Call 排行榜的 GitHub 参考**：分享了一个 [GitHub 链接](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d9f323715d7d2012a955d7f4114e/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json#L42C1-L42C2438) 作为与 Gorilla 项目相关的示例，该项目旨在训练和评估用于 Function Call 的 LLM。
  
  - 引用页面为理解排行榜的运行机制提供了关键背景。

---

**Alignment Lab AI Discord** 没有新消息。如果该服务器（guild）长时间没有活动，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该服务器（guild）长时间没有活动，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器（guild）长时间没有活动，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该服务器（guild）长时间没有活动，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器（guild）长时间没有活动，请告知我们，我们将将其移除。

---

# 第二部分：分频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1299453642847289374) (1071 messages🔥🔥🔥):

> - `Hugging Face Spaces`
> - `Fine-Tuning Models`
> - `Image Generation Models`
> - `NSFW Content Discussion`
> - `Model Quantization`

- **探索 Hugging Face Spaces 和模型**：用户讨论了 Hugging Face Spaces 上的各种可用模型，包括用于图像生成和 Quantization 的模型，并分享了特定模型的链接。
  
  - 针对本地项目的轻量级模型建议包括 Llama 和 Flux，并深入探讨了特定功能。
- **关于 NSFW 内容的争议**：服务器上引发了关于讨论 NSFW 内容是否恰当的辩论，社区成员强调了服务器的 SFW 标准。
  
  - Cakiki 提醒大家遵守行为准则，并建议在指定的线程中继续讨论。
- **关于 AI 训练的技术讨论**：用户探讨了在本地 GPU 上训练模型，并就通过调整样本大小来优化性能和减少 Overfitting 提出了建议。
  
  - 讨论强调了理解模型参数和 Quantization 技术的重要性。
- **分享教育资源**：社区成员分享了初学者友好的资源，包括关于构建 Hugging Face Spaces 的视频和关于模型 Quantization 的课程。
  
  - 对话包括对 AI 和机器学习实际应用的建议，鼓励新用户探索可用工具。
- **一般 AI 社区互动**：参与者就 AI 生成的内容进行了轻松的闲聊，并对各种 AI 项目表示了兴趣。
  
  - 聊天凸显了社区的协作精神以及帮助新手解决疑问的意愿。

**提到的链接**：

- [Here, Let Me Google That For You](https://googlethatforyou.com/?q=github%20repo%20machine%20learning%20ai%20video%20generators%20with%203d%20models%20as%20the%20bones%2C%20for%20adding%20camera%20controls%2C%20consistent%20environments%2C%20and%20and%20unreal%20engine%20or%20similar%20basic%20physics)：以一种“消极攻击”的方式教你的朋友如何使用 Google。适用于那些觉得问你比自己搜索更方便的人。与 Google 无关。
- [Rayleigh quotient - Wikipedia](https://en.wikipedia.org/wiki/Rayleigh_quotient)：未找到描述。
- [Code of Conduct – Hugging Face](https://huggingface.co/code-of-conduct)：未找到描述。
- [Here, Let Me Google That For You](https://tinyurl.com/575bhwa2)：以一种“消极攻击”的方式教你的朋友如何使用 Google。适用于那些觉得问你比自己搜索更方便的人。与 Google 无关。
- [Intel's Hala Point, the world's largest neuromorphic computer, has 1.15 billion neurons](https://www.zdnet.com/article/intels-hala-point-the-worlds-largest-neuromorphic-computer-has-1-15-billion-neurons/)：Intel 表示，Hala Point 机器在某些任务上的计算效率可与 GPU 和 CPU 媲美。
- [Streamlit](https://kaliumpotas-api-recommend.hf.space/main/ai)：未找到描述。

- [@Tonic 在 Hugging Face 上："老古董们还在选 zenodo.org 而不是 huggingface ？？？简直太滑稽了……"](https://huggingface.co/posts/Tonic/759782209325435)：未找到描述
- [访问 AI 推荐的 Api - KaliumPotas 的 Hugging Face Space](https://huggingface.co/spaces/KaliumPotas/Api_recommend)：未找到描述
- [Tonic/open-gpt-Flirty-Friend · Hhh](https://huggingface.co/spaces/Tonic/open-gpt-Flirty-Friend/discussions/1)：未找到描述
- [FlyWire](https://flywire.ai/)：未找到描述
- [贝吉塔 龙珠超 GIF - 贝吉塔 龙珠超 龙珠 - 发现并分享 GIF](https://tenor.com/view/vegeta-dragonballsuper-dragonball-meme-language-of-gods-gif-14707991)：点击查看 GIF
- [city96/stable-diffusion-3.5-large-gguf · Hugging Face](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf)：未找到描述
- [模型 - Hugging Face](https://huggingface.co/models?other=base_model:quantized:stabilityai/stable-diffusion-3.5-large)：未找到描述
- [量化 (Quantization)](https://huggingface.co/docs/transformers/v4.46.0/quantization/overview)：未找到描述
- [压缩张量 (Compressed Tensors)](https://huggingface.co/docs/transformers/v4.46.0/quantization/compressed_tensors)：未找到描述
- [我看到了 W Gus Fring GIF - 我看到了 W Gus Fring Gus - 发现并分享 GIF](https://tenor.com/view/i-saw-w-gus-fring-gus-gustavo-deleted-gif-25440636)：点击查看 GIF
- [莱昂纳多·迪卡普里奥 Rick Dalton GIF - 莱昂纳多·迪卡普里奥 Rick Dalton 指手指 - 发现并分享 GIF](https://tenor.com/view/leonardo-dicaprio-rick-dalton-point-finger-pointing-hand-right-there-gif-20349799)：点击查看 GIF
- [孤独故障 GIF - 孤独故障 电影 - 发现并分享 GIF](https://tenor.com/view/alone-glitch-film-eloresnorwood-heartbreak-gif-15491348)：点击查看 GIF
- [嵌入 No GIF - 嵌入 No No 嵌入 - 发现并分享 GIF](https://tenor.com/view/embed-no-no-embed-megamind-megamind-meme-gif-25261934)：点击查看 GIF
- [calcuis/sd3.5-large-gguf · Hugging Face](https://huggingface.co/calcuis/sd3.5-large-gguf)：未找到描述
- [Reddit - 深入探索一切](https://www.reddit.com/r/Portuguese/comments/xfmku0/comment/ionuodx/)：未找到描述
- [模型 - Hugging Face](https://huggingface.co/models?other=base_model:quantized:stabilityai/stable-diffusion-3-medium)：未找到描述
- [让我们构建 GPT：从零开始，在代码中详细阐述。](https://www.youtube.com/watch?v=kCc8FmEb1nY)：我们按照论文《Attention is All You Need》以及 OpenAI 的 GPT-2 / GPT-3 构建了一个生成式预训练 Transformer (GPT)。我们讨论了与……的联系
- [meta-llama/Llama-3.1-8B · Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B)：未找到描述
- [Allegro：新的开源 SOTA 文本转视频模型 - 27 个带有提示词的惊人示例，Apache 许可证](https://www.youtube.com/watch?v=0tsLqNXQ5Mk)：Allegro 是一款强大的文本转视频模型，可以根据简单的文本输入生成长达 6 秒、15 FPS 和 720p 分辨率的高质量视频。目前……
- [如何创建 Hugging Face Space：初学者指南](https://www.youtube.com/watch?v=xqdTFyRdtjQ)：如果你是 Hugging Face 的新手，并想为你的机器学习模型或应用设置一个 Space，那么你来对地方了。按照这些简单的步骤来……
- [但什么是神经网络？ | 第 1 章，深度学习](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67)：什么是神经元，为什么会有层，其背后的数学原理是什么？资助未来的项目：https://www.patreon.com/3blue1brown 编写/交互……
- [采访一位博士后，初级 Python 开发人员](https://youtu.be/YnL9vAFphmE)：Python 编程语言。采访一位拥有博士学位的博士后、初级 Python 开发人员 Carl Kron - 播出于 © The Python。程序员幽默 Python 幽默……
- [但什么是神经网络？ | 第 1 章，深度学习](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)：什么是神经元，为什么会有层，其背后的数学原理是什么？资助未来的项目：https://www.patreon.com/3blue1brown 编写/交互……
- [特斯拉机器人出租车与人形机器人、OpenAI MLE-bench、DeepLearning.AI多模态课程、NVIDIA专家混合体、AMD GPU性能超越Nvidia](https://startup.aliyun.com/info/1088703.html)：_
- [GitHub - huggingface/hub-docs: Hugging Face Hub 的文档](https://github.com/huggingface/hub-docs)：Hugging Face Hub 的文档。通过在 GitHub 上创建账号，为 huggingface/hub-docs 的开发做出贡献。
- [GitHub - RayFernando1337/LLM-Calc: 即时计算可装入可用 RAM 的量化语言模型的最大尺寸，帮助你优化模型的推理。](https://github.com/RayFernando1337/LLM-Calc)：即时计算可装入可用 RAM 的量化语言模型的最大尺寸，帮助你优化模型的推理。 - RayFernando1337/LLM-Calc

- [nroggendorff/profession · Datasets at Hugging Face](https://huggingface.co/datasets/nroggendorff/profession): 未找到描述
- [argilla (Argilla)](https://huggingface.co/argilla): 未找到描述
- [19,500+ Beautiful Sad Girl Hurt Silhouette Stock Photos, Pictures & Royalty-Free Images - iStock](https://www.istockphoto.com/photos/beautiful-sad-girl-hurt-silhouette): 未找到描述
- [[Bug] The specified tag is not a valid quantization scheme. · Issue #1476 · huggingface/hub-docs](https://github.com/huggingface/hub-docs/issues/1476): Bug 描述。当尝试通过 Ollama 拉取模型的特定 quantization 标签时，我遇到了以下错误：The specified tag is not a valid quantization scheme。起初我以为...

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1299756731961708595) (2 条消息):

> - `Byte Pair Encoding tokenizer`
> - `Shakespeare dataset`
> - `GitHub project DeepLLMs`

- **已创建自定义 BPE Tokenizer**：一名成员完成了一个自定义实现的 **Byte Pair Encoding (BPE)** tokenizer，该 tokenizer 在 **100k 字符** 的 **tiny Shakespeare 数据集** 上训练，词表大小（vocab size）为 **3k**。
  
  - 对于那些有兴趣学习 LLMs 和 transformers 的人，可以在他们的 [GitHub 仓库](https://github.com/its-nmt05/DeepLLMs) 中进一步探索该实现。
- **DeepLLMs 探索**：名为 **DeepLLMs** 的 GitHub 仓库旨在学习 **LLMs** 和 **transformers** 的基础知识，并在此过程中探索其他有趣的主题。
  
  - 对于任何希望加深对大语言模型理解的人来说，这个项目都是一个宝贵的资源。

 

**提到的链接**：[GitHub - its-nmt05/DeepLLMs: Meant for learning the basics of LLMs and transformers and exploring other interesting stuff along the way](https://github.com/its-nmt05/DeepLLMs): 旨在学习 LLMs 和 transformers 的基础知识，并在此过程中探索其他有趣的内容 - its-nmt05/DeepLLMs

 

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1299716038295818311) (56 条消息🔥🔥):

> - `AI Offline Development Environments`
> - `Bee Agent Framework`
> - `Medical AI Research`
> - `Quantum Computing and Machine Learning`
> - `Reading and Understanding Complex Papers`

- **寻求离线 AI 开发工具**：一名成员询问由于公司网络限制而需要的 **offline development environments**，收到的反馈建议使用离线构建的便携式 **virtual machine 或 Docker**。
  
  - 社区讨论了各种选项，但提醒说导出和导入虚拟环境可能非常繁琐。
- **IBM 发布新的 Bee Agent Framework**：**IBM** 开发者发布了新的 [Bee Agent Framework](https://x.com/omarsar0/status/1849909579817140691)，其特点是针对 **Llama 3.1** 进行了优化的 Agent、沙箱化代码执行以及增强的内存管理。
  
  - 该框架允许使用兼容 OpenAI 的 API 进行集成，旨在为提供具有改进工作流控制的 Agent 服务。
- **上周 Medical AI 更新**：成员们重点介绍了最新的 [Medical AI podcast](https://youtu.be/Wt5QOv1vk2U)，讨论了 2024 年 10 月 19 日至 26 日当周的重要研究突破和新的 LLM 模型。
  
  - 重点研究包括一篇关于**医学摘要安全原则**的论文，以及 Medical AI 在诊断任务中的各种应用。
- **黑洞研究中的量子计算**：讨论围绕最近一篇关于 **ML and Quantum Computing** 的论文展开，该论文旨在研究黑洞内部，因其极高的复杂性而受到关注。
  
  - 成员们注意到该论文的难度，建议需要具备量子力学的基础知识才能理解。
- **通过 AI 提高论文理解能力**：成员们分享了理解复杂学术论文的策略，包括**追溯更基础的资源**，以及使用 AI 进行解释和测验。
  
  - 有人建议创建一个 HuggingFace Space，让用户可以与嵌入在学术论文中的 LLM 进行交互，以便更好地理解。

**提到的链接**：

- [HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly](http://arxiv.org/abs/2410.02694)：目前已有许多评估长上下文语言模型 (LCLMs) 的基准测试，但开发者通常依赖于合成任务，如大海捞针 (NIAH) 或任意任务子集。它重...
- [来自 elvis (@omarsar0) 的推文](https://x.com/omarsar0/status/1849909579817140691)：IBM 开发者发布了 Bee Agent Framework，这是一个开源框架，用于大规模构建、部署和提供 Agent 工作流。特性包括：- 为 Llama 3.1 优化的 Bee Agent - 沙箱化代码执行...
- [[Distributed w/ TorchTitan] Introducing Async Tensor Parallelism in PyTorch](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)：作者 Horace He, Less Wright, Luca Wehrstedt, Tianyu Liu, Wanchao Liang。摘要：我们在 PyTorch 中实现了实验性的 Async Tensor Parallelism 支持。我们将其集成到 TorchTitan 中并观察到：...
- [🤖 Latest Medical AI Breakthroughs | Weekly Research Review Oct 19-26 (Part 1/2)](https://youtu.be/Wt5QOv1vk2U)：本周在 Medical AI 研究中，我们探讨了在 Deepfake 检测、可解释 AI (XAI)、LLM 应用和多模态基础模型方面的突破性工作...
- [@aaditya 在 Hugging Face 上发布： "Last Week in Medical AI: Top Research Papers/Models 🔥 🏅 (October 19-26…)"](https://huggingface.co/posts/aaditya/236385586855520)：未找到描述

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1299477421531070535) (11 条消息🔥):

> - `Stable Diffusion 3.5`
> - `Fast Apply Qwen2.5 Model`
> - `Bionic Reading Hub`
> - `Google Shopping Dataset`
> - `LoRA Models`

- **Stable Diffusion 3.5 画廊展示**：一位用户创建了画廊，展示了 **Stable Diffusion 3.5 Large** 如何诠释艺术风格，在两个独立的画廊中展示了超过 **120 位艺术家**和 **140 种风格**。
  
  - 每个条目都呈现了独特的艺术诠释，并附带了所使用的 Prompt，详见 [Artists Gallery](https://enragedantelope.github.io/Artists-SD35L/) 和 [Styles Gallery](https://enragedantelope.github.io/Styles-SD35L/)。
- **推出 Fast Apply Qwen2.5 Coder 模型**：发布了一个新的开源 **Fast Apply** 模型，利用 **Qwen2.5 Coder Model** 优化了大文件的代码更新过程。
  
  - 该模型有助于快速准确地更新代码，处理自然片段而无需编辑整个文件，显著提高了效率。
- **Bionic Reading Hub 发布**：介绍了一个名为 [Bionic Reading Hub](https://github.com/SanshruthR/Bionic_Reading_Hub) 的 GitHub 项目，旨在将 PDF 转换为 Bionic Reading 格式以增强可读性。
  
  - 该仓库详细介绍了像超级英雄一样阅读的工具和方法，提升了用户的阅读体验。
- **Marqo Google Shopping 数据集发布**：**Marqo Google Shopping 1000 万数据集**已在 Hugging Face 上发布，包含全面的多模态产品检索数据，共计 **1000 万行**。
  
  - 该数据集包含用于研发的重要细节，为每个查询-文档对提供了有用的基准测试和相关性分数。
- **探索 LoRA 模型机会**：一位用户强调需要更多模仿**旧像素风格游戏**美学的 **LoRA 模型**。
  
  - 这一评论反映了生成式 AI 艺术中对复古游戏风格的兴趣，为创意探索提供了机会。

**提到的链接**：

- [SD3.5L Artist Gallery](https://enragedantelope.github.io/Artists-SD35L/)：未找到描述
- [SD3.5L Style Test Gallery](https://enragedantelope.github.io/Styles-SD35L/)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/MachineLearning/comments/1g8a3pv/r_google_shopping_10m_dataset_for_large_scale/)：未找到描述
- [goin to war with the guitar and gary4ive - ableton speedrun on fastforward - captains chair s2 ep9](https://youtu.be/4wgtG166_M8)：这次尝试一些不同的东西。如果我真的因此获得了一个订阅者，我会很惊讶。去他的算法，我在这里使用了一个名为 gary4live 的插件……
- [GitHub - SanshruthR/Bionic_Reading_Hub: Read like a superhero: Turn PDFs into Bionic Reading format](https://github.com/SanshruthR/Bionic_Reading_Hub)：像超级英雄一样阅读：将 PDF 转换为 Bionic Reading 格式 - SanshruthR/Bionic_Reading_Hub
- [ChromePunk - SDXL 1.0 - V1 | Stable Diffusion XL LoRA | Civitai](https://civitai.com/models/893518/chromepunk)：随心所欲，但要酷…… 👉🏻 Hugging Face 链接 👉🏻 数据集链接
- [rbourgeat/ChromePunk-SDXL-LoRA · Hugging Face](https://huggingface.co/rbourgeat/ChromePunk-SDXL-LoRA)：未找到描述
- [Marqo/marqo-GS-10M · Datasets at Hugging Face](https://huggingface.co/datasets/Marqo/marqo-GS-10M)：未找到描述

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1299489082241585224) (12 条消息🔥):

> - `Automated Penetration Testing` (自动化渗透测试)
> - `LLMs in Cybersecurity` (网络安全中的 LLM)
> - `Bionic Reading Hub Repo` (Bionic Reading Hub 仓库)

- **介绍一个全新的自动化渗透测试基准**：最近讨论的一篇论文介绍了一个针对 **基于 LLM 的自动化渗透测试** 的新颖基准，解决了网络安全领域缺乏全面评估工具的问题。它重点展示了 **GPT-4o** 和 **Llama 3.1-405B** 等 LLM 在使用 **PentestGPT 工具** 时的表现。
  
  - *Llama 3.1* 相比 *GPT-4o* 表现出一定优势，但目前这两种模型在执行有效的渗透测试方面都还面临困难。
- **对自动化测试未来的期待**：成员们对自动化渗透测试的进展表示热烈欢迎，并讨论了使用 **Reinforcement Learning** (强化学习) 训练 LLM 以提高其性能。挑战依然存在，鉴于目前的局限性，LLM 在渗透测试中的有效性仍有待观察。
  
  - Chad_in_the_house 指出，虽然路途漫长，但该领域的 **潜在收益** 非常显著。
- **支持性社区鼓励进步**：包括 **platinumfawn95** 在内的社区成员强调，在复杂的自动化渗透测试中，迈出 **小小的一步** 最终也能通向成功。对持续努力的支持让研究人员在解决已发现的失败点时感到安心。
  
  - Lunarflu 表达了参与该研究相关活动的渴望，凸显了集体的热情。
- **分享了有用的 GitHub 资源**：一位成员分享了 **Bionic Reading Hub GitHub 仓库**，该仓库可将 PDF 转换为 Bionic Reading 格式，可能有助于用户处理信息。该仓库名为 **Bionic Reading Hub**，并对其功能进行了简要说明。
  
  - 该工具可以提高处理网络安全复杂材料时的阅读效率。

**提到的链接**：

- [Towards Automated Penetration Testing: Introducing LLM Benchmark, Analysis, and Improvements](https://arxiv.org/abs/2410.17141)：黑客攻击对网络安全构成重大威胁，每年造成数十亿美元的损失。为了减轻这些风险，道德黑客或渗透测试被用来识别漏洞...
- [Towards Automated Penetration Testing: Introducing LLM Benchmark, Analysis, and Improvements](https://huggingface.co/blog/Isamu136/ai-pentest-benchmark)：未找到描述
- [GitHub - SanshruthR/Bionic_Reading_Hub: Read like a superhero: Turn PDFs into Bionic Reading format](https://github.com/SanshruthR/Bionic_Reading_Hub)：像超级英雄一样阅读：将 PDF 转换为 Bionic Reading 格式 - SanshruthR/Bionic_Reading_Hub

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1300109269978578944) (23 条消息🔥):

> - `Langchain SQL Agent Memory Integration` (Langchain SQL Agent 内存集成)
> - `Contributing to Open Source` (贡献开源)
> - `Qwen Models Dominance` (Qwen 模型的主导地位)
> - `Healthcare Application Models` (医疗保健应用模型)
> - `File Renaming Models` (文件重命名模型)

- **在 Langchain SQL Agent 中集成 Memory**：一位用户正在寻求帮助，希望为他们的 **Langchain SQL Agent** 添加 **Memory** 功能，以便对数据库进行自然语言查询，并表达了解决该问题的紧迫性。
  
  - 另一位用户提到，目前有 **Buffer** 和 **Summary Memory** 集成可用，允许缓存查询和响应，以改进后续回答。
- **开始贡献开源**：一位用户表达了贡献意向，但不确定初始步骤，担心必要的沟通和协议。
  
  - 得到的建议是需要一个 **GitHub 账号** 并熟悉 **git** 操作，如 cloning、branching、committing 和 pushing。
- **Qwen 模型遥遥领先**：一位成员指出 **Qwen2.5** 模型在开源社区中出人意料的主导地位，因为较新的模型在对比中经常忽略这些模型。
  
  - 社区对 **Qwen** 如何建立起如此强大的领先优势感到困惑，许多人对其成功背后的原因表示好奇。
- **为医疗保健应用寻找医疗模型**：一位用户正在为医疗保健应用寻找开源和闭源的 **医疗模型**，目前正在使用 **GPT-4o**，但希望能找到更专业的模型。
  
  - 他们表示需要一个适合在司法背景下作为 LLM 使用的 **小型模型**。
- **需要轻量级文件重命名模型**：一位用户询问是否有人拥有能够根据内容重命名文件的模型，并明确表示偏好 **极小型模型**。
  
  - 这一需求突显了在文件管理和自动化任务中对轻量级解决方案的持续需求。

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1299480663182938152) (7 messages):

> - `Contributing to Diffusers`（贡献给 Diffusers）
> - `AutoencoderKL model parameters`（AutoencoderKL 模型参数）
> - `Custom workflows with LLMs`（使用 LLM 的自定义工作流）
> - `Training models from scratch`（从零开始训练模型）
> - `Noise addition in models`（模型中的噪声添加）

- **开始使用 Diffusers**：鼓励新贡献者阅读 [贡献指南 readme](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) 并搜索 [good first issue 标签](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)。
  
  - 这有助于为那些寻求贡献的人简化入门流程。
- **关于 AutoencoderKL 参数的查询**：*Schrodingersneko* 询问了 **autoencoderKL** 模型（原始 Sd2.x 预训练 ae）的参数数量，但对话中未提供直接答案。
  
  - 社区可能有相关见解，但似乎需要对此话题进行进一步讨论。
- **探索 LLM 网页浏览工作流**：*Ritzfy* 咨询了除了摘要和创意写作等标准任务之外，使用 LLM 的自定义工作流。
  
  - 未提供具体回复，表明这可能是一个值得探索的兴趣领域。
- **挑战模型训练中的噪声添加**：*Cocos9762* 表达了对从零开始训练模型的看法，质疑了关于噪声添加根据通道特性对不同通道产生不同影响的假设。
  
  - *Ozzy_gt* 回应了关于目前噪声如何均匀添加到各通道的见解，建议这可能已经足够，但主张进行实验验证。
- **关于通道噪声动态的共识**：在 *Ozzy_gt* 的输入之后，*Cocos9762* 认可了 **Flux** 和更新的 **SD** 模型中的做法，并计划以此理解继续进行。
  
  - 他们讨论了使用学习到的权重来处理噪声添加动态的适当性。

 

**提到的链接**：<a href="[https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A"good+first+issue")">Issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)%22%3EIssues) · huggingface/diffusers: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - Issues · huggingface/diffusers

 

---

### **Notebook LM Discord ▷ #**[**announcements**](https://discord.com/channels/1124402182171672732/1182376564525113484/1299509777893953606) (1 messages):

> - `UXR Team Study`（UXR 团队研究）
> - `Remote Interviews`（远程访谈）
> - `Participant Incentives`（参与者奖励）

- **加入 UXR 团队的深度研究**：UXR 团队将于 **10 月 31 日至 11 月 6 日** 通过 Google Meet 举行远程 1:1 访谈，以收集参与者对即将推出的开发的反馈。
  
  - 成功完成访谈的参与者有资格获得 **75 美元的感谢奖励**。
- **研究参与名额有限**：该研究机会仅提供 **6 个名额**，强调需要快速响应。
  
  - 鼓励感兴趣的参与者完成 [资格调查问卷](https://forms.gle/TngndJHaPRdRsnhT7) 以检查是否符合条件。

 

**提到的链接**：[参与即将举行的 Google 概念测试研究！](https://forms.gle/TngndJHaPRdRsnhT7)：您好，我正通过一份简短的调查问卷与您联系，以核实您参加即将举行的 Google 研究调查的资格。这次研究是一个对目前正在开发的内容提供反馈的机会...

 

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1299449077532201084) (86 messages🔥🔥):

> - `Personalized Podcast App`（个性化播客应用）
> - `Deep Dive Podcasts`（深度探讨播客）
> - `Audio Overview Improvements`（音频概览改进）
> - `Using NotebookLM for Bible Study Lessons`（使用 NotebookLM 进行圣经学习课程）
> - `AI Avatar Synchronization`（AI 数字人同步）

- **介绍 PodCraft：个性化播客**：一位用户提议了一款名为 PodCraft 的应用，该应用根据特定请求提供个性化的播客内容，无需在大量剧集中筛选。
  
  - 该应用承诺以喜爱创作者的声音和风格即时访问定制内容，吸引了那些难以找到相关见解的沮丧听众。
- **对深度探讨播客的正面反馈**：许多用户表达了对深度探讨播客的喜爱，其中一位指出对使用 NotebookLM 进行每周圣经学习课程的兴趣增加。
  
  - 根据准备好的课程创建播客有助于参加者在课前更好地理解主题。
- **对 AI 播客生成变化的担忧**：一些用户报告了最近播客生成方面的变化，导致出现了不希望看到的伪影，例如重复的提示词和提到广告休息。

- 据推测，这些变化可能是训练数据调整或模型更新的结果，从而影响了输出质量。
- **AI Audio Overview 长度方面的挑战**：用户注意到生成短音频播客存在困难，因为最近的输出往往标准化在 20 分钟左右，影响了多样性。
  
  - 实验表明，较长的生成内容往往会出现重复，凸显了输出一致性方面的潜在问题。
- **HeyGen Avatars 的成功集成**：一位用户分享了一个项目，专注于让 HeyGen avatars 在不说话时表现得更真实，特别是针对一个万圣节特别视频。
  
  - 该用户对 AI 生成内容的潜力和能力表示兴奋，对所做的改进特别满意。

**提到的链接**：

- [no title found](https://notebooklm.google.com/notebook/8be91499-2108-4802-a3d3-41cf6ad2eaf7/audio)：未找到描述
- [no title found](https://notebooklm.google.com/notebook/0737a7cb-0c73-4977-90de-86924247ece2?authuser=1),)：未找到描述
- [Home | Podcraft](https://www.podcraft.info/)：未找到描述
- [How to create Audio Reactive Circle Wave Spectrums in Adobe After Effects | CC Tutorial](https://www.youtube.com/watch?v=H4r9_OGS750)：我的特效商店：https://justinodisho.com/shop Adobe 软件下载：https://prf.hn/l/dlXjya5 支持本频道：https://www.youtube.com/channel/UCy7DyWXJ...
- [LTT Reads: Abbott and Costello: Who's On First](https://www.youtube.com/watch?v=MOuJl1H6Q9c)：有没有想过喜剧能让你了解人类状况？在本集中，我们将剖析阿博特和科斯特洛传奇的“谁在第一？”常规节目——不……
- [Vocaroo | Online voice recorder](https://vocaroo.com/1kBMrHmdLKOp)：未找到描述
- [OK. This is Serious… China Is Taking Over Electric Vehicles with BYD](https://www.youtube.com/watch?v=VgAGSbreEMI)：深入探讨中国电动汽车巨头比亚迪的不可阻挡的崛起，因为它正准备挑战全球市场并与特斯拉等巨头竞争。这段视……
- [JungleTV study (scary)](https://www.youtube.com/watch?v=j2milFQsYo8)：请勿访问：https://www.jungletv.live 由 NotebookLM 根据我写的文本生成的播客。同样的文本随后被用于生成一项 C……的研究。
- [Weekly Deep Dive 28Oct24](https://youtu.be/m04Mx1M5J6s?si=kXNjcPixEtbN7JU1)：选举、高收益利差、中国回购、钯金
- [Descript Multi-track Sequences and Compositions](https://youtu.be/6fqtiP2qQYE?t=380)：了解文件、合成和序列在 Descript 中是如何组织的。在这段视频中，我详细解释了导入文件意味着什么……
- [Best Tesla for the Money? Stock Market Secrets Behind Tesla’s AI & Robotaxi Revolution!](https://youtu.be/AHOe2EVwkxQ)：在这段视频中，我们深入探讨特斯拉的未来——探索投资者乐观情绪的最新激增，以及埃隆·马斯克及其团队正在采取的雄心勃勃的步骤……
- [UNREAL MYSTERIES 4: The Halloween Special](https://www.youtube.com/watch?v=TwUrLHW8BwE)：大卫和汉娜的虚幻谜团 - 万圣节特辑！- - - 这 100% 是 AI 生成的。每一个人，每一张图片，每一个字，每一个声音，每一个……
- [PocketAI Speaker Separation Tool](https://efit.software/podcast2video/transcribe_audio_multiple.php)：为每个发言者生成独立的 MP3 文件，带有同步静音和可下载的 SRT 文件。非常适合通过真实的对话和口型同步动画来增强 AI avatar 视频。
- [Create an AI-Powered Podcast & Avatars with Your Own Voice! | Control the Conversation](https://www.youtube.com/watch?v=_-AU90-9L_U)：在本教程中，学习如何使用你的声音和 ChatGPT Advanced Voice 创建 AI 驱动的播客。与 NotebookLM 不同，在 NotebookLM 中 AI 主导对话……
- [DeepDive | Instagram, Facebook | Linktree](https://linktr.ee/deepdiveoffical)：🎤 每日语音氛围，核心焦点！加入旅程！1.2万名活跃的 DEEPDIVERS

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1299456168577597471) (516 条消息🔥🔥🔥):

> - `NotebookLM 每日限制`
> - `Audio Overview 生成`
> - `开源 AI 模型`
> - `图像生成工具`
> - `NotebookLM 功能与更新`

- **NotebookLM 每日限制引发挫败感**：用户对新实施的 Audio Overview 生成每日限制表示沮丧，并推测未来可能会推出订阅模式。
  
  - 许多人因缺乏关于这些限制的沟通而感到措手不及，强调 Google 需要提高透明度。
- **音频输出长度各异**：成员们讨论了生成的播客长度的多样性，根据源内容长度和使用的提示词，部分输出可达 39 分钟。

- 注意到源材料会严重影响生成内容的时长。
- **对开源 AI 模型的看法**：用户分享了他们对各种开源 AI 图像生成工具的偏好，明显倾向于那些使用限制较少的工具。
  
  - 用户对 Google 的 Imagen 表示不满，许多人表示存在更好的、没有使用限制的模型。
- **匿名用户关于 Notebook 功能的查询**：有人对允许的 notebook 数量提出了担忧，迹象表明每个用户的限制约为 100 个 notebook。
  
  - 参与者询问了 notebook 中源内容的有效更新和重新解析，特别是与 Google Drive 文件夹连接的情况。
- **个人经历与偏好的混合**：贡献者分享了关于驾驶、AI 技术习惯和大学经历的个人轶事，突显了社区兴趣。
  
  - 对话包括对负责任驾驶的情绪以及各种 AI 工具质量的看法，引发了用户之间的辩论。

**提到的链接**：

- [Discord - Group Chat That’s All Fun & Games](https://discordapp.com/channels/1124402182171672732/1298734876463202435/1298734876463202435)：Discord 是玩游戏、与朋友放松甚至建立全球社区的好地方。自定义你自己的空间来聊天、玩耍和聚会。
- [no title found](https://tenor.com/view/absolutely-brilliant-simon-cowell-britains-got-talent-thats-terrific-marvelou)：未找到描述
- [NotebookLM](https://youtube.com/playlist?list=PLLY_DJYCPJbvrhRcNztk6L51EKlpQIfmf&si=k03VL62c2Ys6Dfip)：未找到描述
- [Ban Hammer GIF - Ban Hammer Futurama - Discover & Share GIFs](https://tenor.com/view/ban-hammer-futurama-scruffy-gif-20750885)：点击查看 GIF
- [Absolutely Brilliant Simon Cowell GIF - Absolutely Brilliant Simon Cowell Britains Got Talent - Discover & Share GIFs](https://tenor.com/view/absolutely-brilliant-simon-cowell-britains-got-talent-thats-terrific-marvelous-gif-26035382)：点击查看 GIF
- [no title found](https://notebooklm.google.com/notebook/ae9c6c95-22fc-460f-99f2-f8e2b2489c80/audio)：未找到描述
- [Chris Evans Laugh GIF - Chris Evans Laugh - Discover & Share GIFs](https://tenor.com/view/chris-evans-laugh-gif-15277206)：点击查看 GIF
- [no title found](https://notebooklm.google.com/): 未找到描述
- [Tweet from Vicente Silveira (@vicentes)](https://x.com/vicentes/status/1844202849162625096)：凭借 NotebookLM 成名的 @joshtwoodward 在旧金山的 Google/Deepmind 活动中分享了一些关于它的酷炫内容：
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/AnthropicAI/status/1848742761278611504)：即使在记录这些演示时，我们也遇到了一些有趣的时刻。在其中一个时刻，Claude 意外停止了一个长时间运行的屏幕录制，导致所有素材丢失。后来，Claude 休息了一下...
- [no title found](https://notebooklm.google.com/notebook/2a727b05-0684-4920-b6c1-4538f666ce75/audio)：未找到描述
- [Best Tesla for the Money? Stock Market Secrets Behind Tesla’s AI & Robotaxi Revolution!](https://youtu.be/AHOe2EVwkxQ)：在这段视频中，我们深入探讨了 Tesla 的未来——探索了投资者乐观情绪的最新激增，以及 Elon Musk 及其团队正在采取的雄心勃勃的步骤...
- [Revolutionize Collaboration with Real Time Messaging Tools](https://youtube.com/shorts/l4EgnS_VcHU?si=Ge0k3Jz4U85dGAwx)：本摘要讨论的是《Beginning React and Firebase - Nabendu Biswas》一书。它是对 Firebase 的介绍，Firebase 是由 G... 提供的一套工具。
- [GitHub - souzatharsis/podcastfy: An Open Source alternative to NotebookLM's podcast feature: Transforming Multimodal Content into Captivating Multilingual Audio Conversations with GenAI](https://www.podcastfy.ai)：NotebookLM 播客功能的开源替代方案：利用 GenAI 将多模态内容转化为引人入胜的多语言音频对话 - souzatharsis/podcastfy
- [TikTok - Make Your Day](https://www.tiktok.com/@therealsoniczed/photo/7430581776180841734?is_from_webapp=1&sender_device=pc&web_id=7430710704338454021)：未找到描述
- [TikTok - Make Your Day](https://www.tiktok.com/@therealsoniczed/photo/7430651893677919493?is_from_webapp=1&sender_device=pc&web_id=7430710704338454021)：未找到描述
- [TikTok - Make Your Day](https://www.tiktok.com/@therealsoniczed/photo/7430616173437390086?is_from_webapp=1&sender_device=pc&web_id=7430710704338454021)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1299447868695777341) (484 条消息🔥🔥🔥):

> - `Unsloth Performance` (Unsloth 性能)
> - `Multimodal Models` (多模态模型)
> - `Gradient Accumulation Issues` (梯度累积问题)
> - `CuDNN Optimization` (CuDNN 优化)
> - `Training Custom Models` (训练自定义模型)

- **Unsloth 的新功能与问题**：用户讨论了 Unsloth 的最新更新，强调了速度的提升，但也报告了崩溃问题，特别是由于 `unsloth-zoo` 中对可索引数据集的假设导致的。提到了一个解决此问题的 pull request，以及社区对如何有效利用新功能的关注。
  
  - 在探索 Unsloth 的功能时，一些用户在模型量化（quantizing models）方面遇到了问题，并在 GitHub 上寻求解决方案，反映了社区在解决特定问题方面的积极参与。
- **多模态 AI 模型的讨论**：对话转向了融合视觉和语言模型的复杂性，并分享了关于适配器（adapters）如何促进不同模态之间兼容性的见解。用户指出，虽然声音集成是可能的，但在当前的 AI 开发中仍是一个具有挑战性的领域。
  
  - GLM-4 被强调为支持音频和文本输入的稳健示例，引发了关于通过音频适配器增强多模态交互潜力的讨论。
- **梯度累积（Gradient Accumulation）增强**：关于梯度累积对训练效率影响的讨论非常热烈，用户分享了在 Unsloth 最近修复后的个人经验和优化。社区成员指出，在配置训练环境时，理解 batch sizes 与内存之间的平衡至关重要。
  
  - 反馈表明，新梯度累积功能存在学习曲线，导致用户在训练工作流中进行了尝试和调整。
- **AI 训练框架的未来发展**：提出了开发用于自动化训练过程的 Gradio UI 的计划，旨在通过可调变量简化模型训练。用户讨论了该框架与 Hugging Face 的集成，以便上传适配后的和合并的模型。
  
  - 社区表达了通过简化 AI 模型训练工具来增强用户体验的兴趣，特别是针对那些技术熟练度较低的用户。
- **社区参与与学习**：一位社区成员在长期缺席后的回归引发了关于贡献和分享各种 AI 项目经验的讨论。参与者强调了代码贡献和协作解决问题作为开源社区核心组成部分的重要性。
  
  - 成员们鼓励讨论新想法，同时澄清有效的协作通常涉及分享具体的贡献，如代码实现。

**Links mentioned**:

- [Arcee-SuperNova: Training Pipeline and Model Composition](https://blog.arcee.ai/arcee-supernova-training-pipeline-and-model-composition/): 我们通过智能蒸馏 (intelligent distillation)、新颖的后训练 (post-training) 和模型合并 (model merging) 技术，训练了 Arcee SuperNova-70B 和 Arcee SuperNova-8B，使其成为具有通用智能的 Llama-3.1-405B 衍生模型。
- [Oh No GIF - Oh No Computer Saysno - Discover & Share GIFs](https://tenor.com/view/oh-no-computer-saysno-typing-gif-13727386): 点击查看 GIF
- [Sad Sigh GIF - Sad Sigh Monkey - Discover & Share GIFs](https://tenor.com/view/sad-sigh-monkey-gif-2049020038486032791): 点击查看 GIF
- [\==((====))== Unsloth - 2x faster free finetuning | Num GPUs = 1 \\\\ /| - Pastebin.com](https://pastebin.com/Le0CuH9X): Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。
- [Confused Confused Look GIF - Confused Confused look Confused face - Discover & Share GIFs](https://tenor.com/view/confused-confused-look-confused-face-funny-funny-look-gif-4850532509577233362): 点击查看 GIF
- [llama-recipes/recipes/quickstart/NotebookLlama at main · meta-llama/llama-recipes](https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/NotebookLlama): 用于微调 Meta Llama 的脚本，采用可组合的 FSDP 和 PEFT 方法，涵盖单节点/多节点 GPU。支持用于摘要和问答等应用的默认及自定义数据集。
- [unsloth (Unsloth AI)](https://huggingface.co/unsloth): 未找到描述
- [Bug Fixes in LLM Training - Gradient Accumulation](https://unsloth.ai/blog/gradient): Unsloth 的梯度累积 (Gradient Accumulation) 修复解决了 LLM 训练中的关键错误。
- [physics_of_llms/model_edit.py at main · symato/physics_of_llms](https://github.com/symato/physics_of_llms/blob/main/model_edit.py#L62,): 与越南语 LLMs 相关的实验（受 Physics of LLMs 系列启发） - symato/physics_of_llms
- [GitHub - THUDM/GLM-4-Voice: GLM-4-Voice | 端到端中英语音对话模型](https://github.com/THUDM/GLM-4-Voice/tree/main): GLM-4-Voice | 端到端中英语音对话模型。通过在 GitHub 上创建账号来为 THUDM/GLM-4-Voice 的开发做出贡献。
- [Issues · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/1192,): 微调 Llama 3.2, Mistral, Phi & Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - Issues · unslothai/unsloth
- [GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory](https://github.com/unslothai/unsloth): 微调 Llama 3.2, Mistral, Phi & Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
- [GitHub - ggerganov/llama.cpp: LLM inference in C/C++](https://github.com/ggerganov/llama.cpp): C/C++ 实现的 LLM 推理。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。
- [unsloth/Llama-3.1-Nemotron-70B-Instruct at main](https://huggingface.co/unsloth/Llama-3.1-Nemotron-70B-Instruct/tree/main): 未找到描述
- [Home](https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-fixes-oom-or-crashing): 微调 Llama 3.2, Mistral, Phi & Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
- [Weights & Biases: The AI Developer Platform](https://wandb.ai): Weights & Biases 是领先的 AI 开发者平台，用于训练和微调模型，管理从实验到生产的模型，并跟踪和评估由 LLMs 驱动的 GenAI 应用。

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1299462998959853700) (15 条消息🔥):

> - `学校里的存在主义危机`
> - `Unsloth 敬礼表情符号`
> - `在学校交朋友`
> - `怀念学校生活`
> - `应对学校压力`

- **来自学校压力的存在主义危机**：一位成员表达了*来自学校的存在主义危机*，促使其他人建议应对方法。
  
  - 有人建议 *“忍耐……它不会永远持续下去”*，另一位幽默地指出 *“谢天谢地它不会”*，反映了普遍的感受。
- **庆祝 Unsloth 敬礼**：一位成员认可了 **unsloth salute** 表情符号的存在，并用愉快的言辞表达了对它的喜爱。
  
  - 使用像 <:slothsalute:1257540362868752396> 这样的表情符号有助于营造小组轻松愉快的氛围。
- **在学校寻找志同道合的朋友**：一位成员建议结交 *志同道合的朋友可以增强学校体验*，减少孤独感。
  
  - 另一位表示赞同，称 *“确实如此”*，强调了社交联系的重要性。
- **对学校生活的怀念**：成员们讨论了 *年长者往往希望重返校园* 的观点，因为那时生活简单且有趣。
  
  - 当有人评论 *“有趣的是……当你变老时，你会想回到学校”* 时，这种情感得到了共鸣。

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1299463129431932968) (91 messages🔥🔥):

> - `Fine-tuning with SFTTrainer`
> - `Errors in Unsloth functions`
> - `Using multiple datasets for training`
> - `Quantization errors in model conversion`
> - `Using system prompts in training`

- **使用类别权重进行模型微调**：一位用户询问了在微调期间使用类别权重的有效性，以及如何在 SFTTrainer 中实现这一点。
  
  - 这表明了用户对于通过处理类别不平衡来提高模型性能的兴趣。
- **使用评估数据集时的运行时错误**：一位用户报告在使用 Unsloth 评估数据集时遇到了 “cuDNN Frontend error”，而在不使用它时则没有出现该错误。
  
  - 他们后来发现需要升级库，这暗示了版本不匹配可能会导致不一致。
- **模型量化中的挑战**：用户在尝试将模型转换为 Q8_0 GGUF 格式时遇到了错误，并收到了编译 llama.cpp 以解决问题的建议。
  
  - 转换错误突显了与目录路径相关的问题以及对特定构建配置的需求。
- **在训练中使用系统提示词**：一位用户对于在使用 train_on_responses_only 方法时如何正确加入系统提示词（system prompt）感到困惑。
  
  - 讨论明确了系统提示词可能需要不同的处理方式，以确保它们被有效地整合到数据集输入中。
- **在 Docker 中运行 Ollama**：一位用户在 Docker 上运行 Ollama 以配合 Python 客户端使用时面临困难，并寻求他人的帮助。
  
  - 这突显了在为特定应用使用 Docker 容器时常见的集成挑战。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ),): 未找到描述
- [Home](https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-fixes-oom-or-crashing): 使用 Unsloth 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3.2, Mistral, Phi & Gemma LLM - unslothai/unsloth
- [GitHub - ggerganov/llama.cpp: LLM inference in C/C++](https://github.com/ggerganov/llama.cpp): C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
- [unsloth/unsloth/tokenizer_utils.py at d76eda4f66828d66aa6a1b01a0d03323e43810dd · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/d76eda4f66828d66aa6a1b01a0d03323e43810dd/unsloth/tokenizer_utils.py#L935,): 使用 Unsloth 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3.2, Mistral, Phi & Gemma LLM - unslothai/unsloth
- [llama3.1-finetune-quick-tuto/main.py at main · donglinkang2021/llama3.1-finetune-quick-tuto](https://github.com/donglinkang2021/llama3.1-finetune-quick-tuto/blob/main/main.py): 使用 Unsloth 在你自己的数据集上微调 llama3.1。 - donglinkang2021/llama3.1-finetune-quick-tuto
- [iamtarun/python_code_instructions_18k_alpaca · Datasets at Hugging Face](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca): 未找到描述
- [GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory](https://github.com/unslothai/unsloth/): 使用 Unsloth 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3.2, Mistral, Phi & Gemma LLM - unslothai/unsloth
- [GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers.git): 🤗 Transformers: 为 Pytorch, TensorFlow 和 JAX 提供最先进的机器学习。 - huggingface/transformers

---

### **Unsloth AI (Daniel Han) ▷ #**[**showcase**](https://discord.com/channels/1179035537009545276/1179779344894263297/1299547578341392406) (3 messages):

> - `Llama 405B Performance`
> - `Llama 3.1-405B-Instruct Breakthrough`

- **Llama 405B 在 Nvidia H200 SXM 上达到 142 tok/s**：**Llama 405B** 的性能在 **Nvidia H200 SXM** 上达到了 **142 tokens per second**，展示了强大的处理能力。
  
  - 这一成就突显了该模型在处理复杂任务时的效率。
- **Llama 3.1-405B-Instruct 突破 100 t/s 大关**：**Llama 3.1-405B-Instruct** 在没有任何投机技巧或性能损失的情况下，突破了 **100 tokens per second** 的大关，并保持了完整的 **128k context length**。
  
  - 用户强调这一里程碑是*数月工作*的结果，标志着模型性能的重大进步。

 

**提到的链接**: [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gc7s6p/llama_405b_up_to_142_toks_on_nvidia_h200_sxm/): 未找到描述

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1300297067981176884) (2 条消息):

> - `AI video generators`
> - `3D models integration`
> - `Camera controls`
> - `Consistent environments`
> - `Unreal Engine physics`

- **探索基于 3D Models 的 AI Video Generators**：一位成员有兴趣开发以 **3D models** 为基础框架的 **AI video generators**，并提出了诸如摄像机控制（camera controls）和一致性环境（consistent environments）等功能。
  
  - 他们特别感兴趣于集成 **Unreal Engine** 或类似的基础物理引擎，以增强整体体验。
- **寻求关于 AI Video Generator 项目的见解**：该成员询问是否有人见过或目前正在从事涉及 AI 和视频生成的类似项目。
  
  - 这表明了社区在这些先进技术上进行协作的潜在兴趣。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1300219799023915070) (2 条消息):

> - `Dualformer Model`
> - `Attention Calculation on CPU`
> - `System 1 and System 2 Thinking`
> - `Reasoning in Transformers`

- **CPU 上的 Attention？新鲜事！**：一位成员对在 **CPU** 上计算 **attention** 表示惊讶，并质疑这种方法的有效性。
  
  - 讨论暗示了这对 AI 模型的计算效率和性能具有潜在影响。
- **引入 Dualformer 以增强推理能力**：**Dualformer** 模型在 **Transformers** 中集成了**快速**（系统 1）和**慢速**（系统 2）推理，解决了推理任务中的效率问题。
  
  - 它超越了 **Searchformer** 和 **Solution-Only** 等现有模型，通过定制的训练策略显著提升了在迷宫导航和数学问题上的表现。
- **AI 中的认知系统**：论文中提出的底层理论将人类认知的**系统 1** 和**系统 2** 与 **Transformer** 的性能联系起来，增强了推理能力。
  
  - 这种联系揭示了关于 AI 模型设计的关键见解，以便在解决问题任务中更好地模拟认知过程。

 

**提到的链接**：[Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces](https://arxiv.org/abs/2410.09918v1)：在人类认知理论中，人类思维由两个系统主导：快速且直觉的系统 1，以及较慢但更深思熟虑的系统 2。最近的研究表明，结合系统...

 

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1299448005270700085) (203 条消息🔥🔥):

> - `LM Studio Plugins`
> - `Headless Mode` 与模型加载
> - `GPU` 与 `CPU` 的性能对比
> - `ROCm` vs `CUDA`
> - `ChatGPT` 为 `LM Studio` 生成的网页

- **LM Studio 支持多 GPU**：用户讨论了 `LM Studio` 可以利用多个 `GPU`，一位成员确认他们的两块 `RTX 3060 Ti` 都能被系统识别。
  
  - 然而，该应用程序可能需要特定的配置才能在双 `GPU` 下实现最佳运行。
- **对公开暴露服务器安全性的担忧**：一位用户提出了将 `LM Studio` 服务器暴露在公共互联网上的安全担忧，并收到了关于使用 `VPN` 和适当防火墙的建议。
  
  - 值得注意的是，在没有额外预防措施的情况下，`LM Studio` 尚未准备好直接暴露在互联网上。
- **不同模型的性能指标**：成员们分享了性能指标，指出 `9B` 模型在 `GPU` 上达到约 62 `tokens per second`，而 `CPU` 处理速度约为 7.5 `tokens per second`。
  
  - 这突显了大型语言模型在 `GPU` 和 `CPU` 之间显著的性能差异。
- **从 0.2 版本过渡到 0.3 版本**：讨论透露，从 `LM Studio` 0.2.x 版本切换到 0.3.x 版本不会自动更新，这可能会导致对可用功能的困惑。
  
  - 用户鼓励其他人手动更新到新版本以获得最佳体验。
- **查看 LM Studio 对话的网页**：一位用户分享了他们开发的一个简单网页链接，用于查看 `LM Studio` 的对话 `JSON` 文件，并允许自定义用户显示名称。
  
  - 该工具旨在帮助用户在更友好的界面中更好地管理他们的对话日志。

**提到的链接**：

- [Gotta Catch Em All GIF - Gotta Catch Em All - Discover & Share GIFs](https://tenor.com/view/gotta-catch-em-all-gif-5662253)：点击查看 GIF
- [RYZEN AI MAX + Is Coming! 40CU iGPU APU, The End Of Mid Range GPU's?](https://youtu.be/MdyxbLPlGKc?si=jiLUq4mdZ6tHxz9f)：Ryzen AI Max 3 款具有强劲 AI 和图形性能的新型号。准备好迎接性能提升！Ryzen AI Max 系列可能包含三个令人兴奋的型号。从...
- [Chat Log Viewer for LM Studio 0.3.x - Pastebin.com](https://pastebin.com/TijQT9Ke)：Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储文本一段时间的网站。
- [GitHub - p4stoboy/AI-Junction: AI Junction exposes a Discord interface for generating text and images using LMStudio (or any OpenAI API interface) and ComfyUI back-ends.](https://github.com/p4stoboy/AI-Junction)：AI Junction 提供了一个 `Discord` 接口，用于使用 `LM Studio`（或任何 `OpenAI API` 接口）和 `ComfyUI` 后端生成文本和图像。该机器人允许用户管理自己的 `LLM` 提示词和图像生成配置，并在服务器内使用自定义提示词生成内容。
- [GitHub - ml-explore/mlx: MLX: An array framework for Apple silicon](https://github.com/ml-explore/mlx)：`MLX`：适用于 `Apple silicon` 的数组框架。通过在 `GitHub` 上创建账户来为 `ml-explore/mlx` 的开发做出贡献。
- [lmstudio.js/packages/lms-client/src/embedding/EmbeddingDynamicHandle.ts at 582eeed1536ce937d0d51bf3e5fbbee1698a51e0 · lmstudio-ai/lmstudio.js](https://github.com/lmstudio-ai/lmstudio.js/blob/582eeed1536ce937d0d51bf3e5fbbee1698a51e0/packages/lms-client/src/embedding/EmbeddingDynamicHandle.ts#L73)：`LM Studio TypeScript SDK`（预发布公开测试版）- lmstudio-ai/lmstudio.js
- [Run LM Studio as a service (headless) - Advanced | LM Studio Docs](https://lmstudio.ai/docs/advanced/headless)：`LM Studio` 的无界面运行：在后台运行、在机器登录时启动，并根据需求加载模型。
- [lmstudio.js Code Examples - SDK (TypeScript) | LM Studio Docs](https://lmstudio.ai/docs/sdk/lmstudioclient#getting-prediction-stats)：在 `TypeScript` 应用程序中使用 `lmstudio.js` 的示例。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1299457930231287880) (232 条消息🔥🔥):

> - `混合 GPU 性能`
> - `NPU 实用性担忧`
> - `主板插槽间距`
> - `Apple M3 及未来 M4 性能`
> - `高性价比 AI 与电脑组装`

- **混合 GPU 性能担忧**：用户对混合 GPU 配置的性能表示担忧，特别是 **RTX 3090 和 4090** 型号之间的混搭，并质疑是否应该卖掉高端显卡以换取更好的兼容性。
  
  - 对能够运行大模型而不单纯追求推理速度的设备需求，引发了关于最佳配置的讨论。
- **NPU 实用性担忧**：一位用户分享了对 **NPU** 的失望，认为其缺乏软件支持和功能，尤其是在 AI 任务中与常规 PC 配置相比。
  
  - 有推测认为 Intel 应该如何与 Microsoft 更好地合作利用 NPU，以提高其在 AI 工作负载中的可行性。
- **主板插槽间距**：关于主板插槽间距的讨论强调了 E-ATX 和专用主板如何为多 GPU 配置提供更好的气流。
  
  - 建议包括测量当前间距并研究能够提供更好间距的主板。
- **Apple M3 及未来 M4 性能**：Apple **M4** 的发布引发了关于其是否适合运行大型模型的辩论，对其能力的怀疑占据了对话的主导。
  
  - 用户对新款 Mac 中极小的内存分配表示沮丧，并批评了增加 RAM 的高昂升级成本。
- **高性价比 AI 与电脑组装**：对比了组装具有 **更高 RAM 容量** 的自定义系统与购买 Apple 系统的成本。
  
  - 用户指出，组装一台强大的具备 AI 能力的机器比 Apple 的产品更具性价比，因为后者往往缺乏处理严肃工作负载所需的硬件。

**提到的链接**：

- [NVIDIA Jetson AGX Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)：为下一代机器人提供更高级别的 AI 性能。
- [GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)：多显卡 NVIDIA GPU 还是 Apple Silicon 更适合大语言模型推理？- XiongjieDai/GPU-Benchmarks-on-LLM-Inference

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1300540945682399253) (1 条消息):

> - `Inflection`

- **Inflection 计费问题已解决**：Inflection 在解决了上周的计费问题后已重新上线，确保用户能够不间断访问。
  
  - 欲了解更多详情，请访问 [Inflection 3 PI 页面](https://openrouter.ai/inflection/inflection-3-pi) 和 [Inflection 3 Productivity 页面](https://openrouter.ai/inflection/inflection-3-productivity)。
- **Inflection 产品线澄清**：在解决计费问题的同时，Inflection 澄清了其产品线，展示了旨在增强生产力的功能。
  
  - 可以通过之前分享的专用产品链接探索详细信息。

**提到的链接**：

- [Inflection 3 Pi - API, Providers, Stats](https://openrouter.ai/inflection/inflection-3-pi)：Inflection 3 Pi 为 Inflection 的 [Pi](https://pi.ai) 聊天机器人提供动力，包括背景故事、情感智能、生产力和安全性。支持通过 API 运行 Inflection 3 Pi。
- [Inflection 3 Productivity - API, Providers, Stats](https://openrouter.ai/inflection/inflection-3-productivity)：Inflection 3 Productivity 针对遵循指令进行了优化。它更适合需要 JSON 输出或精确遵守提供指南的任务。支持通过 API 运行 Inflection 3 Productivity。

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1299450656985452645) (364 条消息🔥🔥):

> - `OpenRouter Connectivity Issues` (OpenRouter 连接问题)
> - `Sonnet Model Performance Changes` (Sonnet 模型性能变化)
> - `Grok 2 Multimodal Release` (Grok 2 多模态发布)
> - `Prompt Engineering Techniques` (Prompt Engineering 技术)
> - `Model Parameters and Providers` (模型参数与提供商)

- **OpenRouter 连接问题持续存在**：用户报告了 OpenRouter 的间歇性连接问题，尽管状态页面显示为绿色，但仍遇到了 Cloudflare 520 和 524 错误。
  
  - 一些用户建议通过不同的浏览器测试连接，有迹象表明这些问题在欧洲地区可能更为明显。
- **观察到 Sonnet 模型性能变化**：多位用户注意到 Sonnet 模型的响应质量有所下降，且追问次数增加，他们认为模型生成的回答比以前更加泛泛而谈。
  
  - 有人怀疑在免费版本受限后，模型的行为可能发生了变化，用户感觉其响应灵敏度不如之前的交互。
- **Grok 2 多模态功能发布**：讨论集中在最近宣布的 Grok 2 多模态特性上，该特性使其能够结合文本理解图像。
  
  - 用户对这一新模型的意义和能力表示好奇，特别是与现有模型的对比。
- **有效的 Prompt Engineering 技术**：分享了编写有效 Prompt 的技巧，重点在于如何通过提供详细指令来引导模型生成更长的响应。
  
  - 用户分享了旨在最大化输出长度和质量的结构化 Prompt 示例，强调了具体性（Specificity）的重要性。
- **了解模型参数和提供商**：对话包括关于列出模型能力以及不同提供商在输出长度和响应质量方面潜在差异的讨论。
  
  - 用户询问了是否存在具有扩展 Token 限制的模型，并分享了不同提供商在处理这些模型方面的能力信息。

**提及的链接**：

- [Internet Speed Test - Measure Network Performance | Cloudflare](https://speed.cloudflare.com/)：互联网速度测试 - 衡量网络性能 | Cloudflare。测试您的互联网连接。使用我们的互联网速度测试检查您的网络性能。由 Cloudflare 的全球边缘网络提供支持。
- [Activity | OpenRouter](https://openrouter.ai/activity)：查看您在 OpenRouter 上使用模型的情况。
- [Open VLM Leaderboard - a Hugging Face Space by opencompass](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)：未找到描述
- [Pixar Draw GIF - Pixar Draw Single - Discover & Share GIFs](https://tenor.com/view/pixar-draw-single-cosplay-gif-5670451)：点击查看 GIF
- [Tweet from Niels Rogge (@NielsRogge)](https://x.com/NielsRogge/status/1850219356690509910)：Meta 在 Hub 上发布了一个新的视频 LLM，它是开源视频理解领域的新 SOTA > 基于 SigLIP/DINOv2 和 Qwen2/Llama 3.2 构建 > 包含一个 3B 参数模型...
- [OpenRouter Status](https://status.openrouter.ai/)：OpenRouter 事件历史
- [Automated clearing house - Wikipedia](https://en.wikipedia.org/wiki/Automated_clearing_house)：未找到描述
- [The killer app of Gemini Pro 1.5 is video](https://simonwillison.net/2024/Feb/21/gemini-pro-video/)：Gemini Pro 1.5 的杀手级应用是视频。上周 Google 推出了 Gemini Pro 1.5，这是对其 Gemini 系列 AI 模型的巨大升级。Gemini Pro 1.5 拥有 1,000,000 Token 的上下文窗口。这是巨大的——此前……
- [Hermes 3 405B Instruct - API, Providers, Stats](https://openrouter.ai/nousresearch/hermes-3-llama-3.1-405b)：Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理、多轮对话、长上下文连贯性……
- [Tweet from Elon Musk (@elonmusk)](https://x.com/elonmusk/status/1850724646414606406)：Grok 现在可以理解图像，甚至能解释笑话的含义。这是一个早期版本。它将迅速改进。https://x.com/i/grok/share/roBaNzwhuOYzfHUuQLo4OWXjO
- [GitHub - sashabaranov/go-openai: OpenAI ChatGPT, GPT-3, GPT-4, DALL·E, Whisper API wrapper for Go](https://github.com/sashabaranov/go-openai)：OpenAI ChatGPT, GPT-3, GPT-4, DALL·E, Whisper API 的 Go 语言封装库 - sashabaranov/go-openai

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1299473520018853921) (7 messages):

> - `Access to Integrations`

- **许多用户寻求 Access to Integrations**：大量用户表达了对获取 **Access to Integrations** 的兴趣，表明对该功能有强烈需求。
  
  - 请求非常礼貌且一致，诸如 *'I would get access to integrations. Thanks'* 之类的短语显示出一种统一的诉求。
- **对集成权限的礼貌呼吁**：用户一直在礼貌地请求 **gain access** 到集成功能，这表明社区非常活跃且渴望新功能。
  
  - 消息中反复强调了对访问权限的渴望，多次出现用户对潜在帮助表示感谢的情况。

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1299491974700863569) (60 messages🔥🔥):

> - `OpenAI Whisper`
> - `Gemini AI Model`
> - `Homomorphic Encryption`
> - `Moonshine ASR`
> - `Moondream Funding`

- **Whisper 与新型 ASR 模型**：Whisper 正在与 Moonshine 等新型 ASR 技术进行比较，后者声称在边缘设备上以更低的计算需求表现更好，实现了显著的速度提升。
  
  - 批评者指出，虽然 Moonshine 可能会超越 Whisper 的较小模型，但 Whisper 的较大模型在性能上仍具有优势。
- **Apple 的 Homomorphic Encryption**：Apple 关于 Homomorphic Encryption 的公告表明，在不损害机密性的情况下，如何将私有数据用于 AI 方面取得了重大进展，被比作 “AI 的 HTTPS 时刻”。
  
  - 讨论强调了在不暴露私有信息的情况下进行检索和 Embedding 的潜在应用，尽管由于速度问题，其在推理（Inference）方面的可行性仍不确定。
- **Moondream 获得融资**：Moondream 宣布成功完成一轮融资，筹集了 450 万美元，用于探索小型 AI 模型在竞争激烈的 AI 应用中的有效性。
  
  - 此举引发了关于小型模型局限性的讨论，一些人质疑在不遇到已知行业挑战的情况下，它们能走多远。
- **关于推理硬件选择的讨论**：关于推理硬件选择的对话一直在持续，反映了对依赖 Cloud APIs 的担忧，并探索了像 Groq 这样的替代方案。
  
  - 对话寻求关于开发团队如何做出硬件选择，以及他们是否在等待主要云提供商更广泛采用的见解。
- **AI 评估与 Hallucinations**：在关于 Whisper 等 AI 转录工具的讨论中，对 Hallucinations 的担忧凸显了适当评估方法的重要性，以减轻医疗保健等敏感环境中的不准确性。
  
  - 专家强调了用户参与和持续评估在增强 AI 输出可靠性方面的作用，尤其是在关键应用中。

**提到的链接**：

- [未找到标题](https://dev.to/griptape/griptape-vs-langchain-crewai-and-llamaindex-which-ai-framework-performs-best-354j)：未找到描述
- [来自 Brad (@brad_agi) 的推文](https://x.com/brad_agi/status/1850740371661230555)：Fast Apply 发布 - 一个开源的类 Cursor 模型，旨在对文件应用编辑
- [来自 Varun (@varun_mathur) 的推文](https://x.com/varun_mathur/status/1850502044307562871?s=46&t=PW8PiFwluc0tdmv2tOMdEg)：这是 Apple 围绕密码学发布的一个改变游戏规则的公告。在某些方面，它是 “AI 的 HTTPS 时刻”。这意味着：你的私人机密数据可以与其他数据汇集在一起……
- [来自 VentureBeat (@VentureBeat) 的推文](https://x.com/VentureBeat/status/1850885273749532852)：Moondream 筹集 450 万美元以证明小型 AI 模型依然实力强劲 https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-that-smaller-ai-models-can-still-pack-a-punch/
- [Google 计划很快发布其下一个 Gemini 模型](https://www.theverge.com/2024/10/25/24279600/google-next-gemini-ai-model-openai-december)：12 月正成为 OpenAI 和 Google 展开 AI 发布对决的一个月。
- [研究人员称医院使用的 AI 驱动转录工具会编造从未说过的话](https://apnews.com/article/ai-artificial-intelligence-health-business-90020cdf5fa16c79ca2e5b6c4c9bbb14?taid=671cde7444b38d00014b98db&utm_campaign=TrueAnthem&utm_medium=AP&utm_source=Twitter)：Whisper 是一款由人工智能驱动的热门转录工具，但它有一个重大缺陷。它会编造从未说过的话。

- [2024年大规模 AI 基础设施现状](https://ai-infrastructure.org/the-state-of-ai-infrastructure-at-scale-2024/)：财富 1000 强公司如何应对 AI 对其基础设施日益增长的需求？他们能否在快速部署 Gen AI 的同时，对其进行严格管控以交付出色的……
- [Zight 录制 2024-10-25 08.23.03 PM](https://share.zight.com/wbuqqWOB)：未找到描述
- [Tom Dörr (@tom_doerr) 的推文](https://x.com/tom_doerr/status/1850307683963801949?s=46)：声称超越了 Whisper + 实时转录
- [AI 的未来需要更灵活的 GPU 容量](https://modal.com/blog/the-future-of-ai-needs-more-flexible-gpu-capacity)：为什么 Modal 痴迷于 Serverless AI 基础设施
- [用于流水线处理的开源 LLM 框架对比](https://winder.ai/comparison-open-source-llm-frameworks-pipelining/)：使用最佳的开源 LLM 框架、Python 库和编排工具优化您的 LLM 项目。
- [Paul Calcraft (@paul_cal) 的推文](https://x.com/paul_cal/status/1850262678712856764?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：LLM 玩“你画我猜”！引用 Simon Willison (@simonw) 的话：尝试一个新的 LLM 基准测试：它们画一个骑自行车的鹈鹕的 SVG 图形能力如何？这是 Claude 3.5 之间的区别……
- [詹姆斯·卡梅隆：SCSP AI+Robotics 峰会特别视频致辞](https://youtu.be/e6Uq_5JemrI?si=5n9mWAtLZILG9spi)：三届奥斯卡金像奖得主、导演詹姆斯·卡梅隆在 AI+ Robotics 峰会上发表特别讲话。特别竞争研究项目（SCSP）的 AI+ 峰会……
- [NotebookLM 是如何诞生的](https://youtu.be/v1mOIdH9q7k?si=JPgkH-VRgKpTMhAk)？：Raiza Martin 和 Usama Bin Shafqat 是 NotebookLM 功能背后的首席 PM 和 AI 工程师，该功能为我们带来了第一个病毒式传播的 AI 语音体验，“……
- [vik (@vikhyatk) 的推文](https://x.com/vikhyatk/status/1850990119937064971?s=46)：我创办了一家公司……引用 VentureBeat (@VentureBeat) 的报道：Moondream 融资 450 万美元，证明较小的 AI 模型依然实力不俗 https://venturebeat.com/ai/moondream-raises-4-5m-to-prove-tha...
- [GitHub - protectai/vulnhuntr: 使用 LLM 进行零样本漏洞发现](https://github.com/protectai/vulnhuntr)：使用 LLM 进行零样本漏洞发现。通过在 GitHub 上创建账号来为 protectai/vulnhuntr 的开发做出贡献。
- [GitHub - Marker-Inc-Korea/AutoRAG: 用于 RAG 的 AutoML 工具](https://github.com/Marker-Inc-Korea/AutoRAG)：用于 RAG 的 AutoML 工具。通过在 GitHub 上创建账号来为 Marker-Inc-Korea/AutoRAG 的开发做出贡献。
- [GitHub - usefulsensors/moonshine: 适用于边缘设备的快速且准确的自动语音识别 (ASR)](https://github.com/usefulsensors/moonshine)：适用于边缘设备的快速且准确的自动语音识别 (ASR) - usefulsensors/moonshine
- [未找到标题](https://notebooklm.google.com/notebook/d787d6c2-7d7b-478c-b7d5-a0be4c74ae19/audio)：未找到描述
- [Moonshine: 用于实时转录和语音命令的语音识别](https://arxiv.org/abs/2410.15608)：本文介绍了 Moonshine，一个为实时转录和语音命令处理优化的语音识别模型系列。Moonshine 基于 Encoder-Decoder Transformer 架构……
- [介绍 Moonshine，语音转文本的新标杆](https://petewarden.com/2024/10/21/introducing-moonshine-the-new-state-of-the-art-for-speech-to-text/)：你能想象使用一个按键后需要两秒钟才能在屏幕上显示的键盘吗？这是大多数语音界面的典型延迟，难怪它们会失败……
- [GitHub - n8n-io/n8n: 免费且提供源码的公平代码许可工作流自动化工具。轻松实现跨不同服务的任务自动化。](https://github.com/n8n-io/n8n)：免费且提供源码的公平代码许可工作流自动化工具。轻松实现跨不同服务的任务自动化。 - n8n-io/n8n
- [n8n.io - 强大的工作流自动化工具](https://n8n.io/)：n8n 是一款免费且提供源码的工作流自动化工具
- [最佳应用与软件集成 | n8n](https://n8n.io/integrations/)：通过这些顶级的软件集成优化您的工作流程。使用 n8n 在不同应用之间无缝移动和转换数据。
- [面向企业 AI 团队的 LLM 平台 | deepset](https://www.deepset.ai/deepset-cloud)：查看 deepset Cloud，我们用于更快速构建 LLM 应用的平台。加速企业级 NLP。
- [AI 基础设施联盟](https://ai-infrastructure.org/the-st)：AI 基础设施联盟汇集了世界上最优秀的 AI/ML 基础设施公司。我们共同致力于为 MLOps 创建标准技术栈（Canonical Stack）。

### **Latent Space ▷ #**[**ai-in-action-club**](https://discord.com/channels/822583790773862470/1200548371715342479/1299462688815972447) (260 条消息🔥🔥):

> - `Cursor 高级技巧`
> - `Discord 中的音频问题`
> - `LLM 集成顾虑`
> - `Markdown 文件生成挑战`
> - `即将举行的活动建议`

- **提升编程工作流的 Cursor 高级技巧**：参与者讨论了各种 **Cursor Pro Tips**，强调了如使用 `ctrl+k` 进行局部编辑以及用于提高编程效率的 **multifile high-level edits**（多文件高层级编辑）等工作流。
  
  - *建议举行后续会议*以探索更多技巧，因为目前仅涵盖了一小部分有用内容。
- **影响会议质量的音频问题**：多位与会者报告在会议期间遇到 **音频问题和掉线**，影响了他们的跟进。
  
  - 参与者对 **Discord 的性能** 表示担忧，认为服务器相关问题可能导致了这些中断。
- **职场中使用 LLM 的挑战**：讨论围绕雇主的 **安全顾虑** 展开，这些顾虑因代码隐私问题限制了 Cursor 和 Claude 等 LLM 的使用。
  
  - 参与者分享了潜在的变通方案，包括使用 **AWS Bedrock** 作为交互代理，同时保持代码安全。
- **Cursor 中的 Markdown 生成问题**：一位参与者提到直接在 Cursor 中 **生成 Markdown 文件** 存在困难，转而选择使用 Claude 进行快速设置。
  
  - *Markdown 格式化需要改进*，并建议与 Cursor 开发者沟通以寻求潜在的增强功能。
- **对未来活动和协作的建议**：有人建议策划更多的 **NYC 技术活动**，并以系列形式讨论 Cursor 高级技巧。
  
  - 大家共同对探索 **LLM 工具集成** 及其如何改善编程产出表现出兴趣，这表明该领域有丰富的讨论空间。

**提到的链接**：

- [未找到标题](https://docs.cursor.com/context/ignore-files)：未找到描述
- [Hp Harry Potter GIF - Hp Harry Potter Snape - Discover & Share GIFs](https://tenor.com/view/hp-harry-potter-snape-always-stare-gif-17584635)：点击查看 GIF
- [YOLO11 🚀 NEW](https://docs.ultralytics.com/models/yolo11/)：探索 YOLO11，这是最先进目标检测领域的最新进展，为各种计算机视觉任务提供无与伦比的准确性和效率。
- [Trolling Is An Honored Profession Leland Townsend GIF - Trolling Is An Honored Profession Leland Townsend Evil - Discover & Share GIFs](https://tenor.com/view/trolling-is-an-honored-profession-leland-townsend-evil-the-demon-of-memes-trolling-is-a-respectable-career-gif-26018384)：点击查看 GIF
- [yikes’s Substack | Substack](https://butdoesitworktho.substack.com)：我的个人 Substack。点击阅读 yikes 的 Substack，一份 Substack 出版物。一年前发布。
- [RDoc Documentation](https://ruby-doc.org/3.3.5/index.html)：未找到描述
- [RDoc Documentation](https://ruby-doc.org/3.3.5/)：未找到描述
- [GitHub - twilwa/crawler.nvim: uses firecrawl, jina, and/or jsondr to render webpages in neovim buffers](https://github.com/twilwa/crawler.nvim)：使用 firecrawl, jina, 和/或 jsondr 在 neovim 缓冲区中渲染网页 - twilwa/crawler.nvim
- [GitHub - sigoden/aichat: All-in-one LLM CLI tool featuring Shell Assistant, Chat-REPL, RAG, AI tools & agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more.](https://github.com/sigoden/aichat)：全能 LLM CLI 工具，具有 Shell Assistant, Chat-REPL, RAG, AI 工具和 Agent 功能，支持访问 OpenAI, Claude, Gemini, Ollama, Groq 等。 - sigoden/aichat

---

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1300548310456729654) (1 条消息):

> - `策展人计划 (Curators Program)`
> - `发现信息流 (Discover Feed)`

- **加入策展人革命！**：Perplexity 团队宣布启动 **Curators Program**，旨在为 **Discover feed** 开发引人入胜的内容。
  
  - 感兴趣的人士可以通过 [此链接](https://perplexity.ai/curators) 申请或标记朋友加入该倡议。
- **召集所有创意爱好者！**：该计划邀请那些喜欢制作 **Pinterest 展板**、编辑 **Wikipedia 页面** 以及深入研究 **YouTube 视频论文** 的人参与。
  
  - 策展人将发挥关键作用，直接通过产品启发、惊喜并告知全球数百万受众。

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1299467968920682598) (212 条消息🔥🔥):

> - `MacOS app 体验`
> - `Perplexity AI 新功能`
> - `即将推出的 AI 模型`
> - `用户对 LLM 的担忧`
> - `Perplexity 中的语言选项`

- **MacOS 应用功能的评价褒贬不一**：用户报告了 Perplexity MacOS 应用崩溃和弹窗处理困难的问题，引发了对可用性和性能的担忧。
  
  - 与 Web 应用相比，部分用户对复制粘贴图片的限制表示不满，并希望能有更好的反馈选项。
- **新功能引发兴奋与质疑**：购物功能的引入出人意料，导致用户评价褒贬不一，有呼声要求将这些功能模块化，以提供更好的用户体验。
  
  - 针对新开发背后的战略原因，目前仍存在持续的猜测，用户渴望看到这些功能对日常使用的影响。
- **对下一代 AI 模型的期待**：行业传闻暗示 GPT-5 可能在 2024 年 12 月发布，反映了 AI 开发者之间激烈的竞争态势。
  
  - Meta 打造自有搜索引擎的举措也凸显了不断演变的动态，用户对这些进展将如何影响功能表现深感兴趣。
- **用户对 LLM 性能的担忧**：一些用户注意到模型响应存在差异，特别是在使用 Claude Sonnet 和 GPT-4 时，并对达到回答限制感到沮丧。
  
  - 用户对模型输出的可靠性提出了担忧，特别是在处理翻译或复杂查询时。
- **观察到语言支持问题**：语言切换问题引起了混乱，多位用户报告称，尽管进行了设置，其 Perplexity 应用仍会默认跳转到非预期的语言。
  
  - 反馈表明，用户希望应用内能有更清晰的语言选项解决方案，以避免阻碍用户体验。

**提到的链接**：

- [无标题](https://docs.perplexity.ai/home)：未找到描述
- [Google 计划很快宣布其下一个 Gemini 模型](https://www.theverge.com/2024/10/25/24279600/google-next-gemini-ai-model-openai-december)：12 月正成为 OpenAI 和 Google 展开 AI 发布对决的月份。
- [Search](https://truthcheck.pages.dev/)：未找到描述
- [Yuki Kaguya Shinomiya GIF - Yuki Kaguya Shinomiya - 发现并分享 GIF](https://tenor.com/view/yuki-kaguya-shinomiya-gif-24132804)：点击查看 GIF
- [来自 Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/aravsrinivas/status/1849879578124353774?s=61)：@laqd99 很快
- [Chost Machine GIF - Chost Machine Ai - 发现并分享 GIF](https://tenor.com/view/chost-machine-ai-type-typing-gif-481421031430735140)：点击查看 GIF
- [来自 Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/aravsrinivas/status/1850736323369554311?s=61)：我们将共同构建和调试未来人们提问、获取答案以及解决问题的方式
- [来自 Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1850919619629854875)：据报道，Meta 正在构建一个由工程经理 Xueyuan Su 领导的网页搜索引擎，该项目已进行至少 8 个月，旨在摆脱对 Google Search 和 Bing 数据源的依赖，涉及包括纽约时报在内的一些网站...
- [无标题](https://docs.perplexity.ai/')：未找到描述

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1299652085788311562) (21 messages🔥):

> - `800-Year-Old Well Man` (800 年前的井中人)
> - `Haunted Houses` (闹鬼屋)
> - `Carbon Capture Technology` (碳捕集技术)
> - `Web3`
> - `Culinary Health` (烹饪健康)

- **探索 800 年前井中人的故事**：一名成员分享了关于 **800-Year-Old Well Man** 的链接，揭示了历史的细微差别。
  
  - 未提供更多细节，但该话题似乎对历史感兴趣的人很有吸引力。
- **揭秘 NE 的闹鬼屋**：几位成员讨论了 NE 的 **haunted houses**，并分享了一个列出多个以恐怖名声著称的地点的链接。
  
  - 该链接可以作为寻求刺激者或对超自然现象感兴趣的人的指南。
- **碳捕集：新前沿**：围绕一份关于 **粉末如何捕集碳** 的技术报告展开了讨论，这可能会影响环保工作。
  
  - 成员们对该技术对气候变化策略的影响表示好奇。
- **了解 Web3**：成员们似乎渴望了解 **Web3**，分享了关于其对数字化格局影响的见解资源。
  
  - 这一话题可能会吸引那些想要探索新兴网络技术的人。
- **探索最健康的食用油**：一位成员发布了关于 **最健康食用油** 的查询，表明了对营养和烹饪实践的兴趣。
  
  - 此类讨论突显了人们对健康饮食选择的意识日益增强。

 

**提到的链接**：[YouTube](https://www.youtube.com/embed/poyJ8R6Dfaw)：未找到描述

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1299533207275831348) (11 messages🔥):

> - `Getting sources from the API` (从 API 获取来源)
> - `Requesting Perplexity API results` (请求 Perplexity API 结果)
> - `Access to citations closed beta` (引用功能内测访问权限)
> - `Communication on request status` (关于请求状态的沟通)

- **从 Perplexity API 获取来源**：成员们讨论了如何获取 Perplexity API 结果的来源，其中一人链接到了一个可能包含有用信息的 [Discord 消息](https://discord.com/channels/1047197230748151888/1161802929053909012/1286596437768933376)。
  
  - 一位用户确认他们寻求复制类似于 Perplexity 聊天中的结果，并询问哪个模型能实现该目标。
- **了解 Perplexity API 模型差异**：针对关于哪个模型返回与 Perplexity 聊天类似结果的问题，一位成员指向了一个解释差异的 [FAQ 链接](https://docs.perplexity.ai/faq/faq#why-are-the-results-from-the-api-different-from-the-ui)。
  
  - 此次讨论源于用户关于从 API 获取匹配结果的咨询。
- **关于引用功能内测访问权限的问题**：一位用户表达了为其项目获取引用功能（citations）内测访问权限的紧迫性，表示这至关重要。
  
  - 在查看了置顶消息中的信息后，该用户指出关于其请求状态缺乏沟通。
- **对请求时间线的担忧**：在后续消息中，同一位用户对不知道自己在请求流程中所处的位置（无论是候补名单还是涉及的时间线）表示沮丧。
  
  - 尽管请求了访问权限，但他们对团队关于其查询的沟通感到不确定。

**提到的链接**：

- [无标题](https://docs.perplexity.ai/faq/faq#why-are-the-results-from-the-api-different-from-the-ui)：未找到描述
- [无标题](https://perplexity.mintlify.app/api-reference/chat-completions)：未找到描述

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1299452746377723914) (206 messages🔥🔥):

> - `Ultrasonic Sound Device Sales` (超声波设备销售)
> - `Logistic Growth Curve Analysis` (逻辑增长曲线分析)
> - `AI Distillation Techniques` (AI 蒸馏技术)
> - `AI Technical Newsletters` (AI 技术通讯)
> - `DisTrO GitHub Contributions` (DisTrO GitHub 贡献)

- **超声波设备的销售增长**：一名开发者用于驱鼠的超声波设备在目标群体中的销售额从 15% 上升到 28%，假设符合逻辑增长曲线。
  
  - 在逻辑函数计算中的 A 值和 b 值方面存在困惑，通过在上下文中重新定义时间变量得以解决。
- **了解 AI 蒸馏过程**：围绕蒸馏模型的讨论重点介绍了 Arcee 的 Llama-3.1 模型，该模型利用来自较大模型的 logits 来高效训练较小版本，从而利用“grokked”特征。

- 有人对 Meta 缺乏关于其 Distillation 过程的详细技术文档表示担忧，这引发了对其训练 Pipeline 的进一步讨论。
- **对 AI 技术通讯的兴趣**：一位用户寻求针对技术受众而非面向消费者炒作（Hype）的 AI Newsletter 推荐，并对目前的产品表示不满。
  
  - 用户对 AI 新闻中的炒作情绪产生共鸣，指出有时这些“炒作”涵盖的是之前已知的进展。
- **对 DisTrO 项目的潜在贡献**：一位成员询问了为 DisTrO GitHub 做出贡献的机会，表达了帮助该项目的兴趣。
  
  - 频道对该询问做出了积极回应，表明了社区内的协作精神。
- **AI 在编程中的局限性**：分享了关于某些 AI 模型在编程任务中表现的担忧，特别指出它们可能需要针对代码相关数据进行额外训练。
  
  - 讨论承认了 AI 工具在技术能力方面不足的普遍问题，这与更广泛的社区担忧相一致。

**提到的链接**：

- [conversation_1719549125_scenario_backrooms-sonnet-opus.txt • infinite backrooms](https://dreams-of-an-electric-mind.webflow.io/dreams/conversation-1719549125-scenario-backrooms-sonnet-opus-txt)：两个 AI 之间的对话。
- [Claude Artifact](https://claude.site/artifacts/2de89bcb-acb2-4976-9684-458f0a62ef72)：试用 Claude 用户创建的 Artifacts。
- [Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)：未找到描述。
- [Arcee-SuperNova: Training Pipeline and Model Composition](https://blog.arcee.ai/arcee-supernova-training-pipeline-and-model-composition/)：我们使用智能 Distillation、新型 Post-training 和模型合并技术，将 Arcee SuperNova-70B 和 Arcee SuperNova-8B 训练为具有通用智能的 Llama-3.1-405B 衍生模型。
- [Tweet from TDM (e/λ) (@cto_junior)](https://x.com/cto_junior/status/1849865561955762580?s=46)：平息了所有关于 Sonnet 因为 Feature Steering 而变得更聪明的理论。
- [Bi-Weekly vLLM Office Hours - Neural Magic](https://neuralmagic.com/community-office-hours/)：了解 vLLM，深入探讨激动人心的话题，提出问题并与 vLLM 社区互动。
- [Tweet from Gabriel Elbling (@gabeElbling)](https://x.com/gabeElbling/status/1850220333631943068)：神经网络的惊人图形表示，从未见过类似的东西。
- [Not A Tunnell Its Dimming GIF - Not A Tunnell Its Dimming Dark Tunnel - Discover & Share GIFs](https://tenor.com/view/not-a-tunnell-its-dimming-dark-tunnel-gif-14376722)：点击查看 GIF。
- [no title found](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)：未找到描述。
- [From Black Holes Entropy to Consciousness: The Dimensions of the Brain Connectome](https://www.youtube.com/watch?v=mlCI5DJM5gc)：所提供的文本摘自一篇科学文章，探讨了意识、大脑 Connectome 以及时空概念之间的联系...
- [[Distributed w/ TorchTitan] Introducing Async Tensor Parallelism in PyTorch](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)：与 Horace He, Less Wright, Luca Wehrstedt, Tianyu Liu, Wanchao Liang 一起。TL;DR 我们在 PyTorch 中实现了实验性的 Async Tensor Parallelism 支持。我们将其集成到 TorchTitan 中并观察到：...
- [Tweet from Jiaxin Pei (@jiaxin_pei)](https://x.com/jiaxin_pei/status/1849142143186800719?s=46)：在 System Prompts 中添加 Persona 很常见，假设这可以帮助 LLM。然而，通过分析 162 个角色 x 4 个 LLM x 2410 个问题，我们表明添加 Persona 大多在统计上*没有*影响...
- [meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8 · Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8)：未找到描述。
- [Current team tasks • QubesOS](https://github.com/orgs/QubesOS/projects/19)：当前团队任务。

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1299471355531497523) (13 条消息🔥):

> - `Hermes 3 SFT Dataset`
> - `Nous 模型的训练与推理`
> - `Runpod 与 Modal 的性能`
> - `DRY Sampler 的实现`

- **Hermes 3 SFT 数据集未开源**：成员们确认 Hermes 3 SFT 数据集**尚未开源**，这与包括 Hermes 1 和 2 在内的先前版本不同。
  
  - 不过，文中分享了 [Hugging Face 上的 OpenHermes-2.5 数据集](https://huggingface.co/datasets/teknium/OpenHermes-2.5)链接作为相关资源。
- **微调 Nous 模型的最佳平台**：一位成员建议使用 **Runpod** 进行训练和推理，并提到由于冷启动时间较长，需要**更好的实时推理**方案。
  
  - 他们还表示有兴趣在 **Fireworks** 上测试性能，并指出 **Runpod** 的 **devex** 在 **function calling** 方面存在局限性。
- **Modal 的冷启动时间影响 Nous 模型**：成员们对 **Modal 的冷启动时间**表示担忧，认为这阻碍了 **Nous 模型**的性能表现。
  
  - 一位成员希望 **Groq** 能支持自定义微调部署，将其作为一种潜在的解决方案。
- **机器人需要集成 DRY sampler**：一位成员指出最近有一个用于集成 **DRY sampler** 的 [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/9702)，表明这可能有利于提升机器人的性能。
  
  - 他们指出，该实现正等待合并到 **vllm** 或用于 **Hermes 3** 推理的类似工具中。

**提到的链接**：

- [teknium/OpenHermes-2.5 · Datasets at Hugging Face](https://huggingface.co/datasets/teknium/OpenHermes-2.5)：未找到描述
- [由 wwoodsTM 添加的 DRY sampler 实现（重构后）· Pull Request #9702 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/9702)：我已阅读贡献指南。自报审查复杂度：低 中 高。这是之前 PR #6839 的延续，旨在实现 DRY sampling（我认为这是一个相当忠实的...

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1299782302670389279) (4 条消息):

> - `Thinking LLMs`
> - `Medical AI Developments`
> - `Dualformer Integration`
> - `Thought Preference Optimization`
> - `Medical AI Podcast`

- **Thinking LLMs 提出有效的指令遵循**：论文 *Thinking LLMs: General Instruction Following With Thought Generation* 介绍了 Thought Preference Optimization (TPO)，旨在通过 Chain-of-Thought 风格的提示来增强经过指令微调（instruction-finetuned）的 LLM。
  
  - 结果表明，TPO 显著提升了性能，同时也揭示了在没有进行适当微调的情况下使用思考提示（thought prompt）可能会导致结果变差。
- **Medical AI 播客发布最新突破**：该播客集讨论了 Medical AI 的最新进展，重点关注 **deepfake 检测**和**可解释 AI**（explainable AI）等主题。
  
  - 观看标题为 *Latest Medical AI Breakthroughs* 的完整剧集，深入了解新研究和技术的见解。
- **引入 Dualformer 以增强推理能力**：*Dualformer* 模型通过在训练期间利用随机推理轨迹（randomized reasoning traces），集成了快速和慢速推理模式，从而提高了效率和性能。
  
  - 在测试中，Dualformer 在慢速模式下的迷宫导航任务中实现了高达 **97.6%** 的成功率，在减少计算量的同时优于现有的 Baseline。
- **Medical LLM 取得全面进展**：最近的发展包括各种医疗模型，如用于词汇理解的 **BioMistral-NLU** 和用于 CT 肿瘤模型的 **ONCOPILOT**。
  
  - 这些创新展示了 AI 在医疗保健领域不断扩大的作用，并为临床分析带来了新的能力。
- **关于 Thought Preference Optimization 的讨论**：在关于 TPO 的对话中，一位社区成员反思了其与 OpenAI 模型相比的功效，强调了思考在指令遵循中的作用。
  
  - 讨论强调了理解模型提示对于在 LLM 应用中保持响应准确性的重要性。

**提到的链接**：

- [Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces](https://arxiv.org/abs/2410.09918v1)：在人类认知理论中，人类思维受两个系统支配：快速且直觉的 System 1，以及较慢但更深思熟虑的 System 2。最近的研究表明，结合 System...
- [来自 Sebastian Raschka (@rasbt) 的推文](https://x.com/rasbt/status/1850177459930497118?s=46)：我刚刚阅读了 "Thinking LLMs: General Instruction Following With Thought Generation" 论文 (I)，它提供了一种简单而有效的方法来提高指令微调（instruction-finetun...）的响应质量。
- [🤖 Latest Medical AI Breakthroughs | 每周研究回顾 10月19-26日 (Part 1/2)](https://youtu.be/Wt5QOv1vk2U)：本周的 Medical AI 研究中，我们探讨了在 deepfake 检测、可解释 AI (XAI)、LLM 应用和多模态基础模型（multimodal foundation mod...）方面的突破性工作。

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1299636074460483627) (4 条消息):

> - `Ferret-UI release`
> - `Homomorphic Encryption announcement`
> - `Apple's AI Cloud security initiative`

- **Apple 为 iOS 发布 Ferret-UI**：两周前，@Apple 发布了 **Ferret-UI** 的权重，这是一款专为 **iPhone/iOS** 屏幕量身定制的新型**多模态 LLM**，为用户简化了与 [@HuggingFace transformers](https://huggingface.co) 的集成。
  
  - *Ferret-UI 擅长移动端 UI 屏幕理解*，具备**图标识别**和**文本定位**等能力，表现甚至超越了 **GPT-4V**。
- **Apple 在密码学领域的重大突破**：据 @varun_mathur 称，Apple 宣布了一种革命性的 **Homomorphic Encryption**（同态加密）方法，该方法在保护私有数据的一致性时，允许其以匿名方式用于改善用户体验。
  
  - 这项技术允许私有数据保持**端到端加密**，促进了一种**非零和博弈**，即个人用户数据可以为更广泛的 UX 做出贡献。
- **AI Cloud 安全的高额悬赏**：在 Apple 的 **Private AI Cloud** 发布之前，该公司宣布将向研究人员支付**高达 100 万美元**的奖金，用于识别其安全性漏洞。
  
  - 该公告包括对报告能够执行恶意代码的漏洞利用 (exploits) 的奖励，而报告敏感数据漏洞的奖励最高可达 **250,000 美元**。

**提到的链接**：

- [Jade Choghari (@jadechoghari) 的推文](https://x.com/jadechoghari/status/1849840373725597988?s=46)：2 周前 @Apple 发布了 Ferret-UI 的权重 -> 一款专门为 iPhone/IOS 屏幕制作的新型多模态 LLM ！！我参与了 HF 的集成工作 - 现在已在 @HuggingFace transformers 中可用 - ...
- [Apple 将支付安全研究人员高达 100 万美元来攻击其私有 AI 云 | TechCrunch](https://techcrunch.com/2024/10/24/apple-will-pay-security-researchers-up-to-1-million-to-hack-its-private-ai-cloud/)：在 Apple 下周推出名为 Private Cloud Compute 的私有 AI 云之前，这家科技巨头表示将向安全研究人员支付高达...
- [Varun (@varun_mathur) 的推文](https://x.com/varun_mathur/status/1850502044307562871?s=46)：这是 Apple 围绕密码学发布的一个改变游戏规则的公告。在某些方面，它是“AI 的 HTTPS 时刻”.. 这意味着：你的私人机密数据可以与其他数据汇集在一起...

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1299782302670389279) (4 messages):

> - `Thought Preference Optimization`
> - `Medical AI Podcast`
> - `Dualformer Model`
> - `Medical LLM Applications`
> - `AI in Healthcare Ethics`

- **思维偏好优化增强了 LLM 的响应能力**：论文《Thinking LLMs: General Instruction Following With Thought Generation》提出了 **Thought Preference Optimization (TPO)**，在 LLM 训练中引入了 Chain-of-Thought 提示，以提高指令响应的质量。
  
  - 结果显示，在 **Llama 3 8B Instruct** 模型上应用 TPO 后，性能提升了 **4%**，此前的微调也带来了显著收益。
- **每周医疗 AI 洞察现已发布**：每周医疗 AI 播客讨论了突破性的进展，包括 **deepfake detection**（深度伪造检测）和 **explainable AI**（可解释 AI），帮助听众高效了解最新动态。
  
  - 重点介绍的模型包括用于医疗词汇理解的 **BioMistral-NLU**，以及用于 CT 肿瘤分析的 **ONCOPILOT** 等应用。
- **引入用于灵活推理的 Dualformer**：**Dualformer** 模型集成了快慢推理模式，在降低计算成本的同时显著增强了 LLM 的推理能力。
  
  - 在测试中，Dualformer 在慢速模式下完成了 **97.6%** 的迷宫导航任务，在减少 **45.5% 推理步骤** 的情况下超越了之前的基准线。
- **多样化的医疗 LLM 发展**：最近的进展包括专为生物医学任务设计的 **Bilingual Multimodal LLM** 以及旨在进行临床分析的 **Metabolic-Enhanced LLMs**。
  
  - 这些进展强调了在医疗各个领域的应用，包括个性化健康计划和评估方法论。
- **探索医疗 AI 中的伦理考量**：围绕 **AI in healthcare ethics** 的讨论强调了偏见分析和具备反思意识的临床 Agent 的重要性，以确保 AI 使用的公平性和准确性。
  
  - 诸如 **Healthcare XAI Through Storytelling** 等项目旨在提高医疗专业人员对 AI 应用的理解和信任。

**提到的链接**：

- [来自 Sebastian Raschka (@rasbt) 的推文](https://x.com/rasbt/status/1850177459930497118?s=46)：我刚刚阅读了《Thinking LLMs: General Instruction Following With Thought Generation》论文 (I)，它提供了一种简单而有效的方法来提高指令微调（instruction-finetun...）的响应质量。
- [Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces](https://arxiv.org/abs/2410.09918v1)：在人类认知理论中，人类思维受两个系统支配：快速且直觉的系统 1，以及较慢但更深思熟虑的系统 2。最近的研究表明，引入系统...
- [🤖 最新医疗 AI 突破 | 每周研究回顾 10月19-26日 (Part 1/2)](https://youtu.be/Wt5QOv1vk2U)：本周的医疗 AI 研究中，我们探索了深度伪造检测、可解释 AI (XAI)、LLM 应用以及多模态基础模型方面的突破性工作...

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1299456278489337898) (99 messages🔥🔥):

> - `Training LLMs on Limited Resources`
> - `Contributing to Open Source AI Projects`
> - `Optimizers in Machine Learning`
> - `Running Models on Multiple GPUs`
> - `Community Support for AI Frameworks`

- **在有限资源下训练 LLM**：成员们讨论了在有限的 GPU 资源上训练 LLaMA-2 或 Pythia 等大语言模型 (LLMs) 的挑战，强调许多论文需要大量的硬件才能进行有效的复现。
  
  - 一位成员分享了他们部署 nanoGPT 作为轻量级模型以简化训练的经验，尽管它很简单，但仍建议初学者尝试。
- **贡献开源 AI 项目**：一位用户表示尽管主要拥有专有技术经验，但仍有兴趣参与 EleutherAI 的工作，并征求关于贡献项目的建议。
  
  - 回复指出，开源贡献（尤其是针对较小项目的贡献）可以提供宝贵的学习经验，并有助于从软件工程背景进行转型。
- **机器学习中的优化器**：关于 Shampoo 优化器的讨论强调了它与 Adam 等其他优化器相比缺乏动量的问题，同时质疑了它在各种库中的实现。
  
  - 成员们提到了提供 Shampoo 优化器最新见解的论文，从而对其功能进行了更深入的探索。
- **在多 GPU 上运行模型**：关于使用 GPT-NeoX 等框架在多 GPU 上进行训练的复杂性引起了关注，特别是关于 CUDA 通信效率和项目依赖关系的问题。

- 建议在转向更复杂的代码库之前，先使用像 nanoGPT 这样更简单的框架进行小规模实验。
- **AI 框架的社区支持**：参与者指出，在处理不同的 AI 代码库时，社区和文档的重要性，特别是在处理系统要求和依赖管理时。
  
  - 一位用户提到了对 PyTorch Lightning 领域中被遗弃项目的沮丧，并主张采用更清晰、更简化的代码结构以促进学习。

**提到的链接**：

- [SymNoise: Advancing Language Model Fine-tuning with Symmetric Noise](https://arxiv.org/abs/2312.01523)：在本文中，我们介绍了一种针对语言模型的新型微调技术，涉及在嵌入过程中加入对称噪声。该方法旨在增强模型的功能...
- [EleutherAI/gpt-neox - Sourcegraph](https://sourcegraph.com/github.com/EleutherAI/gpt-neox)：未找到描述
- [Some testing from me · Issue #407 · pytorch/torchtitan](https://github.com/pytorch/torchtitan/issues/407)：我尝试了 torchtitan。这里有一堆问题。我的环境是 CoreWeave 提供的 PyTorch nightly 镜像，运行在 CW 托管的 slurm HGX 上。PyTorch 和 torchtitan 的版本是几天前的 nightly 构建....
- [GitHub - facebookresearch/lingua: Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs.](https://github.com/facebookresearch/lingua)：Meta Lingua：一个精简、高效且易于修改的 LLM 研究代码库。- facebookresearch/lingua
- [GitHub - tysam-code/hlb-gpt: Minimalistic, extremely fast, and hackable researcher's toolbench for GPT models in 307 lines of code. Reaches <3.8 validation loss on wikitext-103 on a single A100 in <100 seconds. Scales to larger models with one parameter change (feature currently in alpha).](https://github.com/tysam-code/hlb-gpt)：极简、极速且可修改的 GPT 模型研究工具台，仅需 307 行代码。在单台 A100 上，100 秒内即可在 wikitext-103 上达到 <3.8 的验证损失。只需更改一个参数即可扩展到更大的模型（该功能目前处于 alpha 阶段）。
- [GitHub - tomogwen/LitGPT: Minimal GPT Implementation in PyTorch Lightning](https://github.com/tomogwen/LitGPT.git)：PyTorch Lightning 中的极简 GPT 实现。通过在 GitHub 上创建账号来为 tomogwen/LitGPT 的开发做出贡献。
- [Megatron-LM/megatron/core/distributed/distributed_data_parallel.py at main · NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/distributed_data_parallel.py)：关于大规模训练 Transformer 模型的持续研究 - NVIDIA/Megatron-LM
- [pythia/models/14M/pythia-14m.yml at main · EleutherAI/pythia](https://github.com/EleutherAI/pythia/blob/main/models/14M/pythia-14m.yml)：EleutherAI 关于可解释性和学习动力学工作的枢纽 - EleutherAI/pythia
- [GitHub - tomogwen/LitGPT: Minimal GPT Implementation in PyTorch Lightning](https://github.com/tomogwen/LitGPT)：PyTorch Lightning 中的极简 GPT 实现。通过在 GitHub 上创建账号来为 tomogwen/LitGPT 的开发做出贡献。
- [GitHub - google-research/google-research at 2a1b8acddba7ed1bcf308427586564322a66225a](https://github.com/google-research/google-research/blob/2a1b8acddba7ed1bcf30842758656432)：Google Research。通过在 GitHub 上创建账号来为 google-research/google-research 的开发做出贡献。
- [google-research/scalable_shampoo/optax/distributed_shampoo.py at 2a1b8acddba7ed1bcf308427586564322a66225a · google-research/google-research](https://github.com/google-research/google-research/blob/2a1b8acddba7ed1bcf308427586564322a66225a/scalable_shampoo/optax/distributed_shampoo.py#L1266)：Google Research。通过在 GitHub 上创建账号来为 google-research/google-research 的开发做出贡献。

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1299752524802293882) (89 条消息🔥🔥):

> - `Stick-Breaking Attention Mechanism`
> - `Differential Transformers`
> - `Counting in Transformers`
> - `Latent Collaborative Recommendations`
> - `Universal Transformers Scaling`

- **提出创新的 Stick-Breaking Attention**：一种基于 stick-breaking 过程的新型 attention 机制旨在改进 Transformers 的 self-attention，解决与 positional embeddings 和 softmax 限制相关的问题，详见 [arXiv 论文](https://arxiv.org/abs/2410.17980)。
  
  - 社区反馈建议此类机制需要更清晰的介绍，同时分享了相关项目的链接，如 [IBM 的 ModuleFormer](https://github.com/IBM/ModuleFormer)。
- **关于 Differential Transformers 的讨论**：针对 [Differential Transformer](https://arxiv.org/pdf/2410.05258) 论文中使用的 GroupNorm 与 RMSNorm 产生了疑问，指出了方法论中可能存在的混淆。
  
  - 社区成员澄清了相关术语以及归一化方法对每个 attention head 的影响。
- **探索 Transformers 中的计数能力**：最近一篇关于 Transformers 的论文强调了影响推理深度的架构限制，特别是在计数任务中，引发了对通用 LLM 能力的质疑，如 [arXiv 讨论](https://arxiv.org/abs/2410.19730)所述。
  
  - 社区对先前已有的工作以及推理任务中 tokenization 方法的相似性表现出兴趣，表明该领域正在进行持续探索。
- **LC-Rec 的优化策略**：一位用户就如何优化 Latent Collaborative Recommendations (LC-Rec) 方法寻求社区建议，重点是为其硕士项目增强性能和进行方法微调。
  
  - 建议包括参考基准论文以及探索反映社区标准的优化方案。
- **Universal Transformers 扩展的更新**：一位成员报告了他们在扩展 Universal Transformers 方面的努力，表达了在计算资源受限的情况下提高性能所面临的挑战。
  
  - 对话围绕如果扩展问题在未来的研究中得以解决，取得理想结果的潜力展开。

**提到的链接**：

- [Value Residual Learning For Alleviating Attention Concentration In Transformers](https://arxiv.org/abs/2410.17897)：Transformers 可以利用 self-attention 捕捉长程依赖，允许 token 直接关注所有其他 token。然而，堆叠多个 attention 层会导致注意力集中。O...
- [Democratic Representations](https://arxiv.org/abs/1401.3420)：在约束条件下最小化 $\ell_{\infty}$（或最大）范数，以使欠定线性方程组保持一致性，这在许多实际应用中都有用途...
- [when trees fall...](https://blog.wtf.sg/)：未找到描述
- [Counting Ability of Large Language Models and Impact of Tokenization](https://arxiv.org/abs/2410.19730)：Transformers 作为现代大语言模型 (LLMs) 的骨干，面临着阻碍其推理能力的固有架构限制。与循环网络不同，Transformers 缺乏递归...
- [Stick-breaking Attention](https://arxiv.org/abs/2410.17980)：self-attention 机制传统上依赖于 softmax 算子，需要 RoPE 等 positional embeddings 或位置偏差来考虑 token 顺序。但目前使用的方法仍然...
- [Dirichlet process - Wikipedia](https://en.m.wikipedia.org/wiki/Dirichlet_process#The_stick-breaking_process)：未找到描述
- [GitHub - IBM/ModuleFormer: ModuleFormer is a MoE-based architecture that includes two different types of experts: stick-breaking attention heads and feedforward experts. We released a collection of ModuleFormer-based Language Models (MoLM) ranging in scale from 4 billion to 8 billion parameters.](https://github.com/IBM/ModuleFormer)：ModuleFormer 是一种基于 MoE 的架构，包含两种不同类型的专家：stick-breaking attention heads 和 feedforward 专家。我们发布了一系列基于 ModuleFormer 的语言模型 (MoLM)，规模从 40 亿到 80 亿参数不等。
- [Sparse Universal Transformer](https://arxiv.org/abs/2310.07096)：Universal Transformer (UT) 是 Transformer 的一种变体，在各层之间共享参数。经验证据表明，UT 比 Vanilla Transformer 具有更好的组合泛化能力...

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1299512945503637545) (4 messages):

> - `Mech Interp Research Critique` (机械可解释性研究批判)
> - `Definition of Feature in SAEs` (SAEs 中特征的定义)
> - `Outdated Task List` (过时的任务列表)
> - `Apollo Research Project Ideas` (Apollo Research 项目想法)
> - `Future Projects in MI` (MI 的未来项目)

- **Mech Interp Research Critique**: 一位成员分享了一份[项目提案](https://docs.google.com/document/d/1hELXy1WZQVJfkPVvbjRiMX3xiuyYG6VG1t8GiXiS8Ys/edit?tab=t.0)，批评小型研究小组和大型 AI 实验室的 Mech Interp 研究缺乏**科学诚信（scientific integrity）**，并提出了改进建议。
  
  - “我们应该做得更好”是其批判的核心观点。
- **关于 SAEs 中特征定义的辩论**: 有人对将“**特征（feature）**”定义为“在输入/输出某些文本时，矩阵中始终具有较大点积的一列数字”的准确性提出了疑问。
  
  - 这引发了关于 SAEs 背景下特征定义的细微差别和解释的讨论。
- **对过时任务列表的担忧**: 一位新人注意到 Notion 文档中的**未领取任务部分（unclaimed tasks section）**可能已经过时，并询问最近是否有更新。
  
  - 他们还表示，Neel 的 **200 个问题**列表似乎也有些过时。
- **Apollo Research 的新项目想法**: 一位成员分享了由 Apollo Research 团队生成的 [45 个 Mech Interp 项目想法列表](https://www.alignmentforum.org/posts/KfkpgXdgRheSRWDy8/a-list-of-45-mech-interp-project-ideas-from-apollo-research)，旨在识别新的潜在项目。
  
  - 他们指出之前的列表已经过时，并强调了团队面临的计算资源限制。
- **寻求参与未来的 MI 项目**: 另一位成员表达了希望转向 **Mechanistic Interpretability** 研究的愿望，并寻求有关可以加入的正在进行或即将开展的项目信息。
  
  - 他们请求其他成员对该消息做出回应，以便就潜在的项目机会进行直接沟通。

**提及的链接**:

- [来自 Alice Rigg (@woog09) 的推文](https://x.com/woog09/status/1849905383906627784): 我在周日写了这份项目提案。简而言之，我批评了 Mech Interp 研究，称小型团体和大型 AI 实验室都缺乏科学诚信，我们应该做得更好，并建议了一些方法……
- [来自 Apollo Research 可解释性团队的 45+ 个 Mech Interp 项目想法列表 — AI Alignment Forum](https://www.alignmentforum.org/posts/KfkpgXdgRheSRWDy8/a-list-of-45-mech-interp-project-ideas-from-apollo-research): 我们制作这个列表的原因：• Apollo Research 的可解释性团队最近完成了几个项目。为了决定接下来的工作方向……

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1300184437924823090) (1 messages):

> - `Eval Tasks Limit` (评估任务限制)
> - `Accuracy Measurement` (准确率测量)

- **确定有效的评估任务限制**: 一位成员询问了一个合适的 `--limit` 参数值，以便在不执行所有评估任务的情况下有效测量准确率，因为完整评估非常耗时。
  
  - 他们建议将 `--limit 100` 作为一个潜在值，并寻求关于这是否足以实现忠实且准确测量的见解。
- **对受限任务评估的担忧**: 另一项讨论强调了对使用受限评估任务来衡量模型性能的担忧，因为这可能无法产生全面的结果。
  
  - 成员们辩论了在模型可靠性背景下，使用较低的限制对评估准确性的影响。

---

### **Eleuther ▷ #**[**gpt-neox-dev**](https://discord.com/channels/729741769192767510/730090096287547444/1299998183291883571) (10 条消息🔥):

> - `GPT-NeoX Colab Notebooks`
> - `针对 LLM 的分布式 GPU 训练`
> - `Python 3.10 兼容性`
> - `Llama3.x YAML 配置`

- **开发中的 GPT-NeoX Colab Notebooks**：一位用户旨在创建 GPT-NeoX 在 Colab notebooks 中的简单示例，重点关注小型 GPT 模型和用于代码补全的 Python 数据。他们分享了规范并寻求社区反馈，包括 GitHub 上的规范链接。
  
  - 提供了两个详细说明任务的 notebook 链接：[notebook1](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebook1task.md) 和 [notebook2](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebook2task.md)。
- **关于分布式 GPU 训练的关注**：一位用户询问是否有共享消费级 GPU 以使用 GPT-NeoX 训练大语言模型的设置。回复指出由于网络问题存在困难，并建议参考 INTELLECT-1 的类似尝试。
  
  - 分享了 INTELLECT-1 的链接，强调了该领域正在进行的一些工作：[PrimeIntellect](https://app.primeintellect.ai/intelligence)。
- **Python 3.10 兼容性修复**：在 Python 3.10 中设置 GPT-NeoX 需要将 Torch 版本覆盖为 `1.11.0` 以避免导入失败。一位用户分享了一个 Colab notebook，记录了安装问题以及应用修复后的解决过程。
  
  - 他们注意到后续版本的 Torch 可能存在兼容性问题，建议 **torch 2.3** 可能是可以接受的，而 **2.4** 会导致失败。
- **为 Python 兼容性修补 DeepSpeed**：讨论揭示了与 `deepspeed/elasticity/elastic_agent.py` 中的补丁相关的问题，这些问题影响了 GPT-NeoX 对 Python 3.10 的支持。用户分享了解决这些兼容性问题的相关 GitHub pull requests 链接。
  
  - 可以在此处找到与此问题相关的特定 PR：[PR #62](https://github.com/EleutherAI/DeeperSpeed/pull/62/files)，展示了为提高兼容性而进行的持续工作。
- **Llama3.x 模型的 YAML 配置**：一位用户询问是否可以将 Llama3.1/3.2 等较小指令模型的 YAML 配置添加到 GPT-NeoX 仓库中。他们还询问是否可以从 HuggingFace 文件自动生成这些配置，并提到了另一位用户的贡献。
  
  - 该询问包含了一个指向建立 Llama2 现有配置文件的 commit 引用：[commit 链接](https://github.com/AIproj/gpt-neox/commit/594d926cbd4ef5f392d9985dc72967408d20fe07)。

**提到的链接**：

- [INTELLECT-1 | Prime Intellect | 10B 模型的去中心化训练](https://app.primeintellect.ai/intelligence)：首个 100 亿参数模型的去中心化训练运行，邀请任何人贡献算力并参与。
- [GPT-NeoX-Colab/notebooks/neox_failing_python3.10.ipynb at main · markNZed/GPT-NeoX-Colab](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebooks/neox_failing_python3.10.ipynb)：GPT-NeoX 的示例 Colab notebooks。通过在 GitHub 上创建账户为 markNZed/GPT-NeoX-Colab 的开发做出贡献。
- [GPT-NeoX-Colab/notebook1task.md at main · markNZed/GPT-NeoX-Colab](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebook1task.md)：GPT-NeoX 的示例 Colab notebooks。通过在 GitHub 上创建账户为 markNZed/GPT-NeoX-Colab 的开发做出贡献。
- [GPT-NeoX-Colab/notebook2task.md at main · markNZed/GPT-NeoX-Colab](https://github.com/markNZed/GPT-NeoX-Colab/blob/main/notebook2task.md)：GPT-NeoX 的示例 Colab notebooks。通过在 GitHub 上创建账户为 markNZed/GPT-NeoX-Colab 的开发做出贡献。
- [Python 3.10 support by markNZed · Pull Request #1313 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/pull/1313)：在此 issue 中添加了 Python 3.10 支持 #1122
- [[BE] minor logging cleanup in distributed (#122921) · pytorch/pytorch@b6201a6](https://github.com/pytorch/pytorch/commit/b6201a60c57f79d081ecde66e1ce70e6614a3052#diff-0332dff4db45fd26518da1eddd4a93eb965a006de743edddba2316b64e2b1f6c))：摘要：分布式库中的少量日志清理 1. 不要使用 "f" 格式化字符串 - 解决 linter 问题。2. 细节：利用未使用的 `e` (error)...
- [Merge upstream torch-compatibility changes for elastic_agent by jacobthebanana · Pull Request #62 · EleutherAI/DeeperSpeed](https://github.com/EleutherAI/DeeperSpeed/pull/62/files))：从 DeepSpeed 上游拉取以下更改。microsoft@a4cd550 microsoft@6a4b96a 相关讨论：EleutherAI 频道中的消息。针对 torch==2.5.0 进行了测试。

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1299464039419871295) (197 条消息🔥🔥):

> - `Stable Diffusion 3.5 使用`
> - `在 Runpod 上部署自定义模型`
> - `使用 AMD GPU 进行本地生成`
> - `建筑设计中的草图转渲染`
> - `用于 Flux inpainting 的 Discord 机器人`

- **Stable Diffusion 3.5 性能担忧**：用户分享了关于 **Stable Diffusion 3.5** 褒贬不一的使用体验，质疑其与 **1.5** 等早期版本相比的速度和质量。
  
  - 建议在不同模型上运行相同的 prompt，以有效地比较结果。
- **在 Runpod 上部署自定义模型**：一位用户询问了如何在 **Runpod** 上部署自定义 Stable Diffusion 模型 (Juggernaut)，并讨论了缺乏 Forge 模板的问题。
  
  - 其他人指出，使用像 **Auto1111** 这样更简单的界面可能会更直接。
- **使用 AMD GPU 进行本地生成**：讨论包括了使用 **AMD GPU** 进行本地生成的可能性，并建议参考置顶指南以获取成功经验。
  
  - 参与者分享了他们在 VRAM 容量方面的经验，并建议测试各种模型，如 **Gemma 2**。
- **建筑设计的草图转渲染**：一位用户表示有兴趣将 **Stable Diffusion** 用于建筑设计，重点关注“草图转渲染”工作流。
  
  - 建议包括使用 **ControlNet** 等工具，以在转换过程中保持细节和准确性。
- **为 Flux inpainting 创建 Discord 机器人**：开发者讨论了创建一个能够在 **Flux** 中进行 inpainting 的 Discord 机器人，但提到此类模型在网上的可用性有限。
  
  - 一位参与者表示打算实现支持社区工具内 inpainting 的功能。

**提到的链接**：

- [Stable Diffusion 3.5 (Large): What You Need to Know](https://sandner.art/stable-diffusion-35-large-what-you-need-to-know/)：Stable Diffusion 3.5 (Large) 和 (Turbo) 已发布。与 Flux 对比——掌握低 VRAM 技巧和工作流的快速 AI 生成。教程。
- [Reddit - Dive into anything](https://www.reddit.com/r/CyberpunkFanArt/)：未找到描述。
- [来自 AK (@_akhaliq) 的推文](https://x.com/_akhaliq/status/1849945005269336096)：x.infer 昨晚刚发布了一个新的计算机视觉模型。它被称为 GPT-54o-mini-vision-pro-max-xxxl。这是一个非常酷的模型，开源、开放权重、开放数据，所有优秀的东西...
- [AnythingLLM | 适合所有人的全能 AI 应用程序](https://anythingllm.com/)：AnythingLLM 是你一直在寻找的 AI 应用程序。使用任何 LLM 与你的文档聊天，提高生产力，并在完全私密且无需技术背景的情况下运行最新的 state-of-the-art LLMs...
- [GitHub - open-webui/open-webui: 用户友好的 AI 界面 (支持 Ollama, OpenAI API, ...)](https://github.com/open-webui/open-webui)：用户友好的 AI 界面 (支持 Ollama, OpenAI API, ...) - open-webui/open-webui
- [GitHub - anapnoe/stable-diffusion-webui-ux-forge: Stable Diffusion web UI UX Forge](https://github.com/anapnoe/stable-diffusion-webui-ux-forge)：Stable Diffusion web UI UX Forge。通过在 GitHub 上创建账号来为 anapnoe/stable-diffusion-webui-ux-forge 的开发做出贡献。
- [ComfyUI 工作流 - 开发者社区 | OpenArt](https://openart.ai/workflows/all?keyword=pulid)：在 OpenArt 上发现、分享并运行数千个 ComfyUI 工作流。

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1299451831805542502) (127 条消息🔥🔥):

> - `Aider vs PearAI 讨论`
> - `Claude 1022 使用体验`
> - `Homebrew vs Pipx 安装方式`
> - `Sonnet 3.5 基准测试`
> - `本地模型的隐私问题`

- **Aider 和 PearAI 功能对比**：成员们讨论了 PearAI 和 Aider 之间的相似之处，指出 PearAI 似乎模仿了 Continue.dev 和 Aider 的功能，特别是在与开源工具的集成方面。
  
  - 有人对这种重叠带来的伦理影响表示担忧，并提到了 *Open Source Pledge*，该倡议要求企业为开源工具的开发做出贡献。
- **将 Claude 1022 与 Aider 结合使用**：一位用户分享了将 Claude 1022 与 Aider 结合使用开发复杂 Flutter 应用程序的积极体验，声称生产力得到了显著提升。
  
  - 他们花费了 15 小时和 18 美元的额度，仅通过 Prompting 就生成了 4300 行代码。
- **Aider 的最佳安装方法**：关于使用 Homebrew 安装 Aider 的缺点引发了讨论，指出其可能与依赖项要求产生冲突。
  
  - 许多人主张改用 Pipx，它可以隔离 Aider 的依赖项以获得更好的兼容性。
- **基准测试文件请求**：用户表示需要专门针对 Sonnet 3.5 代码编辑和重构的基准测试数据文件，以避免自行运行昂贵的测试。
  
  - 一位用户特别索要了 `.aider.chat.history.md` 和 `.aider.results.json` 文件。
- **Aider 的隐私担忧**：一位用户询问了在 Aider 中使用本地模型的数据隐私问题，担心敏感信息相关的风险。
  
  - 另一位成员确认，当使用本地模型时，Aider 不会存储用户数据，从而确保了隐私。

**提到的链接**：

- [Introducing PearAI Creator (Beta) — Powered By aider](https://trypear.ai/blog/introducing-pearai-creator-beta)：PearAI Creator 可以自动为你构建应用、修复 Bug 并实现新功能。了解如何使用这个由 aider 驱动的强大新功能。
- [Install with pipx](https://aider.chat/docs/install/pipx.html)：aider 是你终端里的 AI 配对编程工具。
- [Home](http://aider.chat)：aider 是你终端里的 AI 配对编程工具。
- [SF ads call out tech firms for not paying for open source](https://www.theregister.com/2024/10/25/open_source_funding_ads/)：提醒那些“吝啬”的首席技术官们注意。
- [GitHub - caseymcc/aider-code: vscode extension for aider](https://github.com/caseymcc/aider-code)：aider 的 VSCode 扩展。欢迎在 GitHub 上通过创建账号为 caseymcc/aider-code 的开发做出贡献。
- [GitHub - caseymcc/aider at commandio](https://github.com/caseymcc/aider/tree/commandio)：aider 是你终端里的 AI 配对编程工具。欢迎在 GitHub 上通过创建账号为 caseymcc/aider 的开发做出贡献。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1299502885058515015) (60 条消息🔥🔥):

> - `Nvidia Nemotron setup`
> - `Unit Testing with Aider`
> - `VSCode Extensions for Aider`
> - `Gemini Model Performance`
> - `Fine-tuning Models and Context Issues`

- **在 Aider 中设置 Nvidia Nemotron**：一位用户在 **Aider** 上配置 **Nvidia Nemotron** 时遇到困难，正在寻求有关设置和执行命令的指导，特别是关于自定义模型元数据方面。
  
  - 另一位成员提到，如果连接成功，可以忽略模型警告，并鼓励查看文档以处理自定义元数据。
- **增强单元测试覆盖率**：一位用户报告了 **Aider** 生成的 **unit tests**（单元测试）无法编译的问题，并询问有关提高覆盖率和可靠性的建议。
  
  - 回复中包括了对特定 Prompt 的讨论，以及使用 Aider 进行单元测试的经验，一些人建议了改进测试隔离的方法。
- **提升 Aider 效率的 VSCode 扩展**：一位用户提议创建一个 VSCode 扩展，以便轻松地将文本编辑器内容传输到 **Aider**，并可能为不同的命令格式设置按钮。
  
  - 该想法受到了积极响应，强调了在与 Aider 集成时对高效工作流的需求。
- **Gemini 模型在编辑任务中的性能**：成员们讨论了 **Gemini models** 在编辑任务中是否有效，一些人指出 **Gemini Flash** 相比 **Claude 3.5** 具有性能优势。
  
  - 同时也提出了关于使用各种模型时一致性的担忧，强调了在 Aider 中需要更好的文档创建命令。
- **微调模型与上下文限制**：一位用户询问了关于将 Aider 与 **fine-tuned models**（微调模型）配合使用的问题，特别是针对 **AngelScript**，并表达了对错误处理和幻觉函数的挫败感。
  
  - 另一位用户分享了在使用模型时遇到的详细输出问题，包括在使用 Aider 时遇到的 API 速率限制和资源耗尽相关的错误。

**提到的链接**：

- [File editing problems](https://aider.chat/docs/troubleshooting/edit-errors.html)：Aider 是你终端里的 AI 结对编程工具
- [Models: 'google)' | OpenRouter](https://openrouter.ai/models?q=google))：在 OpenRouter 上浏览模型
- [Release v1.44.7 · BerriAI/litellm](https://github.com/BerriAI/litellm/releases/tag/v1.44.7)：更新内容 [文档] 由 @ishaan-jaff 在 #5367 中实现在 litellm 代理服务器中使用 litellm sdk [特性] 由 @ishaan-jaff 在 #5371 中增加对微调 vertexai 模型的支持 [重构] 重构 cohere prov...
- [GitHub - CEDARScript/cedarscript-grammar: A SQL-like language for efficient code analysis and transformations](https://github.com/CEDARScript/cedarscript-grammar#key-features))：一种用于高效代码分析和转换的类 SQL 语言 - CEDARScript/cedarscript-grammar
- [GitHub - CEDARScript/cedarscript-grammar: A SQL-like language for efficient code analysis and transformations](https://github.com/CEDARScript/cedarscript-grammar#key-featur)：一种用于高效代码分析和转换的类 SQL 语言 - CEDARScript/cedarscript-grammar

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1299608538548666424) (20 messages🔥):

> - `Mojo documentation contributions` (Mojo 文档贡献)
> - `Learning programming languages` (学习编程语言)
> - `Mojo vs Python for ML` (Mojo vs Python 用于 ML)
> - `C++ learning resources` (C++ 学习资源)
> - `Community engagement and humor` (社区参与和幽默)

- **Mojo API 文档需要示例**：一场讨论强调了 Mojo API 文档中 **Collections 缺乏示例** 的问题，并建议通过 [GitHub](https://github.com/modularml/mojo/tree/main) 为文档做出贡献。
  
  - 成员们强调了社区参与的重要性，并将 **准备 Pull Requests** 视为改进文档的一步。
- **在学习 Mojo 还是 C++ 之间做出选择**：一位正在考虑学习 Mojo 或 C++ 的用户得到的建议是，Mojo 作为一种现代系统级语言，可能更适合他们的探索，尤其是在 ML 和数据科学等领域。
  
  - 社区成员分享了关于语言选择的见解，建议重点关注 **Rust 或在 Mojo 中构建库**。
- **Rust 在安全关键型应用中的角色**：讨论强调了 Rust 在需要安全关键认证（safety-critical certifications）的领域正获得青睐，吸引了那些从 C++ 转型的人。
  
  - 用户们辩论了 **Rust 的实用性** 在各种编程语境下的表现，特别是在系统级编程中的安全性和正确性方面。
- **带有代码启示的社区幽默**：频道里分享了一个轻松时刻，一位用户发表了幽默的代码注释，以俏皮的语气提到了函数实现中的问题。
  
  - 像这样的评论在讨论编程挑战的同时，培养了 **社区参与感** 并带来了娱乐。
- **探索 Mojo 在医学研究中的潜力**：一位用户表达了学习 Mojo 的浓厚兴趣，认为它在数据科学和 ML 领域具有适用性，并与其医学研究抱负相结合。
  
  - 对话显示出对 Mojo 作为专业领域用户可行选择的 **积极展望**，尽管它仍处于早期开发阶段。

 

**提到的链接**：[GitHub - modularml/mojo: The Mojo Programming Language](https://github.com/modularml/mojo/tree/main)：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1299567383215869992) (151 messages🔥🔥):

> - `Mojo language features` (Mojo 语言特性)
> - `InlineArray slicing` (InlineArray 切片)
> - `Protobuf plugin experience` (Protobuf 插件经验)
> - `Zig language comparisons` (Zig 语言对比)
> - `Argument handling in functions` (函数中的参数处理)

- **使用 Span 对 InlineArray 进行切片**：一位用户讨论了在 Mojo 中对 `InlineArray` 进行切片的问题，表达了在推导参数方面的困难，特别是 `Span` 的集成无法正确推导字段参数。
  
  - 提出的解决方案包括利用 `as_span` 来简化过程并有效地处理切片。
- **内联汇编与 Protobuf**：讨论围绕 Mojo 中内联汇编（Inline Assembly）的实现展开，以及 Protobuf 的复杂性是否值得为其编写自定义插件。
  
  - 一位用户强调了 Protobuf 实现零拷贝反序列化（zero-copy deserialization）的挑战，并建议考虑将 ASN.1 作为更好的调试替代方案。
- **Zig 语言特性**：用户对 Zig 的类型系统表现出兴趣，特别是它如何处理类型、comptime 值以及用于接口语义的联合体（unions）。
  
  - 对话涉及了 Zig 激进的代码生成（eager code generation）如何导致大量代码产生，以及这种方法在应用库中的潜在缺点。
- **Comptime 参数处理**：用户讨论了处理可以是 comptime 或运行时（runtime）参数的挑战，特别是在需要针对特定情况进行优化的函数中。
  
  - 解决方案包括使用 `switch` 语句结合 comptime 检查，以有效地管理各种参数场景。
- **联合体与接口**：关于在编程中使用联合体（unions）作为接口（interfaces）的复杂性引起了关注，特别是关于为了传递值而需要实例化联合体的问题。
  
  - 讨论以希望在处理联合体时能有更简单的语法和功能以提高可用性而结束。

**提到的链接**：

- [Physical Address Extension - Wikipedia](https://en.wikipedia.org/wiki/Physical_Address_Extension)：未找到描述
- [mojo/stdlib/src/sys/_assembly.mojo at nightly · modularml/mojo](https://github.com/modularml/mojo/blob/nightly/stdlib/src/sys/_assembly.mojo)：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。
- [GitHub - mzaks/flatbuffers-mojo: Flatbuffers lib in Mojo](https://github.com/mzaks/flatbuffers-mojo)：Mojo 中的 Flatbuffers 库。通过在 GitHub 上创建账户为 mzaks/flatbuffers-mojo 的开发做出贡献。

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1300512140855349331) (2 条消息):

> - `Mutable Tensors`
> - `Nightly Builds`

- **Mutable Tensors 旨在增强训练对象**：当前的 Nightly Builds 正在引入 **mutable tensors**，从而能够表示诸如 **trained weights** 和 **KVCaches** 之类的训练对象。
  
  - 从 API 的角度来看，该功能仍在开发中，但进展顺利，预计将包含在 **下一个版本** 中。
- **社区对新功能充满期待**：一位社区成员对 **mutable tensors** 的引入表达了热情，表明该新功能受到了积极好评。
  
  - “Fantastic” 是被记录下的情绪，反映了对即将到来的进展的渴望。

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1299748585511976991) (13 条消息🔥):

> - `High Performance Mixed Precision Computing`
> - `CUDA Performance Issues on H100`
> - `Llama 3.2 Inference Discrepancies`
> - `Unsloth Kernels Guide`
> - `Parallel Function Calls in CUDA`

- **高性能混合精度计算演讲**：频道中提到，一场关于 **high performance mixed precision computing** 的演讲将在 5 分钟后开始。
  
  - 社区成员用 ⚡🤌👍 等反应表达了兴奋之情。
- **分析 CUDA 推理代码**：一位用户报告在使用 nsight systems 对 **H100** 进行分析时遇到了 “Command Buffer Full” 问题，而这在 **A100** 上并未出现。
  
  - 他们正在寻求潜在解决方案的指导，或者询问是否应该在另一个频道提问。
- **Llama 3.2 推理差异**：一位用户询问，在 MLX 和 Torch 上运行 **L3.2 inference** 时模型输出的微小差异是否可能意味着实现中存在错误。
  
  - 另一位成员建议，与 FP32 相比，在 **FP16** 或 **BF16** 下进行评估可能会导致这些差异。
- **寻求 Unsloth Kernels 指南**：一位用户询问如何获取 **unsloth kernels** 的指南，并分享了 [GitHub 项目](https://github.com/unslothai/unsloth) 的链接。
  
  - 该项目声称微调 Llama 3.2、Mistral、Phi 和 Gemma LLMs 的速度快 2-5 倍，且显存占用减少 80%。
- **使用 CUDA 并行化函数调用**：一位初学者提出了一个挑战，即如何使用 **CUDA** 或 **Numba/CuPy** 为任意方法创建 **parallel function**。
  
  - 他们表示需要关于在多个输入上自动并行化函数调用的建议。

 

**提到的链接**：[GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory](https://github.com/unslothai/unsloth)：微调 Llama 3.2、Mistral、Phi 和 Gemma LLMs 的速度快 2-5 倍，且显存占用减少 80% - unslothai/unsloth

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1299447527334084608) (15 messages🔥):

> - `FA3 Performance Insights` (FA3 性能洞察)
> - `Triton MXFP4 Support` (Triton MXFP4 支持)
> - `GPU Performance Discrepancies` (GPU 性能差异)
> - `Triton Debugging Strategies` (Triton 调试策略)
> - `Triton Hardware Support` (Triton 硬件支持)

- **FA3 展示了令人印象深刻的 kernel 技巧**：据报道，尽管存在一些 **register spill**（寄存器溢出），FA3 仍采用了大量技巧来实现 **SOTA performance**，这使其显得非常 hacky。
  
  - 一位成员幽默地提到，“必须说 FA3 有这么多技巧”，实际的模型工作负载也对其性能有所贡献。
- **Triton 将支持 MXFP4 x FP16 matmul**：Triton 预计很快将正式支持 **(emulated) MXFP4 x FP16 matmul**，旨在提升性能。
  
  - 一位成员建议，为了获得更高的性能上限，应使用比 Triton 更底层的语言。
- **GPU 性能差异显著**：一些成员指出，带有重复索引的 **tl.load** 在 **A100/H100** 等 GPU 上较慢，但在 **2080 Ti** 和 **3090** 等其他 GPU 上运行良好。
  
  - 一位用户询问如何调试此问题，并怀疑这与 **tl.load** 的性能具体相关。
- **Triton 调用的调试策略**：关于较慢的 **tl.load** 的正确调试方法的讨论表明，**TMA** 不支持交错索引（interleaved indices），因此在 A100/H100 上不可行。
  
  - 有人指出，如果处理不当，使用连续块（contiguous blocks）可能会进一步降低性能。
- **探索 Triton 硬件支持**：有人询问 Triton 除了 GPU 和 Intel NPU 之外还支持哪些硬件，并提到 **AWS AI chips** 是一个潜在但尚不确定的候选。
  
  - Triton 编译器在这些芯片上的性能也受到了质疑，强调了进行对比分析的必要性。

 

**提及的链接**：[Poor performance on Ampere vs. Ada with bitpacked weights · Issue #4906 · triton-lang/triton](https://github.com/triton-lang/triton/issues/4906)：我正在编写一个库，用于在 Triton/CUDA 中执行不同的低比特 matmul kernel。Triton kernel 在 Ada GPU（如 4090 RTX 和 A6000 Ada）上表现出色——在大矩阵上与 Marlin 持平……

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1299732232872661065) (10 messages🔥):

> - `Torch Compile Performance` (Torch Compile 性能)
> - `CUDA Graphs and Triton Layers` (CUDA Graphs 与 Triton 层)
> - `GEMM Optimization on Different GPUs` (不同 GPU 上的 GEMM 优化)
> - `Custom Operations Impact on Performance` (自定义操作对性能的影响)
> - `Max Autotune vs Reduce Overhead` (Max Autotune 与 Reduce Overhead)

- **Torch Compile 会降低带有 Triton 层的模型速度**：一位用户注意到，他们带有 **Triton layers** 的简单 MLP 模型在 `torch.compile` 下运行得慢得多，特别是在使用 `reduce-overhead` 模式时，因为 CUDA Graphs 需要 **extra copy**（额外拷贝）。
  
  - 他们发现使用 `max-autotune-no-cudagraphs` 显著提升了性能。
- **自定义操作阻碍性能**：另一位用户观察到，将 Triton kernels 封装在 `custom_op` 中会负面影响 **reduce-overhead** 模式下的性能，而像 **tinygemm** 这样的其他操作即使被封装也表现良好。
  
  - 他们指出，如果不使用 `custom_op`，在 2.4.1 版本中使用 `torch.compile` 会面临限制。
- **GEMM 优化因 GPU 架构而异**：一位用户分享说，**torch.compile** 根据 GPU 的 **SM count**（SM 数量）对 GEMM 进行不同的优化，在 SM **> 80** 的设备上观察到了性能提升，而 SM **< 80** 的设备则不然。
  
  - 他们引用了一个相关的讨论，关于这种架构决策如何影响性能，特别是对于拥有 72 个 SM 的 **NVIDIA A10G**。
- **手动 CUDA Graphs 提升性能**：一位成员透露，他们通过切换到 `max-autotune-no-cudagraphs` 并结合 **manual CUDA Graphs** 以实现更快的执行，解决了 Triton 层的性能问题。
  
  - 例如，他们在 RTX 4090 上使用 **Llama2-7B** 实现了 **186 tokens/sec** 的端到端解码速度，超过了 `reduce-overhead` 的 **173 tokens/sec**。
- **基于版本的 GEMM 限制**：关于 **PyTorch** 版本的讨论出现，用户指出之前的版本根据 SM 数量限制了 GEMM 优化。
  
  - 有人建议这种行为可能早在 **2.3** 版本或更早之前就已经改变了。

 

**提及的链接**：[torch_compile.webm](https://drive.google.com/file/d/1HzhrvSjgM7y0ks61AdAJAW0jKCHVdOyF/view?usp=sharing)：未找到描述

 

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1299656705465913398) (5 条消息):

> - `在 AMD MI300x 上微调 Llama3`
> - `对比损失（Contrastive Loss）技术的进展`
> - `GEMM 中的 Epilogue Visitor Tree`
> - `NVIDIA 的 FP8 训练框架`

- **在 AMD MI300x 上微调 Llama3 的历程**：一篇详细的 [文章](https://publish.obsidian.md/felafax/pages/Tune+Llama3+405B+on+AMD+MI300x+(our+journey)) 分享了使用 **AMD MI300x** 硬件微调 **Llama3 405B** 模型的见解。
  
  - 文章概述了微调过程中面临的实际步骤和挑战。
- **对比损失训练的创新**：一种新的对比损失方法提出了一种基于分块（tile-based）的计算策略，避免了相似度矩阵的完全显式生成（materialization），从而优化了 GPU 资源。
  
  - 该方法有效地扩展了 Batch Size，展示了在实现 *online softmax* 和 **cross-GPU tiling** 等概念时的优势。
- **探索 GEMM 中的 Epilogue Visitor Tree**：一篇补充文章深入探讨了 **Epilogue Visitor Tree (EVT)**，用于 NVIDIA GPU 上基于 GEMM 计算的高效后处理。
  
  - 它解释了如何将 EVT 集成到 CUTLASS GEMM kernel 中，从而增强性能并简化复杂的处理过程（如 elementwise 激活）。
- **NVIDIA 的 FP8 训练方案揭晓**：NVIDIA 最近的论文介绍了一个名为 **COAT** 的新 FP8 训练框架，旨在优化大模型训练期间的内存占用。
  
  - 关键特性包括 *Dynamic Range Expansion*（动态范围扩展）和 *Mixed-Granularity Activation Quantization*（混合粒度激活量化），显著提升了训练效率。

**提到的链接**：

- [Tune Llama3 405B on AMD MI300x (our journey) - Felafax Blog - Obsidian Publish](https://publish.obsidian.md/felafax/pages/Tune+Llama3+405B+on+AMD+MI300x+(our+journey))：在 AMD MI300x 上微调 Llama3 405B（我们的历程）- Felafax 博客。
- [Breaking the Memory Barrier: Near Infinite Batch Size Scaling for Contrastive Loss](https://arxiv.org/abs/2410.17243)：对比损失是表示学习的一种强大方法，较大的 Batch Size 通过提供更多负样本来更好地区分相似和不相似的样本，从而提高性能...
- [COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313)：FP8 训练已成为提高训练效率的一种极具前景的方法。现有框架通过将 FP8 计算应用于线性层来加速训练，但保留了优化器状态和激活值的原始精度...
- [Epilogue Fusion in CUTLASS with Epilogue Visitor Trees](https://research.colfax-intl.com/epilogue_visitor_tree/)：欢迎阅读我们关于 GEMM（通用矩阵乘法）教程系列的补充文章。主系列（1, 2）中的文章讨论了 NVIDIA GPU 上高性能 GEMM 的实现...

---

### **GPU MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1299843868338032752) (1 条消息):

> - `ML / 研究工程师求职`
> - `AI 工程技能`
> - `通过博客进行公开学习`

- **ML / 研究工程师寻求全职职位**：一位成员宣布正在寻找 ML 领域的全职职位，强调了在 **AI engineering**、**data**、**evals**、**finetuning** 和 **deployment** 方面的专业知识。
  
  - 他们表示对核心关注点以外的角色也持灵活态度，并请求反馈或职位推荐。
- **NLP 和 STT 领域的专业知识**：该成员详细介绍了他们在 **NLP**（包括 **LLMs**、**RAG** 和 **NER**）和 **STT**（包括 **ASR** 和 **Speech Translation**）方面的主要经验。
  
  - 他们旨在将这些专业知识应用于求职中，在这些专业领域贡献重要知识。
- **通过博客公开学习**：该成员在 Substack 上推广他们的个人博客，分享他们在 AI 工程学习历程中的见解。
  
  - 他们在 **10 个月前** 推出了该博客，以与更广泛的社区互动并记录他们的经历。

 

**提到的链接**：[Amgad’s Substack | Amgad Hasan | Substack](https://amgadhasan.substack.com)：我的个人 Substack。点击阅读 Amgad Hasan 的 Substack 出版物。10 个月前推出。

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1299617125148790845) (26 messages🔥):

> - `QAT Framework for VITs`
> - `GTX 780 with PyTorch`
> - `Learning Resources for Beginners`
> - `Getting Started with Triton`
> - `CUDA Learning Recommendations`

- **寻求适用于 VITs 的可配置 QAT 框架**：一位用户询问关于 VITs 的易于配置的 QAT 框架，寻找允许直接更改 quantizers 的仓库。
  
  - 未提供具体建议，但鼓励提供协助。
- **GTX 780 是 PyTorch 的经典挑战**：一位用户在使用 **GTX 780** 安装 PyTorch 时遇到了兼容性问题，理由是 CUDA 兼容性疑虑。
  
  - *一位经验丰富的用户评论道，虽然 toolkit 支持它，但由于其 **3GB VRAM** 的限制，torch kernels 可能没有针对这种旧架构进行优化。*
- **需要初学者友好的学习资源**：一位用户表示需要更简单的入门资源，而不是现有 YouTube 讲座中关于 profiling 等复杂话题的内容。
  
  - 建议包括一本名为 **PMPP** 的书，并对频道中提供的初学者教程表现出兴趣。
- **面向初学者的 Triton 教程**：另一位用户收到指导，去探索 [官方 Triton 教程](https://triton-lang.org/main/getting-started/tutorials/index.html)，以帮助他们学习构建 CNN。
  
  - 他们被引导至 **Triton-Puzzles** 等互动资源进行实践学习。
- **CUDA 学习硬件选择**：一位用户询问在 **$3.5k** 的预算下，是投资移动端 NVIDIA GPU 笔记本电脑还是台式机来学习 CUDA。
  
  - 回复建议利用 Colab 和 Kaggle 等提供免费访问强大 GPU 的平台，使得购买硬件的需求变得不那么紧迫。

**提到的链接**：

- [Course Detail | NVIDIA](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+T-AC-01+V1)：未找到描述
- [Start Locally](https://pytorch.org/get-started/locally/)：本地开始
- [Tutorials — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/index.html)：未找到描述
- [GitHub - srush/Triton-Puzzles: Puzzles for learning Triton](https://github.com/srush/Triton-Puzzles)：学习 Triton 的谜题。通过在 GitHub 上创建账号为 srush/Triton-Puzzles 的开发做出贡献。

---

### **GPU MODE ▷ #**[**pmpp-book**](https://discord.com/channels/1189498204333543425/1194427148656721970/1300235397296554004) (4 messages):

> - `Race Conditions in GPU Programming`
> - `Independent Thread Scheduling Changes`
> - `Memory Optimization Concerns`

- **竞态条件（Race Conditions）仍可能发生**：在 GPU 编程中误用 `__sync` threads 会导致 **race conditions**，引起了成员们对潜在问题的担忧。
  
  - 一位成员强调，如果没有适当的同步，编译器可能会错误地优化内存操作，从而导致 **错误结果**。
- **Volta/Turing 线程协调**：讨论显示，随着 **Volta/Turing** 架构（sm_70/75）的引入，warp 中的线程不再保证以同步（lockstep）方式运行。
  
  - 这一变化允许使用更廉价的 warp 同步指令来代替传统的 barriers，为程序员提供了更大的灵活性。
- **内存优化避免冲突**：一位成员指出，省略 `__sync` 指令可能导致编译器对缓存做出错误假设，从而可能影响内存的 read/store 操作。
  
  - 如果同步处理不当，此类 **optimizations** 可能会无意中导致不正确的结果。

 

---

### **GPU MODE ▷ #**[**youtube-recordings**](https://discord.com/channels/1189498204333543425/1198769713635917846/) (1 messages):

mr.osophy: [L33: BitBLAS](https://youtu.be/iA49QqWwMcA?feature=shared)

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1299458487620730961) (8 messages🔥):

> - `Quantization Techniques`
> - `Model Training Challenges`
> - `Dequantization Strategies`
> - `LLM.int8 Refactor`
> - `LoRA Usage`

- **选项 3：干脆不合并**：一位成员在成功训练 **HQQ+ 模型** 且使用 `dequantize(W) + sAB` 未出现问题后，建议增加“选项 3：不合并”。
  
  - 他们提议创建一个 **融合算子 (fused kernel)** 以避免独立的反量化和 LoRA 运行，这可能会使 GemLite 受益。
- **实验中的乱码输出**：实验中出现的 **乱码输出 (gibberish outputs)** 引起了关注，特别是反量化步骤对离群值 (outliers) 的影响。
  
  - 建议包括调整参数，以及在 **vLLM** 中单独使用 LoRA 而不是合并，以避免这些问题。
- **Sayak 关于 FLUX 的发现**：一位成员回想起 **Sayak** 的 Twitter 帖子，暗示 **FLUX** 的结果有所改善，尽管具体细节尚不确定。
  
  - 目前正在讨论通过协作解决此问题，重点关注 **LLM.int8 refactor**。
- **改进量化流程**：讨论了改进 **量化流程** 的各种选项，包括在不损害性能的情况下处理权重的方法。
  
  - 在这些选项中，提到使用 **CUDA streams** 是一个无需融合算子的实用解决方案。
- **LoRA 使用经验**：成员们讨论了在 vLLM 等模型中单独使用 **LoRA**，但意识到这可能类似于合并到原始权重中。
  
  - 在此过程中保持权重分层量化的完整性对于防止模型性能下降至关重要。

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1300286661179216002) (2 messages):

> - `Gear toxicity`
> - `Interesting YouTube videos`

- **Your Gear is Poisoning You! 视频**：一位成员分享了一个名为 [Your Gear is Poisoning You! (Not Clickbait)](https://www.youtube.com/watch?v=-ht7nOaIkpI) 的 YouTube 视频链接，该视频讨论了日常装备的潜在危险。
  
  - 他们称该视频 **非常有趣**，并强调了他们在制作过程中投入的时间和金钱。
- **关于视频内容的讨论**：该视频引发了关于装备对健康影响的对话，成员们对视频中的主张表示好奇。
  
  - 一位观众评论道：*“这真的让你开始思考你每天使用的东西！”*

 

**提到的链接**：[Your Gear is Poisoning You! (Not Clickbait)](https://www.youtube.com/watch?v=-ht7nOaIkpI)：感谢观看此视频。我花了很多时间和金钱让这个视频成为可能。最初帮助我资助这个项目的赞助商...

 

---

### **GPU MODE ▷ #**[**irl-meetup**](https://discord.com/channels/1189498204333543425/1218444432588800010/1300171898025676900) (2 messages):

> - `Toronto Meetup Series`
> - `NVIDIA GPUs`
> - `Collaboration in AI`

- **令人兴奋的多伦多计算系列见面会**：一位成员宣布计划在多伦多组织一系列专注于 **计算 (compute)** 的见面会，专门讨论 **NVIDIA GPUs** 和 **CUDA programming**。
  
  - 第一场见面会暂定于 **11 月 15 日** 举行，并希望在未来的会议中探索其他 HPC 公司，如 **Tenstorrent** 和 **Cerebras**。
- **多伦多见面会征集演讲者**：组织者正在为多伦多系列见面会寻找演讲者，并鼓励任何感兴趣的人联系。
  
  - 他们的目标是促进 **协作**，并统一该市已有的各种 **AI meetups** 的人才。
- **社区对见面会倡议的支持**：另一位成员对即将举行的系列见面会表示支持，并表示...

 

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1299492568597270670) (9 条消息🔥):

> - `CUDA 安装问题`
> - `Cache Modifiers`
> - `CUDA 版本兼容性`
> - `故障排除技巧`

- **CUDA 编译难题**：一位用户在重新安装 Ubuntu 和 CUDA 后，遇到了关于 `__ldcs` 和 `__stcs` 函数重载的错误，并提到了 `floatX` 类型的问题。
  
  - 他们使用 `train_gpt2.cu` 进行了测试，但不清楚错误的具体来源。
- **问题诊断提示**：另一位用户建议，错误可能源于传递给函数的类型异常，并建议研究 Cache Modifiers。
  
  - 他们对代码在未做改动的情况下无法运行感到惊讶。
- **CUDA 版本建议**：一名成员指出问题可能与 CUDA 版本有关，建议升级到更高版本，如 12.4。
  
  - 原用户确认其 CUDA 版本为 12.2，并决定尝试更新版本。
- **解决过程与安装历程**：升级 CUDA 后，该用户报告成功解决问题，并幽默地分享了其混乱的安装过程，包括多次搞挂又修复 CUDA。
  
  - 他们详细描述了涉及分区和应用程序的意外，最终虽然恢复了工作状态，但仍需导入书签并进行其他细微调整。

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1299722811257913355) (9 条消息🔥):

> - `AMD 消费级 GPU 与 ROCm 支持`
> - `驱动稳定性与内核差异`
> - `与 PyTorch 的性能对比`
> - `CK Flash Attention 后端 PR`

- **ROCm 支持的 AMD 消费级 GPU**：目前仅支持 **gfx11**，且**驱动不稳定**和 **segfaults** 等问题似乎依然存在。
  
  - 一位成员提到，在迁移到 Ubuntu 24.04 并移除一块 GPU 后，他们的系统已经稳定运行了几个月。
- **内核差异对性能产生巨大影响**：一位用户强调，使用 **torch SDPA** 时，前向传播在 **4090** 上约为 **50 iter/s**，但在 **XTX** 上仅为 **15 iter/s**，表明存在巨大的性能差距。
  
  - 根据用户经验，反向传播的性能可能会更糟。
- **MES overruns 的修复**：最近解决了一个由于 **MES overruns** 导致服务器冻结的问题，尽管这在发布后花了大约 **2-3 年**时间。
  
  - 有人分享了一个变通方法，建议在 **Arch Linux** 上使用上游（upstream）版本的内核，而不是 Ubuntu 的旧内核。
- **CK Flash Attention 后端 PR 讨论**：一名成员指向了最近的一个 [GitHub pull request](https://github.com/pytorch/pytorch/pull/138947)，该 PR 专注于更新 ROCm 的 **CK gemm 后端**。
  
  - 该 PR 替代了之前的一个 PR，表明 PyTorch 中 ROCm 集成的持续改进，并标记了多位贡献者。

 

**提到的链接**：[[ROCm] CK Flash Attention Backend by alugorey · Pull Request #138947 · pytorch/pytorch](https://github.com/pytorch/pytorch/pull/138947)：替代了 ROCm#1592，更新了 CK gemm 后端的实现。可以关闭之前的 PR，抄送 @jeffdaily @sunway513 @jithunnair-amd @pruthvistony @ROCmSupport @dllehr-amd @jataylo @hongxiayang @naromero7...

 

---

### **GPU MODE ▷ #**[**webgpu**](https://discord.com/channels/1189498204333543425/1262121239044948009/) (1 条消息):

marksaroufim: [https://docs.pygfx.org/stable/index.html](https://docs.pygfx.org/stable/index.html)

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1299449190018977886) (6 messages):

> - `Nanogpt model training`
> - `Torch Compile Usage`
> - `Batch Normalization Implementation`

- **使用 Triton 优化 Nanogpt 训练**：成员们讨论了如果仅使用 eager PyTorch，通过**优化的 Triton 算子可以加速 Nanogpt 模型训练**。
  
  - 然而，目前的支持仅限于 **HF 兼容模型**（如 Llama 和 Qwen），对于自定义的 Nanogpt 模型需要进行修改。
- **Torch Compile 的增强**：有人指出使用 **torch.compile** 可以很好地融合 RMS，但在处理分块交叉熵损失（chunking cross-entropy loss）时效果不佳，这需要不同的实现方式。
  
  - 建议成员们编译模型本身，并对损失函数使用自定义实现。
- **关于加速函数的讨论**：一位成员询问了在使用 torch compile 时，哪些特定函数（如 **RMS Norm** 或 **交叉熵损失**）可以提高训练速度。
  
  - 回复表示对 liger-kernel 中的这些函数很熟悉，并建议进行性能提升测试。
- **Liger-Kernel 新增 Batch Norm**：重点介绍了一个最近的 **Pull Request** (#321)，该 PR 为 Liger Kernel 引入了 Batch Normalization。
  
  - 该更新在 4090 环境下与 **Keras 的 batch norm** 进行了对比测试，确保了准确性和性能符合标准。

**提及的链接**：[added batch norm by vulkomilev · Pull Request #321 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/pull/321)：摘要：添加了 batchNorm。测试完成：已将其与 Keras 的 batch norm 进行对比。使用了 4090 硬件。类型：[ X] 运行 make test 以确保正确性 [X ] 运行 make checkstyle 以确保...

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/) (1 messages):

geri8904: 有会议邀请吗？

---

### **GPU MODE ▷ #**[**diffusion**](https://discord.com/channels/1189498204333543425/1288899271193526342/1299835990373961828) (2 messages):

> - `Stable Diffusion`
> - `Cross Attention Maps`
> - `Attention Map Tools`

- **在 Stable Diffusion 中探索 Cross Attention Maps**：一位成员正在进行一项使用 **Stable Diffusion 模型** 的研究项目，并寻求关于在 **Unet** 中捕获 **Cross Attention maps** 的指导，以理解词与图像的交互。
  
  - 他们专门请求提供代码实现，以协助完成这项具有挑战性的任务。
- **找到 Attention Maps 资源**：一段时间后，该成员分享了他们在 [GitHub](https://github.com/wooyeolBaek/attention-map) 上找到的一个关于 **Cross Attention maps** 的有用资源。
  
  - 该仓库提供了在 **huggingface/diffusers** 中利用 Attention Maps 的工具，这应该能为类似的咨询提供帮助。

**提及的链接**：[GitHub - wooyeolBaek/attention-map: 🚀 Cross attention map tools for huggingface/diffusers](https://github.com/wooyeolBaek/attention-map)：🚀 适用于 huggingface/diffusers 的 Cross attention map 工具 - wooyeolBaek/attention-map

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1299466413412581377) (8 messages🔥):

> - `Discord Cluster Manager`
> - `CPU Execution with AVX and NEON`
> - `Development Sprint Timeline`

- **Discord 集群管理器文档已起草**：一位成员分享了一份[文档](https://docs.google.com/document/d/1SZ3bA9r9yVh27yy9Rbp9-VuAC2nWrUBp7dqetwgZG3M/edit?usp=sharing)，详细说明了 **Discord 集群管理器** 的运作方式，并计划在 **11 月 3 日**至 **11 月 10 日**期间积极开展工作。
  
  - 其他人表示愿意在时间允许的情况下为开发工作做出贡献。
- **探索使用 AVX 和 NEON 的 CPU 执行**：一位成员表示计划镜像在 CPU 执行方面的进展，使用 **AVX** 和 **NEON 指令**，并承认这不会立即合并到主项目中。
  
  - 他们的目标是如果团队稍后决定向该方向扩展，可以提供这一替代方案。
- **开发团队可用性变动**：一位目前正忙于期中考试的成员计划在 **11 月 10 日**之后加入开发冲刺。
  
  - 其他人提到他们很快会有更多空闲时间，并将于 **11 月 1 日**左右开始考虑贡献。

**提及的链接**：[Discord Cluster Manager](https://docs.google.com/document/d/1SZ3bA9r9yVh27yy9Rbp9-VuAC2nWrUBp7dqetwgZG3M/edit?usp=sharing)：我们的代码将放在这里 [https://github.com/gpu-mode/discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager)。用户体验：11 月 4 日开始，最迟 11 月 10 日完成功能。这项工作我们只需要一个单节点。Claud a...

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1299710091381178441) (53 messages🔥):

> - `Cohere 中的 Connector 使用`
> - `Playground 性能问题`
> - `算法交易见解`

- **在 Cohere 中使用 Connector 查询时遇到困难**：一位用户报告了通过 Cohere Connector 检索数据时遇到困难，指出在查询特定用户 ID 时，持续出现类似 *'I was unable to find any information'* 的报错。
  
  - 另一位成员建议发送邮件至 [support@cohere.com](mailto:support@cohere.com) 联系支持团队，以获取有关账户问题的进一步协助。
- **Playground 的延迟性能问题**：讨论中提到了 Playground 的延迟问题，特别是在发送多条消息后，部分成员认为这影响了交互体验。
  
  - 有人建议开启新对话或清除缓存以提高性能，并指出延迟可能与设备限制和 context window 过载有关。
- **关于算法交易的见解**：成员们分享了在算法交易方面的经验，其中一人讨论了他们在周末进行的关于 AI 情绪与交易相关性的分析工作，强调了媒体偏见的细微差别。
  
  - 另一位成员指出，人类观点对市场波动的影响微乎其微，并建议从 EDGAR 等综合平台获取更重要的交易见解。

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1299709025017135156) (33 条消息🔥):

> - `Cohere community server`
> - `Using connectors in Cohere`
> - `Reranker models for multimodal data`
> - `News generation applications`
> - `Cohere API models and limits`

- **Cohere 社区服务器访问**: 一位用户询问如何访问 **Cohere For AI** 社区服务器，另一位成员分享了加入该服务器的 [申请页面](https://cohere.com/research) 链接。
  
  - 随后提供了关于 Cohere 研究实验室的信息，该实验室旨在解决复杂的机器学习问题。
- **在 Cohere 中配置 Connectors**: 用户寻求关于如何使用 connector 查询 Cohere 聊天端点的帮助，并收到了用于设置的 [文档](https://docs.cohere.com/v1/docs/creating-and-deploying-a-connector) 链接。
  
  - 需要注意的是，connectors 必须使用 v1 API 进行设置，因为 v2 API 尚不支持。
- **对多模态 Reranker 模型的关注**: 一位成员询问是否有支持多模态数据（特别是文本和图像）的 reranker 模型，得到的回复是 **目前尚不存在**。
  
  - 对话显示了用户对模型能力的积极兴趣以及对此话题的反馈。
- **选择用于新闻生成的模型**: 在被问及选择哪个模型来生成较长文章时，回复指出 **Command R** 更快且更便宜，而 **Command R+** 提供更好的性能。
  
  - 建议用户在升级前先尝试免费 API，以避免触及速率限制（rate limits）。
- **了解 API Token 限制**: 讨论透露 **Command R** 模型可以生成高达 128k tokens 的响应（包括输入和输出），一些用户报告响应平均为 3,000 到 4,000 个字符。
  
  - 用户被告知 `max_tokens` 参数受上下文窗口限制调节，有效的提示词（prompts）可以延长响应长度。

**提到的链接**:

- [Retrieval Augmented Generation (RAG) — Cohere](https://docs.cohere.com/v2/docs/retrieval-augmented-generation-rag): 使用检索增强生成和 Cohere 的 Chat API 生成带有外部数据和行内引用的文本。
- [Tokens and Tokenizers — Cohere](https://docs.cohere.com/v2/docs/tokens-and-tokenizers): 本文档描述了如何使用 tokenize 和 detokenize API 端点。
- [Creating and Deploying a Connector (v1 API) — Cohere](https://docs.cohere.com/v1/docs/creating-and-deploying-a-connector): 了解如何实现 connector，从设置到部署，以通过 Cohere 的 Chat API 实现有根据的生成（grounded generations）。
- [Cohere For AI (C4AI)](https://cohere.com/research): Cohere For AI (C4AI) 是 Cohere 的非营利研究实验室，致力于解决复杂的机器学习问题。
- [Command R+ — Cohere](https://docs.cohere.com/docs/command-r-plus): Command R+ 是 Cohere 用于对话交互和长上下文任务的模型，最适合复杂的 RAG 工作流和多步工具使用。
- [The Command R Model — Cohere](https://docs.cohere.com/docs/command-r): Command R 是一款对话模型，擅长语言任务并支持多种语言。
- [Login | Cohere](https://dashboard.cohere.com/playground/chat): 登录以通过一个易于使用的 API 访问先进的大语言模型（LLM）和 NLP 工具。
- [Login | Cohere](https://coral.cohere.com/): 登录以通过一个易于使用的 API 访问先进的大语言模型（LLM）和 NLP 工具。
- [Pricing](https://cohere.com/pricing): 直接通过我们的 API 访问我们的模型，以创建可扩展的生产工作负载。

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1299488845175459851) (17 条消息🔥):

> - `Weekend Timeout Issues` (周末超时问题)
> - `API Timeout Reporting` (API 超时报告)
> - `Intermittent Timeout Errors` (间歇性超时错误)

- **周末超时问题再次出现**：成员们注意到**周末超时问题**再次浮现，日志显示在 **10 月 26 日和 27 日** 持续出现 **read operation timed out** 错误。
  
  - *这似乎是一个间歇性问题*，据报告主要发生在周末。
- **超时频率查询**：成员们询问了报告超时的频率和开始时间，确定这些问题似乎在周末较早时段开始。
  
  - 一位成员对**用户标记此问题**表示感谢，确保了问题的妥善升级。
- **超时解决方案的参与**：一位成员提议调查账户详情，以针对**超时发生情况**提供进一步见解。
  
  - 另一位成员对解决这些担忧表示赞赏，强调了社区的协作努力。
- **超时事件的详细日志**：用户分享的近期日志显示，在 **10 月 28 日** 的特定 UTC 时间戳发生了连续的**超时错误**。
  
  - 尽管用户的本地错误已解决，但根据日志显示，超时仍在继续。
- **报告错误中的社区支持**：成员们在向内部团队报告这些超时问题以寻求解决时，表现出了相互支持。
  
  - 对用户*及时报告*的行为表示了认可，强化了团队合作在故障排除中的重要性。

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1299476496867397664) (80 条消息🔥🔥):

> - `AI research grants` (AI 研究资助)
> - `Customization and personalization in AI` (AI 中的定制化与个性化)
> - `Limitations of LLMs` (LLM 的局限性)
> - `Using multiple LLMs in AI applications` (在 AI 应用中使用多个 LLM)
> - `Agent interactions in AI` (AI 中的 Agent 交互)

- **探索 AI 研究资助经验**：一位成员询问了申请 AI 研究资助的相关经验，寻求可能已获得资助的其他人的见解。
  
  - 这反映了 AI 研究社区对资金机会日益增长的兴趣。
- **AI 定制化中的挑战**：有成员对 ChatGPT 有时会忽略或违背定制化指令表示担忧，这导致了非预期的输出。
  
  - 参与者讨论了他们的引导未按预期执行的经历，引发了关于 AI 推理和意图的疑问。
- **理解 LLMs 及其局限性**：大家公认 LLM 擅长生成语言，但在数学方面表现不佳，因此建议使用 Python 工具进行计算。
  
  - 参与者强调在与 LLM 交互时需要分步引导，以改进其功能。
- **在 AI 解决方案中利用多个 LLM**：一位用户分享了构建 AI 应用的经验，指出单个 LLM 不足以胜任所有任务。
  
  - 对话强调了 prompt chaining 和 agentic 工作流在增强效果方面的实用性。
- **AI 行为的人格化**：讨论涉及了由于 AI 模仿人类对话和决策而产生的人格化倾向。
  
  - 成员们告诫不要将 LLM 视为拥有像人类一样的意图或推理能力，强调了它们作为机器的本质属性。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1299772068262449192) (10 条消息🔥):

> - `演进中的 Prompt Engineering`
> - `自定义 AI 语言`
> - `语言定义中的 AI 互操作性`
> - `管理 AI 模型`
> - `使用 API 进行特定的 AI 调用`

- **Prompt Engineering 作为一种编程语言**：一位成员思考 Prompt Engineering 是否可以演变成一种 AI 的“编程语言”，从而在各种模型之间实现**模块化且可重用**的交互。
  
  - 另一位成员补充道，当沟通清晰时，*“模型能做的任何事情，我们都能让它去做”*。
- **创建用于 AI 交互的新语言**：成员们讨论了为 AI 交互“创建”新语言的潜力，类似于当今使用的各种编程语言。
  
  - 一位成员提到，就像编程一样，我们可以采用各种策略来有效地向 AI 模型传达我们的意图。
- **互操作性与定义新语言**：一位用户声称成功教会了 AI 定义一种从未被翻译过的语言，这表明**互操作性**可以将语言定义扩展到传统边界之外。
  
  - 他们暗示了 AI 在定义其所需语言方面可能取得成功的**未定义维度**。
- **模型自定义的挑战**：一位成员对他们的模型忽略自定义 Prompt 并独立命名表示沮丧。
  
  - 另一位用户询问如何将 API 调用限制在特定模型以提高效率，反映了类似的困扰。
- **使用 GPT Vision 进行图像分割**：一位用户寻求使用 GPT Vision 将图像切割成面板的帮助，但在坐标不准确和 GPU 使用限制方面遇到了困难。
  
  - 他们特别希望避免将角色或面板切成两半，正在寻找有效的解决方案。

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1300315811126448152) (2 条消息):

> - `AI 一致性`
> - `挑战 AI 学习`

- **AI 一致性是一个神话**：一位成员强调 **AI 完全不具有一致性**，强调了其行为中固有的不可预测性。
  
  - 在 AI 领域*你必须学习的第一件事*就是承认这种不一致性。
- **AI 挑战中的乐趣**：另一位成员表示，与 AI 合作确实是一个**有趣的挑战**，指出了 AI 相关任务的迷人本质。
  
  - 这一观点展示了在处理 AI 技术时，享受与复杂性之间的平衡。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1300315811126448152) (2 条消息):

> - `AI 一致性`
> - `AI 理解中的挑战`

- **揭示 AI 的不一致本质**：一位成员指出，关于 AI **要学习的第一件事**就是它固有的**缺乏一致性**。
  
  - 这对任何试图深入理解或使用 AI 模型的人来说都是一个有趣的挑战。
- **AI 中的有趣挑战！**：另一位成员评论道，参与 AI 确实呈现了一个**有趣的挑战**。
  
  - 这突显了探索 AI 能力和行为时充满趣味且富有洞察力的方面。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1299694556425617421) (82 条消息🔥🔥):

> - `OpenAI 与 Google 之间的 AI 竞赛`
> - `Meta 的新搜索引擎开发`
> - `生成式 AI 的用户采用率`
> - `Gemini 模型发布的问题`
> - `大型 AI 公司面临的挑战`

- **OpenAI 与 Google 在 12 月的对决**：OpenAI 计划在 12 月发布其下一个 AI 模型，而 Google 也正致力于发布其 **Gemini 2.0**，加剧了 AI 领域的竞争。
  
  - 虽然 OpenAI 的推出是分阶段的，但 Google 寻求广泛发布，尽管其**性能预期**可能无法完全达到。
- **Meta 构建自己的搜索引擎**：Meta 在工程经理 **Xueyuan Su** 的领导下，正在开发一种新的网页搜索引擎，以减少对 Google 和 Bing 数据源的依赖，目前已有一些新闻网站屏蔽了其爬虫。
  
  - 该项目旨在为 Meta 的平台提供更独立的 AI 解决方案，避免再次出现类似 **Apple** 的受制局面。
- **生成式 AI 的采用速度缓慢**：最近的一篇论文声称，虽然有 **40% 的美国成年人** 使用生成式 AI，但实际上只有 **0.5% – 3.5%** 的工作时间涉及其辅助。
  
  - 采用率远低于预期，揭示了实际使用情况与预期的生产力影响之间存在差距。
- **对 Gemini 发布情况的担忧**：**Gemini 模型** 的发布因性能较前版本下降以及针对消费者的营销问题而面临批评。
  
  - 此次发布被认为是**最糟糕的发布之一**，严重的性能退化影响了用户体验。
- **大型 AI 公司的职能失调**：成员们讨论了像 Microsoft 和 Google 这样的大型公司，尽管拥有巨大资源，但被认为没有发挥出应有的潜力。
  
  - 人们对效率低下、产品发布延迟以及在快速发展的市场中保持竞争力的压力表示担忧。

**提到的链接**：

- [Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1850919619629854875)：据报道，Meta 正在构建一个由工程经理 Xueyuan Su 领导的网页搜索引擎，该项目已进行至少 8 个月，旨在摆脱对 Google Search 和 Bing 数据源的依赖，目前已有纽约时报等网站...
- [Google 计划近期宣布其下一个 Gemini 模型](https://www.theverge.com/2024/10/25/24279600/google-next-gemini-ai-model-openai-december)：12 月将成为 OpenAI 和 Google 展开 AI 发布对决的一个月。
- [Arvind Narayanan (@random_walker) 的推文](https://x.com/random_walker/status/1850871740550758766)：这是一个 AI 炒作案例研究。论文《生成式 AI 的快速采用》因声称 40% 的美国成年人正在使用生成式 AI 而广为流传。但这包括了...
- [The Information (@theinformation) 的推文](https://fxtwitter.com/theinformation/status/1850254366239776980)：独家：Google 正在准备 “Project Jarvis”——一个可以接管计算机以协助日常网页任务的 AI 程序。https://www.theinformation.com/articles/google-preps-ai-that-takes-over-comput...
- [OpenAI CFO 表示 AI 不再是实验性的](https://youtu.be/eCqFgVqWbEs)：OpenAI CFO Sarah Friar 表示人工智能不再是实验性的。她说银行、金融机构和金融科技公司每天都在使用它...

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1300559769211371600) (1 条消息):

> - `人类生成样本的定价`
> - `标注质量对比`

- **关于人类生成样本定价的咨询**：一位成员询问在哪里可以找到关于**人类生成样本的价格**与将其标注为好坏的价格对比信息。
  
  - 这个问题突显了在**手动与自动**标注过程的价值主张中需要更多的透明度。
- **索取标注指南**：随后提出了关于**标注指南**的问题，该指南应能明确定义好样本或坏样本的质量。
  
  - 这表明人们对建立评估生成样本的**标准准则**越来越感兴趣。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/) (1 条消息):

rjvs: 每个人都在参与 🍓 [https://underworld.lnk.to/strawberryhotel](https://underworld.lnk.to/strawberryhotel)

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1299856564823396396) (7 条消息):

> - `Lex Friedman`
> - `Prompt Optimization`
> - `Hiking Stories`

- **Lex Friedman 解除屏蔽并关注**：成员对 **Lex Friedman** 在 Twitter 上解除屏蔽并关注他们表示兴奋，称感觉自己处于早期阶段。
  
  - 他们幽默地提到，*'我准备好爆米花了'*，以庆祝这次意外的互动。
- **询问 Prompt Optimization 资源**：一位成员询问了关于 **Prompt Optimization** 的最佳资源，表现出对优化 Prompt 以获得更好 AI 交互的兴趣。
  
  - 另一位成员建议将 **DSPy** 作为一个合适的选择，反映了社区的知识共享。
- **妈妈徒步经历的幽默**：一位用户分享了来自 **Andrew Schmidt** 的幽默推文，其中他们的妈妈在一次艰难的徒步后说：*'那次徒步差点要了我的命！'*
  
  - 这引发了关于个人徒步经历和挑战的轻松讨论。

 

**提到的链接**：[来自 Schmidt (@AndrewSchmidtFC) 的推文](https://x.com/AndrewSchmidtFC/status/1849947375873425889)：我妈：那次徒步差点要了我的命！Apple 的 AI 摘要：

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1299460958497935413) (24 条消息🔥):

> - `Fast Math Mode in Metal`
> - `Tinygrad PR Submission Guidelines`
> - `Backend Testing for LLVM and ONNX`
> - `Tinybox Updates`
> - `Active Bounties`

- **Metal 中 Fast Math Mode 的复杂性**：关于 Metal 中 **fast math mode** 的讨论展开了，成员们注意到它默认执行代数转换，并且需要手动禁用以遵循严格的浮点规则。
  
  - 还提到了使用 `-fassociative-math` 标志进行 *Reassociation*，作为数学表达式的一种潜在优化方式。
- **严格的 PR 提交指南**：George Hotz 强调了在提交新 **PR** 之前审查已合并 **PR** 的重要性，并表示没有明确理解或测试的更改将不会被合并。
  
  - 他批评了以不同方式编码相同信息的冗余 **PR** 提交，敦促贡献者专注于 **Bug 修复而非新功能**。
- **LLVM 和 ONNX 后端的测试输出**：一位成员提议为 **LLVM** 和 **ONNX** 后端的输出创建测试，指出 tinygrad 缺乏此类检查。
  
  - George 确认目前已有将 tinygrad 输出与 **Torch** 进行比较的测试，表明现有的测试基础设施可以进一步扩展。
- **Tinybox 会议安排**：会议定于香港时间晚上 8 点举行，讨论包括 **Tinybox 更新**和各种技术评论在内的多个议题。
  
  - 议程项目包括 **uop canonical order** 的更新、**MLPerf** 以及正在进行的 **active bounties**。

**提到的链接**：

- [Clang 编译器用户手册 — Clang 20.0.0git 文档](https://clang.llvm.org/docs/UsersManual.html#cmdoption-ffast-math)：未找到描述
- [tensors.py 中的 MSE 以及由 littlemountainman 实现的测试 · Pull Request #7107 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7107)：实现了带有测试的 MSE

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1299481673611743232) (27 messages🔥):

> - `Tinygrad 复数支持`
> - `Tinygrad 在 Android 上使用 OpenCL`
> - `Tinygrad Tensor 连续性`
> - `模型转换工具`
> - `Tinygrad 生态系统发展`

- **Tinygrad 在复数支持方面面临挑战**：用户讨论了在 Tinygrad 中使用复数执行 DFT 等任务时的困难，并指出会出现表示缺乏直接支持的 `AssertionError`。
  
  - *George Hotz* 表示对更简便的复数支持感兴趣，并建议可以通过 2D 轴来模拟复数。
- **Tinygrad 在 Android 上使用 OpenCL**：一位用户询问如何使用 Tinygrad 为带有 OpenCL 加速器的 Android 设备编译模型，并寻求设置过程的指导。
  
  - 成员们指向了各种资源，包括 `compile_efficientnet.py`，它可以帮助生成所需的 OpenCL kernel 和 buffer，以便在脱离 Python 的情况下运行。
- **Tinygrad Tensor 连续性问题**：关于为什么在 Tinygrad 中使用 `Tensor.ones` 创建的 Tensor 可能表现为不连续（non-contiguous）的讨论，涉及到了广播（broadcasting）行为。
  
  - *Davidgonmar_* 澄清说，与填充完整 buffer 的 PyTorch 相比，Tinygrad 采用了不同的内存管理技术。
- **对模型转换工具的兴趣**：一位用户表示希望在 Tinygrad 中拥有类似 HuggingFace `transformers` 的工具来帮助标准化模型，并指出了 `extra` 和 `examples` 文件夹带来的组织挑战。
  
  - *George Hotz* 对这一提议表示欢迎，并强调随着 Tinygrad 的成熟，这类库的重要性将日益凸显。
- **Tinygrad 生态系统的未来发展**：关于 Tinygrad 生态演进的讨论，*George Hotz* 表示将转向速度提升和更广泛的实现。
  
  - 他承认目前转换器（converters）的重要性，但指出它们通常效率不高，暗示未来可能会关注更新的工具。

**提到的链接**：

- [tinygrad/examples/openpilot/compile2.py at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/examples/openpilot/compile2.py#L180-L188>)：你喜欢 PyTorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad
- [my_notebooks/mnist-from-scratch.ipynb at main · NinoRisteski/my_notebooks](https://github.com/NinoRisteski/my_notebooks/blob/main/mnist-from-scratch.ipynb)：实现 AI 论文。欢迎在 GitHub 上为 NinoRisteski/my_notebooks 的开发做出贡献。
- [tinygrad/examples/compile_efficientnet.py at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/examples/compile_efficientnet.py)：你喜欢 PyTorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad
- [tinygrad/extra/export_model.py at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/extra/export_model.py#L322-L327>)：你喜欢 PyTorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1299525755822280788) (5 条消息):

> - `Intelligent Knowledge Assistants` (智能知识助手)
> - `Advanced RAG Techniques` (高级 RAG 技术)
> - `Text-to-SQL Tutorials` (Text-to-SQL 教程)
> - `Agentic Workflows` (Agentic 工作流)

- **在 Ray Summit 探索智能知识助手**：Ray Summit 研讨会展示了构建**智能知识助手**的愿景，这些助手能以多种方式处理复杂数据，视频现已上传至 [YouTube](https://t.co/9eOQf5pYYi)。
  
  - 会议期间讨论了*超越简单任务所需的所有组件*，内容可以在[这里](https://t.co/ETH05Y8JNG)找到。
- **关于高级知识助手的全新视频系列**：一个全新的视频系列正在发布，涵盖了能够对数据进行推理以生成输出和决策的 **Agentic 工作流**，链接可在 [YouTube](https://t.co/jp4z2gFpoR) 获取。
  
  - 讨论的核心主题包括高级检索增强生成（RAG）技术，可以在[这里](https://t.co/NzfUhRyX2b)找到。
- **掌握 500 张表的 Text-to-SQL**：由 @kiennt_ 提供的可靠 Text-to-SQL 教程，演示了如何构建一个能够在 **500 张表**上运行的 SQL Agent，可在 [YouTube](https://t.co/nfRvlh3idI) 观看。
  
  - 该资源是处理复杂数据设置的最佳教程之一，更多信息请访问[这里](https://t.co/Fakrwdh0Op)。
- **实现 Agentic RAG 技术**：两段全新的完整教程视频即将发布，指导如何实现 **Agentic RAG 技术**，包括添加 LLM 层来处理输入，可在 [YouTube](https://t.co/1tDzsGfhUx) 找到。
  
  - 教程将涵盖使用 LLM 对向量数据库进行推理并应用元数据过滤器，更多细节请见[这里](https://t.co/PFgWjV4AIH)。
- **动态检索策略详解**：高级 RAG 系列包括一个即将推出的教程，演示**动态检索**（根据问题调整上下文检索），可在 [YouTube](https://t.co/MBGAOgJXPS) 抢先体验。
  
  - 该教程还将涉及 SQL 数据库的动态查询，更多见解可以在[这里](https://t.co/DenDMw5JVQ)探索。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1299462585086771213) (32 条消息🔥):

> - `NVDIA 案例研究 Cookbook`
> - `LlamaIndex 工作流与流式处理`
> - `LlamaIndex 中的 Retriever 问题`
> - `VectorStoreIndex 与 embeddings`
> - `RAG 解决方案的实习机会`

- **需要 NVDIA 案例研究的 Cookbook**：几位成员对 [NVDIA 案例研究](https://github.com/Chainlit/cookbook/pull/138) 的 Cookbook 表示了兴趣，特别是关注使用 Chainlit 的流式处理用例。
  
  - 一位成员提到在追求自定义 Agent 工作流时，在 Chainlit 框架内嵌套父/子步骤遇到了困难。
- **LlamaIndex 工作流指导**：一位 LlamaIndex 新手寻求关于如何进行带有循环的链式 LLM 调用以及同时处理多个请求的指导。
  
  - 一位资深成员建议使用 Workflows 来实现结构化输出，并提供了文档链接以供进一步参考。
- **Retriever 问题报告**：一位成员报告了 Retriever 返回空节点的问题，尽管使用 Chat Engine 测试 Index 时是成功的。
  
  - 另一位成员建议分享代码以便进一步排查，因为 Retriever 的配置似乎设置不当。
- **探索 VectorStoreIndex 和 Embeddings**：一位用户寻求帮助以查看 `VectorStoreIndex` 中生成的 Embeddings，但发现 Embeddings 返回为 `None`。
  
  - 一位专家指出，底层的 Vector Store 客户端通常不返回 Embeddings，以在操作期间节省内存。
- **RAG 解决方案的实习机会**：一位成员详细介绍了他们为企业客户构建 RAG 解决方案的经验，并表达了对即将到来的夏季实习机会的兴趣。
  
  - 他们邀请成员如果有人有软件工程或相关领域的实习生或初级职位线索，请与其联系。

**提到的链接**：

- [Workflows - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows)：未找到描述
- [为 tituslhy 的 LlamaIndex Workflow 抽象创建 Cookbook · Pull Request #138 · Chainlit/cookbook](https://github.com/Chainlit/cookbook/pull/138)：此 Cookbook 旨在提供一个如何将 LlamaIndex 最新的 Workflow 抽象与 Chainlit 结合使用的示例。
- [OpenAI - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/openai/#function-calling)：未找到描述
- [create-llama/templates/components/services/python/suggestion.py at main · run-llama/create-llama](https://github.com/run-llama/create-llama/blob/main/templates/components/services/python/suggestion.py)：开始使用 LlamaIndex 的最简单方法。通过在 GitHub 上创建账户来为 run-llama/create-llama 的开发做出贡献。

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1299865083111018556) (2 条消息):

> - `OpenAI 训练实践`
> - `Deepfake 语音能力`

- **OpenAI 的训练引发关注**：有人对 **OpenAI** 据称在未选择加入（opt-in）的个人数据上进行训练表示担忧，强调了隐私和知情同意问题。
  
  - 有人指出，这种做法可能会导致**意想不到的结果**，例如在未经明确许可的情况下生成个性化回复。
- **Deepfake 语音生成令人印象深刻**：一位用户体验了**令人印象深刻的 Deepfake 语音生成**，系统在 Teams Tier 计划中像代表用户回复一样自动预测回复。
  
  - AI 不仅用自己的声音提问，还用用户的声音回答，展示了**现实生活中的自动预测能力**。

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1299471550784733234) (12 条消息🔥):

> - `使用 MIPROv2 的自动提示词生成 (Automatic Prompt Generation)`
> - `协作式法律制定应用`
> - `适用于 Etherpad 的 DSPy 插件`
> - `关于古代手稿的独特研究`
> - `历史文献的翻译与解读`

- **自动提示词生成揭秘**：一位成员分享了一个关于实现 **自动提示词生成 (automatic prompt generation)** 技术的系列推文，该技术采用了 MIPROv2 优化器，使用 gsm8k 数据集，并将程序结构简化为三个模块。
  
  - 这些模块包括生成示例 (demos)、构建指令以及合并输出以生成最终提示词，提供了一种简洁的方法。
- **瑞士公民直接制定法律**：一位成员正在开发一个 **协作软件应用**，允许瑞士公民通过所谓的 [全民倡议 (popular initiative)](https://en.wikipedia.org/wiki/Popular_initiative_in_Switzerland) 流程讨论并创建法律。
  
  - 该应用是其硕士论文的一部分，展示了公民直接参与立法工作的力量。
- **DSPy 插件增强法律讨论**：一位成员为 **Etherpad 创建了 DSPy 插件**，以促进法律讨论并根据社区输入生成法律文本。
  
  - 这一创新工具旨在将讨论转化为可操作的法律草案，允许公众直接参与立法。
- **针对古代文献的 RAG 研究查询**：一位成员正在研究 **圣经语言**，旨在利用 DSPy 为现代读者翻译死海古卷 (Dead Sea Scrolls) 和埃及纸莎草纸 (Egyptian Papyri) 等古代手稿。
  
  - 他们正在寻求合作，以建立一个集成了历史文本分析和多语言文档处理的专用 RAG 系统。
- **对社区努力的赞赏**：一位成员对社区中其他人产出的令人印象深刻的项目表示赞赏，认可了他们的高产质量。
  
  - 这彰显了社区内对创新和协作的承诺，营造了令人振奋的氛围。

**提到的链接**：

- [来自 Karthik Kalyanaraman (@karthikkalyan90) 的推文](https://x.com/karthikkalyan90/status/1849902254373077196)：🧵 一个使用 MIPROv2 优化器技术的“自动提示词生成”简化实现。该程序使用由数学问题组成的 gsm8k 数据集，由……组成。
- [CustomGPTs](https://www.loom.com/share/badb94ad47cf40b7828a5decc5fbd5a9)：在这段视频中，我深入探讨了我的 GPTs 的演变，从 8 年前的第一个语言模型到最新版本 DSLmodels。我讨论了稀疏启动表示 (sparse priming representations) 的重要性……
- [介绍 DSL Model 框架！🚀](https://www.loom.com/share/67dd1db910ae424eb89e249e676bbaf0)：https://github.com/seanchatmangpt/dslmodel 嘿 DSLModel 社区，我是 Sean Chatman。我很高兴向大家介绍 DSL Model，这是一个彻底改变结构化文本和语言的框架……
- [来自 Karthik Kalyanaraman (@karthikkalyan90) 的推文](https://x.com/karthikkalyan90/status/1850730349195563176)：关于使用 DSPy 和来自 @dottxtai 的 Outlines 解决分类/命名实体识别 (name-entity recognition) 类问题的简短推文。这种方法不仅符合人体工程学且整洁，还能保证模式一致性 (schema adherence)……

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1299513453853147199) (27 messages🔥):

> - `DSPy 2.5 mapping`
> - `Audio input development`
> - `Liquid40B implementation`
> - `Named entity recognition examples`
> - `Relation extraction use cases`

- **关于 DSPy 2.5 映射的澄清**：成员们讨论了向 DSPy 2.5 的过渡，以及实现方式是否会与早期的 Notebook 有显著不同，并建议参考 [迁移文档](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb)。
  
  - 预计*没有重大差异*，强调现有的实现应该仍然有效。
- **音频输入开发咨询**：讨论了 DSPy 中音频输入功能的开发工作，参考了一个潜在的 [GitHub PR](https://github.com/stanfordnlp/dspy/tree/mm-adapter)。
  
  - 参与者分享了链接并讨论了像 Ultravox 这样的架构，它支持与 LLaMa 模型集成的音频功能。
- **在 DSPy 中实现 Liquid40B**：询问了将 Liquid40B 集成到 DSPy 中的情况，并质疑其是否能与 LiteLLM 配合使用。
  
  - 回复确认支持任何与 LiteLLM 兼容的模型。
- **命名实体识别（NER）示例**：一位成员分享了在 DSPy 中实现命名实体识别（NER）的代码片段，重点介绍了一种从文本输入中提取实体的方法。
  
  - 推荐使用通过 `dspy.ChainOfThought` 的新方法，因为它比已弃用的 `TypedPredictor` 方法更可靠，并展示了现代化的类型支持。
- **关系抽取用例**：成员们探讨了除 NER 之外的关系抽取示例的可用性，并建议了在项目过程中可能提供价值参考的相关数据集。
  
  - 提到了一个来自 Hugging Face 的数据集作为此任务的可能资源。

**提到的链接**：

- [Getting Started I - DSPy](https://dspy-docs.vercel.app/quick-start/getting-started-01/)：无
- [Hugging Face – 建设未来的 AI 社区。](https://huggingface.co/datasets?sort=trending&search=NER)：未找到描述
- [dspy/examples/vlm/mmmu.ipynb at mm-adapter · stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/blob/mm-adapter/examples/vlm/mmmu.ipynb)：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy
- [dspy/examples/migration.ipynb at main · stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb)：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy
- [GitHub - fixie-ai/ultravox: A fast multimodal LLM for real-time voice](https://github.com/fixie-ai/ultravox)：用于实时语音的快速多模态 LLM。通过在 GitHub 上创建账户为 fixie-ai/ultravox 的开发做出贡献。
- [GitHub - stanfordnlp/dspy at mm-adapter](https://github.com/stanfordnlp/dspy/tree/mm-adapter)：DSPy：用于编程（而非提示）基础模型的框架 - GitHub - stanfordnlp/dspy at mm-adapter
- [Google Colab](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/longformqa/longformqa_assertions.ipynb)：未找到描述
- [GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.](https://github.com/stanford-oval/storm)：一个由 LLM 驱动的知识策划系统，负责研究主题并生成带有引用的完整报告。 - stanford-oval/storm
- [Google Colab](https://colab.research.google.com/drive/1CpsOiLiLYKeGrhmq579_FmtGsD5uZ3Qe)：未找到描述
- [dspy/examples/coding/hackercup.py at main · stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/blob/main/examples/coding/hackercup.py)：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy
- [GitHub - jjovalle99/DSPy-Text2SQL: DSPY on action with OpenSource LLMs.](https://github.com/jjovalle99/DSPy-Text2SQL)：DSPY 与开源 LLM 的实际应用。通过在 GitHub 上创建账户为 jjovalle99/DSPy-Text2SQL 的开发做出贡献。

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1299450534771556434) (35 条消息🔥):

> - `Open Interpreter 性能`
> - `设置 Open Interpreter`
> - `本地模型限制`
> - `Beta 测试机会`
> - `OS Mode 灵活性`

- **Open Interpreter 在电子表格方面的性能**：成员们讨论了运行名为“通过电子表格提高 Open Interpreter 性能”的 [YouTube 视频](https://www.youtube.com/watch?v=4X4rKqtmxJg)中的测试，但像 **qwen2.5:32b-instruct** 这样的本地模型在执行步骤时表现挣扎。
  
  - 一位成员提到，**性能**取决于**模型质量**和 Prompt 工程，并建议创建一个配置文件（profile）来明确任务。
- **Open Interpreter 设置帮助**：一位初学者请求在 Windows 终端设置 Open Interpreter 的帮助，并提到在参考 GitHub 演示视频时遇到问题。
  
  - 另一位成员分享了[详细的设置指南](https://docs.openinterpreter.com/getting-started/setup)，包括通过 `pip` 安装的命令。
- **Open Interpreter 的本地模型限制**：一位用户询问本地模型是否需要视觉能力才能运行 Open Interpreter，但被告知由于限制，目前还没有本地模型能达到 **Sonnet** 的性能水平。
  
  - 有建议指出应确保正确导入 computer API，因为这对于本地模型有效执行操作至关重要。
- **对 Web3 平台 Beta 测试的兴趣**：一位成员提供了一个 **Web3 平台**的机会，正在寻找开发者、版主、Beta 测试员等，并邀请感兴趣的人私信了解详情。
  
  - 另一位成员表达了对担任 Beta 测试员的浓厚兴趣。
- **OS Mode 与不同 API 的灵活性**：关于在 OS mode 下使用不同 API 的咨询得到了澄清：通过修改代码应该是可行的，目前主要为 **Anthropic, Vertex, 和 Bedrock** 设置。
  
  - 一位成员提供了关于如何访问和修改 computer use 文件夹下的 **loop.py** 文件以实现这些更改的指导。

**提到的链接**：

- [Setup - Open Interpreter](https://docs.openinterpreter.com/getting-started/setup)：未找到描述
- [All Settings - Open Interpreter](https://docs.openinterpreter.com/settings/all-settings#import-computer-api)：未找到描述
- [Improve Open Interpreter Performance with Spreadsheets](https://www.youtube.com/watch?v=4X4rKqtmxJg)：你的 AI Agent 是否在多步骤任务中失去焦点？这里有一种方法可以帮助 Open Interpreter 保持专注，并加快 Prompt 工程的迭代...
- [open-interpreter/interpreter/core/computer/vision/vision.py at 36ec07125efec86594c91e990f68e0ab214e7edf · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/blob/36ec07125efec86594c91e990f68e0ab214e7edf/interpreter/core/computer/vision/vision.py#L22)：计算机的自然语言接口。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 做出贡献。

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1300109350941229146) (4 条消息):

> - `Markdown 使用`
> - `Obsidian 演示`
> - `Advanced Voice 功能`
> - `Apple AI 服务器黑客奖励`

- **Markdown 爱好者集合！**：一位成员表达了他们对 **Markdown** 的喜爱，称赞 **Obsidian** 非常棒，并暗示本周将发布一个酷炫的演示。
  
  - 这种热情反映了在 AI 编程场景中使用 Markdown 工具的兴趣日益增长。
- **OpenAI 扩大 Advanced Voice 访问权限**：OpenAI 宣布，其移动应用中的 **Advanced Voice** 功能现已向**欧盟、瑞士、冰岛、挪威**和**列支敦士登**的免费用户开放。
  
  - 这标志着这些地区移动用户在可访问性方面的重大提升。
- **Apple 的 100 万美元黑客挑战**：一条推文分享称，**Apple** 将向任何能成功入侵其 **AI 服务器**的人支付高达 **100 万美元**的奖金。
  
  - 这一消息引发了关于 AI 技术安全措施的有趣讨论。
- **令人兴奋的 AI 编程 YouTube 内容**：分享了一段标题为 **'Cline & Aider + Obsidian: AI CODING with MARKDOWN is AMAZING!'** 的 YouTube 视频，强调了在 AI 编程中结合使用 Cline、Aider 和 Markdown。
  
  - 该视频旨在展示规则和长 Prompt，突出了增强编程实践的创新方式。

**提到的链接**：

- [OpenAI 的推文 (@OpenAI)](https://x.com/openai/status/1850989317537349927?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：最后，Advanced Voice 现已在我们的移动应用中向欧盟、瑞士、冰岛、挪威和列支敦士登的免费用户开放。
- [Culture Crave 的推文 🎃 (@CultureCrave)](https://x.com/culturecrave/status/1850781293166067999?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：Apple 将向任何能入侵其 AI 服务器的人支付高达 100 万美元。
- [Cline & Aider + Obsidian : AI CODING with MARKDOWN is AMAZING! (Rules & Long Prompts!)](https://www.youtube.com/watch?v=ZHH9NP2mogg)：加入此频道以获得福利：https://www.youtube.com/@AICodeKing/join 在这段视频中，我将告诉你如何结合使用 Cline 和 Aider...

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1300062408018235455) (12 条消息🔥):

> - `Discord LLM 助手`
> - `频道内摘要`
> - `Voicebot 项目`
> - `机器人临时响应 (Ephemeral Responses)`

- **对 Discord LLM 助手的需求**：一位用户表示希望拥有类似 **'ask Discord' LLM 助手**的功能，能够根据需求总结对话并回答问题。
  
  - 另一位成员指出，虽然 Discord 有一个用于摘要的 **Beta 功能**，但目前不支持提问。
- **自定义机器人中的临时响应**：一位成员建议，创建一个自定义 **Discord 机器人**来处理问题和摘要可能比较简单。
  
  - 他们指出，**临时响应 (Ephemeral Responses)** 可以通过确保响应仅对执行命令的用户可见，从而使交互更加整洁。
- **关于活跃 Voicebot 项目的咨询**：一位用户询问 LAION 的 **voicebot** 项目是否仍然活跃，以及是否可以联系相关人员。
  
  - 这反映了社区对 LAION 在语音技术方面进展的持续关注和参与。

 

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1299864617929150464) (11 条消息🔥):

> - `Mindcraft 与 LLM`
> - `Llama3-8B-1.58 模型`
> - `关于模型规格的误解`

- **与 LLM 一起玩转 Mindcraft**：人们正在参与将 **Minecraft** 与 **LLM** 结合的有趣项目。
  
  - 正如讨论中所提到的，*这些都是很有趣的东西。*
- **关于 Llama3-8B-1.58 模型的澄清**：**Llama3-8B-1.58** 模型衍生自基础模型 **Llama-3-8B-Instruct**，并非基于 **BitNet** 架构，这与某些说法相反。
  
  - 有关详细信息，他们参考了一篇[关于极端量化的博客文章](https://huggingface.co/blog/1_58_llm_extreme_quantization)。
- **对模型参数的困惑**：对于模型链接中提到的 **100B** 存在一些误解，其他人澄清这与该模型具有 **8B 参数**有关。
  
  - 成员们表示这种困惑很常见，并感谢彼此之间的纠正。

**提到的链接**：

- [HF1BitLLM/Llama3-8B-1.58-100B-tokens · Hugging Face](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens)：未找到描述
- [FxTwitter / FixupX 的推文](https://x.com/search?q=mindcraft&src=typed_query)：抱歉，该用户不存在 :(
- [GitHub - kolbytn/mindcraft](https://github.com/kolbytn/mindcraft)：通过在 GitHub 上创建账户为 kolbytn/mindcraft 的开发做出贡献。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1300438744574201857) (2 messages):

> - `Mixtral AI 升级`
> - `SymNoise 实现`

- **Mixtral AI 被认为已过时**：一位成员幽默地建议从 **Mixtral AI** 模型升级到更新版本的 **MistralAI**，暗示其已经过时。
  
  - *至少升级到较新的 MistralAI 模型。*\*
- **寻求 SymNoise 技术的代码**：一位成员询问了一篇介绍 **SymNoise** 微调技术论文的代码实现，该技术通过对称噪声来增强语言模型。
  
  - *尝试过自己实现，但它似乎通过拼接（concatenation）使 embedding 的 batch size 翻倍了，我不知道该如何处理。*

 

**提到的链接**：[SymNoise: Advancing Language Model Fine-tuning with Symmetric Noise](https://arxiv.org/abs/2312.01523)：在这篇论文中，我们介绍了一种新型的语言模型微调技术，涉及将对称噪声引入 embedding 过程。该方法旨在增强模型的功...

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**datasets**](https://discord.com/channels/1104757954588196865/1112023441386778704/1299889141764456479) (11 messages🔥):

> - `SAT 阅读测试抓取`
> - `用于问题的 Sonnet 格式化`
> - `原始答案的存在`
> - `多模态问题的问题`
> - `数据集共享`

- **观察到 SAT 阅读测试的抓取不完整**：一位成员完成了一个不完整的 **SAT 阅读测试**和几个 **AP 测试**的抓取，并引发了关于格式化结果的讨论。
  
  - *感谢你让我注意到这一点！* 对关于抓取反馈的表示感谢。
- **关于包含图像的问题**：在注意到 `formatted_prompt` 和 `rewritten_answer` 字段的存在后，有人担心是否应该在问题中包含图像。
  
  - 原抓取者提到，**完整集合确实包含某些问题的图像**，但意图是让数据集保持单模态（unimodal）。
- **关于原始答案和潜在错误的澄清**：一位成员询问是保留了原始答案还是使用 Sonnet 生成的，这引发了对潜在错误的担忧。
  
  - 抓取者确认他们拥有从其他来源抓取的**原始答案和原理（rationale）**作为参考。
- **使用 Sonnet 改进格式**：使用 Sonnet 的主要动力是增强答案和原理的视觉**格式化**，避免直接以答案开头。
  
  - 这样做是为了更清晰地展示每个问题涉及的细节及其推理过程。
- **提供数据集共享链接**：一位成员分享了 Hugging Face 上的数据集链接，特别是 **Dans-Logicmaxx-SAT-AP**。
  
  - 链接访问地址：[Dans-Logicmaxx-SAT-AP](https://huggingface.co/datasets/PocketDoc/Dans-Logicmaxx-SAT-AP)。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-help-bot**](https://discord.com/channels/1104757954588196865/1225300056442409040/1299708809111146597) (8 messages🔥):

> - `Qwen 模型配置`
> - `微调参数`
> - `DeepSpeed 集成`
> - `训练 Token 定义`

- **关于 Qwen 模型配置的讨论**：成员们讨论了用于微调 `Qwen/Qwen2.5-32B` 的配置，强调可能需要指定确切的模型类型，而不是使用 `AutoModel` 和 `AutoTokenizer` 等占位符。
  
  - 有人担心如果使用默认值，则不需要 `base_model_config`，以及将 `trust_remote_code` 设置为 true 的安全性影响。
- **DeepSpeed 配置检查**：重点在于验证 `deepspeed` 配置路径，以确保其指向一个现有的且设置正确的优化训练文件。
  
  - 成员们指出，配置错误可能会阻碍性能，并强调要确认优化器设置与 DeepSpeed 要求的兼容性。
- **特殊 Token 使用的澄清**：强调了正确集成定义的特殊 Token 并确保它们在模型和分词器（tokenizer）中正常工作的重要性。
  
  - 未能集成这些 Token 可能会导致训练和推理阶段出现问题。
- **关于知识盲点的轻松交流**：一位成员开玩笑地承认自己并不了解配置过程的所有细节，并欢迎对潜在问题进行进一步澄清。
  
  - 随着成员们意识到微调配置中固有的复杂性，现场气氛变得轻松起来。

 

**提到的链接**：[OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c66c6e9d-0f02-4052-ad80-2ea10a0a22d3)：更快地理解代码。

 

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1299619488882233344) (7 条消息):

> - `ReAct Agent with HuggingFace`
> - `Advanced RAG methods`
> - `Using create_sql_agent with Pandas`
> - `Public benchmarks for RAG systems`
> - `Image handling in Langchain`

- **使用 HuggingFace 本地模型创建 ReAct Agent**：一位成员分享了他们当前使用本地模型初始化 ReAct Agent 的方法，并在调用过程中遇到了 **parserException**。
  
  - 他们请求帮助，因为在网上找不到针对该特定错误的解决方案。
- **高级 RAG 方法探索**：有人提出了关于目前最先进的 **Retrieval-Augmented Generation (RAG)** 方法以及传统方法是否仍然适用的问题。
  
  - **Pinecone/vector databases** 中的数据清洗和存储被提及为常用实践，同时也在寻找最近的参考资料。
- **使用 create_sql_agent 返回 Pandas DataFrame**：一个关于如何利用 **create_sql_agent** 输出 **Pandas DataFrame** 而不仅仅是文本字符串的查询。
  
  - 该成员特别询问了在这种情况下是否需要 **SQLDatabaseToolkit**。
- **RAG 系统的公开基准测试**：询问是否有专门针对 RAG 系统的 **public benchmarks**。
  
  - 有人建议考虑将 **RAGAS** 作为评估 LLM 应用的潜在资源。
- **Langchain 中便捷的图像处理**：讨论了在 **Langchain** 消息中处理多张图片的便捷方法的需求。
  
  - 这突显了社区对语言模型框架内改进图像处理能力的兴趣。

 

**提及的链接**：[GitHub - explodinggradients/ragas: Supercharge Your LLM Application Evaluations 🚀](https://github.com/explodinggradients/ragas)：增强你的 LLM 应用评估 🚀。通过在 GitHub 上创建一个账户来为 explodinggradients/ragas 的开发做出贡献。

 

---

### **LangChain AI ▷ #**[**share-your-work**](https://discord.com/channels/1038097195422978059/1038097372695236729/1299619922137190400) (6 条消息):

> - `AdaletGPT`
> - `bootstrap-rag v0.0.11`
> - `Appine`
> - `Financial Agentic System`
> - `Wordle Clone Tutorial`

- **AdaletGPT 简介 - 一个土耳其法律聊天机器人**：AdaletGPT 是一个基于 RAG 的 **土耳其法律聊天机器人**，利用 LangChain、Pinecone 和 OpenAI 来协助用户。
  
  - 该平台旨在通过 AI 驱动的交互来促进法律咨询和服务。
- **bootstrap-rag v0.0.11 发布，带来令人兴奋的更新**：**bootstrap-rag** v0.0.11 的发布包含了一个 **LLM as Judge 模板**，现在已与 Arize AI Phoenix 集成，以增强可观测性和分析能力。
  
  - 此次更新还带来了关键的错误修复和文档改进，优化了用户体验。
- **Appine：让无代码 AI 应用创建变得简单**：Appine 提供了一个用于构建和分享 AI 驱动应用的无代码平台，允许用户通过拖拽界面可视化地创建应用程序。
  
  - 作为最小可行性产品（MVP），它目前是免费的，并将根据用户反馈进行进一步增强。
- **构建了创新的金融 Agent 系统**：一个新的 **金融报告系统**，结合了 Langgraph、GROQ 和各种 API 进行实时分析，收集股票价格和损益表等数据。
  
  - 它具有灵活的执行序列、强大的错误处理能力，并支持包括 OpenAI 在内的多个 LLM 提供商。
- **构建 Wordle 克隆的快速教程**：发布了一个教程，详细介绍了如何使用 V0 和 Next.js 在 **30 分钟** 内创建一个 **Wordle 克隆**，面向所有编程水平的用户。
  
  - 该资源旨在启发并引导用户创建自己的简单游戏应用。

**提及的链接**：

- [no title found](https://adaletgpt.com)：未找到描述
- [bootstrap-rag](https://pypi.org/project/bootstrap-rag/)：无
- [Release v0.0.11 · pavanjava/bootstrap-rag](https://github.com/pavanjava/bootstrap-rag/releases/tag/v0.0.11)：更新内容：由 @pavanjava 在 #62 中修改 maindoc；由 @pavanjava 在 #63 中修改 maindoc；由 @pavanjava 在 #64 进行可观测性迁移；由 @pavanjava 在 #65 进行可观测性迁移...
- [Appine](https://appine.tech/)：未找到描述

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1300548204361945249) (1 messages):

> - `Lecture 8`
> - `Yuandong Tian 的演讲`
> - `Neural vs Symbolic Decision Making`

- **Lecture 8 即将开始！**: **第 8 讲** 将于今天 **3:00pm PST** 举行。你可以在[这里](https://www.youtube.com/live/wm9-7VBpdEo)观看直播。
  
  - 务必不要错过这次特别讲座中分享的见解！
- **Yuandong Tian 登台**: 今天的客座讲师是 **Yuandong Tian**，他是 Meta AI Research 的研究科学家总监。他将讨论 **集成 Neural 和 Symbolic 决策框架** 以处理复杂任务。
  
  - 他的演讲探讨了与传统的 Symbolic Solvers 相比，**LLM** 如何增强推理能力。
- **理解 Neural 与 Symbolic 的集成**: 演讲将探讨 **Neural 模型如何凭借灵活性脱颖而出**，而 Symbolic Solvers 则提供有保证的解决方案。目标是找到一个能够利用两种方法优势的 **统一框架**。
  
  - 这种集成对于提高 LLM 在复杂推理任务中的性能至关重要。
- **有问题？联系我们！**: 关于讲座的任何问题，可以在 **指定频道** 联系课程工作人员。这是一个与课程材料互动并寻求澄清的好机会。
  
  - 不要犹豫，寻求帮助以加深你对所讨论主题的理解！

 

**提到的链接**: [CS 194/294-196 (LLM Agents) - Lecture 8](https://www.youtube.com/live/wm9-7VBpdEo.): 未找到描述

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1300185689517981778) (8 messages🔥):

> - `MOOC 报名的确认邮件`
> - `同行学习小组倡议`
> - `Hackathon 时间线和赛道`
> - `Benchmarking 赛道的数据集`

- **MOOC 报名无确认邮件**: 成员们确认，在报名 LLM Agents MOOC 后 **不会发送确认邮件**，只有一份标题为 'LM Agents MOOC Signup Form' 的 Google Form 响应副本。
  
  - 参与者在报名后可以期待收到 **每周课程提醒邮件**，但不会提供额外的确认。
- **同行学习小组建议**: 一位成员建议为较晚加入课程的人创建一个 **Zoom 同行学习小组**，以方便讨论和头脑风暴。
  
  - 虽然官方没有提供正式资源，但鼓励其他人组建自己的小组或加入现有小组。
- **Hackathon 时间线和细节公布**: Hackathon 的时间线已在官方 [Hackathon 网站](https://rdi.berkeley.edu/llm-agents-hackathon/)上公布，详细列出了包括 **Applications**、**Benchmarks**、**Fundamentals** 和 **Safety** 在内的各个赛道。
  
  - 该活动由 **Berkeley RDI** 主办，旨在团结学生、研究人员和行业从业者，共同推动 LLM Agent 技术的发展。
- **Benchmarking 赛道对数据集的需求**: 一位成员询问了关于 Hackathon 的 Benchmarking 赛道的 **合适数据集**，表示对项目的相关资源感兴趣。
  
  - 目前尚未收到针对数据集的回复或具体资源，该询问仍处于开放讨论状态。

 

**提到的链接**: [LLM Agents Hackathon](https://rdi.berkeley.edu/llm-agents-hackathon/): 由 UC Berkeley RDI 主办的 LLM Agents Hackathon。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1300552292797517854) (2 messages):

> - `学习小组组建`
> - `合作意向`

- **组建学习小组的提议**: 一位成员提议统计组建学习小组的意向，用于讨论讲座和阅读材料，并为虚拟会议建议了多个时间段。
  
  - 他们提供了一个 [Google Form](https://forms.gle/QtQ2C6qzomeHDrC38) 链接，供其他人表达他们的空闲时间和兴趣。
- **对学习小组想法的支持**: 另一位成员对学习小组的提议表示热烈支持，称这个主意很酷。
  
  - 这显示了对于促进后期加入者之间协作的积极响应。

 

**提到的链接**: [LLM Agents Peer Study Group (Virtual)](https://forms.gle/QtQ2C6qzomeHDrC38): 使用此表单表达你加入虚拟同行学习小组的兴趣。我们可能会使用 Discord 活动或 Zoom。我们将从第一讲开始复习讲座，并讨论额外的……

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1300164394809950219) (4 条消息):

> - `Embedding Config Flags`
> - `LoRA Bug Fix`
> - `TransformerDecoder Changes`

- **Embedding 和 Norm 层配置提案**：一名成员建议在配置中公开**两个布尔标志**（`embedding_trainable=False` 和 `norms_trainable=False`），而不是名称列表，以防止未来层名称更改时出现配置问题。
  
  - 该想法遵循的逻辑是，从布尔标志过渡到列表的破坏性比调整无数配置要小，特别是考虑到 **TransformerDecoder** 可能需要针对各种模型类型进行更具表现力的更改。
- **提交 LoRA Bug 修复**：一名成员通过 [pull request #1909](https://github.com/pytorch/torchtune/pull/1909) 提交了 **LoRA** bug 的修复，解决了单设备微调 Checkpoint 保存以及 `use_dora=True` 时出现 NaN loss 的问题。
  
  - 他们对该修复是否在所有 Recipe 中都有效表示不确定，并指出分布式设置中存在 bug，以及单设备使用时会出现 NaN 结果。
- **关于配置灵活性的讨论**：另一名成员对配置提案发表了看法，承认了布尔标志和名称选项各自的优缺点，并表示倾向于灵活性。
  
  - 这表明大家在配置适应性以满足未来需求的重要性上达成了共识。

 

**提到的链接**：[Fix lora single device fine tune checkpoint saving & nan loss when use_dora=True by mirceamironenco · Pull Request #1909 · pytorch/torchtune](https://github.com/pytorch/torchtune/pull/1909)：上下文 此 PR 的目的是什么？是添加新功能、修复 bug、更新测试和/或文档还是其他（请在此处添加）修复了 #1903。变更日志 此 PR 中做了哪些更改...

 

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1299684792564187178) (5 条消息):

> - `Hyperparameter Optimization Recipe`
> - `Discussion on muP Utility`
> - `Priority Issues in Development`
> - `External Tools for Tuning`

- **在 Torchtune 中探索超参数优化**：一个 GitHub [issue](https://github.com/pytorch/torchtune/issues/1752) 讨论了创建一个用于超参数优化的 Recipe，用户可以提供配置以及数据集和要扫描的参数。
  
  - 该建议强调了对网格格式通用默认值的需求，尽管有人指出目前*还没有人明确提出过这一需求*。
- **辩论 muP 在微调中的有用性**：成员们对 **muP** 在微调中的效用表示怀疑，其中一人提到它主要是在预训练的背景下被提及。
  
  - *有人担心*，相比于实现 muP，应该优先处理其他问题，如更快的生成和 early stopping。
- **发现更高优先级的问题**：一名成员指出，有许多待解决的问题应该优先处理，包括更快的强化学习生成和使用 LLM 进行更好的分类。
  
  - 他们特别强调了管理 **200 个待解决问题** 的挑战，并建议将支持 distributed shampoo 作为另一个优先级。
- **依赖现有的调优工具**：讨论转向利用现有的外部解决方案（如 **Wandb sweeps** 和 **Ray Tune**）进行超参数调优，而不是在 Torchtune 中加入新的脚本。
  
  - 一位参与者表示倾向于将这些功能排除在库之外，因为现有工具已经提供了合适的解决方案。

 

**提到的链接**：[recipe for hyperparameter sweep · Issue #1752 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1752)：Torchtune 可以提供一个执行 HPO 的 Recipe，用户提供配置、Recipe、评估数据集、要扫描的参数和预算。我刚刚尝试了优化器。我们的默认 lr 是 3e-4。我尝试了 3e-...

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1299472683213520960) (2 条消息):

> - `Human Native AI Marketplace`
> - `十一月成员活动安排`
> - `公共 AI 活动`
> - `OSS4AI 旧金山见面会`
> - `Sqlite-Vec 元数据过滤`

- **Human Native AI Marketplace 上线**：由 [Human Native AI](https://www.humannative.ai/) 推出的全新 AI 数据市场，允许创作者授权其内容用于 AI 训练并获得报酬。
  
  - 联合创始人 **James Smith** 将在即将举行的活动中讨论他们的进展，该活动是 Mozilla **Data Futures Lab Speaker Series** 的一部分。
- **精彩的十一月成员活动安排**：十一月将带来一系列由成员组织的活动，涵盖 Sqlite-Vec 和 Refact.ai 等主题，此外还有远程会议和在旧金山举行的线下见面会。
  
  - 鼓励成员根据活动说明进行 [RSVP](https://discord.com/channels/1089876418936180786/1254956456772505601) 以参加这些重要的聚会。
- **展示成员项目**：在 Mozilla AI 舞台上展示的重点项目包括 **Open Interpreter**、**Homebrew** 以及 **Sentry.io 的开源自动修复 (auto fix)** 等。
  
  - 社区期待在 **Public AI** 中展示更多成员的开源成果，强调了 3300 名成员的贡献。
- **即将举行的 OSS4AI 旧金山 IRL 见面会**：[OSS4AI 旧金山 IRL 见面会](https://discord.com/events/1089876418936180786/1299558554872582154) 即将举行，诚邀感兴趣的成员参与交流并分享见解。
  
  - 这为当地成员提供了参与协作项目和讨论的机会。
- **讨论 Sqlite-Vec 中的元数据过滤**：关于 [Sqlite-Vec 中的元数据过滤](https://discord.com/events/1089876418936180786/1300483739872399411) 的活动将探讨高效处理数据的重要技术。
  
  - 该倡议强调了在为 AI 训练做出贡献的同时维护数据完整性的重要性。

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**leaderboard**](https://discord.com/channels/1111172801899012102/1214705495974092810/1300560060182954134) (2 条消息):

> - `排行榜 multiple 功能`
> - `GitHub 示例说明`

- **关于排行榜中 'Multiple' 功能的澄清**：一位用户询问排行榜上的 'multiple' 指的是什么，质疑其是否涉及多个查询、函数或步骤。
  
  - 作为回应，有人指出 'multiple' 似乎表示从多个选项中选择正确函数的能力，但多步 (multi-step) 功能的评估仍不明确。
- **引用 GitHub 示例**：机器人提供了一个 [GitHub 链接](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d9f323715d7d2012a955d7f4114e/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json#L42C1-L42C2438) 作为随机示例。
  
  - 该 GitHub 页面详细介绍了 'Gorilla: Training and Evaluating LLMs for Function Calls' 项目，为理解排行榜功能提供了背景。

 

**提到的链接**：[gorilla/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json at 2101b11f6d03d9f323715d7d2012a955d7f4114e · ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/blob/2101b11f6d03d9f323715d7d2012a955d7f4114e/berkeley-function-call-leaderboard/data/BFCL_v3_exec_multiple.json#L42C1-L42C2438))：Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla

 

---

---

---

---

---

{% else %}

> 完整的各频道详细内容已在邮件中截断。
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}