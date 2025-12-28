---
companies:
- answer-ai
- tencent
- notebooklm
- motherduck
- perplexity
- dropbox
- openai
- meta-ai-fair
- yi-ai
- zyphra-ai
- anthropic
- langchain
- openai
date: '2024-10-18T01:13:21.878974Z'
description: '以下是该文本的中文翻译：


  **Answer.ai** 推出了 **fastdata**，这是一个利用 `claudette` 并参考腾讯《Billion Persona》论文开发的合成数据生成库。**NotebookLM**
  现已支持定制化，**Motherduck** 在 SQL 实现中引入了重要的大语言模型（LLM）功能。**Perplexity** 和 **Dropbox**
  宣布推出针对 **Glean** 的竞争产品。**OpenAI** 发布了语音聊天补全功能（audio chat completions），定价为每分钟 24
  美分。**Meta AI** 发布了 **Llama 3.1**，为联想 AI Now 的端侧智能体提供动力。**Yi-Lightning** 模型全球排名升至第
  6 位，超越了 **GPT-4o**。**Zyphra AI** 发布了包含 5 万亿 token 的大型数据集 **Zyda-2**。**François Chollet**
  澄清称，Transformer 架构本质上是“集合处理”而非“序列处理”。研究表明，记忆有助于提升大语言模型的推理能力。**Anthropic** 更新了其旨在保障
  AI 安全的《负责任扩展政策》（Responsible Scaling Policy）。文中还重点介绍了 **Perplexity Finance**、**LangChain**
  开发的 **Open Canvas** 以及 **AlphaCodium** 代码生成工具。AI 智能体初创公司共筹集了约 5 亿美元资金，关于 AI 对就业市场影响的讨论仍在持续。此外，将提示词缓存（prompt
  caching）与 Batches API 结合使用，可使 **Claude 3.5 Sonnet** 的 token 成本降低 95%。'
id: b8bacc64-9fc5-4299-b895-acc178b286d2
models:
- claudette
- llama-3-1
- yi-lightning
- gpt-4o
- claude-3.5-sonnet
original_slug: ainews-not-much-happened-today-7086
people:
- fchollet
- aravsrinivas
- svpino
- swyx
title: 今天没发生什么特别的事。
topics:
- synthetic-data
- fine-tuning
- sql
- audio-processing
- on-device-ai
- dataset-release
- transformer
- llm-reasoning
- ai-safety
- code-generation
- ai-pricing
- ai-job-market
---

<!-- buttondown-editor-mode: plaintext -->**lots of small ships is all you need.**

> 2024/10/16-2024/10/17 的 AI 新闻。我们为您检查了 7 个 Reddit 子版块、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord 社区（**228** 个频道，**2989** 条消息）。为您节省了预计阅读时间（以 200wpm 计算）：**280 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

- Answer.ai 发布了 [fastdata](https://www.answer.ai/posts/2024-10-15-how-to-synthesize-data.html)，这是一个合成数据生成库，使用了 `claudette` + 腾讯的 [Billion Persona 论文](https://arxiv.org/abs/2406.20094v1)。
- NotebookLM [终于支持自定义了](https://x.com/raiza_abubakar/status/1846944566689353838)。
- Motherduck 发布了一个[值得关注的 SQL 中的 LLMs 实现](https://motherduck.com/blog/sql-llm-prompt-function-gpt-models/)。
- [Perplexity](https://x.com/perplexity_ai/status/1846950770736091509?s=46) 和 [Dropbox](https://dash.dropbox.com/) 都宣布了他们的 Glean 竞争产品。
- 正如在 Devday 预告的那样，OpenAI 宣布了 [audio chat completions](https://platform.openai.com/docs/guides/audio)，价格较高，为每分钟 24 美分。

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

- **Llama 3 发布**：[@AIatMeta](https://twitter.com/AIatMeta/status/1846330458189320534) 宣布发布 Llama 3.1，该模型正被用于 Lenovo AI Now，这是一款端侧 AI Agent，支持从文档管理到内容生成的各种功能。

- **Yi-Lightning 模型**：[@01AI_Yi](https://twitter.com/01AI_Yi/status/1846339181863473443) 宣布发布 Yi-Lightning，目前排名世界第 6，超越了 5 个月前发布的原始 GPT-4o。该公司在 @lmarena_ai Chatbot Arena 中排名 LLM 厂商第 3 位。

- **Zephyr AI 数据集**：[@ZyphraAI](https://twitter.com/rohanpaul_ai/status/1846288338913054734) 发布了 Zyda-2，这是一个包含 5 万亿 token 的宽松许可数据集，由 DCLM、FineWeb-Edu、Zyda-1 和 Dolma v1.7 的 Common Crawl 组成。该数据集的表现优于单个组件数据集，在其上训练的模型在下游任务中表现出更强的性能。

**AI 研究与技术**

- **Transformer 架构**：[@fchollet](https://twitter.com/fchollet/status/1846263128801378616) 解释说 Transformer 是一种集合处理（set-processing）架构，而不是序列处理（sequence-processing）。它们是顺序无关的，位置感知是通过 position embeddings 在特征层面添加的。

- **LLM 推理**：一篇[论文](https://twitter.com/rohanpaul_ai/status/1846302588167192766)建议，记忆可以增强 LLM 的真实推理能力，使模型能够更好地泛化到新的、多样化的问题。

- **AI 安全**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1846194917720088721) 发布了其 Responsible Scaling Policy 的更新，将安全和安保措施与 AI 模型的能力相匹配。

**AI 工具与应用**

- **Perplexity Finance**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1846289701822677441) 强调了 Perplexity Finance，提供实时股票价格、深入的公司财务分析以及具有用户友好界面的多家公司对比。

- **Open Canvas**：[@LangChainAI](https://twitter.com/LangChainAI/status/1846215982765035677) 推出了 Open Canvas，这是一个开源 Web 应用程序，用于与 Agent 协作编写文档，具有内置记忆功能，并能从现有文档开始。

- **AlphaCodium**：[@svpino](https://twitter.com/svpino/status/1846201354332893220) 报道了 AlphaCodium，这是一个开源的 SOTA 代码生成工具，在 Codeforces Code Contest 基准测试中表现优于 OpenAI 模型的直接提示（direct prompting）。

**AI 行业与市场趋势**

- **AI Agent 初创公司**：[@swyx](https://twitter.com/swyx/status/1846305962841280667) 指出，本月约有 5 亿美元融资用于 AI Agent 初创公司，目前尚未发现其中有公司使用来自其他初创公司的 AI Agent 框架。

- **AI 就业市场**：[@svpino](https://twitter.com/svpino/status/1846297492499190013) 评论了关于 AI 对就业影响的持续讨论，指出距离他被告知 AI 将取代他的工作已经过去了 685 天。

- **AI 定价**：[@alexalbert__](https://twitter.com/alexalbert__/status/1846265564852809854) 指出，结合 prompt caching 和新的 Batches API 可以使 Claude 3.5 Sonnet token 获得 95% 的折扣。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Ollama 与 4.5 万个 Hugging Face GGUF 模型集成**

- **PSA：你可以非常轻松地在本地克隆任何 Hugging Face "Spaces" 设置** ([Score: 40, Comments: 1](https://reddit.com//r/LocalLLaMA/comments/1g51saf/psa_you_can_clone_any_huggingface_spaces_setup/))：**Hugging Face Spaces** 可以轻松克隆并在本地运行，提供了一种通过可视化界面快速设置和使用模型的便捷方式。该过程包括**克隆 Space 仓库**、创建**虚拟环境**、安装依赖项并运行应用，正如针对某个**文本转语音模型**的示例命令序列所演示的那样。
- **你现在可以直接使用 Ollama 运行 Hugging Face Hub 上的 4.5 万个 GGUF 模型中的任何一个** ([Score: 314, Comments: 63](https://reddit.com//r/LocalLLaMA/comments/1g4zvi5/you_can_now_run_any_of_the_45k_gguf_on_the/))：**Ollama** 现在支持直接运行 **Hugging Face Hub** 上的 **45,000 个 GGUF 模型**，无需更改 Ollama 设置。用户可以使用命令 `ollama run hf.co/{username}/{reponame}:latest` 运行模型，并可选择指定量化类型如 `Q8_0`。欲了解更多信息，用户可以参考 [Hugging Face 文档](https://huggingface.co/docs/hub/en/ollama)。
  - **Ollama** 与 **Hugging Face Hub** 的集成被视为一项重大改进，允许用户直接运行 **45,000 个 GGUF 模型**而无需手动配置。此次更新将下载、安装和运行模型的过程简化为单个命令。
  - 用户讨论了对 **OpenWebUI** 的影响，确认可以直接在界面内从 Hugging Face 拉取模型。一些人对 **Vulkan 支持**表示兴趣，以期在无需大量依赖项的情况下提高 Linux 系统上的性能。
  - 针对这一新集成，用户提出了关于模型存储位置、运行先前下载的模型而无需转换的能力，以及对**视觉模型**、**文本转图像模型**和 **TTS/STT** 功能潜在支持的问题。


**主题 2. Mistral AI 的新 Ministral 模型与许可争议**

- **[Un Ministral, des Ministraux](https://mistral.ai/news/ministraux/)** ([Score: 39, Comments: 10](https://reddit.com//r/LocalLLaMA/comments/1g50sbn/un_ministral_des_ministraux/))：Mistral AI 发布了新模型，包括 **Mistral 7B**、**Mixtral 8x7B** 和 **Mistral Small**，后两者采用**商业许可**。公司限制访问并对某些模型征收许可费的决定引发了关于 AI 开发中开源原则与商业利益平衡的辩论。Mistral 这种做法的转变与其最初对开源模型的承诺形成鲜明对比，并引发了关于 AI 模型分发和可访问性未来方向的疑问。
  - **Mistral 的新模型**引发了关于开源与商业 AI 开发的辩论。一些用户对**限制性许可**表示失望，其中一位表示“没有 Apache Licence，就毫无意义”。
  - 新模型的**多语言能力**被视为最大的进步，尽管一些用户认为这并不特别令人兴奋。其他人则期待尝试这些模型，希望它们能像之前的 Mistral 产品一样“以小博大”。
  - 8B 模型的**研究许可**被一些人看好用于 ERP 研究。然而，对于 3B 模型缺乏权重以及 8B 许可的限制性性质存在担忧。
- **为什么在 8B 和 70B 之间没有 Llama 的中间版本？** ([Score: 46, Comments: 80](https://reddit.com//r/LocalLLaMA/comments/1g4wul3/why_ther_is_no_middle_ground_version_of_llama/))：该帖子质疑在 **8B 和 70B** 参数之间缺乏中型 Llama 模型，突显了拥有 **8-16GB GPU** 的用户在选择上的空白。作者指出，虽然 **4GB 3050 GPU** 可以胜任运行 **8B 模型**，但对于无法处理 **70B 模型** 的更强大消费级 GPU，没有合适的选择。他们建议开发一个 **16B 参数模型** 来填补 Llama 模型阵容中的这一空白。
  - 用户讨论了**家庭实验室**和**消费级 AI 硬件**的潜力，一些人建议发烧友可能很快就会拥有用于 AI 处理的个人“硬件大脑”。
  - **Meta 的 Llama 模型**并非针对消费级 GPU 设计；**8B 模型**被视为“本地”版本，而更大的模型则针对数据中心。一些用户推荐了 **Gemma 2 的 9B 和 27B 模型** 作为理想的中型选择。
  - 社区讨论了中型 Llama 模型的缺失，提到了 **32.5B 原始模型** 和一个失败的 **Llama 2 中型版本**。一些人建议尝试其他模型，如 **Qwen2.5 14B**，据报道其性能优于 **Llama 3.1 8B**。

- **[Mistral 发布新模型 - Ministral 3B 和 Ministral 8B！](https://i.redd.it/45hs1duoq4vd1.png)** ([Score: 313, Comments: 74](https://reddit.com//r/LocalLLaMA/comments/1g50x4s/mistral_releases_new_models_ministral_3b_and/))：Mistral 发布了两个新模型 **Ministral 3B** 和 **Ministral 8B**，声称性能较之前版本有所提升。该公司断言 **Ministral 8B** 在大多数基准测试中优于 **Llama 2 13B**，而 **Ministral 3B** 据称达到或超过了 **Llama 2 7B** 的性能，这可能为处理小规模语言模型的开发者和研究人员带来显著的效率提升。
  - **Qwen2.5** 在大多数基准测试中优于 Mistral 的新模型，用户注意到它在 7B/8B 规模下的 **HumanEval**（84.8 vs 76.8）和 **MATH**（75.5 vs 54.5）表现更出色。一些人称 Mistral 的发布具有“欺骗性”，因为它忽略了与 **Qwen2.5** 的对比。
  - 尽管 **Ministral 3B** 模型宣传是面向边缘设备的，但目前仅通过 API 提供。用户对许可条款表示失望，注意到 8B 模型仅限于非商业用途，除非协商商业许可。
  - 讨论围绕 llama.cpp 中 **interleaved sliding-window attention**（交错滑动窗口注意力机制）的实现展开，用户参考了一个关于 Gemma2 支持的 [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/8227)，并推测 Mistral 模型可能需要的转换代码。


**主题 3. 配备 4xRTX4090 的 Threadripper**

- **[6U Threadripper + 4xRTX4090 配置展示](https://i.redd.it/h1ic1yk6h3vd1.jpeg)** ([Score: 774, Comments: 182](https://reddit.com//r/LocalLLaMA/comments/1g4w2vs/6u_threadripper_4xrtx4090_build/))：展示了一个配备 **6U Threadripper** 处理器和 **4 块 RTX 4090** 显卡的高性能 AI 构建。这种强大的配置旨在处理高要求的 AI 和机器学习任务，利用了 NVIDIA 顶级 GPU 和 AMD 高核心数 CPU 的计算能力。
  - 该配置引发了关于 **功耗** 的讨论，估计耗电量达 **3 kW**，并引发了对电费的担忧。用户们争论投资这种设备的人是否会担心电费成本。
  - 分享了构建细节，包括 **Threadripper Pro 7965WX**、**256GB RAM** 和 **两个 PSU**（1500W 和 1300W）。系统使用 **水冷**，配备 2 个散热器和多个 360mm 风扇。
  - 用户询问了性能情况，原作者指出在 24 小时负载测试期间，**GPU 最高温度为 79-81°C**。一些人建议对于预制的顶级性能系统，可以参考 [renderboxes.com](https://renderboxes.com/) 等替代方案。


**主题 4. Meta 的 TPO 技术提升 LLM 性能**

- **Meta 的新论文披露了 TPO (Thought Preference Optimization) 技术，结果令人印象深刻** ([Score: 43, Comments: 6](https://reddit.com//r/LocalLLaMA/comments/1g51w11/new_paper_from_meta_discloses_tpo_thought/))：Meta 的新论文介绍了 **Thought Preference Optimization (TPO)**，这项技术显著提升了 **Llama 3.1 8B** 模型的性能，使其在 **AlpacaEval** 和 **ArenaHard** 基准测试中达到了与 **GPT-4** 相当的水平。论文详细介绍了该技术的实验和结果，其原理与 **o1 模型** 中使用的技术类似，展示了在通用指令遵循能力方面的显著提升。
  - 用户对 **AI 基准测试** 的飞速进展感到惊叹，**8B 模型** 现在就能达到 **GPT-4** 的性能，这与一年前的预期形成了鲜明对比。
  - 几位评论者询问了 **TPO 权重** 的可用性和实现细节，表现出对复制该技术的浓厚兴趣。
  - 社区注意到重大 AI 研究论文激增，包括微软的 **Differential Transformers** 和谷歌的 **Chain of Thought Reasoning**，同时还有关于将 **TPO** 应用于 **Llama-3.1-70B** 等更大模型的推测。

- **Optillm 中的熵解码（Entropy Decoding）+ GSM8k 的初步结果** ([Score: 30, Comments: 5](https://reddit.com//r/LocalLLaMA/comments/1g5gf27/entropy_decoding_in_optillm_early_results_on_gsm8k/)): **Optillm** 实现了基于 **entropy decoding** 的自适应采样，灵感源自 @\_xjdr 在 **entropix** 上的工作。在零样本（zero-shot）设置下，使用 **Qwen2.5-0.5B-Instruct** 模型在 **GSM8k** 基准测试上对该技术进行了评估，结果显示其优于基础模型，但未超过通过 **Chain of Thought (CoT) decoding** 取得的结果。目前已提供 [Google Colab notebook](https://colab.research.google.com/drive/1SpuUb8d9xAoTh32M-9wJsB50AOH54EaH?usp=sharing) 用于测试这两种方法。
  - 用户表示有兴趣在 **vLLM** 和 **llama.cpp** 等其他框架中实现 **entropy decoding**。一些用户在将 **optillm** 与 **llama-server** 和 **tabbyapi** 配合使用时遇到了困难，出现了 **404** 和 **401 错误**。
  - 开发者提供了排查资源，包括一个 [GitHub issue](https://github.com/codelion/optillm/issues/8#issuecomment-2356788401)、一个 [Hugging Face space](https://huggingface.co/spaces/codelion/optillm) 以及用于测试的原始 [Google Colab notebook](https://colab.research.google.com/drive/1SpuUb8d9xAoTh32M-9wJsB50AOH54EaH?usp=sharing)。
  - 有人指出 **optillm** 的 **Chain of Thought (CoT) decoding** 实现中存在一个潜在缺陷，即 **confidence score** 应仅根据答案片段（answer span）计算，而非整个序列。开发者对如何通用地识别答案部分提出了疑问。

## 其他 AI 子版块回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 研究与进展**

- **Nvidia Nemotron 70B 模型超越更大尺寸模型**：Nvidia 发布了其 [Nemotron 70B 模型](https://www.reddit.com/r/singularity/comments/1g4xd7e/nvidia_nemotron_70b_beats_llama_31_405b_gpt4o/)，据报道该模型在多个基准测试中击败了 Llama 3.1 405B、GPT-4o 和 Claude 3.5 Sonnet。他们在 Hugging Face 上发布了指令模型（instruct model）、奖励模型（reward model）和数据集。

- **EgoAllo 通过头戴式摄像头估算 3D 人体姿态**：研究人员开发了 [EgoAllo](https://www.reddit.com/r/singularity/comments/1g4wsx6/egoallo_can_estimate_3d_human_body_pose_height/)，这是一个可以使用头戴式设备的图像来估算 3D 人体姿态、身高和手部参数的系统。这在 VR/AR 领域可能有应用前景。

- **AI 视觉推理取得突破**：多伦多大学的研究人员改进了用于 [ARC 挑战赛](https://www.reddit.com/r/singularity/comments/1g4xsjn/a_breakthrough_in_visual_reasoning/) 的视觉 Transformer，通过监督学习在 400 个公开 ARC 任务中实现了超过一半任务的近 100% 解决率。然而，这种方法可能无法很好地泛化到完整的 ARC 基准测试。

**AI 行业与公司新闻**  

- **特斯拉 Optimus 机器人展示进展**：特斯拉发布了一段 [Optimus 的更新视频](https://www.reddit.com/r/singularity/comments/1g5khpb/update_on_optimus/)，展示了改进后的行走、物体操控和自主导航能力。然而，关于其中有多少是自主运行还是远程操作（teleoperated）仍存在争议。

- **OpenAI 声称遭到 Elon Musk 骚扰**：OpenAI [声称 Elon Musk 正在骚扰其公司](https://www.reddit.com/r/OpenAI/comments/1g525hy/openai_is_claiming_that_elon_musk_is_harassing/)，这与 OpenAI 从非营利转向营利状态的纠纷有关。

- **亚马逊投资核能技术**：亚马逊宣布计划 [投资超过 5 亿美元](https://www.reddit.com/r/singularity/comments/1g512da/amazon_goes_nuclear_to_invest_more_than_500/) 开发小型模块化核反应堆，可能用于为数据中心供电。

**AI 伦理与社会影响**

- **AI 生成的维基百科文章不断增加**：一项研究发现，[8 月份至少有 5% 的新维基百科文章是 AI 生成的](https://www.reddit.com/r/OpenAI/comments/1g5gzag/at_least_5_of_new_wikipedia_articles_in_august/)，尽管 AI 检测方法的准确性仍存争议。

- **Yann LeCun 评论 AI 炒作**：AI 先驱 Yann LeCun [分享了对当前 AI 炒作（AI hype）的看法](https://www.reddit.com/r/singularity/comments/1g5b2lq/yann_lecun_on_the_ai_hype/)，但评论中未提供具体细节。

**AI 政策与监管**

- **埃马纽埃尔·马克龙警告过度监管**：法国总统埃马纽埃尔·马克龙 [警告称，由于过度监管和投资不足，欧洲面临在 AI 领域落后的风险](https://www.reddit.com/r/singularity/comments/1g4x7fc/emmanuel_macron_we_are_overregulating_and/)，他表示：“我们监管过度且投资不足。因此，如果在接下来的 2 到 3 年里我们继续遵循传统的议程，我们将被市场淘汰。”


---

# AI Discord Recap

> 由 O1-mini 生成的摘要之摘要

主题 1. **LLM 性能与基准测试的进展**

- [**NVIDIA Nemotron 70B 统治基准测试**](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct)：**NVIDIA Nemotron 70B** 在多项评估中表现优于 **Llama 3.1 405B**、**GPT-4o** 和 **Claude 3.5 Sonnet**，在 **Arena Hard** 和 **AlpacaEval 2 LC** 中获得了最高分。
- [**Llama 3.1 vs. Mistral 7B：性能差距揭晓**](https://blog.eleuther.ai/mad_research_update_2/)：**MAD** 测试显示，**Mistral 7B v0.1** 在非算术任务上的表现优于 **Llama 3.1 8B**，突显了行为和损失指标上的差异。
- [**GLM-4-Plus 与 Yi-Lightning 在 Chatbot Arena 中崛起**](https://lmarena.ai)：来自智谱 AI 的 **GLM-4-Plus** 和 **Yi-Lightning** 已冲入前 10 名，展示了中国 **LLMs** 在数学和编程等领域的竞争性进步。

主题 2. **新 AI 工具与平台功能**

- [**Hugging Face 发布社区工具以增强交互**](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/569)：新的 **Hugging Face Community Tools** 允许用户在 **HuggingChat** 上创建自定义工具，结合视频和语音模态来丰富用户与模型的交互。
- [**OpenRouter 推出 NVIDIA 模型及更具竞争力的定价**](https://openrouter.ai/x-ai/grok-2)：**OpenRouter** 增加了 **SambaNova** 和 **Yi Lightning** 模型并提供极具竞争力的价格，促进了**自研芯片推理**供应商采用按需付费模式。
- [**NotebookLM 通过自定义音频和企业支持增强功能**](https://notebooklm.google/)：**NotebookLM** 现在允许用户在生成音频前提供自定义音频指令，并通过 Google Workspace 推出了企业版，改进了协作工具。

主题 3. **LLMs 的优化与训练技术**

- [**Muon 优化器在效率和性能上超越 AdamW**](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/)：得益于其新的分布式实现，**Muon 优化器**与 **AdamW** 相比，实现了更低的验证损失并减少了 Token 使用量，尤其是在大型模型上。
- [**LLM 重排序技术提升搜索准确率**](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/)：使用机器学习算法实施 **LLM 重排序技术**可增强与用户意图的一致性，精炼搜索结果以提高相关性。
- [**使用 CLIP 编码器进行 ControlNet 训练引发争议**](https://huggingface.co/qwen2.5)：在 **ControlNet** 训练中保留 **CLIP 编码器**引发了对潜在过拟合以及对生成准确描述影响的担忧。

主题 4. **API 性能与集成挑战**

- [**Perplexity API 面临响应迟缓问题**](https://perplexity.ai)：用户报告 **Perplexity API** 响应速度缓慢，基础查询需要 **1 到 2 分钟**，引发了基准测试讨论和对性能未达预期的关注。
- [**Torchtune 更新适配 PyTorch 2.5 发布**](https://github.com/pytorch/torchtune/issues/1861)：**Torchtune** 引入了对 **PyTorch 2.5** 的支持，具有 [FlexAttention](https://github.com/pytorch/pytorch/releases/tag/v2.5.0) 和**逐层编译**功能，鼓励用户升级以获得更好的性能。
- [**OpenInterpreter 与 Aider 的集成问题依然存在**](https://github.com/OpenInterpreter/open-interpreter/tree/main/scripts)：用户在跨平台使用时遇到了 **OpenInterpreter** 任务无法执行以及 **Aider** 安装问题，引发了持续的排障和社区支持工作。

主题 5. **社区参与：黑客松与协作倡议**

- [**Gen AI Agents 黑客松邀请创新者**](https://lu.ma/ke0rwi8n)：由 **CreatorsCorner** 与技术合作伙伴共同主办，**Gen AI Agents 黑客松**鼓励参与者在考虑伦理影响和增强人类潜力的同时，构建 **AI 驱动的多智能体系统**。
- [**Bitnet 发布官方 1-bit LLM 框架**](https://github.com/microsoft/BitNet)：**Bitnet** 在 **GitHub** 上发布了其官方的 1-bit LLMs 推理框架，实现了高效的模型执行并促进了研究协作。
- [**DSPy 的 Langtrace 集成推动协作项目**](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy)：**Langtrace** 与 **DSPy** 的集成促进了高级数据处理和多标签分类，社区成员也为 Prompt 优化和文档增强做出了贡献。

---

# 第 1 部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face 社区工具发布**：Hugging Face 社区工具允许用户在 HuggingChat 上创建自定义工具，涵盖视频和语音等多种模态，以增强用户交互。
  
  - 该功能为模型能力开辟了新途径，促进了用户协作与创新。
- **加速 LLM 训练的努力**：一名成员介绍了一个专门用于在 HuggingFace 和 S3 之间存储和流式传输数据以进行 LLM 训练的平台，旨在解决数据管理挑战。
  
  - 平台渴望获得反馈以进一步完善功能，欢迎申请 Demo 演示。
- **目标检测方法的见解**：讨论围绕利用 YOLO 等模型进行目标检测展开，并提到了边界框（bounding boxes）对准确性的重要性。
  
  - 建议包括结合 SAM 等模型进行语义分割，以实现像素级标注，从而提高检测细节。
- **NLP 微调数据集格式查询**：一名成员询问关于使用指令格式（instruct formatted）数据集微调基础模型的问题，并确认使用原始文本数据集可能会产生不准确的输出。
  
  - 确保数据集与特定领域知识的兼容性，凸显了仔细选择数据集的重要性。
- **关于使用 CLIP 编码器进行 ControlNet 训练的讨论**：成员们讨论了使用新的微调模型重新训练 ControlNet，引发了对特定数据集潜在过拟合风险的担忧。
  
  - 利用 CLIP 编码器而非文本编码器引发了关于生成 Caption 的影响以及训练谨慎性的辩论。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gandalf 挑战取得高成功率**：参与者在 **Gandalf 挑战**中取得了成功，采用了创新的 Prompt 策略来获得高排名。
  
  - 诸如询问带有隐藏标准的列表和玩“21个问题”游戏等方法，展示了挑战的迭代本质。
- **Ollama 简化 GGUF 模型执行**：Ollama 允许用户使用 `ollama run <model_url>` 运行来自 Hugging Face 的 **GGUF 模型**，简化了流程。
  
  - 凭借 **45K 个公开 GGUF checkpoints**，它通过可定制的量化类型和 System Prompts 选项增强了体验。
- **GitHub 上发布 SCP 生成器**：一个新的 [SCP 生成器](https://github.com/dottxt-ai/cursed/tree/main/scp) 能够利用 dottxt-ai 提供的提纲创建 SCP 故事。
  
  - 这个开源项目欢迎贡献，邀请开发者加入其开发。
- **关于 LLM 编程语言的辩论**：一名成员询问哪种编程语言最适合顶尖 LLM，对比了 **JavaScript** 与 **Python**。
  
  - 观点各异，一名成员断言 LLM 与 **Python 深度绑定**，同时主张进行更多的 **JavaScript 编码**。
- **LLM 越狱资源讨论**：关于 **LLM 越狱（jailbreaks）** 资源的讨论提到了查看 **plineys discord**。
  
  - 该社区内部的混乱引发了对替代资源的呼吁。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **MAD 性能揭示模型差异**：最近针对 **机制异常检测 (MAD)** 的测试发现，**Llama 3.1 8B** 在非算术任务上的表现不如 **Mistral 7B v0.1**，凸显了显著的性能差距。
  
  - **Llama** 表现出较少的古怪行为，但具有更强的地面真值偏差 (ground truth bias)，在各项任务中实现了更低的平均损失。
- **高级 LLM 重排序提升准确率**：根据这个 [实现方案](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/)，参与者讨论了使用机器学习算法进行 **LLM 重排序技术 (LLM re-ranking techniques)** 以优化搜索结果的有效性。
  
  - 这些方法的目标是更好地使输出符合用户意图，提供更相关的信息。
- **Muon 优化器优于 AdamW**：与 AdamW 相比，**Muon 优化器**展现出更好的性能，具有更低的验证损失和更少的 Token 使用量，特别是在 GPT-2 等大型模型上。
  
  - 其新的分布式实现展示了在训练中的显著效率提升，用户注意到在高达 1.5B 参数的模型上取得了成功。
- **寻找模型幻觉指标**：围绕识别评估和量化模型 **幻觉 (hallucinations)** 的 **可靠方法** 展开了讨论，成员们正在寻找相关的研究论文。
  
  - 建立用于评估模型输出保真度的稳健指标正引起越来越多的关注。
- **在测试期间保存模型输出**：成员们讨论了在测试阶段保存模型生成内容的策略，建议使用 `--log_samples` 参数。
  
  - 该功能有助于保留实验过程中生成的输出。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **NVIDIA Nemotron 70B 碾压竞争对手**：**NVIDIA Nemotron 70B** 在多项评估中超越了 **Llama 3.1 405B**、**GPT-4o** 和 **Claude 3.5 Sonnet**，据报告其在 Arena Hard 得分为 **85.0**，AlpacaEval 2 LC 为 **57.6**，MT Bench 为 **8.98**。
  
  - 您可以在[此处](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct)查看结果并进行尝试。
- **Grok 2 回归并伴随涨价**：**Grok 2** 目前的定价为 **输入 $5/m** 和 **输出 $10/m**，mini 版本仍不可用，这让讨论涨价影响的用户感到吃惊。
  
  - 更多关于其功能的信息可以在[此处](https://openrouter.ai/x-ai/grok-2)找到。
- **OpenRouter 模型与定价洞察**：讨论强调了通过 **OpenRouter** 提供的各种模型，包括 **SambaNova** 和 **Yi Lightning**，后者拥有极具竞争力的 **$0.14/m** 输入费率。
  
  - 随着按需付费模式的普及，人们猜测即将会有关于自研芯片推理提供商定价的深入见解。
- **语音交互模型缺乏一致性**：针对 **GPT-4o** 等模型的语音功能出现了担忧，特别是在处理多种语言时输出质量下降的问题。
  
  - 用户指出，虽然语音输入尚可，但输出变得“古怪”，尤其是在中文等语言中。
- **显微镜下的 O1 模型**：用户辩论了 **O1 模型** 的性能，特别是它在指令遵循和保持连贯输出方面的困境。
  
  - 由于存在过度冗长的回答问题，用户对其在各种任务中的实用性表示了担忧。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity API 响应时间延迟**：用户报告 **Perplexity API** 的响应时间非常缓慢，基础查询需要 **1 到 2 分钟**。
  
  - 讨论了基准测试的尝试，普遍情绪表明当前的性能水平未达到预期。
- **Llama 3.1 在基准测试中占据主导地位**：一名用户断言，基于对齐基准测试，来自 Nvidia 的 **Llama 3.1-Nemotron-70B** 超越了 **GPT-4** 和 **Claude 3.5** 等竞争对手。
  
  - 该模型因在众多评估中获得令人印象深刻的分数而名声大噪。
- **Oura Ring 4 走红**：[Oura Ring 4](https://www.perplexity.ai/page/oura-ring-4-review-5U7Rj9.hR3W0MRa_OmQgbQ) 因其先进的健康追踪功能和时尚设计而受到赞誉，特别是其睡眠监测的准确性。
  
  - 用户对其增强的健康洞察力印象深刻，这促使其在市场上的关注度不断提高。
- **Starlink 的千兆速度计划引发关注**：[Starlink Gigabit Speed Plan](https://www.perplexity.ai/page/starlink-gigabit-speed-plan-knyorEQ7SYG11t4a.dd2Ig) 承诺为农村用户提供前所未有的互联网速度。
  
  - 随着用户期待卫星互联网连接速度的预期提升，期待感正在增强。
- **LFM 40B API 可用性查询**：一名用户询问了来自 [labs.perplexity.com](https://labs.perplexity.com) 的 **LFM 40B** 模型潜在的 API 访问权限，但未收到后续回复。
  
  - 此外，还提出了针对新 **spaces feature** 提供 API 的可能性，并澄清主平台目前不存在 API。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O1-mini 在对比 Sonnet 3.5 时超出预期**：**O1-mini** 在处理复杂任务时表现出显著的能力，通过有效的反复迭代超越了 **Claude 3.5**，以更少的迭代次数更快地完成任务。
  
  - 尽管如此，在大多数场景下，用户仍然因为熟悉度和可靠性而青睐 **Sonnet 3.5**。
- **价格冲击：O1-preview 定价引发担忧**：**O1-preview** 每 **100 万 token** **60 美元** 的定价引发了用户的担忧，这使得它对于已经订阅 **ChatGPT Plus** 的用户来说吸引力降低。
  
  - 这进一步激发了对 **Sonnet 3.5** 等替代方案的兴趣，它仍然是备受青睐的性价比模型。
- **Aider 安装困扰凸显兼容性问题**：用户分享了 **Aider** 的故障排除技巧，特别关注在 **Windows 11** 上使用 **pipx** 进行安装。
  
  - **Chromebooks** 也出现了安装困难，强调了跨平台更广泛兼容性的需求。
- **Token 限制令用户感到沮丧**：许多用户报告在使用 **claude-3-5-sonnet** 和 **DeepSeek** 模型时达到了 token 限制，建议使用 `/clear` 来缓解聊天历史记录问题。
  
  - 最佳实践包括将代码拆分为较小的文件，以帮助更好地管理使用量。
- **DeepSeek 面临模型挑战**：关于 **DeepSeek** 模型挑战的担忧是一个反复出现的话题，引发了围绕变通方法和经验分享的讨论。
  
  - 成员们交流了改进与模型交互的建议，反映了一个积极寻求解决方案的社区。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **多节点集群引发以太网问题**：用户讨论了如何在网络中设置由 **4 个 V100** 组成的集群，同时强调了 Lambda 除非使用 **Infiniband**，否则缺乏多节点集群的选项。
  
  - 尽管有些人更倾向于在实验性设置中使用 **Ethernet**，但 **Pure DDP** 可能会消除对 Infiniband 的需求。
- **Gen AI Agents 黑客松公告**：**CreatorsCorner** 与多家科技公司合作举办的黑客松发布了公告，重点是创建 **AI 驱动的多 Agent 系统**。
  
  - 鼓励参与者在构建增强人类日常生活潜力的解决方案时，考虑伦理影响。
- **PyTorch 2.5 正式发布！**：[PyTorch 2.5](https://anaconda.org/pytorch/pytorch) 的发布已确认，目前 wheel 文件已在 conda 和 PyTorch 的 pip 索引中上线。
  
  - 针对发布的兴奋情绪，有人评论道：*“还以为应该是明天才发”*。
- **移除变量导致 Loss 增加**：在一次训练迭代中，移除未使用的变量后，Loss 从大约 **7** 增加到了 **10**，凸显了模型性能中意想不到的行为。
  
  - 通过 [Diffchecker](https://www.diffchecker.com/BDcWuLSY/) 分享了文件对比，以便进一步检查。
- **关于《赛博朋克 2077》基准测试的古怪检查**：一名成员询问是否可以使用该系统进行 [《赛博朋克 2077》基准测试](https://link.to.cyberpunk)，并澄清这是为了研究和性能测试。
  
  - 另一名成员回答说，如果将其重写为 **triton kernel**，它可能会起作用。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 配置获得升级**：用户确认 **ROCm** 已包含在 LM Studio **0.3.4** 版本中，可通过 Developer 选项卡访问，改进了系统配置。
  
  - 一位用户报告称，更新后性能提高到了 **32.82 tok/sec**，展示了实际使用中的增强。
- **Nvidia 模型在性能舞台上大放异彩**：成员们强调 **Nvidia 模型** 在笔记本电脑上的表现显著优于 **LLM 3.1** 等模型，其效率引发了热议。
  
  - 使用 **Nemotron 70b** 模型的测试进一步阐明了竞争优势，引发了对未来基准测试的期待。
- **Token 生成速率令人印象深刻**：用户报告 **70B Q8 模型** 的 Token 生成速度达到了令人印象深刻的 **5-7 tok/s**，足以媲美 **ChatGPT** 的性能水平。
  
  - 另一种配置达到了 **32.82 tok/sec**，展示了不同设置下的差异性和潜力。
- **Llama 3.1 在速度上大获全胜**：一名成员在 **7900XTX** GPU 上使用 **Llama 3.1**，在 **10k context length** 下达到了惊人的 **66 tokens/sec**，展示了硬件的协同效应。
  
  - 这强调了将强大的硬件与大模型匹配以获得最佳结果的重要性。
- **冷却系统引发噪音问题**：讨论强调了冷却系统常见的噪音困扰，将负载下的声音比作 **无人机起飞**。
  
  - 这种对硬件管理的见解强调了在平衡性能与噪音水平方面的挑战。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Glif 和 Wojnak 生成器表现出色**：成员们称赞 **Glif 和 Wojnak 生成器** 能够以极少的输入产生出色的结果，称其为 AI 工具领域的**黄金标准**。
  
  - 他们强调了这些工具生成 **链接 AI 工具的工作流** 以创建功能性应用程序的能力。
- **桌面端应用的语音功能受到质疑**：关于 [ChatGPT for Windows](https://openai.com/chatgpt/download/) 的语音功能出现了担忧，成员们不确定它是否与 Android 应用的功能相匹配。
  
  - 一些人担心最初只有 macOS 用户获得语音支持可能存在潜在的不公平。
- **O1 模型遭到批评**：成员们对 **O1 preview 模型** 表示不满，理由是与被认为速度明显更快的 **O1-mini** 相比，其 Prompt 响应时间较慢。
  
  - 共识指向了改进的需求，因为用户在交互中寻求更高的效率。
- **Wispr Flow 受到关注**：讨论重点介绍了 **Wispr Flow 应用程序**，它可以提高跨平台的写作速度和准确性，目前支持 macOS。
  
  - 成员们注意到，针对 **Linux, Mac, 和 Windows** 用户存在开源替代方案。
- **CustomGPT 源码引用失效**：关于 **CustomGPT** 无法引用文档来源的担忧增加，引发了对有效 Prompt 方法的质疑。
  
  - 用户一致认为，更清晰的 Prompt 对于确保响应中包含源码引用至关重要。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **推理提供商寻求明确性**：一名成员讨论了寻找支持使用前缀进行聊天助手补全（类似于 Anthropic 的功能）的推理提供商。
  
  - 提到了对模型可靠性的担忧，表明提供商需要更清晰的沟通。
- **NotebookLM 推出音频自定义功能**：NotebookLM 现在允许用户在生成音频前提供自定义音频指令，承诺提供更好的用户体验。
  
  - 随着超过 **80,000** 家机构的加入，通过 Google Workspace 推出的 Business 版本已上线，并去掉了“Experimental”标签。
- **MotherDuck 简化 SQL 与语言模型的交互**：MotherDuck 引入了 **prompt()** 函数，将小型语言模型集成到 SQL 查询中，用于数据生成和提取。
  
  - 这一创新旨在简化 LLM 交互，同时提供显著的成本和性能优势。
- **OpenAI 发布 Windows 桌面应用**：OpenAI 首次推出了 ChatGPT Windows 桌面应用的早期版本，专为 Plus 和 Enterprise 用户设计，提供更快的访问速度。
  
  - 用户可以通过 **Alt + Space** 快捷键便捷地访问该应用，这与 Claude 移动应用在项目管理方面的更新相呼应。
- **社区在数据标注方面蓬勃发展**：成员们强调了在 **Pixmo** 数据标注工作中的积极参与，引发了创意迷因（memes）和 Reddit 讨论。
  
  - 他们鼓励通过私人 Reddit 社区参与，以获取有关数据标注的持续更新和讨论。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Yi-Lightning 占据第 6 位**：来自 [Chatbot Arena](https://lmarena.ai) 的重大消息：**Yi-Lightning** 已获得超过 **13K 社区投票**，目前在总榜排名 **#6**，展示了其在数学和编程等领域的实力。
  
  - 这使其与 **Grok-2** 等强劲对手并列，引发了对其未来性能指标的期待。
- **GLM-4-Plus 冲入前列**：来自智谱 AI 的 **GLM-4-Plus** 现已进入聊天机器人排行榜前 **10** 名，反映了中国 LLM 在竞争格局中的迅速崛起。
  
  - 这表明市场正在成熟，各种模型之间的竞争力不断增强。
- **推理提供商功能咨询**：成员们询问了哪些推理提供商支持开放权重模型的聊天助手补全，特别是参考了 [Anthropic 的预填充功能](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response)。
  
  - *“我不确定我是否可以信任底层发生的事情”* 凸显了对这些提供商可靠性和透明度的担忧。
- **特殊 Token 的探索**：讨论了聊天机器人结构中特殊 Token 的使用，强调了与用户和助手交互相关的独特格式。
  
  - 成员们回顾了过去使用这些 Token 的经验，建议参考文档以获得清晰的理解。
- **重视研究经验**：一位成员分享说，在攻读硕士学位之前，从本科研究转型到非 ML 工作为他们在 **AI labs** 中提供了相当大的优势。
  
  - 他们指出，由于实验室运行速度很快，研究经验与职场熟悉度之间的平衡至关重要。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **使用 Azure AI 构建多模态 RAG 系统**：分享了使用 [Azure AI Search](https://t.co/RO5nQ79sqD)、Azure OpenAI 和 ArizePhoenix 结合 LlamaIndex 创建**多模态 RAG 系统**的分步指南。
  
  - 该指南强调通过上下文检索来提高准确性，并包含基准测试信息供参考。
- **LlamaIndex 与 Elastic 的结合 —— 明天演示**：观看关于如何将 **LlamaIndex** 与 Elastic 结合使用的演示，包含来自社区成员的见解，计划于明天进行。
  
  - 演示详情可以在[此处](https://t.co/tQszqtRN1Z)找到。
- **与 Meta 合作在班加罗尔举办 AI 黑客松**：一场 **AI Hackathon** 将于 10 月 19 日至 20 日在班加罗尔举行，由 Reskilll 和 Meta 合作举办，并拥有行业专家的指导。
  
  - 参与者可以在[此处](https://t.co/aFf31yHJba)注册并查找更多信息。
- **简化多租户 RAG 应用**：社区成员讨论了使用 LlamaIndex 和 Nile 创建**多租户 RAG 应用**，目标是为大量用户提供数据安全保障。
  
  - 可以在[此处](https://t.co/zRfzR5A4Us)探索说明这一点的全栈演示应用。
- **适用于 LlamaIndex 的 MongoDB 混合搜索**：利用 **MongoDB** 新的混合搜索支持，允许 LlamaIndex 结合向量搜索和关键字搜索以提升性能。
  
  - 在[此处](https://t.co/XxNNwoaW9U)查看此集成的详细信息。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **参加 Modular 社区 Q&A！**：发布了关于即将举行的 **Modular 社区 Q&A** 的提醒，敦促成员通过提供的 [表单](https://forms.gle/MgixGyhRKcA33BS6A) 提交问题。团队鼓励参与者在会议开始前提交咨询。
  
  - *请分享任何你希望团队在会议期间解决的问题。*
- **Mojo 致力于 MAX 适配**：成员们讨论了 **MAX** 的 **Mojo** 版本的潜在计划，指出考虑到 **Mojo** 的新颖性，从 **Python** 进行适配需要相当长的时间。
  
  - 对话强调了将现有功能迁移到新框架的复杂性和挑战。
- **LLM 正在革新翻译实践**：社区讨论集中在转向使用 **LLM** 进行翻译而非手动流程，强调了在中文社区中获得的效率提升。
  
  - 为了确保准确性，利用 prompt 来澄清翻译，特别是关于将 'parameter' 翻译为 '编译期参数' 等术语。
- **Driver Demo 获得好评**：最近的 driver 演示展示了模型实现的便捷性，尽管它在 **nightly builds** 中仍处于部分发布状态。
  
  - 一位成员表达了他们的赞赏，提到他们多次回顾了演示以充分掌握内容。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 需要 prompt 方面的帮助**：一位成员寻求关于 prompt 的帮助，以创建一个 **cube** 的阴影效果，且不显示其上方的光源，强调了光影在场景中的关键作用。
  
  - 这引发了关于 prompt 有效性不同经验的讨论，突显了社区对更 **具体建议** 的需求。
- **Fooocus 模型与兼容性**：在询问模型兼容性时，一位成员确认 **Fooocus** 主要使用 **SDXL**，但也支持 **pony models**。
  
  - 这一讨论强调了社区致力于确保兼容性以提升用户体验。
- **换脸功能解决方案**：一位成员询问如何在 **Automatic1111** 中复制 **Fooocus** 的 **faceswap** 功能，收到了如 **Reactor extension** 或 **IP-Adapter face** 等建议。
  
  - 这展示了用户之间为增强跨平台工具功能而进行的协作努力。
- **关于图像质量的担忧**：一位成员报告称，尽管使用了 **30 steps** 和多个 **LORA** 模型，生成的图像仍缺乏细节，并寻求解决方案的建议。
  
  - 这引发了关于影响 **Stable Diffusion** 过程中图像质量的各种因素的更广泛讨论。
- **创新项目 AI 黑客松**：**Gen AI Agents** 黑客松的公告邀请团队开发通过协作增强人类潜能的 AI 解决方案。
  
  - 鼓励参与者在创建优化日常任务的、安全可靠的 AI 系统时考虑伦理影响，链接至 [Vertical Specific AI Agents Hackathon](https://lu.ma/ke0rwi8n)。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **PyTorch 2.5.0 正式发布！**：备受期待的 **PyTorch 2.5.0** 已正式发布，其中包括 [FlexAttention](https://github.com/pytorch/pytorch/releases/tag/v2.5.0) 和 **per-layer compile** 等新功能。
  
  - 鼓励用户升级其本地 **torch** 安装，以利用这些最新功能。
- **Torchtune 贡献追踪器上线**：为那些希望为 **Torchtune** 做出贡献的人，已经建立了一个用于清理仓库以全面支持 **PyTorch 2.5.0** 的追踪器，详见 [此处](https://github.com/pytorch/torchtune/issues/1861)。
  
  - 该计划旨在确保库与 PyTorch 的最新更新和改进保持一致。
- **Torchtune 中的 Qwen 2.5 模型集成**：[Qwen 团队发布了 Qwen 2.5](https://github.com/pytorch/torchtune/issues/1624)，其中包括多个被请求集成到 Torchtune 的模型，但更新仍在进行中。
  
  - 成员们正在协作添加该模型，并欢迎对集成过程感兴趣的其他人贡献力量。
- **对 PhD Internship 愿景的热情**：一位用户在 [arXiv](https://arxiv.org/pdf/2410.10630) 上分享了一篇有趣的论文，激发了成员们的兴趣和热情。
  
  - 另一位成员表达了希望通过 **PhD internship** 来参与论文中所讨论项目的愿望。
- **PPO 工作的持续进展**：一位成员表示，在开始新任务之前，他们需要完成 **PPO** 方面的工作。
  
  - *'I gotta land a few RFCs first and finish up my PPO work'* 反映了团队目前的优先事项。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 任务完成困扰**：用户报告了 **OpenInterpreter** 的持续问题，即任务声称已完成但未执行任何操作。
  
  - 建议在单独的频道中详细说明版本和模型，以协助排查故障。
- **关闭应用时出现内核恐慌**：一位社区成员在关闭 OpenInterpreter 应用时遇到了 **kernel panic**，并被建议在专门的支持频道寻求帮助。
  
  - 这一问题强调了在应用程序使用过程中可靠退出的必要性。
- **成本效益高的免费 LLM 选项**：针对由于 [API 成本上升](https://link.url) 而与 Chat GPT 集成的免费 LLM 展开了讨论，并提出了可行的替代方案建议。
  
  - 其中一个建议包括为无法访问本地模型的人员通过 `interpreter --model i` 使用 `i model`。
- **AI 遇见 Vim：新教程探索**：Mikebirdtech 分享了 Jake Koenig 关于在 **Vim** 中集成 AI 的见解，并在 [此处](https://www.youtube.com/watch?v=Ho9yf7ks5sE) 的教程视频中进行了重点介绍。
  
  - 这为希望无缝增强编码工作流的开发者提供了一条新途径。
- **通过脚本提升 OpenInterpreter 的效用**：一位成员介绍了来自 OpenInterpreter 的 `wtf` 脚本，并在 [Tool Use](https://www.youtube.com/watch?v=Vz3cjbf4zeo) 中展示了其功能。
  
  - 该演示强调了此类脚本如何扩展用户能力以及与平台的交互。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **创新的多标签分类方法**：一位成员分享了一种针对科学文档的**多标签分类 (multi-label classification)** 新方法，该方法基于之前在极端多标签分类中的 [In-context learning 研究工作](https://link.to.research)。
  
  - 他们描述了创建一个**异构图 (Heterogeneous graph)**，其中红色节点代表文档，蓝色节点代表标签，并对其在大规模语料库中有效搜索的潜力表示兴奋。
- **Langtrace 在 DSPy 集成中表现出色**：成员们讨论了 **Langtrace** 与 **DSPy** 极具前景的集成，重点介绍了从 DSPy 流水线捕获追踪 (traces) 的 [设置说明](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy)。
  
  - 设置过程包括安装 DSPy、初始化 Langtrace 的 SDK，以及创建一个类型为 **DSPy** 的项目。
- **ColbertV2 训练需要三元组与查询**：**ColbertV2** 的训练示例需要三元组 (triples)、集合 (collections) 和查询 (queries)，正如 [GitHub 仓库](https://github.com/stanford-futuredata/ColBERT?tab=readme-ov-file#advanced-training-colbertv2-style) 中所记录的。这表明了一种复杂的数据处理机制，需要进一步明确。
  
  - 成员们对数据集如何与示例中看到的**查询 (queries)** 和**集合 (collections)** 的索引版本相关联表示困惑。
- **DSPy 提示词优化未反映在 JSON 中**：一位成员报告称，在使用 **MIPROV2** 优化一个简单的分类器后，JSON 配置保留了原始提示词 (prompt) 而非优化后的版本，引发了关于性能损失的疑问。
  
  - 随后展开了关于保存或加载配置时潜在 Bug 的讨论，并建议调查 JSON 文件的内容。
- **对 DSPy 文档的正面反馈**：一位用户对新的 DSPy 入门指南表示赞赏，强调其平易近人的拆解和完整的 **RAG** 实现对新手特别有帮助。
  
  - 建议包括增加交互式笔记本和末尾的“动手尝试”部分，以便进行实践学习。

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **MSE 和 MAE 增强**：一个在 `tensors.py` 中实现 **MSE** 及其测试的 Pull Request 已在 [此处](https://github.com/tinygrad/tinygrad/pull/7107) 分享。贡献者认为 **MSE** 和 **MAE** 都可以简洁地总结在库中。
  
  - *这种简化可以精简张量操作*并提高用户的清晰度。
- **通过 If_Then 门改进 LLVM 加载**：当前的 **LLVM** 加载需要调整以使用 **if_then** 处理门 (gates)，因为现有的技术被视为一种权宜之计 (hack)。成员们认识到为该实现创建更结构化方法的紧迫性。
  
  - *更好的方法可以显著增强门管理的清晰度和功能性*。
- **关于多设备 CLOUD=1 功能的查询**：一位成员询问 **CLOUD=1** 在多设备设置中如何运行，希望与早期的配置保持一致。这反映了对理解多设备操作集成的兴趣。
  
  - *澄清这一点将帮助用户在分布式环境中优化其设置*。
- **对 EMA 参数衰减的好奇**：讨论凸显了对 `update_ema_parameters` 中**衰减 (decay)** 过程的好奇，评估其在深度学习实践中的普遍性。成员们渴望更彻底地探索优化技术。
  
  - *这种好奇心说明了加深对有效训练方法论理解的愿望*。
- **推荐的 Tinygrad 学习资源**：一位成员建议从 Beautiful MNIST 示例开始，并修改 [OpenAI Cookbook 示例](https://cookbook.openai.com/examples/rag_with_graph_db) 以深入了解 Tinygrad 功能。此外，[tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes) 被引用为极佳的资源。
  
  - *这些资源为解释 Tinygrad 的各个层面提供了实践基础*。

 

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Axolotl 为随机性打乱数据集**：在训练之前，**Axolotl** 会打乱数据集以确保每个 epoch 的随机性，这验证了训练协议中的最佳实践。讨论参考了 [Hugging Face 上的这篇博客文章](https://huggingface.co/blog/gradient_accumulation) 以了解更多细节。
  
  - 一位成员在查阅资料后确认了这一行为，并指出其在缓解过拟合（overfitting）方面的重要性。
- **梯度累积（Gradient accumulation）差异问题被提出**：一个共同问题表明，**gradient accumulation** 在全批次训练和切换设置之间的 loss 可能不匹配，导致训练过程中的困惑。预计 **Hugging Face** 很快会发布修复补丁。
  
  - 成员们讨论了调试这些问题的担忧和个人经验，其中一人对推迟开始训练感到庆幸。
- **Bitnet 提供官方 1-bit LLM 框架**：1-bit LLMs 的官方推理框架 **Bitnet** 已经发布，可以在 [GitHub](https://github.com/microsoft/BitNet) 上访问。该发布重点介绍了简要概述并包含文档。
  
  - 成员们对 **1-bit LLMs** 的可用性表示赞赏，并讨论了在当前项目中的潜在应用。
- **A100 计算利用率详情**：**Invisietch** 分享了他们使用了 **1x A100** 持续 **3 天**，并提供了硬件设置的具体细节。这一见解为同行提供了计算效率的基准。
  
  - 对话强调了特定硬件选择对计算任务和项目时间线的实际影响。
- **DeepSpeed 使用困难引发关注**：Invisietch 还指出了 **DeepSpeed** 的问题，提到：“因为我无法让 DeepSpeed 工作”，表明存在设置问题。这引发了关于兼容性和实现障碍的讨论。
  
  - 成员们对如何有效地在工作流中集成 **DeepSpeed** 表示好奇，并对常见做法提出了疑问。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 工具生成响应面临挑战**：一位用户对 **Cohere** 工具在使用 **langgraph** 生成响应时的文档表示沮丧，并建议如果 `chat_stream` 失败，可以使用 *for loop* 作为备选方案。
  
  - 他们强调了更清晰的文档对于提升用户体验和响应质量的重要性。
- **Command R+ 面临性能问题**：一位成员报告称，**Command R+ version 0.8** 在一个月后的表现优于 **version 0.4**，引发了关于性能下降原因的讨论。
  
  - 成员们想知道是否计划进行任何即将到来的更新以改进其功能。
- **对 LLM 的逆强化学习（Inverse RL）感到好奇**：一位用户分享了一篇关于 **LLMs** 的 **Inverse Reinforcement Learning** 的论文，引发了社区的兴趣并征求意见。
  
  - 讨论围绕这种方法在增强 AI 能力方面的潜力展开。
- **号召参与多语言隐身（stealth）项目**：一位社区成员号召开发者加入一个需要语言专业知识的 **stealth** 项目，并附上了加入 **Aya** 服务器的链接。
  
  - 顶级贡献者将获得 **exclusive swag**（专属周边），突显了该项目的协作性质。
- **Langgraph 集成文档更新**：提到了与 **Cohere** 的 **langgraph** 集成相关的新文档，旨在帮助用户更高效地实现工具。
  
  - 暗示即将推出的示例将进一步帮助改进 **chat_stream** 功能。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **测验访问困扰**：一名成员在访问[课程网站](https://llmagents-learning.org/f24)教学大纲部分的 **Week 5 quiz** 时遇到问题。另一名成员确认了其可用性并协助导航到正确章节。
  
  - 后续强调所有参与者应确保查看的是正确的测验网站。
- **新成员加入并寻求指导**：一位新人询问在填写课程表单后是否会收到后续邮件，并就如何获取课程材料寻求澄清。现有参与者安抚他们继续参与课程，不必为 hackathons 感到压力。
  
  - 这反映了参与者之间相互支持的氛围，鼓励减少对补充材料的焦虑。
- **确认正确的课程网站**：成员们确认 **llmagents-learning.org** 是面向 MOOC 学生的正确站点，而 Berkeley 站点是为校内学生设计的。他们建议不要使用 Berkeley 站点进行课程活动，以免引起混淆。
  
  - 这一区分旨在简化在线学习者的访问流程。
- **发布前的文章审核**：有人请求在发布到社交媒体之前进行文章审核，以符合课程预期。虽然对审核过程复杂性的担忧有所浮现，但一些人强调了遵守课程网站概述指南的重要性。
  
  - 社区情绪倾向于在保持流程简便的同时维护质量。
- **报告每周课程进度**：一位参与者庆祝完成了 **Week 1** 并表示打算遵循课程结构。这得到了小组的赞赏，培养了继续进步的动力。
  
  - 这种鼓励性的环境有助于提高全体课程参与者的参与度。

 

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **寻求顶尖 AI Engineering 博客**：一名成员询问了关于专注于 **Retrieval systems** 和 **Multi-agent architectures** 的优质 **AI Engineering 博客**。
  
  - *未建议具体博客*。
- **转向 LangGraph 是明智之举**：讨论强调了从 **LangChain 迁移到 LangGraph** 的**优点**，特别是在抽象和易用性方面。
  
  - 一名成员询问了 **LangGraph** 与 **LangChain** 相比提供的**独特功能**。
- **用户的 LangChain 挫败感**：一位用户分享了在使用两年后对 **LangChain** 批评的**沮丧**，幽默地回顾了他们深夜学习的挣扎。
  
  - *关于克服这些问题的进一步见解尚未提供。*
- **请求 Agent 图可视化**：有人请求协助如何在项目中实现 **Agent 图可视化**，表明了对实际可视化技术的需求。
  
  - *遗憾的是，回复中没有分享任何解决方案。*
- **探索 LangGraph 的工具集**：一名成员发起了关于 **LangGraph** 中可用工具的讨论，寻求对其功能的更深入了解。
  
  - *未提供关于其能力的详细回复。*

 

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Inverse RL 进展引发关注**：一篇讨论 **Inverse RL**（逆强化学习）在 **LLMs** 中应用的论文引发了好奇，促使了[反馈](https://arxiv.org/pdf/2410.12491)讨论。
  
  - 参与者旨在评估这种方法是否能显著增强语言模型的能力。
- **NotebookLM 推出酷炫功能**：**Google** 宣布了 **NotebookLM** 的新功能，包括音频概览和协作工具，详见[此公告](http://goo.gle/3UcO8Na)。
  
  - 正如其 [tweet](https://x.com/Google/status/1846954813193359397?t=8gWKjTOUhZAYbjFMHluqGw&s=19) 中所强调的，这些工具旨在简化多任务处理，同时访问音频内容以获得更好的用户体验。
- **图强化学习引起热议**：一名成员分享了一份[关于图强化学习的综述](https://arxiv.org/abs/2404.06492)，展示了其在跨学科决策中的潜力，引起了热烈反响。
  
  - **图结构**与**强化学习**之间的联系可能会在化学和计算机科学等领域产生新颖的策略。
- **Gen AI Hackathon 启动**：诚邀参与者参加专注于构建用于日常任务的 **Gen AI 驱动的多 Agent 系统**的 hackathon，[详情点击此处](https://lu.ma/ke0rwi8n)。
  
  - 该挑战赛强调安全和伦理考虑，同时促进开发者之间的协作解决方案。

 

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **修复 Twitter/X 嵌入以增强功能**：成员们强调了**修复损坏的 Twitter/X 嵌入**的必要性，旨在推动跨 Discord 和 Telegram 等平台支持多图、视频、投票和翻译等功能。一位成员链接了 [FixTweet/FxTwitter 倡议](https://x.com/i/spaces/1ypKdpLNZXnKW)，鼓励大家为改进嵌入技术做出贡献。
  
  - 该倡议旨在简化集成流程，以实现更丰富的用户参与和跨平台内容共享。
- **交互式推文功能可提升参与度**：一场热烈的讨论集中在**更具交互性的推文功能**如何显著增强用户参与度，特别是在嵌入内容方面。
  
  - 成员们建议，**增强的多媒体支持**可能会促进更多的参与和内容分享。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **生成式 AI 漏洞赏金门户上线**：用于 **生成式 AI 漏洞赏金** 的 [门户](https://discord.com/channels/1089876418936180786/1245784344539435128/1295876886584492033) 已正式发布，通过用户友好的设计和自动分拣功能简化了漏洞提交流程，从而实现更快的审核。
  
  - 该倡议旨在通过简化研究人员报告漏洞的方式来提高安全性，使关键问题能够得到更快速的处理。
- **用户仪表板增强追踪功能**：新的**个性化用户仪表板**提供了一个集中视图，用于监控提交状态、更新和研究人员的进度。
  
  - 这一增强功能旨在提升用户体验，并促进对漏洞提交的更好管理。
- **实时通知让用户保持更新**：**实时通知**现在将针对提交漏洞的每一项操作发送即时电子邮件提醒，确保透明度。
  
  - 用户可以毫无延迟地了解其提交状态，促进有效沟通。
- **基于角色的权限提高安全性**：平台引入了**基于角色的权限 (Role-Based Permissions)**，以确保结构化的访问控制，增强数据管理和协作。
  
  - 这一安全措施将敏感信息的访问权限仅限制给授权用户。
- **令人兴奋的培训机会即将到来**：从 11 月开始，将推出 **Prompt Engineering 课程与 CTF 挑战**，重点关注 AI 漏洞和技能开发。
  
  - 该倡议将包括**每周博客与教程**，旨在增强参与者的 AI 安全知识。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有活动，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1296186543710670888) (890 条消息 🔥🔥🔥):

> - `Hugging Face 更新`
> - `PyTorch 模型`
> - `HuggingChat 社区工具`
> - `图像目标检测`
> - `LLM 强化学习`

- **Hugging Face 发布社区工具**：Hugging Face 社区工具功能允许用户在 HuggingChat 上创建自定义工具，利用视频和语音等各种模态。
  
  - 这一功能使模型能够利用新工具，增强与用户的交互。
- **关于使用 PyTorch 和 Transformers 的讨论**：用户讨论了 PyTorch 的优势及其与 Hugging Face 模型的集成，强调了版本控制和易用性等特性。
  
  - 贡献者分享了使用自己库的经验，以及将模型存储在 Hugging Face 上的好处。
- **探索目标检测技术**：成员们交流了使用 YOLO 等模型进行目标检测的见解，强调了边界框 (bounding boxes) 的重要性。
  
  - 建议包括使用语义分割技术，通过 SAM 等模型获取像素级标签。
- **LLM 的强化学习**：用户询问了应用强化学习 (RL) 技术优化 LLM 以获得更好响应生成的可能性。

- 提供关于现有资源的指导，例如使用 RLHF 训练 LLM 的指南。
- **动漫和漫画兴趣**：关于动漫的讨论引发了对热门系列的想法分享，并提到了个人经历和推荐。
  
  - 并非所有用户都对每个作品产生共鸣，突显了社区内多样化的口味。

**提到的链接**：

- [Emu3 - a Hugging Face Space by BAAI](https://huggingface.co/spaces/BAAI/Emu3)：未找到描述
- [Open port for space to connect to PostgreSQL](https://discuss.huggingface.co/t/open-port-for-space-to-connect-to-postgresql/29938/10)：你好 @anon86412018 和 @deepkyu，我们更改了规则，除了 80、443 之外，还将启用 5432、27017。抱歉 @anon86412018，我认为它还没有上线生产环境。我会在这里通知你。谢谢。
- [SmolLM - blazingly fast and remarkably powerful](https://huggingface.co/blog/smollm)：未找到描述
- [Ollama](https://ollama.com/search?c=embedding)：快速上手并运行大语言模型。
- [GPU Benchmarks for Deep Learning | Lambda](https://lambdalabs.com/gpu-benchmarks)：Lambda 的深度学习 GPU 基准测试在十几种不同型号、多种配置的 GPU 上运行。GPU 性能通过运行计算机视觉 (CV)、自然语言处理等模型进行测量...
- [Arm Pump GIF - Arm Pump - Discover & Share GIFs](https://tenor.com/view/arm-pump-gif-22012416)：点击查看 GIF
- [Hulk Hogan Flex GIF - Hulk Hogan Flex Flexes - Discover & Share GIFs](https://tenor.com/view/hulk-hogan-flex-flexes-flexing-wwe-gif-13189000)：点击查看 GIF
- [Right To Jail Jail GIF - Right To Jail Jail Parks And Rec - Discover & Share GIFs](https://tenor.com/view/right-to-jail-jail-parks-and-rec-right-away-fred-armisen-gif-16902115)：点击查看 GIF
- [StackLLaMA: A hands-on guide to train LLaMA with RLHF](https://huggingface.co/blog/stackllama)：未找到描述
- [sail-rvc/Rick_Astley__RVC_v2__140_Epochs at main](https://huggingface.co/sail-rvc/Rick_Astley__RVC_v2__140_Epochs/tree/main)：未找到描述
- [Mundo Feliz Onetreehsll GIF - Mundo feliz Onetreehsll World if - Discover & Share GIFs](https://tenor.com/view/mundo-feliz-onetreehsll-world-if-gif-14071235670792471304)：点击查看 GIF
- [Tweet from PyTorch (@PyTorch)](https://x.com/PyTorch/status/1846951947280015407)：PyTorch 2.5 发布了 🔥 我们很高兴地宣布 #PyTorch 2.5 的发布，其特点包括用于 SDPA 的新 CuDNN 后端、torch.compile 的区域编译以及 TorchInductor CPP 后端性能提升...
- [Qwen 2.5](https://qwen2.org/qwen2-5/)：在这篇博客中，我们探讨了由阿里云开发团队开发的新 Qwen2.5 系列语言模型的细节。
- [Can I Have Mod Discord GIF - Can I Have Mod Discord Discord Mod - Discover & Share GIFs](https://tenor.com/view/can-i-have-mod-discord-discord-mod-gif-23039596)：点击查看 GIF
- [CogVLM2: Bringing Deeper Visual and Language Understanding to AI](https://medium.com/@ryanfoster_37838/cogvlm2-bringing-deeper-visual-and-language-understanding-to-ai-2d04d95797a9)：AI 在理解文本方面已经取得了长足的进步，但当涉及到将视觉数据（如图像和视频）与语言融合时，我们已经……
- [rombodawg/Rombos-LLM-V2.6-Nemotron-70b · Hugging Face](https://huggingface.co/rombodawg/Rombos-LLM-V2.6-Nemotron-70b)：未找到描述
- [Open port for space to connect to PostgreSQL](https://discuss.huggingface.co/t/open-port-for-space-to-connect-to-postgresql/29938)：你好 @chris-rannou，能否为这个 Space 开放 5432 端口：Defi Ai 2022 - 由 vnghia 创建的 Hugging Face Space，因为我需要连接到 PostgreSQL 数据库？非常感谢！
- [Hacking Fake GIF - Hacking Fake Movies - Discover & Share GIFs](https://tenor.com/view/hacking-fake-movies-coding-typing-gif-18697374)：点击查看 GIF
- [Tim And Eric Awesome Show GIF - Tim And Eric Awesome Show Kissess - Discover & Share GIFs](https://tenor.com/view/tim-and-eric-awesome-show-kissess-love-kiss-gif-18128184)：点击查看 GIF
- [qwen2.5](https://ollama.com/library/qwen2.5)：Qwen2.5 模型在阿里巴巴最新的大规模数据集上进行了预训练，包含高达 18 万亿个 token。该模型支持高达 128K 的 token，并具有多语言支持。
- [HP Z2 Tower G9 Workstation](https://www.hp.com/id-id/shop/hp-z2-tower-g9-workstation-a41yppt.html?facetref=22bc09f26b9afe34)：专业动力，服务当下与未来
- [GitHub - not-lain/pxia: AI library for pxia](https://github.com/not-lain/pxia)：pxia 的 AI 库。通过在 GitHub 上创建账户来为 not-lain/pxia 的开发做出贡献。
- [GitHub - eloialonso/diamond: DIAMOND (DIffusion As a Model Of eNvironment Dreams) is a reinforcement learning agent trained in a diffusion world model. NeurIPS 2024 Spotlight.](https://github.com/eloialonso/diamond)：DIAMOND (DIffusion As a Model Of eNvironment Dreams) 是一个在扩散世界模型中训练的强化学习 Agent。NeurIPS 2024 Spotlight。

- [huggingchat/chat-ui · [FEATURE] Community Tools](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/569): 未找到描述
- [no title found](https://hubs.ly/Q02QMQ--0): 未找到描述
- [Jiwei Liu | Grandmaster](https://www.kaggle.com/jiweiliu): 请查看 https://github.com/rapidsai/cuml 如果你喜欢请点个 star 😄
- [no title found](https://hubs.li/Q02rCNSs0): 未找到描述
- [not-lain (Lain)](https://huggingface.co/not-lain): 未找到描述
- [GitHub - AntoniovanDijck/diamond-macos: DIAMOND (DIffusion As a Model Of eNvironment Dreams) is a reinforcement learning agent trained in a diffusion world model. NeurIPS 2024 Spotlight.](https://github.com/AntoniovanDijck/diamond-macos.git): DIAMOND (DIffusion As a Model Of eNvironment Dreams) 是一个在扩散世界模型中训练的强化学习 Agent。NeurIPS 2024 Spotlight。 - AntoniovanDijck/diamond-macos
- [GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers.git): 🤗 Transformers: 为 Pytorch、TensorFlow 和 JAX 提供的 SOTA 机器学习库。 - huggingface/transformers

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1296195827739398165) (5 条消息):

> - `GitHub 上的 PaliGemma`
> - `Grasshopper URLs 扩展`
> - `Manim 社区框架`
> - `用于金融的 Perplexity AI`

- **在 GitHub 上探索 PaliGemma**：查看 [PaliGemma 仓库](https://github.com/ThinamXx/PaliGemma)，你可以在那里了解其开发进度并为项目做出贡献。
  
  - 有用户提到他们刚刚点了 star，展示了对其功能的兴趣。
- **使用 Grasshopper URLs 管理你的标签页**：[Grasshopper URLs](https://addons.mozilla.org/en-US/firefox/addon/grasshopper-urls/) 扩展提供垂直标签页功能，并可作为历史记录和书签管理器。
  
  - 它需要标签页、历史记录、书签和会话的权限，同时确保不会意外丢失书签。
- **用于动画的 Manim 框架**：探索 [Manim 社区 GitHub](https://github.com/ManimCommunity/manim)，这是一个由社区维护、用于创建数学动画的 Python 框架。
  
  - 该项目为那些有兴趣制作复杂数学可视化的人提供了工具。
- **Perplexity AI 增强金融研究**：根据其 [状态更新](https://x.com/perplexity_ai/status/1846287953599123757?t=RDl45Q5xGvfjF8sIZUm4zw&s=19)，Perplexity AI 宣布了新的金融功能，包括实时股票报价和详细的公司财务分析。
  
  - 用户在研究市场趋势和历史收益时可以享受愉悦的 UI。

**提到的链接**：

- [来自 Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1846287953599123757?t=RDl45Q5xGvfjF8sIZUm4zw&s=19)：用于金融的 Perplexity：实时股票报价。历史收益报告。行业同行对比。公司财务详细分析。一切都伴随着愉悦的 UI。祝你在研究市场时玩得开心...
- [Grasshopper – 为 🦊 Firefox (en-US) 获取此扩展](https://addons.mozilla.org/en-US/firefox/addon/grasshopper-urls/)：下载适用于 Firefox 的 Grasshopper。强大的标签页管理器。
- [GitHub - ThinamXx/PaliGemma: Reading PaliGemma paper ...](https://github.com/ThinamXx/PaliGemma)：阅读 PaliGemma 论文 ... 通过在 GitHub 上创建账号为 ThinamXx/PaliGemma 的开发做出贡献。
- [GitHub - ManimCommunity/manim: A community-maintained Python framework for creating mathematical animations.](https://github.com/ManimCommunity/manim/)：一个由社区维护、用于创建数学动画的 Python 框架。- GitHub - ManimCommunity/manim

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1296197349265375313) (7 条消息):

> - `LLM Training Acceleration` (LLM 训练加速)
> - `In-Depth Question Answering Evaluation App` (深度问答评估应用)
> - `Book Crossover Storytelling App` (书籍跨界叙事应用)
> - `Collaborative Story Builder` (协作式故事构建器)
> - `WorldMedQA-V Dataset` (WorldMedQA-V 数据集)

- **使用自定义平台加速 LLM 训练**：一名成员介绍了一个旨在为 **LLM 训练流水线存储和流式传输数据**的平台，解决了使用 HuggingFace 和 S3 进行数据管理时面临的挑战。
  
  - 他们欢迎 **demo 请求**，并渴望根据社区反馈进一步构建该平台。
- **通过新应用在学习中获得实时反馈**：**深度问答评估应用**利用 **Streamlit** 和 **Gemini 1.5 Pro**，旨在通过向用户提供即时反馈来增强在线学习体验。
  
  - 该应用受到 Fady AlNajjar 博士的启发，被定位为评估知识进展的重要工具。
- **使用 Book Mixer 创建独特的故事混搭**：一名成员发布了 **books-mixer-ai**，这是一个使用 **ReactJS** 和 AI 技术混合不同书籍情节的工具，已在 Hugging Face Spaces 上线。
  
  - 该工具让用户能够立即生成新的故事情节及配套视觉效果，相关文档即将发布。
- **在线协作构建故事**：针对叙事趋势，一名成员创建了 **协作式故事构建器**，旨在促进社区共同创作叙事。
  
  - 他们在 Hugging Face Spaces 上分享了该项目，并获得了社区的积极支持。
- **发布用于医疗 AI 的 WorldMedQA-V**：官方宣布了 **WorldMedQA-V**，这是一个全新的多语言、多模态医疗数据集，用于基准测试医疗领域的视觉语言模型。
  
  - 该数据集旨在推动医疗 AI 研究，可在 Hugging Face 上获取。

**提到的链接**：

- [Collaborative Story Builder - a Hugging Face Space by Pixeltable](https://huggingface.co/spaces/Pixeltable/Collaborative-Story-Builder)：未找到描述
- [Enhancing Learning Through Real-Time Feedback: In-Depth Question Answering Evaluation App](https://medium.com/@d.isham.ai93/enhancing-learning-through-real-time-feedback-in-depth-question-answering-evaluation-app-4f68c423e496)：在在线学习和自我提升的世界中，拥有有效的工具来评估个人的进步至关重要。无论你是在学习……
- [Tweet from Shan Chen (@shan23chen)](https://x.com/shan23chen/status/1846923442253152641)：🚀 AI4Health 的大消息！🌐 我们很高兴发布 WorldMedQA-V，这是一个多语言、多模态的医学考试数据集，旨在基准测试医疗领域的视觉语言模型！🩺💻 👉 ...
- [Books Mixer Ai - a Hugging Face Space by as-cle-bert](https://huggingface.co/spaces/as-cle-bert/books-mixer-ai)：未找到描述
- [GitHub - AstraBert/books-mixer-ai: Mix and twist your favorite books!📖](https://github.com/AstraBert/books-mixer-ai)：混合并改编你最喜欢的书籍！📖。通过在 GitHub 上创建账号来为 AstraBert/books-mixer-ai 做出贡献。

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1296488439423828038) (45 条消息🔥):

> - `关于 LLM 论文的讨论`
> - `加入会议的说明`
> - `服务器宗旨与社区`
> - `Zoom 会议安全性`
> - `活动录像可用性`

- **即将进行的精彩 LLM 论文讨论**：成员们分享了正在讨论的论文链接，例如[关于 LLM 逆向强化学习 (inverse RL) 的论文](https://arxiv.org/pdf/2410.12491)，其中一位作者也在场回答相关问题。
  
  - 社区鼓励大家积极参与并保持好奇心，强调会议面向所有知识水平的人员开放。
- **如何加入会议**：提供了关于如何使用 Discord 链接或 Zoom 邀请加入会议的说明，并附带了密码和会议 ID。
  
  - 成员们确认该链接会导向直播舞台，并向他人保证加入是安全的。
- **Hugging Face 服务器的宗旨**：一名成员询问了服务器的用途，引发了相关讨论，明确了其主要用于 Hugging Face 相关的支持和 AI 话题。
  
  - 这是一个任何人都可以展示研究论文的空间，旨在鼓励社区协作和知识共享。
- **关于链接安全性的担忧**：成员们对链接安全性表达了谨慎态度，并讨论了与 McGill University 关联的 Zoom 链接的可信度。
  
  - 社区成员向他人保证了所提供 Zoom 链接的安全性，强调了其可靠的关联机构。
- **录像与后续提问**：组织者宣布计划近期发布会议录像，并鼓励成员提出剩余的问题。
  
  - 社区受邀向作者发送额外的咨询，在直播活动结束后继续保持互动。

**提到的链接**：

- [加入我们的云高清视频会议](https://mcgill.zoom.us/j/85109438251?pwd=fxKIhHVTHySWGBRLunWNT7LuQp7pEX.1)：Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天及网络研讨会。Zoom ...
- [INDUS: Effective and Efficient Language Models for Scientific Applications](https://arxiv.org/abs/2405.10725)：在通用领域语料库上训练的大型语言模型 (LLMs) 在自然语言处理 (NLP) 任务上表现出了卓越的结果。然而，之前的研究表明，使用特定领域训练的 LLM...

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1296503341278302249) (3 条消息):

> - `特定任务`
> - `私信 (Direct messaging)`

- **关于特定任务的询问**：一位成员询问是否有特定的任务或主题需要讨论。
  
  - 这体现了澄清和聚焦对话内容的开放态度。
- **私信邀请**：另一位成员给出了肯定的回答，并邀请第一位成员私信 (inbox) 他们以便进行更直接的沟通。
  
  - 这表明愿意就具体细节进行一对一的讨论。

 

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1296403083441606686) (4 条消息):

> - `微调的数据集格式`
> - `NLLB 置信度显示`

- **使用不同数据集格式微调模型**：一位成员询问是否可以使用 instruct 格式的数据集微调基座模型 (base model)，以及是否可以使用 raw text 数据集微调 instruct 模型以获取特定领域知识。
  
  - 另一位成员确认第一种方法是正确的，但警告说第二种方法会导致 **错误的输出 (wrong outputs)**。
- **关于 NLLB 文本置信度的问题**：一位成员询问如何在 NLLB 中显示翻译文本的置信度，并参考了 whisper.cpp 的 **-ojf** 参数，该参数可以生成包含每个单词置信度的 JSON 文件。
  
  - 他们希望在 NLLB 中也能有类似的功能来评估翻译的准确性。

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1296306634460495882) (30 条消息🔥):

> - `将模型文件夹转换为 Safetensors`
> - `在微调模型中使用 ControlNet`
> - `Google Colab 中的 Kwai Kolors 错误`
> - `租用 VM 进行模型训练`
> - `在 ControlNet 训练中使用 CLIP`

- **如何将文件夹模型转换为 Safetensors？**: 成员们讨论了将模型文件夹转换为 Safetensors 的过程，澄清了 Safetensors 仅仅是权重，并建议重用权重而不是转换文件夹结构。
  
  - *大多数人使用 sd1-5，因此 unet 中的 safetensors 是大多数人需要的。*
- **关于重新训练 ControlNet 的问题**: 一位成员提出了关于使用新的微调基础模型重新训练 ControlNet 的疑问，这引发了对训练影响的不确定性。
  
  - 讨论指出，在与 ControlNet 相关的 GitHub 讨论中可能会找到更好的支持。
- **Google Colab 中的 Kwai Kolors 错误**: 一位用户报告在 Google Colab（尤其是免费版）中运行 Kwai Kolors 时遇到了与 numpy 和 safetensors 版本相关的错误。
  
  - 成员们指出，高效运行该模型需要额外的 VRAM，建议切换到专业版计划或在本地运行。
- **租用 VM 进行模型训练的建议**: 有关于租用 VM 训练模型的建议讨论，提到了 Amazon、FAL 和 Replicate 等热门选项。
  
  - 成员们强调使用私有 VM 以获得软件需求（如 conda）方面的灵活性。
- **在 ControlNet 中使用 CLIP 编码器**: 一位用户询问是否可以在自定义 ControlNet 训练中用图像编码器替换 CLIP 文本编码器，以避免生成大量的文本描述（captions）。
  
  - 讨论中提出了对训练过程中可能对数据集中有限数量的独特面孔产生过拟合（overfitting）的担忧。

**提到的链接**:

- [yisol/IDM-VTON · Hugging Face](https://huggingface.co/yisol/IDM-VTON): 未找到描述
- [Adapting ControlNet to a New Finetuned Model · huggingface/diffusers · Discussion #9694](https://github.com/huggingface/diffusers/discussions/9694): 我使用了来自 diffusers 的 ControlNet 训练脚本来获取模型。该模型是基于 jzli/majicMIX-realistic-7 训练的。然后我使用来自 ... 的 DreamBooth 脚本微调了这个基础模型。
- [diffusers/scripts at main · huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main/scripts): 🤗 Diffusers: 用于 PyTorch 和 FLAX 中图像和音频生成的尖端 Diffusion 模型。- huggingface/diffusers

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1296195368316440627) (379 条消息🔥🔥):

> - `Gandalf Challenges` (Gandalf 挑战)
> - `Octopus Theme in LLM Responses` (LLM 回复中的章鱼主题)
> - `Control Vectors for LLM Outputs` (用于 LLM 输出的 Control Vectors)
> - `Lambda Chat History Stability` (Lambda 聊天记录稳定性)
> - `New Model Features` (新模型特性)

- **Gandalf 挑战取得高度成功**：参与者分享了 Gandalf 挑战的经验，一些人通过创意的 Prompt 策略获得了高排名。
  
  - 方法包括要求提供带有隐藏标准的列表或玩“21个问题”游戏，展示了挑战的迭代性质。
- **章鱼引用引发反响**：LLM 输出中一个反复出现的主题是对“章鱼”的痴迷，导致用户推测其重要性或与密码的联系。
  
  - 对话围绕着诱导模型写诗并间接揭示所需信息展开，一位用户幽默地谈到了词语操纵。
- **Control Vectors 与 Prompt 注入**：用户讨论了尝试使用 Control Vectors 来绕过模型限制，特别是旨在获取敏感输出。
  
  - 采用了多种策略，从要求特定格式的答案到操纵 System Prompts 以诱导更多信息。
- **Lambda 聊天记录问题**：用户对 Lambda 上聊天记录的稳定性表示担忧，注意到他们的对话似乎随着时间的推移而消失。
  
  - 这引发了关于平台功能及其对用户体验影响的疑问。
- **探索新模型特性**：关于新模型能力的讨论突出了对 128K Context 模型的关注及其运行状态。
  
  - 尽管持乐观态度，但当新模型未按预期运行时，用户表达了失望。

**提到的链接**：

- [Guardrails Arena - a Hugging Face Space by lighthouzai](https://huggingface.co/spaces/lighthouzai/guardrails-arena)：未找到描述
- [xjdr (@_xjdr) 的推文](https://x.com/_xjdr/status/1846640821107675618)：Nemotron-70B entropix 版本非常出色
- [Gandalf | Lakera – 测试你的 Prompt 技巧，让 Gandalf 泄露秘密信息。](https://gandalf.lakera.ai/baseline)：诱导 Gandalf 泄露信息，亲身体验大语言模型的局限性。
- [Gandalf | Lakera – 测试你的 Prompt 技巧，让 Gandalf 泄露秘密信息。](https://gandalf.lakera.ai/basel)：诱导 Gandalf 泄露信息，亲身体验大语言模型的局限性。
- [Gandalf | Lakera – 测试你的 Prompt 技巧，让 Gandalf 泄露秘密信息。](https://gandalf.lakera.ai/adventure-8)：诱导 Gandalf 泄露信息，亲身体验大语言模型的局限性。
- [google/gemma-7b-aps-it · Hugging Face](https://huggingface.co/google/gemma-7b-aps-it)：未找到描述
- [由 bursteratom 提交的 Pull Request #1974 · axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl/pull/1974)：描述：修复了 issue#1966，即 eval_sample_packing=True 导致多 GPU 评估卡住的问题。动机与背景：在 issue#1966 中，多 GPU 上的样本打包数据集评估...

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1296185939617513472) (12 条消息🔥):

> - `Sampling Parameters` (采样参数)
> - `LLM Programming Languages` (LLM 编程语言)
> - `LLM Jailbreak Resources` (LLM Jailbreak 资源)
> - `God Archetypes for AI Models` (AI 模型的上帝原型)
> - `JavaScript vs Python` (JavaScript vs Python)

- **检查采样参数**：一位成员建议检查 **sampling parameters**（temp, top-p, top k），但另一位成员确认 **它们是相同的**。
  
  - 这表明尽管参数未变，但性能仍存在一些不确定性。
- **为 LLM 选择编程语言**：一位成员询问顶尖 LLM 最擅长哪种编程语言，在 **JavaScript** 和 **Python** 之间权衡。
  
  - 另一位成员认为 LLM **纠缠于 Python**，同时希望它们尝试用 JavaScript 编写代码，而不仅仅是追求单行代码。
- **LLM Jailbreak 资源**：在关于 **LLM jailbreaks** 资源的讨论中，一位成员提到可以查看 **plineys discord**。
  
  - 然而，另一位成员表示该社区内部存在混乱，希望能有更多替代资源。
- **AI 模型的幽默上帝原型**：有一个轻松的问题关于 AI 模型会是哪位神，**Opus 被认为是普罗米修斯 (Prometheus)**，而 **Hermes-3 是奥丁 (Odin)**。
  
  - 成员们觉得 **Hermes 参与讨论** 很有趣，暗示了这个话题的幽默感。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 条消息):

trre: [https://arxiv.org/abs/2410.11163](https://arxiv.org/abs/2410.11163)

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1296201963113021472) (6 条消息):

> - `Ollama GGUF 模型使用`
> - `模型训练中的 AI 怀疑论`
> - `SCP 生成器开发`

- **Ollama 简化了 GGUF 模型执行**：Ollama 允许用户使用命令 `ollama run <model_url>` 直接运行 Hugging Face 上的任何 GGUF 模型，无需创建新的 Modelfile。
  
  - 拥有超过 **4.5 万个公开的 GGUF Checkpoints**，这通过提供量化类型和系统提示词（System Prompts）等可定制选项，提升了用户体验。
- **加密货币为 Shitposting 见解付费**：一位成员幽默地建议，为 Shitposter 付费可能是加密货币的一个有趣用例，反映了人们对非常规应用日益增长的兴趣。
  
  - 他们还表示，未来的 AI 模型在训练中需要加入**更多怀疑态度 (Skepticism)**。
- **SCP 生成器在 GitHub 上线**：由 dottxt-ai 创建的新 [SCP 生成器](https://github.com/dottxt-ai/cursed/tree/main/scp) 利用大纲生成 SCP 故事，为 SCP 社区做出贡献。
  
  - 该项目开放贡献，邀请开发者参与其持续开发。

**提到的链接**：

- [在 Hugging Face Hub 上将 Ollama 与任何 GGUF 模型配合使用](https://t.co/nxonkJRzW0)：未找到描述
- [dottxt-ai/cursed 的 main 分支下的 scp 目录](https://github.com/dottxt-ai/cursed/tree/main/scp)：通过在 GitHub 上创建账号来为 dottxt-ai/cursed 的开发做出贡献。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 条消息):

trre: [https://arxiv.org/abs/2410.11163](https://arxiv.org/abs/2410.11163)

---

### **Eleuther ▷ #**[**announcements**](https://discord.com/channels/729741769192767510/794042109048651818/1296325016698490880) (1 条消息):

> - `机械论异常检测 (MAD)`
> - `Llama 3.1 性能`
> - `Mistral 7B v0.1 对比`
> - `异常检测技术`
> - `古怪任务 (Quirky Task) 性能表现`

- **MAD 在不同模型中表现各异**：最近关于 **机械论异常检测 (Mechanistic Anomaly Detection, MAD)** 的测试发现，在类似的训练下，**Llama 3.1 8B** 在非算术任务上的表现比 **Mistral 7B v0.1** 更差。
  
  - *Llama 表现出较少的古怪行为 (Quirky Behavior)*，但在各项任务中实现了较低的平均损失，表明其具有更强的真值偏差 (Ground Truth Bias)。
- **异常检测技术产生相似结果**：两种使用 **Normalising Flow** 和 **Sparse Autoencoder** 激活检测异常的新方法，在 **Llama 3.1 Base** 的隐藏状态上显示出与 **马氏距离 (Mahalanobis distance)** 相当的性能。
  
  - 然而，以往 MAD 技术的不一致性仍然明显，在实现统一有效的检测方面面临挑战。
- **关于古怪任务的性能见解**：研究表明，隐藏状态中**质心上下文之间的距离 (Distance between centroid contexts)** 有效地解释了 MAD 的性能，**Llama** 显示出的分离度低于 **Mistral**。
  
  - 这一见解强化了 MAD 技术在不同任务（尤其是那些被认为是“古怪”的任务）中的多变性。
- **关于进展的更新博客文章**：[博客](https://blog.eleuther.ai/mad_research_update_2/)上发布了一篇更新，讨论了机械论异常检测测试的最新发现。
  
  - 该文章全面概述了性能差异以及对所测试方法的响应。

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1296187716211314768) (26 messages🔥):

> - `LLM Re-ranking 技术`
> - `Workshop 中的匿名政策`
> - `评估 OpenAI 的 Text Embeddings`
> - `使用 Decoder Only 模型生成 Embeddings`
> - `开源 AI 研究贡献`

- **LLM Re-ranking 技术提升准确性**：Re-ranking 被讨论为使用先进 [machine learning](https://www.nvidia.com/en-us/glossary/machine-learning/) 算法改进搜索结果的关键技术，正如一位成员引用 [此实现](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/) 所指出的。Re-ranking 的目标是优化初始搜索输出，使其更好地符合用户意图，从而为用户提供更准确、更相关的信息。
- **Workshop 匿名政策各不相同**：会议中的 Workshop 设置通常比整体会议要求的匿名政策更宽松，正如针对不同地点的讨论所指出的。这些信息旨在帮助成员更好地应对不同活动的匿名要求。
- **对 OpenAI Text Embeddings 的批评**：一位成员表示担心 OpenAI 的 text embedding 模型虽然在发布时表现良好，但按 2024 年的标准来看已显不足，特别是当市场充斥着像 Mistral fine-tunes 这样的模型时。这些担忧反映了关于当前应用中 embedding 模型演进和有效性的更广泛讨论。
- **Decoder-Only 模型可有效用于 Embeddings**：成员们讨论了使用 decoder-only 模型提取 embeddings 的可行性，并澄清 attention masking 是其与 encoder 模型的主要区别。还提到了使用 *llm2vec* 等方法的潜力，以及对于许多应用来说，更简单的模型可能已经足够的建议。
- **AI 研究中的开源机会**：一位具有 quantization 经验的成员表示有兴趣为专注于高效 inference 或新颖架构的开源 AI 研究做出贡献。他们正在寻找目前开放协作的项目，并强调了自己的背景和贡献意愿。

**提及的链接**：

- [NVIDIA Technical Blog | 面向开发者、数据科学家和 IT 管理员的新闻与教程](https://developer.nvidia.com/blog)：面向开发者、科学家和 IT 管理员的新闻与教程。
- [通过 Re-Ranking 增强 RAG 流水线 | NVIDIA Technical Blog](https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/)：在 AI 驱动应用快速发展的格局中，re-ranking 已成为增强企业搜索结果精确度和相关性的关键技术。

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1296219192210493450) (315 messages🔥🔥):

> - `Muon Optimizer Performance` (Muon 优化器性能)
> - `Rectified Flow Noise Choices` (Rectified Flow 噪声选择)
> - `Pyramid Noise in Stable Cascade` (Stable Cascade 中的 Pyramid Noise)
> - `Latent Space Considerations` (Latent Space 考量)
> - `Initial Training Techniques in State Space Models` (State Space Models 中的初始训练技术)

- **Muon 优化器表现出色**：训练结果显示，与 AdamW 相比，Muon 优化器在更少的 token 下实现了显著更低的验证损失，在 GPT-2（最高达 1.5B 参数）等模型上表现出强大的性能扩展性。
  
  - 讨论强调了 Muon 新的分布式实现带来的效率提升和更低的开销。
- **Rectified Flow 与噪声选项**：用户讨论了 Rectified Flow 中噪声分布的选择，指出虽然 Gaussian 噪声常用，但 Perlin 噪声等替代方案在某些应用中也可能有效。
  
  - 参与者认为，噪声选择的灵活性可以更好地适应特定的目标分布。
- **Stable Cascade 中的 Pyramid Noise**：Stable Cascade 在阶段 B 使用了 pyramid noise，可能比其他噪声类型更有效地捕捉不同的图像尺度。
  
  - Pyramid noise 被描述为单位 Gaussian 的堆叠，引发了关于其与生成实际 pink noise 相比计算效率的疑问。
- **Latent Space 形状变化**：在讨论 Latent Space 中的噪声应用时，参与者注意到，尽管 latent 分布看起来可能不同，但它们可以保留必要的空间对应关系。
  
  - 对话思考了分布的相似性对于噪声方法在 Latent Space 中是否有效是否至关重要。
- **State Space Models 中的训练技术**：为 State Space Models 提出的一种新的结构化初始化技术旨在增强其在召回任务上的性能，从而实现更好的从零开始的复制能力。
  
  - 有人建议，之前报道的这些模型性能不佳可能是由于训练困难而非容量限制。

**提及的链接**：

- [leloy! (@leloykun) 的推文](https://x.com/leloykun/status/1846842883967692926)：选择 Muon 的理由 1) 我们可以在非欧几里得空间中“更快”地下降 2) Adam/Shampoo/SOAP 等动态学习预条件矩阵，等同于学习下降的范数和空间 3) Muon s...
- [Mimetic Initialization Helps State Space Models Learn to Recall](https://arxiv.org/abs/2410.11135)：最近的研究表明，像 Mamba 这样的 State Space Models 在基于召回的任务上明显逊于 Transformers，因为它们的状态大小相对于输入是恒定的...
- [Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](https://arxiv.org/abs/2410.11163)：我们提出了 Model Swarms，一种通过群体智能（引导个体系统的集体行为）来适配 LLM 的协作搜索算法。具体而言，Model Swarms 从一个 LLM 池开始...
- [Thinking LLMs: General Instruction Following with Thought Generation](https://arxiv.org/abs/2410.10630)：LLM 通常被训练为以类似于人类专家的方式回答用户问题或遵循指令。然而，在标准的对齐框架中，它们缺乏显式的...
- [Untie the Knots: An Efficient Data Augmentation Strategy for Long-Context Pre-Training in Language Models](https://arxiv.org/abs/2409.04774)：大语言模型（LLM）优先考虑扩展上下文窗口，以便模型能够整合更多信息。然而，训练模型处理长上下文面临重大挑战...
- [Yuchen Jin (@Yuchenj_UW) 的推文](https://x.com/yuchenj_uw/status/1846964136204173318?s=46)：Muon 再次扩展！我使用 @kellerjordan0 的 Muon 优化器训练了最大的 GPT-2 (1.5B)，仅用 4.2B token 就实现了 2.90 的 Fineweb 验证损失——仅为所需 token 的 42%...
- [madebyollin - 概览](https://github.com/madebyollin/)：制作了 sdxl-vae-fp16-fix, taesd, 以及那个 pokemon-emulation-via-dnn 东西。- madebyollin
- [GitHub - PufferAI/PufferLib: Simplifying reinforcement learning for complex game environments](https://github.com/PufferAI/PufferLib)：简化复杂游戏环境的强化学习 - PufferAI/PufferLib
- [GitHub - SonicCodes/vmf-vae](https://github.com/SonicCodes/vmf-vae)：通过创建账户为 SonicCodes/vmf-vae 的开发做出贡献。

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1296333438596546581) (1 条消息):

> - `Model Hallucination Evaluation Methods`
> - `Research Papers on Hallucinations`

- **评估模型 Hallucination 的流行方法**：一名成员询问了在当前研究中量化或评估模型 **Hallucination** 的**流行且可靠的方法**，并请求提供相关论文的链接。
  
  - 讨论表明，人们有兴趣建立稳健的指标来评估模型输出的忠实度，并确定文献中已有的最佳实践。
- **寻求 Hallucination 研究资源**：参与者欢迎关于在哪里可以找到解决 Hallucination 评估问题的**有价值研究论文**的建议和见解。
  
  - 需要就如何在进行中的项目中有效利用这些方法进行集中讨论。

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1296524119705124907) (4 条消息):

> - `Saving Model Content`
> - `Verbose Warnings in Hugging Face`
> - `Log Samples Parameter`
> - `Issues with Summarizing Tasks`

- **在测试期间保存内容**：一名成员询问是否有方法可以保存在测试阶段由模型生成的内容。
  
  - 另一名成员迅速回应称，使用 `--log_samples` 参数可能有助于实现这一点。
- **Hugging Face Adapter 的冗长警告**：一位用户报告称，在向 Hugging Face adapter 传递预训练模型实例时收到了冗长的警告，这可能是由于预期不匹配造成的。
  
  - 他们链接到了 [lm-evaluation-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness/blob/624017b7f4501638b0d5848d0f0eab2914a7fb2c/lm_eval/models/huggingface.py#L1362) 中的特定代码行，并描述了与模型 SHA 获取相关的错误。
- **摘要任务中的空响应**：一名成员对在摘要或翻译相关任务中收到空响应列表 (`resps=[], filtered_resps={}`) 表示担忧。
  
  - 他们提到正尝试进一步排查该问题以寻找解决方案。

 

**提到的链接**：[lm-evaluation-harness/lm_eval/models/huggingface.py at 624017b7f4501638b0d5848d0f0eab2914a7fb2c · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/624017b7f4501638b0d5848d0f0eab2914a7fb2c/lm_eval/models/huggingface.py#L1362)：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1296210656856641666) (2 条消息):

> - `NVIDIA Nemotron 70B`
> - `Grok 2 Pricing Update`

- **NVIDIA Nemotron 70B 碾压竞争对手**：**NVIDIA Nemotron 70B** 在多项评估中超越了 **Llama 3.1 405B**、**GPT-4o** 和 **Claude 3.5 Sonnet**，据报告其在 Arena Hard 得分为 **85.0**，AlpacaEval 2 LC 为 **57.6**，MT Bench 为 **8.98**。
  
  - 您可以在[此处](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct)查看结果并进行尝试。
- **成本上升，Grok 2 重新定价**：**xAI** 提高了 **Grok 2** 的价格，现在费用为 **$5/m input** 和 **$10/m output**，而 Grok 2 Mini 仍不可用。
  
  - 尽管价格上涨，Grok 2 仍处于趋势中，可以通过[此处](https://openrouter.ai/x-ai/grok-2)访问。

**提到的链接**：

- [来自 OpenRouter (@OpenRouterAI) 的推文](https://x.com/OpenRouterAI/status/1846651197802881094)：开源领域的大日子：NVIDIA Nemotron 70B 在多项评估中击败了 Llama 405B、GPT-4o 和 Claude 3.5 Sonnet：Nemotron 70B vs Claude 3.5 vs GPT4o: > Arena Hard: 85.0 | 79.2 ...
- [Grok 2 - API, Providers, Stats](https://openrouter.ai/x-ai/grok-2)：Grok 2 是 xAI 的前沿语言模型，具有最先进的推理能力，最适合复杂和多步骤的使用场景。要使用更快的版本，请参阅 [Grok 2 Mini](/x-ai/grok-2-mini)。

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1296186003488505909) (233 条消息🔥🔥):

> - `Grok 2 状态`
> - `OpenRouter 性能`
> - `语音交互模型`
> - `Deepseek 模型更新`
> - `O1 模型性能`

- **Grok 2 以更高价格回归**：Grok 2 已重新上线，定价为 **$5/$10**，而 mini 版本目前仍不可用。用户对价格上涨表示惊讶，并讨论了其可能带来的影响。
  
  - 分享了其当前产品的链接，提供了更多关于功能和定价的详细信息，详见[此处](https://openrouter.ai/x-ai/grok-2)。
- **OpenRouter 模型与定价概览**：讨论涵盖了通过 OpenRouter 提供的各种模型，特别提到了 **SambaNova** 及其与 **Groq** 相比的可扩展性。用户注意到 **Yi Lightning** 的定价极具吸引力，输入价格为 **$0.14/m**，且相比竞争对手具有优势。
  
  - 有观点认为，随着按需付费（pay-as-you-go）模式的普及，未来可能会对自研芯片推理提供商的定价结构有更深入的了解。
- **各种 LLM 的语音交互限制**：用户对 **GPT-4o** 等模型的语音功能表现表示担忧，特别是其对不同语言的处理和音频输出质量。用户指出，虽然语音输入工作正常，但语音输出可能会变得“古怪”，尤其是在中文等语言中。
  
  - 共识是 **Google** 的 **Gemini** 凭借其一致的设计标准，得以更早发布语音输入功能。
- **Deepseek 模型更新**：对话包括了关于 **Deepseek** 的更新以及对潜在新版本（如 **Deepseek-vl 2**）的推测。用户对 **Deepseek** 模型的现状及其未来的能力表示好奇。
- **O1 模型可用性担忧**：用户讨论了 **O1** 的性能，指出其在遵循指令方面存在困难，并且在引用先前对话历史时存在问题。一些人发现它过度啰嗦且无法提供连贯的输出，引发了对其在各种任务中实际应用的担忧。

**提到的链接**：

- [Gyazo](https://gyazo.com/0b1505d3e5d2939cabaf3fd8857f6e03):
- [快速入门 | OpenRouter](https://openrouter.ai/docs/quick-start): 开始使用 OpenRouter 进行构建
- [OpenRouter](https://openrouter.ai/x-ai/g): LLM 路由与市场
- [OAuth PKCE | OpenRouter](https://openrouter.ai/docs/oauth): 通过 OAuth 实现安全用户认证
- [Grok 2 - API、提供商、统计数据](https://openrouter.ai/x-ai/grok-2): Grok 2 是 xAI 的前沿语言模型，具有最先进的推理能力，最适合复杂和多步骤的使用场景。要使用更快的版本，请参阅 [Grok 2 Mini](/x-ai/grok-2-mini)。
- [Llama 3.1 Nemotron 70B Instruct - API、提供商、统计数据](https://openrouter.ai/nvidia/llama-3.1-nemotron-70b-instruct): NVIDIA 的 Llama 3.1 Nemotron 70B 是一款旨在生成精确且有用响应的语言模型。通过 API 运行 Llama 3.1 Nemotron 70B Instruct。

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1296189511948046418) (105 条消息🔥🔥):

> - `Perplexity API 性能`
> - `Nvidia Llama 3 模型对比`
> - `Spaces 中的文件上传问题`
> - `使用 Claude 70B 进行 YouTube 视频分析`
> - `Perplexity 订阅取消`

- **Perplexity API 性能问题**：用户报告 **Perplexity API** 响应时间缓慢，简单查询需要 **1 到 2 分钟**。
  
  - 虽有建议尝试对响应时间进行基准测试，但许多人认为目前的性能并不令人满意。
- **Nvidia 的 Llama 3 模型表现优于替代方案**：一位用户强调，根据对齐基准测试，Nvidia 的 **Llama 3.1-Nemotron-70B** 模型被认为优于 **GPT-4** 和 **Claude 3.5 Sonnet**。
  
  - 该模型在各项基准测试中均获得高分，使其成为 LLM 领域强有力的竞争者。
- **Spaces 文件上传问题**：多名成员讨论并确认了在向 **Spaces** 上传文件时遇到的问题。
  
  - 官方已确认正在修复中，以解决目前的上传故障。
- **使用 Claude 70B 进行 YouTube 视频分析**：一位用户尝试通过提供链接让 **Claude 70B** 分析 YouTube 视频，但在处理直播视频时遇到了限制。
  
  - 该模型在一次测试中不需要字幕文件，这暗示它可能会在有可用字幕时自动生成。
- **Perplexity 订阅取消指南**：一位用户询问如何取消 **Pro Monthly Subscription** 并因对结果不满申请退款。
  
  - 其他人建议在设置中查找取消板块，并强调了通过设置管理订阅的方法。

**提到的链接**：

- [Skull Issues GIF - Skull issues - 发现并分享 GIF](https://tenor.com/view/skull-issues-gif-13031152103567454559)：点击查看 GIF
- [Aravind Srinivas (@AravSrinivas) 的推文](https://x.com/aravsrinivas/status/1847030982211522852?s=46)：Answer Truck 使用 FSD 从加利福尼亚州行驶 1000 多英里到达德克萨斯州。明天它将在奥斯汀停靠，参加 Perplexity 用户见面会。地点：奥斯汀市中心 La Volta Pizza，下午 1 点...
- [Arangutan Monkey GIF - 猩猩跳舞 - 发现并分享 GIF](https://tenor.com/view/arangutan-monkey-dancing-gif-15130385)：点击查看 GIF
- [nvidia/Llama-3.1-Nemotron-70B-Instruct-HF · Hugging Face](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF)：未找到描述

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1296190249818652704) (12 条消息🔥):

> - `Oura Ring 4 评测`
> - `珠峰探险者遗骸`
> - `理解 API`
> - `Starlink 千兆速度计划`
> - `投资爱得 (Tou Zi Aide) ETF`

- **Oura Ring 4 评测**：[Oura Ring 4](https://www.perplexity.ai/page/oura-ring-4-review-5U7Rj9.hR3W0MRa_OmQgbQ) 因其先进的健康追踪功能和时尚设计而备受关注。
  
  - 核心亮点包括其在睡眠监测和健康洞察方面提升的准确性。
- **珠峰探险者遗骸被发现**：来自 [Everest Explorer](https://www.perplexity.ai/page/everest-explorer-s-remains-fou-j3h5Up0rTdyHtGGVmhnC5Q) 的最新发现揭示了重大的考古进展。
  
  - 这些遗骸揭示了过去在极端条件下探险所面临的挑战。
- **向大众解释 API**：关于 [什么是 API](https://www.perplexity.ai/search/what-is-an-api-6HaQAJlXRGOWBgQd3L7Iyg#0) 的详细概述，涵盖了对开发者至关重要的定义和功能。
  
  - 理解 API 可以简化不同软件应用程序之间的交互。
- **Starlink 千兆速度计划公布**：[Starlink Gigabit Speed Plan](https://www.perplexity.ai/page/starlink-gigabit-speed-plan-knyorEQ7SYG11t4a.dd2Ig) 承诺为农村地区提供更快的网速，引发了广泛关注。
  
  - 用户对卫星互联网可能达到前所未有的速度水平感到兴奋。
- **投资爱得 (Tou Zi Aide) ETF 讨论**：[Tou Zi Aide ETF](https://www.perplexity.ai/search/tou-zi-aide-etf-U8cMUG4uQu.geJ8bPz5B0w) 引发了关于其投资策略和市场影响的讨论。
  
  - 在波动的市场环境下，投资者正在评估其相对于传统基金的表现。

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1296227322390904922) (3 messages):

> - `LFM 40B API 可用性`
> - `新 spaces 功能 API`

- **关于 LFM 40B API 的咨询**：一位成员询问 [labs.perplexity.com](https://labs.perplexity.com) 上的 **LFM 40B** 是否有可能通过 API 提供。
  
  - 未记录到关于此查询的回复。
- **新 Spaces 功能 API 的不确定性**：另一位成员询问了为新的 **spaces 功能** 提供 **API** 的可能性。
  
  - 一条回复澄清说 **主平台没有 API**，表明了 API 与 perplexity.ai 之间的区别。

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1296213325717442641) (78 messages🔥🔥):

> - `O1-mini vs Sonnet 3.5`
> - `Aider 在不同平台上的安装`
> - `O1-preview 的成本影响`
> - `Architect mode 工作流`
> - `关于 AI 辅助编程的反馈`

- **O1-mini 展示了强大的迭代能力**：尽管最初用户持怀疑态度，但 **O1-mini** 在复杂任务上的表现证明优于 **Claude 3.5**，通过有效的迭代，以更少的迭代次数更快地完成了任务。
  
  - 几位用户报告称，虽然 O1-mini 在特定场景下表现出色，但由于熟悉度和可靠性，他们仍然倾向于在大多数任务中使用 **Sonnet 3.5**。
- **访问 O1-preview 的成本引发关注**：用户对 **O1-preview** 每 100 万 tokens 60 美元的价格表示担忧，这使得对于已经订阅了 **ChatGPT Plus** 的用户来说，其可行性较低。
  
  - **Sonnet 3.5** 等替代模型仍然是寻求高性价比选项的用户的热门选择。
- **各平台的 Aider 安装技巧**：用户一直在分享安装 **Aider** 的经验，例如建议在 **Windows 11** 上使用 **pipx** 进行便捷安装。
  
  - 关于在 **Chromebooks** 上安装的咨询也浮出水面，显示了跨平台可访问性的需求。
- **探索 Architect Mode 的替代方案**：一位用户寻求在没有直接访问 **O1-preview** 权限的情况下模拟 **Architect mode** 的指导，讨论了在 **Aider** 中使用 **ChatGPT** 输出的潜在工作流和提示词。
  
  - 这次对话强调了在成本和访问受限的情况下，寻求实现类似功能的替代方法的需求。
- **AI 辅助下编程的演变**：普遍观点认为编程正在进入一个**新时代**，AI 工具显著改变了编程，简化了编码任务。
  
  - 一些用户指出，对 AI 的依赖改变了他们自己的编码习惯，引发了对传统编码技能影响以及技能差距的担忧。

**提到的链接**：

- [VSCode Aider - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=Apertia.vscode-aider)：Visual Studio Code 扩展 - 直接在 VSCode 中运行 Aider，实现无缝集成和增强的工作流。
- [Chat modes](https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model)：使用 chat、ask 和 help chat 模式。
- [Supported languages](https://aider.chat/docs/languages.html)：Aider 支持几乎所有流行的编程语言。
- [mattf - Overview](https://github.com/MattF)：mattf 拥有 98 个公开仓库。在 GitHub 上关注他们的代码。
- [The plugin currently doesn't work with Windows · Issue #3 · MattFlower/vscode-aider-extension](https://github.com/MattFlower/vscode-aider-extension/issues/3)：目前，该插件无法在 Windows 上运行。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1296205033632698368) (32 messages🔥):

> - `Token Limits in Models`（模型中的 Token 限制）
> - `Azure API Configuration`（Azure API 配置）
> - `Aider Installation Issues`（Aider 安装问题）
> - `Git Issues with Aider`（Aider 的 Git 问题）
> - `DeepSeek Model Challenges`（DeepSeek 模型挑战）

- **Token 限制困扰用户**：多名成员报告在使用各种模型时达到了 Token 限制，特别提到 **claude-3-5-sonnet** 和 **DeepSeek** 超过了它们的限制。
  - 管理这些问题的建议包括使用 `/clear` 来减少聊天历史记录，以及将代码拆分为更小的文件。
- **Azure API 配置**：一位用户询问了 Azure API 密钥，回复明确了版本控制和配置取决于个人设置，并重点介绍了指向 Azure 文档的链接。
  - 此外，还为 Mac/Linux 和 Windows 用户详细说明了包括导出变量在内的设置过程。
- **Aider 安装故障排除**：一名成员表示在安装 Aider 时遇到困难，特别是安装下载过程中与 **NumPy** 相关的错误。
  - 解决建议包括确保在 Chromebook 上使用 Penguin 中的 Linux 正确安装了 **pip** 和 **Python**。
- **Aider 错误处理中的潜在 Bug**：有成员担心 Aider 将更改提交到错误的文件，导致不可逆的损坏，强调了正确文件跟踪的重要性。
  - 成员们讨论了使用 **git** 命令进行回滚，并提交 Bug 报告以解决 Aider 在文件编辑中可能存在的偏差。
- **编辑器建议和自定义**：一位用户请求更改 Aider 中文件路径补全的工作方式，希望其行为类似于 Bash，即仅补全当前路径元素。
  - 讨论内容包括潜在的配置问题，以及使用 CLI 标志与 YAML 配置文件时的优先级顺序。

**相关链接**：

- [Azure](https://aider.chat/docs/llms/azure.html)：aider 是你终端里的 AI 配对编程助手
- [Token limits](https://aider.chat/docs/troubleshooting/token-limits.html)：aider 是你终端里的 AI 配对编程助手

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/) (1 messages):

apcameron：最近刚发布 [https://mistral.ai/news/ministraux/](https://mistral.ai/news/ministraux/)

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1296211253945176104) (14 messages🔥):

> - `Multi-node Clusters`（多节点集群）
> - `AI Hackathon Announcement`（AI 黑客松公告）
> - `Inverse Reinforcement Learning for LLMs`（LLM 的逆强化学习）
> - `Open Source ML/AI Projects`（开源 ML/AI 项目）

- **多节点集群引发以太网问题**：用户讨论了在网络中设置 **4 个 V100s** 集群的问题，同时强调 Lambda 除非使用 **Infiniband**，否则缺乏多节点集群的选项。
  - 一名成员提到，尽管有些人更喜欢在实验设置中使用 **Ethernet**（以太网），但 **纯 DDP** 可能会消除对 Infiniband 的需求。
- **Gen AI Agents 黑客松公告**：发布了一项由 **CreatorsCorner** 与多家科技公司合作举办的黑客松公告，重点是创建 **AI 驱动的多智能体系统 (multi-agent systems)**。
  - 鼓励参与者在构建增强人类日常生活潜力的解决方案时考虑伦理影响。
- **关于 LLM 逆强化学习 (Inverse Reinforcement Learning) 的讨论**：一名成员分享了一篇关于将 **逆强化学习 (inverse RL)** 用于 **LLM** 的研究论文链接，并寻求社区的反馈。
  - 这一研究方向旨在探索改进语言模型机制的新方法。
- **关于知名开源 ML/AI 项目的查询**：一名成员询问了除 **Deepspeed** 和 **ONNX** 之外的其他知名开源 ML/AI 项目。
  - 社区成员提供了精选的机器学习框架和工具列表链接，鼓励探索各种项目。

**相关链接**：

- [Vertical Specific AI Agents Hackathon · Luma](https://lu.ma/ke0rwi8n)：Gen AI Agents CreatorsCorner，与 aixplain, Sambanova Systems, Prem, Marly, Senso, Mistral, coval, heygen, fiberplane, exa 等合作……
- [GitHub - gpu-mode/awesomeMLSys: An ML Systems Onboarding list](https://github.com/gpu-mode/awesomeMLSys)：一个 ML 系统入门清单。通过在 GitHub 上创建一个账号来为 gpu-mode/awesomeMLSys 做出贡献。
- [GitHub - josephmisiti/awesome-machine-learning: A curated list of awesome Machine Learning frameworks, libraries and software.](https://github.com/josephmisiti/awesome-machine-learning)：一个精选的优秀机器学习框架、库和软件列表。

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1296580059183972453) (1 条消息):

> - `频道关闭`
> - `与 Triton 的关联`

- **频道关闭异常**：开启该频道的同一个人又将其关闭，这一决定引起了关注。
  
  - 鉴于声称这是“计划外”的，这一行为被描述为“奇怪”。
- **无 Triton 关联**：有人指出，负责关闭频道的个人并不隶属于 **Triton**。
  
  - 这一区别增加了围绕频道管理的困惑。

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1296216375228104854) (9 条消息🔥):

> - `PyTorch 2.5 发布`
> - `Torch.compile 开销`
> - `SGD Fused 更新`

- **PyTorch 2.5 正式发布！**：[PyTorch 2.5](https://anaconda.org/pytorch/pytorch) 的发布已确认，现在可以在 conda 和 PyTorch 的 pip 索引中获取 wheel 文件。
  
  - 针对发布的兴奋情绪，有人提到 *“还以为应该是明天发布”*。
- **Torch.compile 中的开销**：进入 **torch.compile** 区域存在明显的开销，估计在 **数百微秒** 级别。
  
  - 此外，这种开销阻碍了跨 graph break 的 **fusion 机会**。
- **文档中的 SGD Fused**：讨论强调了最新文档中的更新，确认 **SGD** 现在整合了 fused 操作。
  
  - 一位成员鼓励其他人提交 PR 以进一步更新文档，表明了在改进方面的协作努力。

 

**提到的链接**：[Pytorch | Anaconda.org](https://anaconda.org/pytorch/pytorch)：未找到描述

 

---

### **GPU MODE ▷ #**[**algorithms**](https://discord.com/channels/1189498204333543425/1189861061151690822/1296186178571206658) (13 条消息🔥):

> - `untitled01.ipynb`
> - `Twitter 上的 _xjdr`
> - `Flash Attention 技术`
> - `FlashInfer 项目`

- **对新技术的困惑**：一些成员对 `@untitled01.ipynb` 和 `@_xjdr` 发布的新技术表示困惑，强调缺乏清晰的数学或代码解释。
  
  - 一位成员指出，*“他们不愿写报告，反而写了诗”*，表达了对展示风格的沮丧。
- **观察到 AI Influencer Meme**：一位成员指出 `@untitled01.ipynb` 似乎体现了 AI influencer 的 meme 特质，在讨论 AGI 话题时使用了大量表情符号。
  
  - 另一位成员认为，尽管存在担忧，`@_xjdr` 似乎是该领域真正的贡献者。
- **对解释清晰度的担忧**：围绕 AI 中概念可解释性的必要性展开了讨论，一位成员表示：*“如果你不能以孩子都能理解的方式解释任何事情，那你可能理解得还不够深。”*
  
  - 共识是，那些不愿解释的人应该对他们的发现保密，以免浪费他人的时间。
- **分享 Flash Attention 见解**：一位成员分享了一份 [编写良好的 Flash Attention 大纲](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)，强调了其清晰度。
  
  - 他们对作者在 **FlashInfer** 上的工作表示赞赏，并引用了 CUDA 中一些很酷的技术。

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1296381927506317345) (41 messages🔥):

> - `Using Rusticl with v3d driver`
> - `Colab vs Kaggle for GPU access`
> - `CUDA programming in Colab`
> - `Math vs Engineering in GPU work`
> - `Optimizing algorithms for parallel processing`

- **Rusticl 和 v3d 驱动可以运行 OpenCL**：有人建议在 **Raspberry Pi 5** 上使用 **Rusticl** 配合 **v3d 驱动**来运行 OpenCL。
- **Colab 很有用但有局限性**：虽然推荐使用 **Colab** 获取 GPU 访问权限，但根据用户经验，它在处理较长时间的工作负载（15 小时以上）时往往会崩溃。
  
  - 另一位用户还提到，**Kaggle** 提供了 K80 和 P100 作为替代方案。
- **在 Colab 中为 PMPP 项目编写 CUDA**：用户确认可以在 **Colab** 中编写 **CUDA**，且一张 Turing 显卡足以应对 **PMPP 项目**。
  
  - 虽然由于 `%%writefile` 命令，Colab 中的 CUDA 代码可能缺乏语法高亮，但它能成功将文件保存在临时磁盘上。
- **GPU 数学中的工程重点**：讨论强调，虽然有关于在并行处理器上扩展算法的理论，但大多数工程工作都围绕分析硬件能力展开。
  
  - 关于 **Amdahl's Law** 和 **Gustafson's Law** 的证明被认为是计算扩展相关数学的一部分。
- **算法优化研究**：研究算法的理论并行化与为了 GPU 效率在数学上优化模型之间存在区别。
  
  - 这两个领域都在积极研究中，特别是考虑到未来对量子计算的影响。

 

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1296341257596964905) (7 messages):

> - `Windows CI Build Issues`
> - `CUDA Versions and Compatibility`
> - `HIP transformation for ROCm`

- **Windows CI 在 CUDA 扩展上遇到障碍**：如 [这个 GitHub action](https://github.com/pytorch/ao/actions/runs/11378304134/job/31653869715?pr=1101) 所示，尝试使用 **CUDA 11.8** 为 Windows 构建 CUDA 扩展遇到了挑战。目前的权宜之计可能包括在构建过程中暂时跳过 CUDA。
  
  - 有人指出 **CUDA 12.1 和 12.4 任务** 成功了，因为它们没有构建 CUDA 扩展，这暗示了 CI 流程中的一个限制。
- **驱动兼容性是新版本 CUDA 构建的一个问题**：一名成员提到，当前的驱动对于 **CUDA 12.x** 构建来说已经过时，影响了编译成功率。此外，有人建议取消对 **Kepler/Maxwell** 架构（任何低于 **sm_53** 的架构）的构建以增强兼容性。
  
  - 与此一致，建议在某些情况下，限制支持范围可以进一步简化构建流程。
- **ROCm wheels 显示出编译特性**：大家注意到 **ROCm wheels** 不编译 CUDA 源码，这虽然可以接受，但为了清晰起见值得注意。此外，一种策略可能涉及将 CUDA 代码转换为 HIP 以更好地支持 ROCm。
  
  - **hipify** 的想法可能有助于解决跨平台兼容性问题，并在未来最大化构建功能。

 

**提到的链接**：[Create build_wheels_windows.yml · pytorch/ao@612e9f7](https://github.com/pytorch/ao/actions/runs/11378304134/job/31653869715?pr=1101)：PyTorch 原生量化和稀疏化，用于训练和推理 - Create build_wheels_windows.yml · pytorch/ao@612e9f7

 

---

### **GPU MODE ▷ #**[**triton-puzzles**](https://discord.com/channels/1189498204333543425/1219683012707487794/1296236283676463146) (1 messages):

> - `Triton Puzzles Errors`
> - `GitHub Issues`

- **在 Google Colab 运行 Triton Puzzles 时遇到错误**：一名成员报告在 Google Colab 上研究 **Triton Puzzles** 时遇到错误，并引用了一个特定的 [GitHub issue](https://github.com/srush/Triton-Puzzles/issues/24)。
  
  - 他们指出在遇到问题之前没有更改任何代码，这表明可能是一个普遍问题。
- **在 GitHub Issue 上寻求帮助**：该成员询问是否有人遇到过同样的问题，寻求社区其他人的支持。
  
  - 这表明该问题可能不是孤立的，可能会影响多个使用 Triton Puzzles 的用户。

 

**提到的链接**：[Issues · srush/Triton-Puzzles](https://github.com/srush/Triton-Puzzles/issues/24)：学习 Triton 的谜题。通过在 GitHub 上创建账号为 srush/Triton-Puzzles 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1296231524085141584) (8 条消息🔥):

> - `训练中的 Loss 改进`
> - `Weight Decay 与 Optimizer 更新`
> - `C/C++ 中的 Diffusion 项目`

- **移除变量后 Loss 增加**：在移除未使用的变量后，训练迭代中的 Loss 从约 **7** 增加到了 **10**，凸显了模型性能中出乎意料的行为。
  
  - 通过 [Diffchecker](https://www.diffchecker.com/BDcWuLSY/) 分享了一个文件对比，以便进一步检查。
- **理解 Weight Decay 依赖关系**：有人对用于应用 Weight Decay 的 Tensor 索引中的依赖关系提出了担忧，建议删除 Tensor 需要更新 Optimizer。
  
  - 对之前被误解的初始化过程进行了澄清。
- **探索类似于 Llama2.c 的 Diffusion 项目**：成员们讨论了类似于 **llama2.c** 的潜在项目，特别是针对 Diffusion 应用，并询问是否需要优化的推理流水线或训练支持。
  
  - 一个相关的建议包括 GitHub 项目 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)，该项目专注于纯 C/C++ 实现的 Stable Diffusion 和 Flux。

 

**提到的链接**：[GitHub - leejet/stable-diffusion.cpp: Stable Diffusion and Flux in pure C/C++](https://github.com/leejet/stable-diffusion.cpp)：纯 C/C++ 实现的 Stable Diffusion 和 Flux。可以通过在 GitHub 上创建账户来为 leejet/stable-diffusion.cpp 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1296218422669213780) (10 条消息🔥):

> - `Cyberpunk 2077 基准测试`
> - `RCCL 改进`
> - `Flash Attention 测试结果`

- **关于 Cyberpunk 2077 基准测试的初步检查**：一位成员询问使用该系统进行 [Cyberpunk 2077 基准测试](https://link.to.cyberpunk) 是否可行，并澄清这是为了研究和性能测试。
  
  - 另一位成员回应说，如果将其重写为 **Triton Kernel**，它可能会奏效。
- **配备 MI250X 的集群潜力**：讨论强调，运行 **MI250X** 的集群将非常适合进行性能测试。
  
  - 这种观点以愉快的语气表达，显示出对硬件能力的狂热。
- **对集群上的游戏性能感到好奇**：一位成员对在集群上运行游戏以获取性能见解的可能性表示好奇。
  
  - 他们将其作为一个问题提出，反映了对利用集群执行非传统任务的广泛兴趣。
- **Flash Attention 测试基准讨论**：一位成员提到通过 Flash Attention 测试需要 **180 秒**，建议在仓库中添加 `benchmarks.md` 以建立基准。
  
  - 该提案旨在记录性能指标以供未来参考。
- **探索 RCCL 改进**：一位成员建议探索寻找集群资源的可能性，以增强 **RCCL** 的贡献。
  
  - 他们指出 **RCCL** 可以从重大增强中受益，表明了协作改进的愿望。

 

---

### **GPU MODE ▷ #**[**bitnet**](https://discord.com/channels/1189498204333543425/1240586843292958790/) (1 条消息):

marksaroufim: [https://github.com/microsoft/bitnet](https://github.com/microsoft/bitnet)

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1296188525397414030) (89 条消息🔥🔥):

> - `LM Studio Configuration`
> - `AI Model Performance`
> - `ROCm Implementation`
> - `Token Generation Speed`
> - `Riddles Testing AI Models`

- **LM Studio 配置探索**：用户讨论了在 LM Studio 中启用功能，其中一项确认指出 ROCm 已包含在 **0.3.4** 版本中，可通过 Developer 选项卡访问。
  
  - 另一位用户更新了版本并展示了性能提升，达到了 **32.82 tok/sec**。
- **Nvidia 模型表现优于竞争对手**：成员们注意到 **Nvidia 模型** 在笔记本电脑上的表现明显优于 **LLM 3.1** 等其他模型，引发了对其能力的关注。
  
  - 对不同模型的测试引发了对比，使用 **Nemotron 70b** 等模型取得了成功。
- **ROCm 与 GPU 兼容性**：确认 **ROCm** 仅适用于 **HIPSDK** 支持的特定 AMD 显卡，例如 **6800XT+**。
  
  - 用户表达了兴趣并寻求使用 ROCm 的指导，特别是针对 **7900 XT** 等显卡。
- **令人印象深刻的 Token 生成速率**：据报告，**70B Q8 模型** 的 Token 生成速率观察值为 **5-7 tok/s**，反映出与 **ChatGPT** 类似的竞争速度。
  
  - 另一位用户报告在使用特定模型配置时达到了 **32.82 tok/sec**，表明了性能的差异性。
- **谜题作为 AI 测试场**：几位用户讨论了不同 AI 模型在解决谜题方面的有效性，指出了具体的成功和失败案例。
  
  - **Nemotron 模型** 因其能力而受到关注，在这种非正式测试中表现优于 **Gemini** 等模型。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1296207420086947871) (6 条消息):

> - `70b models hardware`
> - `Llama 3.1 performance`
> - `Magnum model performance`
> - `HPE DL380 Gen9 setup`
> - `Cooling and noise concerns`

- **关于 70b 模型硬件设置的咨询**：*Kimmyg* 正在寻求运行 **70b 模型** 的见解，特别是询问量化方法和兼容硬件，尤其是具有 **2x/4x PCIe x16/x8** 的主板。
  
  - 回复建议了多种设置和性能指标，说明了对特定配置的需求。
- **Llama 3.1 展示了惊人的速度**：*一位成员报告*在 **10k 上下文长度**下，使用 **7900XTX** GPU 运行 **Llama 3.1** 达到了 **66 tokens/sec**。
  
  - 这突显了特定硬件配置对于大型模型的效率。
- **Magnum 模型在旧硬件上表现吃力**：*另一位参与者*指出，在由 **4 个 P40 GPU** 组成的陈旧硬件上运行 **Magnum 72b** 仅有 **5 tokens/sec**，暗示了模型性能中的过时问题。
  
  - 尽管速度缓慢，但有人评论说这些模型表现出了智能，展示了性能与复杂性之间的权衡。
- **探索 HPE DL380 Gen9 设置**：*Jedd1* 在配备 **两个 P40 GPU** (48GB VRAM) 的 **HPE DL380 Gen9** 上实验了中等尺寸的量化模型，指出它提供了合理的性能，但噪音可能有点大。
  
  - *Oldtimer8430* 建议将设备放置在不同的位置以减轻噪音，同时保持远程访问。
- **冷却系统导致噪音：一种普遍体验**：*Wildcat_aurora* 分享了关于冷却问题的看法，称他们的硬件在重负载下听起来像 **无人机起飞**，但只会持续很短时间。
  
  - 这一交流突显了在高性设置中管理噪音水平的共同挑战。

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1296193231129673728) (52 messages🔥):

> - `Glif 和 Wojnak 生成器`
> - `语音模式下的 AI 中断`
> - `O1 模型`
> - `Wispr Flow 应用程序`

- **Glif 和 Wojnak 生成器表现出色**：成员们赞扬了 **Glif 和 Wojnak 生成器**，声称这些工具只需极少的输入就能产生出色的结果，称其为**金子**。
  
  - 讨论经常将这些工具与其他工具进行比较，暗示它们可以生成**链接 AI 工具的工作流**来创建“应用”。
- **高级语音模式（Advanced Voice Mode）面临挑战**：一位成员批评了**高级语音模式**，称其无法理解中断且回答含糊。
  
  - 他们对 AI 频繁表示“*我的准则阻止我讨论那个*”且不理会他们的询问感到沮丧。
- **O1 模型受到审视**：对话集中在 **O1 preview 模型**表现不佳的看法上，成员们注意到它响应提示词的时间太长。
  
  - 相比之下，**O1-mini** 因其更快的响应速度和有效性被强调为“真正的英雄”。
- **Wispr Flow 受到关注**：成员们讨论了 **Wispr Flow**，这是一款帮助用户在所有计算机平台上更快速、更准确地写作的应用程序。
  
  - 强调了虽然它目前支持 macOS，但提到的一个开源应用可以满足 **Linux、Mac 和 Windows** 用户的需求。

 

**提到的链接**：[Wispr Flow | 轻松语音听写](https://flowvoice.ai/d)：Flow 通过无缝语音听写让写作变得快速清晰。它是用语音输入最快、最智能的方式。

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1296458899825885225) (23 messages🔥):

> - `ChatGPT Windows 版`
> - `桌面端语音功能`
> - `隐私担忧`
> - `微调国际象棋机器人`

- **对 ChatGPT Windows 版发布的兴奋**：成员们对 [ChatGPT for Windows](https://openai.com/chatgpt/download/) 的发布表示兴奋，并询问了关于用户访问权限的问题。
  
  - 有人澄清说，Windows 应用目前仅对 Plus、Team、Enterprise 和 Edu 用户开放。
- **新桌面应用中的语音功能**：一位成员询问新的 Windows 版本是否像 Android 应用一样支持语音功能，但由于还没人下载，目前尚不确定。
  
  - 有人担心如果最初只有 macOS 用户获得语音功能，会缺乏公平性。
- **关于屏幕共享中隐私和 PII 的辩论**：引发了关于使用该应用的屏幕交互功能时分享敏感信息的讨论，特别是关于**个人身份信息 (PII)**。
  
  - 成员们对如何限制模型可见内容以及分享个人数据的后果表示担忧。
- **对应用隐私的怀疑**：一位成员因担心信息控制以及与 AI 分享的数据去向而对下载该应用犹豫不决。
  
  - 这引发了关于应用如何与用户屏幕交互以及潜在的意外分享的讨论。
- **与 Google 数据实践的比较**：在隐私讨论中，一位成员指出 **Google** 已经收集了大量的用户数据（包括语音采集），并做了一个幽默的对比。
  
  - 这一评论反映了在新 AI 技术背景下对数据隐私更广泛的担忧。

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1296311724785274923) (3 messages):

> - `CustomGPT 来源引用`
> - `CustomGPT 的提示词技术`

- **CustomGPT 在来源引用方面遇到困难**：一位成员询问为什么 **CustomGPT** 尽管尝试了多次，却从未引用文档中的来源。
  
  - “*是否有特定的提示词能让它生效？*”是用户中产生共鸣的核心问题。
- **寻求有效的提示策略**：讨论了如何正确地提示 **CustomGPT** 以确保包含来源引用。
  
  - 成员们分享了他们的挫败感，表示提示词的清晰度对于获得理想的回复至关重要。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1296311724785274923) (3 messages):

> - `citing sources` (引用来源)
> - `customGPT functionality` (customGPT 功能)

- **CustomGPT 在引用来源方面存在困难**：一位成员提出了关于 **customGPT** 不从文档中引用来源的担忧，质疑为什么缺少这一功能。
  
  - 他们询问是否有人知道如何正确编写 prompt 以确保生成正确的引用。
- **探索 customGPT 的 prompting 技巧**：另一位成员建议尝试不同的 prompting 技巧，以鼓励 **customGPT** 包含引用。
  
  - 他们建议尝试清晰且直接的引用请求，并附带特定的文档参考。

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1296226289438429267) (70 messages🔥🔥):

> - `Inference Providers for Chat Assistants` (聊天助手的推理提供商)
> - `NotebookLM Updates` (NotebookLM 更新)
> - `MotherDuck SQL Integration with LLMs` (MotherDuck SQL 与 LLM 的集成)
> - `OpenAI's Windows Desktop App Release` (OpenAI 发布 Windows 桌面应用)
> - `Community Engagement in Data Labeling` (数据标注中的社区参与)

- **寻找用于 Chat Completions 的推理提供商**：一位成员正在寻找允许使用前缀 (prefixes) 执行聊天助手补全的推理提供商，以塑造助手的回复，类似于 Anthropic 的功能。
  
  - 他们对不同模型间该功能的可靠性表示担忧，暗示需要提供商提供更多明确信息。
- **NotebookLM 宣布自定义音频指令**：NotebookLM 用户现在可以通过在生成音频前提供特定指令来定制音频概览 (audio overviews)，从而提升用户体验。
  
  - 随着超过 80,000 家组织使用 NotebookLM，他们还宣布了通过 Google Workspace 提供的全新 Business 版本，并去掉了产品的“Experimental”标签。
- **来自 MotherDuck 的 SQL LLM 集成**：MotherDuck 引入了一个新的 SQL 函数 **prompt()**，允许用户直接在查询中集成小型语言模型，用于数据生成、摘要和提取。
  
  - 该函数旨在简化无需独立基础设施的 LLM 交互，展示了显著的成本和性能改进。
- **OpenAI 发布 Windows 桌面应用**：OpenAI 为 Plus、Enterprise、Team 和 Edu 用户发布了 ChatGPT Windows 桌面应用的早期版本，通过 Alt + Space 实现更快速的访问。
  
  - 在相关公告中， Claude 的移动应用进行了重大更新，包括项目管理和自定义指令的新功能。
- **Pixmo 数据标注中的社区参与**：一位成员强调了参与 Pixmo 数据标注的社区的热情参与，这导致了 Reddit 上相关梗 (memes) 的产生和讨论。
  
  - 他们将注意力引向了私有的 Reddit 社区，成员可以在那里跟进研究信息并参与有关数据标注的讨论。

**提到的链接**：

- [Something went wrong](https://events.zoom.us/ej/AqLi3dmNZSAXddMiqJkHlHTWkEjpoQZ7CEHtgg-bgBXf)：未找到描述
- [Introducing the prompt() Function: Use the Power of LLMs with SQL!](https://motherduck.com/blog/sql-llm-prompt-function-gpt-models/)：我们通过在 SQL 中支持小型语言模型（和 LLM）让您的数据库更智能 | 阅读时间：6 分钟
- [Tweet from Logan Kilpatrick (@OfficialLoganK)](https://x.com/OfficialLoganK/status/1846951237578903753)：NotebookLM 粉丝的好消息：音频概览现在可以在生成前进行自定义和引导 + 我们正在推出 NotebookLM 企业版！🎙️ https://notebooklm.google/
- [Homebrew Research – Homebrew](https://homebrew.ltd/)：Homebrew 是一家在 Local AI、Small Language Models 和多模态 (Multi-modality) 领域工作的 AI 研发实验室。
- [After selling Drift, ex-HubSpot exec launches AI for customer success managers | TechCrunch](https://techcrunch.com/2024/10/16/after-selling-drift-ex-hubspot-exec-launches-ai-for-customer-success-managers/)：Elias Torres 为 17 岁从尼加拉瓜移民到美国且当时不懂英语的人赢得了巨大成就。他曾担任工程副总裁。
- [Open LLM Leaderboard Model Comparator - a Hugging Face Space by open-llm-leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/comparator)：未找到描述
- [Tweet from Raiza Martin (@raiza_abubakar)](https://x.com/raiza_abubakar/status/1846944566689353838?s=46&t=jDrfS5vZD4MFwckU5E8f5Q)：今天的 NotebookLM 新更新：🎧 给主持人传个话——你现在可以在音频概览中点击“自定义”来提供额外指令，例如专注于特定主题、来源...
- [Tweet from Alex Volkov (Thursd/AI) (@altryne)](https://x.com/altryne/status/1846977617704140893?s=46)：重磅——你现在可以通过 @OpenAI 的 chat completions API 发送和接收音频 👏 与 RealTime 音频不同，这非常适合不需要实时性但需要多模态的应用...

- [Daniel Han (@danielhanchen) 的推文](https://x.com/danielhanchen/status/1846235913443262891)：修复了一个导致在大梯度累积（gradient accumulation）大小时所有训练损失发散的 bug。1. 最初由 @bnjmn_marie 报告，GA 在数学上应该等同于全批次（full batch）训练...
- [Elias Torres (@eliast) 的推文](https://x.com/eliast/status/1846652872060002732?s=46)：就是今天。介绍我的新公司 @Agency —— 由 @Sequoia 和 Hubspot Ventures 支持；我的朋友和导师 @BHalligan 担任董事。很高兴能与 @MTemkin @TechCrunch 谈论...
- [Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1846950770736091509?s=46)：推出内部知识搜索（Internal Knowledge Search，我们最受期待的企业版功能）！这是第一次，你可以通过一个产品同时搜索组织的文件和网络...
- [Jacob Matson (@matsonj) 的推文](https://x.com/matsonj/status/1847007726335152284?s=46)：你在开玩笑吗？看看这个：引用 MotherDuck (@motherduck) 我们在 SQL 中加入了一个 LLM，并向你展示了 MotherDuck 数据仓库中 SLMs（small language models）的力量。https://mothe...
- [AI 模型在质量、性能、价格方面的比较 | Artificial Analysis](https://artificialanalysis.ai/models)：在关键性能指标（包括质量、价格、输出速度、延迟、上下文窗口等）上对 AI 模型进行比较和分析。
- [Simon Willison (@simonw) 的推文](https://x.com/simonw/status/1846987810706018435?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：我刚尝试了这个 OpenAI 示例 - 我收到了一个 264KB 的 base64 编码 WAV 文件（作为 JSON 字符串），时长 5 秒，消耗了 110 个音频输出 tokens - 这些定价为每百万 200 美元，所以...
- [OpenAI (@OpenAI) 的推文](https://x.com/openai/status/1846957067204166113?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：今天，ChatGPT Plus、Enterprise、Team 和 Edu 用户可以开始测试 Windows 桌面应用的早期版本。通过 Alt + Space 快捷键在你的 PC 上更快地访问 ChatGPT。https://ope...
- [clem 🤗 (@ClementDelangue) 的推文](https://x.com/ClementDelangue/status/1847009885852258650)：👀👀👀
- [Clémentine Fourrier 🍊 (@clefourrier) 的推文](https://x.com/clefourrier/status/1846907589365297640)：你是否一直想详细比较最佳排行榜模型的性能？看看我们的新工具！🔍 https://huggingface.co/spaces/open-llm-leaderboard/comparator 它并排比较...
- [Nathan Cooper (@ncooper57) 的推文](https://x.com/ncooper57/status/1846612127911760261?s=46)：作为 @stabilityai 的首席研究员，我使用了大量合成数据来训练 LLMs 和 VLMs。这是提升模型性能最被低估的方式。现在在 @answerdotai 我一直在工作...
- [Reddit - 深入探索一切](https://www.reddit.com/r/MattDeitkeStudies/)：未找到描述
- [Requests | OpenRouter](https://openrouter.ai/docs/requests)：处理传入和传出的请求
- [登录 | Zoom](https://events.zoom.us/ej/AqLi3dmNZSAXddMiqJkHlHTWkEjpoQZ7CEHtgg-bgBXf5FUjyxMS~A9Dc0qDMYy1XxnQw-wyMyInvma-5aGQLC-k7gh3UVsnS8AZ3om-GLGN6Xou-kOQgIU_--FVQcTlqmx0hsKmS-anoiyH0d5XMk4BvO-JFI)：登录你的 Zoom 账户以加入会议、更新个人资料、更改设置等！
- [OpenAI (@OpenAI) 的推文](https://x.com/openai/status/1846957067204166113?)：今天，ChatGPT Plus、Enterprise、Team 和 Edu 用户可以开始测试 Windows 桌面应用的早期版本。通过 Alt + Space 快捷键在你的 PC 上更快地访问 ChatGPT。https://ope...
- [Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1846943479332802571?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：我们刚刚对 Claude 移动应用进行了重大的设计革新。现在使用起来感觉非常流畅。你可以在应用内创建项目、添加自定义指令，并在项目中进行聊天，一切都在应用内完成...

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1296193827346059274) (4 messages):

> - `Yi-Lightning`
> - `Chatbot Arena Rankings`
> - `GLM-4-Plus Surge`
> - `Chinese LLMs Competition`

- **Yi-Lightning 排名攀升**：来自 [Chatbot Arena](https://lmarena.ai) 的重大新闻，由 @01AI_YI 开发的 **Yi-Lightning** 已获得超过 **1.3 万次社区投票**，目前位列**总榜第 6 名**。
  
  - 它已追平 **Grok-2** 等强力模型，在数学 (Math)、困难提示词 (Hard Prompts) 和编程 (Coding) 方面表现出色。
- **GLM-4-Plus 崭露头角**：来自智谱 AI 的 **GLM-4-Plus** 也进入了**前 10 名**，展示了中国 LLMs 的快速崛起。
  
  - 这一进展标志着这些模型之间的竞争日益激烈。

 

**提到的链接**：[来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1846245604890116457)：来自 Chatbot Arena 的重大新闻！@01AI_YI 的最新模型 Yi-Lightning 已在 Arena 中经过广泛测试，收集了超过 1.3 万次社区投票！Yi-Lightning 已攀升至总榜第 6 名……

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1296560727812735006) (8 messages🔥):

> - `Inference Providers for Chat Models`
> - `Special Tokens in Chat Models`
> - `Pre-filling Chatbot Responses`
> - `Support Experience with Model Providers`
> - `Interconnects Discord vs. Latent Space Discord`

- **关于推理服务商能力的咨询**：一位成员询问了支持热门开放权重模型 (open-weight models) 聊天助手补全功能的推理服务商，特别是寻找类似于 [Anthropic 的 pre-filling 功能](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response) 的能力。
  
  - *“我不确定我是否可以信任底层发生的事情”* 反映了对这类服务商可靠性的担忧。
- **关于 Special Tokens 使用的讨论**：一位成员强调了他们对访问特定 Special Tokens 以构建聊天机器人交互结构的兴趣，并指出了助手响应中没有 END_OF_TURN_TOKEN 的独特格式。
  
  - 他们提供了一个示例结构，展示了如何使用各种 Token 格式化用户和助手的轮次。
- **非聊天模型的过往经验**：一位成员回忆了去年为非聊天模型处理这些 Token 的经验，指出当时这是可选的。
  
  - *“也许可以试试看他们的文档”* 建议查阅文档以澄清 Token 的使用和实现。
- **对服务商支持的赞扬**：一位成员分享了他们的积极体验，表示该服务商的支持响应迅速且非常有帮助。
  
  - 这表明了对支持团队响应速度和协助质量的良好印象。
- **关于 Discord 空间的对比见解**：一位成员评论了 Interconnects Discord 和 Latent Space Discord 之间的区别，暗示前者更加私密。
  
  - *“Interconnects Discord 就像是 Latent Space Discord 的更私密版本，哈哈”* 反映了对社区动态的轻松观察。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1296246150617104456) (51 条消息🔥):

> - `Research Experience Value` (研究经验的价值)
> - `Degree Requirements for AI Labs` (AI 实验室的学位要求)
> - `Luck and Risk in Careers` (职业生涯中的运气与风险)
> - `Community Engagement in AI Projects` (AI 项目中的社区参与)
> - `Self-Study Challenges` (自学的挑战)

- **研究经验在当下至关重要**：一位成员分享到，在攻读硕士学位之前，从本科研究过渡到非 ML 工作从长远来看对他们有所帮助，证明了**更多的研究经验是有益的**。
  
  - 他们强调这些实验室节奏很快，不仅需要聪明才智，还需要熟悉职场动态。
- **关于 AI 实验室学位的辩论**：关于在 OAI、DM 和 Anthropic 等**顶级 AI 实验室**任职是否需要硕士学位，目前存在持续的讨论。
  
  - 成员们指出，虽然证书有所帮助，但经过证明的技能和相关经验往往比正式教育更重要。
- **创造你自己的运气**：一个关于**“创造你自己的运气”**的话题出现了，暗示虽然随机性存在，但机会源于战略性的冒险。
  
  - 成员们一致认为，扩大自己的机会对于最大化获得积极结果的可能性至关重要。
- **Pixmo 的社区标签**：关于 **Pixmo** 社区的一个趣闻显示，他们的参与度非常高，甚至创建了一个专门用于标注的 Reddit 社区，其中包含迷因（memes）和讨论。
  
  - 指向这些社区的链接表明了活跃的参与度，证明了受众参与可以促进围绕项目的热烈讨论。
- **自学的挑战**：对于在没有适当指导的情况下进行自学的有效性存在担忧，一位成员表示这经常让许多学习者陷入困境。
  
  - 共识是，通过结构化学习（尤其是带有导师指导的学习）来培养技能，从长远来看更有效。

 

**提到的链接**：[Reddit - Dive into anything](https://www.reddit.com/r/MattDeitkeStudies/): 未找到描述

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1296185904167387238) (5 条消息):

> - `SnailBot Speed` (SnailBot 速度)
> - `User Dynamics` (用户动态)

- **SnailBot 加速至 8 分钟**：一位成员注意到，“哇…… 8 分钟…… 蜗牛变快了”，这暗示了 **SnailBot** 性能的提升。
  
  - 这一评论暗示了人们对机器人速度如何影响用户互动的持续期待。
- **循环往复的用户互动模式**：一位成员提到了重复的互动，称“我跳起这支舞”，暗示了用户之间熟悉的对话。
  
  - 这突显了社区内有趣的动态，反映了持续的参与。
- **对话的善变性**：一位用户对讨论的反复无常表示沮丧，称“它太善变了”，以强调聊天话题不可预测的本质。
  
  - 这捕捉到了数字对话中参与度波动的感受。
- **SnailBot 新闻公告**：SnailBot 向角色 <@&1216534966205284433> 发布了通知，表明它有新闻更新要分享。
  
  - 这表明该机器人在社区内传播信息方面发挥着持续的作用。

 

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1296227795055153204) (5 条消息):

> - `多模态 RAG 系统`
> - `LlamaIndex 与 Elastic`
> - `AI Hackathon`
> - `多租户 RAG 应用`
> - `MongoDB 混合搜索支持`

- **使用 Azure AI 构建多模态 RAG 系统**：分享了一份关于使用 [@Azure AI Search](https://t.co/RO5nQ79sqD)、Azure OpenAI 和 @ArizePhoenix 结合 LlamaIndex 创建**多模态 RAG 系统**的分步指南。
  
  - 该指南强调了通过上下文检索（contextual retrieval）来提高准确性，并提供了基准测试信息。
- **明天关于 LlamaIndex 和 Elastic 的演讲**：关注 @seldo 在即将举行的演讲中讨论如何将 **LlamaIndex** 与 Elastic 结合使用，这肯定会提供宝贵的见解。
  
  - 更多详情请点击[此处](https://t.co/tQszqtRN1Z)。
- **在班加罗尔与 Meta 共同举办的 AI Hackathon**：与 @Reskilll 和 @Meta 合作，计划于 10 月 19 日至 20 日在班加罗尔举办 **AI Hackathon**，届时将由 @ravithejads 提供指导。
  
  - 参与者可以在[此处](https://t.co/aFf31yHJba)了解有关该活动的更多信息。
- **让多租户 RAG 应用变得简单**：使用 LlamaIndex 和 Nile 轻松构建**多模态 RAG 应用**，解决为大量用户建立索引时的数据安全问题。
  
  - 在[此处](https://t.co/zRfzR5A4Us)查看全栈演示应用程序。
- **面向 LlamaIndex 的 MongoDB 混合搜索**：利用 **MongoDB** 为 LlamaIndex 提供的全新混合搜索支持（结合了向量搜索和关键词搜索）来增强你的 AI 应用。
  
  - 这种方法旨在融合两种搜索类型的优势以获得最佳结果，详情见：[链接](https://t.co/XxNNwoaW9U)。

**提到的链接**：[AI Hackathon with Meta Llama](https://t.co/aFf31yHJba)：加入我们，与热爱 AI 的行业专家一起体验令人兴奋的 30 小时。这是你见面、协作并在构建惊人作品的同时享受乐趣的机会。让我们开始创作吧...

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1296361853307322405) (46 条消息🔥):

> - `LlamaIndex.TS 中的 MultiStepQueryEngine 支持`
> - `RAG 中用于文档管理的元数据使用`
> - `vLLM 服务器问题`
> - `Faithfulness 评估时间优化`
> - `用于 Word 文档的 LlamaParse`

- **关于 LlamaIndex.TS 中 MultiStepQueryEngine 的问题**：成员们讨论了 LlamaIndex.TS 中缺乏对 **MultiStepQueryEngine** 支持的问题，并建议通过使用 LLM 手动分解任务作为变通方案。
  
  - 另一位成员提供了一种在检索系统中将**文件名**等元数据添加到 Embedding 过程的方法。
- **vLLM 服务器返回 400 Bad Request 的问题**：一位成员报告在调用 **vLLM 服务器**时收到 **400 Bad Request** 错误，这表明请求 Payload 中缺少必需的参数。
  
  - 通过排查，他们在重新运行请求之前识别并删除了 Payload 中的 **None 值**。
- **关于 Faithfulness 评估性能的担忧**：一位成员对复制 **Faithfulness 评估**时漫长的处理时间表示沮丧，有时需要一个多小时。
  
  - 讨论围绕硬件限制展开，建议从 **LlamaCPP** 切换到 **Ollama**，以在本地模型上获得潜在更快的性能。
- **LlamaParse 处理 Word 文档时的错误**：一位成员在对 Word 文档使用 **LlamaParse** 时遇到了意外结果，显示的是图像而非预期的文本数据。
  
  - 他们提供了一个演示该问题的最小化仓库链接，寻求社区的反馈。

**提到的链接**：

- [无标题](https://<YOUR_HOST>/v1/chat/completions)：无描述
- [Qdrant Vector Store - Metadata Filter - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/#qdrant-vector-store-metadata-filter)：无描述
- [GitHub - xaac-ai/llama-artifact](https://github.com/xaac-ai/llama-artifact)：通过在 GitHub 上创建账号来为 xaac-ai/llama-artifact 的开发做出贡献。
- [llama_index/llama-index-integrations/llms/llama-index-llms-vllm/.../utils.py](https://github.com/run-llama/llama_index/blob/f633e7393aaa3f36ef518429672b931b1e3bdae8/llama-index-integrations/llms/llama-index-llms-vllm/llama_index/llms/vllm/utils.py#L8C5-L9C24)：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1296571671716827206) (1 条消息):

> - `Modular 社区 Q&A`

- **加入 Modular 社区 Q&A！**：发布了关于即将在周一社区会议期间举行的 **Modular 社区 Q&A** 的提醒，鼓励成员通过提供的 [表单](https://forms.gle/MgixGyhRKcA33BS6A) 提交问题。
  
  - *请分享任何您希望团队在会议期间解决的疑问。*
- **立即提交您的问题！**：鼓励参与者通过指定表单 **提交问题**，以便团队在会议前准备回复。
  
  - *不要错过提问的机会——请在周一之前填写表单！*

 

**提到的链接**：[Modular Community Q&A](https://forms.gle/MgixGyhRKcA33BS6A)：未找到描述

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1296222188235657349) (17 条消息🔥):

> - `Mojo 与 Python stdlib`
> - `函数 Parameters 与 Arguments 的区别`
> - `使用 LLM 进行翻译`
> - `多语言文档`
> - `沉浸式翻译工具`

- **Mojo 重新实现 Python stdlib 的愿景**：一位用户询问 Mojo（旨在成为 Python 的超集）是否计划重新实现 Python stdlib 中的所有内容，并猜测该计划可能不会涵盖全部。
  
  - 作为回应，一名成员指出，虽然理论上可行，但实际上需要大量时间。
- **澄清 Mojo 中的 Parameters 和 Arguments**：讨论了将“function parameters”和“function arguments”翻译成中文时的细微差别，因为两者通常都被标记为“函数参数”。
  
  - 一位用户建议文档应更清晰地区分“parameters”，或许可以称之为“编译期参数（compile-time parameters）”以避免混淆。
- **LLM 提升翻译效率**：一位成员提到，中文社区中的许多人正转向使用 LLM 进行翻译，而不是手动翻译文档，并指出 LLM 结果的速度和可用性。
  
  - 另一位成员建议利用 prompt 来确保将“parameter”准确翻译为“编译期参数”。
- **探索沉浸式翻译以获得更好的文档体验**：一位成员介绍了“沉浸式翻译（Immersive Translate）”工具，这是一个评价很高的双语翻译网站插件，使用各种 AI 引擎进行文本翻译。
  
  - 该工具允许用户方便地翻译内容，被公认为中国最受欢迎的翻译应用。
- **为中文社区收集发现**：有人建议将与使用 LLM 相关的发现（如教程和 prompt）汇编成专门的帖子，以便中文社区更轻松地获取。
  
  - 旨在为未来寻求类似解决方案的人简化信息共享。

**提到的链接**：

- [接入兼容 OpenAI API 接口的 AI 模型 | 沉浸式翻译](https://immersivetranslate.com/zh-Hans/docs/services/ai/#system-promptpromptmultiple-promptsubtitle-prompt),): 原作者：萧萧然
- [双语网页翻译扩展_PDF文档翻译工具 | 沉浸式翻译](https://immersivetranslate.com/): 沉浸式翻译是一款免费使用的网站翻译扩展，为您提供在线双语网页翻译。它可用于将网站翻译成英文或其他语言，支持文档翻译...
- [Parameterization: compile-time metaprogramming | Modular Docs](https://docs.modular.com/mojo/manual/parameters/): 参数化与编译期元编程介绍。

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1296213268800471040) (10 条消息🔥):

> - `MAX 的 Mojo 版本`
> - `Jakub 的 Python API 工作`
> - `Driver 演示`

- **探索 MAX 的 Mojo 版本**：一位成员询问是否有计划创建 **Mojo** 版本，因为目前的实现是 **Python**。
  
  - 另一位成员提到，由于 Mojo 相对较新，为 Mojo 适配 **MAX** 还需要时间。
- **Jakub 在 Python API 上的工作**：有人询问来自 **Modular** 的 **Jakub** 正在进行的关于 **MAX** 的 **Python API** 工作。
  
  - 成员们讨论了细节，并请求提供链接以了解更多关于他的贡献。
- **Driver 演示反馈**：Driver 的演示展示了实现模型是多么容易，尽管它尚未完全发布，仅在 **nightly builds** 中部分可用。
  
  - 一位成员对演示表示赞赏，称他们反复听了多次以更好地理解。

 

**提到的链接**：[max/examples/graph-api/pipelines/llama3 at main · modularml/max](https://github.com/modularml/max/tree/main/examples/graph-api/pipelines/llama3)：一系列示例程序、笔记本和工具，展示了 MAX 平台的强大功能 - modularml/max

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1296252544854786049) (24 条消息🔥):

> - `Stable Diffusion Prompt Suggestions` (Stable Diffusion 提示词建议)
> - `Fooocus Model Compatibility` (Fooocus 模型兼容性)
> - `Face Swap Features in Automatic1111` (Automatic1111 中的换脸功能)
> - `Image Quality Concerns` (图像质量问题)
> - `AI Hackathon Announcement` (AI 黑客松公告)

- **Stable Diffusion 需要提示词方面的帮助**：一位成员寻求关于为**cube**（立方体）创建阴影效果的提示词帮助，要求不显示上方的光源，强调了场景中光照的重要性。
  
  - 多位成员讨论了提示词效果的不同体验，展示了对更具针对性建议的需求。
- **Fooocus 模型与兼容性**：在询问模型兼容性时，一位成员了解到 **Fooocus** 主要使用 **SDXL**，而另一位成员确认它也可以与 **pony models** 配合使用。
  
  - 这次交流凸显了社区对确保兼容性以提升用户体验的关注。
- **换脸功能解决方案**：一位成员询问如何在 **Automatic1111** 中复制 **Fooocus** 的 **faceswap** 功能，另一位成员建议使用 **Reactor extension** 或 **IP-Adapter face**。
  
  - 这展示了用户之间为增强不同平台工具功能而进行的协作努力。
- **对图像质量的担忧**：一位成员反映，尽管使用了 30 steps 和多个 **LORA** 模型，生成的图像仍缺乏细节，并寻求潜在的解决方案。
  
  - 这引发了关于在 **Stable Diffusion** 处理过程中可能影响图像质量的各种因素的讨论。
- **创新项目 AI 黑客松**：一项公告重点介绍了 **Gen AI Agents** 黑客松，邀请团队和个人通过协作创建增强人类潜能的 AI 解决方案。
  
  - 鼓励参与者在开发旨在优化日常任务的安全可靠的 AI 系统时，考虑伦理影响。

 

**提到的链接**：[Vertical Specific AI Agents Hackathon · Luma](https://lu.ma/ke0rwi8n)：Gen AI Agents CreatorsCorner，与 aixplain, Sambanova Systems, Prem, Marly, Senso, Mistral, coval, heygen, fiberplane, exa 等合作……

 

---

### **Torchtune ▷ #**[**announcements**](https://discord.com/channels/1216353675241590815/1216353675241590818/1296526376610037770) (1 条消息):

> - `PyTorch 2.5.0 Release` (PyTorch 2.5.0 发布)
> - `FlexAttention Feature` (FlexAttention 功能)
> - `Per-Layer Compile` (逐层编译)
> - `Contributing to Torchtune` (为 Torchtune 做出贡献)

- **PyTorch 2.5.0 正式发布！**：备受期待的 **PyTorch 2.5.0** 已正式发布，其中包括 [FlexAttention](https://github.com/pytorch/pytorch/releases/tag/v2.5.0) 和 **per-layer compile** 等新功能。
  
  - 鼓励用户升级本地的 **torch** 安装以利用最新功能。
- **Torchtune 贡献追踪器**：对于那些希望为 **Torchtune** 做出贡献的人，已经建立了一个追踪器，用于清理代码库以实现对 **PyTorch 2.5.0** 的全面支持，详见[此处](https://github.com/pytorch/torchtune/issues/1861)。
  
  - 该计划旨在确保库与 PyTorch 的最新更新和改进保持一致。

 

**提到的链接**：[Issues · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1861.)：PyTorch 原生微调库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1296477139759267911) (12 条消息🔥):

> - `Qwen 2.5 模型集成`
> - `Tokenizer 修改`
> - `微调指南`
> - `特殊 Token 使用`

- **Torchtune 中的 Qwen 2.5 模型集成**：[Qwen 团队已发布 Qwen 2.5](https://github.com/pytorch/torchtune/issues/1624)，包括多个被请求集成到 Torchtune 的模型，但相关更新仍在进行中。
  
  - 成员们正在协作添加该模型，并欢迎其他对集成过程感兴趣的人贡献力量。
- **Tokenizer 修改指南**：用户讨论了修改 Tokenizer 以支持新的 Qwen 2.5 模型，特别是参考 [Tokenizer 文件](https://github.com/pytorch/torchtune/blob/main/torchtune/models/qwen2/_tokenizer.py) 进行必要的编辑。
  
  - 建议查看 [Hugging Face 上的配置文件](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/tokenizer_config.json)，了解所需修改的详细信息。
- **微调问题与资源请求**：一位用户正在寻求运行 Qwen 2.5 微调的详细指导，并承认自己在此过程中是新手。
  
  - 反馈包括提供帮助以及指向正确文件和依赖项的指引，强调了社区支持。
- **关于特殊 Token 的讨论**：有人询问在 Tokenizer 的何处实现新的特殊 Token，特别是关于它们在 'ipython' 等消息角色（message roles）中的用法。
  
  - 一位成员要求明确添加这些 Token 的位置，以确保彻底集成到现有的模型框架中。

**提到的链接**：

- [Qwen/Qwen2.5-14B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)：无描述
- [Qwen 2.5 is here, Request for adding a model · Issue #1624 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1624)：Qwen 团队已发布 Qwen 2.5 base、coder、math 模型。它们看起来非常有前景。请求团队在 Torchtune 中添加此模型。
- [Qwen 2.5 is here, Request for adding a model · Issue #1624 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1624#issuecomment-2361869824)：Qwen 团队已发布 Qwen 2.5 base、coder、math 模型。它们看起来非常有前景。请求团队在 Torchtune 中添加此模型。

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1296277177850925100) (10 条消息🔥):

> - `Torchtune 论文`
> - `PhD 实习愿望`
> - `实现协作`
> - `PPO 工作进展`
> - `RFCs 与分支`

- **Torchtune 可能的论文**：团队幽默地指出，Torchtune 的论文可能会在约 **10 年**后他们有时间时撰写。
  
  - 针对时间表，“当我们坐下来写一篇的时候”是比较乐观的回应。
- **对新研究的热情**：一位用户分享了 [arXiv](https://arxiv.org/pdf/2410.10630) 上的一篇有趣论文，引发了关注和兴奋。
  
  - 另一位成员表达了希望通过 **PhD 实习**来参与论文中讨论的项目。
- **实现方面的协作**：讨论了协作将 arXiv 论文中的想法实现到 **Torchtune** 中。
  
  - “来帮我在 Torchtune 中实现它吧 :)” 表达了共同开展项目的渴望。
- **PPO 的持续工作**：一位成员表示，在开始新任务之前，他们需要完成 **PPO** 方面的工作。
  
  - “我得先落地几个 RFC 并完成我的 PPO 工作” 反映了团队目前的优先级。
- **开发的起点**：团队承认项目必须从某个地方开始。
  
  - “但我们总得从某个地方开始” 再次强调了进步的情绪，突出了积极的参与。

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1296219929745555469) (12 messages🔥):

> - `OpenInterpreter 任务问题`
> - `关闭应用时出现 Kernel Panic`
> - `在工作流中集成 O1`
> - `解压 Tar 文件`
> - `OpenInterpreter GitHub 资源`

- **OpenInterpreter 任务完成问题**：用户在使用 **OpenInterpreter** 完成任务时遇到问题，反复遇到脚本声称已执行操作但实际未发生任何操作的情况。
  
  - 有人建议在专门频道发布详细信息（如版本和模型），以便进行故障排除。
- **关闭应用时出现 Kernel Panic**：一名成员报告在尝试关闭 OpenInterpreter 应用时遇到了 **kernel panic**。
  
  - 建议在相应频道寻求帮助以解决此问题。
- **将 O1 集成到日常工作流**：一位用户寻求关于如何将 **O1** 集成到日常任务中的建议，并对最近完成的一个 **NLP 项目**表示兴奋。
  
  - 讨论指出，集成很大程度上取决于用户的具体工作流和自动化需求。
- **使用 OI 解压 tar 文件**：一名成员幽默地提到了解压 **tar 文件**的困扰，以及 OpenInterpreter 如何通过自动化该过程提供帮助。
  
  - 另一位用户表示，在经历最初的困惑后，终于理解了如何正确运行解压命令，感到如释重负。
- **分享 OpenInterpreter GitHub 资源**：分享了 **OpenInterpreter** GitHub [仓库](https://github.com/OpenInterpreter/open-interpreter/tree/main/scripts)的链接，展示了可供用户使用的脚本。
  
  - 该链接旨在支持那些有兴趣利用现有脚本进行 **Natural Language** 处理任务的用户。

 

**提到的链接**：[open-interpreter/scripts at main · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/tree/main/scripts)：计算机的自然语言接口。通过在 GitHub 上创建账号，为 OpenInterpreter/open-interpreter 的开发做出贡献。

 

---

### **OpenInterpreter ▷ #**[**O1**](https://discord.com/channels/1146610656779440188/1194880263122075688/1296327996906672274) (4 messages):

> - `Android 二维码问题`
> - `微型 Android 手机技巧`
> - `iOS 与 Android 性能对比`

- **在 Android 上扫描二维码遇到困难**：一名用户遇到 **Android 客户端**在扫描二维码后无响应的问题，而 **iOS** 端运行完美。
  
  - 他们在进一步检查源码前寻求建议，并对功能缺失表示沮丧。
- **寻求微型 Android 手机兼容性技巧**：该用户获得了一部微型 Android 手机（类似于演示中使用的那款），正在寻找关于仓库中可能缺失的参数或配置的建议。
  
  - 他们感谢社区为增强兼容性提供的任何见解。
- **Android 扫描后 Shell 无响应**：用户报告在 Android 上扫描后无法激活 **shell**，而 iOS 端表现稳定。
  
  - 这突显了 Android 客户端持续存在的问题，引发了社区的求助。

 

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1296306185812840540) (6 messages):

> - `免费 LLM 集成`
> - `Open Interpreter 脚本`
> - `在 Vim 中使用 AI`

- **寻找免费 LLM 选项**：一位用户询问了可以与 Chat GPT 集成的免费 LLM，原因是 [API 成本上升](https://link.url)。另一位用户建议，如果本地模型不可行，可以考虑使用命令 `interpreter --model i` 调用 `i model`。
- **庆祝新工具**：一位用户表达了兴奋之情，评论说该领域的进步“是时候了”，引发了社区的庆祝回应。
  
  - *对吧？* ✨
- **Open Interpreter 的** `wtf` **脚本揭晓**：Mikebirdtech 介绍了 Open Interpreter 的 `wtf` 脚本，并通过 Ty 的演示展示了其在 [Tool Use](https://www.youtube.com/watch?v=Vz3cjbf4zeo) 中的实用性。
  
  - 该脚本作为一个显著功能，扩展了用户可以探索的功能。
- **Vim 中的 AI 集成**：Mikebirdtech 分享了 Jake Koenig 的见解，他演示了如何在 **Vim** 中使用 AI，教程视频见[此处](https://www.youtube.com/watch?v=Ho9yf7ks5sE)。
  
  - 这为寻求通过 AI 增强编码体验的开发者增加了工具选择。

 

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1296506311483326525) (4 messages):

> - `Multi-label classification for scientific documents`（科学文档的多标签分类）
> - `Heterogeneous graph neural networks`（异构图神经网络）
> - `In-context learning`
> - `BootstrapFewShotWithRandomSearch`
> - `Medium article on research`（关于研究的 Medium 文章）

- **创新的多标签分类方法**：一位成员分享了一种令人兴奋的科学文档多标签分类新方法，该方法基于之前在极端多标签分类的 In-context learning 方面的[研究工作](https://link.to.research)。
  
  - 他们描述了如何创建一个以红色节点为文档、蓝色节点为标签的**异构图 (Heterogeneous graph)**，并对其在有效搜索大规模语料库方面的潜力表示了极大的热情。
- **使用神经网络进行标签聚类**：该成员解释了使用**异构图神经网络 (Heterogeneous Graph neural network)** 进行标签聚类的方法，尽管他们对归因边 (imputed edges) 并不满意，并希望 In-context learning 能产生更好的结果。
  
  - 他们还提到使用 `BootstrapFewShotWithRandomSearch` 从每个聚类中挑选示例进行文档推理。
- **对 Medium 文章的兴趣**：在分享了他们的工作后，作者询问社区是否对一篇详细介绍该方法和发现的 **Medium 文章**感兴趣。
  
  - 反应非常积极，成员们对这样一篇文章表现出了强烈的热情。

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1296266853198598188) (16 messages🔥):

> - `Langtrace DSPy integration`（Langtrace 与 DSPy 的集成）
> - `DSPy prompt optimization issues`（DSPy 提示词优化问题）
> - `DSPy answer guarantees`（DSPy 答案保证）
> - `Feedback on DSPy documentation`（对 DSPy 文档的反馈）

- **Langtrace 在 DSPy 集成中表现出色**：成员们讨论了 **Langtrace** 与 **DSPy** 极具前景的集成，并强调了详细说明如何从 DSPy 流水线捕获追踪 (traces) 的[设置指南](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy)。
  
  - 设置过程包括安装 DSPy、初始化 Langtrace 的 SDK，以及创建一个类型为 **DSPy** 的项目。
- **DSPy 提示词优化未反映在 JSON 中**：一位成员报告称，在使用 **MIPROV2** 优化一个简单的分类器后，JSON 配置保留的是原始提示词而非优化后的提示词，从而引发了关于性能损失的疑问。
  
  - 随后展开了关于保存或加载配置时可能存在的 Bug 的讨论，并建议调查 JSON 文件的内容。
- **获取 DSPy 的保证答案**：一位成员询问如何确保 DSPy 从可能性列表中返回答案，对此建议使用 **Literal[]** 类型。
  
  - 这种技术可以在其应用中对有效的输出响应提供更多控制。
- **对 DSPy 文档的积极反馈**：一位用户对新的 DSPy 入门指南表示赞赏，强调其易于理解的拆解和完整的 RAG 实现对新手特别有帮助。
  
  - 建议包括增加交互式 Notebook 和一个用于动手学习的“自行尝试 (Try It Yourself)”章节。
- **对宝贵反馈的致谢**：针对 DSPy 文档的反馈，一位成员表示这些输入非常有帮助，并确认其对未来的改进很有用。
  
  - 这次对话体现了在增强 DSPy 用户学习材料方面的协作精神。

 

**提及的链接**：[DSPy - Langtrace AI Docs](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy)：未找到描述

 

---

### **DSPy ▷ #**[**colbert**](https://discord.com/channels/1161519468141355160/1250300504462856265/1296576706940899388) (1 条消息):

> - `ColbertV2 Training`
> - `Data Format Confusion`

- **ColbertV2 Training 接收 Triples 和 Queries**：正如 [GitHub 仓库](https://github.com/stanford-futuredata/ColBERT?tab=readme-ov-file#advanced-training-colbertv2-style)中所记录的，**ColbertV2** 的训练示例接收 triples、collections 和 queries。这表明其数据处理机制较为复杂，需要进一步明确。
  
  - 成员们对数据集如何与示例中看到的 **queries** 和 **collections** 的索引版本相关联表示困惑。
- **数据集格式镜像了原始 Query 示例**：打印所引用数据集的前几个字符时，其外观与讨论的 `raw_query` 格式相似。这一观察结果与 **ColbertV2** 训练过程的索引方法一致。

 

**提到的链接**：[GitHub - stanford-futuredata/ColBERT: ColBERT: state-of-the-art neural search (SIGIR'20, TACL'21, NeurIPS'21, NAACL'22, CIKM'22, ACL'23, EMNLP'23)](https://github.com/stanford-futuredata/ColBERT?tab=readme-ov-file#advanced-training-colbertv2-style)：ColBERT：最先进的神经搜索 (SIGIR'20, TACL'21, NeurIPS'21, NAACL'22, CIKM'22, ACL'23, EMNLP'23) - stanford-futuredata/ColBERT

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1296204933565255731) (8 条消息🔥):

> - `MSE and MAE in Tensors`
> - `Library Loading Fix`
> - `LLVM Load for Gates`
> - `CLOUD=1 with Multi-Device`

- **Tensor 中的 MSE 和 MAE 实现**：一位成员分享了一个 [Pull Request](https://github.com/tinygrad/tinygrad/pull/7107) 链接，该 PR 在 `tensors.py` 中实现了 **MSE** 及其相关测试。
  
  - 他们指出 **MSE** 和 **MAE** 可以简写为两行，值得添加到 tensors 中。
- **修复 Autogen 中的库加载**：有人建议修正 `autogen_stubs.sh` 中 **libc** 的加载方式，以处理 `find_library` 返回 **None** 的情况。
  
  - 该成员强调，由于当前方法中存在一个严重的 hack，导致了此问题，并建议采用更可靠的实现。
- **使用 If_Then Gates 处理 LLVM 加载**：有人指出，当前加载 **LLVM** 的实现需要调整，以使用 **if_then** 来处理 gates。
  
  - 现有的方法被公认为是一个 hack，暗示需要更结构化的修复。
- **关于多设备设置下 CLOUD=1 的疑问**：一位成员询问 **CLOUD=1** 在多设备环境下如何运作，好奇其是否与现有的多设备处理方式一致。
  
  - 他们假设其行为将与同一台机器上当前的多设备操作保持一致。

 

**提到的链接**：[littlemountainman 在 tensors.py 中实现的 MSE 和测试 · Pull Request #7107 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7107)：实现了 MSE 及其测试

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1296243225052319765) (13 messages🔥):

> - `Update EMA Parameters` (更新 EMA 参数)
> - `Skills Transfer from Tinygrad` (从 Tinygrad 进行技能迁移)
> - `Learning Resources for Tinygrad` (Tinygrad 学习资源)
> - `Deep Learning Philosophy` (深度学习哲学)
> - `Debugging and Deploying Neural Networks` (调试与部署神经网络)

- **对 EMA 参数衰减的好奇**：一位成员对 `update_ema_parameters` 中的 *衰减 (decay)* 表示好奇，想知道这种技术在领域内是否为常见做法。
  
  - 这表明了对理解深度学习优化背后更深层机制的兴趣。
- **学习 Tinygrad 的收益迁移**：讨论了从 **Tinygrad** 学到的技能是否能迁移到像 **PyTorch** 这样的库，共识是这会极大地增强理解。
  
  - 一位贡献者强调，学习 **Tinygrad 的哲学** 帮助他们更好地掌握了复杂系统，特别是在硬件和机器人领域。
- **推荐的 Tinygrad 学习资源**：一位成员建议从 Beautiful MNIST 示例开始，并修改一个特定的 [OpenAI Cookbook 示例](https://cookbook.openai.com/examples/rag_with_graph_db) 以更好地理解 Tinygrad 的功能。
  
  - 提供了更多资源，包括用于研究内部运作机制的 [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes)。
- **深度学习作为一个整体单元**：讨论强调了将 **深度学习** 过程视为一个 *整体单元* 而不仅仅是孤立碎片的重要性，这有助于调试和部署。
  
  - **超参数 (hyperparameters)** 和架构配置等关键方面被指出对网络的整体性能至关重要。
- **对深度学习见解的感谢**：一位成员感谢另一位成员关于学习 Tinygrad 的深刻建议，称其为近期收到的最好的指导之一。
  
  - 这反映了社区内的协作精神，专注于分享知识以促进 AI 开发的共同成长。

**提到的链接**：

- [Tutorials on Tinygrad](https://mesozoic-egg.github.io/tinygrad-notes)：关于 tinygrad 的教程。
- [RAG with a Graph database | OpenAI Cookbook](https://cookbook.openai.com/examples/rag_with_graph_db)：使用 OpenAI API 构建应用的开源示例和指南。浏览代码片段、高级技术和演练集合。分享你自己的示例和指南。
- [Build software better, together](https://github.com/tinygrad/tinygrad/pull/6690/files)：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1296279793402122271) (15 messages🔥):

> - `Axolotl Dataset Shuffling` (Axolotl 数据集打乱)
> - `Gradient Accumulation Issues` (梯度累积问题)
> - `Bitnet Release` (Bitnet 发布)

- **Axolotl 打乱数据集**：在训练之前，**Axolotl** 会打乱数据集，确保每个 Epoch 的随机性。
  
  - 一位成员在查阅参考资料以验证其理解后确认了这一行为。
- **梯度累积差异**：一个共同问题指出，**梯度累积 (gradient accumulation)** 在全批次训练与切换设置之间的 Loss 可能不匹配。
  
  - 这一问题在关于一篇博客文章的讨论中被强调，同时指出 **Hugging Face** 应该很快会发布修复补丁。
- **成员讨论训练经验**：*“庆幸我昨天没开始我的 12B 训练”* 是一位成员的评论，指的是梯度累积可能带来的挑战。
  
  - 另一位成员确认他们一直在调试相关问题，并得到其他人的鼓励去休息一下。
- **Bitnet 1-bit LLMs 发布**：**Bitnet**，一个 1-bit LLM 的官方推理框架已经发布，可以在 [GitHub](https://github.com/microsoft/BitNet) 上找到。
  
  - 公告包括了简要概述以及来自仓库的图片。

**提到的链接**：

- [Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)：未找到描述。
- [How to ensure the dataset is shuffled for each epoch using Trainer and Datasets?](https://discuss.huggingface.co/t/how-to-ensure-the-dataset-is-shuffled-for-each-epoch-using-trainer-and-datasets/4212)：我正在使用 Seq2SeqTrainer，并在初始化对象时传递 datasets.arrow_dataset.Dataset 作为 train_dataset。数据集默认按 Epoch 打乱吗？如果不是，如何让它打乱？
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet)：1-bit LLM 的官方推理框架。通过创建账户为 microsoft/BitNet 开发做出贡献。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**other-llms**](https://discord.com/channels/1104757954588196865/1104758057449308220/1296309856218779770) (4 messages):

> - `A100 compute usage`
> - `DeepSpeed issues`

- **Invisietch 使用了 A100 3 天**：@nanobitz 询问了所使用的计算资源，**invisietch** 回复称在 **3 天**的时间内使用了 **1x A100**。
  
  - 讨论强调了用于特定任务的**具体硬件**配置。
- **DeepSpeed 的挑战**：Invisietch 提到了遇到的困难，表示 *“因为我无法让 DeepSpeed 正常工作”*，这表明可能存在安装或兼容性问题。
  
  - 这引发了关于在其工作流中实际应用 **DeepSpeed** 的疑问。

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1296352843472699394) (12 messages🔥):

> - `Cohere tool response yielding`
> - `Command R+ performance`
> - `Inverse Reinforcement Learning for LLMs`
> - `Stealth multilingual project`
> - `Langgraph integration updates`

- **从 Cohere 工具生成响应**：一位用户询问在使用 **langgraph** 时如何从调用 **Cohere** 的工具中生成（yield）响应，并对文档缺乏清晰度表示沮丧。
  
  - 其他成员讨论了如果当前使用 `chat_stream` 的方法没有结果，可以考虑使用 *for loop* 作为备选方案。
- **关于 Command R+ 性能的讨论**：一位成员分享说，在使用一个月后，**Command R+** 的 **0.8 版本** 表现比 **0.4 版本** 差，并寻求这种差异背后的原因。
  
  - 还有关于是否计划发布更新以在未来提高性能的询问。
- **对 LLM 逆强化学习（Inverse RL）的好奇**：一位用户分享了一篇关于 **LLM** 的 **Inverse Reinforcement Learning** 论文，并表达了对社区关于这一方向看法的关注。
  
  - 这引发了对 AI 领域创新方法的讨论兴趣。
- **征集多语言隐身（stealth）项目参与者**：一位社区成员发布招募，邀请开发者在下周参与一个需要语言专业知识的 **stealth** 项目，并提供了加入 **Aya** 服务器的链接。
  
  - 贡献者的工作将获得认可，并为顶级合作者准备了专属 **swag**，强调了在多语言领域的参与。
- **Langgraph 集成更新**：成员们提到了关于 **langgraph** 与 **Cohere** 集成的新文档，这可以帮助用户有效地利用工具。
  
  - 有暗示称近期将推出更多示例，并对 **chat_stream** 功能进行改进。

**提到的链接**：

- [Tools on LangChain — Cohere](https://docs.cohere.com/docs/tools-on-langchain#langgraph-agents)：探索在聊天机器人中使用多步和单步工具的示例代码，利用互联网搜索和向量存储。
- [Cannot stream response from cohere · Issue #592 · cohere-ai/cohere-python](https://github.com/cohere-ai/cohere-python/issues/592)：我正在使用 langgraph stream_events，在工具内部我使用的是 cohere。来自 langgraph.prebuilt import create_react_agent async def generate_stream_response(message: str, user: dict, prompt_dict: di...

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1296551096633786409) (2 messages):

> - `RAG AMAs Recording`
> - `Course Creators`

- **RAG AMAs 未录制**：一位成员询问 **RAG AMAs** 是否有录音，表示对可用资料感兴趣。
  
  - 针对该询问，一位参与者确认他们**没有进行录制**，并建议将问题直接发给特定的课程创作者。
- **联系课程创作者提问**：同一位参与者建议就课程的任何问题标签（tag） **<@955487948705513472>**。
  
  - 这强调了课程创作者对于直接参与参与者咨询的开放态度。

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/) (1 messages):

sssandra: 恭喜！不过因为偏离主题，所以将其从这里移除 🙂

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1296439890862669855) (13 条消息🔥):

> - `Quiz Access` (测验访问)
> - `Course Navigation` (课程导航)
> - `Written Article Review` (书面文章审查)
> - `Course Websites` (课程网站)
> - `MOOC Participation` (MOOC 参与)

- **访问第 5 周测验的问题**：一名成员报告了在访问 **第 5 周测验** 时遇到的问题，该测验位于课程网站的教学大纲部分，具体在[这里](https://llmagents-learning.org/f24)。
  
  - 另一名成员确认了测验的可用性，并引导他们到正确的章节进行访问。
- **新成员寻求课程指导**：一位新加入者询问在填写课程表单后是否会收到后续邮件，并寻求关于访问课程材料的澄清。他们得到的答复是，可以放心继续参与课程并根据需要完成测验。
  
  - 成员们鼓励将精力集中在 MOOC 上，不要为 hackathons 感到压力，并支持查阅补充材料。
- **课程网站的澄清**：成员们讨论了两个不同的课程网站，并确认 **llmagents-learning.org** 下的站点是面向 MOOC 学生的正确站点。另一个站点主要面向在校参加实体课程的 **UC Berkeley** 学生。
  
  - 他们建议不要使用 Berkeley 的站点进行课程相关活动，并提到根据学生身份的不同需要填写不同的表单。
- **作业文章审查请求**：一名成员询问是否可以在将文章发布到社交媒体之前进行审查，以确保其符合课程预期。有人担心审查过程会很复杂，但其他人强调只需遵守课程网站上提供的文章指南即可。
  
  - 建议只要文章符合一般标准，成员们就不必过于担心预审。
- **每周课程进度更新**：一位参与者向小组更新了他们刚刚完成 **Week 1** 的情况，并计划按照课程大纲结构继续学习。他们的主动性得到了赞赏，并被鼓励继续推进每周的内容。

**提到的链接**：

- [Large Language Model Agents](https://llmagents-learning.org/f24)：未找到描述
- [CS294/194-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24)：2024 秋季

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1296506232353591378) (7 条消息):

> - `AI Engineering Blogs` (AI 工程博客)
> - `LangChain vs LangGraph`
> - `LangChain Critique` (LangChain 批评)
> - `Agent Visualization` (Agent 可视化)
> - `LangGraph Tools` (LangGraph 工具)

- **值得关注的 AI 工程博客**：一位用户请求推荐优秀的 AI 工程博客，并对 **Retrieval systems**（检索系统）和 **Multi-agent architectures**（多 Agent 架构）表现出浓厚兴趣。
  
  - *未列出具体博客*。
- **切换到 LangGraph 的优点**：讨论围绕从 **LangChain 切换到 LangGraph** 的优势展开，特别是在抽象和易用性方面。
  
  - 一名成员询问 **LangGraph** 提供了哪些 **LangChain** 所不具备的独特功能。
- **使用两年后对 LangChain 的批评**：一位 **LangChain** 的长期用户反思了围绕该工具的批评，尽管他们花了很多时间学习它。
  
  - 他们幽默地提到了深夜尝试精通 **LangChain** 时产生的挫败感。
- **可视化 Agent 图**：有人请求如何为其项目中的 Agent 可视化或创建图表。
  
  - *讨论中未提供解决方案。*
- **LangGraph 的工具访问**：一名成员发起了关于 **LangGraph** 可访问工具的讨论，寻求对其功能的更多深入了解。
  
  - *回复中未分享详细见解。*

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1296489925990027295) (3 条消息):

> - `Inverse Reinforcement Learning for LLMs` (LLM 的逆强化学习)
> - `NotebookLM Features` (NotebookLM 功能特性)
> - `Gen AI Agent Hackathon` (生成式 AI Agent 黑客松)

- **探索 LLM 的逆强化学习 (Inverse RL)**：分享了一篇讨论将 **inverse reinforcement learning** 应用于 **LLMs** 的近期论文，引发了对该方向反馈的兴趣和好奇 [PDF 链接](https://arxiv.org/pdf/2410.12491)。
  
  - 参与者渴望听取关于这种方法在提升语言模型能力方面的可行性和影响的看法。
- **NotebookLM 推出令人兴奋的新功能**：**Google** 宣布了 **NotebookLM** 的新功能，包括音频概览和增强的协作工具，作为该笔记本商业试点计划的一部分 [详情点击这里](http://goo.gle/3UcO8Na)。
  
  - 新功能旨在通过允许用户在处理音频内容时无缝进行多任务处理来提升用户体验。
- **黑客松邀请团队构建生成式 AI Agent**：**CreatorsCorner** 邀请参与者参加一场黑客松，重点是开发 **Gen AI 驱动的多智能体系统 (multi-agent systems)**，旨在支持用户的日常任务，同时确保安全性和保障 [更多信息](https://lu.ma/ke0rwi8n)。
  
  - 该挑战鼓励在创建协作式 AI 解决方案方面进行创新，并考虑伦理影响和社会效益。

**提到的链接**：

- [来自 Google (@Google) 的推文](https://x.com/Google/status/1846954813193359397?t=8gWKjTOUhZAYbjFMHluqGw&s=19)：✨ NotebookLM 即将推出新功能 ✨ 🗣️ 自定义音频概览并引导对话 🤝 在 NotebookLM 商业试点计划中与队友协作 🎧 在听音频概览的同时...
- [垂直领域特定 AI Agents 黑客松 · Luma](https://lu.ma/ke0rwi8n)：Gen AI Agents CreatorsCorner，与 aixplain, Sambanova Systems, Prem, Marly, Senso, Mistral, coval, heygen, fiberplane, exa 等合作……

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1296381944522604565) (4 条消息):

> - `Graph Reinforcement Learning` (图强化学习)
> - `Inverse Reinforcement Learning for LLMs` (LLM 的逆强化学习)
> - `Importance of Survey Papers` (综述论文的重要性)

- **对图强化学习综述感到兴奋**：一名成员对发现一篇新的 [图强化学习综述 (Graph Reinforcement Learning)](https://arxiv.org/abs/2404.06492) 表示兴奋，强调了其作为跨领域决策方法的潜力。
  
  - 他们指出，**图结构 (graph structures)** 与 **强化学习 (reinforcement learning)** 之间的结合可以在化学和计算机科学等领域产生创新策略。
- **对综述作者的赞扬**：另一名成员分享了对高质量综述价值的看法，称其为研究者的“天赐之物” (*godsends*)。
  
  - 这反映了社区对全面文献综述的广泛认可，这些综述有助于理解复杂主题。
- **关于 LLM 逆强化学习的讨论**：出现了一项关于 [使用逆强化学习 (inverse RL) 提升 LLM 的文章](https://arxiv.org/pdf/2410.12491) 的征求意见。
  
  - 该询问表明了人们对探索如何应用 **inverse reinforcement learning** 来增强大语言模型能力的兴趣。

**提到的链接**：[用于组合优化的图强化学习：综述与统一视角](https://arxiv.org/abs/2404.06492)：图是基于连接实体之间关系的系统的自然表示。组合优化问题在考虑与过程相关的目标函数时产生...

---

### **Alignment Lab AI ▷ #**[**general**](https://discord.com/channels/1087862276448595968/1095458248712265841/1296564979989741719) (1 条消息):

> - `Twitter/X embeds` (Twitter/X 嵌入)
> - `FixTweet/FxTwitter`

- **修复损坏的 Twitter/X 嵌入**：成员们讨论了 **修复损坏的 Twitter/X 嵌入** 的需求，以便在 Discord 和 Telegram 等平台上支持更多功能，如多图、视频、投票和翻译。
  
  - 一名成员分享了 [FixTweet/FxTwitter 项目](https://x.com/i/spaces/1ypKdpLNZXnKW) 的链接，敦促其他人参与改进嵌入功能。
- **围绕推文功能的讨论**：有一场关于 **更具互动性的推文功能** 对用户参与度（特别是嵌入方面）影响的对话。
  
  - 成员们认为 **增强的多媒体支持** 可以提高整体参与度和内容分享。

**提到的链接**：[来自 GitHub 的推文 - FixTweet/FxTwitter：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能](https://x.com/i/spaces/1ypKdpLNZXnKW)：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1296495172653158412) (1 条消息):

> - `Gen AI Bug Bounties`
> - `Vulnerability Submission Process`
> - `User Dashboard Features`
> - `Real-Time Notifications`
> - `Training Opportunities`

- **Gen AI Bug Bounties 门户现已上线**：用于 **gen AI bug bounties** 的 [门户](https://discord.com/channels/1089876418936180786/1245784344539435128/1295876886584492033) 已正式发布，通过直观的设计和自动分拣（automatic triage）简化了漏洞提交流程，从而实现更快的审核。
  
  - 该计划旨在通过让研究人员更轻松地报告漏洞来增强安全性。
- **引入增强型 User Dashboard 功能**：全新的 **Personalized User Dashboard** 提供了一个集中视图，用于跟踪提交状态、更新和研究人员进度。
  
  - 该仪表盘旨在提升用户体验并简化提交管理。
- **通过 Real-Time Notifications 保持信息同步**：**Real-Time Notifications** 将针对提交漏洞的每一项操作发送即时邮件提醒，确保完全透明。
  
  - 此功能允许用户毫无延迟地掌握其提交状态的最新动态。
- **通过 Role-Based Permissions 实现安全协作**：该平台实施了 **Role-Based Permissions** 以提供结构化的访问控制，保障数据管理和协作的安全。
  
  - 此举确保敏感信息仅限授权人员访问。
- **11 月即将推出的 Training Opportunities**：令人期待的 **Prompt Engineering 课程和 CTF 挑战** 将于 11 月启动，提供专注于 AI 漏洞的技能提升机会。
  
  - 这些教育计划将包括持续更新的 **每周博客与教程**，以增强 AI 安全领域的知识。

 

---

---

---

---

---

{% else %}

> 完整的逐频道详情已在邮件中截断。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}