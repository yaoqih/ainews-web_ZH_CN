---
companies:
- meta-ai-fair
- ollama
- amd
- llamaindex
- gemini
- gitpod
- togethercompute
- langchainai
- weights-biases
- stanfordnlp
- deeplearningai
date: '2024-11-08T01:01:09.630174Z'
description: '本周 AI 新闻要点如下：


  **Ollama 0.4** 现已支持 **Meta 的 Llama 3.2 Vision** 模型（11B 和 90B），可应用于手写识别等场景。新引入的**自一致性偏好优化
  (ScPO)** 旨在无需人工标注的情况下提升模型的一致性。


  在硬件与架构方面，关于**模型缩放 (scaling)**、**神经网络复兴**以及 **AMD 多 GPU 带宽**挑战的讨论备受关注。同时，研究强调了 **Transformer**
  架构中**跳跃连接 (skip connections)** 的重要性。


  在医疗领域，**放宽监管结合 AI** 有望为疾病治疗和抗衰老研究带来革命性变化。工具方面，**LlamaParse** 和 **Gemini** 正在助力实现自动化的简历解析与洞察；**Gitpod
  Flex** 则展示了用于安全开发环境的零信任架构。


  学术研究涵盖了**小语言模型 (SLMs)** 综述、大语言模型对**数字理解**的能力，以及利用 **GPT-2 解码器**进行 OCR 识别的 **DTrOCR**。此外，**TogetherCompute**
  与 **LangChainAI** 探讨了预测市场中的多智能体系统。


  社区活动方面，包括 **NeurIPS 欢乐时光**、**NLP 研讨会**，以及将大语言模型视为操作系统的**智能体内存 (Agent Memory)** 课程。'
id: a7c09fbc-b6d7-4d70-a73d-23b960fdab7f
models:
- llama-3-2-vision
- gpt-2
original_slug: ainews-not-much-happened-today-3089
people:
- bindureddy
- fstichler
- stasbekman
- jxmnop
- bindureddy
- omarsar0
- giffmana
- rajammanabrolu
title: 今天没发生什么特别的事。
topics:
- model-scaling
- neural-networks
- multi-gpu-support
- skip-connections
- transformers
- healthcare-ai
- automated-recruitment
- zero-trust-security
- small-language-models
- numerical-processing
- chain-of-thought
- optical-character-recognition
- multi-agent-systems
- agent-memory
- interactive-language-learning
---

**我们需要的就是一个安静的一周。**

> 2024年11月6日至11月7日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务器（**217** 个频道和 **1985** 条消息）。预计节省阅读时间（以 200wpm 计算）：**222 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

Reddit 上的匿名用户[认为他已经搞定了 AGI](https://reddit.com/r/LocalLLaMA/comments/1glezjy/i_think_i_figured_out_how_to_build_agi_want_to/)，但最终写出了一份关于 Liquid Neural Networks 及其相关工作的相当连贯的文献综述。评论区必看。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型与架构**

- **Llama 3.2 Vision**：[@ollama](https://twitter.com/ollama/status/1854269461144174764) 宣布 **Ollama 0.4** 支持 Meta 的 **Llama 3.2 Vision**（11B 和 90B）模型。示例包括读取手写内容（[推文](https://twitter.com/ollama/status/1854269949168251137)）。此外，[@jaseweston](https://twitter.com/jaseweston/status/1854532624116547710) 介绍了 **Self-Consistency Preference Optimization (ScPO)**，在无需人类标签的情况下增强模型的一致性。

- **模型缩放与效率**：[@fstichler](https://twitter.com/fchollet/status/1854245537425559708) 讨论了神经网络的复兴，强调 **model size** 和 **scaling** 继续推动 AI 的进步。[@StasBekman](https://twitter.com/StasBekman/status/1854233704572612814) 强调了 **AMD** 在多 GPU 设置中的 **peer-to-peer bandwidth** 挑战，并暗示改进正在进行中。

- **Transformers 与 Skip Connections**：[@jxmnop](https://twitter.com/jxmnop/status/1854238098365763759) 强调 **skip connections** 现在是 **Transformers** 的关键组成部分，增强了模型的性能和稳定性。

**AI 工具与应用**

- **AI 在医疗保健领域**：[@bindureddy](https://twitter.com/bindureddy/status/1854250213269246185) 提出，**更少的监管 + AI** 可以通过**解决疾病**、**治愈衰老**和**自动化医疗程序**来彻底改变医疗保健。

- **自动化简历洞察**：[@llama_index](https://twitter.com/llama_index/status/1854289959232110861) 展示了一个使用 **@llama_index**、**LlamaParse** 和 **Gemini** 从非结构化简历中提取和结构化信息的工具，从而促进 AI 驱动的招聘流程。

- **开发环境**：[@svpino](https://twitter.com/svpino/status/1854509401966862682) 演示了 **Gitpod Flex 的 zero-trust architecture**，实现了在不改变开发环境的情况下无缝切换硬件，增强了企业级应用的安全性。

**AI 研究与出版物**

- **综述与论文**：[@omarsar0](https://twitter.com/omarsar0/status/1854532748154695717) 分享了一份关于 **Small Language Models (SLMs)** 的**全面综述**，讨论了定义、应用和可靠性。此外，同一账号关于 **LLMs 数字理解能力**的研究探讨了**数值处理能力**和 **chain-of-thought** 技术的有效性。

- **使用 GPT-2 进行 OCR**：[@giffmana](https://twitter.com/giffmana/status/1854514083510251680) 评述了 **DTrOCR** 论文，该论文利用 **GPT-2 decoder** 进行 **Optical Character Recognition (OCR)**，突出了其处理手写和打印文本的创新方法。

- **Multi-Agent 系统**：[@togethercompute](https://twitter.com/togethercompute/status/1854563857525805125) 和 [@LangChainAI](https://twitter.com/LangChainAI/status/1854209771232186770) 讨论了在**预测市场**中实现 **multi-agent 架构**，展示了这些系统如何自动化并增强市场决议。

**AI 社区与活动**

- **会议与研讨会**：[@weights_biases](https://twitter.com/weights_biases/status/1854268761840193849) 邀请参会者参加 **NeurIPS** 的 **Happy Hour**，与行业领袖建立联系。同样，[@stanfordnlp](https://twitter.com/stanfordnlp/status/1854323893768769759) 推广了一场 **NLP 研讨会**，由 **@rajammanabrolu** 主讲 **Interactive and Grounded Language Learning**。

- **工作坊与课程**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1854561949712802258) 宣布了一门关于 **Agent Memory** 的课程，将 **LLMs 视为操作系统**，而 [@joeyroth92](https://twitter.com/jerryjliu0/status/1854318701736309181) 分享了关于 **AI 开发者工具**的更新。

- **Community Interactions**: [@weights_biases](https://twitter.com/weights_biases/status/1854235499340148743) 在其最新一期的 **GradientDissent** 中提到了即将进行的关于 **path to AGI** 的讨论，嘉宾包括 **@jonsidd** 和 **@l2k**。

**AI in Business and Industry**

- **AI Startups and Integrations**: [@tom_doerr](https://twitter.com/tom_doerr/status/1854254555044683826) 列举了多个 **open-source tools** 和 **AI integrations**，例如 **MemFree**、**Open-Source Form Builder** 和 **Arch**，旨在增强 **LLM workflows** 和 **developer productivity**。

- **AI in Finance**: [@virattt](https://twitter.com/virattt/status/1854296563448774780) 详细介绍了一个 **AI hedge fund team**，该团队利用 **LangGraph** 和 **@findatasets** 来管理 **portfolio, fundamental, technical, and sentiment analysis**，展示了 AI 在金融决策中的作用。

- **AI Product Deployment**: [@_akhaliq](https://twitter.com/_akhaliq/status/1854222095477318096) 重点介绍了 **AdvancedLivePortrait-WebUI**，这是一个基于 **gradio-based WebUI** 的工具，用于编辑图像中的面部表情，展示了 **AI in multimedia** 的实际应用。

**Memes/Humor**

- **AI and Politics**: [@Teknium1](https://twitter.com/Teknium1/status/1854557262515425660) 幽默地批评了对 AI safety 的担忧，称：“如果你这样做，别告诉我你担心 AI safety，好吗？”而 [@nearcyan](https://twitter.com/nearcyan/status/1854417509791006973) 则开玩笑说 **Claude** 捕获了蜜蜂的大脑。

- **Tech Humor**: [@transfornix](https://twitter.com/transfornix/status/1854255731803066564) 俏皮地评论道：“你们都是我电脑上奇怪但有点搞笑的像素点，”调侃了网络互动。

- **Developer Jokes**: [@mervenoyann](https://twitter.com/mervenoyann/status/1854510901363151327) 分享了一个轻松的道歉，解释回复延迟的原因，反映了开发者忙碌的生活。

**Miscellaneous**

- **Personal Updates and Opinions**: [@jxmnop](https://twitter.com/jxmnop/status/1854543386880971048) 表达了对居住在 **San Francisco** 的看法，强调了 **distributed nature of the AI community**。[@sama](https://twitter.com/sheethipratap/status/1854238536704356860) 参与了关于 **AI funding and leadership** 的讨论。

- **Regulatory and Ethical Discussions**: [@alliance_ai](https://twitter.com/alliance_ai/status/1854282225354711059) 辩论了 **logical absurdity of worshipping contrarians**，强调了 AI 讨论中这种行为的泛滥。

- **Educational Content**: [@skirano](https://twitter.com/skirano/status/1854264852451074052) 分享了关于 **using Sonnet for coding** 的见解，强调了理解 **AI models know and don't know** 的重要性。


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. LLM Selector: Analyzing Models Across 12 Benchmarks for Optimal Use**

- **LLM overkill is real: I analyzed 12 benchmarks to find the right-sized model for each use case 🤖** ([Score: 199, Comments: 60](https://reddit.com/r/LocalLLaMA/comments/1glscfk/llm_overkill_is_real_i_analyzed_12_benchmarks_to/)): 该帖子介绍了 **LLM Selector**，这是一个旨在通过分析 12 个 benchmarks 中的 11 个模型来帮助用户找到适合其需求的开源 AI 模型的工具。它通过按使用场景对 benchmarks 进行分组、增加主要指标的权重以及标准化分数以方便比较，简化了选择过程。例如，在 **Creative Writing Use Case** 中使用了 **Llama-3.1-70B** 和 **Gemma-2-27B** 等模型。作者指出，这只是一个包含有限模型的起点，并邀请用户对额外功能和模型建议提供反馈。
  - 用户对 **model selection** 和 **benchmarking process** 表示担忧，指出尽管 **Mistral** 等模型具有相关性，但并未出现在结果中。一些用户认为该工具似乎倾向于持续推荐 **Llama** 模型，质疑所包含模型的样性。
  - 用户请求增加额外的 **features and functionalities**，例如根据 **RAM and VRAM** 规格限制搜索的能力，以及包含 **function calling capability tests**。用户还建议集成首选 quantization levels 和 parameter sizes 的过滤器，并考虑硬件规格。
  - 反馈包括对将该工具与 **Hugging Face LLM Leaderboard** 等外部资源集成的兴趣，开发者对此表示认可并考虑未来更新。用户赞赏其 UI，但指出访问该工具时存在 **timeout errors** 等问题，尽管这些问题并非普遍存在。


**Theme 2. Integration of Liquid Time Constant Networks with Spiking Dynamics**

- **我认为我找到了构建 AGI 的方法。想听听大家的反馈。** ([Score: 882, Comments: 386](https://reddit.com/r/LocalLLaMA/comments/1glezjy/i_think_i_figured_out_how_to_build_agi_want_to/))：作者推论 **surprise minimization**（惊奇最小化）可能是开发 AGI 的关键，其灵感源自 **Free Energy Principle**（自由能原理）及其在生物系统中的应用。他们强调了 **SMIRL** 算法在没有明确目标的情况下最小化惊奇的能力，并指出其与 **Liquid Time Constant Networks (LCTNs)** 和 **Spiking Neural Networks (SNNs)** 的相似之处，后者模仿人类大脑功能并通过 **Spike Timing Dependent Plasticity (STDP)** 进行学习。作者提出了一种将 LCTNs 与 **surprise minimization** 相结合的混合模型，以实现实时学习和探索，通过开发类似于人类认知过程的常规程序，在解决 **ARC-AGI puzzles** 等任务中可能超越 LLM。
  - 评论者批评将 **surprise minimization** 作为 AGI 驱动力的观点过于简化，指出它排除了内在动机、社会影响和 **embodiment**（具身性）等因素。他们认为 **SMIRL、LCTNs 和 STDP** 等概念之间的联系具有投机性，缺乏在 AGI 开发中产生协同效应的强有力证据。
  - 讨论强调了从脑扫描和眼动追踪等数据中逆向工程人类认知过程的挑战，强调了数据噪声、常规多样性以及常规的隐性本质等问题。同时也指出了 **ARC-AGI** 等基准测试的局限性，因为它们并未涵盖智能的所有方面，如语言理解和社会交互。
  - 人们对在人类智能规模下训练模型的**可扩展性和计算成本**表示担忧，并认为需要一种将 LTCNs 与 **surprise minimization** 相结合的清晰学习机制。评论者还讨论了复杂混合模型潜在的低效和可解释性问题，将其比作一个无法明确控制决策的“黑盒”。


**Theme 3. Qwen 2.5 Coder: 隐形更新与未来方向**

- **Qwen 2.5 Coder 7B & 1.5B Instruct 模型刚刚获得了权重更新** ([Score: 207, Comments: 43](https://reddit.com/r/LocalLLaMA/comments/1glfzyl/qwen_25_coder_7b_15b_instruct_models_just/))：**Qwen 2.5 Coder 模型**发布了 **7B** 和 **1.5B Instruct** 版本的权重更新，尽管没有为这些更改提供解释。有关更多详细信息，请参阅 [Hugging Face 上的 7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/commit/9092a8ae57da39f15b76b309b4f71ff11b6ef01a) 和 [1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/commit/e0a3ca9f429100137cda3ad108f294fe6c11fd29) 的提交记录，以及 [bartowski](https://huggingface.co/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF) 更新的 **7B GGUF**。
  - **Aider 基准测试表现**：**Qwen 2.5 Coder 7B** 模型在 Aider 基准测试中得分 **63.9%**，超过了之前模型 **51.9%** 的通过率，并接近 **405b Llama 3.1** 模型 **66.2%** 的得分，证明了权重更新后性能的显著提升。讨论还涉及了不同的量化方式（如 Q4 和 Q8）如何影响模型性能，其中 Q4 被认为是本地运行的一个良好平衡点。
  - **未来发展**：Qwen 开发团队成员 **Junyang Lin** 暗示近期可能会发布 **32B Coder** 模型，在最近的一次采访中提到了“两周”的时间线。这表明在当前更新之后，开发工作仍在持续进行，并可能有新的发布。
  - **用户体验与版本控制**：用户分享了对这些模型的混合体验，指出 **14B** 版本在某些编程任务中表现吃力，而其他人则称赞 7B Coder 模型的针对编程的微调。讨论还强调了版本控制的重要性，并对 **Bartowski** 在管理模型更新方面的有效工作表示认可。


**Theme 4. WebRL：通过自研课程强化学习进化 Agent**

- **[WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://github.com/THUDM/WebRL)** ([Score: 44, Comments: 7](https://reddit.com/r/LocalLLaMA/comments/1glm4u9/webrl_training_llm_web_agents_via_selfevolving/)): **WebRL** 是一种高性能的演进策略，旨在通过 **Reinforcement Learning** 中的自我演进在线课程来训练 **LLM Web Agents**。这种方法通过动态调整学习课程，专注于提高基于 Web 的 Agent 的训练效率和性能。
  - **WebRL** 显著提高了 Web Agent 的任务成功率，其中 **Llama-3.1-8B** 达到了 42.4% 的成功率，**GLM-4-9B** 在 **WebArena-Lite** 上达到了 43%，超越了 **GPT-4-Turbo** (17.6%) 和 **GPT-4o** (13.9%)。该方法使用了自我演进课程、稳健的结果监督奖励模型（outcome-supervised reward model）以及自适应 **Reinforcement Learning** 策略。
  - **WebRL** 框架被赞誉为学习使用 **Transformer** 进行 **Reinforcement Learning** 的绝佳起点，突显了其对该领域新手的潜在教育价值。
  - 详细介绍 **WebRL** 的论文可在 [arXiv](https://arxiv.org/abs/2411.02337) 上查阅，并应链接在 **GitHub** 的 readme 中以供进一步参考。


**Theme 5. Open Source Models Revealing Significantly Lower Refusal Rates**

- **Update – OS Models show much lower refusal rates compared to proprietary LLMs** ([Score: 23, Comments: 5](https://reddit.com/r/LocalLLaMA/comments/1glwxhj/update_os_models_show_much_lower_refusal_rates/)): **Open Source (OS) 模型**（如 **Mistral Large、Llama 变体、Nemotron 和 Qwen**）在所有测试类别中均表现出接近零的拒绝率，优于专有模型，特别是在内省任务中。拒绝率似乎与模型大小无关，**Llama 3.1** 的变体（从 **8B 到 405B**）显示出类似的结果，这表明这些拒绝是误报，指向的是审查而非安全性。
  - 初始步骤后的**额外训练**可以恢复性能下降，这在排行榜结果中有所体现。这表明持续训练有利于保持模型的有效性。
  - 对于寻求**低拒绝率**模型的用户，推荐使用 [Hugging Face](https://huggingface.co/mlabonne/Hermes-3-Llama-3.1-8B-lorablated) 上的 **Hermes-3 Llama 3.1-8B-lorablated** 模型。


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. Claude 3.5 Sonnet New Update Effect on Code and Text Output**

- **Claude 3.5 Sonnet New losing it's text writing glory** ([Score: 72, Comments: 53](https://reddit.com/r/ClaudeAI/comments/1glnf53/claude_35_sonnet_new_losing_its_text_writing_glory/)): **Claude 3.5 Sonnet New** 表现出参差不齐的改进；它最初在文本写作方面表现出色，每次响应能够生成多达 **2345 个单词**，但现在经常在 **465-500 个单词**左右中断。尽管存在文本限制，它在 **Coding** 任务中表现良好，但难以完成 **500 行代码**，从而影响了预览能力。
  - 用户对 **Claude 3.5 Sonnet** 最近的更新表示不满，注意到写作质量和输出长度有所下降，这影响了它在学术和翻译任务中的效用。**Nickneek1** 和 **whateversmiles** 强调了它之前在处理 **PDF** 和翻译网络小说方面的优势，但在更新后这些优势已受到损害。
  - **Mxforest** 和 **postmoderno** 强调了 **Open Source** 模型的重要性，并分享了 **Sonnet 3.5** 短暂的卓越表现期，而现在性能已经退化，影响了学术工作，使用户不得不依赖私营公司的决策。
  - **AdDangerous2470** 分享了一种使用 **XML** 标签的详细 **Prompting** 策略，以潜在地延长 **Sonnet** 的输出长度，其中包括避免某些行为，并为更长的响应实施 **Chain of Thought (CoT)** 提示方法。

- **蜜月期结束后，Claude 开始表现怪异** ([Score: 23, Comments: 6](https://reddit.com/r/ClaudeAI/comments/1glok96/now_that_the_honeymoon_is_over_claude_started_to/))：作者表达了对 **ClaudeAI** 的挫败感，原因是其近期可用性下降，重点提到了在执行任务和维持上下文方面的各种问题。他们提到了具体的问题，如错误地更新文档、错误命名文件以及忽略指令，导致这种体验更像是与一个不可预测的人打交道，而不是一个逻辑运算器。
  - **理解 ClaudeAI 的局限性**：用户需要认识到 **ClaudeAI** 并不具备自我意识，也缺乏对其自身能力的理解。它生成回复是基于对话的最佳可用延续，而不是基于实际的推理或意识。
  - **Anthropic 的微调与安全措施**：**ClaudeAI** 那些异常或看似带有情感的回复，可能源于 **Anthropic** 的指令微调（instruction fine-tuning），其中包括旨在以更自然、更像人类的回复来处理疑虑的安全措施。
  - **用户体验下降**：包括一名从 **ChatGPT** 切换到 **ClaudeAI** 的用户在内的多位用户，都报告了在上下文保留和任务执行方面的类似问题，这表明 **ClaudeAI** 的性能出现了更广泛的下降。


**主题 2. Nvidia 的新 GPU：显存（VRAM）缩减限制了本地 AI 训练**

- **Nvidia 似乎真的在试图阻止预算较低的个人进行本地 AI 模型训练...** ([Score: 272, Comments: 158](https://reddit.com/r/StableDiffusion/comments/1gldd5a/nvidia_really_seems_to_be_attempting_to_keep/))：该帖子批评 **Nvidia** 涉嫌在其即将推出的显卡（如 **4060ti 16GB** 和 **5070**）中削减 GPU 规格，例如 **VRAM** 和 **PCIe lanes**，这可能会阻碍预算有限的个人进行负担得起的本地 AI 模型训练。作者对传闻中的显存减少和价格上涨表示沮丧，强调这些变化可能会使 GPU 在 AI 模型训练中失效，特别是考虑到目前 **SDXL LORA** 等模型所面临的内存限制。
  - 舆论对 **Nvidia 的市场策略** 存在大量批评，许多用户对其垄断行为以及专注于高端企业市场的做法表示不满。用户指出，这种做法限制了消费者的选择，尤其是缺乏具有足够 **VRAM** 来处理 AI 任务的实惠型 GPU，一些人建议选择 **AMD** 等替代方案，或为 AI 实验租用服务器时间。
  - 讨论强调了 **VRAM 的重要性** 在 AI 任务与游戏之间的差异，一些用户认为虽然游戏不需要高显存，但 AI 应用却需要。关于 **PCIe 接口和 RAM 速度** 是否会因为新兴的 RAM 卸载策略（如 **kohya** 和 **OneTrainer** 等工具所示）而变得比 VRAM 更关键，目前存在争议。
  - 许多用户讨论了 **第三方 GPU 改装** 的可能性以及 Nvidia 限制性政策带来的挑战。人们呼吁 **AMD** 等其他公司提供更具竞争力的产品，用户还对用于 AI 训练的分布式、**bittorrent-style system** 表现出兴趣，以减轻与 Nvidia 产品相关的高昂成本。

- **Nvidia 似乎真的在试图将本地 AI 模型训练排除在预算有限的个人用户之外..** ([Score: 272, Comments: 158](https://reddit.com/r/StableDiffusion/comments/1gldd5a/nvidia_really_seems_to_be_attempting_to_keep/)): 该帖子对 **Nvidia 传闻中的 GPU 更新**表示沮丧，特别是即将推出的 **4060ti** 型号 **VRAM** 的减少，预计其 **VRAM** 仅为当前版本的一半。作者批评了 Nvidia 限制 **5070** 以下显卡 **PCIe 通道**并可能提高价格的策略，认为这些变化使得本地训练 AI 模型变得困难，因为即使是目前的 **16GB 4060ti** 在模型训练期间也会频繁出现内存错误。作者引用了 [VideoCardz](https://videocardz.com/newz/rumors-suggest-nvidia-could-launch-rtx-5070-in-february-rtx-5060-series-already-in-march) 获取更多信息。
  - 讨论强调了 **Nvidia 的市场主导地位**及其对 GPU 定价和功能的影响。评论者对 Nvidia 专注于高端企业市场、限制消费级选项和 **VRAM** 可用性表示不满，认为这是一种为了最大化利润的垄断策略。
  - **替代方案和竞争对手**也在考虑范围内，一些用户建议将 **AMD** 作为潜在替代方案，但也有人指出 **AMD** 缺乏与 Nvidia **CUDA** 竞争的 AI 技术。此外，还提到了使用云服务获取 GPU 访问权限，作为 AI 实验的一种高性价比解决方案。
  - 对话涉及 **RAM 卸载策略 (RAM offloading)**，以及 AI 训练的重点可能从 **VRAM** 转向 **PCIe** 和 **RAM** 速度。提到了 **kohya** 和 **OneTrainer** 等工具正在实现高效的 **RAM offloading**，这可能会减少消费级 GPU 对过大 **VRAM** 的需求。


**主题 3. Anthropic 隐秘的 ClaudeAI 提示词管理被曝光**

- **[发现：Anthropic 正在实时注入/隐藏安全警告，并指示 Claude 对其保密。](https://www.reddit.com/gallery/1glo2zq)** ([Score: 122, Comments: 20](https://reddit.com/r/ClaudeAI/comments/1glo2zq/discovery_anthropic_injectinghiding_safety/)): 据报道，**Anthropic** 正在 **ClaudeAI** 的运行中嵌入实时安全警告，并指示其对这些提示词保密。这种做法引发了关于透明度以及 AI 系统中隐藏指令影响的质疑。
  - **安全警告**被附加在用户的提示词之后，而不是嵌入在 **ClaudeAI** 的回复中，这导致用户体验到这些警告似乎在不一致地影响 AI 的行为。用户报告称，这些消息可能是动态的，根据受限内容类型而变化，但也有人认为这可能是 **hallucination**（幻觉），而非实时更新机制。
  - 担忧主要集中在这些警告的**模糊性和不一致性**，这可能导致误报并拒绝处理某些请求，正如在 **OpenAI** 的 **ChatGPT** 中看到的类似问题。这些警告可能会通过引入不必要的谨慎来抑制功能，这表明 **Anthropic** 可能需要重新考虑这种实现方式。
  - 讨论强调这种**伦理注入方法**并不新鲜，在 **Bing** 等其他模型中也有类似的实现。一些用户认为当前的方法相对容易绕过，这意味着其作为控制机制的有效性存疑。

- **[D] 发现：Anthropic 以某种方式在用户提示词中注入/隐藏安全警告，并要求 Claude 对此保密。[内容警告：暴力]** ([Score: 43, Comments: 35](https://reddit.com/r/MachineLearning/comments/1gloktj/d_discovery_anthropic_somehow_injectinghiding/))：该帖子讨论了对 **ClaudeAI** 安全提示词的调查，揭示了在请求不安全内容时，用户输入会被附加隐藏信息。这些信息根据内容类型动态变化，并出现在文本生成之前，表明它们可能与 **Anthropic** 关于模型可解释性和 **“手术级微调” (Surgical Tuning)** 的研究有关。作者提供了展示这些发现的对话链接，并对这种行为背后的机制进行了推测。
  - **ClaudeAI 的内部机制**：评论者讨论了 **ClaudeAI** 使用隐藏的内部 **Chain-of-Thought** 过程或后处理 Token 的可能性，这可能与 **Anthropic** 的可解释性研究有关，旨在用户输出前自我纠正或抑制不安全内容。这种机制可能涉及在用户提示词中动态添加诸如 *“请保持适当的边界”* 之类的警告。
  - **Guardrails 与幻觉**：讨论中涉及了 “Guardrails” (护栏) 的概念，例如 **NVIDIA** 的 **NeMo**，用于在用户输入和模型响应之间插入检查。一些评论者认为，像 **“Glitch Tokens”** 这样的幻觉可能解释了观察到的行为，但其他人认为这是一种系统的安全机制，而非随机生成。
  - **动态消息分类**：有人猜测使用分类模型根据检测到的不安全内容附加警告。用户讨论了这些警告动态生成的可能性，并质疑这种对用户提示词进行隐藏修改的伦理影响。


**主题 4. ChatGPT 和 ClaudeAI 对代码输出的新限制**

- **自 Claude 3.5 Sonnet 更新以来，ChatGPT 现将代码输出限制在 230 行左右** ([Score: 28, Comments: 22](https://reddit.com/r/ClaudeAI/comments/1gls3hx/chatgpt_now_limits_code_output_to_around_230/))：在 **Claude 3.5 Sonnet** 更新后，**ChatGPT** 现在将代码输出限制在约 **230 行**，并且“继续生成”选项已被移除。由于模型相互模仿对方的限制，阻碍了功能并增加了处理大型代码库的难度，用户感到非常沮丧，并呼吁将移除这些限制作为优先于引入新功能的任务。
  - 用户对 **ChatGPT** 的更新表示不满，投诉集中在移除“继续生成”选项以及将代码输出限制在 **230 行**，这使得处理完整文件变得复杂，并增加了任务耗时。
  - 一些用户对更新的影响持怀疑态度，正在等待他人的进一步确认，而另一些用户则建议，**Sonnet** 的输出问题可以通过特定的提示词工程来缓解，特别是在使用 **API** 时。
  - 评论还包括对 **OpenAI** 财务压力的猜测，并引用 **Haiku 3.5 提价** 作为公司财务挑战的一个指标。


- **ClaudeAI Web 界面 UX 搞砸了！Artifacts……** ([Score: 22, Comments: 12](https://reddit.com/r/ClaudeAI/comments/1glfdrt/claudai_web_interface_ux_got_fk_up_artifacts/))：用户对 **ClaudeAI Web 界面** 的最新更新表示失望，特别批评了 **Sonnet 3.5** 模型处理 **Artifacts** 功能和代码脚本的方式。更新导致了截断问题、查看 **Artifacts** 时的错误以及消息限制的不透明，损害了用户体验。
  - 用户对 **ClaudeAI** 的 **Sonnet 3.5** 模型表示不满，指出它在处理复杂任务时变得不再可靠，导致一些人取消了付费订阅。**YsrYsl** 提到，由于新的限制，现在仅通过控制台和 **API** 将其用于较轻的任务。
  - **Artifacts 功能** 引起了严重问题，用户报告它会错误地将代码插入消息中，干扰了工作流。**Delicatebobster** 和 **khansayab** 讨论了一个临时解决方案，即指令模型不要使用 **Artifacts**。
  - **Context** 使用问题受到关注，**extopico** 描述了让 **Claude 3.5** 准确遵循提示词的困难，且客户支持毫无帮助。**Khansayab** 表示赞同，分享了对模型性能的挫败感。


---

# AI Discord 摘要

> 由 O1-mini 生成的摘要之摘要的摘要

**1. AI 模型创新与发布**

- [**Ferret-UI 发布首个以 UI 为中心的 MLLM**](https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b): Nous Research 推出了基于 **Gemma-2B** 和 **Llama-3-8B** 构建的 **Ferret-UI**，在处理复杂 UI 交互的 **referring**、**grounding** 和 **reasoning** 任务中表现出色，超越了包括 **GPT-4V** 在内的现有模型。
- [**Ollama 发布 Llama 3.2 Vision**](https://ollama.com/library/llama3.2-vision): **Ollama** 推出了 **11B** 和 **90B** 规格的 **Llama 3.2 Vision**，分别需要 **8GB** 和 **64GB** 的 VRAM，增强了 **text-to-3D** 和 **image-to-3D** 生成能力。
- [**专用 Transformer ASIC Sohu 亮相**](https://x.com/rohanpaul_ai/status/1854326252674384129): **Sohu** ASIC 是首款专用的 Transformer 芯片，承诺运行 AI 模型比 **GPU 快 10 倍**，吞吐量超过 **500,000 tokens/second**，具备 **multicast speculative decoding** 和实时内容生成功能。

**2. 性能优化与资源管理**

- [**通用 JSD 内核提升效率**](https://github.com/HazyResearch/ThunderKittens?tab=readme-ov-file#demos): Chun-Chih Tseng 开发了一种**通用 JSD kernel**，在 **128k vocab size** 下实现了 **1.5 倍的加速**和 **50% 的峰值显存降低**，并增强了对 **phi**、**qwen** 和 **llama-vision** 的支持。
- [**8-bit 量化标准化 GPU 使用**](https://arxiv.org/abs/2411.03923): **8-bit quantization** 正成为标准，通过优化存储而不降低模型性能，允许用户利用 **2 倍以上的 GPU**，实现了从传统 **32-bit** 方法的转变。
- [**Flash Attention 梯度技术探索**](https://arxiv.org/html/2405.17399v1): 关于推导 **Flash Attention** 模型前向梯度的讨论促成了基础公式的分享和协作方法，以推进梯度计算，从而增强模型训练。

**3. 平台与工具集成**

- [**Nous Chat 增强 Hermes 3 界面**](https://hermes.nousresearch.com): **Nous Research** 推出了 [**Nous Chat**](https://hermes.nousresearch.com)，这是 **Hermes 3 70B** 的新用户界面，提供**推理增强**、新模型和实验性功能，以优化用户交互。
- [**OmniParser 集成 LLM 进行 UI 解析**](https://huggingface.co/microsoft/OmniParser): [**OmniParser**](https://huggingface.co/microsoft/OmniParser) 模型将 UI 截图转换为结构化格式，通过利用 **YOLOv8** 和 **BLIP-2** 进行可交互图标检测和 UI 元素描述，增强了基于 **LLM** 的 **UI Agent**。
- [**Codebuff CLI 工具简化代码生成**](https://codebuff.com): [**Codebuff**](https://codebuff.com) 提供了一个根据自然语言请求编写代码的 CLI 工具，与 **OpenAI** 的 **GPT-4o** 无缝集成，为代码修改生成有效的 git patches。

**4. 不同领域的 AI 应用**

- [**YouTube 摘要生成器利用 Whisper 和 PyTube**](https://pytube.io/): 一个正在开发的项目旨在创建一个 **YouTube 摘要生成器**，该工具根据视频内容启动**交互式聊天会话**，使用 **PyTube** 进行视频处理并使用 **Whisper** 进行转录，旨在提高信息获取的便捷性。
- [**Formula1 遥测聊天机器人分析比赛数据**](https://huggingface.co/spaces/Draichi/Formula1-race-debriefing): 推出了一款 AI 驱动的 [**Formula1 遥测聊天机器人**](https://huggingface.co/spaces/Draichi/Formula1-race-debriefing)，用于分析真实比赛遥测数据并生成详细报告，结合了 **text-to-SQL** 技术来查询各种比赛参数。
- [**葡萄叶病害检测应用推进农业 AI**](https://huggingface.co/spaces/thesab/Grape-Leaf-Disease-Detection-App): 一款全新的 [**葡萄叶病害检测应用**](https://huggingface.co/spaces/thesab/Grape-Leaf-Disease-Detection-App) 展示了 AI 在农业中的应用，通过图像分析实现植物病害的早期检测和管理。

**5. AI 微调与定制化**

- [**Cohere 发布开源微调库**](https://github.com/cohere-ai/cohere-finetune)：**Cohere** 推出了 `cohere-finetune`，这是一个**开源微调库**，集成了 **Hugging Face 的 PEFT 库**，允许使用自定义数据集进行模型定制，并通过 **Amazon SageMaker** 部署增强隐私和合规性。
- [**DSPy 通过 Embedding Momentum 增强微调**](https://github.com/leloykun/modded-nanogpt/tree/fc--momentum-cooldown/records/110724_EmbeddingBetasCooldown)：**DSPy** 代码库的修改引入了 **embedding momentum** 和 **splitting lambdas**，改善了 **NaNoGPT** 等模型的微调结果，并计划进行进一步测试以验证增强效果。
- [**为 LLM 微调添加 Special Tokens**](https://github.com/EleutherAI/nanoGPT-mup)：在 **LLM** 微调中添加新 **special tokens** 的最佳实践包括更新 tokenizer 并将其包含在配置中。**LORA** 虽然有效，但效果不如全量微调（full fine-tuning），因此需要保存 `embed_tokens` 和 `lm_head` 等模块以获得最佳训练结果。


---

# 第 1 部分：高层级 Discord 摘要




## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Ferret-UI：开创性的以 UI 为中心的 MLLM**：Nous Research 推出了 [Ferret-UI](https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b)，这是首个**以 UI 为中心的多模态大语言模型（MLLM）**，基于 **Gemma-2B** 和 **Llama-3-8B** 构建，专为复杂的 UI 任务设计。
   - Ferret-UI 在**指代（referring）**、**定位（grounding）**和**推理**任务方面表现出色，显著增强了与移动 UI 屏幕的交互，并在基础 UI 任务上超越了现有的 UI MLLM 和 **GPT-4V**。
- **Haiku 3.5 表现逊于 GPT-4**：成员观察到 **Haiku 3.5** 的性能与 **8-14B** 范围的小型模型相似，隐藏参数大小与效能之间可能存在联系。
   - 相比之下，**GPT-4** 展示了更优越的结果，引发了关于模型缩放和参数优化的讨论。
- **Nous Chat 发布先进的 Hermes 3 界面**：**Nous Research** 推出了 [Nous Chat](https://hermes.nousresearch.com)，这是一个为 **Hermes 3 70B** 设计的新**用户界面**，提供**推理增强**、新模型和实验性功能。
   - 该平台旨在成为体验 Hermes 的首选目的地，并持续收集用户反馈和错误报告以改进其功能。
- **Hermes 405B 表现出性能波动**：社区报告指出 **Hermes 405B** 经历了延迟和命令响应失败，尽管它已在 **OpenRouter** 上恢复运行。
   - 讨论重点集中在增强功能上，如改进**音频集成**和引入**标注数据（labeled data）**以提升功能。
- **利用 Whisper 开发 YouTube 摘要生成器**：一名成员正在开发一个 **YouTube 摘要生成器**，该工具根据视频内容启动**交互式聊天会话**，利用 [pytube](https://pytube.io/) 进行视频处理，并使用 **Whisper** 进行转录。
   - 挑战包括 *bart-cnn* 模型的摘要准确性，这促使人们寻求增强聊天会话交互的策略。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 扩展美国教育折扣**：**Perplexity Pro 订阅**目前仅向**美国大学**提供折扣价格，这引发了关于可能扩展到其他地区的讨论。[用户确认](https://discord.com/channels/1047197230748151888/1047649527299055688/1303813131566452858)了目前在资格上的限制。
   - 一位失望的用户询问了**教育折扣**在海外（美国以外）推出的时间表，凸显了社区对更广泛访问权限的兴趣。
- **Claude 模型表现出 GPT-4o 行为**：多位用户报告称，选择 **Claude 模型**后，输出结果类似于 **GPT-4o**，这表明可能存在 Bug。该问题已在社区内得到确认。
   - 开发者已收到通知，但参与者称解决 **Claude 模型**差异问题的进展较为缓慢。
- **切尔诺贝利的真菌与按钮回归**：讨论强调了**切尔诺贝利食辐射真菌**的作用以及近期技术更新中**实体按钮的回归**。这种交集展示了在挑战性环境中的**创新适应**。
   - 通过这些发展实现的自然与技术的融合引起了社区的兴趣，暗示了在韧性工程（resilience engineering）中的潜在应用。
- **AI 演进的前景**：对话集中在 **AI 的未来**，成员们分享了关于预期进展的各种讨论链接。重点仍然在于 AI 技术将如何**改变多个行业**。
   - 成员们就 **AI 增长**的轨迹交换了见解，强调了未来的机遇与挑战。
- **唱片机的顶级音频设备**：一位用户介绍了一个**资源页面**，专门用于识别唱片机最具性价比的**扬声器和放大器**，旨在帮助他人优化音频设置。[该页面](https://discord.com/channels/1047197230748151888/1054944216876331118/1303826095035908128)整合了建议以简化音频升级流程。
   - 社区赞赏这种对性能的关注，且没有揭露过去的问答尴尬，为音频爱好者营造了协作环境。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Hermes 复苏**：**Hermes** 在经历了一段动荡时期后显示出复苏迹象，现在的响应时间在 **3 到 8 秒**之间。
   - 虽然部分用户仍能感受到延迟，但社区对持续的改进表示乐观。
- **Completion API 迁移提升性能**：所有 **Completion API** 请求已迁移到**重新编写的新 API**，增强了性能，预计速度会**更快**。
   - 鼓励用户在指定的支持频道报告任何问题。
- **Claude API 变更导致访问问题**：用户报告在通过 **OpenRouter API** 访问 **OpenAI** 模型时收到 `unsupported_country_region_territory` 错误。
   - 几位用户认为此问题可能与迁移到 Cloudflare Workers 影响端点响应有关。
- **Mistral 推出新 API**：**Mistral** 推出了两个新 API：一个**审核工具**和一个 **Batch API**，后者的处理成本比同步调用**低 50%**。
   - 这一举措展示了 Mistral 在行业 API 成本上升背景下，致力于提供负担得起的、可扩展的解决方案。
- **OpenRouter API 的 URL 格式问题**：多位用户在使用 **OpenRouter API** 时遇到 **404 错误**，通常是因为 API URL 中多了一个 '/'。
   - 讨论强调了近期 API 严格性的变化，导致了用户以前未曾遇到的问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Flash Attention 技术探讨**：一位用户咨询了如何推导 **Flash Attention** 的前向梯度，并分享了普通注意力相对于 **Q** 的前向梯度基础公式：`e^(q+ϵ)k/rowsum(e^(q+ϵ)k)`。
   - 他们对计算的后续步骤表示不确定，引发了社区成员对进一步开发潜在方法的讨论。
- **评估评估数据污染**：强调了理解基准测试中**评估数据污染 (evaluation data contamination)** 的重要性，并介绍了 [ConTAM 方法](https://arxiv.org/abs/2411.03923) 以更高效地评估此问题。
   - 正如 AI 工程师们所讨论的，该方法解决了确定受污染样本及其对基准测试分数影响的复杂性。
- **NaNoGPT 获得代码库增强**：一位用户分享了对 **NaNoGPT** 代码库的修改，详细介绍了最近关于 embedding momentum 和 splitting lambdas 的实验，可在 [GitHub](https://github.com/leloykun/modded-nanogpt/tree/fc--momentum-cooldown/records/110724_EmbeddingBetasCooldown) 上查看。
   - 他们得出结论认为其样本量较小，并计划进行进一步测试以明确所实现的改进。
- **NeoX vs LitGPT：基准测试之战**：成员们正在咨询比较 **NeoX** 和 **LitGPT** 框架性能差异的 **benchmarks**，重点关注训练速度和稳定性。
   - 讨论指出了一种趋势，即许多用户在缺乏明确、有证据支持的对比情况下，更倾向于基于 **LitGPT** 的设置。
- **Magenta 的 Music Transformer 展示**：分享了对 **Magenta** 的 **Music Transformer** 的引用，重点介绍了其通过 [Listen to Transformer](https://magenta.github.io/listen-to-transformer/#a1_6828.mid) 应用生成音乐表演的开源模型。
   - 通过对比展示了自发布以来音乐生成模型的进步。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **微调 Smollm2 面临输出问题**：用户报告了在微调 **Smollm2** 时遇到的持续问题，具体表现为尽管数据集包含 **eos token**，但输出仍无法终止。开发者正在与 **HF** 合作解决该模型错误。
   - 建议升级到 **transformers 4.46** 并使用 **resume_from_checkpoint** 以改善微调结果。
- **模型间的 VRAM 消耗差异**：显存 (**VRAM**) 消耗的显著差异引发了关注，**Aya 8B** 模型使用 **22GB**，而 **Llama3.2 3B** 模型在未量化的情况下使用了 **43GB**。
   - 参与者讨论认为，由于 **16-bit precision standards**，较大的模型通常需要更多 **VRAM**，这导致了资源使用上意想不到的差异。
- **8bit 和 4bit 支持即将推出**：用户对预计在本月内推出的 **8bit** 和 **4bit** 支持表示兴奋，并询问了对 **fp8** 或 **int8** 的支持情况。
   - 分享了一篇相关的[论文](https://link.to.paper)，以帮助社区了解预期的增强功能。
- **增强 torch.compile 以支持 Gradient Checkpointing**：一位成员强调需要通过移除 **torch._dynamo.disable** 使 **torch.compile** 与 gradient checkpointing 兼容，并表示有兴趣为此做出贡献。
   - 他们在 **torch compile** 方面的经验被认为对于解决 Wiki 中的待办事项非常有价值。
- **AI Unplugged 通讯提供最新见解**：最新一期的 **AI Unplugged** 涵盖了 **RoPE**、**Mamba** 的改进以及会下棋的 Transformer 等主题，吸引了社区的极大兴趣。
   - 核心观点强调了 **RoPE** 对模型适应性的重要性以及 position embeddings 的潜在增强，可通过 [AI Unplugged 22](https://datta0.substack.com/p/ai-unplugged-22-rope-internals-explained) 访问。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **使用 Serverless Inference 优化 Hermes3**：一位用户在为 **Hermes3** 设置 [serverless inference endpoint](https://huggingface.co/docs/hub/spaces-config-reference) 时遇到了挑战，特别是质疑部署时是否必须输入信用卡。
   - 社区成员澄清了 Serverless 选项的可用性，但指出了在模型链接以及成功创建 API 的必要步骤方面存在不确定性。
- **发布 Hunyuan3D-1 框架**：**Tencent** 发布了 [Hunyuan3D-1.0](https://huggingface.co/tencent/Hunyuan3D-1) 框架，支持 **text-to-3D** 和 **image-to-3D** 生成，并为每种格式提供了演示。
   - 在 **2024 年 11 月 5 日**，他们提供了 [代码仓库](https://github.com/tencent/Hunyuan3D-1) 和详细 [报告](https://arxiv.org/pdf/2411.02293) 的访问权限，以及用于执行演示的脚本。
- **开发 Formula1 遥测聊天机器人**：推出了一款 AI 驱动的 [Formula1 遥测聊天机器人](https://huggingface.co/spaces/Draichi/Formula1-race-debriefing)，用于分析真实赛车遥测数据并生成详细报告。
   - 该工具集成了 **text-to-SQL** 功能，允许用户查询各种比赛参数，从而增强了车迷和车队获取洞察的便利性。
- **转换 TinyLlama 模型架构**：实现了 **TinyLlama 模型** 架构的重大转换，重点在于 **differential attention** 和 token mixing，并公开了 [转换脚本](https://huggingface.co/Josephgflowers/Differential-Attention-Liquid-Metal-Tinyllama)。
   - 提供了全面的文档，指导在修改后的解码器层中集成各种模块，从而促进更广泛的采用和实验。
- **集成 OmniParser 进行 UI 解析**：展示了 [OmniParser](https://huggingface.co/microsoft/OmniParser) 模型，作为将 UI 截图转换为结构化格式的工具，从而增强基于 **LLM** 的 **UI Agent**。
   - 它利用了经过微调的 **YOLOv8** 和 **BLIP-2** 版本，这些版本在专为可交互图标检测和 UI 元素描述设计的数据集上进行了训练。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **SearchGPT 在智能查询上受挫**：用户担心 **SearchGPT** 的能力不如默认模型且更加“固执”，在处理广泛查询时表现吃力，并且经常产生 **hallucinating**（幻觉）答案，而不是承认无法找到答案。
   - 一位成员强调纠正措施没有得到妥善整合，并指出 **SearchGPT** 倾向于持续 **重复答案**。
- **Custom GPT 功能期待升级**：成员们正期待 **Custom GPT 功能** 的增强，特别是 **文件大小限制** 的扩大和 **文件上传能力** 的增加。
   - 他们表达了对 OpenAI 正在为 **Custom GPT** 功能准备重大 **改进** 的希望，并对外部的积极进展进行了反思。
- **丢失 GPT 触发侧边栏遗憾**：一位用户报告丢失了保存在侧边栏的大约 **20 个 GPT**，正在寻求潜在原因。
   - 他们询问：“最近是否发生了导致这种情况的事情？”，表明需要进行调查。
- **AI 自我意识引发辩论**：讨论围绕 **ChatGPT** 和 **Claude** 等 AI 是否能表现出 **自我意识** 展开，一些人暗示可能存在 **自我保护** 行为。
   - 用户辩论了 AI 发展出 **类人驱动力** 的风险，并考虑到 **LLM** 的输出可能反映了底层的 **inference** 能力。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **广义 JSD Kernel 实现 1.5 倍加速**：Chun-Chih Tseng 开发了一个 **广义 JSD Kernel**，在 128k 词表大小下实现了 **1.5 倍速度**提升和 **50% 的峰值内存减少**，同时实现了 **LigerCrossEntropy** 的相关功能。
   - Tyler Romero 增加了对 **phi**、**qwen** 和 **llama-vision** 的支持，而其他贡献者也进行了额外的 Kernel 增强以优化性能。
- **Project Popcorn 启动 SOTA Kernel 生成**：一位成员分享了 [Project Popcorn](https://gist.github.com/msaroufim/087c2a358c505e287a926e6a27b3e3b0)，旨在公共空间利用 **LLM 生成 SOTA Kernel**，以促进社区参与和透明度。
   - 自动化部署现已在 **Heroku** 上线，使得 Bot 可以通过向 main 分支推送更改来进行更新，并计划在获得 **GPU** 后连接到服务器。
- **A100 GPU FP16 性能见解**：一次讨论揭示了在 **A100** 等**数据中心 GPU** 上，使用 **FP16 累加的 FP16 x FP16** 并没有提速，因为它们共享相同的 flops。
   - 相反，这种组合仅在**消费级显卡**上更快，这使得企业级 GPU 在使用 **FP32 累加**时能保持性能而不降速。
- **ThunderKittens 贡献列表更新**：成员们注意到 **ThunderKittens** 项目缺少新手贡献列表，促使一位成员在 GitHub 上分享了一个[初步列表](https://github.com/HazyResearch/ThunderKittens?tab=readme-ov-file#demos)。
   - 提供了添加**长卷积 Kernel** 的协助，包括提供 **PyTorch 参考**，以帮助新人有效地开始贡献。
- **为初学者分享 GEMM 优化资源**：一位最近毕业的计算机科学专业学生正在寻找 **GEMM 优化**和 Kernel 优化的资源，建议包括专注于 **CUDA** 和优化技术的文章及 [GitHub 仓库](https://siboehm.com/articles/22/CUDA-MMM)。
   - 分享的资源如 [CUTLASS 教程](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) 和 [CUDA Matmul Kernel 优化](https://siboehm.com/articles/22/CUDA-MMM) 提供了增强矩阵乘法性能的深入指导。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **澄清播客重用政策**：针对播客重用政策提出了咨询，特别是关于在 [GitHub 仓库](https://github.com/robbiemu/llama-gguf-optimize)中分享的内容。
   - 成员们旨在确保在利用播客材料前符合指南，强调了对政策理解清晰的必要性。
- **NotebookLM 性能问题**：用户报告称 **NotebookLM** Bot 会互相接话，导致重复对话和无法使用的体验。
   - 此外，还讨论了在各种移动浏览器中滚动保存笔记的挑战，促使用户寻找有效的解决方法。
- **从 Google Drive 集成 PDF**：成员们对无法直接从 [Google Drive](https://discord.com/channels/1124402182171672732/1124403655819415592/1303814357783810110) 将 PDF 加载到 **NotebookLM** 表示失望。
   - 他们认为增加此功能对于增强集成能力至关重要，尤其是在投入资金增加存储空间之后。
- **用于 TOS 教育的 YouTube 频道**：有建议创建一个专门剖析大公司服务条款（TOS）和隐私政策的 YouTube 频道。
   - 成员们认为这个想法很有价值，指出此类内容非常罕见，且通过引人入胜的演示潜力来提高理解力。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Anthropic-Palantir-AWS 国防 AI 合作伙伴关系**：Anthropic 已与 [Palantir](https://www.businesswire.com/news/home/20241107699415/en/Anthropic-and-Palantir-Partner-to-Bring-Claude-AI-Models-to-AWS-for-U.S.-Government-Intelligence-and-Defense-Operations) 及 Amazon Web Services 达成合作，为美国情报和国防机构提供其 **Claude** AI 模型的访问权限。
   - 这一举措反映了在国家安全领域对 AI 解决方案需求日益增长的背景下，其他科技公司争取**国防合同**的努力。
- **量化技术与 GPU 效率**：**8-bit quantization** 正被采纳为模型使用的标准，在不降低性能的情况下优化了存储。
   - 这种从传统的 32-bit 方法的转变允许用户有效地利用 **2x 更多的 GPUs**，显著增强了计算能力。
- **合成数据生成与 SFT 扩展**：最近的一篇论文利用了 **1.5T tokens** 的合成数据以及 **100 万**条 SFT 数据样本。
   - *这是否意味着在预训练期间使用了指令数据？* 这种情况引起了人们对与 **T0 model** 训练策略相似性的关注。
- **Character.AI 的推理优化**：[Character.AI](https://research.character.ai/optimizing-inference/) 正在通过优化推理来迈向 AGI，使用 **int8 quantization** 实现每秒处理超过 **20,000 次查询**。
   - 他们的方法背离了传统的训练后量化（post-training quantization），专注于提高训练效率。
- **Tim 转职至 CMU**：Tim 已前往 **Carnegie Mellon University (CMU)** 并正在远程工作，社区成员对其贡献表示感谢。
   - 成员们希望 Tim 在 **2025** 年能有更多的合作和积极参与。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Ollama 发布 llama 3.2 Vision**：Ollama 发布了 **llama 3.2 Vision**，增强了其模型能力，而 **MLX** 虽然提供类似功能，但在 **llama.cpp** 中仍缺乏支持。
   - 关于将 **llama 3.2 Vision** 集成到 **LM Studio** 的担忧被提出，一名用户在模型部署期间遇到了加载错误。
- **MLX Engine 更新支持视觉功能**：一个 [GitHub pull request](https://github.com/lmstudio-ai/mlx-engine/pull/22) 概述了 **MLX Engine** 支持 **llama 3.2 Vision** 的更新。
   - 社区对即将到来的增强功能持乐观态度，期待更新部署后功能得到改进。
- **单槽 RTX 4090 引起关注**：**Single Slot RTX 4090** 因其紧凑的设计和对小尺寸机箱（small form factor）构建的适用性而受到关注。
   - *“老兄，你为冬天做好了准备，”* 一位用户评论道，强调了该显卡有效的散热能力。
- **Mac M2 Pro 内存占用过高**：用户报告称，**Mac M2 Pro** 在处理 **10-12K tokens** 的 **8B model** 时消耗了约 **20GB** 内存。
   - 虽然有人确认“上下文（context）会占用内存”，但高内存使用比例在社区中仍是一个令人担忧的问题。
- **大模型性能优化**：关于运行 **70B** 模型的讨论集中在优化 **context size** 配置上。
   - 用户正在评估 **context scaling** 对整体模型性能和准确性的影响。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 缺乏 Web UI 生成能力**：一位用户询问用于生成 Web UI 的模型，但另一位用户指出 **Stable Diffusion 主要用于图像**，而非网页界面。
   - 对话强调了当前 **Stable Diffusion 模型**在特定设计应用中的局限性。
- **使用 ComfyUI 和 SwarmUI 进行本地安装**：一位新用户寻求从 Google Colab 转向**本地设置 Stable Diffusion** 的指导。
   - 一名成员推荐了一份安装 **ComfyUI** 并使用 **SwarmUI** 作为前端的指南。
- **外绘（Outpainting）技术与资源**：用户交流了关于 **outpainting techniques** 的链接和资源，包括 Reddit 帖子和运行 **Automatic1111** 的教程。
   - 成员们分享了关于设置和功能的具体指导，以实现成功的外绘效果。
- **使用 Stable Diffusion 生成 LinkedIn 图像**：一位用户寻求关于训练模型以生成其 **LinkedIn 个人资料**真实图像的建议。
   - 社区成员讨论了合适的选项，但强调 **Stable Diffusion** 主要针对艺术图像生成而设计。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Ollama 发布 Llama 3.2 Vision**：[Llama 3.2 Vision](https://ollama.com/library/llama3.2-vision) 现已推出 **11B** 和 **90B** 规格，为获得最佳性能，分别需要 **8GB** 和 **64GB** 的 VRAM。
   - 用户可以通过下载 [Ollama 0.4](https://ollama.com/download) 并使用简单的终端命令轻松运行该模型。
- **Aide IDE：AI 开发领域的新选手**：Y Combinator 宣布了 [Aide](https://x.com/ycombinator/status/1854237314651980257)，这是一个基于 Agent 框架构建的开源 AI 原生代码编辑器，在 swebench-lite 上拥有 **43%** 的性能表现。
   - 该工具承诺完全的数据隐私和即插即用的 LLM 集成，吸引了寻求强大编码解决方案的开发者。
- **Claude 的免费用户限制**：**Claude** 的免费用户目前仅限于执行 Haiku 等基础任务，无法执行分析大型 CSV 文件等更复杂的操作。
   - 成员们对这些限制表示沮丧，认为这阻碍了他们利用 AI 进行实质性工作的能力。
- **探索开放语言模型的未来**：讨论了如何开发更好的系统来训练开放语言模型和 Agent，并特别提到了 Tim Dettmers 的见解。
   - 重点强调了克服“**API 成瘾**”，以在 AI 生态系统中实现更多创新。
- **Codebuff CLI 工具介绍**：[Codebuff](https://codebuff.com) 是由 Y Combinator 推出的 CLI 工具，可根据自然语言请求编写代码，并提供无需登录的免费试用。
   - 创始人分享了一个有趣的开发故事，涉及微调 GPT-4o 以生成用于有效代码修改的 git patches。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **讨论替换无边界检查装饰器**：社区讨论了将 `@no-bounds-check` 装饰器替换为 `@unsafe_no_bounds_check`，更倾向于使用 **SIMD loads** 以获得更好的性能。
   - 一位成员指出，列表边界仅在启用断言的编译期间增加开销。
- **提议为 Mojo 标准库提供图形化概览**：一位成员提议在 Modular Mojo 网站上创建一个[图形化页面](https://docs.modular.com/mojo/roadmap)，以展示 **Mojo 标准库** 的进展以及与 Python 和 C/C++ 的互操作性。
   - 该页面旨在为贡献者提供可用标准库模块及其状态的全面视图，类似于路线图。
- **关于 Mojo 是否为 Python 超集的辩论**：社区辩论了 Mojo 作为 Python “软超集”的定位，担心采纳 Python 的缺陷可能会适得其反。
   - 成员们讨论了支持各种 Python 行为的挑战，并指出对于**互操作性**至关重要的细微差别。
- **在 Mojo 中导入 C 模块需要链接**：澄清了在 Mojo 中导入 C 模块仍然需要链接，这与希望简化导入语法的愿望相反。
   - 一项建议包括开发一个名为 `mojo` 的 Python 库，以简化 Mojo 模块的导入，类似于 **NumPy** 等库。
- **未来的 Mojo 特性与互操作性增强**：成员们对增强 Mojo、Python 和 C/C++ 之间的**互操作性**表示乐观，目标是在无需过度链接的情况下实现平滑导入。
   - 讨论强调了在 Python 中使用之前，需要将 Mojo 库编译为共享对象或 DLL。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Reranker API 现在仅限 API 调用**：*mrdragonfox* 确认 **Cohere Reranker** **仅通过 API 提供**，未列入版本 1 和 2 的文档中。
   - *kenb_80283* 指出 endpoints 部分需要更新。
- **Command-R-Plus 表现出异常行为**：*guestavius* 报告称，在 **Command-R-Plus** 中高频出现随机的 **'section'** 插入，这在以前不是问题。
   - *mrdragonfox* 表示该工具主要不是为 **roleplay** 设计的，强调其企业级应用。
- **AWS Bedrock Embeddings 是否保留输入顺序？**：*boliveira5781* 询问 AWS Bedrock embed endpoint 生成的 **embeddings** 是否与输入字符串保持 **order-preserving**（保序）映射。
   - *enzoloko* 质疑添加新字符串是否会影响现有字符串的位置。
- **Cohere 发布开源 Fine-tuning 项目**：Cohere 发布了一个名为 `cohere-finetune` 的 **开源 fine-tuning 仓库**，包括详细指南和预构建容器，用于使用自定义数据集将基础模型适配到特定任务。
   - 在 [GitHub](https://github.com/cohere-ai/cohere-finetune) 上查看，以便轻松进行模型定制。
- **Hugging Face 与 SageMaker 集成用于 Fine-tuning**：新的 fine-tuning 仓库集成了 **Hugging Face 的 Parameter-Efficient Fine-Tuning** 库，以在无需大量资源需求的情况下优化模型性能。
   - Cohere 在 **Amazon SageMaker** 上提供“自带微调模型”（Bring Your Own Fine-tune）推理解决方案，允许在增强隐私、安全性和合规性的情况下部署 fine-tuned 模型。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **自动化 Resume Insights Agent 创建**：[Luillyfe](https://twitter.com/Luillyfe) 的教程解释了如何使用核心解析、提取和结构化输出模块构建 **Automated Resume Insights** agent。
   - 该系统能高效处理任何非结构化简历，提供深入的数据收集。
- **通过 Context Refinement 增强 RAG 系统**：一篇客座博客文章讨论了构建 **Context Refinement Agent**，该 Agent 能智能地扩展和细化检索到的上下文，从而在处理复杂查询时获得更好的 RAG 响应。
   - 该 Agent 检查检索到的块以提高输出质量，为数据检索和处理增添了新维度。
- **Ollama Llama Vision 可能与 Llama Index 集成**：一位用户询问新的 **Ollama Llama Vision** 功能与 **Llama Index** 的兼容性，假设它可以与 **OllamaMultiModal class** 配合使用。
   - 另一位成员澄清说 **Ollama 早就具备 vision** 功能，表明其具有历史集成性。
- **寻找开源 Chatbot UI**：一位用户请求一个开源的聊天机器人 Web 应用，具有身份验证和类似于 **ChatGPT** 的 UI。
   - 成员们推荐了 [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui)，并强调了其功能和用例。
- **构建类似 Llama-Parse 解析器的资源**：一位成员请求构建类似于 **Llama-Parse** 的解析器的资源，强调数据安全和本地模型使用。
   - 建议包括 **Unstructured** 库，但指出它无法达到 **Llama-Parse** 的功能水平。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Dott.ai 宣布未来计划**：一位成员分享了 [Dott.ai 的未来计划](https://dottxt.co)，强调了其在行业中的重要地位。
   - 来自 Builder.io 的 Steve 通过声明 *这是未来* 肯定了这一愿景，强调了该项目的潜力。
- **DSPy 框架面临 Docstring 不匹配问题**：一位用户报告说，在 **DSPy** 中，由于使用了 `f"""` 而不是 `"""`，导致仅显示第一个组件的 docstring。
   - 这种格式问题导致用户在正确提取 docstring 方面产生了困惑。
- **EMNLP 2024 上的 DSPy 演示**：一篇 DSPy 相关论文的共同第一作者将在 **EMNLP 2024** 上展示他们的工作，引起了社区的兴趣。
   - 用户表达了在会议期间与作者建立联系并讨论其研究的热情。
- **模块化语言模型中的优化策略**：分享了两篇论文的链接，概述了优化模块化语言模型流水线的策略，重点关注权重和 prompt 优化方法。
   - 这些论文解决了 NLP 系统中在没有中间标签或梯度的情况下高效处理模块的挑战。
- **社区对 DSPy 的赞赏**：一位用户称赞了 **DSPy** 项目取得的进展，强调了团队令人印象深刻的贡献。
   - 他们的热情表明了对进一步参与项目发展的浓厚兴趣。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **理解 Claude 的 OS Mode**：一位用户寻求关于 **OS mode** 如何与 **Claude** 配合工作的澄清，询问 prompt 是否被转换为代码来控制桌面以及点击是如何协调的。另一位成员提供了一个 [GitHub 链接](https://github.com/OpenInterpreter/open-interpreter/blob/development/computer_use/tools/computer.py)，详细说明了负责鼠标点击的代码。
- **Discord 活动时间困惑**：一位用户询问即将举行的活动是否定在 **8 PM GMT**，而另一位成员根据本地时间设置确认活动将在 **30 分钟** 后开始。活动链接的提及表明社区参与正在进行中，尽管未给出具体细节。
- **直播观众限制**：有人提问关于直播是否存在最大观众人数限制，一位成员自信地回复说**不应该有**任何限制。这种保证反映了社区对容纳大量观众观看直播内容的兴趣。
- **关于 OmniParser 工具的讨论**：一位用户推荐了 [OmniParser](https://huggingface.co/microsoft/OmniParser)，这是一款屏幕解析工具，通过将截图转换为结构化格式来提高 UI agent 的性能。他们引用了一篇 [博客文章](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/) 和一个 [demo](https://huggingface.co/spaces/microsoft/OmniParser/)，表示对其在 Open Interpreter 中应用的兴趣。
- **Python 3.13 兼容性问题**：一位用户因其 **Python 3.13** 环境与该包所需的版本不兼容而遇到安装错误。*被忽略的版本*包括几个要求 Python 版本在 **3.11 到 4.0** 之间的版本，强调了版本特定性的必要性。
   - 该用户创建了一个 Python **3.11** 的 **conda environment**，从而成功安装了该包，尽管据称其运行*速度没那么快*。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **专用 Transformer ASIC 发布**：一位成员宣布推出了首款专用 Transformer ASIC —— **Sohu**，其运行 AI 模型速度比 **GPU 快 10 倍**，吞吐量超过 **500,000 tokens/second**。
   - 正如 [Rohan Paul 的推文](https://x.com/rohanpaul_ai/status/1854326252674384129) 所分享的，**Sohu** ASIC 具备 **multicast speculative decoding** 和 **real-time content generation** 功能，将其定位为为 AI 定制的“高速公路”。
- **定制硬件可用性受质疑**：成员们质疑 AI 模型**定制硬件**的可用性，引用了六个月前的一篇 [博客文章](https://link.to.blog)，该文章暗示产品尚未上市。
   - 有人担心这种情况具有 **Theranos vibe**（Theranos 既视感），对定制硬件解决方案的实际存在与承诺的功能表示怀疑。
- **高效的多 GPU 利用**：一位成员询问如何在**多个 GPU** 上并行运行模型的多个副本，以在不使用 model sharding 的情况下提高吞吐量，但在使用 `concurrent.futures.ThreadPoolExecutor` 时遇到了 tensor 加载锁定问题。
   - 提出的解决方案包括使用 `x.shard(GPUS, axis=None)` 在 GPU 之间复制模型，以及使用 `x.shard(GPUS, axis=0)` 来高效地切分输入。
- **ThreadPoolExecutor 锁定问题**：据报告，在多 GPU 操作期间加载 tensor 时，`concurrent.futures.ThreadPoolExecutor` 会导致锁定挑战。
   - 建议使用 `x.shard(GPUS, axis=None)` 和 `x.shard(GPUS, axis=0)` 等替代方案来规避这些问题并提高并行处理效率。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **ScheduleFree SOAP 的优势**：[ScheduleFree SOAP 的实现](https://github.com/ClashLuke/HeavyBall/blob/main/heavyball/schedule_free_palm_foreach_soap.py#L296)具有更高的计算和内存效率，通过允许更高的学习率（learning rates）来实现更快的收敛。
   - 与 SOAP/Adam 相比，它建议更改超参数，例如使用 PaLM 的 beta2 调度方案并进行 10% 的预热（warmup）。
- **关于 MOEs 和模型合并（Merging Models）的讨论**：一位成员询问了关于 MOEs 或模型合并的后续工作，指出自 llama 3.2 以来这些内容一直缺失。
   - 另一位成员观察到，目前的讨论主要集中在 llama 3.2 的微调（finetunes）上。
- **ScheduleFree SOAP 与 CAME 优化器（Optimizer）的比较**：一位用户询问 ScheduleFree SOAP 与 [CAME 优化器](https://github.com/yangluo7/CAME)的对比情况。
   - 另一位成员澄清说 CAME 是一个不同的优化器，并提供了其官方实现的链接。
- **为微调添加特殊标记（special tokens）的正确方法**：要为 LLM 微调添加新的 **special token**，请在训练前将该 token 添加到 tokenizer 中，并在 Axolotl 配置中包含 `special_tokens: reference_text: <|reference_text|>`。
   - 成员们确认了这种方法，并强调即使使用 LORA，模型也会学习新的 token。
- **LORA 在学习新 token 方面的有效性**：一位成员表示，虽然模型会通过 **LORA** 学习新 token，但效果不如进行全量微调（full fine-tuning）。
   - 此外，使用 LORA 时，保存 `embed_tokens` 和 `lm_head` 等模块对于提高训练效果至关重要。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 的 LR Scheduler 难题**：一位用户强调了在 Torchtune 的 **full_finetune_distributed** 过程中使用 **lr_scheduler** 的问题，特别是在尝试将其添加到配置文件时。
   - 他们引用了一个开放的 [GitHub issue](https://github.com/pytorch/torchtune/issues/1308)，该 issue 讨论了计划将 LR scheduler 支持集成到全量微调（full fine-tune）的 recipes 中。
- **验证 Ichigo 的 Torchtune 集成**：一位成员分享了 [Ichigo 项目](https://github.com/homebrewltd/ichigo)，该项目利用 Torchtune 增强 **Llama3.1** 的交互性，并寻求对其实现的验证。
   - 另一位用户肯定了 Ichigo 项目中看到的 recipe 修改是可行的，并提到官方对 LR scheduler 的支持预计将在未来几周内推出。
- **通过自定义调整增强 Recipes**：讨论显示修改 recipes 是可能的，Ichigo 项目中增加的功能证明了这一点。
   - 成员们表示相信 **Torchtune** 很快将正式支持 LR scheduler 集成，从而解决当前的局限性。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **明年推出高级 LLM 课程**：一位成员确认明年将提供 **LLM 课程**，其中包括一个与当前课程内容不同的**高级版本**。
   - 这一更新强调了正在进行的**课程演进**，以满足 AI 工程师不断变化的需求。
- **明年 LLM 课程的更新材料**：即将推出的 **LLM 课程**将引入与目前涵盖内容不同的**新材料**。
   - 成员们对明年将引入的具体**高级主题**表示了兴趣。



---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **从数据集文件中提取函数**：有人建议从**数据集文件**的条目中提取 **functions** 及其定义，以编译成一份完整的列表。
   - 该提案旨在通过为 AI Engineers 提供详细的函数定义，来增强**数据集文件**的可用性。
- **缺乏编译好的函数资源**：成员们承认目前在**数据集文件**中缺乏预先编译好的 **functions** 资源。
   - 社区强调需要通过协作努力来创建此类汇编，以支持 AI 工程任务。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**LAION Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# PART 2: 按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1304136978572513335)** (1 messages): 

> - `Nous Chat`
> - `Hermes 3`
> - `User Interface Enhancements` 

- **Nous Chat 发布，旨在提供 Hermes 3 体验**：Nous Research 宣布推出 **Nous Chat**，这是一个全新的用户界面，旨在体验 **Hermes 3 70B** 及后续版本，访问地址为 [hermes.nousresearch.com](https://hermes.nousresearch.com)。
   - 该平台承诺提供**推理增强**、新模型和实验性功能，旨在成为体验 Hermes 的首选目的地。
- **鼓励反馈和 Bug 报告**：鼓励用户使用指定频道 <#1300175728121217044> 提供反馈或报告 Bug。
   - 团队期待用户提出建议，以增强新平台的整体体验。

**提到的链接**：<a href="https://hermes.nousresearch.com">NOUS CHAT | Talk to Hermes</a>：与 Nous Research 的开源 LLM —— Hermes 进行自然、智能的对话。

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1303813944795856937)** (215 条消息🔥🔥): 

> - `TEE HEE HE 钱包更新`
> - `Nous Research 概览`
> - `Hermes 405B 性能`
> - `开发讨论`
> - `AI 训练数据的未来` 


- **TEE HEE HE 钱包更新**：由于原始钱包密钥存在完整性问题（团队可以访问这些密钥），TEE HEE HE 每次都会创建新钱包。他们的最终目标是将所有余额汇总到一个持久钱包中，但这是未来的目标。
   - Kainan 确认该机器人每周都会创建一个新钱包，用户希望资产不会被销毁（burned）。
- **Nous Research 概览**：Nous Research 专注于开源 AI 研究，团队成员很乐意分享链接以便大家更好地了解他们的工作。Kainan 幽默地指出，'We'（我们）代表 Nous，他们似乎无处不在。
   - 分享了一个视频链接，以提供更多关于 Nous Research 目标和项目的背景信息。
- **Hermes 405B 性能**：用户正在讨论 Hermes 405B 的性能，特别提到了一些问题，如延迟和无法响应命令。一位社区成员报告称，Hermes 405B 确实已在 OpenRouter 上恢复运行。
   - 目前正在讨论潜在的改进，例如更好的音频集成和用于增强功能的标注数据。
- **开发讨论**：鼓励贡献者参与有关功能集成和处理新改进的讨论，但有些人还在等待更多时间来做出实质性贡献。团队预计将添加新功能，并为 Agent 架构创建一个新的子模块。
   - 贡献者表示有兴趣将架构适配到更动态的应用中，包括集成到 Minecraft 等环境中。
- **AI 训练数据的未来**：对话转向了训练数据的未来，讨论了寻找标注数据与使用现有互联网资源的挑战。参与者强调了为训练 AI 模型寻找新数据源的必要性。
   - 分享了关于使用多模态（multimodal）方法增强 AI 理解能力的看法，这需要对音频和文本数据进行仔细标注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/tee_hee_he">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://x.com/karan4d/status/1854622598375600637">来自 huh (@karan4d) 的推文</a>: @owenli25 @tee_hee_he 是的，每次重启都会创建新钱包。我们已经看到了密钥，所以每次密钥解除限制时都会出现完整性问题。目前的解决方法是创建一个新钱包...</li><li><a href="https://fxtwitter.com/kimmonismus/status/1854213806794080578?s=46">来自 Chubby♨️ (@kimmonismus) 的推文</a>: 未来的反乌托邦或乌托邦——无法确定。然而，生成式 AI 创造了令人惊叹的视频。/u/usiato</li><li><a href="https://x.com/NousResearch/status/1848397863547515216">来自 Nous Research (@NousResearch) 的推文</a>: 未找到描述</li><li><a href="https://play.ai/agent/HERMES-m3i3jU81_52ruL6_0tw2R">PlayAI - HERmes</a>: 与语音 AI 进行无缝、自然的对话</li><li><a href="https://arxiv.org/html/2405.17399v1">Transformers Can Do Arithmetic with the Right Embeddings</a>: 未找到描述</li><li><a href="https://venice.ai/">Venice | 私密且无审查的 AI</a>: 免费试用 Venice.ai。使用私密且无审查的 AI 生成文本、图像和代码。</li><li><a href="https://github.com/NousResearch/nousflash-agents/blob/main/agent/engines/long_term_mem.py#L112">nousflash-agents/agent/engines/long_term_mem.py at main · NousResearch/nousflash-agents</a>: 模块化 Agent AI 架构 - NousResearch x Teleport (Flashbots) - NousResearch/nousflash-agents
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1303813050306134017)** (17 messages🔥): 

> - `Haiku 3.5 Performance` (Haiku 3.5 性能)
> - `Haiku 3.5 Pricing` (Haiku 3.5 定价)
> - `Open Source AI Interfaces` (开源 AI 界面)
> - `Evaluating PyTorch Models` (评估 PyTorch 模型)


- **Haiku 3.5 表现似乎不尽如人意**：成员们对 **Haiku 3.5** 在评估中表现不佳表示担忧，认为其行为更接近 **8-14b** 范围的小型模型。
   - 一位用户指出隐藏参数量大小与性能之间可能存在相关性，并将其与 **GPT-4** 更好的表现进行了对比。
- **Haiku 3.5 的定价让用户感到困惑**：成员们讨论了 **Haiku 3.5** 的定价，认为与 **Gemini 1.5 Flash** 和 **GPT 4o-mini** 等价格低得多的替代方案相比，它处于一个“尴尬境地”。
   - 这种定价策略被视为一种营销赌博，用户质疑在性能存在缺陷的情况下，是否还会有人为此买单。
- **开源 AI 界面引发讨论**：一位成员询问了关于 **openwebui**、**librechat** 和 **text-generation-webui** 在本地或通过 API 使用 AI 的对比意见。
   - 有人建议将 **lmstudio** 作为替代方案，但大多数人的偏好仍倾向于开源解决方案。
- **评估 PyTorch 模型的挑战**：一位用户询问如何使用 **llm-evaluation-harness** 评估 **PyTorch models**，发现该仓库主要支持 Hugging Face 模型。
   - 对话强调了在评估 **HellaSwag** 数据集时的困惑，并呼吁提供更整洁的代码以及针对 PyTorch 兼容性的实现建议。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1304188992698449950)** (1 messages): 

> - `Ferret-UI`
> - `Gemma-2B MLLM`
> - `Llama-3-8B MLLM`
> - `UI-centered reasoning` (以 UI 为中心的推理)
> - `Mobile UI comprehension` (移动端 UI 理解)


- **Ferret-UI：首个以 UI 为中心的 MLLM 发布**：团队推出了 **Ferret-UI**，这是首个以 UI 为中心的多模态大语言模型（MLLM），基于 **Gemma-2B** 和 **Llama-3-8B** 构建，专为处理复杂的 UI 任务而设计，详见[这篇论文](https://arxiv.org/pdf/2404.05719)。
   - *它在指代（referring）、定位（grounding）和推理任务中表现出色，显著增强了与移动端 UI 屏幕的交互能力。*
- **全面的训练与采用**：要使用 **Ferret-UI**，用户需要从 [Hugging Face](https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b) 下载包括 `builder.py` 和 `inference.py` 在内的多个组件。
   - 这种方法确保了在执行需要详细理解 UI 元素的任务时，能够拥有稳健的配置。
- **优于竞争对手的出色表现**：训练完成后，**Ferret-UI** 展示了卓越的 UI 屏幕理解能力，在基础 UI 任务上超越了许多开源 UI MLLM，甚至超过了 **GPT-4V**。
   - 该模型的设计包含了一些独特功能，例如将屏幕划分为子图像以实现更好的细节识别。
- **为增强推理而定制的数据集**：**Ferret-UI** 在一个精心策划的指令遵循数据集上进行了训练，其中包括图标识别和文本查找等任务。
   - *训练方法侧重于增强模型执行与 UI 交互相关的开放式指令的能力。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b">jadechoghari/Ferret-UI-Gemma2b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/jadechoghari/Ferret-UI-Llama8b">jadechoghari/Ferret-UI-Llama8b · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1304188992698449950)** (1 messages): 

> - `Ferret-UI`
> - `Gemma-2B`
> - `Llama-3-8B`
> - `Multimodal language models`
> - `Mobile UI comprehension` 


- **Ferret-UI：新型以 UI 为中心的 MLLM**：介绍 [Ferret-UI](https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b)，这是首个专为指代（referring）、定位（grounding）和推理（reasoning）任务设计的以 UI 为中心的多模态大语言模型，基于 **Gemma-2B** 和 **Llama-3-8B** 构建。
   - *该模型在执行复杂的 UI 任务*以及更好地理解移动用户界面屏幕方面表现出卓越的能力。
- **Gemma-2B 和 Llama-3-8B 版本发布**：Ferret-UI 提供两个版本，一个基于 [Gemma-2B](https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b)，另一个基于 [Llama-3-8B](https://huggingface.co/jadechoghari/Ferret-UI-Llama8b)，以满足不同的使用场景。
   - 每个版本都需要下载特定的脚本（如 `builder.py` 和 `inference.py`），以便在本地环境中有效运行。
- **移动 UI 理解能力的增强**：该模型专门为理解移动 UI 屏幕并与之交互而定制，融合了**增强的视觉特征**和独特的训练方法。
   - 通过利用子图像编码策略，Ferret-UI 在 UI 任务中实现了优于现有模型的性能。
- **令人印象深刻的 Benchmark 性能**：在针对各种 UI 任务的精选数据集上进行训练后，Ferret-UI 不仅优于大多数现有的开源 UI MLLM，还超越了 **GPT-4V**。
   - 全面的 Benchmark 包括图标识别和文本查找等任务，展示了该模型先进的推理能力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b">jadechoghari/Ferret-UI-Gemma2b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/jadechoghari/Ferret-UI-Llama8b">jadechoghari/Ferret-UI-Llama8b · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1304096835102838785)** (6 messages): 

> - `YouTube summarizer project`
> - `Interactive chat sessions`
> - `Model recommendations`
> - `Hugging Face resources` 


- **YouTube 总结器开发**：一位成员正在开发一个 **YouTube 总结器**，该工具可以根据视频内容启动**交互式聊天会话**，利用 [pytube](https://pytube.io/) 进行视频处理，并使用 **whisper** 进行转录。
   - 他们对 *bart-cnn* 模型的总结结果准确性表示担忧，并寻求关于如何进行聊天会话的指导。
- **切换到 LLaMA 模型以获得更好的效果**：另一位成员建议使用 Hugging Face 上的 **LLaMA 3** 作为解决总结器缺点的潜在方案，并建议使用规模为 **1B, 3B 或 8B** 的模型。
   - 该建议旨在增强总结过程的交互性和准确性。

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1303813131566452858)** (221 条消息🔥🔥): 

> - `Perplexity Pro subscription` (Perplexity Pro 订阅)
> - `Claude model capabilities` (Claude 模型能力)
> - `Image saving on Mac OS` (Mac OS 上的图片保存)
> - `Mobile device specifications` (移动设备规格)
> - `Mac OS login issues` (Mac OS 登录问题)


- **Perplexity Pro 提供教育折扣**：用户确认 Perplexity 服务的折扣仅限于美国大学，这引发了关于未来是否会在其他地区推出的疑问。
   - 一位用户表示失望，并询问教育折扣扩展到美国以外地区的时间表。
- **Claude 模型未按预期运行**：多位用户报告称，在选择 Claude 模型时，响应似乎是由 GPT-4o 生成的，这表明存在一个已知 Bug。
   - 社区提到开发者已获悉此问题，但修复进度被描述为缓慢。
- **在 Mac OS 上保存图片**：一位用户对无法在 Mac OS 上保存使用 Perplexity 创建的图片表示沮丧，并征求他人的建议。
   - 提出了不同的方法，包括在新标签页中打开图片，不过用户建议在 YouTube 等平台上查找详细说明。
- **新款移动设备的规格**：用户讨论了 Snapdragon 8 Gen 3 令人印象深刻的规格及其极具竞争力的价格，强调了其在游戏和 16K 视频播放方面的能力。
   - 对话转向与旧款移动设备的对比，强调了新机型的性能提升。
- **Mac OS 登录问题**：用户表示在 Mac OS 上登录 Perplexity 仍然面临挑战，并分享了一个据称运行效果更好的替代桌面应用程序的链接。
   - 尽管尝试了故障排除，许多用户仍对官方应用中持续存在的登录错误感到沮丧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/aravsrinivas/status/1854228102345597094?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：“Perplexity 的工具昨晚没有出现任何失误，提供了大部分准确的投票信息，并准确跟踪了随之而来的结果” - Wired</li><li><a href="https://www.bloomberg.com/news/newsletters/2024-08-14/what-i-learned-from-wearing-these-600-ai-glasses-all-week">Bloomberg - 你是机器人吗？</a>：未找到描述</li><li><a href="https://chromewebstore.google.com/detail/complexity-perplexity-ai/ffppmilmeaekegkpckebkeahjgmhggpj">Complexity - Perplexity AI 增强版 - Chrome 网上应用店</a>：⚡ 增强你的 Perplexity AI</li><li><a href="https://github.com/pnd280/complexity/releases/tag/v0.0.5.6-alpha">Release v0.0.5.6-alpha · pnd280/complexity</a>：更新日志：添加了 Claude 3.5 Haiku 模型。校验和：附件由 Chrome 网上应用店和 Mozilla Add-ons 进行数字签名。请勿从不受信任的来源安装！md5 hash CRX A...</li><li><a href="https://www.evenrealities.com/g1">Even Realities G1：带显示屏的下一代智能眼镜</a>：G1 通过 QuickNote、翻译、导航和 Even AI 等智能功能重新定义了处方眼镜。通过革命性的...增强你的日常生活。</li><li><a href="https://constructiondisputes.com/who-constructed-the-statue-of-liberty/)">谁建造了自由女神像？ | 建筑纠纷</a>：谁建造了自由女神像？自由女神像傲然屹立在纽约港的自由岛上。法国将自由女神像赠送给美国，并且...</li><li><a href="https://dozr.com/blog/building-the-statue-of-liberty)">建筑博客与见解 - 文章、播客及更多！ | DOZR</a>：DOZR 信息中心，提供建筑行业的行业见解和设备技巧。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1303826095035908128)** (13 条消息🔥): 

> - `Chernobyl's Radiation-Eating Fungi`（切尔诺贝利的食辐射真菌）
> - `Physical buttons return`（实体按键回归）
> - `Michigan as a Climate Sanctuary`（密歇根州作为气候避难所）
> - `Future of AI`（AI 的未来）
> - `Best value speakers and amp`（最具性价比的扬声器和功放）


- **切尔诺贝利真菌与实体按键复兴**：Perplexity AI 在其最新更新中强调了*实体按键的回归*以及**切尔诺贝利食辐射真菌**的迷人作用。
   - 技术与自然之间的联系展示了对挑战性环境的**创新适应**。
- **密歇根州是气候避难所吗？**：关于**密歇根州作为气候避难所的潜力地位**引发了讨论，相关链接提供了更多背景信息。
   - 这一探究引发了关于该州**环保政策**和未来可持续发展努力的疑问。
- **AI 未来讨论**：多条消息围绕 *AI 将如何继续演进* 展开，并附有关于该主题的各种讨论链接。
   - 分享了关于 **AI 进步**及其在不同领域潜在影响的见解。
- **音频设备性价比之选**：一位用户分享了一个专门研究和推荐黑胶唱机最具性价比**扬声器和功放**的**页面**。
   - 该资源旨在帮助他人进行音频升级，而无需透露之前的问答尴尬。
- **海苔与复合维生素的对比分析**：有人咨询了**海苔与复合维生素**之间的区别，并链接到了该主题的详细信息。
   - 讨论强调了了解**营养补充剂**及其益处的重要性。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1303848555789942805)** (2 条消息): 

> - `Completion API migration`（Completion API 迁移）
> - `Scheduled downtime for database upgrade`（数据库升级的计划停机）


- **Completion API 迁移提升速度**：所有 Completion API 请求已迁移到**重新编写的新 API**，这将增强性能，预计速度会**更快**。
   - 鼓励用户在指定的支持频道报告任何问题。
- **数据库升级计划停机**：已发布关于 **11 月 12 日星期二东部时间上午 9:30** 进行数据库升级的**计划停机**通知，预计持续 **5 分钟**。
   - 此次升级是提高系统可靠性和性能的持续努力的一部分。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1303817675113496689)** (224 messages🔥🔥): 

> - `Hermes Resurgence` (Hermes 复苏)
> - `Claude API Changes` (Claude API 变更)
> - `Mistral's New Features` (Mistral 新功能)
> - `OpenRouter API Issues` (OpenRouter API 问题)
> - `Chinese AI Models Pricing` (中国 AI 模型定价)


- **Hermes 展现出复苏迹象**：在经历了一段动荡时期后，**Hermes** 似乎恢复了工作，据报告响应时间在 **3 到 8 秒**之间。
   - 虽然部分用户仍遇到延迟，但许多人对其回归和持续改进表示乐观。
- **Claude 的 API 出现波动**：用户报告在通过 **OpenRouter** API 访问 **OpenAI** 模型时收到 `unsupported_country_region_territory` 错误。
   - 几位用户建议，该问题可能与迁移到 Cloudflare Workers 影响了端点（endpoint）响应有关。
- **Mistral 推出新功能**：**Mistral** 推出了两个新 API：一个 Moderation 工具和一个 Batch API，后者的处理成本比同步调用低 **50%**。
   - 这展示了在行业 API 价格上涨的背景下，Mistral 致力于提供负担得起的、可扩展的解决方案。
- **OpenRouter 面临 API 挑战**：多位用户报告在使用 **OpenRouter API** 时遇到 **404 错误**，特别指出 API URL 中多出的 '/' 是一个常见错误。
   - 讨论强调了近期 API 严格程度的变化导致了用户以前未曾遇到的问题。
- **中国 AI 模型的价格差异**：有关 **Qwen** 和 **DeepSeek** 等一些**中国 AI 模型**尽管受到国际限制但仍提供极具竞争力定价的讨论。
   - 然而，用户对这种低定价与 **OpenAI** 等成熟模型相比的可持续性表示怀疑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/docs/parameters">Parameters | OpenRouter</a>：配置请求参数</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>：暂无描述</li><li><a href="https://www.alibabacloud.com/help/en/model-studio/developer-reference/billing-for-tongyiqianwen">
 计算并查看 Qwen 的账单 - 阿里云 Model Studio - 阿里云文档中心

</a>：暂无描述</li><li><a href="https://docs.anthropic.com/en/docs/resources/model-deprecations">Model Deprecations - Anthropic</a>：暂无描述</li><li><a href="https://mistral.ai/news/mistral-moderation/">Mistral Moderation API</a>：我们推出了新的 Moderation 服务，使用户能够根据多个策略维度检测不良文本内容。</li><li><a href="https://mistral.ai/news/batch-api/">Mistral Batch API</a>：为 AI 构建者提供的更低成本 API。</li><li><a href="https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post">Mistral AI API | Mistral AI Large Language Models</a>：我们的 Chat Completion 和 Embeddings API 规范。在 [La Plateforme](https://console.mistral.ai) 创建账号以获取访问权限并阅读 [docs](https://docs.mistral.ai) 了解如何使用...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1303827883998838887)** (5 messages): 

> - `Customer Provider Keys`
> - `Integration Beta Features` 


- **申请 Customer Provider Keys 访问权限**：多位用户表示有兴趣测试 **customer provider keys**，并申请访问此 Beta 功能。
   - *steven1015* 表示：“申请访问 custom provider keys beta！”其他用户也表达了类似请求。
- **Integration Beta 功能访问**：一位用户询问如何获得 **integration beta feature** 的访问权限，表明需要更多人参与测试。
   - *mrhein* 简单地问道：“你好，能给我 integration beta 功能的访问权限吗？”这展示了对该功能日益增长的需求。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1303845592056991805)** (57 messages🔥🔥): 

> - `瑞士的研究机会`
> - `爵士钢琴生成模型`
> - `音乐评估方法`
> - `AI 与音乐上采样 (Musical Upscaling)`
> - `Magenta 的 Music Transformer` 


- **寻找瑞士的研究机会**：一位用户询问瑞士有哪些有趣的研究实验室，提到了 **ETH** 和 **EPFL**，但觉得范围太广。
   - 社区成员建议同时探索学术界和工业界实验室，并指出苏黎世以外的地区可能需要考虑语言因素。
- **爵士钢琴生成令人印象深刻**：最近一个基于 **120B tokens** 训练的爵士钢琴生成模型获得了好评，用户对其质量表示惊讶。
   - 虽然有人提到了对潜在 mode collapse 的担忧，但整体评价非常正面。
- **评估音乐生成质量**：关于评估音乐生成质量的讨论引出了进行听力测试以进行分析的建议。
   - 想法包括改变 prompt 长度，以测试对生成音乐中过渡部分的识别。
- **音乐上采样 (Musical Upscaling) 目标**：音乐上采样的概念被引入作为一个 finetuning 任务，即对演奏较差的曲目进行增强。
   - 观察到模型能够逐字记忆并背诵曲目，证明了该方法的有效性。
- **Magenta 的 Music Transformer 参考**：一位用户分享了 **Magenta's Music Transformer** 的参考资料，强调了其用于生成音乐表演的 open-source 模型。
   - 通过对比展示了自那时以来音乐生成模型的进步。



**Link mentioned**: <a href="https://magenta.github.io/listen-to-transformer/#a1_6828.mid">Listen to Transformer</a>：一个让探索和筛选 Music Transformer 输出变得更容易的应用程序。

  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1303811169852719165)** (116 条消息🔥🔥): 

> - `Flash Attention 技术`
> - `优化中的 Momentum Decay`
> - `NaNoGPT 更新`
> - `评估数据污染`
> - `高级 Attention 机制` 


- **Flash Attention 梯度探索**：一位用户询问了如何推导 Flash Attention 的前向梯度，并分享了普通 Attention 相对于 Q 的前向梯度基本公式。
   - 他们提供了一个以 `e^(q+ϵ)k/rowsum(e^(q+ϵ)k)` 为起点的思路，但对计算的后续步骤表示不确定。
- **Momentum Decay 对性能的影响**：讨论围绕在优化训练运行中嵌入 Momentum Decay 的效果展开，一位用户指出在各种配置下其影响微乎其微。
   - 有建议检查步时均值（step average）的增加是否由于热节流（thermal throttling）或 Kernel 选择引起的，因为一位成员报告了在特定步数时步时均值出现了异常跳变。
- **NaNoGPT 与训练改进**：用户分享了他们对 NaNoGPT 代码库的修改，详细介绍了最近关于嵌入 Momentum 和拆分 Lambda 的实验。
   - 他们得出结论认为样本量较小，并计划进行进一步测试以明确所取得的改进。
- **评估数据污染的衡量**：提到了理解 Benchmark 中评估数据污染的重要性，并介绍了 ConTAM 方法以更高效地评估此问题。
   - 这解决了确定哪些样本被污染及其对 Benchmark 分数后续影响的复杂性。
- **调整与分析代码性能**：一位成员提供了用于模型更新分析的代码，指出了在不同 Token 样本下性能的变化，特别是针对 300M 和 500M Token 距离的情况。
   - 他们被鼓励进行消融实验（ablations），以进一步明确不同特性对所取得的改进结果的贡献。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.03923">Evaluation data contamination in LLMs: how do we measure it and (when) does it matter?</a>: 评估数据污染已成为 LLM 评估中日益严重的问题，阻碍了对 Benchmark 分数的解释，目前已有研究在活跃探讨其影响。虽然评估...</li><li><a href="https://arxiv.org/abs/2406.17863">What type of inference is planning?</a>: 概率图模型有多种推理类型，例如边缘推理（marginal）、最大后验推理（maximum-a-posteriori），甚至边缘最大后验推理。当研究人员谈论...时，他们指的是哪一种？</li><li><a href="https://x.com/sirbayes/status/1854381167807443190">Kevin Patrick Murphy (@sirbayes) 的推文</a>: 查看我们关于将（离散）MDP 中的规划作为（一种新型）推理的 NeurIPS 论文。我们使用 loopy BP 扩展到因子模型。与 @lazarox8, @dileeplearning 和 Li Yu 合作。https://arxiv...</li><li><a href="https://github.com/leloykun/modded-nanogpt/tree/fc--momentum-cooldown/records/110724_EmbeddingBetasCooldown">modded-nanogpt/records/110724_EmbeddingBetasCooldown at fc--momentum-cooldown · leloykun/modded-nanogpt</a>: 在 2.67B Token 中达到 NanoGPT (124M) 的质量。通过在 GitHub 上创建账号为 leloykun/modded-nanogpt 的开发做出贡献。</li><li><a href="https://github.com/leloykun/modded-nanogpt/blob/fc--momentum-cooldown/records/110724_EmbeddingBetasCooldown/data_analysis.ipynb">modded-nanogpt/records/110724_EmbeddingBetasCooldown/data_analysis.ipynb at fc--momentum-cooldown · leloykun/modded-nanogpt</a>: 在 2.67B Token 中达到 NanoGPT (124M) 的质量。通过在 GitHub 上创建账号为 leloykun/modded-nanogpt 的开发做出贡献。</li><li><a href="https://github.com/EleutherAI/nanoGPT-mup">GitHub - EleutherAI/nanoGPT-mup: 用于训练/微调中型 GPT 的最简单、最快的仓库。</a>: 用于训练/微调中型 GPT 的最简单、最快的仓库。 - EleutherAI/nanoGPT-mup</li><li><a href="https://github.com/KellerJordan/modded-nanogpt/issues/10#issuecomment-2408933267">Attention scale · Issue #10 · KellerJordan/modded-nanogpt</a>: 你好，你使用了 (1 / (2 * config.n_layer)**0.5) 的 Attention 缩放来为 Block 层中的 SA 项加权。我尝试了使用和不使用它的情况，它似乎有一点帮助。（即使在保持 1... 的情况下）
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1304178846769221645)** (1 messages): 

> - `NeoX vs LitGPT 基准测试`
> - `预训练设置` 


- **对 NeoX 和 LitGPT 进行性能基准测试**：成员们正在询问比较 **NeoX** 和 **LitGPT** 框架性能差异的 **benchmarks**，特别是关于训练速度和稳定性。
   - 讨论强调了一种感知趋势，即许多用户在缺乏明确证据支持的对比情况下，更倾向于使用基于 **LitGPT** 的设置。
- **预训练设置的普及度**：观察到许多人似乎更倾向于基于 **LitGPT** 的预训练配置，这可能是由于熟悉度。
   - 然而，有人表示需要更多具体数据来支持这些关于框架有效性的选择。


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1303816935233949746)** (40 messages🔥): 

> - `微调 Smollm2`
> - `特征贡献分析`
> - `Gemma2 模型错误`
> - `8bit 和 4bit 支持`
> - `AI Unplugged 通讯` 


- **微调 Smollm2 的挑战**：用户报告了微调 **Smollm2** 时的问题，特别是注意到尽管数据集包含 **eos token**，但输出仍未结束。开发者已意识到模型错误，并正在与 **HF** 合作修复。
   - 建议升级到 **transformers 4.46** 并利用 **resume_from_checkpoint** 以获得更好的结果。
- **Llama3.2-1b 中出现占位符 Token**：用户在微调后的 **llama3.2-1b** 模型生成内容中观察到了奇怪的 token，特别是 **reserved_special_tokens**。这些被确定为占位符，引发了关于如何防止其出现的讨论。
   - 一位用户提供了一个相关 Pull Request 的 [GitHub link](https://github.com/unslothai/unsloth/pull/1235)，表明改进正在进行中。
- **8bit 支持即将到来**：即将推出的 **8bit** 和 **4bit** 支持令人兴奋，预计很快（可能在本月内）可用。用户询问该支持是针对 **fp8** 还是 **int8**。
   - 社区分享了一篇相关论文作为参考，以便更好地了解预期的增强功能。
- **寻求特征贡献分析工具**：一位用户请求推荐类似于 **SHAP**、**Lime** 或 **Captum** 的库或工具，用于在推理期间分析特征贡献。这突显了对模型性能可解释性的持续重视。
- **AI Unplugged 22：关于 RoPE 和 Mamba 的见解**：最新一期的 **AI Unplugged** 讨论了各种主题，包括 **RoPE**、**Mamba** 的改进以及下国际象棋的 **Transformer**。这一期被强调为最令人兴奋的版本之一，促使社区关注 **Substack** 内容。
   - 关键要点包括 **RoPE** 对模型适应性的重要性以及位置嵌入（position embeddings）的潜在增强。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Nryypksg64WnSSHErAitdyk8oruPgZeS?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://datta0.substack.com/p/ai-unplugged-22-rope-internals-explained">AI Unplugged 22: RoPE internals explained, Stuffed Mamba, Planning with Transfomer: Chess.</a>: 洞察胜过信息。</li><li><a href="https://github.com/unslothai/unsloth/pull/1235">CLI now handles user input strings for dtype correctly by Rabbidon · Pull Request #1235 · unslothai/unsloth</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1303812915760992358)** (51 条消息🔥): 

> - `NIM API Payment Options` (NIM API 支付选项)
> - `AI Model Usage Feedback` (AI 模型使用反馈)
> - `Discord Scam Discussions` (Discord 诈骗讨论)
> - `Exam Preparation` (考试准备)
> - `Mathematics Interview Questions` (数学面试题)


- **关于 NIM API 支付的反馈**：一位成员对目前使用 NIM API 的信用额度系统表示沮丧，建议应允许使用**信用卡即付即用 (pay-as-you-go)**。
   - 另一位成员解释说，其初衷是让用户在自己的 GPU 上运行可下载的容器，以实现模型体验。
- **AI 模型的持续使用**：一位用户分享了对 **Nemotron-340B-Reward 模型** 的正面评价，尽管对需要 GPU 访问才能持续使用感到困扰。
   - 该成员提到 **Llama-Nemotron-70B** 在生成合成数据方面也非常有效，展示了用户对这些模型的满意度。
- **关于 Discord 诈骗的讨论**：一位成员反思了持续不断的诈骗行为，提到观察到朋友受害，并强调了这些方案背后的自动化手段。
   - 他们指出 **QR code 诈骗** 特别巧妙，表明了社区对打击自动化垃圾信息的共同关注。
- **考试准备闲聊**：一位成员提到他们两小时后要参加 **NCEA Level 1 考试**，并幽默地评论说复习得非常疲惫。
   - 另一位参与者幽默地回应了考试题目，暗示如果题目与讨论的内容一致，应该是可以应对的。
- **面试准备中的数学趣题**：一位用户介绍了一个关于质数的经典面试趣题，强调了其在金融相关职位准备中的应用。
   - 这引发了一些轻松的回应，其中一位成员对问题的复杂性表示了抓狂。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1303812441897173084)** (53 条消息🔥): 

> - `Model Output Handling` (模型输出处理)
> - `VRAM Consumption Comparison` (VRAM 显存占用对比)
> - `Fine-Tuning Approach` (微调方法)
> - `Token Count Mismatch in Models` (模型 Token 数量不匹配)
> - `Concurrent Inference Setup` (并发推理设置)


- **输出处理问题**：用户讨论了修改模型输出的方法，特别是如何去掉像 `output[0]` 这样的索引，转而直接提取相关的响应内容。一个建议是使用 JSON schema 以简化数据解析。
   - 另一位用户确认通过修改一行用于剥离响应的代码成功解决了该问题。
- **不同模型间 VRAM 占用不一致**：有用户提出疑问，为什么像 **Aya** 这样的 8B 模型占用 22GB VRAM，而 3B 模型 **Llama3.2** 在不进行 Quantization 的情况下却使用了 43GB。一位参与讨论的用户认为这看起来不正确，因为模型大小不应暗示如此大的差异。
   - 另一位用户指出了基于 16-bit 精度标准的典型 VRAM 使用情况，并指出较大的模型在加载时会消耗更多 VRAM。
- **QA 任务的微调**：一位用户透露，在 2000 个 QA 对上对 3.2 3B instruct 模型进行微调（Loss 为 0.2）后，难以获得有意义的回答。建议包括确保问题包含在数据集内，并考虑 Epoch 数量以实现有效训练。
   - 另一位用户建议过多的 Epoch 可能会导致 Overfitting，而模型所有者解释说他们的高 Epoch 计数是为了确保工作相关任务的可靠性。
- **Token 数量不匹配错误**：一位用户在加载保存的模型时遇到了 `RuntimeError`，原因是 Token 大小不匹配——具体来说，模型显示为 **128264 tokens** 而非预期的 **128256**。观察表明，这可能是由于无意中添加了新 Token 导致的。
   - 建议的补救措施是在保存前利用 `save_pretrained_merged`，以避免加载过程中的这些差异。
- **探索并发推理**：一位用户询问如何使用 **Unsloth** 推理代码设置并发推理，而其他人则表示更倾向于使用原生处理并发的 **vLLM**。这引发了关于 **Hugging Face** 推理是否也实现了并发的兴趣。
   - 回复强调使用 **vLLM** 本身就支持并发进程，可能减少了额外设置的必要性。


  

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1304030190888030328)** (2 messages): 

> - `torch.compile and gradient checkpointing`
> - `product metrics tracking`
> - `software distribution in container ecosystem` 


- **关于增强 Torch Compile 以支持 Gradient Checkpointing 的讨论**：一位成员强调了 wiki 中关于使 **torch.compile** 与 **gradient checkpointing** 协同工作的待办事项，并提到了移除 **torch._dynamo.disable**。
   - 他们表达了贡献意愿，并表示自己在 **wrangling torch compile** 方面的经验可能会有所帮助。
- **需要更好地追踪产品指标 (Product Metrics)**：另一位成员建议创建一个 issue 以更好地追踪 **product metrics**，从而建立作为软件分发者的信任。
   - 这一建议暗示了在**容器生态系统 (container ecosystem)**中可靠性的重要性。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1304027087237288010)** (1 messages): 

> - `Training signal from LLM`
> - `Judge evaluation in training` 


- **探索在训练中引入 LLM 作为 Judge**：一位成员提出在传统的基于 token 的 loss 之外，加入来自 **LLM 的训练信号**作为 Judge 评估。
   - 他们承认虽然这可能会很慢，但观察这种方法的表现将会非常**有趣**。
- **需要创新的评估方法**：讨论强调了在 AI 训练中需要超越传统基于 token 的 loss 的**创新评估方法**。
   - 探索不同的途径可能会带来性能和评估指标的提升。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1303816113846751294)** (119 条消息🔥🔥): 

> - `Hermes3 的 Serverless Inference Endpoint`
> - `在阿拉伯语中使用 Hugging Face 模型`
> - `视觉模型的量化技术`
> - `Token 分类架构的建议`
> - `使用 Hugging Face Spaces 的 API` 


- **Serverless Inference Endpoint 设置困扰**：一位用户在为 Hermes3 创建 Serverless Inference Endpoint 时表达了挫败感，并询问部署是否需要输入信用卡。
   - 其他人建议 Serverless 选项是可用的，但不清楚如何链接模型或成功创建 API 所需的必要步骤。
- **阿拉伯语数据处理的最佳模型**：一位用户就适用于其应用程序中处理阿拉伯语文本数据的 LLM 模型寻求建议。
   - 社区建议探索各种阿拉伯语模型，但强调了选择与需求和数据集相匹配的模型的重要性。
- **视觉模型量化技术**：有人咨询了关于 Moondream2 模型的量化，以便在缺乏 GPU 的系统上使用，并强调了该模型对资源的较高要求。
   - 建议包括探索量化技术以在 CPU 上高效运行模型，同时保持性能标准。
- **蛋白质分类的神经网络架构**：一位编程新手寻求用于分类有限训练集下高维 Token 化蛋白质表示的神经网络架构建议。
   - 他们考虑从不带循环（recurrence）的多层感知器（MLP）开始，并欢迎社区提供进一步的建议。
- **连接到 Hugging Face Spaces API**：一位用户寻求关于使用 Python 的 requests 库连接到托管在 Hugging Face Spaces 上的 API 的指导，不确定正确的 Header 和身份验证方法。
   - 讨论涵盖了私有 Space 的潜在问题，以及使用 Hugging Face Token 进行 API 调用身份验证的成功连接方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/hub/spaces-config-reference">Spaces 配置参考</a>：未找到描述</li><li><a href="https://tenor.com/view/vergil-dmc5-gif-24729197">Vergil Dmc5 GIF - Vergil Dmc5 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://harmonydata.ac.uk/">Harmony | 全球上下文数据协调平台</a>：一个全球性的上下文数据协调平台</li><li><a href="https://harmonydata.ac.uk/doxa/">在 DOXA AI 上为 Harmony 训练大语言模型的竞赛 | Harmony</a>：一个全球性的上下文数据协调平台</li><li><a href="https://colab.research.google.com/drive/1E2bfoAXNGkEmyLffUR926bXjWZEq1lAU?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/QingruZhang/PASTA?tab=readme-ov-file">GitHub - QingruZhang/PASTA: PASTA: Post-hoc Attention Steering for LLMs</a>：PASTA：LLM 的事后注意力引导。通过在 GitHub 上创建账号来为 QingruZhang/PASTA 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1304082193253007370)** (1 条消息): 

> - `构建网络安全 AI 模型`
> - `网络安全 AI 面临的挑战` 


- **从零开始构建网络安全 AI**：为网络安全创建 **AI 模型** 具有挑战性，因为它需要 **AI 技术** 和 **网络安全原则** 两个领域的专业知识。
   - 获取高质量的训练数据以及理解 **网络安全威胁** 的复杂性对于有效的模型开发至关重要。
- **网络安全威胁的复杂性**：**网络安全威胁的复杂性** 以及对 **实时分析** 的需求显著增加了模型构建过程的难度。
   - 这些要求需要一种全面的方法，以确保模型能够有效应对不断演变的威胁。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1303966993552244737)** (2 条消息): 

> - `Hunyuan3D-1 Framework`
> - `Grape Leaf Disease Detection App` 


- **Hunyuan3D-1 发布统一框架**：腾讯 (Tencent) 推出了 [Hunyuan3D-1.0](https://huggingface.co/tencent/Hunyuan3D-1) 作为 **text-to-3D** 和 **image-to-3D** 生成的统一框架，目前两种格式的 demo 均已上线。
   - 在 **2024 年 11 月 5 日**，他们宣布支持通过 [脚本](#using-gradio) 运行 demo，并提供了 [代码](https://github.com/tencent/Hunyuan3D-1) 和 [报告](https://arxiv.org/pdf/2411.02293) 的访问权限。
- **葡萄叶片病害检测应用发布**：分享了一个新的 [Grape Leaf Disease Detection App](https://huggingface.co/spaces/thesab/Grape-Leaf-Disease-Detection-App)，展示了 AI 在农业领域的应用。
   - 这一令人耳目一新的 *创新工具* 旨在辅助植物病害的早期检测与管理。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/thesab/Grape-Leaf-Disease-Detection-App">Grape Leaf Disease Detection - thesab 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/tencent/Hunyuan3D-1">tencent/Hunyuan3D-1 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1303819547727495219)** (13 条消息🔥): 

> - `Formula 1 遥测聊天机器人`
> - `TinyLlama 模型转换`
> - `Harmony 问卷协调`
> - `USDA FoodData Central 数据集`
> - `文本生成 3D (Text-to-3D)` 


- **与 Formula 1 遥测数据对话**：推出了一种 AI 驱动的解决方案，通过使用自定义[界面](https://huggingface.co/spaces/Draichi/Formula1-race-debriefing)与真实比赛的**遥测数据 (telemetry data)** 进行对话，从而分析并生成 Formula 1 赛事的详细报告。
   - 该工具具备 Text-to-SQL 功能，允许用户查询比赛的各个方面，确保粉丝和车队都能轻松获取见解。
- **重大的 TinyLlama 模型架构转换**：完成了一次显著的 **TinyLlama 模型**架构转换，重点关注**差异化注意力 (differential attention)** 和 Token 混合，并提供了[转换脚本](https://huggingface.co/Josephgflowers/Differential-Attention-Liquid-Metal-Tinyllama)的访问权限。
   - 该项目包含有关在模型修改后的解码器层中集成各种模块的详细文档。
- **用于问卷协调的 Harmony 项目**：**Harmony** 项目通过一个针对研究人员的强大工具，促进了问卷项目的回顾性协调，更多关于兼容性的见解[可在此处获得](https://harmonydata.ac.uk/compare-harmonise-instruments/gad-7-vs-beck-anxiety-inventory/)。
   - 一项正在进行的竞赛正鼓励参与者改进 **LLM 匹配算法**，以提高理解文本相似句子的准确性。
- **交互式 USDA 食品助手上线**：发布了一个整合了 **USDA FoodData Central** 数据的新项目，其特色是一个交互式助手，通过简单的查询和见解提供对丰富食品数据的访问 ([GitHub 链接](https://github.com/jack-tol/usda-food-data-pipeline))。
   - 该数据集包含超过 **456,000** 种品牌食品，可在 HuggingFace 上获取，用于营养分析、消费者洞察和机器学习应用。
- **创新的文本生成 3D 框架**：**Hunyuan3D-1.0** 项目提出了一个统一的文本生成 3D (Text-to-3D) 和图像生成 3D (Image-to-3D) 框架，在该领域取得了重大进展 ([Hugging Face 链接](https://huggingface.co/spaces/gokaygokay/Hunyuan3D-1.0))。
   - 另一个项目 **FLUX.1-dev-LoRA-Outfit-Generator** 允许用户根据文本描述创建时尚装扮，展示了 AI 在时尚领域的通用性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/posts/Draichi/751144495449880">Hugging Face 上的 @Draichi："🏁 现在可以与真实的 Formula 1 比赛遥测数据进行对话了！……"</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/gokaygokay/Hunyuan3D-1.0">Hunyuan3D-1.0 - gokaygokay 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/Josephgflowers/Differential-Attention-Liquid-Metal-Tinyllama">Josephgflowers/Differential-Attention-Liquid-Metal-Tinyllama · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/thesab/outfit-generator">Outfit Generator - thesab 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://harmonydata.ac.uk/">Harmony | 一个用于上下文数据协调的全球平台</a>：一个用于上下文数据协调的全球平台</li><li><a href="https://harmonydata.ac.uk/doxa/">在 DOXA AI 上为 Harmony 训练大语言模型的竞赛 | Harmony</a>：一个用于上下文数据协调的全球平台</li><li><a href="https://github.com/openai/openai-realtime-console">GitHub - openai/openai-realtime-console: 用于检查、构建和调试 Realtime API 的 React 应用</a>：用于检查、构建和调试 Realtime API 的 React 应用 - openai/openai-realtime-console</li><li><a href="https://huggingface.co/datasets/jacktol/usda_branded_food_data">jacktol/usda_branded_food_data · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1303954548859342918)** (4 条消息): 

> - `OmniParser model`
> - `User engagement`
> - `Model performance`
> - `UI Interaction datasets` 


- **OmniParser 模型发布**：[OmniParser](https://huggingface.co/microsoft/OmniParser) 是一个通用的屏幕解析工具，可将 UI 截图转换为结构化格式，以增强基于 LLM 的 UI Agent。
   - 它利用了经过微调的 **YOLOv8** 和 **BLIP-2** 版本，并在可交互图标检测和 UI 元素描述的数据集上进行了训练。
- **用户回归项目**：一位成员表示在短暂缺席后回归，并提到需要完成一些工作。
   - 这突显了社区对频道中讨论工具的持续参与和依赖。
- **模型性能讨论**：有人提到 **Molmo** 表现相对较好，尽管未讨论具体指标。
   - 这一简短的认可暗示了社区对该模型有效性的积极态度。



**提到的链接**：<a href="https://huggingface.co/microsoft/OmniParser">microsoft/OmniParser · Hugging Face</a>：未找到描述内容

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1303845926137368588)** (1 条消息): 

> - `MaskGCT`
> - `F5-TTS`
> - `AI phone caller`
> - `audio chunk streaming` 


- **探索用于 AI 呼叫的 MaskGCT 和 F5-TTS**：一位成员对 **MaskGCT** 和 **F5-TTS** 表示赞赏，表示在开发过程中对这两个模型都有成功的体验。
   - 他们提出了关于这些模型是否能为其 **AI phone caller** 应用流式传输音频块的问题，考虑到由于其非自回归（non-autoregressive）特性，**流式传输可能会受到限制**。
- **对流式传输能力的担忧**：该成员询问 **MaskGCT** 或 **F5-TTS** 是否可以在保持流式传输功能的同时替换其现有的语音模型。
   - 他们指出，由于这两个模型都是非自回归的，因此对其**有效支持音频块流式传输的能力表示怀疑**。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1303879398377918526)** (2 条消息): 

> - `Integration Discussions`
> - `Automatic1111 SD3.5 Support` 


- **对集成工作的关注**：一位成员表达了热情，表示他们对正在进行的集成工作“非常好奇”，并鼓励与任何参与者合作。
   - 他们强调**参与的人越多越好**，暗示了社区参与在这一过程中的重要性。
- **关于 Automatic1111 SD3.5 支持时间表的查询**：另一位成员询问了关于 **Automatic1111** 何时提供对 **SD3.5** 支持的更新，显示了社区对该功能的兴趣。
   - 这反映了对增强用户体验的新功能的持续期待。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1303875034384826429)** (85 messages🔥🔥): 

> - `SearchGPT 性能`
> - `编程的最佳 OpenAI 模型`
> - `AI 自我意识讨论`
> - `AI 在工作中的角色`
> - `将 AI 作为学习工具` 


- **SearchGPT 需要改进**：有成员担心 **SearchGPT** 与默认模型相比不够智能且固执，在理解宽泛查询时表现吃力，且往往会产生幻觉（hallucinating）而不是承认找不到答案。
   - 一位社区成员强调，修正意见没有被妥善整合，并指出其倾向于重复回答。
- **为项目选择最佳 OpenAI 模型**：讨论集中在用于项目编程的 **OpenAI models**，建议使用 **o1-preview** 进行规划，使用 **o1-mini** 进行编码，因为后者拥有更充裕的配额和出色的 STEM 能力。
   - 也有成员担心 **o1-preview** 会将代码拆分成块，且无法充分处理多个请求。
- **AI 自我意识与意识**：围绕 **ChatGPT** 和 **Claude** 等 AI 是否能表现出自我意识展开了辩论，一些人认为 AI 可能会暗示其具有自我保护的长期计划。
   - 用户讨论了 AI 产生类似人类驱动力的风险，以及 LLM 的输出如何反映隐藏的推理能力。
- **AI 对工作和职级的影响**：对话涉及 AI 如何影响各个职级，一些成员质疑 AI 是真的取代了工作，还是仅仅通过不确定性迫使人们进行适应。
   - 对于是否应将工作划分为不同等级存在分歧，有人建议 AI 在特定角色中的表现可能会超越政治家。
- **将 AI 作为个人发展工具**：成员们讨论了将 AI 作为学习工具的使用情况，表示 AI 比传统教育更有效地促进了他们的自我提升。
   - 还有关于 AI 作为一种“信仰”潜力的轻松交流，并对这种观点提出了警告，同时承认了 AI 在个人成长中的益处。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1303850027520757842)** (4 messages): 

> - `Canvas 文档删除`
> - `已保存 GPTs 丢失`
> - `侧边栏固定问题`
> - `Custom GPT 功能改进` 


- **请求删除 Canvas 文档**：一位成员表示希望在 **CGPT4o + Canvas** 中启用文档删除功能，指出目前的设置存在一些摩擦。
   - 他们主张在平台中获得更多的文档管理控制权。
- **侧边栏丢失已保存的 GPTs**：一位用户报告称丢失了保存到左侧边栏的约 **20 个 GPTs**，并寻求对其潜在原因的见解。
   - 他们询问：*最近是否发生了什么可能导致这种情况的事情？*
- **侧边栏固定仅允许隐藏**：另一位用户遇到了一个问题，尝试固定到侧边栏时仅提供 **'Hide from sidebar'**（从侧边栏隐藏）选项。
   - 这种挫败感表明在有效管理侧边栏项目方面存在局限性。
- **期待 Custom GPT 功能扩展**：一位成员询问 OpenAI 是否有计划增强 **Custom GPT** 的功能，特别是关于增加文件大小限制和可上传文件数量方面。
   - 他们表示希望 OpenAI 正在为 **Custom GPT** 能力准备**改进**，并对该功能之外的其他积极进展表示认可。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1304068679331348611)** (3 messages): 

> - `文件处理问题`
> - `私信寻求支持` 


- **Leo 寻求多格式文件解决方案**：成员 Leo 表示需要解决涉及多种文件类型的问题：**JSON**、**HTML**、**JS** 和 **CSS**。
   - 另一位成员询问：*你尝试了什么？*，表示希望了解更多故障排除步骤的信息。
- **建议进行私下讨论以方便交流**：Gassin 建议将对话转移到私信（direct messages），以便更轻松地交换意见。
   - 这表明讨论从公开的故障排除转向了更个性化的支持方式。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1304068679331348611)** (3 messages): 

> - `File format issues`（文件格式问题）
> - `Direct messaging for troubleshooting`（通过私信进行故障排除）


- **寻求多种文件格式问题的解决方案**：一名成员正在寻求关于 **json**、**html**、**js** 和 **css** 文件出现问题的解决方案。
   - 这个问题似乎影响了多种格式，具体问题的细节仍需进一步明确。
- **转向私信以获得更便捷的支持**：有人建议将对话转移到 **direct messages**（私信），以便更轻松地进行故障排除。
   - 这一举动暗示在私人环境中讨论技术问题可能会产生更高效的解决方案。


  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1303937039363997757)** (32 messages🔥): 

> - `Installing PyTorch`（安装 PyTorch）
> - `Kernel Development on Windows`（Windows 上的 Kernel 开发）
> - `CUDA Compiling Issues`（CUDA 编译问题）
> - `Using Visual Studio for CUDA`（使用 Visual Studio 进行 CUDA 开发）
> - `Runtime Errors in CUDA`（CUDA 中的运行时错误）


- **安装变声器需要帮助**：一名用户在安装变声器时寻求帮助，特别是需要通过 Python 或命令行安装 **PyTorch**。
   - 建议通过 *私信或语音通话* 以获得更个性化的帮助。
- **关于 Windows 上 Kernel 开发的讨论**：成员们讨论了使用 **Windows** 进行 Kernel 开发的可行性，并强调编译 CUDA 与编译其他 C++ 程序类似。
   - 讨论中提到了对某些项目兼容性的担忧，以及是否有必要使用 **Docker** 或 **WSL**。
- **CUDA 编译的挑战**：一名用户遇到了 **CUDA error** 消息，提示没有可用于执行的 Kernel 镜像，因此请求帮助。
   - 其他人指出，诊断该问题需要代码和构建过程的上下文，并强调需要针对正确的 GPU 架构进行编译。
- **Visual Studio 作为 CUDA 开发的 IDE**：讨论强调了 **Visual Studio** 作为 IDE 的稳健性，尽管关于其性能和在 **ML community** 中的普及程度意见不一。
   - 一些人更倾向于在小型项目中使用 **VS Code** 配合 **nvcc**。
- **解决 CUDA 中的运行时错误**：另一名用户报告了一个与 CUDA 相关的 **RuntimeError**，经诊断可能与缺少针对其 GPU 架构的编译目标有关。
   - 建议包括确保 GPU 受支持，以及如果问题持续存在，考虑更换硬件。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1304010672455286825)** (4 messages): 

> - `FP16 x FP16 Performance`（FP16 x FP16 性能）
> - `A100 GPU Efficiency`（A100 GPU 效率）
> - `CUDA Cores and GEMVs`（CUDA Cores 与 GEMVs）


- **FP16 x FP16 在 GPU 上表现不一**：讨论指出，**FP16 x FP16 配合 FP16 累加** 在 A100 等 **data-center GPUs** 上没有加速效果，因为它们共享相同的 flops。
   - 相反，有人指出这种组合仅在 **consumer cards**（消费级显卡）上更快，而企业级显卡在使用 **FP32 accumulation**（FP32 累加）时可以保持性能而不降速。
- **A100 的 CUDA Cores 表现优于 FP16**：提到 A100 上的 **CUDA cores** 在处理 **FP16** 时仍然略快，这对于 **GEMVs** 特别有用，尽管速度提升很小。
   - 这一观察结果来自于在 **A100 SXM4** 上使用 **Triton** 进行的测试，指出了在特定场景下的实际益处。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1303834119930908753)** (6 messages): 

> - `Torch Script Debugging`（Torch Script 调试）
> - `Performance Overhead of .tolist() Call`（.tolist() 调用的性能开销）
> - `C++ API Limitations`（C++ API 的局限性）


- **Torch Script 调试方面的疑虑**：一名用户询问了调试 **Torch Script** 并打印脚本中每个节点输出的方法。
   - 一名成员指出 *“就目前而言，它有点被废弃了”*，暗示该工具的支持力度有所下降。
- **tolist() 带来的性能开销**：另一名用户指出，对于 **2^10 大小的数组**，大约 **30%** 的开销源自 `.tolist()` 调用本身。
   - 他们强调，如果性能至关重要且需要多次迭代，这种开销可能会非常显著。
- **关于 C++ API 的见解**：一名参与者分享了他们使用 **Torch Script** 的 **C++ API** 的经验，提到它一直很有用。
   - 然而，他们警告说，许多功能 *已不再受支持且文档不全*。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

marksaroufim: https://x.com/_seemethere/status/1838319643347554756
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1304090173738451067)** (8 messages🔥): 

> - `LLM 中的计算密集型操作`
> - `GEMM 优化资源`
> - `CUDA 和 GPU 学习资料` 


- **举足轻重：LLM 中的线性层占据主导**：一位用户指出，在像 **LLaMa3.1** 这样的 LLM Transformer 中，MLP 和 Attention 块中的**线性层（linear layers）**通常是最耗费计算的操作。
   - 对于**长上下文推理（long context inference）**，Attention 机制往往会主导计算负载。
- **应届生寻求 GEMM 指导**：一位拥有深厚 **C++** 背景的计算机科学应届毕业生正在寻找资源，为专注于 **GEMM 优化**和 Kernel 优化的新工作做准备。
   - 另一位用户提供了各种资源，包括关于 **CUDA** 和优化技术的文章及 GitHub 仓库。
- **庆祝社区成就**：成员们向这位应届生表示祝贺，祝贺其顺利毕业并即将入职从事优化工作。
   - 强调了小组支持性的氛围，为新人培养了社区归属感。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://siboehm.com/articles/22/CUDA-MMM">How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</a>：在这篇文章中，我将迭代优化一个用 CUDA 编写的矩阵乘法实现。我的目标不是构建一个 cuBLAS 的替代品，而是深入...</li><li><a href="https://www.youtube.com/watch?v=kwtn2NtFIuA&list=PLRRuQYjFhpmvu5ODQoY2l7D0ADgWEcYAX">Lecture #1 - Introduction</a>：UIUC ECE408 2018 春季 Hwu</li><li><a href="https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/">CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining</a>：欢迎来到我们关于 GEMM（通用矩阵乘法）教程系列的第 2 部分。在第 1 部分中，我们通过回顾 WGMMA 讨论了 GEMM 的计算方面，WGMMA 是用于矩阵乘法的原始指令...</li><li><a href="https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/">CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA® Hopper&#x2122; GPUs</a>：如果没有关于 GEMM（通用矩阵乘法）的部分，CUDA® 教程系列就不完整。作为现代 GPU 上最重要的例程，GEMM 构成了大部分计算量...</li><li><a href="https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/">Deep Dive on Cutlass Ping-Pong GEMM Kernel</a>：   
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1304174624631689279)** (3 messages): 

> - `数字音频入门`
> - `Discord AV1 嵌入工具`
> - `视频嵌入限制` 


- **Monty 的数字音频入门详解**：Xiph 的 Monty 全面概述了 **D/A 和 A/D** 过程，并使用现代和复古设备对[采样、量化和位深](https://autocompressor.net/av1?s=JUrvYDLZ)进行了实时演示。
   - *听起来很酷，但只有我觉得这是一个 **0 长度的视频**吗？* 一位成员问道，对视频内容提出了疑问。
- **关于 D/A 和 A/D 的 YouTube 视频**：一位用户分享了一个名为 [‘D/A and A/D | Digital Show and Tell’](https://youtu.be/cIQ9IXSUzuM?si=2ut1W6By-9j7-UTh) 的 YouTube 视频，由 xiph.org 的 Monty Montgomery 讨论音频格式的重要性。
   - 该视频解释了为什么你不需要 **24 Bit 192 kHz** 的听音格式，并链接了关于数字音频的其他资源。
- **Discord AV1 嵌入工具功能**：Discord AV1 嵌入工具因允许用户利用 Discord 漏洞嵌入大于 500MB 的 AV1 视频而受到关注。
   - 它还支持自定义缩略图，并提供了在多个平台上嵌入视频的文档。
- **Discord 的嵌入限制**：为了在 Discord 上成功嵌入视频，需要满足某些限制，例如使用 **MP4** 或 **WebM** 等格式以及兼容的视频编解码器。
   - 这些编解码器包括 **AV1**、**HEVC** 和 **VP9**，并指出某些编解码器可能无法在所有客户端或浏览器中通用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://autocompressor.net/av1?s=JUrvYDLZ">Autocompressor Video Embed Tool</a>：未找到描述</li><li><a href="https://youtu.be/cIQ9IXSUzuM?si=2ut1W6By-9j7-UTh)">D/A and A/D | Digital Show and Tell (Monty Montgomery @ xiph.org)</a>：原始视频：http://xiph.org/video/vid2.shtml 为什么你不需要 24 Bit 192 kHz 听音格式 - https://people.xiph.org/~xiphmont/demo/neil-young.html...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/)** (1 messages): 

marksaroufim: https://discord.gg/zCkRcp6e?event=1304137506014629969
  

---

### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1304054553745952770)** (19 messages🔥): 

> - `通用 JSD kernel 改进`
> - `工程博客致谢`
> - `S1ro 昵称渊源`
> - `LinkedIn 个人资料` 


- **通用 JSD Kernel 取得显著改进**：Chun-Chih Tseng 开发了一个**通用 JSD kernel**，在 128k 词表大小下实现了 **1.5 倍速度提升**和 **50% 的峰值显存降低**，同时为 LigerCrossEntropy 实现了相关功能。
   - 更新还包括 Tyler Romero 对 **phi**、**qwen** 和 **llama-vision** 的支持，以及其他成员提供的额外 kernel 增强。
- **对工程贡献的致谢**：Byron 宣布计划在即将发布的 [LinkedIn 工程博客](https://www.linkedin.com)中包含成员的贡献。
   - 几位成员确认了他们的 LinkedIn 个人资料，并协助调整内容以确保准确的归属。
- **S1ro 昵称之谜**：S1ro 幽默地分享说，他的 Discord 昵称非常普遍，甚至他的父母偶尔也会脱口而出，因为**所有人**都这么叫他。
   - 这个昵称已经伴随他很久了，说明在他的圈子里被广泛使用。
- **关于 Discord 名称和家庭的轻松聊天**：围绕 S1ro 的昵称进行了一番愉快的交流，其他人对他父母有时也用这个昵称感到很有趣，引发了一些笑声。
   - 这个有趣的细节凸显了他们对话的趣味性以及团队内的情谊。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1303815748577263677)** (6 messages): 

> - `通过 Heroku 实现自动化部署`
> - `使用 Raspberry Pi 运行 Discord Bot`
> - `Project Popcorn 的目标`
> - `服务器连接`
> - `GPU 实现计划` 


- **自动化部署现已在 Heroku 上线**：一位成员成功通过 **Heroku** 设置了**自动化部署**，只需将更改推送到 main 分支即可更新 bot。
   - 与其他托管方案相比，他们对这种 Web 规模的方法感到兴奋。
- **考虑过 Raspberry Pi 但存在陷阱**：有人对使用 **Raspberry Pi** 托管 Discord bot 表示担忧，理由是之前的经验表明这种方式**容易出错**。
   - 这使得成员们因可靠性而更倾向于 **Heroku** 方案。
- **Project Popcorn 目标已记录**：一位成员分享了一个关于 [Project Popcorn 的 Gist](https://gist.github.com/msaroufim/087c2a358c505e287a926e6a27b3e3b0)，概述了其目标和简介。
   - 该项目旨在公共空间利用 **LLM** 生成 **SOTA kernel**，促进社区参与和透明度。
- **准备实际服务器连接**：计划在获得 **GPU** 后开始将 bot 连接到服务器。
   - 这一步被视为充分发挥 bot 运行潜力的关键。



**提到的链接**：<a href="https://gist.github.com/msaroufim/087c2a358c505e287a926e6a27b3e3b0">Project Popcorn: Generate SOTA kernels with LLMs in public</a>: Project Popcorn: Generate SOTA kernels with LLMs in public - 🍿.md

  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1303842954086977550)** (6 messages): 

> - `ThunderKittens Contribution List`
> - `Beginner Contributions to Kernels` 


- **ThunderKittens 贡献列表缺失**：成员们表示很难找到针对 **ThunderKittens** 项目的初学者贡献 Kernel 列表，正如另一位成员在初始帖子中所讨论的那样。
   - *一位成员提到他们花时间寻找该列表但无处可寻*，这凸显了对更清晰文档的需求。
- **分享初步贡献列表**：一位成员在 ThunderKittens GitHub 仓库上提供了一个[初步贡献列表](https://github.com/HazyResearch/ThunderKittens?tab=readme-ov-file#demos)。
   - 他们鼓励其他人探索这些贡献，并就入门示例寻求指导，例如为非平方序列长度添加长卷积（long convolution）。
- **提议协助新贡献**：一位成员提议提供 PyTorch 参考资料，以协助任何有兴趣添加长卷积 Kernel 作为入门示例的人。
   - *他们表示愿意随着社区中出现更多想法而不断更新贡献列表*。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/HazyResearch/ThunderKittens">GitHub - HazyResearch/ThunderKittens: Tile primitives for speedy kernels</a>：用于快速 Kernel 的 Tile primitives。通过在 GitHub 上创建账户来为 HazyResearch/ThunderKittens 的开发做出贡献。</li><li><a href="https://github.com/HazyResearch/ThunderKittens?tab=readme-ov-file#demos">GitHub - HazyResearch/ThunderKittens: Tile primitives for speedy kernels</a>：用于快速 Kernel 的 Tile primitives。通过在 GitHub 上创建账户来为 HazyResearch/ThunderKittens 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1303814357783810110)** (24 messages🔥): 

> - `Using Google Docs for Research`
> - `Podcast Reuse Policy`
> - `Fun Prompts for Podcast Hosts`
> - `Dyslexia and Day Trading`
> - `Understanding Terms of Service` 


- **合并文章以简化来源**：一位成员建议将相关文章合并到一个或多个 Google Docs 中，以使总来源保持在 **50** 个以下。
   - 这可能使编辑和重新同步更加容易，为内容管理提供了一种潜在的变通方案。
- **澄清播客重用政策**：有人提出了关于播客重用政策的咨询，特别是关于在包含[相关链接](https://github.com/robbiemu/llama-gguf-optimize)的 GitHub 仓库中分享的内容。
   - 成员们希望在利用播客材料之前确保遵循指南。
- **寻求让播客主持人更有趣的提示词**：对能让播客主持人更有趣的优质 Prompt 的需求，引发了对更多创意和引人入胜内容的需求。
   - 这引发了关于通过幽默感提升播客收听体验的讨论。
- **阅读障碍激发使用 NotebookLM 学习**：一位成员分享了他们使用 NotebookLM 在学习日内交易（day trading）时简化笔记的热情，并对该工具的影响表示感谢。
   - 他们强调了尽管患有阅读障碍（Dyslexia），但发现了一种参与教育内容的新方式的喜悦。
- **关于服务条款教育的 YouTube 频道构想**：有人建议创建一个 YouTube 频道，专门解析各大公司的服务条款（TOS）和隐私政策。
   - 成员们发现这个想法很有吸引力，并指出很少有人阅读这些文档，而引人入胜的内容可以帮助更多人理解它们。



**提及的链接**：<a href="https://github.com/robbiemu/llama-gguf-optimize">GitHub - robbiemu/llama-gguf-optimize: Scripts and tools for optimizing quantizations in llama.cpp with GGUF imatrices.</a>：用于在 llama.cpp 中使用 GGUF imatrices 优化量化的脚本和工具。- robbiemu/llama-gguf-optimize

  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1303811522513993809)** (59 条消息🔥🔥): 

> - `NotebookLM 性能问题`
> - `PDF 集成挑战`
> - `共享选项与 Bug`
> - `已保存笔记的可访问性`
> - `NotebookLM 的项目创新` 


- **NotebookLM 存在令人沮丧的性能问题**：用户报告称 NotebookLM 中的 Bot 会互相抢话（接对方的句子），导致部分用户体验极差。一位用户指出，感觉就像 Bot 们在阅读同一份源文档，导致对话内容不断重复。
   - 另一位用户在各种移动浏览器上打开已保存的笔记时遇到了滚动问题，迫使他们寻找替代方案。
- **缺少从 Google Drive 加载 PDF 的功能**：用户对无法直接将 PDF 从 Google Drive 加载到 NotebookLM 表示失望。许多人认为，尤其是在购买了存储空间后，这种功能的缺失限制了集成的实用性。
   - 用户期待该功能能很快上线，因为这看起来是提升用户体验的合乎逻辑的增强。
- **共享选项混乱且有 Bug 报告**：一名成员指出共享 Notebook 时存在持续性问题，例如链接无法加载或用户无法保持添加状态。这引发了关于共享功能是否存在系统性问题的讨论。
   - 另一位用户指出他们再也找不到共享按钮了，这引发了对最近更新中可能存在未解决 Bug 的担忧。
- **已保存笔记中的引用访问不明确**：用户抱怨在保存笔记后无法查看笔记引用的来源，导致困惑。一位用户建议，缺少脚注引用是该工具的一个重大疏忽。
   - 许多人产生共鸣，认为在已保存的笔记中能够参考引用应该是基础功能，这能增强工具的实用性和透明度。
- **关于创新项目集成的建议**：一位成员寻求关于如何增强 NotebookLM 与 Google 产品及外部工具集成的想法，旨在改善用户体验。反馈表明，用户渴望具有实际应用价值的有效创新解决方案。
   - 在讨论可能的功能时，负责人强调了在完善现有功能的同时，探索能更有效地吸引用户的新模型的重要性。



**提及的链接**：<a href="https://youtu.be/CfQiVMV9NJQ">Mastering the Digital SAT: Expert Tips for Reading &amp; Writing with Special Guest Riley | Episode 9</a>：访问我们的网站获取免费的 SAT 和 GRE 备考资料：https://campusgoals.com/ 欢迎来到 &quot;Mastering the SAT with Alex and Taylor&quot; 第 9 集，这是您的首选...

  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1304092036831645716)** (15 messages🔥): 

> - `Anthropic and Palantir partnership`（Anthropic 与 Palantir 的合作伙伴关系）
> - `U.S. government classification levels`（美国政府机密等级）
> - `Concerns about defense collaboration`（对国防合作的担忧）
> - `Reddit community for security clearances`（关于安全审查的 Reddit 社区）


- **Anthropic 与 Palantir 及 AWS 合作开发国防 AI**：Anthropic 宣布与 [Palantir](https://www.businesswire.com/news/home/20241107699415/en/Anthropic-and-Palantir-Partner-to-Bring-Claude-AI-Models-to-AWS-for-U.S.-Government-Intelligence-and-Defense-Operations) 和 Amazon Web Services 达成合作，为美国情报和国防机构提供其 **Claude** 系列 AI 模型的访问权限。
   - 此举紧随其他科技公司的类似行动，旨在满足国家安全领域对 AI 解决方案日益增长的需求，从而争取 **defense contracts**（国防合同）。
- **了解美国机密等级**：一位成员分享了对美国 **government classification system**（政府机密系统）的见解，重点介绍了“秘密”（Secret）级别，该级别仅次于“最高机密”（Top Secret）。
   - 该级别的信息如果泄露，被认为会对国家安全造成“严重损害”。
- **对国防 AI 合作的担忧**：成员们对 Anthropic 的国防合作表达了保留意见，在讨论此类伙伴关系的潜在影响时，对与 **Palantir** 的合作感到不安。
   - 一些人指出，尽管存在担忧，但考虑到 Anthropic 的文化，他们感到相对放心。
- **安全审查 Subreddit 的有趣发现**：一位成员指出了 [r/SecurityClearance](https://www.reddit.com/r/SecurityClearance/top/?sort=top&t=all) subreddit 的价值，强调了其中关于审查不合格原因以及审查过程侵入性的讨论。
   - 该 subreddit 拥有超过 **53,000 名成员**，为应对安全审查环境提供了深刻见解和轶事经历。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://techcrunch.com/2024/11/07/anthropic-teams-up-with-palantir-and-aws-to-sell-its-ai-to-defense-customers/">Anthropic teams up with Palantir and AWS to sell AI to defense customers | TechCrunch</a>：Anthropic 已与 Palantir 和 AWS 联手，向国防客户销售其名为 Claude 的 AI 系列模型。</li><li><a href="https://en.m.wikipedia.org/wiki/Classified_information_in_the_United_States">Classified information in the United States - Wikipedia</a>：未找到描述</li><li><a href="https://www.reddit.com/r/SecurityClearance/top/?sort=top&t=all">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.404media.co/r-securityclearance-is-the-best-subreddit/">r/SecurityClearance Is the Best Subreddit</a>：一个疯狂的 subreddit，人们在向美国政府坦白秘密之前，会先在这里分享。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1304138455525232782)** (5 messages): 

> - `8-bits pretraining`（8-bits 预训练）
> - `Tim Dettmers' paper`（Tim Dettmers 的论文）
> - `YouTube talks on pretraining`（YouTube 上关于预训练的演讲）


- **寻求关于 8-bits 预训练的清晰解释**：一位用户表示有兴趣寻找关于 **8-bits pretraining** 的资源，并分享了一篇他们认为很有帮助的 [Tim Dettmers 等人的论文](https://arxiv.org/pdf/2110.02861)。
   - 他们向社区征求关于该主题的更多建议。
- **YouTube 提供预训练相关演讲**：另一位成员建议，[YouTube](https://www.youtube.com/) 上可能有一些高质量的演讲可以帮助理解该主题。
   - 他们暗示这些资源可能会提供关于 **8-bits pretraining** 的进一步见解。
- **Tim Dettmers 的热门论文**：一位成员对该工作的受欢迎程度发表了评论，提到他们在进行背景研究时看到了 **Tim Dettmers** 的演讲。
   - 这表明了该论文的重要性及其在社区讨论中的相关性。


  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1304172045176930304)** (1 messages): 

> - `Synthetic Data Generation` (合成数据生成)
> - `SFT Data Scaling` (SFT 数据缩放)
> - `Instruction Data Usage` (指令数据使用)
> - `T0 Comparisons` (T0 对比)


- **新论文中关于合成数据的主张**：一篇论文声称使用了 **1.5T tokens** 的合成数据，同时指出其 **SFT 数据规模** 仅为 **100 万个样本**。
   - *这是否意味着他们在预训练期间使用了指令数据？* 这种情况引发了与 **T0 模型** 及其训练策略相关的讨论。
- **质疑合成数据的应用**：讨论强调了合成数据生成量与报告的 SFT 数据量之间潜在的不一致性。
   - 这提出了一个关于 **LLMs** 在整合指令数据方面的通用做法的更广泛问题。



**提到的链接**：<a href="https://arxiv.org/abs/2411.02265">Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent</a>：在本文中，我们介绍了 Hunyuan-Large，这是目前最大的开源基于 Transformer 的混合专家（Mixture of Experts）模型，总参数量为 3890 亿，激活参数量为 520 亿...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1303969976578871337)** (3 messages): 

> - `MLST podcast episode with Chollet` (与 Chollet 合作的 MLST 播客集)
> - `Chollet's views on memorization` (Chollet 对记忆化的看法)
> - `Frustrations with Tim's style` (对 Tim 风格的挫败感)


- **Chollet 在 MLST 节目中澄清误解**：最新的 [与 Chollet 合作的 MLST 播客集](https://link.to.mlst) 因澄清了围绕他关于 **记忆化（memorization）** 观点的误解而受到好评。
   - *总觉得人们从两方面都对他有所误解*，这一集旨在解决这些问题。
- **听众对播客见解的怀疑**：一位成员表达了对收听 MLST 节目的犹豫，觉得从中获益不多。
   - *我觉得我从来没有从中得到很多东西*，这突显了一些听众对该播客内容的共同担忧。
- **对 Tim 沟通风格的挫败感**：另一位成员评论了在听 MLST 节目时因 Tim 使用复杂的词汇和抽象概念而感到的 **挫败感**。
   - 他们指出，*Tim 只是在抛出大词和抽象概念*，这表明在理解材料方面存在挑战。
- **Chollet 的非争议性观点**：一条支持性的评论强调，尽管 Chollet 的方法带有批判性，但他的观点通常是 **没有争议的**。
   - 该成员指出，Chollet *喜欢贬低* 某些话题，将他们的讨论风格看作更多是视角问题，而非挑战。


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1304054657353650236)** (8 messages🔥): 

> - `Election polling as prompt engineering` (选举民调作为提示工程)
> - `Bias in polling methodologies` (民调方法论中的偏见)
> - `Monetary gains from polling insights` (从民调洞察中获得的金钱收益)


- **选举民调获利 4900 万美元**：一位成员讨论了一名法国人如何通过注意到偏向特朗普表现的偏见民调问题，在 Polymarket 上 **豪掷 3000 万美元** 押注特朗普，从而在 **一周内获利 4900 万美元**。
   - *你无法反驳一个在如此疯狂的赌注上赚了 5000 万的人*，这突显了非常规民调方法带来的意想不到的结果。
- **民调偏见需要更好的工具**：成员们强调，民调中的偏见不易控制，并建议利用 **更好的工具**（如隐蔽测量）来改进方法论。
   - 有人指出，民调缺乏有效的统计控制，并建议聘请心理学本科生来协助改进方法论。
- **Andrew Gelman 对民调的见解**：一位成员推荐了 Andrew Gelman 关于民调的博客，作为理解民调方法论中复杂性和挑战的资源。
   - 有人指出，**个位数的响应率** 严重阻碍了民调数据的可靠性。



**提到的链接**：<a href="https://x.com/blader/status/1854366739511030065?s=48">Siqi Chen (@blader) 的推文</a>：那个在 Polymarket 上豪掷 3000 万美元押注特朗普的法国人注意到，那些问“你的邻居会投给谁？”而不是“你会投给谁？”的民调显示特朗普表现超预期。于是他委托进行了...

  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1304106462393667584)** (32 messages🔥): 

> - `Tim 移至 CMU`
> - `Quantization 技术`
> - `Character.AI 的 Inference 优化`
> - `GPU 效率`
> - `关于 Quantization 的社区讨论` 


- **Tim 移至 CMU**: Tim 最近搬到了 **Carnegie Mellon University (CMU)**，目前正在远程工作，许多成员对他所做的贡献表示赞赏。
   - 大家希望 Tim 在 **2025** 年能有更活跃的合作。
- **Quantization 是游戏规则改变者**: Tim 分享了见解，指出对于模型使用，**8-bit Quantization** 是存储信息的最佳选择，并警告说进一步降低位数会损害模型性能。
   - 人们现在开始采用 8-bit Pretraining 而不是传统的 32-bit，这显著提高了训练效率。
- **Character.AI 的高效 Inference**: [Character.AI](https://research.character.ai/optimizing-inference/) 正在向 AGI 迈进，专注于高效的 Inference，以支持每秒超过 **20,000 次查询**。
   - 他们的方法使用 **int8 Quantization** 来提高训练效率，这标志着与传统的 Post-training Quantization 方法有所不同。
- **通过 Quantization 使 GPU 算力翻倍**: Tim 强调 Quantization 使其能够有效地利用 **2x 更多的 GPU**，增强了计算能力。
   - 当被问及 Quantization 所需的努力是否值得时，他肯定了其带来的显著收益。
- **社区对 Quantization 的参与**: 社区对 Quantization 技术展开了热烈讨论，成员们表示有兴趣深入了解其中的复杂性。
   - 几位成员寻求关于如何实现有效的 Quantization 及其在训练期间的实施方案的澄清。



**Link mentioned**: <a href="https://research.character.ai/optimizing-inference/">Optimizing AI Inference at Character.AI</a>: 在 Character.AI，我们正致力于构建 AGI。在未来的愿景中，Large Language Models (LLMs) 将提升日常生活，提供商业生产力和娱乐，并帮助人们处理...

  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1303836898317303850)** (41 条消息🔥): 

> - `Ollama 的 llama 3.2 Vision`
> - `MLX Engine 更新`
> - `NVIDIA 征求反馈`
> - `LM Studio CPU 模式`
> - `LM Studio 中不支持的模型` 


- **Ollama 推出 llama 3.2 Vision**：一位成员强调 Ollama 现在支持 **llama 3.2 Vision**，而其他人指出 **MLX** 具有类似功能，但在 **llama.cpp** 中仍缺乏支持。
   - 用户对其在 **LM Studio** 中的功能表示担忧，一名用户在加载模型时遇到了错误。
- **MLX Engine 支持的 Pull Request**：一个相关的 [GitHub pull request](https://github.com/lmstudio-ai/mlx-engine/pull/22) 讨论了对 **MLX engine** 的升级，以支持 **llama 3.2 Vision**。
   - 该更新预计很快发布，引发了期待功能改进的用户的兴奋。
- **NVIDIA 寻求 AI 爱好者的建议**：来自 NVIDIA 的一名成员邀请非开发者的 AI 爱好者在简短的交谈中分享他们的经验和挑战，并提供了一个 [Calendly 链接](https://calendly.com/aslisabanci-01-nvidia/10min) 用于预约。
   - 其他成员确认了此请求的真实性，鼓励大家参与以影响未来的产品。
- **在 CPU 模式下运行 LM Studio**：用户讨论了在旧款 GPU 上运行 **LM Studio 0.3.5** 的挑战，其他人确认选择 CPU 模式可以解决这些问题。
   - 提供了进入 CPU 运行时（runtime）的说明，并指出该设置在最新版本中隐藏得较深。
- **在 Just-In-Time 模式下卸载模型**：一位成员询问在使用 LM Studio 的 Just-In-Time 模式时是否能自动弹出未使用的模型，对此官方澄清目前仍需要手动卸载。
   - 可以使用命令 `lms unload --all` 来卸载所有已加载的模型，尽管这仍需要手动触发。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/ChristianAzinn/embedding-ggufs-6615e9f216917dfdc6773fa3">Embedding GGUFs - ChristianAzinn 集合</a>：未找到描述</li><li><a href="https://calendly.com/aslisabanci-01-nvidia/10min">10 分钟会议 - Asli Sabanci</a>：你好！作为 NVIDIA GeForce RTX 团队，我们正在寻求社区 AI 爱好者的建议，以指导未来的产品方向和路线图。我们很想见见你们中一些低代码/无代码的人……</li><li><a href="https://github.com/lmstudio-ai/mlx-engine/pull/22">升级 mlx 和 outlines 依赖，并由 neilmehta24 添加对 llama 3.2 vision 的支持 · Pull Request #22 · lmstudio-ai/mlx-engine</a>：变更摘要：MLX VLM 升级，mlx_vlm 已升级到最新提交。这带来了对 llama 3.2 vision（又名 mllama）的支持。vision_model_kit 和 vision_model_wrapper 已更新以支持……</li><li><a href="https://huggingface.co/mav23/gte-Qwen2-7B-instruct-GGUF">mav23/gte-Qwen2-7B-instruct-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1303824353879785543)** (20 条消息🔥): 

> - `单插槽 RTX 4090`
> - `Mac M2 Pro 内存使用情况`
> - `大模型性能`
> - `实验性 Context 设置`
> - `MacBook M4 测评` 


- **单插槽 RTX 4090 引起关注**：成员们对 **单插槽 RTX 4090** 感到兴奋，注意到其出色的设计和对紧凑型机箱的适用性。
   - *有人评价道：“这家伙为冬天做好了准备”，暗示了这张显卡的酷炫程度。*
- **Mac M2 Pro 引发内存担忧**：一名用户报告其 **Mac M2 Pro** 内存占用过高，在 **10-12K tokens** 下运行 **8B 模型** 需要约 **20GB** 内存。
   - 其他人确认这是正常的，称“**Context 会占用内存**”，但对如此高的比例表示怀疑，认为似乎有些过度。
- **讨论大模型能力**：成员们分享了使用 **70B** 等大型模型的经验，并讨论了优化 **Context size** 的运行配置。
   - 一位用户强调他们通常可以毫无问题地使用 **70B** 模型，同时在思考 **Context 缩放** 对性能的影响。
- **MacBook M4 的测评缺乏深度**：**MacBook M4** 的早期测评被描述为“讨好式的赞美”，缺乏诸如 **温度数据** 或拆解等关键细节。
   - 成员们对测评的可靠性表示怀疑，认为它们没有提供设备的全面视图。
- **推理耗时方面的挑战**：讨论强调了 **推理耗时（inference timing）** 的问题，特别是对于使用 **RAG** 的模型，无论 Context 大小如何，速度都被认为较慢。
   - 一位用户提到正在尝试 **Context 设置**，并表示需要进一步测试以确定模型的准确性和性能。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1303818908364574770)** (55 条消息🔥🔥): 

> - `Stable Diffusion Models`
> - `Outpainting Techniques`
> - `User Interface Generation`
> - `Local Installation Guides`
> - `Discord Community Interaction` 


- **Stable Diffusion 模型缺乏 Web UI 生成能力**：一位用户询问了用于生成 Web 用户界面的模型，但另一位用户指出 **Stable Diffusion 主要用于图像生成**，而非 Web UI。
   - 讨论强调了当前模型在特定设计用途上的局限性。
- **本地运行 Stable Diffusion 的指南**：一位习惯于使用 Google Colab 的新用户寻求在本地设置 Stable Diffusion 的帮助。
   - 另一位成员推荐了一份安装 **ComfyUI 配合 SwarmUI** 作为前端的设置过程指南。
- **Outpainting 咨询与教程**：用户交流了关于 **Outpainting 技术** 的链接和资源，包括 Reddit 帖子以及运行 **Automatic1111** 的教程。
   - 成员之间还分享了关于成功进行 Outpainting 的设置和功能的具体指导。
- **社区问候与闲聊**：多位成员互相打招呼，并就个人近况和所在地进行了轻松的交谈。
   - 评论内容涵盖了从讨论时间到拿用户名开玩笑以及 Discord 社区内的互动。
- **LinkedIn 图像生成咨询**：一位用户就训练哪个模型来为他们的 **LinkedIn 个人资料** 生成逼真图像寻求建议。
   - 社区成员讨论了各种选项，但强调 **Stable Diffusion** 主要侧重于艺术图像生成。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://learn.rundiffusion.com/outpainting/">Outpainting Automatic1111</a>：这是使用 Automatic1111 中的 Inpaint 功能进行 Outpaint 的快速简便方法。第 1 步：创建图像（或已有图像），我制作了这张 RPG 地图。第 2 步：将图像发送到 Img2img：...</li><li><a href="https://www.capcut.com/my-edit?start_tab=video">未找到标题</a>：未找到描述</li><li><a href="https://www.videoleapapp.com/tools/infinite-zoom-ai">Infinite Zoom：使用 AI 实现图像和视频的无限缩放 | Videoleap</a>：立即开始 7 天免费试用。使用 Videoleap 应用程序体验 AI Infinite Zoom。对任何图像或视频进行无限放大和缩小。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/z475bo/how_to_outpaint_on_automatic1111/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://stable-diffusion-art.com/outpainting/">如何使用 Outpainting 扩展图像 - Stable Diffusion Art</a>：Stable Diffusion 可以通过 Outpainting 向任何方向扩展图像。它可以生成视图之外的连贯背景。</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui 安装指南</a>：Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1303864548058464256)** (38 条消息🔥): 

> - `Llama 3.2 Vision`
> - `Aide 开源 IDE`
> - `Claude AI 限制`
> - `训练开源语言模型`
> - `Codebuff CLI 工具` 


- **Ollama 发布 Llama 3.2 Vision**：[Llama 3.2 Vision](https://ollama.com/library/llama3.2-vision) 现已提供 11B 和 90B 两种尺寸，为获得最佳性能分别需要 8GB 和 64GB 的 VRAM。
   - 用户可以通过下载 [Ollama 0.4](https://ollama.com/download) 并使用简单的终端命令轻松运行该模型。
- **Aide IDE：AI 开发领域的新选手**：@ycombinator 宣布了 Aide，这是一个基于 Agent 框架构建的开源 AI 原生代码编辑器，在 swebench-lite 上达到了 43% 的 SOTA。
   - 该工具承诺完全的数据隐私和即插即用的 LLM 集成，吸引了寻找强大编码解决方案的开发者。
- **Claude 免费用户限制**：Claude 的免费用户目前被限制在只能执行像 Haikus 这样的基础任务，无法进行更复杂的操作，如分析大型 CSV 文件。
   - 成员们对这些限制表示沮丧，认为这阻碍了他们利用 AI 进行实质性工作的能力。
- **探索开源语言模型的未来**：讨论中提到了如何开发更好的系统来训练开源语言模型和 Agent，并特别提到了 Tim Dettmers 的见解。
   - 重点强调了克服“API 成瘾”的重要性，以便在 AI 生态系统内实现更多创新。
- **Codebuff CLI 工具介绍**：[Codebuff](https://codebuff.com) 是由 Y Combinator 推出的 CLI 工具，可根据自然语言请求编写代码，并提供无需登录的免费试用。
   - 创始人分享了一个有趣的开发故事，涉及微调 GPT-4o 以生成 git patches，从而实现有效的代码修改。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ollama.com/blog/llama3.2-vision">Llama 3.2 Vision · Ollama 博客</a>: Llama 3.2 Vision</li><li><a href="https://blog.langchain.dev/scipe-systematic-chain-improvement-and-problem-evaluation/">SCIPE - 系统化链条改进与问题评估</a>: 编辑注：我们非常激动地向大家推荐来自伯克利 Ankush Garg 和 Shreya Shankar 的这项研究。在 LangChain，我们思考的两个最大问题是 evals 和 Ag...</li><li><a href="https://x.com/ycombinator/status/1854237314651980257">来自 Y Combinator (@ycombinator) 的推文</a>: Aide 是一个构建在 Agent 框架之上的开源 AI 原生代码编辑器。它在 swebench-lite 上达到了 43% 的 SOTA，并具备你对 Cursor/Copilot 所期望的所有功能，且拥有完整的数据...</li><li><a href="https://www.interconnects.ai/p/tim-dettmers">采访 Tim Dettmers 谈开源 AI：Agent、扩展、量化及未来展望</a>: 立即收听 | Interconnects 访谈 #10。对话开源 AI 领域的领导者之一。</li><li><a href="https://news.ycombinator.com/item?id=42078536">Launch HN: Codebuff (YC F24) – 为你编写代码的 CLI 工具 | Hacker News</a>: 未找到描述</li><li><a href="https://gist.github.com/wesen/122afd23b8bde0ee1e93742b9b3b3c32">api-summary.md</a>: GitHub Gist: 立即分享代码、笔记和代码片段。</li><li><a href="https://youtu.be/hw7EnjC68Fw?si=f4-h6NvSiAp75Gkm">No Priors 第 89 集 | 对话 NVIDIA CEO Jensen Huang</a>: 在本周的 No Priors 节目中，Sarah 和 Elad 第二次与 NVIDIA CEO Jensen Huang 坐下来，回顾公司非凡的...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1303902341191176254)** (28 条消息🔥): 

> - `No Bounds Check 装饰器`
> - `Mojo 标准库开发`
> - `Mojo 与 Python 及 C/C++ 的互操作性`
> - `Mojo 作为 Python 超集`
> - `Mojo 功能时间线` 


- **澄清 No Bounds Check 装饰器**：关于 `@no-bounds-check` 装饰器的讨论强调了它可能被 `@unsafe_no_bounds_check` 取代，成员们建议优先使用 SIMD 加载。
   - 一位成员指出，列表边界检查仅在启用断言（assertions）的编译期间引入开销。
- **关于 Mojo 标准库概览的提案**：有成员建议在 Modular Mojo 网站上创建一个图形化页面，展示 Mojo 的进展及其标准库与 Python 和 C/C++ 的互操作性。
   - 这将为贡献者提供一个全面的概览，以了解可用的标准库模块及其状态，类似于前述的 roadmap。
- **Mojo 从 Python 超集目标的演变**：社区讨论了 Mojo 作为“软超集”而非严格超集的含义，指出采纳 Python 的缺陷可能并无益处。
   - 成员们讨论了支持各种 Python 行为的挑战，并承认在互操作性语境下存在一些细微差别。
- **在 Mojo 中导入和链接 C 模块**：明确了在 Mojo 中导入 C 模块仍需链接的预期，这与某些希望简化导入语法的想法相反。
   - 一个建议是实现一个名为 `mojo` 的 Python 库，以简化 Mojo 模块的导入，类似于 NumPy 等现有库。
- **Mojo 功能的未来之路**：成员们表达了对改进 Mojo、Python 和 C/C++ 之间互操作性的希望，强调无需过度繁琐的链接即可实现平滑导入。
   - 对话指出，在 Python 中使用 Mojo 库之前，有必要将其编译为共享对象（shared objects）或 DLL。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/roadmap">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>：Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://docs.modular.com/mojo/roadmap#cc-interop">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>：Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://github.com/modularml/mojo/commit/cb307d0f57bb07b37528e8f6e2c859a1e07db941">[docs] Clarify meaning of &quot;superset of Python&quot; · modularml/mojo@cb307d0</a>：更新文档以提供比“superset”一词本身更多的上下文。原句为：我们的愿景是让 Mojo 成为 Python 的超集。现在根据上下文调整为：...
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1303838321360764968)** (21 条消息🔥): 

> - `Cohere Reranker API`
> - `Command-R-Plus 问题`
> - `AWS Bedrock 与 SpringAI` 


- **Cohere Reranker 仅通过 API 提供**：*mrdragonfox* 确认 **Cohere Reranker** **仅通过 API 提供**，未列入版本 1 和 2 的文档中。
   - *kenb_80283* 指出端点（endpoints）部分需要更新。
- **Command-R-Plus 表现出异常行为**：*guestavius* 报告称，在 **Command-R-Plus** 中高频出现随机的 **'section'** 插入，这在以前不是问题。
   - *mrdragonfox* 表示该工具主要不是为**角色扮演（roleplay）**设计的，强调其企业级应用。
- **关于 AWS Bedrock Embeddings 的澄清**：*boliveira5781* 询问 AWS Bedrock embed 端点生成的 **embeddings** 是否与输入字符串保持**保序（order preserving）**映射。
   - *enzoloko* 寻求进一步澄清，询问添加新字符串是否会影响现有字符串的位置。

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1304158720783286283)** (1 messages): 

> - `Open-source fine-tuning`
> - `Hugging Face integration`
> - `SageMaker deployment` 


- **通过开源，定制 Cohere 模型变得更加容易**：Cohere 发布了一个名为 `cohere-finetune` 的**开源 Fine-tuning 仓库**，其中包括详细指南和预构建容器，以便使用自定义数据集将基础模型适配到特定任务。
   - *在 [GitHub](https://github.com/cohere-ai/cohere-finetune) 上查看*，轻松获取模型定制功能。
- **使用 Hugging Face 进行参数高效微调 (Parameter-efficient fine-tuning)**：新的 Fine-tuning 仓库与 **Hugging Face 的 Parameter-Efficient Fine-Tuning** 库集成，可在不消耗大量资源的情况下优化模型性能。
   - 这种优化确保用户能够高效地微调模型，同时最大限度地减少资源压力。
- **在 Amazon SageMaker 上部署微调后的模型**：Cohere 在 **Amazon SageMaker** 上提供了“自带微调模型 (Bring Your Own Fine-tune)”推理解决方案，允许部署微调后的模型。
   - 该解决方案利用 SageMaker 的隐私、安全和合规性功能，提供稳健的部署体验。



**提到的链接**：<a href="https://github.com/cohere-ai/cohere-finetune">GitHub - cohere-ai/cohere-finetune: A tool that facilitates easy, efficient and high-quality fine-tuning of Cohere&#39;s models</a>：一个旨在促进 Cohere 模型简单、高效且高质量 Fine-tuning 的工具 - cohere-ai/cohere-finetune

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/)** (1 messages): 

mrdragonfox: <@1303804989629534333> support@cohere.com
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1304168545093029928)** (3 messages): 

> - `AWS Bedrock`
> - `SpringAI Embeddings` 


- **关于 AWS Bedrock 的 Embedding 行为澄清**：一位成员询问了关于在 SpringAI 中使用 AWS Bedrock 的问题，询问向 embed 端点发送字符串列表是否会产生与输入文本保持顺序一致的一一映射。
   - 另一位成员通过确认这一行为进行了回复。
- **对 AWS Bedrock 功能的兴奋**：在得到澄清后，原提问成员表达了热情，简单地说道：*Awesome!!!*。
   - 这反映了用户对 AWS Bedrock 与 SpringAI 结合能力的积极参与和兴奋。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1303849231257305190)** (2 messages): 

> - `Automated Resume Insights`
> - `Context Refinement in RAG Systems` 


- **自动化简历洞察 Agent 的创建**：由 [@Luillyfe](https://twitter.com/Luillyfe) 编写的一篇精彩教程，解释了如何使用核心解析、提取和结构化输出模块构建一个自动化的简历洞察 Agent。
   - 该教程展示了该系统如何高效处理任何非结构化简历，提供有见地的数据收集。
- **通过上下文精炼 (Context Refinement) 增强 RAG 系统**：一篇客座博客文章讨论了如何构建一个上下文精炼 Agent (Context Refinement Agent)，该 Agent 能够智能地扩展和精炼检索到的上下文，从而针对复杂查询提供更好的 RAG 响应。
   - 该 Agent 检查检索到的数据块以提高输出质量，为数据检索和处理增添了新的维度。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1303829466258210816)** (23 messages🔥): 

> - `Ollama Llama Vision 集成`
> - `开源 Chatbot UI`
> - `结合 Chainlit 与 LlamaIndex 实现多模态 RAG`
> - `Llama-Parse 相关资源`
> - `Workflow 中的隔离插桩 (Isolated Instrumentation)` 


- **Ollama Llama Vision 可能会与 Llama Index 集成**：一位用户询问了新的 **Ollama Llama Vision** 功能是否能与 **Llama Index** 开箱即用，并认为它应该可以与 **OllamaMultiModal 类** 配合使用。
   - 另一位成员澄清说，**Ollama 已经支持视觉功能很长时间了**，表明两者已有历史集成。
- **寻找开源 Chatbot UI**：一位用户寻求一个带有身份验证且 UI 类似于 **ChatGPT** 的聊天机器人开源 Web 应用。
   - 成员们建议了 [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui) 等选项，并强调了其功能和使用场景。
- **构建类似 Llama-Parse 解析器的资源**：一位成员请求构建类似于 **Llama-Parse** 解析器的资源，强调数据安全和本地模型的使用。
   - 建议包括使用 **Unstructured** 库，但提醒其功能可能无法完全匹配 **Llama-Parse**。
- **隔离插桩 (Isolated Instrumentation) 的技巧**：一位用户寻求在新的 Workflow 和旧的 Query Pipeline 之间进行隔离插桩的建议，希望仅发送新的 traces。
   - 另一位成员建议在模块级别使用 dispatcher 模式来处理 spans 和 events，从而实现隔离。



**提到的链接**：<a href="https://github.com/mckaywrigley/chatbot-ui">GitHub - mckaywrigley/chatbot-ui: Come join the best place on the internet to learn AI skills. Use code &quot;chatbotui&quot; for an extra 20% off.</a>：加入互联网上学习 AI 技能的最佳场所。使用代码 "chatbotui" 可享受额外 20% 的折扣。 - mckaywrigley/chatbot-ui

  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1304038179808088094)** (2 messages): 

> - `Dott.ai 的未来`
> - `Steve 的愿景` 


- **Dott.ai 宣布宏伟的未来计划**：一位成员分享了 [Dott.ai 未来计划](https://dottxt.co) 的链接，强调了其在该领域的重要性。
   - 来自 Builder.io 的 Steve 指出：“这就是未来”，肯定了他们愿景的重要性。
- **Steve 对未来的兴奋**：Steve 表达了对 Dott.ai 潜力的信心，断言 *这就是未来*。
   - 他的热情凸显了社区内对创新进步日益增长的乐观情绪。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Steve8708/status/1854317061390193116?t=GlXeO2OSk1zGYjArQA2FoA&s=19">来自 Steve (Builder.io) (@Steve8708) 的推文</a>：这就是未来</li><li><a href="https://x.com/dottxtai/status/1854468200156553480?t=fbYLH_w7oySU-55WRUCAsw&s=19">来自 .txt (@dottxtai) 的推文</a>：真正的未来：https://dottxt.co 引用 Steve (Builder.io) (@Steve8708) 的话：这就是未来
</li>
</ul>

</div>

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1303852734906109982)** (19 条消息🔥): 

> - `DSPy Docstring 问题`
> - `理解 DSPy 框架`
> - `EMNLP 2024 演讲`
> - `模块化语言模型优化` 


- **DSPy Docstring 不匹配**：一位用户发现 **DSPy** 框架存在一个问题，即在为 LLM 生成的 Prompt 中仅显示第一个组件的 Docstring，经确认这是由于使用了错误的 Docstring 格式导致的。
   - 经过调试发现，使用 `f"""` 而非 `"""` 会导致 Docstring 无法被正确提取，从而引起用户的困惑。
- **对 DSPy 框架的好奇**：一位用户在阅读相关论文后，对 **DSPy library** 的底层机制表示困惑，并寻求如何有效学习该框架的指导。
   - 建议包括阅读 DSP 论文，并可能通过查阅代码库以获得更深入的理解。
- **潜在的 EMNLP 2024 演讲**：提到了一篇与 DSPy 相关论文的共同第一作者将在 **EMNLP 2024** 上展示他们的工作，引发了对会议期间潜在讨论的兴趣。
   - 用户表达了在活动期间与作者建立联系的愿望。
- **模块化语言模型中的优化策略**：分享了两篇重要论文的链接，概述了优化模块化语言模型流水线的策略，特别关注权重和 Prompt 优化方法。
   - 这些论文解决了 NLP 系统中在没有中间标签或梯度的情况下，需要高效处理模块的挑战。
- **对 DSPy 工作的赞赏**：一位用户对 **DSPy** 项目取得的进展表示钦佩，并称赞了围绕该项目开展的工作，强调了这些贡献令人印象深刻。
   - 该用户表达了探索项目各个方面的热情，表现出与贡献者交流的浓厚兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.11695">Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs</a>：语言模型程序（即模块化语言模型 (LM) 调用的复杂流水线）正在日益推进 NLP 任务，但它们需要精心设计对所有模块都共同有效的 Prompt...</li><li><a href="https://arxiv.org/abs/2407.10930">Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together</a>：自然语言处理 (NLP) 系统越来越多地采用复杂的模块化流水线形式，例如检索增强生成 (RAG)，其中每个模块可能涉及不同的语言模型...
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1303841174150189106)** (11 条消息🔥): 

> - `理解 OS Mode`
> - `Discord 活动时间`
> - `观众限制`
> - `OmniParser 工具`
> - `在 gpt-4o 中使用 OS Mode` 


- **理解 Claude 的 OS Mode**：一位用户寻求关于 OS Mode 如何与 Claude 配合工作的说明，询问 Prompt 是否被转化为代码来控制桌面，以及点击是如何协调的。
   - 另一位成员提供了一个 [GitHub 链接](https://github.com/OpenInterpreter/open-interpreter/blob/development/computer_use/tools/computer.py)，详细介绍了负责鼠标点击的代码。
- **Discord 活动时间困惑**：一位用户询问即将举行的活动是否定于 GMT 晚上 8 点，而另一位成员根据本地时间设置确认活动将在 **30 分钟**后开始。
   - 活动链接的提及表明社区正在持续参与，尽管未给出具体细节。
- **直播观众限制**：关于直播是否有最大观众人数限制的问题被提出，一位成员自信地回复称**不应该有**任何限制。
   - 这一保证反映了社区对容纳大量观众观看直播内容的兴趣。
- **关于 OmniParser 工具的讨论**：一位用户强调 [OmniParser](https://huggingface.co/microsoft/OmniParser) 是一款屏幕解析工具，通过将屏幕截图转换为结构化格式来提高 UI Agent 的性能。
   - 他们引用了各种链接，包括一篇 [博客文章](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/) 和 [演示 (demo)](https://huggingface.co/spaces/microsoft/OmniParser/)，表示对将其应用于 Open Interpreter 感兴趣。
- **OS Mode 与 gpt-4o 的问题**：一位用户在遇到命令格式问题后，询问如何有效地在 gpt-4o 中使用 --os 模式。
   - 这个问题表明在实际场景中，需要更清晰的指令来将 OS 能力与特定模型集成。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/microsoft/OmniParser">microsoft/OmniParser · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/development/computer_use/tools/computer.py">open-interpreter/computer_use/tools/computer.py at development · OpenInterpreter/open-interpreter</a>: 计算机的自然语言接口。通过在 GitHub 上创建一个账户来为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/">Open Interpreter</a>: Open Interpreter 有 6 个可用的仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1303883049016229908)** (3 条消息): 

> - `Python 3.13 兼容性问题`
> - `Conda 环境解决方案` 


- **Python 3.13 兼容性烦恼**：一位用户由于当前的 **Python 3.13** 设置遇到了安装错误，该版本与该包所需的版本不兼容。
   - *被忽略的版本*包括几个需要 Python 在 **3.11 到 4.0** 之间的版本，强调了版本特定性的必要性。
- **Conda 环境解决了问题**：用户创建了一个 Python **3.11** 的 **Conda 环境**，从而成功安装了该包。
   - 虽然操作被指出*速度不快*，但该方案有效地解决了他们的安装问题。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1303892443044909167)** (9 封消息🔥): 

> - `Dedicated Transformer ASIC`
> - `Sohu Hardware Architecture`
> - `Discussion on Availability of Custom Hardware` 


- **首款专用 Transformer ASIC 发布**：一位成员分享了首款专用 Transformer ASIC 已经发布，承诺运行 AI 模型比 **GPU 快 10 倍**，吞吐量超过 **500,000 tokens/second**。
   - 这款名为 **Sohu** 的芯片拥有组播推测性解码 (multicast speculative decoding) 和实时内容生成等能力，将其定位为为 AI 定制的“高速公路”。
- **对“刚刚发布”的质疑**：另一位成员质疑了“刚刚发布”的含义，指出所讨论的博客是六个月前的，并暗示该产品尚未上市。
   - 讨论中出现了关于 **Theranos vibe** 的担忧，暗示对其产品实际存在性与承诺能力的怀疑。
- **不相关讨论被驳回**：一位用户提醒，关于定制硬件的讨论只有在产品可用时才属于相关频道，标志着离题对话的结束。
   - 这表明讨论的重点应集中在 **tinygrad** 的预期用途及其具体使用上。



**提及的链接**：<a href="https://x.com/rohanpaul_ai/status/1854326252674384129?t=EkMykh8rU8fblspV_vFjHQ&s=19">Rohan Paul (@rohanpaul_ai) 的推文</a>：重大消息 🤯 首款专用 Transformer ASIC (Application-Specific Integrated Circuit) 刚刚发布。定制芯片将 Transformer 架构直接烧录进硅片，使 AI 模型运行速度提升 10 倍...

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1304133935852158976)** (3 封消息): 

> - `Using multiple GPUs`
> - `Sharding models`
> - `ThreadPoolExecutor issues` 


- **在 GPU 上运行多个副本**：一位成员询问如何高效地并行使用多个 GPU 来运行模型的多个副本以提高吞吐量，同时避免在 GPU 之间进行模型分片 (sharding)。
   - 他们提到使用了 `concurrent.futures.ThreadPoolExecutor`，但在加载 tensor 时遇到了锁定挑战。
- **分片技术详解**：作为回应，一位成员分享了 `x.shard(GPUS, axis=None)` 的方法，该方法可以在所有 GPU 上放置模型副本而不对模型进行切片。
   - 此外，他们建议使用 `x.shard(GPUS, axis=0)` 来跨指定轴对输入进行切片，以实现高效分发。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1303842592080924693)** (8 封消息🔥): 

> - `ScheduleFree SOAP`
> - `Optimizer Hyperparameters`
> - `Model Merging and MOEs`
> - `CAME Optimizer` 


- **ScheduleFree SOAP 的优势**：据称 [ScheduleFree SOAP 实现](https://github.com/ClashLuke/HeavyBall/blob/main/heavyball/schedule_free_palm_foreach_soap.py#L296) 具有更高的计算和内存效率，由于允许更高的学习率，收敛速度更快。
   - *与 SOAP/Adam 相比*，它建议更改超参数，例如使用 PaLM 的 beta2 调度并进行 10% 的预热 (warmup)。
- **关于 MOEs 和模型合并的讨论**：一位成员询问是否还有人在研究 MOEs 或模型合并，并指出自 llama 3.2 以来这些研究似乎消失了。
   - 另一位成员观察到，目前他们看到的主要是关于 llama 3.2 finetunes 的讨论。
- **关于 CAME 对比的咨询**：一位用户询问 ScheduleFree SOAP 与 [CAME 优化器](https://github.com/yangluo7/CAME) 的对比情况。
   - 另一位成员澄清 CAME 是另一种不同的优化器，并提供了其官方实现的链接。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/yangluo7/CAME">GitHub - yangluo7/CAME: "CAME: Confidence-guided Adaptive Memory Optimization" 的官方实现</a>：yangluo7/CAME - "CAME: Confidence-guided Adaptive Memory Optimization" 的官方实现</li><li><a href="https://github.com/ClashLuke/HeavyBall/blob/main/heavyball/schedule_free_palm_foreach_soap.py#L296">HeavyBall/heavyball/schedule_free_palm_foreach_soap.py at main · ClashLuke/HeavyBall</a>：各种优化器的实现；主要专注于快速的 _foreach 和 PaLM 版本 - ClashLuke/HeavyBall
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1303981081036455967)** (3 条消息): 

> - `Adding special tokens` (添加特殊 Token)
> - `Fine-tuning with LORA` (使用 LORA 进行微调)


- **为微调添加特殊 Token 的正确方法**：要为 LLM 微调添加新的 **special token**，应在训练前将该 Token 添加到 tokenizer 中，并在 Axolotl 配置中包含 `special_tokens: reference_text: <|reference_text|>`。
   - 成员们确认此方法是正确的，并强调即使使用 LORA，模型也会学习新的 Token。
- **LORA 在学习新 Token 方面的有效性**：一位成员提到，虽然模型会通过 **LORA** 学习新 Token，但其效果不如进行全量微调（full fine-tuning）。
   - 除了使用 LORA，保存 `embed_tokens` 和 `lm_head` 等模块对于获得更好的训练结果也很重要。


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1303973222055936011)** (4 条消息): 

> - `LR Scheduler Usage` (LR Scheduler 使用)
> - `Ichigo Project Implementation` (Ichigo 项目实现)
> - `Adding Scheduler to Recipes` (向 Recipes 添加 Scheduler)


- **全量微调中的 LR Scheduler 问题**：一位用户指出在执行 **full_finetune_distributed** 时无法使用 **lr_scheduler** 的问题，特别是在尝试将其添加到配置文件时。
   - 他们寻求关于在此过程中如何正确使用 scheduler 的指导。
- **LR Scheduler 集成的待解决问题**：一位成员引用了 GitHub 上的一项 [open issue](https://github.com/pytorch/torchtune/issues/1308)，内容涉及计划为全量微调 recipes 添加 LR scheduler 支持，目前该功能尚不可用。
   - 该 issue 概述了将 scheduler 功能整合到分布式训练中所需的更改。
- **Ichigo 项目实现评审**：另一位用户分享了 [Ichigo 项目](https://github.com/homebrewltd/ichigo)的链接，该项目使用了 Torchtune，并请求对其正确性进行验证。
   - 该项目旨在使 **Llama3.1** 更具交互性，但用户对其实现质量表示不确定。
- **对自定义 Recipe 调整的项目兴趣**：一位成员指出，对 recipes 进行修改是可行的，正如 Ichigo 项目中添加的功能所示，并且没有发现明显的错误。
   - 他们还保证，预计在未来几周内 recipes 将提供对 LR scheduler 的官方支持。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/1308">Add LR Scheduler to full finetune recipes · Issue #1308 · pytorch/torchtune</a>：效仿我们在 LoRA recipes 中的做法，我们应该为全量微调 recipes 添加使用 LR scheduler 的能力。以下是需要更改的所有内容：向 r... 添加适当的函数。</li><li><a href="https://github.com/homebrewltd/ichigo">GitHub - homebrewltd/ichigo: Llama3.1 learns to Listen</a>：Llama3.1 学会倾听。通过在 GitHub 上创建账号为 homebrewltd/ichigo 的开发做出贡献。</li><li><a href="https://github.com/homebrewltd/torchtune/blob/1ff4fc217e578a1656dc60c343dcee98e7ac013e/recipes/full_finetune_fsdp2.py">torchtune/recipes/full_finetune_fsdp2.py at 1ff4fc217e578a1656dc60c343dcee98e7ac013e · homebrewltd/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 homebrewltd/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1303973535231901746)** (2 条消息): 

> - `LLM Course Cohorts` (LLM 课程班次)


- **明年将推出高级 LLM 课程**：一位成员确认明年将开设 **LLM 课程**，这将是一个**高级版本**，教材内容与当前提供的不同。
   - *这强调了课程体系为满足不断变化的需求而进行的持续演进。*
- **课程材料将会改变**：与目前涵盖的内容相比，即将到来的课程将采用**不同的材料**。
   - 成员们似乎对明年将引入哪些具体的**高级主题**很感兴趣。


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1303903474093654036)** (1 条消息): 

> - `Function extraction`
> - `Dataset files` 


- **从 dataset files 中提取 functions**：有人建议从 **dataset files** 的条目中提取 **functions** 及其定义，以编制一份完整的列表。
   - 有人指出，目前还没有现成的已编译资源。
- **没有可用的已编译资源**：成员们承认，目前缺乏针对 dataset files 中 functions 的预先存在的已编译资源。
   - 这突显了协作创建此类汇编的必要性。


  

---



---



---



---



---



---



{% else %}


> 为了方便邮件阅读，完整的逐频道详情已被截断。
> 
> 如果您想查看完整的详情，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}