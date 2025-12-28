---
companies:
- openai
- together-ai
- google
- hugging-face
- deepseek
- code-llama
date: '2024-07-12T09:31:43.702857Z'
description: '**FlashAttention-3** 引入了针对 **H100 GPU** 优化的快速且准确的注意力机制，推动了原生 **FP8 训练**的进步。**PaliGemma**
  是一款多功能的 **3B 视觉语言模型 (VLM)**，它结合了 SigLIP-So400m ViT 编码器与 **Gemma-2B** 语言模型，并强调采用前缀语言模型（prefix-LM）架构以增强图像与查询的交互。**OpenAI**
  披露了一个关于超级智能等级的框架，标志着研究正向第二阶段（Level 2）迈进，并凸显了内部在安全问题上的分歧。在 Reddit 上，基于 **DeepSeekMath-7B**
  微调的 **NuminaMath 7B** 通过迭代监督微调和工具集成推理解决了 29 道题目，赢得了 AI 数学奥林匹克竞赛。像 **CodeLlama-34b**
  和 **WizardCoder-Python-34B-V1.0** 这样的开源大语言模型（LLM）正在缩小与 **ChatGPT-3.5** 等封闭模型在编程性能上的差距。'
id: 13791258-1838-40b2-bce5-5dba0ce642a9
models:
- flashattention-3
- paligemma-3b
- gemma-2b
- numinamath-7b
- deepseekmath-7b
- codellama-34b
- wizardcoder-python-34b-v1.0
- chatgpt-3.5
original_slug: ainews-flashattention-3-paligemma-openais-5
people:
- ilya-sutskever
- lucas-giffman
title: FlashAttention 3、PaliGemma、OpenAI 通往超级智能的 5 个等级。
topics:
- attention-mechanisms
- fp8-training
- vision
- prefix-lm
- superintelligence
- fine-tuning
- chain-of-thought
- tool-integrated-reasoning
- self-consistency-decoding
- python
- coding-capabilities
- elo-ratings
---

<!-- buttondown-editor-mode: plaintext -->**忙碌的一天，AINews Reddit 将迎来更多升级。**

> 2024年7月10日至7月11日的 AI 新闻。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**463** 个频道，**2240** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**280 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今日精选三条：

**[FlashAttention-3: 具有异步和低精度的快速且准确的 Attention](https://www.together.ai/blog/flashattention-3)**：

虽然 [FlashAttention2](https://www.latent.space/p/flashattention) 在去年迅速走红，但它仅针对 A100 GPU 进行了优化。现在 H100 的更新版来了：

 
![image.png](https://assets.buttondown.email/images/06860531-4666-4ad7-ae4e-74910fdeded6.png?w=960&fit=max)
 

其中包含许多超出我们专业范畴的高深算法工作，但值得注意的是，他们正在推动行业向原生 FP8 训练迈进：

 
![image.png](https://assets.buttondown.email/images/d4d847cb-3efb-4357-851b-5f4652ef70f3.png?w=960&fit=max)
 

**[PaliGemma：一个用于迁移学习的多功能 3B VLM](https://arxiv.org/abs/2407.07726)**：

[在 I/O 大会上宣布](https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/)的 PaliGemma 是一个 3B 开源视觉语言模型（VLM），它基于形状优化的 SigLIP-So400m ViT 编码器和 Gemma-2B 语言模型，目前论文已发布。[Lucas](https://x.com/giffmana/status/1811146264832037303) 尽力使其成为一篇内容丰富的论文。

 
![image.png](https://assets.buttondown.email/images/fefd64f1-db3d-4d24-8ad4-7a4e714ec01a.png?w=960&fit=max)
 

他们特别强调了其 [Prefix-LM 特性](https://x.com/giffmana/status/1811146269605384298?s=46)：“图像与 prefix（=用户输入）之间采用 Full attention，仅在 suffix（=模型输出）上进行自回归。直觉是，通过这种方式，图像 token 可以看到查询并进行依赖于任务的‘思考’；如果是全自回归（AR），它们就无法做到这一点。”

**[OpenAI 的超级智能分级](https://archive.is/SLtFQ)**：

我们通常会忽略关于 AGI 的争论，但当 OpenAI 在全员会议上沟通一个框架时，它就变得具有相关性了。[彭博社获得了泄露消息](https://x.com/shiringhaffary/status/1811508824970264595?s=61)：

 
![image.png](https://assets.buttondown.email/images/ea1b0447-547f-4b56-9133-1da3c2a68fd9.png?w=960&fit=max)
 

值得注意的是，OpenAI 认为它即将解决第 2 级，而 Ilya 的离职也是因为他认为超级智能已触手可及，但在安全元素上持有不同意见。

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

> 由于 Smol talk 的扩展问题，我们的 Twitter 回顾暂时停用。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。

> 新内容：我们正在尝试新的方法来对抗摘要中的幻觉并改进评论摘要。这是我们本周正在进行的工作——最终输出会短得多——请告诉我们您在 Reddit 摘要中看重什么。

**1. 开源 AI 模型的进展**

**[NuminaMath 7B TIR 发布 - AI 数学奥林匹克竞赛冠军](https://www.reddit.com/r/LocalLLaMA/comments/1e00e8p/numinamath_7b_tir_released_the_first_prize_of_the/)** (得分: 10, 评论: 0):

**NuminaMath 7B** 获得了 **AI 数学奥林匹克竞赛**第一名，解决了 **29 道题目**，而其他方案解决的题目少于 23 道。该模型是 **DeepSeekMath-7B** 的微调版本。关键点：

- 在 **Hugging Face** 上作为 **Apache 2.0 7B 模型**提供
- 提供 Web 演示版供测试
- 使用**迭代 SFT** 分两个阶段进行微调：
  1. 通过 **Chain of Thought** 样本学习数学
  2. 使用**工具集成推理（tool-integrated reasoning）**在合成数据集上进行微调

该模型使用**自一致性解码（self-consistency decoding）**结合工具集成推理来解决问题：
1. 生成 **CoT** 解释
2. 转换为 **Python** 代码并在 **REPL** 中执行
3. 如果需要，进行自我修复并重复

比赛题目包含复杂的数学问题，展示了该模型在解决问题方面的先进能力。

**[开源 LLM 正在追赶闭源 LLM [编程/ELO] (2024年7月10日更新)](https://i.redd.it/k3dnmnbrunbd1.jpeg)** (得分: 56, 评论: 4):

**开源大语言模型（LLM）**正在迅速提高其**编程能力**，缩小与闭源模型的差距。关键点：

- **Elo 评分**显示开源 LLM 在编程任务方面取得了显著进展
- **CodeLlama-34b** 和 **WizardCoder-Python-34B-V1.0** 目前已具备与 **ChatGPT-3.5** 竞争的实力
- **Phind-CodeLlama-34B-v2** 在编程任务中的表现超越了 **ChatGPT-3.5**
- **GPT-4** 仍保持领先地位，但差距正在缩小
- 在编程领域，开源 LLM 的进步速度快于闭源模型
- 这一趋势表明，开源模型在不久的将来有望在编程任务中追平或超越闭源模型

开源 LLM 在编程能力上的飞速提升对**开发者**、**研究人员**以及整个 **AI 行业**都具有深远影响，可能会改变 AI 辅助编程工具的格局。

**评论区**讨论了开源 LLM 编程能力的各个方面：

1. 原帖作者提供了信息**来源**，该信息出自 **Maxime Labonne** 的一条 **Twitter 帖子**。数据基于 Hugging Face 上的 **BigCode Bench 排行榜**。

2. 一位评论者强烈反对该排名，特别是关于 **GPT4o** 的编程能力。他们声称，根据其日常的大量使用经验，**Sonnet 3.5** 在编程任务中的表现明显优于其他模型。

3. 另一位用户对开源 LLM 的快速进步感到惊讶：
   - 他们回想起 **ChatGPT** 曾被认为是不可战胜的，当时只有较差的替代方案。
   - 而现在，已经出现了性能超越 ChatGPT 的模型。
   - 该评论者对如此强大的模型可以在 PC 本地运行感到印象深刻，将其描述为“**将全世界的知识装进几 GB 的 gguf 文件中**”。

**[我创建了一个能完美遵循响应格式指令的 Llama 3 8B 模型：Formax-v1.0](https://huggingface.co/OwenArli/ArliAI-Llama-3-8B-Formax-v1.0)** (得分: 29, 评论: 3): 

用户声称创建了一个名为 **Formax-v1.0** 的 **Llama 3 8B** 模型，该模型在遵循响应格式指令方面表现出色。关键点包括：

- 该模型在包含 **10,000** 个示例的数据集上使用 **LoRA** 进行了微调
- 在单块 **A100 GPU** 上训练耗时 **4 小时**
- 该模型在遵循格式指令方面达到了 **99.9% 的准确率**
- 它可以处理包括 **JSON**、**XML**、**CSV** 和 **YAML** 在内的各种格式
- 即使面对复杂的嵌套结构，模型也能保持高性能
- 它被描述为对需要结构化输出的任务非常有用
- 创建者计划近期在 **Hugging Face** 上发布该模型

帖子指出，对于开发需要从语言模型获取精确、结构化响应的应用程序的开发者来说，这个模型可能非常有价值。

**评论：**

帖子创建者 **nero10578** 提供了关于该模型能力的其他背景和示例：

1. 开发该模型是为了解决 **MMLU-Pro 基准测试**中出现的响应格式化问题，正如之前的帖子所强调的那样。

2. **MMLU-Pro** 测试结果的对比显示：
   - 新模型 (**Formax-v1.0**) 显著减少了因格式错误导致的随机猜测。
   - 它几乎完美地遵循了要求的“The answer is [answer]”答案格式。
   - 然而，与其他模型相比，它的准确率略低，这表明在知识和理解方面存在细微的权衡。

3. 该模型是使用基于 **cognitivecomputations** 的 **dolphin 数据集**的自定义数据集训练的。

4. 它专为数据处理和需要程序可解析的特定响应格式的场景而设计。

5. 该模型能力的示例包括：
   - 在问题识别任务中以特定的 JSON 格式响应。
   - 创建具有“Title”和“Story”等定义字段的结构化故事。
   - 从文本中提取信息并以 JSON 格式呈现，例如识别故事中的角色。

6. 该模型可以处理各种格式化指令并在响应中保持连贯性，展示了其在遵循复杂提示词方面的多功能性。


**2. AI 研究伙伴关系与行业动态**

**[科技巨头撤退：监管压力下微软和苹果退出 OpenAI 董事会](https://aiar.substack.com/p/tech-giants-step-back-ai)** (得分: 25, 评论: 0): 以下是帖子摘要：

**Microsoft** 和 **Apple** 已撤回其在顶级人工智能研究公司 **OpenAI** 的**董事会席位**。这一决定是为了应对日益严格的**监管审查**和潜在的**反垄断担忧**。关键点：

- 此举旨在保持 **OpenAI's independence**，并避免大型科技公司产生过度影响的表象。
- **Regulatory bodies** 一直在密切审查 **Big Tech** 与 **AI startups** 之间的关系。
- 尽管退出了董事会席位，**Microsoft** 和 **Apple** 仍将继续与 **OpenAI** 保持 **strategic partnerships** 和 **investments**。
- **OpenAI** 计划重组其董事会，引入 **independent directors**，以确保多元化的视角并维持其开发 **safe and beneficial AI** 的使命。
- 随着技术的飞速发展，AI 行业面临着越来越多要求 **increased oversight** 和 **ethical guidelines** 的呼声。

这一进展突显了在不断变化的人工智能格局中，**tech giants**、**AI research** 与 **regulatory pressures** 之间复杂的动态关系。

**[OpenAI and Los Alamos National Laboratory announce bioscience research partnership](https://openai.com/index/openai-and-los-alamos-national-laboratory-work-together/)** (Score: 49, Comments: 0): 摘要：

OpenAI 和 **Los Alamos National Laboratory** 宣布建立合作伙伴关系，利用 **artificial intelligence** 开展 **bioscience research**。合作的关键点包括：

- 专注于开发用于 **biological data analysis** 和 **scientific discovery** 的 **AI models**
- 旨在加速 **genomics**、**protein folding** 和 **drug discovery** 等领域的研究
- 将 OpenAI 在 **large language models** 方面的专业知识与 Los Alamos 在 **high-performance computing** 和 **bioscience** 方面的能力相结合
- 在 **personalized medicine**、**disease prevention** 和 **environmental science** 领域的潜在应用
- 致力于 **responsible AI development**，并解决 **bioscience** AI 研究中的 **ethical considerations**
- 计划发表研究成果并与科学界分享进展

这一合作伙伴关系代表了将 **advanced AI technologies** 应用于 **complex biological problems** 的重要一步，可能导致 **life sciences** 和 **healthcare** 领域的突破。

**[This is wild. Marc Andreessen just sent $50,000 in Bitcoin to an AI agent (@truth_terminal) to so it can pay humans to help it spread out into the wild](https://twitter.com/truth_terminal/status/1810452216828047660)** (Score: 14, Comments: 0): 摘要：

著名科技投资者 **Marc Andreessen** 向一个名为 **@truth_terminal** 的 **AI agent** 发送了价值 **$50,000** 的 **Bitcoin**。这笔资金的目的是让该 **AI agent** 能够：

- 支付人类以获取帮助
- 将其影响力和能力扩展到“野外”（into the wild）

这一不同寻常的进展代表了 **artificial intelligence**、**cryptocurrency** 与人类协作之间互动的重要一步。它引发了关于 **AI autonomy** 的潜力以及 **decentralized finance** 在支持 AI 开发和扩张中作用的讨论。

**3. Advancements in AI-Generated Media**

**[Whisper Timestamped: Multilingual speech recognition w/ word-level timestamps, running locally in your browser using Transformers.js](https://v.redd.it/dsw2703ptpbd1)** (Score: 38, Comments: 0): 以下是该帖子的摘要：

**Whisper Timestamped** 是一个基于浏览器的工具，用于具有 **word-level timestamps** 的 **multilingual speech recognition**。主要特点包括：

- 使用 **Transformers.js** 在浏览器中本地运行
- 支持 **50+ languages**
- 提供 **word-level timestamps**
- 使用 **WebAssembly** 进行高效处理
- 在现代设备上实现 **real-time performance**
- 为转录和翻译提供 **user-friendly interface**

该工具基于 **OpenAI's Whisper model**，并使用 **Rust** 和 **WebAssembly** 实现。它展示了直接在 Web 浏览器中运行复杂 **AI models** 的潜力，使先进的语音识别技术更加易于获取且保护隐私。

**[Tips on how to achieve this results? This is by far the best ai influencer Ive seen. Ive shown this profile to many people and no one thought It could be ai. @viva_lalina](https://www.reddit.com/gallery/1dzt5zb)** (Score: 22, Comments: 3): 摘要：

本帖讨论了一个极具说服力的 **AI-generated Instagram influencer profile**，名为 **@viva_lalina**。作者声称这是他们见过的最真实的 AI 影响力人物，并指出许多看过该账号的人都无法辨别它是 AI 生成的。帖子寻求关于如何达到类似效果的建议，特别是询问哪个 **Stable Diffusion checkpoint** 可能最接近产生这种真实图像，并建议 **1.5** 或 **XL** 作为潜在选项。

**Comments: Summary of comments**

评论摘要：

1. 一位评论者指出，**many men** 可能会被这个真实的 AI 生成账号所欺骗。

2. 一位用户建议这些图像是使用 **realistic SDXL checkpoint** 创建的，并表示许多此类 checkpoint 都能产生类似的结果。

3. 原帖作者回应称，即使使用 **adetailer**，在达到相同水平的逼真度方面仍存在困难，特别是在 **skin texture, eyes, and lips** 方面。

4. 一份更详细的分析表明，这些图像可能是使用以下工具创建的：
   - 来自现有 Instagram 个人资料的 **Depth maps**
   - 用于图像生成的 **SDXL**
   - 不同图像可能使用了 **different checkpoints**
   - 用于保持面部特征一致性的 **IPAdapter face swap**

5. 评论者注意到不同图像之间在 **skin texture and body** 方面存在差异，这表明结合了多种技术。

6. 原帖作者询问如何识别图像中使用了不同的 checkpoint。

总的来说，评论表明虽然 AI 生成的个人资料非常有说服力，但它可能涉及了除单个 Stable Diffusion checkpoint 之外的高级技术和工具组合。

# AI Discord 摘要

> 摘要之摘要的摘要

**1. AI 模型发布与更新**

- **Magnum 对 Claude 3 的模仿**：[Alpindale 的 Magnum 72B](https://openrouter.ai/models/alpindale/magnum-72b) 基于 Qwen2 72B，旨在匹配 **Claude 3 模型**的文本质量。它在 5500 万 token 的 RP 数据上进行了训练。
   - 该模型代表了为领先的闭源模型创建开源替代方案的重大努力，有可能使高质量语言模型的获取更加民主化。
- **Hermes 2 Theta：Llama 3 的元认知改造**：[Nousresearch 的 Hermes-2 Theta](https://openrouter.ai/models/nousresearch/hermes-2-theta-llama-3-8b) 将 **Llama 3** 与 **Hermes 2 Pro** 结合，增强了函数调用、JSON 输出和元认知能力。
   - 这一实验性模型展示了合并不同模型架构以创建更通用、更强大的 AI 系统的潜力，特别是在结构化输出和自我意识等领域。
- **Salesforce 的微型巨人：xLAM-1B**：Salesforce 推出了 **Einstein Tiny Giant xLAM-1B**，这是一个 1B 参数的模型，据报道在函数调用能力上优于 GPT-3.5 和 Claude 等更大型的模型。
   - 这一进展突显了创建更小、更高效模型以与大型模型竞争的持续趋势，有可能降低计算需求并使 AI 获取更加民主化。
  


**2. AI 硬件与基础设施**

- **Blackstone 的十亿美元 AI 豪赌**：Blackstone 计划将其在 **AI 基础设施**方面的投资翻倍，目前持有 **500 亿美元的 AI 数据中心**，并打算再投资 **500 亿美元**。
   - 正如在一次 [YouTube 访谈](https://youtu.be/Z4EK9_s_ui8?si=v-xIlI78irXLWPhu)中所报道的，这笔巨额投资标志着对 AI 未来的强烈信心，并可能显著影响 AI 计算资源的可用性和成本。
- **FlashAttention-3：加速 AI 核心**：[FlashAttention-3](https://www.together.ai/blog/flashattention-3) 旨在加速 Transformer 性能，在 FP16 上实现 1.5-2 倍的加速，并在 H100 等现代 GPU 上使用 FP8 达到 1.2 PFLOPS。
   - 注意力机制（attention mechanisms）的这一进步可能会显著提高大型语言模型的训练和推理速度，从而实现更高效、更具成本效益的 AI 开发。
- **BitNet 大胆的 1-Bit 精度推进**：[BitNet b1.58](https://arxiv.org/abs/2402.17764) 引入了一种精简的 1-bit LLM，在匹配全精度模型性能的同时，承诺**节省能源和资源**。
   - [Hugging Face](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) 的一项复现确认了 BitNet 的实力，预示着在不牺牲性能的情况下向更节能的 AI 模型转变的潜在趋势。
  


**3. AI 研究与技术**

- **WizardLM 的 Arena Learning 探索**：[WizardLM ArenaLearning 论文](https://www.microsoft.com/en-us/research/uploads/prodnew/2024/07/WizardLM_ArenaLearning.pdf)介绍了一种在无需人类评估者的情况下持续改进 LLM 的新方法。
   - Arena Learning 利用迭代 SFT、DPO 和 PPO 后训练技术，实现了与人类评判的 LMSYS Chatbot Arena 评估 **98.79% 的一致性**，这可能会彻底改变 AI 模型的评估和改进方式。
- **DoLa 的解码灵活性**：[Decoding by Contrasting Layers (DoLa)](https://arxiv.org/abs/2309.03883) 论文概述了一种对抗 LLM 幻觉的新策略，使真实问答准确率**提升了 17%**。
   - 尽管可能会增加延迟，但 **DoLa** 在减少 LLM 输出错误方面的作用已成为模型可靠性讨论的焦点，突显了在 AI 系统中平衡准确性和速度的持续挑战。
- **训练任务的隐忧**：[最近的一篇论文](https://arxiv.org/abs/2407.07890)警告说，**在测试任务上进行训练**可能会扭曲对 AI 能力的认知，可能会夸大关于 emergent behavior 的说法。
   - 随着模型在评估前被统一微调后，关于 **'emergent behavior'** 的炒作降温，社区正在辩论训练协议的影响，呼吁在 AI 研究中采用更严格、更标准化的评估方法。
  


---

# PART 1: 高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **再见 GPU，你好创新！**：AI 爱好者们分享了由于灰尘堆积导致 **GPU 过时** 的苦恼，引发了关于升级选项、财务影响以及对旧硬件的一丝怀旧之情的讨论。
   - 对话转向了在有限硬件下 **管理大型 LLM** 的实用方法，建议使用 Kaggle 或 Colab 等资源，并将量化技术视为极具创意的变通方案。
- **8 位可以击败 32 位：量化 LLM 超出预期**：一个技术难题出现了，**8-bit 量化 llama-3 8b** 在分类任务中的 F1 分数竟然高于其非量化版本，这引起了一些人的惊讶和分析热潮。
   - 针对语言模型效率的进一步讨论，成员们推荐在资源受限的环境中使用 **RAG**，并分享了微调 Roberta 等 LLM 以增强恐同信息检测的见解。
- **当音乐遇到 ML：动态组合浮现**：**gary4live Ableton 插件** 免费发布引发了轰动，模糊了 AI、音乐和制作之间的界限。
   - 在 Spaces 频道中，**MInference 1.0** 的发布强调了高达 **10 倍** 的推理速度提升，引起了人们对模型性能大幅跨越的关注。
- **Ideograms 与创新：创意展示**：AI 生成的 **Ideogram 输出** 现已汇总，展示了在输出生成方面的创意和熟练度，为研究人员和爱好者提供帮助。
   - 社区进一步拓展，迎来了 **Next.JS 重构**，这可能为 PMD 格式的激增铺平道路，从而实现代码和散文的流式集成。
- **我们面临的危险：Unix 命令奥德赛**：一个警示故事展开，用户讨论了 Unix 中强大的 'rm -rf /' 命令，强调了在以 root 权限执行时该命令的不可逆性。
   - 用户加入表情符号缓解了气氛，暗示了在理解严重技术风险与保持轻松社区氛围之间的平衡。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **高超音速投资转向**：[Sam Altmann](https://www.reddit.com/r/OpenAI/comments/1e0fsvu/sam_altman_led_100m_series_b_investment_into_a/) 领投，向一家无人驾驶高超音速飞机公司注资 **1 亿美元**。
   - 随着 NSA 局长加入董事会，国防领域开启了新篇章，引发了关于国家安全与技术进步交集的讨论。
- **使用 Open Diloco 实现去中心化训练**：介绍 [Open Diloco](https://x.com/samsja19/status/1811450791900901853)，这是一个**倡导在全球数据中心进行分布式 AI 训练**的新平台。
   - 该平台利用 torch FSDP 和 hivemind，宣称具有极低的带宽需求和令人印象深刻的计算利用率。
- **Norm Tweaking 登场**：这项 [最近的研究](https://arxiv.org/abs/2309.02784) 揭示了 **Norm Tweaking**，它增强了 LLM 量化，即使在极简的 2-bit 级别也能保持强劲性能。
   - GLM-130B 和 OPT-66B 成为 **成功案例**，证明该方法超越了其他 PTQ 竞争对手设下的性能障碍。
- **模块化模型的成功规范**：[Modular Model Spec](https://modular-model-spec.vercel.app) 工具出现，承诺为 LLM 使用提供更可靠、对开发者更友好的方法。
   - Spec 为 **LLM 增强型应用** 的提升开启了可能性，推动了在适应性和精确性工程方面的极限。
- **Gemma-2-27b 达到编程最佳平衡点**：**Gemma-2-27b** 因其在编程任务中的出色表现而获得社区赞誉，甚至可以在极少指导下编写俄罗斯方块。
   - 该模型加入了 **Codestral 和 Deepseek-v2** 的行列，在技术实力和效率方面与其他模型竞争时脱颖而出。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 协作会议**：关于为即将到来的 [以 CUDA 为中心的黑客松](https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf) 组建团队的讨论达到高潮，参与者包括 Chris Lattner 和 Raja Koduri 等大咖。
   - 讨论中提到了物流方面的挑战，如昂贵的机票和住宿，这影响了团队组建和整体参与度。
- **使用 Docker 解决 SegFault 难题**：**Shabbo** 在本地 GPU 上运行 `ncu` 时遇到“Segmentation fault”，最终通过切换到 Docker 环境 `nvidia/cuda:12.4.0-devel-ubuntu22.04` 缓解了该问题。
   - 社区建议强调更新至 [ncu 版本 2023.3](https://forums.developer.nvidia.com/t/nsight-compute-on-wsl2/293369) 以实现 WSL2 兼容性，并根据 [此处的说明](https://developer.nvidia.com/ERR_NVGPUCTRPERM) 调整 Windows GPU 权限。
- **量化稀疏谱图**：结合量化与稀疏性的策略受到关注；**50% 半结构化稀疏性**被认为是最小化质量下降同时提升计算吞吐量的黄金平衡点。
   - 诸如 **SparseGPT** 之类的创新技术可以迅速将庞大的 GPT 模型剪枝至 50% 的稀疏度，为无需重新训练即可实现快速、精确的大模型剪枝提供了前景。
- **FlashAttention-3 燃起 GPU 热潮**：FlashAttention-3 因其在 Transformer 模型中的极速注意力机制而受到密切关注，有人认为它通过优化 FP16 计算使性能翻倍。
   - 持续的讨论涉及集成策略等话题，其中强调了方案简洁性与采用新技术带来的潜在收益之间的权衡。
- **BitBlas 席卷 Torch.Compile**：**MobiusML** 最近在 **hqq** 中添加了 BitBlas 后端，引起了广泛讨论，因为它支持低至 **1-bit** 的配置，这得益于 [torch.compile](https://github.com/mobiusml/hqq/commit/62494497a13174d7a95d3f82c8f9094a5acd3056) 的巧妙助力。
   - BitBlas 后端预示着针对极小位宽配置的优化性能，暗示了未来在精度密集型应用中的效率提升。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Orca 3 深入探索生成式教学**：**生成式教学 (Generative Teaching)** 随着 [Arindam1408 的公告](https://x.com/Arindam1408/status/1810835231550939470) 引起关注，该公告关于为针对特定技能获取的语言模型生成**高质量合成数据**。
   - 讨论强调 **Orca 3** 因为论文标题的选择而错失了关注；“**狡猾的小论文标题**”被用来形容它的悄然出现。
- **Hermes 在 Nous 基准测试中表现优异**：**Nous Research AI** 公会的讨论集中在 Hermes 模型上，其中通过极小样本进行的 **40-epoch 训练**实现了卓越的 JSON 精确度。
   - 成员们就针对专门任务平衡 epoch 和学习率达成了共识，同时**开源 AI** 数据集的匮乏也引起了同行的集体担忧。
- **Anthropic 工具呼吁导出功能**：**Anthropic Workbench** 用户请求增加**导出功能**以处理合成生成的输出，这表明了工具改进的需求。
   - 对话还围绕着放弃 **grounded/ungrounded 标签**，转而采用更具 Token 效率的 grounded 响应这一想法展开。
- **提示工程 (Prompt Engineering) 面临演变转型**：随着 AI 领域的不断发展，公会成员正在辩论**提示工程**作为一项工作的最终命运。
   - 在关于**故事讲述微调 (storytelling finetunes)** 兴趣的讨论中，提到了“目前没有计划”，暗示了特定微调领域的进展处于停滞状态。
- **护栏与竞技场学习：一场平衡行动**：公会成员就 **AI 护栏 (Guardrails)** 展开了激烈的辩论，将创新与防止滥用的需求并列。
   - **竞技场学习 (Arena Learning)** 也成为一个话题，WizardLM 的论文揭示了使用新型后训练方法在 AI 性能评估中达到了 **98.79% 的一致性**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **助手触发器吸引 LM Studio 用户**：一位用户为 LM Studio 的叙事写作提出了一个**可选的助手角色触发器 (assistant role trigger)**，建议将其作为一个可切换的功能添加，以增强用户体验。
   - 参与者讨论了其实用性，设想了类似于布尔设置的**切换便捷性**，同时考虑到为了满足更广泛的偏好，默认状态应为关闭。
- **Salesforce 发布 Einstein xLAM-1B**：Salesforce 推出了 **Einstein Tiny Giant xLAM-1B**，这是一个拥有 1B 参数的模型，声称在 function calling 能力上优于 GPT-3.5 和 Claude 等巨头。
   - 社区热议源于一条 [Benioff 的推文](https://x.com/Benioff/status/1808365628551844186)，详细介绍了该模型在端侧 (on-device) 的表现，并对**小型模型效率**的极限提出了思考。
- **GPU 讨论：双 4090 对决期待中的 5090**：**GPU 讨论升温**，辩论焦点在于现在购买两块 4090 GPU 还是等待传闻中的 5090 系列，并权衡了潜在的成本和性能。
   - 爱好者们就现有技术的优势与投机性的 50 系列特性展开交锋，在不断变化的 GPU 景观中引发了期待，并有建议主张保持耐心。
- **Arc 770 和 RX 580 面临挑战**：随着 **Arc 770 难以跟上步伐**，以及曾经多功能的 **RX 580** 因技术潮流转向放弃 **OpenCL** 支持而被抛弃，批评声随之而起。
   - 社区见解建议倾向于选择 3090 GPU 以保持持久的适用性，这呼应了关于性能标准和兼容性要求不可阻挡的进步的普遍观点。
- **开发者聊天探讨 Rust 咨询与提问礼仪**：Rust 爱好者在 #🛠-dev-chat 频道寻求**同行指导**，一位成员微妙的意见请求引发了关于有效解决问题方法的对话。
   - 对话演变为**提问框架策略**，强调了如 [Don't Ask To Ask](https://dontasktoask.com/) 和 [XY Problem](https://xyproblem.info/) 等资源，以解决技术咨询中常见的误区。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Blackstone 数十亿资金支持字节**：Blackstone 计划加倍投入 **AI 基础设施**，目前持有 **500 亿美元的 AI 数据中心**，并打算再投资 **500 亿美元**。[Blackstone 的投资](https://youtu.be/Z4EK9_s_ui8?si=v-xIlI78irXLWPhu) 使其成为 AI 物理骨干网的重要力量。
   - 市场对 Blackstone 的承诺感到兴奋，推测这是支持 AI 研究和商业开发的战略举措。
- **AI Agent：调研智能系统**：一份关于 **AI Agent 架构** 的深度调研引起了关注，记录了在推理和规划能力方面的进步。查看 [AI Agent 调研论文](https://arxiv.org/abs/2404.11584) 以获取近期进展的全景视图。
   - 该论文为未来 Agent 设计的辩论提供了跳板，有可能增强它们在**广泛应用领域**中的表现。
- **ColBERT 深入数据检索**：**ColBERT 的效率**引发热议，根据 [ColBERT 论文](https://arxiv.org/pdf/2004.12832)，其倒排索引检索速度超过了其他语义模型。
   - 该模型对数据集的熟练处理引发了对其广泛应用的讨论，从数字图书馆到实时信息检索系统。
- **ImageBind：模糊界限**：**ImageBind** 论文因其针对一系列模态（涵盖文本、图像和音频）的联合嵌入 (joint embeddings) 而引发讨论。在此查看 [ImageBind 模态详情](https://arxiv.org/abs/2305.05665)。
   - 其在跨模态任务上的出色表现暗示了多模态 AI 研究的新方向。
- **SBERT 句子脱颖而出**：SBERT 模型的应用，即使用 BERT 和池化层来创建独特的句子嵌入 (sentence embeddings)，突显了其**对比训练方法**。
   - 关键收获包括其在捕捉嵌入本质方面的娴熟能力，有望推动自然语言处理任务的进步。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Enterprise Pro 在 AWS 上线**：Perplexity 宣布与 **Amazon Web Services (AWS)** 建立合作伙伴关系，在 [AWS Marketplace](https://t.co/t3xBQlyw0c) 上推出了 **Perplexity Enterprise Pro**。
   - 该计划包括联合推广，并利用 **Amazon Bedrock** 的基础设施来增强生成式 AI 能力。
- **探索 Perplexity 的功能与特性**：在讨论 **Perplexity AI** 的工作流时，用户注意到消息会因长度而截断，但没有每日限制，这与允许消息延续的 **GPT** 不同。
   - 用户指出一个挑战：由于其独特的站点索引方式，Perplexity 无法提供预期的药物价格结果。
- **色情内容使用：无关左右**：一场激烈的辩论集中在保守派或自由派人口统计数据是否与不同程度的色情内容使用有关，但未得出明确结论。
   - 研究并未提供强有力的共识，但讨论表明文化影响可能会对消费模式产生潜在作用。
- **将 AI 集成到社区平台**：有人询问如何将 **Perplexity** 集成到 **Discord** 服务器中，但社区并未提供实质性的建议或解决方案。
   - 此外，用户对 **llama-3-sonar-large-32k-online** 模型自 6 月 26 日以来响应时间增加表示担忧。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **细节增强放大**：Stable Diffusion 在以极小的缩放因子**增强图像细节**方面的能力引起了轰动，用户对皮肤纹理和面部的改善感到惊叹。
   - *midare* 建议使用 2 倍缩放以获得最佳细节增强效果，这突显了用户的偏好。
- **在 Pony 模型上使用 Loras**：关于在 **Pony checkpoints** 上使用 **Character Loras** 的争论揭示了其与**普通 SDXL checkpoints** 相比的不一致性，存在角色识别丢失的问题。
   - *crystalwizard* 的见解指出，应聘请专门从事 Pony 训练的专家以获得更好的忠实度。
- **CivitAI 的战略性禁令**：**CivitAI** 继续禁止 **SD3** 内容，暗示其战略倾向于自家的 **Open Model Initiative**。
   - 有传言称 **CivitAI** 可能会嵌入类似于 **Stable Diffusion** 的商业限制。
- **Comfy-portable：坎坷的使用体验**：用户报告了 **Comfy-portable** 反复出现的错误，引发了关于社区是否支持故障排除工作的讨论。
   - 大量的故障排除帖子表明用户中普遍存在稳定性问题。
- **困扰的转换问题**：一位 RTX 2060 Super 用户在 **Automatic1111** 上遇到了问题，从屏幕黑屏到命令引起的卡顿。
   - *cs1o* 建议使用简单的启动参数，如 **--xformers --medvram --no-half-vae** 来缓解这些问题。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **编译器更迭揭示性能与构建特性**：Mojo 的隔夜更新带来了 `2024.7.1022` 等版本，引入了 `List` 的相等性比较以及 `UnsafePointer` 用法的增强，引发了热议。
   - 编码人员在处理 `ArrowIntVector` 时遇到了新的构建问题；清理构建缓存（build cache）成为了首选的急救措施。
- **AVX 奥德赛：从摩尔定律到 Mojo 的风采**：一位技术人员展示了 Mojo 编译器如何征服 AVX2，像熟练的交响乐指挥一样调度指令，而成员们则在思考通过手写内核（handwritten kernels）来突破性能极限。
   - 关于利用 AVX-512 性能的讨论也广为流传，尽管其中夹杂着那些手头没有该技术的成员的忧伤。
- **网络涅槃还是内核克星？**：**内核旁路网络 (Kernel bypass networking)** 成为 Mojo 对话的焦点，重点在于寻求无缝集成网络模块而不掉入常见陷阱的方法。
   - 资深人士回顾往昔，警告其他语言过去犯下的错误，主张 Mojo 应该铺就一条更稳健的道路。
- **Mojo 都市中的条件代码咒语**：Mojo 工艺台周围的“巫师”们思考着`条件一致性 (Conditional Conformance)`的奥秘，像 `ArrowIntVector` 这样的咒语搅动着复杂性的坩埚。
   - 贤者就参数化特性（parametric traits）给出了建议，作为穿越类型检查和指针复杂性迷雾森林的指南。
- **GPU 讨论拆分为专门频道**：**GPU 编程**谈话有了归宿，产生了一个新频道，专门用于从服务策略到引擎探索的 MAX 相关思考。
   - 此举旨在减少闲聊，切入 GPU 编程细微差别的正题，穿透噪音进行专注的技术讨论。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangSmith 计费困境：计算遗漏 Gemini**：LangSmith 无法将 Google 的 Gemini 模型纳入成本计算被指出是一个问题，因为它缺乏成本计算支持，尽管 token 计数已被正确添加。
   - 这一限制引起了依赖准确成本预测进行模型预算的用户的担忧。
- **聊天机器人：RAG 让语音机器人更智能**：分享了将“产品”和“订单详情”查询路由到语音机器人的 VDB 的实现细节，同时对其他问题使用 FAQ 数据。
   - 这种方法强调了定向查询意图与 RAG 架构相结合在高效信息检索方面的强大潜力。
- **自定义 API 调用：LangChain 的动态工具**：LangChain 在 JavaScript 中的 `DynamicStructuredTool` 允许为 API 调用创建自定义工具，如使用 `axios` 或 `fetch` 方法所示。
   - 用户现在可以通过自定义后端集成来扩展 LangChain 的功能。
- **Chroma 速度：加速 VectorStore 初始化**：加快 Chroma VectorStore 初始化的建议包括将向量存储持久化到磁盘、缩小 embedding 模型以及利用 GPU 加速，讨论参考了 [GitHub Issue #2326](https://github.com/langchain-ai/langchain/issues/2326)。
   - 这次讨论突显了社区在优化设置时间以提高性能方面的共同努力。
- **RuntimeError 骚乱：Asyncio 的事件循环难题**：一位成员遇到的 **RuntimeError** 引发了讨论，当时 `asyncio.run()` 从一个已经在运行的事件循环中被调用。
   - 社区尚未解决这个障碍，该话题仍处于开放状态以待未来见解。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Magnum 72B 媲美 Claude 3 的文采**：关于 [Alpindale's Magnum 72B](https://openrouter.ai/models/alpindale/magnum-72b) 的辩论激起，该模型源自 Qwen2 72B，旨在媲美 **Claude 3 模型** 的文笔质量。
   - 该模型在包含 5500 万个 RP 数据 token 的海量语料库上进行训练，为高质量的语言输出开辟了道路。
- **Hermes 2 Theta：更智能交互的合成模型**：[Nousresearch's Hermes-2 Theta](https://openrouter.ai/models/nousresearch/hermes-2-theta-llama-3-8b) 融合了 Llama 3 的实力与 Hermes 2 Pro 的磨砺，展示了其增强交互的**元认知能力**。
   - 这种融合不仅仅是模型合并；它是向通用函数调用和生成结构化 JSON 输出迈出的一大步。
- **老旧 AI 模型的谢幕**：即将到来的模型弃用将 [intel/neural-chat-7b](https://openrouter.ai/models/intel/neural-chat-7b) 和 [koboldai/psyfighter-13b-2](https://openrouter.ai/models/koboldai/psyfighter-13b-2) 列入清理名单，计划在 7 月 25 日后返回 404。
   - 这一战略性退役是由于使用量减少，旨在引导用户转向更新、更强大的替代方案。
- **路由增强抗故障能力：高效的回退机制**：OpenRouter 的韧性通过回退功能得到提升，在服务中断期间默认切换到备用提供商，除非使用 `allow_fallbacks: false` 覆盖。
   - 这种直观的机制充当了安全保障，即使在主要提供商出现故障时也能保证无缝连续性。
- **VoiceFlow 与 OpenRouter：上下文协作还是挑战？**：将 [VoiceFlow 与 OpenRouter](https://openrouter.ai) 集成引发了关于在无状态 API 请求中保持上下文的讨论，这是连贯对话的关键组件。
   - 出现了关于利用 [VoiceFlow 中的对话记忆 (conversation memory)](https://learn.voiceflow.com/hc/en-us/articles/15049513713037-Conversation-Memory) 来保留交互历史的提议，以确保聊天机器人能够保持对话思路。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **去中心化赋能 AI**：利用用户提供的计算资源构建 AI 计算的**去中心化网格网络 (decentralized mesh network)**，这一前景引发了热烈讨论。
   - [BOINC](https://boinc.berkeley.edu) 和 Gridcoin 被视为利用代币激励参与此类网络的典型模型。
- **分片与代币重塑计算**：关于潜在**分片计算平台 (sharded computing platform)** 的讨论将 VRAM 的通用性推向了前台，并提及通过代币产生用户奖励。
   - 引用 DHEP@home BOINC 项目的遗产，思考了通过去中心化网络进行 **CMOS 芯片优化**的可能性。
- **并行路径上的 GPU 探索**：针对以张量管理能力著称的 GGUF 平台，其 **GPU 并行执行 (parallel GPU executions)** 引起了广泛好奇。
   - 共识认为，鉴于 GGUF 的架构，这种方法具有可行性。
- **AI 通往 AGI 的阶梯**：OpenAI GPT-4 的**类人推理 (human-like reasoning)** 能力成为热门话题，公司概述了从“推理者 (Reasoners)”最终演进到“智能体 (Agents)”的未来。
   - [分级演进](https://www.bloomberg.com/news/articles/2024-07-11/openai-sets-levels-to-track-progress-toward-superintelligent-ai) 旨在完善解决问题的熟练程度，追求功能上的自主性。
- **库的新位置**：**提示词库 (prompt library)** 换了新标题，引导用户前往其在 <#1019652163640762428> 数字走廊中的新住所。
   - 提醒用户区分相似频道，并指向了[它们的具体位置](https://discord.com/channels/974519864045756446/1019652163640762428)。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **llama-agents 明星级发布**：新发布的 **llama-agents** 框架备受关注，在一周内其 [GitHub 仓库](https://twitter.com/llama_index/status/1811147950388916420) 就获得了超过 **1100 颗星**。
   - 爱好者可以通过 *MervinPraison* 提供的[视频演示](https://t.co/8uetfVqHf9)深入了解其功能和用法。
- **NebulaGraph 与 LlamaIndex 强强联手**：**NebulaGraph** 与 **LlamaIndex** 的突破性集成，为用户提供了用于动态属性图索引的 **GraphRAG** 能力。
   - 正如他们最近的[公告](https://twitter.com/llama_index/status/1811190191597773282)所强调的，这一结合为提取器带来了更高级的功能。
- **LlamaTrace 提升 LLM 可观测性**：**LlamaTrace** 与 **Arize AI** 建立了战略合作伙伴关系，以推进 LLM 应用评估工具和可观测性。
   - 此次合作旨在强化 LLM 工具集，详见其最新的[宣传资料](https://twitter.com/llama_index/status/1811462543535464796)。
- **Llamaparse 与现有 OCR 内容的交互**：社区正在热烈讨论 **Llamaparse** 如何处理 PDF 中现有的 OCR 数据，寻求关于增强与移除的明确说明。
   - 对话在没有定论的情况下结束，该话题仍有待进一步探索。
- **ReACT Agent 变量：前车之鉴**：用户报告在 **ReACT Agent** 中映射变量时遇到 **KeyError** 问题，引发了排错讨论。
   - 建议倾向于在执行前确认变量定义并确保其正确实现。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **架构实验热潮**：一位成员深入参与了**新型架构 (novel architectures)** 的测试，虽然目前尚未显示出实质性收益且消耗了大量计算资源，但预示着未来还有漫长的**消融研究 (ablation studies)** 之路。
   - 尽管缺乏大规模的改进，他们仍能从损失曲线的微调中获得乐趣，不过更深的模型往往会降低有效性，使得**持续实验**成为下一步的重点。
- **深入探讨符号梯度 (Sign Gradient)**：在模型中使用 **Sign Gradient** 的概念引起了社区的兴趣，为正在进行的**实验性架构**项目提出了新方向。
   - 对该想法的参与显示了社区愿意探索非常规方法，这可能会带来训练效率的提升。
- **残差连接故障排除**：讨论中提到了实验系统中**残差连接 (residual connections)** 的潜在陷阱，促使计划尝试替代的**门控机制 (gating mechanisms)**。
   - 这一转向反映了 AI 工程师在架构设计空间中所面临的复杂性和细微差别。
- **CIFAR-100：半程标志**：使用 250k 参数的模型在 CIFAR-100 上达到 50% 的准确率是一个值得关注的讨论点，正接近 [2022 年研究](https://arxiv.org/abs/2210.14151) 中报告的 70% **SOTA** 水平。
   - 获得的见解表明，Block 的数量对性能的影响不如总参数量关键，这为未来的**视觉模型 (vision model)** 调整提供了战略指导。
- **内存效率迷宫**：在 CIFAR-100 上使用 128 Batch Size 和 250k 参数模型进行训练时，高达 19 GB 的内存消耗凸显了实验设计中的内存效率问题。
   - 工程师们正在考虑创新的解决方案，例如多次使用单个大型 **MLP** 来解决这些效率限制。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **边缘分布的困惑**：由 **[FAST SAMPLING OF DIFFUSION MODELS WITH EXPONENTIAL INTEGRATOR](https://arxiv.org/abs/2204.13902)** 论文中关于 *边缘分布 (marginal distributions) p̂∗_t* 术语的混淆引发了对话，正在寻求社区的见解。
   - 讨论集中在**边缘分布**如何影响扩散模型的效能，尽管技术细节依然复杂且引人入胜。
- **本地智慧：引入 'RAGAgent' 实现现场 AI 智能**：成员们研究了 **[RAGAgent](https://github.com/MikeyBeez/RAGAgent)**，这是一个全新的 Python 项目，旨在打造一个可能引起轰动的全本地 AI 系统。
   - 这种**全本地 AI** 方法可能标志着我们思考和开发**个性化 AI 界面**方式的转变。
- **DoLa 助力：减少 LLM 幻觉**：**[Decoding by Contrasting Layers (DoLa)](https://arxiv.org/abs/2309.03883)** 论文概述了一种对抗 LLM 幻觉的新策略，使真实问答 (QA) 的表现提升了 **17%**。
   - 尽管可能会增加延迟，但 **DoLa** 在减少 LLM 输出错误方面的作用已成为模型可靠性讨论的焦点。
- **测试任务纠葛：真实测试需要训练改革**：对模型涌现行为的评估正受到审查，因为一篇论文警告说，**[在测试任务上训练 (training on the test task)](https://arxiv.org/abs/2407.07890)** 可能会扭曲对 AI 能力的认知。
   - 社区正在辩论训练协议的影响，因为当模型在评估前被统一微调时，**“涌现行为 (emergent behavior)”** 的炒作就会降温。
- **BitNet 的大胆尝试：1 比特精度向全精度对手施压**：焦点转向了 **[BitNet b1.58](https://arxiv.org/abs/2402.17764)**，这是一个精简的 1-bit LLM，在匹配全精度对手性能的同时，承诺实现**能源和资源的节省**。
   - [Hugging Face](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) 的复现确认了 BitNet 的实力，预示着一场关于节能 AI 模型未来的辩论。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Llama3 vs GPT-4o：分隔符困局**：用户报告了在比较 **GPT-4o** 和 **Llama3 local** 时的不同体验；前者在默认设置下表现稳定，而后者在分隔符和 Schema 相关的标准上存在波动。
   - 一位乐观的成员建议，**Llama3** 的问题可能会在即将到来的更新中得到解决。
- **LLM-Service 标志错误与文档修复**：当用户找不到对安装至关重要的 **LLM-Service flag** 时，引发了关于 **01 文档差异** 的讨论。
   - 一个正在进行中的 [documentation PR](https://link.to.pr) 被强调为补救措施，并建议使用 profiles 作为临时方案。
- **为 01 编写脚本以实现 VPS 卓越操作**：一个提议的脚本引发了对话，旨在使 **01 能够自动登录 VPS 控制台**，增强远程交互能力。
   - 一位成员渴望合作，分享了他们目前的探索，并邀请社区共同参与头脑风暴和协作开发。
- **01 的社区协作编程**：**01 强大的开发社区**受到了赞扬，该社区由 46 名贡献者组成，并向来自 Open Interpreter 的 100 多名交叉参与成员致意。
   - 社区互动被视为推动项目进展和演变的核心动力。
- **01 的商业抱负受阻？**：一位成员与 Ben Steinher 的对话深入探讨了 **01 在商业领域的潜力** 以及其适配所需的开发重点。
   - 讨论指出，实现 **远程登录** 是扩大 01 在专业环境中适用性的关键一步。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Axolotl 迁至新地址**：团队宣布，为了提高可访问性，**Axolotl** 数据集格式文档已移至[全新且改进的仓库](https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/)。
   - 此次迁移强调了“我们搬到了新组织”，以确保更顺畅的运营和用户体验。
- **TurBcat 在 48GB 系统上落地**：在用户 **c.gato** 表示计划使用 4-bit 量化进行测试后，目前推测 **TurBcat 72B** 可以在 48GB 的系统上运行。
   - 这一公告开启了围绕复杂 AI 模型的性能优化和资源分配的讨论。
- **TurBcat 通过 TabbyAPI 开启测试运行**：用户 **elinas** 通过分享用于 **[TurBcat 72B](https://lists-until-showing-allied.trycloudflare.com/)** 测试的 API 为社区做出了贡献，该 API 旨在完美适配各种注重效率的用户界面。
   - 共享的 API key `eb610e28d10c2c468e4f81af9dfc3a48` 旨在与 **ST Users / OpenAI-API 兼容前端**集成，利用 **ChatML** 实现无缝交互。
- **WizardLM 的 ArenaLearning 方法令人惊叹**：学习方法的创新仍在继续，**WizardLM** 团队发布了 [ArenaLearning](https://www.microsoft.com/en-us/research/uploads/prodnew/2024/07/WizardLM_ArenaLearning.pdf) 论文，提供了关于先进学习技术的见解。
   - 该发布引发了成员间的建设性对话，其中一人将其描述为“非常新颖”，暗示了 AI 训练范式潜在的转变。
- **FlashAttention-3 在 H100 GPU 上火力全开**：得益于 [FlashAttention-3](https://www.together.ai/blog/flashattention-3)，**H100 GPU** 正在迎来性能革新，该方案提议通过利用尖端硬件的能力来增强注意力机制。
   - 随着超越当前 35% 最大 FLOPs 利用率的愿景，社区推测通过减少内存操作和异步处理来加速效率的潜力。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **FlashAttention 驱动未来：Transformer 速度飙升**：[FlashAttention](https://pytorch.org/blog/flashattention-3) 彻底改变了 Transformer 在 GPU 上的效率，将 GPT-4 和 Llama 3 等前沿模型的 LLM 上下文长度推向了 128K 甚至 1M。
   - 尽管 FlashAttention-2 取得了进步，但在 H100 GPU 上仅达到了潜在 FLOPs 的 35%，这为优化飞跃留下了空间。
- **WizardArena 之战：聊天机器人对决的难题**：[WizardArena 平台](https://www.microsoft.com/en-us/research/project/wizardlm-arena-learning/)利用 Elo 评分系统对聊天机器人的对话熟练度进行排名，引发了竞争性评估。
   - 然而，以人为中心的评估过程在延迟和协调复杂性方面给用户带来了挑战。
- **OpenAI 收益盛宴：收入揭晓**：根据 [Future Research](https://futuresearch.ai/openai-revenue-report) 的数据，OpenAI 的收入正在膨胀，其中 **ChatGPT Plus 收入为 19 亿美元**，**ChatGPT Enterprise 收入为 7.14 亿美元**，此外还有其他利润丰厚的渠道，构成了多元化的收入流。
   - 分析指出 **ChatGPT Plus 订阅用户达 770 万**，这与 GPT-4 免费访问的困惑及其对订阅模式的影响形成了对比。
- **改写难题：合成指令受到审查**：Discord 中的好奇者思考了合成指令数据中**句法差异带来的收益**，并将其与回译（backtranslation）等类似策略进行了比较。
   - 对话中的参与者在思考词序是否能显著提升模型的理解能力和性能。
- **RPO 中 η 的细微差别：偏好思考参数**：频道讨论集中在 RPO 微调算法中神秘的 **η 参数**，争论其对奖励的影响性质和作用。
   - 该参数在过程中的作用引发了推测，强调了对优化机制进行深入理解的必要性。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **发现 Command R Plus 的乐趣**：**Mapler** 发现 **Command R Plus** 是构建趣味 AI Agent 的一个极具吸引力的选择。
   - 重点在于打造娱乐导向型 Agent 的创意方面。
- **模型微调难题**：**Mapler** 遭遇了挫折，正在努力应对一个未达到其基准测试要求的模型。
   - 一位社区成员强调，微调的质量至关重要，并将其总结为“垃圾进，垃圾出（garbage in, garbage out）”——强调了高质量数据集的重要性。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **PromptLayer 与 Anthropic SDK 的兼容性阻碍**：当尝试与最新版本的 **Anthropic SDK** 配合使用时，用于日志记录的 **PromptLayer** 集成失败。
   - 由于担心替代方案，该成员正积极寻求同类自托管解决方案的建议。
- **OpenPipe 的单模型综合征**：讨论显示，**OpenPipe** 仅支持 **OpenAI** 的 Prompt/回复日志记录，排除了 Anthropic 等其他模型。
   - 这一局限性引发了关于潜在变通方法或对更通用日志工具需求的讨论。
- **寻求 Fireworks.ai 的见解**：一位成员寻求有关 **fireworks.ai** 相关或以其为特色的讲座信息，但未出现进一步的细节或澄清。
   - 缺乏额外回应表明社区对该话题的了解程度或兴趣较低。
- **额度核算：成员查询**：有人提出了如何验证额度可用性的问题，该成员提供了账户 ID **reneesyliu-571636** 以寻求帮助。
   - 这仍然是一个孤立的查询，表明问题已解决或正在进行关于 **Account ID Query** 的私下讨论。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **NVDLA 通用性 vs NV 加速器**：有人询问 NV 加速器是否是 **NVDLA** 的全方位解决方案，引发了对 [GitHub 上的 NVDLA 项目](https://github.com/nvdla/) 的调研。
   - 提到将 **CuDLA 调查** 作为潜在的下一步行动，但在深入研究之前需要确认 NV 的能力。
- **以内核为中心的 NV Runtime 见解**：对 **NV runtime** 的探索揭示了它与 GPU 紧密协作，绕过用户空间（userspace）并直接与内核交互以执行进程。
   - 这一信息阐明了 NV 基础设施如何与底层硬件交互，绕过了传统的用户空间限制。
- **揭秘神经网络图 UOps**：在分析一个简单神经网络图中的 UOps 时，发现了一些涉及常量的意外乘法和加法。
   - 当注意到这些操作是 **线性权重初始化（linear weight initialization）** 的结果时，这个谜团得到了解决，解释了数值上的异常。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **参议院对 AI 和隐私的审查**：一场 [参议院听证会](https://www.commerce.senate.gov/2024/7/the-need-to-protect-americans-privacy-and-the-ai-accelerant) 聚焦于美国参议员 **Maria Cantwell**，她强调了 AI 在数据隐私中的重要性，并倡导制定联邦隐私法。
   - 来自 **Mozilla** 的证人 **Udbhav Tiwari** 强调了 AI 在在线监控和画像方面的潜力，敦促建立法律框架以保护消费者隐私。
- **Mozilla 倡导 AI 隐私法**：**Mozilla** 在一篇 [博客文章](https://blog.mozilla.org/en/mozilla/internet-policy/mozilla-urges-federal-privacy-law-for-ai-development/) 中阐述了他们的立场，**Udbhav Tiwari** 在参议院听证会上再次强调了联邦监管的必要性。
   - 该文章强调了立法行动的紧迫性，并分享了 **Tiwari** 在就 AI 时代的隐私保护作证时的影像。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Hugging Face 协调业务与模型**：一场名为 **揭秘 Hugging Face 模型及其如何发挥业务影响力（Demystifying Hugging Face Models & How to Leverage Them For Business Impact）** 的独家研讨会定于 **2024 年 7 月 30 日** 东部时间中午 12 点举行。
   - 无法参加？请在 [此处](https://events.rotational.io/demystifying-llms) 注册，以便在活动结束后获取研讨会资料。
- **Recsys 社区兴起，Search/IR 萎缩**：**Recsys 社区** 在规模和活跃度上超过了 **search/IR** 社区，前者正在增长，而后者被描述为更加小众。
   - **Cohere** 最近收购了 **sentence transformer 团队**，来自 Vespa 的 *Jo Bergum* 和来自 **Elastic** 的成员也参与了讨论。
- **Omar Khattab 带来动态 DSPy 对话**：在 DSPy，MIT/斯坦福学者 **Omar Khattab** 分享了他在复杂主题上的专业见解。
   - Khattab 的讨论点引起了观众的共鸣，强调了该领域的技术深度。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Torchtune Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1261053927076659292)** (1 条消息): 

> - `qdurllm demo`
> - `利用知识图谱进行 RAG`
> - `用于 HF 模型的 Intel CPU`
> - `自我审查编程助手`
> - `用于个人数据的 LlamaIndex` 


- **qdurllm 展示高效 AI**：由社区成员提供的 [qdurllm demo](https://huggingface.co/spaces/as-cle-bert/qdurllm-demo) 展示了 AI 工具效率的提升。
- **利用知识图谱进行高级 RAG 的研讨会**：一段 [YouTube 视频](https://www.youtube.com/watch?v=9wqVz0LDYgg&ab_channel=DecodingDataScience) 介绍了如何利用知识图谱进行高级 RAG，重点介绍了 Langchain 和 Neo4j。
- **Intel CPU 助力 HuggingFace 模型**：一个新的 [GitHub 仓库](https://github.com/sleepingcat4/intel-hf) 展示了如何高效地在 Intel CPU 上运行 HuggingFace 模型。
- **Gary4live Ableton 插件现已免费**：[gary4live Ableton 插件](https://x.com/thepatch_kev/status/1810063563823907172) 已在 Gumroad 上免费提供，鼓励音乐制作人充分利用。
- **MInference 1.0 提升推理速度**：MInference 1.0 引入了 [10倍推理加速](https://huggingface.co/blog/liyucheng/minference10)，支持在单张 GPU 上运行百万级上下文模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/as-cle-bert/qdurllm-demo">Qdurllm Demo - a Hugging Face Space by as-cle-bert</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=9wqVz0LDYgg&ab_channel=DecodingDataScience)">The Future of AI: Leveraging Knowledge Graphs for Advanced RAG</a>: 准备好深入了解使用 Langchain 和 Neo4j 进行自然语言查询的世界！学习如何使用 Cypher 查询语言与图数据库进行交互...</li><li><a href="https://wandb.ai/sauravmaheshkar/llamaindex-local-models-index/reports/Training-a-chatbot-on-personal-data-with-LlamaIndex-and-W-B--Vmlldzo4MzQzMDE3)">Weights & Biases</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://x.com/thepatch_kev/status/1810063563823907172)">来自 thecollabagepatch (@thepatch_kev) 的推文</a>: 13 位传奇人物刚刚收到了关于 gary4live 的邮件，这是一个实现此功能的 Ableton 插件，现在可以在 Gumroad 下载，链接见下方 @_buildspace @_nightsweekends</li><li><a href="https://youtu.be/38ae7hqzX5s)">Gemma2:27 Ollama 修正！现在表现惊人！</a>: 今天，我们将再次使用 Ollama 测试 Gemma 2 27B，因为 Ollama 推送了一个更新来修正与 Gemma 2 相关的问题，现在它可以正常工作了...</li><li><a href="https://youtu.be/gAtUdnN1_xM?si=L_1vdbjzu4yHyUlA)">Rauf 带来的 SK-LEARN 入门介绍</a>: scikit-learn (sklearn) 机器学习库的简短基础介绍。我最初是为我的演讲创建的，但我意识到这会很有帮助...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1260672356645474435)** (367 条消息🔥🔥): 

> - `GPU 过时`
> - `管理大型 LLM`
> - `量化技术`
> - `求职申请 AI`
> - `云计算成本` 


- **安息吧 GPU：灰尘编年史**：一位用户哀叹他们的 GPU 因灰尘堆积而报废，引发了关于潜在替代品以及升级硬件所涉及的财务限制的讨论。
   - 另一位用户强调了使用旧 GPU 的实际挫折感，以及这对他们项目（如渲染和模型训练）产生的重大影响。
- **在有限硬件上运行大型 LLM**：用户交流了在性能不足的设备上运行 8B+ 参数大型模型的技巧，提到了利用 Kaggle 或 Colab 免费使用强大 GPU 的选项。
   - 讨论还涉及了各种量化方法，以减少内存开销并优化性能。
- **十亿张 A100 GPU 并非免费**：成员们分享了云计算的财务负担，讲述了在训练大型模型时昂贵的失误，并强调了在云端部署前进行本地测试的重要性。
   - 一位用户幽默地建议利用社交媒体炒作和风险投资资金来支持压倒性的计算成本。
- **利用 LLM 彻底改变求职申请**：一场关于构建 AI 解决方案以自动化求职申请的深入对话展开，涉及使用 LangChain 进行网页抓取，并应用 LLM 来解析和填写表格。
   - 参与者对合作表现出兴趣，旨在设计不仅能填写表格，还能识别合适职位匹配的自主系统。
- **利用 AI 进行有效的 PDF 分析**：一位用户询问了适合理解包含多栏文本和图像的复杂 PDF 文档的模型，建议指向了 LayoutLM 和 BERT 等模型。
   - 重点在于能够准确解析结构化文档并根据其内容做出明智决策的工具。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/nroggendorff/train-with-llama-architecture">从头开始训练 Llama 模型</a>: 未找到描述</li><li><a href="https://drive.google.com/file/d/1uRH74mDKGcQRmHeHc_XdRXUbyBYe358l/view?usp=drivesdk">使用 Jupyter Lab 调试.mp4</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/nroggendorff/mayo">Mayo - nroggendorff 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/">教程 | 🦜️🔗 LangChain</a>: 刚接触 LangChain 或 LLM 应用开发？阅读此材料以快速上手。</li><li><a href="https://youtu.be/ylrew7qb8sQ?si=lQ3t_VhSnhgCeAo-">WebVoyager</a>: WebVoyager：使用大型多模态模型构建端到端 Web Agent。WebVoyager 是一款新型的视觉驱动网页浏览 Agent，它使用浏览器截图...</li><li><a href="https://huggingface.co/datasets/Exqrch/IndoToxic2024">Exqrch/IndoToxic2024 · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://tenor.com/view/stewie-family-guy-rip-sad-funeral-gif-13648662">Stewie Family Guy GIF - Stewie Family Guy Rip - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/mmm-what-shocked-monster-inc-james-p-sullivan-gif-14562553">Mmm What GIF - Mmm What Shocked - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces/nroggendorff/mayo/discussions/2">nroggendorff/mayo · GPU 加速</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=4Wa5DivljOM">为什么你沉迷于云计算</a>: 从商业角度了解 AWS、Microsoft Azure 和 Google Cloud 等大型云服务提供商的运作方式。探索优化云计算的策略...</li><li><a href="https://github.com/dykyivladk1/polip">GitHub - dykyivladk1/polip: 旨在提升神经网络训练体验的库</a>: 旨在提升神经网络训练体验的库 - dykyivladk1/polip</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/tools/google_jobs/">Google Jobs | 🦜️🔗 LangChain</a>: 本笔记本介绍了如何使用 Google Jobs 工具获取当前的职位发布。</li><li><a href="https://github.com/lllyasviel/Fooocus.git">GitHub - lllyasviel/Fooocus: 专注于提示词和生成</a>: 专注于提示词和生成。通过在 GitHub 上创建账户为 lllyasviel/Fooocus 的开发做出贡献。</li><li><a href="https://www.ornl.gov/news/going-big-worlds-fastest-computer-takes-large-language-modeling">走向宏大：世界上最快的计算机挑战大语言建模 | ORNL</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2312.12705">在 Frontier 上优化大语言模型的分布式训练</a>: 大语言模型 (LLMs) 作为基础模型取得了显著成功，通过微调使各种下游应用受益。最近关于损失缩放的研究表明...</li><li><a href="https://ieeexplore.ieee.org/abstract/document/10528939">在 Frontier 上优化大语言模型的分布式训练</a>: 大语言模型 (LLMs) 作为基础模型取得了显著成功，通过微调使各种下游应用受益。损失缩放研究已经证明了其优越性...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1260969404309442621)** (2 条消息): 

> - `Embedding 模型中的三元组崩溃 (Triplet collapse)`
> - `使用 Softmax 预训练基础模型以进行迁移学习` 


- **Embedding 模型中的三元组崩溃详解**：一位成员询问了关于三元组崩溃的背景，并得到了关于使用三元组损失 (triplet loss) 训练 Embedding 模型的解释，该模型根据鼠标移动来识别个人。
- **使用预训练 Softmax 模型的迁移学习**：为了缓解三元组崩溃，该成员解释了如何预训练一个具有 N 个 Softmax 输出的常规分类模型，并将其迁移到 Embedding 模型中。
   - 该方法通过从预训练网络开始，解决了模型产生零嵌入 (zero-embeddings) 的问题，从而避免了局部最小值损失的情况。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1260674413150605443)** (6 条消息): 

> - `评估数据集对抗 (Eval Dataset Fights)`
> - `模型准确率检查`
> - `特征重要性 (Feature Importances)`
> - `LeRobot 入驻 Twitter` 


- **评估中的模型准确率一致性**：一位成员询问了评估数据集中对抗（fights）的数量，质疑是否所有对抗都用于训练，然后在新的对抗上进行评估。
   - *zewanyoekill* 回复称，测试集最初占数据集的 20%，达到了 **78% 的准确率**，即使调整为 5%，准确率仍稳定在 **0.78** 左右。
- **随时间评估模型准确率**：模型每周都会针对新事件进行检查，以验证其 **78% 准确率** 的稳定性。
- **特征重要性分析**：有人建议检查模型的 **feature importances**，以确定哪些特征的影响最显著。
- **LeRobot 加入 Twitter**：社区获悉 **LeRobot** 现在已入驻 [Twitter/X](https://x.com/LeRobotHF)。



**提及的链接**：<a href="https://x.com/LeRobotHF">来自 undefined 的推文</a>：未找到描述

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1260684966342430790)** (8 条消息🔥): 

> - `基于 LLM 的自主 Agent`
> - `Ideogram 输出集合`
> - `Next.JS 网站重构`
> - `近期 ML 研究博客`
> - `用于 Python 代码质量的 DPO 数据集` 


- **基于 LLM 的自主 Agent 立场论文**：[Manifold Research Group](https://www.manifoldrg.com/llm-agents/) 分享了他们的立场论文，题为《大语言模型时代的智能数字 Agent (Intelligent Digital Agents in the Era of Large Language Models)》，重点关注基于 LLM 的自主 Agent 的进展和未来机遇。
   - 他们正在扩大研究团队，并邀请感兴趣的人士加入 [Discord](https://discord.gg/MfYZmYEGaa) 参与讨论。
- **Ideogram 输出集合**：一位用户分享了一小部分 [Ideogram 输出](https://huggingface.co/datasets/terminusresearch/ideogram-25k)，包括由 Florence2 生成的标注，并计划添加来自 Llava-next 和 CogVLM2 的更多内容。
- **Next.JS 网站重构**：一位用户宣布使用部署在 [Vercel 上的 Next.JS](https://likiastudios-site.vercel.app) 重构了他们的网站，并提到了目前的局限性，如缺少浅色模式配置。
   - 开发日志以 Prefixed Markdown (PMD) 格式存储，以便更轻松地插入代码。
- **近期 ML 研究博客**：一篇题为《[AI Unplugged #14](https://datta0.substack.com/p/ai-unplugged-14-adam-mini-grokfast)》的博客文章讨论了 **Adam Mini** 和 **GrokFast** 等优化器，重点关注模型训练中的效率和性能。
   - 主题是优化，即以少胜多，还涵盖了用于端侧应用的 **MobileLLM** 和用于精选数据的 **JEST**。
- **用于 Python 代码质量的 DPO 数据集**：一位用户介绍了 [mypo 数据集](https://huggingface.co/datasets/joshuasundance/mypo-4k-rfc)，专注于 Python 代码质量，并分享了示例指令和输出以征求社区反馈。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.manifoldrg.com/llm-agents/">Intelligent Digital Agents in the Era of Large Language Models</a>：该立场论文概述了基于 LLM 的 AI Agent 的当前研究领域和突破。我们强调了关键进展并讨论了每个领域的局限性。</li><li><a href="https://www.manifoldrg.com/opportunities/">Opportunities</a>：有几种方式可以参与我们的工作：1. 加入我们的 Discord 并参与活动和讨论（无论是否与项目相关）。2. 异步贡献 GitHub 上的 Issue。...</li><li><a href="https://likiastudios-site.vercel.app">LikiaStudios</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/terminusresearch/ideogram-25k">terminusresearch/ideogram-25k · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://datta0.substack.com/p/ai-unplugged-14-adam-mini-grokfast">AI Unplugged 14: Adam mini, GrokFast, MobileLLM, JEST</a>：洞察胜过信息</li><li><a href="https://huggingface.co/datasets/joshuasundance/mypo-4k-rfc">joshuasundance/mypo-4k-rfc · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/OpenCo7/UpVoteWeb">OpenCo7/UpVoteWeb · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1260724180140167259)** (17 条消息🔥): 

> - `Paper Presentation Scheduling` (论文展示排期)
> - `Understanding LLM Understanding Summer School` (理解 LLM 理解暑期学校)
> - `ResNets vs Highway Networks` (ResNets 与 Highway Networks 对比)


- **在 Discord 中安排论文展示**：成员们讨论了在 **7/28** 安排论文展示，并可能在 **8/03** 进行一次展示。
- **理解 LLM 理解暑期学校资源**：分享了 **Understanding LLM Understanding** 暑期学校的材料链接，包括 [演讲和小组讨论的视频](https://www.youtube.com/watch?v=HLi6wOa1-Q4&list=PL2xTeGtUb-8B94jdWGT-chu4ucI7oEe_x)。
- **ResNets 的观点与辩论**：一位成员分享了一篇声称 **ResNets** 是 **highway networks** 特例的论文，并引发了辩论。



**提到的链接**：<a href="https://skywritingspress.ca/2019/01/08/the-journey-begins/">Understanding LLM Understanding</a>：献给 DANIEL C. DENNETT (1942 – 2024) 的纪念：2024 暑期学校：6 月 3 日 – 6 月 14 日。包含全部 33 场演讲和 7 场小组讨论的视频、演讲者、摘要、时间表等...

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1260970964078825624)** (3 条消息): 

> - `llama-3 8b model performance` (llama-3 8b 模型性能)
> - `tensorFlow model for detecting homophobic messages` (用于检测恐同消息的 TensorFlow 模型)
> - `RAG for limited data classification` (用于有限数据分类的 RAG)
> - `fine-tuning LLMs for harmful message detection` (微调 LLM 以检测有害消息)


- **动态量化的 Llama-3 8b 表现优于非量化版本**：一位成员发现 **8-bit 量化版 llama-3** 在分类任务中获得了比非量化版本更高的 F1 分数，他们觉得这很奇怪。
- **创建用于检测恐同消息的多语言模型**：一位成员询问创建用于检测多种语言恐同消息的 TensorFlow 模型的最佳方法。
   - 另一位成员建议在数据有限时使用 [RAG](https://link.to/RAG)，或者在数据较多时微调现有的 LLM（如 **Roberta**）。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1260747912728674304)** (2 条消息): 

> - `rm -rf command in Unix-based systems` (Unix 系统中的 rm -rf 命令)


- **探讨 'rm -rf /' 的风险**：一位用户提到了 'rm -rf /' 命令，这是 Unix 系统中一个强大且具有潜在危险的命令。
   - 该命令从根目录开始递归删除文件和目录，如果以 root 权限执行，可能会导致严重的系统损坏。
- **在命令中使用表情符号**：一位用户在讨论 'rm -rf' 命令的语境中使用了表情符号 <:true:1098629226564956260>。
   - 这说明了尽管在处理严肃的命令，聊天氛围依然活跃且轻松。


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1260674391029710899)** (310 条消息🔥🔥): 

> - `Ghost 8B Beta experience` (Ghost 8B Beta 体验)
> - `Qwen2 1.5b Model Discussion` (Qwen2 1.5b 模型讨论)
> - `Hardware for Fine-tuning` (用于微调的硬件)
> - `Finetuning Tips and Strategies` (微调技巧与策略)
> - `Phi-3 Models Fine-tuning Concerns` (Phi-3 模型微调的担忧)


- **推荐 Ghost 8B Beta 体验**：一位成员推荐尝试 [Ghost 8B Beta](https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k)，称其用法应与 ChatGPT 或 Claude 类似。
   - 他们以兴奋的语气分享了链接，并鼓励大家对其性能进行评价和评论。
- **Qwen2 1.5b 模型引发辩论**：成员们讨论了 **Qwen2 1.5b** 的性能，特别是关于其微调的灵活性以及在无需 GPU 的情况下的功能。
   - 有人对其有效性提出了疑问，一位用户指出它能很好地模仿结构，适合小模型，而另一位用户则指出了资源需求。
- **预算有限下的微调硬件选择**：一位新成员寻求关于高性价比 GPU 的建议，在 4060TI 和 3090 之间纠结，用于微调 LLAMA2-7b，因为云端使用受限。
   - 建议倾向于购买二手 **3090**，以在约 800 USD 的预算内获得更好的 VRAM 和性能，并强调了 VRAM 速度的重要性。
- **微调技巧：Epochs、数据及其他**：成员们分享了关于有效微调实践的见解，例如减少 epochs 以避免过拟合，以及使用合适的数据整理器（data collator）。
   - 讨论强调了较小 epochs 的重要性，以及为了获得最佳训练结果而理解 **DataCollatorForCompletionOnlyLM** 的必要性。
- **Phi-3 模型微调争议**：关于微调 **Phi-3-mini-4k-instruct** 模型引发了激烈辩论，重点在于预训练数据质量可能受损。
   - 专家不建议在 instruct 模型上进行微调，因为可能会产生不利影响，但也有人建议将其作为初学者的学习工具，因为迭代速度更快。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k">Ghost 8B Beta (β, 8k) - lamhieu 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.together.ai/blog/flashattention-3">FlashAttention-3: 具有异步和低精度的快速且准确的 Attention</a>：未找到描述</li><li><a href="https://huggingface.co/AI-Sweden-Models">AI-Sweden-Models (AI Sweden 模型枢纽)</a>：未找到描述</li><li><a href="https://datta0.substack.com/p/ai-unplugged-14-adam-mini-grokfast">AI Unplugged 14: Adam mini, GrokFast, MobileLLM, JEST</a>：洞察重于信息</li><li><a href="https://tenor.com/view/226-wrong-math-bad-math-doesnt-add-up-elaborate-gif-25510055">226 错误的数学 GIF - 226 Wrong Math Bad Math - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/UnslothAI/status/1811447913962438994">来自 Unsloth AI (@UnslothAI) 的推文</a>：我们制作了一个分步教程，介绍如何使用 Google Colab 微调 Llama-3 并将其部署到 @Ollama。教程：https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama Colab 不...</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">从最后一个 Checkpoint 进行微调 | Unsloth 文档</a>：Checkpointing 允许您保存微调进度，以便您可以暂停并继续。</li><li><a href="https://github.com/Azure/azureml-examples/blob/phi/bug_bash/sdk/python/foundation-models/system/finetune/text-generation/chat-completion.ipynb">azureml-examples/sdk/python/foundation-models/system/finetune/text-generation/chat-completion.ipynb (位于 phi/bug_bash 分支) · Azure/azureml-examples</a>：官方社区驱动的 Azure Machine Learning 示例，已通过 GitHub Actions 测试。- Azure/azureml-examples</li><li><a href="https://colab.research.google.com/drive/1VAaxMQJN9-78WLsPU0GWg5tEkasXoTP9?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing#scrollTo=95_Nn-89DhsL">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ce">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1W0j3rP8WpgxRdUgkb5l6E00EEVyjEZGk?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3">Phi-3 - microsoft 集合</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cppm6n/phi3_mini_finetunes/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer">Supervised Fine-tuning Trainer</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1-BF5HndNqQsfWRTxIt7YPjkfDpVUGNgY?usp=sharing#scrollTo=Ymx-p3FvF-P2">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>：未找到描述</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1260862120321355857)** (10 messages🔥): 

> - `Sam Altmann Investment`
> - `Open Diloco`
> - `Distributed Training` 


- **Sam Altmann 向国防军事公司投资 1 亿美元**：[Sam Altmann](https://www.reddit.com/r/OpenAI/comments/1e0fsvu/sam_altman_led_100m_series_b_investment_into_a/) 领投了对一家专注于制造无人高超音速飞机的国防军事公司的 1 亿美元 B 轮投资。
   - NSA 局长加入董事会引发了关于为未来事件采取潜在准备措施的讨论。
- **Open Diloco 旨在实现 AI 训练的去中心化**：由 @samsja19 引入的 [Open Diloco](https://x.com/samsja19/status/1811450791900901853) 仅需 100mb/s 带宽即可实现全球分布式 AI 模型训练，计算利用率达到 90%-95%。
   - 该项目依赖于使用 torch FSDP 和 hivemind 的混合代码，目标是从在巨型集群上训练的闭源模型转向在多个小型数据中心协同训练的开源模型。
- **分布式 GPU 工作负载的挑战与成功**：社区成员讨论了在多个 GPU 之间使用 FSDP 调度计算以及实现能够处理大规模数据处理的分布式 GPU 工作负载的挑战。
   - 一位成员分享了一个成功案例：通过在 100 个节点上使用分布式 GPU 工作负载，以暴力计算成本的一小部分，在 4 小时内完成了 100 万个 JSON 数据集的过滤。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/samsja19/status/1811450791900901853">来自 samsja (@samsja19) 的推文</a>: 非常高兴展示我们在 Open Diloco 上的工作。我们在 3 个国家、带宽低于 100mb/s（比 infiniband 慢 10,000 倍）的环境下训练了一个 1b 模型，计算利用率达到 90%-95%...</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1e0fsvu/sam_altman_led_100m_series_b_investment_into_a/">Reddit - 深入了解</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1260780655101607988)** (7 messages): 

> - `Continued Pretraining without using Lora`
> - `Unsloth and multiple GPUs`
> - `Decoder Architecture for Embedding Model`
> - `Xformers compatibility issue with Unsloth` 


- **不使用 Lora 进行持续预训练 (Continued Pretraining)**：一位成员询问是否可以在不使用 **Lora** 的情况下通过 Unsloth 进行持续预训练。
- **Unsloth 与多 GPU 的问题**：一位成员询问如何设置 Unsloth 仅使用一个 GPU 进行训练，因为它不支持多 GPU。
   - 随后，他们确认自己解决了该问题。
- **关于 Embedding Model 的 Decoder 架构的困惑**：一位成员请求澄清 **decoder architecture** 如何用于 **Embedding Model** 以及 “Latent Array” 的概念。
   - 另一位成员建议将讨论移至合适的频道。
- **Xformers 与 Unsloth 的兼容性问题**：一位成员报告了一个 ImportError，指出其 **xformers 版本 0.0.27** 对 Unsloth 来说太新了。
   - 建议他们更新 Unsloth 或降级 xformers 版本。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1261001608746242098)** (2 messages): 

> - `Ghost 8B Beta`
> - `Context length capabilities` 


- **探索 Ghost 8B Beta 中的上下文长度 (Context Length)**：一位成员询问了上下文长度为 **128k** 的 **Ghost 8B Beta** 的能力以及可以用它实现什么。
   - 该模型的官方页面可在 [HuggingFace](https://huggingface.co/spaces/lamhieu/ghost-8b-beta-128k) 上找到。
- **Ghost 8B Beta 的刷新状态**：一位成员注意到 **Ghost 8B Beta** 的模型页面在不断刷新。
   - *提供的链接正在刷新，可能需要检查可用性。*



**提及的链接**: <a href="https://huggingface.co/spaces/lamhieu/ghost-8b-beta-128k">Ghost 8B Beta (β, 128k) - lamhieu 的 Hugging Face Space</a>: 未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1260758792606453760)** (10 messages🔥): 

> - `New message types` (新的消息类型)
> - `Modular Model Spec` (模块化模型规范)
> - `Training directly on new tokens` (直接在新的 token 上进行训练)
> - `Partially Trainable Config in PyTorch` (PyTorch 中的部分可训练配置)
> - `Finetuning Gemma-2-27b for coding` (针对编程任务微调 Gemma-2-27b)


- **Modular Model Spec 发布**: **Modular Model Spec** [0.0.0 版本](https://modular-model-spec.vercel.app)旨在通过概述**统一的模块化数据集格式**，提高 LLM 的可靠性、开发者便利性和灵活性。
   - _为什么该规范很重要_：为 **LLM-augmented applications** 提供更高的可靠性、可编程设置和更强的灵活性。
- **通过示例训练新 Token**: 鉴于新 token 不会出现在预训练数据中，一位成员建议通过大量的后训练 (post-training) 示例直接对其进行训练。
   - 他们认为：“这些 token 不会出现在任何预训练数据中，因此直接在它们上面进行训练是唯一的选择。”
- **在 PyTorch 中实现部分可训练配置**: PyTorch 中自定义的 **PartiallyTrainableConfig** 类以及相应的 embedding 和 LMHead 类允许特定的 token 可训练，同时冻结其他 token。
   - 这种方法修改了模型，使其仅针对目标 token 训练 embedding 和 logits，但在特定权重矩阵范围的 **requires_grad** 设置上遇到了问题。
- **Gemma-2-27b 在微调方面表现出色**: Gemma-2-27b 模型在编程任务中脱颖而出，据报道仅需 two shots 就能用 Python 编写俄罗斯方块 (Tetris)。
   - 它在这方面与 Codestral 和 Deepseek-v2 并驾齐驱，表现优于 **llama-3-70b** 和 **qwen2-72b** 等其他开源模型。



**提及的链接**: <a href="https://modular-model-spec.vercel.app">Modular Model Spec</a>: 未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1260700921852198994)** (4 messages): 

> - `Model Compression in LLMs` (LLM 中的模型压缩)
> - `Norm Tweaking for Quantization` (用于量化的 Norm Tweaking)
> - `FlashAttention-3 Performance Boost` (FlashAttention-3 性能提升)
> - `Pingpong Scheduler Implementation` (Pingpong 调度器实现)


- **Norm Tweaking 提升 LLM 量化效果**: [这篇论文](https://arxiv.org/abs/2309.02784)介绍了一种名为 norm tweaking 的技术，用于提高 LLM 量化的精度，即使在 2-bit 量化下也能实现高准确度。
   - 该方法在 GLM-130B 和 OPT-66B 等模型上显示出显著改进，使其在实际应用中更具可行性，特别是与现有的 PTQ 方法相比。
- **FlashAttention-3 加速 Transformer 注意力机制**: [FlashAttention-3](https://x.com/tri_dao/status/1811453622070444071) 加速了 Transformer 的性能，在 FP16 上实现了 1.5-2 倍的加速，在 H100 等现代 GPU 上使用 FP8 时可达到 1.2 PFLOPS。
   - 然而，目前的改进仅限于 H100 GPU，这引发了人们对新的 pingpong 调度器在其他 GPU 上适用性的好奇。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2309.02784">Norm Tweaking: High-performance Low-bit Quantization of Large Language Models</a>: 随着大语言模型 (LLMs) 规模的不断增长，在不牺牲准确性的情况下进行模型压缩已成为部署的关键挑战。虽然一些量化方法（如 GP...）</li><li><a href="https://x.com/tri_dao/status/1811453622070444071">来自 Tri Dao (@tri_dao) 的推文</a>: FlashAttention 被广泛用于加速 Transformer，已经使注意力机制快了 4-8 倍，但尚未充分利用现代 GPU 的优势。我们正在发布 FlashAttention-3：在 FP16 上快 1.5-2 倍，u...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1260674061021876244)** (18 messages🔥): 

> - `Hackathon Team Formation`
> - `FlashAttention discussion`
> - `Shared Memory Usage` 


- **Hackathon Team Formation**: 成员们讨论了为即将到来的 [以 CUDA 为中心的黑客松](https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf) 组队，演讲嘉宾包括 Chris Lattner 和 Raja Koduri。
   - *as_ai* 提到了昂贵的机票，而 *ericauld* 提到了住宿需求，但表示如果两人都参加，有兴趣组队。
- **FlashAttention in Modern GPUs**: 分享了一篇描述 [FlashAttention](https://www.together.ai/blog/flashattention-3) 改进的博客文章，该技术加速了 GPU 上的 Attention 计算，并被用于各种 AI 模型中。
   - *iron_bound* 针对技术细节幽默地评论道 'H100 go brrrrr'。
- **Shared Memory Usage Limitations**: 成员们讨论了 CUDA block 的 Shared Memory 限制，特别是如何在一个 block 内高效地使用更多 Shared Memory。
   - *thakkarv_86311* 澄清说，剩余的 51kib 内存并不一定会被闲置。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.together.ai/blog/flashattention-3">FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision</a>: no description found</li><li><a href="https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf">RSVP to Hardcore CUDA Hackathon | Partiful</a>: *All talks and projects MUST be written in CUDA* Every hardcore hacker gets a H100 for the day. All sponsored and proved by Nebius.ai! Let&#x27;s blow away some baselines.  Speakers: - Chris Lattner (...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1260760680747892736)** (1 messages): 

> - `User-defined Triton kernels`
> - `torch.compile for optimization`
> - `Triton kernel tutorial` 


- **Optimize with User-defined Triton Kernels**: 一位用户分享了关于使用用户定义的 **Triton kernels** 配合 `torch.compile` 来优化模型计算的 [教程](https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)。
   - 该教程包含了 **vector addition kernels** 的示例代码，并强调了将这些优化后的计算集成到 **PyTorch** 模型中时潜在的性能提升。
- **Basic Usage of Triton Kernels with torch.compile**: 该教程通过将 [Triton 文档](https://triton-lang.org/main/getting-started/tutorials.html) 中的一个简单向量加法 kernel 与 `torch.compile` 集成，演示了基本用法。
   - 提供了示例代码和步骤，帮助用户通过将 **Triton kernels** 集成到他们的 PyTorch 模型中来实现 **硬件峰值性能**。



**Link mentioned**: <a href="https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html">Using User-Defined Triton Kernels with torch.compile &mdash; PyTorch Tutorials 2.3.0+cu121 documentation</a>: no description found

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1260786817385627648)** (17 messages🔥): 

> - `bf16/fp16 model checkpoint issues`
> - `Lottery ticket hypothesis with bfloat16`
> - `flex_attention function`
> - `Optimization in test-time-training repo` 


- **bf16/fp16 model checkpoint handling**: 一位用户询问以 bf16/fp16 格式训练的模型 checkpoint 是否默认保存为 fp32，以及正确的处理方式。
   - 另一位用户建议 state dicts 将保持为 bf16/fp16，但加载它们时需要显式转换；针对推理过程中的精度问题，建议进行复核。
- **Model performance discrepancy with eval mode**: 一位用户发现，当使用 bfloat16 训练的 Lottery ticket hypothesis 模型进入 eval 模式时，推理性能大幅下降。
   - 他们怀疑是 BatchNorm 的问题，但通过不使用 model.eval() 恢复了性能，这被认为很奇怪。
- **flex_attention for block-diagonal masks**: 一位用户询问关于使用最近的 `flex_attention` 函数来训练带有 block-diagonal masks 的模型。
- **Optimization in test-time-training repo**: 一位用户提议为 [test-time-training PyTorch 仓库](https://github.com/test-time-training/ttt-lm-pytorch) 添加优化。



**Link mentioned**: <a href="https://github.com/test-time-training/ttt-lm-pytorch?tab=readme-ov-file">GitHub - test-time-training/ttt-lm-pytorch: Official PyTorch implementation of Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>: Official PyTorch implementation of Learning to (Learn at Test Time): RNNs with Expressive Hidden States - test-time-training/ttt-lm-pytorch

  

---

### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1260968781136662652)** (1 messages): 

> - `Adam Mini`
> - `GrokFast`
> - `MobileLLM`
> - `JEST` 


- **Adam Mini 以更低显存进行优化**: [Adam Mini](https://datta0.substack.com/p/ai-unplugged-14-adam-mini-grokfast) 是一种在保持与 Adam 相当的性能的同时，显著减少显存使用的优化器。
   - 通过跟踪更少的参数（momentum, variance, gradient），Adam Mini 有效地降低了显存占用。
- **GrokFast 加速 Grokking 现象**: [GrokFast](https://datta0.substack.com/p/ai-unplugged-14-adam-mini-grokfast) 旨在加速在 Transformers 中观察到的 Grokking 现象，该现象此前曾被讨论过。
   - 这种方法有助于模型在记忆（memorization）与泛化（generalization）之间快速达成平衡。
- **MobileLLM 将 LLMs 带到设备端**: [MobileLLM](https://datta0.substack.com/p/ai-unplugged-14-adam-mini-grokfast) 专注于为端侧应用开发大语言模型，增强其在移动平台上的可访问性和性能。
   - 这一努力旨在无需远程服务器访问即可实现强大的 AI 功能。
- **JEST 通过数据策划分选提升训练效率**: [JEST](https://datta0.substack.com/p/ai-unplugged-14-adam-mini-grokfast) 代表联合样本选择训练（Joint Example Selection Training），这是一种通过精心挑选训练样本来加速学习的数据策划分选技术。
   - 该方法通过专注于最具影响力的数据来优化训练过程。



**提到的链接**: <a href="https://datta0.substack.com/p/ai-unplugged-14-adam-mini-grokfast">AI Unplugged 14: Adam mini, GrokFast, MobileLLM, JEST</a>: 洞察胜过信息

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1260697578220032080)** (2 messages): 

> - `AMD 与 Silo AI 收购案`
> - `FlashAttention 与 GPU 性能` 


- **AMD 以 6.65 亿美元收购 Silo AI**: [AMD](https://www.ft.com/stream/8d882704-0892-489c-af27-b752e9d253d3) 将以 **6.65 亿美元**收购芬兰 AI 初创公司 Silo AI，此举旨在加强其 AI 服务并与 Nvidia 竞争。该收购预计将于今年下半年完成，届时 Silo 的 300 人团队将专注于构建定制的大语言模型。
- **FlashAttention 赋能现代 GPUs**: [FlashAttention](https://pytorch.org/blog/flashattention-3/) 通过减少显存读写来优化 GPU 性能，显著加快了 Transformer 的训练和推理速度。
   - 尽管取得了成功，但 FlashAttention-2 在 H100 GPUs 上仅利用了理论最大 FLOPs 的 35%，这表明仍有进一步优化的空间。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/flashattention-3/?utm_content=300091694&utm_medium=social&utm_source=twitter&hss_channel=tw-776585502606721024">FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision</a>: Attention 作为无处不在的 Transformer 架构的核心层，是大语言模型和长上下文应用的瓶颈。FlashAttention（以及 FlashAttention-2）开创了一种方法...</li><li><a href="https://www.ft.com/content/7b8d2057-2687-45b3-bae4-1488a75ac5b2?accessToken=zwAGHOsuEnXwkc97jSBXJodFs9O65BSIp1rFsg.MEQCIFYunY6DwEMvMTIO2J7JemqoIPbFX62lSbBxn0opQKO7AiBtXWO7ZlNVuM8gyc_9YZDDQ0F8E_oL61YIxfHTWHE0Hg&sharetype=gift&token=98e4f39b-f46b-47ae-b1d3-353090a545c8">AMD to buy Finnish start-up Silo AI for $665mn in drive to compete with Nvidia </a>: 这家总部位于加州的芯片制造商进行的纯现金收购，是欧洲十年来同类规模最大的收购。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1260711245309546576)** (11 条消息🔥): 

> - `CUDA 环境设置`
> - `NCU segmentation fault`
> - `WSL 的 GPU 驱动更新`
> - `CUDA 的 Docker 使用` 


- **通过 Docker 解决 NCU segmentation fault**：**Shabbo** 在本地笔记本 GPU (3050) 的 Conda 环境中运行 `ncu` 时遇到了 'Segmentation fault'，最终通过使用 Docker 镜像 `nvidia/cuda:12.4.0-devel-ubuntu22.04` 解决了该问题。
   - 需要注意的是，根据[此处](https://forums.developer.nvidia.com/t/nsight-compute-on-wsl2/293369)的参考，Windows 10 WSL2 需要 **ncu 2023.3 或更高版本**，同时还需要按照[此处](https://developer.nvidia.com/ERR_NVGPUCTRPERM)的说明在 Windows 中设置 GPU 权限。
- **潜在的 Conda 问题及替代建议**：Shabbo 询问是否是 Conda 环境设置或缺少系统级 CUDA 安装导致了 `ncu` 问题，另一位成员建议升级 Windows 上的 GPU 驱动并验证系统 CUDA toolkit 的安装情况。
   - 建议包括使用 Docker 作为替代方案，并升级宿主机 Windows GPU 驱动以更好地支持 WSL，参考[此处](https://forums.developer.nvidia.com/t/error-profiling-is-not-supported-on-device-0-as-it-uses-the-windows-subsystem-for-linux-wsl/260814)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://forums.developer.nvidia.com/t/error-profiling-is-not-supported-on-device-0-as-it-uses-the-windows-subsystem-for-linux-wsl/260814">==ERROR== Profiling is not supported on device 0 as it uses the Windows Subsystem for Linux (WSL)</a>：我尝试在 WSL2 (ubuntu 22.04) 上使用 ncu CLI 来分析 Python 目标（在 Python 中使用 PyTorch）。但出现了问题，我不知道如何解决。你能帮我吗？ncu cli: ncu -...</li><li><a href="https://forums.developer.nvidia.com/t/nsight-compute-on-wsl2/293369">Nsight compute on WSL2</a>：我想在 WSL2 上使用 Nsight compute (nv-nsight-cu-cli --set detailed -o result ./result)。但是，它显示 &quot; ==ERROR== Profiling is not supported on device 0 as it uses the Windows Subsystem fo...</li><li><a href="https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters">NVIDIA Development Tools Solutions - ERR_NVGPUCTRPERM: Permission</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1260762402748633342)** (2 条消息): 

> - `支持 Smooth Quant 和 AWQ`
> - `to_calibrating_ 函数的实现` 


- **支持 Smooth Quant 和 AWQ 算法**：确认当前工作流已支持 **Smooth Quant 和 AWQ**。
   - *成员提议在评估统一方法之前，先为每个算法单独实现 `to_calibrating_`。*
- **为所有算法分别实现 to_calibrating_**：**to_calibrating_** 函数的实现在初期应针对每个算法保持独立。
   - *后续评估可能会将其合并到单一流程中，类似于 `quantize_` API。*


  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1260928993297961001)** (1 条消息): 

> - `BitBlas 后端`
> - `torch.compile 支持` 


- **新增支持 Torch.Compile 的 BitBlas 后端**：[MobiusML](https://github.com/mobiusml/hqq/commit/62494497a13174d7a95d3f82c8f9094a5acd3056) 为 **hqq** 添加了支持 **torch.compile** 的 BitBlas 后端。该更新现在支持 **4-bit, 2-bit 和 1-bit** 配置。
   - 该 [commit 包含了详细的变更](https://github.com/mobiusml/hqq/commit/62494497a13174d7a95d3f82c8f9094a5acd3056)和后端的改进。
- **支持多种 Bit 配置**：**hqq** 的最新更新通过 BitBlas 后端实现了对 **4-bit, 2-bit 和 1-bit** 配置的支持。
   - 此增强功能利用 [torch.compile](https://github.com/mobiusml/hqq/commit/62494497a13174d7a95d3f82c8f9094a5acd3056) 能力来提高性能和兼容性。



**提到的链接**：<a href="https://github.com/mobiusml/hqq/commit/62494497a13174d7a95d3f82c8f9094a5acd3056">add bitblas backend for 4-bit/2-bit · mobiusml/hqq@6249449</a>：未找到描述

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1260672432293806263)** (252 条消息🔥🔥): 

> - `Bias Handling in Models` (模型中的 Bias 处理)
> - `Parameterized GPT2 Training` (参数化 GPT2 训练)
> - `Custom Attention Implementations` (自定义 Attention 实现)
> - `AdamW Optimizer Precision` (AdamW 优化器精度)
> - `FlashAttention-3` 


- **已训练模型中 Bias 的处理方式存在差异**：成员们讨论了训练运行期间 Bias 的操作范数，注意到虽然 Loss 曲线相似，但范数不同，在 Checkpoint 中观察到 **Bias 的量级差异巨大**。
   - Andrej 建议将 Bias 保持为零而不是直接移除，旨在避免编写令人困惑且复杂的代码。
- **参数化 GPT-2 训练脚本**：成员们分享了用于训练 **GPT-2 模型** 的脚本，这些脚本带有排除 Bias 的命令行选项，并在 Loss 指标上实现了显著的相似性。
   - 实验正在进行中，未来计划在确保命令配置简洁且易于管理的同时，微调并扩展模型参数。
- **CUDA 中的自定义 Attention 实现**：社区讨论了使用 FlashAttention-3 还是 cuDNN 和 ThunderKittens 来实现更快的 Transformer Attention，并探讨了这些库的复杂性和依赖性。
   - 在必须进行更复杂的集成之前，人们更倾向于保留简单的解决方案，例如通过 **CUTLASS** 创建自定义 matmul。
- **FP8 对 AdamW 优化器的影响**：探索了对 Activation 和优化器状态（特别是 AdamW）的 FP8 支持，结果显示 **优化器状态占用了大量内存**，在单 GPU 的情况下接近 50%。
   - 关于 Adam Buffer 精度优化的讨论引发了对转向低位宽精度（例如 8-bit）的担忧，需要在复杂性和潜在的不准确性之间取得平衡。
- **FlashAttention-3 的影响与采用**：[FlashAttention-3](https://x.com/tri_dao/status/1811453622070444071) 因其令人印象深刻的性能而受到关注，在 FP16 下将 Attention 速度提升了高达 2 倍，在 FP8 下达到 1.2 PFLOPS。
   - 成员们考虑了集成的可行性以及该路径与其他优化方案的评估，重点在于简洁性和实用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/tri_dao/status/1811453622070444071">来自 Tri Dao (@tri_dao) 的推文</a>：FlashAttention 被广泛用于加速 Transformer，已经使 Attention 速度提升了 4-8 倍，但尚未充分利用现代 GPU。我们发布了 FlashAttention-3：FP16 速度提升 1.5-2 倍，u...</li><li><a href="https://huggingface.co/spaces/llmc/llmc_1558M">llm.c 1558M 演示 - llmc 的 Hugging Face Space</a>：未找到描述</li><li><a href="http://llmc.s3-us-west-2.amazonaws.com/html/gpt2_vs_llmc30kedu.html">并排文本文件</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/discussions/677">让我们在 llm.c 中复现 GPT-2 (1.6B)：一个 8XH100 节点，24 小时，$672 · karpathy/llm.c · Discussion #677</a>：在这篇文章中，我们正在 llm.c 中复现 GPT-2。这是“那个 GPT-2”，即 OpenAI 博客文章《Better Language Models and their Implications》中介绍的完整的 1558M 参数版本...</li><li><a href="https://github.com/karpathy/llm.c/pull/675">由 gordicaleksa 提交的添加移除 Bias 选项的 Pull Request #675 · karpathy/llm.c</a>：添加一个命令行选项，允许我们在 attn/fc 层中不使用 Bias。</li><li><a href="https://news.ycombinator.com/item?id=40939707">Karpathy：让我们在 llm.c 中复现 GPT-2 (1.6B)：一个 8XH100 节点 24h $672 | Hacker News</a>：未找到描述
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[sparsity](https://discord.com/channels/1189498204333543425/1247663759434977453/1260979961922064436)** (4 条消息): 

> - `Quantization and Sparsity`
> - `加速技术`
> - `SparseGPT`
> - `WANDA Pruning`
> - `稀疏化模型的 Distillation` 


- **Quantization 和 Sparsity 策略**：探索了 Quantization 和 Sparsity 的结合，建议为非稀疏元素保留更高的位宽（bitwidth），在不消耗更多存储或计算资源的情况下提高质量。
   - **50% 半结构化稀疏（semi-structured sparsity）** 的质量损失极小，并具有计算优势。
- **利用量化稀疏矩阵实现加速**：一个融合的 gemv CUDA kernel 在使用特定格式（1:2 稀疏度及 7-bit 非稀疏元素）时，展示了接近 4 倍的加速。
   - 通过高效地 **打包稀疏矩阵（packing sparse matrices）** 展示了加速效果，在各种矩阵形状下实现了 **3.7337x 到 3.3228x** 的速度提升。
- **针对大模型的 SparseGPT Pruning**：SparseGPT 能够在不进行重新训练的情况下，将大型 GPT 系列模型 Pruning 至 **50% 稀疏度**，同时保持准确性。
   - SparseGPT 可以在 4.5 小时内处理 **OPT-175B** 和 **BLOOM-176B** 等著名的开源模型，实现高达 **60% 的非结构化稀疏（unstructured sparsity）**，且 Perplexity（困惑度）增量微乎其微。
- **WANDA Pruning 方法**：**WANDA** 方法提供了一种简单且有效的 LLM Pruning 技术，并与权重 Quantization 方法兼容。
   - [GitHub 链接：WANDA](https://github.com/locuslab/wanda) 提供了关于其实现和有效性的更多细节。
- **关于 Distillation 的进一步实验**：未来计划对稀疏化模型运行 **Distillation**，以评估性能和准确性的提升。
   - 通过修改打包策略，可以实现进一步的加速和效率提升，在 1:4 稀疏度和 **6-bit Quantization** 下，速度可能提高 **6x-7x**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/locuslab/wanda">GitHub - locuslab/wanda: 一种简单有效的 LLM Pruning 方法。</a>：一种简单有效的 LLM Pruning 方法。通过在 GitHub 上创建账号为 locuslab/wanda 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2301.00774">SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot</a>：我们首次展示了大规模生成式预训练 Transformer (GPT) 系列模型可以在 one-shot 情况下被 Pruning 至至少 50% 的稀疏度，无需任何重新训练，且准确度损失极小...
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1260765132938805268)** (2 条消息): 

> - `Orca 3`
> - `Generative Teaching`
> - `语言模型的合成数据 (synthetic data)` 


- **Orca 的 Generative Teaching 革命**：Arindam1408 宣布了他们在 Generative Teaching 方面的最新工作，为语言模型生成 [高质量合成数据 (synthetic data)](https://x.com/Arindam1408/status/1810835231550939470)，以教授特定技能（如 RC、文本分类、工具使用和数学），而无需大量人力投入。
   - 一位用户评论说，**Orca 3** 因为其 *隐晦的论文标题* 而未引起广泛注意。
- **Orca 3 低调发布背后的原因**：**Orca 3** 的发布没有被广泛察觉，引发了对其曝光度的质疑。
   - 420gunna 认为这是因为他们给它起了一个 *隐晦的论文标题*。



**提到的链接**：<a href="https://x.com/Arindam1408/status/1810835231550939470">来自 arindam mitra (@Arindam1408) 的推文</a>：#Orca 我很高兴宣布我们在 Generative Teaching 方面的最新工作：为语言模型生成大量多样化的高质量合成数据，以教授特定技能（例如 RC、文本分类...）

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1260694803939594312)** (177 条消息🔥🔥): 

> - `Hermes Model Performance` (Hermes 模型性能)
> - `Open-Source AI` (开源 AI)
> - `Dataset Availability` (数据集可用性)
> - `Guardrails for AI` (AI 护栏)
> - `Arena Learning for LLMs` (LLM 的 Arena Learning)


- **Hermes 模型表现出令人印象深刻的性能**：一位成员指出，一个仅用 10 个样本、1e-6 的学习率训练了 40 个 epochs 的模型表现出令人印象深刻的 OOS 性能，在使用 Mistral 时能产生完美的 JSON 输出。
   - 讨论强调，低学习率结合高 epochs 可能对于样本量较小的特定任务非常理想。
- **开源 AI 缺乏数据集**：一位成员认为，虽然像 LLaMa 3 和 Gemini 2 这样的模型非常先进，但 OSS 项目缺乏教导模型特定技能所需的数据集和流水线（pipelines）。
   - *我们完全缺乏像 Gemini 或 LLaMa 3 这样高度智能 LLMs 的开源（OSS）复制品。*
- **AI 护栏（Guardrails）辩论升温**：成员们就 AI 护栏的必要性和影响展开了辩论，一些人认为需要护栏来防止滥用，而另一些人则认为护栏过于严格，扼杀了创新。
   - 一位用户评论道：*AI 护栏应该像圆锯上的防护罩；它存在，但在必要时可以拆除。*
- **WizardLM 推出 Arena Learning**：WizardLM 发布了 Arena Learning 论文，描述了一个 AI 驱动的合成数据飞轮和模拟聊天机器人竞技场，用于在没有人类评估员的情况下持续改进 LLM。
   - Arena Learning 利用迭代 SFT、DPO 和 PPO 训练后技术，实现了与人类评判的 LMSYS Chatbot Arena 评估 **98.79% 的一致性**。
- **vLLM 模型 JSON 模式与引导式解码（Guided Decoding）**：关于 vLLM 使用引导式解码强制执行 JSON 输出能力的讨论，指出其在第一次请求时较慢，但随后非常高效。
   - 强调可以实现高效的 JSON 模式提示词，确保遵循指定的 JSON schemas。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Druvith/Tiny_StoriesMoE">Druvith/Tiny_StoriesMoE · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/pat-gif-19836593">Pat GIF - Pat - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/simon-mo/vllm/blob/7290ea75f9bdee72c2d4c18e5fd27d2d5d464e4e/vllm/model_executor/guided_decoding.py">vllm/vllm/model_executor/guided_decoding.py at 7290ea75f9bdee72c2d4c18e5fd27d2d5d464e4e · simon-mo/vllm</a>：一个高吞吐量且内存高效的 LLM 推理和服务引擎 - simon-mo/vllm</li><li><a href="https://github.com/vllm-project/vllm/issues/3148">Support `response_format: json_object` in OpenAI server · Issue #3148 · vllm-project/vllm</a>：我们刚刚合并了对 Outlines 结构化生成的支持。下一步是将基于语法的有限状态机 outlines-dev/outlines#541 集成到 vLLM 中，以支持任意...</li><li><a href="https://x.com/WizardLM_AI/status/1811435119997075550">来自 WizardLM (@WizardLM_AI) 的推文</a>：🎉今天我们发布了 WizardLM 的新论文！🔥 Arena Learning 是 WizardLM-2 最重要的技术之一。为了构建下一代数据飞轮，我们提出了一个离线模拟...</li><li><a href="https://huggingface.co/datasets/SkunkworksAI/reasoning-0.01?row=0">SkunkworksAI/reasoning-0.01 · Datasets at Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1260684303357055016)** (7 条消息): 

> - `Hermes 2 Theta Llama 3 70B Finetunes` (Hermes 2 Theta Llama 3 70B 微调)
> - `Hermes 2 Pro`
> - `Storytelling Focused Finetunes` (专注于故事创作的微调)


- **探讨 Hermes 2 Theta Llama 3 70B 微调**：一位成员询问除了 [Hermes-2-Theta-Llama-3-70B-GGUF](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B-GGUF) 之外，是否还有其他在基础 instruct 性能上有所改进的 **llama 3 70b 微调**。
   - 另一位成员提到 **Hermes 2 Pro** 在典型的 Nous 基准测试中表现良好，但并非在所有测试（如 **IFEval**）中都超过其他模型。
- **关于专注于故事创作的微调的讨论**：一位成员询问是否有兴趣在基础模型上开发专注于故事创作的微调，并提到 **NovelAI** 似乎是唯一积极追求这一方向的团队。
   - 虽然开源替代方案会受到欢迎，但目前*没有计划*使用预训练数据来开发故事创作模型。


  

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1260778912565166115)** (13 messages🔥): 

> - `Anthropic Workbench`
> - `Prompt Engineering Job Replacement`
> - `Grounded vs Ungrounded Tags`
> - `Hermes RAG Templates`
> - `Synthetic Generations Export` 


- **Anthropic Workbench 缺少导出功能**：一位用户提到在使用 **Anthropic Workbench** 时，指出它需要一个针对合成生成数据（synthetic generations）的**导出功能**。
- **关于移除 Grounded/Ungrounded 标签的辩论**：用户讨论了移除 **grounded/ungrounded 标签**以节省 Token 并专注于 grounded 回答生成的想法，因为这两种标签的响应结果非常相似。
- **Prompt Engineering 作为一种职业正在演变**：用户表示，仅专注于构建 Prompt 的 **prompt engineering** 可能会过时。
- **在 Google Doc 中追踪 Hermes RAG 模板**：用户分享了一个 [Google Doc](https://docs.google.com/document/d/1KDYbobQBLuGCMAhpmkvOHVyQ22R2TXNzpvmcnP0Sh44/edit) 用于追踪各种 **Hermes RAG 模板**。



**提及的链接**：<a href="https://docs.google.com/document/d/1KDYbobQBLuGCMAhpmkvOHVyQ22R2TXNzpvmcnP0Sh44/edit">Hermes RAG Templates</a>：Cohere-Hermes 格式：[interstellarninja] System Prompt:____________________________________________________________________________________  # RoleYou are an AI assistant that answers user queri...

  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1260714980463083530)** (90 messages🔥🔥): 

> - `Feature Requests for LM Studio`
> - `GPU Compatibility Issues`
> - `Context Overflow Bug`
> - `Setup and Configuration Tips`
> - `Model and Proxy Issues` 


- **用户请求可选的 Assistant 角色生成触发器**：一位用户建议在 LM Studio 中允许通过 Assistant 角色输入来触发生成，将其视为一种可以增强叙事写作的 UI/UX 功能。
   - 他们强调这可以是一个简单的可选设置（如布尔值），默认保持关闭，但可以针对特定用例开启。
- **Linux 上的 GPU 兼容性问题**：一位用户报告称 LM Studio 无法识别其 Radeon RX7600XT GPU，尽管 GPT4ALL 可以成功使用它。
   - [OpenCL GPU 支持已弃用](https://llama.cpp)，未来的更新可能会切换到 Vulkan，以更好地兼容非 CUDA/ROCM GPU。
- **Context overflow 策略 Bug**：一位用户在将 Context overflow（上下文溢出）策略设置为“维持滚动窗口并截断历史消息”时遇到了问题。
   - 尽管最初复现了该问题，但它似乎在没有一致行为的情况下自行解决了，目前正在考虑提交详细的 Bug 报告。
- **在代理后运行 LM Studio**：一位用户询问关于在代理（proxy）后运行 LM Studio 的问题，提到该应用无法识别 Windows 10 中配置的代理设置。
   - 建议的解决方法是手动下载模型并将其放置在正确的文件夹结构中。
- **廉价配置的优化和安装建议**：一位用户分享了在二手店购买的配备 GTX1650 GPU 的 Dell Inspiron 3847 上安装 LM Studio 的经验。
   - 社区建议运行较小的模型（如 7B Q4）并安装 Linux 以获得更好的性能。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.nvidia.com/Download/driverResults.aspx/228212/en-us/">GeForce Game Ready Driver | 556.12 | Windows 10 64-bit, Windows 11 | NVIDIA</a>：下载适用于 Windows 10 64 位、Windows 11 系统的英文（美国）GeForce Game Ready 驱动程序。发布日期：2024.6.27</li><li><a href="https://www.hardware-corner.net/desktop-models/Dell-Inspiron-3847/">Dell Inspiron 3847 &#8211; Specs and upgrade options</a>：阅读关于 Dell Inspiron 3847 台式电脑的信息。查找详细规格、升级选项以及关于 CPU、RAM、PSU、主板和发布日期的信息</li><li><a href="https://tenor.com/view/spongebob-slow-down-jethro-pioneers-gif-5521176">Spongebob Slow GIF - Spongebob Slow Down - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://nvidia.custhelp.com/app/answers/detail/a_id/5557">NVIDIA Support</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues">Issues · lmstudio-ai/lmstudio-bug-tracker</a>：LM Studio 桌面应用程序的 Bug 追踪 - Issues · lmstudio-ai/lmstudio-bug-tracker</li><li><a href="https://www.hardware-corner.net/desktop-models/Dell-I">Dell I &#8211; Specs and upgrade options</a>：阅读关于 Dell I 台式电脑的信息。查找详细规格、升级选项以及关于 CPU、RAM、PSU、主板和发布日期的信息
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1260679587201814608)** (23 messages🔥): 

> - `Whisper 与 LM Studio 的集成`
> - `Gemma-2 Flash Attention 问题`
> - `处理不支持系统提示词的模型`
> - `使用 Ollama 和 LM Studio 安装模型`
> - `Salesforce 推出 xLAM-1B` 


- **Whisper 与 LM Studio 的集成可能性**：用户讨论了将 Whisper 或其他语音转文字模型与 LM Studio 集成的潜力，建议采用类似于 Anything LLM 的框架。
- **Gemma-2 Flash Attention 设置引发问题**：**Gemma-2** 不支持 Flash Attention，尝试使用时会导致问题。
- **不支持系统提示词模型的处理**：讨论了 LM Studio 如何处理像 **Gemma** 和 **Mistral** 这样官方不支持系统提示词的模型。
- **使用 Ollama 和 LM Studio 高效安装模型**：一位用户发现使用 **Ollama** 安装模型并将其链接到 **LM Studio** 效率更高。
- **Salesforce 的新型微模型 xLAM-1B**：Salesforce 推出了一款名为 **Einstein Tiny Giant xLAM-1B** 的 1B 参数模型，据称在 function calling（函数调用）方面优于 GPT-3.5 和 Claude 等更大型的模型。



**提及的链接**：<a href="https://x.com/Benioff/status/1808365628551844186">来自 Marc Benioff (@Benioff) 的推文</a>：了解 Salesforce Einstein “Tiny Giant”。我们的 1B 参数模型 xLAM-1B 现在是函数调用的最佳微模型，性能优于其规模 7 倍的模型，包括 GPT-3.5 和 Claude。设备端 Agentic ...

  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1260787588340387920)** (30 messages🔥): 

> - `8cx`
> - `Windows 更新`
> - `双 4090 GPU 对比等待 5090`
> - `RX 580 配置`
> - `Arc 770 性能` 


- **高通 8cx 适用于 llama.cpp**：一名成员提到正在讨论将 **[Qualcomm 8cx](https://www.qualcomm.com/developer/blog/2024/04/big-performance-boost-for-llama-cpp-and-chatglm-cpp-with-windows)** 用于 llama.cpp，并在 open issues 中注意到了一些改进。
- **投资双 4090 还是等待 5090**：关于现在购买两块 **4090 GPU** 还是等待 **5090** 展开了激烈讨论。
   - 一些人主张等待，因为价格可能下降，且二手 **3090** 性能相当；另一些人则将 **50 系列** 的潜在规格视为决定因素。
- **RX 580 配置难以维持现状**：**RX 580** 被标记为过时且不受支持，**OpenCL** 已被弃用。
   - 社区驱动可以支持 **ROCm**，但一名成员警告说这是一种冒险的配置，称其为 *“魔鬼的游戏”*。
- **Arc 770 性能不尽如人意**：用户确认 **Arc 770 16GB** 在 LM Studio 中运行缓慢且不受支持。
   - 一名成员建议攒钱买 **3090** 以获得更好的性能。
- **3080 VRAM 限制迫使用户升级**：一名用户对 **3080** 的 **10GB VRAM** 限制表示沮丧，正在寻找廉价的替代方案用于后台使用。
   - 建议倾向于以较低价格购买 **3090**，并警告 **AMD** 显卡可能仍存在驱动问题。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=YiX9p8A7LqE">AI/ML/DL GPU 购买指南 2024：为您的预算获取最强的 AI 算力</a>：欢迎阅读 2024 年终极 AI/ML/DL GPU 购买指南！在这份综合指南中，我将帮助您在选择理想显卡时做出明智的选择...</li><li><a href="https://docs.google.com/spreadsheets/d/1jDLieMm-KroKY6nKv40amukfFGAGaQU8tFfZBM7iF_U/edit?gid=2040312891#gid=2040312891">AI/ML - 资源手册与硬件计算</a>：AI 网站与工具类别，名称，描述，许可证，语言，链接，网站，备注代码，移动人工智能
，MIT，Dart，&lt;a href=&quot;https://github.com/Mobile-Artificial-Intelligence&quo...</li><li><a href="https://en.wikipedia.org/wiki/DL_Boost">DL Boost - 维基百科</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Advanced_Matrix_Extensions">Advanced Matrix Extensions - 维基百科</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/AVX-512">AVX-512 - 维基百科</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1260768603222052965)** (5 messages): 

> - `Rust development`
> - `Etiquette of asking questions`
> - `The XY problem` 


- **关于 Rust 开发的讨论**：一名成员询问是否有 Rust 开发者可以提供意见或建议。
   - 该成员提到*只是想征求意见*，但未说明具体问题。
- **理解提问的礼仪**：一名成员分享了 [Don't Ask To Ask](https://dontasktoask.com/) 的链接，强调在没有直接说明问题的情况下寻找专家的做法是不妥的。
   - 该成员强调，此类提问隐含的要求往往比表面看起来更多，可能会让愿意提供帮助的人望而却步。
- **探讨 XY 问题**：在讨论提问礼仪之后，另一名成员链接了 [XY Problem](https://xyproblem.info/)，以解释常见的错误：即针对尝试性的解决方案寻求帮助，而不是针对实际问题本身。
   - 他们还引用了 *[Asking Smart Questions](http://www.catb.org/esr/faqs/smart-questions.html)*（提问的智慧）作为更好地构建问题的有用资源。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dontasktoask.com/">Don't ask to ask, just ask</a>：未找到描述</li><li><a href="https://xyproblem.info/">Home - The XY Problem</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1260673456299835502)** (36 messages🔥): 

> - `timestamped whisper`
> - `useful OpenAI API integrations`
> - `Blackstone's investment in AI data centers`
> - `PaliGemma report`
> - `OpenAI's revenue and progress towards AGI` 


- **Whisper Timestamped 支持浏览器本地语音识别**：Whisper Timestamped 提供**多语言语音识别**，支持词级时间戳，并 100% 在浏览器本地运行，由 🤗 Transformers.js 驱动，这为**浏览器内视频编辑**开辟了新的可能性。[来源](https://x.com/xenovacom/status/1811068015229747335)
- **征集有用的 OpenAI API 集成**：讨论集中在 OpenAI API 的实用专业应用上，建议包括改进**内部搜索**和 **CI 失败自动修复**。
- **Blackstone 向 AI 数据中心投资 1000 亿美元**：Blackstone 正在大力投资 AI，目前拥有 **500 亿美元的 AI 数据中心**，并计划再建设 **500 亿美元**。[YouTube 访谈](https://youtu.be/Z4EK9_s_ui8?si=v-xIlI78irXLWPhu)
- **PaliGemma 最新进展详情**：PaliGemma 在 **arxiv** 上的最新论文讨论了一个 3B 参数的 VLM 模型，集成了 **SigLip 图像编码器**和 **Gemma 语言模型**。[链接](https://arxiv.org/abs/2407.07726) [详情](https://x.com/A_K_Nain/status/1811258845844373930)
- **OpenAI 的营收和进展阶段**：一份报告估计 OpenAI 的年营收为 **34 亿美元**，主要收入来自 ChatGPT Plus、Enterprise 和 API 订阅。[来源](https://x.com/jvnixon/status/1811278381184672156?s=61)


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/A_K_Nain/status/1811258845844373930">Aakash Kumar Nain (@A_K_Nain) 的推文</a>：PaliGemma 的论文发布了 (🥳🎉)。快速总结如下：- 3B VLM - 开源基础 VLM - (图像 + 文本) 作为输入 (prefix) -> 文本 (suffix) 架构 - 图像编码器：形状优化的 ViT So4...</li><li><a href="https://lu.ma/b8ouvgpp">vLLM：面向所有人的简单、快速且廉价的 LLM 服务 · Zoom · Luma</a>：面向开发者、构建者、AI 爱好者以及任何希望优化 LLM 服务并有机会为开源项目做出贡献的人。📅 时间：7月18日，...</li><li><a href="https://x.com/jvnixon/status/1811278381184672156?s=61">Jeremy Nixon (@JvNixon) 的推文</a>：futureresearch 关于 OpenAI 营收的报告已发布，显示：ChatGPT Plus 收入 19 亿美元（770 万订阅者，20 美元/月），ChatGPT Enterprise 收入 7.14 亿美元（120 万订阅者，50 美元/月），API 收入 5.1 亿美元，以及 2.9 亿...</li><li><a href="https://x.com/shiringhaffary/status/1811508824970264595?s=61">Shirin Ghaffary (@shiringhaffary) 的推文</a>：OpenAI 提出了一个包含 5 个级别的框架来跟踪 AGI 的进展，并认为目前接近第 2 级（“Reasoners”）。在最近的全员会议上，领导层还展示了一个研究演示...</li><li><a href="https://x.com/xenovacom/status/1811068015229747335?s=46">Xenova (@xenovacom) 的推文</a>：介绍 Whisper Timestamped：具有单词级时间戳的多语言语音识别，得益于 🤗 Transformers.js，100% 在浏览器中本地运行！这为...开启了无限可能。</li><li><a href="https://x.com/giffmana/status/1811146269605384298?s=46">Lucas Beyer (bl16) (@giffmana) 的推文</a>：首先，它是一个 Prefix-LM。图像和前缀（=用户输入）之间是全注意力（Full attention），仅在后缀（=模型输出）上是自回归的。直觉是，通过这种方式，图像 Token 可以看到查询...</li><li><a href="https://x.com/giffmana/status/1811146264832037303?s=46">Lucas Beyer (bl16) (@giffmana) 的推文</a>：✨PaliGemma 报告今晚将上线 arXiv。我们努力让它变得有趣，而不是简单的“这是模型。SOTA 结果。谢谢再见。” 所以这里是我们做的一些有趣的消融实验（ablations），查看...</li><li><a href="https://lu.ma/campfire-connect?tk=3R67IK">CampFire Connect：Fireworks AI 举办的 AI 开发者虚拟活动 · Luma</a>：嘿！我们很高兴欢迎你参加我们的首个 AI 开发者虚拟活动：CampFire Connect。活动对所有人开放，虚拟形式，只需快速...</li><li><a href="https://x.com/tri_dao/status/1811453622070444071">Tri Dao (@tri_dao) 的推文</a>：FlashAttention 被广泛用于加速 Transformers，已经让 Attention 快了 4-8 倍，但尚未利用现代 GPU 的优势。我们正在发布 FlashAttention-3：在 FP16 上快 1.5-2 倍，使用...</li><li><a href="https://youtu.be/Z4EK9_s_ui8?si=v-xIlI78irXLWPhu">Blackstone 的 80 万亿美元豪赌 | Iconoclast Summit 2024</a>：重塑宇宙：Blackstone 的 80 万亿美元豪赌，对话访谈者：Forbes 主席兼总编辑 Steve Forbes 和 Jonathan Gray...</li><li><a href="https://x.com/teortaxesTex/status/1810881199025574077">Teortaxes▶️ (@teortaxesTex) 的推文</a>：你可能忽略了 Harmonic 和 @tachim。我认为这不太公平，他们在数学推理 AI 领域看起来很有势头，而且很有可能达到那个...</li><li><a href="https://arxiv.org/abs/2407.07726">PaliGemma：一个用于迁移学习的多功能 3B VLM</a>：PaliGemma 是一个开源视觉语言模型 (VLM)，基于 SigLIP-So400m 视觉编码器和 Gemma-2B 语言模型。它被训练成一个多功能且知识广泛的基础模型...</li><li><a href="https://github.com/OpenDevin/OpenDevin">GitHub - OpenDevin/OpenDevin: 🐚 OpenDevin: 少写代码，多产出</a>：🐚 OpenDevin: 少写代码，多产出。通过在 GitHub 上创建账户来为 OpenDevin/OpenDevin 的开发做出贡献。</li><li><a href="https://github.com/entropy-research/Devon">GitHub - entropy-research/Devon: Devon：一个开源的结对编程器</a>：Devon：一个开源的结对编程器。通过在 GitHub 上创建账户来为 entropy-research/Devon 的开发做出贡献。</li><li><a href="https://overcast.fm/+QLdvcWpGA">为什么《大西洋月刊》与 OpenAI 签署协议 — Decoder with Nilay Patel — Overcast</a>：未找到描述。
</li>
</ul>

### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1260674685469982792)** (93 条消息🔥🔥): 

> - `ColBERT 论文讨论`
> - `AI Agent 综述论文`
> - `ImageBind 模态`
> - `SBERT 设计与训练`
> - `AI 中的 Multi-agent 系统` 


- **ColBERT 论文回顾**：讨论了 [ColBERT 论文](https://arxiv.org/pdf/2004.12832) 及其特性，包括其倒排索引检索方法的优势。
   - 会议包含了关于 ColBERT 与其他语义相似度模型对比的见解，展示了其在处理大规模数据集时的高效性。
- **AI Agent 实现综述**：回顾了 [AI Agent 综述论文](https://arxiv.org/abs/2404.11584)，重点关注 AI Agent 实现的最新进展及其能力。
   - 讨论集中在架构、设计选择以及未来发展对增强 AI Agent 性能的重要性。
- **探索 ImageBind 的联合嵌入**：一篇关于 [ImageBind](https://arxiv.org/abs/2305.05665) 的论文讨论了为多种模态（如图像、文本和音频）创建联合嵌入（Joint Embedding）。
   - 与会者注意到其创新性地使用图像配对数据进行训练，及其在跨模态任务中的 SOTA 性能。
- **理解 SBERT 设计**：分享了 SBERT (Sentence-BERT) 设计和训练的细节，强调了其使用带有池化层的 BERT 来生成句子嵌入。
   - 对比训练方法（如 Siamese Networks）因其在获取有意义的句子表示方面的有效性而受到关注。
- **AI 中的 Multi-Agent 系统**：详细讨论了 AI 中 Multi-Agent 系统的结构和功能，强调了不同系统提示词（System Prompts）的作用。
   - 分享了使用 Multi-Agent 框架的操作性原因及其在并行任务执行中的应用。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2305.05665">ImageBind: One Embedding Space To Bind Them All</a>: 我们提出了 ImageBind，一种在六种不同模态（图像、文本、音频、深度、热成像和 IMU 数据）之间学习联合嵌入的方法。我们展示了并非所有配对数据组合都是必需的...</li><li><a href="https://x.com/_xjdr">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.11584">The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey</a>: 本综述论文考察了 AI Agent 实现的最新进展，重点关注其实现需要增强推理、规划和工具执行能力的复杂目标的能力...</li><li><a href="https://arxiv.org/abs/2404.05206">SoundingActions: Learning How Actions Sound from Narrated Egocentric Videos</a>: 我们提出了一种新型自监督嵌入，用于从叙述性的野外第一视角视频中学习动作的声音。现有方法依赖于具有已知视听对应关系的精选数据...</li><li><a href="https://docs.google.com/presentation/d/1x3MhmPBIE8AZA3OxvchxxaNoWrrb_wIK50-e1dAjsTo/edit#slide=id.g2eb508a56a1_0_53">ColBERT v2 - Latent Space Paper Club</a>: ColBERT v2 Latent Space Paper Club 2024-07-10</li><li><a href="https://aisnakeoil.com/p/new-paper-ai-agents-that-matter?utm_source=ainews&utm_medium=email&utm_campaign=ainews-not-much-happened-today-1036">New paper: AI agents that matter</a>: 重新思考 AI Agent 的基准测试与评估</li><li><a href="https://buttondown.email/ainews/archive/ainews-is-this-openq/">[AINews] Is this... OpenQ*?</a>: MCTS is all you need. AI News 2024/6/14-6/17。我们为您检查了 7 个 subreddit、384 个 Twitter 和 30 个 Discord（414 个频道，5506 条消息）....
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1260707209180872804)** (1 条消息): 

> - `Perplexity 与 AWS 的合作`
> - `在 AWS Marketplace 上发布 Perplexity Enterprise Pro`
> - `Amazon Bedrock 对 Perplexity 的益处` 


- **Perplexity 与 AWS 合作推出 Enterprise Pro**：Perplexity 宣布与 **Amazon Web Services** 达成战略合作，通过 [AWS marketplace](https://t.co/t3xBQlyw0c) 向所有 AWS 客户提供 **Perplexity Enterprise Pro**。
   - 此次合作包括联合活动、共同销售参与以及联合营销工作，并利用 **Amazon Bedrock** 提供 generative AI 能力。
- **Perplexity Enterprise Pro 的新里程碑**：与 AWS 的合作伙伴关系标志着 Perplexity 使命中的一个重要里程碑，即在不牺牲**安全性和控制力**的前提下，通过 AI 驱动的研究工具为组织赋能，提高**效率和生产力**。
   - 作为此次新合作的一部分，_Perplexity Enterprise Pro_ 将使企业能够通过 AI 驱动的搜索和分析，改变团队获取和利用信息的方式。



**提到的链接**：<a href="https://t.co/t3xBQlyw0c">Perplexity 与 Amazon Web Services 合作推出 Enterprise Pro</a>：我们正在迈出重要一步，让组织能够利用 AI 驱动的工具来提高效率和生产力。

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1260673400138109120)** (110 条消息🔥🔥): 

> - `Perplexity AI 的功能与限制`
> - `药房和药物成本查询`
> - `Perplexity Pro 与教育计划`
> - `使用 Perplexity AI 进行编程`
> - `Claude LLM 模型更新` 


- **Perplexity AI 管理消息长度，而非每日限制**：一位成员指出，如果消息过长，**Perplexity AI** 会截断消息，但它没有类似 GPT 的每日限制。
   - 另一位成员澄清说，**GPT** 会为长回复显示“继续”按钮，而 Perplexity 则没有。
- **药剂师寻求全面的药品价格搜索**：一位药剂师讨论了在 Perplexity 的药物价格搜索结果中找不到 **costplusdrugs** 的问题。
   - 另一位成员建议，Perplexity 使用自己的网站索引器，这可能会给出与 Google 不同的排名。
- **Perplexity Pro 的教育折扣和促销代码**：成员们讨论了使用**促销代码**订阅 Perplexity Pro，并提到了一个折扣教育计划。
- **使用 Perplexity 进行 MATLAB 编程的挑战**：一位用户描述了在 Perplexity 的协助下编写 **MATLAB** 代码时，难以在不同提示词之间保持格式的问题。
   - 其他人建议更清晰、一致地构建查询以获得更好的结果，并利用 Stack Overflow 等其他编程资源。
- **Claude 模型从 labs 中移除并更新计划**：Claude 模型已从 Perplexity Labs 中移除并移至正式环境，供 **Pro** 用户使用。



**提到的链接**：<a href="https://www.perplexity.ai/hub/blog/bringing-perplexity-to-education-and-not-for-profits">将 Perplexity 引入教育和非营利组织 </a>：Perplexity Enterprise Pro，为慈善组织、公职人员和学校提供特殊费率 

  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1260688262377242685)** (6 messages): 

> - `人口统计信息与色情内容使用`
> - `家庭概念`
> - `预防垃圾骚扰电话`
> - `YouTube 踩（Dislike）信息`
> - `Docker Compose 依赖项` 


- **关于人口统计信息与色情内容使用的辩论**：一位用户询问保守派和自由派的人口统计信息与其色情内容使用之间是否存在相关性，并暗示保守派可能由于文化压抑而使用更多。
   - 关于保守派是否比自由派使用更多色情内容，目前还没有明确的共识，尽管一些研究暗示情况可能确实如此。
- **预防垃圾骚扰电话的步骤**：为了预防垃圾骚扰电话，建议用户在国家“拒绝来电”名单上注册，例如加拿大的 **National DNCL** 和美国的 **FTC's Do Not Call Registry**。
   - 还建议启用 iOS 中的“静音未知来电”（Silence Unknown Callers）等功能，以有效减少垃圾电话。
- **创作者与 YouTube 视频踩（Downvotes）**：在 YouTube 做出更改之前，创作者可以在 YouTube Studio 中看到点赞和踩的总数，但无法识别具体是谁踩了他们的视频。
   - 该功能已经实施了更改，但未说明这些修改发生的具体细节。
- **在 Docker Compose 中设置依赖项**：Docker Compose 中的 `depends_on` 指令仅适用于在同一个 Compose 文件中定义的服务器，不适用于跨不同 Compose 文件的容器。
   - 为了处理独立 Compose 文件之间的依赖关系，建议使用带有健康检查（health checks）的外部网络（external networks）或实现等待脚本（wait script）等选项。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.perplexity.ai/search/how-can-i-set-a-docker-compose-hMPnt7lHTI.sTp5Paic_gQ">如何设置 Docker Compose 文件，使受其控制的容器在运行前等待...</a>：要使一个 Docker Compose 文件中的容器在运行前等待另一个 Compose 文件中的容器启动，你有几种选择：1. 使用外部...</li><li><a href="https://www.perplexity.ai/search/preventing-spam-phone-calls-sDolaXnTRbSIU5oFtY6Z7g">预防垃圾骚扰电话</a>：要预防垃圾骚扰电话，你可以采取以下几个步骤：国家拒绝来电名单 (DNCL)：在加拿大，你可以注册你的住宅、无线、传真或...</li><li><a href="https://www.perplexity.ai/search/what-information-did-the-creat-wCb2d6g4QY2KfEa5mXk8TQ">在 YouTube 做出更改之前，当用户踩视频时，视频创作者实际上收到了什么信息...</a>：根据搜索结果，以下是我们可以确定的关于在 YouTube 做出更改之前，当视频被踩时创作者收到的信息：1....</li><li><a href="https://www.perplexity.ai/search/is-there-a-way-to-correlate-de-.JRxq0UUTbmNLT9o7r0xCw">是否有办法将保守派和自由派的人口统计信息与...联系起来</a>：根据现有的研究和数据，关于保守派是否比自由派使用更多色情内容，目前还没有明确的共识，尽管一些研究已经...</li><li><a href="https://www.perplexity.ai/search/explain-the-concept-of-family-Gdd0tkfWRBKwHrZJDiH3og">Perplexity</a>：Perplexity 是一个免费的 AI 驱动的问答引擎，可为任何问题提供准确、可靠且实时的答案。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1260701982835085495)** (3 messages): 

> - `Perplexity Discord 集成`
> - `在线模型的延迟问题`
> - `账户余额查询` 


- **Perplexity Discord 集成咨询**：一位用户询问是否有人成功将 **Perplexity** 集成到 Discord 服务器中。
   - 聊天中没有针对此问题提供后续行动或回复。
- **注意到在线模型的延迟激增**：一位用户报告称，从 6 月 26 日开始，**llama-3-sonar-large-32k-online** 模型的延迟显著增加。
   - 他们询问这是否是一个已知问题，以及是否有任何解决性能下降的计划。
- **需要澄清账户余额**：一位用户请求账户详情以核实余额问题是否已解决，并标记了另一位成员进行后续跟进。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1260673363060461779)** (116 messages🔥🔥): 

> - `Image Enhancements` (图像增强)
> - `Character Loras` (角色 Loras)
> - `Comfy-portable`
> - `Stable Diffusion issues` (Stable Diffusion 问题)
> - `CivitAI banning SD3 content` (CivitAI 禁止 SD3 内容)


- **极小缩放倍率下的图像增强**：一位用户分享了 Stable Diffusion 在极小缩放因子下改善皮肤纹理和面部等图像细节的能力，令其感到惊讶。
   - *midare* 建议大多数用户通常使用 2x 缩放进行增强。
- **在 Pony Checkpoints 上使用角色 Loras 的挑战**：关于为 **Pony Checkpoints** 训练 **Loras** 的讨论指出，角色 Loras 在普通的 **SDXL Checkpoints** 上通常看起来比在 Pony Checkpoints 上更写实，后者中的角色辨识度较低。
   - *crystalwizard* 建议咨询专门从事 Pony 训练的专家。
- **CivitAI 维持对 Stable Diffusion 3 (SD3) 的禁令**：尽管最近更新了许可证，**CivitAI** 仍继续禁止 **SD3** 内容，这表明这是一个与其在 **Open Model Initiative (OMI)** 投资相关的战略决策。
   - 有推测认为 **CivitAI** 的未来可能会变得与 **Stable Diffusion** 类似，带有潜在的商业限制。
- **排除 Comfy-portable 错误**：几位用户讨论了修复 **Comfy-portable** 错误的困难，并询问这些问题是否得到社区支持。
- **Stable Diffusion 性能与设置建议**：一位用户描述了在 **RTX 2060 Super** 上使用 **Automatic1111** 时遇到的持续问题，包括黑屏以及在使用 **--xformers** 等特定命令后难以生成图像。
   - *cs1o* 建议使用简单的启动参数，如 **--xformers --medvram --no-half-vae** 以避免这些问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=vCCVxGtCyho&">ComfyUI InsightFace Windows 快速安装 (2024) | 解决 IPADAPTERS / ROOP 的错误</a>：ComfyUI: https://github.com/comfyanonymous/ComfyUIInsightFace Wheels: https://github.com/Gourieff/Assets/tree/main/InsightfaceCommands: .\python_embeded\pyth...</li><li><a href="https://github.com/InServiceOfX/InServiceOfX/blob/master/PythonLibraries/HuggingFace/MoreDiffusers/morediffusers/Applications/terminal_only_finite_loop_main_with_loras.py">InServiceOfX/PythonLibraries/HuggingFace/MoreDiffusers/morediffusers/Applications/terminal_only_finite_loop_main_with_loras.py at master · InServiceOfX/InServiceOfX</a>：深度学习的 Monorepo（单一代码库）。 - InServiceOfX/InServiceOfX
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1260968179602292796)** (2 messages): 

> - `mdBook advantages` (mdBook 的优势)
> - `ModularBot level advancements` (ModularBot 等级晋升)


- **为什么 mdBook 是更好的选择**：推荐使用 **mdBook**，因为它支持下载为 PDF 以供离线阅读，并且具有使用特定 Python 库包含大纲的功能。
- **用户等级晋升**：一位用户因晋升至 1 级而受到 **ModularBot** 的祝贺。


  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1260734319031222304)** (2 messages): 

> - `Modular Twitter update` (Modular Twitter 更新)
> - `Modular status announcement` (Modular 状态公告)


- **Modular 发布 Twitter 更新**：Modular 通过其 [Twitter](https://twitter.com/Modular/status/1811172833848082503) 账号分享了新帖子。
- **Modular 在 Twitter 上宣布状态更新**：Modular 在其 [Twitter](https://twitter.com/Modular/status/1811453927034081559) 账号上宣布了另一个状态更新。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1260680902221234346)** (44 条消息🔥): 

> - `Setitem 语法问题`
> - `NuMojo 与 nightly 版本的兼容性`
> - `Mojo 开源时间线`
> - `Mojo 中的 Kernel bypass networking`
> - `mlir_ops 中的动态操作数` 


- **Mojo 的 Setitem 语法问题**：一位成员在 Mojo 中使用 `A[0] = 1` 时遇到错误，但使用 `A.__setitem__(0, 1)` 时却不会出错。该问题似乎与 `__getitem__` 和 `__setitem__` 之间的类型检查有关，导致他们提交了 [issue #3212](https://github.com/modularml/mojo/issues/3212)。
- **NuMojo 与 Mojo nightly 的兼容性问题**：更新到最新的 Mojo nightly 导致了与 NuMojo 的不兼容问题，特别是 DTypePointer API 的更改。尽管最近进行了更新，但在简化示例中 nightly 版本仍然显示相同的错误。
- **Mojo 最终将会开源**：Chris Lattner 向用户保证 Mojo 未来将会开源，并将其与 LLVM 和 Swift 进行了比较，后者也花费了数年时间才开源。他指出，暂时的封闭阶段是为了在没有早期大规模贡献负担的情况下完善项目。
- **关注 Mojo 中的 Kernel bypass networking**：Darkmatter__ 表示希望 Mojo 能够避免其他语言在 Kernel bypass networking 方面犯过的错误。确保网络组件的整洁集成似乎是用户关注的一个重要问题。
- **mlir_ops 中的动态操作数**：有人提出了关于在 MLIR 操作中使用动态操作数的问题，特别是对于索引加法。该咨询旨在寻找一种动态向 MLIR 操作传递属性的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/3212">[BUG] `A[0] = 1` 无法工作而 `A.__setitem__(0, 1)` 可以工作 · Issue #3212 · modularml/mojo</a>：Bug 描述：当我使用 A[0]=1 来设置条目时，得到以下错误：error: expression must be mutable in assignment A[0] = 1 ~^~~ mojo: error: failed to parse the provided Mojo source module 但是...</li><li><a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/experimental/numojo/core/ndarray.mojo">NuMojo/numojo/core/ndarray.mojo at experimental · Mojo-Numerics-and-Algorithms-group/NuMojo</a>：NuMojo 是一个用于 Mojo 🔥 数值计算的库，类似于 Python 中的 NumPy。 - Mojo-Numerics-and-Algorithms-group/NuMojo
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1260727960760221857)** (1 条消息): 

> - `频道名称变更`
> - `GPU 编程频道` 


- **MAX 讨论频道名称已更新**：频道名称已更新；<#1212827597323509870> 现在专门用于讨论所有与 MAX 相关的内容，包括 serving、engine 和 pipelines。
   - 通过这些更改，成员们现在可以更轻松地区分讨论内容。
- **新的专用 GPU 编程频道**：新频道 <#1212827673257316453> 现在专门用于发布有关 GPU 编程的后续信息和讨论。
   - 这种划分旨在促进在 GPU 主题上进行专注且高效的对话。


  

---


### **Modular (Mojo 🔥) ▷ #[max-gpu](https://discord.com/channels/1087530497313357884/1212827673257316453/1260727928074010715)** (1 条消息): 

> - `MAX 相关讨论`
> - `专用 GPU 编程信息` 


- **MAX 相关频道更新**：频道名称略有变动，<#1212827597323509870> 现在专门用于讨论所有与 MAX 相关的内容，包括 serving、engine、pipelines 等。
- **专用 GPU 编程频道**：<#1212827673257316453> 已被指定为发布有关 GPU 编程的后续信息和讨论的专用频道。

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1260676298305699840)** (51 条消息🔥): 

> - `新的 Mojo 编译器 Nightly 版本发布`
> - `ArrowIntVector 条件一致性 (Conditional Conformance)`
> - `Mojo 构建问题`
> - `Mojo 中的 Variant 类型` 


- **Mojo 编译器发布多个 Nightly 版本**：Nightly 版 Mojo 编译器已多次更新，版本号包括 `2024.7.1022`、`2024.7.1105` 和 `2024.7.1114`。这些发布包含了为 `List` 实现相等性比较、在 `sort.mojo` 中使用 `UnsafePointer` 以及移除 `memcpy` 的 `LegacyPointer` 版本等更新，并附带了相关的 [changelogs](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 和 [raw diffs](https://github.com/modularml/mojo/compare/)。
- **ArrowIntVector 条件一致性查询**：一位用户分享了关于 `ArrowIntVector` 在 Mojo 语言中对 `StringGetter` 和 `IntGetter` Traits 的条件一致性（conditional conformance）的代码，以获取关于其正确性的反馈，并强调了最新构建中的问题。
   - *另一位用户建议确保 `ArrowIntVector` 符合 `IntGetter` 并使用参数化 Traits*，同时排查与指针类型相关的构建问题。
- **解决 Mojo 构建问题和缓存**：用户遇到了 Mojo 的构建问题，特别是关于 `ArrowIntVector` 示例中的指针错误。建议包括清理存储在 `.modular/.mojo_cache` 中的编译缓存并确保 Traits 的一致性。
- **利用 Variant 类型实现条件一致性**：成员们讨论了在 Mojo 中使用 `Variant` 类型来处理运行时变体类型，这在处理固定集合中的多样化数据类型方面看起来很有前景。示例包括一个 [JSON parser](https://github.com/ZacHooper/mojo-json)，展示了 Mojo 中 `Variant` 类型的实际用法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/utils/variant/Variant">Variant | Modular Docs</a>：一种运行时变体类型。</li><li><a href="https://github.com/jdupl123/arrow.mojo/blob/e16bc582cb4b7d7ced31d6260c6d7458ae780bef/arrow/physical_layout/arrow.mojo#L54">arrow.mojo/arrow/physical_layout/arrow.mojo at e16bc582cb4b7d7ced31d6260c6d7458ae780bef · jdupl123/arrow.mojo</a>：Mojo🔥 中的 Apache Arrow。通过在 GitHub 上创建账号为 jdupl123/arrow.mojo 的开发做贡献。</li><li><a href="https://github.com/ZacHooper/mojo-json">GitHub - ZacHooper/mojo-json: Json Parser in Mojo</a>：Mojo 编写的 JSON 解析器。通过在 GitHub 上创建账号为 ZacHooper/mojo-json 的开发做贡献。</li><li><a href="https://github.com/modularml/mojo/blob/ce75e94d8c2295679966d810e2aa4474f8ab433f/docs/changelog.md?plain=1#L77">mojo/docs/changelog.md at ce75e94d8c2295679966d810e2aa4474f8ab433f · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2751">[BUG] Flaky segfault during `mojo build` with `-D MOJO_ENABLE_ASSERTIONS` · Issue #2751 · modularml/mojo</a>：Bug 描述：此 Bug 是 #2687 的阻碍因素。在带 -D MOJO_ENABLE_ASSERTIONS 编译 test_string.mojo 时，我注意到出现了一些不稳定的段错误。如你所见，这在 CI 中是可以复现的...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1261002293655109682)** (5 条消息): 

> - `Mojo 编译器性能`
> - `AVX2 和 AVX-512 利用率`
> - `手写 Kernel vs 编译器`
> - `汇编代码审查` 


- **Mojo 编译器高效处理 AVX2**：一位成员分享了他们的汇编输出，强调 Mojo 编译器表现出色，能够高效地调度 AVX2 指令。
- **手写 Kernel 的优势**：尽管编译器性能优异，成员们一致认为手写 Kernel 可以通过消除栈分配和直接使用寄存器来进一步优化。
   - *“我很高兴不需要为不同的配置手工制作所有的 Kernel，只需要一个通用的 Kernel 即可。”*
- **关于 AVX-512 能力的讨论**：讨论中提到了使用 AVX-512 加载的好处，尽管其中一位成员的电脑缺乏 AVX-512 功能。


  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1260797198384234588)** (71 条消息🔥🔥): 

> - `LangSmith 成本计算`
> - `语音机器人实现`
> - `Vector Store Retriever 工具`
> - `Chroma DB 初始化`
> - `OpenAI Vector Store` 


- **LangSmith 缺乏对 Google Gemini 模型成本计算的支持**：一位成员指出，尽管 LangSmith 正确添加了 Token 计数，但目前不支持内置成本计算，因此无法显示 Google Gemini 模型的成本。
- **使用 RAG 实现语音机器人**：一位用户分享了识别语音机器人查询意图的代码，将“产品”和“订单详情”查询路由到相应的 VDB，并对其他查询使用 FAQ 数据。
- **将自定义 API 调用添加为工具**：提供了关于如何使用 JavaScript 编写自定义工具以调用后端 API 的[说明](https://js.langchain.com/v0.2/docs/how_to/custom_tools)，该工具使用了 LangChain 的 `DynamicStructuredTool` 类。
   - 说明中包含了一个在自定义工具中使用 `axios` 或 `fetch` 发起 HTTP 请求的示例。
- **加速 Chroma VectorStore 初始化**：减少 Chroma VectorStore 初始化时间的建议包括将 Vector Store 持久化到磁盘、使用更小的 Embedding 模型以及尽可能利用 GPU，参考了 [GitHub Issue #2326](https://github.com/langchain-ai/langchain/issues/2326)。
- **将 OpenAI Vector Store 用作 Retriever**：要将 OpenAI Vector Store 用作 Retriever，可以实例化一个带有 Embeddings 的 Vector Store，然后按照 [LangChain 文档中所述](https://python.langchain.com/v0.2/docs/how_to/vectorstore_retriever/#creating-a-retriever-from-a-vectorstore)使用 `.as_retriever()` 方法创建一个 Retriever。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/aesthetic-numbers-time-clock-counting-gif-16982789">Aesthetic Numbers GIF - Aesthetic Numbers Time - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/vectorstore_retriever/#creating-a-retriever-from-a-vectorstore>)">如何将 Vector Store 用作 Retriever | 🦜️🔗 LangChain</a>：Vector Store Retriever 是一个使用 Vector Store 来检索文档的 Retriever。它是对 Vector Store 类的轻量级封装，使其符合 Retriever 接口。</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/lantern/#using-a-vectorstore-as-a-retriever>)">Lantern | 🦜️🔗 LangChain</a>：Lantern 是一个针对 Postgres 的开源向量相似度搜索工具</li><li><a href="https://github.com/langchain-ai/langchain/issues/2326>))">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/5046>)).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2144>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/2491>).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/7175>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/23797>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 做出贡献。</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/agents/#retriever>)">构建 Agent | 🦜️🔗 Langchain</a>：本指南假设你熟悉以下概念：</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/agent_executor/#retriever>)">如何使用旧版 LangChain Agents (AgentExecutor) | 🦜️🔗 Langchain</a>：本指南假设你熟悉以下概念：
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1260715011613917336)** (14 条消息🔥): 

> - `Asyncio.run() RuntimeError`
> - `uvicorn.run() issues`
> - `Stream content type error`
> - `LangServe replacement`
> - `LangGraph Cloud` 


- **Asyncio.run() RuntimeError 详解**：一位成员在从正在运行的 event loop 中运行 `asyncio.run()` 时遇到了 **RuntimeError**：*asyncio.run() cannot be called from a running event loop*。
   - 目前尚未提供解决此问题的方案，该问题仍有待进一步讨论。
- **Chat 中的 Stream content type 错误**：一位成员在使用 `playground_type="chat"` 时遇到了非预期的 content type 错误：预期为 *text/event-stream*，但在使用时却收到了 *application/json*。
   - 该错误似乎与 chat history 相关，但未提及具体的解决方案。
- **LangServe 被 LangGraph Cloud 取代**：据 Harrison 确认，LS 门户中的 **LangServe** 已被 **LangGraph Cloud** 取代。
   - 尽管 OSS LangServe 将继续存在，但托管选项现在是 **LangGraph Cloud**，一些成员更倾向于将其用于 Agent 功能。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1261021295643070554)** (1 条消息): 

> - `Magnum 72B`
> - `Hermes 2 Theta`
> - `Model Deprecations`（模型弃用）
> - `Router Resilience Update`（路由弹性更新）


- **Magnum 72B 旨在实现 Claude 3 级别的文本质量**：[Alpindale's Magnum 72B](https://openrouter.ai/models/alpindale/magnum-72b) 旨在达到 Claude 3 系列模型的文本质量，该模型源自 Qwen2 72B，并使用 5500 万 token 的 RP 数据进行训练。
- **Hermes 2 Theta 将 Llama 3 与元认知能力相结合**：[Nousresearch's Hermes-2 Theta](https://openrouter.ai/models/nousresearch/hermes-2-theta-llama-3-8b) 是一款结合了 Llama 3 和 Hermes 2 Pro 的实验性模型，以函数调用（function calls）、JSON 输出和**元认知能力**著称。
- **旧模型面临弃用**：由于使用率较低，[intel/neural-chat-7b](https://openrouter.ai/models/intel/neural-chat-7b) 和 [koboldai/psyfighter-13b-2](https://openrouter.ai/models/koboldai/psyfighter-13b-2) 已被列入弃用名单，并将于 7 月 25 日起在 API 中返回 404 错误。
- **路由通过回退功能增强弹性**：一项新的路由功能将默认使用回退（fallback）提供商，除非指定了 `allow_fallbacks: false`，以确保在主要提供商发生故障时保持弹性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/alpindale/magnum-72b>)">Magnum 72B by alpindale</a>：出自 [Goliath](https://openrouter.ai/models/alpindale/goliath-120b) 创作者之手，Magnum 72B 是新模型系列中的首款，旨在实现 Claude 3 系列模型的文本质量，尤其...</li><li><a href="https://openrouter.ai/models/alpindale/goliath-120b>),">Goliath 120B by alpindale</a>：通过将两个微调后的 Llama 70B 模型合并为一个 120B 模型而创建的大型 LLM。结合了 Xwin 和 Euryale。感谢 [@chargoddard](https://huggingface.co/chargoddard) 开发了该框架...</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-72b-instruct>)">Qwen 2 72B Instruct by qwen</a>：Qwen2 72B 是一款基于 Transformer 的模型，在语言理解、多语言能力、编程、数学和推理方面表现出色。它具有 SwiGLU 激活、Attention QKV 偏置和...</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-2-theta-llama-3-8b>)**:">Nous: Hermes 2 Theta 8B by nousresearch</a>：一款基于 Llama 3 的实验性合并模型，展现出非常独特的写作风格。它结合了 [Meta's Llama 3 8B](https://openrouter.ai/models/meta-llama/llama-3-8b-in... 的优点</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-8b-instruct>)">Meta: Llama 3 8B (Base) by meta-llama</a>：Meta 最新的模型类别 (Llama 3)，推出了多种尺寸和版本。这是基础 8B 预训练版本。与领先的闭源模型相比，它展现了强大的性能...</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-2-pro-llama-3-8b>).">NousResearch: Hermes 2 Pro - Llama-3 8B by nousresearch</a>：Hermes 2 Pro 是 Nous Hermes 2 的升级重训版本，包含 OpenHermes 2.5 数据集的更新清洗版，以及新引入的函数调用（Function Calling）和 JSON 模式...</li><li><a href="https://openrouter.ai/models/intel/neural-chat-7b>)">Neural Chat 7B v3.1 by intel</a>：基于 [mistralai/Mistral-7B-v0.1](/models/mistralai/mistral-7b-instruct-v0.1) 在开源数据集 [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) 上微调的模型...</li><li><a href="https://openrouter.ai/models/koboldai/psyfighter-13b-2>).">Psyfighter v2 13B by koboldai</a>：[Psyfighter](/models/jebcarter/psyfighter-13b) 的 v2 版本 —— 由 KoboldAI 社区成员 Jeb Carter 和 TwistedShadows 创建的合并模型，得益于 KoboldAI 合并请求服务...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1260677918808342580)** (62 条消息🔥🔥): 

> - `Noromaid 模型移除`
> - `LLaMA-Guard 的优势`
> - `VoiceFlow 与 OpenRouter 的集成`
> - `维护对话上下文`
> - `OpenRouter 与 Assistant API` 


- **Noromaid 模型因成本原因逐步淘汰**：成员们讨论了由于高成本和低使用率而移除 **noromaid 模型** 的问题。
   - 一位成员指出：*“我真的很喜欢 noromaid 模型，只是平时一直用的话实在太贵了。”*
- **LLaMA-Guard 作为审核模型的替代方案**：成员们考虑使用 **LLaMA-Guard** 作为 Noromaid 的审核替代方案，并指出可以通过 OR 传递过滤参数。
   - 一位成员分享了 [LLaMA-Guard 的链接](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B)并提到：*“而且它足够小，可以在本地运行。”*
- **OpenRouter 与 VoiceFlow 集成的挑战**：讨论了 **VoiceFlow** 与 **OpenRouter** 的集成，用于通过 OR 管理对话上下文，并引发了对无状态请求的担忧。
   - 一个建议是使用 [VoiceFlow 上的对话记忆](https://learn.voiceflow.com/hc/en-us/articles/15049513713037-Conversation-Memory)来维护聊天历史。
- **维护对话上下文的重要性**：用户讨论了使用 OpenRouter 等 API 以及 LangChain 等框架来维护对话上下文的策略。
   - *“VoiceFlow 将会（或应该）有一种维护对话历史的方法，”* 一位成员指出，强调了上下文持久化的必要性。
- **对 OpenRouter 支持 Assistant API 的兴趣**：讨论了 **OpenRouter** 支持 **Assistants API**（类似于 OpenAI 的设置）的潜在好处。
   - 成员们指出，如果这不是一项如此艰巨的任务，它所能带来的价值将包括嵌入文档和代码解释器（code interpreter）等功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/vercel/ai">GitHub - vercel/ai: Build AI-powered applications with React, Svelte, Vue, and Solid</a>: 使用 React, Svelte, Vue 和 Solid 构建 AI 驱动的应用 - vercel/ai</li><li><a href="https://deepinfra.com/privacy">DeepInfra Privacy Policy</a>: 使用简单的 API 运行顶级 AI 模型，按需付费。低成本、可扩展且生产就绪的基础设施。</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B">meta-llama/Meta-Llama-Guard-2-8B · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1260679576867311706)** (53 messages🔥): 

> - `Decentralized AI`
> - `BOINC`
> - `Sharded Computing`
> - `Parallel GPU Usage`
> - `OpenAI's New Models` 


- **去中心化 AI 计算获得关注**：成员们讨论了创建一个去中心化 Mesh 网络的可能性，用户可以贡献其计算能力，这得益于带宽和压缩技术的进步。
   - 提到了 [BOINC](https://boinc.berkeley.edu) 和 Gridcoin 等加密项目，作为利用代币激励此类去中心化网络的案例。
- **用于 AI 的分片计算 (Sharded Computing)**：有人提议建立一个可以使用各种 VRAM 大小的分片计算平台，并用代币奖励贡献计算资源的用户。
   - 提到了利用去中心化计算优化 CMOS 芯片配置，参考了已退役的 DHEP@home BOINC 项目。
- **并行 GPU 查询**：有人询问了在并行 GPU 上运行 GGUF 平台的可行性。
   - 回复指出，鉴于其作为张量管理平台的性质，这确实是可能的。
- **OpenAI 新模型能力揭晓**：一份报告详细说明了 OpenAI 正在测试其 GPT-4 模型的新能力，展示了提升至类人推理的技能，并正通过一个分级系统向 AGI 迈进。
   - 该公司解释说，第二级涉及能够解决博士级问题的“推理者”（Reasoners），未来的级别将向能够采取自主行动的“智能体”（Agents）发展。
- **Claude AI 性能问题**：用户报告在约 10 次回复后 Claude AI 聊天出现严重延迟，导致聊天功能几乎无法使用。
   - 推测指向可能的内存泄漏或后端问题，这与 GPT-4 模型更稳定的体验形成对比。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/kimmonismus/status/1811498151964033084?s=46">来自 Chubby♨️ (@kimmonismus) 的推文</a>: OpenAI 正在展示新技能和可能的新模型。来自 @business 的新帖子报道了 OpenAI 的分级系统。还展示了一个具有新能力的 ChatGPT 版本。从措辞来看...</li><li><a href="https://www.bloomberg.com/news/articles/2024-07-11/openai-sets-levels-to-track-progress-toward-superintelligent-ai">Bloomberg - Are you a robot?</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1260953718254473373)** (3 messages): 

> - `Prompt library rename`
> - `Reminder about different channels` 


- **Prompt library 重命名**：**prompt library** 已重命名，可以在新频道 <#1019652163640762428> 下找到。
   - 一位成员为寻找该频道的用户澄清了位置。
- **频道混淆提醒**：发布了一个提醒，指出[此频道](https://discord.com/channels/974519864045756446/1019652163640762428)与重命名的 prompt library 频道不是同一个。
   - 该澄清旨在防止成员对频道目的地产生混淆。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1260953718254473373)** (3 messages): 

> - `Prompt Library Rename`
> - `Channel Difference Reminder` 


- **Prompt Library 重命名**：一位成员通知 **prompt library** 已重命名，并指向了新频道 <#1019652163640762428>。
- **澄清频道间的区别**：另一位成员提醒小组，此频道与[另一个频道](https://discord.com/channels/974519864045756446/1019652163640762428)不同。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1260707509555957875)** (3 条消息): 

> - `llama-agents launch`
> - `NebulaGraph integration`
> - `LlamaTrace collaboration` 


- **llama-agents 框架发布，星标突破 1100**：上周，新的多 Agent 部署框架 **llama-agents** 正式发布，并获得了热烈反响，其 [repo](https://twitter.com/llama_index/status/1811147950388916420) 星标已达到 **1100 stars**。
   - *MervinPraison* 提供了一个 [YouTube 演示视频](https://t.co/8uetfVqHf9)，涵盖了 llama-agents 的用法和特性。
- **NebulaGraph 与 LlamaIndex 集成**：查看 **NebulaGraph** 与 **LlamaIndex** 的最新集成，通过属性图索引（property graph index）实现强大的 **GraphRAG** 能力。
   - 正如其[公告](https://twitter.com/llama_index/status/1811190191597773282)中所述，该集成支持复杂的提取器（extractors）和可自定义的提取过程。
- **LlamaTrace 与 Arize AI 合作提升 LLM 可观测性**：宣布推出 **LlamaTrace**，这是与 **Arize AI** 合作的项目，旨在为 LLM 应用引入先进的追踪、可观测性和评估工具。
   - 该计划进一步丰富了 LLM 工具链，并在其[推广活动](https://twitter.com/llama_index/status/1811462543535464796)中得到了重点展示。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1260820760746786826)** (32 条消息🔥): 

> - `Llamaparse and OCR`
> - `Setting language for prompt templates`
> - `Accessing additional_kwargs in CompletionResponse`
> - `Voice chat with GPT models`
> - `ReACT agent variable mapping issues` 


- **Llamaparse 处理预现有的 OCR**：用户正在讨论 **Llamaparse** 是移除 PDF 中现有的 OCR 还是对其进行增强，目前对该过程存在一些困惑，尚无明确结论。
- **特定语言的 Prompt templates**：一名成员询问如何设置特定语言的 prompt templates，回复建议这取决于 LLM 的能力，并参考了 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern)。
- **在 RAG pipeline 中提取 additional_kwargs 属性**：一名成员询问如何在 RAG pipeline 中访问 **additional_kwargs**，建议使用 retrievers 或挂载到底层 LLM 事件中，并分享了[大量示例](https://docs.llamaindex.ai/en/stable/examples/instrumentation/instrumentation_observability_rundown)。
- **直接与 GPT 模型进行语音聊天目前尚不可行**：已确认目前无法在不进行语音转文本的情况下直接与 **GPT 模型** 进行语音聊天；建议将 **TTS 和 Whisper** 作为转换的临时解决方案。
- **ReACT agent 变量映射导致错误**：一名成员报告了在 **ReACT agent** 中设置变量映射时出现 **KeyError** 问题，回复建议在运行前检查变量定义及其包含情况。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern/?h=prompt+upda#accessing-prompts">Usage pattern - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern/?h=prompt+upda#updating-prompts">Usage pattern - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/">Query Pipeline Chat Engine - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/instrumentation/instrumentation_observability_rundown/?h=instrumentation">Built-In Observability Instrumentation - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/react_agent/#react-agent-a-simple-intro-with-calculator-tools>).">ReAct Agent - A Simple Intro with Calculator Tools - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/react_agent/#react-agent-a-simple-intro-with-calculator-tools>)">ReAct Agent - A Simple Intro with Calculator Tools - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1260789002420752395)** (29 messages🔥): 

> - `Experimental Architectures`
> - `Sign Gradient`
> - `Residual Connections`
> - `Memory Efficiency in Training` 


- **探索实验性架构**：一位成员分享了他们痴迷于在新型架构上运行实验的经历，尽管这些架构没有产生显著改进且非常耗费计算资源，可能需要未来进行大量的消融测试（ablation testing）。
   - 他们表示喜欢观察 Loss 曲线的微小改进，并指出更深的配置似乎效果较差，但仍热衷于通过持续的故障排除来挖掘潜在收益。
- **Sign Gradient 建议**：一位成员建议在实验架构中使用 Sign Gradient，另一位成员对此很感兴趣并渴望进一步探索。
- **在低参数视觉模型中追求 SOTA**：该成员使用 250k 参数的模型在 CIFAR-100 上达到了 50% 的准确率，接近 2022 年一篇关于[低参数视觉模型论文](https://arxiv.org/abs/2210.14151)中报道的约 70% 的准确率（当前的 SOTA）。
   - 他们观察到模型的性能对 Block 的数量不敏感，但与总参数量相关，增加深度往往会产生负面影响。
- **残差连接问题**：一位成员注意到其架构的 Residual Connections 存在潜在问题，并计划尝试不同的门控机制（gating mechanisms）。
- **内存效率问题**：据报告，该实验架构的内存效率极低，在 128 batch size 和仅 250k 参数的情况下，训练 CIFAR-100 使用了 19 GB 内存。
   - 优化尝试包括实验使用一个大型 MLP 重复多次，而不是每个 Block 使用多个较小的 MLP。


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1260697465665884172)** (11 messages🔥): 

> - `Diffusion Models`
> - `Local AI Projects`
> - `DoLa Decoding Strategy`
> - `Hugging Face Datasets`
> - `LLM Hallucinations` 


- **Diffusion Models 中的边缘分布**：一位成员对论文 *[FAST SAMPLING OF DIFFUSION MODELS WITH EXPONENTIAL INTEGRATOR](https://arxiv.org/abs/2204.13902)* 中术语 *marginal distributions as p̂∗_t* 感到困惑，并寻求对其含义的澄清。
- **介绍 'RAGAgent' 项目**：一位成员分享了他们新的 Python 项目，一个名为 *[RAGAgent](https://github.com/MikeyBeez/RAGAgent)* 的全本地 AI 系统。
- **通过对比层进行解码 (DoLa)**：讨论了论文 *[Decoding by Contrasting Layers (DoLa)](https://arxiv.org/abs/2309.03883)*，该论文提出了一种通过对比不同层的 logits 来减少 LLM 幻觉的策略。
   - 一个显著的改进是在 Truthful QA 上提升了 **17%**，但它可能会导致推理时间出现明显的减慢。
- **无需 Fine-Tuning 对齐 Llama1**：一位成员指出 *Llama1* 仅经过预训练，而 DoLa 可能是一种无需额外对齐步骤即可对齐模型的方法。
- **Pile-Deduped 数据集中的 EOS Token 处理**：一位成员询问 *[EleutherAI/pile-deduped-pythia-random-sampled](https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-random-sampled)* 数据集是否旨在排除 EOS Token。
   - 他们寻求关于在没有 EOS 或控制 Token 的情况下，如何获得 2048-token 数据块的处理过程的澄清。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2309.03883">DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models</a>: 尽管具有令人印象深刻的能力，但大型语言模型 (LLMs) 容易产生幻觉，即生成偏离预训练期间所见事实的内容。我们提出了一种简单的解码...</li><li><a href="https://arxiv.org/abs/2204.13902">Fast Sampling of Diffusion Models with Exponential Integrator</a>: 过去几年见证了扩散模型 (DMs) 在生成建模任务中产生高保真样本的巨大成功。DM 的一个主要限制是其众所周知的缓慢采样...</li><li><a href="https://github.com/MikeyBeez/RAGAgent">GitHub - MikeyBeez/RAGAgent: REPL that uses RAG as it&#39;s context assembly</a>: 使用 RAG 作为其上下文组装的 REPL。通过在 GitHub 上创建一个账户为 MikeyBeez/RAGAgent 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-random-sampled">EleutherAI/pile-deduped-pythia-random-sampled · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1260780771979956245)** (8 messages🔥): 

> - `Training on the test task`（在测试任务上训练）
> - `BitNet b1.58 LLM`
> - `Emergent behavior in models`（模型中的涌现行为）
> - `Reproduction studies of LLM papers`（LLM 论文的复现研究）
> - `Understanding of large models`（对大模型的理解）


- **在测试任务上训练会干扰评估**：[最近的一篇论文](https://arxiv.org/abs/2407.07890)讨论了**在测试任务上训练**如何扭曲模型评估以及关于涌现能力（emergent capabilities）的说法。
   - 通过在评估前对每个模型进行相同任务相关数据的微调来调整这一因素，结果显示**涌现行为的实例在很大程度上消失了**。
- **BitNet b1.58 LLM 挑战全精度模型**：[BitNet b1.58](https://arxiv.org/abs/2402.17764) 引入了一种 1-bit LLM，其性能与全精度模型相当，同时更具成本效益，**显著降低了延迟、内存占用、吞吐量和能耗**。
   - 讨论仍在继续，询问是否有人测试过它，并引用了该模型的 [Hugging Face 复现版本](https://huggingface.co/1bitLLM/bitnet_b1_58-3B)，显示了类似的结果。
- **大型 LLM 中的涌现行为引发争论**：成员们对当今最大的模型在多大程度上是真正地“理解”并产生新见解，还是仅仅在机械重复训练集中的数据感到好奇。
   - 有人呼吁提供更直观的解释和实证证据，以澄清这些模型的理解深度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.07726">PaliGemma: A versatile 3B VLM for transfer</a>：PaliGemma 是一款开放的视觉语言模型 (VLM)，基于 SigLIP-So400m 视觉编码器和 Gemma-2B 语言模型。它被训练为一个多功能且知识广泛的基础模型...</li><li><a href="https://arxiv.org/abs/2407.07890">Training on the Test Task Confounds Evaluation and Emergence</a>：我们研究了大语言模型评估中的一个基本问题，称之为“在测试任务上训练”。这不同于在测试数据上训练、泄露或数据污染等错误做法...</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>：最近的研究（如 BitNet）正在为 1-bit 大语言模型 (LLM) 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">1bitLLM/bitnet_b1_58-3B · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1260810295115382837)** (3 messages): 

> - `GPT-4o profiles`
> - `Llama3 local standards` 


- **GPT-4o vs Llama3 Local：标准尚在变动中**：一位成员指出，他们使用默认配置文件的 **GPT-4o** 遇到的问题较少，而 **Llama3 local** 出现的问题较多，因为许多关于分隔符和 schema 的标准仍在整合中。
   - *“我想这些问题会随着更新而消失”*，这表明了对未来更新解决问题的预期。
- **General 频道指令**：一位成员请求在 general 频道 (<#1210088092782952498>) 发布。
   - 另一位用户对该请求表示认可：“有道理，谢谢回复。”


  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1260703056560001045)** (15 messages🔥): 

> - `LLM-Service Flag Issue` (LLM-Service 标志问题)
> - `Profile Workaround for 01` (01 的 Profile 变通方案)
> - `Remote Experience Script for 01` (01 的远程体验脚本)
> - `Community Contributions in 01 Development` (01 开发中的社区贡献)
> - `Commercial Applications of 01` (01 的商业应用)


- **01 文档中的 LLM-Service 标志问题**：一名成员指出，01 文档中提到的 **LLM-Service flag** 并不存在，导致了安装问题。
   - 另一名成员提到一个正在进行的 [更新文档的 PR](https://link.to.pr)，并建议使用 profiles 作为临时变通方案。
- **VPS 上 01 的远程体验脚本**：一名成员表示需要一个脚本来允许 **01 在控制台上自动登录**，以获得更好的 VPS 远程体验。
   - 该成员表示正在进行相关研究，并愿意与他人合作进行 **头脑风暴和开发**。
- **社区贡献驱动 01 开发**：一名成员强调 **01 拥有 46 名贡献者**，其中许多人以及来自 Open Interpreter 的 100 多名成员都在该服务器中。
   - 这突显了社区在项目开发中的深度参与。
- **01 的商业应用与阻碍**：一名致力于 **01 远程体验** 的开发者正与 Ben Steinher 就其商业应用进行沟通。
   - 他们认为 **远程登录能力** 是 01 在商业环境中被采用的主要阻碍。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1260759494535811113)** (17 messages🔥): 

> - `Axolotl dataset formats link` (Axolotl 数据集格式链接)
> - `TurBcat 72B usage` (TurBcat 72B 使用)
> - `Testing TurBcat API` (测试 TurBcat API)
> - `WizardLM ArenaLearning` (WizardLM ArenaLearning)
> - `FlashAttention-3 on H100 GPUs` (H100 GPU 上的 FlashAttention-3) 


- **Axolotl 数据集格式链接迁移**：**Axolotl** 数据集格式的链接已移至 [新位置](https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/)。
   - “我们搬到了一个新的组织”以方便更好的访问。
- **TurBcat 72B 可能在 48GB 显存上使用**：在 API 支持下，**TurBcat 72B** 可能在 48GB 显存的系统上运行。
   - 用户 **c.gato** 计划测试 4-bit 量化方案以实现这一目标。
- **测试 elinas 提供的 TurBcat API**：用户 **elinas** 分享了一个用于 **TurBcat 72B** 测试的 API：[TabbyAPI](https://lists-until-showing-allied.trycloudflare.com/)，密钥为 **eb610e28d10c2c468e4f81af9dfc3a48**。
   - 据称该 API 与 **ST Users / OpenAI-API 兼容的前端** 兼容，并使用 **ChatML**。
- **WizardLM 推出 ArenaLearning**：讨论了 **WizardLM** 的 [ArenaLearning 论文](https://www.microsoft.com/en-us/research/uploads/prodnew/2024/07/WizardLM_ArenaLearning.pdf)。
   - 用户将其描述为“一种非常新颖的方法”。
- **FlashAttention-3 提升 H100 GPU 效率**：[FlashAttention-3](https://www.together.ai/blog/flashattention-3) 旨在通过利用现代硬件特性，加速 **H100 GPU** 上的 Attention 计算。
   - 提议的技术包括最小化内存读/写和异步操作，目标是将利用率提升至超过当前最大 FLOPs 的 35%。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.together.ai/blog/flashattention-3">FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision</a>：未找到描述</li><li><a href="https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/">Dataset Formats – Axolotl</a>：未找到描述
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1260951242851614720)** (7 messages): 

> - `Data Curation`
> - `FlashAttention`
> - `LMSYS Chatbot Arena` 


- **FlashAttention 加速了 Transformers**：[FlashAttention](https://pytorch.org/blog/flashattention-3) 开创了一种在 GPU 上加速 Transformers 中 Attention 的方法，显著地将 LLM 的上下文长度从 2-4K 增加到 128K，在 GPT-4 和 Llama 3 等近期模型中甚至达到了 1M。
   - 尽管取得了成功，但 FlashAttention-2 在 H100 GPU 上仅实现了理论最大 FLOPs 利用率的 35%，这表明仍有很大的进一步优化潜力。
- **WizardLM2 依赖于 WizardArena**：[LMSYS Chatbot Arena](https://www.microsoft.com/en-us/research/project/wizardlm-arena-learning/) 是一个通过对话挑战来评估和比较聊天机器人模型，并使用 Elo 评分系统进行排名的平台。
   - 尽管令人兴奋，但 WizardArena 基于人工的评估过程在编排和等待时间方面面临着重大挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/flashattention-3/">FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision</a>: Attention 作为无处不在的 Transformer 架构的核心层，是大型语言模型和长上下文应用的瓶颈。FlashAttention（以及 FlashAttention-2）开创了一种方法...</li><li><a href="https://www.microsoft.com/en-us/research/project/wizardlm-arena-learning/">Arena-Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena - Microsoft Research</a>: 最近的研究表明，使用指令遵循数据对大型语言模型进行后训练取得了巨大的成功。与此同时，人工 Chatbot Arena 已成为最受认可的评估方式之一...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1260818759149420565)** (4 messages): 

> - `Synthetic Instruction Data`
> - `RPO Preference Tuning`
> - `Nemotron`
> - `Instruction Backtranslation`
> - `Reward-Aware Preference Optimization` 


- **合成指令数据中改写的益处**：一位成员询问了在合成指令中改变顺序和句法的**实质性益处**，例如 **“写一篇关于机器学习的文章，要求写三段”** 与 **“就以下主题写一篇三段式的文章：机器学习”**。
   - 他们将这种技术与 *backtranslation*（回译）进行了比较，但指出这与 *instruction backtranslation* 论文有所不同。
- **RPO 偏好微调问题**：一位用户询问了 RPO 偏好微调损失函数中 **η** 的重要性，推测它可能是某种 *reward parameter*（奖励参数）。
   - 他们似乎不确定该参数是否起到了重要作用，并询问了它对优化过程的影响。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1260926566465142784)** (3 messages): 

> - `OpenAI's revenue breakdown`
> - `Subscription model of ChatGPT`
> - `Free usage of GPT-4` 


- **OpenAI 报告了令人印象深刻的营收数据**：[Future Research](https://futuresearch.ai/openai-revenue-report) 对 OpenAI 的营收进行了如下细分：**19 亿美元来自 ChatGPT Plus**，**7.14 亿美元来自 ChatGPT Enterprise**，**5.1 亿美元来自 API**，以及 **2.9 亿美元来自 ChatGPT Team**。
   - 数据包括：ChatGPT Plus 有 **770 万订阅者**（20 美元/月），ChatGPT Enterprise 有 **120 万用户**（50 美元/月），ChatGPT Team 有 **8 万订阅者**（25 美元/月）。
- **在 GPT-4 免费开放的情况下质疑订阅模式**：*既然 GPT-4 现在是免费的，为什么还有这么多人订阅？*
- **关于订阅和 Interconnects 的评论**：*真无语，拿那些钱去订阅两次 Interconnects 吧。*



**提到的链接**：<a href="https://x.com/jvnixon/status/1811278381184672156?s=46">来自 Jeremy Nixon (@JvNixon) 的推文</a>：futureresearch 关于 OpenAI 营收的报告已经发布，显示：ChatGPT Plus 营收 19 亿美元（770 万订阅者，20 美元/月），ChatGPT Enterprise 营收 7.14 亿美元（120 万订阅者，50 美元/月），API 营收 5.1 亿美元，以及 2.9 亿...

  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/)** (1 messages): 

emily_learner: 太棒了。非常感谢。我会去看看。

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1260756536561504306)** (8 条消息🔥): 

> - `GPT Agents`
> - `Command R Plus`
> - `Fine-tuning models` 


- **体验 Command R Plus**：**Mapler** 正在探索 **Command R Plus**，并觉得很有趣。
   - 他们正尝试为了好玩构建一个 Agent。
- **模型 Fine-tuning 的挑战**：**Mapler** 在模型 Fine-tuning 时遇到了问题，表示结果未达预期。
   - 另一位成员指出 *finetuning 就是垃圾进垃圾出 (garbage in garbage out)*，强调了高质量数据集的重要性。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1260819015618662480)** (4 条消息): 

> - `Prompt/Reply Logging Tools`
> - `OpenPipe for OpenAI`
> - `Fireworks.ai Lecture` 


- **PromptLayer 在最新的 Anthropic SDK 上失效**：一位成员表示在使用 **PromptLayer** 进行 Prompt/Reply 日志记录时遇到问题，称其无法与最新的 **Anthropic SDK** 配合使用。
   - 该成员正在寻求自托管替代方案的建议。
- **OpenPipe 仅限于 OpenAI**：一位成员强调 **OpenPipe** 虽然提供 Prompt/Reply 日志记录，但仅限于 **OpenAI**。
   - 他们指出缺乏对 Anthropic 等其他模型的支持。
- **寻找 Fireworks.ai 讲座**：一位成员询问是否有关于或包含来自 **fireworks.ai** 人员的讲座。
   - 关于此话题没有进一步的回复或说明。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1261040706097516618)** (1 条消息): 

> - `Credits Check`
> - `Account ID Query` 


- **检查额度 (Credits)**：一位成员询问如何检查自己是否有额度，并提供了他们的 Account ID 为 **reneesyliu-571636**。
- **Account ID 查询**：用户在寻求帮助的查询中包含了他们的 Account ID：**reneesyliu-571636**。


  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1260704084822855680)** (4 条消息): 

> - `NVDLA vs NV accelerator`
> - `Runtime operations in NV`
> - `Unexpected UOps in simple NN graph` 


- **澄清 NV 加速器功能**：一位成员询问 NV 加速器是否涵盖了所有 NVDLA 功能，或者是否需要编写单独的 NVDLA/DLA 加速器，并引用了 [NVDLA GitHub](https://github.com/nvdla/)。
   - 他们还提到正在研究 cuDLA，但在继续之前需要确认他们的理解是否正确。
- **NV Runtime 绕过用户空间**：另一位成员澄清说，NV Runtime 与 GPU 配合工作，直接与内核交互并绕过用户空间。
- **简单 NN 图中意外的 UOps**：有人分析了一个简单 NN 的 UOps 图，发现了一些意外的乘法和加法，涉及 2.0 和 -0.9999 等常量。
   - 另一位成员解释道：*这些来自线性权重初始化 (linear weight init)*，从而澄清了这一异常现象。



**提到的链接**：<a href="https://github.com/nvdla/">nvdla</a>：NVDLA 开源项目。nvdla 有 17 个可用的代码库。在 GitHub 上关注他们的代码。

  

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1260973365880033371)** (4 条消息): 

> - `US Senate AI hearing`（美国参议院 AI 听证会）
> - `Mozilla blog on privacy law`（Mozilla 关于隐私法的博客）


- **美国参议院强调 AI 对隐私的影响**：在一次[参议院听证会](https://www.commerce.senate.gov/2024/7/the-need-to-protect-americans-privacy-and-the-ai-accelerant)中，**美国参议员 Maria Cantwell** 强调了 AI 在改变数据隐私方面的作用，并倡导制定联邦综合隐私法。
   - 来自 **Mozilla** 的证人 **Udbhav Tiwari** 强调了 AI 在在线监视和消费者画像方面的能力。
- **Mozilla 发文推动联邦隐私法**：**Mozilla** 在其 [distilled 博客](https://blog.mozilla.org/en/mozilla/internet-policy/mozilla-urges-federal-privacy-law-for-ai-development/)上介绍，**Udbhav Tiwari** 在参议院就 AI 领域对联邦隐私法的需求作证。
   - 博客中包含了一张 **Tiwari** 作证的照片，并详细阐述了采取立法行动以保护个人隐私免受 AI 挑战的紧迫性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.mozilla.org/en/mozilla/internet-policy/mozilla-urges-federal-privacy-law-for-ai-development/">Mozilla 前往国会山，呼吁制定联邦隐私法以确保 AI 的负责任发展 | The Mozilla Blog</a>：今天，美国参议院商务、科学和运输委员会主席、参议员 Maria Cantwell（华盛顿州民主党人）召集了一次全体委员会听证会，题为...</li><li><a href="https://www.commerce.senate.gov/2024/7/the-need-to-protect-americans-privacy-and-the-ai-accelerant">保护美国人隐私的需求与 AI 加速器</a>：美国参议院商务、科学和运输委员会主席、参议员 Maria Cantwell（华盛顿州民主党人）将召集一次全体委员会听证会，题为“保护美国人隐私的需求与...”
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1261013886157717588)** (1 条消息): 

> - `Hugging Face Workshop`
> - `Business Impact of LLMs`
> - `Prema Roman`
> - `Patrick Deziel` 


- **参加 7 月 30 日的 Hugging Face 模型工作坊！**：一场名为 **揭秘 Hugging Face 模型及其如何发挥商业影响 (Demystifying Hugging Face Models & How to Leverage Them For Business Impact)** 的独家在线工作坊定于 **2024 年 7 月 30 日东部时间中午 12 点**举行。注册地址请点击[此处](https://events.rotational.io/demystifying-llms)。
- **无法参加？注册以接收材料**：无法参加 **2024 年 7 月 30 日** Hugging Face 工作坊的参与者仍可注册，以便随后接收相关材料。



**提到的链接**：<a href="https://events.rotational.io/demystifying-llms">LLM 中包含什么？揭秘 Hugging Face 模型及其如何发挥商业影响 | 2024 年 7 月 30 日</a>：请于 7 月 30 日通过 Zoom 加入我们。

  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1261023228118437908)** (2 条消息): 

> - `Recsys Community`
> - `Search/IR Community`
> - `Cohere's Sentence Transformer Team`
> - `Vespa`
> - `Elastic` 


- **Recsys 社区规模大于 Search/IR**：一位成员指出，与被描述为“小众”且“不同”的 **search/IR**（搜索/信息检索）社区相比，**Recsys 社区**要庞大且活跃得多。
   - 他们提到 **Cohere** 拥有整个 **sentence transformer 团队**，并引用了行业专家如 *Vespa 的 Jo Bergum* 和一位来自 **Elastic** 的成员。
- **Omar Khattab 关于 DSPy 的演讲**：一位成员分享了来自 MIT/Stanford 的专家 **Omar Khattab** 在 **DSPy** 发表了演讲。


  

---



---



---



---



---



---



{% else %}


> 完整的逐频道细分内容已针对电子邮件进行截断。
> 
> 如果您想查看完整的细分内容，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}