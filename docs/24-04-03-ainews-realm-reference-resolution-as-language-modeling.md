---
companies:
- apple
- openai
- hugging-face
- stability-ai
date: '2024-04-04T00:00:20.574613Z'
description: '**苹果公司**正在推进其 AI 技术，推出了一种名为 **ReALM（将引用解析视为语言建模）**的新方法。该方法通过三种上下文改进了对歧义引用的理解，并微调了一个较小的
  **FLAN-T5** 模型，其在该任务上的表现优于 **GPT-4**。


  在 Reddit AI 新闻方面，开源编程智能体 **SWE-agent** 在 SWE-bench 基准测试中达到了 **12.29%** 的成绩，**RAGFlow**
  则推出了一款可定制的检索增强生成（RAG）引擎。一种名为 **QuaRot** 的新量化方法实现了高效的 4 位推理。


  AI 应用方面包括 T 恤设计生成器、基于 GPT-4 生成播客的 **podgenai**，以及来自 **HuggingFace** 的无需 GPU 即可运行的开源模型。行业讨论集中在大语言模型对
  AI 领域的影响以及推动 AI 开发去中心化的努力上。**泷泽拓人（Takuto Takizawa）**加入 **Stability AI 日本分部**，担任销售与合作伙伴关系负责人。'
id: c6ea6f49-0170-43c6-93c1-308fd26576f2
models:
- flan-t5
- gpt-4
original_slug: ainews-realm-reference-resolution-as-language
people:
- takuto-takizawa
title: '**ReALM：将指代消解视作语言建模**'
topics:
- reference-resolution
- finetuning
- quantization
- retrieval-augmented-generation
- open-source
- coding-agents
- podcast-generation
- image-generation
- ai-industry-trends
---

<!-- buttondown-editor-mode: plaintext -->> 2024年4月2日至4月3日的 AI 新闻。我们为您检查了 5 个 subreddits、[**364** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **26** 个 Discord（**382** 个频道，**4673** 条消息）。预计节省阅读时间（按 200wpm 计算）：**512 分钟**。

在 [WWDC 之前](https://analyticsindiamag.com/what-to-expect-at-the-absolutely-incredible-apple-wwdc-2024/)，Apple 终于开始在 AI 领域大展拳脚。[我们几周前介绍了 MM1](https://buttondown.email/ainews/archive/ainews-mm1-apples-first-large-multimodal-model/)，现在另一个团队展示了 [ReALM: Reference Resolution As Language Modeling](https://arxiv.org/abs/2403.20329)。在他们的术语中，Reference Resolution 是指根据 3 种上下文——1) 屏幕内容，2) 与对话相关的实体，以及 3) 背景实体——来理解诸如“他们”、“那个”、“最底下的那个”或“屏幕上的这个数字”等模糊指代。它们支持各种类似 Assistant 的用例：

 
![image.png](https://assets.buttondown.email/images/d38ff8cd-58f1-4e6c-8ff0-227006ec7cfd.png?w=960&fit=max)
 

考虑到它基本上需要读取你的心思，这是一个极具挑战性的任务。

作者结合使用标注数据和合成数据来 Finetune 一个更小的 FLAN-T5 模型，该模型在此任务上击败了 GPT4：
 
![image.png](https://assets.buttondown.email/images/1e53fed9-dd25-43bd-83ce-ef33cfb2875c.png?w=960&fit=max)
 

没有模型发布，也没有 Demo。但很高兴看到他们是如何处理这个问题的，而且数据集和模型足够小，任何有决心的人都可以复现。

当然，[AI 内容创作者群体对此已经疯狂了](https://www.emergentmind.com/papers/2403.20329)。在这种说法本身变得陈词滥调之前，大概还有几个月的时间可以利用“击败 GPT4”来制造新闻头条。

---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取功能尚未实现，但即将推出。

**AI 研究与开发**

- **开源编程 Agent**：在 /r/MachineLearning 中，研究人员开发了 [**SWE-agent，这是一个在 SWE-bench 基准测试中达到 12.29% 的开源编程 Agent**](https://www.reddit.com/r/MachineLearning/comments/1btwl37/p_sweagent_an_open_source_coding_agent_that/)。该 Agent 可以将 GitHub issues 转化为 pull requests，但研究人员发现，经过 6 个月的工作后，构建有效的 Agent 比预期的要困难。
- **新 RAG 引擎**：同样在 /r/MachineLearning 中，[RAGFlow 被介绍为一个**可定制、可靠且可解释的检索增强生成 (RAG) 引擎**](https://www.reddit.com/r/MachineLearning/comments/1btycwl/d_ragflow_customizable_credible_explainable_rag/)，它基于文档结构识别模型。
- **高效量化**：在 /r/LocalLLaMA 中，[QuaRot 被宣布为一种**支持 4-bit 推理的新量化方法**](https://www.reddit.com/r/LocalLLaMA/comments/1bu8j03/quarot_new_quant_that_offers_4_bit_inference_w4a4/)，比目前需要反量化（dequantization）的方法（如 GPTQ）更高效。它还支持无需校准数据（calibration data）的无损 8-bit 量化。

**AI 应用与工具**

- **T 恤设计生成器**：在一个视频帖子中，一位 Reddit 用户[分享了他们制作的一个**使用 AI 生成 T 恤设计**的工具](https://v.redd.it/ukxhydz1r0sc1)。
- **播客生成**：在 /r/OpenAI 中，[podgenai 作为一款**基于 GPT-4 的免费软件发布，用于生成长达一小时的任何主题的信息性有声读物/播客**](https://www.reddit.com/r/OpenAI/comments/1bujeh6/podgenai_a_free_gpt4_api_based_software_to/)，需要 OpenAI API key。
- **开源语言模型**：HuggingFace CEO 转发了 [PipableAI/pip-library-etl-1.3b 的发布，这是一个**无需 GPU 即可试用的开源模型**](https://i.redd.it/i5ylmzrmo0sc1.jpeg)。

**AI 行业与趋势**

- **大语言模型的影响**：在 /r/MachineLearning 中，发起了一场关于 [**大语言模型 (LLMs) 对 AI 领域是否弊大于利**](https://www.reddit.com/r/MachineLearning/comments/1btuizd/d_llms_causing_more_harm_than_good_for_the_field/) 的讨论，原因是炒作导致会议和工作的重心发生了表面化的改变，过度承诺可能导致另一个 AI 寒冬。
- **去中心化 AI**：分享了一篇 Axios 的文章，内容关于[**去中心化 AI 开发并打破大科技公司垄断的努力**](https://www.axios.com/2024/04/02/ai-decentralized-big-tech-blockchain)。
- **Stability AI 日本人事任命**：发布了关于 [**Takuto Takizawa 加入 Stability AI 日本担任日本销售与合作伙伴关系负责人**](https://i.redd.it/8mt9bei0s5sc1.png) 的新闻。

**Stable Diffusion 讨论**

- **生成任意分辨率**：在 /r/StableDiffusion 中，一位用户询问 [Stable Diffusion **如何在 VAE 输入/输出尺寸固定的情况下生成 512x512 以外的分辨率图像**](https://www.reddit.com/r/StableDiffusion/comments/1bueyze/how_does_stable_diffusion_generate_arbitrary/)，寻求相关解释和代码指引。
- **叙事适用性**：同样在 /r/StableDiffusion 中，一位初学者询问 [Stable Diffusion 是否**适合为叙事和漫画创建特定的角色、姿势和场景**](https://www.reddit.com/r/StableDiffusion/comments/1bukoqt/beginner_question_is_ai_stable_diffusion_good_to/)，因为他们难以控制输出，并考虑将 3D 工具作为替代方案。
- **UI 中的批量生成**：/r/StableDiffusion 的另一位用户正在[寻找如何让 **Automatic1111 的 Stable Diffusion UI 在夜间重复进行批量图像生成**的设置](https://www.reddit.com/r/StableDiffusion/comments/1bula9s/where_is_the_option_to_repeat_generation_in/)。

# AI Twitter 摘要回顾

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和 flow engineering。

**Anthropic 关于 LLMs 越狱的研究**

- **Many-shot jailbreaking 技术**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1775211248239464837) 发布了一篇研究论文，研究了一种在大多数 Large Language Models 上都有效的长上下文越狱技术。研究表明，增加上下文窗口是一把双刃剑，在使模型更有用的同时也使其更容易受到对抗性攻击。
- **有原则且可预测的技术**：[@EthanJPerez](https://twitter.com/EthanJPerez/status/1775230994087543155) 指出，这是目前已知最有效、最可靠且难以通过训练消除的越狱方式，它基于 in-context learning。这种风险会随着模型规模和上下文长度的增加而可预测地加剧。
- **令人担忧的结果**：[@sleepinyourhat](https://twitter.com/sleepinyourhat/status/1775212287214981207) 发现结果既有趣又令人担忧，研究显示，诱导有害行为的 many-shot prompting 在克服安全训练方面，随着示例数量的增加，其有效性遵循幂律（power law）并可预测地提升。

**用于识别分布偏移的 Adversarial Validation 技术**

- **检查训练/测试分布的巧妙技巧**：[@svpino](https://twitter.com/svpino/status/1775154270708396215) 分享了一个名为 Adversarial Validation 的技巧，用于确定训练数据和测试数据是否来自同一分布。将它们放在一起，移除目标变量，为训练/测试集添加二进制特征，训练一个简单模型。如果 AUC 接近 0.5，则为同一分布；如果接近 1，则为不同分布。
- **有助于识别问题特征**：Adversarial Validation 可以识别导致分布偏移的问题特征。计算特征重要性，移除最重要的特征，重建模型，重新计算 AUC。重复此过程直到 AUC 接近 0.5。这在生产环境中识别分布偏移非常有用。

**台湾地震对半导体供应的影响**

- **地震与晶圆厂的距离**：[@nearcyan](https://twitter.com/nearcyan/status/1775382258767016116) 指出，7.4 级地震距离台湾中部科学园区 64 英里。1999 年晶圆厂附近的 7.7 级地震曾导致生产损失。2016 年的 6.6 级地震仅延迟了约 1% 的 TSMC 订单。
- **TSMC 的准备工作**：TSMC 对大地震准备充分。政府优先恢复晶圆厂的公用设施。目前尚未报告结构性损坏。预计新竹/台中受到的干扰将超过台南的 3nm 晶圆厂。
- **潜在延迟**：预计会有至少几周、运气不好甚至几个月的实质性延迟。这可能会导致短期半导体价格波动。

**AI 进展与发展** 

- **来自 DeepMind 的 Genie AI 模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1775162077696360804) 宣布了 Genie，这是一个基础世界模型，可以从单个图像提示、草图或文本描述中创建可玩的 2D 平台游戏世界。它可以帮助训练 AI Agent。
- **Replit Code Repair AI Agent**：[@pirroh](https://twitter.com/pirroh/status/1775327316157358564) 宣布了 Replit Code Repair，这是一个使用 GPT-4 的低延迟代码修复 AI Agent。它在速度和准确性上大幅超越了开源模型。
- **Sonnet 模型替代 GPT-4**：[@jxnlco](https://twitter.com/jxnlco/status/1775169368781209980) 正在 3 家公司的多数用例中用 Sonnet 替代 GPT-4，这显示出向更专业化模型转变的趋势。

**迷因与幽默**

- **编程寿命迷因**：[@svpino](https://twitter.com/svpino/status/1775266990128812314) 调侃说，1994 年有人告诉他编程将在 5 年内消亡，然而 30 年后他仍在编程。
- **Anthropic 越狱暴力迷因**：[@goodside](https://twitter.com/goodside/status/1775271932382068844) 调侃道，如果暴力不能解决你的 LLM 越狱问题，那说明你用的暴力还不够多。

---

# AI Discord 简报

> 摘要的摘要之摘要

1. **内存高效 LLM 训练的进展**：
   - 一种名为 **[DISTFLASHATTN](https://arxiv.org/abs/2310.03294)** 的新注意力机制声称可以将训练长上下文 LLM 时的二次方峰值内存占用降低到线性，从而支持高达 **8 倍长的序列**。然而，该论文缺乏反向传播（backward pass）的伪代码，引发了对可复现性的担忧。
   - 围绕 **[CUDA 优化技术](https://github.com/cuda-mode)**（如 DISTFLASHATTN）的讨论，及其通过内存效率和速度提升（优于 Ring Self-Attention 等现有方案）来彻底改变 LLM 训练的潜力。

2. **AI 模型评估与基准测试**：
   - 开源系统 **[SWE-agent](http://github.com/princeton-nlp/SWE-agent)** 声称在自主解决 GitHub issue 的 SWE-bench 测试中，拥有与 Devin 相当的准确率。
   - **GPT-4**、**Claude** 和 **Opus** 等模型在解决历史提示词、数学谜题和代码生成等任务上的表现各异，凸显了进行全面评估的必要性。
   - **[Chaiverse.com](https://chaiverse.com)** 等平台用于快速获取 RP-LLM 模型的反馈，而 **[LMSys Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)** 则用于模型基准测试。

3. **Prompt Engineering 与多模态 AI**：
   - 讨论了 **Prompt Engineering 技术**，用于在保留 Markdown 的情况下进行翻译、生成管理者提示词，以及使用 Chain of Thought 改进多模态问答。
   - 与 LangChain 和 LlamaIndex 等其他框架相比，**[DSPy](https://arxiv.org/abs/2310.03714)** 在 Prompt 优化方面的潜力。
   - 对**多模态 AI** 的探索，例如使用 Stable Diffusion 从立体图像进行深度映射，以及发布用于高质量音乐生成的 **[Stable Audio 2.0](http://stableaudio.com)**。

4. **开源 AI 发展与部署**：
   - **[Open Interpreter iPhone app](https://github.com/tyfiero/01iOS)** 的开发工作，并将其移植到 Android Termux、M5 Cardputer，实现语音界面并探索本地 STT 解决方案。
   - **[Octopus 2](https://huggingface.co/spaces/Tonic/Octopus/)** 演示版亮相，这是一个具备 Function Calling（函数调用）能力的模型，引发了对端侧（on-device）模型的关注。
   - 发布了 **[Axolotl 文档更新](https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a)** 以及 **[Mojo 标准库](https://github.com/modularml/mojo/tree/nightly/stdlib)** 的开源。

5. **其他主题**：

- **LLM 中的优化挑战与突破**：工程师们致力于解决训练大语言模型时的内存和性能瓶颈，引入了如 **DISTFLASHATTN** 等新技术，声称与现有方案相比可实现[线性内存使用和 8 倍长的序列](https://arxiv.org/abs/2310.03294)。讨论还涉及利用 **bf16 优化器**、**tinyBLAS** 以及 **IPEX-LLM** ([GitHub](https://github.com/intel-analytics/ipex-llm)) 等框架在特定硬件上进行推理加速。

- **对新 AI 模型的期待与分析**：社区对新发布或即将发布的模型反应热烈，如 **Apple 的 ReALM** ([论文](https://arxiv.org/pdf/2403.20329.pdf))、**Stable Diffusion 3.0**、**Stable Audio 2.0** ([网站](http://stableaudio.com))，以及在 SWE-bench 上表现媲美 Devin 的 **SWE-agent** ([GitHub](http://github.com/princeton-nlp/SWE-agent))。对 **Claude**、**Opus** 和 **Haiku** 等指令遵循和对话模型的对比评估也很常见。

- **AI 系统中的伦理问题与 Jailbreaking**：讨论涉及使用受版权保护的数据训练 AI 的法律影响（如音乐平台 Suno），以及语言模型中 Jailbreak 防御的有效性，并引用了一篇关于定义不安全输出重要性的 [arXiv 论文](https://arxiv.org/abs/2403.14725)。聊天机器人的情感模拟能力引发了将 AI 类比为精神病态（psychopathy）的哲学辩论。

- **AI 界面与应用的创新**：**CallStar AI** 等应用突显了与 AI 进行语音交互的潜力，同时社区致力于通过**对话式 UI** 使技术更易于获取。**Open Interpreter** 等项目旨在将 AI 能力带到移动和嵌入式设备。AI 的新颖用例涵盖了从 **WorldSim 的游戏化模拟** ([Notion](https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4)) 到 **AI 生成的艺术和音乐**。


---



# PART 1: 高层级 Discord 摘要

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **优化器难题与建议**：技术讨论揭示了 `torch.compile` 和优化器函数面临的挑战。讨论中提出的一种新兴解决方案涉及一个包含 `bf16 optimizer` 的 Python 包，旨在解决 dtype 冲突和设备兼容性问题。

- **AI 音乐的法律警报**：社区关注了 AI 音乐平台 Suno 可能面临的法律问题，强调了因使用受版权保护的内容进行训练而可能导致唱片公司提起版权侵权诉讼的风险。

- **Apple MPS 中的内存占用与崩溃**：Apple 的 MPS 框架因在高内存分配时（即使内存充足）发生崩溃而受到审查。理论上的内部限制和作为权宜之计的 attention slicing 是热门话题，尽管人们担心这会导致 NaN 错误。

- **文本细节提升图像质量**：研究表明，使用精确的空间描述对 text-to-image 模型进行微调可以增强生成图像的空间一致性，正如一篇 [arXiv 论文](https://arxiv.org/pdf/2404.01197.pdf)所指出的。

- **解码 AI 最佳性能**：从对 SD3 Turbo 声称的效率的怀疑，到关于模型微调和 scheduler 有效性的建议，该公会对各种 AI 策略进行了分析。此外，正如最近的一项[实证研究](https://arxiv.org/abs/2404.01367)所示，在相同的 inference budget 内，较小的模型表现可能会优于较大的模型。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**推进 Stable Diffusion**：用户报告称，Stable Diffusion 的用户界面 **Forge** 提供了卓越的性能，尤其是在 RTX 3060 和 RTX 4080 显卡上。推荐使用 DreamShaper Lightning (SDXL) 模型以提高图像生成的效率和速度。

**对 SD3 的高度期待**：Stable Diffusion 社区正积极期待 **Stable Diffusion 3.0** 的发布，预计将在未来 3-5 周内推出，预计文本渲染将有所改进，尽管完美的拼写可能仍然难以实现。

**创意 AI 的释放**：成员们正在尝试使用 Stable Diffusion 为桌面 RPG 等项目创作艺术，并考虑通过 AI 生成的视觉叙事（可能是漫画或电影格式）进行讲故事。

**疑难解答技术技巧**：讨论集中在解决图像生成缓慢和出现多余文本等问题上，参与者提出了优化建议，并提到 GitHub 链接作为故障排除的起点。

**功能预测**：对于即将推出的功能（如 sparse control net、SegMOE 和 audiosparx 模型），社区表现出明显的兴奋，大家分享了资源并期待 AI 生成内容的新可能性。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Cortana 1.0 聊天模型引发好奇**：工程师们讨论了基于《光环》（*Halo*）系列 AI 创建名为 **Cortana 1.0** 的 AI Prompt 模型，重点是创建有效的聊天模式和 Prompt 结构以简化交互。

**Unsloth Enterprise 能力澄清**：官方澄清 **Unsloth Enterprise** 确实支持全模型训练，速度比 FA2 提升 2-5 倍，而非预期的 30-40 倍。

**AI 优化交流**：一系列活跃的讨论涵盖了各种优化主题，包括 **Unsloth AI** 的进展（提到了 [Daniel Han 的推文](https://twitter.com/danielhanchen/status/1775120334305607781)）、用于加速 AI 推理的 GitHub 资源（如 [ipex-llm](https://github.com/intel-analytics/ipex-llm)），以及 AI 模型的故障排除，特别是 *SFTTrainer* 与 **Gemma models** 的兼容性。

**小行星采矿的创新方法**：[Open Asteroid Impact](https://openasteroidimpact.org/) 项目以一种新颖的概念吸引了人们的兴趣，即通过将小行星带回地球来更有效地利用资源。

**全栈人才招募**：社区内正在征集熟练的 Full Stack 开发人员，并鼓励用户如果能推荐或提供帮助请私信（DM）。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**解读 PDF 的字里行间**：工程师们讨论了使用 **Claude** 和 **Haiku** 等 AI 模型来解析 PDF，重点关注 *context windows*（上下文窗口）和 **Perplexity** 的 Pro 功能，特别是 “Writing” 焦点模式以及开启 “Pro” 模式以提高准确性。一些用户更倾向于使用 **Sonar** 以获得更快的响应。

**广告话题引发用户争论**：**Perplexity 引入广告**的可能性引发了辩论，此前 Perplexity 的首席商务官发表了关于整合赞助建议的言论。用户对 Pro 订阅者的体验可能受到的影响表示担忧，并引用了一篇关于该主题的 [Verge 文章](https://www.theverge.com/2024/4/1/24118154/perplexity-will-try-a-form-of-ads-on-its-ai-search-platform)。

**PDF 障碍与图像生成**：在解决技术问题时，用户明确了 **Perplexity 的移动端 App 缺乏图像生成支持**——虽然这一不便可以通过在移动设备上使用网页版（具备类似桌面端的功能）来进行图像生成来缓解。另外的讨论指出，用户希望取消 25MB 的 PDF 限制以提高效率。

**工程师交流“推荐链接”**：推荐计划和折扣成为了热门话题，提到了通过提供的**链接**（links）来节省费用。

**API 的烦恼与变通方案**：在 Perplexity API 领域，用户正在努力应对缺乏 **team support**（团队支持）和 API 额度**支付问题**，同时也分享了对 **rate limits**（速率限制）以及从 sonar-medium-online 模型接收到过时响应的沮丧。建议范围从准确的请求日志记录到优化 system prompts 以获取最新新闻。

**好奇心驱动深度探索**：
- 用户利用 AI 探索了从 **Fritz Haber 的生平**和伦理困境到 **random forest classifiers**（随机森林分类器）以及《希腊左巴》(Zorba the Greek) 等一系列主题，这取决于 AI 是否适合满足多样且复杂的查询。
- 他们利用 Perplexity 高效地为时事通讯汇编综合数据，表明了利用 AI 简化内容创作的强烈倾向。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**开源 AI 媲美 Devin**：作为 Devin 开源替代方案推出的 [SWE-agent](http://github.com/princeton-nlp/SWE-agent) 在 SWE-bench 上展示了相当的性能，引发了关于其潜在集成和应用的讨论。

**Apple 的 AI 研究蓄势待发**：[Apple 的一篇新论文](https://arxiv.org/pdf/2403.20329.pdf)展示了 ReALM，暗示其 AI 进展可能超越 GPT-4 的能力，并与即将推出的 iOS 18 紧密集成，以改进 Siri 的交互。

**Claude 的难题**：用户正在测试 Claude Opus，但发现它在处理复杂任务时面临挑战，因此推荐参考[提示工程互动教程](https://www.anthropic.com/index/claude-2-1-prompting)（Prompt Engineering Interactive Tutorial）以增强与模型的交互。

**Stable Audio 2.0 强化音频体验**：StabilityAI 推出了 [Stable Audio 2.0](http://stableaudio.com)，凭借其制作完整长度、高质量曲目的能力，推向了 AI 生成音乐的新边界。

**DALL-E 新增编辑按钮**：ChatGPT Plus 现在包含允许用户**编辑 DALL-E 生成的图像**和**编辑对话提示词**的功能，带来了自定义和控制的新维度，详情见 [OpenAI 帮助页面](https://help.openai.com/en/articles/9055440-editing-your-images-with-dall-e)。

**DSPy 框架讨论升温**：LLM Paper Club 详细审查了 [DSPy 框架](https://arxiv.org/abs/2310.03714)的功能及其在 prompt 优化方面优于其他框架的优势，激发了将其应用于各种项目的想法，如语音 API 日志应用和学术论文摘要平台。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **SWE-agent 崛起，Devin 趋于平稳**：一个名为 **SWE-agent** 的尖端系统被推出，声称在解决 GitHub issues 方面能与前作 **Devin** 媲美，平均处理时间仅为惊人的 93 秒，且已在 [GitHub](http://github.com/princeton-nlp/SWE-agent) 开源。

- **80M 模型引发质疑**：工程师们讨论了一个 **80M 模型** 在分布外（out-of-distribution）数据上取得的惊人成功，引发了关于误差幅度的推测，并激起了对其性能有效性的辩论。

- **中国处理器表现超出预期**：关于 AI 硬件的讨论提到了 **云天励飞（Intellifusion）的 DeepEyes**，这是一款中国 14nm AI 处理器，以显著降低的成本提供具有竞争力的 AI 性能，可能对硬件市场发起挑战（[Tom's Hardware 报告](https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus)）。

- **微调英雄与模型故障**：社区分享了模型微调的经验，例如 **Lhl** 对 *jamba 模型* 的工作，以及 *Mvds1* 因元数据问题在向 Hugging Face 上传模型时遇到的困难，指出需要手动调整 `SafeTensorsInfo`。

- **WorldSim 激发社区想象力**：工程师们热情地探索了 **WorldSim** 的功能，从 text-to-video 集成到社区路线图，讨论了技术增强并分享了资源，如 [Notion 上的 WorldSim 命令索引](https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4)。WorldSim 的技术限制和游戏化是热门话题，展示了社区在模拟平台上的创新动力和参与度。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 缺乏 Embedding 模型支持**：用户确认 **LM Studio** 目前不支持 embedding 模型，强调 embedding 功能尚未实现。
- **AI 推荐查询受到关注**：一个用户请求能够提供成人动漫推荐的模型，引发了使用 **MyAnimeList (MAL)** ([myanimelist.net](https://myanimelist.net/)) 的建议，社区对这一不寻常的咨询感到有趣。
- **优化 LLM 配置的悬念**：硬件频道的讨论揭示了关于 LM Studio **无需 SLI 的多 GPU 配置** 的见解，推荐了如 Nvidia **Tesla P40** 等 GPU，并对由于 [影响台积电 (TSMC) 的大地震](https://www.bloomberg.com/news/articles/2024-04-03/tsmc-evacuates-production-lines-after-major-taiwan-quake) 导致的未来硬件价格表示担忧。
- **API 类型对 Autogen 集成至关重要**：LM Studio 的故障排除强调了指定 API 类型的重要性，以确保与 **Autogen** 的正常协作。
- **针对 CrewAI 的跨源资源共享 (CORS)**：讨论了启用 CORS 作为解决 **LM Studio** 本地模型使用问题的潜在方案，并通过一篇 [Medium 文章](https://medium.com/@tayyibali4300/implementing-lm-studio-in-crewai-270cc577acee) 提供了额外指导。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DALL·E 进入 ChatGPT 领域**：**ChatGPT** 界面中引入了直接在对话中编辑图像和风格灵感的功能，针对 **DALL·E** 图像，兼顾了便利性和创意探索。

- **Bing API 陷入沉寂**：持续 12 小时的 **Bing API** 故障引起了用户的担忧，影响了依赖它的服务（如 DALL·E 和 Bing Image Creator），表明需要强大的备用方案。

- **对情感的困惑**：围绕类 GPT 的 LLM 是否能真实模拟情感展开了激烈辩论，指出 AI 缺乏内在动力，并将其与精神病态以及著名的 Eliza 效应进行比较。

- **盒子里的经理**：征集用于处理管理任务的 prompt 编写，强调了 AI 社区对自动化复杂领导角色的兴趣，尽管讨论中尚未产出实际的策略或解决方案。

- **翻译难题与 Markdown 困扰**：尝试制作保留 Markdown 语法的翻译 prompt 时遇到了阻碍；翻译的不一致性（尤其是阿拉伯语）让 AI 工程师质疑当前语言模型处理复杂格式和语言细微差别时的能力极限。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**向 Linux GPU 先驱告别**：John Bridgman 从 AMD 退休引发了关于他对 Linux 驱动程序贡献的讨论，George Hotz 对 AMD 的管理现状和未来方向发表了评论。Hotz 呼吁 AMD 员工提供匿名爆料，以便可能撰写一篇博客揭露文章，此时社区正对 AMD 在驱动问题和开源承诺上的执行力表示担忧，正如辩论和一篇 [Phoronix 文章](https://www.phoronix.com/news/AMD-Bridgman-Retires)中所强调的那样。

**Linux Kernel 与 NVIDIA 的开源举措**：讨论延伸到了不同 Kernel 版本的影响，特别是围绕 Intel 的 Xe 和 i915 驱动程序，以及 Linux 发行版之间的迁移偏好，例如从 Ubuntu 22.04 LTS 转向 24.04 LTS。此外，George Hotz 提到了他对 [NVIDIA 开源驱动倡议](https://github.com/NVIDIA/open-gpu-kernel-modules)的贡献，引发了关于开源 GPU 驱动与专有驱动现状的对话。

**Tinygrad 的 V1.0 之路离不开社区**：对 **tinygrad 的 beam search 启发式算法**和 **CommandQueue** 功能的探索，突显了 George Hotz 对改进文档以帮助用户学习和贡献的重视，包括提议一个受 ["Write Yourself a Scheme in 48 Hours"](https://en.wikibooks.org/wiki/Write_Yourself_a_Scheme_in_48_Hours) 启发的教程。这与社区贡献相辅相成，例如[这个 command queue 教程](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md)，旨在完善 tinygrad。

**活跃的成员参与增强了 Tinygrad**：社区在创建学习材料方面的积极性受到了赞赏，成员们提供资源并主动直播他们在 tinygrad 上的实践经验，营造了协作学习的环境。这与达成 tinygrad 1.0 版本的共同目标一致，巩固了该平台作为教育和创新工具的地位。

**重新思考 AI 模型中的内存使用**：一场关于模型前向传播过程中内存优化的技术辩论随之展开，特别是关于利用 [反函数法则 (inverse function rule)](https://en.wikipedia.org/wiki/Inverse_function_rule) 使用具有反函数的激活函数。这体现了社区不仅参与工具开发，还参与基础原理的研究，以提高 AI 计算的处理效率。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**OpenInterpreter 深入应用开发**：**Open Interpreter iPhone 应用**的开发正在取得进展，完成度约为 40%，该项目由 [GitHub](https://github.com/tyfiero/01iOS) 上的社区协作推动，灵感源自 [Jordan Singer 的 Twitter 概念](https://x.com/jsngr/status/1774110742070882478?s=46&t=kwbSfLYCOimQnegJhHK_iA)。

**让技术更易于触达**：**Open Interpreter** 社区正在推动引入 **对话式 UI 层 (Conversational UI layer)**，以帮助老年人和残障人士，旨在显著简化他们与技术的交互。

**数字时代的安全措施**：成员们被警告要远离一个疑似被盗的 **Open Interpreter X** 账号发布的潜在危险帖子，以防止加密货币钱包被侵入。

**创新的移植计划**：**OpenInterpreter** 正在模糊平台界限，推出了用于 Android Termux 安装的新仓库，正在进行 **M5 Cardputer** 移植工作，并讨论了在 GPT-4 成本担忧下实施本地 STT 解决方案。

**对 AI 见解的期待**：社区分享了对深入理解 LLM 的热情，这可能表明人们对获取 AI 系统高级技术知识有着浓厚兴趣。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Tinystories 饱和警报**：据报道，**Tinystories** 数据集在约 5M 参数时达到饱和点，引发了转向更大规模 `minipile` 数据集的讨论，尽管后者对处理能力的要求更高。

- **呼吁组建 AI 竞赛团队**：社区对 EleutherAI 支持 AI 竞赛团队表现出浓厚兴趣，建议利用 llema 等模型和 RLHF 专业知识，并提议设立专门频道并寻求 compute grants 支持。

- **防御语言模型 Jailbreaking**：最近的一篇 [论文](https://arxiv.org/abs/2403.14725) 指出，定义不安全回答时的歧义是保护语言模型免受 'jailbreak' 攻击的关键挑战，重点应放在后处理输出的精确性上。

- **AI 模型反馈提交亮点**：关于 AI 模型政策的 [公开评论](https://www.regulations.gov/document/NTIA-2023-0009-0001/comment) 显示了对开源模型开发的偏好，如 EleutherAI 的 LaTeX 风格贡献所示，讨论中既流露出自豪感，也提到了社区参与中错失的机会。

- **LLM 安全过滤器增强建议**：关于在 LLM 微调数据中混入拒绝示例的对话引用了 @BlancheMinerva 的推文和相关研究，证实了 [ArXiv 论文](https://arxiv.org/pdf/2402.18540.pdf) 中提到的对安全过滤器鲁棒性的日益关注。

- **ChemNLP 带来的化学领域突破**：首篇 ChemNLP 项目论文在 [ArXiv](https://arxiv.org/abs/2404.01475) 上的发布预示着 AI 驱动化学领域的重大进展，引发了关注并可能带动未来研究方向的讨论。

- **开源 AI 面临的法律阴影**：深入探讨加州 SB 1047 法案对开源 AI 项目的影响，鼓励签署抗议公开信，这表明社区对该法案限制创新后果的担忧。详细评论见 [此处](https://www.context.fund/policy/sb_1047_analysis.html)。

- **抽象与具体之间的难题**：一个关于“房子”如何介于“具体长颈鹿”和“抽象长颈鹿”之间的奇特澄清请求，得到了一个轻松的数字耸肩回应，展示了社区讨论中幽默而神秘的一面。

- **Neel Nanda 的 MATS Stream 开放申请**：提醒 Neel Nanda 的 MATS stream 申请截止日期临近（少于 10 天），完整详情见此 [Google Doc](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit#heading=h.y0ohi6l5z9qn)。

- **多语言生成式 QA 的参与**：讨论了使用 **Chain of Thought** (CoT) 提升多语言 QA 任务的潜力，涉及 **MGSM** 等数据集，并生成了一个展示包含 `generate until` 功能任务的列表。

- **CUDA 疑难解答呼吁社区帮助**：一位用户在 H100 GPU 上遇到 `CUDA error: no kernel image is available for execution on the device` 错误（在 A100 GPU 上未出现），排除了 flash attention 的原因，进一步建议检查 `context_layer` 设备以解决问题。

- **PyTorch 的弹性探索**：关于预训练期间 **GPU/TPU 弹性调整** 的问题，得到了采用 [PyTorch Elastic](https://pytorch.org/docs/stable/elastic/quickstart.html) 的建议，该工具展示了适应故障和动态调整计算资源的能力，引起了寻求可扩展训练方案者的兴趣。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**提升仓库隐私**：Hugging Face 现在允许企业组织默认将仓库可见性设置为公开或私有，从而增强隐私控制。[他们的推文](https://twitter.com/julien_c/status/1772688542289822073)中有更多详情。

**通过命令发布**：Quarto 用户可以使用 `use quarto publish hugging-face` 在 Hugging Face 上部署网站，正如最近在 [Twitter](https://twitter.com/gshotwell/status/1772661727856914720) 和 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7178422723503673345?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7178422723503673345%29) 帖子中分享的那样。

**Gradio 的新特性**：Gradio 在最新的 4.25.0 版本中引入了**状态变量自动删除**和**延迟示例缓存**，详见其 [changelog](https://gradio.app/changelog)。

**探索 CLI 前沿**：一段分享的 [YouTube 视频](https://www.youtube.com/watch?v=PKYPKRoCW2c)向开发者解释了如何在命令行界面（CLI）中使用 Linux 命令、容器、Rust 和 Groq。

**推动 LLM 达到高效运行状态**：一位用户询问如何在计算资源受限的情况下，针对 PDF 微调语言模型，并重点关注使用开源模型的推理。同时，一场关于在微调 LLM 时修改 tokenizer 中特殊 token 的讨论也在展开。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**聊天记录中的持久上下文探索**：工程师们讨论了如何维护**聊天中的持久上下文**，特别是在与“问题：回答”对数据库交互时，但尚未达成统一的解决方案。讨论中引用了 [LangChain 的 issue 和文档](https://github.com/langchain-ai/langchain/issues/8066)以寻求潜在的推进方法。

**LangServe Playground 视频教程**：分享了一个介绍 LangServe 中 **Chat Playground** 特性的[视频教程](https://www.youtube.com/watch?v=stWiNP1o2_g)，旨在简化初始设置并展示其与 Langsmith 的集成。

**语音指令未来**：宣布推出了多个 AI 语音应用，如 **CallStar AI** 和 **AllMind AI**，这暗示了语音作为 AI 交互接口的趋势。提供了 [Product Hunt](https://www.producthunt.com/posts/callstar) 和 [Hacker News](https://news.ycombinator.com/item?id=39914442) 等平台的链接以寻求社区支持。

**AI 工程问题与教程**：[langchain-ai/langserve 的一个 pull request](https://github.com/langchain-ai/langserve/pull/580) 报告了 CI 问题；此外，有人在采用 LangChain 的 `ChatOpenAI` 和 `ChatPromptTemplate` 时遇到了 `NotFoundError` 并寻求指导。同时，新手被引导至一份全面的 [LangChain 快速入门指南](https://python.langchain.com/docs/get_started/quickstart)。

**GalaxyAI 服务与提示词熟练度测试**：GalaxyAI 提供了**免费访问高级 AI 模型**的机会，并强调了与 Langchain 的 API 兼容性，尽管服务链接缺失。另一个项目 [GitGud LangChain](https://tinyurl.com/gitgud-langchain) 则挑战**资深提示词工程师**测试一种新的代码转换工具，以维护代码质量。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 与内存安全**：将 **Mojo 语言**集成到 [ROS 2](https://github.com/ros2) 中显示出对机器人开发的潜在益处，这得益于 Mojo 的内存安全实践。C++ 和 Rust 的对比显示出人们对机器人环境性能和安全性的日益关注。

**Docker 构建启航**：即将发布的 **Modular 24.3** 将包含一个旨在提高**自动化 Docker 构建**效率的修复方案，这受到了社区的好评。

**日志记录器的灵活性飞跃**：Mojo 中的 logger 库已更新，可接受**任意参数和关键字参数**，从而实现更动态的日志记录，在消息之外容纳更多样化的信息。

**Mojo Dict 需要更快的速度**：社区在 [One Billion Row Challenge](https://github.com/VMois/1brc-mojo)（十亿行挑战）中的参与表明，**Mojo** 中 `Dict` 的性能需要增强。目前正在努力讨论实现自定义的、可能基于 SIMD 的 `Dict`，以跟上 swiss tables 等解决方案的步伐。

**共同推动 Mojo Nightly 版本的改进**：成员们表达了希望在 **Mojo 标准库（stdlib）开发**中拥有更清晰的贡献和排错路径。GitHub 上的讨论澄清了诸如解析错误和 `Optional` 类型行为等挑战，这表明了社区正在积极协作以完善 Mojo 的功能。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **TogetherAI 遭遇超时故障**：用户报告 **NOUSRESEARCH/NOUS-HERMES-2-MIXTRAL** 模型出现故障，具体表现为错误代码 524，这表明 **TogetherAI** 的 API 可能存在上游问题。建议使用 **Nous Capybara 34B** 作为替代方案。
  
- **聊天机器人的历史准确性测试结果参差不齐**：在识别二战背景下的日本将军**山本五十六 (Isoroku Yamamoto)** 时，**Claude**、**Opus** 和 **Haiku** 等 LLM 表现出不同程度的准确性，凸显了当前聊天机器人在处理历史事实方面的挑战。

- **OpenRouter 触及 4MB 上限**：OpenRouter 存在一项技术限制，即 Body 内容的**最大 Payload 大小为 4MB**，已确认目前没有绕过该限制的方法。

- **AI 助力角色扮演**：在 AI 辅助角色扮演领域，**Claude 3 Haiku** 成为关注焦点，用户分享了优化策略，包括对模型进行 Jailbreaking 以及应用 Few-shot Learning 来磨合交互效果。

- **社区众包 Prompt 游乐场**：推荐 **SillyTavern** 和 **Chub** 的 Discord 服务器给那些寻求丰富 Prompt 资源和 Jailbroken 模型的人，并指出了诸如 **pancatstack jailbreak** 等特定技术。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**RankZephyr 在竞争中脱颖而出**：建议将 *RankZephyr* 集成到高级 *Retrieval-Augmented Generation* (RAG) 系统中以增强 Reranking，*RankLLM* 系列因其 [Fine-tuning 能力](https://twitter.com/llama_index/status/1775166279911186930)而受到认可。

**利用 AI Copilot 提升研究敏捷性**：一份**网络研讨会总结**揭示了构建 **AI Browser Copilot** 的关键策略，重点在于 *Prompt Engineering Pipeline*、*KNN Few-shot 示例*和*向量检索 (Vector Retrieval)*，更多见解可在 [LlamaIndex 的 Twitter](https://twitter.com/llama_index/status/1775264340465381536) 上查看。

**及时的数据检索创新**：据称 *KDB.AI* 通过为**混合搜索 (Hybrid Searching)** 引入*时间敏感查询*，改进了 **Retrieval-Augmented Generation**，从而实现了对财务报告等场景至关重要的更细致的搜索能力，如[代码片段](https://twitter.com/llama_index/status/1775269014849359925)所示。

**智能库重新定义知识管理**：一款面向专业人士和团队的**新型 LLM 驱动数字图书馆**据称将彻底改变知识组织方式，其功能允许在高级数字环境中进行创建、组织和注释，正如 [LlamaIndex 的推文](https://twitter.com/llama_index/status/1775537091272937933)所宣布的那样。

**社区对话提出技术问题**：社区讨论包括索引大型 PDF 的挑战、*qDrant* 在 *IngestionPipeline* 后不释放锁的问题、**HuggingFace API** 的限制、使用 **Ollama 类**进行模型集成，以及 **RAG** 递归查询引擎中的文档缺失问题。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Axolotl 文档焕然一新**：**Axolotl 文档**进行了外观更新，虽然最初遗漏了**目录 (Table of Contents)**，但已通过此 [GitHub commit](https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a) 迅速修正，不过标题与目录之间的一致性仍需进一步清理。

**Serverless vLLM 部署的苦与乐**：分享了关于 **Runpod** 和 Serverless **vLLM** 的经验，强调了挑战，并提供了关于如何部署 [LLM Endpoints](https://github.com/runpod-workers/worker-vllm) 的资源。

**数据聚合难题**：整合包含数百 GB 数据的多个数据集的努力面临复杂情况，包括文件对齐问题。目前使用 TSV 文件和 Pickle 格式的索引数据进行快速检索，同时也在讨论更高效的解决方案。

**轻松的 AI 模型大比拼**：一场轻松的辩论比较了对 'qwen mow' 与 'jamba' 等 AI 模型的偏好，社区开玩笑说需要更多的数据和资源。

**征集高清数据**：一位社区成员正在寻找获取 **4K 和 8K 图像**集的资源，这表明某个项目或研究需要高分辨率图像数据。



---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Windows ARM 上的 Llamafile 困境**：为 Windows ARM 编译 **llama.cpp** 需要进行源码编译，因为目前尚不提供预构建支持。由于 Windows 上 Cosmopolitan 开发环境存在问题（如 [Cosmopolitan issue #1010](https://github.com/jart/cosmopolitan/issues/1010) 中所述），开发者已被建议使用其他平台来构建 **llamafile**。

- **Mixtral 的智力随参数规模提升**：**Mixtral** 版本 **`mixtral-8x7b-instruct-v0.1.Q4_0.llamafile`** 擅长解决数学谜题；然而，为了确保事实记忆准确无误，建议使用 **`Q5_K_M`** 或更高版本。感兴趣的用户可以在 [Hugging Face](https://huggingface.co/jartine/Mixtral-8x7B-Instruct-v0.1-llamafile/tree/main) 上找到具体细节。

- **TinyBLAS 带来的性能提升**：在使用 **llamafile** 时，通过使用 *`--tinyblas`* 标志可以大幅提升 GPU 性能，该标志无需额外 SDK 即可提供支持，不过效果可能取决于所使用的 GPU 型号。

- **PE 文件可包含 ARM64 和 ARM64EC**：Windows on ARM 支持带有 ARM64X 二进制文件的 PE 格式，这种格式结合了 Arm64 和 Arm64EC 代码，详见微软的 [Arm64X PE Files 文档](https://learn.microsoft.com/en-us/windows/arm/arm64x-pe)。由于 ARM64EC 中不支持 AVX/AVX2 指令仿真，这可能会阻碍 LLM 通常需要的操作，从而带来潜在挑战。

- **延伸阅读参考**：分享了包括 Windows 上 HIP SDK 安装指南以及使用 Llamafile 进行性能增强的详细信息等文章和资源，例如 [The Register](https://www.theregister.com/2024/04/03/llamafile_performance_gains/) 上的“Llamafile LLM 驱动项目提升 CPU 核心性能”，以及可在[此处](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html)获取的 HIP SDK 安装文档。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Opus Judgement 预示 AI 性能提升**：讨论强调了 *Opus Judgement* 在解锁研究级 AI 微调（**RLAIF**）性能提升方面的潜力，其确定性取决于其准确性。

- **Google 的 AI 强势举措**：工程师们对 Logan K 转型领导 Google 的 AI Studio 议论纷纷，对其动机的猜测从个人生活方式到战略职业定位不等。[官方公告](https://fxtwitter.com/OfficialLoganK/status/1775222819439149424)引发了人们对他领导下 Gemini API 未来的期待。

- **Logan K 引发更广泛的 AI Alignment 辩论**：Logan K 的举动引发了关于 AI Alignment 价值观与企业诱惑之间的讨论，人们在思考这一选择是为了在 Google 实现更开放的模型共享，还是仅仅因为丰厚的薪酬而无视个人对齐原则。

- **AI 进展中的神秘氛围**：一位成员指出，GPT-4 技术报告缺乏透明度所产生的连锁反应，标志着 AI 公司之间保密性增加、模型细节分享减少的趋势。

- **无法获取 AI 财务分析**：人们对 AI 财务影响的兴趣被《金融时报》一篇讨论 Google AI 搜索变现的文章所激发，但由于 [Financial Times](https://www.ft.com/content/2f4bfeb4-6579-4819-9f5f-b3a46ff59ed1) 内容的访问受限，限制了技术社区内的讨论。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 撞上 LLM 优化**：[DISTFLASHATTN 机制](https://arxiv.org/abs/2310.03294)声称在训练长上下文大语言模型（LLM）期间实现了**线性内存占用**，相比传统的二次方峰值内存占用，它允许处理高达 **8 倍长的序列**。然而，社区注意到论文中缺少 backward pass 的伪代码，引发了对可复现性的担忧。

- **代码讨论**：对于那些寻求 CUDA 实战经验的人，推荐将 [CUDA MODE YouTube 频道](https://www.youtube.com/@CUDAMODE)和相关的 [GitHub 资料](https://github.com/cuda-mode)作为从 Python 和 Rust 转向 CUDA 的初学者的起点。

- **内存高效训练引起关注**：专注于优化 LLM 训练的 DISTFLASHATTN 论文正受到关注，一名成员标记了即将进行的详细审查，暗示将围绕其内存高效训练的优势展开进一步讨论。

- **对 Backward Pass 的负面反馈**：一名成员对 DISTFLASHATTN 论文中缺乏 backward pass 伪代码的批评引起了社区的共鸣，呼吁在 Attention 机制研究中提高科学可重复性。

- **指向 Intel Analytics 仓库的指针**：分享了一个指向 Intel Analytics 的 ipex-llm GitHub 仓库的链接，未提供额外上下文，可能暗示了 LLM 领域的新工具或进展。



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

**Token 效率讨论**：一位用户强调了一篇论文的发现，即**吞吐效率**随着**每个 token** 的测量而增加，其计算方式是端到端吞吐量（包括 encoding 和 decoding）与 **token 总数**的比率。

**速度辩论升温**：关于增加 token 如何影响生成速度存在分歧——虽然 encoding 可以并行完成，但 decoding 固有的顺序特性意味着每个新 token 都会增加处理时间。

**关注 Encoding 性能**：讨论中的澄清指出了一张绘制生成固定 512 个 token 速度的图表，暗示图中观察到的速度提升应归功于**更快的 encoding** 而非 decoding。

**Decoding：顺序减速困境**：有人询问尽管 decoding 存在顺序依赖性（理论上要求等待每个 token 的前序 token），是否仍有提高其速度的可能性。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **热情的 Python 开发者加入**：一位拥有 **Python**、**软件工程**背景和**数据科学**硕士学位的新贡献者正寻求加入团队并为入职流程做出贡献，他带来了 AI 医学研究和数据流水线（data pipeline）构建方面的专业知识。
  
- **GPT-4 在没有上下文的情况下被数学题难住**：即使是像 **GPT-4** 和 **Claude** 这样先进的 AI，除非问题用清晰的自然语言提出，否则在解方程时也会遇到困难，这表明目前的 AI 模型仍有改进空间。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

似乎没有足够的上下文来生成摘要。请提供更多关于该 Discord 服务器内各频道的讨论信息，以便输出有意义的摘要。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **对话数据澄清**：一位 AI 工程师澄清了对话日志中使用的术语，引用了 `logs.db` 中的 `responses` 表。建议将对话的初始部分称为“speaker turn”或简称“turn”，并因此将其应用的表重命名为 `turns`。



---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1224692800659001356)** (699 条消息🔥🔥🔥): 

- **优化器实现中的故障排除**：成员们就使用 `torch.compile` 和 `add_stochastic_` 函数时遇到的问题进行了技术讨论，指出了在 NVIDIA, AMD 和 MPS 等不同设备上的兼容性问题。讨论了一个涉及为 bf16 optim 创建 Python 包的潜在解决方案，以及为防止操作期间出现 dtype 冲突错误而进行的可能修改。

- **对 SD3 效率提升的怀疑**：在一名成员因质疑有限 token 训练和该方法的长期可行性而被服务器封禁后，人们对 SD3 Turbo 效率提升的说法产生了怀疑。还有建议认为，依赖 CLIP 等工具可能会引入伪影，从而阻碍全面学习。

- **AI 生成音乐的法律风险**：关于 AI 音乐平台 Suno 的讨论凸显了潜在的版权侵权问题。有人担心，如果 Suno 使用受版权保护的音乐进行训练，唱片公司强大的法律团队可能会带来严峻挑战。用户们讨论了在法庭上证明侵权的复杂性。

- **高内存占用下的 MPS 限制与崩溃**：有人指出，尽管内存充足，Apple 的 MPS 框架在训练期间分配超过 2^32 字节的数据时会发生崩溃，这表明可能存在内部限制。文中还提到了 attention slicing 等实际解决方法，尽管这些方法可能会导致 backward pass 期间出现 NaN 等其他问题。

- **模型 Fine-Tuning 与 Scheduler 选择建议**：关于如何将 CLIP 与 T5 等其他模型结合使用以获得更好性能存在争论，其中一名成员支持最终排除 CLIP，转而使用纯 T5 模型以避免长期问题。进一步的讨论涉及社区内关于 sampler 效率和理想采样次数的不一致信息及误传。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.972mag.com/lavender-ai-israeli-army-gaza/">‘Lavender’：指挥以色列在加沙进行轰炸行动的 AI 机器</a>：以色列军队利用一个几乎没有人类监督且对伤亡政策宽松的 AI 目标定位系统，将数万名加沙人标记为暗杀嫌疑人，+972 和 Local C...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1axbjrp/psa_recent_pytorch_nightlies_support_enough/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://tenor.com/8a9w.gif">Ian Malcolm GIF - Ian Malcolm Jurassic - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://discuss.pytorch.org/t/runtimeerror-required-rank-4-tensor-to-use-channels-last-format/159729">RuntimeError: required rank 4 tensor to use channels_last format</a>：我的 Transformer 训练循环在 CPU 上运行时似乎正常，但当我切换到 MPS 时，在计算 Cross Entropy loss 的 loss.backward() 时遇到了以下错误。我正在进行机器学习...</li><li><a href="https://arxiv.org/abs/2404.01292">衡量 Diffusion Models 中的风格相似性</a>：生成模型现在被图形设计师和艺术家广泛使用。之前的研究表明，这些模型在生成过程中会记住并经常复制其训练数据中的内容。因此...</li><li><a href="https://www.youtube.com/watch?v=_D3GACF-Bsk">Galileo</a>：未找到描述</li><li><a href="https://www.musicbusinessworldwide.com/suno-is-a-music-ai-company-aiming-to-generate-120-billion-per-year-newton-rex/">Suno 是一家旨在每年创造 1200 亿美元收入的音乐 AI 公司。但它是否在受版权保护的录音上进行过训练？ - Music Business Worldwide</a>：Ed Newton-Rex 发现 Suno 创作的音乐与经典版权作品有着惊人的相似之处...</li><li><a href="https://www.youtube.com/watch?v=5pidokakU4I">Axis of Awesome - 4 Four Chord Song (附歌曲名称)</a>：澳大利亚喜剧团体 'Axis Of Awesome' 表演了 2009 年墨尔本国际喜剧节的一个片段。视频由 Network Ten Australia 提供。...</li><li><a href="https://github.com/pytorch/pytorch/issues/120930>">Issues · pytorch/pytorch</a>：Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - Issues · pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/issues/71631>">Issues · pytorch/pytorch</a>：Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - Issues · pytorch/pytorch</li><li><a href="https://github.com/Nerogar/OneTrainer/blob/9a35e7f8596988f672af668f474f8d489ff8f962/modules/util/optimizer/adafactor_extensions.py">OneTrainer/modules/util/optimizer/adafactor_extensions.py at 9a35e7f8596988f672af668f474f8d489ff8f962 · Nerogar/OneTrainer</a>：OneTrainer 是满足你所有 Stable Diffusion 训练需求的一站式解决方案。- Nerogar/OneTrainer</li><li><a href="https://github.com/huggingface/diffusers/issues/7563">[mps] 训练 / 推理 dtype 问题 · Issue #7563 · huggingface/diffusers</a>：当在没有 attention slicing 的情况下在 Diffusers 上进行训练时，我们看到：/AppleInternal/Library/BuildRoots/ce725a5f-c761-11ee-a4ec-b6ef2fd8d87b/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPS...</li><li><a href="https://github.com/steffen74/ConstitutionalAiTuning/">GitHub - steffen74/ConstitutionalAiTuning：一个用于微调具有自定义伦理或上下文对齐的 LLM 的 Python 库，利用了 Anthropic 提出的 Constitutional AI 原则。简化了 Prompt 生成、模型交互和微调的过程，以实现更负责任的 AI 开发。</a>：一个用于微调具有自定义伦理或上下文对齐的 LLM 的 Python 库，利用了 Anthropic 提出的 Constitutional AI 原则。简化了 Prompt 生成...</li><li><a href="https://github.com/huggingface/diffusers/pull/7530#discussion_r1547822696">7529 不要为 cuda 设备禁用 autocast，由 bghira 提交 · Pull Request #7530 · huggingface/diffusers</a>：这个 PR 做了什么？修复了 #7529。在提交之前，这个 PR 修复了一个拼写错误或改进了文档（如果是这种情况，你可以忽略其他检查）。你阅读了贡献者指南吗？...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1224691527167643719)** (11 条消息🔥): 

- **扩展与采样效率分析**：[这篇文章](https://arxiv.org/abs/2404.01367)重点介绍的一项实证研究探讨了模型大小对 Latent Diffusion Models (LDMs) 采样效率的影响。与预期相反，研究发现，在相同的推理预算下，较小的模型通常比较大的模型表现更好。

- **寻找可扩展的爬虫技术**：一位成员询问了关于可扩展爬虫方法的研究，这些方法可以协助构建用于模型训练的数据集。然而，回复中并未引用具体的团队或资源。

- **揭秘赚取 5 万美元的奥秘**：一次幽默的交流，涉及一个 [Discord 版主封禁 GIF](https://tenor.com/view/discord-mod-moderation-ban-mod-ban-gif-9351874248631360646) 的链接，并猜测在 72 小时内赚取 5 万美元的秘密可能涉及当“毒骡”（drug mule），引用了一个与 MLM 相关的梗。

- **Twitter 上预热的新优化器**：Twitter 上讨论的一个 [新优化器](https://twitter.com/aaron_defazio/status/1775521495298588956) 备受期待，有望在该领域带来潜在的进步。

- **通过具体化增强视觉效果**：在讨论一篇 [arXiv 论文](https://arxiv.org/pdf/2404.01197.pdf) 时提到，使用包含更好空间描述的提示词（captions）对文本生成图像（t2i）模型进行微调，可以使生成的图像具有更好的空间一致性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.01367">Bigger is not Always Better: Scaling Properties of Latent Diffusion Models</a>：我们研究了 Latent Diffusion Models (LDMs) 的缩放特性，重点关注其采样效率。虽然改进的网络架构和推理算法已被证明能有效...</li><li><a href="https://tenor.com/view/discord-mod-moderation-ban-mod-ban-gif-9351874248631360646">Discord Mod Moderation Ban GIF - Discord mod Moderation ban Mod ban - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1224793707606184117)** (1 条消息): 

- **LangChain 的 Harrison Chase 将阐明 LLM 挑战**：诚邀参加与 **LangChain** 联合创始人兼 CEO **Harrison Chase** 的*独家活动*。他将讨论公司在从原型转向生产环境时面临的挑战，以及 **LangSmith** 如何帮助克服这些障碍。此次活动将于 4 月 17 日 18:30 在线举行。[在此注册](https://www.meetup.com/fr-FR/langchain-and-llm-france-meetup/events/300045589/)。
- **通过 LangChain 获取 LLM 框架趋势的内部信息**：LangChain 联合创始人 **Harrison Chase** 将分享他在使用 **LLMs (Large Language Models)** 开发上下文感知推理应用方面的专业知识。作为第三届 **LangChain and LLM France Meetup** 的一部分，本次演讲将探讨公司遇到的挑战及实施的解决方案。

**提到的链接**：<a href="https://www.meetup.com/fr-FR/langchain-and-llm-france-meetup/events/300045589/">Meetup #3 LangChain and LLM: Using LangSmith to go from prototype to production, mer. 17 avr. 2024, 18:30   | Meetup</a>：我们很高兴邀请到 LangChain 的联合创始人兼 CEO Harrison Chase 参加我们的第三届 LangChain and LLM France Meetup！不要错过这个难得的机会。

  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1224640668421459988)** (568 条消息🔥🔥🔥): 

- **Stable Diffusion 秘诀揭晓**：成员们正在讨论各种版本 Stable Diffusion 的性能。Forge 被强调为目前最快的 UI，DreamShaper Lightning (SDXL) 等模型也备受青睐。使用 RTX 3060 和 RTX 4080 等显卡的用户注意到，与 A1111 相比，使用 Forge 时速度显著提升，图像生成时间大幅缩短。
- **对 SD3 的期待与日俱增**：社区正急切等待 Stable Diffusion 3.0 的发布，预计发布时间在 3-5 周之间。不过有人指出，虽然 SD3 将改进文本渲染，但由于其局限性和模型大小，可能仍无法实现完美的拼写。
- **利用 SD 进行创意项目**：用户正在探索将 Stable Diffusion 用于各种创意尝试，例如为桌面 RPG 生成艺术图，或考虑通过图像进行叙事（可能是漫画或电影格式）。
- **技术攻关与技巧**：围绕图像生成过程中可能遇到的问题（如速度慢或一个提示词中的文本出现在另一个提示词中）展开了讨论，并建议利用特定的 Stable Diffusion 优化方案并尝试替代界面（如 Forge）。
- **即将推出的新模型和功能**：社区对 Sparse control net、SegMOE 和 Audiosparx 模型等新功能感到兴奋，并分享了有用的 GitHub 链接以及如何更好利用 AI 生成内容的技巧。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://app.leonardo.ai/">Leonardo.Ai</a>: 为您的项目创建具有前所未有的质量、速度和风格一致性的生产级视觉资产。</li><li><a href="https://tenor.com/view/anime-help-tears-cry-sad-gif-17104681">Anime Help GIF - Anime Help Tears - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://remix.ai/">Remix</a>: 创建、分享和混剪 AI 图像及视频。</li><li><a href="https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus">BFloat16: The secret to high performance on Cloud TPUs | Google Cloud Blog</a>: Google Cloud TPU 的高性能如何由 Brain Floating Point 格式（即 bfloat16）驱动</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Optimizations">Optimizations</a>: Stable Diffusion web UI。通过在 GitHub 上创建账号为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/3Frame">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://civitai.com/models/229002">ICBINP XL - v4 | Stable Diffusion Checkpoint | Civitai</a>: 如果你喜欢这个作品，请考虑请我喝杯咖啡 :) 在 Stable Horde 上免费使用此模型。这个期待已久的 ICBINP 后续模型是...</li><li><a href="https://forms.gle/9i4jM9BQu9bVVAAF6">Survey Form - 5day.io</a>: 作为一名刚入职几年的年轻专业人士，对于证明自己和寻找每个人都在谈论的神秘工作与生活平衡，总有一种隐约的焦虑。有时...</li><li><a href="https://www.youtube.com/watch?v=yvOXZ6SV2Rk">Stable Radio 24/7</a>: Stable Radio，一个 24/7 全天候直播流，专门播放由 Stable Audio 生成的曲目。在 stableaudio.com 探索模型并开始免费创作</li><li><a href="https://tenor.com/tKgaYjwJq16.gif">Cool Fun GIF - Cool Fun White cat - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/continue-revolution/sd-webui-animatediff/blob/master/docs/features.md#controlnet-v2v">sd-webui-animatediff/docs/features.md at master · continue-revolution/sd-webui-animatediff</a>: 适用于 AUTOMATIC1111 Stable Diffusion WebUI 的 AnimateDiff - continue-revolution/sd-webui-animatediff</li><li><a href="https://github.com/princeton-nlp/SWE-agent">GitHub - princeton-nlp/SWE-agent: SWE-agent: Agent Computer Interfaces Enable Software Engineering Language Models</a>: SWE-agent: Agent 计算机接口赋能软件工程语言模型 - princeton-nlp/SWE-agent</li><li><a href="https://www.reddit.com/r/3FrameMovies/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.</a>: 最强大且模块化的 Stable Diffusion GUI、API 和后端，具有图形/节点界面。 - comfyanonymous/ComfyUI</li><li><a href="https://github.com/kijai/ComfyUI-DiffusionLight">GitHub - kijai/ComfyUI-DiffusionLight: Using DiffusionLight in ComfyUI</a>: 在 ComfyUI 中使用 DiffusionLight。通过在 GitHub 上创建账号为 kijai/ComfyUI-DiffusionLight 的开发做出贡献。</li><li><a href="https://github.com/ZHO-ZHO-ZHO/ComfyUI-SegMoE">GitHub - ZHO-ZHO-ZHO/ComfyUI-SegMoE: Unofficial implementation of SegMoE for ComfyUI</a>: SegMoE 的 ComfyUI 非官方实现。通过在 GitHub 上创建账号为 ZHO-ZHO-ZHO/ComfyUI-SegMoE 的开发做出贡献。</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: 通过在 GitHub 上创建账号为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1224650862958346270)** (241 条消息🔥🔥): 

- **全栈开发人员推荐请求**：一位成员寻求优秀全栈开发人员的推荐，并邀请任何能提供帮助的人直接私信。

- **关于 Unsloth Enterprise 模型训练的咨询**：有人提问 Unsloth Enterprise 是否支持全量模型训练；回复澄清说支持，但加速倍数将比 FA2 快 2-5 倍，而不是 30-40 倍。

- **关于 Prompt 格式和实现的讨论**：成员们讨论了自定义 AI 模型和 Prompt 格式，特别提到了创建一个名为 Cortana 1.0 的模型，该模型是根据《士官长》视频游戏中的 AI 设计的。讨论中还涉及了寻找适合聊天模式的模型，以及利用正确的 Prompt 结构以实现高效运行的担忧。

- **AI 开发中的更新与成就分享**：他们分享了 [Daniel Han 的推文](https://twitter.com/danielhanchen/status/1775120334305607781)，反思了鉴于目前较短的开发时间，AI 在未来几个月内的潜力。此外还讨论了 Unsloth AI 的基准测试，包括其 'Ye' 模型在 SWE Bench 上取得的 12.29% 的成绩。

- **AI 性能的关注与优化**：多位成员询问了针对不同 AI 模型和平台的优化与支持。例如，讨论围绕 Unsloth 对 Galore 的支持、GPT 模型开源的可能性，以及在 Intel CPU 和 GPU 上加速本地 LLM 推理和微调的努力。一段包含 [GitHub 链接](https://github.com/intel-analytics/ipex-llm) 的交流强调了在特定硬件上加速 AI 推理的资源。此外，还讨论了 Unsloth 团队即将推出的潜在性能改进和更新。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1-uKmQzhh8ftxEdipiqGu4sVdRb8MgWv2?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://tenor.com/view/am-ia-joke-to-you-am-ia-joke-is-this-a-joke-do-you-think-this-is-funny-do-you-think-this-is-a-joke-gif-14191111">Am Ia Joke To You Is This A Joke GIF - Am IA Joke To You Am IA Joke Is This A Joke - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/i-aint-no-fool-wiz-khalifa-still-wiz-song-im-not-a-fool-im-not-an-idiot-gif-21822363">I Aint No Fool Wiz Khalifa GIF - I Aint No Fool Wiz Khalifa Still Wiz Song - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/jondurbin/airoboros-gpt-3.5-turbo-100k-7b">jondurbin/airoboros-gpt-3.5-turbo-100k-7b · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>：提速 2-5 倍，节省 70% 显存的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>：提速 2-5 倍，节省 70% 显存的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/intel/neural-speed">GitHub - intel/neural-speed: An innovative library for efficient LLM inference via low-bit quantization</a>：一个通过低比特量化实现高效 LLM 推理的创新库 - intel/neural-speed</li><li><a href="https://github.com/intel-analytics/ipex-llm">GitHub - intel-analytics/ipex-llm: Accelerate local LLM inference and finetuning (LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma, etc.) on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max). A PyTorch LLM library that seamlessly integrates with llama.cpp, HuggingFace, LangChain, LlamaIndex, DeepSpeed, vLLM, FastChat, ModelScope, etc.</a>：在 Intel CPU 和 GPU（例如带有 iGPU 的本地 PC，以及 Arc、Flex 和 Max 等独立 GPU）上加速本地 LLM 推理和微调（LLaMA、Mistral、ChatGLM、Qwen、Baichuan、Mixtral、Gemma 等）。一个与 llama.cpp、HuggingFace、LangChain、LlamaIndex、DeepSpeed、vLLM、FastChat、ModelScope 等无缝集成的 PyTorch LLM 库...</li><li><a href="https://github.com/toranb/sloth/blob/master/sftune.py">sloth/sftune.py at master · toranb/sloth</a>：使用 unsloth 的 Python sftune、qmerge 和 dpo 脚本 - toranb/sloth</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1546dvc/24gb_vram_on_a_budget/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1224808992300335278)** (12 条消息🔥): 

- **别具一格的小行星采矿公司**：[Open Asteroid Impact](https://openasteroidimpact.org/) 倡议是一种独特的小行星采矿方法，提议将小行星投向地球，而不是在太空中进行采矿。提供的链接展示了他们的 Logo，并强调了他们在获取太空资源时优先考虑安全和效率的目标。

- **对 Unsloth 网站设计的赞赏**：一位成员称赞了 Unsloth 的网站设计，指出该网站非常有吸引力。

- **预算有限下的创意**：由于预算限制，Unsloth 网站上的树懒图像是使用 Bing DALL-E 设计的。设计者还表示，打算最终委托 3D 艺术家创作一个形象统一的吉祥物。

- **通过努力实现设计一致性**：在回答关于设计统一性的询问时，Unsloth 网站设计者提到生成了数百张树懒图像，并在 Photoshop 中手动进行了精修。

- **为了速度选择 Bing DALL-E 而非 Hugging Face**：设计者在图像生成方面选择了 Bing DALL-E 而非 Hugging Face 的 DALL-E，因为其能够快速生成多张图像且拥有可用额度。

**提到的链接**：<a href="https://openasteroidimpact.org/">Open Asteroid Impact</a>：未找到描述

  

---

**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1224663029539668029)** (278 messages🔥🔥): 

- **训练过程中的评估详解**：成员们讨论了为什么默认情况下在 Fine-tuning 期间不添加评估数据集——添加它们会减慢过程。Training loss 是使用 Cross-entropy loss 计算的，Evaluation loss 也使用相同的指标。

- **使用 SFTTrainer 进行智能打包 (Packing)**：在使用 `SFTTrainer` 时，成员们分享了如何配置和优化训练，包括使用 `packing` 以及避免在 Gemma 模型中使用它，因为这可能会导致问题。

- **应对数据集大小挑战**：用户排查了与 OOM 错误和数据集大小相关的问题，包括讨论对大数据量使用 Streaming datasets，以及在使用 PyArrow 处理极大量数据时的挑战。

- **GGUF 转换困惑**：一位成员在将模型转换为 GGUF 格式时遇到问题并讨论了合适的方法，探讨了在转换脚本中进行手动架构调整的可能需求。

- **推理故障与 Unsloth 更新**：出现了一个 GemmaForCausalLM 对象导致属性错误的情况，在更新并重新安装 Unsloth 库后得到解决。一位成员提到使用 16-bit 模型推理导致了 OOM 错误，还有人在设置 Fine-tuning 环境时遇到了 Python.h 缺失的问题。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing#scrollTo=2eSvM9zX_2d3">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/qwp4w3hyb/deepseek-coder-7b-instruct-v1.5-iMat-GGUF">qwp4w3hyb/deepseek-coder-7b-instruct-v1.5-iMat-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://discuss.huggingface.co/t/adding-accuracy-precision-recall-and-f1-score-metrics-during-training/16419/2">在训练期间添加准确率、精确率、召回率和 F1 分数指标</a>：你好，你可以定义你的计算指标函数并将其传递给 Trainer。这里有一个计算指标的示例。使用 sklearn.metrics 定义准确率指标函数...</li><li><a href="https://huggingface.co/danielhanchen/model_21032024">danielhanchen/model_21032024 · Hugging Face</a>：未找到描述</li><li><a href="https://docs.wandb.ai/guides/integrations/huggingface">Hugging Face Transformers | Weights &amp; Biases 文档</a>：Hugging Face Transformers 库使 BERT 等最先进的 NLP 模型以及混合精度和梯度检查点等训练技术易于使用。W&B 集成增加了丰富的...</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">主页</a>：快 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/trl/main/en/sft_trainer#trl.trainer.ConstantLengthDataset">Supervised Fine-tuning Trainer</a>：未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GPTQ-Int4">Qwen/Qwen1.5-14B-Chat-GPTQ-Int4 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat/">deepseek-ai/deepseek-vl-7b-chat · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">主页</a>：快 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">主页</a>：快 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF">TheBloke/deepseek-coder-6.7B-instruct-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1224640434211389500)** (469 messages🔥🔥🔥): 

- **关于 Pro 模型和用法的讨论**：用户交流了使用不同 AI 模型（如 Claude 和 Haiku）阅读和解释 PDF 的见解。他们辩论了 Perplexity 的 Pro 功能和模型 Context windows 的优势，建议使用 "Writing" 焦点模式以获得详细回复，并启用 "Pro" 以获得更简洁准确的答案。一些人建议使用 Sonar 以获得更快的响应。

- **Perplexity 将引入广告？**：针对 Perplexity 计划引入广告的报道，用户表达了极大的担忧。用户引用了 Perplexity 首席商务官关于潜在赞助建议问题的言论，部分用户表示失望，并希望广告整合不会影响 Pro 用户的体验。

- **图像生成查询与可访问性**：用户询问了在桌面端和移动端生成图像的问题，得到的回复确认虽然移动端 App 不支持图像生成，但移动设备上的网页版支持该功能。

- **推荐链接与折扣**：用户分享了 Perplexity.ai 的推荐链接，并提到通过这些链接可以获得 10 美元的折扣。

- **技术支持与功能请求**：用户咨询了 API 限制和响应速度慢等技术问题，以及取消 25MB PDF 限制等功能更新。有建议使用 Sonar 以提高速度，并讨论了 Perplexity 是否已经取消了某些限制。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://groq.com/">GroqChat</a>: 未找到描述</li><li><a href="https://community.spiceworks.com/">未找到标题</a>: 未找到描述</li><li><a href="https://www.theverge.com/2024/4/1/24118154/perplexity-will-try-a-form-of-ads-on-its-ai-search-platform">Perplexity 将在其 AI 搜索平台上尝试广告形式。</a>: Perplexity 的首席业务官 Dmitry Shevelenko 告诉 Adweek，公司正在考虑在其平台中添加赞助的建议问题。如果用户继续搜索有关...的更多信息</li><li><a href="https://www.tomsguide.com/ai/apple-reveals-realm-new-ai-model-could-make-siri-way-faster-and-smarter">Apple 发布 ReALM —— 新的 AI 模型可能让 Siri 变得更快、更智能</a>: ReALM 可能是 Siri 2.0 的一部分</li><li><a href="https://www.adweek.com/media/gen-ai-search-engine-perplexity-has-a-plan-to-sell-ads/">生成式 AI 搜索引擎 Perplexity 计划出售广告</a>: 未找到描述</li><li><a href="https://www.theverge.com/2024/4/1/24118154/perplexity-will-try-a-form-of-ads-on-its-ai-search-platfo">Perplexity 将在其 AI 搜索平台上尝试广告形式。</a>: Perplexity 的首席业务官 Dmitry Shevelenko 告诉 Adweek，公司正在考虑在其平台中添加赞助的建议问题。如果用户继续搜索有关...的更多信息</li><li><a href="https://tenor.com/view/when-server-down-iceeramen-monkey-gif-23229726">当服务器宕机时的 Iceeramen GIF - 当服务器宕机时的 Iceeramen 猴子 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://fxtwitter.com/AravSrinivas/status/1775229252973334902?t=p2-h_dWeQhz6swoCVL66SA&s=19">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: 良好的氛围至关重要</li><li><a href="https://x.com/aravsrinivas/status/1775244089505845610?s=46">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: 迫不及待。</li><li><a href="https://fxtwitter.com/apostraphi/status/1775240129264730438?t=8XB64t2ExHGixP06DHvAHw&s=19">来自 Phi Hoang (@apostraphi) 的推文</a>: 本月发布周边商品。与 @Smith_Diction 合作，为 @perplexity_ai 打造。</li><li><a href="https://www.reddit.com/r/singularity/comments/1bp885i/claude_3_haiku_is_the_new_budget_king/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/docs/getting-started">pplx-api 入门指南</a>: 未找到描述</li><li><a href="https://www.gadgets360.com/ai/news/perplexity-ai-powered-search-engine-could-soon-show-ads-report-5357479">报告：AI 搜索引擎 Perplexity 可能很快向用户展示广告</a>: 根据报告，Perplexity 将在相关问题部分显示广告。</li><li><a href="https://slashdot.org/story/24/04/01/1653221/perplexity-an-ai-startup-attempting-to-challenge-google-plans-to-sell-ads">Perplexity，一家试图挑战 Google 的 AI 初创公司，计划出售广告 - Slashdot</a>: 一位匿名读者分享了一份报告：生成式 AI 搜索引擎 Perplexity 声称是 Google 的竞争对手，最近从 Jeff Bezos 等投资者那里获得了 7360 万美元的 Series B 融资...</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/11s44ry/discord_bo">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/11s44ry/discord_bot_public_releaseintroducing/?rdt=64126">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.quora.com">Quora - 分享知识、更好地了解世界的场所</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/AskReddit">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/NoStupidQuestions">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/explainlikeimfive">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/subreddits/search">搜索结果</a>: 未找到描述</li><li><a href="https://community.spiceworks.com">未找到标题</a>: 未找到描述</li><li><a href="https://discuss.codecademy.com">Codecademy 论坛</a>: Codecademy 的社区讨论论坛。</li><li><a href="https://hashnode.com">开始一个开发者博客：Hashnode - 自定义域名、子路径、托管/Headless CMS。</a>: 具有自定义域名、托管/headless CMS 选项的开发者博客。我们新的 headless CMS 为开发者工具公司简化了内容管理。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1224647504826601513)** (23 条消息🔥):

- **定制文章的魔力**：一位成员发现他们可以创建高度定制化的文章，突显了利用 Perplexity 深入研究特定主题的能力。
- **高效的新闻通讯调研**：Perplexity 协助用户迅速收集准确信息，显著加快了为其新闻通讯（Newsletter）订阅者制作“欢迎礼”的速度。
- **对 Fritz Haber 的深入探讨**：通过 Perplexity 搜索，一位成员深入研究了 **Fritz Haber** 的生平，揭示了他通过哈伯-博施法（Haber-Bosch process）对粮食生产做出的关键贡献、他在化学武器方面的复杂历史，以及他反对纳粹政权的道德立场。细节包括他获得诺贝尔奖的成就，以及围绕他的不幸家庭和历史境遇。
- **好奇心驱动的学习**：用户正利用 Perplexity 满足对各种主题的好奇心，从机器学习中的卷积（convolutions）到 **《希腊左巴》（Zorba the Greek）**，展示了该平台在处理各种咨询方面的多功能性。
- **随机森林的概念澄清**：多位成员寻求理解 **random forest classifier**（随机森林分类器），表明了社区内对机器学习算法的共同兴趣。

---

**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1224751637936079020)** (24 messages🔥): 

- **Perplexity API 暂不支持团队注册**：一位用户询问是否可以使用团队计划注册 Perplexity API，但已确认 **目前不支持团队注册**。

- **关于速率限制的困惑**：一位成员分享了对 **速率限制（rate limits）的困惑**，特别是在使用 sonar-medium-online 模型时。尽管遵守了 20req/m 的限制，他们 **仍然遇到 429 错误**；建议记录带有时间戳的请求，以确保速率限制被正确执行。

- **时效性准确结果的问题**：一位用户报告称，在使用 sonar-medium-online 模型询问当天的顶级科技新闻时，得到了不准确的过时信息。建议在 system prompt 中加入 **"Ensure responses are aligned with the Current date."**（确保回答与当前日期一致），以引导模型的结果。

- **澄清 Perplexity API 的功能**：有人寻求关于 Perplexity API 工作原理的澄清。要点包括生成 API key、在请求中将 key 作为 bearer token 发送，以及管理信用余额（支持自动充值）。

- **API 额度支付挂起问题**：一位成员表达了对购买 API 额度时出现问题的担忧——过程显示为 **"Pending" 状态** 且账户未更新。一名工作人员要求提供账户详情，以便在后台检查该问题。

---

**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1224726965886648423)** (76 messages🔥🔥): 

- **开源 SWE-agent 媲美 Devin**：一个名为 [SWE-agent](http://github.com/princeton-nlp/SWE-agent) 的新系统已经推出，在 SWE-bench 上拥有与 Devin 相似的准确率，其显著特点是开源。
- **苹果的研究暗示 AI 将超越 GPT-4**：苹果的一篇研究 [论文](https://arxiv.org/pdf/2403.20329.pdf) 讨论了一个名为 ReALM 的系统，暗示其能力超越了 ChatGPT 4.0，并与 iOS 18 的 Siri 开发同步。
- **Claude Opus 的性能困境**：对话中提到 Claude Opus 和 GPT-4 之间存在显著的性能差距，Opus 在某些任务（如“大海捞针”测试）中表现吃力。提到了一个 [提示工程互动教程](https://www.anthropic.com/index/claude-2-1-prompting) 以提高 Claude 的输出效果。
- **Stable Audio 2.0 发布**：[StabilityAI 发布了 Stable Audio 2.0](http://stableaudio.com)，这是一款能够生成高质量、完整长度音乐曲目的 AI，提升了音频 AI 领域的能力。
- **ChatGPT Plus 功能增强**：ChatGPT Plus 现在允许用户在网页或 App 端 **编辑 DALL-E 图像**，且最近的 iOS 更新包含了一个 **编辑对话提示词** 的选项。详细说明可在 [OpenAI 帮助页面](https://help.openai.com/en/articles/9055440-editing-your-images-with-dall-e) 查看。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://devday.replit.com/">Replit</a>: Replit 开发者大会直播</li><li><a href="https://x.com/officiallogank/status/1775222819439149424?s">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：很高兴分享我已加入 @Google，负责 AI Studio 的产品领导工作并支持 Gemini API。前方有很多艰巨的工作，但我们将努力让 Google 成为 AI 开发者构建应用的最佳家园。...</li><li><a href="https://9to5mac.com/2024/04/01/apple-ai-gpt-4/">Apple AI 研究人员夸赞其实用的端侧模型“大幅超越” GPT-4 - 9to5Mac</a>：Siri 最近一直在尝试描述在使用 CarPlay 或播报通知功能时在“信息”中收到的图像。在...</li><li><a href="https://x.com/sullyomarr/status/1774960295393538519?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Sully (@SullyOmarr) 的推文</a>：我使用 Cursor 作为我的 IDE，而 Claude 在使用 API 时似乎明显变差了。代码写一半、逻辑糟糕、编码风格极差。但在他们的官网上运行完美。还有人遇到这种情况吗？</li><li><a href="https://x.com/officiallogank/status/1775222819439149424?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：很高兴分享我已加入 @Google，负责 AI Studio 的产品领导工作并支持 Gemini API。前方有很多艰巨的工作，但我们将努力让 Google 成为 AI 开发者构建应用的最佳家园。...</li><li><a href="https://x.com/zswitten/status/1775187565219631155?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Zack Witten (@zswitten) 的推文</a>：六个月来我一直渴望大力推销这个，现在 Anthropic API 已经 GA，我终于可以了…… Prompt Engineering 互动教程！https://docs.google.com/spreadsheets/d/19jzLgR...</li><li><a href="https://blog.replit.com/code-repair">Replit — 为代码修复构建 LLM</a>：简介 在 Replit，我们正在重新思考开发者体验，将 AI 作为开发环境的一等公民。为了实现这一愿景，我们正将 AI 工具与我们的 I...</li><li><a href="https://x.com/gregkamradt/status/1727018183608193393?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Greg Kamradt (@GregKamradt) 的推文</a>：Claude 2.1 (200K Tokens) - 长上下文召回压力测试。我们都喜欢增加上下文长度——但性能如何？Anthropic 提供了 Claude 2.1 的早期访问权限，所以我...</li><li><a href="https://x.com/anthropicai/status/1732527908139552951?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Anthropic (@AnthropicAI) 的推文</a>：Claude 2.1 的 200K token 上下文窗口非常强大，但需要仔细的 Prompt 才能有效使用。了解如何让 Claude 在长文档中高保真地召回单个句子...</li><li><a href="https://x.com/jyangballin/status/1775114444370051582?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 John Yang (@jyangballin) 的推文</a>：SWE-agent 是我们用于自主解决 GitHub 仓库问题的新系统。它在 SWE-bench 上获得了与 Devin 类似的准确率，平均耗时 93 秒，而且它是开源的！我们设计了一个新的 Agent-co...</li><li><a href="https://x.com/ofirpress/status/1775226081575915661?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Ofir Press (@OfirPress) 的推文</a>：人们在问我们 Claude 3 在 SWE-agent 上的表现如何——并不好。在 SWE-bench Lite（测试集的 10% 子集）上，它的表现比 GPT-4 低了近 6%（绝对值）。而且它也慢得多。我们将...</li><li><a href="https://x.com/jd_pressman/status/1775295848509026659?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 John David Pressman (@jd_pressman) 的推文</a>：“Many Shot Jailbreaking”是我一段时间以来见过的主要实验室发布的最令人尴尬的出版物，我把 OpenAI 的 superalignment 文章也算在内。↘️ 引用 lumpen spac...</li><li><a href="https://x.com/StabilityAI/status/1775501906321793266?s=20">来自 Stability AI (@StabilityAI) 的推文</a>：隆重推出 Stable Audio 2.0 —— 一个能够根据单个 Prompt 生成高质量、完整曲目、具有连贯音乐结构、长达三分钟、采样率为 44.1 kHz 立体声的新模型。探索...</li><li><a href="https://x.com/teortaxestex/status/1775003753055228391?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：Opus 是一个极其强大的模型，是诗人，也是 @repligate 的天赐之物。它在事实性（胡编乱造或不知道）和指令遵循方面表现欠佳；GPT-4，甚至 Mistral 模型可能做得...</li><li><a href="https://x.com/_cgustavo/status/1775139142948552748?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w">来自 Gustavo Cid (@_cgustavo) 的推文</a>：我过去常乞求 LLM 提供结构化输出。大多数时候，它们理解任务并返回有效的 JSON。然而，大约有 5% 的时间它们做不到，我不得不编写胶水代码来避免...</li><li><a href="https://overcast.fm/+HaNOG0VjE/19:08">孩子们还应该学习编程吗？(Practical AI #263) &mdash; Changelog Master Feed &mdash; Overcast</a>：未找到描述</li>

<li><a href="https://x.com/_cgustavo/status/1775139142948552748?s=46&t=Tc6nPt_FP2Ybqya">来自 Gustavo Cid (@_cgustavo) 的推文</a>：我以前总是恳求 LLM 提供结构化输出。大多数时候，它们能理解任务并返回有效的 JSON。然而，大约有 5% 的时间它们做不到，我不得不编写胶水代码来避免……</li><li><a href="https://youtu.be/tVw3CwrN5-8?si=d_0EPgMCRL9mhva_">使用 DSPy 实现结构化输出</a>：不幸的是，大语言模型（LLM）并不会始终如一地遵循你给出的指令。当你构建 AI 系统时，这是一个巨大的问题……</li><li><a href="https://x.com/gblazex/status/1775558982645547236?s=20">来自 Blaze (Balázs Galambosi) (@gblazex) 的推文</a>：哇。当 OpenAI API 还停留在 Whisper-2 时，@AssemblyAI 发布了甚至超越 Whisper-3 的产品：比 Whisper-3 准确率高 13.5% + 幻觉减少高达 30% + 处理 60 秒只需 38 秒……</li><li><a href="https://www.youtube.com/watch?v=N1TEjTeQeg0">Geoffrey Hinton 教授 - “数字智能会取代生物智能吗？” Romanes 讲座</a>：AI 教父 Geoffrey Hinton 教授，CC, FRS, FRSC，于 2 月 19 日星期一在 Sheldonian 剧院发表了牛津大学年度 Romanes 讲座……
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1225158181316067422)** (356 条消息🔥🔥): 

- **DSPy 成为焦点**：LLM Paper Club 讨论了 **DSPy 框架**，并将其效用与 **LangChain** 和 **LlamaIndex** 进行了比较。重点强调了它为不同大语言模型（LLM）优化 **Prompt** 以及轻松**迁移**模型的能力，这一能力在 [DSPy 的 arXiv 论文](https://arxiv.org/abs/2310.03714)中得到了强调。

- **Devin 的亮相引发讨论**：提到了 **Devin** 的概念，这是一款拥有数千美元 OpenAI 额度支持其演示的 AI，引发了对其潜在演示用途的兴奋和期待。

- **探索 DSPy 的深度**：提出了关于 **DSPy 运行**和执行的问题，包括它是否可以**编译为更小的模型**、对调用进行**速率限制（Rate Limit）**以避免 OpenAI API 饱和，以及使用 `.save` 函数将优化结果保存到磁盘。

- **Prompt 优化的潜力**：人们对 **DSPy 优化单一指标**的能力以及是否可以将**多个指标**组合成复合分数进行优化感兴趣。讨论点强调了 DSPy 的 **teleprompter/optimizer** 功能，该功能不要求指标是可微的。

- **提出的实际应用**：俱乐部成员提出了 LLM 的各种**实际应用**，包括用于记录语音 API 对话的 **iOS 应用**、根据 URL 总结 arXiv 论文的**前端平台**、用于 **PII 检测的 DSPy 流水线**，以及重写 **DSPy 的文档**。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb">加入 Slido：输入 #code 进行投票和提问</a>：参与实时投票、测验或问答。无需登录。</li><li><a href="https://colab.research.google.com/drive/1KZR1sGTp_RLWUJPAiK1FKPKI-Qn9neUm?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2310.03714">DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines</a>：ML 社区正在迅速探索提示语言模型 (LM) 以及将其堆叠成解决复杂任务的流水线 (pipelines) 的技术。不幸的是，现有的 LM 流水线通常是...</li><li><a href="https://eugeneyan.com/writing/abstractive/">生成式摘要的评估与幻觉检测</a>：基于参考、上下文和偏好的指标，自我一致性，以及捕捉幻觉。</li><li><a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions">加入 Slido：输入 #code 进行投票和提问</a>：参与实时投票、测验或问答。无需登录。</li><li><a href="https://eugeneyan.com/writing/evals/#summ">有效与无效的 LLM 特定任务评估</a>：针对分类、摘要、翻译、版权重复和毒性的评估。</li><li><a href="https://eugeneyan.com/writing/evals/#summarization-consistency-relevance-length">有效与无效的 LLM 特定任务评估</a>：针对分类、摘要、翻译、版权重复和毒性的评估。</li><li><a href="https://www.spotery.com/">你是人类吗？</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.17764">1-bit LLM 时代：所有大语言模型都是 1.58 Bits</a>：最近的研究（如 BitNet）正在为 1-bit 大语言模型 (LLM) 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://hamel.dev/blog/posts/prompt/#dspy">- Fuck You, Show Me The Prompt.</a>：通过拦截 API 调用快速理解难以捉摸的 LLM 框架。</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/examples/knn.ipynb">stanfordnlp/dspy 项目 main 分支下的 dspy/examples/knn.ipynb</a>：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://github.com/seanchatmangpt/dspygen">GitHub - seanchatmangpt/dspygen：一个为 DSPy (Demonstrate, Search, Predict) 项目提供的 Ruby on Rails 风格框架，适用于 GPT、BERT 和 LLama 等语言模型。</a>：一个为 DSPy (Demonstrate, Search, Predict) 项目提供的 Ruby on Rails 风格框架，适用于 GPT、BERT 和 LLama 等语言模型。 - seanchatmangpt/dspygen</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy：DSPy：用于编程（而非提示）基础模型的框架</a>：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://x.com/HamelHusain/status/1774999027538612652?s=20">来自 Hamel Husain (@HamelHusain) 的推文</a>：@swyx 一个人 + 一群狂热粉丝
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1224830090073084036)** (4 条消息): 

- **自主 GitHub Issue 解决工具发布**：一个名为 **SWE-agent** 的新系统已发布，它在 SWE-bench 上的准确率与前身 Devin 相当，并提供了一个创新的 Agent-计算机接口。它处理任务的平均时间为 93 秒，并已在 [GitHub 仓库](http://github.com/princeton-nlp/SWE-agent) 开源。

- **Devin 的兴衰**：一段简单的评论强调了 AI 工具的飞速演进，在 SWE-agent 推出前仅仅两周，**Devin** 还被认为令人印象深刻。

- **探索可扩展的数据抓取**：一位成员询问了关于创建大型数据集的可扩展抓取方法的研究，回复表示人们对扩大数据集规模和提高质量都有着广泛的兴趣。

**提到的链接**：<a href="https://fxtwitter.com/jyangballin/status/1775114444370051582">来自 John Yang (@jyangballin) 的推文</a>：SWE-agent 是我们用于自主解决 GitHub 仓库中 issue 的新系统。它在 SWE-bench 上获得了与 Devin 类似的准确率，平均耗时 93 秒，而且它是开源的！我们设计了一个新的 Agent-co...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1224658774166343751)** (17 条消息🔥):

- **理解未见性能 (Unseen Performance)**：讨论涉及到一个奇特的现象，即一个 **80M 模型在未见性能上优于更大的模型**。人们对这一结果的有效性表示怀疑，并建议针对未见领域的评估应考虑较高的 **误差幅度 (margin of error)**。
- **奇特的 OOD 数据结果**：成员们评论了 **80M 模型** 在分布外 (OOD) 数据上得分极高的怪异现象，引发了对评估过程中潜在错误的推测。
- **探索 LLM 漏洞**：提到了由 @enkryptai 创建的红队测试套件，旨在检查 Large Language Models (LLMs) 的漏洞，包括对 **@databricks 的 DBRX** 和 MoE SSM LLM **Jamba** 的测试。分享的结果表明发现了一些重大问题（[关于 LLM 漏洞的推文](https://x.com/divyanshutwt/status/1775241719740535149?s=20)）。
- **Lollms & Ollama Server 教程**：推荐了一个 YouTube 教程，展示了如何安装和使用 **lollms 与 Ollama Server**，面向技术爱好者（[关于 lollms & Ollama Server 的 YouTube 教程](https://www.youtube.com/watch?v=RuQSQmolXGE)）。
- **中国的替代 AI 硬件**：讨论了中国芯片制造商 **Intellifusion**（云天励飞）推出的一款名为“DeepEyes”的 14nm AI 处理器，其价格显著低于同类 GPU。该处理器的 AI 性能和竞争性定价可能会挑战 AI 市场的高端硬件（[Tom's Hardware 关于 Intellifusion 的文章](https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus)）。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus">中国芯片制造商推出比 GPU 便宜 90% 的 14nm AI 处理器 —— 140 美元的芯片采用旧工艺规避美国制裁</a>：如果有规避制裁的方法，你知道中国一定会紧跟其后。</li><li><a href="https://x.com/divyanshutwt/status/1775241719740535149?s=20">来自 Divyanshu (@divyanshutwt) 的推文</a>：在 @enkryptai，我们构建了一个红队测试套件来识别 LLMs 的缺陷。最近，我们测试了 @databricks 的 DBRX 和 🐍Jamba（一种 MoE SSM LLM）的漏洞。得到了一些有趣的结...</li><li><a href="https://en.wikipedia.org/wiki/ISO_8601">ISO 8601 - 维基百科</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=RuQSQmolXGE">安装并释放 lollms 与 Ollama Server 的力量：一个有趣的技术教程 🚀</a>：🌟 嘿 YouTube 的家人们！🤓 我非常激动地向大家展示我的最新视频！在这个启发性的教程中，我将带你完成安装...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1224667853735858187)** (137 条消息🔥🔥): 

- **关于账号封禁和工具限制的查询**：一位用户询问了关于快速封禁（instaban）的问题，要求澄清是否允许同时拥有 API 和网页端账号。另一位用户提到，像 *worldsim* 这样的工具可以生成 Anthropic 禁止的内容。

- **分享 Jamba 模型微调经验**：Lhl 分享了周末使用 [shisa-v1 双语微调数据集](https://huggingface.co/datasets/augmxnt/ultra-orca-boros-en-ja-v1) 微调 *jamba 模型* 的结果，尽管结果“微乎其微”。提供了[训练脚本和配置](https://huggingface.co/shisa-ai/shisa-jamba-v1-checkpoint-4228)的直接链接，并承认 JA MT-Bench 的结果并不理想。

- **基础 NLP 论文咨询**：一位用户在读完 "Attention Is All You Need" 后，正在寻找 NLP 领域的基础论文。回复中包括建议观看 Andrej Karpathy 的所有 YouTube 视频。

- **在 Hugging Face 上分享模型的问题**：Mvds1 报告了由于 *safetensors.sharded* 键的元数据问题导致无法向 Hugging Face 上传模型的问题，并分享了一个来自讨论区的解决方法，即在 `SafeTensorsInfo` 定义中手动添加 `sharded: None` 参数。

- **讨论新型 LLM 压缩机制**：随后进行了一场关于 LLM 效率的理论和前沿方法的激烈讨论，涉及使用 Coq 等求解器来增强模型压缩，并引用了 Goertzel 关于使用超相容概率逻辑（paraconsistent probabilistic logic）实现 AGI 的著作。讨论的具体研究包括在 LLM 中内化 *PDLU: Proof Driven Logic Unit* 的概念，以及 (DSPy + Solver) Hylomorphic Recursor 在实现显著模型压缩方面的潜力。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.19928">DiJiang: Efficient Large Language Models through Compact Kernelization</a>: 为了减轻 Transformer 的计算负载，线性注意力机制的研究获得了显著动力。然而，注意力机制的改进策略通常需要...</li><li><a href="https://x.com/p00ssh/status/1775185708887539864?s=20">来自 poosh (e/λcc) (@p00ssh) 的推文</a>: attention 是你所需要的，匿名者</li><li><a href="https://huggingface.co/shisa-ai/shisa-jamba-v1-checkpoint-4228">shisa-ai/shisa-jamba-v1-checkpoint-4228 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.archives.gov/citizen-archivist">公民档案保管员 (Citizen Archivist)</a>: 总有一天，我们所有的记录都会上线。你可以帮助实现这一目标。你可以成为一名公民档案保管员——只需点击下面的选项即可开始。你可以标记它！为图像添加标签...</li><li><a href="https://tenor.com/view/unchained-foxx-silent-django-gif-4956511">Unchained Foxx GIF - Unchained Foxx Silent - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/sam_paech/status/1770497691593974021?s=46">来自 Sam Paech (@sam_paech) 的推文</a>: 使用 Claude-3-opus 作为评委的新型自动化创意写作基准测试：https://eqbench.com/creative_writing.html 更多信息：https://eqbench.com/about.html</li><li><a href="https://arxiv.org/abs/2012.14474">Paraconsistent Foundations for Probabilistic Reasoning, Programming and Concept Formation</a>: 文章认为，四值超协调真值（此处称为 "p-bits"）可以作为 AI 高度相关的概率逻辑形式的概念、数学和实践基础...</li><li><a href="https://youtu.be/Y94tw4eDHW0?si=cbH5-LV2dkXkkb0_&t=549">Programming Foundation Models with DSPy / Multivector Semantic Search with ColBERT - Omar Khattab</a>: Omar Khattab 是斯坦福大学的博士候选人，也是 AI/ML 领域的 Apple 学者。在这次对话中，Omar 解释了如何编写基础模型流水线...</li><li><a href="https://www.youtube.com/watch?v=ZYf9V2fSFwU">AI Pioneer Shows The Power of AI AGENTS - "The Future Is Agentic"</a>: Andrew Ng（吴恩达），Google Brain 和 Coursera 创始人，讨论了 Agent 的力量以及如何使用它们。加入我的通讯以获取定期 AI 更新 👇🏼https://www.matthewb...</li><li><a href="https://github.com/YuchuanTian/DiJiang">GitHub - YuchuanTian/DiJiang: "DiJiang: Efficient Large Language Models through Compact Kernelization" 的官方实现，一种基于 DCT 的新型线性注意力机制。</a>: "DiJiang: Efficient Large Language Models through Compact Kernelization" 的官方实现，一种基于 DCT 的新型线性注意力机制。 - YuchuanTian/DiJiang</li><li><a href="https://huggingface.co/datasets/aneeshas/imsdb-genre-movie-scripts">aneeshas/imsdb-genre-movie-scripts · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/fnlp/character-llm-data">fnlp/character-llm-data · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/Oobabooga/s/ApIzWEdZu7">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://huggingface.co/TroyDoesAI/MermaidMistral">TroyDoesAI/MermaidMistral · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/TheBritishLibrary/blbooks">TheBritishLibrary/blbooks · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/storytracer/US-PD-Books">storytracer/US-PD-Books · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1224666673450192916)** (34 条消息🔥):

- **探索 Agent 研究**：一位具有认知科学和强化学习背景的成员建议，通过校准 LLM 已知内容来进行高效探索是 Agent 研究中一个尚未被充分探索的领域。
- **Hermes 2 Pro 获得好评**：在测试 Hermes 2 Pro 后，一位用户称赞了该模型，特别是其 Function Calling 能力，在大型聊天会话中表现可靠，没有对不存在的工具产生幻觉。
- **多语言 LLM 训练说明**：针对有关多语言 LLM 训练的问题，会议澄清了 **Mistral** 主要在英语和部分欧洲语言上进行预训练，但微调训练数据中包含的非英语内容极少。该模型在其他语言中的连贯性可能归功于以英语为主的训练集中存在的语言片段。
- **用于 Function Calling 的 JSON 流式传输**：一位对 Function Calling 流式参数感兴趣的用户被引导至 oboe.js 库，该库提供了一种流式 JSON 解析技术。
- **Genstruct 7B 被推崇用于指令生成**：在关于生成不同领域合成数据的讨论中，成员们建议使用 Genstruct 7B，这是一个指令生成模型，旨在从原始文本语料库中创建有效的指令，作为构建用于微调的多样化指令数据的参考点。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B/discussions/7">NousResearch/Nous-Hermes-2-SOLAR-10.7B · Is added_tokens.json missing?</a>：未找到描述</li><li><a href="https://github.com/jimhigson/oboe.js/">GitHub - jimhigson/oboe.js: A streaming approach to JSON. Oboe.js speeds up web applications by providing parsed objects before the response completes.</a>：一种 JSON 流式处理方法。Oboe.js 通过在响应完成前提供已解析的对象来加速 Web 应用程序。- jimhigson/oboe.js</li><li><a href="https://arxiv.org/abs/2401.03462">Soaring from 4K to 400K: Extending LLM&#39;s Context with Activation Beacon</a>：长上下文的利用对 LLM 构成了巨大挑战，因为其上下文窗口大小有限。虽然可以通过微调扩展上下文窗口，但这会导致相当大的...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1224728581096345641)** (3 条消息): 

- **意向表达**：一位成员表达了热情，可能是对项目中正在进行的讨论或最近更新的回应。
- **数据集开发潜力**：同一位成员承认了构建数据集的潜力，暗示这与频道内讨论的工作或主题有关。
- **对时间限制的致歉**：该成员还因没有时间尝试可能与项目相关的内容而表示歉意。
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1224713092542365716)** (2 条消息): 

- **Huggingface 模型上传问题**：一位成员报告了上传到链（chain）时的问题，指出原因是 Huggingface 自动向模型元数据添加了 `safetensors.sharded = true/false` 键。该键不被 Huggingface Python 库识别，由于无法加载 `ModelInfo`，导致模型上传过程出现障碍。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1224642070178697216)** (7 条消息):

- **Scratchpad 在工作流中的定位**：*gabriel_syme* 讨论了在工作流中使用 Scratchpad 存储中间结果的价值，并提到一个具体的用例，即 `notes` 功能可以作为用户的 Scratchpad。
- **Glaive 发布 RAG 样本数据集**：*sahilch* 分享了 Glaive 新创建的样本数据集链接，该数据集有助于 RAG 数据的生成，可在 [Hugging Face 上的 GlaiveAI RAG 样本](https://huggingface.co/datasets/glaiveai/rag_sample)获取。
- **DiscoResearch 合成高级 RAG 数据**：来自 ellamind/DiscoResearch 的 *bjoernp* 强调了他们在高级 RAG 应用合成数据生成方面的工作，并表示有兴趣合作开发一个健壮且多样化的数据集。
- **增强功能的 RAG 愿景**：*bjoernp* 吹捧了将 RAG 与 Function calling 功能集成的潜力，使 LLM 能够管理查询分解（query decomposition）、多搜索协调和动态检索策略。
- **Ellamind 的早期 RAG 数据集及意图**：*rasdani* 介绍了 ellamind/DiscoResearch 的初步德语 RAG 数据集，并概述了他们为 RAG 能力的 Finetuning 和增强做出贡献的愿望，对 Nous Research 之前的工作表现出极大的热情。

**提及的链接**：<a href="https://huggingface.co/datasets/glaiveai/rag_sample">glaiveai/rag_sample · Datasets at Hugging Face</a>：未找到描述

---

**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1224672342492581958)** (88 条消息🔥🔥): 

- **基于 WorldSim 的创意竞赛**：成员们构思了一个 **WorldSim** 竞技平台，提议通过比赛在模拟世界中达到特定状态，拥抱复杂性，并讨论了规则和裁判的角色，表现出对游戏化模拟的浓厚兴趣。他们引用了一篇 [Twitter 帖子](https://x.com/karan4d/status/1768836844207378463?s=20) 作为 **WorldSim 系统提示词（system prompt）** 的来源，并分享了一个 [Pastebin 链接](https://pastebin.com/Gj7CpdSE) 以方便获取。
  
- **讨论中的 WorldSim 潜在功能**：构思了 WorldSim 的几项增强功能，例如 **Text-to-video 集成**（可能使用像 **ModelScope 的 MotionAgent** 这样的开源项目），以及用于加深模拟交互的 **持久化用户实体和数据**。一些提议的高级概念涉及 **对实际 Kernel 的读写权限**，为用户创造一种多元宇宙体验。

- **路线图规划与沟通**：讨论了为 WorldSim 创建 **社区驱动的 Roadmap 和 Newsletter**，以便让用户了解潜在的更新，并希望对 **WorldSim 的开发** 进行更清晰的沟通。有人建议使用视觉组织工具，并分享了 [Dwarf Fortress Roadmap](https://www.bay12games.com/dwarves/dev.html) 链接作为参考。

- **技术故障排除与增强**：改进 WorldSim 的建议包括：提高模拟器内的 **复制/粘贴便利性**、管理资源减速问题，以及 **保存/加载** 模拟状态。用户们自发提供了各种解决方案，分享了他们在 **Copilot** 和 **AI Dungeon** 等平台集成不同版本 WorldSim 的经验。

- **多样化的贡献与资源**：社区分享并赞赏了多种资源，例如 **Notion 上的 WorldSim 指令索引**，并进行了轻松的闲谈，欢迎其他用户来到“数字来世”。他们在交互过程中还遇到了用户个人资料被错误标记为 **垃圾信息（spam flags）** 的问题。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4">Notion —— 集笔记、任务、维基和数据库于一体的工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://lostpedia.fandom.com/wiki/Hanso_Foundation">Hanso Foundation</a>：Hanso Foundation 是由 Alvar Hanso 创立的组织，旨在通过研究保护人类生命和促进福祉的方法来“迈向更美好的明天”。它曾...</li><li><a href="https://en.wikipedia.org/wiki/Core_War">Core War - 维基百科</a>：未找到描述</li><li><a href="https://copilot.microsoft.com/sl/j7kIWW89XQ4">Microsoft Copilot：您的日常 AI 助手</a>：Microsoft Copilot 利用人工智能的力量，通过简单的聊天体验来提高生产力、释放创造力并更好地理解信息。</li><li><a href="https://arxiv.org/abs/2402.19459">随机时空导致的星系旋转曲线异常贡献</a>：我们考虑了一种量子引力的替代方案，其中时空度规被视为经典的，而物质场保持量子化。该理论的一致性必然要求...</li><li><a href="https://pastebin.com/Gj7CpdSE">Karan4D 的 WorldSim 系统提示词开源 - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://x.com/karan4d/status/1768836844207378463?s=20">来自 mephisto 🤡7 (@karan4d) 的推文</a>：我当然要开源 WorldSim。WorldSim 系统提示词和初始化对话：sysprompt：&lt;sys&gt;Assistant 今天处于 CLI 模式。人类正在直接与模拟器交互...</li><li><a href="https://www.bay12games.com/dwarves/dev.html">Bay 12 Games: Dwarf Fortress</a>：未找到描述
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1224648828888813569)** (170 条消息🔥🔥): 

- **LM Studio 与 Embedding 模型并不兼容**：用户澄清说 **LM Studio** 目前无法支持 Embedding 模型，并指出 *Embedding 模型尚未得到支持*。
- **在某些 CPU 上运行 LLM Studio 的问题**：讨论了在不支持 **AVX2 指令集** 的处理器上可能会出现 **LLM Studio** 安装问题，提到虽然有一个旧的测试版本可用，但已弃用且支持度不高。
- **模型加载错误排查**：几位成员在尝试将模型加载到 LM Studio 时遇到了错误，建议包括查看预设列表、修改配置文件，以及在特定的帮助频道发布系统规格以寻求进一步协助。
- **本地服务器的使用与稳定性担忧**：对话中有人称赞了本地服务器模式，而另一些人则表示在 LLM 性能下降或无法维持对话上下文方面遇到困难，建议调整 Context Size 并调查日志记录。
- **GPU 性能与多用户环境处理**：出现了关于在 LM Studio 中运行模型的硬件要求的询问，提到了卸载 GPU 层（offload GPU layers）的设置，并讨论了并行处理多个用户聊天请求的可行性，建议公司使用 **Nvidia DGX H100 服务器** 等企业级解决方案。

**提到的链接**：<a href="https://useanything.com/">AnythingLLM | 终极 AI 商业智能工具</a>：AnythingLLM 是为您的组织打造的终极企业级商业智能工具。为您的 LLM 提供无限控制、多用户支持、对内和对外工具，以及...

  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1224656872716505259)** (13 条消息🔥):

- **Databricks 下载困境**：一位成员询问如何将 **databricks/dbrx-instruct** 下载到 LM Studio 中，但被告知目前尚不支持且资源消耗巨大，甚至在拥有 128GB 内存的 M3 Max 上使用 Apple MLX 加载也宣告失败。
- **寻求 Hentai 动漫推荐模型**：一位用户询问是否有能够推荐 Hentai 动漫的模型，但被建议使用 **MyAnimeList (MAL)** 作为常规替代方案，并提供了链接：[myanimelist.net](https://myanimelist.net/)。
- **Hentai 推荐查询引发幽默**：社区对专门推荐 Hentai 动漫模型的需求反应幽默，赞赏该用户的胆量。
- **使用 System Prompts 训练 LLM**：讨论了使用带有复杂 System Prompt 的 LLM 输出结果来训练另一个 LLM 以继承该 Prompt 功能的可能性，这可以作为一种模型 Fine-tuning 的形式。
- **雇主模型的奇怪响应**：一位成员报告了其公司模型出现的奇怪行为，该模型始终提供与填字游戏相关的无关响应，暗示可能存在 Presets 问题。

**提到的链接**：<a href="https://myanimelist.net/">MyAnimeList.net - 动漫与漫画数据库及社区 </a>：欢迎来到 MyAnimeList，全球最活跃的在线动漫和漫画社区及数据库。加入在线社区，创建你的动漫和漫画列表，阅读评论，探索论坛，关注...

---

**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1225000508557754408)** (3 条消息): 

- **Embedding 模型咨询**：一位成员询问关于在 LM Studio 中使用 **embedding models** 的问题，并提到从 Hugging Face 下载了 **SFR embedding gguf model**。
- **目前不支持 Embedding**：作为回应，另一位参与者澄清说，目前 LM Studio 内部*不支持 Embedding 模型*。

---

**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1224685466885034054)** (69 条消息 🔥🔥): 

- **打破 LM Studio 的 SLI 迷思**：讨论澄清了使用两个 GPU **不需要** SLI，并且在 3090 之后的一代已经逐步淘汰。成员们确认在没有 SLI 的情况下，多 GPU 运行 LM Studio 性能良好，包括 2x 3090 和 2x 48GB RTX8000s 的配置。
- **P40 GPU 引起关注**：一位成员分享了一篇关于 Nvidia Tesla P40 性能的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/)，而另一位成员则详细介绍了一个使用三块 P40 的配置，能够高效运行 70B 参数模型。
- **LM Studio 中的性能惊喜**：用户报告了不同系统之间的显著性能差异，其中一位指出 **AMD 系统运行速度比预期慢**。然而，切换到 ROCm 预览版后，性能跃升至约 **65 tokens/sec**，这表明软件和驱动程序的选择对性能有巨大影响。
- **考虑升级 GPU 以获得更快的 LLM 响应**：一位考虑通过硬件升级来提高 LLM 性能的用户得到建议，称 **4090 GPU 和 PSU 升级**就足够了，无需更换 CPU。
- **对未来硬件价格的担忧**：讨论涉及了 TSMC 生产线发生[重大地震](https://www.bloomberg.com/news/articles/2024-04-03/tsmc-evacuates-production-lines-after-major-taiwan-quake)后对 GPU 和 Mac 价格的潜在影响，暗示这些商品可能会变得更加昂贵或稀缺。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-04-03/tsmc-evacuates-production-lines-after-major-taiwan-quake">Bloomberg - 你是机器人吗？</a>：未找到描述</li><li><a href="https://rentry.org/Mikubox-Triple-P40">Mikubox Triple-P40 配置</a>：来自 ebay 的 Dell T7910 “准系统”，包含散热器。我推荐 “digitalmind2000” 卖家，因为他们使用现场发泡包装，确保工作站完好无损地送达。你可以选择 Xe...</li><li><a href="https://www.techpowerup.com/gpu-specs/quadro-rtx-8000.c3306#:~:text=The%20card%20also%20has%2072,MHz%20(14%20Gbps%20effective).">NVIDIA Quadro RTX 8000 规格</a>：NVIDIA TU102, 1770 MHz, 4608 Cores, 288 TMUs, 96 ROPs, 49152 MB GDDR6, 1750 MHz, 384 bit</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>

---

**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1224718735068364962)** (3 条消息):

- **LM Studio 与 Autogen 的故障排除**：一位用户遇到了 Autogen 仅返回几个 token 就停止的问题。他们不确定 LM Studio 与 Autogen 之间的正确集成是否需要特殊步骤。
- **模型和 API 规范至关重要**：另一位成员暗示问题可能是由于不正确的模型名称以及在配置中可能忽略了 API 类型。他们建议检查 LM Studio 中的模型详情部分以获取准确信息。
- **API 类型至关重要**：已确认指定 API 类型对于 LM Studio 与 Autogen 的配合工作至关重要。
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1224968419774955583)** (3 messages): 

- **LM Studio 连接故障排除**：一位成员报告成功将项目与 **OpenAI GPT-4** 集成，但在将其连接到 **LM Studio** 以使用本地模型时遇到问题。尽管 LM Studio Server 显示有流式响应，但本地模型 "TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_S.gguf" 并没有返回响应。

- **CORS 可能是罪魁祸首**：针对连接问题，另一位成员建议启用 **CORS** 作为解决 **LM Studio** 与 **crewai** 通信问题的可能方案。

- **集成的有用资源**：为了进一步协助在 crewai 中实现 LM Studio，一位成员提供了一份有用的 [Medium 文章](https://medium.com/@tayyibali4300/implementing-lm-studio-in-crewai-270cc577acee) 指南。
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1225128391813103706)** (1 messages): 

- **DALL·E 图像现在可以在 ChatGPT 中编辑**：用户现在可以在 web、iOS 和 Android 平台的 **ChatGPT** 中直接编辑 **DALL·E** 图像。此外，在 GPT 中使用 DALL·E 创建图像时，现在可以获取*风格灵感*。
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1224671467820683334)** (173 messages🔥🔥): 

<ul>
<li><strong>Bing API 宕机数小时</strong>：用户报告 <strong>Bing API</strong> 宕机 12 小时，影响了与 DALL-E 和 Bing Image Creator 相关的服务。</li>
<li><strong>Android 上的应用访问问题</strong>：一位成员对无法在三星 Galaxy Note 9 上访问应用表示沮丧，提到了诸如 "request is not allowed"（请求不允许）之类的错误消息，以及该应用在 Google Play Store 中被列为不兼容。</li>
<li><strong>GPT 的模拟情感引发辩论</strong>：一场关于 LLM（如 GPT）是否能真正模拟情感的讨论展开，导致了与精神病态和 Eliza 效应的比较，并强调了当前 AI 模型缺乏“动力引擎”。</li>
<li><strong>OpenAI 承诺功能的缓慢推出</strong>：用户讨论了他们对 OpenAI 宣布新工具和功能（如记忆系统）但未跟进提供广泛访问权限（特别是对付费订阅者）的模式感到不满。</li>
<li><strong>界定模拟与感知之间的界限</strong>：聊天涉及了当前 AI 在模拟情感方面的局限性，参考了神经科学中类似的各种概念问题，并呼吁对意识有更精细的理解，以指导 AI 的开发。</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2310.17567">Skill-Mix: a Flexible and Expandable Family of Evaluations for AI models</a>: 随着 LLM 的角色从语言统计建模转向作为通用 AI Agent，LLM 评估应该如何改变？可以说，AI Agent 的一项关键能力是灵活地协作...</li><li><a href="https://arstechnica.com/tech-policy/2024/03/microsoft-compares-nyts-openai-lawsuit-to-movie-studios-trying-to-kill-the-vcr/">Microsoft argues Supreme Court’s VCR ruling should doom NYT’s OpenAI lawsuit</a>: 微软辩称：版权法“对 LLM 的障碍并不比对 VCR 的障碍大。”</li><li><a href="https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00567/116616/How-Much-Do-Language-Models-Copy-From-Their">How Much Do Language Models Copy From Their Training Data? Evaluating Linguistic Novelty in Text Generation Using RAVEN</a>: 摘要。目前的语言模型可以生成高质量文本。它们只是在复制以前见过的文本，还是已经学会了可泛化的语言抽象？为了区分这些...</li><li><a href="https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators">Simulators — LessWrong</a>: 感谢 Chris Scammell, Adam Shimi, Lee Sharkey, Evan Hubinger, Nicholas Dupuis, Leo Gao, Johannes Treutlein, 和 Jonathan Low 对草案的反馈…
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1224681472775032882)** (57 messages🔥🔥): 

- **DALL-E 的 Inpainting 预览**: 成员们正在讨论 DALL-E 的新编辑功能，该功能支持风格建议和局部重绘（Inpainting），可以编辑图像的特定部分。此功能似乎仅对 Plus 计划或更高级别的会员开放，且尚未完全推广，部分用户反映无法访问。

- **ChatGPT 性能讨论**: 社区对 GPT-4 和 Anthropic 的 Opus 等不同模型的性能持有不同意见。有人认为 GPT-4 在推理任务中表现更好，且逻辑连贯性更强；而另一些人则认为 Opus 在某些领域超越了 GPT-4。

- **利用自定义 GPTs**: 关于使用自定义 GPTs 与基础 ChatGPT 模型的辩论十分激烈。虽然一些人喜欢这些定制模型带来的效率，但一位用户更倾向于基础模型的灵活性和直接交互。

- **探索自定义 Prompt Engineering**: 讨论涉及了自定义 GPTs 在构建复杂 Prompt 方面的优势。用户们分享了如何利用构建器菜单将 Prompt 链接在一起的技术，并对比了自定义 GPTs 与指导基础 GPT 模型过程的便捷性。

- **Plus 计划福利**: Plus 计划用户正在询问如何使用图像编辑等新功能，因为该功能并非对所有人可见或正常运行。如果功能已推送至用户账户，选择图像后应会出现一个明显的编辑按钮。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1224771961897484319)** (11 messages🔥): 

- **寻求管理替代 Prompt**: 一位成员正在寻找用于管理任务的 Prompt，例如为中层管理和 C-suite 角色划分指令和绩效计划。讨论中未提供具体的建议或回复。
- **Numpy 新手寻求帮助**: Amayl_ 表达了在学习 Numpy 时的困难，提到这与其正在参加的机器学习课程有关。他们请求协助，但未提供具体练习的细节。
- **使用 ChatGPT 辅助练习**: Eskcanta 建议通过复制粘贴练习细节，向 ChatGPT（甚至是 3.5 模型）寻求帮助。该建议暂无后续进展。
- **Markdown 翻译难题**: Mat_adore 在翻译 Markdown 文本时遇到问题，阿拉伯语的回复翻译不一致或完全未翻译。他们分享了多个版本的 Prompt，目标是保留 Markdown 格式、链接和专有名词。
- **Prompt Engineering 挫败感加剧**: Mat_adore 多次调整了翻译 Prompt 以解决 Markdown 和语言转换问题，但仍面临挑战，并对结果的不一致表示沮丧。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1224771961897484319)** (11 messages🔥):

- **寻求管理类 Prompt 指导**：一位用户询问了针对**经理替代 (manager replacement)** 任务的有效 Prompt 策略，特别是针对中层和高层管理人员，涉及指令拆解和绩效计划。
- **机器学习课程 Numpy 求助**：一名成员就其机器学习课程相关的 **numpy** 练习寻求帮助，但未提供所遇问题的具体细节。
- **翻译故障排除**：一位用户报告了将 Markdown 内容翻译成不同语言时结果不一致的问题；对于阿拉伯语等某些语言，输出偶尔未翻译，导致用户感到沮丧。他们正在寻求一种 Prompt 修改方案，以确保在**保持 Markdown 格式的同时实现一致的翻译**。
- **标记保留挑战**：同一用户尝试了多次 Prompt 迭代以维持 Markdown 标记和正确的翻译，但仍然遇到问题——特别是在翻译文本中维持语言一致性和适当的 Markdown 格式方面。
- **寻求万无一失的翻译 Prompt**：为 **Markdown 内容**制作准确翻译 Prompt 的持续努力产生了好坏参半的结果，用户在实现翻译一致性和目标语言正确性，同时保留链接和 Markdown 标记方面仍面临挑战。
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1224686984484753510)** (148 messages🔥🔥): 

- **告别 GPU 巨头**：John Bridgman 从 AMD 退休，他因将驱动程序合并到 Linux kernel 主线而受到认可。George Hotz 评论了他的影响力，并对 AMD 的管理层及其处理 GPU 问题的方式表示怀疑，邀请 AMD 员工提供匿名见解以撰写潜在的博客文章。参见 [Phoronix](https://www.phoronix.com/news/AMD-Bridgman-Retires) 上的讨论和 [Twitter 线程](https://twitter.com/__tinygrad__/status/1775264921468764205)。
  
- **开源 GPU 的挑战与承诺**：AMD 团队在 GPU 驱动程序和开源承诺上的消极表现引发了辩论；George Hotz 强调了过去未兑现承诺的历史，并认为重大项目的取消可能是 AMD 所需的警钟。对于 [AMD 推文](https://twitter.com/amdradeon/status/1775261152987271614) 标志的开源方法存在谨慎乐观，但其承诺的可信度正受到审查。

- **Kernel 讨论与发行版演进**：对话转向讨论标记 Intel Xe 和 i915 驱动程序的各种 Kernel 版本的含义和支持挑战，以及从 Ubuntu 22.04 LTS 迁移到 24.04 LTS 的可能性。最后，George Hotz 表示一旦依赖项对齐，他将切换到 24.04 LTS，这与 [com.apple](https://apple.com) 从 20.04 迁移到 24.04 的时间一致。

- **Logo 重构贡献**：社区参与了 tinygrad 文档的更新，包括引入和调整适应浅色和深色模式的新 SVG Logo。George Hotz 提交了最终更改，并指出删除了“多余内容”，并对发现有助于更新的 'source media' 属性表示感谢。

- **NVIDIA 开源 GPU 驱动推测**：George Hotz 分享了他对开源 NVIDIA 驱动程序贡献的链接，澄清这不是 Nouveau 驱动程序，而是 [NVIDIA 的开源 GPU kernel 模块](https://github.com/NVIDIA/open-gpu-kernel-modules)。这引发了关于不同硬件制造商开源 GPU 驱动程序的优缺点和支持情况的讨论。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://www.phoronix.com/review/intel-xe-benchmark/)">推文：尝试并基准测试新的实验性 Intel Xe Linux 图形驱动程序 - Phoronix</a>：未找到描述</li><li><a href="https://www.phoronix.com/news/AMD-Bridgman-Retires">推文：AMD 长期开源 Linux 图形驱动程序倡导者退休 - Phoronix</a>：未找到描述</li><li><a href="https://tinygrad.org">">未找到标题</a>：未找到描述</li><li><a href="https://fedoramagazine.org/contribute-at-the-fedora-linux-test-week-for-kernel-6-8/">在 Fedora Linux Kernel 6.8 测试周中做出贡献 - Fedora Magazine</a>：宣布 Fedora Kernel 6.8 测试周并招募参与者</li><li><a href="https://lwn.net/ml/dri-devel/20221222222127.34560-1-matthew.brost@intel.com/">[RFC PATCH 00/20] 初始 Xe 驱动程序提交 [LWN.net]</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4044">[WIP] nimlgen 开发的 nv 驱动程序 · Pull Request #4044 · tinygrad/tinygrad</a>：未找到描述</li><li><a href="https://github.com/NVIDIA/open-gpu-kernel-modules">GitHub - NVIDIA/open-gpu-kernel-modules: NVIDIA Linux 开源 GPU 内核模块源码</a>：NVIDIA Linux 开源 GPU 内核模块源码。通过在 GitHub 上创建账号来为 NVIDIA/open-gpu-kernel-modules 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad">GitHub - tinygrad/tinygrad: 你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！ ❤️</a>：你喜欢 PyTorch？你喜欢 micrograd？你会爱上 tinygrad！ ❤️ - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</li><li><a href="http://www.catb.org/~esr/faqs/smart-questions.html">提问的智慧</a>：未找到描述
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1224687461242769478)** (28 条消息🔥): 

- **理解 Tinygrad 的束搜索（Beam Search）启发式算法**：一位成员询问 **tinygrad 的束搜索启发式算法** 是否与耗时有关，引发了讨论但未达成具体结论。
- **CommandQueue 揭示 Tinygrad 的功能**：**George Hotz** 指出 **CommandQueue** 是 **tinygrad** 内部 `run_schedule` 函数的替代品。为了深入了解，**alveoli3358** 分享了一份[关于新命令队列实现的教程](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md)。
- **内存优化查询引发技术评估**：一位成员通过询问在正向传播过程中是否可以释放内存（特别是针对具有反函数的激活函数）引发了讨论。他们引用了 [维基百科中的反函数规则](https://en.wikipedia.org/wiki/Inverse_function_rule) 来进一步阐述观点。
- **迈向更完善的 Tinygrad**：为了实现 1.0 版本，**George Hotz** 强调了 **tinygrad 对更多文档和教程** 的迫切需求。他还建议创建一个类似于 [“48 小时内自己写一个 Scheme”](https://en.wikibooks.org/wiki/Write_Yourself_a_Scheme_in_48_Hours) 的教程，帮助用户通过自己实现部分代码来进行学习。
- **社区参与和教程贡献**：成员们正积极为 **tinygrad 的学习资源** 做出贡献，并得到了其他用户的积极反馈。诸如编写教程和直播自己阅读快速入门指南等贡献，正在帮助用户（尤其是新手）理解和参与这项技术。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikibooks.org/wiki/Write_Yourself_a_Scheme_in_48_Hours">48 小时内自己写一个 Scheme - Wikibooks，开放世界的开放书籍</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Inverse_function_rule">反函数规则 - 维基百科</a>：未找到描述</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/commandqueue.md">tinygrad-notes/commandqueue.md at main · mesozoic-egg/tinygrad-notes</a>：通过在 GitHub 上创建账号来为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/me">me - 概览</a>：me 有 45 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://jax.readthedocs.io/en/latest/autodidax.html">Autodidax：从零开始的 JAX 核心 —— JAX 文档</a>：未找到描述
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1224693342177333318)** (93 条消息🔥🔥):

- **Open Interpreter 应用开发**：成员们讨论了开发一个与 Open Interpreter 通信的 **iPhone 应用** 的潜力，并引用了 [Jordan Singer 的 Twitter 帖子](https://x.com/jsngr/status/1774110742070882478?s=46&t=kwbSfLYCOimQnegJhHK_iA)。目前一个 **React Native 应用** 正在开发中，已完成约 40%，代码仓库已在 [GitHub](https://github.com/tyfiero/01iOS) 上共享以供社区协作。

- **Open Interpreter 的无障碍关注点**：一位成员强调了 **对话式 UI 层 (Conversational UI layer)** 对协助老年人和残障人士的重要性，旨在通过减少搜索、点击和数据管理工作来简化人机交互。

- **安全警报：Open Interpreter 的 X 账号可能被盗**：Open Interpreter 社区提醒用户不要点击来自疑似被盗的 Open Interpreter X 账号发布的异常帖子中的链接，并鼓励举报该账号以防止加密货币钱包被盗。

- **社区参与提醒**：Mike Bird 提醒大家参加四月的 House Party，并提供了 Discord 活动[链接](https://discord.gg/fjPmtRk8?event=1221828294811586572)，同时发起了关于 **Open Interpreter** 如何普遍改善人类境况的讨论。

- **交互式安装问题已解决**：一位用户询问了与 **chroma-hnswlib** 相关的安装问题，该问题已被引导至更合适的频道，强调了社区参与和共享技术故障解决方案的价值。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/fjPmtRk8?event=1221828294811586572">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。</li><li><a href="https://docs.openinterpreter.com/getting-started/setup">未找到标题</a>：未找到描述</li><li><a href="https://api.openinterpreter.com/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/jsngr/status/1774110742070882478?s=46&t=kwbSfLYCOimQnegJhHK_iA">jordan singer (@jsngr) 的推文</a>：✨ 通过手机远程与你的电脑对话，我称之为 Teleport</li><li><a href="https://github.com/tyfiero/01iOS">GitHub - tyfiero/01iOS</a>：通过在 GitHub 上创建账号来为 tyfiero/01iOS 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1224681843710754816)** (66 条消息🔥🔥): 

- **Android Termux 上的 OI**：分享了一个仓库，提供了使用 Termux 在 Android 设备上安装 OpenInterpreter 的说明，详见[此处](https://github.com/MikeBirdTech/open-interpreter-termux)。
- **Linux 服务器障碍**：多位用户在不同的 Linux 发行版上运行 01 服务器时遇到困难，问题涉及音频和 `portaudio19-dev` 等依赖项。
- **本地 STT 使用建议**：为了降低成本，建议在将文本输出发送给 OpenAI 之前，使用本地语音转文本 (STT) 代替云服务，可以利用 `Whisper.cpp` 等工具。
- **在 M5 Cardputer 上移植**：将 OpenInterpreter 移植到 M5 Cardputer 的工作正在进行中，并分享了更新和分支，包括一个向串口和屏幕同时发送消息的函数。相关的 GitHub 仓库可以在[此处](https://github.com/Clinteastman/c0mputer)找到。
- **GPT-4 成本担忧及替代方案**：关于使用 GPT-4 进行测试的高昂成本的讨论，引出了对 `gpt-4-turbo` 和 Claude 的 Haiku 等更具成本效益的替代方案的建议；OpenInterpreter 未来默认模型的选择正在考虑这些担忧。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://learn.microsoft.com/en-us/windows/package-manager/winget/">使用 winget 工具安装和管理应用程序</a>：winget 命令行工具使开发人员能够在 Windows 计算机上发现、安装、升级、移除和配置应用程序。</li><li><a href="https://scoop.sh/">未找到标题</a>：未找到描述</li><li><a href="https://github.com/OpenInterpreter/01/issues/219">Ubuntu 21+ 不受支持 [wayland] · Issue #219 · OpenInterpreter/01</a>：某些依赖项使用 x11，与 wayland 不兼容 https://github.com/Kalmat/PyWinCtl?tab=readme-ov-file#linux-notice https://github.com/asweigart/pyautogui/issues?q=is%3Aissue+is%3Aopen...</li><li><a href="https://github.com/Clinteastman/c0mputer">GitHub - Clinteastman/c0mputer: 将 open-interpreter 移植到 M5 Cardputer</a>：将 open-interpreter 移植到 M5 Cardputer。通过在 GitHub 上创建账户来为 Clinteastman/c0mputer 的开发做出贡献。</li><li><a href="https://github.com/MikeBirdTech/open-interpreter-termux">GitHub - MikeBirdTech/open-interpreter-termux: 在 Android 设备上安装 Open Interpreter 的说明。</a>：在 Android 设备上安装 Open Interpreter 的说明。 - MikeBirdTech/open-interpreter-termux</li><li><a href="https://github.com/m5stack/M5Unified/tree/develop">GitHub - m5stack/M5Unified 的 develop 分支</a>：M5Stack 系列的统一库。通过在 GitHub 上创建账户来为 m5stack/M5Unified 的开发做出贡献。</li><li><a href="https://git-scm.com/download/win">Git - 下载包</a>：未找到描述</li><li><a href="https://visualstudio.microsoft.com/visual-cpp-build-tools.">Microsoft C++ Build Tools - Visual Studio</a>：未找到描述</li><li><a href="https://ngrok.com/docs/getting-started/?os=linux">快速入门 | ngrok 文档</a>：本快速入门将使用 ngrok agent 将您的应用程序部署到</li><li><a href="https://github.com/rhasspy/piper/?tab=readme-ov-file#running-in-python">GitHub - rhasspy/piper: 一个快速、本地的神经文本转语音系统</a>：一个快速、本地的神经文本转语音系统。通过在 GitHub 上创建账户来为 rhasspy/piper 的开发做出贡献。</li><li><a href="https://dashboard.ngrok.com/get-started/setup/linux">ngrok - 一行命令上线</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1224640470337191996)** (2 条消息): 

- **对深度探索的兴奋**：一位成员对深入了解驱动大语言模型 (LLM) 的底层机制表达了热情。火箭表情符号强调了该成员对这些前沿知识的兴奋。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1224649158565036143)** (67 条消息🔥🔥): 

- **Tinystories 的性能限制**：讨论强调了 *Tinystories* 数据集在模型训练中的局限性，提到它在 5M 参数左右开始趋于饱和。成员建议改用 `minipile` 数据集，因为它大约大 4 倍，尽管处理起来更耗资源。

- **对 AI 竞赛的兴趣**：社区成员希望 EleutherAI 赞助团队参加 AI 竞赛，特别提到了利用 llema 模型、carperai 以及其他具有 RLHF 经验的合作伙伴的潜力。为了方便参与竞赛，有人建议在指定的聊天频道组建小组，并讨论计算资源补助的资格。

- **EAI 对越狱防御和不安全输出的立场**：分享了一篇 [arXiv 论文](https://arxiv.org/abs/2403.14725)，对现有的针对语言模型“越狱”攻击的执行机制的有效性提出了质疑。论文认为，为了制定更好的防御策略，明确定义不安全响应非常重要，并强调了后处理输出的充分性。

- **寻求研究工程职位的 PyTorch 面试技巧**：由于成员在寻求针对 PyTorch 知识的研究工程面试建议，大家一致认为自信地讨论自己的工作非常重要。技巧包括在行为面试题中依靠 STAR 法则，并掌握大多数候选人都能做对的中等难度编程题。

- **关于 AI 模型的公开评论**：分享了一个来自 [regulations.gov](https://www.regulations.gov/document/NTIA-2023-0009-0001/comment) 的链接，讨论了开源 AI 模型，其中 EleutherAI 的评论因其 LaTeX 格式而受到关注。一些成员对没有做出贡献表示遗憾，而另一些成员则对评论区中对开源模型的主流支持以及对散布恐惧行为的抵制表示赞同。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/overview">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>: 未找到描述</li><li><a href="https://www.regulations.gov/document/NTIA-2023-0009-0001/comment">Regulations.gov</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2403.14725">Jailbreaking is Best Solved by Definition</a>: 针对语言模型的“越狱（jailbreak）”攻击的兴起引发了一系列旨在防止输出不良响应的防御措施。在这项工作中，我们批判性地审视了两个阶段...</li><li><a href="https://github.com/UpstageAI/evalverse?fbclid=IwAR3IhfKfnHlkHWfmuAKDqcZZIP3mIZE5NfnsxBowp-ZuqiyVSndZfnYVTG4">GitHub - UpstageAI/evalverse: The Universe of Evaluation. All about the evaluation for LLMs.</a>: 评估的宇宙。关于 LLM 评估的一切。 - UpstageAI/evalverse
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1224720471774466088)** (53 条消息🔥): 

- **探索 LLM 鲁棒性思路**：分享了一个关于调查 LLM 安全过滤器鲁棒性的建议，引用了 **BlancheMinerva** 的一条推文，讨论了在微调数据中混入拒绝示例的潜力。该概念与提供的 [ArXiv 论文](https://arxiv.org/pdf/2402.18540.pdf) 中指出的当前研究一致。
  
- **监测开源 AI 立法**：重点分析了加州 SB 1047 法案对开源 AI 开发的影响，并提供了一封可供签署的公开信。对该法案的批评非常广泛，涉及 AI 领域的法律责任和效率问题，完整分析可以在[这里](https://www.context.fund/policy/sb_1047_analysis.html)找到。

- **AI 越狱（Jailbreaking）的新发现**：讨论了 Anthropic 关于“多样本越狱（many-shot jailbreaking）”的新研究，这种技术对包括他们自己在内的各种 LLM 都有效；同时还讨论了对该论文关于 In-context Learning 如何遵循幂律（power laws）发现的原创性的批评。完整论文可以在其[研究页面](https://www.anthropic.com/research/many-shot-jailbreaking)查看。

- **ChemNLP 首篇论文发表**：来自 OpenBioML.org 的 ChemNLP 项目的第一篇论文已在 [ArXiv](https://arxiv.org/abs/2404.01475) 上发布，这可能是 AI 驱动化学领域的重要一步。

- **讨论研究中的梯度符号**：随后进行了一场关于梯度符号的对话，建议根据梯度是指代模型参数还是其他内容，来决定使用偏导数符号还是 nabla 符号。讨论还涉及了报告中对不同版本 epsilon 符号的偏好。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.context.fund/policy/sb_1047_analysis.html">SB 1047 Analysis - Context Fund</a>: 未找到描述</li><li><a href="https://x.com/cem__anil/status/1775282571070591220?s=20">Cem Anil (@cem__anil) 的推文</a>: 我们最清晰的发现之一是，In-context Learning 通常作为演示数量的函数遵循简单的幂律。我们很惊讶没有发现这一点被明确地阐述在...</li><li><a href="https://swe-agent.com/">SWE-Agent</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.01475">Are large language models superhuman chemists?</a>: 大语言模型（LLMs）因其处理人类语言和执行未经显式训练的任务的能力而受到广泛关注。这与...</li><li><a href="https://x.com/blancheminerva/status/1774901289773584531?s=46">Stella Biderman (@BlancheMinerva) 的推文</a>: 众所周知，微调可能会无意中移除 RLHF 防护措施 https://arxiv.org/abs/2310.03693。能否通过在数据中混入拒绝示例来解决这个问题？这些是否重要...</li><li><a href="https://x.com/anthropicai/status/1775211248239464837?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Anthropic (@AnthropicAI) 的推文</a>: Anthropic 新研究论文：多样本越狱（Many-shot jailbreaking）。我们研究了一种长上下文越狱技术，该技术对大多数大语言模型有效，包括由 Anthropic 开发的模型以及许多其他...</li><li><a href="https://www.youtube.com/watch?v=rJIwO31uv5c">Louis Castricato - RLAIF, User Autonomy, and Controllability (Eleuther / Synthlabs)</a>: 来自 Cornell Tech 开源生成式 AI 研讨会的演讲。网站：https://www.louiscastricato.com/ 幻灯片：https://drive.google.com/file/d/14Qldg0E1c...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1224765514140876871)** (4 条消息):

- **抽象房屋难题 (Abstract House Conundrum)**：一位成员幽默地质疑，一栋房子如何能被认为介于**具体长颈鹿 (concrete giraffe)**和**抽象长颈鹿 (abstract giraffe)**之间。
- **保持冷静并耸肩 (Keep Calm and Shrug On)**：针对抽象/具体长颈鹿房子的难题，另一位成员提供了一个经典的互联网耸肩表情符号作为淡定的回答。
- **Neel Nanda 的 MATS 方向申请即将截止**：**Neel Nanda 的 MATS 方向**录取程序将在不到 10 天内关闭，附有 [Google Docs](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit#heading=h.y0ohi6l5z9qn) 上的申请详情链接。
- **神秘的 Twitter 提及**：一位成员分享了一条推文，但消息中未包含该推文的背景和内容。

**提到的链接**：<a href="https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit#heading=h.y0ohi6l5z9qn">Neel Nanda MATS Stream - Admissions Procedure + FAQ</a>：未找到描述

---

**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1224708347899412482)** (24 messages🔥): 

- **探索多语言生成式问答 (Multilingual Generative QA)**：参与者承认使用 **Chain of Thought** (CoT) 变体来提高多语言 QA 任务性能的潜力，并正在考虑 **MGSM** 以及 `nq_open` 或 `triviaqa` 等数据集。
- **生成直到任务引起兴趣 (Generate Until Task Arouses Interest)**：调试工作发现，使用 `generate until` 函数的任务并不多，已确认的有 gsm8k、bigbench 和 mgsm。随后，发现了一个包含实现 `generate until` 任务的完整列表。
- **排查 LM Eval 中的多选输出问题**：讨论了在使用 CSV 格式的评估数据集进行多项选择输出时，如何解决 "index out of range" 问题，暗示需要调整答案的索引。
- **不同 GPU 架构上的 CUDA 错误难题**：一位用户在 H100 GPU 上运行旧版 LM Eval Harness 时遇到 `CUDA error: no kernel image is available for execution on the device`，而 A100 GPU 运行正常。该问题被确定并非由 flash attention 引起。
- **CUDA 错误调查**：对 CUDA 错误的进一步调查表明，它不是由 `.contiguous()` 函数引起的，因为包含此操作的最小示例运行正常。建议检查 `context_layer` 所在的设备，以进一步排查问题。

**提到的链接**：<a href="https://github.com/search?q=repo%3AEleutherAI%2Flm-evaluation-harness+output_type%3A+generate_until&type=code">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

---

**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1224940664991649832)** (2 messages): 

- **弹性预训练框架 (Elastic Pretraining Frameworks)**：一位用户询问了能够在训练期间进行**弹性 GPU/TPU 调整**的预训练框架。另一位用户提供了使用 [PyTorch Elastic](https://pytorch.org/docs/stable/elastic/quickstart.html) 的解决方案，该方案允许作业以指定的重启次数容错运行，并能以弹性方式处理节点加入。

**提到的链接**：<a href="https://pytorch.org/docs/stable/elastic/quickstart.html">Quickstart &mdash; PyTorch 2.2 documentation</a>：未找到描述

---

**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1225114862640959720)** (3 messages):

- **设置可见性**：🤗 上的 Enterprise 组织现在可以选择将默认仓库可见性设置为公开、私有或默认私有。[查看推文了解更多信息](https://twitter.com/julien_c/status/1772688542289822073)。
- **Quarto 发布**：Quarto! 现在允许用户通过简单的命令 `use quarto publish hugging-face` 在 Hugging Face 上部署站点。详细说明可以在这些 [Twitter](https://twitter.com/gshotwell/status/1772661727856914720) 和 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7178422723503673345?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7178422723503673345%29) 帖子中找到。
- **新 HF Enterprise 页面和 HuggingChat 更新**：新的 HuggingFace Hub Enterprise 页面已上线，且 HuggingChat 助手现在支持自定义生成参数设置。探索 [新 Enterprise 页面](https://x.com/victormustar/status/1772742275744850137) 和 [HuggingChat 功能](https://x.com/victormustar/status/1772993404437205289)。
- **Hub 上的细粒度控制与 GGUF**：Enterprise 组织现在可以对每个 repo 进行细粒度的访问控制，并且 Hub 上已实现 GGUF 支持更新。在这一 [推文](https://twitter.com/Thom_Wolf/status/1770504033452573077) 中了解更多关于访问控制的信息，在这一 [状态帖子](https://x.com/lunarflu1/status/1775232743070220559) 中了解 GGUF 更新。
- **Datasets 2.18.0 发布**：Datasets 2.18.0 版本的发布带来了新功能、JSON 构建器支持，并确保了与 PyTorch 数据类型的兼容性。[探索新版本](https://github.com/huggingface/datasets/releases/tag/2.18.0)。
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1224649181352824843)** (70 条消息🔥🔥): 

- **寻找多语言 Image-Captioning 模型**：一位用户询问是否有支持包括葡萄牙语在内的多种语言的预训练图像字幕（Image-Captioning）模型，但未给出具体解决方案。
- **用于照片光照的 Stable Diffusion**：进行了一场关于使用 **Stable Diffusion** 均衡照片光照的讨论，一名成员指出应进行亮度（luma）归一化，而不是直接操作图像纹理。对话还涉及了批量处理具有各种光照偏差的图像的愿望。
- **NLP 项目中的精度目标**：成员们就 NLP 项目可接受的精度水平展开了讨论，一位用户询问 0.68 的精度对于第一个项目是否足够。另一位建议目标至少应达到 80% 的精度。
- **微调挑战与解决方案**：用户分享了与微调 **Mistral** 模型相关的经验和挑战，并提到了成功微调的版本，如 [Mistral Alpaca LoRA](https://huggingface.co/JoPmt/mistral_alpaca_lora)，以及在该过程中使用 Google Colab 的技巧。
- **摘要流水线简洁性调整**：一位用户寻求关于使用 Hugging Face 摘要流水线（summarization pipeline）生成更短摘要的建议。对话包括提示调整 `max_new_tokens` 而不是 `max_length` 以避免输出被截断，更多讨论被引导至 Hugging Face 的 Discord 频道。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/audio/ipynb/wav2vec2_audiocls.ipynb#scrollTo=9_TjMTIGL46g">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/JoPmt/mistral_alpaca_lora">JoPmt/mistral_alpaca_lora · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation">文本生成策略</a>: 未找到描述</li><li><a href="https://pytorch.org/blog/inside-the-matrix/?hss_channel=tw-776585502606721024">矩阵内部：可视化矩阵乘法、Attention 及更多</a>: 使用 3D 可视化矩阵乘法表达式、具有真实权重的 Attention Head 等。</li><li><a href="https://www.reddit.com/r/photoshop/comments/r7c2bh/evenout_lighting_for_a_tileable_texture/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/huggingface/cookbook">GitHub - huggingface/cookbook: 开源 AI 手册</a>: 开源 AI 手册。通过在 GitHub 上创建账号为 huggingface/cookbook 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth#installation-instructions---conda">GitHub - unslothai/unsloth: 快 2-5 倍、节省 70% 内存的 QLoRA &amp; LoRA 微调</a>: 快 2-5 倍、节省 70% 内存的 QLoRA &amp; LoRA 微调 - unslothai/unsloth
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1224855264096948304)** (2 条消息):

- **探索命令行世界**：一位频道成员分享了一个名为 [“Super User Do - Tinkering with Linux commands, Containers, Rust, and Groq”](https://www.youtube.com/watch?v=PKYPKRoCW2c) 的 YouTube 视频，介绍了如何使用命令行界面 (CLI) 导航计算机。
- **扩展的新视角**：讨论暗示了一个以**指数级**而非线性扩展的未知主题，尽管未提供具体细节。

**提到的链接**：<a href="https://www.youtube.com/watch?v=PKYPKRoCW2c">Super User Do- Tinkering with Linux commands, Containers, Rust, and Groq</a>：简要介绍如何通过所谓的“命令行界面”或“CLI”使用基本命令导航计算机。如何进行 update、upgrade、切进切出目录等...

---

**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1224700005152194570)** (5 条消息): 

- **文本生成领域的创新**：分享了一篇讨论 **IPEX-LLM 和 LlamaIndex** 的 Medium 文章，强调了它们在塑造文本生成和聊天应用未来方面的潜力。点击[此处](https://medium.com/ai-advances/unlocking-the-future-of-text-generation-and-chat-with-ipex-llm-and-llamaindex-c98b84cdb3a2)阅读全文了解这些进展。
- **测试 LLM 安全性**：开发了一套新的红队测试套件，用于测试 LLM 的漏洞，特别关注 **DBRX 和 Jamba**。分享的 [推文](https://x.com/divyanshutwt/status/1775241719740535149?s=20) 中提到了他们的调查结果详情。
- **教育视频：揭秘 GPT**：来自 3blue1brown 的 YouTube 视频，题为“But what is a GPT? Visual intro to Transformers”，对 Transformer 和 GPT 架构进行了引人入胜的解释。在[此处](https://www.youtube.com/watch?v=wjZofJX0v4M)预览教育内容，并向视频的支持者致谢。
- **Apple 宣称其 AI 优于 OpenAI**：一则简短通知透露，**Apple** 宣布其最新模型比 **OpenAI 的 GPT-4** 更强大，但未提供更多细节或支持证据。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/divyanshutwt/status/1775241719740535149?s=20">来自 Divyanshu (@divyanshutwt) 的推文</a>：在 @enkryptai，我们构建了一个红队测试套件来识别 LLM 的缺陷。最近，我们测试了 @databricks 的 DBRX 和 🐍Jamba（一种 MoE SSM LLM）的漏洞。得到了一些有趣的结...</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M">But what is a GPT? Visual intro to Transformers | Chapter 5, Deep Learning</a>：Transformer 及其先决条件介绍。赞助者可提前观看下一章节：https://3b1b.co/early-attention 特别感谢这些支...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1224691782516740227)** (14 条消息🔥): 

- **音乐创新引起共鸣**：创建了一个名为“音乐老虎机”的新 Gradio 应用，集成了 **Musiclang** 用于随机种子生成或输入和弦，并允许用户从社区制作的微调模型中进行选择。其结果是一种 **text2midi2audio** 的转换形式，在 [YouTube 视频](https://www.youtube.com/watch?v=p77U2eyJFPU) 中有详细介绍。虽然该应用是为测试微调模型而制作的，但它也可以作为音乐家的趣味乐器。

- **利用超图可视化化繁为简**：构建了一个用于可视化高维超图数据集的 **Space**，可处理多达 15 万行数据，作为一种理清复杂信息的方式。分享了指向该 [Space 的简短链接](https://huggingface.co/spaces/SauravMaheshkar/CornellTemporalHyperGraphDatasets)，以及对[原始集合](https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05)的引用和随附的 [Twitter 线程](https://x.com/MaheshkarSaurav/status/1775176529414086787?s=20)。

- **Octopus 2 以功能性吸引开发者**：具有函数调用（function calling）能力的模型 **Octopus 2** 的演示版首次亮相。尽管渲染需要 1500 秒，但该模型承诺了新的可能性，尤其是围绕端侧模型（on-device models）日益增长的热情，在 [Space](https://huggingface.co/spaces/Tonic/Octopus/) 中有详细展示。

- **本地曲调合成大获成功**：讨论强调，音乐模型在本地运行可能会有更好的可访问性和可用性，这与端侧模型更便捷的概念一致。

- **GPU 成本高昂激发了对 CPU 优化的乐观预期**：GPU 的高昂成本引发了关于在不久的将来 AI 和 ML 应用在 CPU 优化方面取得重大进展的期待。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/Tonic/Octopus/">Octopus - Tonic 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05">HyperGraph Datasets - SauravMaheshkar 的集合</a>：未找到描述</li><li><a href="https://x.com/MaheshkarSaurav/status/1775176529414086787?s=20">Saurav Maheshkar ☕️ (@MaheshkarSaurav) 的推文</a>：我目前正在研究 HyperGraph Representation Learning，过去几天一直在创建 @huggingface 集合，包括：👉 处理后的数据集 👉 论文 👉 @Gradio space...</li><li><a href="https://www.youtube.com/watch?v=p77U2eyJFPU">制作了一个音乐老虎机，然后用它创作了一首歌 - captains chair 21</a>：00:00 - 开始 01:35 - 制作音轨 08:28 - 音轨 我们的第一个 @HuggingFace space。这非常有趣。https://huggingface.co/spaces/thepatch/the-slot-...</li><li><a href="https://huggingface.co/spaces/thepatch/the-slot-machine">The Slot Machine - thepatch 创建的 Hugging Face Space</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1224734225853317231)** (3 messages): 

- **有效的 Batch Size 优化**：Seanb2792 提到，虽然计算成本可能相似，但可以在不使用额外 VRAM 的情况下增加有效 Batch Size。这特别有价值，因为较大的 Batch Size 可能会增强某些模型的性能。
- **Batch Size 影响模型性能**：在医疗数据的测试中，Huzuni 发现**较大的 Batch Size 通常会带来更好的性能**，即使改进是微小的或不显著的。
- **Batch Normalization 引起关注**：Huzuni 还观察到，根据他们最新的测试，累积超过两个 Batch 可能会对性能产生不利影响，这可能是由于 Batch Normalization 导致的。
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1224841862506872973)** (13 messages🔥): 

- **预算有限下的 LLM 微调**：一位用户正在探索如何在计算资源有限的情况下，基于 PDF 构建语言模型，倾向于使用 llama2、mistral、phi 等开源模型进行推理。有人询问了 LLM 模型的最低要求，提到 *phi-2* 在具有 16GB RAM 的 PC 上运行需要超过 10GB 的空闲空间。

- **Transformers 中的 KV Cache 查询**：一位成员询问了在 HuggingFace 中使用 KV Cache 的用例或示例，并链接到了 transformers GitHub 仓库中特定的 [Dynamic Cache](https://github.com/huggingface/transformers/blob/c9f6e5e35156e068b227dd9b15521767f6afd4d2/src/transformers/cache_utils.py#L76)。

- **修改 Tokenizer 中的特殊 Token**：讨论了在微调 LLM 时如何修改 Tokenizer 中的特殊 Token。一位成员提供了使用 `tokenizer.add_special_tokens(special_tokens)` 添加新特殊 Token 的解决方案，另一位建议直接修改 Tokenizer 的字典，但警告在 Tokenization 过程中可能会出现合并问题。

- **多节点微调问题**：一位用户在尝试使用 Docker、deepspeed 和 axolotl 进行多节点微调 llama2 时遇到了超时。尽管节点之间通信正常且 GPU 在其堆栈中可见，但微调过程在给定的 deepspeed 命令下冻结。

- **呼吁结构化的训练示例**：一位用户在训练 GPT2 进行文本摘要时遇到困难，遇到了 OOM 错误和验证指标停滞等问题。他们建议 HuggingFace 应该提供关于如何使用各种模型执行特定任务的结构化示例，以帮助用户进行训练。

**提及的链接**：<a href="https://github.com/huggingface/transformers/blob/c9f6e5e35156e068b227dd9b15521767f6afd4d2/src/transformers/cache_utils.py#L76">transformers/src/transformers/cache_utils.py at c9f6e5e35156e068b227dd9b15521767f6afd4d2 · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1225105529903251536)** (8 messages🔥): 

- **寻求增强型调节的 DiT**：一位用户询问了一种改进的 **DiT (Diffusion Image Transformer)**，它支持使用 Cross-attention 进行文本、图像或其他模态的调节。Hugging Face Diffusers 上唯一可用的 DiT 是按类别调节的，且原始团队的源代码未公开分享，如[论文链接](https://arxiv.org/html/2312.04557v1)所述。

- **公开 DiTs 的成本担忧**：一位成员指出，公开可用的 **DiTs** 是类别条件化（class-conditioned）的，因为与交叉注意力（cross-attention）方法相比，这种方式更具成本效益，这呼应了关于此类模型开销的讨论。

- **探索用于深度映射的扩散模型**：一位用户正在考虑修改 **Stable Diffusion (SD)** 以将立体图像转换为深度图，因为目前用于此类任务的最佳公开模型无法满足其挑战性需求。

- **Stable Diffusion 架构的潜在修改**：该用户询问是否可以使用超过三个通道的输入图像来微调 Stable Diffusion，探索在他们的任务中将 **LoRA** 或 **ControlNet** 与 Stable Diffusion 结合使用的可行性。

- **提倡修改 SD 而非从头开始训练**：针对该疑问，另一位参与者建议稍微修改 SD 架构以适应用户的需求，并表示从头开始训练应该是最后的手段。
  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1224778620971515994)** (1 messages): 

- **Gradio 发布 4.25.0 版本**：Gradio 发布了新更新，引入了 `gr.State` 变量的**自动删除**以实现更好的流量管理，以及针对浏览器标签页关闭的 **unload 事件**。该更新还具有适用于 ZeroGPU 的**延迟示例缓存**（使用 `cache_examples="lazy"`），修复了流式音频输出的 bug，并增强了 `gr.ChatInterface`，包括支持从剪贴板粘贴图像。
- **变更日志已准备就绪**：可以在 [changelog](https://gradio.app/changelog) 中查看 Gradio 4.25.0 的完整变更和修复列表。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1224663111391252520)** (104 messages🔥🔥): 

- **寻找持久化聊天历史解决方案**：一位成员询问了在与格式为“问题：答案”对的数据库聊天时保持持久上下文的技术，并对应该应用哪些方法表示不确定。
- **结构化工具验证查询**：围绕使用 LangChain 验证 `StructuredTool` 中的字段展开了讨论。对话提到了利用 Pydantic 的 `BaseModel` 和 `root_validator` 进行字段验证，并引用了特定的 [GitHub issues 和文档](https://github.com/langchain-ai/langchain/issues/8066)。
- **结构化工具中的异常处理**：成员们探索了在满足错误条件时如何在结构化工具中捕获并显示 `ValueError` 文本的策略，并参考了[相关方法的 GitHub issues](https://github.com/langchain-ai/langchain/issues/1358)。
- **将 LangChain 与外部 API 集成**：出现了关于将 LangChain 与 Azure API Management (APIM) 集成的问题，特别是使用 AzureOpenAI 获取结果，为此建议了一个指向特定 [GitHub issue](https://github.com/langchain-ai/langchain/issues/16930) 的故障排除链接。
- **创建一个连接数据库的预约机器人**：一位成员寻求使用 LangChain 和 JavaScript 创建机器人的帮助，该机器人不仅可以安排预约，还可以处理数据库中日期的存储和检索，从而引发了对 [Sequelize](https://github.com/sequelize/sequelize/tree/9e141880230a7f2a9a8c1e66a31f29fea7b5a65a) 和 [node-postgres](https://github.com/brianc/node-postgres/tree/master) 等库的推荐。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://smith.langchain.com/hub/wfh/web-voyager?organizationId=0ef50744-2e28-5e1f-8f50-1a7afa359cb9]">LangSmith</a>: 未找到描述</li><li><a href="http://localhost:8000.>">未找到标题</a>: 未找到描述</li><li><a href="https://python.langchain.com/docs/templates/openai-functions-agent#usage>).">openai-functions-agent | 🦜️🔗 Langchain</a>: 此模板创建了一个 Agent，它使用 OpenAI function calling 来传达其关于采取何种行动的决策。</li><li><a href="https://python.langchain.com/docs/guides/structured_output">[beta] Structured Output | 🦜️🔗 Langchain</a>: 让 LLM 返回结构化输出通常至关重要。这是</li><li><a href="https://python.langchain.com/docs/langgraph#how-to-guides>).">🦜🕸️LangGraph | 🦜️🔗 Langchain</a>: 下载</li><li><a href="https://js.langchain.com/docs/use_cases/chatbots/quickstart#quickstart-1>):">快速入门 | 🦜️🔗 Langchain</a>: 概览</li><li><a href="https://js.langchain.com/docs/integrations/llms/azure#llm-usage-example>).">Azure OpenAI | 🦜️🔗 Langchain</a>: Azure OpenAI 是一项云服务，旨在通过来自 OpenAI、Meta 等公司的各种预构建和精选模型，帮助您快速开发生成式 AI 体验。</li><li><a href="https://js.langchain.com/docs/integrations/llms/azure#using-the-openai-sdk>).">Azure OpenAI | 🦜️🔗 Langchain</a>: Azure OpenAI 是一项云服务，旨在通过来自 OpenAI、Meta 等公司的各种预构建和精选模型，帮助您快速开发生成式 AI 体验。</li><li><a href="https://api.python.langchain.com/en/latest/chains/langchain.chains.structured_output.base.create_structured_output_runnable.html#langchain.chains.structured_output.base.create_structured_output_runnable.">langchain.chains.structured_output.base.create_structured_output_runnable &mdash; 🦜🔗 LangChain 0.1.14</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/1358>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/sequelize/sequelize/tree/9e141880230a7f2a9a8c1e66a31f29fea7b5a65a">GitHub - sequelize/sequelize at 9e141880230a7f2a9a8c1e66a31f29fea7b5a65a</a>: 适用于现代 Node.js 和 TypeScript 的功能丰富的 ORM，支持 PostgreSQL（支持 JSON 和 JSONB）、MySQL、MariaDB、SQLite、MS SQL Server、Snowflake、Oracle DB (v6)、DB2 和 DB2 for IBM i。 - ...</li><li><a href="https://github.com/langchain-ai/langchain/issues/8406>):">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/16930>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/brianc/node-postgres/tree/master">GitHub - brianc/node-postgres: PostgreSQL client for node.js.</a>: 适用于 node.js 的 PostgreSQL 客户端。通过在 GitHub 上创建账号，为 brianc/node-postgres 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/use_cases/tool_use/tool_error_handling#tryexcept-tool-call>).">工具错误处理 | 🦜️🔗 Langchain</a>: 使用模型调用工具存在一些明显的潜在失败模式。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13662>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/8066>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/facebookresearch/fairseq/tree/nllb">GitHub - facebookresearch/fairseq at nllb</a>: 用 Python 编写的 Facebook AI Research 序列到序列工具包。 - GitHub - facebookresearch/fairseq at nllb</li><li><a href="https://huggingface.co/facebook/nllb-200-distilled-600M">facebook/nllb-200-distilled-600M · Hugging Face</a>: 未找到描述</li><li><a href="https://opennmt.net/CTranslate2/guides/transformers.html">Transformers &mdash; CTranslate2 4.1.0 文档</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1225024321806667787)** (2 条消息): 

- **Langserve 上的 CI 困惑**：一位成员就与 **[langchain-ai/langserve 的 pull request #580](https://github.com/langchain-ai/langserve/pull/580)** 相关的 CI 失败寻求帮助。他们表示已在本地使用 Python 3.10 测试了更改，所有测试均已通过。

- **Langserve Chat Playground 的新教程**：分享了一个完整的视频教程，解释了如何利用 Langserve 的新 **Chat Playground** 功能，特别是在无法开箱即用的情况下。这里是[视频链接](https://www.youtube.com/watch?v=stWiNP1o2_g)，其中还包括了 Langsmith 的展示以及描述中的最终代码。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=stWiNP1o2_g">The NEW Langserve Chat Playground with Agents | Coding Showcase</a>：在这次技术深度探讨中，我们将引导您进入令人兴奋的 LangChain 和 LangServe 框架世界。在 17 分钟内，我们将为您呈现一个全面的...</li><li><a href="https://github.com/langchain-ai/langserve/pull/580">WIP: Serve playground from correct route if nested APIrouters within one another by StreetLamb · Pull Request #580 · langchain-ai/langserve</a>：更新 playground 测试以检查 index.html 中正确的 playground 资源路径。#578
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1224759029935505522)** (7 messages): 

- **发布 Prompt-Breaking 挑战**：一位成员介绍了一个自动生成代码转换的工具，以确保生产环境中的**代码质量和标准**。征求**熟练的 prompters** 的反馈来测试该工具，并为此分享了链接 [GitGud LangChain](https://tinyurl.com/gitgud-langchain)。
  
- **CallStar AI 语音应用发布**：该成员宣布发布了多款 AI 语音应用，包括 **CallStar AI**、**Call Jesus AI**、**Call PDF AI**、**Call Tube AI**、**Call Website AI** 和 **Call Hacker News AI**。表达了对语音作为 AI 交互未来的热情，并提供了在 [Product Hunt](https://www.producthunt.com/posts/callstar)、[Reddit](https://www.reddit.com/r/SideProject/comments/1bumj6s/launching_callpdf_5_more_ai_voice_apps_today/) 和 [Hacker News](https://news.ycombinator.com/item?id=39914442) 上支持该项目的链接。

- **AllMind AI 亮相金融分析领域**：推出了一款名为 **AllMind AI** 的新型大语言模型，用于金融分析和研究。该 LLM 旨在通过在单一平台上提供洞察和全面的金融数据，彻底改变金融研究，并在 [AllMind Investments](https://allmindinvestments.com/) 和 [Product Hunt](https://www.producthunt.com/products/allmind-ai) 上提供了推广链接。

- **Galaxy AI 揭晓**：GalaxyAI 宣布了一项**免费 API 服务**，提供对高级 AI 模型的访问，包括各种版本的 GPT-3.5、GPT-4 和 **Gemini-PRO API**，全部兼容 Langchain 集成，并采用类似 OpenAI 的 API 格式。他们鼓励将该服务集成到项目中，并提供了一个试用服务的链接，尽管消息中未包含该 URL。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>: 未找到描述</li><li><a href="https://tinyurl.com/gitgud-langchain">GitGud</a>: 未找到描述</li><li><a href="https://callstar.ai/">CallStar</a>: 与角色和名人的 AI 语音通话</li><li><a href="https://www.kleinanzeigen.de/s-anzeige/mona-bild-repost/2724274253-246-1564?utm_source=other&utm_campaign=socialbuttons&utm_medium=social&utm_content=app_ios">Mona Bild repost</a>: 来自 tiktok 的知名图片 -，Mona Bild repost 在 Wuppertal - Elberfeld-West</li><li><a href="https://allmindinvestments.com/">AllMind AI</a>: 未找到描述</li><li><a href="https://www.producthunt.com/products/allmind-ai"> AllMind AI - Product Information, Latest Updates, and Reviews 2024 | Product Hunt</a>: AllMind AI 是一款专为金融分析和研究设计的全新大语言模型。该 LLM 通过为用户提供洞察并提供实时...来彻底改变金融研究。</li><li><a href="https://calljesus.ai/">Call Jesus</a>: 与耶稣的逼真 AI 语音聊天</li><li><a href="https://callpdf.ai/">CallPDF</a>: 呼叫任何 PDF - 逼真 AI 语音聊天</li><li><a href="https://calltube.ai/">CallTube</a>: 呼叫任何 YouTube 视频 - 逼真 AI 语音聊天</li><li><a href="https://callwebsite.ai/">Call Website</a>: 呼叫任何网站 - 逼真 AI 语音聊天</li><li><a href="https://callhackernews.com/">Call Hacker News</a>: Hacker News 的 AI 语音界面</li><li><a href="https://www.producthunt.com/posts/callstar"> CallStar - Realistic AI voice calls with characters, YT-videos &amp; PDFs | Product Hunt</a>: 下一代 AI 语音通话！与名人聊天，通过语音理解您的文档并探索灵性。通过最出色的 AI 语音让 AI 对话感觉真实且个性化。呼叫 PDF、YouTube...</li><li><a href="https://www.reddit.com/r/SideProject/comments/1bumj6s">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://news.ycombinator.com/item?id=39914442">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1225149749418659880)** (1 条消息): 

- **你的 LangChain 宇宙指南**：一位成员重点介绍了 [LangChain 快速入门指南](https://python.langchain.com/docs/get_started/quickstart)，该指南全面介绍了 **LangChain**，包括设置 **LangSmith** 和 **LangServe**，使用 prompt 模板、模型、输出解析器以及构建简单的应用程序。

- **遭遇 404 深渊**：在尝试运行涉及 `ChatOpenAI` 和 `ChatPromptTemplate` 的 **LangChain** 代码时，一位成员遇到了 `NotFoundError`，错误代码为 **404**，提示“未找到资源”问题。这一小插曲发生在成员在虚拟环境中执行程序期间。

**提到的链接**：<a href="https://python.langchain.com/docs/get_started/quickstart">快速入门 | 🦜️🔗 Langchain</a>：在此快速入门中，我们将向您展示如何：

  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1224668076906385429)** (38 条消息🔥): 

- **全面的 Mojo 文档备受赞誉**：[Mojo 文档](https://docs.modular.com/mojo/roadmap#cc-interop)被提到相当全面，提供了对未来实现的见解，包括 MAX Engine 和 C/C++ interop，这些预计将增强开发和效率。
- **Mojo 与数学变量名**：关于 Mojo 是否支持像 Julia 那样的数学变量名的问题得到了澄清，目前 Mojo 仅支持 ASCII 变量名，并遵循 Python 的变量命名约定，即以字符或下划线开头。
- **关于 Mojo 使用表情符号命名变量的辩论**：一场关于 Mojo 是否支持非传统变量名的讨论展开，确认了如果用反引号括起来，表情符号和其他符号可以用作变量名。
- **Mojo 的维基百科页面需要更新**：人们对 [Mojo 的维基百科页面](https://en.wikipedia.org/wiki/Mojo_(programming_language))状态不佳和信息过时表示担忧，最近的一次编辑纠正了关于 Mojo 仍是专有软件的误解。
- **代码片段故障排除**：有一场关于代码片段的故障排除讨论，其中 `listdir` 返回了一个引用列表，需要使用 `[]` 进行解引用才能使 `print` 正常工作，该解决方案已被找到并成功应用。

**提到的链接**：<a href="https://docs.modular.com/mojo/roadmap#cc-interop">Mojo🔥 路线图与注意事项 | Modular 文档</a>：我们的 Mojo 计划摘要，包括即将推出的功能和我们需要修复的问题。

  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1224789636891803670)** (3 条消息):

- **Modular 发布动态**：Modular 在其官方 Twitter 账号上分享了一条推文，可以在[这里](https://twitter.com/Modular/status/1775225130882564529)查看。
- **Modular 的另一条推文**：Modular 在其 Twitter 账号上发布了另一条[推文](https://twitter.com/Modular/status/1775549728400572660)。
- **Modular 继续更新 Twitter**：Modular 在其 Twitter 动态中又发布了一条[推文](https://twitter.com/Modular/status/1775583583530524987)。
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1224783263936024597)** (1 条消息): 

- **MAXimum Mojo 势头**：Modular **Mojo 24.2** 更新已发布，详情请见最近的[博客文章](https://www.modular.com/blog/whats-new-in-mojo-24-2-mojo-nightly-enhanced-python-interop-oss-stdlib-and-more)。此次发布对于采用 Mojo 的 Python 开发者尤为重要，提供了一系列新功能和增强。

**提到的链接**：<a href="https://www.modular.com/blog/whats-new-in-mojo-24-2-mojo-nightly-enhanced-python-interop-oss-stdlib-and-more">Modular: Mojo 24.2 的新特性：Mojo Nightly、增强的 Python 互操作性、开源标准库等</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo 24.2 的新特性：Mojo Nightly、增强的 Python 互操作性、开源标准库等。

  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1225139017121529927)** (4 条消息): 

- **提议在 ROS 2 上使用 Mojo**：一位成员建议将 [Mojo 支持集成到 ROS 2](https://github.com/ros2)（一种广泛使用的机器人中间件框架）中，由于 Mojo 的内存安全实践，这具有潜在优势。ROS 2 社区拥有 [原生 Rust 支持](https://github.com/ros2-rust/ros2_rust)，并正转向基于 Rust 的中间件，如 [Zenoh](https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds)。

- **ROS 2 中的 Rust 与 Python**：有人指出，尽管大多数 ROS 2 社区成员由于研究背景而偏好 Python，但 Rust 在性能和安全性方面提供了一个极具吸引力的替代方案。

- **为了性能重写 Python 代码**：该成员提到，虽然许多机器人系统最初为了方便使用 Python 编写，但在严肃的应用中，为了速度通常会用 C++ 重写。

- **Mojo 在 Nvidia Jetson 上的潜力**：有人指出，Mojo 可以更好地利用像 Nvidia Jetson 系列产品这样的硬件（这些产品在机器人领域的使用日益增加），而不像 Python 那样受限于全局解释器锁 (GIL)。

**提到的链接**：<a href="https://github.com/ros2-rust/ros2_rust">GitHub - ros2-rust/ros2_rust: ROS 2 的 Rust 绑定</a>：ROS 2 的 Rust 绑定。通过在 GitHub 上创建账号来为 ros2-rust/ros2_rust 的开发做出贡献。

  

---


**Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1225083116880662580)** (2 条消息): 

- **Docker 自动构建**：24.3 版本已准备好一个修复程序，旨在解决 **自动化 Docker 构建** 的方案。成员们对这一消息反应积极。
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1224657765461528577)** (30 条消息🔥): 

- **非平凡结构体（Non-Trivial Structs）的风险**：已确认由于共享指针问题，**@register_passable("trivial")** 不能用于具有内存分配的结构体，需要使用 **@register_passable** 才能正常工作。

- **开始 SIMD 搜索**：一位成员打算在 Mojo 中实现 SIMD Naïve Search，但不清楚如何实现 'found' 和 'SIMDcompare' 函数。另一位成员将其与用于 SIMD 操作的原生 Mojo 代码进行了比较，并指出 **[Mojo 的 SIMD 文档](https://arxiv.org/pdf/1612.01506.pdf)** 是一个起点。

- **顶层代码（Top-Level Code）暂时搁置**：关于在 Mojo 中引入顶层代码的讨论显示出其复杂性，目前尚无预计发布时间。关于“escaping”操作符页面缺失的问题已被提出，并已通知文档团队。

- **装饰器的困境**：Mojo 目前还无法实现自定义装饰器，因为它们是硬编码在编译器中的；分享了一个手动装饰函数的变通方法，同时也承认了这一局限性。

- **迭代中的等值检查之谜**：在一个场景中，成员尝试在 Mojo 的 List 迭代中检查字符串等值，这引出了一个澄清：由于 Mojo 处理 Reference 类型的方式，需要使用方括号 `x[]` 进行显式解引用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/decorators/always-inline">@always_inline | Modular Docs</a>: 将函数体直接复制到调用函数的函数体中。</li><li><a href="https://docs.modular.com/search?q=escaping+">Modular Docs</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1225176876079911101)** (1 messages): 

- **Logger 库获得更新**: Logger 库现在可以接受 **任意 args 和 kwargs** 来记录消息。此次更新增强了功能，允许在记录日志消息时输入变量信息，例如 `key=value` 或 `erroring=True`。
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1224847356277817344)** (7 messages): 

- **Mojo 在 1BRC 中落后于 Go**: 一位成员分享了他们在 Mojo 语言中进行 [One Billion Row Challenge](https://github.com/VMois/1brc-mojo) (1BRC) 的经验，指出在 MacBook Air M1 上经过优化后的性能约为 23 分钟，与在大约 96 秒内完成的 Go 实现相比，耗时显著更长。

- **寻找更快的 Dict**: 该成员对 Mojo 中 `Dict` 的性能表示担忧，认为它进行了过多的内存复制，并讨论了包括 SIMD 版本在内的潜在改进方案。

- **新的 Dict 实现即将到来**: 另一位成员提到他们有一个自定义的 `Dict` 实现，比 Mojo 标准库中的实现更快，为性能提升带来了希望。

- **与 Swiss Table 进行基准测试**: 当被问及与 Swiss Table 的对比时，一位成员回答说他们尚未对其进行基准测试，且此类基准测试需要用 C++ 或 Rust 编写。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/VMois/1brc-mojo/tree/main">GitHub - VMois/1brc-mojo: One Billion Row Challenge (1brc) in Mojo language</a>: Mojo 语言实现的十亿行挑战赛 (1BRC)。可以通过在 GitHub 上创建账号来为 VMois/1brc-mojo 的开发做出贡献。</li><li><a href="https://r2p.dev/b/2024-03-18-1brc-go/#:~:text=One%20Billion%20Row%20Challenge%20in%20Golang%20%2D%20From%2095s%20to%201.96s">One Billion Row Challenge in Golang - From 95s to 1.96s</a>: 在十亿行挑战赛中，任务是编写一个能够读取 10 亿行文件（约 13GB）、处理并汇总来自各个气象站温度读数的程序...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/1225099690790355036)** (2 messages): 

- **Max⚡ 和 Mojo🔥 24.2 发布**: 上周发布了 **Max⚡ 和 Mojo🔥 24.2**，同时标准库开源并推出了 nightly 构建版本。社区表现活跃，提出了约 50 个 Pull Requests 并合并了 10 个；感兴趣的用户可以在 [GitHub](https://github.com/modularml/mojo/tree/nightly/stdlib) 上探索并做出贡献。

- **探索 Mojo🔥 的最新动态**: 对于渴望深入了解最新更新和贡献的用户，Modular 提供了多项资源：*Mojo🔥 开源的下一个重大步骤*、**Mojo 发布博客**、**Mojo 24.2 更新详情**（包括 *Mojo nightly、增强的 Python 互操作性、开源标准库*）等。
  - 在 Modular 的博客中查找关于 [开源进展](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source) 的开发见解。
  - 通过阅读 [Mojo 发布博客](https://www.modular.com/blog/max-24-2-is-here-whats-new) 和关于 [24.2 更新内容](https://www.modular.com/blog/whats-new-in-mojo-24-2-mojo-nightly-enhanced-python-interop-oss-stdlib-and-more) 的详细说明来发现 Mojo 24.2 的新特性。

**提及的链接**: <a href="https://www.modular.com/newsletters/modverse-weekly-issue-28">Modverse Weekly - 第 28 期</a>: 欢迎阅读第 28 期 Modverse 周报，涵盖专题报道、Max 平台、Mojo 以及社区活动。

  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1224732356536045689)** (13 messages🔥):

- **GitHub 协作路径**：一位用户建议使用 Discord 进行一般性讨论，并将更具体的话题转移到 GitHub 上以进行协作。
- **将 Python stdlib 导入 Mojo**：一位开发者询问是否可以参考 Python 标准库来为 **Mojo stdlib** 贡献代码。回复指出，这种方法会引入对 CPython 解释器的依赖，这与实现独立二进制文件（standalone binaries）的目标相悖。
- **寻求 Mojo stdlib 开发指导**：一位希望为 **Mojo stdlib** 做出贡献的用户表示，虽然 `stdlib/docs/development.md` 等现有文档很有帮助，但发现开始实际开发仍具挑战性。
- **解决 stdlib 中的解析错误和测试失败**：一位用户遇到了解析错误和测试失败，包括 `FileCheck command not found` 错误。得到的指导是关于如何在 WSL 中定位 `FileCheck` 并将其添加到路径中，从而解决了该问题。
- **关于 Mojo 中 `Optional` 行为的讨论**：分享了一个 GitHub 链接，讨论在当前解引用（dereferencing）值的行为下，Mojo 标准库中的 `Optional` 是否可以为 `value()` 返回引用。

**提到的链接**：<a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/collections/optional.mojo#L117-L118).">mojo/stdlib/src/collections/optional.mojo at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。

---

**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1224643934186373151)** (91 条消息🔥🔥): 

- **超时关注与模型故障**：消息显示 **NOUSRESEARCH/NOUS-HERMES-2-MIXTRAL** 模型出现错误代码 524 的故障，一位成员提到使用 **TogetherAI's API** 时出现问题，表明这是一个上游问题。另一位成员提到备用模型 **Nous Capybara 34B** 可作为潜在替代方案。
- **使用历史问题测试 LLM**：一些成员正在讨论不同 LLM 在回答涉及二战日本将军的历史提示词时的准确性差异。**Isoroku Yamamoto**（山本五十六）被确定为正确答案，但 **claude**、**opus** 和 **haiku** 等模型表现参差不齐。
- **OpenRouter 的最大 Payload 大小**：讨论了 OpenRouter **4MB 最大 body size** 的限制，并确认该限制目前没有变通方法。
- **使用 AI 模型进行角色扮演**：成员们正在寻求关于使用各种 AI 模型（特别是 **Claude 3 Haiku**）进行角色扮演的建议。对话包括对模型进行越狱（jailbreaking）以及使用 few-shot 示例来提高性能的建议。
- **Prompt 资源的 Discord 服务器**：寻找 Prompt 示例和越狱 Prompt 的成员被引导至 **SillyTavern** 和 **Chub** 的 Discord 服务器，在那里他们可以找到诸如建议的 **pancatstack jailbreak** 等资源。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/abhishek/autotrain-mixtral-dgx-cloud-local">使用 AutoTrain 微调 Mixtral 8x7B</a>：未找到描述</li><li><a href="https://sillytavern.app/">SillyTavern - 面向高级用户的 LLM 前端</a>：未找到描述
</li>
</ul>

</div>

---

**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1224725406058283039)** (4 条消息): 

- **RankZephyr 在高级 RAG 中领先**：针对高级 RAG（检索增强生成），建议使用特定的 **rerankers**。IFTTT 建议使用像 **RankZephyr** 这样的 **LLM** 以获得更好的效果。一个名为 **RankLLM** 的开源 LLM 集合也因其在[重排序微调（finetuning for reranking）](https://twitter.com/llama_index/status/1775166279911186930)方面的卓越表现而被强调。

- **网络研讨会揭秘 AI 浏览器 Copilot 秘诀**：最近由 @dhuynh95 主持的网络研讨会为构建 **AI Browser Copilot** 提供了宝贵的见解，强调了 **prompt engineering pipeline** 以及 **KNN few-shot 示例**和 *vector retrieval*（向量检索）的重要性。更多详情可在 [LlamaIndex Twitter 页面](https://twitter.com/llama_index/status/1775264340465381536)查看。

- **通过时间敏感查询提升 RAG 性能**：**KDB.AI** 与 **RAG** 的集成允许进行 **hybrid searching**（混合搜索），结合了字面、语义和时间序列分析。这可以通过基于时间索引过滤相关性来实现更准确的结果，这对于季度财报等财务报告至关重要，正如[分享的代码片段](https://twitter.com/llama_index/status/1775269014849359925)所示。

- **介绍一款 AI 驱动的数字图书馆**：推出了一款专为**专业人士和团队**设计的全新 **LLM 驱动的数字图书馆**，承诺提供一套先进的知识组织系统。正如 [这条 LlamaIndex 推文](https://twitter.com/llama_index/status/1775537091272937933) 所述，该平台超越了传统的数据管理，提供了在一个自我管理的数字空间中创建、组织和标注数据的功能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://t.co/nbvRS0Cc9Q">IKI AI – Intelligent Knowledge Interface</a>：面向专业人士和团队的智能图书馆及知识助手。</li><li><a href="https://t.co/5uVy4hbtSw">Home - KDB.AI</a>：未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1224731920492134512)** (45 条消息🔥): 

- **PDF 索引困境**：一名成员就如何在不使用 llamaparse 的情况下索引一份 **2000 页的 PDF** 寻求建议，并提到目前的方法非常耗时。另一名成员建议增加 Embedding 模型的 `embed_batch_size`，但随后被认为无效，这表明需要其他的替代策略。

- **理解 qDrant 锁定文件**：一位用户遇到了 **qDrant** 在运行 **IngestionPipeline** 后不释放锁定的问题，并询问社区这是 LlamaIndex 还是 qDrant 特有的问题。该用户未得到确切答复，凸显了在该问题上集体经验的缺失。

- **讨论 HuggingFace API 限制**：关于在使用带有 Token 的 **HuggingFaceInferenceAPIEmbedding 和 HuggingFaceInferenceAPI** 时可能存在的速率限制和费用问题存在困惑。虽然一名成员最初认为没有速率限制，但另一名成员随后确认了速率限制错误以及 Hugging Face 收费的可能性。

- **替代模型的集成挑战**：一位用户尝试将名为 **"llama2"** 的模型集成到 LlamaIndex Agent 中，并被建议使用 Ollama 类，该类使用 REST API 进行交互。分享了有用的文档，并详细讨论了与 **Ollama** 的集成过程。

- **结合递归查询引擎的 RAGAs**：引发了一场关于 RAGAs 缺乏递归查询引擎文档的讨论，导致人们意识到 **langchain 和 ragas** 之间可能存在的问题，并强调了在该领域需要更清晰的指导或修复。

**提到的链接**：<a href="https://docs.llamaindex.ai/en/stable/api_reference/llms/ollama/">Ollama - LlamaIndex</a>：未找到描述

  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1224699971501297745)** (7 条消息): 

- **探索文本和聊天生成的未来**：一篇题为《利用 IPEX-LLM 和 LlamaIndex 开启文本生成和聊天的未来》的 Medium 文章探讨了文本生成的进展。文章链接见 [Unlocking the future of text generation](https://medium.com/ai-advances/unlocking-the-future-of-text-generation-and-chat-with-ipex-llm-and-llamaindex-c98b84cdb3a2)。

- **分享 RAG 应用分步教程**：一名成员分享了一个关于使用 **LlamaIndex, Pinecone 和 Gemini Pro** 构建 RAG 应用的 YouTube 教程链接。教程观看地址：[How to build a RAG app using Gemini Pro](https://youtu.be/B9mRMw0Jhfo)。

- **RAG 教程获得社区支持**：另一名成员对之前分享的 RAG 应用视频教程表示热烈欢迎，显示了社区对此类教育内容的支持。

- **比较多步任务的 Fine-Tuning 与 Few-Shot Learning**：一名成员询问有关比较 **Fine-Tuning 与 Few-Shot Learning** 在提高模型执行多步 Agent 任务能力的调研，考虑了两种方法：在 Prompt 中包含推理示例，或者构建数据集并进行 Fine-Tuning。

- **寻求本地文本增强解决方案**：一名成员征求有关构建本地应用程序的技术建议，旨在纠正错误而不改变原始含义，目标是避免使用像 ChatGPT 这样的第三方服务。

**提到的链接**：<a href="https://youtu.be/B9mRMw0Jhfo">How to build a RAG app using Gemini Pro, LlamaIndex (v0.10+), and Pinecone</a>：让我们来谈谈如何使用 LlamaIndex (v0.10+)、Pinecone 和 Google 的 Gemini Pro 模型构建一个简单的 RAG 应用。如果你刚刚开始，这是一个分步教程...

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1224746398155542569)** (48 条消息🔥):

- **Mistral Office Hour 提醒**：频道收到关于 **Mistral office hour** 开放提问的通知。
- **数据集统一挑战**：一位成员描述了统一总计数百 GB 的众多数据集的复杂过程，涉及文件对齐等问题。他们目前使用 TSV 文件和 pickle 格式的索引数据进行快速检索，但理想的解决方案和基础设施仍在考虑中。
- **Runpod Serverless vLLM 经验**：围绕 **Runpod** 和 serverless **vLLM** 的讨论包括设置和操作方面的挑战。GitHub 上的共享资源展示了如何部署 [large language model endpoints](https://github.com/runpod-workers/worker-vllm)。
- **评估 RP-LLM**：一位成员介绍了 **Chaiverse.com** 作为一个快速获取 RP-LLM 模型反馈的平台，并强调该平台已经评估了 1k 个模型和 5k 个变体。他们邀请大家对该服务提供反馈，并讨论了非公开评估数据集在防止“针对测试集训练（training to the test）”方面的优势。
- **Qwen Mow 对比 Jamba**：关于 AI 模型偏好的趣味辩论（如 'qwen mow' 对比 'jamba'）暗示了对不同模型在 RAG 或通用场景下有效性的不同看法。此外还有关于需要更多训练数据和集体投资以获得更好服务器的幽默调侃。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://bit.ly/3TFIsKt">Salad - GPU Cloud | 10k+ GPUs for Generative AI</a>：节省高达 90% 的云账单。轻松部署 AI/ML 生产模型。每美元可多生成 600% 的图像和 10 倍的推理。立即免费试用 SaladCloud。</li><li><a href="https://github.com/runpod-workers/worker-vllm">GitHub - runpod-workers/worker-vllm: 用于提供大语言模型端点服务的 RunPod worker 模板。由 vLLM 驱动。</a>：用于提供大语言模型端点服务的 RunPod worker 模板。由 vLLM 驱动。 - runpod-workers/worker-vllm
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1224741705954099303)** (5 messages): 

- **文档更新获赞**：Axolotl 的更新文档因新外观受到称赞，但有人提出了关于缺失 **Table of Contents**（目录）的问题，该目录本应包含 *Axolotl supports*、*Quickstart*、*Common Errors* 等部分，如[此处](https://openaccess-ai-collective.github.io/axolotl/)所示。
- **目录已修复**：一位成员修复了缺失的 **Table of Contents**，并通过 [GitHub commit](https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a) 确认了更新。
- **注意到目录差异**：观察到 README 中的目录与 Markdown 标题不完全匹配，这意味着需要进一步清理以确保一致性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/">Axolotl</a>：未找到描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a">fix toc · OpenAccess-AI-Collective/axolotl@5760099</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1224713792160534689)** (2 messages): 

- **模型表现异常**：一位成员表示沮丧，某些模型在配置与其他正常运行的模型相同的情况下却卡住了。

- **寻求高分辨率图像**：另一位成员询问了关于抓取大量 **4K 和 8K 图像**资源的建议。
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1224644605455372309)** (36 messages🔥):

- **Llamafile Builds for Windows ARM**: 要为 Windows ARM 构建 **llama.cpp**，你需要从源代码进行编译，因为 Windows ARM 目前不在支持范围内。
- **Mixtral's Math-Riddle Solving Capability**: **`mixtral-8x7b-instruct-v0.1.Q4_0.llamafile`** 可以简洁地解决数学谜题，但为了在没有 Hallucinations 的情况下回忆冷门事实，需要 **`Q5_K_M`** 或更高版本。在 [Hugging Face](https://huggingface.co/jartine/Mixtral-8x7B-Instruct-v0.1-llamafile/tree/main) 上查找相关详情。
- **Optimizing GPU Performance with TinyBLAS**: 使用 **llamafile** 时，GPU 性能可能会有显著差异，通常取决于供应商提供的线性代数库。提供了一个 *`--tinyblas`* 标志，可以在不需要额外 SDK 的情况下启用 GPU 支持，尽管其性能可能因具体的 GPU 型号而异。
- **Windows Executable Formats for ARM**: Windows on ARM 通过包含 Arm64 和 Arm64EC 代码的 ARM64X 二进制文件支持 PE 格式。ARM64EC 缺乏对 AVX/AVX2 的模拟，这给通常需要 SVE 或 NEON 等指令的 LLM 操作带来了挑战。更多详情请参阅 Microsoft 的 [文档](https://learn.microsoft.com/en-us/windows/arm/arm64x-pe)。
- **Compiling Issues on Windows for Llamafile**: 鼓励 Windows 用户在 Linux、Mac 或 BSD 上构建 **llamafile**，因为在 Windows 上设置 Cosmopolitan 开发环境非常复杂，如 Cosmopolitan [issue #1010](https://github.com/jart/cosmopolitan/issues/1010) 中所述。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html">Install HIP SDK — HIP SDK installation Windows</a>: 未找到描述</li><li><a href="https://www.theregister.com/2024/04/03/llamafile_performance_gains/">Llamafile LLM driver project boosts performance on CPU cores</a>: 提升 LLaMA 性能的方法</li><li><a href="https://huggingface.co/jartine/Mixtral-8x7B-Instruct-v0.1-llamafile/tree/main">jartine/Mixtral-8x7B-Instruct-v0.1-llamafile at main</a>: 未找到描述</li><li><a href="https://learn.microsoft.com/en-us/windows/arm/arm64x-pe">Arm64X PE Files</a>: Arm64X 是 Windows 11 SDK 中的一种 PE 文件类型，用于 Arm64 上的 x64 兼容性。对于中间件或插件开发人员来说，Arm64X 可能是一个很好的解决方案，因为代码可能会被加载到 x64 或 A...</li><li><a href="http://www.emulators.com/docs/abc_arm64ec_explained.htm">ARM64 Boot Camp: ARM64EC and ARM64X Explained</a>: 未找到描述</li><li><a href="https://github.com/jart/cosmopolitan/issues/1010">execve() should polyfill #! on windows · Issue #1010 · jart/cosmopolitan</a>: 复制自 bellard/quickjs#197: #!/bin/qjs console.log(&quot;Hello&quot;); 当从 bash 作为脚本调用时不起作用: $ ./test.qjs ./test.qjs: line 2: syntax error near unexpected token `&...
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1224755324100149390)** (1 messages): 

- **Potential for Opus Judgement to Boost Performance**: 有推测认为，如果 *Opus Judgement* 是准确的，那么可能存在尚未利用的潜力，可以通过进一步的研究级 AI 微调 (**RLAIF**) 来增强结果。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1224783900036042772)** (29 messages🔥): 

- **Google's New AI Lead Excites Discord**: 成员们对 [Logan K 宣布](https://fxtwitter.com/OfficialLoganK/status/1775222819439149424) 加入 Google 领导 AI Studio 的产品工作并支持 Gemini API 表示惊讶和幽默，反应从震惊到对地理位置等实际原因的猜测不等。
- **The Logan Strategy: Lifestyle or Poaching?**: 对话推测了影响 Logan 移师 Google 的各种因素，包括芝加哥的吸引力、被感知的 HR 挖角策略、未来股票增值的机会，以及 Google 相比 OpenAI 在发布模型权重方面相对开放的态度。
- **Ideology or Opportunity?**: 成员们讨论了 Logan 离开 OpenAI 的潜在意识形态原因，例如对更多开放性的渴望，但也考虑了尽管有个人价值观，但仍被 Google 的待遇所吸引的可能性。
- **Startup Ambitions or Strategic Move?**: 对话包括猜测 Logan 是否有其之前的 "building at" 简历所暗示的创业抱负，或者由于 Google 目前在 AI 领域的良好势头，这一举动是否是一个战略选择。
- **Financial Times and the AI Buzz**: 一位成员分享了关于 AI 的《金融时报》文章链接，但内容被锁定在订阅墙后，使得相关讨论未能充分展开 ([FT 内容](https://www.ft.com/content/2f4bfeb4-6579-4819-9f5f-b3a46ff59ed1))。

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/OfficialLoganK/status/1775222819439149424">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：很高兴分享我已加入 @Google，负责 AI Studio 的产品工作并支持 Gemini API。前方还有很多艰巨的工作，但我们将努力让 Google 成为开发者使用 AI 构建应用的最佳家园。...</li><li><a href="https://www.ft.com/content/2f4bfeb4-6579-4819-9f5f-b3a46ff59ed1">Google 考虑对 AI 驱动的搜索收费，这是其商业模式的重大变革</a>：未找到描述
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/1224755945343811624)** (1 条消息): 

- **Open Science 变得不透明**：在 **GPT-4 技术报告**发布后（该报告保留了模型细节），出现了一种趋势，其他公司也开始对其模型信息保密。成员回忆称，这是该领域向增加保密性转变的标志。
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 条消息): 

iron_bound: https://github.com/intel-analytics/ipex-llm
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1224978234559696968)** (4 条消息): 

- **通过 DISTFLASHATTN 彻底改变 LLM**：[DISTFLASHATTN](https://arxiv.org/abs/2310.03294) 提出了一种内存高效的 Attention 机制，声称能将**二次项峰值内存占用降低至线性**，并优化长上下文 LLM 训练。据报道，与 Ring Self-Attention 和带有 FlashAttention 的 Megatron-LM 等现有解决方案相比，它能实现高达 **8 倍的序列长度**和显著的速度优势。

- **尖端 LLM 训练代码发布**：研究人员可以通过提供的 [GitHub 仓库](https://github.com)获取 DISTFLASHATTN 的代码，该代码在 Llama-7B 等模型的训练序列长度和速度方面有显著改进。

- **对 DISTFLASHATTN 缺少反向传播伪代码的批评**：一位成员指出 DISTFLASHATTN 论文中的一个疏漏；它**没有包含反向传播（backward pass）的伪代码**。

- **先前存在类似问题的 Attention 机制**：同一位成员指出，**Ring Attention**（之前的一项技术）也未能提供其反向传播的伪代码。

- **呼吁科学可重复性**：一条评论强调了对科学缺乏可重复性的沮丧，这可能与已发表作品中省略了详细实现细节（如伪代码）有关。

**提到的链接**：<a href="https://arxiv.org/abs/2310.03294">DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training</a>：FlashAttention (Dao, 2023) 在单 GPU 上训练基于 Transformer 的大语言模型 (LLMs) 时，有效地将二次项峰值内存占用降低到线性。在本文中，我们介绍了 DISTFLA...

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1225047565670547496)** (2 条消息): 

- **针对初学者的 CUDA 学习资源**：一位新成员询问了在有 Python 和 Rust 背景的情况下学习 CUDA 编程基础的建议。另一位成员建议从 [CUDA MODE YouTube 频道](https://www.youtube.com/@CUDAMODE)的一系列讲座以及其 [GitHub 页面](https://github.com/cuda-mode)上的补充内容开始。

**提到的链接**：<a href="https://www.youtube.com/@CUDAMODE">CUDA MODE</a>：一个 CUDA 阅读小组和社区 https://discord.gg/cudamode 补充内容见此处 https://github.com/cuda-mode 由 Mark Saroufim 和 Andreas Köpf 创建

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1224979333785849907)** (2 条消息): 

- **用于内存高效 LLM 训练的 DISTFLASHATTN**：介绍了一种名为 DISTFLASHATTN 的新型**[分布式内存高效 Attention 机制](https://arxiv.org/abs/2310.03294)**，通过 Token 级工作负载平衡等技术优化了长上下文大语言模型 (LLMs) 的训练。它的表现优于现有模型，与 Ring Self-Attention 和带有 FlashAttention 的 Megatron-LM 相比，实现了高达 **8 倍的序列长度**和**速度提升**，**源代码可在 GitHub 上获得**。

- **DISTFLASHATTN 论文阅读计划**：一位成员分享了在第二天审阅 DISTFLASHATTN 论文的意向，表示了兴趣并可能随后展开讨论。

**提到的链接**：<a href="https://arxiv.org/abs/2310.03294">DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training</a>：FlashAttention (Dao, 2023) 在单 GPU 上训练基于 Transformer 的大语言模型 (LLMs) 时，有效地将二次方峰值显存占用降低到了线性水平。在本文中，我们介绍了 DISTFLA...

---

**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1224674334778327071)** (6 messages): 

- **澄清吞吐量效率**：一位用户引用了一篇论文，强调**每个 token** 的效率会随着端到端吞吐量（编码 + 解码）除以 **token 总数**的测量而提高。
- **辩论 Token 生成速度**：一场关于增加更多 token 如何导致速度提升的讨论展开了。提出的观点是，虽然编码可以并行运行，但解码是顺序执行的，因此预期每个额外的 token 解码所需的时间应该相同。
- **编码速度见解**：进一步的解释澄清了所讨论的图表显示的是生成恒定 512 个 token 的速度，这意味着图中任何速度提升都与**编码过程 (encoding process)** 相关。
- **质疑解码速度**：在理解该过程方面存在持续的疑问，即由于解码本质上是顺序的，需要每个 token 等待其前序 token，那么在更大的上下文下解码如何能变得更快。

---

**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1224849362052452372)** (1 messages): 

- **渴望加入的新贡献者**：一位成员表达了对 **onboarding session** 的兴趣，强调了他们在 Python、软件工程方面的背景以及数据科学硕士学位。他们拥有与 StonyBrook 的人员合作进行 **AI 医疗研究**的经验，并擅长编写**数据流水线 (data pipelines)**。

---

**Skunkworks AI ▷ #[finetuning](https://discord.com/channels/1131084849432768614/1131669354912678028/1224826866137960590)** (1 messages): 

- **自然语言对方程式至关重要**：尽管 **GPT-4** 和 **Claude** 具备高水平的能力，但除非用自然语言仔细解释问题，否则它们有时仍难以解出方程式。这表明在当前的 AI 规模下，重大挑战依然存在。

---

**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

jinastico: <@748528982034612226>

---

**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1225067602288574554)** (1 messages): 

- **对话日志中的术语纠葛**：一位参与者对 `logs.db` 中的 `responses` 表进行了观察，表达了对如何命名对话部分的兴趣。他们分享到，对话中第一人发言的初始部分被称为 "speaker turn" 或 "turn"，这促使他们将应用的表命名为 `turns`。