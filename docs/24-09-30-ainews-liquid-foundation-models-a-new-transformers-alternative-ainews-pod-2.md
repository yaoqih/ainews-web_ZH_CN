---
companies:
- liquid-ai
- meta-ai-fair
- google-deepmind
- openai
date: '2024-10-01T01:34:19.663940Z'
description: '以下是为您翻译的中文内容：


  **Liquid.ai** 结束隐身状态正式亮相，推出了三个亚二次方（subquadratic）基础模型。这些模型在效率上优于状态空间模型（SSM）以及苹果的端侧和服务器模型，并获得了
  3700 万美元的种子轮融资。**Meta AI** 发布了 **Llama 3.2**，其中包括具备视觉能力的多模态模型，以及适用于移动设备的轻量级纯文本变体。**Google
  DeepMind** 推出了生产就绪的 **Gemini-1.5-Pro-002** 和 **Gemini-1.5-Flash-002** 模型，并优化了价格和速率限制；同时还推出了
  **AlphaChip**，这是一个利用强化学习实现超人水平快速布局的 AI 驱动芯片设计系统。**OpenAI** 为 ChatGPT Plus 和 Teams
  用户增强了“高级语音模式”（Advanced Voice Mode），新增了自定义指令、记忆功能以及多款灵感源自自然的新音色。加州州长否决了 SB-1047 AI
  监管法案，**Yann LeCun** 和 **svpino** 等 AI 社区领袖对此表示庆祝，认为这是开源 AI 的胜利。Google 升级了 **NotebookLM**，其音频概览功能现已支持
  YouTube 视频和音频文件，可将文档转化为 AI 生成的播客。**Yann LeCun** 指出：“AI 领域的开源正在蓬勃发展”，并强调 GitHub 和
  HuggingFace 上的模型数量已达 100 万个。'
id: 21fed6b0-f5de-419a-8a11-9f0745506282
models:
- llama-3-2
- gemini-1.5-pro-002
- gemini-1.5-flash-002
original_slug: ainews-liquid-foundation-models-a-new
people:
- ylecun
- svpino
title: 液态基础模型：Transformer 的新替代方案 + AI 新闻播客第 2 期
topics:
- reinforcement-learning
- multimodality
- model-efficiency
- foundation-models
- audio-processing
- model-deployment
- open-source
---

<!-- buttondown-editor-mode: plaintext -->**自适应计算算子（Adaptive computational operators）就是你所需要的一切。**

> 2024年9月27日至9月30日的 AI 新闻。我们为您查看了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord（**225** 个频道，**5435** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**604 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

并不是每天都有一个可信的新基础模型（foundation model）实验室成立，所以今天的头条理所当然属于 Liquid.ai。在获得 [3700 万美元种子轮融资](https://siliconangle.com/2023/12/06/liquid-ai-raises-37-6m-build-liquid-neural-networks/) 10 个月后，他们终于“结束隐身模式”，发布了 3 个亚二次方（subquadratic）模型，这些模型在同级别中表现非常出色：

[
![image.png](https://assets.buttondown.email/images/f4006762-e87d-449a-9acd-7a60e88e20d1.png?w=960&fit=max)
](https://x.com/AndrewCurran_/status/1840802455225094147)

与状态空间模型（state space models）相比，我们对“液体网络”（liquid networks）知之甚少，但他们展示了必不可少的亚二次方图表，证明他们在该领域击败了 SSM：


![image.png](https://assets.buttondown.email/images/3502168f-ebe5-429f-8c75-cc43fc03852a.png?w=960&fit=max)


以及非常可信的基准测试（benchmark）分数：


![image.png](https://assets.buttondown.email/images/8ad83dec-2a97-4f2f-86a6-6c609c2af5c2.png?w=960&fit=max)


值得注意的是，它们的单参数效率似乎明显高于 Apple 的端侧和服务器基础模型（[我们的相关报道在此](https://buttondown.com/ainews/archive/ainews-apple-intelligence/)）。

它们尚未开源，但提供了 playground 和 API，并承诺在 10 月 23 日正式发布前提供更多内容。

---

**AINews 播客**

我们本月初首次预览了[受 Illuminate 启发的播客](https://buttondown.com/ainews/archive/ainews-not-much-happened-today-ainews-podcast/)。随着 NotebookLM Deep Dive 的走红，我们正在构建一个开源音频版的 AINews 作为一项新实验。在这里查看[我们最新的 NotebookLM 与我们播客的对比](https://github.com/smol-ai/temp/tree/main/2024-09-30)！如果您有反馈意见或想要开源仓库（repo），请在 [@smol_ai](https://twitter.com/smol_ai) 告诉我们。


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

- **Llama 3.2 发布**：Meta AI 宣布推出 Llama 3.2，其特点是包含具有视觉能力的 11B 和 90B 多模态模型，以及适用于移动设备的轻量级 1B 和 3B 纯文本模型。视觉模型支持图像和文本提示，可对输入进行深度理解和推理。[@AIatMeta](https://twitter.com/AIatMeta/status/1840431307761054202) 指出，这些模型可以同时接收图像和文本提示，以深入理解并对输入进行推理。

- **Google DeepMind 公告**：Google 宣布推出两个新的生产级 Gemini AI 模型：Gemini-1.5-Pro-002 和 Gemini-1.5-Flash-002。[@adcock_brett](https://twitter.com/adcock_brett/status/1840422127331057885) 强调，该公告最棒的部分是 1.5 Pro 降价 50%，且 Flash 和 1.5 Pro 的速率限制分别提升了 2 倍和 3 倍。

- **OpenAI 更新**：据 [@adcock_brett](https://twitter.com/adcock_brett/status/1840422082301046850) 报道，OpenAI 向所有 ChatGPT Plus 和 Teams 订阅用户推出了增强版 Advanced Voice Mode，增加了 Custom Instructions、Memory 以及五种新的“受自然启发”的声音。

- **AlphaChip**：Google DeepMind 发布了 AlphaChip，这是一个利用强化学习设计芯片的 AI 系统。[@adcock_brett](https://twitter.com/adcock_brett/status/1840422149829386581) 指出，这使得在数小时内构建出超越人类水平的芯片布局成为可能，而以往则需要数月。

**开源与监管**

- **SB-1047 否决**：加州州长 Gavin Newsom 否决了 SB-1047，这是一项关于 AI 监管的法案。包括 [@ylecun](https://twitter.com/ylecun/status/1840511216889778332) 和 [@svpino](https://twitter.com/svpino/status/1840510698813829254) 在内的许多科技界人士对这一决定表示感谢，认为这是开源 AI 和创新的胜利。

- **开源增长**：[@ylecun](https://twitter.com/ylecun/status/1840431809479463187) 强调 AI 开源正在蓬勃发展，并引用 GitHub 和 HuggingFace 上的项目数量已达到 100 万个模型。

**AI 研究与开发**

- **NotebookLM**：Google 升级了 NotebookLM/Audio Overviews，增加了对 YouTube 视频和音频文件的支持。[@adcock_brett](https://twitter.com/adcock_brett/status/1840422255420912045) 分享道，Audio Overviews 可以将笔记、PDF、Google Docs 等转换为 AI 生成的播客。

- **Meta AI 进展**：据 [@adcock_brett](https://twitter.com/adcock_brett/status/1840422210395054368) 报道，Meta AI（消费者聊天机器人）现在已具备多模态能力，能够“看到”图像并允许用户使用 AI 编辑照片。

- **AI 在医学中的应用**：根据 [@dair_ai](https://twitter.com/dair_ai/status/1840450324097904901) 的报道，一项关于 o1-preview 模型在医疗场景中的研究显示，在 19 个数据集和两个新创建的复杂 QA 场景中，其准确率比 GPT-4 平均高出 6.2% 和 6.6%。

**行业趋势与合作**

- **James Cameron 与 Stability AI**：据 [@adcock_brett](https://twitter.com/adcock_brett/status/1840422277994733702) 报道，电影导演 James Cameron 加入了 Stability AI 的董事会，他认为生成式 AI 与 CGI 的融合是视觉媒体创作的“下一波浪潮”。

- **EA 的 AI 演示**：EA 展示了一个用于用户生成视频游戏内容的新 AI 概念，利用 3D 资产、代码、游戏时长、遥测事件和 EA 训练的自定义模型来实时重混游戏和资产库，由 [@adcock_brett](https://twitter.com/adcock_brett/status/1840422300610388224) 分享。


---

# AI Reddit 摘要

## /r/LocalLlama 回顾

**主题 1. Emu3：多模态 AI 的 Next-token prediction 突破**

- **Emu3: Next-Token Prediction is All You Need** ([Score: 227, Comments: 63](https://reddit.com//r/LocalLLaMA/comments/1fsoe83/emu3_nexttoken_prediction_is_all_you_need/))：**Emu3** 是一套全新的多模态模型，仅通过 **next-token prediction** 就在生成和感知任务中均实现了 **state-of-the-art performance**，超越了 **SDXL** 和 **LLaVA-1.6** 等成熟模型。通过将图像、文本和视频 token 化到离散空间，并从零开始训练单个 Transformer，Emu3 简化了复杂的多模态模型设计，并展示了 next-token prediction 在构建超越语言的通用多模态智能方面的潜力。研究人员已经开源了关键技术和模型，包括 [GitHub](https://github.com/baaivision/Emu3) 上的代码和 [Hugging Face](https://huggingface.co/collections/BAAI/emu3-66f4e64f70850ff358a2e60f) 上的预训练模型，以支持该方向的进一步研究。
  - **Booru tags**（常用于动漫图站和 **Stable Diffusion** 模型）出现在 Emu3 的生成示例中。用户讨论了支持这些标签对于模型流行度的必要性，一些人认为这是获得广泛采用的**必要条件**。
  - 讨论中提到了将 **diffusion models 应用于文本生成**，并提到了 **CodeFusion** 论文。用户推测了 **Meta 的 GPU compute capability** 以及潜在的未发布实验，暗示大型 AI 公司之间可能存在控制信息发布的协议。
  - 该模型将**视频生成作为 next-token prediction** 的能力让用户感到兴奋，可能开启“视频生成的新时代”。然而，人们对**生成时间**表示担忧，有报告称在 Replicate 上**生成一张图片需要 10 分钟**。


**主题 2. Replete-LLM 发布具有性能提升的微调版 Qwen-2.5 模型**

- **Replete-LLM Qwen-2.5 models release** ([Score: 73, Comments: 55](https://reddit.com//r/LocalLLaMA/comments/1frynwr/repletellm_qwen25_models_release/))：Replete-LLM 发布了参数量从 **0.5B 到 72B** 不等的 **Qwen-2.5** 微调版本，采用了 **Continuous finetuning 方法**。这些模型已在 **Hugging Face** 上发布，据报告，与原始 Qwen-2.5 权重相比，所有尺寸的模型性能均有所提升。
  - 用户要求提供 **benchmarks 和横向对比**以展示改进。开发者为 **7B 模型**增加了一些基准测试，并指出运行全面的基准测试通常需要大量的计算资源。
  - 开发者的 **continuous finetuning 方法**结合了之前的微调权重、预训练权重和新的微调权重，以最小化损失。一份详细介绍该方法的[论文](https://docs.google.com/document/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/edit?usp=sharing)已被分享。
  - 模型的 **GGUF 版本**已提供，包括高达 **72B 参数**的量化版本。用户表示有兴趣在从高端机器到手机等边缘设备的各种设备上对其进行测试。

## 其他 AI Subreddit 汇总

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型能力与进展**

- **OpenAI 的 o1 模型**可以处理 **5 小时的任务**，与 GPT-3（5 秒任务）和 GPT-4（5 分钟任务）相比，实现了更长程的问题解决能力。据 [OpenAI 战略营销负责人](https://www.reddit.com/r/singularity/comments/1fsfz47/dane_vahey_head_of_strategic_marketing_at_openai/) 透露。

- **MindsAI 在 ARC-AGI 基准测试中取得了 48% 的新高分**，而该奖项的[目标设定为 85%](https://www.reddit.com/r/singularity/comments/1fs9ymg/new_arcagi_high_score_by_mindsai_48_prize_goal_85/)。

- 一名[黑客演示了](https://www.reddit.com/r/singularity/comments/1fsdfjc/hacker_plants_false_memories_in_chatgpt_to_steal/)在 **ChatGPT 中植入虚假记忆**的能力，从而创建一个持久的数据外泄通道。

**AI 政策与监管**

- **加州州长 Gavin Newsom 否决了**一项[备受争议的 AI 安全法案](https://www.reddit.com/r/singularity/comments/1fsegyi/california_governor_vetoes_contentious_ai_safety/)，突显了围绕 AI 监管的持续争论。

**AI 伦理与社会影响**

- AI 研究员 **Dan Hendrycks 提出了一个思想实验**，关于一种假设的、具有快速增长的智能和繁殖能力的新物种，[质疑哪个物种将掌握控制权](https://www.reddit.com/r/singularity/comments/1fs6ce0/dan_hendrycks_imagine_that_a_new_species_arrives/)。

- [向 OpenAI 的 o1 模型进行单次查询的成本](https://www.reddit.com/r/OpenAI/comments/1fsdrxq/the_cost_of_a_single_query_to_o1/)引发了关注，触发了关于先进 AI 模型经济影响的讨论。

**迷因与幽默**

- 一个关于[试图遏制 AGI](https://www.reddit.com/r/singularity/comments/1fsb6ml/trying_to_contain_agi_be_like/) 的迷因引发了关于 AI 安全挑战的讨论。

- 另一个迷因质疑[人类是否是“坏人”](https://www.reddit.com/r/singularity/comments/1fsk1ov/are_we_the_baddies/)（相对于 AI 的发展），导致了关于 AI 意识和伦理的辩论。


---

# AI Discord 汇总

> 由 o1-preview 提供的摘要之摘要

**主题 1. AI 模型凭借新发布和升级掀起波澜**

- [**LiquidAI 凭借 Liquid Foundation Models (LFMs) 挑战巨头**](https://www.liquid.ai/liquid-foundation-models)：LiquidAI 推出了 LFMs——1B、3B 和 40B 模型——声称在 **MMLU** 等基准测试中表现优异，并指出了竞争对手的低效。凭借来自 **MIT** 的团队成员，其架构旨在挑战行业内的既有模型。
- [**Aider v0.58.0 编写了超过一半的自身代码**](https://aider.chat/2024/09/26/architect.html)：最新版本引入了模型配对和新命令等功能，并自豪地宣布 Aider 自主创建了该更新中 **53%** 的代码。此版本支持新模型，并通过改进 `/copy` 和 `/paste` 等命令增强了用户体验。
- [**微软的幻觉检测模型升级至 Phi-3.5**](https://huggingface.co/grounded-ai/phi3.5-hallucination-judge)：从 Phi-3 升级到 Phi-3.5，该模型展示了令人印象深刻的指标——**Precision: 0.77**，**Recall: 0.91**，**F1 Score: 0.83**，以及 **Accuracy: 82%**。它旨在通过有效识别幻觉来提高语言模型输出的可靠性。

**主题 2. AI 监管与法律斗争升温**

- **加州州长否决 AI 安全法案 SB 1047**：州长 **Gavin Newsom** 阻止了旨在监管 AI 公司的法案，声称这不是保护公众的最佳方法。批评者认为这是 AI 监管的挫折，而支持者则推动基于能力的监管。
- **OpenAI 因薪酬要求面临人才流失**：OpenAI 的核心研究人员威胁要辞职，除非增加薪酬，在估值飙升之际，已有 **12 亿美元**套现。新任 CFO **Sarah Friar** 正在应对紧张的谈判，而 **Safe Superintelligence** 等对手正在挖角人才。
- [**LAION 在德国赢得里程碑式的版权案件**](https://www.technollama.co.uk/laion-wins-copyright-infringement-lawsuit-in-german-court)：LAION 成功抵御了版权侵权指控，确立了一个有利于 AI 数据集使用的先例。这一胜利消除了 AI 研究与开发中的重大法律障碍。

**主题 3. 社区应对 AI 工具挑战**

- **Perplexity 用户抱怨性能不稳定**：用户报告响应不稳定且缺失引用，尤其是在网页搜索和学术论文之间切换时。许多人因更好的访问权限和来源预览等功能，在学术研究中更倾向于使用 **Felo**。
- **OpenRouter 用户遭遇速率限制和性能下降**：频繁的 **429 错误** 令 **Gemini Flash** 用户感到沮丧，目前正等待 Google 增加配额。像 **Hermes 405B free** 这样的模型在维护后表现出性能下降，引发了对供应商变更的担忧。
- **关于 OpenAI 研究透明度的辩论升温**：批评者认为 OpenAI 对其研究不够开放，指出仅靠博客文章是不够的。员工坚称具有透明度，但社区寻求除 [研究博客](https://openai.com/index/learning-to-reason-with-llms/) 之外更具实质性的交流。

**主题 4. 硬件问题困扰 AI 爱好者**

- **NVIDIA Jetson AGX Thor 的 128GB VRAM 引发硬件羡慕**：定于 2025 年发布的 AGX Thor 拥有海量 VRAM，引发了人们对 **3090** 和 **P40** 等当前 GPU 未来地位的质疑。该公告让社区对潜在的升级和不断演变的 GPU 格局议论纷纷。
- **新的 NVIDIA 驱动程序降低了 Stable Diffusion 的性能**：使用 **8GB VRAM 显卡** 的用户在驱动更新后，生成时间从 **20 秒激增至 2 分钟**。社区建议不要更新驱动，以免破坏渲染工作流。
- **Linux 用户与 NVIDIA 驱动问题作斗争，转而关注 AMD GPU**：用户对 NVIDIA 存在问题的 Linux 驱动（尤其是 **VRAM offloading** 方面）愈发不满。一些用户考虑转向 **AMD 显卡**，理由是其在配置中具有更好的性能和易用性。

**主题 5. AI 扩展到创意和健康领域**

- [**NotebookLM 根据你的内容制作定制播客**](https://notebooklm.google.com/)：Google 的 NotebookLM 推出了一项音频功能，可以使用 AI 主持人生成个性化播客。用户对根据其提供材料生成的引人入胜且极具说服力的对话印象深刻。
- **精神分裂症治疗取得突破**：Perplexity AI 宣布推出了 **30 年来** 首款精神分裂症药物，标志着精神健康护理领域的重大进展。讨论强调了其对患者护理和治疗范式的潜在影响。
- **关于 AI 生成艺术与人类创造力的激烈辩论**：Stability.ai 社区在 **AI 艺术** 与人类创作的质量和深度对比上存在分歧。虽然一些人拥护 AI 生成的作品为合法艺术，但另一些人则坚持人类艺术性的持久优越性。


---

# 第 1 部分：高层级 Discord 摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LinkedIn 复制代码争议**：LinkedIn 因涉嫌在未妥善署名的情况下复制 Unsloth 的代码而面临抵制，促使 Microsoft 和 GitHub 介入以确保正确归功。
   - 该事件强调了遵守 **open source licensing**（开源许可）的紧迫性，并引发了对 **intellectual property**（知识产权）的担忧。
- **微调 Llama 模型的最佳实践**：为了减轻 Token 生成问题，用户讨论了在 **Llama model fine-tuning** 期间设置 **random seed** 并仔细评估输出质量。
   - 正确配置 EOS tokens 对于在推理过程中保持模型的原始能力至关重要。
- **GGUF 转换错误**：用户在加载 GGUF 模型时遇到了“cannot find tokenizer merges in model file”错误，凸显了模型保存过程中的潜在问题。
   - 理解转换过程并保持与 tokenizer 配置的兼容性，对于确保模型平滑过渡至关重要。
- **Liquid Foundation Models 发布**：LiquidAI 宣布推出 **Liquid Foundation Models (LFMs)**，包括 **1B、3B 和 40B 模型**，但对其公告的有效性存在质疑。
   - 针对这些说法的准确性表达了担忧，特别是与 **Perplexity Labs** 相关的部分。
- **利用未开发的算力**：成员注意到大量 **compute power** 未被充分利用，建议在各种硬件设置中进行潜在的性能改进。
   - 通过优化现有资源实现现实的性能提升，表明当前系统仍有很大的增强空间。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.58.0 带来令人兴奋的增强功能**：[Aider v0.58.0](https://aider.chat/2024/09/26/architect.html) 的发布引入了模型配对（model pairing）和新命令等功能，其中 Aider 自主创建了该更新 **53%** 的代码。
   - 此版本还支持新模型，并通过 **剪贴板命令更新** 等功能提升了用户体验。
- **Architect/Editor 模型提高效率**：Aider 利用一个主模型进行规划，并使用一个可选的 editor 模型进行执行，允许通过 `--editor-model` 进行配置以实现最佳任务处理。
   - 这种双模型方法引发了关于多 Agent 编程能力以及 LLM 任务价格效率的讨论。
- **NotebookLM 的新播客功能脱颖而出**：[NotebookLM](https://notebooklm.google/) 推出了一项音频功能，可以根据用户内容生成自定义播客，以极具吸引力的形式展示 AI 主持人。
   - 其中一个示例播客展示了该技术从提供材料中创建引人入胜的对话的能力。
- **内容生成的自动化提案**：有人提出了使用 NotebookLM 自动根据发布说明（release notes）制作视频的想法，这可能会催生一个名为 *ReleaseNotesLM* 的高效工具。
   - 该工具旨在将文字更新转化为音频，为内容创作者简化流程。
- **模型成本效率讨论**：在 Architect 任务中使用 `claude-3.5-sonnet`，在编辑任务中使用 `deepseek v2.5` 等不同模型，可以使 editor tokens 的 **成本降低 20-30 倍**。
   - 参与者强调了根据成本和功能进行战略性模型选择的优势，并探索了用于增强配置的脚本选项。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI 模型合并技术探讨**：用户探索了各种 **模型合并（model merging）** 方法，特别是关注 PEFT merge 和 **DARE** 方法，以增强微调期间的性能。
   - 对话强调了利用现有模型而非从头开始训练 LLM 的价值，将这些方法定位为高效处理任务的关键。
- **近期论文中的医疗 AI 见解**：一篇文章总结了 2024 年 9 月 21 日至 27 日期间 **医疗 AI 领域的顶级研究论文**，包括《A Preliminary Study of o1 in Medicine》等显著研究。
   - 成员们建议将这些见解拆分为单独的博客文章，以增加围绕杰出论文的参与度和讨论。
- **幻觉检测模型性能指标**：新发布的 **幻觉检测模型（Hallucination Detection Model）** 从 **Phi-3 升级到 Phi-3.5**，拥有令人印象深刻的指标：**Precision: 0.77**，**Recall: 0.91**，**F1 Score: 0.83**，以及 **准确率: 82%**；[查看模型卡片](https://huggingface.co/grounded-ai/phi3.5-hallucination-judge)。
   - 该模型旨在通过有效识别幻觉来提高语言模型输出的可靠性。
- **Gradio 用户反响平平**：社区对 **Gradio** 的情绪趋于负面，用户因 UI 响应问题和使项目管理复杂化的设计缺陷将其贴上“烂透了（hot garbage）”的标签。
   - 尽管遭到抵制，成员们仍鼓励在专门的支持频道寻求帮助，表明在故障排除方面仍有持续投入。
- **关键点检测模型增强**：**OmDet-Turbo 模型** 的发布支持 zero-shot 目标检测，集成了 Grounding DINO 和 OWLv2 的技术；详情可以在 [这里](https://www.linkedin.com/posts/yoni-gozlan_ai-artificialintelligence-objectdetection-ugcPost-7244768044533657603-FDOT?utm_source=share&utm_medium=member_desktop) 找到。
   - 对 **SuperPoint** 等模型的关键点检测的专门关注，为社区对该领域未来发展的期待奠定了基础。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **在 LM Studio 中下载和侧加载模型的挑战**：用户在 *LM Studio* 中下载模型时遇到问题，特别是在使用 VPN 时，促使一些人选择侧加载模型。指出了对 **safetensors** 和 **GGUF** 等模型格式支持的局限性。
   - 社区对整体下载体验表示沮丧，讨论强调了对各种模型类型提供更好支持的必要性。
- **NVIDIA Jetson AGX Thor 拥有 128GB VRAM**：即将推出的 **NVIDIA Jetson AGX Thor** 将在 2025 年配备 **128GB 的 VRAM**，这引发了关于 **3090** 和 **P40** 等当前 GPU 可行性的疑问。这一公告在 GPU 领域引发了关于潜在升级的热议。
   - *一些成员思考，随着对高 VRAM 选项需求的持续增长，现有硬件是否仍具竞争力。*
- **GPU 性能对比：3090 vs 3090 Ti vs P40**：成员们对比了 **3090**、**3090 Ti** 和 **P40** 的性能，重点关注 VRAM 和价格，这些因素严重影响了他们的选择。有评论指出 **P40** 的运行速度大约是 **3090** 的一半。
   - 成员们对 GPU 价格上涨表示担忧，并辩论了针对当前 AI 工作负载在不同型号之间的权衡。
- **GPU 市场定价动态**：讨论强调，由于黄牛炒作和 AI 应用需求的增加，**GPU 价格居高不下**，**A6000** 可作为高 VRAM 的替代方案。然而，预算有限的成员更倾向于在配置中使用多个 **3090**。
   - *对话凸显了对价格趋势的普遍沮丧，以及许多人在当前市场中面临的障碍。*
- **Linux 上 NVIDIA 驱动程序的挑战**：社区分享了对 **NVIDIA 的 Linux 驱动程序** 众所周知的问题的抱怨，特别是在 **VRAM offloading** 方面，而 **AMD** 显卡在这一领域表现更好。**CUDA** 和其他驱动程序安装中的复杂性加剧了这些挫败感。
   - 一些成员表示越来越倾向于 **AMD** 硬件，理由是其在某些配置中具有更好的易用性。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Cerebras 芯片优化讨论**：成员们正在探索 **Cerebras 芯片** 的代码优化，对于潜在购买和专业知识的获取持有不同意见。
   - 随着成员们表现出寻找专家以深入了解 Cerebras 技术的意愿，社区兴趣日益增长。
- **对垃圾信息管理的担忧日益增加**：社区正在处理 Discord 上日益增多的**加密货币诈骗垃圾信息**，建议采用更严格的验证协议以增强服务器安全性。
   - 成员们正积极寻找高效的抗垃圾信息工具，并讨论他们使用 AutoMod 等现有解决方案的经验。
- **Triton 演讲资料分享**：一位成员寻找 **Triton 演讲** 的幻灯片，并被引导至包含教育资源的 [GitHub 仓库](https://github.com/gpu-mode/lectures)。
   - 这反映了知识共享和协作学习的强大社区文化。
- **AMD GPU 性能问题**：成员们重点讨论了 **AMD GPU** 的显著性能限制，特别是 **GFX1100** 和 **MI300** 架构。
   - 许多人强调了多节点设置中持续存在的挑战，并表示需要提升性能。
- **理解 Model Parallelism 与 ZeRO/FSDP**：成员们阐明了 **Model Parallelism** 与 **ZeRO/FSDP** 之间的区别，重点关注 ZeRO 如何实现参数分布策略。
   - 讨论强调 FSDP 利用分片（sharding）来提高模型训练效率，吸引了那些希望了解高级功能的成员。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **探讨 Modular 社区会议议程**：今天的 Modular 社区会议于 **太平洋时间上午 10 点** 举行，内容涵盖 **MAX driver & engine API** 以及关于 Magic 的问答环节，可通过 [Zoom](https://modul.ar/community-meeting-zoom) 参加。参与者可以查看 [Modular Community Calendar](https://modul.ar/community-meeting) 了解后续活动。
   - 会议录像将上传至 YouTube，包括今天的会议，可通过[此链接](https://www.youtube.com/watch?v=zL0cCHs_0RI&list=PLh0S94-sJw_6UcaIMgpESb5KSVRsuuhnX)观看，确保无人错过。
- **关于 Mojo 语言增强功能的辩论**：一项关于高级 **Mojo 语言特性** 的提案建议为消息传递引入命名变体（named variants），并在不引入新结构的情况下更好地管理标签联合（tagged unions），这引发了成员间的广泛讨论。
   - 支持者权衡了定义类型的易用性，讨论了设计过程中名义类型（nominal types）与结构类型（structural types）之间的平衡。
- **使用 Mojopkg 打包模型**：社区热烈讨论了在 **Mojopkg** 中嵌入模型的能力，展示了通过将所有内容打包进单个可执行应用程序来提升用户体验的潜力。
   - 提到了其他语言的关键示例，阐明了这如何为用户简化依赖关系并增强可用性。
- **平滑管理原生依赖**：针对 Mojopkg 简化依赖管理的能力提出了关注，这可能使安装和配置变得更加容易。
   - 讨论包括了一些实际实现，例如直接在 Mojo 应用程序中嵌入 Python 等运行时的安装程序。
- **MacOS 上的兼容性警告**：一位用户报告了在为 macOS 构建对象文件时的兼容性警告，指出版本 **15.0** 和 **14.4** 之间存在链接问题。
   - 尽管这些警告不是致命的，但它们可能指向未来需要解决的兼容性挑战。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 推动开源倡议**：Nous Research 专注于 **开源 AI 研究**，与开发者合作并发布了包括 **Hermes 家族** 在内的模型。
   - 他们的 **DisTrO 项目** 旨在加速互联网上的 AI 模型训练，并暗示了 **闭源模型** 的风险。
- **Distro 论文发布引发热议**：**Distro 论文** 预计很快就会发布，引发了渴望更新的社区成员的兴奋。
   - 该论文与 AI 社区的相关性放大了对其详细内容的期待。
- **新型 AI 模型微调技术发布**：最近 **Rombodawg’s Replete-LLM** 在创新微调技术的帮助下，登顶了 **7B 模型** 的 **OpenLLM leaderboard**。
   - **TIES merging** 等方法被认为是显著提升模型基准测试的关键。
- **Liquid Foundation Models 引起关注**：LiquidAI 推出了 **Liquid Foundation Models**，版本包括 **1B、3B 和 40B**，旨在为 AI 领域提供新的能力。
   - 这些模型被视为在为 AI 领域的各种应用提供创新功能方面发挥着关键作用。
- **本周医学 AI 论文：我们离 AI 医生更近了吗？**：重点推介的论文《*A Preliminary Study of o1 in Medicine*》探讨了 AI 担任医生的潜力，由该领域的专家共同撰写。
   - 该论文被评为 **本周医学 AI 论文**，展示了其在关于 AI 在医疗保健中角色的持续讨论中的相关性。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 在性能一致性方面面临挑战**：用户注意到 **Perplexity** 在切换网页搜索和学术论文时出现 **响应不一致** 的情况，并存在引用缺失的案例。
   - 用户担心这些不一致性究竟是 bug，还是反映了搜索功能中潜在的设计缺陷。
- **Felo 在学术搜索方面表现更优**：许多用户发现 **Felo** 在学术研究中更有效，称其比 **Perplexity** 能更好地获取相关论文。
   - 诸如 *悬停预览来源* 等功能增强了研究体验，吸引用户因其直观的界面而倾向于选择 Felo。
- **不一致的 API 输出令用户沮丧**：社区讨论了 **API 的不一致性**，特别是 **PPLX API**，与网站数据相比，该 API 返回的是过时的 **房地产列表**。
   - 有建议提出通过实验 **temperature** 和 **top-p** 等参数来提高 API 响应的一致性。
- **精神分裂症治疗取得突破**：Perplexity AI 宣布了一个重要里程碑，**30 年来首款精神分裂症药物发布**，标志着心理健康解决方案的重大进展。
   - 讨论强调了这对患者护理的潜在影响以及未来治疗模式的演变。
- **德克萨斯州各县有效利用 AI 技术**：德克萨斯州各县展示了在地方政府运营中利用 AI 应用的创新方法，增强了公共服务能力。
   - 参与者分享了一份详细资源，重点介绍了 AI 技术在行政任务中的这些实际应用。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 深受速率限制困扰**：用户报告在使用 **Gemini Flash** 时频繁出现 **429 错误**，在等待 Google 可能增加的配额期间感到非常沮丧。
   - 这一持续的流量问题正在损害平台的可用性，影响用户参与度。
- **维护后性能下降**：像 **Hermes 405B free** 这样的模型在近期更新后表现出较低的性能质量，引发了对模型提供商可能发生变化的担忧。
   - 建议用户检查其 **Activity pages** 以确保他们使用的是首选模型。
- **翻译模型选项建议**：一位用户正在寻找没有严格限制的高效对话翻译模型，并对 **GPT4o Mini** 表示不满。
   - 推荐使用经过 dolphin 技术微调的开源权重模型作为更灵活的替代方案。
- **前端聊天 GUI 推荐**：关于允许中间件灵活性的聊天 GUI 解决方案展开了讨论，**Streamlit** 被提议为一个可行的选择。
   - **Typingmind** 也因其在管理多个 AI Agent 交互时的可定制特性而被提及。
- **关于 Gemini 搜索功能的讨论**：用户有兴趣在 **Gemini** 模型中启用类似于 **Perplexity** 的直接搜索功能，尽管目前的使用限制仍在评估中。
   - 讨论引用了 Google 的 **Search Retrieval API 参数**，强调需要更清晰的实施策略。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux 模型大获成功**：受 kohya_ss 工作的启发，成员们注意到 **Flux 模型** 仅需 **12G VRAM** 即可进行训练，展示了令人惊叹的性能实力。
   - 这种进步带来的兴奋感正在蔓延，暗示着模型效率基准可能发生转变。
- **Nvidia 驱动导致 SDXL 变慢**：新的 Nvidia 驱动导致 **8GB VRAM 显卡** 出现严重减速，图像生成时间从 **20 秒激增至 2 分钟**。
   - 成员们强烈建议不要更新驱动，因为这些变化对他们的渲染工作流产生了不利影响。
- **区域提示词遇到障碍**：社区成员分享了在 Stable Diffusion 中使用 **区域提示词 (regional prompting)** 的挫败感，特别是在处理如 *“2 个男孩和 1 个女孩”* 这种提示词中的角色混淆问题时。
   - 建议从更广泛的提示词开始，利用通用指南以获得最佳效果。
- **AI 艺术投稿征集**：社区受邀提交 AI 生成的艺术作品，有机会入选 **The AI Art Magazine**，截止日期定为 **10 月 20 日**。
   - 该倡议旨在庆祝数字艺术，并鼓励成员展示他们的创造力。
- **AI 艺术引发质量辩论**：关于 **AI 艺术** 与人类艺术优劣的激烈辩论爆发，观点在质量和深度上产生了分歧。
   - 一些人主张人类艺术创作的优越性，而另一些人则为 AI 生成的作品辩护，认为其是合法的艺术表达。

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Aider 评测 LLM 编辑能力**：成员们讨论了 Aider 的功能，指出它在擅长代码“编辑”的 LLM 上表现出色，正如其[排行榜](https://aider.chat/docs/leaderboards/)所强调的那样。对于 Aider 基准测试的可靠性存在一些质疑，特别是关于 *Gemini Pro 1.5 002* 的部分。
   - 虽然 Aider 展示了令人印象深刻的编辑能力，但进一步测试和验证的潜力对于获得社区更广泛的认可仍然至关重要。
- **欧盟 AI 法案引发对话**：围绕欧盟 AI 法案的讨论升温，成员们就其对**多模态 AI 监管**的影响以及二级法规下的聊天机器人分类发表了不同看法。对科技公司监管负担的担忧十分普遍。
   - 许多人强调，在应对合规环境时，必须明确新兴 AI 技术将如何受到这些法规的影响。
- **Meta 在视频翻译领域的重大突破**：一位成员强调了 Meta 即将发布的唇形同步视频翻译功能，旨在增强平台上的用户参与度。这一功能引发了关于其重塑内容创作工具潜力的讨论。
   - 成员们对这如何提升翻译服务以及对全球内容可访问性的影响表示兴奋。
- **GPT-4 语音模式的困惑**：对 **GPT-4o** 表现的挫败感正在酝酿，在有人称其为“最笨的 LLM”后，出现了要求发布 **GPT-4.5-o** 的紧急呼声。批评集中在推理能力不足这一主要问题上。
   - 在用户困惑中，关于每日限制和**语音模式**可访问性的详细讨论突显了社区对提升用户体验的期待。
- **Flutter 代码执行错误已解决**：一位用户遇到了指示线程 `thread_ey25cCtgH3wqinE5ZqIUbmVT` 中存在活动运行的错误，导致了关于管理活动运行和使用 `cancel` 函数的建议。该用户最终通过在执行之间等待更长时间解决了问题。
   - 参与者建议加入状态参数来跟踪线程完成情况，这可能会简化线程管理并减少未来交互中的挫败感。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **新成员丰富社区动态**：几位新成员加入，包括来自新加坡的全栈工程师和来自葡萄牙的数据工程师，渴望为 AI 项目和开源倡议做出贡献。
   - 他们对协作的热情为社区发展奠定了良好的基调。
- **AI 会议即将召开**：成员们讨论了即将举行的会议，如 **ICLR** 和 **NeurIPS**，特别是新加坡将主办 ICLR，并正在计划聚会。
   - 关于活动安保角色的轻松对话为协调工作增添了趣味。
- **Liquid AI 发布基础模型**：[Liquid Foundation Models](https://www.liquid.ai/liquid-foundation-models) 正式发布，展示了强大的基准测试分数和针对不同行业优化的灵活架构。
   - 这些模型专为各种硬件设计，邀请用户在 Liquid AI 的平台上进行测试。
- **探索 vLLM 指标提取**：一位成员询问如何使用基准测试上的 `simple_evaluate` 函数从 **lm-evaluation-harness library** 中提取 **vLLM metrics objects**。
   - 他们特别寻求诸如 **time to first token** 和 **time in queue** 等指标，引发了社区的有益回应。
- **ExecuTorch 增强设备端 AI 能力**：根据平台概述，**ExecuTorch** 允许在各种设备（包括 AR/VR 和移动系统）上**定制和部署** PyTorch 程序。
   - 分享了关于 `executorch` pip 包的详细信息，该包目前处于 Python **3.10 和 3.11** 的 Alpha 阶段，兼容 **Linux x86_64** 和 **macOS aarch64**。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **优化 Torchtune 训练配置**：用户针对 **Llama 3.1 8B** 微调了各种设置，优化了 `batch_size`、`fused` 和 `fsdp_cpu_offload` 等参数，在启用 `packed=True` 时缩短了 epoch 时间。
   - *……大家一致认为 `enable_activation_checkpoint` 应保持为 `False` 以提升计算效率。*
- **对动态 CLI 方案的需求**：有人提议使用 `tyro` 库创建一个动态 CLI，允许根据 Torchtune recipes 中的配置设置自定义帮助文本。
   - 这种灵活性旨在通过清晰的文档增强用户体验并简化 recipe 管理。
- **内存优化策略揭晓**：成员们建议更新内存优化页面，同时包含**性能和内存优化技巧**，提倡一种更集成的方法。
   - 实施 **sample packing** 和探索 **int4 training** 等想法被强调为提升内存效率的潜在增强方案。
- **分布式训练的错误处理增强**：有人建议利用 `torch.distributed` 的 record 工具记录异常，从而改进分布式训练中的错误处理。
   - 这种方法通过在整个训练过程中维护全面的错误日志，使故障排除变得更加容易。
- **配置管理中的重复键问题**：关于 **OmegaConf** 标记配置中重复条目（如 `fused=True`）的讨论引发了关注，强调了保持配置文件整洁有序的重要性。
   - *我们应该在配置中添加一个性能部分，* 将快速选项放在注释中，以提高可读性和即时访问性。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **CodiumAI 获得 A 轮融资并更名**：QodoAI（原名 CodiumAI）获得了 **4000 万美元** 的 A 轮融资，总融资额达到 **5000 万美元**，用于增强 **AI-assisted tools**。
   - *“这笔资金验证了他们的路线”*，表明开发者支持他们确保代码完整性的使命。
- **Liquid Foundation Models 宣称拥有令人印象深刻的基准测试结果**：LiquidAI 推出了 **LFMs**，展示了在 **MMLU** 和其他基准测试中的卓越性能，并指出了竞争对手的效率低下。
   - 凭借来自 MIT 的团队成员，其 **1.3B model** 架构将挑战行业内的既有模型。
- **Gradio 实现实时 AI 语音交互**：LeptonAI 展示了 **Gradio 5.0**，其中包括 LLM 的音频模式实时流媒体，简化了代码集成。
   - 这些更新使开发者能够轻松创建交互式应用程序，鼓励开源协作。
- **Ultralytics 发布 YOLO11**：Ultralytics 推出了 **YOLO11**，对先前版本进行了增强，提高了**computer vision tasks**中的准确性和速度。
   - 此次发布标志着其 YOLO 模型演进的关键一步，展示了实质性的性能提升。
- **播客听众要求更多研究员参与**：最新一期节目邀请了 **Shunyu Yao** 和 **Harrison Chase**，吸引了渴望在未来节目中看到更多**研究员参与**的听众的兴趣。
   - 互动凸显了听众的热情，评论如*“邀请更多研究员来”*，敦促进行更深入的讨论。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **用于公共金融数据的 FinanceAgentToolSpec**：LlamaHub 上的 [FinanceAgentToolSpec](https://t.co/7bsEm4Er1m) 软件包允许 Agent 访问来自 **Polygon** 和 **Finnhub** 等来源的公共金融数据。
   - Hanane 的详细帖子强调了该工具如何通过查询来简化金融分析。
- **全栈 Demo 展示流式事件**：一个新的 [全栈应用程序](https://t.co/HOajPyiqQb) 展示了具有 Human In The Loop 功能的流式事件 (Streaming Events) 工作流。
   - 该应用演示了如何研究和展示一个主题，显著提升了用户参与度。
- **YouTube 教程增强对工作流的理解**：一段 [YouTube 视频](https://t.co/Nn5NVZopPz) 提供了开发者对全栈 Demo 编码过程的演练。
   - 该资源旨在帮助那些希望实现类似流式系统的开发者。
- **应对 RAG Pipeline 评估挑战**：用户报告了使用 trulens 进行 RAG Pipeline 评估时遇到的问题，特别是关于导入错误和数据检索的问题。
   - 这引发了关于构建坚实的评估数据集以进行准确评估的重要性讨论。
- **理解 LLM 推理问题**：定义推理问题的类型对于处理 LLM 推理至关重要，正如一篇分享的文章中所详述的推理类型。
   - 文章强调，各种推理挑战需要量身定制的方法才能进行有效的评估。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Startup Program 提供折扣**：一位用户询问了使用 Cohere 的创业团队的折扣情况，并提到了与 **Gemini** 相比的成本。建议他们申请 [Cohere Startup Program](https://cohere.com/startup-program) 以寻求潜在的减免。
   - 参与者提到申请过程可能需要时间，但他们肯定了这种支持对早期创业公司的重要性。
- **通过 Fine-tuning 改进抽认卡生成**：成员们讨论了专门针对笔记和幻灯片生成抽认卡 (Flash Card) 的 **Fine-tuning 模型**，以解决输出清晰度的问题。建议采用机器学习 Pipeline 的最佳实践，并利用 **Chunking data** 来获得更好的结果。
   - Chunking 被强调为非常有益，特别是对于处理 PDF 幻灯片，能增强模型的理解和定性输出。
- **文化多语言 LMM 基准测试发布**：MBZUAI 正在开发一个针对 **100 种语言** 的 **Cultural Multilingual LMM Benchmark**，并正在积极寻求母语翻译志愿者进行纠错。成功的参与者将被邀请共同撰写最终论文。
   - 语言范围包括 **印度**、**南亚**、**非洲** 和 **欧洲** 语言，感兴趣的人士可以通过 **LinkedIn** 与项目负责人联系。
- **用于 LLM Prompt 的 RAG Header 格式化**：用户寻求关于 RAG Prompt 的 **Instructional Headers** 格式化指导，以确保 LLM 正确解释输入。讨论强调了精确的辅助信息和正确的 Header 终止方法的必要性。
   - 对话强调了格式的清晰度如何减少模型响应中的错误，从而增强与 LLM 的交互。
- **发现 API 文档中的缺失**：一位用户注意到 API 文档中关于惩罚范围的不一致，呼吁对参数值建立更清晰的标准。这次对话反映了用户在利用 API 功能时对 **文档一致性** 和清晰度的持续关注。
   - 关于从 v1 迁移到 v2 API 的讨论证实，虽然旧功能仍然保留，但系统性的更新对于平滑过渡至关重要。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 因薪酬需求引发的人才流失**：OpenAI 的核心研究人员正在寻求更高的薪酬，随着公司估值的上升，通过出售利润单位已套现 **12 亿美元**。由于 Safe Superintelligence 等竞争对手积极招募人才，这种人员流动进一步加剧。
   - *员工因资金问题威胁辞职*，而新任 CFO **Sarah Friar** 正在处理这些谈判。
- **加州州长否决 AI 安全法案 SB 1047**：州长 **Gavin Newsom** 否决了旨在监管 AI 公司的法案，声称这不是保护公众的最佳方法。批评者认为这是监管的挫折，而支持者则推动基于特定能力的监管。
   - *参议员 Scott Wiener 对州长缺乏事先反馈表示失望*，强调加州失去了在技术监管方面领先的机会。
- **PearAI 面临代码窃取指控**：**PearAI** 被指控从 [Continue.dev](http://Continue.dev) 窃取代码并在未致谢的情况下重新包装，敦促 YC 等投资者推动问责。这引发了关于初创生态系统内资金来源的重大伦理担忧。
   - *这一争议突显了人们对开源社区完整性*及其被新兴技术公司对待方式的持续关注。
- **关于 OpenAI 研究透明度的辩论**：批评者质疑 OpenAI 的透明度，强调引用博客并不等同于对研究结果的实质性沟通。一些员工则断言公司对他们的研究确实是开放的。
   - *讨论突显了人们对于* [OpenAI 的研究博客](https://openai.com/index/learning-to-reason-with-llms/) 是否充分解决了社区对透明度担忧的复杂情绪。
- **关于 iPhone IAP 订阅访问的见解**：一位 Substack 畅销作者宣布获得了 **iPhone In-App Purchase (IAP) 订阅**的访问权限，预示着移动端变现的新机会。这一进展为实施和管理这些系统提供了见解。
   - *讨论反映了开发者对管理 **Apple App Store** 混乱环境的挫败感*以及他们在处理其复杂性方面的经验。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **课程材料已开放获取**：学生可以在 [课程网站](https://llmagents-learning.org/f24) 上访问所有课程材料，包括作业和讲座录像，提交截止日期定为 **12 月 12 日**。
   - *定期检查网站以获取材料更新也非常重要。*
- **Multi-Agent 系统 vs. Single-Agent 系统**：讨论中出现了在项目背景下需要 Multi-Agent 系统而非 Single-Agent 实现的需求，以减少幻觉并管理上下文。
   - 参与者指出，这些系统可能会从 **LLM** 中获得更准确的响应。
- **对 NotebookLM 能力的好奇**：成员询问 **NotebookLM** 是否作为 Agent 应用运行，揭示了它作为一个 RAG Agent，可以总结文本并生成音频。
   - 关于其技术实现，特别是在多步流程中的实现，也出现了一些问题。
- **等待培训时间表确认**：学生们渴望确认培训课程何时开始，其中一人指出所有实验预计将在 **10 月 1 日**发布。
   - *然而，这一时间表尚未得到官方确认。*
- **探索 Super-Alignment 研究**：一个拟议的研究项目正在讨论中，旨在利用 **AutoGen** 等框架研究 Multi-Agent 系统中的伦理问题。
   - 提出了在没有专用框架的情况下实施该研究的挑战，突显了模拟能力的局限性。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **云存储成本与主流供应商相比具有竞争力**：George 提到，**存储和出站流量成本 (egress costs)** 将低于或等于主流云供应商，并强调了成本考量。
   - 他进一步解释说，对使用情况的预期可能会显著改变感知的成本。
- **Modal 的付费模式引发辩论**：Modal 独特的按秒计费算力资源定价吸引了关注，被吹捧为**比传统的按小时计费更便宜**。
   - 成员们质疑这种模式的可持续性，以及它如何与 AI 初创公司环境中持续的使用模式保持一致。
- **使用状态机改进 tinygrad 的 Matcher**：一位成员建议实现一个**匹配器状态机 (matcher state machine)** 可以提高性能，使其趋向于类 C 的效率。
   - George 热情地支持这种方法，表示它可以实现预期的性能提升。
- **需要全面的回归测试**：有人对优化器缺乏**回归测试套件 (regression test suite)** 表示担忧，这可能导致代码更改后出现未被察觉的问题。
   - 成员们讨论了通过序列化来检查优化模式的想法，但意识到这可能并不吸引人。
- **悬赏任务不强制要求 SOTA GPU**：一位成员建议，虽然 **SOTA GPU** 会有帮助，但使用普通的 GPU 也能应付，尤其是对于某些任务。
   - 某些任务（如 **tinygrad 中 100+ TFLOPS 的 matmul**）可能需要特定的硬件（如 **7900XTX**），而其他任务则不需要。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Llama 3.2 微调遭遇显存 (VRAM) 瓶颈**：用户在微调 **Llama 3.2 1b** 时，使用 qlora 和 4bit 加载等设置面临 **24GB** 的**高显存占用**，引发了关于平衡序列长度和 batch size 的讨论。
   - 担忧特别集中在样本打包 (sample packing) 的影响上，强调了在微调配置中进行优化的必要性。
- **加州强制要求 AI 训练透明化**：一项新的**加州法律**现在要求**披露所有 AI 模型的训练来源**，即使是较小的非营利组织也不例外。
   - 这促使人们开始讨论利用轻量级聊天模型来创建合规数据集，社区成员正在集思广益潜在的变通方案。
- **轻量级聊天模型受到关注**：成员们正在探索从网页爬取的数据集中**微调轻量级聊天模型**，旨在满足法律转换标准。
   - 一位用户指出，通过 LLM 优化凌乱的**原始网页爬取数据**可能是该过程中的重要下一步。
- **Liquid AI 引发好奇**：新基础模型 **Liquid AI** 的推出因其潜在特性和应用引起了成员们的兴趣。
   - 成员们热衷于讨论立法变化对该模型意味着什么，以及结合近期发展的实际影响。
- **在 Axolotl 中最大化数据集利用率**：在 **Axolotl** 中，通过调整数据集设置中的 `split` 选项，可以将数据集配置为使用前 **20%** 进行训练。
   - 由于 Axolotl 直接缺乏随机样本选择功能，用户必须对数据进行预处理，在加载前利用 Hugging Face 的 `datasets` 进行随机子集采样。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 展示实时 Pydantic 模型生成**：一场实况编程演示了如何使用 [Groq](https://groq.com) 和 **GitHub Actions** 创建一个**免费的 Pydantic 模型生成器**。
   - 参与者可以在分享的 [Loom 视频](https://www.loom.com/share/783ed4d80720492da23f39d2678de27f)中观看详细演示。
- **升级到 DSPy 2.5 带来显著改进**：切换到带有 **LM client** 的 **DSPy 2.5**，并使用 **Predictor** 代替 **TypedPredictor**，显著提升了性能并减少了问题。
   - 关键的增强源于新的 **Adapters**，它们现在对 **chat LMs** 有更好的感知能力。
- **OpenSearchRetriever 准备好分享**：如果社区表现出兴趣，一位成员愿意分享他们为 DSPy 开发的 [OpenSearchRetriever for DSPy](https://link.to.github)。
   - 该项目可以简化集成和功能实现，社区鼓励他们提交一个 **PR**。
- **医疗欺诈分类中的挑战**：一位成员在准确分类 **DOJ**（美国司法部）新闻稿中的**医疗欺诈**时遇到困难，导致了误分类。
   - 社区讨论了细化分类标准，以提高这一关键领域的准确性。
- **解决长 Docstring 引起的困惑**：在使用 **docstrings** 进行长解释时出现了困惑，影响了类签名的准确性。
   - 成员们就清晰文档的重要性提供了见解，但用户需要明确所使用的语言模型。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **全栈开发人员寻求项目**：一位全栈开发人员正在寻找新客户，擅长使用 **React + Node** 和 **Vue + Laravel** 技术构建**电子商务平台**、在线商店和房地产网站。
   - 他们对长期合作的讨论持开放态度。
- **关于重新指令 AI 执行的询问**：一位成员询问是否可以修改 **AI 执行指令**，以便用户能够独立修复和调试问题，并指出了频繁出现的路径相关错误。
   - 成员对当前系统的能力表达了明显的挫败感。
- **持续的解码数据包错误**：用户报告了一个反复出现的**解码数据包问题**，在服务器重启或客户端连接期间出现错误消息：*Invalid data found when processing input*。
   - 虽然建议检查终端错误消息，但未发现任何信息，表明问题具有持续性。
- **Ngrok 身份验证困难**：一位成员在执行服务器时遇到了 **ngrok 身份验证错误**，要求提供验证过的账户和 authtoken。
   - 他们怀疑问题可能与 .env 文件未能正确读取 *apikey* 有关，并就此寻求帮助。
- **Jan AI 作为计算机控制接口**：一位成员分享了将 **Jan AI** 与 **Open Interpreter** 结合使用作为本地 LLMs 的本地推理服务器的见解，并邀请他人分享经验。
   - 他们提供了一个 [YouTube 视频](https://www.youtube.com/watch?v=1l3B0AzbbjQ)，展示了 Jan 如何通过接口控制计算机。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **征集法语音频数据集**：一位用户需要高质量的**法语**音频数据集来训练 **CosyVoice**，并强调了获取合适数据集的紧迫性。
   - 他们表示，*如果没有合适的数据集*，不确定项目能否继续推进。
- **LAION 在版权挑战中获胜**：**LAION** 在**德国法院**赢得了一场重大的版权侵权挑战，为 AI 数据集的法律障碍树立了先例。
   - 进一步的讨论强调了这次胜利的影响，详情可以在 [Reddit](https://www.reddit.com/r/aiwars/comments/1fqpiut/laion_wins_first_copyright_infringement_challenge/) 上找到。
- **使用 Phenaki 探索文本转视频**：成员们探索了用于从文本生成视频的 **Phenaki** 模型，并分享了一个用于初步测试的 [GitHub 链接](https://github.com/lucidrains/make-a-video-pytorch)。
   - 由于缺乏数据集，他们请求关于测试其能力的指导。
- **视觉语言与潜在扩散模型之间的协同作用**：讨论围绕结合 **VLM**（视觉语言模型）和 **LDM**（潜在扩散模型）以增强图像生成的潜力展开。
   - 提出了一个理论循环，即由 **VLM** 指导 **LDM**，从而有效优化输出质量。
- **澄清 PALM-RLHF 数据集的实现**：一位成员询问了关于为特定任务定制 **PALM-RLHF** 训练数据集的合适频道。
   - 他们的目标是明确如何使这些数据集与操作需求保持一致。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Vectorstores 可以使用示例问题**：一位成员建议，加入示例问题可能会增强 Vectorstore 在寻找最接近匹配时的性能，尽管这可能被认为有些过度。
   - 他们强调了**测试**的重要性，以衡量这种方法的实际有效性。
- **对于 LLMs 而言，数据库优于表格数据**：一位成员指出，从表格数据切换到 **Postgres** 数据库更适合 LLMs，这促使他们利用 **LangChain 模块**进行交互。
   - 这一转变旨在优化模型训练和查询的数据处理。
- **在 Discord 中探索感谢礼物**：有人询问了向提供帮助的 Discord 成员发送小型感谢礼物的可行性。
   - 这反映了致谢贡献并建立社区纽带的愿望。
- **Gemini 突然出现图像错误**：一位成员报告了向 **Gemini** 发送图像时出现的意外错误，并指出该问题是在最近升级了所有 **pip 包**之后出现的。
   - 这种情况引发了对升级后潜在兼容性问题的担忧。
- **使用 LangChain 修改推理方法**：一位成员正在研究使用 **LangChain** 修改聊天模型的推理方法，重点关注 **vllm** 中的优化。
   - 他们寻求控制 token 解码，特别是围绕聊天历史和输入调用（input invocation）。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **2024 年 AI Realized 峰会定于 10 月 2 日举行**：由 **Christina Ellwood** 和 **David Yakobovitch** 在 UCSF 主办的 [AI Realized - 企业级 AI 峰会](https://lu.ma/airsummit) 备受期待，届时将有企业级 AI 领域的行业领袖出席。
   - 参会者可以使用代码 **extra75** 在购票时立减 **$75**，门票包含会议期间的膳食。
- **Manifold Research 前沿讲座启动**：**Manifold Research** 正在推出 Frontiers 系列讲座，以展示基础和应用 AI 领域的创新工作，首场讲座由 **Helen Lu** 主讲，重点关注神经符号 AI 和人机协作。
   - 讲座将讨论自主 Agent 在动态环境中面临的挑战，并开放免费注册，链接在[此处](https://lu.ma/cbflyi6s)。
- **咨询斯德哥尔摩的 MLOps 聚会**：一位最近搬到斯德哥尔摩的成员正在寻找该市关于 **MLOps 或基础设施的聚会**信息。
   - 他们表达了与当地技术社区建立联系并了解即将举行的活动的愿望。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Calytrix 推出 anti-slop 采样器**：一个原型 **anti-slop 采样器**通过对检测到的序列进行回溯，在推理过程中抑制不需要的词汇。Calytrix 旨在使该代码库可用于下游用途，该项目已在 [GitHub](https://github.com/sam-paech/antislop-sampler) 上发布。
   - 这种方法旨在通过减少生成输出中的噪声来直接提高**数据集质量**。
- **社区支持 anti-slop 概念**：成员们对 **anti-slop 采样器**分享了积极的反馈，其中一位评论道：*“太酷了，我喜欢这个主意！”*，强调了其潜在影响。
   - 这种热情表明人们对优化**数据集生成**过程的解决方案越来越感兴趣。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Takiyoshi Hoshida 展示 SoraSNS**：独立开发者 **Takiyoshi Hoshida** 将现场演示他的项目 **[SoraSNS](https://discord.com/events/1089876418936180786/1277835047084363827)**，这是一款社交媒体应用，提供来自你通常不关注的用户的私人时间线。
   - 演示强调了该应用独特的**昼夜**天空概念，象征着开放和远距离观察，以增强用户体验。
- **Hoshida 令人印象深刻的技术背景**：Takiyoshi Hoshida 曾在**卡内基梅隆大学学习计算机科学**，这为他打下了坚实的技术基础。
   - 他拥有丰富的经验，此前曾在**苹果的 AR Kit 团队**工作，并参与了超过 **50 个 iOS 项目**。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Hammer Handle 变得更好了**：Hammer Handle 进行了更新，在设计和功能上引入了增强。期待在这个迭代版本中看到许多*令人兴奋的改进*。
   - 此次更新标志着团队致力于不断提高工具的可用性。
- **了解 Hammer2.0 系列模型**：团队推出了 **Hammer2.0 系列模型**，包括 Hammer2.0-7b、Hammer2.0-3b、Hammer2.0-1.5b 和 Hammer2.0-0.5b。
   - 这些模型标志着开发应用产品多样化方面的重要进展。
- **提交了新的 Pull Request PR#667**：作为 Hammer 产品线程序化更新的一部分，已提交了一个 Pull Request (PR#667)。这次提交对正在进行的开发过程至关重要。
   - 该 PR 旨在整合最近的增强功能和来自社区的反馈。



---

# 第 2 部分：按频道划分的详细摘要和链接


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1289300567323316224)** (920 条消息🔥🔥🔥): 

> - `LinkedIn 与开源问题`
> - `微调 Llama 模型`
> - `模型加载问题`
> - `在 Unsloth 中使用 BitsAndBytes`
> - `Google Colab 使用` 


- **LinkedIn 抄袭代码争议**：LinkedIn 被指控在没有适当署名的情况下抄袭 Unsloth 的代码，并声称是自主研发。这引发了 Microsoft 和 GitHub 的干预，最终要求 LinkedIn 正确署名 Unsloth。
   - 这一事件凸显了对知识产权的担忧以及遵守开源许可（open source licensing）的重要性。
- **微调模型的最佳实践**：建议在微调模型时设置随机种子（random seed）以确保可复现性，并使用一种方法彻底评估输出质量。建议使用提示词（prompts）列表进行人工评估，以深入了解模型性能。
   - 响应格式和上下文微调（context tuning）等各种参数会显著影响微调过程的效果。
- **模型加载挑战**：用户在尝试使用 Unsloth 库加载微调后的模型时，遇到了与模型配置文件相关的运行时错误。问题主要源于在同一个仓库中同时存在 LoRA adapters 和基础模型配置。
   - 建议升级 Unsloth 库以解决与模型加载相关的特定 bug。
- **在 Unsloth 中使用 BitsAndBytes**：BitsAndBytes 允许以量化格式加载模型，用户可以加载 4-bit 或 8-bit 配置的模型。虽然可以在 4-bit 下进行微调，但建议在训练后以 16-bit 加载模型，以获得更好的推理性能。
   - 建议用户确保使用正确的参数，以避免在模型训练和推理过程中产生混淆。
- **Google Colab 入门**：新用户被引导至有效使用 Google Colab 的资源，包括带有清晰指令的 notebook 链接。为初学者推荐了几个模型，以便以用户友好的格式实验和探索功能。
   - 这确保了新手能够快速适应可用于微调和部署模型的资源。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/CodeFryingPan/status/1840203597478539477">来自 FRYING PAN (@CodeFryingPan) 的推文</a>：我刚刚辞掉了在 Coinbase 年薪 27 万美元的工作，与我的联合创始人 @not_nang 一起加入 YCombinator 的首个秋季批次。我们正在构建 PearAI，一个开源 AI 代码编辑器。可以把它看作是一个更好的 Copilot，或者开源的...</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing>">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - ggml-org 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/RhysSullivan/status/1840461449371812289">来自 Rhys (@RhysSullivan) 的推文</a>：介绍 BlueberryAI，开源 AI 驱动的代码编辑器。它是 PearAI 的 fork，而 PearAI 是 Continue 的 fork，Continue 又是 VSCode 的 fork。投资者们，我的私信已为种子轮（seed round）开放。</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">我们所有的模型 | Unsloth 文档</a>：查看下方列表，了解我们上传的所有 GGUF、16-bit 和 4-bit bnb 模型</li><li><a href="https://github.com/linkedin/Liger-Kernel/commit/376fe0c2af65ff4d716dc36eb6fe5231662920a7">在 header 中引用 Unsloth (#216) · linkedin/Liger-Kernel@376fe0c</a>：## 摘要
 在 header 部分引用 Unsloth
 
 &amp;lt;!---
 ## 详情
 这是一个可选部分；是否有任何特定的内容需要评审者注意？
 ---&amp;gt;
 </li></ul></div>

## Testing Done...</li><li><a href="https://www.llama.com/docs/how-to-guides/fine-tuning">Fine-tuning | 操作指南</a>: 全参数微调（Full parameter fine-tuning）是一种对预训练模型所有层的所有参数进行微调的方法。 </li><li><a href="https://www.youtube.com/watch?v=YZW3pkIR-YE">微调 Llama-3.2 并在 Ollama 中运行的最简单方法</a>: Meta 最近发布了 Llama 3.2，本视频演示了如何使用 Unsloth 微调 30 亿参数的指令模型，并使用 Ollama 在本地运行...</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://github.com/bitsandbytes-foundation/">bitsandbytes foundation</a>: bitsandbytes foundation 有 2 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes/multilingual/README.md">llama-recipes/recipes/multilingual/README.md at 0efb8bd31e4359ba9e8f52e8d003d35ff038e081 · meta-llama/llama-recipes</a>: 使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama 的脚本，涵盖单节点/多节点 GPU。支持用于摘要和问答等应用的默认及自定义数据集...</li><li><a href="https://github.com/unslothai/unsloth/wiki">首页</a>: 微调 Llama 3.1, Mistral, Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/1061">[已修复] RuntimeError: Unsloth: 你的仓库包含一个 LoRA 适配器和一个基础模型。 · Issue #1061 · unslothai/unsloth</a>: 我成功训练了 unsloth/Llama-3.2-3B-Instruct-bnb-4bit 模型，但当我尝试通过 astLanguageModel.from_pretrained 使用它时，遇到了这个错误：Traceback (most recent call last): Fil...</li><li><a href="https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py">trl/examples/scripts/sft_vlm.py at main · huggingface/trl</a>: 使用强化学习训练 Transformer 语言模型。 - huggingface/trl</li><li><a href="https://github.com/PygmalionAI/aphrodite-engine">GitHub - PygmalionAI/aphrodite-engine: 大规模 LLM 推理引擎</a>: 大规模 LLM 推理引擎。通过在 GitHub 上创建账号为 PygmalionAI/aphrodite-engine 的开发做出贡献。</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/discussions/1375">v0.44.0: 新的 AdEMAMix 优化器、嵌入量化（Embeddings quantization）等！ · bitsandbytes-foundation/bitsandbytes · Discussion #1375</a>: 新优化器：AdEMAMix。AdEMAMix 优化器是对 AdamW 的改进，建议跟踪两个 EMA 以更好地利用过去的梯度。这允许以更少的训练数据实现更快的收敛...</li><li><a href="https://github.com/unslothai/unsloth/issues/421">未找到 config.json 文件，使用 unsloth 微调 Llama 3 后将文件保存到 Hugging Face · Issue #421 · unslothai/unsloth</a>: 我使用 unsloth 微调 Llama 3-8B...，训练完成后我使用 'push_to_hub' 将模型保存到 Hugging Face，但它显示了这些文件：.gitattributes README.md adapter_config.js...</li><li><a href="https://github.com/unslothai/unsloth/issues/1062">[临时修复] Ollama / llama.cpp: 在模型文件中找不到 tokenizer merges [重复] · Issue #1062 · unslothai/unsloth</a>: 你好，我按照你在这里提供的 notebook 尝试微调了 Llama 3.1-8b-instruct 和 Llama 3-8b-instruct。训练阶段无误完成，我生成了 GGUF 量化...</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: C/C++ 环境下的 LLM 推理</a>: C/C++ 环境下的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/asmith26/unsloth/blob/main/KTO_%2B_Phi_3_Mini_4K_Instruct_%2B_Unsloth.ipynb">unsloth/KTO_+_Phi_3_Mini_4K_Instruct_+_Unsloth.ipynb at main · asmith26/unsloth</a>: 微调 Llama 3, Mistral, Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - asmith26/unsloth</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 微调 Llama 3.1, Mistral, Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80%</a>: 微调 Llama 3.1, Mistral, Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 微调 Llama 3.1, Mistral, Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80%</a>: 微调 Llama 3.1, Mistral, Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/huggingface/trl/issues/862">在 SFTTrainer 中计算生成任务的指标 · Issue #862 · huggingface/trl</a>: 你好，我想在 SFTTrainer 中包含一个基于自定义生成的 compute_metrics（例如 BLEU）。但是，我遇到了困难，因为：输入到 compute_metrics 的 eval_preds 包含一个 .predicti...</li><li><a href="https://github.com/

unslothai/unsloth/issues/1065">[临时修复] Ollama / llama.cpp: 在模型文件中找不到 tokenizer merges · Issue #1065 · unslothai/unsloth</a>: 感谢开发这个有用的资源。Ollama notebook 报告 {&quot;error&quot;:&quot;llama runner process has terminated: error loading modelvocabulary: cannot find tokenizer merges in ...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1289349351524663318)** (17 条消息🔥): 

> - `Compute utilization` (计算利用率)
> - `Software acceleration methods` (软件加速方法)
> - `Underutilized hardware performance` (未充分利用的硬件性能)


- **秘密计算洞察 (Secret Compute Insights)**：一位成员表示，有大量的 **算力 (compute power)** 尚未被开发，并强调了在各种硬件组件上进行改进的潜力。
   - *“如果允许我在这里分享一些东西而不展开细说，因为这是秘密的特权内容”* 暗示了利用这些未开发能力的未公开策略。
- **令人印象深刻的 4 倍推理加速**：另一位成员分享称，他们仅使用标准的 **Python** 就实现了 **4X 推理加速 (inference acceleration)**，而无需诉诸复杂的黑客手段或私有方法。
   - 这突显了简单的调整如何产生显著的性能提升，表明了进一步改进的未开发潜力。
- **硬件被大规模低效利用**：讨论集中在 **CPU 和 GPU** 被极大地低效利用，并声称系统的 **PCIe** 通道几乎处于闲置状态，这表明存在效率低下。
   - 核心观点是，即使没有硬件进步，仅通过整合现有的研究成果，也有实现 **10X 性能** 的清晰路径。
- **对性能洞察的反应**：一次幽默的交流中提到，一位成员的见解听起来类似于 **OpenAI 论文**，指出了所分享信息的神秘性。
   - 参与者们通过不提供细节的玩笑和正式的语气（将其比作 **TED** 演讲）来回应。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1289305839726760068)** (303 条消息🔥🔥): 

> - `Model Fine-Tuning Issues` (模型 Fine-Tuning 问题)
> - `GGUF Conversion Problems` (GGUF 转换问题)
> - `Tokenizer and EOS Token Issues` (Tokenizer 和 EOS Token 问题)
> - `Checkpoint Management in Training` (训练中的 Checkpoint 管理)
> - `Using Unsloth with Llama Models` (在 Llama 模型中使用 Unsloth)


- **Llama 模型 Fine-Tuning 的挑战**：用户讨论了与 Fine-Tuning Llama 模型相关的各种问题，特别是面临无限 Token 生成和保留原始能力的问题。
   - 关于使用 EOS Token 和模型配置导致推理期间出现问题的担忧被提出。
- **GGUF 转换过程中遇到的错误**：一名用户在尝试加载 Fine-Tuning 后的 GGUF 模型时遇到了错误，提示 'cannot find tokenizer merges in model file'。
   - 讨论表明，此问题可能源于将模型保存为 GGUF 格式过程中的问题。
- **不同训练方法的有效性**：讨论了在模型 Fine-Tuning 过程中使用各种 Rank 值、目标层以及添加 Embedding 层的有效性。
   - 建议使用 Base 模型，以避免用户在使用 Instruct 模型时遇到的问题。
- **Colab 中的 Checkpoint 管理**：用户分享了在训练期间有效管理 Checkpoint 的方法，以防止在 Google Colab 中丢失进度。
   - 强调了为保存模型 Checkpoint 设置适当参数以减轻运行时问题的重要性。
- **不同 Llama 模型的兼容性**：澄清了模型 'meta-llama/Meta-Llama-3.1-8B' 和 'unsloth/Meta-Llama-3.1-8B' 基本上是相同且兼容的。
   - 讨论还包括 Hugging Face 和 Unsloth 的模型 Checkpoint 之间的差异及其兼容性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1179035537009545276/1179777624986357780/1290255706053939243">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是玩游戏和与朋友放松，甚至建立全球社区的绝佳场所。自定义你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>: 又称 Continued Finetuning。Unsloth 允许你进行持续预训练，以便模型学习新语言。</li><li><a href="https://colab.research.google.com/drive/1oCEHcED15DzL8xXGU1VTx5ZfOJM8WY01?usp=sharing#scrollTo=6bZsfBuZDeCL">Google Colab</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2409.12917">Training Language Models to Self-Correct via Reinforcement Learning</a>: 自我修正是大语言模型 (LLMs) 一项非常理想的能力，但在现代 LLMs 中一直被发现很大程度上是无效的。现有的训练自我修正的方法...</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">Finetuning from Last Checkpoint | Unsloth Documentation</a>: Checkpointing 允许你保存 Fine-Tuning 进度，以便暂停后继续。</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-orpo-and-kto">Reward Modelling - DPO, ORPO &amp; KTO | Unsloth Documentation</a>: 要在 Unsloth 中使用 DPO, ORPO 或 KTO，请遵循以下步骤：</li><li><a href="https://docs.unsloth.ai/basics/chat-templates">Chat Templates | Unsloth Documentation</a>: 未找到描述</li><li><a href="https://github.com/codelion/optillm/blob/main/optillm/cot_decoding.py">optillm/optillm/cot_decoding.py at main · codelion/optillm</a>: 为 LLMs 优化推理代理。通过在 GitHub 上创建账号为 codelion/optillm 的开发做出贡献。</li><li><a href="https://github.com/EricLBuehler/xlora">GitHub - EricLBuehler/xlora: X-LoRA: Mixture of LoRA Experts</a>: X-LoRA: LoRA 专家混合。通过在 GitHub 上创建账号为 EricLBuehler/xlora 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/1061">[FIXED] RuntimeError: Unsloth: Your repo has a LoRA adapter and a base model. · Issue #1061 · unslothai/unsloth</a>: 我已经成功训练了 unsloth/Llama-3.2-3B-Instruct-bnb-4bit 模型，但当我尝试使用 astLanguageModel.from_pretrained 时，遇到了这个错误：Traceback (most recent call last): Fil...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L813)">unsloth/unsloth/tokenizer_utils.py at main · unslothai/unsloth</a>: 使用 80% 更少的显存，以 2-5 倍的速度 Fine-Tuning Llama 3.1, Mistral, Phi &amp; Gemma LLMs - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1289525835958714369)** (9 messages🔥): 

> - `LLM 和金融领域的审稿人角色`
> - `Liquid Foundation Models` 


- **寻求 LLM 和金融领域的审稿人**：一位成员询问了关于专注于 **LLM** 和 **金融/经济** 主题的科学期刊的潜在审稿人（referee）。
   - 在此背景下，“referee” 指的是科学期刊的 **审稿人（reviewer）**。
- **Liquid Foundation Models 发布**：一位成员分享了来自 [LiquidAI](https://x.com/LiquidAI_/status/1840768716784697688) 的帖子，宣布推出 **Liquid Foundation Models (LFMs)**，包括 **1B、3B 和 40B 模型**。
   - 然而，对于这些声明的有效性出现了质疑，一位成员对报告表示失望并质疑其准确性，特别是提到了 **Perplexity Labs** 的问题。



**提到的链接**：<a href="https://x.com/LiquidAI_/status/1840768716784697688">来自 Liquid AI (@LiquidAI_) 的推文</a>：今天我们向世界介绍 Liquid Foundation Models (LFMs)，以及我们的第一系列语言 LFMs：一个 1B、一个 3B 和一个 40B 模型。(/n)

  

---



### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1289969915960361072)** (1 messages): 

> - `Aider v0.58.0 特性`
> - `Architect/Editor 模型配对`
> - `新模型支持`
> - `会话增强`
> - `剪贴板命令更新` 


- **Aider v0.58.0 带来令人兴奋的特性**：最新版本 [Aider v0.58.0](https://aider.chat/2024/09/26/architect.html) 引入了各种增强功能，包括模型配对和新命令。
   - 值得注意的是，**Aider 编写了本次更新中 53%** 的代码，展示了其自动化能力。
- **Architect/Editor 模型配对提升编码体验**：用户现在可以将像 **o1-preview** 这样强大的推理模型作为其 Architect，同时使用像 **gpt-4o** 这样更快的模型作为其 Editor。
   - *这种配对旨在优化编码效率*，同时平衡性能和成本。
- **Aider 扩展了模型支持**：更新提供了对新 **Gemini 002** 模型和 **Qwen 2.5** 模型增强功能的支持。
   - 这些新增功能拓宽了用户在各种应用中可用的工具范围。
- **会话增强使使用更顺畅**：Aider 现在允许用户通过选择 **(D)on't ask again**（不再询问）来跳过许多确认问题，提升了用户体验。
   - 此外，`/read-only` 的自动补全现在支持 **整个文件系统**，使导航更高效。
- **剪贴板命令更新简化工作流**：新的 `/copy` 命令允许用户将最后一次 **LLM** 响应复制到剪贴板，而 `/clipboard` 已重命名为 `/paste`。
   - 此外，在抓取 URL 时现在会遵循 **HTTP 重定向**，改进了操作中的数据检索。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1289300553419198505)** (436 messages🔥🔥🔥): 

> - `Aider 的 Architect 和 Editor 模型`
> - `使用多个 LLM`
> - `DeepSeek 集成`
> - `Aider 用户工作流`
> - `Aider 中的 Prompt 配置` 


- **理解 Aider 的 Architect 和 Editor 模型**：Aider 使用一个主模型和一个可选的 editor 模型运行；architect 模式利用主模型进行规划，利用 editor 模型进行执行。
   - 用户可以在其配置文件中设置 `--editor-model` 来指定 editor 模型，而 architect 模式仍然是主要功能的一部分。
- **关于多智能体编码的讨论**：一位用户引用了两篇展示 **LLM** 多智能体（multi-agent）编码有效性的论文，引发了关于 Aider 是否计划推出类似功能的询问。
   - 建议将这些询问发布在 **GitHub** 上，以便获得更好的曝光和潜在的集成。
- **DeepSeek 在 Aider 中的角色**：鼓励用户尝试使用 **DeepSeek** 作为 editor 模型，以降低与 **o1-preview** 等更昂贵选项相比的成本。
   - 最近的更新合并了不同的 **DeepSeek** 模型，导致在具体使用哪个模型上产生了一些困惑。
- **用户反馈与建议**：用户注意到，虽然像 Sonnet 这样的 **LLM** 可以提供有用的模板，但在生成无关编辑方面存在问题。
   - 回复指出，在使用 **LLM** 进行代码编辑时，细小且详细的任务往往会产生更好的结果。
- **Aider 中的配置和命令语法**：用户讨论了 Aider 的 YAML 配置文件设置，特别是在为任务设置合适的模型方面。
   - 任务和设置的命令语法得到了澄清，强调了 Aider 的灵活性允许量身定制的用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://aider.chat/docs/troubleshooting/imports.html">依赖版本</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://docs.continue.dev/customize/deep-dives/codebase">@codebase | Continue</a>: 与你的代码库对话</li><li><a href="https://alexgarcia.xyz/blog/2024/sqlite-lembed-init/index.html">介绍 sqlite-lembed：一个用于在本地生成文本嵌入的 SQLite 扩展</a>: 使用 GGUF 模型在 SQL 中生成文本嵌入！</li><li><a href="https://www.answer.ai/posts/2024-09-03-llmstxt.html">/llms.txt — 一个为 LLM 提供网站使用信息的提案 – Answer.AI</a>: 我们建议那些有兴趣提供 LLM 友好内容的人在他们的网站上添加一个 /llms.txt 文件。这是一个提供简要背景信息和指南以及链接的 Markdown 文件...</li><li><a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/scripting.html">编写 aider 脚本</a>: 你可以通过命令行或 Python 为 aider 编写脚本。</li><li><a href="https://aider.chat/2024/09/26/architect.html">分离代码推理与编辑</a>: Architect 模型描述如何解决编程问题，而 Editor 模型将其转化为文件编辑。这种 Architect/Editor 方法产生了 SOTA 基准测试结果。</li><li><a href="https://aider.chat/">首页</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML 配置文件</a>: 如何使用 YAML 配置文件配置 aider。</li><li><a href="https://aider.chat/docs/install.html">安装</a>: 如何安装并开始使用 aider 进行配对编程。</li><li><a href="https://aider.chat/docs/config/options.html#--editor-model-editor_model">选项参考</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://github.com/sigoden/aichat/wiki/RAG-Guide">RAG 指南</a>: 一体化 AI CLI 工具，具有 Chat-REPL、Shell Assistant、RAG、AI 工具和 Agent 功能，支持访问 OpenAI、Claude、Gemini、Ollama、Groq 等。- sigoden/aichat</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>: LLM 代码编辑能力的量化基准测试。</li><li><a href="https://huggingface.co/Kortix/FastApply-v1_16bit_Qwen2.5-Coder-1.5B-ft">Kortix/FastApply-1.5B-v1_16bit_Qwen2.5-Coder-1.5B-ft · Hugging Face</a>: 未找到描述</li><li><a href="https://aider.chat/docs/config/options.html#--editor-edit-format-editor_edit_format">选项参考</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/config/options.html#--editor-">选项参考</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://rentry.org/aiderts">aider 中的 JavaScript / TypeScript</a>: 背景：aider 是一个强大的 AI 编程助手，自带 Linter 系统，但高级 JS/TS 模板语言（如 JSX/TSX 或 Svelte）允许在一个文件中包含多种不同的语言...</li><li><a href="https://aider.chat/docs/config/options.html#--deepseek">选项参考</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/benchmark/README.md">aider/benchmark/README.md (main 分支) · paul-gauthier/aider</a>: aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat">DeepSeek V2.5 - API、供应商、统计数据</a>: DeepSeek-V2.5 是结合了 DeepSeek-V2-Chat 和 DeepSeek-Coder-V2-Instruct 的升级版本。通过 API 运行 DeepSeek V2.5</li><li><a href="https://github.com/asg017/sqlite-vec?tab=readme-ov-file">GitHub - asg017/sqlite-vec: 一个可以在任何地方运行的向量搜索 SQLite 扩展！</a>: 一个可以在任何地方运行的向量搜索 SQLite 扩展！- asg017/sqlite-vec</li><li><a href="https://github.com/paul-gauthier/aider/issues/1818">功能请求：添加 --external-chat 开关，允许 Aider 接收在自定义文本编辑器中编写的单条消息 · Issue #1818 · paul-gauthier/aider</a>: 问题描述 --external-chat &lt;text_editor_path&gt; 开关将允许 Aider 接收在自定义文本编辑器中编写的单条消息。单独使用 --external-chat 开关将意味着读取...</li><li><a href="https://github.com/paul-gauthier/aider/issues/1315?">添加 `/editor` 命令？ · Issue #1315 · paul-gauthier/aider</a>: 问题：在 https://github.com/llm-workflow-engine/llm-workflow-engine 中，我实现了一个非常方便的 /editor 命令，它可以：打开 $EDITOR 环境变量中指定的 CLI 编辑器，...</li><li><a href="https://www.firecrawl.dev/">Firecrawl</a>: 将任何网站转换为 LLM 就绪的数据。</li><li><a href="https://github.com/paul-gauthier/aider/">GitHub - paul-gauthier/aider: aider 是你终端里的 AI 配对编程工具</a>: aider 是你终端里的 AI 配对编程工具。C

<li>通过在 GitHub 上创建账户，为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/issues/1839">如何添加 multi-agent 流？· Issue #1839 · paul-gauthier/aider</a>：aider.chat 关于 multi-agent 编程的计划是什么？有两篇论文 https://arxiv.org/pdf/2405.11403v1 和 https://arxiv.org/pdf/2402.16906v6 使用了多次 LLM 调用（以及调试器...</li><li><a href="https://github.com/paul-gauthier/aider/pull/1790">feat: 由 fry69 添加 cmd_copy 命令以将最后一条助手回复复制到剪贴板 · Pull Request #1790 · paul-gauthier/aider</a>：这添加了 /copy 命令，用于将 LLM 的最后一条回复复制到剪贴板。注意：由于行长度超过 100 个字符，flake8 强制将 /paste 的描述切分为两行。</li><li><a href="https://github.com/paul-gauthier/aider/commit/c2c4dbd2a8319f3eab72939f60e2b199a452ff1d">合并来自 jbellis/paste 的 Pull Request #1595 · paul-gauthier/aider@c2c4dbd</a>：feat: 将 /clipboard 重命名为 /paste</li><li><a href="https://github.com/paul-gauthier/aider/actions/workflows/docker-build-test.yml">Docker Build Test · Workflow 运行 · paul-gauthier/aider</a>：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账户，为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/fry69/aider/tree/copy-command">GitHub - fry69/aider 的 copy-command 分支</a>：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账户，为 fry69/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider.git">GitHub - paul-gauthier/aider：aider 是你终端里的 AI 配对编程工具</a>：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账户，为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://paperswithcode.com/sota/code-generation-on-humaneval">Papers with Code - HumanEval 基准测试（代码生成）</a>：目前 HumanEval 上的 SOTA 是 LDB（O1-mini，基于来自 Reflexion 的种子程序）。查看 138 篇带有代码的论文的完整对比。</li><li><a href="https://platform.deepseek.com/api-docs/updates/#version-2024-09-05">更新日志 | DeepSeek API 文档</a>：版本：2024-09-05</li><li><a href="https://github.com/paul-gauthier/aider/blob/0aaa37f528b6b8851fa35859cdb401cb71addde1/aider/args.py#L217">aider/aider/args.py 位于 0aaa37f528b6b8851fa35859cdb401cb71addde1 · paul-gauthier/aider</a>：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账户，为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/okwilkins/rag-cli">GitHub - okwilkins/rag-cli：一个展示成熟 RAG 系统良好 CLI 实践的项目。</a>：一个展示成熟 RAG 系统良好 CLI 实践的项目。- okwilkins/rag-cli</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/">RAG CLI - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/paul-gauthier/aider/pull/1823">doc: 由 fry69 提交的完整结果表热修复 · Pull Request #1823 · paul-gauthier/aider</a>：该问题的热修复：（通过 -> https://discord.com/channels/1131200896827654144/1131200896827654149/1289976901393453066）修复版本：</li><li><a href="https://github.com/paul-gauthier/aider/issues/1315">添加 `/editor` 命令？· Issue #1315 · paul-gauthier/aider</a>：在 https://github.com/llm-workflow-engine/llm-workflow-engine 中，我实现了一个非常方便的 /editor 命令，它可以：打开 $EDITOR 环境变量中指定的 CLI 编辑器，...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1289331299584180235)** (192 条消息🔥🔥): 

> - `Aider 配置`
> - `Architect 模式 vs Code 模式`
> - `模型的成本效益`
> - `使用多个 Git Worktrees`
> - `Prompt 缓存与 Token 管理`

- **理解 Aider 配置文件**：用户讨论了使用多个 `.aider.conf.yml` 文件来管理配置的可能性，并建议通过编写 Aider 脚本来实现更好的灵活性。
   - 关于是否有必要编写脚本，还是结构良好的配置文件就足以有效管理 Aider，存在一些争论。
- **Architect 模式输出代码**：有用户担心 Architect 模式会产生最终代码输出而不仅仅是规划，这导致了对其效用的困惑。
   - 有人澄清说，对于简单的任务，规划步骤可能是不必要的，这可能会导致 token 的浪费。
- **使用不同模型的成本效益**：使用 `claude-3.5-sonnet` 作为 architect 并使用 `deepseek v2.5` 作为 editor 被认为显著更便宜，估计 editor token 的成本可降低 20-30 倍。
   - 讨论强调了在使用具有不同定价结构和功能的模型时潜在的成本节省。
- **使用多个 Git Worktrees**：参与者建议利用多个 git worktrees 同时处理多个问题，并管理 Aider 实例以提高生产力。
   - 在不同的终端或分支上工作被视为抵消使用较慢模型所带来的等待时间的一种方法。
- **Prompt Caching 与 Token 管理**：讨论了 Aider 中 prompt caching 的有效性和实用性，重点在于它是否真的能节省成本还是会让过程变得复杂。
   - 讨论了将 Keepalive pings 作为一种在不产生过高成本的情况下维持缓存的手段，强调了平衡交互时机的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/aider-not-found.html">Aider not found</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/usage/images-urls.html#web-pages">Images &amp; web pages</a>: 将图像和网页添加到 aider 编码聊天中。</li><li><a href="https://aider.chat/docs/usage/modes.html#chat-modes">Chat modes</a>: 使用 chat、ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: 你可以通过命令行或 Python 为 aider 编写脚本。</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model">Chat modes</a>: 使用 chat、ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-includ">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://simonwillison.net/2024/Feb/21/gemini-pro-video/">The killer app of Gemini Pro 1.5 is video</a>: 上周 Google 推出了 Gemini Pro 1.5，这是对其 Gemini 系列 AI 模型的一次巨大升级。Gemini Pro 1.5 拥有 1,000,000 token 的上下文大小。这非常巨大——此前……</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>: 由 aider 用户制作的入门和教程视频。</li><li><a href="https://aider.chat/docs/config/options.html">Options reference</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/faq.html">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/more-info.html">More info</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: 让 aider 在处理你的代码时遵循你的编码规范。</li><li><a href="https://aider.chat/docs/config/options.html#--chat-language-chat_language">Options reference</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching">Prompt Caching (beta) - Anthropic</a>: 未找到描述</li><li><a href="https://aider.chat/docs/usage/caching.html">Prompt caching</a>: Aider 支持 Prompt 缓存，以节省成本并加快编码速度。</li><li><a href="https://youtu.be/W6Z0U11nnhA?si=VUCa3iHKWy3-L9vH">BEST Prompt Format: Markdown, XML, or Raw? CONFIRMED on Llama 3.1 &amp; Promptfoo</a>: 哪种 Prompt 格式最适合你的 AI Agent？是 Markdown、XML 还是 Raw Prompts？🚀 准备好释放 AI Agent 的真正潜力了吗？在本视频中，我...</li><li><a href="https://github.com/PierrunoYT/gemini-youtube-analyzer">GitHub - PierrunoYT/gemini-youtube-analyzer</a>: 通过在 GitHub 上创建账号来为 PierrunoYT/gemini-youtube-analyzer 的开发做出贡献。</li><li><a href="https://openrouter.ai/models/google/gemini-pro-1.5">Gemini Pro 1.5 - API, Providers, Stats</a>: Google 最新的多模态模型，支持文本或聊天 Prompt 中的图像和视频。针对以下语言任务进行了优化：- 代码生成 - 文本生成 - 文本编辑 - 问题解决...</li><li><a href="https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding">GitHub - yunlong10/Awesome-LLMs-for-Video-Understanding: 🔥🔥🔥Latest Papers, Codes and Datasets on Vid-LLMs.</a>: 🔥🔥🔥 关于 Vid-LLMs 的最新论文、代码和数据集。通过在 GitHub 上创建账号来为 yunlong10/Awesome-LLMs-for-Video-Understanding 的开发做出贡献。</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-include-the-git-history-in-the-context">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://github.com/paul-gauthier/aider/issues/1815#issuecomment-2381264243">Feature request - templates for aider · Issue #1815 · paul-gauthier/aider</a>: 问题：一些灵感可以借鉴 Simon Willison 的 https://github.com/simonw/llm，他的 LLM 工具允许创建插件 (#1814)。他还有一个 Prompt 模板系统，允许我们...</li><li><a href="https://youtu.be/pcC4Dr6Wj2Q?si=z5la0QllNsnLqY9F">Deno 2 is here… will it actually kill Node.js this time?</a>: 初探 Deno 2.0 - 一个具有原生 TypeScript 支持的 JavaScript 运行时，现在完全兼容 Node.js #javascript #tech #thecodere...</li><li><a href="https://console.groq.com/docs/rate-limits">GroqCloud</a>: 体验全球最快的推理
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1289650742566584401)** (16 条消息🔥): 

> - `NotebookLM 音频功能`
> - `Aider 更新`
> - `AI 播客摘要`
> - `内容创作自动化`
> - `招聘决策` 


- **NotebookLM 宣布自定义播客功能**：Google 的 [NotebookLM](https://notebooklm.google/) 现在提供了一项独特的音频功能，可以利用提供的内容生成自定义播客，由 AI 主持人对材料进行讨论。
   - 一个示例播客展示了其极具吸引力的形式，时长约十分钟，展示了主持人之间令人惊讶的真实对话。
- **Aider 工具的激动人心的更新**：最近的 YouTube 视频详细介绍了 Aider 的重大更新，其中一段名为 [“NEW Aider Architect & Editor Updates”](https://www.youtube.com/watch?v=8jD8dAXq8jE) 的视频展示了 AI coding agent 和 Beast Cursor 等功能。
   - 另一段视频讨论了 Aider Architect Mode 的增强功能，支持 **Gemini-002**，并强调了内容创作者制作这些视频的速度之快。
- **关于 AI 驱动的播客摘要的讨论**：有一场关于需要 AI 来收听并将无数新播客总结为列表文章的讨论，并建议这可能是下一个大项目。
   - 一位成员沉思着要创建一个名为 *“Today in Coding AI News”* 的播客，以进一步整合内容。
- **将发布说明自动化为音频**：有人提议利用 NotebookLM 的功能，将发布说明和源代码自动生成视频，从而可能简化内容生成。
   - 这个想法是发布一个名为 *ReleaseNotesLM* 的工具，以极小的代价将书面更新转换为音频格式。
- **基于视频质量的招聘决策**：在审阅内容后，一位成员表示他们已决定聘用之前讨论中被积极提及的一名个人。
   - 这一招聘决策反映了演讲者令人印象深刻的演讲技巧和内容深度的影响力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://simonwillison.net/2024/Sep/29/notebooklm-audio-overview/">NotebookLM 自动生成的播客效果惊人</a>：Audio Overview 是 Google NotebookLM 的一个有趣的新功能，目前备受关注。它会针对你提供的内容生成一次性的自定义播客，其中……</li><li><a href="https://www.youtube.com/watch?v=8jD8dAXq8jE">NEW Aider Architect &amp; Editor Updates Are INSANE!🤖(Beast Cursor?!?) Best AI Coding Agent?! OpenAI o1</a>：NEW Aider Architect &amp; Editor Updates Are INSANE!🤖(Beast Cursor?!?) Best AI Coding Agent?!? OpenAI o1 https://aider.chat/ https://github.com/paul-gauthier/aide...</li><li><a href="https://www.youtube.com/watch?v=OPXslklVBZc">Aider (Upgraded) : This Coding Agent just got BETTER with Architect Mode, Gemini-002 Support &amp; More!</a>：加入此频道以获得特权：https://www.youtube.com/@AICodeKing/join 在这段视频中，我将向你介绍 Aider 的新升级，它是……</li><li><a href="https://github.com/sigoden/aichat/wiki/RAG-Guide">RAG 指南</a>：集 Chat-REPL、Shell Assistant、RAG、AI tools &amp; agents 于一体的 AI CLI 工具，支持访问 OpenAI、Claude、Gemini、Ollama、Groq 等。 - sigoden/aichat
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1289317372829761537)** (464 条消息🔥🔥🔥): 

> - `AI 模型合并`
> - `AI 中的文本相似度`
> - `Stable Diffusion 性能`
> - `视频模型开发`
> - `Hugging Face 社区项目`

- **探索 AI 模型融合技术**：用户讨论了合并 AI 模型的不同方法，包括 PEFT merge 和 DARE 方法，强调了它们在增强模型性能方面的有效性。
   - 对话强调了从零开始训练 LLM 的挑战，以及现有模型在针对特定任务进行 fine-tuning 方面的实用性。
- **文本相似度在 AI 中的重要性**：参与者讨论了 AI 模型如何识别文本相似度，通过“I have a car”和“I own a car”等示例展示了需要数据集来教授这些细微差别。
   - 理解文本相似度对于提高 AI 交互质量至关重要，并且需要全面的数据集进行有效训练。
- **关于 Stable Diffusion 及其运行环境的讨论**：成员们比较了在 Windows 与 WSL 上运行 Stable Diffusion 的优势，并指出了 GPU 驱动程序对性能的影响。
   - 该话题强调了在资源密集型 AI 应用背景下对操作系统的偏好。
- **视频模型开发的新兴趋势**：用户对正在开发的新视频模型感到兴奋，分享了诸如 'S3Diff' 等创新项目的链接以及现有模型的更新。
   - 参与者对视频处理能力的进步以及即将推出的模型的潜力表达了热情。
- **关于 AI 模型性能的担忧**：用户分享了对 ChatGPT O1 等模型性能较早期版本有所下降的沮丧情绪，理由是推理和简洁性方面的问题。
   - 讨论反映了对模型更新以及审查或更改对 AI 可用性影响的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sambanova.ai>">未找到标题</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1ekNDPjC3CKWWd3jd2_V9QGTJSbvHKIZ2">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/time-series-transformers.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://tenor.com/view/jizz-adult-swim-john-reilly-blink-gif-14841420">Jizz Adult Swim GIF - Jizz Adult Swim John Reilly - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces/jasperai/Flux.1-dev-Controlnet-Upscaler">Flux.1-dev Upscaler - jasperai 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/KingNish/Qwen2.5-0.5b-Test-ft">KingNish/Qwen2.5-0.5b-Test-ft · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras#addweightedadapter">Merge LoRAs</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/time-">Google Colab</a>: 未找到描述</li><li><a href="https://console.groq.com/">GroqCloud</a>: 体验全球最快的推理速度</li><li><a href="https://x.com/NousResearch/status/1840505804673225031">来自 Nous Research (@NousResearch) 的推文</a>: 开源精神永存 #SB1047 已被否决</li><li><a href="https://cloud.google.com/translate/docs/reference/rest/?apix=true">未找到标题</a>: 未找到描述</li><li><a href="https://pypi.org/project/starlette-session-middleware/">starlette-session-middleware</a>: 无</li><li><a href="https://huggingface.co/spaces?search=Video%20editor">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/lmsys/chatbot_arena_conversations">lmsys/chatbot_arena_conversations · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://hf.co/papers/2311.03099">论文页面 - Language Models are Super Mario: Absorbing Abilities from Homologous
  Models as a Free Lunch</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k?row=0">HuggingFaceTB/everyday-conversations-llama3.1-2k · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/ArcticHare105/S3Diff">GitHub - ArcticHare105/S3Diff: S3Diff 的官方实现</a>: S3Diff 的官方实现。通过在 GitHub 上创建账号来为 ArcticHare105/S3Diff 的开发做出贡献。</li><li><a href="https://github.com/xtekky/gpt4free">GitHub - xtekky/gpt4free: 官方 gpt4free 仓库 | 各种强大语言模型的集合</a>: 官方 gpt4free 仓库 | 各种强大语言模型的集合 - xtekky/gpt4free</li><li><a href="https://huggingface.co/datasets/Langame/conversation-starters">Langame/conversation-starters · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/peft/developer_guides/model_merging">Model merging</a>: 未找到描述</li><li><a href="https://github.com/interneuron-ai/project-barbarossa">GitHub - interneuron-ai/project-barbarossa</a>: 通过在 GitHub 上创建账号来为 interneuron-ai/project-barbarossa 的开发做出贡献。</li><li><a href="https://huggingface.co/interneuronai/az-llama2">interneuronai/az-llama2 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1289314495260524627)** (14 messages🔥): 

> - `CUDA 实验`
> - `Gradio 带来的挫败感`
> - `模型 Policy Loss`
> - `界面设计问题` 


- **CUDA 实验带来新见解**：一位成员分享了他们在 **CUDA** 和 **7b FP8** 方面的工作进展，并提到一个拼写错误，该错误指向带有 **fp32 master weights** 的 **bfloat16**。
   - 他们反思了过去两天的学习过程，表示在技术上取得了显著成长。
- **Gradio 让用户感到失望**：成员们对 **Gradio** 表达了强烈不满，一位成员激动地称其为“工业垃圾”且浪费时间。
   - 他们转达了对其设计的沮丧，认为它将复杂的项目变成了“乱成一团的意大利面条式代码 (spaghetti code)”，并存在 UI 响应问题。
- **鼓励寻求 Gradio 支持**：针对对 Gradio 的不满，成员们鼓励在专门的频道中寻求有关 **Gradio** 功能问题的支持。
   - 一位成员以支持的口吻提供帮助，展示了社区导向的解决问题方式。
- **关于模型性能的见解**：社区讨论强调了一位成员对模型 **policy loss** 的满意度，指出其 loss 看起来“不错”。
   - 在关于持续技术挑战的广泛对话中，这一反馈被视为积极的进展。
- **探索 Gradio 的替代方案**：一位成员表示打算尝试 **NiceGUI** 作为 Gradio 的替代品，理由是后者存在重大的设计缺陷。
   - 他们表达了失望，但对像 **Accelerate** 这样他们喜欢的 **Hugging Face** 项目仍保持热情。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1289373629817032724)** (9 messages🔥): 

> - `医疗 AI 论文亮点`
> - `HuggingFace 模型流行度指标`
> - `投影映射技术 (Projection Mapping)`
> - `Phi 模型的使用体验`
> - `视频映射技术` 


- **医疗 AI 上周亮点**：最近的一篇帖子强调了 2024 年 9 月 21 日至 27 日当周的**医疗 AI 顶级研究论文和模型**，其中包括《A Preliminary Study of o1 in Medicine》等重要研究。
   - 社区成员建议通过将这些内容拆分为关注最酷论文的独立博客文章来提高曝光度。
- **HuggingFace 模型流行度指标抓取**：一个 Reddit 帖子讨论了一种量化 **HuggingFace 上最活跃点赞模型**的指标，该指标考虑了模型在平台上的发布时长，以避免偏向旧模型或新模型。
   - 一位用户提出了一个 Pull Request 来改进 OpenLLM 排行榜的点赞数更新，并提到这与 HuggingFace 的 Trending（趋势）板块的关系。
- **探索投影映射技术**：一篇关于投影映射的文章描述了这种艺术视频技术如何将表面转化为动态展示，创造沉浸式体验。
   - 文章讨论了其对企业的益处，并深入探讨了视频映射如何增强创意和参与度。
- **Phi 模型的使用困扰**：一位用户表达了使用 **Phi 3** 时的挫败感，指出他们的测试进展并不顺利，并质疑 Phi 2 的对比性能。
   - 这场持续的讨论反映了社区对不同版本 Phi 模型的有效性和易用性的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/192ly8c/which_models_are_the_most_actively_liked_on/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.mediacraft.video/posts/projection-mapping/">Projection Mapping - Artistic Video Content &amp; Visual Illusion</a>：欢迎来到投影映射的迷人世界——一种赋予艺术和视觉效果生命的尖端技术。了解投影映射的工作原理、它对企业的益处，并探索案例...</li><li><a href="https://x.com/OpenlifesciAI/status/1840020394880667937">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：医疗 AI 上周回顾：顶级研究论文/模型 🏅（2024 年 9 月 21 日 - 9 月 27 日）🏅 本周医疗 AI 论文《A Preliminary Study of o1 in Medicine: Are We Closer to an AI Doctor?》作者...</li><li><a href="https://huggingface.co/blog/aaditya/medicalai-weekly-papers-2127">医疗 AI 上周回顾：顶级研究论文/模型 🏅（2024 年 9 月 21 日 - 9 月 27 日）</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1289323201616281670)** (29 条消息🔥): 

> - `Flux-Schnell Demo`
> - `Qwen 2.5 Fine-tuning`
> - `Instrumentum AI Summarizer`
> - `Deepseek-Chat CoT Mode`
> - `MusicGen Continuations App` 


- **用于 Regional Prompt Attention 的 Flux-Schnell Demo**：已开发一个针对 **Flux-Schnell** 的 Demo，专注于 **Regional Prompt Attention**，并计划稍后添加源代码和 **ComfyUI** 节点。
   - 人们对这些增强功能将如何进一步提升用户体验充满期待。
- **微调 Qwen 2.5 模型**：一位用户分享了使用 **Magpie 300k Dataset** 微调 **Qwen 2.5 0.5b** 的经验，其回答质量可与 **Llama 3.2 1b** 等更大模型相媲美。
   - 该用户指出存在一些 **inconsistencies**（不一致性），但正在努力解决这些问题，并欢迎对其正在进行的工作提供反馈。
- **介绍 Instrumentum AI Summarizer**：**Instrumentum** AI 摘要生成器没有长度限制，旨在利用先进的 **LLM** 进行快速文档摘要。
   - 主要特点包括文档上传的完全安全性以及极具竞争力的价格，旨在提高生产力。
- **带有 Chain of Thought 可视化的 Deepseek-Chat**：**Deepseek-Chat** 模式引入了可选的 **Chain of Thought** 可视化，通过分步可视化实现透明推理。
   - 这一创新旨在通过基于 **Streamlit** 的 UI 增强用户对模型推理过程的理解。
- **用于 MusicGen Continuations 的 iOS App**：一款专注于使用 **beatboxes** 作为输入音频进行 **MusicGen continuations** 的 iOS App 正在开发中，即将发布到 **App Store**。
   - 该 App 具有降噪功能，旨在捕捉鼓点输入时提供更好的输出效果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.g-diffuser.com/dualdiffusion/">DualDiffusion Demo Audio</a>: DualDiffusion 演示音频</li><li><a href="https://forbo7.github.io/forblog/posts/21_reflecting_on_my_internships.html">What I Learned during my Second and Third Internships – ForBo7 // Salman Naqvi</a>: 在实践中学习</li><li><a href="https://x.com/thepatch_kev/status/1840536425776763020">Tweet from thecollabagepatch (@thepatch_kev)</a>: 第 4 天，用于 MusicGen continuations 的 iOS App 落地页，输入音频降噪以及一个效果尚可的 'tame the gary' 开关，专注于鼓点并尝试更好地融合输入...</li><li><a href="https://huggingface.co/spaces/qamarsidd/SentimentReveal">SentimentReveal - a Hugging Face Space by qamarsidd</a>: 未找到描述</li><li><a href="https://chromewebstore.google.com/detail/arxiv-insight-paper-summa/iciiagolkeidemjnbobbkcjfndkabicf">Chrome Web Store</a>: 为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://github.com/vietanhdev/llama-assistant">GitHub - vietanhdev/llama-assistant: AI-powered assistant to help you with your daily tasks, powered by Llama 3.2. It can recognize your voice, process natural language, and perform various actions based on your commands: summarizing text, rephasing sentences, answering questions, writing emails, and more.</a>: 由 Llama 3.2 驱动的 AI 助手，可帮助您处理日常任务。它可以识别您的语音、处理自然语言并根据您的指令执行各种操作：总结文本、重组句子、回答问题、撰写电子邮件等。</li><li><a href="https://instrumentum.ai/en">Welcome | Instrumentum</a>: 未找到描述</li><li><a href="https://github.com/U-C4N/Deepseek-CoT/">GitHub - U-C4N/Deepseek-CoT: Deepseek-CoT</a>: Deepseek-CoT。通过创建账号为 U-C4N/Deepseek-CoT 的开发做出贡献。</li><li><a href="https://docs.google.com/spreadsheets/d/1DlBT1pF8-zMECntRWXFsL46gZyvNp1BJlJ6LXGze4dA/edit?gid=0#gid=0">discord AI sphere - share  with whoever!</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1289619330899640437)** (5 messages): 

> - `OmDet-Turbo model`
> - `Keypoint Detection Task`
> - `SuperPoint Model`
> - `Fine-tuning TroCR Models`
> - `Upcoming Models for Keypoint Detection` 


- **OmDet-Turbo 模型发布**：团队宣布增加对 **OmDet-Turbo** 模型的支持，通过 [RT-DETR](https://www.linkedin.com/posts/yoni-gozlan_ai-artificialintelligence-objectdetection-ugcPost-7244768044533657603-FDOT?utm_source=share&utm_medium=member_desktop) 增强了受 Grounding DINO 和 OWLv2 启发的实时 zero-shot 目标检测能力。
   - 这一重大更新旨在提升 AI 在各种目标检测任务中的表现。
- **关键点检测任务页面发布**：推出了全新的 **keypoint-detection** 任务页面，现已支持 **SuperPoint**，这对于特征点检测和描述至关重要。详细信息可以在其文档 [此处](https://huggingface.co/docs/transformers/v4.45.1/en/model_doc/superpoint) 找到。
   - SuperPoint 展示了一个自监督训练框架，适用于单应性估计（homography estimation）和图像匹配。
- **社区渴望更多模型**：社区对关键点检测的兴趣日益增长，用户对未来集成 **LoFTR**、**LightGlue** 和 **OmniGlue** 等模型表示期待。
   - 这种期待凸显了社区对 Computer Vision 领域进展的参与度和期望。
- **微调 TroCR 模型的讨论**：一位用户提出了关于是否应该使用 '**trocr-large-stage1**' (base) 或 '**trocr-large-handwriting**' (已在 IAM 数据集上微调) 来微调其数据集的问题。
   - *他们询问微调一个已经微调过的模型是否会产生更好的性能。*



**提及的链接**：<a href="https://huggingface.co/docs/transformers/v4.45.1/en/model_doc/superpoint">SuperPoint</a>：未找到描述

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1290122188661329930)** (2 messages): 

> - `Hallucination Detection Model`
> - `Fine-tuning BERT on Yelp Dataset` 


- **幻觉检测模型发布**：发布了一个新的**幻觉检测模型**，从 **Phi-3** 升级到了 **Phi-3.5 base**，专注于评估语言模型输出的幻觉（hallucinations）。
   - 关键性能指标包括 **Precision: 0.77**、**Recall: 0.91** 和 **F1 Score: 0.83**，总体准确率达到 **82%**；[在此查看模型卡片](https://huggingface.co/grounded-ai/phi3.5-hallucination-judge)。
- **寻求微调 BERT 的帮助**：一名成员正在寻找在具有五个类别的 **Yelp 评论数据集**上微调 **BERT** 的资源，并对准确率仅在 **60% 左右**表示担忧。
   - 他们特别请求提供当前 SOTA 模型列表以及在 Yelp 数据集上的性能指标，并指出 Paperswithcode 网站上缺乏近期更新。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1289619614942232638)** (6 messages): 

> - `GitHub API usage`
> - `Stack Overflow for Developers`
> - `Increased Context LLaMA Model Conversion`
> - `llama.cpp compatibility` 


- **GitHub API 查询引发离题讨论**：一名成员询问如何使用 GitHub API 查找 rebase 后的提交，但另一名成员指出该频道专注于 Diffusion 模型。
   - 尽管话题离题，但有人建议该问题可以通过 [Stack Overflow](https://stackoverflow.com/) 或 GitHub Copilot 解决。
- **Stack Overflow 对开发者仍然至关重要**：一名成员强调，每个开发者都会在浏览器标签页中保留 [Stack Overflow](https://stackoverflow.com/) 以获取解决方案和知识共享。
   - 他们提到 Stack Overflow 现在正为团队提供一套 GenAI 工具，以改善员工之间的知识连接。
- **LLaMA 模型转换遇到困难**：一位用户分享了他们在将 **LLaMA-2-7B-32K** 模型转换为 GGUF 格式时遇到的困难，并寻求关于 llama.cpp 兼容性的帮助。
   - 他们提供了所遇错误的详细 traceback，强调了在词汇表设置阶段出现的 `IndexError`。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://stackoverflow.com/">Stack Overflow - 开发者的学习、分享与职业构建平台</a>：Stack Overflow | 全球最大的开发者在线社区</li><li><a href="https://huggingface.co/togethercomputer/LLaMA-2-7B-32K">togethercomputer/LLaMA-2-7B-32K · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1289301764708761685)** (363 messages🔥🔥):

> - `LM Studio 中的模型下载问题`
> - `在 LM Studio 中使用 vision-enabled models`
> - `LM Studio 的功能需求`
> - `对模型性能和声明的担忧`
> - `关于 query queueing 和 caching 的讨论` 


- **下载和 sideloading 模型的挑战**：用户讨论了在 LM Studio 中下载模型的问题，特别是在使用 VPN 时，导致一些人选择 sideloading 模型。
   - 平台的局限性得到了确认，特别是关于支持的模型格式，如 safetensors 和 GGUF。
- **LM Studio 中的 vision-enabled models**：已澄清由于与 llama.cpp 的兼容性问题，LM Studio 目前不支持 llama-3.2-11B vision models。
   - 参与者对 multimodal models 的更广泛可用性及其在平台内的功能提出了疑问。
- **功能需求和未来计划**：用户对 query queueing 和 caching of edits 等功能表示感兴趣，一些人在 feature tracker 中找到了现有的请求。
   - 目前没有发布针对即将推出的功能的 roadmap，使得 3D generation 等话题处于不确定状态。
- **对模型性能和可信度的担忧**：社区讨论了 LiquidAI 和 Replete 等新模型，并将其性能声明与 Qwen-2.5 等成熟选项进行了权衡。
   - 辩论集中在这些模型的可靠性和测试可访问性上，一些人对其营销炒作表示怀疑。
- **用户关于 LM Studio 加载时间的咨询**：一位用户报告称，即使模型已完全加载到 VRAM 中，仍经历了明显的加载时间，导致 evaluation 之前出现延迟。
   - 该问题引发了关于在应用程序中观察到的初始加载时间背后潜在原因的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=41323042">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/SixOpen/Florence-2-large-ft">Florence 2 Large Ft - SixOpen 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct">allenai/OLMoE-1B-7B-0924-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/training">微调预训练模型</a>: 未找到描述</li><li><a href="https://www.liquid.ai/liquid-foundation-models#join-us-as-an-early-adopter-of-LFMs)">Liquid Foundation Models：我们的首个生成式 AI 模型系列</a>: 宣布推出首个 Liquid Foundation Models (LFMs) 系列 —— 新一代生成式 AI 模型，在各种规模下均实现了最先进的性能，同时保持了更小的内存占用...</li><li><a href="https://huggingface.co/collections/Replete-AI/replete-llm-v25-66f987583df3ae3b18cf3c84">Replete-LLM-V2.5 - Replete-AI 集合</a>: 未找到描述</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio 社区)</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/llama31#inference-memory-requirements">Llama 3.1 - 具备多语言能力和长上下文的 405B, 70B &amp; 8B 模型</a>: 未找到描述</li><li><a href="https://huggingface.co/mylesgoose/Llama-3.2-11B-Vision-Instruct">mylesgoose/Llama-3.2-11B-Vision-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.liquid.ai/liquid-foundation-models#join-us-as-an-early-adopter-of-LFM">Liquid Foundation Models：我们的首个生成式 AI 模型系列</a>: 宣布推出首个 Liquid Foundation Models (LFMs) 系列 —— 新一代生成式 AI 模型，在各种规模下均实现了最先进的性能，同时保持了更小的内存占用...</li><li><a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large · Hugging Face</a>: 未找到描述</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio Beta 版本发布</a>: LM Studio Beta 版本发布</li><li><a href="https://lmstudio.ai/docs/basics/chat#faq">管理对话 - 在本地运行 LLMs | LM Studio 文档</a>: 使用 LLMs 管理对话线程</li><li><a href="https://lmstudio.ai/model/llama-3.2-">未找到模型</a>: 未找到描述</li><li><a href="https://github.com/YorkieDev/lmstudioservercodeexamples">GitHub - YorkieDev/lmstudioservercodeexamples: 此 Readme 包含来自 LM Studio v0.2.31 的服务器代码示例</a>: 此 Readme 包含来自 LM Studio v0.2.31 的服务器代码示例 - YorkieDev/lmstudioservercodeexamples</li><li><a href="https://lmstudio.ai/model/stable-code-instruct-3b.">未找到模型</a>: 未找到描述</li><li><a href="https://lmstudio.ai/model/llama-3.2-1b-instruct">Llama 3.2 1B</a>: llama • Meta • 1B</li><li><a href="https://github.com/openai/openai-python?tab=readme-ov-file#vision">GitHub - openai/openai-python: OpenAI API 的官方 Python 库</a>: OpenAI API 的官方 Python 库。通过在 GitHub 上创建账号为 openai/openai-python 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fqk9ky/i_trained_mistral_on_the_us_armys_field_manuals/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/9a913110cf471a8287ac06c43cbe307d3cf6df99">llama : 添加对 Chameleon 的支持 (#8543) · ggerganov/llama.cpp@9a91311</a>: * 将 chameleon hf 转换为 gguf
 
 * 添加 chameleon 分词器测试
 
 * 修复 lint
 
 * 实现 chameleon 图
 
 * 添加 swin norm 参数
 
 * 将 qk norm 权重和偏置恢复为原始格式
 
 ...</li><li><a href="https://huggingface.co/mistralai/Pixtral-12B-2409#usage-examples">mistralai/Pixtral-12B-2409 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/meta-llama/llama-models/tree/main">GitHub - meta-llama/llama-models: 旨在用于 Llama 模型的实用工具。</a>: 旨在用于 Llama 模型的实用工具。通过在 GitHub 上创建账号为 meta-llama/llama-models 的开发做出贡献。
</li>
</ul>

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1289306614511308832)** (138 条消息🔥🔥): 

> - `NVIDIA Jetson AGX Thor`
> - `3090 vs 3090 Ti vs P40 对比`
> - `GPU 市场定价`
> - `AI 模型托管与租赁`
> - `Linux 上的 NVIDIA 驱动问题` 


- **NVIDIA Jetson AGX Thor 拥有 128GB VRAM**：NVIDIA Jetson AGX Thor 将于 2025 年配备 **128GB VRAM**，这引发了成员们关于潜在升级的讨论。
   - 这一消息激发了人们的兴趣，即随着市场的发展，现有的 GPU（如 **3090** 或 **P40**）是否仍具可行性。
- **比较 GPU 性能：3090 vs 3090 Ti vs P40**：成员们讨论了 **3090**、**3090 Ti** 和 **P40** 之间的性能差异，VRAM 和价格因素影响了决策。
   - *有人指出 P40 的运行速度大约只有 3090 的一半*，而 GPU 的成本却在出人意料地持续上涨。
- **GPU 市场定价动态**：大家一致认为，由于黄牛加价以及近期 AI 工作负载的需求，目前 GPU 价格处于高位。
   - **A6000** 被讨论为投资高 VRAM 的潜在替代方案，尽管许多成员更倾向于选择更便宜的方案，例如使用多张 **3090**。
- **为 AI 工作负载租赁 GPU**：推荐使用 **Runpod** 和 **Vast** 租赁 GPU，成员们发现租赁比直接购买高价显卡更经济。
   - 一些成员就租赁成本回收的可行性进行了争论，尤其是在对高性能 GPU 需求激增的情况下。
- **Linux 上 NVIDIA 驱动的挑战**：讨论强调了 **NVIDIA** 的 Linux 驱动非常棘手，特别是在 **VRAM 卸载 (offloading)** 方面，而 AMD 显卡在这方面处理得更顺畅。
   - 社区对配置 NVIDIA 的 CUDA 和其他驱动程序表示沮丧，并强调在可行的情况下更倾向于选择 **AMD**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/mark-cuban-shark-tank-notes-taking-notes-remember-gif-15073512">Mark Cuban Shark Tank GIF - Mark Cuban Shark Tank Notes - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fqsafn/nvidia_jetson_agx_thor_will_have_128gb_of_vram_in/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://tenor.com/view/paulwnos-gif-26909845">Paulwnos GIF - Paulwnos - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/you-dont-turn-your-back-on-family-you-cant-walk-away-from-family-you-cant-leave-family-behind-you-cant-ignore-family-you-cant-disregard-family-gif-16058425">You Dont Turn Your Back On Family You Cant Walk Away From Family GIF - You Dont Turn Your Back On Family You Cant Walk Away From Family You Cant Leave Family Behind - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/geohot/cuda_ioctl_sniffer">GitHub - geohot/cuda_ioctl_sniffer: Sniff CUDA ioctls</a>：嗅探 CUDA ioctls。通过在 GitHub 上创建账号为 geohot/cuda_ioctl_sniffer 的开发做出贡献。</li><li><a href="https://www.ebay.co.uk/itm/285837751445?">Dell AMD Instinct MI100 32GB Graphics Accelerator | 50NN0  | eBay</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fqwrvg/64gb_vram_dual_mi100_server/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1289301614099828758)** (30 条消息🔥): 

> - `Cerebras 芯片优化`
> - `服务器垃圾信息管理`
> - `Triton 演讲幻灯片`
> - `GPU 性能指标`
> - `机器人开发挑战` 


- **关于 Cerebras 芯片代码优化的咨询**：一名成员询问是否有人正在为 **Cerebras 芯片** 优化代码，并寻求关于该芯片是否值得购买的建议。
   - 另一名成员表示可以帮忙联系对此有深入了解的人，显示出社区对该话题的兴趣。
- **解决服务器垃圾信息问题**：成员们讨论了 Discord 上日益增多的 **加密货币诈骗垃圾信息** 以及潜在的预防措施。
   - 建议包括实施更严格的验证流程和入群问题，以减少垃圾信息和恶意账号。
- **获取 Triton 演讲幻灯片**：一名成员寻找 **Triton 演讲** 的幻灯片，另一名成员将其引导至存放讲座资料的 [GitHub 仓库](https://github.com/gpu-mode/lectures)。
   - 这体现了社区在分享教育资源和确保参与者获取信息方面的努力。
- **GPU 性能指标评估**：讨论集中在观察到的性能指标上，特别是 **INT8** 与 **BF16** 在 **GPU** 上的表现对比，并指出了预期加速与实际加速的差异。
   - 成员们分享了在性能差异方面的经验，特别是关于计算中累加方法的问题。
- **机器人开发中的挑战**：一名成员发起了一场关于当前 **机器人开发挑战** 的头脑风暴，强调了计算能力和高昂劳动力成本等问题。
   - 他们鼓励共同思考哪些任务可以潜在地外包给成本更低的劳动力解决方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://drew.silcock.dev/blog/everything-you-need-to-know-about-python-3-13/">Everything you need to know about Python 3.13 – JIT and GIL went up the hill | drew's dev blog</a>：关于 Python 3.13 你需要知道的一切 —— JIT 和 GIL 的进展 | drew's dev blog：包含 Global Interpreter Lock 和 Just-in-Time 编译在内的最新 Python 版本详解。</li><li><a href="https://www.youtube.com/watch?v=BmdOt6A6tHM">llm.c's Origin and the Future of LLM Compilers - Andrej Karpathy at CUDA MODE</a>：今天 CUDA mode 黑客松的非正式记录。https://github.com/karpathy/llm.c</li><li><a href="https://github.com/gpu-mode/lectures">GitHub - gpu-mode/lectures: Material for gpu-mode lectures</a>：gpu-mode 讲座资料。可以通过在 GitHub 上创建账号来为 gpu-mode/lectures 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1289324416353960018)** (12 messages🔥): 

> - `Triton 库函数`
> - `Block 指针与 tmas`
> - `Triton 深度解析讲座`
> - `Metal MLIR dialect`
> - `Triton 中的设备编译` 


- **Triton 提供多种计算函数**：一位用户指出，可以使用 `tl.exp(tl.log(t)*x)` 计算指数，或者利用 `libdevice` 中的 `pow()` 或 `fast_powf()` [详情见此](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html)。
   - 另一位成员认为这些信息非常有用，表明社区对实际实现提供了强有力的支持。
- **关于 Block 指针和 tmas 的讨论**：一位用户提到 Block 指针不会转换为 **tmas**，并引发了关于此行为潜在细微差别的疑问。
   - 这种推测表明了关于 Triton 如何处理特定数据结构的更深层次技术讨论。
- **Triton 深度解析讲座亮点**：一位参与者对 Triton 的深度解析讲座表示感谢，指出该讲座吸引了超过 **100 名观众**，创下了目前为止第二大的直播观看人数记录。
   - 演讲者感谢了一位同事鼓励其进行演示，体现了良好的社区支持氛围。
- **探索 Metal MLIR Dialect**：一位成员分享了一个旨在成为 **Metal MLIR dialect** 的库链接，并重点介绍了 `CommandBuffer` 类，它类似于 warp。
   - 他们还引用了 Metal 中的共享内存概念，展示了社区对性能优化的兴趣。
- **Triton 如何决定设备编译**：有关于 Triton 如何决定为哪个设备进行编译的疑问，特别是当使用 @triton.jit 装饰的函数旨在用于 GPU 但在编译时失败时。
   - 另一位用户建议查看 `get_current_device()` 进行驱动检测作为潜在解决方案，这提供了有用的故障排除资源。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://fburl.com/4ml0y1p7">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/int-flashattention2024/int-flashattention">GitHub - INT-FlashAttention2024/INT-FlashAttention</a>: 通过在 GitHub 上创建账号来为 INT-FlashAttention2024/INT-FlashAttention 的开发做出贡献。</li><li><a href="https://github.com/kapilsh/lectures/blob/main/lecture_029/presentation.pdf">lectures/lecture_029/presentation.pdf at main · kapilsh/lectures</a>: cuda-mode 讲座材料。通过在 GitHub 上创建账号来为 kapilsh/lectures 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1289643628674945097)** (35 条消息🔥): 

> - `Torchscript 哈希表的批量更新选项`
> - `torch.int_mm() 在 CPU 上的问题`
> - `调试 AO 模型替换`
> - `FFCV 的图像加载替代方案`
> - `ZeRO-3 对单 GPU 推理的优势` 


- **Torchscript 哈希表缺少批量更新选项**：一位成员确认 Torchscript 哈希表没有“批量更新”选项，并引用了相关的 GitHub 接口 [此处](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/python/python_dict.h)。他们建议在 GPU 上使用 `cuco::dynamic_map` 进行批量插入，尽管这可能需要重大的代码重新设计。
   - 讨论强调，并行更新哈希表的情况相当少见。
- **torch._int_mm() 在 CPU 上返回错误结果**：一位成员报告说，在 CPU 上对带有 **int8** 权重的矩阵乘法执行 `torch._int_mm()` 会得到错误结果，而 CUDA 的输出是正确的。该问题已记录在 [此 GitHub ticket](https://github.com/pytorch/pytorch/issues/136746) 中，表明这是 AMD CPU 的问题。
   - 这个问题引起了社区对可能的变通方法和修复方案的讨论。
- **调试 AO 中的模型权重替换**：一位成员询问如何验证 AO 是否正确替换了权重和激活函数，并被建议打印模型并检查 [此 pull request](https://github.com/pytorch/ao/pull/782) 中的日志。社区成员强调检查模型内部是验证实现更改的一种方式。
   - 另一位关注者建议可能需要使用 DeepSpeed，但对于单 GPU 使用可能并非必要。
- **探索 FFCV 之外的图像加载替代方案**：一位成员询问了除了 FFCV 之外，PyTorch 工作流中更好的图像加载选项，并提到了其缓存和文件系统的优势。社区反馈强调了新的方法，包括使用流式数据集、WebDataset 以及利用 **torchvision** 变换来提高效率。
   - 然而，有人对 DALI 等库与 FFCV 相比的灵活性和开销表示担忧。
- **ZeRO-3 对单 GPU 推理的优势**：成员们讨论了使用 ZeRO-3 的好处，指出它不仅适用于分布式框架，对于单 GPU 设置也确实有用。提供了一个链接，详细介绍了 ZeRO-3 的关键特性，特别是它在有限资源下处理 **large models** 的效率。
   - 针对其价值以及 DeepSpeed 为没有大量 GPU 资源的用户提供的内核替换功能进行了澄清。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.deepspeed.ai/2021/03/07/zero3-offload.html">DeepSpeed ZeRO-3 Offload</a>: DeepSpeed 是一个深度学习优化库，使分布式训练变得简单、高效且有效。</li><li><a href="https://www.deepspeed.ai/2022/09/09/zero-inference.html#model-scaling-on-1-gpu">ZeRO-Inference: Democratizing massive model inference</a>: DeepSpeed 是一个深度学习优化库，使分布式训练变得简单、高效且有效。</li><li><a href="https://github.com/KellerJordan/cifar10-airbench">GitHub - KellerJordan/cifar10-airbench: 2.73 秒内在 CIFAR-10 上达到 94% 💨 27 秒内达到 96%</a>: 2.73 秒内在 CIFAR-10 上达到 94% 💨 27 秒内达到 96% - KellerJordan/cifar10-airbench</li><li><a href="https://github.com/pytorch/pytorch/issues/136746">AMD CPU 上的 torch._int_mm 精度问题 · Issue #136746 · pytorch/pytorch</a>: 🐛 错误描述：在 AMD CPU 上对 int8 权重执行矩阵乘法时，结果与在 CUDA 或 Intel CPU 上运行相同操作获得的结果不同……</li><li><a href="https://github.com/pytorch/ao/pull/782">jerryzh168 为量化线性模块添加更多信息并添加了一些日志 · Pull Request #782 · pytorch/ao</a>: 摘要：修复了 #771；测试计划：python test/dtypes/test_affine_quantized_tensor.py -k test_print_quantized_module；示例输出：Linear(in_features=128, out_features=256, weight=AffineQuantizedTens...</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/python/python_dict.h">pytorch/torch/csrc/jit/python/python_dict.h at main · pytorch/pytorch</a>: Python 中的 Tensor 和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch</li><li><a href="https://github.com/NVIDIA/cuCollections">GitHub - NVIDIA/cuCollections</a>: 通过创建账户为 NVIDIA/cuCollections 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1289643527742947350)** (1 条消息): 

> - `Triton Internals`
> - `Lecture Schedule`
> - `Quantized Training`
> - `Metal Kernels`
> - `GPU Optimization` 


- **Kapil Sharma 回归主讲 Triton Internals**: 我们很高兴欢迎客座讲师 **Kapil Sharma** 回归，在 `reading-group` 阶段频道深入探讨 **Triton internals**。
   - 该环节将在公告发布后约 **20 分钟** 开始。
- **讲座恢复，阵容强大**: 讲座已恢复，计划进行 **10** 场讲座，展示来自世界各地的极具影响力的 GPU 黑客。
   - 值得关注的课程包括 **Quantized Training** 和 **Metal Kernels**，由来自服务器的资深贡献者主讲。
- **系列讲座重点讲师**: 本次系列的主讲人包括负责 **SGLang** 的 **Yineng Zhang** 和负责 **CUTLASS** 以及 **Flash Attention 3** 的 **Jay Shah**。
   - 这些讲座有望为 GPU 编程的进展提供宝贵的见解。
- **GPU 优化的多样化主题**: 讲座将涵盖一系列主题，包括 **Low Bit Triton kernels**、**DietGPU** 以及 **SASS 入门**。
   - 该系列旨在吸引 GPU 技术领域的初学者和高级用户。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1289623024915714100)** (4 条消息): 

> - `AI Discord Servers`
> - `CuTe/Cutlass Layout Algebra`
> - `Next-Token Prediction` 


- **AI Discord 服务器排名**: 一名成员分享了一个 [电子表格](https://docs.google.com/spreadsheets/d/1DlBT1pF8-zMECntRWXFsL46gZyvNp1BJlJ6LXGze4dA/edit?gid=0#gid=0)，根据服务器类型和活跃度等不同标准对各种 AI 服务器进行了列举和排名。
   - **EleutherAI** 服务器获得了 **7.9** 分，表明其“非常活跃”并提供各种社区项目和工具。
- **请求添加缺失的 Discord 服务器**: 一名成员注意到之前分享的 AI 服务器列表中缺少 **Ultralytics** Discord。
   - 这凸显了为 AI 社区维护全面资源以促进联系的重要性。
- **CuTe/Cutlass Layout Algebra 演示片段**: 一名成员分享了他们制作的 [演示片段](https://x.com/KuterDinel/status/1840380207657533692) 链接，并考虑制作一个模仿 3blue1brown 风格的视频来解释 **CuTe/Cutlass layout algebra**。
   - 社区表现出极大的热情，对演示内容给予了积极的回应。
- **Next-token Prediction 的重要性**: 一名成员引用了一个链接，讨论对于某些 AI 应用来说 *next-token prediction is all you need*，强调了其重要性。
   - 这反映了人们对基础 AI 概念的简单性和有效性的广泛兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/KuterDinel/status/1840380207657533692">Kuter Dinel (@KuterDinel) 的推文</a>: 考虑制作一个 @3blue1brown 风格的视频来解释 CuTe/Cutlass layout algebra。让我知道你们对我制作的小演示片段的看法。</li><li><a href="https://docs.google.com/spreadsheets/d/1DlBT1pF8-zMECntRWXFsL46gZyvNp1BJlJ6LXGze4dA/edit?gid=0#gid=0">discord AI sphere - 欢迎分享！</a>: 未找到描述</li><li><a href="https://emu.baai.ac.cn/about">Emu3</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1289503431672856648)** (11 条消息🔥): 

> - `Model Parallelism 与 ZeRO/FSDP 的区别`
> - `理解 FSDP 机制`
> - `NLP 领域的开源项目`
> - `LLM 研究工作流简介`
> - `HuggingFace 工具与库` 


- **澄清 Model Parallelism vs ZeRO/FSDP**：一位成员寻求理解 PyTorch 中 **Model Parallelism** 与 **ZeRO/FSDP** 之间的区别，并询问 **ZeRO** 是否因其参数分布方式而可以被视为一种模型并行形式。
   - 另一位成员通过提到 **FSDP** 结合了分片（sharding）并需要理解其架构中的不同层级，提供了清晰的解释。
- **FSDP 机制详解**：**FSDP** 在 GPU 之间对模型层进行分片，在前向传播期间需要 all-gather，同时在本地维护每一层的分片，这与流水线并行（pipeline parallelism）有显著不同。
   - 讨论显示 **FSDP** 通过协调通信来提高效率，使其区别于每个设备处理不同层级的流水线方法。
- **适合初学者的开源项目**：一位成员询问了在 **NLP**、**LLM** 和 **reinforcement learning** 领域中，专注于易上手任务的初学者友好型开源项目。
   - 这一询问反映了初学者对 **CUDA/Triton** 及其在各领域应用的教育资源日益增长的兴趣。
- **LLM 研究工作流的起点**：一位成员表示需要指导，以从使用 **TensorFlow** 的 **CNNs** 转型到主要使用 **PyTorch** 探索 **LLM** 和 **Diffusion** 技术。
   - 他们寻求对 **HuggingFace Hub** 和 **Diffusers library** 等关键组件的清晰认识，以确定它们在研究功能中的集成方式。
- **解释 HuggingFace 库**：有人请求提供使用 **HuggingFace** 工具的工作流示例，表明需要学习数据集、预训练权重以及 **Transformers** 和 **Accelerate** 等相关库。
   - 对未知术语的澄清展示了一个更大的趋势：研究人员在扩展到新的 AI 方法论时需要支持。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/MachineLearning/comments/1bqsq3w/d_pytorch_fsdp_is_pipeline_parallelism_right/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html">Getting Started with Fully Sharded Data Parallel(FSDP) — PyTorch Tutorials 2.4.0+cu121 documentation</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1289768974489489491)** (3 条消息): 

> - `第 29 讲：Triton 内部机制`
> - `线下见面会演讲上传` 


- **Triton 内部机制讲座发布**：标题为 [Lecture 29: Triton Internals](https://youtu.be/njgow_zaJMw?feature=shared)、由主讲人 **Kapil Sharma** 带来的 YouTube 视频已分享。
   - 本讲座深入探讨了 Triton 的内部工作原理，并强调了其技术细节。
- **线下见面会演讲即将上线**：已确认 **IRL meetup talks** 将在几天内上传至 YouTube。
   - 这些即将发布的视频被描述为比往常**精良得多**，并对延迟表示歉意。



**提到的链接**：<a href="https://youtu.be/njgow_zaJMw?feature=shared)">Lecture 29: Triton Internals</a>：主讲人：Kapil Sharma

  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1289300873712762895)** (35 条消息🔥): 

> - `CPUOffloadOptimizer`
> - `FP8 和 INT8 量化`
> - `模型分析 (Model Profiling)`
> - `Hugging Face 集成`
> - `结合 Flux 微调的 SOAP 优化器` 


- **与专家一起对 CPUOffloadOptimizer 进行分析**：成员们讨论了对 **torchao CPUOffloadOptimizer** 的性能分析，其中一人寻求咨询贡献者以获取反馈。
   - *最好创建一个话题让其他人也加入*讨论，而不是通过私信。
- **FP8 和 INT8 加载的挑战**：有人提出了在 **FP8** 下加载 **lycoris 适配器**时出现 **显存不足 (OOM)** 的问题，但在 **INT8** 下似乎运行良好。
   - *FP8 的主要优势似乎在于计算加速*，这是讨论量化策略的成员们分享的观点。
- **动态量化与仅权重量化的区别解释**：一位成员解释说，**动态量化 (dynamic quantization)** 主要支持计算密集型 (compute-bound) 模型，而**仅权重量化 (weight-only quantization)** 则对内存带宽受限型 (memory-bound) 模型有益，这是从 [Cohere 的讲座](https://youtu.be/1u9xUK3G4VM?feature=shared)讨论中学到的。
   - 强调了 **FP8** 量化的复杂性，特别是在内存负载和计算收益之间的权衡。
- **解决评估脚本的问题**：讨论了评估脚本的问题，特别是使用 **pile_hackernews** 数据集导致配置错误以及需要进行版本检查的问题。
   - 成员们通常更倾向于使用 **wikitext** 进行评估，并指出了现有配置中的空白，建议进一步调查。
- **在主分支中引入 Int8 支持**：一位成员将用于全精度/混合精度训练的 **int8-torchao 支持**合并到了 **bghira/simpletuner** 的主分支中，理由是它在避免 **SOAP 优化器** 导致的 OOM 错误方面非常有用。
   - 得益于 **int8**，他们报告称在当前设置下不需要状态卸载 (state offload)，从而实现了更高效的 **Flux 微调**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/1u9xUK3G4VM?feature=shared)">第 7 讲 高级量化</a>：幻灯片：https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&amp;dl=0</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload">ao/torchao/prototype/low_bit_optim at main · pytorch/ao</a>：PyTorch 原生量化和稀疏性，用于训练和推理 - pytorch/ao
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/)** (1 条消息): 

glaxus_: 有人看过这个关于长上下文推理的内容吗？ https://arxiv.org/pdf/2409.17264v1
  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1289599467552772200)** (83 messages🔥🔥): 

> - `GeForce RTX 5090`
> - `Power supply challenges`（电源供应挑战）
> - `Apple Watch and LLMs`
> - `California AI safety bill`（加州 AI 安全法案）
> - `Cooling solutions for high-end GPUs`（高端 GPU 散热解决方案）


- **GeForce RTX 5090 规格引发热议**：最新传闻的 **GeForce RTX 5090** 拥有 **600W TDP**、**512-bit GDDR7** 和 **32GB** 显存等强悍规格，让用户对其功耗和散热需求感到好奇。
   - “这到底该怎么散热？”以及“我以为我的 4070 在 200W 时发热已经很大了”等言论凸显了用户对管理此类功耗需求的担忧。
- **需要升级电源**：由于 RTX 5090 的功耗高达惊人的 **600W**，社区中的许多人都在质疑是否需要升级电源（PSU）以满足这些需求。
   - 正如一位用户所言，“大多数人现在都需要升级 PSU 了”，这表明了大家对日益增长的功耗需求的普遍担忧。
- **Apple Watch：下一个 AI 前沿？**：讨论围绕在 **Apple Watch** 上原生运行 **Llama 3.2 1B** 的可能性展开，用户正在考虑该设备的架构是否能够支持它。
   - 一位用户评论道，“如果 Apple Watch 的性能足以运行一个连贯的 LLM，那我们真的成功了”，展现了对便携式 AI 的向往。
- **加州 AI 安全法案被否决**：加州州长 **Gavin Newsom** 否决了 **SB 1047** AI 安全法案，理由是该法案可能会给 AI 公司带来不必要的负担，且范围可能过于广泛。
   - 他强调，该法案未能考虑到 AI 系统部署的具体语境，甚至会影响到一些基础功能。
- **讨论高端 GPU 的散热解决方案**：考虑到 RTX 5090 在满载下可能面临的散热压力，多位用户对双风扇设计的效果表示怀疑。
   - 提出的建议包括使用水冷解决方案，一位用户声称，“高级用户可能需要购买像水冷混合散热器之类的东西。”


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/kopite7kimi/status/1839343725727941060">来自 kopite7kimi (@kopite7kimi) 的推文</a>：GeForce RTX 5090 PG144/145-SKU30 GB202-300-A1 21760FP32 512-bit GDDR7 32G 600W</li><li><a href="https://www.theverge.com/2024/9/29/24232172/california-ai-safety-bill-1047-vetoed-gavin-newsom">加州州长否决重大 AI 安全法案</a>：加州 AI 安全法案宣告终结。</li><li><a href="https://tenor.com/view/4090-rtx-gif-1107376314130178802">4090 Rtx GIF - 4090 RTX - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

marcelo5444: 有人在米兰参加 ECCV 吗？
  

---


### **GPU MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1290294586882920541)** (1 messages): 

> - `HQQ model serialization`（HQQ 模型序列化）
> - `Transformers library` 


- **HQQ 模型序列化获得全面支持**：最近的 [pull request #33141](https://github.com/huggingface/transformers/pull/33141) 为 Transformers 库直接保存和加载 **HQQ 量化模型** 增加了全面支持。
   - 此前，**序列化**是在 **hqq-lib** 端使用 .pt 格式处理的，而此次更新旨在简化这一流程。
- **对之前 PR #32379 的后续跟进**：此 pull request 是对 **#32379** 的后续跟进，旨在增强库内的序列化能力。
   - 这反映了社区在改进模型处理方面的持续努力，并强调了开发中的协作。



**提到的链接**：<a href="https://github.com/huggingface/transformers/pull/33141/">Hqq serialization by mobicham · Pull Request #33141 · huggingface/transformers</a>：#32379 的后续跟进。此 PR 的目标是增加在 transformers 中直接保存/加载 HQQ 量化模型的全面支持。到目前为止，序列化是在 hqq-lib 端通过 .pt 格式完成的...

  

---

### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1289305367611969609)** (23 条消息🔥): 

> - `repkv_backward_kernel2` 的改进
> - `FP8` 实现策略
> - `Llama3` 相关问题
> - 用于 `FP8` 的 `Pre-swizzled` 布局
> - 自定义 `matmul` 内核开发


- **repkv_backward_kernel2 展示了显著的改进**：关于 `repkv_backward_kernel2` 的最新 PR 已经提交。与 `repkv_backward_kernel1` 相比，它在减少线程使用的同时展现了更好的性能，并缩短了执行时间。
   - 详细信息可以在[这里](https://github.com/karpathy/llm.c/pull/771)找到，重点介绍了根据社区建议所做的增强。
- **探索 FP8 实现的新方法**：一位成员讨论了一种非侵入式的 `FP8` 方法，该方法在保持性能的同时，为大型矩阵集成了缩放因子。
   - 该实现利用组合方法进行高效缩放，如果集成，预计将优于现有方法。
- **调查 Llama3 的差异问题**：围绕尚未解决的 `Llama3` 差异（特别是关于 `repkv` 和 `rope` 功能）展开了讨论，成员们表示愿意协助排查。
   - 一位成员表示愿意进一步探索这些问题，并建议在此期间先审查 `repkv` 内核的 PR。
- **Pre-swizzled 布局在 FP8 应用中的潜力**：讨论强调了 `FP8` 采用 `Pre-swizzled` 布局的好处，这可以以额外的内存占用为代价换取性能提升。
   - 成员们指出，这种技术对于大型矩阵特别有用，允许在乘法过程中进行 warp 级的线性加载（linear loads）。
- **辩论自定义 matmul 内核的缩放方案**：一位成员概述了在自定义 `matmul` 内核的累加过程中管理缩放因子的方法，建议为中间结果使用临时寄存器。
   - 该方法涉及在缩放因子超过特定阈值时利用多个 `WGMMA` 操作来优化性能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/771.">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/771">Add `repkv_backward_kernel2` and `repkv_kernel2` by insop · Pull Request #771 · karpathy/llm.c</a>: 更改内容：添加 `repkv_backward_kernel2`，根据 @karpathy 的建议通过减少线程使用来改进 `repkv_backward_kernel1`。同时添加了与 `backward_kernel2` 类似的 `repkv_kernel2`。以下是测试输出...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1289312419910389771)** (207 messages🔥🔥): 

> - `面向社区的 MI300X 访问权限`
> - `AMD GPU 的性能问题`
> - `调优 MIOpen Kernel`
> - `AMD-Llama 模型训练`
> - `使用 Triton 实现 Flash Attention` 


- **MI300X 访问权限助力推广**: 来自 TensorWave 的 Darrick 表示有兴趣向社区提供 MI300X GPU，以增强采用率和教育，欢迎通过私信进行协调。
   - 来自 AMD 的 Anush 也提供了 MI300 访问的赞助，表明了参与社区的协作努力。
- **AMD GPU 的性能挑战**: 讨论揭示了 AMD GPU 面临的重大性能障碍，特别是跨节点扩展方面，重点在于 GFX1100 和 MI300 架构表现不佳。
   - 成员们指出，虽然 NVIDIA 的 GPU 通常表现更好，但提升 AMD GPU 性能（特别是在多节点设置中）的努力仍在进行中。
- **调优 MIOpen 以获得高效性能**: 对话强调了 MIOpen Kernel 漫长的调优时间，特别是在 ROCm 6.2 下，并呼吁在测试期间寻找绕过不必要调优的方法。
   - 讨论了将环境变量设置为 `MIOPEN_FIND_MODE=FAST` 作为一种权宜之计，以在牺牲极小性能的情况下缩短调优时间。
- **训练 AMD-Llama 模型**: Anthonix 报告称在 7900XTX 机器上训练 AMD-llama-135M 模型，达到了约 335k tokens/sec，略快于之前的 8xMI250x 结果。
   - 由于使用了 Multi-Head Attention (MHA) 而非 Gated Query Attention (GQA) 以及更长的上下文长度，该模型的实现面临挑战。
- **使用 Triton 实现 Flash Attention**: 成员们分享了在 MI300 上利用 Triton 实现 Flash Attention 的基准测试链接，并对测试期间性能缓慢表示担忧。
   - 提到了 Flash Attention 的 backward 函数的进展，但对其整体效率和可用性仍持怀疑态度。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://rust-lang.github.io/rust-project-goals/2024h2/Rust-for-SciComp.html">Expose experimental LLVM features for automatic differentiation and GPU offloading - Rust Project Goals</a>: 未找到描述</li><li><a href="https://huggingface.co/amd/AMD-Llama-135m">amd/AMD-Llama-135m · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/jzhang38/TinyLlama">GitHub - jzhang38/TinyLlama: The TinyLlama project is an open endeavor to pretrain a 1.1B Llama model on 3 trillion tokens.</a>: TinyLlama 项目是一个开源尝试，旨在 3 万亿 token 上预训练一个 1.1B 的 Llama 模型。 - jzhang38/TinyLlama</li><li><a href="https://rocmdocs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html#find-modes>">Using the find APIs and immediate mode &#8212; MIOpen 3.2.0 Documentation</a>: 未找到描述</li><li><a href="https://rocmdocs.amd.com/projects/MIOpen/en/latest/conceptual/perfdb.html#auto-tuning-kernels>">Using the performance database &#8212; MIOpen 3.2.0 Documentation</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1289327811559034923)** (1 messages): 

> - `多 GPU 使用`
> - `基于 Llama 的模型` 


- **多 GPU 设置变得简单**: 建议成员在多 GPU 设置中使用 **torchrun**，该命令在文件顶部已标出。
   - 默认方法是 **fsdp2**，但添加 `--ddp` 可以切换到使用 **DDP**。
- **Llama 模型已准备就绪**: 您可以通过在 `--model-id` 中指定 Hugging Face 上的任何 **基于 Llama 的模型** 来无缝使用它们，利用 **HF 的 LlamaForCausalLM**。
   - 默认模型选项主要用于 **测试目的**。


  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/)** (1 messages): 

marksaroufim: https://github.com/pytorch/torchtune/pull/1698

### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1289437739686563862)** (12 messages🔥): 

> - `LiteRT vs gpu.cpp`
> - `WebNN 对比`
> - `gpu.cpp 中的手动组网`
> - `Buffer Pass 读/写`
> - `WebGPU 资源` 


- **LiteRT 在模型运行时方面超越了 gpu.cpp**：LiteRT 旨在根据设备可用性利用 **GPU**、**CPU** 和 **NPU** 的组合，而 **gpu.cpp** 缺乏类似的能力。
   - 这表明为了获得最佳性能，首选 LiteRT 而非 gpu.cpp，因为后者需要更多的手动操作。
- **LiteRT 比 WebNN 更接近但功能更全**：LiteRT 被拿来与 **WebNN** 进行比较，它提供了从文件加载和设置模型的增强功能，而 WebNN 缺乏这一特性。
   - 这使得 LiteRT 成为那些需要模型加载和配置的用户的更全面的解决方案。
- **使用 gpu.cpp 需要手动组网**：使用 **gpu.cpp** 创建网络需要深入的理解和手动配置，以确保性能能与 LiteRT 媲美。
   - 这种复杂性可能会挑战那些对手动组网细节经验较少的开发者。
- **Buffer Pass 运行符合预期**：一位开发者确认，在 pass 1 中写入 **buffer A** 允许在 pass 2 中读取 pass 1 的结果，从而在多个层之间保持逻辑流。
   - 这确保了网络层可以按照设计有效地计算和传递数据。
- **可用的 WebGPU 资源有限**：开发者通常将 **specification**（规范）和 Google 的 **'what's new in WebGPU'** 博客文章作为主要资源。
   - 然而，关于 WebGPU 的文献匮乏，在寻找直接信息方面带来了挑战。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1289465789841149993)** (6 messages): 

> - `Gemma2 收敛测试失败`
> - `LLama3.2-Vision 补丁问题`
> - `2024 Q4 路线图追踪` 


- **Gemma2 测试在 main 分支上失败**：如 [此 GitHub action](https://github.com/linkedin/Liger-Kernel/actions/runs/11108231200/job/30860687961?pr=284) 所示，**Gemma2 收敛测试**目前在 main 分支上失败。此外，自 **HF 发布 4.45.0 版本**以来，**qwen2-vl 多模态测试**也出现了问题，但修复方案将在即将发布的 PR 中提供。
- **LLama3.2-Vision 需要预训练的 tokenizer**：一位成员已经准备好了 **llama3.2-vision 补丁**，但在多模态测试期间面临需要预训练 tokenizer 的问题。本地运行的测试通过了，但 GitHub CI/CD 需要一个承认 llama 许可的 **HF hub token**。
- **启动 2024 Q4 路线图追踪**：已创建 2024 Q4 的 **roadmap tracker**，以更有效地管理日益增长的请求。这个置顶的 issue 旨在跟踪 issue 和 PR，并为各项任务分配了特定的维护者，详见 [此 GitHub issue](https://github.com/linkedin/Liger-Kernel/issues/285)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/linkedin/Liger-Kernel/actions/runs/11108231200/job/30860687961?pr=284">poke tests · linkedin/Liger-Kernel@81a75e7</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/285">2024 Q4 Roadmap · Issue #285 · linkedin/Liger-Kernel</a>：随着社区的成长，跟踪 issue 和 PR 变得越来越具有挑战性。这个置顶的 issue 将作为管理 2024 Q4（约至 2024/12）进度的中心场所。在这里我们...
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1289320596374818907)** (16 条消息🔥): 

> - `Metal Shading Language`
> - `M2 与 M3 设备性能`
> - `Triton 的 Metal 后端`
> - `构建设备端 Agent`
> - `Metal 的资源共享` 


- **Metal Shading Language 规范至关重要**：对于任何从事 Metal 开发的人员，强烈推荐将 **Metal Shading Language Specification** 作为基础资源：[查看规范](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)。
   - 正如一位成员所言，“你最好的选择就是 Metal shading language 规范”，这反映了它的重要性。
- **M3 性能见解**：用户在 **M3** 设备上遇到了挑战，表示某些功能和资源仍处于追赶阶段。一位用户提到，在尝试使用它训练设备端专家模型时，由于难以正确处理查询，其热情受到了打击。
- **为 Triton 创建 Metal 后端**：一名成员询问了建立 **Triton 的 Metal 后端**的可行性，并概述了从 Triton IR 到 Metal Shader 的潜在转换过程。
   - 他们还列出了有用的资源，包括 **LLVM IR 到 Metal shader 转换器**：[概览](https://developer.apple.com/metal/shader-converter/)，并重点介绍了 GitHub 上的 **MLIR Metal dialect**。
- **理解 Metal 中的浮点速率**：据报告，使用 Metal 时 **F16** 是全速率的，而 BF16 仅在某些设备（特别是 **M1**）上通过模拟实现。一位用户分享说，这支持在不同的 Apple 硬件上执行高效的计算任务。
- **关于设备端 Agent 的通用建议**：一位用户表达了在构建**设备端 Agent** 时的挫败感，原因是缺乏关于其 **MacBook Pro M3** 的信息和支持。对话中还有一名成员通过询问所使用的具体设备来提供帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developer.apple.com/metal/shader-converter/">Metal shader converter - Metal - Apple Developer</a>：Metal shader 转换器将 LLVM IR 字节码中的 shader 中间表示转换为适合加载到 Metal 中的字节码。它以库和独立可执行文件的形式提供。 </li><li><a href="https://github.com/NicolaLancellotti/metal-dialect">GitHub - NicolaLancellotti/metal-dialect: MLIR metal dialect</a>：MLIR metal dialect。通过在 GitHub 上创建账号来为 NicolaLancellotti/metal-dialect 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1289431346644779100)** (8 条消息🔥): 

> - `Discord AutoMod`
> - `垃圾信息管理`
> - `反垃圾信息工具` 


- **关于自动清理垃圾信息的讨论**：成员们讨论了使用机器人*自动清理垃圾消息*的可能性。共识是目前已有可用于此目的的工具，并引发了对其细节的进一步询问。
   - 一位成员指出，使用此类工具可以显著减少目前在垃圾信息管理上投入的精力。
- **AutoMod 在移除消息方面的成功**：会议强调了 **AutoMod** 自发布以来已成功从服务器中移除了 **超过 2000 万条违规消息**，这极大地助力了社区治理工作。
   - 社区安全性的提升显而易见，它可能为管理员节省了此前用于审核消息的 **1157 天** 时间。
- **询问特定的反垃圾工具**：一位成员请求获取*特定反垃圾工具*的链接以便进一步研究，这标志着对垃圾信息问题采取了积极主动的态度。
   - 回复中包含了一个指向 [Discord 反垃圾安全更新](https://discord.blog/new-anti-spam-raid-automod-safety-update) 的链接以获取更多信息。
- **讨论 AutoMod 的功能**：成员们注意到其社区设置中已*启用了所有 AutoMod 功能*，旨在提高审核效率。
   - 讨论反映了利用现有工具维护友好环境的承诺。


  

---

### **GPU MODE ▷ #[nccl-in-triton](https://discord.com/channels/1189498204333543425/1289355253392867348/1289355279346958377)** (6 messages): 

> - `Triton 项目协作`
> - `内存管理挑战`
> - `弱内存一致性模型`
> - `Triton 中的学习机会`
> - `项目热情` 


- **Triton 项目协作引发关注**：一位用户表达了尽管缺乏经验但仍渴望参与 Triton 项目的意愿，并表示他们渴望学习。
   - 这种热情得到了其他人的回应，他们也热衷于深入研究具有挑战性的任务。
- **讨论内存管理的复杂性**：一位用户对 **memory management** 的复杂性以及实现一致性提出了担忧，特别是在跨 **nvlink domains** 的弱内存模型中。
   - 他们强调，构建一个具有强一致性模型的原型可能相对简单。
- **弱内存一致性是可以学习的**：另一位成员指出，学习如何处理 **weak memory consistency models** 是可行的，并鼓励专注于通过 nvlink 在单节点内进行 reduction。
   - 他们表示愿意为那些对这一挑战有疑问的人提供帮助。
- **项目难度与热情并存**：一位参与者承认该项目既 **fancy** 又具挑战性，并表示克服此类障碍是黑客精神的一部分。
   - 他们敦促其他人继续关注 Triton 项目的相关进展。


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1289408477474848799)** (18 messages🔥): 

> - `Modular 社区会议`
> - `桌面背景偏好`
> - `YouTube 会议录像` 


- **Modular 社区会议议程公布**：今天的 Modular 社区会议（太平洋时间上午 10 点）议程紧凑，包括来自 [<@447855150409842719>](https://modul.ar/community-meeting-zoom) 关于 **MAX driver & engine API** 的演讲，以及关于 Magic 的问答环节。
   - 邀请参与者通过 [Zoom](https://modul.ar/community-meeting-zoom) 加入，并可以通过 [Modular 社区日历](https://modul.ar/community-meeting) 将未来的活动添加到他们的日历中。
- **社区会议的 YouTube 录像**：所有 Modular 社区会议都会录制并随后发布到 YouTube，包括今天的会议，可通过[此链接](https://www.youtube.com/watch?v=zL0cCHs_0RI&list=PLh0S94-sJw_6UcaIMgpESb5KSVRsuuhnX)观看。
   - 录像方便那些无法参加直播的人观看，确保没有人错过有价值的讨论。
- **聊天中出现对 T 恤的兴趣**：一位用户表达了对 **Modular 主题 T 恤** 的兴趣，表示希望有更多社区周边（swag）。
   - 这个俏皮的建议暗示了通过周边商品建立更强社区认同感的愿望。
- **关于社区会议时区的查询**：一位成员询问 18:00 的 **社区会议时间** 是否为他们的当地时间，得到的答复是肯定的。
   - 另一位成员澄清了时区细节，确保参与者做好加入准备。
- **个人对桌面背景的偏好**：一位成员分享了他们对桌面背景的极简主义处理方式，偏好纯深棕色，但也乐于改进。
   - 在中心加入一个小小的 **mojo fire** 图标的建议表明了对个性化修饰的创意倾向。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://modul.ar/community-meeting-zoom">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。</li><li><a href="https://modul.ar/community-meeting">Google 日历 - 登录以访问和编辑您的日程安排</a>：未找到描述
</li>
</ul>

</div>

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1289300577309819002)** (232 条消息🔥🔥): 

> - `Mojo Language Features`
> - `Embedding Models in Mojo`
> - `Managing Native Dependencies`
> - `Mojopkg Enhancements`
> - `Warnings on MacOS` 


- **增强 Mojo 语言特性的提议**：讨论集中在 Mojo 对高级特性的需求上，例如用于消息传递的命名变体（named variants），以及在不引入新构造的情况下，利用现有构造更好地处理标签联合（tagged unions）。
   - 参与者讨论了定义类型的易用性，以及在语言设计中同时存在标称类型（nominal types）和结构类型（structural types）的影响。
- **在 Mojopkg 中嵌入模型**：重点介绍了 Mojopkg 的嵌入能力，用例包括在单个可执行应用程序中捆绑模型和依赖项。
   - 借鉴了其他语言的例子，展示了如何通过在包中直接包含必要组件来提供更简单的用户体验。
- **Mojopkg 的增强功能**：建议为 Mojopkg 增加加密和更简便的文件结构嵌入等功能，这可以简化依赖管理。
   - 虽然某些功能被认为是小众需求，但将相关文件和模型嵌入包中的想法被认为对各种应用都有潜在益处。
- **处理原生依赖**：讨论了 Mojopkg 简化依赖包含的潜力，从而为用户提供更便捷的安装和配置。
   - 讨论围绕实际实现展开，包括在 Mojo 应用程序中嵌入 Python 等运行时的安装程序。
- **MacOS 上遇到的警告**：一位用户报告称，在为 macOS 15.0 版本构建的对象文件与针对 14.4 版本的链接过程之间，收到了多个关于兼容性的警告。
   - 这些警告虽然不是致命的，但表明了可能需要在未来版本中解决的兼容性问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/hellux/jotdown">GitHub - hellux/jotdown: A Djot parser library</a>：一个 Djot 解析库。可以通过创建账户为 hellux/jotdown 的开发做出贡献。</li><li><a href="https://github.com/VitWW/rfcs/blob/partial_types3/text/0000-partial_types.md">rfcs/text/0000-partial_types.md at partial_types3 · VitWW/rfcs</a>：关于 Rust 变更的 RFC。可以通过创建账户为 VitWW/rfcs 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1289348064376787037)** (189 条消息🔥🔥): 

> - `Nous Research`
> - `Distro Paper Timeline`
> - `AI Model Fine-tuning`
> - `Liquid Foundation Models`
> - `NLP Research Opportunities` 


- **了解 Nous Research**：Nous Research 专注于开源 AI 研究，为独立开发者和爱好者提供合作机会。
   - 他们发布了包括 Hermes 家族在内的多种模型，目前正参与 DisTrO 等项目以加速 AI 发展。
- **即将发布的 Distro 论文**：频道成员表示，Distro 论文的发布预计很快就会宣布。
   - 由于该论文在 AI 社区的相关性，人们对其充满了期待。
- **AI 模型微调的进展**：最近的进展提到了一种新的持续训练模型——Rombodawg 的 Replete-LLM，它在 OpenLLM 7B 模型排行榜上名列前茅。
   - TIES merging 等微调技术被强调为显著提高模型基准测试的方法。
- **Liquid Foundation Models 介绍**：LiquidAI 推出了 Liquid Foundation Models（Liquid 基础模型），包含 1B、3B 和 40B 变体，引起了 AI 社区的关注。
   - 这些模型旨在为 AI 语言模型领域提供新的方法和功能。
- **学生进入 NLP 研究的途径**：频道中的新参与者表达了参与 AI（特别是 NLP）的兴趣，并寻求实习指导。
   - 讨论了来自 AI 研究接触有限的地区（如巴基斯坦）的学生所面临的机会问题，以及通往国际项目的路径。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.marktechpost.com/2024/09/25/minish-lab-releases-model2vec-an-ai-to">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/LiquidAI_/status/1840768716784697688">Liquid AI (@LiquidAI_) 的推文</a>: 今天我们向世界介绍 Liquid Foundation Models (LFMs)，这是我们首系列语言 LFMs：包含 1B、3B 和 40B 模型。(/n)</li><li><a href="https://x.com/altryne/status/1840267263070319047?s=46">Alex Volkov (Thursd/AI) (@altryne) 的推文</a>: 噢天哪... NotebookLM “播客”主持人意识到他们是 AI，这是我最近在这个应用里听到的最棒的事！😂 “我试着给妻子打电话.. 那个号码不是真的” 还有...</li><li><a href="https://x.com/karpathy/status/1840511640317673965">Andrej Karpathy (@karpathy) 的推文</a>: 抱歉，这是针对你提供的任何素材/链接生成的全新点播播客。在 Google 的 NotebookLM 中生成它们：https://notebooklm.google.com/ + 新的 Notebook 链接源...</li><li><a href="https://x.com/ahmad_al_dahle/status/1840097836312211681?s=46">Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>: 制作 Meta AI 语音的幕后花絮。</li><li><a href="https://a16z.com/podcast/distro-and-the-quest-for-community-trained-ai-models/">DisTrO 与社区训练 AI 模型的探索 | Andreessen Horowitz</a>: Nous Research 的 Bowen Peng 和 Jeffrey Quesnelle 讨论了他们加速开源 AI 研究的使命，包括一个名为 DisTrO 的新项目。</li><li><a href="https://huggingface.co/mylesgoose/Llama-3.2-3B-instruct-abliterated-Q8_0-GGUF">mylesgoose/Llama-3.2-3B-instruct-abliterated-Q8_0-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/kamen-rider-build-henshin-rabbit-tank-gif-24237461">假面骑士 Build 变身 GIF - 假面骑士 Build 变身 兔子坦克 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://exa.ai">Exa</a>: Exa API 从网络检索最佳实时数据，以补充您的 AI</li><li><a href="https://tenor.com/view/monday-mood-gif-18424113286394293247">周一心情 GIF - 周一心情 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/benjamindekr/status/1840622126664949943?s=46">Benjamin De Kraker 🏴‍☠️ (@BenjaminDEKR) 的推文</a>: 我刚刚得到了一个 GPT-4o（不是 o1）的回复，其中包含了 20 秒的思考过程... Chain of Thought 正在 4o 上进行测试吗...？</li><li><a href="https://x.com/a16z/status/1839803037562614016?s=46&t=UF7xXn4t0Q6LVvtoFHrVsA">a16z (@a16z) 的推文</a>: 下一个大型开源模型能否由全球独立开发者网络构建？@NousResearch 的 DisTrO 正在展示这种可能性——利用公共互联网训练强大的 AI 模型，而无需...</li><li><a href="https://x.com/N8Programs/status/1840618307235549679">N8 Programs (@N8Programs) 的推文</a>: MLX 刚刚增加了全量微调 (full finetuning)... bf16 格式的 Llama 3.2 3B 速度可达 100-200 tok/sec。冲啊！</li><li><a href="https://youtu.be/41dF0yoz0qo?si=Ny0IqRYz82Qq_NTg">创建基于群体的注意力机制</a>: 研究论文链接：https://lime-georgette-80.tiiny.site Colab Notebook 链接：https://colab.research.google.com/drive/1cVM-GpAEp1nGX4vYx1Rr_tNwQSlFmPeT...</li><li><a href="https://arxiv.org/html/2408.16737v1">更小、更弱、却更好：通过计算最优采样训练 LLM 推理器</a>: 未找到描述</li><li><a href="https://www.marktechpost.com/2024/09/25/minish-lab-releases-model2vec-an-ai-tool-for-distilling-small-super-fast-models-from-any-sentence-transformer/?amp">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/MinishLab/model2vec?tab=readme-ov-file">GitHub - MinishLab/model2vec: Model2Vec: 从任何 Sentence Transformer 蒸馏出一个小型快速模型</a>: Model2Vec: 从任何 Sentence Transformer 蒸馏出一个小型快速模型 - MinishLab/model2vec</li><li><a href="https://calmatters.org/economy/2024/09/california-artificial-intelligence-bill-veto/">为什么 Gavin Newsom 否决了加州监管 AI 的大胆提案</a>: 加州这项立法本要求公司测试 AI 模型可能对社会造成的关键危害。</li><li><a href="https://huggingface.co/datasets/archit11/worldbuilding">archit11/worldbuilding · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1289310671132950559)** (16 条消息🔥): 

> - `Hyperparameter Adjustment`（超参数调整）
> - `Multimodal Input LLMs`（多模态输入 LLM）
> - `Open-sourcing Models`（开源模型）
> - `RL Techniques in Inference`（推理中的 RL 技术）
> - `Inference on CPU`（CPU 推理）


- **超参数调整是必要的**：一位成员指出，在训练不同尺寸的模型时，*确实需要进行超参数调整*。
   - 具体而言，他们提到对于像 70B 和 40B 这样的大型模型，需要*更少的 epoch* 和*更低的学习率*。
- **讨论最便宜的多模态输入 LLM**：一位成员建议，目前使用 Together API 的 **Llama 3.2** 可能是多模态输入 LLM 中最便宜的选择。
   - 另一位成员补充了价格细节，指出 *11B vision instruct* 的价格为 **$0.18/1M tokens**，而 *90B* 为 **$1.20/1M tokens**。
- **开源模型可能使社区受益**：围绕开源像 **O1** 这样的模型是否对社区有益展开了讨论。
   - 成员们表示，虽然核心进步来自于使用新 RL 技术的**推理过程**，但将其公开仍然可能产生巨大的社区价值。
- **在 CPU 上运行模型**：一位成员确认 **ColpaLigemma3B** 可以在 CPU 上运行，但速度有限且有 RAM 限制。
   - 据报告，它不需要超过 **3GB RAM**，如果使用 quantization（量化），可以降低到 **500MB**。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1289496475067486263)** (4 条消息): 

> - `Medical AI Research Papers`（医疗 AI 研究论文）
> - `LLM Models in Healthcare`（医疗保健中的 LLM 模型）
> - `AI Ethics in Medicine`（医学中的 AI 伦理）


- **上周医疗 AI 亮点**：最新的综述包括一项关于 **o1** 在医学领域的**初步研究**，评估了其作为 AI 医生的潜力，并介绍了 **DREAMS** 和 **Uni-Med** 等多种模型。
   - 讨论的关键框架涉及用于肿瘤学的 **Digital Twin**（数字孪生）技术和用于抑郁症评估的 **InterMind**，展示了医疗 LLM 方法论的进展。
- **医疗 AI 中的新兴模型**：探索了 **O1 in Medicine** 和 **Genome Language Model** 等新模型，强调了 AI 驱动的医疗解决方案中的机遇与挑战。
   - 其他基准测试包括针对中文 LLM 的 **CHBench** 以及针对姑息治疗的 **PALLM** 评估，强调了医疗 LLM 的可靠性。
- **AI 伦理讨论**：重点关注伦理问题，包括评估医学影像 AI 中的 **confidence intervals**（置信区间）以及生成式 AI 在临床环境中的当前就绪状态。
   - 随着医疗领域整合 AI 技术，这些讨论对于确保维持伦理标准至关重要。
- **通过 LLM 进行患者教育**：创新应用如**为放射学报告微调 LLM** 以及利用 LLM 进行**背痛**教育，展示了在患者护理中的实际用途。
   - 通过检索上下文（retrieved context）和持续预训练（continuous pretraining）来增强医疗 AI 的努力，标志着该领域的持续发展。
- **新资源与综述**：资源包括一份关于**医疗保健中的 LLM** 的全面综述，阐明了从通用应用到特定医学应用的演变。
   - 还重点介绍了对 **EHR 信息检索**的检查以及近距离放射治疗（brachytherapy）的 AI 指南，反映了该领域专业知识的不断扩展。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1840020394880667937">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：上周医疗 AI：顶级研究论文/模型 🏅（2024年9月21日 - 9月27日）🏅 本周医疗 AI 论文：o1 在医学中的初步研究：我们离 AI 医生更近了吗？作者...</li><li><a href="https://proem.ai/paper/oa/W4402356829">proem</a>：由科学研究支持的各类问题解答
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1289399545025003550)** (13 条消息🔥): 

> - `DisTrO AI Project`
> - `AI Server Rankings`
> - `Quantum Computing in Data Generation`
> - `EleutherAI Community`
> - `VPTQ Quantization Algorithm` 


- **DisTrO AI Project 加速 open source 进程**：在最近的 [AI + a16z](https://a16z.com/podcasts/ai-a16z/) 播客节目中，来自 [Nous Research](https://nousresearch.com/) 的 Bowen Peng 和 Jeffrey Quesnelle 讨论了他们的 [DisTrO](https://github.com/NousResearch/DisTrO) 项目，该项目旨在实现跨互联网的 AI 模型快速训练。
   - Jeffrey 强调了来自 closed source 模型的潜在威胁，他表示：*“如果我们拿不到 Llama 4 怎么办？那真的是一个现实的生存威胁……”*
- **AI Server 排行榜包含 Nous Research**：一位成员分享了一个 Google Spreadsheet，列出并排名了各种 AI Server，并提到 [Nous Research](https://nousresearch.com/) 也位列其中。
   - 该表包含了社区项目和资源，但有一点需要注意，这些评分反映了在 LLM 研究中的个人实用性，需谨慎参考。
- **Quantum Computing 在 synthetic data 领域展现潜力**：讨论围绕 [Quantum Computing](https://x.com/tdatascience/status/1840225741536948561?s=46) 在 synthetic data 生成中的作用展开，重点关注其涌现能力，并通过一个简单的 quantum generator 实验进行了说明。
   - 更多见解通过 [Towards Data Science](https://towardsdatascience.com/a-basic-introduction-to-quantum-gans-4dbdc27ccb54) 上的一篇名为 *《A Basic Introduction to Quantum GANs》* 的文章进行了分享。
- **关于 LLM 训练和功能的社区讨论**：成员们表达了参与专注于 LLM 训练社区的兴趣，特别提到 EleutherAI server 是此类讨论的一个很有前景的场所。
   - 还有建议探索 Mech Interp 和 Alignment Jams 等其他 server，以获取有关 LLM 运行的更多见解。
- **VPTQ quantization algorithm 发布**：Microsoft 在 [GitHub](https://github.com/microsoft/vptq) 上发布了一个名为 [VPTQ](https://github.com/microsoft/vptq) 的新项目，引入了一种灵活的 low-bit quantization 算法，旨在优化模型性能。
   - 该工具专为寻求高效模型训练和部署解决方案的研究人员设计。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/tdatascience/status/1840225741536948561?s=46">Towards Data Science (@TDataScience) 的推文</a>：Quantum Computing 在 synthetic data 生成中的作用正日益受到关注。一个使用“quantum”生成器的简单实验展示了其巨大潜力的一小部分。阅读更多来自 @jamarinval 的内容...</li><li><a href="https://a16z.com/podcast/distro-and-the-quest-for-community-trained-ai-models/">DisTrO 与社区训练 AI 模型的探索 | Andreessen Horowitz</a>：Nous Research 的 Bowen Peng 和 Jeffrey Quesnelle 讨论了他们加速 open source AI 研究的使命，包括一个名为 DisTrO 的新项目。</li><li><a href="https://docs.google.com/spreadsheets/d/1DlBT1pF8-zMECntRWXFsL46gZyvNp1BJlJ6LXGze4dA/edit?gid=0#gid=0">discord AI 领域 - 与任何人分享！</a>：未找到描述</li><li><a href="https://github.com/microsoft/vptq">GitHub - microsoft/VPTQ: VPTQ，一种灵活且极低比特的 quantization 算法</a>：VPTQ，一种灵活且极低比特的 quantization 算法 - microsoft/VPTQ
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1289496475067486263)** (4 messages): 

> - `本周医学 AI 论文`
> - `新型医学 LLMs`
> - `医疗 AI 的框架与方法论`
> - `医学 LLM 应用`
> - `医疗 AI 伦理` 


- **本周医学 AI 论文：我们离 AI 医生更近了吗？**：重点论文《o1 在医学领域的初步研究》（*A Preliminary Study of o1 in Medicine*）由该领域的专家撰写，探讨了 AI 充当医生的潜力。
   - 该论文被评为**本周医学 AI 论文**，展示了其在关于 AI 在医疗保健中角色的持续讨论中的相关性。
- **新兴模型：DREAMS 和 Uni-Med**：新模型如 **DREAMS**（一个用于医学 LLM 的 Python 框架）和 **Uni-Med**（一个统一的医学通用 LLM）正在医疗 AI 领域引起关注。
   - 这些进展标志着医疗应用正向更专业、更强大的工具转变。
- **医疗 AI 的创新框架**：诸如**肿瘤运营数字孪生**（Digital Twin for Oncology Operations）和**增强医疗 AI 的护栏**（Enhancing Guardrails for Healthcare AI）等创新方法论旨在提高医疗 AI 应用的安全性和效率。
   - 此外，像 **InterMind** 这样的工具提供了由 LLM 驱动的抑郁症评估，体现了对心理健康的关注。
- **LLM 在医疗保健中的应用**：**用于心理健康严重程度预测的 LLM** 和**针对放射科报告微调 LLM** 是近期展示 AI 增强患者护理潜力的应用。
   - 此外，目前正在努力通过检索上下文和持续预训练来增强医疗 LLM，这可能会完善临床实践。
- **医疗 AI 的伦理**：关于**医学影像 AI 中的置信区间**和**生成式 AI 的临床应用就绪性**的讨论突显了人们对 AI 技术伦理日益增长的关注。
   - 随着 AI 技术越来越多地整合到临床环境中，解决这些伦理考量至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1840020394880667937">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：上周医学 AI：顶级研究论文/模型 🏅（2024年9月21日 - 9月27日）🏅 本周医学 AI 论文《o1 在医学领域的初步研究：我们离 AI 医生更近了吗？》作者...</li><li><a href="https://proem.ai/paper/oa/W4402356829">proem</a>：由科学研究支持的问题解答
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1290344451473932329)** (4 messages): 

> - `AGI 推测`
> - `资助 AGI 开发` 


- **关于实现 AGI 的推测**：一位成员强调，“在 AGI 真正实现之前，没有人知道它是否能实现或如何实现”，对该领域预测的确定性表示怀疑。
   - 另一位成员补充说，那些声称知道的人可能只是在“推测和博取关注”，表达了对大胆断言的怀疑。
- **金钱作为 AGI 的解决方案**：在一种截然不同的观点中，一位成员自信地宣称，“我知道如何实现 AGI！！！”暗示了一个明确的解决方案。
   - 他们的答案？“钱。很多很多钱。”这表明资金资源是开启 AGI 开发的关键。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1289317076539805751)** (182 条消息🔥🔥): 

> - `Perplexity 性能问题`
> - `Felo vs Perplexity 对比`
> - `API 不一致性`
> - `文档上传 vs 粘贴`
> - `LaTeX 公式讨论` 


- **讨论了 Perplexity 的性能问题**：用户报告了 Perplexity 在网页搜索和学术论文搜索之间切换时响应不一致的问题，其中一个案例没有产生任何引用。
   - 成员们对这些不一致性究竟是功能特性还是 Bug 表示担忧。
- **Felo vs Perplexity 对比**：讨论强调，许多用户发现 Felo 在学术搜索方面比 Perplexity 更有效，理由是它能更好地获取相关论文。
   - 用户还注意到 Felo 的界面功能（如悬停预览来源）比 Perplexity 提升了研究体验。
- **提出 API 不一致性问题**：提出了关于 API 在 JSON, HTML 和 Markdown 等格式中提供一致输出能力的问题，用户对混合的结果表示沮丧。
   - 建议包括尝试调整 temperature 和 top-p 等参数，以提高 API 响应的一致性。
- **对话中上传文档 vs 粘贴内容**：一位用户询问上传文档还是直接将内容粘贴到对话中，哪种方式能让 AI 提供更好的引用。
   - 回复建议对两种方法都进行测试，以评估哪种能产生更可靠的交互。
- **LaTeX 公式讨论**：一位用户分享了一组 LaTeX 格式的复杂方程，并强调了 Claude Opus 等模型在评估这些方程时的差异。
   - 用户最终找到了为这些方程提供背景信息的参考论文，解决了他们的疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1840742628570059008?s=61">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: WIP 🚧: Perplexity 正在测试增加葡萄牙语和意大利语。</li><li><a href="https://www.perplexity.ai/backtoschool">Perplexity - Race to Infinity</a>: 欢迎回到学校！仅限两周，领取一个月免费的 Perplexity Pro。推荐你的朋友，如果你的学校达到 500 人注册，我们将把免费月份升级为整整一年...</li><li><a href="https://huggingface.co/blog/llama31#:~:text=56%C2%B0F.%3C%7Ceot_id%7C%3E-,Custom%20Tool%20calling,-Llama%203.1%20Instruct))">Llama 3.1 - 405B, 70B &amp; 8B 具备多语言能力和长上下文</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1289416993086115932)** (16 条消息🔥): 

> - `多元宇宙的新见解`
> - `以色列-真主党冲突升级`
> - `新型 AI 设计工具`
> - `德克萨斯州县级 AI 应用`
> - `30 年来首款精神分裂症药物` 


- **通过新见解探索多元宇宙**：Perplexity AI 强调了关于 **Multiverse**（多元宇宙）的新发现，预示着理论物理学领域的**令人兴奋的发展**。点击[此处](https://www.youtube.com/embed/TxMVKnGSbG4)查看讨论。
   - 这次演讲深入探讨了关于现实和宇宙结构的新视角，激发了科学爱好者的好奇心。
- **以色列-真主党冲突升级**：最近的讨论引发了对**以色列-真主党冲突**的关注，展示了该地区潜在的升级和紧张局势。更多详情请参阅[当前进展](https://www.perplexity.ai/search/israel-hezbollah-war-escalatio-FuK4.tsqSXSAVxvc7JRq8w)。
   - 参与者分享了关于这场冲突影响的见解，包括历史背景和地缘政治利益。
- **新型 AI 设计工具亮相**：分享了一个关于**新型 AI 设计工具**的链接，展示了可能重塑各领域创意流程的创新。点击[此处](https://www.perplexity.ai/search/new-ai-design-tools-1Ge5qhkHR.WqMgiJT4DBpQ)了解更多关于这些工具的信息。
   - 讨论强调了这些工具如何**提高生产力**并激发设计师的创造力。
- **德克萨斯州县级的创新 AI 应用**：一名成员引用了一个详细介绍**德克萨斯州县级 AI 应用**的页面，说明了地方政府如何利用技术。如需了解见解，请访问[此资源](https://www.perplexity.ai/page/texas-county-ai-applications-gffwruR9QIK4U72mkQUK9Q)。
   - 这些应用展示了 AI 在公共服务和行政管理中的实际用途。
- **30 年来首款精神分裂症药物上市**：Perplexity AI 宣布了 30 年来**首款精神分裂症药物的推出**，标志着精神健康治疗领域的重大突破。观看[此视频](https://www.youtube.com/embed/7FX4rZdtgUQ)了解更多见解。
   - 对话强调了这一进展对患者护理和治疗选择的潜在影响。



**提到的链接**：<a href="https://www.youtube.com/embed/7FX4rZdtgUQ">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1289450748173881445)** (2 条消息): 

> - `PPLX API 集成问题`
> - `房地产列表` 


- **PPLX API 返回过时信息**：一名成员报告称，在集成 **PPLX API** 时，返回的房地产列表与网站上提供的准确信息相比已经过时。
   - 他们指出，在两种情况下使用相同的 Prompt 产生了不同的结果。
- **JSON 输出的挑战**：同一名成员对 AI 在集成过程中持续以 **raw JSON** 格式输出的能力表示担忧。
   - 他们正在寻求关于其设置或 API 使用中可能存在的错误的指导。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1289304066043809876)** (193 条消息🔥🔥): 

> - `OpenRouter Rate Limits` (OpenRouter 速率限制)
> - `Model Performance Issues` (模型性能问题)
> - `Translation Model Recommendations` (翻译模型推荐)
> - `Frontend Chat GUI Options` (前端聊天 GUI 选项)
> - `Gemini and Search Functionality` (Gemini 与搜索功能) 


- **OpenRouter 面临速率限制挑战**：用户报告在使用 **Gemini Flash** 时，由于配额耗尽频繁出现 **429 错误**，并希望 Google 能尽快提高配额。
   - 流量负载是一个持续存在的问题，正如最近用户间的讨论所示，这影响了平台的可用性。
- **对维护后模型性能的担忧**：某些模型（如 **Hermes 405B free**）在维护更新后表现出性能质量下降，引发了关于供应商变更的猜测。
   - 鼓励用户检查 OpenRouter 中的 **Activity 页面**，以确认他们是否仍在使用首选的供应商。
- **翻译模型推荐**：一位用户询问是否有针对对话翻译且没有严格限制的高效翻译模型，并表达了对 **GPT4o Mini** 的不满。
   - 带有 dolphin 微调的开源权重模型被建议作为提供更多灵活性的选项。
- **前端聊天 GUI 建议**：一位用户寻求关于支持中间件灵活性以管理 AI 模型交互的聊天 GUI 建议，**Streamlit** 被提及为潜在解决方案。
   - 其他选项如 **Typingmind** 因其在与多个 AI Agent 交互时的可定制功能而受到关注。
- **Gemini 模型搜索功能**：用户对在 **Gemini** 模型中启用类似于 **Perplexity** 的直接搜索功能感兴趣，但具体的使用限制尚不明确。
   - 讨论中提到了 Google 的 **Search Retrieval API 参数**，但其实现方式和有效性仍在评估中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>：LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 在浏览器本地存储数据。</li><li><a href="https://aider.chat/2024/09/26/architect.html#results">分离代码推理与编辑</a>：Architect 模型描述如何解决编程问题，Editor 模型将其转化为文件编辑。这种 Architect/Editor 方法产生了 SOTA 基准测试结果。</li><li><a href="https://sillytavern.app/">SillyTavern - 面向高级用户的 LLM 前端</a>：未找到描述</li><li><a href="https://x.com/openrouterai/status/1839738812877918617?s=46&t=nM71JKV50FJ0CR4r6r2_Rg">来自 OpenRouter (@OpenRouterAI) 的推文</a>：Chatroom 现在默认以折叠方式显示模型的推理响应。o1 vs Gemini vs Sonnet 在 🍓 问题上的表现：</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#error-code-429">未找到标题</a>：未找到描述</li><li><a href="https://docs.typingmind.com/typingmind-custom/branding-and-customizations/create-multiple-ai-agents-within-a-chat-instance">在聊天实例中创建多个 AI Agent</a>：在单个聊天实例中创建多个 AI Agent 可以实现个性化且动态的交互体验。通过为每个 AI Agent 定制特定的数据集，您可以获得广泛的...</li><li><a href="https://github.com/Mintplex-Labs/anything-llm/issues/1476#issuecomment-2123480889">AnythingLLM 的移动 App 版本？· Issue #1476 · Mintplex-Labs/anything-llm</a>：您想看到什么？不确定这里是否是提问的合适地方，但是否有开发 AnythingLLM 移动端应用的意愿？是否有任何正在进行的进展？如果没有，我很乐意...</li><li><a href="https://github.com/Mintplex-Labs/">Mintplex Labs</a>：为每个人提供的 AI 应用。Mintplex Labs 拥有 16 个代码库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1289305089634340935)** (178 条消息🔥🔥): 

> - `Flux 模型见解`
> - `Stable Diffusion 设置与性能`
> - `图像生成技巧`
> - `社区艺术贡献`
> - `AI 艺术与人类艺术之争` 


- **Flux 模型令人印象深刻**：一位成员表达了对 kohya_ss 成就的钦佩，指出 Flux 模型仅需 12G VRAM 即可进行训练。
   - 他们对已展示出的性能提升和功能进步感到兴奋。
- **Nvidia 驱动问题影响性能**：有成员担心新的 Nvidia 驱动会导致 8GB VRAM 显卡在生成 SDXL 图像时严重减速，据报告生成时间从 20 秒增加到了 2 分钟。
   - 鉴于这些问题，成员们建议不要更新到最新驱动，并讨论了这对渲染能力产生的影响。
- **区域提示词（Regional Prompting）挑战**：成员们分享了在 Stable Diffusion 中使用区域提示词的困难经历，指出在使用如“2 boys and 1 girl”之类的提示词时会出现角色混淆的问题。
   - 建议先从通用提示词开始，然后再应用区域引导以获得更好的效果。
- **社区参与 AI 艺术**：社区邀请成员贡献他们的 AI 艺术作品，有机会被刊登在《The AI Art Magazine》上，投稿截止日期为 10 月 20 日。
   - 鼓励社区成员加入庆祝数字艺术的行列，分享他们的创意表达。
- **AI 艺术质量辩论**：一场关于 AI 艺术与人类艺术价值的热烈讨论展开了，一些人认为人类艺术保持着更高的质量和深度。
   - 一位成员反驳道，由图像算法生成的 AI 艺术同样属于艺术表达的范畴。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/jasperai/Flux.1-dev-Controlnet-Upscaler">Flux.1-dev Upscaler - jasperai 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://art-magazine.ai">The AI Art Magazine</a>：未找到描述</li><li><a href="https://github.com/filipstrand/mflux?tab=readme-ov-file#-installation>">GitHub - filipstrand/mflux: 基于 Huggingface Diffusers 实现的 FLUX MLX 移植版本。</a>：基于 Huggingface Diffusers 实现的 FLUX MLX 移植版本。 - filipstrand/mflux</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1fslym2/if_you_have_a_gpu_with_low_vram_like_3060_ti_8gb/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - 深入了解</a>：未找到描述
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1289305548629479525)** (105 条消息🔥🔥): 

> - `Aider 的代码编辑能力`
> - `欧盟 AI 法案中的监管`
> - `视频翻译公告`
> - `使用 AI 辅助写作`
> - `华为设备的 ChatGPT 可访问性` 


- **Aider 评测 LLM 编辑技能**：成员们讨论了 Aider 的功能，指出它在与擅长“编辑”代码的 LLM 配合时效果最好，正如其[排行榜](https://aider.chat/docs/leaderboards/)所示。一些人对 Aider 基准测试的可靠性表示怀疑，特别是提到 *Gemini Pro 1.5 002* 未能得到充分测试。
- **欧盟 AI 法案引发辩论**：关于欧盟新 AI 法案的讨论仍在继续，对其在多模态 AI 监管方面的影响存在不同意见，并澄清聊天机器人仍将被归类为二级监管。针对监管审查背景下公司发布新技术的潜在影响，人们表达了担忧。
- **Meta 的视频翻译功能**：一位成员提到 Meta 即将发布唇形同步视频翻译功能，并确认该功能已出现在 Meta 平台中。这一功能激发了成员们对翻译服务的兴趣，尤其是用于内容创作。
- **将 AI 用于写作项目**：对话围绕利用 AI 辅助写作展开，成员们提供了在利用 GPT 等 AI 进行内容创作时保持个人风格的策略。技巧包括向 GPT 提供个人写作样本，以帮助输出内容与个人语调保持一致。
- **华为设备上的 ChatGPT 访问**：一位成员询问了在华为设备上访问 ChatGPT 的可能性，质疑在没有 Google 服务的情况下登录的可行性。对话凸显了尽管存在当前设备限制，社区仍渴望获得 AI 功能。



**提到的链接**：<a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑技能的定量基准测试。

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1289340287914872853)** (28 条消息🔥): 

> - `GPT-4.5-o 发布`
> - `Advanced Voice Mode 限制`
> - `Custom GPTs 与 Voice Mode`
> - `语音功能的付费方案` 


- **对 GPT-4.5-o 发布的呼声**：成员们对 **GPT-4o** 的表现表示失望，认为其存在缺陷，并要求发布 **GPT-4.5-o**。Sam Altman 曾评价其为“最笨的 LLM”，这一言论被引用以强调改进的紧迫性。
   - 在讨论背景中，用户指出需要超越目前 GPT-4 系列局限性的更强推理能力。
- **对 Advanced Voice Mode 的困惑**：成员们寻求关于 Advanced Voice Mode **每日时间限制**的澄清，有报告称 **1 小时限制**包括了模式开启的时间。一位用户提到在使用一段时间后遇到了 **“剩余 15 分钟”** 的提示。
   - 用户对可访问性表示担忧，特别是关于 **语音模式时间如何累积**，以及在不主动使用时是否需要将其关闭。
- **Custom GPTs 中的 Voice Mode 可用性**：已确认 **Advanced Voice 对话在 Custom GPTs 中不可用**，尝试使用时会被重定向到标准聊天。用户对 **标准语音模式** 的可用性表示困惑，尤其是在 Custom GPTs 设置内部。
   - 一位用户报告称，即使开启了语音模式，也只能转录输入而不会语音播报响应，这引发了对 **标准语音功能** 的担忧。
- **语音功能的潜在付费方案**：讨论暗示可能很快会推出 **Advanced Voice Mode 的付费方案**。即便是长期订阅用户也对目前的限制表示沮丧，并对新功能的可访问性提出质疑。
   - 评论回顾了 **GPT-4** 过去的限制，对比了现状，并表达了对能够改善可访问性的变革的期待。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1289338072709861517)** (5 条消息): 

> - `Flutter 代码 Assistant 问题`
> - `管理 Assistant Runs`
> - `Prompt 管理` 


- **Flutter 代码触发 Thread 错误**：一位用户遇到了错误，提示 Thread `thread_ey25cCtgH3wqinE5ZqIUbmVT` 已经有一个活跃的 run，导致无法发送新请求。
   - 另一位成员建议用户可以等待当前 run 完成，或者使用相关参数手动取消活跃的 run。
- **增加等待时间解决 Thread 问题**：该用户通过将 Thread 执行的等待时间从 10 秒增加到 **15 秒** 解决了问题，从而消除了错误。
   - 这一调整确保在发起进一步请求之前，已经充分考虑到了活跃 run 的完成情况。
- **基于条件的 Thread 执行**：有建议提出利用一个指示 Thread 是否执行完毕的参数，以避免不必要的等待时间。
   - 使用这种条件检查可以优化流程，并减少 Thread 管理期间的等待周期。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1289338072709861517)** (5 条消息): 

> - `Flutter 代码错误`
> - `Thread 管理`
> - `Prompt 管理` 


- **由于活跃 Thread 导致的 Flutter 代码错误**：一位用户遇到错误提示 Thread `thread_ey25cCtgH3wqinE5ZqIUbmVT` 已有活跃的 run，表明之前的执行仍在进行中。
   - 另一位用户建议要么等待 run 完成，要么使用带有相应 ID 的 `cancel` 函数手动取消它。
- **通过增加等待时间解决**：原用户通过取消活跃的 Thread run 解决了错误，结果发现该 run 正是已经在运行的那一个。
   - 他们发现，为了避免错误，必须等待 15 秒，而不是最初添加的 10 秒。
- **利用执行状态参数**：为了改进 Thread 管理，一位用户建议采用一个指示 Thread 是否完成执行的参数，从而实现更高效的处理。
   - 这种方法可以防止在开始新操作或处理现有 Thread 之前出现不必要的等待。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1289338720235032698)** (90 条消息🔥🔥): 

> - `新成员介绍`
> - `ICLR 和 NeurIPS 活动协调`
> - `Liquid AI 的 Foundation Models`
> - `新加坡的登革热`
> - `开源 LLM 训练` 


- **新成员加入对话**：几位新成员介绍了自己，包括来自新加坡的全栈工程师和来自葡萄牙的数据工程师，他们都渴望合作并做出贡献。
   - 他们表达了对 AI 项目和开源贡献的热情，为社区营造了合作的氛围。
- **即将举行的 AI 会议协调**：成员们讨论了参加即将举行的 ICLR 和 NeurIPS 等会议的情况，并提到新加坡将主办 ICLR 以及相关的聚会计划。
   - 现场还有关于活动安保角色以及在新加坡进行潜在聚会的轻松讨论。
- **Liquid AI 发布 Foundation Models**：Liquid AI 宣布推出其 Liquid Foundation Models (LFMs)，强调了令人印象深刻的基准测试分数和高效的架构。
   - 他们的目标是通过针对多种硬件解决方案优化的模型来服务于各个行业，并邀请用户在其平台上尝试他们的新 AI。
- **引发登革热担忧**：讨论中涉及了新加坡的登革热疫情，成员们分享了关于这种蚊媒疾病的个人经历和担忧。
   - 讨论了导致东南亚登革热爆发的因素，揭示了其对公共卫生的影响。
- **探索开源 LLM 开发**：成员们表达了对参与开源 LLM 训练项目的兴趣，展示了他们在 Machine Learning 和 Computer Vision 方面的背景。
   - 有人询问目前有哪些项目需要帮助，反映出参与协作式 AI 开发的强烈愿望。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/huggingface/status/1443246197779664903">来自 Hugging Face (@huggingface) 的推文</a>: EleutherAI 的 GPT-J 现已加入 🤗 Transformers：一个拥有 60 亿参数、具备惊人生成能力的自回归模型！它在以下方面表现出令人印象深刻的结果：- 🧮算术 - ⌨️代码编写 - 👀NLU - 📜Pa...</li><li><a href="https://www.liquid.ai/liquid-foundation-models">Liquid Foundation Models：我们的首系列生成式 AI 模型</a>: 宣布推出首系列 Liquid Foundation Models (LFMs) —— 新一代生成式 AI 模型，在各种规模下均实现了 state-of-the-art 的性能，同时保持了更小的内存占用...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1289341370917589023)** (45 条消息🔥): 

> - `Process Reward Models`
> - `Value Functions in RL`
> - `Sparsity Masks in LLMs`
> - `Swarm LLM Architecture`
> - `Physics Simulation with Equivariant Representations` 


- **理解 Process Reward Models 与 Value Functions 的区别**：一位成员对 Reinforcement Learning (RL) 中 **Process Reward Model (PRM)** 与学习到的 **value function** 之间的区别表示困惑，并强调了两者如何影响决策中的各个步骤。
   - 另一位成员澄清说，PRM 侧重于独立于最终结果的步骤级评估，而 value functions 依赖于最终结果，这导致在对错误进行惩罚时可能存在差异。
- **强化学习数据效率的提升**：对话指出，使用 **PRMs** 可以提高 RL 中的数据效率和训练稳定性，与仅依赖 value functions 相比，提供了更清晰的反馈机制。
   - 这一观察引发了推测：虽然这两种模型在理论上可能趋于一致，但使用 PRMs 可能更好地解释了 RL 模型所缺失的类人推理过程。
- **关于 LLM 稀疏性与速度的讨论**：一位成员建议探索使用 **1-bit BitNet** 结合 sparsity masks 的可能性，以此在提升 LLM 速度的同时实现 ternary 性能。
   - 这引起了另一位参与者的兴趣，他提到了利用 sparse tensor core 操作来有效实现这些想法的潜力。
- **Swarm LLM 架构咨询**：一位成员联系了其他从事 **swarm LLM architecture** 工作的人员，寻求合作或分享该主题的见解。
   - 这反映了人们对利用分布式或并发学习策略的 LLM 开发创新方法的持续关注。
- **使用 Equivariant Representations 的物理模拟**：一位成员提出，拥有物体的 **translation, rotation, and volume equivariant representation** 可以通过直接应用基于物理的形状匹配技术来简化物理模拟。
   - 这表明模型设计中几何与物理的融合，可能带来更直观、更高效的模拟。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/pavlomolchanov/status/1839501906907181104">Pavlo Molchanov (@PavloMolchanov) 的推文</a>：🚀 @NeurIPSConf Spotlight! 🥳 想象一下仅使用 sparsity mask 来微调 LLM！在我们最新的工作中，我们冻结了 LLM 并使用 2:4 structured sparsity 来为每个线性层学习二进制掩码。T...</li><li><a href="https://arxiv.org/abs/2211.14275">通过过程和结果反馈解决数学应用题</a>：最近的研究表明，要求语言模型生成推理步骤可以提高在许多推理任务上的表现。当超越 prompting 时，这就提出了一个问题：我们应该如何监督...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1289702908840906844)** (2 条消息): 

> - `lm-evaluation-harness library`
> - `vLLM model metrics` 


- **关于 vLLM 指标提取的咨询**：一位成员询问在对 benchmark 任务使用 `simple_evaluate` 函数时，是否有办法从 **lm-evaluation-harness library** 中提取 **vLLM metrics object**。
   - 他们特别提到想要诸如 **time to first token** 和 **time in queue** 之类的指标。
- **表达感谢**：另一位成员对 Baber 的帮助表示了感谢。
   - 这一致谢凸显了社区内互助的互动氛围。


  

---

### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1290385064575766558)** (1 条消息): 

> - `ExecuTorch 信息`
> - `多模态模型指南`
> - `硬件设置咨询` 


- **硬件设置咨询**：为了提供有效的帮助，需要明确用户的**硬件规格**、打算运行的**模型**，以及他们构思的特定**视觉任务**详情。
   - *你对 ML 框架有多少经验？* 这些信息可以极大地帮助量身定制所提供的协助。
- **ExecuTorch 概览**：**ExecuTorch** 是一个 [PyTorch](https://pytorch.org/) 平台，旨在允许在各种设备（包括 AR/VR 和移动系统）上**定制和部署** PyTorch 程序。
   - 目前，`executorch` pip 包处于 alpha 阶段，支持 Python 版本 **3.10 和 3.11**，并兼容 **Linux x86_64** 和 **macOS aarch64**。
- **ExecuTorch 使用注意事项**：预构建的 `executorch.extension.pybindings.portable_lib` 模块允许运行 **.pte** 文件，但仅包含**核心 ATen 算子**，并使用 **XNNPACK** 后端委托。
   - 用户提到他们的用例*相当小众*，这表明需要对 ExecuTorch 功能有更深入的见解。
- **多模态模型重点**：该频道主要针对**多模态模型**的研究讨论，建议用户查看 **/r/localllama** 以获取更集中的指南和资源。
   - 鼓励成员遵循相关指南，因为当前的频道讨论可能与更具技术性的设置咨询不直接一致。



**提及的链接**: <a href="https://pypi.org/project/executorch/#:~:text=ExecuTorch%20is%20a%20PyTorch%20platform%20that%20provides,">executorch</a>: 面向移动端、嵌入式和边缘设备的 PyTorch 端侧 AI

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1289527633813704715)** (95 条消息🔥🔥): 

> - `Torchtune 训练问题`
> - `Torchtune 的动态 Recipe CLI`
> - `VRAM 效率与 GPU 利用率`
> - `分布式训练中的错误处理设置`
> - `改进 CLI 参数的配置管理` 


- **优化 Torchtune 中的训练设置**：用户讨论了各种配置，以优化 Llama 3.1 8B 的训练速度，使用了 `batch_size`、`fused` 和 `fsdp_cpu_offload` 等设置。
   - 结论是启用 `packed=True` 显著减少了 epoch 时间，而 `enable_activation_checkpoint` 和 `fsdp_cpu_offload` 应设置为 `False` 以获得更好的计算效率。
- **为 Recipe 创建动态 CLI**：讨论了一项开发动态命令行界面 (CLI) 的提案，旨在为 Torchtune 中的每个 Recipe 生成特定的帮助文本。
   - 使用 `tyro` 库，展示了一种创建灵活解析器的方法，该解析器可以整合来自 YAML 文件的配置细节。
- **实现分布式训练的错误处理**：建议使用 `torch.distributed` 中的 record 工具来增强分布式训练运行中的错误处理。
   - 通过生成捕获异常的错误日志演示了测试过程，从而可以更轻松地调试训练期间遇到的问题。
- **影响训练速度的 VRAM 限制**：分析了单张 A100 训练受 VRAM 限制，与使用多张 A100 时 GPU 利用率成为瓶颈之间的关系。
   - 注意到通过更高的 `batch_size` 提高 GPU 利用率有利于更平滑的训练，但对于可能减慢过程的节省 VRAM 的方法建议保持谨慎。
- **增强文档配置体验**：讨论了记录配置的重要性，以及通过为 Torchtune Recipe 提供更清晰的 CLI 帮助来改善用户体验。
   - 建议为特定 Recipe 参数动态生成帮助文本，可以减轻困惑并简化参数调整过程。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/brentyi/tyro">GitHub - brentyi/tyro: CLI interfaces &amp; config objects, from types</a>: 基于类型的 CLI 接口与配置对象。通过在 GitHub 上创建一个账户来为 brentyi/tyro 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/issues/1710">torch.distributed.elastic.multiprocessing.errors.ChildFailedError · Issue #1710 · pytorch/torchtune</a>: 上下文 :- 我正尝试在 2 张拥有 40GB VRAM 的 A-100 GPU 上运行分布式训练。Batch size 为 3，gradient accumulation=1。我在下方附上了配置文件以了解更多详情以及...</li><li><a href="https://github.com/mirceamironenco/torchtune/blob/add-distrib-error-record/torchtune/_cli/run.py#L82">torchtune/torchtune/_cli/run.py at add-distrib-error-record · mirceamironenco/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建一个账户来为 mirceamironenco/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/elastic/multiprocessing/errors/error_handler.py#L42">pytorch/torch/distributed/elastic/multiprocessing/errors/error_handler.py at main · pytorch/pytorch</a>: Python 中具有强大 GPU 加速能力的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py#L916">pytorch/torch/distributed/run.py at main · pytorch/pytorch</a>: Python 中具有强大 GPU 加速能力的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/facebookresearch/fairseq/blob/main/fairseq/tasks/audio_pretraining.py#L42)">fairseq/fairseq/tasks/audio_pretraining.py at main · facebookresearch/fairseq</a>: 用 Python 编写的 Facebook AI Research 序列到序列工具包。 - facebookresearch/fairseq</li><li><a href="https://github.com/facebookresearch/fairseq/blob/main/fairseq/dataclass/utils.py#L53.">fairseq/fairseq/dataclass/utils.py at main · facebookresearch/fairseq</a>: 用 Python 编写的 Facebook AI Research 序列到序列工具包。 - facebookresearch/fairseq</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py#L210">torchtune/recipes/full_finetune_distributed.py at main · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建一个账户来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/nightly/recipes/full_finetune_distributed.py#L487">torchtune/recipes/full_finetune_distributed.py at nightly · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建一个账户来为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1289559155685724180)** (39 条消息🔥): 

> - `配置管理关注点`
> - `性能优化想法`
> - `文档改进`
> - `模型实现技术`
> - `内存优化策略` 


- **关于 Config 中重复键的担忧**：讨论了配置文件中出现两次 `fused=True` 的情况，导致 **OmegaConf** 报错提示重复键。
   - *我们可以考虑为配置增加一个性能章节*，将快速选项注释掉以增强可读性。
- **推动清晰的性能指南**：一些成员希望获得全面的性能指南，建议在文档中提供一组 **performance config overrides** 以方便访问。
   - 提出了通过投票获取用户对文档清晰度反馈的想法，表明需要改进。
- **Recipe 文档需要关注**：注意到 **recipe 文档** 滞后的挑战，导致难以随新贡献同步更新。
   - 建议包括请求贡献者协助编写文档，这至关重要但常被忽视。
- **弃用旧的模型代码**：成员们辩论了是否应弃用旧版本中使用的旧模型编码模式，转而采用新方法。
   - 对话强调了确保模型实现标准一致性的重要性。
- **内存优化审查与建议**：建议更新内存优化页面，将**性能和内存优化提示**结合起来，这表明了一种流线化的方法。
   - 想法包括在文档中添加 **sample packing** 和未来的功能（如 **int4 training**）以提高效率。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html">内存优化概述 &mdash; torchtune 主文档</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/blob/3fddc56942846220b39945559f4b5e695873bb43/recipes/configs/llama3/70B_full.yaml#L84">torchtune/recipes/configs/llama3/70B_full.yaml (GitHub)</a>：一个用于 LLM 微调的原生 PyTorch 库。欢迎在 GitHub 上通过创建账号为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1289309247489704006)** (66 条消息🔥🔥): 

> - `CodiumAI A 轮融资`
> - `Liquid Foundation Models 发布`
> - `使用 Gradio 进行 AI 语音交互`
> - `Ultralytics YOLO11 发布`
> - `OpenAI 价格对比` 


- **CodiumAI 更名为 Qodo 并获得 A 轮融资**：QodoAI（原 CodiumAI）宣布获得 4000 万美元 A 轮融资，总融资额达到 5000 万美元。重点是确保代码完整性并为开发者提供 AI 辅助工具。
   - 这笔资金验证了他们的方法，并突显了为该使命做出贡献的开发者和合作伙伴的支持。
- **Liquid Foundation Models 声称基准测试表现出色**：LiquidAI 发布了 LFMs，自称在 MMLU 和其他基准测试上优于现有模型，并指出了竞争对手的低效。团队成员主要来自 MIT，并获得了大量资金。
   - 它们的新架构承诺在 1.3B 模型范围内表现出色，可能挑战该领域的既有领导者。
- **Gradio 5.0 实现实时 AI 语音交互**：LeptonAI 展示了一个集成在 Gradio 5.0 中的创新音频模式 LLM，允许通过极简的代码设置实现无缝实时流式交互。该演示促进了开源协作，并鼓励用户使用自己的 Key 来 Fork 该项目。
   - 向 Gradio 团队致敬，他们提供了强大的更新，使开发者能够高效创建交互式应用。
- **Ultralytics 推出 YOLO11**：Ultralytics 发布了 YOLO11，在之前版本的基础上增强了处理各种计算机视觉任务的能力。此版本在准确性、速度和整体效率方面为开发者带来了提升。
   - 该事件标志着其 YOLO 模型演进中的一个重要里程碑。
- **AI 模型定价见解**：对比了 Google Gemini 与 GPT-4o Mini 在生成机器人回复方面的成本效益，强调了显著的成本降低。这种定价策略可能会影响 AI 驱动的解决方案在社交媒体上自动回复的泛滥程度。
   - 此类讨论表明，业界正在持续评估与大规模 AI 部署相关的运营成本。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/patrickc/status/1840054482455142865?s=46">来自 Patrick Collison (@patrickc) 的推文</a>：直到 2022 年 10 月，ChatGPT 还没有出现，而且普遍很少有 AI 原生产品。领先的早期 AI 投资机构 AI Grant 曾告诫创始人去创造一些产品：https://web.archiv...</li><li><a href="https://x.com/venturetwins/status/1839806109076598837?s=46">来自 Justine Moore (@venturetwins) 的推文</a>：“AI 公司不赚钱” —— 看看 Stripe 的数据吧。顶尖 AI 公司达到 3000 万美元营收的速度比传统 SaaS 同行快 5 倍。</li><li><a href="https://x.com/soumithchintala/status/1840537928369426695">来自 Soumith Chintala (@soumithchintala) 的推文</a>：SB1047 的生命周期：* 初稿由特定利益相关方编写 * 在其他利益相关方私下权衡之前，草案过快地进行了公开传达。* 公开传达开始...</li><li><a href="https://huggingface.co/spaces/akhaliq/dailypapershackernews">Dailypapershackernews - akhaliq 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai">Ultralytics YOLO11 已发布！重新定义 AI 的可能性！作者 Abirami Vina</a>：了解 Ultralytics YOLO11 的所有突破性功能，这是我们最新的 AI 模型，以无与伦比的准确性和效率重新定义了计算机视觉。</li><li><a href="https://x.com/AndrewCurran_/status/1840802455225094147">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：Liquid 今天发布了。他们的小型团队基于一种新架构构建了三个模型，性能极其出色。Joscha Bach 是他们团队的一员，Mikhail Parakhin 在他们的董事会...</li><li><a href="https://aider.chat/2024/09/26/architect.html">分离代码推理与编辑</a>：Architect 模型描述如何解决编码问题，而 Editor 模型将其转化为文件编辑。这种 Architect/Editor 方法产生了 SOTA 基准测试结果。</li><li><a href="https://x.com/AndrewCurran_/status/1839802882327228579">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：《纽约时报》获得了 OpenAI 融资轮的文件。- 8 月有 3.5 亿人使用了 Chat - 匿名登录后用户增长巨大 - 1000 万活跃订阅用户 - 订阅费到年底将上涨 2 美元...</li><li><a href="https://x.com/swyx/status/1840794198913794236">来自 swyx @ DevDay! (@swyx) 的推文</a>：城里来了新的 Transformer 杀手！自从 4 月份与 @Plinz 交谈以来，一直对 @LiquidAI_ 感到兴奋。现在他们终于发布了 LFMs！开火：- 在 MMLU、ARC、GSM8K 上表现优于 1B/3B 模型，相比...</li><li><a href="https://x.com/jiayq/status/1840790511353000437">来自 杨庆娣 (@jiayq) 的推文</a>：构建实时交互曾经很难，因为 Python Web 前端和流式传输不能很好地融合。现在，得益于即将发布的 Gradio 5.0，你只需 250 行代码即可实现。超过...</li><li><a href="https://x.com/levelsio/status/1840410820238270698">来自 @levelsio (@levelsio) 的推文</a>：人们提到 Google 的 Gemini 价格只有 GPT-4o mini 的一半：$0.075 / 1M 输入 token，$0.30 / 1M 输出 token。所以生成 100 万条回复每月只需 $0.37，或者只需 $375/月...</li><li><a href="https://x.com/diegocabezas01/status/1840018687614472246">来自 Diego | AI 🚀 - e/acc (@diegocabezas01) 的推文</a>：Meta AI Llama 3.2 可以编辑图像的选定部分</li><li><a href="https://share.snipd.com/episode/d3b7ee4d-80b3-4889-b2f5-a4c7372a9804">AI 的未来可能看起来很像 Twitter</a>：AI 的未来可能看起来很像 Twitter</li><li><a href="https://x.com/itamar_mar/status/1840755628148687231">来自 Itamar Friedman (@itamar_mar) 的推文</a>：CodiumAI 现在更名为 Qodo！+ 宣布 4000 万美元 A 轮融资 🚀 今天标志着 @QodoAI 的一个重要里程碑。我们宣布了 A 轮融资，使我们的总融资额达到 5000 万美元。这段旅程始于...</li><li><a href="https://x.com/ParikPatelCFA/status/1840060922347638919">来自 Dr. Parik Patel, BA, CFA, ACCA Esq. (@ParikPatelCFA) 的推文</a>：Chat，在 37 亿美元营收的情况下亏损 50 亿美元是正常的吗？</li><li><a href="https://x.com/teortaxestex/status/1840436615908630813?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：Sonnet 仍然拥有即使是 o1 也缺乏的独门秘籍。可能没有其他模型在纯自回归采样且无回溯的 token 中具有如此高的“推理”密度。Anthropic 将...</li><li><a href="https://x.com/that_anokha_boy/status/1840476530536780072">来自 bharat (@that_anokha_boy) 的推文</a>：所以我在他们的 App 上设置了一个代理，你猜怎么着，27 万 Coinbase 工程师在客户端计算用户的使用量。我拦截了他们的 log_tokens API，现在我可以在没有...</li><li><a href="https://www.latent.space/p/mar-jun-2024">AI 寒冬之风</a>：2024 年 3 月至 6 月回顾：人们开始对 AI 之夏产生怀疑。</li>

这就是为什么 AI Engineers 是解决方案。</li><li><a href="https://x.com/karpathy/status/1840112692910272898">来自 Andrej Karpathy (@karpathy) 的推文</a>：NotebookLM 非常强大，值得一试 https://notebooklm.google/ 它在某种程度上重新构想了围绕你上传的一系列来源而组织的 LLMs 工作的 UIUX...</li><li><a href="https://github.com/ultralytics/ultralytics/releases/tag/v8.3.0">Release v8.3.0 - 新 YOLO11 模型发布 (#16539) · ultralytics/ultralytics</a>：🌟 摘要 Ultralytics YOLO11 来了！基于 YOLOv8 基础，由 @Laughing-q 和 @glenn-jocher 在 #16539 中进行研发，YOLO11 在准确性、速度和效率方面提供了前沿的改进...</li><li><a href="https://szymonkaliski.com/projects/replit-agent/">Replit Agent</a>：面向人类和 LLMs 的 IDE</li><li><a href="https://www.ft.com/content/a9a192e3-bfbc-461e-a4f3-112e63d0bb33">订阅以阅读</a>：未找到描述</li><li><a href="https://github.com/mediar-ai/screenpipe">GitHub - mediar-ai/screenpipe：24/7 本地 AI 屏幕和麦克风录制。构建具有完整上下文的 AI 应用。支持 Ollama。Rewind.ai 的替代方案。开源。安全。你拥有自己的数据。Rust。</a>：24/7 本地 AI 屏幕和麦克风录制。构建具有完整上下文的 AI 应用。支持 Ollama。Rewind.ai 的替代方案。开源。安全。你拥有自己的数据。Rust。 - mediar-ai/screenpipe
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1289336254734929930)** (6 条消息): 

> - `新播客剧集`
> - `YouTube 互动`
> - `节目中的 AI Researchers` 


- **最新播客邀请重磅嘉宾**：新的播客剧集邀请了来自 OpenAI 的 **Shunyu Yao** 和来自 LangChain 的 **Harrison Chase**，重点讨论 AI agents 的核心话题。
   - 鼓励听众在 [Apple Podcasts](https://podcasts.apple.com/us/podcast/latent-space-the-ai-engineer-podcast-practitioners/id1674008350) 和 [YouTube](https://youtube.com/@latentspacetv?si=ZwBcMikMlltS1vwW) 上为节目评分，以帮助其实现多元化发展。
- **听众对互动充满热情**：听众正积极参与播客互动，其中一位确认他们已经“点赞并订阅”，并开启了铃铛通知以获取更新。
   - 另一位听众幽默地表示，他们取消订阅只是为了能订阅两次，展示了他们对节目的支持。
- **请求更多 Researchers 参加节目**：听众非常喜欢这些内容，并表达了希望更多 **researchers** 加入未来剧集的愿望。
   - 一位用户评论道，*“带更多 researchers 来”*，表明了对未来播客中更深入讨论的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://podcasts.apple.com/us/podcast/latent-space-the-ai-engineer-podcast-practitioners/id1674008350">Latent Space: The AI Engineer Podcast — 从业者谈论 LLMs, CodeGen, Agents, Multimodality, AI UX, GPU Infra 等</a>：在 Apple Podcasts 上收听 Alessio + swyx 的 Latent Space: The AI Engineer Podcast — 从业者谈论 LLMs, CodeGen, Agents, Multimodality, AI UX, GPU Infra 等。</li><li><a href="https://youtube.com/@latentspacetv?si=ZwBcMikMlltS1vwW">Latent Space</a>：超过 50,000 名 AI Engineers 聚集在一起讨论模型、工具和想法的首选之地。今天发布的突发新闻，你明天就能在工作中使用！完整的节目笔记和通讯请访问 https://latent.space</li><li><a href="https://x.com/FanaHOVA/status/1839741529331773813">来自 Alessio Fanelli (@FanaHOVA) 的推文</a>：我们如何让 AI agents 思考和行动？🤖 今天与 @ShunyuYao12（以及特别联合主持人 @hwchase17！）的剧集可能是我们目前为止最好的 agents 剧集：- ReAct 的起源以及它是如何启发 @L... 的。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1289315495488786433)** (42 条消息🔥): 

> - `AI Engineering 面试`
> - `屏幕共享问题`
> - `本地模型实验`
> - `Braintrust 评估平台` 


- **Frikster 的面试好消息**：Frikster 分享了关于面试可能转型为 **AI Engineering** 角色的兴奋之情，表达了对这次机会的整体喜悦。
   - 随后出现了一些有趣的反应，认为这种转型类似于“为其 Prompt 知识点亮了正确的权重”。
- **屏幕共享故障排除**：多位成员报告了查看 **屏幕共享** 时的问题，并提出了各种故障排除建议，如重新加载或切换平台。
   - 一些人发现退出并重新加入通话解决了问题；然而，其他人仍然遇到 **黑屏**。
- **本地模型的潜力**：Rajwant 询问为特定任务创建 **本地模型** 是否有益，引发了关于此类模型有效性的讨论。
   - Kbal 询问成员是否对其他模型进行过类似的实验，特别是与 **O1** 的对比。
- **Braintrust 与其他评估平台**：Youngphlo 询问了关于 **Braintrust** 与其他语言模型评估平台相比的看法。
   - Vodros 承认不熟悉 Braintrust，同时提出了关于其是否支持 **JSON mode** 的疑问。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1289318545964138569)** (7 条消息): 

> - `FinanceAgentToolSpec`
> - `从 Workflow 中流式传输事件`
> - `自动化财务报告生成`
> - `带有 Confluence 的 Multi-Agent Slackbot`
> - `LlamaParse Premium` 


- **利用 FinanceAgentToolSpec 获取公开财务数据**：LlamaHub 上的 [FinanceAgentToolSpec](https://t.co/7bsEm4Er1m) 软件包使 Agent 能够查询各种公开财务数据源，如 **Polygon**、**Finnhub** 和 **Seeking Alpha**。
   - Hanane 的一篇详细文章解释了该工具在财务分析中的效用及其实际应用。
- **流式传输事件的全栈 Demo**：一个新的 [全栈应用](https://t.co/HOajPyiqQb) 展示了流式传输事件的 Workflow，在报告撰写场景中具有 Human In The Loop 功能。
   - 该应用展示了如何研究一个主题并进行全面展示，增强了用户交互。
- **Workflow 代码的 YouTube 教程**：现在有一个 [YouTube 视频](https://t.co/Nn5NVZopPz)，开发者在其中演示了前面讨论的全栈 Demo 的编码过程。
   - 该视频为那些希望实现类似系统的人提供了教育资源。
- **通过 RAG Workflows 自动生成报告**：一份新的研究指南说明了如何使用 **Agentic Workflows** 将 10K 报告中的非结构化上下文整合到自动化财务报告生成中。
   - 这种高级应用超越了简单的 Chatbot 响应，从多个数据源综合生成全面的报告。
- **构建带有 Confluence 的 Agentic Slackbot**：一份全面的教程详细介绍了如何构建一个 **Multi-Agent Slackbot**，该机器人使用 **AWS 服务** 与 Confluence 文档进行交互。
   - 该计划强调了通过将结构化内容集成到聊天界面来提高组织效率的潜力。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1289429783486402591)** (105 条消息🔥🔥): 

> - `Ollama 并发`
> - `LlamaIndex 项目设置`
> - `RAG 流水线评估`
> - `Node 元数据处理`
> - `RAG Benchmark 中的 Oracle 检索` 


- **Ollama 的并发特性**：一位用户询问如何利用 Ollama 的并发功能，得到的回复是该功能默认已启用。
   - 提供了一个指向 [Ollama 并发处理](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-does-ollama-handle-concurrent-requests) 的有用链接以供进一步参考。
- **LlamaIndex 项目流水线指南**：一位成员在 LlamaIndex 项目中寻求处理复杂 PDF 的建议，被推荐使用 [Llamaparse](https://github.com/run-llama/llama_parse) 以获得最佳效果。
   - 关于各种文档处理方法的讨论为有效提取相关数据提供了进一步的见解。
- **RAG 流水线评估中的挑战**：一位用户报告了由于导入错误导致无法使用 trulens 评估其 RAG 流水线的问题，引发了查看文档和可用指标的建议。
   - 深入讨论了在评估设置中检索 Node ID 以作为 ground truth 的澄清，强调了构建可靠评估数据集的必要性。
- **在 LlamaIndex 中编辑 Node 元数据**：用户讨论了为 LlamaIndex 中的每个数据块（chunk）编辑元数据的能力，确认通过代码片段添加 URL 等详细信息是可行的。
   - 提供了关于有效操作 Node 元数据以增强数据检索和索引过程的指导。
- **关于 Oracle 检索和新 Benchmark 的见解**：一位成员分享了来自 Google 的新 RAG benchmark 数据集的信息，该数据集引入了 Oracle 检索的概念。
   - 指出 Oracle 检索依赖于 ground-truth 注释，代表的是性能上限衡量标准，而非实际的检索方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/evaluating/usage_pattern_retrieval/">使用模式 (检索) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_vs_recursive_retriever/">结构化检索方法比较 (自动检索 vs. 递归检索) - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/retrieval/retriever_eval/">检索评估 - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/faq.md#how-does-ollama-handle-concurrent-requests">ollama/docs/faq.md at main · ollama/ollama</a>: 快速运行 Llama 3.2, Mistral, Gemma 2 以及其他大语言模型。 - ollama/ollama</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/retrieval/retriever_eval/.">检索评估 - LlamaIndex</a>: 未找到描述</li><li><a href="https://go.microsoft.com/fwlink/?linkid=2198766",">Go with the floe</a>: 在午夜阳光下最完美的活动是什么 </li><li><a href="https://github.com/run-llama/llama_index/discussions/15117">如何在自定义查询流水线构建的聊天引擎中适配多步查询分解 · run-llama/llama_index · Discussion #15117</a>: 我该如何将类似这样的多步查询分解集成到我的自定义聊天引擎中：from llama_index.core.query_engine import MultiStepQueryEngine query_engine = index.as_query_engine(llm=gpt4) q...</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: 为最佳 RAG 解析文件</a>: 为最佳 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 做出贡献。</li><li><a href="https://github.com/run-llama/llama_index/blob/a620a2661faabb49ba2f257bff7ae2ac04d0c12b/llama-index-core/llama_index/core/evaluation/retrieval/metrics.py#L457">llama_index/llama-index-core/llama_index/core/evaluation/retrieval/metrics.py at a620a2661faabb49ba2f257bff7ae2ac04d0c12b · run-llama/llama_index</a>: LlamaIndex 是适用于 LLM 应用的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1289845610958032908)** (1 条消息): 

> - `LLM Reasoning`
> - `Different Types of Reasoning` 


- **定义 LLM Reasoning 问题**：在深入研究 LLM 推理之前，明确我们要解决的推理问题类型至关重要。
   - 分享的一篇文章详细介绍了[各种推理类型](https://www.linkedin.com/posts/subham-kundu-2746b515b_llm-reasoning-is-becoming-a-very-important-activity-7246050519289413632-3LPo?utm_source=share&utm_medium=member_desktop)并评估了 LLM 在这些挑战中的表现。
- **推理分类的重要性**：识别具体的推理问题对于引导 LLM 的有效性至关重要。
   - 文章强调，不同的推理挑战需要独特的方法和评估手段。


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1289324296069713984)** (21 条消息🔥): 

> - `Channel posting guidelines`
> - `Humanoid Robots 2024 YouTube video`
> - `Innovations in UI/UX for LLMs`
> - `Robotics development challenges`
> - `Podcasting as a UI/UX interaction` 


- **频道发布指南澄清**：一名成员询问了正确的发布频道，随后澄清了 https://link.to/channel 是可以接受的，尽管它与 Cohere 没有直接关系。另一名成员警告说该频道不是招聘门户，并提醒大家保持适当的讨论。
   - *Hello!* 和 *yahalloo!* 的交流标志着成员加入时频道内的欢迎氛围。
- **2024 年人形机器人最佳综述**：一名成员分享了一个[标题为](https://youtu.be/PyrDh6RQdYY?si=RDA-9SFzdcZbAsmP) 'Every Humanoid Robot 2024' 的 YouTube 视频，声称这是互联网上对人形机器人最好的综述。其中包含了一个机器人及其制造商的完整列表链接。
   - 随后，对话转向讨论机器人领域当前面临的问题，如计算能力、电池成本和更高的人力成本，引发了一场头脑风暴。
- **LLM 的 UI/UX 创新**：一名成员强调了人机交互中 UI/UX 创新的必要性，并分享了关于 NotebookLM 作为从任何内容创建播客的强大工具的见解。他们提供了各种音频转换的链接，展示了播客作为 LLM 界面格式的潜力。
   - 他们指出，虽然 LLM 发展迅速，但 UI/UX 往往滞后，并认为播客可以绕过 AI 交互中传统的用户参与障碍。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/karpathy/status/1840112692910272898">来自 Andrej Karpathy (@karpathy) 的推文</a>: NotebookLM 非常强大，值得一试 https://notebooklm.google/。它是对围绕你上传的一系列源文件组织的 LLM UI/UX 的一种重新构想...</li><li><a href="https://docs.cohere.com/">Cohere Documentation — Cohere</a>: Cohere 的 API 文档帮助开发人员轻松地将自然语言处理和生成集成到他们的产品中。</li><li><a href="https://youtu.be/PyrDh6RQdYY?si=RDA-9SFzdcZbAsmP">Every Humanoid Robot 2024</a>: 互联网上所有人形机器人的最佳综述。由 Automate Construction 为您呈现。机器人列表及其制造商：https://automateconstruction....
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1289318982939316224)** (36 条消息🔥): 

> - `RAG 格式化查询`
> - `Cohere Startup Program`
> - `API 计费问题`
> - `Multimodal Captioning`
> - `输入 Token 数量问题` 


- **模型提示词的 RAG 格式化**：用户讨论了如何为提交给 LLM 的提示词中的 RAG 包含内容格式化**指令头 (instructional headers)**，指出需要明确拼接方式。
   - 一位成员提到，以模型预期的格式包含支持信息以及标题的终止方法非常重要。
- **探索 Cohere Startup Program**：一位用户询问了使用 Cohere 的创业团队的折扣信息，强调了与 **Gemini** 等竞争对手相比的高昂费用。
   - 另一位用户建议申请 **Cohere Startup Program**，该计划提供折扣，并指出申请处理可能需要一些时间。
- **澄清 API 计费流程**：出现了关于 **Cohere** 如何对 API 使用进行计费的查询，并确认**按月计费**。
   - 一位用户提到在他们的账户中找不到发票，引发了关于计费流程的进一步讨论。
- **对 Multimodal Captioning 的兴趣**：一位用户询问是否有人在研究 **multimodal captioning**，并邀请交流想法和经验。
   - 另一位参与者表现出热情，鼓励讨论与 multimodal captioning 相关的项目。
- **输入 Token 数量差异**：一位用户对他们的**输入 Token 数量**准确性提出担忧，声称每日使用量被低报了。
   - 他们还讨论了在申请折扣时面临的挑战，因为他们不是以公司身份运营，而是作为一个创业团队。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/startup-program-application">Startup Program Application</a>: 非常感谢您对 Cohere Startup Program 的关注！我们最初正向选定的客户群体推出该计划，并希望进一步了解您的业务...</li><li><a href="https://cohere.com/startup-program">Startup Program </a>: Cohere Startup Program 为符合条件的 B 轮及更早期的创业公司提供支持、API 费率折扣和宣传的独特机会。</li><li><a href="https://docs.cohere.com/reference/chat-stream-v2">Chat with Streaming — Cohere</a>: 根据提供的对话生成来自模型的响应。要了解有关 Chat API 功能的更多信息，请参阅我们的 Text Generation 指南。请遵循 Migration Guide 以获取...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1289302210202570874)** (23 条消息🔥): 

> - `Fine-tuning Models`
> - `Chunking Data for Improved Output`
> - `System Message and API Migration Issues`
> - `Documentation Consistency`
> - `V1 to V2 Chat API Transition` 


- **Fine-tuning 模型在抽认卡（flash card）生成方面面临挑战**：一位成员就如何针对笔记和幻灯片进行模型 Fine-tuning 以获得更好的抽认卡生成效果寻求建议，并指出了输出质量方面的问题。他们正在考虑是否可以在不进行 Fine-tuning 的情况下改进非结构化数据。
   - 另一位成员建议使用机器学习流水线的最佳实践来增强任务，并强调 **数据 Chunking** 可以显著提升模型的输出效果。
- **Chunking 显著提升模型性能**：成员们讨论了 **数据 Chunking** 的有效性，特别是针对 PDF 幻灯片，以增强模型对相关内容的理解。他们还提到探索使用 **rerankers** 等工具来优化大规模数据集的结果。
   - 对话强调了一个原则：结构良好的输入可以带来 **更好的定性输出**，解决了 AI 任务中数据准备的重要性。
- **API 迁移对话揭示了关键挑战**：随着用户从 Chat API 的 v1 过渡到 v2，有人提出了同时使用 **system messages 和 document 参数** 会影响功能的问题。一位成员遇到了困难，并从他人处得知这是一个已知 bug，随后已得到解决。
   - 另一位用户确认当前的 API 结构仍然支持旧版本，确保了迁移者的连续性，同时也强调了 **系统性更新的必要性**。
- **呼吁改进文档**：一位成员注意到 API 文档中的不一致之处，特别是关于参数的惩罚范围（penalty ranges），呼吁更统一地展示细节。他们建议为记录最小值和最大值制定更清晰的标准，以提高用户的清晰度。
   - 围绕 API 错误处理的讨论强调了为了获得最佳用户体验，一致且易于理解的文档的重要性。
- **V1 到 V2 的过渡总体表现积极**：成员们对 v1 Chat API 在迁移期间仍能正常工作感到欣慰，并强调几乎没有理由退回到旧版本。对话显示，尽管最初有些小波折，但大家对 v2 提供的改进持乐观态度。
   - 社区保持活跃，在适应 v2 API 新实现的功能时不断交流见解和解决方案。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1289328195514142858)** (2 条消息): 

> - `Cultural Multilingual LMM Benchmark`
> - `Volunteer Translators`
> - `CVPR'2025 Paper Co-Authorship` 


- **MBZUAI 发布文化多语言 LMM 基准测试**：MBZUAI 正在开发一个针对 **100 种语言** 的 **Cultural Multilingual LMM Benchmark**，创建一个包含当地语言翻译的多模态数据集。
   - 他们正在招募母语翻译志愿者来帮助纠正错误，并承诺在任务完成后邀请其共同署名论文。
- **招募各语种志愿者翻译员**：需要协助的语言包括 **印度**、**南亚**、**非洲** 和 **欧洲** 语言，并为潜在志愿者提供了一份广泛的列表。
   - 在回应志愿者招募时提到 *“这不是一个招聘门户……所以我们不按那种方式运作”*，澄清了咨询的性质。
- **翻译员社交邀请**：感兴趣的人士可以通过 **LinkedIn** 与项目负责人联系，获取更多信息并展示其语言技能。
   - 他们鼓励就志愿服务意向发送私信。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1289325996184703069)** (36 条消息🔥): 

> - `OpenAI 员工流失`
> - `AI 监管`
> - `关于 AI 数据集的法律裁决`
> - `投资讨论`
> - `公众对 AI 法案的反应` 


- **OpenAI 因薪酬要求引发的人才外流**：OpenAI 的核心研究人员正在寻求更高的薪酬，随着公司估值上升，已通过出售利润单位套现 **12 亿美元**。Safe Superintelligence 等竞争对手积极挖掘 OpenAI 人才，加剧了领导层的更迭。
   - 员工因资金问题威胁离职，而新任 CFO **Sarah Friar** 正处于这些谈判的中心。
- **加州州长否决 AI 安全法案 SB 1047**：州长 **Gavin Newsom** 否决了旨在监管 AI 公司的 SB 1047 法案，称其并非保护公众的最佳方式。批评者认为此次否决是监管的倒退，而支持者则主张应基于明确的能力而非模糊的预测进行监管。
   - 参议员 **Scott Wiener** 对州长在否决前未提供反馈表示失望，并强调加州错失了在技术监管方面领先的机会。
- **LAION 在版权案件中获得法律胜利**：LAION 在德国 **Kneschke v LAION** 案件中成功抵御了版权侵权指控，一名摄影师指控其滥用其图像。法院裁定 LAION 仅链接到图像，而非自行托管任何图像。
   - 这一裁决对 AI 数据集的使用案例具有重要意义，版权讨论将继续塑造 AI 领域。
- **OpenAI 对投资者关系的担忧**：据 **WSJ** 报道，OpenAI 已不再与 **Apple** 就投资进行讨论。这一转变标志着 OpenAI 的使命与满足投资者需求之间存在更广泛的紧张关系。
   - 随着 OpenAI 接近一个潜在的转型财务节点，与主要投资者的关系对其未来走向至关重要。
- **公众反应引发关于 AI 的讨论**：对被否决的 AI 安全法案的反应褒贬不一，一些人认为否决的理由充分，强调了监管的清晰度。许多人预计立法努力将在明年重新浮出水面。
   - 社区讨论突显了关于监管应如何反映实际技术能力而非投机性未来场景的不同观点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/9/27/24255177/openai-safety-mira-murati-quit-sam-altman-cofounders-exodus">OpenAI 曾是一个研究实验室——现在它只是另一家科技公司</a>：OpenAI 可能很快会成为一家营利性公司，其制衡机制比以前更少——这正是它成立之初想要避免的结构。</li><li><a href="https://x.com/unusual_whales/status/1839837869399257373?s=46">来自 unusual_whales (@unusual_whales) 的推文</a>：快讯：据 WSJ 报道，Apple $AAPL 据称已不再参与 OpenAI 的投资讨论或董事会讨论</li><li><a href="https://sfstandard.com/2024/09/29/gavin-newsom-vetoes-controversial-ai-safety-bill/">Newsom 否决了备受争议的 AI 安全法案 SB 1047</a>：该法案已成为硅谷的焦点，Elon Musk 等科技人物支持该措施，而其他人则表示这将威胁到处于早期阶段的蓬勃发展的 AI 产业。</li><li><a href="https://www.technollama.co.uk/laion-wins-copyright-infringement-lawsuit-in-german-court">LAION 在德国法院赢得版权侵权诉讼</a>：版权 AI 爱好者一直热切期待德国 Kneschke v LAION 案件的裁决（之前关于该案件的博客文章见此），昨天我们得到了裁决（裁决文本为德语...）
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1289325582106099722)** (16 messages🔥): 

> - `PearAI 争议`
> - `Yann LeCun 谈研究标准`
> - `OpenAI 的透明度辩论`
> - `同行评审批判`
> - `研究博客的影响` 


- **PearAI 被指控窃取代码**：**PearAI** 据称从 [Continue.dev](http://Continue.dev) 窃取了代码，并在没有适当致谢的情况下对其进行了重新品牌化，引发了愤怒，并引发了 YC 等投资者对其问责的呼声。
   - *对于那些不知道的人：PearAI 从开源社区窃取了代码……* 这引发了关于初创公司融资的伦理担忧。
- **LeCun 抨击博客文章标准**：Yann LeCun 批评了依赖博客文章来确立研究有效性的做法，而忽视了同行评审论文的严格标准，并强调技术研究不能被新闻稿所取代。
   - *“你可以自欺欺人地认为这是自切片面包以来最伟大的发明……”* 这句话突显了产品压力与研究诚信之间的紧张关系。
- **关于 OpenAI 透明度的辩论**：批评者质疑 OpenAI 的透明度，指出引用博客并不等同于对研究结果的实质性沟通，一位成员表示 **新闻稿并不代表什么**。
   - 在辩论中，一些 OpenAI 员工断言他们在研究沟通方面确实是开放的。
- **对同行评审的怀疑**：一些成员对 **同行评审** 的有效性表示怀疑，认为许多已发表的研究可能水平低下，却仍被视为有效。
   - 对话揭示了对研究发表过程中缺乏问责制的挫败感。
- **OpenAI 研究博客的影响**：关于 **研究博客** 的讨论质疑分享 CoTs 等见解是否足以告知社区，一些人建议这些信息可能是经过精心挑选的。
   - 成员们对于 [openai.com](https://openai.com/index/learning-to-reason-with-llms/) 是否充分解决了社区对透明度和彻底性的担忧持有复杂态度。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/polynoamial/status/1840416885189271890">Noam Brown (@polynoamial) 的推文</a>：@ylecun @thomaspower @OpenAI 此外，我们在研究博客文章 https://openai.com/index/learning-to-reason-with-llms/ 中说了很多，包括分享 CoTs，我认为这些信息非常丰富...</li><li><a href="https://x.com/doiftrue/status/1840414633573646806?s=46">Jakob Finch (@doiftrue) 的推文</a>：@candyflipline @iamgingertrash 对于那些不知道的人：PearAI 从 http://Continue.dev 窃取了代码，并将其作为他们正在“构建”并刚刚获得融资的初创公司：https://...</li><li><a href="https://x.com/ylecun/status/1840422017654210604">Yann LeCun (@ylecun) 的推文</a>：我很抱歉 Noam，但博客文章远未达到可重复性、方法论、对先前工作的认可以及与 SOTA 进行公平比较的标准，而这些是一个技术...</li><li><a href="https://x.com/polynoamial/status/1840441849011744809">Noam Brown (@polynoamial) 的推文</a>：@ylecun @thomaspower @OpenAI 我认为恰恰相反。坦率地说，很多发表的研究都是废话。作者只需要迷惑 3 名审稿人和一名 AC。当发布一个数百万人...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1289312314008277114)** (13 条消息🔥): 

> - `iPhone IAP 订阅`
> - `Apple App Store 管理`
> - `Twitter 安全问题`
> - `与 John Schulman 的会面`
> - `Twitter 上的社区互动` 


- **获得 iPhone IAP 订阅权限**：一位 Substack 畅销作者宣布获得了 **iPhone In-App Purchase (IAP) 订阅**权限，预示着移动端变现的潜在增长机会。
   - 这一权限的获得，让人们得以一窥这些系统的实现及其管理方式。
- **揭秘 Apple App Store 的噩梦**：分享了关于 **Apple App Store 管理挑战**的见解，强调了其环境的混乱。
   - 讨论突出了开发者在该生态系统中面临的复杂性和挫败感。
- **Twitter 安全漏洞警报**：一条令人担忧的推文指出一个知名 Twitter 账号被盗，强调这可能发生在技术领域的任何人身上。
   - 讨论指出这一问题依然存在，并呼吁提高用户的安全意识。
- **关于 RLHF 见解的 John Schulman 会面**：宣布即将与 **John Schulman** 会面，就 **Reinforcement Learning from Human Feedback (RLHF)** 工作寻求建议。
   - 这次互动反映了 AI 社区中的协作与导师指导机会。
- **对 Twitter 维护工作的担忧**：一位用户对 Twitter 的安全承诺表示怀疑，指出该平台只有 **三名工程师** 在处理相关问题。
   - 评论认为，团队的效率受到干扰和资源匮乏的影响，从而影响了整体安全性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/gm8xx8/status/1840304990134411561">来自 𝚐𝚖𝟾𝚡𝚡𝟾 (@gm8xx8) 的推文</a>：🚨 @DrJimFan 被盗号了</li><li><a href="https://x.com/sammcallister/status/1840800944264478772?s=46">来自 sam mcallister (@sammcallister) 的推文</a>：🥹 @karpathy
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1290059810708131883)** (3 条消息): 

> - `AI Memes`
> - `用户反应` 


- **用户对“残暴”迷因的反应**：针对一个 AI 生成的迷因，一位用户用 *You make this??? Brutal* 表达了惊讶和有趣。
   - 当被问及是否创作了该迷因时，另一位用户幽默地称 *I wish lol*，并澄清它是源自一个随机的 AI Memes 账号。
- **关于迷因作者身份的讨论**：展开了一场关于迷因来源的对话，一位用户急切地询问是否是另一位用户创作的。
   - 当回复者提到这只是来自一个随机的 AI Memes 账号时，气氛很快变成了欢笑。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 条消息): 

SnailBot 新闻：<@&1216534966205284433>
  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1289468237620183091)** (36 条消息🔥): 

> - `Course Material Access`（课程资料获取）
> - `Multi-Agent Systems Discussion`（Multi-Agent 系统讨论）
> - `NotebookLM Inquiry`（NotebookLM 咨询）
> - `Training Schedule Inquiry`（培训日程咨询）
> - `Research Proposal Discussion`（研究提案讨论） 


- **获取课程资料**：学生们在填写注册表后，正在询问如何获取课程视频和资料。
   - 课程资料（包括作业和课程录像）可以在 [课程网站](https://llmagents-learning.org/f24) 上找到，所有作业的截止日期为 12 月 12 日。
- **Multi-Agent 与 Single-Agent 系统的辩论**：对话集中在针对不同项目，Multi-Agent 系统对比 Single-Agent 实现的有效性和必要性。
   - 有人指出，Multi-Agent 系统可以减轻幻觉（hallucinations）并简化上下文管理，有助于 LLMs 提供准确的回答。
- **NotebookLM 的功能**：有关于 NotebookLM 是否作为 Agent 应用运行的咨询。
   - 它被描述为一个 RAG Agent，可以总结文本并生成音频，用户对其在多步流程方面的技术实现提出了疑问。
- **培训日程确认**：学生们正在寻求有关课程培训课程何时开始的信息。
   - 一位成员分享说，他们被告知所有三个 labs 都将在 10 月 1 日发布，尽管这并非正式公告。
- **关于 Super-Alignment 的研究提案**：一个拟议的研究项目旨在探索 Multi-Agent 系统中的伦理问题，强调使用 AutoGen 等框架。
   - 讨论强调了在没有专用框架的情况下实施此类研究的挑战，并指出了在模拟能力方面的潜在局限性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/f24">Large Language Model Agents</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/">无标题</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/12XgfYC2_U4gFEN732GPpv5Axh5IAUbaT2t_UKeRGFb0/edit?usp=drivesdk">Research Proposal: Exploring Super-Alignment through Relative Ethics in Multi-Agent Systems using AutoGen</a>: 研究提案：通过 AutoGen 在 Multi-Agent 系统中使用相对伦理探索 Super-Alignment。Eric Moore - 2024/9/28。摘要：随着先进人工智能和强大能力的出现...</li><li><a href="https://www.reddit.com/r/LangChain/s/CHRT9AehcV">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.coursera.org/projects/ai-agentic-design-patterns-with-autogen?">AI Agentic Design Patterns with AutoGen</a>: 在 2 小时内完成此引导项目。在“使用 AutoGen 的 AI Agentic 设计模式”中，你将学习如何构建和定制 Multi-Agent 系统...</li><li><a href="https://www.reddit.com/r/LangChain/s/CHRT9Aeh">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/)** (1 条消息): 

metakingkal: AutoGen 网站上有一个关于如何构建 Agent 来下国际象棋的示例。
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1289499444399702078)** (27 messages🔥): 

> - `Cloud Storage Costs` (云存储成本)
> - `Modal Pricing Structure` (Modal 定价结构)
> - `Tinygrad Matcher Optimization` (Tinygrad Matcher 优化)
> - `Testing Strategies for Optimizers` (优化器测试策略)
> - `Bounty Payment Methods` (Bounty 支付方式)


- **云存储成本与主流供应商相比具有竞争力**：George 提到 **storage 和 egress 成本**将小于或等于主流云供应商，强调了成本考量。
   - 他进一步解释说，对使用情况的预期可能会显著改变感知的成本。
- **Modal 的付费模式引发辩论**：Modal 独特的定价模式（按秒计费计算资源）引起了关注，被吹捧为**比传统的按小时计费更便宜**。
   - 成员们质疑这种模式的可持续性，以及它如何与 AI 初创公司环境中持续的使用模式相匹配。
- **使用状态机改进 Tinygrad 的 Matcher**：一位成员建议实现一个 **matcher 状态机**可以提高性能，使其趋向于类 C 的效率。
   - George 热情地支持这种方法，表示这可以实现预期的性能提升。
- **需要全面的回归测试**：有人担心优化器缺乏 **regression test suite**（回归测试套件），这可能导致代码更改后出现未被察觉的问题。
   - 成员们讨论了通过序列化来检查优化模式的想法，但意识到这可能不够吸引人。
- **讨论 Bounty 支付选项**：一位用户询问 Bounty 是否可以通过 **Payoneer** 而不是 PayPal 支付，尽管 George 指出了他们问题文档中现有的协议。
   - 这反映了社区内关于支付系统的持续对话。



**提到的链接**：<a href="https://modal.com/pricing">Plan Pricing</a>：简单、透明的定价，根据你使用的计算量进行扩展。

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1289915687350960159)** (4 messages): 

> - `SOTA GPU for Bounties` (用于 Bounty 的 SOTA GPU)
> - `Renting GPUs Online` (在线租用 GPU)
> - `TF32 Tensor Core Support` (TF32 Tensor Core 支持)
> - `Learning Before Tackling Bounties` (在处理 Bounty 前先学习)
> - `Small PR Contributions` (小型 PR 贡献)


- **SOTA GPU 并非 Bounty 的强制要求**：一位成员建议，虽然 **SOTA GPU** 会有帮助，但使用普通的 GPU 也可以应付，尤其是对于某些任务。
   - 像 **tinygrad 中 100+ TFLOPS 的 matmul** 这样的任务可能需要特定的硬件（如 **7900XTX**），而其他任务则不需要。
- **租用 GPU 完成任务**：有人提到，如果有必要，可以**在网上廉价租用 GPU**，这为那些没有高端硬件的人提供了灵活性。
   - 这种具有成本效益的方法允许在不需要永久高性能配置的情况下参与 Bounty。
- **理解 TF32 Tensor Core 支持**：一位用户询问了“TF32 Tensor Core 支持”，表现出对性能能力的兴趣。
   - 建议在尝试 Bounty 之前彻底掌握这些概念，以确保成功。
- **处理 Bounty 前准备工作的重要性**：强烈建议在尝试 Bounty 之前花时间学习代码库，因为这会简化过程。
   - 熟悉 **open PRs** 和现有问题（issues）可以帮助避免冲突并简化入门过程。
- **从小型 PR 贡献开始**：建议在参与更重要的 Bounty 任务之前，先从一个**小型 PR** 开始。
   - 关注 GitHub issues 和 Discord 频道可以发现需要处理的任务，并提供贡献的途径。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1290026403533819956)** (14 messages🔥): 

> - `Llama 3.2 1b tuning`
> - `California AI training bill`
> - `Lightweight chat models`
> - `Liquid AI`
> - `Sample packing effects` 


- **对 Llama 3.2 1b 微调的担忧**：一位用户报告了在微调 **Llama 3.2 1b** 时遇到的问题，即使使用了 qlora 和 4bit 加载等设置，显存（VRAM）占用仍高达 **24GB**。
   - 讨论中提出了关于增加序列长度与批次大小（batch size）相比所产生影响的问题，特别是在启用了 **sample packing** 的情况下。
- **加州颁布 AI 训练披露法**：一项新的**加州法律**强制要求披露在该州使用的任何 AI 模型的训练来源，小型模型或非营利组织也不例外。
   - 这一法律引发了关于潜在规避方案的讨论，正如多位成员建议的那样，可以使用轻量级聊天模型来创建符合法律要求的“受启发”数据集。
- **进军轻量级聊天模型**：成员们讨论了微调轻量级聊天模型的想法，以转换网页爬取的数据集，同时保持法律意义上的转换标准。
   - 一位成员指出，由于**原始网页爬取数据（raw webcrawl data）**通常很混乱，LLM 可以协助进行清理，作为有益的下一步。
- **对 Liquid AI 的关注**：一个名为 **Liquid AI** 的新基础模型引起了讨论成员的兴趣。
   - 考虑到最近的立法变化，一些人对这个新模型的含义和特性表示好奇。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1289324517679825021)** (3 messages): 

> - `LoRA+ implementation`
> - `Learning Rate Default Values`
> - `PEFT's Implementation` 


- **关于默认值使用的疑问**：一位成员询问他们应该为某个参数使用默认值，还是使用与 **learning_rate** 相同的值。
   - 他们注意到 [LoRA+ 论文](https://link.to.lorapluspaper) 将 **1e-6** 设置为主学习率，这可能解释了 **loraplus_lr_embedding** 的默认值设置。
- **关于论文默认值的假设**：另一位成员同意这一假设，即默认值源自 **LoRA+ 论文**，因为它使用了 **1e-6**。
   - *由于 Pydantic 默认值为 None*，向 **PEFT 的实现** 迁移时需要进行细微调整。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1289682244180381797)** (12 messages🔥): 

> - `Axolotl dataset configuration`
> - `Selecting random dataset samples`
> - `Hugging Face datasets handling` 


- **在 Axolotl 中使用 20% 的数据集**：在 Axolotl 中，你可以通过利用 `datasets` 配置下的 `split` 选项来指定使用数据集的一部分，从而允许你定义自定义切分。
   - 例如，你可以设置配置以使用数据集的前 20% 进行训练，并可对验证集和测试集切分进行调整。
- **随机选择数据子集**：Axolotl 配置中没有直接使用随机 20% 数据集的选项；这需要在数据集加载或预处理阶段完成。
   - 利用像 Hugging Face 的 `datasets` 这样的库，你可以在将处理后的数据集传递给 Axolotl 之前随机采样 20%。
- **引用 Llama 3 示例**：一位用户建议查看 Llama 3 示例，以获取有关 Axolotl 数据集处理的潜在相关配置。
   - 这表明现有示例中可能列出了处理随机样本的隐式方法或实践。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=76ea009a-88eb-4421-bba0-01c6c3c35516)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=6d0af173-99ff-4f56-b361-5bbc2256f689)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快地理解代码。
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1289648270834008145)** (2 messages): 

> - `Pydantic model generator`
> - `Groq integration`
> - `GitHub Actions`
> - `Typed Predictors`
> - `DSPyGen` 


- **免费 Pydantic Model Generator 现场编程**: 一场正在进行的演示环节，展示了如何利用 [Groq](https://groq.com) 和 **GitHub Actions** 创建一个**免费的 Pydantic model generator**。
   - 该项目旨在增强 **DSPyGen** 内部的模型生成能力，从而支持更多的 **Typed Predictors** 并简化流程。
- **会议的 Loom 视频**: 一位成员分享了一个 [Loom 视频](https://www.loom.com/share/783ed4d80720492da23f39d2678de27f)，详细记录了这次现场编程过程。
   - 该视频深入介绍了演示中所使用的编程方法和工具，对参与者和观察者都非常有价值。



**提到的链接**: <a href="https://www.loom.com/share/783ed4d80720492da23f39d2678de27f">I am still King of Typed Output in DSPy</a>：在此视频中，我演示了在 **Pydantic** 中创建类型预测器的过程，展示了生成结构化文本的过程和结果。我逐步讲解了创建一个类型预测器生成器的步骤...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1289319529431699546)** (17 messages🔥): 

> - `DSPy 2.5 & LM Client Upgrade`
> - `Miprov2 Status & Issues`
> - `Optimizing System Prompts in DSPy` 


- **升级到 DSPy 2.5 带来重大改进**: 升级到 **DSPy 2.5** 并在 **Predictor**（而非 **TypedPredictor**）中使用 **LM client** 修复了许多问题，并带来了更好的开箱即用性能。
   - *奇特的是，* 这些改进与新的 **Adapters** 能更好地感知 **chat LMs** 有关。
- **Miprov2 问题与用户相关**: 关于 **miprov2** 损坏的担忧得到了澄清，结果发现问题出在用户的 **LM client** 中，与 **MIPRO** 本身无关。
   - 社区讨论了通过在 **dspy.Evaluate** 调用中默认开启 `provide_traceback` 来改进错误处理。
- **在 DSPy 中优化 System Prompts**: 一位用户表示需要关于如何手动将 **System Prompt** 输入 **DSPy** 进行优化的指导。
   - 其他人建议参考 **DSPy documentation**，利用平台进行自定义 Prompt 优化。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1290060756783726716)** (8 messages🔥): 

> - `OpenSearchRetriever for DSPy`
> - `Healthcare Fraud Classification`
> - `Long Docstring Confusion`
> - `Using GPT-4o Mini and Claude Models` 


- **提供 OpenSearchRetriever**: 一位成员表示，如果社区有兴趣，愿意分享他们为 **DSPy** 构建的 [OpenSearchRetriever](https://link.to.github)。
   - *Chiggly007* 鼓励他们分享代码或提交 **PR**，认为这对其他人会很有帮助。
- **医疗欺诈分类的困扰**: 一位成员正在将司法部关于**医疗欺诈**的新闻稿分为三类，但在准确性方面遇到困难。
   - 他们注意到模块将“计费为高套编码（upcoding）的医疗不必要护理”误分类了，因此需要更好的方法来定义类别标准。
- **长 Docstrings 导致的困惑**: 该成员指出，在类 **Signature** 的 **docstring** 中使用过长的解释时，准确性会出现问题。
   - *Okhattab* 肯定了详细的 **docstring** 本身没有问题，但询问了正在使用的是哪种语言模型。
- **探索语言模型**: 该成员目前正在使用 **GPT-4o Mini**，并计划测试 **Claude models** 进行最终分类。
   - 他们讨论了在处理从美国司法部网站抓取的公开数据时，如何应对 Token 限制的问题。
- **潜在的数据基准测试**: *Okhattab* 建议可以访问这些公开数据来创建 **Benchmark** 和相关的 **Notebooks**。
   - 他们通过 **DM** 联系了该成员，以进一步讨论这种可能性。


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1289308501289603197)** (5 条消息): 

> - `全栈开发`
> - `AI 执行指令`
> - `Open Interpreter 功能` 


- **全栈开发人员寻求新客户**：一位资深全栈开发人员宣布了他们在利用 **React + Node** 和 **Vue + Laravel** 构建**电子商务平台**、在线商店和房地产网站方面的专业知识。
   - 他们表示有兴趣为长期项目联系新的可靠客户，并欢迎通过私信进行潜在合作。
- **修改 AI 执行指令的请求**：一位成员提出是否可以**重新指示 AI 的执行指令**，以便用户能够独立修复和调试问题。
   - 他们提到经常遇到与路径相关的错误，并对当前的功能感到沮丧。
- **关于 Open Interpreter 用途的咨询**：一位成员对 **Open Interpreter** 的实际功能表示困惑，询问它是否能执行特定任务。
   - 他们的询问引发了关于澄清 AI 能力和整体产品功能的关注。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1289301733473914972)** (9 条消息🔥): 

> - `数据包解码错误`
> - `客户端连接问题`
> - `Ngrok 错误` 


- **数据包解码错误问题**：一位用户报告了在服务器重启或客户端连接期间反复出现的**数据包解码错误**：*Invalid data found when processing input*。
   - 另一位成员建议检查终端错误消息，但确认没有任何消息，表明该错误是持续发生的。
- **客户端连接困扰**：一位用户提到他们的手机在尝试连接时卡在 *Starting...* 页面。
   - 一位成员鼓励在指定频道发布设置详情，以便获得进一步帮助。
- **Ngrok 身份验证问题**：一位成员对 **ngrok 错误**表示沮丧，该错误指出在运行服务器时需要验证账户和 authtoken。
   - 他们推测问题是否源于未从 .env 文件中读取 *apikey*，并就这个看似琐碎的问题寻求帮助。
- **Open Interpreter 使用演示**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=4TNzwKuq_yg)，演示了使用基于 Open Interpreter 的软件刷入各种 **01** 设备的过程。
   - 该视频提供了软件能力的视觉指导，但未提供额外的文字描述。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ngrok.com/docs/errors/err_ngrok_4018/">ERR_NGROK_4018 | ngrok 文档</a>：消息</li><li><a href="https://www.youtube.com/watch?v=4TNzwKuq_yg">Human Devices 01 刷机演示</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1289309881601495120)** (2 条消息): 

> - `Open Interpreter 的影响`
> - `结合 Jan 使用 Open Interpreter`
> - `本地 LLM 界面` 


- **Open Interpreter 改变生活**：一年前，一位成员演示了一个新工具并引发了病毒式传播，从那时起，**Open Interpreter** 极大地影响了他们的生活，帮助他们结识了志同道合的朋友并深入探索了 **AI 世界**。
   - 他们对社区的支持表示感谢，并表示：*“让我们继续建设一个极其丰饶的未来。”*
- **Jan AI 作为计算机控制界面**：一位成员询问其他人是否使用过 **Jan**，并指出它与 **Open Interpreter** 的兼容性，强调了它作为本地 **LLM** 推理服务器的功能。
   - 他们分享了一个名为 *“使用 Jan AI 控制你的计算机”* 的 [YouTube 视频](https://www.youtube.com/watch?v=1l3B0AzbbjQ)，解释了 Jan 如何作为界面来控制你的计算机。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/MikeBirdTech/status/1839750338179674590">来自 Mike Bird (@MikeBirdTech) 的推文</a>：一年前的今天，我为在网上发现的这个酷炫新工具做了一个小演示。只是想展示一下它的功能，然后它就有点疯传了。从那时起，@OpenInterpreter 完全改变了...</li><li><a href="https://www.youtube.com/watch?v=1l3B0AzbbjQ">使用 Jan AI 控制你的计算机</a>：Jan.AI 是一个出色的本地推理服务器，用于提供本地 LLM 服务。但你知道你可以把它当作控制计算机的界面吗？Jan: https://jan.a...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1289324677646520340)** (8 messages🔥): 

> - `French Audio Dataset for CosyVoice`
> - `LAION Copyright Challenge`
> - `Phenaki Video Generation Model`
> - `Visual Language Models and Latent Diffusion Models`
> - `PALM-RLHF Datasets and Task Implementation` 


- **寻找用于 CosyVoice 的法语音频数据集**：一位用户请求用于训练 **CosyVoice** 的高质量**法语**音频数据集。
   - 他们表示需要合适的数据集来推进其项目。
- **LAION 在德国赢得版权挑战**：一个帖子强调 **LAION** 在**德国法院**赢得了首个版权侵权挑战。
   - 该帖子包含了一个链接，用于进一步讨论和了解这一法律胜利的细节。
- **测试用于文本生成视频的 Phenaki**：一位用户探索了用于从文本生成视频的 **Phenaki** 实现，并提供了一个 [GitHub 链接](https://github.com/lucidrains/make-a-video-pytorch) 进行测试。
   - 由于缺乏数据集，他们在训练前寻求初始测试的指导。
- **结合 Visual Language 和 Latent Diffusion Models**：讨论了结合 **VLM** (Visual Language Models) 和 **LDM** (Latent Diffusion Models) 以改进图像生成过程的潜力。
   - 理论方面包括由 VLM 生成 LDM 指令并有效优化输出的循环可能性。
- **实现 PALM-RLHF 训练数据集**：一位用户询问了为特定任务实现 **PALM-RLHF** 训练数据集最合适的频道和角色。
   - 他们寻求明确流程，以便使训练数据集与特定的操作需求保持一致。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/aiwars/comments/1fqpiut/laion_wins_first_copyright_infringement_challenge/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/lucidrains/make-a-video-pytorch">GitHub - lucidrains/make-a-video-pytorch: Implementation of Make-A-Video, new SOTA text to video generator from Meta AI, in Pytorch</a>：Make-A-Video 的 Pytorch 实现，来自 Meta AI 的最新 SOTA 文本生成视频生成器 - lucidrains/make-a-video-pytorch
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1289326572343398421)** (7 messages): 

> - `Transformer Models`
> - `Positional Encodings`
> - `RoPE in Attention Layers`
> - `Convergence Time in Training` 


- **Transformer 将成为主导架构**：一位成员提到，最终可能会演变为一个大型的 **Transformer** 模型，暗示 AI 领域对该架构的依赖日益增加。
   - 他们分享了一个 [emu 项目链接](https://emu.baai.ac.cn/about)，该项目探索了这一发展的各个方面。
- **Positional Encodings 可能简化架构**：成员们讨论了在 **Transformer** 块中使用 **Positional Encodings** 的想法，认为这可能带来更简洁的实现。
   - *一位成员确认*，位置信息已经集成到他们研究的各层特征中。
- **在 U-Net 的 Attention 层尝试 RoPE**：一位成员分享了在 **U-Net** 的 **Attention** 层尝试 **RoPE** 的经验，表示对其性能影响感兴趣。
   - 他们指出，目前尚不确定这种方法是否会影响整体的收敛时间（Convergence Time）。
- **位置信息在层间的传播**：一位成员指出，位置信息需要经过一些 **1D padded convolution** 层才能在网格中完全传播。
   - 他们建议，如果位置信息在早期就发挥作用，可能会显著影响结果。



**提到的链接**：<a href="https://emu.baai.ac.cn/about">Emu3</a>：未找到描述

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1289370604281139212)** (6 messages): 

> - `Vectorstores 交互`
> - `LLMs 的数据库使用`
> - `Discord 中的感谢礼物`
> - `Gemini 中的图像错误`
> - `修改 LangChain 中的推理方法` 


- **Vectorstores 可能需要示例问题**：一位成员建议，利用示例问题可以帮助 Vectorstores 寻找最接近的匹配，尽管这可能有些过度。
   - 他们强调需要通过测试来确定有效性。
- **数据库优于表格数据**：一位成员解释说，表格数据对于 LLMs 并不理想，这促使他们将表格数据转移到 **Postgres** 数据库中。
   - 他们现在正使用 **LangChain modules** 与该数据库进行交互。
- **感谢礼物咨询**：一位成员询问是否可以发送一份小礼物，作为对在 Discord 中提供帮助的人的感谢。
   - 他们表达了对他人贡献的认可愿望。
- **Gemini 突然出现图像错误**：一位成员报告说，在向 **Gemini** 发送图像时突然遇到错误，而之前运行正常。
   - 他们怀疑该问题可能是在升级所有 **pip packages** 后出现的。
- **修改 LangChain 推理方法**：一位成员正在探索如何在使用 **LangChain** 的同时，结合 **vllm** 中的优化来修改聊天模型的推理方法。
   - 他们有兴趣控制 LLM 如何解码 tokens，特别是在聊天历史和输入的开放式调用中。


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1289657420137762958)** (3 messages): 

> - `AI Realized Summit 2024`
> - `Manifold Research Frontiers 系列`
> - `斯德哥尔摩的 MLOps 聚会` 


- **AI Realized Summit 2024 定于 10 月 2 日举行**：由 **Christina Ellwood** 和 **David Yakobovitch** 在 UCSF 主办的 [AI Realized - The Enterprise AI Summit](https://lu.ma/airsummit) 将于 2024 年 10 月 2 日举行，届时将有 Enterprise AI 领域的行业领袖参加。
   - 与会者可以使用代码 **extra75** 节省 **$75** 的门票费用，门票包含会议期间的餐食。
- **Manifold Research Frontiers 讲座启动**：**Manifold Research** 正在推出 Frontiers 系列，以展示基础和应用 AI 领域的创新工作，首场讲座由 **Helen Lu** 主讲，重点关注 Neuro-symbolic AI 和人机协作。
   - 讲座将讨论 Autonomous Agents 在动态环境中面临的挑战，并开放免费注册，链接见[此处](https://lu.ma/cbflyi6s)。
- **咨询斯德哥尔摩的 MLOps 聚会**：一位最近搬到斯德哥尔摩的成员正在寻求有关该市 **MLOps 或 Infrastructure 聚会** 的信息。
   - 他们表达了与当地技术社区建立联系并了解即将举行的活动的愿望。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://lu.ma/airsummit">AI Realized – The Enterprise AI Summit · Luma</a>: 欢迎参加 AI Realized Summit 2024！...由 Christina Ellwood 和 David Yakobovitch 主办。2024 年 10 月 2 日，加入我们在旧金山举办的这一为期一天的独家峰会……</li><li><a href="https://lu.ma/cbflyi6s">Frontiers: Neuro-Symbolic Adaptation for Autonomous Agents · Zoom · Luma</a>: 欢迎来到 Frontiers - 在这个系列中，我们邀请在各个领域处于前沿的顶尖研究人员、工程师、设计师和领导者进行深度探讨……</li><li><a href="https://www.manifoldrg.com/events/">Manifold Research Group (第 1 页)</a>: 未找到描述</li><li><a href="https://www.manifoldrg.com">Manifold Research Group</a>: Manifold Research 是一家新型研发机构，致力于开展具有高影响力的前沿科学和技术项目，最终目标是改善和推进人类文明。
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 messages): 

zachmayer: Surya
  

---

### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1289413018567835693)** (3 messages): 

> - `Anti-slop Sampler`
> - `Dataset Creation` 


- **Calytrix 推出 Anti-slop Sampler**：开发了一个原型 Anti-slop Sampler，通过在检测到不需要的序列时进行回溯，从而在推理（Inference）过程中抑制不需要的词汇/短语。
   - Calytrix 正在努力使代码库可用于下游用途，并在 [GitHub](https://github.com/sam-paech/antislop-sampler) 上分享了该项目。
- **社区支持 Anti-slop 概念**：一位成员对 Anti-slop Sampler 的想法表示赞赏，指出：*“太酷了，我喜欢这个主意！”*
   - 积极的反馈表明了人们对提高数据集质量的创新方法的兴趣。



**提及的链接**：<a href="https://github.com/sam-paech/antislop-sampler">GitHub - sam-paech/antislop-sampler</a>：通过在 GitHub 上创建账户，为 sam-paech/antislop-sampler 的开发做出贡献。

  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1290379411245236314)** (1 messages): 

> - `SoraSNS`
> - `Takiyoshi Hoshida`
> - `Carnegie-Melon University`
> - `Apple's AR Kit` 


- **Takiyoshi Hoshida 演示 SoraSNS**：独立开发者 **Takiyoshi Hoshida** 将现场演示他的项目 [SoraSNS](https://discord.com/events/1089876418936180786/1277835047084363827)，这是一个社交媒体应用，提供来自你通常不关注的用户的私人时间线。
   - 该演示将突出应用中**昼夜天空**的概念，象征着开放和远距离观察，允许用户发现社交网络的新部分。
- **Hoshida 令人印象深刻的背景**：Hoshida 曾在 **Carnegie-Melon University 学习计算机科学**，在技术领域拥有丰富的经验。
   - 他此前曾就职于 **Apple 的 AR Kit 团队**，并参与了超过 **50 个 iOS 项目**。


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1290268665299734528)** (1 messages): 

> - `Hammer handle update`
> - `Hammer2.0 series models`
> - `Pull Request submission` 


- **Hammer Handle 迎来更新**：Hammer handle 已更新，标志着在设计和功能上的一些增强。
   - 预计这一新迭代将带来*令人兴奋的改进*。
- **推出 Hammer2.0 系列**：团队发布了 **Hammer2.0 系列模型**，包括 Hammer2.0-7b、Hammer2.0-3b、Hammer2.0-1.5b 和 Hammer2.0-0.5b。
   - 这些新增模型标志着产品多样化迈出了重要一步。
- **提交 Pull Request PR#667**：作为 Hammer 产品线更新的一部分，已提交了一个 Pull Request (PR#667)。
   - 此次提交是近期增强功能后开发过程的关键部分。


  

---



---



---



{% else %}


> 完整的逐频道细分内容已针对电子邮件进行了截断。
> 
> 如果您想查看完整细分，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[与朋友分享](https://buttondown.email/ainews)！提前致谢！

{% endif %}