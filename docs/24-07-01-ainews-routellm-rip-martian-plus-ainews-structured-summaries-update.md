---
companies:
- lmsys
- openai
date: '2024-07-02T00:23:08.479510Z'
description: '**LMSys** 推出了 **RouteLLM**，这是一个基于 Chatbot Arena **偏好数据**训练的开源路由框架。在保持
  **GPT-4 95% 性能**的同时，它在 MT Bench 上实现了**超过 85% 的成本降低**，在 MMLU 上降低了 45%，在 GSM8K 上则降低了
  35%。


  该方法通过使用基于语法的混合专家（MoE）路由和数据增强，超越了以往针对特定任务的路由方案，表现优于商业解决方案 40%。此次更新突显了 **LLM 路由**、**成本效益**以及**跨多模型性能优化**（而非仅限于单一模型或
  MoE 级别的改进）方面的进展。


  此外，AI Twitter 摘要还提到：**Gemma 2 模型系列**被视为顶级的开源模型；**Block Transformer 架构**可提升推理吞吐量；以及
  **karpathy** 提出的构建全“软件 2.0”计算机视觉系统的方案。'
id: b9088584-8ef0-45e9-9390-6a5c1547cfc2
models:
- gpt-4
- gemma-2-27b
- gemma-2-9b
original_slug: ainews-to-be-named-5628
people:
- karpathy
- bindureddy
- armand-joulin
title: RouteLLM：RIP Martian？（外加：AINews 结构化摘要更新）
topics:
- llm-routing
- cost-efficiency
- model-performance
- model-optimization
- data-augmentation
- syntax-based-routing
- mixture-of-experts
- inference-throughput
- software-2.0
- computer-vision
---

<!-- buttondown-editor-mode: plaintext -->**LLM Preference data is all you need.**

> 2024年6月28日至7月1日的 AI 新闻。
我们为您检查了 7 个 subreddits、[**384** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discords（**419** 个频道和 **6896** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**746 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

还记得 [4 月份的 Mistral Convex Hull](https://buttondown.email/ainews/archive/ainews-mixtral-8x22b-instruct-defines-frontier/)，以及 [5 月份 DeepSeekV2 的胜利](https://buttondown.email/ainews/archive/ainews-deepseek-v2-beats-mixtral-8x22b/)吗？成本与性能的有效边界再次被推高，但这次不是在单模型或 MoE 层面，而是在所有模型之间：

 
![image.png](https://assets.buttondown.email/images/15826a27-7536-4595-9b00-e75ad39574f7.png?w=960&fit=max)
 

值得注意的核心特性是这句话：我们利用来自 Chatbot Arena 的公开数据训练了四种不同的 router，并证明它们可以在不降低质量的情况下显著降低成本。与仅使用 GPT-4 相比，在 **MT Bench 上成本降低了 85% 以上，在 MMLU 上降低了 45%，在 GSM8K 上降低了 35%**，同时仍能达到 **GPT-4 性能的 95%**。

LLM routing 的想法并不新鲜；[model-router](https://x.com/withmartian/status/1641884426161520640) 是 2023 年初“Woodstock of AI”聚会上的一个特色项目，随后凭借该概念筹集了 [900 万美元的可观种子轮融资](https://www.businessinsider.com/martian-ai-startup-pitchdeck-seed-vc-funding-nea-prosus-2023-12)。然而，这些 routing 解决方案是基于**特定任务（task-specific）**的 routing，即不同模型擅长不同任务的概念，这与基于**语法（syntax-based）**的 MoE routing 形成鲜明对比。

LMSys 的新型开源 router 框架 [RouteLLM](https://github.com/lm-sys/RouteLLM) 进行了创新，它使用来自 The Arena 的 **preference data** 来训练其 router，其基础是根据 prompt 预测用户最喜欢的最佳模型。他们还对 Arena 数据进行了 **data augmentation**，以进一步提高其 routing 效益：

 
![image.png](https://assets.buttondown.email/images/6ac3c6bf-5899-4141-974f-9ac867341cc9.png?w=960&fit=max)
 

或许最残酷的是，LMSys 声称在相同性能下，其表现优于现有的商业解决方案 40%。

 
![image.png](https://assets.buttondown.email/images/463c1ea5-b1f7-4545-84eb-f1455b8659ad.png?w=960&fit=max)
 

> **特别 AINEWS 更新：结构化摘要**
>
> 我们修改了核心摘要代码以使用 structured output，重点在于实现：1) 更好的主题选择，2) 事实与观点/反应的分离，以及 3) 更好的链接和高亮显示。您可以看到相应的效果。我们确实发现，通过这次更新，内容变得更加冗长，但我们希望这种结构使其更易于浏览，我们即将推出的网页版也将更易于导航。


![image.png](https://assets.buttondown.email/images/d09386ee-eed1-4cd4-9918-cc0cc92c8618.png?w=960&fit=max)


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要回顾

> 所有摘要由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**AI 模型与架构**

- **Gemma 2 模型系列**：[@armandjoulin](https://twitter.com/armandjoulin/status/1807412150144491910) 指出 Gemma 2 27B 目前是**最佳的开源模型，且体积比替代方案小 2.5 倍**，验证了团队的工作。[@bindureddy](https://twitter.com/bindureddy/status/1807485457048953010) 表示 Gemma 2 27B 的性能接近 Llama 3 70B，而 Gemma 2 9B 凭借出色的训练后处理（post-training）超越了帕累托前沿（Pareto front）。
- **Block Transformer 架构**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1807551317759455559) 分享了一篇论文，显示 **Block Transformer 与原生 Transformer 相比，在困惑度（perplexity）相当的情况下，推理吞吐量提升了高达 20 倍**。它通过将 KV cache 的 IO 开销从平方级降低到线性级来实现。它将昂贵的全局建模隔离到较低层，并在较高层应用快速的局部建模。
- **全 Software 2.0 计算机视觉**：[@karpathy](https://twitter.com/karpathy/status/1807497426816946333) 提出了一个 **100% 全 Software 2.0 计算机，仅由单个神经网络组成，没有经典软件**。设备输入直接馈入神经网络，其输出显示为音频/视频。

**AI Agent 与推理**

- **视频生成模型的局限性**：[@ylecun](https://twitter.com/ylecun/status/1807497091964449266) 认为视频生成模型不理解基础物理学或人体。[@giffmana](https://twitter.com/giffmana/status/1807511985807908926) 对 AI 领导者使用笨拙的体操 AI 视频来声称人体物理很复杂感到恼火，这就像展示 DALL-E mini 的生成结果来说明当前的图像生成注定失败一样。
- **用于多步推理的 Q***：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1807561583079367086) 分享了一篇关于 Q* 的论文，该论文通过审慎规划引导 LLM 解码，从而在无需任务特定微调的情况下提高多步推理能力。它将推理形式化为 MDP，并使用拟合 Q 迭代（fitted Q-iteration）和 A* 搜索。
- **用于长期记忆的 HippoRAG**：[@LangChainAI](https://twitter.com/LangChainAI/status/1807466126097650112) 分享了 HippoRAG，这是一个受神经生物学启发的 LLM 长期记忆框架，用于**持续集成知识**。它使用 Unstructured API 为文档丰富元数据。

**AI 应用**

- **用于法律工作流的 AI**：[@scottastevenson](https://twitter.com/scottastevenson/status/1807540320982433945) 指出 **Agent 正在进入法律工作流**，并艾特了 @SpellbookLegal。
- **来自 Eureka Health 的 AI 医生**：[@adcock_brett](https://twitter.com/adcock_brett/status/1807444895902138368) 分享道，Eureka Health 推出了 Eureka，这是“首位 AI 医生”，根据早期测试，其提供的**个性化护理比大多数美国护理快 90 倍**。
- **AI 生成的奥运会回顾**：[@adcock_brett](https://twitter.com/adcock_brett/status/1807444918337519673) 报道称 NBC 将为 2024 年奥运会推出 **10 分钟的 AI 生成回顾**，克隆 Al Michaels 的声音在 Peacock 上解说精彩片段。演示效果与人类制作的内容几乎无法区分。

**迷因与幽默**

- [@nearcyan](https://twitter.com/nearcyan/status/1807557363840520355) 针对 AI 的闪烁图标（sparkle icons）开玩笑说：“我要变成小丑了，你们到底怎么了”。
- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1807443823259230276) 幽默地建议，如果你不能生成 6-7 只鸟，你在社交上就完蛋了，因为 4-5 只已被验证，但 0-5 和 2-5 太难，而 5-5 又太容易。
- [@kylebrussell](https://twitter.com/kylebrussell/status/1807462686566826356) 分享了一张图片，开玩笑说“你值得拥有让你产生这种感觉的对手”。

---

# AI Reddit 摘要回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**幽默/迷因**

- **AI 生成的幽默视频**：在 /r/StableDiffusion 中，用户分享了 [幽默内容的 AI 生成视频](https://v.redd.it/7boa86nsbu9d1)，例如一个人打开一盒巧克力，结果巧克力在脸上炸开。评论指出 AI 视频中那种梦幻般的质感和诡异的动作，并将其与人类大脑的处理方式进行了类比。
- **AI 结合迷因**：另一个 [AI 生成视频](https://v.redd.it/t3wzai9b0g9d1) 结合了各种迷因，用户认为这非常有趣，是 AI 的一个很好的应用场景。
- **超人工智能 (Superintelligent AI) 迷因**：一些表情包迷因被分享，描绘了 [人类与超人工智能之间的关系](https://i.redd.it/9ww15hhkst9d1.png)，以及一个人 [试图用遥控器控制先进 AI](https://i.redd.it/rgkq6s1mrt9d1.png) 的场景。另一个迷因展示了 [当 AI 说它不会杀死人类时人们松了一口气的样子](https://i.redd.it/04v5eb7isq9d1.png)。

**AI 艺术**

- **恶魔版越南战争**：在 /r/StableDiffusion 中，[AI 生成的图像](https://www.reddit.com/gallery/1dsfnje) 描绘了一个虚构的越南战争场景，海军陆战队员与恶魔作战，灵感源自 Beksinski、Szukalski 和 Giger 的恐怖艺术风格。用户分享了用于创建这些图像的详细 Prompt。
- **穿越时空的 Blåhaj**：一系列 [图像](https://imgur.com/a/T6ELspY) 展示了宜家毛绒鲨鱼玩具 Blåhaj 在各种历史和未来场景中的样子。
- **19 世纪玩电子游戏的孩子**：一段来自 Luma Dream machine 的 [AI 生成视频](https://v.redd.it/dqdvym45lu9d1) 描绘了 19 世纪的孩子们违背时代背景地玩着电子游戏。

**AI 扩展与能力**

- **Kurzweil 对智能扩张的预测**：在《卫报》的一篇 [文章](https://www.theguardian.com/technology/article/2024/jun/29/ray-kurzweil-google-ai-the-singularity-is-nearer) 中，AI 科学家 Ray Kurzweil 预测到 2045 年，AI 将使人类智能扩展一百万倍。
- **AI 的扩展极限**：一段 [YouTube 视频](https://www.youtube.com/watch?v=ZyMzHG9eUFo) 探讨了 AI 的扩展（Scaling）究竟能走多远。
- **模型大小 vs 数据质量**：/r/LocalLLaMA 中的一个 [帖子](https://i.redd.it/16iuw5kemu9d1.png) 认为，有时拥有高质量数据的 9B 小模型在推理任务上可以超越 2T 模型，这引发了关于模型大小与数据质量相对重要性的讨论。不过，评论者指出这更多是一个特例。
- **处理器性能进入平台期**：一张带有对数 y 轴的 [处理器性能图表](https://i.redd.it/7xi5yqjy3r9d1.jpeg) 被分享，暗示性能提升实际上并非指数级的，并且即将进入平台期。

**AI 模型与基准测试**

- **Gemma 2 9B 模型**：在 /r/LocalLLaMA 中，一位用户发布了 [Gemma 2 9B 的好评贴](https://www.reddit.com/r/LocalLLaMA/comments/1drxhlh/gemma_2_9b_appreciation_post/)，发现它在自己的使用场景中比 Llama 3 8B 表现更好。
- **Llama 400B 发布时机**：另一个 [帖子](https://www.reddit.com/r/LocalLLaMA/comments/1drw01y/400b_llama3_might_not_be_impactful_if_not/) 讨论了 Llama 400B 的潜在影响，认为它需要尽快发布才能产生影响力。评论者指出，对于大多数用户来说，400B 模型不如 ~70B 模型实用。
- **Gemma2-27B 在 LMSYS 的表现**：围绕 Gemma2-27B 在 LMSYS 基准测试中超越更大的 Qwe2-72B 和 Llama3-70B 模型的 [讨论](https://www.reddit.com/r/LocalLLaMA/comments/1ds2tv9/gemma227b_outperforms_both_qwe272b_and_llama370b/)，质疑这反映的是真实能力还是 LMSYS 特有的因素。
- **关于 Llama 400B 的猜测**：根据一张所谓的截图，有人 [猜测](https://www.reddit.com/r/LocalLLaMA/comments/1ds2p09/llama_400_released_internally_at_meta_available/) Meta 可能已经在内部发布了 Llama 400B，并已在 WhatsApp 上提供。
- **Llama 3 405B 发布的意义**：一张 [图片](https://i.redd.it/cvzbd8cfwp9d1.png) 暗示 Llama 3 405B 的发布可能会促使其他大型科技公司发布强大的开源模型。
- **Gemma-2-9B 在 AlpacaEval2.0 的表现**：根据 Hugging Face 页面显示，[UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3 模型](https://huggingface.co/UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3) 在 AlpacaEval2.0 基准测试中实现了 53.27% 的胜率。

---

# AI Discord 回顾

> 摘要的摘要之摘要

1. **模型训练、量化与优化**：
   - **[Adam-mini Optimizer](http://arxiv.org/abs/2406.16793)：节省 VRAM 达 45-50%。** 实现与 AdamW 类似的性能，且没有过高的内存开销，适用于 **llama 70b** 和 **GPT-4** 等模型。
   - **Hugging Face 的新型 [低精度推理](https://github.com/huggingface/diffusers/discussions/8746)** 提升了 transformer pipeline 的性能。针对 **SD3** 和 **PixArt-Sigma** 等模型，它提高了计算效率。
   - **[CAME Optimizer](https://arxiv.org/abs/2307.02047)：内存高效优化。** 在减少内存需求的情况下表现出更好或相当的性能，有利于 stable diffusion 训练。

2. **新 AI 模型与基准测试**：
   - **[Gemma 2](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF)** 表现参差不齐，但在进一步优化后显示出对抗 Phi3 和 Mistral 的潜力。
   - **[Claude 3.5](https://youtu.be/B45s_qWYUt8?si=_c7sQUFUN6bZa61m)** 尽管最初期望很高，但面临上下文保留问题；**Claude Opus** 等替代模型表现可靠。
   - **[Persona Hub](https://arxiv.org/abs/2406.20094)** 利用多样化的数据应用使 MATH 基准测试分数飙升，证明了合成数据在更广泛 AI 应用中的有效性。

3. **开源 AI 工具与社区参与**：
   - **[Rig Library](https://bit.ly/Rig-Feeback-Form)**：与 Cohere 模型完全集成，面向 Rust 开发者，提供 100 美元的反馈奖励以获取见解。
   - **[LlamaIndex](https://t.co/YsYoVOIirb)** 推出了迄今为止最好的 Jina reranker，并提供了混合检索设置的全面教程，有望推动检索流水线的进步。
   - **[Jina Reranker](https://t.co/cTxW2UwuZ0)**：新的混合检索器教程详细介绍了结合多种方法以获得更好性能，允许与 Langchain 和 Postgres 等工具集成。

4. **技术挑战与故障排除**：
   - **[BPE Tokenizer Visualizer](https://screenrec.com/share/SV7cw9vryx)** 帮助理解 LLM 中的 tokenizer 机制，邀请社区反馈以完善该工具。
   - **数据库队列问题** 困扰着 Eleuther 和 Hugging Face 模型的基准测试工作，敦促用户寻找像 **[vllm](https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm)** 这样更高效的替代方案。
   - **[跨多个系统训练 GPT 模型](https://github.com/tinygrad/tinygrad/pull/5159/files#diff-91ca5e2e75ef3ea1982c8ca6cc175ee88f20efa0d8e4b96f305b970dc6df71e7R291)**：讨论强调了处理 GPU 限制和优化规模以实现有效的资源利用。

5. **AI 在现实世界中的应用**：
   - **[Featherless.ai](https://featherless.ai)** 推出以固定费率提供 LLM 的 serverless 访问，无需 GPU 设置即可轻松开发 AI persona 应用。
   - **[Deepseek Code V2](https://openrouter.ai/models/deepseek/deepseek-coder)** 因其在高效解决复杂微积分和编码任务方面的表现而受到高度赞扬。
   - **[医疗保健中的计算机视觉](https://youtu.be/QIjB4tqLqcs?si=bHfDg3WuRn5rEYbs)**：探索使用 CV 的 Agentic 医院，强调计算资源整合以增强患者护理并减少行政工作量。

---

# 第一部分：高层级 Discord 摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **蛋白质预测精度：ESM3-SM-open-v1 的新前沿**：Evolutionary Scale 发布了 **ESM3-SM-open-v1**，这是一个用于预测蛋白质序列的变革性模型，其基础是对生物特性的深刻理解。通过 [GitHub 仓库](https://github.com/evolutionaryscale/esm) 和交互式的 [Hugging Face space](https://huggingface.co/spaces/as-cle-bert/proteins-with-esm)，其效用得到了极大提升，这种协同作用有望进一步推动生物研究。
   - 随着从业者被号召利用该模型在 [Hugging Face](https://huggingface.co/spaces/as-cle-bert/proteins-with-esm) 上开辟研究路径，势头日益强劲，社区已将其誉为生物 AI 应用领域的一次**令人振奋**的飞跃。
- **提取审美精华：揭秘增强模型精炼的数据集**：**Terminusresearch** 策划了一个视觉数据宝库，其包含 33.1k 张图像的 [photo-aesthetics 数据集](https://huggingface.co/datasets/terminusresearch/photo-aesthetics)，用于磨练 AI 的审美辨别力。该数据集充满了真实的摄影照片，为细致的模型训练奠定了基础。
   - 捕捉建筑以及人与物体互动图像的补充数据集对主数据集进行了完善，这一初创尝试在提升模型以审美眼光导航和解释视觉领域的能力方面展现出前景。
- **驯服 Tokenization：为 LLM 清晰度而生的 BPE 描绘**：BPE Tokenizer 是 LLM 运作的基石，通过一位热心社区贡献者制作的新型 [可视化工具](https://github.com/mdabir1203/BPE_Tokenizer_Visualizer-)，其透明度得到了提升。这个可部署的实用程序旨在阐明 Tokenizer 的机制，增强开发者对 LLM 复杂性的掌握。
   - 众包改进工作正在进行中，呼吁反馈以纠正问题并完善可视化工具——这一倡议旨在丰富 LLM 的可访问性。
- **治愈的远见场所：用 CV 规划 Agentic 医院**：一位来自悉尼和海德拉巴的医生发出了响亮的号召，要求用 **computer vision** 开创 **agentic hospitals**——这一构想旨在减轻沉重的行政负担。**鱼眼摄像头**被预定为协调平稳运行和增强以患者为中心的护理的关键环节。
   - 随着对计算资源的渴求，对算力资源的请求也在扩大，预示着一场潜在的革命，AI 可能会改变医疗动态，培育技术驱动的医疗生态系统。
- **推理的独创性：在 Transformer 中拥抱低精度目标**：对 **SD3** 和 **PixArt-Sigma** 等 **transformer pipelines** 中低精度推理的探索，标志着向计算经济性的转变。[GitHub 讨论](https://github.com/huggingface/diffusers/discussions/8746) 揭示了该技术在提升模型性能敏捷性方面的潜力。
   - 虽然具有优化的潜力，但这种新兴方法也充满了挑战，需要审慎解决才能释放其全部效益。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **RAM 优化配置：Unsloth 在 WSL2 上平稳运行**：工程师们推荐了在 WSL2 上运行 Unsloth AI 的优化方案，以充分利用系统资源。交流了如在 `.wslconfig` 中设置内存限制以及使用特定安装命令等配置，以在各种硬件设置上获得性能提升。
   - 分享了解决安装障碍的排错技巧，并就内存分配调整和全新的命令行达成共识，这是在 Intel 和 AMD 架构上释放 Unsloth 效率的关键。
- **DRM 的双重角色：Unsloth 中的多 GPU 支持与授权**：Unsloth 在多 GPU 训练中引入了 DRM 系统，概述了受 NDA 保护的严格测试，确保 GPU ID 绑定和持久化 ID，揭示了在授权控制与功能灵活性之间取得平衡的幕后努力。
   - 社区热烈讨论了多 GPU 设置的配置和边界，DRM 稳定性的更新对于扩展 AI 训练能力至关重要。
- **精细化微调：探索 Unsloth 的微调机制**：AI 爱好者们剖析了微调 Lexi 和 Gemma 模型的复杂过程，指出了诸如 system tokens 等特定怪癖，解决了微调后输出无限生成的故障，并强调使用加粗 Markdown 语法以提高清晰度。
   - 优化微调过程的共享技术包括多语言数据集翻译、通过仔细策划避免灾难性遗忘，以及恰当地使用 tokenizer 函数以匹配自定义训练数据。
- **合成角色塑造数据：MATH 评分的巨幅提升**：Persona Hub 拥有十亿级角色的合成数据生成方法论一直是热门话题，它在数学推理分数上实现了飞跃，其 [摘要](https://arxiv.org/abs/2406.20094) 中展示的易用性引发了热议。
   - 在对该项目的各种观点中，一些人强调了聚合数据本身比生成代码更重要，引发了关于大规模合成数据应用关键要素的辩论。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemma 的 GPU 策略**：虽然 Gemma 2 支持将 GPU 任务卸载到 **CUDA** 和 **Metal**，但根据社区反馈，*0.2.23* 版本之后的更新对于解决困扰该 AI 模型性能的问题至关重要。[深度探索](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF)显示，Gemma 2 在特定配置下存在模型加载时间过长的问题；AI 专家们呼吁持续进行补丁修复。
   - 一个被提出的中肯观点是，Gemma 2 目前受限于 **CUDA** 和 **Metal**，并建议未来应拓宽支持范围。社区对话详细讨论了技术挑战，并提出了旨在增强模型鲁棒性和兼容性的潜在更新方案。
- **AutoUpdater 的敏捷进展**：LM Studio 的 **AutoUpdater** 运行流畅，将用户版本推升至 **v0.2.26**，并对 **v0.3** 充满期待。一份[公开帖子](https://lmstudio.ai)传达了专门针对 **LLama 3 8B model** 的增强功能，修复了其忽略停止序列（stop sequences）的棘手问题——这是一个备受欢迎的改进。
   - 围绕近期功能发布的讨论集中在它们解决以往痛点的能力上，例如拒绝遵守停止序列的问题。用户们以积极的态度认可了这些承诺带来和谐人机交互的更新。
- **Deepseek 的 CPU 历程**：Deepseek v2 展示了其强大的实力，在高性能 CPU 上仅调用其 **200B+ 参数中的 21B**，在强悍的 **Threadripper** 系统上跑出了 **3-4 tokens/sec** 的速度。大量用户生成的性能数据汇入社区对话，为 CPU 的可行性声明提供了依据。
   - **共享用户测试** 遍历了从 RAM 占用率到加载速度的各种指标，涵盖了 Deepseek v2 在顶级 CPU 上强劲但管理出奇良好的性能体验。这些稳健的评估旨在为未来的开发和实际场景中的应用指明方向。
- **量化探索的试验**：[量化查询](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md)占据了讨论的主导地位，特别关注在性能和效率之间取得平衡的 “GGUF quants”。针对 **q5_k_l** 的实验性尝试引发了分析性轶事，旨在追求资源消耗的敏捷与轻量。
   - 伴随着一系列经过基准测试的产物（如 **q5_k_l** 和 **GGUF quants**），AI 开发者群体在追求性能精准度的同时，也强调保留模型原有的实力。记录在案的讨论深入探讨了数据和用户反馈，力求为这些前沿的性能增强手段寻找最佳实践。
- **滑动窗口 (Sliding Window) 的平稳运行**：Gemma 2 传奇的最新篇章是 **Sliding Window Attention**，它刚刚被融合进最新的 **llama.cpp** 中。这一精明的更新允许 AI 熟练地筛选过去的 Token，巧妙地加深其上下文理解。用户们正屏息以待针对遗留质量问题的[修复](https://github.com/ggerganov/llama.cpp/pull/8227)。
   - 随着强大的 **Sliding Window Attention** 功能在 Gemma 2 场景中首次亮相，并带来了通过熟练管理 Token 历史来增强性能的承诺，虚拟社区中涌现出了各种成功的案例。然而，在这些希望中，谨慎的声音依然坚定，对未来全面完善的预期保持着审慎的态度。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 中处于困境的 Loras**：一场激烈的讨论集中在为 **Stable Diffusion 3** 训练 **Loras** 的挑战上，重点是需要强大的支持。人们对 **SD 8b** 潜在的易用性充满热情。
   - 虽然有些人急于开始，但主流观点倾向于保持谨慎，敦促耐心等待更好的**训练工具**和数据，以防止产生质量不合格的 Loras 或 Checkpoints。
- **GPU 专家指导新手**：关于运行 Stable Diffusion 的**硬件要求**展开了辩论，普遍共识倾向于使用至少具备 **12GB VRAM** 的 **Nvidia GPU**。
   - 值得注意的是，现有的 RTX 3090 显卡受到了称赞，而最新的 RTX 4080 和 4090 尽管价格昂贵，但因其面向未来的属性而受到关注。
- **安装入门**：用户们团结起来解决 Stable Diffusion 的**安装问题**，分享了关于 **Automatic1111** 和 **ComfyUI** 等各种界面的知识，以及关键的设置命令。
   - 交流了有用的资源和指南，包括特定的配置建议，如加入 'xformers' 和 'medvram-sdxl' 以增强复杂工作流的性能。
- **高分辨率技巧提升艺术清晰度**：社区深入探讨了在 SDXL 中使用**高分辨率修复设置 (high-resolution fix settings)** 以获得更清晰的图像，强调了精确参数设置的重要性，例如 '10 hires steps'。
   - 参与者放大了 **adetailer** 等插件的好处，强调了它细化图像关键方面的能力，特别是动漫风格图形中的面部和眼睛。
- **模型挖掘与 Loras 知识**：对话挖掘出了寻找**模型和 Loras** 的来源，点名了 **Civitai** 等平台，因其广泛的收藏和用户贡献。
   - 分享了关于使用 Prompt 示例作为引导以准确利用这些资产的见解，强调了在模型训练和分发方面的集体努力。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **同步时间：时区工具解决调度难题**：分享了两个工具 **[Discord Timestamps](https://r.3v.fi/discord-timestamps/)** 和 **[When2meet](https://www.when2meet.com/)**，以**简化跨时区的会议协调**。
   - **Discord Timestamps** 和 **When2meet** 都通过转换时间和汇总空闲状态来缓解调度烦恼，促进了**轻松的集体调度**。
- **对数尺度的数值稳定性基础**：一篇[博客文章](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)激发了关于 **log exp 函数**在**数值稳定性**中作用的讨论，强调其在计算中用于**防止下溢 (underflow)**。
   - **辩论随之而来**，涉及其必要性的频率，并对对数尺度在**建模**和 **ML** 中的不可或缺性提出了疑问，突显了**实践中的分歧**。
- **Tensor 故事：PyTorch 工作坊中的元数据难题**：提出了在操作中保留 `torch.Tensor` 中**元数据**的挑战，并提出了诸如**子类化 (subclassing)** 之类的建议，但尚未出现明确的解决方案。
   - 面对支持**动态输入形状**的限制，`torch.compile` 出现了**编译难题**，这增加了 **HuggingFace transformers** 使用的复杂性，虽然提出了解决方案但尚未被采用。
- **闪现的洞见：AI 工程师赞赏著名演讲系列**：聊天中充满了对富有启发性的演讲系列的认可。
   - **Stephen Jones 的演讲**因其持续输出的**深刻见解内容**而赢得赞誉，巩固了他在 **AI 工程师**圈子中的声誉。
- **性能的高峰与低谷：使用 FP16 穿越 Kernel 景观**：Kernel 性能成为热门话题，**FP16** 显示出**单次启动效率**，而 **bfloat** 则需要多次启动，这引发了优化**大 Tensor 操作**的策略讨论。
   - **Bitpacking 优化**在较小的 Tensor 上显示出潜力，但在处理大体积 Tensor 时效果减弱，促使对 **Kernel 性能**增强进行**进一步探索**。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemma 的盛大发布：Gemma 2 模型登陆 Perplexity Labs**：**Gemma 2 模型**已上线 [Perplexity Labs](https://labs.pplx.ai)，旨在通过用户对其性能的反馈来促进社区参与。
   - 热情的参与者现在可以试用新模型并贡献见解，该公告鼓励互动式的社区体验。
- **会聊天的 Android 设备：按住说话 vs 免提模式**：Perplexity AI 的 Android 应用现在推出了**语音对语音功能**，通过免提和按住说话模式增强了用户可访问性，正如其最新更新中所详述的那样。
   - 免提模式在屏幕激活后立即开始监听，而按住说话则等待用户指令，旨在丰富用户交互。
- **Claude 的上下文处理不力：用户抱怨其“健忘”**：有关 **Claude 3.5** 无法保持上下文的报告激增，尽管在讨论特定话题，它却转向概括性的回答，这给工程师带来了意想不到的对话转折挑战。
   - 切换到 **Opus** 模型对某些人来说有所改善，这暗示了 **Claude 3.5** 中可能存在影响交互的 Bug，社区呼吁更智能的 Pro 搜索以保留上下文。
- **API 烦恼：Perplexity API 出现差异**：Perplexity API 用户正在应对不一致性，发现像 `after:2024-05-28` 这样的日期特定过滤器可能会诱导 API 生成超前日期的内容，引发了关于其预测能力的辩论。
   - 由于 Apple ID 识别问题，用户在 Perplexity Labs Playground 的交互遇到障碍，引发了关于用户包容性和体验改进的对话。
- **游戏机制：Minecraft 的机制会误导未成年人吗？**：一个激烈的帖子批评了 [Minecraft 的修复机制](https://www.perplexity.ai/page/Minecraft-Repair-Mechanics-NdRggXKXRXyGY8LgKsp1dQ)，认为游戏内的逻辑可能会扭曲青少年对现实世界工具修复的理解。
   - 这场数字辩论深入探讨了教育影响，敦促对 Minecraft 等虚拟环境如何播下误解种子进行现实检查，促使程序员思考其影响。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **十亿角色突破：数学基准测试飙升**：Aran Komatsuzaki 的推文强调了成功创建的 [十亿角色数据集 (billion persona dataset)](https://x.com/arankomatsuzaki/status/1807593343007818065)，将 MATH 基准测试分数从 49.6 提升至 64.9。其背后的方法，包括多样化的数据应用，详见 [Persona Hub GitHub](https://github.com/tencent-ailab/persona-hub) 和相应的 [arXiv 论文](https://arxiv.org/abs/2406.20094)。
   - 该新数据集实现了**高质量数学问题**和游戏 NPC 场景的合成生成。这一创新展示了合成数据带来的显著性能提升，为学术和娱乐 AI 应用提供了众多用例。
- **梦境数据二元性：Android & Human 数据集**：该数据集将 10,000 个真实梦境与由 **Oneirogen 模型**合成的 10,000 个梦境进行了对比，展示在 [Hugging Face](https://huggingface.co/datasets/gustavecortal/the-android-and-the-human) 上。Oneirogen 的变体（0.5B、1.5B 和 7B）为梦境叙事评估提供了新标准。
   - 该语料库可用于辨别真实梦境与生成梦境内容之间的差异，为高级分类任务和心理学 AI 研究铺平了道路。
- **科技巨头的巨额投资：Microsoft 与 OpenAI 的数据中心**：据 [The Information](https://www.theinformation.com/articles/microsoft-and-openai-plot-100-billion-stargate-ai-supercomputer) 报道，Microsoft 透露了与 OpenAI 在 Stargate 项目上的合作，可能为该项目注入超过 1000 亿美元。两家公司旨在应对 AI 对巨大算力日益增长的需求。
   - 考虑到 Microsoft 为维持如此庞大的计算需求而制定的核能战略，这一举措可能会显著影响能源行业。
- **投机速度：SpecExec 的 LLM 解码创新**：SpecExec 提供了一种全新的 LLM 推理方法，通过在消费级 GPU 上使用 4-bit 量化，实现了高达 18.7 倍的 [速度提升](https://www.together.ai/blog/specexec)。这一突破促进了更快的 LLM 运行，可能简化 AI 在更广泛应用中的集成。
   - 该模型对序列进行投机解码，并由核心算法快速验证，引发了关于其与不同 LLM 家族的兼容性以及集成到现有平台中的讨论。
- **利用 PhyloLM 绘制 LLM 系统发育图**：PhyloLM 的新颖方法引入了系统发育学原理来评估 LLM 的谱系和性能，详见 [arXiv 报告](https://arxiv.org/abs/2404.04671)。该方法根据 LLM 输出的相似性制作树状图，评估了 111 个开源模型和 45 个闭源模型。
   - 这种方法在训练数据不完全透明的情况下，梳理出 LLM 之间的性能特征和关系，提供了一种具有成本效益的基准测试技术。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **“错误即值”引发设计讨论**：Mojo 社区的讨论揭示了将**错误作为值（errors as values）**处理与传统异常处理之间的细微差别。讨论显示出对 **Variant[T, Error]** 的偏好，以及对更完善的 **match 语句**的需求。
   - 贡献者们指出 Mojo 中 try/except 的使用仅仅是“语法糖”，并建议在语言设计中深入考虑更优雅的错误解决方式。
- **Ubuntu 用户齐心协力掌握 Mojo**：用户们在 **Ubuntu** 上安装 **Mojo** 时展开了协作。在 **Raspberry Pi 5** 的 **Ubuntu 24.04** 上安装成功的案例展示了社区共同解决问题的特质。
   - 对话强调了社区支持在克服安装难题中的重要性，特别是对于在不同 Ubuntu 版本中摸索的新手而言。
- **Mojo 马拉松：编程挑战赛开始**：[每月编程挑战](https://discord.com/channels/1087530497313357884/1255303604894437388)已启动，为展示和磨练 Mojo 社区技能提供了一个动态平台。
   - 该倡议由 **@Benny-Nottonson** 发起，重点关注优化的**矩阵乘法（matrix multiplication）**等实际问题，详细的参与指南可在 [GitHub 仓库](https://github.com/Benny-Nottonson/Mojo-Marathons)中找到。
- **AI 愿景：Cody 使用 Mojo 编程**：Cody 与 Mojo 的交集引发了兴趣，讨论集中在利用 Cody 预测语言特性上。类 Python 的语法为流畅集成铺平了道路。
   - 随着探索 SIMD 等高级 Mojo 特定特性的愿景，社区正准备推向像 Cody 这样的辅助机器人所能实现的极限。
- **异步 I/O 与 Mojo 的系统级优势**：关于 I/O 模块当前限制的讨论非常热烈。成员们主张引入像 `io_uring` 这样的**异步 API**，旨在提升网络性能。
   - **Darkmatter__** 和 **Lukashermann.com** 辩论了强大但复杂的 API 与用户友好型抽象之间的权衡，强调了可维护性的必要性。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 的小失误**：根据[公告](https://discord.com/channels/1091220969173028894/1092729520181739581/1257248877497552948)，由于**数据库操作失误**，**OpenRouter** 的分析功能暂时下线。客户数据保持安全，未受此次事故影响。
   - 团队迅速向用户表示歉意，澄清客户额度未受损，并正在积极处理数据修复。
- **DeepSeek Code 攻克微积分**：通过 OpenRouter API 使用的 **DeepSeek Code V2** 在处理微积分问题时表现出令人印象深刻的准确性，这一消息分享在 [general 频道](https://discord.com/channels/1091220969173028894/1094454198688546826/1256356791814455306)中。其经济高效的特性受到了关注。
   - 确认所使用的模型是 **full 263B** 版本，这表明它在各种任务中具有相当大的能力和通用性。详情可见 [DeepSeek-Coder-V2 页面](https://openrouter.ai/models/deepseek/deepseek-coder)。
- **Mistral API 混淆问题**：有报告称在 Aider 聊天中使用 **Sonnet 3.5** 时出现了 **Mistral API 错误**，这让当时并未使用 Mistral 的用户感到困惑。
   - 用户被引导联系 Aider 的支持团队进行具体排查，这暗示在服务中断期间可能触发了自动回退到 Mistral 的机制。详情在 [general 频道](https://discord.com/channels/1091220969173028894/1094454198688546826/1256356791814455306)中讨论。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Adept 与 Amazon 的结盟**：Adept AI Labs 宣布了战略更新和领导层变动，联合创始人将前往 Amazon，详情见其 [blog post](https://www.adept.ai/blog/adept-update) 和 [GeekWire article](https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/)。此举旨在让 Adept 在保持独立性的同时，通过非独占许可使用 Amazon 的技术。
   - 社区反馈指出，最初的博客文章引起了困惑，引发了关于**与 Amazon 合作伙伴关系性质**的讨论，促使读者倾向于阅读 [GeekWire article](https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/) 以获取更清晰的见解。
- **关于 AIEWF 的坦诚反馈**：AI Engineer World's Fair (AIEWF) 的组织者和参与者进行了反馈会议，讨论如何改进会议时长和物流方面，参考了关于经验教训和未来规划的 [GitHub discussions](https://github.com/swyxio/swyxdotio/issues/510)。
   - 建议包括延长活动时间或进行更结构化的反馈，并呼吁设立专门的 Hackathon 空间，灵感来自其他会议在促进深入讨论方面的成功经验。
- **Runway 的视频生成之旅**：Runway 发布了他们的 Gen-3 Alpha Text to Video 功能，这是高保真和可控视频生成领域的重大进步，已在他们的 [official account](https://x.com/runwayml/status/1807822396415467686) 上宣布。
   - 该功能面向所有人开放，承诺在视频生成技术上实现重大飞跃，可通过 Runway 的 [website](http://runwayml.com) 访问，激发了创作者的好奇心和实验热情。
- **Prompt 规划中的隐私问题**：出现了关于 GPT system prompts 隐私的讨论，强调应将 prompts 视为潜在的公开信息，参考了 [GitHub](https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts) 上的示例。
   - 建议避免在 GPT 系统定义中包含敏感数据，社区支持这一观点，建议对 prompts 中分享的内容采取谨慎态度。
- **CoALA 论文引起关注**：社区讨论了关于 Cognitive Architectures for Language Agents (CoALA) 的新论文，该论文可在 [arXiv](https://arxiv.org/abs/2309.02427) 上找到，它引入了一个组织现有 Language Agent 模型的框架。
   - 基于 CoALA 论文的 Language Agent 仓库 [awesome-language-agents](https://github.com/ysymyth/awesome-language-agents) 成为那些希望深入研究 Language Agent 的人的重点资源。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Web 编年史：React 与 FastAPI 的结合**：一位公会成员询问了如何将 **LangGraph** 与 React 前端和 FastAPI 后端集成，并获得了指向 [GitHub 上的 chat-langchain](https://github.com/langchain-ai/chat-langchain) 和 [Semantic Router 文档](https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb) 的指引。
   - 他们收到了关于创建 Agent 或利用语义相似性进行路由的建议，并参考了为工具实现奠定坚实基础的概述方法。
- **Embedding 嵌套博弈：通过 Matryoshka 加速**：Prashant Dixit 展示了一种使用 **Matryoshka RAG** 和 **llama_index** 提升检索速度的解决方案，详情见 [Twitter 帖子](https://x.com/Prashant_Dixit0/status/1806580075447590974) 和一份详尽的 [Colab 教程](https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/tutorials/RAG-with_MatryoshkaEmbed-Llamaindex/RAG_with_MatryoshkaEmbedding_and_Llamaindex.ipynb)。
   - 该技术采用了多样的 Embedding 维度（从 768 到 64），承诺增强性能并提高内存效率。
- **智能自动化与分析：EDA-GPT 革命**：Shaunak 宣布了 **EDA-GPT**，这是一个使用 **LLMs** 进行自动化数据分析的 GitHub 项目，可通过 **Streamlit** 部署，并在[项目的 README](https://github.com/shaunthecomputerscientist/EDA-GPT) 中提供了设置视频教程。
   - 这一创新简化了数据分析过程，为工程师优化了工作流。
- **Postgres 遇见 LangChain：持久化的绝佳组合**：Andy Singal 的 [Medium 文章](https://medium.com/ai-advances/unleashing-the-power-of-persistence-langchain-meets-postgres-9cc7f069b260) 强调了将 **LangChain** 与 **Postgres** 结合以优化持久化，将 Postgres 的可靠性引入 LangChain 项目。
   - 这一协同组合旨在利用 Postgres 稳健的存储能力来加强状态管理。
- **在 LangChain 中施展 MoA 魔法**：一段名为 [“使用 langchain 实现 Mixture of Agents (MoA)”](https://www.youtube.com/watch?v=VNy7CM23WA0) 的 YouTube 教程向观众演示了在 LangChain 中创建多 Agent 系统的过程，旨在提升任务性能。
   - 该视频为 MoA 提供了一个切入点，并为有兴趣应用组合 Agent 优势的工程受众提供了具体的代码示例。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Adept 进入 Amazon 版图**：**Adept** 转变策略并与 **Amazon** 整合；联合创始人加入 *Amazon*，详见其 [博客文章](https://www.adept.ai/blog/adept-update)。公司在人员流失后，目前约有 20 名员工留守。
   - **Adept** 的这一战略举措类似于 **Microsoft** 对 Inflection AI 的策略，引发了关于 **Adept 的发展方向**和**组织文化**变化的猜测，如[这条推文](https://x.com/anissagardizy8/status/1806812006009442671?s=46)中所讨论的。
- **AI 的精通还是迷局？**：关于 **AI Agent 开发**是否滞后的辩论正在进行，有人将其类比为**早期阶段的自动驾驶汽车**。像 **Multion** 这样的项目因其在**基础网页抓取**之外的能力提升微乎其微而受到批评。
   - 社区推测，创新的**数据收集方法**是 **AI Agent** 的游戏规则改变者，转向生成**高质量、模型特定数据**是克服当前局限性的关键环节。
- **Cohere 对 AI 清晰度的巧妙尝试**：**Cohere 的 CEO** **Aidan Gomez** 在一次 [YouTube 讨论](https://youtu.be/B45s_qWYUt8?si=qs1u6p7wiXFP46PT)中分享了关于对抗 **AI 幻觉**和提升**推理能力**的见解，暗示了合成数据生成的潜力。
   - 社区将这些努力与 **Generative Active Learning** 以及针对 LLM 的 *hard negative/positive mining*（难负/正样本挖掘）实践进行了比较，呼应了视频中 **5:30** 和 **15:00** 处的重要性。
- **模型估值难题**：拥有丰富用户的 **Character.ai** 估值为 **10 亿美元**，而尚未投入运营的 **Cognition AI** 估值却高达 **20 亿美元**，这引发了围绕**融资宣讲能力**和**筹款技巧**的讨论。
   - **Cognition AI** 打着创始人获得 **IMO（国际数学奥林匹克）奖项**的旗号，瞄准开发者群体，但在来自**大厂 AI 实体**的激烈竞争中，其价值正面临审查。
- **外行的 RL 飞跃**：一位 AI 爱好者完成了 **Silver 的 RL 入门**和 **Abeel 的 Deep RL** 课程，下一步目标是 **Sutton & Barto**，并寻找任何关于 **LM alignment**（语言模型对齐）的非传统建议。
   - RL 新手们得到的建议是浏览 **Spinning Up in Deep RL** 并尝试真实的**代码库**，进行手动的 *CPU 驱动任务*以获得扎实的理解，或许可以参考 **HF Deep RL 课程**。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **跨操作系统安装历程**：成员们反映在 **macOS 和 Windows** 上设置 `01` 时存在摩擦，尽管遵循了预设步骤，但仍面临 **API key** 依赖和终端命令混淆等障碍。[@killian](https://www.reddit.com/r/LocalLLaMA/comments/1dl3a13/killian_showed_a_fully_local_computercontrolling/) 和其他人提出了潜在的解决方案，大家对安装过程中的阻碍达成了普遍共识。
   - 针对文档清晰度的持续呼吁贯穿了多次尝试，正如一个讨论 **桌面应用冗余** 的线程所强调的那样。GitHub 上的一个 Pull Request 带来了希望，[dheavy](https://github.com/OpenInterpreter/01/pull/203) 展示了简化的 Windows 操作流程。
- **健忘的助手？解决 AI 遗忘问题**：关于赋予 **Open Interpreter** 更好的 **长期记忆** 的询问浮出水面，指出了 **Sonnet** 模型从过去交互中学习能力的痛点。
   - 关于记忆增强的讨论反映了大家对 **OI 记忆限制** 的集体尝试，但尽管有关于特定命令使用和深奥预训练尝试的建议，目前仍缺乏明确的高级策略。
- **征集向量搜索先锋**：一位积极的成员在 [Colab notebook](https://colab.research.google.com/github/onefact/loving-the-baseline/blob/main/nearest-neighbors.ipynb) 中展示了将向量搜索集成到公共数据集的动手教程，为在 Fed 的前沿演示奠定了基础。
   - 该合作者为进一步的 **向量搜索增强项目** 伸出了橄榄枝，预示着社区创新和应用 AI 研究可能开启新篇章。
- **多模态模型搜寻升温**：关于为受限和非受限项目选择顶级开源多模态模型的咨询不断涌现，促使了如 **Moondream**（用于视觉精细化）配合 **强健的 LLMs** 的建议。
   - 讨论导致了对模型充分性的碎片化观点，反映了对多模态实现策略的多样化看法，目前尚未出现统一的胜出者。
- **Windows 难题与安装波动**：对 **OpenInterpreter** 在 Windows 上 **typer 安装麻烦** 的不满情绪蔓延，成员们通过微调 pyproject.toml 文件并操作 `poetry install` 来获得成功。
   - 文档问题的叙述贯穿了整个社区，放大了对透明、**最新指南** 的呼声，并引发了对其 **01 Light** 设置实用性的审查。[@Shadowdoggie](https://github.com/OpenInterpreter/01) 强调了 macOS 的便捷与 Windows 的困扰之间的两极分化。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Reranker 革命**：随着 **Jina reranker** 新版本的发布，兴奋感在蔓延，据称这是他们目前最有效的版本，详见[此处](https://t.co/YsYoVOIirb)。社区赞扬了它对检索策略和 **结果组合** 的影响。
   - 由 **@kingzzm** 分享的构建自定义混合检索器的指南因其结合检索方法的详尽方式而受到欢迎，详见[此处](https://t.co/cTxW2UwuZ0)。反馈显示它在高级 **检索流水线** 方面具有深远的潜力。
- **LlamaIndex 工具包扩展**：随着用户思考 **Langchain Tools** 与 **LlamaIndex** 的兼容性，集成不确定性随之增加，这一询问由一位社区成员提出。
   - 围绕在 **LlamaIndex agents** 中使用 **Langchain Tools** 的讨论非常热烈，大家对如何合并它们的功能以提高效率表现出浓厚兴趣。
- **查询困惑得以解决**：用户正在努力解决 LlamaIndex 中的 **查询流水线配置**，并提出了诸如利用 kwargs 管理 **`source_key`** 以改进输入分离和检索设置的见解。
   - 针对大型 CSV 文件的 Embedding 性能担忧，促使了一项提高 **`embed_batch_size`** 的提议，从而为引入更大型的 LLMs 以进行更好的代码评估拓宽了路径。
- **子 Agent 专业化**：对 **子 Agent** 的好奇心显现，用户寻求通过提示词和输入来自定义它们，以增强特定任务的操作。
   - **CodeSplitter** 工具因其在优化元数据提取方面的潜力而受到关注，暗示了 **LlamaIndex** 内部向更高效的节点操作转变。
- **Kubernetes 与多 Agent 协同**：由 **@_nerdai_** 推出的全新 **多 Agent 系统部署** 入门套件，为轻松将本地 Agent 服务迁移到 Kubernetes 铺平了道路，详见[此处](https://t.co/wfcI0wSmFG)。
   - 该套件实现了向 k8s 部署的无缝过渡，标志着在 **扩展服务能力方面迈出了重要一步**。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **以少胜多的 Llama**：社区成员对 **llama3 70b** 模型仅凭 270 亿参数就实现高性能感到震惊，引发了关于这种壮举可行性的讨论。与此同时，一些人仍坚持使用 **Mixtral**，因其性能全面且对消费级硬件的许可协议友好。
   - **Hugging Face** 上展开了关于 **Hermes 2 Theta** 和 **Pro** 迭代版本的辩论——一个是新颖的实验，另一个是经过打磨的 finetune——同时用户也在思考 **Pro** 版本独有的结构化 JSON 输出的价值。
- **格式化的魅力与挫折**：**Axolotl** 自定义 ORPO 格式化程序的问题引发了讨论，原因是 tokenization 不当以及 ChatML 中系统角色（system roles）的管理方式。
   - 使用替代角色来应对挑战的建议遇到了冲突方面的担忧，这表明需要更无缝的自定义解决方案。
- **合成模型减速引发猜测**：**Nvidia synthetic model** 因其数据生成速度缓慢而受到关注，与 **llama 70b** 或 **GPT-4** 等更快的模型相比，其速度慢如蜗牛。
   - 这引发了关于小模型可能具有的优势的询问，特别是在效率和实际应用方面。
- **优化中的前沿压缩技术**：AI 爱好者探讨了创新的内存高效优化器，如 **CAME** 和 **Adam-mini**，它们承诺在不牺牲性能的情况下减少内存占用。
   - 技术发烧友被引导至 [CAME 的论文](https://arxiv.org/abs/2307.02047)和 [Adam-mini 的研究](https://arxiv.org/abs/2406.16793)，以深入了解细节及其在 Stable Diffusion 训练等领域的潜在应用。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **深陷队列泥潭的 Benchmark**：热门话题包括影响 **leaderboard benchmark** 排队时间的计算资源瓶颈，推测源于 HF 的基础设施。**Stellaathena** 暗示无法控制队列，表明需要替代方案。
   - [@dimfeld](https://x.com/dimfeld/status/1806116419995844947) 建议将 **vllm** 作为替代方案，并指向了一个[有用的 wiki](https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm)，提供模型内存优化的技术指导。
- **Im-Eval 思考 'HumanEval' 能力**：关于 **im-eval** 处理 `HumanEval` 和 `HumanEvalPlus` 能力的问题浮出水面，**Johnl5945** 发起了关于为该评估工具配置评估温度（evaluation temperatures）的讨论。
   - 对话在没有明确结论的情况下结束，凸显了对 im-eval 功能和温度控制进行后续研究或澄清的潜在领域。
- **Adam-mini：轻量级优化器**：[Adam-mini 优化器](http://arxiv.org/abs/2406.16793)是一个值得关注的主题，它通过使用分块式的单一学习率提供了显著的内存节省，并承诺具有与 AdamW 相当的性能。
   - 成员们评估了它的功效，认识到在不影响模型结果的情况下缩减优化器内存占用的潜力，这可能会在 ML 工作流中引入更具内存效率的实践。
- **Gemma 2 的指标令用户困惑**：尽管努力遵循了推荐做法（如将 `dtype` 设置为 `bfloat16`），但在复现 **Gemma 2** 指标时的差异仍导致了困惑。人们对 **piqa** 和 **hellaswag** 等 benchmark 中报告的准确率存在实质性差异表示担忧。
   - 在正确的调试命令似乎返回了正确但不一致的结果后，人们敦促进一步调查潜在问题，正如 [@LysandreJik 的一条推文](https://x.com/LysandreJik/status/1807779464849273343)中所述。
- **揭秘 Token 表示的“擦除效应”**：最近的一项[研究](https://arxiv.org/abs/2406.20086)揭示了 LLM 中 token 表示的“擦除效应”（erasure effect），在多 token 命名实体中尤为显著，引发了围绕其影响的激烈讨论。
   - 学术交流集中在这种效应如何影响语义复杂的 token 组的解释，以及旨在解决这一表示挑战的增强型模型设计的潜力。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **代码库清理大行动**：George Hotz 发起了精简 **tinygrad** 的行动号召，要求将 **RMSNorm** 从 **LLaMA** 移至 `nn/__init__.py`，并配齐测试和文档。
   - 社区对此做出了响应，建议对组织结构进行增强，并可能在整个项目中统一代码标准。
- **tinygrad 周一例会**：在最近的会议中，参与者讨论了多项进展，包括 [sharding 更新](https://github.com/tinygrad/tinygrad/pull/5123) 和 [单次图重写 (single pass graph rewrite)](https://github.com/tinygrad/tinygrad/pull/5159/files#diff-91ca5e2e75ef3ea1982c8ca6cc175ee88f20efa0d8e4b96f305b970dc6df71e7R291)，并涉及了 **tensor cores** 和新的悬赏任务。
   - 详细交流涵盖了 **lowerer continuation**、**Qualcomm runtime**，并确定了进一步改进 tinygrad 开发流程的后续步骤。
- **独立 tinygrad 讨论**：有疑问提出关于将 **tinygrad** 程序编译为适用于 Raspberry Pi 等设备的独立 C 代码的可能性，大家对针对低功耗硬件的开发表现出共同兴趣。
   - 成员们分享了 [tinygrad for ESP32](https://github.com/wozeparrot/tinygrad) 等资源，以激发在传统环境之外的应用探索。
- **悬赏任务亮点**：一场深入的讨论明确了 **llama 70b lora 悬赏** 的要求，包括遵循 [MLPerf 参考实现](https://github.com/mlcommons/training/tree/master/llama2_70b_lora)，但在计算方法上保持灵活性。
   - 社区探讨了使用 qlora 的可能性，并分享了在不同硬件配置上实现该悬赏任务的见解。
- **图重写新进展**：关于 **图重写 (graph rewrite)** 的交流包括对在流程中采用新算法的兴趣，重点在于优化调度器 (scheduler)。
   - **ChenYuy** 对会议的总结指出，虽然尚未选定具体的图算法，但将更多功能迁移到图重写框架中的势头正盛。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **27B 模型：大质疑与小胜利**：在褒贬不一的评价中，**27B 模型**面临质疑，但也因其潜力获得了一些认可，甚至在某些表现不佳的场景下也超过了 **Command R+**。一位成员指出 **Gemma 2 9B** 表现出了出人意料的优越性能，从而加强了讨论。
   - 这种带有疑虑的热情在讨论中蔓延，如 *“确实，但即使在最坏的情况下（-15），它也比 command r+ 更好”* 这种观点表明，人们更倾向于 27B 模型相对于同类产品的优越潜力。
- **ChatGPT 走下坡路，沦为二流**：关于 **ChatGPT 4 和 4o 模型** 在处理细微编程任务时失去掌控力的担忧被提出，用户更倾向于 **3.5 版本**。几位用户觉得最新的模型过于“死板”，对提示词理解得太字面化。
   - 挫败感在蔓延，一位成员评论道 *“有时付费的 4 和 4o 模型在编程时感觉完全没用”*，这捕捉到了社区向更可靠的**免费替代方案**转移的趋势。
- **Gemini 崛起，ChatGPT 衰落**：**Gemini 1.5 Pro** 因其响应迅速的交互而备受瞩目，而 **ChatGPT** 则面临效率日益降低的投诉，尤其是在编程任务中。用户纷纷称赞 Gemini 积极解决问题的态度，与 ChatGPT 逐渐减退的热情形成鲜明对比。
   - 用户如是评价：*“与 ChatGPT 日益严重的懒惰相比，Gemini 1.5 pro 做得非常出色”*，这凸显了用户正转向那些能长期保持活力和参与度的替代模型。
- **Claude 的 Artifact 功能大获好评**：**Claude 的 artifact 功能** 赢得了用户的青睐，提供了比 **ChatGPT** 建立的现状更具沉浸感且高效的体验。这一特定功能吸引了越来越多的受众准备转换阵营。
   - 社区共识反映在 *“artifacts 功能体验好得多”* 等陈述中，标志着更符合用户体验预期的发烧友工具正日益流行。
- **语言迷宫与 LLM**：讨论转向全球受众，**非英语母语者**正在寻找擅长多种语言的 LLM，他们优先考虑母语的对话能力，而非特定任务的效率。尽管模型在特定任务中的有效性各异，但这种全球化的倾向仍在继续。
   - *“它之所以能排在前面，不是因为它能解决难题，而是因为它的多语言能力”* 这一说法展示了对推动语言包容性和支持本地化用户交互的模型的需求激增。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **JSONL 处理者集结**：简化 JSONL 编辑的努力促成了经验分享，包括一个用于快速编辑的 [simple JSONL editor](https://github.com/aastroza/simple-jsonl-editor)，以及一套用于处理各种 JSON 结构化文件的脚本方法。
   - 社区交流建议直接编辑和使用 Prompt Engineering 从 JSON 格式的结构化患者数据中提取摘要，避免引入新的包实现，并最大限度地减少 LLM 评估中的幻觉发生率。
- **Kv-Caching 启发 Vision-LLMs**：LLM 用户探索了针对视觉中心模型的 [kv-caching 增强方案](https://sachinruk.github.io/blog/2024-06-29-kv-cache.html)，发现预测概率有了显著提升。
   - 该指南为受限 GPU 设置下的视觉模型提供了可操作的优化建议，吸引了实际关注和实现反馈。
- **在 Kubernetes 上进行 LLM 推理**：对在 **Kubernetes** 上实现 ML 推理存在质疑，一条轻松的推文引发了关于 ML 工作负载替代云基础设施的深入讨论。
   - 尽管在 **Modal** 上的工具扩展存在一些 [共同困难](https://github.com/modal-labs/llm-finetuning)，但在特定的分布式系统中，人们对 Modal 的信心高于 Kubernetes。
- **Hugging Face 积分：用户的慰藉**：关于 **Hugging Face 积分** 的说明已经发布，确认有效期为 2 年，缓解了用户对积分立即过期的担忧。
   - 讨论指出社区需要更好的关于 **Hugging Face** 积分状态和管理的沟通渠道。
- **追求最优 IDE**：[Zed IDE](https://zed.dev) 凭借出色的功能和 AI 集成赢得了 Sublime Text 老用户的青睐，但人们对 **Cursor** 的功能仍保持高度好奇。
   - 社区征求关于 **Cursor** 用户体验的反馈，建议更广泛地探索开发环境中的 AI 集成。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 培养实习生**：一位渴望获得 Cohere 实习机会的社区成员提到了他们的 AAAI 会议论文和过去的实习经历，并询问了 **Cohere API** 的进展和新型 LLM 功能任务。
   - 对话围绕将 LLM 与强化学习结合的资源展开，丰富了 **Cohere** 的技术版图。
- **Coral 的烦恼：速率限制之谜**：用户对 **Coral API** 的速率限制感到沮丧，抱怨每分钟仅 5 次调用的严格限制。
   - 通过一份 [实用指南](https://docs.cohere.com/docs/rate-limits) 交流了心得，生产环境密钥（Production Key）以每分钟 10,000 次调用的慷慨额度成为了救星。
- **Aya-23 模型的混杂信息**：社区在 **Aya-23** 模型的版本迷雾中穿行，重点关注了 [Hugging Face](https://huggingface.co/CohereForAI/aya-23-8B) 上的 8B 和 35B 模型，同时追踪着关于 9B 变体的虚假传闻。
   - 共识澄清目前没有这些模型版本用于运行推理的应用，并重申了它们的充分性。
- **Cohere 迈向更清晰的认知**：成员们热议 **Cohere** 遏制 AI 幻觉的计划，此前 [Aidan Gomez 的分享](https://youtu.be/B45s_qWYUt8?si=_c7sQUFUN6bZa61m) 探讨了如何增强 AI 推理能力。
   - CEO 的路线图并未涉及外部合作，而是强调自主开发。
- **Rig 解决 Cohere 兼容性**：随着 **Rig** 库宣布与 **Cohere 模型** 全面集成， Rust 爱好者们欢欣鼓舞，并鼓励通过 [奖励评审计划](https://bit.ly/Rig-Feeback-Form) 提供反馈。
   - 贡献者可以通过提供改进 Rig 的见解来获得 100 美元的酬金，使其成为 LLM 驱动项目的核心组件。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **深入探讨数据驱动精度的 DPO**：针对拥有稳健数据的数据集进行 **DPO training** 的益处引发了辩论，一些人考虑立即使用 **DPO/PPO**，而另一些人则表示 **hesitation**。建议在这些应用中使用 **PreferenceDataset**。
   - 讨论强调 **Socket experts** 应该指导此类决策，并引用了过去在 **llama2** 和 **Pythia 1B-7B** 上进行直接 **DPO/PPO** 训练的成功案例。
- **WandB 助力 Phi Mini 微调**：一位 AI 爱好者成功微调了 **Phi Mini (LoRA)** 模型，并寻求关于 **evaluating logs** 的指导。共识是采用 **WandBLogger** 进行高级日志管理和可视化。
   - 有人对 **yaml configuration** 的陷阱提出了警告，并强调了设置良好的 **WandBLogger** 对于防止错误和增强训练监督的重要性。
- **微调细节：日志与梯度治理**：技术讨论涉及了 **gradient size** 的适当性，并建议根据数据集的具体情况进行调整。分享的日志引发了对 **overfitting** 迹象的审查，以及关于延长 **training epochs** 的讨论。
   - 日志显示了 **loss and learning rate** 指标的异常，特别是在较小的数据集中，这突显了像 **WandB** 这样的工具在微调过程中提供清晰度的效用。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Featherless 以固定费用起飞**：最近推出的 **Featherless.ai** 提供基于订阅的访问权限，可以访问 Hugging Face 上提供的所有 LLM 模型，起价为每月 10 美元，无需本地 GPU 设置，详见[此处](https://featherless.ai/)。该平台在本地 AI 角色应用（如 Sillytaven）以及语言微调和利用 SQL 模型的需求方面，使用量有所增加。
   - **Text-to-Speech (TTS) 的诱惑**：随着用户要求增强在线游戏中 NPC 语音多样性的呼声日益增高，**Featherless.ai** 考虑集成像 [Piper](https://github.com/rhasspy/piper/blob/master/VOICES.md) 这样的 TTS 系统，同时保持对本地 CPU 设置无法运行的热门模型的关注。
- **带有 WSL README 的 Windows 智慧**：新成员 Niashamina 通过创建一个使用 **WSL** 在 Windows 上运行 AI Town 的 **README**，为公会带来了 **Windows wisdom**，并提到了在不久的[将来](https://github.com/)集成 Docker 的可能性。
   - 虽然 Docker 的集成仍在进行中，且 README 的草案正等待在 GitHub 上首次亮相，但 Niashamina 调侃了它最终的实用性，暗示了他们正在开拓的动手实践式 **Windows progress**。
- **Hexagen.World 的新地理瑰宝**：一个简短但值得注意的公告揭晓了 [Hexagen.World](https://Hexagen.World) 提供的 **fresh locations**，扩展了该领域的虚拟景观。
   - 此次发布并未深入细节，但为那些倾向于探索新增虚拟地形的人播下了好奇的种子，开启了通往 **new localizations** 的窗口。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Facebook 打造编译器能力：LLM Compiler 模型**：Facebook 发布了其 **LLM Compiler 模型**，具备编译 C、优化汇编和 LLVM IR 的能力，现已由 Mozilla 便捷地打包成[适用于各种操作系统的 llamafiles](https://github.com/Mozilla-Ocho/llamafile)。
   - 支持 AMD64 和 ARM64 架构的 **llamafile** [已由 Mozilla 上传至 Hugging Face](https://huggingface.co/Mozilla/llm-compiler-13b-ftd-llamafile)，以提高其用户的可访问性。
- **Llamafile 迈向官方化：Hugging Face 上的集成历程**：为了让 llamafile 在 Hugging Face 上获得官方地位，贡献者们准备创建 Pull Request 来更新 [model libraries](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts) 和相应的 [code snippets](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries-snippets.ts) 文件。
   - 此次集成将通过在支持 **llamafile** 的仓库中添加按钮，并自动填充代码以无缝加载模型，从而优化用户体验。
- **技术规格揭秘：llamafile 的硬件要求**：社区讨论了在各种设备上运行 **llamafile** 的可行性；然而，它需要 **64-bit system**，使得 Raspberry Pi Zero 无法参与其中。
   - 虽然 **llamafile server v2.0** 的内存占用极低（使用 **all-MiniLM-L6-v2.Q6_K.gguf** 为 HTTP 客户端托管 Embedding 仅需 23mb），但对 iPhone 13 的支持仍未确认。
- **llamafile v0.8.9 登陆 Android：Gemma2 获得支持**：**llamafile v0.8.9** 发布，带来了官方 Android 兼容性，并完善了对 Google Gemma2 架构的支持，同时修复了 Windows GPU 提取问题。
   - [新发布的 v0.8.9 版本](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.9)还强调了服务器模式操作的进展，并巩固了 Google Gemma v2 的增强功能。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **聚焦 AI 工程领域的混乱**：一场备受期待的 **'From ML Engineering to AI(cntEngr)' 活动录像** 变成了一场数字捉迷藏，成员们遇到了 [无效的 Zoom 链接](https://events.zoom.us/ejl/AuK2GHiGErlhKuJWYRGND-S4fQfnvCgfPO50QMGDk0trhQ2ykP5H~A2Qw5SEBU-CKEpNn-eBw) 和访问码问题。
   - 尽管做出了努力，社区仍无法获取录像，这凸显了 AI 工程领域在活动共享基础设施方面的差距。
- **流水线魔法研讨会奇迹**：[Data Talks Club 即将举行的 Zoomcamp](https://lu.ma/cnpdoc5n?tk=uEvsB6) 为 AI 工程师承诺了一场实战之旅，重点是使用 **dlt** 和 **LanceDB** 构建 **开源数据流水线**，计划于 7 月 8 日举行。
   - 在来自 **dltHub** 的 Akela Drissner 指导下，参与者将深入研究 REST APIs、数据向量化和编排工具，旨在将流水线部署到包括 Python notebooks 和 Airflow 在内的各种环境中。



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **数据库主导地位的竞争加剧**：关于数据库的发展，**dbreunig** 强调了自 **5 月 19 日** 开始的一个显著趋势，显示竞争对手正在缩小与数据库技术领先者的差距。
   - 评论表明 AI 数据库格局正在发生变化，多个参与者正在进步并争夺领先地位。
- **计算领域的追赶态势**：在 **dbreunig** 最近的见解中，自 **5 月 19 日** 以来的数据表明领先的数据库竞争者之间的竞争日益激烈。
   - 这一观察结果指出了一个关键时期，即竞争技术开始显示出显著增长，正在追赶行业领导者。



---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **陷入内存困境的程序员**：一位工程师尝试在 50 GB 的海量语料库上训练 **Panjabi 语言的 BPE tokenizer**，但在拥有 1TB RAM 的机器上遭遇了 **OOM 问题**。他们通过分享类似的 [GitHub issues](https://github.com/huggingface/tokenizers/issues/1434) 揭示了这一困境。
   - 尽管 **Pre-processing sequences 步骤在超出 len(ds) 后仍在继续**，内存消耗却持续飙升，这暗示了 `train_from_iterator` 函数可能存在故障，详见此 [相关 issue](https://github.com/huggingface/tokenizers/issues/1345)。针对这一棘手问题，目前迫切需要技术见解或替代训练方法。
- **调试困境：深入 Rust 底层**：为了破解 BPE tokenizer 训练过程中的 OOM 谜团，一位勇敢的程序员陷入了僵局，因为 `tokenization_utils_fast.py` 中的 `train_from_iterator` 函数变得像一座无法攻破的堡垒。
   - 有推测认为问题可能源于可执行/二进制 Rust 代码，这一理论得到了社区其他遭遇者的支持，这让我们的工程师百思不得其解并寻求 [专家援助](https://github.com/huggingface/transformers/blob/e65502951593a76844e872fee9c56b805598538a/src/transformers/tokenization_utils_fast.py#L817)。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1257065010312646818)** (1 条消息): 

- **使用 ESM3-SM-open-v1 预测蛋白质序列**：一位认证用户分享了一个用于预测掩码蛋白质序列及其结构的 **ESM3-SM-open-v1** 演示 [Space](https://huggingface.co/spaces/as-cle-bert/proteins-with-esm)。
   - *演示 Space 中重点展示了预测掩码蛋白质序列的细节。*
- **BPE Tokenizer 可视化工具上线**：介绍了由另一位社区成员创建的 **BPE Tokenizer** [可视化工具](https://github.com/mdabir1203/BPE_Tokenizer_Visualizer-)。
   - *该可视化工具使理解和使用 BPE tokenization 变得更加容易。*
- **terminusresearch 发布令人兴奋的新数据集**：terminusresearch 发布了 [人物持物数据集](https://huggingface.co/datasets/terminusresearch/photo-anatomy)、[建筑数据集](https://huggingface.co/datasets/terminusresearch/photo-architecture) 以及 [美学数据集](https://huggingface.co/datasets/terminusresearch/photo-aesthetics)。
   - *这些数据集为各个领域的研究和开发提供了多样化的选择。*
- **探索 Fast whisper playground**：社区成员分享了一个 [Fast whisper playground](https://huggingface.co/spaces/Iatalking/fast-whisper-server)，用于快速实验。
   - *用户现在可以在这个新的交互式设置中快速测试 whisper 模型。*
- **Gemma 2 27B 测试结果**：一段 YouTube 视频讨论了用户分享的 [Gemma 2 27B](https://youtu.be/vIKNRiVxWeo) 测试结果。
   - *该视频探讨了 Google 最新发布的 Gemma 版本的性能。*

**提到的链接**：<a href="https://youtu.be/vIKNRiVxWeo)">Gemma2:27B 首次测试！怎么会这么差？！</a>：让我们使用 ollama 测试 Google 一小时前发布的最大的 gemma2 版本 (27B)。

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1256323280126738562)** (952 条消息 🔥🔥🔥):

- **模型下载与缓存问题**：一位用户在下载 Falcon40B 时遇到问题，发现模型文件在下载后消失了。事实证明，模型文件由 HuggingFace 缓存，可以使用 `snapshot_download` 函数进行访问。
   - 另一位用户建议使用更高效的模型（如 Llama 3 8B）来替代 Falcon 40B，认为这样可以在消耗更少资源的情况下获得更好的性能。
- **HuggingFace 平台上的 Spaces 卡住问题**：一位用户的 Space 卡在“preparing”状态超过一天。尽管尝试了重新构建和重启，问题依然存在。
   - 版主建议通过重启或暂停/取消暂停 Space 来解决问题；然而，问题仍未解决，需要进一步的支持。
- **法律 LLM 与 RAG 实现**：一位用户就构建用于法律咨询的 LLM 寻求建议，涉及处理 Token 大小和幻觉（hallucinations）问题。建议包括使用检索增强生成（RAG）以实现更有效的数据利用。
   - 讨论强调了使用中间 LLM 评估批量生成输出以进行质量排序的重要性，并建议避免对可能过时的法律数据进行频繁的微调（fine-tuning）。
- **使用 Whisper 和 Wav2Vec 进行音频分类**：关于使用 Whisper 和 Wav2Vec 模型进行音频分类的问题被提出，特别是围绕音频录音的分段。
   - 建议将音频分成 15-30 秒的区块以获得更好的模型性能，并在处理不同操作系统时使用特定的格式。
- **Gradio 和 Spaces 脚本中的 Token 管理**：一位用户需要帮助改进一个使用 Gradio 和 HuggingFace Spaces 将模型转换并推送到 Hub 的脚本安全性。
   - 人们对脚本中安全管理 HF Token 表示担忧，从而提出了改进异常处理和 Token 重置实践的建议。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/xyplon/text-to-image-models-playground">Text To Image Models Playground - xyplon 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/aheedsajid/Edge-TTS">Edge TTS - aheedsajid 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/hub/models-download-stats">模型下载统计</a>: 未找到描述</li><li><a href="https://huggingface.co/dev-mode-explorers">dev-mode-explorers (开发模式探索者)</a>: 未找到描述</li><li><a href="https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents">为什么我们不再使用 LangChain 来构建我们的 AI Agent</a>: 当抽象弊大于利时 —— 在生产环境中使用 LangChain 的教训以及我们本该怎么做</li><li><a href="https://sachinruk.github.io/blog/2024-06-29-kv-cache.html">Prompt 缓存：穷人版的零样本视觉-LLM 分类指南 – deepschool.ai</a>: 使用 KV caching 和 logit 比率来加速和控制 LLM/ VLM 输出。</li><li><a href="https://huggingface.co/posts">Hugging Face – 帖子</a>: 未找到描述</li><li><a href="https://huggingface.co/nerijs/pixel-art-xl">nerijs/pixel-art-xl · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-mistral">在你的数据集上微调 Mistral</a>: 未找到描述</li><li><a href="https://huggingface.co/VAP36/EltonJohn70s/resolve/main/Ej1979.zip">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/TheStinger/UVR-Test/discussions/1">TheStinger/UVR-Test · 更新 requirements.txt</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/81">zero-gpu-explorers/README · 带有 ZeroGPU Space 的 Pro 账户：“你已超出 GPU 配额（剩余 59s，请求 60s）。请在 0:00:56 后重试”</a>: 未找到描述</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/models?pipeline_tag=image">模型 - Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/cat-cat-memes-cat-images-cat-meme-gif-4644773688486402896">Cat Cat Memes GIF - Cat Cat 表情包 Cat 图片 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.google.it/search?q=tensorflow+transformer">tensorflow transformer - Google 搜索</a>: 未找到描述</li><li><a href="https://tenor.com/view/soldier-ww2-traumatized-meme-eyes-gif-12257475272172704406">二战士兵 GIF - 二战士兵创伤 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/drake-notebook-gif-20708336">Drake 笔记本 GIF - Drake 笔记本 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/coBTzvQsPpQ?si=PgVtXG5hw_XgjpBX">逐步讲解 RAG，直到 GROKKED RAG 系统</a>: 今天我尝试回答订阅者关于我最后三个视频的所有问题，重点关注新的 Grokked LLM 与传统 RAG 系统的集成。我...</li><li><a href="https://github.com/huggingface/transformers/issues/31293>">Issues · huggingface/transformers</a>: 🤗 Transformers: 面向 Pytorch、TensorFlow 和 JAX 的前沿机器学习。- Issues · huggingface/transformers</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE">LICENSE · meta-llama/Meta-Llama-3-8B (main 分支)</a>: 未找到描述</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5">nomic-ai/nomic-embed-text-v1.5 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.tensorflow.org/text/tutorials/transformer#define_the_components">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/Vipitis/shadertoys-dataset">GitHub - Vipitis/shadertoys-dataset: 数据集重构进行中 (WIP)</a>: 数据集重构进行中 (WIP)。通过在 GitHub 上创建账户来为 Vipitis/shadertoys-dataset 的开发做出贡献。</li><li><a href="https://huggingface.co/nroggendorff/mayo-7b-it">nroggendorff/mayo-7b-it · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/carwash-car-cat-gif-13432273130992663014">洗车猫 GIF - 洗车猫 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/intone/Ammino-1.1B">intone/Ammino-1.1B · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Cf_aWv4LyUQam80gNHVRygNUtneovVmp#scrollTo=ua9MQA3WXCtr">Google Colab</a>: 未找到描述</li><li><a href="https://tenor.com/bA1GjIzjUL.gif">披甲龙龟 OK GIF - Rammus Ok Okay - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1256350504708083795)** (7 条消息): 

- **Huggingface 课程更新状态**：**Huggingface courses** 似乎在定期维护，与 2022 年 5 月版的 NLP 书籍相比，可能提供更新的内容。关于 **Diffusion 课程和社区计算机视觉课程** 的最新状态仍不清楚。
   - 建议用户直接从 [Huggingface 官网](https://huggingface.co/course) 验证课程内容，以确保获取最新的更新和信息。
- **使用 2D 摄像机的生物特征步态识别**：**使用 2D 摄像机的步态识别**在 23 人中基于单帧识别个人的准确率达到了 70%。接下来的步骤包括寻找更多数据集、结合多帧以供 RNN 使用，以及使用 **triplet loss** 进行训练以生成 **embeddings**。
   - 鼓励对 **Triplet Collapse** 和高级步态识别方法等领域感兴趣的成员通过私信作者参与协作和知识共享。
- **适用于 Garmin 设备的自定义 ML 库**：一个令人兴奋的项目正在进行中，旨在使用 Monkey C 语言编写的自定义 **ML** 库在 **Garmin** 设备上克隆 **Apple 的双击功能**。该项目处于初始阶段，重点是开发专门为 Garmin 硬件量身定制的这一功能。
   - 邀请 Monkey C 或类似项目的合作者和专家分享见解或加入开发过程。
- **理解 Tree-sitter S-Expressions**：通过其 [官方文档](https://tree-sitter.github.io/tree-sitter/using-parsers#pattern-matching-with-queries) 和各种编程语言中的绑定库探索 **Tree-sitter s-expressions**。开发者可以使用 **node-tree-sitter** 或 **tree-sitter Rust crate** 来扩展功能。
   - **Tree-sitter 项目** 通过 **C APIs** 和高级绑定支持多种语言，使用户能够将强大的解析功能集成到他们的应用程序中。
- **引人入胜的 ML 概念讲解**：一位成员赞扬了 Ashpun 对 **attention 模块** 的解释，强调了此类知识在 **ML 面试** 中的重要性。理解这些概念可以激发新想法并改进大语言模型 (**LLMs**)。
   - Ashpun 对反馈表示感谢，并强调了清晰、详细的解释在 **ML** 概念的学习和应用中的重要性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">1 分钟介绍前 10 名深度学习算法</a>：欢迎深入了解前 10 名深度学习算法！在本视频中，我们用简洁的 10 个词解释来分解每个算法。非常适合...</li><li><a href="https://tree-sitter.github.io/tree-sitter/using-parsers#pattern-matching-with-queries">Tree-sitter｜使用解析器</a>：未找到描述
</li>
</ul>

</div>

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1256856303767130233)** (11 条消息🔥): 

- **Kickstarter 项目融合动漫与 AI**：分享了一个专注于利用 AI 创作动漫和漫画艺术作品的 Kickstarter 项目。更多详情可以在其 [Kickstarter 页面](https://www.kickstarter.com/projects/mobinet-games/creating-anime-and-manga-artwork-with-ai)查看。
   - 该项目承诺提供创新的艺术生成技术，对于动漫和漫画设计中 AI 赋能的创意开发来说，这可能是一个令人兴奋的进展。
- **Firecrawl 将网站转换为数据**：**Firecrawl** 是一款开源工具，可将网站转换为干净的 Markdown 或结构化数据，初始提供 500 个免费额度。他们宣布在 GitHub 上获得了 [7000+ stars](https://github.com/mendableai/firecrawl)。
   - 该工具无需站点地图即可抓取所有可访问的子页面，为 LLM 应用转换数据。示例和功能在其 [官网](https://www.firecrawl.dev/) 有详细说明。
- **SDXL 的 Lora 模型在 HuggingFace 发布**：一个用于 SDXL 的 **Lora 模型** 在 HuggingFace 上发布，其特色是 TOK 风格的独特生物提示词。在 [HuggingFace](https://huggingface.co/alvdansen/m3lt) 探索更多创意输出和提示词。
   - 该模型受到了社区的好评，用户对该作品表示赞赏，并引用了相关模型如 `alvdansen/BandW-Manga`。
- **Langchain 与 Postgres 集成**：分享了一篇讨论 **Langchain** 与 **Postgres** 集成时持久化能力的文章。如需深入阅读，请查看这篇 [Medium 文章](https://medium.com/ai-advances/unleashing-the-power-of-persistence-langchain-meets-postgres-9cc7f069b260)。
   - 这种集成实现了改进的数据管理和检索，使需要稳健数据库解决方案的 AI 工作流受益。
- **AI 探索网格生成技术**：一段名为“AI 刚刚搞定了网格（Meshes）”的 [YouTube 视频](https://www.youtube.com/watch?v=rQolOT4tuUY&ab_channel=IndividualKex) 详细介绍了 AI 在网格生成方面的进展。链接中包含原始论文和 HuggingFace 上的 Demo。
   - 视频解释了网格生成的能力和应用，并附带了 [代码](https://github.com/buaacyw/MeshAnything) 和研究资料以供深入探索。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/alvdansen/m3lt">alvdansen/m3lt · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=rQolOT4tuUY&ab_channel=IndividualKex">AI just figured out Meshes</a>: 原始论文: https://huggingface.co/papers/2406.10163 demo: https://huggingface.co/spaces/Yiwen-ntu/MeshAnything code: https://github.com/buaacyw/MeshAnythi...</li><li><a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">Top 10 Deep Learning Algorithms intro in 1 min</a>: 欢迎来到前 10 名深度学习算法的深入探讨！在本视频中，我们用简洁的 10 个词解释来拆解每个算法。非常适合...</li><li><a href="https://getdoks.org/">Build an amazing docs site</a>: 构建一流文档网站所需的一切。快速、易用且易于上手。</li><li><a href="https://mintlify.com/">Mintlify - The modern standard for documentation</a>: 未找到描述</li><li><a href="https://www.firecrawl.dev/">Firecrawl</a>: 将任何网站转换为 LLM 就绪的数据。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1256396266070474824)** (17 条消息🔥): 

- **用于生物研究的 esm3-sm-open-v1 模型**：[Evolutionary Scale](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) 发布了 *esm3-sm-open-v1* 模型，该模型被描述为生物学领域的前沿生成模型，能够跨蛋白质的三个基本生物学特性进行推理。该模型可在 [GitHub](https://github.com/evolutionaryscale/esm) 上获取，并可通过 Hugging Face 上用户构建的 Space 进行体验。
   - 鼓励用户通过访问 [此链接](https://huggingface.co/spaces/as-cle-bert/proteins-with-esm) 的 Space 来尝试将该模型用于研究，创建者称其“令人耳目一新”。
- **发布内部美学数据集**：[Terminusresearch](https://huggingface.co/datasets/terminusresearch/photo-aesthetics) 发布了其内部美学数据集之一，包含 33.1k 张从 Pexels 筛选出的具有特定品质的真实照片图像。该数据集旨在帮助微调用于美学判断的模型。
   - 数据集的更多组件也已发布，包括人们手持物品的图像和用于正则化数据的小型建筑集，希望能提高模型在相关任务中的性能。
- **用于 LLM 的 BPE Tokenizer 可视化工具**：一个新的 [BPE Tokenizer Visualizer](https://github.com/mdabir1203/BPE_Tokenizer_Visualizer-) 已创建，旨在帮助可视化 BPE Tokenizer 在 LLM 中的工作方式。该可视化工具的演示可在 [此处](https://screenrec.com/share/SV7cw9vryx) 查看。
   - 创建者正在寻求修复问题的帮助以及来自社区的反馈，以改进该工具。
- **在机器人硬件上轻松运行 Transformer**：[Embodied Agents](https://github.com/mbodiai/embodied-agents) 项目允许仅用几行 Python 代码在机器人硬件上运行 Transformer 模型。该工具旨在无缝集成到机器人技术栈中。
   - GitHub 页面提供了详细的说明和示例代码，供用户快速上手。
- **Stable Cypher Instruct 3B 模型发布**：[Stable Cypher Instruct 3B](https://huggingface.co/lakkeo/stable-cypher-instruct-3b) 是新发布的 3B 参数模型，旨在在生成 CYPHER 查询方面超越 GPT-4o 等 SoA 模型。它是 stable-code-instruct-3b 的微调版本，专门在来自 Neo4j Labs 的合成数据集上进行了训练。
   - 该模型可通过 Hugging Face 访问，旨在促进 Neo4j 等 GraphDB 数据库的 text-to-CYPHER 查询生成。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://marketing.startyparty.dev/">startyparty</a>：一切的开始</li><li><a href="https://huggingface.co/blog/alvdansen/enhancing-lora-training-through-effective-captions">通过有效的标注增强图像模型 Dreambooth 训练：关键观察</a>：未找到描述</li><li><a href="https://huggingface.co/lakkeo/stable-cypher-instruct-3b">lakkeo/stable-cypher-instruct-3b · Hugging Face</a>：未找到描述</li><li><a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">1 分钟介绍前 10 名深度学习算法</a>：欢迎来到我们的前 10 名深度学习算法深入探讨！在本视频中，我们用简洁的 10 个词解释来分解每个算法。非常适合...</li><li><a href="https://huggingface.co/blog/alvdansen/training-lora-m3lt">我如何训练 LoRA：m3lt 风格训练概述</a>：未找到描述</li><li><a href="https://github.com/U-C4N/UMBOT">GitHub - U-C4N/UMBOT</a>：通过在 GitHub 上创建账户，为 U-C4N/UMBOT 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/as-cle-bert/proteins-with-esm">Proteins With Esm - 由 as-cle-bert 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/mbodiai/embodied-agents">GitHub - mbodiai/embodied-agents: 将最先进的 Transformer 模型无缝集成到机器人技术栈中</a>：将最先进的 Transformer 模型无缝集成到机器人技术栈中 - mbodiai/embodied-agents</li><li><a href="https://huggingface.co/datasets/terminusresearch/photo-aesthetics">terminusresearch/photo-aesthetics · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/terminusresearch/photo-anatomy">terminusresearch/photo-anatomy · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/mdabir1203/BPE_Tokenizer_Visualizer-">GitHub - mdabir1203/BPE_Tokenizer_Visualizer-: 用于检查 LLM 中 BPE Tokenizer 如何工作的可视化工具</a>：用于检查 LLM 中 BPE Tokenizer 如何工作的可视化工具 - mdabir1203/BPE_Tokenizer_Visualizer-</li><li><a href="https://screenrec.com/share/SV7cw9vryx">24.05.2024_00.18.02_REC</a>：使用 ScreenRec 录制
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1256355527202701332)** (30 messages🔥): 

- **LLM 推理引发成员关注**：成员们讨论了论文 [Reasoning with LLMs](https://arxiv.org/pdf/2405.16506)，该论文因在检索适配中应用 GNN 而受到关注。
   - 组织了一次会议来深入探讨 [LLM 推理的当前研究](https://github.com/atfortes/Awesome-LLM-Reasoning) 和 [符号推理 (symbolic reasoning)](https://github.com/luban-agi/Awesome-LLM-reasoning)，其中包括[详细的报告](https://isamu-website.medium.com/understanding-the-current-state-of-reasoning-with-llms-dbd9fa3fc1a0)和 [YouTube 演示](https://www.youtube.com/watch?v=vbji1PvXgBc&ab_channel=IsamuIsozaki)。
- **Discord 技术问题导致切换至 Zoom**：在会议期间，成员们遇到了 Discord 音频的技术困难。作为权宜之计，会议切换到了 [Zoom](https://drexel.zoom.us/j/86571034095)，从而解决了问题。
   - 成员们推测问题可能源于客户端版本不匹配，但一致认为在功能正常时，Discord 是一个极佳的通话平台。
- **Terminator 架构有望实现快速训练**：[Terminator 架构的代码](https://github.com/hyperevolnet/Terminator)已发布，因其训练收敛的高效率而受到推崇。
   - 测试表明，Terminator 可以在 50-100 个 epochs 内取得理想结果，显著少于其他架构，有望缩短训练时间。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=vbji1PvXgBc&ab_channel=IsamuIsozaki">Hugging Face Reading Group 24: Understanding Current State of Reasoning with LLMs</a>: 演讲者: Isamu Isozaki。往期演示: https://github.com/isamu-isozaki/huggingface-reading-group</li><li><a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">Top 10 Deep Learning Algorithms intro in 1 min</a>: 欢迎来到前 10 名深度学习算法的深入探讨！在本视频中，我们用简练的 10 个词解释来分解每个算法。非常适合...</li><li><a href="https://github.com/hyperevolnet/Terminator">GitHub - hyperevolnet/Terminator: The official repository for HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks for Full Context Interaction.</a>: HyperZ⋅Z⋅W 算子连接慢速-快速网络以实现全上下文交互的官方仓库。</li><li><a href="https://github.com/atfortes/Awesome-LLM-Reasoning">GitHub - atfortes/Awesome-LLM-Reasoning: Reasoning in Large Language Models: Papers and Resources, including Chain-of-Thought, Instruction-Tuning and Multimodality.</a>: 大语言模型中的推理：论文和资源，包括 Chain-of-Thought、指令微调和多模态。</li><li><a href="https://drexel.zoom.us/j/86571034095">Join our Cloud HD Video Meeting</a>: Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。</li><li><a href="https://drexel.zoom.us/j/8657103">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1257175065590763662)** (1 messages): 

- **Transformer 的低精度推理**：我们调查了在 **SD3** 和 **PixArt-Sigma** 等 **基于 Transformer 的流水线 (transformer-based pipelines)** 中使用低精度推理的情况。一些有趣的发现已在 [GitHub 讨论串](https://github.com/huggingface/diffusers/discussions/8746)中展开讨论。
   - 该讨论强调了在这些模型中实现低精度推理的潜在性能提升和挑战。
- **探索 Transformer 流水线**：对 **SD3** 和 **PixArt-Sigma** 等基于 Transformer 的流水线进行了详细分析。见解和技术细节可在 [讨论串](https://github.com/huggingface/diffusers/discussions/8746)中查阅。
   - 关键点包括低精度推理的优势及其对模型性能和效率的影响。
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1256491748583280763)** (12 messages🔥): 

- **手势绘图应用需要改进**：一位用户分享了一个 [YouTube 视频](https://youtu.be/QIjB4tqLqcs?si=bHfDg3WuRn5rEYbs)，展示了他们使用 **OpenCV** 和 **Mediapipe** 将**手势转化为数字艺术**的项目。他们征求了改进建议。
   - 关于改进应用的建议较少，但几位成员讨论了**目标尺寸（object size）和卷积层配置（convolutional layer configuration）**在模型性能中的重要性。
- **ViTMAEForPretraining 设置挑战**：用户讨论了将 **ViTMAEForPretraining** 用于自定义图像的问题，并对技术挑战表示担忧。一位用户分享的代码显示了 **mask_ratio 设置**以及拆除编码器进行推理时遇到的问题。
   - 该实现看起来比较**粗糙（janky）**且存在不确定性，需要在预训练和推理期间的模型配置方面采用更精细的方法。
- **由于 Numpy 版本导致的 face_recognition 模块错误**：关于 **face_recognition** 模块错误的讨论暗示了由 **numpy** 版本引起的问题。一位用户建议回退到 2.0 以下的版本以解决该问题。
   - 经确认这可以解决问题，为遇到类似问题的其他用户提供了一个快速的变通方案。
- **利用计算机视觉构建 Agentic 医院**：一位来自悉尼和海得拉巴的医生分享了关于 **agentic 医院**的愿景，即利用**计算机视觉**来减轻行政负担。他们的目标是在集装箱中使用**鱼眼摄像头（fisheye cameras）**来简化操作并改善患者护理。
   - 他们请求计算资源以帮助实现这一愿景，强调了对高效且技术驱动的医疗解决方案的需求。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/QIjB4tqLqcs?si=bHfDg3WuRn5rEYbs">🎨 手势绘图应用演示 - 艺术创作进行中！ 🖐️✍️</a>：观看我尝试最新的项目，该项目使用 OpenCV 和 Mediapipe 将手势转化为数字艺术。加入我，看我实时创作（或尝试创作）一些东西...</li><li><a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">1 分钟介绍前 10 名深度学习算法</a>：欢迎来到我们对前 10 名深度学习算法的深入探讨！在这个视频中，我们用简短的 10 个词解释来分解每个算法。非常适合...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1256362763849171057)** (9 messages🔥): 

- **GEC 预测列表急需帮助**：一位成员表达了关于其 **GEC 预测列表** 形状不匹配（out of shape）的紧急问题，并请求立即提供建议。
   - 该成员强调了情况的紧迫性，寻求社区的帮助。
- **前 10 名深度学习算法视频**：一位用户分享了一个名为 **"Top 10 Deep Learning Algorithms intro in 1 min"** 的 [YouTube 视频](https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC)，该视频为每个算法提供了简短的 10 词解释。
   - 该视频旨在提供深度学习算法的快速概览，但因在多个频道**跨频道发布（cross-posting）**而收到了警告。
- **寻求 LLM 实操学习方法**：一位成员询问了参与 **LLM 实操学习** 的有效方法。
   - 针对该询问，目前还没有直接的建议或资源。
- **在 Transformer 模型中加入 RBF 层**：一位用户询问是否有人尝试过将 **Transformer 模型** 内部的一个层替换为 **RBF 层**。
   - 该用户未收到任何回复或进一步的讨论。

**提到的链接**：<a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">1 分钟介绍前 10 名深度学习算法</a>：欢迎来到我们对前 10 名深度学习算法的深入探讨！在这个视频中，我们用简短的 10 个词解释来分解每个算法。非常适合...

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1257186167829434408)** (10 messages🔥): 

- **如何在 Diffusers 中触发 LoRA**：一位用户询问如何在使用 **Diffusers** 时触发 **LoRA**，并提到他们已经加载了 LoRA 但没有效果。
   - 另一位用户回答说 **Diffusers** 不会解析 prompt，你需要手动加载 **LoRA 权重**。
- **LoRA 权重不起作用的问题**：用户分享了他们使用 `text2imgPipe` 加载 LoRA 权重并设置适配器权重的代码，但指出该代码在 A1111 中可以工作。
   - 另一位用户指出问题可能是由于将权重设置为了 **0.0**。原用户澄清说其中一个 LoRA 权重实际上设置为了 0.5。
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1256326241364217886)** (1031 messages🔥🔥🔥):

- **WSL2 上的 Unsloth 设置**：用户讨论了在 WSL2 上设置 Unsloth，包括最大化资源利用率的命令和配置。分享了一些针对安装问题的排障步骤和教程。
   - 多位用户分享了他们的配置和修复方案，例如在 `.wslconfig` 中将内存设置为 0 以避免限制 RAM，以及使用更新的安装命令。讨论内容包括在 Intel 和 AMD 系统上运行 Unsloth。
- **Unsloth 中的多 GPU 支持和 DRM**：讨论了 Unsloth 用于多 GPU 支持的新 DRM 系统，包括 GPU ID 固定和唯一 GPU ID 的持久性。该 DRM 系统正在严格的 NDA 下进行测试，展示了早期访问阶段以及在平衡许可和灵活性方面的努力。
   - 用户对使用 Unsloth 进行多 GPU 训练表示了兴趣，包括配置和潜在限制。分享了进度更新和可能的发布时间表，强调了稳定 DRM 实现的重要性。
- **微调和数据集准备的挑战**：提出了关于微调不同模型（包括 Lexi 和 Gemma）的问题。用户分享了集成系统 Token 以及处理微调后出现无限生成等挑战的方法。
   - 讨论内容包括将数据集翻译成不同语言以及保持未经过滤（uncensored）的训练数据等技术。建议了数据集策划和避免灾难性遗忘（critical forgetting）的最佳实践。
- **有效使用 AI 工具和平台**：讨论了 Runpod 和 Ollama 等各种 AI 工具和平台，包括它们的优势以及如何将它们集成到现有工作流中。用户注意到了用于训练的额度可用性以及租用计算资源的实用性。
   - 讨论了用于部署模型和翻译数据集以获得更好性能的自动化流水线。对比了本地方案和基于 API 的方案，以突出效率和成本效益。
- **人格驱动的数据合成及其应用**：介绍了一种人格驱动（persona-driven）数据合成的新方法，利用 LLM 中的多样化视角来大规模创建合成数据。提到了包含 10 亿个人格的 Persona Hub，以促进这一过程。
   - 该方法的应用包括为数学推理、用户提示词和游戏 NPC 等各种场景创建高质量数据集。强调了其对 LLM 研究和实际应用的潜在影响。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://foleycrafter.github.io/">FoleyCrafter</a>：未找到描述</li><li><a href="https://www.scmp.com/lifestyle/gadgets/article/3268063/china-made-play-dreams-headset-cheaper-higher-resolution-apple-vision-pro-clone">Play For Dream 混合现实头显：一款改进的 Apple Vision Pro 克隆产品</a>：它看起来可能与 Apple Vision Pro 一模一样，但这款来自上海 Play For Dream 的头显提供了更高分辨率的视觉效果，佩戴更舒适，并且可以运行 Android 应用。</li><li><a href="https://docs.continue.dev/walkthroughs/tab-autocomplete#:~:text=The%20models%20that%20we%20suggest%20for%20autocomplete%20are%20trained%20with%20a%20highly%20specific%20prompt%20format%2C%20which%20allows%20them%20to%20respond%20to%20requests%20for%20completing%20code">Tab 自动补全 (beta) | Continue</a>：Continue 现在支持在 VS Code 和 JetBrains IDE 中进行 Tab 自动补全。我们将在接下来的几个版本中大幅改进体验，听到反馈总是很有帮助。如果 ...</li><li><a href="https://ar5iv.labs.arxiv.org/html/2304.06035">Choose Your Weapon: Survival Strategies for Depressed AI Academics</a>：未找到描述</li><li><a href="https://huggingface.co/google/recurrentgemma-2b-it">google/recurrentgemma-2b-it · Hugging Face</a>：未找到描述</li><li><a href="https://www.numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction">NuExtract：用于结构化提取的基础模型 - NuMind</a>：我们推出了 NuExtract，这是一个轻量级的文本转 JSON LLM。NuExtract 允许从文本中提取任意复杂的信息，并将其转换为结构化数据。该模型可以直接用于 ze...</li><li><a href="https://huggingface.co/CohereForAI/aya-101">CohereForAI/aya-101 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/sweaty-sweat-heat-hot-wipe-sweat-gif-17716050">Sweaty Heat GIF - Sweaty Sweat Heat - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/fireworks-ai/llama-3-firefunction-v2">fireworks-ai/llama-3-firefunction-v2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Kearm/UnslothAIWorldFair">Kearm/UnslothAIWorldFair · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/failspy/Llama-3-8B-Instruct-MopeyMule">failspy/Llama-3-8B-Instruct-MopeyMule · Hugging Face</a>：未找到描述</li><li><a href="https://share.hsforms.com/1tvg18CtoSH6EYna-eQzpgAecykq">表单</a>：未找到描述</li><li><a href="https://github.com/MC-E/ReVideo">GitHub - MC-E/ReVideo</a>：通过在 GitHub 上创建账号，为 MC-E/ReVideo 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/NeuralNovel/Unsloth-DPO">NeuralNovel/Unsloth-DPO · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/SafeAILab/EAGLE">GitHub - SafeAILab/EAGLE：EAGLE 的官方实现</a>：EAGLE 的官方实现。通过在 GitHub 上创建账号，为 SafeAILab/EAGLE 的开发做出贡献。</li><li><a href="https://github.com/b4rtaz/distributed-llama">GitHub - b4rtaz/distributed-llama：张量并行（Tensor parallelism）就是你所需要的一切。在性能较弱的设备上运行 LLM，或者通过分配工作负载和划分 RAM 使用量，让强大的设备变得更加强大。</a>：张量并行就是你所需要的一切。在性能较弱的设备上运行 LLM，或者通过分配工作负载和划分 RAM 使用量，让强大的设备变得更加强大。 - b4rtaz/distributed-llama</li><li><a href="https://github.com/camenduru/FoleyCrafter-jupyter">GitHub - camenduru/FoleyCrafter-jupyter</a>：通过在 GitHub 上创建账号，为 camenduru/FoleyCrafter-jupyter 的开发做出贡献。</li><li><a href="https://rocm.docs.amd.com/en/latest/">AMD ROCm™ 文档 — ROCm 文档</a>：未找到描述</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth：微调 Llama 3, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80%</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/microsoft/WSL2-Linux-Kernel">GitHub - microsoft/WSL2-Linux-Kernel：用于 Windows Subsystem for Linux 2 (WSL2) 的 Linux 内核源码</a>：用于 Windows Subsystem for Linux 2 (WSL2) 的 Linux 内核源码 - microsoft/WSL2-Linux-Kernel</li><li><a href="https://github.com/unslothai/unsloth/pull/708">darkacorn 提交的 yaml 和 cli · Pull Request #708 · unslothai/unsloth</a>：应该可以开箱即用 - seb 请审阅/ daniel 和 mike 请对该想法发表评论，因为它非常简单直接 - 中央配置 vs 过度更改文件 - 并且通过 c... 更易于维护</li><li><a href="https://ai.meta.com/research/cicero/diplomacy/">未找到标题</a>：未找到描述</li>

nd</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Qwen2-1.5b">Replete-AI/Replete-Coder-Qwen2-1.5b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/proj-persona/PersonaHub">proj-persona/PersonaHub · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1drk3kc/gemma_2_betrayed_us/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1WHlelph_UAPksquDgsZ2Q-H6VUrAjS7H?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://x.com/arankomatsuzaki/status/1807593343007818065?s=46">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Scaling Synthetic Data Creation with 1,000,000,000 Personas - 展示了从网络数据中自动策划的 10 亿个多样化 Persona 集合 - 在 MATH 数据集上取得巨大进步：49.6 -> 64.9 仓库：https://g...</li><li><a href="https://arxiv.org/abs/2406.20094">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>: 我们提出了一种新颖的 Persona 驱动的数据合成方法，利用大语言模型 (LLM) 中的各种视角来创建多样化的合成数据。为了充分利用这种方法...</li><li><a href="https://www.bigscreenvr.com/">no title found</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues?q=wsl">Issues · unslothai/unsloth</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi 和 Gemma LLM - Issues · unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: 我在原生 Windows（无 WSL）上运行了 Unsloth。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有完整的安装教程，我本想在这里全部写出来，但我现在在用手机...</li><li><a href="https://github.com/unslothai/unsloth/blob/933d9fe2cb2459f949ee2250e90a5b610d277eab/unsloth/tokenizer_utils.py#L962">unsloth/unsloth/tokenizer_utils.py at 933d9fe2cb2459f949ee2250e90a5b610d277eab · unslothai/unsloth</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi 和 Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/933d9fe2cb2459f949ee2250e90a5b610d277eab/unsloth/models/llama.py#L1199">unsloth/unsloth/models/llama.py at 933d9fe2cb2459f949ee2250e90a5b610d277eab · unslothai/unsloth</a>: 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi 和 Gemma LLM - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1256945644120178688)** (14 条消息🔥): 

- **Meta 内部发布 Llama 400**：一篇 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1ds2p09/llama_400_released_internally_at_meta_available) 讨论了 Meta 内部发布的 **Llama 400** 及其在 WhatsApp 上的可用性，并附带了一张据称是 WhatsApp 的截图。讨论集中在这些信息是否暗示该模型具有顶尖性能。
   - 成员们表达了好奇心并对高性能寄予厚望，其中一位成员指出：*'如果分数很高（至少前三名），那么我希望如此。'* 另一位成员也表示赞同。
- **LMSYS 模型类型澄清**：讨论透露 **LMSYS** 有两个可用模型：'base' 和 'instruct'。**用户 'edd0302'** 和另一位成员在交流中确认了这一信息。
   - 似乎存在性能预期的比较，用户 **'mahiatlinux'** 同意 LMSYS 模型的多样性及其潜在的高分。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1ds2p09/llama_400_released_internally_at_meta_available/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ds2p09/llama_400_released_internally_at_meta_available">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1256350984800829491)** (170 条消息🔥🔥): 

- **预训练与微调的区别详解**：一位用户询问了持续预训练（continued pretraining）与微调（finetuning）之间的区别，提到预训练包含的目标模块更多。另一位用户建议，如果是针对另一种语言进行预训练，则需要额外的模块，并参考了 TrellisResearch 的 [YouTube 视频](https://youtu.be/GZqYr8_Q7DE)。
   - 该视频解释了在针对不同语言进行微调时，训练 embedding 和 LM head 为何如此重要。
- **预训练后的推理错误**：一位用户在预训练后的推理过程中遇到错误，怀疑可能是 16GB T4 GPU 的显存问题。另一位用户建议查看 [GitHub issue](https://github.com/unslothai/unsloth/issues/702) 以寻找类似问题。
   - 有建议指出，该问题可能是由于新版本的 PyTorch 导致的。
- **微调 Llama3 模型**：一位用户询问了关于微调 Llama3 的 instruct 版本，并对聊天模板（chat template）和映射（mapping）表示担忧。建议使用基础模型（base models）进行微调，而不是 instruct 模型。
   - 其他建议包括尝试不同的 tokenizer 和格式化函数，以确保与训练数据的兼容性。
- **处理数据集格式错误**：一位用户在将 ShareGPT 格式的数据集上传到 Hugging Face 时遇到问题，并在加载数据集时报错。另一位用户建议确保格式为 `jsonl`，并提供了[示例格式](https://huggingface.co/docs/datasets/loading_datasets.html)。
   - 建议使用 Hugging Face 库读取数据集，然后再推送到 Hub，以避免错误。
- **在 AMD 和 Windows 上运行 Unsloth**：一位用户询问了在 AMD GPU 和 Windows 上运行 Unsloth 的问题，遇到了与 nvidia-smi 和 AMD 驱动相关的错误。解决方案包括使用不同的安装方法并确保驱动程序正确初始化。
   - 进一步的讨论涉及在向 Hugging Face 推送大型模型时处理限制和兼容性问题。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/tomasonjo/text2cypher-qwen2-72b-4bit-gguf/tree/main">tomasonjo/text2cypher-qwen2-72b-4bit-gguf at main</a>: 未找到描述</li><li><a href="https://lightning.ai/lightning-ai/studios/train-a-gpt-classifier-from-scratch?section=tutorials">Train a GPT Classifier from Scratch - a Lightning Studio by sebastian</a>: 该 Studio 提供了一个 Jupyter notebook，解释了如何从头开始微调 GPT 模型，以 96% 的准确率对垃圾短信进行分类</li><li><a href="https://github.com/unslothai/unsloth/issues/711">ROCm + WSL2 incompatibility · Issue #711 · unslothai/unsloth</a>: AMD 的 HIP/ROCm WSL2 驱动目前不使用内核模块。/home/musclez/ComfyUI/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:691: UserWarning: Can&#39;t initialize amdsm...</li><li><a href="https://github.com/unslothai/unsloth/issues/702#issuecomment-2197477362">Cache only has 0 layers, attempted to access layer with index 0 · Issue #702 · unslothai/unsloth</a>: 在尝试使用 unsloth 库训练 Phi-3 时遇到 KeyError。错误发生在 model.generate 的生成步骤中。以下是代码和错误的详细信息...</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: 使用 unslothai/unsloth 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLM</li><li><a href="https://github.com/unslothai/unsloth/wiki#fin">Home</a>: 使用 unslothai/unsloth 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLM</li><li><a href="https://huggingface.co/datasets/xingyaoww/opendevin-code-act">xingyaoww/opendevin-code-act · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://docs.vllm.ai/en/latest/models/lora.html">Using LoRA adapters &#8212; vLLM</a>: 未找到描述</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 使用 unslothai/unsloth 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLM</li><li><a href="https://colab.research.google.com/drive/1bd_VkH4aszEVvzuRXNtFNU70yWf38UJE?usp=sharing">Google Colab</a>: 未找到描述</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1257009330092507207)** (2 messages): 

- **使用 GGUF 高效修复 OSS 模型 bug**：**Llama-3** 等 OSS 模型中的常见 bug 包括 *Double BOS tokens 问题*和导致 *NaNs 的 Untrained tokens*。修复方法包括使用 **GGUF 的 CPU 转换**而非 GPU 来正确处理 BOS tokens。
   - 通过适当处理特定 tokens，可以减轻这些计算错误，确保模型性能更平稳。
- **轻松安装 Unsloth 进行 LLMs 微调**：要安装 *Unsloth*，请使用特定包创建一个 Conda 环境并激活该环境。安装命令为：`pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`。
   - 遵循这些步骤可确保获得一个高效微调 LLMs 的就绪环境。
- **Unsloth 加速模型训练**：使用 *Unsloth* 配合 Huggingface 的 TRL 库，[UnslothAIWorldFair](https://huggingface.co/datasets/Kearm/UnslothAIWorldFair) 模型的训练速度提升了 **2 倍**。
   - 该模型遵循 **Llama 3 Community License**，参数量为 8.03B，并使用 **BF16 张量类型**。
- **Qwen2-Wukong-0.5B 引入 chat 微调**：[Qwen2-Wukong-0.5B](https://huggingface.co/RESMPDEV/Qwen2-Wukong-0.5B) 是原始 Qwen2-0.5B 模型的**去对齐（dealigned）chat 微调版**。
   - 在 **teknium OpenHeremes-2.5 数据集**上训练了三个 epoch，其在数据分类方面的表现优于 1.1B TinyLlama 微调版。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/RESMPDEV/Qwen2-Wukong-0.5B">RESMPDEV/Qwen2-Wukong-0.5B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Kearm/UnslothAIWorldFair">Kearm/UnslothAIWorldFair · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Kearm/UnslothAIWorldFair">Kearm/UnslothAIWorldFair · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1257120870552571965)** (10 messages🔥): 

- **10 亿个 Personas 驱动数据合成**：[Persona Hub](https://github.com/tencent-ailab/persona-hub) 引入了一种新颖的 persona 驱动的数据合成方法论，利用多样化的 personas 从网络数据中策划的 10 亿个 personas 中创建合成数据。这种方法显著提升了 MATH 分数，从 **49.6** 提高到 **64.9**。
   - 社区讨论了这种合成数据创建的影响，强调了[摘要](https://arxiv.org/abs/2406.20094)中展示的多功能性和易用性。一位用户指出，“这看起来非常重要，但没有代码”，另一位则建议“复制这个并不难，数据比代码重要得多”。
- **数据与代码之争**：成员们在 Persona Hub 的背景下辩论了数据与代码的重要性。虽然一位用户对缺乏代码表示失望，但其他人认为数据才是关键要素。
   - 讨论突显了社区对于复制和利用此类大规模项目所需核心组件的不同观点，诸如“数据比代码重要得多”之类的评论反映了这一点。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/arankomatsuzaki/status/1807593343007818065?s=46">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Scaling Synthetic Data Creation with 1,000,000,000 Personas  - Presents a collection of 1B diverse personas automatically curated from web data - Massive gains on MATH: 49.6 -&gt;64.9  repo: https://g...</li><li><a href="https://arxiv.org/abs/2406.20094">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>: We propose a novel persona-driven data synthesis methodology that leverages various perspectives within a large language model (LLM) to create diverse synthetic data. To fully exploit this methodology...
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1256323607676719146)** (332 messages🔥🔥):

- **Gemma 2 GPU Offloads - 目前受限**：一位成员询问了关于 Gemma 2 GPU offload 的支持情况，但目前仅支持 **Cuda** 和 **Metal**。此外，还强调了 0.2.23 之后的版本需要进行更新以修复相关问题。
   - 在某些配置下，加载 Gemma 2 模型的时间过长，LM Studio 社区已收到更新请求，并将其列为正在进行的修复项目。
- **LM Studio 中的 AutoUpdater 已重构**：LM Studio 的 **AutoUpdater** 现在已恢复工作，确保用户可以轻松更新到 **0.2.26** 版本。社区对即将发布的 0.3 版本表示期待。
   - 讨论强调，最近的迭代应该会解决一些问题，特别是关于 **LLama 3 8B model** 消除其持久存在的忽略 stop sequence 的问题。
- **Deepseek v2 与 CPU 性能**：探讨了 Deepseek v2 的性能，指出尽管它拥有 200B+ 参数，但只有 21B 被激活，这意味着在高性能 CPU 上其性能是可控的。报告提到在配备大容量内存的高端 Threadripper 系统上，速度可达 3-4 tokens/sec。
   - **共享用户测试**：用户分享了他们的实测性能指标，包括 RAM 使用情况、模型加载和生成速度，揭示了实际的能力和局限性。
- **LM Studio 中的新实验性 Quants**：对于像 **q5_k_l** 这样旨在减少内存占用同时保持输出质量的实验性量化，反馈褒贬不一。**GGUF quants** 因其在性能和效率之间的平衡而被特别提及。
   - 社区反馈和测试仍在继续，旨在完善这些量化方案，以便更广泛地采用并实现持续的性能提升。
- **Gemma 2 的 Sliding Window Attention 已合并**：针对 **Gemma 2** 的 **Sliding Window Attention** 功能已合并到最新的 **llama.cpp** 中。这旨在通过有效处理过去的 tokens 来增强上下文理解，从而提高模型性能。
   - 尽管有此更新，一些用户仍注意到存在质量问题，预计将进行进一步调查和更新以全面解决这些疑虑。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6">未找到标题</a>: 未找到描述</li><li><a href="https://learn.microsoft.com/en-us/windows/wsl/install">安装 WSL</a>: 使用命令 `wsl --install` 安装 Windows Subsystem for Linux。在你的 Windows 机器上使用由你偏好的 Linux 发行版（Ubuntu, Debian, SUSE, Kali, Fedora, Pengwin...）运行的 Bash 终端。</li><li><a href="https://www.youtube.com/watch?v=l8pRSuU81PU&">让我们复现 GPT-2 (124M)</a>: 我们从零开始复现 GPT-2 (124M)。这段视频涵盖了整个过程：首先我们构建 GPT-2 网络，然后优化其训练以使其真正...</li><li><a href="https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF">lmstudio-community/gemma-2-9b-it-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=iOdFUJiB0Zc">微调 LLM 模型 – 生成式 AI 课程</a>: 学习如何微调 LLM 模型。本课程将教你使用 QLORA 和 LORA 进行微调，以及使用 LLama2, Gradient 等进行 Quantization...</li><li><a href="https://huggingface.co/thethinkmachine">thethinkmachine (Shreyan Chaubey)</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式和示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://unsloth.ai">Unsloth AI | 微调 Llama 3 &amp; Mistral LLMs</a>: 为 AI 和 LLM 提供简单的微调。开源且适合初学者。使用 Unsloth 变得更快。 </li><li><a href="https://huggingface.co/Joseph717171">Joseph717171 (Joseph)</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio CLI。使用 TypeScript/Node 编写</a>: LM Studio CLI。使用 TypeScript/Node 编写。通过在 GitHub 上创建账号来为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://huggingface.co/bartowski/Samantha-Qwen-2-7B-GGUF/discussions/2">bartowski/Samantha-Qwen-2-7B-GGUF · 测试实验性量化</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度快 2-5 倍，内存占用减少 80%</a>: 微调 Llama 3, Mistral, Phi &amp; Gemma LLMs，速度快 2-5 倍，内存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8183">Bug: 量化版 gemma 27b 在分词器修复和 soft capping 后输出仍然错误 · Issue #8183 · ggerganov/llama.cpp</a>: 发生了什么？量化版本的 gemma 27b (Q8_0) 即使在处理简单问题时仍然得到错误答案。AI Studio 上的 gemma 版本正确回答了我所有的问题。示例问题...</li><li><a href="https://huggingface.co/bartowski/Samantha-Qwen-2-7B-GGUF/discussions/2#6673e67852c02322ba7ee01c">bartowski/Samantha-Qwen-2-7B-GGUF · 测试实验性量化</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski">bartowski (Bartowski)</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8197">添加注意力层和最终 logit soft-capping，更新 Gemma2 的缩放因子，由 abetlen 提交 · Pull Request #8197 · ggerganov/llama.cpp</a>: 此 PR 添加了缺失的注意力层和最终 logit soft-capping。实现参考自 Hugging Face Transformers。此外，Gemma2 应用了 hidden_size / ... 的预注意力缩放。</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8227">gemma2: 由 ngxson 添加滑动窗口掩码 · Pull Request #8227 · ggerganov/llama.cpp</a>: 这是一个通过掩码过去 Token 来支持 Gemma 2 滑动窗口注意力 (sliding window attention) 的临时方案。目标是使其工作。虽然理想的解决方案是拥有每层 KV cache 管理（具有不同的...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8240)),">Issues · ggerganov/llama.cpp</a>: 使用 C/C++ 进行 LLM 推理。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://ubuntu.com/desktop/wsl">Windows Subsystem for Linux (WSL) | Ubuntu</a>: 在 Windows 上通过 WSL 访问 Ubuntu 终端。无需离开 Windows 即可开发跨平台应用并管理 IT 基础设施。 
</li>
</ul>

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1256328974762508309)** (221 messages🔥🔥): 

- **Gemma 2 的性能波动**：关于 Gemma 2 的性能讨论非常广泛，评价褒贬不一，有的认为它和 **Mistral** 一样出色，有的则指出它存在**持续重复**等严重问题。用户探讨了多种技术调整，并分享了关于修复这些问题的**待发布更新**的见解。
   - 存在对兼容性的担忧，特别是与 **AMD ROCm** 的兼容性，以及不同量化（如 **Q4KM 和 IQ3_M**）的效率。讨论显示，对于这些修复措施的有效性意见不一，用户分享了各自的测试和体验。
- **模型偏好与对比**：成员们频繁对比了 **ChatGPT, Claude 3.5, Deepseek** 等多种模型，分享了性能基准测试和个人体验。**Deepseek Coder** 被特别强调为 7b 范围内最好的本地代码模型。
   - 由于不满而从 **ChatGPT** 转向其他模型是一种普遍情绪，一些用户更倾向于 **Anthropic 的 Claude 3.5**。尽管用户指出了 UI 等有待改进的地方，但像 **LLM Studio** 这样的新模型因其效率而经常受到称赞。
- **量化及其障碍**：量化讨论通常围绕平衡模型大小与性能展开，重点关注 **Q4KM, IQ4_XS 和 IQ3_M** 等变体。测试后的量化模型显示 **5KM** 更好，但如果硬件受限，**IQ3-M** 也是可行的。
   - 报告强调了某些量化模型的问题，指出存在不匹配和损坏的量化文件。用户分享了可以正常运行且没有重大问题的解决方案和替代量化模型，例如 **bartowski 的量化版本**。
- **用于图像和文本任务的视觉模型**：用户讨论了视觉模型处理**图像描述（image captioning）**等任务的能力，并针对漫画分析等特定需求寻求建议。**Florence-2 和 LLaVa** 被推荐为胜任的选择。
   - 讨论扩展到了这些模型的具体用例及其在现有工作流中的集成。**Meta Chameleon** 被提及为视觉任务的另一个潜在模型，尽管该群体尚未对其进行广泛测试。
- **本地模型实验与设置**：用户经常分享他们的设置和本地模型配置，强调了**嵌入（embedding）选项和 GPU 设置**的重要性。由于硬件限制，许多人更倾向于更小、更高效的模型。
   - 优化本地 LLM 性能的实用建议包括配置 **GPU 层或微调（fine-tuning）**。对话还详细阐述了使用 **Nomic-Embed-Text-v1.5** 等模型针对特定任务优化嵌入的技术。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/cartoons-tom-and-jerry-ok-mouse-ok-i-got-it-gif-17005831">汤姆和杰瑞动画 GIF - Cartoons Tom And Jerry Ok - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9">GGUF 量化概览</a>：GGUF 量化概览。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://huggingface.co/bartowski/Yi-1.5-34B-Chat-GGUF/blob/main/Yi-1.5-34B-Chat-Q6_K.gguf">Yi-1.5-34B-Chat-Q6_K.gguf · bartowski/Yi-1.5-34B-Chat-GGUF at main</a>：未找到描述</li><li><a href="https://rentry.org/LMSTudioFAQ">非官方 LMStudio FAQ！</a>：欢迎来到非官方 LMStudio FAQ。在这里你可以找到 LMStudio Discord 中最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源软件...</li><li><a href="https://huggingface.co/mxs980/gte-Qwen2-1.5B-instruct-Q8_0-GGUF">mxs980/gte-Qwen2-1.5B-instruct-Q8_0-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://link.springer.com/chapter/10.1007/978-3-540-30493-7_19">混合 CDN 的设计</a>：基于点对点（P2P）的网络在内容分发方面具有低成本、可扩展性和容错性等多种理想特性。然而，它们通常无法提供协同保证...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1256323247704641609)** (1 条消息): 

- **LM Studio 0.2.26 发布**: LM Studio 0.2.26 现已支持 Mac (M1/M2/M3), Windows (x86/ARM64), 以及 Linux (x86)。你可以从 [lmstudio.ai](https://lmstudio.ai) 下载。
   - 此版本包含了对 Google Gemma 2 模型的支持，并基于 `llama.cpp` 的 commit `97877eb10bd8e7f8023420b5b5300bcbdadd62dc` 构建。
- **支持 Google Gemma 2 模型**: 最新版本增加了对 **Google Gemma 2** 模型（特别是 **9B** 和 **27B** 版本）的支持。你可以在此处下载 [9B 版本](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) 以及 [27B 版本](https://huggingface.co/lmstudio-community/gemma-2-27b-it-GGUF)。
   - 该功能的实现得益于社区对 `llama.cpp` 的贡献。
- **适用于 Windows on ARM 的 LM Studio**: **LM Studio 0.2.26** 现已支持 Windows **ARM64** (Snapdragon X Elite 电脑)。这是通过与 Qualcomm 合作实现的，更多讨论请见 [LinkedIn](https://www.linkedin.com/posts/qualcomm_ai-snapdragon-snapdragonxseries-activity-7212153031222513666-7s0N?utm_source=share&utm_medium=member_desktop)。
   - ARM64 版本可从 [lmstudio.ai/snapdragon](https://lmstudio.ai/snapdragon) 下载。
- **即将进行的 LM Studio 0.3.0 Beta 测试**: LM Studio 的一次**重大更新**已准备就绪，团队正在招募 Beta 测试人员。感兴趣的用户可以[在此报名](https://forms.gle/K7pTWgTJsdHBmUaWA)。
   - 包含新 Beta 版本的邮件将在可用时发送给参与者。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并实验本地 LLM</li><li><a href="https://lmstudio.ai/snapdragon">👾 LM Studio - 发现并运行本地 LLM</a>: 查找、下载并实验本地 LLM</li><li><a href="https://forms.gle/K7pTWgTJsdHBmUaWA">LM Studio 0.3.0 - 私测报名</a>: 感谢您有兴趣帮助测试我们即将发布的版本。LM Studio 0.3.0 充满了新功能，我们希望在向全球发布之前，得到您的帮助来排查 Bug...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1257011322353225908)** (13 条消息🔥): 

- **V0.2.26 的性能问题**: 有用户反馈 V0.2.26 感觉比 V0.2.24 慢，**5B 模型**的运行速度仅相当于旧版本的 **8B 模型**。
   - 另一位成员建议，变慢可能是由于**旧的聊天记录**以及达到了 **Context** 限制，并建议使用**更强大的 GPU** 作为解决方案。
- **输入框缺少浏览器拼写检查**: 一位成员注意到，尽管拼写错误的单词会被画下划线，但在各个输入框中缺少浏览器的拼写检查功能。
   - 另一位成员确认，虽然**拼写检查**功能已被请求很久，但目前尚不可用。
- **删除聊天弹窗中的键盘导航**: 用户反馈“删除聊天”弹窗应该为键盘导航提供按钮焦点。
   - 他们提到，虽然 **Escape/Enter** 键可以控制弹窗，但初始的 Tab 选择仍需要鼠标点击。
- **创建新语言频道的指南**: 有用户询问如何根据建议请求新语言的指南来创建一个挪威语频道。
   - 成员确认用户应该能够**自行创建 Thread**，该询问只是为了遵循**合规规则**。
  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1256364080504242318)** (149 条消息🔥🔥): 

- **混合低硬件需求的小型模型**：一位用户建议，组合 0.5b 模型可以使其在 4GB 内存的电脑上高度可用，从而可能弥补硬件差距。
   - 讨论了 CPU 和 GPU 的速度差异，其中 Intel 和 IPEX 的 llama.cpp 分支因更好的 GPU 性能而受到关注。
- **服务器功耗：电力十足！**：用户分享了各种 LLM 模型和配置的功耗详细统计数据，包括一台配备双 Xeon E5 CPU 和 Tesla P40 GPU 的服务器，功耗高达 858 瓦。
   - 另一位用户对比了自己的服务器配置，指出 Nvidia-smi 报告的数值约为 70W 和 180W，引发了关于高效监控和硬件优化的讨论。
- **4090 对阵双 4080：GPU 大决战**：一位用户向社区询问了单张 4090 与双 4080 GPU 之间的性能权衡，得到的建议是单张 4090 通常表现更好，因为模型在两张显卡之间拆分时速度会受损。
   - 进一步的讨论建议，对于像 70b 这样的大型模型，单张 24GB GPU 优于双 16GB 配置，这加强了对单张 4090 的推荐。
- **交易配置：Mac 对阵 PC**：一位用户考虑了在 Mac Studio 上运行交易软件相对于 PC 配置的优势，提到他们可以在任何地方运行其 Python/C# 交易机器人，但界面是基于 Windows 的。
   - 另一位用户提出了一个“价值百万美元”的想法，即开发适用于 Mac 的交易软件以使切换更可行，并强调硬件灵活性对 AI 工作负载至关重要。
- **ASIC：AI 的未来**：Pooxid 介绍了专门为 Transformer 优化的 Sohu ASIC，声称它可以以每秒超过 500,000 个 token 的速度运行 LLaMA 70B——比 NVIDIA 的 GPU 更快且更便宜。
   - 讨论指出，虽然像 Sohu 这样的 ASIC 无法运行传统的 AI 模型，但它们可以显著加速基于 Transformer 的任务，引发了关注和审慎的乐观。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.etched.com/announcing-etched">Etched 正在 AI 领域进行最大的豪赌</a>：未找到描述</li><li><a href="https://tenor.com/view/doja-cat-star-wars-gif-25078126">Doja Cat GIF - Doja Cat Star - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/bartowski/Smaug-Llama-3-70B-Instruct-32K-GGUF">bartowski/Smaug-Llama-3-70B-Instruct-32K-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1256323614081421395)** (14 条消息🔥): 

- **ROCm 0.2.26 版本发布**：宣布 **0.2.26 ROCm** 版本于[今天](https://discord.com/channels/1110598183144399058/1195858490338594866/1256374959651946616)可用。具体的发布时间尚不明确，引发了一些关于时差的有趣调侃。
   - 一次有趣的交流中包含了一个 **Futurama Angry GIF** ([链接](https://media1.tenor.com/m/V45VFWvSVyUAAAAC/futurama-angry.gif))。用户们似乎对部署充满期待但保持耐心。
- **排查新版本中的空响应问题**：一位用户在尝试使用更新了 ROCm 和 Gemma 的最新版本后遇到了**空响应**。详细日志显示进程已启动，但未完成任何输出。
   - 经过一番沟通，发现使用 **SillyTavern** 作为前端是导致问题的原因。直接在 UI 中测试运行正常，这表明第三方集成中可能存在 Bug。
- **使用 LM Studio CLI 调试问题**：为了帮助调试问题，一位成员建议安装 **LM Studio CLI** 工具 (`lms`)。提供了[安装指南](https://lmstudio.ai/blog/lms)的链接。
   - 使用命令 `lms log stream` 可以帮助识别发送给模型的 Prompt，从而辅助排查过程。这是进一步调试的推荐做法。

**提及的链接**：<a href="https://tenor.com/view/futurama-angry-gif-13063135">Futurama Angry GIF - Futurama Angry - 发现并分享 GIF</a>：点击查看 GIF

  

---

### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1256340898820653086)** (29 条消息🔥): 

- **Windows 版 ROCm 扩展包 0.2.26 现已发布**：一位成员宣布发布了适用于 **Windows** 的 **0.2.26 ROCm "扩展包"**，并提供了下载链接和安装说明。用户可以从 [LMStudio.ai](https://lmstudio.ai) 获取，且作为安装的一部分，需要在 Powershell 中运行一条命令。
   - 按照正确的安装步骤操作后，多位用户确认更新成功且性能有所提升，特别是 Gemma 2 模型，现在可以无缝运行。一位用户强调，在这些更新下，**RX 7900 XTX** 在满载使用时功耗达到 350W。
- **ROCm 0.2.26 带来显著性能提升**：成员们报告称，在升级到 **ROCm 0.2.26** 并使用最新的 **24.6.1** AMD 驱动程序后，性能有明显增强。例如，在 **6900 XT** 上运行 **Codestral 22B Q4_K_M**，性能从旧版本的 **9.29 tok/s** 跃升至 **26.75 tok/s**。
   - 一位用户指出，尽管之前为了更好的 AI 性能购买了 NVIDIA GPU，但最近的改进可能使其变得不再必要。另一位用户强调了对比不同硬件性能差异的重要性。
- **Qwen2 模型问题与修复**：多位用户报告 Qwen2 存在 Bug，但一位成员指出启用 **flash attention** 可以解决这些问题。讨论中涉及了具体的配置和设置说明，以确保模型正常运行。
   - *heyitsyorkie* 澄清说 Qwen2 需要 **flash attention** 才能正常工作，确认了针对出现的 Bug 的解决方案。用户反馈了哪些配置对他们有效或无效。
- **ROCm 支持与兼容性讨论**：随后进行了关于某些 GPU 缺乏 **ROCm 支持** 以及特定版本可用性的技术讨论。**Blue.balls** 提到了 ROCm 未作为选项出现的问题，经澄清这是由于该用户的 GPU 不支持 ROCm。
   - 成员们讨论了拥有正确的扩展包和兼容硬件的必要性。*heyitsyorkie* 确认 **0.2.26 Linux ROCm** 支持仍在开发中，并建议等待官方更新。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai,">无标题</a>：未找到描述</li><li><a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>：查找、下载并实验本地 LLMs</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#installation-on-windows-0226-">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1256339154602688622)** (11 条消息🔥): 

- **在 Unraid/Docker 上安装 LM Studio 具有挑战性**：一位成员询问是否有办法在 **Unraid** 或 **Docker** 上安装 **LM Studio**，但得到的反馈称目前尚不可能。
   - 讨论未产生任何解决方案，凸显了目前在这些环境中部署 LM Studio 的局限性。
- **LM Studio SDK 版本更新导致 loadModel 错误**：在将 LM Studio 从 **0.2.25** 更新到 **0.2.26**，并将 **lmstudio.js** 从 **0.0.3** 更新到 **0.0.12** 后，一位用户在执行 `await client.llm.load(modelPath)` 命令时遇到错误。
   - 报告了 *'Received invalid creationParameter for channel'* 错误，建议在 [GitHub](https://github.com/lmstudio-ai/lmstudio.js/issues) 上提交 Issue。
- **SDK 问题的社区支持**：包括 **@heyitsyorkie** 和 **yagilb** 在内的成员回应了用户关于 SDK 命令的问题并提供了帮助。
   - 他们建议在 GitHub 上提交 Issue 以进行进一步调查和排查，展示了**活跃的社区支持**。

**提到的链接**：<a href="https://github.com/lmstudio-ai/lmstudio.js/issues">Issues · lmstudio-ai/lmstudio.js</a>：LM Studio TypeScript SDK (pre-release public alpha) - Issues · lmstudio-ai/lmstudio.js

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1256323904729907272)** (716 条消息🔥🔥🔥):

- **SD3 Lora 训练的难点**：用户讨论了在 Stable Diffusion 3 模型上创建和使用 Lora 的挑战，提到了其复杂性以及目前缺乏有效训练支持的问题。大家对 SD 8b 充满期待，该版本可能不需要过多的微调。
   - 一些用户表示，等待 SD3 完善的微调数据和训练工具至关重要，而不是仓促创建质量欠佳的 Lora 或 Checkpoints。
- **为 SD 选择合适的硬件**：对运行 Stable Diffusion 感兴趣的新用户讨论了硬件要求，如 GPU 显存和处理能力。建议倾向于使用至少具备 12GB VRAM 或更高显存的 Nvidia GPU，以实现高效运行。
   - 用户提到，虽然像拥有 24GB VRAM 的 RTX 3090 这种旧款 GPU 仍有价值，但最新的显卡（RTX 4080, 4090）尽管价格高昂，却更具前瞻性。
- **安装与设置问题**：记录了在使用各种界面（如 Automatic1111, ComfyUI）安装和设置 Stable Diffusion 时遇到的问题，特别提到了为实现最佳性能而使用的特定设置命令，尤其是针对高分辨率和复杂工作流。
   - 用户分享了资源和安装指南，提供了特定的配置技巧，例如在启动命令中使用 'xformers' 和 'medvram-sdxl'。
- **在 SDXL 中使用高分辨率修复 (High-Resolution Fix)**：解释了使用高分辨率修复设置来创建更清晰图像的方法，这需要正确设置特定参数（例如 10 步 Hires 步数、合适的分辨率）。
   - 用户指出，利用 adetailer 等高级插件来增强动漫风格艺术作品中人脸和眼睛等关键图像组件的重要性。
- **寻找和使用模型及 Lora**：讨论了在哪里可以找到模型和 Lora，Civitai 等热门网站因其丰富的仓库而受到关注。
   - 重点提到了检查提示词示例的重要性，以确保 Lora 和模型被正确使用，并强调了社区对模型训练和分享的贡献。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.twitch.tv/timelesstakes">TimelessTakes - Twitch</a>: AI 实时回应 Twitter | 爱因斯坦、乔布斯、梦露和 MJ 🔴 在聊天中向他们提问！</li><li><a href="https://rog.asus.com/graphics-cards/graphics-cards/rog-strix/rog-strix-rtx3090-o24g-gaming-model/">ROG Strix GeForce RTX 3090 OC Edition 24GB GDDR6X | 显卡</a>: ROG Strix GeForce RTX 3090 OC Edition 24GB GDDR6X 通过采用轴流风扇设计、0dB 技术、2.9 插槽设计、双 BIOS、Aut... 释放了 NVIDIA Ampere 架构的最大性能。</li><li><a href="https://youtu.be/Azj9Kkpif0M">越狱 GEMMA 2</a>: 通过几个简单的提示词越狱 Google AI 的 Gemma 2。Gemma 2 是 Google 发布的一个新的 Large Language Model。我正在使用...</li><li><a href="https://tenor.com/view/michael-jackson-eating-popcorn-enjoy-i-like-nom-nom-gif-11040065238845078056">迈克尔·杰克逊吃爆米花 GIF - 迈克尔·杰克逊吃爆米花享受 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://archvizartist.com/article/how-to-install-stable-diffusion-on-windows-automatic1111/">如何在 Windows 上安装 Stable Diffusion (AUTOMATIC1111) – Arch Viz Artist</a>: 未找到描述</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Installation-Guides">安装指南</a>: Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://github.com/TheLastBen/fast-stable-diffusion">GitHub - TheLastBen/fast-stable-diffusion: fast-stable-diffusion + DreamBooth</a>: fast-stable-diffusion + DreamBooth。通过在 GitHub 上创建账户为 TheLastBen/fast-stable-diffusion 的开发做出贡献。</li><li><a href="https://pics.io/photo-metadata-viewer">图像元数据查看器 - 在线 EXIF 数据查看器</a>: 免费服务：用于照片和图像的在线元数据 (EXIF) 查看器。</li><li><a href="https://github.com/openvinotoolkit/stable-diffusion-webui">GitHub - openvinotoolkit/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户为 openvinotoolkit/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://colab.research.google.com/github/R3gm/SD_diffusers_interactive/blob/main/Stable_diffusion_interactive_notebook.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://civitai.com/models/140737/albedobase-xl">AlbedoBase XL - v2.1 | Stable Diffusion Checkpoint | Civitai</a>: 如果您发现该模型的价值，请考虑提供支持。您的贡献将完全用于推动 SDXL 社区...</li><li><a href="https://civitai.com/models/261336/animapencil-xl">anima_pencil-XL - v5.0.0 | Stable Diffusion Checkpoint | Civitai</a>: 简单、便捷、高质量（在多样性上略有权衡）blue_pencil-XL 结合了 ANIMAGINE XL 3.0 / ANIMAGINE XL 3.1 许可证：Fair AI Public L...</li><li><a href="https://www.newegg.com/abs-aqa14700kf4060ti16g-stratos-aqua/p/N82E16883360436">ABS Aquilon Aqua 游戏电脑 - Windows 11 家庭版 - Intel Core i7 第 14 代 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI 驱动性能 - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI16G - Newegg.com</a>: 购买 ABS Aquilon Aqua 游戏电脑 - Windows 11 家庭版 - Intel Core i7 第 14 代 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI 驱动性能 - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/13a9avh/quick_question_does_putting_break_in_a_prompt/">Reddit - 深入探索一切</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1257353627534622871)** (1 条消息): 

- **会议协调必备工具**: 分享了两个用于协调会议时间的工具：**[Discord Timestamps](https://r.3v.fi/discord-timestamps/)**，它允许消息以查看者所在时区显示时间；以及 **[When2meet](https://www.when2meet.com/)**，它有助于收集小组的空闲时间以找到最佳会议时间。
   - 这些工具旨在通过处理时区转换和空闲时间投票来简化日程安排，使所有参与者更容易找到共同方便的会议时间。
- **简化小组日程安排**: **Discord Timestamps** 允许您使用类似 `<t:1717964400:f>` 的格式发送消息，该格式会根据查看者的时区显示特定时间。
   - **When2meet** 简化了收集小组空闲时间以确定最佳会议时间的过程。
  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1256686956440715354)** (4 messages): 

- **Log Exp 工具效用辩论**：**chr0nomaton** 提到了在减少统计建模和机器学习中的数值稳定性问题时，log exp 函数的效用。他分享了一篇[博客文章](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)，解释了 **logarithmic scale**（对数尺度）如何将乘法转换为加法，以获得更好的数值稳定性。
   - **chhillee** 反驳说通常不需要对数尺度，从而引发了关于其必要应用场景的进一步提问。**doubleart** 询问是否存在它变得至关重要的特定场景。
- **对数尺度稳定小浮点数**：**chr0nomaton** 强调，在处理微小的浮点数时，使用 **logarithmic scale** 特别有用，可以避免 underflow（下溢）。他强调这种技术使得处理 log likelihoods（对数似然）和概率在数值上更加稳定。
   - **chhillee** 回应称对数尺度的效用通常是不必要的，引发了进一步讨论。**doubleart** 请求澄清哪些特定情况是必不可少的。

**提到的链接**：<a href="https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/">The Log-Sum-Exp Trick</a>：未找到描述

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1256846056218562622)** (24 messages🔥): 

- **在使用 torch.compile 时向 torch.Tensor 附加元数据**：一位成员询问如何在保持对 `torch.compile` 支持的情况下，以 tuple 的形式向 `torch.Tensor` 附加元数据，且不触发 `UserDefinedVariable` 错误。他们探索了包括使用来自 `torchvision` 带有元数据的 Tensor 子类在内的多种方案，但在 Tensor 操作过程中遇到了元数据丢失的问题。
   - 建议包括将元数据存储为 torch Tensor 或使用类似 `torchvision` 中的子类；然而，这些并没有解决问题。用户分享了在各种 Tensor 变换和显式 Tensor 传递过程中维持元数据的困难。
- **在 torch.compile 中强制使用编译后的函数**：一位用户询问是否有办法强制 `torch.compile` 对预定义的输入形状使用编译后的函数，而对其他形状回退到 eager mode，以避免重新编译。背景涉及优化 HuggingFace transformers 的 `.generate()` 函数，其中 prefill 和 decoding 阶段具有不同的输入形状要求。
   - 建议包括使用 `torch.compile(dynamic=True)`，使用细粒度 API 标记要在 eager mode 下运行的代码部分，以及设置重新编译限制。用户确认他们已经在利用自定义包装器，并对跨新版本的维护表示担忧。
- **HuggingFace trainer save_model 超时问题**：一位成员在多机使用 `HuggingFace` trainer 的 `save_model` 时遇到超时问题，尽管中间 checkpoint 保存成功。他们发现调用 `state_dict = trainer.accelerator.get_state_dict(trainer.model)` 会导致挂起，从而引发 socket 超时。
   - 该设置在单机上运行良好，表明问题特定于多机配置。错误日志显示了与 socket 超时相关的 `torch.distributed.elastic` 问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html">TorchDynamo APIs for fine-grained tracing &mdash; PyTorch 2.3 documentation</a>：未找到描述</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/utils/generation_hf.py">hqq/hqq/utils/generation_hf.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch/vision/blob/main/torchvision/tv_tensors/_bounding_boxes.py">vision/torchvision/tv_tensors/_bounding_boxes.py at main · pytorch/vision</a>：特定于计算机视觉的数据集、转换和模型 - pytorch/vision</li><li><a href="https://docs.google.com/presentation/d/1piuv9nBzyoqdH49D1SoE5OZUPSMpOOFqfSKOhr-ab2c/edit#slide=id.p1">BackToPython PTC 2022 Poster</a>：1 Back to Python: 在不接触 C++ 的情况下扩展 PyTorch。Alban Desmaison Meta 摘要：直接从 Python 扩展 PyTorch。在过去的一年里，PyTorch 团队一直致力于改进扩展性...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1256713771242754099)** (3 messages): 

- **AI Engineer World’s Fair 2024 展示**: 一位成员推荐了名为 **"AI Engineer World’s Fair 2024 – GPUs & Inference Track"** 的 [YouTube 视频](https://www.youtube.com/watch?v=JVSKlEmUr0k)，该视频由 **AI Engineer World** 频道发布。他们强调*这整个专题*都值得快速浏览。
   - 在同一专题中，*Dylan 的演讲*获得了积极反馈，强调了其在社区中的重要性。
- **Stephen Jones 的演讲总是令人印象深刻**: 一位成员为 **Stephen Jones** 的所有演讲背书，称其内容始终具有价值且见解深刻。他们指出，*"字面上 Stephen Jones 的每一场演讲都值得观看"*。
   - 另一条评论呼应了这一观点，并补充说 Stephen Jones 的演讲受到社区的高度推荐。

**提到的链接**: <a href="https://www.youtube.com/watch?v=JVSKlEmUr0k">AI Engineer World’s Fair 2024 — GPUs &amp; Inference Track</a>: https://twitter.com/aidotengineer

  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1256639961550291060)** (2 messages): 

- **书中有空白页**: 一位用户报告他们的书中有**空白页**。这表明可能存在印刷错误。
   - 目前尚不清楚该问题是孤立案例还是在多个副本中普遍存在。
- **PMPP 版本差异**: 一位用户询问了 PMPP **第三版和第四版**之间的区别。他们担心是否需要购买新版。
   - 目前还没有关于具体变化或升级必要性的进一步细节或回复。
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1256361464252465265)** (42 messages🔥): 

- **FP16 解决了 Kernel launch 瓶颈**: 在量化 API 中使用 bfloat 时，**Kernel launches** 导致了性能问题，但切换到 **FP16** 后将其减少到了单次 launch。详细的 profiling 揭示了多次 Kernel launch 的情况。
   - 尽管进行了优化，讨论中还提到了进一步的改进，例如针对 **large tensors** 的特殊 unpack+dequant+matmul kernels。社区建议整合 **gemv 工作**以实现未来的进步。
- **Bitpacking 优化在大张量上遇到困难**: **提升 bitpack 函数速度**的努力在小张量上取得了成功，但在 2048 或更大的张量上效果不佳，性能甚至比 **FP16** 更差。
   - 未来的**研究**可能涉及开发特殊的 kernels 以实现显著加速，并承认目前 PyTorch 在高效融合操作方面存在局限性。
- **关于 torchao 贡献的 Flash attention 讨论**: 建议在 Triton 中添加 **Flash attention 变体**作为 torchao 新贡献者的良好入门议题（good first issue），特别是围绕来自 **sam-fast** 等工具的相对位置编码（relative positional encodings）。
   - **Marksaroufim** 表达了保留意见，因为重点是 dtype 和 layout 库，但也承认如果范围不会过度扩张，这对**架构优化**可能是有益的。
- **FlexAttention API PR 亮相**: **Drisspg** 及其合作者一直在为 PyTorch 开发新的 **FlexAttention API**，旨在扩展现有 attention 模块的功能。
   - 初始的 **pull request** 已与社区分享（[FlexAttention API PR](https://github.com/pytorch/pytorch/issues/121845)），并强调在公开 API 发布后将进行广泛公告。
- **关于 CUDA 贡献者入门议题的讨论**: 讨论了如何为新贡献者确定**良好的入门议题**，潜在任务涉及为 torchao 开发 CUDA kernel。
   - 讨论中提出了对直接编写原始 CUDA kernel 复杂性的担忧。建议提供更易上手的任务和高层级的引导，以帮助新贡献者入门。

**提到的链接**: <a href="https://github.com/pytorch/pytorch/issues/121845">FlexAttention API by drisspg · Pull Request #121845 · pytorch/pytorch</a>: 摘要：此 PR 添加了一个新的 higher-order_op: templated_attention。该算子旨在扩展 torch.nn.functional.scaled_dot_product_attention 的功能。PyTorch 拥有高效的预编写...

  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1257054602830811199)** (2 条消息): 

- **分享了隔壁 10x 软件工程师视频**: *Iron_bound* 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=kKAue9DiHc0)，标题为 "**Next-door 10x Software Engineer** [FULL]"。
   - *as_ai* 对视频发表了评论，表示 *"我喜欢这家伙的视频。非常有趣"*。
- **对搞笑视频的赞赏**: *As_ai* 表达了对分享视频中幽默内容的喜爱。
   - *As_ai* 提到，*"我喜欢这家伙的视频。非常有趣"*，显然很享受这些内容。

**提到的链接**: <a href="https://www.youtube.com/watch?v=kKAue9DiHc0">*Next-door 10x Software Engineer* [FULL]</a>: 未找到描述

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1256323417297125456)** (473 条消息🔥🔥🔥): 

- **FineWeb 与 FineWeb-EDU 的稳定性对比**: FineWeb 100B 样本存在训练不稳定的情况，而 FineWeb-EDU 样本在 1.5B 模型上表现更稳定。多位成员证实了这一点，进一步的分析正在进行中。
   - 团队正在调查 FineWeb 样本中的潜在问题，并计划切换到 FineWeb-EDU，后者在训练运行中表现出了持续的稳定性。
- **追踪性能问题**: 对追踪性能的深入研究揭示了潜在的微调领域，特别是在不同的 batch size 和 recompute 设置下。某些配置显示出的改进微乎其微，表明在扩展时可能存在收益递减。
   - 目前有一项计划旨在更好地理解 batch size、recompute 设置与整体训练性能之间的相互作用，研究结果表明，更激进的策略可能不会产生成比例的改进。
- **使用 MuP 和 FP8 进行协调与调试**: 将 Maximum Update Parametrization (MuP) 集成到训练工作流中的努力显示出稳定训练的潜力。协调检查和进一步的稳定性测试正在进行中，以微调超参数。
   - 正在探索 FP8 的集成，重点是最小化性能开销并确保架构的合理性。团队还专注于清理 matmul 操作以获得更好的性能。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/661">修复 gordicaleksa 提交的周期性 Loss 尖峰 · Pull Request #661 · karpathy/llm.c</a>：我们在主代码中引入了一个与之前 PyTorch 代码中类似原因导致的 Bug：由于 zero grad 发生在 backward 步骤之后，而不是在我们执行 ba... 之前。</li><li><a href="https://github.com/karpathy/llm.c/pull/653">由 ademeure 提交的仅使用 cuBLASLt + GELU Fusion 的 Matmul 重构 · Pull Request #653 · karpathy/llm.c</a>：为了给 FP8 做准备，此 PR 将所有 cuBLAS 调用替换为 cuBLASLt，并封装在单个 matmul_cublaslt() 函数中。它还增加了对 GELU fusion 的支持，可以通过 c... 进行控制。</li><li><a href="https://huggingface.co/mlx-community">mlx-community (MLX 社区)</a>：未找到描述</li><li><a href="https://github.com/LambdaLabsML/llm.c-1cc">GitHub - LambdaLabsML/llm.c-1cc</a>：通过在 GitHub 上创建账号来为 LambdaLabsML/llm.c-1cc 的开发做出贡献。</li><li><a href="https://github.com/clu0/unet.cu">GitHub - clu0/unet.cu: 纯 CUDA 实现的 UNet 扩散模型</a>：纯 CUDA 实现的 UNet 扩散模型。通过在 GitHub 上创建账号来为 clu0/unet.cu 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/commit/a876282eb845f89aef70c780033ee150aba044b0">合并来自 ademeure/cublaslt_refactor 的 Pull Request #653 · karpathy/llm.c@a876282</a>：仅使用 cuBLASLt + GELU Fusion 的 Matmul 重构</li><li><a href="https://github.com/karpathy/llm.c/pull/657">由 ademeure 提交的移除每层 attproj 和 fcproj 激活张量 · Pull Request #657 · karpathy/llm.c</a>：我不确定我们是怎么漏掉这个的，但我们在 backward pass 中根本不需要这些张量！可能在实现 residual/layernorm/recompute 时情况并非如此...</li><li><a href="https://github.com/microsoft/mup?tab=readme-ov-file#how-mup-works-under-the-hood">GitHub - microsoft/mup: 最大更新参数化 (µP)</a>：最大更新参数化 (µP)。通过在 GitHub 上创建账号来为 microsoft/mup 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/6">并非所有英雄都披着斗篷 · Issue #6 · karpathy/llm.c</a>：只是想代表社区说声谢谢。谢谢你，Andrej。❤️ 我在提交此 Issue 后会将其关闭，这样你就不用动手了。</li><li><a href="https://github.com/karpathy/llm.c/blame/master/.github/workflows/ci_gpu.yml#L90">查看 llm.c/.github/workflows/ci_gpu.yml 的 Git Blame（位于 master 分支）· karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/650#issuecomment-2198342446">由 gordicaleksa 提交的 muP (最大更新参数化) · Pull Request #650 · karpathy/llm.c</a>：主要变更：修改随机初始化；按 1/d 而非 1/sqrt(d) 缩放 Attention 分数并添加 attn_mult；在映射到 Logits 之前按 1/width_mult 缩放激活值；更新学习率 &...</li><li><a href="https://github.com/karpathy/llm.c/pull/650/">由 gordicaleksa 提交的 muP (最大更新参数化) · Pull Request #650 · karpathy/llm.c</a>：主要变更：修改随机初始化；按 1/d 而非 1/sqrt(d) 缩放 Attention 分数并添加 attn_mult；在映射到 Logits 之前按 1/width_mult 缩放激活值；更新学习率 &...</li><li><a href="https://github.com/karpathy/llm.c/pull/650">由 gordicaleksa 提交的 muP (最大更新参数化) · Pull Request #650 · karpathy/llm.c</a>：主要变更：修改随机初始化；按 1/d 而非 1/sqrt(d) 缩放 Attention 分数并添加 attn_mult；在映射到 Logits 之前按 1/width_mult 缩放激活值；更新学习率 &...</li><li><a href="https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards">karpathy/fineweb-edu-100B-gpt2-token-shards · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/karpathy/llmc-starter-pack">karpathy/llmc-starter-pack · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/662">由 karpathy 提交的修复过拟合单 Batch 行为，使其真正过拟合 Batch 而非... · Pull Request #662 · karpathy/llm.c</a>：...microbatch，我们通过在每一步简单地重置 dataloader 来更干净地实现这一点。</li><li><a href="https://github.com/Azure/MS-AMP">GitHub - Azure/MS-AMP: 微软自动混合精度库</a>：微软自动混合精度库。通过在 GitHub 上创建账号来为 Azure/MS-AMP 的开发做出贡献。
</li>
</ul>

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1256535488454721568)** (1 messages): 

- **Nscale 对 AMD MI300X GPU 进行基准测试**：Nscale [对 AMD MI300X GPU 进行基准测试](https://www.nscale.com/blog/nscale-benchmarks-amd-mi300x-gpus-with-gemm-tuning-improves-throughput-and-latency-by-up-to-7-2x) 显示，通过 GEMM 调优，性能提升高达 **7.2倍**。他们专注于增强**吞吐量**、**降低延迟**以及高效处理复杂模型。
   - 该博客强调了使用 **rocBLAS** 和 **hipBLASlt** 等库进行 **GEMM 调优**以优化 **GEMM 操作**的重要性。这些工具对于最大化 GPU 加速任务的性能至关重要，确保了**更高的吞吐量**和高效处理。
- **GEMM 调优提升 GPU 性能**：Nscale 的技术深度解析讨论了 **GEMM 调优**对 GPU 性能的影响。博客重点介绍了显著的吞吐量基准测试和性能调优技术。
   - 通过使用 **rocBLAS** 和 **hipBLASlt** 等优化库，博客阐述了 **GEMM 调优**如何大幅降低延迟并提高计算效率。这些优化对于处理 AI 任务中复杂的模型和数据集至关重要。

**提到的链接**：<a href="https://www.nscale.com/blog/nscale-benchmarks-amd-mi300x-gpus-with-gemm-tuning-improves-throughput-and-latency-by-up-to-7-2x">Nscale 基准测试：通过 GEMM 调优将 AMD MI300x GPU 的吞吐量和延迟提升高达 7.2 倍</a>：优化 AI 模型性能：vLLM 吞吐量和延迟基准测试，以及使用 rocBLAS 和 hipBLASlt 进行 GEMM 调优。

  

---


### **CUDA MODE ▷ #[sparsity](https://discord.com/channels/1189498204333543425/1247663759434977453/1256708419558047825)** (1 messages): 

- **关于 RF 中剪枝（pruning）的问题**：一位成员引用了 [来自 PyTorch 的图表](https://pytorch.org/assets/images/accelerating-neural-network-training/fg8.png) 并要求澄清术语 **"Pruned in RF"**，询问 **RF** 是否指 **register file**（寄存器堆）。
   - 此外，还寻求了关于 kernel PR 以及相关论文/文章中 **"RF"** 含义和重要性的进一步背景信息。
- **CUDA 线程中的内存访问大小说明**：一位成员引用道：“这意味着在一个 CUDA 线程内，我们希望一次读/写 128 字节的块”，并询问 **128** 是否是单次列选择（column select）可以访问的内存大小。
   - 该成员正在寻求对 CUDA 线程中内存访问模式和大小的进一步理解，特别是关于最佳块大小（chunk size）的问题。
- **明确了 tile 处理中 FP16 (BF16) 的重要性**：有人针对计算公式“线程处理的不是单个 4x4 tile（仅 4x4x2 = 32 字节）”提出疑问，询问其中的 **"x2"** 是否代表 **FP16 (BF16)** 的 2 字节。
   - 这个问题突显了对 **FP16 (BF16)** 数据类型及其在内存处理计算中表示方式进行澄清的需求。
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1257006579518013540)** (2 messages): 

- **Gemma 2 模型在 Perplexity Labs 发布**：**Gemma 2 模型**现已在 [Perplexity Labs](https://labs.pplx.ai) 上线。鼓励用户尝试并分享反馈。
   - 该公告发布时充满期待，旨在寻求社区对新模型的意见。
- **Android 应用中的语音对语音功能**：最新版本的 Android 应用引入了**语音对语音功能**，包含两种模式：免提（Hands-free）和按住说话（Push-to-talk）。用户可以尝试并在指定频道提供反馈。
   - 免提模式在屏幕打开时立即开始监听，而按住说话模式则需要按住麦克风按钮才能说话。
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1256323813021585448)** (367 messages🔥🔥): 

- **Claude 3.5 在上下文处理上遇到困难**：用户报告称 **Claude 3.5** 经常遗忘上下文，在回答后续问题时给出笼统的建议而非具体的答案。一些用户指出，在开启网页搜索（web search）时会出现问题，因为它会干扰模型，强调了手动调整的必要性。
   - 尽管存在问题，从 **Claude 3.5** 切换到 **Opus** 帮助一些用户获得了更好的上下文保留效果。社区认为 **Claude 3.5** 可能存在影响性能的 bug，并建议 **Pro search** 应该智能地关闭网页搜索以提高连贯性。
- **Perplexity AI 的 Pro Search 表现不一致**：许多用户在 **Pro search** 中遇到了波动；有时它会执行多步搜索，而有时则不会。一位成员指出，除非手动切换到“写作（writing）”模式，否则短段落的处理效果很差。
   - 用户对 Perplexity 模型从网页搜索抓取内容而非参考之前的 Prompt 感到沮丧。建议包括改进网页搜索中上下文的优先级，以维持对话的连贯性。
- **不同模型的上下文窗口（Context Windows）测试**：社区成员 **dailyfocus_daily** 分享了多种模型的上下文窗口测试结果，指出 **Opus** 在 27k tokens 时表现正确，但在 46k 时失败，表明代码中可能存在 32k 的上下文限制。**Sonnet 3.5** 和 **GPT-4o** 因其稳定的 64k token 保留能力而受到赞赏。
   - 另一位用户询问 **Claude 3.5 Sonnet** 的信息是否真的截止于 2022 年 12 月，回复确认由于 **Claude.com** 和 **Perplexity.com** 使用了不同的 system prompts，导致了知识截止日期（knowledge cut-off）的问题。
- **Perplexity 上的 Claude 及其用户限制**：据报告，Perplexity 上的 **Claude 3.5** 并没有提供宣称的完整 200k tokens，通常限制在 32k tokens 左右。来自 Complexity 的用户确认了这些约束，同时讨论了平台上 token 限制不一致的更广泛问题。
   - 反馈涉及 **Perplexity Pro 计划**，指出 24 小时内最多使用 600 次，其中 **Opus** 限制为 50次/24小时。用户被引导至官方资源以了解限制，并提出了改进可用性和可访问性的建议。
- **用户常规反馈与支持查询**：用户频繁询问关于 **Perplexity AI 的功能与限制**，例如自定义指令（custom instruction）的影响、线程中重新上传文档的问题，以及特定模型行为的差异。**用户输入**反映了对改进 UI 以及使模型能力更好地符合用户预期的需求。
   - 出现了关于订阅计划和教育折扣的问题，要求为团队账户和 Pro 使用提供明确的指南。一些**非 Pro 用户**的反馈强调了模型性能的不稳定以及在不同搜索焦点（search focuses）之间切换的限制。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/discord-light-mode-light-mode-benny-beni-gif-25360134">Discord Light Mode Benny GIF - Discord Light Mode Light Mode Benny - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://youtu.be/aoikSxHXBYw?si=3jPRoPJmoJhZ6x1k">Mixture of Agents (MoA) BEATS GPT4o With Open-Source (Fully Tested)</a>：Mixture of Experts 实现的完整测试。订阅我的时事通讯有机会赢取 Dell 显示器：https://gleam.io/otvyy/dell-nvidia-monitor-1 (O...</li><li><a href="https://open.spotify.com/track/1wf9F3L1B11i9WTfvwnfMo?si=IXCqWpDxTz6CvZZKj4E1zg">The Chemical Worker's Song (Process Man)</a>：Great Big Sea · 歌曲 · 1995</li><li><a href="https://by-ai-monnef-9ff5d9c2460ae15d70e737f77eab719c6e8a4c64c2f99ca1c2.gitlab.io/2024/opus_50-ball-game/">opus_50 Game</a>：未找到描述</li><li><a href="https://open.spotify.com/track/6pPCkAzVYapjObH73BWu9t?si=aeucsMekQxG28MzDt0Gz2Q&context=spotify%3Aalbum%3A53VtJpStdfdFG2MSJRgZgC)">Father</a>：Sabaton · 歌曲 · 2022
</li>
</ul>

</div>

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1256330807539400734)** (16 条消息🔥): 

- **Minecraft 维修机制误导儿童**：[Minecraft 维修机制](https://www.perplexity.ai/page/Minecraft-Repair-Mechanics-NdRggXKXRXyGY8LgKsp1dQ) 因可能让儿童对现实世界的工具维修产生错误理解而受到批评。提供的链接中讨论了该机制的细节。
   - 社区对 Minecraft 等游戏的教育方面表示担忧。他们强调，这类机制可能会让孩子在现实工具维护方面感到困惑。
- **全球变暖对智能手机的影响**：一场关于 [全球变暖对智能手机充电性能日益增长的影响](https://www.perplexity.ai/page/the-increasing-impact-of-globa-xEkc7Nd9RA2tXXMoC4VT3g) 的讨论展开了。该页面详细介绍了气温升高如何影响电池效率。
   - 成员们分享了他们对这一问题的经验和担忧。他们辩论了潜在的解决方案，重点强调了开发更具韧性的电池技术的必要性。
- **创立 Perplexity：一个故事**：通过多个链接分享了 [创立 Perplexity 背后的故事](https://www.perplexity.ai/search/the-story-behind-starting-perp-DnZ.yJgfSM28Ra9_h2uKWg)。这段叙述不仅涵盖了公司创立初期，还深入探讨了所面临的挑战。
   - 成员们对创业历程进行了反思，提供了见解和观点。直接引用的话语强调了启动和维持创新项目所需的毅力。

**提到的链接**：<a href="https://www.youtube.com/embed/lJbAw0wCc0I">YouTube</a>：未找到描述

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1256349528102277161)** (19 条消息🔥): 

- **Perplexity API 中的特定日期结果**：一位成员建议在查询中使用 `after:2024-05-28`，以将 Perplexity API 的结果限制在过去 30 天内的新信息。然而，有人指出这可能会增加幻觉（hallucination），生成来自未来的文章。
   - 另一位成员建议申请封闭测试（closed beta）以获取 `search_domain_filter` 功能，作为该问题的潜在解决方案。他们分享了表单链接和电子邮件地址（[填写此表单](https://perplexity.typeform.com/to/j50rnNiB) 并 [发送邮件](mailto:api@perplexity.ai)）。
- **Perplexity API POST 请求示例**：一位成员分享了一个向 Perplexity API 发送 POST 请求的示例，包括所需的请求头（headers）和数据。他们详细说明了检索特定日期后新闻头条的具体用法。
   - 演示了这种方法的响应结果，展示了其在实际操作中的多功能性。该示例涉及使用适当的提示词和 API 配置从 NPR 网站获取最近的头条新闻。
- **Perplexity Labs Playground 的 Pro 账户问题**：一位用户在使用 Apple ID 登录时，遇到了 Perplexity Labs Playground 无法识别其 Pro 账户的问题。另一位成员澄清说，Playground 对所有人都是免费开放的，“Try Perplexity”按钮主要是一个营销工具。
   - 还讨论了如果用户注销并切换到普通电子邮箱账户可能导致的计费重叠问题。这解决了与免费版和 Pro 版账户功能相关的误解。
- **Perplexity API 与 Web UI 结果之间的差异**：一位成员观察到 Perplexity API 与其 Web UI 之间的结果质量存在显著差异。他们分享了一个示例查询，其中 Web UI 提供的信息比 API 好得多。
   - 建议他们将问题发布在 Perplexity 论坛上，并申请封闭测试中提供的更好的过滤选项。这表明 API 正在进行持续改进并考虑用户反馈。
- **API 设置页面的访问问题**：一位用户报告了访问 API 设置页面的问题，在网页上经历了持续加载。另一位成员建议更换浏览器，特别提到 Safari 是一个可以正常工作的解决方案。
   - 建议清除 Chrome 上的缓存作为潜在的修复方法。切换浏览器后，用户确认问题已解决，这表明浏览器设置可能会干扰某些功能的访问。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.npr.org"">无标题</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/discuss">Discussions</a>：未找到描述</li><li><a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api 表单</a>：使用 Typeform 将数据收集转变为一种体验。创建精美的在线表单、调查、测验等。免费试用。
</li>
</ul>

</div>

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1257320294259888290)** (3 messages): 

- **合成数据收集通过 10 亿个 Personas 提速**：[一个包含 10 亿个多样化 Personas 的集合](https://x.com/arankomatsuzaki/status/1807593343007818065)已从 Web 数据中自动策划完成，旨在扩展合成数据的创建，使 **MATH 基准测试大幅提升**，从 49.6 提高到 64.9。该方法利用这些 Personas 在各种场景中促进多样化且可扩展的合成数据生成。
   - [Persona Hub GitHub 仓库](https://github.com/tencent-ailab/persona-hub)和 [arXiv 论文](https://arxiv.org/abs/2406.20094)详细介绍了所使用的方法论，展示了包括合成**高质量数学问题**、逻辑推理任务以及游戏 NPC 在内的用例。
- **PhyloLM 将 Phylogenetics 引入 LLMs**：[PhyloLM](https://arxiv.org/abs/2404.04671) 介绍了一种将 Phylogenetic 算法适配到 Large Language Models (LLMs) 的方法，以探索它们之间的关系并预测性能特征。该方法利用 LLMs 输出的相似性指标构建树状图（dendrograms），捕捉了 111 个开源模型和 45 个闭源模型之间的已知关系。
   - 该方法的 Phylogenetic 距离成功预测了标准基准测试中的性能，[验证了其功能实用性](https://arxiv.org/html/2404.04671)，并能够在没有透明训练数据的情况下实现低成本的 LLM 能力评估。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.04671">PhyloLM : Inferring the Phylogeny of Large Language Models and Predicting their Performances in Benchmarks</a>: 本文介绍了 PhyloLM，一种将 Phylogenetic 算法适配到 Large Language Models (LLMs) 的方法，用于探索它们之间是否存在关联、如何关联，并预测其性能特征...</li><li><a href="https://x.com/arankomatsuzaki/status/1807593343007818065">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>: 通过 1,000,000,000 个 Personas 扩展合成数据创建 - 展示了从 Web 数据中自动策划的 10 亿个多样化 Personas 集合 - MATH 显著提升：49.6 -> 64.9  仓库: https://g...</li><li><a href="https://arxiv.org/abs/2406.20094">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>: 我们提出了一种新型的 Persona 驱动的数据合成方法论，利用 LLM 中的各种视角来创建多样化的合成数据。为了充分利用这一方法论...
</li>
</ul>

</div>

### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1256455180774408192)** (4 messages): 

- **探索 Android and Human 数据集中的梦境**：查看 [Android and the Human 数据集](https://huggingface.co/datasets/gustavecortal/the-android-and-the-human)，其中包含来自 DreamBank 的 10,000 个真实梦境和使用 **Oneirogen 模型** 生成的 10,000 个梦境。该资源允许区分真实梦境与生成梦境，并支持分类任务。
   - 生成的梦境是使用 **Oneirogen (0.5B, 1.5B, 和 7B)** 语言模型产生的，为分析梦境叙事提供了基准。该数据集是研究真实梦境与合成梦境内容差异的理想选择。
- **Snorkel AI 的聊天优化模型**：Snorkel AI 的 [Snorkel-Mistral-PairRM-DPO](https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO) 现已可用于聊天，并在 [Together AI playground](https://api.together.xyz/playground/chat/snorkelai/Snorkel-Mistral-PairRM-DPO) 上进行实时测试。该模型也可通过 Together AI 的 API 获取，以供更广泛的使用。
   - 最近的一篇 [Snorkel AI 博客文章](https://snorkel.ai/new-benchmark-results-demonstrate-value-of-snorkel-ai-approach-to-llm-alignment/) 详细介绍了该模型的对齐优势。得益于 Together AI 团队的集成工作，该模型在 HF 的 7B 模型文本推理端点上以标准速度运行。
- **PersonaHub 彻底改变数据合成**：[PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub) 提供了一种角色驱动（persona-driven）的数据集方法论，通过超过 10 亿个角色来扩展合成数据的创建。这些角色涵盖了多种视角，促进了各种应用场景下的丰富数据合成。
   - 在论文 *Scaling Synthetic Data Creation with 1,000,000,000 Personas* 中介绍，该方法论通过模拟复杂的指令、逻辑和数学场景来增强 LLM 训练。PersonaHub 在为复杂研究和实际应用生成合成数据方面展示了通用性和可扩展性。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO">snorkelai/Snorkel-Mistral-PairRM-DPO · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/gustavecortal/the-android-and-the-human">gustavecortal/the-android-and-the-human · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/proj-persona/PersonaHub">proj-persona/PersonaHub · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1256328863865241630)** (10 messages🔥): 

- **实验室培育血液进入临床试验**：剑桥大学的研究人员已经开始了全球首个用于输血的**实验室培育红细胞**临床试验。观看 [YouTube 视频](https://youtu.be/o0IM-FcX_9U) 了解更多详情。
   - 围绕该话题的兴奋和幽默讨论包括：吸血鬼更喜欢非人工血液，以及**阻碍研究进展**。
- **Open Model Initiative 启动**：**Open Model Initiative** 已经启动，旨在推广用于图像、视频和音频生成的开源 AI 模型。在 [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1do5gvz/the_open_model_initiative_invoke_comfy_org/) 上阅读完整公告。
   - 该倡议旨在生产具有开放许可的高质量、有竞争力的模型，确保所有人都能**免费且不受限制地访问**。
- **使用 Langchain 实现 Mixture of Agents**：[YouTube 视频](https://www.youtube.com/watch?v=VNy7CM23WA0) 探讨了如何使用 langchain 实现 **Mixture of Agents (MoA)**。这种方法旨在利用多个 Agent 的集体优势。
   - 视频提供了一个动手实践教程，重点关注使用各种 Agent 协作以提高功能和性能。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=VNy7CM23WA0">Mixture of Agents (MoA) using langchain</a>: Today we will implement mixture of agents using langchain.We introduce Mixture of Agents (MoA), an approach to harness the collective strengths of multiple L...</li><li><a href="https://youtu.be/o0IM-FcX_9U?feature=shared">First ever clinical trial of lab-grown red blood cell transfusion</a>: Cambridge researchers are taking part in the world’s first clinical trial of red blood cells that have been grown in a laboratory for transfusion into anothe...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1do5gvz/the_open_model_initiative_invoke_comfy_org/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1256373456300146718)** (60 条消息🔥🔥): 

- **Gemma 2 9B 挑战 Phi3 medium**：讨论集中在 **Gemma 2 9B** 在合成生成方面是否优于 **Phi3 medium**，相关的基准测试对比见于 [Hugging Face](https://huggingface.co/google/gemma-2-9b)。成员们辩论了包括上下文长度在内的优劣，但尚未达成明确共识。
   - 成员们指出 **Phi3** 拥有更大的上下文窗口，这让人们对 **Gemma 2 9B** 的竞争优势持怀疑态度。对 **Gemma 2 9B** 的初步评估 (vibe check) 仍在进行中，多位用户正在测试其性能并提供反馈。
- **SpecExec 加速 LLM 推理**：[SpecExec](https://www.together.ai/blog/specexec) 承诺为 LLM 推理提供投机解码 (speculative decoding)，在消费级 GPU 上通过 4-bit 量化实现每秒 4-6 个 token 的速度。该方法可能将推理速度提高多达 18.7 倍，使其成为在消费级硬件上高效运行大型模型的极具吸引力的选择。
   - 该技术涉及使用草稿模型 (draft model) 预测 token 序列，然后由主模型快速验证。讨论围绕适用于投机解码的模型系列以及兼容词表大小 (vocab sizes) 的重要性展开。
- **Meta Chameleon 对比 CogVLM2 和 LlavaNext 进行视觉描述**：用户寻求视觉描述 (vision-captioning) 模型的推荐，权衡了包括 **Meta Chameleon**、**CogVLM2** 和 **LlavaNext** 在内的选项。报告指出 **Florence 2 large** 也加入了竞争，但在某些情况下其详细描述能力可能稍逊一筹。
   - 成员们提醒道：“查询提示词 (Query prompts) 会显著影响视觉语言模型的基准测试结果。” 详细讨论强调了 **Florence 2** 可能不是最佳选择，并由于输出的可变性，强调了亲自测试的重要性。
- **FireFunction V2：新型函数调用模型**：由 Fireworks 推出的 **FireFunction V2** 提供了最先进的函数调用 (function calling) 能力，在函数调用评分上达到 0.81（对比 GPT-4o）。它基于 **Llama 3** 构建，支持并行函数调用，并可同时处理多达 20 个函数规范，旨在改进指令遵循任务。
   - [详细信息](https://huggingface.co/fireworks-ai/llama-3-firefunction-v2)显示其较前代有显著改进，使其成为函数调用需求的稳健选择。该模型保留了 Llama 3 强大的对话能力，同时在多轮对话和结构化信息提取方面引入了新的效率。
- **Hugging Face 上的 UGI 排行榜揭晓**：Hugging Face 上 [Uncensored General Intelligence Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)（无审查通用智能排行榜）的发布引发了关注。该排行榜根据易用性和通用智能对模型进行排名，为最新且最有效的模型提供见解。
   - 具体而言，**Teknium** 质疑像 **L3 70B** 这样经过 Abliterated 处理（移除拒绝机制）的模型与基础版 **Mixtral** 相比排名如何，因为模型拒绝行为存在差异。讨论强调了社区对 AI 模型基准测试和对比性能评估的参与度。

<ul>
<li>
<a href="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard">UGI Leaderboard - DontPlanToEnd 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - open-llm-leaderboard 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/fireworks-ai/llama-3-firefunction-v2">fireworks-ai/llama-3-firefunction-v2 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/WABetaInfo/status/1806101428609622181">来自 WABetaInfo (@WABetaInfo) 的推文</a>：📝 WhatsApp Android 测试版 2.24.14.7：有什么新内容？WhatsApp 正在开发一项选择 Meta AI Llama 模型的功能，并将在未来的更新中推出！https://wabetainfo.com/whatsa...</li><li><a href="https://x.com/nicolas__yax/status/1807761080917045489">来自 Nicolas Yax (@nicolas__yax) 的推文</a>：从相似性矩阵中，我们可以计算出树状图。这些图能够准确捕捉模型之间不同的微调关系（例如 OH @Teknium1 @maximelabonne 等人）当...</li><li><a href="https://x.com/honghuazhang2/status/1806727439823102325?s=46">来自 Honghua Zhang (@HonghuaZhang2) 的推文</a>：提出 Ctrl-G，一个神经符号框架，使任意 LLM 能够以 100% 的保证遵循逻辑约束（长度控制、填充……）。Ctrl-G 在文本编辑任务上击败了 GPT4...</li><li><a href="https://www.together.ai/blog/specexec">SpecExec：用于消费级设备上交互式 LLM 推理的大规模并行推测解码</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2305.17333v3">仅通过前向传递微调语言模型</a>：微调语言模型 (LM) 在各种下游任务上取得了成功，但随着 LM 规模的增长，反向传播需要极大量的内存。零阶 (ZO) 方法可以...</li><li><a href="https://x.com/NousResearch/status/1793637803701780797">来自 Nous Research (@NousResearch) 的推文</a>：Nous Research 正在招聘！在此申请：https://forms.gle/UWx2Pht8qioi1bjAA</li><li><a href="https://x.com/rauchg/status/1806899014312595948">来自 Guillermo Rauch (@rauchg) 的推文</a>：如果你在没有任何额外上下文的情况下问 Google Chrome 的嵌入式模型 Gemini Nano “他是谁”，它会可靠地回答 @elonmusk 🤨</li><li><a href="https://huggingface.co/google/gemma-2-9b">google/gemma-2-9b · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1256903083829624882)** (6 条消息): 

- **10k DPO 数据集大小验证**：一位成员询问约 10k 个样本对于 DPO 数据集是否足够。另一位成员确认，虽然样本越多越好，但 **10k 已经相当不错**，并且能很好地对齐模型。
   - 同一位成员分享说，他们从 10k 个样本开始，并观察到随着逐渐增加数据集大小，模型有了显著改进。
- **关于 Hermes Gemma 27b 的推测**：在关于其发布或存在的问题中简要提到了 **Hermes Gemma 27b**。关于此话题没有提供额外的细节或确认。
   - 没有关于 **Hermes Gemma 27b** 的进一步讨论，其状态和规格仍存疑问。
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1256356923113214004)** (163 条消息🔥🔥):

- **Self-Retrieval 论文深度解析**：针对 [self-retrieval 论文](https://arxiv.org/abs/2403.00801) 的详细讨论检查了其数据集构建和训练方法，并指出了训练过程中可能存在重复的担忧。具体而言，文档中的每个句子都被索引，且文档本身被配对用于训练，这可能导致模型过拟合。
   - 对话强调了潜在的缺点和过拟合问题，一位成员表达了挫败感，说道：*“这个我懂，简单的 next token prediction；至少我觉得我懂了”。* 另一位成员由于这些担忧决定不使用该论文的方法。
- **面向 HF 的创新数据集搜索工具**：一位成员分享了一个新更新的数据集搜索工具 [holdingfaceSEARCH](https://github.com/EveryOneIsGross/holdingfaceSEARCH)，它可以根据选定的键下载并搜索 HF 数据集，支持为 RAG 任务生成本地 embeddings。该工具旨在通过实现“no frills”的 RAG-chat 来简化流程，以便快速摄取和搜索。
   - 开发者强调了该工具的灵活性，指出其能够满足不同的数据集需求，并鼓励其他成员分享他们的项目和工具。这引发了关于高效数据集处理技术的更广泛讨论。
- **Groq 与定制芯片的未来**：Groq 成为 AI 硬件领域备受关注的参与者，可能为 70B 模型提供每秒 1000 个 tokens 的速度，挑战 Nvidia 的主导地位。对话表明，如果 Groq 能提供有效的训练和 serving 能力，可能会彻底改变 AI 计算市场。
   - 成员们讨论了竞争格局，指出 Nvidia 的 CUDA 在训练应用方面仍具有优势。关于 LPUs 的未来也引发了争论，一位成员强调：*“我希望 Groq 能成功。我仍然认为 etched 只是一个营销噱头”。*
- **Microsoft 与 OpenAI 的巨型数据中心**：据 [The Information](https://www.theinformation.com/articles/microsoft-and-openai-plot-100-billion-stargate-ai-supercomputer) 报道，Microsoft 和 OpenAI 正在合作开展一个名为 Stargate 的巨型数据中心项目，耗资可能超过 1000 亿美元。这一举措与 Microsoft 利用核能为其耗能巨大的 AI 雄心提供动力的战略相一致。
   - 讨论还涉及了这样一个宏大项目可能面临的环境和物流挑战，暗示其对未来数据中心的能源领域将产生重大影响。
- **Anthropic 与 OpenAI 的市场策略对比**：成员们称赞了 Anthropic 的方法，并将其与 OpenAI 进行了积极对比。讨论公认 Anthropic 尚未面临重大的公众投诉，不像 OpenAI，其联合创始人已经开始在别处寻求更安全的 superintelligence 方案。
   - 一位成员幽默地评论道：*“看到那个新初创公司了吗？Sutskever 的？”*，这反映了他们对 Anthropic 目前发展轨迹的偏好。这体现了 AI 公司之间更广泛的行业动态和战略转变。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2023/9/26/23889956/microsoft-next-generation-nuclear-energy-smr-job-hiring">微软正转向核能以支持其 AI 雄心</a>：微软正在招聘人员来领导其核能战略。</li><li><a href="https://arxiv.org/abs/2406.19292">从人造针到真实的干草堆：通过合成数据微调提升 LLM 的检索能力</a>：最近的研究表明，大语言模型 (LLM) 在处理长上下文输入时，难以准确检索信息并保持推理能力。为了解决这些限制...</li><li><a href="https://arxiv.org/abs/2403.00801">Self-Retrieval：使用单个大语言模型构建信息检索系统</a>：大语言模型 (LLM) 的兴起改变了信息检索 (IR) 系统在人类获取信息方式中的角色。由于孤立的架构和有限的...</li><li><a href="https://www.complex.com/music/a/jaelaniturnerwilliams/the-weeknd-interviews-10-year-old-self">The Weeknd 采访 10 岁 AI 版自己的推文，告诉他“你经常说‘超级力量’”</a>：当 The Weeknd 使用 OpenAI 采访年轻时的自己时，事情变得有些奇妙。</li><li><a href="https://github.com/tencent-ailab/persona-hub">GitHub - tencent-ailab/persona-hub</a>：通过在 GitHub 上创建账户，为 tencent-ailab/persona-hub 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2310.11511">Self-RAG：通过自我反思学习检索、生成和评判</a>：尽管大语言模型 (LLM) 具有卓越的能力，但由于仅依赖其封装的参数化知识，它们经常产生包含事实错误的回应。检索...</li><li><a href="https://github.com/EveryOneIsGross/holdingfaceSEARCH">GitHub - EveryOneIsGross/holdingfaceSEARCH：一个用于下载和搜索 HF 数据集的简单导入或 CLI 工具</a>：一个用于下载和搜索 HF 数据集的简单导入或 CLI 工具 - EveryOneIsGross/holdingfaceSEARCH</li><li><a href="https://www.forbes.com/sites/cindygordon/2024/03/31/microsoft-and-openai-partnering-on-stargate-a-100b-us-data-center/">据报道，微软和 OpenAI 合作建设价值 1000 亿美元的美国数据中心</a>：对生成式人工智能飙升的需求加速了对能够处理更高级任务的以 AI 为中心的数据中心的需求。</li><li><a href="https://docs.google.com/spreadsheets/d/1f5fbPxhjGrmPqhbM0exOCX2vAzffRWufhF7QBp24OMw/edit?gid=0#gid=0">RAG 数据合成</a>：Sheet1 领域、课程文件、来源/链接、HF 仓库、大小（行）、状态、负责人、审核人、审核笔记 Websearch Wikipedia Codebase, WIP, Bexboy Academic Papers Books, WIP, EveryoneIsGross Finance...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1256513728971804724)** (31 messages🔥): 

- **WorldSim 中的命令问题**：用户报告了在 **WorldSim** 中使用 **!save** 命令的问题，该命令导致了数据清空而不是保存讨论。另一个问题涉及 **!back** 命令，它清空了整个聊天记录，而不是返回到上一次对话。
   - Apyh 确认了这两个问题并保证会进行调查，其中 **!back** 已经修复。Vuyp 还注意到使用 **Sonnet 3.5** 时，生成内容在一行后会随机停止。
- **Claude 3.5 Sonnet 支持已推出**：**WorldSim** 现在支持 Claude 3.5 Sonnet 模型选项。此更新发布时带有谨慎提示，提到它在 **World Client** 中表现出色，但在 **WorldSim** 中创意稍逊。
   - Apyh 和其他用户讨论了 **Claude 3.5 Sonnet** 往往会突然停止生成响应的问题，Apyh 保证正在开发新的 Prompt 以解决此行为。
- **Beta 测试者的额度查询**：Rundeen 询问额度为何从 $50 降至 $4，并询问是否存在每周津贴制度。Apyh 澄清说，只有使用与正式版相同的电子邮件注册的 Beta 测试者才会获得免费额度，并表示愿意帮助迁移额度。
   - Apyh 邀请 Rundeen 私信其账号 ID 或电子邮件以解决额度问题，展示了亲力亲为的客户支持方式。
- **WorldSim 中的意外重置**：Keksimus.maximus 因点击 **WorldClient** 按钮而不小心重置了涉及龙的 **WorldSim** 场景。Apyh 解释了如何使用 `**!list**` 和 `**!load**` 命令来恢复之前的聊天。
   - 这次重置事件凸显了对清晰指令的需求，以及可能需要改进 UI 以防止在 **WorldSim** 中发生意外重置。
- **未来 WorldSim Bot 的增强功能**：Rezonaut 建议针对问答功能强化一些 Bot，使其能够链接到 Nous 的各个领域及其可用资源。Teknium 将此类 Bot 的使用引导至了相应的频道。
   - 对话指出了 Bot 功能可能存在的增强方向，突显了为改善 Nous 用户体验所做的持续努力。
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1257051471598321684)** (7 messages): 

- **“错误即值”引发辩论**：成员们讨论了在 Mojo 中使用 `Variant[T, Error]` 将**错误作为值**（errors as values）的处理方式，并指出了当前的局限性。**Errors** 和 **Exceptions** 被对比为具有不同处理挑战的不同类型。
   - 一位用户评论道：“Mojo 的 try/except 是语法糖”，强调了 Mojo 内部处理错误方式的区别。缺少 **match 语句**也因阻碍了优雅的错误处理而受到批评。
- **在 Ubuntu 上设置 Mojo**：一位新用户寻求在 **Ubuntu** 上设置 **Mojo** 的帮助，特别提到了遇到的问题。另一位成员分享了在 **Raspberry Pi 5** 的 **Ubuntu 24.04** 上成功运行 Mojo 的经验并表示愿意提供帮助。
   - 故障排除请求凸显了新用户在不同 **Ubuntu 版本**上设置 Mojo 时遇到的常见挑战。社区支持在解决这些问题中发挥了关键作用。
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1257366341191471115)** (1 messages): 

- **新的 Mojo 编程挑战频道上线**：一个用于[每月 Mojo 编程挑战](https://discord.com/channels/1087530497313357884/1255303604894437388)的新频道已上线。社区成员可以通过点击 <:mojomarathon:1255306448632807424> 表情符号进行回应来订阅提醒。
   - 每月挑战由一位社区成员、版主以及 `basalt` ML Framework 的维护者组织。更多详情请见频道和 [GitHub 仓库](https://github.com/Benny-Nottonson/Mojo-Marathons)。
- **订阅每月 Mojo 挑战提醒**：成员可以通过点击 <:mojomarathon:1255306448632807424> 表情符号来获取新挑战的提醒。此功能可确保你及时了解最新的 Mojo 编程挑战。
   - 这些挑战提供了一个与社区互动并提升你在 `basalt` ML Framework 中技能的绝佳机会。查看 [GitHub 仓库](https://github.com/Benny-Nottonson/Mojo-Marathons)了解更多详情。

**提到的链接**：<a href="https://github.com/Benny-Nottonson/Mojo-Marathons">GitHub - Benny-Nottonson/Mojo-Marathons</a>：通过在 GitHub 上创建账号来为 Benny-Nottonson/Mojo-Marathons 的开发做出贡献。

  

---

### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1256346953474768947)** (32 条消息🔥): 

- **PyTorch 与 Mojo 见解**：**Mojo 中的 PyTorch**：讨论了在 Mojo 中使用 PyTorch，强调了其灵活性以及在 AI 开发者中的普及程度。分享了一部关于 PyTorch 演变和影响的 [YouTube 纪录片](https://www.youtube.com/watch?v=rgP_LBtaUEc)。
   - **快速迭代的采用**：一个有趣的观点是，快速迭代最初促成了 PyTorch 的采用，而性能改进则是在后期阶段进行的。
- **TacoBot 的复活**：**ModularBot 退役，TacoBot 引入**：ModularBot 已退役，随后作为 TacoBot 重新引入，用户们幽默地与新机器人互动。TacoBot 依然有趣地分享着它对塔可（tacos）的热爱。
   - **TacoBot 头像**：用户建议并赞赏了由 ChatGPT 提供的 TacoBot 新头像，这似乎完美地捕捉到了它的神韵。
- **Facebook 的新 LLM Compiler**：**Facebook 的 LLM Compiler**：热烈讨论了 Facebook 专门用于编译器优化的新 [LLM Compiler 模型](https://huggingface.co/facebook/llm-compiler-13b-ftd)。它已在各种汇编代码上进行了训练，并能复制 clang 编译器的功能。
   - **免费使用**：提到 LLM Compiler 可用于研究和商业用途，有两个版本分别满足基础模型和微调模型的需求。
- **Cody 支持 Mojo**：**Cody 与 Mojo 集成**：用户讨论了使用 Cody 辅助 Mojo 编程，注意到由于其类 Python 的语法，Cody 有潜力推测语言特性。
   - **测试高级功能**：提到了一个计划，即使用更高级的、针对 Mojo 的任务（特别是与 SIMD 相关的任务）来测试 Cody，以评估其能力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=rgP_LBtaUEc">Official PyTorch Documentary: Powering the AI Revolution</a>：这部影片揭示了 PyTorch 诞生的真实叙事，将其存在归功于一群推动技术创新的无名英雄...</li><li><a href="https://huggingface.co/facebook/llm-compiler-13b-ftd">facebook/llm-compiler-13b-ftd · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1256368212367507577)** (149 messages🔥🔥): 

- **Mojo 讨论 I/O 和 Async API**：成员们讨论了 Mojo 目前 I/O 模块的局限性，特别是需要通过 Python 接口来读取 stdin。对话转向了未来 Async API 的改进，许多人主张使用 io_uring 以获得更好的性能。
   - Darkmatter__ 强调了高性能 I/O 子系统对于高速网络的重要性，认为尽管 io_uring 和基于完成（completion-based）的 API 较为复杂，但仍应优先考虑。Lukashermann.com 则反驳称，对于普通用户来说，可维护性和易理解性也至关重要，并建议采用更简单的抽象。
- **Mojo 作为系统语言的未来**：Twilight_muse 和 darkmatter__ 对 Mojo 彻底改变系统编程的潜力表示兴奋，赞扬了其 MLIR 后端和所有权模型（ownership model）。他们强调 Mojo 需要吸取 Rust 的教训，特别是在 Async I/O 实现方面，以成为一个强大的系统语言。
   - Darkmatter__ 分享了关于网络性能瓶颈的见解，并强调了 io_uring 相比传统基于轮询（polling-based）的 I/O 的优势。讨论强调了 Mojo 如何弥合高性能系统编程与易用的通用编程之间的差距。
- **Sum Types 与 Union Types 之争**：Nick.sm 和 soracc 辩论了 Sum Types（和类型/枚举）与 Union Types（联合类型）的优缺点。Nick.sm 认为需要描述语义相关的不同选项集。Soracc 建议 Sum Types 是植根于直觉类型论（intuitionistic type theory）的基础概念，而 Nick.sm 将其与继承进行了比较，暗示了它们的局限性。
   - 讨论转向了实际应用，以及 Mojo 如何实现这些类型以避免其他语言中遇到的问题。这场持续的辩论反映了 Mojo 在语言设计中平衡数学严谨性与实际可用性的深度思考。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/get-started">Get started with Mojo🔥 | Modular Docs</a>：立即安装 Mojo 并开始开发</li><li><a href="https://docs.modular.com/mojo/manual/python/">Python integration | Modular Docs</a>：同时使用 Python 和 Mojo。</li><li><a href="https://learn.microsoft.com/en-us/windows/win32/api/ioringapi/">ioringapi - Win32 apps</a>：提供用于创建和管理 I/O rings 的 API。</li><li><a href="https://www.youtube.com/watch?v=7qvVMUSxqz4">2024 EuroLLVM - How Slow is MLIR</a>：2024 欧洲 LLVM 开发者大会。演讲者：Mehdi Amini, Jeff Niu。幻灯片：https://llvm.org/devm...</li><li><a href="https://github.com/modularml/mojo/discussions/3049">Self-holding and open-sourced, which comes first? · modularml/mojo · Discussion #3049</a>：据我所知，Mojo 编译器是用 C++ 编写的。我想知道 Mojo 是会先实现自举（用 Mojo 编写编译器），还是先将编译器开源。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1257368905618292806)** (7 messages): 

- **为 Mojo Marathons 设置环境**：一名成员请求关于 Mojo Marathons 环境设置的**详细信息**，包括核心隔离（core isolation）和定时器滴答（timer tick）配置。他们提出协助设置，并提到将定时器切换到 **ARM MSRs** 以避免系统调用开销。
   - 另一名成员回应称，他们现有一个用于 Benchmark 的脚本，并有兴趣交流心得。他们计划写下建议并提交一个 PR，以使用 **ARM MSRs 或 rtdsc**。
- **定时器滴答隔离的影响**：一名成员提到 **timer tick isolation** 并不常用，因为它需要在启动时配置，并且会干扰大多数 Benchmark 库。但他们指出，这可以避免**随机中断**。
   - 他们还表示，除非有人尝试将此过程**并行化**，否则设置应该很直接，主要涉及内核命令行参数和 **taskset** 的使用。
- **使用 libhugetlb 隔离 TLB 状态**：一名成员建议使用 **libhugetlb** 来隔离 Benchmark 的 TLB 状态。他们暗示这将增强 Benchmark 结果的一致性。
   - 对话没有进一步深入探讨 **libhugetlb** 使用的细节，但提到了它在维持隔离性和一致性方面的潜在好处。
  

---

### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1256636887276326973)** (6 条消息): 

- **测试新的应用板块 (apps section)**：一位成员提到他们正在尝试使用**新的应用板块**，但没意识到它会出现在聊天中。
   - 另一位成员询问了 **apps section** 的具体含义，表明对这一新功能存在一些困惑或了解不足。
- **版主处理垃圾信息**：一位成员举报了**垃圾信息 (spam)**，并被告知如果自动审核系统未能拦截，请直接 ping 版主。
   - 垃圾信息随后被一名版主迅速处理，该版主向成员表示感谢，并鼓励他们继续举报此类事件。
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1256492979745591296)** (28 条消息🔥): 

- **Mojo Nightly 编译器已更新**：发布了新的 nightly Mojo 编译器 `2024.6.2905`，包含多项更新，包括为 `UInt` 新增的 `min` 和 `max` 重载，以及 `UInt` 类型更新日志。点击[此处](https://github.com/modularml/mojo/compare/439d86d608d3b6c12cead112eb651752ba1ad40d...9cf3a83e0eb661a2263ae117c921bb004df6721c)查看原始 diff，以及[当前的更新日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。
   - 发现了最近 nightly 版本中的问题，目前正在修复中。导致内部依赖问题的更改已被回滚，以确保未来版本的稳定性。
- **寻求 Stdin 流测试方法**：用户讨论了测试涉及 Stdin 流的函数的方法，例如为文件中的每个测试写入 Stdin。建议包括使用独立进程、线程或 `tee(2)` 等工具。
   - 还提到了处理 EOF 等额外复杂性，强调了在测试之间清除 Stdin 的潜在需求。
- **Mojo 中 TLS 的挑战**：一位成员询问如何使用 Mojo 构建一个必须处理带 TLS 的 JSON HTTP 请求的 Linux CLI 工具。社区指出，目前 Mojo 中的 IO 通常涉及调用 C 或 Python，这使得 TLS 集成特别具有挑战性。
   - 鉴于 Mojo 的 FFI 尚不成熟，封装 OpenSSL 或 BoringSSL 等库会很困难。这为任何需要 TLS 的开发者带来了显著障碍。
- **Mojo 在早期挑战之外的潜力**：尽管目前存在局限性，成员们对 Mojo 替代 Java 或 .NET 等语言用于商业应用的潜力感到兴奋。对 Wasm 作为编译目标的期待很高。
   - 尽管承认存在成长的烦恼，贡献者们对 Mojo 功能的逐步改进和增强仍保持乐观。
- **Mojo 标准库贡献者有资格获得周边**：Mojo 标准库的贡献者可以联系团队成员领取 Mojo 周边商品。Jack Clayton 在聊天中回应询问时确认了这一点。
   - 鼓励参与者积极参与社区并为库做贡献，以获得这些奖励的机会。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=DJtef410XaM">Python 中的整洁架构 (The Clean Architecture in Python)</a>：Brandon Rhodes。即使是注重设计的程序员也会发现……</li><li><a href="https://docs.modular.com/mojo/stdlib/tensor/tensor/Tensor#__init__">Tensor | Modular 文档</a>：一种拥有底层数据并以 DType 为参数的 Tensor 类型。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1256662493963751487)** (9 messages🔥): 

- **Mojo Marathons 正式开启！**: 欢迎来到 **Mojo Marathons**！由 [@Benny-Nottonson](https://github.com/Benny-Nottonson) 主办，这项每月竞赛挑战 Mojicians 展示他们的技能，以赢取奖品和社区认可。
   - 第一个挑战涉及使用 **Mojo 创建优化的矩阵乘法算法**，参与者需 fork [仓库](https://github.com/Benny-Nottonson/Mojo-Marathons)并在月底前提交他们的解决方案。
- **提高精度的基准测试建议**: 一名成员建议使用 **stdlib benchmark 模块**，以便在 Mojo Marathons 挑战中获得更精确的结果。这包括多次运行操作并进行预热（warmups）的工具。
   - [@Benny-Nottonson](https://github.com/Benny-Nottonson) 表示同意，并邀请该成员编写一个改进的测试模块并提交 PR。
- **main.mojo 的编译问题**: 一名成员在 7 月的挑战中编译 `main.mojo` 时遇到错误。错误指向 `DTypePointer[Type, 0]` 中缺少 'load' 和 'store' 属性。
   - [@Benny-Nottonson](https://github.com/Benny-Nottonson) 澄清说，代码旨在最新的 **Mojo** 稳定版本（stable build）上运行，而非 nightly 版本。
- **矩阵乘法函数输出问题**: 一名成员在比较两个矩阵乘法函数的输出时遇到了 *AssertionError*，`24.4375` 和 `24.453125` 之间存在 `0.015625` 的差异。
   - [@Benny-Nottonson](https://github.com/Benny-Nottonson) 确认这表明用户的函数输出不正确。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/benchmark/bencher/ThroughputMeasure">ThroughputMeasure | Modular Docs</a>: 记录 BenchMetric 指标和值的吞吐量度量。</li><li><a href="https://github.com/Benny-Nottonson/Mojo-Marathons/blob/main/July2024/src/test.mojo#L72">Mojo-Marathons/July2024/src/test.mojo at main · Benny-Nottonson/Mojo-Marathons</a>: 通过在 GitHub 上创建账户来为 Benny-Nottonson/Mojo-Marathons 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1257248877497552948)** (3 messages): 

- **OpenRouter 上的数据统计暂时下线**: 由于**数据库操作失误**，我们在 [OpenRouter](https://openrouter.ai/rankings) 上的数据统计暂时下线。团队正在努力修复数据，并对带来的不便深表歉意。
   - 该错误不会影响客户数据和积分（credit）。用户收到了“对带来的不便表示抱歉”的确认消息。
- **客户数据不受影响**: 尽管 [OpenRouter](https://openrouter.ai/rankings) 的统计功能下线，但**客户数据和积分仍不受影响**。用户已获知团队正在解决此问题。
   - 通知中包含了对造成不便的歉意，并向用户保证**客户数据安全**完好无损。

**提及的链接**: <a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>: 根据各应用的用量对语言模型进行排名和分析

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

wyverndryke: 伙计们，这些太棒了！ 😄

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1256356791814455306)** (155 条消息🔥🔥): 

- **OpenRouter API 响应质量问题**：一位用户担心在 OpenRouter 界面测试的提示词，通过 OpenRouter API 执行时会产生不同且质量较低的响应。调试建议包括检查请求示例（request sample）并确保其与 OpenRouter 上的设置匹配。
   - 其他成员讨论了如何正确设置请求示例，并考虑使用 Bedrock 或其他受控模型以获得更高质量的响应。一位成员提到为 Bedrock 设置使用反向代理（reverse proxy）。
- **使用 Gemini 模型时的上下文差异**：一位用户发现网站上显示 Gemini 的上下文为 2.8M，但实际仅观察到 1M，从而导致困惑。另一位成员澄清说 OpenRouter 计算 Token 的方式不同，通常会更短。
   - 进一步的见解指出，与通常标准相比，OpenRouter 的 Token 计数结果更短，因此产生了感知上的差异。讨论了反馈和报告错误的替代途径。
- **DeepSeek Code V2 的准确性受到赞赏**：一位成员称赞通过 OpenRouter API 使用的 DeepSeek Code V2 在解决微积分问题和代码实现方面具有极高的准确性。他们认为该模型既高效又经济。
   - 另一位成员确认该模型是完整的 263B 版本，因为它通过 DeepSeek API 进行路由，这表明了它在各种用例中的准确性和强大功能。提供了外部链接以获取有关该模型的更多详细信息。
- **Embedding 模型问题及本地替代方案**：一位成员询问 OpenRouter 对 Embedding 模型的支持，但另一位成员澄清说，由于 Embedding 成本低且存在兼容性问题，建议直接使用 API。
   - 建议使用本地 Embedding 模型，强调了它们在保持与已生成 Embedding 一致性方面的优势。HuggingFace 上的 Nomic 模型被推荐为热门选择。
- **使用 Sonnet 3.5 时出现 Mistral API 错误**：一位用户报告在 Aider 聊天中使用 Sonnet 3.5 时遇到了 Mistral API 错误，尽管他们并未使用 Mistral。有人建议回退（fallback）机制可能自动切换到了 Mistral。
   - 为了解决此问题，建议用户联系 Aider 的支持团队进行更具体的调试。该问题被标记为当主要请求被阻止时，可能会回退到不同的 API。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/deepseek/deepseek-coder">DeepSeek-Coder-V2 by deepseek</a>：DeepSeek-Coder-V2，一个开源的混合专家（MoE）代码语言模型。它是在 DeepSeek-V2 的中间检查点基础上，通过额外的 6 万亿 Token 进一步预训练而成的。</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-70b-instruct">Meta: Llama 3 70B Instruct by meta-llama</a>：Meta 最新的模型系列（Llama 3），推出了多种尺寸和版本。这个 70B 指令微调版本针对高质量对话用例进行了优化。它展示了强大的...</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>：未找到描述</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5">nomic-ai/nomic-embed-text-v1.5 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct">deepseek-ai/DeepSeek-Coder-V2-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://gitgud.io/khanon/oai-reverse-proxy#what-is-this">khanon / oai-reverse-proxy · GitLab</a>：适用于各种 LLM API 的反向代理服务器。功能包括 API 格式转换、用户管理、反滥用、API 密钥轮换、DALL-E 支持以及可选的提示词/响应日志记录。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/)** (1 条消息): 

salnegyeron: 不算太差，但我觉得 Cohere 的 Aya23 更好一些。
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1256338211668496506)** (74 条消息🔥🔥):

- **Adept 策略调整：联合创始人加入 Amazon**：Adept AI Labs 宣布了其策略和领导层的更新，包括 CEO David Luan 在内的联合创始人将加入 Amazon，正如其 [博客文章](https://www.adept.ai/blog/adept-update) 和更详细的 [GeekWire 文章](https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/) 所确认的。文章指出，Amazon 打算在非独占许可下使用 Adept 的技术，而 Adept 将继续作为一家独立公司运营。
   - *“那个链接比他们的博客文章更好地解释了发生的事情。”* 一位社区成员评论道，并强调博客文章令人困惑，导致读者对 Adept 和 Amazon 之间关系的具体性质产生猜测。
- **AI Engineer World’s Fair 回顾与经验教训**：AI Engineer World’s Fair (AIEWF) 的组织者收集了关于该活动的反馈，重点在于改进会议时长和整体物流。[讨论](https://github.com/swyxio/swyxdotio/issues/510) 强调了应如何调整来自 AI Engineer Summit 的限制，以便为演讲者提供更多时间进行深入演示。
   - 一位与会者建议延长活动时间或进行调查以收集更结构化的反馈，而另一位则强调需要为 Hackathon 或即兴演讲提供专门的空间，并参考了其他会议的成功模式。
- **Runway Gen 3 发布：高保真文本生成视频**：Runway 宣布其 Gen-3 Alpha 文本生成视频功能正式开放使用，并将其宣传为高保真、快速且可控视频生成的新前沿。他们通过其 [官方账号](https://x.com/runwayml/status/1807822396415467686) 分享了这一令人兴奋的更新。
   - 该功能现在可以在其 [网站](http://runwayml.com) 上供所有人使用，允许用户尝试并利用先进的视频生成能力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/AdeptAILabs/status/1806773469155381705?t=HevOdjCZ31VyPecgHs5KoQ&s=19">来自 Adept (@AdeptAILabs) 的推文</a>：今天，我们宣布了战略方面的一些更新，以及领导层和团队的一些变动。更多详情请见我们的博客：https://www.adept.ai/blog/adept-update</li><li><a href="https://x.com/lmsysorg/status/1807812671238258931">来自 lmsys.org (@lmsysorg) 的推文</a>：并非所有问题都需要 GPT-4！我们推出了 RouteLLM —— 一个基于人类偏好数据的路由框架，可将简单查询引导至更便宜的模型。通过数据增强技术，RouteLLM...</li><li><a href="https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/">亚马逊从资金充足的企业级 AI 初创公司 Adept 聘请创始人，以助力这家科技巨头的 “AGI” 团队</a>：(GeekWire 资料照片 / Kevin Lisota) 亚马逊正在通过聘请来自 Adept 的高管来加大其 AI 投入，Adept 是一家总部位于旧金山、致力于构建 “Agent” 的初创公司。</li><li><a href="https://x.com/ashvanth_s1/status/1806994830062493805)">来自 Ashvanth.S (@ashvanth_s1) 的推文</a>：正在阅读介绍 GPT-3 的论文《Language models are few shot learners》。这是在 2020 年，当时他们在训练时遇到了这些问题。很想阅读更多关于...</li><li><a href="https://maven.com/parlance-labs/fine-tuning">Mastering LLMs：由 Dan Becker 和 Hamel Husain 在 Maven 上举办的面向开发者和数据科学家的会议</a>：一个关于 LLM 方方面面的在线会议。</li><li><a href="https://x.com/karpathy/status/1807121265502965802?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Andrej Karpathy (@karpathy) 的推文</a>：@realSharonZhou 并购界的背越式跳高（Fosbury flop）</li><li><a href="https://x.com/runwayml/status/1807822396415467686">来自 Runway (@runwayml) 的推文</a>：Gen-3 Alpha 文本生成视频（Text to Video）现已向所有人开放。高保真、快速且可控的视频生成新前沿。立即在 http://runwayml.com 体验。</li><li><a href="https://asciinema.org/a/76uAJPY1825TxnuUlZ4yhvosu">无题</a>：由 wesen3000 录制</li><li><a href="https://asciinema.org/a/0fT2j4M8xWoJZ23A2PTjBTn4r">无题</a>：由 wesen3000 录制</li><li><a href="https://github.com/simonw/llm-cmd">GitHub - simonw/llm-cmd：在你的 shell 中使用 LLM 生成并执行命令</a>：在你的 shell 中使用 LLM 生成并执行命令 - simonw/llm-cmd</li><li><a href="https://lmsys.org/blog/2024-07-01-routellm/">RouteLLM：一个用于高性价比 LLM 路由的开源框架 | LMSYS Org</a>：<p>LLM 在一系列任务中展示了卓越的能力，但它们的成本和能力存在巨大差异，正如从...中所见。</p></li><li><a href="https://github.com/swyxio/swyxdotio/issues/510">筹备 2024 年 AI Engineer World's Fair · Issue #510 · swyxio/swyxdotio</a>：slug: aiewf-2024 我们刚刚结束了第一届 AI Engineer World's Fair 紧张的筹备期，这是由我的业务伙伴...发起的 AI Engineer 系列会议的大型多轨道形式。</li><li><a href="https://github.com/gorilla-llm/gorilla-cli">GitHub - gorilla-llm/gorilla-cli：适用于 CLI 的 LLM</a>：适用于 CLI 的 LLM。通过在 GitHub 上创建一个账号来为 gorilla-llm/gorilla-cli 的开发做出贡献。</li><li><a href="https://github.com/pgibler/cmdh">GitHub - pgibler/cmdh：在 shell 中通过自然语言创建 Linux 命令。</a>：在 shell 中通过自然语言创建 Linux 命令。 - pgibler/cmdh</li><li><a href="https://github.com/go-go-golems/geppetto">GitHub - go-go-golems/geppetto：Golang GPT3 工具</a>：Golang GPT3 工具。通过在 GitHub 上创建一个账号来为 go-go-golems/geppetto 的开发做出贡献。</li><li><a href="https://github.com/ShishirPatil/gorilla">GitHub - ShishirPatil/gorilla：Gorilla：LLM 的 API 商店</a>：Gorilla：LLM 的 API 商店。通过在 GitHub 上创建一个账号来为 ShishirPatil/gorilla 的开发做出贡献。</li><li><a href="https://x.com/iamgingertrash/status/1807798608374411630).">来自 simp 4 satoshi (@iamgingertrash) 的推文</a>：删除了我关于 Cursor 的推文 —— 看起来本地模式（local mode）只是个临时方案，他们从未打算让 Cursor 在本地使用。所以他们对自己的数据处理实践非常明确，因为它是...
</li>
</ul>

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1256325180741914694)** (1 条消息): 

- **OSS GPT Store 概览发布**：一名成员宣布，<@383468192535937026> 将在大约一小时后进行 **OSS GPT Store 概览**。
   - *提醒*：如果你在 **AIEF** <@&1254604002000244837> 忙得过来，请进入 <#1209303473263485011> 并领取提到的角色，以便接收未来的通知。
- **领取角色以获取未来通知**：温馨提醒，如果你想接收有关即将举行活动的通知，请在 <#1209303473263485011> 中领取提到的角色。
   - 这将帮助你及时了解 **OSS GPT Store** 的后续动态及其他相关更新。
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1256338676573667530)** (34 条消息🔥): 

- **GPT System Prompts 并非私有**：成员们讨论了 GPT System Prompts 是否公开，强调虽然它们不是*专门公开*的，但可以在[这个仓库](https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts)等地方找到。
   - 建议假设任何人都可以访问这些 Prompts，因此建议不要在 GPT 系统定义中包含任何秘密。
- **讨论 CoALA 框架**：关于 **Cognitive Architectures for Language Agents (CoALA)** 的最新论文（链接至 [arXiv](https://arxiv.org/abs/2309.02427)）引起了关注。
   - 该论文将现有的 Language Agents 组织成一个连贯的框架，并提出了未来可行的行动方向。
- **Awesome Language Agents 仓库**：分享了 [awesome-language-agents](https://github.com/ysymyth/awesome-language-agents) GitHub 仓库。
   - 该仓库根据 CoALA 论文列出了 Language Agents，为进一步探索提供了宝贵的资源。
- **AI 会议回顾**：有人提议对包含多个闪电演讲（lightning talks）的 **AI 工程师会议**进行回顾。
   - 成员们似乎对下次会议的这个想法很感兴趣，考虑到过去的演讲有很好的参与度和反馈。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/ysymyth/awesome-language-agents">GitHub - ysymyth/awesome-language-agents: 基于论文 "Cognitive Architectures for Language Agents" 的 Language Agents 列表</a>：基于论文 "Cognitive Architectures for Language Agents" 的 Language Agents 列表 - ysymyth/awesome-language-agents</li><li><a href="https://arxiv.org/abs/2309.02427">Cognitive Architectures for Language Agents</a>：最近的研究通过外部资源（如互联网）或内部控制流（如 Prompt Chaining）增强了 Large Language Models (LLMs)，以处理需要落地 (grounding) 或推理的任务...</li><li><a href="https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts">chatgpt_system_prompt/prompts/gpts at main · LouisShark/chatgpt_system_prompt</a>：GPT System Prompts 的集合以及各种 Prompt 注入/泄露知识。 - LouisShark/chatgpt_system_prompt
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1256494640383655976)** (97 条消息🔥🔥):

- **LangGraph React 和 FastAPI 示例**：一位用户询问了如何结合 React 前端和 FastAPI 后端使用 LangGraph 的示例。他们被引导至 [GitHub](https://github.com/langchain-ai/chat-langchain) 上的 *chat-langchain* 项目以及语义路由库 [Semantic Router](https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb)。
   - 建议通过创建一个 Agent 和 Tool 来处理用户输入，或者使用语义相似度方法进行路由决策。这些资源为实现这些技术提供了基础。
- **在 Tool 中访问可配置参数**：询问了如何在 Tool 中访问诸如 `thread_id` 之类的配置参数。发现可以通过 [langchain-community](https://v02.api.js.langchain.com/types/langchain_community_tools_gmail.GetThreadSchema.html) 中的特定 Tool Schema（如 *GetThreadSchema*）进行访问。
   - 关于使用 `.astream_events()` 流式传输 LangGraph 状态的进一步讨论建议，通过遍历事件来捕获最终状态，从而确保使用 `thread_id` 在多次执行之间进行状态管理。
- **LangGraph 中的 Streaming 和 Checkpointing**：一位用户询问了如何恢复 LangGraph 执行以及如何使用 Checkpoint 访问保存的状态。代码片段显示，Checkpoint 确保了之前的状态得以保留，并可以通过 `thread_id` 等唯一标识符进行访问。
   - 描述了使用 `.astream_events()` 进行实时状态更新的方法。引用了关键事件和配置来有效地管理状态转换，这对于需要历史上下文的 Tool 特别有用。
- **RAG 与语义相似度示例检索（Semantic Similarity Example Retrieval）的对比**：用户询问了 RAG 与语义相似度示例检索之间的区别。RAG 将检索式模型与生成式模型结合用于上下文路由，而相似度示例检索则根据 Embedding 选择相关的示例。
   - 这两种方法可以结合使用以构建稳健的应用，利用 RAG 进行路由，并利用相似度方法进行特定示例的选择。参考文档以获取更深入的理解。
- **LangGraph 状态管理技术**：提出了关于在流式传输期间和执行后管理及访问 LangGraph 状态的问题。状态信息可以通过 Checkpoint 系统进行访问和持久化，并讨论了利用已保存状态的 Tool。
   - 重点强调了 `stream` 方法和 `thread_id` 配置，以维持执行的连续性。提供的代码示例增强了这些技术的可应用性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.langflow.org/">Langflow - 创建你的 AI 应用！</a>: LangFlow 是一个 LangChain 的 GUI，采用 react-flow 设计，通过拖拽组件和聊天框提供了一种轻松的流程实验和原型设计方式。</li><li><a href="https://js.langchain.com/v0.2/docs/concepts/#routing>).">概念指南 | 🦜️🔗 Langchain</a>: 本节包含 LangChain 核心部分的介绍。</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#astream_events>).">概念指南 | 🦜️🔗 LangChain</a>: 本节包含 LangChain 核心部分的介绍。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/migrate_agent/#in-langgraph-2>).">如何从旧版 LangChain agents 迁移到 LangGraph | 🦜️🔗 LangChain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://github.com/langchain-ai/langchain/issues/12304>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/streaming/#using-stream-events>).">如何流式传输 runnables | 🦜️🔗 LangChain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#astream_events>)">概念指南 | 🦜️🔗 LangChain</a>: 本节包含 LangChain 核心部分的介绍。</li><li><a href="https://ai.google.dev/competition">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=i5cEuq9yuS4">Academia Braker，第二个演示。</a>: 这是我为 NVIDIA 和 LangChain 举办的 Generative AI Agents 开发者大赛创建的应用的第二个演示视频。#NVIDIADevContest #LangChain @nvi...</li><li><a href="https://github.com/langchain-ai/langchain/issues/16640>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/discussions/22529,">如何在 tool 中访问配置变量？ · langchain-ai/langchain · Discussion #22529</a>: 检查了其他资源，我为这个问题添加了一个非常详细的标题。我使用集成搜索搜索了 LangChain 文档。我使用 GitHub 搜索来查找类似的问题和...</li><li><a href="https://github.com/langchain-ai/chat-langchain">GitHub - langchain-ai/chat-langchain</a>: 通过在 GitHub 上创建账号，为 langchain-ai/chat-langchain 的开发做出贡献。</li><li><a href="https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb">semantic-router/docs/03-basic-langchain-agent.ipynb (main 分支) · aurelio-labs/semantic-router</a>: 超快速的 AI 决策和多模态数据的智能处理。 - aurelio-labs/semantic-router</li><li><a href="https://api.python.langchain.com/en/stable/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig>">langchain_core.runnables.config.RunnableConfig &mdash; 🦜🔗 LangChain 0.1.4</a>: 未找到描述</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/migrate_agent/#in-langgraph-2>)">如何从旧版 LangChain agents 迁移到 LangGraph | 🦜️🔗 LangChain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/migrate_agent/#basic-usage>).">如何从旧版 LangChain agents 迁移到 LangGraph | 🦜️🔗 Langchain</a>: 本指南假设你已熟悉以下概念：-</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/#retrieval-tool>)">如何添加聊天历史 | 🦜️🔗 LangChain</a>: 在许多问答应用中，我们希望允许用户进行来回对话，这意味着应用需要某种形式的过去问题和答案的“记忆”，以及一些逻辑...</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/#agent-constructor>)">对话式 RAG | 🦜️🔗 LangChain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/agents/#adding-in-memory>)">构建一个 Agent | 🦜️🔗 LangChain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/message_history/#invoking-with-config>)">如何添加消息历史 | 🦜️🔗 Langchain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/chatbot/#message-history>)">构建一个聊天机器人 (Chatbot) | 🦜️🔗 LangChain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/chatbot/#message-history>)">构建一个聊天机器人 (Chatbot) | 🦜️🔗 Langchain</a>: 概览
</li>
</ul>

</div>

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1256370823069962333)** (7 条消息): 

- **使用 Matryoshka Embeddings 提升检索速度**：Prashant Dixit 分享了一篇关于使用 **llama_index** 构建 Matryoshka RAG 的[帖子](https://x.com/Prashant_Dixit0/status/1806580075447590974)，旨在加速检索速度并减少内存占用。查看他的[教程](https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/tutorials/RAG-with_MatryoshkaEmbed-Llamaindex/RAG_with_MatryoshkaEmbedding_and_Llamaindex.ipynb)，其中使用了 **lancedb** 和 **huggingface**。
   - 他强调这些 embedding 模型提供多种维度（768, 512, 256, 128 和 64），可以**提升性能**并**减少内存使用**。
- **使用 EDA-GPT 自动化数据分析**：Shaunak 分享了他利用 **LLMs** 进行自动化数据分析的新 [GitHub 项目](https://github.com/shaunthecomputerscientist/EDA-GPT)。该项目可部署在 **Streamlit** 上或在本地设置，README 中附带了视频教程。
   - 该应用集成了 **LLMs** 以简化数据分析工作流。
- **社区包的 Todoist 集成**：mbbrainz 介绍了一个在 ChatGPT 帮助下创建的新 **Todoist 集成**，并寻求与社区分享。他建议[这个对话](https://chatgpt.com/share/e5c10477-0db9-4941-b684-5fc0ec5556e2)可能对创建完整的教程很有用。
   - 此集成作为社区包的潜在增强功能，提供了简化的任务管理。
- **LangChain 遇上 Postgres**：Andy Singal 发表了一篇关于将 **LangChain** 与 **Postgres** 集成的 [Medium 文章](https://medium.com/ai-advances/unleashing-the-power-of-persistence-langchain-meets-postgres-9cc7f069b260)。他探讨了如何通过结合使用这些技术来优化持久化。
   - 文章深入探讨了利用 **Postgres 的可靠性** 为 LangChain 项目提供稳健的持久化。
- **自动化竞争对手研究工具指南**：Sheru 分享了关于创建**自动化竞争对手研究员**的详细指南。这份分步指南展示了如何跟踪竞争对手在其网站上所做的任何更改——请点击[此处](https://dub.composio.dev/compete)查看。
   - 该工具旨在让你随时了解竞争对手的活动和网站修改。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/5-QV3lVI8uo">使用 Visual Agents 和 LangChain 与任何网页或应用聊天</a>: 在这个视频中，我展示了如何使用 OpenAI 和 LangChain RAG 构建你自己的 AI 专家，并带有聊天界面。然后可以从浏览器立即访问...</li><li><a href="https://dub.composio.dev/compete">竞争对手研究员</a>: 通过这个项目研究竞争对手的网站并跟踪他们所做的每一项更改。</li><li><a href="https://x.com/Prashant_Dixit0/status/1806580075447590974">Prashant Dixit (@Prashant_Dixit0) 的推文</a>: 使用 @llama_index 构建 Matryoshka RAG。这些 embedding 模型产生一系列 embedding 维度 (768, 512, 256, 128 和 64)。🌟 优势 ✅ 提升检索速度性能 ✅ 减少内存...</li><li><a href="https://github.com/shaunthecomputerscientist/EDA-GPT">GitHub - shaunthecomputerscientist/EDA-GPT: 利用 LLMs 进行自动化数据分析</a>: 利用 LLMs 进行自动化数据分析。通过在 GitHub 上创建账号为 shaunthecomputerscientist/EDA-GPT 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1256848855232938025)** (3 messages): 

- **Mixture of Agents (MoA) 大放异彩**：一段名为 ["Mixture of Agents (MoA) using langchain"](https://www.youtube.com/watch?v=VNy7CM23WA0) 的 YouTube 视频演示了如何使用 LangChain **实现 MoA**。该视频强调了结合多个 Agent 以发挥其各自优势的好处和所需步骤。
   - 该视频专注于实际应用并 *介绍了 MoA*，旨在通过利用集体能力来优化任务。演示者概述了核心代码片段和详细指南，以帮助更好地理解。
- **使用 LangChain 和 Google API 自动化网络搜索**：一篇 [Analytics Vidhya 博客文章](https://www.analyticsvidhya.com/blog/2024/06/automating-web-search-using-langchain-and-google-search-apis/) 描述了如何使用 LangChain 和 Google Search API **自动化网络搜索**。它解释了 AI 和 **NLP 创新** 如何通过总结搜索结果来简化相关信息的查找。
   - 文章详细阐述了使用 **LangChain 和 OpenAI** 来提高搜索效率，突破了传统搜索的局限。*用户可以获得针对其查询的简洁、总结性的回答* 以及相关链接。
- **详细介绍使用 LangChain 创建 PR Agent**：一份 [详细指南](https://git.new/pr-agent-langchain) 概述了使用 **Composio, LangChain, OpenAI 和 ChatGPT** 创建 PR Agent 的步骤。它建议确保安装了 Python 3.8+，并展示了 PR Agent 设置的 **示意图**。
   - 该指南强调了集成 **Slackbot** 进行 PR 审查，提供了插图和详细的实现步骤。从设置到执行，它涵盖了有效 PR 自动化所需的所有方面。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=VNy7CM23WA0">Mixture of Agents (MoA) using langchain</a>: 今天我们将使用 LangChain 实现 Mixture of Agents。我们介绍了 Mixture of Agents (MoA)，这是一种利用多个 L... 集体优势的方法。</li><li><a href="https://git.new/pr-agent-langchain">composio/python/examples/pr_agent/pr_agent_langchain at master · ComposioHQ/composio</a>: Composio 为 Agent 配备了精心设计的工具，使它们能够处理复杂的任务 - ComposioHQ/composio</li><li><a href="https://www.analyticsvidhya.com/blog/2024/06/automating-web-search-using-langchain-and-google-search-apis/">Automating Web Search Using LangChain and Google Search APIs</a>: 使用 LangChain 和 Google Search API 自动化网络研究，以实现高效的数据提取和 AI 驱动的查询回答。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1256371087906967595)** (40 messages🔥): 

- **Adept 策略转型及联合创始人加入 Amazon**：Adept 最近在其[博客](https://www.adept.ai/blog/adept-update)上宣布了策略转型和公司更新详情。这包括 Adept 的联合创始人加入 Amazon 并授权 Adept 的部分技术，正如 Adept 和 Amazon 内部沟通所确认的那样。
   - 根据一条 [推文](https://x.com/anissagardizy8/status/1806812006009442671?s=46)，在联合创始人离职前往 Amazon 后，这家初创公司仅剩约 20 名员工。此举模仿了 Microsoft 对 Inflection AI 的做法，引发了关于 Adept 未来和文化变革的讨论。
- **Transformer 作者离开 Adept**：两位曾是 Transformer 论文原作者的联合创始人离开了 Adept 并开始了自己的创业项目。这导致了多次领导层变动，包括公司目前已更换至第三任 CEO。
   - 有传言称 Adept 的毒性文化可能是他们离职的原因，一些人将其归咎于其中一位联合创始人。据报道，这种内部动荡是策略随着 Amazon 的介入而发生重大转变的部分原因。
- **AI Agent 开发中的挑战**：成员们将 AI Agent 的现状与早期的自动驾驶汽车炒作进行了比较，认为它们“似乎触手可及”但尚未能可靠运行。他们注意到像 Multion 这样的项目仅展示了基础的网页抓取能力。
   - 有推测认为数据收集是 AI Agent 进展的瓶颈，一些人认为更广泛或合成数据（synthetic data）对进步至关重要。重点正在从扩展现有数据转向生成专门针对模型需求的优质数据。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.adept.ai/blog/adept-update">An Update to Adept</a>：宣布我们的策略和公司的一些更新。</li><li><a href="https://x.com/anissagardizy8/status/1806812006009442671?s=46">Anissa Gardizy (@anissagardizy8) 的推文</a>：根据该初创公司的帖子和 Amazon 高管的内部邮件，Amazon 已聘请人工智能初创公司 Adept 的联合创始人并授权了其部分技术。Adept 目前仅剩...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1256724693654704148)** (6 messages): 

- **Cohere 改进 AI 的方法**：一段名为“[Cohere 今年将如何改进 AI 推理](https://youtu.be/B45s_qWYUt8?si=qs1u6p7wiXFP46PT)”的 YouTube 视频讨论了 **Cohere** 的 CEO **Aidan Gomez** 计划如何解决 **AI 幻觉 (hallucinations)** 并增强 **推理能力 (reasoning abilities)**。视频重点介绍了旨在识别模型弱点并缩小性能差距的合成数据生成技术。
   - 一位成员指出，**Generative Active Learning**（生成对标注/摄取最关键的示例）可能是这些技术的正确术语，并将其比作 LLM 的 *hard negative/positive mining*。在视频的 **5:30** 和 **15:00** 时间点有相关观察。
- **MIT 课程中提到的 Generative Active Learning**：在观看 MIT Data Centric AI 讲座时，提到了一个名为 **Generative Active Learning** 的概念。该子领域专注于生成用于标注和摄取的最重要示例。
   - 这一概念被讨论为可能类似于 LLM 的 *hard negative/positive mining*，这有助于识别和减轻模型弱点。该术语与正在进行的关于改进 AI 模型的讨论产生了共鸣。

**提到的链接**：<a href="https://youtu.be/B45s_qWYUt8?si=qs1u6p7wiXFP46PT">How Cohere will improve AI Reasoning this year</a>：Cohere 的 CEO Aidan Gomez 揭示了他们如何应对 AI 幻觉并提高推理能力。他还解释了为什么 Cohere 不使用任何外部...

  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1257050918826938498)** (23 messages🔥): 

- **Llama 405B 让 WhatsApp 用户感到惊喜**：一条 [推文](https://x.com/teknium1/status/1807490685387591983) 提到 **Llama 405B** 显然出现在了 WhatsApp 中，引起了用户的惊讶。这一发现引发了社区的询问和兴奋。
   - 成员们对这一进展表达了好奇和兴趣，想知道 **Llama** 何时会公开可用。
- **Llama 3 的发布仍不确定**：**Nathan Lambert** 对 **Llama 3** 的发布表示怀疑，认为考虑到现有模型的状况，可能没有必要发布它。**政治压力**也被提及为不确定性背后的原因。
   - 进一步的讨论显示，由于 **Meta 的战略利益**和 **ML 组织**的整体格局，对 **Llama 3** 的预期有所降低。Lambert 强调社区应相应调整预期。
- **Chatbot Arena 推动模型评估透明化**：**Chatbot Arena** 旨在通过人类偏好辅助 LLM 评估，强调 [推文](https://x.com/lmsysorg/status/1807503885181006236) 中概述的**数据透明度和社区参与**。该倡议鼓励研究真实世界的 Prompt，并利用这些数据来改进模型。
   - **Arena** 解决了关于 **Prompt 重复**和**数据分布**的担忧，通过 Kaggle 挑战赛和其他平台邀请社区参与，以解决模型评估中的人类偏好问题。
- **开源社区的希望寄托在 Deepseek 身上**：鉴于 Llama 3 发布的不确定性，社区开玩笑地表示哀叹，并将 **Deepseek** 视为开源模型的希望替代方案。这种情绪反映了社区对 **Deepseek** 满足其需求的更广泛依赖。
   - 一位成员认为，拥有像 **Deepseek** 这样的大型模型可以帮助计算资源有限的团队在其基础上进行构建，尽管 **Nathan Lambert** 对其可行性表示怀疑。
- **在开源社区训练大型模型是不现实的**：**Nathan Lambert** 认为，考虑到计算和数据的限制，期望开源社区在 400B 模型上进行训练是不现实的。尽管有讨论认为这对于拥有大量计算能力的团队具有潜在效用，但他仍持此观点。
   - Lambert 强调，**Meta** 发布此类模型会无意中帮助许多 ML 组织，这可能不符合其战略利益。对话强调了开源社区在与顶级专有模型保持竞争力方面面临的挑战。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/lmsysorg/status/1807503885181006236">来自 lmsys.org (@lmsysorg) 的推文</a>：似乎存在一些困惑。Chatbot Arena 自第一天起的目标就是帮助通过人类偏好解决 LLM 评估问题。通过开放我们的数据集/论文，我们鼓励社区研究真实世界的 Prompt...</li><li><a href="https://x.com/teknium1/status/1807490685387591983?s=46">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：显然 Llama 405B 出现在了 WhatsApp 中？🙃
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1256561795695509558)** (5 messages): 

- **Character.ai 与 Cognition AI 估值之谜**：拥有 2000 万用户并由 **Noam Shazeer** 创立的 **Character.ai** 估值为 10 亿美元，而没有用户且产品演示较弱的 **Cognition AI** 却坐拥 20 亿美元估值。讨论推测这是由于**融资策略**和市场推介的差异。
   - 评论指出，**Cognition AI** 描绘的愿景基于其创始成员是 **IMO 奖牌获得者**，并瞄准了利润丰厚的开发者市场。人们对其主张的稳健性表示担忧，因为面临来自**大厂**和成熟 AI 厂商的激烈竞争。
- **IMO 奖牌与估值**：成员 **an1lam** 幽默地指出 **Noam Shazeer** 也拥有一枚 **IMO 金牌**。鉴于对 **Cognition AI** 仅凭创始人数学荣誉获得高估值的批评，这增加了讽刺意味。
   - 讨论提到了来自 **OAI** 和 **Anthropic** 等大型 AI 公司的竞争，这些公司也有许多 IMO 奖牌获得者和更实质性的 AI 成就。人们对 **Cognition AI** 业务主张的可持续性表示担忧。
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1257110677668237352)** (7 条消息): 

- **AI 创新相关的 YouTube 视频**：成员分享了一个名为 [“Live from The Lakehouse, Day 2 Segment 3: AI Regulation and Industry Innovations”](https://youtu.be/cz6q-9sv-pU?si=XeckJ9U2V6XqZtzM&t=1290) 的 YouTube 视频。该视频讨论了 AI 监管和行业创新的最新趋势。
   - 另一个名为 [“Marines commercial (sword classic)”](https://www.youtube.com/watch?v=7C0kHFVHSDs) 的视频被提及，这是 80 年代经典的海军陆战队挥剑广告。一位用户幽默地指出，一个只有 8 次观看的未列出视频散发着一种“强者（alpha）气息”。
- **DBRex 法务团队“拔剑”事件**：成员们开玩笑说 DBRex 法务团队已经“拔剑相向”。讨论中没有进一步涉及该事件的背景或具体细节。
   - 一位用户表达了对 Frankle 的钦佩，称他“太酷了”。这是在之前关于视频和法务团队讨论的背景下提到的。
- **Chatbot Arena 上的 DBRX-Next 模型**：一位用户提到了 Chatbot Arena 上的 **DBRX-Next** 模型，但记不起更多细节。他们不确定是之前听说过然后忘了，这表明社区对此有一定兴趣但缺乏信息。
   - 另一位用户评论了修复其微调（fine-tuning）的努力，可能指的是同一个模型或另一个模型，这表明社区中存在持续的工作和挑战。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=7C0kHFVHSDs">Marines commercial (sword classic)</a>：80 年代经典的海军陆战队挥剑广告。</li><li><a href="https://youtu.be/cz6q-9sv-pU?si=XeckJ9U2V6XqZtzM&t=1290">Live from The Lakehouse, Day 2 Segment 3: AI Regulation and Industry Innovations</a>：加入我们，在 Live from The Lakehouse 第二天的这一环节中，就 AI 监管和行业创新的最新趋势进行深入讨论。</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1256493245848879226)** (4 条消息): 

- **以 LM 对齐为重点学习 RL**：一位用户快要看完 **David Silver 的 RL 入门**和 **Pieter Abeel 的深度 RL 基础讲座**，并计划接下来阅读 **Sutton & Barto**。他们询问关于学习 RL 的任何意外建议，特别是从 LM 对齐的角度。
   - 建议包括**阅读 Spinning Up in Deep RL** 以及利用**代码库来解决基础任务**。建议许多任务可以在 CPU 上执行，强调实践实现至关重要，且 **HF Deep RL 课程**中可能有用的示例。
- **Spinning Up in Deep RL 非常有用**：一位成员建议将 **Spinning Up in Deep RL** 作为一个非常有用的资源。另一位成员表示赞同，并补充说使用代码库解决基础任务是关键。
   - 建议在 **CPU** 上执行任务，并专注于**进行实际实现**以获得更好的学习效果。
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1256449307767476305)** (33 messages🔥): 

- **Ollama 在 01 配置上遇到困难**：一位成员分享了他们在遵循设置说明并使用 Ollama 的 `--local` 启动参数后，仍难以在 macOS 和 Windows 上运行 `01` 的经历。[一位用户建议](https://www.reddit.com/r/LocalLLaMA/comments/1dl3a13/killian_showed_a_fully_local_computercontrolling/)在终端中使用命令，但该成员面临需要 API key 的问题。
   - 其他用户在最近几个月也报告了类似问题，目前尚未找到广泛认可的解决方案。另一位成员提到桌面应用是一个不必要的步骤，但确切的修复方法仍然难以捉摸。
- **Open Interpreter 的长期记忆**：一位成员询问如何赋予 **Open Interpreter** 长期记忆，以避免在日常任务中重复犯错。他们指出，使用 **Sonnet** 模型会导致需要多次尝试才能纠正的错误。
   - 另一位成员建议使用特定命令或预训练方法，但未提供进一步的详细解决方案。**OI 记忆限制**带来的持续困扰显而易见。
- **向量搜索集成演示**：一位成员提议帮助进行大规模向量搜索的集成，并分享了最近在 lightning.ai 演讲中使用的 [Colab notebook](https://colab.research.google.com/github/onefact/loving-the-baseline/blob/main/nearest-neighbors.ipynb)。他们计划不久后在 Fed 演示类似的演讲。
   - 该成员对进一步的合作和分享他们在向量搜索方面的专业知识持开放态度，强调了其实际应用以及对社区的持续支持。
- **多模态 AI 模型推荐**：另一位成员征求用于构建多模态 AI 的开源模型建议，包括受限（censored）和不受限（uncensored）版本。**Moondream** 被推荐为视觉理解的一个极佳选择，可能可以与更大的 LLM 结合使用。
   - 这引发了关于不同模型在特定用例下的能力和可行性的简短讨论。然而，尚未就最佳模型达成共识。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://ai.google.dev/gemini-api/docs/code-execution?utm_source=gais&utm_medium=email&utm_campaign=june">未找到标题</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Q_p82HtBqoc&t">Open Interpreter's 01 Lite - WORLD'S FIRST Fully Open-Source Personal AI AGENT Device</a>：Open Interpreter 的 01 Lite 是一款 100% 开源的个人 AI 助手，可以控制你的电脑。让我们来评测一下，我将向你展示如何安装 open...</li><li><a href="https://colab.research.google.com/github/onefact/loving-the-baseline/blob/main/nearest-neighbors.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1dl3a13/killian_showed_a_fully_local_computercontrolling/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1256415466499342357)** (42 条消息🔥): 

- **OpenInterpreter 在 Windows 上遇到困难**：一名成员在 Windows 上为 OpenInterpreter 安装 **typer** 时遇到问题，并建议改进 pyproject.toml 文件。他们最终通过跳过某些依赖项并重试 poetry install 成功了。
   - 社区成员对 Windows 和 Ubuntu 上的**过时文档**和安装问题表示沮丧，强调需要紧急更新文档。Mikebirdtech 建议不要在频道中发送大量重复问题。
- **Windows 安装的 GitHub 资源**：分享了一个 [GitHub pull request](https://github.com/OpenInterpreter/01/pull/203) 的链接，其中包含更新后的 Windows 安装文档。该 pull request 旨在汇总之前安装尝试中的经验教训。
   - Shadowdoggie 称赞了 OpenInterpreter 与 macOS 的兼容性，但强调希望有更好的 **Windows 支持**。其他用户也表达了对清晰、更新的说明的需求，并对他们购买的 **01 Light** 系统的可行性表示担忧。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01: 开源语言模型计算机</a>：开源语言模型计算机。通过在 GitHub 上创建账户为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/01/pull/203">由 dheavy 更新 Windows 安装文档 · Pull Request #203 · OpenInterpreter/01</a>：问题：文档中未提供 Windows 安装及其关键差异。解决方案：汇总之前用户的尝试（包括 Discord 上的 Zorcon 和 ...</li><li><a href="https://01.openinterpreter.com/getting-started/setup">Setup - 01</a>：未找到描述
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1256353985959559218)** (7 条消息): 

- **Jina Reranker 获得升级**：LlamaIndex 用户对新的 **Jina reranker** 发布感到兴奋，它被誉为迄今为止最好的 reranker。有关此发布的更多信息可以在[这里](https://t.co/YsYoVOIirb)找到。
   - *一位用户*表示，这款 reranker 是改进检索策略和有效合并结果的游戏规则改变者。
- **超越朴素 RAG**：在 **AI World's Fair** 上，LlamaIndex 的 **Jerry Liu** 讨论了使用 Agent 进行比朴素 RAG 更深入的分析，并介绍了新的 llama-agents 框架。更多详情请访问[此链接](https://t.co/XWA6qnF7mn)。
   - *另一位用户*称赞了这次演讲，强调了其深度以及关于高级 Agent 使用的实用见解。
- **从零开始的混合检索教程**：**@kingzzm** 展示了一个关于创建自定义混合检索器（hybrid retriever）的详细教程，通过 reranker 结合不同的检索策略。教程请见[这里](https://t.co/cTxW2UwuZ0)。
   - *参与者*认为该教程内容详尽，对于构建有效的检索流水线（retrieval pipelines）非常有价值。
- **面向初学者的报告生成 Agent**：**@gswithai** 分享了一个入门指南，介绍如何使用一系列工具（如针对指南文档的 RAG）构建报告生成 Agent。查看教程请点击[这里](https://t.co/CJW6wefxLT)。
   - 该指南因其在设置用于报告生成的 ReAct Agent 方面的简单性和实用性而受到称赞。
- **将多 Agent 系统部署到 Kubernetes**：**@_nerdai_** 发布了一个新的入门套件，用于使用 Docker 和 Kubernetes 部署多 Agent 系统。更多信息可以在[这里](https://t.co/wfcI0wSmFG)找到。
   - 该套件旨在让用户从本地服务到 k8s 部署的转变变得无缝且高效。
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1256494605336318105)** (65 messages🔥🔥): 

- **Query Pipeline 故障排除**：一位用户在 **LlamaIndex** 的查询管道中使用非常规输入时遇到困难，特别是关于如何使用 **`source_key`** 在模块之间分离参数。另一位用户建议使用 kwargs 来正确路由输入。
   - 进一步的讨论包括在管道中集成检索以及设置和管理检索的适当方法。**Source_key** 被再次确认是正确的，但需要仔细实现。
- **Embedding 性能问题**：由于嵌入行的性能缓慢（约 2048 行/50 秒），一位用户在嵌入大型 CSV 文件时感到吃力。另一位用户建议将 **`embed_batch_size`** 增加到 500 以提高性能。
   - 尽管性能有所提高，但使用像 Phi3 这样的紧凑型 LLM 在生成可评估代码时存在问题。用户考虑将 **Ollama 上的量化 Llama3** 作为潜在的解决方案。
- **查询中的元数据处理**：一位用户在 **LlamaIndex** 中通过元数据值检索节点时遇到问题，因为检索器将不相关的节点排在更高位置。建议的解决方案是引入 Reranker 以获得更好的准确性。
   - 另一项讨论包括基于元数据字符串包含关系过滤节点的可行性，以及元数据在增强 LlamaIndex 检索和响应准确性方面的重要性。
- **Agent 服务中的状态与规模管理**：提出了关于管理 **LlamaIndex Agent 服务** 状态的问题，特别是针对具有多个服务副本的场景。讨论提到 **AgentService** 的状态管理目前是在内存中的，但正在向使用键值存储的无状态方法演进。
   - 讨论了运行具有共享状态的多个 AgentService 实例的可能性，并正在考虑实现用于水平扩展的无状态服务的解决方案。此外还提到了 Sidecar 模式作为一种前瞻性方法。
- **教程与子 Agent 功能**：用户寻求在 **LlamaIndex** 中实现 **sub-agents** 的教程或示例，旨在为特定任务分配预定义的 Prompt 和输入。讨论了子 Agent 在增强特定任务功能方面的作用。
   - 此外，还提出了一个关于如何有效使用和自定义 **CodeSplitter** 等工具，以更高效的方式简化元数据提取和节点处理任务的问题。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/@llamaindex?si=1hMPLukiNN8dMJzx">LlamaIndex</a>: LlamaIndex 官方 YouTube 频道 - 适用于你 LLM 应用的数据框架 </li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline/">LlamaIndex 查询管道简介 - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">入门教程（本地模型） - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/">查询管道聊天引擎 - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/">提取元数据以实现更好的文档索引和理解 - LlamaIndex</a>: 暂无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC">提取元数据以实现更好的文档索引和理解 - LlamaIndex</a>: 暂无描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1257039529098088512)** (1 messages): 

- **LlamaIndex 可以使用 Langchain 工具吗？**：一位用户询问 **Langchain Tools** 是否可以与 **LlamaIndex agents** 一起使用。他们强调反向使用 Langchain 工具（在 Langchain 中使用 LlamaIndex 工具）已经是已知的。
   - 社区正在寻求关于将这些工具与 LlamaIndex agents 集成的进一步确认和讨论。
- **关于 Langchain 和 LlamaIndex 的集成咨询**：成员们讨论了将 **Langchain Tools** 与 **LlamaIndex agents** 结合使用的潜在方法。用户提到已知晓反向集成的操作。
   - 这引发了关于这两个工具集之间兼容性和集成过程的疑问。
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1256551595718610995)** (54 条消息🔥): 

- **仅凭 27b 参数就达到 llama3 70b 的水平令人惊叹**：成员们对仅用 270 亿参数就实现的 **llama3 70b** 模型性能表示惊讶，并讨论了其可行性。
   - 一位成员提到他们仍在使用 **Mixtral**，因为它在性能和准确性之间取得了平衡，尤其是在消费级显卡上，并称赞了其多语言能力和许可协议。
- **Hermes 2 Theta vs Hermes 2 Pro**：围绕 Hugging Face 上的 **Hermes 2 Theta** 和 **Hermes 2 Pro** 模型展开了讨论，其中一个是实验性的 merge，另一个是纯净的 finetune。
   - 成员们辩论了哪个选项更好，提到了不同的训练数据集和功能，例如 **Pro** 版本中的 JSON Structured Outputs。
- **Axolotl 中的自定义 ORPO 格式化器**：用户讨论了 **Axolotl** 中自定义 ORPO 格式化器的问题，例如错误的 tokenization 以及 ChatML 中 system roles 的处理。
   - 建议包括使用 custom roles 和 input roles 来克服限制，但成员们对潜在的冲突表示担忧。
- **Nvidia 合成模型性能**：一些成员尝试了 **Nvidia synthetic model**，注意到其数据生成速度较慢——与 llama 70b 或 GPT-4 等模型相比耗时更长。
   - 该模型的性能引发了关于更小、更高效版本在实际用途中优势的疑问。
- **CAME 和 Adam-mini 优化器**：社区探索了新的内存高效优化器，如 **CAME** 和 **Adam-mini**，它们声称在减少内存使用的同时，性能优于或等同于传统方法。
   - 分享了关于 [CAME 论文](https://arxiv.org/abs/2307.02047) 和 [Adam-mini 论文](https://arxiv.org/abs/2406.16793) 的链接，供对技术细节和在 Stable Diffusion 训练等任务中潜在用途感兴趣的人参考。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B">NousResearch/Hermes-2-Pro-Llama-3-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B">NousResearch/Hermes-2-Theta-Llama-3-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2307.02047">CAME: Confidence-guided Adaptive Memory Efficient Optimization</a>: 自适应梯度方法（如 Adam 和 LAMB）在大语言模型训练中表现出色。然而，自适应性需要维持二阶矩...</li><li><a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: 我们提出了 Adam-mini，这是一种优化器，它能以减少 45% 到 50% 的内存占用实现与 AdamW 相当或更好的性能。Adam-mini 通过减少学习率资源来降低内存...
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1256551947344023653)** (11 条消息🔥): 

- **训练 Prompt 格式保持一致**：一位成员询问训练期间是否使用了相同的 Prompt 格式。另一位成员确认确实如此。
   - 尽管如此，关于与 Prompt 重复惩罚或训练/测试 Prompt 无关的其他潜在问题的讨论仍在继续。
- **解决 FT phi-3-small 加载错误**：一位成员在加载 8k 和 128k 版本的 phi-3-small 时遇到困难，遇到了与 warmup_steps 和 warmup_ratio 相关的验证错误。另一位成员澄清说，这些参数是互斥的，只能使用其中一个。
   - 在重新检查配置后，该成员承认了错误并找到了隐藏的 ratio 参数，这要归功于有用的反馈。
- **Tiktoken 导入错误困扰离线设置**：一位在离线机器上工作的成员遇到了 tiktoken 导入错误：由于网络不可用导致 *requests.exceptions.ConnectionError*。该错误是由于尝试连接 [openaipublic.blob.core.windows.net](https://openaipublic.blob.core.windows.net) 失败引起的。
   - *ConnectionError* 是由于在没有可用网络连接的情况下尝试访问 tiktoken 编码资源而发生的。

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1256323358560096337)** (9 条消息🔥): 

- **排行榜基准测试受限于计算资源瓶颈**：成员们讨论了当前排行榜基准测试漫长的排队时间，感觉队列已经积压了 2 天。**Stellaathena** 提到 HF 上的队列完全超出了他们的控制，很可能受限于计算资源瓶颈。
   - “HF 上的队列完全超出了我们的控制，但推测是受限于计算资源瓶颈，”一位成员强调道。这表明 HF 在及时处理基准测试请求的能力上存在持久性问题。
- **im-eval 与 HumanEval 和 HumanEvalPlus 的兼容性**：**Johnl5945** 询问 im-eval 是否可以执行 `HumanEval` 和 `HumanEvalPlus`，并询问如何设置评估温度（temperature）。这一询问突出了 im-eval 的潜在使用场景。
   - 成员们没有提供直接回答，使得关于 im-eval 功能和温度设置配置的查询悬而未决，暗示需要进一步探索。
- **HF 队列问题及 vllm 等替代方案**：成员们注意到了 HF 队列延迟的问题，并建议探索替代方案，例如 **vllm**。分享了一个**有用的 wiki**：[GitHub Wiki](https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm) 以指导模型保存。
   - “哦，是的，那是 HF 的已知问题——经常发生。你试过用 vllm 代替吗？”一位热心的成员建议道，并链接到了 [GitHub Wiki](https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm) 以获取进一步指导。

**提到的链接**：<a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 使用 unslothai/unsloth 以 80% 更少的显存将 Llama 3, Mistral, Phi &amp; Gemma LLMs 的微调速度提升 2-5 倍

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1256325117173170377)** (46 条消息🔥): 

- **$\Delta$-IRIS Agent 创下新基准**：研究人员提出了 [$\Delta$-IRIS](https://arxiv.org/abs/2406.19320)，这是一种带有离散自编码器和自回归 Transformer 的新型 RL Agent，在 Crafter 基准测试中创下了新的 SOTA。与之前的方法相比，该 Agent 的训练速度显著加快。
   - 社区成员对性能提升表示赞赏，并指出该 Agent 可以通过连续 Token 高效地编码时间步之间的随机增量（stochastic deltas）。
- **RoPE-before-QK 定位问题得到解决**：RoPE-before-QK 的问题（即 QK 混合了跨 RoPE 通道的信息，从而破坏了位置嵌入）已通过绑定每个 Head 通道的线性变换得到解决。该方法有效地保留了 **RoPE 的相对交互**。
   - 讨论还强调，相位偏移（phase offsets）可以是可学习的且依赖于数据的，以进行进一步改进，这可以增强 Transformer 中的位置嵌入。
- **Adam-mini 优化器将内存占用减少一半**：[Adam-mini 优化器](http://arxiv.org/abs/2406.16793) 通过划分参数并为块分配单一学习率，以减少 45% 到 50% 的内存占用，实现了与 AdamW 相当或更好的性能。这种减少是在没有显著牺牲模型性能的情况下实现的。
   - 社区讨论承认，划分和分块优化使其成为当前流行优化器的有力替代方案，且没有过多的内存开销。
- **Flora 挑战低秩适配器局限性**：正如这篇 [论文](https://arxiv.org/abs/2402.03293) 所介绍的，Flora 提出了一种通过随机投影实现高秩更新的方法，减少了优化状态。它保留了 LoRA 减少内存占用的优势，但改善了其性能局限。
   - 社区成员讨论了其理论价值，认为它可能面临与 LoRA 类似的局限性，但投影的动态重采样可能会带来改进。
- **Token 表示中的擦除效应被发现**：一项关于 LLM 中 Token 表示的新研究揭示了一种“擦除效应（erasure effect）”，即在包含命名实体和多 Token 词的表示中，关于前一个 Token 的信息会减弱或被“擦除”。在多 Token 概念的最后一个 Token 表示中，Token 擦除效应非常强烈。
   - 该 [研究](https://arxiv.org/abs/2406.20086) 因揭示了 LLM 如何处理语义无关的 Token 组而受到赞赏，并为改进模型对复杂 Token 序列的理解提供了见解。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>：我们提出了 Adam-mini，这是一种优化器，其性能与 AdamW 相当或更好，但内存占用减少了 45% 到 50%。Adam-mini 通过削减...中的学习率资源来减少内存。</li><li><a href="https://arxiv.org/abs/2406.19320">Efficient World Models with Context-Aware Tokenization</a>：扩展深度强化学习 (RL) 方法面临着重大挑战。随着生成式建模的发展，基于模型的 RL 成为强有力的竞争者。最近的进展...</li><li><a href="https://arxiv.org/abs/2406.20086">Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs</a>：LLM 将文本处理为大致对应于单词的 Token 序列，其中较不常见的单词由多个 Token 表示。然而，单个 Token 通常在语义上与...无关。</li><li><a href="https://arxiv.org/abs/2402.03293">Flora: Low-Rank Adapters Are Secretly Gradient Compressors</a>：尽管大型神经网络在完成不同任务方面表现出卓越的能力，但它们需要过多的内存来存储训练的优化状态。为了缓解这一问题，...</li><li><a href="https://arxiv.org/abs/2404.14507">Align Your Steps: Optimizing Sampling Schedules in Diffusion Models</a>：扩散模型 (DM) 已确立其作为视觉领域及其他领域最先进生成建模方法的地位。DM 的一个关键缺点是采样速度慢，依赖于...</li><li><a href="https://github.com/BorealisAI/flora-opt">GitHub - BorealisAI/flora-opt: This is the official repository for the paper &quot;Flora: Low-Rank Adapters Are Secretly Gradient Compressors&quot; in ICML 2024.</a>：这是 ICML 2024 论文 "Flora: Low-Rank Adapters Are Secretly Gradient Compressors" 的官方仓库。 - BorealisAI/flora-opt</li><li><a href="https://arxiv.org/abs/2311.00537">Machine Learning Without a Processor: Emergent Learning in a Nonlinear Electronic Metamaterial</a>：标准的深度学习算法需要对大型非线性网络进行微分，这一过程缓慢且耗能。电子学习超材料提供了潜在的快速、高效且...</li><li><a href="https://colab.research.google.com/drive/13WO33fQzhnSV4daghFXUmVCwVJ0OV5MI?usp=sharing">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1256326613503578212)** (7 条消息): 

- **自定义配置 YAML 错误：格式问题**：**新用户**：一名成员询问关于使用自定义配置 YAML 运行评估的问题，尽管创建了包含 `dataset_name`、`description` 等参数的新 YAML，但仍面临错误。包含参数 `--verbosity DEBUG` 有助于识别错误所在，并指出命名约定必须唯一有助于解决问题。
   - 添加适当的标志后发现，由于存在同名的已注册任务，导致 `task named 'mmlu_clinical_knowledge'` 冲突。最终，运行适当的调试命令成功解决了该问题。
- **Gemma 2 指标故障排除：注意到差异**：**复现问题**：一位用户报告称，尽管遵循了将 `dtype` 设置为 `bfloat16` 并使用 `4.42.3` 版本的 transformers 库等设置说明，但在使用 `lm_eval` 复现 **Gemma 2** 指标时仍存在显著差异。观察到的指标差异包括 **piqa** 上的准确率为 0.5990，而模型卡上为 0.817，**hellaswag** 和 **winogrande** 也有类似差异。
   - 使用的命令（`lm_eval --model hf --model_args pretrained=google/gemma-2-9b,parallelize=True,trust_remote_code=True,dtype=bfloat16 --tasks piqa --batch_size auto:4 --log_samples --output_path output/gemma-2-9b`）看起来格式正确，因此需要进一步调查潜在的未知问题。

**提到的链接**：<a href="https://x.com/LysandreJik/status/1807779464849273343)">来自 Lysandre (@LysandreJik) 的推文</a>：上周，Gemma 2 发布了。从那时起，实现已根据模型性能进行了调整：`pip install -U transformers==4.42.3`。我们看到有报告称工具（transformers, llama.cpp）未能...

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 条消息): 

arctodus_: 谢谢！这正是我在寻找的。我会看看。
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1256362059277533256)** (27 条消息🔥): 

- **将 RMSNorm 移动到 nn/init.py**: George Hotz 询问是否有人能以*优雅的方式*将 `RMSNorm` 从 **LLaMA** 移动到 `nn/__init__.py`，并添加文档和测试。
   - 这将有助于规范化并改进 **tinygrad** 代码库的组织结构。
- **Tinygrad 周一会议议题**: 周一会议涵盖了多个主题，例如新的 tinybox 更新、[更快的 sharded llama](https://github.com/tinygrad/tinygrad/pull/5123) 以及 [更整洁的 graph_rewrite](https://github.com/tinygrad/tinygrad/pull/5159/files#diff-91ca5e2e75ef3ea1982c8ca6cc175ee88f20efa0d8e4b96f305b970dc6df71e7R291)。
   - 讨论内容包括 **lowerer continuation**、**tensor cores** 以及各种悬赏任务（bounties），如 standard mean one kernel 和 **qualcomm runtime**。
- **为 Raspberry Pi 编译 Tinygrad 程序**: 一位用户询问 **tinygrad** 程序是否可以编译为适用于 Raspberry Pi 等设备的独立 C 程序。
   - 另一位成员分享了 [此链接](https://github.com/wozeparrot/tinygrad-on-esp32) 关于 **tinygrad on ESP32** 的内容，表明了潜在的兴趣和用例。
- **Llama 70B LoRA 悬赏要求**: 讨论了关于 **llama 70b lora bounty** 的要求，以及是否接受 **qlora**。
   - **ChenYuy** 解释说它必须遵循 [MLPerf 参考实现](https://github.com/mlcommons/training/tree/master/llama2_70b_lora)，但可以在具有 offloading 功能的不同机器或多个 tinybox 上完成。
- **Graph Rewrite 后续讨论**: 一位成员错过了之前的讨论，询问关于 **graph rewrite followup** 的情况，特别是 Egraphs/muGraphs。
   - **ChenYuy** 转述称，目前有意将更多算法（如 scheduler）移入 graph rewrite，尽管具体的图算法尚未准备好被优先处理。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/5159/files#diff-91ca5e2e75ef3ea1982c8ca6cc175ee88f20efa0d8e4b96f305b970dc6df71e7R291),">single pass rewrite by geohot · Pull Request #5159 · tinygrad/tinygrad</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/issues/5244)">Issues · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️  - Issues · tinygrad/tinygrad
</li>
</ul>

</div>

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1256418416206545009)** (34 messages🔥): 

- **tinygrad 计算与梯度更新中的挑战**：一位成员在 tinygrad 中定义中间张量时遇到问题，如果先定义中间张量 (`emb = C[X]`)，**C** 会从计算图中切断。他们发现将 `emb = C[X]` 下移到 logits 之前，可以保持 `C` 与 loss 之间的链接，从而使梯度计算正常工作。
   - 另一个问题出现在遍历 `parameters` 进行更新步骤时，原始张量在下一个 epoch 会回退。解决方案是使用 `p.assign(p.realize().detach() - lr * p.grad)` 以确保张量被正确更新。
- **对 tinygrad 中 text2video 功能的兴趣**：一位成员询问是否可以在 tinygrad 中添加类似 **SORA** 的 text2video 功能，并寻找可以贡献的缺失部分。目前没有直接回应表明有此计划。
   - 这次对话显示了社区对扩展 tinygrad 能力的兴趣以及用户的潜在贡献意愿。
- **跨多台机器运行 tinygrad**：George Hotz 分享了在 **AGIHouse Hackathon** 期间在 4 台 mac mini 上运行 tinygrad 的兴趣，并链接到了 [wozeparrot/tinygrad](https://github.com/wozeparrot/tinygrad/tree/ops_remote)。
   - 进一步的讨论包括在几台 **mac studio M2 ultra** 设备上运行 **nemotron-340b inference** 的可行性咨询。
- **TinyJit 装饰器导致训练异常**：一位成员报告称，在 `train_step` 上添加 `TinyJit` 装饰器会导致 loss 值迅速降至零。**Chenyuy** 建议该问题可能是由于重复使用同一个 mini-batch 造成的，指出在步骤中需要不同的训练样本。
   - 其他人参与了关于 JIT 在梯度内存重置和梯度清零方面行为的技术讨论。
- **调试 tinygrad 问题与 PR 建议**：__gip__ 在使用 `DEBUG >= 6` 时遇到问题，提到 tinygrad 的调试输出中缺少 `applegpu` 工具。他们提交了一个 [PR](https://github.com/tinygrad/tinygrad/pull/5236) 以改进这些工具缺失时的错误提示。
   - __gip__ 在调试时还遇到了另一个 Apple GPU 反汇编问题，并在 [dougallj/applegpu GitHub](https://github.com/dougallj/applegpu/issues/61) 上进行了报告。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/wozeparrot/tinygrad/tree/ops_remote">GitHub - wozeparrot/tinygrad at ops_remote</a>：你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - GitHub - wozeparrot/tinygrad at ops_remote</li><li><a href="https://github.com/dougallj/applegpu.git">GitHub - dougallj/applegpu: Apple G13 GPU architecture docs and tools</a>：Apple G13 GPU 架构文档和工具。欢迎在 GitHub 上为 dougallj/applegpu 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/blob/b2ea610df830f8b2b25dd06ce67c2304b0f9d08a/examples/llm.c/train_gpt2.py#L168">tinygrad/examples/llm.c/train_gpt2.py at b2ea610df830f8b2b25dd06ce67c2304b0f9d08a · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你爱 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="http://www.catb.org/~esr/faqs/smart-questions.html">How To Ask Questions The Smart Way</a>：提问的智慧</li><li><a href="https://github.com/tinygrad/tinygrad/pull/5236">fix: message when applegpu tools missing by gip · Pull Request #5236 · tinygrad/tinygrad</a>：当 applegpu 工具缺失时显示错误信息。很乐意在另一个 PR 中增加文档章节介绍如何安装 applegpu 工具。虽然如果我们这样做，同时也增加如何安装...</li><li><a href="https://github.com/dougallj/applegpu/issues/61">AssertionError during disassembly · Issue #61 · dougallj/applegpu</a>：在使用 tinygrad 调试模式时，我在 Apple GPU 反汇编过程中遇到了这个崩溃。复现方法：python3 compiler_explorer.py tmp_p36do0r 在 https://github.com/dougallj/app... 触发断言错误。
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1256324486643581110)** (25 条消息🔥): 

- **对 27B 模型质量的怀疑**：**27B 模型**的有效性受到质疑，但即使在最坏的情况下，也被认为优于 **Command R+**。尽管有置信区间，人们对其真实能力仍持怀疑态度。
   - 提到 **Gemma 2 9B** 表现更好增加了辩论的激烈程度，但总体情绪倾向于 27B 的潜力。*“确实，但即使在最坏的情况下（-15），它也比 Command R+ 更好”*。
- **ChatGPT 4 和 4o 模型陷入困境**：报告显示 **ChatGPT 4 和 4o 模型** 的性能正在下降，特别是在编程任务中。用户注意到 **ChatGPT 3.5** 处理 Prompt 的效果更好，不会过于死板地理解字面意思。
   - *“有时付费的 4 和 4o 模型在编程时感觉完全没用”*，强调了由于付费版本效率下降，用户更倾向于选择**免费替代方案**。
- **Gemini 1.5 Pro 胜过 ChatGPT**：**Gemini 1.5 Pro** 因表现优异而受到称赞，相比之下，**ChatGPT** 被认为在遵循 Prompt 方面越来越懒惰。关于 GPT4 在编程中表现不佳的投诉正在增加。
   - *“与 ChatGPT 日益严重的懒惰相比，Gemini 1.5 Pro 的表现非常出色”*。用户正转向像 Gemini 这样响应更积极的模型。
- **Claude 的 Artifacts 功能赢得用户**：用户表达了对 **Claude 的 Artifacts 功能** 的偏好，认为其体验更好。由于这一功能，一些人考虑完全从 **ChatGPT** 切换过来。
   - *“Artifacts 功能的体验要好得多”* 总结了促使用户转向 Claude 等替代方案的情绪。
- **非英语母语者寻求多语言 LLM**：讨论表明，**非英语母语者**正转向 LLM Arena 寻找在母语方面表现出色的模型。尽管模型在特定任务中的有效性各异，但这推动了它们的普及。
   - *“它之所以能排在前面，不是因为它能回答难题，而是因为它具备多语言能力”*。使用趋势显示出对特定语言对话能力的偏好。

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1256526950604738681)** (31 messages🔥): 

- **Block-wise Adam 优化器的效率**：一场讨论强调了 Block-wise Adam 优化器相比逐参数 $v$ 状态可节省 **90% VRAM**，**总体显存减少 45%**。据称它不仅在节省 VRAM 方面表现出色，在性能上，特别是在阶梯式损失曲线中也更具优势。
   - 多位成员对该优化器的性能和节省效果表示惊讶。一位用户提到，该优化器不需要在内存中保留三组参数，并思考其与 **8bit 优化**的兼容性。
- **面向多样化场景的角色驱动数据合成**：讨论了一种针对 LLM 的新型 [角色驱动数据合成 (persona-driven data synthesis)](https://arxiv.org/html/2406.20094v1) 方法论。该方法可以利用来自网络数据的 **10 亿个多样化角色 (personas)** 来创建合成数据，可能引发数据创建的范式转移。
   - 讨论强调了该方法在创建多样化合成数据（包括**数学问题**和**游戏 NPC**）方面的通用性和可扩展性。这种方法利用了模型中封装的各种视角，显著增强了合成数据的生成。
- **全量微调模型评估的挑战**：分享了一篇关于 [全量微调模型评估 (full-finetuned model evaluation)](https://mlops.systems/posts/2024-07-01-full-finetuned-model-evaluation.html) 的博客文章，强调了从微调模型中提取结构化数据的复杂性和挑战。**准确率 (Accuracy)** 被强调为核心关注指标。
   - 发布者在没有专用系统的情况下遇到了显著的实现困难和性能问题，并指出管理这些评估的复杂性正在增加。这反映了在有效维护和扩展微调模型方面面临的更广泛挑战。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.16008">Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization</a>：LLM 即使经过专门的长文本处理训练，也很难捕捉位于输入中间的相关信息。这种现象被称为...</li><li><a href="https://arxiv.org/html/2406.20094v1">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>：未找到描述</li><li><a href="https://github.com/zyushun/Adam-mini/blob/main/Adam_mini.py">Adam-mini/Adam_mini.py at main · zyushun/Adam-mini</a>：Adam-mini 的代码。通过在 GitHub 上创建账号为 Adam-mini 的开发做出贡献。</li><li><a href="https://mlops.systems/posts/2024-07-01-full-finetuned-model-evaluation.html">Alex Strick van Linschoten - My finetuned models beat OpenAI’s GPT-4</a>：对于我的测试数据，Mistral、Llama3 和 Solar LLM 的微调版本比 OpenAI 的模型更准确。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1256332407850668032)** (16 条消息🔥): 

- **简单的 JSONL 编辑工具**：一位用户正在寻找一种简单快速的 JSONL 文件编辑工具。[提供的一个示例](https://github.com/aastroza/simple-jsonl-editor)包含一个用于此目的的简单 GUI。
   - 另一位用户分享了他们用于 JSONL 编辑的个人脚本方法，表示他们维护着一个用于编辑 JSONL 文件的 Python 脚本文件夹。
- **结构化数据摘要**：一位用户正在探索从 JSON 格式的结构化患者数据中生成摘要的方法。他们提到正在测试 Llama 模型，并对幻觉（hallucinations）感到担忧，同时考虑了人工和 LLM 评估方法。
   - [建议包括](https://github.com/aastroza/simple-jsonl-editor)提示词工程（prompt engineering）和少样本学习（few-shot learning），并建议最初先对基准方法进行故障排除，而不是直接实现新的软件包。
- **LLM 推理的 KV-Caching**：一位用户分享了一篇关于实现 KV-Caching 以增强 LLM 推理的博文，特别是针对像 Phi-3-vision 这样的 Vision-LLM。[这种技术](https://sachinruk.github.io/blog/2024-06-29-kv-cache.html)通过存储常用提示词并使用词汇表 Logits 来提高预测概率。
   - 该博文因其在优化视觉模型方面的指导而受到另一位用户的赞赏，强调了为那些没有充足 GPU 资源的人提供的实用步骤。
- **语音聊天产品资源**：关于开发语音聊天产品的讨论建议结合使用 STT (Whisper)、LLM 处理和 TTS 模型。提到了 GPT-4o 语音对这一流水线（pipeline）的潜在影响。
   - 用户推荐了特定的工具，如用于语音检测的 SileroVAD，以及用于快速测试语音助手原型的 [Chainlit cookbook](https://github.com/Chainlit/cookbook/tree/main/audio-assistant)。
- **知识图谱与 Lang Graph 专业知识**：一位资深的 Python AI 全栈开发人员分享了他们在 AI 驱动软件方面的丰富经验，包括在电子商务、图像处理和气候分析项目中的深入工作。他们强调了通过使用高级 RAG 和定制 LLM 显著提高了准确率。
   - 进一步的询问涉及该开发人员在知识图谱（knowledge graphs）和 Lang Graph 方面的工作，表明了对深入理解和实际案例的需求。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/515379">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>：未找到描述</li><li><a href="https://sachinruk.github.io/blog/2024-06-29-kv-cache.html">Prompt Caching: Poor man’s guide to zero shot vision-LLM classification – deepschool.ai</a>：使用 KV-Caching 和 Logit 比率来加速并控制 LLM/ VLM 输出。</li><li><a href="https://github.com/Chainlit/cookbook/tree/main/audio-assistant">cookbook/audio-assistant at main · Chainlit/cookbook</a>：Chainlit 的 cookbook 仓库。通过在 GitHub 上创建账号来为 Chainlit/cookbook 的开发做出贡献。</li><li><a href="https://github.com/aastroza/simple-jsonl-editor">GitHub - aastroza/simple-jsonl-editor: Simple GUI for editing .jsonl files</a>：用于编辑 .jsonl 文件的简单 GUI。通过在 GitHub 上创建账号来为 aastroza/simple-jsonl-editor 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1256961192065499136)** (4 messages): 

- **Kubernetes 在 ML 推理方面面临质疑**：一位成员注意到一条推文在嘲讽为 ML 在 **Kubernetes (k8s)** 上设置推理，这引发了对该立场背后原因的好奇。他们询问了关于此话题的见解，强调了经验丰富的从业者在管理高流量生产系统方面的讨论重要性。
   - 作为回应，推文作者澄清说该评论*主要是为了好玩*，并表示相信 **Modal** 是一个更优的 ML 推理云基础设施。
- **在分布式 Modal 设置上运行 axolotl 遇到困难**：一位成员分享了他们在 **Modal** 的分布式设置上运行 **axolotl** 时遇到的挑战，并指向了 [GitHub](https://github.com/modal-labs/llm-finetuning) 上的一个克隆仓库。他们提到在单张 **A100** GPU 上运行成功，但在尝试横向扩展时遇到了错误。
   - 报告的错误包括与 **NCCL** 通信相关的 *Distributed Training Error* 和 *Socket Timeout*，这表明可能存在网络连接或设置问题。此外，多个 rank 无法通信，导致进程失败。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/charles_irl/status/1807265742506639739">Charles 🎉 Frye (@charles_irl) 的推文</a>：当有人说他们正在 k8s 上设置推理时我的表情</li><li><a href="https://github.com/modal-labs/llm-finetuning">GitHub - modal-labs/llm-finetuning: Llama/Mistral/CodeLlama 等模型的微调指南</a>：微调 Llama/Mistral/CodeLlama 等模型的指南 - modal-labs/llm-finetuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1256638752680575110)** (7 messages): 

- **Hugging Face 积分有效期为 2 年**：一位成员询问积分是否适用于 **Hugging Face** 上的任何用途，另一位成员确认确实如此。他们澄清说积分**有效期为 2 年**，缓解了对 2 天过期的担忧。
   - 多位成员在得知积分有效期为 2 年后感到宽慰，其中一位表达了感谢，另一位提到他们也剩有很多积分。另一位成员寻求关于其积分状态的额外信息，但未得到回复，凸显了沟通问题。
- **成员管理未使用的积分**：成员们讨论了使用剩余积分进行 **LLM** 和 embedding 推理，担心积分即将过期。关于积分 2 年有效期的澄清缓解了他们的担忧。
   - 还有一些关于积分状态的信息请求未得到答复。这表明需要更好的关于积分管理的沟通。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1256662452519567360)** (1 messages): 

- **在 Replicate 上获取 Mistral 7B Instruct v3**：[一场讨论](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)开始于如何使 **Mistral 7B Instruct v3** 在 **Replicate** 上可用，并附带了 **Mistral-7B-Instruct-v0.3** 的模型卡链接。说明建议使用 `mistral_common` 进行分词（tokenization）。
   - 目前，[Replicate](https://replicate.com/mistralai) 上只有 **v1** 和 **v2** 版本。一位用户询问他们是否可以从个人账户发布 **v3** 作为“社区”贡献。
- **Mistral-7B Instruct 版本差异说明**：**Mistral-7B-Instruct-v0.3** 大语言模型与 [v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/edit/main/README.md) 相比有显著更新，包括扩展的词汇表、支持 v3 tokenizer 以及 **function calling**。用户咨询澄清了这些差异如何影响在 Replicate 等平台上的可用性。
   - 项目文档建议使用 **mistralai/Mistral-7B-Instruct-v0.3** 配合 [mistral-inference](https://github.com/mistralai/mistral-inference) 以获得最佳效果。提供了通过 `pip install mistral_inference` 进行安装的进一步细节。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>：暂无描述</li><li><a href="https://replicate.com/mistralai">mistralai – Replicate</a>：暂无描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[ankurgoyal_textsql_llmevals](https://discord.com/channels/1238365980128706560/1242222674835538012/1256404374700425216)** (9 条消息🔥): 

- **Braintrust 测量输出的工具**：一位用户询问了用于测量工具调用输出以及复杂 Agentic 工具调用工作流的工具。他们被引导至 Braintrust [cookbook recipes](https://www.braintrust.dev/docs/cookbook/recipes/Assertions) 以获取这些工具的详细信息。
   - 用户对这些信息表示感谢，并提到在 Braintrust 中通过 `Eval()` 调试和集成数据、任务以及评分部分的便利性。他们表达了对该工具 UX 的喜爱和对其帮助的认可。
- **通过 Logs 标签页实现实时可观测性**：用户询问了实时可观测性以及是否使用 'Logs' 标签页来实现此目的，指出需要文档或 cookbook 参考。他们被推荐参考 [logs 通用文档](https://www.braintrust.dev/docs/guides/logging) 以及一个关于[结合使用 logs 和 evals 的特定 cookbook](https://www.braintrust.dev/docs/cookbook/recipes/HTMLGenerator)。
   - 文中澄清，为 evaluations 设置的 tracing 也适用于 logs，并且 Braintrust UI 中的 Logs 标签页会自动实时更新。文档还涵盖了记录交互、调试客户问题以及捕获用户反馈的内容。
- **人工审核功能**：有人简要询问了 Braintrust 的 'human review' 功能，以及它是否适用于 logs、experiments 或两者。经确认，experiments 和 logs 之间的 'Trace' 数据结构是相同的，这使得人工审核功能在各处都适用。
   - 还提到了数据集中的人工审核新功能，能够集成来自不同来源的人工反馈。更多详细信息请参阅 Braintrust 的 [人工审核指南](https://www.braintrust.dev/docs/guides/human-review#writing-categorical-scores-to-expected-field)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.braintrust.dev/docs/guides/human-review#writing-categorical-scores-to-expected-field">人工审核</a>: 未找到描述</li><li><a href="https://www.braintrust.dev/docs/cookbook/recipes/Assertions">Zapier 如何使用断言评估聊天机器人中的工具使用情况</a>: 未找到描述</li><li><a href="https://www.braintrust.dev/docs/guides/logging">日志记录</a>: 未找到描述</li><li><a href="https://www.braintrust.dev/docs/cookbook/recipes/HTMLGenerator">生成美观的 HTML 组件</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[paige_when_finetune](https://discord.com/channels/1238365980128706560/1242224662142779530/1257279259047886868)** (2 条消息): 

- **本次会议录像仍不可用**：一位成员指出**会议录像**仍不可用。**Dan Becker** 迅速做出回应，确认录像现在应该可用，并要求如果有任何问题请告知。
   - Dan Becker 实时处理了成员关于**会议录像可用性**的疑虑，并承诺迅速纠正任何潜在问题。
- **会议录像更新**：Dan Becker 提到**会议录像**现在应该已经上线。他要求如果有进一步的问题请告知。
   - **Dan Becker** 迅速处理了会议录像问题，确保其得到解决，并征求关于任何剩余问题的反馈。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1257396427227660379)** (1 条消息): 

- **使用 axolotl 微调 AutoModelForSequenceClassification**：一位成员询问了是否可以使用 **axolotl** 微调 **AutoModelForSequenceClassification**。他们指出该领域的文档清晰度存在问题。
   - 该成员寻求确认这种微调方法是否可行，或者仅仅是文档说明不足。
- **文档清晰度问题**：强调了使用 **axolotl** 微调 **AutoModelForSequenceClassification** 缺乏完善的文档流程。改进此类用例的文档可能会对用户产生重大帮助。
   - 更清晰的指南将帮助用户确定使用 axolotl 进行微调的可行性和工作流。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1257083590798217377)** (1 条消息): 

- **较轻松的 Accelerate 演讲：Hugging Face Accelerate**：Zachary Mueller 在最近的一次会议上分享了名为 *'Making Device-Agnostic ML Training and Inference Easy at Scale'* 的 **Hugging Face Accelerate** 演示。该 [YouTube 视频](https://www.youtube.com/watch?v=IBBeLNgGIIo) 概述了该工具的功能和应用。
   - 演讲涵盖了 **Hugging Face Accelerate** 如何简化**设备无关（device-agnostic）的 ML 训练和推理**，重点在于大规模使用的易用性。对于想要全面但不过于深奥的概述的人来说，强烈推荐这个演示。
- **Hugging Face Accelerate 深度探讨**：Zachary Mueller 最近的会议演讲介绍了用于实现 ML 训练和推理设备无关化的 **Hugging Face Accelerate**。该 [YouTube 视频](https://www.youtube.com/watch?v=IBBeLNgGIIo) 深入探讨了该工具的技术细节和可扩展性。
   - 演示强调了在各种设备环境中实现 **Hugging Face Accelerate** 的便捷性，使其适用于大规模 ML 应用。对于对设备无关解决方案感兴趣的人来说，这是必看内容。

**提到的链接**：<a href="https://www.youtube.com/watch?v=IBBeLNgGIIo">Hugging Face Accelerate: Making Device-Agnostic ML Training and Inference Easy... - Zachary Mueller</a>：Hugging Face Accelerate: Making Device-Agnostic ML Training and Inference Easy at Scale - Zachary Mueller, Hugging Face。Hugging Face Accelerate 是一个开源...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/)** (1 条消息): 

1dingyao: 嗨 <@466291653154439169>，

请协助处理 `adingyao-41fa41`。

非常感谢！
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/)** (1 条消息): 

saharn_34789: 有人在纽约吗？如果在波士顿见面，我可以安排。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1256875456142118942)** (4 条消息): 

- **Zed IDE 给 Sublime Text 用户留下深刻印象**：最近有人尝试了 [Zed IDE](https://zed.dev)，发现它非常令人印象深刻，特别是对于那些伴随 Sublime Text 成长的人来说。他们将这种体验描述为 *“哇哦”*。
   - 出于尝试其他选择的好奇，他们提到虽然 Zed 有不错的 AI 集成，但不如 Cursor 先进。他们表示有兴趣听听是否还有其他人尝试过 Zed。
- **对 Cursor 的好奇**：在体验过 Zed 的积极感受后，用户表达了尝试 **Cursor** 的好奇心。他们认为与 Zed 相比，Cursor 可能会提供更强大的 AI 集成。
   - 鉴于他们对比较不同 IDE 的兴趣，他们很想看看社区中是否还有其他人尝试过 **Cursor**。他们正在寻求有关其 AI 能力的反馈和意见。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1256959592483782768)** (1 条消息): 

- **Mistral-7b 表现优于竞争对手**：对于用户的特定用例，表现最好的模型是在 [OpenPipe](https://mlops.systems/posts/2024-07-01-full-finetuned-model-evaluation.html) 上微调的 **Mistral-7b-optimised**。该模型在评估中优于其他几个模型。
   - 使用其他供应商或本地机器上的**普通开放权重 Mistral** 也可以复现这些结果。用户指出，使用 **OpenPipe** 时，这个过程非常简单。
- **使用 OpenPipe 轻松进行微调**：一位用户强调，使用 **OpenPipe** 的微调过程非常直接。这使得使用**普通开放权重 Mistral** 复现结果变得毫不费力。
   - 易用性被特别强调。这有助于提升 **OpenPipe** 在微调 **Mistral-7b** 等模型时的可用性。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[bergum_rag](https://discord.com/channels/1238365980128706560/1252713659243827251/1256658846617964625)** (5 条消息): 

- **Reciprocal Rank Fusion 是混合搜索的稳健选择**：一位成员指出，尽管 **BM25** 和**向量搜索（vector search）**结果都有各自的问题，但 **Reciprocal Rank Fusion** 是结合两者的强大起点。
   - 另一位成员对此表示赞同，并说：*“所以 Reciprocal Rank Fusion 是混合搜索的一个很好的起点。谢谢！”*
- **查找幻灯片的困扰**：一位成员询问视频中提到的幻灯片（slide deck）在哪里，想知道是否已经分享。
   - 有人澄清说 **幻灯片通常在 Maven 上**，但在那里没有找到相关的幻灯片。
  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1256415982763901048)** (32 条消息🔥): 

- **Cohere 实习见解**：一位成员分享了他们对 Cohere 实习职位的兴趣，提到了他们目前正在为 AAAI 会议准备的一篇 workshop 论文以及之前的 co-op 经验。他们询问了具体的挑战和项目，以及开发新功能或改进 Cohere 的 LLM API 的机会。
   - 讨论涉及了 Cohere 对将 LLMs 与 reinforcement learning 等其他 AI 框架集成的研究项目的支持和资源。
- **Coral API 速率限制令用户沮丧**：几位用户对 Coral trial keys 的速率限制表示不满，该限制被设定为每分钟 5 次调用。分享了一个解释性[链接](https://docs.cohere.com/docs/rate-limits)，详细说明了 trial keys 和 production keys 之间的区别，后者提供每分钟 10,000 次调用。
   - “Trial keys 是为试用服务而设计的，”一位用户强调道，并建议升级到 production keys 以获得更高的吞吐量。
- **Aya-23 9B 模型查询**：成员们讨论了 Aya-23 模型不同版本的发布，特别是 [Hugging Face](https://huggingface.co/CohereForAI/aya-23-8B) 上提供的 8B 和 35B 版本。
   - 关于是否存在 Aya-23 9B 模型及其相关性存在困惑，并澄清目前模型被认为是足够的，且不用于推理。
- **Cohere 的 AI 推理增强**：分享了一个名为[“Cohere 今年将如何改进 AI 推理”](https://youtu.be/B45s_qWYUt8?si=_c7sQUFUN6bZa61m)的 YouTube 视频，首席执行官 Aidan Gomez 在视频中讨论了解决 AI 幻觉和增强推理能力的努力。
   - 该视频解释了为什么 Cohere 避免使用任何外部模型，而专注于内部开发。
- **学术研究支持**：针对有关学术研究开发工具的咨询，分享了一个 [GitHub 仓库](https://github.com/cohere-ai/cohere-toolkit)。它包含用于构建和部署 RAG 应用的预构建组件。
   - 该工具包可以作为学术项目的基础，确保简化的开发和部署。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/B45s_qWYUt8?si=_c7sQUFUN6bZa61m">How Cohere will improve AI Reasoning this year</a>：Cohere 首席执行官 Aidan Gomez 揭示了他们如何应对 AI 幻觉并提高推理能力。他还解释了为什么 Cohere 不使用任何外部...</li><li><a href="https://docs.cohere.com/docs/rate-limits">API Keys and Rate Limits</a>：未找到描述</li><li><a href="https://github.com/cohere-ai/cohere-toolkit">GitHub - cohere-ai/cohere-toolkit: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications.</a>：Cohere Toolkit 是一个预构建组件的集合，使用户能够快速构建和部署 RAG 应用。 - cohere-ai/cohere-toolkit</li><li><a href="https://huggingface.co/CohereForAI/aya-23-8B">CohereForAI/aya-23-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/CohereForAI/aya-23-35B">CohereForAI/aya-23-35B · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1257183563833409596)** (3 messages): 

- **Rig 全面集成 Cohere 模型**：**Rig** 库，一个专为构建 LLM 驱动的应用而设计的 Rust 库，现在已全面集成所有 **Cohere 模型**。查看社区中的[公告链接](https://discord.com/channels/954421988141711382/1218409701339828245/1250476945821532195)了解更多信息。
   - 发起了一项征求反馈的激励计划，为 Rust 开发者提供 **100 美元奖励**，旨在收集用户体验和改进 Rig 的建议。参与详情请访问[反馈表单链接](https://bit.ly/Rig-Feeback-Form)。
- **用于映射引用的社区 Notebook**：一位社区成员分享了一个 **Notebook**，用于将引用和文档映射回应用中的响应。Notebook 链接可在 [Google Colab](https://colab.research.google.com/drive/1o89bvd_JGRijQSFwiyXK-61oTD6okf_0?usp=sharing) 上获取。
   - 该 Notebook 允许存储 Markdown 响应，并高亮显示引用文本及展示来源。它还提供可自定义的参数来调整外观和感觉，欢迎社区进一步增强功能。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://bit.ly/Rig-Feeback-Form">Rig 反馈与奖励计划</a>: 感谢您对 Rig 的关注，这是一个使用 Rust 构建 LLM 驱动应用的开源库！由于 Rig 是一个全新的库，我们有兴趣收集 Rust 开发者的反馈...</li><li><a href="https://colab.research.google.com/drive/1o89bvd_JGRijQSFwiyXK-61oTD6okf_0?usp=sharing">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1256387194248237139)** (22 messages🔥): 

- **直接进入 DPO 训练**：一位用户询问 DPO 训练是否应该在常规训练之后进行，一些成员建议如果有足够的数据，直接进入 DPO/PPO **可能有效**。然而，目前还存在**不确定性**，建议咨询 DPO 专家。
   - 成员指出，对于这种特定设置应使用 **PreferenceDataset**，但 **Socket 专家**将对策略拥有最终决定权。一位用户强调，在 **llama2** 和 **Pythia 1B-7B** 等模型上直接进行 DPO/PPO 已取得成功。
- **使用 WandB 微调第一个模型**：一位用户使用 **Phi Mini (LoRA)** 完成了他们的第一个模型微调，并就**评估日志**寻求建议。建议使用 **WandBLogger** 以实现更好的日志管理和指标可视化。
   - 用户被提醒注意日志 **YAML 配置**中可能出现的重复键问题。关于如何正确设置 **WandBLogger** 以避免错误并改进训练监督，提供了相关建议。
- **评估微调日志**：用户讨论了**梯度大小**是否适合给定的数据集，并提出了潜在的调整建议。一位用户分享了日志文件，并就潜在的**过拟合**或增加 Epochs 所需的调整寻求见解。
   - 日志显示，在小数据集中，**Loss 和学习率**指标可能会有所波动，导致仅通过日志文件难以理解结果。强调了使用 **WandB** 来简化这一过程。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/nbeerbower/llama-3-gutenberg-8B">nbeerbower/llama-3-gutenberg-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://pytorch.org/torchtune/main/deep_dives/wandb_logging.html">记录到 Weights &amp; Biases &mdash; torchtune 主文档</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/jondurbin/gutenberg-dpo-v0.1">jondurbin/gutenberg-dpo-v0.1 · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/)** (1 messages): 

mikhail_ee: 来自 https://Hexagen.World 的一些新鲜地点。
  

---

### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1256332208185151652)** (12 messages🔥): 

- **Featherless.ai 发布新平台**：**Featherless.ai** 平台最近上线，提供对 Hugging Face 上所有 🦙 模型的即时访问，采用每月 10 美元起的固定订阅模式。用户无需任何 GPU 设置或下载模型，详情参见[此处](https://featherless.ai/)。
   - 早期用户主要将该平台用于 AI 角色本地应用（如 Sillytaven），少数用户则专注于特定用途，如语言微调或 SQL 模型。
- **Featherless.ai 探索 TTS 模型集成**：根据用户反馈，**Featherless.ai** 可能会探索将 [Piper](https://github.com/rhasspy/piper/blob/master/VOICES.md) 等 TTS 模型集成到其平台中。一位用户请求此功能，用于为在线游戏生成多样化的 NPC 语音。
   - 团队解释说，足够小的模型（如 100MB）可以在 CPU 上本地运行，而他们主要关注那些无法在 CPU 上高效运行的热门模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/rhasspy/piper/blob/master/VOICES.md">piper/VOICES.md at master · rhasspy/piper</a>：一个快速、本地的神经文本转语音系统。通过在 GitHub 上创建账户，为 rhasspy/piper 的开发做出贡献。</li><li><a href="https://huggingface.co/rhasspy/piper-voices">rhasspy/piper-voices · Hugging Face</a>：未找到描述</li><li><a href="https://featherless.ai/"> Featherless - Serverless LLM</a>：Featherless - 最新的 LLM 模型，Serverless 架构，可根据您的需求随时使用。
</li>
</ul>

</div>
  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1257021855739547811)** (2 messages): 

- **介绍 Niashamina**：新成员 Niashamina 介绍了自己，他是一位具备 Windows 技能的 **AI engineer**。
   - 他们提到已经创建了一个关于使用 **WSL** 在 Windows 上启动 **AI Town** 的 **README**，目前正在努力将其集成到 Docker 中。
- **用于 Windows 上 AI Town 的 WSL README**：Niashamina 宣布他们制作了一个关于使用 **WSL** 在 Windows 上启动 **AI Town** 的 **README**。
   - 他们询问了在 GitHub 上分享该 **README** 的合适位置。
- **Docker 集成进展**：Niashamina 提到他们正在进行 **将 AI Town 集成到 Docker** 的工作。
   - 他们指出工作仍在进行中，并幽默地质疑了其有用性。
  

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1256421679412940830)** (12 条消息🔥): 

- **Facebook 的新 LLM Compiler 模型**：Facebook 发布了其 **LLM Compiler 模型**，能够编译 C 语言、优化汇编代码和 LLVM IR。查看由 [Mozilla 打包的模型](https://huggingface.co/Mozilla/llm-compiler-13b-ftd-llamafile)，该模型被封装为名为 llamafiles 的可执行权重文件。
   - [llamafile](https://github.com/Mozilla-Ocho/llamafile) 可在 Linux, MacOS, Windows, FreeBSD, OpenBSD 和 NetBSD 上运行，支持 AMD64 和 ARM64 架构。Mozilla 已将其上传至 Hugging Face 以提高易用性。
- **在 Hugging Face 上使 llamafile 正式化**：为了让 llamafile 在 Hugging Face 上正式化，一名成员建议向 [模型库文件](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts) 提交 PR。此外，[代码片段文件](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries-snippets.ts) 也需要修改。
   - 这将在所有使用 llamafile 的仓库中添加一个按钮，并填充加载模型的代码。这对于无缝集成非常重要。
- **llama.cpp 的实际硬件需求**：在讨论实际硬件时，一名成员询问了在 iPhone 13 和 Raspberry Pi Zero W 等设备上运行 **llamafile** 的情况。会议澄清了 **llamafile** 需要 **64-bit 系统**，且无法在 Raspberry Pi Zero 上运行。
   - RAM 需求各不相同；例如，使用 **all-MiniLM-L6-v2.Q6_K.gguf** 向 HTTP 客户端提供 Embedding 服务仅占用 23mb 内存。因此，**llamafile server v2.0** 几乎不占用 RAM。
- **llamafile v0.8.9 发布**：**llamafile v0.8.9** 版本确认了对 **Android 的支持**，并引入了更好的 **Gemma2 支持**。该版本包含了一些修复，如 Windows 上的 GPU 提取问题，并增加了对 Google Gemma v2 的支持。
   - 此外还强调了对新服务器模式的进一步改进。查看 [发布详情](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.9) 以获取更多信息。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Mozilla/llm-compiler-13b-ftd-llamafile">Mozilla/llm-compiler-13b-ftd-llamafile · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.9">Release llamafile v0.8.9 · Mozilla-Ocho/llamafile</a>: 此版本使 Gemma2 的运行更接近 Google 的预期。af22695 使 gemma2-27b-it 与 aistudio.google.com 保持一致 41678c8 为 Gemma2 添加滑动窗口掩码 140eed5 为 Ge 添加 soft-capping...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.8">Release llamafile v0.8.8 · Mozilla-Ocho/llamafile</a>: 571b4e5 修复了阻止 Windows 上 GPU 提取的错误 4aea606 在 --server 模式下支持 flash attention 7fd9101 不要将 bf16 subnormals 刷新为零 7692b85 添加 Google Gemma v2 支持 72fb8ca 引入...
</li>
</ul>

</div>
  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1256490833641734164)** (8 messages🔥): 

- **“From ML Engineering to AI Engineering” 活动录像**：一位成员错过了 “From ML Engineering to AI Engineering” 活动并询问是否有录像。社区成员分享了一个 [Zoom 链接](https://events.zoom.us/ejl/AuK2GHiGErlhKuJWYRGND-S4fQfnvCgfPO50QMGDk0trhQ2ykP5H~A2Qw5SEBU-CKEpNn-eBw)，但该链接已被证实无效。
   - 尽管经过多次沟通，用户分享了多个链接，但结果均为无效页面。一位成员提到该链接需要代码，这使得访问变得更加复杂。
- **Data Talks Club LLM Zoomcamp：构建数据流水线**：即将举行的 Zoomcamp 专注于使用 dlt 和 LanceDB 创建开源数据流水线，计划于 7 月 8 日星期一举行，时长 90 分钟。工作坊内容涵盖从 REST API 提取数据、向量化并加载到 LanceDB，以及增量加载方法。
   - 参与者将学习如何在 Python notebooks、虚拟机以及 Airflow、Dagster 或 Mage 等编排器等不同环境中部署这些流水线。该活动由 dltHub 解决方案工程主管 Akela Drissner 赞助并主持。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://events.zoom.us/ejl/AuK2GHiGErlhKuJWYRGND-S4fQfnvCgfP">Something went wrong</a>：未找到描述</li><li><a href="https://lu.ma/cnpdoc5n?tk=uEvsB6">Open source data ingestion for RAGs with dlt · Luma</a>：创建可扩展的数据流水线 - Akela Drissner 关于活动 在本次实战工作坊中，我们将学习如何使用 dlt 构建数据摄取流水线以……</li><li><a href="https://events.zoom.us/ejl/AuK2GHiGErlhKuJWYRGND-S4fQfnvCgfPO50QMGDk0trhQ2ykP5H~A2Qw5SEBU-CKEpNn-eBw">Something went wrong</a>：未找到描述</li><li><a href="https://events.zoom.us/ejl/AuK2GHiGErlhKuJWYRGND-S4fQfnvCgfPO50QMGDk0trhQ2ykP5H~A2Qw5SEBU-CKEpNn-eBwjM_fMUTbTOlHM-LoWdsP8pDkCYNvYe-h892C3_JDfnGrM48-PFMLhBmVxJ43wdd3-9_kpWLMCWgHe6UIk-PCEp85k/home">All-in-one virtual event platform | Zoom Events</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/)** (1 messages): 

dbreunig：虽然只是从 5 月 19 日开始，但你肯定能看到顶部的竞争者们正在赶上来。
  

---

### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1257361863906431128)** (1 messages): 

- **海量语料库导致 tokenizer 训练崩溃**：一名成员正在 50 GB 的文本语料库上训练**针对旁遮普语（Panjabi language）的 BPE tokenizer**，但即使使用 1TB RAM 实例也遇到了**内存溢出（OOM）问题**。他们分享了[相关的 GitHub issues](https://github.com/huggingface/tokenizers/issues/1434)作为参考，并寻求更高效模型训练的建议。
   - 用户提到，即使在**预处理序列步骤超过 len(ds) 后**，内存仍在持续增加，类似于[这个 issue](https://github.com/huggingface/tokenizers/issues/1345)。他们怀疑问题可能出在 `tokenization_utils_fast.py` 文件中的 **train_from_iterator 函数**，但无法确定确切原因。
- **无法调试 tokenizer 训练函数**：用户尝试调试代码以了解 OOM 问题，但无法进入 `tokenization_utils_fast.py` 中的 `train_from_iterator` 函数。他们推测这可能是因为调用了在 Rust 中运行的可执行/二进制代码。
   - 用户追踪内存占用过高具体原因的尝试均未成功，导致了进一步的困惑，并需要社区建议或替代方案。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/tokenizers/issues/1434>)">Issues · huggingface/tokenizers</a>: 💥 Fast State-of-the-Art Tokenizers optimized for Research and Production - Issues · huggingface/tokenizers</li><li><a href="https://github.com/huggingface/tokenizers/issues/1345)">Issues · huggingface/tokenizers</a>: 💥 Fast State-of-the-Art Tokenizers optimized for Research and Production - Issues · huggingface/tokenizers</li><li><a href="https://github.com/huggingface/transformers/blob/e65502951593a76844e872fee9c56b805598538a/src/transformers/tokenization_utils_fast.py#L817>).">transformers/src/transformers/tokenization_utils_fast.py at e65502951593a76844e872fee9c56b805598538a · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---



---



---



{% else %}


> 完整的频道分类详情已在邮件中截断。
> 
> 如果你想查看完整的详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}