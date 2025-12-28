---
companies:
- anthropic
- scale-ai
- suno-ai
- microsoft
date: '2024-05-21T22:47:46.990001Z'
description: '**Anthropic** 发布了其机械可解释性（MechInterp）系列的第三篇论文 **《扩展单语义性》（Scaling Monosemanticity）**，将可解释性分析扩展到了
  **Claude 3 Sonnet** 上的 **3400 万个特征**。


  这项工作引入了**字典学习（dictionary learning）**的概念，用以分离重复出现的神经元激活模式，通过组合特征而非神经元，使模型的内部状态更具可解释性。论文揭示了与代码、错误、阿谀奉承（sycophancy）、犯罪、自我表征和欺骗相关的抽象特征，并通过固定（clamping）特征值展示了对模型行为进行有意识修改的可能性。


  这项研究标志着在前沿规模的**模型可解释性**和**神经网络分析**领域取得了重大进展。'
id: de6fed18-27ad-4a29-829f-3d6a4ec67722
models:
- claude-3-sonnet
- claude-3
original_slug: ainews-anthropic-cracks-the-llm-genome-project
people:
- emmanuel-ameisen
- alex-albert
title: Anthropic 的“LLM 基因组计划”：在 Claude Sonnet 上学习与钳制 3400 万个特征。
topics:
- model-interpretability
- dictionary-learning
- neural-networks
- feature-activation
- intentional-modifiability
- scaling
- mechanistic-interpretability
---

<!-- buttondown-editor-mode: plaintext -->**Dictionary Learning is All You Need.**

> 2024年5月20日至5月21日的 AI 新闻。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**376** 个频道，**6363** 条消息）。
预计节省阅读时间（按 200wpm 计算）：**738 分钟**。目录和 Discord 摘要已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

今天是新闻相对密集的一天，有来自 **Scale AI** 和 **Suno AI** 的巨额融资，以及对 **Microsoft Build** 发布内容的持续反响（如 [Microsoft Recall](https://x.com/dsiroker/status/1792956339515273537)），但我们在这里尽量保持技术性。

今天最大的新闻可能是 Anthropic 的 [Scaling Monosemanticity](https://www.anthropic.com/research/mapping-mind-language-model)，这是继 [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html#strategic-ways-out) (2022) 和 [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (2023) 之后，其现代 MechInterp 三部曲的第三部。第一篇论文专注于极小 ReLU 网络（5 个神经元上最多 8 个特征）的“主成分分析”，第二篇在真实的 Transformer（512 个神经元上 4096 个特征）上应用了 Sparse Autoencoders，而这篇论文现在扩展到了 **Claude 3 Sonnet 上的 100万/400万/3400万个特征**。这在真实的、前沿级模型上开启了各种可解释性的魔法：

 
![image.png](https://assets.buttondown.email/images/74a296cf-65a2-45c6-9c6a-46ad01c4fdb4.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/d96d1cac-e45b-40b0-8011-83b9223c0096.png?w=960&fit=max)
 

> 一定要看看 [特征 UMAPs](https://transformer-circuits.pub/2024/scaling-monosemanticity/umap.html?targetId=1m_1013764)

与其使用相对高深莫测的“superposition（叠加）”概念，现在的类比是“**dictionary learning**（字典学习）”，Anthropic 将其解释为：

> 借鉴自经典机器学习，它**隔离了在许多不同上下文中重复出现的神经元激活模式**。反过来，模型的任何内部状态都可以用少数活跃特征而不是许多活跃神经元来表示。正如字典中的每个英语单词都是由字母组合而成，每个句子都是由单词组合而成，AI 模型中的每个特征都是由神经元组合而成，每个内部状态都是由特征组合而成。（更多阅读请见[注释](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#related-work-dictionary)）

Anthropic 的 3400 万个特征编码了一些非常有趣的“抽象特征”，比如代码特征，甚至是 [错误](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#assessing-sophisticated-code-error)：

 
![image.png](https://assets.buttondown.email/images/8dd74aaf-5d74-4869-af68-55ca90142411.png?w=960&fit=max)
 

[sycophancy](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#safety-relevant-sycophancy)（谄媚）、[crime/harm](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#safety-relevant-criminal)（犯罪/伤害）、[self representation](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#safety-relevant-self)（自我表征），以及 [deception and power seeking](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#safety-relevant-deception)（欺骗和追求权力）：

 
![image.png](https://assets.buttondown.email/images/ca16bd0c-da17-45d1-b6bd-d010bf3f9c8b.png?w=960&fit=max)
 

完整可解释性研究的标志性证明是意图可修改性，Anthropic 通过将特征值钳制在其最大值的 -2 倍到 10 倍之间展示了这一点：

{% if medium == 'web' %}
 
![image.png](https://assets.buttondown.email/images/2b5bdf89-5b41-4350-96df-09b1825efbec.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/874d1492-5ac9-435f-be00-5afb8dea588e.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/92206ea3-5e0d-48d4-9ccc-ef90aedfaf7f.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/b675d446-aa5c-45e3-9528-c00efa8adade.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/646a3f7c-63e0-4e99-8c16-0479d3d73a7f.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/d619163e-0536-4d75-b82e-a145030cdf91.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/0f8d5ee9-d72e-42d3-bd25-bab68efe196d.png?w=960&fit=max)
 

{% else %}

> 您正在通过电子邮件阅读此内容。我们正将更多内容移至网页版，以创造更多空间并节省您的收件箱容量。**如果您愿意，请在 [网页版]({{ email_url }}) 查看摘录的图表。**

{% endif %}

不要错过来自 [Emmanuel Ameisen](https://x.com/mlpowered/status/1792948212728524917)、[Alex Albert](https://x.com/alexalbert__/status/1792936647665107108?s=46&t=90xQ8sGy63D2OtiaoGJuww)、[Linus Lee](https://x.com/thesephist/status/1793031719244734923) 和 [HN](https://news.ycombinator.com/item?id=40429326) 的详细分析。

---


{% if medium == 'web' %}


**目录**

[TOC] 


{% endif %}



---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 生成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**微软为 AI 时代推出 Copilot+ PC**

- **Copilot+ PC 被称为 Windows 40 年来最大的更新**：[@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1792624197572653351) 指出，Copilot+ PC 是目前速度最快、性能最强大的 AI 就绪型 PC，通过围绕 Copilot 重新构建整个技术栈，为 AI 时代重新定义了 PC。
- **在 Copilot+ PC 上演示了实时 AI 共同创作和相机控制**：[@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1792632555797180553) 展示了 Copilot 控制 Minecraft 游戏的过程，同时 [@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1792626471552336101) 演示了 PC 上的实时 AI 共同创作。
- **Copilot+ PC 具备“过目不忘”的记忆能力和顶尖性能**：[@yusuf_i_mehdi](https://twitter.com/yusuf_i_mehdi/status/1792637482988614027) 强调了 Copilot 对 PC 上执行的所有操作具有“过目不忘”（photographic memory）的记忆力。他还称其为[史上速度最快、最强大且最智能的 Windows PC](https://twitter.com/yusuf_i_mehdi/status/1792620591930826879)。

**Scale AI 以 138 亿美元估值融资 10 亿美元**

- **Scale AI 在由 Accel 领投的融资轮中以 138 亿美元估值筹集了 10 亿美元**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1792905417065914858) 宣布了这一融资消息，并表示 Scale AI 在加速前沿数据供应和铺平 AGI 之路方面处于前所未有的有利地位。
- **Scale AI 通过提供数据为几乎所有领先的 AI 模型提供动力**：作为与算力和算法并列的三大 AI 支柱之一，[@alexandr_wang](https://twitter.com/alexandr_wang/status/1792905420744581597) 解释说 Scale 正在为几乎所有领先的 AI 模型提供数据支持。
- **资金将用于加速前沿数据供应并为 AGI 铺路**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1792905424251060575) 表示，这笔资金将帮助 Scale AI 进入下一阶段，即加速前沿数据的丰度，为通往 AGI 铺平道路。

**Suno 融资 1.25 亿美元以构建 AI 驱动的音乐创作工具**

- **Suno 融资 1.25 亿美元，让任何人都能用 AI 创作音乐**：[@suno_ai_](https://twitter.com/suno_ai_/status/1792922276683297162) 将利用这笔资金加速产品开发并扩大团队，通过技术放大人类的创造力，构建一个每个人都能创作音乐的未来。
- **Suno 正在招聘人才，为音乐人社区构建最佳工具**：Suno 认为他们的社区值得拥有[最好的工具](https://twitter.com/suno_ai_/status/1792922276683297162)，这需要具备技术专长且真正热爱音乐的顶尖人才。他们邀请人们加入，共同塑造音乐的未来。

**Meta 自动测试生成工具的开源实现发布**

- **Cover-Agent 作为 Meta 自动测试生成论文的首个开源实现发布**：[@svpino](https://twitter.com/svpino/status/1792897013920538944) 分享了 Cover-Agent，这是一个开源工具，实现了 Meta 在 2 月份发表的关于自动增加现有代码库测试覆盖率的论文。
- **Cover-Agent 生成独特且有效的测试以提高覆盖率，表现优于 ChatGPT**：[@svpino](https://twitter.com/svpino/status/1792897013920538944) 强调，虽然自动单元测试生成并不新鲜，但要做好却很难。Cover-Agent 只生成能够运行并增加覆盖率的独特测试，而 ChatGPT 则会产生重复、无效且无意义的测试。

**Anthropic 发布关于解释领先大语言模型的研究**

- **Anthropic 在最新研究中首次详细展示了领先大语言模型的内部机制**：在名为 "Scaling Monosemanticity" 的[新研究论文和博客文章](https://twitter.com/AnthropicAI/status/1792935506587656625)中，Anthropic 对领先的 Large Language Model 内部进行了前所未有的详细展示。
- **从 Anthropic 的 Claude 3 Sonnet 模型中提取出数百万个可解释特征**：利用无监督学习技术，[@AnthropicAI](https://twitter.com/AnthropicAI/status/1792935511582986466) 从 Claude 3 Sonnet 的激活层（activations）中提取了可解释的“特征（features）”，这些特征对应于模型学习到的抽象概念。
- **部分提取的特征与安全相关，为潜在的模型故障提供了见解**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1792935524220481777) 发现了与令人担忧的能力或行为（如不安全代码、偏见、不诚实等）相对应的安全相关特征。研究这些特征可以深入了解模型的潜在故障模式（failure modes）。

**迷因与幽默**

- **OpenAI 未经许可克隆 Scarlett Johansson 的声音，引发《小美人鱼》式的类比**：[@bindureddy](https://twitter.com/bindureddy/status/1792683787647848880) 和 [@realSharonZhou](https://twitter.com/realSharonZhou/status/1792688472861573192) 对 OpenAI 未经许可为其 AI Agent 助手克隆 Scarlett Johansson 声音的新闻做出反应，并将其与《小美人鱼》的情节进行了类比。
- **由于电子马克杯，收藏的加热咖啡杯遗憾被闲置**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1792901873696956867) 思考电池能量密度是否足以支撑一个加热搅拌棒，从而为任何杯子提供电子温度控制，因为他妻子的 Ember 马克杯让她其他的杯子都派不上用场了。
- **针对 Microsoft Copilot 的“过目不忘”，Linux 权限迷因引发关注**：[@svpino](https://twitter.com/svpino/status/1792957041612337331) 分享了一个关于 Linux 文件权限的迷因，以回应 Microsoft Copilot 具备“过目不忘”记忆力（photographic memory）的功能。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**OpenAI 争议与法律问题**

- **Scarlett Johansson 考虑对 OpenAI 采取法律行动**：在 /r/OpenAI 中，讨论指出 [**Scarlett Johansson 发表声明，谴责 OpenAI 在她拒绝其请求后，在 GPT-4o 演示中使用了与她非常相似的 AI 语音**](https://www.reddit.com/r/OpenAI/comments/1cwucf9/psa_yes_scarlett_johansson_has_a_legitimate_case/)。OpenAI 声称该语音属于另一位女演员，但 Johansson 正在寻求法律途径。/r/OpenAI 的进一步讨论表明，[OpenAI CEO Sam Altman 在发布前引用电影《Her》的推文以及再次联系 Johansson 的行为，可能会加强她关于 OpenAI 故意复制其形象的指控](https://www.reddit.com/r/OpenAI/comments/1cw9gxj/open_ai_respondes_to_sky_sounding_like_scarlett/)。
- **OpenAI 移除 "Sky" 语音选项**：针对争议，OpenAI 已[移除了听起来与 Scarlett Johansson 相似的 "Sky" 语音](https://www.reddit.com/r/OpenAI/comments/1cwenul/sky_assistant_voice_is_paused_for_sounding_like/)，并声称该女演员是在联系 Johansson 之前聘请的。/r/OpenAI 正在辩论[名人是否应该对相似的声音拥有所有权](https://www.reddit.com/r/OpenAI/comments/1cwwhnt/openai_appears_to_be_expanding_their_legal_team/)。

**GPT-4o 和 Copilot 演示与功能**

- **微软在 Windows 11 中演示由 GPT-4o 驱动的 Copilot**：[Twitter 上分享的一段视频](https://x.com/msftcopilot/status/1792626848641274342?s=46)显示，微软演示了集成到 Windows 11 中的基于 GPT-4o 的 Copilot 功能，包括[**游戏时的实时语音辅助和生活指导**](https://www.reddit.com/r/OpenAI/comments/1cwman0/is_this_why_openai_didnt_release_their_desktop/)。/r/OpenAI 的一些人推测，这种深度的 OS 集成是 OpenAI 尚未发布自家桌面应用的原因。
- **GPT-4o 语音/视觉功能将面向 Plus 用户推出**：/r/OpenAI 分享的 GPT-4o 演示图像显示，[新的语音和视觉功能将在未来几个月内向 Plus 用户推出](https://www.reddit.com/gallery/1cwqhcb)，而不是最初暗示的几周内。([图片来源](https://i.redd.it/qh9kczvw1n1d1.png))
- **令人印象深刻的 OCR 能力**：/r/singularity 的一个帖子分享了 [GPT-4o 的 OCR 成功读取并修正图像中部分遮挡文本的示例](https://www.reddit.com/r/singularity/comments/1cwil8s/just_had_an_interesting_experience_with_4o_doing/)，展示了先进的计算机视觉技术。
- **幻觉可能增加**：/r/OpenAI 的一些用户报告称，[与基础 GPT-4 模型相比，GPT-4o 似乎更容易产生幻觉（hallucinations）](https://www.reddit.com/r/OpenAI/comments/1cwi1dl/does_4o_seem_more_prone_to_hallucinating_than_4/)，这可能是由于增加了额外的模态。

**AI 进展与通往 AGI 之路**

- **GPT-4 展现出人类水平的心智理论**：一篇[新的 Nature 论文](https://www.nature.com/articles/s41562-024-01882-z)发现，GPT-4 展现出了人类水平的心智理论（theory of mind），在检测讽刺和暗示方面比人类表现更好。其主要局限性似乎源于对表达观点所设置的护栏（guardrails）。
- **对推理能力进展的担忧**：/r/singularity 的一个帖子表达了[担忧：尽管 GPT-4 功能强大，但在发布后的一年里，推理能力和智能并没有显著提高](https://www.reddit.com/r/singularity/comments/1cwe0yc/is_anyone_else_concerned_thats_its_been_over_a/)，这减缓了通往 AGI 的道路。

**幽默与梗图**

- 一张[梗图开玩笑地暗示 Joaquin Phoenix 正在考虑起诉 OpenAI](https://www.reddit.com/r/singularity/comments/1cwwhnt/breaking_joaquin_phoenix_now_considering_suing/)，理由是他们聘请了一个胡子相似的男人，以此嘲讽 Scarlett Johansson 争议。
- 一张 [image macro 梗图调侃了 /r/singularity 对 GPT-4o 热潮的反应](https://i.imgur.com/63WoZO2.png)。
- 分享了一个 AI 生成的荒诞幽默示例，描绘了 [1864 年亚伯拉罕·林肯会见 Hello Kitty 讨论国家安全](https://i.redd.it/e0bvhii75m1d1.jpeg)的场景。

---

# AI Discord 回顾

> 摘要的摘要的摘要

1. **优化模型以突破界限**：

  - **Transformer 集成与模型贡献引发热议**：工程师们正在将 [ImageBind](https://arxiv.org/abs/2305.05665) 集成到 `transformers` 库中，同时另一位工程师的 [PR 已被合并](https://github.com/huggingface/transformers/pull/29004)，修复了微调 AI 模型的一个问题。此外，**[llama-cpp-agent](https://huggingface.co/spaces/pabloce/llama-cpp-agent)** 通过利用 ZeroGPU 暗示了在计算效率方面的进展。
   - **[Modular 带来的 LLM 效率提升](https://www.modular.com/blog/fast-k-means-clustering-in-mojo-guide-to-porting-python-to-mojo-for-accelerated-k-means-clustering)**：Modular 的新 nightly 版本在改进的 SIMD 优化和异步编程技术的支持下，有望通过 Mojo 中的 k-means 聚类等方法带来巨大的性能提升。

   - 成员们强调了像 [Torch 的 mul_()](https://x.com/mlpowered/status/1792948212728524917) 这样的工具的重要性，以及 [vLLM 和内存优化技术](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)在显存（VRAM）受限系统上增强模型性能的实际用途。

2. **斯嘉丽·约翰逊反击 AI 语音克隆**：

   - **[斯嘉丽·约翰逊对 OpenAI 的诉讼](https://www.npr.org/2024/05/20/1252495087/openai-pulls-ai-voice-that-was-compared-to-scarlett-johansson-in-the-movie-her)**：约翰逊因语音复制争议起诉 OpenAI，迫使该公司移除该模型，并可能重塑围绕 AI 生成语音克隆的法律格局。

   - 讨论强调了关于[声音肖像权和知情同意](https://platformer.news/open-ai-scarlett-johansson-her-voice-sam-altman)的伦理与法律辩论，行业内将其与涉及 Drake 等音乐家的未经授权内容移除事件进行了对比。

3. **新 AI 模型引爆基准测试**：

   - **Phi-3 模型与 ZeroGPU 激励 AI 开发者**：微软在 HuggingFace 上发布了 **Phi-3 small (7B)** 和 **Phi-3 medium (14B)** 模型，它们具有 128k 的上下文窗口，在 [MMLU 和 AGI Eval 任务](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)中表现出色。与此同时，HuggingFace 新推出的 **[ZeroGPU 计划](https://huggingface.co/zero-gpu-explorers)提供价值 1000 万美元的免费 GPU 访问权限**，旨在推动独立开发者和学术界的 AI Demo 创作。

   - **发现 PaliGemma 的文档处理能力**：[Merve 强调了](https://x.com/giffmana/status/1791541209883717973?s=46) **PaliGemma** 在文档理解方面的强大实力，并分享了一系列 Hugging Face 链接和相关推文。关于 Mozilla 的 DeepSpeech 以及从 [LangChain](https://python.langchain.com/v0.1/docs/integrations/document_loaders/airtable/) 到 [3D Gaussian Splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#editing) 的各种资源咨询，显示了社区对多样化 AI 技术的广泛兴趣。

   - **[适用于 LLM 的 M3 Max](https://x.com/osanseviero/status/1791567896482635801)** 的性能受到称赞，尤其是在配备 96GB 内存的情况下，这为模型能力的重大跨越提供了动力，并为大语言模型训练效率设定了新标准。

4. **协作努力塑造 AI 的未来**：

   - **[Hugging Face 的 LangChain 集成](https://huggingface.co/blog/langchain)**：新软件包旨在促进模型无缝集成到 LangChain 中，为社区项目提供新的架构并优化交互能力。

   - **[Memary 网络研讨会](https://lu.ma/nzh3o83f)** 展示了一种用于自主 Agent 的开源长期记忆解决方案，解决了知识图谱生成和记忆流管理中的关键需求。

5. **AI 社区关于伦理和实用 AI 实现的热议**：

   - **[Anthropic 的负责任缩放政策](https://www.anthropic.com/news/reflections-on-our-responsible-scaling-policy)**：计算能力的提升预示着即将到来的重大创新，并与新的负责任缩放政策（Responsible Scaling Policy）保持一致，以管理 AI 开发中的伦理问题。

   - **[AI 领域的协作](https://lu.ma/fp0xr460)** 在巴黎和旧金山的 PizzerIA 聚会等活动中继续蓬勃发展，增强了检索增强生成（RAG）技术以及社区对 AI 创新的参与度。


---

{% if medium == 'web' %}



# 第一部分：高层级 Discord 摘要

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **使用 PyMuPDF 和 Tesseract 进行 PDF 提取**：工程师们分享了使用 [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/tutorial.html) 和 OCR 进行 PDF 文本提取的工具和工作流，提到了 `fitz` 和 `sort=True` 选项，以及用于处理复杂 PDF 的 ABBYY 和 MarkerAPI。

- **优化 LLM 训练和 Fine-Tuning**：技术讨论重点介绍了用于多用户服务的 vllm 等工具，并参考了使用 pyenv 和 virtualenv 的工作流，以及 Axolotl 中的依赖项。分享了来自 Anthropic 关于模型可解释性研究的见解，并提及了 [Claude Sonnet 的研究](https://www.anthropic.com/research/mapping-mind-language-model)。

- **创新学习与协作**：工程师们就 Vik 的 Marker API 和用于 Fine-Tuning 模型的 GitHub 仓库等资源进行了头脑风暴，重点关注多语言模型 Fine-Tuning 和共同解决问题。

- **Modal 上的模型服务技巧**：为了高效地提供 LLM 模型服务，建议工程师使用 `modal serve` 而非 `modal run`，并分享了关于成本管理和最小化容器空闲时间的见解。可以通过 [此表单](https://bit.ly/modal-credits) 获取 Modal 额度，注册即可获得 500 美元额度以及免费层级每月 30 美元的额度。

- **班加罗尔见面会的热情**：人们对班加罗尔见面会表现出浓厚兴趣。在不损害模型性能的情况下融入新语言的技术、关于日语 LLM 的性能讨论以及特定地区的见面会都是讨论的热点。

- **课程结构与参与**：新解释的课程结构包括 Fine-Tuning 工作坊、答疑时间（Office Hours）和会议演讲。资深参与者交流了关于 Llama3 的技术挑战、超参数以及用于 Fine-Tuning 的资源，如斯坦福大学的 [Pyvene](https://github.com/stanfordnlp/pyvene/issues/46)。

- **Hugging Face 的 Accelerate 备受推崇**：鼓励成员查看 Accelerate，它对于在不同配置中分发 PyTorch 代码非常有用，并提供了在 [Hugging Face 的 GitHub](hhttps://github.com/huggingface/accelerate/tree/main/examples) 上从 `nlp_example` 开始的示例。还重点介绍了用于估算模型内存和 FLOPS 的资源，例如 [Model Memory Utility](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)。

- **Axolotl 和 BitsandBytes 查询**：针对 bitsandbytes 和 macOS 上的 MLX 支持的技术咨询得到了解答，特别参考了 [GitHub 上的 issue](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1436)。关于 OpenAI 和 Axolotl 之间 Fine-Tuning 对比的提议引发了对 OpenAI 基于 Token 的 30 分钟服务的兴趣。

- **对系统化 Prompt Engineering 的好奇**：人们对 Jason 的系统化 Prompt Engineering 技术产生了浓厚兴趣，并热切期待他在即将举行的工作坊环节中分享他的“秘籍”。

- **Gradio 易于上手的界面开发**：Gradio 的维护者邀请大家进行提问和分享 Demo，提倡其在开发 AI 模型用户界面方面的易用性，并分享了有用的指南，如 [快速入门教程](https://www.gradio.app/guides/quickstart) 以及如何 [快速构建聊天机器人](https://www.gradio.app/guides/creating-a-chatbot-fast)。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 与 Tako 联手**：Perplexity AI 与 Tako 合作，通过**高级知识搜索与可视化**增强用户体验，目前已在美国上线并支持英文，移动版预计很快推出。详情请见[此处](https://trytako.com/blog/introducing-tako-and-perplexity-integration)。

- **Perplexity 助力深度讨论**：工程师们就使用 **Perplexity AI** 交换了见解，并对平台忠诚度展开了热烈辩论，讨论了 GPT-4 和 Claude 3 Opus 的模型用例，并对 Tako 图表等新功能表示期待。在面临服务宕机时，他们也团结一致，显示出强大的用户社区凝聚力。

- **Perplexity API 的挑战与收获**：AI 工程师指出了将 Perplexity API 与 Open WebUI 集成时面临的挑战，特别是围绕模型兼容性的困惑。解决方案涉及代理服务器和精确的 Docker 命令，工程师们积极分享了进展和建议。

- **Perplexity：通往知识的门户**：**sharing** 频道的贡献强调了 Perplexity AI 处理多样化话题的能力，从历史、数学到脚本创建和技术计算概念，体现了该平台作为知识资源的通用性。

- **API 集成策略与初期困难**：**pplx-api** 频道充满了关于配置 Docker 以优化 **Perplexity API** 使用的战术讨论，确认了缺少 `/models` 端点的问题，并澄清了目前 API 不支持图像等局限性。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Phi-3 模型与 ZeroGPU 激励 AI 开发者**：微软在 HuggingFace 上发布了 **Phi-3 small (7B)** 和 **Phi-3 medium (14B)** 模型，具备 128k 上下文窗口，在 MMLU 和 AGI Eval 任务中表现出色。与此同时，HuggingFace 新推出的 **ZeroGPU 计划提供价值 1000 万美元的免费 GPU 访问权限**，旨在推动独立开发者和学术界的 AI Demo 创作。

**探索 PaliGemma 的文档处理能力**：Merve 通过一系列 Hugging Face 链接和相关推文，强调了 **PaliGemma** 的文档理解能力。关于 Mozilla 的 DeepSpeech 以及从 LangChain 到 3D Gaussian Splatting 的各种资源咨询，反映了社区对多样化 AI 技术的广泛兴趣。

**LangChain 记忆技巧**：针对机器人遗忘先前交互的常见挑战，社区提供了将对话历史整合到基于 **LLM** 的聊天机器人中的实用建议。同时，一位用户批评了 **llama3 8b 4bit** 的故事增强能力，揭示了该模型在创意过程中的局限性。

**Transformer 集成与模型贡献引发热议**：工程师们正在将 ImageBind 集成到 `transformers` 库中，另一位工程师的 PR 已被合并，修复了微调 AI 模型的一个问题。此外，**llama-cpp-agent** 建议通过利用 ZeroGPU 来提升计算效率。

**视觉技术查询与方案交流**：在计算机视觉领域，关于 Vision Transformers 中高级补丁技术（patching techniques）的论文请求以及截图中 Zero-shot 目标检测的方法成为焦点。这些讨论表明在目标识别任务中需要更复杂的方法和 Zero-shot 方法论。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **斯嘉丽·约翰逊反击 AI 语音克隆**：斯嘉丽·约翰逊（Scarlett Johansson）因未经授权复制其声音而起诉 OpenAI。受此影响，在公众关注度日益增高的情况下，OpenAI 已经下架了该语音模型。

- **Phi-3 在 Hugging Face 首次亮相**：微软在 [Hugging Face](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) 上发布了 **Phi-3-Medium-128K-Instruct 模型**，宣称其拥有更强的基准测试表现和 128k 的超长上下文。公会的工程师们目前正在讨论其优点以及大上下文窗口带来的挑战。

- **Colab T4 GPU 难题已解决**：由于 PyTorch 在 Colab 上对 T4 GPU 的检测不完善，导致 Notebook 运行混乱，直到 Unsloth 的[更新](https://x.com/danielhanchen/status/1792985678030221464)发布后才得以解决。该修复程序解决了 PyTorch 错误地假设 T4 支持 bfloat16 的问题。

- **围绕 MoRA 展开讨论**：一场关于名为 **MoRA** 的新微调方法的讨论已经开启，并提供了 [arXiv 论文](https://arxiv.org/abs/2405.12130)链接。公会成员对在工作流中测试其原生实现表现出了浓厚的早期兴趣。

- **Dolphin-Mistral 的精简成功**：有传闻称 **dolphin-mistral-2.6** 仅通过约 2 万条样本进行精炼，就达到了原版（使用了数百万条样本）的指令遵循性能。这种新颖的训练方法引起了关注，预计今年晚些时候发布的论文将详细介绍这一过程。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **针对 AI 爱好者的诈骗警报**：建议用户避开 **Stable Diffusion** 服务的诈骗订阅，并仅通过官方网站 [stability.ai](https://stability.ai) 获取合法访问权限。

- **Stable Diffusion 也支持离线运行**：确认了 **Stable Diffusion** 具备在无网络连接的情况下本地运行的能力，减少了对持续在线连接的依赖。

- **Stable Diffusion 设置的技术支持**：社区为那些在 **Stable Diffusion** 和 **ComfyUI** 等工具设置中遇到困难的用户提供支持，用户们正在分享解决安装问题的建议。

- **欧盟 AI 法案引发关注**：新推出的**欧盟 AI 法案（EU AI Act）**引发了关于其对 AI 生成内容影响的辩论，包括对强制水印和执法挑战的担忧。

- **缓解硬件性能瓶颈**：关于 **Stable Diffusion** 性能问题的讨论建议检查系统配置并使用 diffusers 脚本，并推测新硬件设置上可能存在热节流（thermal throttling）现象。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **实时 AI**：GPT-4o 以 **每秒 2-4 帧** 处理视频的能力引发了讨论，预计 **GPT-4o** 集成到 Microsoft Copilot 后将带来实时语音和视频功能。OpenAI 的 Sky 功能因声音酷似斯嘉丽·约翰逊而引发了法律和伦理辩论。

- **模型精度与特性**：GPT-4 的 128,000 token 上下文窗口同时包含 Prompt 和响应。此外，如何让 AI 实现精准的语言表达和特定行为（类似于电影《她》中的 AI）也是热门话题。

- **追求简洁的 Prompt Engineering**：分享了一些巧妙的 Prompt 构建技巧，以使 GPT-4 的输出保持在特定的字符限制内，重点在于清晰的模板和对 token 计数的战略性使用，以确保响应简洁且相关。

- **AI 中的伦理与法律**：出售 AI 生成艺术的能力得到了确认，但版权问题的复杂性被重点强调，社区成员还对 GPT-4 评估数值的能力表示了担忧。

- **安全与更新**：在 AI 首尔峰会上宣布了一项重大的安全更新，更多详情可见 [OpenAI 安全更新](https://openai.com/index/openai-safety-update/)，这进一步强化了 OpenAI 对负责任 AI 开发的承诺。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**以管理员身份运行 LM Studio 以访问日志**：以管理员权限运行 **LM Studio** 可以解决服务器日志空白的问题，让用户能够访问用于故障排除所需的日志文件。

**AVX2 是运行 LM Studio 的必备条件**：了解 **AVX2 指令集**是运行 LM Studio 的必要条件，用户可以使用 [HWInfo](https://www.hwinfo.com/download/) 等工具检查 CPU 对 AVX2 的兼容性。缺乏 AVX2 支持的旧款 CPU 将面临软件兼容性问题。

**通过 Civit.ai 进行高效图像生成**：为了提高图像质量，成员们建议使用 **Automatic1111** 和 **ComfyUI** 等本地模型，并结合来自 [Civit.ai](https://civitai.com/) 的支持资源，同时提醒系统配置中需要充足的 VRAM 和 RAM。

**针对特定模型进行优化**：为确保 **LM Studio** 中响应的完整性，将 **max_tokens** 设置为 **-1** 可以解决当该值设为 null 时遇到的响应提前截断问题。社区还讨论了使用特定模型的 Prompt，如 **MPT-7b-WizardLM** 所示；可参考 [Hugging Face](https://huggingface.co/DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF) 获取所需的量化级别和模板。

**ROCm 与 Linux 在 AMD GPU 上的结合**：拥有 AMD GPU 的 Linux 爱好者受邀测试集成了 ROCm 的 **LM Studio** 早期版本，该版本已列在 [AMD 支持的 GPU 列表](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)中。使用不受支持 GPU 的用户也发回了成功报告，用户分享了他们不同的 Linux 发行版体验以及关于 **infinity fabric (fclk)** 频率同步影响系统性能的发现。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**聚焦 Mojo 社区会议**：[Mojo 社区会议](https://modular.zoom.us/j/89417554201?pwd=Vj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1)已举行，尽管部分人遇到了通知问题，但录像现已上传至 [YouTube](https://www.youtube.com/playlist?list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D)。最初关于是否需要商业 Zoom 账号的困惑已得到澄清，确认并非必要。

**通过 k-means 聚类提升 Mojo 性能**：一篇[博客文章](https://www.modular.com/blog/fast-k-means-clustering-in-mojo-guide-to-porting-python-to-mojo-for-accelerated-k-means-clustering)向读者介绍了如何在 Mojo 中使用 k-means 聚类算法，并承诺与 Python 相比会有显著的性能提升。

**代码难题与编译器纪事**：讨论内容包括处理字符串中的空终止符（null terminators）、探索异步编程，以及在 Mojo 中使用 Lightbug HTTP 框架。社区制定了解决方案和变通方法，一些技术查询引发了 [GitHub issue 讨论](https://github.com/saviorand/lightbug_http/issues/41)。

**Nightly 更新应对编译器复杂性**：详细介绍了[最新的 nightly Mojo 编译器版本](https://github.com/modularml/mojo/compare/7e8cd37ff8fe2ddbe69a3cca787e59abf6357d76...69e92d0040af838de8f3f0fdba1cea92f1904986)，并围绕字典中的 `pop` 方法、字符串中的 Unicode 支持以及其他 [GitHub issue](https://github.com/modularml/mojo/issues/2696) 和 [PR 审议](https://github.com/modularml/mojo/pull/2739)展开了对话。

**深入研究 SIMD 优化**：成员们参与了关于在 Mojo 中优化 SIMD gather 和 scatter 操作的讨论，克服了 ARM SVE 和内存对齐等挑战，并就减少 gather/scatter 操作提出了建议，还分享了为迭代解码器排序离散内存的技巧。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Kubernetes：是必需品还是过度设计？**：一些成员认为，**像 EKS 这样的托管 Kubernetes 服务**可以有效地取代本地 ML 服务器，尽管其他人指出 Kubernetes 对于 ML 基础设施并非必不可少；决策应根据项目需求量身定制。

**Triton 焕然一新**：**Triton** 库的更新包括一个改进教程可读性的 [pull request](https://github.com/triton-lang/triton/pull/3959)，以及关于 **GPU kernel 特性**如何影响最大 block size 的新见解。

**处理 SASS 和复杂操作**：工程师们讨论了关于 **SASS** 的学术资源，并商讨了在先进 **NVIDIA 架构**上进行原子操作时，使用 "cucomplex" 还是 "cuda::std::complex" 的优劣。

**高效内存使用的 Torch 技巧**：用户发现 **Torch 原生的 `*` 运算符**会使内存占用翻倍，而 `mul_()` 则不会，且在 CUDA 设备分配方面，`torch.empty_like` 的性能优于 `torch.empty`。

**激活量化成为 CUDA 的焦点**：重点转向利用新一代 GPU 上的 **2:4 稀疏性**和 **fp6/fp4** 等特性进行**激活量化**，并着眼于将这些特性集成到 **torch.compile** 中，以增强图级优化。

**Torchao 0.2 引入自定义扩展**：[GitHub](https://github.com/pytorch/ao/releases/tag/v0.2.0) 上的 **torchao 0.2 版本**引入了自定义 CUDA 和 CPU 扩展，以及 **NF4 张量与 FSDP** 的集成，以改进模型训练。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **旧金山招募安全专家**：新成立的 **英国人工智能安全研究所 (AISI)** 旧金山办公室正提供具有竞争力的薪资以吸引人才。他们正在开展合作，包括 [英加 AI 安全伙伴关系](https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership)。

- **反对 SB 1047 的行动号召**：AI 社区的利益相关者正动员起来反对 **加利福尼亚州的 SB 1047 法案**，认为该法案严厉的监管措施可能会威胁开源 AI 的发展，详见[此分析](https://context.fund/policy/sb_1047_analysis.html)。

- **FLOPs 计算细节探讨**：针对 Attention 机制的 FLOPs 计算展开了深入讨论，引用了 [EleutherAI cookbook](https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py) 进行 FLOPs 计算，强调了包含 QKVO 投影的必要性。

- **多模态模型成为焦点**：讨论集中在通过多模态训练改进 AI 模型，包括在 **CLIP** 中加入音频进行零样本分类时观察到的益处。在 [ImageBind](https://arxiv.org/abs/2305.05665) 等模型中注意到了性能提升，但未发现涌现能力。

- **MoE 效率受到关注**：新研究介绍了 **[MegaBlocks](https://arxiv.org/abs/2211.15841)**，这是一个用于 MoE 训练的资源高效型系统，它放弃了 token dropping 并利用块稀疏操作，显著提升了训练效率。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Temporal 征服工作流管理**：在讨论工作流编排后，一名成员确认选择了 **Temporal.io** 而非 Apache Airflow，理由是其功能更强大。

- **在 AI 迷宫中穿行**：成员们指出了各种挑战，例如**效果不佳的 LLM 排行榜**和 **Chatbot Arena 偏颇的评分**。微软的 **Copilot+** 发布会引发了讨论，而 **Yi-1.5** 模型的发布因满足了不同的上下文窗口大小需求而受到关注。

- **研究计划蓬勃发展**：**Manifold Research Group** 在 NEKO 项目中的持续进展反映了社区开发综合模型的动力，**Phi-3 Vision** 的发布进一步强调了这一点，该模型通过微调和优化技术实现了视觉与文本的对齐。

- **描绘 AI 边界**：通过 ASCII 和生成的模拟图像进行的创意探索，引发了关于 AI 功能和符号能力的讨论，特别是 **WorldSim** 的应用。

- **流动中的知识**：分享的 **Obsidian 知识图谱**延时摄影以及对 **rerankers** 公开评估方法的支持呼吁，反映了工程社区动态协作的本质。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

**Sky 语音被停用**：由于用户反馈，OpenAI 已暂时停止在 ChatGPT 中使用 **Sky 语音**；该公司正在努力解决这些担忧。这一决定引发了关于 AI 生成语音以及此类技术固有的伦理考量的持续讨论。[阅读推文](https://x.com/OpenAI/status/1792443575839678909)

**CogVLM2：谨慎使用**：**CogVLM2 模型**因支持 8K 内容长度而受到关注，但其附带的争议性许可协议限制了不利于中国国家利益的使用，引发了关于真实开源原则的讨论。该许可还规定任何争议均受中国司法管辖。[查看许可证](https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE)

**AI Copilot：从代码助手到生活伴侣？**：Mustafa Suleyman 对即将推出的 **Copilot AI** 的预告引发了各种反应，该 AI 可以实时与物理世界互动，反映了社区对于 AI 辅助与隐私之间日益模糊的界限的复杂情绪。[查看推文](https://fxtwitter.com/mustafasuleyman/status/1792623877744623806)

**斯嘉丽·约翰逊的语音“替身”困境**：OpenAI 语音助手使用与女演员 **Scarlett Johansson** 相似的语音，引发了关于 AI 模仿人类语音（尤其是名人语音）的伦理边界和法律问题的辩论。

**Sakuga-42M 数据集在机器人冲击下消失**：高需求和自动化下载导致 **Sakuga-42M 数据集**从托管平台移除，这引发了关于在激进的网络爬虫面前维护可访问数据集挑战的对话。[Hacker News 讨论](https://news.ycombinator.com/item?id=40389711)



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 语音争议导致“Sky”停用**：由于法律压力和负面的公众认知，OpenAI 已停止使用与 Scarlett Johansson 相似的语音 AI “Sky”，突显了语音模仿和知情同意的伦理担忧。此事件让人联想到涉及冒充音乐家等公众人物的争议，引发了对问责制和 AI 行业明确伦理准则需求的讨论。

- **Anthropic 算力实现飞跃**：[Anthropic 已将其算力资源提升至其前代模型 Opus 的四倍](https://www.anthropic.com/news/reflections-on-our-responsible-scaling-policy)，引发了社区对该公司正在研发项目的兴趣。细节尚不明确，但算力的巨大增幅指向了重大进展。

- **AI Arena 面临 Hard Prompts 挑战**：[LMsysorg](https://fxtwitter.com/lmsysorg/status/1792625968865026427) 引入的 “Hard Prompts” 类别加剧了 AI 模型评估的竞争，事实证明这对 Llama-3-8B 等模型来说尤为吃力，该模型在与 GPT-4-0314 的对比中表现出明显的性能下滑。这种严苛的评估引发了对当前评测模型（如 Llama-3-70B-Instruct）有效性的质疑。

- **OpenAI 违反超级对齐承诺**：OpenAI 正面临来自 [Fortune 文章](https://fortune.com/2024/05/21/openai-superalignment-20-compute-commitment-never-fulfilled-sutskever-leike-altman-brockman-murati/) 指控的审查，称其违背了将 20% 算力分配给其 Superalignment 团队的承诺，导致团队重组。这一披露引发了关于产品开发与 AI 安全之间优先级的对话，一些人认为该公司的举动是其背离承诺的预料之中。

- **域名交易与 AI 数据集困境**：Nathan Lambert 以每年 7 美元的成交价购买了域名 rlhfbook.com，并开玩笑地谈论了使用 AI Books4 数据集训练 LLM 的潜在法律风险，这既展示了 AI 开发有趣的一面，也体现了数据使用的严肃法律考量。关于 Microsoft Surface AI 出现延迟的提及引发了对本地处理与依赖云端的安全验证之间平衡的疑问，暗示了一个潜在的优化领域。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **“Memory Tuning” 为 LLM 树立新标杆**：Lamini 的 Sharon Zhou 介绍了一种名为 “Memory Tuning” 的新技术，声称该技术能显著减少大语言模型 (LLM) 的幻觉（降至 <5%），性能超越了 LoRA 和传统的微调方法。关于早期访问的细节和进一步解释尚待公布 ([Sharon Zhou 的推文](https://x.com/realsharonzhou/status/1792578913572429878))。

- **Scarlett Johansson 的 AI 声音争议**：在 Scarlett Johansson 的律师暗示将采取法律行动后，OpenAI 暂时停止使用一种与她相似的 AI 生成声音，引发了关于肖像权和代言的辩论 ([NPR 文章](https://www.npr.org/2024/05/20/1252495087/openai-pulls-ai-voice-that-was-compared-to-scarlett-johansson-in-the-movie-her))。

- **规模扩张：Scale AI 获得 10 亿美元注资**：Scale AI 以 138 亿美元的估值获得了 10 亿美元融资，计划利用这笔投资增强前沿数据，并目标在 2024 年底前实现盈利，此轮融资由 Accel 领投 ([Fortune 文章](https://fortune.com/2024/05/21/scale-ai-funding-valuation-ceo-alexandr-wang-profitability/))。

- **微软发布 Phi 3 模型系列**：微软在 MS Build 上发布了 Phi 3 模型，其基准测试性能可与 Llama 3 70B 和 GPT 3.5 媲美，支持高达 128K 的上下文长度，并以 MIT 许可证发布 ([关于 Phi 3 模型的推文](https://x.com/reach_vb/status/1792949163249791383))。

- **Pi 简介：具有情感智能的 LLM**：Inflection AI 宣布转向创建更具情感和认知能力的 AI 模型，每天有超过 100 万用户与其共情式 LLM “Pi” 互动，展示了 AI 的变革潜力 ([Inflection AI 的公告](https://inflection.ai/redefining-the-future-of-ai))。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **速率限制引发不满**：Azure 的 GPT-32k 模型一直触及 Token 速率限制，用户指出了在 Azure OpenAI API 2023-07-01-preview 版本下向 ChatCompletions_Create 操作发起请求时的具体问题。

- **Phi-3 模型受到关注**：社区一直在探索 Phi-3 模型在数据推理方面的卓越表现，研究了使用监督微调的 [Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) 以及具有直接偏好优化 (Direct Preference Optimization) 功能的 [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)。

- **LLM 交互的新花样即将到来**：一种名为 “Action Commands” 的新型 LLM 交互方法正在流传，分享经验和寻求反馈的讨论帖可以在[这里](https://x.com/leonjcoe/status/1792946945528320382)找到。

- **简洁与冗长之争仍在继续**：管理 Wizard8x22 等模型冗长程度的策略正在评估中，一些成员主张降低重复惩罚 (repetition penalty) 以确保更简洁的输出。

- **OpenRouter 对非营利组织展现开放态度**：针对用户的 Error 400 账单问题及其对非营利组织折扣的请求，OpenRouter 讨论了其 20% 利润率的定价政策。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Grok Enthusiasts Gear Up**: AI 工程师们正表现出对使用 [PyTorch 版本](https://huggingface.co/hpcai-tech/grok-1) 训练 **Grok** 的极大热情，讨论通过 **torchtune integration** 进行潜在增强，并对比了包括 **Mi300x** 与 **H100s** 在内的计算平台。
- **Sharp Turn in Mistral Finetuning**: 成员们正在排查 **Mistral 7B** 微调中的问题，提议范围涵盖了从全量微调 (full finetuning) 到 **Retrieval-Augmented Generation (RAG)** 技术，以解决内容保留问题，详见分享的 [配置指南](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/lora.yml)。
- **OOM Woes and Wisdom**: 显存溢出 (OOM) 错误是一个核心话题，为了应对 VRAM 限制，提出了多种解决方案，包括 **gradient accumulation steps**、**mixed precision training**、**model parallelism**、Batch Size 调整以及 **DeepSpeed ZeRO optimization**，更多详情见 [Phorm.ai](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=79ea3546-1ab6-4fe1-8984-1d8eb8183eda)。
- **M3 Max Takes the Stage**: **M3 Max** 芯片因其 LLM 性能表现而获得赞誉，建议配备 96GB RAM 以充分发挥大语言模型的潜力。
- **Code Debacles and Python Queries**: 对话内容包括排查 **Transformers** 库中涉及 `CohereTokenizer` 的语法错误 (Syntax Errors)，并探索了更快的替代方案（如 [GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files) 中所述），以及寻找能够加速“语音转文本 (speech-to-text) 到 LLM 再到语音合成 (speech synthesis)”链路的 Python 库。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Memary Makes Memories**: 即将举行的网络研讨会将聚焦于 **memary**，这是一个为 autonomous agents 设计的开源长期记忆系统，承诺深入探讨其如何利用 LLMs 和 neo4j 进行知识图谱 (knowledge graph) 生成。活动定于太平洋时间周四上午 9 点，工程师可以通过 [此处](https://lu.ma/nzh3o83f) 注册参加。

**Knack for Stacking RAG Techniques**: 在检索增强生成 (RAG) 领域，@hexapode 将在巴黎的 PizzerIA 分享高级策略，而 Tryolabs 和 ActiveLoop 将于下周二在旧金山的首次线下见面会上进行演讲——在此 [注册](https://t.co/qIGOmCW62G)。

**GPT-4o Integrates with LlamaParse**: LlamaIndex.TS 文档已增强，**GPT-4o** 现在可以与 LlamaParse 无缝协作以分析复杂文档。此外，根据其 [最新产品](https://t.co/2cnsBH411k)，你可以使用 Azure Container Apps 安全地执行 LLM 生成的代码。

**Resolving Twin Data Quandaries**: 工程师们讨论了为文档计算唯一哈希值的方法，以避免在 Pinecone 中出现重复，并研究了处理 VectorStoreIndex 中空节点的权宜之计。

**Streamlining Systems and Storage**: 分享了关于如何使用 `chat_agent.agent_worker.prefix_messages` 修改 OpenAI agent 的 system prompt 的见解，以及由于其 Langchain 集成而使用 Airtable 优于 Excel/Sqlite 的优点——信息可见 [此处](https://python.langchain.com/v0.1/docs/integrations/document_loaders/airtable/)。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **情感 AI 即将到来**：据 [VentureBeat 文章](https://venturebeat.com/ai/exclusive-inflection-ai-reveals-new-team-and-plan-to-embed-emotional-ai-in-business-bots)报道，Inflection AI 计划将**情感 AI 整合到商业机器人**中，提升了开发更具共情能力的 AI 伴侣的前景。对话还触及了 AI 角色，其中通过 [Tenor 的 GIF](https://tenor.com/view/ddlc-doki-doki-literature-club-just-monika-monika-gif-20717242) 澄清了来自《心跳文学部》（*Doki Doki Literature Club*）的 *Just Monika* 梗。
  
- **解决 AI Town 的记忆难题**：社区反馈表明，AI Town 中的 AI 角色经常**无法记住过去的互动**，导致对话重复。建议调整 `convex/constants.ts` 以修改 `NUM_MEMORIES_TO_SEARCH`，从而优化过去交流内容的检索。

- **克服 SQL Schema 困惑**：工程师们分享了用于导出 AI Town 对话数据的 **SQL 查询**和工具，包括 [townplayer](https://github.com/cocktailpeanut/townplayer/blob/main/index.html) 等 GitHub 仓库链接以及一段解释性的 [Twitter 线程](https://x.com/cocktailpeanut/status/1786421948638965870)，方便了数据的操作和理解。

- **3D AI 介绍**：提及了一个涉及 **3D 角色聊天机器人**的在研项目预告，并建议在社区内的另一个频道查看更多详情。 

- **动画解释缺乏冲击力**：围绕 *AI waifus* 文化影响的趣味讨论，强调了 AI 角色开发在用户界面中的幽默感与重要性。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LLM 与文本类型的纠缠**：包括 [Hermes 2 Pro - Mistral 7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) 和 [OpenAI 的 chatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) 在内的结构化和非结构化数据处理 LLM，对文本类型没有天生的偏好，但在 finetuning 后表现出色。

**LangChain 的社区贡献**：`langchain-core` 包针对基础抽象进行了精简，而 `langchain-openai` 和 `langchain-community` 则包含更多细分领域的集成，详见[架构概览](https://python.langchain.com/v0.2/docs/concepts/#architecture)。

**顺序链（Sequential Chains）实战**：推荐了一个 [YouTube 教程](https://youtu.be/2xxziIWmaSA?si=3wkNt_huJKu3xK3t&t=1694)，用于设置顺序链，即一个链的输出成为下一个链的输入。

**聊天定制化的佣金**：一个联盟计划以 25% 的佣金吸引用户推广 **ChatGPT Chrome Extension - Easy Folders**，详情见[此处](https://easyfolders.promotekit.com/)，尽管一些用户反映该扩展的性能存在问题。

**Agent 升级与 PDF 洞察**：一篇 [Medium 文章](https://medium.com/ai-advances/upgrading-your-agents-a-smooth-transition-from-legacy-langchain-to-langgraph-c552cb60fcb3)阐述了从 LangChain 迁移到更新的 **LangGraph** 平台的过程，同时还提供了一份使用 **Upstage AI solar 模型**查询 PDF 的指南，可在[此处](https://medium.com/@sonam.gupta1105/creating-a-pdf-query-assistant-with-upstage-ai-solar-and-langchain-integration-6631280093b5)查看。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**AI 赋能的 DevOps 正在兴起**：一位全栈初级 DevOps 工程师正在创建一个 **lite O1 AI 项目**，旨在为各种 DevOps 任务提供**隐蔽的语音辅助**，并寻求社区对开发和实际应用的见解。

**OpenInterpreter 与日常技术的共生**：工程师们正在探索 **Open Interpreter** 如何简化他们的工作流程，从跨设备的代考参考到总结技术文档，强调了 AI 在日常技术任务中的实际影响。

**语音技术与 OpenInterpreter 的结合**：一位社区成员正在将 **Text-to-Speech** 与 Open Interpreter 集成，并被引导至相关的 [GitHub 仓库](https://github.com/OpenInterpreter/01)以进一步推进其项目。

**连接查询与缺失的手册**：一位成员在指南缺失说明的情况下寻求将笔记本电脑连接到灯光应用的帮助，而另一位成员则在为其版本的 **Open Interpreter lite 01** 组装 3D 打印零件寻求建议。

**对错失机会的幽默调侃**：用户 *ashthescholar.* 轻松地指出了命名规范中错失的一个机会，展示了技术社区有趣的一面。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Codegen-350M-mono 解决兼容性问题**：针对在 Transformers.js 中使用 **Codegen-350M-mono** 遇到的兼容性问题，成员们分享了一个 [ONNX 版本](https://huggingface.co/Xenova/codegen-350M-mono)，标志着跨平台实现的成功。
- **使用 CommandR+ 进行翻译**：对于韩英翻译任务，**CommandR+** 被强调为一个有效的工具，[Chat API 文档](https://docs.cohere.com/docs/chat-api)提供了包含示例代码和使用说明的资源。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Johansson 与 OpenAI 的语音争议**：在版权指控和 Scarlett Johansson 发表[声明](https://x.com/BobbyAllyn/status/1792679435701014908)后，OpenAI 已暂停在 GPT-4o 中使用 Sky 语音，并用 Juniper 代替。
- **GPT-4o 的统一模态方法**：GPT-4o 通过集成文本、视觉和音频的统一模型增强了功能，提升了交互中的情感理解，但也可能使模型的性能和潜在用例复杂化。
- **Lem 对系统可靠性的看法**：工程师们分享了来自 Stanisław Lem 作品的观点，主张构建具有韧性而非完美可靠的系统，并承认系统故障的不可避免性。
- **语音克隆的道德迷宫**：工程师们讨论了语音克隆技术带来的微妙伦理和法律挑战，警告不要仅依赖立法来保护身份。
- **万众瞩目的 Qualcomm 新套件**：Qualcomm 发布了适用于 Windows 的 Snapdragon Dev Kit，备受期待。其规格包括 4.6 TFLOP GPU、32GB RAM 和 512GB 存储空间；售价 899.99 美元，常被拿来与 Apple 的 Mac Mini 比较。[阅读更多](https://www.theverge.com/2024/5/21/24158603/qualcomm-windows-snapdragon-dev-kit-x-elite)关于该开发套件的信息。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **SFT 与 Preference Optimization 之争**：一位社区成员质疑 **Supervised Fine-Tuning (SFT)** 的必要性，因为 **Preference Optimization** 似乎可以通过调整期望和非期望输出的概率分布来达到类似的效果。
- **Phi3 Vision 获得认可**：拥有 42 亿参数的模型 **Phi3 Vision** 因其在图像流上令人印象深刻的低延迟实时推理能力而受到赞誉，[Jan P. Harries 的帖子](https://x.com/jphme/status/1792950682695479734)强调了其在机器人领域的潜在应用。
- **模型对比：Phi3 Vision vs Moondream2**：社区在图像推理任务上对比了 **Phi3 Vision** 和 [Moondream2](https://huggingface.co/spaces/vikhyatk/moondream2)，指出 Moondream2 的幻觉较少，但在某些数据集上存在问题。
- **Microsoft 发布新模型**：Microsoft 推出了 **70 亿和 140 亿参数**的新 AI 模型，提到这些发布仅提供了 instruct 版本，引发了社区成员的兴趣和讨论。
- **需要进一步讨论**：提供的见解引发了进一步讨论，社区可能会深入探讨这些模型的功效和应用。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **SQLite-VeC 成为焦点**：Alex 介绍了 [`sqlite-vec`](https://github.com/asg017/sqlite-vec)，这是一个新的 **SQLite 向量搜索扩展**，描述了其在 RAG 和语义搜索等功能中的应用；该扩展与 **cosmopolitan** 兼容，目前处于 beta 阶段。
- **深入了解 'sqlite-vec'**：Alex 的一篇[详细博客文章](https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html)揭示了 `sqlite-vec` 的愿景，即通过更好的性能和更简单的应用嵌入来超越 `sqlite-vss`；二进制文件和包将提供给各种编程环境。
- **呼吁协作与实验**：考虑到 `sqlite-vec` 处于 beta 阶段，Alex 表示愿意为任何有兴趣在项目中集成该扩展或进行故障排除的人提供支持。
- **社区对 Llamafile 集成的热议**：`sqlite-vec` 与 **Llamafile** 集成的可能性引发了成员们的兴奋，突显了该扩展提升当前项目能力的潜力。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**GPT-4o 表现优于其前代产品**：一位 Discord 频道成员详细描述了 **GPT-4o** 在复杂法律推理领域相比 GPT-4 和 GPT-4-Turbo 的显著性能飞跃，并通过一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/evan-harris-387375b2_the-release-of-gpt-4o-from-openai-has-been-activity-7196856963454959617-w1i1)强调了这一进步的重要性。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Manifold Research Group 寻求合作：** Manifold Research Group 是一个专注于 *generalist models* 和 AI Agent 的开源研发实验室，目前正在寻求合作者，并分享了其 [研究日志](https://www.manifoldrg.com/research-log-038/)、[Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) 和 [GitHub](https://github.com/ManifoldRG?ref=manifoldrg.com) 的链接。
- **NEKO Project 为开源 AI 规划路线：** NEKO Project 正雄心勃勃地构建一个大规模开源 generalist model，该模型融合了多种模态，包括控制和机器人任务，其详细内容已在 [项目文档](https://docs.google.com/document/d/e/2PACX-1vQELDXCIT9tn7Uq5vxQG4_3HsrkQcuBRqvXm-MkxW06Zkh-LP3G9z7TP7a-2MNWyA/pub?ref=manifoldrg.com) 中列出。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1242043650851078205)** (225 条消息🔥🔥): 

- **掌握使用 Python 和 OCR 进行 PDF 提取**：成员们分享了使用 PyMuPDF 和 tesseract 进行 PDF 文本提取的工具和代码片段。有人强调了 `fitz` 配合 `sort=True` 选项的高效性，而其他人则讨论了 ABBYY 和 MarkerAPI 等 OCR 解决方案，用于处理复杂和低质量的 PDF（[PyMuPDF 教程](https://pymupdf.readthedocs.io/en/latest/tutorial.html)）。
  
- **探索与优化 LLM 训练和 Fine-Tuning**：详细讨论了优化 LLM 训练设置，并引用了 vLLM 等工具来同时服务多个用户。用户还分享了使用 pyenv、virtualenv 的 Fine-Tuning 工作流，并解决了 Axolotl 中的依赖问题（[StarCoder2-instruct](https://github.com/bigcode-project/starcoder2-self-align)）。

- **处理大语言模型与内存优化**：参与者探索了处理大语言模型的方法，特别是在 GPU 上，并分享了新研究的见解。讨论内容包括内存调优、使用 vLLM 进行高效模型推理服务，以及 Anthropic 关于模型可解释性的最新发现（[Claude Sonnet 研究](https://www.anthropic.com/research/mapping-mind-language-model)）。

- **协作学习与资源共享**：参与者通过共享资源和工具建立联系，例如用于 PDF 处理的 Vik's Marker API 以及各种用于 Fine-Tuning 模型的 GitHub 仓库。许多人还分享了经验，并寻求在多语言和特定领域模型 Fine-Tuning 方面的合作（[Marker API](https://github.com/satish860/PDF-Extraction-API)）。

- **工作坊后勤与参与**：讨论了关于课程录像、时区管理和访问课程材料的问题，并确认所有课程都将进行录像。参与者还反思了赞助商的额度分配以及课程 Discord 会议的组织结构（[Modal 示例](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving)）。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://github.com">GitHub: 从这里开始构建</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。在这里为开源社区做贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪 Bug 和功能...</li><li><a href="https://huggingface.co/spaces/hf-accelerate/model-memory-usage">Model Memory Utility - hf-accelerate 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#post-installation-actions">Linux 版 CUDA 安装指南</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2403.10131">RAFT: 使语言模型适配特定领域的 RAG</a>: 在大规模文本语料库上预训练大语言模型 (LLMs) 现已成为标准范式。在将这些 LLMs 用于许多下游应用时，通常会额外加入新的知识...</li><li><a href="https://github.com/satish860/PDF-Extraction-API">GitHub - satish860/PDF-Extraction-API: 一个基于 Marker 库的 API，用于执行 Marker 响应。</a>: 一个基于 Marker 库的 API，用于执行 Marker 响应。 - satish860/PDF-Extraction-API</li><li><a href="https://github.com/poloclub/unitable">GitHub - poloclub/unitable: UniTable: 迈向统一的表格基础模型</a>: UniTable: 迈向统一的表格基础模型 - poloclub/unitable</li><li><a href="https://github.com/VikParuchuri">VikParuchuri - 概览</a>: VikParuchuri 有 88 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://pymupdf.readthedocs.io/en/latest/tutorial.html">教程 - PyMuPDF 1.24.4 文档</a>: 未找到描述</li><li><a href="https://x.com/jxnlco/status/1792549015273513102">来自 jason liu (@jxnlco) 的推文</a>: 如果你是一家正在构建 RAG 的公司并希望提升你的工程团队水平，请填写此表单。 https://q7gjsgfstrp.typeform.com/to/SL656ADC 我们将邀请其他运营者分享他们的故事，提供...</li><li><a href="https://x.com/mlpowered/status/1792948212728524917">来自 Emmanuel Ameisen (@mlpowered) 的推文</a>: 今天，我们宣布已经在 Sonnet 上实现了字典学习，从世界上最好的模型之一中提取了数百万个特征。这是该技术首次成功...</li><li><a href="https://github.com/pyenv/pyenv?tab=readme-ov-file#automat">GitHub - pyenv/pyenv: 简单的 Python 版本管理</a>: 简单的 Python 版本管理。通过在 GitHub 上创建账户为 pyenv/pyenv 的开发做出贡献。</li><li><a href="https://github.com/VikParuchuri/surya">GitHub - VikParuchuri/surya: 支持 90 多种语言的 OCR、布局分析、阅读顺序、行检测</a>: 支持 90 多种语言的 OCR、布局分析、阅读顺序、行检测 - VikParuchuri/surya</li><li><a href="https://x.com/Kyrannio/status/1792440824355332313">来自 Kiri (@Kyrannio) 的推文</a>: 出于好奇，我找到了 GPT-4o iOS 系统提示词：“你是 ChatGPT，一个由 OpenAI 训练的大语言模型，基于 GPT-4 架构。你正在通过 ChatGPT iOS 界面与用户聊天...”</li><li><a href="https://x.com/llama_index/status/1791258285993230786">来自 LlamaIndex 🦙 (@llama_index) 的推文</a>: 使用 GPT-4o 进行结构化图像提取 🖼️ GPT-4o 在整合图像/文本理解方面处于领先地位，我们创建了一个完整的 cookbook，向你展示如何使用 GPT-4o 提取结构化...</li><li><a href="https://x.com/VikParuchuri/status/1788966758742982696">来自 Vik Paruchuri (@VikParuchuri) 的推文</a>: Marker v2 发布了！主要新功能： - 提取图像/图表 - 更好的表格解析 - Pip 包安装 - 可商用 - 改进了支持更多语言的 OCR - 更好的列排序...</li><li><a href="https://x.com/rohanpaul_ai/status/1792640477641970029?s=">来自 Rohan Paul (@rohanpaul_ai) 的推文</a>: DPO (Direct Preference Optimization) 可能不如 PPO (Proximal Policy Optimization) —— 来自 Google 的最新研究 🤔 它调查了为什么在线强化学习算法（如 PPO）失败...</li><li><a href="https://x.com/rohanpaul_ai/status/1792640477641970029?s=46&t=mgKHGVn_Owt0fh3SjofSeg">来自 Rohan Paul (@rohanpaul_ai) 的推文</a>: DPO (Direct Preference Optimization) 可能不如 PPO (Proximal Policy Optimization) —— 来自 Google 的最新研究 🤔 它调查了为什么在线强化学习算法（如 PPO）失败...</li><li><a href="https://x.com/realSharonZhou/status/1792576516444065967">来自 Sharon Zhou (@realSharonZhou) 的推文</a>: 幻觉是生产级 LLMs 和 Agent 的最大阻碍之一。我们在内部以及为客户实现了零幻觉（<5%）。我们已经能够微调 LLMs 以召回特定...</li><li><a href="https://github.com/shisa-ai/shisa-v2/wiki/Ablations">消融实验</a>: 通过在 GitHub 上创建账户为 shisa-ai/shisa-v2 的开发做出贡献。</li><li><a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/text_generation_">GitHub - modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/text_generation_</a>: </li>

inference.py">modal-examples/06_gpu_and_ml/llm-serving/text_generation_inference.py at main · modal-labs/modal-examples</a>: 使用 Modal 构建的程序示例。通过在 GitHub 上创建账号为 modal-labs/modal-examples 的开发做出贡献。</li><li><a href="https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving">modal-examples/06_gpu_and_ml/llm-serving at main · modal-labs/modal-examples</a>: 使用 Modal 构建的程序示例。通过在 GitHub 上创建账号为 modal-labs/modal-examples 的开发做出贡献。</li><li><a href="https://www.anthropic.com/research/mapping-mind-language-model">映射大语言模型的思想</a>：我们已经确定了数百万个概念是如何在 Claude Sonnet（我们部署的大语言模型之一）内部表示的。这是有史以来第一次对现代生产级大语言模型内部进行的详细观察...</li><li><a href="https://github.com/bigcode-project/starcoder2-self-align/tree/main?tab=readme-ov-file#data-generation-pipeline">GitHub - bigcode-project/starcoder2-self-align: StarCoder2-Instruct: 用于代码生成的全透明且许可宽松的自我对齐（Self-Alignment）</a>：StarCoder2-Instruct：用于代码生成的全透明且许可宽松的自我对齐 - bigcode-project/starcoder2-self-align</li><li><a href="https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts">依赖解析 - pip 文档 v24.1.dev1</a>：未找到描述</li><li><a href="https://github.com/explosion/prodigy-segment">GitHub - explosion/prodigy-segment: 通过 Facebook 的 Segment-Anything 模型在 Prodigy 中选择像素。</a>：通过 Facebook 的 Segment-Anything 模型在 Prodigy 中选择像素。 - explosion/prodigy-segment</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/453">pip install flash-attn 总是出现 ModuleNotFoundError: No module named 'packaging'，但实际上我已经 pip install packaging 了 · Issue #453 · Dao-AILab/flash-attention</a>：正在收集 flash-attn，使用缓存的 flash_attn-2.0.7.tar.gz (2.2 MB)，正在安装构建依赖 ... 完成，正在获取构建 wheel 的要求 ... 错误 error: subprocess-exited-with-error × Gettin...</li><li><a href="https://github.com/pyenv/pyenv?tab=readme-ov-file#automatic-installer">GitHub - pyenv/pyenv: 简单的 Python 版本管理</a>：简单的 Python 版本管理。通过在 GitHub 上创建账号为 pyenv/pyenv 的开发做出贡献。</li><li><a href="https://github.com/pyenv/pyenv-virtualenv">GitHub - pyenv/pyenv-virtualenv: 一个用于管理 virtualenv（又名 python-virtualenv）的 pyenv 插件</a>：一个用于管理 virtualenv（又名 python-virtualenv）的 pyenv 插件 - pyenv/pyenv-virtualenv</li><li><a href="https://buttondown.email/ainews/archive/ainews-to-be-named-3447/#llm-finetuning-hamel-dan-discord">[AINews] Skyfall</a>：不考虑超级对齐（superalignment），Google Scarlett Johansson 就是你所需的一切。2024/5/17-2024/5/20 的 AI 新闻。我们检查了 7 个 subreddit，384 个 Twitter 和 29...</li><li><a href="https://github.com/xl0">xl0 - 概览</a>：全职学习者。(Linux, Biology, Electronics) -> AI :heart: 编写一些可爱的软件。 :two_hearts: 欢迎各种令人兴奋的机会！ - xl0</li><li><a href="https://chinese-reader.vercel.app">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1242011294861627402)** (141 条消息🔥🔥): 

- **创意写作 AI 引起关注**：成员们讨论了开发用于辅助创意写作的 AI，重点关注通过 **prompt engineering** 生成创意并克服写作障碍。建议通过 Fine-tuning 使模型符合特定流派或写作风格。
- **BERT 和 Sentence Transformers 的实际应用**：成员们介绍了使用 **BERT-type models** 和 **sentence-transformers**（如 [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)）来执行聚类和语义搜索等任务。示例代码展示了该模型在句子编码方面的实际用法。
- **法律文档摘要讨论**：讨论了使用 **LLMs** 总结法律文档并提供客户支持。探索了结合 Fine-tuning、RAG 和 prompt engineering 来完成法律研究和策略制定等任务。
- **客服场景下的 RAG vs Prompting**：一位成员重新考虑了在设计用于帮助客户创建功能工单的 LLM 时，是使用 Fine-tuning 还是 prompt engineering。最初倾向于通过 Fine-tuning 来调整语气和流程，但后来出于实际考虑，更倾向于使用 Prompting。
- **心理健康和医疗 AI 用例出现**：多位成员提议利用 **Fine-tuning 和 RAG** 创建用于医疗编码、总结患者记录和提供心理健康建议的 AI 系统。示例包括总结 ICD-10 代码和提供针对性的心理健康见解。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">sentence-transformers/all-MiniLM-L6-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/">LlamaParse - LlamaIndex</a>: 未找到描述</li><li><a href="https://unstructured.io/">Unstructured | 为您的 LLM 提供的非结构化数据 ETL</a>: Unstructured 通过将数据转换为大语言模型可以理解的格式，帮助您为 AI 准备好数据。轻松将您的数据连接到 LLMs。</li><li><a href="https://www.youtube.com/watch?v=sTQaJyrI-zg&list=PLVVTN-yNn8rvEwlY8ClxDUWeVPVfdifYj&index=8&ab_channel=StanfordOnline">Stanford CS25: V2 I Common Sense Reasoning</a>: 2023年2月14日，常识推理，Yejin Choi。在本系列讲座中，我们将研究 Transformers 的工作细节，并深入探讨不同类型的...</li><li><a href="https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax-Pattern.html">pattern - Amazon CloudWatch Logs</a>: 未找到描述</li><li><a href="https://www.onetonline.org/find/all">查看所有职业</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1242056437182632046)** (49 条消息🔥): 

- **班加罗尔（Bangalore）见面会获得关注**：多位成员表示有兴趣在**班加罗尔组织见面会**。该提议获得了极大的热情，来自班加罗尔的用户纷纷响应。
  
- **关于非英语语言模型微调的咨询**：针对如何在不降低性能的情况下为模型添加新语言，进行了一场有趣的交流。建议包括使用 **90/10% 的数据混合**以减少灾难性遗忘（catastrophic forgetting），并可能采用 **layer freezing** 等技术。

- **日语 LLM 性能讨论**：一位成员分享了关于日语语言模型开发的广泛更新，提到了各种模型和 Benchmark。提供了指向其 [benchmark framework](https://github.com/shisa-ai/shaberi) 和一个在日语表现上媲美 GPT-3.5-turbo 的 [Hugging Face 模型](https://huggingface.co/shisa-ai/shisa-v1-llama3-70b) 的链接。

- **训练数据集详细评论链接**：特别提到了 [Shisa 项目](https://huggingface.co/augmxnt/shisa-7b-v1) 以及对 [公开日语训练集](https://github.com/AUGMXNT/shisa/wiki/A-Review-of-Public-Japanese-Training-Sets#analysis) 的评论，为日语 LLM 开发中的挑战和方法论提供了见解。

- **多城市见面会倡议**：发出了在其他多个地点举行见面会的邀请，包括 [NCR](https://x.com/sivil_taram/status/1791159335999201380)、浦那（Pune）、新加坡和马来西亚。来自这些地区的几位成员表达了热烈的响应和参与意愿。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/sivil_taram/status/1791159335999201380">来自 Qian Liu 🔭 (@sivil_taram) 的推文</a>: 介绍 Sailor-14B 模型和 Sailor2 项目 🚢 我们很高兴宣布发布 Sailor-14B 模型，包括 Base 和 Chat 版本！✅基于 Qwen1.5-14B 模型构建，...</li><li><a href="https://huggingface.co/blog/leonardlin/llm-jp-eval-eval">评估 llm-jp-eval（评估很难）</a>: 未找到描述</li><li><a href="https://huggingface.co/shisa-ai/shisa-v1-llama3-70b">shisa-ai/shisa-v1-llama3-70b · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/AUGMXNT/shisa/wiki/A-Review-of-Public-Japanese-Training-Sets#analysis">公开日语训练集评论</a>: 通过在 GitHub 上创建账户为 AUGMXNT/shisa 的开发做出贡献。</li><li><a href="https://gist.github.com/cedrickchee/c3d9f8fed88f1c486b883153a64ee7dc">软件工程师的 LLM 微调</a>: 软件工程师的 LLM 微调。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://x.com/cedric_chee/status/1790638025397117031">来自 Cedric Chee (@cedric_chee) 的推文</a>: 何时以及为什么要微调 LLM：- 极其狭窄的问题 - Prompt engineering 不切实际 - 质量与延迟的权衡 - 数据隐私。模型微调万岁。</li><li><a href="https://huggingface.co/augmxnt/shisa-7b-v1">augmxnt/shisa-7b-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/augmxnt/shisa-gamma-7b-v1">augmxnt/shisa-gamma-7b-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://wandb.ai/wandb-japan/llm-leaderboard/reports/Nejumi-LLM-Leaderboard-Evaluating-Japanese-Language-Proficiency--Vmlldzo2MzU3NzIy)">Weights & Biases</a>: Weights & Biases，机器学习开发者工具。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1242129546334175393)** (37 条消息🔥): 

- **解锁 Modal 额度，获取解码权限**：成员们收到了关于获取 **Modal credits** 的指导，在 [modal.com](https://modal.com/signup) 注册后填写 [Modal 黑客松额度表单](https://bit.ly/modal-credits) 即可。额度共计 $500，有效期一年，此外免费层级（free tier）每月还有额外的 $30。

- **保持活跃以节省 Modal 成本**：一位成员分享了管理 **Modal 服务成本** 的技巧，通过设置 `container_idle_timeout` 来减少测试期间的费用。强调了在 LLM serving 等工作负载中谨慎使用 GPU 服务的重要性，并参考了 [GitHub 示例](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_mixtral.py)。

- **微调与模型服务技巧**：为了在 Modal 上进行有效的 **LLM 模型微调与服务**，开发时建议使用 `modal serve` 而非 `modal run`。为了获得优化后的结果，可以参考 [TensorRT-LLM 服务指南](https://modal.com/docs/examples/trtllm_llama) 并进行批处理（batch processing）。

- **更流畅的 Modal 部署体验**：成员们讨论了操作性问题，如正确设置 `container_idle_timeout` 以及避免重复加载模型。通过社区见解和相关 GitHub 项目链接，明确了 `modal serve` 与 `modal deploy` 的正确用法。

- **加入 Modal Slack 获取更快速的支持**：成员们被引导至 [Modal Slack](https://modal.com/slack) 以获取工程团队的专业支持。鼓励在 general 或 LLMS 频道提问，以便获得更快速、全天候的响应。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/vllm_mixtral.py">modal-labs/modal-examples 中的 vllm_mixtral.py</a>：使用 Modal 构建的程序示例。可以通过在 GitHub 上创建账号为 modal-labs/modal-examples 做出贡献。</li><li><a href="https://bit.ly/modal-credits.">Modal 黑客松额度</a>：要领取您的 Modal 额度，请先在 https://modal.com/ 注册账号。然后，通过此表单告知我们您的用户名。如需支持，请加入 Modal Slack。这里有一些示例可以帮助您开始...</li><li><a href="https://github.com/modal-labs/llm-finetuning/">GitHub - modal-labs/llm-finetuning</a>：微调 Llama/Mistral/CodeLlama 等模型的指南。</li><li><a href="https://github.com/satish860/PDF-Extraction-API/blob/main/app.py#L58">satish860/PDF-Extraction-API 中的 app.py</a>：一个基于 Marker 库的 API，用于执行 Marker 响应。</li><li><a href="https://modal.com/docs/examples/trtllm_llama">Serverless TensorRT-LLM (LLaMA 3 8B)</a>：在此示例中，我们演示了如何使用 TensorRT-LLM 框架在单张 NVIDIA A100 40GB GPU 上以每秒约 4,500 个输出 Token 的总吞吐量提供 Meta 的 LLaMA 3 8B 模型服务。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl.git">GitHub - OpenAccess-AI-Collective/axolotl</a>：尽管提问（Axolotl）。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 做出贡献。</li><li><a href="https://modal.com/settings/YOURUSERNAME/usage">登录</a>：欢迎回到 Modal！通过下方选择身份提供商登录您的 Modal 账号。
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis](https://discord.com/channels/1238365980128706560/1241117895740625099/1242545274329763912)** (3 条消息): 

- **用户对运行 Axolotl 感兴趣**：一位用户表达了对运行 Axolotl 的兴趣，并特别请求某位成员的关注。他们分享了相关讨论的直接链接。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1242350581474136136)** (10 条消息🔥): 

- **积分（Credits）将很快处理**：关于积分发放的更新将很快提供。对社区的耐心表示感谢。

- **确认 Axolotl 模型搜索问题**：用户观察到在 [HuggingFace](https://huggingface.co/models?other=axolotl) 上可以过滤但无法搜索 axolotl 模型。解释称搜索栏使用预定义标签以避免混淆，并讨论了改进 UI 以更好处理额外标签的可能性。

- **通过代码过滤 axolotl 模型的替代方法**：一位用户分享了使用 Hugging Face API 过滤所有 axolotl 模型的代码片段：
  ``` 
  from huggingface_hub import HfApi
  hf_api = HfApi()
  models = hf_api.list_models(filter="axolotl")
  ```

- **对混合分片策略（hybrid sharding strategy）的正面反馈**：一位成员对专注于 HYBRID_SHARD 策略的精力和努力表示热忱，该策略涉及使用 Fully Sharded Data Parallel (FSDP) 和 DeepSpeed (DS) 技术对模型进行分片。

**提到的链接**：<a href="https://huggingface.co/models?other=axolotl)">Models - Hugging Face</a>：未找到描述

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1242342595267395594)** (4 条消息): 

- **额度发放咨询引起困惑**：一位成员表示他们使用电子邮件地址注册但尚未收到积分。作为回应，另一位成员保证积分问题将很快得到解决，并感谢大家的耐心。

- **澄清 Replicate 的使用场景**：一位成员询问了 Replicate 的主要使用场景，质疑其是否旨在为公司或个人提供下游任务的 API 端点。他们还提到了 Fine-tuning 和自定义数据集等特定功能。

- **注册不匹配是一个常见问题**：另一位成员指出，他们的情况反映了另一位用户关于 Replicate 和会议之间注册方式不同的问题。这突显了用户注册方式一致性方面的反复出现的担忧。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1242284773599215687)** (5 条消息): 

- **新成员加入课程**：两名新成员宣布参加课程。一位用户提到注册后未收到 LangSmith 积分。

- **关于免费额度的咨询**：一位成员询问是否需要设置账单才能在现有的 250 个积分基础上再获得 250 个免费积分。另一位成员安慰说积分分配将很快处理，并会提供更新。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1242485573386637455)** (613 条消息🔥🔥🔥): 

- **关于 Discord Stages 和 Zoom 聊天集成的讨论**：成员们讨论了使用 Discord stages 的优缺点。一位参与者指出 stages “仅限音频”，另一位参与者确认了这一点，并建议将其作为语音/视频/屏幕共享频道。
  
- **新课程结构说明**：Hamelm 概述了课程的三种会议类型：Fine-Tuning 工作坊、深入问答的 Office Hours 以及会议演讲（Conference Talks）。日历邀请标题已更新以明确会议类型。

- **Fine-tuning 技术讨论**：关于 Llama3 模型问题、超参数重要性和多语言能力的深入对话。参与者提到了具体的挑战并分享了资源，如 [Stanford's Pyvene](https://github.com/stanfordnlp/pyvene/issues/46)。

- **分享的资源和技巧**：分享了大量链接、博客文章和论文，用于进一步阅读和资源汇总，例如 [Practical Tips for Finetuning LLMs](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) 和 [Axolotl's GitHub](https://github.com/OpenAccess-AI-Collective/axolotl)。

- **Apple Silicon 用于 Fine-tuning 的问题**：用户讨论了在 Apple M1 上使用 Axolotl 的困难，原因是 bitsandbytes 不支持该架构。提供了使用 Docker 或 mlx 等建议作为潜在的变通方案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://arxiv.org/abs/2405.09673">LoRA Learns Less and Forgets Less</a>: 低秩自适应 (LoRA) 是一种广泛使用的针对大语言模型的高效参数微调方法。LoRA 通过仅对选定的权重矩阵训练低秩扰动来节省内存。在...</li><li><a href="https://arxiv.org/abs/2305.11206">LIMA: Less Is More for Alignment</a>: 大语言模型的训练分为两个阶段：(1) 从原始文本进行无监督预训练，以学习通用表示；(2) 大规模指令微调和强化学习...</li><li><a href="https://www.malwarebytes.com/blog/news/2024/04/billions-of-scraped-discord-messages-up-for-sale">Billions of scraped Discord messages up for sale | Malwarebytes</a>: 一个网络抓取平台正在提供对一个包含超过 40 亿条 Discord 消息和合并用户配置文件的数据库的访问权限。</li><li><a href="https://huggingface.co/docs/peft/main/en/conceptual_guides/lora">LoRA</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/GAIR/lima">GAIR/lima · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/bhutanisanyam1/status/1758159687051350189">Tweet from Sanyam Bhutani (@bhutanisanyam1)</a>: LLM 微调基准测试！🙏 非常激动终于发布了这份比较不同 GPU 和精度的报告：- 首先，为什么要这样做以及它是什么？- 虽然有很多 GPU 基准测试，但很少有专门针对...</li><li><a href="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms">Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)</a>: 我从数百次实验中学到的经验</li><li><a href="https://huggingface.co/parlance-labs/hc-mistral-alpaca/tree/main/data">parlance-labs/hc-mistral-alpaca at main</a>: 未找到描述</li><li><a href="https://hamel.dev/blog/posts/evals/">- Your AI Product Needs Evals</a>: 如何构建特定领域的 LLM 评估系统。</li><li><a href="https://x.com/danielhanchen/status/1789659394302718373">Tweet from Daniel Han (@danielhanchen)</a>: 正在修复 LLM 微调 bug 并发现了 4 个问题：1. Mistral: HF 的 batch_decode 输出错误 2. Llama-3: 注意双重 BOS 3. Gemma: 第二个 token 有一个额外的空格 - GGUF(_Below) = 3064...</li><li><a href="https://huggingface.co/spaces/muellerzr/llm-conf">LLM Conf talk - a Hugging Face Space by muellerzr</a>: 未找到描述</li><li><a href="https://huggingface.co/parlance-labs/hc-mistral-alpaca/tree/main/configs">parlance-labs/hc-mistral-alpaca at main</a>: 未找到描述</li><li><a href="https://huggingface.co/parlance-labs/hc-mistral-alpaca">parlance-labs/hc-mistral-alpaca · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/TheZachMueller/status/1696157965890339148">Tweet from Zach Mueller (@TheZachMueller)</a>: 很高兴宣布一个新的 @huggingface space，旨在帮助解决机器学习中最大的问题之一：{X} 模型占用多少 vRAM？最重要的是：当使用 `device_map=&#3...</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>: 未找到描述</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/pretraining.html">Axolotl - Pre-training</a>: 未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-small-8k-instruct">microsoft/Phi-3-small-8k-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer#packing-dataset--constantlengthdataset-">Supervised Fine-tuning Trainer</a>: 未找到描述</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/config.html">Axolotl - Config options</a>: 未找到描述</li><li><a href="https://outlines-dev.github.io/outlines/">Outlines</a>: 使用 LLM 进行结构化文本生成</li><li><a href="https://en.wiktionary.org/wiki/OTTOMH">OTTOMH - Wiktionary, the free dictionary</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=eGVDKegRdgM">Scaling Up “Vibe Checks” for LLMs - Shreya Shankar | Stanford MLSys #97</a>: Stanford MLSys 研讨会系列第 97 集！为 LLM 扩展“感官检查 (Vibe Checks)”。演讲者：Shreya Shankar。简介：Shreya Shankar 是计算机科学专业的博士生...</li><li><a href="https://www.honeycomb.io/blog/introducing-query-assistant">Observability, Meet Natural Language Querying with Query Assistant </a>: 发布 Query Assistant，这是 AI 首次引入 Honeycomb。通过 Query Assistant，你可以用简单的英语描述/提问。</li><li><a href="https://huggingface.co/collections/leonardlin/multilingual-6594d0ea075245eadd6aa99c">multilingual - a leonardlin Collection</a>: 未找到描述</li><li><a href="https://x.com/HamelHusain/status/1784769559364608222">Tweet from Hamel Husain (@HamelHusain)</a>: Llama 3 70b 的 function calling 仅通过 prompting 即可开箱即用地良好运行</li>

<li>🚀💰 查看下方演示（提示词和代码在下一条推文中）</li><li><a href="https://github.com/TimDettmers/bitsandbytes/blob/main/CHANGELOG.md">bitsandbytes/CHANGELOG.md at main · TimDettmers/bitsandbytes</a>：通过针对 PyTorch 的 k-bit 量化实现易于使用的 LLM。 - TimDettmers/bitsandbytes</li><li><a href="https://github.com/stanfordnlp/pyvene/issues/46">[P1] 支持更多 Hugging Face (基于 Transformer) 的模型 · Issue #46 · stanfordnlp/pyvene</a>：描述：理想情况下，此库应支持此处列出的所有模型，而无需向用户暴露模型细节。这需要我们为所有模型设置模型文件夹...</li><li><a href="https://github.com/argilla-io/distilabel/blob/main/examples/structured_generation_with_outlines.py">distilabel/examples/structured_generation_with_outlines.py at main · argilla-io/distilabel</a>：⚗️ distilabel 是一个为需要高质量输出、完整数据所有权和整体效率的 AI 工程师提供的合成数据和 AI 反馈框架。 - argilla-io/distilabel</li><li><a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">安装 NVIDIA Container Toolkit &mdash; NVIDIA Container Toolkit 1.15.0 文档</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=jkrNMKz9pWU">黑客的语言模型指南</a>：在这段信息量巨大的视频中，fast.ai 联合创始人、所有现代语言模型 (LMs) 所基于的 ULMFiT 方法的创造者 Jeremy Howard...</li><li><a href="https://huggingface.co/models?other=axolotl">Models - Hugging Face</a>：未找到描述</li><li><a href="https://poe.com/s/c0BFLNhTwiyPXOulPCnO">你有一列，每个元素包含一个元组列表。获取每个元组出现的频率</a>：TrinoAgentEx：你想了解哪个 SQL 关键字？TrinoAgentEx：要在单个 Trino SQL 查询中查询列表中元组的频率分布，你需要执行几个操作...</li><li><a href="https://discord.gg/2YkbgY5TQj">加入 Axolotl AI Discord 服务器！</a>：在 Discord 上查看 Axolotl AI 社区 - 与 2197 名其他成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://x.com/abacaj/status/1782835550396850449">来自 anton (@abacaj) 的推文</a>：Phi-3 看起来相当不错，肯定比 phi-2 有所改进。128k 的长上下文对于提取信息和文档处理非常有用，考虑到该模型非常小，它可以被部署在...</li><li><a href="https://lake-scilla-bc6.notion.site/LLM-fine-tuning-workshop-6832ed2266a14957831ed8e2b3a959b3">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间</li><li><a href="https://github.com/ml-explore/mlx">GitHub - ml-explore/mlx: MLX：适用于 Apple Silicon 的数组框架</a>：MLX：适用于 Apple Silicon 的数组框架。通过在 GitHub 上创建账号来为 ml-explore/mlx 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/949">启用 eval_table_size 时评估耗时大幅增加 · Issue #949 · OpenAccess-AI-Collective/axolotl</a>：请检查此问题之前是否已报告过。我搜索了之前的 Bug 报告，没有发现类似的报告。预期行为：评估时间预计会增加，但不应...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb">axolotl/examples/colab-notebooks/colab-axolotl-example.ipynb at main · OpenAccess-AI-Collective/axolotl</a>：尽管提出 Axolotl 问题。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/parlance-labs/ftcourse">GitHub - parlance-labs/ftcourse</a>：通过在 GitHub 上创建账号来为 parlance-labs/ftcourse 的开发做出贡献。</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: 结构化文本生成</a>：结构化文本生成。通过在 GitHub 上创建账号来为 outlines-dev/outlines 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docker/Dockerfile-cloud#L8">axolotl/docker/Dockerfile-cloud at main · OpenAccess-AI-Collective/axolotl</a>：尽管提出 Axolotl 问题。通过在 GitHub 上创建账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html">Axolotl - 无模板提示词构建</a>：未找到描述</li><li><a href="https://docs.google.com/presentation/d/1MC8JqXf9SU9fEYh6RhXPzF8LjAjpmdrmUMTWnPpi79Y/edit?usp=sharing">前沿技巧</a>：SANYAM BHUTANI，H2O.ai 高级数据科学家</li><li><a href="https://lightning.ai/pages/community/l">lightning.ai/pages/community/l</a>

ora-insights/">使用 LoRA 和 QLoRA 微调 LLM：来自数百次实验的见解 - Lightning AI</a>：LoRA 是训练自定义 LLM 最广泛使用的参数高效微调技术之一。从使用 QLoRA 节省内存到选择最佳 LoRA 设置，本文提供了实用的...</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Chat Models 模板</a>：未找到描述</li><li><a href="https://github.com/parlance-labs/ftcourse/tree/master">GitHub - parlance-labs/ftcourse</a>：通过在 GitHub 上创建账号，为 parlance-labs/ftcourse 的开发做出贡献。</li><li><a href="https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2">微调 LLM：LLAMA-2 的深入分析 | Anyscale</a>：在本博客中，我们比较了全参数微调与 LoRA，并回答了关于这两种技术优缺点的相关问题。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples/llama-3">axolotl/examples/llama-3 (main 分支) · OpenAccess-AI-Collective/axolotl</a>：尽管提问（axolotl questions）。通过在 GitHub 上创建账号，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/parlance-labs/ftcourse/tree/master/sample_data">ftcourse/sample_data (master 分支) · parlance-labs/ftcourse</a>：通过在 GitHub 上创建账号，为 parlance-labs/ftcourse 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=PXWYUTMt-AU">LoRA：大语言模型的低秩自适应 - 视觉化解释 + 从零开始的 PyTorch 代码</a>：LoRA 的完整视觉化解释，包含从零开始的 PyTorch 代码！完整代码和幻灯片可在我的 GitHub 上获取：https://github.com/hkproj/pytorch-loraChap...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1436">错误：未找到适用于 macOS 的 bitsandbytes==0.43.0 匹配发行版 · Issue #1436 · OpenAccess-AI-Collective/axolotl</a>：请检查此问题之前是否已被报告。我搜索了之前的 Bug 报告，未发现类似报告。预期行为：命令 pip3 install -e '.[flash-attn,deeps...</li><li><a href="https://buttondown.email/ainews">AI News</a>：我们汇总顶尖的 AI Discord + AI Reddit + AI X/Twitter，并每天为您发送综述！查看存档以获取示例。“这是我每天花费的最具杠杆作用的 45 分钟” - Soumith “最好的 AI 新闻...”</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1589">使用 FSDP 进行 8-Bit DoRA 训练无效，但 4-bit QDoRA 有效 / peft_use_dora 被忽略了？ · Issue #1589 · OpenAccess-AI-Collective/axolotl</a>：请检查此问题之前是否已被报告。我搜索了之前的 Bug 报告，未发现类似报告。预期行为：在启用 8-bit LoRA 且 peft_use_dora: true 的情况下，th...</li><li><a href="https://x.com/sroecker/status/1757103619705299061?t=uajfu81xkUp7x80xgQ7i1A&s=19">来自 Steffen Röcker (@sroecker) 的推文</a>：有没有想过如何使用 @axolotl_ai 和 @Podman_io 微调 LLM？按照 NVIDIA toolkit CDI 的说明操作，只需运行 "podman run --rm --device http://nvidia.com/gpu=all --security-...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl：尽管提问</a>：尽管提问。通过在 GitHub 上创建账号，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/huggingface/peft/pull/1724">修复：在使用 BNB 时允许在 CPU 上进行 DoRA 初始化，作者 BenjaminBossan · Pull Request #1724 · huggingface/peft</a>：解决 #1674。对于某些用户，即使使用最终需要 GPU 的 BitsAndBytes，也有必要在 CPU 上初始化模型。由于 DoRA 需要在初始化时对 BNB 权重进行反量化...</li><li><a href="https://lu.ma/terrible-ai-systems?utm_source=llm">如何与 Jason Liu 一起构建糟糕的 AI 系统 · Luma</a>：Jason 是一位独立顾问，他利用自己在推荐系统方面的专业知识，帮助快速成长的初创公司构建其 RAG 应用程序。他曾是...</li><li><a href="https://nbsanity.com/static/d06085f1dacae8c9de9402f2d7428de2/demo.html">Llama-3 Function Calling 演示</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1791900967472140583">来自 Daniel Han (@danielhanchen) 的推文</a>：我对“LoRA 学得更少，遗忘也更少”的看法：1) "MLP/All" 不包括 gate_proj。训练了 QKVO、up 和 down，但没有 gate（第 3 页脚注） 2) 为什么 LoRA 在数学和...方面表现良好...</li><li><a href="https://www.guardrailsai.com/">Guardrails AI</a>：为 LLM 应用程序强制执行保障。
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1242526610024824942)** (3 messages): 

```html
- **Jason 的 W&B 课程大获好评**：一位用户表达了对 Jason 课程的兴奋，并提到已经完成了一半的 **Weights & Biases (W&B) 课程**。他们使用了老师表情符号来表达敬意。
- **对 Prompt engineering 的好奇心达到顶峰**：另一位用户询问了 Jason 处理 Prompt engineering 的系统化方法，称赞他在优化 Prompt 方面的广泛工作。他们渴望在研讨会期间学习他的“秘诀”。
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1242489403129987194)** (2 messages): 

- **Gradio 维护者自我介绍**：Freddy 是 **Gradio**（一个用于为 AI 模型开发用户界面的 Python 库）的维护者，他邀请成员们提问并分享 Demo。他提供了 [Gradio 快速入门指南](https://www.gradio.app/guides/quickstart) 的链接，以及另一份关于如何 [用 5 行代码构建聊天机器人](https://www.gradio.app/guides/creating-a-chatbot-fast) 的指南。
- **成员对 Gradio 表现出兴趣**：一位成员对分享的资源表示感谢，并提到他们最终会有一些问题，特别是与他们之前开发过且觉得具有挑战性的 **A1111-extension** 相关的问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.gradio.app/guides/quickstart">Quickstart</a>：Gradio 分步教程</li><li><a href="https://www.gradio.app/guides/creating-a-chatbot-fast">Creating A Chatbot Fast</a>：Gradio 分步教程
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[askolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1242543726312689705)** (13 messages🔥): 

- **macOS 上的 bitsandbytes 问题**：[这个 GitHub 线程](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1436) 讨论了在 macOS 上安装 bitsandbytes 的相关问题。具体错误为 *"No matching distribution found for bitsandbytes==0.43.0 for macOS"*。
- **尚不支持 MLX**：一位成员指出 Axolotl 目前还不支持 MLX，并引用了 [GitHub 上的一个未解决问题](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1119)。MLX 因其在消费级硬件上高效微调大语言模型的能力而受到称赞。
- **微调对比：OpenAI vs Axolotl**：一位用户分享了他们使用 OpenAI 进行微调的经验，称大约需要 30 分钟并按 token 收费。他们询问 Axolotl 在微调的时间和成本方面表现如何。
- **Apple M1 不适合微调**：有观点指出 Apple ARM (M1) 不支持 q4 和 q8，因此不太适合微调。建议用户在 RunPod 上租用 Linux GPU 服务器。
- **MLX-examples 指南**：对于那些有兴趣使用 MLX 的用户，参考了 GitHub 上的 [MLX examples 文档](https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md) 以获取进一步指导。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md">mlx-examples/lora/README.md at main · ml-explore/mlx-examples</a>：MLX 框架中的示例。通过在 GitHub 上创建账号为 ml-explore/mlx-examples 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1436">ERROR: No matching distribution found for bitsandbytes==0.43.0 for macOS · Issue #1436 · OpenAccess-AI-Collective/axolotl</a>：请检查此问题之前是否已被报告。我搜索了之前的 Bug 报告，没有发现类似的报告。预期行为：执行命令 pip3 install -e &#39;.[flash-attn,deeps...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/issues/1119">MLX Support · Issue #1119 · OpenAccess-AI-Collective/axolotl</a>：你好，如果 Axolotl 能支持 MLX 就太棒了。MLX 已被证明能够快速高效地在消费级硬件上微调许多 LLM，包括 7B LLM。谢谢！（编辑：更新）
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1242565467562967152)** (1 条消息): 

- **使用 Accelerate 加速你的 PyTorch**：成员分享了一个 [Hugging Face Spaces 上的演示文稿](https://huggingface.co/spaces/muellerzr/llm-conf)，介绍了 Accelerate。这是一个简化在任何分布式配置上运行 PyTorch 代码的库。链接中的 [Accelerate 文档](https://huggingface.co/docs/accelerate)展示了如何仅用几行代码来实现它。

- **Accelerate 功能快速入门**：[Hugging Face 上的快速入门指南](https://huggingface.co/docs/accelerate/quicktour)阐述了 Accelerate 的功能，包括用于分布式训练脚本的统一命令行界面、PyTorch 训练库，以及针对大型模型的 Big Model Inference 支持。

- **入门示例**：在 [Hugging Face 的 GitHub](https://github.com/huggingface/accelerate/tree/main/examples) 上可以找到一系列示例，建议从 `nlp_example` 开始。这些示例展示了 Accelerate 在处理各种分布式训练设置中的多功能性。

- **深度模型显存估算器**：成员们分享了 [显存使用估算器](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) 和 [TransformerAnalyzer 工具](https://huggingface.co/spaces/cllatMTK/TransformerAnalyzer) 的链接，后者提供详细的 FLOPS 和其他参数估计，有助于理解模型需求。

- **高效运行大语言模型**：在 Hugging Face 上讨论的 [Can I Run it LLM Edition](https://huggingface.co/spaces/Vokturz/can-it-run-llm) Space 专注于推理能力，强调了 LoRa 在高效大语言模型部署中的适用性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/muellerzr/llm-conf">LLM Conf talk - 由 muellerzr 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/docs/accelerate">Accelerate</a>：未找到描述</li><li><a href="https://huggingface.co/docs/accelerate/quicktour">Quicktour</a>：未找到描述</li><li><a href="https://github.com/huggingface/accelerate/tree/main/examples">accelerate/examples at main · huggingface/accelerate</a>：🚀 一种在几乎任何设备和分布式配置上启动、训练和使用 PyTorch 模型的简单方法，支持自动混合精度（包括 fp8），以及易于配置的 FSDP 和 DeepSpeed 支持……</li><li><a href="https://huggingface.co/spaces/hf-accelerate/model-memory-usage">Model Memory Utility - 由 hf-accelerate 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Vokturz/can-it-run-llm">Can You Run It? LLM version - 由 Vokturz 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/cllatMTK/TransformerAnalyzer">TransformerAnalyzer - 由 cllatMTK 创建的 Hugging Face Space</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1242522009758470174)** (1 条消息): 

- **Perplexity AI 与 Tako 合作提供高级知识搜索**：*"我们正与 Tako 联手，为我们的用户带来高级知识搜索和可视化功能。"* 这允许用户在 Perplexity 内部搜索、对比并分享权威的知识卡片，最初在美国以英文提供，移动端访问即将推出。[阅读关于我们合作伙伴关系的信息](https://trytako.com/blog/introducing-tako-and-perplexity-integration)。

**提及的链接**：<a href="https://trytako.com/blog/introducing-tako-and-perplexity-integration">Tako</a>：未找到描述

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1242024539555106878)** (735 条消息🔥🔥🔥): 

```html
- **平台忠诚度引发讨论**：一位成员分享了使用 Perplexity 和 Gemini 的经验，强调用户具有“零忠诚度”，并称赞 Perplexity 提供的直接回答 ([Tenor GIF](https://tenor.com/view/oh-no-homer-simpsons-hide-disappear-gif-16799752))。
- **Perplexity 功能技巧分享**：讨论了使用 Perplexity 的各种功能，包括理解 API、调整 Firefox 等浏览器中的搜索引擎选项，以及处理 system prompts。
- **Perplexity 暂时宕机**：多位用户报告了 Perplexity 宕机的问题；他们对失去该服务表示同情，并推测是在进行维护和更新。
- **模型偏好与用途讨论**：成员们对比了 GPT-4o 和 Claude 3 Opus 等模型，讨论了它们在创意写作和 coding 等任务中的优势和偏好 ([Spectrum IEEE 文章](https://spectrum.ieee.org/perplexity-ai))。
- **Perplexity 的交互功能**：成员们对 Perplexity 的新功能（如 Tako 图表）感到好奇并分享了使用技巧，有人提到添加 `since:YYYY/01/01` 等提示词可以改善搜索结果。 
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://violentmonkey.github.io/">未找到标题</a>: 未找到描述</li><li><a href="https://apps.apple.com/us/app/elevenlabs-reader-ai-audio/id6479373050">‎ElevenLabs Reader: AI Audio</a>: ‎将文本转换为自然、富有表现力的语音。非常适合文章、ePubs、PDF 或任何文本。ElevenLabs Reader 将我们最强大的 Text to Speech (TTS) 模型带入您的口袋。应用功能：文本阅读器...</li><li><a href="https://docs.perplexity.ai/docs/perplexitybot">PerplexityBot</a>: 未找到描述</li><li><a href="https://spectrum.ieee.org/perplexity-ai">Perplexity.ai Turns Tables on Google, Upends SEO Credos</a>: AI 搜索领导者将 Meta 构建的智能与初创公司的拼搏精神相结合</li><li><a href="https://x.com/bobbyallyn/status/1792679435701014908?s=46">Bobby Allyn (@BobbyAllyn) 的推文</a>: Scarlett Johansson 关于 OpenAI 情况的声明。哇：</li><li><a href="https://greasyfork.org/en/scripts/490634-perplexity-model-selection">Perplexity Model Selection</a>: 使用 jQuery 为 Perplexity AI 添加模型选择按钮</li><li><a href="https://tenor.com/view/oh-no-homer-simpsons-hide-disappear-gif-16799752">Oh No Homer GIF - Oh No Homer Simpsons - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://blogs.microsoft.com/blog/2024/05/20/introducing-copilot-pcs/">Introducing Copilot+ PCs - Microsoft 官方博客</a>: 提供 5 月 20 日活动的按需录像。今天，在我们新的 Microsoft 园区举行的特别活动中，我们向世界介绍了一类专为 AI 设计的新型 Windows PC，即 Copilot+ PCs。...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1242134770088149032)** (9 条消息🔥): 

- **通过 Perplexity AI 回答历史问题**：一位成员分享了一个询问 *"Qui est Adolf?"* 的链接，其中包含详细的历史见解。[在此探索](https://www.perplexity.ai/search/Qui-est-adolf-TQqGm0aDRRWWqeblJpYUgg#5)。

- **理解数学中的理想结构**：发布了一个针对 *"Does every ideal?"* 问题的链接，深入探讨了复杂的数学理论。[在此探索](https://www.perplexity.ai/search/Does-every-ideal-hQP30OxPQjqQIg4cK.sFDA#0)。

- **通过 Perplexity 查询脚本创建**：一位用户分享了对 *"Create a script"* 的搜索，可能旨在生成特定的脚本或代码片段。[在此探索](https://www.perplexity.ai/search/Create-a-script-ZkKbE43aRhyXn3HIlXADUg)。

- **探索计算机技术概念**：一位成员在 Perplexity AI 搜索中询问了 *"what is layer?"*，涉及计算机或 Machine Learning 中的详细讨论。[在此探索](https://www.perplexity.ai/search/what-is-layer-xXVSIKHpT2uGOqogIZmOVw)。

- **室内话题讨论**：另一个名为 *"talk about indoor"* 的搜索表明其关注点在于室内环境或活动。[在此探索](https://www.perplexity.ai/search/talk-about-indoor-Wkghx1CeTwuZWH_gcxDpJw)。
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1242229643336224879)** (98 条消息🔥🔥): 

```html
- **在 Open WebUI 上使用 Perplexity API 的困扰**：一位用户报告了模型兼容性问题，指出：“它在 OpenAI (Closed) 和 Groq 上运行良好，但可能他们没有设置好与 PPLX 配合使用的模型名称。”另一位用户建议直接使用 `api.perplexity.ai`，但发现 Perplexity 没有 `/models` 端点，导致了进一步的复杂化。
- **代理服务器解决方案与执行协助**：有人提议了一个变通方案，即创建一个本地服务器来代理 models 和 chat completions 端点。一位用户提到已完成代理，并指导说：“你需要在 docker 命令中添加 `--network=host`”以解决 localhost 问题。
- **Docker 配置讨论**：用户讨论了 Docker 配置的复杂性，其中一人在排查连接问题时总结了正确的命令：“docker run -d --network=host -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main”，同时进行连接故障排查。
- **关于发送图片的咨询**：当被问及“是否有办法通过 API 发送图片？”时，得到的澄清是目前 Perplexity 的 API 仅支持文本，并表示“他们只是在使用 Claude 和 OpenAI vision api”，且支持图片的 LLAVA 模型无法通过 API 使用。
- **用户感谢与最终调整**：一位用户表达了感谢，说：“谢谢你，🙂”，而另一位用户确认他们需要调整 Docker 配置以确保 API 功能正常。这表明了为解决问题而进行的持续努力和协作。
```

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://localhost:8080`">未找到标题</a>: 未找到描述</li><li><a href="https://docs.openwebui.com/">🏡 首页 | Open WebUI</a>: Open WebUI 是一个可扩展、功能丰富且用户友好的自托管 WebUI，旨在完全离线运行。它支持各种 LLM 运行器，包括 Ollama 和 OpenAI 兼容的 API。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1242544374726791270)** (1 条消息): 

- **Phi-3 模型登场**：Microsoft 发布了 **Phi-3 small (7B)** 和 **Phi-3 medium (14B)** 模型，支持高达 128k 的上下文（context），在 MMLU 和 AGI Eval 上取得了令人印象深刻的分数。点击[这里](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)查看！

- **1000 万美元算力资源开放**：Hugging Face 宣布通过 ZeroGPU 承诺提供 **1000 万美元的免费 GPU 访问额度**，旨在帮助独立开发者和学术界 AI 构建者创建 AI demo。点击[这里](https://huggingface.co/zero-gpu-explorers)了解更多关于该计划的信息。

- **Transformers 4.41.0 带来大量新特性**：最新更新包括 **Phi3, JetMoE, PaliGemma, VideoLlava 和 Falcon 2**，并改进了对 **GGUF、水印（watermarking）以及 HQQ 和 EETQ 等新量化方法（quant methods）**的支持。完整的发布说明请见[这里](https://github.com/huggingface/transformers/releases/tag/v4.41.0)。

- **LangChain 集成简化**：全新的 **langchain-huggingface 软件包** 助力 Hugging Face 模型无缝集成到 LangChain 中。查看[公告和详情](https://huggingface.co/blog/langchain)。

- **CommonCanvas 和 Moondream 更新**：**CommonCanvas** 发布了首个基于 Creative Commons 图像训练的开源文本生成图像（text-to-image）模型，并在 Hugging Face 上提供了[最大的数据集](https://huggingface.co/common-canvas)。**Moondream** 现在可以通过 WebGPU 直接在浏览器中运行，提升了用户隐私。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/ClementDelangue/status/1791115403734778185)">clem 🤗 (@ClementDelangue) 的推文</a>: 不再是 GPU 贫困户：非常激动今天正式发布 ZeroGPU 的 beta 版本。祝贺 @victormustar 及其团队发布！在过去的几个月里，开源 AI 社区蓬勃发展。不...</li><li><a href="https://x.com/LysandreJik/status/1792923587340390733)">Lysandre (@LysandreJik) 的推文</a>: 从模型页面到本地应用只需几秒，@huggingface Hub 迎来 Local Apps！建议你最喜欢的利用 Hub 的本地应用添加到下拉列表中，并实现 ✨ 深度链接...</li><li><a href="https://x.com/osanseviero/status/1792904237153722569)">Omar Sanseviero (@osanseviero) 的推文</a>: Transformers 4.41.0 有很多好东西🤗 🥳 新模型：Phi3, JetMoE, PaliGemma, VideoLlava 和 Falcon 2。🤯 通过 from_pretrained 支持 GGUF 🤏 新量化方法：HQQ 和 EETQ 🔍 水印支持...</li><li><a href="https://x.com/_philschmid/status/1790419788931416466)">Philipp Schmid (@_philschmid) 的推文</a>: 我们很高兴宣布 huggingface-langchain🚀 一个新的开源包，可将来自 @huggingface 的最新开源模型无缝集成到 @LangChainAI，支持本地模型和托管模型！...</li><li><a href="https://x.com/multimodalart/status/1791201296357142663)">apolinario (multimodal.art) (@multimodalart) 的推文</a>: 非常激动 CommonCanvas 刚刚发布！🖼️ • 首个完全基于公开许可图像训练的开源文本生成图像模型（SD2 和 SDXL 架构） • 该数据集包含约 70M 张公开许可...</li><li><a href="https://x.com/xenovacom/status/1791436796498174047)">Xenova (@xenovacom) 的推文</a>: Moondream，你最喜欢的由 @vikhyatk 开发的小型视觉语言模型，现在可以直接在浏览器上通过 WebGPU 运行！🤯 当然，这是由 Transformers.js 和 ONNX Runtime Web 驱动的！🤗 本地推理意味着...</li><li><a href="https://x.com/xenovacom/status/1792570966272336074)">Xenova (@xenovacom) 的推文</a>: 你现在可以将 🤗 Transformers.js 与 Google Visual Blocks 结合使用，这是一个可视化编程框架，让你可以在无代码图形编辑器中创建机器学习流水线！🛠️ 快速工作流原型设计...</li><li><a href="https://x.com/IlysMoutawwakil/status/1791406503112704455)">Ilyas Moutawwakil (@IlysMoutawwakil) 的推文</a>: Optimum-Benchmark 登陆 PyPI 🎉 但为什么是现在？🤔 因为它正被集成到 Transformers 的基准测试工作流中 😍 你最喜欢的 transformers 将变得更快、更轻量；感谢 @...</li><li><a href="https://x.com/osanseviero/status/1791567896482635801)">Omar Sanseviero (@osanseviero) 的推文</a>: 对 LLM 感兴趣？加入这个由顶尖专家授课的微调（Fine-Tuning）课程！🚀 @huggingface 为 Space demo、微调、推理等提供 $501.42 的 GPU 额度！尽情享受 🤗 https://maven.co...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1242025543457833002)** (678 条消息🔥🔥🔥): 

- **语音模型大对决**：一位用户分享了两个著名的文本转语音（text-to-speech）模型的链接，分别是 [Hugging Face 上的 Suno bark](https://huggingface.co/suno/bark) 和付费服务 [Eleven Labs](https://elevenlabs.io/)，并询问了 [Udio](https://www.udio.com) 所使用的底层模型。
- **Git LFS 上传问题**：多位用户讨论了关于使用 git LFS 向 Hugging Face 仓库上传大文件时的故障排除问题。建议包括使用 `huggingface_hub` 库中的 `upload_file` 函数。
- **语言模型规格**：讨论围绕着最大的语言模型展开，提到了 [GPT-4](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) 和 Google 的 1.5 万亿参数模型，并探讨了如何优化 Falcon-180B 和 Llama 模型。
- **Hugging Face 商店期待**：用户对 Hugging Face 周边商店的重新开业表达了兴奋和期待，凸显了社区对官方周边（swag）的强烈渴望。
- **求职申请成功**：社区成员向申请了 Hugging Face 职位的成员表示祝贺和祝福，体现了社区的支持与鼓励。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pypi.org/project/ratelimiter/">ratelimiter</a>: 简单的 Python 速率限制对象</li><li><a href="https://huggingface.co/spaces/parler-tts/parler-tts-expresso">Parler TTS Expresso - a Hugging Face Space by parler-tts</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/parler-tts/parler_tts_mini">Parler-TTS Mini - a Hugging Face Space by parler-tts</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/deep-rl-course/unit1/hands-on#install-dependencies-and-create-a-virtual-screen-">Train your first Deep Reinforcement Learning Agent 🤖 - Hugging Face Deep RL Course</a>: 训练你的第一个深度强化学习 Agent 🤖 - Hugging Face 深度强化学习课程</li><li><a href="https://x.com/kuldeep_s_s/status/1792296168111628717">Tweet from Kuldeep Singh Sidhu (@kuldeep_s_s)</a>: 你很高兴 @Meta 开源了 Llama 3 😃... 所以你冲向 HuggingFace Hub 下载闪亮的 Llama 3 模型，结果却看到了无数个 Llama 3！🦙✨ 你该用哪一个...</li><li><a href="https://huggingface.co/mlabonne/Meta-Llama-3-225B-Instruct">mlabonne/Meta-Llama-3-225B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/gemma-peft">Fine-Tuning Gemma Models in Hugging Face</a>: 在 Hugging Face 中微调 Gemma 模型</li><li><a href="https://huggingface.co/google/switch-c-2048">google/switch-c-2048 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/kanye-ye-kanye-west-ty-dolla-vultures-gif-3313542573422740922">Kanye Kanye West GIF - Kanye Ye Kanye west - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="http://hf.co/papers">Daily Papers - Hugging Face</a>: 每日论文 - Hugging Face</li><li><a href="https://tenor.com/p9BpiQov0bB.gif">Skibidi Toilet GIF - Skibidi toilet Skibidi Toilet - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/minor-spelling-mistake-gif-21179057">Minor Spelling Mistake GIF - Minor Spelling Mistake - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/cowboy-hug-brokeback-mountain-couple-gay-gif-5066019591388392130">Cowboy Hug GIF - Cowboy Hug Brokeback Mountain - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot/blob/main/app.py#:~:text=if%20reaction.message.author.id%20!%3D%20user.id%3A%20%23%20can%27t%20earn%20while%20self%2Dreacting%2C%20which%20is%20abuseable)">app.py · huggingface-projects/LevelBot at main</a>: 未找到描述</li><li><a href="https://tenor.com/bjKth.gif">Idk Shrug GIF - Idk Shrug Meme - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/datasets/H-D-T/Buzz">H-D-T/Buzz · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://elevenlabs.io/">Text to Speech &amp; AI Voice Generator</a>: 使用有史以来最强大的在线 AI 文本转语音 (TTS) 软件，免费创建任何风格和语言的高级 AI 语音。在几分钟内生成文本转语音配音...</li><li><a href="https://www.udio.com/">Udio | AI Music Generator - Official Website</a>: 发现、创作并与世界分享音乐。利用最新技术在几秒钟内创作 AI 音乐。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1242134561761267755)** (2 条消息): 

- **致力于为 Transformers 集成 ImageBind**：一位成员提到，*"正在努力将 ImageBind 添加到 `transformers` 中。"* 虽然细节较少，但这表明正在持续努力增强 Transformers 库的功能。
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1242139476772388884)** (13 条消息🔥): 

- **Merve 展示 PaliGemma 的文档模型**：*"引用 merve (@mervenoyann) 的话：有人问我关于 PaliGemma 的文档理解能力..."*。更多详情请参阅该 [推文](https://x.com/giffmana/status/1791541209883717973?s=46)。
  
- **DeepSpeech 咨询**：一位成员询问，*"这里有人用过 Mozilla 的 DeepSpeech 吗？"*，引发了对 Mozilla DeepSpeech 项目的关注。

- **LangChain 到 LangGraph 的迁移指南**：通过一门 [文章](https://medium.com/ai-advances/upgrading-your-agents-a-smooth-transition-from-legacy-langchain-to-langgraph-c552cb60fcb3) 分享了关于从旧版 LangChain 升级到 LangGraph 的深度指南。

- **在 Magnolia CMS 中利用 LLM**：一位成员通过 [这篇 Medium 文章](https://joaquin-alfaro.medium.com/openai-as-writing-assistant-in-magnolia-cms-7052a4715201) 分享了在 Magnolia CMS 中使用 LLM 进行内容创作的见解。

- **精选 3D Gaussian Splatting 资源**：这个 [GitHub 仓库](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#editing) 重点介绍了一份 3D Gaussian Splatting 论文和资源的综合列表，在机器人和 Embodied AI 领域具有巨大潜力。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/giffmana/status/1791541209883717973?s=46">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>：Merve 表现非常出色：引用 merve (@mervenoyann) 的话，有人问我关于 PaliGemma 的文档理解能力，所以我构建了一个包含所有 PaliGemma 微调文档模型的 Space...</li><li><a href="https://huggingface.co/papers/2301.13276">论文页面 - Distributed Swarm Intelligence</a>：未找到描述</li><li><a href="https://huggingface.co/docs/evaluate/base_evaluator#evaluate-models-on-the-hub">使用 `evaluator`</a>：未找到描述</li><li><a href="https://github.com/anthonyrussano/wikitweet/blob/main/tweet-natural-healing-thread.py">wikitweet/tweet-natural-healing-thread.py at main · anthonyrussano/wikitweet</a>：通过在 GitHub 上创建账号来为 anthonyrussano/wikitweet 的开发做出贡献。</li><li><a href="https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#editing">GitHub - MrNeRF/awesome-3D-gaussian-splatting：专注于 3D Gaussian Splatting 的论文和资源精选列表，旨在紧跟未来几个月预期涌现的研究浪潮。</a>：专注于 3D Gaussian Splatting 的论文和资源精选列表，旨在紧跟未来几个月预期涌现的研究浪潮。 - MrNeRF/awesome-3D-gaussian-splatting
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1242054556188151858)** (15 条消息🔥): 

- **发布 Sdxl Flash Mini**：一名成员宣布与 [Project Fluently](https://hf.co/fluently) 合作发布 **SDXL Flash Mini**。该模型被描述为快速且高效，在保持可观质量的同时减少了资源消耗 [SDXL Flash Mini](https://huggingface.co/sd-community/sdxl-flash-mini)。

- **KingNish 展示 SDXL Flash Demo**：KingNish 在 Hugging Face Spaces 上展示了 **SDXL Flash** 的精彩新 Demo。这为其实际能力提供了直观展示 [SDXL Flash Demo](https://huggingface.co/spaces/KingNish/SDXL-Flash)。

- **Tokun Tokenizer 发布**：受 Andrej Karpathy 启发，一名成员开发了一种名为 **Tokun** 的新 Tokenizer，旨在显著减小模型体积并增强能力。分享了 [GitHub 项目](https://github.com/apehex/tokun)和[关于测试的文章](https://x.com/4pe0x/status/1792638900059385942)。

- **Transformers 库贡献**：一名成员庆祝他们的 PR 被合并到 **Transformers 库**中，该 PR 修复了微调 AI 模型和自定义 Pipeline 的问题。点击[此处](https://github.com/huggingface/transformers/pull/29004)查看 PR 链接。

- **使用 ZeroGPU 的 llama-cpp-agent**：成员分享了在 Hugging Face Spaces 上利用 ZeroGPU 技术创建的 **llama-cpp-agent**，这标志着计算效率方面的进步 [llama-cpp-agent](https://huggingface.co/spaces/pabloce/llama-cpp-agent)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://swiftapi.pro/">Swift API</a>：未找到描述</li><li><a href="https://x.com/4pe0x/status/1792638900059385942">来自 Apehex (@4pe0x) 的推文</a>：很高兴介绍 `tokun`，一个为 #LLM 带来变革的 #tokenizer。它可以将 #llama3 的体积缩小 10 倍，同时提升能力！https://github.com/apehex/tokun/blob/main/arti...</li><li><a href="https://huggingface.co/spaces/KingNish/SDXL-Flash">SDXL Flash - KingNish 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/sd-community/sdxl-flash-mini">sd-community/sdxl-flash-mini · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/pabloce/llama-cpp-agent">Llama Cpp Agent - pabloce 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/formentor-studio/magnolia-ai-contents">GitHub - formentor-studio/magnolia-ai-contents: 使用 AI 在 Magnolia CMS 中生成内容</a>：使用 AI 在 Magnolia CMS 中生成内容。欢迎通过 GitHub 账号为 formentor-studio/magnolia-ai-contents 的开发做出贡献。</li><li><a href="https://github.com/apehex/tokun">GitHub - apehex/tokun: tokun to can tokens</a>：tokun to can tokens。欢迎通过 GitHub 账号为 apehex/tokun 的开发做出贡献。</li><li><a href="https://github.com/wikip-co/wikip.co">GitHub - wikip-co/wikip.co: 使用 node.js 构建的静态 wiki</a>：使用 node.js 构建的静态 wiki。欢迎通过 GitHub 账号为 wikip-co/wikip.co 的开发做出贡献。</li><li><a href="https://github.com/branyang02/notie">GitHub - branyang02/notie: 个人 Markdown 笔记应用。</a>：个人 Markdown 笔记应用。欢迎通过 GitHub 账号为 branyang02/notie 的开发做出贡献。</li><li><a href="https://notie-nine.vercel.app/">Notie</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1242324285541056522)** (4 条消息): 

- **LLM 在故事增强方面表现不佳**：一位成员发现，使用 **llama3 8b 4bit** 来实现《Creating Suspenseful Stories: Iterative Planning with Large Language Models》中的方法效果不佳。LLM 虽然能熟练地评价情节，但在根据评价进行改进时却失败了，这体现了当前模型的一个显著局限性。
- **需要更好的 Prompt 或更大的模型**：另一位成员承认了 **LLM 更擅长评价而非改进**的趋势，建议至少需要 **13b 模型或更好的 Prompt**（如 Chain-of-Thought, CoT）才能获得更有效的结果。
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1242062334118199306)** (2 条消息): 

- **寻求高级 Vision Transformer 技术**：一位用户询问是否有**解释 Vision Transformer 中 patching 技术**的论文，且要求比 VIT 更为先进。他们正在寻找深入的资源来扩展在该领域的知识。
- **屏幕截图中的 Zero-Shot 目标检测**：另一位用户描述了一个任务，涉及**在网页截图中查找所有与参考图像相似的对象**，并强调由于参考图像经常变化，需要使用 Zero-Shot 方法。他们正在寻求高效实现这一能力的指导或解决方案。
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1242029949796225064)** (12 条消息🔥): 

- **LLM 会遗忘对话，需手动存储历史记录**：一位用户反映他们的机器人不会考虑对话历史。成员们建议手动拼接之前的消息，因为 **LLM** 本质上不会记住之前的交流。[该机器人的 GitHub 仓库](https://github.com/jakobdylanc/discord-llm-chatbot)。

- **运行时间对比：Gemini 1.5 Flash vs Llama3-70B**：一位用户指出 **Llama3-70B** 能提供准确的数据模式分析和真实的回答，而 **Gemini Flash** 往往会产生幻觉（hallucinate）。这表明 Llama3-70B 在复杂数据场景中表现更强。

- **用于幻觉检测的集成模型**：一位正在撰写硕士论文的成员分享了他们的方法，即使用 **Mistral 7B** 模型的集成来测量不同类型的不确定性。他们征集可能超出模型训练数据范围的问题，以测试增加的认知不确定性（epistemic uncertainty）是否可以作为幻觉的指标。

- **在 HuggingFace 上托管微调后的 LLM**：一位用户询问关于在 HuggingFace 上托管微调后的 **LLM** 并使用 API 进行请求的问题。他们非常有信心，表示“99.9%”确定这是可以实现的。
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1242026772761940049)** (10 条消息🔥): 

- **Diffusion 频道中的法译英请求**：一位用户最初用法语发布消息，随后将其翻译成英语，解释了 [llmcord chatbot](https://github.com/jakobdylanc/discord-llm-chatbot) 无法保留对话历史的问题。另一位成员建议此类查询更适合 NLP 频道，而非 Diffusion Discussions 频道。

- **LLMcord 聊天机器人对话历史技巧**：另一位用户建议通过在 Prompt 中发送历史记录来解决对话历史问题。他们分享了 [LangChain 文档](https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/)的链接，该文档解释了如何管理聊天消息历史。

- **Diffusion 模型去噪器问题及数学咨询**：一位用户分享了他们在实现 Diffusion 模型时的困扰，提到前向扩散过程（forward diffusion process）很成功，但去噪器（denoiser）存在问题。他们询问应该学习哪个数学领域，特别是关于高斯分布和正态分布的领域；另一位用户建议学习 *变分推理（variational inference）*。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/jakobdylanc/discord-llm-chatbot">GitHub - jakobdylanc/discord-llm-chatbot: llmcord.py • Talk to LLMs with your friends!</a>: llmcord.py • 与你的朋友一起与 LLM 交谈！通过在 GitHub 上创建账户为 jakobdylanc/discord-llm-chatbot 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/">添加消息历史 (memory) | 🦜️🔗 LangChain</a>: RunnableWithMessageHistory 允许我们为特定类型的链添加消息历史。它包装了另一个 Runnable 并为其管理聊天消息历史。
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1242018499040116736)** (402 条消息🔥🔥):

- **Scarlett Johansson 起诉 OpenAI 语音复制**：*报道了 Scarlett Johansson 因 OpenAI 生成其语音而提起诉讼的细节*，并讨论了潜在的法律影响。成员们注意到，在公众的强烈反对下，*OpenAI 随后删除了该语音*。
- **Phi-3 模型发布引起轰动**：Microsoft 在 [Hugging Face](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) 上发布了 **Phi-3-Medium-128K-Instruct 模型**，宣称其改进了 benchmarks 并支持高达 128k 的 context。参与者讨论了其性能以及 context 长度可能存在的问题。
- **Colab 问题与 PyTorch 的 T4 GPU 检测有关**：由于 PyTorch 错误识别了 Tesla T4 的能力，**Colab notebooks 表现异常**，直到 [Unsloth 方面发布更新](https://x.com/danielhanchen/status/1792985678030221464) 后才得以解决。Daniel Hanchen 的一条推文证实了这一识别故障。
- **多样化的 finetuning 讨论**：讨论范围从 **多 GPU 的使用** 到 **在 Google Cloud 与 Colab 上进行模型 fine-tuning**。fine-tuning 的实践细节包括 **dataset handling**、**epoch 配置** 以及 **为 curriculum learning 避免 dataset shuffling**。
- **Optimizers 与 FSDP 更新**：关于 **将 8bit optimizers 与 Fully Sharded Data Parallel (FSDP) 结合使用** 的复杂细节进行了深入交流。参与者分享了解决保存 checkpoint 问题以及跨不同 GPUs 管理 optimizer states 的排错方法。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1792985678030221464">来自 Daniel Han (@danielhanchen) 的推文</a>：@GoogleColab @PyTorch @thechrisperry 更新：一位 @UnslothAI 社区成员 (Edd) 发现 PyTorch 2.3 无法正确检测 Tesla T4s —— PyTorch 认为 Tesla T4 可以支持 bfloat16，但实际上并不支持。...</li><li><a href="https://huggingface.co/microsoft/Phi-3-medium-128k-instruct">microsoft/Phi-3-medium-128k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/fai">fai (fai)</a>：未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/loading#csv">Load</a>：未找到描述</li><li><a href="https://tenor.com/view/explosion-boom-iron-man-gif-14282225">Explosion Boom GIF - Explosion Boom Iron Man - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/oKatanaaa/kolibrify/tree/master/examples/training_mini_dolphin">kolibrify/examples/training_mini_dolphin at master · oKatanaaa/kolibrify</a>：使用 Unsloth 对指令遵循 LLMs 进行课程学习训练 - oKatanaaa/kolibrify</li><li><a href="https://tenor.com/view/no-no-wait-wait-gif-8174347161288218584">No No Wait Wait GIF - No no wait wait - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_optim_utils.py#L1369>">pytorch/torch/distributed/fsdp/_optim_utils.py at main · pytorch/pytorch</a>：Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/unslothai/unsloth/wiki#gguf-quantization-options">Home</a>：使用 Unsloth 微调 Llama 3, Mistral & Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/issues/11693">Flag to disable shuffling for data loader · Issue #11693 · huggingface/transformers</a>：🚀 功能请求 目前，Trainer 默认会对 train_dataset 进行洗牌，且没有开启/禁用的标志。@sgugger 动机 即使对数据集进行洗牌会带来很多好处 ......</li><li><a href="https://github.com/hsiehjackson/RULER?tab=readme-ov-file>">GitHub - hsiehjackson/RULER: 此仓库包含 RULER 的源代码：你的长上下文语言模型的真实上下文大小是多少？</a>：此仓库包含 RULER 的源代码：你的长上下文语言模型的真实上下文大小是多少？ - hsiehjackson/RULER</li><li><a href="https://www.npmjs.com/package/grammar-builder">grammar-builder</a>：一个与 GBNF (llama.cpp) 兼容的简单语法构建器。最新版本：0.0.5，最后发布于 11 天前。通过运行 `npm i grammar-builder` 在你的项目中使用 grammar-builder。这里有...</li><li><a href="https://imgur.com/FhBnfFP">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行的模因、有趣的 GIF、鼓舞人心的故事、病毒式视频等来振奋你的精神...</li><li><a href="https://tenor.com/view/sad-sad-cat-cat-depressed-depression-gif-13240550249247957481">Sad Sad Cat GIF - Sad Sad cat Cat - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 使用 Unsloth 微调 Llama 3, Mistral & Gemma LLMs，速度提升 2-5 倍，显存占用减少 80%</a>：使用 Unsloth 微调 Llama 3, Mistral & Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=IqM-T1RTzY6C">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1242432957030076466)** (4 条消息): 

- **新方法预警：MoRA**：一位用户提到了一种名为 **MoRA** 的新方法，并表示有兴趣尝试其原始实现。另一位用户热情地回应道，它“看起来很史诗”。[arxiv 链接](https://arxiv.org/abs/2405.12130)。

**提及的链接**：<a href="https://arxiv.org/abs/2405.12130">MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning</a>：低秩自适应（Low-rank adaptation）是一种流行的大型语言模型参数高效微调方法。在本文中，我们分析了 LoRA 中实现的低秩更新的影响。我们的研究结果表明...

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1242033910376955938)** (246 条消息🔥🔥): 

```html
- **上传使用 Unsloth 训练的模型**：一位用户分享了一个使用 Unsloth 微调并上传到 Hugging Face 的模型，询问运行该模型的最佳方式，特别是提到担心 Ollama 仅适用于预定义模型。另一位用户推荐了 Ollama、LM Studio、Jan 和 GPT4ALL 等工具，并指出仅上传了 LORA 适配器。
- **Mistral 微调中的数据集依赖问题**：一位用户面临 Mistral-instruct-7b 过度依赖数据集的问题，导致对新输入产生错误或空的输出。其他人建议混合数据集以帮助模型更好地泛化。
- **T4 上 TRT 和 Flash Attention 的问题**：由于 PyTorch 2.3 的更新以及 Flash Attention 的问题，多位用户在 Google Colab 的 T4 GPU 上运行 Unsloth 时遇到错误。指定 dtype 或遵循更新后的安装说明有助于缓解该问题。
- **由于 VRAM 限制使用 4bit 模型**：用户讨论了在 VRAM 有限的设备上微调模型的挑战。提到了利用 4bit 量化模型以在 VRAM 限制内适配更大的模型，特别是针对像拥有 6GB VRAM 的 GTX 3060 这样的硬件。
- **微调数据集中重复指令的确认**：用户探讨了在微调数据集中使用重复指令的有效性。对话显示了对该方法的好奇和积极实验，但对其整体影响尚无定论。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/issues/485">使用 llama-cpp-python · Issue #485 · unslothai/unsloth</a>: 你好，感谢创建这个精彩的包！目前的 save_to_gguf 失败了，因为 llama.cpp 的安装似乎损坏了。可以使用像 llama-cpp-python 这样的工具代替吗？</li><li><a href="https://x.com/danielhanchen/status/1792982364894929083">来自 Daniel Han (@danielhanchen) 的推文</a>: 噢不，@GoogleColab 升级到了 @PyTorch 2.3，而 T4 GPU 无法与 Triton 2.3 配合使用！我尝试将 Triton 降级到 2.2，但仍然失败。这似乎是 Torch 2.3 的问题。@thechrisperr...</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-gguf">主页</a>: 使用 Unsloth 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3, Mistral & Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/pull/506/commits/2b23b9357aba25ab2f3a49d899045547d7dde1d7">danielhanchen 的 Nightly 版本 · Pull Request #506 · unslothai/unsloth</a>: 未找到描述</li><li><a href="https://www.unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>: 通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，支持 6 倍长的上下文长度！</li><li><a href="https://huggingface.co/omar8/bpm_v2_gguf">omar8/bpm_v2_gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/omar8/bpm__v1/tree/main">omar8/bpm__v1 在 main 分支</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">量化</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://youtu.be/AK4IzTQJI9E?si=ppEisWZUs0DXl9hp">Windows 微调综合流</a>: LLM 微调是使 LLM 在特定场景中表现更好的首选技术之一。在这篇文章中，我将向你展示如何准备本地 Windows 环境...</li><li><a href="https://github.com/pytorch/pytorch/blob/b40fb2de5934afea63231eb6d18cc999e228100f/torch/cuda/__init__.py#L130C1-L151C1">pytorch/torch/cuda/__init__.py 在 b40fb2de5934afea63231eb6d18cc999e228100f · pytorch/pytorch</a>: Python 中的张量和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 使用 Unsloth 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3, Mistral & Gemma LLM</a>: 使用 Unsloth 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3, Mistral & Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/210">我在原生 Windows 上成功运行了 Unsloth · Issue #210 · unslothai/unsloth</a>: 我在原生 Windows（非 WSL）上运行了 Unsloth。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一个完整的安装教程，我本想在这里全部写出来，但我现在在用手机...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1242035477075333211)** (13 条消息🔥): 

- **Dolphin-Mistral-2.6 仅需更少样本即可匹敌**：一名成员报告称，在指令遵循评估中，仅使用约 20k 个样本就成功达到了 **dolphin-mistral-2.6** 的性能水平，而原始模型使用了数百万个样本。讨论中提到了 **kolibri-mistral-0427** 和 **kolibri-mistral-0426-upd** 模型，并强调了训练数据流水线（pipelines）的差异。

- **即将发布的模型**：该用户计划在几天内发布该模型，并承诺很快分享训练“配方（recipe）”，尽管其中包含一些专有数据，可能会对可复现性产生轻微影响。关于这些发现的论文可能会在今年晚些时候发表。

- **社区反应**：社区对这一消息反应热烈，多名成员表示祝贺并表达了期待。一位成员分享了他们对详细介绍低样本训练方法文章的期待，并提到他们个人面临的挑战是无法将训练样本减少到 52k 以下。
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1242014000162537532)** (618 条消息🔥🔥🔥): 

```html
- **解决订阅困惑**：用户对提供 Stable Diffusion 订阅的不同网站表示困惑，其中一些被识别为诈骗。官方网站 [stability.ai](https://stability.ai) 被推荐为访问 Stable Diffusion 服务的合法来源。
- **离线运行软件**：讨论了在没有互联网连接的情况下在本地运行 Kohya 的问题。用户确认，只要正确下载模型并完成设置，就可以离线运行。
- **Stable Diffusion 安装难题**：几位用户在安装和运行 Stable Diffusion 及 ComfyUI 等相关工具时寻求帮助。社区提供了关于处理依赖项和通过终端命令进行故障排除的指导。
- **对欧盟 AI 法案的担忧**：欧盟 AI 法案（EU AI Act）的通过引起了用户的担忧，特别是其对 AI 生成内容的潜在影响以及水印要求的引入。许多人对这类法规的实用性和执行力表示怀疑。
- **基准测试性能困惑**：一位用户强调了在新硬件上生成 SD 时的性能问题，怀疑原因是热节流（thermal throttling）。社区成员建议检查配置并使用 diffusers 脚本进行更好的诊断。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.stablediffusionai.ai/">Stable Diffusion AI Generator Online | Stable Diffusion XL Powered</a>: 未找到描述</li><li><a href="https://invideo.io/">Invideo AI - Turn ideas into videos - AI video creator </a>: 通过向 invideo AI 提供提示词轻松制作视频。invideo AI 为内容创作者、YouTuber 和营销人员提供了一种将创意转化为 AI 生成的即用型视频的无缝方式。</li><li><a href="https://youtu.be/G7mihAy691g">Stable Video Diffusion</a>: 未找到描述</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.</a>: 最强大且模块化的 Stable Diffusion GUI、API 和后端，采用图/节点界面。 - comfyanonymous/ComfyUI</li><li><a href="https://tenor.com/view/alvin-and-the-chipmunks-alvin-whoops-my-bad-oops-gif-15512287650458333097">Alvin And The Chipmunks Alvin GIF - Alvin And The Chipmunks Alvin Whoops - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/welcome-gif-26939290">Welcome GIF - Welcome - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://download.pytorch.org/whl/nightly/cu121/torch-2.4.0.dev20240520%2Bcu121-cp311-cp311-win_amd64.whl">未找到标题</a>: 未找到描述</li><li><a href="https://stability.ai/">Stability AI</a>: 通过生成式 AI 激发人类潜力。为每个人、每个地方提供各种模态的开源模型。</li><li><a href="https://stability.ai/stable-assistant">Stable Assistant &mdash; Stability AI</a>: Stable Assistant 是由 Stability AI 开发的友好聊天机器人，配备了 Stability AI 的文本和图像生成技术，具有 Stable Diffusion 3 和 Stable LM 2 12B 的特点。</li><li><a href="https://tenor.com/view/trollszn123-ronaldo-gif-18268194">Trollszn123 Ronaldo GIF - Trollszn123 Ronaldo - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1242476097174507561)** (1 条消息): 

- **AI Seoul Summit 上宣布安全更新**：配合 AI Seoul Summit，分享了一项新的安全更新。欲了解更多详情，请访问 [OpenAI Safety Update](https://openai.com/index/openai-safety-update/)。
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1242053124718329876)** (229 条消息🔥🔥): 

- **GPT-4o 帧采样讨论**：成员们讨论了 GPT-4o 的视频处理能力，推测其处理视频的速度为 **每秒 2-4 帧**。一位成员分享了一个 [社区讨论](https://community.openai.com/t/announcing-gpt-4o-in-the-api/744700) 的链接，描述了将视频转换为模型所需帧的过程。
- **向 GPT-4o API 传递图像 Buffer**：一位成员在向 GPT-4o Vision API 传递 `Buffer` 对象时遇到困难，其他人建议将其编码为 base64 data URL。他们讨论了确保正确设置 base64 字符串的 MIME 类型，以避免 API 响应中出现静默失败。
- **Microsoft Copilot 与 GPT-4o 集成**：成员们讨论了将 **GPT-4o** 集成到 Microsoft Copilot 中的公告，承诺提供实时语音和视频功能。他们预计在“未来几周”内可用，并推测了集成系统的优势。
- **关于 Scarlett Johansson 声音的争议**：讨论了 OpenAI 在其 Sky 语音功能中使用与 Scarlett Johansson 相似声音的争议。在 Johansson 的律师介入后，社区指出了潜在的法律和伦理影响。
- **Microsoft 的新 Phi-3 模型**：Microsoft 宣布了新的 **Phi-3** 模型，包括集成了语言和视觉能力的多模态模型，可在 [Azure](https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/) 上使用。成员们反应不一，并分享了进一步阅读的链接。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.platformer.news/open-ai-scarlett-johansson-her-voice-sam-altman/?ref=platformer-newsletter">OpenAI 失去了它的声音</a>：自 Sam Altman 回归以来，这家公司就变得不一样了——它对 Scarlett Johansson 的态度应该引起所有人的警惕。</li><li><a href="https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/">Phi-3 家族新增模型，可在 Microsoft Azure 上使用 | Microsoft Azure 博客</a>：我们推出了 Phi-3-vision，这是一款结合了语言和视觉能力的多模态模型，现已在 Microsoft Azure 上提供。了解更多。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1242027479007363134)** (38 条消息🔥): 

- **理解 GPT-4 的 Context Window**：一位成员询问了 GPT-4 Omni 128,000 个 token 的 “context window”，寻求澄清这是否指 prompt 大小。另一位成员澄清说，context window 是 prompt 和响应组合后的最大尺寸，并引用了一篇 [帮助文章](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)。

- **JSON 响应长度问题**：一位成员在配置了系统指令限制内容为 200 个 token 的情况下，仍然收到大型 JSON 响应。他们注意到使用 GPT-4 turbo 会产生较短的响应，并计划进一步调整系统指令。

- **销售 AI 生成的艺术作品**：讨论确认了销售 AI 生成的艺术作品是可能的，尽管此类艺术的版权性仍然是一个独立且复杂的问题。一位成员提到，公共领域可以作为可销售作品的来源，因为有效地向 AI 发送 prompt 具有挑战性。

- **关于 GPT-4 评估数值的担忧**：一场关于 GPT-4 难以正确评估简单数值表达式的讨论展开，揭示了它可能需要依赖 code interpreter 来提高准确性。

- **下载 Mac 版 GPT 应用的注意事项**：成员们建议等待其账户上出现官方提示后再下载 macOS 版 ChatGPT 应用，并警告不要点击非官方链接。
  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1242029779935432816)** (73 条消息🔥🔥): 

- **在 GPT-4 中设置字符限制**：为了实现 150 个字符的回复限制，建议在 Prompt 中提供约 120 个字符的示例输出，以防止超出限制。一位用户分享了[模型尝试此任务的示例](https://chatgpt.com/share/f38d2248-ff7f-4cc3-989e-526e68dc54f4)，展示了其中的难度。
- **针对特定行为训练模型**：为了复制电影《她》（Her）中的 AI，需要定义准确的行为参数，并使用输入/输出对来塑造回复。避免使用否定指令，以提供更清晰的引导。
- **回复精确度不一致**：用户讨论了从模型获取准确答案的挑战，例如要求特定范围而非泛泛而谈。反复要求细节会有所帮助，但当模型无法提供准确数据时，可能会出现“幻觉”或自动补全。
- **管理 Token 限制以避免过度输出**：设置 max token 参数并编写具体、简洁的 Prompt 可以帮助管理输出的冗余度。包含清晰的输出模板并将回复限制在一个段落或一句话内可以提高简洁性。
- **在代码中高效使用 Prompt Engineering**：用户分享了高效生成代码的 Prompt 策略，强调了在协作编程环境中的精确缩进和基于角色的 Character Prompt。示例包括用于创建和调试全栈应用程序的详细 Prompt。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1242029779935432816)** (73 条消息🔥🔥): 

- **在 GPT-4 中设置字符限制很棘手**：一名成员征求关于设置 150 个字符回复限制的建议。建议包括提供约 120 个字符的示例输出，因为模型经常会超出限制（*“它会超出，因此目标要设得比限制小，这样你就有望不超标”*）。

- **训练像《她》中 AI 一样的模型引发讨论**：一位用户询问如何训练模型使其表现得像电影《她》中的 AI。建议包括使用输入/输出对以及避免否定指令。

- **精确语言使用方面的困扰**：一位成员讨论了尽管有提供精确数据的指令（如营养标签或薪资范围），模型仍给出模糊答案的问题。有人建议这可能是由于自动补全和指令冲突造成的（*“它往往不太仔细地遵循格式”*）。

- **防止 LLM 啰嗦不止**：成员们讨论了尽管在 API 中设置了 Token 限制，模型仍产生冗长回复的问题。建议包括使用具体问题、要求简洁回答以及采用输出模板（*“Prompt 应该要求它将回答限制在一句简洁的话内”*）。

- **Prompt 分享与改进**：一位用户提出分享用于构建全栈应用程序的有效 Prompt，并指出了 Prompt Engineering 中的错误。另一位成员指出这可能更适合 Prompt Labs 频道，并提到他们对模型冗长以及 Explore GPTs 菜单使用的挫败感。
  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1242041743239544912)** (191 条消息🔥🔥): 

- **排查 LM Studio Server 问题**：一位用户遇到了 LM Studio server 日志空白且无响应的问题。通过以管理员权限运行 LM Studio 以正确访问日志文件，该问题得到了解决。

- **澄清 AVX2 指令集困惑**：几位成员澄清了 **AVX2 指令集**对于运行 LM Studio 是必不可少的，用户可以使用 HWInfo 等工具检查其 CPU 是否支持。AVX2 是硬性要求，不支持该指令集的旧款 CPU 将无法运行 LM Studio。

- **加载与管理模型**：用户讨论了在 LM Studio 中下载和运行模型的各种问题。一个有效的策略包括下载 **GGUF 格式**的模型，并确保所有系统提示词（system prompts）和设置都已正确配置。

- **LM Studio 与其他工具的集成**：有用户提出了关于将 LM Studio 与 StarCoderEx 和 Continue.dev 等工具集成以增强功能的问题。一些对这些集成有经验的用户提供了有用的链接 [Continue.dev 集成指南](https://docs.continue.dev/walkthroughs/tab-autocomplete#setting-up-with-lm-studio)。

- **常见 GPU 与性能查询**：针对频繁出现的性能问题，会议强调 GPU 应至少具备 8GB VRAM 才能高效运行。用户还分享了提到显存不足和驱动程序过时是常见原因的具体错误，并建议进行更新和调整 GPU offload 设置。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - 由 ggml-org 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://docs.continue.dev/walkthroughs/tab-autocomplete#setting-up-with-lm-studio">Tab 自动补全 (beta) | Continue</a>: Continue 现在支持 VS Code 和 JetBrains IDE 中的 Tab 自动补全。我们将在接下来的几个版本中大幅提升体验，随时欢迎反馈。如果...</li><li><a href="https://pinokio.computer/">Pinokio</a>: AI 浏览器</li><li><a href="https://github.com/Lisoveliy/StarCoderEx">GitHub - Lisoveliy/StarCoderEx: 在 VSCode 中使用替代 GitHub Copilot (StarCoder API) 的扩展</a>: Extension for using alternative GitHub Copilot (StarCoder API) in VSCode - Lisoveliy/StarCoderEx</li><li><a href="https://www.hwinfo.com/download/">免费下载 HWiNFO 软件 | Windows、DOS 安装版与便携版</a>: 立即开始分析您的硬件！HWiNFO 提供适用于 Windows（32/64 位）的安装版和便携版，以及适用于 DOS 的便携版。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1242026637193773099)** (57 条消息🔥🔥): 

- **成功的模型设置需要正确的 Prompt**：一位成员询问如何在 **LM Studio** 上使用 **MPT-7b-WizardLM** 模型，另一位成员建议使用正确的量化级别和模板，并指向了 [Hugging Face](https://huggingface.co/DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF) 上的模型特定详情。
- **图像生成质量技巧**：几位成员讨论了如何使用 **Automatic1111** 和 **ComfyUI** 等本地 AI 模型提高图像质量。建议包括使用来自 [Civit.ai](https://civitai.com/) 的资源，并考虑 VRAM 和 RAM 等系统规格。
- **Phi-3-Small 和 Medium 模型发布**：成员们提到了在 Hugging Face 上发布的具有 4K、8K 和 128K token 上下文长度的新 **Phi-3** 模型。[Phi-3-Small-8K](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) 和 [Phi-3-Medium-4K](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) 被特别强调。
- **使用专业模型改进 LLM 响应**：一位用户提到使用 **codeqwen** 模型以获得更好的编程能力。改进建议包括使用微调模型，并利用 **ComfyUI** 等高级设置来处理专门任务。
- **本地视觉模型在处理特定 Prompt 时表现不佳**：一位用户报告了 **vision models** 无法遵循特定 Prompt 查询的问题。多位用户建议，本地视觉模型通常无法有效处理多轮对话。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF">DavidAU/MPT-7b-WizardLM_Uncensored-Storywriter-Merge-Q6_K-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-medium-4k-instruct">microsoft/Phi-3-medium-4k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-small-8k-instruct">microsoft/Phi-3-small-8k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7439>">Issues · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>：通过在 GitHub 上创建账户为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://huggingface.co/collections/DavidAU/roleplay-creative-writing-uncensored-nsfw-66163c580c61496c340afe32">Roleplay, Creative Writing, Uncensored, NSFW - a DavidAU Collection</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1242144163001143406)** (9 条消息🔥): 

- **介绍 Hugging Face 与 LM Studio 的集成**：用户现在可以通过点击 "Use this model" 直接将 Hugging Face 模型集成到 LM Studio 中，这需要 LM Studio 0.2.23 或更高版本。正如所强调的，该功能确保了 *“无云端、无成本、不向任何人发送数据、没问题”*。
- **模型下载自定义**：在当前版本中，用户在选择 Hugging Face 模型后必须手动选择想要下载的文件。讨论了诸如设置默认量化级别或根据可用 RAM 自动下载等建议。
- **兼容性限制**：有人指出并非所有模型都受 LM Studio 支持，特别是许多 safetensor 模型。目前仅兼容 GGUF 格式的模型。

**提到的链接**：<a href="https://x.com/LMStudioAI/status/1792576553601102024">来自 LM Studio (@LMStudioAI) 的推文</a>：1. 浏览 HF 2. 这个模型看起来很有趣 3. 在 LM Studio 中使用它 👾🤗 引用 clem 🤗 (@ClementDelangue) 无云端、无成本、不向任何人发送数据、没问题。欢迎来到 Hugging Face 上的本地 AI...

  

---

### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1242118716351844423)** (3 messages): 

- **System Prompt 调优防止过早截断**：一位成员建议在 [system] prompt 中加入 "Do not prematurely cut off a response"（不要过早截断响应），这将有助于解决响应不完整的持续问题。这一见解旨在增强聊天机器人的响应可靠性。
- **直接引用提高指令清晰度**：该成员建议直接引用所需文本，并在 prompt 中添加指令，例如 *"Considering the following text alone as input, <insert subsequent instructions here>."*（仅将以下文本视为输入，<在此插入后续指令>）。该方法旨在细化 prompt 的针对性以获得更好的结果。
- **幽默地确认旧帖**：一位成员幽默地承认了之前帖子的年份，并表示 *"Didn't realize how old that post was. 😆"*（没意识到那是多久以前的帖子了）。这为讨论背景增添了轻松的氛围。
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1242229857321095270)** (1 messages): 

- **LM Studio 在 Linux 上使用 VPN 时遇到困难**：一位用户报告了一个问题，即在 Linux 上通过 VPN 连接时，**LM Studio** 无法进行模型搜索。他们正在寻找遇到过此问题的其他用户以及任何可能的解决方案。
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1242031209039532074)** (27 messages🔥): 

- **Infinity Fabric 速度同步影响性能**：一位成员强调了保持 **infinity fabric (fclk)** 速度与内存速度同步对获得最佳性能的重要性，并建议 *“fclk 应该与内存速度同步，否则你会看到性能下降。”*
- **免费服务与能耗担忧**：推荐使用 **Groq** 和 OpenRouter 等免费服务以避免高昂成本。一位用户分享说，他们拥有 144GB VRAM 的强大设备在温暖的天气里会让房子明显升温。
- **RAM 速度对模型的影响**：将 RAM 速度从 **2133MHz 升级到 3200MHz** 提升了 Goliath 模型的性能，但对于其他模型，超过 2666MHz 后的提升微乎其微。有人建议，一旦超过 VRAM 容量，**iQuant 的表现可能会变差**。
- **尝试不同的模型**：对各种 **Quant 模型** 的测试揭示了 iQuant 和常规 Quant 之间的性能差异，当超过 VRAM 容量时，iQuant 表现不佳。
- **在双 GPU 上运行 LM Studio**：关于使用两个不同 GPU 运行 LM Studio 的咨询得到了肯定的回答，只要两个 GPU 品牌相同即可，例如 *“同为 nvidia 或同为 amd。”*


  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1242268966928388267)** (4 messages): 

- **寻求能强制执行 60 字限制的模型**：一位成员寻求帮助，希望让 **Meta Lama 3 Instruct** 遵守 60 个单词的响应限制。另一位成员建议列出已尝试的方法及其结果，以便更好地排查问题。
- **寻找更合适的模型**：原帖作者询问是否有比 Meta Lama 3 更适合强制执行严格响应限制的模型。他们接受了建议，并计划提供有关其尝试的更多细节。


  

---


### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1242096330893950996)** (13 messages🔥): 

- **Null max_tokens 导致截断问题**：据指出，在 LM Studio 中将 **max_tokens** 设置为 **null** 会导致响应在两个 token 后截断。解决方法是将其设置为 **-1**，这有助于本地服务器正常运行。
- **分享 CLI LMStudio Client 解决方案**：一位正在构建 CLI LMStudio Client 的成员确认，将 **max_tokens** 设置为 **-1** 可以解决响应被截断的问题。另一位贡献者提到必须手动编辑 autogpt 中的代码才能使其工作。
- **Autogen Studio 修复方法讨论**：讨论了此修复程序是仅适用于命令行版本，还是可以在 Autogen Studio 中实现。一些人通过更改 root autogen 包中的值确认成功，暗示在 Autogen Studio 中具有类似的有效性。
- **Manager agents 可靠性担忧**：有人建议 manager agents 仅在 OpenAI 模型下可靠。测试者注意到在选择合适的 agent 时存在 bug 且性能较差，建议在改进之前使用 round-robin 或硬编码的工作流。
- **删除缓存可能有所帮助**：为了解决截断问题，建议在将 **max_tokens** 设置为 **-1** 后删除应用程序缓存。成员们经常遇到这个问题，并发现删除缓存对于修复生效是必要的。
  

---

### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1242203464889536552)** (42 条消息🔥): 

- **招募拥有 AMD GPU 的 Linux 爱好者**：一名成员宣布招募**拥有较新 AMD GPU 的 Linux 用户**，以测试适用于 Linux + ROCm 的 LM Studio 早期版本，并提供了 [受支持 GPU 列表的链接](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。从 6600xt 到 7900XT 的各类 GPU 用户都表达了兴趣，评论包括 “6900xt 前来报到” 和 “这里有块 6600xt”。
  
- **不受支持的 GPU 似乎也能运行**：几位用户报告在未列入官方支持名单的 GPU 上成功运行了 ROCm。一位拥有 6600xt 的成员提到：*“它不在 ROCm 的支持列表中，但我已经用它运行 ROCm 来跑 Stable diffusion 了。”*

- **ROCm 测试小组涵盖多种 Linux 发行版**：运行 Arch, Fedora 和 Ubuntu 等多种 Linux 发行版的用户分享了他们的经验。甚至有人指出，通过使用 *“HSA_OVERRIDE_GFX_VERSION=10.3.0”*，在 RX 6600xt 上成功使用了 ROCm。

- **CPU 占用观察与讨论**：围绕 Linux 上 ROCm 的 CPU 占用情况展开了讨论，一位成员幽默地指出 *“啊是的，229% 的 CPU 占用率”*，另一位则认为 Linux 加快了处理速度。关于 Linux 性能的评论包括 *“它确实很快”*，以及关于 Linux 与 Windows RAM 占用情况的辩论。

- **Arch Linux 与 ROCm 兼容性获得好评**：成员们称赞在 Arch Linux 上设置 ROCm 和 HIP SDK 非常容易。Quickdive 指出：*“Arch 让 ROCm 和 HIP SDK 变得如此简单，”* 许多人表示赞同并分享了类似的成功案例。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rocm.docs.am">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/reunited-peaches-and-herb-and-it-feels-so-good-cause-we-understood-old-skool-gif-17279659">Reunited Peaches And Herb GIF - Reunited Peaches And Herb And It Feels So Good - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1242144891186970695)** (38 条消息🔥): 

- **Mojo 开放社区会议开始**：Mojo 的开放社区会议正在直播，可以通过提供的 [Zoom 链接](https://modular.zoom.us/j/89417554201?pwd=Vj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1) 加入。一位成员询问了录像的回放，录像稍后会分享。
- **录像已上传至 YouTube**：Mojo 社区会议的录像现在可以在 [YouTube](https://www.youtube.com/playlist?list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D) 上观看。 
- **澄清 Zoom 账号困惑**：一些成员对于是否需要商业版 Zoom 账号才能加入感到困惑。会议澄清只需基础账号即可，尽管最初可能存在配置错误。
- **错过会议的遗憾**：Helehex 因为没有收到通知提醒而错过了会议，感到很遗憾。随后提供了未来会议的详情，包括日历订阅选项。
- **Python 中的 IPC 讨论**：Moosems_yeehaw 寻求关于 Python 中 IPC（进程间通信）的建议，以避免 Tkinter 应用示例中的主线程卡顿。成员们给出了各种建议，包括 threading、消息队列和异步 IPC 模块。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.google.com/url?q=https://modular.zoom.us/j/89417554201?pwd%3DVj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1&sa=D&source=calendar&ust=1716255791532130&usg=AOvVaw2IgLzFgI9-S5vkyEC7_b2v">重定向通知</a>：未找到描述</li><li><a href="https://www.google.com/url?q=https://modular.zoom.us/j">重定向通知</a>：未找到描述</li><li><a href="https://docs.google.com/document/d/1Hdy52tJXbUR2jZSYt-IFdaEJRRBHvHCQkODAZnuXsNc/edit">[公开] Mojo 社区会议</a>：未找到描述</li><li><a href="https://tenor.com/view/cloudy-with-a-chance-of-meatballs-enough-to-make-a-grown-man-cry-police-officer-make-a-man-cry-gif-15227532">Cloudy With A Chance Of Meatballs Enough To Make A Grown Man Cry GIF - Cloudy With A Chance Of Meatballs Enough To Make A Grown Man Cry Police Officer - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1242261353507590265)** (2 条消息): 

- **Modular 分享最新推文**：分享了一个 [Modular 推文](https://twitter.com/Modular/status/1792701156122415589) 的链接。
- **Modular 的另一条推文**：同时也分享了另一个 [Modular 推文](https://twitter.com/Modular/status/1792701170634699243) 的链接。
  

---

### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1242261428564525057)** (1 条消息): 

- **在 Mojo 中实现 K-means 聚类以提升速度**：一篇新的博客文章旨在教读者如何从零开始在 Python 和 Mojo🔥 中实现 k-means 聚类算法，重点强调了 Mojo 的性能优势。该文章还提供了将 Python 代码移植到 Mojo 以实现显著速度提升的详细指南。阅读更多请访问 [Modular 博客](https://www.modular.com/blog/fast-k-means-clustering-in-mojo-guide-to-porting-python-to-mojo-for-accelerated-k-means-clustering)。

**提到的链接**：<a href="https://www.modular.com/blog/fast-k-means-clustering-in-mojo-guide-to-porting-python-to-mojo-for-accelerated-k-means-clustering">Modular: Mojo🔥 中的快速⚡ k-means 聚类：将 Python 移植到 Mojo🔥 以加速 k-means 聚类的指南</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo🔥 中的快速⚡ k-means 聚类：将 Python 移植到 Mojo🔥 以加速 k-means 聚类...

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1242078427310198785)** (258 条消息🔥🔥): 

- **通过教程学习 Mojo 和 ML**：一位用户询问是否应该在 Mojo 中实现一个机器学习教程，以便同时学习 Mojo 和 ML。另一位用户建议尝试一下，并指出 Mojo 目前不支持 classes，但可以使用 structs，并且可能需要实现一些 numpy 的功能。

- **Modular 社区会议通知**：一位用户在频道中通知了正在进行的 Modular 社区会议，并分享了 Zoom 链接。另一位用户评论了 Chris Lattner 在会议期间关于将 Tensor 移出标准库的发言。

- **字符串中的空终止符（Null Terminator）处理**：一位用户在将 bytes 转换为 strings 并进行迭代时，在处理空终止符方面遇到了困难。他们分享了自己的尝试以及通过社区帮助找到的解决方案，包括使用 append(0) 方法来正确处理空终止符。

- **Mojo 异步编程辩论**：成员们讨论了异步编程中函数着色（function coloring）的优缺点。一些人主张探索无着色（colorless）的异步编程，以简化 API 使用并减轻负担，而另一些人则强调了保留函数着色在安全性以及推断代码行为方面的优势。

- **Lightbug HTTP 框架使用**：一位用户询问了如何使用 Lightbug HTTP 框架发送 GET 请求并解码响应。在实现过程中遇到困难后，维护者和社区提供了帮助，并将对话转移到了 [GitHub 上的 issue](https://github.com/saviorand/lightbug_http/issues/41) 以进行进一步讨论。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.google.com/url?q=https://modular.zoom.us/j/89417554201?pwd%3DVj17RNBZG7QMbrT2GKodMHoKx6Wvtr.1&sa=D&source=calendar&usd=2&usg=AOvVaw37jsmYkBEWm4CHK4NwSCMB">重定向通知</a>: 未找到描述</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read_bytes)">FileHandle | Modular 文档</a>: 已打开文件的文件句柄。</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read">FileHandle | Modular 文档</a>: 已打开文件的文件句柄。</li><li><a href="https://without.boats/blog/the-registers-of-rust/">Rust 的寄存器</a>: 未找到描述</li><li><a href="https://github.com/modularml/mojo/blob/main/proposals/inferred-parameters.md">mojo/proposals/inferred-parameters.md 分支 main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/saviorand/lightbug_http/issues/41),">Issues · saviorand/lightbug_http</a>: 简单且快速的 Mojo HTTP 框架！🔥。通过在 GitHub 上创建账号来为 saviorand/lightbug_http 的开发做出贡献。</li><li><a href="https://github.com/saviorand/lightbug_http/blob/1eb9242ce0ddeeec39ac858028a7117dde627523/lightbug_http/tests/test_client.mojo#L13">lightbug_http/lightbug_http/tests/test_client.mojo 位于 1eb9242ce0ddeeec39ac858028a7117dde627523 · saviorand/lightbug_http</a>: 简单且快速的 Mojo HTTP 框架！🔥。通过在 GitHub 上创建账号来为 saviorand/lightbug_http 的开发做出贡献。</li><li><a href="https://github.com/saviorand/lightbug_http/releases/tag/latest-build">发布 latest-build: 合并来自 Moosems/main 的拉取请求 #27 · saviorand/lightbug_http</a>: 未找到描述</li><li><a href="https://github.com/saviorand/lightbug_http?tab=readme-ov-file>">GitHub - saviorand/lightbug_http: 简单且快速的 Mojo HTTP 框架！🔥</a>: 简单且快速的 Mojo HTTP 框架！🔥。通过在 GitHub 上创建账号来为 saviorand/lightbug_http 的开发做出贡献。</li><li><a href="https://github.com/laspy/laspy/tree/master/laspy">laspy/laspy 分支 master · laspy/laspy</a>: Laspy 是一个用于读取/修改/创建符合 1.0-1.4 规范的 .LAS LIDAR 文件的 Python 风格接口。 - laspy/laspy</li><li><a href="https://github.com/saviorand/lightbug_http/blob/main/lightbug_http/http.mojo">lightbug_http/lightbug_http/http.mojo 分支 main · saviorand/lightbug_http</a>: 简单且快速的 Mojo HTTP 框架！🔥。通过在 GitHub 上创建账号来为 saviorand/lightbug_http 的开发做出贡献。</li><li><a href="https://github.com/saviorand/lightbug_http/blob/bd2f4ef57765505210256165b5386b890a2aa0be/lightbug_http/http.mojo#L12">lightbug_http/lightbug_http/http.mojo 位于 bd2f4ef57765505210256165b5386b890a2aa0be · saviorand/lightbug_http</a>: 简单且快速的 Mojo HTTP 框架！🔥。通过在 GitHub 上创建账号来为 saviorand/lightbug_http 的开发做出贡献。</li><li><a href="https://victorzhou.com/blog/intro-to-neural-networks/">初学者机器学习：神经网络简介 - victorzhou.com</a>: 简单解释它们的工作原理以及如何使用 Python 从零开始实现一个。</li><li><a href="https://github.com/modularml/mojo/issues/2678">[功能请求] 更好地处理字符串中的空终止符 · Issue #2678 · modularml/mojo</a>: 审查 Mojo 的优先级。我已经阅读了路线图和优先级，并相信此请求符合优先级。你的请求是什么？我希望通过讨论来回答以下问题...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1242152864957337641)** (13 条消息🔥): 

- **在 Mojo 中优化 SIMD Gather 和 Scatter**：一位成员询问 Mojo 的 SIMD gather 和 scatter 操作是否已完全优化，并讨论了将值对齐到 32-bit 边界以寻求潜在速度提升的可能性。另一位成员分享了经验，表示 gather 和 scatter 已经得到了很好的优化，尽管对齐带来的收益尚不确定。

- **ARM SVE 与 SIMD 宽度的挑战**：讨论强调了 ARM Scalable Vector Extension (SVE) 的复杂性、可变向量宽度以及跨页边界的投机加载（speculative loads）。一位成员指出 LLVM 在处理 SVE 格式时比较吃力，而有限的 CPU 可用性使这一问题更加复杂。

- **考虑减少 SIMD 操作**：一位成员建议通过始终使用尽可能最高的 SIMD 宽度来减少 gather/scatter 操作的数量，这涉及更多的索引操作（index manipulation）以获得更好的性能。他们计划据此更新并分享其 MoCodes 项目的结果。

- **针对分散内存访问进行排序**：另一位成员建议在处理数 KB 的分散内存时，通过对指针数组进行排序来优化性能，特别是对于迭代解码器（iterative decoders）。

- **向量化的 DTypePointer Memset 实现**：一位成员分享称，针对 100,000 字节的 memset 向量化实现在性能上比 LLVM 的调用快 20%，但在处理 1,000,000 字节时性能优势发生了反转。该成员对可靠性表示担忧，并提到了对 "clobber memory" 的使用。
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1242096661845774386)** (31 条消息🔥): 

- **新的 Mojo nightly 编译器版本发布**：发布了 Mojo 编译器的最新 nightly 构建版本（版本号 `2024.5.2012`）。你可以查看[自上次发布以来的差异](https://github.com/modularml/mojo/compare/7e8cd37ff8fe2ddbe69a3cca787e59abf6357d76...69e92d0040af838de8f3f0fdba1cea92f1904986)以及自[上一个稳定版本](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)以来的变更。

- **Dict pop 方法问题**：讨论了字典中 `pop` 方法的问题，特别是关于从 `DictEntry` 中移出值以及正确调用 `__del__` 的困难。提议的解决方案包括将值字段类型从 `V` 更改为 `Optional[V]`。

- **GitHub issue 和 PR 讨论**：用户讨论了几个 GitHub issue 和 PR，例如关于 "while 循环逻辑导致段错误 (seg fault)" 的 issue [#2696](https://github.com/modularml/mojo/issues/2696)，以及将断言中的参数消息更改为仅限关键字 (keyword-only) 的 PR [#2739](https://github.com/modularml/mojo/pull/2739)。

- **5/21 nightly 版本延迟发布**：多位用户注意到 nightly 版本发布延迟，这归因于潜在的 CI 基础设施/发布问题。该问题随后得到解决，并确认 5/21 的 nightly 构建版本已可用。

- **字符串 Unicode 支持提案**：针对在字符串中实现 Unicode 支持进行了详细讨论，提出了各种内部表示形式，并辩论了空终止符 (null termination) 的权衡。其核心思想是进行激进优化，确保高效的内存使用和互操作性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/lifecycle/death#field-lifetimes))*">值的销毁 | Modular 文档</a>：关于 Mojo 何时以及如何销毁值的解释。</li><li><a href="https://peps.python.org/pep-0393/">PEP 393 – 灵活的字符串表示 | peps.python.org</a>：未找到描述</li><li><a href="https://www.githubstatus.com/">GitHub 状态</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/2696">[BUG] While 循环逻辑导致段错误 · Issue #2696 · modularml/mojo</a>：Bug 描述。此问题始于几天前的 Mojo nightly 版本。不确定具体是哪一个。在下方代码中，函数 app_run 导致段错误，但单独的 app_close 可以编译...</li><li><a href="https://github.com/modularml/mojo/pull/2771">[stdlib] 由 rd4com 提交的为 StringLiteral 添加 format_simple() · Pull Request #2771 · modularml/mojo</a>：为 #2761 提供了一个“小”修复。它不是非常高级，只是提供一个有用的小功能："{name} is awesome {emoji}".format_simple(name="Mojo", emoji="🔥")...</li><li><a href="https://github.com/modularml/mojo/pull/2739">[stdlib] Issue #2487：由 softmaxer 提交的将 assert_true/assert_false/... 中的参数 msg 更改为仅限关键字 · Pull Request #2739 · modularml/mojo</a>：变更内容：在 stdlib/src/testing/testing.mojo 的函数定义中添加 *，以区分变长参数和仅限关键字参数。扫描这些断言函数的调用点并替换 assert_true(val...</li><li><a href="https://github.com/modularml/mojo/pull/2613#discussion_r1599235527">[stdlib] 由 gabrieldemarmiesse 提交的在 `List` 中添加可选的小缓冲区优化 (SSO) · Pull Request #2613 · modularml/mojo</a>：与 #2467 相关。这是 SSO 的开发工作。我正在尝试一些方案，并希望收集社区反馈。起初，我想使用 Variant[InlineList, List] 来实现 SSO，虽然那样可以...
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1242220990331621426)** (5 条消息): 

- **关于 ML 工作负载的托管 Kubernetes 的辩论**：一位成员质疑管理用于 ML 推理的本地服务器的必要性，建议使用 **EKS 等托管 Kubernetes 服务** 作为替代方案。他们对扩展 Web 服务器和 ML 任务之间感知的差异表示困惑，除了**偶尔对 GPU 的需求**之外。

- **Kubernetes 对 ML 基础设施并非必不可少**：有人澄清说 **Kubernetes** 主要用于基础设施目的，并非天生与 ML 工作绑定。选择是否使用 Kubernetes 取决于**具体的项目需求**。
  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1242011136090701824)** (13 messages🔥): 

- **硬件与 GPU kernel 的细节差异**：最大 block size 受硬件、kernel 特性以及 dtype 的影响，因为每个线程会加载多个元素，以有效利用 GPU 上的向量指令。
- **CUDA 调度原理依然适用**：Block 被调度到一个 SM 并在 block 内共享内存，这与 CUDA 类似，确保了 GPU 处理的一致性。
- **团队赞誉与建议**：Byronhsu1230 对 Horace 提供的丰富信息表示感谢，并建议需要一篇关于 Triton 编译器的文章。团队非常欣赏 Horace 提供的宝贵见解。
- **改进 Triton 教程**：Lancerts 分享了一个 [GitHub pull request](https://github.com/triton-lang/triton/pull/3959)，详细介绍了对 Triton 教程进行的细微修改，以提高可读性和一致性，这些修改已在 GPU 上测试并成功运行。

**提到的链接**：<a href="https://github.com/triton-lang/triton/pull/3959">lancerts 对 tutorial5 的小幅重构以及对 tutorial1 的小幅修改 · Pull Request #3959 · triton-lang/triton</a>：更改已在 GPU 上测试，执行结果一致。在 tutorial 1 中，将 gbps = lambda ms: 12 * size / ms * 1e-6 修改为 gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6。这是 m...

  

---


### **CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1242185192727253172)** (2 messages): 

- **寻求 SASS 论文**：一位成员询问是否有人推荐与 **SASS** 相关的论文。该查询非常直接，旨在寻找学术资源。
- **关于 cucomplex 与 cuda::std::complex 的讨论**：另一位成员针对 **Volta, Ampere 和 Hopper** 架构，讨论了在原子操作中使用 "cucomplex" 或 "cuda::std::complex" 的问题。他们就哪种方式更适合其需求（特别是针对 x 和 y 的 atomic add 操作）寻求建议。
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1242026656122798100)** (21 messages🔥): 

- **Torch 原生乘法导致内存占用翻倍**：一位成员注意到 **Torch** 中的原生 `*` 运算符似乎会使内存翻倍，即使是原地执行也是如此。在检查该问题后，他们发现使用 `mul_()` 可以解决此问题并实现平稳的内存消耗。

- **`torch.empty_like` 与 `torch.empty` 的性能差异**：一位用户分享了一个 **PSA**，强调 `torch.empty_like` 比 `torch.empty` 快得多，同样地，`torch.empty(..., device='cuda')` 的性能优于 `torch.empty(...).to('cuda')`。另一位用户确认这种行为在 **NumPy** 中也存在，特别是 `np.zeros_like`。
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1242221629971370124)** (2 messages): 

```html
- **Member finds the discussion amazing**: One member described the talk as *"amazing."* 
- **Clarification requested**: Another member asked for elaboration on why the talk was considered *"amazing."*
```
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1242124174332395561)** (3 messages): 

- **专注于 CUDA 的激活量化**："我们的首要重点是激活量化 (fp8/int8)。" 一位成员讨论了需要通过 **Cutlass epilogue fusion** 将 GEMMs 周围的小操作进行融合，以实现推理加速。
- **利用下一代 GPU 特性**：该成员强调了在新型 GPU 中使用 **2:4 sparsity 和 fp6/fp4** 的计划。
- **Torch.compile 后端开发**：团队正在为 **torch.compile** 开发用户自定义后端，以实现图级优化并通过更多融合来提高性能。
- **未充分优化的 vLLM 组件**：确定了 **MoE kernel 和 sampling kernels** 是 vLLM 中未充分优化的区域，也是目前的优先事项。
- **LinkedIn 提供协助**：另一位来自 LinkedIn 的成员表示有兴趣合作，并询问了关于 "图级优化" 的细节。
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

norton1971: 有人能帮帮我吗？
  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1242263465977053298)** (1 条消息): 

- **torchao 0.2 版本发布备受关注**：torchao 0.2 的新版本现已在 [GitHub](https://github.com/pytorch/ao/releases/tag/v0.2.0) 上发布。它的特点是包含支持二进制文件的自定义 CUDA 和 CPU 扩展，以及其他增强功能。
- **自定义扩展实战**：一位成员使用新版本设置了一些 **fp6** kernel。这突显了新的自定义算子（custom op）注册机制所提供的灵活性和可扩展性。
- **高速 kernel 已合并**：另一位成员合并了针对 GaLoRe、DoRA 和 int4/fp16 的高速 kernel。这些改进旨在提升性能和效率。
- **NF4 tensor 与 FSDP 的兼容性**：在之前工作的基础上，此版本支持可与 **FSDP** 组合使用的 **NF4 tensor**。官方提供了一份将更小的数据类型（dtypes）与 FSDP 集成的详细蓝图，以确保更好的资源利用率。

**提及的链接**：<a href="https://github.com/pytorch/ao/releases/tag/v0.2.0">Release v0.2.0 · pytorch/ao</a>：更新亮点包括用于分发 CPU/CUDA 二进制文件的自定义 CPU/CUDA 扩展。PyTorch 核心最近发布了一个带有 torch.library 的新自定义算子注册机制，其优势在于...

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

iron_bound: 光线投射（Ray casting） https://frankforce.com/city-in-a-bottle-a-256-byte-raycasting-system/
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1242025337848856596)** (193 条消息🔥🔥): 

```html
- **关于移动边界检查的辩论**：成员们讨论了是否将边界检查（bounds checks）从 kernel 内部移至 assert 中，并对性能影响表示担忧。有人提到，“为了性能，通常应该关闭 assert”，并指出了潜在的隐藏维度约束问题。

- **GPT-2 复现的阻碍因素**：一位成员列出了阻碍 GPT-2 复现的剩余任务，包括初始化、权重衰减（weight decay）管理和学习率调度（learning rate schedules）。Checkpoint 的保存与加载功能被强调为必不可少。

- **DataLoader 重构提案**：一位成员概述了对 DataLoader 的重构，旨在引入新功能，如规范的 .bin 标头、uint16 数据存储和数据集分片（sharding）。目标是改进对 FineWeb 等大型数据集的数据处理。

- **关于 CI 兼容性的讨论**：成员们讨论了确保 fp32.cu 文件与旧版本 CUDA 的兼容性，建议包含 C11 和 C++14 标准。他们强调使用旧版本 CUDA 进行测试以发现问题。

- **数据集重构合并**：DataLoader 的重构已合并至 master 分支，导致了破坏性变更（breaking changes）。一位成员建议，拉取更改会破坏当前的实现，并建议重新运行数据预处理脚本以解决问题。
```
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/439">修复 matmul_backward_bias kernel 1 中不支持的 block_size | lancerts 提交的 Pull Request #439 · karpathy/llm.c</a>: 由于 https://github.com/karpathy/llm.c/blob/master/dev/cuda/matmul_backward_bias.cu#L67 中的 reduction 操作，kernel 1 的 block size 需要是 2 的幂。否则 GPU 结果...</li><li><a href="https://github.com/karpathy/llm.c/pull/442">完全确定性的 encoder backward kernel | ademeure 提交的 Pull Request #442 · karpathy/llm.c</a>: 这是对 encoder backward pass 的完全重写，将其拆分为两个 kernel（wte 和 wpe），两者都是完全确定性的，因为它们不使用 atomics（假设随机种子的种子...</li><li><a href="https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md">cudnn-frontend/docs/operations/Attention.md at main · NVIDIA/cudnn-frontend</a>: cudnn_frontend 提供了 cudnn backend API 的 C++ 封装以及如何使用它的示例 - NVIDIA/cudnn-frontend</li><li><a href="https://github.com/karpathy/llm.c/pull/440">重构数据集 | karpathy 提交的 Pull Request #440 · karpathy/llm.c</a>: 重构我们处理数据集的方式，因为我们即将拥有更多数据集，且不希望它们堆满根目录等。这只是第一步，我正准备重构一系列 datalo...</li><li><a href="https://github.com/karpathy/llm.c/discussions/84#discussioncomment-9486746)">llm.c 讨论区 · karpathy/llm.c · Discussion #84</a>: 🔥 llm.c 🔥 开启讨论功能，作为一个供人们提问、分享和交流的地方，而无需创建 Issue。</li><li><a href="https://github.com/karpathy/llm.c/pull/427/files)">权重重排：尝试 1 | ngc92 提交的 Pull Request #427 · karpathy/llm.c</a>: 非功能性尝试，探索如何按 block 布局重新排列权重。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1242034953424736316)** (11 条消息🔥): 

- **预分配 Tensor 以加速解包**：建议避免使用 `torch.stack`，而是使用 `torch.empty` 预分配 Tensor，以便在通过 `torch.compile` 编译时实现更快的解包（unpacking）。分享了一个从 uint8 格式解包的示例代码，强调了这种方法。
  
- **在 torchao uint4 中实现更改**：Vayuda 建议更新 torchao uint4 的实现，以反映所提议的预分配优化。Coffeevampir3 已确认并分享了一个包含相关更改的 [GitHub notebook](https://github.com/CoffeeVampir3/ao-bitnet/blob/main/bitnet-testing.ipynb)。

- **优化解包代码**：Mobicham 指出了一个额外的优化点，即在解包函数中移除不必要的 uint8 类型转换。Coffeevampir3 在更新后的代码示例中采纳了这一反馈。

- **确保数值正确性和效率**：Coffeevampir3 建议通过添加偏移（shift）来正确处理无符号整数的 Tensor 数据打包和解包。该方法已通过量化过程的示例调整得到验证。

- **为自定义算子使用 `opcheck()`**：Vayuda 提醒使用 `opcheck()` 来确保自定义算子（custom ops）满足各种要求，并暗示需要在 `__torch_dispatch__` 中实现必要的函数。他们询问是否存在基于使用场景的函数列表。
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1242019666847862815)** (127 条消息🔥🔥): 

- **AI Safety Institute 在旧金山开设新办公室并提供更高薪资**：英国 AISI 宣布在旧金山开设办公室，其薪资水平较伦敦办公室有所上调。他们正在积极寻求人才并与加拿大合作，详见[此合作伙伴关系公告](https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership)。

- **关于 OpenAI 员工流动和 AISI 招聘的讨论**：几位成员推测前 OpenAI 对齐（aligners）人员是否加入了新的 AISI。讨论中涉及了对加拿大办公室开设的兴趣以及录用标准，并指向了英国-加拿大 AI 安全合作伙伴关系的[详情](https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership)。

- **语言模型中 Dropout 的评估**：成员们辩论了在现代语言模型中使用 Dropout 的相关性，一些人对其目前的用法表示困惑。为了缓解 Overfitting（过拟合），还考虑了 Label Smoothing 等替代策略。

- **关于加州 SB 1047 对 AI 发展影响的公益公告**：呼吁通过立法参与来反对加州的 SB 1047 法案。正如[此分析](https://context.fund/policy/sb_1047_analysis.html)所述，该法案可能通过引入缺乏问责制的监管措施，并对开发者处以潜在的监禁，从而严重影响开源 AI。

- **AI 模型开发的工具与技术分享**：成员们分享了关于 JAX 中 Flash Attention 实现及其相比原生实现性能提升的资源和见解。这包括指向对话引用的链接，如 [JAX 中的 Flash Attention](https://github.com/nshepperd/flash_attn_jax) 及相关的性能基准测试。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.aisi.gov.uk/">The AI Safety Institute (AISI)</a>: AI Safety Institute 是科学、创新和技术部的一个部门，旨在促进严谨的研究以实现先进的 AI 治理。</li><li><a href="https://affuture.org/post/9-context/">Call-To-Action on SB 1047</a>: 在 Effective Altruism 活动人士的影响下，加州立法者正试图通过一项对开源 AI 和整个科技行业来说都是灾难性的法案。SB 1047 创造了一个...</li><li><a href="https://github.com/huggingface/transformers/issues/30810">tracker: `generate` composability refactor  · Issue #30810 · huggingface/transformers</a>: generate + 可组合性 = 以最少的重写实现更多用例。在我撰写此 issue 时，generate 基本上是一个顺序单体。在过去两年中，许多内部模块被拆分为函数...</li><li><a href="https://github.com/nshepperd/flash_attn_jax">GitHub - nshepperd/flash_attn_jax: JAX bindings for Flash Attention v2</a>: Flash Attention v2 的 JAX 绑定。通过在 GitHub 上创建账号来为 nshepperd/flash_attn_jax 的开发做出贡献。</li><li><a href="https://www.gov.uk/government/publications/uk-canada-science-of-ai-safety-partnership/uk-canada-science-of-ai-safety-partnership">UK-Canada science of AI safety partnership</a>: 英国-加拿大 AI 安全科学合作伙伴关系。</li><li><a href="https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py">jax/jax/experimental/pallas/ops/tpu/flash_attention.py at main · google/jax</a>: Python+NumPy 程序的可组合转换：微分、向量化、JIT 到 GPU/TPU 等 - google/jax
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1242119063053013103)** (72 messages🔥🔥): 

```html
- **探讨 CLIP 中的多模态训练**：讨论重点在于使用音频等额外模态训练 CLIP 是否能提高 zero-shot ImageNet 分类性能。提到了 [ImageBind](https://arxiv.org/abs/2305.05665)，该研究展示了使用组合 Embedding 在跨模态检索方面的改进，但未涉及非涌现能力的提升。
  
- **GPT-3 在 Temperature 0 下的非确定性**：针对 GPT-3 即使在 Temperature 为 0 时也表现出非确定性行为的疑问，分享了几篇论文和资源，包括[一篇关于 Mixture of Experts 攻击的论文](https://arxiv.org/abs/2402.05526)以及关于分布式系统中一致性哈希溢出的讨论。

- **自我意识模拟（Simulacra）能力**：用户分享了关于语言模型意识到其虚构身份的经历，以及这对其后续行为的影响。共识是，像 Llama 2 70b 和自定义微调模型这样的大型模型，在逐步引导下可以表现出细微的理解和适应能力。

- **多模态学习中的正向迁移**：辩论了多模态训练对单模态任务的潜在益处，并引用了 Gato 和 PaLM-E 等模型，这些模型在任务之间展示了“正向迁移（positive transfer）”，表明额外模态确实可能增强任务性能。
  
- **使用 MegaBlocks 进行高效 MoE 训练**：介绍了 [MegaBlocks](https://arxiv.org/abs/2211.15841) 系统，强调其通过使用块稀疏操作重新制定 MoE 计算，从而避免 Token 丢弃的能力，在不损失模型质量的情况下实现了显著的训练效率提升。
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2211.15841">MegaBlocks: Efficient Sparse Training with Mixture-of-Experts</a>：我们介绍了 MegaBlocks，一个用于在 GPU 上高效训练 Mixture-of-Experts (MoE) 的系统。我们的系统针对当前框架的局限性而设计，这些局限性限制了 MoE 层中的动态路由...</li><li><a href="https://arxiv.org/abs/2402.05526">Buffer Overflow in Mixture of Experts</a>：Mixture of Experts (MoE) 已成为扩展大型基础模型同时保持推理成本稳定的关键要素。我们展示了具有跨 Batch 依赖性的 Expert 路由策略...</li><li><a href="https://arxiv.org/abs/2305.05665">ImageBind: One Embedding Space To Bind Them All</a>：我们提出了 ImageBind，一种学习跨六种不同模态（图像、文本、音频、深度、热能和 IMU 数据）联合 Embedding 的方法。我们展示了并非所有配对数据的组合都是必需的...</li><li><a href="https://arxiv.org/abs/2205.06175">A Generalist Agent</a>：受大规模语言建模进展的启发，我们采用类似的方法构建了一个超越文本输出领域的单一通用 Agent。该 Agent 被称为 Gato...</li><li><a href="https://arxiv.org/abs/2303.03378">PaLM-E: An Embodied Multimodal Language Model</a>：大型语言模型擅长处理各种复杂任务。然而，在现实世界中实现通用推理（例如机器人问题）提出了 Grounding 的挑战。我们提出了具身...</li><li><a href="https://community.openai.com/t/run-same-query-many-times-different-results/140588">Run same query many times - different results</a>：我想知道为什么连续多次运行相同的 Prompt 会得到不同的结果。我在很多实验中注意到，如果你在两次运行之间设置冷却时间...</li><li><a href="https://152334h.github.io/blog/non-determinism-in-gpt-4/">Non-determinism in GPT-4 is caused by Sparse MoE</a>：目前众所周知，即使在 temperature=0.0 时，GPT-4/GPT-3.5-turbo 也是非确定性的。如果你习惯了 Dense Decoder-only 模型，这是一种奇怪的行为，因为在这些模型中 temp=0 应该...</li><li><a href="https://rmarcus.info/blog/2018/09/14/consistent-hashing-overflow.html">
      
      一致性哈希中的溢出 &middot; Ryan Marcus
      
    </a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1242014385501634580)** (12 messages🔥): 

- **新论文提出替代的 Scaling Laws**：分享了一篇 [arXiv 上的新论文](https://arxiv.org/abs/2405.10938) 链接，该论文提出了一种通过约 80 个公开模型构建 Scaling Laws 的观测方法，绕过了在多个尺度上训练模型的需要。论文假设语言模型的性能是一个低维能力空间的函数，其中不同家族的模型在训练计算效率上有所不同。

- **关于 Attention 机制的 FLOPs 计算讨论**：针对如何计算 Attention 机制前向和后向传递的 FLOPs 进行了详细讨论，引用了多处参考资料，如 [PALM 论文解释](https://link.to.paper) 和 [EleutherAI cookbook](https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py)。成员们明确了在 FLOPs 计算中应包含 QKVO 投影。

- **对 Bitnet 架构最优 Scaling Laws 的质疑**：一位成员思考 Chinchilla Scaling Laws 是否会建议在使用显著更少计算量的 Bitnet 时，采用更高或更低的参数与 Token 比例。另一位成员建议，如果计算速度神奇地变快，Scaling Laws 可能会保持不变，但由于计算预算增加，将允许使用更大的模型。

- **与 Scaling Laws 相关的样本效率**：样本效率（Sample efficiency）的定义和衡量被认为是理解 Scaling Laws 的关键。讨论集中在资源管理应如何随数据集增长而调整，暗示高效的缩放是资源利用的关键。

- **在小数据集上训练难度的认知**：一位成员澄清说，在小数据集上训练通常不如在大数据集上进行预训练再进行微调有效，暗示弥合这一性能差距具有挑战性。这是背景是关于小数据集训练“极其困难”的普遍观点。

**提到的链接**：<a href="https://arxiv.org/abs/2405.10938">Observational Scaling Laws and the Predictability of Language Model Performance</a>：了解语言模型性能如何随规模变化对于基准测试和算法开发至关重要。Scaling Laws 是构建这种理解的一种方法，但其要求……

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1242533689271779329)** (1 messages): 

- **Anthropic 在可解释特征方面的工作令人振奋**：一位成员分享了他们对 Anthropic 最近关于 Transformer 中可解释特征（interpretable features）工作的热情。他们提供了[研究出版物](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)的链接以供进一步阅读。
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1242034152828829747)** (30 messages🔥): 

- **评估的 Batch Size 技巧**：成员们讨论了 `--batch_size` 参数的设置，指出可以将其设置为正整数或 "auto" 以优化内存使用。有人建议使用 "auto:N" 在评估期间多次动态重新选择最大 Batch Size，这有助于加快进程。

- **翻译评估集的命名规范**：一位用户询问了机器翻译的 ARC Challenge 评估集的命名规范。建议包括 `arc_challenge_mt_language` 或 `mt_arc_challenge_language` 等名称。

- **没有专门的 AI Safety 活动频道**：有人询问是否有用于推广 AI Safety/基准测试活动的频道。确认 EleutherAI 没有此类专用频道。

- **对基准测试答案随机化的担忧**：用户讨论了如果答案选项不随机化，多选题（MCQs）中潜在的偏见。提到对于 SciQ，随机化并不重要，因为选项不在 Context 中，但对于 MMLU，它是相关的，尽管目前尚未实现。

- **对医学基准测试的担忧**：一位成员分享了他们对医学基准测试可能产生危害的关注，强调了改进基准测试解释的重要性。大家对即将开展的相关工作感到兴奋，包括 Pile 数据集的更新和关于基于种族的医学论文。

**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/1710b42d52d0f327cb0eb3cb1bfbbeca992836ca/lm_eval/tasks/sciq/sciq.yaml#L11">lm-evaluation-harness/lm_eval/tasks/sciq/sciq.yaml at 1710b42d52d0f327cb0eb3cb1bfbbeca992836ca · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

  

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1242065231459385458)** (7 messages): 

```html
<ul>
    <li><strong>Temporal.io 胜出</strong>：一位成员询问了关于 Airflow 和 Temporal.io 的使用经验，最终决定选择 <strong>Temporal</strong>。</li>
    <li><strong>Manifold Research Group 更新</strong>：来自 <strong>Manifold Research Group</strong> 的一位成员分享了他们的<a href="https://www.manifoldrg.com/research-log-038/">最新研究日志</a>，详细介绍了 NEKO Project 等项目的进展，该项目旨在构建大规模开源“Generalist”模型。他们正在扩大团队，并邀请其他人通过 Discord 或 GitHub 加入。</li>
    <li><strong>虚构文明模拟</strong>：分享了一个 <a href="https://websim.ai/">Websim</a> 项目的链接，该项目模拟了黑海沿岸古代安纳托利亚的一个虚构文明。</li>
    <li><strong>LLM 课程发布</strong>：分享了一门新课程“通过项目式学习应用大语言模型 (LLMs)”的详细信息，重点关注实际应用，如语义电影搜索、用于食物推荐的 RAG，以及使用 LLM 创建软件和网站。感兴趣的成员可以私信了解更多信息。</li>
</ul>
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.manifoldrg.com/research-log-038/">研究日志 #038</a>：欢迎阅读研究日志 #038！我们记录了 Manifold Research Group 各项计划的每周研究进展，并重点介绍了我们认为来自更广泛研究社区的突破...</li><li><a href="https://websim.ai/c/i4l0yMB06Ie8AI3BG">赫斯珀里亚历史 (公元前 3000 年 - 公元 1460 年) - 维基百科</a>：未找到描述</li><li><a href="https://websim.ai/c/Eh7h07aUo3LsEGeh6">Kidin-Erra - 维基百科</a>：未找到描述</li><li><a href="https://websim.ai/c/jfdPjPqRWqsXoUow2)">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

mautonomy: https://fxtwitter.com/vikhyatk/status/1792512588431159480?s=19
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1242012707075522601)** (172 messages🔥🔥): 

- **Yi-1.5 Context 版本发布**：宣布发布 Yi-1.5 的 16k 和 32k Context 版本，并附带 [Hugging Face 链接](https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8)。这些版本满足了不同的 Context 长度需求，可能会影响模型性能。

- **LLM Leaderboard 批评**：成员们批评了 LLM Leaderboard 的实用性，称其由于过多的噪音和难以过滤相关模型而“官方性地不可用”。LLM Leaderboard 充斥着大量条目，导致难以辨别质量排名。

- **Chatbot Arena 的客观性受到质疑**：人们对 Chatbot Arena 评分的客观性表示担忧，特别是用户偏好倾向于简单、易于验证的测试。该平台引入了 “Hard Prompts” 类别来解决这些偏见，详见其 [博客文章](https://lmsys.org/blog/2024-05-17-category-hard/)。

- **微软 AI 发布会**：成员们讨论了最近的微软发布会，展示了全新的 Copilot+ PC，录像已上传至 [YouTube](https://www.youtube.com/watch?v=aZbHd4suAnQ)。该活动备受期待但未进行直播，引发了关于通过观看回放获取详细见解的评论。

- **提及 Qwen MoE 模型**：一位成员强调了 Qwen 发布的 MoE 模型，该模型总参数量为 140 亿，但在运行时仅有 27 亿激活参数，命名为 [Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)。与他们的 7B 模型相比，该模型在 Inference 期间的性能提升了 1.75 倍。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://lmsys.org/blog/2024-05-17-category-hard/">在 Chatbot Arena 中引入 Hard Prompts 类别 | LMSYS Org</a>: &lt;h3&gt;&lt;a id=&quot;background&quot; class=&quot;anchor&quot; href=&quot;#background&quot; aria-hidden=&quot;true&quot;&gt;&lt;svg aria-hidden=&quot;true&quot; class=&quot;octicon octicon-link&qu...</li><li><a href="https://huggingface.co/blog/maywell/llm-feature-transfer">一键扩展模型 Context 并创建 Chat 模型</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8">Yi-1.5 (2024/05) - 01-ai 集合</a>: 未找到描述</li><li><a href="https://lmsys.org/blog/2024-04-19-arena-hard/">从实时数据到高质量基准测试：Arena-Hard 流水线 | LMSYS Org</a>: &lt;p&gt;为 LLM 聊天机器人构建一个负担得起且可靠的基准测试已成为一项关键挑战。高质量的基准测试应该 1) 稳健地分离模型...</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat">Qwen/Qwen1.5-MoE-A2.7B-Chat · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/bobbyallyn/status/1792679435701014908?s=46">Bobby Allyn (@BobbyAllyn) 的推文</a>: Scarlett Johansson 关于 OpenAI 情况的声明。哇：</li><li><a href="https://youtu.be/jcvatirXHXU?si=-zJBGCohaoKFvOkw">通过 Microsoft CoPilot AI 提前体验 GPT-4o 语音与视觉功能！</a>: 微软的 AI 活动 Microsoft Build 揭晓了关于 Copilot 和 GPT-4o 的激动人心的更新。虽然没有直播，但细节迅速传开。值得注意的是， GPT-4o...</li><li><a href="https://www.youtube.com/watch?v=aZbHd4suAnQ">完整主题演讲：介绍 Copilot+ PC</a>: Copilot+ PC 是有史以来速度最快、最智能、续航最长的 Windows PC。在此订阅微软 YouTube 频道：https://aka.ms/SubscribeT...</li><li><a href="https://github.com/huggingface/datatrove">GitHub - huggingface/datatrove: 通过提供一套与平台无关的可定制流水线处理块，将数据处理从脚本疯狂中解放出来。</a>: 将数据处理从脚本疯狂中解放出来。 - huggingface/datatrove
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1242166807222292683)** (1 messages): 

- **难以找到用于 Reranker 基准测试的公开评估**：一位成员表示，很难为他们制作的 Finetuned Reranker 找到公开评估。他们观察到其他 Reranker 使用了各种数据集，但对具体的查询和 Benchmark 方法仍感到困惑。
  

---

### **Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1242514928150249573)** (1 条消息): 

- **Phi-3 Vision 发布：** 一位成员分享了 [Phi-3 Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) 现已可用，并将其描述为一个**轻量级、最先进的开源多模态模型**，具有 128K token 的上下文长度。**它专注于来自文本和视觉来源的高质量、推理密集型数据**，并使用监督微调（supervised fine-tuning）和直接偏好优化（direct preference optimization）来增强指令遵循能力和安全性。
- **探索 Phi-3 Vision 资源：** 关键资源包括 [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024)、[Phi-3 Technical Report](https://aka.ms/phi3-tech-report) 以及 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook)。此外还有一个指向 [Azure AI Studio 上的 Phi-3](https://aka.ms/try-phi3vision) 的链接，用于实际部署。

**提及的链接**：<a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>：未找到描述

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1242101350473404417)** (20 条消息🔥): 

- **CLI 提示词创建搞怪图像**：成员们欣赏了一些使用 CLI 提示词生成的搞怪图像，其中一位成员非常喜欢生成图像中的猫。**ASCII 艺术**也受到了关注，图像分享在[这里](https://www.bing.com/images/create/ascii-art-of-a-dream-like-simulation-framework-wit/1-664ba0e66d06426b8d19b219b95859bf?id=FriXjEz08JVgjTHO3NCJ6w%3d%3d&view=detailv2&idpp=genimg&idpclose=1&thId=OIG3.s.gLVkq2qEbWmuzpYHrU&frame=sydedg&FORM=SYDBIC)。

- **WorldSim 的潜力**：参与者讨论了 WorldSim 演变为**全球智能平台**和一种新型协作思考形式的潜力。有人评论称其有潜力成为“世界上最智能的玩具”，能够培养一种新的全球思维状态，并建议再举行一次讨论会以进一步探索这些想法。

- **符号意义知识图谱**：受 **Tek 的映射（mapping）** 启发，成员们考虑了 AI 框架内的符号意义，提到了创建入门级知识图谱，并将其视为**罗夏墨迹测试（Rorschach test）**与 AI **语义网（semantic web）**的结合体。

- **构想 WorldSim 世界**：成员们分享了代表构想中 WorldSim 世界的生成图像，灵感源自多样化的建筑风格和景观。这些作品可以在[这里](https://copilot.microsoft.com/images/create/a-palace-with-indonesian-and-south-indian-architec/1-664bded7a83d47099a43b3bff31da0ff?id=1G89%2bELerXo%2felOtiC%2foQw%3d%3d&view=detailv2&idpp=genimg&idpclose=1&thId=OIG4.68GjfK8kliVnM3sKbk_b&lng=en-US&ineditshare=1)和[这里](https://copilot.microsoft.com/images/create/a-small-mountainous-island-with-southern-african-a/1-664bdf507c354bcbad606c5b223eb24d?id=6bFJfczZZ4CJykjB90rkYQ%3d%3d&view=detailv2&idpp=genimg&idpclose=1&thId=OIG2.L8Gf2dcTsVepM.tkLuzx&lng=en-US&ineditshare=1)找到。

- **Obsidian 知识图谱延时摄影**：一位成员分享了一段**令人印象深刻的延时摄影**，展示了一位 Obsidian 用户的知识图谱形成过程，将其描述为一件类似于运行中的合成大脑的艺术品。在 [YouTube](https://youtube.com/shorts/4YQhH61tvOc?si=0Dx1KyJP8VMz-pXY) 上可以观看该延时摄影。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://cdixon.org/2010/01/03/the-next-big-thing-will-start-out-looking-like-a-toy">下一件大事最初看起来会像个玩具</a>：Chris Dixon 的博客。</li><li><a href="https://youtube.com/shorts/4YQhH61tvOc?si=0Dx1KyJP8VMz-pXY">我的 Obsidian 图谱延时摄影：2 年浓缩为 30 秒</a>：这是一段关于我的 Obsidian 笔记库如何在 2 年多时间里缓慢增长到 8,000 多条笔记的延时摄影。---// 关于我 网站: https://nicolevanderhoeven.com Mastodon: https...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1242122372488888340)** (105 条消息🔥🔥): 

- **OpenAI 因用户疑虑暂停 Sky 语音**：OpenAI 的一份状态更新回应了关于 ChatGPT 语音选择的疑虑，特别是 Sky 语音。在处理这些问题的同时，他们正在*暂停使用 Sky 语音* [来源](https://x.com/OpenAI/status/1792443575839678909)。

- **CogVLM2 模型凭借关键特性备受关注**：社区对 [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B) 的发布表现出极高热情，重点强调了其 **8K 内容长度支持**以及在 `TextVQA` 等基准测试中显著提升的性能。

- **对 Copilot AI 进展的反应不一**：Mustafa Suleyman 宣布了下一代 Copilot，它可以*“实时看、听、说并提供帮助”*，这引发了好奇与质疑。一些用户觉得这*令人毛骨悚然*，而另一些人则开玩笑说，这可能会变成一个对所有操作指手画脚的“游戏旁观者”版本 [来源](https://fxtwitter.com/mustafasuleyman/status/1792623877744623806)。

- **OpenAI 的 Scarlett Johansson 语音争议**：成员们讨论了 OpenAI 语音助手据称模仿电影《Her》中 Scarlett Johansson 声音的伦理和法律影响。大家一致认为，在联系 Johansson 后又复制她的声音导致了严重的舆论反弹和*“冒充（passing off）”*的指控。

- **Sakuga-42M 数据集下架原因查明**：Sakuga-42M 数据集从 Hugging Face 和 GitHub 下架归因于网站因下载量过大而采取的反机器人措施。这引发了关于在高流量压力下维护开源数据集难度的讨论 [来源](https://news.ycombinator.com/item?id=40389711)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/_ebehrens_/status/1792569302773555250">来自 Eva Behrens (@_ebehrens_) 的推文</a>：这是我和我在 ICFG 的同事为即将到来的首尔 AI 安全峰会提出的 5 项政策建议。在布莱切利，世界领导人讨论了前沿 AI 发展的主要风险。在首尔...</li><li><a href="https://arxiv.org/html/2405.07425v1">Sakuga-42M 数据集：扩大动画研究规模</a>：未找到描述</li><li><a href="https://fxtwitter.com/mustafasuleyman/status/1792623877744623806?t=t5EX1E--TJ-mAJJZtzX4eg&s=19">来自 Mustafa Suleyman (@mustafasuleyman) 的推文</a>：我们正在将 Copilot 提升到新的水平。🚀 Copilot 将实时看、听、说并提供帮助。观看此演示以了解我的意思。很快，你的 AI 伴侣将开始与你一起生活，无论...</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B">THUDM/cogvlm2-llama3-chat-19B · Hugging Face</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=40389711">未找到标题</a>：未找到描述</li><li><a href="https://x.com/OpenAI/status/1792443575839678909">来自 OpenAI (@OpenAI) 的推文</a>：我们收到了关于我们如何选择 ChatGPT 语音（尤其是 Sky）的问题。在处理这些问题期间，我们正致力于暂停使用 Sky。阅读更多关于我们如何选择这些语音的信息：https://openai...</li><li><a href="https://forum.effectivealtruism.org/posts/twMs8xsgwnYvaowWX/database-of-orgs-relevant-to-longtermist-x-risk-work>">与长期主义/生存风险工作相关的组织数据库 — EA 论坛</a>：这是该数据库的一个版本，你可以根据需要进行筛选和排序，这里是一个你可以添加评论的版本。…
</li>
</ul>

</div>

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1242140234423210035)** (24 条消息🔥): 

- **CogVLM2 许可证限制**：一位用户警告 CogVLM2 新许可证中的限制性条款，这些条款禁止可能损害中国国家安全或公共利益的使用。该许可证和争议解决受中国司法管辖，引发了对“虚假开源”以及许可证条款中潜在恶意行为的担忧。[GitHub 上的 CogVLM2 许可证](https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE)

- **Mamba 架构在视觉领域表现平平**：最近的一篇 arXiv 论文讨论了带有 SSM token mixer 的 Mamba 架构，并得出结论认为它对于图像分类任务并不理想。该研究引入了 MambaOut 模型，在图像分类方面表现更好，但强调了 Mamba 在长序列视觉任务中的潜力。[arXiv 上的 Mamba 论文](https://arxiv.org/abs/2405.07992)

- **基于字符的 Embeddings 实验**：一位用户描述了一个实验，将句子 Embeddings 转换为字符串，并将其输入到一个小型 LLM (Smol 101M) 中，用于 MS COCO 标题预测。该方法在 Colab T4 实例上实现，生成了“有点相关”的标题，表明其在廉价标注或概念验证方面具有潜在用途。

- **关于改进模型论文的讨论**：成员们讨论了各种模型改进，引用了 Meta 的新论文，该论文延续了他们的 cm3leon 工作，并增加了用于扩展和效率的增强技巧。对话中包含了一篇近期论文的链接 [arXiv 上的 Meta 论文](https://arxiv.org/abs/2309.02591)，并与其他先进模型（如 GPT-4O）进行了比较。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.12130">MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning</a>：低秩自适应（Low-rank adaptation）是一种流行的针对大语言模型的参数高效微调方法。在本文中，我们分析了 LoRA 中实现的低秩更新的影响。我们的发现表明...</li><li><a href="https://arxiv.org/abs/2405.07992">MambaOut: Do We Really Need Mamba for Vision?</a>：Mamba 是一种带有类似 RNN 的状态空间模型 (SSM) token mixer 的架构，最近被引入以解决 Attention 机制的二次复杂度问题，并随后应用于视觉任务...</li><li><a href="https://github.com/THUDM/CogVLM2/blob/main/MODEL_LICENSE">CogVLM2/MODEL_LICENSE at main · THUDM/CogVLM2</a>：基于 Llama3-8B 的 GPT4V 级开源多模态模型 - THUDM/CogVLM2
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1242181148902293564)** (18 条消息🔥): 

```html
- **Anthropic 扩大算力规模**：[Anthropic 的最新更新](https://www.anthropic.com/news/reflections-on-our-responsible-scaling-policy)提到使用的算力是 Opus 的 4 倍，引发了对其新进展的好奇。一位用户惊叹道：“*yo what is anthropic cookin*”。

- **Arena 引入 Hard Prompts 增加难度**：[LMsysorg 引入了“Hard Prompts”类别](https://fxtwitter.com/lmsysorg/status/1792625968865026427)，以在更具挑战性的任务上评估模型，导致排名发生显著变化。例如，在这些 Hard Prompts 下，Llama-3-8B 的表现相比 GPT-4-0314 有所下降。

- **关于 Llama-3-70B-Instruct 作为评判者的争议**：[Llama-3-70B-Instruct](https://fxtwitter.com/lmsysorg/status/1792625977207468315) 被用作评判模型来对 Arena 对战中的标准进行分类，引发了对其有效性的担忧。一位用户认为它“*只是增加了噪声*”而非有用的评估，尽管训练可能会缓解这个问题。

- **视觉模型 Phi-3 Vision 亮相**：用户确认 Phi-3 Vision 是新款模型，与其前代相比体积略大。这在关于模型发布和尺寸的简短交流中得到了强调。 
```
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/aidan_mclau/status/1792610354255769919">来自 Aidan McLau (@aidan_mclau) 的推文</a>：yo what is anthropic cookin，算力是 opus 的 4 倍，太强了</li><li><a href="https://fxtwitter.com/lmsysorg/status/1792625968865026427">来自 lmsys.org (@lmsysorg) 的推文</a>：在 Arena 中引入“Hard Prompts”类别！响应社区对在更具挑战性的任务上评估模型日益增长的兴趣，我们很高兴推出新的“Hard Pr...</li><li><a href="https://fxtwitter.com/lmsysorg/status/1792625977207468315">来自 lmsys.org (@lmsysorg) 的推文</a>：我们如何分类这些标准？我们采用 Llama-3-70B-Instruct 作为评判模型，帮助我们标注超过 100 万场 Arena 对战。总体而言，我们的分析显示 Arena 用户提示词的质量...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1242009212968767528)** (31 条消息🔥): 

- **Nathan Lambert 思考是否撰写关于 OpenAI 争议的文章**：Nathan Lambert 讨论了是否要再写一篇关于 OpenAI 的文章，并表示除了说“我是对的”之外，没有太多可补充的内容。他们提议了一个标题：“OpenAI 的第二个糟糕透顶的一周”。
- **Scarlett Johansson 关于 OpenAI 的声明**：一位 Twitter 用户分享了 Scarlett Johansson 关于 OpenAI 未经许可使用与其相似声音的声明，这促使她采取了法律行动。争议的核心在于 OpenAI 被指控有意为其 "Sky" 系统模仿她的声音。
- **公众对 Sky Johansson 声音问题的反应**：Nathan Lambert 和其他人讨论了 Johansson 声明的重大影响，并将其与之前备受关注的《纽约时报》起诉 AI 发展的案件进行了比较。Nathan 反思了更广泛的影响，并提到了移除包含 Drake 等音乐人未经授权内容的类似案例。
- **OpenAI 与 Superalignment 团队的争议**：《财富》杂志的一篇文章指出，OpenAI 未能履行将其 20% 计算资源分配给 Superalignment 团队的承诺，导致了人员辞职以及对公司优先考虑产品发布而非 AI 安全的指控。在讨论中，一些人认为这一事件是预料之中的结果。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/natolambert/status/1792752615933276165">来自 Nathan Lambert (@natolambert) 的推文</a>：末日论者以暂停 Sky Johansson 声音的形式实现了他们对 AI 发展的暂停。引用 Hayden Field (@haydenfield) 的话：刚刚收到 OpenAI CEO Sam Altman 关于 Sc... 的声明。</li><li><a href="https://fortune.com/2024/05/21/openai-superalignment-20-compute-commitment-never-fulfilled-sutskever-leike-altman-brockman-murati/">OpenAI 承诺将其 20% 的算力用于应对最危险的 AI——但据消息人士称，这一承诺从未兑现</a>：消息人士称，尽管该公司的 Superalignment 团队从未接近 20% 的阈值，但其对算力的请求仍多次遭到拒绝。</li><li><a href="https://x.com/arankomatsuzaki/status/1792713233331355867">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：本质上，你的声音并不真正属于你；它属于那个声音最相似且最有钱的人。引用 OpenAI (@OpenAI) 的话：我们收到了关于我们如何选择声音的问题...</li><li><a href="https://x.com/yashar/status/1792682664845254683">来自 Yashar Ali 🐘 (@yashar) 的推文</a>：新闻：Scarlett Johansson 刚刚就 OpenAI 发表了这份声明。我已经直接向她的公关人员确认了其真实性。“去年 9 月，我收到了 Sam Altman 的邀请，他想要...”
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1242138261011562597)** (30 条消息🔥): 

- **Nathan 讨论并购买了域名 rlhfbook.com**：Nathan Lambert 考虑购买域名 rlhfbook.com，并最终以 7 美元/年的价格从 Porkbun 购得，认为这非常划算且易于持有。

- **新 AI 数据集的潜在法律风险**：一位成员幽默地警告了使用新的 AI Books4 数据集训练 LLM 可能带来的法律后果，并参考了“原始 pile 数据集”的类似情况。

- **MSFT Surface AI 因云端检查导致性能缓慢**：讨论围绕微软新的 Surface 绘图 AI 展开，尽管该功能在本地运行，但由于向云端发送安全检查而产生了延迟。一位成员引用了 Ben Thompson 的文章作为该信息的来源。

- **对前同事诚信的批评**：Nathan Lambert 批评了一位前同事在简历中误导性地声称曾与知名人士合作。他表示希望在会议上当面质问这种不诚实行为。

**提到的链接**：<a href="https://web.archive.org/web/20240519104217/https://www.reddit.com/r/datasets/comments/1cvi151/ai_books4_dataset_for_training_llms_further/">未找到标题</a>：未找到描述

  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1242131792413200474)** (9 messages🔥): 

- **OpenAI 暂停使用类 Scarlett Johansson 语音**：在引起广泛关注后，OpenAI 停止使用 Sky，这是一个听起来像 Scarlett Johansson 的语音 AI。该公司坚持认为 Sky 的声音并非模仿，而是由另一位拥有自己声音的女演员完成的，详见 [The Verge 文章](https://www.theverge.com/2024/5/20/24160621/openai-chatgpt-gpt4o-sky-scarlett-johansson-voice-assistant-her)。

- **博客影响带来的产品决策**：一位成员幽默地指出，一位首席产品负责人阅读他们的博客得到了回报，暗示这种影响可能左右了产品决策。有人猜测这是否导致了退订，但被一笑置之。

- **对 AI 实验室的批评**：[Liron Shapira 的一条推文](https://x.com/liron/status/1792649595454976123) 批评了 AI 实验室，将他们比作“负责任的成年人”，但警告说：“你们根本不知道自己在做什么，我们都会因此而死”。这引发了一些反应，但没有进一步的评论。

- **宣传幽默**：一位成员发布了“你对宣传并非免疫 (🤞)”，并分享了一个充满表情符号的回复。这种轻松的玩笑得到了大家的认可，反映了该频道的随意性质。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/liron/status/1792649595454976123">来自 Liron Shapira (@liron) 的推文</a>: AI 实验室希望我们认为他们是负责任的成年人。事实是：你们根本不知道自己在做什么，我们都会因此而死</li><li><a href="https://www.theverge.com/2024/5/20/24160621/openai-chatgpt-gpt4o-sky-scarlett-johansson-voice-assistant-her">OpenAI 为 ChatGPT 撤回类 Scarlett Johansson 语音</a>: 也许《她》(Her, 2014) 不应该成为 AI 语音功能的蓝图。
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1242188236902760629)** (78 messages🔥🔥): 

```html
<ul>
  <li><strong>Memory Tuning Explained</strong>: Sharon Zhou from Lamini introduced "Memory Tuning" as a technique to enhance LLMs' accuracy in critical domains like healthcare and finance, achieving up to <em>"no hallucinations (&lt;5%)"</em>. This method outperforms LoRA and traditional fine-tuning, and Zhou promises more details and early access soon (<a href="https://x.com/realsharonzhou/status/1792578913572429878">link tweet</a>).</li>
  <li><strong>Lawyers demand OpenAI disclose AI voice origin</strong>: Lawyers for Scarlett Johansson are asking OpenAI how it developed its latest ChatGPT voice, which has been compared to Johansson's from the movie "Her." OpenAI has paused using the voice amid public debate, as users point out the tenuous legal arguments around likeness and endorsements (<a href="https://www.npr.org/2024/05/20/1252495087/openai-pulls-ai-voice-that-was-compared-to-scarlett-johansson-in-the-movie-her">NPR article</a>).</li>
  <li><strong>Scale AI raises $1B funding</strong>: Scale AI has announced $1 billion in new funding at a $13.8 billion valuation, led by Accel with participation from prominent investors like Wellington Management and Amazon. CEO Alex Wang stated this positions Scale AI to accelerate the abundance of frontier data and aims for profitability by the end of 2024 (<a href="https://fortune.com/2024/05/21/scale-ai-funding-valuation-ceo-alexandr-wang-profitability/">Fortune article</a>).</li>
  <li><strong>MS Phi 3 Models Released</strong>: Microsoft unveiled the Phi 3 models at MS Build, touting major benchmarks such as the Medium model being competitive with Llama 3 70B and GPT 3.5. The models offer context lengths up to 128K and utilize heavily filtered and synthetic data, released under the MIT license (<a href="https://x.com/reach_vb/status/1792949163249791383">link tweet</a>).</li>
  <li><strong>Emotionally Intelligent AI from Inflection</strong>: Inflection AI's new CEO announced a focus on integrating emotional and cognitive AI abilities, with their empathetic LLM "Pi" now used by over 1 million people daily. This move is aimed at helping organizations harness AI's transformative potential (<a href="https://inflection.ai/redefining-the-future-of-ai">Inflection announcement</a>).</li>
</ul>
```
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

*   [Sharon Zhou (@realSharonZhou) 的推文](https://x.com/realSharonZhou/status/1792578913572429878)：@MichaelBiGong 正在研究如何以合理的方式解释它——基本上是我们一直在研究的 LoRA 微调的一种更好方法，因为 finetuning 很难获得超过 90%、95% 等的结果...
*   [博客](https://inflection.ai/redefining-the-future-of-ai)：重新定义 AI 的未来
*   [Sharon Zhou (@realSharonZhou) 的推文](https://x.com/realsharonzhou/status/1792576516444065967?s=46&t=PW8PiFwluc0tdmv2tOMdEg)：Hallucinations 是生产级 LLM 和 Agent 的最大阻碍之一。我们在内部以及为客户实现了无 Hallucinations (<5%)。我们已经能够微调 LLM 来 Recall 特定的...
*   [Emmanuel Ameisen (@mlpowered) 的推文](https://x.com/mlpowered/status/1792948212728524917?s=46&t=90xQ8sGy63D2OtiaoGJuww)：今天，我们宣布已在 Sonnet 上实现了 Dictionary learning，从世界上最好的模型之一中提取了数百万个特征。这是首次成功...
*   [Teknium (e/λ) (@Teknium1) 的推文](https://x.com/teknium1/status/1792640772526813679?s=46&t=90xQ8sGy63D2OtiaoGJuww)：Inflection 已死。引用 Paolo (@The_Real_Paolo)：在蓝色对勾之后，黄色的组织认证对勾也消失了，之后什么都没有，没有公告... 闻起来不妙，@inflectionAI 可能...
*   [Lamini LLM Photographic Memory Evaluation Suite | Lamini - Enterprise LLM Platform](https://www.lamini.ai/blog/lamini-llm-photographic-memory-evaluation-suite)：未找到描述
*   [undefined 的推文](https://x.com/BobbyAll)：未找到描述
*   [建立 AI 游戏工作室：我们目前的收获 - Braindump Incorporated](https://braindump.me/blog-posts/building-an-ai-game-studio)：使用 AI 创建世界和游戏
*   [Suno 已融资 1.25 亿美元，旨在打造一个人人都能创作音乐的未来](http://suno.com/blog/fundraising-announcement-may-2024)：我们的音乐家社区值得拥有最好的工具，而打造最好的工具需要最优秀的人才。我们将利用这笔资金加速产品开发并扩大我们的世界级...
*   [Bobby Allyn (@BobbyAllyn) 的推文](https://x.com/BobbyAllyn/status/1792679435701014908)：Scarlett Johansson 就 OpenAI 事件发表的声明。哇：
*   [Scarlett Johansson 表示对 ChatGPT 的新声音感到“震惊和愤怒”，该声音被拿来与电影《Her》中的声音作比较](https://www.npr.org/2024/05/20/1252495087/openai-pulls-ai-voice-that-was-compared-to-scarlett-johansson-in-the-movie-her)：Johansson 表示 OpenAI 曾多次联系她担任 ChatGPT 的配音，但她拒绝了。随后该公司发布了一个听起来与她极其相似的语音助手。
*   [Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1792936647665107108?s=46&t=90xQ8sGy63D2OtiaoGJuww)：我们新的 Interpretability 论文首次详细展示了前沿 LLM 的内部情况，并包含了一些精彩的故事。我想分享其中两个自阅读以来一直让我印象深刻的故事。背景是...
*   [独家：Scale AI 获得 10 亿美元融资，估值达 140 亿美元，其 CEO 预测到年底收入将大幅增长并实现盈利](https://fortune.com/2024/05/21/scale-ai-funding-valuation-ceo-alexandr-wang-profitability/)：Scale AI 帮助公司为 AI 模型训练标注和测试数据，已完成 10 亿美元的新一轮融资，估值达 140 亿美元。
*   [Dan Siroker (@dsiroker) 的推文](https://x.com/dsiroker/status/1792956339515273537)：很多人问我关于 Microsoft Recall 的看法，所以这是我的观点！
*   [lmsys.org (@lmsysorg) 的推文](https://x.com/lmsysorg/status/1792677208185794906)：令人兴奋的排行榜更新🔥 我们已将 @01AI_Yi Yi-Large 添加到 Arena 中，并在过去一周收集了 1.5 万多张选票。Yi-Large 的表现非常出色，位列第 7，几乎与...
*   [Alexandr Wang (@alexandr_wang) 的推文](https://x.com/alexandr_wang/status/1792905417065914858?s=46&t=90xQ8sGy63D2OtiaoGJuww)：1/ 今天，@Scale_AI 宣布以 13.8 亿美元的估值获得 10 亿美元融资。本轮融资由 @Accel 领投，现有投资者参投。@Scale_AI 在加速...
*   [Hayden Field (@haydenfield) 的推文](https://x.com/haydenfield/status/1792748249272795348?s=46&t=90xQ8sGy63D2OtiaoGJuww)：刚刚收到 OpenAI CEO Sam Altman 关于 Scarlett Johansson 声音争议的声明。
*   [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1792949163249791383?s=46&t=90xQ8sGy63D2OtiaoGJuww)：冲啊！Phi 3 - Small, Medium & Vision 发布了！🔥 > Medium 与 Mixtral 8x22B, Llama 3 70B 具有竞争力，并且击败了...

s Command R+ 104B & GPT 3.5 > Small 击败了 Mistral 7B & Llama 3 8B > 4K & 128K ...</li><li><a href="https://youtu.be/uHEPBzYick0?si=ajbDL9agnubNAECO&t=203">微软 CEO 谈新款 Windows AI Copilot+ PC 如何击败苹果 Mac | WSJ</a>：微软搭载 Qualcomm 芯片和 AI Windows 功能的新款 Copilot+ PC 旨在击败苹果的 MacBook。WSJ 的 Joanna Stern 试用了这些新款笔记本电脑并坐下来...</li><li><a href="https://buttondown.email/ainews/archive/ainews-to-be-named-3447/">[AINews] Skyfall</a>：不考虑 superalignment，Google Scarlett Johansson 就是你所需的一切。2024/5/17-2024/5/20 的 AI 新闻。我们检查了 7 个 subreddit，384 个 Twitter 和 29 个...
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1242071749131239505)** (76 messages🔥🔥): 

- **GPT-32k 面临速率限制问题**：用户报告在使用 Azure 的 GPT-32k 模型时遇到 Token 速率限制问题。一位用户表示：“Azure OpenAI API 2023-07-01-preview 版本下的 ChatCompletions_Create 操作请求已超过 Token 速率限制。”

- **讨论 Phi-3 模型的高性能表现**：成员们讨论了使用 [Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) 和 [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) 处理高质量、推理密集型数据。这两个模型都结合了 supervised fine-tuning 和 direct preference optimization 以增强性能。

- **与 LLM 交互的新方法**：一位用户分享了一个关于使用“Action Commands”[与 LLM 交互的新方法](https://x.com/leonjcoe/status/1792946945528320382)的帖子。他们寻求他人的反馈，看看是否有人有类似的经验。

- **处理模型中的冗长问题**：成员们讨论了如何处理像 Wizard8x22 这样模型中的冗长问题。有人建议降低 repetition penalty 以减少冗长，而另一位则指出不同的模型可能更适合特定的任务。

- **非营利组织的折扣请求和信用额度问题**：一位用户遇到了与 [账单地址相关的 Error 400](#) 问题，并为非营利组织申请折扣。管理员解释说，OpenRouter 将批量折扣转让给用户，并保留 20% 的利润空间。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/leonjcoe/status/1792946945528320382">来自 Leon Builds Agents (@leonjcoe) 的推文</a>：有一种没人讨论的与 LLM 交互的新方式。Action Commands。那么它们是什么，为什么如此有价值？让我展示给你看。</li><li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-medium-4k-instruct">microsoft/Phi-3-medium-4k-instruct · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1242146692376170617)** (43 messages🔥): 

- **M3 Max 在 LLM 方面表现出色**：一位用户赞扬了 **M3 Max** 的性能，称其“非常惊人”，并建议配置 96GB RAM 以更好地配合 LLM 使用。
- **Git Patch 合并困扰**：一位用户在自行合并 Git patch 时遇到问题，并讨论了更新特定文件进行测试的事宜。他们提到：“使用 git 有点棘手，因为我把它推到了自己的仓库而不是上游仓库”。
- **Unsloth 与 ROCm 的兼容性问题**：另一位用户报告了由于对 xformers 的依赖，导致最新的 unsloth 更新在 ROCm 上存在兼容性问题。尽管如此，“unsloth 的 `gradient_checkpointing` 仍然有效，并且带来了显著的内存优化”。
- **Transformers 库中的语法错误排查**：用户们协作解决了 transformers 库中与 `CohereTokenizer` 相关的 `ValidationError` 和 `AttributeError`。他们探索了使用 `CohereTokenizerFast` 和 `AutoTokenizer` 作为替代方案 [GitHub pull request 链接](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files)。
- **寻找更快的 STT -> LLM -> SST Python 库**：一位用户询问是否有人记得某个旨在加速语音转文本（STT）到 LLM 再到语音合成（SST）的 Python 库名称。日志中未提供具体答案。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547/files">Feat: Add cohere (commandr) by NanoCode012 · Pull Request #1547 · OpenAccess-AI-Collective/axolotl</a>：描述、动机和背景、测试方式（未测试！）、截图（如有）、变更类型、社交账号（可选）。</li><li><a href="https://github.com/huggingface/transformers/blob/d24097e0229485287ff4959258c55">GitHub - huggingface/transformers at d24097e0229485287ff4959258c552168bd898c6</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。</li><li><a href="https://github.com/huggingface/transformers/blob/d24097e0229485287ff4959258c552168bd898c6/src/transformers/models/cohere/tokenization_cohere_fast.py#L51C7-L51C26">transformers/src/transformers/models/cohere/tokenization_cohere_fast.py at d24097e0229485287ff4959258c552168bd898c6 · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1242126360860819567)** (8 messages🔥): 

- **使用 Grok-1 PyTorch 版本训练 Grok**：一位用户分享了他们使用 [Grok-1 PyTorch 版本](https://huggingface.co/hpcai-tech/grok-1) 训练 **Grok** 的计划，并征求意见。另一位用户表示赞同，并提到了 Axolotl 即将进行的 **torchtune 集成**。
- **Torchtune 集成备受关注**：有人猜测 **torchtune** 是否会取代 Hugging Face 后端，或者作为其并列的一个选项。一些用户持有强烈观点，其中一人建议“拆解 hf”。
- **算力情况确认**：当有人询问此次训练任务使用的算力时，引起了大家的兴趣。得到的回复是 **Mi300x**，这引发了关于用户满意度以及与 **H100s** 对比的好奇。

**提及的链接**：<a href="https://huggingface.co/hpcai-tech/grok-1">hpcai-tech/grok-1 · Hugging Face</a>：未找到描述。

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1242217905366634507)** (15 条消息🔥): 

- **Mistral 7B 微调困境**：一位成员在对其数据进行 **Mistral 7B** 微调时遇到问题，尽管 loss 在下降，但模型会出现信息混淆。他们分享了一个 [配置链接](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/lora.yml)，并对模型为何无法正常学习表示困惑。
- **全量微调 vs. LoRA**：另一位成员建议尝试全量微调或利用 **Retrieval-Augmented Generation (RAG)** 以获得更好的模型记忆保留效果，并指出 **LoRA** 可能对风格保留更有效，而非内容保留。
- **推理配置问题**：讨论了确保在推理期间手动添加 chat template 的必要性，因为目前的设置可能不会自动包含它。一位成员分享了关于潜在 tokenization 不匹配问题的链接 [点击此处](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#tokenization-mismatch-bw-inference--training)。
- **共享配置以进行故障排除**：一位参与者被要求分享其配置，以帮助他人了解设置并提供更好的指导。
- **下个稳定版本咨询**：一位用户询问了 **axolotl** 下一个稳定大版本的发布时间。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#tokenizati">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: 尽管提问 (axolotl questions)。通过在 GitHub 上创建一个账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#tokenization-mismatch-bw-inference--training">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: 尽管提问 (axolotl questions)。通过在 GitHub 上创建一个账号来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1242038505085997057)** (5 条消息): 

- **尽管有 24GB 显存仍遇到 OOM 问题**：一位用户在拥有 24GB VRAM 的情况下，在训练期间遇到了 **Out-of-Memory (OOM)** 错误。他们分享了自己的配置以及尝试过但未成功的各种设置。
- **Phorm 针对 OOM 问题建议的解决方案**：为了解决 OOM 问题，Phorm 建议增加 **gradient accumulation steps**、启用 **mixed precision training**、使用 **model parallelism**、减小 batch size 以及利用 **DeepSpeed ZeRO optimization** 等方法。提供的详细配置包括 *mixed_precision: 'fp16'* 和 *zero_optimization with stage: 3*。
- **DeepSpeed 和 ZeRO 优化策略**：通过利用 DeepSpeed 的 **ZeRO-2 和 ZeRO-3** 阶段，可以显著减少内存占用。分享了将 optimizer 和 parameter 状态 offload 到 CPU 的示例配置。
- **管理内存的混合策略**：其他方法包括 **CPU and Disk Offloading**、利用高效的模型和算子、使用内存分析工具（如 *torch.cuda.memory_summary()*）以及针对变长序列的 dynamic padding。这些技术可以通过优化内存管理来帮助训练更大的模型。
- **更多详情请访问 Phorm.ai**：建议用户查看 [Phorm.ai](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=79ea3546-1ab6-4fe1-8984-1d8eb8183eda) 以获取有关防止 OOM 错误解决方案的更多信息和更新。

**提到的链接**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=79ea3546-1ab6-4fe1-8984-1d8eb8183eda)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。

  

---

### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1242138352224964679)** (1 条消息): 

- **关于开源长期记忆的精彩网络研讨会 (Webinar)**：一个新的网络研讨会定于太平洋时间周四上午 9 点举行，届时将邀请 **memary** 的作者——这是一个用于自主 Agent (autonomous agents) 长期记忆的全开源参考实现。参与者可以通过[点击此处](https://lu.ma/nzh3o83f)报名参加。

- **深入探讨 memary**：网络研讨会将包括关于 **memary** 的深入讨论和问答环节，涵盖其功能，如使用 LLM 和 neo4j 将 Agent 的输入/响应提取到知识图谱中，利用记忆流 (memory stream) 记录交互时间线，以及对热门实体进行排名。

**提到的链接**：<a href="https://lu.ma/nzh3o83f">LlamaIndex Webinar: Open-Source Longterm Memory for Autonomous Agents · Zoom · Luma</a>：在本次网络研讨会中，我们很高兴能邀请到 memary 的作者——这是一个用于自主 Agent (autonomous agents) 长期记忆的全开源参考实现 🧠🕸️ 在……

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1242138908398194751)** (6 条消息): 

- **关于自主 Agent 的 Memary 新网络研讨会**：本周四太平洋时间上午 9 点，将举行一场深入探讨会议，邀请 **memary** 的作者，这是一个用于自主 Agent (autonomous agents) 长期记忆的全开源参考实现。[参加网络研讨会](https://t.co/XycydBSfTp)以了解更多信息。
- **关于高级 RAG 技术的 PizzerIA 演讲**：本周四在巴黎的 PizzerIA 关注 @hexapode，他将讨论高级检索增强生成 (RAG) 技术。[活动详情](https://t.co/dytY4VKdj3)已发布，供感兴趣的人查看。
- **旧金山首次线下见面会 (Meetup)**：下周二，在 LlamaIndex 的旧金山总部与团队见面，并听取 @jerryjliu0、Tryolabs 和 ActiveLoop 的分享。[在此预约 (RSVP)](https://t.co/qIGOmCW62G) 参加，了解如何将 RAG 系统推进到基础配置之外。
- **LlamaIndex 的 TypeScript 文档升级**：**LlamaIndex.TS** 文档已升级，包括新的入门教程和构建 Agent 的分步指南。查看[更新后的文档](https://t.co/UKycgYpq1F)。
- **使用 GPT-4o 进行复杂文档 RAG**：**GPT-4o** 现已原生集成到 LlamaParse 中，利用多模态 (multimodal) 能力处理复杂的 PDF 和幻灯片。更多详情请见[公告](https://t.co/g5TG7brSwt)。
- **在 Azure 沙箱中安全运行 LLM 生成的代码**：今天发布，使用 Azure Container Apps 动态会话在沙箱中安全运行 LLM 生成的代码。更多信息请见[发布详情](https://t.co/2cnsBH411k)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://t.co/qIGOmCW62G">RSVP to GenAI Summit Pre-Game: Why RAG Is Not Enough? | Partiful</a>：注意：这是在旧金山 LlamaIndex 总部举行的线下见面会！顺便来参加我们的见面会，了解为您的公司构建生产级检索增强生成 (RAG) 引擎的最新创新……</li><li><a href="https://t.co/koCp84KfYb">What is LlamaIndex? | LlamaIndex.TS</a>：LlamaIndex 是一个用于构建 LLM 驱动的应用程序的框架。LlamaIndex 帮助您摄取、结构化和访问私有或特定领域的数据。它提供 Python 包和 Type...</li><li><a href="https://t.co/UKycgYpq1F">Getting started | LlamaIndex.TS</a>：在本指南中，我们将引导您使用 LlamaIndex.TS 库在 JavaScript 中构建 Agent 的过程，从零开始并分阶段增加复杂性。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1242097125215436842)** (45 messages🔥): 

- **解决 Pinecone 的文档哈希问题**：一位成员寻求关于如何计算文档或节点哈希以防止在 Pinecone 中出现重复条目的建议。他们解释了涉及网页抓取内容和 PDF 文档重叠的使用场景。

- **修改 OpenAI Agent 的系统提示词**：一位成员询问如何在不创建新对象的情况下修改 OpenAI Agent 的系统提示词（System Prompt）。另一位成员建议使用 `chat_agent.agent_worker.prefix_messages` 属性。

- **在 LlamaIndex 中运行 gguf 格式模型**：有人提问如何在不使用 OpenAI 的情况下，将 LlamaIndex 与来自 Hugging Face 的 gguf 格式模型配合使用。讨论明确了 LlamaIndex 可以通过使用 HuggingFaceLLM 加载模型和分词器（tokenizer）来工作。

- **使用 Airtable 对比 Excel/Sqlite**：讨论了 Airtable 相比 Excel 和 Sqlite 的优势，强调了 Airtable 与 Langchain 的集成，可直接使用相关功能。此外还分享了 Langchain Airtable 集成文档的链接。

- **处理 VectorStoreIndex 中的空节点问题**：讨论集中在解决使用 `VectorStoreIndex.from_vector_store` 加载索引时的空节点问题。建议确保从数据库正确地将文档加载到 docstore 中。

**提到的链接**：<a href="https://python.langchain.com/v0.1/docs/integrations/document_loaders/airtable/">Airtable | 🦜️🔗 LangChain</a>：在此获取你的 API key。

  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1242531297750941899)** (7 messages): 

- **AI Waifus 引发简短趣谈**：一位用户断言“AI waifus 拯救生命！”，引发了与另一位回复“Just monika”的用户的俏皮互动。
- **3D 角色聊天机器人项目预告**：一位用户提到了他们在 4Wall AI 开发 3D 角色聊天机器人的工作，并引导其他人去另一个频道查看预告。
- **Inflection AI 计划嵌入情感 AI**：一位用户分享了来自 [VentureBeat 的链接](https://venturebeat.com/ai/exclusive-inflection-ai-reveals-new-team-and-plan-to-embed-emotional-ai-in-business-bots)，内容关于 Inflection AI 计划在商业机器人中集成情感 AI，暗示了 AI waifus 理解和处理情感的可能性。
- **对角色引用的困惑**：针对“Just monika”，另一位用户询问“那是谁？”，并收到了来自 Tenor.com 的 [GIF 链接](https://tenor.com/view/ddlc-doki-doki-literature-club-just-monika-monika-gif-20717242) 以澄清该引用。

**提到的链接**：<a href="https://tenor.com/view/ddlc-doki-doki-literature-club-just-monika-monika-gif-20717242">Ddlc Doki Doki Literature Club GIF - Ddlc Doki Doki Literature Club Just Monika - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1242058484627931136)** (18 messages🔥): 

- **AI Town 对话缺乏上下文**：一位用户报告说，角色“对另一个角色说的话完全没有反应”，导致出现重复的问候语，如 *“嗨！终于能和你聊天真是太棒了！”* 另一位用户建议，虽然有一个 *vector memory system*（向量记忆系统）用于检索过去的对话，但可能受到了设置或配置的影响。

- **调整 Convex 设置以减少记忆获取**：为了解决 AI Town 对话中出现 *empty bubbles*（空白气泡）的问题，建议用户调整 `convex/constants.ts` 中的数值，特别是将 `NUM_MEMORIES_TO_SEARCH` 从默认值 3 更改为 1。

- **从 SQLite 导出 AI Town 对话**：一位用户因对 Schema 的误解而在导出对话数据时遇到困难。另一位用户提供了一个有用的 SQL 查询，并推荐使用 DB Browser for SQLite。同时，还分享了一个 GitHub 仓库 ([townplayer](https://github.com/cocktailpeanut/townplayer/blob/main/index.html)) 和相关的 [Twitter 线程](https://x.com/cocktailpeanut/status/1786421948638965870)，用于更高级的 AI Town 数据提取查询和工具。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/cocktailpeanut/townplayer/blob/main/index.html">townplayer/index.html at main · cocktailpeanut/townplayer</a>：重放 AI Town。通过在 GitHub 上创建账号为 cocktailpeanut/townplayer 的开发做出贡献。</li><li><a href="https://github.com/cocktailpeanut/">cocktailpeanut - 概览</a>：cocktailpeanut 拥有 142 个代码仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://x.com/cocktailpeanut/status/1786421948638965870">来自 cocktail peanut (@cocktailpeanut) 的推文</a>：介绍 AI Town Player。你知道整个 AI Town 都通过 @convex_dev 存储在单个 sqlite 文件中吗？我逆向工程了它的 Schema 并构建了一个 Web 应用，让任何人都可以重放任何 A...
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1242052916525793341)** (18 messages🔥): 

- **澄清 LLM 中的结构化数据**：一位成员询问 LLM 处理结构化数据的方式是否与非结构化文本不同。另一位成员解释说，LLM 处理结构化数据（如 JSON）和非结构化文本的方式类似，但可以针对特定结构进行 Fine-tuning（微调），并提到了 [Hermes 2 Pro - Mistral 7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) 和 [OpenAI 的 chatML](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) 格式等示例。

- **解释 LangChain 包的区别**：一位成员询问 `langchain` 和 `langchain_community` 之间的区别。回复指出，`langchain-core` 包含具有轻量级依赖的基础抽象，而流行的集成位于独立的包中（如 `langchain-openai`），较不常见的集成则位于 `langchain-community` [架构](https://python.langchain.com/v0.2/docs/concepts/#architecture)中。

- **LangChain 中的序列链 (Sequential Chains)**：一位成员分享了代码，展示了如何设置序列链，其中一个链的输出作为另一个链的输入。这得到了一个演示该概念的 [YouTube 教程](https://youtu.be/2xxziIWmaSA?si=3wkNt_huJKu3xK3t&t=1694) 的支持。

- **处理 LangServe 中的并发请求**：另一位成员报告了在 LangServe 中处理多个并发请求时遇到困难。目前该问题尚无回复。

- **保护 LLM 响应中的敏感数据**：一位新用户询问是否可以通过隐藏客户姓名或卡号等敏感数据来保护 RAG 应用中的 LLM 响应。讨论中尚未提供解决方案。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B#prompt-format">NousResearch/Hermes-2-Pro-Mistral-7B · Hugging Face</a>：暂无描述</li><li><a href="https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md">openai-python/chatml.md at release-v0.28.0 · openai/openai-python</a>：OpenAI API 的官方 Python 库。通过在 GitHub 上创建账号为 openai/openai-python 的开发做出贡献。</li><li><a href="https://youtu.be/2xxziIWmaSA?si=3wkNt_huJKu3xK3t&t=1694">The LangChain Cookbook - 7 个核心概念初学者指南</a>：Twitter: https://twitter.com/GregKamradt Newsletter: https://mail.gregkamradt.com/signup Cookbook Part 2: https://youtu.be/vGP4pQdCocw Wild Belle - Keep You: ht...</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#architecture">概念指南 | 🦜️🔗 LangChain</a>：本节包含对 LangChain 关键部分的介绍。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1242111918571913408)** (3 messages): 

- **Easy Folders 联盟计划启动**：针对 **ChatGPT Chrome Extension - Easy Folders** 的新联盟计划已启动。推广者可获得 25% 的佣金，客户可享受 10% 的折扣。更多详情请点击[此处](https://easyfolders.promotekit.com/)，扩展程序可从 [Chrome Web Store](https://chromewebstore.google.com/detail/easy-folders-chatgpt-clau/gdocioajfidpnaejbgmbnkflgmppibfe) 下载。
- **Easy Folders 扩展受到批评与赞扬**：用户对 Easy Folders 扩展的评价褒贬不一。一位用户批评其增加了界面杂乱感且性能缓慢，而另一位用户在丢失保存的文件夹和聊天记录之前表示非常满意。
- **从 LangChain 升级到 LangGraph**：一位用户分享了一篇 Medium 博客文章，介绍如何将旧版 LangChain Agent 迁移到新的 **LangGraph** 平台。感兴趣的用户可以在[此处](https://medium.com/ai-advances/upgrading-your-agents-a-smooth-transition-from-legacy-langchain-to-langgraph-c552cb60fcb3)阅读更多内容。
- **使用 Upstage AI 和 LangChain 查询 PDF 文件**：分享了一篇博客文章，详细介绍了如何使用集成在 LangChain 中的 **Upstage AI solar models** 创建 PDF 查询助手。点击[此处](https://medium.com/@sonam.gupta1105/creating-a-pdf-query-assistant-with-upstage-ai-solar-and-langchain-integration-6631280093b5)查看博客文章。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://easyfolders.promotekit.com/">注册</a>：适用于 Stripe 的联盟营销软件</li><li><a href="https://chromewebstore.google.com/detail/easy-folders-chatgpt-clau/gdocioajfidpnaejbgmbnkflgmppibfe?hl=en-GB&authuser=0">Easy Folders: ChatGPT &amp; Claude 聊天整理器</a>：适用于 ChatGPT &amp; Claude 的拖放式文件夹。彩色文件夹。嵌套文件夹。历史搜索。书签。批量删除聊天。</li><li><a href="https://chatgpt-easy-folders.vercel.app/">ChatGPT Easy Folders</a>：一个通过文件夹、书签和搜索来整理 ChatGPT 历史记录的浏览器扩展。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages): 

bayraktar47: <@1043024658812895333>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1242222144281120768)** (13 messages🔥): 

- **电影《她》(Her) 中的 OS1 引用引发感悟**：一位用户分享了一个有趣的观察，即 Open Interpreter **O1** 是对电影《她》中 **OS1** 的致敬。这一发现激发了成员们的好奇心。

- **寻求 DevOps AI 模块的帮助**：一位初级全栈 DevOps 工程师正寻求构建一个 **lite O1** AI，以协助处理 DevOps 工具、配置终端和云计算。目标是通过**隐形耳机**提供这些资源，以便在各种工作环境中实现不显眼的 AI 辅助。

- **安装与开发环境设置咨询**：成员们正在讨论 **Open Interpreter** 如何访问文件系统并审查项目结构。针对更高效的开发设置，提出了具体的问题。

- **Open Interpreter 的日常用途与问题解决**：针对关于日常用途和复杂问题解决的开放性提问，多位用户对已记录的成功案例表示感兴趣，并分享了他们的具体使用场景。示例包括设备间的无缝引用、编码时查询特定上下文的数据以及总结研究论文。

- **将 Text-to-Speech 与 Open Interpreter 集成**：一位成员就如何将 **Text-to-Speech** 引擎和语音识别与 Open Interpreter 结合寻求建议。他们被引导至相关的 [GitHub](https://github.com/OpenInterpreter/01) 仓库，并被鼓励探索其他的支持渠道。

**提到的链接**：<a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01: 开源语言模型计算机</a>：开源语言模型计算机。通过在 GitHub 上创建账号为 OpenInterpreter/01 的开发做出贡献。

  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1242101799167725579)** (3 messages): 

- **寻求将笔记本电脑连接到 Light App 的步骤**：一位成员请求关于将笔记本电脑连接到 Light App 的指导，并指出相关步骤未在指南中列出。消息中未提供该 App 的详细信息或具体的连接方式。
- **初级 DevOps 工程师需要 Lite 01 项目的帮助**：一位初级全栈 DevOps 工程师表示需要协助构建 Lite 01，旨在简化日常任务并使担任类似角色的其他人受益。他们正在开发一个用于提供资源和谨慎协助的 AI 模块，寻求帮助以创建一个 Open Interpreter Lite 01，因为预订要到明年秋天才能发货。
- **请求组装 3D 打印零件的指导**：同一位初级 DevOps 工程师表现出学习如何组装 Open Interpreter Lite 01 的零件和 3D 打印外壳的兴趣。他们向已经完成组装的人征求组装过程中的技巧或指导。
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

ashthescholar.: 错失了把它变成 moo 的机会
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1242062452250771456)** (15 messages🔥): 

- **Transformers.js 中的 Codegen-350M-mono**：成员们讨论了在 Huggingface 的 Transformers.js 中使用 Codegen-350M-mono 模型。分享了一个带有 ONNX 权重的 [Xenova's codegen-350M-mono](https://huggingface.co/Xenova/codegen-350M-mono) 链接，作为解决兼容性问题的方案。
- **用于翻译的 CommandR+**：有人询问使用 CommandR+ 进行翻译的情况，提到它在韩语翻译成英语方面表现良好。他们被引导至 [Chat API 文档](https://docs.cohere.com/docs/chat-api) 以获取示例代码和更多详细信息。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Xenova/codegen-350M-mono">Xenova/codegen-350M-mono · Hugging Face</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/chat-api">使用 Chat API</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1242045816584147024)** (10 messages🔥): 

- **Sky 语音模式因争议暂停**：OpenAI 已暂停在 GPT-4o 演示中使用 Sky 语音，原因是涉嫌模仿 Scarlett Johansson。一位用户注意到 Sky 已被另一种女性声音 Juniper 取代，而 [Scarlett Johansson 发表了一份声明](https://x.com/BobbyAllyn/status/1792679435701014908) 针对此事进行了说明。

- **GPT-4o 集成多模态模型**：据一位用户称，之前版本的 GPT 使用不同的模型来处理音频和文本，导致无法识别语调或背景噪音等限制。GPT-4o 现在对文本、视觉和音频使用单一模型，这可能会增加情感深度，但也带来了复杂性和潜在的缺点。

- **韧性优于完美**：一位用户引用了 Stainslaw Lem 的短篇小说，认为复杂系统中的完美可靠性是无法实现的。相反，重点应该放在构建能够应对不可避免故障的韧性系统上。

- **语音克隆中的法律复杂性**：用户讨论了语音克隆的法律和伦理影响，特别是考虑到 Scarlett Johansson 的担忧。一位用户批评了依赖立法来保护肖像权的局限性，强调了执法方面的限制以及现有的开源语音克隆技术。

- **高通 Snapdragon 开发套件发布**：一位成员分享了对高通新款售价 899.99 美元的 [Snapdragon Dev Kit for Windows](https://www.theverge.com/2024/5/21/24158603/qualcomm-windows-snapdragon-dev-kit-x-elite) 的热情。该开发套件凭借其 4.6 TFLOP GPU、32GB RAM 和 512GB 存储空间提供了强大的性能，包装类似于 Apple 的迷你台式机。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.theverge.com/2024/5/21/24158603/qualcomm-windows-snapdragon-dev-kit-x-elite">这是用于 Windows on Arm 实验的 8 英寸 Snapdragon PC</a>：高通正在销售黑色版本。</li><li><a href="https://x.com/BobbyAllyn/status/1792679435701014908">Bobby Allyn (@BobbyAllyn) 的推文</a>：Scarlett Johansson 就 OpenAI 事件发表的声明。哇：
</li>
</ul>

</div>
  

---

### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1242451432444006400)** (6 messages): 

- **Supervised Fine-Tuning (SFT) vs Preference Optimization**：一位成员询问了 **Supervised Fine-Tuning (SFT)** 与 **Preference Optimization** 之间的区别。他们提出，虽然 SFT 提高了 SFT 数据集的概率分布，但 Preference Optimization 会同时调整非期望和期望的概率，因此质疑为什么 SFT 是必要的。
  
- **Phi3 Vision 的效率令人印象深刻**：一位成员表达了对 **Phi3 Vision**（一个 42 亿参数的模型）的赞赏，称赞其在图像流的低延迟/实时推理中的表现。他们分享了一篇 [X 上的帖子](https://x.com/jphme/status/1792950682695479734)，讨论了其在机器人领域的潜在应用。
  
- **比较 Phi3 Vision 和 Moondream2**：另一位成员鼓励在同一张图片上使用 [Moondream2](https://huggingface.co/spaces/vikhyatk/moondream2) 与 Phi3 Vision 进行结果对比。反馈显示 Moondream2 表现良好且减少了幻觉，尽管某些数据集仍存在问题。
  
- **Microsoft 发布新模型**：**Microsoft 发布了 70 亿和 140 亿参数的模型**。社区成员观察到，目前仅提供 Instruct 版本。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/vikhyatk/moondream2">moondream2 - a Hugging Face Space by vikhyatk</a>: 无描述</li><li><a href="https://x.com/jphme/status/1792950682695479734">Jan P. Harries (@jphme) 的推文</a>: Phi3 vision 刚刚发布 - 它只有 4.2b 参数，非常令人印象深刻。🤩 我觉得这是图像流低延迟/实时推理的一个突破 - 想象一下更小/更多的模型会怎样...
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1242187117501743406)** (2 messages): 

- **Alex 向社区介绍 `sqlite-vec`**：Alex 分享了他的新项目 [`sqlite-vec`](https://github.com/asg017/sqlite-vec)，这是一个用于向量搜索的 SQLite 扩展，并提到它可能会与 Llamafile 集成，用于 RAG、记忆、语义搜索等功能。*“它完全由 C 语言编写，应该可以与 cosmopolitan 配合使用，尽管我自己还没有测试过。”*
  
- **详细的项目描述**：Alex 提供了一篇[详细的博客文章](https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html)，解释了 `sqlite-vec` 的潜力和进展，该项目旨在取代 `sqlite-vss` 并提供更高性能和更易嵌入的解决方案。该扩展目前仍处于 Beta 阶段，但已提供早期试用，支持 C/C++ 项目以及 pip/npm/gem 平台上的包分发。

- **开放合作与支持**：Alex 表示愿意支持并帮助任何人开始使用 `sqlite-vec`，并解决用户在 Beta 阶段可能遇到的任何问题。“更多内容即将推出，但很高兴能帮助这里的任何人入门或解决任何问题！”

- **社区反响热烈**：一位成员对 Alex 表示欢迎，并对该项目与 Llamafile 集成的潜力表示热切期待。*“对你的项目感到非常兴奋，也对将其与 llamafile 集成所呈现的可能性感到兴奋。”*
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/asg017/sqlite-vec">GitHub - asg017/sqlite-vec: 正在开发中的、可随处运行的向量搜索 SQLite 扩展。</a>: 正在开发中的、可随处运行的向量搜索 SQLite 扩展。 - asg017/sqlite-vec</li><li><a href="https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html">我正在编写一个新的向量搜索 SQLite 扩展</a>: sqlite-vec 是一个新的向量搜索 SQLite 扩展，即将推出！</li><li><a href="https://github.com/asg017/sqlite-vec/releases">Releases · asg017/sqlite-vec</a>: 正在开发中的、可随处运行的向量搜索 SQLite 扩展。 - asg017/sqlite-vec
</li>
</ul>

</div>
  

---



### **LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1242201665835237448)** (1 messages): 

- **GPT-4o 在法律推理方面表现出色**：一位成员分享了他们在 **GPT-4o** 上进行复杂法律推理任务内部评估测试的经验。他们报告称，相比 GPT-4 和 GPT-4-Turbo 有“非同寻常的提升”，并链接了一篇关于 GPT-4o 发布的 [LinkedIn 帖子](https://www.linkedin.com/posts/evan-harris-387375b2_the-release-of-gpt-4o-from-openai-has-been-activity-7196856963454959617-w1i1)。

### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1242233137828728903)** (1 条消息): 

- **Manifold Research Group 寻求合作者**：来自 Manifold Research Group 的代表介绍了他们专注于 *generalist models* 和 AI Agents 的 OS R&D Lab。他们邀请感兴趣的人士[了解更多](https://www.manifoldrg.com/research-log-038/)，或通过他们的 [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) 和 [GitHub](https://github.com/ManifoldRG?ref=manifoldrg.com) 加入团队。
- **NEKO Project 致力于开源 generalist models**：NEKO Project 正在构建首个大规模开源 generalist model，该模型在包括控制和机器人任务在内的多种模态上进行训练。更多信息可以在其详细的 [项目文档](https://docs.google.com/document/d/e/2PACX-1vQELDXCIT9tn7Uq5vxQG4_3HsrkQcuBRqvXm-MkxW06Zkh-LP3G9z7TP7a-2MNWyA/pub?ref=manifoldrg.com) 中找到。

**提到的链接**：<a href="https://www.manifoldrg.com/research-log-038/">Research log #038</a>：欢迎来到 Research Log #038！我们记录了 Manifold Research Group 各项计划的每周研究进展，并重点介绍了我们认为来自更广泛研究社区的突破性成果...

  
---




{% else %}

> 邮件中的各频道详细解析现已截断。
> 
> 如果您想查看完整的解析，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！

如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}