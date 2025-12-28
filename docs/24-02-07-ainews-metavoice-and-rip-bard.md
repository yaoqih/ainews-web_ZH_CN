---
companies:
- coqui
- metavoice
- google
- openai
- thebloke
date: '2024-02-07T22:41:50.157897Z'
description: '最近倒闭的 TTS（文本转语音）初创公司 **Coqui** 启发了一家名为 **MetaVoice** 的小型初创公司，后者推出了一款支持语音克隆和长文本合成的新
  **TTS 模型**。**谷歌**停用了 **Bard** 品牌，转而采用 **Gemini**。


  在 **TheBloke Discord** 社区，讨论集中在 **Mixtral**、**Nous Mixtral DPO** 和 **Miqu 70B**
  等模型的 AI 训练上，并将其与 **OpenAI 的 GPT** 模型进行了对比；此外，还辩论了提示词工程（prompt engineering）、设定集（lorebooks）以及通过对
  **Llama2 70B instruct** 等模型进行 **LoRA 微调**来移除安全限制等话题。技术讨论还涉及 Transformer 层卸载（layer
  offloading）的限制以及针对 Apple Silicon 适配 **LLaMa 2**。


  在 **OpenAI Discord** 社区，**DALL-E** 生成的图像现在包含用于验证内容真实性的 **C2PA 元数据**，这引发了关于 AI 审查、元数据篡改以及开源
  AI 模型与 **GPT-4** 等商业巨头之间博弈的辩论。用户还讨论了 GPT-4 的易用性、局限性和实际应用。'
id: 914ee360-1e87-4dc6-90f8-6de50ace445b
models:
- mixtral
- nous-mixtral-dpo
- miqu-70b
- gpt-4
- llama-2-70b-instruct
- llama-2
- llama-2-70b
- llama-2-70b-instruct
original_slug: ainews-metavoice-rip-bard
people: []
title: MetaVoice 与 告别 Bard (或：别了，Bard)
topics:
- text-to-speech
- voice-cloning
- longform-synthesis
- prompt-engineering
- direct-preference-optimization
- lora-fine-tuning
- transformers
- gpu-acceleration
- apple-silicon
- content-authenticity
- metadata
- ai-censorship
- open-source-ai
- model-comparison
- usability
- model-limitations
---

<!-- buttondown-editor-mode: plaintext -->> 2024年2月6日的 AI Discord 动态。我们为您检查了 **20** 个公会、**308** 个频道和 **5284** 条消息。预计节省阅读时间（以 200wpm 计算）：**437 分钟**。

还记得上个月[倒闭](https://buttondown.email/ainews/archive/ainews-132024-rip-coqui/)的 TTS 初创公司 Coqui 吗？现在，一个新的支持语音克隆和长文本合成的 TTS 模型已经发布（[点击尝试](https://ttsdemo.themetavoice.xyz/)）。

 
![image.png](https://assets.buttondown.email/images/fa979c16-2f6e-4893-b68b-3c295d3d3ef2.png?w=960&fit=max)
 

这是一家[小型](https://metavoice.notion.site/Join-MetaVoice-e4c907cb6a2f4c33af2b148f635adda4)初创公司，但首个发布的产品非常有前景。

另外，[Google 为了 Gemini 弃用了 Bard 品牌](https://twitter.com/AndrewCurran_/status/1754546359460590002)。

---

**目录**

[TOC] 


# 第一部分：Discord 高层级摘要




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

- **AI 训练讨论升温**：讨论涉及 **Mixtral**、**Nous Mixtral DPO** 和 **Miqu 70B** 等 AI 模型，并将其在效率和能力方面与 OpenAI 的 GPT 模型进行了对比。关于 **Reddit /r/LocalLLaMA 子版块**审核制度的辩论十分激烈，并分享了讨论社区内 AI 进展和问题的 GitHub、Hugging Face 链接及 YouTube 视频。

- **AI 命名法中的古典与现代交汇**：在 #characters-roleplay-stories 频道中，`Thespis 0.8` 引发了关于其希腊悲剧起源的辩论，使对话转向了将神话用于 AI 上下文。角色扮演中的 **lorebooks** 被讨论作为 **Prompt Engineering** 的工具，并提到了 **DPO (Direct Preference Optimization)**，同时提供了 wandb 示例链接。

- **移除安全特性以释放 AI 的全部潜力**：用户分享了关于通过 **LoRA fine-tuning** 移除 **Llama2 70B instruct** 等模型安全护栏的见解，并引用了一篇 [LessWrong 文章](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from)。讨论还建议**合并数据集**可以增强 fine-tuning 期间的模型控制。

- **探索 AI 开发中的技术极限**：关于是否可以像 `llama.cpp` 那样将 Transformer 层卸载（offloading）到 GPU 的咨询，得出的结论是 **Transformers 库不支持在 CPU 和 GPU 之间进行层拆分**。讨论对 **Meta 的 Sphere 项目**表现出兴趣，认为其具有利用大数据工具整合**频繁更新**的潜力。

- **在替代平台上探索 AI 实现**：出现了关于在 **Apple Silicon 的 MLX 上实现 LLaMa 2** 的问题，特别是如何使模型的查询参数适应此平台。正在考虑 Multi-head 或 Grouped-query 格式的技术复杂性。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **DALL-E 采用内容真实性倡议（Content Authenticity Initiative）标准**：据 `@abdubs` 宣布，DALL-E 生成的图像现在包含符合 **C2PA 规范** 的元数据，有助于验证 OpenAI 生成的内容。此举旨在协助社交平台和内容分发者进行内容验证。详情请参阅 [官方帮助文章](https://help.openai.com/en/articles/8912793-c2pa-in-dall-e-3)。

- **AI 审查与真实性引发辩论**：AI 审查对用户体验的影响引发了激烈讨论；用户 `@kotykd` 和 `@arturmentado` 分别对 AI 审查提出了批评和辩护，重点讨论了用户自由与防止滥用的平衡。此外，社区还对将 AI 生成的艺术作品冒充人类创作的伦理问题表达了担忧，强调了根据 OpenAI 的 TOS 进行诚实披露的必要性。

- **社区认为元数据篡改非常简单**：关于元数据在图像溯源中的重要性引发了广泛讨论，用户一致认为删除此类数据非常容易，这使得它在图像来源验证方面成为一种不可靠的手段。这反映了保障数字图像真实性所面临的技术挑战。

- **开源 AI 模型与商业巨头的博弈**：一场关于开源 AI 模型与 GPT-4 等商业方案对比的辩论展开了，涉及对创新的影响以及具有竞争力的开源替代方案的潜在增长。该讨论反映了工程社区对 AI 技术发展格局的关注。

- **围绕 GPT-4 可用性与开发的讨论**：包括 `@mikreyyy`、`@glory_72072` 在内的多位用户就 GPT-4 的使用问题寻求帮助，如登出问题、寻找 Demo 以及故事创作能力。对话还涉及回答限制以及使用自定义 GPT 的复杂性，表明用户关注点集中在 GPT-4 在现实场景中的实际应用和局限性。

- **社区寻求 AI 驱动项目的改进与互动**：在 **Prompt Engineering 和 API 讨论** 领域，社区成员 `@loier` 和 `@_fresnic` 寻求关于改进 GPT 模块和优化角色交互 Prompt 的协作建议。同时，`@novumclassicum` 试图直接向 OpenAI 提供反馈，表达了开发者与 OpenAI 团队之间建立更高效沟通渠道的愿望。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 总结

- **Hugging Chat Assistant 个性化**：Hugging Face 推出了 **Hugging Chat Assistant**，允许用户构建具有可自定义名称、头像和行为的个性化助手。它支持 **Llama2** 和 **Mixtral** 等 LLM，通过消除对单独自定义 Prompt 存储的需求来简化用户体验。点击 [Hugging Chat](https://huggingface.co/chat/assistants) 查看新功能。

- **面向 PRO 和 Enterprise 的 Dataset Viewer**：HuggingFace 上的 Dataset Viewer 现在支持私有数据集，但该功能仅限 PRO 和 Enterprise Hub 用户使用。此更新旨在增强数据分析和探索工具（[来源](https://x.com/julien_c/status/1752716726012129577)）。

- **合成数据趋势**：HuggingFace Hub 添加了 `synthetic` 标签，以促进合成数据集的共享和发现，标志着合成数据在 AI 中日益增长的重要性。

- **AI 驱动的四足动物与 AI 时尚**：Cat Game Research 正在开发一款视频游戏，其特色是**首个利用 ML 和 AI 的四足角色控制器**，而 Sketch to Fashion Collection 则将草图转化为时尚设计。访问 [badcatgame.com](https://badcatgame.com) 和 [Hugging Face Spaces](https://huggingface.co/spaces/tonyassi/sketch-to-fashion-collection) 探索 AI 驱动的游戏和时尚创新。

- **BLOOMChat-v2 提升多语言对话能力**：**BLOOMChat-v2** 的 176B 参数多语言语言模型具有 32K 序列长度，正在其前代产品的基础上进行改进。API 和进一步的进展值得期待；详情见 [Twitter 总结](https://twitter.com/SambaNovaAI/status/1754928815590277146)和[详细博客文章](https://sambanova.ai/blog/bloomchat-v2)。

- **阅读小组动态与资源**：HuggingFace Reading Group 安排了一场关于时间序列预测的 decoder-only 基础模型的演示，并建立了一个 GitHub 仓库（[链接](https://github.com/isamu-isozaki/huggingface-reading-group)）来汇集往期会议资源，以加强知识共享。

- **Diffusers 与 Transformers 学习**：对于扩散模型的新手，社区建议通过 HuggingFace 和 FastAI 的课程进行深入学习，而关于 Timestep 设置和 Conditioning 模型验证等细节的咨询表明该领域内存在活跃的实验和学习。

- **NLP 频道引入教科书内容**：关于使用教科书内容微调 **LLama chat** 或 **Mistral** 等 LLM 的讨论，旨在增强特定领域的理解，从而提高模型的教育聊天能力。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **LM Studio 发布 v0.2.14**：新版本 **LM Studio v0.2.14** 发布，解决了 UI 冻结和输入挂起等关键 Bug，你可以通过 [LM Studio 官网](https://lmstudio.ai/)获取。请记得更新以获得更流畅的体验。

- **LM Studio 的易用性备受青睐**：用户被 [LM Studio 简单的用户界面](https://blog.stackademic.com/lm-studio-experience-the-magic-of-llms-with-zero-technical-expertise-4561039a01ed)所吸引，使任何人无需编程技能即可使用 LLM。但请注意，更新后 LLM 文件夹位置可能会重置为默认值。

- **本地模型运行挑战**：虽然用户遇到了模型生成冻结和 GPU 利用率低等问题，但 **LM Studio 最新更新**中的补丁旨在解决这些问题。此外，有关详细的模型微调说明，请查看 [YouTube 教程](https://www.youtube.com/watch?v=MDA3LUKNl1E)。

- **硬件中心**：关于 AI 任务最佳硬件配置的辩论仍在继续，一位用户正在准备 AMD 8700g 测试平台，并引发了对 **7950x3d** 可能升级的好奇。散热风扇配置涉及 **2x180mm 风扇对比 3x120mm Arctic P120 风扇**，但也有人提醒不要高估 APU 在 AI 相关计算中的性能。

- **LM Studio 的反馈循环**：用户关注 Beta 版本中的一些问题，如弹出模型时应用挂起以及过时的非 AVX2 Beta 版本；LM Studio 团队似乎对这些问题做出了回应。macOS 用户强调了一个持久的 Bug，即应用关闭后服务器仍保持活动状态。

- **专业 AI 应用咨询**：有关于用于 Python 脚本的 **chain-compatible seq2seq LLM 模型**的咨询，以及 Crew-AI 是否具有与 AutoGen Studio 等其他平台类似的 **UI 或 Web 界面**。

- **模型偏好与经验分享**：用户讨论了他们使用各种模型的经验，phoenix2574 在 open-interpreter 频道中随口提到 **Mixtral** 运行“还可以”。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

**Mistral 在编程领域表现卓越**：`@carsonpoole` 发现，在相同的采样场景下，**Mistral** 在 **OpenHermes 2.5** 的代码部分表现显著优于 **phi2**。讨论内容包括对 GPT-4 编程能力的启示，并引发了对 20 亿参数模型预期技能组合的好奇，其中引用了来自 Microsoft Research 的预期。

**Sparsetral 发布与数学基准测试热潮**：稀疏 MoE 模型 **Sparsetral** 正式推出，并提供了[原始论文](https://arxiv.org/abs/2401.02731)和 [GitHub 仓库](https://github.com/wuhy68/Parameter-Efficient-MoE)等资源。同时，`.benxh` 对 **Deepseek** 表示赞赏，该模型结合了名为 DPO 的技术，在以数学为重点的评估中达到了新的熟练水平。

**量化微调与 EQ-Bench**：`@tsunemoto` 对 Senku-70B（假设的 Mistral-70B 的微调版本）进行了量化，获得了 84.89 的 EQ-Bench 评分，并将其分享在 [HuggingFace](https://huggingface.co/ShinojiResearch/Senku-70B-Full) 上。这引发了关于数学在评估语言模型能力中的重要性以及举办由 LLM 驱动的机器人黑客松的广泛讨论。

**语言模型怪癖与 Mixtral 问题说明**：用户反映 Mixtral 在接收中文指令时会出现混合语言回答的情况，OpenHermes 也有类似问题。Cloudflare 的 AI 平台采用这些模型的消息通过 [推文](https://x.com/teknium1/status/1755020133398155269?s=46) 得到了关注。

**机器人控制框架支持**：`@babycommando` 寻求关于微调多模态模型的建议，并发布了 **MachinaScript for Robots** 及其 [GitHub 仓库](https://github.com/babycommando/machinascript-for-robots)。他们就微调 Obsidian 以及使用其 LLM 驱动框架进行机器人交互的技术规范寻求指导。

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **VRAM 狂热者与芯片对决**：讨论强调了运行全 fp16 AI 模型的硬件需求，建议至少需要 **100GB vRAM** 以获得最佳性能，并推测了 **Nvidia 4090** 和双 **4080 配置** 的充分性。辩论还转向了 **Intel Macs** 在 **Apple Silicon Macs** 面前的优劣与过时问题，反映了在升级哲学和实际寿命考量上的严重分歧。

- **低成本 AI 建模秘诀揭晓**：用户探讨了降低 **Mistral 模型** 计算成本的挑战，引用了 [DeepInfra 定价](https://deepinfra.com/pricing) 并建议使用 Serverless 平台、硬件加速器和 **LlamaCPP** 等解决方案。运营成本讨论还涉及数据敏感性、微调以及本地推理与专业托管服务之间的平衡。

- **从微调挫折到推理创新**：技术挫折主要集中在微调中的 Padding 不一致问题，尽管参考了 [QLoRa 教程](https://youtu.be/OQdp-OeG1as) 等资源，一名成员仍表示困惑。其他人分享了模型启动时间改进的经验，有人报告在 **2-10 秒** 内即可就绪，社区还集思广益，探讨了使用 **Llama 1.6** 等工具为 **Mistral-8x7B** 模型进行有效的提示词工程（Prompt Engineering）。

- **对开源发布的期待**：简短的交流显示社区对某个未具名工具表现出兴趣，并承诺在可行时将其**开源**；然而，目前尚未提供更多细节或时间表。

- **记下 Office Hours 的时间**：Mistral 社区的下一场 **Office Hour** 会议已正式列入日程，感兴趣的参与者可以通过 [Discord 活动链接](https://discord.gg/mistralai?event=1204405056825327677) 获取访问权限。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **被盗 Discord 账号已找回**：`@astropulse` 遭受了鱼叉式网络钓鱼攻击，导致其 Discord 账号被盗。用户们分享了网络安全建议，并强调了 [Have I Been Pwned](https://haveibeenpwned.com/) 在检查电子邮件地址是否受数据泄露影响方面的实用性。

- **创建稀疏神经网络**：`@mkaic` 正在研究创新的神经网络架构，该架构允许在训练期间动态重构连接，这可能会增强模型的稀疏性和性能。

- **需要关注的新颖 AI 项目**：`@SegmentationFault` 提到了 [GitHub 上的 PolyMind](https://github.com/itsme2417/PolyMind)，该项目旨在将多种 AI 能力整合到一个平台中，强调了该项目在实用价值上优于娱乐导向的应用。

- **无需训练的文本生成图像一致性**：`@thejonasbrothers` 讨论了 [ConsiStory](https://arxiv.org/abs/2402.03286)，这是一个无需训练的模型，旨在通过新颖的注意力机制和特征注入来提高文本生成图像的一致性。

- **Google Research 用于文本生成视频的 Lumiere**：`@spirit_from_germany` 分享了一个 [YouTube 视频](https://youtu.be/Pl8BET_K1mc)，展示了 Google Research 的模型 *Lumiere*，该模型专注于根据文本输入创建全局一致的视频内容。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 摘要

**GPT 竞争与机器人**：GPT-3.5 在为冷门语言生成代码方面表现出比 GPT-4 更惊人的实力，同时 Eleuther 服务器正在讨论开放性与垃圾邮件机器人干扰之间的权衡。

**MetaVoice TTS 模型发布**：MetaVoice 1B 是一款新的 TTS 模型，已采用开源许可发布，引发了关于其性能的讨论，包括零样本语音克隆和情感语音合成等功能，详见这篇 [推文](https://x.com/reach_vb/status/1754984949654904988?s=46)。

**评估模型外推与优化**：评述了多种理解和提升模型能力的方法，从分析损失与序列长度的关系，到 SELF-DISCOVER 框架在推理基准测试中超越传统方法，如[这篇论文](https://arxiv.org/abs/2402.03620)所述。

**无限极限与可解释性**：关于深度学习无限深度极限和损失景观的询问引发了对现有研究的兴趣，同时[一篇研究论文](https://arxiv.org/abs/2402.01702)提出了一种名为进化提示优化 (EPO) 的语言模型解释新方法。

**剖析 LLM 提示词影响力**：对 LLM 提示词中可靠输入显著性方法的探索仍在继续，目前对 Integrated Gradients 持怀疑态度，[一篇令人担忧的论文](https://arxiv.org/abs/2212.11870)进一步强调了这一点，该论文对归因方法推断模型行为的能力提出了质疑。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

- **GCP 正在应对 A100 可用性问题**：社区成员在 Google Cloud Platform 上获取 **A100 GPUs** 时遇到困难，引发了对潜在短缺的担忧。讨论还涉及了各种模型和工具的 benchmark 时间，例如 **lm-eval-harness**，其中 7b 模型的 MMLU 测试在 4090 GPU 上大约需要 12 小时。

- **寻求 Axolotl UI**：Hugging Face 为创建 Axolotl Spaces 训练 UI 提供了 5000 美元的悬赏，促使前端（偏向使用 **Tailwind CSS**）和后端开发者（理想情况下使用 Python）进行协作。关于 UI 使用 **Shiny** 还是 **Gradio** 展开了辩论，来自 Posit 的 Shiny 团队提供了原型和支持。

- **多机模型保存问题**：用户报告在尝试多 GPU、多节点（multi-node）配置下保存模型时存在持续性问题，怀疑 Axolotl 中可能未正确实现分布式保存。尽管使用了最新的 transformers 库版本（4.37.2），但针对多节点训练的修复 pull requests 仍在审查中，社区成员正积极寻求代码适配以解决 **mistral fft** 保存错误。

- **使用 Alpacas 和 ChatML 调优 DPO**：社区互动揭示了在获得可靠 **DPO** 结果方面的挑战，建议显著降低学习率。尽管早期在 Alpaca 上取得了成功，但目前正在探索从 **Alpaca format** 向 **ChatML** 的转变，并分享了涉及 Metharme 的个人变通方法。

- **Jupyter 的困惑与修正**：遇到 Jupyter notebooks 关键错误和警告的用户被引导至潜在的修复方案，包括一个针对挂载卷影响 workspace 目录问题的 Github pull request ([Cloud motd by winglian](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1235))。建议重新克隆仓库作为排查故障的一部分。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 摘要

- **寻求关于 Claude Pro 必要性的澄清**：在 **general** 频道中，`@brewingbrews_92072` 询问了在极简使用场景下是否需要订阅 **Claude Pro**，反映了在升级前的深思熟虑。
- **评估 AI 服务价格点**：**general** 频道中的另一场讨论深入探讨了 **Perplexity's API**（定价为 0.07/0.28）相对于其他 AI 服务的性价比，以及每月约 12 美元的 AI 扩展程序的普遍支出。
- **API Credit 经济**：`@general3d` 同样在 **general** 频道分享了他们使用每月 5 美元的 API credit 经济运行 Discord bot 的经验，强调了本地托管时的负担能力。
- **Gemini Pro 对标高级 AI 竞争对手**：`@luke_____________` 在 **general** 频道讨论了 **Gemini Pro** 与 **GPT-4** 等高级模型的性能对比，并展望了即将推出的 Gemini Ultra 所提供的潜力。
- **API 利用与摘要方面的挑战**：在 **pplx-api** 频道，用户在创建持续对话跟踪快捷方式、复制摘要功能以及将 Perplexity 的 API key 格式与 OpenAI 匹配以实现更广泛的工具兼容性等任务中遇到了困难。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 总结

- **LLM 处理表格数据的研讨会**：即将举行的研讨会将聚焦于 *使用 LLM 理解表格数据*，探讨 **Chain-of-Table** 方法以及如何利用多条推理路径增强 LLM 的性能。点击[此处](https://lu.ma/1cq5hvi4)注册参加太平洋时间周五上午 9 点的会议，并深入研究诸如 "[Chain-of-Table](https://arxiv.org/abs/2401.04398v1)" 和 "[Rethinking Tabular Data Understanding with Large Language Models](https://arxiv.org/abs/2312.16702)" 等论文。

- **为企业和研究利用 RAG**：`@seldo` 讨论了面向企业的 **语言模型** 和 **RAG**，提供的资源包括 [self-RAG 演进](https://t.co/na6n0kw2kX)、[Mistral 的 RAG 文档](https://t.co/LVMTG0YJ43) 以及 [研讨会信息](https://t.co/1yo21Z5QDN)。面向 RAG 初学者的 **简单 GitHub 仓库** 可在[此处](https://github.com/jotarretx/RAG_Tester)访问。

- **来自常规讨论的技术查询**：解决了使用 `ServiceContext.from_defaults` 进行 PDF 解析的问题，处理了 Neo4j 中的标签限制，提高了节点内容提取的效率，澄清了 `VectorStoreIndex` 中文档（documents）与节点（nodes）的区别，并排查了 LlamaIndex 的 SQL 查询合成故障。

- **Hacker News 和 Medium 揭示 SQL 与 RAG 见解**：讨论围绕寻找可靠的 NL to SQL 解决方案展开，重点参考了 [此处](https://news.ycombinator.com/item?id=39261486) 的 Hacker News 线程，以及一篇关于使用 RAG 和 LlamaIndex 进行自分块（Self-Chunking）的 Medium 文章（[链接在此](https://medium.com/ai-advances/self-chunking-brilliance-with-rag-analysis-and-llamaindex-revolution-dd590d734484)），涉及准确性挑战和文档分析的未来。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **ChromaDB 加速 RAG 系统**：`@bwo_28` 询问了如何优化 RAG 系统的性能，`@david1542` 建议使用 **ChromaDB 的持久化客户端 (persistent client)**，通过将 embeddings 保存到磁盘来加速相似度搜索，从而避免重新创建 embeddings，潜在地缩短加载时间（[ChromaDB 文档](https://docs.trychroma.com/usage-guide#initiating-a-persistent-chroma-client)）。

- **LangChain 取得进展**：关于 LangChain 的集成和功能引起了热烈讨论。讨论内容包括在 **AWS SageMaker** 上使用 LangChain 与 **Mistral** 的指南请求，关于使用 `OpenAIWhisperAudio` 将音频文件的 "response_format" 设置为 "vtt" 的查询，以及解决涉及从 `langchain` 导入 `SQLDatabase` 时出现的 `ModuleNotFoundError`。

- **Langserve 的稳健更新与修复**：`@veryboldbagel` 更新了新的事件流 API Agent 示例，并附带详细注释，可在 [GitHub](https://github.com/langchain-ai/langserve/tree/main/examples/agent) 获取；同时 `@albertperez.` 报告了一个已自行解决的 LangServe 部署循环问题。

- **对个人 AI 工作教练的需求**：`@bartst.` 寻求创建个人 AI 工作教练，引发了讨论，`@david1542` 表示有兴趣为此类倡议贡献想法。

- **MLBlocks 亮相**：`@neil6430` 分享了 [MLBlocks](https://mlblocks.com/) 的介绍，这是一个无代码平台，支持使用 AI 模型和传统方法构建图像处理工作流，并将流程简化为单个 REST API 端点。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **AI 助手成为日常英雄**：`@ashpreetbedi` 支持 AI 个人助手在 **总结每日站会** 等任务中的实用性，展示了它们在工作流程中日益增加的集成度。
- **通过 AI 实现代码自动化**：`@slono` 通过一个 [GitHub Gist](https://gist.github.com/wesen/a4ca759275f1a2bb2a9d4bf4b4b57769) 分享了一种自动化编程方法，揭示了像 'Aider' 这样的 AI 助手在简化开发流程方面的潜力。
- **RPA 的 AI 革命**：`@pennepitstop` 引发了关于 AI 在机器人流程自动化 (RPA) 中变革作用的讨论，引用 **Adept** 作为挑战 **UiPath** 等巨头的个人自动化技术领域的新秀。
- **利用向量数据库查询未来**：关于具有 API 端点的生产级 **向量数据库** 的对话促使 `@swyxio` 推荐了 **Supabase pgvector**，凸显了向更强大的数据查询工具发展的趋势。
- **关注 AI 模型扩展**：`@swyxio` 讨论了 Stella Biderman 关于 AI 规模的一条深刻 [推文](https://twitter.com/BlancheMinerva/status/1754960269250339286)，引起了社区共鸣，强调了 **RWKV** 和 **Mamba** 等模型的发展。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord 总结

- **GPU 愿景与实用性的结合**：`@timjones1` 异想天开地表示有兴趣搭建个人计算环境，而 `@joseph_en` 建议从单张 3090 GPU 或更具性价比的 12GB VRAM 3060 开始。同时，`@vim410` 讨论了揭示未被利用硬件特性的增量工作，这表明硬件性能仍有优化空间。

- **性能调优与监控**：`@cudawarped` 建议使用 Nvidia 的 `ncu` 工具进行更好的 Benchmarking，并分享了一个 [示例命令](https://github.com/cudawarped/cuda_mode_lectures/blob/rgb_to_grey/lecture3/rgb_to_grey.py)。`@iron_bound` 讨论了使用 [CLTune](https://github.com/CNugteren/CLTune) 进行 Kernel 调优，该工具对于现代 CUDA 可能已经过时。`@smexy3` 介绍了 `gmon`，这是一个简化 GPU 监控的工具，并提供了其 [GitHub 链接](https://github.com/AdamLouly/gmon)。

- **解决 PyTorch 中的量化难题**：`@hdcharles_74684` 在 torchao 中面临动态量化的挑战，并通过添加反量化（dequant）epilogue 提升了性能，详见此 [GitHub commit](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L295)。

- **旧款 MacBook 的 GPU 替代方案与云端选项**：`@boredmgr2005` 询问了使用 2015 款 MacBook Pro 进行 CUDA 编程的可行性，而 `@joseph_en` 建议在这种情况下使用 Google Colab 作为免费且功能强大的云端解决方案。

- **Jax 生态系统势头强劲**：`@joseph_en` 注意到 Jax 的受欢迎程度正在上升，并询问了 Google 对 Jax 的战略方向，暗示其在 AI 和 Machine Learning 社区中与 TensorFlow 存在竞争。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **寻找具有代表性的数据集**：`@bjoernp` 等成员建议使用**原始 SFT 数据集**进行模型训练，并对使用相同数据集的 Benchmark 以及多语言 c4 德语部分、德语维基百科和 *malteos wechsel_de* 等增强资源进行 Perplexity 测试表现出兴趣。

- **避免内存陷阱**：`@philipmay` 报告了 **Axolotl** 模型的 Out of Memory (OOM) 问题，暗示可能是 Deepspeed 的 `stage3_gather_16bit_weights_on_model_save` 设置配置不当，导致模型无法在单张 GPU 上容纳。

- **Jina Embeddings 表现不佳**：`@sebastian.bodza` 等用户批评 **Jina embeddings** 表现欠佳，尤其是在处理分布外（OOD）代码文档时；`@rasdani` 对此表示赞同，并显露出失望。

- **德语推理定价模型**：`@lightningralf` 介绍了一个拟议的德语推理服务两级价格模型，引发了关于企业赞助可能提供免费服务的讨论。

- **自主服务器管理以提升效率**：`@flozi00` 透露了一项内部建设数据中心的计划，反映了转向专有服务器解决方案和设立专门内部服务器管理部门的趋势。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **Lindy AI 通过初步测试**：`@thebaghdaddy` 发现 **Lindy AI** 能够执行数据检索和撰写报告等基础任务，但暗示可能需要专门的系统来提高特定任务的效率。

- **对 Azure AI 产品的疑问**：用户 `.psychickoala` 询问 Azure 是否有 **GPT-4 vision 模型**，但对话没有进一步得到解答。

- **Super JSON Mode 承诺提速**：`@res6969` 通过 `@varunshenoy_` 的一条 [推文](https://x.com/varunshenoy_/status/1754967233141633513?s=46) 介绍了 **Super JSON Mode**，声称在无需非常规方法的情况下，语言模型的结构化输出生成速度可提升 **20 倍**。

- **优化 MythoMax 的托管**：`@jmak` 正在寻找部署 **MythoMax** LLM 更具成本效益的托管方案，但目前缺乏社区的进一步建议。

- **苦于 PDF 处理？全部 OCR 化！**：`@pantsforbirds` 正在寻求增强 PDF 处理的方法，主要是针对那些文本编码较差的文件，而 `@res6969` 提倡普遍使用 OCR 来解决文本提取问题，尽管这会增加额外成本。

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

- **在硅谷寻求协同**：一位名为 `@blankcoo` 的成员联系并寻求与 Alignment Lab 项目的潜在**合作机会**。
- **热心新人提醒**：`@craigba` 分享了他加入 Alignment Lab AI Discord 的热情，并提供了他在**网络安全**方面的专业知识，参考了他在 [Threat Prompt](https://threatprompt.com) 的工作。
- **巧妙的代码生成工具**：`@craigba` 引起了大家对 **AlphaCodium** 的关注，这是一个利用类似于 **GANs** 的对抗性技术来生成高质量代码的工具，并邀请其他人查看 [Tamar Friedman 的简短介绍](https://twitter.com/itamar_mar/status/1747957348293824676) 并探索其 [GitHub 仓库](https://github.com/Codium-ai/AlphaCodium)。
- **对 AI 访谈见解的赞赏**：对 **Jeremy 的 Latent Space 访谈**中的对话表示认可，特别是赞扬了一个围绕深度学习和大型科技公司之外的生产力的问题，详见 "[The End of Finetuning](https://www.latent.space/p/fastai)"。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 总结

- **寻找免费发型应用无果**：`@soundblaster__` 正在寻找一款**免费的更换发型应用**，但即使在 **Google 的第一页和第二页**搜索了不需要注册后付费的选项后，仍然碰壁。

---

# 第二部分：渠道详细总结与链接

### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1204334882218713088) (1312 条消息🔥🔥🔥): 

- **AI 性能与训练讨论**：用户讨论了各种 AI 模型和训练技术，特别关注了 **Mixtral**、**Nous Mixtral DPO** 和 **Miqu 70B** 等模型。他们在效率和能力方面将这些模型与 OpenAI 的 GPT 模型进行了比较。
- **LLM 社区与资源共享**：对话涉及 **PolyMind**，这是一个针对 Mixtral Instruct 的项目，具有 Python 解释和语义 PDF 搜索等功能。[分享了其 GitHub 仓库](https://github.com/itsme2417/PolyMind)，但也提到了关于它被删除的 Reddit 帖子，表明 `/r/LocalLLaMA` 子版块可能存在审核问题。
- **技术规格与设备讨论**：用户交流了适合在本地运行大型模型的计算硬件见解。他们辩论了像 Apple 的 M2 和 AMD 的 Epyc 等芯片的内存带宽，以及为 AI 推理任务配置大容量 RAM 的实用性。
- **社区动态**：讨论了各个以 AI 为中心的 Discord 服务器的基调，并对包括 **TheBloke 的服务器**、**SillyTavern** 和 **EleutherAI** 在内的服务器的对话性质和社区行为发表了看法。
- **Reddit 审核与政策**：对 Reddit 上模糊的审核做法提出了批评，特别是关于 `/r/LocalLLaMA` 子版块的帖子和透明度问题。用户对在分享有用的 AI 相关内容时规则执行明显不公表示沮丧。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1111983596572520458/1111984430945402960/1202079366134382633)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [bartowski/dolphin-2.6-mistral-7b-dpo-exl2 · Hugging Face](https://huggingface.co/bartowski/dolphin-2.6-mistral-7b-dpo-exl2)：未找到描述
- [Sfm Soldier GIF - Sfm Soldier Tf2 - 发现并分享 GIF](https://tenor.com/view/sfm-soldier-tf2-meme-american-gif-24728385)：点击查看 GIF
- [Oh My God Its Happening GIF - Oh My God Its Happening Ok - 发现并分享 GIF](https://tenor.com/zIIQ.gif)：点击查看 GIF
- [Transformer 推理算术 | kipply 的博客](https://kipp.ly/transformer-inference-arithmetic/)：kipply 关于她所做、所读或所观察事物的博客
- [Apple Apple Mac GIF - Apple Apple Mac Apple Mac Studio - 发现并分享 GIF](https://tenor.com/view/apple-apple-mac-apple-mac-studio-apple-mac-studio2022-apple-mac-studio-m1-gif-25082394)：点击查看 GIF
- [GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k · Hugging Face](https://huggingface.co/GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k)：未找到描述
- [Qwen-1.5 72B：中国的 AI 巨头击败了 Mistral 7B 和 GPT4！（AI 新闻）🐉](https://www.youtube.com/watch?v=-oD2JVPD9Nc)：东方凭借 Qwen 1.5 下了战书，这是一款打破边界、改变游戏规则的 LLM。它不仅在数学和编程能力上与 ChatGPT4 旗鼓相当，而且...
- [GitHub - itsme2417/PolyMind：一个由多模态、函数调用驱动的 LLM webui。](https://github.com/itsme2417/PolyMind)：一个由多模态、函数调用驱动的 LLM webui。 - GitHub - itsme2417/PolyMind：一个由多模态、函数调用驱动的 LLM webui。
- [The Voices GIF - The Voices - 发现并分享 GIF](https://tenor.com/view/the-voices-gif-26307682)：点击查看 GIF
- [完全无审查的 GPT 来了 🚨 使用时请极度谨慎](https://www.youtube.com/watch?v=BntGOaMrB90)：在这段视频中，我们评测了 Wizard Vicuna 30B Uncensored。这款 LLM 已移除了所有审查。你们期待已久，现在它终于来了...
- [Hugging Face – 构建未来的 AI 社区。](https://huggingface.co/posts)：未找到描述
- [Reddit - 深入探索任何事物](https://www.reddit.com/r/LocalLLaMA/comments/1akvwdp/we_need_to_talk_about_pol)：未找到描述
- [The Voices Meme GIF - The Voices Meme Cat - 发现并分享 GIF](https://tenor.com/view/the-voices-meme-cat-gif-23917781)：点击查看 GIF
- [GitHub - ml-explore/mlx-examples：MLX 框架中的示例](https://github.com/ml-explore/mlx-examples)：MLX 框架中的示例。通过在 GitHub 上创建账号来为 ml-explore/mlx-examples 的开发做出贡献。
- [abacusai/Smaug-72B-v0.1 · Hugging Face](https://huggingface.co/abacusai/Smaug-72B-v0.1)：未找到描述
- [Open LLM 排行榜 - 由 HuggingFaceH4 创建的 Hugging Face Space](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)：未找到描述
- [Reddit - 深入探索任何事物](https://www.reddit.com/r/LocalLLaMA/comments/1akgebk/how_i_got_finetuning_mistral7b_to_not_suck/)：未找到描述
- [Reddit - 深入探索任何事物](https://www.reddit.com/r/LocalLLaMA/comments/1akvwdp/we_need_to_talk_about_polymind/)：未找到描述
- [Reddit - 深入探索任何事物](https://www.reddit.com/r/LocalLLa)：未找到描述
- [THE DECODER](https://www.google.com/amp/s/the-decoder.com/ccp-releases-politically)：人工智能正在改变世界。THE DECODER 为您带来关于 AI 的所有新闻。

---

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1204340730848350218) (503 messages🔥🔥🔥): 

- **希腊悲剧还是编码策略？**：Discord 用户讨论了 `Thespis 0.8` 的起源和性质，`@c.gato` 澄清其名称源于希腊悲剧，并提供了命名背后的历史背景。`@billynotreally` 和其他人对该术语进行了戏谑性的混淆，将其比作 "sepsis"（败血症）和一个挪威语单词。
- **详述 DPO 结果**：`@dreamgen` 请求公开 DPO (Direct Preference Optimization) 运行指标的 Weights & Biases (wandb)。`@c.gato` 指出准确率应上升至 100%，且边际（margins）应随时间增加，并提供了一个 wandb 项目示例[链接](https://wandb.ai/jondurbin/projects)（此链接是原始聊天的一部分，并不指向真实目的地）。
- **Lorebooks 作为 Prompt Engineering 工具**：在关于使用 Lorebooks 增强角色扮演故事的对话中，`@johnrobertsmith` 质疑了它们在 Prompt Engineering 之外的有效性。`@mrdragonfox` 建议它们在无需用户提示的情况下注入信息，并在角色扮演过程中强化重要元素。
- **关于合并与模型训练的讨论**：用户辩论了合并模型的优缺点。`@mrdragonfox` 对合并表示反对，而 `@mrg` 则认为用户只关心最终结果。讨论趋向于一个普遍共识：数据集的创建比单纯的模型合并更重要。
- **思考 Benchmark 和数据集策略**：关于为了 Benchmark 而合并模型的价值与创建数据集的价值之间存在争论，`@flail_` 认为 Benchmark 可能会被合并扭曲，而 `@mrdragonfox` 肯定了真正的价值在于数据集，而不仅仅是模型的合并。

**提到的链接**：

- [Artefact2/BagelMIsteryTour-v2-8x7B-GGUF · Hugging Face](https://huggingface.co/Artefact2/BagelMIsteryTour-v2-8x7B-GGUF)：未找到描述
- [PotatoOff/HamSter-0.2 · Hugging Face](https://huggingface.co/PotatoOff/HamSter-0.2)：未找到描述
- [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning)：未找到描述
- [Objective | docs.ST.app](https://docs.sillytavern.app/extras/extensions/objective/)：Objective 扩展允许用户指定一个 Objective（目标），供 AI 在聊天过程中努力实现。
- [TheBloke/Beyonder-4x7B-v2-GGUF · Hugging Face](https://huggingface.co/TheBloke/Beyonder-4x7B-v2-GGUF)：未找到描述
- [jondurbin](https://wandb.ai/jondurbin/projects)：Weights & Biases，机器学习开发者工具

---

### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1204406569534754847) (11 messages🔥): 

- **寻求规避安全限制**：用户 `@mmarkd` 询问如何从 Llama2 70B instruct 中**移除安全护栏**，并分享了一篇 [LessWrong 帖子](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from)，讨论了 LoRA 微调如何以低成本撤销安全训练，但指出该帖缺乏具体的操作细节。
- **对 AI 伦理限制的沮丧**：`@mmarkd` 提到由于过度的伦理护栏，**很难使用** Llama2 70B instruct 等模型执行任务，甚至无法协助进行无害的代码重构。
- **减少限制的建议**：`@flail_` 建议使用 toxicdpo、spicyboros、airoboros、dolphin 和 hermes 等*替代微调模型*，以规避过度的安全特性。
- **精细微调技巧**：`@london` 建议在**微调期间合并数据集**可以提高对各种模型参数的控制。
- **LoRA 的训练属性受到质疑**：`@cogbuji` 讨论了 LoRA 微调提供的改变程度，引用了 QLoRa 论文，该论文表明将 LoRA 应用于所有 Transformer 层可能**达到全量微调（full fine-tuning）的性能**。
- **寻求在 Colab 上使用 LM 的支持**：用户 `@thiagoribeirosnts` 在遇到困难后，正寻求在 Google Colab 上使用 **wizardLM 或 LLaMA 2** 的帮助。
- **思考 LoRA 的正确方向**：`@gandolphthewicked_87678` 权衡是继续使用 **Mistral 7b** 进行 LoRA 微调还是切换到基础模型（base model），并寻求建议。

**提到的链接**：

[LoRA Fine-tuning Efficiently Undoes Safety Training from Llama 2-Chat 70B — LessWrong](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from)：作为 SERI ML Alignment Theory Scholars Program - 2023 夏季班的一部分产出，由 Jeffrey Ladish 指导。…

---

### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1204422463107112991) (6 messages): 

- **Transformers 在层卸载（Layer Offloading）方面的局限性**：用户 `@mmarkd` 询问如何使用 Transformers 将特定层卸载到 GPU，类似于 llama.cpp 中的 `-ngl` 命令，但 `@itsme9316` 回复称 **Transformers 库无法在 CPU 和 GPU 之间拆分层**。

- **Meta 的 Sphere 项目引发关注**：`@spottyluck` 分享了 Facebook **Sphere 项目**的 [GitHub 链接](https://github.com/facebookresearch/Sphere)，并思考了 Meta 战略的转变（该战略曾被视为对 Google 的挑战）。他们提供了关于利用 Sphere 结合 common crawl 和大数据工具进行**频繁更新**的见解。

- **在 MLX 上实现 LLaMa 2 的疑问**：`@lushboi` 正在探索在 **Apple Silicon 的 MLX** 上实现 **LLaMa 2**，并询问了模型查询参数的结构，思考是否应将其调整为 multi-head 或 grouped-query 格式。

**提到的链接**：

[GitHub - facebookresearch/Sphere: Web-scale retrieval for knowledge-intensive NLP](https://github.com/facebookresearch/Sphere)：知识密集型 NLP 的 Web 规模检索。欢迎在 GitHub 上为 facebookresearch/Sphere 的开发做出贡献。

  

---



### OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1204505065105133640) (1 messages): 

- **DALL-E 图像获得元数据升级**：`@abdubs` 宣布在 **ChatGPT** 和 **OpenAI API** 中生成的图像现在包含符合 **C2PA 规范**的元数据。这使得验证图像是否由 OpenAI 产品生成成为可能，对社交平台和内容分发者很有帮助。在[帮助文章](https://help.openai.com/en/articles/8912793-c2pa-in-dall-e-3)中阅读详细信息。
  

---

### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1204337278982561852) (300 条消息🔥🔥): 

- **AI 审查疑虑阻碍使用**：用户 `@kotykd` 和 `@arturmentado` 讨论了 AI 审查对用户体验的影响，`@kotykd` 主张 AI 输出应给予用户自由，而 `@arturmentado` 则解释了防止滥用的必要性。然而，`@kotykd` 认为这属于过度干预，而 `@arturmentado` 坚持认为保护措施有其正当理由。
  
- **AI 时代真实艺术性的代价**：`@infidelis` 等人辩论了在 Artstation 等平台上将 AI 生成的艺术作品伪装成人类创作的伦理问题，强调了披露的重要性。有人担心如果虚假呈现 AI 艺术，平台会受损，`@lugui` 强调 OpenAI 的 TOS 要求在内容创作中诚实说明 AI 的作用。

- **元数据在溯源映射中的作用**：讨论重点在于从图像中删除元数据的重要性和简易性，`@whereyamomsat.com` 提供了关于 EXIF 数据的资源，`@heavygee` 评论了删除元数据后文件大小的变化。许多用户认为，由于元数据极易被篡改，对其进行操作是一种无关紧要的手段。

- **探索开源 AI 与商业解决方案**：`@infidelis` 和 `@arturmentado` 等参与者讨论了开源模型与 GPT-4 等商业 AI 的优劣。他们思考了这对创新进程的影响，一些用户预测具有竞争力的开源解决方案将会崛起。

- **AI 学习辅助工具备受争议的便利性**：用户辩论了 AI 的教育应用，`@germ_storm` 和 `@chief_executive` 讨论了其在学习和研究中相对于传统方法的有效性，认为 AI 在某些研究领域可以成为高效学习的强大工具。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/977259063052234752/1204505065105133640)：Discord 是通过语音、视频和文本进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。
- [Overview - C2PA](https://c2pa.org)：未找到描述
- [Portals - [TouchDesigner + Stable WarpFusion Project Files]](https://www.youtube.com/watch?v=zptPQbTScto)：由我本人创作。您可以访问这些 TouchDesigner 项目文件 [及其相应的 WarpFusion 设置]，以及更多项目文件、教程和实验...
- [Online photo metadata and EXIF data viewer | Jimpl](https://jimpl.com)：在线查看图像的 EXIF 数据。查找照片拍摄的时间和地点。从照片中删除元数据和位置以保护您的隐私。
- [NO C2PA - Remove C2PA Metadata](https://noc2pa.com/)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/midjourney/s/Hr8aYtbcYW)：未找到描述
- [Content Credentials](https://contentcredentials.org/verify…)：介绍内容认证的新标准。Content Credentials 为内容的创建或编辑方式提供了更深层次的透明度。

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1204348340935065621) (46 条消息🔥): 

- **登录问题排查**：用户 `@mikreyyy` 询问如何从所有设备注销其账号，但讨论线程中未提供解决方案或后续评论。

- **寻找 GPT 演示**：用户 `@glory_72072` 询问如何找到并使用 GPT 演示，但后续消息中缺少引导他们的细节或回复。

- **体验增强的叙事能力**：用户 `@blckreaper` 对 GPT 改进的叙事输出表示满意，提到响应更长且能更准确地遵循指令，尽管他们报告了 AI 未能完全遵守自定义指令（Custom Instructions）的问题。

- **关于回答限制的疑问**：`@ytzhak` 在与 GPT 进行 12 到 20 次交互后被封禁，面临可能的用量上限问题，这引发了关于自定义 GPT 使用限制的讨论——`@blckreaper` 将此归因于重新生成（Regeneration）的上限，即每次重新生成也计为一条消息。

- **Custom GPT 的问题与技巧**：用户报告了 Custom GPT 的各种挑战和解决方案，从超时错误（`@realspacekangaroo`）到 Explore 按钮消失（`@hawk8225`）——`@blckreaper` 建议通过退出并重新登录来修复。其他用户如 `@woodenrobot` 讨论了自定义指令中的指令冲突和 Token 限制，而 `@drinkoblog.weebly.com` 则评论了不同 GPT 模型的性价比。
  

---

### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1204475546591825991) (5 条消息): 

- **寻找受众与 GPTs 建议**：用户 `@loier` 正在寻求建议，想知道在哪里可以找到对他们的 GPTs 感兴趣的人，并希望学习如何改进模块和脚本设置。
- **优化 Prompt 中的角色交互**：`@_fresnic` 建议在角色交互对话的每个片段中加入提示，以更好地微调系统的响应，并表示如果提供截图或 gist，愿意审查他人的 Prompt/对话流。
- **寻求 OpenAI 联系方式以反馈 Teams 相关建议**：`@novumclassicum` 表达了与 OpenAI 讨论 Teams 集成的强烈愿望，并正在寻找联系该组织成员（包括 Sam Altman）的途径，以提供充满热情的反馈和改进建议。
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1204475546591825991) (5 条消息): 

- **寻求协作与指导**：用户 `@loier` 正在寻找一个讨论 **GPTs usage** 的社区，寻求关于 **setting up modules and scripts** 的建议，以增强其 GPTs 的性能。
- **优化 Prompt 中的角色交互**：用户 `@fresnic` 建议在 **整个对话片段中穿插角色交互提示**，而不仅仅是在初始的 system prompt 中，并表示如果通过 **screenshot 或 gist** 分享示例，他愿意进行审查。
- **寻求与 OpenAI 团队直接联系**：用户 `@novumclassicum` 表达了与 **OpenAI 代表** 讨论 **Teams integration** 的强烈愿望，并邀请像 **Sam Altman** 这样的人进行深入交流。
  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1204421684006879252) (1 条消息): 

- **创建你自己的聊天助手**：一项名为 **Hugging Chat Assistant** 的新功能允许用户轻松构建个人助手。正如 `@_philschmid` 所描述的，它包括可自定义的元素，如名称、头像和行为控制，并使用不同的 LLM，如 **Llama2** 或 **Mixtral**。该功能因无需单独存储自定义 Prompt 而受到赞誉。在 [Hugging Face Chat](https://huggingface.co/chat/assistants) 发现你的助手。

- **现在可以查看私有数据集**：`@julien_c` 宣布了一项更新，使 Dataset Viewer 可用于私有数据集。但是，此功能仅限 PRO 和 Enterprise Hub 用户使用，为数据探索和分析提供增强工具。阅读更多来自 [数据团队的工作](https://x.com/julien_c/status/1752716726012129577)。

- **Hub 上的 Synthetic 和 Croissant 数据标签**：预见到合成数据日益增长的重要性，`@vanstriendaniel` 在 HuggingFace Hub 上发布了一个新的 `synthetic` 标签。添加它是为了方便发现和共享合成数据集；只需在你的数据集卡片元数据中包含此标签即可。

- **在你的 HF 个人资料中展示博客文章**：据 `@not_so_lain` 称，当 HuggingFace 用户撰写博客文章时，它们现在将出现在他们自己的个人资料中。此功能作为一种新方式，用于突出社区内的个人贡献和见解。

- **Spaces 的迷你页眉**：`@lunarflu1` 为 HuggingFace Spaces 引入了 `header: mini` 选项，允许带有极简页眉的全屏显示，从而增强用户界面并专注于内容。

**提到的链接**：

- [Philipp Schmid (@_philschmid) 的推文](https://x.com/_philschmid/status/1753429249363452274)：介绍 Hugging Chat Assistant！🤵 只需点击两次，即可在 Hugging Face Chat 中构建你自己的个人 Assistant！类似于 @OpenAI GPTs，你现在可以创建自定义版本的 @huggingface Chat！🤯 一个 Ass...
- [HuggingChat - Assistants](https://huggingface.co/chat/assistants)：浏览由社区创建的 HuggingChat assistants。
- [Julien Chaumond (@julien_c) 的推文](https://x.com/julien_c/status/1752716726012129577)：@huggingface Hub 新功能：Dataset Viewer 现在也支持私有数据集了。你需要是 PRO 或 Enterprise Hub 用户。🔥 Dataset Viewer 允许团队了解他们的数据...
- [Daniel van Strien (@vanstriendaniel) 的推文](https://x.com/vanstriendaniel/status/1754466661321879814)：合成数据（Synthetic data）在 2024 年将变得非常重要，因此我们最近在 @huggingface Hub 上推出了一个新标签，以方便合成数据集的发现和共享。要添加此标签...
- [hafedh (@not_so_lain) 的推文](https://x.com/not_so_lain/status/1754302175159701910)：@huggingface 刚发现当你写博客文章（blogpost）时，它会显示在你的个人资料中 ❤️
- [lunarflu (@lunarflu1) 的推文](https://x.com/lunarflu1/status/1754800761303683436)：@huggingface Spaces 有新选项了 🤗！在元数据（metadata）中添加 `header: mini`，Space 将以全屏显示，并带有一个浮动的迷你页眉。
- [Sayak Paul (@RisingSayak) 的推文](https://x.com/RisingSayak/status/1753643552301617585)：在周末发布新版本已成为新常态 🤷🚀 推出 Diffusers 0.26.0，包含两个新的视频模型，支持多 IP-adapter 推理等 📹 发布说明 📜 https://github...
- [Sourab Mangrulkar (@sourab_m) 的推文](https://x.com/sourab_m/status/1752648062877798867)：新版本发布警报！🚨 PEFT v0.8.0 现已发布！🔥🚀✨ 查看完整发布说明：https://github.com/huggingface/peft/releases/tag/v0.8.0 [1/9]
- [Titus.von.Koeller (@Titus_vK) 的推文](https://x.com/Titus_vK/status/1754358165343461704)：bitsandbytes 的激动人心消息！我们很高兴地宣布发布新文档的初始版本！🧵https://huggingface.co/docs/bitsandbytes/main/en/index
- [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1753168382017327472)：冲啊！通过 AWQ 和 Flash Attention 2 实现更快的 CodeLlama 70B ⚡ 由 AutoAWQ、Transformers 和 @tri_dao 的 Flash Attention 2 提供支持。GPU VRAM 约 40GB 🔥 想亲自尝试吗？你需要做两个...
- [Sayak Paul (@RisingSayak) 的推文](https://x.com/RisingSayak/status/1754556329887166553)：感谢 @multimodalart，Stable Video Diffusion (SVD) 现在可以与 🧨 diffusers 一起使用了 ❤️ SVD v1.1 👉 https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1 指南 👉 https://hugg...
- [使用开源 LLMs 的 Constitutional AI](https://huggingface.co/blog/constitutional_ai)：未找到描述
- [NPHardEval 排行榜：通过复杂度类别和动态更新揭示 Large Language Models 的推理能力](https://huggingface.co/blog/leaderboards-on-the-hub-nphardeval)：未找到描述
- [Hugging Face 中的 Patch Time Series Transformer](https://huggingface.co/blog/patchtst)：未找到描述
- [Hugging Face Text Generation Inference 已支持 AWS Inferentia2](https://huggingface.co/blog/text-generation-inference-on-inferentia2)：未找到描述
- [SegMoE：Segmind Mixture of Diffusion Experts](https://huggingface.co/blog/segmoe)：未找到描述
- [ai geek (wishesh) ⚡️ (@aigeek__) 的推文](https://x.com/aigeek__/status/1753554577490690305)：终于有了一个最重要的排行榜。@huggingface 的新 Enterprise Scenarios 排行榜刚刚发布。它评估了语言模型在真实企业用例中的性能。...
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/events/879548962464493619/1201999637360148520)：Discord 是通过语音、视频和文本进行交流的最简单方式。聊天、聚会，并与您的朋友和社区保持联系。

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1204338674117115964) (170 条消息🔥🔥): 

- **XP Boost 辩论**：`@lunarflu` 讨论了为 HuggingFace 社区中的特定角色增加 **XP 获取倍率** 的想法，并提到 `server booster` 是获得此奖励的合适候选者。
- **关于 Docker AI 处理位置的咨询**：`@criticaldevx` 提出了一个关于 **Docker 文本生成推理 (text generation inference)** 是在用户设备还是 HuggingFace 服务器上处理的问题，消息中未提供明确答案。
- **错误排查**：包括 `@leifer_` 和 `@criticaldevx` 在内的多位用户报告在 [HuggingFace 的聊天功能](https://huggingface.co/chat/) 中遇到 ***504 错误***，表明服务可能已宕机。
- **协作与贡献**：`@lunarflu` 表示愿意研究 **fellowships** 以帮助提升用户的影响力，而 `@ufukhury` 寻求关于如何向 HuggingFace 贡献的建议，但未获得具体的后续步骤。
- **Accelerate 加载状态问题**：`@bit0r` 和 `@doctorpangloss` 就使用 **Accelerate** 的 load_state 功能恢复 checkpoint 时遇到的问题进行了 ***详细的排查*** 对话，其中对使用 lxd 等容器以及代码细节的有效性提出了质疑。

**相关链接**：

- [LoRA Studio - 由 enzostvs 创建的 Hugging Face Space](https://huggingface.co/spaces/enzostvs/lora-studio)：未找到描述
- [快速入门 (Quick tour)](https://huggingface.co/docs/accelerate/quicktour#saveload-entire-states)：未找到描述
- [Rockwell Retro Encabulator](https://youtu.be/RXJKdh1KZ0w?si=9sF3L2f4S2YXCaq8)：Rockwell Automation 的最新技术
- [GitHub - HSG-AIML/MaskedSST: Scheibenreif, L., Mommert, M., & Borth, D. (2023) 的代码库。Masked Vision Transformers for Hyperspectral Image Classification, In CVPRW EarthVision 2023](https://github.com/HSG-AIML/MaskedSST)：Scheibenreif, L., Mommert, M., & Borth, D. (2023) 的代码库。Masked Vision Transformers for Hyperspectral Image Classification, In CVPRW EarthVision 2023 - GitHub - HSG-AIML/MaskedSS...
- [GitHub - Sanster/tldream: 一个微型扩散绘图应用](https://github.com/Sanster/tldream)：一个微型扩散绘图应用。通过在 GitHub 上创建账号来为 Sanster/tldream 的开发做出贡献。

---

### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1204442473263276103) (7 条消息): 

- **Cat Game Research 释放四足毛茸力量**：`@technosourceressextraordinaire` 分享了他们为 Bat Country Entertainment 正在开发的视频游戏创建的名为 Leela 的猫咪模型。据说该游戏正在开创**首个使用 ML 和 AI 的四足角色控制器**，更多详情请访问 [badcatgame.com](https://badcatgame.com)。

- **对语言模型和代码的学术深入探讨**：`@vipitis` 正沉浸在一篇综述论文中，该论文系统回顾了使用语言模型进行代码处理的进展。该论文涵盖了广泛的模型、数据集和 700 多项工作，并在 GitHub 上有一个持续更新的线程，以及要添加到 [HuggingFace collection](https://huggingface.co/collections/Vipitis/code-evaluation-6530478d8e4767ecfe1bc489) 的参考资料。完整论文可在 [arXiv](https://arxiv.org/abs/2311.07989) 获取。

- **请勿发送 Discord 邀请**：`@cakiki` 发布了提醒，要求遵守频道指南，禁止发送 Discord 邀请。该规则执行针对用户 `@1134164664721350676`，随后又对 `@985187584684736632` 再次强调。

- **阿里巴巴的 AI 表现优于竞争对手**：`@dreamer1618` 强调了一篇文章，指出阿里巴巴最新的人工智能模型 **Qwen 1.5** 在多项基准测试中表现优于 ChatGPT 和 Claude。讨论这些进展的文章可在 [wccftech.com](https://wccftech.com/alibabas-latest-a-i-beats-gpt-3-5-claude-in-multple-benchmark-tests/) 查看。

- **探索创新微调论文 'RA-DIT'**：`@austintb.` 讨论了实施一篇名为 "RA-DIT: Retrieval-Augmented Dual Instruction Tuning" 的极具前景的论文中的技术的计划。该论文为检索器（retriever）和语言模型微调提出了先进的方法论，完整文档见 [arXiv](https://arxiv.org/abs/2310.01352)。

**提到的链接**：

- [Alibaba's Latest A.I. Beats GPT-3.5, Claude In Multple Benchmark Tests](https://wccftech.com/alibabas-latest-a-i-beats-gpt-3-5-claude-in-multple-benchmark-tests/)：随着 2024 年全球人工智能竞赛的强劲开局，中国科技巨头阿里巴巴集团也宣布了其 Qwen 人工智能模型的最新迭代。一个...
- [Bad Cat Game](https://badcatgame.com)：你是一只猫，而且是个混蛋 —— 这是一款由 Bat Country Entertainment LLC 目前正在开发的动作冒险 RPG 游戏。
- [RA-DIT: Retrieval-Augmented Dual Instruction Tuning](https://arxiv.org/abs/2310.01352)：检索增强语言模型 (RALMs) 通过访问外部数据存储中的长尾和最新知识来提高性能，但构建起来具有挑战性。现有方法需要...
- [Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code](https://arxiv.org/abs/2311.07989)：在这项工作中，我们系统地回顾了使用语言模型进行代码处理的最新进展，涵盖了 50 多个模型、30 多个评估任务、170 多个数据集和 700 多篇相关论文。我们分解了...

  

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1204373522416279563) (10 条消息🔥): 

- **作品赞赏**：`@furquan.sal` 在一条简单的消息中对某项作品表示了赞赏，称：“*Impressive Bro! Liked it 💛*”。
  
- **前端技术咨询**：针对 `@furquan.sal` 关于所用前端框架的提问，`@wubs_` 详细说明了他们的项目前端是使用 **React** 构建的，并使用 **Jotai** 进行状态管理。他们还分享了其 [开发时间线](https://www.artforgelabs.com/post/art-forge-labs-development-timeline-ai-art-innovation)，并欢迎提问和互动。

- **TensorLM-webui 发布**：`@ehristoforu` 宣布了 **TensorLM-webui**，这是一个基于 **LLaMA** 的 **GGML 格式 LLM** 的 Gradio Web UI，并鼓励用户从 [GitHub](https://github.com/ehristoforu/TensorLM-webui) 克隆该项目，或在 [Hugging Face Spaces](https://hf.co/spaces/ehristoforu/TensorLM-for-HF) 上测试小型 Demo。

- **从草图到时尚**：`@tony_assi` 展示了 **Sketch to Fashion Collection**，这是一个将草图转化为时尚设计的应用，可在 [Hugging Face Spaces](https://huggingface.co/spaces/tonyassi/sketch-to-fashion-collection) 上使用。他们随后询问了关于图像生成 API 的可能性。

- **BLOOMChat-v2 发布**：`@urmish.` 分享了关于 **BLOOMChat-v2** 的信息，这是一个拥有 176B 参数、支持 32K 序列长度的多语言语言模型。该模型很快将配套提供 API，且相比早期模型有显著改进；更多详情可见 [Twitter 总结](https://twitter.com/SambaNovaAI/status/1754928815590277146) 和 [详细博客文章](https://sambanova.ai/blog/bloomchat-v2)。

**提到的链接**：

- [Sketch To Fashion Collection - tonyassi 的 Hugging Face Space](https://huggingface.co/spaces/tonyassi/sketch-to-fashion-collection)：未找到描述
- [Introducing BLOOMChat 176B - 基于多语言对话的 LLM](https://sambanova.ai/blog/bloomchat-v2)：我们很自豪地发布了 BLOOMChat-v2，一个 32K 序列长度、176B 参数的多语言语言模型。
- [GitHub - ehristoforu/TensorLM-webui: 基于 LLaMA 的 LLM 模型简单现代的 Web UI。](https://github.com/ehristoforu/TensorLM-webui)：基于 LLaMA 的 LLM 模型简单现代的 Web UI。 - GitHub - ehristoforu/TensorLM-webui: Simple and modern webui for LLM models based LLaMA.
- [TensorLM - Llama.cpp UI - ehristoforu 的 Hugging Face Space](https://hf.co/spaces/ehristoforu/TensorLM-for-HF)：未找到描述
- [Art Forge Labs 开发时间线 - AI 艺术创新](https://www.artforgelabs.com/post/art-forge-labs-development-timeline-ai-art-innovation)：欢迎来到 Art Forge Labs —— 我们在 AI 艺术创新领域的旅程既迅速又具有革命性。从早期的基础设置到引领 AI 驱动艺术的潮流，我们的道路...

---

### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1204419830711717928) (101 条消息🔥🔥): 

- **Mamba 演讲推迟**：`@lunarflu` 宣布将 **Mamba** 论文演讲推迟到下周，并邀请其他人在期间进行分享。本周尚未确定具体的论文或主题。
- **读书小组关注点**：`@tonic_1` 和一位朋友表示有兴趣在下一次 **Reading Group** 中分享一篇关于 [用于时间序列预测的 decoder-only 基础模型论文](https://arxiv.org/pdf/2310.10688.pdf)，并协调在周五进行演示，引发了热烈讨论（“极度兴奋”）。
- **读书小组资源 GitHub 仓库**：`@chad_in_the_house` 创建了一个 [GitHub 仓库](https://github.com/isamu-isozaki/huggingface-reading-group)，用于汇集 **HuggingFace Reading Group** 过去的演讲内容和录音，以便于访问以及未来可能在 YouTube 上进行传播。
- **S4 和 Mamba 讨论预热**：`@ericauld` 正准备讲解 **Mamba** 和 **S4**，并征求社区意见，了解大家认为演讲中哪些方面最有价值。他们建议重点关注他人对这些论文的迭代以及潜在的未来发展。
- **ML/AI 学习路径**：多位用户分享了针对机器学习和 AI 初学者的建议、资源和入门点。建议的方法包括参与读书小组、使用高级库、开展特定项目以及学习线性代数等基础知识。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/events/879548962464493619/1203285706949009448)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。
- [Discord - 与朋友和社区聊天的新方式](https://discord.com/events/8795)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。
- [在 Hugging Face 使用 ML-Agents](https://huggingface.co/docs/hub/ml-agents)：未找到描述
- [Spaces - Hugging Face](https://huggingface.co/spaces)：未找到描述
- [Civitai | 分享你的模型](https://civitai.com/user/Yamer)：未找到描述
- [GitHub - isamu-isozaki/huggingface-reading-group: 该仓库的目标是预先汇编 Huggingface 读书小组过去所有的演讲内容](https://github.com/isamu-isozaki/huggingface-reading-group)：该仓库的目标是预先汇编 Huggingface 读书小组过去所有的演讲内容 - GitHub - isamu-isozaki/huggingface-reading-group...
- [SDXL Unstable Diffusers ヤメールの帝国 ☛ YamerMIX - V11 + RunDiffusion | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/84040/sdxl-unstable-diffusers-yamermix)：有关商业咨询、商业许可、定制模型/委托、数据集的大规模图像标注和咨询，请联系我...
- [Mobile ALOHA](https://sota.beehiiv.com/p/mobile-aloha?utm_source=sota.beehiiv.com&utm_medium=newsletter&utm_campaign=mobile-aloha)：未找到描述
- [这款将取代你在麦当劳工作的 AI](https://www.youtube.com/watch?v=HNlS7GyVYK4)：这里展示了由 AI 驱动并使用远程操作数据训练的机器人的未来。查看我的排行榜网站：https://leaderboard.bycloud....

---

### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1204341495663169536) (7 条消息): 

- **离题指导**：`@juancopi81` 指导 `@eugenekormin` 将服务器端代码讨论移至适当频道，强调 **diffusion-discussions** 是用于 Diffusion 模型相关话题的。

- **Diffusion 新手寻求指导**：`@_elab` 是 Diffusion 模型的新手，就几个关键参数寻求建议，包括时间步 (`T`)、beta 调度、训练速度以及针对使用 Stanford Cars 数据集进行图像合成的硬件要求。

- **训练尝试与错误**：`@bitpattern` 分享了使用 **Stable Diffusion** 进行图像生成的训练参数日志，包括 batch sizes、梯度累积 (gradient accumulation) 和优化步骤，同时提到打算通过可能减少图像数量来优化流程。

- **推荐 Diffusion 课程**：`@juancopi81` 向 `@_elab` 推荐了 HuggingFace 和 FastAI 关于 Diffusion 模型的课程，以便更深入地理解 Diffusion 概念并解答 `@_elab` 的疑问。

- **Unet2dconditionmodel 咨询**：`@blankspace1586` 正在寻求关于使用 **Unet2dconditionmodel** 进行非文本 Embedding 验证的指导，因为该场景似乎缺乏示例。

- **Cross Attention 中的 Per-Token Mask 咨询**：`@jfischoff` 询问了在 diffusers pipeline 中为 Cross Attention 实现 per-token mask 的可能性，旨在将 token 的影响限制在 latent space 的特定区域。
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1204343311486427146) (5 条消息): 

- **寻求特定领域的对话模型**：`@alexkstern` 正在考虑使用教科书切片在课程大纲上微调 (fine-tuning) **LLama chat** 或 **Mistral model**，以培养模型在特定主题上的专业知识。目标是创建一个在进一步微调以提高其教育对话能力之前，能够理解领域特定内容的模型。
- **超越教科书 - 音频作为数据**：`@technosourceressextraordinaire` 建议内容可能已经存在于预训练数据 ("the pile") 中，但建议考虑使用音频转文本的转录（例如来自 **Whisper** 的转录）来创建数据集。
- **关于数据集效用的澄清**：`@alexkstern` 寻求确认，先使用教科书内容进行领域知识微调，然后再进行对话上下文微调，是否是一个有效的策略。
- **微调失败的挫折**：`@zen_maypole_40488` 在尝试在 OpenAI 平台上微调模型时遇到了 `InvalidRequestError`，这表明微调请求 URL 或 API 使用可能存在问题。
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1204341495663169536) (7 条消息): 

- **Diffusion 新手寻求指导**：`@_elab` 询问了关于为使用 Stanford Cars 数据库进行图像合成的 Diffusion 模型设置参数的建议，特别是关于 `T` (时间步) 的最佳值、计算 `betas` 以及训练的硬件要求。他们担心训练速度以及是否需要多个 GPU。
  
- **图像合成训练进行中**：`@bitpattern` 分享了他们使用混合精度 (mixed precision) 和梯度累积 (gradient accumulation) 等技术的图像合成模型训练日志快照。日志显示使用了预训练模型，且数据集分辨率为 512。

- **为 Diffusion 模型学习者提供指导**：针对 `@_elab` 的提问，`@juancopi81` 建议查看 HuggingFace 和 FastAI 关于 Diffusion 模型的课程，这些是理解 Diffusion 模型参数和训练过程的有用资源。

- **关于 Unet2dconditionmodel Pipeline 的咨询**：`@blankspace1586` 讨论了他们成功为以非文本 Embedding 作为条件的 Unet2dconditionmodel 实现了训练循环，但对于在验证时应使用哪个可以传递该 Embedding 的 pipeline 表示不确定。

- **寻求 Pipeline 的 Cross Attention Mask**：`@jfischoff` 提出了一个技术疑问，关于是否可以在 diffusers pipeline 中对 Cross Attention 应用 per-token mask，旨在将 token 的影响限制在特定的 latent 区域。
  

---

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1204347504569618462) (141 条消息🔥🔥): 

- **LM Studio 对 Intel Mac 的支持**：`@robert.bou.infinite` 建议，虽然 **LM Studio** 理论上可以支持 Intel Mac，但旧款 Intel Mac 缺乏强大的 GPU，可能会导致性能不佳。他们建议仍在使用 Intel Mac 的用户通过远程控制兼容机器来操作，并可以通过私信（DM）获取云服务商的推荐。

- **TTS 和图像支持问题**：用户正在探索与 **text-to-speech (TTS)** 和 **图像支持** 相关的特性。`@joelthebuilder` 在 iOS 上运行 AI 语音时遇到困难，`@enragedantelope` 询问如何筛选带有 "vision adapters" 的模型，`@lyracon` 则在寻求 OCR 后处理的技巧。

- **模型兼容性与操作**：`@justmarky` 和 `@robert.bou.infinite` 为 `@xermiz.` 提供了关于 RTX 3060 兼容模型的指导；`@robert.bou.infinite` 和 `@heyitsyorkie` 提供了反馈渠道，用于讨论 **LM Studio** 的功能和报告 Bug。

- **提示策略、量化与内存需求**：`@kujila` 讨论了构建最低配置要求的嵌入式 llama 应用；`@robert.bou.infinite` 谈到了量化和运行超大型模型的挑战，例如 **Giant Hydra MOE 240b**，该模型甚至在 Hugging Face 强大的 A100x4 配置上也未能成功加载。

- **执行代码与其他离线 AI 工具**：`@artik.ua` 询问是否有像 ChatGPT 一样可以执行代码和浏览网页的软件；`@curiouslycory` 推荐了 **ollama + ollama-webui** 等工具，这些工具支持图像输入和文档对话，是针对不同 AI 任务的 **LM Studio** 替代方案。

**提到的链接**：

- [Hugging Face – 构建未来的 AI 社区。](https://huggingface.co): 未找到描述
- [coqui (Coqui.ai)](https://huggingface.co/coqui): 未找到描述
- [ibivibiv/giant-hydra-moe-240b · Hugging Face](https://huggingface.co/ibivibiv/giant-hydra-moe-240b): 未找到描述
- [为什么 STACK 这么快？](https://www.youtube.com/watch?v=N3o5yHYLviQ): 在这段视频中，我们探讨了 Stack，有时也被称为硬件栈（Hardware Stack）、调用栈（Call Stack）、程序栈（Program Stack）……请记住，这是为了教学目的而制作的……
- [使用 AutoGPTQ 和 transformers 让 LLM 更轻量化](https://huggingface.co/blog/gptq-integration): 未找到描述
- [非官方 LMStudio FAQ！](https://rentry.org/LMSTudioFAQ): 欢迎来到非官方 LMStudio FAQ。在这里，你可以找到我们在 LMStudio Discord 中收到的最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源软件……
- [使用 LM Studio 在没有网络连接的情况下在本地电脑上运行助手 LLM (ChatGPT)！](https://youtu.be/sLOOLbKM1ys?si=T8H50jrrY8_toqO8): 我们探讨了在本地运行基于大语言模型的助手的需求、如何运行它们，以及使用 LM Studio 的应用场景。
- [通过 RoPE 缩放扩展上下文窗口大小 · ggerganov/llama.cpp · Discussion #1965](https://github.com/ggerganov/llama.cpp/discussions/1965): 简介：这是一场关于最近提出的扩展 LLaMA 模型上下文大小策略的讨论。最初的想法提出于：https://kaiokendev.github.io/til#extending-context-t...

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1204375720147689483) (73 messages🔥🔥): 

- **用户在模型偏好上纠结**：`@hexacube` 取消了 ChatGPT 的订阅，并对 Guanaco 33b q5 等替代方案表达了复杂的感受。与此同时，`@fabguy` 分享了使用 120B 模型生成长篇故事的积极体验，建议配合特定策略使用大模型。`@goldensun3ds` 详细介绍了一种结构化的叙事方法，并对 AI 给出特定指令，尽管在转向本地模型和测试扩展上下文（extended contexts）时遇到了一些初期磨合问题。
  
- **针对个人数据微调 AI 模型**：针对 `@goofy_navigator` 关于在个人数据上训练模型的询问，`@heyitsyorkie` 指出这可以在外部实现，但无法在 LM Studio 内部完成。Heyitsyorkie 进一步分享了一个 [YouTube 教程](https://www.youtube.com/watch?v=MDA3LUKNl1E)，演示了使用自定义数据集微调模型的过程。

- **本地模型故障排除**：在包括 `@rumpelstilforeskin` 和 `@joelthebuilder` 在内的几位用户报告某些模型输出结果欠佳或行为异常后，其他用户建议检查正在使用的 LM Studio 版本，或切换到较新的模型，如 Dolphin 或 Nous Hermes。

- **硬件讨论与模型建议**：`@kujila` 和 `@goldensun3ds` 讨论了在不同系统上运行 Goliath 120B LongLORA 模型的可行性和性能，结果各异，并强调了 RAM 和 VRAM 的需求。`@heyitsyorkie` 就笔记本电脑使用的理想模型量化（quant）级别向 `@goofy_navigator` 提供了建议，并建议在硬件受限的情况下坚持使用 7b 模型。

- **寻找适合专业 AI 用途的模型**：`@supersnow17` 正在寻找针对数学和物理进行微调的模型，对此 `@fabguy` 建议，对于专门解决数学问题，其他类型的工具可能更合适。`@juanrinta` 询问了用于阅读杂乱文本的 AI，并被引导至 RAG 相关资源。

**提到的链接**：

- [Fine-tuning Llama 2 on Your Own Dataset | Train an LLM for Your Use Case with QLoRA on a Single GPU](https://www.youtube.com/watch?v=MDA3LUKNl1E)：全文教程（需要 MLExpert Pro）：https://www.mlexpert.io/prompt-engineering/fine-tuning-llama-2-on-custom-dataset 学习如何微调 Llama ...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18ljvxb/llm_prompt_format_comparisontest_mixtral_8x7b/)：未找到描述

  

---


### LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1204472573740716053) (1 messages): 

- **LM Studio v0.2.14 中的关键 Bug 修复**：`@yagilb` 宣布了 **LM Studio v0.2.14** 中的重要 Bug 修复，解决了中断模型生成时 UI 冻结以及粘贴长输入导致卡死的问题。敦促用户通过 [LM Studio 官网](https://lmstudio.ai/)或应用的“Check for updates...”功能进行更新。

**提到的链接**：

[👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/)：发现、下载并实验本地 LLM

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1204335083981504514) (12 messages🔥): 

- **LM Studio 为所有人简化 LLM**：`@drawless111` 强调了 [LM Studio 运行 LLM 的能力](https://blog.stackademic.com/lm-studio-experience-the-magic-of-llms-with-zero-technical-expertise-4561039a01ed)，其简单的界面无需编程，使任何人都能轻松下载并使用预训练模型。
- **关于 LLM 文件夹默认重置的提醒**：`@msz_mgs` 观察到在最近的一次更新后，LLM 文件夹位置被重置为默认值，但其他设置未受影响。
- **禁用 GPU 以加载模型**：`@georde` 报告了一个问题，即尝试使用 GPU 时模型加载失败，但在禁用 GPU 后可以正常工作。
- **对提升长文本粘贴速度的赞赏**：`@msz_mgs` 对 LM Studio 中粘贴长文本速度的提升表示赞赏。
- **对 LM Studio 增强功能的建议**：`@justmarky` 建议增加新功能，例如模型下载完成时的提示音、收藏像 TheBloke 这样的发布者（release users）的能力，以及按大小和用户过滤模型。`@fabguy` 引导他们在指定频道发布功能请求，并告知了如何按特定发布者过滤模型。

**提到的链接**：

[LM Studio: experience the magic of LLMs with Zero technical expertise](https://blog.stackademic.com/lm-studio-experience-the-magic-of-llms-with-zero-technical-expertise-4561039a01ed)：在任何电脑上实现零配置本地 LLM 的指南。

  

---

### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1204340234620239912) (27 messages🔥): 

- **Tornado 机箱风扇 vs. 紧凑型压力方案**：`@666siegfried666` 讨论了硬件的最佳冷却方案，提到使用配备 **2x180mm 风扇**并产生显著气流的机箱，或者选择较小的机箱配备 **3x120mm Arctic P120 风扇**，以获得良好的静压和性价比。
  
- **探索用于双本地模型的 AMD 8700g**：`@bobzdar` 询问了将 **AMD 8700g 与 DDR5 RAM** 以及 4090 GPU 配对运行语言和代码模型时的性能数据，并建议使用 128 GB RAM 以可能避免瓶颈。
  
- **关于 APU 运行模型潜力的讨论**：`@ptable` 和 `@bobzdar` 交流了关于 APU 的系统 RAM 是否会成为运行模型限制的看法。`@bobzdar` 分享说 APU 可以直接寻址 32GB，且内存控制器具有很高的超频潜力，系统 RAM 速度约为 **100GB/s**。

- **对 APU 的 AI 性能持怀疑态度**：`@goldensun3ds` 建议对 APU 在 AI 任务中的性能采取谨慎态度，类似于对待 ARC GPU 的方式，而 `@rugg0064` 指出，尽管比系统 RAM 快，但与 VRAM 相比仍有差距。
  
- **等待 APU 实际测试结果**：`@bobzdar` 决定通过订购一套配置来测试 APU 的性能，如果结果不理想，准备选择 **7950x3d**，这引发了 `@quickdive.` 等人的回应，他们认为 7950x3d 从一开始可能就是更好的选择。

**提到的链接**：

[I Saw W Gus Fring GIF - I Saw W Gus Fring Gus - Discover &amp; Share GIFs](https://tenor.com/view/i-saw-w-gus-fring-gus-gustavo-deleted-gif-25440636)：点击查看 GIF

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1204382807263019008) (15 messages🔥): 

- **弹出问题需要修复**：用户 `@goldensun3ds` 提到一个棘手的 Bug：在消息处理过程中弹出（eject）模型会导致程序无限期挂起。他们建议需要一种无需等待 Token 输出即可取消生成的方法，以避免必须重启程序。
  
- **非 AVX2 Beta 版本落后**：`@mike_50363` 指出非 AVX2 Beta 版本落后了两个版本，影响了他们在几台配备 128GB RAM 的 Sandy Bridge 系统上使用该软件。`@yagilb` 承认了这个问题，并承诺在发布新的 AVX 构建版本时会提醒他们。

- **LM Studio MacOS 上的持续 Bug**：`@laurentcrivello` 报告了 MacOS 上跨越 3-4 个版本的重复 Bug，即在应用窗口关闭后服务器仍然可以访问。`@yagilb` 确认了该问题，并澄清了点击红叉时的预期行为。

- **优化服务器指示的 UI**：`@laurentcrivello` 解释了他们偏好在 MacOS 上减少活跃应用的指示，希望服务器在没有多个应用图标的情况下运行。他们提议在顶部栏设置一个根据服务器活动而变化的图标，`@wolfspyre` 询问了具体的 UI 预期。

- **对创建快捷方式的抱怨**：`@jiha` 询问是否有选项可以阻止 Beta 版本在每次安装时创建桌面快捷方式，暗示这是一种不受欢迎的行为。
  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1204378574539333632) (1 messages): 

- **关于 Chain 兼容模型的咨询**：用户 `@eugenekormin` 寻求帮助，以识别支持 **chain 和 invoke 方法** 的 **小型 seq2seq LLM 模型**（约几十亿参数），用于使用 langchain 的 Python 脚本。他们请求协助或指引以获取支持这些方法的模型列表。
  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1204505828154023966) (1 messages): 

- **关于 Crew-AI UI 的咨询**：用户 `@docorange88` 询问是否有类似于 AutoGen Studio 的 CrewAI **UI 界面** 或 **Web 界面**。他们表示 Crew-AI 似乎更好，并正在征求意见。
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/) (1 messages): 

phoenix2574: <@294336444393324545> 我正在使用 Mixtral，它看起来运行得还可以。
  

---

### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1204397687668215838) (8 messages🔥): 

- **Mistral 在编程方面胜过 Phi2**：用户 `@carsonpoole` 观察到，在 **OpenHermes 2.5** 的代码部分对 **phi2 和 Mistral** 进行微调时，两者存在显著差异；在相同的采样设置下，Mistral 的表现明显优于 phi2。
- **关于 GPT-4 编程能力的辩论**：鉴于 `@carsonpoole` 的发现，用户 `@teknium` 对 GPT-4 的编程性能调侃了一番，引发了相关讨论。
- **GPT-4 的规模讨论**：针对编程性能，`@n8programs` 指出所讨论的模型仅有 20 亿参数，暗示了其规模带来的局限性。
- **GPT-4 技能水平的预期与现实**：`@teknium` 引用了 Microsoft Research 的一项声明进行反驳，该声明暗示他们已达到 GPT-4 级别的技能，从而设定了对该模型性能的预期。
- **表达难以置信**：用户 `@Error.PDF` 分享了一个来自 Tenor 的幽默[震惊猫咪 gif](https://tenor.com/view/shocked-shocked-cat-silly-cat-cat-kitten-gif-7414586676150300212)，可能是对所讨论的性能结果做出的反应。

**提到的链接**：

[Shocked Shocked Cat GIF - Shocked Shocked cat Silly cat - Discover &amp; Share GIFs](https://tenor.com/view/shocked-shocked-cat-silly-cat-cat-kitten-gif-7414586676150300212)：点击查看 GIF

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1204334563212398632) (16 messages🔥): 

- **Sparsetral MoE 发布**：`@dreamgen` 介绍了 **Sparsetral**，这是一个源自稠密模型 Mistral 的稀疏 MoE 模型，并提供了[原始论文](https://arxiv.org/abs/2401.02731)、[原始仓库](https://github.com/wuhy68/Parameter-Efficient-MoE)以及 [Sparsetral 集成仓库](https://github.com/serp-ai/Parameter-Efficient-MoE)等资源。他们还强调了 fork [unsloth](https://github.com/serp-ai/unsloth) 以进行高效训练，并指出 **Sparsetral on vLLM** 可以在 4090 等硬件上以 bf16 精度运行，同时在 [Hugging Face](https://huggingface.co/serpdotai/sparsetral-16x7B-v2) 上分享了该模型。
  
- **DeepSeek 创下数学领域新 SOTA**：`.benxh` 对 **Deepseek** 表达了热情，该工具通过引入一种称为 DPO 的技术和一种构建数据集的新方法，显然在数学相关基准测试中创下了新的 state-of-the-art。

- **PanGu-$\pi$-1 Tiny Language Model 研究**：`@bozoid` 分享了一篇研究论文（[链接](https://arxiv.org/abs/2402.02791)），重点关注优化具有 1B 参数的 PanGu-$\pi$-1 等微型语言模型，研究了架构、初始化和优化策略，以提高微型 LLM 的性能。

- **针对 LLM 的 EQ-Bench**：`@nonameusr` 介绍了 [EQ-Bench](https://eqbench.com/)，这是一个针对 Large Language Models 的情商基准测试，包括 [GitHub 仓库](https://github.com/EQ-bench/EQ-Bench)和[相关论文](https://arxiv.org/abs/2312.06281)的链接，并指出了其评分系统的更新。

- **Audio Flamingo 在音频理解方面表现出色**：`@2bit3thn` 分享了 **Audio Flamingo** 的详细信息，这是一个在各种音频理解基准测试中表现出色的音频语言模型，提到该模型通过 in-context learning 能很好地适应未见过的任务，并具有强大的多轮对话能力（[项目链接](https://audioflamingo.github.io/)）。

**提到的链接**：

- [LLM check](https://rahulschand.github.io/gpu_poor/)：未找到描述
- [EQ-Bench Leaderboard](https://eqbench.com/)：未找到描述
- [Audio Flamingo](https://audioflamingo.github.io/)：未找到描述
- [Rethinking Optimization and Architecture for Tiny Language Models](https://arxiv.org/abs/2402.02791)：Large Language Models (LLM) 的力量已通过大量数据和计算资源得到证明。然而，语言模型在移动设备上的应用正面临巨大挑战...
- [GitHub - babycommando/machinascript-for-robots: Build LLM-powered robots in your garage with MachinaScript For Robots!](https://github.com/babycommando/machinascript-for-robots)：在你的车库里用 MachinaScript For Robots 构建由 LLM 驱动的机器人！ - GitHub - babycommando/machinascript-for-robots: Build LLM-powered robots in your garage with MachinaScript For Robots!
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ajwijf/model_release_sparsetral/)：未找到描述

  

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1204344002741149716) (198 条消息🔥🔥): 

- **对 Miqu Finetune 进行量化**：`@tsunemoto` 分享了他们对 Senku-70B-Full 的量化版本，这是一个据称是早期 Mistral-70B medium 模型的 Finetune 版本，其 EQ-Bench 评分高达 84.89。针对这是否是基于 FLAN 的 Finetune 的疑问，作者回答称其为 OpenOrca Finetune，你可以在 [HuggingFace](https://huggingface.co/ShinojiResearch/Senku-70B-Full) 上找到它。

- **探索 LLM 中的数学性能**：`@gabriel_syme` 发起了一场关于数学在评估语言模型中重要性的讨论。有人建议，良好的数学表现可能与（非特定任务的）LLM 中扎实的逻辑和推理能力相关，且模型应该在数学和编程语料库上进行训练。

- **旧金山即将举行的活动热潮**：`@teknium` 和其他用户讨论了在旧金山举行的 Ollama AI 开发者活动，`@coffeebean6887` 提供了 [Starter to SF Guide](https://www.startertosf.guide/) 和 [Cerebral Valley](https://cerebralvalley.ai/) 等链接，以获取该地区的额外资源和活动信息。尽管容量很大，但名额已接近爆满，建议迅速采取 RSVP 行动。

- **Mixtral 的混合语言难题**：`@light4bear` 表示，当用中文指示 Mixtral 时，它会以中英混合的方式回答，而相反地，OpenHermes 偶尔会显示中文回答。后者由 `@teknium` 提到，已被添加到 Cloudflare 的 AI 平台，Cloudflare 和 Teknium 的[官方推文](https://x.com/teknium1/status/1755020133398155269?s=46)证明了这一点。

- **多模态模型 Finetuning 的问题与支持**：`@babycommando` 介绍了 MachinaScript For Robots，这是一个旨在利用 LLM 控制机器人的项目，并在 `[channel]` 类别中寻求关于 Finetuning 多模态模型的建议。他们还对 Nous 的工作及其对该领域的贡献表示了感谢。

**提到的链接**：

- [Cerebral Valley](https://cerebralvalley.ai/)：一个由创始人及构建者组成的社区，致力于创造下一代技术。
- [LiPO: Listwise Preference Optimization through Learning-to-Rank](https://arxiv.org/abs/2402.01878)：将语言模型 (LM) 与精选的人类反馈对齐，对于在实际应用中控制其行为至关重要。最近的几种策略优化方法，如 DPO 和 SLiC，旨在……
- [tsunemoto/Senku-70B-Full-GGUF · Hugging Face](https://huggingface.co/tsunemoto/Senku-70B-Full-GGUF)：未找到描述。
- [ShinojiResearch/Senku-70B-Full · Hugging Face](https://huggingface.co/ShinojiResearch/Senku-70B-Full)：未找到描述。
- [Teknium (e/λ) (@Teknium1) 的推文](https://x.com/teknium1/status/1755020133398155269?s=46)：Cloudflare 已将我的 OpenHermes 2.5 7b 添加到他们的 Workers AI 平台！↘️ 引用 Cloudflare (@Cloudflare) 的话：在过去的几个月里，Workers AI 团队一直致力于改进我们的 AI 平台……
- [RSVP to Chat (Ro)bots Hackathon @ AGI House | Partiful](https://partiful.com/e/d2fCE2WW4MGeUr8pEZLV)：欢迎来到 Robotics x LLMs Hack，这里是创意、协作和尖端技术的交汇点。无论你是资深编码员还是问题解决大师，这都是你参与构建的绝佳机会……
- [Alice (e/nya) (@Alice_comfy) 的推文](https://x.com/Alice_comfy/status/1754965801147490418?s=20)：好吧，这只是一个基准测试，但 Senku-70B（泄露的 Mistral Finetune）在 EQ Bench 中击败了 GPT-4。不确定如何将其添加到网站上。Senku-70B 可以在这里获取。https://huggin...
- [Local & open-source AI developer meetup · Luma](https://lu.ma/devs2)：Ollamas 及其朋友们又回来了，举办另一场以开发者为中心的见面会！我们将前往旧金山渡轮大厦的 Cerebral Valley！开源 AI 演示日，提供免费餐饮及……
- [Cloudflare (@Cloudflare) 的推文](https://x.com/cloudflare/status/1754958644326604930?s=46)：在过去的几个月里，Workers AI 团队一直致力于改进我们的 AI 平台。在添加了 Code Llama、Stable Diffusion、Mistral 等模型后，今天，我们很高兴地宣布……
- [Starter Guide to SF for Founders](https://www.startertosf.guide/)：为任何新到或考虑搬到旧金山的人提供的入门资源。

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1204507713783078942) (11 条消息🔥): 

- **寻求数据生成方面的研究**: `@bigdatamike` 正在探索用于微调内部 LLM 的问答对生成，并讨论了利用一篇 Microsoft 论文中的策略。他们征求了关于高质量数据生成的其他论文推荐。
  
- **挖掘模型架构的限制**: `@lunarsylph_67003` 询问了 `LlamaForCausalLM` 支持的模型限制，质疑是否允许使用新颖的架构。`@teknium` 澄清说 Bittensor 仅允许使用 **Mistral**。

- **进一步微调的 Loss 值**: `@gabriel_syme` 提出了一个关于在微调已经微调过的模型时典型的 Loss 值问题，以及 Loss 是会逐渐减小还是保持在初始微调值附近。

- **发布新框架**: `@babycommando` 介绍了 **MachinaScript for Robots**，这是一个允许构建由 LLM 驱动的机器人的框架和语言。该框架以类 JSON 语法处理 LLM 输出，然后由机器人解析并执行；代码仓库可以在 [MachinaScript for Robots on GitHub](https://github.com/babycommando/machinascript-for-robots) 找到。

- **如何微调 Obsidian**: `@babycommando` 寻求关于微调 **Obsidian** 的指导，包括流程、步骤、工具（如 Lora 或 Qlora）和系统规格。此外，他们还请求了关于数据集格式的信息，以及是否有人能说明其基于 MachinaScript 的机器人交互项目的微调过程。

**提到的链接**:

- [LiPO: Listwise Preference Optimization through Learning-to-Rank](https://arxiv.org/html/2402.01878v1): 未找到描述
- [GitHub - babycommando/machinascript-for-robots: Build LLM-powered robots in your garage with MachinaScript For Robots!](https://github.com/babycommando/machinascript-for-robots): 在你的车库里用 MachinaScript For Robots 构建 LLM 驱动的机器人！ - GitHub - babycommando/machinascript-for-robots: Build LLM-powered robots in your garage with MachinaScript For Robots!

---

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1204334580543520799) (159 messages🔥🔥): 

- **VRAM 之争**：对话围绕运行大模型的硬件需求展开，`@ethux` 建议至少需要 **100GB VRAM**，而 `@dawn.dusk` 和 `@i_am_dom` 等人则讨论了 **NVIDIA 4090** 甚至 **双 4080**（累计 32GB VRAM）是否足够。语境暗示这是为了运行全量 fp16 模型，很可能是 AI 或 ML 应用。
  
- **Intel Mac 与 Apple Silicon 之争**：关于 **基于 Intel 的 Mac** 与较新的 **Apple Silicon Mac** 相关性的激烈辩论，`@frosty04212` 对 Intel Mac 的过时表达了强烈观点并主张升级。其他用户如 `@firesonwires` 则认为笔记本电脑具有实际的使用寿命，反对不必要的升级。

- **硬件升级理念**：一场关于何时升级技术的争议性讨论，由 `@mrdragonfox` 和 `@firesonwires` 的评论引发，强调了对最新硬件需求的不同看法。虽然 `@frosty04212` 认为 **Apple Silicon** 是一次重大且终极的升级，但其他人强调了技术的持续演进和个人财务状况的考量。

- **AI 模型访问与性能**：`@ethux` 和 `@mrdragonfox` 等用户分享了使用 AI 模型的资源和见解，包括 **Mistral 模型** 和 **Mistral 指南**。对话涉及基于 MoE 的模型、API 访问、**Google 的 LocalLLM** 以及在 CPU 上运行的量化模型等话题。

- **加速卡与技术演进**：`@mrdragonfox` 讨论了高端硬件，如 **Groq 加速器** 和 **NVIDIA A100 的停产公告 (EOL)**，反映了硬件生命周期的快速更迭以及尖端算力相关的成本。一些人幽默地思考了这些加速器未来在 eBay 等平台上的价值。

**提到的链接**：

- [Mistral AI | Open-weight models](https://mistral.ai/)：触手可及的前沿 AI
- [Transformer Inference Arithmetic | kipply's blog](https://kipp.ly/transformer-inference-arithmetic/)：kipply 关于她所做、所读或所观察事物的博客
- [Prompting Capabilities | Mistral AI Large Language Models](https://docs.mistral.ai/guides/prompting-capabilities/)：当你开始使用 Mistral 模型时，你的第一次交互将围绕提示词展开。编写有效提示词的艺术对于从 Mistral 模型中获得理想响应至关重要……
- [New localllm lets you develop gen AI apps locally, without GPUs | Google Cloud Blog](https://cloud.google.com/blog/products/application-development/new-localllm-lets-you-develop-gen-ai-apps-locally-without-gpus?utm_source=twitter&utm_medium=unpaidsoc&utm_campaign=fy24q1-googlecloudtech-blog-ai-in_feed-no-brand-global&utm_content=-&utm_term=-&linkId=9418398&s=09)：想在本地开发环境中使用 Hugging Face 的开源 LLM 模型吗？通过 localllm 和 Cloud Workstations，你可以实现这一目标。
- [Exploring the Latency/Throughput & Cost Space for LLM Inference // Timothée Lacroix // LLM 3 Talk 3](https://www.youtube.com/watch?v=mYRqvB1_gRk)：// 摘要：获得正确的 LLM 推理栈意味着为你的任务选择正确的模型，并在正确的硬件上运行它，配合适当的推理协作……
- [Chat with Open Large Language Models](https://chat.lmsys.org)：未找到描述
- [HuggingChat](https://huggingface.co/chat)：让每个人都能使用社区最好的 AI 聊天模型。
- [Google Cloud Blog](https://cloud.google.com/blog/products/application-development/new-localll)：未找到描述

  

---

### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1204404525725646859) (47 messages🔥): 

- **Mistral 推理成本计算查询**：`@kushagra_67246` 对 **Mistral-7b 模型** 的每 Token 成本计算表示困惑，并指出他们的计算结果显示成本远高于 ChatGPT 3.5，即使使用了量化（quantization）来减少内存占用（memory footprint）。他们引用了 [DeepInfra](https://deepinfra.com/pricing) 的定价作为基准，寻求降低自身成本的方法。
- **社区关于降低成本的建议**：社区成员向 `@kushagra_67246` 提出了多种省钱策略，包括利用 **Runpod** 等 Serverless 平台，考虑 **Groq** 的加速器，以及探索使用 **LlamaCPP** 来实现 Mixtral-8x7B。专用硬件和提示词工程（prompt engineering）也被提及为影响成本和性能的因素。
- **对数据敏感性和微调的担忧**：`@kushagra_67246` 强调由于输入数据的敏感性以及在自定义数据集上微调（fine-tune）模型的需求，需要保持对模型托管的控制。关于隐私成本、内部推理运行与专业托管服务的对比，在运营需求和数据保护方面展开了讨论。
- **模型加载和启动时间的经验分享**：`@casper_ai` 和 `.superintendent` 分享了个人经验，指出模型启动时间并不算高，有时模型在 **2-10 秒** 内即可准备好进行推理。`.superintendent` 提到了近期启动时间的改进。
- **针对 LlamaCPP 和 Mistral 模型的提示词工程**：`@aiman1993` 向社区询问在使用 **LlamaCPP** 运行 **Mistral-8x7B** 时提示词工程技术的有效性，因为他们观察到尝试的结果并不标准。对话表明，虽然技术通常通用，但特定的配置和更新（如用于 autoAWQ 的 **Llama 1.6**）可能需要进行适配。

**提到的链接**：

[Mistral 7B 比 GPT-4 便宜 187 倍](https://www.linkedin.com/pulse/mistral-7b-187x-cheaper-compared-gpt-4-tzejf)：Mistral AI 7B 模型可以作为 GPT 3.5 或 4 模型的绝佳替代方案，成本便宜 187 倍。内含计算过程以了解模型间的成本对比。

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1204344730541101066) (16 messages🔥): 

- **微调中的填充 (Padding) 难题**：`@ramin2024` 对使用填充 Token 生成文本时的不一致行为表示困惑，即使在参考了 [Fine-tuning Language Models for Structured Responses with QLoRa](https://youtu.be/OQdp-OeG1as) 等教程之后也是如此。他们质疑是否需要在分词器（tokenizer）和模型配置中定义多个填充 Token。
  
- **寻求微调实践的指导**：`@ramin2024` 分享了一个他们参考的关于微调语言模型的 YouTube [教程](https://youtu.be/OQdp-OeG1as)，并指出尽管该教程侧重于 Llama 2，而他们在 Mistral 上使用的是 LlamaTokenizer，但仍具有参考价值。

- **微调平台推荐**：`@js06` 在遇到 Huggingface 的 Autotrain 错误后寻求平台建议；`@mrdragonfox` 建议使用本地训练或租用 GPU，并考虑使用 [togetherai](https://togetherai.com/) 等服务。

- **Mistral 训练问题**：`@xzuyn` 寻求关于在不使用 SWA 的情况下将 Mistral v0.1 7B 微调至超过 8k Token 的指导，并询问了这种方法的性能表现，提到他们知道该模型预训练时使用的是 8k。

- **模型预训练的澄清**：`@xzuyn` 提出了关于在 Axolotl 中使用 `s2_attention` 和样本打包（sample packing）的疑问，以及这与 Mistral v0.1 7B 预训练方法论的关系。

**提到的链接**：

- [Discord - 与好友和社区聊天的新方式](https://discord.com/channels/1144547040454508606/1144547040928481394/1157033013335576796)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的好友和社区保持紧密联系。
- [使用 QLoRa 微调语言模型以获得结构化响应](https://youtu.be/OQdp-OeG1as?si=ZtD9ld9qqF4xaSAT)：我介绍了如何微调语言模型以返回*结构化响应*，例如返回函数调用、JSON 对象或数组。讲义地址：https://c...

  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1204681413299929138) (2 messages): 

- **等待开源发布**：`@stdewinter` 对使用某个工具表示兴趣，询问如何使用。`@hugoduprez` 回应称打算在有时间后将其**开源发布**。

---

### Mistral ▷ #[office-hour](https://discord.com/channels/1144547040454508606/1192781286008422441/1204405187343552594) (1 条消息): 

- **下一次 Office Hour 已排期**：`@sophiamyang` 宣布了下一次 Office Hour 的安排，可以通过 [此 Discord 链接](https://discord.gg/mistralai?event=1204405056825327677) 访问。

**提到的链接**：

[Discord - 与好友及社区聊天的新方式](https://discord.gg/mistralai?event=1204405056825327677)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、闲逛，与您的朋友和社区保持紧密联系。

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1204401703860830239) (114 条消息🔥🔥): 

- **Astro 的 Discord 账号被盗**：`@astropulse` 确认他们在一次针对其 Discord 账号的定向鱼叉式网络钓鱼攻击中被黑。在进行了彻底的清理和恢复操作后，账号目前已安全。`@vrus0188` 和 `@nodja` 等其他用户给出了预防性建议，如重置恢复代码，并使用 [Have I Been Pwned](https://haveibeenpwned.com/) 检查邮箱是否出现在已泄露网站的名单中。

- **AI 架构开发的磨难与尝试**：`@mkaic` 分享了他们在开发新型 AI 架构时的挫折与希望，表示他们目前的模型在 CIFAR100 上表现不佳，但仍保持乐观，将问题归因于梯度流（gradient flow）不畅。
  
- **诡异的黑客故事**：幽默的是，`@pseudoterminalx` 开玩笑说 `@astropulse` 被绑架并被当作算力的“电池”，而 `@progamergov` 则讲述了自己账号被僵尸网络（botnet）攻击的经历。

- **建设中的神经形态网络**：`@mkaic` 讨论了他们创建真正稀疏神经网络的方法，即允许神经元在训练期间改变连接点，旨在为当前 AI 模型中的稀疏性（sparsity）概念提供一种替代方案。

- **被低估的项目**：`@SegmentationFault` 重点介绍了 [GitHub 上的 PolyMind](https://github.com/itsme2417/PolyMind)，这是一个将多种高级 AI 功能整合到一个界面的项目，并感叹这类实用性项目往往被那些更具娱乐性的 AI 应用所掩盖。

**提到的链接**：

- [Discord - 与好友及社区聊天的新方式](https://discord.com/channels/823813159592001537/823813160075132991/1204474029080182785)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、闲逛，与您的朋友和社区保持紧密联系。
- [Have I Been Pwned：检查您的电子邮件是否在数据泄露中受损](https://haveibeenpwned.com/)：Have I Been Pwned 允许您在多个数据泄露事件中进行搜索，以查看您的电子邮件地址或电话号码是否已泄露。
- [Good Boy Dance GIF - Good Boy Dance - 发现并分享 GIF](https://tenor.com/view/good-boy-dance-gif-25381375)：点击查看 GIF
- [GitHub - itsme2417/PolyMind: 一个多模态、由 function calling 驱动的 LLM webui。](https://github.com/itsme2417/PolyMind)：一个多模态、由 function calling 驱动的 LLM webui。 - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.
- [Matrix Morpheus GIF - Matrix Morpheus Battery - 发现并分享 GIF](https://tenor.com/bg2bf.gif)：点击查看 GIF
- [Astropulse](https://astropulse.co/#retrodiffusionhack)：Retro Diffusion 开发者 Astropulse 的主页。

  

---

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1204448100803158017) (9 messages🔥): 

- **一致性图像生成的新方法**：`@thejonasbrothers` 介绍了 [ConsiStory](https://arxiv.org/abs/2402.03286)，这是一种无需训练的新颖方法，旨在解决 text-to-image 生成中的一致性挑战。`ConsiStory` 模型利用了主体驱动的共享注意力块（subject-driven shared attention blocks）和基于对应关系的特征注入（correspondence-based feature injection）。

- **提升预测与承诺能力**：`@chad_in_the_house` 提到了一篇引发关注的论文 [论文链接](https://arxiv.org/abs/2401.14953)，DeepMind 的研究人员在其中考察了 AI 的预测（anticipation）和承诺（commitments）等认知能力。

- **晚读预告**：用户 `@twoabove` 对分享的论文做出了回应，表示由于这些论文的复杂性和深度，分析它们需要专门抽出一个晚上的时间。

- **Lumiere，点亮视频生成**：`@spirit_from_germany` 分享了一个 [YouTube 视频](https://youtu.be/Pl8BET_K1mc)，讨论了 *Lumiere*，这是 Google Research 开发的 text-to-video 生成模型，解决了视频中全局一致性的挑战。

- **推特热门 AI 话题**：`@helium__` 提供了一个与 AI 相关的 [danlyth 的推文链接](https://twitter.com/danlyth/status/1754823375208280430)，但未提供关于内容的额外上下文。

**提到的链接**：

- [LiPO: Listwise Preference Optimization through Learning-to-Rank](https://arxiv.org/abs/2402.01878)：使语言模型（LMs）与精选的人类反馈保持一致，对于在现实应用中控制其行为至关重要。最近的一些策略优化方法，如 DPO 和 SLiC，被用作……
- [Training-Free Consistent Text-to-Image Generation](https://arxiv.org/abs/2402.03286)：Text-to-image 模型通过允许用户使用自然语言引导图像生成过程，提供了全新的创意灵活性。然而，使用这些模型来一致地描绘……
- [Learning Universal Predictors](https://arxiv.org/abs/2401.14953)：Meta-learning 已成为一种强大的方法，可以训练神经网络从有限的数据中快速学习新任务。广泛接触不同的任务可以产生通用的表示，从而实现……
- [Lumiere: A Space-Time Diffusion Model for Video Generation (Paper Explained)](https://youtu.be/Pl8BET_K1mc)：#lumiere #texttovideoai #google Google Research 的 LUMIERE 通过将 U-Net 下采样概念扩展到……来解决全局一致的 text-to-video 生成问题。

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1204373557392703488) (42 messages🔥): 

- **极简互动的早安问候**：Pagangpegus 在聊天中简单地发了一句 "morning yall"。

- **关于 GPT-3.5 与 GPT-4 代码生成的辩论**：`@catboy_slim_` 讨论了在为冷门语言生成代码时，GPT-3.5 似乎比 GPT-4 更好地遵守复杂的指令，而 GPT-4 会更快地退回到类似教程的回答。他们指出，GPT-4 在全局范围内可能看起来更聪明，但在处理复杂的边缘情况 Prompt 时可能会表现出不一致性。

- **垃圾机器人入侵服务器**：成员 `@.undeleted` 和 `@random_string_of_character` 交流了服务器中垃圾机器人持续存在的问题，指出服务器的开放性和网站上可见的邀请链接可能是诱因。`@stellaathena` 指出，服务器更看重可访问性而非限制性的审核，因此接受偶尔出现的机器人。

- **MetaVoice TTS 模型发布并采用开源许可证**：`@stellaathena` 分享了[一条推文](https://x.com/reach_vb/status/1754984949654904988?s=46)，庆祝 *MetaVoice 1B* 的发布，这是一个新的 text-to-speech 模型。包括 `@random_string_of_character` 在内的用户讨论了它的表现，提到了一个 *open demo* 以及该模型克隆声音和生成情感演讲的能力，但效果参差不齐。

- **通过大量使用来判断模型能力**：在一次关于理解模型能力的讨论中，`@rallio.` 强调，真正了解一个模型的有效性需要大量的使用，这在标准测试（如针对 LLM 的 SAT/GRE）出现之前可以作为一种替代方案。`@alexanderrgriffing` 暗示未来会有一个名为 *gollarkbench™* 的解决方案。

**提到的链接**：

- [TTS by MetaVoice](https://ttsdemo.themetavoice.xyz/)：未找到描述
- [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1754984949654904988?s=46)：出发！MetaVoice 1B 🔉 > 12 亿参数模型。> 在 10 万小时数据上训练。> 支持 zero-shot 语音克隆。> 短文本和长文本合成。> 情感演讲。> 最佳……

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1204401102074810428) (71 messages🔥🔥):

- **探索外推研究的困境**：`@paganpegasus` 寻求关于使用 150M 模型进行长度外推（length extrapolation）/泛化研究的基准测试建议。其他用户如 `@ai_waifu` 和 `@nverix` 建议查看 loss 与序列长度的关系图、induction，并使用给定 completions 的 log likelihood 作为性能衡量指标。

- **SELF-DISCOVER 框架表现优于传统方法**：讨论了一个名为 SELF-DISCOVER 的新框架（见[这篇论文](https://arxiv.org/abs/2402.03620)），与现有技术相比，该框架显著提高了 GPT-4 和 PaLM 2 在推理基准测试中的表现，并将推理计算量减少了高达 40 倍。

- **通过基于意图的校准进行 Prompt 优化**：`@elad7318` 分享了他们关于大语言模型（LLM）自动 Prompt 工程的最新论文，其中包括生成具有挑战性的边界案例以迭代优化 Prompt。他们提到了开源系统 AutoPrompt，可在 [GitHub](https://github.com/Eladlev/AutoPrompt) 上获取。

- **结合 Hurst 指数与 BPB 以提升下游性能**：`@random_string_of_character` 分享了 @ibomohsin 的一条 Twitter 帖子，讨论了将 Hurst 指数与 bits per byte (BPB) 相结合，以更好地预测语言模型的下游性能，这一想法在 [arXiv 论文](https://arxiv.org/abs/2402.01825)中得到了进一步解释。

- **围绕 Moirai 论文的争议与澄清**：围绕一篇关于通用时间序列预测模型的论文（可能来自 Google）展开了讨论。Twitter 上出现了关于该论文主张中可能存在误导和疏忽错误的指控。

**提到的链接**：

- [Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620): 我们介绍了 SELF-DISCOVER，这是一个通用的框架，让 LLM 自我发现任务内在的推理结构，以解决对于典型 Prompting 方法具有挑战性的复杂推理问题...
- [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592#:~:text=Deep%20learning%20for%20time%20series,of%20large%20pre%2Dtrained%20models.): 时间序列预测的深度学习传统上在“一个数据集一个模型”的框架内运行，限制了其利用大型预训练模型（Large Pre-trained Models）带来的变革性影响的潜力。...
- [Hungry Hungry Hippos: Towards Language Modeling with State Space Models](http://arxiv.org/abs/2212.14052): 状态空间模型（SSM）在某些模态中展示了最先进的序列建模性能，但在语言建模方面表现不如 Attention。此外，尽管其扩展性接近线性...
- [Read to Play (R2-Play): Decision Transformer with Multimodal Game Instruction](https://arxiv.org/abs/2402.04154): 开发通用型 Agent 是人工智能领域的一个长期目标。之前利用来自各种任务的大量离线数据集的努力，在多任务中展示了卓越的性能...
- [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592#:~:text=Deep%20learning%20for%20time%20series,of%20large%20pre%2Dtr): 时间序列预测的深度学习传统上在“一个数据集一个模型”的框架内运行，限制了其利用大型预训练模型带来的变革性影响的潜力。...
- [A decoder-only foundation model for time-series forecasting &#8211; Google Research Blog](https://blog.research.google/2024/02/a-decoder-only-foundation-model-for.html): 未找到描述
- [Tweet from Valeriy M., PhD, MBA, CQF (@predict_addict)](https://x.com/predict_addict/status/1754134502895460421): 来自 Google 的一篇推销时间序列预测“基础模型”的新论文，既是初学者错误的典型，又采用了具有欺骗性的“基准测试（Benchmarks）”。在图 6 中，作者...
- [Scaling Laws for Downstream Task Performance of Large Language Models](https://arxiv.org/abs/2402.04177): Scaling Laws 提供了可以指导大型语言模型（LLM）设计的重要见解。现有的工作主要集中在研究预训练（上游）损失的 Scaling Laws。然而...
- [Tweet from Ibrahim Alabdulmohsin | إبراهيم العبدالمحسن (@ibomohsin)](https://x.com/ibomohsin/status/1754912619985604818): 那么，我们可以将 H 与 BPB 结合来预测下游性能吗？是的：取 H + 1/BPB 的平均值（我们将 BPB 取倒数，以便数值越高越好）。这个简单的平均值能更好地预测下游性能...
- [Tweet from Ibrahim Alabdulmohsin | إبراهيم العبدالمحسن (@ibomohsin)](https://x.com/ibomohsin/status/1754912601165775296): Next-token Prediction 是如何实现这种智能行为的？我非常激动地分享我们的工作，我们在其中研究了语言的分形结构。简而言之（TLDR）：将语言中的 Next-token Prediction 视为...
- [Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases](https://arxiv.org/abs/2402.03099): 由于大型语言模型（LLM）对给定 Prompt 的高度敏感性以及文本任务指令固有的模糊性，Prompt Engineering 是一项具有挑战性且重要的任务。自动...
- [GitHub - Eladlev/AutoPrompt: A framework for prompt tuning using Intent-based Prompt Calibration](https://github.com/Eladlev/AutoPrompt): 一个使用基于意图的 Prompt Calibration 进行 Prompt Tuning 的框架 - GitHub - Eladlev/AutoPrompt: 一个使用基于意图的 Prompt Calibration 进行 Prompt Tuning 的框架
- [Hurst exponent - Wikipedia](https://en.wikipedia.org/wiki/Hurst_exponent): 未找到描述
- [Induction heads - illustrated — LessWrong](https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated): 非常感谢所有提供有用反馈的人，特别是 Aryan Bhatt 和 Lawrence Chan！...
- [Tweet from Dimitris Papailiopoulos (@DimitrisPapail)](https://x.com/DimitrisPapail/status/1754962834113356231): 今晚发布 arXiv 论文 "Can Mamba Learn How to Learn?: A Comparative Study on In-Context Learning Tasks"，合作者阵容豪华，来自 @Krafton_inc @SeoulNatlUni @UMich 和 @UWMadison
- [Tweet from Dimitris Papailiopoulos (@DimitrisPapail)](https://x.com/DimitrisPapail/status/1754965567004389427): 我先把它放在这里

---

### Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1204511912994410638) (3 messages): 

- **讨论无限深度极限 (Infinite Depth Limit)**：`@niket` 提到了 Greg Yang 和 Soufiane Hayou 等研究人员关于 **infinite depth limit** 工作相关性。
- **扩展 Tensor Programs**：`@niket` 在深度学习和 **scaling laws** 的背景下引用了 **tensor programs**，可能是第六次迭代。
- **探索 Scaling Laws 中的 Loss Landscapes**：`@niket` 表示有兴趣了解 **loss landscape** 在这些无限极限下的表现，并指出缺乏相关资源，请求提供该领域现有研究的参考文献。
  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1204490850713866250) (2 messages): 

- **发现 LLMs 的 Fluent Dreaming**：`@tbenthompson` 分享了一篇名为 “Fluent dreaming for language models” 的 [新研究论文](https://arxiv.org/abs/2402.01702)，该论文提出了一种名为 Evolutionary Prompt Optimization (EPO) 的方法，用于优化 prompts 以在保持低 **perplexity** 的同时最大化激活。这种方法被比作视觉模型中的特征可视化（feature visualization），可能是 **LLM** 可解释性工具包的一个重要补充。

- **寻求 LLM Prompts 的 Saliency 解决方案**：`@bingbingan.` 正在寻找确定 **LLM** prompts 中输入 **saliency** 的方法，特别是为了理解哪些 tokens 会影响模型拒绝回答敏感查询（如“如何制造炸弹”）。他们考虑了 Integrated Gradients，但分享了一篇 [令人担忧的论文](https://arxiv.org/abs/2212.11870)，指出此类特征归因（feature attribution）方法在推断模型行为方面可能并不可靠。

**提到的链接**：

- [Fluent dreaming for language models](https://arxiv.org/abs/2402.01702)：特征可视化，也称为 "dreaming"，通过优化输入以最大化神经元激活或其他内部组件，提供了对视觉模型的见解。然而，dreamin...
- [Impossibility Theorems for Feature Attribution](https://arxiv.org/abs/2212.11870)：尽管有大量的可解释性方法可以产生看似合理的解释，但该领域在经验上也看到了许多此类方法的失败案例。鉴于这些结果，目前尚不清楚...

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1204354051857326150) (3 messages): 

- **确认会前准备**：`@asuglia` 对 `@1072629185346019358` 表示感谢，并提到计划在即将召开的会议之前审查某些材料。
- **提议协助 VLM 集成**：`@hailey_schoelkopf` 建议在 ACL 截止日期后重新召集讨论，并提议协助将 **VLMs** 与基于 **loglikelihood** 的请求类型进行集成，邀请 `@1072629185346019358` 在需要时联系。
- **简短的告别**：`@jbdel.` 保持简短，简单感谢了 `@hailey_schoelkopf`。
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1204348713649184789) (21 messages🔥): 

- **GCP A100 短缺**：`@dangfutures` 表示在 Google Cloud Platform (GCP) 上很难找到 **A100 GPUs**，表明可能存在短缺或可用性有限。
- **LLM Autoeval 基准测试时长查询**：`@c.gato` 询问了 7b 模型在 **LLM Autoeval** 上的基准测试时间，并提到在 4090 GPU 上进行测试需要等待 4 小时。
- **了解 LLM Autoeval**：`@teknium` 提供信息称，在 4090 GPU 上使用 7b 模型进行 MMLU 测试，复制 HF Leaderboard 基准测试大约需要 12 小时，并澄清他们使用的是 **lm-eval-harness** 而不是 LLM Autoeval。
- **基准测试时间线澄清**：在与 `@teknium` 讨论后，`@c.gato` 意识到他们的基准测试可能比预期的要长，因为 **lm-eval-harness** 在单个 4090 上完成一组基准测试大约需要 1-2 小时。
- **报告技术问题**：`@youraveragedev` 分享了一个错误消息，指出 `ServerApp` 配置已弃用，以及与 Jupyter server 根内容目录相关的初始化配置错误。

**提到的链接**：

[GitHub - mlabonne/llm-autoeval: Automatically evaluate your LLMs in Google Colab](https://github.com/mlabonne/llm-autoeval)：在 Google Colab 中自动评估你的 LLMs。通过在 GitHub 上创建账户来为 mlabonne/llm-autoeval 的开发做出贡献。

  

---

### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1204393097979760707) (26 messages🔥): 

- **Hugging Face 资助 Axolotl 悬赏**：`@dctanner` 透露了 Hugging Face (HF) 提供的一项提议，即为开发 **Axolotl Spaces 训练 UI** 提供价值 5000 美元的悬赏和算力额度。该 UI 应允许用户将一个 UI Space 克隆到自己的账户中，指定训练运行，并管理带有所需硬件的 Spaces。

- **前后端协作呼吁**：`@dctanner` 表示有兴趣使用 **Tailwind CSS** 开发 Axolotl Spaces 训练 UI 的前端，并寻求后端开发方面的协助，首选 Python，因为其更易于维护和适配。

- **训练 UI 的 Shiny vs Gradio 之争**：`@jameshwade` 建议使用 **Shiny** 构建原型，甚至提供了一个应用概念的 [mockup](https://huggingface.co/spaces/jameshwade/axolotl-ui) 链接。然而，`@le_mess` 反驳认为 **Gradio** 可能更合适，并提到 Shiny 容易变得臃肿和复杂。

- **Shiny 团队提供支持**：在关于 UI 工具的讨论中，来自 **Posit 的 Shiny 团队** 的 `@gordon_shotwell` 表示愿意协助开发。

- **Runpod 模板中的持久卷挂载点问题**：`@m4ttfl0` 建议更改 Runpod 模板中持久卷的挂载点，以避免覆盖现有的 Axolotl 目录，并指向了一个包含更多细节的 [GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1235)。`@nanobitz` 参与了对话，指出该问题最近才出现，并可能对已建立的工作流产生影响。

**提及的链接**：

- [Shiny](https://shiny.posit.co/)：Shiny 是一个可以轻松使用 R 和 Python 创建交互式 Web 应用的包。
- [Axolotl Launcher 🚀 - jameshwade 的 Hugging Face Space](https://huggingface.co/spaces/jameshwade/axolotl-ui)：未找到描述
- [使用 🤗 AutoTrain SpaceRunner 在 NER 数据集上微调 Flair 模型](https://huggingface.co/blog/stefan-it/autotrain-flair-mobie#start-fine-tuning-with-🤗-autotrain-spacerunner)：未找到描述
- [winglian 提交的 Cloud motd · Pull Request #1235 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1235)：描述：有时在 Runpod 中，额外的磁盘被挂载并覆盖了 Axolotl 目录。添加一个 motd 以帮助用户解决此问题。

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1204396049964339230) (17 messages🔥): 

- **多 GPU 多节点保存问题**：`@lykon78` 报告了在多 GPU、多节点配置上保存最终模型 (Mistral FFT) 时出现的错误。他们怀疑分布式保存可能未正确实现，特别是考虑到错误源自非主节点。
  
- **相同配置下无差异**：`@lykon78` 确认两个节点具有相同的配置、文件夹结构和相同版本的 Python 库。运行模型使用的命令包括带有 DeepSpeed 配置的 `torchrun`。

- **分布式训练挑战**：`@nanobitz` 和 `@caseus_` 讨论了多节点设置的潜在问题，承认缺乏对 `axolotl` 的广泛测试，以及跨节点同步 Checkpoint 数据的必要性。

- **研究 Transformer 库的修复方案**：`@lykon78` 调查了 Transformers 库中多节点训练的类似问题，参考了一个可能包含修复方案的 Pull Request，但指出即使在应该已经解决该问题的最新版本 (4.37.2) 中，问题依然存在。

- **多节点保存需要代码调整**：`@caseus_` 建议需要通过修改 `TrainingArguments` 来更改代码，以便在每个节点上启用模型保存，`@lykon78` 表示愿意测试任何提议的代码更改。

**提及的链接**：

- [多节点微调无法从 Checkpoint 恢复 · Issue #884 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/884)：请检查此问题之前是否已被报告。我搜索了之前的 Bug 报告，没有发现类似的报告。预期行为：我有两个节点，我想进行微调...
- [dumpmemory 提交的修复多节点训练设置下 Checkpoint 保存的 Bug · Pull Request #28078 · huggingface/transformers](https://github.com/huggingface/transformers/pull/28078/files)：此 PR 做了什么？修复了 # (issue) 修复了共享文件系统下多节点训练设置的 Bug。在提交之前，此 PR 修复了一个拼写错误或改进了文档（你可以忽略其他检查...）

  

---

### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1204357412631486485) (9 条消息🔥): 

- **DPO 结果故障排除**：`@fred_fups` 质疑 **DPO** 结果不可靠，并在使用 **Qlora** 对基于非 DPO 数据训练的 Mistral 模型进行处理后寻求指导。`@xzuyn` 建议学习率可能过高，并建议显著降低学习率。
- **DPO 与学习率**：`@xzuyn` 指出，与标准微调相比，**DPO** 通常需要低得多的学习率，并建议 `@fred_fups` 在学习率中“再增加一两个 0”以获得更好的效果。
- **DPO 中的 Alpaca 与 ChatML**：`@fred_fups` 询问在 **DPO** 中使用 **Alpaca format** 的可能性，但 `@xzuyn` 表示 **ChatML** 是唯一受支持的格式，尽管自定义格式或许可行。
- **格式切换建议**：在从 `@xzuyn` 处获知 **ChatML** 是 DPO 的标准格式后，`@fred_fups` 决定切换到 **ChatML**，尽管此前在小数据集的单一任务上使用 **Alpaca format** 取得了成功。
- **个人方法与协助**：为了替代 **ChatML**，`@xzuyn` 分享了一种个人方法，包括使用 **Metharme** 并“不断尝试修改直到成功”。
  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1204431673203363840) (7 条消息): 

- **Jupyter 日志困惑**：`@caseus_` 澄清说，显示的 Jupyter 日志不会影响系统内的常规训练。
- **Jupyter Notebook 启动问题**：`@nruaif` 提到 Jupyter notebook 无法启动的问题，`@caseus_` 随后建议，这可能是因为 **axolotl** 位于 workspace 目录中，而该目录受到了挂载卷的影响。
- **弃用的配置警告与初始化错误**：`@youraveragedev` 因弃用的 ServerApp 配置而遇到错误，并收到一条关键消息称 **"/workspace is outside root contents directory"**。
- **针对 Jupyter 问题的潜在修复建议**：针对 `@youraveragedev` 的错误，`@nanobitz` 建议运行来自 GitHub pull request 的代码，该 PR 解决了 runpod 中额外磁盘覆盖 axolotl 目录的问题。该 pull request 可以在此处找到：[Cloud motd by winglian · Pull Request #1235](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1235)。
- **重新克隆仓库作为解决方案**：`@nanobitz` 进一步建议 `@youraveragedev` 重新克隆仓库并再次运行 pip install，作为故障排除过程的一部分。

**提到的链接**：

[Cloud motd by winglian · Pull Request #1235 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1235)：描述：有时在 runpod 中，额外磁盘被挂载并覆盖了 axolotl 目录。添加一个 motd 以帮助用户解决此问题。

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1204392073978183730) (48 条消息🔥): 

- **Claude Pro 订阅咨询**：`@brewingbrews_92072` 询问即使使用量极少是否有必要订阅 **Claude Pro**，并在购买 Pro 版本前考虑了这一点。
- **AI 服务定价讨论**：`@archient` 将 **Perplexity 的 API** 成本（0.07/0.28）与其他 AI 服务进行了比较，发现其更便宜，并讨论了 AI 扩展通常较高的价格，即每月约 12 美元且 token 使用量有限。
- **Autopilot 使用辩论**：`@stocktown` 询问了使用 Autopilot 的频率，而 `@brknclock1215` 建议通常开启，但也承认在某些情况下禁用可能更好。
- **API 额度使用分享**：`@general3d` 分享了使用每月 5 美元的 API 额度为个人服务器运行 Discord bot 的经验，强调了极低的费用并将其托管在本地。
- **Gemini Pro 与高级模型的对比**：`@luke_____________` 认为虽然 **Gemini Pro** 可能优于其他*免费* AI 聊天机器人，但仍无法与 **GPT-4** 或 **Claude 2.1** 等高级模型相提并论，但期待 Gemini Ultra 的发布可能会改变性能格局。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1191690469592289280)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。
- [Perplexity Blog](https://blog.perplexity.ai/faq/how-does-file-upload-work)：浏览 Perplexity 的博客以获取文章、公告、产品更新以及优化体验的技巧。保持关注并充分利用 Perplexity。

  

---

### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1204383639278985246) (9 messages🔥): 

- **探索商业领域的 AI 应用**：`@krisograbek` 提到学习商业领袖可以利用 AI 的方法。
- **编程语言回忆**：`@grigAI` 分享了快速回忆起多年前使用的编程语言的经历。
- **寻找知识线程**：`@ok.alex` 向 `@1056184370014191686` 和 `@912748706640564244` 询问是否有关于商业 AI 和编程知识回忆的有价值线程。
- **通过 Perplexity 提高效率**：`@glnarayanan` 提供了使用 Perplexity AI 进行研究的反馈，通过分享关于印度 LLP 注册和成本的搜索链接来强调其效率：[LLP 注册与成本](https://www.perplexity.ai/search/What-is-the-Y6lauTzyTFWrcJ0SKGdctw?s=c)。
- **提醒将线程设置为公开**：`@me.lk` 提醒 `@1204584104662929488` 确保将线程设置为公开以便访问，并提供了如何调整线程隐私设置的说明。
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1204426483678978128) (9 messages🔥): 

- **GPT-4 API 访问咨询**：`@erlebach123` 询问是否可以通过 Perplexity API 访问 GPT-4，`@icelavaman` 回复称不可以，只能通过 OpenAI 访问。
- **失效的 iCloud Shortcut 链接**：`@the_only_alexander` 分享了一个 iCloud shortcut 链接，但被报告未找到。他们还确认所讨论的使用场景不需要 API。
- **请求基于 API 的对话追踪**：`@loyah` 请求一个可以使用 API 维持持续对话（而不是每条消息都开始新对话）的 shortcut，但 `@the_only_alexander` 表示他们没有这样的 shortcut。
- **模仿 Perplexity 总结功能的挑战**：`@adventurous_lynx_53570` 寻求关于使用 API 复制 Perplexity Chrome 扩展总结能力的建议，并在通过 API 上传文件时遇到困难。
- **API Key 格式兼容性问题**：`@juan_sc2` 质疑为什么将 Perplexity 的 API Key 格式与 OpenAI 的格式匹配如此困难，以便于在已支持 OpenAI API Key 的工具中使用。

**提到的链接**：

[Shortcuts](https://www.icloud.com/shortcuts/ba386a9ff0de41c7b51a40a01f0cd10f)：未找到描述

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1204594969537683498) (1 messages): 

- **深入探讨 LLM 处理表格数据**：`@jerryjliu0` 宣布了一场网络研讨会，展示 **LLM 表格数据理解** 的先进技术，邀请了两篇重要论文的作者，重点话题包括 **Chain-of-Table** 以及通过聚合多个推理路径来提高模型性能。研讨会定于本周五太平洋时间上午 9 点举行，感兴趣的人员可以[在此注册](https://lu.ma/1cq5hvi4)。
- **探索 LLM 的鲁棒技术**：参与者将学习如何克服在基于表格的问答中面临的朴素方法的典型挑战，并通过 "[Chain-of-Table](https://arxiv.org/abs/2401.04398v1)" 和 "[Rethinking Tabular Data Understanding with Large Language Models](https://arxiv.org/abs/2312.16702)" 等论文深入了解相关研究。活动承诺讨论 *鲁棒表格数据推理* 的新方法。
- **LlamaPack 实现前沿研究**：论文中的先进技术已通过 *LlamaPack 模板* 转化为实用工具，使用户能够在自己的 LLM 应用中应用这些突破。该研讨会是探索这些新实施策略的效率和潜力的机会。

**提到的链接**：

- [Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding](https://arxiv.org/abs/2401.04398v1)：利用大语言模型（LLM）进行基于表格的推理是解决许多表格理解任务（如基于表格的问答和事实验证）的一个有前景的方向。与 g... 相比
- [Rethinking Tabular Data Understanding with Large Language Models](https://arxiv.org/abs/2312.16702)：大语言模型（LLM）已被证明能够胜任各种任务，但其在解释和推理表格数据方面的能力仍是一个尚未充分探索的领域。在此背景下，本研究...
- [LlamaIndex Webinar: Advanced Tabular Data Understanding with LLMs · Zoom · Luma](https://lu.ma/1cq5hvi4)：使用 LLM 对表格数据进行问答和理解具有挑战性。朴素方法（将文本直接丢入 prompt、text-to-SQL）通常效果不佳...

  

---

### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1204449406473211945) (5 条消息): 

- **关于企业级 LLM 和 RAG 的演讲**：关注 `@seldo` 今天关于企业级**语言模型 (LLM)**和**检索增强生成 (RAG)**的讨论。有关演讲的更多详情可以在[这里](https://t.co/ddp5VRiby3)找到。
- **检索动态中的 Self-RAG 演进**：`@AkariAsai` 提出的 **Self-RAG** 方法允许语言模型执行动态检索，利用检索令牌并评估检索是否必要，从而提高模型的生成能力。欲了解更多信息，请点击[此链接](https://t.co/na6n0kw2kX)。
- **Mistral 的新 RAG 文档**：`@MistralAI` 发布了关于 RAG 的新文档。查看 `@sophiamyang` 的贡献，或通过访问[此指南](https://t.co/LVMTG0YJ43)学习如何用 10 行代码通过 **Mistral 和 LlamaIndex** 实现 RAG。
- **构建 Agentic RAG 的 Cookbook 条目**：通过其全新 Cookbook 中的条目探索如何使用 Mistral 构建 **agentic RAG**，链接见[此处](https://t.co/g6QWClaKa9)。
- **关于使用 LLM 进行高级表格数据理解的网络研讨会**：在即将举行的 LlamaIndex Webinar 中，探索如何使用 LLM 增强对**表格数据 (tabular data)**的问答和理解。这挑战了诸如文本转储 (text dumping) 和 text-to-SQL 转换等传统方法。更多信息请见[此处](https://t.co/1yo21Z5QDN)。

**提到的链接**：

- [cookbook/llamaindex_agentic_rag.ipynb at main · mistralai/cookbook](https://t.co/g6QWClaKa9)：通过在 GitHub 上创建账号，为 mistralai/cookbook 的开发做出贡献。
- [Basic RAG | Mistral AI Large Language Models](https://t.co/LVMTG0YJ43)：检索增强生成 (RAG) 是一种 AI 框架，它协同了 LLM 和信息检索系统的能力。它对于利用...来回答问题或生成内容非常有用。

---

### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1204438156628197447) (38 条消息🔥): 

- **PDF 解析难题**：`@nbulkz` 寻求关于使用 SummaryIndex 解析单个 PDF 以进行 LlamaIndex 问答的建议，并说明了相比于使用向量索引 (vector indices)，他们更倾向于此，因为需要精确的文本匹配。`@cheesyfishes` 建议使用 `ServiceContext.from_defaults(llm=llm, embed_model=None)`，以便在将每个查询直接与上传的 PDF 动态关联时不使用 embeddings。

- **初学者 RAG 游乐场分享**：`@wrapdepollo` 为 RAG 和 LlamaIndex 的初学者上传了一个[简单的 GitHub 仓库](https://github.com/jotarretx/RAG_Tester)，欢迎反馈并提供帮助。

- **Neo4j 中的图标签限制**：`@mikefseaff` 询问在使用 LlamaIndex 在 Neo4j 中生成图时是否可以向节点添加额外的标签，`@cheesyfishes` 回复称在当前设置下无法实现，并建议考虑知识图谱 (KGs) 的替代方案。

- **已解决节点内容提取缓慢的问题**：`@gkossakowski` 在从索引中的 990 个节点获取内容时遇到了性能缓慢的问题，但发现了一种更有效的方法，即直接遍历 document store 的值。

- **澄清 Documents 与 Nodes 的区别**：`@bin4ry_d3struct0r` 询问了在处理 `VectorStoreIndex` 时使用 documents 或 nodes 的实际区别。`@cheesyfishes` 澄清说，documents 会被解析器分块 (chunked)，而 nodes 不会，这意味着 nodes 通常是预先分块好的。

- **SQL 查询合成困扰**：`@dieghox90` 在尝试使用 LlamaIndex 和本地 LLAMA2 模型从 SQL 查询合成响应时遇到错误，感到十分沮丧，收到的错误提示为无效的 SQL 语句。

**提到的链接**：

- [mrm8488/longformer-base-4096-finetuned-squadv2 · Hugging Face](https://huggingface.co/mrm8488/longformer-base-4096-finetuned-squadv2)：未找到描述
- [GitHub - jotarretx/RAG_Tester: Simple playground for RAG parameters](https://github.com/jotarretx/RAG_Tester)：用于 RAG 参数的简单游乐场。通过在 GitHub 上创建账号，为 jotarretx/RAG_Tester 的开发做出贡献。

### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1204449204399775864) (6 messages): 

- **寻求最佳 NL to SQL 转换器**：`@prodicer21` 询问目前可用的、不会产生幻觉的最佳自然语言到 SQL (NL to SQL) 解决方案。
- **Hacker News 关于 SQL 解决方案的讨论帖**：`@the_xyt` 分享了一个 [Hacker News 帖子](https://news.ycombinator.com/item?id=39261486)，讨论了一个在 SQL-Eval 上准确率为 76.5% 的 NL to SQL 解决方案，并将其与 GPT-4 和 sqlcoder-15b 进行了对比。
- **高准确率 SQL 解决方案亮点**：`@the_xyt` 重点介绍了一个在 Hacker News 帖子中提到的解决方案，该方案在 NL to SQL 转换中达到了 93% 的准确率。
- **对 SQL 查询准确性的担忧**：`@the_xyt` 表达了对 SQL 查询准确性的担忧，认为对于不精通 SQL 的用户，即使是 90% 准确率的系统也可能生成有害的查询。
- **RAG 分析与 LlamaIndex 的演进**：`@andysingal` 发布了一篇文章，讨论了 RAG 分块 (chunking) 分析与 LlamaIndex 的集成，暗示了文档分析方面的重大进展。文章标题为 "Self-Chunking Brilliance with RAG Analysis and LlamaIndex Revolution"，可以在 [Medium](https://medium.com/ai-advances/self-chunking-brilliance-with-rag-analysis-and-llamaindex-revolution-dd590d734484) 上找到。

**提到的链接**：

- [Self-Chunking Brilliance with RAG Analysis and LlamaIndex Revolution](https://medium.com/ai-advances/self-chunking-brilliance-with-rag-analysis-and-llamaindex-revolution-dd590d734484)：Ankush k Singal
- [Show HN: Natural-SQL-7B, a strong text-to-SQL model | Hacker News](https://news.ycombinator.com/item?id=39261486)：未找到描述

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1204357371275644938) (15 messages🔥): 

- **优化 RAG 系统性能**：`@bwo_28` 询问如何减少 RAG 系统中相似度搜索所需的时间。`@david1542` 建议使用 **ChromaDB 的持久化客户端 (persistent client)** 将 embeddings 保存到磁盘，通过从磁盘加载数据而不是每次重新创建 embeddings 来加速过程 ([ChromaDB 文档](https://docs.trychroma.com/usage-guide#initiating-a-persistent-chroma-client))。

- **Agent 创建中的异步阻塞**：`@ferasawadi` 在使用异步 Agent 时遇到了阻塞问题，正在寻求关于代码为何没有按预期异步运行的见解。

- **带有自定义参数的 Langchain 音频文件加载**：`@risatoga` 询问如何在 Langchain 中设置模型参数，使 "response_format" 为 "vtt" 而不是 json，特别是在使用 `OpenAIWhisperAudio` 时。

- **Langchain 与 AWS SageMaker 集成**：`@jbower` 寻求在 AWS SageMaker 端点上托管的 Mistral 使用 Langchain 的指导。

- **Langchain 导入中的 ModuleNotFoundError**：`@metaverxe.` 在尝试从 `langchain` 导入 `SQLDatabase` 时遇到了 `ModuleNotFoundError`，正在寻找该错误的解决方案。

- **Langgraph 使用案例查询**：`@sdfjo` 正在寻求目前使用 **langgraph** 的项目或用例信息，旨在了解其应用。

**提到的链接**：

- [WebVoyager](https://www.youtube.com/watch?v=ylrew7qb8sQ)：WebVoyager：使用大型多模态模型构建端到端 Web Agent。WebVoyager 是一款新型的视觉驱动网页浏览 Agent，利用浏览器截图进行操作...
- [🧪 Usage Guide | Chroma](https://docs.trychroma.com/usage-guide#initiating-a-persistent-chroma-client)：选择语言

  

---

### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1204477691391250453) (8 条消息🔥): 

- **成功执行 Curl**: 用户 `@johnda98` 确认其 curl 命令执行成功，并对此表示满意。
- **Agent 示例已更新 Event Stream API**: `@veryboldbagel` 更新了 Agent 示例以演示新的 event stream API，并在 [GitHub](https://github.com/langchain-ai/langserve/tree/main/examples/agent) 上提供了更新后的客户端 notebook 和示例。这些示例展示了逐 token 输出和 tool calls，以及如何使用 Runnable Lambda 自定义 Agent 流式传输。
- **对及时更新示例的称赞**: `@gitmaxd` 称赞 `@veryboldbagel` 最近更新的示例是 *出色工作 (amazing work)*，指出这些示例对于使用 Alpha 服务非常有用，并赞扬了代码中详细的注释。
- **部署循环问题似乎已解决**: `@albertperez.` 遇到了 LangServe 的部署循环问题，在第 9 步不断重启。`@gitmaxd` 询问了项目名称中大写字母可能导致的问题，然而 `@albertperez.` 报告称该问题在未干预的情况下已自行解决。

**相关链接**:

- [langserve/examples/agent at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/tree/main/examples/agent): LangServe 🦜️🏓。通过在 GitHub 上创建账号来为 langchain-ai/langserve 的开发做出贡献。
- [langserve/examples/agent_with_history at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/tree/main/examples/agent_with_history): LangServe 🦜️🏓。通过在 GitHub 上创建账号来为 langchain-ai/langserve 的开发做出贡献。
- [langserve/examples/agent_custom_streaming at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/tree/main/examples/agent_custom_streaming): LangServe 🦜️🏓。通过在 GitHub 上创建账号来为 langchain-ai/langserve 的开发做出贡献。

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1204348485080584202) (8 条消息🔥): 

- **寻找个人 AI 工作教练**: 用户 `@bartst.` 表示有兴趣寻找一个个人（工作）教练 AI，促使 `@david1542` 提议参与头脑风暴并贡献想法，尽管他目前无法参与实际开发工作。
- **rez0 关于安全 AI Agent 的博文**: 用户 `@rez0` 分享了其博文链接，探讨了对功能强大且安全的 AI Agent 的需求，[称其为一篇你会喜欢的读物](https://josephthacker.com/ai/2024/02/05/secure-ai-agents.html)，并强调了开发上述功能的**巨大潜力**。
- **MLBlocks：无代码图像处理 API 构建器**: 用户 `@neil6430` 介绍了 [MLBlocks](https://mlblocks.com/)，这是一个无代码工具，允许用户通过**单个 REST API 调用**，结合 AI 模型和传统函数构建多步图像处理工作流。
- **加密项目寻求人才**: 用户 `@hinayoka` 为一个令人兴奋的加密项目发布了多个**职位空缺**，包括 Web3 Developer、Game Developer、Web Developer、Moderator 和 UI/UX Designer，要求具备相关技术经验和强大的团队协作能力。
- **对无代码图像处理项目的称赞**: 用户 `@djabatt` 表扬了 `@neil6430` 的无代码图像处理 API 构建器项目，称其**非常棒**并认可了其中的付出。

**相关链接**:

- [ML Blocks | Home](https://mlblocks.com/): ML Blocks 让你无需编写任何代码即可构建 AI 驱动的图像生成和分析工作流。
- [Joseph Thacker (@rez0__) 的推文](https://x.com/rez0__/status/1754860746070999076?s=20): 我写了关于我们需要更强大、更安全的 AI Agent 的文章。我想你会喜欢的 😊 https://josephthacker.com/ai/2024/02/05/secure-ai-agents.html

  

---

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1204367956813946920) (23 messages🔥): 

- **AI 个人助手与现实世界的使用**：`@ashpreetbedi` 回应了 `@jozexotic` 对 AI 个人助手的怀疑，提到了它们在特定场景下的实用性，例如总结每日站会（daily standups）。
- **AI 辅助编程**：`@slono` 分享了一个 [GitHub Gist 链接](https://gist.github.com/wesen/a4ca759275f1a2bb2a9d4bf4b4b57769)，展示了一种自动化的编程方法，这引发了关于其是否与助手 'Aider' 相似的简短讨论。
- **AI 对 RPA 的影响受到关注**：`@pennepitstop` 强调了人们对 AI 赋能机器人流程自动化（RPA）以实现个人自动化的兴趣日益浓厚，并提到了 Adept 等公司及其对 UiPath 等老牌企业的挑战。
- **探索带有 API 端点的向量数据库**：`@henriqueln7` 询问了哪些向量数据库为生产环境提供 API 端点，`@swyxio` 建议尝试 Supabase pgvector。
- **强调 AI Scaling 见解**：`@swyxio` 指出了 [Stella Biderman 的一条推文](https://twitter.com/BlancheMinerva/status/1754960269250339286)，讨论了 AI 中的 Scale 问题，该推文因概括了对 RWKV 和 Mamba 等模型的看法而获得认可。

**提到的链接**：

- [create-notices.md](https://gist.github.com/wesen/a4ca759275f1a2bb2a9d4bf4b4b57769)：GitHub Gist：即时分享代码、笔记和片段。
- [Effective Data Augmentation With Diffusion Models [NeurIPS 2023]](https://youtu.be/IKDWOOWzwns?si=kr_ErWuS1RciGvF0)：来自 NeurIPS 2023 生成式 AI 合成数据生成研讨会的 DA-Fusion 25 分钟演讲。完整论文：arxiv.org/abs/2302.07944
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind)：一个多模态、支持 function calling 的 LLM WebUI。

  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1204337689013256192) (3 messages): 

- **对 AI 挖矿设备的幻想**：`@timjones1` 表达了对挖矿设备和车库的奇思妙想，暗示了建立个人计算环境的愿望。
- **在 GPU 上从小处着手，怀揣大梦想**：`@joseph_en` 建议从单块 3090 GPU 开始，并考虑扩展到双卡配置。他还建议了一个性价比高的选择：带有 12GB VRAM 的 3060，可以运行量化后的 13b 参数模型，价格约为 250 美元。
- **硬件优化方面的进展**：`@vim410` 对一些看似增量的工作发表了评论，但承认这些工作揭示了未被利用的硬件特性，暗示通过优化和硬件定制仍有进一步提升性能的空间。
  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1204399651034505226) (7 messages): 

- **使用 NVIDIA 工具进行更好的基准测试**：`@cudawarped` 建议使用 Nvidia 的 `ncu` 工具来**测量 Kernel 执行时间**，从而避免启动延迟（launch latency）和设备到主机（device-to-host）内存复制的开销。他们分享了该工具的使用方法，并提供了一个运行 `ncu` 的 [示例命令](https://github.com/cudawarped/cuda_mode_lectures/blob/rgb_to_grey/lecture3/rgb_to_grey.py)。

- **讨论 OpenCL 和 CUDA Kernel 调优**：`@iron_bound` 询问了 [CLTune](https://github.com/CNugteren/CLTune) 的实用性，这是一个用于 OpenCL 和 CUDA 的自动 Kernel 调优器。_tvi_ 回应强调了候选生成（candidate generation）和动态形状（dynamic shapes）处理的重要性，而 `@cudawarped` 指出 CLTune 对于现代 CUDA 版本可能已经过时。

- **介绍用于 GPU 监控的 GMON**：`@smexy3` 开发了一个名为 `gmon` 的工具，旨在**简化训练任务期间的 GPU 监控**。他们提供了该 [工具的 GitHub 仓库](https://github.com/AdamLouly/gmon) 链接以及安装和使用说明，包括在训练结束时生成 HTML 格式的 GPU 显存使用报告的功能。

**提到的链接**：

- [GitHub - CNugteren/CLTune: CLTune: An automatic OpenCL &amp; CUDA kernel tuner](https://github.com/CNugteren/CLTune)：CLTune：一个自动 OpenCL 和 CUDA Kernel 调优器。
- [GitHub - AdamLouly/gmon: GPU Monitor for python.](https://github.com/AdamLouly/gmon)：Python 的 GPU 监控器。
- [GitHub - AdamLouly/gmon: GPU Monitor for python.](https://github.com/adamlouly/gmon.git)：Python 的 GPU 监控器。

  

---

### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1204511384415633439) (1 条消息): 

- **TorchAO 中的动态量化障碍**：`@hdcharles_74684` 在尝试于 torchao 中实现高性能动态量化时遇到了问题，原因是 Torch Compile 中出现了非预期的**操作融合顺序 (order of operation fusions)**。为了解决逐元素操作在矩阵乘法之前被融合，从而阻止了与反量化的最佳融合这一问题，他们手动向内核添加了一个 **dequant epilogue**，并链接到了相关的 [GitHub commit](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L295)。

**提到的链接**：

[pytorch/torch/_inductor/kernel/mm.py at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L295)：Python 中具有强大 GPU 加速功能的张量和动态神经网络 - pytorch/pytorch

  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1204582112427118612) (4 条消息): 

- **为 2015 款 MacBook Pro 寻找 GPU 替代方案**：用户 `@boredmgr2005` 询问是否可以在配备 Intel Iris Graphics 的 2015 款 MacBook Pro 上学习课程材料，或者从云端租用 GPU 是否是唯一选择。
- **云资源解围**：`@joseph_en` 建议尝试 Google Colab，并强调了其免费服务以及处理课程需求的能力。

**提到的链接**：

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1189498204333543425/1191300313928433664/1198770808122785912)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、闲逛，并与你的朋友和社区保持紧密联系。

  

---


### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1204566165662212126) (2 条消息): 

- **Jax 在技术圈兴起**：`@joseph_en` 表达了学习 **Jax** 的兴趣，并提到了它的流行程度。他们询问了 Google 将 Jax 开发为 TensorFlow 潜在竞争对手的战略举措。
  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1204376620803170365) (6 条消息): 

- **SFT 数据集建议**：`@bjoernp` 主张使用**原始 SFT 数据集**进行训练，因为这最能代表模型的预期任务。
- **探索多语言适配**：`@johannhartmann` 确认他们将尝试使用原始 SFT 数据集进行基准测试，并提到在 laserRMT perplexity 中使用了多语言 c4 的德语部分、德语维基百科以及 *malteos wechsel_de*。
- **Axolotl 训练后的 OOM 困扰**：`@philipmay` 报告称 **Axolotl** 在训练完成后仍然出现显存溢出 (OOM)。
- **Deepspeed 配置注意事项**：针对 OOM 问题，`@bjoernp` 建议这可能是 Deepspeed 的配置问题，特别是 `stage3_gather_16bit_weights_on_model_save` 设置，该设置要求模型能够装入单个 GPU。
  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1204535226814959696) (4 条消息): 

- **Jina Embeddings 表现不佳**：`@sebastian.bodza` 对 **Jina embeddings** 表示不满，称其往往**达不到预期**。
- **OOD 代码文档问题**：`@sebastian.bodza` 还提到，对于代码文档，这些 embeddings 有点**分布外 (OOD)**。
- **对 Embedding 性能表示失望**：`@rasdani` 对 `@sebastian.bodza` 分享的问题做出了失望的回应："*hm what a pitty :/*"。
  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1204362164408811540) (5 条消息): 

- **内部硬件雄心**：`@flozi00` 提到他们的雇主已经建设了自己的数据中心，提供服务器解决方案并通过专门部门进行管理。
- **分层推理方案建议**：`@lightningralf` 建议为德语推理服务创建两个价格层级，其中一个包含作为开源的数据，另一个作为私有推理。
- **免费服务考量**：针对双层定价模型，`@flozi00` 指出他们正在探索某种包含使用赞助的免费服务。
  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1204488573575897088) (2 条消息): 

- **Lindy AI 的初步印象**：`@thebaghdaddy` 分享了他们使用 **Lindy** 的个人经验，表示他们对该功能进行了“有限的测试”，发现它在提取数据和撰写报告等任务上表现尚可，尽管他们认为更定制化的系统可能对特定任务效果更好。
  

---

### LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/) (1 messages): 

.psychickoala: Azure 是否有 GPT 4 Vision 模型
  

---


### LLM Perf Enthusiasts AI ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/1204487819360206918) (2 messages): 

- **用户 sourya4 正在寻求建议**：`@sourya4` 在对话中确认收到了建议，但询问其他人是否已经尝试过，表现出好奇并寻求社区反馈。
- **介绍 Super JSON Mode**：`@res6969` 分享了来自 `@varunshenoy_` 的推文，宣布推出 **Super JSON Mode**。这是一个专为 LLM **低延迟结构化输出生成**设计的新框架，声称在 OpenAI 和开源模型上的生成速度最高可提升 **20 倍**。该框架无需使用诸如威胁模型或给 AI 打赏等非常规手段。

**提到的链接**：

[Varun Shenoy (@varunshenoy_) 的推文](https://x.com/varunshenoy_/status/1754967233141633513?s=46)：介绍 𝗦𝘂𝗽𝗲𝗿 𝗝𝗦𝗢𝗡 𝗠𝗼𝗱𝗲，一个用于 LLM 低延迟结构化输出生成的框架。在 OpenAI 和开源模型上生成 JSON 的速度最高可提升 𝟮𝟬 倍。❌ 无需...

  

---


### LLM Perf Enthusiasts AI ▷ #[cost](https://discord.com/channels/1168579740391710851/1169026016887459961/1204693376478482432) (1 messages): 

- **为 MythoMax 寻求更好的托管服务**：用户 `@jmak` 询问是否有人在使用 **MythoMax**，并正在寻找更高效的托管服务来向用户部署该 LLM。现有消息中未提供建议或后续讨论。
  

---


### LLM Perf Enthusiasts AI ▷ #[reliability](https://discord.com/channels/1168579740391710851/1169378117865963580/1204510444333703208) (3 messages): 

- **寻求更好的 PDF 处理方案**：`@pantsforbirds` 表达了对处理 **PDF 文档**的担忧，特别是处理来自可选中文档中编码不良的文本。他们在需要时使用基于 **AWS Textract** 的基础提取和生成流水线进行 OCR，但正在寻找预处理或运行时策略来提高可靠性。

- **OCR 作为通用解决方案**：`@res6969` 宣布他们的团队**对所有文档进行 OCR**，无论其是否为可选中文档。尽管成本较高，但这种方法在处理边缘情况方面已被证明有效，且实施该方案是为了加快部署速度。
  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1204507737048875068) (2 messages): 

- **寻求合作对接**：用户 `@blankcoo` 询问应联系谁以进行项目**合作**。
- **新成员问候**：用户 `@mosessamuel` 在 **general-chat** 频道向大家问好。
  

---


### Alignment Lab AI ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/1204432724086235186) (4 messages): 

- **Craig 加入 Alignment Lab**：`@craigba` 在听了 **Jeremy** 的 Latent Space 访谈后，表达了加入 **Alignment Lab AI Discord** 的兴奋之情。Craig 表示愿意在任何与**网络安全**相关的事情上提供帮助，并分享了他在 [threatprompt.com](https://threatprompt.com) 的工作。
- **代码生成中的对抗概念**：`@craigba` 重点介绍了 **AlphaCodium**，这是一个开源 AI 工具，它使用类似于 **GANs** 的对抗概念来创建高完整性代码。感兴趣的观众可以从 [Tamar Friedman 的 5 分钟视频](https://twitter.com/itamar_mar/status/1747957348293824676)中了解更多信息，并通过其 [GitHub 页面](https://github.com/Codium-ai/AlphaCodium)探索 AlphaCodium。
- **感谢深度学习见解**：`@craigba` 感谢 `@fanahova` 在“[微调的终结](https://www.latent.space/p/fastai)”访谈中向 **Jeremy** 提出了一个关键问题，并赞赏了关于在不加入大型科技公司的情况下，如何高效利用时间学习深度学习所分享的实用且诚恳的建议。

**提到的链接**：

[Threat Prompt - AI 安全](https://threatprompt.com.)：未找到描述

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1204534233020891226) (2 messages): 

- **寻找免费的发型应用**：`@soundblaster__` 询问是否有**可以更换发型的免费应用**，但反映很难找到注册后不需要付费的应用，尽管他已经查看了 **Google 搜索结果的第一页和第二页**。
  

---



### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/) (1 messages): 

._z: @everyone 每周例会现在开始。 😄
  

---