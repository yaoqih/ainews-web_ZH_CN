---
companies:
- replit
- anthropic
- togethercompute
date: '2024-09-06T01:54:59.572225Z'
description: '**Replit Agent** 正式发布，作为一款全集成的 Web IDE，它支持具备规划和自愈能力的“文本生成应用”功能，付费用户无需排队即可立即使用。其他值得关注的进展包括：新型文本生成音乐模型
  **Melodio**，以及 **Together AI** 在内核和投机性解码（speculative decoding）方面的工作。**Anthropic
  AI** 宣布了全新的企业版方案，其特色是拥有 **50万（500K）上下文窗口**并增强了安全性。此外，讨论还涵盖了旨在改进图像和视频生成的 **JPEG-LM**
  和 **AVC-LM** 模型，以及围绕 **H100 GPU** 定价的 GPU 市场趋势。**Andrej Karpathy** 等具有影响力的专家也分享了关于
  AI 智能体和自动化的见解。'
id: 8cafe5a8-f419-4ab3-a27a-2c45411199cb
models:
- jpeg-lm
- avc-lm
original_slug: ainews-replit-agent-how-did-everybody-beat-devin
people:
- andrej-karpathy
- mervenoyann
- bindureddy
- rohanpaul_ai
- leptonai
- teortaxestex
title: Replit Agent —— 为什么大家都抢在 Devin 之前发布了产品（抢占了市场）？
topics:
- document-retrieval
- retrieval-augmented-generation
- ai-agents
- image-generation
- video-generation
- context-windows
- gpu-pricing
- enterprise-ai
- self-healing
- text-to-music
---

<!-- buttondown-editor-mode: plaintext -->**一个全集成的 Web IDE 就是你所需的一切。**

> 2024年9月4日至9月5日的 AI 新闻。我们为你检查了 7 个 subreddits，[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord (**214** 个频道，共 **2723** 条消息)。预计节省阅读时间（以 200wpm 计算）：**303 分钟**。你现在可以标签 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

充实的一天。一年一度的 [Time 100 AI 争议](https://x.com/time/status/1831665580241293772?s=46)文章。[Maitai](https://news.ycombinator.com/item?id=41456552)、[AnythingLLM](https://news.ycombinator.com/item?id=41457633)、[Laminar](https://news.ycombinator.com/item?id=41451698) 发布了。[Melodio - 新的 text-to-music 模型](https://x.com/mjlbach/status/1831323536788791595?s=46)。Together ai 宣布了[一些 kernel 工作](https://x.com/togethercompute/status/1831783919718690877?s=46)和 [speculative decoding 工作](https://x.com/togethercompute/status/1831755763615674412)。[Andrej Karpathy 参加了播客](https://x.com/swyx/status/1831742418053689853)。[$2000/月 的 ChatGPT](https://www.theinformation.com/articles/openai-considers-higher-priced-subscriptions-to-its-chatbot-ai-preview-of-the-informations-ai-summit?rc=sy0ihq)。我们几乎要把 [Matt Shumer + Sahil Chaudhary 的 Reflection Tuned 版 Llama 3.1 70B 微调模型作为今天的头条新闻](https://x.com/mattshumer_/status/1831767014341538166)，但 405B 版本和论文下周才会发布，所以我们先提醒你它即将到来。

今日的重磅发布是 [**Replit Agent**](https://x.com/amasad/status/1831730911685308857)。

 
![image.png](https://assets.buttondown.email/images/77a21acd-0945-4a5c-9eff-43aff8e23207.png?w=960&fit=max)
 

如果你一直关注 coding agent 公司的发布——比如 Claude Artifacts、[Cursor Composer](https://x.com/shaoruu/status/1812412514350858634)、[Val.town Townie](https://news.ycombinator.com/item?id=41322818)、[Cosie Genie](https://www.latent.space/p/cosine)、[Honeycomb](https://x.com/snowmaker/status/1831219441327394886)，甚至是[昨天的 You.com 转型](https://buttondown.com/ainews/archive/ainews-1150m-for-ssi-sakana-youcom-claude-500m/)，这基本上就是你对 Replit 的期待，只是执行得非常出色——实现了从纯文本到运行中应用的生成，并具备规划和[自我修复 (self healing)](https://x.com/frankismartinez/status/1831766482642202881)能力。值得称赞的是**没有等待名单**——它今天已对付费用户开放——并且可以部署在带有 postgres 后端的实时 URL 上，甚至适用于[不会写代码的人](https://x.com/emollick/status/1831855794356379914)，包括[在手机上](https://x.com/amasad/status/1831759801736626341)。当然，Replit Agent [甚至可以制作一个 Replit 克隆版](https://x.com/amasad/status/1831858971847880873)。

遗憾的是，目前还没有 benchmark 甚至博客文章可写。这让我们的工作变得简单：观看视频，亲自尝试，或者继续往下看。


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

**AI 发展与模型**

- **文档检索技术**：[@mervenoyann](https://twitter.com/mervenoyann/status/1831467222012920164) 强调了多模态 **RAG**（检索增强生成）的方法，建议使用 Donut 或 LayoutLM 等模型，以改进从标记数据中获取结构化响应的效果。
- **AI Agents 功能**：[@bindureddy](https://twitter.com/bindureddy/status/1831468638882427178) 解释说 **AI Agents** 可以自动执行各种任务，如文档生成和技术图像生成，使用户能够指定高层级任务并由 AI 执行。
- **图像与视频生成**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1831477152769774051) 详细介绍了 **JPEG-LM** 和 **AVC-LM** 的开发，这些模型利用文件编码来增强图像和视频生成。该方法在降低数据复杂性的同时，提供了令人印象深刻的输出质量。

**AI 工具与技术**

- **企业级新功能**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1831475071250223411) 透露了 AnthropicAI 的新企业计划，具有 **500K context window**（上下文窗口）和改进的安全措施等重大功能，针对营销和工程领域的特定用例。
- **GPU 市场趋势**：[@LeptonAI](https://twitter.com/rohanpaul_ai/status/1831480368592990507) 讨论了 **H100 GPU** 定价模型的趋势，预测其成本将出现类似于 A100 GPUs 的下降，并强调了监控和测试可靠性的重要性。

**AI 哲学与伦理**

- **探究的重要性**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1831468511123927197) 批评了科学家缺乏好奇心的现象，认为需要对基本问题进行更深入的探究，而不是接受肤浅的解释。
- **研究影响力**：[@stanfordnlp](https://twitter.com/stanfordnlp/status/1831470314108416314) 分享了关于研究生如何参与具有影响力的 AI 研究的见解，这与关于对该领域做出有意义贡献的广泛讨论相一致。

**社区与协作**

- **NLP 活动社交**：[@stanfordnlp](https://twitter.com/stanfordnlp/status/1831468959000051985) 发布的一则研讨会公告宣传了关于 "The State of Prompt Hacking" 的演讲，邀请各界参与并强调了社区参与在讨论 NLP 突破中的重要性。
- **来自领导层的底层洞察**：[@RamaswmySridhar](https://twitter.com/RamaswmySridhar/status/1831473429590991075) 分享了关于扩展组织的思考，强调了**透明度**和**问责制**作为高增长公司关键驱动力的必要性。
- **导师指导与机会**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1831474077309207030) 认可了社区联系的影响力，主张年轻工程师利用协作关系来实现职业成长。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. GitHub 自动标记：对 AI 模型仓库的影响**

- **Qwen 仓库在 GitHub 上被下架 - 突发新闻** ([Score: 183, Comments: 75](https://reddit.com//r/LocalLLaMA/comments/1f9fa6g/qwen_repo_has_been_deplatformed_on_github/))：据主要贡献者 **Junyang Lin** 报告，**GitHub** 因未知原因暂时标记并移除了 **Qwen** 仓库。该项目在 **Gitee**（中国版 GitHub）和 **Hugging Face** 上仍可访问，文档可在 [qwen.readthedocs.io](https://qwen.readthedocs.io/en/latest/) 查看。帖子作者敦促开源社区建立存档，以防止未来的下架事件。
  - **Qwen 仓库**已在 **GitHub** 上恢复，贡献者 **Justin Lin** 发布推文宣布：*"We are fucking back!!! Go visit our github now!"* 用户讨论了对**备份方案**和**分布式 AI 系统**的需求。
  - 出现了关于 **GitHub** 替代方案的讨论，包括专注于 AI 的 **torrent trackers**（如 [aitracker.art](https://aitracker.art/)）以及去中心化平台（如 [Codeberg](https://codeberg.org/) 和 [Radicle](https://radicle.xyz)）。用户强调了代码托管和协作中独立于平台解决方案的重要性。
  - 一些用户猜测这可能是针对**中国模型**的行为，或者是 **Microsoft** 的介入，并引用了该公司过去的反竞争行为历史。其他人则警告不要过早下结论，建议等待 **GitHub** 对此次临时移除的官方解释。

## AI Reddit 全回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 研究与开发**

- **Logan Kilpatrick** 表示，如果“足够关注”，就会发现 [AI 的进步并未放缓](https://www.reddit.com/r/singularity/comments/1f99mvb/logan_from_google/) (336.5 points)
  - 评论指出 AI 视频和图像生成领域进步神速
  - 部分用户对 AI 研究人员发布的晦涩推文和炒作表示不满

- **OpenAI 联合创始人 Ilya Sutskever** 发布推文 [“是时候攀登了 (time to climb)”](https://www.reddit.com/r/singularity/comments/1f8v5jr/ilya_sutskever_time_to_climb/) (302.5 points)

- **OpenAI** 发布推文 [“我们有很多话想对你说”](https://www.reddit.com/r/singularity/comments/1f8zvlv/openai_we_have_so_much_to_tell_you/) (233 points)

- 根据一条推文，**Anthropic** 正在 [“疯狂交付 (shipping so hard)”](https://www.reddit.com/r/singularity/comments/1f8xdof/anthropic_is_shipping_so_hard/) (190.5 points)

- **Christian Szegedy** 预测 [2026 年前将出现超越人类的 AI 数学家](https://www.reddit.com/r/singularity/comments/1f94ay5/progress_is_faster_than_my_past_expectation_my/)，甚至可能在 2025 年 (140.5 points)

**AI 融资与竞争**

- **Sutskever 的新 AI 安全初创公司 SSI** 已 [融资 10 亿美元](https://www.reddit.com/r/singularity/comments/1f8tl1y/ssi_has_raised_1_billion/) (268 points)
  - [路透社文章](https://www.reddit.com/r/singularity/comments/1f8tnxr/exclusive_openai_cofounder_sutskevers_new/) 报道了此次融资 (118 points)

- 据报道，**OpenAI 及其竞争对手** [对 xAI 的算力感到担忧](https://www.reddit.com/r/singularity/comments/1f92ad8/openai_and_other_competitors_are_reportedly/) (141 points)

**AI 图像生成**

- 一段 [Stable Diffusion 5 分钟之旅](https://www.reddit.com/r/StableDiffusion/comments/1f8xr10/5_minutes_journey_with_stable_diffusion/) 视频展示了该模型的能力 (366 points)

- [Flux Icon Maker](https://www.reddit.com/r/StableDiffusion/comments/1f9eyr7/flux_icon_maker_ready_to_use_vector_outputs/) 使用自定义训练的 Lora 和 ComfyUI 工作流生成矢量图标输出 (213 points)
  - 支持直接转换为矢量图形以实现可扩展性
  - 使用 ComfyUI-ToSVG 仓库进行矢量转换

---

# AI Discord 回顾

> 由 Claude 3.5 Sonnet 生成的摘要之摘要的摘要

**1. LLM 进展与基准测试**

- **DeepSeek V2.5 发布**：**DeepSeek V2.5** 合并了其 **Coder** 和 **Chat** 模型，在多项性能指标上表现出显著提升，例如 **ArenaHard 胜率**从 **68.3% 提升至 76.3%**。[点击此处阅读更多](https://platform.deepseek.com/api-docs/news/news0802/)。
  - 用户对这些升级表示赞赏，在保持指令遵循能力的同时增强了整体可用性。[更新日志](https://platform.deepseek.com/api-docs/updates)。
- **Reflection 70B 模型发布**：新的 **Reflection 70B** 模型引入了用于自我修复的 **Reflection-Tuning**，在社区中引起了轰动。[Matt Shumer 的发布公告](https://x.com/mattshumer_/status/1831767014341538166?s=46&t=2a7uDiV3mox9o-E5jIFbLQ)。
  - 成员们热切期待即将推出的 **405B** 版本，预计其表现将超越现有的替代方案。[推文](https://x.com/mattshumer_/status/1831767014341538166?t=MKrJQ-X4VjS_MpTLpP4jDg&s=19)。
  - 这种创新方法可以显著提高模型性能，引发了关于其潜在应用及对模型设计影响的讨论。[研究论文](https://openreview.net/forum?id=xaqoZZqkPU)。

**2. AI 行业新闻与融资**

- **xAI 的集群引发竞争担忧**：Elon Musk 在构建 xAI 的 **100k GPU 集群**方面取得的进展引起了竞争对手模型开发者的担忧，OpenAI 的 Sam Altman 对潜在的算力差距表示担忧。
   - 这一消息引发了关于 AI 军备竞赛升级的讨论，一位社区成员幽默地指出：*“最终我们都会变成 GPU 穷人”*。
- **OpenAI 雄心勃勃的定价策略**：有报道称 OpenAI 正在考虑为访问其下一代模型设定每月高达 **2,000 美元**的订阅费，这暗示其能力可能比低端版本提升 100 倍。
   - 社区反应充满怀疑，一位成员表示：*“这将是 Vision-Pro 级别的灾难。我希望这只是个玩笑”*。其他人推测这可能更适合 B2B 定价模式。

**3. 多模态 AI 创新**

- **Transfusion 模型见解**：Meta 发布了一篇关于 **Transfusion** 模型的论文，这是一种在 **1T** 文本 Token 和 **692M** 图像上集成语言和扩散训练的多任务处理方法。[Transfusion 论文](https://www.arxiv.org/abs/2408.11039)。
  - 强调了该方法与传统的离散 Token 训练相比，具有更好的扩展性能。[Transfusion 论文](https://www.arxiv.org/abs/2408.11039)。
- **Loopy：音频驱动的视频生成**：论文介绍了 **Loopy**，这是一种端到端的音频条件视频扩散模型，旨在合成自然运动而无需手动空间模板。[Loopy 论文](https://huggingface.co/papers/2409.02634)。
  - Loopy 增强了音频与肖像动作的相关性，并根据广泛的实验结果展示了性能的显著提升。[Loopy 论文](https://huggingface.co/papers/2409.02634)。
- **Comfy 重写项目受到关注**：**Julien Blanchon** 宣布从零开始进行极简的 **Comfy** 重写，旨在创建一个高度可扩展且无依赖的用户界面。该项目邀请合作，以在保持灵活性的同时简化使用。
  - 成员们对旨在增强用户体验和降低复杂性的改革表示了兴趣，[更多详情请点击此处](https://x.com/JulienBlanchon/status/1831719118434709868)。


---

# 第 1 部分：高层级 Discord 摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Hash Rosin 模型狂热**：一位用户正在寻找生成逼真的 hash rosin 图像的最佳模型，并参考了一个提供详细微距特写镜头的特定 [Lora](https://civitai.com/models/487689/hash-rosin)。
   - 建议包括将 Lora 与 **SDXL** 或 **Flux** 等模型配对，以提升输出质量。
- **ControlNet 难题**：一位用户在 ComfyUI 中使用 ControlNet 预处理器时遇到困难，特别是除了 tile 预处理器之外缺少其他选项。
   - 用户建议尝试 tiled ksamplers 并检查设置的准确性，同时推荐了教程资源。
- **安装见解**：讨论围绕尝试各种模型组合展开，重点是使用 **Flux** 和 **SDXL** 来获得卓越的图像生成效果。
   - 参与者热衷于学习如何将不同模型与 Lora 集成以达到预期效果。
- **GPU 性能困境**：用户讨论了 GPU 性能限制，特别是在使用 **SDXL** 和 **Flux** 等大型模型时对 VRAM 的关注。
   - 对生成时间过长的担忧促使人们建议探索云服务，以提高容量和效率。
- **云计算好奇心**：许多人推荐使用 Vast.ai 等云平台来访问高性能 GPU，以运行高需求模型。
   - 对云解决方案的需求引起了共鸣，尤其是对于使用笔记本电脑等低配置设备的用户。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 获得 Y Combinator 支持**：Unsloth 宣布获得 [Y Combinator](https://www.ycombinator.com/companies/unsloth-ai) 的支持，这标志着其发展过程中的一个重要里程碑。
   - 团队对未来的发展感到兴奋，包括他们新庆祝的 **200 万次月下载量**。
- **Unsloth 新功能揭晓**：Unsloth 将推出用于模型微调的 **Unsloth Studio**，而对于用户来说，**Dora** 集成仍需要设置 `use_dora = True` 才能使用。
   - 讨论还强调了热门模型推荐，如 **Gemma 2 27B** 和 **Llama 3.1 8B**，社区成员分享了他们的实验心得。
- **Illya 为 AGI 筹集 10 亿美元**：Illya 最近为 Safe SuperIntelligence 筹集的 **10 亿美元** 资金引发了关于其对扩展 AGI 和 LLM 推理影响的困惑。
   - 成员们指出，*没有证据表明扩展（scaling）会导致 AGI*，并指出这些投资通常是由炒作驱动的。
- **LLM 推理研究**：社区讨论了 LLM 中推理和规划的挑战，断言单纯的扩展（scaling）无法提高这些能力。
   - 见解表明，有效的推理可能需要 *架构创新或显式的推理机制*。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **关于 AI 与人类认知的辩论**：一场关于 AI 推理与人类理解之间差异的热烈讨论展开，强调 LLM 利用的是统计预测而非真正的认知。
   - 参与者指出，虽然 AI 模拟了意识，但它本质上缺乏生物实体所拥有的真正理解。
- **Perplexity 成为宠儿**：成员们经常称赞 **Perplexity** 的速度和可靠性，特别是在研究和项目任务中，其免费层级对许多用户来说已经足够。
   - 这使得 **Perplexity** 成为 AI 领域其他付费订阅工具的有力竞争替代品。
- **Gemini AI 性能不及预期**：用户分享了对 **Gemini AI** 褒贬不一的体验，特别指出在编程任务中输出不可靠，以及幻觉（hallucinations）影响了回答的准确性。
   - 尽管有这些挫折，一些用户报告说新版本有所改进，这让他们继续探索该工具。
- **OpenAI 达到重大订阅里程碑**：OpenAI 庆祝其付费用户达到 100 万，这主要由 ChatGPT Team 和 Enterprise 等面向业务的产品驱动。
   - 订阅费用起价为每用户每月 **$60**，这突显了在持续的运营成本中存在的重大收入机会。
- **UI 变更引起用户困惑**：ChatGPT 用户界面的最新变化，特别是重新生成按钮的缺失，让用户感到困惑且不确定如何导航。
   - 一些用户推测界面元素被移到了模型选择下拉菜单中，影响了易用性。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **视觉语言模型 (Vision Language Models) 概述**：一篇新的 [博客文章](https://www.lightly.ai/post/introduction-to-vision-language-models) 介绍了 **Vision Language Models** 的基础知识，旨在面向该领域的新手。
   - 它作为理解视觉与语言集成应用背后关键原理的资源。
- **Tau LLM 的流线型优化**：[Tau LLM](https://youtube.com/live/flwqvE4aSzA?feature=share) 系列探讨了增强训练过程和性能指标的方法。
   - 来自社区专家的见解指导了模型效率和部署策略的改进。
- **InkubaLM-0.4B 扩展语言表示**：[InkubaLM-0.4B](https://huggingface.co/spaces/Tonic/Inkuba-0.4B) 的发布解决了对非洲语言的支持，展示了多语言能力的进步。
   - 该项目代表了社区在增强 **AI 应用多样性** 方面的广泛努力。
- **Kyber Odyssey 应对后量子加密**：团队宣布其关于实现 NIST 后量子加密协议的提交已被 AMA 研究挑战赛接收，代码可在 [GitHub](https://github.com/qompassai/KO) 上获取。
   - 他们的工作优先考虑 **学习者和社区的可访问性**，以极低的成本增强安全协议。
- **Qwen2-VL-7B-Instruct 处理程序发布**：针对 **Qwen2-VL-7B-Instruct** 的可用 [handler.py](https://huggingface.co/hperkins/Qwen2-VL-7B-Instruct/tree/main) 和更新的 requirements.txt 展示了其在 T4、A100 和 L4 等端点上的功能。
   - 这些更新侧重于保持兼容性和性能改进，确保在不同配置下的稳健运行。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.2 下载错误报告**：用户在 **LM Studio 0.3.2** 更新后遇到了“无法获取本地发行者证书 (unable to get local issuer certificate)”错误，阻碍了模型下载。此问题可能与公司网络安全变更或 SSL 证书有关。
   - 这一不便凸显了可能影响企业环境中模型部署时间表的连接挑战。
- **图像 API (Image API) 探索正在进行中**：用户正在寻找高限额的 **免费图像 API 提供商**，并将 **Stable Diffusion** 作为起点。该请求还包括对提供高级成像工具的替代方案的咨询。
   - 对扩展 API 能力的寻求反映了项目工作流中对多样化成像资源日益增长的需求。
- **Reflection 70B 模型受到关注**：以纠正推理错误著称的 **Reflection 70B** 模型现已在 [Hugging Face](https://huggingface.co/mattshumer/Reflection-70B) 上可用。在最近上传后，用户渴望将其集成到 **LM Studio** 中。
   - 该模型的能力被视为社区内开源 LLM 讨论的一个重大进展。
- **用户对新 LM Studio UI 的反馈**：一些用户对 **LM Studio 0.3.2** 中的新 UI 提出了批评，指出元素过大和缺少预设下拉菜单是主要问题。许多人表示希望拥有更紧凑的 UI 并重新引入预设选项。
   - 这些反馈可能会指导未来的 UI 开发，以增强用户体验和功能性。
- **建议 Mac 用户配置最大 RAM**：讨论强调 Apple 用户应追求尽可能大的 **RAM**，*64GB* 是专业 AI 使用的基准。用户鼓励投资 **NAS** 系统以获得高效的存储解决方案。
   - 提升 RAM 将有助于增强处理高要求工作负载时的模型处理能力和性能。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLM 的 Reflection-Tuning**：新推出的 [Reflection-Tuning](https://huggingface.co/mattshumer/Reflection-70B) 方法旨在通过使用故意设计的包含错误的数据集，教导模型在输出生成过程中进行自我修正，从而增强 LLM 的能力。
   - 这种创新方法可能会显著提高模型性能，引发了关于其潜在应用和对模型设计影响的讨论。
- **对 Mergekit 停滞的沮丧**：用户报告在 Colab 中合并微调后的 **Llama 3.1** 模型时，**Mergekit** 停滞在 'Executing graph: 0% 0/1457'，导致无法创建可用的模型。
   - 解决此问题的指导对于社区内顺利进行模型合并过程至关重要。
- **Illya 的 10 亿美元 AGI 融资**：Illya 为 **Safe Superintelligence** 成功筹集了 **10 亿美元**，旨在通过扩展（scaling）努力来解决 **AGI** 的复杂性。
   - 成员们对于仅靠 scaling 是否能解决 **LLM** 的推理局限性仍感到困惑，这反映了 AI 社区中正在进行的辩论。
- **Falcon Mamba 模型发布**：由 Technology Innovation Institute 根据 **TII Falcon Mamba 7B License 1.0** 推出的 [Falcon Mamba](https://falconllm.tii.ae/tii-releases-first-sslm-with-falcon-mamba-7b.html) 现已在 [Hugging Face](https://huggingface.co/tiiuae/falcon-mamba-7b) 上开放获取。
   - 发布博客强调了该模型的竞争优势以及在 Hugging Face 生态系统中的集成，邀请进一步探索。
- **Loopy：音频驱动视频生成的进展**：该论文介绍了 **Loopy**，这是一种端到端的音频条件视频扩散模型，旨在合成自然运动而无需手动空间模板。
   - Loopy 增强了音频与肖像动作的相关性，并根据广泛的实验结果展示了性能的显著提升。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **xAI 的 GPU 集群引发关注**：Elon Musk 为 xAI 开发的 **10 万个 GPU 集群** 引起了竞争对手的担忧，*OpenAI 的 Sam Altman* 对竞争性算力差距表示了担忧。
   - *一位成员自嘲道，我们最终都不可避免地变得 GPU 贫困（GPU poor），* 凸显了 AI 基础设施中不断升级的赌注。
- **Unsloth 与 YCombinator 合作**：**Unsloth** 已获得 **YCombinator** 的支持，利用 **Triton** 和 **CUDA** 开发集成模型创建解决方案，专注于速度和易用性。
   - 鼓励感兴趣的各方加入其 [等待名单](https://unsloth.ai/waitlist) 并查看其 [路线图](https://unsloth.ai/roadmap-yc)。
- **Reflection Llama-3.1 成为顶尖开源 LLM**：**Reflection Llama-3.1 70B** 被誉为领先的开源 LLM，它利用名为 **Reflection-Tuning** 的技术来提高推理准确性，并由 [Glaive](https://glaive.ai) 使用合成数据进行训练。
   - 用户可以在[此处](https://reflection-playground-production.up.railway.app/)体验该模型。
- **寻求有效的推理数据集**：一位成员寻求关于 **推理数据集** 的建议，特别是那些包含 **chain-of-thought reasoning** 的数据集，反映出市场上众多的选择。
   - 著名的建议包括 **MATH** 和 **GSM8k** 基准测试，这些基准因评估 LLM 推理能力而备受推崇。
- **OpenAI 的定价策略引发辩论**：报告显示 OpenAI 可能会考虑高达 **每月 2,000 美元** 的订阅费，鉴于竞争激烈的价格环境，这引发了对市场可行性的怀疑。
   - 成员们对潜在的 **B2B 定价模型** 感到好奇，质疑如此高昂的消费者成本在实践中如何合理化。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Magic 包管理器接管工作**：全新的 **Magic 包管理器** 正式支持 **MAX** 和 **Mojo** 项目，现已提供单个 Conda 包，从而简化了虚拟环境管理。
   - 敦促用户迁移到 **Magic** 或兼容工具，因为旧版的 `modular` CLI 将从周一开始停止更新。
- **Mojo 面临性能审查**：测试显示 Mojo 中的 **ord() 函数** 运行速度比 C++ 和 Python 慢约 **30 倍**，引发了优化呼声。
   - 社区讨论建议检查 **ord 实现** 以及诸如“小字符串优化”（Small String Optimization）等潜在特性以提升性能。
- **模型序列化格式前景不明**：团队目前没有平台无关模型序列化格式的 ETA（预计发布时间），该功能被描述为未来的增强项，预计将有助于容器化。
   - 反馈强调了对该功能的期待，希望能以此简化模型在 Docker 容器中的部署。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **无限银行账户困境**：一位成员幽默地提出了*将银行账户压缩为无限金额*的想法，引发了关于财务限制的激烈辩论。
   - 这引发了一场哲学讨论，另一位成员质疑*压缩成无限量*是否真的意味着扩张。
- **Opus 在特定任务中优于 Sonnet**：一位成员指出，**Opus** 在特定提示词（如计算数字时钟显示屏上的角度）上的表现优于 **Sonnet**。
   - 然而，许多人认为综合基准测试仍然倾向于 **Sonnet**，导致性能评估出现分歧。
- **DeepSeek V2.5 模型取得更高分数**：**DeepSeek V2.5** 的发布合并了其 **Coder** 和 **Chat** 模型，展示了显著的指标提升，例如 **ArenaHard 胜率** 从 **68.3% 跃升至 76.3%**。
   - 用户对这些升级表示赞赏，认为在保持指令遵循能力的同时增强了整体可用性。
- **Reflection 70B 模型发布公告**：新的 **Reflection 70B** 模型将引入 **Reflection-Tuning** 以实现自我修正，在社区中引起轰动。
   - 根据 [Matt Shumer 的公告](https://x.com/mattshumer_/status/1831767014341538166?s=46&t=2a7uDiV3mox9o-E5jIFbLQ)，成员们正迫切期待即将推出的 **405B** 版本，预计其表现将超越现有替代方案。
- **AI Studio 密钥配置失败**：**AI Studio** 用户报告了一个严重问题，即密钥输入无法保存配置，会回退到 **Not Configured**（未配置）状态。
   - 虽然 **Hyperbolic** 和 **Lambda** 密钥功能正常，但这种不一致性引发了用户对可靠性的担忧。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 为学生提供免费会员**：Perplexity 宣布，对于有 **500** 名使用 `.edu` 邮箱的学生注册的大学，将提供为期 **1 年的免费 Pro 会员**，这引发了关于资格和注册标准的疑问。
   - 用户必须在特定日期前注册，对话中还提到了对其所在大学是否参与的不确定性。
- **xAI 的 Colossus 成为焦点**：Perplexity AI 介绍了**世界上最强大的超级计算机**——xAI 的 Colossus，并讨论了**已知最古老的棋盘游戏**塞尼特棋（Senet）。
   - 欲了解更多关于这一突破性发现的信息，请查看此处的 [YouTube 视频](https://www.youtube.com/embed/kb_DJSrHOy4)。
- **使用 Perplexity API 轻松实现文件上传**：一位成员概述了在 Flask 中使用 **Perplexity API** 实现文件上传的方法，详细说明了客户端和服务器端的配置。
   - 该方法修改了 **/query** 路由以接收文件数据，从而实现与 API 提示词的无缝集成。
- **冷水澡引起关注**：成员们深入探讨了[洗冷水澡的好处](https://www.perplexity.ai/search/benefits-of-cold-showers-hMZf7v0AR1KmXfwENQ_xag)，强调了改善血液循环和提升情绪等健康优势。
   - 这一趋势引发了关于日常习惯及其心理益处的讨论。
- **提升 Perplexity API 响应质量**：一位用户寻求关于配置 **Perplexity API** 请求的建议，以模拟 Perplexity 网站的响应质量。
   - 虽然没有提供具体的解决方案，但对增强 API 响应的追求表明了社区对模型性能的兴趣。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Cursor AI 工具评价褒贬不一**：在讨论 **Cursor** AI 编程工具时，几位成员表达了怀疑，称其有时感觉并不好用，尽管在代码检索方面它比免费版更出色。
   - 一位成员指出：“真的有人尝试用它来处理工单（tickets）吗？”，质疑其在实际场景中的有效性。
- **新模型 Reflection 70B 标志着开源 LLM 的里程碑**：通过 **Reflection-Tuning** 精调的开源 LLM **Reflection 70B** 的发布令许多人感到兴奋，后续模型 **405B** 预计将于下周发布并设定新标准。
   - 一位社区成员分享了来自 Matt Shumer 的一条 [推文](https://x.com/mattshumer_/status/1831767014341538166)，强调了该模型自我纠错的能力。
- **深入研究 Pallas Kernel**：成员们探索了在 **Pallas** 中实现的各种 Kernel，可在 [GitHub](https://github.com/google/jax/tree/main/jax/experimental/pallas/ops/tpu) 上找到，展示了针对 Python+NumPy 程序的转换。
   - **Splash Attention kernel** 被重点提及，其实现链接在 [此处](https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py) 以供深入审查。
- **探索 Open Sora 的 CUDA 实现**：一位成员正在尝试用 **CUDA** 和 **C++** 实现 **Open Sora**，并指出这个庞大项目的难度大且进展缓慢。
   - 他们表达了对图形学领域更多进展的期待，希望能看到技术领域的进步。
- **Triton 中的内存受限性能分析**：在**内存受限（memory-bound）**的设置下，性能仍然受限且缓慢，但在较大的 Batch Size 下速度接近 **FP16**，这表明在效率提升方面仍需努力。
   - 讨论还倾向于使用 **Autotuning** 来随着 Batch Size 的增加潜在地提高速度。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **图像生成中的 MCTS：一场辩论**：关于在图像任务中应用 **MCTS (Monte Carlo Tree Search)** 的讨论引发了对其与 AlphaZero 和 AlphaProof 等模型相比逻辑反转的疑问。
   - *一位参与者强调了 MCTS 如何严重依赖之前的步骤*，指出其重点在于增强策略（policies）而非生成策略。
- **对 Creative AI 工作坊的兴趣**：成员们正在寻求有关即将举行的 **Creative AI** 工作坊的信息，旨在利用他们最近关于扩散模型（diffusion models）论文中的见解。
   - 考虑到迫在眉睫的投稿截止日期，人们对其在 **ICCV** 时间线内的相关性表示怀疑。
- **参数缩放：一个陷阱**：人们开始担心在不相应增加数据集大小的情况下增加参数数量的低效性，并引用了 **Chinchilla** 论文。
   - 一位用户建议研究该论文的公式，以更清楚地了解缩放的影响。
- **Transfusion 模型见解**：讨论集中在 [Transfusion 论文](https://www.arxiv.org/abs/2408.11039) 上，该论文提供了在离散和连续数据上训练多模态模型的见解。
   - 讨论强调了与传统的离散 Token 训练相比，该方法产生了更好的缩放性能。
- **AI 提升开发者生产力**：一篇名为 *The Effects of Generative AI on High Skilled Work* 的论文发现，使用 GPT 3.5 等 AI 工具的开发者任务完成率提高了 **26.08%**。
   - 这表明在开发中引入 AI 技术可以显著提高生产力。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SSI Inc 获得 10 亿美元巨额融资**：SSI Inc 在一轮融资中成功筹集了 **$1B**，与此同时 **Sakana** 也获得了 **$100M** 的融资。
   - 在工程讨论中，出现了关于这笔资金可能分配给 **Nvidia** 的推测。
- **You.com 凭借 5000 万美元注资转型策略**：[You.com](https://you.com) 正在从 AI 搜索业务转型，专注于更深层次的生产力 Agent，这得益于最近获得的 **$50M** 融资。
   - 创始人 Richard Socher 强调，在简单查询上与 Google 竞争的效果，不如增强复杂查询的能力。
- **Karpathy 在自动驾驶领域支持 Tesla**：在一段引人入胜的播客中，Andrej Karpathy 预测 **Tesla** 将在自动驾驶技术方面处于领先地位，尽管 **Waymo** 取得了进展，他指出这是一个至关重要的软件与硬件挑战。
   - 他强调了 Tesla 的人形机器人 **Optimus** 在未来工厂应用中的变革潜力。
- **OpenAI 考虑推出每月 2000 美元的模式**：据报道，OpenAI 正在考虑为其下一代模型推出 **$2000/month** 的订阅服务，这暗示其能力可能比低层级版本提升 **100x**。
   - 讨论暗示这要么是为了显著提升模型性能，要么是为了覆盖不断攀升的运营成本。
- **Replit Agent 自动化开发任务**：Replit 推出了 **Replit Agent**，用于在早期访问期间自动执行软件开发任务，包括设置开发环境。
   - 该计划旨在通过将 AI 更深入地集成到编程工作流中，来增强 Replit 的产品能力。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 迎来又一周年**：成员们庆祝了 **Open Interpreter** 的生日，强调了其在 AI 与人类协作方面的成就，并引发了关于“*AGI 已实现，我们现在都可以回家了*”的幽默评论。
   - 这一反思时刻强调了该工具在当今 AI 讨论中的相关性。
- **教 Open Interpreter 新技能**：讨论集中在 **Teach Mode**，用户可以说“*我想教你一些东西*”，以帮助系统根据用户输入开发新技能。
   - 该系统的适应性与 Rabbit Tech 分享的原则一致，展示了其在多种应用中的潜力。
- **开源仓库鼓励协作**：**Open Interpreter** 和 **01** 仓库现已开源，邀请开发者将创新功能集成到他们的应用程序中。
   - 一位用户表达了利用这些开源资源实现 Web 任务自动化的愿望。
- **关于 AGI 的热议**：一位好奇的成员提出了关于 AGI 公告的问题，引发了参与者中兴奋与怀疑交织的情绪，并再次提到“*AGI 已实现，我们现在都可以回家了*”。
   - 这些讨论反映了社区对先进 AI 概念的活跃参与。
- **Fulcra App：仍在等待探索**：用户对 **Fulcra app** 国际发布的兴趣持续升温，新西兰以外的用户寄予厚望。
   - 预期的发布时间表仍不明确，让用户保持期待。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **PyTorch 2.4 编译错误出现**：成员们报告了 **PyTorch 2.4 的编译错误**，特别是在处理 fake tensors 时，建议使用 `os.environ['TORCH_COMPILE_BACKEND'] = 'aot_eager'` 来屏蔽 CI 中的错误。
   - 提出了一个关于 **默认后端 (default backend)** 的潜在 CI 问题，强调了 CI worker 需要更新 gcc 安装。
- **输入填充严重影响性能**：测试显示，使用 **Alpaca 数据集** 进行输入填充（input padding）会导致速度大幅下降，尽管内存占用（memory footprint）有所改善。
   - 建议同时报告已填充和未填充的 token，以便更有效地量化填充对性能的影响。
- **DeepFusionModel 测试增强**：**DeepFusionModel** 的最新更新包括增加了 kv caching 测试，并分享了一个 Pull Request 以供详细审查和反馈。
   - [Pull Request #1449](https://github.com/pytorch/torchtune/pull/1449) 提议覆盖 max cache sequence length，引发了关于其必要性的讨论。
- **Unsloth 获得 Y Combinator 支持**：**Unsloth** 已获得 Y Combinator 的支持，引发了社区对未来支持计划的热情。
   - 随着一名成员表达对类似机会的希望，人们对社区项目格局变化的期待也在增加。
- **关于 Meta 雇佣关系的澄清**：一名成员澄清了关于在 **Meta** 工作的误解，强调并非所有参与者都隶属于该公司。
   - 一名成员指出 *Salman 纯粹是出于对游戏的热爱而参与*，消除了对其职业关系的假设。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **处理 System Prompt 错误**：一位用户在优化其 system prompt 时遇到问题，收到错误提示 **Could not parse & validate the given body**。
   - 另一名成员建议在指定频道提供详细的 prompt，以便获得针对性的帮助。
- **Cohere 有什么新动态？**：成员们渴望了解 **Cohere** 的最新更新，有人指向 [Cohere blog](https://cohere.com/blog) 获取新鲜见解。
   - 该资源重点介绍了客户用例和近期进展，对于理解持续改进至关重要。
- **实现类似 Gmail 的文本建议功能**：一名成员寻求关于使用 Cohere 模型复制类似于 Gmail **Smart Compose** 的 **文本建议功能 (text suggestions feature)** 的建议。
   - 另一名成员强调了上下文提示（contextual prompting）对于实现该功能的重要性。
- **使用 LLM Agent 生成报告**：人们对利用 **LLM Agent** 根据之前的写作风格和会议记录生成利益相关者报告表现出兴趣。
   - 建议范围从针对会议记录的 **结合 Nimble rerank 的 RAG** 到保持写作风格一致性的 **meta prompting 技术**。
- **OpenSesame 2.0 发布重大更新**：**OpenSesame 2.0** 发布，增强功能包括不再需要 ground truth 输入，以及与 **vector DBs** 集成以进行语义搜索。
   - 它还支持多个模型，包括针对 [OpenAI](https://www.loom.com/share/9569b031ddd343b792856fb23e95d77a?sid=341fa6b2-d295-4c4d-aea5-362accc30c7f)、**Gemini** 和 **Cohere** 等平台的功能。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Netchex AI 革新员工支持**：Netchex 使用 LlamaIndex 实现了 **AskHR + Netchex AI**，仅由两名工程师在短短一个月内就为中小型企业转型了员工支持模式。他们采用了 **advanced RAG pipelines** 来实现上下文感知响应，展示了 HR 领域的快速开发能力。[阅读更多](https://t.co/JWz8sgqRj7)。
   - 这一实现展示了 AI 在增强员工互动方面的有效应用，标志着 HR 领域的重大演进。
- **create-llama 引入 Multi-Agent 工作流**：**create-llama** 的最新更新提供了 Python 环境下的 multi-agent 工作流，强调其在各种用例快速部署中的作用。一个示例展示了利用三个 agents 生成博客文章，体现了其灵活性和效率。[点击查看！](https://t.co/nmrtjUw7iL)。
   - 该功能旨在简化内容创作流程，赋能开发者轻松利用 AI 能力进行创新。
- **用于微服务的 llama-deploy 发布**：**llama-deploy** 实现了基于 LlamaIndex Workflows 的无缝微服务部署，标志着部署效率的实质性提升。此次发布借鉴了 **llama-agents** 和 **Workflows** 的经验，增强了开发者的能力。[获取详情](https://t.co/6TmgpPiZxp)。
   - 该系统旨在简化以 AI 为核心的应用部署，这对于快速扩展服务至关重要。
- **安装 llama-index-experimental-param-tuner**：要安装该实验性软件包，请针对 **llama-index** 版本 **0.11.3** 运行 `pip install llama-index-experimental`。一位用户确认此安装步骤对于实现该功能是必需的。
   - 该软件包预计将为寻求利用 LlamaIndex 最新改进的用户提供高级功能。
- **在 LlamaIndex 中设置 Claude**：分享了一份在 LlamaIndex 中使用 Claude 最新模型的全面指南，包括设置说明和 tokenizer 设置。模型涵盖从 **Claude 3 Opus** 到 **Claude 3 Haiku**，并强调需遵循官方文档。
   - 这一集成通过利用先进的语言模型，为构建复杂的应用程序开启了机会。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **征集 AI Agent 平台社区意见**：一位成员正在探索一个用于构建、部署和变现 **AI agents** 的平台，并在研究阶段征求其他开发者的见解。
   - 他们提供 Beta 测试访问权限以换取简短的交流，旨在根据社区反馈优化功能。
- **文档驱动聊天机器人的挑战**：寻求关于一个需要使用 **两个 PDF 文件** 内容进行交互的聊天机器人的帮助，重点在于用户体验。
   - 核心需求包括文档加载、响应生成以及高效的对话管理。
- **探索 Vision Language Models 的进展**：一篇博客文章揭示了从 **CLIP** 等早期模型到 **Flamingo** 和 **LLaVA** 等复杂解决方案的发展历程，强调了视觉和文本数据的联合训练。
   - 引用作品包括 [DALL-E 2](https://openai.com/index/dall-e-2-extending-creativity/) 以及来自 [GPT-4](https://arxiv.org/abs/2303.08774) 和 [PaLM 2](https://arxiv.org/abs/2305.10403) 等著名模型的见解。
- **CodeMaster App 的游戏化学习**：**CodeMaster** 应用已发布，旨在通过游戏化和有科学依据的学习技术提升编程技能。
   - 社区反馈称赞其 **spaced repetition**（间隔复习）功能显著提高了用户参与度和知识留存。
- **从 SQLite 迁移到云解决方案**：讨论了将部署在 **GCP AppEngine** 上的 **ReAct agent** 从 **SQLite** 迁移到 **Postgres** 或 **MySQL** 的方案。
   - 同时也提出了关于重新部署时丢失本地 SQLite 上下文的担忧。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Comfy 重写项目获得关注**：**Julien Blanchon** 宣布了一个从零开始的极简主义 **Comfy** 重写项目，旨在创建一个高度可扩展且无依赖的用户界面。该项目邀请合作，以在保持灵活性的同时简化使用。
   - 成员们对旨在提升用户体验和降低复杂性的改革表示了兴趣，[更多详情请点击此处](https://x.com/JulienBlanchon/status/1831719118434709868)。
- **Reflection 70B 声称具备自我纠错能力**：**Reflection 70B** 被宣布为顶级的开源模型，能够通过 **Reflection-Tuning** 修复自身的错误。报告显示，它在各项基准测试中优于 **GPT-4o** 等模型，且 **405B** 版本即将推出。
   - AI 社区反响热烈，一条值得关注的 [推文强调了其革命性的特性](https://x.com/mattshumer_/status/1831767014341538166?t=DbIKb0tk5JYIwYIMQVB8sQ&s=19)。
- **Transfusion 模型结合多种模态**：Meta 发布了一篇关于 **Transfusion** 模型的论文，这是一种在 **1T** text tokens 和 **692M** 图像上集成语言和扩散训练的多任务方法。它显示出未来扩展到 **audio** 和潜在的 **video** 的潜力。
   - 该研究提出了创新性地使用 VAE 进行无缝媒体转换，这可能对多模态 AI 的发展产生广泛影响，详见 [arXiv 论文](https://www.arxiv.org/abs/2408.11039)。
- **SwarmUI 专注于模块化易用性**：**SwarmUI** 项目旨在为 **Stable Diffusion** 提供一个模块化的 Web 用户界面，优先考虑用户友好性和性能增强。分享了一个 GitHub 链接，强调其目标是让强大的工具变得易于获取。
   - 成员们指出，其可扩展性是一个关键特性，迎合了寻求在 AI 应用中简化操作的用户。更多内容可以在其 [GitHub 页面](https://github.com/mcmonkeyprojects/SwarmUI) 上探索。
- **提出了统一多模态模型**：成员们讨论了 **Transfusion+GameNGen** 模型的愿景，该模型将语言、视觉、音频和游戏引擎集成到一个单一框架中。这种进步可能会重新定义跨 AI 和模态的交互。
   - 这一概念引发了关于集成 AI 解决方案未来的辩论，许多人热衷于探索此类模型的实际意义。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **赏金支付已完成**：所有通过电子邮件申请赏金的人员均已**获得支付**，并鼓励未收到补偿的接收者在未收到时进行报告。
   - 这促进了 tinygrad 社区管理用户奖励的**透明度和效率**。
- **Tinyboxes 租赁方案初具雏形**：分享了一个关于制造 **tinyboxes** 用于销售或从数据中心租赁的概念，强调了硬件的升级路径。
   - 该计划旨在出售过时的硬件，以保持**库存新鲜**，从而实现持续租赁。
- **关于性能定价模型的讨论**：成员们探讨了定价模型，建议成本以 **$/exaflops** 和 **$/tflops*month** 表示。
   - 这突显了**定价结构的复杂性**以及它们如何满足不同用户的需求。
- **对 IR 中 phi 操作的困惑**：一位成员询问了 **IR** 中的 **phi 操作**，询问它与 LLVM IR 在循环体中的放置方式有何不同。
   - 讨论澄清了它不是真正的 phi 操作，并建议将其重命名为 **ASSIGN** 或 **UOps.UPDATE**。
- **关于 cstyle 渲染器的见解**：George Hotz 引导大家关注 **cstyle renderer**，以便更好地理解其在当前讨论中的作用。
   - 这被寻求深入理解的成员认为是一个有用的参考。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Unsloth Phi 可无缝转换为 Llama**：**Unsloth Phi** 架构现在可以无缝转换为 **Llama**，允许使用 **Llama3 配置** 以实现更高效的实验设置。
   - *这一调整为实验效率提供了潜在的提升。*
- **关于 Phi3 挑战的持续讨论**：虽然 **Phi3** 被认为是安全的，但 **Discord 历史记录**中强调了一些需要持续关注的挑战。
   - *成员们表示，虽然它可以使用，但由于性能方面存在歧义，可能需要进一步调查。*
- **Invisietch 寻找小模型**：**Invisietch** 正在寻找一个小模型进行快速实验，反映了社区对易获取资源的需求。
   - *这一追求展示了对敏捷开发策略的广泛兴趣。*
- **Dora 支持已正式确认**：如 [GitHub issue](https://github.com/axolotl-ai-cloud/axolotl/issues/1328) 中所述，**Axolotl** 现在通过使用参数 `peft_use_dora: true` 正式支持 **Dora**。
   - *鼓励成员回顾之前的讨论，以探索类似的功能请求。*
- **Llama-3.1-8B 转型为分子设计引擎**：通过微调和 **DPO**，成功将 **Llama-3.1-8B** 转换为一个根据用户定义属性生成分子的模型。
   - *这一进步使得只需极少的输入指令即可按需创建分子。*

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 用例列表公布**：**DSPy 用例列表**已正式公布，详细介绍了在生产环境中使用大模型（LMs）构建的近 **100 个产品**，详见 [推文](https://x.com/isaacbmiller1/status/1831715783556395369)。
   - 该倡议由核心贡献者领导，旨在收集社区意见并探索 DSPy 背景下的当前部署情况。
- **ColPali 增强文档检索**：一种名为 **ColPali** 的新方法已发布，通过针对视觉丰富文档的延迟交互（late interaction）机制，有效增强了文档检索，详见[此处](https://www.lycee.ai/blog/colpali-efficient-document-retrieval)。
   - 由 **Manuel Faysse** 和 **Hugues Sibille** 开发的 ColPali 通过整合表格和插图等非文本元素，解决了现有系统的局限性。
- **视觉文档检索基准测试发布**：**视觉文档检索基准测试 (ViDoRe)** 已推出，旨在评估跨多种语言和文档类型的检索性能。
   - 该基准测试旨在通过整合比纯文本更广泛的文档元素来改进评估方法。
- **实时编程环节火热进行中**：提醒成员可以通过 [此链接](https://discord.com/channels/1161519468141355160/1161519469777133580) 参加正在进行的 **livecoding** 环节。
   - 这些环节旨在加强社区内的动手编程技能。
- **新论文预警**：分享了一篇新研究论文的链接，见[此处](https://huggingface.co/papers/2409.02889)，重点介绍了与 AI 和模型开发相关的主题。
   - 这一贡献为该领域不断发展的讨论增添了新内容。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **成员寻求多模态 LLM 经验**：一位成员询问了关于结合文本和语音输入的 **多模态 LLM** 的经验，特别是关注训练和微调方面的工作。
   - 这反映了将 **语音能力** 融入 LLM 框架的兴趣日益增长。
- **关于多模态见解的 YouTube 视频**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=GUdoNdTNNaU)，该视频可能涵盖了多模态模型的各个方面。
   - 对于那些旨在项目中实现多模态能力的人来说，该资源可以作为一个宝贵的入门介绍。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **会议需要转录文本**：与会者强调需要一份包含参会者姓名的 **完整会议转录文本**，以提高问责制。
   - *这可以增强未来讨论的参考准确性和问责制。*
- **正在开发重点概念验证**：一位成员正在开发 **报告的概念验证 (PoC)**，表明了项目实施的动手实践方法。
   - *这在保持范围可控的同时，向实际落地迈进。*
- **Agent 工作流的复杂性**：对话中包含了关于利用 **Agent 工作流** 的想法，暗示了项目方法论的潜在转变。
   - *然而，由于缺乏既定标准，对评估 Agent 的复杂性出现了担忧。*

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI 企业峰会定于旧金山举行**：**AI Enterprise Summit** 定于 **2024 年 10 月 2 日**在**旧金山**举行，面向专注于扩展 AI 产品的高管和 AI 爱好者。*使用代码 AIR50 可在购买此独家活动门票时获得 50 美元折扣*。
   - 峰会预计将吸引大批雄心勃勃的专业人士，旨在为与会者提供建立联系和学习的机会。
- **行业领袖登台演讲**：峰会的主旨演讲嘉宾包括 **Paul Baier**（GAInsights CEO）、**Ted Shelton**（Inflection AI COO）和 **Jeremiah Owyang**（Blitzscaling Ventures），他们将分享关于实际商业应用的见解。
   - 这些领导者将提供来自行业的宝贵观点，使其成为所有参与者的重要学习体验。
- **AI 专业人士的社交网络**：峰会提倡**精心策划的聚会**，让 AI 专业人士可以就 AI 产品开发进行社交和协作。这种环境旨在促进该领域领导者之间的建设性对话。
   - 参与者将有机会直接与思想领袖交流，确保高效的思想交换并促进潜在的合作。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM 问题确认**：一名成员确认了关于 **Gorilla LLM** 的问题，并保证他们会*查看*一下。
   - 虽然没有提供更多细节，但这表明了在处理潜在改进方面的参与度。
- **Berkeley Function Calling 见解**：围绕 **Berkeley Function Calling** 的讨论包括对该方法在 **Gorilla LLM** 集成中实用性的查询。
   - 尽管没有具体的评论，但这种兴趣反映了在新型模型中增强 Function Calling 和接口的趋势。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1280980226553413744)** (321 条消息🔥🔥): 

> - `Hash Rosin 的模型推荐`
> - `在 ComfyUI 中使用 ControlNet`
> - `安装与模型搭配`
> - `技术挑战与性能`
> - `云计算选项` 


- **Hash Rosin 的模型推荐**：一位用户正在寻求生成逼真 Hash Rosin 图像的最佳模型建议，并参考了一个特定的 [Lora](https://civitai.com/models/487689/hash-rosin)，该 Lora 可以重现 Hash Rosin 的近距离宏观镜头。
   - 建议包括将该 Lora 与 SDXL 或 Flux 等模型搭配使用，以获得更高质量的输出效果。
- **在 ComfyUI 中使用 ControlNet**：一位用户询问在 ComfyUI 中使用 ControlNet 预处理器时遇到的困难，特别是除了 tile 预处理器外看不到其他选项。
   - 建议用户尝试 tiled ksamplers 并确保其设置正确；相关教程可能也会有所帮助。
- **安装与模型搭配**：讨论中涉及尝试各种模型，强调使用 Flux 和 SDXL 以实现最佳图像生成。
   - 用户表示有兴趣了解如何将不同模型与 Lora 结合以获得理想效果。
- **技术挑战与性能**：用户讨论了其 GPU 的性能，重点关注运行 SDXL 和 Flux 等大型模型时的 VRAM 限制。
   - 用户对生成时间表示担忧，一些用户建议使用云服务以获得更高的容量和更快的处理速度。
- **云计算选项**：建议指向使用 Vast.ai 等云服务来获取强大的 GPU 访问权限，以处理高需求模型。
   - 讨论强调了云端配置的优势，特别是对于使用笔记本电脑等低配置本地设备的用户。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/JulienBlanchon/status/1831719118434709868">Julien Blanchon (@JulienBlanchon) 的推文</a>：尝试找出如何修复 Comfy 👀</li><li><a href="https://civitai.green">Civitai：开源生成式 AI 之家</a>：探索数千个高质量的 Stable Diffusion 模型，分享您的 AI 生成艺术，并与充满活力的创作者社区互动</li><li><a href="https://huggingface.co/h94/IP-Adapter">h94/IP-Adapter · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main">lllyasviel/ControlNet-v1-1 at main</a>：未找到描述</li><li><a href="https://civitai.com/models/155256/stable-diffusion-v15-bf16fp16-no-emaema-only-no-vae-safetensors-checkpoint">Stable Diffusion v1.5 [bf16/fp16] [no-ema/ema-only] [no-vae] [SafeTensors] [Checkpoint] - v1.5-no-ema | Stable Diffusion Checkpoint | Civitai</a>：Stable Diffusion v1.5 [bf16/fp16] [no-ema/ema-only] [no-vae] [SafeTensors] [Checkpoint] ===================== 免责声明：我只是个脚本小子....</li><li><a href="https://huggingface.co/lllyasviel/sd_control_collection/tree/main">lllyasviel/sd_control_collection at main</a>：未找到描述</li><li><a href="https://civitai.com/models/487689/hash-rosin">Hash Rosin - v1.0 | Stable Diffusion LoRA | Civitai</a>：此 Lora 可以重现罐中和点胶工具上的 Hash Rosin 近距离宏观镜头。它也足够灵活，可以用 Rosin 制作各种东西，比如动物...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1280971928370282518)** (254 messages🔥🔥): 

> - `Y Combinator 支持`
> - `Unsloth 模型更新`
> - `模型推荐`
> - `使用 Dora 进行微调`
> - `Reflection Llama-3.1 70B` 


- **Unsloth 获得 Y Combinator 支持**：Unsloth 最近宣布获得 [Y Combinator](https://www.ycombinator.com/companies/unsloth-ai) 的支持，这标志着该团队及其使命的一个重要里程碑。
   - 团队分享了对这一成就的兴奋之情以及未来的发展计划。
- **Unsloth 令人兴奋的新功能**：团队透露了一个名为 **Unsloth Studio** 的即将推出的 UI，用于微调模型，并庆祝月下载量达到 **200 万次**。
   - 鼓励对多 GPU 测试感兴趣的用户表达意向，以获取未来的机会。
- **针对 4090 的模型推荐**：推荐用于实验的热门模型包括 **Gemma 2 27B**、**Mistral Nemotron 12B**、**Phi-3 medium** 和 **Llama 3.1 8B**。
   - 社区积极讨论这些推荐并分享他们的经验。
- **使用 Dora 微调模型**：用于微调的 **Dora** 集成已可用，用户可能需要设置 `use_dora = True` 才能使用。
   - 提醒用户在考虑显存限制的同时，微调模型是可行的。
- **Reflection Llama-3.1 70B 模型发布**：**Reflection Llama-3.1 70B** 模型结合了一种名为 **Reflection-Tuning** 的新技术，以增强推理能力。
   - 社区对其性能感到好奇，并邀请大家进行测试和对比讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/roadmap-yc">Unsloth x YCombinator</a>: 你最喜欢的开源微调包 Unsloth 现在得到了 YCombinator 的支持，我们打算让开源比以往任何时候都更有活力！</li><li><a href="https://huggingface.co/spaces/unclemusclez/ollamafy">Ollamafy (开发中) - unclemusclez 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/republica-de-fifidonia-rick-idk-fake-it-looks-fake-gif-17266845">Republica De Fifidonia Rick GIF - Republica De Fifidonia Rick Idk - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/danielhanchen/status/1831370671756341348">Daniel Han (@danielhanchen) 的推文</a>: 上传了更多 4bit bnb 量化模型到 http://huggingface.co/unsloth，下载速度快 4 倍！1. @NousResearch Hermes 8, 70 & 405b 2. @cohere Command R 32b, R+104b 3. @akjindal53244 Llama 3.1 Storm 4. Reu...</li><li><a href="https://ollama.com/unsloth/unsloth-tutorial">unsloth/unsloth-tutorial</a>: 快速上手大语言模型。</li><li><a href="https://llama-cpp-python.readthedocs.io/en/latest/server/">OpenAI 兼容 Web 服务器 - llama-cpp-python</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/845">在应使用 &lt;|eom_id|&gt; 时，get_chat_template 在工具使用中使用了 &lt;|eot_id|&gt;。 · Issue #845 · unslothai/unsloth</a>: 在 Llama 3.1 中引入了 &lt;|eom_id|&gt; 以支持多轮推理。消息结束。消息代表执行中可能的一个停止点，模型可以在此处通知执行...</li><li><a href="https://ollama.com/unclemusclez/smollm-135m-instruct-devinator">unclemusclez/smollm-135m-instruct-devinator</a>: 在 DEVINator 数据上训练的 SmolLM 135M Instruct，用于 Open Hands (Open Devin)</li><li><a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct">microsoft/Phi-3.5-mini-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Phi-3.5-mini-instruct-bnb-4bit">unsloth/Phi-3.5-mini-instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Phi-3.5-mini-instruct">unsloth/Phi-3.5-mini-instruct · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1281040695024550012)** (35 条消息🔥): 

> - `返校幽默`
> - `对话中的年龄认知`
> - `关于年龄的讨论`
> - `Meme 分享` 


- **Infinit3e 的返校兴奋感**：Infinit3e 宣布为了 **AI** 重返校园，引发了一场关于年龄认知的幽默交流。
   - *Theyruinedelise* 讽刺地评论道，他原以为 Infinit3e 要年长得多，这引发了对年龄刻板印象的讨论。
- **年龄误解引发笑声**：当成员们开玩笑猜测 Infinit3e 的年龄可能在 **20-22** 岁左右时，引发了一场有趣的讨论。
   - Infinit3e 幽默地回应说他实际上已经 **35** 岁了，而其他人也纷纷加入，拿自己的年龄开玩笑。
- **酷老头？一场理论辩论**：MrDragonFox 俏皮地辩论说 **老头也可以很酷**，即便他觉得自己并不符合那个描述。
   - 对话在成员们轻松愉快地互相调侃年龄中继续。
- **分享 Meme 表达幽默**：Infinit3e 分享了一个展示角色好友请求的 **Meme**，将其与正在进行的关于年龄差异的笑谈联系起来。
   - 这个 GIF 幽默地展示了好友请求的数量，为聊天增添了俏皮的氛围。



**提到的链接**：<a href="https://tenor.com/view/fivie-kuu0001-lynxdenis-gif-17972849">Fivie Kuu0001 GIF - Fivie Kuu0001 Lynxdenis - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1280997715974553621)** (18 条消息🔥): 

> - `Vim 中的尾随换行符问题`
> - `使用 Unsloth 进行文本摘要`
> - `在本地运行 Unsloth 处理私有数据`
> - `Gemma 模型对比`
> - `使用用户数据微调 Chatbot` 


- **已实现尾随换行符修复**：一位成员指出 Vim 会添加 **尾随换行符** 的问题，并提交了一个 [PR 来解决它](https://github.com/unslothai/unsloth/pull/993)。此更改与另一个关于聊天格式的问题 (#992) 相关。
   - *Theyruinedelise* 回应道，“*谢谢，我们会检查的！*”，表示社区已收到反馈。
- **Unsloth 无法直接进行片段摘要**：一位用户询问 Unsloth 是否可以使用 AI 来摘要文本片段，但一位成员澄清说 **Unsloth 本身不直接提供此功能**。他们建议使用任何 AI 模型（如 **ChatGPT**）来完成摘要任务。
   - 这表明用户可能不完全了解 Unsloth 提供的模型能力，并被鼓励探索其他 AI 解决方案。
- **文档助力本地模型训练**：一位新用户获悉，应首先查阅 [Unsloth 文档](https://docs.unsloth.ai/) 以获取 **在本地微调模型** 的指导。文档涵盖了创建数据集和部署自定义模型的内容。
   - 成员们强调了帮助有效进行微调过程的关键资源。
- **Gemma 模型对比得到确认**：一位成员询问 **unsloth/gemma-2-9b-it** 是否与 **google/gemma-2-9b-it** 相同，另一位成员确认 **它们确实是相同的**。这一澄清有助于防止在使用模型时产生困惑。
   - 详细的讨论还表明共享资源可能是可以互换的。
- **构建用于 Chatbot 微调的数据集**：一位用户表示有兴趣微调一个 Chatbot，并寻求关于如何从以前的工单和实时聊天中构建数据集的建议。另一位成员建议定义数据格式并专注于特定任务，以实现有效的训练。
   - 对话反映了量身定制的数据集对于在 Chatbot 性能中实现预期结果的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>：Unsloth 新手？从这里开始！</li><li><a href="https://github.com/unslothai/unsloth/pull/993">将 lstrip 修改为 strip 以解决 Ollama 聊天格式中的尾随空格/换行符问题 (#992) 由 rodrigomeireles 提交 · Pull Request #993 · unslothai/unsloth</a>：这与 #992 相关
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 条消息): 

rodrigo_meireles: 你有以某种方式对比它们的报告吗？读起来应该会很有趣。
  

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1281180358099996714)** (2 条消息): 

> - `频道礼仪`
> - `建议频道` 


- **避免消息重复**：一位成员敦促其他人不要在服务器中多次发布相同的消息，以促进更好的频道使用。
   - 这一关于节制的呼吁旨在提高社区内的沟通效率。
- **最佳发布频道**：一位成员建议频道 <#1257011997250424842> 可能是分享某些消息的最佳场所。
   - 这一建议表明，人们正在努力将话题讨论引导至合适的空间。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1281158047875469372)** (4 条消息): 

> - `Illya 的十亿美元融资`
> - `LLM 扩展与 AGI`
> - `LLM 中的推理与规划` 


- **对 Illya 为 AGI 融资的困惑**：一位成员对 Illya 为专注于扩展 AGI 的 Safe SuperIntelligence 筹集 **10 亿美元**的意义表示困惑，质疑扩展（scaling）是否真的能增强 LLM 的推理能力。
   - 另一位成员回应称，*没有证据表明扩展 LLM 会导致 AGI*，并指出这些投资主要是由炒作驱动的。
- **LLM 中令人印象深刻的研究与推理**：一位成员询问了似乎能有效解决 LLM 面临的推理和规划挑战的卓越研究。
   - 作为回应，有人指出，仅仅扩大 LLM 的规模不会产生高级推理能力，真正的推理可能需要*架构创新或显式的推理机制*。


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1280965860474753134)** (259 条消息🔥🔥): 

> - `AI 意识辩论`
> - `Perplexity 使用案例`
> - `Gemini 性能`
> - `OpenAI 订阅增长`
> - `ChatGPT 中的 UI 变化` 


- **关于 AI 意识与认知的辩论**：一场讨论强调了 AI 推理与人类理解之间的区别，强调 LLM 是基于统计预测而非真正的认知来运行的。
   - 参与者认为，虽然 AI 可以模拟意识，但它缺乏生物体固有的真正理解和自我保存本能。
- **Perplexity 作为首选工具**：成员们表达了对使用 Perplexity 的偏好，认为其速度和可靠性是处理研究和学校项目等任务的显著优势。
   - Perplexity 的免费层级被强调为对用户来说已经足够，使其成为付费订阅的一个有吸引力的替代方案。
- **对 Gemini AI 的褒贬不一**：用户报告称 Gemini AI 的性能不稳定，特别是在编程任务中，突出了 Hallucinations（幻觉）和不可靠响应的问题。
   - 尽管存在这些挑战，一些用户注意到新版本的 Gemini 正在显示出改进，并正在尝试使用它们。
- **OpenAI 付费用户达到 100 万**：OpenAI 宣布其面向业务的产品付费用户达到 100 万，这些产品可能包括 ChatGPT Team 和 Enterprise 服务。
   - 企业订阅模式可能相当昂贵，基础价格约为每用户每月 60 美元，这突显了尽管持续运营亏损，但仍具有巨大的收入潜力。
- **ChatGPT 用户界面的更改**：用户注意到 ChatGPT 中的 regenerate（重新生成）按钮消失了，对 UI 的变化感到不确定，一些人建议它被移到了模型选择下拉菜单中。
   - 一些用户报告看不到某些按钮，因此界面似乎正在经历可能未统一应用的更改。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/jasonkneen/status/1831457484134908341?s=46&t=c8IsgKcZo3KjR_KlSmJj_w">来自 Jason Kneen (@jasonkneen) 的推文</a>: http://x.com/i/article/1831453865201340416</li><li><a href="https://reflection-playground-production.up.railway.app">Reflection 70B Playground</a>: 未找到描述</li><li><a href="https://www.bloomberg.com/news/articles/2024-09-05/openai-hits-1-million-paid-users-for-business-ver">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://www.bloomberg.com/news/articles/2024-09-05/openai-hits-1-million-paid-users-for-business-version-of-chatgpt">Bloomberg - Are you a robot?</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1281007630546108546)** (3 条消息): 

> - `GPT 响应问题`
> - `图标消失`
> - `浏览器兼容性`
> - `App 使用挫败感` 


- **GPT 响应的随机问题**：一位用户报告了 GPT 的随机问题，生成新响应会覆盖之前的响应，并导致网站上的图标消失。
   - 他们表达了挫败感，表示无法查看过去的响应，并对 App 感到不满。
- **浏览器兼容性解决方案**：另一位成员建议使用 Chrome 以避免该用户遇到的问题，并建议在不同浏览器中进行测试。
   - 他们还引导用户前往 [OpenAI 帮助中心](https://help.openai.com/) 报告 Bug 或寻求相关问题的协助。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1280983639647064190)** (25 条消息🔥): 

> - `字体问题`
> - `AI 作者模仿`
> - `导出带有错误的输出`
> - `Prompt 中的 Tool Calls` 


- **字体问题导致奇怪符号**：成员们讨论了一个潜在的 **font** 问题，导致生成的输出中出现类似 `youâre` 的 **奇怪符号**。
   - 这与一个进行 API 请求的 **Flutter app** 有关，并提到了可能的转义字符翻译错误。
- **AI 拒绝模仿近现代作者**：一位成员指出，AI 被设计为避免模仿 **近现代或受版权保护的作者**，而是专注于 **Shakespeare** 和 **Dante** 等较早的人物。
   - 他们建议创建风格指南很容易，定义自己的沟通风格可能会更有效。
- **API 调用中的变量输出响应**：一位用户报告了来自 OpenAI API 的不一致响应，偶尔能收到正确的输出，但其他时候会遇到错误。
   - 讨论表明问题可能与用于同 API 交互的 **wrapper** 有关，构建一个更好的 **wrapper** 可能会有帮助。
- **成功实现 Tool Calls**：成员们分享了在 Prompt 中加入 **tool calls** 的经验，表示工具名称必须正确才能成功。
   - 一位成员通过意识到在调用工具后需要包含 **tool result** 来确保结构正确，从而成功解决了他们的问题。
- **分享资源以获得更好建议**：在讨论中，分享了指向外部资源的链接，以帮助寻求问题协助的用户，特别是关于 **tool calls** 的问题。
   - 成员们鼓励查阅社区资源，以获取关于有效使用 OpenAI 功能的更具针对性的建议。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1280983639647064190)** (25 条消息🔥): 

> - `字体缺失问题`
> - `加入 Tool Calls`
> - `字符编码错误`
> - `API 响应一致性`
> - `创建有效的工具链` 


- **识别字体缺失问题**：一位用户指出可能存在 **font missing issue** 影响了他们的 Prompt 和响应，这引发了关于语言兼容性的讨论。
   - 一位成员建议检查 App 中可用的字体以解决此问题。
- **在 Prompt 中加入 Tool Calls**：一位用户询问了如何在 Prompt 中成功加入 **tool calls**，并对来自 OpenAI 的错误消息表示沮丧。
   - 另一位成员分享说，他们经常在单个输出中创建多个 **tool calls**，并强调了使用 **正确工具名称** 的重要性。
- **响应中的字符编码错误**：一位用户报告在 API 响应中收到 **奇怪符号**，并确认这些问题有时涉及转义字符。
   - 有人建议这些可能是被其 **wrapper** 错误翻译的 **撇号**，并指出该问题是不连贯的。
- **API 响应的一致性**：用户讨论了接收 API 响应的不一致性，有些格式正确，有些则不然。
   - 提出了需要构建更好的 **wrapper** 作为实现一致输出的潜在解决方案。
- **澄清 Tool Call 结构**：一位成员澄清了他们的 **tool call** 结构，其中包括带有内容的 **Assistant message**，随后是匹配的带有结果的 **Tool Message**。
   - 提供此信息是为了解决之前在有效实现 **tool calls** 方面的困难。


  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1281347738113675306)** (1 条消息): 

> - `Vision Language Models`
> - `Tau LLM 的优化`
> - `InkubaLM-0.4B 发布`
> - `Shadowbox 工具`
> - `选择性 Fine-tuning` 


- **Vision Language Models 简介探索**：一位认证用户发布的新[博客文章](https://www.lightly.ai/post/introduction-to-vision-language-models)简要概述了 Vision Language Models。
   - *该资源旨在简化初学者对这一概念的理解。*
- **Tau LLM 的优化前景**：查看 [Tau LLM](https://youtube.com/live/flwqvE4aSzA?feature=share) 系列，该系列专注于改进训练过程和模型性能。
   - *该系列承诺提供来自社区领先成员的详细见解。*
- **InkubaLM-0.4B 推进语言支持**：社区欢迎 [InkubaLM-0.4B](https://huggingface.co/spaces/Tonic/Inkuba-0.4B) 的发布，这是一个旨在支持非洲语言的模型。
   - *这一举措展示了在 AI 领域扩大代表性的承诺。*
- **使用 Shadowbox 工具实现无代码 AI**：推出了一款名为 [Shadowbox](https://github.com/darkshapes/singularity) 的无代码构建器，使用户能够使用 FOSS AI 模型创建任务。
   - *该工具旨在让社区中的非编程人员更容易接触到 AI。*
- **语言模型 Fine-tuning 变得简单**：探索关于使用 Spectrum 方法对语言模型进行选择性 [fine-tuning](https://huggingface.co/blog/anakin87/spectrum) 的文章。
   - *内容强调了实现定制化模型性能的实用策略。*


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1280966166784901120)** (193 条消息🔥🔥): 

> - `Comparison of Coding Models` (编程模型对比)
> - `Transformer Attention Explainer` (Transformer Attention 解释器)
> - `Evaluation of Code Generation` (代码生成评估)
> - `Coding Benchmark Quality` (编程基准测试质量)
> - `Future Research Ideas in Code Generation` (代码生成领域的未来研究思路)


- **Comparison of Coding Models**: 成员们讨论了如何确定最佳编程模型，其中 **Llama 3.1 70B** 被提议为首选。
   - 一位成员寻求推荐，而其他人指出存在多个模型在基准测试上过拟合的情况。
- **Transformer Attention Explainer**: 一位成员请求澄清 Transformer 如何将给定 Token 的 Attention 表示为一个单一数值。
   - 问题集中在理解潜在向量空间（latent vector space）中的距离与 Attention 表示之间的联系。
- **Evaluation of Code Generation**: 讨论了为代码输出建立“正确”标签的难度，并探讨了使用错误率进行评估的方法。
   - 成员们指出了代码评估中语义正确性和语用学的重要性，并指出了 LLM 作为评审员（judges）的局限性。
- **Coding Benchmark Quality**: 大家一致认为当前的编程基准测试需要更严谨的评估方法，特别是缺乏良好的正确性标签。
   - 成员们讨论了为不同模型输出创建交互式对比，强调了实用代码的重要性。
- **Future Research Ideas in Code Generation**: 讨论了未来的研究方向，包括使用视觉模型来评估代码的语义和语用。
   - 模型从代码预测渲染帧（以及反之亦然）的潜力被强调为一个令人兴奋的研究方向。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/osanseviero/status/1831415565518565780">来自 Omar Sanseviero (@osanseviero) 的推文</a>: 使用 Flux 的 Latent Navigation 来了 🤯  https://huggingface.co/spaces/latentexplorers/latentnavigation-flux  看看一位 CEO 如何从“守序善良”变为“混乱邪恶”</li><li><a href="https://medium.com/rapids-ai/cybert-28b35a4c81c4">cyBERT</a>: 神经网络，就是这项技术；让你的员工从糟糕的 regex 中解脱</li><li><a href="https://tenor.com/view/shocked-surprised-gasp-what-cat-shock-gif-635629308990545194">Shocked Surprised GIF - Shocked Surprised Gasp - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/spaces/latentexplorers/latentnavigation-flux">Latent Navigation - 由 latentexplorers 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/mike-ravkine/can-ai-code-results">Can Ai Code Results - 由 mike-ravkine 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/aaaaaaa-gif-18466099">Aaaaaaa GIF - Aaaaaaa - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/deep-learning-ai-better-the-devil-you-know-overkill-ai-what-risks-we-come-in-peace-gif-25432236">Deep Learning Ai Better The Devil You Know GIF - Deep Learning AI Better The Devil You Know Overkill - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/smg4-mario-you-dropped-this-king-crown-dropped-crown-gif-26121821">Smg4 Mario GIF - Smg4 Mario You Dropped This King - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/CopyleftCultivars/llama-3.1-natural-farmer-16bit">CopyleftCultivars/llama-3.1-natural-farmer-16bit · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1281141424162607137)** (8 messages🔥): 

> - `AI 中的 Residual Connections`
> - `Jeeds Agent 模型`
> - `Transformers 与 Attention 机制`
> - `使用 Ollama 的 Python 微服务` 


- **探索 Residual Connections**：一位成员提到今天学习了 **Residual Connections 的实现**及其底层机制。
   - 他们的目标是深入理解为什么 **Residual Connections** 在模型架构中如此有效。
- **编写新的 Jeeds Agent 模型**：另一位用户今天专注于 **编写采用 Jeeds 架构的新 Agent 模型**。
   - 这体现了在 AI 开发中应用新方法的显著努力。
- **理解 Transformers 中的 Attention**：一位用户提出了关于在 Transformers 架构中，单个数字如何代表 **给定 token 的 attention** 的问题。
   - 他们询问该值是否源自 **latent vector space 中的距离**，并请求更多讨论该主题的资料。
- **对跨频道发布消息（Cross-Posting）的担忧**：一位用户对在不同频道间 **跨频道发布消息** 表示担忧，要求另一位成员停止这种行为。
   - 这一反复讨论强调了社区关于维护频道讨论清晰度的指南。
- **使用 Ollama 构建 Python 微服务**：一位参与者有兴趣创建一个 **使用 Ollama 的 Python 微服务**，以多种方式对句子进行改写（paraphrase）。
   - 这一尝试暗示了语言模型在开发多功能文本处理解决方案中的应用。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1280981274433032386)** (6 messages): 

> - `GPT4FREE`
> - `Kyber Odyssey 加密实现`
> - `Yi-Coder 发布`
> - `Vision Language Models 的进展`
> - `Comfy 的极简 UI` 


- **探索 GPT4FREE！**：一位成员发现了 [GPT4FREE](https://cf4c-34-32-164-170.ngrok-free.app) 并提议创建一个 Web UI 的在线版本。
   - 该倡议旨在让 GPT 的访问更加用户友好且易于获取。
- **Kyber Odyssey 应对后量子加密（Post-Quantum Encryption）**：一个小组自豪地宣布，他们关于实现 NIST 新后量子加密协议的提交已被 AMA 研究挑战赛接受，并强调通过 [GitHub](https://github.com/qompassai/KO) 上的开源代码为学习者提供可访问性。
   - 他们的目标是以极低的成本赋能传统上被忽视的社区，以增强安全性和隐私。
- **Yi-Coder 已上线！**：[Yi-Coder](https://huggingface.co/spaces/Tonic/Yi-Coder-9B) 已由 01ai 发布，邀请用户试用并贡献示例。
   - 此次发布提供了一个新工具，并通过 PR 展示了社区的参与。
- **Vision Language Models 的最新进展**：一位成员分享了关于一篇博客文章的见解，内容涉及从 CLIP 等早期对比学习方法向 Flamingo 和 LLaVA 等高级模型的过渡，强调了它们的联合训练能力。
   - 像 [DALL-E 2](https://openai.com/index/dall-e-2-extending-creativity/) 和 [Flamingo](https://arxiv.org/abs/2204.14198) 这样的突破代表了该领域的关键进展。
- **极简 Comfy 重写项目**：一位成员宣布了他们的实验性项目，旨在从头开始重写 Comfy，专注于创建一个没有依赖项的极简 UI 和服务器。
   - 他们邀请其他有兴趣创建可扩展解决方案的人联系他们进行协作。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/JulienBlanchon/status/1831719118434709868">Julien Blanchon (@JulienBlanchon) 的推文</a>: 正在尝试弄清楚如何修复 Comfy 👀</li><li><a href="https://huggingface.co/spaces/Tonic/Yi-Coder-9B">Yi Coder 9B - Tonic 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://www.lightly.ai/post/introduction-to-vision-language-models">Vision Language Models 简要介绍</a>: Vision Language Models 领域最新进展概述。从早期的对比学习方法（如 CLIP）到更先进的模型（如 Flamingo 和 LLaVA）。</li><li><a href="https://github.com/qompassai/KO">GitHub - qompassai/KO: Kyber Odyssey: 在后 Crowdstrike 世界中为安全创新指明方向</a>: Kyber Odyssey: Charting a course for secure innovation in a post-Crowdstrike world - qompassai/KO
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

quantaussie99: 得读读这个……我没看懂
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1280989801440481290)** (3 messages): 

> - `Tracking Algorithms for Multi-Object Tracking`
> - `Retrieving Screen Items from Internal Data`
> - `Running BLIP-2 on AWS SageMaker` 


- **讨论跟踪算法**：成员们提到了使用 **ByteTrack** 和 **DeepSORT** 等多种跟踪算法进行 Multi-Object Tracking。
   - 他们正在就这些选项的优缺点交换意见。
- **关于内部数据检索的问题**：一位成员提出了关于是否可以通过读取某些**内部数据**来检索屏幕上项目的问题。
   - 这引发了关于访问此类数据的可行性和方法的讨论。
- **关于在 AWS SageMaker 上运行 BLIP-2 的咨询**：一位成员就如何在 AWS SageMaker 上运行 **BLIP-2 模型**以对 **19,000 张图像**进行推理寻求建议。
   - 他们请求关于配置、实例类型、性能优化和集成步骤的建议。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1281039769001791612)** (2 messages): 

> - `Qwen2-VL-7B-Instruct`
> - `requirements.txt update`
> - `fp16 performance` 


- **Qwen2-VL-7B-Instruct Handler 已创建**：已分享适用于 **Qwen2-VL-7B-Instruct** 的可用 [handler.py](https://huggingface.co/hperkins/Qwen2-VL-7B-Instruct/tree/main) 和 requirements.txt，并确认可在 **T4 64GB**、**A100 80GB** 和 **L4 96GB** 等专用端点上运行。
   - 同时也链接了 commit，显示最近一次更新是在 **1 天前**。
- **Requirements.txt 正在更新**：记录了 requirements.txt 的一次更新，并提供了具体的 commit [链接](https://huggingface.co/hperkins/Qwen2-VL-7B-Instruct/commit/1dfb1806d850a7c85411b46ab1577310f8120324)供参考。
   - 此更新是确保项目兼容性和功能的持续维护工作的一部分。
- **fp16 实现缺乏 Flash-Attention**：目前该实现使用的是不带 **Flash-Attention** 的 **fp16**，这被视为一个局限性。
   - 这一情况已得到确认，并表示未来会有相关的增强功能。



**提到的链接**: <a href="https://huggingface.co/hperkins/Qwen2-VL-7B-Instruct/tree/main">hperkins/Qwen2-VL-7B-Instruct at main</a>: 未找到描述

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1281090719523799132)** (2 messages): 

> - `PixArt-Alpha performance`
> - `FluxImageToImagePipeline availability` 


- **PixArt-Alpha 展示了令人印象深刻的性能**：一位成员强调 **PixArt-Alpha** 表现出色，但未详细说明其具体功效或用例。
   - 这表明可能有一些值得进一步探索的显著特性。
- **diffusers 中缺失 FluxImageToImagePipeline**：一位成员询问为何 diffusers 中缺少 **FluxImageToImagePipeline**，尽管它出现在 HF 文档中。
   - 这引发了关于库中可能存在的差异或尚未同步更新的问题。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1280966639705391137)** (111 messages🔥🔥): 

> - `LM Studio 0.3.2 Issues` (LM Studio 0.3.2 问题)
> - `Image API Providers` (图像 API 提供商)
> - `Reflection 70B Model` (Reflection 70B 模型)
> - `Change in UI Elements` (UI 元素变更)
> - `Advanced Model Techniques` (高级模型技术)


- **LM Studio 0.3.2 下载错误**：用户报告在更新到 **LM Studio 0.3.2** 后遇到“unable to get local issuer certificate”错误，导致无法下载模型。
   - 据建议，这可能与公司网络安全设置的变更或影响软件连接性的 SSL 证书有关。
- **探索图像 API 选项**：一位用户表示有兴趣寻找具有高额度的免费图像 API 提供商，提到了 **Stable Diffusion** 但希望能有更多选择。
   - 他们询问是否有提供商为高级成像工具提供 API 访问权限。
- **Reflection 70B 模型讨论**：**Reflection 70B** 模型被强调为领先的开源 LLM，经过训练可以纠正其推理错误，目前已在 **Hugging Face** 上提供。
   - 在该模型最近上传后，人们对其何时能在 **LM Studio** 中使用充满了期待。
- **对新 UI 元素的担忧**：一些用户批评了 **LM Studio 0.3.2** 中的新 UI，认为元素过大且缺少预设下拉菜单，使用不便。
   - 反馈表明，用户希望在未来版本中看到更小的 UI 元素并恢复预设选项。
- **高级量化 AGI 模型**：关于 AI 未来的一项幽默预测指出，未来可能会围绕 **AGI 模型** 的量化展开一场争斗。
   - 用户对 AI 和模型技术的进步表示乐观。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/hello-hi-cute-kitten-cat-gif-6917710866304482943">Hello Hi GIF - Hello Hi Cute - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://releases.lmstudio.ai/windows/0.2.31/candidate/LM-Studio-0.2.31-Setup.exe">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-70B · Hugging Face</a>：未找到描述</li><li><a href="https://medium.com/@ianormy/microsoft-graphrag-with-an-rdf-knowledge-graph-part-1-00a354afdb09">Microsoft GraphRAG with an RDF Knowledge Graph — Part 1</a>：使用本地 LLM 和 Encoder 实现 Microsoft 的 GraphRAG
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1281064488715681923)** (60 messages🔥🔥): 

> - `Mac RAM and storage needs` (Mac RAM 和存储需求)
> - `Local server versus cloud options` (本地服务器与云端选项)
> - `Raspberry Pi and LMStudio compatibility` (Raspberry Pi 与 LMStudio 兼容性)
> - `Performance of RTX 3060 for inference` (RTX 3060 的推理性能)
> - `NAS advantages for Apple users` (NAS 对 Apple 用户的优势)


- **Mac 用户应为模型配置最大内存**：用户讨论认为，对于 Apple 硬件，应目标购买**最大的 RAM**，特别是为了处理大型模型。
   - *64GB* 被认为是 AI 严肃使用的最低配置，并建议投资 **NAS** 作为存储解决方案。
- **关于构建具备 AI 能力的本地服务器的辩论**：一些成员辩论是购买本地服务器还是使用云端选项来满足 AI 需求，强调了组装个人设备的**经济负担**。
   - 一位成员提到，与构建本地机器相比，**云端订阅**能以更低的成本提供更强大的能力。
- **Raspberry Pi 无法运行 LMStudio**：一位成员询问在 **Raspberry Pi** 上运行 LMStudio 的可行性，但已确认目前无法实现。
   - 讨论了 LMStudio 与 Ollama 之间的区别，强调了 Ollama 具有更广泛的硬件兼容性。
- **模型的 GPU 性能讨论**：一位使用 RTX 3060 的成员分享了对当前配置（**6GB VRAM** 和 **64GB DDR4 RAM**）增加上下文长度的担忧。
   - 其他人建议攒钱投资新 GPU，并强调硬件升级带来的**性能**提升至关重要。
- **NAS 设置对 Apple 用户的益处**：用户分享了使用 NAS 系统的经验，表示非常喜欢将存储从主桌面移出后带来的更好组织和效率。
   - 提到了特定的 **Asustor NAS**，以及将其用于多部 iPhone 的 **Time Machine** 备份的想法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.asustor.com/en/product?p_id=79">未找到标题</a>：未找到描述</li><li><a href="https://www.reddit.com/r/MacOS/comments/1ae3m3z/a_nas_that_actually_works_on_macos/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1280969268808257608)** (140 条消息🔥🔥): 

> - `Reflection-Tuning 技术`
> - `Hermes 模型推测`
> - `Fine-tuning LLMs`
> - `AI 模型的 Dataset 创建`
> - `Nvidia 驱动问题` 


- **Reflection-Tuning 创新**: 这种名为 [Reflection-Tuning](https://huggingface.co/mattshumer/Reflection-70B) 的新技术旨在通过教导 LLMs 在输出生成过程中纠正自己的错误并对响应进行反思，从而提高模型的能力。
   - 该方法强调使用一个特意包含错误的 Dataset，以辅助模型的自我纠正能力。
- **关于 Hermes 模型 Reflection 能力的讨论**: 在关于 Hermes 模型的讨论中，成员们推测原始训练数据可能不支持即时纠正，这对改进模型响应提出了挑战。
   - 关于如果文本中没有错误，Pretraining 如何解释即时错误的问题存在一些困惑，这引发了围绕 Fine-tuning 策略的更深层次讨论。
- **Fine-tuning 技术与 Datasets**: 参与者分享了 Fine-tuning 模型的方法，并表示希望看到 GPT-4o 和 Llama 70B 等不同模型之间的对比。
   - 有建议提出对包含 reflection tokens 和修订技术的模型进行 Fine-tuning，以增强输出评估。
- **Nvidia 驱动与 Vulkan 兼容性问题**: 用户在使 Vulkan 与其 Nvidia 驱动协同工作时遇到了问题，收到的消息要求使用 nouveau 驱动而非 Nvidia 官方驱动。
   - 参与者呼吁寻求在使用当前 Nvidia 配置的同时，如何开启 Vulkan 以获得更好性能的解决方案。
- **通用 AI 社区参与**: 包括计算机科学专业学生在内的参与者分享了入门 AI 领域的资源和建议，强调了实践与理论知识的重要性。
   - 社区对 AI 模型实验的协作努力感到兴奋，凸显了社区中积极主动的精神。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/mattshumer_/status/1831826171107144090?t=k5R0qg02Qr5azpPjQtfgaw&s=19">来自 Matt Shumer (@mattshumer_) 的推文</a>: @EnricoShippole @binary_racoon @GlaiveAI 不同的 reflection —— 只是不想引起任何混淆，我们正在做完全不同的事情</li><li><a href="https://openreview.net/forum?id=xaqoZZqkPU">Reflection-Tuning: Recycling Data for Better Instruction-Tuning</a>: 大语言模型 (LLMs) 的最新进展扩展了自然语言理解和生成的视野。值得注意的是，LLMs 的输出控制和与输入的对齐可以...</li><li><a href="https://x.com/mattshumer_/status/1831768677605155174">来自 Matt Shumer (@mattshumer_) 的推文</a>: @abacaj 不完全是 —— 我们发现目前的模型很难做好这一点（它们不知道什么时候该进行 reflect）。这需要通过一个特意制造错误的 Dataset 将其训练到模型中 ->...</li><li><a href="https://huggingface.co/matts">matts (Matt Szydlik)</a>: 未找到描述</li><li><a href="https://tenor.com/view/cat-cute-cat-yap-yapper-yapping-gif-5642199211123099306">Cat Cute Cat GIF - Cat Cute cat Yap - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/ZeyuanAllenZhu/status/1829326495757853005?t=VibYJ-3VXqmPmp9QWPYqSA&s=19">来自 Zeyuan Allen-Zhu (@ZeyuanAllenZhu) 的推文</a>: (1/7) Physics of LM, Part 2.2，包含关于 “LLM 如何从错误中学习” 的 8 个结果，现已发布在 arXiv: https://arxiv.org/abs/2408.16293。我们探索了使模型能够纠正错误的可能性...</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-70B · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/mattshumer_/status/1831767014341538166?t=ldUBdhhdmxU0qMgsmVaTUg&s=19">来自 Matt Shumer (@mattshumer_) 的推文</a>: 我很高兴地宣布 Reflection 70B，世界上顶级的开源模型。使用 Reflection-Tuning 训练，这是一种为使 LLMs 能够修复自身错误而开发的技术。405B 将于下周推出...</li><li><a href="https://github.com/tianyi-lab/Reflection_Tuning">GitHub - tianyi-lab/Reflection_Tuning: [ACL'24] Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning</a>: [ACL'24] Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning - tianyi-lab/Reflection_Tuning
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1281005764420440185)** (20 messages🔥): 

> - `Mamba API inquiries`
> - `Mergekit issues`
> - `Scaling and LLM reasoning`
> - `Llama 3.1 utilization`
> - `Open reasoning tasks` 


- **对 Mamba API 的好奇**：成员们询问 **Mamba API** 是否存在，并讨论了除 Google 和 Hugging Face 等常见平台之外的多种免费 API 替代方案。
   - *Arthrod* 特别询问了其他免费 API，并邀请社区提供建议。
- **对 Mergekit 停滞的挫败感**：一位成员报告称，在 Colab 中尝试合并两个微调后的 **Llama 3.1** 模型时，**Mergekit 始终停滞**在 'Executing graph: 0% 0/1457'。
   - 执行中断且未在 HF hub 仓库中创建可用的模型，导致用户困惑。
- **Scaling 与 LLM 推理查询**：一位成员就 **Scaling** 与 **Ilya** 为 AGI 筹集的 **10 亿美元** 资金之间的关系提出了疑问，以及这是否能真正提升 LLM 的推理能力。
   - *Kingsd* 寻求可能在该领域投入大量时间研究的其他人的见解，以求明确。
- **Llama 3.1 在交易中的实际应用**：一位用户分享了使用 **Llama.cpp** 作为交易基础设施推理引擎的经验，特别提到了使用 **mistral-7B-instruct-v0.2.Q6_K.gguf** 进行编码查询。
   - 他们收到了在资源允许的情况下使用 **Llama 3.1 8B Instruct** 的建议，并讨论了 GPU 规格。
- **获取开放推理任务资源**：一位成员询问专注于 **推理任务** 的数据集，并被引导至一个开放推理任务项目，该项目列出了用于训练或评估的潜在任务类型。
   - 该项目本身不是一个数据集，但鼓励参与者根据建议的任务开发自己的数据集。



**Link mentioned**: <a href="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/tree/main">bartowski/Meta-Llama-3.1-8B-Instruct-GGUF at main</a>: no description found

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1281006674156453998)** (2 messages): 

> - `Falcon Mamba release`
> - `Loopy video diffusion model` 


- **TII 发布 Falcon Mamba**：[Falcon Mamba](https://falconllm.tii.ae/tii-releases-first-sslm-with-falcon-mamba-7b.html) 是由 [Technology Innovation Institute](https://www.tii.ae/ai-and-digital-science) 开发的新模型，已根据 **TII Falcon Mamba 7B License 1.0** 发布，可在 [Hugging Face](https://huggingface.co/tiiuae/falcon-mamba-7b) 上公开获取。博客详细介绍了设计决策、该模型相对于 SoTA 模型的竞争优势，以及其在 Hugging Face 生态系统中的集成。
- **用于纯音频视频生成的创新模型 Loopy**：论文介绍了 **Loopy**，这是一种端到端的音频条件视频扩散模型，无需手动空间模板即可增强自然运动和肖像合成。该模型采用独特的片段间和片段内时间模块，以更好地将音频与人体运动关联，从而提高视频生成的整体性能。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/papers/2409.02634">Paper page - Loopy: Taming Audio-Driven Portrait Avatar with Long-Term Motion
  Dependency</a>: no description found</li><li><a href="https://huggingface.co/blog/falconmamba">Welcome Falcon Mamba: The first strong attention-free 7B model</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

adjectiveallison: https://github.com/Cognitive-AI-Systems/MAPF-GPT
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1281006674156453998)** (2 messages): 

> - `Falcon Mamba 模型`
> - `Loopy 视频扩散模型` 


- **TII 推出 Falcon Mamba**：[Falcon Mamba](https://falconllm.tii.ae/tii-releases-first-sslm-with-falcon-mamba-7b.html) 是阿布扎比技术创新研究院（TII）在 TII Falcon Mamba 7B License 1.0 许可下发布的新模型，旨在 Hugging Face 生态系统中提供开放访问。
   - 博客讨论了该模型的架构设计决策及其与现有 SoTA 模型的竞争力，并强调该模型可用于研究和应用目的，详见 [此处](https://huggingface.co/tiiuae/falcon-mamba-7b)。
- **Loopy：纯音频驱动视频生成的突破**：该论文介绍了 **Loopy**，这是一种端到端的纯音频驱动视频扩散模型，通过利用长期运动信息，克服了通过音频信号控制人物动作的局限性。
   - Loopy 通过消除对手动指定空间运动模板的需求，改善了**音频-肖像动作的相关性**，在广泛的实验中展示了在自然运动合成和细节方面的**显著进步**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/falconmamba">欢迎 Falcon Mamba：首个强大的无 Attention 机制的 7B 模型</a>：未找到描述</li><li><a href="https://huggingface.co/papers/2409.02634">论文页面 - Loopy: Taming Audio-Driven Portrait Avatar with Long-Term Motion
  Dependency</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1281159974600314910)** (1 messages): 

> - `Ilya 为 AGI 筹资`
> - `Scaling 与 LLM 推理` 


- **Ilya 为 Safe Superintelligence 筹集 10 亿美元**：Ilya 为其新创公司 **Safe Superintelligence** 成功筹集了 **10 亿美元**，该公司致力于通过 Scaling 努力实现 **AGI**。
   - 成员们对 Scaling 是否能有效解决与 **LLM 推理** 相关的问题表示了**困惑**。
- **质疑 Scaling 对 LLM 的影响**：一位成员质疑 **Scaling** 是否真正解决了 **Large Language Models (LLMs)** 的推理能力及其运作方式。
   - 他们询问小组中是否有人认真投入时间研究过这一课题。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1280999611674263563)** (11 messages🔥): 

> - `xAI's GPU Cluster`
> - `Unsloth backed by YCombinator`
> - `Reflection Llama-3.1`
> - `Intrinsic-self correction technique` 


- **xAI 的 GPU Cluster 算力引发关注**：Elon Musk 在构建 xAI 的 **100k GPU Cluster** 方面取得的进展引起了竞争对手模型开发者的担忧，*OpenAI 的 Sam Altman 对潜在的算力差距表达了忧虑*。
   - 一位成员幽默地评论道，*最终我们都会变得 GPU poor*。
- **Unsloth 与 YCombinator 合作**：Unsloth 宣布获得 **YCombinator** 的支持，旨在通过软件创新为模型创建者打造一个专注于速度和易用性的全方位解决方案。
   - 他们使用 **Triton** 和 **CUDA** 等底层语言，并邀请感兴趣的人士加入他们的 [waitlist](https://unsloth.ai/waitlist) 并查看其 [roadmap](https://unsloth.ai/roadmap-yc)。
- **Reflection Llama-3.1 被誉为顶尖开源 LLM**：**Reflection Llama-3.1 70B** 被强调为全球领先的开源 LLM，它采用了一种名为 **Reflection-Tuning** 的新技术来提高推理准确性。
   - 该模型由 [Glaive](https://glaive.ai) 使用合成数据训练，可以在[此处](https://reflection-playground-production.up.railway.app/)进行试用。
- **关于 intrinsic-self correction 的讨论**：有人对在没有外部工具的情况下 **intrinsic-self correction** 的有效性表示怀疑，并引用了常见的 **GDM 论文**。
   - 一位用户对这种方法表示惊讶，并质疑其在最近讨论的 Reflection Tuning 背景下的可行性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/xDaily/status/1831405867641802834">来自 X Daily News (@xDaily) 的推文</a>：新闻：Elon 在构建 xAI 的 100k GPU Cluster 方面的进展让一些竞争对手模型开发者感到担忧。例如，OpenAI CEO Sam Altman 告诉一些 Microsoft 高管，他担心...</li><li><a href="https://huggingface.co/mattshumer/Reflection-70B">mattshumer/Reflection-70B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/UnslothAI/status/1831715700031025455">来自 Unsloth AI (@UnslothAI) 的推文</a>：我们很高兴地宣布 Unsloth 现在得到了 @YCombinator 的支持！基于我们在开源微调方面的基础，我们正在创建全方位的解决方案，以便您可以专注于制作模型...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1281300417598459935)** (6 messages): 

> - `Reasoning Datasets`
> - `HuggingFace Numina`
> - `MATH Benchmark`
> - `GSM8k Benchmark`
> - `CHAMP Dataset` 


- **推理数据集的热门选择**：一位成员寻求关于 **推理数据集/基准测试 (reasoning datasets/benchmarks)** 的建议，特别是那些包含 **思维链 (chain-of-thought) 推理轨迹** 的数据集。
   - 另一位成员幽默地指出选项非常丰富，暗示他们被众多的选择淹没了。
- **HuggingFace 的 Numina 备受关注**：一位参与者推荐了最近的 **HuggingFace Numina** 资源，认为它是推理任务数据的绝佳来源。
   - 这被视为对该领域感兴趣的人来说，基准测试库中的一个宝贵补充。
- **标准基准测试：MATH 和 GSM8k**：当被问及值得注意的基准测试时，几位成员指出 **MATH** 和 **GSM8k** 是推理评估中的标准参考。
   - 这些基准测试经常被用于评估大型语言模型 (LLM) 的推理能力。
- **CHAMP 数据集提供独特见解**：一位成员强调了 **CHAMP** 数据集，该数据集专注于带有注释提示的高中数学问题，为推理任务提供了额外的上下文。
   - 该基准测试旨在研究特定问题的提示和概念对 LLM 性能的影响，详见 [论文](https://arxiv.org/abs/2401.06961)。
- **寻找冷门见解**：发帖者表示，在 **HuggingFace** 上为一个研究项目搜寻时，希望能找到一些较少人知的推理数据集。
   - 他们对那些在讨论中不常被提及的数据集特别感兴趣。



**提到的链接**：<a href="https://arxiv.org/abs/2401.06961">CHAMP: A Competition-level Dataset for Fine-Grained Analyses of LLMs&#39; Mathematical Reasoning Capabilities</a>：最近的大型语言模型 (LLM) 在具有挑战性的竞赛级问题上显示出数学推理能力的迹象，特别是在自生成的中间推理过程方面...

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1280985733594742846)** (42 条消息🔥): 

> - `Cursor 聊天文档`
> - `QwenLM GitHub 消失`
> - `OpenAI 模型命名混淆`
> - `Llama 微调供应商`
> - `Artificial Analysis 与新图像模型` 


- **Cursor 聊天文档受到关注**：讨论强调了在软件开发中缺乏用于记录 AI 交互的标准化 `chats.txt` 文件，并重点讨论了 Cursor 如何使其变得有用。
   - 成员们对行业内**缺乏此类标准感到震惊**，认为这可以显著增强代码库的文档化。
- **QwenLM 从 GitHub 神秘消失**：随着 QwenLM 组织从 GitHub 消失，引发了担忧，人们猜测该平台可能触发了某些未知标记。
   - 成员们对 GitHub 缺乏沟通表示难以置信，认为类似的往事非常**荒谬**。
- **对 OpenAI 模型名称的混淆**：关于两个不同模型 **GPT-4o-latest** 和 **GPT-4o-2024-08-06** 存在混淆，尽管命名方案相似，但它们并非同一个模型。
   - 成员们幽默地指出 OpenAI 的命名策略让许多人感到困惑，有人开玩笑说 *Scale* 也被它绊倒了。
- **寻求 Llama 微调建议**：一位成员询问了微调 Llama 模型的首选供应商，有人建议聘请一名有能力的工程师。
   - 回复中提到了 Fireworks 和 Together 等公司，指出它们还不错，但并非 100% 可靠。
- **关于即将推出的图像模型的讨论**：一位参与者询问是否有组织准备发布新的图像模型，特别提到 **Saturn-Next** 是一个很有前景的候选者。
   - 推测包括这些模型可能仅限于 Artificial Analysis，这与 Midjourney 预期的更新形成对比。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://llmstxt.org/">The /llms.txt file – llms-txt</a>：一项关于标准化使用 /llms.txt 文件的提案，旨在提供信息以帮助 LLM 在推理时使用网站。</li><li><a href="https://forum.cursor.com/t/how-do-i-export-chat-with-ai/144/13">如何导出与 AI 的聊天记录？</a>：我认为这是一个非常有用的功能。我最近与我的朋友 Sonnet 进行了很长时间的互动，我很想导出它，以便对其进行格式化，删除所有不必要的部分，并将其保存为 ...</li><li><a href="https://x.com/simonw/status/1831392171850969456?s=46">来自 Simon Willison (@simonw) 的推文</a>：@Lingster888 这太奇怪了，他们的整个 GitHub 组织都消失了 https://github.com/qwenlm。昨天还在，这是几天前的网页存档 https://web.archive.org/...</li><li><a href="https://fxtwitter.com/thexeophon/status/1831678356745597031?s=46">来自 Xeophon (@TheXeophon) 的推文</a>：哎呀，Scale 也被 OpenAI 惊人的命名方案给绊倒了。GPT-4o-latest 和 GPT-4o-2024-08-06 是两个不同的模型 🙃 引用 Alexandr Wang (@alexandr_wang) 的 SEAL 排行榜更新...</li><li><a href="https://x.com/justinlin610/status/1831489518467477529?s=46">来自 Junyang Lin (@JustinLin610) 的推文</a>：我们还活着... GitHub 出于未知原因标记了我们的组织，我们正在尝试联系他们寻求解决方案。引用 Simon Willison (@simonw) 询问为什么 @Alibaba_Qwen AI 团队...</li><li><a href="https://fxtwitter.com/phill__1/status/1831405607641059588">来自 Phil (@phill__1) 的推文</a>：等等，Artificial Analysis 图像竞技场中是否有隐身模型？似乎有三个新模型正在测试中，其中 Saturn-Next 表现非常出色。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1281223504288681984)** (74 messages🔥🔥): 

> - `AI 中的 Autoformalization`
> - `2026 年前实现超人类 AI 数学家`
> - `OpenAI 的定价策略`
> - `Google 在 AI 部署方面的挑战`
> - `SnailBot 的性能` 


- **Autoformalization 作为关键策略**：一位成员强调，**Autoformalization**（自动形式化）对于 AI 的进步至关重要，特别是在大实验室已经暗示的合成数据机制背景下。
   - 他们指出 **Google** 正在积极追求这一领域，表明了市场中的竞争压力。
- **Szegedy 预测 2026 年实现超人类 AI**：Christian Szegedy 表示，他现在相信到 2026 年我们将拥有**超人类 AI 数学家**，这与他之前预测的 2029 年相比有了重大提前。
   - 他的断言引发了关于该目标可行性的辩论，特别是关于数学证明所需的非正式推理（informal reasoning）。
- **OpenAI 潜在的高昂定价**：有报告称 OpenAI 可能会考虑为新模型提供高达**每月 2,000 美元**的订阅服务，许多人认为考虑到市场竞争，这可能并不现实。
   - 成员们推测 **B2B 定价** 可能更容易被接受，但质疑家庭如何能为消费级 AI 支付此类成本。
- **Google 在 AI 策略中挣扎**：讨论强调了 Google 在有效部署其 AI 框架方面的持续困难，**Vertex AI** 的易用性受到了批评。
   - 尽管拥有顶尖工程师，该组织似乎在执行力上表现不佳，引发了对其在 AI 领域领导地位的担忧。
- **SnailBot 的古怪之处**：一位成员幽默地称 **SnailBot** 为有史以来编写的最慢的 Rust 程序，强调了它的娱乐性质。
   - 尽管有这些古怪之处，大家仍认为 **SnailBot** 依然是社区中一个免费且有趣的补充。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Ch">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/btibor91/status/1831705162349494551?s=46">来自 Tibor Blaho (@btibor91) 的推文</a>：据报道，OpenAI 正在考虑为专注于推理的 Strawberry 和旗舰级 Orion LLM 等新 AI 模型提供高达每月 2,000 美元的高价订阅（尽管最终价格可能会更低）...</li><li><a href="https://x.com/ChrSzegedy/status/1831330997239255186).">来自 Christian Szegedy (@ChrSzegedy) 的推文</a>：在过去的八年里，我对这一点的看法没有改变：我在 2019 年做过多次演讲并撰写了立场论文（链接见下文）。进展比我过去的预期要快。我之前的目标日期是...
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1281300926979903509)** (1 messages): 

> - `Magic 包管理器`
> - `MAX 与 Mojo 集成`
> - `Conda 生态系统`
> - `虚拟环境管理` 


- **Magic 🪄 正式掌舵**：今天我们宣布，**Magic** 成为 **MAX** 和 **Mojo** 项目新的官方包管理器和虚拟环境管理器，相关包以单个 Conda 包 `max` 的形式提供。
   - 从本周一开始，由于 `modular` CLI 将不再接收更新，鼓励用户**迁移**到 Magic 或其他支持 Conda 包管理的工具。
- **与 Conda 生态系统的无缝集成**：选择采用 **Conda 生态系统** 作为标准，旨在增强与流行包管理系统的兼容性，在减少冲突的同时提高代码的可复现性。
   - 使用 **Magic**，你可以立即启动代码示例并创建新项目，确保管理依赖项的体验更加精简。
- **告别打包冲突**：管理包依赖和**虚拟环境**对于稳定性和兼容性至关重要，**Magic** 有效地解决了这一挑战。
   - 目前 `magic` 的稳定版本为 **0.2.3**，为 Modular 流水线带来了特定的改进，并为未来的管理和部署提供了增强功能。
- **查看新的 Magic 文档**：有关开始使用 **Magic** 的更多信息，用户可以访问我们新的 [magic 文档页面](https://docs.modular.com/magic/)。
   - **Magic** 构建于 Conda 和 PyPi 生态系统之上，提供了访问数千个包的权限，以及为 MAX 和 Mojo 项目量身定制的附加功能。
- **感谢社区支持与反馈**：对社区在过渡期间提供的反馈和支持表示由衷感谢。
   - 鼓励用户在指定频道 <#1267269207372988597> 分享他们的问题和反馈。



**提到的链接**：<a href="https://docs.modular.com/magic/">开始使用 Magic | Modular 文档</a>：Magic 是用于 MAX 和 Mojo 的包管理器和虚拟环境管理器。

  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1281046521474842767)** (117 messages🔥🔥): 

> - `Mojo performance`
> - `Async function support`
> - `Memory management in Mojo`
> - `Mojo standard library enhancements`
> - `Compiler and debugging tools` 


- **Mojo 中 ord() 函数的性能问题**：一位用户注意到在 Mojo 中使用 **ord()** 与 C++ 和 Python 相比存在显著的性能差异，指出在基准测试场景中它大约**慢了 30 倍**。
   - 讨论包括建议使用调试器检查 **ord** 的实现，并推测了诸如 Small String Optimization（小字符串优化）之类的优化方案。
- **Mojo 中异步函数的问题**：在 Mojo 中尝试使用 **async fn** 和 **async def** 导致了各种错误，这主要归因于用户运行的是稳定版（stable build）而非支持异步特性的 nightly 版本。
   - 会议明确了将 **fn main** 标记为 **async** 可能尚不受支持，这表明了该语言目前的局限性。
- **Mojo 中的内存管理与借用 (Borrowing)**：对话集中在如何使用 **Arc** 和 **Weak** 等结构处理对象的局部借用，并考虑了由此产生的开销。
   - 提出了一种替代方法，即可能通过一个单独的类型来实现弱引用（weak references），并讨论了使用 Omit 处理可选字段。
- **利用 Mojo 的调试工具**：有建议提出在 Mojo 中使用编译技巧来获取汇编输出，以帮助理解生成的代码并辅助调试工作。
   - 还讨论了创建一个支持 MLIR 的 Mojo 编译器浏览器（compiler explorer）的可能性，并强调了其在教育方面的益处。
- **Mojo 库的增强与特性**：讨论包括在标准库中添加 **Omit** 类型的可能性，这可以避免与未使用字段相关的开销。
   - 讨论了对类型和构造函数的改进和细化，以在不牺牲代码效率的情况下确保功能性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/roadmap">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>：Mojo 计划摘要，包括即将推出的特性和需要修复的问题。</li><li><a href="https://docs.modular.com/mojo/roadmap#calling-mojo-from-python">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>：Mojo 计划摘要，包括即将推出的特性和需要修复的问题。</li><li><a href="https://docs.modular.com/max/roadmap">Roadmap &amp; known issues | Modular Docs</a>：MAX 平台已知问题和即将推出的特性的摘要。</li><li><a href="https://docs.google.com/presentation/d/1vkM05Ld8nEfLalxSuWmjDv_wfYQ9IVb7HbQ07MR3Xxs/edit?usp=drivesdk">Small string optimization in Mojo’s stdlib</a>：Mojo 标准库中的小字符串优化，以及顺便提到的小缓冲区优化。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/string.mojo#L32">mojo/stdlib/src/builtin/string.mojo at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1280967560514506843)** (8 messages🔥): 

> - `Model Serialization Format`
> - `Containerization Techniques`
> - `MAX Engine Support` 


- **等待模型序列化格式**：一位用户询问了**平台无关模型格式的预计发布时间 (ETA)**，但回复指出目前该模型序列化格式没有确切的 ETA，因为它更多属于一项功能增强。
   - 反馈表达了对这一即将推出的特性的期待，该特性旨在辅助容器化，但也强调了平台无关性并非核心需求。
- **寻求容器化见解**：用户询问了**推荐的容器化方法**，表示有兴趣在 Docker 容器中部署模型，并提到了使用 **tvm** 等其他工具时遇到的问题。
   - 回复强调模型序列化将促进 Docker 容器化，并希望在一个月内发布。
- **MAX Engine 与 GGUF**：明确了 MAX Engine 不支持 **gguf**，可以参考提供的 GitHub 链接了解替代流水线。
   - 这为探索类似功能或寻求 MAX Engine 替代方案的用户提供了背景信息。



**提到的链接**：<a href="https://github.com/modularml/max/tree/main/examples/graph-api">max/examples/graph-api at main · modularml/max</a>：示例程序、笔记本和工具的集合，展示了 MAX 平台的强大功能 - modularml/max

  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1281185503923671062)** (3 条消息): 

> - `Bank Account Expansion` (银行账户扩张)
> - `Infinite Dilution Concept` (无限稀释概念)


- **无限银行账户概念**：一位成员幽默地表达了希望将他们的*银行账户压缩成无限金额*的愿望。
   - 这一机智的请求引发了关于财务限制和可能性的讨论。
- **关于扩张与压缩的困惑**：另一位成员质疑*压缩成无限金额*是否实际上意味着扩张。
   - 这引发了一个发人深省的时刻，促使人们对财务概念进行更深层次的思考。
- **无限扩张的危险**：一位成员提出了一个重要观点，指出*如果你无限扩张某样东西*，你可能会将其稀释到虚无。
   - 这一评论警告了在金融等背景下追求无限数量的潜在负面影响。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1280965782033141890)** (91 条消息🔥🔥): 

> - `Opus vs Sonnet Performance` (Opus 与 Sonnet 性能对比)
> - `DeepSeek V2.5 Release` (DeepSeek V2.5 发布)
> - `Reflection 70B Announcement` (Reflection 70B 发布公告)
> - `Claude Caching Feature` (Claude 缓存功能)
> - `Model Throughput Comparisons` (模型吞吐量对比)


- **Opus 声称在特定任务上的性能优于 Sonnet**：一位成员指出 Opus 在特定提示词（如计算数字时钟显示屏上的角度）上的表现优于 Sonnet。
   - 相反，其他人认为大多数基准测试一致显示 **Sonnet** 在整体上更胜一筹。
- **DeepSeek V2.5 模型发布**：DeepSeek 已将其 **Coder** 和 **Chat** 模型合并并升级为新的 V2.5 版本，在各项性能指标上都有显著提升。
   - 例如，**ArenaHard 胜率**从 **68.3% 提升至 76.3%**，增强了通用能力和指令遵循（instruction following）能力。
- **对 Reflection 70B 模型的兴奋**：新的 **Reflection 70B** 模型已发布，号称通过一种名为 **Reflection-Tuning** 的技术具备自我修正能力。
   - 随着下周将推出 **405B** 版本的承诺，社区预期其表现将超越现有模型。
- **关于 Claude 上下文缓存的疑问**：有关于 **Claude** 模型中上下文缓存（context caching）可用性的咨询，一些成员分享了关于速率限制（rate limits）和成本的经验。
   - 据透露，目前的情况还不允许通过缓存降低价格，尽管预计未来会有实施计划。
- **对模型吞吐量的担忧**：尽管新的 V2.5 模型有所进步，但有人担心 DeepSeek 模型的吞吐量（throughput）低于 **Sonnet 3.5**。
   - 一些成员评论说，虽然该模型非常适合个人使用，但其较慢的性能给生产环境用例带来了挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://nian.llmonpy.ai/intro">GPT-4o’s Memory Breakthrough! (NIAN code)</a>: 未找到描述</li><li><a href="https://x.com/mattshumer_/status/1831767014341538166?s=46&t=2a7uDiV3mox9o-E5jIFbLQ">Matt Shumer (@mattshumer_) 的推文</a>: 我很高兴宣布 Reflection 70B，全球顶尖的开源模型。使用 Reflection-Tuning 训练，这是一种为使 LLM 能够修复自身错误而开发的技术。405B 将于下周推出...</li><li><a href="https://x.com/mattshumer_/status/1831767014341538166?t=MKrJQ-X4VjS_MpTLpP4jDg&s=19">Matt Shumer (@mattshumer_) 的推文</a>: 我很高兴宣布 Reflection 70B，全球顶尖的开源模型。使用 Reflection-Tuning 训练，这是一种为使 LLM 能够修复自身错误而开发的技术。405B 将于下周推出...</li><li><a href="https://platform.deepseek.com/api-docs/news/news0802/">DeepSeek API 推出磁盘上下文缓存，将价格降低了一个数量级 | DeepSeek API 文档</a>: 在大语言模型 API 使用中，很大一部分用户输入往往是重复的。例如，用户提示词经常包含重复的引用，而在多轮对话中，之前的...</li><li><a href="https://platform.deepseek.com/api-docs/updates">更新日志 | DeepSeek API 文档</a>: 版本: 2024-09-05
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1281023122786484329)** (5 messages): 

> - `AI Studio key 问题`
> - `Bug 报告`
> - `Activity 日志记录` 


- **AI Studio key 无法保存配置**：输入 **AI Studio key** 时，页面显示更新成功，但在输入后又恢复为 **Not Configured**。
   - *Daun.ai* 将此确定为潜在的 **bug**，并正在进行修复。
- **Hyperbolic 和 Lambda key 运行正常**：尽管 **AI Studio key** 存在问题，但据报告 **Hyperbolic** 和 **Lambda** key 均可正常使用。
   - 用户对不同 key 之间表现不一致的情况表示担忧。
- **提出 Activity 日志记录相关问题**：一位用户询问是否可以在 **Activity** 下验证 **AI Studio key** 是否已被使用。
   - 这引发了关于用户如何有效监控其 key 使用情况的疑问。


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1280967921161732190)** (77 messages🔥🔥): 

> - `Perplexity 订阅优惠`
> - `推荐计划详情`
> - `会员变更`
> - `面向学生的周边商品促销`
> - `技术支持与咨询` 


- **Perplexity 面向学生的年度会员**：Perplexity 宣布，对于达到 **500** 名使用 `.edu` 邮箱注册学生的大学，将提供免费的 **1 年 Pro 会员**资格，这引发了关于资格和注册标准的讨论。
   - 用户讨论了需要在特定日期前注册，部分用户对其大学的状态表示不确定。
- **关于推荐链接的说明**：成员询问如何查找其 **affiliate 推荐链接**并分享会员福利，其中一位提到特定 **URL** 可以提供访问权限。
   - 关于唯一 promo code 可以使用多少次出现了困惑，随后澄清其最多可使用 **8 次**。
- **学生推荐的周边促销**：分享了关于可通过推荐获得 **面向学生的新周边商品** 的公告，鼓励成员参与分享链接。
   - 提供了关于如何通过推荐朋友使用 Perplexity 来获得这些促销的具体说明。
- **语言设置的技术问题**：用户遇到了语言设置在不同浏览器中无法正确应用的问题，一名成员通过切换选项成功解决了该问题。
   - 解决方案表明，切换到另一种语言再切换回来可以解决显示问题。
- **关于免费 Perplexity 访问的咨询**：存在关于学生获取免费 Perplexity 功能的问题，特别是与推荐和大学注册人数挂钩的部分。
   - 成员对订阅到期以及解锁延长访问权限的必要条件表示担忧。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1831762895220383807?s=61">来自 Perplexity (@perplexity_ai) 的推文</a>: 学生新周边 🔜 获取方式之一：推荐你的朋友使用 Perplexity！分享越多，收获越多：http://perplexity.ai/backtoschool</li><li><a href="https://x.com/perplexity_ai/status/1831469659067195613?s=46">来自 Perplexity (@perplexity_ai) 的推文</a>: 从未如此出色。感谢 @tryramp 的专题报道！引用 Aravind Srinivas (@AravSrinivas)：今天的时代广场被点亮了
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1281000431870279751)** (10 messages🔥): 

> - `全球最强大的超级计算机`
> - `冷水澡的好处`
> - `大脑中的记忆存储`
> - `已知最古老的棋盘游戏`
> - `Dark Souls 的创新` 


- **探索 xAI 的 Colossus 超级计算机**：Perplexity AI 重点介绍了**全球最强大的超级计算机**——xAI 的 Colossus，以及**已知最古老的棋盘游戏** Senet。
   - 您可以在此 [YouTube 视频](https://www.youtube.com/embed/kb_DJSrHOy4)中了解更多关于这一惊人发现的信息。
- **冷水澡带来益处**：多位成员分享了讨论[冷水澡好处](https://www.perplexity.ai/search/benefits-of-cold-showers-hMZf7v0AR1KmXfwENQ_xag)的链接，展示了各种健康优势。
   - 这些益处包括改善血液循环和提升情绪，使其成为热门讨论话题。
- **大脑的记忆存储机制**：有一个关于大脑如何以 [triplets](https://www.perplexity.ai/page/brain-stores-memories-in-tripl-SYcH2HZjQH6FQyly7e8keA)（三元组）形式**存储记忆**的有趣引用，这是一个引人入胜的研究领域。
   - 它详细阐述了记忆之间的联系以及它们如何在我们的脑中形成复杂的网络。
- **Dark Souls 的创新**：对话涉及了 **Dark Souls** 游戏的最新创新，引发了对其机制和设计的询问。
   - 一位成员在[相关讨论](https://www.perplexity.ai/search/in-dark-souls-1-what-are-some-upmbYYY3QeaWQ0SJjZxJxA)中寻求了解更多关于这些创新的信息。
- **Perplexity 用户界面更新**：一位成员收到提醒，将其 Thread 设置为 **Shareable**（可共享），以增强社区内的协作。
   - 这强调了在提高讨论中的用户参与度和可访问性方面所做的持续努力。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1280999067903725709)** (2 messages): 

> - `使用 Perplexity API 实现文件上传`
> - `配置 Perplexity API 请求` 


- **在 Flask 中集成 Perplexity API 文件上传**：一位成员分享了在 Python Flask 应用中使用 **Perplexity API** 实现文件上传的方法，详细说明了该实现的客户端和服务器端组件。
   - 核心功能包括修改 **/query** 路由以接收文件数据，并将文件内容整合到发送给 API 的 Prompt 中。
- **从 Perplexity API 获取高质量响应**：一位用户询问如何配置其 **Perplexity API** 请求，以复制 Perplexity 官网回答的质量和风格。
   - 虽然未提供具体细节，但他们正在寻找基于现有参考模型来提升 API 响应质量的方法。


  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1280996350770806814)** (36 messages🔥): 

> - `Cursor 游戏规则改变者`
> - `AI 编程工具`
> - `vLLM 开放办公时间`
> - `Reflection 70B 发布公告`
> - `利用 AI 的 SaaS 初创公司` 


- **Cursor 变革者评价褒贬不一**：几位成员对 **Cursor** AI 工具表示怀疑，其中一人表示发现它没什么用，甚至将其归结为“技术水平问题”（skill issue）。
   - 另一位成员称赞了它的代码检索能力，但最终认为与免费版相比，它不值得投资。
- **对过度依赖 AI 编程助手的担忧**：讨论了使用 AI 编程工具可能带来的负面影响，一些人担心这种依赖会导致“大脑退化”（brainrot）。
   - 正如一位成员所说，“真的有人尝试用它来处理工单（tickets）吗？”，表达了对其有效性的怀疑。
- **vLLM 开放办公时间提供见解**：vLLM 团队每两周举办一次开放办公时间，今天的会议重点是用于高性能推理的 **NVIDIA CUTLASS**。
   - 参与者可以期待在即将发布的版本中获得巨大的性能提升，录音可在 [YouTube](https://www.youtube.com/watch?v=ZlIr_QsXqOM) 上观看。
- **Reflection 70B：开源 LLM 的新里程碑**：推出了一款名为 **Reflection 70B** 的新模型，它是领先的开源 LLM，使用 **Reflection-Tuning** 技术进行训练以实现自我纠错。
   - 随后将在下周推出 **405B** 模型，该模型被吹捧为全球最强，是与 **GlaiveAI** 共同开发的。
- **SaaS 初创公司与 AI 工具**：成员们讨论了 **SaaS 初创公司** 声称通过 AI 工具提升效率的趋势，尽管仍存在怀疑的声音。
   - 有人指出，社交媒体上的励志内容往往过度简化了这些技术的潜在优势。



**提到的链接**：<a href="https://x.com/mattshumer_/status/1831767014341538166">Matt Shumer (@mattshumer_) 的推文</a>：我很高兴宣布 Reflection 70B，全球顶级的开源模型。采用 Reflection-Tuning 训练，这是一种为使 LLM 能够修复自身错误而开发的技术。405B 将于下周推出...

  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1281345758913626255)** (6 messages): 

> - `MLIR_DEBUGGING`
> - `Triton 环境变量` 


- **启用 MLIR Dump 进行调试**：一位用户建议使用 `MLIR_ENABLE_DUMP=1` 来输出每个编译器 Pass 后的 MLIR，这有助于理解 Triton 的底层编译方式。
   - 他们表示可以对比两个 Dump 文件进行有效调试，并指出 LLM 可以协助更好地解释 MLIR。
- **利用 TRITON_INTERPRET 进行增强调试**：另一位成员强调，设置 `TRITON_INTERPRET=1` 是 Triton 中可用的最佳调试工具之一。
   - 该变量在调试过程中提供了宝贵的见解。
- **参考 README 获取调试变量**：一位用户建议参考之前链接的 README，其中包含许多用于调试 Triton 的有用环境变量。
   - 他们提到，虽然大多数变量可能不是必需的，但某些变量对于解决复杂问题至关重要。



**提到的链接**：<a href="https://github.com/triton-lang/triton/tree/7480ef5028b724cb434b7841b016c6d6debf3b84?tab=readme-ov-file#tips-for-hacking">GitHub - triton-lang/triton at 7480ef5028b724cb434b7841b016c6d6debf3b84</a>：Triton 语言和编译器的开发仓库 - GitHub - triton-lang/triton at 7480ef5028b724cb434b7841b016c6d6debf3b84

  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1281340616894058506)** (2 条消息): 

> - `卷积优化技术`
> - `CUDA 中的内存访问模式` 


- **使用 Constant Memory 优化卷积**：一位成员报告称，将 Constant Memory 用于卷积矩阵使执行时间从 **850 ms 降低到 705 ms**，但预期的寄存器数量为 **19**，而实际观察到的是 **20**。
   - 他们询问了为什么寄存器数量没有进一步下降，并表示需要对优化过程进行更清晰的说明。
- **Local Memory 意想不到的影响**：将 Local Memory 用于卷积矩阵导致运行时间从 **850 ms 减少到 702 ms**，这与预期相反，且每个线程的寄存器数量降至 **19**。
   - 该成员询问了为什么使用 Local Memory 会导致更低的 Constant Load，从而引发了关于 Local Memory 与 Global Memory 影响的讨论。
- **Local Memory 的编译器行为**：另一位成员解释说，编译器可能无法将 Local Memory 放入寄存器中，并且当涉及动态寻址时，Local Memory 会变成交错的 Global Memory。
   - 他们提供了 NVIDIA 关于 Local Memory 文档的链接，以引导对内存访问模式的进一步理解。


  

---


### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1281260496443412522)** (1 条消息): 

> - `Pallas kernel`
> - `Splash Attention kernel`
> - `Pallas 视频入门指南` 


- **探索 JAX 的 Pallas Kernel**：成员们正在分享各种**用 Pallas 实现的 kernel**，可以在[这个 GitHub 仓库](https://github.com/google/jax/tree/main/jax/experimental/pallas/ops/tpu)中找到。该仓库展示了 Python+NumPy 程序的组合式变换，包括微分和 JIT 到 GPU/TPU。
   - 同时也包含了一张仓库的图片，为贡献者提供视觉参考。
- **深入研究 Splash Attention Kernel**：分享的一个具体 kernel 示例是 **Splash Attention kernel**，其实现可以在[这里](https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py)找到。这直接链接到代码，突出了 **Pallas** 操作的重要组成部分。
   - 鼓励成员查看 kernel 的细节，以更好地理解其在 Pallas 框架中的功能。
- **查看 Pallas 视频入门指南**：通过[此链接](https://youtu.be/liKrhX2gm44?si=QX_xZKD_oentvMiV)分享了一段关于 **Pallas** 的**简短入门视频**，由其主要发明者之一 **Sharad** 主讲。该视频介绍了 Pallas 的概念和功能。
   - 对于那些希望熟悉 Pallas 特性和用例的人来说，这是一个非常有用的资源。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/google/jax/tree/main/jax/experimental/pallas/ops/tpu">google/jax 仓库 main 分支下的 jax/jax/experimental/pallas/ops/tpu</a>: Python+NumPy 程序的组合式变换：微分、向量化、JIT 到 GPU/TPU 等 - google/jax</li><li><a href="https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py">google/jax 仓库 main 分支下的 jax/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py</a>: Python+NumPy 程序的组合式变换：微分、向量化、JIT 到 GPU/TPU 等 - google/jax
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1281010043361296558)** (5 条消息): 

> - `Llm.c 替代方案`
> - `孟买 AI 峰会`
> - `倦怠预防策略` 


- **对 LLM 替代方案的期待**：一位成员对潜在的 **llm.c** 替代方案表示乐观，该方案与大语言模型无关，表明了对**单一用途解决方案**的需求。
   - 另一位成员补充说 **PyTorch** 已经可以提供更广泛的功能。
- **NVIDIA AI 峰会公告**：**NVIDIA AI 峰会**将于 **2024 年 10 月 23 日至 25 日**在**孟买**举行，届时将有超过 50 场会议涵盖各种 AI 主题，包括**生成式 AI**。
   - 鼓励成员们[立即注册](https://register.nvidia.com/flow/nvidia/aisummitindia/registration/login)，并在活动中与行业领袖和参展商交流。
- **关于预防倦怠的见解**：一位成员分享了关于避免倦怠的见解，强调了了解**个人极限**的重要性，并建议保持 **95%** 的努力而非 **100%**，以实现可持续性。
   - 他们建议专注于自己可以控制的事情，设定现实的目标，并原谅自己过去的错误，以鼓励持续进步。



**提到的链接**：<a href="https://www.nvidia.com/en-in/events/ai-summit/">加入 NVIDIA AI 峰会 2024</a>：10 月 23-25 日，印度孟买

  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1281061080679976993)** (1 条消息): 

> - `加入 Tenstorrent`
> - `CUDA kernel 开发`
> - `CUDA Mode IRL 活动` 


- **Atul Krishnadas 加入 Tenstorrent 担任 kernel 开发人员**：Atul Krishnadas 宣布他即将在圣克拉拉担任 **Tenstorrent 的 kernel 开发人员**。
   - 他表达了对 CUDA 的热情，并强调了他在该领域的开发背景。
- **开发 PyTorch/cuDNN 克隆版**：Atul 分享了他创建 **PyTorch/cuDNN 克隆版**的经验，他从零开始为各种功能编写了所有的 **CUDA kernels**。
   - 他展示了他的工作 Demo，体现了他在**前向/反向传播**和 mini-batch 训练方面的熟练程度。
- **询问 CUDA Mode IRL 活动**：Atul 询问了 21 日举行的 **CUDA Mode IRL 活动**是否还有名额，并提到他一段时间前已经申请了。
   - 他预先感谢社区提供有关活动名额的任何更新。


  

---


### **CUDA MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1281199356166410291)** (17 条消息🔥): 

> - `Batch Size 的性能见解`
> - `来自 PyTorch 的 Autotune 配置`
> - `GROUP_M 对 Triton 代码的限制`
> - `GemV 实现挑战`
> - `内存受限 (Memory-Bound) 性能分析` 


- **Batch Size 的性能见解**：在 batch size 达到 **16-32** 之前，加速效果保持一致，因为它利用了 1 个 **16x16** / **8x32** tensor core 指令，但在之后会变慢，而在较大的 batch size 下保持接近 **1x**。
   - Mobicham 指出，随着 autotune 参数的增多，速度有进一步提升的潜力。
- **来自 PyTorch 的 Autotune 配置**：一位成员分享了在 PyTorch 仓库中发现的**额外 autotune 配置**，这对于 **int8 mm** 的挑战非常有用。
   - 这些配置可能有助于 Mobicham 的性能测试，特别是在优化 tensor core 使用方面。
- **GROUP_M 对 Triton 代码的限制**：Mobicham 指出，将 **GROUP_M** 降低到 **8** 以下可能会对性能产生负面影响，因为 `tl.dot` 仅支持特定的 tensor core 形状。
   - 使用较小形状时出现的断言错误（assertion error）突显了实现高效方案的挑战。
- **GemV 实现挑战**：在 Triton 中苦于无法实现良好的 **gemv** 后，Mobicham 转向了 **CUDA**，从而开发了 **GemLite**。
   - 测试表明，使用乘法 + 加法是可行的，但最终性能不如使用 `tl.dot`。
- **内存受限 (Memory-Bound) 性能分析**：在使用高级配置设置时，**内存受限设置**下的性能仍然较慢，但在处理大 batch 时能达到接近 **FP16** 的速度。
   - 这对于大上下文的 prefill 和训练特别有利，表明整体进展有效。



**提到的链接**：<a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm_common.py">pytorch/torch/_inductor/kernel/mm_common.py at main · pytorch/pytorch</a>：Python 中的张量和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1281040467785551973)** (4 messages): 

> - `Open Sora Implementation`
> - `Graphics Progress` 


- **Open Sora 的 CUDA 实现工作**：一位成员分享了他们在 **CUDA** 和 **C++** 中实现 **Open Sora** 的努力，并指出这是一项巨大的任务，进展缓慢。
   - “*我真的希望图形学能有所突破……*”反映了对该领域更多进展的期待。
- **同行间的启发**：一位成员建议，这些讨论可能已经**激励了足够多的人**去贡献或进一步探索。
   - 这一评论突显了社区在面临挑战时的协作氛围。


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1281071034132598845)** (1 messages): 

> - `Third Wave Delays`
> - `Inbox Notifications` 


- **等待第三波**：一位成员对收件箱**未收到**任何更新表示沮丧，并提到必须等待**第三波**。
   - 他们指出，缺乏通知导致了对预期信息的延迟感。
- **对收件箱通知的不满**：同一位成员表示他们的收件箱仍然是空的，这暗示了在预期更新方面的脱节。
   - 这一评论反映了对群组内及时沟通的更广泛关注。


  

---


### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1280978503009042475)** (8 messages🔥): 

> - `Jupyter Notebook Versioning`
> - `Python Script for Benchmark Visualizations`
> - `Implementation of MoE Models` 


- **Jupyter Notebook 版本控制问题**：成员们讨论了 **Jupyter Notebook 版本控制的低效性**，称其通常很笨重且繁琐。
   - 他们提议创建一个 **Python 脚本**来生成 PNG 可视化图表，并将其存储在包含在 **.gitignore** 中的文件夹中。
- **为 PNG 存储方案创建 PoC**：**s1r_o** 提到正在为 PNG 存储方案准备概念验证 (PoC)，并建议通过将图像放置在指定的**忽略文件夹**中来确保方案万无一失。
   - **Byronhsu1230** 同意该方法，并表示将咨询一位此前实现过类似方案的同事。
- **讨论转移至 PR**：s1r_o 创建了一个 **Pull Request (PR)** 和一个分支，以进一步讨论所提方案的实现细节。
   - 他们表示实现不会花费很长时间，并鼓励在 PR 中继续讨论。
- **探索来自 Huggingface 的 MoE 模型**：s1r_o 提出了关于实现来自 Huggingface 的 **MoE 模型**（如 **Mixtral** 或 **Nllb_moe**）的想法。
   - 其想法是支持多项操作，并在开发完成后集成 **MoE kernel**。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1281061554720211050)** (19 messages🔥): 

> - `MCTS Application in Image Tasks`
> - `Creative AI Workshops`
> - `Keyword-Driven Generative Models`
> - `Undergraduate Internships in Labs`
> - `Minimalist UI Development` 


- **关于 MCTS 在图像识别中应用的讨论**：有一场关于 **Monte Carlo Tree Search (MCTS)** 如何应用于图像生成的辩论，并将其逻辑反转与 AlphaZero 和 AlphaProof 等模型进行了比较。
   - *一位参与者质疑 MCTS 如何被反转*，特别是当每一步都严重依赖前一步时，并强调 MCTS 是增强 Policy（策略）而非生成它们。
- **寻找创意 AI 工作坊**：一位成员询问即将举行的专注于 **创意 AI** 的 **workshops**，希望应用他们在关于 Diffusion 模型和 LoRA 组合（LoRA composition）论文中的研究成果。
   - 另一位成员对这类工作坊在 ICCV 期间的相关性表示怀疑，考虑到投稿截止日期。
- **从标题中提取元数据**：**关键词驱动的生成模型**（如 Stable Diffusion）需要对训练数据进行仔细的预处理，这引发了对其方法论的好奇。
   - 一位用户正在集思广益，探讨从 120 万个标题中提取 **metadata tags** 的方法，并将讨论与数据策展（data curation）的最佳实践联系起来。
- **学术实验室的本科生实习**：对话强调学术实验室**可以**聘请本科实习生，特别是如果 PI 有精力且学生有合适的背景。
   - 一位实习生分享了他们从兼职开始并转为全职角色的经验，为潜在的职业路径提供了参考。
- **极简 UI 的开发**：一位用户宣布他们计划重写一个极简 UI，目标是实现**超强可扩展性**的设计，且不含不必要的依赖。
   - 他们表达了对合作的兴趣，邀请他人加入他们的项目，旨在创建一个可定制的用户界面和服务器。



**提到的链接**：<a href="https://x.com/JulienBlanchon/status/1831719118434709868">来自 Julien Blanchon (@JulienBlanchon) 的推文</a>：正在尝试弄清楚如何修复 Comfy 👀

  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1280994704443314256)** (52 条消息🔥): 

> - `Inefficiency of Scaling Parameters` (参数缩放的低效性)
> - `Transfusion Model Insights` (Transfusion 模型见解)
> - `Gradient Behavior During Training` (训练过程中的梯度行为)
> - `Effects of Generative AI on Work` (生成式 AI 对工作的影响)
> - `Numerical Stability in Optimizers` (优化器中的数值稳定性)


- **低效的参数缩放会影响训练**：一位成员对在不增加数据集大小的情况下显著增加参数数量的低效性提出了疑问，并引用了 Chinchilla 论文进行计算。
   - 另一位成员建议直接查阅该论文的公式，以更准确地理解参数缩放的后果。
- **来自 Transfusion 论文的见解**：讨论指向了 [Transfusion 论文](https://www.arxiv.org/abs/2408.11039)，该论文探索了在离散和连续数据上训练多模态模型。
   - 据观察，与在离散图像 Token 上训练语言模型相比，作者实现了更好的缩放性能。
- **训练梯度中的异常模式**：一位成员讨论了在蒸馏训练期间观察到的梯度之间 Hamming 相似度的峰值，认为某些数据点序列可能是有益的。
   - 他们考虑了数值精度影响梯度的可能性，并促使对优化器行为进行进一步检查，特别是在他们的 Lion 实现中。
- **生成式 AI 提升开发者生产力**：一篇题为 *The Effects of Generative AI on High Skilled Work* 的共享论文显示，使用 AI 工具 (GPT 3.5) 的开发者完成任务的数量增加了 **26.08%**。
   - 这一发现表明，将 AI 技术集成到软件开发中可以带来显著的生产力提升。
- **优化器中的数值稳定性问题**：人们对 Lion 优化器中潜在的数值稳定性问题表示担忧，特别是可能影响训练一致性的梯度离散跳变。
   - 有建议称，将参数调整为标准的 32-bit 格式可能有助于缓解训练过程中报告的一些数值不一致问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/distily/distily_attn_mlp_sweep/tensorboard">distily/distily_attn_mlp_sweep · 训练指标</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2409.02426">Diffusion Models Learn Low-Dimensional Distributions via Subspace Clustering</a>：最近的实证研究表明，扩散模型可以有效地学习图像分布并生成新样本。值得注意的是，即使在样本数量较少的情况下，这些模型也能实现这一目标...</li><li><a href="https://www.arxiv.org/abs/2408.11039">Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model</a>：我们介绍了 Transfusion，这是一种在离散和连续数据上训练多模态模型的方法。Transfusion 将语言建模损失函数（Next Token Prediction）与扩散技术相结合...</li><li><a href="https://en.m.wikipedia.org/wiki/Parametric_design">参数化设计 - 维基百科</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1280968526018117726)** (2 条消息): 

> - `Leaderboard IFeval`
> - `IFeval differences` 


- **理解排行榜 IFeval**：一位成员询问了 **leaderboard_ifeval** 和 **ifeval** 之间的区别。
   - *关于它们的功能或用途的澄清仍待定。*
- **寻求系统组件的澄清**：一位成员表示需要澄清两个组件之间的区别：**leaderboard_ifeval** 和 **ifeval**。
   - *讨论暗示了它们在角色上的差异，但仍有待进一步详细说明。*


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 条消息): 

bennib2407: 什么是 SOTA 的视频字幕 (video captioning) 模型？
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1281375430204457012)** (1 条消息): 

> - `RoPE Compatibility` (RoPE 兼容性)
> - `Attention Output Discrepancies` (Attention 输出差异) 


- **RoPE 实现兼容性问题**：一位成员询问 **GPTNeoX / Pythia** 的 RoPE [Hugging Face 实现](https://huggingface.co/) 是否与 **LLaMA** 和 **GPT-Fast** 模型中使用的实现兼容。
   - 他们提供了一段频率和旋转嵌入 (Rotary Embedding) 计算的代码片段供参考。
- **Attention 输出的对比分析**：该成员注意到他们的 **Pythia 模型** 实现与他们自己的实现之间的 Attention 输出存在显著差异 (>95%)。
   - 这种差异促使他们寻求关于 RoPE 应用中潜在不兼容或实现错误的见解。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1280974362887065777)** (68 条消息🔥🔥): 

> - `SSI Inc 融资`
> - `You.com 融资`
> - `Karpathy 见解`
> - `OpenAI 定价`
> - `Replit Agent 发布` 


- **SSI Inc 获得 10 亿美元融资**：SSI Inc 获得了一笔高达 **10 亿美元** 的融资，而 Sakana 也获得了 **1 亿美元**。
   - 讨论中出现了关于这笔资金中有多少可能会分配给 **Nvidia** 的猜测。
- **You.com 凭借新融资重新调整重心**：[You.com](https://you.com) 已从 AI 搜索产品转向开发更深层次的生产力 Agent，并获得了 **5000 万美元** 的新融资，旨在探索处理复杂查询的创新方法。
   - 创始人 Richard Socher 强调，在简单查询上与 Google 竞争的可行性，不如增强以生产力为中心的能力。
- **Karpathy 对 Tesla 和自动驾驶技术的看法**：在最近的一期播客中，Andrej Karpathy 表示，虽然 **Waymo** 取得了长足进步，但他相信 **Tesla** 将在自动驾驶技术方面长期领先，并将其归结为基础软件与硬件的问题。
   - 他还讨论了 Tesla 的人形机器人 **Optimus** 的变革潜力，强调了其在工厂中的应用。
- **OpenAI 考虑推出高端订阅模式**：据报道，OpenAI 正在评估为其下一代模型推出每月 **2000 美元** 的订阅服务，这暗示其能力可能比低端方案提升 100 倍。
   - 定价讨论暗示了模型性能的实质性增强，或者是为了在成本上升的情况下覆盖运营开支。
- **Replit Agent 发布**：Replit 推出了 **Replit Agent**，旨在为订阅者提供早期访问权限，以自动化软件开发任务（如设置开发环境）。
   - 此举被视为 Replit 增强其产品供应并可能利用 AI 集成到编程工作流中的战略努力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://news.ycombinator.com/item?id=41453237">Yi-Coder: A Small but Mighty LLM for Code | Hacker News</a>：未找到描述</li><li><a href="https://gamengen.github.io/">GameNGen</a>：Diffusion Models 是实时游戏引擎</li><li><a href="https://x.com/annarmonaco/status/1831347029202915478?s=46">来自 Anna Monaco (@annarmonaco) 的推文</a>：介绍 Paradigm —— 一个以 AI 为核心的重新构想的工作空间。Paradigm 以电子表格为基础原语，让成群的智能 Agent 触手可及。Pa 的真正力量...</li><li><a href="https://x.com/time/status/1831665580241293772?s=46">来自 TIME (@TIME) 的推文</a>：《时代》杂志新封面：AI 领域最具影响力的 100 人 https://ti.me/4dQcJ1Q</li><li><a href="https://x.com/kennandavison/status/1831432265768808872?s=46">来自 Kennan Davison (@kennandavison) 的推文</a>：很高兴介绍 Icon：我们帮助品牌与真实的创作者一起创作制胜的 AI 广告。我们得到了 Peter Thiel 的 Founders Fund 以及 Ramp、Flexport、Pika 和 Cognition (Devin) 创始人的支持。未来的...</li><li><a href="https://x.com/cto_junior/status/1831705018224754931?s=46">来自 TDM (e/λ) (@cto_junior) 的推文</a>：1) 什么？引用 TIME (@TIME) 《时代》杂志新封面：AI 领域最具影响力的 100 人 https://ti.me/4dQcJ1Q</li><li><a href="https://x.com/bindureddy/status/1831746158752088178">来自 Bindu Reddy (@bindureddy) 的推文</a>：OpenAI 正在考虑收取每月 2000 美元来访问其顶级模型。说正经的，这将是 Vision-Pro 级别的灾难。我希望这只是个玩笑</li><li><a href="https://x.com/mjlbach/status/1831323536788791595?s=46">来自 Michael Lingelbach (@mjlbach) 的推文</a>：看来一个 SOTA 开源文本转音乐模型（一个 rectified flow dit）发布了。论文地址：https://arxiv.org/abs/2409.00587 代码地址：https://github.com/feizc/FluxMusic 示例听起来非常...</li><li><a href="https://x.com/aiexplainedyt/status/1831710902636228694?s=46">来自 AI Explained (@AIExplainedYT) 的推文</a>：你会为 ChatGPT 每月支付 2000 美元吗？根据 The Information 刚刚发布的关于 OpenAI 的报告，这是订阅方案中“摆在桌面上的”最高价格。这将是...</li><li><a href="https://x.com/teortaxesTex/status/1831717316121243947">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：是的，再次得到证实：@deepseek_ai 确实合并了代码和通用模型。（变更日志被追溯修改了）。Tao 与算力回收的艺术。喜闻乐见。引用 Teortaxes▶️ (@teortaxes...</li><li><a href="https://x.com/swyx/status/1831742418053689853">来自 shawn swyx wang (@swyx) 的推文</a>：@karpathy 的准则：- 站在人类这一边 - 选择可扩展的事物 - 将它们扩展到全人类。事物 = { @tesla 摄像头视觉 | 人形机器人 | Transformers | AI 教育 @EurekaLabsAI} 最喜欢的...</li><li><a href="https://podcasts.apple.com/us/podcast/no-priors-artificial-intelligence-technology-startups/id1668002688?i=1000668455289">与 Andrej Karpathy 探讨通往自主智能之路</a>：Andrej Karpathy 在本周的 No Priors 节目中加入了 Sarah 和 Elad。作为 OpenAI 的创始团队成员和特斯拉前 AI 高级总监，Andrej 需要...</li><li><a href="https://x.com/mattshumer_/status/1831767014341538166?s=46">来自 Matt Shumer (@mattshumer_) 的推文</a>：我很高兴地宣布 Reflection 70B，全球顶尖的开源模型。使用 Reflection-Tuning 训练，这是一种旨在让 LLM 能够纠正自身错误的技术。405B 将于下周推出...</li><li><a href="https://x.com/Techmeme/status/1831696947914404181">来自 Techmeme (@Techmeme) 的推文</a>：OpenAI 表示其 ChatGPT 企业版（包括 ChatGPT Team、Enterprise 和 Edu）现在拥有超过 100 万付费用户 (@rachelmetz / Bloomberg) https://www.bloomberg.com/news/articles/2024-09-05/o...</li><li><a href="https://x.com/natolambert/status/1831353405585195121?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>：Ai2 今天发布了 OLMoE。这是我们迄今为止最好的模型。- 1.3B 激活参数， 6.9B 总参数，每层 64 个专家 - 在来自 DCLM 基准 + Dolma 的 5T token 上训练 - Tulu 3 后训练的新预览...</li><li><a href="https://x.com/togethercompute/status/1831783919718690877?s=46">来自 Together AI (@togethercompute) 的推文</a>：🚀 NVIDIA H200 和 Together Kernel Collection (TKC) 即将登陆 Together GPU Clusters：为 AI 训练、微调和推理提供加速的性能、效率和可扩展性...</li><li><a href="https://x.com/natolambert/status/1831701773721203164?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>：就像 Q* 一样，OpenAI 的 Strawberry 系统泄露的信息已经足够让我们对其训练设置和用例产生实质性且有趣的假设。一些想法：* 将自言自语作为推理...</li><li><a href="https://x.com/amasad/status/1831730911685308857">来自 Amjad Masad (@amasad) 的推文</a>：AI 在编写代码方面表现惊人。但这不足以创建软件...</li>

你需要设置开发环境、安装包、配置数据库，如果运气好的话，还要进行部署。是时候自动化了……</li><li><a href="https://news.ycombinator.com/item?id=41456552">Launch HN: Maitai (YC S24) – Self-Optimizing LLM Platform | Hacker News</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=41457633">Show HN: AnythingLLM – 开源、一体化桌面 AI 助手 | Hacker News</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=41451698">Show HN: Laminar – 为 LLM 应用打造的开源 DataDog + PostHog，使用 Rust 构建 | Hacker News</a>：未找到描述</li><li><a href="https://x.com/moyix/status/1831528226331521293?s=46">Brendan Dolan-Gavitt (@moyix) 的推文</a>：OpenAI：`pip install openai` 并设置 `OPENAI_API_KEY`；Anthropic：差不多，但执行 `s/openai/anthropic/g`；Google：噢天哪。好吧，你有 GCP 账号吗？没有？去开一个。还有付款方式。现在创建一个……</li><li><a href="https://techcrunch.com/2024/09/04/you-com-refocuses-from-ai-search-to-deeper-productivity-agents-with-new-50m-round/">获得 5000 万美元新融资，You.com 认为其 AI 能在难题上击败 Google | TechCrunch</a>：如果你构建 AI 搜索产品，你就是在与 Google 竞争。但 Google 在回答单一、简单的查询（如“如何……”）时要容易得多。</li><li><a href="https://buttondown.email/ainews/archive/ainews-to-be-named-5745/">[AINews] SciCode：HumanEval 获得 STEM 博士级升级</a>：博士级基准测试就是你所需要的一切。2024/7/15-2024/7/16 的 AI 新闻。我们检查了 7 个 subreddits、384 个 Twitter 账号和 29 个 Discord 社区（466 个频道和 2228...
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1280967019239575655)** (55 条消息🔥🔥): 

> - `Open Interpreter 生日`
> - `教学模式 (Teach Mode)`
> - `Open Interpreter 仓库`
> - `AGI 讨论`
> - `Fulcra App 可用性` 


- **Open Interpreter 庆祝生日**：成员们庆祝了 **Open Interpreter** 的生日，并指出了它对 AI 与人类交互及创新的影响。
   - 一位参与者幽默地评论道：*“AGI 已实现，我们现在都可以回家了”*。
- **探索教学模式 (Teach Mode) 功能**：讨论了 Open Interpreter 的 **Teach Mode**；用户可以说 *“我想教你一些东西”* 来引导系统创建新技能。
   - 它可以根据所教的任务调整技能，重点是灵活执行并与 Rabbit Tech 的方法论保持一致。
- **访问 Open Interpreter 仓库**：**Open Interpreter** 和 **01** 仓库是开源的，邀请用户在这些基础上构建自己的应用。
   - 一位用户表示有兴趣将这些功能集成到他们的软件中，特别是用于 Web 自动化实例。
- **AGI 发布查询**：一位成员询问了关于 AGI 发布的消息，另一位成员幽默地回应道：*“AGI 已实现，我们现在都可以回家了”*。
   - 成员们似乎对这个想法很感兴趣，后续消息中反映出兴奋与怀疑交织的情绪。
- **Fulcra App 区域可用性**：一位成员对 **Fulcra app** 表示了兴趣，并询问其在新西兰以外地区的发布情况。
   - 目前还没有关于发布时间表的直接回复，表明用户仍在持续期待。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/youre-a-wizard-hagrid-afirmation-magic-magical-gif-16533730">你是名巫师，海格 GIF - Youre A Wizard Hagrid Afirmation - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/clapping-yay-excited-soexcited-greatnews-gif-8845875809066863059">鼓掌耶 GIF - Clapping Yay Excited - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/frankenstein-its-alive-happy-excited-gif-5625959">科学怪人它活了 GIF - Frankenstein Its Alive Happy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言界面</a>：计算机的自然语言界面。通过在 GitHub 上创建账号为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01: 桌面、移动端和 ESP32 芯片的首选开源语音界面。</a>：桌面、移动端和 ESP32 芯片的首选开源语音界面。 - OpenInterpreter/01
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1281147494499225610)** (7 条消息): 

> - `O1 最近的 Demo`
> - `O1 发货日期`
> - `House Party 活动`
> - `Discord 链接` 


- **请求 O1 的最近 Demo**：*有人*询问了关于 **O1** 的任何最新 Demo，表明了对该产品更新的持续关注。
   - 这反映了在产品接近发货之际，用户对其功能进行直观展示的渴望。
- **O1 发货日期不确定**：一位用户对他们预订的 **O1** 的**发货日期**表示沮丧，并提到产品尚未送达。
   - 这突显了对延迟的担忧，因为预订通常意味着更早获得产品的承诺。
- **House Party 活动公告**：一名成员鼓励其他人稍后关注 **House Party**，这标志着该活动在社区中的重要性。
   - 这预示着成员之间即将迎来讨论和社交的机会。
- **分享 House Party 活动链接**：分享了几个用于访问 House Party 活动的 **Discord 链接**，方便感兴趣的成员参与。
   - 这些分享的链接促进了围绕 O1 讨论的参与度和社区互动。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1281096298732191847)** (27 条消息🔥): 

> - `使用 PyTorch 2.4 时的编译错误`
> - `输入填充（Input padding）性能`
> - `训练期间的内存占用（Memory footprint）`
> - `数据集中的 Token 分配`
> - `针对 torch.compile 的 CI 测试` 


- **使用 PyTorch 2.4 时的编译错误**：成员们报告了在 PyTorch 2.4 的最新 main 分支上出现的编译错误，特别是关于 fake tensors 的问题。有人指出，使用 `os.environ['TORCH_COMPILE_BACKEND'] = 'aot_eager'` 可能会在 CI 中掩盖这些错误。
   - 一位成员建议关注与默认后端测试相关的潜在 CI 问题，暗示 CI worker 需要安装更高版本的 gcc。
- **输入填充（Input padding）的性能影响**：一位成员在 Alpaca 数据集上使用默认配置对输入填充进行了测试运行，发现速度明显下降。他们指出，虽然由于碎片减少内存占用（memory footprint）有所改善，但性能优化方面的收益并不理想。
   - 另一位成员建议同时报告填充和未填充的 token，以便深入了解填充造成的浪费，并强调填充的 token 仍然会被处理。
- **内存占用（Memory footprint）考量**：关于内存管理的讨论带来了对更好内存占用的见解，以及对训练期间 OOM 问题的启示。使用 expandable segments 似乎无法解决较长序列长度下的内存激增问题。
   - 成员们强调预留内存对于避免 OOM 至关重要，一位成员指出内存的增加可能与序列长度的增加相对应。
- **对 CI 测试标准的需求**：有建议针对使用默认后端的 torch.compile CI 测试开启一个单独的 issue，因为错误报告不一致。此话题随后在现有 GitHub issues 的背景下被再次讨论。
   - 围绕 CI 标准的参与包括讨论如何建立一个环境，以更好地复现不同 PyTorch 版本所面临的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/676.">Issues · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/ao/pull/812">[Low-bit optim] Improve compile time + Fix PyTorch 2.3 support for 4-bit optim by gau-nernst · Pull Request #812 · pytorch/ao</a>: 针对单个参数的静态形状编译优化步骤 + 禁用缓存大小限制。对于给定的模型，single_param_adam() 的不同参数组合数量是固定的 -> 可以安全地禁用...</li><li><a href="https://github.com/huggingface/transformers/blob/5c1027bf09717f664b579e01cbb8ec3ef5aeb140/src/transformers/trainer.py#L1535.">transformers/src/transformers/trainer.py at 5c1027bf09717f664b579e01cbb8ec3ef5aeb140 · huggingface/transformers</a>: 🤗 Transformers: 为 Pytorch, TensorFlow, 和 JAX 提供的最先进的机器学习库。 - huggingface/transformers</li><li><a href="https://github.com/pytorch/torchtune/actions/runs/10723116777/job/29735707305?pr=1315">Prevent OOM during checkpoint save on colab for llama3-8b qlora recipe · pytorch/torchtune@a02ccd6</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/f437639f6abf101cc2b40793d5d86dbda35e24ec/tests/recipes/test_full_finetune_single_device.py#L61">torchtune/tests/recipes/test_full_finetune_single_device.py at f437639f6abf101cc2b40793d5d86dbda35e24ec · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1280983342354927638)** (13 messages🔥): 

> - `DeepFusionModel Caches`
> - `Testing DeepFusionModel`
> - `Unsloth Backed by YC`
> - `Daniel Han's Contribution`
> - `Meta Employment Clarification` 


- **DeepFusionModel Caches 误解**：讨论集中在如果 encoder 缺少 `setup_caches` 函数，`deepfusionmodel.setup_caches` 是否应该忽略 `encoder_max_seq_len`。
   - *这有点反直觉，但 encoder 的序列长度是为 decoder 中的 cross attention 层准备的*。
- **DeepFusionModel 测试增强**：一名成员更新称已向 DeepFusionModel 添加了 kv caching 测试，并分享了一个 Pull Request 进行评审。
   - [Pull Request #1449](https://github.com/pytorch/torchtune/pull/1449) 引入了对最大缓存序列长度的覆盖（overrides），并围绕其目的展开了进一步讨论。
- **Unsloth 获得 Y Combinator 支持**：一名成员指出 Unsloth 现在由 Y Combinator 支持，引发了社区对未来可能获得支持的兴趣。
   - 随着有人表示希望接下来能获得类似的严肃支持，期待感有所增加。
- **对 Daniel Han 的赞赏**：社区成员表达了对 Daniel Han 的赞赏，称其为传奇人物，标志着他的重大贡献。
   - 成员们认可了来自 AI 社区知名人士的努力和支持。
- **Meta 入职情况澄清**：针对关于在 Meta 工作的假设分享了一份评价，澄清并非所有成员都隶属于该公司。
   - 一名成员强调 *Salman 纯粹是出于对这项事业的热爱在做这件事*，而其他人确认他们确实为 Meta 工作。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/pull/1449">[RFC] Adding overrides for max cache seq length by SalmanMohammadi · Pull Request #1449 · pytorch/torchtune</a>：上下文 此 PR 的目的是什么？是添加新功能、修复 bug、更新测试和/或文档还是其他（请在此处添加）#1364 Changelog 此 PR：增加了对覆盖 th... 的支持。

  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1281113762731917344)** (13 messages🔥): 

> - `System prompt optimization`
> - `Cohere updates`
> - `Community engagement` 


- **System Prompt 优化困境**：一位用户寻求优化其 system prompt 的帮助，但遇到了报错：**Could not parse & validate the given body**。
   - 另一位成员建议在特定频道分享细节以获得更好的帮助。
- **探索 Cohere 的新动态**：一名成员询问了 **Cohere** 的最新更新以及其他人如何使用该平台。
   - 回复引导他们前往 **Cohere blog**，通过 [cohere.com/blog](https://cohere.com/blog) 快速了解近期进展和客户用例。
- **新成员寻求社区联系**：一位新成员表示打算与 **Cohere 社区** 建立联系，以更好地了解其产品。
   - 他们确认已经查看了文档作为起点。
- **对新用户的鼓励**：一位社区成员向新人保证，这里是学习和协作的正确场所。
   - 他们鼓励查看平台的全面文档以有效开始。



**提到的链接**：<a href="https://cohere.com/blog">The Cohere Blog</a>：探索我们收集的富有洞察力的博客文章，涵盖各种生成式 AI 主题。我们的文章提供深入分析、专家意见和实用建议，以提供信息和启发。 

  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1280971909311369279)** (8 messages🔥): 

> - `文本建议功能`
> - `用于报告生成的 LLM Agents`
> - `Cohere 使用最佳实践` 


- **实现类似 Gmail Smart Compose 的文本建议功能**：一位成员寻求关于使用 Cohere 模型在其消息平台中实现类似于 **Gmail Smart Compose** 的**文本建议功能**的指导。
   - 另一位成员建议，这可以通过有效地利用邮件上下文对模型进行 Prompting 来实现。
- **使用 LLM Agents 生成报告**：一位成员询问如何使用 **LLM Agents** 根据之前的写作风格和利益相关者的会议记录来生成报告。
   - 回复中包括建议对会议记录采用 **RAG 结合 Nimble rerank**，并对写作风格采用 **meta prompting** 技术。
- **熟练掌握 Cohere**：一位成员询问了关于如何有效使用 Cohere 并产出高质量结果的建议。
   - 另一位成员建议查阅 [Cohere 文档](https://docs.cohere.com/docs/the-cohere-platform) 以了解最佳实践和模型功能。



**提到的链接**：<a href="https://docs.cohere.com/docs/the-cohere-platform">The Cohere Platform — Cohere</a>：Cohere 提供世界级的 Large Language Models (LLMs)，如 Command、Rerank 和 Embed。这些模型帮助开发者和企业构建由 LLM 驱动的应用，如对话式 Agents、摘要生成...

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1281069250844561490)** (3 messages): 

> - `用于报告生成的 LLM Agents`
> - `OpenSesame 2.0 发布` 


- **探索用于报告创建的 LLM Agents**：一位成员询问如何为**内部审计团队**使用 **LLM Agents**，根据之前的写作风格和利益相关者的会议记录来生成报告。
   - *有人尝试过这种方法吗？*
- **OpenSesame 2.0 带来重大增强**：**OpenSesame 2.0** 已经发布，包含重大更新，包括消除对 ground truth 输入的需求，并连接到 **vector DBs** 以进行实时语义搜索。
   - 该更新还具有对 [OpenAI](https://www.loom.com/share/9569b031ddd343b792856fb23e95d77a?sid=341fa6b2-d295-4c4d-aea5-362accc30c7f)、**Gemini** 和 **Cohere** 等平台的多模型支持。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1280996290422898774)** (3 messages): 

> - `使用 LlamaIndex 的 Netchex AI`
> - `create-llama 模板`
> - `llama-deploy 微服务` 


- **Netchex AI 彻底改变员工支持**：@Netchex 使用 LlamaIndex 实现了 **AskHR + Netchex AI**，仅用一个月时间和两名工程师就为中小型企业转型了员工支持模式。
   - 他们使用了**高级 RAG 流水线**来实现上下文感知响应，展示了 HR 领域的快速开发能力。[点击此处阅读更多](https://t.co/JWz8sgqRj7)。
- **create-llama 引入多 Agent 工作流**：**create-llama** 的最新更新提供了 Python 中的多 Agent 工作流，强调了其在各种用例快速部署中的作用。
   - 一个示例工作流利用三个 Agents 来生成博客文章，展示了其灵活性和效率。[去看看吧！](https://t.co/nmrtjUw7iL)。
- **为微服务推出的 llama-deploy 发布**：新的 **llama-deploy** 系统允许无缝部署基于 LlamaIndex Workflows 的微服务，这代表了其演进中的重要一步。
   - 此次发布建立在自 **llama-agents** 和 **Workflows** 发布以来积累的经验之上，增强了开发者的部署能力。[在此获取详情](https://t.co/6TmgpPiZxp)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1281143212509106239)** (20 条消息🔥): 

> - `llama-index-experimental-param-tuner` 安装
> - 从 `ChromaVectorStore` 获取 embedding 向量
> - 将 Claude 与 LlamaIndex 集成
> - Text-to-SQL 功能与 embeddings
> - 优化 RAG 应用中的 prompt 


- **安装 llama-index-experimental-param-tuner**：要安装该实验性软件包，请针对 **llama-index** 版本 **0.11.3** 运行命令 `pip install llama-index-experimental`。
   - 一位用户确认此安装步骤对于该功能是必需的。
- **ChromaVectorStore 中的 Embedding 向量**：一位用户在从相关节点获取 embedding 向量时遇到问题，导致出现 ValueError，提示 embedding 未设置。
   - 其他人讨论认为，重构 Chroma 类可能会解决无法返回 embeddings 的问题。
- **在 LlamaIndex 中设置 Claude**：分享了一份在 LlamaIndex 中利用 Claude 最新模型的全面指南，包括设置说明和 tokenizer 设置。
   - 模型包括 **Claude 3 Opus**、**Claude 3 Sonnet** 和 **Claude 3 Haiku**，重点强调了遵循 chat engine 设置文档的重要性。
- **将 Text-to-SQL 与语义搜索结合**：一位用户询问如何在特定表列上实现 Text-to-SQL 功能，其中一些列包含用于语义搜索的 embeddings。
   - 讨论中未提供直接解决方案，表明需要对该集成进行进一步探索。
- **RAG 应用中的 Prompt 优化**：成员们讨论了从 QueryPipelines 到 Workflows 的过渡，并指出在 LlamaIndex 中使用 DSPy 进行优化的潜力。
   - 提到了有用的集成示例以及维护高效 RAG pipeline 的复杂性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/llamaindex.py#L249)">dspy/dspy/predict/llamaindex.py at main · stanfordnlp/dspy</a>：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://github.com/run-llama/llama_index/blob/fd4a2e6b2da51fb6b3c50f636f795c0599341ff8/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py#L378">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py at fd4a2e6b2da51fb6b3c50f636f795c0599341ff8 · run-llama/llama_index</a>：LlamaIndex 是适用于 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/anthropic/">Anthropic - LlamaIndex</a>：无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_plus_context/">Chat Engine - Condense Plus Context Mode - LlamaIndex</a>：无描述
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1281260297771946055)** (14 条消息🔥): 

> - `构建 AI Agents`
> - `Chatbot 开发`
> - `ReAct Agent 部署`
> - `AI Agents 的数据库解决方案` 


- **社区寻求关于 AI Agent 平台的建议**：一位成员正在开发一个用于构建、部署和变现 **AI agents** 的新平台，并在调研阶段寻求现有 Agent 构建者的见解。
   - 他们提供感谢和 Beta 测试访问权限，以换取简短的交流。
- **构建文档驱动型 Chatbot 的指南**：另一位成员请求协助创建一个能有效利用 **两个 PDF 文件** 内容进行交互的 Chatbot，并强调流畅的用户体验。
   - 讨论强调了文档加载、响应生成和对话管理等关键需求。
- **Chatbot 的 FAISS Vector DB 集成**：一位参与者询问了包括将文档存储在 **FAISS vector DB** 中以检索答案的端到端解决方案。
   - 他们收到了关于文档加载、Embeddings 创建以及使用 LangChain 设置 Retriever 的指导。
- **从 SQLite 迁移到云数据库**：针对在 **GCP** AppEngine 上运行的 **ReAct agent**，有成员请求使用 **Postgres** 或 **MySQL** Saver 实现作为 SQLite 的替代方案。
   - 贡献者对重新部署时丢失本地 SQLite 数据库上下文表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/issues/4950>):">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/11857>):">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss_async/#saving-and-loading>)">Faiss (Async) | 🦜️🔗 LangChain</a>: Facebook AI Similarity Search (Faiss) 是一个用于高效相似性搜索和密集向量聚类的库。它包含搜索任意大小向量集的算法，最高可达...</li><li><a href="https://github.com/langchain-ai/langchain/issues/17576>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17412>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/11661>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/8170>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1281264762134462498)** (2 messages): 

> - `Vision Language Models`
> - `CodeMaster App`
> - `Gamification in Learning`
> - `DSA Learning Techniques` 


- **探索 Vision Language Models 的最新进展**：一篇新的博客文章深入探讨了 Vision Language Models (VLMs) 的演变，从早期的 **CLIP** 到先进的模型如 **Flamingo** 和 **LLaVA**。文章强调了通过视觉和文本数据的联合训练如何提升在 Segmentation 和 Classification 等各种任务中的表现，并引用了 [DALL-E 2](https://openai.com/index/dall-e-2-extending-creativity/) 等作品。
   - 该博客强调了基础模型的成功，并提供了对该领域近期突破的见解，参考了 [GPT-4](https://arxiv.org/abs/2303.08774) 和 [PaLM 2](https://arxiv.org/abs/2305.10403) 等著名模型。
- **旨在增强学习效果的 CodeMaster App 发布**：新推出的 **CodeMaster** 应用旨在通过 Gamification 和基于科学的知识保留技术来提高编程技能。用户可以参与社区竞赛并在巩固学习的同时获得奖励。
   - 关于 CodeMaster 的反馈强调了它对编程教育的影响，用户称赞其 **Spaced Repetition** 功能能够有效掌握概念，正如 **Alex Chen** 和 **Sarah Johnson** 的证言所展示的那样。
- **征求 DSA 学习项目的反馈**：一个讨论学习 Data Structures and Algorithms (DSA) 的趣味方法的项目正在寻求社区反馈。其目标是将每日解题与科学支持的知识保留方法相结合。
   - 该倡议仍处于初期阶段，仅开发了 **8 小时**，旨在通过学习 DSA 中的 Gamified 体验来激励用户。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.lightly.ai/post/introduction-to-vision-language-models">A Brief Introduction to Vision Language Models</a>：Vision Language Models 领域近期进展概述。从早期的对比学习方法如 CLIP 到更先进的模型如 Flamingo 和 LLaVA。</li><li><a href="https://codehelper.koesterjannik.com/">Code Helper</a>：未找到描述
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1281081298815025153)** (10 messages🔥): 

> - `Comfy Rewrite Project`
> - `Complimentary GUI for Comfy`
> - `SwarmUI`
> - `ComfyBox Project` 


- **Julien Blanchon 开始 Comfy 重写**：成员 **Julien Blanchon** 宣布尝试从头开始进行极简主义的 **Comfy** 重写，目标是建立一个无依赖且具有超强扩展性的用户界面。
   - 该项目邀请协作，并寻求在不牺牲灵活性的情况下简化使用。
- **讨论配套 GUI 的想法**：另一位成员建议开发一个 **配套 GUI**，在后端使用 Comfy，同时提供类似于 **A1111** 的更简便的用户体验。
   - 其目标是允许快速执行 Inpainting 和 Upscaling 等任务，而无需处理加载节点的复杂性。
- **探索 ComfyBox 项目**：一位成员提到了过去创建类似界面的尝试，指出了 GitHub 上似乎已废弃的 **ComfyBox 项目**。
   - 有人对其笨重的 UI 提出了批评，认为它缺乏所期望的流线型体验。
- **关于 SwarmUI 的讨论**：成员们认可了 **SwarmUI**，它被定位为一个模块化的 Web 用户界面，专注于 Stable Diffusion 的易用性和性能。
   - 有人指出 SwarmUI 强调可扩展性，对寻求更多用户友好选项的用户很有吸引力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/JulienBlanchon/status/1831719118434709868">Julien Blanchon (@JulienBlanchon) 的推文</a>：试图找出如何修复 Comfy 👀</li><li><a href="https://x.com/JulienBl">undefined 的推文</a>：未找到描述</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (原 StableSwarmUI)，一个模块化的 Stable Diffusion Web 用户界面，强调让强大工具易于获取、高性能且具可扩展性。</a>：SwarmUI (原 StableSwarmUI)，一个模块化的 Stable Diffusion Web 用户界面，强调让强大工具易于获取、高性能且具可扩展性。 - mcmonkeyprojects/Swa...
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1281137852524003400)** (6 messages): 

> - `Transfusion Model`
> - `Reflection 70B`
> - `Causal UNET Performance`
> - `Unified Multi-modal Models` 


- **Meta 发布全新 Transfusion 模型**：Meta 发布了关于 **Transfusion** 的论文，这是一种多模态模型，结合了离散和连续数据的语言与扩散训练技术，在 **1T** 文本 token 和 **692M** 图像上预训练了一个 7B 模型。
   - 该研究强调了该模型扩展到 **audio** 和 **video** 的潜力，使用 VAE 在不同媒体类型之间实现平滑过渡。
- **Reflection 70B 承诺重大进展**：围绕 **Reflection 70B** 的发布引发了热烈讨论，据称它是全球顶尖的开源模型，能通过 **Reflection-Tuning** 独立修复自身错误。
   - 报告称其在多个基准测试中超越了包括 **GPT-4o** 在内的现有模型，**405B** 版本定于下周发布，引起了 AI 社区的关注。
- **Causal UNET 表现与 Dense Linear 相当**：讨论强调使用 **UNET** 进行因果建模产生的性能与稠密线性模型相当，引发了开发者的兴趣。
   - 这表明模型架构调整的新途径，可能增强语言处理的效率。
- **统一多模态模型的愿景**：一名成员提出了 **Transfusion+GameNGen** 的想法，设想将语言、视觉、音频甚至游戏引擎集成到一个单一框架中。
   - 这种模型的意义可能会从根本上重塑各种模态与 AI 应用之间的交互。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.arxiv.org/abs/2408.11039">Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model</a>：我们介绍了 Transfusion，这是一种在离散和连续数据上训练多模态模型的方法。Transfusion 将语言建模损失函数（Next Token Prediction）与扩散技术相结合...</li><li><a href="https://x.com/mattshumer_/status/1831767014341538166?t=DbIKb0tk5JYIwYIMQVB8sQ&s=19">Matt Shumer (@mattshumer_) 的推文</a>：我很高兴宣布 Reflection 70B，全球顶尖的开源模型。使用 Reflection-Tuning 训练，这是一种旨在让 LLM 修复自身错误的技术。405B 版本下周发布...</li><li><a href="https://x.com/kimmonismus/status/1831772661296345333?t=DbIKb0tk5JYIwYIMQVB8sQ&s=19">Chubby♨️ (@kimmonismus) 的推文</a>：我简直不敢相信我读到的内容：一个能修复自身 bug、自我纠正并在所有基准测试中击败包括 GPT-4o 在内的所有当前模型的 LLM？而且该模型仍然是开源的？...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1281057533280124999)** (8 messages🔥): 

> - `Bounty Payments`
> - `Tinyboxes Rental Model`
> - `Pricing Models for Performance` 


- **赏金支付已完成**：所有通过电子邮件领取赏金的人员应该都已获得支付，并公开征询是否有人尚未收到款项。
   - 这确保了管理用户奖励的透明度和效率。
- **创新的 Tinyboxes 租赁概念**：分享了一项制造 **tinyboxes** 的提案，这些设备既可以出售，也可以从数据中心租出，并提供硬件升级路径。
   - 该概念侧重于出售过时硬件以维持新鲜库存，从而进行持续租赁。
- **性能指标定价**：围绕定价模型展开了讨论，建议以 **$/exaflops** 和 **$/tflops*month** 来表示成本。
   - 这一讨论突显了用户定价结构的复杂性和不同考量。
- **内存带宽考虑的复杂性**：对话指出，在定价时假设固定的 flop 与内存带宽比率会带来复杂性。
   - 成员们提到了划分 GPU 以使性能比率匹配的挑战，表明需要更清晰的指南。
- **内存带宽对推理的影响**：指出 **memory bandwidth** 的考量对于在自有硬件上进行 **bs=1 inference** 的用户尤为关键。
   - 这突显了用户根据其特定用例和工作负载需求而产生的不同需求。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1281357362237669386)** (6 条消息): 

> - `IR 中的 phi 操作`
> - `UOps.UPDATE`
> - `cstyle renderer 见解` 


- **对 IR 中 phi 操作的困惑**：一名成员询问了 **IR** 中 **phi 操作** 的工作原理，并将其与通常位于循环体开头的 LLVM IR 进行了比较。
   - 随后另一名成员进行了澄清，解释说它并非真正的 phi 操作，并建议将其重命名为 **ASSIGN**。
- **关于 Cstyle Renderer 的见解**：George Hotz 建议查看 **cstyle renderer** 以了解与其讨论相关的功能。
   - 最初的询问者接受了这一建议，并表示打算对其进行研究。
- **phi 的替代命名建议**：另一名成员建议该操作也可以称为 **UOps.UPDATE**，以更好地反映其用途。
   - 这一贡献为 IR 实现中正在进行的命名规范讨论增添了内容。



**提到的链接**：<a href="https://mesozoic-egg.github.io/tinygrad-notes/uops.html">Kernel Fusion part 3: the linear layer UOps</a>：关于 tinygrad 的教程

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1281178640855597076)** (7 条消息): 

> - `Unsloth Phi 到 Llama 的转换`
> - `Phi3 面临的挑战`
> - `用于快速迭代的小模型`
> - `Axolotl 中的 Dora 支持` 


- **Unsloth Phi 成功转换为 Llama**：据指出，目前存在一种将架构转换为 **Llama** 的 **Unsloth Phi**，从而可以使用 **Llama3 配置**。
   - 这种调整为实验提供了一个潜在更高效的设置。
- **讨论强调了 Phi3 的挑战**：成员们指出，虽然 **Phi3** 应该是可以安全使用的，但 **Discord 历史记录** 中仍有关于其相关挑战的持续讨论。
   - 这种担忧表明，虽然它可以使用，但仍可能出现需要进一步调查的问题。
- **Invisietch 寻找用于实验的小模型**：**Invisietch** 正在寻找一个小模型来进行快速迭代实验，强调了对易获取资源的需求。
   - 这反映了在寻找敏捷开发高效解决方案方面的广泛兴趣。
- **确认 Axolotl 支持 Dora**：已确认 **Axolotl** 通过传递参数 `peft_use_dora: true` 来支持 **Dora**。
   - 此信息已记录在 [GitHub issue](https://github.com/axolotl-ai-cloud/axolotl/issues/1328) 中，该 issue 还鼓励在提出功能请求前先搜索类似内容。



**提到的链接**：<a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1328">DoRA Support · Issue #1328 · axolotl-ai-cloud/axolotl</a>：⚠️ 请检查此功能请求之前是否已被提出。我搜索了 Discussions 中之前的 Ideas，没有找到任何类似的功能请求。我搜索了之前的 Issues，也没有……

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1281295639690350695)** (5 messages): 

> - `Llama-3.1-8B fine-tuning`
> - `Chemical Language Model`
> - `Molecule generation`
> - `DPO optimization`
> - `SmileyLlama` 


- **Llama-3.1-8B 转型为分子设计引擎**：通过 **fine-tuning** 和 **DPO**，成功将 **Llama-3.1-8B** 转化为一个强大的模型，能够根据指定的属性生成分子，展示了其在分子设计方面的能力。
   - 该技术允许用户通过提供一些关于所需特性的提示，按需生成分子。
- **SFT 和 DPO 创造了革命性的 Chemical Language Model**：一项研究表明，当使用 **supervised fine-tuning (SFT)** 和 **direct preference optimization (DPO)** 进行训练时，**Large Language Model (LLM)** 可以作为 **Chemical Language Model (CLM)** 发挥作用。
   - 这种方法使 **LLM** 能够生成与药物开发相关的分子，其性能可与仅依赖化学数据的 **CLM** 相媲美。
- **对新分子设计能力的兴奋**：*太酷了！*
   - 社区成员对这个微调模型的潜力表示了极大的热情，并考虑在社交媒体上广泛分享。
- **SmileyLlama 在社交媒体上首次亮相**：该模型被称为 **SmileyLlama**，是一个旨在根据属性提示词创建分子的 **Chemical Language Model**，已在 X 上引起关注。
   - Axolotl 账号的一篇帖子强调，它在使用 **Axolotl** 框架构建的同时，性能与其它纯 **CLM** **并驾齐驱**。
- **即将开放测试**：大家对 **HF** 模型的到来充满期待，届时成员们可以直接体验这个经过微调的 **Llama** 模型。
   - 这是继最近在利用 **Llama** 处理化学任务方面取得进展后的又一举措，预示着该领域正向更广泛的可访问性迈进。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/axolotl_ai/status/1831771214445945148">来自 Axolotl (@axolotl_ai) 的推文</a>：SmileyLlama，一个经过微调的 Chemical Language Model，用于根据提示词中指定的属性设计分子。这是一个基于 SFT+DPO 的模型，性能与其它纯 CLM 相当，但是使用 Axolotl 构建的。</li><li><a href="https://arxiv.org/abs/2409.02231">SmileyLlama: Modifying Large Language Models for Directed Chemical Space Exploration</a>：在这里我们展示了 Large Language Model (LLM) 可以作为 Chemical Language Model (CLM) 的基础模型，其表现达到或超过了仅在化学 SMILES 字符串上训练的 CLM 水平...
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1281275026968023143)** (2 messages): 

> - `DSPy usecase list`
> - `Livecoding sessions` 


- **DSPy 用例列表发布**：**DSPy usecase list** 已正式公布，旨在探索人们正在使用 **Large Models (LMs)** 构建什么以及在生产环境中部署什么。目前已编制了一份包含近 **100 个产品**和 **OSS** 系统的初步列表，详情见 [推文](https://x.com/isaacbmiller1/status/1831715783556395369) 和链接文档。
   - 该倡议由 @isaacbmiller1 和 @lateinteraction 发起，旨在通过 **DSPy** 的视角收集见解。
- **Livecoding 活动公告**：分享了一个关于当前在指定 Discord 频道进行的 **livecoding** 环节的提醒。参与者可以通过 [此链接](https://discord.com/channels/1161519468141355160/1161519469777133580) 加入。
   - 该活动旨在促进社区内的动手编程体验。



**提及的链接**：<a href="https://x.com/isaacbmiller1/status/1831715783556395369)">来自 isaac 🧩 (@isaacbmiller1) 的推文</a>：人们正在用 LM 构建什么？他们在生产中部署了什么？@lateinteraction 和我想开始通过 DSPy 的视角来回答这个问题。我们编制了一份包含近 100 个项目的初步列表...

  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

batmanosama: https://huggingface.co/papers/2409.02889
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1281172588298637314)** (1 messages): 

> - `ColPali`
> - `Visual Document Retrieval Benchmark` 


- **ColPali 彻底改变文档检索**：发布了一种名为 **ColPali** 的新方法，利用延迟交互（late interaction）机制增强了文档检索。根据[这篇博文](https://www.lycee.ai/blog/colpali-efficient-document-retrieval)，该方法对于视觉丰富的文档非常高效。
   - 由 **Manuel Faysse** 和 **Hugues Sibille** 等人组成的团队设计，ColPali 通过利用表格和插图等非文本元素，克服了现有系统的局限性。
- **推出视觉文档检索基准 (Visual Document Retrieval Benchmark)**：论文介绍了 **Visual Document Retrieval Benchmark (ViDoRe)**，用于评估跨多种语言、领域和文档类型的检索性能。
   - 该基准旨在通过整合除文本之外更广泛的文档元素，来增强对检索系统的评估。



**提及的链接**：<a href="https://www.lycee.ai/blog/colpali-efficient-document-retrieval">ColPaLi: Efficient Document Retrieval with Contextualized Language Model</a>：ColPaLi 是一种新的文档检索系统，利用视觉语言模型 (VLMs) 高效处理视觉丰富的文档。通过结合视觉和文本信息，ColPaLi 的表现优于现有的...

  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1281259287263314012)** (2 messages): 

> - `Multimodal LLMs`
> - `Training/Finetuning` 


- **关于多模态 LLM 经验的咨询**：一位成员询问是否有人具有使用文本和语音作为输入的 **multimodal LLMs** 的经验，特别是在训练或 Finetuning 方面。
   - 这反映了将 **语音能力 (speech capabilities)** 集成到 LLM 框架中的日益增长的兴趣。
- **关于多模态模型的 YouTube 资源**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=GUdoNdTNNaU)，推测与多模态 LLM 相关，暗示了关于该主题的有用见解。
   - 对于那些对多模态模型运营化感兴趣的人来说，这可能是一个很好的起点。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1281062678428909672)** (1 messages): 

> - `Meeting Transcription`
> - `Agent Workflows`
> - `Evaluation Challenges` 


- **会议与会者转录**：讨论强调了对**整个会议转录**的需求，包括所有与会者的姓名。
   - *这可以提高未来讨论的参考准确性和问责制*。
- **报告的概念验证 (Proof of Concept)**：一名参与者正在进行**一份报告的概念验证**，表明其项目采用了聚焦的方法。
   - *这在保持范围可控的同时，向实际落地迈进*。
- **对 Agent 工作流的担忧**：考虑在项目中使用 **Agent 工作流**，这暗示了一种创新的方法。
   - *然而，由于缺乏既定标准，人们对评估 Agent 的复杂性感到担忧*。


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1280997763911385182)** (1 messages): 

> - `AI Enterprise Summit`
> - `San Francisco event`
> - `Keynote speakers`
> - `Networking opportunities` 


- **AI 企业峰会即将在旧金山开幕**：**AI Enterprise Summit** 定于 **2024 年 10 月 2 日**在**旧金山**举行，旨在为高管、企业家和 AI 爱好者提供聚集并讨论扩展 AI 产品的机会。
   - _使用代码 AIR50 可享受 50 美元的特别优惠_，购买这一专属单日活动的门票。
- **峰会的主讲嘉宾**：峰会将邀请行业领袖，包括 **Paul Baier**（GAInsights CEO）、**Ted Shelton**（Inflection AI COO）和 **Jeremiah Owyang**（Blitzscaling Ventures）等。
   - 这些演讲者将基于真实的业务用例提供见解，增强所有参与者的学习体验。
- **为 AI 专业人士精心策划的聚会**：此次活动承诺将是一场雄心勃勃的高管和 AI 专业人士的**精心聚会**，提供社交和学习的机会。
   - 参与者将与思想领袖互动，并探索 AI 产品开发的各个方面。



**提及的链接**：<a href="https://lu.ma/airsummit">AI Realized – The Enterprise AI Summit · Luma</a>：Christina Ellwood 和 David Yakobovitch 呈现... 2024 年 AI Realized 峰会，面向企业高管、企业家和 AI 创新者。加入我们在旧金山的活动...

  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/)** (1 条消息): 

huanzhimao: 感谢提交 issue！我会查看一下。
  

---



---



---



{% else %}


> 完整的逐个频道详细分析已针对邮件进行了截断。 
> 
> 如果您想查看完整的详细分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}