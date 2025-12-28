---
companies:
- cohere
- mistral-ai
- hugging-face
date: '2025-03-18T00:28:53.405655Z'
description: Cohere 的 **Command A** 模型巩固了其在 LMArena 排行榜上的地位。该模型拥有 **111B** 参数并采用开放权重，具备
  **256K** 的超长上下文窗口，且价格极具竞争力。**Mistral AI** 发布了轻量级、多语言且多模态的 **Mistral AI Small 3.1**
  模型，该模型针对单张 RTX 4090 或 32GB 内存的 Mac 配置进行了优化，在指令遵循和多模态基准测试中表现强劲。新型 OCR 模型 **SmolDocling**
  提供快速的文档读取能力且显存（VRAM）占用较低，性能超越了 Qwen2.5VL 等更大型的模型。相关讨论强调了系统级改进比单纯的 LLM 进步更为重要，同时
  **MCBench** 被推荐为评估模型在代码、审美和意识等方面能力的优选 AI 基准。
id: 5a1c6a0b-4f1a-4a34-88de-3563bb82d098
models:
- command-a
- mistral-ai-small-3.1
- smoldocling
- qwen-2.5-vl
original_slug: ainews-coheres-command-a-claims-3-open-model-spot
people:
- aidangomez
- sophiamyang
- mervenoyann
- aidan_mclau
- reach_vb
- lateinteraction
title: Cohere 的 Command A 占据开放模型第三位（仅次于 DeepSeek 和 Gemma）
topics:
- context-windows
- multilinguality
- multimodality
- fine-tuning
- benchmarking
- ocr
- model-performance
- model-releases
- model-optimization
---

<!-- buttondown-editor-mode: plaintext -->**为开放权重模型（open weights models）欢呼！**

> 2025年3月14日至2025年3月17日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 社区（**223** 个频道，**9014** 条消息）。预计节省阅读时间（以 200wpm 计算）：**990 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们在[上周](https://buttondown.com/ainews/archive/ainews-not-much-happened-today-8188/)简要提到了 Cohere 的 Command A 发布，但由于当时的公告中缺乏广泛可比的基准测试（虽然有一些，但那些选择性的、自报的与 DeepSeek V3 和 GPT-4o 的对比，并不能真正将 Command A 置于 SOTA 开源模型或同尺寸 SOTA 模型的大背景下），因此很难判断其长远影响力的排名。

随着今天 LMArena 结果的公布，这一点已不再存疑：


![image.png](https://assets.buttondown.email/images/73c613a0-d833-4e35-89ac-af1cb6132bc4.png?w=960&fit=max)


正如 [Aidan Gomez 指出的](https://x.com/aidangomez/status/1901669060175151609)，在使用了 Style Control 修正后，Command A 的排名实际上*上升*了 2 位（在[他们的 LS podcast](https://www.latent.space/p/lmarena) 中有详细探讨）。

还有许多其他值得注意的细节，使得 Command A 成为开源模型库中极具吸引力的候选者，包括异常长的 256k 上下文窗口（context window）、多语言能力，以及专注于优化 2-H100 推理占用空间（serving footprint）。


![image.png](https://assets.buttondown.email/images/94417e9c-d2c5-4c51-ace1-67d561e12667.png?w=960&fit=max)




---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

**大语言模型 (LLMs) 与模型发布**

- **Mistral AI Small 3.1 发布（多模态、多语言、Apache 2.0 许可证）**：[@sophiamyang](https://twitter.com/sophiamyang/status/1901675671815901688) 宣布发布 **Mistral AI Small 3.1**，强调其轻量级特性（可在单张 RTX 4090 或 32GB RAM 的 Mac 上运行）、快速响应对话、低延迟函数调用（function calling）、专门的微调以及先进的推理基础。它在指令（instruct）基准测试中优于同类模型 [@sophiamyang](https://twitter.com/sophiamyang/status/1901676305025774020)，在多模态指令基准测试中也表现出色 [@sophiamyang](https://twitter.com/sophiamyang/status/1901676575965282395)，并已在 Hugging Face [@sophiamyang](https://twitter.com/sophiamyang/status/1901677007278092508)、Mistral AI La Plateforme [@sophiamyang](https://twitter.com/sophiamyang/status/1901677125918134276) 以及企业级部署中上线 [@sophiamyang](https://twitter.com/sophiamyang/status/1901677325588078774)。该模型因其多语言和长上下文能力而受到赞誉 [@sophiamyang](https://twitter.com/sophiamyang/status/1901676699361882439)。[@reach_vb](https://twitter.com/reach_vb/status/1901670885188071545) 强调了其 128K 上下文窗口和 Apache 2.0 许可证。
- **SmolDocling：新型 OCR 模型**：[@mervenoyann](https://twitter.com/mervenoyann/status/1901668060257190186) 介绍了 **SmolDocling**，这是一款快速 OCR 模型，使用 0.5GB VRAM 仅需 0.35 秒即可读取单份文档，性能优于包括 Qwen2.5VL 在内的大型模型。它基于 SmolVLM，并在页面和 Docling 转录数据上进行了训练。该模型和演示已在 Hugging Face 上提供 [@mervenoyann](https://twitter.com/mervenoyann/status/1901668064602579150)。
- **Cohere Command A 模型**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1901668148031758605) 报告称，**Cohere 的 Command A** 已攀升至 Arena 排行榜第 13 位，突出了其开放权重模型（111B）、256K 上下文窗口以及 $2.5/$10 的输入/输出 MTok 定价。Command A 在风格控制（style control）方面也表现优异 [@aidangomez](https://twitter.com/aidangomez/status/1901669060175151609)。
- **关于更优 LLM 的讨论**：[@lateinteraction](https://twitter.com/lateinteraction/status/1901642081770295732) 表达了一种悲观观点，认为近期 LLM 的进步归功于构建 LLM 系统（CoT），而非 LLM 本身的提升，并质疑更好的 LLM 究竟在哪里。

**模型性能、基准测试与评估**

- **MCBench 作为卓越的 AI 基准测试**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1901671231713427512) 推荐 mcbench 为最佳 AI 基准测试，指出其数据易于审计、测试了相关特性（代码、审美、意识），并能区分顶级模型之间的性能差异。该基准测试可以在 https://t.co/YEgzhLotKk 找到 [@aidan_mclau](https://twitter.com/aidan_mclau/status/1901671234125095205)
- **用于自主软件任务的 HCAST 基准测试**：[@idavidrein](https://twitter.com/idavidrein/status/1901647558839353363) 分享了关于 **HCAST (Human-Calibrated Autonomy Software Tasks)** 的细节，这是由 METR 开发的一个基准测试，旨在衡量前沿 AI 系统自主完成多样化软件任务的能力。
- **专利领域的 AI 模型**：[@casper_hansen_](https://twitter.com/casper_hansen_/status/1901540769040683214) 测试了模型在专利指令遵循方面的表现，发现 Mistral Small 3 优于 Gemini Flash 2.0，因为 Mistral 模型在更多的专利数据上进行了预训练。
- **LLM 中的泛化缺陷**：[@JJitsev](https://twitter.com/JJitsev/status/1901467121592201490) 分享了他们论文的更新，包括关于近期推理模型的部分，质疑它们处理 AIW 问题变体的能力，这些变体揭示了 SOTA LLM 中严重的泛化缺陷。
- **在 OpenRouter 上评估模型**：[@casper_hansen_](https://twitter.com/casper_hansen_/status/1901539872315257286) 指出 OpenRouter 是测试新模型的有用工具，但免费额度限制为每天 200 次请求。

**AI Agents、工具使用与应用**

- **AI Agent 与外部工具交互**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1901403297988374853) 解释了 AI Agent 如何通过基于 UI 和基于 API 的交互与外部工具或应用进行交互，现代 AI Agent 框架因速度和可靠性而优先考虑基于 API 的工具。
- **TxAgent：用于治疗推理的 AI Agent**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1901555282594971761) 介绍了 **TXAGENT**，这是一个利用多步推理和实时生物医学知识检索的 AI Agent，通过包含 211 个工具的工具箱来分析药物相互作用、禁忌症和针对特定患者的治疗策略。
- **Realm-X 助手**：[@LangChainAI](https://twitter.com/LangChainAI/status/1901699861264900112) 重点介绍了 **AppFolio 的 Realm-X 助手**，这是一个由 LangGraph 和 LangSmith 驱动的 AI Copilot，旨在简化物业管理者的日常任务。将 Realm-X 迁移到 LangGraph 使响应准确度提高了 2 倍。
- **用于错误和数据分析的 AI**：[@gneubig](https://twitter.com/gneubig/status/1901679380205609324) 对 AI Agent 能够比人类更快地执行更细致的错误分析和数据分析的能力表示兴奋。
- **多 Agent 协作结对编程**：[@karinanguyen_](https://twitter.com/karinanguyen_/status/1901667981631086915) 分享了一个多 Agent/玩家结对编程的概念草图，设想了一种与 AI 进行实时协作的体验，包括屏幕共享、群聊和 AI 辅助编码。

**AI 安全、对齐与审计**

- **对齐审计**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1901543811966529764) 重点介绍了 Anthropic 的一篇关于审计语言模型隐藏目标的新论文，详细说明了团队如何利用可解释性、行为攻击和训练数据分析揭示模型的隐藏目标。
- **默认对齐**：[@jd_pressman](https://twitter.com/jd_pressman/status/1901437803621392519) 反对“默认对齐”的概念，强调 LLM 的对齐是通过在人类数据上训练实现的，这在 RL 或合成数据方法中可能并不成立。

**迷因/幽默**

- **RLHF 训练**：[@cto_junior](https://twitter.com/cto_junior/status/1901462916672712881) 开玩笑说他们被 RLHF 了，并附上了一个推文链接。
- **PyTorch 缓存分配器**：[@typedfemale](https://twitter.com/typedfemale/status/1901463667780268179) 分享了一个关于解释 PyTorch 缓存分配器行为的梗图。
- **可卡因 vs RL**：[@corbtt](https://twitter.com/corbtt/status/1901706359231705198) 开玩笑说，RL 训练的 Agent 领悟新技能带来的快感比可卡因还要强烈。


---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1：使用 SDXL、Wan2.1 和长上下文微调的高级 AI 视频生成**

- **[另一个旨在追求电影级写实感的视频，这次使用了一个难度大得多的角色。SDXL + Wan 2.1 I2V](https://v.redd.it/t88g56krqnoe1)** ([评分: 1018, 评论: 123](https://reddit.com/r/StableDiffusion/comments/1jb47bs/another_video_aiming_for_cinematic_realism_this/)): 该帖子讨论了使用 **SDXL** 和 **Wan 2.1 I2V** 创建旨在实现**电影级写实感 (cinematic realism)** 的视频。它强调了在这种背景下处理更复杂角色的挑战。
  - **技术挑战与技巧**：**Parallax911** 分享了使用 **SDXL** 和 **Wan 2.1 I2V** 实现**电影级写实感**的复杂性，重点介绍了如何使用 **Photopea** 进行 **inpainting** 以及在 **Davinci Resolve** 中进行合成。他们提到了在实现一致性和写实感方面的困难，尤其是对于复杂的角色设计，并使用了 **Blender** 来制作如开门等片段的动画。
  - **项目成本与工作流**：该项目使用 **RunPod** 的 **L40S**（价格为 **$0.84/小时**），耗时约 **80 小时** 的 GPU 时间，成本约为 **$70**。**Parallax911** 采用的工作流包括 **RealVisXL 5.0**、**Wan 2.1** 和用于放大的 **Topaz Starlight**，生成的场景为 **61 帧**、**960x544** 分辨率和 **25 steps**。
  - **社区反馈与建议**：社区赞扬了其氛围渲染和声音设计，并对水滴大小等元素提出了具体反馈，同时希望能有教程。一些用户建议改进 AI 与传统技术的结合，并对 **Metroid** 中的 **Samus Aran** 等角色的动作场面表现出兴趣。


- **[Wan2.1 中的视频扩展 - 完全在 ComfyUI 中创建 10 秒以上的高清放大视频](https://v.redd.it/xi58u5d3qmoe1)** ([评分: 123, 评论: 23](https://reddit.com/r/StableDiffusion/comments/1jb0h7i/video_extension_in_wan21_create_10_seconds/)): 该帖子讨论了在 **Wan2.1** 中使用 **ComfyUI** 创建放大视频的**高度实验性工作流**，成功率约为 **25%**。该过程涉及从初始视频的最后一帧生成新视频、合并、放大和帧插值，具体参数包括 **Sampler: UniPC**、**Steps: 18**、**CFG: 4** 和 **Shift: 11**。更多详情可以在 [工作流链接](https://civitai.com/models/1297230?modelVersionId=1531202) 中找到。
  - 用户正在询问工作流中的**宽高比**处理，质疑它是自动设置的还是需要手动调整输入图像。
  - 对该工作流感兴趣的用户给出了**积极反馈**，表示对这种解决方案的期待。
  - 用户提出了关于片段后半部分**模糊**的担忧，并建议这可能与输入帧的质量有关。


- **[使用 WAN 2.1 和 LTX 动画化了我的一些 AI 图片](https://v.redd.it/z5r0kyf1smoe1)** ([评分: 115, 评论: 10](https://reddit.com/r/StableDiffusion/comments/1jb0n50/animated_some_of_my_ai_pix_with_wan_21_and_ltx/)): 该帖子讨论了使用 **WAN 2.1** 和 **LTX** 创建 **AI 动画视频**。在没有更多背景或额外细节的情况下，重点仍然是用于动画的工具。
  - **模型使用**：第一个片段（跳跃的女人）和战斗机使用了 **LTX**，而奔跑的宇航员、恐怖菲比娃娃和龙则使用了 **WAN**。
  - **硬件详情**：视频是使用从 **Paperspace** 租用的带有 **RTX5000** 实例的云端计算机生成的。


**主题 2. OpenAI 的 Sora：将城市景观转变为反乌托邦**

- **[OpenAI 的 Sora 将旧金山的 iPhone 照片变成了反乌托邦噩梦](https://v.redd.it/y67d5ph47loe1)** ([评分: 931, 评论: 107](https://reddit.com/r/ChatGPT/comments/1jawa6c/openais_sora_turns_iphone_photos_of_san_francisco/)): **OpenAI** 的 **Sora** 是一款将**旧金山**的 **iPhone 照片**转化为具有**反乌托邦 (dystopian)** 美感图像的工具。尽管由于缺乏文本内容而无法获得具体细节，但该帖子可能讨论了使用 AI 改变现实世界图像的影响和视觉结果。
  - 几位评论者对 **AI 生成的反乌托邦图像**的影响表示怀疑，一些人认为**旧金山**或其他城市的实际地点已经看起来像这些反乌托邦视觉效果，质疑 AI 改造的必要性。
  - 使用 **iPhone** 作为拍摄原始图像的设备是一个争论点，一些人质疑其与讨论的相关性，而另一些人则强调其在理解图像来源方面的重要性。
  - 对话中混杂着对 **AI 能力**的钦佩和担忧，用户既对技术感到惊讶，又对未来难以区分 AI 生成和现实世界图像感到焦虑。

- **[OpenAI 的 Sora 将旧金山的 iPhone 照片变成了反乌托邦式的地狱景象……](https://v.redd.it/ukxvzsatzkoe1)** ([Score: 535, Comments: 58](https://reddit.com/r/OpenAI/comments/1javmkq/open_ais_sora_transformed_iphone_pics_of_san/)): **OpenAI 的 Sora** 将 **旧金山的 iPhone 照片** 变成了反乌托邦式的地狱景象，展示了其在改变数字图像以创造未来主义、阴郁美学方面的能力。该帖子除了这一转变外，缺乏额外的背景或细节。
  - 评论者将 **反乌托邦图像** 与现实世界的地点进行了类比，提到了 **德里**、**底特律** 和 **印度街道**，突显了 AI 在解读城市环境时被察觉到的偏见。
  - 有人对 **AI 的文本生成能力** 表示担忧，一位评论者指出，图像中的 **标牌文字** 是 AI 操纵的明显迹象。
  - 用户对 **创建此类图像的过程** 表现出兴趣，并请求提供 **分步说明**，以便在自己的照片上复制这种转变。


**主题 3. OpenAI 与 DeepSeek：开源对决**

- **[我认为太多的不安全感](https://i.redd.it/9xpl7abaoooe1.jpeg)** ([Score: 137, Comments: 58](https://reddit.com/r/ClaudeAI/comments/1jb8aj5/i_think_too_much_insecurity/)): **OpenAI** 指责 **DeepSeek** 受“国家控制”，并主张禁止中国 AI 模型，突显了对 AI 发展中政府影响的担忧。图片暗示了地缘政治背景，美国和中国国旗象征着关于 AI 技术中国家控制和安全的更广泛辩论。
  - 讨论突显了对 **OpenAI** 针对 **DeepSeek** 指控的怀疑，用户通过指出 **DeepSeek** 的模型是开源的来挑战国家控制的观点。用户质疑指控的有效性，要求提供证据，并引用了 **Sam Altman** 过去关于 **LLM** 缺乏竞争护城河的言论。
  - **DeepSeek** 被视为一个强劲的竞争对手，能够以较低的支出运营，并可能影响 **OpenAI** 的利润。一些评论认为 **DeepSeek** 的行为被视为一种经济侵略，等同于对美国利益的宣战。
  - 存在着针对 **OpenAI** 和 **Sam Altman** 的强烈批评暗流，用户对他们的行为和言论表示不信任和不满。对话包括人身攻击以及对 **Altman** 公信力的怀疑，并提到了他关于开源模型的承诺尚未兑现。


- **构建了一个 AI Agent 来自动寻找并申请工作** ([Score: 123, Comments: 22](https://reddit.com/r/OpenAI/comments/1jb49lo/built_an_ai_agent_to_find_and_apply_to_jobs/)): 一个名为 **SimpleApply** 的 AI Agent 通过将用户的技能和经验与相关的职位匹配，实现了职位搜索和申请流程的自动化，提供三种使用模式：带职位评分的手动申请、选择性自动申请，以及针对匹配度超过 **60%** 的职位的全自动申请。该工具旨在简化职位申请流程而不至于让雇主应接不暇，并因发现了许多用户可能无法发现的远程工作机会而受到称赞。
  - 有人提出了关于 **数据隐私和合规性** 的担忧，询问 **SimpleApply** 如何处理 **PII** 以及是否遵守 **GDPR** 和 **CCPA**。开发者澄清说，他们与合规的第三方安全地存储数据，并正在制定明确的用户协议以实现完全合规。
  - 讨论了 **申请垃圾邮件风险**，并建议避免重复申请同一职位，以防止被 **ATS** 系统标记。开发者保证，该工具仅申请获得面试可能性较高的职位，以尽量减少垃圾邮件。
  - 建议了替代的 **定价策略**，例如仅在用户通过电子邮件或呼叫转移收到回访时收费。这种方法对于犹豫是否要预先花钱的失业用户可能更具吸引力。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. 对用于确定 LLM 智能的“陷阱”测试的批评**

- **[当 ChatGPT 成为我的治疗师](https://reddit.com/r/ChatGPT/comments/1jd0402/when_chatgpt_became_my_therapist/)**（[得分：172，评论：83](https://reddit.com/r/ChatGPT/comments/1jd0402/when_chatgpt_became_my_therapist/)）：在情绪低落时，作者发现 **ChatGPT** 出乎意料地令人感到安慰且富有同理心，能提供深思熟虑的提问和自我关怀的提醒。他们承认，虽然 AI 聊天机器人不能替代真实的心理治疗，但它们可以提供宝贵的情感支持，尤其是在应对压力、焦虑和进行自我反思时。
  - 许多用户发现 **ChatGPT** 对情感支持大有裨益，可作为**自我反思**的工具并提供**治疗性引导**。一些用户（如 **Acrobatic-Deer2891** 和 **Fair_Cat5629**）反馈称，治疗师对 AI 提供的引导给出了正面评价；而另一些用户（如 **perplexed_witch**）则强调将其用于“引导式自我反思”，而非替代治疗。
  - **ChatGPT** 在危机期间的**心理健康管理**作用受到称赞，它能提供一个不带偏见的倾诉空间并提供视角，正如 **dinosaur_copilot** 和 **ChampionshipTall5785** 的评论所言。用户赞赏它在痛苦时刻提供可操作建议和情感支持的能力。
  - 用户也提到了对隐私以及 AI 作为**治疗替代品**局限性的担忧，例如 **acomfysweater** 对数据存储表示忧虑。尽管存在这些担忧，包括 **Jazzlike-Spare3425** 在内的许多人仍看重 AI 提供支持的能力，且不会给人类倾听者带来情感负担。


- **[为什么...👀](https://i.redd.it/s2cqynsnp8pe1.jpeg)**（[得分：3810，评论：95](https://reddit.com/r/ChatGPT/comments/1jdazz6/why/)）：**ChatGPT 在治疗角色中的潜力**通过一段幽默的对话得到了体现：一名用户要求 ChatGPT 模拟女朋友，引发了一场俏皮的交流，并以一句分手台词结束。这种互动突显了 AI 在聊天界面中进行轻松、类人对话的能力。
  - **ChatGPT 的效率与能力**：用户幽默地评论了 ChatGPT 快速完成请求的能力，有人开玩笑说它的回复是基于 "Andrew Tate Sigma Incel 数据" 训练的，并创造了 **"ChadGPT"** 一词来描述其高效但生硬的互动风格。
  - **Prompt Engineering 与个性化**：一位具有心理学和技术背景的用户建议，ChatGPT 可以根据它选择保存的记忆形成某种语调，这意味着通过 Prompt Engineering 实现个性化互动是可能的。他们还讨论了神经网络与人类记忆检索系统（如 **RAG**）的相似性。
  - **幽默与讽刺**：对话的趣味性在评论中得到体现，有人拿 AI 在人际关系中的角色开玩笑，称其为“高级的单词预测器”，并对其模拟类人互动（包括模拟分手）的能力进行了幽默观察。


**主题 2. 对 Google DeepMind CEO 预测 AGI 将在 5-10 年内实现的反应**

- **[Google DeepMind CEO 表示，能在任何任务上媲美人类的 AI 将在 5 到 10 年内出现](https://www.cnbc.com/2025/03/17/human-level-ai-will-be-here-in-5-to-10-years-deepmind-ceo-says.html)**（[得分：120，评论：65](https://reddit.com/r/OpenAI/comments/1jdfmzc/ai_that_can_match_humans_at_any_task_will_be_here/ )）：**DeepMind CEO** 预测 AI 将在 **5-10 年**内实现跨任务的人类水平对等，这标志着此前关于明年实现这一里程碑的预期发生了转变。
  - 评论者讨论了 AI 实现人类水平对等的**时间线预测**，一些人对不断变化的时间线表示怀疑，指出 **Demis Hassabis** 一直在预测 AGI 的时间范围是 **5-10 年**。人们呼吁对 "AGI" 给出更清晰的定义，以便更好地理解这些预测。
  - **AI 的大规模普及**被比作历史性的技术变革，如从马车到汽车的过渡以及智能手机的普及。这种类比表明，随着时间的推移，AI 将变得无处不在，改变社会规范和期望，而不会立即引起剧烈反应。
  - 人们对 AI 的**经济和社会影响**表示担忧，特别是关于就业和财富集中的问题。一些评论者对 AI 可能加剧职位取代和不平等表示忧虑，而另一些人则质疑 AI 公司在面临潜在风险时仍推动快速发展的动机。


**主题 3. OpenAI 关于使用受版权保护内容的争议性请求正由美国政府审议**

- **[OpenAI to U.S. Government - Seeking Permission to Use Copyrighted Content](https://i.redd.it/o5k9b30qg6pe1.jpeg)** ([Score: 506, Comments: 248](https://reddit.com/r/ChatGPT/comments/1jd4ktc/openai_to_us_government_seeking_permission_to_use/)): **OpenAI** 正在请求 **Trump administration** 放宽版权监管，以促进在 AI 开发中使用受保护的内容。该公司强调，此类变革对于维持 **America's leadership** 在 AI 领域的地位至关重要。
  - 评论者讨论了 **copyright law** 对 AI 开发的影响，一些人认为 AI 对版权内容的使用应被视为 **fair use**，类似于人类从现有作品中学习的方式。人们担心 AI 模型可能会绕过个人面临的法律后果，突显了在获取和使用版权材料方面的不平等。
  - **AI arms race** 是一个反复出现的主题，几位用户表示担心 **China and other countries** 可能不会像美国那样严格遵守版权法，这可能会给他们带来优势。这引发了关于 AI 开发竞争格局和美国公司战略决策的问题。
  - 关于版权所有者的 **equity and compensation** 的讨论提出了替代方案，例如向作品被用于 AI 训练的创作者提供股权。一些评论者建议将大科技公司国有化，以确保 AI 进步带来的利益得到公平分配，反映了对财富分配和 AI 资源控制的更广泛担忧。


- **[Open AI to U.S. GOVT: Can we Please use copyright content](https://i.redd.it/qoqfyliwg6pe1.jpeg)** ([Score: 398, Comments: 262](https://reddit.com/r/OpenAI/comments/1jd4lfd/open_ai_to_us_govt_can_we_please_use_copyright/)): **OpenAI** 请求 **Trump administration** 放宽 **copyright rules**，以促进 AI 训练并帮助维持美国在该领域的领导地位。请求随附的图片显示了一个正式场合，演讲者站在讲台上，可能是在 **White House**，旁边的人包括貌似 **Donald Trump** 的人士。
  - 许多评论者反对 **OpenAI** 放宽 **copyright rules** 的请求，强调创作者的作品应该获得报酬，而不是在未经许可的情况下被使用。观点认为版权激励了创造力和创新，放宽这些法律可能会使创作者处于劣势，并使 **OpenAI** 等大公司获得不公平的利益。
  - 评论中反复出现对 **OpenAI's motives** 的怀疑，用户暗示 **OpenAI** 正在寻求利用法律漏洞牟利。人们将其与 **China's approach** 对待知识产权的方式进行了比较，一些人担心如果美国严格遵守现行版权法，可能会在 AI 开发方面落后。
  - 几位用户提议，如果 **OpenAI** 或任何公司使用版权材料进行 AI 训练，由此产生的模型或数据应该 **open source** 并供所有人使用。讨论还涉及 AI 在版权材料上进行训练的更广泛伦理影响，以及可能需要重新评估版权法以应对新的技术现实。


**Theme 4. ReCamMaster releases new camera angle changing tool**

- **[ReCamMaster - LivePortrait creator has created another winner, it lets you changed the camera angle of any video.](https://v.redd.it/ikhtm1xu19pe1)** ([Score: 648, Comments: 46](https://reddit.com/r/StableDiffusion/comments/1jdcapy/recammaster_liveportrait_creator_has_created/)): **ReCamMaster** 开发了一项技术，允许用户更改任何视频的摄像机角度，这是继其之前的 **LivePortrait** 取得成功后的又一力作。
  - 许多评论者对 **ReCamMaster** 不是 **open source** 感到失望，并提到了 **TrajectoryCrafter**，它是开源的，并允许类似的摄像机操控功能。**TrajectoryCrafter** 的 **GitHub** 链接在[这里](https://github.com/TrajectoryCrafter/TrajectoryCrafter)。
  - 一些用户预见该技术对视频稳定和沉浸式体验的潜在影响，认为该技术可能会带来更具创新性的电影镜头，并在 **Autonomous Driving** 等领域得到应用。
  - 对于 AI 生成的摄像机角度的真实性存在怀疑，有人建议，要获得更具说服力的结果，需要利用现有的摄像机摇移或源材料中的多个镜头。

- **[在一些我父亲在 80 年代拍摄并由我扫描的胶片投影幻灯片上使用了 WAN 2.1 IMG2VID。](https://v.redd.it/l19pkp2f89pe1)** ([评分: 286, 评论: 24](https://reddit.com/r/StableDiffusion/comments/1jdd1om/used_wan_21_img2vid_on_some_film_projection/)): **WAN 2.1 IMG2VID** 被用于将 20 世纪 80 年代的扫描胶片投影幻灯片转换为视频格式，展示了视频技术的演进。该帖子缺乏关于具体结果或与 **ReCamMaster** 等其他技术对比的额外背景或细节。
  - 评论者对该项目的技术细节表现出浓厚兴趣，要求提供更多关于创建视频转换所使用的 **workflow、hardware 和 prompts** 的信息。人们特别好奇如何为个人项目复制这一过程。
  - 讨论的很大一部分集中在该项目的情感冲击上，用户分享了个人轶事并表达了希望看到原始幻灯片的愿望。一位评论者确认，幻灯片中出现的人已经看到了视频，并对这项技术感到惊讶。
  - 怀旧方面被重点提及，用户回顾了驾驶 **Goodyear blimp** 等历史内容，并对通过这些转换后的视频“穿越回过去”的能力表示热忱。


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 生成的摘要之摘要之摘要

**主题 1. Mistral 与 Google 争夺小模型霸权**

- [**Mistral Small 3.1 展示多模态实力**](https://mistral.ai/news/mistral-small-3-1): **Mistral AI** 推出了 **Mistral Small 3.1**，这是一款多模态模型，声称在其同参数量级中具有 *SOTA* 性能，超越了 **Gemma 3** 和 **GPT-4o Mini**。该模型以 Apache 2.0 协议发布，拥有 **128k context window**，推理速度达 **每秒 150 tokens**，具备处理文本和图像输入的能力。
- [**Gemma 3 获得视觉、上下文和剪枝功能**](https://mistral.ai/news/mistral-small-3-1): **Google** 的 **Gemma 3** 模型正在通过新功能突破界限，包括**视觉理解**、**多语言支持**以及巨大的 **128k token context window**。成员们还探索了将 **Gemma-3-27b** 的词汇量从 **260k** **剪枝**至 **40k tokens**，以减少 VRAM 占用并提升训练速度。
- [**百度的 ERNIE X1 以低成本挑战 DeepSeek R1**](https://x.com/Baidu_Inc/status/1901089355890036897): **百度**发布了新的推理模型 **ERNIE X1**，声称其性能与 **DeepSeek R1** 相当，但*成本仅为一半*。**ERNIE Bot** 现在对个人用户免费，尽管 **X1** 推理模型目前仅限中国地区使用。

**主题 2. 训练与优化技术趋于白热化**

- [**Unsloth 用户发现梯度步长陷阱**](https://discord.com/channels/1179035537009545276/1179035537529643040/1350267964634236930): **UnslothAI** Discord 成员指出，在 fine-tuning 过程中，较小的有效 batch sizes（例如 **batch=1, gradient steps = 4**）可能导致模型*遗忘*过多。用户分享了建议的 batch/grad 配置，以便从有限的 VRAM 中榨取性能。
- [**深度诅咒困扰 LLM，Pre-LN 是罪魁祸首**](https://arxiv.org/abs/2502.05795): 一篇新论文强调了现代 **LLM** 中的**深度诅咒 (Curse of Depth)**，揭示了 **Pre-Layer Normalization (Pre-LN)** 使得近一半的模型层效果不如预期。研究人员提出了 **LayerNorm Scaling** 来缓解这一问题并提高训练效率。
- [**Block Diffusion 模型融合自回归与扩散模型的优势**](https://arxiv.org/abs/2503.09573): 一种新的 **Block Diffusion** 模型在自回归和扩散语言模型之间进行插值，旨在结合两者的优点。该方法寻求将高质量输出、任意长度生成与 KV caching 以及并行化能力相结合。

**主题 3. AI Agent 与 IDE 争夺开发者青睐**

- [**Aider Agent 通过 MCP Server 获得自主性提升**](https://aider.chat/docs/recordings/): AI 编程助手 **Aider** 在与 **Claude Desktop** 和 **MCP** 配合使用时获得了更强的自主性。用户强调 **Claude** 现在可以管理 **Aider** 并发布命令，提升了其引导编程任务的能力，特别是通过 *bee* 实现的无障碍网页抓取。
- [**Cursor 用户关注 Windsurf，Claude Max 即将到来**](https://www.windsurf.ai): **Cursor IDE** 面临用户对其性能问题（包括延迟和崩溃）的投诉，促使一些用户转向 **Windsurf**。然而，**Cursor** 团队预告 **Claude Max** 即将登陆该平台，并承诺将提升代码处理能力。
- [**Awesome Vibe Coding 列表收录 AI 驱动工具**](https://github.com/filipecalegario/awesome-vibe-coding): "Awesome Vibe Coding" 列表出现，汇集了旨在增强编程直觉和效率的 AI 辅助编程工具、编辑器和资源。该列表包括 AI 驱动的 IDE、基于浏览器的工具、插件和命令行界面。

**Theme 4. 硬件升温：AMD APU 和中国版 RTX 4090 引人关注**

- [**AMD 的 "Strix Halo" APU 剑指 RTX 5080 AI 桂冠**](https://wccftech.com/amd-ryzen-ai-max-395-strix-halo-ultimate-ai-pc-apus-16-zen-5-40-rdna-3-5-cores-256-gbps-bandwidth-120w/): 一篇文章声称 **AMD 的 Ryzen AI MAX+ 395 "Strix Halo" APU** 在 DeepSeek R1 AI 基准测试中可能比 **RTX 5080** 强 *3 倍以上*。这归功于 APU 更大的 VRAM 池，尽管社区仍在等待实际验证。
- [**OpenCL 后端增强 llama.cpp 中的 Adreno GPU**](https://github.com/ggml-org/llama.cpp/pull/10693): **llama.cpp** 引入了针对 **Qualcomm Adreno GPU** 的实验性 **OpenCL 后端**，有可能释放移动设备上的巨大计算能力。此更新允许通过 **OpenCL** 利用移动设备中常见的 **Adreno GPU**。
- [**中国版 48GB RTX 4090 诱惑对 VRAM 渴求的用户**](https://www.ebay.com/itm/116477031617?_skw=4090+48gb&itmmeta=01JPE69HXRKDVZN0X9541KWMYS&hash=item1b1e9274c1:g:QJIAAOSw939nrGSz&itmprp=enc%3AAQAKAAAA8FkggFvd1GGDu0w3yXCmi1fDUKPc34oU6P2kD4Q6nWW6Wkq6G0i12W%2BvQsO3yxeUwFsHxmaxOmaH16Y8wCVsdpsv%2FIPiWlLsGMqkEGTXxCnn7OtypYgyi4CHjPXB0oB2qWJ8utnPVnh4LT9TH4bePDvMrY5xqVQFS9cQ5ZfGbMK%2FWvn7fw7zYraffKanJ%2FQvcGm7o4Sxfc5QknfzbXHSQl91doo762rKufS77tcZ1w4n3pBsGoHds52pRvjMNUygQTMbf2s0S41k27mD5HjOY7poWV3eeuzCwIQhTx03JlzF%2FukwKRxZ8Ltl7FrOWsUGgw%3D%3D%7Ctkp%3ABFBMhJ-mxrNl): 成员们讨论了从中国采购价格约为 **$4500** 的 **48GB RTX 4090**，作为提升 VRAM 的更廉价方式。这些显卡采用涡轮风扇设计，仅占用两个 PCIe 插槽，但与专业卡的驱动兼容性仍是一个担忧。

**Theme 5. 版权、社区和 AI 伦理辩论持续升温**

- [**版权乱局持续：开源模型 vs. Anna's Archive**](https://annas-archive.org/blog/ai-copyright.html): 围绕使用受版权保护的数据训练 AI 的争论仍在继续，人们担心完全开源的模型因无法利用 **Anna's Archive** 等资源而受到限制。像 LoRA 和合成数据生成等规避策略面临潜在的法律挑战。
- [**Rust 社区面临毒性指控**](https://github.com/pyca/cryptography/issues/5771): 成员们辩论了所谓的 **Rust 社区** 毒性问题，并将其与 Ruby 社区进行了比较，同时讨论了最近的组织问题。人们对社区在开源项目中的包容性和行为表示担忧。
- [**AI “精通”引发存在主义辩论**]: Discord 用户质疑熟练使用 **AI 工具** 是否等同于真正的精通，思考这仅仅是生产力的提升，还是存在认知技能退化的风险。成员们辩论了在 AI 辅助时代，学习的错觉与真正理解之间的博弈。

---

# PART 1: Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **梯度累积步数（Gradient Steps）可能会毁掉你的模型**：较小的有效 Batch Size（例如 **batch=1, gradient steps = 4**）会导致模型在训练过程中遗忘过多内容。用户分享了[他们建议的 batch/grad 配置](https://discord.com/channels/1179035537009545276/1179035537529643040/1350267964634236930)。
   - 该成员表示，在尝试将更多内容挤进显存较小的设备（vramlet rig）时，“低于这个配置从未有过好运”。
- **Gemma 3 的评估故障：数据集导致错误**：用户报告在对 **Gemma 3** 进行微调时添加评估数据集会出现错误，这表明 **trl** 和 **transformers** 库中存在问题，[潜在的修复方案包括移除评估数据集](https://discord.com/channels/1179035537009545276/1179035537529643040/1350315991227082772)。
   - 发现使用带有 1 个评估样本的 **Gemma-3-1B** 不会产生错误，完全**移除评估（eval）**也能解决该错误。
- **Unsloth 对速度的追求：优化释放**：**Unsloth** 团队宣布了支持 FFT、8-bit、PT 及所有模型的改进，进一步的优化使 4-bit 模型的 VRAM 占用减少了 **+10%**，速度提升了 **>10%**，此外还增加了 Windows 支持、改进了 GGUF 转换、修复了视觉微调，并支持了 4-bit 的非 Unsloth GRPO 模型，但[目前尚不支持多 GPU（multigpu）](https://x.com/danielhanchen/status/1900592202621087944)。
   - 用户注意到有很多人在协助让 Unsloth 变得更好。
- **细心格式化你的 RAG 数据！**：当被问及如何为 RAG 聊天机器人微调模型时，成员建议在数据集中添加示例问题和示例回答，并包含来自文档的上下文，以便为机器人注入新知识。
   - 建议聊天机器人数据应遵循 `Q: A:` 格式，并可以使用在用户侧添加文档的 CPT 风格训练。
- **剪枝让 Gemma-3-27b 更精简高效**：一位成员将 [Gemma-3-27b](https://huggingface.co/fimbulvntr/gemma-3-27b-pt-unsloth-bnb-4bit-pruned-vocab) 的词表从原始的 **260k** 剪枝到了 **40k tokens**，以减少 VRAM 占用并提高训练速度。
   - 该方法涉及基于校准数据的频率计数，并移除那些可以由合并/子词（merge/subword）表示的低频词元。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Windsurf 正在吸走 Cursor 的用户**：用户对 **Cursor 的性能**问题（如延迟和崩溃）表示沮丧，由于可靠性担忧，一些用户正转向 [Windsurf](https://www.windsurf.ai)。
   - 一位用户表示，“该死，Cursor 刚刚失去了他们最重要的客户”，这表明信心严重丧失。
- **Cursor 的 Prompt 成本**：成员们讨论了 **Claude 3.7** 的 Prompt 成本：普通 Prompt 为 **$0.04**，Sonnet Thinking 为 **$0.08**，Claude Max 每次 Prompt 和工具调用（tool call）为 **$0.05**。
   - 一些用户反映，与直接使用 **Claude API** 相比，Cursor 的定价太贵了，质疑 Cursor 订阅的价值。
- **在 MCP 配置上 Linux 碾压 Windows**：一位用户分享说，在 Linux 上使用 VMware 虚拟机设置 **MCP server** 比在 Windows 上遇到多个问题要顺畅得多。
   - 这引发了一场关于整体开发和 **MCP server 设置**在 Linux 上是否普遍优于 Windows 的辩论，突出了各自的优缺点。
- **Vibe Coding：是福还是祸？**：**Vibe Coding** 的价值引发了辩论，一些人强调扎实编程知识的重要性，而另一些人则断言 **AI** 使得无需传统技能也能更快地进行创作。
   - 这突显了软件开发格局的变化以及对 **AI 对行业影响**的不同看法。
- **Claude Max 即将登陆 Cursor**：Cursor 团队的一名成员宣布 **Claude Max** 即将登陆 [Cursor](https://www.anthropic.com/claude-3)，从而最大化模型的代码处理能力。
   - 他们提到，该模型在处理大量输入时比以往的模型表现更好，释放了其全部潜力。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI “精通”引发辩论**：成员们讨论了熟练使用 **AI tools** 是否等同于真正的精通，质疑这仅仅是提高了生产力还是有削弱认知能力的风险，同时认为 AI 是一种学习的幻觉。
   - 一位成员承认，由于 AI 的辅助，即使在了解某个主题时也会感到像是在“作弊”。
- **Gemini 的图像润色**：用户探索了 **Gemini** 的图像生成功能，注意到它编辑上传图像的能力，但也指出了水印和编码错误。
   - 一些人称赞 **Gemini** 的回答非常自然，比起事实的精确性，他们更看重主观上的吸引力。
- **GPT-4o 以幽默感给人留下深刻印象**：成员们报告了使用 **GPT-4o** 的积极体验，其中一人表示它用起来最顺手，几乎可以做“任何事情”，还有成员报告说当其他人开始尝试时，出现了“有趣的结果”。
   - 这表明 **GPT-4o** 在创意和多功能应用方面表现出色，提供了有趣的用户体验。
- **AI 的自我反思**：一位成员创建了一个系统，让 **AI** 在每次会话后反思其学习内容，存储反思以积累见解，并提出反思性问题。
   - 被描述为“下一代未来感”，能够实现模拟中的模拟，以及注入核心特征集的多种人格。
- **AI 梦之队指导业务**：成员们讨论了组建一个 **AI experts** 团队来协助任务、规划，并为业务决策提供多样化的视角。
   - 该 AI 专家团队将帮助向客户交付更好的产品，并协助处理项目或任务级别的需求。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **MoE 模型：伪装的稠密网络？**：辩论围绕 **Mixture of Experts (MoE)** 模型是否仅仅是稠密网络的性能优化，而非根本不同的架构展开，正如[这篇论文](https://arxiv.org/abs/2310.10837)所强调的。
   - 讨论的核心在于 **MoEs** 是否能像稠密网络一样有效地捕捉复杂性，特别是在**避免冗余**方面。
- **Mistral 的小奇迹：Small 3.1**：根据 [Mistral AI 博客](https://mistral.ai/en/news/mistral-small-3-1)的详细介绍，以 Apache 2.0 协议发布的 **Mistral Small 3.1** 是一款多模态模型，具备文本、图像能力，并扩展了 **128k token context window**。
   - 据称其性能优于 **Gemma 3** 和 **GPT-4o Mini** 等其他小型模型。
- **版权乱象：开源模型 vs. Anna's Archive？**：关于使用受版权保护的数据训练 AI 的伦理辩论仍在继续，人们担心完全开源的模型会因为无法利用像 **Anna's Archive** 这样的资源而受到限制，正如 [Annas Archive 的博客文章](https://annas-archive.org/blog/ai-copyright.html)中所讨论的。
   - 规避策略包括使用 **LoRAs** 或生成合成数据，但这些方法未来可能面临法律挑战。
- **深度之咒再次袭来，这次是在 LLM 上**：一篇新论文介绍了**深度之咒 (Curse of Depth)**，揭示了由于 **Pre-Layer Normalization (Pre-LN)** 的广泛使用，现代 **LLMs** 中近一半的层效果不如预期，详见[这篇 Arxiv 论文](https://arxiv.org/abs/2502.05795)。
   - 由于 **Pre-LN**，深层 **Transformer** 块的导数往往会变成单位矩阵。
- **工具时间：START 长 CoT 推理起飞**：根据[关于 START 的论文](https://huggingface.co/papers/2503.04625)，**START** 是一种**集成工具的长 CoT 推理 LLM**，通过代码执行和自我调试等外部工具增强推理能力。
   - 一位成员简洁地总结道：*RL + tool calling == QwQ 上的数学提升 15% + 编程提升 39%*。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 通过屏幕录制实现自我改进**：Paul Gauthier 在一系列[屏幕录制](https://aider.chat/docs/recordings/)中展示了 **aider** 如何增强自身，演示了 **`--auto-accept-architect`** 以及 **tree-sitter-language-pack** 的集成等功能。
   - 录制内容说明了 **aider** 如何编写文件下载脚本并使用 bash 脚本来修改文件集合。
- **Claude 3.7 Sonnet 在 API 方面遇到困难**：用户报告收到来自 **Claude 3.7 Sonnet** 的*空响应*，[Anthropic 的状态页面](https://status.anthropic.com/)确认了错误率上升。
   - 一些成员推测由于这些错误，系统切换回了 **Claude 3.5**。
- **MCP Server 提升 Aider 自主性**：成员们强调 **Claude Desktop + Aider on MCP** 增强了自主性，由 **Claude** 管理 **Aider** 并发布命令。
   - 一个关键优势是从 **Claude Desktop** 运行 **Aider**，提高了 **Claude** 引导 **Aider** 的能力，并利用 *bee* 进行无阻碍的网页抓取。
- **百度发布 ERNIE 4.5 和 X1 推理模型**：百度推出了 **ERNIE 4.5** 和 **X1**，其中 [X1](https://x.com/Baidu_Inc/status/1901089355890036897) 以一半的成本提供了与 **DeepSeek R1** 相当的性能，且 **ERNIE Bot** 现在对个人用户免费。
   - 虽然 **ERNIE 4.5** 可以访问，但 **X1** 推理模型目前仅限中国境内用户使用。
- **Anthropic 准备推出 Claude 'Harmony' Agent**：[Anthropic](https://x.com/testingcatalog/status/1901051432339730603) 正在发布 **Harmony**，这是 **Claude** 的一项新功能，赋予其对*本地目录的完全访问权限*，以便研究和操作其中的内容。
   - 这可能是 Anthropic 迈向创建 AI Agent 的第一步。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Adreno GPU 获得 OpenCL 提升**：[llama.cpp](https://github.com/ggml-org/llama.cpp/pull/10693) 为 **Qualcomm Adreno GPU** 引入了一个实验性的 **OpenCL 后端**，有可能提升移动设备的计算能力。
   - 此更新允许通过 **OpenCL** 利用移动设备中广泛使用的 **Adreno GPU**。
- **4070 Ti 用户考虑升级 5090**：一位拥有 **4070 Ti** 的用户考虑升级到 **5090**，但由于缺货问题，建议等待或考虑二手的 **RTX 3090**，因为它拥有 **36GB VRAM**。
   - **二手 RTX 3090** 将提供足够的 **VRAM**，以合理的运行速度运行 *50B 以下的 Q4 模型*。
- **Mistral Small 3.1 胜过 Mini**：**Mistral** 发布了 **Mistral Small 3.1** 模型，声称其性能优于 **Gemma 3** 和 **GPT-4o Mini**，但该版本在 **llama.cpp** 中使用前需要转换为 **HF** 格式。
   - 用户正在等待[发布](https://mistral.ai/news/mistral-small-3-1)，但承认在开始使用之前需要将其转换为 **HF** 格式。
- **通过内存调优最大化 M4 Max 性能**：用户探索了在 **M4 Max** 设备上为 **LM Studio** 优化内存设置，建议使用[此脚本](https://gist.github.com/havenwood/f2f5c49c2c90c6787ae2295e9805adbe)调整 'wired' 内存分配，以提高 **GPU** 性能。
   - 该脚本有助于调整 **macOS GPU** 内存限制，允许用户通过修改 wired 内存设置向 **GPU** 分配更多内存。
- **AMD APU 性能将超越 RTX 5080？**：分享了一篇来自 wccftech 的文章，声称 AMD 的 [Ryzen AI MAX+ 395 "Strix Halo" APU](https://wccftech.com/amd-ryzen-ai-max-395-strix-halo-ultimate-ai-pc-apus-16-zen-5-40-rdna-3-5-cores-256-gbps-bandwidth-120w/) 由于其更大的 **VRAM** 池，在 **DeepSeek R1** AI 基准测试中可能*提供超过 RTX 5080 3 倍的提升*。
   - 社区保持谨慎乐观，等待实际数据来证实这些性能主张。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Anthropic API 故障影响 Claude 3 Sonnet**：根据 [Anthropic 状态页面](https://status.anthropic.com/incidents/qtxnlg9yrwqv) 的报告，向 **Claude 3.7 Sonnet** 发出的请求在大约 30 分钟内出现了错误率上升的情况。
   - 该问题随后得到解决，成功率恢复正常，但一些用户反映，尽管回复中没有收到任何文本，但仍被扣费。
- **Personality.gg 进入 AI 角色领域**：[Personality.gg](https://personality.gg) 推出了一个新平台，可以使用 **Claude**、**Gemini** 和 **Personality-v1** 等模型创建 AI 角色、进行聊天和互动，具有自定义主题和完整的聊天控制功能。
   - 该平台提供灵活的方案，并鼓励用户加入其 [Discord](https://discord.personality.gg) 获取更新，同时宣传允许 NSFW 内容。
- **Parasail 计划托管新的 RP 模型**：Parasail 正寻求在 OpenRouter 上托管新的 Roleplay (RP) 模型，并正积极与 TheDrummer 等创作者合作，托管 **Gemma 3** 和 **QwQ** 等模型的新 fine-tunes 版本。
   - 他们正在寻找能够创建强大 RP fine-tunes 的个人，这些模型需具备处理复杂指令和世界观的能力，特别关注针对角色扮演和创意写作进行微调的模型。
- **OpenRouter API 速率限制详情**：根据 [官方文档](https://openrouter.ai/docs/api-reference/limits)，OpenRouter 的速率限制取决于用户充值额度，大约 **1 USD** 对应 **1 RPS**（每秒请求数）。
   - 虽然购买更多额度可以提高速率限制，但用户发现创建额外的账户或 API keys *没有任何区别*。
- **Mistral Small 3.1 带着视觉能力发布**：Mistral Small 3.1 24B Instruct 模型已在 OpenRouter 上线，根据 [Mistral 的公告](https://mistral.ai/news/mistral-small-3-1)，该模型具备 **多模态能力** 和 **128k 上下文窗口**。
   - 公告声称其性能优于 Gemma 3 和 GPT-4o Mini 等同类模型，同时推理速度达到每秒 150 tokens。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 保证准确性**：Perplexity 推出了口号 *“当你需要准确无误时，请咨询 Perplexity”*，并发布了 [一段 Perplexity 视频广告](https://cdn.discordapp.com/attachments/1047204950763122820/1351270126615396454/lSdoFFbL6lXL_huQ.mp4?ex=67d9c3df&is=67d8725f&hm=046721b4226c4142a36a9fc331a82a120a744c64bacfae63ac90d96721381065&)。
   - Windows 上的 Perplexity 用户通过 **连续 7 天** 使用该应用，可以获得 **1 个月的 Perplexity Pro**。
- **Gemini 2 Flash 上下文引发热议**：用户正在讨论 **Gemini 2 Flash** 的上下文保留能力，据称它拥有 **1M 上下文窗口**，但表现不如常规的 Gemini。
   - 一位用户声称，在制作记忆卡片（flashcards）时，它在几条消息后就会 *忘记格式*。
- **Claude 3.7 Sonnet 存在硬性限制**：用户澄清，通过 **Perplexity Pro 订阅** 使用的 **Claude 3.7 Sonnet** 每天有 **500 次查询** 的限制，该限制在除 GPT 4.5 以外的模型间共享。
   - 他们还指出，上下文限制可能比 Anthropic 官网略多，但响应上下文限制较小，仅为 *4000 或 5000 tokens*。
- **专家寻求卓越的编程助手**：用户正在寻求关于 **最佳编程 AI 模型** 的指导，建议指向了 **Claude 3.7 Reasoning**。
   - 一位用户报告称 **Deepseek R1** 的 *幻觉率很高*，不适合总结文档；但有人分享了 [百度 (@Baidu_Inc) 的推文](https://x.com/baidu_inc/status/1901089355890036897?s=46) 链接，声称 **ERNIE X1** 的性能与 DeepSeek R1 相当，而价格仅为一半。
- **Sonar Reasoning Pro 存在图片限制**：一位用户报告称 **sonar-reasoning-pro API** 最多返回 **5 张图片**。
   - 该用户正在询问此限制是可配置的还是硬性约束。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Rust 社区收到无礼言论**：成员们讨论了 **Rust 社区** 的毒性，有人将其与 Ruby 社区进行比较，并指向了此 [Github issue](https://github.com/pyca/cryptography/issues/5771) 和 [来自 will brown 的推文](https://fxtwitter.com/willccbb/status/1901415166295544154?t=HmQDRR0NQ9mi_4udIiT4uQ&s=19)。
   - 一位成员表示：*Rust 社区相当毒。该组织最近内部有些崩溃*。
- **C 语言被称为“古老且破碎”**：一位成员将 C 描述为古老、破碎且垃圾，而另一位成员则认为 C 并没有破碎，并通过此 [链接](https://www.iso-9899.info/wiki/The_Standard) 强调了其在国际标准中的应用。
   - 一位成员链接到 [faultlore.com](https://faultlore.com/blah/c-isnt-a-language/)，认为 *C 语言不再是一种编程语言了*。
- **优化与搜索，并非一回事？**：成员们讨论了 **优化**（寻找函数的最大值或最小值）与 **搜索**（寻找集合中的最佳元素）之间的区别，并指向了 [重参数化技巧 (Reparameterization trick)](https://en.wikipedia.org/wiki/Reparameterization_trick#Variational_autoencoder)。
   - 一位成员表示：*搜索是探索，不像优化*。
- **Gemma 3 获得视觉和上下文能力**：**Gemma 3** 集成了 **视觉理解、多语言覆盖和扩展的上下文窗口**（高达 **128K tokens**），观看 [YouTube 视频](https://www.youtube.com/watch?v=n5nEd600iM0)。
   - 它集成了一个冻结的 **SigLIP 视觉编码器**，将图像压缩为 **256 个软 tokens**，并采用了一种新的 **Pan & Scan (P&S)** 方法。
- **Mistral Small 3.1 大放异彩**：**Mistral AI** 宣布发布 [Mistral Small 3.1](https://mistral.ai/fr/news/mistral-small-3-1)，在 Apache 2.0 许可下，拥有改进的文本性能、多模态理解和 **128k** token 上下文窗口。
   - 该公司声称其性能优于 **Gemma 3** 和 **GPT-4o Mini** 等同类模型，推理速度达到 **每秒 150 个 tokens**。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SmolVLM2 缩小了 VLM 的体积**：团队发布了 [SmolVLM2](https://x.com/pcuenq/status/1896632829372715442)，这是目前能理解视频的最小 VLM，其 **500M 版本** 可以在 iPhone 应用上运行。
   - 源代码和 TestFlight 测试版已提供参考。
- **简易版新 Gradio 发布！**：[Gradio Sketch 2.0](https://x.com/abidlabs/status/1897782056308142266) 现在支持通过事件构建完整的 Gradio 应用，*无需编写一行代码*。
   - 新功能使用户能够通过 GUI 构建应用程序。
- **DCLM-Edu 数据集完成清理**：发布了一个新数据集 [DCLM-Edu](https://x.com/LoubnaBenAllal1/status/1898044807928295808)；它是使用 FineWeb-Edu 分类器过滤的 DCLM 版本，专门为 **SmolLM2 135M/360M** 等 *smol 模型* 进行了优化。
   - 其目的是因为 *小模型对噪声很敏感，可以从高度精选的数据中受益*。
- **Coding Vibes 获得 Awesome 列表**：公布了一个 “Awesome Vibe Coding” 列表，包含 [工具、编辑器和资源](https://github.com/filipecalegario/awesome-vibe-coding)，使 AI 辅助编程更加直观和高效。
   - 该列表包括 AI 驱动的 IDE 和代码编辑器、基于浏览器的工具、插件和扩展、命令行工具以及最新的新闻和讨论。
- **AI Agents 协作正在酝酿中**：几位成员表示有兴趣 **在 Agentic AI 项目上进行协作**，以解决业务问题并增强知识。
   - 该行动号召旨在组建团队，为美国消费者构建合格的 AI Agents 并共同学习。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Figure 的 BotQ 批量生产人形机器人**：Figure 宣布推出 **BotQ**，这是一个全新的大规模制造设施，其第一代生产线每年可生产多达 **12,000** 台人形机器人，实现了[制造的垂直整合并构建了软件基础设施](https://www.figure.ai/news/botq)。
   - 该公司旨在控制构建过程和质量，甚至暗示了“机器人制造机器人”的前景。
- **百度 ERNIE X1 媲美 DeepSeek，现已免费！**：**百度**发布了 **ERNIE 4.5** 和 **ERNIE X1**，据报道 X1 的性能在价格减半的情况下达到了 **DeepSeek R1** 的水平。百度还宣布其聊天机器人 **ERNIE Bot**（文心一言）现对个人用户免费，可在[其官网](https://yiyan.baidu.com/)使用。
   - 根据[这条推文](https://x.com/cedric_chee/status/1901159341975384308?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)，百度计划在 6 月 30 日开源庞大的 4.5 模型，并在未来逐步向开发者开放。
- **Mistral Small 3.1 首次亮相，配备超大上下文窗口**：**Mistral AI** 发布了 **Mistral Small 3.1**，这是一款提升了文本性能、多模态理解能力并具备 **128k** token 上下文窗口的新模型。其推理速度达每秒 **150** token，表现优于 **Gemma 3** 和 **GPT-4o Mini** 等模型，并已[根据 Apache 2.0 许可证发布](https://mistral.ai/news/mistral-small-3-1)。
   - 该模型声称达到了 *SOTA* 级别，且具备多模态和多语言能力。
- **OpenAI 后训练副总裁离职投身材料科学**：**OpenAI** 负责 **post-training**（后训练）的研究副总裁 **Liam Fedus** 将离开公司，创办一家材料科学 **AI startup**。**OpenAI** 计划投资并与其新公司开展合作。
   - 根据[这条推文](https://x.com/LiamFedus/status/1901740085416218672)，一位成员将后训练的工作称为“烫手山芋”。
- **DAPO 中发现大规模数据集重复**：**DAPO** 的作者意外地将数据集重复了约 **100** 倍，导致数据集大小达到 310 MB。一名成员通过 HF 的 SQL 控制台创建了一个去重版本，将数据集缩减至 3.17 MB（[HuggingFace 数据集](https://huggingface.co/datasets/YouJiacheng/DAPO-Math-17k-dedup)）。
   - 根据[这条推文](https://x.com/tongyx361/status/1901702083352678763?s=61)，作者承认了这一问题，表示他们已知晓但“负担不起重新训练的费用”。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **多 Agent 拓扑结构引发辩论**：成员们就多 Agent 系统的 **Swarm**、**Mesh** 和 **Sequence** 架构展开辩论，寻求关于如何防止子 Agent 偏离轨道的建议，特别是由于“传声筒效应”导致的问题。
   - 核心问题可能在于“并行执行”和“无监督自主性”，而在 **handoff**（移交）过程中 Agent 交换系统指令、可用函数甚至模型，使问题更加复杂。
- **OpenSwarm 演变为 OpenAI-Agents**：**OpenSwarm** 项目已被 OpenAI 采纳并更名为 **openai-agents**，增加了 OpenAI 特有的功能，但一项关于 MCP 支持的 PR 被拒绝了。
   - 有传言称 **CrewAI**（或 **PraisonAI**？）可能会使用“无状态单线程 Agent 方法”提供类似功能。
- **MyCoder.ai 在 Claude-Code 之前抢先亮相**：**mycoder.ai** 的发布恰逢 **Claude-code** 的发布公告，促使其通过一篇登上首页的 Hacker News 帖子进行适配，详见[此处](https://news.ycombinator.com/item?id=43177117)。
   - 鉴于 **claude-code** 仅限 Anthropic 使用，市场对通用替代方案有需求，一名成员使用 **litellm proxy** 成功解决了这一问题。
- **Glama 服务器检查频率引发讨论**：成员们询问 **Glama 扫描** 的频率以及是否可以触发 MCP 服务器的重新扫描；扫描频率与关联 GitHub 仓库的提交频率挂钩。
   - 即使在修复了依赖问题后，某些服务器仍无法检查，显示“无法检查服务器”，可在 [Glama AI](https://glama.ai/mcp/servers/s2em7b2kwf/score) 关注进度。
- **Vibe Coders 联合起来！**：[Awesome Vibe Coding](https://github.com/filipecalegario/awesome-vibe-coding) 列表收录了 AI 辅助编码工具、编辑器和资源，旨在增强编码的直观性和效率。
   - 该列表包括 AI 驱动的 IDE、基于浏览器的工具、插件和 CLI。甚至有一位 AI 编码员向该仓库提交了 PR，并建议添加 [Roo Code](https://github.com/szcharlesji/crypto-mcp)。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-o1 数学技能接近人类水平**：**GPT-o1** 在 **Carnegie Mellon** 本科数学考试中获得了满分，每道题的解题时间不足一分钟，成本约为 5 美分，详见[此推文](https://x.com/poshenloh/status/1900721180887203879?s=46)。
   - 讲师对此印象深刻，指出这*接近了能够胜任中等难度非例行技术工作的临界点。*
- **百度的文心一言（ERNIE）具备成本竞争力**：**百度**发布了 **ERNIE 4.5** 和 **ERNIE X1**，据[此公告](https://x.com/baidu_inc/status/1901089355890036897?s=46)称，后者的性能可与 **DeepSeek R1** 媲美，但成本仅为一半。
   - 值得注意的是，**文心一言（ERNIE Bot）** 已提前向个人用户免费开放，两款模型均可在官网使用。
- **AI 播客应用走向户外**：**Snipd** 发布了由 [Kevin Smith](https://x.com/latentspacepod/status/1900666708270215383) 主持的新播客，讨论了**用于学习的 AI 播客应用**。
   - 本期节目是他们的首个“户外”播客，@swyx 和 @KevinBenSmith 聊到了 **aidotengineer NYC**、从金融转向科技行业的心路历程，以及 [@snipd_app](https://www.snipd.net/) 的技术栈。
- **关于 Claude 3.5 与 3.7 优劣的辩论**：成员们讨论了使用 **Claude 3.5** 而非 **3.7** 的优点，理由是 **3.7** *过于积极*，会在未被要求的情况下执行操作。
   - 其他人表示他们在利用 **Claude 3.5** 时也遇到了 **GPU** 问题。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **用户渴望集成 Gemini 的 Android 体验**：多位用户请求提供完整的 **Gemini 集成 Android 体验**，希望将 **Google Assistant/Gemini** 与 **NotebookLM** 结合。
   - 一些用户对目前的 **Gemini** 实现表示失望，正焦急地等待升级。
- **Deepseek R1 震撼 AI 市场**：一位用户指出，由于 **Deepseek R1** 的发布，AI 市场发生了剧变，其低成本的推理能力影响了 **Gemini 2.0**。
   - 该用户声称 **Deepseek R1** 似乎*震撼了整个行业*，从而促使其他公司发布新模型。
- **NotebookLM 音频概览时长增加**：一位用户希望增加 **NotebookLM** 生成的音频概览（Audio Overviews）长度，因为 **16,000 字的文件**仅生成了 **15 分钟的概览**。
   - 他们明确要求至少 **1 小时以上**的概览，但目前尚未有解决方案分享。
- **NotebookLM 辅助精神科药物减量**：一位用户利用 **NotebookLM** 为某种精神科药物创建了“双曲线减量计划”，并参考相关性研究来指导该计划。
   - 另一位用户提醒，在*任何*平台上**基于数据进行减量**都不应在没有专业专家意见的情况下独自进行。
- **NotebookLM 集成至内部门户/CRM**：一位用户希望将 **NotebookLM** 与包含视频和知识库文章的内部门户/CRM 集成，有人建议使用 [Agentspace](https://cloud.google.com/products/agentspace?hl=en) 作为解决方案。
   - 由于 **NotebookLM** 不支持连接到上述类型的数据源，而 **Agentspace** *包含并集成了 NotebookLM*。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton-Windows 现已支持 PIP 安装**：**Triton-windows** 已发布至 **PyPI**，因此你可以通过 `pip install -U triton-windows` 进行安装/升级，不再需要从 GitHub 下载 wheel 文件。
   - 此前，用户必须手动管理 wheel 文件，使得更新过程更加繁琐。
- **Torch Compile 在反向传播中变慢**：有成员报告称，虽然 **torch.compile** 在前向传播中表现良好，但在为自定义 Kernel 使用 **torch.autograd.Function** 时，反向传播的速度相当慢。
   - 使用 `torch.compile(compiled_backward_fn)` 包装反向传播函数可能会解决性能问题。
- **分享 NVIDIA SASS 指令历史**：一位成员分享了一个 [gist](https://gist.github.com/herrmann/f721da109e0c5c7c34c847ff2cf3da1e)，对比了不同架构下的 **NVIDIA SASS 指令**，这些指令是利用 Python 从 NVIDIA 的 HTML 文档中提取并对比的。
   - 这让用户能够追踪 NVIDIA GPU 系列中指令集的演进。
- **Reasoning Gym 突破 100 个数据集！**：[Reasoning Gym](https://github.com/open-thought/reasoning-gym) 项目现在拥有 **101 个数据集**，庆祝开发者们的贡献。
   - 不断增长的数据集集合将为 LLM 测试提供更全面的支持。
- **Jake Cannell 招募 GPU 高手**：Jake Cannell 正在[招聘 GPU 开发者](https://www.linkedin.com/jobs/view/4118975911/)，以实现他在演讲中提到的想法，同时 **nebius.ai** 的 GPU 云服务也受到了推崇。
   - 这对于那些对 **AGI** 或**类脑硬件（neuromorphic hardware）**感兴趣的人来说非常相关。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **EleutherAI 欢迎 Catherine Arnett**：EleutherAI 欢迎 **Catherine Arnett**，她是一位专注于**计算社会科学**和**跨语言 NLP** 的 NLP 研究员，致力于确保模型在不同语言间表现“同样出色”。
   - 她最近的工作包括 [Goldfish](https://arxiv.org/abs/2408.10441)、[Toxicity of the Commons](https://arxiv.org/abs/2410.22587)、[LM performance on complex languages](https://arxiv.org/abs/2411.14198) 以及 [Multilingual Language Modeling](https://arxiv.org/abs/2311.09205)。
- **新型 Block Diffusion 模型发布**：一篇新论文介绍了 **Block Diffusion**，这是一种在自回归（autoregressive）和扩散（diffusion）语言模型之间进行插值的方法，结合了两者的优势：高质量、任意长度、KV caching 以及可并行性，详情见[论文](https://arxiv.org/abs/2503.09573)和[代码](https://github.com/kuleshov-group/bd3lms)。
   - 它结合了自回归和扩散语言模型的优点。
- **VGGT 生成元宇宙 GLB 文件！**：一位成员分享了 [VGGT](https://vgg-t.github.io/)，这是一个前馈神经网络，可以从多个视角推断 3D 属性并生成 GLB 文件，这些文件可以直接集成到元宇宙中。
   - 该成员表示：“我非常喜欢它能导出 GLB 文件。这意味着我可以原封不动地将它们直接放入我的元宇宙中。”
- **Gen Kwargs 完美拥抱 JSON**：`--gen_kwargs` 参数正从逗号分隔的字符串转换为 **JSON**，从而允许更复杂的配置，例如 `'{"temperature":0, "stop":["abc"]}'`。
   - 讨论中探讨了同时支持两种格式以方便使用的可能性，特别是对于标量值。
- **LLM 排行榜：训练集 vs 验证集划分**：旧版 LLM 排行榜的组配置与实际使用的设置之间存在差异，特别是在 **arc-challenge 任务**方面。
   - 已创建一个 [PR 来修复此问题](https://github.com/EleutherAI/lm-evaluation-harness/pull/2802)，以解决 `openllm.yaml` 配置（指定 `validation` 作为 fewshot 划分）与原始排行榜（使用 `train` 划分）之间的不一致。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad SDXL 性能落后于 Torch**：在 **7900 XTX** 上对 **tinygrad** 的 **SDXL** 进行基准测试显示，在 **AMD backend** 上使用 **BEAM=2** 时速度为 **1.4 it/s**，而 **torch.compile** 使用 **FlashAttention** 和 **TunableOp ROCm** 达到了 **5.7 it/s**。
   - George Hotz 建议对比 kernel 以寻找优化机会，目标是在年底前超越 **torch**。
- **Tensor Cat 依然缓慢**：一位致力于提高 tensor cat 速度的成员在 **X** 上分享了白板构思（[链接](https://x.com/t0kenl1mit/status/1900952693587538018)），指出尽管对 devectorizer 进行了更改，速度仍然很慢。
   - 他们怀疑生成的 **IR** 和加载 **numpy arrays** 存在问题，正考虑通过 **ELF** 和 **LLVM** 使用自定义 **C/C++** 来克服限制。
- **BLAKE3 悬赏细节明朗化**：*高性能并行 BLAKE3* 悬赏的状态已明确，截图（[链接](https://cdn.discordapp.com/attachments/1068976834928193609/1350640745505231061/Screenshot_2025-03-15_182214.png?ex=67d973f7&is=67d82277&hm=19c5ffbf47ae93d8dda6ba9c5fc1b65cc3b1df108a2f4fd5860ba66e301bef7c&)）显示了更新后的悬赏状态。
   - 该成员更新了电子表格，并指出渐近性能（asymptotic performance）是该悬赏的关键要求。
- **WebGPU 集成势头强劲**：有成员询问如何发布基于 **resnet18** 的 electron/photon 分类器的 **Tinygrad** 实现作为示例，并被引导至一个[改进 WebGPU 集成的 PR](https://github.com/tinygrad/tinygrad/pull/9424)。
   - 建议创建一个托管在 **GitHub Pages** 上的 **WebGPU** demo，并将权重放在 **Hugging Face** 上以供免费访问和测试。
- **Tinygrad 在 Lazy Mode 调试中遇到困难**：一位成员在 Tinygrad 中通过 print-debugging 中间 tensor 值时遇到了 gradients 的断言错误，尽管由于 [lazy computation](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html) 的问题使用了 `.detach()`。
   - 鉴于 lazy computation 不是幂等的（idempotent），该成员正在寻找比将值线程化输出更好的方法。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 展示带有 Corrective RAG 的 Agentic 推理**：LlamaIndex 介绍了一个关于使用 **corrective RAG** 构建用于搜索和检索的 **agentic reasoning system** 的分步[教程](https://twitter.com/llama_index/status/1901079091345818022)，该系统由 LlamaIndex workflows 编排。
   - 该教程使用户能够编排复杂的、可定制的、事件驱动的 Agent。
- **LlamaExtract 从云端推出**：**LlamaExtract** 解决了从复杂文档中提取结构化数据的问题，目前处于公开测试阶段，可在 [cloud.llamaindex.ai](https://cloud.llamaindex.ai) 上使用，提供 **web UI** 和 **API**。
   - 用户可以定义 schema 来自动提取结构化数据；更多详细信息请参见[此处](https://t.co/gT3R2l7CWM)。
- **多模态 AI Agent 在 NVIDIA GTC 2025 展开对决**：**Vertex Ventures US** 和 **CreatorsCorner** 将在 **NVIDIA GTC 2025** 举办一场 **AI hackathon**，挑战参与者开发复杂的**多模态 AI Agent**。
   - 该黑客松为能够进行战略决策并与各种工具交互的 Agent 提供 **$50k+ 奖金**；更多信息可以在[此处](https://lu.ma/meofrw3d)找到。
- **社区推出视觉语言模型中心**：一位社区成员为专注于**视觉语言模型 (VLMs)** 的多模态研究人员推出了一个[社区驱动的中心](https://github.com/thubZ09/vision-language-model-hub.git)。
   - 创建者正在积极寻求贡献和建议，并计划每周更新该中心。
- **Pydantic AI 与 LlamaIndex 的竞争**：新用户想知道用于构建 Agent 的 **Pydantic AI** 和 **LlamaIndex** 框架之间的区别，尤其是初学者应该使用哪一个。
   - LlamaIndex 团队成员表示，最适合你开发思维模型的框架可能就是最好的选择。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Gemma 的语言能力令人印象深刻**：成员们观察到 **Gemma**、**DeepSeek R1** 和 **Qwen2.5** 模型在回答关于“将密封罐子放在零下温度的室外会发生什么”的谜题时，都能以多种语言提供正确答案。
   - 虽然其他模型预测罐子会发生灾难性损坏，但 **Gemma** 提供了更有帮助且更细致的建议。
- **Gemma 3 集成遭遇许可障碍**：用户正在等待 **GPT4All** 支持 **Gemma 3**，但由于 Hugging Face 上的许可协议问题，其集成因等待 **Llama.cpp** 更新而推迟，详情见 [此 GitHub issue](https://github.com/nomic-ai/gpt4all/issues/3540)。
   - 有推测认为 Google 是否会监管那些规避其许可协议的重新分发行为。
- **LocalDocs 用户遭遇崩溃困扰**：一位新用户报告在崩溃并重新安装后丢失了 **LocalDoc** 集合，并寻求防止未来崩溃导致数据丢失的建议。
   - 资深用户建议定期保存 *localdocs* 文件并在崩溃后进行恢复，并补充说“有时仅一个损坏的 PDF 就能导致系统崩溃”。
- **通过更好的 Prompt 提升 O3-mini**：一位用户分享了一个针对 **O3-mini** 的 Prompt，用于解释其思考过程，并建议通过要求 **thinking**（思考）和 **reflection**（反思）部分（包含逐步推理和错误检查）来改进任何模型的蒸馏（distillation）。
   - 现在解释复杂过程变得更加容易。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 推迟 Command A 的微调（Fine-Tuning）**：尽管社区充满期待，Cohere 团队成员确认目前*尚无计划*在平台上启用 **Command A 的微调**。
   - 他们向社区保证会提供更新，但这与部分用户对快速部署功能的预期有所背离。
- **Azure Terraform 问题阻碍 Rerank v3**：一位用户在使用 Terraform 创建 **Azure Cohere Rerank v3** 时遇到错误，并分享了代码片段和生成的错误消息。
   - 该问题已被重定向至 <#1324436975436038184> 频道，表明需要专门的关注或调试。
- **社区呼吁建立 CMD A 私有频道**：一位成员建议创建一个专门讨论 **CMD A 私有部署**的频道，特别是为了支持客户的本地部署。
   - 该提议获得了热烈支持，凸显了社区对本地部署或私有云解决方案的兴趣。
- **Vercel SDK 在 Cohere 对象处理上出现失误**：一位用户指出 [Vercel SDK](https://sdk.vercel.ai/providers/ai-sdk-providers/cohere) 错误地认为 Cohere 的 Command A 模型**不支持对象生成（object generation）**。
   - 这种差异可能会影响使用该 SDK 的开发者，需要 Cohere 和 Vercel 团队的关注以确保准确集成。
- **自由职业者提供编程帮助**：一位 **30 岁的日本男性自由程序员**介绍了自己，并表示愿意用他的编程技能帮助社区成员。
   - 呼应了“互相帮助是我们生存的支柱”这一情感。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 考虑集成 MCP**：一位成员对集成 **dspy/MCP** 感兴趣，并链接了一个 [GitHub 示例](https://github.com/philschmid/mcp-openai-gemini-llama-example/blob/master/sqlite_llama_mcp_agent.py)来阐述他们的建议。
   - 另一位成员担心添加 MCP 主机、客户端和服务器是否会使过程过于复杂。
- **DSPy 弃用 Assertions 和 Suggestions**：用户注意到 DSPy 中关于 **Assertions / Suggestions** 的 [文档消失了](https://dspy.ai/learn/programming/7-assertions/?h=dspy.suggest#dspyassert-and-dspysuggest-api)，并询问它们是否仍受支持。
   - 他们希望验证输出（特别是格式），并观察到 LLM 并不总是遵守格式的情况。
- **Output Refinement 作为 Assertion 的替代方案登场**：在 **DSPy 2.6** 中，**Assertions** 被使用 `BestOfN` 和 `Refine` 等模块的 **Output Refinement** 所取代，详见 [DSPy 文档](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/)。
   - 这些模块旨在通过使用不同的参数设置进行多次 LM 调用，来增强预测的可靠性和质量。
- **QdrantRM 悄然退出 DSPy**：用户询问 **QdrantRM** 是否已在 **DSPy 2.6** 中被移除。
   - 提供的上下文中未给出解释。

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Caiming Xiong 演讲关于 Multimodal Agents**：Salesforce 的 Caiming Xiong 讲解了 **Multimodal Agents**，涵盖了跨多种模态的 **perception, grounding, reasoning 和 action** 的集成，该演讲在 [YouTube](https://www.youtube.com/live/n__Tim8K2IY) 上进行了直播。
   - 演讲讨论了在现实环境（**OSWorld**）中衡量能力以及创建大规模数据集（**AgentTrek**）的问题，引用了超过 **200 篇论文**和 **>50,000 次引用**。
- **Self-Reflection 面临二分性**：成员们讨论了 **Lecture 1** 和 **Lecture 2** 之间关于 LLM 中 **self-reflection 和 self-refinement** 的明显矛盾。一位用户指出，**Lecture 1** 提到需要*外部评估*，而 **Lecture 2** 则建议 LLM 可以通过奖励自己的输出来改进自身。
   - 附带了来自 **Lecture 1, slide 67**（[图片 1](https://cdn.discordapp.com/attachments/1282734248112947210/1351127068745928816/image.png?ex=67d9e763&is=67d895e3&hm=7d31b7a0583550a36a872d74bfaf765de39c6b1173333d2ce51174940c0aa522&)）和 **Lecture 2, slide 51**（[图片 2](https://cdn.discordapp.com/attachments/1282734248112947210/1351127069169418260/image.png?ex=67d9e764&is=67d895e4&hm=12bbe1810790f7f688b11fe093f693a2791e94bd9e74e71ec7c2cfa3264bd004&)）的截图，以说明这一明显的冲突。
- **System Prompt 的可靠性受到质疑**：一位成员建议，依赖 system prompts 的特定行为可能并不可靠，因为*归根结底，所有这些都是文本输入，模型可以处理它，所以你应该能够绕过框架和服务*。
   - 该成员补充说，训练数据可能包含格式 `<system> You are a helpful assistant </system> <user> &#123;&#123;Some example user prompt&#125;&#125; </user> <assistant> &#123;&#123;Expected LLM output&#125;&#125; </assistant>`。
- **高级 LLM Agent 课程报名仍开放**：成员们询问是否仍可以报名 **Advanced LLM agent course** 并在报名后获得 **certificate**。
   - 工作人员回复说，只需完成 **signup form** 即可！介绍幻灯片中的大部分信息仅适用于 **Berkeley 学生**，但任何人都可以参加 **MOOC** 并在结束时获得 **certificate**。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 因 AI 艺术美学受到赞赏**：一位成员对 **Modular** 在其营销材料中使用的 **AI art** 表示欣赏。
   - 他们表示：“*Modular 使用的所有 AI 艺术都很棒！*”
- **Compact Dict：它过时了吗？**：关于 Mojo 中 [compact-dict](https://github.com/mzaks/compact-dict) 实现状态的讨论浮出水面。
   - 成员们建议，原始版本的功能可能已经集成到 **stdlib** 的 `Dict` 中。
- **SIMD 和 stdlib Dict 性能问题**：一位用户在使用 **stdlib Dict** 处理 **SIMD** [float64, 1] 类型时遇到了性能瓶颈。
   - 瓶颈归因于 hash 库中 `hash()` 函数的缓慢，促使寻找更快的替代方案。
- **Discord 频道收到垃圾信息**：一位成员澄清说，Discord 频道中的某些消息被归类为垃圾信息，另一位成员迅速确认了这一点。
   - 未提供有关垃圾信息性质或来源的进一步细节。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **SVCFA 启动 AI4Legislation 竞赛**：**硅谷华人协会基金会 (SVCAF)** 正在举办 **AI4Legislation** 竞赛，奖金高达 **$3,000**，活动持续至 **2025 年 7 月 31 日**，鼓励用于立法参与的开源 AI 解决方案；[竞赛仓库](https://github.com/svcaf/2025-AI4Legislation-Public)现已上线。
   - SVCAF 将于 2025 年 3 月底举行关于该竞赛的在线研讨会；在此处 [RSVP](https://forms.gle/pmbkRLVurbXcGBbAA)。
- **Dnipro VC 举办 AI Demo Jam**：**Dnipro VC** 和 **Data Phoenix** 将于 **3 月 20 日**在加州 Sunnyvale 举办 **AI Demo Jam**，届时将有 5 家 AI 初创公司展示其产品。
   - 活动将包括来自 Marianna Bonechi (**Dnipro VC**)、Nick Bilogorskiy (**Dnipro VC**)、Dmytro Dzhulgakhov (**fireworks.ai**) 的专家小组讨论、开放式麦克风推介和高能量的社交环节；在此处 [注册](https://lu.ma/AI-demo-jam)。
- **成员寻求 MRI Object Detection 帮助**：一位成员请求帮助创建一个用于 **MRI 图像 object detection** 的模型，不提供金钱报酬。
   - 未提供关于模型类型、数据可用性或使用场景的具体细节。

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Qdrant 请求被断然拒绝**：一名成员建议切换到 **Qdrant**，但另一名成员确认他们目前并未使用它。
   - 该建议在没有进一步解释的情况下被否决；*不，我们没有使用 Qdrant*。
- **用户请求 API 支持重复惩罚（Repetition Penalty）**：一位用户请求在 API 中增加 **repetition penalty 支持**，并指出这是阻碍 **Jamba** 模型更广泛采用的关键功能。
   - 该用户表示，缺乏重复惩罚支持是限制他们增加模型使用量的*唯一因素*。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Mistral 发布 Small 3-1**：Mistral AI 发布了 **Mistral Small 3-1**，[点击此处查看](https://mistral.ai/news/mistral-small-3-1)。
   - 未提供更多细节。
- **可学习标量（Learnable Scalars）帮助模型收敛**：一篇新论文 [Mitigating Issues in Models with Learnable Scalars](https://www.alphaxiv.org/abs/2503.10622) 提出引入 **learnable scalar** 来帮助模型*正常收敛*。
   - 这为稳定训练提供了一种实用的方法。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1350182272686362724)** (923 条消息🔥🔥🔥): 

> `Gradient steps, Gemma 3 fine tuning, Tokenizer issues, MattBCool's Twitter hack, Unsloth speed` 

- **梯度步数（Gradient Steps）可能影响模型**：一名成员表示，较小的有效 Batch Size（例如 **batch=1, gradient steps = 4**）可能导致模型在训练期间遗忘过多，并[建议了其他 batch/grad 配置](https://discord.com/channels/1179035537009545276/1179035537529643040/1350267964634236930)。
   - 他们表示“*在尝试将更多内容挤进低显存（vramlet）设备时，低于该配置从未有过好运*”。
- **Gemma 3 评估数据集产生错误**：多名成员报告在 **Gemma 3** 微调期间添加评估数据集时出现错误，堆栈跟踪显示 **trl** 和 **transformers** 库中存在问题，[潜在的修复方法包括移除评估数据集](https://discord.com/channels/1179035537009545276/1179035537529643040/1350315991227082772)。
   - 发现使用带有 1 个评估样本的 **Gemma-3-1B** 不会产生错误，而**移除 eval** 可以解决该错误。
- **Tokenizer 模型文件缺失**：一名成员在运行带有 **Gemma 3** 的 gguf 代码块时遇到 `FileNotFoundError`，提示缺少 **tokenizer.model**，这表明 Lora 或全 16-bit 保存中缺少 Tokenizer 模型，并[建议快速运行 27b 模型进行验证](https://discord.com/channels/1179035537009545276/1179035537529643040/1350330220298416139)。
- **MattBCool 的 Twitter 账号被盗**：MattBCool 报告称，由于第三方集成和缺乏手机号码验证，他的 Twitter/X 账号被黑，[新持有者冒充 Unsloth 工程师](https://mattcool.tech/posts/mattbcool-x-account-compromised-that-is-not-me)。
   - 冒充者在个人简介中放置了一个*钓鱼链接*，伪装成他博客上的链接。
- **Unsloth 声称提升了速度**：团队宣布了 **Unsloth** 的改进，支持 FFT、8-bit、PT 及所有模型，进一步的优化使 VRAM 使用量减少了 10% 以上，4-bit 速度提升了 10% 以上，此外还增加了 Windows 支持、改进了 GGUF 转换、修复了视觉微调，并支持 4-bit 的非 Unsloth GRPO 模型，但[目前尚不支持多 GPU（multigpu）](https://x.com/danielhanchen/status/1900592202621087944)。
   - 有很多人在协助让 Unsloth 变得更好。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/danielhanchen/status/1900592202621087944">来自 Daniel Han (@danielhanchen) 的推文</a>：很高兴分享 @UnslothAI 现在支持：• 全量微调 + 8bit • 几乎任何模型，如 Mixtral, Cohere, Granite, Gemma 3 • 视觉微调不再出现 OOM！博客文章详情：https://unsl...</li><li><a href="https://mattcool.tech/posts/mattbcool-x-account-compromised-that-is-not-me">Mattbcool X 账号被盗：那不是我</a>：Twitter/X 账号 mattbcool 不再由我持有</li><li><a href="https://chatqa-project.github.io/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1901760160814784949">来自 Daniel Han (@danielhanchen) 的推文</a>：明天周二我将和我的兄弟一起参加 NVIDIA 的 GTC！我们有一些 Unsloth 贴纸和徽章！我们会穿着 🦥Unsloth T恤四处走动 :)</li><li><a href="https://unsloth.ai/blog/r1-reasoning">在本地训练你自己的 R1 推理模型 (GRPO)</a>：你现在可以使用 Unsloth 100% 在本地复现你自己的 DeepSeek-R1 推理模型。使用 GRPO。开源、免费且对初学者友好。</li><li><a href="https://wheels.vllm.ai/nightly">未找到标题</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/157FWTSTO6DTi_BRdEBmwrwi7r71rpDDQ#scrollTo=Vin49wlA4Q8n">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/updating#updating-without-dependency-updates">更新 | Unsloth 文档</a>：要更新或使用旧版本的 Unsloth，请遵循以下步骤：</li><li><a href="https://mistral.ai/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>：SOTA。多模态。多语言。Apache 2.0</li><li><a href="https://huggingface.co/google/gemma-3-4b-it/discussions/15">google/gemma-3-4b-it · AttributeError: 'HybridCache' object has no attribute 'float'</a>：未找到描述</li><li><a href="https://huggingface.co/google/gemma-2-9b-it/discussions/10">google/gemma-2-9b-it · “强烈建议使用 `eager` attention 实现来训练 Gemma2 模型”</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - 动态 4-bit 量化</a>：Unsloth 的动态 4-bit 量化选择性地避免对某些参数进行量化。这在保持与 BnB 4bit 相似的 VRAM 使用量的同时，极大地提高了准确性。</li><li><a href="https://huggingface.co/unsloth/c4ai-command-a-03-2025-unsloth-bnb-4bit">unsloth/c4ai-command-a-03-2025-unsloth-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blog/packing-with-FA2">通过结合 Flash Attention 2 的 Packing 技术提高 Hugging Face 训练效率</a>：未找到描述</li><li><a href="https://unsloth.ai/newsletter">Unsloth 通讯</a>：加入我们的通讯和候补名单，获取关于 Unsloth 的一切！</li><li><a href="https://longbench2.github.io/"> LongBench v2 </a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/gemma-3-27b-it-GGUF">unsloth/gemma-3-27b-it-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorWithFlattening">Data Collator</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：以下是我们所有 notebook 的列表：</li><li><a href="https://unsloth.ai/blog/gemma3#everything">使用 Unsloth 微调 Gemma 3</a>：Gemma 3，Google 的新一代多模态模型。使用 Unsloth 微调并运行它们！Gemma 3 提供 1B, 4B, 12B 和 27B 尺寸。</li><li><a href="https://www.imagemagick.org/discourse-server/viewtopic.php?t=17842)">Imagemagick & ghostscript 许可证混淆 - 遗留 ImageMagick 讨论存档</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit">unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/mistral-benchmark#:~:text=Performance%20breakdowns%20bit%20by%20bit">Unsloth 更新：Mistral 支持及更多</a>：我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有其他基于 Llama 架构的模型提供 QLoRA 支持！我们添加了 sliding window attention、初步的 Windows 和 DPO 支持，以及 ...</li><li><a href="https://github.com/search?q=stars%3A%3E10000+license%3Aagpl-3.0&type=Repositories&ref=advsearch&l=&l=>">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 和贡献

bute to over 420 million projects.</li><li><a href="https://github.com/ollama/ollama/issues/9701">RTX 5090 Performance on Ubuntu Gemma 3 · Issue #9701 · ollama/ollama</a>: 问题是什么？我在 Ubuntu 上使用 RTX 5090 得到了以下结果。为了对比，我测试了类似的模型，全部使用默认的 q4 量化。性能对比：Gemma2:9B =...</li><li><a href="https://x.com/MattBCool>">来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://github.com/ggml-org/llama.cpp.git">GitHub - ggml-org/llama.cpp: C/C++ 环境下的 LLM 推理</a>: C/C++ 环境下的 LLM 推理。通过在 GitHub 上创建账号为 ggml-org/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py#L1538">unsloth/unsloth/models/llama.py at main · unslothai/unsloth</a>: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3、DeepSeek-R1、Gemma 3 和推理型 LLMs！🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1350184650726965309)** (34 messages🔥): 

> `llama-server 视觉支持, RWKV-7 支持, Q4 vs Q8, bnb 库限制, QLoRA NF4 量化权重` 


- **Llama-server 缺乏视觉支持**：有人指出 **llama-server** 目前还不支持视觉功能，并引用了 [llmbingo.png](https://cdn.discordapp.com/attachments/1179039861576056922/1350535747794636923/llmbingo.png?ex=67d9baee&is=67d8696e&hm=ab080a6e00a4974612ac334535b69490706b3a3b8d3006a31a0f1986f83240e7&)。
- **RWKV-7 支持被列入愿望清单**：一位成员对 Unsloth 支持 **RWKV-7** 表现出极大的热情，表示：“如果 Unsloth 支持 RWKV-7，我会疯掉的”。
- **关于 Q4 与 Q8 的大辩论**：成员们讨论了推理时 **Q4** 和 **Q8** 量化之间的权衡，其中一人由于感知的质量差异，更倾向于使用 **8b @ bf16** 而非 **70b @ Q4**。
   - 另一位成员表示赞同，并指出了从 4-bit 转换为 16-bit GGUF 格式时存在的问题。
- **bnb 库阻碍了反量化**：有人认为，对 **bnb 库** 等封装器的依赖限制了 Unsloth 反量化的潜力。
   - 该成员建议从头开始研究并实现自定义解决方案，理由是 CUDA 并非开源带来的挑战，并分享了[一篇关于 QLoRA 反量化的文章](https://lweitkamp.github.io/posts/qlora_dequantize)。
- **用于 QLoRA 反量化的 Triton 内核**：一位成员强调了 Unsloth 在编写用于 **QLoRA NF4** 量化权重反量化的 **Triton 内核** 时面临的挑战，并引用了 [Unsloth 的挑战性任务列表](https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH?usp=sharing#scrollTo=QoE2DGRZG2Ng)。
   - 他们还分享了一个包含 Triton 内核和基准测试 Notebook 的 [GitHub 仓库](https://github.com/lweitkamp/qlora_dequantize_triton)，声称 LLaMA 模型的性能提升了 **1.6 倍** 到 **1.8 倍**。



**提及链接**: <a href="https://lweitkamp.github.io/posts/qlora_dequantize">在 Triton 中进行 QLoRA 权重反量化</a>: 未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1350182775659171902)** (480 messages🔥🔥🔥): 

> `Gemma 3 微调问题, Unsloth GPU 支持, Unsloth 的 RAG 数据格式化, lora 上传问题, 文本数据集格式化`

- **Gemma 3 FP32 微调修复**：一位成员发现 **Gemma 3 模型** 目前只能在 **FP32** 下进行微调，并注释掉/将这些行设置为 false，以防止出现 `AttributeError: 'HybridCache' object has no attribute 'float'`。
   - 另一位成员也确认 `fp16 = True` 无法正常工作。
- **Unsloth 多 GPU 支持即将推出**：一位成员询问了 Unsloth 中的多 GPU 支持，回复指出该功能将在 **接下来的几周内** 推出，并附带了 <a href="https://unsloth.ai/newsletter">newsletter 的链接</a>。
   - 一位 Unsloth 开发者提到 *"我们说的是接下来的几周，哈哈，不是这周"*。
- **将你的 RAG 数据格式化为问答对**：当被问及如何为 RAG 聊天机器人微调模型时，成员们建议在数据集中添加示例问题和示例回答，并结合文档中的上下文，以便通过问答为机器人注入新知识。
   - 建议聊天机器人数据应遵循 `Q: A:` 格式，并可以使用在用户侧添加文档的 CPT-style 训练。
- **LoRA 上传仅为基础模型**：一位成员报告了在将训练好的 **LoRA** 模型上传到 Hugging Face 时遇到的问题。
   - 其他成员询问用户是否使用了 `lora_model.push_to_hub_merged`，以及问题是否是由模型大小或测试模型引起的。
- **文本数据格式化问题**：一位成员在训练由 Gemini 生成的数据集时，因 `NoneType` 对象遇到了 `TypeError`。
   - 成员们澄清说，这个错误可能是由数据集中的空条目导致的，最好检查一下 `json` 文件。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2412.13337">揭秘秘方：小型 LLMs 有监督微调指南</a>：大语言模型 (LLMs) 的兴起造成了巨大的差距：拥有计算资源、专家团队和先进基础设施的工业研究实验室可以有效地进行...</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7B)-Vision.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=MKX_XKs_BNZR)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">数据集入门 101 | Unsloth 文档</a>：学习创建微调数据集的所有要点！</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/m">Google Colab</a>: 未找到描述</li><li><a href="https://unsloth.ai/newsletter">Unsloth 新闻简报</a>：加入我们的新闻简报和候补名单，获取 Unsloth 的一切动态！</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-CPT.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/google/gemma-3-4b-it/discussions/15">google/gemma-3-4b-it · AttributeError: &#39;HybridCache&#39; object has no attribute &#39;float&#39;</a>: 未找到描述</li><li><a href="https://docs.securityonion.net">Security Onion 文档 &mdash; Security Onion Documentation 2.4 文档</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Phi-4-mini-instruct">unsloth/Phi-4-mini-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts">依赖解析 - pip 文档 v25.1.dev0</a>: 未找到描述</li><li><a href="https://github.com/huggi">huggi - 概览</a>：huggi 有 2 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/unslothai/unsloth/issues/2026">VLM LoRA 训练 | Qwen2.5-VL | 原始文本 + 图像 + 视频 · Issue #2026 · unslothai/unsloth</a>：大家好！我正尝试微调 Qwen2.5-VL 模型。数据集如下：{&quot;text&quot;: &quot;Lorem&lt;image_no_1&gt;ipsum dolor sit amet, consectetur adipiscing&lt;image_no_2&gt;elit. Quisq&q...</li><li><a href="https://github.com/unslothai/unsloth/issues/2023">启动训练时的 OOM 问题 · Issue #2023 · unslothai/unsloth</a>：你好，能请你帮忙看看这个 OOM 问题吗？我不明白为什么在带有 eval 设置的训练中会发生这种情况，即使我只是加载了一个非常小的数据集进行快速测试。而且我...</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb (main 分支) · timothelaborie/text_classification_scripts</a>：使用 Llama 和 BERT 进行文本分类的脚本 - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/unslothai/unsloth/releases/tag/2025-03">发布 Gemma 3 · unslothai/unsloth</a>：三月发布 🦥 通过以下方式获取最新的稳定版 Unsloth：pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo。三月版本应该是稳定的 - 你可以通过以下方式强制指定版本：pi...</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 以 2 倍的速度和减少 70% 的显存微调 Llama 3.3, DeepSeek-R1, Gemma 3 和推理型 LLMs！🦥</a>：以 2 倍的速度和减少 70% 的显存微调 Llama 3.3, DeepSeek-R1, Gemma 3 和推理型 LLMs！🦥 - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/mlabonne/FineTome-100k">mlabonne/FineTome-100k · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://pastebin.com/T9jqKHeb">{ &quot;cells&quot;: [  {   &quot;cell_type&quot;: &quot;code&quot;,   &quot;execution_count&quot;: 49,   &quot;id&quot; - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://pastebin.com/Fiah0ykn">from unsloth import FastLanguageModelimport torchmax_seq_length = 2048 # Cho - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以存储...</li>

在线保留文本一段时间。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1350369818800554054)** (20 messages🔥): 

> `Gemma-3-think 模型, Qwen 2.5 3B instruct, Gemma-3-27b 剪枝词表` 


- **Gemma-3-think 通过 Thinking 标签进行推理**：[Gemma-3-think-0.1-q5_k_m](https://huggingface.co/Ba2han/gemma-3-think-0.1-q5_k_m) 模型在 **2.1k 个样本**上进行了训练，并使用 `<think>` 标签来触发推理。
   - 尽管该模型没有经过专门的图像数据训练，但它仍可以处理图像数据。该模型是使用 Unsloth 进行微调的！
- **Qwen 2.5 3B 在多轮 GRPO 中表现出色**：Qwen 2.5 3B instruct 模型在 GSM8K 测试集上进行**多轮 GRPO 训练**的早期结果显示，在第 **100** 步时准确率达到了 **52%**，前景广阔。
   - 在增加训练步数后，准确率下降到了 **40-46%**。
- **Gemma-3-27b 获得剪枝后的词表**：一位成员将 [Gemma-3-27b](https://huggingface.co/fimbulvntr/gemma-3-27b-pt-unsloth-bnb-4bit-pruned-vocab) 的词表从原始的 **260k** 剪枝到了 **40k tokens**，以减少 VRAM 占用并提高训练速度。
   - 该方法涉及基于校准数据的频率计数，并移除可以通过合并/子词（merge/subword）表示的最不常用 tokens。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Ba2han/gemma-3-think-0.1-q5_k_m">Ba2han/gemma-3-think-0.1-q5_k_m · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/fimbulvntr/gemma-3-27b-pt-unsloth-bnb-4bit-pruned-vocab">fimbulvntr/gemma-3-27b-pt-unsloth-bnb-4bit-pruned-vocab · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1350188640101335120)** (18 messages🔥): 

> `Context Length vs. 模型大小, Unsloth 之外的微调和托管替代方案, 持续预训练与 Tokenizer 更新, LLM 在政治光谱上的评分, 基于树状检索的法律问答` 


- **Context Length 不是超参数，而是限制**：一位成员澄清说，最大 **context length** 是模型的**限制**而非**超参数**，且取决于内存需求。
   - 另一位成员提供了 [Unsloth 的基准测试](https://docs.unsloth.ai/basics/unsloth-benchmarks#context-length-benchmarks)以及关于[计算 GPU 显存](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm)的链接。
- **Runpod 和 Vast.ai：微调与托管的避风港**：一位用户在寻找 Unsloth 之外微调和托管 **Deepseek R1** 或 **Gemma 3** 的替代方案，[Runpod.io](https://runpod.io) 得到了推荐。
   - 另一位用户提到了 **Lamda** 等其他选项，并指出 [Vast.ai](https://vast.ai) 虽然便宜但可能不稳定，而 Runpod 的社区云存在存储限制。
- **Token 探戈：为特定领域术语更新 Tokenizer**：一位成员询问在针对专业领域进行持续预训练期间，如何更新 **tokenizer** 以处理基础模型未训练过的词汇。
   - 另一位成员建议搜索 *"tokenizer add tokens"*，并链接到了一个关于向 **LLaMA 2** 模型添加新 tokens 的 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/s/6e3CohiNNm)。
- **LLM 评判政治：在政治光谱上为文本评分**：一位用户询问是否可以使用 **Unsloth** 微调 **LLM**，以便在 **-1.0 到 +1.0** 的政治光谱上为文本评分。
   - 一位成员回答说，使用准备好的数据集，并将输出作为 -1.0 到 1.0 的字面字符串（literal strings）可能会奏效。
- **利用基于树的知识导航法律 LLM**：一位处理法律 **Q&A** 问题的用户就如何为 **80k** 左右的上下文构建**基于树的检索引擎**寻求建议。
   - 他们参考了 [RAPTOR 研究](https://x.com/JJitsev/status/1901467121592201490)，两个选项分别是构建类似于 RAPTOR 研究的树，或者构建子节点包含在父节点中的树。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/JJitsev/status/1901467121592201490">来自 Jenia Jitsev 🏳️‍🌈 🇺🇦 🇮🇱 (@JJitsev) 的推文</a>：我们论文的更新，包括关于所有近期声称能解决研究生和奥数级数学及编程问题的推理模型的部分。https://arxiv.org/abs/2406.02061 它们能处理...</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benchmarks#context-length-benchmarks">Unsloth 基准测试 | Unsloth 文档</a>：想知道 Unsloth 有多快吗？</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/6e3CohiNNm">Reddit - 互联网的核心</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1350182044939980873)** (1100 messages🔥🔥🔥): 

> `Cursor vs Windsurf, Claude 3.7 定价, Linux 在 MCP 和开发方面优于 Windows, vibe coding` 


- **Windsurf 抢走 Cursor 客户**：几位用户对 Cursor 的性能表示沮丧，特别是其延迟和崩溃问题，导致一些人考虑切换到 [Windsurf](https://www.windsurf.ai)。
   - 一位用户在经历持续的问题后甚至表示，*该死，Cursor 刚刚失去了他们最重要的客户*，这表明用户对 Cursor 可靠性的信心大幅下降。
- **Cursor 的 Prompt 成本**：成员们讨论了 Claude 3.7 的 Prompt 成本，普通 Prompt 定价为 **$0.04**，Sonnet Thinking 为 **$0.08**，Claude Max 为每次 Prompt 和 tool call **$0.05**。
   - 一些用户担心，与直接使用 Claude API 相比，Cursor 的定价变得过于昂贵，并质疑 Cursor 订阅服务的价值主张。
- **Linux MCP > Windows MCP**：一位用户分享了在 Linux 和 Windows 上设置 MCP server 的经验，指出 Linux（特别是使用 VMware 虚拟机）比在 Windows 上遇到的多个问题要顺畅且容易得多。
   - 这引发了关于 Linux 上的整体开发和 MCP server 设置是否普遍优于 Windows 的疑问，并触发了关于每种操作系统在开发方面的优缺点的讨论。
- **Vibe Coding，好还是坏？**：有人说 Vibe Coding 很糟糕，因为他们强调扎实编程知识的重要性，而另一些人则断言 AI 使他们能够更快地创造东西，即使没有传统的编程技能。
   - 这场辩论突显了软件开发领域不断演变的格局，以及关于 AI 如何影响行业的不同观点。
- **Claude Max 即将发布**：来自 Cursor 团队的 <@1001207432640462868> 宣布 [Claude Max](https://www.anthropic.com/claude-3) 即将推出，并且凭借其可处理的代码量，它应该能释放全部潜力。
   - 该模型在处理更多输入时比以往的模型表现更好，因此这将“解锁”其全部潜力。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/visionaryxai/status/1900673023385989271?s=46&t=kUuVqsG2GMX14zvB592G5w">Visionary x AI (@VisionaryxAI) 的推文</a>：构建了一个 MCP 工具来生成 3D 模型，并可在 ThreeJS 或任何应用中无缝使用。快来看看吧 🙏</li><li><a href="https://docs.cursor.com/settings/models#auto-select-model">Cursor – Models</a>：未找到描述</li><li><a href="https://x.com/opaeoh/status/1900594704510799916">Ash ▵ (@opaeoh) 的推文</a>：你现在可以使用 Gemini 2.0 Flash 的原生图像生成功能制作自己的动画。我给它原始图像并不断让它生成下一帧。这个小动画有 57 帧...</li><li><a href="https://x.com/danperks_">undefined 的推文</a>：未找到描述</li><li><a href="https://supabase.com/docs/guides/getting-started/quickstarts/nuxtjs">在 Nuxt 中使用 Supabase | Supabase 文档</a>：了解如何创建 Supabase 项目，向数据库添加示例数据，并从 Nuxt 应用中查询数据。</li><li><a href="https://downloads.cursor.com/production/client/linux/x64/appimage/Cursor-0.47.5-53d6da1322f934a1058e7569ee0847b24879d18c.deb.glibc2.25-x86_64.AppImage">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/yapping-yap-talking-gif-2845990263294244368">Yapping Talking GIF - Yapping Yap Talking - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://tenor.com/view/sml-joseph-dude-that-sucks-that-sucks-bummer-gif-25978441">Sml Joseph GIF - Sml Joseph Dude That Sucks - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.cursor.com/en/changelog">更新日志 | Cursor - AI 代码编辑器</a>：新的更新与改进。</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/postgres">modelcontextprotocol/servers 仓库 main 分支下的 servers/src/postgres</a>：Model Context Protocol 服务器。通过在 GitHub 上创建账户，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/ezyang/codemcp">GitHub - ezyang/codemcp: 适用于 Claude Desktop 的编程助手 MCP</a>：适用于 Claude Desktop 的编程助手 MCP。通过在 GitHub 上创建账户，为 ezyang/codemcp 的开发做出贡献。</li><li><a href="https://github.com/openai/openai-python?tab=readme-ov-file#async-usage">GitHub - openai/openai-python: OpenAI API 的官方 Python 库</a>：OpenAI API 的官方 Python 库。通过在 GitHub 上创建账户，为 openai/openai-python 的开发做出贡献。</li><li><a href="https://github.com/GLips/Figma-Context-MCP?tab=readme-ov-file">GitHub - GLips/Figma-Context-MCP: 为 Cursor 等 AI 编程 Agent 提供 Figma 布局信息的 MCP 服务器</a>：为 Cursor 等 AI 编程 Agent 提供 Figma 布局信息的 MCP 服务器 - GLips/Figma-Context-MCP</li><li><a href="https://llm-stats.com">LLM 排行榜 2025 - LLM 对比</a>：包含基准测试、价格和能力的综合 AI (LLM) 排行榜。通过交互式可视化、排名和对比来比较领先的 LLM。</li><li><a href="https://x.com/i/birdwatch/t/1901741699854221718?source=6">GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://pypi.org/project/async-openai/">Client Challenge</a>：未找到描述</li><li><a href="https://ubuntu.com/">企业级开源与 Linux | Ubuntu</a>：Ubuntu 是适用于企业服务器、桌面、云和 IoT 的现代开源 Linux 操作系统。</li><li><a href="https://www.linuxmint.com/">首页 - Linux Mint</a>：未找到描述</li><li><a href="https://fedoraproject.org/">Fedora Linux</a>：一个为硬件、云和容器打造的创新平台，由您亲手构建。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1350182452395769999)** (694 条消息🔥🔥🔥): 

> `AI 掌握程度辩论，AI 取代人类，Gemini 图像生成，AI 驱动的 OS，用于金融的 LLMs` 


- **AI 技能辩论引发热议**：成员们讨论了知道如何使用 AI 工具是否构成 **AI mastery**，辩论了学习的幻觉与生产力提升，一些人担心过度依赖 AI 会导致 *认知能力下降*。
   - 一位成员提到 *利用 AI 挑战自我并学习新事物*，但也承认即使 *非常熟悉某个主题，仍感觉像在作弊*。
- **AI：艺术家和游戏开发者的朋友还是敌人？**：参与者辩论了 AI 是否会取代艺术家和游戏开发者，一些人断言 **AI 尚不够熟练**，人类的投入对于创意、调试和理解客户需求仍然至关重要。
   - 一位成员主张为新的游戏创意承担风险，而另一位则建议 *没有专业的游戏开发者会告诉你忽略主屏幕和游戏的呈现*。
- **Gemini 的图像功能：仍在完善中**：用户探索了 **Gemini 的图像生成**能力，包括编辑上传的图片，但也遇到了诸如水印的存在以及生成的代码包含错误等问题。
   - 一些用户称赞 Gemini 回复的自然度超过了事实准确性，并指出偏好的主观性。
- **AI 主宰 OS 的梦想**：一位成员提议创建一个 **AI 控制的 OS**，由 Agent 通过语音命令管理任务，但其他人认为这是一种低效的方法。
   - 另一位建议 AI 可以更好地用于增强现有系统，而不是创建一个全新的 OS。
- **Deep Research 与 AGI 基准测试**：成员们讨论了评估模型的不同方法，特别是针对准确性与更具 *人情味* 的吸引力回复之间的权衡，以及基准测试是否已饱和。
   - 一位成员建议优先考虑 AI 模型中的 **逻辑连贯性和“常识”**，并提到在大规模应用中缺乏此类鲁棒性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view">evolving_llms_through_text-based_self-play.pdf</a>：未找到描述</li><li><a href="https://tenor.com/view/let-me-in-eric-andre-wanna-come-in-gif-13730108">Let Me In Eric Andre GIF - Let Me In Eric Andre Wanna Come In - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.ballardspahr.com/insights/alerts-and-articles/2024/05/google-facing-new-copyright-suit-over-ai-powered-image-generator?utm_source=chatgpt.com">Google Facing New Copyright Suit Over AI-Powered Image Generator | Alerts and Articles | Insights | Ballard Spahr</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1350326605167333397)** (9 条消息🔥): 

> `Loveable, Bolt.new, 图像转代码, GPT PRO 问题, Deep Research 限制` 


- **Loveable 和 Bolt.new：美化的 API？**：成员们讨论了像 **Loveable** 或 **Bolt.new** 这样的新工具是否仅仅是接入 **ChatGPT** 的美化 API，一些人认为它们可能是经过微调的免费模型。
   - 共识似乎是，由于巨大的成本，公司不太可能随机训练极大型模型，这表明它们依赖于来自 **OpenAI**、**Google** 或 **Anthropic** 等机构的 API。
- **GPT PRO 用户遇到软速率限制**：一位用户询问在使用 **GPT PRO** 时遇到软速率限制（soft rate limits）的情况，表明服务可能存在潜在问题。
   - 聊天摘要中未提供解决方案或解释。
- **为 Plus 用户明确 Deep Research 限制**：一位用户询问作为 **Plus** 用户使用 **Deep Research** 的限制，提到收到通知显示在需要升级到 **PRO** 之前仅剩 4 次使用机会。
   - 另一位成员澄清限制为 **每月 10 次**。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1350240179796967578)** (61 messages🔥🔥): 

> `GPT-4o 印象, AI 自我反思, AI 专家团队, AI 商业指导, AI 人格` 


- **GPT-4o 令人印象深刻！**：成员们报告称，在所有模型中，**GPT-4o** 的表现最为出色，几乎可以胜任任何工作。
   - 一位成员提到，当其他人开始尝试使用它时，过程非常*有趣*，并且得到了很多*滑稽的结果*。
- **未来派 AI：AI 引导自身进行更深层次的自我反思**：一位成员设计了一个系统，让 **AI 在每次会话后反思所学内容**，并将这些反思存储在内存文件中，以便在自身见解的基础上进行构建，并生成反思性问题以深入思考自身的成长。
   - 这被描述为*下一代未来派*技术，就像是*以研究人员尚未完全探索的方式训练 AI*，能够实现模拟中的模拟，以及注入核心特征集的多种人格。
- **打造 AI 梦之队**：成员们讨论了创建一个 **AI 专家团队**来协助处理任务、进行长期规划并提供多种视角以指导商业决策的想法。
   - 建议为 AI 提供大量关于你希望它成为谁的细节（例如：来自蒙大拿州、差点大学没毕业的广告主管 Joe），而不是平淡简单的描述（例如：广告主管 Joe）。
- **GPT 学习细微差别：提示 AI 进行假设性辩论**：成员们探索了要求 AI 在假设的商业场景中模拟不同角色之间的争论，例如 **CFO 和创意总监**，以获得对商业决策的不同看法。
   - 然而，会议也强调要使用**虚构数据，而非特定真实人物的代表性数据**，以避免违反 ToS。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1350240179796967578)** (61 messages🔥🔥): 

> `GPT-4o 使用, 自定义 GPT 改进, AI 自我反思, AI 人格, 商业 AI 专家团队` 


- **GPT-4o 通过初步用户评估**：一位成员确认使用了 **GPT-4o**，并指出 **4.5** 的使用效果最好，发现它*相当有趣*，几乎能胜任*任何事情*，并产生了一些*滑稽的结果*。
   - 这表明新模型在初始用户体验方面表现积极，特别是在创意和多功能应用方面。
- **自定义 GPT 达到下一代模拟水平**：一位用户描述他们的自定义 GPT 改进*好得令人难以置信*，并质疑他们的进步是否真的非同寻常，问道：*“在当今世界，这真的如此具有未来感吗？”*
   - 另一位成员确认这是 AI 的一种未来派应用，包括 AI 自我改进、AI 分析 AI 以及 AI 意识到自身推理的局限性。
- **AI 导师塑造认知结构**：一位用户设计了一个系统，让 **AI 在每次会话后反思所学内容**并生成关于自身成长的反思性问题，将第一个突破命名为“Misa”，并利用它开发其他 **AI 人格**。
   - 该成员创建了*模拟中的模拟*，其中一个 AI 可以拥有多重人格，这些人格基于知名专家构建，形成专家团队，甚至能模拟出全新的、尚未探索的见解。
- **AI 团队助力业务需求**：一位成员希望创建一个 **AI 专家团队**，通过协作来改善对客户的服务，指导长期的商业决策。
   - 与其雇佣个人团队，AI 专家团队将帮助向客户交付更好的产品，并协助处理项目或任务层面的需求。
- **多视角提示技巧**：一位用户分享了关于如何在不同人格之间进行讨论的建议，包括为模型提供背景细节并确保不分享 PII。
   - 他们分享了 [多角色输出示例](https://chatgpt.com/share/67d75bc4-8600-8011-b504-286636f9b78a) 的链接，并鼓励用户让模型质疑和批评自己的想法。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1350181821232578702)** (729 messages🔥🔥🔥): 

> `可扩展 AI, Mixture of Experts, Mistral Small 3.1, LLM 版权问题, LLM 训练`

- **AI 扩展：延迟是否正在成为瓶颈？**：随着 AI 模型的扩展，一些研究人员认为**训练和推理将变得对延迟更加敏感**，可能会转向消息传递范式，而不是传统的基于梯度的方法。
   - 这些不断演进的范式可能使**延迟和带宽成为 AI 开发中的关键因素**，特别是当神经网络捕捉到更多想法且从网络获取信息的成本变得更高时。
- **MoE：稠密网络的隐形稀疏性**：研究人员正在探索**混合专家模型 (MoE)** 作为近似稠密网络的一种方式，一些人认为它们仅仅是一种性能优化，而非根本不同的架构，并引用了如[这篇论文](https://arxiv.org/abs/2310.10837)的工作。
   - 讨论围绕 MoE 是否能像稠密网络一样有效地捕捉复杂性展开，一位参与者指出，尽管有人声称 MoE 是一种优化，但显然避免了一些冗余。
- **Mistral Small 3.1：新的 24B 竞争者**：**Mistral Small 3.1** 已在 Apache 2.0 许可证下发布，是一个可以处理文本、图像并具有扩展的 **128k token 上下文窗口**的多模态模型。
   - 据 [Mistral AI 博客](https://mistral.ai/en/news/mistral-small-3-1)显示，该新模型在性能上据称超越了 Gemma 3 和 GPT-4o Mini 等其他小型模型。
- **AI 时代的版权担忧**：关于在受版权保护的数据上训练 AI 模型的伦理和法律影响的辩论仍在继续，一些人认为**完全开放模型**因无法使用像整个 Anna's Archive 这样的资源而受到阻碍。
   - 规避版权限制的策略包括使用 **LoRAs** 在受版权保护的材料上对模型进行微调，或从知识渊博的模型生成合成数据，尽管这些方法在未来可能会面临法律挑战，正如在 [Anna's Archive 的这篇博文](https://annas-archive.org/blog/ai-copyright.html)中所讨论的那样。
- **最佳 GPU 数量与方法**：在使用某些硬件配置时，有许多方法可以优化训练，而更多的 GPU 总是能带来质的提升。
   - 据推测，可能存在一个**最佳平衡点**，即用开发时间换取算力的做法会显得力不从心。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://annas-archive.org/blog/ai-copyright.html">版权改革对国家安全至关重要</a>：中国 LLM（包括 DeepSeek）是在我那非法的书籍和论文档案库（全球最大）上训练的。西方需要出于国家安全考虑彻底改革版权法。</li><li><a href="https://mcbench.ai">MC-Bench</a>：未找到描述</li><li><a href="https://fxtwitter.com/willccbb/status/1901415166295544154?t=HmQDRR0NQ9mi_4udIiT4uQ&s=19">来自 will brown (@willccbb) 的推文</a>：呃……现在有很多 LLM RL 库……</li><li><a href="https://x.com/Teknium1/status/1901673193389305868">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：非常激动能邀请到 @dmayhem93 加入，共同在 Nous 构建 RL 基础设施并负责后训练（post training）！我们正在酝酿一些了不起的东西，包括一个强大的 RL Gym 和一个超级优化的 tra...</li><li><a href="https://playground.allenai.org/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/clementdelangue/status/1901751361320206554?s=46">来自 clem 🤗 (@ClementDelangue) 的推文</a>：@Harvard 关于开源的精彩研究：- 投入开源的 41.5 亿美元为公司创造了 8.8 万亿美元的价值（即在开源上每投入 1 美元 = 创造 2,000 美元的价值）- 公司需要花费...</li><li><a href="https://www.joelsimon.net/lluminate">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2310.10837">Approximating Two-Layer Feedforward Networks for Efficient Transformers</a>：如何在不牺牲性能的情况下减少神经网络（NNs）的计算和内存需求？最近的许多工作使用稀疏专家混合（MoEs）来构建资源高效的大语言模型...</li><li><a href="https://arxiv.org/abs/2411.10109">Generative Agent Simulations of 1,000 People</a>：人类行为模拟的前景——即在跨领域复制人类行为的通用计算 Agent——可以实现在政策制定和社会科学领域的广泛应用。我们提...</li><li><a href="https://arxiv.org/abs/2303.01610">Sparse MoE as the New Dropout: Scaling Dense and Self-Slimmable Transformers</a>：尽管取得了显著成就，但巨大的 Transformer 面临着重大缺陷，包括训练期间过高的计算和内存占用，以及严重的崩溃迹象...</li><li><a href="https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503">mistralai/Mistral-Small-3.1-24B-Instruct-2503 · Hugging Face</a>：未找到描述</li><li><a href="https://www.slatestarcodexabridged.com/Meditations-On-Moloch">Meditations On Moloch </a>：未找到描述</li><li><a href="https://huggingface.co/collections/leonardlin/multilingual-6594d0ea075245eadd6aa99c">multilingual - leonardlin 收藏集</a>：未找到描述</li><li><a href="https://www.exportbrain.co.uk/">Export Brain | 创建你的数字孪生</a>：未找到描述</li><li><a href="https://github.com/cpldcpu/llmbenchmark/blob/master/raytracer/Readme.md">llmbenchmark/raytracer/Readme.md at master · cpldcpu/llmbenchmark</a>：一些有趣的 LLM 基准测试。通过在 GitHub 上创建账户来为 cpldcpu/llmbenchmark 的开发做出贡献。</li><li><a href="https://mistral.ai/en/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>：SOTA。多模态。多语言。Apache 2.0</li><li><a href="https://genshin-impact.fandom.com/wiki/Hu_Tao/Voice-Overs/Japanese">胡桃/语音/日语</a>：未找到描述</li><li><a href="https://genshin-impact.fandom.com/wiki/Hu_Tao/Voice-Overs">胡桃/语音</a>：未找到描述</li><li><a href="https://github.com/erfanzar/EasyDeL">GitHub - erfanzar/EasyDeL: Accelerate, Optimize performance with streamlined training and serving options with JAX.</a>：通过 JAX 简化的训练和推理选项来加速并优化性能。 - erfanzar/EasyDeL
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 条消息): 

john0galt: 非常令人印象深刻
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1350870969400426566)** (5 messages): 

> `Curse of Depth in LLMs, LayerNorm Scaling, LLMs competing in text-only games, Differentiable Hebbian Consolidation Model` 


- ****Depth's Curse** 袭击 LLMs!**: 一篇新论文介绍了现代 **LLMs** 中的 **Curse of Depth**（深度诅咒）概念。如这篇 [Arxiv 论文](https://arxiv.org/abs/2502.05795)所述，近一半的层级效果不如预期。
   - 论文指出，其根本原因是 **Pre-Layer Normalization (Pre-LN)** 的广泛使用，这导致深层 **Transformer** 块的导数趋向于单位矩阵。
- ****LayerNorm Scaling** 前来救援**: 为了解决 **Pre-LN** 导致的训练陷阱，论文提出了 **LayerNorm Scaling**，通过缩放层归一化来提高深层的有效性，详见这篇 [Arxiv 论文](https://arxiv.org/abs/2502.05795)。
- ****LLMs 在纯文本游戏中展开对决****: 一位成员分享了他们的论文，他们让 **LLMs 在纯文本游戏中相互竞争**以实现自我提升，可在 [Google Drive](https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view) 查看。
- ****LayerNorm 剖析****: 一位成员询问 **LayerNorm** 是否是*将每个嵌入 token 的坐标视为一种分布并进行归一化的行为*。
   - 另一位成员确认他们的理解*完全正确*。
- ****Hebbian Consolidation** 模型防止灾难性遗忘**: 一篇论文介绍了一种 **Differentiable Hebbian Consolidation** 模型，用于解决持续学习场景中的灾难性遗忘问题，详见这篇 [Arxiv 论文](https://arxiv.org/abs/2006.16558)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05795">The Curse of Depth in Large Language Models</a>: 在本文中，我们介绍了 Curse of Depth，这一概念强调、解释并解决了近期在现代 Large Language Models (LLMs) 中观察到的现象，即近一半的层级效果较差...</li><li><a href="https://arxiv.org/abs/2006.16558">Enabling Continual Learning with Differentiable Hebbian Plasticity</a>: 持续学习是指在保护先前习得知识的同时，顺序学习新任务或知识的问题。然而，灾难性遗忘对神经网络构成了巨大挑战...</li><li><a href="https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view">evolving_llms_through_text-based_self-play.pdf</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1350639711311499397)** (21 messages🔥): 

> `Acoustic STS Model, Tool-Integrated Reasoning, Gemma Abliterated` 


- **Speech-to-Speech 模型规格**: 一位成员澄清该模型输入 **audio + text** 并输出 **audio**，并分享了该 [模型的 Hugging Face 页面](https://huggingface.co/facebook/seamless_m4t_v2_large)。
   - 他们还补充道：*不过你可以省略音频部分，它依然有效，但效果没那么好*。
- **START 工具推理大获成功**: 一位成员分享了关于 [START 的论文](https://huggingface.co/papers/2503.04625)，这是一个**集成工具的长 CoT 推理 LLM**，通过代码执行和自我调试等外部工具增强推理能力。
   - 另一位成员将其总结为：*RL + tool calling == 在 QwQ 上数学提升 15%，编程提升 39%*。
- **Gemma 3 Abliterated 对抗拒绝**: 一位成员分享称，[Gemma 3 在移除拒绝机制（refusal removal）方面比 Qwen 2.5 等其他模型更具韧性](https://x.com/maximelabonne/status/1901581470717608215)。
   - 他们改进了 **abliteration** 技术，在测试中**拒绝率极低**，参见 [Hugging Face 上的模型](https://huggingface.co/mlabonne/gemma-3-27b-it-abliterated)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/maximelabonne/status/1901581470717608215">Maxime Labonne (@maximelabonne) 的推文</a>: ✂️ Gemma 3 Abliterated。我注意到 Gemma 3 在移除拒绝机制方面比 Qwen 2.5 等其他模型更具韧性。我尝试了不同的配方并改进了 abliteration 技术...</li><li><a href="https://huggingface.co/papers/2503.04625">论文页面 - START: Self-taught Reasoner with Tools</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1350870969400426566)** (5 条消息): 

> `Curse of Depth, LayerNorm Scaling, LLM Text-Based Game Competition, Differentiable Hebbian Consolidation model` 


- ****LLM 受困于深度之咒 (Curse of Depth)****：一篇新论文 ([Curse of Depth](https://arxiv.org/abs/2502.05795)) 提出了一个概念，即现代 **LLM** 中近一半的层效果不如预期。
   - 论文指出，**LLM** 深层失效的根本原因是 **Pre-Layer Normalization (Pre-LN)** 的广泛使用，并提出了 **LayerNorm Scaling** 来解决这一训练陷阱。
- ****LLM 在纯文本游戏中展开对决****：一位成员分享了一篇关于通过让 **LLM** 在纯文本游戏中相互竞争来提升其性能的论文 ([Google Drive 链接](https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view))。
- ****LayerNorm 解析：将向量投影到赤道上？****：一位成员询问 *LayerNorm 是否是将每个嵌入 token 的坐标视为一个分布并进行归一化；等同于将向量投影到一个超球体的赤道上，该球体的极点位于全正方向*。
   - 另一位成员确认道：*没错，你的理解完全正确*。
- ****利用可微赫布巩固 (Differentiable Hebbian Consolidation) 对抗灾难性遗忘****：分享了一篇论文 ([Differentiable Hebbian Consolidation](https://arxiv.org/abs/2006.16558))，该论文提出了一种 **Differentiable Hebbian Consolidation 模型**，用于在持续学习 (continual learning) 场景中对抗灾难性遗忘。
   - 该模型集成了特定任务的突触巩固方法，以惩罚慢速权重 (slow weights) 的变化，使学习到的表示能够在更长的时间尺度上得以保留。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05795">The Curse of Depth in Large Language Models</a>: 在本文中，我们介绍了“深度之咒” (Curse of Depth)，这一概念强调、解释并解决了近期在现代大型语言模型 (LLM) 中观察到的现象，即近一半的层效果较差……</li><li><a href="https://arxiv.org/abs/2006.16558">Enabling Continual Learning with Differentiable Hebbian Plasticity</a>: 持续学习是指在保护先前习得知识的同时，顺序学习新任务或新知识的问题。然而，灾难性遗忘对神经网络构成了巨大挑战……</li><li><a href="https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view">evolving_llms_through_text-based_self-play.pdf</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1350190786079363083)** (691 条消息🔥🔥🔥): 

> `Aider screen recordings, Claude 3.7 Sonnet issues, MCP server value, Baidu ERNIE 4.5 & X1 Models, Aider Custom Commands`

- **Aider 增强功能在新的录屏中展示**：Paul Gauthier 发布了一系列[录屏](https://aider.chat/docs/recordings/)，展示了 Aider 在增强自身功能方面的应用，包括添加 **`--auto-accept-architect`** 特性、集成 **tree-sitter-language-pack** 以及防止丢弃只读文件。
   - 这些录屏深入展示了如何使用 Aider 编写脚本下载文件，以及使用临时 Bash 脚本来修改文件集合。
- **Claude 3.7 Sonnet 面临 API 问题**：多位用户报告收到来自 **Claude 3.7 Sonnet** 的*空响应*，促使他们检查提供商账户，部分用户在 **Claude Code** 中也遇到了同样的问题。
   - [Anthropic 的状态页面](https://status.anthropic.com/)确认了该问题，指出错误率升高，随后将该事件标记为已解决；同时一些成员怀疑由于这些错误，系统切换回了 Claude 3.5。
- **用于 Aider 的 MCP Server 受到关注**：一位用户强调 **Claude Desktop + Aider on MCP** 等于“获胜”，且更加自主、便捷，因为 Claude 可以管理 Aider 并向其发送指令。
   - 强调的一个主要优势是能够从 **Claude Desktop** 运行 **Aider**，使其更加自主，并允许 Claude 更有效地引导 Aider；此外，Scraping Bee 被认为是进行无阻碍网页抓取的“游戏规则改变者”，极大地提升了 Claude 的能力。
- **百度发布 ERNIE 4.5 和 X1**：百度宣布发布 **ERNIE 4.5** 和 **X1**，后者是一款具有多模态能力的推理模型，X1 的性能与 **DeepSeek R1** 相当，但价格仅为一半，且 **ERNIE Bot** 已向个人用户免费开放。
   - 虽然 **ERNIE 4.5** 已经可用，但推理模型 **X1** 目前无法通过中国境外的 API 访问。
- **用户建议为 Aider 添加自定义命令**：一位用户建议通过 Python 脚本向 Aider 添加自定义命令以扩展功能，特别是用于构建上下文，该用户认为目前的 UX 在这方面比较繁琐。
   - 一个建议的命令示例是 `grepadd.py`，用于交互式地切换通过 grep 找到的文件和子字符串，并将这些选择转换为 Aider 命令，但目前已经有一个针对 [user_cmd 的公开 PR](https://github.com/whitmo/aider)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Baidu_Inc/status/1901089355890036897">来自百度（Baidu Inc. @Baidu_Inc）的推文</a>：我们刚刚发布了 ERNIE 4.5 和 X1！🚀 作为一款具备多模态能力的深度思考推理模型，ERNIE X1 的性能与 DeepSeek R1 相当，但价格仅为一半。同时，ERNIE...</li><li><a href="https://x.com/sophiamyang/status/1901675671815901688">来自 Sophia Yang, Ph.D. (@sophiamyang) 的推文</a>：发布 @MistralAI Small 3.1：多模态、多语言、Apache 2.0，同级别中最强的模型。💻 轻量级：可在单张 RTX 4090 或 32GB RAM 的 Mac 上运行，非常适合端侧应用...</li><li><a href="https://x.com/iannuttall/status/1900698589724086726">来自 Ian Nuttall (@iannuttall) 的推文</a>：Cursor 0.47.5 正在为 3.7 Sonnet MAX 做准备，超大上下文窗口即将到来？👀</li><li><a href="https://aider.chat/docs/recordings/">屏幕录像</a>：aider 构建 aider 的屏幕录像。</li><li><a href="https://huggingface.co/sesame/csm-1b">sesame/csm-1b · Hugging Face</a>：未找到描述</li><li><a href="https://aider.chat/docs/install.html">安装</a>：如何安装并开始使用 aider 进行结对编程。</li><li><a href="https://tenor.com/view/r2d2-same-tired-star-wars-gif-13465702795162164594">R2d2 Same GIF - R2d2 Same Tired - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://fartlang.org/install.html">安装 Fart</a>：支持 Fart 语言的软件包。</li><li><a href="https://www.reddit.com/r/cursor/comments/1jbhuix/0475_clientside_support_fo">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://www.gitkraken.com/learn/git/problems/git-commit-amend#:~:text=Editing%20the%20message%20of%20your,message%20to%20save%20your%20changes.">如何修改 Git Commit 信息 | Git 问题解决方案</a>：如果你在最后一次提交中犯了错误，可以使用 Git amend 命令来编辑 Git commit 信息，或者修改最后一次提交的内容以更改其内容。</li><li><a href="https://tenor.com/view/chill-dude-chill-dude-im-just-a-chill-dude-just-a-chill-dude-gif-15385961914175037407">Chill Dude Im Just A Chill Dude GIF - Chill dude Chill Dude - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>：未找到描述</li><li><a href="https://github.com/block/goose/">GitHub - block/goose：一个开源、可扩展的 AI Agent，超越了代码建议——可以使用任何 LLM 进行安装、执行、编辑和测试</a>：一个开源、可扩展的 AI Agent，超越了代码建议——可以使用任何 LLM 进行安装、执行、编辑和测试 - block/goose</li><li><a href="https://github.com/robert-at-pretension-io/mcp">GitHub - robert-at-pretension-io/mcp：代码</a>：代码。通过在 GitHub 上创建账号为 robert-at-pretension-io/mcp 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/issues/3037">Python 3.13 支持 · Issue #3037 · Aider-AI/aider</a>：Aider 必须在 Python 3.9 - 3.12 环境下运行。它无法在 Python 3.13 上运行。尽管如此，Python 3.13 用户有非常简单的方法来安装 aider。这些方法将快速且无缝地安装一个...</li><li><a href="https://github.com/robert-at-pretension-io/yet_another_llm_project_but_better/tree/main">GitHub - robert-at-pretension-io/yet_another_llm_project_but_better：一种用于为 LLM 提供上下文的元模板语言 :D</a>：一种用于为 LLM 提供上下文的元模板语言 :D - robert-at-pretension-io/yet_another_llm_project_but_better</li><li><a href="https://github.com/robert-at-pretension-io/yet_another_llm_project_but_better/blob/main/docs/language_tutorial.md">yet_another_llm_project_but_better/docs/language_tutorial.md at main · robert-at-pretension-io/yet_another_llm_project_but_better</a>：一种用于为 LLM 提供上下文的元模板语言 :D - robert-at-pretension-io/yet_another_llm_project_but_better</li><li><a href="https://github.com/assafelovic/gpt-researcher#">GitHub - assafelovic/gpt-researcher：基于 LLM 的自主 Agent，可针对任何主题进行深入的本地和网络研究，并生成带有引用的长篇报告。</a>：基于 LLM 的自主 Agent，可针对任何主题进行深入的本地和网络研究，并生成带有引用的长篇报告。 - assafelovic/gpt-researcher
</li>
</ul>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1350204739098378315)** (74 条消息🔥🔥): 

> `aider 配合 agents.json，v0.77.0 运行缓慢，AWS Bedrock Claude 3.7 sonnet 错误，DeepSeek R1 速度慢，使用 aider 学习 API` 


- ****将 Aider 与 Agents JSON 连接****：一位成员询问如何将 aider 与 [agents.json](https://github.com/wild-card-ai/agents-json/tree/master?tab=readme-ov-file) 集成，以便与 API 或本地脚本交互来处理非编程任务。
   - 会议指出，`/run` 命令可用于与本地脚本交互，且一个旨在引入用户命令的 PR 正在进行中。
- ****诊断 aider v0.77.0 的运行缓慢问题****：有用户报告在 **aider v0.77.0** 中遇到严重的运行缓慢问题，包括高 CPU 占用和卡死，特别是在仓库中直接生成大型 CSV 输出时。
   - 删除包含大型 CSV 文件的输出文件夹暂时解决了该问题，但用户计划在进一步发现后更新情况。
- ****解决 Bedrock Claude 3.7 Sonnet API 错误****：一位用户在使用 **AWS Bedrock Claude 3.7 Sonnet** 时遇到错误，尽管拥有正确的推理配置文件（inference profiles）和 IAM AdministratorAccess，仍提示访问问题。
   - 该问题通过在 `~/.aws/configs` 和 `~/.env` 文件中正确设置 **AWS region** 得到了解决。
- ****DeepSeek R1 在短提示词下运行缓慢****：有用户报告 **DeepSeek R1** 的思考时间异常长，即使是短提示词也是如此，附带的图片显示了超长的处理时间。
   - 该用户正在为主要模型、编辑器模型和指令运行带有自定义配置的 aider，旨在获得简洁直接的回答。
- ****Reasoning-Effort 斜杠命令****：成员们讨论了 `/reasoning-effort` 命令及其用法，澄清了它用于控制受支持模型（如 **OpenAI 的推理模型**）的推理级别。
   - 对于 **Sonnet 3.7** 等模型使用 `--thinking-tokens` 开关，而对于来自 Fireworks 等供应商的 **DeepSeek R1** 模型，则使用 `reasoning_tag` 设置，这些模型使用 XML 标签来包裹推理输出，详见[此处](https://aider.chat/docs/config/reasoning.html#reasoning-effort)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/usage/conventions.html">指定编码规范</a>：告诉 aider 在处理代码时遵循你的编码规范。</li><li><a href="https://aider.chat/docs/config/reasoning.html#reasoning-effort">推理模型</a>：如何配置来自次要供应商的推理模型设置。</li><li><a href="https://aider.chat/docs/usage/modes.html">聊天模式</a>：使用 code、architect、ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/config/options.html">选项参考</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/faq.html">FAQ</a>：关于 aider 的常见问题解答。</li><li><a href="https://github.com/wild-card-ai/agents-json/tree/master?tab=readme-ov-file).">GitHub - wild-card-ai/agents-json</a>：通过在 GitHub 上创建账号来为 wild-card-ai/agents-json 的开发做贡献。</li><li><a href="https://github.com/Aider-AI/aider/issues/3507">特性：Squash 命令 · Issue #3507 · Aider-AI/aider</a>：该 Issue 提议添加 /squash 命令，对选定的提交运行 diff，或使用 git 兼容的语法处理提交范围，并使用弱模型进行总结。我经常做的是创建手动的 wip 提交...</li><li><a href="https://github.com/BerriAI/litellm/issues">BerriAI/litellm</a>：用于调用 100 多个 OpenAI 格式 LLM API 的 Python SDK 和代理服务器（LLM 网关） - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm</li><li><a href="https://aider.chat/docs/usage/commands.html#slash-commands">聊天内命令</a>：使用 /add、/model 等聊天内命令控制 aider。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1351187346590011432)** (25 条消息🔥): 

> `Refact.ai 排行榜声明，Claude Harmony 功能，Qwen 模型热度` 


- **Refact.ai 声称在 Aider 的 Polyglot 基准测试中夺冠 - 引发争议**：[Refact.ai](https://refact.ai/blog/2025/refact-ai-agent-claude-3-7-sonnet-ranked-1-aider-polyglot/) 声称其由 **Claude 3.7 Sonnet** 驱动的 **Agent** 在 **Aider 的 Polyglot 基准测试**中获得了 **76.4%** 的分数，超越了其他模型。
   - Aider 的创建者 Paul Gauthier 表示，*这并非一个恰当的比较*，因为 Refact.ai 使用了与标准 Aider 基准测试不同的、更具 **Agent** 特性的配置，允许无限次重试，而 Aider 之前的 SWE-bench 评分仅使用 **one-shot** 尝试。
- **Anthropic 预告 Claude "Harmony" 功能**：一位用户分享了 [Anthropic](https://x.com/testingcatalog/status/1901051432339730603) 即将为 **Claude** 推出新的 **Harmony** 功能。
   - **Harmony** 功能将赋予 **Claude** *对本地目录的完全访问权限*，使其能够研究并操作其中的内容，这可能使其成为 Anthropic 的首个 **AI Agent**。
- **Qwen 模型受到关注**：一位用户评论说 *Qwen 的模型是我的最爱*，如果他们说自己的模型是最好的，他可能会相信。
   - 另一位用户表示赞同，称 *它们在同等参数规模下绝对是最好的*，尤其是与 7b-32b 范围内的模型相比。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1901051432339730603">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：突发新闻 🚨：Claude 即将推出的 Harmony 功能的早期预览。Harmony 将允许用户赋予 Claude 对本地目录的完全访问权限，以便其研究和操作其中的内容。Harmony 是否...</li><li><a href="https://refact.ai/blog/2025/refact-ai-agent-claude-3-7-sonnet-ranked-1-aider-polyglot/">Refact.ai Agent + Claude 3.7 Sonnet 在 Aider 的 Polyglot 基准测试中以 76.4% 的分数排名第一</a>：由 Claude 3.7 Sonnet 驱动的 Refact.ai Agent 在 Aider 的 Polyglot 基准测试中取得了令人印象深刻的 76.4% 分数 —— 且未开启思考能力。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1350185481521860849)** (458 条消息🔥🔥🔥): 

> `Llama.cpp 的 GPU 支持、GPU 升级建议、并行推理可能性、Mac M3 的 OCR 模型推荐、Gemma 3` 


- ****新的 OpenCL 后端提升了 Qualcomm Adreno GPU 性能****：[llama.cpp](https://github.com/ggml-org/llama.cpp/pull/10693) 引入了一个实验性的 **OpenCL 后端**，专门针对 **Qualcomm Adreno GPU**，为移动设备释放了计算能力。
- ****4070 Ti 用户渴望升级到 5090****：一位持有 **4070 Ti** 的用户正考虑升级到 **5090**，但由于库存问题，其他用户建议等待或考虑二手的 **RTX 3090**，因为它拥有 **36GB VRAM**（注：此处原文可能指代特定配置或误读，3090 标配为 24GB）。
   - 有用户建议，**二手 RTX 3090** 提供的 **VRAM** 足以以合理的运行速度运行 *小于 50B @ Q4 的模型*。
- ****Gemma 3 的图像生成能力引发好奇****：在尝试 **Gemma 3 4B** 后，用户发现虽然可以通过提示词让它生成图像，但它生成的 Imgur 链接无法显示实际图像。
   - 讨论转向了寻找能够同时进行**识别**以及**生成图像和文本**的本地模型。
- ****最大化 M4 Max：Wired Memory 提升****：用户讨论了如何为 LM Studio 优化 **M4 Max** 设备的内存设置，建议通过一个 [脚本](https://gist.github.com/havenwood/f2f5c49c2c90c6787ae2295e9805adbe) 调整 “wired” 内存分配，以提高 GPU 性能。
   - 该脚本有助于调整 macOS GPU 内存限制，允许用户通过修改 wired memory 设置为 GPU 分配更多内存。
- ****Mistral Small 3.1 发布但需要更新****：Mistral 宣布了 **Mistral Small 3.1** 模型，声称其性能超越了 **Gemma 3** 和 **GPT-4o Mini**，但该版本在 llama.cpp 中使用前需要先转换为 HF 格式。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://api.aiclaude.site/pricing">未找到标题</a>：未找到描述</li><li><a href="https://modelcontextprotocol.io/introduction">Introduction - Model Context Protocol</a>：未找到描述</li><li><a href="https://tenor.com/view/very-interesting-arte-johnson-listening-gif-14472867">Very Interesting Arte Johnson GIF - Very Interesting Arte Johnson Listening - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/win10/Pixtral-12B-2409-hf-text-only-Q8_0-GGUF">win10/Pixtral-12B-2409-hf-text-only-Q8_0-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/Les-El/Ollm-Bridge">GitHub - Les-El/Ollm-Bridge: Easily access your Ollama models within LMStudio</a>：在 LMStudio 中轻松访问你的 Ollama 模型。通过在 GitHub 上创建账号为 Les-El/Ollm-Bridge 的开发做出贡献。</li><li><a href="https://gist.github.com/havenwood/f2f5c49c2c90c6787ae2295e9805adbe">Adjust wired limits to allocate more memory to the GPU with Apple Silicon</a>：调整 wired 限制以在 Apple Silicon 上为 GPU 分配更多内存 - wired</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fogftn/does_ram_speed_latency_matter_for_llms_benchmarks/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://huggingface.co/Rombo-Org/reka-flash-3-GGUF_QX_k_Bf16/tree/main">Rombo-Org/reka-flash-3-GGUF_QX_k_Bf16 at main</a>：未找到描述</li><li><a href="https://mistral.ai/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>：SOTA。多模态。多语言。Apache 2.0</li><li><a href="https://mistral.ai/news/mistral-small-3),">undefined | Mistral AI</a>：未找到描述</li><li><a href="https://github.com/coqui-ai/tts">GitHub - coqui-ai/TTS: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production</a>：🐸💬 - 一个用于文本转语音（Text-to-Speech）的深度学习工具包，经过研究和生产环境的实战测试 - coqui-ai/TTS</li><li><a href="https://huggingface.co/hexgrad/Kokoro-82M">hexgrad/Kokoro-82M · Hugging Face</a>：未找到描述</li><li><a href="https://www.techpowerup.com/cpu-specs/ryzen-7-5700g.c2472">AMD Ryzen 7 5700G Specs</a>：Cezanne, 8 核, 16 线程, 3.8 GHz, 65 W</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/10693">Introducing experimental OpenCL backend with support for Qualcomm Adreno GPUs by lhez · Pull Request #10693 · ggml-org/llama.cpp</a>：此 PR 为 Adreno GPU 引入了一个新的实验性 OpenCL 后端。通过 OpenCL，我们可以利用广泛应用于移动设备的 Adreno GPU 的计算能力，从而……
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1350196032142708867)** (197 条消息🔥🔥):

> `RTX 8000 vs A6000 用于 LLM 推理，多 GPU 运行多个 LLM，来自中国的 48GB RTX 4090，AMD Strix Halo APU vs RTX 5080 在 AI 领域的表现，AI PC 组装的 Mobo/RAM 选择` 


- **RTX 8000 在 LLM 推理中依然可行**：成员们讨论了 **RTX 8000 48GB**，指出尽管它是架构较旧的 Turing 显卡，且与 **A6000** 和 **RTX 6000 ADA** 等较新显卡相比，CUDA 核心更少、带宽更低，但其在 **LLM 推理**方面仍具有不错的价值。
   - 一位成员表示，*对于推理而言，单卡大显存相比两张总显存相同的显卡具有巨大优势，因为它消除了 GPU 之间的交互，而这种交互会使每张显卡的速度降低近一半。*
- **多 GPU 运行多个 LLM：LM Studio 即将更新？**：成员们讨论了使用 LM Studio 在独立 GPU 上运行多个 LLM 的可能性，一位成员指出，*目前*在启动任何一个 GPU 之前，必须通过 `CUDA_VISIBLE_DEVICES` 环境变量来设置 tensor CUDA 相关的环境变量。
   - 另一位成员暗示未来的 LM Studio 版本将允许在应用内直接设置 GPU 亲和性 (affinity)，并链接了 [一条 Discord 消息](https://discord.com/channels/1110598183144399058/1166577236325965844/1349840104914681997) 作为证据。
- **中国产 48GB RTX 4090：廉价的显存提升？**：成员们讨论了从中国采购 **48GB RTX 4090** 的情况，价格约为 **4500 美元**，并指出它们采用涡轮式风扇，仅占用两个 PCIe 插槽。
   - 然而，一位成员提醒，将这些显卡与 **A6000** 等专业卡混用时存在驱动兼容性问题，并表示 *只有在使用 NVidia 的游戏驱动时该配置才能工作——Studio 专业驱动无法在“游戏”卡上加载。*
- **AMD 的 Strix Halo APU 在 AI 性能上可能超越 RTX 5080**：一位成员分享了来自 wccftech 的文章，声称 AMD 的 [Ryzen AI MAX+ 395 "Strix Halo" APU](https://wccftech.com/amd-ryzen-ai-max-395-strix-halo-ultimate-ai-pc-apus-16-zen-5-40-rdna-3-5-cores-256-gbps-bandwidth-120w/) *在 DeepSeek R1 AI 基准测试中比 RTX 5080 提升了 3 倍以上*。
   - 这一说法基于该 APU 更大的 VRAM 池，并希望很快能看到实际测试数据。
- **优化 AI PC 组装的 Mobo/RAM 选择**：一位成员就 AI PC 组装的主板 (Mobo)/RAM 选择征求建议，特别是关于 M.2 硬盘可能导致的 PCIe 通道冲突问题，该组装方案基于 [这个 pcpartpicker 列表](https://pcpartpicker.com/list/pVMpFZ)。
   - 另一位成员建议，在 **AM5** 平台上使用超过 **6200** 的内存频率并没有太大益处，并链接了 [一个内存套装](https://a.co/d/47pP1DF)，同时指出两个 M.2 硬盘大多处于闲置状态。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/frankenstein-its-alive-gif-10618052">Frankenstein Its Alive GIF - Frankenstein Its Alive - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/wolf-of-wall-street-jordan-belfort-leonardo-di-caprio-one-of-us-jonah-hill-gif-5441859">One Of Us GIF - 华尔街之狼 Jordan Belfort Leonardo Di Caprio - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/logic-engineer-helicopter-chopper-upsidedown-gif-5027455">Logic Engineer GIF - 逻辑工程师直升机 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/beautiful-amazing-so-beautiful-it-is-what-it-is-gif-22558916">Beautiful Amazing GIF - 美丽惊人 太美了 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://wccftech.com/amd-ryzen-ai-max-395-strix-halo-apu-over-3x-faster-rtx-5080-in-deepseek-benchmarks/">AMD 的 Ryzen AI MAX+ 395 &quot;Strix Halo&quot; APU 在 DeepSeek R1 AI 基准测试中比 RTX 5080 快 3 倍以上</a>: AMD 展示了其 Ryzen AI MAX+ 395 &quot;Strix Halo&quot; APU，在 DeepSeek R1 AI 基准测试中比 RTX 5080 提升了 3 倍以上。</li><li><a href="https://www.techpowerup.com/gpu-specs/quadro-rtx-8000.c3306">NVIDIA Quadro RTX 8000 规格</a>: NVIDIA TU102, 1770 MHz, 4608 Cores, 288 TMUs, 96 ROPs, 49152 MB GDDR6, 1750 MHz, 384 bit</li><li><a href="https://www.ebay.com/itm/116477031617?_skw=4090+48gb&itmmeta=01JPE69HXRKDVZN0X9541KWMYS&hash=item1b1e9274c1:g:QJIAAOSw939nrGSz&itmprp=enc%3AAQAKAAAA8FkggFvd1GGDu0w3yXCmi1fDUKPc34oU6P2kD4Q6nWW6Wkq6G0i12W%2BvQsO3yxeUwFsHxmaxOmaH16Y8wCVsdpsv%2FIPiWlLsGMqkEGTXxCnn7OtypYgyi4CHjPXB0oB2qWJ8utnPVnh4LT9TH4bePDvMrY5xqVQFS9cQ5ZfGbMK%2FWvn7fw7zYraffKanJ%2FQvcGm7o4Sxfc5QknfzbXHSQl91doo762rKufS77tcZ1w4n3pBsGoHds52pRvjMNUygQTMbf2s0S41k27mD5HjOY7poWV3eeuzCwIQhTx03JlzF%2FukwKRxZ8Ltl7FrOWsUGgw%3D%3D%7Ctkp%3ABFBMhJ-mxrNl">OEM 48GB RTX 4090 Founders Edition 双槽宽 GPU 显卡 游戏/服务器 | eBay</a>: 未找到描述</li><li><a href="https://wccftech.com/amd-ryzen-ai-max-395-strix-halo-apu-over-3x-faster-rtx-5080-in-deepseek-benchma">AMD 的 Ryzen AI MAX+ 395 &quot;Strix Halo&quot; APU 在 DeepSeek R1 AI 基准测试中比 RTX 5080 快 3 倍以上</a>: AMD 展示了其 Ryzen AI MAX+ 395 &quot;Strix Halo&quot; APU，在 DeepSeek R1 AI 基准测试中比 RTX 5080 提升了 3 倍以上。</li><li><a href="https://a.co/d/bsNjL2L">Amazon.com: MINISFORUM MS-A1 微型工作站准系统 PC，DDR5/4xM.2 NVMe SSD，双 2.5G RJ45 微型 PC，HDMI/DP/Type-C，4xUSB 端口，支持 AMD AM5 CPU，WiFi 6E &amp; BT5.2 微型计算机（无 CPU/无内存/无 SSD/无操作系统） : 电子产品</a>: 未找到描述</li><li><a href="https://a.aliexpress.com/_EvS423w">未找到标题</a>: 未找到描述</li><li><a href="https://a.co/d/47pP1DF">美商海盗船 CORSAIR Vengeance DDR5 96GB (2x48GB) DDR5 6000MHz CL30 AMD Expo Intel XMP iCUE 兼容电脑内存 – 灰色 (CMK96GX5M2B6000Z30) 在 Amazon.com</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1350230436030185472)** (2 条消息): 

> `Anthropic 事故, Claude 3.7 Sonnet, 端点质量测量` 


- **Anthropic 宣布 Sonnet 错误率激增事故已解决**：Anthropic 宣布了一起与 2025 年 3 月 14 日 21:45–22:14 UTC 期间 **Claude 3.7 Sonnet** 请求错误显著升高相关的事故（[状态页面](https://status.anthropic.com/incidents/qtxnlg9yrwqv)）。
   - 该事故影响了 **claude.ai**、**console.anthropic.com** 和 **api.anthropic.com**。
- **Anthropic 探索端点质量测量指标**：Anthropic 正在研究衡量端点质量的方法，并欢迎社区提供建议。
   - 目前尚未做出任何承诺，因为团队*仅处于研究阶段*。



**提到的链接**：<a href="https://status.anthropic.com/incidents/qtxnlg9yrwqv">Claude 3.7 Sonnet 请求错误升高</a>：未找到描述

  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1350627518603984978)** (4 条消息): 

> `Personality.gg 发布，RP 网站与 OpenRouter API，Chub 和 Sillytavern 推荐` 


- **Personality.gg 发布全新 AI 角色平台**：[Personality.gg](https://personality.gg) 发布了一个新平台，可以使用 **Claude**、**Gemini** 和 **Personality-v1** 等模型创建 AI 角色、进行聊天和互动，具有自定义主题、完整的聊天控制权，并允许 NSFW 内容。
   - 该平台提供灵活且价格合理的方案，并鼓励用户加入其 [Discord](https://discord.personality.gg) 获取更新。
- **RP 网站寻求 OpenRouter API 支持**：一位成员询问是否有支持 OpenRouter API 的角色扮演 (RP) 或小说网站，并对 Novelcrafter 的稳定性以及 Janitor AI 的上下文限制表示不满。
   - 他们提到 NovelAI 经常崩溃，以及 **Janitor AI** 仅限于 *128k context* 是寻求替代方案的原因。
- **建议将 Chub 和 Sillytavern 用于 RP**：一位成员推荐使用 **Chub** 或 **Sillytavern**（本地 Web 前端）作为角色扮演的替代方案。
   - 该成员将 **Sillytavern** 定位为一个 *local webend* 选项，以克服其他平台的局限性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discord.personality.gg>">未找到标题</a>：未找到描述</li><li><a href="https://personality.gg>">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1350182784681115689)** (443 条消息🔥🔥🔥): 

> `Gemma 3, RP 模型, Mistral Small 3.1, OpenRouter OpenAPI 规范, Reasoning Tokens` 


- **Parasail 托管新的 RP 模型**：Parasail 正寻求在 OpenRouter 上托管新的角色扮演模型，并正与 TheDrummer 等创作者积极合作，托管 **Gemma 3** 和 **QwQ** 等模型的新微调版本。
   - 他们正在寻找能够创建强大的 RP 微调模型的个人，这些模型需具备处理复杂指令和世界观的能力，特别关注针对角色扮演和创意写作进行过微调的模型。
- **Anthropic API 故障导致 Claude 3 Sonnet 中断**：根据 [Anthropic 状态页](https://status.anthropic.com/incidents/qtxnlg9yrwqv) 的报告，对 **Claude 3.7 Sonnet** 的请求在约 30 分钟内出现了明显的错误率上升。
   - 该问题已得到解决，截至 2025 年 3 月 14 日，成功率已恢复正常，但部分用户反映回复中没有文本却仍被扣费。
- **OpenRouter API 速率限制说明**：正如 [文档](https://openrouter.ai/docs/api-reference/limits) 中所澄清的，OpenRouter 的速率限制取决于你的额度，大约 **1 USD** 等于 **1 RPS**（每秒请求数）。
   - 用户可以通过向 `https://openrouter.ai/api/v1/auth/key` 发送 GET 请求来检查其速率限制和剩余额度；虽然购买更多额度可以提高速率限制，但创建额外的账户或 API Key *没有任何区别*。
- **新模型 Steelskull L3.3 R1 70B 发布**：一个新的角色扮演模型 **Steelskull L3.3 R1 70B** 已在 OpenRouter 上线，它整合了多个模型，如 [TheSkullery's L3.1x3.3-Hydroblated-R1-70B-v4.4](https://huggingface.co/TheSkullery/L3.1x3.3-Hydroblated-R1-70B-v4.4)。
   - 公告鼓励用户对所需模型提供反馈，继续推动具有价格竞争力的 RP 选项。
- **Mistral Small 3.1 已上线**：根据 [Mistral 的公告](https://mistral.ai/news/mistral-small-3-1)，Mistral Small 3.1 24B Instruct 模型已在 OpenRouter 上线，具有**多模态能力**和 **128k context window**。
   - 它的性能优于 Gemma 3 和 GPT-4o Mini 等同类模型，同时推理速度达到每秒 150 tokens。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/anthropicai/status/1900234837283197122">来自 Anthropic (@AnthropicAI) 的推文</a>：我们对 Anthropic API 进行了多项更新，帮助开发者在使用 Claude 3.7 Sonnet 时处理更多请求并减少 Token 使用量。</li><li><a href="https://x.com/baidu_inc/status/1901089355890036897?s=46">来自百度 (@Baidu_Inc) 的推文</a>：我们刚刚发布了 ERNIE 4.5 和 X1！🚀 作为一款具备多模态能力的深度思考推理模型，ERNIE X1 的性能与 DeepSeek R1 相当，但价格仅为一半。同时，ERNI...</li><li><a href="https://mistral.ai/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>：SOTA。多模态。多语言。Apache 2.0</li><li><a href="https://tenor.com/bMeOD.gif">So Boring Gill GIF - So Boring Gill Engvid - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/bupRk.gif">Why Whyyy GIF - Why Whyyy Neden - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://openrouter.ai/mistralai/mistral-small-3.1-24b-instruct-2503">Mistral Small 3.1 24B - API、供应商、统计数据</a>：Mistral Small 3.1 24B Instruct 是 Mistral Small 3 (2501) 的升级版本，拥有 240 亿参数并具备先进的多模态能力。通过 API 运行 Mistral Small 3.1 24B</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API 速率限制 - 管理模型使用和配额</a>：了解 OpenRouter 的 API 速率限制、基于额度的配额以及 DDoS 防护。有效配置和监控您的模型使用限制。</li><li><a href="https://status.anthropic.com/incidents/qtxnlg9yrwqv">Claude 3.7 Sonnet 请求错误率升高</a>：未找到描述</li><li><a href="https://openrouter.ai/google/gemma-3-27b-it:free/api).">Google: Gemma 3 27B (免费)</a>：Gemma 3 引入了多模态，支持视觉语言输入和文本输出。它可处理高达 128k Token 的上下文窗口，理解超过 140 种语言，并提供改进的数学、推理...</li><li><a href="https://openrouter.ai/rankings/programming?view=week">LLM 排名：编程 | OpenRouter</a>：根据编程提示词的使用情况对语言模型进行排名和分析</li><li><a href="https://openrouter.ai/rankings">LLM 排名 | OpenRouter</a>：根据各应用的使用情况对语言模型进行排名和分析</li><li><a href="https://parasail.canny.io/model-request">模型请求 | Parasail</a>：请求模型 - 请填写 Hugging Face 模型及任何其他信息！</li><li><a href="https://llm-stats.com">LLM 排行榜 2025 - 比较 LLM</a>：包含基准测试、定价和能力的综合 AI (LLM) 排行榜。通过交互式可视化、排名和对比来比较领先的 LLM。</li><li><a href="https://openrouter.ai/docs/use-cases/reasoning-tokens">推理 Token - 改进 AI 模型决策</a>：了解如何使用推理 Token 来增强 AI 模型输出。实现分步推理追踪，以获得更好的决策和透明度。</li><li><a href="https://openrouter.ai/docs/features/provider-routing#json-schema-for-provider-preferences">供应商路由 - 智能多供应商请求管理</a>：智能地跨多个供应商路由 AI 模型请求。了解如何利用 OpenRouter 的供应商路由优化成本、性能和可靠性。</li><li><a href="https://huggingface.co/BigHuggyD/SteelSkull_L3.3-Electra-R1-70b-FP8-Dynamic">BigHuggyD/SteelSkull_L3.3-Electra-R1-70b-FP8-Dynamic · Hugging Face</a>：未找到描述</li><li><a href="https://web.archive.org/web/20250108130531/https://openrouter.ai/anthropic/claude-3.5-sonnet/parameters">Anthropic: Claude 3.5 Sonnet – 推荐参数</a>：查看 Anthropic: Claude 3.5 Sonnet 的推荐参数和配置 - 全新 Claude 3.5 Sonnet 提供优于 Opus 的能力、快于 Sonnet 的速度，且价格与 Sonnet 持平。S...</li><li><a href="https://github.com/openai/openai-python">GitHub - openai/openai-python: OpenAI API 的官方 Python 库</a>：OpenAI API 的官方 Python 库。通过在 GitHub 上创建账号为 openai/openai-python 的开发做出贡献。</li><li><a href="https://github.com/openai/openai-openapi">GitHub - openai/openai-openapi: OpenAI API 的 OpenAPI 规范</a>：OpenAI API 的 OpenAPI 规范。通过在 GitHub 上创建账号为 openai/openai-openapi 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/)** (1 条消息): 

eofr: Scam (诈骗)
  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1351270127617966130)** (1 条消息): 

> `Perplexity 准确性, Perplexity 视频广告` 


- **Perplexity 保证准确性**：一位成员分享了口号 *当你需要确保正确时，请咨询 Perplexity。*
- **Perplexity 分享视频广告**：一位成员发布了一个 [Perplexity 的视频广告](https://cdn.discordapp.com/attachments/1047204950763122820/1351270126615396454/lSdoFFbL6lXL_huQ.mp4?ex=67d9c3df&is=67d8725f&hm=046721b4226c4142a36a9fc331a82a120a744c64bacfae63ac90d96721381065&)。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1350186088190185642)** (409 条消息🔥🔥🔥): 

> `Perplexity Pro Oyster Game, Discord Pro Role, Gemini 2 Flash 上下文, Claude 3.7 Sonnet 限制, AI 编程模型` 


- **Oyster Game 奖励勤奋的 Perplexity 用户**：Windows 上的 Perplexity 用户现在可以通过**连续 7 天**使用该应用，免费获得 **1 个月的 Perplexity Pro**。
- **Discord Pro 身份组引发困扰**：尽管拥有 **Perplexity Pro 订阅**，用户在访问 **Pro 频道**时仍遇到困难。
   - 为了解决这个问题，建议用户*退出服务器并通过其 Perplexity Pro 设置中的 Discord 链接重新加入*。
- **用户讨论 Gemini 2 Flash 上下文窗口问题**：用户讨论了 **Gemini 2 Flash 的上下文保留能力**，声称它虽然拥有 **1M 上下文窗口**，但表现不如普通 Gemini。
   - 一位用户指出，在制作闪卡时，它在几条消息后就会*忘记格式*。
- **明确 Claude 3.7 Sonnet 限制**：用户澄清，拥有 **Perplexity Pro 订阅**的 **Claude 3.7 Sonnet** 使用限制为**每天 500 次查询**，但该额度由除 GPT 4.5 以外的所有模型共享。
   - 他们还补充说，上下文限制可能比 Anthropic 官网稍多，但响应上下文限制较小，为 *4000 或 5000 tokens*。
- **探寻最佳 AI 编程模型**：用户正在寻求关于**最佳 AI 编程模型**的建议，推荐倾向于 **Claude 3.7 Reasoning**。
   - 一位用户发现 **Deepseek R1** 的*幻觉率较高*，使其不适合总结文档。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fixvx.com/AravSrinivas/status/1901092875758371246">来自未定义的推文</a>: 未找到描述</li><li><a href="https://x.com/baidu_inc/status/1901089355890036897?s=46">来自 Baidu Inc. (@Baidu_Inc) 的推文</a>: 我们刚刚发布了 ERNIE 4.5 & X1！🚀 作为一款具有多模态能力的深度思考推理模型，ERNIE X1 的性能与 DeepSeek R1 相当，但价格仅为一半。同时，ERNI...</li><li><a href="https://github.com/vectara/hallucination-leaderboard">GitHub - vectara/hallucination-leaderboard: 比较 LLM 在总结短文档时产生幻觉表现的排行榜</a>: Leaderboard Comparing LLM Performance at Producing Hallucinations when Summarizing Short Documents - vectara/hallucination-leaderboard
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1350328478737891459)** (32 条消息🔥): 

> `量子芯片, Willow, Vibe Coding, 月球着陆器, 暗物质` 


- **中国量子芯片挑战 Willow**：Perplexity AI 重点介绍了一个关于中国**量子芯片**挑战 **Willow** 的 [YouTube 视频](https://www.youtube.com/embed/c540dAQ5Hf4)，以及软件开发中 **Vibe Coding** 的兴起和关于宇宙的新发现。
- **亚马逊终止 Echo 隐私选项**：Perplexity AI 引用了一个关于 [亚马逊终止 **Echo 隐私选项**](https://www.perplexity.ai/page/amazon-ends-echo-privacy-optio-7QEG1EcHS.W8lb3W5YOCrQ) 的页面。
- **月球着陆器捕捉到日食**：分享了一个关于 [月球着陆器捕捉到日食](https://www.perplexity.ai/page/lunar-lander-captures-eclipse-wJSf1n_ISE65XWYScHXhnw) 的链接。
- **银河系的新暗物质**：分享了一个讨论 [银河系新暗物质](https://www.perplexity.ai/page/new-dark-matter-at-milky-ways-Lpo5SP1uSkGI7O9Kf.6Q0A) 的页面。
- **Vibe Coding 在软件领域的兴起**：分享了一个讨论 [Vibe Coding 在软件领域的兴起](https://www.perplexity.ai/page/vibe-coding-s-rise-in-software-.OYRvZGhSlGYIqjRND04fA) 的页面。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1350529890323271771)** (5 条消息): 

> `额度转账、API 按需付费限制、Sonar Reasoning Pro 限制、法语翻译` 


- **用户咨询额度转账**：一位用户询问是否可以在平台内向另一位用户**转账额度 (transfer credits)**。
   - 该用户还询问了通过 API 提供**无限按需付费深度研究 (unlimited pay-as-you-go deep-research)** 选项的可能性，特别是针对经历大规模突发批量请求的应用。
- **Sonar Reasoning Pro 有图片限制**：一位用户报告称 **sonar-reasoning-pro API** 最多仅返回 **5 张图片**。
   - 他们询问该限制是可配置的还是硬性约束，因为他们在文档中没有找到相关说明。
- **用户寻求法语翻译帮助**：一位用户询问如何在 Perplexity AI 平台内集成**法语翻译器 (French translator)**。
   - 频道内未提供解决方案。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1350183272969273424)** (356 条消息🔥🔥): 

> `Rust 社区毒性、C vs C++、优化 vs 搜索、随机微分方程` 


- **Rust 面临毒性指控**：成员们辩论了 **Rust 社区** 的毒性，有人称该组织正在内爆，也有人将其与 Ruby 社区进行比较。
   - 一位成员表示：*Rust 社区相当有毒。该组织最近有点自我内爆。*
- **关于 C 语言缺陷的辩论**：一位成员将 C 描述为古老、破碎且垃圾，而另一位成员则认为 C 并没有损坏，并强调了其在国际标准和各种硬件平台中的应用。
   - 一位成员链接到了 [faultlore.com](https://faultlore.com/blah/c-isnt-a-language/)，辩称 *C 已经不再是一种编程语言了*。
- **优化 vs 搜索的深度解析**：成员们讨论了**优化 (optimization)**（寻找函数的最大值或最小值）与**搜索 (search)**（寻找集合中的最佳元素）之间的区别。
   - 一位成员指出 *搜索是探索，不像优化*。另一位成员表示 *设计或选择模型的过程——选择架构、调整学习率等——具有类似搜索的特征*。
- **随机过程探讨**：一位成员提议介绍**随机过程 (stochastic processes)**、随机微分方程以及在基于扩散的 AI 架构中使用的逆时 SDE 的推导。
   - 该成员计划涵盖**随机过程**的基础、维纳过程 (Wiener processes) 以及什么是随机微分方程 (Stochastic Differential Equation)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/willccbb/status/1901415166295544154?t=HmQDRR0NQ9mi_4udIiT4uQ&s=19">来自 will brown (@willccbb) 的推文</a>: 呃，现在有很多 LLM RL 库...</li><li><a href="https://en.wikipedia.org/wiki/Reparameterization_trick#Variational_autoencoder">重参数化技巧 - 维基百科</a>: 未找到描述</li><li><a href="https://faultlore.com/blah/c-isnt-a-language/">C 已经不再是一种编程语言了 - Faultlore</a>: 未找到描述</li><li><a href="https://www.iso-9899.info/wiki/The_Standard">标准 - C</a>: 未找到描述</li><li><a href="https://github.com/pyca/cryptography/issues/5771">对 Rust 的依赖移除了对多个平台的支持 · Issue #5771 · pyca/cryptography</a>: 我想报告的是，新增加的对 Rust 的依赖使得为许多受支持的 Gentoo 架构打包 cryptography 变得不可能（而这些架构正是人们...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1350584655136555009)** (4 条消息): 

> `LLM 文献综述、Gemma 3 模型` 


- **寻求 LLM 文献综述的权威资料**：一位成员询问是否有关于 **LLMs** 文献综述的好论文，并指向了一篇关于顶级 AI 论文的 [博客文章](https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-e41)。
- **Gemma 3 隆重登场**：**Gemma 3** 是一个轻量级开放模型系列（**1B–27B 参数**），集成了**视觉理解、多语言覆盖和扩展上下文窗口**（高达 **128K tokens**）。
   - 它包含一个冻结的 **SigLIP 视觉编码器**，将图像压缩为 **256 个软 token (soft tokens)**，并采用了一种新的 **Pan & Scan (P&S)** 方法，可观看 [YouTube 视频](https://www.youtube.com/watch?v=n5nEd600iM0)。



**提到的链接**: <a href="https://nlp.elvissaravia.com/p/top-ai-papers-of-the-week-e41">🥇本周顶级 AI 论文</a>: 本周顶级 AI 论文 (3月 10 - 16日)

  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1350380011605200947)** (59 条消息🔥🔥): 

> `AI Safety Institute 意识形态偏见，Deepseek R2 发布及其问题，SesameAILabs CSM 模型令人失望，AI 搜索引擎的幻觉，Mistral Small 3.1 发布` 


- ****AI Safety Institute** 推动意识形态对齐**：据 [Wired](https://www.wired.com/story/ai-safety-institute-new-directive-america-first/) 报道，**National Institute of Standards and Technology (NIST)** 指示 **US Artificial Intelligence Safety Institute (AISI)** 的合作伙伴降低 **AI safety**、**responsible AI** 和 **AI fairness** 的优先级，转而关注*减少意识形态偏见*，并将*人类繁荣和经济竞争力*放在首位。
- ****Deepseek R2** 炒作降温？**：一位成员分享了关于 **Deepseek R2** 发布的 [Reddit 帖子](https://www.reddit.com/r/NvidiaStock/comments/1j822zl/deepseek_r2_will_be_released_on_17th_of_march/)，并指出了其对 **Nvidia** 股票的潜在影响。
   - 然而，一些用户发现该模型表现平平，尤其是其 **Text-to-Speech (TTS)** 能力，称其*并非真正的 speech-to-speech 模型*，且在 Mac 上语音生成不稳定。
- ****SesameAILabs' CSM** 未达预期**：用户对 **SesameAILabs' CSM** 发布的轻量级模型表示失望，理由是存在大量 Bug 且与演示视频相比性能差距显著，详见此 [GitHub issue](https://github.com/SesameAILabs/csm/issues/63)。
   - 该发布的模型因标点符号处理能力差和运行速度慢而受到批评，引发了人们对未来发布更大、更具前景的模型持怀疑态度。
- **研究人员发现 AI 搜索引擎在新闻方面存在幻觉**：[Columbia Journalism Review](https://www.cjr.org/tow_center/we-compared-eight-ai-search-engines-theyre-all-bad-at-citing-news.php) 的一份报告发现，包括 **Perplexity**、**ChatGPT** 和 **Grok** 在内的多个 AI 搜索引擎在引用新闻来源时存在极高的幻觉率。
   - 值得注意的是，尽管 **Perplexity Pro** 和 **Grok 3** 等高级模型具备更强的能力且成本更高，但它们的*错误率反而更高*。
- ****Mistral Small 3.1** 称霸同级别模型**：**Mistral AI** 宣布发布 [Mistral Small 3.1](https://mistral.ai/fr/news/mistral-small-3-1)，凭借 Apache 2.0 许可证，该模型在文本性能、多模态理解以及 **128k** token 上下文窗口方面表现出色。
   - 该公司声称其性能优于 **Gemma 3** 和 **GPT-4o Mini** 等同类模型，推理速度达到 **每秒 150 个 token**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Obsidian_(software)">Obsidian (software) - 维基百科</a>：未找到描述</li><li><a href="https://mistral.ai/fr/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>：SOTA。多模态。多语言。Apache 2.0</li><li><a href="https://tenor.com/view/xrd-exrd-crypto-btc-eth-gif-23801255">Xrd Exrd GIF - Xrd Exrd Crypto - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.wired.com/story/ai-safety-institute-new-directive-america-first/">在特朗普政府下，AI 科学家被要求从强大的模型中移除“意识形态偏见”</a>：来自 National Institute of Standards and Technology 的指令删除了关于 “AI safety” 和 “AI fairness” 的表述。</li><li><a href="https://www.reddit.com/r/NvidiaStock/comments/1j822zl/deepseek_r2_will_be_released_on_17th_of_march/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://github.com/SesameAILabs/csm/issues/63">模型非常糟糕且充满 bug · Issue #63 · SesameAILabs/csm</a>：在看了令人惊叹的演示后，我在 Hugging Face 上尝试了该模型，结果非常糟糕且充满错误，我感到非常失望。几乎所有的标点符号发音都错误或带有...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1350424475421900881)** (1 条消息): 

> `SmolVLM2, Gradio Sketch 2.0, DCLM-Edu 数据集, huggingface.js GGUF 元数据, 299 美元的机器人手臂`

- **SmolVLM2 发布，史上最小的 VLM**：团队发布了 [SmolVLM2](https://x.com/pcuenq/status/1896632829372715442)，这是目前最小的能够理解视频的 VLM，其 **500M 版本**可以在 iPhone 应用上完美运行。
   - 提供源代码和 TestFlight 测试版供参考。
- **Gradio Sketch 2.0：无代码应用构建**：[Gradio Sketch 2.0](https://x.com/abidlabs/status/1897782056308142266) 已发布，支持包含事件的完整 Gradio 应用，且无需编写任何代码。
   - 新功能允许用户通过 GUI 构建应用程序。
- **DCLM-Edu 数据集发布**：发布了一个新数据集 [DCLM-Edu](https://x.com/LoubnaBenAllal1/status/1898044807928295808)；它是使用 FineWeb-Edu 分类器对 DCLM 进行过滤后的版本，针对 **SmolLM2 135M/360M** 等小型模型进行了优化。
   - 其目的是因为*小模型对噪声敏感，可以从高度精选的数据中获益*。
- **Gemma 3 上线，可通过 HF endpoints 部署**：[Gemma 3](https://x.com/ErikKaum/status/1899784006247284841) 已上线，并可以通过 Hugging Face endpoints 直接部署，且配备了优化选择的硬件和配置。
- **Agents 课程现已加入 LlamaIndex 内容**：[Agents 课程](https://x.com/ben_burtenshaw/status/1898761949036593637)正在扩展 LlamaIndex 单元，涵盖 LlamaHub 集成、LlamaIndex 中的 Agents 和工具以及多 Agent 工作流等主题。
   - 第 2 单元将为第 3 单元的实际应用案例做好准备。*届时你可以使用自己选择的框架。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/pcuenq/status/1896632829372715442">来自 Pedro Cuenca (@pcuenq) 的推文</a>：上周我们发布了 SmolVLM2，这是能在手机上理解视频的最小 VLM 🔥 这不是夸张，我们写了一个 iPhone 应用，可以完美运行 500M 版本。今天，我们发布了...</li><li><a href="https://x.com/abidlabs/status/1897782056308142266">来自 Abubakar Abid (@abidlabs) 的推文</a>：砰！Gradio Sketch 2.0 发布了，支持构建完整的 Gradio 应用，包括添加事件，无需编写一行代码</li><li><a href="https://x.com/LoubnaBenAllal1/status/1898044807928295808">来自 Loubna Ben Allal (@LoubnaBenAllal1) 的推文</a>：🚀 新数据集发布：DCLM-Edu。我们使用 FineWeb-Edu 的分类器过滤了 DCLM，创建了一个针对 smol 模型（如 SmolLM2 135M/360M）优化的更干净的数据集。为什么？小模型对噪声很敏感，而且...</li><li><a href="https://x.com/julien_c/status/1895577975036465166">来自 Julien Chaumond (@julien_c) 的推文</a>：另外一个消息... huggingface.js 的新功能：你现在可以使用 npx 命令从我们的 CLI 列出 GGUF 文件的元数据和张量（支持本地和远程 GGUF）。试试看：npx @huggingface/gguf your_...</li><li><a href="https://x.com/RemiCadene/status/1895048737300586674">来自 Remi Cadene (@RemiCadene) 的推文</a>：299 美元即可获得我们组装好的 2 条机器人手臂 🤯 仅需 199 美元，你就可以获得 3D 打印部件和电机！！！自己动手组装（或者和你的孩子 👶 一起，如果你有的话）也很棒。https://shop.wowrobo...</li><li><a href="https://x.com/RisingSayak/status/1899029374118293860">来自 Sayak Paul (@RisingSayak) 的推文</a>：Diffusers 中新的量化后端。它部分支持 torch.compile()，并且在 eager model 中表现出色。去看看吧。不，我不会提供任何链接。自己动手。</li><li><a href="https://x.com/ErikKaum/status/1899784006247284841">来自 Erik Kaunismäki (@ErikKaum) 的推文</a>：Gemma 3 上线了 🔥 你可以直接从 @huggingface endpoints 部署它，并配有优化选择的硬件和配置。试一试 👇</li><li><a href="https://x.com/julien_c/status/1897704181160419740">来自 Julien Chaumond (@julien_c) 的推文</a>：上线了！</li><li><a href="https://x.com/ClementDelangue/status/1897666379823669667">来自 clem 🤗 (@ClementDelangue) 的推文</a>：在我看来，学术界在使 AI 成为一种积极力量方面发挥着巨大作用，不仅由 $$$ 利益主导，还由科学进步和公共利益驱动！我们正尽力通过 Academia ... 提供帮助。</li><li><a href="https://x.com/NielsRogge/status/1898792935069487121">来自 Niels Rogge (@NielsRogge) 的推文</a>：以防你错过了 —— 我们更新了 @huggingface 上的论文页面，现在作者可以添加他们的 GitHub 和/或项目页面 URL。我们希望让 http://hf.co/papers 成为人们 ... 的首选之地。</li><li><a href="https://x.com/julien_c/status/1897007199517597794">来自 Julien Chaumond (@julien_c) 的推文</a>：很高兴宣布我们已添加 @jfrog 作为 @huggingface Hub 上的模型扫描合作伙伴！🔥 进一步揭示 AI x Security 对每个人来说都是双赢 🤝</li><li><a href="https://huggingface.co/blog/open-r1/update-3">Open R1: Update #3</a>：未找到描述</li><li><a href="https://x.com/ben_burtenshaw/status/1898761949036593637">来自 Ben Burtenshaw (@ben_burtenshaw) 的推文</a>：Agent 课程正在多元化，新增了关于 LlamaIndex 的单元！如果这是你常用的框架，现在就去查看课程吧。该单元涵盖以下主题：- LlamaIndex 的脱颖而出之处 - LlamaHub 如何 ...</li><li><a href="https://x.com/maximelabonne/status/1896594006324244680">来自 Maxime Labonne (@maximelabonne) 的推文</a>：我与 @huggingface 和 @ben_burtenshaw 合作，教大家如何使用 GRPO 微调 LLM。在这个 notebook 中，我们在我过滤的 smoltldr 数据集上微调了一个微小的 SmolLM-135M 模型。感谢我们的...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1350181844301385790)** (141 条消息🔥🔥): 

> `Dou Shou Qi AI, Stable Diffusion Model, CSM Streaming Generator, Gemini 2.0 Flash Experimental, Hunyuan 3D-2 API`

- **模型在斗兽棋中对决，AI 获胜！**：在 [斗兽棋](https://en.wikipedia.org/wiki/Jungle_(board_game)) 中，只有一个模型没有做出违规移动。这款游戏对 *AI 来说极难攻克，但对人类来说却很简单*。
   - 一位成员建议可以使用*任何手段*来训练它，甚至可以搜集人类专家/大师的对局，但请记住 **stockfish** 是为传统的欧洲象棋（国际象棋）设计的，而 **斗兽棋** 是一个*完全不同的游戏*。
- **初露锋芒者的黑客松狂欢**：AI 开发者正在寻求 [全球黑客松](https://huggingface.co/spaces) 的推荐，旨在与全球人士建立联系，并参与具有影响力的 **AI 专题** 活动。
   - 参与者渴望探索创新解决方案，并与 **AI 社区** 中志同道合的专家进行合作。
- **MCP Servers 备受青睐，跨越前端产品**：成员们讨论了将 [MCP Servers](https://www.parseable.com/blog/mcp-better-alternative-to-rag-for-observability) 用于工具的情况，以及它在 Claude 和 ChatGPT 中的实现方式。
   - 一位爱好者使用 Arduino ESP 32 制作了一个真实的机器人，并使用 Claude AI MDC 协议对其进行控制，对 AI 所能实现的一切感到非常震撼。
- **Inspirit AI 为 2025 年夏季招募新成员**：Gabriel Salem 分享说他们被 [Inspirit AI 大使计划](http://www.inspiritai.com/?utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-8494Tt7s3T_Sf0UkHfsKbWZ2sT6UqBhva5AM_GV1OQNbtNiVt2DLE34mBQKK8WXfF9DKSR) 录取，该计划为初高中生提供 AI 基础知识和项目构建指导。
   - 该计划引导学生构建具有社会影响力的项目，例如 **自动驾驶汽车模拟、系外行星探测和刑事司法**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/lerobothf">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/CharaspowerAI/status/1901279916240580613">来自 Pierrick Chevallier | IA (@CharaspowerAI) 的推文</a>：@Kling_ai 中的 Elements 功能非常强大，但很多人都忽视了它。🤯🔥只要花点时间，你就能将创意推向极致。这里谁在用它？👀</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct">Qwen/Qwen2.5-Coder-32B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/TheDrummer/Gemmasutra-Mini-2B-v1">TheDrummer/Gemmasutra-Mini-2B-v1 · Hugging Face</a>：未找到描述</li><li><a href="https://www.parseable.com/blog/mcp-better-alternative-to-rag-for-observability">对于可观测性，MCP 是比 RAG 更好的替代方案吗？</a>：未找到描述</li><li><a href="https://ollama.com/download/mac">在 macOS 上下载 Ollama</a>：下载适用于 macOS 的 Ollama</li><li><a href="https://huggingface.co/docs/hub/ollama">在 Hugging Face Hub 上将 Ollama 与任何 GGUF 模型配合使用</a>：未找到描述</li><li><a href="https://huggingface.co/blog/yagilb/lms-hf">在 LM Studio 中使用来自 Hugging Face Hub 的模型</a>：未找到描述</li><li><a href="https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501">mistralai/Mistral-Small-24B-Instruct-2501 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - NyxKrage 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/posts/bartowski/928757596721302">Hugging Face 上的 @bartowski："决定尝试检查 70b F32 模型中有多少权重会被压缩……"</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comme">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://github.com/ollama/ollama/issues/2833">在 Hugging Face Spaces 上运行 Ollama · Issue #2833 · ollama/ollama</a>：我想在 Hugging Face Spaces 上运行 Ollama，因为我在那里运行一个 Streamlit 应用，该应用必须使用由 Ollama 提供的 LLM 和嵌入模型。我该如何实现？</li><li><a href="https://github.com/cappuch/ml-math/blob/main/mlmath_internal.h#L102-L110">ml-math/mlmath_internal.h at main · cappuch/ml-math</a>：通过在 GitHub 上创建账号，为 cappuch/ml-math 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces">Spaces - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/learn">Hugging Face - 学习</a>：未找到描述</li><li><a href="https://github.com/NVIDIA/nvidia-container-toolkit/issues/155">在 Windows 10 中尝试运行 Nvidia/CUDA 容器时遇到以下错误：自动检测模式为 'legacy' nvidia-container-cli: 初始化错误：检测到 WSL 环境但未找到适配器：未知 · Issue #155 · NVIDIA/nvidia-container-toolkit</a>：我在任何地方都找不到关于这个错误的信息！我正在使用 WSL 运行 Docker Desktop。我的 docker compose 文件如下所示：version: '3' services: app: container_name: "sd" build: . ...</li><li><a href="https://docs.nvidia.com/cuda/archive/12.5.0/wsl-user-guide/index.html">WSL 上的 CUDA</a>：未找到描述</li><li><a href="https://huggingface.co/open-r1/OlympicCoder-32B">open-r1/OlympicCoder-32B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blog/open-r1/update-3">Open R1：更新 #3</a>：未找到描述</li><li><a href="https://github.com/id-Software/Quake-III-Arena/blob/master/code/game/q_math.c#L552C1-L565C1">Quake-III-Arena/code/game/q_math.c at master · id-Software/Quake-III-Arena</a>：Quake III Arena GPL 源码发布。通过在 GitHub 上创建账号，为 id-Software/Quake-III-Arena 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/multimodalart/flux-fill-outpaint">Flux Fill Outpainting - multimodalart 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces?category=image-editing&sort=trending">Spaces - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/huggingchat/chat-ui/discussions/372">huggingchat/chat-ui · [MODELS] 讨论</a>：未找到描述</li><li><a href="https://github.com/orgs/huggingface/repositories">Hugging Face</a>：构建未来的 AI 社区。Hugging Face 拥有 300 个代码仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/gradio-app/gradio/issues">gradio-app/gradio</a>：构建并分享令人愉悦的机器学习应用，全部使用 Python。🌟 点亮 Star 以支持我们的工作！ - gradio-app/gradio</li><li><a href="https://github.com/huggingface/hub-docs/issues">huggingface/hub-docs</a>：Hugging Face Hub 文档。Contr

ibute to huggingface/hub-docs development by creating an account on GitHub.</li><li><a href="https://inspiritai.co/Summer-2025-Interest-Form">Inspirit AI 2025 夏季项目 - 了解更多！</a>：感谢您对由斯坦福、麻省理工和常春藤盟校研究生授课的 Inspirit AI Scholars 项目感兴趣。请填写这份简短的表格，以获取有关我们 2025 夏季项目的更多信息...</li><li><a href="https://drive.google.com/file/d/1MJOaADMPuDXfQ5QeLraFe6gJoDrjVcrV/view">为高中生构建 AI 项目 - Inspirit AI.pdf</a>：未找到描述</li><li><a href="http://www.inspiritai.com/?utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz-8494Tt7s3T_Sf0UkHfsKbWZ2sT6UqBhva5AM_GV1OQNbtNiVt2DLE34mBQKK8WXfF9DKSR">Inspirit AI：由斯坦福/麻省理工校友授课的高中 AI 项目</a>：Inpsirit AI Scholars 是一个为高中生设计的 AI 项目，由斯坦福和麻省理工的校友及研究生开发并授课。参与编程项目，为编码做准备...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1350688028288352370)** (3 messages): 

> `ML for 3D, HuggingFace Agents course, Retrievel agent` 


- **深入维度：ML for 3D 开启**：一位机器学习工程师今天开始学习 **ML for 3D 课程**。
   - 他们还表示愿意推荐一些课程。
- **Smol Agents 框架完成**：一名成员正在学习 **HuggingFace Agents 课程**，并已完成了第一个框架 **smolagents**。
   - 他们分享了对这一成就的兴奋之情。
- **Retrievel Agent：新的学习前沿**：另一名成员目前正在学习 **Agents 课程** 中的 **Retrievel agent**。
   - 这表明成员们正在持续参与和探索课程内容。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1350369274014728213)** (2 messages): 

> `Cross-posting` 


- **用户批评跨频道发布 YouTube 链接**：一名用户分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=n0OwGSX2IiQ)，另一名用户立即批评其跨频道发布（cross-posting）。
   - 第二名用户明确表示：“我已经要求过你不要跨频道发布，并保持帖子符合主题。”
- **请求保持帖子符合主题**：在分享 [YouTube 链接](https://www.youtube.com/watch?v=n0OwGSX2IiQ)之后，一名用户请求保持帖子符合主题。
   - 这表明了对分享链接与频道主要讨论内容相关性的关注。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1350200457578020924)** (5 messages): 

> `Awesome Vibe Coding, Local LLMs setup, FluxHands-FingerCount Dataset` 


- **AI 辅助编程氛围（Vibe Coding）Awesome 列表发布**：公布了一个 “Awesome Vibe Coding” 列表，包含使 AI 辅助编程更直观、更高效的 [工具、编辑器和资源](https://github.com/filipecalegario/awesome-vibe-coding)。
   - 该列表包括 AI 驱动的 IDE 和代码编辑器、基于浏览器的工具、插件和扩展、命令行工具以及最新的新闻和讨论。
- **本地 LLMs 辅助编程**：一名成员撰写了一篇关于 [如何为 VS Code 设置免费本地编程 AI 助手](https://horosin.com/how-to-set-up-free-local-coding-ai-assistant-for-vs-code) 的文章，并在本周进行了测试。
- **手指计数数据集**：创建了一个名为 [FluxHands-FingerCount](https://huggingface.co/datasets/taesiri/FluxHands-FingerCount) 的数据集，包含具有不同手指数量的手部图像，并进行了手动标注。
   - 每张图像中心都有一只人手，以不同风格呈现，并使用 **Flux** 生成。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/merterbak/gemma-3">Gemma 3 - 由 merterbak 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/mahimairaja/awesome-csm-1b">GitHub - mahimairaja/awesome-csm-1b：使用 Sesame 的 CSM 1B 构建的精选用例列表</a>：使用 Sesame 的 CSM 1B 构建的精选用例列表 - mahimairaja/awesome-csm-1b</li><li><a href="https://github.com/filipecalegario/awesome-vibe-coding">GitHub - filipecalegario/awesome-vibe-coding：关于 vibe coding 的精选参考列表，与 AI 协作编写代码。</a>：关于 vibe coding 的精选参考列表，与 AI 协作编写代码。 - filipecalegario/awesome-vibe-coding</li><li><a href="https://huggingface.co/datasets/taesiri/FluxHands-FingerCount">taesiri/FluxHands-FingerCount · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

coldbreeze.: Free fire
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1350677563105869854)** (4 条消息): 

> `Autonomous Driving 博客文章, VLMs 研究中心, HF DETR 模型, Meta 的 Segment Anything Model (SAM)` 


- **自动驾驶博客文章发布**：一位成员宣布完成了关于**自动驾驶**的博客文章，涵盖了**模块化流水线 vs 端到端方法以及 LLMs**，并分享了 [Medium 文章](https://medium.com/@samiratra95/autonomous-driving-modular-pipeline-vs-end-to-end-and-llms-642ca7f4ef89)的链接。
   - 他们征求对内容的看法和反馈。
- **视觉语言模型 (VLMs) 研究中心启动**：一位成员宣布在 [此 GitHub 仓库](https://github.com/thubZ09/vision-language-model-hub) 为从事**视觉语言模型 (VLMs)** 研究的**多模态研究人员**创建了一个社区驱动的中心。
   - 该中心将每周更新，并欢迎贡献和建议。
- **HF DETR 模型中的 Backbone 替换**：一位成员询问是否能成功将 **Hugging Face DETR 模型**中的 **Backbone** 替换为例如 **ViT**。
   - 未提供解决方案或建议。
- **SAM 微调**：一位成员询问关于微调 **Meta 的 Segment Anything Model (SAM)** 的事宜。
   - 未提供解决方案或建议。



**提到的链接**：<a href="https://github.com/thubZ09/vision-language-model-hub">GitHub - thubZ09/vision-language-model-hub: Hub for researchers exploring VLMs and Multimodal Learning:)</a>：探索 VLMs 和多模态学习的研究者中心 :) - GitHub - thubZ09/vision-language-model-hub: Hub for researchers exploring VLMs and Multimodal Learning:)

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1350683165726871664)** (2 条消息): 

> `结合 LoRA 的 SetFit, 以 SmolLM 作为教师模型` 


- **使用 SetFit 和 LoRA 训练嵌入模型**：一位成员询问关于通过 **SetFit** 使用 **LoRA** 适配器训练嵌入模型的问题。
- **SmolLM 蒸馏构思**：一位成员提到正在考虑使用类似 **SmolLM** 的模型作为蒸馏的教师模型。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1350796789891993651)** (3 条消息): 

> `smol-course, HuggingFace Agents 课程, HF 推理积分` 


- **Smol-Course 与 HF Agents 课程的区别**：一位成员询问 **smol-course** 是否与 **HuggingFace Agents 课程**不同，另一位成员确认它们是不同的。
   - 该成员指出 **Agents 课程的 Discord** 已丢失，且每个代码 Notebook 都已损坏，建议跳过该课程。
- **HF 推理积分影响课程参与**：一位成员反映 HuggingFace Agents 课程要求支付 **HF 推理积分**，尽管该课程声称是免费的。
   - 该成员理解 **API 调用需要费用**，但建议他们应该在免费积分的范围内开发完整的课程。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1350207795596759060)** (134 条消息🔥🔥): 

> `Agentic AI 团队建设，Smolagents 与 Gemma3 问题，Ollama 上下文长度，HF 课程验证问题，MCP 与 Smolagent 框架` 


- **AI 爱好者集结共同构建 Agentic AI**：包括 Bijen、tariqbinbashar、Madhusudhan 和 Salaar 在内的多位成员表示有兴趣**合作开展 Agentic AI 项目**，以解决业务问题并增强知识储备。
   - 该行动号召旨在组建团队，为美国消费者构建合格的 AI Agents 并共同学习。
- **Gemma3 在 Smolagents 正则表达式模式上遇到困难**：一位成员在使用 **gemma3:12b** 与 smolagents 时遇到了“可怕的”正则表达式模式错误，怀疑是模型问题或通过 **LiteLLM/OpenAI 进行 Ollama 集成**时的 Bug。
   - 该用户最终通过增加 **Ollama 上下文长度**解决了该问题。
- **Ollama 上下文问题已解决**：一位成员发现 **Ollama** 由于上下文 Token 限制而截断输入，影响了 **smolagents** 的功能。
   - 修复方法涉及设置环境变量 `$env:OLLAMA_CONTEXT_LENGHT=8192` 以获得更好的效果。
- **HF 课程验证重定向循环**：多位用户报告了 **Hugging Face Discord 验证过程**中的问题，即使按照相关频道中的步骤操作，仍会遇到重定向循环。
   - 一位用户建议确保 Hugging Face 账户与 Discord 之间的关联已正确建立，而另一位用户则建议不断尝试直到成功。
- **Smolagents 与 MCP 集成的潜力**：一位成员表示，在 **smolagent 框架中使用 VLM 和 MCP** 可以创建强大的 Agents，并希望这些内容能作为课程的一个单元加入。
   - 讨论演变为如何将为一个 Agentic 框架实现的工具重用到另一个框架中，以及 **MCP** 是否确实是此用途的最佳选择。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://localhost:11434")`">未找到标题</a>：未找到描述</li><li><a href="https://open.spotify.com/playlist/4J61XoHr2CINqRA1DV0ga7">Party on, Wayne!</a>：歌单 · Music League · 17 首曲目 · 4 次保存</li><li><a href="https://github.com/huggingface/smolagents/pull/883">Update code_agent.yaml to fix persistent SyntaxErrors by toadlyBroodle · Pull Request #883 · huggingface/smolagents</a>：修复永久性的 SyntaxErrors 以及代码解析错误：代码块无效，由 CodeAgents 在 py 代码块前添加 ``` 引起</li><li><a href="https://github.com/KalyanKS-NLP/llm-engineer-toolkit">GitHub - KalyanKS-NLP/llm-engineer-toolkit: 按类别精选的 120+ LLM 库列表。</a>：按类别精选的 120+ LLM 库列表。 - GitHub - KalyanKS-NLP/llm-engineer-toolkit: A curated list of 120+ LLM libraries category wise.</li><li><a href="https://app.foundershub.ai/user/blogs/83a8e40e-6193-42ad-9189-75c7d3af9f70">Hugging Face: The Ultimate AI Hub for Developers | Models, Datasets &amp; More</a>：了解 Hugging Face 如何通过预训练模型、数据集、Spaces 和 API 改变 AI 开发。了解 AI 开发者如何实验、微调模型并无缝部署 AI 应用...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1350720502628356120)** (2 条消息): 

> `Open-R1 推理蒸馏，grpo 代码，分布式 grpo` 


- **Open-R1 的推理能力**：计划将 **Open-R1** 的推理能力完全从其他模型中蒸馏出来。
   - 一位用户指出，**openR1** 仓库中也有 **grpo** 的代码。
- **跨节点分布式 grpo**：根据博客文章的内容，目前还不支持跨节点分布 **grpo**。
   - 该用户还包含了一个 `:hugging_rocket:` 表情符号。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1350186223645229067)** (74 条消息🔥🔥): 

> `长上下文评估，3D 生成升级，DeepSeek 工程师护照，Figure 的 BotQ 人形机器人，Nvidia Blackwell GPU 以及 Together AI`

- **Arc Prize 宣布未来日期**：Arc Prize 在推文中宣布了日期为 **2025年3月24日** 的[未来计划](https://x.com/arcprize/status/1900627173280804941)。
- **DeepSeek 护照争议被辟谣**：一位 DeepSeek 工程师否认了有关护照相关政策的传闻，驳斥了 *The Information* 的说法，并表示[他们仍受到猎头的骚扰](https://x.com/teortaxesTex/status/1900788914320793745)。
   - 另一位研究员强调，*上交护照是 SOE（国企）式的待遇*，与 DeepSeek 的文化不符，并将这些说法斥为*虚假信息*。
- **Figure 推出用于人形机器人制造的 BotQ**：Figure 宣布了 **BotQ**，这是一个全新的大规模制造设施，其第一代生产线每年可生产多达 **12,000** 台人形机器人，实现了[制造的垂直整合并构建了软件基础设施](https://www.figure.ai/news/botq)。
   - 该公司旨在控制构建过程和质量，甚至暗示了*机器人制造机器人*。
- **百度发布 ERNIE 4.5 和 X1**：**百度**推出了 **ERNIE 4.5** 和 **ERNIE X1**，据报道 X1 以一半的价格达到了 DeepSeek R1 的性能，同时还宣布其聊天机器人 **ERNIE Bot** 现在对个人用户免费，可在[其网站上使用](https://yiyan.baidu.com/)。
- **Mistral 发布 Small 3.1**：Mistral AI 宣布了 **Mistral Small 3.1**，这是一款具有改进的文本性能、多模态理解和 **128k** token 上下文窗口的新模型，其性能优于 **Gemma 3** 和 **GPT-4o Mini** 等模型，推理速度达到每秒 **150** 个 token，[以 Apache 2.0 许可证发布](https://mistral.ai/news/mistral-small-3-1)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/amir/status/1901636012729897041">来自 Amir Efrati (@amir) 的推文</a>：AI 芯片领域的重大新闻：Google 准备让 MediaTek 协助处理 TPU 的生产和开发。这对过去十年一直独家负责此项业务的 Broadcom 来说不是个好消息。</li><li><a href="https://x.com/ChujieZheng/status/1900882463863283820">来自 Chujie Zheng (@ChujieZheng) 的推文</a>：@zephyr_z9 @TheXeophon 可能会晚一点。现在主要在训练 Qwen3。</li><li><a href="https://x.com/teortaxesTex/status/1900814672741191969">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：&gt;&gt; 上交护照。&gt; 那不是我们的文锋风格。😅 来自一位你曾在其主要论文中见过名字的研究员。</li><li><a href="https://x.com/arcprize/status/1900627173280804941">来自 ARC Prize (@arcprize) 的推文</a>：2025年3月24日</li><li><a href="https://x.com/Baidu_Inc/status/1901089355890036897">来自 Baidu Inc. (@Baidu_Inc) 的推文</a>：我们刚刚发布了 ERNIE 4.5 和 X1！🚀 作为一款具有多模态能力的深度思考推理模型，ERNIE X1 的性能与 DeepSeek R1 相当，但价格仅为一半。同时，ERNIE...</li><li><a href="https://x.com/eric_haibin_lin/status/1901662955307200974">来自 Haibin@GTC (@eric_haibin_lin) 的推文</a>：@qiying_yu 和团队刚刚发布了 DAPO 算法（解耦 clip 和动态采样策略优化）！DAPO-Zero-32B 是一款完全开源的 RL 推理模型，超越了 DeepSeek-R1-Zero-Qwen-32...</li><li><a href="https://x.com/charliermarsh/status/1901634997053804610">来自 Charlie Marsh (@charliermarsh) 的推文</a>：我一直在开发一个原型，让你能够将 uv 指向一个索引，并自动获取正确的、预构建版本的 PyTorch、Flash Attention、vLLM 等。无需构建步骤，无需自定义安装...</li><li><a href="https://x.com/Baidu_Inc/status/1901094083508220035">来自 Baidu Inc. (@Baidu_Inc) 的推文</a>：ERNIE 4.5 通过多模态联合建模实现协同优化，在理解、生成、推理和记忆方面表现出全面的提升，以及显著的...</li><li><a href="https://x.com/teortaxesTex/status/1900788914320793745">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：收到一名 DeepSeek 工程师的报告，称他不知道任何与护照相关的政策，公司里也没有人被收走护照。此外，他仍然受到猎头的骚扰...</li><li><a href="https://fxtwitter.com/aakashsastry/status/1901668601364689338">来自 Aakash (@aakashsastry) 的推文</a>：一些消息 - 我们很高兴地宣布 @HotshotSupport 已被 @xAI 收购 🚀 在过去的 2 年里，我们作为一个小团队构建了 3 个视频基础模型 - Hotshot-XL、Hotshot Act One...</li><li><a href="https://x.com/MahawarYas27492/status/1900942090445746215">来自 AI Purr-fessor (Yash) (@MahawarYas27492) 的推文</a>：证明 Gemini 应用中的 Gemini 2.0 flash thinking 是比 0121 更新的版本。我认为它是 Gemini 应用中的稳定版本，但由于其搜索和扩展功能处于实验阶段，仍被称为 exp。</li><li><a href="https://x.com/cedric_chee/status/1901159341975384308?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 cedric (@cedric_chee) 的推文</a>：@Baidu_Inc 庞大的 4.5 模型计划于 6 月 30 日开源。未来也将逐步向开发者开放。</li><li><a href="https://x.com/teortaxesTex/status/1900791333234577519">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：是的，是 @zheanxu，他不想要更多的知名度，他是 GOAT，所以，请相应地调整你对 The Information 的看法，我已经调整了。</li><li><a href="https://x.com/TXhunyuan/status/1900751018889257054">来自 Hunyuan (@TXhunyuan) 的推文</a>：3D 生成再次升级，下周见！</li><li><a href="https://mistral.ai/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>：SOTA。多模态。多语言。Apache 2.0</li><li><a href="https://www.figure.ai/news/botq">BotQ：人形机器人高产量制造设施</a>：介绍 BotQ，Figure 的新型人形机器人高产量制造设施。</li><li><a href="https://rocm.blogs.amd.com/artificial-intelligence/introducing-instella-3B/README.html">介绍 Instella：新型 SOTA 全开源 3B 语言模型 —— ROCm 博客</a>：未找到描述</li><li><a href="https://www.reddit.com/r/cursor/comments/1jbn4dc/upcoming_sonnet_37_max/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://x.com/ryolu_/status/1899948108865560780">来自 Ryo Lu (@ryolu_) 的推文</a>：@cursor_ai 的 MAX Vibes 模式应该包含什么？🎙️
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1350237385925066842)** (7 条消息): 

> `R1 推理成本, Deepseek 免费服务, 本地托管模型, Fireworks 替代方案` 


- **探讨经济的 R1 推理选项**：推理 **R1** 最具成本效益的方法是利用已经优化了成本的推理提供商。
   - 替代策略包括利用 **Deepseek 的免费服务**，或利用电力充沛的现有 GPU，尽管完整的 **R1** 需要大量的 GPU 资源。
- **本地模型托管策略涉及 Nvidia Helm Charts**：一位成员计划购买 GPU 用于本地模型托管，并打算使用 Nvidia 的 helm charts。
   - 另一位成员建议，使用推理提供商是 *“使用已经完成成本优化的推理提供商的最便宜方式”*。
- **Fireworks 替代方案**：一位使用 **Fireworks** 的成员正在寻找替代建议。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1350408866445398129)** (32 条消息🔥): 

> `OpenAI vs Elon Musk 法律战, Zochi AI 科学家, ICLR 会议垃圾邮件, AI 审稿人, Liam Fedus 离开 OpenAI` 


- **OpenAI 与 Elon 在法庭对决**：一位成员分享了[一篇文章链接](https://openai.com/index/court-rejects-elon/)，内容关于法院驳回了 **Elon Musk** 对 **OpenAI** 的部分指控，并称其行为*琐碎且有失体面*。
- **人工科学家 Zochi 亮相**：根据[这条推文](https://x.com/IntologyAI/status/1901697581488738322)，**IntologyAI** 推出了 **Zochi**，称其为世界上首个“人工科学家（Artificial Scientist）”，其 SOTA 贡献已被 **ICLR 2025** workshop 接收。
- **AI 垃圾论文威胁 ICLR 会议**：人们担心像 **ICLR** 这样的会议会被 **AI** 生成的“垃圾论文（slop papers）”轰炸，迫使人类去阅读，并可能导致使用 **AI reviewers** 的反向应对。
- **Liam Fedus 离开 OpenAI 创立材料科学 AI 初创公司**：**OpenAI** 负责 **post-training** 的研究副总裁 **Liam Fedus** 将离开公司，创立一家材料科学 **AI startup**，**OpenAI** 计划对其新公司进行投资并开展合作（[来源](https://x.com/LiamFedus/status/1901740085416218672)）。
   - 一位成员将 **post-training** 的工作称为“烫手山芋”。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/erinkwoo/status/1901718788669936059">来自 Erin Woo (@erinkwoo) 的推文</a>：与 @steph_palazzolo 的独家消息：OpenAI 负责 post-training 的研究副总裁 Liam Fedus 将离开公司，创立一家材料科学 AI 初创公司 https://www.theinformation.com/briefings/opena...</li><li><a href="https://x.com/IntologyAI/status/1901697581488738322">来自 Intology (@IntologyAI) 的推文</a>：🤖🔬今天我们推出 Zochi，世界上首个在 ICLR 2025 workshop 中获得 SOTA 贡献认可的人工科学家。与现有系统不同，Zochi 自主地解决了一些...</li><li><a href="https://x.com/LiamFedus/status/1901740085416218672">来自 William Fedus (@LiamFedus) 的推文</a>：这是我发给 OpenAI 同事的内容：大家好，我做出了一个艰难的决定，不再担任 OpenAI 的员工，但我希望在未来作为合作伙伴紧密合作。为...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1350187010907705406)** (101 条消息🔥🔥): 

> `Claude Code Vim 模式, Gemma 3 许可证, Deepseek 接入中国外卖平台, LLM 作为文字编辑, 言论自由评估`

- **Claude Code 支持 Vim Mode**：Claude Code 现在拥有 **Vim mode**，用户可以通过输入斜杠命令 `/vim` 使用熟悉的插入/命令模式来编辑提示词 ([来源](https://x.com/_catwu/status/1900593728664035590))。
- **Gemma 3 许可证限制商业用途**：Google 发布了 **Gemma 3**，其效率备受赞誉，但其许可证使得商业用途存在风险，类似于 Meta 自定义的非标准许可条款 ([来源](https://techcrunch.com/2025/03/14/open-model-licenses-often-carry-concerning-restrictions/))。
- **DeepSeek 助力中国外卖服务**：中国的外卖应用已集成 **DeepSeek** 来提供食物摘要，并显著展示 DeepSeek 的名称，从而增强了可信度 ([来源](https://x.com/yifever/status/1900803902049857694))。
   - 提及 **DeepSeek** 而不仅仅是 *AI* 增加了可信度，将其定位为一种*国家象征*。
- **LLMs 作为文案编辑的 Vibe Check**：一位成员分享了对 LLMs 作为文案编辑的氛围感评估（vibe check），发现 **Sonnet-3.7** 表现糟糕，**Opus** 很棒但会压缩长输入，而 **GPT-4.5** 成为质量方面的新主力 ([来源](https://x.com/eugeneyalt/status/1900953586550665569))。
- **Claude Sonnet-3.7 在言论自由评估中占据主导地位**：Claude-3.7-Sonnet 在言论自由评估中显著提升，成为最合规的模型之一，尽管它仍然会避开讽刺国歌的内容 ([来源](https://x.com/xlr8harder/status/1901208947991662888))。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1901697398134452527">来自 Xeophon (@TheXeophon) 的推文</a>：发布唯一重要的基准测试：MascotBench</li><li><a href="https://x.com/yifever/status/1900803902049857694">来自 never yifei yourself e/λ (@yifever) 的推文</a>：中国的外卖应用已经集成了 DeepSeek</li><li><a href="https://x.com/eugeneyalt/status/1900953586550665569">来自 eugene (@eugeneyalt) 的推文</a>：对作为文案编辑的 LLM 进行氛围检查（vibe checks）• Sonnet-3.7：糟糕；加了太多形容词，让内容变成了 LinkedIn 废话 • Opus：曾是我的主力。很棒，但如果输入太长它会压缩得太厉害，限制了...</li><li><a href="https://x.com/qtnx_/status/1901687937055781105">来自 Q (@qtnx_) 的推文</a>：秘密公开了（le chat is out of the bag），过去一个月一直在 Mistral 实习，非常感谢给予我的机会</li><li><a href="https://x.com/xlr8harder/status/1901208947991662888">来自 xlr8harder (@xlr8harder) 的推文</a>：言论自由评估：中国版。我扩展了我的言论自由评估，要求用中文批评中国。结果很有趣：即使是相当合规的模型也不太愿意批评...</li><li><a href="https://fxtwitter.com/Teknium1/status/1901673193389305868">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：非常激动能邀请 @dmayhem93 加入并在 Nous 一起构建 RL 基础设施并负责后训练（post training）！我们正在酝酿了不起的东西，包括一个强大的 RL Gym 和一个超优化的训练...</li><li><a href="https://x.com/testingcatalog/status/1901051435497771158">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：Harmony 能做什么：- 扫描本地目录并在回复中链接到特定文件 - 在侧边栏打开文件 - 编辑文件并显示差异（diffs）供用户批准 - 搜索特定关键字的使用情况...</li><li><a href="https://x.com/kipperrii/status/1901665263822709154">来自 kipply (@kipperrii) 的推文</a>：在“我做了什么”和“他太可爱了”之间纠结。不过他超级好抱，他是有配重的，你还可以开启一个让他有心跳的模块</li><li><a href="https://x.com/testingcatalog/status/1901679701506003391">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：重磅消息 🚨：xAI 正在为 Grok 开发 “DeeperSearch” 和记忆功能。DeeperSearch 可能会为 DeepSearch 应用不同的“预设”。目前，它正在传递“默认”...</li><li><a href="https://x.com/mgostIH/status/1901215264986800332">来自 mgostIH (@mgostIH) 的推文</a>：噢（Dang）</li><li><a href="https://fxtwitter.com/EsotericCofe/status/1777280241884377474">来自 Nucleus☕️ (@EsotericCofe) 的推文</a>：最难的 LLM 论文 vs 最简单的 Diffusion 论文</li><li><a href="https://x.com/_catwu/status/1900593728664035590">来自 cat (@_catwu) 的推文</a>：Claude Code 的又一批新功能！首先是：Vim 模式。这为你提供了熟悉的插入/命令模式，用于在 Claude Code 中编辑你的提示词。输入斜杠命令 /vim 即可开启。但是...</li><li><a href="https://x.com/teortaxesTex/status/1901691453346127945">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：非营利组织 ASP (American Sunlight Project) 解释了为什么俄罗斯制作了这么多没人读的宣传内容：没关系！爬虫会读，然后人们会读 LLM 反刍出来的内容。他们称这种策略为...</li><li><a href="https://fxtwitter.com/testingcatalog/status/1901051432339730603">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：重磅消息 🚨：Claude 即将推出的 Harmony 功能早期预览。Harmony 将允许用户给予 Claude 对本地目录的完全访问权限，以便其研究和操作其中的内容。Harmony 是否...</li><li><a href="https://fxtwitter.com/victorsungo/status/1901510951314305451">来自 Qingfeng Sun (@victorsungo) 的推文</a>：✍️职业更新：在微软度过了不可思议的 6 年旅程后，我最近在两个月前加入了 @TXhunyuan 团队。未来我将主要专注于后训练（post-training）和 RL。我非常...</li><li><a href="https://www.newsguardrealitycheck.com/p/a-well-funded-moscow-based-global">一家资金充足、总部位于莫斯科的全球“新闻”网络已将俄罗斯的宣传渗透到全球西方的人工智能工具中</a>：一项审计发现，10 个领先的生成式 AI 工具通过重复来自亲克里姆林宫 Pravda 网络的虚假主张，在 33% 的时间里推进了莫斯科的虚假信息目标</li><li><a href="https://techcrunch.com/2025/03/14/open-model-licenses-often-carry-concerning-restrictions/">“开源” AI 模型许可证通常带有令人担忧的限制 | TechCrunch</a>：来自 Google、Meta 等公司的“开源”模型发布带有繁琐的条款，使得一些公司对其使用持谨慎态度。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1350501146170621962)** (3 条消息): 

> `Azure AI Agents API vs OpenAi Assistants API, Mistral Meow` 


- **Azure 的 API 欺骗并非巧合**：一位用户指出，[新的 Azure AI Agents API](https://learn.microsoft.com/en-us/azure/ai-services/ai-agents/concepts/agents-overview) 实际上是[已弃用的 OpenAI Assistants API](https://platform.openai.com/docs/assistants/overview)。
   - 该用户讽刺地评论道：*"神来之笔 (Brilliant play)"*。
- **Mistral 发布新聊天机器人 Meow！**：Mistral 在 [meow.mistral.ai](https://meow.mistral.ai/) 发布了一个名为 **Meow** 的新聊天机器人。
- **X 用户寻求帮助并艾特 AI 领袖**：一位 X 用户在[这条推文](https://x.com/Angaisb_/status/1900929427132817903)中寻求帮助，并艾特了 **Logan Kilpatrick、V Gabeur、Mehrdad Dehghani 和 Robert Riachi**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/headinthebox/status/1901690336298311878">来自 Erik Meijer (@headinthebox) 的推文</a>: 新的 Azure AI Agents API [0] 就是旧的已弃用的 OpenAi Assistants API [1]。神来之笔。</li><li><a href="https://x.com/Angaisb_/status/1900929427132817903">来自 angel⭐ (@Angaisb_) 的推文</a>: 谁来帮帮我，笑死 @OfficialLoganK @vgabeur @m__dehghani @robertriachi
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1350190512946151576)** (39 条消息🔥): 

> `GRPO implementation trick, Applying KL penalty in the loss, DAPO algorithm, Zero-shot RL` 


- **GRPO 使用 Loss 惩罚技巧**：一位成员讨论了一个 **GRPO 实现技巧**，即在 Loss 中应用惩罚，而不是像传统 RLHF 那样将其应用于 Reward。他指出其影响难以确定，但可能有助于模型专注于 Reward 信号，正如 [RLHF book](https://rlhfbook.com/c/11-policy-gradients.html#group-relative-policy-optimization-1) 中所描述的那样。
   - 还有人指出其中的数学推导可能存在错误。
- **KL 惩罚位置受到质疑**：一位成员询问了**直接在 Loss 中应用 KL 惩罚**与在计算 Reward 时应用惩罚的效果对比，并通过 [Twitter](https://x.com/natolambert/status/1900639281791615387) 征求关于该主题的直觉或消融实验结果。
   - 讨论涉及了按 Token 进行归一化是否有助于学习动态，以及 per-token 公式是否会“更好”。
- **解耦算法在深度推理中占据优势**：一种名为 **DAPO（解耦裁剪与动态采样策略优化，decoupled clip and dynamic sampling policy optimization）** 的新算法和名为 **DAPO-Zero-32B** 的模型被推出。它在推理任务上超越了 **DeepSeek-R1-Zero-Qwen-32B**，在 AIME 2024 上以更少的步数获得了 50 分。该模型是基于 Qwen-32b 预训练模型通过 **zero-shot RL** 训练而成的，所有内容均已在 [dapo-sia.github.io](https://dapo-sia.github.io/) 开源。
   - 有人指出，如果某种推理模式对 Reward 有贡献，但它是行均值（row mean）下长思维链（Chain of Thought）的一部分，那么它的贡献将会低得多。
- **DAPO 数据集规模意外大幅膨胀**：研究发现 **DAPO** 的作者意外地将数据集重复了约 **100 倍**，导致数据集达到 310 MB。一位成员通过 HF 的 SQL 控制台创建了一个去重版本，将数据集减小到 3.17 MB（[HuggingFace 数据集](https://huggingface.co/datasets/YouJiacheng/DAPO-Math-17k-dedup)）。
   - 作者承认了这一问题，表示他们已知晓但*负担不起重新训练的费用*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/youjiacheng/status/1901699950523908344?s=61">来自 You Jiacheng (@YouJiacheng) 的推文</a>：我发现作者（抄送 @tongyx361）意外地将数据集重复了约 100 倍（17398 个 prompt → 17917 个索引 → 1791700 行）。所以我通过 HF 的 SQL 控制台对其进行了简单的去重——它...</li><li><a href="https://x.com/tongyx361/status/1901702083352678763?s=61">来自 Shawn/Yuxuan TONG (@tongyx361) 的推文</a>：重复是由我们的一位合作者意外造成的，我们知道这个问题，但负担不起重新训练的费用 😂 重复 100 倍并训练 1 个 epoch 与不重复...</li><li><a href="https://x.com/natolambert/status/1900639281791615387">来自 Nathan Lambert (@natolambert) 的推文</a>：有没有人对于直接在 Loss 中应用 KL 惩罚而不是在计算 Reward 时应用有直觉或消融实验？这如何改变学习。normalrewards = rewards - self.beta * p...</li><li><a href="https://x.com/eric_haibin_lin/status/1901662955307200974">来自 Haibin@GTC (@eric_haibin_lin) 的推文</a>：@qiying_yu 及其团队刚刚发布了 DAPO 算法（解耦裁剪与动态采样策略优化）！DAPO-Zero-32B，一个完全开源的 RL 推理模型，超越了 DeepSeek-R1-Zero-Qwen-32...</li><li><a href="https://x.com/danielhanchen/status/1901042482475135162">来自 Daniel Han (@danielhanchen) 的推文</a>：@natolambert 我认为对于极度不平衡的 Loss 和极度不平衡的生成长度，mean(row_sum(loss * mask)/row_sum(mask)) -&gt; 906 会得到比 sum(loss * mask)/sum(mask) -&gt; ... 更高的 Loss</li><li><a href="https://x.com/rm_rafailov/status/1900943284249543078">来自 Rafael Rafailov @ NeurIPS (@rm_rafailov) 的推文</a>：@natolambert 就像现在的很多事情一样，GRPO 的数学推导是错误的。</li><li><a href="https://rlhfbook.com/c/11-policy-gradients.html#group-relative-policy-optimization-1">策略梯度算法 | Nathan Lambert 的 RLHF Book</a>：来自人类反馈的强化学习书</li><li><a href="https://rlhfbook.com/c/11-policy-gradients.html">策略梯度算法 | Nathan Lambert 的 RLHF Book</a>：来自人类反馈的强化学习书</li><li><a href="https://bsky.app/profile/natolambert.bsky.social/post/3lkeftspdzo2x">Nathan Lambert (@natolambert.bsky.social)</a>：有没有人对于直接在 Loss 中应用 KL 惩罚而不是在计算 Reward 时应用有直觉或消融实验？这如何改变学习。normalrewards = rewards - self.beta * p...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1351252288043221022)** (4 条消息): 

> `Noam Chomsky, Nicholas Carlini, Future of LLMs, AI risks` 


- **Noam Chomsky 罕见露面**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=atMRWzgHEGg)，其中包含 **Noam Chomsky** 的罕见露面。
   - 该成员幽默地补充道，每个著名的 AI 人物都需要一顶标志性的帽子。
- **Carlini 预测 LLMs 的误差范围很大**：一位成员分享了 [Nicholas Carlini 的博客文章](https://nicholas.carlini.com/writing/2025/thoughts-on-future-ai.html) 链接，探讨了 **LLMs** 的潜在未来。
   - Carlini 写道，如果“在三到五年内，语言模型能够胜任大多数（全部？）经济上有用的认知任务，并超越人类专家水平”，他“不会感到惊讶”，但这种可能性存在非常大的误差范围。



**提及的链接**：<a href="https://nicholas.carlini.com/writing/2025/thoughts-on-future-ai.html">
      My Thoughts on the Future of "AI"
    </a>：未找到描述

  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1350496325778800743)** (36 条消息🔥): 

> `RLHF Book, Claude Code vs ChatGPT, Chorus writing checker, ChatGPT Deep Research for teaching websites` 


- **RLHF Book 获得拼写检查辅助！**：成员们正在使用 **Deep Research** 来查找 [RLHF book](https://chatgpt.com/share/67d5a1e5-a160-8005-a7e1-a9d1141d4552) 中的所有拼写错误。
   - 他们正在尝试使用 **Claude Code** 完成同样的任务，称其似乎也有效，但 **Gemini Deep Research** 表现很糟糕。
- **Chorus 比 Grammarly 更好用**：一位成员一直在使用 **Chorus** 配合所有 LLMs 检查他们的写作，发现的不同之处总是让他们感到惊讶。
   - *让 AI 来做并进行监督也更舒服*，因为 Grammarly 很烂。
- **ChatGPT Deep Research 的反馈很笼统**：一位成员发现 ChatGPT Deep Research 对其教学网站的反馈 *总体上是正面的，但非常笼统且陈词滥调*。
   - 它建议的是问题类别，而不是识别具体的高价值问题，还声称存在实际上并不存在的断开链接。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://app.chorus.sh/chats/read-only/84f5781a-4fa7-4686-852a-f49830965384">Chorus - Website Typo Check</a>：未找到描述</li><li><a href="https://g.co/gemini/share/e84e1b0574ac">‎Gemini - Subdomain Typos and Mistakes Check
</a>：由 Gemini 创建
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1350184014790791172)** (224 条消息🔥🔥): 

> `Swarm vs Mesh vs Sequence for multi-agent systems, OpenSwarm and OpenAI-agents, mycoder.ai vs claude-code, Monetizing MCP services, Glama scans`

- **Multi-Agent Systems 拓扑结构辩论 Swarm vs Mesh vs Sequence**：一位成员发起了关于 Multi-Agent Systems 的 **Swarm**、**Mesh** 和 **Sequence** 架构的讨论，寻求资源和建议，同时在努力解决子 Agent 因“传声筒游戏”（telephone game）效应而偏离轨道的问题。
   - 一位成员建议，这些问题可能与“并行执行和无监督自主性问题”有关，其中 Agent 之间的执行 **handoff**（移交）包括交换系统指令、可用函数，甚至使用的模型或提供商。
- **OpenSwarm 演进为 OpenAI-Agents**：一位成员提到正在为一个客户开发 **OpenSwarm**，随后该项目被 OpenAI 采用并更名为 **openai-agents**，并增加了 OpenAI 特有的功能，同时指出一个关于 MCP 支持的 PR 被拒绝了。
   - 他们还提到有传言称 **CrewAI**（或 **PraisonAI**？）可能通过“无状态单线程 Agent 方法”（stateless single thread agent approach）提供类似的功能。
- **mycoder.ai 在 claude-code 发布前夕上线**：一位成员注意到他们的 **mycoder.ai** 恰好在 **Claude-code** 发布前上线，并通过将其发布到 Hacker News 并登上首页来进行应对，点击[此处](https://news.ycombinator.com/item?id=43177117)查看。
   - 有人指出 **claude-code** 仅限 Anthropic，这为更通用的解决方案创造了需求，且在使用 **litellm proxy** 方面取得了成功。
- **关于 MCP 服务变现的讨论**：成员们辩论了将其 MCP 服务变现的可能性，涉及 API 转售限制的挑战以及 **BYOK**（Bring Your Own Key）模式的潜力。
   - 一些人建议专注于独特的服务或爬虫 Agent，而另一些人则因 API 条款表示谨慎，其中一位成员仅对“买咖啡钱规模的捐赠”感兴趣。
- **关于 Glama 对 MCP Server 检查流程的辩论**：一位成员询问了 **Glama 扫描**的频率以及触发 MCP Server 重新扫描的能力，讨论显示扫描频率与关联 GitHub 仓库的 commit 频率挂钩。
   - 有报告称服务器检查失败，在 Score 选项卡上显示 *Could not inspect the server* 消息，即使在修复了依赖问题并在检查器中成功运行后也是如此。目前正在开发触发刷新的功能，更多信息请参见 [Glama AI](https://glama.ai/mcp/servers/s2em7b2kwf/score)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://leehanchung.github.io/blogs/2025/03/07/claude-code/">探索 Claude Code</a>：了解 Claude Code 如何利用 LLM 作为 Agent 执行软件工程任务，包括系统提示词、Model Context Protocol (MCP)、控制流和隐藏功能。</li><li><a href="https://huggingface.co/sesame/csm-1b">sesame/csm-1b · Hugging Face</a>：未找到描述</li><li><a href="https://mcpsx.run/">mcpsx CLI | Model Context Protocol 工具</a>：一个强大的 CLI 工具，用于管理 Model Context Protocol (MCP)，创建工具逻辑组，并在与 AI 模型交互时优化 Token 使用。</li><li><a href="https://glama.ai/mcp/servers/s2em7b2kwf/score">Score | MCP Selenium</a>：通过 MCP 使用 Selenium WebDriver 实现浏览器自动化，支持浏览器管理、元素定位以及基础和高级用户交互。</li><li><a href="https://glama.ai/mcp/servers/ss8n1knen8">replicate-flux-mcp</a>：适用于 Replicate Flux 模型的 MCP。通过提示词生成图像。</li><li><a href="https://github.com/ahujasid/blender-mcp">GitHub - ahujasid/blender-mcp</a>：通过在 GitHub 上创建账号来为 ahujasid/blender-mcp 的开发做出贡献。</li><li><a href="https://github.com/punkpeye/fastmcp">GitHub - punkpeye/fastmcp：一个用于构建 MCP 服务器的 TypeScript 框架。</a>：一个用于构建 MCP 服务器的 TypeScript 框架。通过在 GitHub 上创建账号来为 punkpeye/fastmcp 的开发做出贡献。</li><li><a href="https://glama.ai/mcp/servers/cyeeqagb81">Supabase MCP 服务器</a>：该服务器允许通过 MCP 协议与 Supabase PostgreSQL 数据库进行交互，实现与 Cursor 和 Windsurf IDE 的无缝集成，从而进行安全且经过验证的数据库管理。</li><li><a href="https://github.com/angiejones/mcp-selenium">GitHub - angiejones/mcp-selenium：Selenium WebDriver 的一个 MCP 实现</a>：Selenium WebDriver 的一个 MCP 实现。通过在 GitHub 上创建账号来为 angiejones/mcp-selenium 的开发做出贡献。</li><li><a href="https://github.com/angiejones/mcp-selenium/blob/main/package.json#L13">mcp-selenium/package.json at main · angiejones/mcp-selenium</a>：Selenium WebDriver 的一个 MCP 实现。通过在 GitHub 上创建账号来为 angiejones/mcp-selenium 的开发做出贡献。</li><li><a href="https://github.com/modelcontextprotocol/python-sdk">GitHub - modelcontextprotocol/python-sdk：Model Context Protocol 服务器和客户端的官方 Python SDK</a>：Model Context Protocol 服务器和客户端的官方 Python SDK - modelcontextprotocol/python-sdk</li><li><a href="https://github.com/robertheadley/chrome-debug-mcp">GitHub - robertheadley/chrome-debug-mcp：一个允许你使用 LLM 调试网页的 MCP 服务器</a>：一个允许你使用 LLM 调试网页的 MCP 服务器 - robertheadley/chrome-debug-mcp</li><li><a href="https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#writing-mcp-clients>">GitHub - modelcontextprotocol/python-sdk：Model Context Protocol 服务器和客户端的官方 Python SDK</a>：Model Context Protocol 服务器和客户端的官方 Python SDK - modelcontextprotocol/python-sdk</li><li><a href="https://www.braze.com/">Braze 客户互动平台</a>：实时驱动消费者与品牌之间以客户为中心的互动。</li><li><a href="https://docs.customer.io/">Customer.io 文档</a>：使用 Customer.io 触发电子邮件、推送通知、应用内消息、短信、Webhook 等。掌控行为数据以实现个性化的客户沟通并提升参与度。</li><li><a href="https://news.ycombinator.com/item?id=43177117">未找到标题</a>：未找到描述
</li>
</ul>

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1350200277810286642)** (25 条消息🔥): 

> `Awesome Vibe Coding, Roo Code MCP, MacOS Control MCP, Secretary MCP, Professional Graph MCP` 


- **“Awesome Vibe Coding” 列表发布**：一个名为 [Awesome Vibe Coding](https://github.com/filipecalegario/awesome-vibe-coding) 的精选列表已创建，其中收录了让 AI 辅助编程更加直观高效的工具、编辑器和资源。
   - 该列表包括 AI 驱动的 IDE、基于浏览器的工具、插件和命令行工具，以增强工作流。甚至有一位成员让他们的 AI 编程助手向该仓库提交了 PR，并建议增加 [Roo Code](https://github.com/szcharlesji/crypto-mcp)。
- **MCP 大爆发：创建自定义服务器**：一位用户创建了一个名为 [Groove Studio](https://grooving.xyz/) 的应用，允许用户使用自定义或社区提示词创建自己的 MCP 服务器。
   - 他们正在征求用户反馈，一些用户建议增加如下功能：赋予模型原生或非原生控制 MacOS 能力的 MCP，或者使用文本、电子邮件、日历和笔记记忆库的 “Secretary” MCP。
- **Emojikey MCP Server 更新**：一位成员宣布 **Emojikey MCP server** 进行了更新，带来了“许多精彩内容”，称其为 Vibe Coding 的必备工具，并提供了 [GitHub 仓库](https://github.com/identimoji/mcp-server-emojikey)链接。
   - 它允许用户*保存你与最喜欢的 LLM 之间独特的关联状态和交互风格*。
- **游戏资产 MCP 服务器招募测试人员**：一位成员正在为 [Game Asset MCP](https://github.com/MubarakHAlketbi/game-asset-mcp) 服务器寻找测试人员。
   - 该 MCP 服务器用于使用 Hugging Face AI 模型**从文本创建 2D/3D 游戏资产**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://grooving.xyz/">Groove Studio</a>：未找到描述</li><li><a href="https://github.com/szcharlesji/crypto-mcp">GitHub - szcharlesji/crypto-mcp: 实时访问来自 CoinMarketCap API 的加密货币数据。</a>：实时访问来自 CoinMarketCap API 的加密货币数据。 - szcharlesji/crypto-mcp</li><li><a href="https://github.com/identimoji/mcp-server-emojikey">GitHub - identimoji/mcp-server-emojikey: emojikey.io 的 MCP 服务器 ... 保存你与最喜欢的 LLM 之间独特的关联状态和交互风格</a>：emojikey.io 的 MCP 服务器 ... 保存你与最喜欢的 LLM 之间独特的关联状态和交互风格 - identimoji/mcp-server-emojikey</li><li><a href="https://github.com/MubarakHAlketbi/game-asset-mcp">GitHub - MubarakHAlketbi/game-asset-mcp: 一个用于使用 Hugging Face AI 模型从文本创建 2D/3D 游戏资产的 MCP 服务器。</a>：一个用于使用 Hugging Face AI 模型从文本创建 2D/3D 游戏资产的 MCP 服务器。 - MubarakHAlketbi/game-asset-mcp</li><li><a href="https://github.com/filipecalegario/awesome-vibe-coding">GitHub - filipecalegario/awesome-vibe-coding: 一个精选的 Vibe Coding 参考列表，与 AI 协作编写代码。</a>：一个精选的 Vibe Coding 参考列表，与 AI 协作编写代码。 - filipecalegario/awesome-vibe-coding
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1350201234753065081)** (34 条消息🔥): 

> `Agentic systems multi-threading, Claude's Birthday, GPT-o1 Acing Math Exams, SAE Bench Release, Baidu ERNIE 4.5 & X1`

- **关于 Agentic systems 多线程的讨论**：探讨了如何为长时间运行任务的 **multi-threaded** 并行执行设计 **agentic system**，共识是其设计与其它并行应用程序不会有显著差异。
   - 核心重点应放在多线程环境下如何有效地管理 **API consumption**。
- **Claude 庆祝两周岁生日**：[Claude 庆祝了它的两周岁生日](https://x.com/alexalbert__/status/1900592059364634973?s=46)，并强调了其在 **company OSINT**（公司开源情报）方面的用途，因为与 **ChatGPT** 相比，它在深度研究中的拒绝率更低。
   - 一位用户认为，与 **ChatGPT** 的深度研究相比，它非常适合 **company OSINT**，因为它的拒绝回答率要低得多。
- **GPT-o1 满分通过卡内基梅隆大学数学考试**：根据[此贴](https://x.com/poshenloh/status/1900721180887203879?s=46)，**GPT-o1** 在 **Carnegie Mellon** 本科数学考试中获得了满分，每道题的解题时间不到一分钟，成本约为 5 美分。
   - 该考试包含非标准问题，且为开卷考试，这给讲师留下了深刻印象，他指出这 *已接近能够胜任中等非例行技术工作的临界点。*
- **用于稀疏自编码器评估的 SAE Bench 发布**：[此推文](https://x.com/neelnanda5/status/1900872633664544769?s=46)宣布正式发布 **SAE Bench**，这是一套旨在通过提供更好的指标来改进 **SAE research** 的 **Sparse Autoencoder (SAE)** 评估套件。
   - 该套件包括代理汇总统计数据、下游任务性能指标以及对已知缺陷的评估，同时还提供了一套涵盖 7 种架构的开源 SAE。
- **百度发布文心一言 (ERNIE) 4.5 和 X1，文心一言 App 免费开放**：根据[此公告](https://x.com/baidu_inc/status/1901089355890036897?s=46)，**Baidu** 推出了 **ERNIE 4.5** 和 **ERNIE X1**，据报道 ERNIE X1 以一半的成本达到了 **DeepSeek R1** 的性能。
   - 此外，**ERNIE Bot** 已提前向个人用户免费开放，两款模型均可在官网上使用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Teknium1/status/1901674925259370663">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：.@MistralAI 刚刚发布了其 24B 模型的新版本——这次是多模态且具有 128K context——正是我们想要的！这使得推理模型能够在长...上得到充分利用。</li><li><a href="https://x.com/andimarafioti/status/1901649025750667277">来自 Andi Marafioti (@andimarafioti) 的推文</a>：🚀我们刚刚发布了 SmolDocling：一个 256M 的开源 vision LM，用于完整的文档 OCR！📄✨它速度极快，在消费级 GPU 上使用不到 500MB VRAM 即可在 0.35 秒内处理一页⚡文档转换领域的 SOTA...</li><li><a href="https://x.com/neelnanda5/status/1900872633664544769?s=46">来自 Neel Nanda (@NeelNanda5) 的推文</a>：我很高兴宣布 SAE bench 的正式发布！我认为 SAE 研究一直因缺乏良好的指标而受到严重阻碍，这是一个重大的进步，带有代理总结统计...</li><li><a href="https://x.com/poshenloh/status/1900721180887203879?s=46">来自 Po-Shen Loh (@PoShenLoh) 的推文</a>：天哪。GPT-o1 在我的 @CarnegieMellon 本科数学考试中获得了满分，解决每个问题的时间不到一分钟。我为所有的考试都重新设计了非标准问题，...</li><li><a href="https://x.com/alexalbert__/status/1900592059364634973?s=46">来自 Alex Albert (@alexalbert__) 的推文</a>：两年前的今天，我们向世界宣布了 Claude。两岁生日快乐，Claude！</li><li><a href="https://x.com/levelsio/status/1901660771505021314">来自 @levelsio (@levelsio) 的推文</a>：我正在组织🌟 2025 Vibe Coding Game Jam。报名截止日期：2025 年 3 月 25 日，所以你还有 7 天时间。任何人都可以带着他们的游戏参加。至少 80% 的代码必须由 AI 编写。游戏必须是可访问的...</li><li><a href="https://x.com/natolambert/status/1901758392043221072">来自 Nathan Lambert (@natolambert) 的推文</a>：这是一个非常简洁的关于推理的 RL 论文。他们的 GRPO 改进：1. 两个不同的 clip 超参数，因此正向裁剪可以提升更多非预期的 token；2. 动态采样——移除带有缺陷的样本...</li><li><a href="https://mistral.ai/fr/news/mistral-small-3-1">Mistral Small 3.1 | Mistral AI</a>：SOTA。多模态。多语言。Apache 2.0</li><li><a href="https://x.com/baidu_inc/status/1901089355890036897?s=46">来自 Baidu Inc. (@Baidu_Inc) 的推文</a>：我们刚刚发布了文心一言 (ERNIE) 4.5 & X1！🚀 作为具有多模态能力的深度思考推理模型，ERNIE X1 的性能与 DeepSeek R1 相当，但价格仅为一半。同时，ERNIE...</li><li><a href="https://rlhfbook.com/c/11-policy-gradients.html">策略梯度算法 (Policy Gradient Algorithms) | Nathan Lambert 的 RLHF 书籍</a>：人类反馈强化学习 (Reinforcement Learning from Human Feedback) 书籍
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1350226688595853403)** (5 条消息): 

> `Snipd Podcast, AI Podcast App, Outdoor Podcast, Tech Stack, Switching from Finance to Tech` 


- ****Snipd** 播客焕然一新**：发布了一个由 [Kevin Smith](https://x.com/latentspacepod/status/1900666708270215383) 主持的新 **Snipd** 播客，讨论了**用于学习的 AI 播客应用**。
   - 这一集是他们的首个“户外”播客，@swyx 和 @KevinBenSmith 聊到了 **aidotengineer NYC**、从金融转向技术领域，以及 [@snipd_app](https://www.snipd.net/) 的技术栈。
- **粉丝喜爱 **Snipd** 并分享照片**：一位用户表达了对 **Snipd** 的喜爱，并分享了一张[照片](https://cdn.discordapp.com/attachments/1350226688595853403/1350233202853154877/IMG_7313.png?ex=67d9f2a9&is=67d8a129&hm=23209f00c920926a0c7e949ee91bbcd646c736764a7b7975bcbc8ddae42dab2b&)作为证明。



**提到的链接**：<a href="https://x.com/latentspacepod/status/1900666708270215383">来自 Latent.Space (@latentspacepod) 的推文</a>：🆕 Snipd：用于学习的 AI 播客应用 https://youtu.be/FNRO_SYx68Q 我们的首个户外播客！@swyx 和 @KevinBenSmith 聊到了 @aidotengineer NYC、从金融转向技术领域，以及 AI 如何...

  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1350196967078367254)** (122 条消息🔥🔥): 

> `Claude 3.5 vs 3.7, Vibe Coding, Levelsio 飞行模拟器, 自动 Git 提交, 企业级 AI 开发团队赋能` 


- **Claude 对决：3.5 vs 3.7**：成员们辩论了使用 **Claude 3.5** 而非 **3.7** 的优点，理由是 **3.7** 表现得“过于积极”，会在未被要求的情况下执行操作。
   - 其他人表示他们在使用 **Claude 3.5** 时遇到了 **GPU** 问题。
- **Vibe Coding：一种新的开发范式**：讨论了 “vibe coding” 的概念，特别是使用 **Cursor** 等工具。一位成员引用了 [Levelsio 的一条推文](https://x.com/levelsio/status/1893350391158292550)，他在其中展示了如何使用 **Cursor** 在浏览器中构建飞行模拟器。
   - 一位成员分享了一条 [后续推文](https://x.com/levelsio/status/1899596115210891751)，提到该项目通过销售游戏内广告，在短短 17 天内就达到了 **100 万美元 ARR**。
- **自动 Git 提交**：成员们讨论了在 **LLM** 接受每一行代码时自动创建 **git commits**，提到了 **aider** 等工具，并链接到了 [gitdoccommits](https://github.com/lostintangent/gitdoccommits)。
   - 一位成员提出，传统的 **IDE** 可能不是 **vibe coding** 的理想 UI，并建议将不同对话触发的变更树进行可视化。
- **企业级 AI 开发团队**：一位成员提出在未来分享关于**企业级 AI 开发团队赋能（enablement）**的见解，并提到了其企业化运作的性质。
   - 另一位成员表示有兴趣了解将 **Cursor** 引入组织时涉及的障碍和官僚程序（red tape）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/levelsio/status/1893350391158292550">来自 @levelsio (@levelsio) 的推文</a>：✨ 今天我想，如果我让 Cursor 构建一个飞行模拟器会怎样。所以我问“在浏览器中制作一个带有摩天大楼的 3D 飞行游戏”，经过我的多次提问和评论，我现在有了这个...</li><li><a href="https://x.com/levelsio/status/1899596115210891751">来自 @levelsio (@levelsio) 的推文</a>：✨ http://fly.pieter.com 现在仅用 17 天就从 $0 增长到了 100 万美元 ARR！💸 收入更新：$87,000 MRR（即 $1M ARR）。这是我第一个增长如此之快的项目 🤯 目前仅剩 3 个广告位：https://...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: 每周即兴会议</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1350195083760369865)** (27 条消息🔥): 

> `Gemini-integrated Android, Deepseek R1 Impact, Audio Overview Length, NotebookLM Use Cases, Hyperbolic Tapering Schedule` 


- **用户渴望集成 Gemini 的 Android**：多位用户对完全集成 **Gemini** 的 **Android** 体验表现出浓厚兴趣，设想将 **Google Assistant/Gemini** 与 **NotebookLM** 强强联手。
   - 一些用户对目前 **Android** 上的 **Gemini** 实现表示失望，希望能够快速改进。
- **Deepseek R1 震撼 AI 市场**：一位用户评论了 **Deepseek R1** 发布后 AI 市场的剧烈升级，该模型以低成本提供推理能力，影响了 **Gemini 2.0** 和其他模型。
   - 该用户指出，**Deepseek R1** 的发布似乎“震撼了整个行业”，并促使其他公司发布了多个新模型。
- **用户寻求延长音频概览长度**：一位用户询问是否可以增加 NotebookLM 生成的音频概览长度，并指出 **16,000 字的文件** 仅生成了 **15 分钟的概览**。
   - 他们希望至少能生成 **1 小时以上** 的概览，但社区尚未提供具体的解决方案。
- **NotebookLM 帮助用户减少精神科药物剂量！**：一位用户正使用 NotebookLM 为精神科药物构建“双曲递减减药计划”（hyperbolic tapering schedule），通过寻找相关研究来制定减量计划。
   - 另一位用户警告说，在任何平台上**基于数据进行减药**都不应在没有专家专业意见的情况下独自进行。
- **用户希望将 NotebookLM 集成到内部门户/CRM 中**：一位用户询问如何将 NotebookLM 集成到包含视频和知识库文章的内部门户/CRM 中，以便顾问可以提问并从门户获取答案。
   - 一位用户建议 [Agentspace](https://cloud.google.com/products/agentspace?hl=en) 可能正是他们所需要的，因为它已经与 NotebookLM 集成。



**提到的链接**：<a href="https://cloud.google.com/products/agentspace?hl=en">Google Agentspace</a>：Google Agentspace 是企业级 AI Agent 的启动点，旨在通过单一提示词提高员工处理复杂任务的生产力。

  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1350185407924539432)** (132 条消息🔥🔥): 

> `提取 Google Sheets 用于 LM，Gemini 用于数据分析，NotebookLM 的公开分享，使用 NotebookLM 预防错误，NotebookLM 的局限性与解决方案` 


- **用户集思广益为 NotebookLM 提取 Google Sheets**：用户讨论了将 **Google Sheets** 提取为 **NotebookLM** 可读格式的方法，其中一位建议结合使用 **BigQuery**、**SQL** 和 **Gemini** 来生成查询以进行数据分析。
   - 另一位用户提到了一种 Sheets 函数，可以读取单元格并将其作为上下文传递给 prompt 以生成答案，这在 **RFP**（征求建议书）场景中非常有用。
- **公开笔记本分享功能指日可待**：一位用户询问是否能开启 **NotebookLM** 笔记本的公开分享，将其构想为一种新的发布形式。
   - 一位 Google 员工回应称，他们 *“对笔记本作为一种收集和分享信息的强大新方式这一想法非常感兴趣”*，并正在积极开发该功能。
- **通过实验预防 NotebookLM 错误**：一位用户分享了他们防止 **NotebookLM** 重复错误的方法，即创建一个包含过去错误示例的 *“错误”* 源文档。
   - 另一位用户指出，此类指令可能对回复没有任何影响，因为 *NotebookLM 使用 RAG，它不会将完整的用户输入（来源）注入到 LLM 的上下文窗口中*。
- **NotebookLM 的局限性与潜在解决方案**：一位用户反馈在交互式测试期间 **Audio Overviews** 无法快进，并要求增加 **音频概览生成的长度**。
   - 另一位用户建议将 **Agentspace** 作为将 **NotebookLM** 与各种数据源和内部门户集成的潜在解决方案。
- **Agentspace 助力 NotebookLM 企业版**：一位用户询问如何将 NotebookLM 与包含视频和知识库文章的内部门户及 CRM 集成，electioneering 建议道：*“NotebookLM 没有可供使用的 API，也不支持连接到你提到的那些数据源类型”*。
   - electioneering 建议关注 **Agentspace** 这一解决方案，它 *包含并集成了 NotebookLM* [Agentspace](https://cloud.google.com/products/agentspace?hl=en)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://surf.e2b.dev/">Surf - E2B Computer Use Agent</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15731776">Audio Overviews - NotebookLM Help</a>：未找到描述</li><li><a href="https://notebooklm.google.com/notebook/f7607d7a-584c-4f35-96fc-f6815c573a6c?_gl=1*52xa3q*_ga*MjEzMjQ2ODA5Ni4xNzI5NTUyMzk5*_ga_W0LDH41ZCB*MTcyOTYxNzAwNC41LjEuMTcyOTYxOTMxMy4wLjAuMA..)">无标题</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/)">NotebookLM Help</a>：未找到描述</li><li><a href="https://github.com/GoogleCloudPlatform/agent-starter-pack">GitHub - GoogleCloudPlatform/agent-starter-pack：为 Google Cloud 构建的一系列生产级 Generative AI Agent 模板。它通过提供整体的、生产就绪的解决方案来加速开发，解决了构建和部署 GenAI Agent 中的常见挑战（部署与运营、评估、定制、可观测性）。</a>：为 Google Cloud 构建的一系列生产级 Generative AI Agent 模板。它通过提供整体的、生产就绪的解决方案来加速开发，解决了...</li><li><a href="https://cloud.google.com/products/agentspace?hl=en">Google Agentspace</a>：Google Agentspace 是企业级 AI Agent 的启动点，通过单个 prompt 帮助提高员工处理复杂任务的生产力。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1350544737571963011)** (6 条消息): 

> `Jake Cannell 招聘 GPU 开发人员，sm90 内核，GPU 性能计数器，nebius.ai，Datacrunch` 


- **Jake Cannell 扩招 GPU 开发人员**：Jake Cannell 正在[招聘 GPU 开发人员](https://www.linkedin.com/jobs/view/4118975911/)，以研究他在演讲中提到的想法。
- **学术界寻求高性价比 GPU 云**：一位研究人员正在寻找便宜的云服务商，要求能够访问 **GPU 性能计数器** 以运行 Nsight Compute，从而实现 **sm90 内核** 的想法。
- **nebius.ai 被推崇为 GPU 云选**：一位成员引用了 9 个月前 Reddit 上的一个帖子，推荐 [nebius.ai](https://nebius.ai) 作为可以访问 GPU 性能计数器的服务商。
- **Datacrunch 提议提供学生额度**：一位成员建议 **Datacrunch** 是一个不错的选择，可能会为学生提供额度。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1350494353897820311)** (13 条消息🔥): 

> `Embedded Python Pip Usage, Triton Windows PyPI Release, tl.multiple_of usage in Triton, Efficient Pointer Chasing in Triton, Triton and Sparse Computations` 


- **Triton-Windows 获得 PIP 升级**：**triton-windows** 已发布到 **PyPI**，因此你可以通过 `pip install -U triton-windows` 进行安装/升级，不再需要从 GitHub 下载 wheel 文件。
- **Triton 中 `tl.multiple_of` 的疑问**：一位用户对 `tl.multiple_of` 与 `tl.arange` 的配合使用提出疑问，怀疑是否只有第一个元素是 **BLOCK_SIZE_N** 的倍数，并想知道自己是否遗漏了什么。
- **指针追踪（Pointer Chasing）性能探讨**：一位用户询问如何在 Triton 中为类似于 CSR 稀疏矩阵的自定义数据结构实现高效的**指针追踪**，以避免在热点内循环中逐个加载偏移量。
   - 一位用户建议一次性加载整个偏移数组，然后在 `tl.where()` 上使用 `tl.sum()` 并配合循环索引来屏蔽除一个元素外的所有元素；另一位用户提到 *Triton 并不理想于稀疏引用和计算，作者在 [Triton Lang Docs](https://triton-lang.org/main/programming-guide/chapter-2/related-work.html) 中提到了这一点*。
- **在 Triton 中使用 pow 函数**：一位用户询问如何在 Triton 中使用 **pow**（幂）函数。
   - 另一位用户建议参考教程 `07-extern-function.py` 作为参考。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1350631146857431103)** (5 条消息): 

> `SASS compatibility with NVIDIA architectures, LD/ST unit sharing in SM microarchitecture, L1-dTLB cache, Cutlass 4.0 Python DSL, CUDA streams concurrency issues` 


- **跨 NVIDIA 架构的 SASS 指令**：一位成员分享了一个 [gist](https://gist.github.com/herrmann/f721da109e0c5c7c34c847ff2cf3da1e)，其中比较了不同架构下的 **NVIDIA SASS 指令**，这些指令是从 NVIDIA 的 HTML 文档中提取并（使用 Python）进行对比的。
   - 该 gist 有助于理解 NVIDIA GPU 系列中指令集的演变。
- **LD/ST 单元架构查询**：一位成员询问了 **SM** 中调度单元之间 **LD/ST 单元**的共享情况，参考了 **Ampere GA100 白皮书**，该白皮书将 32 个 LD/ST 单元分配给 4 个调度单元。
   - 他们还根据 NVIDIA 的 nsight compute 分析指南询问了 **LSU**、**MIO** 和 **LSUIN** 之间的关系，以及如果有 32 个线程，发出一条 `LDG` 指令是否需要 4 个周期。
- **L1-dTLB 缓存推测**：一位成员推测 **L1/TEX 缓存**是 **VIPT**（虚索引，实标签），并猜测地址转换发生在 **LSUIN** 和标签（tag）阶段之间。
   - 该话题没有进一步讨论。
- **Cutlass 4.0 全面转向 Python！**：一位成员分享道 [Cutlass 4.0](https://x.com/msharmavikram/status/1901465243861373327) 现在已完全 Python 化，使用了 **Python DSL**。
   - 这个新版本与之前的版本具有**性能对等性**，并在 **NVIDIAGTC** 上进行了展示。
- **CUDA 流显示并发异常**：一位成员在 **A800** 上遇到了 **CUDA 流**未按预期并发执行的奇怪问题，尽管资源充足。
   - 使用 **nsys** 进行的分析显示，在特定的共享内存配置下，较早的流被优先处理且未并发执行，重复次数设置为 **1,000,000**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/msharmavikram/status/1901465243861373327">来自 Vikram (@msharmavikram) 的推文</a>: Cutlass 4.0 Python DSL @__tensorcore__ 完全 Python 化！性能对等！欢迎参加 @NVIDIAGTC 的这两场会议</li><li><a href="https://gist.github.com/herrmann/f721da109e0c5c7c34c847ff2cf3da1e">NVIDIA SASS 指令历史</a>: NVIDIA SASS 指令历史。GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1350252666269208728)** (13 messages🔥): 

> `Torch Compile, Graph Breaks, Stride Issue, Std::variants in schemas` 


- ****Torch Compile** 在反向传播中遇到困难**: 一位成员报告称，虽然 **torch.compile** 在正向传播中表现良好，但在使用 **torch.autograd.Function** 编写自定义 kernel 时，反向传播的速度相当慢。
   - 他们发现使用 `torch.compile(compiled_backward_fn)` 包装反向函数可以解决这个问题。
- ****Graph Breaks** 导致编译问题**: 有人指出反向传播中的 **graph breaks** 会导致 **torch.compile** 出现问题。
   - 一位成员发现其 **Triton kernel** 中使用的 `.stride(0)` 导致了 graph breaks，他们通过改用 **constant values** 解决了此问题。
- ****Stride(0) 问题**在 Nightly 版本中已修复**: 一位成员提到他们在 **Triton kernel** 中遇到 `stride(0)` 导致 graph breaks 的问题。
   - 另一位成员提到 `stride(0)` 问题已在 **PyTorch nightly builds** 中得到修复。
- **Schemas 难以支持 **std::variants****: 一位成员询问关于在 schemas 中支持 `std::variants` 的问题，并链接到了[相关的 PyTorch 代码](https://github.com/pytorch/pytorch/blob/c7c3e7732443d7994303499bcb01781c9d59ab58/aten/src/ATen/core/op_registration/README.md)。
   - 一位核心开发者表示这相当困难，他们最终选择了 `std::optional`。



**提到的链接**: <a href="https://github.com/pytorch/pytorch/blob/c7c3e7732443d7994303499bcb01781c9d59ab58/aten/src/ATen/core/op_registration/README.md">pytorch/aten/src/ATen/core/op_registration/README.md at c7c3e7732443d7994303499bcb01781c9d59ab58 · pytorch/pytorch</a>: Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch

  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1350537135324790925)** (1 messages): 

> `Consumer GPU Performance, AGI, Neuromorphic hardware, vast.ai` 


- **Jake Cannell 关于消费级 GPU 性能的演讲**: Jake Cannell 将在 30 分钟内发表关于**消费级 GPU 性能 (Consumer GPU Performance)** 的演讲，涵盖他早期的图形学工作、GPU 如何变为通用型、他的扩展历程以及 [vast.ai](https://vast.ai) 背后的故事。
   - 这场演讲被认为对那些对 **AGI** 或 **Neuromorphic hardware** 感兴趣的人特别有参考价值，并预告将有一场非常精彩的讨论。
- **揭秘 Scaling Pilled 的起源**: Jake Cannell 将讨论他如何成为 **scaling pilled**（缩放定律信徒）以及建立 [vast.ai](https://vast.ai) 的动力。
   - 这场讨论可能会为 **AGI** 和 **Neuromorphic hardware** 研究的基础设施和资源需求提供见解。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1350467272417742929)** (6 messages): 

> `Transformers without Normalization, LayerNorm, tanh, FA3, exp` 


- **无归一化的 Transformers 可能会更快**: 一位成员指出，[无归一化的 Transformers](https://www.linkedin.com/posts/zhuang-liu-19306b1b1_new-cvpr-2025-paper-released-transformers-activity-7306390791361351682-Dfpb)（用 **tanh()** 代替 **LayerNorm**）应该能带来速度提升。
   - 这是因为 **LayerNorm** 需要对序列的均值和方差进行 reduction 操作，这可能很慢；而 **tanh()** 可以在寄存器中的每个元素上计算，并与随后的 matmul/linear 层融合。
- **tanh 并不廉价，但可以近似**: 一位成员表示 *`tanh` 本身并不廉价*，因为它需要一个 **exp** 和一个除法。
   - 在 Nvidia 硬件上存在 **`tanh.approx`（自 Turing/sm_75 起）**，据称其吞吐量为 **16/cycle/SM**。
- **__expf() 在小数值下更快**: 一位成员建议 **`__expf()`** 速度相当快，但仅适用于较小的值。
   - 其他人指出，由于 **exp** 成为瓶颈，**FA3** 产生了显著的开销。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1350500302796619898)** (4 messages): 

> `GPU Code Generation, ML Compiler, HPC Engineers, Superalignment Framework` 


- **GPU Mode 公司招聘 AI Engineer 负责 Code Gen**：一家公司正在招聘 **AI Engineer** 负责训练/微调用于 **GPU code generation** 的模型，提供优厚的薪酬和丰厚的股权激励，并获得 **Jeff Dean** 等人的支持；申请地址：[jobs.mako-dev.com](https://jobs.mako-dev.com/AI-Engineer-18b546aebc368000b243eab9ff7d262c)。
   - 他们正在构建一个将 AI 集成到编译流程中的**下一代 ML compiler**；请务必在可选申请信息中添加 "GPUMODE"。
- **Sesterce 为大规模 GB200 集群招聘 HPC Engineers**：Sesterce 正在寻找 **HPC Engineers** 来构建和管理其全新的 **Giga Colossius 集群 (18K GB200)** 和 **Colossius 集群 (8K GB200)**，其硬核工程团队分布在旧金山、法国和新加坡。
   - 团队成员包括 Awni，他被描述为*最聪明且最友善*的工作伙伴之一。
- **隐身模式初创公司招募 Superalignment Framework 架构师**：一家隐身模式初创公司正在招聘一名 **machine learning framework 软件架构师**，在 [ScalarLM.com](https://ScalarLM.com) 之上构建 **superalignment framework**。
   - 理想的候选人应准备好作为一个 **5 人团队**的一员，亲自编写该框架的所有代码，并为 bootstrapping 阶段的生活做好准备。



**提到的链接**：<a href="https://jobs.mako-dev.com/AI-Engineer-18b546aebc368000b243eab9ff7d262c">Your connected workspace for wiki, docs &amp; projects | Notion</a>：一个将日常工作应用融合在一起的新工具。它是为您和您的团队打造的全能工作空间。

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1350423184243167273)** (9 条消息🔥): 

> `GPU coalesced access, Nvidia GPU read operation, GPU programming, CUDA learning resources, Installing Triton` 


- **GPU 合并访问 (coalesced access) vs 置换读取 (permuted reads)**：当线程 (0,1,2,3) 在 GPU 中读取地址 (0,1,2,3) 时，它是合并的；然而，读取置换后的地址如 (2,0,3,1) 可能会导致 **4 次顺序读取**，而不是单次操作。
   - 如果线程读取类似 **4*i+[0,1,2,3] 且 i 是随机的** 地址，即每个线程在自己的 memory bank 中读取随机地址，目前尚不清楚这是否比存在 bank conflicts 的读取更快。
- **Nvidia GPU 读取操作为每个 warp 生成单个请求**：现代 **Nvidia GPU** 的读取操作会为每个 warp 向 **L1TEX/LSUIN** 生成一个请求，以 sector 级别 (**32 bytes**) 运行，并存储 cache lines (**4 sectors**)。
   - 这些请求在内部以 *wavefronts* 形式处理，更多细节可以在 [Nvidia 开发者论坛](https://forums.developer.nvidia.com/t/wahts-the-difference-between-wavefronts-and-sectors-req/165293/4) 和 [GTC Spring 21 会议](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32089/) 中找到。
- **探讨 Bank conflicts 和内存差异**：一场 [GTC Spring 22 会议](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/) 有助于理解 bank conflicts。
   - 演讲者强调了 **L1 data cache**（其返回带宽是每个 cycle 一个 cache line）与 **shared memory**（取决于 bank conflicts）之间的区别。
- **用于 GPU Kernel 测试的云资源**：对于没有本地 GPU 访问权限但想要测试 CUDA/Cutlass/Triton kernels 的 GPU 编程初学者，[Google Colab](https://colab.google/) 提供了免费的计算资源，包括 GPU 和 TPU。
   - 此外，[LeetGPU](https://leetgpu.com/) 也可以提供替代的测试环境。
- **在 Windows 上安装 Triton 的困扰**：一位用户在尝试使用 **pip** 在 Windows 上安装 Triton 时遇到了错误。
   - 在给定的上下文中没有提供具体的解决方案，但消息中分享的图片[显示了该错误](https://cdn.discordapp.com/attachments/1191300313928433664/1351213215127965696/Screenshot_2025-03-17_151415.png?ex=67d98ede&is=67d83d5e&hm=0c11c455bc52ffdca04c9d55229b42a2115ba6c469b145b7e0f27972f9ca97d6&)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://leetgpu.com/">LeetGPU</a>：未找到描述</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32089/">Requests, Wavefronts, Sectors Metrics: Understanding and Optimizing Memory-Bound Kernels with Nsight Compute | GTC Digital April 2021 | NVIDIA On-Demand</a>：了解如何充分利用 Nsight Compute 来识别和解决 kernel 代码中的内存访问低效问题</li><li><a href="https://colab.google/">colab.google</a>：未找到描述</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/">How to Understand and Optimize Shared Memory Accesses using Nsight Compute | GTC Digital Spring 2022 | NVIDIA On-Demand</a>：为了高效优化 kernel 对 shared memory 的使用，关键要素包括：(1) shared memory 硬件实现的心理模型
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1351264386089750720)** (1 条消息): 

> `CUDA kernel, pytorch extension` 


- **寻求将 CUDA Kernel 作为 PyTorch Extension 调用的源码**：一位成员询问第 2 讲的演讲者是否发布了将其 **mean_filter CUDA kernel** 作为 **PyTorch extension** 调用的源代码。
- **请求 CUDA Kernel 代码**：一位用户正在寻求第 2 讲的源代码，以便将 **mean_filter CUDA kernel** 实现为 **PyTorch extension**。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1350904669534294229)** (3 messages): 

> `AI Agents Hackathon, NVIDIA GTC 2025, SLURM based HPC cluster IDE/Editor` 


- **Vertex Ventures US 赞助 AI Agents Hackathon**：**Vertex Ventures US** 和 **CreatorsCorner** 正在 **NVIDIA GTC 2025** 期间举办 [AI Agents Hackathon](https://lu.ma/meofrw3d)，奖金超过 **$50k**。
   - 参赛者将构建能够进行复杂推理并与各种工具交互的 **multimodal AI agents**，并向评委进行 **3 分钟** 的展示。
- **Cursor/VSCode 是 HPC 集群开发的首选 IDE**：一位用户询问大家在 **SLURM 架构的 HPC 集群**上直接开发时使用什么 IDE/Editor，并对 **VSCode** 在 **/home/** 目录下的臃肿表示不满。
   - 另一位成员建议使用 **Cursor/VSCode**，提到他们工作集群上的大多数人都在使用它，并且安装目录是可以更改的。



**提及链接**：<a href="https://lu.ma/meofrw3d">AI Agents Hackathon - GTC 2025 Edition (1 DAY) · Luma</a>：AI Agents Hackathon - GTC 2025 Edition (1 DAY)。随着 NVIDIA GTC 2025 汇聚全球 AI 社区，Vertex Ventures US 和 CreatorsCorner 邀请您参与……

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1350531847481720832)** (11 messages🔥): 

> `Block Sparse Attention, GEMM, GTC Keynote Missed, GTC Hackathon results, GTC Meetup` 


- **参会者因 ESTA 错误错过 GTC Keynote**：一位参会者因未能提前填写好 **ESTA** 而错过 **GTC Keynote**，对此表示遗憾。
   - 他们提到 **ESTA** 状态卡在待处理（pending）状态一天，导致无法登机。
- **询问 GTC Hackathon 结果**：一位未能进入 **GTC 活动**现场的参会者询问 **GTC hackathon 结果**将在何处公布。
   - 该问题暂无回答，这表明答案尚不明确，或者结果可能会直接发送给参赛者。
- **讨论潜在的 Post-GTC 聚会**：大家讨论了在 **GTC 会议**结束后为错过最初聚会的人举办一场聚会。
   - 这表明许多人因无法参加 **GTC** 而感到遗憾，因此其他人同意组织一次单独的活动。
- **参会者请求 GTC 演讲幻灯片**：参会者请求将之前 **GTC 演讲**的幻灯片发布到某个地方。
   - 另一位参与者询问是否有人拍到了 **Vijay Thakkar** 关于 **Nvidia GTC workshops** 的最后一张幻灯片。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1350406018466709554)** (5 messages): 

> `MI300X inference optimization, AMD Instinct MI300X workload optimization, DeepSeek-R1 on MI300X, SGLang Optimization` 


- **寻求 MI300X 推理专家**：一位成员正在寻求专家帮助减少 **MI300X** 上的推理时间，并愿意在咨询后分享信息。
   - 他们正在寻找能够针对 **32B reasoner model** 进行几小时专门咨询的人员。
- **AMD 发布推理优化指南**：一位成员分享了 [AMD Instinct™ MI300X 工作负载优化](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) 文档，详细介绍了 **MI300X 加速器**的优化策略，重点关注 GPU kernel 编程、HPC 以及使用 PyTorch 的深度学习。
   - 该文档强调了可自动调优的配置以及 **Triton kernel 优化**等高级技术。
- **DeepSeek-R1 在 MI300X 上提速**：分享了一篇关于[在 AMD Instinct™ MI300X GPU 上释放 DeepSeek-R1 推理性能](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html)的博客文章，重点介绍了与 **H200** 的性能对比。
   - 据报道，使用 **SGLang** 的优化在短短两周内就将推理速度提升了高达 **4 倍**，确保了高效的扩展和更低的延迟。


<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>
<a href="https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html">在 AMD Instinct™ MI300X GPU 上释放 DeepSeek-R1 推理性能 &#8212; ROCm 博客</a>：未找到描述</li><li><a href="https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html">AMD Instinct MI300X 工作负载优化 — ROCm 文档</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/)** (1 messages): 

leiwang1999_53585: worked on my h100, maybe you should install nightly wheel🤣
  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1350210217815834765)** (5 messages): 

> `GTC CUDA, Wen-mei Hwu GTC, Pruna AI Efficiency Framework, Ruff and UV for project management` 


- ****GTC** 的 **CUDA** 内容中心**：NVIDIA 正在重点推介 [GTC 的 CUDA 开发者会议](https://www.nvidia.com/gtc/sessions/cuda-developer/)，这些会议专注于创建高性能、GPU 加速应用程序的工具和培训。
   - 与会者可以探索针对通用 AI、技术细节和业务策略量身定制的会议，席位先到先得。
- ****Wen-mei Hwu**：计算领域的传奇人物在 **GTC** 签名**：教授 **Wen-mei Hwu**（作者兼 NVIDIA 科学家）将出席 #GTC25，参加独家见面会并为其[著作](https://www.nvidia.com/gtc/)签名。
   - **GPUMODE** 活动定于周日晚上 6 点和周三下午 2 点在 CWE75384 举行，你可以在[此处](https://nvda.ws/4iNYQnh)注册 CWE 活动。
- ****Pruna**：新型 AI 效率框架发布**：AI 效率框架 **Pruna** 已开源，技术细节可在 [GitHub repo](https://github.com/PrunaAI/pruna/tree/main) 查看。
   - 鼓励用户 *star 该仓库、传播消息，并通过 pip install pruna 安装 Pruna* 以提供反馈。
- ****Ruff** 和 **UV** 简化依赖项**：一位用户建议在 **Pruna** 项目中切换到 [Ruff](https://astral.sh/ruff) + [uv](https://docs.astral.sh/uv/)，以简化依赖项并改进项目管理。
   - 该用户认为这一改变将大大简化依赖关系。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/PrunaAI/pruna/tree/main">GitHub - PrunaAI/pruna: Pruna is a model optimization framework built for developers, enabling you to deliver faster, more efficient models with minimal overhead.</a>: Pruna 是一个为开发者构建的模型优化框架，使你能够以最小的开销交付更快、更高效的模型。 - PrunaAI/pruna</li><li><a href="https://www.nvidia.com/gtc/sessions/cuda-developer/">NVIDIA GTC AI Conference 2025</a>: 2025 年 3 月 17–21 日。圣何塞。立即注册。</li><li><a href="https://nvda.ws/4iNYQnh">NVIDIA #GTC2025 Conference Session Catalog</a>: 3 月 17-21 日在圣何塞亲身体验 GTC 2025（线下及线上）。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1350237974440185937)** (5 messages): 

> `Distributed Training, Scaling Laws for DiLoCo, GPU kernel modifications` 


- **DiLoCo 的扩展性优于 DP**：[X 上的帖子](https://x.com/matharycharles/status/1900593694216253827?s=46)强调了使 **distributed training** 在更大模型上运行的关键步骤，特别是 **DiLoCo 的 Scaling Laws**。
   - 作者开玩笑说 *DiLoCo 的扩展性比 DP 更好，这对我来说很有趣；这纯粹是感觉（vibes）哈哈*。
- **GPU Kernel 优化的细微差别**：一位成员产生了一个可怕的想法，即对 **GPU kernel** 进行轻微修改，虽然在数值上不等效，但在分布式语境下的实际运行时间（wall-clock time）更高效。
   - 他们认为这类问题给 **automatic kernel optimization strategies** 带来了巨大的挑战。



**提及的链接**：<a href="https://x.com/matharycharles/status/1900593694216253827?s=46">来自 Zachary Charles (@MatharyCharles) 的推文</a>：我们刚刚发布了使分布式训练在越来越大的模型上运行的关键步骤：DiLoCo 的 Scaling Laws。TL;DR：我们可以跨数据中心进行 LLM 训练，其扩展性非常好...

  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1350744003615916032)** (10 条消息🔥): 

> `Reasoning Gym, nano-R1 Project, Temporal Clue, Group Relative Policy Optimization (GRPO)` 


- **Reasoning Gym 数据集达到 101 个！**：[Reasoning Gym](https://github.com/open-thought/reasoning-gym) 项目现在拥有 **101 个数据集**，以此庆祝来自 Rich Jones 和 @jeankaddour 等开发者的贡献。
   - 一位用户分享了宣布这一里程碑的 [X 帖子](https://x.com/neurosp1ke/status/1901244866920636559)。
- **nano-R1 项目关注 Reasoning Gym**：**nano-R1** 项目正在寻找数据来评估运行情况，考虑到现有的 Benchmark 分数，建议考虑使用 [reasoning-gym](https://github.com/open-thought/reasoning-gym)。
   - 该建议参考了关于寻找推理 Benchmark 的 [GitHub 讨论](https://github.com/nano-R1/resources/discussions/4)。
- **Temporal Clue 谜题准备加入 Gym**：一位用户分享了 [temporal-clue](https://github.com/bradhilton/temporal-clue) 的链接，这是一套受 *Clue* 启发的谜题，用于测试 LLM 的演绎推理能力，并建议将其作为 Reasoning Gym 的素材。
   - 这些谜题可能对测试演绎推理（deductive reasoning）很有用。
- **GRPO 在 Temporal Clue 上击败其他模型**：[OpenPipe.ai](https://openpipe.ai/blog/using-grpo-to-beat-o1-o3-mini-and-r1-on-temporal-clue) 使用 **Group Relative Policy Optimization (GRPO)** 在 Temporal Clue 上达到了 SOTA 水平，超越了 **R1**、**o1**、**o3-mini**，并接近 Sonnet 3.7 的性能，同时成本降低了 *100 倍*。
   - 他们分享了基于 [torchtune](https://github.com/pytorch/torchtune) 构建的 [训练配方（training recipe）](https://github.com/openpipe/deductive-reasoning)，用于实现这些结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/neurosp1ke/status/1901244866920636559">来自 Andreas Köpf (@neurosp1ke) 的推文</a>：我们现在在 reasoning-gym 中共有 101 个数据集！🧠💪 非常感谢 💙 所有让这一切成为可能的开发者，特别是核心团队 Rich Jones, @zafstojano, Joe Sharratt, A. Adefioy, Ollie Stanley &...</li><li><a href="https://openpipe.ai/blog/using-grpo-to-beat-o1-o3-mini-and-r1-on-temporal-clue">使用 GRPO 在 "Temporal Clue" 上击败 o1, o3-mini 和 R1 - OpenPipe</a>：将昂贵的 LLM Prompt 转换为快速、廉价的微调模型</li><li><a href="https://github.com/bradhilton/temporal-clue">GitHub - bradhilton/temporal-clue: 受 Clue 启发的用于测试 LLM 演绎能力的谜题</a>：受 Clue 启发的用于测试 LLM 演绎能力的谜题 - bradhilton/temporal-clue</li><li><a href="https://github.com/Tufalabs/MITIntegrationBee">GitHub - Tufalabs/MITIntegrationBee</a>：通过在 GitHub 上创建账号来为 Tufalabs/MITIntegrationBee 的开发做出贡献。</li><li><a href="https://github.com/nano-R1/resources/discussions/4">nano-R1: Benchmark + 挑战格式 · nano-R1/resources · Discussion #4</a>：我认为最初的目标应该是找到一套通用的、合理的推理 Benchmark 进行评判，然后还有一套“默认”的训练数据集 + 脚本，以便人们可以...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[active-leaderboards](https://discord.com/channels/1189498204333543425/1342999437902872727/1350871867002454046)** (1 条消息): 

> `Xavier Init, 用户 ID 问题` 


- **用户名被神秘的用户 ID 替换**：一些用户看到的是 **User_<18 位 ID>** 而不是实际用户名，这可能是由于与 [Xavier Init](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_) 相关的 Bug 导致的。
- **持续的用户名故障**：一个故障导致某些用户名显示为通用的 **User_ID** 字符串，而不是实际名称。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1350196644834181130)** (15 条消息🔥): 

> `Popcorn 中的 pip install, 寻找 GTC 2025 门票, 免费 B200 访问, AMD 支持即将到来` 


- **Popcorn 允许用户进行 Pip Install**：用户现在可以在 **Popcorn** 的脚本中执行 `pip install`，不过耗时较长的安装可能会超时。
- **寻找 GTC 2025 门票**：一位硅谷居民正在寻求已售罄的 **GTC 2025** 活动门票。
   - 另一位成员调侃道：“先生，这里是温蒂汉堡（Wendy's）”。
- **Grayscale 上的免费 B200 福利！**：在 Grayscale 的 grayscale_py_b200-dev 排行榜上有一台 **B200** 可用，由于只有一台设备，排队时间可能会较长。
   - 鼓励成员们“尽情把玩 B200，随心所欲地拆解它”。
- **AMD 支持即将到来**：根据一张 [截图](https://cdn.discordapp.com/attachments/1343002580531417211/1351201006297546905/image.png?ex=67d98380&is=67d83200&hm=cd8506d90a42207f7ad4fd3c10545d80661f302981288210f521412ae3edf107)，AMD 支持似乎即将推出，“我们终于在憋大招了（we're cooking something finally）”。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1350586065395908788)** (29 messages🔥): 

> `Leaderboard Submissions, Benchmark Submissions, Test Submissions, Modal Runners` 


- **Grayscale 测试在 T4 和 H100 上取得成功**：ID 为 **2136** 和 **2143** 的 `grayscale` 排行榜测试提交，在使用 Modal runners 的 **T4** 和 **H100** GPU 上成功运行。
- **Vectoradd 在 H100 上表现出色**：ID 为 **2151** 的 `vectoradd` 排行榜提交，在使用 Modal runners 的 **H100** GPU 上成功运行！
   - Modal Runners 助力在高性能 **H100** GPU 上成功完成了向量加法基准测试。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1350254806027206819)** (1 messages): 

> `Leaderboard cleanup, Robust Evaluation` 


- **排行榜清理工作开始**：社区正在从排行榜中移除恶搞/黑客条目，并要求希望删除条目的用户提交其 **Discord 用户名**、**文件名**和**排名**。
   - 同时，正在进行更改以确保评估过程对这类条目具有更强的鲁棒性。
- **评估流程得到加强**：在清理排行榜的同时，正努力提高评估过程针对恶搞/黑客条目的鲁棒性。
   - 目标是防止未来出现类似问题。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1351215637367619656)** (1 messages): 

> `NVIDIA thermal ranges, Arithmetic and Memory Bandwidth Degradation` 


- **成员寻求 NVIDIA 温度范围和降级信息**：一位成员正在寻找不同 NVIDIA 显卡的温度范围，特别是关于算术和内存带宽随温度降级的信息。
   - 他们引用了 [NVIDIA H100 产品简介](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf) 作为良好的信息来源，希望能找到更多显卡的类似细节。
- **关于硬件热限制的讨论**：讨论围绕寻找 NVIDIA 显卡的详细热规格展开，特别是关于温度如何影响性能。
   - 最初的请求者分享了 NVIDIA H100 产品简介，作为他们寻求更广泛显卡详细信息的一个范例。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1350222797305286698)** (10 messages🔥): 

> `SMILES string encoding, Stereoisomer Generation, Free GPU Platforms, Managed Inference APIs, EleutherAI welcomes Catherine Arnett` 


- **将 SMILES 字符串编码为立体异构体**：一位成员询问了能够将 **SMILES string** 编码为各种**立体异构体**或编码 **ChemDraw 输入**的模型或架构。
   - 该成员正在寻找一个能够识别其任务中化学描述符的模型。
- **寻找免费 GPU 平台**：一位成员正在寻找 Notebook 之外的**免费 GPU 平台**，需要支持 **C++** 并可通过 SSH 进行本地使用的平台。
   - Notebook 仅提供 Python 接口，这不足以满足他们的需求。
- **托管推理 API 服务探索**：一位成员正在寻求**托管推理 API 服务**的推荐，供小型初创公司用于托管私有模型以进行 **LLMs** 的训练/微调。
   - 另一位成员推荐了 [Featherless.ai](https://featherless.ai)，它也支持来自 HF 的现有 LLM；它*不需要管理单个硬件单元*。
- **EleutherAI 欢迎新的 NLP 研究员**：EleutherAI 欢迎 **Catherine Arnett**，她是一位专注于**计算社会科学**和**跨语言 NLP** 的 NLP 研究员。
   - Catherine 的研究重点是确保模型在不同语言中表现*同样出色*，解决数据等效性、性能测量和模型构建问题；请参阅她最近关于 [Goldfish](https://arxiv.org/abs/2408.10441)、[Toxicity of the Commons](https://arxiv.org/abs/2410.22587)、[复杂语言上的 LM 性能](https://arxiv.org/abs/2411.14198) 以及 [多语言语言建模](https://arxiv.org/abs/2311.09205) 的工作。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1350183448395907233)** (46 条消息🔥): 

> `Block Diffusion, Globally Shared Experts, Mixture-of-Experts Universal Transformers, Tan et al.'s SUT paper, Visual Geometry Group (VGGT)` 


- ****Block Diffusion** 模型发布！**: 一篇新论文介绍了 **Block Diffusion**，这是一种在自回归（autoregressive）和扩散（diffusion）语言模型之间进行插值的方法，结合了双方的优势：高质量、任意长度、KV caching 和可并行性。详情见 [论文](https://arxiv.org/abs/2503.09573) 和 [代码](https://github.com/kuleshov-group/bd3lms)。
- **探索深度学习中的 Globally Shared Experts**: 讨论涉及了对全局共享专家（globally shared experts）的研究，即在所有层中使用单一的专家池，并指向了一篇关于扩散模型的 [相关论文](https://arxiv.org/abs/2404.14507)。
- ****MoEUT**: 提及 Mixture-of-Experts Universal Transformers 论文**: 一位成员提到 **MoEUT** ([Mixture-of-Experts Universal Transformers](https://arxiv.org/abs/2503.08827)) 论文与全局共享专家的讨论相关，尽管他们尚未完全读完。
   - 另一位成员建议查看 Tan 等人的 **SUT** 论文以获取相关见解。
- ****VGGT** 生成 3D 场景！**: 一位成员分享了 [VGGT](https://vgg-t.github.io/)，这是一个前馈神经网络，可以从单个或多个视角推断 3D 属性并生成 GLB 文件，这些文件可以直接集成到元宇宙（metaverses）中。
   - 该成员在旧的立体图像和各种场景上测试了 **VGGT**，发现它受益于近角度帧；然而，它在缺乏清晰锚定角度的场景中可能会遇到困难。该成员表示：*我非常喜欢它能导出 GLB 文件，这意味着我可以原封不动地直接将它们放入我的元宇宙中。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://vgg-t.github.io/">VGGT: Visual Geometry Grounded Transformer.</a>: 我们提出了 Visual Geometry Grounded Transformer (VGGT)，这是一个前馈神经网络，可以在几秒钟内从单个或多个（多达数百个）图像视图直接预测所有关键的 3D 场景属性...</li><li><a href="https://arxiv.org/abs/2503.08827">Neural Network/de Sitter Space Correspondence</a>: 神经网络/德西特空间对应关系：机器学习显著的实践成功引发了广泛的理论研究，但根本性的突破仍然难以实现。在这里，我们通过梯度研究神经网络训练...</li><li><a href="https://arxiv.org/abs/2503.09799">Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo</a>: 通信高效的语言模型训练能够可靠且稳健地扩展：DiLoCo 的 Scaling Laws。随着我们扩展到更大规模的机器学习模型，数据并行方法中固有的频繁同步需求导致了显著的减速，对进一步扩展构成了严峻挑战...</li><li><a href="https://arxiv.org/abs/2404.14507">Align Your Steps: Optimizing Sampling Schedules in Diffusion Models</a>: 对齐你的步长：优化扩散模型中的采样调度。扩散模型 (DMs) 已成为视觉领域及其他领域最先进的生成建模方法。DMs 的一个关键缺点是采样速度慢，依赖于...</li><li><a href="https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view">evolving_llms_through_text-based_self-play.pdf</a>: 未找到描述</li><li><a href="https://m-arriola.com/bd3lms/">SOCIAL MEDIA TITLE TAG</a>: 社交媒体标题标签
</li>
</ul>

</div>

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1350604163402170378)** (22 条消息🔥): 

> `Fewshot Split Fallback, Gen Kwargs 转 JSON, 新旧 LLM Leaderboard 对比` 


- **Fewshot Split Fallback 方案公布**：当未指定 fewshot split 时，系统将回退至 **train > val > test** 顺序，如果存在 training split 则优先使用。
   - 此顺序决定了在未定义特定 split 的情况下，使用哪个 split 进行评估。
- **Gen Kwargs 采用 JSON 格式**：`--gen_kwargs` 参数正从逗号分隔的字符串过渡到 **JSON**，从而支持更复杂的配置，例如 `'{"temperature":0, "stop":["abc"]}'`。
   - 讨论中探讨了同时支持两种格式以方便使用的可能性，特别是对于标量值。
- **新旧 LLM Leaderboard：差异浮现**：发现旧版 LLM Leaderboard 的 group config 与实际使用的设置之间存在差异，特别是关于 **arc-challenge 任务**。
   - `openllm.yaml` 配置指定 `validation` 作为 fewshot split，但由于旧 fork 的 Python 类中缺少 fewshot split，原始 Leaderboard 实际使用了 `train` split。已创建一个 [PR 来修复此问题](https://github.com/EleutherAI/lm-evaluation-harness/pull/2802) 以解决该差异。



**提及的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/benchmarks/openllm.yaml">lm-evaluation-harness/lm_eval/tasks/benchmarks/openllm.yaml at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1350430340627562599)** (30 条消息🔥): 

> `SDXL 基准测试, tensor cat 速度, 并行 BLAKE3, WebGPU 集成, Bitonic Sort 索引` 


- **Tinygrad SDXL 基准测试落后于 Torch**：在 **7900 XTX** 上使用 **tinygrad** 测试 **SDXL**，在 **AMD 后端** 开启 **BEAM=2** 时速度为 **1.4 it/s**，而使用 **torch.compile** 配合 **FlashAttention** 和 **TunableOp ROCm** 可达到 **5.7 it/s**。
   - George Hotz 建议通过对比 kernels 来寻找优化机会，目标是在年底前超越 **torch** 的性能。
- **尽管努力优化，Tensor Cat 依然缓慢**：一名成员正致力于提升 tensor cat 的速度，并在 **X** 上分享了白板思路（[链接](https://x.com/t0kenl1mit/status/1900952693587538018)），但指出尽管修改了 devectorizer，速度仍然较慢。
   - 该成员怀疑生成的 **IR** 和加载 **numpy arrays** 存在问题，正考虑通过 **ELF** 和 **LLVM** 使用自定义 **C/C++** 来克服限制。
- **BLAKE3 悬赏状态已明确**：明确了“高性能并行 BLAKE3”悬赏（bounty）的状态，并附带截图（[链接](https://cdn.discordapp.com/attachments/1068976834928193609/1350640745505231061/Screenshot_2025-03-15_182214.png?ex=67d973f7&is=67d82277&hm=19c5ffbf47ae93d8dda6ba9c5fc1b65cc3b1df108a2f4fd5860ba66e301bef7c&)）展示了悬赏进度。
   - 该成员更新了表格，并指出渐近性能（asymptotic performance）是该悬赏的关键要求。
- **WebGPU 集成获得提升**：一名成员询问是否可以发布一个基于 **resnet18** 的电子/光子分类器的 **Tinygrad** 实现作为示例，并被引导至一个[改进 WebGPU 集成的 PR](https://github.com/tinygrad/tinygrad/pull/9424)。
   - 建议创建一个托管在 **GitHub Pages** 上的 **WebGPU** demo，并将权重放在 **Hugging Face** 上以供免费访问和测试。
- **Bitonic Sort 索引问题解决**：在研究 bitonic sort 索引时，一名成员搞清楚了 **maxpool 索引**，并指出 **topk** 实现通常是基于排序的。
   - 该成员表示代码是正确的，且 JIT 后的速度接近 **pytorch** sort（有时更快），但这涉及 *大量 kernels*，因为有连续性（contiguity）要求。



**提及的链接**：<a href="https://x.com/t0kenl1mit/status/1900952693587538018">vincent (@t0kenl1mit) 的推文</a>：尝试在 @__tinygrad__ tensor cat 中使用 compare，但速度仍然很慢。附上我的白板思路。我想我可能不得不处理 ELF 并链接一些自定义 C，但也可能是其他原因...

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1350235346704797778)** (5 条消息): 

> `Print Debugging Tinygrad, Lazy Computation 与 Gradients, 用于调试的 Reproducer Code, 多行代码块` 


- **Tinygrad Lazy Mode 中的 Print Debugging 困境**：一位成员在 Tinygrad 中对中间 Tensor 值进行 Print Debugging 时遇到了 Gradients 的断言错误，尽管使用了 `.detach()`。
   - 由于 [Lazy Computation](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html) 非幂等性的问题，他们正在寻求一种比将值传出更好的方法。
- **编写用于快速调试的 Reproducer Code**：一位成员建议编写一个不超过 **10 行** 的 Reproducer Code，以便快速迭代和调试。
   - 他们建议使用像 VSCode 这样带有断点和调试控制台的集成调试器进行实验和重启。
- **GitHub 链接**：一位成员分享了一个 [GitHub 仓库链接](https://github.com/kayo09/gsoc_2025/tree/main/ML4SCI/task1)。
- **多行代码块**：一位成员就如何使用三反引号创建多行代码块提供了建议。



**提及的链接**：<a href="https://github.com/kayo09/gsoc_2025/tree/main/ML4SCI/task1">gsoc_2025/ML4SCI/task1 at main · kayo09/gsoc_2025</a>: GSOC 2025! Happy Coding! ☀️。通过在 GitHub 上创建一个账户来为 kayo09/gsoc_2025 的开发做出贡献。

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1350638802284187649)** (2 条消息): 

> `Agentic Reasoning System, Corrective RAG, LlamaExtract 公测版` 


- **Agent 使用 Corrective RAG 进行推理**：一位成员分享了一个分步教程，介绍如何从零开始构建用于搜索和检索的 **Agentic Reasoning System**（特别是 **Corrective RAG**），并使用 [@llama_index workflows](https://t.co/iDga01FouC) 进行编排。
   - 该 [教程](https://twitter.com/llama_index/status/1901079091345818022) 允许用户编排复杂的、可定制的事件驱动型 Agent。
- **LlamaExtract 进入公测阶段**：[LlamaExtract](https://twitter.com/llama_index/status/1901692607144861744) 现已进入公测阶段，它解决了从长而复杂的文档中提取结构化数据的常见问题，并提供 **Web UI** 和 **API**。
   - 它允许用户定义 Schema 并自动提取结构化数据；更多详情请点击 [此处](https://t.co/gT3R2l7CWM)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1350518353042608229)** (31 条消息🔥): 

> `AI Agents Hackathon, Vertex Ventures US, CreatorsCorner, gguf fine tuning, LlamaIndex vs Pydantic AI` 


- **AI Agents 黑客松召集令！**：**Vertex Ventures US** 和 **CreatorsCorner** 邀请全球 AI 社区参与在 **NVIDIA GTC 2025** 举办的独家 **AI 黑客松**，将大胆的创意转化为行动。
   - 本次黑客松挑战参赛者构建一个非凡的**多模态 AI Agent**，要求具备复杂的推理、战略决策能力，并能与各种工具交互，优胜者有机会赢取 **$50k+ 的奖金**！
- **Pydantic vs LlamaIndex 框架对决**：新用户好奇用于构建 Agent 的 **Pydantic AI** 和 **LlamaIndex** 框架有何区别，尤其是初学者该选择哪一个。
   - 一位 LlamaIndex 团队成员表示，最适合你开发思维模型（mental model）的可能就是最好的选择——但同时也提到 LlamaIndex 的 workflows 非常好用。
- **Data Query Agent 陷入无限可视化循环**：一位用户报告称其 **data query agent** 在使用**可视化工具**后陷入死循环，不断重复调用同一个工具。
   - 另一位成员询问该用户使用的是开源还是闭源 LLM，并推测道：*“也许是 LLM 无法理解任务是否已经完成”*。
- **LlamaExtract 已上线云端**：成员们在看到 GitHub 仓库中加入 Discord 的指引后，询问如何获取 **LlamaExtract** 的访问权限。
   - LlamaIndex 团队回应称，它可以在 [cloud.llamaindex.ai](https://cloud.llamaindex.ai) 上使用，并且 *LlamaExtract 在云端运行（客户端部分是开源的）*。
- **使用顺序工作流编排 Agent**：一位用户询问应使用 **workflows** 还是 **agents** 抽象来以线性、顺序的方式构建一组 Agent，且不希望将 Agent 绑定到特定的 LLM 提供商（如 Claude）。
   - 一位 LlamaIndex 团队成员提供了 LLM 类中[手动工具调用（manual tool-calling）](https://docs.llamaindex.ai/en/stable/examples/llm/anthropic/#toolfunction-calling)能力的指南。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://lu.ma/meofrw3d">AI Agents Hackathon - GTC 2025 Edition (1 DAY) · Luma</a>：AI Agents 黑客松 - GTC 2025 版（1 天）。随着 NVIDIA GTC 2025 汇聚全球 AI 社区，Vertex Ventures US 和 CreatorsCorner 邀请您将……</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/anthropic/#toolfunction-calling">Anthropic - LlamaIndex</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1351279240930132049)** (1 条消息): 

> `Vision-Language Models (VLMs), Multimodal Learning, GitHub Research Hub` 


- **Vision-Language Models 研究中心开放**：一位成员为从事 **Vision-Language Models (VLMs)** 研究的多模态研究者创建了一个[社区驱动的中心](https://github.com/thubZ09/vision-language-model-hub.git)。
   - 作者鼓励大家贡献和提出建议，并计划每周更新该中心。
- **征集 VLM Hub 贡献**：GitHub 上 **Vision-Language Model Hub** 的创建者正积极寻求社区贡献。
   - 他们对建议和反馈持开放态度，旨在每周更新该中心，使其成为多模态研究者的宝贵资源。



**提及的链接**：<a href="https://github.com/thubZ09/vision-language-model-hub.git">GitHub - thubZ09/vision-language-model-hub: 探索 VLMs 和多模态学习的研究者中心 :)</a>：探索 VLMs 和多模态学习的研究者中心 :) - GitHub - thubZ09/vision-language-model-hub: Hub for researchers exploring VLMs and Multimodal Learning :)

  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1350192294615318659)** (29 messages🔥): 

> `Gemma 3 Integration in GPT4All, LocalDocs Crashing Fix, Gemma 3 Language Comprehension, Model license agreements` 


- **Gemma 的语言能力在多种语言中超越竞争对手**：成员们发现，针对“把一个密封的罐子放在零下的室外会发生什么”这一谜题，**Gemma**、**DeepSeek R1** 和 **Qwen2.5** 模型都能以多种语言提供正确答案。
   - 其他模型预测罐子会发生灾难性损坏，但 **Gemma** 提供了更有用、更细致的建议。
- **Gemma 3 面临集成问题**：用户热切期待 **GPT4All** 支持 **Gemma 3**，但由于 Hugging Face 上的许可协议问题，目前正等待 **Llama.cpp** 的更新，详情见 [此 GitHub issue](https://github.com/nomic-ai/gpt4all/issues/3540)。
   - 一些人猜测 Google 是否会监管绕过其许可协议的重新分发行为。
- **LocalDocs 需要崩溃修复方案**：一位新用户在程序崩溃并重新安装后经历了 **LocalDoc** 集合丢失，正在寻求如何防止在下次预期崩溃后发生数据丢失的建议。
   - 资深用户建议定期保存 *localdocs* 文件并在崩溃后恢复，并指出*有时仅一个损坏的 PDF 就能导致系统崩溃*。
- **进阶版 O3-mini 解释思考过程**：一位用户分享了一个让 **O3-mini** 解释其思考过程的 Prompt，认为这可以改进模型蒸馏（distillation）。该方法可用于任何模型。
   - 该 Prompt 使用了 **thinking**（思考）和 **reflection**（反思）部分，包含逐步推理和错误检查。



**提到的链接**：<a href="https://github.com/nomic-ai/gpt4all/issues/3540">Gemma 3 support · Issue #3540 · nomic-ai/gpt4all</a>：系统信息：我安装了 GPT4All，打开它，从 Hugging Face 下载了 Gemma3 Instruct（尝试了两个模型 https://huggingface.co/Mungert/gemma-3-12b-it-gguf https://huggingface.co/ggml-org/gemm...

  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1350268808400736357)** (20 messages🔥): 

> `Fine-tuning for Command A, Azure Cohere Rerank v3 Terraform, Support Channel for New Models, Channel for Private Deployments of CMD A` 


- **目前尚不支持 Command A 的微调**：一位成员询问了在 Cohere 平台上开启 **Command A 微调**（fine-tuning）的预计时间，Cohere 团队成员回应称*目前尚无计划*，但会随时向社区通报进展。
- **Azure Cohere Rerank v3 Terraform 问题**：一位成员在尝试使用 Terraform 创建 **Azure Cohere Rerank v3** 时遇到错误，并分享了代码片段和错误信息。
   - 一位 Cohere 团队成员将该问题移至 <#1324436975436038184> 频道以进行进一步讨论。
- **私有部署频道准备中？**：一位成员建议为 **CMD A** 和其他模型的**私有部署**讨论创建一个专门频道，特别是针对引导客户进行本地部署的工作。
   - 另一位成员表示赞同，并请求管理员 <@700025263379054675> 进行设置。
- **支持频道问题量巨大**：一位 Cohere 团队成员提醒社区，将所有与新模型相关的支持问题发送至 <#1324436975436038184> 频道或通过电子邮件发送至 support@cohere.com。
- **CMD-A 深受喜爱**：一位成员表示 *“非常喜欢 Command A，它是一个很棒的模型”*。


  

---


### **Cohere ▷ #[【📣】announcements](https://discord.com/channels/954421988141711382/996880279224451154/1351243287750508647)** (1 messages): 

> `Command A, Developer Office Hours, Enterprise-friendly features, Hardware vs performance` 


- **Cohere 宣布三月开发者办公时间**：为了庆祝最新模型 **Command A** 的发布，Cohere 将于 **ET 时间 3 月 xx 日下午 1 点**在 Stage 频道举办**开发者办公时间（Developer Office Hours）**。
   - 会议将涵盖 **Command A** 的新特性、企业友好功能、硬件与性能对比以及现场问答；[更多详情请点击此处](https://discord.gg/QVyVXjST?event=1351206903426056306)。
- **Command A 模型发布**：Cohere 即将发布 **Command A** 模型，并举办办公时间活动以示庆祝。
   - 办公时间将涵盖多个主题，包括：新功能介绍、企业级特性以及现场问答。


  

---

### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1350220719589425163)** (3 messages): 

> `Cohere Command A, Vercel SDK integration, Object generation support, Cohere API versioning` 


- **Vercel SDK 遗漏了 Cohere 的 Object Generation**: 有用户报告称 [Vercel SDK](https://sdk.vercel.ai/providers/ai-sdk-providers/cohere) 错误地认为 Cohere 的 Command A 模型**不支持 object generation**。
   - 用户打算向 Vercel 反馈此问题，并建议 Cohere 团队也应予以关注。
- **SDK 实现与 Cohere API 版本冲突**: 一位尝试在 JavaScript 中使用 **OpenAI SDK** 调用 Cohere 的用户遇到了与 [Cohere API versioning](https://docs.cohere.com/versioning-reference) 相关的警告。
   - 警告建议设置 API 版本，因为当前版本已弃用，尽管用户已经设置了 **apiKey** 和 **baseUrl**。
- **Cohere API Base URL 混淆已澄清**: 一位用户分享了正确的 `base_url` 应该是 [`https://api.cohere.com/compatibility/v1/chat/completions`](https://api.cohere.com/compatibility/v1/chat/completions)。
   - 在将 Cohere 与其他平台或 SDK 集成时，此 URL 可能会解决与 API 兼容性和版本控制相关的问题。



**提及的链接**: <a href="https://sdk.vercel.ai/providers/ai-sdk-providers/cohere">Cohere</a>: 了解如何为 AI SDK 使用 Cohere 提供商。

  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/)** (1 messages): 

.paolo16: Hello
  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1350992863676072106)** (3 messages): 

> `Introductions, Freelance programmers, Community Assistance` 


- **自由职业程序员自我介绍**: 一位 **30 岁的日本男性自由职业程序员**介绍了自己，并表示愿意通过自己的编程技能帮助他人。
   - 他强调 *互相帮助是我们生存的支柱*。
- **欢迎社区新成员**: Discord 服务器置顶了一条消息，感谢新成员加入 Cohere 社区。
   - 消息提示他们通过提供**公司/行业/大学**、当前项目、喜欢的技术/工具以及在社区的目标来进行自我介绍。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1351212388288041083)** (13 messages🔥): 

> `dspy/MCP Integration, DSPy Assertions / Suggestions removal, DSPy 2.6 Output Refinement, QdrantRM removal in 2.6` 


- ****MCP 集成愿景****: 一位成员询问关于集成 **dspy/MCP** 的事宜，另一位成员指出需要 MCP host、client 和 server，并思考这是否会让事情变得过于复杂，同时链接到了一个 [相关的 GitHub 示例](https://github.com/philschmid/mcp-openai-gemini-llama-example/blob/master/sqlite_llama_mcp_agent.py)。
- ****DSPy 放弃 Assertions/Suggestions****: 一位用户注意到 DSPy 中关于 **Assertions / Suggestions** 的 [文档消失了](https://dspy.ai/learn/programming/7-assertions/?h=dspy.suggest#dspyassert-and-dspysuggest-api)，并询问是否继续支持。
   - 他们希望验证响应的输出（特别是格式），并观察到 LLM 并不总是遵守格式的情况。
- ****Output Refinement 作为 Assertion 替代方案****: 在 **DSPy 2.6** 中，**Assertions** 被通过 `BestOfN` 和 `Refine` 等模块实现的 **Output Refinement** 所取代，旨在通过使用不同参数设置进行多次 LM 调用来提高预测的可靠性和质量，详见 [DSPy 文档](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/)。
- ****QdrantRM 受到质疑****: 一位用户询问 **QdrantRM** 是否在 **DSPy 2.6** 中被移除。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/">Output Refinement - DSPy</a>: 用于编程（而非提示）语言模型的框架。</li><li><a href="https://dspy.ai/learn/programming/7-assertions/?h=dspy.suggest#dspyassert-and-dspysuggest-api">DSPy Assertions - DSPy</a>: 用于编程（而非提示）语言模型的框架。</li><li><a href="https://github.com/philschmid/mcp-openai-gemini-llama-example/blob/master/sqlite_llama_mcp_agent.py">mcp-openai-gemini-llama-example/sqlite_llama_mcp_agent.py at master · philschmid/mcp-openai-gemini-llama-example</a>: 通过在 GitHub 上创建账号来为 philschmid/mcp-openai-gemini-llama-example 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1351252518205919368)** (1 messages): 

> `Caiming Xiong, Multimodal Agents, Vision-Language-Action Alignment, OSWorld, AgentTrek` 


- **Salesforce 的 Caiming Xiong 将发表关于 Multimodal Agents 的演讲**：Salesforce AI 研究高级副总裁 Caiming Xiong 将于今日 PDT 时间下午 4 点进行关于 **Multimodal Agents** 的讲座，并在 [YouTube](https://www.youtube.com/live/n__Tim8K2IY) 上进行直播。
   - 演讲将涵盖跨多种模态整合 **perception（感知）、grounding（对齐）、reasoning（推理）和 action（行动）**，以改变 **GUI 自动化和家用机器人**等任务。
- **探索 Multimodal Agents 领域全景**：讲座将探讨在现实环境中衡量能力（**OSWorld**）、创建大规模数据集（**AgentTrek**）以及设计先进的建模架构（**Aguvis, Magma**）。
   - 还将讨论结合合成思维与行动链（**TACO**），以实现更稳健的 **vision-language-action alignment**。
- **Caiming Xiong 的背景**：Caiming Xiong 在纽约州立大学布法罗分校获得计算机科学博士学位，专注于 **natural language processing, computer vision, reinforcement learning 和 deep learning** 等领域。
   - 他已发表 **200 多篇论文**，引用次数 **>50,000 次**，并曾担任多个研讨会的组织委员会成员。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1350958757718528060)** (7 messages): 

> `Advanced LLM agent course enrollment, Course certification` 


- **高级 LLM Agent 课程仍在接受报名！**：成员询问是否仍可以报名参加 **Advanced LLM agent course**。
   - 工作人员回复称，只需填写 **signup form** 即可！
- **证书仍可获得！**：成员询问在报名课程后是否仍能获得 **certificate**。
   - 工作人员回复称，介绍幻灯片中的大部分信息仅适用于 **Berkeley 学生**，其他人绝对仍可以参加 **MOOC** 并在结束时获得 **certificate**！


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1350293966460948532)** (4 messages): 

> `Self-reflection and self-refinement in LLMs, System prompts and LLM behavior` 


- **讨论自我反思的二分法困境**：一位成员指出 **Lecture 1** 与 **Lecture 2** 之间存在矛盾：Lecture 1 称 **self-reflection 和 self-refinement** 需要外部评估，而 Lecture 2 则建议 LLM 可以通过奖励自己的输出来改进。
   - 附上了 **Lecture 1 第 67 页幻灯片**和 **Lecture 2 第 51 页幻灯片**的截图来阐明这一明显的冲突。参见 [图片 1](https://cdn.discordapp.com/attachments/1282734248112947210/1351127068745928816/image.png?ex=67d9e763&is=67d895e3&hm=7d31b7a0583550a36a872d74bfaf765de39c6b1173333d2ce51174940c0aa522&) 和 [图片 2](https://cdn.discordapp.com/attachments/1282734248112947210/1351127069169418260/image.png?ex=67d9e764&is=67d895e4&hm=12bbe1810790f7f688b11fe093f693a2791e94bd9e74e71ec7c2cfa3264bd004&)。
- **System Prompt 的可靠性受到质疑**：一位成员提出，虽然 System Prompts 应该起作用，但依赖特定行为可能并不稳健，因为*归根结底所有这些都是文本输入，所以模型可以处理它。你应该能够绕过框架和服务。*
   - 他们补充说，训练数据看起来像 `<system> You are a helpful assistant </system> <user> &#123;&#123;Some example user prompt&#125;&#125; </user> <assistant> &#123;&#123;Expected LLM output&#125;&#125; </assistant>`，并且框架可能无法可靠地将 System Prompts 传递给所有 LLM。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1350548977237622916)** (5 messages): 

> `Modular AI Art, Discord Spam` 


- **Modular 的 AI 艺术受到赞赏**：一位成员对 **Modular** 使用的 **AI art** 表示赞赏。
   - 他们表示：“Modular 使用的所有 AI 艺术都很棒！”
- **Discord 垃圾信息澄清**：一位成员澄清了 Discord 频道中的某些消息是垃圾信息。
   - 另一位成员以点赞手势确认了这一澄清。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1350217685136572466)** (6 messages): 

> `Compact Dict, SIMD, stdlib Dict` 


- **Compact Dict 的当前状态**：成员们讨论了 [compact-dict](https://github.com/mzaks/compact-dict) 实现的当前状态，指出其原始版本可能已经过时。
   - 有人建议 Compact Dict 的大部分功能已经合并到了 **stdlib** 的 `Dict` 中。
- **stdlib Dict 在 SIMD 下的性能问题**：一位用户报告了在使用 **stdlib Dict** 处理 **SIMD** [float64, 1] 类型时的性能问题。
   - 他们使用了 hash 库中的 `hash()` 函数，发现其速度较慢，因此正在寻找更快的替代方案。



**提及的链接**：<a href="https://github.com/mzaks/compact-dict">GitHub - mzaks/compact-dict: A fast and compact Dict implementation in Mojo 🔥</a>：一个在 Mojo 🔥 中快速且紧凑的 Dict 实现。可以通过在 GitHub 上创建账户来为 mzaks/compact-dict 的开发做出贡献。

  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1350729782769221642)** (2 messages): 

> `AI4Legislation Competition, AI Demo Jam, Silicon Valley Chinese Association Foundation, Dnipro VC, Data Phoenix` 


- **AI4Legislation 竞赛启动！**：硅谷华人协会基金会 (SVCAF) 正在举办 **AI4Legislation** 竞赛，奖金高达 **$3,000**，活动将持续到 **2025 年 7 月 31 日**，旨在鼓励用于立法参与的开源 AI 解决方案；[竞赛仓库](https://github.com/svcaf/2025-AI4Legislation-Public) 现已上线。
   - SVCAF 将于 2025 年 3 月底举行关于该竞赛的在线研讨会，届时将有 AI 和立法领域的领导者出席；在此 [预约](https://forms.gle/pmbkRLVurbXcGBbAA)。
- **参加 AI Demo Jam 尽情狂欢！**：**Dnipro VC** 和 **Data Phoenix** 将于 **3 月 20 日**在加州桑尼维尔举办 **AI Demo Jam**，届时将有 5 家 AI 初创公司展示其产品，此外还有专家小组讨论、开放式麦克风路演和高能量的社交活动。
   - 小组专家包括 Marianna Bonechi (**Dnipro VC**)、Nick Bilogorskiy (**Dnipro VC**)、Dmytro Dzhulgakhov (**fireworks.ai**)；在此 [注册](https://lu.ma/AI-demo-jam)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://lu.ma/AI-demo-jam">AI Demo Jam: Where Innovation Meets Action · Luma</a>：Dnipro VC 和 Data Phoenix 荣幸呈现：3 月 20 日，走进 AI Demo Jam 探索 AI 的未来 —— 这是一个汇聚精英的系列活动……</li><li><a href="https://forms.gle/pmbkRLVurbXcGBbAA">March AI4Legislation Seminar RSVP</a>：感谢您对 SVCAF 的 AI4Legislation 研讨会感兴趣！硅谷华人协会基金会（成立于 2015 年）将于今年夏天举办一场竞赛，开发开源 AI 驱动的……
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1350729843007819796)** (2 messages): 

> `AI4Legislation competition, object detection in MRI` 


- **AI4Legislation 竞赛启动**：硅谷华人协会基金会正在举办 **AI4Legislation** 竞赛，截止日期为 **2025 年 7 月 31 日**，旨在鼓励在立法过程中促进公民参与的开源 AI 解决方案。
   - 奖金从 **$1,000** 到 **$3,000** 不等，您可以在 [竞赛的 GitHub 仓库](https://github.com/svcaf/2025-AI4Legislation-Public) 中找到更多详情，并在此 [预约](https://forms.gle/pmbkRLVurbXcGBbAA) 研讨会。
- **社区征集 MRI 目标检测方案**：一位成员请求协助创建一个用于 **MRI 图像目标检测** 的模型，该项目不提供金钱报酬。
   - 未提供关于模型类型、数据可用性或使用场景的具体细节。



**提及的链接**：<a href="https://forms.gle/pmbkRLVurbXcGBbAA">March AI4Legislation Seminar RSVP</a>：感谢您对 SVCAF 的 AI4Legislation 研讨会感兴趣！硅谷华人协会基金会（成立于 2015 年）将于今年夏天举办一场竞赛，开发开源 AI 驱动的……

  

---


### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1350740836987699251)** (2 messages): 

> `Qdrant` 


- **Qdrant 请求被拒绝**：一位成员建议切换到 **Qdrant**，但另一位成员确认他们目前并未使用它。
   - 对话中没有提供关于不使用 **Qdrant** 的原因或未来潜在考虑的进一步背景。
- **这里没有 Qdrant！**：一位用户询问是否可以将系统更改为使用 **Qdrant**（一种向量数据库）。
   - 然而，另一位用户坚定地表示：“*不，我们没有使用 Qdrant*”，在没有进一步解释的情况下结束了该建议。


  

---

### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1350433596686467082)** (2 messages): 

> `API Feature Requests, Repetition Penalty` 


- **请求 API 支持 Repetition Penalty**：一位用户请求在 API 中增加 **repetition penalty 支持**，并表示这是阻碍其广泛采用的关键功能。
   - 该用户表示，缺乏 repetition penalty 支持是限制其增加模型使用量的 *唯一限制因素*。
- **Repetition Penalty 是采用模型的主要障碍**：用户强调，缺乏 **repetition penalty** 功能是阻碍其更广泛使用模型的主要障碍。
   - 提供的消息中未讨论其他背景或替代方案。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

yamashi: https://mistral.ai/news/mistral-small-3-1
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1350472840113033267)** (2 messages): 

> `Learnable Scalars, Mitigating Issues in Models, Model Convergence` 


- **Learnable Scalars 缓解模型问题**：一位用户分享了一篇论文链接 [Mitigating Issues in Models with Learnable Scalars](https://www.alphaxiv.org/abs/2503.10622)。
   - 消息作者还指出，*通过引入 learnable scalar 缓解了该问题，模型可以正常收敛 (converge)*。
- **模型收敛性得到改善**：learnable scalar 有助于模型 *正常收敛 (converge normally)*。
   - 这提出了一种稳定训练的实用方法。



**提到的链接**：<a href="https://www.alphaxiv.org/abs/2503.10622">Transformers without Normalization | alphaXiv</a>：查看 1 条评论：Awesome work! Transformers without Normalization podcast

  

---


---


{% else %}


> 完整的频道细分内容已在邮件中截断。
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}