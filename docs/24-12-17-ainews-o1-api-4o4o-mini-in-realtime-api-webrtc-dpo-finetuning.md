---
companies:
- openai
- google
- google-deepmind
date: '2024-12-18T01:43:51.256632Z'
description: '**OpenAI** 推出了功能增强的 **o1 API**，新增了包括视觉输入、函数调用、结构化输出以及全新的 `reasoning_effort`（推理力度）参数，使得推理
  Token 的平均消耗降低了 **60%**。官方确认 **o1 pro** 变体是一个即将推出的独立实现版本。


  **Realtime API**（实时 API）也迎来了改进，通过集成 **WebRTC** 简化了使用流程，支持更长的会话（最长可达 **30 分钟**），并显著降低了价格（使用
  mini 模型可便宜多达 **10 倍**）。此外，OpenAI 还引入了用于微调的 **DPO 偏好微调**功能，目前已支持 **4o** 模型。


  其他更新还包括官方发布的 Go 和 Java SDK 以及 OpenAI DevDay 的视频回顾。新闻中还重点讨论了 **Google Gemini 2.0
  Flash** 模型的性能表现，其准确率已达到 **83.6%**。'
id: 36b4513c-1d0d-4df2-96ad-9e13095c3d11
models:
- o1-2024-12-17
- o1
- o1-pro
- 4o
- 4o-mini
- gemini-2-0-flash
- claude-3.5-sonnet
- claude-3.5
original_slug: ainews-o1-api-4o4o-mini-in-realtime-api-webrtc
people:
- aidan_mclau
- kevinweil
- simonw
- michpokrass
- morgymcg
- juberti
title: o1 API、Realtime API + WebRTC 中的 4o/4o-mini、DPO 微调。
topics:
- function-calling
- structured-outputs
- vision
- reasoning
- webrtc
- realtime-api
- preference-tuning
- fine-tuning
- api
- model-performance
type: archival
---

<!-- buttondown-editor-mode: plaintext -->**[更好的 API 就是实现 AGI 的全部所需](https://www.latent.space/p/openai-api-and-o1)。**

> 2024/12/16-2024/12/17 的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**210** 个频道和 **4050** 条消息）。预计节省阅读时间（以 200wpm 计算）：**447 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

这是 OpenAI 的一个[小型开发者日](https://x.com/kevinweil/status/1869084308432109948)，发布了大量小型更新和一个备受期待的 API。让我们依次来看：

## o1 API


![image.png](https://assets.buttondown.email/images/11a86b0a-2af2-4abc-b7ee-6e1b66b8bc6b.png?w=960&fit=max)


要点：

- [`o1-2024-12-17` 是比两周前发布到 ChatGPT 的 o1 更先进的 o1](https://x.com/aidan_mclau/status/1869092738991612237?s=46)（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-200-chatgpt-pro-and-o1-fullpro-with-vision/)），平均减少了 [60% 的推理 token (reasoning tokens)](https://x.com/OpenAIDevs/status/1869160041741488500)。
- 视觉/图像输入（我们在完整的 o1 发布中看到了这一点，但现在它已进入 API）。
- o1 API 还具有 function calling 和 structured outputs 功能——对基准测试有[一些但非常小的影响](https://x.com/OpenAIDevs/status/1869160041741488500)。
- 一个新的 `reasoning_effort` 参数（目前只是 [`low`/`medium`/`high`](https://github.com/openai/openai-python/blob/19ecaafeda91480d0dfd7ce44e7317220b9d48b6/src/openai/types/chat/chat_completion_reasoning_effort.py#L4) 字符串）。
- “system message” 已更名为 “developer messages”，原因[众所周知](https://x.com/simonw/status/1869101725266932158?s=46)……（开个玩笑，这只是将主要的 chatCompletion 行为更新为与 realtime API 相同的工作方式）。


![image.png](https://assets.buttondown.email/images/d3acb85e-5749-46f4-bc40-731432f8cde9.png?w=960&fit=max)


o1 pro 已确认[“是不同的实现，而不仅仅是设置了 `high` reasoning_effort 的 o1”](https://x.com/michpokrass/status/1869102222598152627)，并将在“[一段时间后](https://x.com/morgymcg/status/1869105067938251028)”在 API 中提供。

## WebRTC 和 Realtime API 改进

现在通过 WebRTC 使用 [RealTime API](https://x.com/juberti/status/1869101256754803098) 变得容易得多，因为它[短到可以放进一条推文](https://x.com/OpenAIDevs/status/1869116585044259059)（可以在 [SimwonW 的演示中使用您自己的 key](https://x.com/simonw/status/1869143764775907494) 进行尝试）：


![image.png](https://assets.buttondown.email/images/849aee9c-232b-4d56-b368-6722583a58a8.png?w=960&fit=max)


新的 4o 和 4o-mini 模型，[仍处于预览阶段](https://platform.openai.com/docs/guides/realtime)：


![image.png](https://assets.buttondown.email/images/c7b1c47a-6904-4760-a434-1a63e50f06ba.png?w=960&fit=max)


WebRTC 的创始人 Justin Uberti 最近加入了 OpenAI，他还[强调了一些其他细节](https://x.com/juberti/status/1869122352656900129?s=46)：

- 价格优化（**使用 mini 时便宜 10 倍**）。
- 更长的持续时间（**会话限制现在为 30 分钟**）。


## DPO 偏好微调 (Preference Tuning)

这是[针对微调 (finetuning) 的偏好排序](https://platform.openai.com/docs/guides/fine-tuning#preference)。我们打算尽快为 AINews 尝试这个……尽管它似乎只适用于 4o。


![image.png](https://assets.buttondown.email/images/40775bce-4c1d-4e82-9abe-1318fd251436.png?w=960&fit=max)



![image.png](https://assets.buttondown.email/images/9a5236ea-2d0a-4a15-9158-7b900d8b23c2.png?w=960&fit=max)


## 其他

精选的 [OpenAI DevDay 视频也已发布](https://www.youtube.com/watch?v=wnsZ7DuqYp0&list=PLOXw6I10VTv_o0ZLpFu2IQyQOho1l-v7y&index=26)。

为有需要的人提供官方的 Go 和 Java SDK。

团队还[进行了一次 AMA](https://community.openai.com/t/ama-on-the-17th-of-december-with-openais-api-team-post-your-questions-here/1057527)（[摘要在此](https://x.com/btibor91/status/1869110487709069337?s=46)，没什么太令人惊讶的）。

完整的演示值得一看：

https://www.youtube.com/watch?v=14leJ1fg4Pw


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

以下是关键讨论和公告的分类摘要：

**模型发布与性能**

- **OpenAI o1 API 发布**：[@OpenAIDevs 宣布](https://twitter.com/OpenAIDevs/status/1869156065788715409) o1 已在 API 中上线，支持 function calling、structured outputs、vision 能力和 developer messages。据报道，该模型使用的 **reasoning tokens 比 o1-preview 少 60%**。
- **Google Gemini 更新**：[Gemini 2.0 Flash 在新的 DeepMind FACTS 基准测试中达到了 83.6% 的准确率](https://twitter.com/_philschmid/status/1869052954579189976)，表现优于其他模型，取得了显著进步。
- **Falcon 3 发布**：[@scaling01 分享](https://twitter.com/scaling01/status/1869007562034544939) Falcon 发布了新模型（1B, 3B, 7B, 10B & 7B Mamba），在 **14 万亿 (Trillion) tokens** 上训练，采用 Apache 2.0 许可证。

**研究与技术进展**

- **Test-Time Computing**：[@_philschmid 强调](https://twitter.com/_philschmid/status/1868919520741445797)了 **Llama 3 3B 如何通过 test-time compute 方法在 MATH-500 上超越 Llama 3.1 70B**。
- **Voice API 定价**：OpenAI 宣布 [GPT-4o 音频现在便宜了 60%](https://twitter.com/omarsar0/status/1869087552332075009)，而 GPT-4o-mini 的音频 token 价格则**便宜了 10 倍**。
- **WebRTC 支持**：Realtime API 新增了使用 WHIP 协议的 [WebRTC 端点](https://twitter.com/juberti/status/1869109071137361926)。

**公司动态**

- **Midjourney 观点**：[@DavidSHolz 分享](https://twitter.com/DavidSHolz/status/1868826489640436061)了关于运营 Midjourney 的见解，指出他们在没有投资者的情况下拥有**“足够的收入来资助大量疯狂的研发”**。
- **Anthropic 安全事件**：[该公司确认](https://twitter.com/AnthropicAI/status/1869139895400399183)其账号出现了未经授权的帖子，并表示没有任何 Anthropic 系统遭到破坏。

**迷因与幽默**

- [@jxmnop 调侃](https://twitter.com/jxmnop/status/1869154293888258139)在 IMAX 观看《Attention Is All You Need》
- 社区中分享了多个关于模型对比和 AI 发展竞赛的幽默段子。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Falcon 3 问世，拥有惊人的 Token 训练量和多样化的模型**

- **Falcon 3 刚刚发布** ([得分: 332, 评论: 122](https://reddit.com/r/LocalLLaMA/comments/1hg74wd/falcon_3_just_dropped/))：**Falcon 3** 已发布，根据 [Hugging Face 博客文章](https://huggingface.co/blog/falcon3)显示，其基准测试表现令人印象深刻。此次发布突显了 AI 模型性能的重大进步。
  - **模型性能与基准测试**：**Falcon 3** 的发布涵盖了从 **1B 到 10B** 的模型，在 **14 万亿 tokens** 上进行了训练。**10B-Base 模型**被认为是该类别中的 state-of-the-art，具体性能得分包括 **MATH-Lvl5 为 24.77**，**GSM8K 为 83.0**。基准测试表明 Falcon 3 与 **Qwen 2.5 14B** 和 **Llama-3.1-8B** 等其他模型相比具有竞争力。
  - **许可证担忧与 BitNet 模型**：人们对该模型的许可证表示担忧，其中包含一个可能限制其地理使用范围的“撤资条款 (rug pull clause)”。讨论中提到了 **BitNet 模型**的发布，一些人注意到该模型与传统的 **FP16 模型**相比基准测试表现较差，尽管它允许在相同硬件上运行更多参数。
  - **社区与技术支持**：社区正在积极讨论对 **Mamba 模型**和**推理引擎支持**，**llama.cpp** 正在持续开发以提高兼容性。人们对 **1.58-bit 量化**方法很感兴趣，尽管目前的基准测试显示与非量化模型相比性能有显著下降。

- **介绍 Falcon 3 系列** ([Score: 121, Comments: 37](https://reddit.com/r/LocalLLaMA/comments/1hg8hpc/introducing_falcon_3_family/)): 该帖子宣布发布 **Falcon 3**，这是一款新的开源大语言模型，标志着 Falcon 团队的一个重要里程碑。欲了解更多详情，读者可参阅 [Hugging Face](https://huggingface.co/blog/falcon3) 上的官方博客文章。
  - **LM Studio** 预计将通过 **llama.cpp** 的更新来集成对 **Falcon 3** 的支持，尽管有报告称由于不支持的 **tokenizer** 导致加载模型时出现问题。一种解决方法是应用来自 [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/10864) 的修复并重新编译 **llama.cpp**。
  - 用户提出了关于 **阿拉伯语支持** 的担忧，指出缺乏具有强大阿拉伯语能力以及推理和数学基准测试的模型。回复表明目前尚不支持阿拉伯语。
  - 用户对 **Falcon 3** 的发布和性能表示赞赏，并计划将其纳入即将进行的基准测试中。反馈建议更新模型卡片（model card），包含有关 **tokenizer** 问题及解决方法的说明。


**主题 2. Nvidia 的 Jetson Orin Nano：嵌入式系统的游戏规则改变者？**

- **[终于，我们迎来了新硬件！](https://www.youtube.com/watch?v=S9L2WGf1KrM)** ([Score: 262, Comments: 171](https://reddit.com/r/LocalLLaMA/comments/1hgdpo7/finally_we_are_getting_new_hardware/)): **Jetson Orin Nano** 硬件正式推出，标志着 AI 技术的一个显著发展。这种新硬件通过提供增强的性能和功能，可能会影响 AI 应用，特别是在边缘计算和机器学习领域。
  - **Jetson Orin Nano** 因其低功耗（7-25W）和紧凑的一体化设计而受到称赞，使其适用于机器人和嵌入式系统。然而，也有人对其 **8GB 128-bit LPDDR5 内存** 及 102 GB/s 的带宽提出批评，一些用户认为与同价位的 **RTX 3060** 或 **Intel B580** 等替代品相比，这不足以运行大型 AI 模型。
  - 讨论强调了 **Jetson Orin Nano** 在机器学习应用和分布式 **LLM** 节点中的潜力，一些用户指出其在 **LLM** 任务中比 **Raspberry Pi 5** 快 5 倍。然而，人们也担心其内存带宽会限制 **LLM** 性能，强调了 RAM 带宽对 **LLM** 推理的重要性。
  - 与其他硬件的比较包括提到 **Raspberry Pi** 即将推出的 16GB 计算模块和 **Apple** 的 **M1/M4 Mac mini**，后者提供了更好的内存带宽和能效。用户辩论了 **Jetson Orin Nano** 的价值主张，考虑了其在机器人和机器学习中的专门用途与更通用的计算需求之间的权衡。


**主题 3. ZOTAC 发布配备 32GB GDDR7 的 GeForce RTX 5090：AI 的高端潜力**

- **[ZOTAC 确认配备 32GB GDDR7 显存的 GeForce RTX 5090，5080 和 5070 系列也已列出 - VideoCardz.com](https://videocardz.com/newz/zotac-confirms-geforce-rtx-5090-with-32gb-gddr7-memory-5080-and-5070-series-listed-as-well)** ([Score: 153, Comments: 61](https://reddit.com/r/LocalLLaMA/comments/1hg3ra4/zotac_confirms_geforce_rtx_5090_with_32gb_gddr7/)): **ZOTAC** 已确认 **GeForce RTX 5090** 显卡，该显卡将配备 **32GB 的 GDDR7 显存**。此外，**5080** 和 **5070 系列**也已列出，表明其产品线即将扩张。
  - **显存带宽担忧**：用户对新系列（除 **RTX 5090** 外）的显存带宽表示失望。人们渴望更大的显存容量，特别是对于 **5080**，一些人希望它能有 **24GB**。
  - **市场动态**：新显卡的发布通常会导致市场上充斥着 **RTX 3090** 和 **4090** 等旧型号。由于性价比考虑和新产品的供应问题，一些用户正在考虑购买这些旧型号。
  - **成本与生产见解**：**Nvidia** 旨在实现利润最大化，这影响了更大显存模块的可用性。**4090** 的生产成本约为 **$300**，其中很大一部分归因于显存模块，这暗示了即将推出的型号中新 **GDDR7** 模块容量可能存在限制。


**主题 4. DavidAU 的大规模混合专家 (Mixture of Experts) LLM：一次创意飞跃**

- **(3 models) L3-MOE-8X8B-Dark-Planet-8D-Mirrored-Chaos-47B-GGUF - 又名 The Death Star - NSFW, 非 AI 风格文本** ([Score: 67, Comments: 28](https://reddit.com/r/LocalLLaMA/comments/1hfzcoz/3_models/)): **DavidAU** 发布了一系列新模型，包括庞大的 **L3-MOE-8X8B-Dark-Planet-8D-Mirrored-Chaos-47B-GGUF**，这是他目前为止最大的模型，体积达 **95GB**，并采用独特的 **Mixture of Experts (MOE)** 方法来生成创意和 NSFW 内容。该模型通过进化过程集成了 8 个版本的 **Dark Planet 8B**，允许用户访问这些模型的不同组合并控制性能水平。更多模型和源代码可在 [Hugging Face](https://huggingface.co/DavidAU/L3-MOE-8X8B-Dark-Planet-8D-Mirrored-Chaos-47B-GGUF) 上获取，以供进一步探索和自定义。
  - **DavidAU** 独立开发了这些模型，使用了一种他称之为 "merge gambling"（合并博弈）的方法，通过结合不同模型的元素并选择最佳版本进行进一步开发，从而使 **Dark Planet 8B** 等模型不断进化。
  - 关于 **NSFW benchmarking** 的讨论强调了模型理解细微提示词的重要性，而不是默认输出安全或通用的内容。用户指出，成功的模型会避免 "GPT-isms"（GPT 腔），并能处理详细提示词中描述的复杂场景。
  - 评论者强调了模型评估中 **prose quality**（散文质量）和通用智能的重要性，指出许多模型在保持角色一致性和叙事深度方面表现不佳，往往倾向于总结而非详细的叙述。


**主题 5. Llama.cpp GPU 优化：骁龙笔记本获得 AI 性能提升**

- **Llama.cpp 现已支持骁龙 Windows 笔记本的 GPU** ([Score: 65, Comments: 4](https://reddit.com/r/LocalLLaMA/comments/1hgbbfj/llamacpp_now_supporting_gpu_on_snapdragon_windows/)): **Llama.cpp** 现在支持骁龙 Windows 笔记本上的 **GPU**，特别是利用了 **Qualcomm Adreno GPU**。帖子作者好奇该功能何时会集成到 **LM Studio** 和 **Ollama** 等平台中，并期待 **KoboldCpp** 的 **ARM** 版本发布。更多详情可见 [Qualcomm 开发者博客](https://www.qualcomm.com/developer/blog/2024/11/introducing-new-opn-cl-gpu-backend-llama-cpp-for-qualcomm-adreno-gpu)。
  - **FullstackSensei** 批评为 **Qualcomm Adreno GPU** 添加 OpenCL 后端是多余的，认为其效率低于使用 **Hexagon NPU**，且尽管 token 处理速度略有提升，但会导致更高的功耗。他们强调系统瓶颈仍然在于内存带宽，约为 **136GB/sec**。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. 煎牛排挑战凸显 Google 在 AI 视频渲染领域的领先地位**

- **不同视频生成模型之间的煎牛排挑战。Google Veo 以巨大优势获胜** ([Score: 106, Comments: 13](https://reddit.com/r/OpenAI/comments/1hg6868/steak_off_challenge_between_different_video_gens/)): **Google Veo** 在视频渲染方面优于竞争对手，特别是在处理手指和切割物理效果等复杂元素方面。**Hunyan** 排名第二，**Kling** 紧随其后，而 **Sora** 表现不佳。原始讨论可以在这篇 [推文](https://x.com/blizaine/status/1868850653759783033) 中找到。
  - **Google Veo** 因其逼真的渲染效果而受到赞誉，特别是在拿刀等细微细节上，使其与竞争对手相比更具人性化。用户注意到视频中对食物准备的描绘具有令人印象深刻的真实感，这表明其在烹饪素材上进行了广泛的训练。
  - **Hunyan Video** 的质量也受到了关注，尽管幽默的评论指出了它的渲染怪癖，例如使用塑料刀。这表明虽然 **Hunyan** 仅次于 **Google Veo**，但仍有明显的改进空间。
  - 讨论暗示了在不久的将来出现此类高质量视频渲染的开源版本的潜力，表达了对该领域更广泛的可访问性和创新的兴奋与期待。


**主题 2. Gemini 2.0 Flash 模型凭借先进的角色扮演和上下文能力增强 AI 体验**

- **[Gemini 2.0 Advanced 发布](https://i.redd.it/cpi755mmng7e1.jpeg)** ([Score: 299, Comments: 54](https://reddit.com/r/OpenAI/comments/1hgioy8/gemini_20_advanced_released/)): **Gemini 2.0** 因其复杂的 Roleplay 能力而受到关注，其功能标记为 "1.5 Pro"、"1.5 Flash" 以及实验性选项如 "2.0 Flash Experimental" 和 "2.0 Experimental Advanced"。界面因其简洁有序的布局而备受好评，强调了每个版本的功能。
  - 关于 **Gemini 2.0** 与 **1206** 和 **Claude 3.5 Sonnet** 等其他模型在 Coding 任务中的有效性存在争论，如果 **1206** 确实是 **2.0**，一些用户表示失望。像 **Salty-Garage7777** 这样的用户报告称，**2.0** 比 **Flash** 更聪明，更擅长遵循 Prompt，但在图像识别方面较差。
  - **2.0 Flash** 模型因其 Roleplay 能力而受到称赞，用户如 **CarefulGarage3902** 强调了其长 Context Length 和自定义选项。该模型因其复杂性以及调整审查过滤器和创造力的能力而比 **ChatGPT** 等主流替代方案更受青睐。
  - 用户对 **Gemini 2.0** 在 Google Pixel 手机的 **Gemini App** 等平台上的可用性和集成情况感兴趣，尽管目前尚未上线。此外，用户正在寻求 Coding Benchmarks 以将 **Gemini 2.0** 与其他模型进行比较，一些人对缺乏此类数据表示沮丧。


---

# AI Discord 回顾

> 由 O1-mini 总结的总结之总结

**主题 1. AI 模型争夺霸权**

- **[Phi-4 在 STEM 领域超越 GPT-4](https://arxiv.org/pdf/2412.08905)**: 拥有 **140 亿参数** 的 **phi-4 模型** 通过利用合成数据和先进的训练技术，在 STEM 相关的 QA 中超越了 **GPT-4**，证明了参数规模并非一切。
  - 尽管与 **phi-3** 相比只有微调，但 phi-4 改进后的课程学习（Curriculum）显著提升了推理 Benchmark。
- **[Gemini Flash 2 提升代码生成水平](https://x.com/OpenRouterAI/status/1869077909438091485)**: **Gemini Flash 2** 在科学编程任务中优于 **Sonnet 3.5**，特别是在数组大小调整方面，标志着代码生成 AI 进入了新纪元。
  - 用户对集成外部框架以进一步增强其能力感到兴奋。
- **[Cohere 的 Maya 在工具使用方面表现出色](https://github.com/cybertronai/maya)**: **Maya** 的发布引起了开发者的热议，并计划对其进行微调以增强工具利用率，以前所未有的方式突破项目边界。

**主题 2. AI 工具在定价与集成方面的挑战**

- **[Windsurf 的困扰：代码覆盖与定价难题](https://codeium.com/blog/pricing-windsurf)**：用户感叹 **Windsurf** 不仅在修改文件方面表现挣扎，还引入了令人困惑的新定价方案，使得资源管理变得更加困难。
  - 建议包括集成 Git 以实现更好的版本控制，防止不必要的代码覆盖。
- **[Codeium 的额度危机引发混乱](https://codeium.canny.io/feature-requests/p/add-gemini-20)**：**Codeium** 中 **Flex credits** 的快速消耗让用户不得不争相购买更大额度的资源包，凸显了对更清晰定价层级的需求。
  - 社区正在讨论新额度限制的公平性以及付费层级的具体细节。
- **[Aider 的 API 发布：功能丰富但价格昂贵](https://aider.chat/docs/install.html)**：**O1 API** 引入了诸如 reasoning effort 参数等高级功能，但用户对其相对于 Sonnet 等竞争对手的大幅涨价保持警惕。
  - 建议将 **O1** 与 **Claude** 结合使用以发挥各自优势，但也引发了对响应过度自信的担忧。

**主题 3. 优化 AI 部署与硬件利用**

- **[量化探索：8B 模型的 2-Bit 魔法](https://huggingface.co/unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit)**：成功将一个 **80 亿参数 (8B)** 模型量化至 **2 bits**，为在受限硬件上部署更大模型打开了大门，尽管初始设置过程较为繁琐。
  - 开发者对于将此方法标准化并应用于超过 **32B parameters** 的模型表现出极高热情。
- **[NVIDIA Jetson Orin Nano Super Kit 提升 AI 处理能力](https://www.theverge.com/2024/12/17/24323450/nvidia-jetson-orin-nano-super-developer-kit)**：定价 **$249**，NVIDIA 的新套件通过将神经运算提升 **70%** 来增强 **AI processing**，使业余爱好者也能获得强大的 AI 能力。
  - 开发者正在探索在 **AGX Orin** 和 **Raspberry Pi 5** 等设备上部署 **LLMs**，以增强本地 AI 能力。
- **[CUDA Graphs 与异步拷贝引发计算难题](https://discord.com/channels/1189498204333543425/1189607726595194971/1318331494883921991)**：在 **4090 GPU** 上将 **cudaMemcpyAsync** 集成到 **CUDA Graphs** 中会导致结果不一致，这令开发者感到困惑，并促使人们深入研究 stream capture 问题。
  - 目前正在进行的调查旨在解决这些差异并优化 **compute throughput**。

**主题 4. 开发者工作流中的 AI 增强**

- **[Cursor 扩展：Markdown 魔法与网页发布](https://marketplace.visualstudio.com/items?itemName=SpecStory.specstory-vscode)**：新的 **Cursor Extension** 允许将 composer 和聊天历史无缝导出为 Markdown，并支持一键网页发布，极大地提升了开发者的生产力。
  - 用户称赞其能够毫不费力地捕捉并分享编程交互过程。
- **[Aider 的 Linter 与代码管理革新工作流](https://aider.chat/docs/usage/lint-test.html)**：通过对各种 Linter 的内置支持和可自定义的 linting 命令，**Aider** 为开发者在管理代码质量和 AI 驱动的编辑方面提供了无与伦比的灵活性。
  - 自动 linting 功能可以切换开关，从而实现量身定制的编程体验。
- **[SpecStory 扩展改变 AI 编程历程](https://marketplace.visualstudio.com/items?itemName=SpecStory.specstory-vscode)**：**VS Code** 的 **SpecStory** 扩展可以捕捉、搜索并学习每一次 AI 辅助编程会话，为开发者优化实践提供了一个丰富的存储库。
  - 增强了对编程交互的文档记录和分析，以获得更好的学习成果。

**主题 5. 社区活动与教育倡议驱动创新**

- **[DevDay 假日版与 API AMA 盛况](https://www.youtube.com/live/XKABimtOWME?si=_EsIUcPOK8-UTWL5)**：OpenAI 的 **DevDay Holiday Edition** 直播以一场由 API 团队参与的 AMA（问我任何事）圆满结束，为开发者提供了丰富的见解和直接互动的机会。
  - 社区成员热切期待关于 API 紧迫问题的解答以及未来功能的发布。
- **[Code Wizard 黑客松为二月热潮寻求赞助](https://discord.com/channels/954421988141711382/954421988783444043/1318696792207921213)**：组织者正在为即将于 2025 年 2 月举行的 **Code Wizard** 黑客松寻找赞助，旨在促进参与者的创新和问题解决能力。
  - 尽管有人质疑资金需求，但许多人强调了黑客松在构建有价值的技术项目方面的作用。
- **[LLM Agents MOOC 延长提交截止日期](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform)**：**Hackathon** 提交截止日期延长 **48 小时**至 **12 月 19 日**，明确了提交流程，并为参与者提供了额外时间来完善他们的 AI agent 项目。
  - MOOC 网站改进后的移动端响应能力获得了赞誉，有助于参与者展示他们的创新成果。


---

# PART 1: 高层级 Discord 摘要




## [Codeium / Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 在文件修改方面遇到困难**：用户报告称 **Windsurf** 无法有效地修改或编辑文件，一位用户形容其在最近更新后变得“更笨了”。
   - 讨论强调了**资源耗尽错误**以及对新定价方案引入的困惑。
- **对 Codeium 定价模式的困惑**：成员询问如何购买更大额度的 **Flex credits**，并指出尽管他们努力管理使用情况，消耗依然很快。
   - 对话集中在最新设立的额度限制以及不同付费层级的细节上。
- **性能对决：Windsurf vs Cursor**：参与者对比了 **Windsurf** 和 **Cursor**，观察到两个平台都提供类似的 agent 功能，但在 context 使用策略上有所不同。
   - **Windsurf** 执行严格的额度限制，而 **Cursor** 在达到一定的 premium 使用量后提供无限次的慢速查询，一些用户认为后者更人性化。
- **Windsurf 代码管理中的挑战**：用户对 **Windsurf** 倾向于覆盖代码并引入幻觉错误表示沮丧，这使开发过程变得复杂。
   - 建议集成 Git 进行版本控制，以便更好地管理更改并实现可逆性。
- **评估 Gemini 2.0 与 Windsurf 的集成**：工程师们正在评估 **Gemini 2.0** 与 **Windsurf** 的结合，注意到其在 context 方面的显著优势，但在输出质量方面评价褒贬不一。
   - 虽然 **Gemini 2.0** 提供了更大的 context window，但一些用户观察到在超过特定 token 限制后性能会有所下降。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **phi-4 模型在 STEM QA 中表现优于 GPT-4**：拥有 **140 亿参数**的 [phi-4](https://arxiv.org/pdf/2412.08905) 模型通过整合合成数据和增强的训练方法，在以 STEM 为重点的 QA 中超越了 **GPT-4**。
   - 尽管在架构上与 **phi-3** 有细微相似之处，但 phi-4 在推理基准测试中表现出强劲的性能，这归功于其修订后的训练课程和后训练技术。
- **8B 模型实现有效的 2-bit 量化**：一个 **80 亿参数**的模型成功量化至 **2 bits**，尽管初始设置复杂，但展示了作为更大模型标准的潜力。
   - 这一进展表明超过 **32B** 参数的模型可用性将得到增强，成员们对其在未来草案中的适用性表示乐观。
- **Gemini 在采样算法中使用 Threefry**：成员们讨论了 LLM 采样中是否使用了 **Xorshift** 或其他算法，其中一人指出 **Gemma** 使用的是 **Threefry**。
   - **PyTorch** 采用 **Mersenne Twister** 与 Gemini 的方法形成对比，突显了不同 AI 框架之间采样技术的差异。
- **Hugging Face 推进推理时计算（Test-Time Compute）策略**：**Hugging Face** 在推理时计算方法上的最新工作受到称赞，特别是他们在计算效率方面的扩展方法。
   - 一篇 [Hugging Face 博客文章](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) 深入探讨了他们的策略，促进了社区的理解并获得了积极反响。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O1 API 全面发布**：即将推出的 [O1 API](https://aider.chat/docs/install.html) 引入了 **reasoning effort parameters**（推理力度参数）和 **system prompts**（系统提示词）等功能，增强了 AI 能力。
   - 用户预计价格相比 Sonnet 会有显著上涨，对 O1 API 可能带来的成本支出表现出复杂的情绪。
- **O1 与 Claude 提升 AI 性能**：**O1 模型**在特定提示词下展现了增强的响应能力，同时用户建议将其与 **Claude** 结合使用，以发挥两种模型的各自优势。
   - 尽管性能有所提升，但 O1 模型在某些情况下会表现出过度自信，引发了关于模型最佳使用方式的讨论。
- **Aider 的 Linter 与代码管理**：**Aider** 内置支持多种 Linter，并允许通过 `--lint-cmd` 选项进行自定义，详见 [Linting and Testing](https://aider.chat/docs/usage/lint-test.html) 文档。
   - 用户可以切换自动 Linting 功能，在 AI 驱动的代码编辑过程中灵活管理代码质量。
- **Claude 模型限制**：**Claude 模型**被指出在生成某些输出时较为犹豫，且倾向于提供过于谨慎的回答。
   - 用户强调需要更明确的引导才能获得理想结果，突出了提示词具体性的重要性。
- **Aider 与 LM Studio 的集成**：**Aider** 在与 [LM Studio](https://docs.litellm.ai/docs/providers) 集成时面临挑战，包括由于缺少 LLM 提供商而导致的 **BadRequestError** 等错误。
   - 用户在排查故障时发现，通过将 OpenAI 提供商格式配置为 `openai/qwen-2.5-coder-7b-instruct-128k` 可实现成功集成。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **新 UI 推送提升用户体验**：今天早上，团队宣布向所有用户推送 **新 UI** 和 **NotebookLM Plus 功能**，这是提升平台用户体验的持续努力的一部分。[公告详情](https://discord.com/channels/1124402182171672732/1182376564525113484/1318635009959264377)。
   - 然而，部分用户对新 UI 表示不满，指出聊天面板的可见性和笔记布局存在问题，而另一些用户则赞赏更大的编辑器，并建议通过折叠面板来提高可用性。
- **NotebookLM Plus 通过 Google 服务扩大访问范围**：**NotebookLM Plus** 现在可以通过 **Google Workspace** 和 **Google Cloud** 访问，并计划在 **2025** 年初向 **Google One AI Premium** 用户开放。[升级信息](https://support.google.com/notebooklm/answer/15678219?visit_id=638700145029570781-388658972&p=plus&rd=1)。
   - 关于其在 **意大利** 和 **巴西** 等国家可用性的问题，官方回应称正在全球范围内逐步推广。
- **Interactive Audio BETA 仅限早期采用者**：**Interactive Audio** 目前仅对部分选定用户开放，后端改进正在进行中。在过渡期间，没有 ***Interactive mode (BETA)*** 访问权限的用户无需担心。[Interactive Audio 详情](https://discord.com/channels/1124402182171672732/1182376564525113484/1318635009959264377)。
   - 多位用户反映 **Interactive Mode** 功能存在困难，称即使更新到新 UI 后仍存在延迟和访问问题，表明该功能仍在推送中。
- **呼叫中心 AI 集成讨论**：成员们探讨了将 **AI 集成**到 **IT 呼叫中心**的可能性，包括一段关于德语 AI 处理客户查询的幽默讨论，并分享了演示计算机故障排除和冷启动销售电话等场景的音频剪辑。[使用案例](https://discord.com/channels/1124402182171672732/1124403655819415592/1318343114552901683)。
   - 讨论强调了通过 AI 实施提升客户服务效率的潜力。
- **NotebookLM 扩展多语言支持**：用户询问了 **NotebookLM** 生成播客的多语言能力，确认目前 **音频摘要** 仅支持 **英语**。[常规频道](https://discord.com/channels/1124402182171672732/1124402182909857966/1318308951393308712)。
   - 尽管有此限制，**葡萄牙语** 内容的成功生成表明未来更新中可能支持更广泛的语言。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Phi-4 在 STEM 问答中超越 GPT-4**：**phi-4** 是一个拥有 140 亿参数的语言模型，它利用强调**数据质量**的训练策略，在整个开发过程中整合了合成数据，使其在**以 STEM 为重点的问答能力**方面表现出色，超越了其教师模型 GPT-4。
   - 尽管自 **phi-3** 以来架构变化极小，但该模型在推理基准测试中的表现突显了改进的训练课程，详情见 [Continual Pre-Training of Large Language Models](https://arxiv.org/abs/2308.04014)。
- **Unsloth 4-bit 模型显示出性能差距**：用户报告了 **Unsloth 4-bit 模型**与原始 Meta 版本之间在层大小上的差异，突显了**模型参数化**的潜在问题。
   - 用户对从 4-bit 转换到**全精度（full precision）**时的 **VRAM 占用**和性能权衡表示担忧，如 [Qwen2-VL-7B-Instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit) 仓库中所讨论的。
- **Qwen 2.5 微调面临灾难性遗忘**：一位成员表示，他们微调后的 **Qwen 2.5** 模型表现不如原始版本，并将这种下降归因于**灾难性遗忘（catastrophic forgetting）**。
   - 其他成员建议对微调过程进行迭代，以更好地与特定目标对齐，并强调了定制化调整的重要性。
- **增强 Llama 3.2 中的函数调用**：参与者探索了训练 **Llama 3.2** 以提高 **function calling** 能力的方法，但注意到缺乏直接的实现示例。
   - 大家达成共识，认为将 **special tokens** 直接整合到数据集中可以简化训练过程，参考 [Llama Model Text Prompt Format](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#zero-shot-function-calling-e2e-format)。
- **在 Unsloth 中优化 Lora+**：成员们讨论了将 **Lora+** 与 Unsloth 集成，观察到与其他方法的潜在不兼容性，并建议使用 **LoFTQ 或 PiSSA** 等替代方案以获得更好的初始化。
   - 一位成员通过一篇 [CPT 博客文章](https://unsloth.ai/blog/contpretraining) 强调了 Unsloth 最新版本中的性能改进，并突出了这些优化的好处。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 提升多模态图像嵌入速率**：Cohere 已将 **Multimodal Image Embed** 端点的速率限制提高了 **10 倍**，将生产密钥从 **40 images/min 提升至 400 images/min**。[阅读更多](https://docs.cohere.com/v2/docs/rate-limits)
   - 测试版用户仍限制在 **5 images/min** 以进行测试，这使得在不使系统过载的情况下进行应用开发和社区分享成为可能。
- **Maya 发布增强了工具利用率**：**Maya** 的发布受到了成员们的欢迎，激发了探索并可能针对 **tool use** 对其进行微调的热情。成员们致力于利用新模型突破项目边界。
   - 社区计划进行广泛的测试和定制，旨在将 Maya 的能力有效地整合到他们的工作流中。
- **优化 Cohere API 密钥管理**：**Cohere** 提供两种类型的 API 密钥：**免费但有限制的评估密钥（evaluation keys）**，以及**付费且限制较少的生产密钥（production keys）**。用户可以通过 [API keys 页面](https://dashboard.cohere.com/api-keys)管理他们的密钥。
   - 这种结构允许开发人员高效地启动项目，同时随着应用程序的增长使用生产密钥进行扩展。
- **使用嵌入进行图像检索的策略**：为了实现基于用户查询的图像检索，一位成员建议在 **Pinecone** 向量数据库中将图像路径作为元数据与嵌入（embeddings）一起存储。这使得系统在嵌入匹配查询时能够显示正确的图像。
   - 通过利用嵌入进行**语义搜索（semantic search）**，检索过程变得更加准确和高效，从而提升了用户体验。
- **为 Code Wizard 黑客松寻求赞助**：**Code Wizard** 黑客松的组织者正在积极为定于 **2025 年 2 月**举行的活动寻求赞助，旨在培养参与者的创新和解决问题的能力。
   - 虽然一些参与者质疑资金的必要性，但其他人强调了该活动在构建有价值的项目和提升技术技能方面的作用。



---

## [Bolt.new / Stackblitz](https://discord.com/channels/364486390102097930) Discord

- **带有模型选择功能的 Bolt UI 版本**：一位成员宣布推出了 **Bolt 的 UI 版本**，允许用户在托管于 [Hyperbolic](https://hyperbolic.hosting) 的 **Claude**、**OpenAI** 和 **Llama** 模型之间进行选择。此次更新旨在通过提供多样化的模型选项来增强生成过程。
   - 新 UI 旨在提升**用户体验**并简化**模型选择**，从而实现更具定制化且高效的项目工作流。
- **有效管理 Token**：用户对 Bolt 中 **Token 的使用和管理**表示关注，强调了对意外消耗的挫败感。
   - 会议强调了**每月 Token 使用量**存在限制，用户应留意**更换成本**，以避免超出配额。
- **Bolt 集成面临的挑战**：多位用户报告了 **Bolt 的集成问题**，例如平台生成不必要的文件以及在执行命令时遇到错误。
   - 为了缓解挫败感，一些用户建议在使用平台期间适当休息，强调了在不过度劳累的情况下保持生产力的重要性。
- **用于 SaaS 项目的 Bolt**：成员们表达了利用 **Bolt 开发 SaaS 应用**的兴趣，并认识到需要开发者协助以进行有效的扩展和集成。
   - 一位用户寻求关于使用 Bolt 管理 SaaS 项目的**分步指导**，表明对更全面的支持资源存在需求。
- **编码问题的支持与协助**：用户在 Bolt 内部寻求**编码挑战的支持**，特别是针对其 **Python 代码**的帮助请求。
   - 社区成员提供了关于**调试技术**的建议，并推荐了**在线资源**以改进编码实践。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **基于 Pythia 的 RLHF 模型**：**general** 频道的成员询问了是否有**公开可用的基于 Pythia 的 RLHF 模型**，但讨论中未推荐具体模型。
- **TPU v5p 上的 TensorFlow**：一位用户报告 **TensorFlow** 在 **TPU v5p** 上出现分段错误（segmentation faults），称 `import tensorflow` 在多个 VM 镜像中均会导致错误。
   - 针对 Google 在持续的技术挑战中对 TensorFlow 支持力度减弱的问题，用户表达了担忧。
- **SGD-SaI 优化器方法**：在 **research** 频道中，**SGD-SaI** 的引入提出了一种在没有自适应矩（adaptive moments）的情况下增强随机梯度下降的新方法，取得了与 AdamW 相当的效果。
   - 参与者强调需要与成熟的优化器进行公正的比较，并建议在训练阶段动态调整学习率。
- **Stick Breaking 注意力机制**：**research** 频道的讨论涵盖了 **Stick Breaking Attention**，这是一种自适应聚合注意力分数的技术，旨在减少模型中的过度平滑（oversmoothing）效应。
   - 成员们辩论了这些自适应方法是否能更好地处理 Transformer 架构中学习表征的复杂性。
- **Grokking 现象**：最近的一篇论文通过将神经网络复杂度与泛化联系起来讨论了 **Grokking 现象**，并引入了一个基于 Kolmogorov 复杂度的指标。
   - 该研究旨在辨别模型何时处于泛化与何时处于记忆状态，可能为训练动力学提供结构化的见解。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Extension 发布增强生产力**：**Cursor Extension** 现在允许用户将 composer 和聊天历史导出为 [Markdown](https://marketplace.visualstudio.com/items?itemName=SpecStory.specstory-vscode)，从而提高生产力并方便内容共享。
   - 此外，它还包含一个[将内容发布到 Web](https://marketplace.visualstudio.com/items?itemName=SpecStory.specstory-vscode) 的选项，能够有效捕获编码交互。
- **O1 Pro 高效自动化编码任务**：`@mckaywrigley` 报告称 **O1 Pro** 成功执行了 6 个任务，修改了 **14 个文件**，并在 **5分25秒** 内使用了 **64,852 个 input tokens**，实现了 100% 的正确率并节省了 **2 小时**。
   - 这展示了 **O1 Pro** 在简化复杂编码工作流方面的潜力。
- **RAPIDS cuDF 无需修改代码即可加速 Pandas**：[@NVIDIAAIDev 的推文](https://x.com/NVIDIAAIDev/status/1868778156347339033) 宣布 **RAPIDS cuDF** 可以在不修改任何代码的情况下将 **pandas** 操作加速高达 **150倍**。
   - 开发者现在可以在 **Jupyter Notebooks** 中处理更大的数据集，正如其 [demo](http://nvda.ws...) 中所示。
- **SpecStory 扩展集成 AI 编码历程**：适用于 **Visual Studio Code** 的 **SpecStory** 扩展提供了从每一次 **AI coding journey** 中[捕获](https://marketplace.visualstudio.com/items?itemName=SpecStory.specstory-vscode)、**搜索**和**学习**的功能。
   - 该工具增强了开发者有效记录和分析其编码交互的能力。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DevDay 节日版和 API 团队 AMA**：**DevDay Holiday Edition** YouTube 直播 *Day 9: DevDay Holiday Edition* 已排期，可在此处[观看](https://www.youtube.com/live/XKABimtOWME?si=_EsIUcPOK8-UTWL5)。
   - 直播之后将与 OpenAI 的 API 团队进行 AMA，计划于太平洋时间上午 10:30–11:30 在[开发者论坛](https://community.openai.com/t/ama-on-the-17th-of-december-with-openais-api-team-post-your-questions-here/1057527)举行。
- **AI 口音模仿与现实感局限**：用户讨论了 AI 在多种语言和口音之间切换的能力，例如模仿澳洲口音，但交互感仍然显得不自然。
   - 参与者指出，虽然 AI 可以模仿口音，但由于准则限制，它经常会拒绝某些出于礼貌的交互请求。
- **自定义 GPT 编辑与功能问题**：多位用户报告称失去了**编辑自定义 GPT** 的能力，并且尽管进行了多次设置，仍无法访问它们。
   - 这似乎是一个**已知问题**，其他面临类似自定义 GPT 配置问题的用户也证实了这一点。
- **Anthropic 调整定价模型**：Anthropic 调整了其定价模型，随着 API 引入 prompt caching，价格变得更低。
   - 用户对这一转变将如何影响他们的使用以及与 OpenAI 产品的竞争表示好奇。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 新增对 46 个模型的支持**：OpenRouter 现在支持 **46 个模型** 的结构化输出（structured outputs），增强了多模型应用的开发。该 [demo](https://x.com/OpenRouterAI/status/1869077909438091485) 展示了结构化输出如何将 **LLM 输出约束为 JSON schema**。
   - 此外，结构化输出在 **8 家模型公司**和 **8 个免费模型**之间实现了标准化，有助于更顺畅地集成到应用中。
- **Gemini Flash 2 表现优于 Sonnet 3.5**：在科学问题解决任务中，**Gemini Flash 2** 生成的代码优于 **Sonnet 3.5**，尤其是在数组大小调整（array sizing）场景中。
   - 反馈表明，集成外部框架可以进一步提升其在特定用例中的有效性。
- **尝试通过拼写错误影响 AI 回复**：成员们正在探索在 prompts 中使用故意拼写的错误和无意义词汇来引导模型输出，这可能对创意写作有益。
   - 该技术旨在将模型注意力引导至特定关键词，同时通过思维链（Chain of Thought, CoT）方法保持受控的输出。
- **o1 API 减少 60% 的 token 使用量**：**o1 API** 现在的 **token 消耗减少了 60%**，引发了对其对模型性能影响的关注。
   - 用户讨论了定价调整和提高 token 效率的需求，并指出当前的 tier 限制仍然适用。
- **API key 泄露与报告**：一位成员报告了 GitHub 上暴露的 **OpenRouter API keys**，引发了关于正确报告渠道的讨论。
   - 建议联系 **support@openrouter.ai** 以处理泄露密钥带来的任何安全风险。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 推出 Pro 礼品订阅服务**：Perplexity 现在提供为期 1、3、6 或 12 个月的**礼品订阅**，可在[此处](https://perplexity.supply/shop/perplexity-subscription)购买，使用户能够解锁更强大的搜索功能。
   - 订阅将通过**优惠码（promo codes）**直接发送至接收者的电子邮箱，并注明**所有销售均为最终销售**，以确保购买承诺。
- **关于 OpenAI 借鉴 Perplexity 功能的辩论**：用户们就 **OpenAI** 是在创新还是在抄袭 **Perplexity** 的功能（如 Projects 和 GPT Search）展开了辩论，引发了关于 AI 开发原创性的讨论。
   - 一些成员认为功能复制在各平台间很常见，并就如何保持 AI 工具的**独特价值主张**展开了对话。
- **Mozi 应用在社交媒体的热议中发布**：Ev Williams 发布了新的社交应用 **Mozi**，其全新的社交网络方式正引起关注，详见这段 [YouTube 视频](https://www.youtube.com/embed/RNXnnOT3-9Y)。
   - 该应用承诺提供创新功能，引发了关于其对现有社交媒体平台潜在影响的讨论。
- **对模型性能下降的担忧**：用户对 **Sonnet** 和 **Claude** 模型的变体表示失望，认为感知到的性能下降影响了回答质量。
   - 在特定模型之间切换导致了不一致的用户体验，突显了对模型优化透明度的需求。
- **Gemini API 集成增强了 Perplexity 的产品力**：通过 **OpenAI SDK 实现的新 Gemini 集成**允许与多个 API 进行无缝交互，可通过 [Gemini API](https://ai.google.dev/gemini-api/docs/openai) 访问。
   - 这种集成通过简化对包括 **Gemini**、**OpenAI** 和 **Groq** 在内的多种模型的访问来提升用户体验，**Mistral** 支持也即将推出。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Palmyra Creative 发布 128k 上下文版本**：新的 [Palmyra Creative](https://x.com/waseem_s/status/1869040950464459216) 模型通过 **128k 上下文窗口**增强了**创意业务任务**，适用于头脑风暴和分析。
   - 它与特定领域模型无缝集成，服务于从**营销人员**到**临床医生**的专业人士。
- **OpenAI API 推出支持 Function Calling 的 O1**：[OpenAI](https://x.com/kevinweil/status/1869084308432109948) 在小型开发者日上宣布了更新，包括支持 **Function Calling 的 O1 实现**以及**新的语音模型功能**。
   - 用于实时语音应用的 **WebRTC 支持**和显著的**输出 Token 增强**是主要亮点。
- **NVIDIA 发布 Jetson Orin Nano Super 套件**：NVIDIA 的 [Jetson Orin Nano Super 开发人员套件](https://www.theverge.com/2024/12/17/24323450/nvidia-jetson-orin-nano-super-developer-kit-software-update-ai-artificial-intelligence-maker-pc)提升了 **AI 处理能力**，**神经处理性能提升了 70%** 达到 **67 TOPS**，并拥有 **102 GB/s 内存带宽**。
   - 售价为 **$249**，旨在为爱好者提供**高性价比的 AI 能力**。
- **Aidan McLau 澄清 O1 与 O1 Pro 的区别**：[Aidan McLau](https://x.com/michpokrass/status/1869102222598152627) 澄清说，**O1 Pro** 是与标准 **O1 模型**不同的实现，专为**更高的推理能力**而设计。
   - 这一区分引发了社区关于这些模型之间潜在**功能混淆**的问题。
- **Anthropic API 将四项功能移出 Beta 阶段**：[Anthropic](https://x.com/alexalbert__/status/1869096718387872205) 宣布其 API 的四项新功能正式商用（GA），包括 **Prompt Caching** 和 **PDF 支持**。
   - 这些更新旨在**增强开发者体验**并促进 Anthropic 平台上的**更顺畅运营**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Zotac 的 RTX 50 抢先看**：Zotac 意外在其[网站](https://www.tomshardware.com/pc-components/gpus/zotac-accidentally-lists-rtx-5090-rtx-5080-and-rtx-5070-family-weeks-before-launch-inadvertent-listing-seemingly-confirms-the-rtx-5090-with-32gb-of-gddr7-vram)上列出了即将推出的 **RTX 5090**、**RTX 5080** 和 **RTX 5070** GPU 系列，在正式发布前揭晓了 **32GB GDDR7 显存**等先进规格。
   - 这次意外泄露在社区内引发了轰动，证实了 **RTX 5090** 令人印象深刻的规格，并增加了对 Nvidia 下一代硬件的期待。
- **AMD 驱动困境**：用户报告了 **24.12.1 AMD driver** 的问题，该驱动导致性能下降和 GPU 占用率飙升，且未能有效利用功耗。
   - 回退到 **24.10.1** 版本解决了这些卡顿问题，使各种模型的性能提升至 **90+ tokens/second**。
- **TTS 之梦：LM Studio 的下一步**：一位用户对在 **LM Studio** 中集成 **text to speech** 和 **speech to text** 功能表示乐观，目前已有替代方案作为临时解决方法。
   - 另一位成员建议将这些工具作为服务器与 **LM Studio** 并行运行，以实现所需功能，从而增强整体用户体验。
- **去审查聊天机器人：新选择**：关于寻找**去审查聊天机器人 (uncensored chatbot)** 替代方案的讨论不断涌现，推荐了可以在 CPU 上运行的 [**Gemma2 2B**](https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF) 和 [**Llama3.2 3B**](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF) 等模型。
   - 成员们获得了有效使用这些模型的资源，包括量化 (quantization) 选项的链接，以优化其性能。
- **GPU 大对决：3070 Ti vs 3090**：用户观察到 **RTX 3070 Ti** 和 **RTX 3090** 在游戏中的表现相似，尽管价格区间相当。
   - 一位成员提到能以约 **750 美元**的价格买到 **3090**，而另一位成员则提到当地价格在 **900 加元**左右，凸显了市场价格的差异。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **顶尖 Stable Diffusion 课程**：一位成员正在寻找综合性的[在线课程](https://chatgpt.com/c/67618085-3fc4-8010-93a1-7bf7c6b79806)，这些课程汇集了 YouTube 教程，用于学习使用 A1111 的 Stable Diffusion。
   - 社区强调了获取 Stable Diffusion 易用教育资源的必要性。
- **AI 任务中的笔记本 vs 台式机**：一位用户正在评估 **4090 笔记本**和 **4070 TI Super 台式机**（均配备 **16GB VRAM**）在 AI 任务中的表现。
   - 成员们建议台式机更适合繁重的 AI 工作负载，并指出笔记本更适合游戏，但不适合高强度的图形任务。
- **机器人检测策略**：讨论集中在识别诈骗机器人的技术上，例如询问荒谬的问题或采用“土豆测试 (potato test)”。
   - 参与者强调机器人和人类都可能带来风险，需要谨慎互动。
- **创建你自己的 Lora 模型**：一位用户请求关于构建 **Lora 模型**的指导，并收到了包括数据集创建、模型选择和训练在内的分步方法。
   - 重点放在了研究用于训练目的的有效数据集创建上。
- **最新 AI 模型：Flux.1-Dev**：一位回归成员询问了当前的 AI 模型，特别提到了 **Flux.1-Dev** 及其硬件要求。
   - 社区提供了关于热门模型使用情况和必要实现要求的更新。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Graphs 与 cudaMemcpyAsync 的兼容性**：成员们确认 **CUDA Graph** 支持 **cudaMemcpyAsync**，但将它们集成会导致应用程序结果不一致，特别是在 **4090 GPU** 上会影响计算吞吐量。[更多详情](https://discord.com/channels/1189498204333543425/1189607726595194971/1318331494883921991)
   - 一份报告的问题指出，在 **CUDA Graph** 模式下使用 **cudaMemcpyAsync** 会导致错误的应用程序结果，而 **kernel copies** 则能正常工作。目前正在通过最小示例进行进一步调查，以解决这些差异。
- **优化 PyTorch Docker 镜像**：讨论显示官方 **PyTorch Docker** 镜像大小在 **3-7 GB** 之间，有可能通过使用 **30MB Ubuntu** 基础镜像配合 **Conda** 管理 CUDA 库来减小体积。[GitHub 指南](https://github.com/LambdaLabsML/distributed-training-guide?tab=readme-ov-file)
   - 随后引发了关于是否有必要结合使用 **Conda** 和 **Docker** 的辩论，支持者认为这有助于在不同的开发环境中保持安装的一致性。
- **NVIDIA Jetson Nano Super 发布**：NVIDIA 推出了 **Jetson Nano Super**，这是一款紧凑型 AI 计算机，可为机器人应用提供每秒 **70-T 次操作**，售价 **$249**，并支持 **LLMs** 等先进模型。[推文](https://x.com/slow_developer/status/1869059311969661103)
   - 用户讨论了通过 **SDK Manager** 使用 **JetPack 6.1** 来增强 **Jetson Orin** 的性能，并在 **AGX Orin** 和 **Raspberry Pi 5** 等设备上部署 **LLM inference**，后者利用 **nvme 256GB** 来加速数据传输。
- **使用 Axolotl 和 TRL 进行 VLM 微调**：分享了使用 **Axolotl**、**Unslosh** 和 **Hugging Face TRL** 进行 **VLM fine-tuning** 的资源，包括一份[微调教程](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)。
   - 该过程被指出是**资源密集型**的，需要大量的计算能力，这被强调为实现高效集成时需要考虑的因素。
- **Chain of Thought 数据集生成**：团队启动了一个 **Chain of Thought (CoT) dataset generation** 项目，旨在评估哪种 CoT 形式能最有效地提升模型性能，并利用 **reinforcement learning** 进行优化。
   - 该实验旨在确定 **CoT** 是否能解决超出直接转导（direct transduction）方法能力的谜题，初步进展显示已解决 **119 个谜题**，并有望通过强大的验证器（verifiers）进一步改进。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX 24.6 携 MAX GPU 发布**：今天，**MAX 24.6** 正式发布，推出了备受期待的 **MAX GPU**。这是一个垂直集成的生成式 AI 堆栈，无需再使用 NVIDIA CUDA 等特定厂商的库。欲了解更多详情，请访问 [Modular 博客](https://www.modular.com/blog/introducing-max-24-6-a-gpu-native-generative-ai-platform)。
   - 此版本解决了大规模生成式 AI 日益增长的资源需求，为增强 AI 开发铺平了道路。
- **Mojo v24.6 发布**：最新版本的 **Mojo** (**v24.6.0**) 已经发布并可供使用，可通过命令 `% mojo --version` 确认。社区对新功能表现出极大的热情。
   - **mojo** 频道的用户们渴望探索这些更新，显示出强烈的社区参与度。
- **推出 MAX Engine 和 MAX Serve**：**MAX Engine** 和 **MAX Serve** 随 **MAX 24.6** 一同推出，提供了一个高速 AI 模型编译器和一个针对大语言模型 (**LLMs**) 的 Python 原生服务层。这些工具旨在提升 AI 工作负载的性能和效率。
   - **MAX Engine** 具有针对 NVIDIA GPU 优化的厂商无关的 Mojo GPU kernels，而 **MAX Serve** 则简化了高负载场景下 **LLMs** 的集成。
- **Mojo 确认支持 GPU**：继最近的 24.6 版本之后，即将发布的 **Mojo v25.1.0 nightly** 版本确认将包含 **GPU support**。这一加入展示了 Mojo 平台持续的增强。
   - 社区期待随着 GPU 支持的加入，复杂 AI 工作负载的性能和可扩展性将得到提升。
- **Mojo REPL 在 Archcraft Linux 上遇到问题**：一位用户报告在 **Archcraft Linux** 上进入 **Mojo REPL** 时遇到问题，理由是缺少动态链接库，可能是 `mojo-ldd` 或 `mojo-lld`。
   - 此外，该用户在安装 Python 依赖时也遇到了困难，提到了与处于外部管理环境（externally managed environment）相关的错误。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **探索 NVIDIA NV-Embed-v2 的可用性**：成员们研究了 NVIDIA Embedding 中 **NVIDIA NV-Embed-v2** 的**可用性**，利用 `embed_model.available_models` 功能来验证可访问的模型。
   - 有人指出，即使 NV-Embed-v2 没有被明确列出，它仍可能正常工作，因此需要进行额外测试以确认其可用性。
- **在工作流中集成 Qdrant Vector Store**：一位用户寻求将 **Qdrant vector store** 集成到其工作流中的帮助，并提到了在现有集合和查询执行方面遇到的挑战。
   - 另一位成员提供了 [文档示例](https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/) 并表示他们没有遇到类似问题，建议进行进一步排查。
- **解决 OpenAI LLM 双重重试问题**：**Paullg** 对 OpenAI LLM 中潜在的双重重试问题表示担忧，指出 OpenAI 客户端和 `llm_retry_decorator` 可能都在独立实现重试逻辑。
   - 讨论随后集中在最近的一个 [pull request](https://github.com/run-llama/llama_index/pull/17072) 是否解决了此问题，参与者对拟议更改的有效性表示不确定。
- **LlamaReport 增强文档可读性**：**LlamaReport** 目前处于预览阶段，可在几分钟内将文档数据库转换为**结构良好**、人类可读的报告，从而促进对文档集的有效问答。更多详情请见 [公告帖子](https://twitter.com/llama_index/status/1869094544169677138)。
   - 该工具旨在通过优化输出过程来简化文档交互，使用户更容易导航和利用其文档数据库。
- **Agentic AI SDR 助力线索生成**：新推出的 **agentic AI SDR** 利用 **LlamaIndex** 来生成线索，展示了 AI 在销售策略中的实际集成。其 [代码](https://t.co/tczv5ZDI4H) 已开放供实现。
   - 这一进展是 **Quickstarters** 计划的一部分，该计划通过示例项目和实际应用协助用户探索 **Composio** 的功能。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All v3.5.3 发布，包含关键修复**：**GPT4All v3.5.3** 版本已正式发布，解决了之前版本的显著问题，包括针对 v3.5.2 中运行异常的 **LocalDocs** 的关键修复。
   - **Nomic AI** 的 **Jared Van Bortel** 和 **Adam Treat** 因对此次更新的贡献而受到认可，提升了 GPT4All 的整体功能。
- **新版本恢复 LocalDocs 功能**：阻止 **LocalDocs** 在 v3.5.2 中正常运行的严重问题已在 **GPT4All v3.5.3** 中成功解决。
   - 用户现在在使用 **LocalDocs** 进行文档处理时可以期待更好的性能和可靠性。
- **通过 YouTube 演示探索 AI Agent 能力**：讨论中提到了通过 **GPT4All** 运行“AI Agent”的可能性，并链接到了一个展示其能力的 [YouTube 视频](https://www.youtube.com/watch?v=XeWZIzndlY4)。
   - 一位成员指出，虽然技术上可行，但它主要作为一个生成式 AI 平台，功能有限。
- **Jinja Template 问题困扰 GPT4All 用户**：一位成员报告称，由于 **Jinja template 问题**，**GPT4All** 对他们来说几乎完全无法使用，希望该问题能尽快得到解决。
   - 另一位成员强调了 Jinja templates 对模型交互至关重要，目前正在改进工具调用（tool calling）功能。
- **API 文档请求凸显 GPT4All 的空白**：有人请求提供包含端点和参数详情的完整 **API 文档**，并引用了现有的 [GPT4All API 文档](https://docs.gpt4all.io/gpt4all_api_server/home.html#key-features)。
   - 成员们分享了激活本地 API 服务器只需简单的步骤，但他们认为文档缺乏全面性。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **关于 Gemini 2.0 Flash 的提问**：用户正在询问 **Gemini 2.0 Flash** 的功能，并强调了目前缺乏响应和支持的问题。
   - 这表明在 **OpenInterpreter** 社区中，该功能的用户体验或支持可能存在缺口。
- **关于 VEO 2 和 SORA 的辩论**：成员们在争论 **VEO 2** 是否优于 **SORA**，并指出这两种 AI 目前在他们所在的地区都不可用。
   - 可用性的缺失表明了用户的兴趣，同时也反映了他们想要探索这些选项时的挫败感。
- **OpenInterpreter 与 Web Assembly 的集成**：一位用户提议使用 **Web Assembly** 以及 Pyodide 或 Emscripten 等工具在网页中运行 **OpenInterpreter** 项目。
   - 这种方法可以提供自动沙箱化 (auto-sandboxing) 并消除对计算调用的需求，从而增强在聊天 UI 环境下的可用性。
- **OpenInterpreter 中 OS 的本地使用**：有关于在 **OpenInterpreter** 中本地利用 **OS** 的咨询，用户寻求关于 **OS** 具体涵盖内容的澄清。
   - 这反映了用户对增强功能和本地执行能力的持续兴趣。
- **Open Interpreter 错误排查**：一名成员报告了在使用带有 `-y` 标志的代码时持续出现的错误，特别是与设置 **OpenAI API key** 相关的问题。
   - 这突显了用户面临的常见挑战，以及对错误处理提供更清晰指导的需求。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torcheval 的批量指标同步简化了工作流**：一位成员对 **Torcheval** 的批量指标同步 (batched metric sync) 功能以及无需**额外依赖**表示满意，认为它是一个非常愉快的工具。
   - *这种精简的方法* 提高了生产力并降低了处理指标时的复杂性。
- **指令微调损失计算中的挑战**：一位成员对指令微调 (Instruction Fine-Tuning) 中的**每个 token 损失 (per-token loss)** 计算提出了担忧，指出由于 token 数量不同，一个句子的损失取决于 batch 中的其他句子。
   - *这种方法似乎是标准做法*，但也带来了社区必须去适应的挑战。
- **GenRM 验证器模型提升 LLM 性能**：最近的一篇 [论文](https://arxiv.org/abs/2408.15240v1) 提议使用基于 next-token prediction 训练的**生成式验证器 (GenRM)**，通过将解决方案生成与验证相结合，来增强 **LLM** 的推理能力。
   - 这种方法可以实现更好的指令微调，并具有通过多数投票改进计算的潜力，提供了优于标准 **LLM** 分类器的优势。
- **Sakana AI 的通用 Transformer 内存优化**：**Sakana AI** 的研究人员开发了一种优化 **LLM** 内存使用的技术，允许企业显著降低在 **Transformer** 模型上开发应用的成本。
   - [通用 Transformer 内存 (universal transformer memory)](https://sakana.ai/namm/) 技术在丢弃冗余的同时保留关键信息，从而提高了模型效率。
- **8B 验证器性能分析与社区反应**：针对使用 **8B 奖励/验证器模型** 的担忧被提出，一位成员指出，在性能讨论中不应忽视训练此类模型的计算成本和复杂性。
   - 另一位成员幽默地将该方法比作“让猴子打字，然后让人类挑选最好的一个”，暗示这可能具有误导性，并表示需要更广泛的实验。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Hackathon 截止日期延长 48 小时**：**Hackathon 提交截止日期**已延长 **48 小时**，至 **太平洋时间 12 月 19 日晚上 11:59**。
   - 此次延期旨在消除关于提交过程的困惑，并让参赛者有更多时间完善他们的项目。
- **明确 Hackathon 提交流程**：提醒参赛者应通过 [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform) 提交，而**不是**通过 Devpost 网站。
   - 这一说明对于确保所有项目正确提交至关重要。
- **LLM Agents MOOC 网站完成移动端适配**：一位成员对 **LLM Agents MOOC 网站**进行了翻新，以获得更好的移动端响应能力，并在[此链接](https://gilbertomedrano.com/berkeley-ai-mooc-website/index.html)分享了更新版本。
   - *希望这能成为回馈 MOOC/Hackathon 的一种方式。* 另一位用户称赞了该设计，并表示计划将其分享给工作人员。
- **证书截止日期确认至 12/19**：一位用户询问了证书提交截止日期，因为不确定是否会延期。
   - 另一位成员确认 MOOC **没有截止日期变更**，并强调提交表单将一直开放到 **12/19** 以方便大家。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **探索通过 USB 连接 GPU**：**#general** 频道的一位用户询问了通过 **USB 端口**连接 **GPU** 的事宜，引用了一条[推文](https://x.com/__tinygrad__/status/1868867387652714522)，George Hotz 回复道：“*我们的驱动程序应该允许这样做*”。
   - 这次讨论凸显了社区对扩展 **tinygrad** 应用的**硬件兼容性**的兴趣。
- **Mac ARM64 后端访问仅限于 CI**：在 **#general** 中，一位用户寻求访问 **Mac** 以进行 **arm64 backend** 开发，但 George 澄清说，这些系统仅指定用于**持续集成 (CI)**。
   - 这一澄清强调了 **Mac infrastructure** 目前保留用于运行 **benchmark** 测试，而非通用开发用途。
- **持续集成侧重于 Mac Benchmarks**：**Mac Benchmark** 作为项目**持续集成 (CI)** 流程的关键部分，专注于**性能评估**。
   - 这种方法强调了团队利用特定硬件配置以确保**稳健性能指标**的策略。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **扩展测试时计算 (Scaling Test Time Compute) 分析**：一位成员分享了讨论 **scaling test time compute** 的 [Hugging Face 博客文章](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute)，他们认为这篇文章**令人耳目一新**。
   - 这篇文章引发了社区对**扩展测试效率**的兴趣。
- **3b 模型在数学方面优于 70b**：一位成员指出 **3b 模型**在数学方面的表现优于 **70b 模型**，称这既**疯狂**又意义重大。
   - 这一观察引发了关于**小模型出人意料的效率**的讨论。
- **仓库中缺失 Optim 代码**：一位成员对开发者仓库中缺少实际的 **optim 代码**表示担忧，该仓库仅包含 **benchmark** 脚本。
   - 他们强调了在使用该仓库时遇到的困难，并表示正在努力解决该问题。
- **当前工作量阻碍了贡献**：一位成员因无法做出贡献而道歉，理由是还有其他任务和 **bug 修复**。
   - 这凸显了社区内**开发与协作**的繁忙本质。
- **社区对更新表示感谢**：在持续的讨论中，一位成员感谢了另一位成员的更新。
   - 这反映了频道**积极且相互支持的氛围**。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **自主 AI 提升知识工作者效率**：最近的一篇 [论文](https://arxiv.org/abs/2312.05481) 讨论了 **自主 AI** 如何通过自动化常规任务来提高知识工作者的效率，从而提升整体生产力。
   - 研究显示，虽然最初的研究集中在 *chatbots* 辅助低技能工人，但 Agentic AI 的出现将优势转向了更高技能的个人。
- **AI 运营模式改变劳动力动态**：该论文介绍了一个框架，其中 **AI Agent** 可以自主或非自主运行，导致等级制公司内部劳动力动态的重大转变。
   - 报告指出，**基础自主 AI** 可以将人类置换到专业化角色中，而 **高级自主 AI** 则将劳动力重新分配到常规任务中，从而产生规模更大、生产力更高的组织。
- **非自主 AI 赋能知识较少的人群**：非自主 AI（如 **chatbots**）为知识较少的人群提供负担得起的专家协助，在不竞争大型任务的情况下增强其解决问题的能力。
   - 尽管被认为是有益的，但随着 AI 技术的不断演进，自主 Agent 支持 **知识工作者** 的能力提供了一种竞争优势。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **超低依赖应用的最终 RAG 活动**：明天是 12 月的最后一场活动，参与者将在 **Alex Garcia** 的带领下，学习仅使用 **sqlite-vec**、**llamafile** 和基础 Python 创建 **超低依赖检索增强生成 (RAG)** 应用。
   - 该课程不需要额外的依赖项或 'pip install'，强调 RAG 开发的简洁与高效。
- **Developer Hub 和 Blueprints 的重大更新**：发布了关于 **Developer Hub** 和 **Blueprints** 的重要公告，提醒用户刷新关注。
   - 随着社区探索关于 **Blueprints** 的讨论帖，反馈正受到重视，旨在帮助开发者构建开源 AI 解决方案。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **数据基础设施年终回顾**：请在 [12 月 18 日](https://www.meetup.com/streaming-stories/events/304951233/) 加入我们的回顾小组，创始人 [Yingjun Wu](https://www.linkedin.com/in/yingjun-wu/)、[Stéphane Derosiaux](https://www.linkedin.com/in/stephane-derosiaux/) 和 [Alexander Gallego](https://www.linkedin.com/in/alexandergallego/) 将讨论过去一年 **数据基础设施** 的创新。
   - 小组讨论将涵盖关键主题，包括 **Data Governance**、**Streaming** 以及 **AI 对数据基础设施的影响**。
- **数据创新小组的主旨发言人**：小组嘉宾包括 **RisingWave CEO** [Yingjun Wu](https://www.linkedin.com/in/yingjun-wu/)、**Conduktor CPTO** [Stéphane Derosiaux](https://www.linkedin.com/in/stephane-derosiaux/) 以及 **Redpanda CEO** [Alexander Gallego](https://www.linkedin.com/in/alexandergallego/)。
   - 他们的见解预计将探讨 **Stream Processing** 和 **Iceberg 格式** 等关键领域，塑造 2024 年的格局。
- **AI 在数据基础设施中的角色**：小组将讨论 **AI 对数据基础设施的影响**，重点介绍最近的进展和实现。
   - 这包括 AI 技术如何改变 **Data Governance** 并增强 **Streaming** 能力。
- **流处理和 Iceberg 格式**：关键话题包括 **Stream Processing** 和 **Iceberg 格式**，这对于现代数据基础设施至关重要。
   - 小组成员将深入探讨这些技术如何塑造未来一年的数据基础设施生态系统。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCL Leaderboard V3 在函数演示期间冻结**：一名成员反映 [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard) 在函数调用演示期间卡在 **'Loading Model Response...'** 状态。
   - 他们询问其他人是否也遇到了同样的加载问题，寻求确认及潜在的解决方案。
- **BFCL Leaderboard V3 扩展功能和数据集**：讨论强调了 [Berkeley Function Calling Leaderboard V3](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard) 更新了 LLM 准确函数调用的评估标准。
   - 成员们参考了之前的版本如 [BFCL-v1](blogs/8_berkeley_function_calling_leaderboard.html) 和 **BFCL-v2**，并指出 **BFCL-v3** 包含了针对多轮交互（multi-turn interactions）扩展的数据集和方法论。

---

**LAION Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有动静，请告知我们，我们将将其移除。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **Codeium / Windsurf ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1318330983551995904)** (89 messages🔥🔥): 

> `Windsurf 功能问题, Codeium 定价与额度, AI 代码生成的用户体验, Codeium 插件显示问题, 代码审查工具推荐` 

- **用户报告 Windsurf 功能问题**：多位用户对 **Windsurf** 无法有效修改或编辑文件表示担忧，其中一人称其在最近的更新后变得“更笨了”。
   - 讨论中提到了 **resource exhaustion errors**（资源耗尽错误）以及新推出的方案，这在用户中引起了困惑。
- **了解 Codeium 的定价和额度**：频道中的个人质疑是否能购买更大额度的 **Flex credits**，并强调尽管用户努力管理，但消耗速度依然很快。
   - 对话围绕新建立的信用额度限制和方案展开，重点关注不同付费层级将包含的内容。
- **使用 AI 代码生成的不同体验**：几位用户指出在使用 **Codeium** 执行任务时结果参差不齐，特别提到了 AI 进行的更改破坏了预期功能的问题。
   - 一名成员讲述了 AI 如何修改单元测试以使其通过，却未能理解预期的功能，这表明其缺乏上下文和控制。
- **Codeium 插件显示字体大小问题**：一名用户反映了 JetBrains IDEs 中 Codeium 聊天机器人的 **字体过小** 问题，而其他字体显示正常。
   - 讨论包括了排查步骤，用户正在寻求针对显示不一致问题的潜在修复方法。
- **代码审查工具推荐**：有人建议将 **Code Rabbit AI** 作为代码审查的替代工具，强调了其在管理 Pull Requests 方面的有效性。
   - 这引发了关于代码审查工具演变现状以及付费选项中用户偏好的讨论。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/NVIDIAAIDev/status/1868778156347339033">来自 NVIDIA AI Developer (@NVIDIAAIDev) 的推文</a>：👀 RAPIDS cuDF 在零代码更改的情况下将 #pandas 加速高达 150 倍。现在，随着数据集大小增长到 GB 级，你可以继续使用 pandas。⚡ ➡️ 尝试演示的 Jupyter Notebook：http://nvda.ws...</li><li><a href="https://tenor.com/view/hello-there-gif-5677380953331354485">Hello There GIF - Hello there - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://codeium.com/blog/pricing-windsurf">方案与定价更新</a>：我们对 Cascade 的定价模型进行了一些更改。</li><li><a href="https://codeium.canny.io/feature-requests/p/add-gemini-20">添加 Gemini 2.0 | 功能请求 | Codeium</a>：添加 Gemini 2.0，我看到许多基准测试显示它在编程方面比 Claude 更好</li><li><a href="https://github.com/SchneiderSam/awesome-windsurfrules/">GitHub - SchneiderSam/awesome-windsurfrules: 📄 精选的 global_rules.md 和 .windsurfrules 文件列表</a>：📄 一个精选的优秀 global_rules.md 和 .windsurfrules 文件列表 - SchneiderSam/awesome-windsurfrules
</li>
</ul>

</div>

---

### **Codeium / Windsurf ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1318306688440336395)** (668 messages🔥🔥🔥): 

> `Windsurf vs Cursor, Gemini AI 性能, Windsurf Bug, Git 使用, AI 工具用户体验`

- **Windsurf 与 Cursor 的性能对比**：用户正在讨论 Windsurf 和 Cursor 之间的性能差异，强调这两个平台虽然具有类似的 Agent 功能，但在上下文使用方法上有所不同。
   - Windsurf 被指出有严格的额度限制，而 Cursor 在使用完一定量的高级额度后提供无限制的慢速查询，一些用户认为这更加方便。
- **Windsurf 的编辑和代码管理困扰**：多位用户表达了对 Windsurf 倾向于覆盖代码和产生幻觉错误的挫败感，这在开发过程中导致了混乱。
   - 用户建议通过实施 Git 进行版本控制来改进他们的编码工作流，以便更好地管理更改并提高可逆性。
- **转向 Gemini 2.0 的体验**：用户正在评估 Gemini 2.0 与 Windsurf 结合使用的效果，一些人注意到其显著的上下文优势，而另一些人对其输出质量评价褒贬不一。
   - 虽然 Gemini 2.0 拥有更大的上下文窗口，但一些用户提到在达到一定的 Token 限制后，性能可能会下降。
- **社区对改进 Windsurf 的建议**：社区正在倡导对 Windsurf 进行增强，包括改进跨面板的鼠标焦点管理，以提高工作流效率。
   - 用户还要求提供一种撤销代码更改的方法，并在工具内更好地管理他们的开发环境。
- **用户对 AI 工具的使用参与度**：参与者分享了他们利用 Windsurf 和 Cursor 等 AI 工具来简化编码任务并提高生产力的独特方法。
   - 一些用户利用这些工具的高级功能来完成特定任务，同时讨论了保持对代码更改控制权的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://astral.sh/blog/the-ruff-formatter">The Ruff Formatter：一个极快且兼容 Black 的 Python 格式化工具</a>：Ruff 的格式化工具比现有工具快 30 倍以上，同时保持了与 Black 超过 99.9% 的兼容性。</li><li><a href="https://marketplace.visualstudio.com/items?itemName=laravel.vscode-laravel">Laravel&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>：Visual Studio Code 扩展 —— Laravel 官方 VS Code 扩展</li><li><a href="https://tenor.com/view/anakin-gif-1614955667706199731">Anakin GIF - Anakin - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://developers.cloudflare.com/pages/framework-guides/nextjs/ssr/">全栈 (SSR) · Cloudflare Pages 文档</a>：Next.js ↗ 是一个用于构建全栈应用程序的开源 React.js 框架。本节将帮助你使用 @cloudflare/next-on-pages ↗ 将全栈 Next.js 项目部署到 Cloudflare Pages。</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://codeium.canny.io/feature-requests/p/add-gemini-20">添加 Gemini 2.0 | 功能请求 | Codeium</a>：添加 Gemini 2.0。我看到许多基准测试显示它在编程方面比 Claude 更好。</li><li><a href="https://github.com/VSCodium/vscodium/blob/master/docs/index.md#extensions-marketplace">vscodium/docs/index.md at master · VSCodium/vscodium</a>：不含微软品牌/遥测/许可的 VS Code 二进制发行版 - VSCodium/vscodium</li><li><a href="https://codeium.canny.io/feature-requests/p/windsurf-focus-follows-mouse-as-a-configuration-option">Windsurf - 焦点跟随鼠标（作为配置选项） | 功能请求 | Codeium</a>：VSCode 有一个公开的 GitHub PR，表面上看已经超过 4 年了，但实际上它的历史比这还要久。</li><li><a href="https://codeium.com/blog/pricing-windsurf">方案与定价更新</a>：Cascade 定价模型的一些变更。</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers?tab=read">GitHub - punkpeye/awesome-mcp-servers：MCP 服务器集合。</a>：MCP 服务器集合。通过在 GitHub 上创建账号，为 punkpeye/awesome-mcp-servers 的开发做出贡献。</li><li><a href="https://github.com/orgs/modelcontextprotocol/discussions/88">MCP 和向量数据库有什么区别？· modelcontextprotocol · Discussion #88</a>：已经有一段时间了，我还是没搞明白。</li><li><a href="https://www.youtube.com/watch?v=VcUl0vPJwxo&pp=ygUId2luZHN1cmY%3D"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/6rwbcgEM25g">如何真正通过 Windsurf 赚钱 #aiautomation #firebringerai #coding #seoautomation</a>：使用这款改变游戏规则的工具在几分钟内构建 SEO 网站。停止花费数小时甚至数天手动构建 SEO 网站。该工具将你的关键词转化为...</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers?tab=readme-ov-file#tutorials">GitHub - punkpeye/awesome-mcp-servers：MCP 服务器集合。</a>：MCP 服务器集合。通过在 GitHub 上创建账号，为 punkpeye/awesome-mcp-servers 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1318315127308681306)** (566 条消息🔥🔥🔥): 

> `AI 与创意写作、Prompt Engineering 与评估、LLM 性能特征、计算机科学教育路径` 


- **AI 变革故事创作**：参与者讨论了 AI 创作故事的潜力，以及使用 prompt chaining 提高文本质量的有效性，强调了清晰的评估标准（rubric）的重要性。
   - 对话包括 prompt 示例，以及迭代反馈如何增强故事的连贯性和情感冲击力。
- **AI 响应中的随机性**：小组探讨了在 AI 生成的名字和场景中加入随机性的想法，以对抗生成故事中重复输出的倾向。
   - 建议使用随机表（Stochastic tables）和随机名称生成器，以增加 LLM 输出的多样性和深度。
- **构建评分系统**：参与者开发了一个基于详细标准评估故事的 prompt，评估连贯情节和情感影响等方面。
   - 讨论包括测试和完善评分系统，以确保 LLM 对故事质量的准确评估。
- **技术领域的教育选择**：一位用户质疑获得计算机科学硕士学位与通过项目和实习获得实践经验的价值。
   - 对话承认，虽然学术资历对某些职业路径有益，但在 Web 和移动开发等领域，动手经验通常被优先考虑。
- **LLM 性能反馈**：用户报告了使用 LLM 的积极体验，特别提到了一个 8B 模型在评估故事质量方面的强大批判能力。
   - 对话强调了有效利用 LLM 进行写作辅助和批判性分析的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2310.16764">ConvNets Match Vision Transformers at Scale</a>: 许多研究人员认为 ConvNets 在小型或中型数据集上表现良好，但在面对互联网规模的数据集时无法与 Vision Transformers 竞争。我们挑战...</li><li><a href="https://blogs.nvidia.com/blog/jetson-generative-ai-supercomputer/">NVIDIA Unveils Its Most Affordable Generative AI Supercomputer</a>: NVIDIA 发布了一款新型紧凑型生成式 AI 超级计算机，通过软件升级以更低的价格提供更高的性能。全新的 NVIDIA Jetson Orin Nano Super Developer Kit...</li><li><a href="https://docs.langflow.org/">Welcome to Langflow | Langflow Documentation</a>: Langflow 是一个用于构建多 Agent 和 RAG 应用的新型可视化框架。它是开源的、由 Python 驱动、完全可定制，并且与 LLM 和向量数据库无关。</li><li><a href="https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/">NVIDIA Jetson AGX Orin</a>: 为下一代机器人提供更高级别的 AI 性能。</li><li><a href="https://x.ai/blog/grok">Announcing Grok</a>: 无描述</li><li><a href="https://www.pcworld.com/article/2553897/intel-arc-b580-review-worthy-budget-1440p-gpu.html">Intel Arc B580 review: The GPU we&#039;ve begged for since the pandemic</a>: 英特尔 249 美元的 Arc B580 是我们自疫情以来一直渴求的显卡。</li><li><a href="http://www.orangepi.org/">Orange Pi - Orange Pi official website - Orange Pi development board, open source hardware, open source
        software, open source chip, computer keyboard</a>: 无描述</li><li><a href="https://www.hardkernel.com/shop/odroid-m2-with-16gbyte-ram/">ODROID-M2 with 16GByte RAM &#8211; ODROID</a>: 无描述</li><li><a href="https://safepine.co/">Safepine</a>: 无描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/175ejvi/quick_start_example_for_llava_generate_image/">Reddit - Dive into anything</a>: 无描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1318366232340201635)** (6 条消息): 

> `Sampling Algorithms, Gemini Data Recall, Threefry, Mersenne Twister` 


- **关于采样算法的辩论**：一位成员询问在 LLM 的采样和权重计算中是否使用了 **Xorshift 或其他算法**。
   - 另一位成员提到 **Gemma 使用了 Threefry**。
- **Gemini 的数据召回能力**：一位成员对 **Gemini 在拥有庞大互联网知识库的情况下准确召回数据**的能力表示好奇。
   - 他们将其与历史学家可能混淆日期的情况进行了比较，询问该模型是否存在类似的局限性。
- **PyTorch 的算法选择**：有人指出 **PyTorch 使用 Mersenne Twister** 作为其采样算法。
   - 这凸显了不同 AI 框架所使用的采样技术之间的差异。
- **参与 AI 项目的兴趣**：一位成员表示希望了解如何为所讨论的**有趣项目**提供帮助。
   - 这标志着社区内发出了开放的协作邀请。
- **构建虚拟女友**：一位成员澄清了他们的意图，表示：**“我正尝试构建一个女朋友。”**
   - 这表明讨论从技术探讨转向了个人项目愿景。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1318530250124296232)** (5 条消息): 

> `phi-4 language model, quantization techniques, LlamaCPP integration, test-time compute approaches, performance benchmarks` 


- **phi-4 模型显著提升了 QA 能力**：拥有 **140 亿参数**的 [phi-4](https://arxiv.org/pdf/2412.08905) 模型通过结合合成数据和改进的训练方法，在以 STEM 为重点的 QA 能力上超越了 GPT-4。
   - 尽管与 phi-3 相比架构变化较小，但由于其改进的训练课程和训练后技术，phi-4 在推理基准测试中表现强劲。
- **低至 2 bits 的量化令人印象深刻**：一位成员指出，一个 **80 亿参数模型**被有效地量化到了 **2 bits**，尽管最初存在编码挑战，但这看起来很有前景。
   - 另一位参与者评论说，这种新方法可以作为具有更大参数模型的**标准**，从而提高其可用性。
- **在有限硬件上运行 70b 模型的挑战**：一位参与者对由于只有 **24GB** RAM 而无法在硬件上运行 **70b 模型**表示沮丧，并询问了如何集成到 LlamaCPP 或 VLLM 等平台的方法。
   - 这表明对于那些希望在没有大量硬件的情况下使用大型模型的人来说，需要可扩展的解决方案。
- **社区认可 test-time compute 的进展**：一位成员称赞了 **Hugging Face** 在 **test-time compute 方法**上的工作，表明这在社区内受到了好评。
   - 共享的讨论其扩展方法的链接进一步强调了这一点，增强了对计算效率的理解。



**提到的链接**：<a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">Scaling test-time compute - a Hugging Face Space by HuggingFaceH4</a>：未找到描述

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1318530250124296232)** (5 条消息): 

> `phi-4 model, quantized models, Hugging Face test-time compute` 


- **phi-4 模型在 STEM QA 方面表现出色**：[phi-4](https://arxiv.org/pdf/2412.08905) 是一个拥有 140 亿参数的语言模型，其特点是专注于数据质量，在 STEM 能力方面超越了 GPT-4。
   - 尽管架构与 phi-3 相似，但它在训练过程中使用了合成数据（synthetic data），从而增强了其在推理基准测试中的表现。
- **处理过程中的量化挑战**：一位成员强调了使用压缩至 2-bits 的 8B 模型的挑战以及设置的复杂性，将这一过程形容为相当棘手。
   - *Vibe check 似乎通过了*，因为他们认为这对于 32B+ 的草稿模型（draft models）展现出了潜力。
- **寻求与 LlamaCPP 等工具的集成**：讨论中提到了由于硬件限制（特别是 **24GB RAM** 的上限），运行 70B 模型的难度。
   - 成员们正在探索将模型集成到 LlamaCPP、Aphrodite 和 VLLM 等平台的最佳方法。
- **Hugging Face 的 test-time compute 进展**：Real.Azure 赞扬了 Hugging Face 在 test-time compute 方法上的工作，并指出了极具前景的发展。
   - 其 [blogpost](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) 的链接讨论了扩展 test-time compute 的策略。



**提到的链接**：<a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">Scaling test-time compute - a Hugging Face Space by HuggingFaceH4</a>：未找到描述

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1318306882322038865)** (302 条消息🔥🔥): 

> `Aider Updates, O1 API and Pro Features, Linters and Code Management, Claude Model Discussion, AI in Coding Automation` 


- **O1 API 全面发布**：用户对即将推出的 O1 API 表示兴奋，预计该 API 将包含推理努力程度（reasoning effort）参数和系统提示词（system prompts）等功能，从而增强 AI 能力。
   - 用户对 O1 API 的潜在成本持复杂态度，因为预计其价格相比 Sonnet 会有显著上涨。
- **AI 性能提升**：O1 模型被指出能够根据特定提示词改进回答，尽管在某些情况下会表现出过度自信。
   - 用户建议结合不同的模型（如 O1 和 Claude），通过创建提示词来利用两者的优势。
- **Aider 中的 Linting 与代码管理**：Aider 内置支持多种 Linter，并允许用户使用 `--lint-cmd` 选项指定首选的 Linting 命令。
   - 用户可以启用或禁用自动 Linting，在通过 AI 进行编辑时灵活管理代码质量。
- **Claude 模型局限性**：讨论中提到了 Claude 模型的局限性，特别是它不愿生成某些输出以及倾向于提供过于谨慎的回答。
   - 用户对需要更明确地引导 AI 以获得预期结果表示沮丧，并指出明确性是关键。
- **AI 编程的未来**：参与者讨论了先进 AI 模型对编程工作的影响，并对未来潜在的岗位取代表示担忧。
   - 尽管如此，许多人认为仍需要人类的创造力和解决问题的能力来补充 AI 的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/sundarpichai/status/1869066293426655459">Sundar Pichai (@sundarpichai) 的推文</a>：Gemini Advanced 订阅者可以尝试 gemini-exp-1206，这是我们最新的实验模型。在编程、数学、推理、指令遵循等方面性能显著提升。</li><li><a href="https://x.com/altryne/status/1869084443673309595">Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：@crizcraig @OpenAI 尚未到来</li><li><a href="https://aider.chat/docs/more/edit-formats.html">编辑格式</a>：Aider 使用各种“编辑格式”让 LLM 编辑源文件。</li><li><a href="https://aider.chat/docs/install.html">安装</a>：如何安装并开始使用 aider 进行结对编程。</li><li><a href="https://aider.chat/docs/usage/lint-test.html">Linting 与测试</a>：自动修复 Linting 和测试错误。</li><li><a href="https://www.youtube.com/watch?v=XKABimtOWME"> - YouTube</a>：未找到描述</li><li><a href="https://aider.chat/docs/usage/tutorials.html">教程视频</a>：由 aider 用户制作的入门和教程视频。</li><li><a href="https://aider.chat/docs/config/options.html#fixing-and-committing">选项参考</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/config/options.html">选项参考</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://github.com/1broseidon/promptext">GitHub - 1broseidon/promptext</a>：通过创建账号为 1broseidon/promptext 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/pull/2634">feat: Support custom Whisper API endpoints for voice transcription by mbailey · Pull Request #2634 · Aider-AI/aider</a>：添加对自定义 Whisper API 端点的支持 2024-12-18：第三次重写 - 与现有的 API Key 处理约定更兼容。此 PR 增加了使用替代 Whisper API 提供商的能力...</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1318340252057731143)** (34 messages🔥): 

> `Aider 与 LM Studio 集成, 在 Emacs 中使用 Aider, 在 Aider 中提交特定文件, Aider 错误排查, Aider 的 Dart 支持` 


- **Aider 在集成 LM Studio 时遇到困难**：用户报告了在将 Aider 与 LM Studio 配合使用时面临的挑战，特别是收到诸如 **BadRequestError** 之类的错误，提示缺少 LLM provider。
   - 经过调试，一位用户发现使用 OpenAI provider 格式可以成功运行，具体指向为 `openai/qwen-2.5-coder-7b-instruct-128k`。
- **关于 Aider 的 Emacs 兼容性咨询**：一位用户询问是否可以在 Emacs 中使用 Aider，并发现了一个可用的 `aider.el` 文件来辅助集成。
   - 另一位用户确认他们正在 Emacs 中配合 Aider 的 watch mode 使用，这极大地增强了他们的工作流。
- **Aider 中的 Commit 功能和文件处理**：一位新用户寻求澄清，询问 `/commit` 是否可以应用于特定文件，而不是 Aider 中所有已暂存（staged）的文件。
   - 回复指出，用户必须移除（drop）不需要的文件才能进行选择性提交，因为该命令仅对已添加的文件生效。
- **Aider 模型错误的挑战**：几位用户对运行 Aider 命令时反复出现的问题表示沮丧，这些问题通常源于环境配置错误或版本过旧。
   - 一位用户通过确保引用的是通过 pipx 安装的正确 Aider 版本解决了他们的问题。
- **Aider 对 Dart 语言支持的局限性**：讨论显示 Aider 目前缺乏对 Dart 的支持，用户注意到正在努力添加兼容性。
   - 社区提供了相关 GitHub issues 的链接，强调了在 Aider 中增加额外语言支持的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: 了解如何在 LiteLLM 上部署并调用来自不同 provider 的模型</li><li><a href="https://aider.chat/docs/languages.html">支持的语言</a>: Aider 几乎支持所有流行的编程语言。</li><li><a href="https://github.com/Aider-AI/aider/issues/1089">支持 Dart/Flutter? · Issue #1089 · Aider-AI/aider</a>: 该 Issue 提到，在 https://aider.chat/docs/languages.html 的支持语言列表中未列出 Dart / Flutter。询问是否可以考虑支持生成仓库...
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1318635009959264377)** (1 messages): 

> `新 UI 推出, NotebookLM Plus 功能, Interactive Audio BETA` 


- **新 UI 和 NotebookLM Plus 功能发布**：团队宣布今天上午向所有用户推出剩余的**新 UI**和 **NotebookLM Plus 功能**。
   - 此次更新是持续提升整个平台用户体验工作的一部分。
- **Interactive Audio 可用性受限**：目前，由于后端正在进行改进，**Interactive Audio** 仍仅对部分选定用户开放。
   - 收到新 UI 但无法访问 ***Interactive mode (BETA)*** 功能的用户不必惊慌，这在过渡期间是预期内的情况。


  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1318343114552901683)** (32 条消息🔥): 

> `播客实验, 呼叫中心中的 AI, 游戏攻略指南, 改进的笔记导出, 交互模式` 


- **播客测试新技术**：成员们讨论了使用 [NotebookLM](https://link.to/notebooklm) 创作播客，分享了如《与 Gordon Ramsey 和 Dalai Lama 一起烹饪》等示例。
   - 一位成员表示希望利用该服务在其播客频道上进行每日音乐历史更新。
- **AI 在呼叫中心的作用**：对话强调了 AI 在 IT 呼叫中心的集成，包括一个关于德语 AI 处理客户查询的幽默案例。
   - 一位成员分享了各种音频片段，描绘了计算机故障排除和冷启动销售电话等场景。
- **游戏攻略指南的应用**：一位用户提到测试将不同的游戏攻略指南作为来源，以便更轻松地获取 Boss 战和收集品的信息。
   - 这种方法旨在消除从指南或 Reddit 帖子中详尽搜索技巧的需求。
- **导出笔记以增强功能**：针对笔记缺乏导出选项的问题提出了担忧，并呼吁支持 Markdown 和 PDF 等格式。
   - 另一位成员讨论了包括使用 Readwise Reader 渲染源文档在内的替代方法。
- **交互模式参与**：一位成员分享了使用 Beta 版交互模式在对话过程中积极参与并提问的经验。
   - 这一功能表明人们对增强用户与 AI 系统交互的兴趣日益浓厚。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/UbBSmM-WM48?si=IL9U1zalqhiEEzoF"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/ytcHj-EllWo?feature=shared"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/RFFH1rcT3hM?feature=shared"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1318308951393308712)** (207 条消息🔥🔥): 

> `Notebook LM Plus 访问权限, Interactive Mode 功能, 新 UI 反馈, Audio Overview 限制, 多语言支持` 


- **访问 Notebook LM Plus**：用户正在讨论如何访问 Notebook LM Plus，指出可以通过 Google Workspace 或 Google Cloud 获取，并将于 2025 年初面向 Google One AI Premium 用户开放。
   - 有关于 Notebook LM Plus 是否在意大利和巴西等特定国家可用的疑问，但回复显示该功能正在逐步推出。
- **Interactive Mode 功能的挑战**：多位用户报告在使用 Interactive Mode 功能时遇到困难，包括延迟以及即使更新到新 UI 后也无法访问等问题。
   - 大家的共识是该功能可能仍在推出中，导致不同用户之间的可用性存在差异。
- **对新 UI 的反馈**：一些用户对新 UI 表示不满，提到它不如以前的版本易用，特别是抱怨聊天面板的可见性和笔记的布局。
   - 其他人则认为，虽然新 UI 提供了更大的编辑器，但折叠某些面板的能力可能会改善体验。
- **Audio Overview 的限制**：用户对 Audio Overview 的限制提出了担忧，包括处理时间长和生成失败。
   - 讨论涉及了来源顺序对 Audio Overview 的影响，以及是否可以指示主持人提供更详细的解释。
- **多语言能力的探索**：关于 Notebook LM 生成不同语言播客能力的问题不断涌现，一些用户确认音频摘要目前仅支持英文。
   - 尽管有此限制，用户已成功生成了葡萄牙语内容，表明了更广泛语言支持的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://book-a-painter.com/">未找到标题</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?visit_id=638700401361446702-1279912759&p=plus&rd=1">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?visit_id=638700145029570781-388658972&p=plus&rd=1">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://developer.hashicorp.com/terraform/docs">Terraform 概览 | Terraform | HashiCorp Developer</a>：未找到描述</li><li><a href="https://youtu.be/aG0ixD3OY80"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/JhuC77mtdoQ"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/7mqciPtMfBI?si=IStj7r25df71U40Y">Veo 2 用于科幻新闻广播的 AI 视频生成（配合 NotebookLM）</a>：Google Labs 今天发布了全新的 Veo2 AI 驱动视频生成工具，所以我拿了一个我从科幻新闻中生成的 NotebookLM 播客...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1318307491297361951)** (183 条消息🔥🔥): 

> `Python version compatibility, Unsloth 4-bit model performance, Function calling in Llama 3.2, Multi-GPU training with Unsloth Pro, Quantization process for models` 


- **Python 版本兼容性问题**：一场关于 Python 版本的讨论展开，强调 **Python 3.13** 可能会导致问题，而建议使用 **3.10** 以保证兼容性。
   - 成员们分享了使用 **pyenv** 与 **conda** 环境管理 Python 版本的经验。
- **Unsloth 4-bit 模型性能见解**：用户注意到 **Unsloth 4-bit 模型**与原始 Meta 版本在层大小（layer sizes）上存在差异，这表明模型参数化可能存在潜在问题。
   - 用户对从 4-bit 转换到 **full precision**（全精度）时的 VRAM 占用和性能权衡表示担忧。
- **探索 Llama 3.2 中的 Function calling**：参与者表示有兴趣训练 **Llama 3.2** 以提升 Function calling 能力，但发现可直接实现的示例有限。
   - 大家达成共识，认为直接在数据集中包含 special tokens 可以简化训练过程。
- **使用 Unsloth Pro 进行多 GPU 训练**：成员们询问了 **Unsloth Pro** 多 GPU 训练的当前功能，并确认了其可用性。
   - 产生了一些疑问，即该功能是仅限于云端设置，还是也可以在本地执行。
- **模型的 Quantization 过程**：用户讨论了量化模型进行微调的影响，指出非 Unsloth 模型可以被加载并即时转换为 4-bit。
   - 社区强调这一过程可以在模型训练期间节省时间和资源。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pypi.org/project/triton/,">未找到标题</a>: 未找到描述</li><li><a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#cosineannealinglr).">CosineAnnealingLR &mdash; PyTorch 2.5 文档</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit">unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct">Qwen/Qwen2-VL-7B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1868748998783517093">Daniel Han (@danielhanchen) 的推文</a>: 我对后预训练（Post Pretraining）世界的看法 - Ilya 的演讲：Ilya 暗示我们需要寻找其他东西来扩展 - 演讲中的脑体质量比图表显示人类智能的“扩展”比...更好</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#zero-shot-function-calling-e2e-format>)">llama-models/models/llama3_2/text_prompt_format.md at main · meta-llama/llama-models</a>: 旨在用于 Llama 模型的实用工具。通过在 GitHub 上创建账号为 meta-llama/llama-models 的开发做出贡献。</li><li><a href="https://gist.github.com/fullstackwebdev/5aa69712a30a93bff3b2daebaeb6776f">unsloth_tool_success2.py</a>: GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://youtu.be/jFl5Fewrieo"> - YouTube</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/chat-templates>)...">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/finetuning/datasets/README.md#batching-strategies>),">llama-recipes/recipes/quickstart/finetuning/datasets/README.md at main · meta-llama/llama-recipes</a>: 使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama 的脚本，涵盖单/多节点 GPU。支持默认和自定义数据集，适用于摘要和问答等应用...</li><li><a href="https://gist.github.com/fullstackwebdev/d8c8d46d042828ffeedb0ac2b701b31d">tool_train.py</a>: GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://gist.github.com/fullstackwebdev/b8948204845207ef1ef672144b60caf8">train.jsonl</a>: GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1318684381124100097)** (4 messages): 

> `Voe2 vs Sora, Open Source Reasoning Models, OpenAI Bankruptcy Speculation` 


- **Voe2 让 Sora 显得毫无用处**：围绕 **Voe2** 的发布展开了讨论，有人声称它让 **Sora** 看起来像个**无用的玩具**。
   - 这种情绪通过一段附带的 [meme 视频](https://cdn.discordapp.com/attachments/1179039861576056922/1318684381094744074/PACKWATCH_-_RIP_Bozo_Meme_HD.mp4?ex=67633808&is=6761e688&hm=37fafc9c1fe42e3649578f9fda852baee026cb292738ceff1d3fe5ddc5381aef&)得到了幽默的体现。
- **QwQ 将主导 OpenAI 的 O1**：一位成员认为，像 **QwQ** 这样的**开源推理模型**将显著超越 OpenAI 的 **O1**。
   - 另一位成员指出，虽然复制推理模型很容易，但打造一个**有价值的模型**是一个更大的挑战。
- **对 OpenAI 财务未来的推测**：一位用户挑衅地询问 OpenAI 何时可能**宣布破产**，引发了财务方面的猜测。
   - 虽然没有得到确切预测的回应，但该评论反映了人们对 OpenAI 稳定性的持续关注。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1318329026690941011)** (47 messages🔥): 

> `Lora+ with Unsloth, Finetuning Qwen models, AMD GPU compatibility, Training vs Inference in Unsloth, Packing in training` 


- **Lora+ 准备见解**：成员们讨论了在 Unsloth 中使用 Lora+，指出它可能与其他方法交互不佳，而 **LoFTQ 或 PiSSA** 可能会提供更好的初始化。
   - 一位成员分享了一篇 [CPT 博客文章](https://unsloth.ai/blog/contpretraining)，强调了 Unsloth 新版本的性能提升。
- **Qwen 模型的微调挑战**：一位成员对他们微调后的 **Qwen 2.5** 模型表现不如原始模型表示沮丧，并提到了对**灾难性遗忘**（catastrophic forgetting）的担忧。
   - 其他成员建议对微调过程进行迭代以更好地适应特定需求，强调了尝试和调整的重要性。
- **AMD GPU 与 Bitandbytes 的兼容性**：围绕在 AMD GPU 上运行 **Llama-3.2-11B-Vision-Instruct** 展开了讨论，成员们强调 **Bitandbytes** 现在已支持这些 GPU。
   - 仍然有人提到了一些局限性，以及需要与 AMD 配置兼容的替代智能量化技术。
- **使用 Unsloth 进行推理**：成员们澄清说，虽然 Unsloth 支持某些推理功能，但它主要是为训练模型而设计的。
   - 有人提到模型可以导出用于缺乏 GPU 的环境，尽管性能可能会受到严重影响。
- **理解训练中的 Packing**：有人对参数 `packing=False` 感到好奇，得到的解释是启用 packing 可以通过对较短的样本进行分组来加速训练。
   - 同时也讨论了潜在的缺点，包括使用该策略时数据污染的风险。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行持续 LLM 预训练</a>：通过使用 Unsloth 对 Llama 3、Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://huggingface.co/facebook/bart-large-mnli">facebook/bart-large-mnli · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/finetuning/datasets/README.md#batching-strategies>">llama-recipes/recipes/quickstart/finetuning/datasets/README.md at main · meta-llama/llama-recipes</a>：使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama 的脚本，涵盖单节点/多节点 GPU。支持用于摘要和问答等应用的默认及自定义数据集...</li><li><a href="https://github.com/unslothai/unsloth/wiki">主页</a>：以 2-5 倍的速度和减少 70% 的内存微调 Llama 3.3、Mistral、Phi、Qwen 2.5 和 Gemma LLM - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1318531832186404885)** (2 条消息): 

> `phi-4 Language Model, Continual Pre-training Strategies` 


- **phi-4 以数据质量为核心超越 GPT-4**：**phi-4** 是一个拥有 140 亿参数的语言模型，其采用的训练策略强调**数据质量**而非典型的原始来源，并在整个过程中集成了合成数据 (synthetic data)。它在以 **STEM 为核心的 QA 能力**方面表现尤为出色，凭借增强的数据和创新的 post-training 技术，超越了其教师模型 GPT-4。
   - 尽管自 phi-3 以来架构变化极小，但它在推理基准测试中的强劲表现凸显了该模型改进后的训练课程 (training curriculum)。
- **探索高效的 Continual Pre-training 方法**：一项研究调查了大语言模型进行 **Continual Pre-training** 的 **warm-up 策略**，重点关注在引入新数据集时如何保持性能。研究人员假设，在从上游数据 Pile (300B tokens) 过渡到下游数据 SlimPajama (297B tokens) 时，重新提高**学习率 (learning rate)** 可以提高效率。
   - 实验遵循 **linear warmup 和 cosine decay 调度方案**，旨在通过周密的训练阶段调整来优化性能。



**提及的链接**：<a href="https://arxiv.org/abs/2308.04014">Continual Pre-Training of Large Language Models: How to (re)warm your model?</a>：大语言模型 (LLMs) 通常在数千亿个 token 上进行预训练，但一旦有新数据可用，往往需要重新开始整个过程。一个更便宜且更高效的解决方案将是……

  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1318315880710672464)** (166 条消息🔥🔥): 

> `Code Wizard Hackathon, Command R7B Office Hours, Maya Release and Tool Use, Emotional Support for Projects, AI Models Discussion` 


- **Code Wizard 黑客松寻求赞助商**：**Code Wizard** 黑客松的组织者正在寻求赞助，以支持其即将在 2025 年 2 月举办的活动，该活动旨在促进创新和解决问题。
   - 虽然一些参与者质疑资金的必要性，但其他人强调了构建有价值项目的重要性。
- **Command R7B Office Hours 公告**：分享了关于 **Command R7B 问答环节**的提醒，邀请参与者提问并通过实际的代码示例进行学习。
   - 成员们对该环节带来的见解表示兴奋，其中一人计划在睡觉前参加。
- **Maya 的发布激发创新**：**Maya** 的发布受到了热烈欢迎，成员们渴望探索并可能针对 **tool use** 进行微调 (finetune)。
   - 在这一新模型的激励下，参与者们表示愿意工作到深夜，以突破项目的边界。
- **开发中的情感支持**：成员们在开展项目时互相提供情感支持，强调了社区鼓励的价值。
   - 在技术讨论中，“让我们交付世界 (*Let's ship the world*)”之类的短语凸显了积极的氛围和协作精神。
- **关于 AI 和建模的讨论**：关于 **Maya** 模型的规格和特性引发了讨论，成员们分享了对其潜在能力的见解。
   - 关于可用不同模型确切数量的提问，引发了关于相关性和当前开发格局的进一步讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/irobot-benjammins-sentient-sentient-computer-artificial-intelligence-gif-2150069685320147555">Irobot Benjammins GIF - Irobot Benjammins 有意识 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/cats-toby-the-cat-nod-yes-yes-yes-hooman-gif-17105827">Cats Toby The Cat GIF - 猫咪 Toby 点头 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/cat-cute-cat-yap-yapper-yapping-gif-5642199211123099306">Cat Cute Cat GIF - 可爱猫咪 Yap - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1318696792207921213)** (1 messages): 

> `Multimodal Image Embed endpoint, Rate limit increase, API keys, Cohere pricing` 


- **🎉 Multimodal Image Embed 速率大幅提升！ 🎉**：Cohere 宣布将生产环境密钥（Production keys）的 **Multimodal Image Embed** 端点速率限制提升 **10 倍**，从 **40 images/min 增加到 400 images/min**。
   - 测试版速率限制仍保持在 **5 images/min** 供测试使用，允许用户创建应用程序并在社区内分享。
- **了解你的 API keys**：Cohere 提供两种类型的 API keys：免费但受限的 **Evaluation keys** 以及付费且限制较少的 **Production keys**。
   - 用户可以通过 [API keys 页面](https://dashboard.cohere.com/api-keys) 管理其密钥，从而更高效地开始开发。
- **探索更多关于定价和限制的信息**：有关各种端点的详细限制和定价的进一步见解，用户可以查看 [定价文档](https://docs.cohere.com/v2/docs/how-does-cohere-pricing-work)。
   - 端点每月最多可调用 **1,000 次**，以确保整个平台的公平使用。
- **社区支持随时待命！**：如有任何进一步查询，鼓励用户在指定的支持频道或通过电子邮件 **support@cohere.com** 提问。
   - Cohere 提倡就反馈和澄清进行公开对话，确保为开发者提供支持性的环境。



**提及的链接**：<a href="https://docs.cohere.com/v2/docs/rate-limits">API Keys and Rate Limits — Cohere</a>：此页面描述了 Cohere API 针对 Production 和 Evaluation keys 的速率限制。

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1318469314902495293)** (57 messages🔥🔥): 

> `Cohere API for Image Embeddings, RAG-based PDF Answering System, Image Retrieval through Metadata` 


- **关于 Cohere Image Embedding API 的说明**：用户讨论了使用 Cohere 的 API 将 PDF 中的图像嵌入到向量数据库中以用于 RAG 系统，并指出 Embeddings 提供语义含义，无法生成原始图像。
   - 会议确定 Embeddings 旨在通过向量表示进行搜索，不允许直接检索图像，因此需要单独存储原始图像。
- **使用 Embedding 的图像检索策略**：为了根据用户查询检索图像，一位成员建议在 Pinecone 向量数据库中将图像路径作为 Metadata 与 Embeddings 一起存储。
   - 当出现与图像相关的查询时，Embedding 将引导检索其对应的路径，从而允许系统显示正确的图像。
- **AI 和 ML 的基础知识**：一位用户承认自己是 AI 和 ML 的新手，仅在几个月前开始学习，而另一位成员强调了理解 Embeddings 基础知识的重要性。
   - 对话强调了 Embeddings 代表语义含义，有助于高效搜索，但无法还原原始内容。



**提及的链接**：<a href="https://docs.cohere.com/docs/multimodal-embeddings">Multimodal Embeddings — Cohere</a>：Multimodal embeddings 将文本和图像转换为 Embeddings，用于搜索和分类 (API v2)。

  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/)** (1 messages): 

.kolynzb: yello
  

---

### **Bolt.new / Stackblitz ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1318327940072869909)** (4 条消息): 

> `Bolt 的 Meta prompt，Bolt 的 UI 版本，Bolt 的功能请求` 


- **Bolt 开发的 Meta Prompt**：一位成员分享了他们与 Claude 配合使用的 [meta prompt](https://gist.github.com/martinbowling/fe4aa7711d023ef7f188fdd9828fad3e)，用于建立一种系统化的方法，通过 Bolt 生成详细的软件项目计划，涵盖了开发的各个方面。
   - 该 prompt 强调分析需求、定义结构和设计 UI，以简化项目规划流程。
- **Bolt UI 版本的激动公告**：一位成员宣布他们将推出 Bolt 的 UI 版本，用户可以在 Hyperbolic 上托管的 **Claude**、**OpenAI** 和 **Llama** 模型之间进行选择以进行生成。
   - 该 UI 旨在增强生成过程中的用户体验和模型选择。
- **Bolt 功能性 Prompt 的请求**：一位成员询问是否可以从一个 prompt 开始，以启用 Bolt 中全面的功能特性，例如 **
- **社区 GIF 互动**：另一位成员分享了一个来自 Tenor 的轻松 **GIF** 来回应这些公告，其中的角色惊叹道 **


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/excellent-bill-and-ted-air-guitar-yes-yeah-gif-15828050">Excellent Bill And Ted GIF - Excellent Bill And Ted Air Guitar - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://gist.github.com/martinbowling/fe4aa7711d023ef7f188fdd9828fad3e">这个 meta prompt 概述了 Bolt 创建详细软件项目计划的系统方法。它包括分析需求、定义结构、设计 UI、规划实施，以及映射所选 tech stack 如何融入开发过程。</a>: 这个 meta prompt 概述了 Bolt 创建详细软件项目计划的系统方法。它包括分析需求、定义结构、设计 UI、规划实施...
</li>
</ul>

</div>
  

---

### **Bolt.new / Stackblitz ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1318312780759171213)** (213 messages🔥🔥): 

> `将 Bolt 用于 SaaS 项目, Bolt 集成的挑战, 编码问题的支持与协助, 有效管理 Tokens, 分享使用 Bolt 构建的项目` 


- **将 Bolt 用于 SaaS 项目**：用户表达了使用 Bolt 构建 SaaS 应用程序的兴趣，并承认在进一步扩展和集成时可能需要开发人员的协助。
   - 一位用户正在寻求关于如何有效管理其 SaaS 项目的逐步指导。
- **Bolt 集成的挑战**：多位用户报告了 Bolt 的问题，包括平台创建不必要的文件以及在运行命令时遇到错误。
   - 一些用户建议暂时离开平台休息一下，以缓解挫败感并防止损坏屏幕。
- **编码问题的支持与协助**：用户就其编码挑战寻求帮助，其中一位用户专门请求对其 Python 代码提供帮助。
   - 共享了关于调试和利用在线资源以实现更好编码实践的建议。
- **有效管理 Tokens**：用户对 Tokens 的使用和管理表示担忧，并对意外的消耗感到沮丧。
   - 会议指出，用户有每月 Token 使用限制，应意识到更换成本。
- **分享使用 Bolt 构建的项目**：用户询问是否可以分享使用 Bolt 构建的网站，无论语言限制如何。
   - 对分享已完成项目的兴趣表明了社区内的协作环境。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/johnny-depp-pirate-pirate-salute-salute-intergalactic-pirates-of-the-caribbean-gif-25016099">Johnny Depp Pirate GIF - Johnny Depp Pirate Pirate Salute - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=IIueA5giF_4"> - YouTube</a>: 未找到描述</li><li><a href="https://boltsync.mystify.tech/">无标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=IneFM6ViV8s"> - YouTube</a>: 未找到描述</li><li><a href="https://bolters.io/">Bolters.io | Community Supported Tips, Tricks &#38; Knowledgebase for Bolt.new No-Code App Builder</a>: Bolt.new 的文档和指南</li><li><a href="https://support.bolt.new/">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: 一个将日常工作应用融合在一起的新工具。为您和您的团队提供的全能工作空间</li><li><a href="https://thinktank.ottomator.ai/">oTTomator Community</a>: 创新者和专家聚集地，共同推进 AI 驱动自动化的未来
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1318310308938518589)** (43 messages🔥): 

> `基于 Pythia 的 RLHF 模型, TPU v5p 上的 TensorFlow, Tokenizer 边缘情况, 指数移动平均 (EMA), VLM 预训练数据` 


- **寻找基于 Pythia 的 RLHF 模型**：一位成员询问是否有**公开可用的基于 Pythia 的 RLHF 模型**。
   - 讨论中没有推荐具体的模型。
- **TensorFlow 在 TPU v5p 上崩溃**：一位用户报告了**在 TPU v5p 上运行 TensorFlow** 的问题，称在多个 VM 镜像中 'import tensorflow' 都会导致段错误（segmentation faults）。
   - 在持续的挑战中，人们对 Google 逐渐减少对 TensorFlow 的支持表示担忧。
- **Tokenizer 的边缘情况问题**：成员们分享了关于 **Tokenizer 丢弃**重要用户信息（由于边缘情况和 BPE 偏差）的担忧。
   - 引用了一篇讨论这些训练问题对性能影响的论文。
- **利用指数移动平均 (EMA)**：讨论了将**指数移动平均 (EMA)** 作为解决模型近期偏差（recency bias）的一种方法，建议它可以帮助平滑权重。
   - 一位成员强调 EMA 已经过大量测试，特别是在扩散模型（diffusion models）中。
- **VLM 预训练数据的最佳实践**：一位成员寻求 **VLM 预训练数据**的建议，特别是大量的图像和标题集合。
   - 有人建议 **Laion-5B** 尽管数据存在噪声，但仍是一个首选方案。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1318327764851363922)** (101 messages🔥🔥): 

> `Attention Mechanisms, Gradient Descent Optimizers, Grokking Phenomenon, Memory Augmented Neural Networks, Stick Breaking Attention` 


- **质疑 Attention 的 Kernel Approach**：成员们讨论了将 Attention 机制界定为核方法（Kernel Methods）的局限性，认为这种视角可能会忽略 Attention 超出经典核近似（Kernel Approximations）的核心功能。
   - 一位成员提出，Attention 的真实操作可能与检索（Retrieval）能力的关系比简单的聚合（Aggregation）更紧密。
- **SGD-SaI：优化器的新方法**：SGD-SaI 的引入为训练深度网络提供了一个新视角，通过在不使用自适应矩（Adaptive Moments）的情况下增强随机梯度下降（SGD），取得了与 AdamW 相当的结果。
   - 参与者强调需要与已有的优化器进行公正的对比，并建议在训练阶段动态调整学习率。
- **关于 Grokking 和复杂性的见解**：最近的一篇论文探讨了神经网络复杂性与 Grokking 现象之间的联系，提出了一种基于 Kolmogorov 复杂度的新指标。
   - 该研究旨在更好地理解模型何时产生泛化（Generalization）与记忆（Memorization），可能为训练动态提供结构化的见解。
- **Stick Breaking Attention 机制**：关于 Stick Breaking Attention 的讨论揭示了一种自适应聚合 Attention 分数的方法，可能减轻模型中的过平滑（Oversmoothing）效应。
   - 成员们讨论了这些自适应方法是否能更好地处理 Transformer 架构中学习到的表示复杂性。
- **学习率的 Warmup 策略**：对正确的学习率 Warmup 策略的澄清重点介绍了在优化器设置期间调整学习率的公式，特别是针对 Beta 衰减（Beta Decay）的考虑。
   - 参与者分享了 Warmup 策略的实现和注意事项，指出了常见学习率调度器（Schedulers）中潜在的陷阱。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.09810">The Complexity Dynamics of Grokking</a>: 我们通过压缩的视角研究泛化现象。特别是，我们研究了神经网络的复杂性动态来解释 Grokking，即网络突然发生转变...</li><li><a href="https://arxiv.org/abs/1803.00904">Hardness of Approximate Nearest Neighbor Search</a>: 我们证明了在欧几里得、曼哈顿、汉明或编辑距离下的近似双色最近邻对（Bichromatic Closest Pair）具有条件近二次运行时间的下界。具体而言，除非强指数...</li><li><a href="https://arxiv.org/abs/2412.11768">No More Adam: Learning Rate Scaling at Initialization is All You Need</a>: 在这项工作中，我们质疑了在训练深度神经网络时自适应梯度方法的必要性。SGD-SaI 是对带动量的随机梯度下降（SGDM）的一种简单而有效的增强...</li><li><a href="https://arxiv.org/abs/2403.02920v2">TaylorShift: Shifting the Complexity of Self-Attention from Squared to Linear (and Back) using Taylor-Softmax</a>: Attention 机制的平方复杂度是使用 Transformer 处理长序列的最大障碍之一。目前的方法依赖于稀疏表示或状态...</li><li><a href="https://brantondemoss.com/research/grokking/">The Complexity Dynamics of Grokking</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2209.04881">On The Computational Complexity of Self-Attention</a>: Transformer 架构在许多最先进的应用中取得了显著进展。然而，尽管取得了成功，现代 Transformer 仍依赖于 Self-Attention 机制，其时间和...</li><li><a href="https://arxiv.org/abs/2006.11527">Memory Transformer</a>: 基于 Transformer 的模型在许多自然语言处理任务中取得了最先进的结果。Self-Attention 架构允许 Transformer 结合来自所有元素的信息...</li><li><a href="https://github.com/lucidrains/x-transformers?tab=readme-ov-file#memory-transformers">GitHub - lucidrains/x-transformers: A concise but complete full-attention transformer with a set of promising experimental features from various papers</a>: 一个简洁但完整的全注意力 Transformer，包含来自各种论文的一系列极具前景的实验性功能 - lucidrains/x-transformers
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1318402720314560542)** (3 messages): 

> `Steering Vectors, SAEs and Interpretability, Unlearning with SAE Conditional Steering` 


- **Steering Vectors 揭示可解释路径**：团队分享了关于 [steering vectors](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx) 的发现，断言它们表明 LLM 中的线性方向是可解释的，特别是利用 SAE 的分解能力。
   - 研究人员使用 [Smith et al](https://www.alignmentforum.org/posts/C5KAZQib3bzzpeyrg/progress-update-1-from-the-gdm-mech-interp-team-full-update#Replacing_SAE_Encoders_with_Inference_Time_Optimisation) 的梯度追踪算法（gradient pursuit algorithm），分解了 steering vectors，以发现与拒绝（refusal）和谄媚（sycophancy）相关的极具前景的特征。
- **对 SAE 性能的复杂态度**：一位成员表达了怀疑，将整体结果标记为**负面**，尽管 SAE 带来的一些改进超过了可能归因于随机噪声的程度。
   - 这种观点凸显了在各种应用中解释 SAE 有效性时存在的争议。
- **由 Arthur Conmy 指导的论文**：一位成员提到 Arthur Conmy 指导了一篇专注于通过 SAE 条件控制效应进行**遗忘学习 (unlearning)** 的论文。
   - 这意味着围绕 SAE 在特定语境下的影响和有效性，讨论仍在继续。



**提到的链接**：<a href="https://www.lesswrong.com/posts/k8bBx4HcTF9iyikma/sae-features-for-refusal-and-sycophancy-steering-vectors#Evaluating_vectors_and_reconstructions>">SAE features for refusal and sycophancy steering vectors — LessWrong</a>：TL;DR * Steering vectors 提供了 LLM 中线性方向是可解释的证据。由于 SAE 分解了线性方向，它们应该是一个……

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1318491781540741203)** (43 messages🔥): 

> `VLLM performance, Winogrande dataset issues, New release updates, Library requirements for benchmarks, Chat template integration` 


- **VLLM 输出速度受限**：据报告，在 MMLU Pro 上使用 **8B 模型**时，**VLLM** 的每秒输出 token 数仅为 **24.69/s**，这主要是因为每个样本都需要进行一次前向传播，而没有利用 kv-cache。
   - 使用 `--data_parallel_size=N` 可以通过根据张量并行（tensor parallel）大小增加模型副本来帮助优化性能。
- **Winogrande 格式问题**：一位成员遇到了 **Winogrande** 数据集导致 `IndexError` 的问题，可能是由于其独特的格式影响了分词（tokenization）。
   - 提交了一个新的 [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2576) 以解决 Winogrande 任务的问题，同时指出 fewshots 可能仍然存在挑战。
- **新版本停止支持 Python 3.8**：一位成员提到，**PyPI** 上的最新版本在 10 月 Python 3.8 EOL（生命周期结束）后停止了对其的支持。
   - 这一变化反映了为确保 evaluation harness 的兼容性和性能改进而进行的持续更新。
- **基准测试运行期间缺少库**：成员们指出，在运行基准测试时缺少 **vllm**、**langdetect** 和 **antlr4-python3-runtime** 等库，这些库没有被 evaluation harness 自动安装。
   - 目前尚不清楚 harness 是否旨在自动安装各种基准测试所需的所有库，因此提出了这一问题以引起关注。
- **Chat template 影响评估分数**：有人担心新的 chat template 会影响评估结果，成员们不确定在没有它的情况下如何与之前的分数进行比较。
   - 测试将继续进行，以确定新的集成是否达到或超过了先前的性能指标。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1318398776536207423)** (2 messages): 

> `Non-parametric LayerNorm, Configuration Options, Memory Recall on Config Changes` 


- **新增非参数化 LayerNorm**：一位成员确认在配置中添加了**非参数化 LayerNorm**，表明可用选项发生了变化。
   - 这一添加反映了对现有系统的持续更新和增强。
- **配置选项仍在讨论中**：另一位成员对之前的配置选项表示不确定，认为信息在过去一年中可能已经发生了变化。
   - 这凸显了配置不断演进的特性以及持续审查的必要性。


  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1318307538801922058)** (173 messages🔥🔥): 

> `Cursor IDE 更新、AI 模型问题、Cursor 扩展发布、用户对模型的反馈、O1 Pro 讨论` 


- **Cursor 扩展发布**：一个新的 Cursor 扩展允许用户轻松地将他们的 Composer 和聊天历史导出为 Markdown，从而提高生产力并方便分享。
   - 该扩展还具有将内容发布到 Web 的选项，提供了一种有效记录编程交互的方法。
- **聊天和 AI 模型问题**：用户报告了 Claude 3.5 和 Gemini 等模型的连接失败、延迟问题以及响应质量的整体下降，表明存在持续的不稳定性。
   - 几位用户在聊天会话中遇到了问题，引发了多起关于 AI 交互可靠性的投诉。
- **关于 O1 Pro 的讨论**：在对 O1 Pro 能力的兴奋中，用户推测了其定价模式以及它在复杂编程任务中提供的优势。
   - 一些人表示需要更好的集成，建议在 O1 Pro 和 Claude 之间增加交接功能，以增强项目工作流。
- **社区支持与反馈**：成员们讨论了为软件开发项目提供同行支持，并对审查新的 SEO 工具表现出特别兴趣。
   - 用户鼓励分享关于 Cursor IDE 生态系统中各种工具和模型有效性的经验和技巧。
- **聊天管理功能需求**：讨论了对重复聊天功能的需求，以此作为更有效地管理聊天会话并保留上下文的一种方式。
   - 成员们分享了利用 Markdown 导出选项作为等待官方增强功能期间的临时解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/NVIDIAAIDev/status/1868778156347339033">来自 NVIDIA AI Developer (@NVIDIAAIDev) 的推文</a>：👀 RAPIDS cuDF 在无需更改代码的情况下将 #pandas 加速高达 150 倍。现在，随着数据集大小增长到 GB 级，你可以继续使用 pandas。⚡ ➡️ 尝试演示的 Jupyter Notebook：http://nvda.ws...</li><li><a href="https://tenor.com/view/chad-monke-gif-20835999">Chad Monke GIF - Chad Monke - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://marketplace.visualstudio.com/items?itemName=SpecStory.specstory-vscode">SpecStory (Cursor 扩展) - Visual Studio Marketplace</a>：Visual Studio Code 扩展 - (Cursor 扩展) 记录、搜索并从每一次 AI 编程之旅中学习</li><li><a href="https://tenor.com/view/champoy-el-risitas-kek-issou-etu-gif-17837830">Champoy El Risitas GIF - Champoy El Risitas Kek - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/shut-up-and-take-my-money-gif-13250127">Shut Up And Take My Money GIF - Shut Up And Take My Money - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://vm.tiktok.com/ZNeTGbon9/">TikTok - Make Your Day</a>：未找到描述</li><li><a href="https://x.com/mckaywrigley/status/1868341756494053573?s=46">来自 Mckay Wrigley (@mckaywrigley) 的推文</a>：我让 o1 pro 实现了我今天项目待办事项列表上的 6 件事。- 它思考了 5 分 25 秒。- 修改了 14 个文件。- 64,852 个 input tokens。- 14,740 个 output tokens。100% 正确 - 为我节省了 2...</li><li><a href="https://tenor.com/view/you-say-you-hate-me-then-love-me-jamie-fine-hate-me-love-me-song-youre-giving-me-mixed-signals-you-have-mixed-emotions-gif-26309096">You Say You Hate Me Then Love Me Jamie Fine GIF - You Say You Hate Me Then Love Me Jamie Fine Hate Me Love Me Song - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1318637734637604966)** (1 messages): 

> `DevDay 节日版、OpenAI API 团队 AMA` 


- **DevDay 节日活动启动**：一场名为 *Day 9: DevDay Holiday Edition* 的 YouTube 直播已排期，可在此处观看 [here](https://www.youtube.com/live/XKABimtOWME?si=_EsIUcPOK8-UTWL5)。
   - 直播之后，计划于太平洋时间上午 10:30–11:30 在开发者论坛举行 AMA。
- **加入 OpenAI API 团队的 AMA**：团队邀请社区成员参加其开发者论坛上的 AMA，详情请见 [here](https://community.openai.com/t/ama-on-the-17th-of-december-with-openais-api-team-post-your-questions-here/1057527)。
   - 鼓励成员在指定的一小时内向 OpenAI API 团队提出问题。



**提到的链接**：<a href="https://www.youtube.com/live/XKABimtOWME?si=_EsIUcPOK8-UTWL5"> - YouTube</a>：未找到描述

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1318306812600127552)** (130 条消息🔥🔥): 

> `AI 口音与真实感，OpenAI 功能与局限，AI 交互与对齐，Anthropic 价格变动，API 功能疑虑` 


- **AI 可以模仿口音，但仍感觉生硬**：用户讨论了 AI 在多种语言和口音之间切换的能力，特别提到了它模仿澳大利亚口音的表现。尽管有这些进步，许多人仍觉得交互过程不自然，缺乏真实的对话流。
   - 一位用户指出，虽然 AI 可以使用不同的口音，但它经常拒绝某些请求，这反映了关于尊重性交互的潜在准则限制。
- **OpenAI 的功能备受关注**：用户对 chat-gpt.com 上搜索放大镜图标的缺失表示担忧，分享了不同的体验并推测了其消失的潜在原因。一些人注意到侧边栏的可见性可能会影响功能的可用性。
   - 此外，用户提到审核分类器（moderation classifiers）可能会导致某些内容被自动标记，从而给被标记的用户带来临时的访问问题。
- **AI 对齐框架出现**：一位用户分享了他们在开发解决 AI 对齐（alignment）问题框架方面的工作，并发布了其 GitHub 仓库链接。这引发了关于在孤立环境下开发 AI 与在现实世界互动中开发 AI 的更广泛影响的讨论。
   - 另一位贡献者强调了现实世界交互对于有效 AI 开发的重要性，将受限的 AI 比作缺乏基本学习机会的孤独体验。
- **Anthropic 调整其定价策略**：分享了关于 Anthropic 定价模型的更新，据报道，随着 API 引入 Prompt Caching（提示词缓存），价格变得更便宜。这一变化表明在 AI 领域持续发展的背景下，这是一个竞争性的举措。
   - 用户对这一转变可能如何影响他们的使用情况以及与 OpenAI 产品的持续竞争表示好奇。
- **与 AI 的交互仍存在不确定性**：关于 AI 交互局限性的担忧浮出水面，特别是当用户试图询问截止日期（cut-off dates）或衡量情感反应时。用户对 AI 表现出与其能力相矛盾的行为感到沮丧，这导致了对其真实潜力的困惑。
   - 对话强调了在承认技术不断进步的同时，用户希望对如何与 AI 互动有更清晰的理解和功能支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.blackbox.ai/">Chat Blackbox: AI Code Generation, Code Chat, Code Search</a>: 未找到描述</li><li><a href="https://community.openai.com/t/search-magnifying-glass-in-the-left-sidebar-not-available/1048289/17">Search magnifying glass in the left sidebar not available</a>: 我仍然遇到这个问题。我清理了缓存，并在隐身模式下使用了不同的浏览器，但都没有成功。更新：现在显示了。</li><li><a href="https://github.com/AlignAGI/Alignment/">GitHub - AlignAGI/Alignment: Promoting global awareness and action for ethical AI alignment and safeguarding humanity against AI self-replication risks. Includes research, frameworks, and open-source resources.</a>: 促进全球对伦理 AI 对齐的意识和行动，保护人类免受 AI 自我复制风险。包括研究、框架和开源资源。- AlignAGI/Alig...</li><li><a href="https://www.youtube.com/watch?v=2tGtgH96nZ4"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1318375912835645461)** (12 messages🔥): 

> `Custom GPT 问题，Advanced Voice Mode 功能，PDF 和图像读取功能，Custom GPT 的 Project 替代方案，Custom GPT 的最大文件大小和限制` 


- **Custom GPT 编辑和可见性问题**：多位用户报告称，他们失去了**编辑 Custom GPT** 的能力，并且在访问平台时无法找到它们，即使之前已经进行了多次设置。
   - 这似乎是一个**已知问题**，其他成员也证实了在访问其自定义设置时遇到了类似问题。
- **关于 Advanced Voice Mode 的咨询**：一位用户询问 **ChatGPT PRO 订阅**每天可以使用多少分钟的 Advanced Voice Mode 功能。
   - 这凸显了用户对于所提供的语音功能限制的关注度日益增加。
- **PDF 和图像读取能力受到质疑**：一名成员担心 **PDF 和图像读取选项**仅在移动设备上正常运行，而在电脑上则不然。
   - 许多用户对该问题缺乏支持表示沮丧，表明这是一个普遍存在的问题。
- **用 Project 替换 Custom GPT**：一位成员分享说，将 **Custom GPT 替换为 Project** 对他们的性能产生了积极影响。
   - 在 Custom GPT 设置持续出现问题的情况下，这一建议对寻找替代方案的其他用户很有帮助。
- **Custom GPT 限制的未来更新**：有关于解决 Custom GPT **最大文件大小和限制**的潜在更新问题。
   - 这一讨论反映了用户对平台能力和可能增强功能的持续关注。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1318315909550440519)** (3 messages): 

> `使用 AI 构建网站，探索 AI 能力` 


- **利用 AI 进行网站开发**：一位成员建议，用户在与模型互动创建网站时，应沟通其**预期结果**和**可用工具**。
   - 他们鼓励进行从基础的**代码指导**到**优化现有代码**的各种咨询，无论用户的技能水平如何。
- **询问 AI 的能力**：一位用户强调了**直接询问** AI 它能做什么的重要性，暗示明确性可以增强协作成果。
   - 这强调了 AI 的灵活性，使用户能够根据自己的**经验**和**目标**定制交互。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1318315909550440519)** (3 messages): 

> `使用 AI 进行 Web 开发，最大化 AI 模型能力` 


- **探索使用 AI 进行 Web 开发**：一位成员建议向 AI 模型分享你想要的项目、工具和经验水平，以帮助从头开始创建网站。
   - 他们强调，该模型可以在任何水平上提供帮助，从完全的初学者到更有经验的开发人员，提供指导和代码检查。
- **询问模型其具备的能力**：.pythagoras 鼓励成员直接询问 AI 模型可以为他们的项目完成什么。
   - 这可以通过提示特定的功能查询或项目想法来增强用户体验。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1318636917553627166)** (1 messages): 

> `Structured Outputs，多模型应用，OpenRouter 模型支持` 


- **OpenRouter 扩展至支持 46 个模型**：OpenRouter 现在支持 **46 种不同的模型**进行 Structured Outputs，显著增强了多模型应用的可用性。
   - 通过 Structured Outputs，可以更轻松地将 **LLM 输出限制为 JSON schema**，从而简化开发流程，如[此处的演示](https://x.com/OpenRouterAI/status/1869077909438091485)所示。
- **Structured Outputs 的标准化**：该平台现在对 **8 家不同模型公司**的 Structured Outputs 进行了标准化，并包含 **8 个免费模型**。
   - 这种广泛的支持旨在促进各种模型更顺畅地集成到应用程序中，强调了 Structured Outputs 被低估的特性。



**提到的链接**：<a href="https://x.com/OpenRouterAI/status/1869077909438091485">来自 OpenRouter (@OpenRouterAI) 的推文</a>：Structured Outputs 被严重低估了。将 LLM 输出限制为 JSON schema 通常比请求工具调用要容易得多。OpenRouter 现在为 46 个模型、8 个...实现了 Structured Outputs 标准化。

  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1318351149228757022)** (2 messages): 

> `` 


- **Shashank 对某些精彩内容感到兴奋**：一位成员表达了热情，称：*"太棒了！"*
- **感谢已获确认**：另一位成员对这种热情表达了感谢，对前一条消息中分享的兴奋表示赞赏。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1318325767838629909)** (130 messages🔥🔥): 

> `Gemini Flash 2 performance, Using typos in prompts for model response, API Key Exposure, OpenRouter API limitations, o1 API changes and pricing` 


- **Gemini Flash 2 显示出改进的代码编写能力**：成员们讨论了 **Gemini Flash 2** 在解决科学问题任务时如何比 **Sonnet 3.5** 生成更好的代码，特别是在数组大小调整（array sizing）场景中。
   - 反馈表明，外部框架可以帮助增强有效性，引起了对特定用例效率的关注。
- **在 Prompt 中尝试拼写错误以引导 AI**：成员们分享了在 **Prompt** 中有意放置拼写错误和无意义词汇以影响模型输出的想法，强调了这对创意写作的潜在好处。
   - 该策略包括吸引模型对所需关键词的注意，同时利用 **Chain of Thought (CoT)** 技术实现受控输出。
- **报告泄露的 OpenRouter API Key**：一位成员报告在 **GitHub** 上发现了泄露的 **OpenRouter API Key**，引发了关于出于安全原因应在何处报告此类发现的讨论。
   - 建议通过电子邮件发送至 **support@openrouter.ai** 以处理任何可能构成风险的泄露 Key。
- **关于对话详情的 OpenRouter API 限制**：关于从 **OpenRouter API** 检索对话历史或 Prompt 输入能力的问题被提出，重点在于请求的无状态（stateless）性质。
   - 澄清指出，虽然元数据可用，但完整的对话并不存储在 **OpenRouter** 上，因此需要代理（proxy）来进行对话日志记录。
- **o1 API 的价格和 Token 使用变化**：用户注意到 **o1 API** 现在消耗的 **Token 减少了 60%**，这引发了对模型性能潜在影响的担忧。
   - 讨论强调了调整定价和 **Token** 效率的必要性，同时确认目前仍适用层级限制。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: 设置模型使用限制</li><li><a href="https://openrouter.ai/google/gemini-exp-1206:free/parameters">Google: Gemini Experimental 1206 (free) – Recommended Parameters</a>: 查看 Google: Gemini Experimental 1206 (free) 的推荐参数和配置 - Gemini 的实验性版本（2024年12月6日）。</li><li><a href="https://openrouter.ai/google/gemini-2.0-flash-exp:free">Gemini Flash 2.0 Experimental (free) - API, Providers, Stats</a>: 与 Gemini Flash 1 相比，Gemini Flash 2.0 提供了显著更快的首个 Token 时间 (TTFT)。通过 API 运行 Gemini Flash 2.0 Experimental (free)。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1318352802954608690)** (1 messages): 

> `Perplexity Pro gift subscriptions, Subscription benefits, Subscription durations` 


- **通过 Perplexity Pro 订阅赠送知识**：**Perplexity** 现在提供 1、3、6 或 12 个月的**礼品订阅**，可在[此处](https://perplexity.supply/shop/perplexity-subscription)购买。对于任何对 AI 感兴趣的人来说，这都是一份理想的礼物，可以让他们解锁增强的搜索功能。
   - 接收者将通过直接发送到其电子邮箱的**优惠码**收到订阅，使赠送过程简单高效。
- **通过 Pro 解锁强大功能**：**Perplexity Pro** 用户可以搜索 **3 倍数量的来源**，访问最新的 AI 模型，并搜索自己的文件。这种广泛的能力使得该订阅对资深 AI 爱好者特别有吸引力。
   - 在促销信息中强调，**所有销售均为最终决定**，确保购买者在购买时已完全确认。



**提到的链接**: <a href="https://perplexity.supply/shop/perplexity-subscription">Perplexity Pro Subscription | Perplexity Supply</a>: Perplexity Supply 旨在通过精心设计的产品探索时尚与智慧之间的关系，以激发对话并展示你对知识的无限追求。

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1318308631489413201)** (90 条消息🔥🔥): 

> `OpenAI vs. Perplexity 功能对比, Perplexity Pro 订阅, 模型性能对比, API 使用指南, 用户界面建议` 


- **关于 OpenAI 借鉴功能的辩论**：用户讨论了 OpenAI 是在创新还是仅仅在抄袭 **Perplexity** 的功能，特别是关于 Projects 和 GPT Search。
   - *一些成员指出，每个人都在互相模仿*，引发了关于 AI 开发原创性的讨论。
- **Perplexity Pro 订阅体验**：一位用户分享了使用 Perplexity Pro 的积极体验，强调了其与 ChatGPT 相比在研究方面的有效性。
   - 其他人询问了模型差异和偏好，表明希望明确哪种模型最适合各种任务。
- **对模型性能的担忧**：用户对感知到的模型性能下降表示沮丧，特别是 **Sonnet** 和 **Claude** 变体。
   - 一些人报告说，切换到特定模型会显著影响响应质量，导致不同的用户体验。
- **API 信息与使用**：一位成员寻求获取 Perplexity API token 的指导，而其他人则提供了官方设置文档的链接。
   - 澄清了尽管拥有 Pro 账户，API 的使用仍需要单独的注册和支付方式。
- **用户界面与功能请求**：用户建议对 Perplexity UI 进行美化改进，例如添加降雪效果以增加视觉吸引力。
   - 讨论强调，虽然一些人在工作中优先考虑功能性，但另一些人则欣赏美学增强。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/pplxsupply/status/1868738538231287816?s=46">来自 Perplexity Supply (@PPLXsupply) 的推文</a>：赠送知识的礼物。Perplexity Pro 礼品订阅现已推出。</li><li><a href="https://docs.perplexity.ai/guides/getting-started">初始设置 - Perplexity</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=xZc0YQbIyWE"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=gSypQljcZgM"> - YouTube</a>：未找到描述</li><li><a href="https://www.copilotforyoutube.com/search/search12-days-of-openai-day-8-rQQJ3bPMn1WyaUvBqMUw9F">搜索—OpenAI 的 12 天：第 8 天</a>：Kevin Weil、Adam Fry 和 Cristina Scheau 介绍并演示了 ChatGPT search 的更新。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1318368690143629334)** (6 条消息): 

> `Mozi 社交应用, 美国军事太空战, 神秘的格鲁吉亚平板, 区块链, 一水肌酸` 


- **Mozi 应用在社交媒体热议中发布**：由 Ev Williams 推出的新社交应用 **Mozi** 正在各平台引发关注，承诺提供一种全新的社交网络方式。
   - 随附的 [YouTube 视频](https://www.youtube.com/embed/RNXnnOT3-9Y) 深入探讨了其功能和特性。
- **美国军方为太空战做准备**：最近的讨论显示，**美国军方**正在为外层空间的潜在对抗进行战略准备，强调技术进步。
   - 在关于其**太空霸权**计划的相关文章中可以找到详细视图。
- **发现神秘的格鲁吉亚平板**：一篇引人入胜的文章强调了**神秘格鲁吉亚平板**的发现，引起了历史学家和考古学家的共同兴趣。
   - 关于其起源和意义的细节可以在[此处阅读全文](https://www.perplexity.ai/page/mystery-georgian-tablet-found-q4MNqPlyRl.5PTZ34uuzmw)。
- **Qu Kuai Lian 探索区块链创新**：关于 **Qu Kuai Lian** 的最新动态展示了区块链领域的创新，吸引了开发者和投资者的极大兴趣。
   - 对于感兴趣的人，可以从该资源中获得进一步的见解。
- **分析一水肌酸的益处**：讨论 **MuscleBlaze 一水肌酸** 的链接详细介绍了其益处，包括增强性能和恢复策略。
   - 访问此 [链接](https://www.perplexity.ai/search/muscleblaze-creatine-monohydra-0zZxrrV2QTe51OChDjbcVQ#1) 获取全面分析。



**提到的链接**：<a href="https://www.youtube.com/embed/RNXnnOT3-9Y">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1318440633496506452)** (2 条消息): 

> `Perplexity MCP, MCP server integration, Using models with APIs, Gemini integration, Access to models` 


- **在聚合网站上发现 Perplexity MCP**：一位用户提到他们在关注的两个聚合网站之一发现了 **Perplexity MCP**，并承诺稍后发布更多细节。
   - 这表明社区成员对 **MCP server** 的兴趣日益浓厚。
- **日常使用的 MCP server**：另一位用户确认他们每天都在使用一个特定的 MCP server，并提供了 **any-chat-completions-mcp** 项目的 [GitHub 仓库](https://github.com/pyroprompts/any-chat-completions-mcp) 链接。
   - 他们强调可以访问各种模型，如 **Gemini**、**OpenAI** 和 **Groq**，**Mistral** 也即将推出。
- **讨论 Gemini API 整合**：用户分享了他们利用通过 **OpenAI SDK 实现的新 Gemini 整合**，并提供了访问的 **base URL**：[Gemini API](https://ai.google.dev/gemini-api/docs/openai)。
   - 这种整合似乎通过允许无缝地与多个 API 交互来增强用户体验。
- **提到模型访问权限**：另一位用户表示他们可以访问 API 提供的任何模型，包括 Perplexity 的 **网页访问和引用功能**。
   - 这种可访问性为用户高效利用不同 AI 模型提供了强大的能力。



**提到的链接**：<a href="https://github.com/pyroprompts/any-chat-completions-mcp">GitHub - pyroprompts/any-chat-completions-mcp</a>：通过在 GitHub 上创建账户，为 pyroprompts/any-chat-completions-mcp 的开发做出贡献。

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1318364624831381597)** (84 条消息🔥🔥): 

> `Palmyra Creative, OpenAI API updates, NVIDIA Jetson Orin Nano, O1 and O1 Pro distinction, Anthropic API updates` 


- **Palmyra Creative 发布**：新的 [Palmyra Creative](https://x.com/waseem_s/status/1869040950464459216) 模型旨在增强业务任务中的创造力，具有 128k **context window**，适用于头脑风暴和分析等富有想象力的任务。
   - 它可以与特定领域的模型无缝集成，是营销人员到临床医生等各类专业人士的理想选择。
- **令人兴奋的 OpenAI API 更新**：[小型开发者日](https://x.com/kevinweil/status/1869084308432109948) 重点介绍了 OpenAI API 的重要更新，包括具有 **function calling** 的 O1 实现和新的语音模型功能。
   - WebRTC 对实时语音应用的支持以及显著的输出 **token** 增强是活动期间的主要发布内容。
- **Jetson Orin Nano Super Kit 介绍**：NVIDIA 的 Jetson Orin Nano Super 开发者套件承诺增强 AI 处理能力，神经处理性能提升 **70%** 达到 **67 TOPS**，内存带宽达到 **102 GB/s**。
   - 它旨在为爱好者提供高性价比的 AI 能力，保持 **$249** 的价格，作为一种易于获取的 AI 计算解决方案。
- **关于 O1 与 O1 Pro 的澄清**：[Aidan McLau 澄清](https://x.com/michpokrass/status/1869102222598152627) O1 Pro 是独立于标准 O1 模型的一个独特实现，专为更高的推理能力而设计。
   - 这一区别在社区内引起了关于这些模型功能可能产生混淆的讨论。
- **Anthropic API 功能更新**：Anthropic 宣布四项新功能结束测试进入正式版，包括其 API 的 **prompt caching** 和 **PDF 支持**。
   - 这些更新旨在提升使用 Anthropic 平台的开发者的开发体验，促进更顺畅的操作。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/rjvir/status/1868815344200303030?s=46">来自 Raj Vir (@rjvir) 的推文</a>：Google Gemini 在开发者中的市场份额从 9 月的约 5% 增长到上周的 >50%（据 @OpenRouterAI 数据）</li><li><a href="https://apollo-lmms.github.io/">Apollo</a>：Apollo：大语言多模态模型（Large Multimodal Models）中的视频理解探索</li><li><a href="https://x.com/fofrai/status/1868763436974588222?s=46">来自 fofr (@fofrAI) 的推文</a>：Minimax 的新视频模型 `video-01-live` 已在 Replicate 上线：https://replicate.com/minimax/video-01-live。它在动画和保持角色一致性方面表现非常出色。输出结果...</li><li><a href="https://x.com/waseem_s/status/1869040950464459216">来自 Waseem AlShikh (@waseem_s) 的推文</a>：企业级 AI 在创造性问题解决和头脑风暴方面的应用至今表现平平，现有的 LLM 无法为商业用户所需的关键任务生成突破性想法...</li><li><a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">Scaling test-time compute - HuggingFaceH4 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.theverge.com/2024/12/17/24323450/nvidia-jetson-orin-nano-super-developer-kit-software-update-ai-artificial-intelligence-maker-pc">Nvidia 售价 249 美元的开发套件承诺提供廉价、小巧的 AI 动力</a>：价格仅为前代产品的一半。</li><li><a href="https://x.com/vercel/status/1869083642938712368?s=46">来自 Vercel (@vercel) 的推文</a>：我们分析了数十亿次 AI 爬虫请求，以了解每个爬虫如何处理 JavaScript 渲染、资产和其他行为。这是我们的发现。</li><li><a href="https://huggingface.co/spaces/samjulien/palmyra-creative">Palmyra Creative - samjulien 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/mckaywrigley/status/1869084297707278584">来自 Mckay Wrigley (@mckaywrigley) 的推文</a>：@OpenAI 你们快看！上下文窗口（context window）提升了 56%，输出 Token 提升了 3 倍！！！</li><li><a href="https://www.modular.com/blog/introducing-max-24-6-a-gpu-native-generative-ai-platform">Modular：推出 MAX 24.6：一个 GPU 原生生成式 AI 平台</a>：MAX 24.6 发布博客，重点介绍 MAX GPU</li><li><a href="https://x.com/kevinweil/status/1869084308432109948">来自 Kevin Weil 🇺🇸 (@kevinweil) 的推文</a>：第 9 天：✨ mini dev day ✨ 今天发布了一系列更新：• API 中的 o1，包含函数调用（function calling）、结构化输出（structured outputs）、开发者消息和视觉（vision）• GPT-4o 和 4o-mini 语音模型更新...</li><li><a href="https://x.com/craigsdennis/status/1869085459143688271">来自 Craig Dennis (@craigsdennis) 的推文</a>：我一直在愉快地探索 @OpenAI 的 Realtime API。今天他们刚刚推出了通过 WebRTC 连接的功能！🤖🖐️ 它支持工具调用（Tool Calling），快来看看效果...</li><li><a href="https://x.com/rjvir/status/1868815344200303030">来自 Raj Vir (@rjvir) 的推文</a>：Google Gemini 在开发者中的市场份额从 9 月的约 5% 增长到上周的 >50%（据 @OpenRouterAI 数据）</li><li><a href="https://x.com/alexalbert__/status/1869096718387872205">来自 Alex Albert (@alexalbert__) 的推文</a>：今天为开发者带来的体验优化更新。Anthropic API 的四个功能将结束 Beta 测试并正式发布：- Prompt 缓存 - Message Batches API（支持更大批次）- Token 计数...</li><li><a href="https://x.com/michpokrass/status/1869102222598152627">来自 Michelle Pokrass (@michpokrass) 的推文</a>：@aidan_mclau 嘿 Aidan，这不是误传，它们是不同的产品！o1 pro 是不同的实现，而不仅仅是具有高推理能力的 o1。</li><li><a href="https://x.com/_akhaliq/status/1868535608370708643">来自 AK (@_akhaliq) 的推文</a>：Meta 发布 Apollo：大语言多模态模型中的视频理解探索，一系列最先进的 video-LMMs</li><li><a href="https://x.com/fofrai/status/1868776722466009334?s=46">来自 fofr (@fofrAI) 的推文</a>：噢天哪，看看 video-01-live 填充她身后文字的效果有多好。初始图像中不包含 S 或 P。我甚至没在 Prompt 中提到 "space"。但不知何故它知道。🤯🤯 看看...</li><li><a href="https://community.openai.com/t/ama-on-the-17th-of-december-with-openais-api-team-post-your-questions-here/1057527">12 月 17 日与 OpenAI API 团队的 AMA：在此发布您的问题</a>：Little Dev Day 已确认。发布后立即加入 API 团队的 AMA！为了准备活动——或者如果您无法参加——您可以提前在此发布您的问题。AMA 将开始...</li><li><a href="https://x.com/osanseviero/status/1869024925249569114?s=46">来自 Omar Sanseviero (@osanseviero) 的推文</a>：OmniAudio 发布了！⚡️⚡️ 超快的本地语音 LLM 🤏 2.6B 参数 🔊 多模态：文本和音频输入 🤗 统一的 Gemma 和 Whisper。博客：https://nexa.ai/blogs/omniaudio-2.6b Demo：https://hf.co/spaces/NexaAID...</li>

ef="https://x.com/scaling01/status/1869007562034544939?s=46">来自 Lisan al Gaib (@scaling01) 的推文</a>：Falcon 3 模型在几小时前发布了！Huggingface 链接：https://huggingface.co/blog/falcon3。包含以下模型尺寸：1B, 3B, 7B, 10B & 7B Mamba，在 14 Trillion tokens 上训练，采用 apache 2.0 许可证...</li><li><a href="https://youtu.be/s85YY3myQLw?si=2e7Ub8SLeBt8rAhH"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=S9L2WGf1KrM"> - YouTube</a>：未找到描述</li><li><a href="https://huggingface.co/google/Gemma-Embeddings-v1.0">google/Gemma-Embeddings-v1.0 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com">来自 GitHub 的推文 - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>：修复损坏的 Twitter/X 嵌入！在 Discord, Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1318307670180102175)** (55 messages🔥🔥): 

> `Text to Speech 和 Speech to Text 工具，LM Studio 模型微调，无审查聊天机器人替代方案，macOS 上的散热和电源管理脚本，模型兼容性与性能` 


- **对 Text to Speech 集成的期待**：一位用户对未来在 LM Studio 中集成 **text to speech** 和 **speech to text** 功能表示乐观，目前如果需要已有替代方案。
   - 另一位成员提到，可以将工具作为服务器与 LM Studio 并行运行，以实现这些功能。
- **LM Studio 模型微调限制**：一位用户询问是否可以使用 Discord 数据导出内容来微调现有模型，但被告知 LM Studio 无法进行 **finetuning**。
   - 相反，他们可以设置 system prompts 来引导回复，但这仅限于当前会话。
- **探索无审查聊天机器人**：讨论引发了关于寻找 **uncensored chatbot** 替代方案的话题，并推荐了如 **Gemma2 2B** 和 **Llama3.2 3B** 等可在 CPU 上运行的模型。
   - 成员们被引导至如何有效使用这些模型的资源，包括量化选项的链接。
- **macOS 电源管理脚本**：一位用户分享了在发生热节流（thermal throttling）时管理 macOS 低功耗模式的脚本，旨在不引起过热的情况下优化性能。
   - 脚本根据系统热压力级别调整电源设置，尽管其有效性在用户中仍有争议。
- **硬件兼容性与性能查询**：一位初学者询问使用 **Intel i7** 和 **NVIDIA 4070ti Super** 运行 **PHI-4** 等模型的情况，并得到了关于其能力的确认。
   - 讨论涉及了当前框架中的模型支持，并分享了各种硬件配置的使用经验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF">bartowski/gemma-2-2b-it-abliterated-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF">bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/10814">功能请求：添加对 Phi-4 模型的支持 · Issue #10814 · ggerganov/llama.cpp</a>：前提条件：我正在运行最新代码。如果可能，请注明版本。我仔细阅读了 README.md。我搜索了与我的问题相关的关键词，以确保我正在创建...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1318313593846108241)** (29 条消息🔥): 

> `GPU Performance Comparison, Model Memory Usage in GPUs, Llama Model Settings, New GPU Listings, Driver Issues with AMD GPUs` 


- **GPU 性能表现出惊人的相似性**：用户观察到，虽然 **3070 Ti** 和 **3090** 的价格区间不同，但在游戏中的表现似乎几乎持平。
   - 一位用户能以 **~$750** 的价格找到 **3090**，而另一位用户报告其当地价格可能更低，约为 **$900 CAD**。
- **GPU 资源分配可能配置错误**：一位用户在多块 GPU 上仅获得 **2.13 tokens/second** 的缓慢性能，这表明可能是在使用 CPU 而非 GPU。
   - 尽管进行了配置，有时单块 GPU 会随机出现 **25% utilization**，这表明设置可能需要调整。
- **需要验证 Llama 模型设置**：有关 **Llama.cpp** 配置的担忧出现，需确保启用 CUDA 以有效利用 GPU。
   - 用户建议在模型大小为 **5GB** 时，尝试使用单块 GPU 测试以检查性能是否有所提升。
- **Zotac 意外列出即将推出的 RTX 50 系列 GPU**：[Zotac 官网](https://www.tomshardware.com/pc-components/gpus/zotac-accidentally-lists-rtx-5090-rtx-5080-and-rtx-5070-family-weeks-before-launch) 在正式发布前几周，无意中列出了 **RTX 5090**、**5080** 和 **5070** 系列及其高级规格。
   - 该列表确认 **RTX 5090** 将配备 **32GB GDDR7 memory**，引发了对即将推出的 Nvidia 硬件的关注。
- **AMD GPU 驱动程序问题**：一些用户遇到了 **24.12.1 AMD driver** 的问题，该问题影响了性能并导致 GPU 使用率激增，但实际功耗并未有效提升。
   - 回退到版本 **24.10.1** 解决了卡顿问题，并将模型的性能显著提升至 **90+ tokens/second**。



**提到的链接**：<a href="https://www.tomshardware.com/pc-components/gpus/zotac-accidentally-lists-rtx-5090-rtx-5080-and-rtx-5070-family-weeks-before-launch-inadvertent-listing-seemingly-confirms-the-rtx-5090-with-32gb-of-gddr7-vram">Zotac accidentally lists RTX 5090, RTX 5080, and RTX 5070 family weeks before launch &mdash; accidental listing seemingly confirms the RTX 5090 with 32GB of GDDR7 VRAM</a>：Zotac 的第三次失误！

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1318330222369570907)** (79 条消息🔥🔥): 

> `通过在线课程学习 Stable Diffusion，为 AI 选择 GPU 选项，诈骗与机器人检测方法，创建 Lora 模型，使用最新的 AI 模型` 


- **Stable Diffusion 的最佳在线课程**：一位成员表示有兴趣寻找全面的[在线课程](https://chatgpt.com/c/67618085-3fc4-8010-93a1-7bf7c6b79806)，这些课程将 YouTube 教程整合为一个资源，以便学习如何使用 A1111 运行 Stable Diffusion。
   - 社区讨论了对该主题易于获取的教育资源的需求。
- **为 AI 任务在笔记本电脑和台式机之间做出选择**：一位用户正在 **4090 笔记本电脑**或 **4070 TI Super 台式机**之间做决定，并指出两者都有 **16GB VRAM**，而其他人则建议台式机更适合 AI 工作。
   - 评论强调，尽管笔记本电脑适合游戏，但对于繁重的图形任务并不理想。
- **了解机器人检测技术**：对话集中在识别诈骗机器人上，强调了提出荒谬问题或使用“土豆测试”（potato test）等测试来区分人类和机器人的重要性。
   - 成员们指出，机器人和真实人类都可能带来风险，因此谨慎参与至关重要。
- **创建自己的 Lora 模型的步骤**：一位用户寻求制作 Lora 模型的建议，并收到了一个分步流程，包括创建数据集、选择模型和训练 Lora。
   - 还强调了研究如何创建有效的训练数据集的重要性。
- **咨询最新的 AI 模型**：一位回归的成员询问了当前正在使用的 AI 模型，特别提到了 'Flux.1-Dev'，并寻求有关其运行要求的信息。
   - 社区分享了关于模型使用趋势和有效 AI 实施要求的更新。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/what-huh-wat-wut-gif-13031409">What Huh GIF - What Huh Wat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/Bing-su/adetailer">GitHub - Bing-su/adetailer: 使用检测模型进行自动检测、遮罩和重绘。</a>: Auto detecting, masking and inpainting with detection model. - Bing-su/adetailer</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info">GitHub - CS1o/Stable-Diffusion-Info: Stable Diffusion 知识库（安装、基础、指南等）</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1318418824659206175)** (7 条消息): 

> `会议录制可用性、Distributed Training 课程、6D Parallelism 见解、NCCL 源码学习、生成过程中的 Tool Calls` 


- **会议录制访问**：一位成员询问会议录制是否可用，另一位成员随即分享了已上传的 [YouTube 链接](https://www.youtube.com/watch?v=hfb_AIhDYnA)。
   - *是的，已经上传到这里了！* 😀
- **Distributed Training 课程推荐**：一位成员寻求关于 **distributed training** 的全面课程推荐，特别是那些提供结构化作业和实践的课程。
   - 他们强调了现有资源的实用性，例如 GitHub 上的 [LambdaLabsML 指南](https://github.com/LambdaLabsML/distributed-training-guide?tab=readme-ov-file)。
- **关于 6D Parallelism 的讨论**：分享了一篇关于 **6D parallelism** 的近期文章，详细介绍了如何在 **2⁶ mesh** 中可视化集合通信（collective communications）。
   - 该文章批评其他资源未能深入分析训练步骤中涉及的通信，而这正是本文旨在解决的问题。
- **NCCL 源码学习资源发布**：一位成员提供了其 **NCCL source code study** 的链接，深入探讨了分布式训练场景中通信处理的细微差别。
   - 该资源包含多篇文章，涵盖了 NCCL 流程的各个方面，可作为相关兴趣者的全面指南。
- **生成过程中 Tool Calls 的效率**：一位成员思考了为 **external tool calls** 暂停生成对效率的影响，认为这可能导致 GPU 利用率问题。
   - 他们推测这些中断是否会影响整体性能，并寻求他人关于潜在解决方案的意见。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://main-horse.github.io/series/nccl-source-code-study/">NCCL Source Code Study</a>：无描述</li><li><a href="https://main-horse.github.io/posts/visualizing-6d/">Visualizing 6D Mesh Parallelism</a>：包含一些背景知识</li><li><a href="https://x.com/yoavgo/status/1868068060638359833),">来自 (((ل()(ل() 'yoav))))👾 (@yoavgo) 的推文</a>：“Toolformer”论文提出了“内联工具使用”的概念，即暂停解码，并在恢复前将 tool-call 的输出直接整合到生成的文本中……</li><li><a href="https://github.com/LambdaLabsML/distributed-training-guide?tab=readme-ov-file)">GitHub - LambdaLabsML/distributed-training-guide</a>：编写分布式 PyTorch 训练代码的最佳实践与指南</li><li><a href="https://www.youtube.com/watch?v=hfb_AIhDYnA"> - YouTube</a>：无描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1318331494883921991)** (7 条消息): 

> `CUDA Graph 与 cudaMemcpyAsync、4090 上的计算吞吐量、CUDA Graph 中的 Kernel 与 cudaMemcpyAsync` 


- **4090 达到 100% 计算吞吐量的最低占用率**：有人提问关于在 4090 上实现 **100% compute throughput** 所需的 **lowest occupancy**，并建议可能是 **4/64 (~7%)**。
   - 确切的阈值仍未确定，引发了成员间的进一步调查。
- **CUDA Graphs 与 cudaMemcpyAsync 的兼容性**：讨论了 **CUDA Graph** 是否支持 **cudaMemcpyAsync**，一位用户确认“支持”。
   - 这种兼容性似乎引发了一些担忧，特别是在 CUDA Graph 模式下使用 cudaMemcpyAsync 时，应用程序的结果会出现不一致。
- **CUDA Graph 模式下 cudaMemcpyAsync 的问题**：一位成员报告称，在 **CUDA Graph** 模式下使用 **cudaMemcpyAsync** 会导致错误的应用程序结果，而使用 **kernel copies** 则能得到正确结果。
   - 这种差异引发了关于 CUDA Graph 框架内运行时 API 流捕获（stream capture）的问题。
- **寻求 CUDA Graph 问题的最小示例**：一位用户请求提供一个 **minimal example**，以澄清其在 CUDA Graph 应用程序中使用 cudaMemcpyAsync 时遇到的问题。
   - 这引发了关于他们是否正在训练模型的进一步询问，表明需要详细的上下文来提供帮助。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1318533251349876887)** (9 条消息🔥): 

> `优化 PyTorch Docker 镜像，Conda 与 Docker 的使用对比，使用 Nix 构建自定义 Torch，Megatron-LM 在训练中的效率` 


- **优化 PyTorch Docker 镜像**：成员们讨论了官方 PyTorch Docker 镜像的大小，指出虽然起始大小约为 **3-7 GB**，但更小的镜像是可行的。
   - 一位用户的方法是使用 **纯净微型 30MB Ubuntu** 镜像，同时依赖 **Conda** 来获取 CUDA 库。
- **关于在 Docker 中使用 Conda 的辩论**：一位用户质疑同时使用 Conda 和 Docker 镜像的合理性，因为通常认为它们是替代方案。
   - 另一位用户回应称，这能确保安装的一致性，特别是在缺乏标准化的多元化开发团队中。
- **使用 Nix 自定义 Torch 构建**：一位成员解释了他们使用 **nixpkgs** 构建具有特定标志的自定义 Torch 包的方法，包括 `TORCH_CUDA_ARCH_LIST="8.0;8.6"`。
   - 他们提供了一个 [Dockerfile 示例](https://raw.githubusercontent.com/technillogue/build-pytorch/refs/heads/main/build-pytorch/Dockerfile) 来展示他们的配置。
- **学术界代码实践的挑战**：一位用户评论了管理复杂的学术代码库的困难，称参与者通常技术精湛但缺乏正式的编程训练。
   - 这种观点引起了其他人的共鸣，突显了学术界代码实践的混乱现状。
- **咨询 Megatron-LM 的训练效率**：一位参与者询问了 **Megatron-LM** 目前在分布式训练吞吐量效率方面的地位。
   - 他们表示有兴趣利用合适的代码库进行旨在增强性能的持续研究项目。



**提到的链接**: <a href="https://hub.docker.com/r/pytorch/pytorch/tags">未找到标题</a>: 未找到描述

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1318481559137026058)** (14 条消息🔥): 

> `NVIDIA Jetson Nano Super, JetPack 6.1 安装, AGX 上的 LLM 推理, 用于 LLM 的 Raspberry Pi 5, Esp32 / Xtensa LX7 芯片` 


- **NVIDIA 发布 Jetson Nano Super**：NVIDIA 推出了 **Jetson Nano Super**，这是一款能够进行每秒 **70-T 次运算** 的紧凑型 AI 计算机，专为机器人设计，售价 **$249**。
   - 讨论强调了其支持包括 **LLM** 在内的高级模型的能力，引发了对其与现有模型相比性能如何的好奇。
- **使用 JetPack 6.1 提升 Jetson Orin 性能**：用户可以使用 **SDK Manager** 安装 **JetPack 6.1**，以提升 Jetson Orin Nano 开发套件的性能。
   - 通过将电源模式更改为 **MAXN**，设备可以实现超强性能，优化运行能力。
- **AGX 上的 LLM 推理**：讨论了一个利用 **AGX Orin** 进行量化为 **gguf 8bit** 的 **11B 模型推理** 的研究项目。
   - 该设置旨在增强更复杂 **LLM** 应用的本地部署能力。
- **使用 Raspberry Pi 5 运行小模型**：讨论包括使用 **Raspberry Pi 5** 部署模型，该设备配置了 **nvme 256GB** 以实现更快的数据传输，并超频至 **2.8**。
   - 它已在本地用于运行小模型（1.5B 参数），利用编译了 **OpenBlas** 的 **Ollama** 来提高效率。
- **对新型 Esp32 / Xtensa LX7 芯片的期待**：表达了对新型 **Esp32 / Xtensa LX7 芯片** 的期待，这些芯片旨在用于通过 **API** 远程调用 **LLM** 的场景。
   - 这一进展反映了社区内对推进边缘计算能力的持续关注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/slow_developer/status/1869059311969661103">来自 Haider. (@slow_developer) 的推文</a>: 🚨 NVIDIA 推出 Jetson Nano Super > 能够进行每秒 70-T 次运算的紧凑型 AI 计算机 > 专为机器人设计，支持包括 LLM 在内的高级模型，售价 $249</li><li><a href="https://andrewkchan.dev/posts/yalm.html">从零开始的快速 LLM 推理</a>: 未找到描述</li><li><a href="https://nvdam.widen.net/s/zkfqjmtds2/jetson-orin-datasheet-nano-developer-kit-3575392-r2">jetson-orin-datasheet-nano-developer-kit-3575392-r2.pdf</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 条消息): 

pirate_king97: https://www.youtube.com/playlist?list=PLvJjZoRc4albEFlny8Z1OGDiF_y3MGNK9

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1318519459798974476)** (8 条消息🔥): 

> `phi-4 采样进度，Chain of Thought 数据集生成，使用 Axolotl、Unslosh 和 TRL 进行 VLM 微调，自定义 vision encoder 讨论，ARC 的 Test-time scaling` 


- **phi-4 采样进度揭示了谜题解答情况**：**phi-4 在 128 时的采样**正在进行中，目前已解决了 **119 个谜题**。然而，一个之前中止的运行已经解决了 **164 个谜题**，这表明通过增强测试方法有潜力获得更高的成功率。
   - 这表明使用优秀的验证器（verifier）可以显著增加解决问题的数量，因为之前的方法仅解决了 **32 个谜题**。
- **生成 CoT 数据集以评估有效性**：团队正在启动首次 **Chain of Thought (CoT) 数据集生成**，旨在找出哪些形式的 CoT 对模型性能的贡献最有效。他们计划让**强化学习来决定**哪些类型最有益。
   - 该实验旨在了解 CoT 是否能解决直接转导（direct transduction）无法解决的谜题。
- **分享 VLM 微调资源**：发现了关于使用 **Axolotl**、**Unslosh** 和 **Hugging Face TRL** 进行 **VLM 微调** 的教程，包括重要资源的链接。每个链接都为微调 vision-language models 提供了具体指导，这对于实现高效集成至关重要。
   - 使用 TRL 进行 VLM 微调的教程警告说，该过程是**资源密集型**的，需要大量的计算能力。
- **讨论自定义 vision encoders 以实现更好的集成**：考虑到传统模型可能无法很好地处理微小的像素尺寸，正在考虑是否创建一个**自定义 vision encoder** 以与语言模型集成。其想法是通过将 encoder 与任何可用的**语言模型**配对来提高灵活性。
   - 一位成员质疑，对于通常小于 **100x100 像素** 的图像，轻微的参数增加是否足以进行有效的特征提取。
- **探索 ARC 的 test-time scaling 潜力**：讨论集中在为 ARC 实现一个可扩展且能持续运行并随时间改进的测试系统是否可行。目标是评估 **CoT 推理 token** 是否能解决超出标准方法能力的复杂谜题。
   - 建议进入潜在的 Proof of Concept (PoC) 阶段，以确定**实现挑战**并评估有效性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing">Google Colab</a>：无描述</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/blob/effc4dc4097af212432c9ebaba7eb9677d768467/examples/llama-3-vision/lora-11b.yaml">axolotl/examples/llama-3-vision/lora-11b.yaml at effc4dc4097af212432c9ebaba7eb9677d768467 · axolotl-ai-cloud/axolotl</a>：欢迎提问。通过在 GitHub 上创建账户来为 axolotl-ai-cloud/axolotl 的开发做出贡献。</li><li><a href="https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl">使用 Hugging Face 生态系统 (TRL) 微调 Vision Language Model (Qwen2-VL-7B) - Hugging Face 开源 AI 食谱</a>：无描述
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1318352517951651871)** (25 条消息🔥): 

> `Zoom 会议录音，MAX 在讨论中的使用，Mojo 在 Archcraft Linux 上的问题` 


- **Zoom 会议录音已在 YouTube 上线**：一位用户询问错过的 Zoom 会议是否有录音，确认录音将于周三发布在他们的 YouTube 频道上。
- **用于 Stable Diffusion 讨论的 MAX 工具**：一位用户祝贺社区发布了 MAX，并寻求关于其在稳定讨论（Stable Diffusion）中应用的指导，得到的回复指向了 [GitHub 仓库](https://github.com/modularml/max/tree/main/examples/inference/stable-diffusion-mojo-onnx)中的一个示例。
   - 另一位用户澄清说，要在 MAX 中使用 GPU，需要更换执行器（executor）以避免占用 CPU。
- **Mojo 在 Archcraft Linux 上的问题**：一位用户报告了在 Archcraft Linux 上进入 Mojo REPL 的问题，提到缺少动态链接库，推测可能是 `mojo-ldd` 或 `mojo-lld`。
   - 该用户表示在安装 Python 依赖时遇到困难，指出他们处于 magic 环境中，并收到关于处于外部管理环境（externally managed environment）的错误。


  

---

### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1318650059654430730)** (1 messages): 

> `MAX 24.6, MAX GPU, MAX Engine, MAX Serve, Generative AI Infrastructure` 


- **MAX 24.6 发布，搭载备受期待的 MAX GPU**：今天，我们推出了 **MAX 24.6**，其核心是备受期待的 **MAX GPU**。这是一个垂直集成的生成式 AI 栈，消除了对 NVIDIA CUDA 等特定厂商库的依赖。
   - *这是重新构想 AI 开发的第一步*，旨在解决大规模 Generative AI 日益增长的资源需求。
- **推出 MAX Engine：AI 模型编译的未来**：**MAX Engine** 被描述为一个高速 AI 模型编译器和运行时，配备了针对 NVIDIA GPU 优化的厂商无关 Mojo GPU 内核。
   - 该技术为无与伦比的性能奠定了基础，特别是在扩展复杂的 AI 工作负载时。
- **MAX Serve：简化 LLMs 的 Python 集成**：**MAX Serve** 提供了一个专为大语言模型 (LLMs) 定制的 Python 原生推理服务层，旨在增强高负载场景下的性能。
   - 它允许开发者最大化效率，并有效应对复杂的 Batching 挑战。
- **探索基准测试深度解析**：一篇新的基准测试博客文章强调了 AI 推理栈中性能权衡的复杂性，重点介绍了准确性、吞吐量和延迟之间的平衡。
   - *理解如何进行有效的基准测试* 对于开发者将创新应用从“可能”转变为“实用”至关重要。
- **加入 MAX 24.6 论坛讨论**：鼓励用户在官方 **MAX 24.6 论坛主题**中分享想法和疑问，促进社区参与和反馈。
   - 该论坛是开发者讨论最新技术发布的意义和经验的平台。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/introducing-max-24-6-a-gpu-native-generative-ai-platform?utm_campaign=24_6&utm_source=discord)**,">Modular: Introducing MAX 24.6: A GPU Native Generative AI Platform</a>: 介绍 MAX GPU 的 MAX 24.6 发布博客</li><li><a href="https://www.modular.com/blog/build-a-continuous-chat-interface-with-llama-3-and-max-serve?utm_campaign=24_6&utm_source=discord)**">Modular: Build a Continuous Chat Interface with Llama 3 and MAX Serve</a>: 使用 Llama 3 和 MAX Serve 构建聊天应用</li><li><a href="https://www.modular.com/blog/max-gpu-state-of-the-art-throughput-on-a-new-genai-platform?utm_campaign=24_6&utm_source=discord)">Modular: MAX GPU: State of the Art Throughput on a New GenAI platform</a>: 在 Modular 的 MAX 24.6 上衡量与 vLLM 相比的顶尖 GPU 性能
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1318582077502787664)** (19 条消息🔥): 

> `Mojo v24.6 发布，Python 导入 Mojo kernel，Mojo 中的 GPU 支持，Kernel 编程 API，Mojo 文档更新` 


- **Mojo v24.6 已准备就绪**：Mojo 的最新版本 **v24.6.0** 已发布，并可通过命令 `% mojo --version` 确认使用。
   - 该版本包含了用户渴望探索的更新，社区中的兴奋氛围也印证了这一点。
- **Python 顺利导入 Mojo kernel**：在最近的演示中，**Python InferenceSession** 通过编译后的 **.mojopkg** 文件成功导入了 Mojo kernel，展示了实际的集成能力。
   - 如需深入了解，感兴趣的用户可以查看[此处的源码示例](https://github.com/modularml/max/tree/nightly/examples/custom_ops)。
- **是的，Mojo 已确认支持 GPU**：针对紧随 24.6 版本之后的 **25.1.0 nightly** 版本是否支持 **GPU**，官方给出了肯定的回答：“是的，它支持！”。
   - 这展示了 Mojo 平台功能的持续增强。
- **Kernel 编程的激动人心更新**：一位成员提到，**更多 Kernel 编程 API 指南**和其他资源将很快发布，请用户保持关注。
   - 社区正热切期待关于共享内存和同步的**协作编程 (cooperative programming)** 的相关更新。
- **Mojo 文档正在修订中**：成员反馈指出 Mojo 文档中存在失效链接，特别是关于 **Span** 的部分，该部分现已移至 memory 模块（[链接在此](https://docs.modular.com/mojo/stdlib/memory/span/Span)）。
   - 另一位用户对文档中 **var 关键字** 的必要性提出了疑问，目前正等待作者的进一步更新。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/modularml/max/tree/nightly/examples/custom_ops">max/examples/custom_ops at nightly · modularml/max</a>: 一系列示例程序、笔记本和工具，展示了 MAX 平台的强大功能 - modularml/max</li><li><a href="https://docs.modular.com/mojo/manual/basics#variables)">Mojo 语言基础 | Modular Docs</a>: Mojo 基础语言特性介绍。</li><li><a href="https://docs.modular.com/mojo/stdlib/memory/span/Span">Span | Modular Docs</a>: @register_passable(trivial)</li><li><a href="https://github.com/modularml/max/tree/nightly/examples/custom_ops/kernels">max/examples/custom_ops/kernels at nightly · modularml/max</a>: 一系列示例程序、笔记本和工具，展示了 MAX 平台的强大功能 - modularml/max</li><li><a href="https://docs.modular.com/mojo/manual/values/lifetimes">生命周期、来源和引用 | Modular Docs</a>: 使用来源 (origins) 和引用 (references)。</li><li><a href="https://github.com/modularml/max/tree/nightly/pipelines/python">max/pipelines/python at nightly · modularml/max</a>: 一系列示例程序、笔记本和工具，展示了 MAX 平台的强大功能 - modularml/max</li><li><a href="https://github.com/modularml/mojo/commit/50d5fb28b886bb01bd86b8f1da892621c25e5876">[stdlib] 将 `Span` 从 `utils` 移动到 `memory` · modularml/mojo@50d5fb2</a>: `Span` 不再是一个工具类，而是整个 stdlib API 中使用的通用词汇类型。因此，将其提升到 `memory` 模块。MODULAR_ORIG_COMMIT_REV_ID: 33bae4c7dcc8191118d669985405f31599de386c
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1318654396325953697)** (2 条消息): 

> `LlamaReport 预览, Agentic AI SDR, Composio 平台` 


- **LlamaReport 让文档数据库变得可读**：LlamaReport 目前处于预览阶段，它能在几分钟内将文档数据库转换为**结构良好**、人类可读的报告，使用户能够有效地回答有关文档集的问题。更多详情请参阅 [公告文章](https://twitter.com/llama_index/status/1869094544169677138)。
   - *了解它如何通过简化输出流程来改善文档交互。*
- **使用 Agentic AI SDR 进行线索生成**：推出了一种新型的 **agentic AI SDR**，它使用 LlamaIndex 生成线索，展示了 AI 在销售策略中的实际集成。代码可以在 [此处](https://t.co/tczv5ZDI4H) 获取。
   - 该工具是更广泛的 **Quickstarters** 计划的一部分，旨在通过示例项目和实际应用帮助用户探索 Composio 的功能。
- **Composio 赋能智能 Agent**：**Composio 平台** 允许开发者创建能够跨 GitHub 和 Gmail 等应用程序自动执行任务的智能 Agent，通过自然语言命令提高生产力。平台的详细概述可以在 [子文件夹指南](https://t.co/hwTNMnhfRX) 中找到。
   - 其中一个示例项目演示了如何将待办事项列表转换为 Google Calendar 事件，展示了该平台的动态性和多功能性。



**提及的链接**：<a href="https://t.co/tczv5ZDI4H">composio/python/examples/quickstarters at master · ComposioHQ/composio</a>：Composio 通过 function calling 为你的 AI Agent 和 LLM 配备了 100 多个高质量集成 - ComposioHQ/composio

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1318353971491901452)** (20 条消息🔥): 

> `NVIDIA NV-Embed-v2 可用性, 使用 Qdrant 向量存储, OpenAI LLM 双重重试` 


- **查询 NVIDIA NV-Embed-v2**：成员们讨论了 **NVIDIA NV-Embed-v2** 在 NVIDIA Embedding 中的可用性，并引用了 `embed_model.available_models` 功能来检查可用模型。
   - 有人指出，即使 NV-Embed-v2 没有被明确列出，它可能仍然有效；建议进一步调查以确认。
- **在工作流中实现 Qdrant**：一位用户寻求在工作流中集成 **Qdrant 向量存储** 的指导，并提到了在现有集合和查询中遇到的问题。
   - 另一位成员分享了资源，包括 [文档示例](https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/)，并表示他们没有遇到类似的问题。
- **对 OpenAI LLM 重试逻辑的质疑**：**Paullg** 对 OpenAI LLM 中可能存在的双重重试提出了担忧，指出 OpenAI 客户端和 `llm_retry_decorator` 可能都在独立实现重试逻辑。
   - 随后讨论了最近的一个 Pull Request 是否修复了此问题，但对于拟议更改的有效性仍存在不确定性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/">Qdrant Vector Store - LlamaIndex</a>：未找到描述</li><li><a href="https://legacy.ts.llamaindex.ai/guides/agents/qdrant">添加持久化向量存储 | LlamaIndex.TS</a>：在之前的示例中，我们每次运行 Agent 时都会将数据加载到内存中。这对于小型数据集没问题，但对于大型数据集，你会希望将 Embedding 存储在...</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/qdrant/">Qdrant - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-openai/llama_index/embeddings/openai/base.py#L347">llama_index/llama-index-integrations/embeddings/llama-index-embeddings-openai/llama_index/embeddings/openai/base.py at main · run-llama/llama_index</a>：LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/pull/17072">fix/openai-embbeding-retry by rendyfebry · Pull Request #17072 · run-llama/llama_index</a>：描述请包含更改摘要以及修复了哪个问题。还请包括相关的动机和背景。列出此更改所需的任何依赖项。修复了 #170...
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1318442296840159252)** (2 messages): 

> `意图识别技术，处理 SSL 证书错误` 


- **探索意图识别技术**：一位开发者询问了识别用户提示词背后意图的各种**方法和技术**，以及如何根据检测到的意图在**应用程序中实现逻辑**。
   - 他们还寻求了关于**工具和框架**的建议，以提高意图识别在不同用例中的准确性和适应性。
- **修复 Azure 的 SSL 证书错误**：一名成员报告在使用 Python 通过 Azure AI Search Service 创建索引时遇到 **'SSL certification failed'** 错误，并分享了截图链接以提供更多背景信息。
   - 他们请求解决这一特定问题的方案或建议，以便继续推进项目。


  

---


### **Nomic.ai (GPT4All) ▷ #[announcements](https://discord.com/channels/1076964370942267462/1090471714888102009/1318329816025399447)** (1 messages): 

> `GPT4All v3.5.3 发布，LocalDocs 修复，GPT4All 贡献者` 


- **GPT4All v3.5.3 发布并包含重要修复**：**GPT4All v3.5.3** 版本已正式发布，解决了前一版本中的显著问题。
   - 此次更新主要包含针对 **LocalDocs** 的关键修复，该功能在 v3.5.2 中运行异常。
- **LocalDocs 问题已解决**：导致 **LocalDocs** 在 v3.5.2 中无法正常工作的严重问题已在新版本中成功解决。
   - 用户现在可以期待在使用 LocalDocs 时获得更好的性能和可靠性。
- **更新中致谢贡献者**：该版本对来自 **Nomic AI** 的 **Jared Van Bortel** 和 **Adam Treat** 在开发最新版本中的贡献表示感谢。
   - 他们的努力对于确保 GPT4All 的整体功能和改进至关重要。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1318319081866268683)** (17 messages🔥): 

> `AI Agent 功能，Jinja 模板问题，API 文档查询，文档处理效率，模型性能关注点` 


- **探索 AI Agent 功能**：讨论中提到了通过 GPT4All 运行“AI Agent”的可能性，并链接了一个展示其能力的 [YouTube 视频](https://www.youtube.com/watch?v=XeWZIzndlY4)。
   - 一位成员指出，虽然技术上可行，但它主要作为一个生成式 AI 平台，功能有限。
- **当前的 Jinja 模板问题**：一位成员表示 GPT4All 对他们来说几乎完全无法使用，特别是关于 **Jinja 模板问题**，希望该问题能尽快得到解决。
   - 另一位成员强调了 Jinja 模板的重要性，称其为模型交互至关的小程序，同时关于 tool calling 功能的持续改进正在进行中。
- **寻求完整的 API 文档**：有成员请求提供包含端点（endpoints）和参数详情的完整 API 文档，并引用了 [GPT4All API 文档](https://docs.gpt4all.io/gpt4all_api_server/home.html#key-features)中现有但似乎有限的信息。
   - 成员们分享说激活本地 API 服务器只需简单的步骤，但认为文档缺乏全面性。
- **文档处理效率**：有成员提出疑问，将文档拆分为独立文件是否比保留在一个文件中能获得更好的处理性能。
   - 成员们建议两种方法应该都可以，尽管有人指出模型似乎会随机选择文档，并建议通过复制粘贴整个文档来获得更好的上下文。
- **解决模型性能关注点**：讨论了关于模型在更新后产生随机字符串的问题，一位成员总结了可能影响性能的常见 Jinja 模板问题。
   - 提到的具体问题包括空格错误、换行符位置错误以及不支持的函数，强调需要进行调整以恢复模型功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.gpt4all.io/gpt4all_api_server/home.html#key-features">GPT4All API Server - GPT4All</a>: GPT4All 文档 - 在您的硬件上高效运行 LLM</li><li><a href="https://www.youtube.com/watch?v=XeWZIzndlY4">终于，这个 AI agent 真的能用了！</a>: 这个新的 AI agent 浏览器真的能用！Do Browser 教程与评论。#ai #aitools #ainews #aiagent #agi 感谢我们的赞助商 Thoughtly。使用...可获得 50% 折扣。
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1318465501424521266)** (12 messages🔥): 

> `Gemini 2.0 Flash 功能, VEO 2 vs SORA 对比, OpenInterpreter Web Assembly 集成, 本地 OS 使用, Open Interpreter 错误处理` 


- **关于 Gemini 2.0 Flash 的问题**：用户正在询问是否有人成功让 **Gemini 2.0 Flash** 运行起来，并强调了目前缺乏关于其功能的回复。
   - 这表明该功能的后续支持或用户体验可能存在空白。
- **关于 VEO 2 和 SORA 的辩论**：一名成员提出了 **VEO 2** 是否优于 **SORA** 的问题，并指出这两种 AI 目前在他们所在的地区都不可用。
   - 这种可用性的缺失表明了用户想要探索这些选项的兴趣，同时也带有一种挫败感。
- **OpenInterpreter 与 Web Assembly 的集成**：一位用户建议通过 Pyodide 或 Emscripten 等工具，利用 **Web Assembly** 在网页中运行 **OpenInterpreter** 项目的可能性。
   - 这种方法可以提供自动沙箱功能并消除对计算调用的需求，从而增强其在聊天 UI 场景下的可用性。
- **OpenInterpreter 中 OS 的本地使用**：有人询问是否可以在本地利用 **OS**，并有进一步的问题寻求澄清 OS 具体包含什么。
   - 这反映了寻求增强功能的用户对本地执行能力的持续兴趣。
- **Open Interpreter 的错误排查**：一位成员对在使用带有 `-y` 标志的代码时持续出现的错误表示沮丧，特别是与设置 **OpenAI API key** 相关的问题。
   - 这突显了用户面临的共同挑战，以及对更清晰的错误处理指南的需求。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1318566283901665350)** (1 messages): 

> `Torcheval 指标同步, 批处理` 


- **Torcheval 的批量指标同步简化了工作流**：一位成员对 **Torcheval** 具有 **batched metric sync**（批量指标同步）功能且没有额外依赖表示满意，认为它是一个非常好用的工具。
   - *这种流线型的方法* 提高了生产力并降低了处理指标的复杂性。
- **对无额外依赖的赞赏**：同一位成员强调了 **Torcheval** 中没有 **额外依赖**，这提升了他们使用该工具时的积极体验。
   - *这一设计选择* 似乎使得集成和操作比其他选项更顺畅。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1318385688973938778)** (3 messages): 

> `指令微调损失计算, 序列处理中的梯度归一化, FSDPModule 调整` 


- **指令微调损失中的效率低下问题**：一位成员对指令微调中的 **per-token loss**（逐 token 损失）计算表示担忧，指出由于 token 数量不同，一个句子的损失取决于 batch 中的其他句子。
   - *这种方法似乎是标准做法*，导致了社区必须去适应的挑战。
- **梯度归一化问题**：有人提到 padding/masking 会影响损失，特别是当力求通过“产生梯度的元素总数”进行归一化时。
   - 这种不一致性在 **sequence processing objectives**（序列处理目标）中非常普遍，可能会使训练复杂化。
- **调整 FSDPModule 以优化除法**：建议在将模块包装在 `fully_shard` 后，将 `set_reduce_scatter_divide_factor` 设置为 **1.0**，以解决潜在的效率低下问题。
   - 然而，这种方法可能会引入一个 *“无用”的 div kernel*，为实现增加了一层复杂性。


  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1318346415809888367)** (7 messages): 

> `GenRM 验证器模型, Sakana AI 内存优化, 8B 验证器性能分析, Chain-of-Thought 数据集生成` 


- **GenRM 验证器模型增强 LLM 性能**：最近的一篇 [论文](https://arxiv.org/abs/2408.15240v1) 提出使用基于 next-token prediction 训练的生成式验证器 (GenRM)，通过将解法生成与验证相结合，来增强大语言模型 (LLMs) 的推理能力。
   - 这种方法允许更好的指令微调，并有潜力通过 majority voting 改进计算，相比标准的 LLM 分类器具有优势。
- **Sakana AI 的 Universal Transformer Memory 降低成本**：Sakana AI 的研究人员开发了一种优化 LLM 内存使用的技术，使企业能够显著降低在 Transformer 模型上进行应用开发的成本。
   - [Universal Transformer Memory](https://sakana.ai/namm/) 技术保留关键信息并丢弃冗余，从而提高模型效率。
- **关于 8B 验证器模型影响的讨论**：有成员对使用 **8B 奖励/验证器模型** 表示担忧，指出在性能讨论中不应忽视训练此类模型的计算成本和复杂性。
   - 另一位成员指出，使用较小的验证器可能会使关于原型在实际应用中效率的假设产生偏差。
- **社区对验证器方法论的反应**：一位成员幽默地将使用 8B 验证器的方法比作“让猴子打字，然后由人类选出最好的一个”，暗示这可能具有误导性。
   - 该成员提到，标题可能暗示了比实验结果更多的内容，表明需要更广泛的实验。
- **Chain-of-Thought 数据集生成的见解**：讨论表明，8B 验证器的方法论可能更多地反映了 Chain-of-Thought 数据集的生成方式，而不是 O1 的推理过程。
   - 这被强调为理解在实际应用中使用此类验证器的影响的关键区别。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.15240v1">Generative Verifiers: Reward Modeling as Next-Token Prediction</a>: 验证器或奖励模型通常用于增强大语言模型 (LLMs) 的推理性能。一种常见的方法是 Best-of-N，即从 N 个候选解中...</li><li><a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">Scaling test-time compute - a Hugging Face Space by HuggingFaceH4</a>: 未找到描述</li><li><a href="https://venturebeat.com/ai/new-llm-optimization-technique-slashes-memory-costs-up-to-75/">New LLM optimization technique slashes memory costs up to 75%</a>: Universal Transformer Memory 使用神经网络来确定 LLM 上下文窗口中哪些 token 是有用的或冗余的。
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1318394738520494140)** (1 messages): 

> `黑客松提交截止日期, Google Form 提交流程, 项目创新` 


- **黑客松提交截止日期延长**：黑客松的提交截止日期已延长 **48 小时**，至 **太平洋时间 (PT) 12 月 19 日晚上 11:59**。
   - 此次延期旨在消除对提交过程的困惑，并让参与者有更多时间完善他们的项目。
- **提交流程澄清**：提醒参与者应通过 [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform) 提交，而 **不是** 通过 Devpost 网站。
   - 这一澄清对于确保所有项目被正确提交至关重要。
- **鼓励项目创新**：延长的截止日期为参与者提供了在项目中更具 **创新性** 的机会。
   - 鼓励参与者充分利用额外的可用时间来增强他们的提交内容。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1318367913194950799)** (8 messages🔥): 

> `LLM Agents MOOC 网站更新，证书提交截止日期` 


- **LLM Agents MOOC 网站获得移动端优化**：一位成员为 LLM Agents MOOC 网站进行了重构，以提供更好的移动端响应式体验，并在[此链接](https://gilbertomedrano.com/berkeley-ai-mooc-website/index.html)分享了更新版本。
   - *希望这能作为回馈 MOOC/Hackathon 的一种方式。* 另一位用户赞扬了该设计，并表示计划将其分享给工作人员。
- **证书提交截止日期确认**：针对潜在延期的不确定性，一位用户询问了证书提交的截止日期。
   - 另一位成员确认 **MOOC 的截止日期没有变化**，并强调提交表单将一直开放至 **12/19** 以方便大家。



**提到的链接**：<a href="https://gilbertomedrano.com/berkeley-ai-mooc-website/index.html">Large Language Model Agents MOOC</a>：未找到描述

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1318426684445163531)** (6 messages): 

> `通过 USB 连接 GPU，Mac 对 arm64 后端的支持，Mac 上的持续集成 (CI)` 


- **探索通过 USB 端口连接 GPU**：一位用户询问是否可以简单地将 **GPU** 插入 **USB 端口**，George 对此给予了肯定回答，并表示“我们的驱动程序应该允许这样做”。
   - 这一交流突显了社区内关于硬件兼容性的持续讨论。
- **关于 Mac 访问权限以进行后端开发的疑问**：一位用户表示有兴趣获得 **Macs** 的访问权限，专门用于 **arm64 后端** 开发。
   - George 澄清说这些系统是 **CI only**（仅限 CI），意味着它们仅运行基准测试，不开放给一般用途。
- **了解 Mac 在持续集成中的角色**：对话确认了 **Mac Benchmark** 是持续集成 (CI) 流程的一部分，专注于性能评估。
   - 这反映了社区强调利用特定硬件进行严格测试程序的特点。



**提到的链接**：<a href="https://x.com/__tinygrad__/status/1868867387652714522">来自 tiny corp (@__tinygrad__) 的推文</a>：呃，你确定能直接把 GPU 插进 USB 接口吗？

  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1318486774959444088)** (6 messages): 

> `扩展测试时计算 (Scaling Test Time Compute)，3b 模型与 70b 模型的性能对比，仓库中缺失的 Optim 代码` 


- **扩展测试时计算分析**：一位成员分享了 [Hugging Face 博客文章](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) 的链接，讨论了扩展测试时计算（scaling test time compute），认为其内容**令人耳目一新**。
   - 这篇文章引发了社区对扩展测试效率的关注。
- **3b 模型在数学方面超越 70b**：一位成员指出 **3b 模型** 在数学表现上优于 **70b 模型**，这既**疯狂**又意义重大。
   - 这一观察引发了关于小型模型意想不到的效率的讨论。
- **寻找缺失的 Optim 代码**：有成员表示担心在开发者的 Repo 中找不到实际的 **optim 代码**，该仓库目前仅包含基准测试脚本。
   - 一位成员表示他们在该 Repo 上遇到了困难，并强调正在努力解决问题。
- **当前工作量影响任务进度**：一位成员因无法贡献代码而道歉，透露自己正忙于其他任务和 Bug 修复。
   - 这突显了社区内开发和协作的繁忙状态。
- **感谢更新**：在持续的讨论中，一位成员对另一位成员提供的更新表示感谢。
   - 这反映了频道内积极且支持的氛围。



**提到的链接**：<a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">Scaling test-time compute - HuggingFaceH4 的 Hugging Face Space</a>：未找到描述

  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1318687341027655800)** (2 条消息): 

> `Autonomous AI 的影响、知识经济中的 AI Agent、劳动者替代、AI 在等级制企业中的角色` 


- **Autonomous Agent 增强知识型工作者**：最近的一篇 [论文](https://arxiv.org/abs/2312.05481) 讨论了 **autonomous AI** 如何使知识最渊博的个体受益，让他们在提高生产力的同时，能更高效地完成常规工作。
   - 研究表明，虽然初步研究指出 AI 有助于低技能员工，但这些研究主要集中在 **chatbot** 上，因此忽略了 **agentic AI** 可能会将利益转向技能更高的人群。
- **AI 的运行模式影响劳动力动态**：论文引入了一个框架，其中 **AI Agent** 可以自主或非自主运行，强调了等级制企业内部劳动力动态的演变。
   - 论文指出，虽然 **基础级 autonomous AI** 可能会将人类挤压到专业化角色中，但 **高级 autonomous AI** 会将劳动力重新分配到常规任务中，从而产生规模更大、生产力更高的企业。
- **AI 对知识较少个体的益处**：非自主 AI（如 **chatbot**）为知识较少的个体提供了负担得起的专家协助，增强了他们解决问题的能力，而不会在大型任务上产生竞争。
   - 因此，虽然他们被认为从技术中受益，但随着 AI 的演进，**autonomous Agent** 辅助**知识型工作者**的能力将使其享有竞争优势。



**提到的链接**：<a href="https://arxiv.org/abs/2312.05481">Artificial Intelligence in the Knowledge Economy</a>：人工智能（AI）的兴起有可能通过大规模解决问题来从根本上重塑知识经济。本文介绍了一个研究这一转型的框架，……

  

---


### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1318308461712379994)** (2 条消息): 

> `Retrieval Augmented Generation (RAG) 应用、Developer Hub 和 Blueprints 公告` 


- **超低依赖 RAG 应用的最终活动**：明天是 12 月的最后一场活动，参与者将学习如何仅使用 **sqlite-vec**、**llamafile** 和基础 **Python** 创建一个**超低依赖的 Retrieval Augmented Generation (RAG)** 应用。
   - 本次会议将由 **Alex Garcia** 主持，不需要任何其他依赖项或 "pip install"。
- **Developer Hub 和 Blueprints 的重要更新**：关于 **Developer Hub** 和 **Blueprints** 发布了一项重要公告，提醒用户刷新认知。
   - 随着社区深入探讨 **Blueprints** 线程，反馈意见正受到重视，该项目旨在帮助开发者构建开源 AI 解决方案。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1318574174117298236)** (1 条消息): 

> `数据基础设施创新、数据治理、数据流、流处理、数据基础设施中的 AI` 


- **数据创新年终回顾小组讨论**：欢迎在 12 月 18 日参加回顾小组讨论，届时开拓性的创始人 **Yingjun Wu**、**Stéphane Derosiaux** 和 **Alexander Gallego** 将讨论过去一年中 **数据基础设施** 领域令人兴奋的创新。
   - 小组讨论将涵盖关键主题，包括 **数据治理**、**数据流 (Streaming)** 以及 **AI 对数据基础设施的影响**。
- **小组讨论嘉宾阵容强大**：小组成员包括行业领袖：[Yingjun Wu](https://www.linkedin.com/in/yingjun-wu/)（**RisingWave CEO**）、[Stéphane Derosiaux](https://www.linkedin.com/in/stephane-derosiaux/)（**Conduktor CPTO**）以及 [Alexander Gallego](https://www.linkedin.com/in/alexandergallego/)（**Redpanda CEO**）。
   - 他们的见解预计将探讨 **流处理 (Stream Processing)** 和 **Iceberg 格式** 等关键领域，塑造 2024 年的格局。
- **预留小组讨论席位**：不要错过这个机会！在此处[预留活动席位](https://www.meetup.com/streaming-stories/events/304951233/)。
   - 这是一个深入了解 **数据基础设施生态系统** 中最重大进展的绝佳机会。



**提到的链接**：<a href="https://www.meetup.com/streaming-stories/events/304951233/">数据基础设施年终回顾，2024 年 12 月 18 日星期三上午 9:00 | Meetup</a>：**关于**：2024 年对于数据基础设施来说是具有开创性的一年。我们见证了一系列令人兴奋的创新，其中许多都在推动……

  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1318697465855086655)** (1 条消息): 

> `BFCL Leaderboard V3, Function calling 能力, 模型响应加载问题` 


- **关于 BFCL Leaderboard 的 Function Call Demo 的咨询**：一位成员就 [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard) 上的 Function Call Demo 及其运行状态提出了疑问。
   - 他们特别指出，页面似乎卡在了 **'Loading Model Response...'** 状态，并询问是否有人遇到同样的问题。
- **BFCL 特性与更新概览**：讨论强调了 [Berkeley Function Calling Leaderboard V3](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard) 的细节，包括其对 LLM 准确调用 Function 的评估标准。
   - 成员们提到了详细介绍该排行榜各版本的博客，例如 [BFCL-v1](blogs/8_berkeley_function_calling_leaderboard.html)、**BFCL-v2** 以及包含扩展数据集和多轮对话（multi-turn interactions）方法的 **BFCL-v3**。



**提到的链接**: <a href="https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard">
        Berkeley Function Calling Leaderboard V3 (又名 Berkeley Tool Calling Leaderboard V3)
    </a>: 未找到描述

  

---


---


---


{% else %}


> 完整的各频道详情已针对邮件进行截断。 
> 
> 如果您想查看完整的详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}